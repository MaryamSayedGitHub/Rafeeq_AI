"""Smart lecture montage: the pure helpers (no ffmpeg, no internet) and, when ffmpeg exists, one real run.
    uv run pytest tests/test_montage.py
"""
import shutil
import subprocess

import pytest

from rafeeq.teacher import montage

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def test_keep_segments_cuts_the_silence_but_keeps_padding():
    kept = montage.keep_segments([(2.0, 5.0)], 10.0)
    assert kept == [(0.0, 2.0 + montage.PAD), (5.0 - montage.PAD, 10.0)]


def test_keep_segments_without_silence_keeps_everything():
    assert montage.keep_segments([], 7.0) == [(0.0, 7.0)]


def test_keep_segments_drops_tiny_pieces_and_caps_the_count():
    silences = [(i * 1.0 + 0.5, i * 1.0 + 1.4) for i in range(400)]
    assert len(montage.keep_segments(silences, 400.0)) <= montage.MAX_SEGMENTS


def test_clean_segments_drops_bad_rows_and_keeps_order():
    raw = [{"start": 0, "end": 2, "text": "  hello   world "}, {"start": "x"}, "junk",
           {"start": 1, "end": 4, "text": "overlap"}, {"start": 50, "end": 60, "text": "past the end"},
           {"start": 4, "end": 4.05, "text": "too short"}]
    out = montage.clean_segments(raw, 10.0)
    assert [s["text"] for s in out] == ["hello world", "overlap"]
    assert out[1]["start"] >= out[0]["end"]
    assert montage.clean_segments("not a list", 10.0) == []


def test_to_srt_format_and_offset():
    srt = montage.to_srt([{"start": 0.0, "end": 1.5, "text": "مرحبا"}], offset=3.0)
    assert srt.startswith("1\n00:00:03,000 --> 00:00:04,500\nمرحبا")


def test_clean_chapters_rules():
    good = montage.clean_chapters([{"title": "A", "start": 5}, {"title": "B", "start": 30},
                                   {"title": "B2", "start": 20}, {"title": "", "start": 40},
                                   {"title": "late", "start": 999}], 100.0)
    assert [c["title"] for c in good] == ["A", "B"] and good[0]["start"] == 0.0
    assert montage.clean_chapters([{"title": "only one", "start": 0}], 100.0) == []
    assert montage.clean_chapters(None, 100.0) == []


def test_make_chapters_survives_a_failing_model():
    def boom(*_):
        raise RuntimeError("down")
    assert montage.make_chapters([{"start": 0, "end": 1, "text": "x"}], 10.0, boom) == []
    assert montage.make_chapters([], 10.0, lambda *_: {}) == []


def test_clean_options_is_strict():
    o = montage.clean_options({"cut_silence": "on", "burn_captions": "on", "title": "  Lesson   1 " + "x" * 500,
                               "music_asset_id": "abc", "evil": "<script>"})
    assert o["cut_silence"] and not o["captions"] and not o["burn_captions"]   # burn needs captions
    assert o["title"].startswith("Lesson 1") and len(o["title"]) == montage.MAX_TITLE
    assert o["music"] and "evil" not in o


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_pipeline_cuts_silence(tmp_path):
    src = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=10:r=15",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-filter_complex", "[1:a]apad=whole_dur=10[a]", "-map", "0:v", "-map", "[a]",
                    "-shortest", "-pix_fmt", "yuv420p", str(src)], check=True, capture_output=True)
    work = tmp_path / "w"
    work.mkdir()
    res = montage.run(src, work, montage.clean_options({"cut_silence": True}))
    assert res["file"].exists()
    assert montage.probe(res["file"])["seconds"] < montage.probe(src)["seconds"] - 3


def test_editor_keep_list_is_cleaned():
    assert montage.clean_keep([[5, 9], [0, 3], [2.95, 4], ["x"], [8, 8.05], "junk"], 10.0) == [(0.0, 4.0), (5.0, 9.0)]
    assert montage.clean_keep(None) == []


def test_editor_times_move_onto_the_cut_video():
    keep = [(0.0, 2.0), (7.0, 14.5)]
    assert montage.remap_time(1.0, keep) == 1.0
    assert montage.remap_time(8.0, keep) == 3.0            # 2 s of the first piece + 1 s into the second
    assert montage.remap_time(4.0, keep) == 2.0            # inside a cut-out part: the start of the next piece
    segs = [{"start": 0, "end": 1.5, "text": "a"}, {"start": 3, "end": 4, "text": "gone"}, {"start": 8, "end": 10, "text": "b"}]
    assert [s["text"] for s in montage.remap_segments(segs, keep)] == ["a", "b"]


def test_editor_decisions_survive_clean_options_and_stay_out_of_the_asset_settings():
    o = montage.clean_options({"keep": [[0, 5]], "segments": [{"start": 0, "end": 2, "text": " hi  there "}, {"bad": 1}],
                               "chapter_list": [{"title": "A", "start": 0}], "captions": True})
    assert o["keep"] == [(0.0, 5.0)] and o["segments"][0]["text"] == "hi there" and len(o["segments"]) == 1
    pub = montage.public_options(o)
    assert "segments" not in pub and pub["segments_count"] == 1 and pub["keep_count"] == 1
    assert montage.clean_options({})["keep"] is None


# ------------------------------------------------------------------ the agent team
from rafeeq.teacher.montage_team import agents as team_agents          # noqa: E402
from rafeeq.teacher.montage_team import graph as team_graph            # noqa: E402


def _plan(**opts):
    state = {"opts": montage.clean_options(opts)}
    return team_agents.supervisor(state, {"configurable": {"deps": {"music_file": opts.get("_music")}}})["plan"]


def test_supervisor_plans_only_what_was_asked():
    assert _plan(cut_silence=True) == ["analyst_agent", "cutter_agent", "qa_agent"]
    assert _plan(captions=True) == ["analyst_agent", "transcriber_agent", "qa_agent"]
    assert _plan(chapters=True) == ["analyst_agent", "transcriber_agent", "chapters_agent", "qa_agent"]
    full = _plan(cut_silence=True, captions=True, burn_captions=True, chapters=True, intro=True, outro=True,
                 title="T")
    assert full == ["analyst_agent", "cutter_agent", "transcriber_agent", "chapters_agent", "designer_agent",
                    "assembler_agent", "qa_agent"]


def test_supervisor_skips_the_intro_card_without_a_title():
    assert "designer_agent" not in _plan(intro=True)


def test_next_agent_walks_the_plan_and_qa_can_send_back():
    state = {"plan": ["analyst_agent", "cutter_agent", "qa_agent"]}
    assert team_graph.next_agent(state, "supervisor") == "analyst_agent"
    assert team_graph.next_agent(state, "analyst_agent") == "cutter_agent"
    assert team_graph.next_agent(state, "cutter_agent") == "qa_agent"
    assert team_graph.route_after_qa({"qa_ok": True}) == team_graph.END_NODE
    assert team_graph.route_after_qa({"qa_ok": False}) == "assembler_agent"


def _lecture(tmp_path):
    src = tmp_path / "raw.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=10:r=15",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-filter_complex", "[1:a]apad=whole_dur=10[a]",
                    "-map", "0:v", "-map", "[a]", "-shortest", "-pix_fmt", "yuv420p", str(src)],
                   check=True, capture_output=True)
    return src


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_team_runs_every_agent_in_order(tmp_path):
    fake_lines = lambda audio: [{"start": 0, "end": 2, "text": "hello"}, {"start": 2.5, "end": 3, "text": "bye"}]   # noqa: E731
    fake_llm = lambda system, user: {"chapters": [{"title": "A", "start": 0}, {"title": "B", "start": 2.5}]}      # noqa: E731
    opts = montage.clean_options({"cut_silence": True, "captions": True, "burn_captions": True, "chapters": True,
                                  "intro": True, "outro": True, "title": "Lesson 1"})
    steps = []
    out = montage.run(_lecture(tmp_path), tmp_path, opts, transcribe=fake_lines, propose=fake_llm, progress=steps.append)
    assert out["agents"] == ["supervisor", "analyst_agent", "cutter_agent", "transcriber_agent", "chapters_agent",
                             "designer_agent", "assembler_agent", "qa_agent"]
    assert steps == ["analyzing", "cutting", "transcribing", "chapters", "designing", "assembling", "checking"]
    assert out["captions"] == 2 and out["chapters"][0] == {"title": "Lesson 1", "start": 0.0}
    assert out["srt"].startswith("1\n00:00:03,000")            # shifted by the 3 s title card


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_qa_sends_the_work_back_when_burning_captions_fails(tmp_path, monkeypatch):
    real = montage.assemble

    def flaky(*a, **kw):
        if kw.get("burn_srt"):
            raise montage.MediaError("subtitles filter missing")
        return real(*a, **kw)
    monkeypatch.setattr(montage, "assemble", flaky)
    opts = montage.clean_options({"captions": True, "burn_captions": True})
    out = montage.run(_lecture(tmp_path), tmp_path, opts,
                      transcribe=lambda audio: [{"start": 0, "end": 2, "text": "hello"}])
    assert len([a for a in out["agents"] if a.startswith("assembler_agent")]) == 2 and any("retry" in a for a in out["agents"])
    assert out["file"].exists() and any(n.startswith("burn_failed") for n in out["notes"])


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_team_uses_what_the_lecturer_decided_in_the_editor(tmp_path):
    opts = montage.clean_options({"keep": [[0, 2], [6, 10]], "captions": True, "burn_captions": True, "intro": True,
                                  "title": "T", "segments": [{"start": 0.2, "end": 1.8, "text": "one"},
                                                             {"start": 3, "end": 4, "text": "cut out"},
                                                             {"start": 7, "end": 9, "text": "two"}]})
    out = montage.run(_lecture(tmp_path), tmp_path, opts, transcribe=lambda a: pytest.fail("must not transcribe again"))
    assert "transcriber_agent" not in out["agents"] and "cutter_agent" in out["agents"]
    assert out["cut"]["removed_seconds"] == 4.0
    assert [l for l in out["srt"].splitlines() if l and not l[0].isdigit()] == ["one", "two"]
    assert out["srt"].splitlines()[1].startswith("00:00:03,200")       # 0.2 s + the 3 s title card


# ------------------------------------------------------------------ GIFs / stickers
GIF_URL = "https://media2.giphy.com/media/abc123/giphy.gif"


def test_sticker_urls_must_be_giphy_https():
    ok = montage._giphy_ok
    assert ok(GIF_URL) and ok("https://i.giphy.com/x.gif") and ok("https://media.giphy.com/a.gif")
    for bad in ("http://media.giphy.com/a.gif", "https://giphy.com.evil.io/a.gif", "https://evil.io/media.giphy.com/a.gif",
                "https://media.giphy.com@evil.io/a.gif", "https://localhost/a.gif", "file:///etc/passwd", "", None):
        assert not ok(bad)


def test_clean_stickers_keeps_only_valid_ones_in_bounds():
    raw = [{"start": 5, "end": 8, "x": 0.5, "y": 2, "w": 9, "src": {"giphy": GIF_URL}},
           {"start": 1, "end": 1.1, "x": .5, "y": .5, "w": .2, "src": {"giphy": GIF_URL}},                  # too short
           {"start": 0, "end": 3, "x": .5, "y": .5, "w": .2, "src": {"giphy": "https://evil.io/a.gif"}},    # not Giphy
           {"start": 0, "end": 3, "x": .5, "y": .5, "w": .2, "src": {}},                                    # no source
           {"start": 2, "end": 4, "x": .1, "y": .1, "w": .2, "src": {"asset_id": "a1"}}, "junk", {"start": "x"}]
    out = montage.clean_stickers(raw, total=7.0)
    assert [s["start"] for s in out] == [2.0, 5.0]
    assert out[1]["end"] == 7.0 and out[1]["y"] == 1.0 and out[1]["w"] == 0.8          # clipped to the video / the frame
    assert montage.clean_stickers(None) == [] and len(montage.clean_stickers(raw * 20)) <= montage.MAX_STICKERS


def test_stickers_move_onto_the_cut_video_and_a_cut_out_one_is_dropped():
    keep = [(0.0, 2.0), (6.0, 10.0)]
    st = [{"start": 1.0, "end": 1.9}, {"start": 3.0, "end": 5.0}, {"start": 6.5, "end": 8.0}]
    out = montage.remap_stickers(st, keep)
    assert [(s["start"], s["end"]) for s in out] == [(1.0, 1.9), (2.5, 4.0)]


def test_stickers_reach_the_supervisor_and_stay_out_of_the_asset_settings():
    o = montage.clean_options({"stickers": [{"start": 0, "end": 3, "x": .5, "y": .5, "w": .2, "src": {"giphy": GIF_URL}}]})
    assert len(o["stickers"]) == 1 and montage.public_options(o)["stickers_count"] == 1 and "stickers" not in montage.public_options(o)
    assert _plan(stickers=o["stickers"] and [{"start": 0, "end": 3, "x": .5, "y": .5, "w": .2, "src": {"giphy": GIF_URL}}]) == \
        ["analyst_agent", "sticker_agent", "assembler_agent", "qa_agent"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_team_lays_gifs_over_the_lesson_and_skips_a_broken_one(tmp_path):
    gif = tmp_path / "party.gif"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=96x64:d=1:r=10", str(gif)], check=True, capture_output=True)
    png = tmp_path / "logo.png"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=40x40:d=1", "-frames:v", "1", str(png)], check=True, capture_output=True)

    def fake_fetch(url, dst):
        if "broken" in url:
            raise montage.MediaError("gone")
        dst.write_bytes(gif.read_bytes())
        return dst
    stickers = [{"start": 1, "end": 4, "x": .8, "y": .75, "w": .25, "src": {"giphy": GIF_URL}},
                {"start": 2, "end": 5, "x": .2, "y": .2, "w": .1, "src": {"asset_id": "p1"}},
                {"start": 6, "end": 8, "x": .5, "y": .5, "w": .2, "src": {"giphy": "https://media1.giphy.com/broken.gif"}}]
    opts = montage.clean_options({"stickers": stickers, "captions": True, "burn_captions": True, "intro": True, "title": "T",
                                  "segments": [{"start": 0.2, "end": 1.8, "text": "one"}]})
    steps = []
    out = montage.run(_lecture(tmp_path), tmp_path, opts, sticker_files={"p1": png}, fetch=fake_fetch, progress=steps.append)
    assert "sticker_agent" in out["agents"] and "stickers" in steps
    assert any(n.startswith("sticker_failed") for n in out["notes"])
    assert out["file"].exists() and abs(out["seconds"] - 13.0) < 0.6                  # 10 s lesson + the 3 s title card
    # the GIF really is in the picture: the frame with the GIF differs from the same frame without it
    frame = lambda f, t: subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(f), "-frames:v", "1", "-f", "rawvideo",    # noqa: E731
                                         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    plain = montage.run(_lecture(tmp_path), tmp_path / "p", montage.clean_options({"intro": True, "title": "T"}), progress=lambda s: None) \
        if (tmp_path / "p").mkdir() is None else None
    assert frame(out["file"], 3 + 2.0) != frame(plain["file"], 3 + 2.0)               # at 2 s of the lesson (after the card)
    assert frame(out["file"], 3 + 9.0)[:2000] is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_gifs_follow_the_cut_timeline(tmp_path):
    gif = tmp_path / "g.gif"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=s=64x64:d=1:r=10", str(gif)], check=True, capture_output=True)
    opts = montage.clean_options({"keep": [[0, 2], [6, 10]], "stickers": [
        {"start": 3, "end": 5, "x": .5, "y": .5, "w": .3, "src": {"giphy": GIF_URL}},             # inside the cut-out part: dropped
        {"start": 7, "end": 9, "x": .5, "y": .5, "w": .3, "src": {"giphy": GIF_URL}}]})
    out = montage.run(_lecture(tmp_path), tmp_path, opts, fetch=lambda url, dst: (dst.write_bytes(gif.read_bytes()), dst)[1])
    assert out["cut"]["removed_seconds"] == 4.0 and out["file"].exists() and not any(n.startswith("sticker_failed") for n in out["notes"])

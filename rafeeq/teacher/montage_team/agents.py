"""The montage agents. Each one is a small function: it reads some keys of the state and returns the keys it changed.

    supervisor          decides WHICH agents are needed for what the lecturer ticked (rules, no model)
    analyst_agent       looks at the video: length, sound, where the long pauses are            (ffprobe, silencedetect)
    cutter_agent        removes the pauses                                                      (ffmpeg)
    transcriber_agent   listens to the speech and writes timed caption lines                   (Gemini, audio input)
    chapters_agent      reads the caption lines and names 3 to 8 chapters                      (LLM, JSON)
    designer_agent      draws the title card and the closing card                               (Pillow)
    sticker_agent       fetches the GIFs/pictures the lecturer placed and moves them onto the cut timeline
    assembler_agent     joins cards + lesson, mixes the music, burns the captions               (ffmpeg)
    qa_agent            checks the finished file; if it is broken, sends the work back once; then packages the result

Only the transcriber and chapters agents use a model. The rest is signal processing and rendering, which is
deliberate: the lecturer's words are never rewritten, and the result can be repeated.

Models and the progress callback are injected through `config["configurable"]["deps"]`, so tests run offline.
"""
import logging

from rafeeq.teacher import montage as M
from rafeeq.teacher.media_tools import MediaError
from rafeeq.teacher.montage_team.state import AGENT_ORDER, MAX_ATTEMPTS

log = logging.getLogger(__name__)


def _deps(config) -> dict:
    return ((config or {}).get("configurable") or {}).get("deps") or {}


def _progress(config, step: str) -> None:
    cb = _deps(config).get("progress")
    if cb:
        cb(step)


def _want_cards(opts: dict) -> tuple[bool, bool]:
    return bool(opts.get("intro") and opts.get("title")), bool(opts.get("outro"))


# ---------------------------------------------------------------- supervisor
def supervisor(state, config=None) -> dict:
    """Reads: opts.   Writes: plan, reasoning.   The plan is always in AGENT_ORDER; analyst and qa always run."""
    o, deps = state["opts"], _deps(config)
    intro, outro = _want_cards(o)
    music = bool(o.get("music") and deps.get("music_file"))
    wanted, why = {"analyst_agent", "qa_agent"}, []
    if o.get("cut_silence") or o.get("keep"):
        wanted.add("cutter_agent"); why.append("cutter_agent: cut the parts the lecturer marked" if o.get("keep") else "cutter_agent: cut the long pauses")
    given_captions, given_chapters = o.get("segments") is not None, o.get("chapter_list") is not None
    if (o.get("captions") or o.get("chapters")) and not given_captions:
        wanted.add("transcriber_agent"); why.append("transcriber_agent: the speech must be written down first")
    if o.get("chapters") and not given_chapters:
        wanted.add("chapters_agent"); why.append("chapters_agent: name the chapters from the captions")
    if given_captions or given_chapters:
        why.append("captions/chapters: taken as the lecturer edited them in the editor")
    if intro or outro:
        wanted.add("designer_agent"); why.append("designer_agent: draw the title/closing card")
    if o.get("stickers"):
        wanted.add("sticker_agent"); why.append("sticker_agent: fetch the GIFs the lecturer placed")
    if intro or outro or music or o.get("burn_captions") or o.get("stickers"):
        wanted.add("assembler_agent"); why.append("assembler_agent: join the cards, mix the music, lay the GIFs or burn the captions")
    plan = [a for a in AGENT_ORDER if a in wanted]
    return {"plan": plan, "reasoning": "; ".join(why) or "nothing to do", "attempts": 0, "qa_ok": False,
            "notes": [], "trace": ["supervisor"]}


# ---------------------------------------------------------------- analyst
def analyst_agent(state, config=None) -> dict:
    """Reads: src, opts, plan.   Writes: info, current, total, keep, removed, cut (+ notes)."""
    _progress(config, "analyzing")
    info = M.probe(state["src"])
    if info["seconds"] > M.MAX_SECONDS:
        raise MediaError(f"The video is longer than {M.MAX_SECONDS // 60} minutes")
    out = {"info": info, "current": state["src"], "total": info["seconds"], "keep": [], "removed": 0.0,
           "cut": {"removed_seconds": 0.0, "pieces": 0}, "trace": ["analyst_agent"], "notes": []}
    opts = state["opts"]
    marked = M.clean_keep(opts.get("keep"), info["seconds"]) if opts.get("keep") else []
    if marked and "cutter_agent" in state["plan"]:
        removed = info["seconds"] - sum(b - a for a, b in marked)
        if removed >= 0.5:
            out["keep"], out["removed"] = marked, removed
    elif "cutter_agent" in state["plan"]:
        if not info["audio"]:
            out["notes"].append("no_audio")
        else:
            keep = M.keep_segments(M.detect_silences(state["src"]), info["seconds"])
            removed = info["seconds"] - sum(b - a for a, b in keep)
            if keep and removed >= 2.0:
                out["keep"], out["removed"] = keep, removed
            else:
                out["notes"].append("no_long_pauses")
    # What the lecturer edited in the editor sits on the ORIGINAL timeline: move it onto the cut one.
    if opts.get("segments") is not None:
        out["segments"] = M.remap_segments(M.clean_segments(opts["segments"], info["seconds"]), out["keep"])
    if opts.get("chapter_list") is not None:
        mapped = M.remap_chapters(opts["chapter_list"], out["keep"])
        out["chapters"] = M.clean_chapters(mapped, M.remap_time(info["seconds"], out["keep"]) if out["keep"] else info["seconds"])
    return out


# ---------------------------------------------------------------- cutter
def cutter_agent(state, config=None) -> dict:
    """Reads: keep, src, info, work.   Writes: current, total, cut."""
    if not state.get("keep"):
        return {"trace": ["cutter_agent (nothing to cut)"]}
    _progress(config, "cutting")
    try:
        cut = M.render_cut(state["src"], state["keep"], state["work"] / "cut.mp4", state["info"]["audio"], state["work"])
    except MediaError as e:
        return {"notes": ["cut_failed:" + str(e)[:160]], "trace": ["cutter_agent (failed, original kept)"]}
    return {"current": cut, "total": M.probe(cut)["seconds"],
            "cut": {"removed_seconds": round(state["removed"], 1), "pieces": len(state["keep"])},
            "trace": ["cutter_agent"]}


# ---------------------------------------------------------------- transcriber
def transcriber_agent(state, config=None) -> dict:
    """Reads: current, total, info.   Writes: segments (+ notes). A failure here never stops the montage."""
    if not state["info"]["audio"]:
        return {"notes": ["no_audio"], "trace": ["transcriber_agent (no sound)"]}
    _progress(config, "transcribing")
    try:
        audio = M.extract_audio(state["current"], state["work"] / "speech.mp3")
        segments = M.clean_segments((_deps(config).get("transcribe") or M.gemini_transcribe)(audio), state["total"])
    except MediaError as e:
        return {"segments": [], "notes": ["captions_failed:" + str(e)[:160]], "trace": ["transcriber_agent (failed)"]}
    return {"segments": segments, "notes": [] if segments else ["no_speech_found"], "trace": ["transcriber_agent"]}


# ---------------------------------------------------------------- chapters
def chapters_agent(state, config=None) -> dict:
    """Reads: segments, total.   Writes: chapters (+ notes)."""
    if not state.get("segments"):
        return {"trace": ["chapters_agent (no captions)"]}
    _progress(config, "chapters")
    chapters = M.make_chapters(state["segments"], state["total"], _deps(config).get("propose"))
    return {"chapters": chapters, "notes": [] if chapters else ["no_chapters"], "trace": ["chapters_agent"]}


# ---------------------------------------------------------------- designer
def designer_agent(state, config=None) -> dict:
    """Reads: opts, info, work.   Writes: intro, outro."""
    _progress(config, "designing")
    o, info = state["opts"], state["info"]
    intro, outro = _want_cards(o)
    w, h = info["width"] - info["width"] % 2, info["height"] - info["height"] % 2
    title, subtitle = o.get("title", ""), o.get("subtitle", "")
    return {
        "intro": M.make_card(title, subtitle, (w, h), state["work"] / "intro.png") if intro else None,
        "outro": M.make_card(str(o.get("outro_text") or subtitle or title or "Thank you")[:M.MAX_TITLE], "", (w, h),
                             state["work"] / "outro.png") if outro else None,
        "trace": ["designer_agent"]}


# ---------------------------------------------------------------- stickers
def sticker_agent(state, config=None) -> dict:
    """Reads: opts.stickers, keep, work.   Writes: stickers (+ notes). A picture that cannot be fetched is skipped, never fatal."""
    _progress(config, "stickers")
    deps = _deps(config)
    files, fetch = deps.get("sticker_files") or {}, deps.get("fetch") or M.fetch_giphy
    placed = M.remap_stickers(M.clean_stickers(state["opts"].get("stickers"), state["info"]["seconds"]), state.get("keep") or [])
    out, notes, cache = [], [], {}
    for i, st in enumerate(placed):
        src = st["src"]
        try:
            if src.get("giphy"):
                key = src["giphy"]
                if key not in cache:
                    cache[key] = fetch(key, state["work"] / f"sticker{len(cache)}.gif")
                path = cache[key]
            else:
                path = files.get(src.get("asset_id"))
                if path is None:
                    raise MediaError("picture missing")
            w, h = M.picture_size(path)
            if w <= 0 or h <= 0:
                raise MediaError("unreadable picture")
        except MediaError as e:
            notes.append("sticker_failed:" + str(e)[:100])
            continue
        out.append({"path": path, "start": min(st["start"], state["total"]), "end": min(st["end"], state["total"]),
                    "x": st["x"], "y": st["y"], "w": st["w"], "ratio": h / w})
    if placed and not out:
        notes.append("stickers_none")
    return {"stickers": out, "notes": notes, "trace": ["sticker_agent"]}


# ---------------------------------------------------------------- assembler
def assembler_agent(state, config=None) -> dict:
    """Reads: current, intro, outro, segments, opts, info, burn_off.   Writes: final, burn, attempts.
    If burning the captions fails, it says so and qa sends it back once to render without them."""
    _progress(config, "assembling")
    o, info, work = state["opts"], state["info"], state["work"]
    deps = _deps(config)
    attempts = state.get("attempts", 0) + 1
    burn = bool(o.get("burn_captions") and state.get("segments") and not state.get("burn_off"))
    music = deps.get("music_file") if o.get("music") else None
    intro, outro = state.get("intro"), state.get("outro")
    stickers = [] if state.get("burn_off") else (state.get("stickers") or [])      # a second run is the plain one
    dropped = ["stickers_dropped"] if state.get("burn_off") and state.get("stickers") else []
    if not (intro or outro or music or burn or stickers):
        return {"final": state["current"], "burn": False, "attempts": attempts, "notes": dropped, "trace": ["assembler_agent (nothing to join)"]}
    w, h = info["width"] - info["width"] % 2, info["height"] - info["height"] % 2
    if burn:
        (work / "captions.srt").write_text(M.to_srt(state["segments"], 0.0), encoding="utf-8")
    try:
        final = M.assemble(state["current"], work / "montage.mp4", work, width=w, height=h, has_audio=info["audio"],
                           intro=intro, outro=outro, music=music, burn_srt=(work / "captions.srt") if burn else None,
                           total=state["total"], stickers=stickers)
    except MediaError as e:
        if not burn and not stickers:
            raise
        # captions or GIFs may be what broke it: the second run (asked by qa) leaves both out
        return {"final": None, "burn": False, "burn_off": True, "attempts": attempts,
                "notes": ["burn_failed:" + str(e)[:120]], "trace": ["assembler_agent (captions burn failed)"]}
    return {"final": final, "burn": burn, "attempts": attempts, "notes": dropped, "trace": ["assembler_agent"]}


# ---------------------------------------------------------------- qa
def qa_agent(state, config=None) -> dict:
    """Reads: final/current, info, intro, outro, segments, chapters.   Writes: qa_ok, result.
    Checks the file really plays and still has its sound, and that the length is what the steps add up to.
    A bad file goes back to the assembler (once). It then packages the result for the library."""
    _progress(config, "checking")
    final = state.get("final") or (state["current"] if "assembler_agent" not in state["plan"] else None)
    problem = ""
    if final is None:
        problem = "no output"
    elif not final.exists() or final.stat().st_size == 0:
        problem = "empty file"
    else:
        try:
            got = M.probe(final)
        except Exception:                                           # noqa: BLE001 - ffprobe could not read it
            got, problem = None, "unreadable file"
        if got:
            cards = int(bool(state.get("intro"))) + int(bool(state.get("outro")))
            expected = state["total"] + M.CARD_SECONDS * cards
            if got["seconds"] < max(1.0, expected * 0.5):
                problem = f"too short ({got['seconds']:.1f}s, expected about {expected:.1f}s)"
            elif state["info"]["audio"] and not got["audio"]:
                problem = "the sound is missing"
    if problem:
        if state.get("attempts", 0) < MAX_ATTEMPTS and "assembler_agent" in state["plan"]:
            return {"qa_ok": False, "burn_off": True, "notes": ["qa_retry:" + problem], "trace": [f"qa_agent (retry: {problem})"]}
        raise MediaError(f"The finished video failed the check: {problem}")

    offset = M.CARD_SECONDS if state.get("intro") else 0.0
    chapters = [{**c, "start": round(c["start"] + offset, 1)} for c in state.get("chapters", [])]
    if chapters and offset:
        chapters.insert(0, {"title": state["opts"].get("title") or "Intro", "start": 0.0})
    segments = state.get("segments", [])
    notes = list(dict.fromkeys(state.get("notes", [])))
    return {"qa_ok": True, "trace": ["qa_agent"], "result": {
        "file": final, "srt": M.to_srt(segments, offset) if segments else "", "captions": len(segments),
        "chapters": chapters, "cut": state.get("cut", {"removed_seconds": 0.0, "pieces": 0}), "notes": notes,
        "seconds": round(M.probe(final)["seconds"], 1)}}


NODES = {"analyst_agent": analyst_agent, "cutter_agent": cutter_agent, "transcriber_agent": transcriber_agent,
         "chapters_agent": chapters_agent, "designer_agent": designer_agent, "sticker_agent": sticker_agent, "assembler_agent": assembler_agent,
         "qa_agent": qa_agent}

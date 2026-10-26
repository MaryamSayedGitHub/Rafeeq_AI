/* M5 montage editor. Every control changes the preview at once; "Create final video" sends the decisions
   (pieces to keep, captions, chapters, cards, music) to the montage agent team, which renders them. */
(() => {
  const root = document.getElementById("ed");
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const I18N = JSON.parse($("ed-i18n").textContent);
  const MUSIC = JSON.parse($("ed-music").textContent);
  const PICS = JSON.parse($("ed-pics").textContent);
  const MAX_ST = 12;
  const CARD_SECONDS = 3;
  const tt = (k, v) => (I18N[k] || k).replace(/\{(\w+)\}/g, (_, x) => (v && x in v ? v[x] : ""));
  const csrf = root.dataset.csrf || "";

  const video = $("ed-video");
  video.src = root.dataset.video;
  const S = {
    dur: 0, pad: 0.2, silences: [], off: new Set(), pausesOn: true, minSil: 1.0, analyzed: false,
    trim: { a: 0, b: 0 }, segs: [], chaps: [], preview: true, music: null, introTimer: null, st: [], sel: null, stSeq: 0, gifsLoaded: false,
  };

  // ---------------------------------------------------------------- helpers
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const fmt = (s) => { s = Math.max(0, s); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
  const fmt1 = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
  const parseTime = (txt) => {
    const m = String(txt).trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/);
    return m ? (m[1] ? +m[1] * 60 : 0) + +m[2] : NaN;
  };
  const pct = (t) => (S.dur ? (clamp(t, 0, S.dur) / S.dur) * 100 : 0);
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const post = async (action, body) => {
    const res = await fetch(`${root.dataset.base}/${action}`, {
      method: "POST", credentials: "same-origin",
      headers: { "X-CSRF-Token": csrf, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    return res;
  };

  // ---------------------------------------------------------------- the edit model
  const pauseRanges = () => S.silences.map(([s, e], i) => ({
    i, a: s + S.pad, b: e - S.pad, len: e - s, shown: e - s >= S.minSil && e - s > 2 * S.pad + 0.1, on: !S.off.has(i),
  })).filter((r) => r.shown);
  const activeCuts = () => {
    if (!S.pausesOn) return [];
    return pauseRanges().filter((r) => r.on).map((r) => ({ a: Math.max(r.a, S.trim.a), b: Math.min(r.b, S.trim.b) })).filter((r) => r.b - r.a > 0.05);
  };
  const keepRanges = () => {
    const out = []; let cur = S.trim.a;
    for (const c of activeCuts().sort((x, y) => x.a - y.a)) {
      if (c.a - cur > 0.1) out.push([cur, c.a]);
      cur = Math.max(cur, c.b);
    }
    if (S.trim.b - cur > 0.1) out.push([cur, S.trim.b]);
    return out;
  };
  const isFull = (keep) => keep.length === 1 && keep[0][0] <= 0.05 && keep[0][1] >= S.dur - 0.05;
  const cardsOn = () => ({ intro: $("ed-intro-on").checked, outro: $("ed-outro-on").checked });
  const newLength = () => {
    const c = cardsOn();
    return keepRanges().reduce((s, [a, b]) => s + b - a, 0) + CARD_SECONDS * (+c.intro + +c.outro);
  };

  // ---------------------------------------------------------------- drawing
  function drawCuts() {
    const live = $("ed-cuts"), ghost = $("ed-cuts-v"); live.textContent = ""; ghost.textContent = "";
    for (const r of pauseRanges()) {
      const on = r.on && S.pausesOn, left = pct(r.a) + "%", width = Math.max(0.3, pct(r.b) - pct(r.a)) + "%";
      const b = el("button", "ed-cut" + (on ? "" : " off")); b.type = "button"; b.style.left = left; b.style.width = width;
      b.title = `${fmt1(r.a)} – ${fmt1(r.b)}`; b.setAttribute("aria-pressed", on);
      b.addEventListener("click", (e) => { e.stopPropagation(); r.on ? S.off.add(r.i) : S.off.delete(r.i); refresh(); });
      live.appendChild(b);
      const g = el("div", "ed-cut-ghost" + (on ? "" : " off")); g.style.left = left; g.style.width = width; ghost.appendChild(g);
    }
  }
  function drawSummaries() {
    const set = (k, txt) => { const n = $("sum-" + k); if (n) n.textContent = txt; };
    const rs = pauseRanges().filter((r) => r.on);
    set("pauses", !S.analyzed ? "…" : !S.pausesOn || !rs.length ? tt("sum_off") : `${rs.length} · −${Math.round(rs.reduce((s, r) => s + r.b - r.a, 0))}s`);
    set("trim", `${fmt(S.trim.a)} → ${fmt(S.trim.b)}`);
    set("captions", S.segs.length && $("ed-cap-on").checked ? tt("sum_lines", { n: S.segs.length }) : tt("sum_none"));
    set("chapters", S.chaps.length ? tt("sum_chapters", { n: S.chaps.length }) : tt("sum_none"));
    const c = cardsOn(); set("cards", c.intro || c.outro ? [c.intro ? "▶" : "", c.outro ? "■" : ""].join(" ").trim() : tt("sum_off"));
    set("music", S.music ? S.music.title : tt("sum_none"));
    set("gifs", S.st.length ? String(S.st.length) : tt("sum_none"));
    $("ed-chip-in").hidden = !c.intro; $("ed-chip-out").hidden = !c.outro;
    const mb = $("ed-music-bar"); mb.hidden = !S.music; if (S.music) mb.textContent = "♪ " + S.music.title;
  }
  function drawCaps() {
    const box = $("ed-caps"); box.textContent = "";
    if (!$("ed-cap-on").checked) return;
    for (const s of S.segs) {
      const d = el("div", "ed-capblock");
      d.style.left = pct(s.start) + "%"; d.style.width = Math.max(0.2, pct(s.end) - pct(s.start)) + "%";
      box.appendChild(d);
    }
  }
  function drawChapters() {
    const box = $("ed-chs"); box.textContent = "";
    for (const c of S.chaps) {
      const d = el("button", "ed-flag"); d.type = "button"; d.title = c.title;
      d.style.left = pct(c.start) + "%";
      d.addEventListener("click", (e) => { e.stopPropagation(); seek(c.start); });
      box.appendChild(d);
    }
  }
  function drawTrim() {
    $("ed-shade-a").style.width = pct(S.trim.a) + "%";
    $("ed-shade-b").style.width = 100 - pct(S.trim.b) + "%";
    $("ed-handle-a").style.left = pct(S.trim.a) + "%";
    $("ed-handle-b").style.left = pct(S.trim.b) + "%";
    $("ed-trim-a").value = fmt1(S.trim.a); $("ed-trim-b").value = fmt1(S.trim.b);
  }
  function drawTicks() {
    const box = $("ed-ticks"); box.textContent = "";
    if (!S.dur) return;
    const step = [5, 10, 15, 30, 60, 120, 300].find((s) => S.dur / s <= 12) || 600;
    for (let t = 0; t <= S.dur; t += step) { const d = el("span", "ed-tick", fmt(t)); d.style.left = pct(t) + "%"; box.appendChild(d); }
  }
  function drawLength() {
    const n = newLength();
    $("ed-newlen").textContent = fmt(n);
    $("ed-total").textContent = fmt(S.dur);
    $("ed-bar-new").style.inlineSize = (S.dur ? Math.min(100, (n / S.dur) * 100) : 0) + "%";
    const diff = S.dur ? Math.round(((n - S.dur) / S.dur) * 100) : 0, d = $("ed-delta");
    d.textContent = diff === 0 ? "" : `${diff > 0 ? "+" : "−"}${Math.abs(diff)}%`; d.className = diff < 0 ? "less" : diff > 0 ? "more" : "";
    drawSummaries();
  }
  function drawPauseList() {
    const st = $("ed-pauses-status"), ul = $("ed-pauses-list"); ul.textContent = "";
    if (!S.analyzed) return;
    const rs = pauseRanges();
    const saved = rs.filter((r) => r.on && S.pausesOn).reduce((s, r) => s + (r.b - r.a), 0);
    st.textContent = rs.length ? tt("pauses_found", { n: rs.length, sec: Math.round(saved) }) : tt("no_pauses");
    for (const r of rs) {
      const li = el("li", "ed-item");
      const cb = el("input"); cb.type = "checkbox"; cb.checked = r.on;
      cb.addEventListener("change", () => { cb.checked ? S.off.delete(r.i) : S.off.add(r.i); refresh(); });
      const when = el("button", "ed-link", `${fmt1(r.a)} – ${fmt1(r.b)}`); when.type = "button"; when.dir = "ltr";
      when.addEventListener("click", () => seek(Math.max(0, r.a - 1)));
      li.append(cb, when, el("span", "ed-dim", `${(r.b - r.a).toFixed(1)} s`));
      ul.appendChild(li);
    }
  }
  function drawCapList() {
    const ul = $("ed-cap-list"); ul.textContent = "";
    S.segs.forEach((s, i) => {
      const li = el("li", "ed-item");
      const when = el("button", "ed-link", fmt(s.start)); when.type = "button"; when.dir = "ltr";
      when.addEventListener("click", () => seek(s.start));
      const inp = el("input", "ed-input"); inp.value = s.text; inp.dir = "auto"; inp.placeholder = tt("line_placeholder");
      inp.addEventListener("input", () => { s.text = inp.value; showCaption(); });
      const del = el("button", "ed-x", "×"); del.type = "button"; del.setAttribute("aria-label", "delete");
      del.addEventListener("click", () => { S.segs.splice(i, 1); drawCapList(); drawCaps(); drawSummaries(); });
      li.append(when, inp, del); ul.appendChild(li);
    });
  }
  function drawChapList() {
    const ul = $("ed-ch-list"); ul.textContent = "";
    S.chaps.sort((x, y) => x.start - y.start);
    S.chaps.forEach((c, i) => {
      const li = el("li", "ed-item");
      const t = el("input", "ed-input narrow"); t.value = fmt1(c.start); t.dir = "ltr";
      t.addEventListener("change", () => { const v = parseTime(t.value); if (!isNaN(v)) { c.start = clamp(v, 0, S.dur); drawChapList(); drawChapters(); } });
      const inp = el("input", "ed-input"); inp.value = c.title; inp.dir = "auto"; inp.placeholder = tt("chapter_placeholder");
      inp.addEventListener("input", () => { c.title = inp.value; });
      const go = el("button", "ed-link", "▶"); go.type = "button"; go.addEventListener("click", () => seek(c.start));
      const del = el("button", "ed-x", "×"); del.type = "button"; del.setAttribute("aria-label", "delete");
      del.addEventListener("click", () => { S.chaps.splice(i, 1); drawChapList(); drawChapters(); drawSummaries(); });
      li.append(go, t, inp, del); ul.appendChild(li);
    });
    if (!S.chaps.length) ul.appendChild(el("li", "ed-dim", tt("no_chapters")));
  }
  function refresh() { drawCuts(); drawCaps(); drawChapters(); drawTrim(); drawLength(); drawPauseList(); drawGifLane(); }

  // ---------------------------------------------------------------- the preview
  function showCaption() {
    const box = $("ed-caption"), t = video.currentTime;
    const s = $("ed-cap-on").checked && S.segs.find((x) => x.start <= t && t < x.end);
    box.textContent = s ? s.text : ""; box.hidden = !s;
  }
  function showBadge() {
    const t = video.currentTime, b = $("ed-badge");
    const c = S.pausesOn && pauseRanges().find((r) => r.on && t >= r.a && t < r.b);
    b.hidden = !c; if (c) b.textContent = "✂ " + fmt1(c.b - c.a) + "s";
  }
  function showCard(kind, ms, done) {
    const card = $("ed-card");
    $("ed-card-title").textContent = kind === "intro" ? ($("ed-c-title").value || root.dataset.title) : ($("ed-c-outro").value || $("ed-c-sub").value || $("ed-c-title").value);
    $("ed-card-sub").textContent = kind === "intro" ? $("ed-c-sub").value : "";
    card.hidden = false; card.classList.add("show");
    clearTimeout(S.introTimer);
    S.introTimer = setTimeout(() => { card.hidden = true; card.classList.remove("show"); if (done) done(); }, ms);
  }
  const musicEl = new Audio(); musicEl.loop = true; musicEl.volume = 0.12;
  const musicWanted = () => S.music && ($("ed-music-prev") ? $("ed-music-prev").checked : true);

  function setPlayIcon(playing) {
    root.querySelector(".i-play").hidden = playing; root.querySelector(".i-pause").hidden = !playing;
  }
  function tick() {
    const t = video.currentTime;
    if (!video.paused && S.preview) {
      if (t < S.trim.a - 0.05) video.currentTime = S.trim.a;
      else if (t >= S.trim.b - 0.03) { finishPlayback(); return; }
      else { const c = activeCuts().find((r) => t >= r.a && t < r.b); if (c) video.currentTime = c.b; }
    }
    $("ed-head").style.left = pct(t) + "%";
    $("ed-now").textContent = fmt(t);
    showCaption(); showBadge(); showStickers();
    if (!video.paused) requestAnimationFrame(tick);
  }
  function finishPlayback() {
    video.pause(); video.currentTime = S.trim.b; musicEl.pause();
    if (S.preview && cardsOn().outro) showCard("outro", CARD_SECONDS * 1000);
  }
  function startPlayback() {
    const go = () => { video.play().catch(() => {}); if (musicWanted()) musicEl.play().catch(() => {}); };
    const t = video.currentTime;
    if (S.preview && t >= S.trim.b - 0.05) video.currentTime = S.trim.a;
    if (S.preview && video.currentTime <= S.trim.a + 0.1 && cardsOn().intro) { showCard("intro", CARD_SECONDS * 1000, go); return; }
    go();
  }
  function seek(t) { video.currentTime = clamp(t, 0, S.dur || t); tick(); }
  video.addEventListener("play", () => { setPlayIcon(true); requestAnimationFrame(tick); });
  video.addEventListener("pause", () => { setPlayIcon(false); musicEl.pause(); });
  video.addEventListener("seeked", tick);
  video.addEventListener("timeupdate", () => { if (video.paused) tick(); });
  $("ed-play").addEventListener("click", () => (video.paused ? startPlayback() : video.pause()));
  $("ed-preview").addEventListener("change", (e) => { S.preview = e.target.checked; });
  document.addEventListener("keydown", (e) => {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); $("ed-play").click(); }
    if (e.code === "ArrowLeft") seek(video.currentTime - 5);
    if (e.code === "ArrowRight") seek(video.currentTime + 5);
  });

  // ---------------------------------------------------------------- the timeline
  const track = $("ed-tracks");
  const timeAt = (ev) => { const r = track.getBoundingClientRect(); return clamp((ev.clientX - r.left) / r.width, 0, 1) * S.dur; };
  let drag = null;
  track.addEventListener("pointerdown", (ev) => {
    if (!S.dur) return;
    drag = ev.target.id === "ed-handle-a" ? "a" : ev.target.id === "ed-handle-b" ? "b" : "head";
    track.setPointerCapture(ev.pointerId);
    onDrag(ev);
  });
  track.addEventListener("pointermove", (ev) => { if (drag) onDrag(ev); });
  track.addEventListener("pointerup", () => { drag = null; });
  function onDrag(ev) {
    const t = timeAt(ev);
    if (drag === "a") S.trim.a = clamp(t, 0, S.trim.b - 1);
    else if (drag === "b") S.trim.b = clamp(t, S.trim.a + 1, S.dur);
    else seek(t);
    if (drag !== "head") refresh();
  }
  for (const id of ["ed-handle-a", "ed-handle-b"]) {
    $(id).addEventListener("keydown", (e) => {
      const d = e.code === "ArrowLeft" ? -1 : e.code === "ArrowRight" ? 1 : 0; if (!d) return;
      e.preventDefault();
      if (id === "ed-handle-a") S.trim.a = clamp(S.trim.a + d, 0, S.trim.b - 1); else S.trim.b = clamp(S.trim.b + d, S.trim.a + 1, S.dur);
      refresh();
    });
  }
  async function drawFilm() {
    const cv = $("ed-film"), w = cv.parentElement.clientWidth || 800; cv.width = w; cv.height = 56;
    const ctx = cv.getContext("2d"); ctx.fillStyle = "#e9dfe6"; ctx.fillRect(0, 0, w, 56);
    const v = document.createElement("video"); v.muted = true; v.preload = "auto"; v.src = root.dataset.video;
    const n = Math.max(4, Math.round(w / 100)), tw = w / n;
    const wait = (ev) => new Promise((ok, no) => { const to = setTimeout(() => no(new Error("timeout")), 6000); v.addEventListener(ev, () => { clearTimeout(to); ok(); }, { once: true }); });
    try {
      await wait("loadeddata");
      for (let i = 0; i < n; i++) {
        v.currentTime = ((i + 0.5) / n) * S.dur; await wait("seeked");
        ctx.drawImage(v, i * tw, 0, tw + 1, 56);
      }
    } catch (e) { /* the strip is decoration only */ }
  }

  // ---------------------------------------------------------------- tools
  root.querySelectorAll(".ed-tool").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(".ed-tool").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    root.querySelectorAll(".ed-pane").forEach((p) => { p.hidden = p.dataset.pane !== b.dataset.tool; });
  }));
  $("ed-pauses-on").addEventListener("change", (e) => { S.pausesOn = e.target.checked; refresh(); });
  $("ed-min").addEventListener("input", (e) => { S.minSil = +e.target.value; $("ed-min-label").textContent = `${S.minSil.toFixed(1)} s`; refresh(); });
  for (const [id, key] of [["ed-trim-a", "a"], ["ed-trim-b", "b"]]) {
    $(id).addEventListener("change", (e) => {
      const v = parseTime(e.target.value); if (isNaN(v)) return drawTrim();
      if (key === "a") S.trim.a = clamp(v, 0, S.trim.b - 1); else S.trim.b = clamp(v, S.trim.a + 1, S.dur);
      refresh();
    });
  }
  $("ed-set-a").addEventListener("click", () => { S.trim.a = clamp(video.currentTime, 0, S.trim.b - 1); refresh(); });
  $("ed-set-b").addEventListener("click", () => { S.trim.b = clamp(video.currentTime, S.trim.a + 1, S.dur); refresh(); });
  $("ed-cap-on").addEventListener("change", () => { drawCaps(); showCaption(); drawSummaries(); });
  $("ed-add-ch").addEventListener("click", () => { S.chaps.push({ start: Math.round(video.currentTime * 10) / 10, title: tt("chapter_default") }); drawChapList(); drawChapters(); drawSummaries(); });
  for (const id of ["ed-intro-on", "ed-outro-on"]) $(id).addEventListener("change", drawLength);
  $("ed-prev-intro").addEventListener("click", () => { $("ed-intro-on").checked = true; drawLength(); showCard("intro", CARD_SECONDS * 1000); });
  $("ed-prev-outro").addEventListener("click", () => { $("ed-outro-on").checked = true; drawLength(); showCard("outro", CARD_SECONDS * 1000); });
  const sel = $("ed-music-sel");
  if (sel) sel.addEventListener("change", () => {
    S.music = MUSIC.find((m) => m.id === sel.value) || null; drawSummaries();
    musicEl.pause(); if (S.music) { musicEl.src = S.music.url; if (!video.paused && musicWanted()) musicEl.play().catch(() => {}); }
  });
  if ($("ed-music-prev")) $("ed-music-prev").addEventListener("change", () => { if (!musicWanted()) musicEl.pause(); else if (!video.paused) musicEl.play().catch(() => {}); });

  $("ed-write").addEventListener("click", async () => {
    const st = $("ed-cap-status"), btn = $("ed-write");
    btn.disabled = true; st.textContent = tt("writing"); st.className = "ed-status busy";
    try {
      const res = await post("transcribe"), data = await res.json();
      if (!res.ok) throw new Error(data.error || tt("write_failed"));
      S.segs = data.segments || []; S.chaps = (data.chapters || []).slice(); drawSummaries();
      st.textContent = S.segs.length ? "" : tt("no_captions"); st.className = "ed-status";
      drawCapList(); drawChapList(); refresh();
    } catch (e) { st.textContent = e.message; st.className = "ed-status bad"; }
    btn.disabled = false;
  });

  function reset() {
    S.off.clear(); S.pausesOn = true; S.minSil = 1.0; S.trim = { a: 0, b: S.dur }; S.segs = []; S.chaps = [];
    $("ed-pauses-on").checked = true; $("ed-min").value = 1; $("ed-min-label").textContent = "1.0 s";
    $("ed-intro-on").checked = false; $("ed-outro-on").checked = false; $("ed-cap-on").checked = true;
    if (sel) { sel.value = ""; S.music = null; musicEl.pause(); }
    S.st = []; S.sel = null; buildStickerEls(); drawStList();
    drawCapList(); drawChapList(); refresh();
  }
  $("ed-reset").addEventListener("click", reset);

  // ---------------------------------------------------------------- GIFs and pictures on the video
  const stl = $("ed-stl");
  function fitLayer() {                       // the layer covers exactly the picture of the video (not the black bars)
    const stage = $("ed-stage"), W = stage.clientWidth, H = stage.clientHeight;
    const vw = video.videoWidth || 16, vh = video.videoHeight || 9, k = Math.min(W / vw, H / vh);
    const w = vw * k, h = vh * k;
    Object.assign(stl.style, { left: (W - w) / 2 + "px", top: (H - h) / 2 + "px", width: w + "px", height: h + "px" });
  }
  const stById = (id) => S.st.find((k) => k.id === id);
  function buildStickerEls() {
    stl.textContent = "";
    for (const k of S.st) {
      const im = el("img"); im.src = k.url; im.alt = ""; im.draggable = false; im.dataset.id = k.id;
      im.addEventListener("pointerdown", (ev) => { ev.preventDefault(); ev.stopPropagation(); select(k.id); stDrag = { id: k.id }; im.setPointerCapture(ev.pointerId); });
      im.addEventListener("pointermove", (ev) => {
        if (!stDrag || stDrag.id !== k.id) return;
        const r = stl.getBoundingClientRect();
        k.x = clamp((ev.clientX - r.left) / r.width, 0, 1); k.y = clamp((ev.clientY - r.top) / r.height, 0, 1);
        placeSticker(im, k);
      });
      im.addEventListener("pointerup", () => { stDrag = null; });
      placeSticker(im, k); stl.appendChild(im);
    }
    showStickers();
  }
  let stDrag = null;
  function placeSticker(im, k) { im.style.left = k.x * 100 + "%"; im.style.top = k.y * 100 + "%"; im.style.width = k.w * 100 + "%"; }
  function showStickers() {
    const t = video.currentTime;
    for (const im of stl.children) {
      const k = stById(im.dataset.id); if (!k) continue;
      const on = t >= k.start && t < k.end;
      im.style.display = on || k.id === S.sel && video.paused ? "" : "none";
      im.style.opacity = on ? "1" : ".45";
      im.classList.toggle("sel", k.id === S.sel);
    }
  }
  function select(id) {
    S.sel = id; showStickers(); drawGifLane(); drawStList();
  }
  function drawGifLane() {
    const box = $("ed-gifs"); if (!box) return; box.textContent = "";
    for (const k of S.st) {
      const b = el("button", "ed-gifblock" + (k.id === S.sel ? " sel" : ""), k.label || "GIF"); b.type = "button";
      b.style.left = pct(k.start) + "%"; b.style.width = Math.max(0.5, pct(k.end) - pct(k.start)) + "%"; b.title = `${fmt1(k.start)} – ${fmt1(k.end)}`;
      b.addEventListener("click", (e) => { e.stopPropagation(); select(k.id); seek(k.start); });
      box.appendChild(b);
    }
    drawSummaries();
  }
  function drawStList() {
    const ul = $("ed-st-list"); ul.textContent = "";
    if (!S.st.length) ul.appendChild(el("li", "ed-dim", tt("gifs.empty")));
    for (const k of S.st) {
      const li = el("li", "ed-st" + (k.id === S.sel ? " sel" : ""));
      const im = el("img"); im.src = k.thumb || k.url; im.alt = ""; im.addEventListener("click", () => { select(k.id); seek(k.start); });
      const times = el("div", "ed-sttimes");
      const mk = (key) => {
        const inp = el("input", "ed-input narrow"); inp.value = fmt1(k[key]); inp.dir = "ltr"; inp.inputMode = "decimal";
        inp.addEventListener("change", () => {
          const v = parseTime(inp.value);
          if (!isNaN(v)) { if (key === "start") k.start = clamp(v, 0, k.end - 0.5); else k.end = clamp(v, k.start + 0.5, S.dur); }
          drawGifLane(); drawStList(); showStickers();
        });
        return inp;
      };
      const here = (key) => {
        const b = el("button", "ed-mini", tt(key === "start" ? "gifs.here" : "gifs.end_here")); b.type = "button";
        b.addEventListener("click", () => {
          const t = video.currentTime;
          if (key === "start") k.start = clamp(t, 0, k.end - 0.5); else k.end = clamp(t, k.start + 0.5, S.dur);
          drawGifLane(); drawStList(); showStickers();
        });
        return b;
      };
      times.append(el("span", null, tt("gifs.from")), mk("start"), here("start"), el("span", null, tt("gifs.to")), mk("end"), here("end"));
      const del = el("button", "ed-x", "×"); del.type = "button"; del.title = tt("gifs.remove"); del.setAttribute("aria-label", tt("gifs.remove"));
      del.addEventListener("click", () => { S.st = S.st.filter((x) => x !== k); if (S.sel === k.id) S.sel = null; buildStickerEls(); drawStList(); drawGifLane(); });
      const sz = el("div", "ed-szrow"); const rg = el("input"); rg.type = "range"; rg.min = 5; rg.max = 70; rg.value = Math.round(k.w * 100);
      rg.addEventListener("input", () => { k.w = +rg.value / 100; const im2 = [...stl.children].find((n) => n.dataset.id === k.id); if (im2) placeSticker(im2, k); });
      sz.append(el("span", null, tt("gifs.size")), rg);
      li.append(im, times, del, sz); ul.appendChild(li);
    }
  }
  function addSticker(src, url, thumb, label) {
    if (S.st.length >= MAX_ST) { const st = $("ed-gif-status"); st.textContent = tt("gifs.limit"); st.className = "ed-status bad"; return; }
    const start = clamp(video.currentTime, 0, Math.max(0, S.dur - 0.5));
    const k = { id: "s" + (++S.stSeq), src, url, thumb, label: (label || "GIF").slice(0, 24), start, end: Math.min(S.dur, start + 3), x: 0.8, y: 0.74, w: 0.22 };
    S.st.push(k); S.sel = k.id; buildStickerEls(); drawStList(); drawGifLane();
  }
  function gridButton(url, thumb, label, onclick) {
    const b = el("button", "ed-gifbtn"); b.type = "button"; b.title = label || "";
    const im = el("img"); im.src = thumb; im.alt = label || ""; im.loading = "lazy"; b.appendChild(im);
    b.addEventListener("click", onclick); return b;
  }
  function drawPics() {
    const g = $("ed-pic-grid"); g.textContent = "";
    if (!PICS.length) { g.after(Object.assign(el("p", "ed-hint"), { textContent: tt("gifs.library_none") })); return; }
    for (const p of PICS) g.appendChild(gridButton(p.url, p.url, p.title, () => addSticker({ asset_id: p.id }, p.url, p.url, p.title)));
  }
  async function searchGifs(q) {
    const st = $("ed-gif-status"), grid = $("ed-gif-grid"), go = $("ed-gif-go");
    go.disabled = true; st.textContent = tt("gifs.searching"); st.className = "ed-status busy";
    try {
      const res = await post("gifs", { q }), data = await res.json();
      if (!res.ok) throw new Error(data.error || tt("gifs.failed"));
      grid.textContent = "";
      for (const g of data.gifs || []) grid.appendChild(gridButton(g.url, g.thumb, g.title, () => addSticker({ giphy: g.url }, g.url, g.thumb, g.title)));
      st.textContent = (data.gifs || []).length ? "" : tt("gifs.none"); st.className = "ed-status";
      S.gifsLoaded = true;
    } catch (e) { st.textContent = e.message; st.className = "ed-status bad"; }
    go.disabled = false;
  }
  $("ed-gif-form").addEventListener("submit", (e) => { e.preventDefault(); searchGifs($("ed-gif-q").value.trim()); });
  root.querySelector('[data-tool="gifs"]').addEventListener("click", () => { if (!S.gifsLoaded) searchGifs(""); });
  video.addEventListener("pause", showStickers); video.addEventListener("play", showStickers);
  drawPics(); drawStList();

  // ---------------------------------------------------------------- render
  function decisions() {
    const keep = keepRanges(), c = cardsOn();
    const body = { title: $("ed-c-title").value, subtitle: $("ed-c-sub").value, outro_text: $("ed-c-outro").value, intro: c.intro, outro: c.outro };
    if (keep.length && !isFull(keep)) body.keep = keep.map(([a, b]) => [+a.toFixed(2), +b.toFixed(2)]);
    const segs = S.segs.filter((s) => s.text.trim());
    if ($("ed-cap-on").checked && segs.length) { body.captions = true; body.segments = segs; body.burn_captions = $("ed-burn").checked; }
    const chs = S.chaps.filter((c2) => c2.title.trim());
    if (chs.length) { body.chapters = true; body.chapter_list = chs; }
    if (S.music) body.music_asset_id = S.music.id;
    if (S.st.length) body.stickers = S.st.map((k) => ({ start: +k.start.toFixed(2), end: +k.end.toFixed(2), x: +k.x.toFixed(3), y: +k.y.toFixed(3), w: +k.w.toFixed(3), src: k.src }));
    return body;
  }
  $("ed-render").addEventListener("click", async () => {
    const st = $("ed-render-status"), btn = $("ed-render"), body = decisions();
    if (!body.keep && !body.intro && !body.outro && !body.segments && !body.chapter_list && !body.music_asset_id && !body.stickers) { st.textContent = tt("nothing_to_do"); st.className = "ed-status bad"; return; }
    btn.disabled = true; st.textContent = tt("sending"); st.className = "ed-status busy";
    try {
      const res = await post("render", body), text = await res.text();
      if (!res.ok) { let msg = tt("render_failed"); try { msg = JSON.parse(text).error || msg; } catch (e) { /* keep */ } throw new Error(msg); }
      $("ed-result").innerHTML = text; st.textContent = ""; st.className = "ed-status";
      if (window.htmx) window.htmx.process($("ed-result"));
      $("ed-result").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) { st.textContent = e.message; st.className = "ed-status bad"; }
    btn.disabled = false;
  });

  // ---------------------------------------------------------------- start
  async function analyze() {
    const st = $("ed-pauses-status"); st.textContent = tt("analyzing"); st.className = "ed-status busy";
    try {
      const res = await post("analyze"), data = await res.json();
      if (!res.ok) throw new Error(data.error || tt("analyze_failed"));
      S.silences = data.silences || []; S.pad = data.pad || 0.2; S.analyzed = true;
      if (data.min_silence) { $("ed-min").min = data.min_silence; S.minSil = Math.max(S.minSil, data.min_silence); $("ed-min").value = S.minSil; $("ed-min-label").textContent = `${S.minSil.toFixed(1)} s`; }
      if (!data.audio) st.textContent = tt("no_audio"); else st.textContent = "";
      st.className = "ed-status"; refresh();
    } catch (e) { st.textContent = e.message; st.className = "ed-status bad"; }
  }
  video.addEventListener("loadedmetadata", () => {
    S.dur = video.duration; S.trim = { a: 0, b: S.dur };
    fitLayer(); drawTicks(); refresh(); drawCapList(); drawChapList(); drawFilm(); analyze();
  }, { once: true });
  window.addEventListener("resize", () => { drawTicks(); fitLayer(); });
})();

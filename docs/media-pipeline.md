# Media pipeline: story videos, GIFs, and the local-diffusion plan

What is **built and tested** vs. what is **a plan that needs a GPU server** is marked on every line.

## 1. Built: ~30 s story video (`/app/story`, `POST /teacher/stories`)

Code: `rafeeq/teacher/story_video.py`. Needs `DASHSCOPE_API_KEY` (Wan) and `ffmpeg`.

| Step | What happens | Why it keeps the look stable |
|---|---|---|
| Plan | A text LLM writes ONE character, ONE setting, N scenes (visual + narration) | the character/setting/style text is copied into every scene prompt |
| Scene 1 | text-to-video, 5 s | fixed `seed`, `prompt_extend=false` (the provider no longer rewrites the prompt) |
| Scene k | image-to-video from the **last frame of scene k-1** | the picture itself carries character and background forward |
| Optional | local vision model describes the last frame -> added to the next prompt | "previous scene ended like this: ..." |
| Join | ffmpeg: same size/fps, 0.5 s cross-fade (or cut), optional edge-tts narration | no flicker at the joins |
| Save | one `generated_video` library asset | usable in Media Studio, courses, slides (as GIF) |

6 scenes x 5 s = 30 s = 6 paid jobs. Tested with real ffmpeg: joining, fades, last frame, narration track, GIF;
the chain (submit -> poll -> next scene -> join) is tested with a fake provider. **Not tested against the live Wan service.**
Known limit: faces and hands drift a little over six hops; background and colours hold.

## 2. Built: GIFs on slides

`Media Studio -> video -> "convert to GIF"` makes a looping, palette-optimised GIF (image asset). Put it on a slide in
Presentation Studio like any picture: the `.pptx` keeps the real GIF (checked: `ppt/media/image1.gif`), and PowerPoint
plays it in slideshow mode. GIFs are not editable as pictures (it would flatten them).

## 3. Your two local models (Ollama)

- `qwen3:8b` -> **the story planner and slide-text writer** (text only). Set `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen3:8b`
  or only for stories: `STORY_LLM_PROVIDER=ollama`. Its `<think>` output is stripped. An 8B model is acceptable for
  English prompts, weaker for Arabic JSON: if scenes come back malformed, plan in English and narrate in Arabic.
- `qwen2.5vl:7b` -> **a checker, not a generator**: the "describe the last frame" option in the story page
  (`OLLAMA_VL_MODEL`, default `qwen2.5vl:7b`). Neither model can make images or video.

## 4. Plan (not built): local SDXL / SD3 / ControlNet / AnimateDiff / SAM2

Needs an NVIDIA GPU (12 GB+ VRAM comfortable, 8 GB workable with SDXL fp16) and a separate worker; it does not run inside
the FastAPI process. Suggested shape: a small ComfyUI (or diffusers) HTTP worker behind the same job interface the app
already uses (`submit -> task_id -> poll -> file`), selected by `IMAGE_PROVIDER=wan|comfyui`.

Keeping a character/background across scenes, in order of payoff:
1. **IP-Adapter / reference image** (character sheet generated once): weight 0.6-0.8. This does most of the work.
2. **Fixed seed + identical style/character text** in every prompt (already done for Wan).
3. **ControlNet** (depth or canny from the previous frame, scale 0.6-0.8, end at 0.8) to hold layout between scenes.
4. **SAM 2** only where you need to *edit one element*: click/box the subject on frame 0 -> propagate the mask through
   the clip -> dilate 8-16 px and feather -> use it as the mask for inpainting (swap background, recolour, remove an
   object). It does not create consistency by itself.
5. **AnimateDiff** gives 2-4 s clips per generation, so it needs the same last-frame chaining as above.

Starting settings (tune per model):

| Model | Steps | CFG | Sampler | Notes |
|---|---|---|---|---|
| SDXL | 30-40 | 5-7 | DPM++ 2M Karras | 1024x1024; negative: "text, watermark, blurry, extra fingers" |
| SD3 medium | 28 | 4-5 | euler | lower CFG than SDXL |
| AnimateDiff (SD1.5, motion v3) | 25 | 7-8 | Euler a / DPM++ SDE | 16 frames at 8 fps; img2img denoise 0.5-0.6 when chaining |
| Wan (current) | - | - | - | 720P, `seed` fixed per story, `prompt_extend=false` |

Higher CFG makes colours harsh and repeats artefacts; more steps stop helping after ~40.

## 5. Why presentations ignored the theme

The first `.pptx` that Lesson Studio writes uses the default Rafeeq theme. The chosen theme is applied when you
**export from Presentation Studio** (`/app/presentations/{deck}`): colours, font, and your pictures/GIFs. Export again
after changing the theme; every export is a new version.

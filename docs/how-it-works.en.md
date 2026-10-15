# How DoF Anime works

[日本語](how-it-works.ja.md) ｜ [README](../README.en.md)

DoF Anime is an After Effects effect that adds depth of field (out-of-focus blur) to finished AI-generated anime videos (plain RGB footage).
This document describes how it is built and how to build it from scratch.

---

## 1. Overview

```
Apply DoF Anime to a footage layer
  │
  ├─ Make depth (once; results stay in cache/, shared with Relight Anime)
  │    AE renders the source frames → relight_depth.exe (ONNX Runtime + CUDA) estimates depth
  │    → the 16-bit depth PNGs are imported and wired as a hidden layer "DoF depth"
  │
  ├─ (optional) Make character mattes (Auto Subject / Generate Subject, Sharp by Text / Blur by Text)
  │    SAM 3.1 finds and tracks them through the clip, edges are re-solved from the image colours → wired as hidden layers
  │
  └─ Blur (every frame)
       Upsample depth along the line art, tighten depth steps, turn the distance from focus into a blur radius,
       and gather near and far separately (scatter-as-gather). The CPU and GPU (CUDA) paths share the same code (DofCore.h)
```

Depth estimation and matte generation run outside AE's renderer (in a separate process) because AE asks for frames in any order and in parallel,
while video depth and matte tracking need neighbouring frames together to stay stable over time.

## 2. Depth

Depth is made exactly as in [Relight Anime](https://github.com/yuichi-suzuki-highdrama/relight-anime); the settings (`%APPDATA%\SRLM\RelightAnime.ini`) and the cache are shared.
Footage that already has depth from Relight Anime is ready at once, and a Relight Anime depth layer in the same comp is reused.

- **Model**: Video Depth Anything Small (Apache-2.0), a video depth model that looks at several frames at once.
  See Relight Anime's [how it works](https://github.com/yuichi-suzuki-highdrama/relight-anime/blob/main/docs/how-it-works.en.md) for inference, normalisation and temporal smoothing.
- **Detail** (`depth_input_size`): the short side fed to the model. At the default 518, thin strands of hair were missing from depth and stayed as sharp lines after blurring.
  At 770 they are captured to the tips (still the Small model, so commercial use is fine).
  For 1376×768 × 192 frames: 518 takes about 19 s with a VRAM peak of about 11 GB; 770 about 32 s and 21 GB (fp32). With fp16 models (fp16 inside, fp32 inputs and outputs; mean difference 0.0002), 770 needs about 12 GB and 1036 about 55 s and 20 GB. At 1036, grass blades that were cut off at 770 are captured to the tips. Large inputs pass the encoder one frame at a time (two at a time peaked at about 25 GB).
- **The Large model** gives finer hair and backgrounds but is **CC-BY-NC-4.0 (non-commercial)**. It is used only by Depth Engine = Remote Quality.

## 3. Blurring

### 3.1 Blur radius (CoC)

- Estimated depth behaves like disparity (inverse distance), so, as with a thin lens, the blur radius is proportional to the depth difference from focus.
  Blur Size is the radius (in 1080p pixels) at a depth difference of 0.5.
- Outside the in-focus band (Focus Range, default 0.06) the blur ramps up smoothly over 0.02, so there is no jump at the edge of the band.
- The radius is signed (near = negative, far = positive). Near Blur / Far Blur scale each side; Near Blur defaults to 0 (no foreground blur).

### 3.2 Gathering (scatter-as-gather)

1. Convert the input to linear light and build 1/2, 1/4 and 1/8 downsamples.
2. For each 16×16 tile, find the largest blur radius that can reach it (tile maximum, then spread to neighbouring tiles).
3. Around each pixel, place samples on a golden-angle spiral (Vogel disk) of equal area (Quality: 48 / 96 / 192 samples).
   A sample whose circle reaches this pixel contributes its colour.
4. Gather two layers:
   - **Background and in-focus layer**: a farther sample's spread is limited to this pixel's own blur, so blurred background never bleeds onto an in-focus subject.
   - **Foreground layer**: nearer samples spread over whatever is behind them. Their coverage gives an alpha, and the layer is composited over the background layer.
5. When samples are far apart, colours are read bilinearly from a downsample chosen from the sample spacing, so no sample pattern (dotted bokeh) or texture shows inside the blur.
6. Bright areas (Highlights / Highlight Threshold) get extra weight in linear light and become bokeh balls.

## 4. Clean edges

Estimated depth is coarse; blur edges land 1–3 px off the line art and show up as thin sharp strips or blurred rims.

| Technique | What it does |
|---|---|
| Snap depth to line art | When upsampling depth to the output size, only where depth changes, depth samples whose source colour is similar get more weight (joint bilateral upsampling). Depth edges follow the line art |
| Edge Expand | Grows the nearer object's depth outward, so outlines just outside the depth edge belong to the near object |
| Step tightening | Where the depth range within 8 px (1080p) is large, each pixel snaps to the nearer end (near or far). No in-between depth remains on a step, so no in-focus line appears along edges |
| Edge Bias | Pushes the blurrier side into the sharper side by the given width (uses a clearly larger blur from the neighbourhood). The strip ends up inside the blur |
| Edge Soften | Smooths far blur with a separable Gaussian and lets it fade into the edge of sharp objects, so edges join softly |

## 5. Character mattes

By default no matte is used; blur comes from depth alone. A matte helps when a whole character should be evenly sharp (or evenly blurred).

### 5.1 Automatic mattes (`tools/subject_mask.py`)

SAM 3.1 is used as a library from a ComfyUI install (ComfyUI's implementation and weights, without starting ComfyUI; run with ComfyUI's Python).

1. On the Focus Point frame, find all characters by text ("anime character" etc.). The one containing the point comes first; up to 3
2. Track each character through the clip (SAM 3.1 video tracking)
3. Re-refine each frame's matte with SAM's fine stage (a 1008-px crop around the character), because tracked mattes are upsampled from low resolution and their edges wave
4. Fill small holes and remove small fragments
5. Fix with depth: drop large blobs inside the matte that are clearly farther than their surroundings; add large blobs just outside that sit at the character's depth
6. Re-solve the outer few pixels from the image colours (closed-form matting, pymatting) and drop faint alpha
7. Write RGB PNGs (red, green, blue = characters 1–3). DoF Anime reads them with Subject Channel = Multi (RGB) and handles each character separately

### 5.2 How the matte is used

| Item | How |
|---|---|
| Subject Flatten | Estimated depth wobbles inside a character (hair tips, legs, frills), so depth inside the matte is pulled toward the character's representative depth (median inside the matte). Uneven blur inside the character disappears |
| Subject Inset | Shrinks the matte inward (default 12 px) and feathers it. The matte then only evens out the inside; the character's edge is decided by depth alone, so jagged matte edges and filled gaps between hair strands do not show |
| Focus with Inset | The band outside the shrunken matte keeps its original depth, so focus is taken from the original depth around the point, and the character under the point is flattened to that same depth (with the median, the band and the inside disagreed) |
| Steps with Inset | Step tightening ignores the matte (treating the shrunken edge as a character boundary drew a line inside the character) |
| Smooth matte | The matte is read as a 3×3 grid of samples (3 px apart at 1080p) weighted 1-2-1, which removes jaggies where two characters meet |
| Keep Subject Sharp | Reduces blur inside the matte regardless of focus (default 0%). It applies to every character in the matte, so out-of-focus characters become sharp too |

### 5.3 Manual fixes

- **Sharp Layer / Blur Layer**: draw a mask on a solid. Sharp Layer sets blur to 0; Blur Layer replaces depth with the farthest background (5th percentile).
- **Sharp by Text / Blur by Text**: type object names (e.g. flower, grass); `tools/prompt_mask.py` finds and tracks them with SAM 3.1 and wires an alpha matte.

## 6. Lens character

The items of After Effects' Camera Lens Blur are implemented as weights in the gather. Each one only changes the test "does the aperture of sample q reach pixel p".

| Item | How |
|---|---|
| Shape / Blades | Aperture shape: the distance to the polygon edge in the direction from q to p (Polygon: 3–10 blades) |
| Roundness | Blends the polygon edge distance with a circle (1) |
| Aspect | Stretches the sample disk into an ellipse (×√a horizontally, ÷√a vertically, same area); tile reach grows to match |
| Cat Eye | Optical vignetting: q's aperture is intersected with a disk shifted along the direction from the frame centre to q. The shift grows toward the frame edge, so balls get clipped |
| Bubble | Positive: brighter rims, weights × `1 + 3b(t⁴ − 1/3)` (t = distance from centre / radius). Negative: brighter centre fading to the rim (`1 − 2b(1/2 − t²)`, `2(1 − t²)` at −100%). The area means of t⁴ and t² are 1/3 and 1/2, so total brightness is unchanged either way. On thin blurred hair strands, positive values show doubled edges and negative values melt them together |
| Fringe | Slightly different aperture radii for red, green and blue (+8% red, −8% blue at 100%), with per-channel weights, giving coloured rims |

## 7. Focus pulls

- The **Focus Point** depth is the median depth around the point (radius 1.2% of the frame height). On a character, the character's representative depth is used.
- **Focus Pull Time** (default 0.8 s) averages the Focus Point over neighbouring frames with a Hann window.
  Like a film focus pull it eases in and out, even across hold keyframes.
- Because other frames' parameters are read, the effect tells AE it uses other times (`PF_OutFlag_WIDE_TIME_INPUT`).
  Without it, AE reused cached frames wherever the Focus Point itself did not change, and the pull stopped.

## 8. Compared with AE's Camera Lens Blur

Same depth, same focus pull, same blur strength and aperture (nonagon, 70% roundness):

| Case | Camera Lens Blur | DoF Anime |
|---|---|---|
| Background blur | about the same | about the same |
| Near side of the in-focus character (legs, hands, frills) | blurred by depth, patchy (sharp hand, blurred frill next to it) | sharp throughout (Focus Range and Subject Flatten) |
| Edge of foreground blur (hair strands) | does not spread over what is behind; thin sharp lines remain on the edges | the foreground layer is composited over the background, so it blurs smoothly |
| Speed (1376×768, 192 frames) | about 8 s | a few ms per frame (GPU) |

## 9. Speed (CPU and GPU)

- The blur maths lives in `plugin/src/DofCore.h`; every function runs on both CPU and CUDA (`__host__ __device__`).
- The CPU path (`DofAE.cpp`, rows on worker threads) and the GPU path (`DofGPU.cu`) each drive the same functions. They differ by at most one 8-bit step.
- GPU scratch memory is counted first and taken as a single block.
- 1376×768: GPU about 3–8 ms (depending on mattes and Quality), CPU about 0.1–0.3 s (RTX 5090 / 24 threads).

## 10. Notes on AE integration

| Topic | Approach |
|---|---|
| Frame order | AE requests frames in any order and in parallel; depth and mattes are made outside the renderer (separate process) |
| Automatic generation | Jobs advance from the idle hook of a helper AEGP (`DoFAnimeHelper.aex`), as in Relight Anime |
| Cache | Keyed by the footage file name, location, size and modification time. Press Generate to rebuild |
| Reading layers | Depth, matte and fix layers are checked out whole (not just the output rect) and matched by relative position in the layer |
| Adding parameters | Parameter IDs are fixed and new ones are appended, so saved projects keep their values |
| GPU | GPU is reported only for CUDA; other frameworks (DirectX etc.) use the CPU path |
| Comp view | A "Focus" frame is drawn at the Focus Point; dragging uses AE's standard 2D point handle |

## 11. Test tools (`tools/harness/`)

| Tool | Purpose |
|---|---|
| `dof_cli` (`run_cli.py`) | Compiles the plugin code as is and blurs one frame without AE (CPU / GPU); shows source / blur / blur map side by side |
| `tools/ae_*.jsx` | Drive AE to apply the effect and render focus pulls (source paths come from environment variables) |

## 12. Building from scratch

Requirements: Windows 11, After Effects 2024 or later (tested on AE 2026), Visual Studio 2022 Build Tools (C++), CMake, Ninja, CUDA Toolkit 12.8 or later, the After Effects SDK.

1. **Depth**: prepare the ONNX models and `relight_depth.exe` as described in Relight Anime (their locations are written to `%APPDATA%\SRLM\RelightAnime.ini`).
   For finer depth, build the fp16 models (`python tools/export_vda_onnx.py --split --fp16`), point `encoder` and `head` at them, and add `depth_input_size=1036` to that file (24 GB VRAM or more)
2. Get the **AE SDK** from https://developer.adobe.com/after-effects/ (not included)
3. Build and install the **plugin** (administrator; close AE first):
   `.\plugin\build.ps1 -SdkRoot <SDK path> -Install`
4. (optional) For **character mattes**, put the SAM 3.1 weights (`models/checkpoints/sam3.1_multiplex_fp16.safetensors`) in ComfyUI and install pymatting into ComfyUI's Python
5. In AE, apply "SRLM > DoF Anime" to a footage layer. Depth is wired automatically when ready

To change settings for DoF Anime only, write the same key=value lines to `%APPDATA%\SRLM\DoFAnime.ini` (read after RelightAnime.ini).

## 13. Licences

| Item | Terms |
|---|---|
| This project | MIT ([LICENSE](../LICENSE)) |
| Video Depth Anything Small | Apache-2.0 |
| Video Depth Anything Large (Remote Quality) | **CC-BY-NC-4.0 (non-commercial)**. Use Small for commercial work (raise detail with `depth_input_size`) |
| SAM 3.1 (character mattes) | Meta SAM License |
| pymatting (edge re-solve) | MIT |
| SAM2Matting (only with `subject_matte=sam2matting`) | **CC BY-NC 4.0 (non-commercial)**; not included |
| After Effects SDK | Adobe SDK licence (not included) |

## 14. Known limitations

- Heavily blurred foreground objects reveal background that is not in the source image; it is filled from surrounding colours
- Thin wind-blown hair tips can be missing from depth and stay as sharp lines (raise `depth_input_size`, or name the hair in Blur by Text)
- Automatic mattes can fill thin gaps between hair strands (with Subject Inset the edge comes from depth, so this barely shows)
- Depth and mattes do not follow time-stretched or time-remapped layers

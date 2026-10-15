# DoF Anime

An After Effects effect that adds depth of field (out-of-focus blur) to AI-generated anime footage.
It estimates depth from the video, focuses on the depth under a point you pick, and blurs what is in front of and behind it.

[日本語](README.md) | How it works: [docs/how-it-works.en.md](docs/how-it-works.en.md)

- **Focus**: drag the Focus Point in the Composition panel to focus on the depth there. Keyframe it for a focus pull; Focus Pull Time (seconds, 0.8 by default) eases the pull in and out like a film focus puller, even across hold keyframes. You can also focus by depth value (Focus Mode = Depth).
- **Clean edges**: near and far blur are computed separately. Blurred background does not bleed onto the in-focus subject, and near blur spreads semi-transparently over what is behind it. Edge Bias / Edge Soften join the small offset between depth and drawn outlines smoothly.
- **Bokeh**: colours are mixed in linear light, so bright spots become bokeh balls. Choose blade count, roundness, aspect, cat eye (optical vignetting), bubble (bright rims) and fringe (coloured rims).
- **Character mattes (optional)**: SAM 3.1 finds and tracks characters. The matte only evens out the inside of a character (Subject Inset); edges come from depth.
- **Speed**: about 2–3 ms per frame on the GPU (CUDA, RTX 5090, 1376×768). The CPU path also works (about 0.1 s at the same size). CPU and GPU differ by at most one 8-bit step.
- **Stable in video**: depth is smoothed over neighbouring frames before use.

Depth estimation uses the same pipeline as [Relight Anime](https://github.com/yuichi-suzuki-highdrama/relight-anime) (Video Depth Anything, via `relight_depth.exe` or a remote depth server). Settings and cache are shared, so footage already processed by Relight Anime is used right away, and an existing Relight Anime depth layer in the same comp is reused.

## Layout

```
plugin/
  src/DofCore.h       the blur maths (shared by CPU and GPU, per-pixel functions)
  src/DofAE.cpp       the effect (parameters, CPU pipeline, AE glue, focus marker)
  src/DofGPU.cu       GPU pipeline (CUDA)
  src/DofAuto.cpp     automatic depth generation and wiring (adapted from Relight Anime's RelightAuto)
  src/DofHelper.cpp   helper AEGP (drives the jobs from an idle hook)
  build.ps1           build script (VS2022 Build Tools + CMake + Ninja; GPU path if CUDA is present)
tools/harness/
  dof_cli.cpp         runs the plugin code without AE
  run_cli.py          blurs one frame of a video and shows source / result / blur map
```

## Requirements

- Windows x64, After Effects 2024 or later (tested with AE 2026)
- [After Effects SDK](https://developer.adobe.com/after-effects/)
- Visual Studio 2022 Build Tools (C++), CMake, Ninja
- CUDA Toolkit 12.8 or later for the GPU path (otherwise CPU only)
- For automatic depth: Relight Anime's depth setup (`relight_depth.exe`, models, `%APPDATA%\SRLM\RelightAnime.ini`). You can also point Depth Layer at your own depth sequence.

## Build and install

```powershell
.\plugin\build.ps1 -SdkRoot C:\dev\AE_SDK
# from an elevated PowerShell, with AE closed
.\plugin\build.ps1 -SdkRoot C:\dev\AE_SDK -Install
```

This copies `DoFAnime.aex` and `DoFAnimeHelper.aex` to `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\SRLM`.
To override settings for DoF Anime only, put key=value lines in `%APPDATA%\SRLM\DoFAnime.ini`.

## Usage

1. Apply Effect > SRLM > DoF Anime to a footage layer.
2. Depth is generated and wired as a hidden "DoF depth" layer (instantly if cached).
3. Drag the "Focus" frame in the Composition panel to what should be sharp.
4. Set the amount with Blur Size; adjust near and far separately with Near Blur / Far Blur. Near Blur defaults to 0 (only what is behind the focus blurs); raise it for shots that need foreground bokeh.
5. By default no character matte is used: blur comes from depth alone, with edges joined smoothly by Edge Bias / Edge Soften. To keep a whole character evenly sharp, turn on Auto Subject or press Generate Subject; a character matte is then generated and wired to Subject Layer (you can also pick a cut-out such as a Roto Brush layer yourself). Estimated depth varies within a character (hair tips, legs, frills), so depth inside the matte is pulled towards the character's representative depth (Subject Flatten). The matte is shrunk by Subject Inset (12 px by default) so that only the inside is evened out and the edge comes from depth. Put the Focus Point on the character to bring the whole character into focus; move it to the background and the character blurs evenly. Keep Subject Sharp (0% by default) forces the matte to stay sharp wherever the focus is. For spots the automatic matte misses, draw a mask on a solid and pick it as Sharp Layer (always sharp) or Blur Layer (blurred like the farthest background). Or press "Text..." under Sharp by Text / Blur by Text and type object names (e.g. flower, grass): SAM 3.1 finds and tracks them through the clip and wires the matte for you (optionally only the one at the Focus Point). The automatic matte finds and tracks characters with SAM 3.1, uses the depth map to fix large misses and spills, and re-solves the edge from the image colours (closed-form matting). Choose the edge method with `subject_matte` in %APPDATA%\SRLM\DoFAnime.ini: `depthcf` (default), `cf` (no depth), `sam2matting` (cleanest but **non-commercial**) or `none`.

Output = Blur Map shows the blur size (near = blue, far = orange, in focus = black). Focus Check tints the in-focus area green.

## How it works

See [docs/how-it-works.en.md](docs/how-it-works.en.md) for edge handling, character mattes, lens character, a comparison with AE's Camera Lens Blur, and building from scratch. In short:


1. Convert the input to linear light and build 1/2, 1/4 and 1/8 downsampled copies.
2. Smooth depth with the median of neighbouring frames; take the median depth around the Focus Point as the focus depth.
3. Upsample depth to output size, snapping depth edges to the line art where depth changes (joint bilateral upsampling).
4. Compute a signed circle of confusion from the depth difference. Estimated depth behaves like disparity, so CoC is proportional to the difference, as in the thin-lens model.
5. For each 16×16 tile, find the largest blur radius that can reach it.
6. For each pixel, test points on a golden-angle spiral; add a point's colour if its circle reaches this pixel (scatter-as-gather).
   - Far/focus layer: a farther point's spread is limited to this pixel's own blur, so background never bleeds onto the in-focus subject.
   - Near layer: near points spread over what is behind; their coverage gives an alpha used to composite on top.
   - When points are sparse, colours are read bilinearly from the downsampled copies to avoid patterns inside the blur.

## Known limitations

- Where the estimated depth edge does not match the drawn outline, the blur edge is off too. Too much Edge Expand leaves a sharp sliver of background just outside the outline.
- Heavily blurred foreground cannot reveal background that the image never contained; it is filled from nearby colours.
- Depth varies within a character, so a narrow Focus Range can blur part of a face (a character matte with Subject Flatten evens this out).
- Thin hair tips and grass blades can be missing from depth and stay sharp or blur halfway. Setting `depth_input_size=1036` with the fp16 models captures them (about 20 GB of VRAM; see the Relight Anime README).

## Licence

[MIT](LICENSE). Depth model: Video Depth Anything (Small is Apache-2.0; **Large is CC-BY-NC-4.0, non-commercial**, used by Remote Quality). Character mattes: SAM 3.1 (Meta SAM License) and pymatting (MIT). Only `subject_matte=sam2matting` uses SAM2Matting (**CC BY-NC 4.0, non-commercial**; from your ComfyUI-SAM2Matting install, not included here). The After Effects SDK is not included.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cli.py - dof_cli.exe (プラグインのぼかしのコードそのもの) で 1 フレームをぼかし、比較画像を作る

  python tools/harness/run_cli.py --video a.mp4 --frame 10 --depth-dir cache/x/depth --out out.png -- fx=450 fy=200 blur=40
  python tools/harness/run_cli.py --src image.png --depth depth.png --out out.png -- ...

--depth-dir を渡すと、そのフレームの深度 (depth.%05d.png、1 始まり) と前後 1 フレームを使う (深度を時間方向にならす)。
--panel を付けると 元 / ぼかし / ぼけの大きさ (Blur Map) を横に並べる。付けなければぼかしだけ。
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CLI = HERE / "build" / "dof_cli.exe"


def load_png16(path):
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a.ndim == 3:
        a = a[:, :, ::-1]
    scale = 65535.0 if a.dtype == np.uint16 else 255.0
    a = a.astype(np.float32) / scale
    if a.ndim == 3:
        a = a[:, :, :3].mean(axis=2)
    return a


def load_frame(video, frame):
    tmp = Path(tempfile.gettempdir()) / "dof_cli_frame.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf",
                    f"select=eq(n\\,{frame})", "-vframes", "1", str(tmp)], check=True)
    return np.asarray(Image.open(tmp).convert("RGBA")).astype(np.float32) / 255.0


def run(src, depth, extra, neighbors=None):
    """src: RGBA 0..1 (H, W, 4)。depth: (dh, dw) 近い = 1。neighbors: 前後のフレームの深度のリスト"""
    H, W = src.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src.astype(np.float32).tofile(td / "src.f32")
        args = []
        if depth is not None:
            depth.astype(np.float32).tofile(td / "depth.f32")
            args += [f"dw={depth.shape[1]}", f"dh={depth.shape[0]}"]
        for k, nd in enumerate(neighbors or []):
            p = td / f"nb{k}.f32"
            nd.astype(np.float32).tofile(p)
            args.append(f"nb={p}")
        cmd = [str(CLI), str(td / "src.f32"), str(td / "depth.f32") if depth is not None else "-",
               str(td / "out.f32"), str(W), str(H)] + args + list(extra)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr + r.stdout)
        out = np.fromfile(td / "out.f32", dtype=np.float32).reshape(H, W, 4)
        return out, r.stdout.strip()


def to_img(a):
    return Image.fromarray((np.clip(a[:, :, :3], 0, 1) * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--src")
    ap.add_argument("--depth")
    ap.add_argument("--depth-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", action="store_true")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()

    src = load_frame(a.video, a.frame) if a.video else np.asarray(Image.open(a.src).convert("RGBA")).astype(np.float32) / 255.0
    nbs = []
    if a.depth_dir:
        d = Path(a.depth_dir)
        depth = load_png16(d / f"depth.{a.frame + 1:05d}.png")
        for off in (-1, 1):
            p = d / f"depth.{a.frame + 1 + off:05d}.png"
            if p.exists():
                nbs.append(load_png16(p))
    else:
        depth = load_png16(a.depth) if a.depth else None

    out, info = run(src, depth, a.extra, nbs)
    print(info)
    if a.panel:
        bm, _ = run(src, depth, [e for e in a.extra if not e.startswith("mode=")] + ["mode=3"], nbs)
        tiles = [to_img(src), to_img(out), to_img(bm)]
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * len(tiles), h))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * w, 0))
        sheet.save(a.out)
    else:
        to_img(out).save(a.out)


if __name__ == "__main__":
    main()

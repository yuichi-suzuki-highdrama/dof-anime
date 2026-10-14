#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject_mask.py - ピントを合わせるキャラのマスクを、動画全体について自動で作る (DoF Anime の Subject Layer 用)

  python subject_mask.py --video a.mp4 --out-dir cache/<shot>/subject_xxxx --point 450,190 [--frame 0] [--text "anime character"]

手順:
  1. --frame のコマで、--text の文字でキャラを全員検出する (SAM 3.1)。--point (Focus Point、素材の画素) を含むキャラを 1 人目にし、
     残りは点に近い順に 3 人まで。文字で見つからなければ、点だけで 1 人分のマスクを作る
  2. そのマスクから、キャラごとに別々に動画の後ろへ追いかける (SAM 3.1 の動画の追跡)。--frame が 0 でなければ、前へも逆順に追いかける
  3. 1 コマずつ subject.%05d.png に書く。RGB の PNG で、赤 = 1 人目、緑 = 2 人目、青 = 3 人目のマスク (白 = キャラ)。
     DoF Anime は Subject Channel = Multi (RGB) で読み、キャラごとに自分の代表の深度へ寄せる。
  素材は動画でも静止画 (png / jpg / webp など) でもよい。

SAM 3.1 は ComfyUI に入っている実装と重み (models/checkpoints/sam3.1_multiplex_fp16.safetensors) を、
ComfyUI を立ち上げずに部品として使う。ComfyUI の Python (python_embeded) で動かすこと。
SAM 3 は Meta の SAM License (https://github.com/facebookresearch/sam3/blob/main/LICENSE)。

進み具合は --status (status.json) に書く (DoF Anime の進み具合の窓が読む)。終わったら manifest.json を書く。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SUBJECTS = 3   # RGB の 3 チャンネルに 1 人ずつ


def write_json(path, obj, must=False):
    """書いてから差し替える。DoF Anime の進み具合の窓が同じときに読んでいると差し替えに失敗するので、
    少し待って何度か試す。進み具合 (must=False) なら、それでもだめなら今回は書かない"""
    if not path:
        return
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    if must:
        raise PermissionError(f"書き込めません: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--point", required=True, help="x,y (素材の画素)")
    ap.add_argument("--frame", type=int, default=0, help="点を見るコマ (0 始まり)")
    ap.add_argument("--text", default="anime character")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--comfyui", default=r"C:\ComfyUI_windows_portable\ComfyUI")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--status", default="")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = a.status or str(out / "status.json")
    write_json(status, {"state": "running", "pid": os.getpid(), "stage": "load", "progress": 0.0})

    sys.path.insert(0, a.comfyui)
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]   # ComfyUI の引数の解析に、こちらの引数を渡さない
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    import comfy.sd
    import comfy.utils
    import comfy.model_management
    from comfy.ldm.sam3.tracker import unpack_masks
    sys.argv = saved_argv

    t0 = time.time()
    try:
        # --- 動画 (または静止画) を読む (素材の大きさのまま、0..1 の RGB) ---
        fps = 24.0
        frames = []
        if Path(a.video).suffix.lower() in IMAGE_EXTS:
            # 静止画。Windows の日本語のパスでも読めるよう、バイト列から読む
            buf = np.fromfile(a.video, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            cap = cv2.VideoCapture(a.video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
        if not frames:
            raise RuntimeError("素材からコマを読めませんでした")
        N = len(frames)
        H, W = frames[0].shape[:2]
        k = max(0, min(a.frame, N - 1))
        px, py = [float(v) for v in a.point.split(",")]
        px = min(max(px, 0), W - 1)
        py = min(max(py, 0), H - 1)
        images = torch.from_numpy(np.stack(frames)).float().div_(255.0)   # [N, H, W, 3]
        del frames

        # --- SAM 3.1 を読む ---
        ckpt = a.ckpt or os.path.join(a.comfyui, "models", "checkpoints", "sam3.1_multiplex_fp16.safetensors")
        model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(ckpt, output_vae=False, output_clip=True)
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam = model.model.diffusion_model
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "detect", "progress": 0.02})

        # --- 1. 点のコマでキャラを全員検出する。点を含むキャラを 1 人目にし、残りは確からしい順に MAX_SUBJECTS 人まで ---
        frame = comfy.utils.common_upscale(images[k:k + 1].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        frame = frame.to(device=device, dtype=dtype)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(a.text))
        emb = cond[0][0].to(device=device, dtype=dtype)
        tmask = cond[0][1].get("attention_mask")
        tmask = tmask.to(device) if tmask is not None else torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
        found = []   # (点を含むか, 確からしさ, 点までの距離, マスク)
        with torch.no_grad():
            r = sam(frame, text_embeddings=emb, text_mask=tmask, boxes=None, threshold=a.threshold, orig_size=(H, W))
            probs = r["scores"][0].sigmoid()
            for i in (probs > a.threshold).nonzero().flatten().tolist():
                m = F.interpolate(r["masks"][0][i][None, None].float(), size=(H, W), mode="bilinear")[0, 0] > 0
                ys, xs = m.nonzero(as_tuple=True)
                if len(xs) < 64:
                    continue
                inside = bool(m[int(py), int(px)])
                d = 0.0 if inside else float(((xs.float() - px) ** 2 + (ys.float() - py) ** 2).min().sqrt())
                found.append((inside, float(probs[i]), d, m))
            # 同じキャラを 2 回拾ったもの (大きく重なる) は、確からしい方だけ残す
            found.sort(key=lambda f: -f[1])
            uniq = []
            for f in found:
                if all(float((f[3] & g[3]).sum()) / float(min(f[3].sum(), g[3].sum())) < 0.6 for g in uniq):
                    uniq.append(f)
            # 点を含むもの → 点に近いもの の順。1 人目が「ピントを合わせたキャラ」
            uniq.sort(key=lambda f: (not f[0], f[2], -f[1]))
            masks = [f[3] for f in uniq[:MAX_SUBJECTS]]
            how = f"text '{a.text}' ({len(found)} found, {len(masks)} used" + (", first contains point" if uniq and uniq[0][0] else "") + ")"
            if not masks:
                # 文字で見つからなければ、点だけで 1 人
                pts = {"point_coords": torch.tensor([[[px / W * 1008, py / H * 1008]]], dtype=dtype, device=device),
                       "point_labels": torch.tensor([[1]], dtype=torch.int32, device=device)}
                ml = sam.forward_segment(frame, point_inputs=pts)
                for _ in range(2):
                    ml = sam.forward_segment(frame, mask_inputs=ml)
                masks = [F.interpolate(ml.float(), size=(H, W), mode="bilinear")[0, 0] > 0]
                how = "point only"
        S = len(masks)
        init = torch.stack(masks).float()[:, None]   # [S, 1, H, W]
        print(f"start masks: frame {k}, {how}, areas " + ", ".join(f"{float(m.float().mean()):.3f}" for m in masks), flush=True)

        # --- 2. 追いかける (後ろへ、必要なら前へも)。キャラごとに別々のマスク ---
        result = np.zeros((N, H, W, 3), np.uint8)   # R, G, B = 1 人目, 2 人目, 3 人目
        for s in range(S):
            result[k, :, :, s] = (masks[s].float() * 255 + 0.5).byte().cpu().numpy()
        done = [0]

        def track(order):
            if len(order) < 2:
                return
            imgs = images[order].movedim(-1, 1)   # [n, 3, H, W]
            n = len(order)

            class Bar:
                def update(self, v):
                    done[0] += v
                    write_json(status, {"state": "running", "pid": os.getpid(), "stage": "track",
                                        "progress": 0.05 + 0.9 * min(1.0, done[0] / max(1, N + k))})

                def update_absolute(self, v, total=None, preview=None):
                    pass
            with torch.no_grad():
                res = sam.forward_video(images=imgs, initial_masks=init.to(device=device, dtype=dtype),
                                        pbar=Bar(), text_prompts=None, new_det_thresh=0.5, max_objects=S,
                                        detect_interval=1, target_device=device, target_dtype=dtype)
            packed = res["packed_masks"]
            if packed is None:
                return
            for j in range(n):
                for s in range(min(S, packed.shape[1])):
                    m = unpack_masks(packed[j, s]).float()[None, None]
                    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                    result[order[j], :, :, s] = (m.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()

        track(list(range(k, N)))
        if k > 0:
            track(list(range(k, -1, -1)))

        # --- 3. 書き出す (RGB の PNG。OpenCV は BGR の順で書くので並べ替える) ---
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "write", "progress": 0.97})
        for i in range(N):
            cv2.imwrite(str(out / f"subject.{i + 1:05d}.png"), result[i][:, :, ::-1])
        write_json(out / "manifest.json", {
            "generator": "subject_mask.py", "model": "sam3.1_multiplex", "text": a.text,
            "point": [px, py], "frame": k, "start": how, "subjects": S, "channels": "R,G,B = subject 1,2,3",
            "frames": N, "size": [W, H], "fps": fps,
            "file": "subject.%05d.png", "seconds": round(time.time() - t0, 1)}, must=True)
        write_json(status, {"state": "done", "pid": os.getpid(), "progress": 1.0}, must=True)
        print(f"done: {N} frames, {S} subjects in {time.time() - t0:.1f}s", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        write_json(status, {"state": "failed", "pid": os.getpid(), "message": f"{type(e).__name__}: {e}"})
        raise


if __name__ == "__main__":
    sys.exit(main())

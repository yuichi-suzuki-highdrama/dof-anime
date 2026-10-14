#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_mask.py - 文字で指定した物のマスクを、動画全体について作る (DoF Anime の Sharp Layer / Blur Layer 用)

  python prompt_mask.py --video a.mp4 --out-dir cache/<shot>/prompt_xxxx --text "flower, grass" [--frame 0] [--point 450,190]

手順:
  1. --frame のコマで、--text の文字 (カンマで区切れば複数) に当たる物を SAM 3.1 で探す。
     --point (素材の画素) を渡せば、その点を含む物 (無ければいちばん近い物) を 1 つだけ。渡さなければ見つかった物を全部 (MAX_OBJECTS まで)
  2. 物ごとに別々に動画全体を追いかける (SAM 3.1 の動画の追跡)
  3. 物を合わせて 1 枚のマスクにし、穴と切れ端を片付け、縁を元の絵の色から切り抜き直す (クローズドフォーム・マッティング)
  4. 1 コマずつ prompt.%05d.png に書く。RGBA の PNG で、色は白、アルファがマスク。
     DoF Anime の Sharp Layer / Blur Layer はアルファを読む

部品は subject_mask.py と同じ (SAM 3.1 を ComfyUI を立ち上げずに使う。ComfyUI の Python で動かすこと)。
進み具合は --status (status.json) に書き、終わったら manifest.json を書く。
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I で動かしても subject_mask を読めるように
from subject_mask import refine_mask, clean_binary, cf_matte, load_frames, track_objects, write_json  # noqa: E402

MAX_OBJECTS = 16  # 点を渡さないときに追いかける物の数の上限


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--text", required=True, help="探す物の名前 (英語。カンマで区切れば複数)")
    ap.add_argument("--frame", type=int, default=0, help="探すコマ (0 始まり)")
    ap.add_argument("--point", default="", help="x,y (素材の画素)。渡せばその所の物を 1 つだけ")
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
        frames, fps = load_frames(a.video, np, cv2)
        N = len(frames)
        H, W = frames[0].shape[:2]
        k = max(0, min(a.frame, N - 1))
        point = None
        if a.point:
            px, py = [float(v) for v in a.point.split(",")]
            point = (min(max(px, 0), W - 1), min(max(py, 0), H - 1))
        images = torch.from_numpy(np.stack(frames)).float().div_(255.0)   # [N, H, W, 3]
        del frames

        ckpt = a.ckpt or os.path.join(a.comfyui, "models", "checkpoints", "sam3.1_multiplex_fp16.safetensors")
        model, clip, _, _ = comfy.sd.load_checkpoint_guess_config(ckpt, output_vae=False, output_clip=True)
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        sam = model.model.diffusion_model
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "detect", "progress": 0.02})

        # --- 1. 文字ごとに探す ---
        frame = comfy.utils.common_upscale(images[k:k + 1].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")
        frame = frame.to(device=device, dtype=dtype)
        terms = [t.strip() for t in a.text.split(",") if t.strip()]
        found = []   # (点を含むか, 点までの距離, 確からしさ, マスク, logits, 箱)
        with torch.no_grad():
            for term in terms:
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(term))
                emb = cond[0][0].to(device=device, dtype=dtype)
                tmask = cond[0][1].get("attention_mask")
                tmask = tmask.to(device) if tmask is not None else torch.ones(emb.shape[0], emb.shape[1], dtype=torch.int64, device=device)
                r = sam(frame, text_embeddings=emb, text_mask=tmask, boxes=None, threshold=a.threshold, orig_size=(H, W))
                probs = r["scores"][0].sigmoid()
                for i in (probs > a.threshold).nonzero().flatten().tolist():
                    m = F.interpolate(r["masks"][0][i][None, None].float(), size=(H, W), mode="bilinear")[0, 0] > 0
                    ys, xs = m.nonzero(as_tuple=True)
                    if len(xs) < 16:
                        continue
                    inside, d = False, 0.0
                    if point:
                        inside = bool(m[int(point[1]), int(point[0])])
                        d = 0.0 if inside else float(((xs.float() - point[0]) ** 2 + (ys.float() - point[1]) ** 2).min().sqrt())
                    found.append((inside, d, float(probs[i]), m, r["masks"][0][i], r["boxes"][0][i]))
            # 同じ物を 2 回拾ったもの (大きく重なる) は、確からしい方だけ残す
            found.sort(key=lambda f: -f[2])
            uniq = []
            for f in found:
                if all(float((f[3] & g[3]).sum()) / float(min(f[3].sum(), g[3].sum())) < 0.6 for g in uniq):
                    uniq.append(f)
            if point:
                uniq.sort(key=lambda f: (not f[0], f[1], -f[2]))
                uniq = uniq[:1]
            else:
                uniq = uniq[:MAX_OBJECTS]
            masks = [refine_mask(sam, images[k], f[4], f[5], H, W, device, dtype, F, comfy.utils) for f in uniq]
        if not masks:
            raise RuntimeError(f"「{a.text}」に当たる物が見つかりませんでした (英語の名前で試してください)")
        S = len(masks)
        how = f"text '{a.text}' ({len(found)} found, {S} used" + (", at point" if point else "") + ")"
        print(f"start masks: frame {k}, {how}", flush=True)

        # --- 2. 追いかける ---
        result = np.zeros((N, H, W, S), np.uint8)
        for s in range(S):
            result[k, :, :, s] = (masks[s].float() * 255 + 0.5).byte().cpu().numpy()
        init = torch.stack(masks).float()[:, None]   # [S, 1, H, W]
        track_objects(sam, images, init, k, result, device, dtype, F, unpack_masks,
                      lambda v: write_json(status, {"state": "running", "pid": os.getpid(), "stage": "track", "progress": 0.05 + 0.85 * v}))

        # --- 3. 合わせて 1 枚にし、仕上げて書き出す ---
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "clean", "progress": 0.92})
        for i in range(N):
            m = result[i].max(axis=2)
            b = clean_binary(m, cv2, np)
            alpha = cf_matte(b, images[i].numpy().astype(np.float64), cv2, np)
            rgba = np.full((H, W, 4), 255, np.uint8)
            rgba[:, :, 3] = alpha
            cv2.imwrite(str(out / f"prompt.{i + 1:05d}.png"), rgba)   # 色は白なので BGR の順は気にしない
        write_json(out / "manifest.json", {
            "generator": "prompt_mask.py", "model": "sam3.1_multiplex + closed-form", "text": a.text,
            "point": list(point) if point else None, "frame": k, "start": how, "objects": S,
            "frames": N, "size": [W, H], "fps": fps, "file": "prompt.%05d.png", "channels": "alpha = mask",
            "seconds": round(time.time() - t0, 1)}, must=True)
        write_json(status, {"state": "done", "pid": os.getpid(), "progress": 1.0}, must=True)
        print(f"done: {N} frames, {S} objects in {time.time() - t0:.1f}s", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        write_json(status, {"state": "failed", "pid": os.getpid(), "message": f"{type(e).__name__}: {e}"})
        raise


if __name__ == "__main__":
    sys.exit(main())

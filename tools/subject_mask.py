#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subject_mask.py - ピントを合わせるキャラのマスクを、動画全体について自動で作る (DoF Anime の Subject Layer 用)

  python subject_mask.py --video a.mp4 --out-dir cache/<shot>/subject_xxxx --point 450,190 [--frame 0] [--text "anime character"]

手順:
  1. --frame のコマで、--text の文字でキャラを全員検出する (SAM 3.1)。--point (Focus Point、素材の画素) を含むキャラを 1 人目にし、
     残りは点に近い順に 3 人まで。文字で見つからなければ、点だけで 1 人分のマスクを作る
  2. そのマスクから、キャラごとに別々に動画全体を追いかける (SAM 3.1 の動画の追跡)。--frame が 0 でなければ、前へも逆順に追いかける。
     --matte で縁の仕上げ方を選ぶ:
       depthcf (既定): マスクと深度を混ぜる。マスクの中で深度がまわりのキャラより明らかに奥の大きな塊は外し
                       (浴槽の内側など)、マスクのすぐ外で深度がキャラと同じ高さの大きな塊は足す (取りこぼし)。
                       そのあと縁の数画素を、元の絵の色から切り抜き直す (クローズドフォーム・マッティング、pymatting)。
                       深度は --depth-dir (無ければ --out-dir の隣の depth) の depth.%05d.png。無ければ cf と同じ
       cf:          深度を使わず、縁だけクローズドフォームで切り抜き直す
       sam2matting: 追跡と切り抜きを SAM2Matting (SAM3 版) で行う。いちばんきれいだが CC BY-NC 4.0 (営利目的では使えない)
       none:        元の絵をガイドにしたガイデッドフィルタで縁を線画に寄せるだけ
  3. 1 コマずつ subject.%05d.png に書く。RGB の PNG で、赤 = 1 人目、緑 = 2 人目、青 = 3 人目のマスク (白 = キャラ)。
     DoF Anime は Subject Channel = Multi (RGB) で読み、キャラごとに自分の代表の深度へ寄せる。
  素材は動画でも静止画 (png / jpg / webp など) でもよい。
  マスクは仕上げに、(a) 検出したマスクを SAM の細かい段 (1008 画素、キャラの周りを切り出して) で詰め直し、
  (b) キャラの内側の小さな穴を埋めて小さな切れ端を消し、(c) 上の --matte のやり方で縁を仕上げる。

SAM 3.1 は ComfyUI に入っている実装と重み (models/checkpoints/sam3.1_multiplex_fp16.safetensors) を、
ComfyUI を立ち上げずに部品として使う。ComfyUI の Python (python_embeded) で動かすこと。
SAM 3 は Meta の SAM License (https://github.com/facebookresearch/sam3/blob/main/LICENSE)。
SAM2Matting は ComfyUI-SAM2Matting (custom_nodes) の実装と重み (models/sam2matting/SAM2Matting-SAM3.pt) を使う。
SAM2Matting は CC BY-NC 4.0 (営利目的では使えない)。既定の depthcf / cf / none は SAM 3.1 と pymatting (MIT) だけを使う。

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
REFINE_ITERS = 2   # 検出したマスクを SAM の細かい段で詰め直す回数


def refine_mask(sam, image_hwc, coarse_logits, box, H, W, device, dtype, F, comfy_utils):
    """検出の粗いマスク (logits) を、キャラの周りを 1008 画素に切り出して SAM の細かい段で詰め直す
    (ComfyUI の SAM3 Detect の refine と同じ考え方)。戻り値: [H, W] の bool"""
    import torch
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad = 0.1
    cx1, cy1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad))
    cx2, cy2 = min(W, int(x2 + bw * pad)), min(H, int(y2 + bh * pad))
    coarse_full = F.interpolate(coarse_logits[None, None].float(), size=(H, W), mode="bilinear")[0, 0] > 0
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return coarse_full
    crop = image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_1008 = comfy_utils.common_upscale(crop[None].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled").to(device=device, dtype=dtype)
    mh, mw = coarse_logits.shape[-2:]
    mx1, my1 = int(cx1 / W * mw), int(cy1 / H * mh)
    mx2, my2 = max(mx1 + 1, int(cx2 / W * mw)), max(my1 + 1, int(cy2 / H * mh))
    logit = coarse_logits[my1:my2, mx1:mx2][None, None].to(device=device, dtype=dtype)
    for _ in range(REFINE_ITERS):
        logit = sam.forward_segment(crop_1008, mask_inputs=F.interpolate(logit.float(), size=(1008, 1008), mode="bilinear").to(dtype))
    refined = F.interpolate(logit.float(), size=(cy2 - cy1, cx2 - cx1), mode="bilinear")[0, 0] > 0
    full = torch.zeros(H, W, dtype=torch.bool, device=refined.device)
    full[cy1:cy2, cx1:cx2] = refined
    return full if bool(full.any()) else coarse_full


def clean_mask(m_u8, guide_u8, cv2, np):
    """1 人分のマスク (0..255) を仕上げる: 内側の小さな穴を埋め、小さな切れ端を消し、元の絵の色の境目に縁を寄せる"""
    H, W = m_u8.shape
    b = m_u8 >= 128
    area = int(b.sum())
    if area < 64:
        return m_u8
    # 小さな切れ端 (本体の 0.5% 未満) を消す
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < area * 0.005:
            b[lab == c] = False
    # 内側の小さな穴 (外とつながっていない、本体の 2% 未満) を埋める。腕と体の間の大きなすき間は残す
    inv = (~b).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for c in range(1, n):
        x, y, w, h, a = stats[c]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        if not touches and a < area * 0.02:
            b[lab == c] = True
    # 縁を元の絵の色の境目に寄せる (ガイデッドフィルタ)。少し締めて、ふわっと広がりすぎないようにする
    r = max(2, int(round(min(H, W) / 200)))
    gf = cv2.ximgproc.guidedFilter(guide_u8, b.astype(np.float32), r, (0.03 * 255) ** 2)
    t = np.clip((gf - 0.2) / 0.6, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return (t * 255.0 + 0.5).astype(np.uint8)


def clean_matte(m_u8, cv2, np, band=2.5):
    """SAM2Matting の 1 人分の切り抜き (0..255) を仕上げる: 一度 2 値にして小さな切れ端を消し、内側の小さな穴を埋め、
    縁から band 画素の中だけ切り抜きの半透明の値を戻す (毛先や縁のなめらかさは残し、迷った所の広いシミは消える)"""
    H, W = m_u8.shape
    b = (m_u8 >= 128).astype(np.uint8)
    area = int(b.sum())
    if area < 64:
        return m_u8
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < area * 0.005:
            b[lab == c] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - b, connectivity=4)
    for c in range(1, n):
        x, y, w, h, a = stats[c]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H) and a < area * 0.002:
            b[lab == c] = 1
    d = np.where(b > 0, cv2.distanceTransform(b, cv2.DIST_L2, 3), cv2.distanceTransform(1 - b, cv2.DIST_L2, 3))
    near = np.clip(1.0 - (d - band) / 1.5, 0.0, 1.0)
    out = near * (m_u8.astype(np.float32) / 255.0) + (1.0 - near) * b.astype(np.float32)
    return (out * 255.0 + 0.5).astype(np.uint8)


def clean_binary(m_u8, cv2, np):
    """マスク (0..255) を 2 値にし、小さな切れ端 (本体の 0.5% 未満) を消し、内側の小さな穴 (2% 未満) を埋める"""
    H, W = m_u8.shape
    b = (m_u8 >= 128)
    area = int(b.sum())
    if area < 64:
        return b.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < area * 0.005:
            b[lab == c] = False
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~b).astype(np.uint8), connectivity=4)
    for c in range(1, n):
        x, y, w, h, ar = stats[c]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H) and ar < area * 0.02:
            b[lab == c] = True
    return b.astype(np.uint8)


DEPTH_GROW = 40         # マスクの外へ足す範囲 (画素)
DEPTH_T_ADD = 0.04      # まわりのキャラの深度とこれ以内なら、キャラとつながっているとみなして足す (手前の物は足さない)
DEPTH_T_RM = 0.10       # まわりのキャラの深度よりこれ以上奥なら、キャラではないとみなして外す
DEPTH_MIN_ADD = 0.005   # 足す塊の最小の大きさ (キャラの面積に対する割合)
DEPTH_MIN_RM = 0.002    # 外す塊の最小の大きさ
DEPTH_OPEN_R = 6        # 深度のにじみ (細い帯) を無視する半径 (画素)


def depth_fix(b, depth, cv2, np):
    """2 値のマスク b を深度で直す。depth: [H, W] 0..1 (近い = 1)。深度の地図は粗いので、大きな塊だけ足し引きする"""
    area = int(b.sum())
    if area < 64:
        return b
    bf = b.astype(np.float32)
    # その場所での「キャラの深度」: マスクの中の深度を、まわりへ広げてならしたもの
    num = cv2.GaussianBlur(depth * bf, (0, 0), 25)
    den = cv2.GaussianBlur(bf, (0, 0), 25)
    ref = num / np.maximum(den, 1e-4)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DEPTH_OPEN_R + 1, 2 * DEPTH_OPEN_R + 1))
    dist = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
    add = ((b == 0) & (dist <= DEPTH_GROW) & (np.abs(depth - ref) < DEPTH_T_ADD) & (den > 0.02)).astype(np.uint8)
    rm = ((b > 0) & (depth < ref - DEPTH_T_RM)).astype(np.uint8)
    # 足すのはキャラと同じ深度の物だけなので、キャラのすぐ前の花などが入っても、ピントとぼけはキャラと同じになるだけで害はない。
    # 深度の違う物 (手前の浴槽の縁など) は足さない
    nb = b.copy()
    for m_, val, min_a in ((cv2.morphologyEx(add, cv2.MORPH_OPEN, ker), 1, DEPTH_MIN_ADD),
                           (cv2.morphologyEx(rm, cv2.MORPH_OPEN, ker), 0, DEPTH_MIN_RM)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m_, connectivity=8)
        for c in range(1, n):
            if st[c, cv2.CC_STAT_AREA] >= area * min_a:
                nb[lab == c] = val
    return nb


CF_BAND = 5   # クローズドフォームで計算し直す縁の幅 (画素、内と外それぞれ)


def cf_matte(b, img, cv2, np):
    """2 値のマスク b の縁から CF_BAND 画素の内と外を「未定」にして、元の絵の色から切り抜きを解く
    (クローズドフォーム・マッティング、pymatting = MIT)。img: [H, W, 3] 0..1。戻り値: 0..255"""
    from pymatting import estimate_alpha_cf
    H, W = b.shape
    if int(b.sum()) < 64:
        return (b * 255).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CF_BAND + 1, 2 * CF_BAND + 1))
    tri = np.full((H, W), 0.5)
    tri[cv2.erode(b, ker) > 0] = 1.0
    tri[cv2.dilate(b, ker) == 0] = 0.0
    ys, xs = np.nonzero(b)
    y0, y1 = max(0, ys.min() - 2 * CF_BAND), min(H, ys.max() + 2 * CF_BAND + 1)
    x0, x1 = max(0, xs.min() - 2 * CF_BAND), min(W, xs.max() + 2 * CF_BAND + 1)
    alpha = np.zeros((H, W))
    try:
        alpha[y0:y1, x0:x1] = estimate_alpha_cf(img[y0:y1, x0:x1], tri[y0:y1, x0:x1])
    except Exception as e:  # noqa: BLE001  解けなければ 2 値のまま
        print(f"cf: {type(e).__name__}: {e}", flush=True)
        alpha = b.astype(np.float64)
    # ごく薄い値 (0.1 未満) は解いたときのノイズ (縁の外にトゲのように散らばる) なので消し、そのぶんコントラストを整える
    alpha = np.clip((alpha - 0.1) / 0.8, 0.0, 1.0)
    return (alpha * 255.0 + 0.5).astype(np.uint8)


def load_depth(depth_dir, i, W, H, cv2, np):
    """深度の 1 コマ (depth.%05d.png、1 始まり) を [H, W] 0..1 (近い = 1) で読む。無ければ None"""
    if not depth_dir:
        return None
    path = Path(depth_dir) / f"depth.{i:05d}.png"
    if not path.is_file():
        return None
    d = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[:, :, 0]
    scale = 65535.0 if d.dtype == np.uint16 else 255.0
    return cv2.resize(d.astype(np.float32) / scale, (W, H), interpolation=cv2.INTER_LINEAR)


def matte_subjects(comfy_root, images, masks, k, result, report):
    """SAM2Matting (SAM3 版) で、キャラごとに k コマ目のマスクから動画全体を切り抜く。result[i, :, :, s] に 0..255 で書く。
    report(s, done, total): 進み具合。使えなければ False を返す"""
    import torch
    node_dir = Path(comfy_root) / "custom_nodes" / "ComfyUI-SAM2Matting"
    ckpt = Path(comfy_root) / "models" / "sam2matting" / "SAM2Matting-SAM3.pt"
    if not (node_dir / "video_model.py").is_file() or not ckpt.is_file() or not torch.cuda.is_available():
        print(f"matte: SAM2Matting not available ({node_dir}, {ckpt})", flush=True)
        return False
    sys.path.insert(0, str(node_dir))
    import video_model as vm
    model = vm.SAM2MattingVideoModel("sam3", ckpt, "cuda")
    N, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    # コマは使うときに 1 枚ずつ縮める (動画全体を GPU に載せない)
    frames = vm._prepare_frames(images, int(model.predictor.image_size), model.kind, model.device, True)
    for s, m in enumerate(masks):
        def keep(i, alpha, s=s):
            result[i, :, :, s] = (alpha.float().clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
        model.matte_frame_sequence(frames, H, W, m.float().cpu(), mask_frame=k, alpha_callback=keep,
                                   progress_callback=lambda done, total, s=s: report(s, done, total))
    del model
    torch.cuda.empty_cache()
    return True


def load_frames(path, np, cv2):
    """動画 (または静止画) を読む。戻り値: (コマのリスト [H, W, 3] uint8 RGB, fps)。
    Windows の日本語のパスでも読めるよう、静止画はバイト列から読む"""
    fps = 24.0
    frames = []
    if Path(path).suffix.lower() in IMAGE_EXTS:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames:
        raise RuntimeError("素材からコマを読めませんでした")
    return frames, fps


def track_objects(sam, images, init, k, result, device, dtype, F, unpack_masks, report):
    """k コマ目のマスク init ([S, 1, H, W]) から、物ごとに別々に動画全体を追いかける (SAM 3.1 の動画の追跡。
    後ろへ、k > 0 なら前へも逆順に)。result[i, :, :, s] に 0..255 で書く。report(0..1): 進み具合"""
    import torch
    N, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    S = int(init.shape[0])
    done = [0]

    def track(order):
        if len(order) < 2:
            return
        imgs = images[order].movedim(-1, 1)   # [n, 3, H, W]

        class Bar:
            def update(self, v):
                done[0] += v
                report(min(1.0, done[0] / max(1, N + k)))

            def update_absolute(self, v, total=None, preview=None):
                pass
        with torch.no_grad():
            res = sam.forward_video(images=imgs, initial_masks=init.to(device=device, dtype=dtype),
                                    pbar=Bar(), text_prompts=None, new_det_thresh=0.5, max_objects=S,
                                    detect_interval=1, target_device=device, target_dtype=dtype)
        packed = res["packed_masks"]
        if packed is None:
            return
        for j in range(len(order)):
            for s_ in range(min(S, packed.shape[1])):
                m = unpack_masks(packed[j, s_]).float()[None, None]
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                result[order[j], :, :, s_] = (m.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()

    track(list(range(k, N)))
    if k > 0:
        track(list(range(k, -1, -1)))


def refine_tracked(sam, images, result, k, device, dtype, F, comfy_utils, report):
    """追いかけた各コマのマスクを、SAM の細かい段 (1008 画素、物の周りを切り出して) で詰め直す。
    追跡のマスクは粗い解像度から引き伸ばしたもので、縁が元の絵の輪郭線から 10 画素以上ずれて波打つことがある
    (そのままだと、縁の切り抜き直し (CF_BAND 画素の幅) が本当の輪郭を見つけられずガタガタになる)。
    k コマ目は検出のときに詰め直してあるので飛ばす。result[i, :, :, s] (0..255) を書き換える"""
    import numpy as np
    import torch
    N, H, W, S = result.shape
    for i in range(N):
        if i != k:
            for s_ in range(S):
                m = result[i, :, :, s_]
                ys, xs = np.nonzero(m >= 128)
                if len(xs) < 64:
                    continue
                box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                logits = torch.from_numpy((m.astype(np.float32) / 255.0 * 2.0 - 1.0) * 8.0)
                with torch.no_grad():
                    r = refine_mask(sam, images[i], logits, box, H, W, device, dtype, F, comfy_utils)
                result[i, :, :, s_] = (r.float() * 255 + 0.5).byte().cpu().numpy()
        report((i + 1) / N)


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
    ap.add_argument("--matte", default="depthcf", choices=["depthcf", "cf", "sam2matting", "none"],
                    help="depthcf: マスクと深度を混ぜて直し、縁をクローズドフォームで切り抜く (既定)。cf: 縁だけクローズドフォーム。"
                         "sam2matting: SAM2Matting で切り抜く (CC BY-NC 4.0、営利不可)。none: ガイデッドフィルタ")
    ap.add_argument("--depth-dir", default="", help="depthcf の深度 (depth.%%05d.png)。空なら --out-dir の隣の depth")
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
        frames, fps = load_frames(a.video, np, cv2)
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
                found.append((inside, float(probs[i]), d, m, r["masks"][0][i], r["boxes"][0][i]))
            # 同じキャラを 2 回拾ったもの (大きく重なる) は、確からしい方だけ残す
            found.sort(key=lambda f: -f[1])
            uniq = []
            for f in found:
                if all(float((f[3] & g[3]).sum()) / float(min(f[3].sum(), g[3].sum())) < 0.6 for g in uniq):
                    uniq.append(f)
            # 点を含むもの → 点に近いもの の順。1 人目が「ピントを合わせたキャラ」
            uniq.sort(key=lambda f: (not f[0], f[2], -f[1]))
            # 選んだキャラのマスクを SAM の細かい段で詰め直す (検出のマスクは粗い)
            masks = [refine_mask(sam, images[k], f[4], f[5], H, W, device, dtype, F, comfy.utils) for f in uniq[:MAX_SUBJECTS]]
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
        matted = False
        if a.matte == "sam2matting":
            # SAM 3.1 を GPU から下ろしてから (メモリが足りなくならないよう)、SAM2Matting で切り抜く
            masks = [m.cpu() for m in masks]
            del sam, model, clip, frame, r
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            write_json(status, {"state": "running", "pid": os.getpid(), "stage": "matte", "progress": 0.05})

            def report(s, d, total):
                write_json(status, {"state": "running", "pid": os.getpid(), "stage": "matte",
                                    "progress": 0.05 + 0.9 * min(1.0, (s + d / max(1, total)) / S)})
            matted = matte_subjects(a.comfyui, images, masks, k, result, report)
            if matted:
                how += ", matte sam2matting"
            else:
                raise RuntimeError("SAM2Matting を使えません (ComfyUI-SAM2Matting と models/sam2matting/SAM2Matting-SAM3.pt を確かめるか、"
                                   "設定の subject_matte=none で SAM 3.1 だけにしてください)")

        if not matted:
            track_objects(sam, images, init, k, result, device, dtype, F, unpack_masks,
                          lambda v: write_json(status, {"state": "running", "pid": os.getpid(), "stage": "track", "progress": 0.05 + 0.6 * v}))
            refine_tracked(sam, images, result, k, device, dtype, F, comfy.utils,
                           lambda v: write_json(status, {"state": "running", "pid": os.getpid(), "stage": "refine", "progress": 0.65 + 0.3 * v}))

        # --- 3. 仕上げ: 穴を埋め、切れ端を消し、縁を整える (コマごと、キャラごと) ---
        depth_dir = a.depth_dir or str(out.parent / "depth")
        depth_used = [0]
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "clean", "progress": 0.95})
        for i in range(N):
            if matted:
                for s_ in range(S):
                    result[i, :, :, s_] = clean_matte(result[i, :, :, s_], cv2, np)
                # キャラどうしが重なった所は、合わせて 1 を超えないようにする
                tot = result[i].astype(np.float32).sum(axis=2, keepdims=True)
                if float(tot.max()) > 255.0:
                    result[i] = np.where(tot > 255.0, result[i] * (255.0 / np.maximum(tot, 1.0)) + 0.5, result[i]).astype(np.uint8)
            elif a.matte in ("depthcf", "cf"):
                img = images[i].numpy().astype(np.float64)
                depth = load_depth(depth_dir, i + 1, W, H, cv2, np) if a.matte == "depthcf" else None
                if depth is not None:
                    depth_used[0] += 1
                for s_ in range(S):
                    b = clean_binary(result[i, :, :, s_], cv2, np)
                    if depth is not None:
                        b = depth_fix(b, depth, cv2, np)
                    result[i, :, :, s_] = cf_matte(b, img, cv2, np)
                tot = result[i].astype(np.float32).sum(axis=2, keepdims=True)
                if float(tot.max()) > 255.0:
                    result[i] = np.where(tot > 255.0, result[i] * (255.0 / np.maximum(tot, 1.0)) + 0.5, result[i]).astype(np.uint8)
            else:
                guide = (images[i].numpy() * 255.0 + 0.5).astype(np.uint8)
                for s_ in range(S):
                    result[i, :, :, s_] = clean_mask(result[i, :, :, s_], guide, cv2, np)

        # --- 4. 書き出す (RGB の PNG。OpenCV は BGR の順で書くので並べ替える) ---
        write_json(status, {"state": "running", "pid": os.getpid(), "stage": "write", "progress": 0.97})
        for i in range(N):
            cv2.imwrite(str(out / f"subject.{i + 1:05d}.png"), result[i][:, :, ::-1])
        write_json(out / "manifest.json", {
            "generator": "subject_mask.py", "model": "sam3.1_multiplex" + (" + sam2matting_sam3" if matted else ""), "matte": a.matte,
            "depth_frames_used": depth_used[0], "text": a.text,
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

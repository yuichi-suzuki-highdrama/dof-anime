#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_explainer.py - DoF Anime の解説動画を組み立てる (ffmpeg)

  python build_explainer.py --work <作業フォルダ> --fonts <書体のフォルダ> --bgm <曲> --src <元の動画> --final <完成版の動画>
                            --depth-lo <深度 518 の連番のフォルダ> --depth-hi <深度 1036 の連番のフォルダ> --subject <人物のマスクの連番のフォルダ>

  → <作業フォルダ>/out/dof_explainer.mp4 (1920x1080, 30fps)

素材 (<作業フォルダ>/renders/。AE で書き出したもの。tools/explainer の AE スクリプトを参照):
  EX_dof_only.mp4     DoF Anime だけ (ピント送り)
  EX_depth.mp4        Output = Depth
  EX_blurmap.mp4      Output = Blur Map
  EX_no_exclude.mp4   Exclude Layer なし (草がマスクで切れる)
  EX_nearfill0.mp4    Near Fill 0 (ぼけた髪の束の間がくっきり残る)
書体は BIZ UDPゴシック (BIZUDPGothic.ttf / BIZUDPGothicBold.ttf。Windows の .ttc から取り出したもの)。
場面ごとに segments/NN.mp4 を作り、最後につなぐ。文字は textfile で渡す (エスケープを避ける)。
"""
import argparse
import subprocess
from pathlib import Path

W, H, FPS = 1920, 1080, 30
BG = "0x101418"
BGM_GAIN_DB = -17
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
FIT = f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={BG}"

a = None
R = SEG = OUT = TXT = None
FONT_B = FONT_R = None


def ff_path(p):
    return str(p).replace("\\", "/").replace(":", "\\:")


def run(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error"] + args, check=True)


def text_file(name, s):
    p = TXT / f"{name}.txt"
    p.write_text(s, encoding="utf-8")
    return ff_path(p)


def caption(name, main, sub=""):
    """画面下の帯と文字 (main は大きく、sub は小さく)"""
    f = [f"drawbox=x=0:y=ih-190:w=iw:h=190:color=black@0.62:t=fill",
         f"drawtext=fontfile='{FONT_B}':textfile='{text_file(name + '_m', main)}':fontsize=52:fontcolor=white:x=(w-tw)/2:y=h-160"]
    if sub:
        f.append(f"drawtext=fontfile='{FONT_R}':textfile='{text_file(name + '_s', sub)}':fontsize=34:fontcolor=0xDDDDDD:x=(w-tw)/2:y=h-82")
    return ",".join(f)


def label(name, s, x=56, y=48, size=40):
    return (f"drawtext=fontfile='{FONT_B}':textfile='{text_file(name, s)}':fontsize={size}:fontcolor=white:"
            f"box=1:boxcolor=black@0.55:boxborderw=16:x={x}:y={y}")


def fades(dur):
    return f"fade=t=in:st=0:d=0.35,fade=t=out:st={dur - 0.35:.2f}:d=0.35"


def afades(dur):
    return f"afade=t=in:st=0:d=0.35,afade=t=out:st={dur - 0.35:.2f}:d=0.35"


def slide(t_in, t_out, d=0.5, x0=72):
    """drawtext の x の式: t_in から d 秒で左の外から x0 へ入り (ease-out)、t_out から d 秒で左の外へ出る (ease-in)"""
    ein = f"(1-pow(1-(t-{t_in})/{d},3))"
    eout = f"pow((t-{t_out})/{d},3)"
    off = "(-tw-80)"
    return (f"if(lt(t,{t_in}),{off},if(lt(t,{t_in + d}),{off}+({x0}-{off})*{ein},"
            f"if(lt(t,{t_out}),{x0},if(lt(t,{t_out + d}),{x0}+({off}-{x0})*{eout},{off}))))")


def full(n, src, ss, dur, top_label, cap_name, main, sub, with_audio=True, extra=""):
    """素材を画面いっぱいに置き、左上にラベル、下に文字。with_audio なら元の動画の音を通す"""
    vf = f"[0:v]trim=start={ss}:duration={dur},setpts=PTS-STARTPTS,{FIT},fps={FPS},{label(cap_name + '_lb', top_label)}"
    if main:
        vf += "," + caption(cap_name, main, sub)
    if extra:
        vf += "," + extra
    vf += f",{fades(dur)}[v]"
    if with_audio:
        af = f"[1:a]atrim=start={ss}:duration={dur},asetpts=PTS-STARTPTS,aresample=48000,{afades(dur)}[a]"
        ins = ["-i", str(src), "-i", str(a.src)]
    else:
        af = f"anullsrc=r=48000:cl=stereo,atrim=0:{dur}[a]"
        ins = ["-i", str(src)]
    run(ins + ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{dur}"] + ENC + [str(SEG / f"{n:02d}.mp4")])


def card(n, dur, lines):
    """黒地の文字だけの場面。lines = [(文字, 大きさ, 太字か, y)]"""
    parts = []
    for i, (s, size, bold, y) in enumerate(lines):
        font = FONT_B if bold else FONT_R
        color = "white" if bold else "0xDDDDDD"
        parts.append(f"drawtext=fontfile='{font}':textfile='{text_file(f'card{n}_{i}', s)}':fontsize={size}:fontcolor={color}:x=(w-tw)/2:y={y}")
    vf = ",".join(parts) + "," + fades(dur)
    run(["-f", "lavfi", "-t", f"{dur}", "-i", f"color=c={BG}:s={W}x{H}:r={FPS}",
         "-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo",
         "-vf", vf, "-map", "0:v", "-map", "1:a", "-shortest"] + ENC + [str(SEG / f"{n:02d}.mp4")])


def pair(n, inputs, ss, dur, crop, labels, cap_name, main, sub):
    """2 つの素材の同じ所を切り出して左右に並べる。inputs: [(ファイル, 前処理)]。crop = (x, y, w, h) は 1376x768 での範囲"""
    x, y, cw, ch = crop
    avail_h = H - 190 - 60
    scale = min((W - 60) / 2 / cw, avail_h / ch)
    tw, th = int(cw * scale) // 2 * 2, int(ch * scale) // 2 * 2
    ins, fl = [], []
    for i, (src, pre) in enumerate(inputs):
        ins += ["-i", str(src)]
        fl.append(f"[{i}:v]{pre}trim=start={ss}:duration={dur},setpts=PTS-STARTPTS,scale=1376:768,crop={cw}:{ch}:{x}:{y},"
                  f"scale={tw}:{th}:flags=lanczos,fps={FPS},{label(f'{cap_name}_l{i}', labels[i], x=14, y=14, size=34)}[v{i}]")
    gap = 20
    fl.append(f"[v0][v1]hstack=inputs=2[hs]")
    fl.append(f"[hs]pad={tw * 2 + gap}:{th}:0:0:color={BG}[hs2]")
    # 間を空ける: 右の絵を gap だけずらすため、左右を別に置き直す
    fl[-2] = f"[v0]pad={tw + gap}:{th}:0:0:color={BG}[v0p]"
    fl[-1] = f"[v0p][v1]hstack=inputs=2[hs2]"
    top = (H - 190 - th) // 2
    fl.append(f"[hs2]pad={W}:{H}:(ow-iw)/2:{top}:color={BG},{caption(cap_name, main, sub)},{fades(dur)}[out]")
    fl.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur}[a]")
    run(ins + ["-filter_complex", ";".join(fl), "-map", "[out]", "-map", "[a]", "-t", f"{dur}"] + ENC + [str(SEG / f"{n:02d}.mp4")])


def seq_input(folder, name="depth.%05d.png"):
    return ["-framerate", "24", "-i", str(Path(folder) / name)]


def depth_pair(n, ss, dur, crop):
    """深度 518 と 1036 を左右に並べる (同じ所を切り出す)"""
    x, y, cw, ch = crop
    avail_h = H - 190 - 60
    scale = min((W - 60) / 2 / cw, avail_h / ch)
    tw, th = int(cw * scale) // 2 * 2, int(ch * scale) // 2 * 2
    ins = seq_input(a.depth_lo) + seq_input(a.depth_hi)
    labels = ["深度 518（前）", "深度 1036（今）"]
    fl = []
    for i in range(2):
        fl.append(f"[{i}:v]trim=start={ss}:duration={dur},setpts=PTS-STARTPTS,format=gray,scale=1376:768:flags=bicubic,crop={cw}:{ch}:{x}:{y},"
                  f"scale={tw}:{th}:flags=lanczos,format=yuv420p,fps={FPS},{label(f'dp_l{i}', labels[i], x=14, y=14, size=34)}[v{i}]")
    fl.append(f"[v0]pad={tw + 20}:{th}:0:0:color={BG}[v0p]")
    fl.append(f"[v0p][v1]hstack=inputs=2[hs]")
    top = (H - 190 - th) // 2
    fl.append(f"[hs]pad={W}:{H}:(ow-iw)/2:{top}:color={BG},"
              f"{caption('dp', '細い髪や草は深度に入りにくい → 推定を細かく（518 → 1036）', 'モデルを fp16 にして VRAM を約半分に。毛先や葉の先まで入る')},{fades(dur)}[out]")
    fl.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{dur}[a]")
    run(ins + ["-filter_complex", ";".join(fl), "-map", "[out]", "-map", "[a]", "-t", f"{dur}"] + ENC + [str(SEG / f"{n:02d}.mp4")])


def mask_overlay(n, ss, dur):
    """人物のマスク (赤 = 手前の女性、緑 = 奥の女性) を元の動画に半透明で重ねる"""
    vf = (f"[0:v]trim=start={ss}:duration={dur},setpts=PTS-STARTPTS,scale=1376:768,format=gbrp[s];"
          f"[1:v]trim=start={ss}:duration={dur},setpts=PTS-STARTPTS,scale=1376:768,format=gbrp[m];"
          f"[s][m]blend=all_mode=addition:all_opacity=0.45,format=yuv420p,{FIT},fps={FPS},"
          f"{label('mk_lb', '人物のマスク（赤＝手前の女性、緑＝奥の女性）')},"
          f"{caption('mk', '③ 人物のマスク（SAM 3.1）で、人物の中の深度をそろえる', '顔と胸元で深度が違っても、全身が同じようにぼける')},"
          f"{fades(dur)}[v]")
    run(["-i", str(a.src)] + seq_input(a.subject, "subject.%05d.png") +
        ["-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo",
         "-filter_complex", vf, "-map", "[v]", "-map", "2:a", "-t", f"{dur}"] + ENC + [str(SEG / f"{n:02d}.mp4")])


def main():
    global a, R, SEG, OUT, TXT, FONT_B, FONT_R
    ap = argparse.ArgumentParser()
    for k in ("--work", "--fonts", "--bgm", "--src", "--final", "--depth-lo", "--depth-hi", "--subject"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    work = Path(a.work)
    R, SEG, OUT, TXT = work / "renders", work / "segments", work / "out", work / "text"
    for d in (SEG, OUT, TXT):
        d.mkdir(parents=True, exist_ok=True)
    FONT_B = ff_path(Path(a.fonts) / "BIZUDPGothicBold.ttf")
    FONT_R = ff_path(Path(a.fonts) / "BIZUDPGothic.ttf")
    for old in SEG.glob("*.mp4"):
        old.unlink()

    # 1. 元の動画 (タイトルが左下からすべり込む)
    title = (f"drawtext=fontfile='{FONT_B}':textfile='{text_file('title', 'DoF Anime')}':fontsize=96:fontcolor=white:"
             f"box=1:boxcolor=black@0.5:boxborderw=22:y=h-300:x='{slide(0.5, 3.6)}'")
    tsub = (f"drawtext=fontfile='{FONT_R}':textfile='{text_file('title_sub', 'AI アニメ動画に、あとからピントとぼけを足す After Effects プラグイン')}':"
            f"fontsize=42:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=16:y=h-160:x='{slide(0.7, 3.4)}'")
    full(0, a.src, 0, 5.0, "元の動画", "plain", "", "", extra=f"{title},{tsub}")
    # 2. ぼけあり
    full(1, R / "EX_dof_only.mp4", 0.5, 6.0, "DoF Anime", "dof",
         "手前の女性から奥の女性へ、ピントを送る", "Focus Point のキーフレーム 2 つだけ。0.8 秒かけてなめらかに移る")
    # 3. 作り方
    full(2, R / "EX_depth.mp4", 0, 3.5, "Output：Depth", "depth",
         "① 動画から奥行き（深度）を推定", "Video Depth Anything（Small・商用可）。白いほど手前", with_audio=False)
    full(3, R / "EX_blurmap.mp4", 1.5, 3.5, "Output：Blur Map", "blurmap",
         "② 深度の差から、ぼけの大きさを決める", "手前＝青、奥＝橙、ピント＝黒。手前と奥を分けるので背景がにじまない", with_audio=False)
    mask_overlay(4, 2.0, 3.5)
    # 4. 草や髪の毛の抜けが甘い所 (拡大して見せる)
    depth_pair(5, 1.0, 4.0, (40, 150, 440, 560))
    pair(6, [(R / "EX_no_exclude.mp4", ""), (R / "EX_dof_only.mp4", "")], 0.8, 2.6, (0, 398, 405, 279),
         ["直す前", "Exclude（grass）"], "grass",
         "人物のマスクに入った草が途中で切れる → Exclude で抜く",
         "草自身の手前寄りの深度にそろえ、根元から先まで同じ前ぼけに")
    pair(7, [(R / "EX_nearfill0.mp4", ""), (R / "EX_dof_only.mp4", "")], 5.5, 2.6, (0, 170, 320, 360),
         ["Near Fill 0", "Near Fill 16"], "nearfill",
         "ぼけた髪の束の縁が線で残る → Near Fill",
         "手前のぼけを周りへ広げ、束の間のすき間もまとめてぼかす")
    # 5. 完成版
    full(8, a.final, 0, 8.0, "完成版：Relight Anime ＋ DoF Anime ＋ FilmConvert Nitrate", "final", "", "")

    segs = sorted(SEG.glob("*.mp4"))
    lst = SEG / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in segs), encoding="utf-8")
    joined = OUT / "joined_nobgm.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(joined)],
                                 capture_output=True, text=True, check=True).stdout.strip())
    out = OUT / "dof_explainer.mp4"
    af = (f"[1:a]atrim=0:{total},asetpts=PTS-STARTPTS,aresample=48000,volume={BGM_GAIN_DB}dB,"
          f"afade=t=in:st=0:d=1.5,afade=t=out:st={total - 2.5:.2f}:d=2.5[bgm];"
          f"[0:a][bgm]amix=inputs=2:duration=first:normalize=0[a]")
    run(["-i", str(joined), "-i", str(a.bgm), "-filter_complex", af, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])
    joined.unlink()
    print(f"保存: {out} ({total:.1f}s)")


if __name__ == "__main__":
    main()

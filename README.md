# DoF Anime

AI で作ったアニメ動画に、あとから被写界深度（ピントの外のぼけ）を付ける After Effects エフェクトです。
動画から奥行き（深度）を推定し、画面上の点を選ぶとその深さにピントを合わせて、手前と奥をぼかします。

[English](README.en.md) ｜ 作り方: [docs/how-it-works.ja.md](docs/how-it-works.ja.md)

- **ピント合わせ**: Focus Point をコンポビューでドラッグすると、その位置の深度にピントが合います。キーフレームを打てばピント送りになり、Focus Pull Time の時間で映画のフォーカス送りのようになめらかに移ります。深度の値で指定することもできます（Focus Mode = Depth）。
- **にじまない境目**: 手前と奥を分けてぼかします。ピントの合った人物に、ぼけた背景がにじみません。手前のぼけは、後ろの物の上に半透明に広がります。深度と絵の輪郭のずれは Edge Bias / Edge Soften でなめらかにつなぎます。
- **玉ぼけ**: 色はリニアの明るさで混ぜるので、明るい所が玉ぼけになります。羽根の枚数・丸み・縦横比・口径食（Cat Eye）・縁の明るさ（Bubble）・色ずれ（Fringe）を選べます。
- **キャラのマスク（任意）**: SAM 3.1 でキャラを見つけて動画全体で追いかけます。マスクは人物の内側を整えるだけに使い（Subject Inset）、縁は深度で決めます。
- **速さ**: GPU（CUDA）で 1 コマ約 2〜3ms です（RTX 5090、1376×768）。CPU でも描けます（同じ大きさで約 0.1 秒）。CPU と GPU の差は 8bit で最大 1 段です。
- **ちらつき対策**: 深度は前後のフレームと合わせて時間方向にならしてから使います。

深度の推定は [Relight Anime](https://github.com/yuichi-suzuki-highdrama/relight-anime) と同じ仕組み（Video Depth Anything、`relight_depth.exe` か別の PC の深度サーバー）を使います。設定とキャッシュも共有するので、Relight Anime で深度を作った素材なら作り直さずにすぐ使えます。同じコンポに Relight Anime の深度レイヤーがあれば、それを使い回します。

## 構成

```
plugin/
  src/DofCore.h       ぼかしの計算 (CPU と GPU で共通。1 画素ぶんの関数)
  src/DofAE.cpp       エフェクト本体 (パラメータ、CPU 版の手順、AE とのやりとり、ピントの目印)
  src/DofGPU.cu       GPU 版の手順 (CUDA)
  src/DofAuto.cpp     深度の自動生成と結線 (Relight Anime の RelightAuto を写したもの)
  src/DofHelper.cpp   補助 AEGP (アイドルフックで自動生成のジョブを進める)
  build.ps1           ビルド (VS2022 BuildTools + CMake + Ninja、CUDA があれば GPU 版も)
tools/harness/
  dof_cli.cpp         プラグインのコードを AE なしで動かす確認用ツール
  run_cli.py          動画の 1 コマをぼかして、元 / ぼかし / ぼけの大きさを並べる
```

## 必要なもの

- Windows x64、After Effects 2024 以降（AE 2026 で確認）
- [After Effects SDK](https://developer.adobe.com/after-effects/)
- Visual Studio 2022 Build Tools（C++）、CMake、Ninja
- CUDA Toolkit 12.8 以降（GPU 版。無ければ CPU 版だけを作ります）
- 深度の自動生成を使う場合: Relight Anime の深度推定の準備（`relight_depth.exe` とモデル、`%APPDATA%\SRLM\RelightAnime.ini`）。手で深度の連番を用意して Depth Layer に指定してもかまいません

## ビルドと導入

```powershell
.\plugin\build.ps1 -SdkRoot C:\dev\AE_SDK
# 管理者の PowerShell で (AE を閉じてから)
.\plugin\build.ps1 -SdkRoot C:\dev\AE_SDK -Install
```

`DoFAnime.aex` と `DoFAnimeHelper.aex` が `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\SRLM` に入ります。
DoF Anime だけ設定を変えたいときは、`%APPDATA%\SRLM\DoFAnime.ini` に RelightAnime.ini と同じ形（key=value）で書きます。

## 使い方

1. 動画のフッテージレイヤーに「エフェクト > SRLM > DoF Anime」を掛けます。
2. 深度が自動で作られ、非表示の「DoF depth」レイヤーとして結線されます（キャッシュがあればすぐ）。
3. コンポビューの「Focus」の枠をピントを合わせたい所へドラッグします。
4. Blur Size でぼけの大きさを決めます。手前と奥は Near Blur / Far Blur で別に調整できます。
5. 既定ではキャラのマスクを使わず、深度だけでぼかします（縁は Edge Bias / Edge Soften でなめらかにつなぎます）。キャラ全体をそろえてくっきりさせたいときは、Auto Subject を入れるか Generate Subject を押すと、キャラのマスクが作られて Subject Layer に結線されます（ロトブラシなどで切り抜いたレイヤーを手で指定してもかまいません）。推定した深度はキャラの中でも揺れる（髪の先・脚・フリル）ので、マスクの中の深度をキャラの代表の深度に寄せます（Subject Flatten）。Focus Point をキャラの上に置けばキャラ全体にピントが合い、背景へ動かせばキャラは全体が均一にぼけます。

Output を Blur Map にすると、ぼけの大きさが見えます（手前 = 青、奥 = 橙、ピント = 黒）。Focus Check はピントの合っている所を緑で示します。

## パラメータ

| グループ | 項目 | 内容 |
|---|---|---|
| Auto | Auto Passes / Depth Engine / Generate Depth / Clean Cache... | 深度の自動生成（Relight Anime と同じ） |
| Focus | Focus Mode | Point: Focus Point の位置の深度に合わせる。Depth: Focus Depth の値で合わせる |
| | Focus Point | ピントを合わせる位置（コンポビューでドラッグ） |
| | Focus Pull Time | ピント送りにかける時間（秒、既定 0.8）。映画のフォーカス送りのように、ゆっくり動き始めてゆっくり止まる。停止したキーフレームでパッと切り替えても、この時間で移る。0 でキーフレームのとおり |
| | Focus Depth | Focus Mode = Depth のときの深度（0..1、近い = 1） |
| | Focus Range | ピントの合う幅（深度の差）。広げると人物全体にピントが合う |
| | Subject Layer / Subject Channel | キャラのマスク（自動で作ったもの、またはロトブラシで切り抜いたレイヤーなど） |
| | Subject Flatten | マスクの中の深度を、キャラの代表の深度（マスクの中の深度の中央値）にどれだけ寄せるか。100% でキャラを 1 枚の板として扱い、キャラの中のぼけのむらが無くなる。0% で深度そのまま |
| | Subject Inset | マスクを内側へ縮める幅（既定 12 px）。マスクは人物の内側を整えるだけに使い、人物の縁は深度で決める（マスクの縁のガタガタが出ない）。0 でマスクをそのまま使う |
| | Keep Subject Sharp | ピントに関係なく、マスクの中を必ずくっきり残す強さ（手前のぼけも重ねない）。既定は 0% |
| | Sharp Layer / Blur Layer | 手直し。自動で拾えない所を、平面にマスクを描いて指定する（アルファの所が対象）。Sharp Layer の所は必ずくっきり、Blur Layer の所は画面のいちばん奥と同じだけぼかす |
| | Sharp by Text / Blur by Text | 「Text...」を押して物の名前を英語で入れると（例: flower, grass。カンマで複数）、SAM 3.1 がその物を動画全体で探して追いかけ、マスクを作って Sharp Layer / Blur Layer に結線する。「Focus Point の所の物を 1 つだけ」を選べば、その所の 1 つだけ |
| | Auto Subject / Subject Type / Generate Subject | キャラのマスクを自動で作る（既定は切。SAM 3.1 でキャラを見つけて動画全体で追いかけ、深度で大きな取りこぼし・拾いすぎを直し、縁を元の絵の色から切り抜き直す）。別のキャラにピントを移したら Generate Subject で作り直す。縁の仕上げ方は設定ファイル（%APPDATA%\SRLM\DoFAnime.ini）の `subject_matte` で変えられる: `depthcf`（既定）/ `cf`（深度を使わない）/ `sam2matting`（いちばんきれいだが**非商用**）/ `none` |
| Blur | Blur Size | ピントから深度が 0.5 離れた所のぼけの半径（1080p でのピクセル数） |
| | Near Blur / Far Blur | 手前 / 奥のぼけの倍率。Near Blur の既定は 0（手前はぼかさず奥だけ）。手前ボケが欲しいカットだけ上げる |
| | Edge Expand | 手前の物の輪郭を広げる幅。推定した深度の境目より外にある輪郭線を、手前の物に入れる |
| | Edge Bias / Edge Soften | 深度と絵の輪郭のずれで出る細い帯を消す。Bias: ぼけている側をくっきりした側へ食い込ませる幅。Soften: くっきりした物の縁から内側へ、ぼけを少しずつ弱めていく幅 |
| Bokeh | Shape / Rotation | 玉ぼけの形（円・六角形・八角形・Polygon）と向き |
| | Blades / Roundness | Shape = Polygon の羽根の枚数（3〜10）と、多角形と円の中間 |
| | Aspect / Cat Eye / Bubble / Fringe | 玉の縦横比（アナモルフィック風）、画面の端ほど玉が欠ける口径食、玉の明るさの分け方（正: 縁を明るく。負: 中心を明るく縁をやわらかく。細い髪の束をぼかすときは負がきれい）、玉の縁の色ずれ |
| | Highlights / Highlight Threshold | 明るい所を玉ぼけにする強さと、し始める明るさ |
| | Quality | 点の数（Draft 48 / Normal 96 / High 192） |
| Passes | Depth Layer など | 深度レイヤー、白黒の向き、時間方向のならし、線画に合わせる、GPU を使うか |
| Output | Output / Mix | Final / Depth / Blur Map / Focus Check、元との混ぜ具合 |

## 仕組み

詳しくは [docs/how-it-works.ja.md](docs/how-it-works.ja.md)（境目の工夫、キャラのマスク、レンズらしさ、AE 標準の Camera Lens Blur との違い、一から作る手順）。

1. 入力をリニアの明るさにし、1/2・1/4・1/8 の縮小も作ります。
2. 深度を前後のフレームの中央値でならし、Focus Point のまわりの中央値をピントの深度にします。
3. 深度を出力の大きさへ引き伸ばします。深度が変わる所だけ、素材の色が近い深度の点を重く見て、境目を線画に合わせます（joint bilateral upsampling）。
4. ピントとの深度の差からぼけの半径（CoC）を符号付きで求めます。推定した深度は視差に近い量なので、薄いレンズの式どおり差に比例させます。
5. 16 画素四方の区画ごとに、その区画に届きうるぼけの最大の半径を求めます。
6. 各画素のまわりを黄金角のらせんに並べた点で調べ、点の円がこの画素に届けば色を足します（scatter-as-gather）。
   - 奥とピントの層: 奥の点の広がりは、この画素のぼけまでに抑えます。ピントの合った物に背景がにじみません。
   - 手前の層: 手前の点は後ろの物の上に広がります。届いた量から覆う割合を求めて重ねます。
   - 点の間隔が広いときは縮小した画像からバイリニアで読み、ぼけの中に模様が出ないようにします。

## 既知の限界

- 推定した深度の境目が素材の輪郭からずれる所では、ぼけの境目もずれます。Edge Expand を上げすぎると、輪郭のすぐ外の背景がくっきり残ります。
- 手前の物を大きくぼかすと、本来その裏に見えるはずの背景は元の絵に無いので、まわりの色で埋まります。
- 人物の中でも深度が揺れるので、Focus Range が狭いと顔の一部だけぼけることがあります（キャラのマスクと Subject Flatten でそろえられます）。
- 細い髪の毛先が深度に入らず、線のまま残ることがあります。設定ファイルの `depth_input_size` を 770 にすると毛先まで入ります（VRAM 約 21GB）。

## ライセンス

[MIT](LICENSE)。深度のモデルは Video Depth Anything（Small は Apache-2.0、**Large は CC-BY-NC-4.0 で非商用**。Remote Quality で使います）。キャラのマスクは SAM 3.1（Meta の SAM License）と pymatting（MIT）。`subject_matte=sam2matting` を選んだ時だけ SAM2Matting（**CC BY-NC 4.0 で非商用**。ComfyUI-SAM2Matting に入っているものを使い、このリポジトリには含めません）。After Effects SDK は含めていません。

/*
 * DofCore.h
 * 被写界深度 (ぼかし) の計算そのもの (CPU と GPU で共通)。
 *
 * C++ (CPU 版、確認用ツール) と CUDA (GPU 版、DofGPU.cu) の両方から取り込む。
 * AE の SDK には依存しない。関数は DF_HD で CPU・GPU どちらでも動く。
 * 1 画素ぶんの処理だけをここに置き、全体の手順は
 *   CPU: DofAE.cpp の RenderFrameCPU
 *   GPU: DofGPU.cu の DofGPU_Render
 * がそれぞれ持つ。同じ関数を呼ぶので、CPU と GPU の結果は (浮動小数の誤差を除いて) 一致する。
 *
 * 考え方:
 *   深度 (0..1、近い = 1) は推定した相対的な奥行き (視差に近い量) なので、
 *   ぼけの大きさ (CoC、錯乱円の半径) はピントの深度との差に比例させる (薄いレンズの式で、視差の差に比例する)。
 *   符号付きで持ち、手前 (ピントより近い) を負、奥を正にする。
 *
 *   ぼかしは「各画素が自分の CoC の円に光を広げる」(scatter) のを、出力の画素から集める形 (gather) で計算する。
 *   出力の画素 p のまわりを円形に並べた点で調べ、点 q の円が p に届けば q の色を足す。
 *   - 奥と、ピントの合った所 (後ろの層): q が p より奥なら、q の広がりを p の CoC までに抑える。
 *     そうするとピントの合った人物に、ぼけた背景がにじまない (人物の輪郭が背景に食われない)。
 *     ピントの合った人物の色も、背景のぼけに混ざらない (人物の CoC は 0 なので広がらない)。
 *   - 手前 (前の層): 手前のぼけは後ろの物の上に半透明に広がる。届いた量から覆う割合 (アルファ) を求め、
 *     後ろの層の上に重ねる。
 *   色はリニアの明るさで混ぜる (明るい所が丸い玉ぼけになる)。
 *
 * 座標: x, y は出力の画素。レイヤー座標は (x + originX, y + originY)。
 */
#pragma once

#include <math.h>

#ifdef __CUDACC__
#define DF_HD  __host__ __device__
#define DF_INL __forceinline__
#else
#define DF_HD
#define DF_INL inline
#endif

/* ------------------------------------------------------------------ */
/* 描画の設定                                                           */
/* ------------------------------------------------------------------ */

enum {
	FOCUS_POINT = 1,     /* 画面上の点の深度にピントを合わせる */
	FOCUS_DEPTH          /* 深度の値で指定する */
};

enum {
	SHAPE_CIRCLE = 1,
	SHAPE_HEXAGON,
	SHAPE_OCTAGON
};

enum {
	QUALITY_DRAFT = 1,
	QUALITY_NORMAL,
	QUALITY_HIGH
};

enum {
	OUTPUT_FINAL = 1,
	OUTPUT_DEPTH,        /* 深度 (白 = 近い) */
	OUTPUT_BLUR_MAP,     /* ぼけの大きさ (手前 = 青、奥 = 橙、ピント = 黒) */
	OUTPUT_FOCUS_CHECK   /* ピントの合っている所を緑で示す */
};

/* ピント送りの 1 コマ分: Focus Point が動いている間は、動き始めの位置 A のピントから動き終わりの位置 B のピントへ、
   点の進み具合 t (A→B の上での割合) に合わせて深度を移す (途中で通る背景にはピントを合わせない)。止まっていれば A = B */
struct DfFocusSample {
	float ax, ay, bx, by;
	float t;                  /* A→B の進み具合 (0..1。キーフレームのイーズがそのまま効く) */
	float w;                  /* 重み */
};
static constexpr int DF_PULL_MAX = 25;   /* ピント送りをならすときに見るコマの数の上限 (長い窓は間引く) */

struct DofParams {
	int   focusMode;
	float focusX, focusY;     /* Focus Point (レイヤー座標、ダウンサンプル後) */
	/* ピント送り (Focus Pull Time): 前後のコマのピントを、なめらかな重み (Hann) で平均する。
	   映画のフォーカス送りのように、ゆっくり動き始めてゆっくり止まる (停止したキーフレームでも急に切り替わらない)。
	   0 なら今のコマの Focus Point だけ */
	int   pullCount;
	DfFocusSample pull[DF_PULL_MAX];
	float focusDepth;         /* Focus Mode = Depth のときの深度 (0..1、近い = 1) */
	float focusRange;         /* ピントの合う幅 (深度の差がこれ以下ならぼかさない) */
	int   subjectChannel;     /* Subject Layer の読み方 1: アルファ  2: 明るさ  3: Multi (RGB。赤・緑・青 = 1・2・3 人目のキャラ) */
	float subjectAmount;      /* Keep Subject Sharp: ピントに関係なく、マスクの中をどれだけくっきり残すか (0..1) */
	float subjectFlatten;     /* Subject Flatten: マスクの中の深度を人物の代表の深度にどれだけ寄せるか (0..1)。
	                             1 なら人物を 1 枚の板として扱い、人物の中のぼけのむらが無くなる */
	float blurSize;           /* 深度の差 0.5 のときのぼけの半径 (1080p でのピクセル数) */
	float nearScale;          /* 手前のぼけの倍率 */
	float farScale;           /* 奥のぼけの倍率 */
	float edgeExpand;         /* 手前の物の輪郭を広げる幅 (1080p でのピクセル数。輪郭線を手前の物に含める) */
	int   shape;
	float shapeRotation;      /* 度 */
	float highlights;         /* 明るい所を玉ぼけにする強さ */
	float highlightThreshold; /* 玉ぼけにし始める明るさ (0..1) */
	int   quality;
	/* passes */
	int   depthNear;          /* 1: 白 = 近い  2: 黒 = 近い */
	bool  depthSnap;          /* 深度の境目を素材の線画に合わせる */
	int   depthStabilize;     /* 深度を前後何フレームで時間方向にならすか (0 = しない) */
	int   passEncoding;       /* 1: Linear  2: sRGB (AE が変換した場合に戻す) */
	bool  inputLinear;        /* 入力がリニア (リニア作業空間) */
	/* output */
	int   outputMode;
	float mix;
};

/* ------------------------------------------------------------------ */
/* 定数                                                                 */
/* ------------------------------------------------------------------ */

static constexpr int   DF_MAX_SUBJECTS    = 3;       /* キャラを別々に扱う人数 (マスクの赤・緑・青) */
static constexpr int   DF_LEVELS          = 4;       /* 色の縮小の段 (1, 1/2, 1/4, 1/8) */
static constexpr int   DF_TILE            = 16;      /* ぼけの最大値を求める区画の一辺 (画素) */
static constexpr float DF_MAX_COC_1080    = 160.f;   /* ぼけの半径の上限 (1080p でのピクセル数) */
static constexpr float DF_DEPTH_REF       = 0.5f;    /* Blur Size が指す深度の差 */
static constexpr float DF_FOCUS_SOFT      = 0.02f;   /* ピントの幅の外で、ぼけが立ち上がる幅 (深度) */
static constexpr float DF_STEP_1080       = 8.f;     /* 深度の段差を締めるときに見るまわりの半径 (1080p でのピクセル数) */
static constexpr float DF_STEP_LO         = 0.05f;   /* まわりの深度の幅がこれより小さければ段差ではない (なだらかな坂は締めない) */
static constexpr float DF_STEP_HI         = 0.10f;   /* これより大きければ段差として締め切る */
static constexpr float DF_STEP_NEAR_PENALTY = 0.02f; /* 段差の近くのキャラを選ぶとき、1 画素離れるごとにマスクの値から引く量 (いちばん近いキャラを選ぶ) */
static constexpr float DF_STEP_MASK       = 0.5f;    /* キャラの縁とみなすマスクの値 (ここで決め切る)。自動のマスクは縁を元の絵から切り抜き直して
                                                         輪郭線まで濃い値で入っているので真ん中でよい。低くすると、縁の外の薄い値の点がトゲのようにくっきり残る */
static constexpr float DF_FOCUS_WINDOW    = 0.012f;  /* Focus Point のまわりで深度を調べる範囲 (画面の高さに対する半径) */
static constexpr int   DF_STABILIZE_MAX   = 2;       /* 前後何フレームまで */
static constexpr float DF_SNAP_SIGMA      = 0.08f;   /* 深度を線画に合わせるとき、色の近さの幅 */
static constexpr float DF_GOLDEN_ANGLE    = 2.39996323f;

/* 品質ごとの点の数 (ぼけが大きいとき) */
DF_HD DF_INL int dfMaxSamples(int quality)
{
	return quality == QUALITY_DRAFT ? 48 : (quality == QUALITY_HIGH ? 192 : 96);
}

/* ------------------------------------------------------------------ */
/* 小さな道具                                                           */
/* ------------------------------------------------------------------ */

template <typename T> DF_HD DF_INL T dfMin(T a, T b) { return a < b ? a : b; }
template <typename T> DF_HD DF_INL T dfMax(T a, T b) { return a > b ? a : b; }
template <typename T> DF_HD DF_INL T dfClamp(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

DF_HD DF_INL float dfSat(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }
DF_HD DF_INL float dfMix(float a, float b, float t) { return a + (b - a) * t; }
DF_HD DF_INL float dfSmooth(float e0, float e1, float x)
{
	const float t = dfSat((x - e0) / (e1 - e0));
	return t * t * (3.f - 2.f * t);
}
DF_HD DF_INL int dfRound(float x) { return (int)floorf(x + 0.5f); }

DF_HD DF_INL float dfSrgbToLinear(float c)
{
	if (c <= 0.f) return c;
	return (c <= 0.04045f) ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

/* ガンマ 2.2 の表 (powf を避ける)。dec[i] = (i/1024)^2.2、enc[i] = (i/1024)^(2/2.2) (enc は sqrt を挟む) */
#define DF_LUT_SIZE 1026
inline void dfBuildLut(float *dec, float *enc)
{
	for (int i = 0; i < DF_LUT_SIZE; ++i) {
		const double x = (i / 1024.0) < 1.0 ? i / 1024.0 : 1.0;
		dec[i] = (float)pow(x, 2.2);
		enc[i] = (float)pow(x, 2.0 / 2.2);
	}
}
DF_HD DF_INL float dfLut(const float *t, float x)
{
	const float f = dfSat(x) * 1024.f;
	const int i = (int)f;
	return t[i] + (t[i + 1] - t[i]) * (f - (float)i);
}

/* ------------------------------------------------------------------ */
/* 画像の読み取り (AE のワールドと GPU のワールドの両方)                  */
/* ------------------------------------------------------------------ */

struct DfRgba { float a, r, g, b; };

enum {
	DF_FMT_ARGB8 = 0,    /* PF_Pixel8 */
	DF_FMT_ARGB16,       /* PF_Pixel16 (最大値 32768) */
	DF_FMT_ARGB32F,      /* PF_PixelFloat */
	DF_FMT_BGRA32F,      /* AE の GPU ワールド (PF_PixelFormat_GPU_BGRA128) */
	DF_FMT_GRAY32F       /* 1 チャンネルの float (確認用ツールの深度) */
};

struct DfImage {
	const unsigned char *data;   /* NULL なら無し */
	int width, height, rowbytes, format;
};

/* 端はクランプ。premultiplied のまま返す */
DF_HD DF_INL DfRgba dfRead(const DfImage &im, int x, int y)
{
	DfRgba o;
	o.a = o.r = o.g = o.b = 0.f;
	if (!im.data) return o;
	x = dfClamp(x, 0, im.width - 1);
	y = dfClamp(y, 0, im.height - 1);
	const unsigned char *row = im.data + (size_t)y * (size_t)im.rowbytes;
	switch (im.format) {
	case DF_FMT_ARGB32F: {
		const float *p = (const float *)row + (size_t)x * 4;
		o.a = p[0]; o.r = p[1]; o.g = p[2]; o.b = p[3];
		break;
	}
	case DF_FMT_BGRA32F: {
		const float *p = (const float *)row + (size_t)x * 4;
		o.b = p[0]; o.g = p[1]; o.r = p[2]; o.a = p[3];
		break;
	}
	case DF_FMT_ARGB16: {
		const unsigned short *p = (const unsigned short *)row + (size_t)x * 4;
		const float s = 1.f / 32768.f;
		o.a = p[0] * s; o.r = p[1] * s; o.g = p[2] * s; o.b = p[3] * s;
		break;
	}
	case DF_FMT_GRAY32F: {
		const float v = ((const float *)row)[x];
		o.a = 1.f; o.r = o.g = o.b = v;
		break;
	}
	default: {
		const unsigned char *p = row + (size_t)x * 4;
		const float s = 1.f / 255.f;
		o.a = p[0] * s; o.r = p[1] * s; o.g = p[2] * s; o.b = p[3] * s;
		break;
	}
	}
	return o;
}

/* 深度レイヤーの位置合わせ。深度は素材と大きさが違う (短辺 518px) ので、レイヤー全体を取り出し、
   「素材レイヤーの中での位置の割合」で対応させる。出力の画素 x → 深度ワールドの x * sx + ox */
struct DfAuxMap {
	DfImage img;
	float sx, sy, ox, oy;
};

/* fullW, fullH: 深度レイヤー全体の大きさ (ダウンサンプル後)。originX/Y: 出力 (0,0) のレイヤー座標 */
DF_HD DF_INL DfAuxMap dfMakeAuxMap(const DfImage &img, float fullW, float fullH, float originX, float originY,
                                   float layerW, float layerH)
{
	DfAuxMap m;
	m.img = img;
	m.sx = dfMax(1.f, fullW) / layerW;
	m.sy = dfMax(1.f, fullH) / layerH;
	m.ox = (originX + 0.5f) * m.sx - 0.5f;
	m.oy = (originY + 0.5f) * m.sy - 0.5f;
	return m;
}

/* ------------------------------------------------------------------ */
/* 描画コンテキスト                                                     */
/* ------------------------------------------------------------------ */

struct DofContext {
	DofParams p;
	int   outW, outH;
	float originX, originY;   /* 出力 (0,0) のレイヤー座標 */
	float layerW, layerH;
	const float *lutDec, *lutEnc;
	/* 1 フレームごとに決まる値 (dfSetupFrame) */
	float focus;              /* ピントの深度 (Focus Point なら深度から求めた値) */
	float cocScale;           /* 深度の差 1 あたりのぼけの半径 (出力の画素) */
	float maxCoc;             /* ぼけの半径の上限 (出力の画素) */
	float expand;             /* 手前の輪郭を広げる幅 (出力の画素) */
	float step;               /* 深度の段差を締めるときに見るまわりの半径 (出力の画素) */
	int   maxSamples;
	int   polySides;          /* 0 = 円 */
	float polyRot;            /* ラジアン */
	int   tilesX, tilesY;
	/* 作業領域 (出力の大きさ) */
	const float *depth;       /* 深度 (近い = 1) */
	const float *coc;         /* 符号付きのぼけの半径 (手前 = 負) */
	const float *tileR;       /* 区画ごとの、届きうるぼけの最大の半径 */
	const float *subject;     /* キャラのマスク (全員を合わせたもの、0..1、出力の大きさ。NULL なら無し) */
	const float *subjectTgt;  /* その画素のキャラの代表の深度 (Subject Flatten で寄せる先。キャラでなければ負) */
	float subjectDepth[DF_MAX_SUBJECTS];   /* キャラごとの代表の深度 (マスクの中の深度の中央値)。いなければ負 */
	/* リニアの色 (premultiplied、RGBA の順に 4 個ずつ)。lin[0] が出力の大きさ、lin[k] は 1/2^k */
	const float *lin[DF_LEVELS];
	int   lw[DF_LEVELS], lh[DF_LEVELS];
};

DF_HD DF_INL float dfDecode(const DofContext *c, float x) { return dfLut(c->lutDec, x); }
DF_HD DF_INL float dfEncode(const DofContext *c, float x) { return dfLut(c->lutEnc, sqrtf(dfSat(x))); }

/* ピントの深度が決まったあとに呼ぶ (CPU と GPU のどちらも、描く前にホストで) */
inline void dfSetupFrame(DofContext &c, float focusDepth)
{
	const float s = c.layerH / 1080.f;
	c.focus      = focusDepth;
	c.cocScale   = dfMax(0.f, c.p.blurSize) * s / DF_DEPTH_REF;
	c.maxCoc     = DF_MAX_COC_1080 * s;
	c.expand     = dfMax(0.f, c.p.edgeExpand) * s;
	c.step       = DF_STEP_1080 * s;
	c.maxSamples = dfMaxSamples(c.p.quality);
	c.polySides  = c.p.shape == SHAPE_HEXAGON ? 6 : (c.p.shape == SHAPE_OCTAGON ? 8 : 0);
	c.polyRot    = c.p.shapeRotation * 3.14159265f / 180.f;
	c.tilesX     = (c.outW + DF_TILE - 1) / DF_TILE;
	c.tilesY     = (c.outH + DF_TILE - 1) / DF_TILE;
	c.lw[0] = c.outW; c.lh[0] = c.outH;
	for (int k = 1; k < DF_LEVELS; ++k) { c.lw[k] = (c.lw[k - 1] + 1) / 2; c.lh[k] = (c.lh[k - 1] + 1) / 2; }
}

/* ------------------------------------------------------------------ */
/* 深度                                                                 */
/* ------------------------------------------------------------------ */

/* 深度ワールドの 1 点 → 0..1 (近い = 1) */
DF_HD DF_INL float dfLowDepth(const DfImage &img, int i, int j, int depthNear, bool decodeSrgb)
{
	DfRgba d = dfRead(img, i, j);
	if (decodeSrgb) { d.r = dfSrgbToLinear(d.r); d.g = dfSrgbToLinear(d.g); d.b = dfSrgbToLinear(d.b); }
	float v = dfSat(0.2126f * d.r + 0.7152f * d.g + 0.0722f * d.b);
	if (depthNear == 2) v = 1.f - v;
	return v;
}

DF_HD DF_INL bool dfDepthValid(const DfImage &img, int i, int j) { return img.data && dfRead(img, i, j).a > 0.5f; }

/* 深度を時間方向にならす (1 点)。前後のフレームの同じ位置の深度との中央値 (Relight Anime と同じ)。
   飛び出した値は消え、一定の速さで動く物の輪郭は今のフレームの位置のまま */
DF_HD DF_INL float dfStabilizeDepth(float d0, const DfImage *nb, int n, int i, int j, int depthNear, bool decodeSrgb)
{
	float v[5];
	int m = 0;
	v[m++] = d0;
	for (int k = 0; k < n && m < 5; ++k)
		if (dfDepthValid(nb[k], i, j)) v[m++] = dfLowDepth(nb[k], i, j, depthNear, decodeSrgb);
	for (int a = 1; a < m; ++a) {
		const float x = v[a];
		int b = a - 1;
		while (b >= 0 && v[b] > x) { v[b + 1] = v[b]; --b; }
		v[b + 1] = x;
	}
	return (m & 1) ? v[m / 2] : 0.5f * (v[m / 2 - 1] + v[m / 2]);
}

/* 素材の画素の表示の色 (0..1)。深度を線画に合わせる手がかり */
DF_HD DF_INL void dfGuide(const DofContext *c, const DfImage &src, int x, int y, float g[3])
{
	const DfRgba p = dfRead(src, x, y);
	const float a = p.a > 1e-6f ? p.a : 1.f;
	g[0] = p.r / a; g[1] = p.g / a; g[2] = p.b / a;
	for (int k = 0; k < 3; ++k) {
		if (c->p.inputLinear) g[k] = dfEncode(c, g[k]);
		g[k] = dfSat(g[k]);
	}
}

/* 深度を出力の大きさへ引き伸ばす (1 画素)。Snap Depth To Lines がオンなら、深度が変わる所だけ
   周り 4x4 の深度の点のうち「素材の色がこの画素に近いもの」を重く見て混ぜる (joint bilateral upsampling。Relight Anime と同じ) */
DF_HD DF_INL float dfUpsampleDepth(const DofContext *c, const float *low, int dw, int dh, const DfAuxMap &map,
                                   const DfImage &src, int x, int y)
{
	const float fx = dfClamp((float)x * map.sx + map.ox, 0.f, (float)(dw - 1));
	const float fy = dfClamp((float)y * map.sy + map.oy, 0.f, (float)(dh - 1));
	const int x0 = (int)fx, y0 = (int)fy;
	const float tx = fx - (float)x0, ty = fy - (float)y0;
#define DF_DAT(i, j) low[(size_t)dfClamp((j), 0, dh - 1) * dw + dfClamp((i), 0, dw - 1)]
	const float a = DF_DAT(x0, y0), b = DF_DAT(x0 + 1, y0), cc = DF_DAT(x0, y0 + 1), d = DF_DAT(x0 + 1, y0 + 1);
	const float bil = dfMix(dfMix(a, b, tx), dfMix(cc, d, tx), ty);
	const float lo = dfMin(dfMin(a, b), dfMin(cc, d)), hi = dfMax(dfMax(a, b), dfMax(cc, d));
	if (!c->p.depthSnap || hi - lo < 0.02f) return bil;
	float g0[3];
	dfGuide(c, src, x, y, g0);
	float sum = 0.f, wsum = 0.f;
	for (int j = y0 - 1; j <= y0 + 2; ++j)
		for (int i = x0 - 1; i <= x0 + 2; ++i) {
			const int px = dfClamp(dfRound(((float)i - map.ox) / map.sx), 0, c->outW - 1);
			const int py = dfClamp(dfRound(((float)j - map.oy) / map.sy), 0, c->outH - 1);
			float g[3];
			dfGuide(c, src, px, py, g);
			const float cd = dfMax(fabsf(g[0] - g0[0]), dfMax(fabsf(g[1] - g0[1]), fabsf(g[2] - g0[2])));
			const float ds2 = ((float)i - fx) * ((float)i - fx) + ((float)j - fy) * ((float)j - fy);
			const float w = expf(-0.5f * ds2) * expf(-cd * cd * (1.f / (2.f * DF_SNAP_SIGMA * DF_SNAP_SIGMA)));
			sum += DF_DAT(i, j) * w;
			wsum += w;
		}
#undef DF_DAT
	return wsum > 1e-6f ? sum / wsum : bil;
}

/* 手前の物の輪郭を広げる (1 方向の最大値、1 画素)。手前ほど深度が大きいので最大値を取る。
   推定した深度の境目は人物の輪郭線より少し内側にあることが多く、そのままだと輪郭線が背景としてぼけ、
   背景のぼけに黒い縁が出る。輪郭線ぶん手前の物を広げて、輪郭線を人物の側に入れる */
DF_HD DF_INL float dfExpand1D(const float *in, int W, int H, int rad, bool horizontal, int x, int y)
{
	float m = in[(size_t)y * W + x];
	for (int k = -rad; k <= rad; ++k) {
		const int xx = horizontal ? dfClamp(x + k, 0, W - 1) : x;
		const int yy = horizontal ? y : dfClamp(y + k, 0, H - 1);
		m = dfMax(m, in[(size_t)yy * W + xx]);
	}
	return m;
}

/*
 * 深度の段差を締める。深度の地図は物の境目でなだらかにつながっているので、手前の物と奥の物の境目の途中の深度が
 * ピントの深度を横切り、境目にピントの合った細い線が出る (ぼけた手前のキャラの縁がくっきり切れて見える)。
 * ピントの合ったキャラの縁も、途中の深度だと半端にぼけて筋になる。
 * そこで、まわりの深度の幅が大きい所 (段差) では、深度を近い側か遠い側のどちらか近い方に寄せる。
 */
/* キャラのマスクがあるときは、境目をマスクの縁で決める。深度の地図のキャラはマスクより少し太っているので、
   深度だけで決めると、キャラのすぐ外の背景がキャラの深度に寄って、くっきりした背景の帯が残る。
   そこで、近くにキャラがいてマスクの外の画素は、そのキャラの深度から遠い側に寄せる */
/* まわりの最小と最大 (1 方向)。inMin, inMax: 前の段の最小と最大 (最初の段はどちらも深度)。
   inW, inT (無ければ NULL): 前の段のマスクの最大と、その所のキャラの深度 → wm, ts */
DF_HD DF_INL void dfMinMax1D(const float *inMin, const float *inMax, const float *inW, const float *inT, int W, int H, int rad, bool horizontal,
                            int x, int y, float &mn, float &mx, float &wm, float &ts)
{
	const size_t n0 = (size_t)y * W + x;
	mn = inMin[n0];
	mx = inMax[n0];
	wm = inW ? inW[n0] : 0.f;
	ts = inT ? inT[n0] : -1.f;
	for (int k = -rad; k <= rad; ++k) {
		const int xx = horizontal ? dfClamp(x + k, 0, W - 1) : x;
		const int yy = horizontal ? y : dfClamp(y + k, 0, H - 1);
		const size_t n = (size_t)yy * W + xx;
		mn = dfMin(mn, inMin[n]);
		mx = dfMax(mx, inMax[n]);
		/* まわりにキャラが 2 人いるとき、遠い方のキャラの深度を「近くのキャラ」にしないよう、離れるほど少し割り引く
		   (縦横 2 回で、割り引きは |dx| + |dy| に比例する) */
		const float sc = inW ? inW[n] - DF_STEP_NEAR_PENALTY * fabsf((float)k) : 0.f;
		if (inW && sc > wm) { wm = sc; ts = inT[n]; }
	}
}

/* 深度 d を、まわりの最小 mn と最大 mx のどちらかへ寄せる (段差の大きさに合わせて)。
   ふつうは近い方。w: この画素のマスク、wm, ts: まわりのマスクの最大とそのキャラの深度、focus: ピントの深度。
   境目の 1〜2 画素はなめらかにつなぐ (ギザギザにしない)。ただし両側がピントの深度をはさむときは、
   間の深度にするとピントの合った線が出るので、どちらかに決め切る */
DF_HD DF_INL float dfStepDepth(float d, float mn, float mx, float w, float wm, float ts, float focus)
{
	const float range = mx - mn;
	const float e = dfSmooth(DF_STEP_LO, DF_STEP_HI, range);
	if (e <= 0.f) return d;
	float a;   /* 0: 遠い側 mn、1: 近い側 mx */
	if (wm >= 0.5f && ts >= 0.f && w < 0.95f) {
		/* キャラの縁: マスクで決める。キャラの中はキャラの深度の側、外はその反対の側 */
		/* キャラの縁では中間の深度を作らない (中間だと縁のすぐ外の背景が少ししかぼけず、くっきりしたかけらが残る)。
		   縁のなめらかさは、元の絵の線とマスクのなめらかな境目に任せる */
		const float k = dfSmooth(DF_STEP_MASK - 0.05f, DF_STEP_MASK + 0.05f, w);
		a = (fabsf(mx - ts) < fabsf(mn - ts)) ? k : 1.f - k;
	} else {
		a = dfSmooth(0.35f, 0.65f, (d - mn) / range);
	}
	if ((mn - focus) * (mx - focus) < 0.f) a = a < 0.5f ? 0.f : 1.f;
	return dfMix(d, dfMix(mn, mx, a), e);
}

/* 手直しのレイヤー (Sharp Layer / Blur Layer) の 1 画素のアルファ (0..1)。fx, fy: レイヤーのワールドの画素 */
DF_HD DF_INL float dfFixAt(const DfAuxMap &m, float fx, float fy);

/* Blur Layer: 塗った所 (b) は、画面のいちばん奥 (farD) と同じだけぼかす (奥へだけ寄せる) */
DF_HD DF_INL float dfBlurFix(float d, float b, float farD)
{
	return b > 0.f ? dfMix(d, dfMin(d, farD), b) : d;
}

/* 深度の点のうち、奥から 5% の深度 (近い = 1 なので小さい方から 5%)。ヒストグラムで */
inline float dfFarPercentile(const float *low, size_t n)
{
	if (!low || n == 0) return 0.f;
	int hist[256] = {0};
	for (size_t i = 0; i < n; ++i) hist[dfClamp((int)(dfSat(low[i]) * 255.f + 0.5f), 0, 255)]++;
	const size_t want = n / 20;
	size_t acc = 0;
	for (int k = 0; k < 256; ++k) {
		acc += (size_t)hist[k];
		if (acc > want) return (float)k / 255.f;
	}
	return 0.f;
}

/* キャラのマスク (Subject Layer) の 1 画素 (0..1)。深度と同じく、レイヤー全体を取り出して位置の割合で合わせる */
/* マスクの位置 (fx, fy、マスクのワールドの画素) で、キャラごとのマスクを o[0..2] に入れる。
   Alpha / Luminance は 1 人分 (o[0] だけ)。Multi (RGB) は赤・緑・青を 1・2・3 人目とする */
DF_HD DF_INL void dfSubjectAtF3(const DfAuxMap &m, int channel, float fx, float fy, float o[3])
{
	o[0] = o[1] = o[2] = 0.f;
	if (!m.img.data) return;
	fx = dfClamp(fx, 0.f, (float)(m.img.width - 1));
	fy = dfClamp(fy, 0.f, (float)(m.img.height - 1));
	const int x0 = (int)fx, y0 = (int)fy;
	const float tx = fx - (float)x0, ty = fy - (float)y0;
	for (int k = 0; k < 4; ++k) {
		const DfRgba p = dfRead(m.img, x0 + (k & 1), y0 + (k >> 1));
		const float w = ((k & 1) ? tx : 1.f - tx) * ((k >> 1) ? ty : 1.f - ty);
		if (channel == 3) { o[0] += p.r * w; o[1] += p.g * w; o[2] += p.b * w; }
		else if (channel == 2) o[0] += (0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b) * w;   /* 明るさは premultiplied のまま (透明な所は 0) */
		else o[0] += p.a * w;
	}
	o[0] = dfSat(o[0]); o[1] = dfSat(o[1]); o[2] = dfSat(o[2]);
}

/* キャラ全員を合わせたマスク */
DF_HD DF_INL float dfSubjectAtF(const DfAuxMap &m, int channel, float fx, float fy)
{
	float o[3];
	dfSubjectAtF3(m, channel, fx, fy, o);
	return dfSat(o[0] + o[1] + o[2]);
}

/* 出力の画素 (x, y) のマスク */
DF_HD DF_INL float dfSubjectAt(const DfAuxMap &m, int channel, int x, int y)
{
	return dfSubjectAtF(m, channel, (float)x * m.sx + m.ox, (float)y * m.sy + m.oy);
}

/* 深度の点 (i, j) の位置の、キャラごとのマスク。どちらもレイヤー全体を取り出しているので、レイヤーの中の位置の割合で合わせる */
DF_HD DF_INL void dfSubjectAtLow3(const DfAuxMap &depthMap, const DfAuxMap &m, int channel, int i, int j, float o[3])
{
	const float rx = m.sx / dfMax(depthMap.sx, 1e-6f), ry = m.sy / dfMax(depthMap.sy, 1e-6f);
	dfSubjectAtF3(m, channel, ((float)i + 0.5f) * rx - 0.5f, ((float)j + 0.5f) * ry - 0.5f, o);
}

DF_HD DF_INL float dfFixAt(const DfAuxMap &m, float fx, float fy)
{
	float o[3];
	dfSubjectAtF3(m, 1, fx, fy, o);
	return dfSat(o[0]);
}

/* 出力の画素のキャラごとのマスク o → 全員を合わせたマスク w と、寄せる先の深度 tgt (キャラごとの代表の深度をマスクで重み付け。いなければ負) */
DF_HD DF_INL void dfSubjectWeights(const DofContext *c, const float o[3], float &w, float &tgt)
{
	float ws = 0.f, ts = 0.f;
	for (int s = 0; s < DF_MAX_SUBJECTS; ++s) {
		if (c->subjectDepth[s] < 0.f) continue;
		ws += o[s];
		ts += o[s] * c->subjectDepth[s];
	}
	w = dfSat(o[0] + o[1] + o[2]);
	tgt = ws > 1e-4f ? ts / ws : -1.f;
}

/* マスクの中の深度を、そのキャラの代表の深度へ寄せる (Subject Flatten)。w: マスク、tgt: 寄せる先 */
DF_HD DF_INL float dfSubjectBlendDepth(const DofContext *c, float depth, float w, float tgt)
{
	if (tgt < 0.f || c->p.subjectFlatten <= 0.f) return depth;
	return dfMix(depth, tgt, dfSat(w * c->p.subjectFlatten));
}

/* ピントの深度からの差 → 符号付きのぼけの半径 (出力の画素。手前 = 負) */
DF_HD DF_INL float dfCoc(const DofContext *c, float depth)
{
	const float delta = depth - c->focus;
	const float over = fabsf(delta) - c->p.focusRange;
	if (over <= 0.f) return 0.f;
	/* ピントの幅の外でなめらかに立ち上げる (幅の境目でぼけが急に変わらないように) */
	const float ramp = dfSmooth(0.f, DF_FOCUS_SOFT, over);
	const float r = dfMin(c->maxCoc, over * ramp * c->cocScale * (delta > 0.f ? c->p.nearScale : c->p.farScale));
	return delta > 0.f ? -r : r;
}

/* ------------------------------------------------------------------ */
/* 色 (リニア、premultiplied)                                            */
/* ------------------------------------------------------------------ */

/* 入力の 1 画素 → リニアの色 (premultiplied)。out[0..3] = r, g, b, a */
DF_HD DF_INL void dfLinear(const DofContext *c, const DfImage &src, int x, int y, float *out)
{
	const DfRgba p = dfRead(src, x, y);
	if (c->p.inputLinear) { out[0] = p.r; out[1] = p.g; out[2] = p.b; out[3] = p.a; return; }
	const float a = p.a;
	if (a <= 1e-6f) { out[0] = out[1] = out[2] = out[3] = 0.f; return; }
	const float ia = 1.f / a;
	/* 明るさが 1 を超える値 (32bpc) はそのまま伸ばす */
	auto dec = [&](float v) { v *= ia; return v <= 1.f ? dfDecode(c, v) : powf(v, 2.2f); };
	out[0] = dec(p.r) * a; out[1] = dec(p.g) * a; out[2] = dec(p.b) * a; out[3] = a;
}

/* 2x2 の平均で半分の大きさにする (1 画素)。src は sw x sh、RGBA 4 個ずつ */
DF_HD DF_INL void dfDown(const float *src, int sw, int sh, int x, int y, float *out)
{
	const int x0 = dfMin(2 * x, sw - 1), x1 = dfMin(2 * x + 1, sw - 1);
	const int y0 = dfMin(2 * y, sh - 1), y1 = dfMin(2 * y + 1, sh - 1);
	for (int k = 0; k < 4; ++k)
		out[k] = 0.25f * (src[((size_t)y0 * sw + x0) * 4 + k] + src[((size_t)y0 * sw + x1) * 4 + k] +
		                  src[((size_t)y1 * sw + x0) * 4 + k] + src[((size_t)y1 * sw + x1) * 4 + k]);
}

/* ------------------------------------------------------------------ */
/* ぼけの届く範囲 (区画ごと)                                              */
/* ------------------------------------------------------------------ */

/* 区画 (DF_TILE 四方) の中の |CoC| の最大値 */
DF_HD DF_INL float dfTileMax(const DofContext *c, int tx, int ty)
{
	float m = 0.f;
	const int x0 = tx * DF_TILE, y0 = ty * DF_TILE;
	const int x1 = dfMin(x0 + DF_TILE, c->outW), y1 = dfMin(y0 + DF_TILE, c->outH);
	for (int y = y0; y < y1; ++y)
		for (int x = x0; x < x1; ++x) m = dfMax(m, fabsf(c->coc[(size_t)y * c->outW + x]));
	return m;
}

/* この区画の画素に届きうるぼけの最大の半径: 近くの区画のうち、ぼけがこの区画まで届くものの最大値 */
DF_HD DF_INL float dfTileReach(const DofContext *c, const float *tileMax, int tx, int ty)
{
	const int k = (int)ceilf(c->maxCoc / (float)DF_TILE) + 1;
	float r = tileMax[(size_t)ty * c->tilesX + tx];
	for (int j = dfMax(0, ty - k); j <= dfMin(c->tilesY - 1, ty + k); ++j)
		for (int i = dfMax(0, tx - k); i <= dfMin(c->tilesX - 1, tx + k); ++i) {
			const float m = tileMax[(size_t)j * c->tilesX + i];
			const int d = dfMax(dfMax(i - tx, tx - i), dfMax(j - ty, ty - j));
			/* 区画の間の最短の距離 (画素) より大きなぼけなら届く */
			if (m > (float)((d - 1) * DF_TILE)) r = dfMax(r, m);
		}
	return r;
}

/* ------------------------------------------------------------------ */
/* ぼかし (1 画素)                                                       */
/* ------------------------------------------------------------------ */

/* 多角形の絞りの形: 方向 ang での、外接円に対する縁までの距離の割合 (円なら 1) */
DF_HD DF_INL float dfShapeScale(const DofContext *c, float ang)
{
	if (c->polySides <= 0) return 1.f;
	const float seg = 6.28318531f / (float)c->polySides;
	float a = ang - c->polyRot;
	a = a - seg * floorf(a / seg) - 0.5f * seg;   /* 辺の中央からの角度 */
	return cosf(0.5f * seg) / cosf(a);
}

/* リニアの色を読む (出力の大きさの画像の 1 画素) */
DF_HD DF_INL void dfFetch(const DofContext *c, int level, int x, int y, float *o)
{
	(void)level;
	x = dfClamp(x, 0, c->outW - 1);
	y = dfClamp(y, 0, c->outH - 1);
	const float *p = c->lin[0] + ((size_t)y * c->outW + x) * 4;
	o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; o[3] = p[3];
}

/* リニアの色を縮小画像 level から、出力の画素の位置 (fx, fy、小数) でバイリニアに読む。
   点の間隔が広いときに、間の模様を平均した色を読む (一番近い画素だけだと、ぼけの中にブロックや模様が出る) */
DF_HD DF_INL void dfFetchLevel(const DofContext *c, int level, float fx, float fy, float *o)
{
	if (level <= 0) { dfFetch(c, 0, dfRound(fx), dfRound(fy), o); return; }
	const float s = 1.f / (float)(1 << level);
	const int w = c->lw[level], h = c->lh[level];
	const float u = dfClamp((fx + 0.5f) * s - 0.5f, 0.f, (float)(w - 1));
	const float v = dfClamp((fy + 0.5f) * s - 0.5f, 0.f, (float)(h - 1));
	const int x0 = (int)u, y0 = (int)v;
	const int x1 = dfMin(x0 + 1, w - 1), y1 = dfMin(y0 + 1, h - 1);
	const float tx = u - (float)x0, ty = v - (float)y0;
	const float *img = c->lin[level];
	const float *a = img + ((size_t)y0 * w + x0) * 4, *b = img + ((size_t)y0 * w + x1) * 4;
	const float *d = img + ((size_t)y1 * w + x0) * 4, *e = img + ((size_t)y1 * w + x1) * 4;
	for (int k = 0; k < 4; ++k) o[k] = dfMix(dfMix(a[k], b[k], tx), dfMix(d[k], e[k], tx), ty);
}

/* 明るい所の重み (玉ぼけ)。lum はリニアの明るさ */
DF_HD DF_INL float dfHighlight(const DofContext *c, float lum)
{
	if (c->p.highlights <= 0.f) return 1.f;
	const float t = dfSat((lum - c->p.highlightThreshold) / dfMax(1e-3f, 1.f - c->p.highlightThreshold));
	return 1.f + c->p.highlights * 8.f * t * t + c->p.highlights * dfMax(0.f, lum - 1.f);
}

/*
 * 出力の画素 (x, y) をぼかした色 (リニア、premultiplied) を out[0..3] に入れる。
 *   R: この画素に届きうるぼけの最大の半径 (区画ごとの値)
 * 点は黄金角のらせん (Vogel の円盤) で R の円の中に等しい面積ずつ並べる。
 */
DF_HD DF_INL void dfGather(const DofContext *c, int x, int y, float R, float *out)
{
	const int W = c->outW, H = c->outH;
	const size_t np = (size_t)y * W + x;
	const float cp = c->coc[np];
	const float ap = fabsf(cp);
	float center[4];
	dfFetch(c, 0, x, y, center);
	if (R < 0.5f) { out[0] = center[0]; out[1] = center[1]; out[2] = center[2]; out[3] = center[3]; return; }

	/* 点の数: 小さいぼけは少なく (円の面積に合わせる) */
	const int n = dfClamp((int)ceilf(R * R * 0.8f), 8, c->maxSamples);
	/* 点の間隔が広いときは縮小した画像から読む (細かい模様のちらつきを抑える) */
	const float spacing = R * sqrtf(3.14159265f / (float)n);
	const int level = dfClamp((int)floorf(log2f(dfMax(spacing, 1.f) / 1.5f)), 0, DF_LEVELS - 1);
	const bool pNear = cp < 0.f;

	/* 後ろの層 (ピントと奥)。自分の画素から始める (手前の画素なら、後ろの層には入れない) */
	float bg[4] = {0.f, 0.f, 0.f, 0.f}, bgW = 0.f;
	if (!pNear) {
		const float w = 1.f / dfMax(ap * ap, 0.25f);
		for (int k = 0; k < 4; ++k) bg[k] = center[k] * w;
		bgW = w;
	}
	/* 前の層 (手前のぼけ) と、その覆う割合 */
	float fg[4] = {0.f, 0.f, 0.f, 0.f}, fgW = 0.f, cover = 0.f;
	const float area = R * R / (float)n;   /* 1 点が受け持つ面積 / π */
	if (pNear) {
		const float d = area / dfMax(ap * ap, 0.25f);
		const float hl = dfHighlight(c, (0.2126f * center[0] + 0.7152f * center[1] + 0.0722f * center[2]) / dfMax(center[3], 1e-4f));
		for (int k = 0; k < 4; ++k) fg[k] = center[k] * d * hl;
		fgW = d * hl;
		cover = d;
	}

	for (int i = 0; i < n; ++i) {
		const float r = R * sqrtf(((float)i + 0.5f) / (float)n);
		const float th = (float)i * DF_GOLDEN_ANGLE;
		const float ox = r * cosf(th), oy = r * sinf(th);
		const int qx = dfClamp(x + dfRound(ox), 0, W - 1);
		const int qy = dfClamp(y + dfRound(oy), 0, H - 1);
		const float cq = c->coc[(size_t)qy * W + qx];
		const float aq = fabsf(cq);
		/* q の円が p に届くか: q から p への向き (th + π) での絞りの形 */
		const float shape = dfShapeScale(c, th + 3.14159265f);
		float col[4];
		bool fetched = false;
		if (cq < 0.f) {
			/* 手前の点: 後ろの物の上に広がる */
			const float cov = dfSat(aq * shape - r + 0.5f);
			if (cov <= 0.f) continue;
			dfFetchLevel(c, level, (float)x + ox, (float)y + oy, col);
			fetched = true;
			const float d = cov * area / dfMax(aq * aq, 0.25f);
			const float hl = dfHighlight(c, (0.2126f * col[0] + 0.7152f * col[1] + 0.0722f * col[2]) / dfMax(col[3], 1e-4f));
			for (int k = 0; k < 4; ++k) fg[k] += col[k] * d * hl;
			fgW += d * hl;
			cover += d;
			/* p も手前なら、手前どうしは後ろの層にも入れる (手前の物の内側をなめらかにする) */
			if (!pNear) continue;
		}
		/* 後ろの層: q が p より奥なら、q の広がりを p のぼけまでに抑える (ピントの合った物に背景がにじまない) */
		float reach = aq * shape;
		if (!pNear && cq > cp) reach = dfMin(reach, ap * shape);
		const float cov = dfSat(reach - r + 0.5f);
		if (cov <= 0.f) continue;
		if (!fetched) dfFetchLevel(c, level, (float)x + ox, (float)y + oy, col);
		const float w = cov / dfMax(aq * aq, 0.25f);
		const float hl = dfHighlight(c, (0.2126f * col[0] + 0.7152f * col[1] + 0.0722f * col[2]) / dfMax(col[3], 1e-4f));
		for (int k = 0; k < 4; ++k) bg[k] += col[k] * w * hl;
		bgW += w * hl;
	}

	float back[4];
	if (bgW > 1e-8f) for (int k = 0; k < 4; ++k) back[k] = bg[k] / bgW;
	else for (int k = 0; k < 4; ++k) back[k] = center[k];
	/* キャラの上には手前のぼけを重ねない (キャラはいちばん上のセルとして扱う) */
	const float keep = c->subject ? dfSat(c->subject[np] * c->p.subjectAmount) : 0.f;
	const float alpha = dfSat(cover) * (1.f - keep);
	if (fgW > 1e-8f && alpha > 0.f) {
		for (int k = 0; k < 4; ++k) out[k] = dfMix(back[k], fg[k] / fgW, alpha);
	} else {
		for (int k = 0; k < 4; ++k) out[k] = back[k];
	}
}

/* ------------------------------------------------------------------ */
/* 仕上げ (1 画素)                                                       */
/* ------------------------------------------------------------------ */

/* 出力の色 (premultiplied、表示の値に戻したもの) を作る。lin: ぼかした色 (リニア、premultiplied)。
   src: 元の画素 (premultiplied、Mix 用) */
DF_HD DF_INL DfRgba dfFinish(const DofContext *c, const float *lin, const DfRgba &src, float depth, float coc)
{
	DfRgba o;
	const int mode = c->p.outputMode;
	if (mode == OUTPUT_DEPTH) {
		o.a = 1.f; o.r = o.g = o.b = depth;
		if (c->p.inputLinear) { o.r = o.g = o.b = dfDecode(c, depth); }
		return o;
	}
	if (mode == OUTPUT_BLUR_MAP) {
		/* 手前 = 青、奥 = 橙、ピント = 黒。明るさはぼけの大きさ */
		const float t = dfSat(fabsf(coc) / dfMax(1.f, 0.5f * c->maxCoc));
		const float v = sqrtf(t);
		o.a = 1.f;
		if (coc < 0.f) { o.r = 0.25f * v; o.g = 0.55f * v; o.b = v; }
		else           { o.r = v; o.g = 0.55f * v; o.b = 0.2f * v; }
		if (c->p.inputLinear) { o.r = dfDecode(c, o.r); o.g = dfDecode(c, o.g); o.b = dfDecode(c, o.b); }
		return o;
	}
	o.a = lin[3];
	if (c->p.inputLinear) {
		o.r = lin[0]; o.g = lin[1]; o.b = lin[2];
	} else if (o.a > 1e-6f) {
		const float ia = 1.f / o.a;
		auto enc = [&](float v) { v *= ia; return v <= 1.f ? dfEncode(c, v) : powf(v, 1.f / 2.2f); };
		o.r = enc(lin[0]) * o.a; o.g = enc(lin[1]) * o.a; o.b = enc(lin[2]) * o.a;
	} else {
		o.r = o.g = o.b = 0.f;
	}
	if (mode == OUTPUT_FOCUS_CHECK) {
		/* ピントの合っている所 (ぼけが 1 画素未満) を緑に寄せる */
		const float inFocus = 1.f - dfSmooth(0.5f, 1.5f, fabsf(coc));
		const float g = 0.55f * inFocus;
		o.r = dfMix(o.r, 0.15f * o.a, g); o.g = dfMix(o.g, 0.95f * o.a, g); o.b = dfMix(o.b, 0.3f * o.a, g);
	}
	const float m = dfSat(c->p.mix);
	if (m < 1.f) {
		o.a = dfMix(src.a, o.a, m); o.r = dfMix(src.r, o.r, m); o.g = dfMix(src.g, o.g, m); o.b = dfMix(src.b, o.b, m);
	}
	return o;
}

inline float dfFocusFromPoint(const float *low, int dw, int dh, float sx, float sy, float lx, float ly);

/* キャラ s の代表の深度: そのキャラのマスクが半分以上の深度の点の、深度の中央値 (1024 段の度数で)。点が少なければ -1。
   mask: 深度の点ごとに 3 個ずつ (キャラ 1・2・3) */
inline float dfSubjectMedian(const float *low, const float *mask, int dw, int dh, int s)
{
	static const int BINS = 1024;
	int hist[BINS] = {0};
	int n = 0;
	for (int k = 0; k < dw * dh; ++k) {
		if (mask[(size_t)k * 3 + s] < 0.5f) continue;
		hist[dfClamp((int)(dfSat(low[k]) * (BINS - 1) + 0.5f), 0, BINS - 1)]++;
		++n;
	}
	if (n < 16) return -1.f;
	int acc = 0;
	for (int b = 0; b < BINS; ++b) {
		acc += hist[b];
		if (acc * 2 >= n) return (float)b / (float)(BINS - 1);
	}
	return -1.f;
}

/* Focus Point (レイヤー座標) にいるキャラ (0..2)。いなければ -1 (深度の点で見る) */
inline int dfPointSubject(const float *mask, int dw, int dh, float sx, float sy, float lx, float ly)
{
	const int i = dfClamp(dfRound((lx + 0.5f) * sx - 0.5f), 0, dw - 1);
	const int j = dfClamp(dfRound((ly + 0.5f) * sy - 0.5f), 0, dh - 1);
	const float *m = mask + ((size_t)j * dw + i) * 3;
	int best = -1;
	float bv = 0.5f;
	for (int s = 0; s < DF_MAX_SUBJECTS; ++s)
		if (m[s] >= bv) { bv = m[s]; best = s; }
	return best;
}

/* キャラごとの代表の深度を求め、Focus Point がキャラの上ならそのキャラの深度をピントにする (CPU と GPU で共通、ホストで)。
   low: 深度の点、mask: キャラごとのマスク (3 個ずつ。NULL ならキャラなし)。戻り値: ピントの深度 */
inline float dfResolveFocus(DofContext &c, const float *low, const float *mask, int dw, int dh, float sx, float sy)
{
	for (int s = 0; s < DF_MAX_SUBJECTS; ++s) c.subjectDepth[s] = mask ? dfSubjectMedian(low, mask, dw, dh, s) : -1.f;
	if (c.p.focusMode != FOCUS_POINT) return c.p.focusDepth;
	/* 点 (lx, ly) のピントの深度: キャラの上ならそのキャラの代表の深度 (キャラ全体にピントが合う)、そうでなければ点のまわりの深度 */
	auto at = [&](float lx, float ly) {
		if (mask) {
			const int s = dfPointSubject(mask, dw, dh, sx, sy, lx, ly);
			if (s >= 0 && c.subjectDepth[s] >= 0.f) return c.subjectDepth[s];
		}
		return dfFocusFromPoint(low, dw, dh, sx, sy, lx, ly);
	};
	if (c.p.pullCount > 0) {
		float sum = 0.f, ws = 0.f;
		for (int i = 0; i < c.p.pullCount && i < DF_PULL_MAX; ++i) {
			const DfFocusSample &s = c.p.pull[i];
			float f = at(s.ax, s.ay);
			if (s.t > 0.f && (s.ax != s.bx || s.ay != s.by)) f = dfMix(f, at(s.bx, s.by), dfSat(s.t));
			sum += f * s.w;
			ws += s.w;
		}
		if (ws > 0.f) return sum / ws;
	}
	return at(c.p.focusX, c.p.focusY);
}

/* ピントの深度を求める: Focus Point のまわりの深度の中央値。
   low: 深度 (dw x dh、レイヤー全体)。sx, sy: レイヤー座標 → 深度の点の倍率 */
inline float dfFocusFromPoint(const float *low, int dw, int dh, float sx, float sy, float lx, float ly)
{
	const int cx = dfClamp(dfRound((lx + 0.5f) * sx - 0.5f), 0, dw - 1);
	const int cy = dfClamp(dfRound((ly + 0.5f) * sy - 0.5f), 0, dh - 1);
	const int r = dfMax(1, dfRound(DF_FOCUS_WINDOW * (float)dh));
	float v[(2 * 16 + 1) * (2 * 16 + 1)];
	int m = 0;
	const int rr = dfMin(r, 16);
	for (int j = cy - rr; j <= cy + rr; ++j)
		for (int i = cx - rr; i <= cx + rr; ++i) {
			if ((i - cx) * (i - cx) + (j - cy) * (j - cy) > rr * rr) continue;
			v[m++] = low[(size_t)dfClamp(j, 0, dh - 1) * dw + dfClamp(i, 0, dw - 1)];
		}
	for (int a = 1; a < m; ++a) {
		const float x = v[a];
		int b = a - 1;
		while (b >= 0 && v[b] > x) { v[b + 1] = v[b]; --b; }
		v[b + 1] = x;
	}
	return m > 0 ? v[m / 2] : 0.5f;
}

/*
 * DofGPU.cu
 * GPU 版 (CUDA)。CPU 版 (DofAE.cpp の RenderFrameCPU) と同じ手順を、DofCore.h の同じ関数で行う。
 * AE が GPU で描くとき (Mercury GPU アクセラレーション = CUDA) に PF_Cmd_SMART_RENDER_GPU から呼ばれる。
 *
 * 手順:
 *   入力 → リニアの色 (と 1/2、1/4 の縮小)
 *   深度の点 (前後のフレームでならす) → ピントの深度 (Focus Point のまわりの中央値) → 出力の大きさへ引き伸ばし
 *   → 手前の輪郭を広げる → ぼけの大きさ → 区画ごとの届く範囲 → ぼかし → 出力
 */
#include "DofGPU.h"
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <cstdio>
#include <cstring>

/* ------------------------------------------------------------------ */
/* GPU のメモリ                                                        */
/* ------------------------------------------------------------------ */

/* 作業領域はフレームごとに確保せず使い回す (cudaMalloc は遅い)。同時に描くフレームの数だけ持つ */
struct DevBlock {
	void  *p;
	size_t size;
	bool   inUse;
};
static std::mutex            g_memMtx;
static std::vector<DevBlock> g_blocks;
static float                *g_lut = NULL;   /* ガンマの表 (dec, enc を続けて置く) */

static void *AcquireBlock(size_t size)
{
	std::lock_guard<std::mutex> lk(g_memMtx);
	for (DevBlock &b : g_blocks)
		if (!b.inUse && b.size >= size) { b.inUse = true; return b.p; }
	/* 足りなければ、空いている小さいものを捨てて大きく取り直す */
	for (size_t i = 0; i < g_blocks.size(); ++i)
		if (!g_blocks[i].inUse) { cudaFree(g_blocks[i].p); g_blocks.erase(g_blocks.begin() + i); break; }
	void *p = NULL;
	if (cudaMalloc(&p, size) != cudaSuccess) return NULL;
	DevBlock b = {p, size, true};
	g_blocks.push_back(b);
	return p;
}

static void ReleaseBlock(void *p)
{
	std::lock_guard<std::mutex> lk(g_memMtx);
	for (DevBlock &b : g_blocks)
		if (b.p == p) { b.inUse = false; return; }
}

static const float *DeviceLut()
{
	std::lock_guard<std::mutex> lk(g_memMtx);
	if (g_lut) return g_lut;
	std::vector<float> host(DF_LUT_SIZE * 2);
	dfBuildLut(host.data(), host.data() + DF_LUT_SIZE);
	if (cudaMalloc(&g_lut, sizeof(float) * host.size()) != cudaSuccess) { g_lut = NULL; return NULL; }
	cudaMemcpy(g_lut, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice);
	return g_lut;
}

void DofGPU_Release()
{
	std::lock_guard<std::mutex> lk(g_memMtx);
	for (DevBlock &b : g_blocks) cudaFree(b.p);
	g_blocks.clear();
	if (g_lut) { cudaFree(g_lut); g_lut = NULL; }
}

/* ------------------------------------------------------------------ */
/* カーネル                                                             */
/* ------------------------------------------------------------------ */

#define DF_XY(W, H)                                         \
	const int x = blockIdx.x * blockDim.x + threadIdx.x;    \
	const int y = blockIdx.y * blockDim.y + threadIdx.y;    \
	if (x >= (W) || y >= (H)) return;

static dim3 Grid2(int W, int H) { return dim3((W + 15) / 16, (H + 15) / 16, 1); }
static const dim3 BLOCK2(16, 16, 1);

/* 前後のフレームの深度 (カーネルへ値で渡す) */
struct NbSet {
	DfImage img[4];
	int     n;
};

__global__ void kLinear(DofContext c, DfImage src, float *lin)
{
	DF_XY(c.outW, c.outH);
	dfLinear(&c, src, x, y, lin + ((size_t)y * c.outW + x) * 4);
}

__global__ void kDown(const float *src, int sw, int sh, float *dst, int dw, int dh)
{
	DF_XY(dw, dh);
	dfDown(src, sw, sh, x, y, dst + ((size_t)y * dw + x) * 4);
}

__global__ void kLowDepth(DfImage img, NbSet nb, int depthNear, bool dec, float *low)
{
	DF_XY(img.width, img.height);
	float d = dfLowDepth(img, x, y, depthNear, dec);
	if (nb.n > 0) d = dfStabilizeDepth(d, nb.img, nb.n, x, y, depthNear, dec);
	low[(size_t)y * img.width + x] = d;
}

__global__ void kUpsample(DofContext c, const float *low, int dw, int dh, DfAuxMap map, DfImage src, float *depth)
{
	DF_XY(c.outW, c.outH);
	depth[(size_t)y * c.outW + x] = dfUpsampleDepth(&c, low, dw, dh, map, src, x, y);
}

__global__ void kExpand(const float *in, float *out, int W, int H, int rad, bool horizontal)
{
	DF_XY(W, H);
	out[(size_t)y * W + x] = dfExpand1D(in, W, H, rad, horizontal, x, y);
}

__global__ void kCoc(DofContext c, const float *depth, float *coc)
{
	DF_XY(c.outW, c.outH);
	const size_t n = (size_t)y * c.outW + x;
	coc[n] = dfCoc(&c, depth[n]);
}

__global__ void kTileMax(DofContext c, float *tileMax)
{
	DF_XY(c.tilesX, c.tilesY);
	tileMax[(size_t)y * c.tilesX + x] = dfTileMax(&c, x, y);
}

__global__ void kTileReach(DofContext c, const float *tileMax, float *tileR)
{
	DF_XY(c.tilesX, c.tilesY);
	tileR[(size_t)y * c.tilesX + x] = dfTileReach(&c, tileMax, x, y);
}

__device__ void writeBGRA(void *dst, int rowbytes, int x, int y, const DfRgba &o)
{
	float *p = (float *)((char *)dst + (size_t)y * rowbytes) + (size_t)x * 4;
	p[0] = o.b; p[1] = o.g; p[2] = o.r; p[3] = o.a;
}

__global__ void kGather(DofContext c, DfImage src, void *dst, int dstRowbytes)
{
	DF_XY(c.outW, c.outH);
	float lin[4];
	float depth = 0.f, coc = 0.f;
	if (c.coc) {
		const size_t n = (size_t)y * c.outW + x;
		depth = c.depth[n];
		coc = c.coc[n];
		const float R = c.tileR[(size_t)(y / DF_TILE) * c.tilesX + (x / DF_TILE)];
		if (c.p.outputMode == OUTPUT_FINAL || c.p.outputMode == OUTPUT_FOCUS_CHECK) dfGather(&c, x, y, R, lin);
		else dfFetch(&c, 0, x, y, lin);
	} else {
		dfFetch(&c, 0, x, y, lin);
	}
	writeBGRA(dst, dstRowbytes, x, y, dfFinish(&c, lin, dfRead(src, x, y), depth, coc));
}

/* ------------------------------------------------------------------ */
/* 全体の手順                                                           */
/* ------------------------------------------------------------------ */

/* 作業領域の割り付け (1 つの塊から切り出す) */
struct Carver {
	unsigned char *base;
	size_t used;
	float *take(size_t count)
	{
		float *p = (float *)(base + used);
		used += (count * sizeof(float) + 255) & ~(size_t)255;
		return p;
	}
};

#define DF_CHECK(what)                                                                       \
	do {                                                                                     \
		cudaError_t e_ = cudaGetLastError();                                                 \
		if (e_ != cudaSuccess) {                                                             \
			if (err) snprintf(err, errLen, "%s: %s", what, cudaGetErrorString(e_));          \
			if (block) ReleaseBlock(block);                                                  \
			return (int)e_;                                                                  \
		}                                                                                    \
	} while (0)

int DofGPU_Render(DfGpuFrame &f, char *err, int errLen)
{
	DofContext &c = f.ctx;
	const int W = c.outW, H = c.outH;
	const size_t NP = (size_t)W * H;
	void *block = NULL;

	c.lutDec = DeviceLut();
	if (!c.lutDec) { if (err) snprintf(err, errLen, "LUT alloc failed"); return -1; }
	c.lutEnc = c.lutDec + DF_LUT_SIZE;
	c.depth = c.coc = c.tileR = NULL;
	dfSetupFrame(c, c.p.focusDepth);

	const bool hasDepth = f.depthMap.img.data != NULL;
	const int dw = f.depthMap.img.width, dh = f.depthMap.img.height;

	/* 必要な量を数えてから 1 つの塊で取る */
	size_t floats = 0;
	auto count = [&](size_t n) { floats += n + 64; };
	for (int k = 0; k < DF_LEVELS; ++k) count((size_t)c.lw[k] * c.lh[k] * 4);
	if (hasDepth) {
		count((size_t)dw * dh); count(NP); count(NP); count(NP);
		count((size_t)c.tilesX * c.tilesY); count((size_t)c.tilesX * c.tilesY);
	}
	block = AcquireBlock(floats * sizeof(float));
	if (!block) { if (err) snprintf(err, errLen, "work alloc failed (%zu MB)", floats * 4 / 1048576); return -2; }
	Carver cv = {(unsigned char *)block, 0};

	/* 1. リニアの色と縮小 */
	float *lin[DF_LEVELS];
	for (int k = 0; k < DF_LEVELS; ++k) lin[k] = cv.take((size_t)c.lw[k] * c.lh[k] * 4);
	kLinear<<<Grid2(W, H), BLOCK2>>>(c, f.src, lin[0]);
	for (int k = 1; k < DF_LEVELS; ++k)
		kDown<<<Grid2(c.lw[k], c.lh[k]), BLOCK2>>>(lin[k - 1], c.lw[k - 1], c.lh[k - 1], lin[k], c.lw[k], c.lh[k]);
	DF_CHECK("linear");
	for (int k = 0; k < DF_LEVELS; ++k) c.lin[k] = lin[k];
	f.focusUsed = c.p.focusDepth;

	/* 2. 深度 → ピント → ぼけの大きさ */
	if (hasDepth) {
		float *low = cv.take((size_t)dw * dh);
		NbSet nb;
		std::memset(&nb, 0, sizeof(nb));
		for (int k = 0; k < f.depthNbCount && k < 4; ++k) nb.img[nb.n++] = f.depthNb[k];
		kLowDepth<<<Grid2(dw, dh), BLOCK2>>>(f.depthMap.img, nb, c.p.depthNear, c.p.passEncoding == 2, low);
		DF_CHECK("depth");
		float focus = c.p.focusDepth;
		if (c.p.focusMode == FOCUS_POINT) {
			/* ピントの深度は深度の点 (小さい) をホストへ写して求める */
			std::vector<float> host((size_t)dw * dh);
			cudaMemcpy(host.data(), low, sizeof(float) * host.size(), cudaMemcpyDeviceToHost);
			focus = dfFocusFromPoint(host.data(), dw, dh, f.depthMap.sx, f.depthMap.sy, c.p.focusX, c.p.focusY);
		}
		dfSetupFrame(c, focus);
		f.focusUsed = focus;

		float *depth = cv.take(NP), *tmp = cv.take(NP), *coc = cv.take(NP);
		kUpsample<<<Grid2(W, H), BLOCK2>>>(c, low, dw, dh, f.depthMap, f.src, depth);
		const int rad = dfRound(c.expand);
		if (rad > 0) {
			kExpand<<<Grid2(W, H), BLOCK2>>>(depth, tmp, W, H, rad, true);
			kExpand<<<Grid2(W, H), BLOCK2>>>(tmp, coc, W, H, rad, false);   /* coc を作業に使う */
			cudaMemcpy(depth, coc, sizeof(float) * NP, cudaMemcpyDeviceToDevice);
		}
		kCoc<<<Grid2(W, H), BLOCK2>>>(c, depth, coc);
		DF_CHECK("coc");
		c.depth = depth;
		c.coc = coc;

		float *tileMax = cv.take((size_t)c.tilesX * c.tilesY), *tileR = cv.take((size_t)c.tilesX * c.tilesY);
		kTileMax<<<Grid2(c.tilesX, c.tilesY), BLOCK2>>>(c, tileMax);
		kTileReach<<<Grid2(c.tilesX, c.tilesY), BLOCK2>>>(c, tileMax, tileR);
		DF_CHECK("tiles");
		c.tileR = tileR;
	}

	/* 3. ぼかしと出力 */
	kGather<<<Grid2(W, H), BLOCK2>>>(c, f.src, f.dst, f.dstRowbytes);
	DF_CHECK("gather");
	cudaError_t e = cudaDeviceSynchronize();
	ReleaseBlock(block);
	if (e != cudaSuccess) {
		if (err) snprintf(err, errLen, "sync: %s", cudaGetErrorString(e));
		return (int)e;
	}
	return 0;
}

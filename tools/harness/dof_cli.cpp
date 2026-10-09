/*
 * dof_cli.cpp - プラグインのぼかしのコード (DofAE.cpp) を AE なしで実行する検証用ツール
 *
 * DofAE.cpp をそのまま取り込み、CPU 版 (RenderFrameCPU) か GPU 版 (DofGPU_Render、gpu=1) で描く。入出力は生の float32 RGBA。
 *   dof_cli src.f32 depth.f32 out.f32 W H dw=.. dh=.. key=value ...
 *   キー: fx fy (Focus Point、レイヤー px)、fmode (1 Point / 2 Depth)、fdepth、frange、
 *         blur (Blur Size、1080p の px)、near、far、expand、shape (1 円 / 2 六角形 / 3 八角形)、rot (度)、
 *         hl (Highlights)、hlth (Highlight Threshold)、quality (1..3)、snap (0/1)、mode (1..4)、mix (0..1)、
 *         gpu=1 (GPU 版で描く)、nb=ファイル (前後のフレームの深度、4 個まで)、repeat=N (時間を測る回数)、
 *         subject=ファイル (キャラのマスク、W x H の 1 チャンネル 0..1)、keep (Keep Subject Sharp 0..1)
 * src.f32 は sRGB の RGBA (非乗算、alpha 込み)、depth.f32 は 1 チャンネル (近い = 1)。
 * 標準出力に、使ったピントの深度と処理時間を書く。
 */
#include "../../plugin/src/DofAE.cpp"
#ifdef DOF_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

static bool readFile(const char *path, std::vector<float> &buf, size_t count)
{
	FILE *f = std::fopen(path, "rb");
	if (!f) return false;
	buf.resize(count);
	size_t n = std::fread(buf.data(), sizeof(float), count, f);
	std::fclose(f);
	return n == count;
}

int main(int argc, char **argv)
{
	if (argc < 6) {
		std::fprintf(stderr, "usage: dof_cli src.f32 depth.f32 out.f32 W H [key=value ...]\n");
		return 2;
	}
	const int W = std::atoi(argv[4]), H = std::atoi(argv[5]);
	int dw = W, dh = H, useGpu = 0, repeat = 1;
	DofParams p;
	std::memset(&p, 0, sizeof(p));
	p.focusMode = FOCUS_POINT;
	p.focusX = W * 0.5f; p.focusY = H * 0.45f;
	p.focusDepth = DEF_FOCUS_DEPTH; p.focusRange = DEF_FOCUS_RANGE;
	p.blurSize = DEF_BLUR_SIZE; p.nearScale = 1.f; p.farScale = 1.f; p.edgeExpand = DEF_EDGE_EXPAND;
	p.shape = SHAPE_CIRCLE; p.shapeRotation = 0.f; p.highlights = DEF_HIGHLIGHTS; p.highlightThreshold = DEF_HIGHLIGHT_THR;
	p.quality = QUALITY_NORMAL; p.depthNear = 1; p.depthSnap = true; p.depthStabilize = 0; p.passEncoding = 1;
	p.inputLinear = false; p.outputMode = OUTPUT_FINAL; p.mix = 1.f;
	p.subjectChannel = 1; p.subjectAmount = 1.f;
	std::vector<std::string> nbFiles;
	std::string subjectFile;
	for (int i = 6; i < argc; ++i) {
		std::string a = argv[i];
		size_t eq = a.find('=');
		if (eq == std::string::npos) continue;
		std::string k = a.substr(0, eq), v = a.substr(eq + 1);
		const float f = (float)std::atof(v.c_str());
		if (k == "dw") dw = std::atoi(v.c_str());
		else if (k == "dh") dh = std::atoi(v.c_str());
		else if (k == "gpu") useGpu = std::atoi(v.c_str());
		else if (k == "repeat") repeat = std::max(1, std::atoi(v.c_str()));
		else if (k == "nb") nbFiles.push_back(v);
		else if (k == "subject") subjectFile = v;
		else if (k == "keep") p.subjectAmount = f;
		else if (k == "fx") p.focusX = f;
		else if (k == "fy") p.focusY = f;
		else if (k == "fmode") p.focusMode = (int)f;
		else if (k == "fdepth") p.focusDepth = f;
		else if (k == "frange") p.focusRange = f;
		else if (k == "blur") p.blurSize = f;
		else if (k == "near") p.nearScale = f;
		else if (k == "far") p.farScale = f;
		else if (k == "expand") p.edgeExpand = f;
		else if (k == "shape") p.shape = (int)f;
		else if (k == "rot") p.shapeRotation = f;
		else if (k == "hl") p.highlights = f;
		else if (k == "hlth") p.highlightThreshold = f;
		else if (k == "quality") p.quality = (int)f;
		else if (k == "snap") p.depthSnap = f != 0.f;
		else if (k == "mode") p.outputMode = (int)f;
		else if (k == "mix") p.mix = f;
	}

	std::vector<float> src, depth;
	if (!readFile(argv[1], src, (size_t)W * H * 4)) { std::fprintf(stderr, "src read failed\n"); return 1; }
	const bool hasDepth = std::string(argv[2]) != "-";
	if (hasDepth && !readFile(argv[2], depth, (size_t)dw * dh)) { std::fprintf(stderr, "depth read failed\n"); return 1; }
	/* キャラのマスクは 1 チャンネル → アルファに入れる (Subject Channel = Alpha と同じ) */
	std::vector<float> subjectGray, subjectArgb;
	if (!subjectFile.empty()) {
		if (!readFile(subjectFile.c_str(), subjectGray, (size_t)W * H)) { std::fprintf(stderr, "subject read failed\n"); return 1; }
		subjectArgb.resize((size_t)W * H * 4);
		for (size_t n = 0; n < (size_t)W * H; ++n) {
			const float m = subjectGray[n];
			subjectArgb[n * 4 + 0] = m; subjectArgb[n * 4 + 1] = m; subjectArgb[n * 4 + 2] = m; subjectArgb[n * 4 + 3] = m;
		}
	}
	std::vector<std::vector<float>> nbData;
	for (const std::string &nf : nbFiles) {
		nbData.emplace_back();
		if (!readFile(nf.c_str(), nbData.back(), (size_t)dw * dh)) { std::fprintf(stderr, "nb read failed\n"); return 1; }
	}

	/* 入力は premultiplied の ARGB float にする (AE の 32bpc と同じ形) */
	std::vector<float> argb((size_t)W * H * 4);
	for (size_t n = 0; n < (size_t)W * H; ++n) {
		const float a = src[n * 4 + 3];
		argb[n * 4 + 0] = a;
		argb[n * 4 + 1] = src[n * 4 + 0] * a;
		argb[n * 4 + 2] = src[n * 4 + 1] * a;
		argb[n * 4 + 3] = src[n * 4 + 2] * a;
	}

	DofContext ctx;
	std::memset(&ctx, 0, sizeof(ctx));
	ctx.p = p;
	ctx.outW = W; ctx.outH = H;
	ctx.originX = 0.f; ctx.originY = 0.f;
	ctx.layerW = (float)W; ctx.layerH = (float)H;

	std::vector<DfRgba> out;
	float focus = -1.f;
	double ms = 0.0;
	if (!useGpu) {
		DfImage srcImg = {(const unsigned char *)argb.data(), W, H, W * 16, DF_FMT_ARGB32F};
		DfImage depthImg = {hasDepth ? (const unsigned char *)depth.data() : NULL, dw, dh, dw * 4, DF_FMT_GRAY32F};
		DfAuxMap map = dfMakeAuxMap(depthImg, (float)dw, (float)dh, 0.f, 0.f, (float)W, (float)H);
		DfImage subjImg = {subjectArgb.empty() ? NULL : (const unsigned char *)subjectArgb.data(), W, H, W * 16, DF_FMT_ARGB32F};
		DfAuxMap smap = dfMakeAuxMap(subjImg, (float)W, (float)H, 0.f, 0.f, (float)W, (float)H);
		DfImage nb[4];
		int nbN = 0;
		for (auto &d : nbData) if (nbN < 4) nb[nbN++] = {(const unsigned char *)d.data(), dw, dh, dw * 4, DF_FMT_GRAY32F};
		for (int r = 0; r < repeat; ++r) {
			DofContext c = ctx;
			const auto t0 = std::chrono::steady_clock::now();
			focus = RenderFrameCPU(c, srcImg, map, smap, nb, nbN, out);
			ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		}
	} else {
#ifdef DOF_HAS_CUDA
		/* GPU のワールドと同じ BGRA float にして GPU へ */
		std::vector<float> bgra((size_t)W * H * 4);
		for (size_t n = 0; n < (size_t)W * H; ++n) {
			bgra[n * 4 + 0] = argb[n * 4 + 3]; bgra[n * 4 + 1] = argb[n * 4 + 2];
			bgra[n * 4 + 2] = argb[n * 4 + 1]; bgra[n * 4 + 3] = argb[n * 4 + 0];
		}
		float *dSrc = NULL, *dDst = NULL, *dDepth = NULL, *dSubj = NULL;
		if (!subjectArgb.empty()) {
			/* ARGB → BGRA (どれも同じ値なので並びはそのまま) */
			cudaMalloc(&dSubj, subjectArgb.size() * 4);
			cudaMemcpy(dSubj, subjectArgb.data(), subjectArgb.size() * 4, cudaMemcpyHostToDevice);
		}
		std::vector<float *> dNb;
		cudaMalloc(&dSrc, bgra.size() * 4);
		cudaMalloc(&dDst, bgra.size() * 4);
		cudaMemcpy(dSrc, bgra.data(), bgra.size() * 4, cudaMemcpyHostToDevice);
		if (hasDepth) {
			cudaMalloc(&dDepth, depth.size() * 4);
			cudaMemcpy(dDepth, depth.data(), depth.size() * 4, cudaMemcpyHostToDevice);
		}
		for (auto &d : nbData) {
			float *m = NULL;
			cudaMalloc(&m, d.size() * 4);
			cudaMemcpy(m, d.data(), d.size() * 4, cudaMemcpyHostToDevice);
			dNb.push_back(m);
		}
		for (int r = 0; r < repeat; ++r) {
			DfGpuFrame f;
			std::memset(&f, 0, sizeof(f));
			f.ctx = ctx;
			DfImage depthImg = {hasDepth ? (const unsigned char *)dDepth : NULL, dw, dh, dw * 4, DF_FMT_GRAY32F};
			f.depthMap = dfMakeAuxMap(depthImg, (float)dw, (float)dh, 0.f, 0.f, (float)W, (float)H);
			DfImage subjImg = {dSubj ? (const unsigned char *)dSubj : NULL, W, H, W * 16, DF_FMT_BGRA32F};
			f.subjectMap = dfMakeAuxMap(subjImg, (float)W, (float)H, 0.f, 0.f, (float)W, (float)H);
			f.src = {(const unsigned char *)dSrc, W, H, W * 16, DF_FMT_BGRA32F};
			f.dst = dDst;
			f.dstRowbytes = W * 16;
			for (size_t k = 0; k < dNb.size() && k < 4; ++k) f.depthNb[f.depthNbCount++] = {(const unsigned char *)dNb[k], dw, dh, dw * 4, DF_FMT_GRAY32F};
			char msg[256] = {0};
			const auto t0 = std::chrono::steady_clock::now();
			if (DofGPU_Render(f, msg, sizeof(msg)) != 0) { std::fprintf(stderr, "gpu: %s\n", msg); return 1; }
			ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			focus = f.focusUsed;
		}
		cudaMemcpy(bgra.data(), dDst, bgra.size() * 4, cudaMemcpyDeviceToHost);
		out.resize((size_t)W * H);
		for (size_t n = 0; n < (size_t)W * H; ++n) {
			out[n].b = bgra[n * 4 + 0]; out[n].g = bgra[n * 4 + 1]; out[n].r = bgra[n * 4 + 2]; out[n].a = bgra[n * 4 + 3];
		}
#else
		std::fprintf(stderr, "built without CUDA\n");
		return 1;
#endif
	}

	/* 出力は非乗算の RGBA */
	std::vector<float> rgba((size_t)W * H * 4);
	for (size_t n = 0; n < (size_t)W * H; ++n) {
		const float a = out[n].a, ia = a > 1e-6f ? 1.f / a : 0.f;
		rgba[n * 4 + 0] = out[n].r * ia; rgba[n * 4 + 1] = out[n].g * ia; rgba[n * 4 + 2] = out[n].b * ia; rgba[n * 4 + 3] = a;
	}
	FILE *f = std::fopen(argv[3], "wb");
	if (!f) { std::fprintf(stderr, "out write failed\n"); return 1; }
	std::fwrite(rgba.data(), sizeof(float), rgba.size(), f);
	std::fclose(f);
	std::printf("focus=%.4f ms=%.2f\n", focus, ms);
	return 0;
}

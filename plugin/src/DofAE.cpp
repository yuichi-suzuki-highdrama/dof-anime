/*
 * DofAE.cpp
 * DoF Anime - 推定した深度から被写界深度を付ける SmartFX エフェクト (32bpc / MFR / CUDA 対応)
 *
 * 作りは Relight Anime (RelightAE.cpp) と同じ:
 *   深度レイヤー (自動で作って結線) を全体で取り出し、前後のフレームの深度と合わせて時間方向にならし、
 *   素材の線画に合わせて出力の大きさへ引き伸ばしてから使う。
 * ぼかしの計算は DofCore.h (CPU と GPU で共通)。
 */

#include "DofAE.h"
#include "DofAuto.h"
#include "DofGPU.h"
#include "AE_EffectGPUSuites.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <string>
#include <functional>
#include <memory>
#include <condition_variable>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static_assert(DOF_OUTFLAGS == (PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_CUSTOM_UI | PF_OutFlag_SEND_UPDATE_PARAMS_UI),
	"DofAE_Flags.h の DOF_OUTFLAGS を GlobalSetup のフラグと一致させること");
static_assert(DOF_OUTFLAGS2 == (PF_OutFlag2_FLOAT_COLOR_AWARE |
                                PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                PF_OutFlag2_SUPPORTS_GPU_RENDER_F32),
	"DofAE_Flags.h の DOF_OUTFLAGS2 を GlobalSetup のフラグと一致させること");

/* 既定値 */
static const float DEF_FOCUS_X         = 50.f;    /* レイヤーの幅に対する % */
static const float DEF_FOCUS_Y         = 45.f;
static const float DEF_FOCUS_DEPTH     = 0.7f;
static const float DEF_FOCUS_RANGE     = 0.06f;
static const float DEF_FOCUS_PULL      = 0.8f;    /* ピント送りにかける時間 (秒) */
static const float DEF_BLUR_SIZE       = 30.f;
static const float DEF_EDGE_EXPAND     = 2.f;
static const float DEF_HIGHLIGHTS      = 1.f;
static const float DEF_HIGHLIGHT_THR   = 0.75f;

/* ------------------------------------------------------------------ */
/* 並列処理 (Relight Anime と同じ。作業スレッドを全フレームで共有する)    */
/* ------------------------------------------------------------------ */

struct RowJob {
	std::function<void(A_long)> fn;
	A_long rows;
	std::atomic<A_long> next;
	std::atomic<A_long> finished;
	RowJob(std::function<void(A_long)> f, A_long r) : fn(std::move(f)), rows(r), next(0), finished(0) {}
	void work()
	{
		for (A_long y; (y = next.fetch_add(1)) < rows;) { fn(y); finished.fetch_add(1); }
	}
};

struct RowPool {
	std::mutex m;
	std::condition_variable cv;
	std::vector<std::shared_ptr<RowJob>> jobs;
	unsigned workers = 0;

	static RowPool &get()
	{
		/* 作業スレッドは AE が終わるまで残る。DLL が先に外されないよう固定し、後片付けもしない */
		static RowPool *p = []() {
			HMODULE self = NULL;
			GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
			                   (LPCWSTR)&RowPool::get, &self);
			RowPool *pool = new RowPool;
			unsigned n = std::thread::hardware_concurrency();
			pool->workers = std::max(1u, std::min(n, 32u)) - 1;
			for (unsigned i = 0; i < pool->workers; ++i) std::thread([pool]() { pool->loop(); }).detach();
			return pool;
		}();
		return *p;
	}

	void loop()
	{
		for (;;) {
			std::shared_ptr<RowJob> job;
			{
				std::unique_lock<std::mutex> lk(m);
				cv.wait(lk, [&]() { return !jobs.empty(); });
				job = jobs.front();
				std::rotate(jobs.begin(), jobs.begin() + 1, jobs.end());
			}
			job->work();
			std::lock_guard<std::mutex> lk(m);
			auto it = std::find(jobs.begin(), jobs.end(), job);
			if (it != jobs.end()) jobs.erase(it);
		}
	}

	void run(A_long rows, std::function<void(A_long)> fn)
	{
		auto job = std::make_shared<RowJob>(std::move(fn), rows);
		{
			std::lock_guard<std::mutex> lk(m);
			jobs.push_back(job);
		}
		cv.notify_all();
		job->work();
		while (job->finished.load() < rows) std::this_thread::yield();
		std::lock_guard<std::mutex> lk(m);
		auto it = std::find(jobs.begin(), jobs.end(), job);
		if (it != jobs.end()) jobs.erase(it);
	}
};

template <typename F>
static void parallelRows(A_long rows, F fn)
{
	if (rows < 16) { for (A_long y = 0; y < rows; ++y) fn(y); return; }
	RowPool::get().run(rows, std::function<void(A_long)>(fn));
}

/* ------------------------------------------------------------------ */
/* CPU 版の全体の手順 (GPU 版は DofGPU.cu の DofGPU_Render)              */
/* ------------------------------------------------------------------ */

static const float *HostLut()
{
	static std::vector<float> lut = []() {
		std::vector<float> l(DF_LUT_SIZE * 2);
		dfBuildLut(l.data(), l.data() + DF_LUT_SIZE);
		return l;
	}();
	return lut.data();
}

/*
 * 1 フレームを描く。c には p, outW, outH, originX, originY, layerW, layerH を入れておく。
 * src: 入力 (premultiplied)。depthMap: 深度ワールドと位置合わせ (img.data が NULL なら深度なし)。
 * nb / nbCount: 前後のフレームの深度ワールド。out: 結果 (premultiplied、表示の値、outW x outH)
 * 戻り値: 使ったピントの深度
 */
static float RenderFrameCPU(DofContext &c, const DfImage &src, const DfAuxMap &depthMap, const DfAuxMap &subjectMap,
                            const DfImage *nb, int nbCount, std::vector<DfRgba> &out)
{
	const int W = c.outW, H = c.outH;
	const size_t NP = (size_t)W * H;
	c.lutDec = HostLut();
	c.lutEnc = c.lutDec + DF_LUT_SIZE;
	c.depth = c.coc = c.tileR = c.subject = c.subjectTgt = NULL;
	dfSetupFrame(c, c.p.focusDepth);
	for (int s = 0; s < DF_MAX_SUBJECTS; ++s) c.subjectDepth[s] = -1.f;

	/* 1. リニアの色と縮小 */
	std::vector<float> lin[DF_LEVELS];
	for (int k = 0; k < DF_LEVELS; ++k) lin[k].resize((size_t)c.lw[k] * c.lh[k] * 4);
	parallelRows(H, [&](A_long y) { for (int x = 0; x < W; ++x) dfLinear(&c, src, x, (int)y, &lin[0][((size_t)y * W + x) * 4]); });
	for (int k = 1; k < DF_LEVELS; ++k) {
		const float *prev = lin[k - 1].data();
		const int pw = c.lw[k - 1], ph = c.lh[k - 1], w = c.lw[k];
		float *dst = lin[k].data();
		parallelRows(c.lh[k], [&](A_long y) { for (int x = 0; x < w; ++x) dfDown(prev, pw, ph, x, (int)y, &dst[((size_t)y * w + x) * 4]); });
	}
	for (int k = 0; k < DF_LEVELS; ++k) c.lin[k] = lin[k].data();

	/* 2. 深度 → ピント → ぼけの大きさ */
	std::vector<float> low, depth, tmp, coc, tileMax, tileR, subject, subjectTgt;
	float focus = c.p.focusDepth;
	if (depthMap.img.data) {
		const int dw = depthMap.img.width, dh = depthMap.img.height;
		low.resize((size_t)dw * dh);
		parallelRows(dh, [&](A_long j) {
			for (int i = 0; i < dw; ++i) {
				float d = dfLowDepth(depthMap.img, i, (int)j, c.p.depthNear, c.p.passEncoding == 2);
				if (nbCount > 0) d = dfStabilizeDepth(d, nb, nbCount, i, (int)j, c.p.depthNear, c.p.passEncoding == 2);
				low[(size_t)j * dw + i] = d;
			}
		});
		/* 人物の代表の深度 (マスクの中の深度の中央値) */
		const bool hasSubject = subjectMap.img.data && (c.p.subjectAmount > 0.f || c.p.subjectFlatten > 0.f);
		std::vector<float> lowMask;
		if (hasSubject) {
			lowMask.resize((size_t)dw * dh * 3);
			parallelRows(dh, [&](A_long j) { for (int i = 0; i < dw; ++i) dfSubjectAtLow3(depthMap, subjectMap, c.p.subjectChannel, i, (int)j, &lowMask[((size_t)j * dw + i) * 3]); });
		}
		/* キャラごとの代表の深度。Focus Point がキャラの上なら、そのキャラの深度にピントを合わせる (キャラ全体にピントが合う) */
		focus = dfResolveFocus(c, low.data(), hasSubject ? lowMask.data() : NULL, dw, dh, depthMap.sx, depthMap.sy);
		{
			float sd[DF_MAX_SUBJECTS];
			for (int s = 0; s < DF_MAX_SUBJECTS; ++s) sd[s] = c.subjectDepth[s];
			dfSetupFrame(c, focus);
			for (int s = 0; s < DF_MAX_SUBJECTS; ++s) c.subjectDepth[s] = sd[s];
		}

		depth.resize(NP); tmp.resize(NP); coc.resize(NP);
		parallelRows(H, [&](A_long y) {
			for (int x = 0; x < W; ++x) depth[(size_t)y * W + x] = dfUpsampleDepth(&c, low.data(), dw, dh, depthMap, src, x, (int)y);
		});
		const int rad = dfRound(c.expand);
		if (rad > 0) {
			parallelRows(H, [&](A_long y) { for (int x = 0; x < W; ++x) tmp[(size_t)y * W + x] = dfExpand1D(depth.data(), W, H, rad, true, x, (int)y); });
			parallelRows(H, [&](A_long y) { for (int x = 0; x < W; ++x) depth[(size_t)y * W + x] = dfExpand1D(tmp.data(), W, H, rad, false, x, (int)y); });
		}
		/* キャラのマスク。中の深度を人物の代表の深度に寄せ (Subject Flatten)、Keep Subject Sharp の分だけぼけを減らす */
		if (hasSubject) {
			subject.resize(NP);
			subjectTgt.resize(NP);
			parallelRows(H, [&](A_long y) {
				for (int x = 0; x < W; ++x) {
					float o[3];
					dfSubjectAtF3(subjectMap, c.p.subjectChannel, (float)x * subjectMap.sx + subjectMap.ox, (float)y * subjectMap.sy + subjectMap.oy, o);
					const size_t n = (size_t)y * W + x;
					dfSubjectWeights(&c, o, subject[n], subjectTgt[n]);
				}
			});
			c.subject = subject.data();
			c.subjectTgt = subjectTgt.data();
		}
		if (c.subject) {
			parallelRows(H, [&](A_long y) {
				for (int x = 0; x < W; ++x) {
					const size_t n = (size_t)y * W + x;
					depth[n] = dfSubjectBlendDepth(&c, depth[n], c.subject[n], c.subjectTgt[n]);
				}
			});
		}
		/* 深度の段差を締める (手前と奥の境目にピントの合った線が出ないように)。coc を作業に使う */
		const int step = dfRound(c.step);
		if (step > 0) {
			std::vector<float> wm, ts;
			if (c.subject) { wm.resize(NP); ts.resize(NP); }
			parallelRows(H, [&](A_long y) {
				for (int x = 0; x < W; ++x) {
					const size_t n = (size_t)y * W + x;
					float w, t;
					dfMinMax1D(depth.data(), depth.data(), c.subject, c.subjectTgt, W, H, step, true, x, (int)y, tmp[n], coc[n], w, t);
					if (c.subject) { wm[n] = w; ts[n] = t; }
				}
			});
			parallelRows(H, [&](A_long y) {
				for (int x = 0; x < W; ++x) {
					float mn, mx, w, t;
					dfMinMax1D(tmp.data(), coc.data(), c.subject ? wm.data() : NULL, c.subject ? ts.data() : NULL, W, H, step, false, x, (int)y, mn, mx, w, t);
					const size_t n = (size_t)y * W + x;
					depth[n] = dfStepDepth(depth[n], mn, mx, c.subject ? c.subject[n] : 0.f, w, t, c.focus);
				}
			});
		}
		parallelRows(H, [&](A_long y) {
			for (int x = 0; x < W; ++x) {
				const size_t n = (size_t)y * W + x;
				const float s = c.subject ? c.subject[n] : 0.f;
				coc[n] = dfCoc(&c, depth[n]) * (1.f - dfSat(s * c.p.subjectAmount));
			}
		});
		c.depth = depth.data();
		c.coc = coc.data();

		tileMax.resize((size_t)c.tilesX * c.tilesY);
		tileR.resize(tileMax.size());
		parallelRows(c.tilesY, [&](A_long j) { for (int i = 0; i < c.tilesX; ++i) tileMax[(size_t)j * c.tilesX + i] = dfTileMax(&c, i, (int)j); });
		parallelRows(c.tilesY, [&](A_long j) { for (int i = 0; i < c.tilesX; ++i) tileR[(size_t)j * c.tilesX + i] = dfTileReach(&c, tileMax.data(), i, (int)j); });
		c.tileR = tileR.data();
	}

	/* 3. ぼかしと仕上げ */
	out.resize(NP);
	parallelRows(H, [&](A_long y) {
		for (int x = 0; x < W; ++x) {
			float lin[4];
			float d = 0.f, cc = 0.f;
			if (c.coc) {
				const size_t n = (size_t)y * W + x;
				d = c.depth[n];
				cc = c.coc[n];
				const float R = c.tileR[(size_t)(y / DF_TILE) * c.tilesX + (x / DF_TILE)];
				if (c.p.outputMode == OUTPUT_FINAL || c.p.outputMode == OUTPUT_FOCUS_CHECK) dfGather(&c, x, (int)y, R, lin);
				else dfFetch(&c, 0, x, (int)y, lin);
			} else {
				dfFetch(&c, 0, x, (int)y, lin);
			}
			out[(size_t)y * W + x] = dfFinish(&c, lin, dfRead(src, x, (int)y), d, cc);
		}
	});
	return focus;
}

#ifndef DOF_HAS_CUDA
/* CUDA なしでビルドしたとき (GPU 版は使われない) */
int  DofGPU_Render(DfGpuFrame &, char *err, int errLen) { if (err && errLen > 0) err[0] = 0; return -1; }
void DofGPU_Release() {}
#endif

/* 浮動小数の結果 (premultiplied) を出力ワールドへ書く */
static void WriteWorld(PF_EffectWorld *w, PF_PixelFormat fmt, const std::vector<DfRgba> &buf)
{
	const A_long W = w->width, H = w->height;
	parallelRows(H, [&](A_long y) {
		char *row = (char *)w->data + (size_t)y * (size_t)w->rowbytes;
		for (A_long x = 0; x < W; ++x) {
			const DfRgba &o = buf[(size_t)y * W + x];
			switch (fmt) {
			case PF_PixelFormat_ARGB128: {
				PF_PixelFloat *p = (PF_PixelFloat *)row + x;
				p->alpha = o.a; p->red = o.r; p->green = o.g; p->blue = o.b;
				break;
			}
			case PF_PixelFormat_ARGB64: {
				PF_Pixel16 *p = (PF_Pixel16 *)row + x;
				p->alpha = (A_u_short)(dfSat(o.a) * PF_MAX_CHAN16 + 0.5f);
				p->red   = (A_u_short)(dfSat(o.r) * PF_MAX_CHAN16 + 0.5f);
				p->green = (A_u_short)(dfSat(o.g) * PF_MAX_CHAN16 + 0.5f);
				p->blue  = (A_u_short)(dfSat(o.b) * PF_MAX_CHAN16 + 0.5f);
				break;
			}
			default: {
				PF_Pixel8 *p = (PF_Pixel8 *)row + x;
				p->alpha = (A_u_char)(dfSat(o.a) * PF_MAX_CHAN8 + 0.5f);
				p->red   = (A_u_char)(dfSat(o.r) * PF_MAX_CHAN8 + 0.5f);
				p->green = (A_u_char)(dfSat(o.g) * PF_MAX_CHAN8 + 0.5f);
				p->blue  = (A_u_char)(dfSat(o.b) * PF_MAX_CHAN8 + 0.5f);
				break;
			}
			}
		}
	});
}

/* ------------------------------------------------------------------ */
/* AE コマンド処理                                                     */
/* ------------------------------------------------------------------ */

static PF_Err About(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg, "%s v%d.%d\r%s",
		DOF_NAME, DOF_MAJOR_VERSION, DOF_MINOR_VERSION, DOF_DESCRIPTION);
	return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output)
{
	out_data->my_version = PF_VERSION(DOF_MAJOR_VERSION, DOF_MINOR_VERSION,
	                                  DOF_BUG_VERSION, DOF_STAGE_VERSION, DOF_BUILD_VERSION);
	out_data->out_flags  = DOF_OUTFLAGS;
	out_data->out_flags2 = DOF_OUTFLAGS2;
	Auto_Init(in_data);   /* AEGP 登録とタイマー (深度の自動生成用) */
	return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_LayerDef *output)
{
	PF_Err      err = PF_Err_NONE;
	PF_ParamDef def;

	/* --- Auto --- */
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_START_COLLAPSED;
	PF_ADD_TOPIC("Auto", ID_TOPIC_AUTO);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Auto Passes", TRUE, 0, ID_AUTO_ENABLE);

	/* 深度を作る場所 (Relight Anime と同じ)。Local GPU: この PC。Remote: 別の PC の深度サーバー (設定の remote_server) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Depth Engine", 3, 1, "Local GPU|Remote Fast|Remote Quality", ID_AUTO_ENGINE);

	PF_ADD_BUTTON("Regenerate", "Generate Depth", 0, PF_ParamFlag_SUPERVISE, ID_AUTO_GENERATE);
	PF_ADD_BUTTON("Depth Cache", "Clean Cache...", 0, PF_ParamFlag_SUPERVISE, ID_AUTO_CLEAN);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ID_TOPIC_AUTO_END);

	/* --- Focus --- */
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Focus", ID_TOPIC_FOCUS);

	/* Point: Focus Point の位置の深度にピントを合わせる (コンポビューでドラッグ、キーフレームでピント送り)。
	   Depth: Focus Depth の値で合わせる */
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE;
	PF_ADD_POPUP("Focus Mode", 2, FOCUS_POINT, "Point|Depth", ID_FOCUS_MODE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT("Focus Point", (A_long)DEF_FOCUS_X, (A_long)DEF_FOCUS_Y, FALSE, ID_FOCUS_POINT);

	/* ピント送りにかける時間 (秒)。Focus Point (または Focus Depth) のキーフレームでピントが移るとき、
	   映画のフォーカス送りのように、この時間をかけてゆっくり動き始めてゆっくり止まる。
	   停止したキーフレームでパッと切り替えても、ピントはこの時間で移る。0 でキーフレームのとおり */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Focus Pull Time", 0.f, 5.f, 0.f, 2.f, DEF_FOCUS_PULL,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_FOCUS_PULL);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Focus Depth", 0.f, 1.f, 0.f, 1.f, DEF_FOCUS_DEPTH,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_FOCUS_DEPTH);

	/* ピントの合う幅 (深度の差)。広げると人物全体にピントが合う */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Focus Range", 0.f, 0.5f, 0.f, 0.2f, DEF_FOCUS_RANGE,
		PF_Precision_THOUSANDTHS, PF_ValueDisplayFlag_NONE, 0, ID_FOCUS_RANGE);

	/* キャラのマスク (ロトブラシで切り抜いたレイヤーなど)。この中はぼかさず、手前のぼけも重ねない。
	   推定した深度はキャラの中でも揺れる (髪の先・脚・フリル) ので、体全体をくっきり残したいときに使う */
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Subject Layer", PF_LayerDefault_NONE, ID_SUBJECT_LAYER);

	AEFX_CLR_STRUCT(def);
	/* Multi (RGB): 赤・緑・青をキャラ 1・2・3 人目として、キャラごとに自分の深度へ寄せる (自動のマスクはこれ) */
	PF_ADD_POPUP("Subject Channel", 3, 1, "Alpha|Luminance|Multi (RGB)", ID_SUBJECT_CHANNEL);

	/* マスクの中の深度を、人物の代表の深度 (マスクの中の深度の中央値) に寄せる。100% で人物を 1 枚の板として扱う。
	   Focus Point が人物の上なら人物全体にピントが合い、背景へ動かせば人物は全体が均一にぼける (ピント送りができる) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Subject Flatten", 0.f, 100.f, 0.f, 100.f, 100.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_SUBJECT_FLATTEN);

	/* ピントに関係なく、マスクの中を必ずくっきり残す (手前のぼけも重ねない) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Keep Subject Sharp", 0.f, 100.f, 0.f, 100.f, 0.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_SUBJECT_AMOUNT);

	/* キャラのマスクを自動で作る (SAM 3.1)。深度ができたら、今の時刻の Focus Point にいるキャラを動画全体で追いかける。
	   別のキャラにピントを移したら Generate Subject で作り直す */
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Auto Subject", TRUE, 0, ID_AUTO_SUBJECT);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Subject Type", 3, 1, "Anime Character|Person|Animal", ID_SUBJECT_TYPE);

	PF_ADD_BUTTON("Subject Mask", "Generate Subject", 0, PF_ParamFlag_SUPERVISE, ID_SUBJECT_GENERATE);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ID_TOPIC_FOCUS_END);

	/* --- Blur --- */
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Blur", ID_TOPIC_BLUR);

	/* ぼけの大きさ: ピントから深度が 0.5 離れた所のぼけの半径 (1080p でのピクセル数) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Blur Size", 0.f, 160.f, 0.f, 80.f, DEF_BLUR_SIZE,
		PF_Precision_TENTHS, PF_ValueDisplayFlag_PIXEL, 0, ID_BLUR_SIZE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Near Blur", 0.f, 3.f, 0.f, 2.f, 1.f,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_NEAR_BLUR);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Far Blur", 0.f, 3.f, 0.f, 2.f, 1.f,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_FAR_BLUR);

	/* 手前の物の輪郭を広げる幅。推定した深度の境目は輪郭線より内側にあることが多いので、輪郭線を手前の物に入れる */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Edge Expand", 0.f, 12.f, 0.f, 6.f, DEF_EDGE_EXPAND,
		PF_Precision_TENTHS, PF_ValueDisplayFlag_PIXEL, 0, ID_EDGE_EXPAND);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ID_TOPIC_BLUR_END);

	/* --- Bokeh --- */
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC("Bokeh", ID_TOPIC_BOKEH);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Shape", 3, SHAPE_CIRCLE, "Circle|Hexagon|Octagon", ID_SHAPE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE("Rotation", 0, ID_SHAPE_ROTATION);

	/* 明るい所を玉ぼけにする強さと、玉ぼけにし始める明るさ */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Highlights", 0.f, 5.f, 0.f, 3.f, DEF_HIGHLIGHTS,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_HIGHLIGHTS);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Highlight Threshold", 0.f, 1.f, 0.f, 1.f, DEF_HIGHLIGHT_THR,
		PF_Precision_HUNDREDTHS, PF_ValueDisplayFlag_NONE, 0, ID_HIGHLIGHT_THRESHOLD);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Quality", 3, QUALITY_NORMAL, "Draft|Normal|High", ID_QUALITY);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ID_TOPIC_BOKEH_END);

	/* --- Passes --- */
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_START_COLLAPSED;
	PF_ADD_TOPIC("Passes", ID_TOPIC_PASSES);

	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER("Depth Layer", PF_LayerDefault_NONE, ID_DEPTH_LAYER);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Depth Near Is", 2, 1, "White|Black", ID_DEPTH_NEAR);

	/* 深度を前後のフレームで時間方向にならす (ぼけの境目がちらつくのを抑える) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Depth Stabilize", 3, 2, "Off|1 Frame|2 Frames", ID_DEPTH_STABILIZE);

	/* 深度の境目を素材の線画に合わせる (ぼけの境目が人物の輪郭に沿う) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Snap Depth To Lines", TRUE, 0, ID_DEPTH_SNAP);

	/* AE は Preserve RGB を付けないと EXR をリニア→sRGB 変換して渡してくる。その場合は逆変換で戻す */
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Pass Encoding", 2, 1, "Linear (Preserve RGB)|sRGB (decode)", ID_PASS_ENCODING);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Input Is Linear", FALSE, 0, ID_INPUT_LINEAR);

	/* GPU (CUDA) で描く。AE のプロジェクト設定が Mercury GPU アクセラレーション (CUDA) のときだけ効く */
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX("Use GPU", TRUE, 0, ID_USE_GPU);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ID_TOPIC_PASSES_END);

	/* --- Output --- */
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Output", 4, OUTPUT_FINAL, "Final|Depth|Blur Map|Focus Check", ID_OUTPUT_MODE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Mix", 0.f, 100.f, 0.f, 100.f, 100.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_MIX);

	out_data->num_params = DOF_NUM_PARAMS;

	/* コンポビューにピントの目印を描くため、コンポ窓のイベントを受け取る */
	if (!err) {
		PF_CustomUIInfo ci;
		AEFX_CLR_STRUCT(ci);
		ci.events = PF_CustomEFlag_COMP;
		ci.comp_ui_alignment = PF_UIAlignment_NONE;
		ci.layer_ui_alignment = PF_UIAlignment_NONE;
		ci.preview_ui_alignment = PF_UIAlignment_NONE;
		err = (*(in_data->inter.register_ui))(in_data->effect_ref, &ci);
	}
	return err;
}

/* Focus Mode で使わない項目を灰色にする (見た目だけ。値は変えない) */
static PF_Err UpdateFocusUI(PF_InData *in_data, PF_ParamDef *params[])
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	const bool point = params[DOF_FOCUS_MODE]->u.pd.value == FOCUS_POINT;
	const A_long idx[2] = {DOF_FOCUS_POINT, DOF_FOCUS_DEPTH};
	const bool on[2] = {point, !point};
	for (int i = 0; i < 2 && !err; ++i) {
		const bool disabled = (params[idx[i]]->ui_flags & PF_PUI_DISABLED) != 0;
		if (disabled == !on[i]) continue;
		PF_ParamDef copy = *params[idx[i]];
		if (on[i]) copy.ui_flags &= ~PF_PUI_DISABLED; else copy.ui_flags |= PF_PUI_DISABLED;
		ERR(suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref, idx[i], &copy));
	}
	return err;
}

/* ------------------------------------------------------------------ */
/* パラメータ読み出し (SmartRender 時に checkout)                      */
/* ------------------------------------------------------------------ */

struct ParamReader {
	PF_InData *in_data;
	PF_Err     err;

	explicit ParamReader(PF_InData *d) : in_data(d), err(PF_Err_NONE) {}

	bool checkout(A_long index, PF_ParamDef &def)
	{
		AEFX_CLR_STRUCT(def);
		PF_Err e = PF_CHECKOUT_PARAM(in_data, index, in_data->current_time,
		                             in_data->time_step, in_data->time_scale, &def);
		if (e) { if (!err) err = e; return false; }
		return true;
	}
	void checkin(PF_ParamDef &def)
	{
		PF_Err e = PF_CHECKIN_PARAM(in_data, &def);
		if (e && !err) err = e;
	}
	float fs(A_long index, float dflt = 0.f)
	{
		PF_ParamDef def; float v = dflt;
		if (checkout(index, def)) { v = (float)def.u.fs_d.value; checkin(def); }
		return v;
	}
	A_long popup(A_long index, A_long dflt = 1)
	{
		PF_ParamDef def; A_long v = dflt;
		if (checkout(index, def)) { v = def.u.pd.value; checkin(def); }
		return v;
	}
	bool checkbox(A_long index, bool dflt = false)
	{
		PF_ParamDef def; bool v = dflt;
		if (checkout(index, def)) { v = def.u.bd.value != 0; checkin(def); }
		return v;
	}
	float angle(A_long index, float dflt = 0.f)
	{
		PF_ParamDef def; float v = dflt;
		if (checkout(index, def)) { v = (float)FIX_2_FLOAT(def.u.ad.value); checkin(def); }
		return v;
	}
	/* 2D ポイント。値はレンダー解像度 (ダウンサンプル後) のレイヤー座標で渡される */
	void point2d(A_long index, float *xy)
	{
		PF_ParamDef def;
		xy[0] = xy[1] = 0.f;
		if (checkout(index, def)) {
			xy[0] = (float)FIX_2_FLOAT(def.u.td.x_value);
			xy[1] = (float)FIX_2_FLOAT(def.u.td.y_value);
			checkin(def);
		}
	}
};

/*
 * ピント送り。Focus Pull Time の長さの窓の中の前後のコマについて、そのコマのピントの決め方 (DfFocusSample) を p に入れる
 * (重みは Hann の窓。ピントの深度はあとで深度の地図を見て dfResolveFocus が平均する)。
 * 各コマでは、Focus Point が動いていれば、前後のコマの値を 1 コマずつたどって値が止まった所 (キーフレームの位置や
 * 停止の間) を動き始め A・動き終わり B とする。最大 FOCUS_SEARCH コマまで。
 * Focus Mode = Depth なら、Focus Depth の値を同じ重みで平均する。
 */
static const int FOCUS_SEARCH = 240;

static void ReadFocusPull(PF_InData *in_data, DofParams &p, float pullSec)
{
	p.pullCount = 0;
	if (in_data->time_step <= 0 || in_data->time_scale == 0) return;
	const A_long t0 = in_data->current_time, dt = in_data->time_step;
	/* 窓の半分の長さ (コマ)。長いときは間引いて DF_PULL_MAX 個までにする */
	const double frames = (double)dfMax(0.f, pullSec) * (double)in_data->time_scale / (double)dt;
	const int half = dfClamp((int)floor(frames * 0.5 + 0.5), 0, FOCUS_SEARCH);
	const int maxHalf = (DF_PULL_MAX - 1) / 2;
	const int stride = half > maxHalf ? (half + maxHalf - 1) / maxHalf : 1;
	const int nh = half / stride;
	auto weight = [&](int i) { return 0.5f + 0.5f * cosf(3.14159265f * (float)i / (float)(nh + 1)); };

	if (p.focusMode != FOCUS_POINT) {
		if (nh <= 0) return;
		float sum = 0.f, ws = 0.f;
		for (int i = -nh; i <= nh; ++i) {
			PF_ParamDef def;
			AEFX_CLR_STRUCT(def);
			if (PF_CHECKOUT_PARAM(in_data, DOF_FOCUS_DEPTH, t0 + (A_long)(i * stride) * dt, dt, in_data->time_scale, &def) != PF_Err_NONE) continue;
			const float w = weight(i);
			sum += (float)FIX_2_FLOAT(def.u.fs_d.value) * w;
			PF_CHECKIN_PARAM(in_data, &def);
			ws += w;
		}
		if (ws > 0.f) p.focusDepth = dfSat(sum / ws);
		return;
	}

	/* コマ k (今のコマからのずれ) の Focus Point。同じコマを何度も読まないよう覚えておく */
	struct Pt { float x, y; bool ok; };
	std::map<int, Pt> cache;
	auto pointAt = [&](int k, float *xy) -> bool {
		auto it = cache.find(k);
		if (it == cache.end()) {
			Pt v = {0.f, 0.f, false};
			PF_ParamDef def;
			AEFX_CLR_STRUCT(def);
			if (PF_CHECKOUT_PARAM(in_data, DOF_FOCUS_POINT, t0 + (A_long)k * dt, dt, in_data->time_scale, &def) == PF_Err_NONE) {
				v.x = (float)FIX_2_FLOAT(def.u.td.x_value);
				v.y = (float)FIX_2_FLOAT(def.u.td.y_value);
				v.ok = true;
				PF_CHECKIN_PARAM(in_data, &def);
			}
			it = cache.emplace(k, v).first;
		}
		xy[0] = it->second.x; xy[1] = it->second.y;
		return it->second.ok;
	};
	auto same = [](const float *a, const float *b) { return fabsf(a[0] - b[0]) < 0.01f && fabsf(a[1] - b[1]) < 0.01f; };
	/* コマ k のピントの決め方 */
	auto sampleAt = [&](int k, DfFocusSample &s) -> bool {
		float pk[2], prev[2], next[2];
		if (!pointAt(k, pk)) return false;
		s.ax = s.bx = pk[0]; s.ay = s.by = pk[1]; s.t = 0.f;
		const bool hp = pointAt(k - 1, prev), hn = pointAt(k + 1, next);
		if ((!hp || same(prev, pk)) && (!hn || same(next, pk))) return true;   /* 止まっている */
		/* 後ろへたどって、止まった所 = A */
		float a[2] = {pk[0], pk[1]}, b[2] = {pk[0], pk[1]}, cur[2] = {pk[0], pk[1]};
		for (int j = 1; j <= FOCUS_SEARCH; ++j) {
			float q[2];
			if (!pointAt(k - j, q) || same(q, cur)) break;
			cur[0] = a[0] = q[0]; cur[1] = a[1] = q[1];
		}
		/* 前へたどって、止まった所 = B */
		cur[0] = pk[0]; cur[1] = pk[1];
		for (int j = 1; j <= FOCUS_SEARCH; ++j) {
			float q[2];
			if (!pointAt(k + j, q) || same(q, cur)) break;
			cur[0] = b[0] = q[0]; cur[1] = b[1] = q[1];
		}
		const float vx = b[0] - a[0], vy = b[1] - a[1], len2 = vx * vx + vy * vy;
		if (len2 < 1.f) return true;
		s.ax = a[0]; s.ay = a[1]; s.bx = b[0]; s.by = b[1];
		s.t = dfSat(((pk[0] - a[0]) * vx + (pk[1] - a[1]) * vy) / len2);
		return true;
	};
	bool varies = false;
	for (int i = -nh; i <= nh; ++i) {
		DfFocusSample s;
		if (!sampleAt(i * stride, s)) continue;
		s.w = weight(i);
		if (s.ax != p.focusX || s.ay != p.focusY || s.bx != p.focusX || s.by != p.focusY) varies = true;
		p.pull[p.pullCount++] = s;
	}
	if (!varies) p.pullCount = 0;   /* 窓の中でずっと止まっている */
}

static PF_Err ReadDofParams(PF_InData *in_data, DofParams &p)
{
	ParamReader r(in_data);
	p.focusMode   = (int)r.popup(DOF_FOCUS_MODE, FOCUS_POINT);
	float fp[2];
	r.point2d(DOF_FOCUS_POINT, fp);
	p.focusX = fp[0]; p.focusY = fp[1];
	p.focusDepth  = dfSat(r.fs(DOF_FOCUS_DEPTH, DEF_FOCUS_DEPTH));
	ReadFocusPull(in_data, p, r.fs(DOF_FOCUS_PULL, DEF_FOCUS_PULL));
	p.focusRange  = dfMax(0.f, r.fs(DOF_FOCUS_RANGE, DEF_FOCUS_RANGE));
	p.subjectChannel = (int)r.popup(DOF_SUBJECT_CHANNEL, 1);
	p.subjectAmount  = dfSat(r.fs(DOF_SUBJECT_AMOUNT, 0.f) / 100.f);
	p.subjectFlatten = dfSat(r.fs(DOF_SUBJECT_FLATTEN, 100.f) / 100.f);
	p.blurSize    = dfMax(0.f, r.fs(DOF_BLUR_SIZE, DEF_BLUR_SIZE));
	p.nearScale   = dfMax(0.f, r.fs(DOF_NEAR_BLUR, 1.f));
	p.farScale    = dfMax(0.f, r.fs(DOF_FAR_BLUR, 1.f));
	p.edgeExpand  = dfMax(0.f, r.fs(DOF_EDGE_EXPAND, DEF_EDGE_EXPAND));
	p.shape       = (int)r.popup(DOF_SHAPE, SHAPE_CIRCLE);
	p.shapeRotation = r.angle(DOF_SHAPE_ROTATION, 0.f);
	p.highlights  = dfMax(0.f, r.fs(DOF_HIGHLIGHTS, DEF_HIGHLIGHTS));
	p.highlightThreshold = dfSat(r.fs(DOF_HIGHLIGHT_THRESHOLD, DEF_HIGHLIGHT_THR));
	p.quality     = (int)r.popup(DOF_QUALITY, QUALITY_NORMAL);
	p.depthNear   = (int)r.popup(DOF_DEPTH_NEAR, 1);
	p.depthStabilize = (int)r.popup(DOF_DEPTH_STABILIZE, 2) - 1;
	p.depthSnap   = r.checkbox(DOF_DEPTH_SNAP, true);
	p.passEncoding = (int)r.popup(DOF_PASS_ENCODING, 1);
	p.inputLinear = r.checkbox(DOF_INPUT_LINEAR, false);
	p.outputMode  = (int)r.popup(DOF_OUTPUT_MODE, OUTPUT_FINAL);
	p.mix         = dfSat(r.fs(DOF_MIX, 100.f) / 100.f);
	return r.err;
}

/* ------------------------------------------------------------------ */
/* SmartFX                                                             */
/* ------------------------------------------------------------------ */

static void RenderLog(const std::wstring &msg) { Dof_Log(msg); }

/* 前後のフレームの深度を取り出すときの番号 (他の取り出しと重ならない値) */
static const A_long DOF_NB_ID = 400;

static void DeletePreRenderData(void *pre_render_dataPV)
{
	delete static_cast<PreRenderData *>(pre_render_dataPV);
}

/* ぼけは周りの画素を集めて作るので、描く範囲の外側も入力として取り出す (ぼけの最大の半径ぶん) */
static PF_Err PreRender(PF_InData *in_data, PF_OutData *out_data, PF_PreRenderExtra *extra)
{
	PF_Err err = PF_Err_NONE;
	PF_RenderRequest req = extra->input->output_request;
	req.preserve_rgb_of_zero_alpha = TRUE;

	/* 拡大表示で一部だけ描くときも、まわりの画素がぼけに入るよう、入力は広めに取る */
	const float dsy = (float)in_data->downsample_y.num / (float)std::max<A_long>(1, in_data->downsample_y.den);
	const A_long pad = (A_long)ceilf(DF_MAX_COC_1080 * (float)in_data->height * dsy / 1080.f) + DF_TILE;
	PF_RenderRequest inReq = req;
	inReq.rect.left -= pad; inReq.rect.top -= pad; inReq.rect.right += pad; inReq.rect.bottom += pad;

	PF_CheckoutResult in_result;
	AEFX_CLR_STRUCT(in_result);
	ERR(extra->cb->checkout_layer(in_data->effect_ref, DOF_INPUT, DOF_INPUT, &inReq,
	                              in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));
	if (err) return err;

	/* 出力は入力と同じ範囲 (ぼけで外へは広げない。素材の端はそのまま) */
	extra->output->result_rect     = in_result.result_rect;
	extra->output->max_result_rect = in_result.max_result_rect;
	extra->output->solid           = FALSE;

#ifdef DOF_HAS_CUDA
	if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
		PF_ParamDef def;
		AEFX_CLR_STRUCT(def);
		bool useGpu = true;
		if (PF_CHECKOUT_PARAM(in_data, DOF_USE_GPU, in_data->current_time, in_data->time_step, in_data->time_scale, &def) == PF_Err_NONE) {
			useGpu = def.u.bd.value != 0;
			PF_CHECKIN_PARAM(in_data, &def);
		}
		if (useGpu) extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
	}
#endif

	PreRenderData *prd = new PreRenderData;
	std::memset(prd, 0, sizeof(*prd));
	prd->originX = in_result.result_rect.left;
	prd->originY = in_result.result_rect.top;
	extra->output->pre_render_data = prd;
	extra->output->delete_pre_render_data_func = DeletePreRenderData;

	/* 深度レイヤーとキャラのマスクのレイヤーは全体を取り出す (Relight Anime と同じ。まず 1 画素だけ要求して全体の範囲を知る)。
	   未指定でも番号は予約しておく (予約していない番号を取り出すと AE が「Unknown checkout id」を出す) */
	for (int which = 0; which < 2; ++which) {
		const A_long idx = which == 0 ? DOF_DEPTH_LAYER : DOF_SUBJECT_LAYER;
		PF_LRect &rect = which == 0 ? prd->depthRect : prd->subjectRect;
		bool &checked = which == 0 ? prd->depthChecked : prd->subjectChecked;
		PF_CheckoutResult probe;
		AEFX_CLR_STRUCT(probe);
		PF_RenderRequest probeReq = req;
		probeReq.rect.left = 0; probeReq.rect.top = 0; probeReq.rect.right = 1; probeReq.rect.bottom = 1;
		PF_LRect full;
		AEFX_CLR_STRUCT(full);
		if (extra->cb->checkout_layer(in_data->effect_ref, idx, idx + 100, &probeReq,
		                              in_data->current_time, in_data->time_step, in_data->time_scale, &probe) == PF_Err_NONE)
			full = probe.max_result_rect;
		PF_CheckoutResult aux;
		AEFX_CLR_STRUCT(aux);
		PF_RenderRequest auxReq = req;
		auxReq.rect = (full.right > full.left && full.bottom > full.top) ? full : in_result.result_rect;
		if (extra->cb->checkout_layer(in_data->effect_ref, idx, idx, &auxReq,
		                              in_data->current_time, in_data->time_step, in_data->time_scale, &aux) == PF_Err_NONE) {
			checked = true;
			if (full.right > full.left && full.bottom > full.top) rect = aux.result_rect;
		}
	}

	/* 深度を時間方向にならすため、前後のフレームの深度も予約する */
	prd->nbCount = 0;
	if (prd->depthChecked && prd->depthRect.right > prd->depthRect.left) {
		int stab = 1;
		PF_ParamDef def;
		AEFX_CLR_STRUCT(def);
		if (PF_CHECKOUT_PARAM(in_data, DOF_DEPTH_STABILIZE, in_data->current_time, in_data->time_step, in_data->time_scale, &def) == PF_Err_NONE) {
			stab = def.u.pd.value - 1;
			PF_CHECKIN_PARAM(in_data, &def);
		}
		stab = std::max(0, std::min(stab, DF_STABILIZE_MAX));
		static const int offs[4] = {-1, 1, -2, 2};
		for (int k = 0; k < stab * 2; ++k) {
			const int slot = prd->nbCount;
			PF_CheckoutResult r;
			AEFX_CLR_STRUCT(r);
			PF_RenderRequest q = req;
			q.rect = prd->depthRect;
			const A_long when = in_data->current_time + offs[k] * in_data->time_step;
			if (extra->cb->checkout_layer(in_data->effect_ref, DOF_DEPTH_LAYER, DOF_NB_ID + slot, &q,
			                              when, in_data->time_step, in_data->time_scale, &r) == PF_Err_NONE) {
				prd->nbOff[slot] = offs[k];
				prd->nbChecked[slot] = true;
				prd->nbCount++;
			}
		}
	}
	return err;
}

/* 描画のログ。設定が前回と同じなら 3 秒に 1 回まで */
static void RenderLogThrottled(const wchar_t *line)
{
	static std::mutex mtx;
	static std::wstring last;
	static std::chrono::steady_clock::time_point lastTime;
	std::wstring key = line;
	size_t a = key.find(L" t="), b = key.find(L" ", a + 3);
	if (a != std::wstring::npos && b != std::wstring::npos) key.erase(a, b - a);
	size_t m = key.rfind(L" err=");
	if (m != std::wstring::npos) key.erase(m);
	std::lock_guard<std::mutex> lk(mtx);
	const auto now = std::chrono::steady_clock::now();
	if (key == last && now - lastTime < std::chrono::seconds(3)) return;
	last = key;
	lastTime = now;
	Dof_Log(line);
}

/* AE のワールド → DfImage。data を渡せば (GPU のワールド) それを使う */
static DfImage ImageOf(const PF_EffectWorld *w, int format, const void *data = NULL)
{
	DfImage im;
	im.data = NULL;
	im.width = im.height = im.rowbytes = 0;
	im.format = format;
	if (!w) return im;
	im.data = (const unsigned char *)(data ? data : w->data);
	im.width = w->width;
	im.height = w->height;
	im.rowbytes = w->rowbytes;
	return im;
}

/*
 * 描画。isGPU なら PF_Cmd_SMART_RENDER_GPU (ワールドは GPU のメモリ、BGRA float) で、DofGPU_Render が描く。
 * そうでなければ CPU で RenderFrameCPU が描く。どちらも DofCore.h の同じ計算。
 * 入力は出力より広く取り出してある (PreRender) ので、入力全体を描いてから出力の範囲を写す。
 */
static PF_Err SmartRender(PF_InData *in_data, PF_OutData *out_data, PF_SmartRenderExtra *extra, bool isGPU)
{
	PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;

	PF_EffectWorld *inputP = NULL, *outputP = NULL, *depthP = NULL, *subjectP = NULL;
	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, DOF_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	const PreRenderData *prd = (const PreRenderData *)extra->input->pre_render_data;
	if (!err) {
		if (!prd || !prd->depthChecked || extra->cb->checkout_layer_pixels(in_data->effect_ref, DOF_DEPTH_LAYER, &depthP)) depthP = NULL;
		if (depthP && !isGPU && !depthP->data) depthP = NULL;
		if (depthP && depthP->width <= 0) depthP = NULL;
		if (!prd || !prd->subjectChecked || extra->cb->checkout_layer_pixels(in_data->effect_ref, DOF_SUBJECT_LAYER, &subjectP)) subjectP = NULL;
		if (subjectP && !isGPU && !subjectP->data) subjectP = NULL;
		if (subjectP && subjectP->width <= 0) subjectP = NULL;
	}

	/* 前後のフレームの深度。大きさが違うものは使わない */
	PF_EffectWorld *nbW[4] = {NULL, NULL, NULL, NULL};
	int nbN = 0;
	std::vector<A_long> nbCheckedOut;
	if (!err && depthP && prd) {
		for (int k = 0; k < prd->nbCount && k < 4; ++k) {
			if (!prd->nbChecked[k]) continue;
			PF_EffectWorld *w = NULL;
			if (extra->cb->checkout_layer_pixels(in_data->effect_ref, DOF_NB_ID + k, &w) != PF_Err_NONE) continue;
			nbCheckedOut.push_back(DOF_NB_ID + k);
			if (!w || w->width != depthP->width || w->height != depthP->height || (!isGPU && !w->data)) continue;
			nbW[nbN++] = w;
		}
	}

	const float dsx = (float)in_data->downsample_x.num / (float)in_data->downsample_x.den;
	const float dsy = (float)in_data->downsample_y.num / (float)in_data->downsample_y.den;
	const float layerW = std::max(1.f, (float)in_data->width * dsx);
	const float layerH = std::max(1.f, (float)in_data->height * dsy);

	DofParams p;
	std::memset(&p, 0, sizeof(p));
	if (!err) {
		PF_Err e = ReadDofParams(in_data, p);
		if (e) { RenderLog(L"render: param checkout failed err=" + std::to_wstring((long)e)); err = e; }
	}
	const auto t0 = std::chrono::steady_clock::now();
	float focusUsed = -1.f;
	{
		/* 調べもの: 深度ありの描画を始める前に 1 回だけ記録する (落ちた場所を絞るため) */
		static std::atomic<int> traced(0);
		if (depthP && traced.fetch_add(1) < 3)
			RenderLog(std::wstring(L"render: begin with depth ") + (isGPU ? L"gpu" : L"cpu") +
			          L" in=" + std::to_wstring(inputP ? (long)inputP->width : 0L) + L"x" + std::to_wstring(inputP ? (long)inputP->height : 0L) +
			          L" out=" + std::to_wstring(outputP ? (long)outputP->width : 0L) + L"x" + std::to_wstring(outputP ? (long)outputP->height : 0L) +
			          L" depth=" + std::to_wstring((long)depthP->width) + L"x" + std::to_wstring((long)depthP->height) +
			          L" nb=" + std::to_wstring(nbN) +
			          L" rect=" + std::to_wstring(prd ? (long)(prd->depthRect.right - prd->depthRect.left) : -1L) +
			          L" origin=" + std::to_wstring(prd ? (long)prd->originX : 0L) + L"," + std::to_wstring(prd ? (long)prd->originY : 0L));
	}

	try {
	if (!err && inputP && outputP) {
		/* 入力の左上のレイヤー座標 = 出力の左上 - (出力の左上 - 入力の左上)。
		   PreRender の result_rect は入力の範囲と同じなので、出力と入力は同じ位置・大きさ */
		const float originX = prd ? (float)prd->originX : 0.f;
		const float originY = prd ? (float)prd->originY : 0.f;
		DofContext ctx;
		std::memset(&ctx, 0, sizeof(ctx));
		ctx.p       = p;
		ctx.outW    = std::min(inputP->width, outputP->width);
		ctx.outH    = std::min(inputP->height, outputP->height);
		ctx.originX = originX;
		ctx.originY = originY;
		ctx.layerW  = layerW;
		ctx.layerH  = layerH;
		/* 補助レイヤー全体の大きさ。PreRender で取れなかったときは、取り出したワールドの大きさを全体とみなす */
		auto auxMapOf = [&](const DfImage &img, const PF_LRect &rect, const PF_EffectWorld *w) {
			float fw = 1.f, fh = 1.f;
			if (rect.right > rect.left) {
				fw = (float)(rect.right - rect.left);
				fh = (float)(rect.bottom - rect.top);
			} else if (w) {
				fw = (float)w->width; fh = (float)w->height;
			}
			return dfMakeAuxMap(img, fw, fh, originX, originY, layerW, layerH);
		};
		PF_LRect noRect;
		AEFX_CLR_STRUCT(noRect);
		auto depthMapOf = [&](const DfImage &img) { return auxMapOf(img, prd ? prd->depthRect : noRect, depthP); };
		auto subjectMapOf = [&](const DfImage &img) { return auxMapOf(img, prd ? prd->subjectRect : noRect, subjectP); };

		if (!isGPU) {
			PF_PixelFormat fmt = PF_PixelFormat_ARGB32;
			int code = DF_FMT_ARGB8;
			if (extra->input->bitdepth == 32)      { fmt = PF_PixelFormat_ARGB128; code = DF_FMT_ARGB32F; }
			else if (extra->input->bitdepth == 16) { fmt = PF_PixelFormat_ARGB64;  code = DF_FMT_ARGB16; }
			const DfAuxMap depthMap = depthMapOf(ImageOf(depthP, code));
			DfImage nbImg[4];
			for (int k = 0; k < nbN; ++k) nbImg[k] = ImageOf(nbW[k], code);
			std::vector<DfRgba> buf;
			focusUsed = RenderFrameCPU(ctx, ImageOf(inputP, code), depthMap, subjectMapOf(ImageOf(subjectP, code)), nbImg, nbN, buf);
			WriteWorld(outputP, fmt, buf);
		} else {
			AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
			AEFX_SuiteScoper<PF_WorldSuite2> worldSuite(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
			PF_PixelFormat pf = PF_PixelFormat_INVALID;
			worldSuite->PF_GetPixelFormat(inputP, &pf);
			if (pf != PF_PixelFormat_GPU_BGRA128) {
				RenderLog(L"render(gpu): unsupported pixel format " + std::to_wstring((long)pf));
				err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
			} else {
				auto mem = [&](PF_EffectWorld *w) -> void * {
					void *ptr = NULL;
					if (w && gpu->GetGPUWorldData(in_data->effect_ref, w, &ptr) != PF_Err_NONE) ptr = NULL;
					return ptr;
				};
				void *srcMem = mem(inputP), *dstMem = mem(outputP), *depthMem = mem(depthP), *subjectMem = mem(subjectP);
				DfGpuFrame f;
				std::memset(&f, 0, sizeof(f));
				f.ctx = ctx;
				f.depthMap    = depthMapOf(ImageOf(depthMem ? depthP : NULL, DF_FMT_BGRA32F, depthMem));
				f.subjectMap  = subjectMapOf(ImageOf(subjectMem ? subjectP : NULL, DF_FMT_BGRA32F, subjectMem));
				f.src         = ImageOf(inputP, DF_FMT_BGRA32F, srcMem);
				f.dst         = dstMem;
				f.dstRowbytes = outputP->rowbytes;
				for (int k = 0; k < nbN; ++k) {
					void *m = mem(nbW[k]);
					if (!m) continue;
					f.depthNb[f.depthNbCount++] = ImageOf(nbW[k], DF_FMT_BGRA32F, m);
				}
				if (!srcMem || !dstMem) {
					RenderLog(L"render(gpu): GetGPUWorldData failed");
					err = PF_Err_INTERNAL_STRUCT_DAMAGED;
				} else {
					char msg[256] = {0};
					if (DofGPU_Render(f, msg, sizeof(msg)) != 0) {
						std::string m = msg;
						RenderLog(L"render(gpu): " + std::wstring(m.begin(), m.end()));
						err = PF_Err_INTERNAL_STRUCT_DAMAGED;
					}
					focusUsed = f.focusUsed;
				}
			}
		}
	}
	} catch (PF_Err &e) {
		err = e;
		RenderLog(L"render: PF_Err thrown " + std::to_wstring((long)e));
	} catch (std::exception &ex) {
		std::string what = ex.what();
		RenderLog(L"render: exception " + std::wstring(what.begin(), what.end()));
		err = PF_Err_OUT_OF_MEMORY;
	} catch (...) {
		RenderLog(L"render: unknown exception");
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	{
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		wchar_t line[320];
		swprintf(line, 320, L"render: t=%ld/%lu %ldx%ld %ls ds=%.2f mode=%d focus=%.3f blur=%.1f out=%d depth=%d subject=%d err=%d %.0f ms",
			(long)in_data->current_time, (unsigned long)in_data->time_scale,
			outputP ? (long)outputP->width : 0L, outputP ? (long)outputP->height : 0L, isGPU ? L"gpu" : L"cpu", dsx,
			p.focusMode, focusUsed, p.blurSize, p.outputMode, depthP ? 1 + nbN : 0, subjectP ? 1 : 0, (int)err, ms);
		RenderLogThrottled(line);
	}

	for (A_long id : nbCheckedOut) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, id));
	ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, DOF_INPUT));
	if (depthP) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, DOF_DEPTH_LAYER));
	if (subjectP) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, DOF_SUBJECT_LAYER));
	return err;
}

/* ------------------------------------------------------------------ */
/* コンポビューのピントの目印 (PF_Cmd_EVENT)                             */
/* ------------------------------------------------------------------ */

/* レイヤー座標 → 表示中の窓のピクセル座標 */
static void LayerToFrame(PF_InData *in_data, PF_EventExtra *ex, float lx, float ly, float &fx, float &fy)
{
	PF_FixedPoint pt;
	pt.x = FLOAT2FIX(lx);
	pt.y = FLOAT2FIX(ly);
	if ((*ex->contextH)->w_type == PF_Window_COMP)
		ex->cbs.layer_to_comp(ex->cbs.refcon, ex->contextH, in_data->current_time, in_data->time_scale, &pt);
	ex->cbs.source_to_frame(ex->cbs.refcon, ex->contextH, &pt);
	fx = (float)FIX_2_FLOAT(pt.x);
	fy = (float)FIX_2_FLOAT(pt.y);
}

/* Focus Point の位置に、ピント合わせの枠 (四隅のかぎ) と「Focus」の文字を描く。
   位置のドラッグは AE 標準の 2D ポイントのハンドルに任せる */
static PF_Err DrawFocusGizmo(PF_InData *in_data, PF_ParamDef *params[], PF_EventExtra *ex)
{
	PF_Err err = PF_Err_NONE;
	{
		static std::atomic<int> traced(0);
		if (traced.fetch_add(1) < 2) Dof_Log(L"gizmo: draw");
	}
	if (params[DOF_FOCUS_MODE]->u.pd.value != FOCUS_POINT) return err;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	DRAWBOT_DrawRef drawRef = NULL;
	DRAWBOT_SupplierRef supplier = NULL;
	DRAWBOT_SurfaceRef surface = NULL;
	ERR(suites.EffectCustomUISuite1()->PF_GetDrawingReference(ex->contextH, &drawRef));
	if (err || !drawRef) return err;
	ERR(suites.DrawbotSuiteCurrent()->GetSupplier(drawRef, &supplier));
	ERR(suites.DrawbotSuiteCurrent()->GetSurface(drawRef, &surface));
	if (err || !supplier || !surface) return err;

	DRAWBOT_SupplierSuiteCurrent *sup = suites.SupplierSuiteCurrent();
	DRAWBOT_SurfaceSuiteCurrent  *srf = suites.SurfaceSuiteCurrent();
	DRAWBOT_PathSuiteCurrent     *pth = suites.PathSuiteCurrent();

	const float lx = (float)FIX_2_FLOAT(params[DOF_FOCUS_POINT]->u.td.x_value);
	const float ly = (float)FIX_2_FLOAT(params[DOF_FOCUS_POINT]->u.td.y_value);
	float cx, cy;
	LayerToFrame(in_data, ex, lx, ly, cx, cy);
	const float s = 22.f, k = 8.f;   /* 枠の半分の大きさとかぎの長さ (画面のピクセル) */

	DRAWBOT_PathRef path = NULL;
	if (sup->NewPath(supplier, &path) != kSPNoError || !path) return err;
	for (int i = 0; i < 4; ++i) {
		const float sx = (i & 1) ? 1.f : -1.f, sy = (i & 2) ? 1.f : -1.f;
		pth->MoveTo(path, cx + sx * s, cy + sy * (s - k));
		pth->LineTo(path, cx + sx * s, cy + sy * s);
		pth->LineTo(path, cx + sx * (s - k), cy + sy * s);
	}
	DRAWBOT_ColorRGBA shade, col;
	shade.red = shade.green = shade.blue = 0.f; shade.alpha = 0.6f;
	col.red = 0.45f; col.green = 1.f; col.blue = 0.55f; col.alpha = 1.f;
	DRAWBOT_PenRef under = NULL, pen = NULL;
	if (sup->NewPen(supplier, &shade, 3.5f, &under) == kSPNoError && under) {
		srf->StrokePath(surface, under, path);
		sup->ReleaseObject((DRAWBOT_ObjectRef)under);
	}
	if (sup->NewPen(supplier, &col, 1.5f, &pen) == kSPNoError && pen) {
		srf->StrokePath(surface, pen, path);
		sup->ReleaseObject((DRAWBOT_ObjectRef)pen);
	}
	sup->ReleaseObject((DRAWBOT_ObjectRef)path);

	DRAWBOT_Boolean hasText = FALSE;
	sup->SupportsText(supplier, &hasText);
	if (hasText) {
		float size = 11.f;
		sup->GetDefaultFontSize(supplier, &size);
		DRAWBOT_FontRef font = NULL;
		if (sup->NewDefaultFont(supplier, size, &font) == kSPNoError && font) {
			const wchar_t *label = L"Focus";
			DRAWBOT_UTF16Char text[8];
			for (int i = 0; i < 8; ++i) { text[i] = (DRAWBOT_UTF16Char)label[i]; if (!label[i]) break; }
			DRAWBOT_PointF32 org;
			org.x = cx + s + 5.f; org.y = cy - s + 10.f;
			DRAWBOT_BrushRef dark = NULL, brush = NULL;
			if (sup->NewBrush(supplier, &shade, &dark) == kSPNoError && dark) {
				DRAWBOT_PointF32 o2 = org;
				o2.x += 1.f; o2.y += 1.f;
				srf->DrawString(surface, dark, font, text, &o2, kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.f);
				sup->ReleaseObject((DRAWBOT_ObjectRef)dark);
			}
			if (sup->NewBrush(supplier, &col, &brush) == kSPNoError && brush) {
				srf->DrawString(surface, brush, font, text, &org, kDRAWBOT_TextAlignment_Left, kDRAWBOT_TextTruncation_None, 0.f);
				sup->ReleaseObject((DRAWBOT_ObjectRef)brush);
			}
			sup->ReleaseObject((DRAWBOT_ObjectRef)font);
		}
	}
	return err;
}

static PF_Err HandleEvent(PF_InData *in_data, PF_OutData *out_data, PF_ParamDef *params[], PF_EventExtra *ex)
{
	PF_Err err = PF_Err_NONE;
	if (!ex || !ex->contextH) return err;
	if (ex->e_type == PF_Event_DRAW && (*ex->contextH)->w_type == PF_Window_COMP) {
		try {
			err = DrawFocusGizmo(in_data, params, ex);
		} catch (...) {
			err = PF_Err_NONE;   /* 描けなくても本体の動作には影響させない */
		}
	}
	return err;
}

/* ------------------------------------------------------------------ */
/* エントリポイント                                                    */
/* ------------------------------------------------------------------ */

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite    *inSPBasicSuitePtr,
	const char      *inHostName,
	const char      *inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;   /* PF_REGISTER_EFFECT_EXT2 は result に書く */
	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		DOF_NAME,
		DOF_MATCH_NAME,
		DOF_CATEGORY,
		AE_RESERVED_INFO,
		"EffectMain",
		DOF_SUPPORT_URL);
	return result;
}

PF_Err EffectMain(
	PF_Cmd       cmd,
	PF_InData   *in_data,
	PF_OutData  *out_data,
	PF_ParamDef *params[],
	PF_LayerDef *output,
	void        *extra)
{
	PF_Err err = PF_Err_NONE;
	try {
		switch (cmd) {
		case PF_Cmd_ABOUT:
			err = About(in_data, out_data, params, output);
			break;
		case PF_Cmd_GLOBAL_SETUP:
			err = GlobalSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_PARAMS_SETUP:
			err = ParamsSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_SEQUENCE_SETUP:
			out_data->sequence_data = NULL;
			Auto_Request(in_data, false);
			break;
		case PF_Cmd_SEQUENCE_RESETUP:
			Auto_Request(in_data, false);
			break;
		case PF_Cmd_USER_CHANGED_PARAM: {
			const PF_UserChangedParamExtra *uc = (const PF_UserChangedParamExtra *)extra;
			if (!uc) break;
			if (uc->param_index == DOF_AUTO_GENERATE) {
				Auto_Request(in_data, true, (int)params[DOF_AUTO_ENGINE]->u.pd.value);
			} else if (uc->param_index == DOF_SUBJECT_GENERATE) {
				Auto_RequestSubject(in_data, true);
			} else if (uc->param_index == DOF_AUTO_CLEAN) {
				Auto_CleanCache(in_data);
			} else if (uc->param_index == DOF_FOCUS_MODE) {
				err = UpdateFocusUI(in_data, params);
			}
			break;
		}
		case PF_Cmd_UPDATE_PARAMS_UI:
			err = UpdateFocusUI(in_data, params);
			break;
		case PF_Cmd_EVENT:
			err = HandleEvent(in_data, out_data, params, (PF_EventExtra *)extra);
			break;
		case PF_Cmd_SMART_PRE_RENDER:
			err = PreRender(in_data, out_data, (PF_PreRenderExtra *)extra);
			break;
		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra *)extra, false);
			break;
		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra *)extra, true);
			break;
		case PF_Cmd_GPU_DEVICE_SETUP: {
			/* GPU 版は CUDA だけ。ほか (DirectX / OpenCL / Metal) のときは CPU 版で描く */
			PF_GPUDeviceSetupExtra *gx = (PF_GPUDeviceSetupExtra *)extra;
			out_data->out_flags2 = 0;
#ifdef DOF_HAS_CUDA
			if (gx && gx->input->what_gpu == PF_GPU_Framework_CUDA) out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
#endif
			Dof_Log(std::wstring(L"gpu: device setup framework=") + std::to_wstring(gx ? (long)gx->input->what_gpu : -1L) +
			        (out_data->out_flags2 ? L" (use CUDA)" : L" (CPU)"));
			break;
		}
		case PF_Cmd_GPU_DEVICE_SETDOWN:
			DofGPU_Release();
			break;
		default:
			break;
		}
	} catch (PF_Err &thrown_err) {
		err = thrown_err;
	} catch (...) {
		Dof_Log(L"EffectMain: exception in cmd=" + std::to_wstring((long)cmd));
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	return err;
}

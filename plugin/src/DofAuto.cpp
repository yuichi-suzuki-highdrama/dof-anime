/*
 * DofAuto.cpp
 * 深度の自動生成の実装。詳細は DofAuto.h を参照。
 * Relight Anime (RelightAuto.cpp) から写したもの。深度の作り方・キャッシュ・設定は Relight Anime と共通にしてあり、
 * 同じ素材なら Relight Anime で作った深度をそのまま使う (作り直さない)
 *
 * 流れ (タイマーでメインスレッドから少しずつ進める):
 *   Auto_Request (エフェクト適用 / ボタン) → ジョブをキューへ
 *     QUEUED     : レイヤー上の本エフェクトを調べ、Depth Layer 未設定なら素材を特定。
 *                  キャッシュがあれば IMPORTING。無ければ
 *                    engine=native  : EXTRACTING へ (既定。relight_depth.exe、Python 不要)
 *                    engine=comfyui : gen_passes.py を起動して RUNNING (旧方式)
 *     EXTRACTING : AE に素材を 1 フレームずつ描かせて frames.bin に書く (1 回あたり短く区切る)。
 *                  書き終えたら relight_depth.exe を起動して RUNNING
 *     RUNNING    : プロセス終了を待つ。manifest.json ができていれば IMPORTING。
 *                  CUDA で失敗したら AE 同梱の ONNX Runtime (DirectML) でやり直す
 *     IMPORTING  : 深度の連番をプロジェクトへ読み込み、コンポに非表示レイヤーとして追加し、結線
 */

#include "DofAuto.h"
#include "AE_GeneralPlug.h"
#include "AEGP_SuiteHandler.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shlobj.h>

#include <mutex>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>

/* ------------------------------------------------------------------ */
/* 設定                                                                 */
/* ------------------------------------------------------------------ */

struct AutoConfig {
	std::wstring engine  = L"native";   /* native (relight_depth.exe) / comfyui (gen_passes.py) */
	std::wstring cache   = L"C:\\dev\\RelightAEPlugin\\cache";
	/* native */
	std::wstring depthExe = L"C:\\dev\\RelightAEPlugin\\depth\\build\\relight_depth.exe";
	std::wstring encoder  = L"C:\\dev\\RelightAEPlugin\\models\\vda_small_encoder.onnx";
	std::wstring head     = L"C:\\dev\\RelightAEPlugin\\models\\vda_small_head_t32.onnx";
	std::wstring provider = L"cuda";   /* cuda / dml / cpu */
	std::wstring ort      = L"C:\\ComfyUI_windows_portable\\python_embeded\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime.dll";
	/* cuDNN の場所 (torch の lib など) は PC ごとに違うので、plugin/build.ps1 が設定ファイルに書く */
	std::wstring dllDirs  = L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin";
	/* comfyui (旧方式) */
	std::wstring python  = L"python";
	std::wstring script  = L"C:\\dev\\RelightAEPlugin\\tools\\gen_passes.py";
	std::wstring server  = L"http://127.0.0.1:8188";
	std::wstring extra;                 /* 追加引数 */
	/* remote (Depth Engine = Remote): 別の PC の深度サーバー (tools/depth_server)。gen_passes.py --remote で動画を送って深度を受け取る */
	/* 送り先は設定ファイル (%APPDATA%\SRLM\RelightAnime.ini、DoFAnime.ini) の remote_server。空ならこの PC で作る */
	std::wstring remoteServer;
	/* この PC で作る深度の、モデルに入れる短辺 (設定ファイルの depth_input_size)。既定 518 (VRAM の山 約 11GB)。
	   770 にすると髪の毛などが細かく出る (Small のままなので商用可。時間は約 2 倍、VRAM の山 約 21GB) */
	int depthInputSize = 518;
	/* キャラのマスク (Subject Layer) の自動生成: SAM 3.1 を ComfyUI の Python で動かす (tools/subject_mask.py) */
	std::wstring subjectPython = L"C:\\ComfyUI_windows_portable\\python_embeded\\python.exe";
	std::wstring subjectScript = L"C:\\dev\\DepthOfFieldAEPlugin\\tools\\subject_mask.py";
	std::wstring comfyuiDir    = L"C:\\ComfyUI_windows_portable\\ComfyUI";
	/* キャラのマスクの縁の仕上げ方 (設定の subject_matte で変える。tools/subject_mask.py の --matte):
	   depthcf (既定。マスクと深度を混ぜて直し、縁を元の絵の色から切り抜く) / cf (縁だけ) /
	   sam2matting (SAM2Matting。いちばんきれいだが CC BY-NC 4.0 = 営利目的では使えない) / none (ガイデッドフィルタだけ) */
	std::wstring subjectMatte  = L"depthcf";
	bool loaded = false;
};

static AutoConfig g_cfg;

static std::wstring trim(const std::wstring &s)
{
	size_t a = s.find_first_not_of(L" \t\r\n");
	size_t b = s.find_last_not_of(L" \t\r\n");
	if (a == std::wstring::npos) return L"";
	return s.substr(a, b - a + 1);
}

/* 設定ファイル 1 つを読む (key=value 形式)。無ければ何もしない */
static void LoadIni(const std::wstring &ini)
{
	FILE *f = _wfopen(ini.c_str(), L"rb");
	if (!f) return;
	std::string bytes;
	char buf[4096];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) bytes.append(buf, n);
	fclose(f);
	/* UTF-8 → UTF-16 */
	int wl = MultiByteToWideChar(CP_UTF8, 0, bytes.c_str(), (int)bytes.size(), NULL, 0);
	std::wstring text(wl, 0);
	MultiByteToWideChar(CP_UTF8, 0, bytes.c_str(), (int)bytes.size(), &text[0], wl);
	if (!text.empty() && text[0] == 0xFEFF) text.erase(0, 1);
	size_t pos = 0;
	while (pos < text.size()) {
		size_t nl = text.find(L'\n', pos);
		std::wstring line = trim(text.substr(pos, nl == std::wstring::npos ? std::wstring::npos : nl - pos));
		pos = (nl == std::wstring::npos) ? text.size() : nl + 1;
		if (line.empty() || line[0] == L'#' || line[0] == L';') continue;
		size_t eq = line.find(L'=');
		if (eq == std::wstring::npos) continue;
		std::wstring k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
		if (k == L"python") g_cfg.python = v;
		else if (k == L"script") g_cfg.script = v;
		else if (k == L"cache") g_cfg.cache = v;
		else if (k == L"server") g_cfg.server = v;
		else if (k == L"remote_server") g_cfg.remoteServer = v;
		else if (k == L"depth_input_size") { const int n = _wtoi(v.c_str()); if (n >= 252 && n <= 1400) g_cfg.depthInputSize = n; }
		else if (k == L"extra") g_cfg.extra = v;
		else if (k == L"engine") g_cfg.engine = v;
		else if (k == L"depth_exe") g_cfg.depthExe = v;
		else if (k == L"encoder") g_cfg.encoder = v;
		else if (k == L"head") g_cfg.head = v;
		else if (k == L"provider") g_cfg.provider = v;
		else if (k == L"ort") g_cfg.ort = v;
		else if (k == L"dll_dirs") g_cfg.dllDirs = v;
		else if (k == L"subject_python") g_cfg.subjectPython = v;
		else if (k == L"subject_script") g_cfg.subjectScript = v;
		else if (k == L"comfyui_dir") g_cfg.comfyuiDir = v;
		else if (k == L"subject_matte") {
			if (v == L"depthcf" || v == L"cf" || v == L"sam2matting" || v == L"none") g_cfg.subjectMatte = v;
		}
	}
}


/* 設定を読む。深度の作り方は Relight Anime と共通なので、まず %APPDATA%\SRLM\RelightAnime.ini を読み、
   %APPDATA%\SRLM\DoFAnime.ini があればその値で上書きする。どちらも無ければ既定値。
   キャッシュの場所が同じなら、Relight Anime で作った深度をそのまま使える */
static void LoadConfig()
{
	if (g_cfg.loaded) return;
	g_cfg.loaded = true;
	wchar_t appdata[MAX_PATH] = {0};
	if (FAILED(SHGetFolderPathW(NULL, CSIDL_APPDATA, NULL, 0, appdata))) return;
	LoadIni(std::wstring(appdata) + L"\\SRLM\\RelightAnime.ini");
	LoadIni(std::wstring(appdata) + L"\\SRLM\\DoFAnime.ini");
}

/* <cache>\dof_auto.log に追記する簡易ログ */
static void AutoLog(const std::wstring &msg)
{
	LoadConfig();
	CreateDirectoryW(g_cfg.cache.c_str(), NULL);
	std::wstring path = g_cfg.cache + L"\\dof_auto.log";
	FILE *f = _wfopen(path.c_str(), L"ab");
	if (!f) return;
	SYSTEMTIME st; GetLocalTime(&st);
	wchar_t ts[64];
	swprintf(ts, 64, L"%02d:%02d:%02d.%03d ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
	std::wstring line = ts + msg + L"\n";
	int n = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int)line.size(), NULL, 0, NULL, NULL);
	std::string u8(n, 0);
	WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int)line.size(), &u8[0], n, NULL, NULL);
	fwrite(u8.data(), 1, u8.size(), f);
	fclose(f);
}

static std::wstring ErrStr(A_Err e) { wchar_t b[32]; swprintf(b, 32, L"err=%ld", (long)e); return b; }

/* ------------------------------------------------------------------ */
/* ジョブ                                                               */
/* ------------------------------------------------------------------ */

enum JobState { JOB_QUEUED, JOB_EXTRACTING, JOB_RUNNING, JOB_IMPORTING, JOB_DONE, JOB_FAILED };

struct AutoJob {
	AEGP_CompH      compH;
	AEGP_LayerIDVal layerId;
	bool            force;
	JobState        state;
	std::wstring    srcPath;
	std::wstring    shotKey;
	std::wstring    cacheDir;
	HANDLE          proc;
	A_long          frames;        /* 書き出すフレーム数 */
	bool            isStill;
	/* フレーム書き出し (EXTRACTING) */
	AEGP_ItemH      itemH;
	double          fps;
	A_long          nextFrame;
	A_short         downsample;
	FILE           *framesFile;
	std::wstring    provider;      /* 実際に使う推論方式 (CUDA 失敗時は dml に切り替える) */
	DWORD           startTick;
	std::wstring    engine;        /* native / comfyui (設定の engine) / remote (Depth Engine = Remote) */
	std::wstring    remoteModel;   /* remote のときの深度のモデル (Fast = vits、Quality = vitl) */
	/* 依頼を受けた時刻。これから QUEUE_DELAY_MS たつまで手を付けない */
	DWORD           queuedTick = 0;
	/* キャラのマスクのジョブ (subject = true) */
	bool            subject = false;
	std::wstring    subjectDir;    /* cache\<素材>\subject_<点・コマ・種類> */
	double          pointX = 0.0, pointY = 0.0;   /* Focus Point (レイヤー座標 = 素材の画素) */
	A_long          promptFrame = 0;               /* 点を見るコマ */
	int             subjectType = 1;               /* 1: Anime Character  2: Person  3: Animal */
	/* 文字のマスクのジョブ (promptKind >= 0。0 = Sharp Layer、1 = Blur Layer、2 = Exclude Layer)。出力先は subjectDir */
	int             promptKind = -1;
	std::wstring    promptText;
	bool            promptAtPoint = false;
};

/* 依頼を受けてから、プロジェクトを書き換え始めるまでの待ち時間 (ミリ秒)。
   エフェクトを掛けた直後にスクリプトや操作が続いていると、その途中でアイドルフックが呼ばれることがある。
   そこでレイヤーを足すと AE が落ちた (スクリプトでエフェクトを掛けてすぐ値を書き換えたとき) ので、少し待つ */
static const DWORD QUEUE_DELAY_MS = 1500;

/* この PC で AE にフレームを描かせて relight_depth.exe で作るか */
static bool IsNativeJob(const AutoJob &job) { return job.engine != L"comfyui" && job.engine != L"remote"; }

static std::mutex            g_mtx;
static std::vector<AutoJob>  g_jobs;
static SPBasicSuite         *g_basic = NULL;
static AEGP_PluginID         g_pluginId = 0;
static bool                  g_hookRegistered = false;

static bool FileExists(const std::wstring &p)
{
	DWORD a = GetFileAttributesW(p.c_str());
	return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

static std::wstring ReadTextFile(const std::wstring &p)
{
	FILE *f = _wfopen(p.c_str(), L"rb");
	if (!f) return L"";
	std::string bytes;
	char buf[4096];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) bytes.append(buf, n);
	fclose(f);
	int wl = MultiByteToWideChar(CP_UTF8, 0, bytes.c_str(), (int)bytes.size(), NULL, 0);
	std::wstring text(wl, 0);
	MultiByteToWideChar(CP_UTF8, 0, bytes.c_str(), (int)bytes.size(), &text[0], wl);
	return text;
}

/* "key": value を雑に拾う (manifest.json / status.json 用) */
static std::wstring JsonValue(const std::wstring &json, const wchar_t *key)
{
	std::wstring k = std::wstring(L"\"") + key + L"\"";
	size_t p = json.find(k);
	if (p == std::wstring::npos) return L"";
	p = json.find(L':', p);
	if (p == std::wstring::npos) return L"";
	++p;
	while (p < json.size() && (json[p] == L' ' || json[p] == L'\t')) ++p;
	if (p < json.size() && json[p] == L'"') {
		size_t e = json.find(L'"', p + 1);
		return json.substr(p + 1, e == std::wstring::npos ? std::wstring::npos : e - p - 1);
	}
	size_t e = p;
	while (e < json.size() && json[e] != L',' && json[e] != L'}' && json[e] != L'\n') ++e;
	return trim(json.substr(p, e - p));
}

/* 素材を識別するキー: ファイル名の幹 + パス/サイズ/更新日時の FNV ハッシュ */
static std::wstring MakeShotKey(const std::wstring &path)
{
	unsigned long long h = 1469598103934665603ULL;
	auto mix = [&](const void *d, size_t n) {
		const unsigned char *b = (const unsigned char *)d;
		for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
	};
	mix(path.data(), path.size() * sizeof(wchar_t));
	WIN32_FILE_ATTRIBUTE_DATA fa;
	if (GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &fa)) {
		mix(&fa.nFileSizeLow, sizeof(fa.nFileSizeLow));
		mix(&fa.nFileSizeHigh, sizeof(fa.nFileSizeHigh));
		mix(&fa.ftLastWriteTime, sizeof(fa.ftLastWriteTime));
	}
	size_t s = path.find_last_of(L"\\/");
	std::wstring name = (s == std::wstring::npos) ? path : path.substr(s + 1);
	size_t dot = name.find_last_of(L'.');
	if (dot != std::wstring::npos) name = name.substr(0, dot);
	std::wstring stem;
	for (wchar_t c : name) {
		if ((c >= L'0' && c <= L'9') || (c >= L'A' && c <= L'Z') || (c >= L'a' && c <= L'z') || c == L'-' || c == L'_')
			stem += c;
		if (stem.size() >= 24) break;
	}
	if (stem.empty()) stem = L"src";
	wchar_t hex[20];
	swprintf(hex, 20, L"%08x", (unsigned)(h ^ (h >> 32)));
	return stem + L"_" + hex;
}

static void ReportInfo(AEGP_SuiteHandler &suites, const std::wstring &msg)
{
	try {
		suites.UtilitySuite6()->AEGP_ReportInfoUnicode(g_pluginId, (const A_UTF16Char *)msg.c_str());
	} catch (...) {}
}

static std::wstring MemHandleToWString(AEGP_SuiteHandler &suites, AEGP_MemHandle h)
{
	std::wstring out;
	if (!h) return out;
	A_UTF16Char *p = NULL;
	if (suites.MemorySuite1()->AEGP_LockMemHandle(h, (void **)&p) == A_Err_NONE && p) {
		out = (const wchar_t *)p;
		suites.MemorySuite1()->AEGP_UnlockMemHandle(h);
	}
	suites.MemorySuite1()->AEGP_FreeMemHandle(h);
	return out;
}

/* ------------------------------------------------------------------ */
/* エフェクトのストリーム操作                                           */
/* ------------------------------------------------------------------ */

/* レイヤー上の本エフェクトのうち、自動生成が必要なものを集める */
struct EffectRef {
	AEGP_EffectRefH effectH;
};

static bool EffectIsOurs(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH)
{
	AEGP_InstalledEffectKey key;
	if (suites.EffectSuite4()->AEGP_GetInstalledKeyFromLayerEffect(effectH, &key) != A_Err_NONE) return false;
	A_char match[AEGP_MAX_EFFECT_MATCH_NAME_SIZE] = {0};
	if (suites.EffectSuite4()->AEGP_GetEffectMatchName(key, match) != A_Err_NONE) return false;
	return std::strcmp(match, DOF_MATCH_NAME) == 0;
}

/* 英語名でストリームを探す (呼び出し側が Dispose する) */
static AEGP_StreamRefH FindStream(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name)
{
	A_long n = 0;
	if (suites.StreamSuite6()->AEGP_GetEffectNumParamStreams(effectH, &n) != A_Err_NONE) return NULL;
	for (A_long i = 1; i < n; ++i) {
		AEGP_StreamRefH sH = NULL;
		if (suites.StreamSuite6()->AEGP_GetNewEffectStreamByIndex(g_pluginId, effectH, i, &sH) != A_Err_NONE || !sH) continue;
		AEGP_MemHandle nameH = NULL;
		std::wstring sname;
		if (suites.StreamSuite6()->AEGP_GetStreamName(g_pluginId, sH, TRUE, &nameH) == A_Err_NONE)
			sname = MemHandleToWString(suites, nameH);
		if (sname == name) return sH;
		suites.StreamSuite6()->AEGP_DisposeStream(sH);
	}
	return NULL;
}

static double GetStreamOneD(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name, double dflt)
{
	AEGP_StreamRefH sH = FindStream(suites, effectH, name);
	if (!sH) return dflt;
	double v = dflt;
	AEGP_StreamValue2 val;
	AEFX_CLR_STRUCT(val);
	A_Time t = {0, 1};
	if (suites.StreamSuite6()->AEGP_GetNewStreamValue(g_pluginId, sH, AEGP_LTimeMode_LayerTime, &t, FALSE, &val) == A_Err_NONE) {
		v = val.val.one_d;
		suites.StreamSuite6()->AEGP_DisposeStreamValue(&val);
	}
	suites.StreamSuite6()->AEGP_DisposeStream(sH);
	return v;
}

static AEGP_LayerIDVal GetStreamLayerId(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name)
{
	AEGP_StreamRefH sH = FindStream(suites, effectH, name);
	if (!sH) return 0;
	AEGP_LayerIDVal id = 0;
	AEGP_StreamValue2 val;
	AEFX_CLR_STRUCT(val);
	A_Time t = {0, 1};
	if (suites.StreamSuite6()->AEGP_GetNewStreamValue(g_pluginId, sH, AEGP_LTimeMode_LayerTime, &t, FALSE, &val) == A_Err_NONE) {
		id = val.val.layer_id;
		suites.StreamSuite6()->AEGP_DisposeStreamValue(&val);
	}
	suites.StreamSuite6()->AEGP_DisposeStream(sH);
	return id;
}

static void SetStreamLayerId(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name, AEGP_LayerIDVal id)
{
	AEGP_StreamRefH sH = FindStream(suites, effectH, name);
	if (!sH) return;
	AEGP_StreamValue2 val;
	AEFX_CLR_STRUCT(val);
	val.streamH = sH;
	val.val.layer_id = id;
	suites.StreamSuite6()->AEGP_SetStreamValue(g_pluginId, sH, &val);
	suites.StreamSuite6()->AEGP_DisposeStream(sH);
}

static void SetStreamOneD(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name, double v)
{
	AEGP_StreamRefH sH = FindStream(suites, effectH, name);
	if (!sH) return;
	AEGP_StreamValue2 val;
	AEFX_CLR_STRUCT(val);
	val.streamH = sH;
	val.val.one_d = v;
	suites.StreamSuite6()->AEGP_SetStreamValue(g_pluginId, sH, &val);
	suites.StreamSuite6()->AEGP_DisposeStream(sH);
}

/* 結線済みの深度レイヤーの素材が見つからない (キャッシュを消した、移動した) か */
static bool DepthSourceMissing(AEGP_SuiteHandler &suites, AEGP_LayerH layerH, AEGP_LayerIDVal depthId)
{
	AEGP_CompH compH = NULL;
	AEGP_LayerH depthL = NULL;
	AEGP_ItemH itemH = NULL;
	AEGP_ItemFlags flags = 0;
	if (suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH) != A_Err_NONE || !compH) return false;
	if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(compH, depthId, &depthL) != A_Err_NONE || !depthL) return false;
	if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(depthL, &itemH) != A_Err_NONE || !itemH) return false;
	if (suites.ItemSuite9()->AEGP_GetItemFlags(itemH, &flags) != A_Err_NONE) return false;
	return (flags & AEGP_ItemFlag_MISSING) != 0;
}

/* 対象レイヤー上で、自動生成が必要な本エフェクトを列挙する */
static void CollectEffects(AEGP_SuiteHandler &suites, AEGP_LayerH layerH, bool force, std::vector<AEGP_EffectRefH> &out)
{
	A_long n = 0;
	if (suites.EffectSuite4()->AEGP_GetLayerNumEffects(layerH, &n) != A_Err_NONE) return;
	for (A_long i = 0; i < n; ++i) {
		AEGP_EffectRefH eH = NULL;
		if (suites.EffectSuite4()->AEGP_GetLayerEffectByIndex(g_pluginId, layerH, i, &eH) != A_Err_NONE || !eH) continue;
		bool keep = false;
		if (EffectIsOurs(suites, eH)) {
			bool autoOn = GetStreamOneD(suites, eH, L"Auto Passes", 1.0) >= 0.5;
			/* 照明は深度だけで成り立つので、深度レイヤーが未設定なら生成する */
			AEGP_LayerIDVal depthId = GetStreamLayerId(suites, eH, L"Depth Layer");
			/* 深度の素材が見つからない場合 (キャッシュを消した後に開いたプロジェクトなど) も作り直す */
			bool missing = autoOn && depthId != 0 && DepthSourceMissing(suites, layerH, depthId);
			if (missing) AutoLog(L"CollectEffects: depth footage is missing, regenerate");
			keep = force || (autoOn && (depthId == 0 || missing));
		}
		if (keep) out.push_back(eH);
		else suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	}
}

/* ------------------------------------------------------------------ */
/* 素材パスの取得                                                       */
/* ------------------------------------------------------------------ */

static std::wstring GetLayerSourcePath(AEGP_SuiteHandler &suites, AEGP_LayerH layerH, bool &isStill)
{
	isStill = false;
	AEGP_ItemH itemH = NULL;
	if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(layerH, &itemH) != A_Err_NONE || !itemH) return L"";
	AEGP_ItemType type = AEGP_ItemType_NONE;
	if (suites.ItemSuite9()->AEGP_GetItemType(itemH, &type) != A_Err_NONE || type != AEGP_ItemType_FOOTAGE) return L"";
	AEGP_FootageH footH = NULL;
	if (suites.FootageSuite5()->AEGP_GetMainFootageFromItem(itemH, &footH) != A_Err_NONE || !footH) return L"";
	AEGP_MemHandle pathH = NULL;
	if (suites.FootageSuite5()->AEGP_GetFootagePath(footH, 0, AEGP_FOOTAGE_MAIN_FILE_INDEX, &pathH) != A_Err_NONE) return L"";
	std::wstring path = MemHandleToWString(suites, pathH);
	A_Time dur = {0, 1};
	if (suites.ItemSuite9()->AEGP_GetItemDuration(itemH, &dur) == A_Err_NONE) isStill = (dur.value == 0);
	return path;
}

/* status.json が "running" で、その pid の python がまだ生きていればハンドルを返す */
static HANDLE FindRunningGenerator(const std::wstring &cacheDir)
{
	std::wstring st = ReadTextFile(cacheDir + L"\\status.json");
	if (JsonValue(st, L"state") != L"running") return NULL;
	DWORD pid = (DWORD)_wtol(JsonValue(st, L"pid").c_str());
	if (!pid) return NULL;
	HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
	if (!h) return NULL;
	DWORD code = 0;
	wchar_t image[MAX_PATH] = {0};
	DWORD len = MAX_PATH;
	bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
	std::wstring img = QueryFullProcessImageNameW(h, 0, image, &len) ? std::wstring(image) : L"";
	bool ours = img.find(L"python") != std::wstring::npos || img.find(L"relight_depth") != std::wstring::npos;
	if (alive && ours) return h;
	CloseHandle(h);
	return NULL;
}

/* ------------------------------------------------------------------ */
/* 外部プロセス起動                                                     */
/* ------------------------------------------------------------------ */

/* AE 本体のフォルダ (同梱の onnxruntime.dll / DirectML.dll がある) */
static std::wstring AeSupportDir()
{
	wchar_t exe[MAX_PATH] = {0};
	GetModuleFileNameW(NULL, exe, MAX_PATH);
	std::wstring p = exe;
	size_t s = p.find_last_of(L"\\/");
	return s == std::wstring::npos ? L"" : p.substr(0, s);
}

static std::wstring NativeCommand(const AutoJob &job)
{
	const bool dml = (job.provider == L"dml");
	/* DirectML は AE 同梱の ONNX Runtime を使う (CUDA 版の DLL が無くても動く) */
	std::wstring ort = dml ? AeSupportDir() + L"\\onnxruntime.dll" : g_cfg.ort;
	std::wstring cmd = L"\"" + g_cfg.depthExe + L"\""
		+ L" --frames \"" + job.cacheDir + L"\\frames.bin\""
		+ L" --png-dir \"" + job.cacheDir + L"\\depth\""
		+ L" --manifest \"" + job.cacheDir + L"\\manifest.json\""
		+ L" --status \"" + job.cacheDir + L"\\status.json\""
		+ L" --encoder \"" + g_cfg.encoder + L"\""
		+ L" --head \"" + g_cfg.head + L"\""
		+ L" --ort \"" + ort + L"\""
		+ L" --provider " + job.provider;
	/* 大きく作るときは、エンコーダに一度に通すフレームを減らして VRAM の山を抑える */
	if (g_cfg.depthInputSize != 518) cmd += L" --input-size " + std::to_wstring(g_cfg.depthInputSize) + (g_cfg.depthInputSize > 518 ? L" --chunk 2" : L"");
	if (!dml) {
		size_t pos = 0;
		while (pos <= g_cfg.dllDirs.size()) {
			size_t sc = g_cfg.dllDirs.find(L';', pos);
			std::wstring d = trim(g_cfg.dllDirs.substr(pos, sc == std::wstring::npos ? std::wstring::npos : sc - pos));
			if (!d.empty()) cmd += L" --dll-dir \"" + d + L"\"";
			if (sc == std::wstring::npos) break;
			pos = sc + 1;
		}
	}
	return cmd;
}

static HANDLE SpawnGenerator(const AutoJob &job, std::wstring &errOut)
{
	LoadConfig();
	CreateDirectoryW(g_cfg.cache.c_str(), NULL);
	CreateDirectoryW(job.cacheDir.c_str(), NULL);

	std::wstring cmd;
	if (job.engine == L"remote") {
		/* 別の PC の深度サーバー: 動画を送り、範囲の正規化と時間方向のならしまで済んだ深度を受け取る */
		cmd = L"\"" + g_cfg.python + L"\" \"" + g_cfg.script + L"\" --remote"
			+ L" --video \"" + job.srcPath + L"\""
			+ L" --shot \"" + job.shotKey + L"\""
			+ L" --cache \"" + g_cfg.cache + L"\""
			+ L" --server \"" + g_cfg.remoteServer + L"\""
			+ L" --depth-model \"" + job.remoteModel + L"\"";
	} else if (job.engine == L"comfyui") {
		cmd = L"\"" + g_cfg.python + L"\" \"" + g_cfg.script + L"\""
			+ L" --video \"" + job.srcPath + L"\""
			+ L" --shot \"" + job.shotKey + L"\""
			+ L" --cache \"" + g_cfg.cache + L"\""
			+ L" --server \"" + g_cfg.server + L"\"";
		if (!g_cfg.extra.empty()) cmd += L" " + g_cfg.extra;
	} else {
		cmd = NativeCommand(job);
	}
	AutoLog(L"spawn: " + cmd);

	/* 標準出力をログへ */
	SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
	std::wstring logPath = job.cacheDir + L"\\gen.log";
	HANDLE hLog = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	STARTUPINFOW si;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	if (hLog != INVALID_HANDLE_VALUE) {
		si.dwFlags = STARTF_USESTDHANDLES;
		si.hStdOutput = hLog;
		si.hStdError = hLog;
		si.hStdInput = NULL;
	}
	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));
	std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
	cmdBuf.push_back(0);
	/* 環境変数: Python 側の標準出力を UTF-8 にする */
	SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
	BOOL ok = CreateProcessW(NULL, cmdBuf.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
	if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
	if (!ok) {
		wchar_t buf[256];
		swprintf(buf, 256, L"深度推定を起動できませんでした (Win32 error %lu)\n", GetLastError());
		errOut = buf + cmd;
		return NULL;
	}
	CloseHandle(pi.hThread);
	/* 推論中も AE の操作を優先する */
	SetPriorityClass(pi.hProcess, BELOW_NORMAL_PRIORITY_CLASS);
	return pi.hProcess;
}

/* ------------------------------------------------------------------ */
/* フレーム書き出し (EXTRACTING)                                         */
/* ------------------------------------------------------------------ */

#pragma pack(push, 1)
struct FramesHeader {   /* relight_depth.cpp と同じ形 */
	char     magic[4];
	uint32_t version, width, height, count, channels;
	float    fps;
};
#pragma pack(pop)

/* SuiteHandler に無い版のスイートを直接取得する */
template <typename T>
static T *AcquireSuite(const char *name, int32_t version)
{
	const void *p = NULL;
	if (!g_basic || g_basic->AcquireSuite(name, version, &p) != kSPNoError) return NULL;
	return (T *)p;
}

/* 書き出しの準備: フレーム数・fps・縮小率を決め、frames.bin を開く */
static bool BeginExtract(AEGP_SuiteHandler &suites, AutoJob &job, AEGP_LayerH layerH, std::wstring &err)
{
	if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(layerH, &job.itemH) != A_Err_NONE || !job.itemH) {
		err = L"ソースを取得できません";
		return false;
	}
	AEGP_FootageInterp interp;
	AEFX_CLR_STRUCT(interp);
	suites.FootageSuite5()->AEGP_GetFootageInterpretation(job.itemH, FALSE, &interp);
	job.fps = interp.conform_fpsF > 0 ? interp.conform_fpsF : (interp.native_fpsF > 0 ? interp.native_fpsF : 24.0);
	A_Time dur = {0, 1};
	suites.ItemSuite9()->AEGP_GetItemDuration(job.itemH, &dur);
	double sec = dur.scale ? (double)dur.value / (double)dur.scale : 0.0;
	job.frames = job.isStill ? 1 : std::max<A_long>(1, (A_long)std::llround(sec * job.fps));
	A_long w = 0, h = 0;
	suites.ItemSuite9()->AEGP_GetItemDimensions(job.itemH, &w, &h);
	/* モデルは短辺 518 px で推定するので、短辺が 518 を下回らない範囲で縮小して描かせる */
	A_long shortSide = std::min(w, h);
	job.downsample = (A_short)std::max<A_long>(1, std::min<A_long>(8, shortSide / 518));
	job.nextFrame = 0;
	CreateDirectoryW(g_cfg.cache.c_str(), NULL);
	CreateDirectoryW(job.cacheDir.c_str(), NULL);
	/* 前回の状態ファイルが残っていると、進み具合の表示が古い値になる */
	DeleteFileW((job.cacheDir + L"\\status.json").c_str());
	std::wstring path = job.cacheDir + L"\\frames.bin";
	job.framesFile = _wfopen(path.c_str(), L"wb");
	if (!job.framesFile) { err = L"書き出せません: " + path; return false; }
	wchar_t msg[256];
	swprintf(msg, 256, L"extract: %ld frames @ %.3f fps, %ldx%ld, downsample %d", (long)job.frames, job.fps, (long)w, (long)h, job.downsample);
	AutoLog(msg);
	job.startTick = GetTickCount();
	return true;
}

/* 時間の許す範囲でフレームを描いて書き出す。全部書き終えたら true */
static bool ExtractSome(AEGP_SuiteHandler &suites, AutoJob &job, DWORD budgetMs, std::wstring &err)
{
	AEGP_RenderOptionsSuite4 *ro = AcquireSuite<AEGP_RenderOptionsSuite4>(kAEGPRenderOptionsSuite, kAEGPRenderOptionsSuiteVersion4);
	if (!ro) { err = L"RenderOptionsSuite4 を取得できません"; return false; }
	AEGP_RenderOptionsH optH = NULL;
	if (ro->AEGP_NewFromItem(g_pluginId, job.itemH, &optH) != A_Err_NONE || !optH) {
		g_basic->ReleaseSuite(kAEGPRenderOptionsSuite, kAEGPRenderOptionsSuiteVersion4);
		err = L"描画設定を作れません";
		return false;
	}
	ro->AEGP_SetWorldType(optH, AEGP_WorldType_8);
	ro->AEGP_SetDownsampleFactor(optH, job.downsample, job.downsample);

	const DWORD t0 = GetTickCount();
	std::vector<uint8_t> rgb;
	bool ok = true;
	while (job.nextFrame < job.frames && GetTickCount() - t0 < budgetMs) {
		/* フレームの中央の時刻で描く (端だと丸めで前のフレームになることがある) */
		const A_u_long scale = 100000;
		A_Time t;
		t.scale = scale;
		t.value = (A_long)std::llround(((double)job.nextFrame + 0.5) / job.fps * scale);
		if (job.isStill) t.value = 0;
		ro->AEGP_SetTime(optH, t);
		AEGP_FrameReceiptH receipt = NULL;
		if (suites.RenderSuite5()->AEGP_RenderAndCheckoutFrame(optH, NULL, NULL, &receipt) != A_Err_NONE || !receipt) {
			err = L"フレームを描けません"; ok = false; break;
		}
		AEGP_WorldH world = NULL;
		A_long w = 0, h = 0;
		A_u_long rowbytes = 0;
		PF_Pixel8 *base = NULL;
		suites.RenderSuite5()->AEGP_GetReceiptWorld(receipt, &world);
		suites.WorldSuite3()->AEGP_GetSize(world, &w, &h);
		suites.WorldSuite3()->AEGP_GetRowBytes(world, &rowbytes);
		suites.WorldSuite3()->AEGP_GetBaseAddr8(world, &base);
		if (job.nextFrame == 0) {
			FramesHeader fh;
			std::memcpy(fh.magic, "RLFR", 4);
			fh.version = 1; fh.width = (uint32_t)w; fh.height = (uint32_t)h;
			fh.count = (uint32_t)job.frames; fh.channels = 3; fh.fps = (float)job.fps;
			fwrite(&fh, sizeof(fh), 1, job.framesFile);
		}
		rgb.resize((size_t)w * h * 3);
		for (A_long y = 0; y < h && base; ++y) {
			const PF_Pixel8 *row = (const PF_Pixel8 *)((const char *)base + (size_t)y * rowbytes);
			uint8_t *d = &rgb[(size_t)y * w * 3];
			for (A_long x = 0; x < w; ++x) { d[0] = row[x].red; d[1] = row[x].green; d[2] = row[x].blue; d += 3; }
		}
		fwrite(rgb.data(), 1, rgb.size(), job.framesFile);
		suites.RenderSuite5()->AEGP_CheckinFrame(receipt);
		++job.nextFrame;
	}
	ro->AEGP_Dispose(optH);
	g_basic->ReleaseSuite(kAEGPRenderOptionsSuite, kAEGPRenderOptionsSuiteVersion4);
	if (!ok) return false;
	if (job.nextFrame >= job.frames) {
		fclose(job.framesFile);
		job.framesFile = NULL;
		wchar_t msg[128];
		swprintf(msg, 128, L"extract: done in %.1f s", (GetTickCount() - job.startTick) / 1000.0);
		AutoLog(msg);
		return true;
	}
	return true;
}

/* ------------------------------------------------------------------ */
/* 読み込みと結線                                                       */
/* ------------------------------------------------------------------ */

static AEGP_LayerH AddPassLayer(AEGP_SuiteHandler &suites, AutoJob &job, AEGP_LayerH srcLayerH,
                                const std::wstring &first, const wchar_t *layerName, std::wstring &err)
{
	if (!FileExists(first)) { err = L"パスが見つかりません: " + first; return NULL; }

	AEGP_FileSequenceImportOptions seq;
	AEFX_CLR_STRUCT(seq);
	seq.all_in_folderB = job.isStill ? FALSE : TRUE;
	seq.force_alphabeticalB = TRUE;
	seq.start_frameL = AEGP_ANY_FRAME;
	seq.end_frameL = AEGP_ANY_FRAME;

	AEGP_FootageH footH = NULL;
	A_Err e = suites.FootageSuite5()->AEGP_NewFootage(g_pluginId, (const A_UTF16Char *)first.c_str(), NULL,
	                                                  job.isStill ? NULL : &seq,
	                                                  AEGP_InterpretationStyle_NO_DIALOG_GUESS, NULL, &footH);
	if (e != A_Err_NONE || !footH) { err = L"フッテージを作成できません: " + first; return NULL; }

	AEGP_ProjectH projH = NULL;
	AEGP_ItemH rootH = NULL;
	suites.ProjSuite6()->AEGP_GetProjectByIndex(0, &projH);
	if (projH) suites.ProjSuite6()->AEGP_GetProjectRootFolder(projH, &rootH);

	AEGP_ItemH itemH = NULL;
	e = suites.FootageSuite5()->AEGP_AddFootageToProject(footH, rootH, &itemH);
	if (e != A_Err_NONE || !itemH) { err = L"プロジェクトへ追加できません: " + first; return NULL; }
	suites.ItemSuite9()->AEGP_SetItemName(itemH, (const A_UTF16Char *)layerName);

	/* フレームレートを元素材に合わせる */
	if (!job.isStill) {
		AEGP_ItemH srcItemH = NULL;
		AEGP_FootageInterp srcInterp, interp;
		AEFX_CLR_STRUCT(srcInterp);
		AEFX_CLR_STRUCT(interp);
		if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(srcLayerH, &srcItemH) == A_Err_NONE && srcItemH &&
		    suites.FootageSuite5()->AEGP_GetFootageInterpretation(srcItemH, FALSE, &srcInterp) == A_Err_NONE &&
		    suites.FootageSuite5()->AEGP_GetFootageInterpretation(itemH, FALSE, &interp) == A_Err_NONE) {
			double fps = srcInterp.conform_fpsF > 0 ? srcInterp.conform_fpsF : srcInterp.native_fpsF;
			if (fps > 0) {
				interp.conform_fpsF = fps;
				suites.FootageSuite5()->AEGP_SetFootageInterpretation(itemH, FALSE, &interp);
			}
		}
	}

	AEGP_LayerH newLayerH = NULL;
	e = suites.LayerSuite9()->AEGP_AddLayer(itemH, job.compH, &newLayerH);
	if (e != A_Err_NONE || !newLayerH) { err = L"レイヤーを追加できません"; return NULL; }
	suites.LayerSuite9()->AEGP_SetLayerName(newLayerH, (const A_UTF16Char *)layerName);
	suites.LayerSuite9()->AEGP_SetLayerFlag(newLayerH, AEGP_LayerFlag_VIDEO_ACTIVE, FALSE);
	suites.LayerSuite9()->AEGP_SetLayerFlag(newLayerH, AEGP_LayerFlag_SHY, TRUE);

	/* 元レイヤーの直下に置き、開始時刻を合わせる。
	   追加直後は新しいレイヤーが 0 番 (一番上) にあり、元レイヤーは 1 つ下がっている。
	   新しいレイヤーを元レイヤーの今の番号へ移すと、元レイヤーが 1 つ上がり、そのすぐ下に入る。
	   (以前は +1 していたため、2 枚だけのコンポで範囲外になり layer_indexL invalid が出た) */
	A_long srcIdx = 0;
	if (suites.LayerSuite9()->AEGP_GetLayerIndex(srcLayerH, &srcIdx) == A_Err_NONE && srcIdx > 0)
		suites.LayerSuite9()->AEGP_ReorderLayer(newLayerH, srcIdx);   /* srcIdx は必ず範囲内 (元レイヤーの今の番号) */
	A_Time offset = {0, 1};
	if (suites.LayerSuite9()->AEGP_GetLayerOffset(srcLayerH, &offset) == A_Err_NONE)
		suites.LayerSuite9()->AEGP_SetLayerOffset(newLayerH, &offset);
	return newLayerH;
}

/* パスの比較用 (小文字、区切りを \ に) */
static std::wstring NormPath(std::wstring p)
{
	for (wchar_t &c : p) { if (c == L'/') c = L'\\'; c = (wchar_t)towlower(c); }
	return p;
}

/* コンポの中に、同じ深度の連番を素材にしたレイヤーがあれば返す (Relight Anime が読み込んだ「Relight depth」など)。
   同じ素材に Relight Anime と DoF Anime を両方掛けたとき、深度のレイヤーを 2 枚にしない */
static AEGP_LayerH FindLayerWithSource(AEGP_SuiteHandler &suites, AEGP_CompH compH, const std::wstring &firstFile)
{
	const std::wstring want = NormPath(firstFile);
	A_long n = 0;
	if (suites.LayerSuite9()->AEGP_GetCompNumLayers(compH, &n) != A_Err_NONE) return NULL;
	for (A_long i = 0; i < n; ++i) {
		AEGP_LayerH l = NULL;
		AEGP_ItemH itemH = NULL;
		AEGP_ItemType type = AEGP_ItemType_NONE;
		AEGP_FootageH footH = NULL;
		if (suites.LayerSuite9()->AEGP_GetCompLayerByIndex(compH, i, &l) != A_Err_NONE || !l) continue;
		if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(l, &itemH) != A_Err_NONE || !itemH) continue;
		if (suites.ItemSuite9()->AEGP_GetItemType(itemH, &type) != A_Err_NONE || type != AEGP_ItemType_FOOTAGE) continue;
		if (suites.FootageSuite5()->AEGP_GetMainFootageFromItem(itemH, &footH) != A_Err_NONE || !footH) continue;
		AEGP_MemHandle pathH = NULL;
		if (suites.FootageSuite5()->AEGP_GetFootagePath(footH, 0, AEGP_FOOTAGE_MAIN_FILE_INDEX, &pathH) != A_Err_NONE) continue;
		if (NormPath(MemHandleToWString(suites, pathH)) == want) return l;
	}
	return NULL;
}

static bool ImportAndWire(AEGP_SuiteHandler &suites, AutoJob &job, std::wstring &err)
{
	AEGP_LayerH srcLayerH = NULL;
	if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &srcLayerH) != A_Err_NONE || !srcLayerH) {
		err = L"対象レイヤーが見つかりません (削除された可能性)";
		return false;
	}
	std::vector<AEGP_EffectRefH> effects;
	CollectEffects(suites, srcLayerH, job.force, effects);
	if (effects.empty()) return true;   /* 既に結線済みなど。何もしない */

	/* 作り直しのときは、以前に自動で追加した深度レイヤーを後で消す (名前で判別。手動のレイヤーと Relight Anime のレイヤーは残す) */
	std::vector<AEGP_LayerIDVal> oldIds;
	for (AEGP_EffectRefH eH : effects) {
		AEGP_LayerIDVal id = GetStreamLayerId(suites, eH, L"Depth Layer");
		if (id) oldIds.push_back(id);
	}

	std::wstring manifest = ReadTextFile(job.cacheDir + L"\\manifest.json");
	std::wstring fmt = JsonValue(manifest, L"format");
	const wchar_t *ext = (fmt == L"exr") ? L".exr" : L".png";

	AutoLog(L"import: start (effects=" + std::to_wstring(effects.size()) + L")");
	suites.UtilitySuite6()->AEGP_StartUndoGroup("DoF Anime: Auto Depth");
	/* 同じ深度の連番のレイヤーがコンポにあればそれを使う。無ければ読み込んで追加する */
	AEGP_LayerH depthL = job.force ? NULL : FindLayerWithSource(suites, job.compH, job.cacheDir + L"\\depth\\depth.00001" + ext);
	AutoLog(L"import: searched the comp");
	if (depthL) AutoLog(L"import: reuse the depth layer already in the comp");
	else depthL = AddPassLayer(suites, job, srcLayerH, job.cacheDir + L"\\depth\\depth.00001" + ext, L"DoF depth", err);
	bool ok = (depthL != NULL);
	AEGP_LayerIDVal depthId = 0;
	if (ok) {
		suites.LayerSuite9()->AEGP_GetLayerID(depthL, &depthId);
		AutoLog(L"import: layer added, wiring");
		for (AEGP_EffectRefH eH : effects) {
			SetStreamLayerId(suites, eH, L"Depth Layer", depthId);
			AutoLog(L"import: wired Depth Layer");
			/* PNG は変換無しで読めるので Linear、EXR は AE が変換するので sRGB decode */
			SetStreamOneD(suites, eH, L"Pass Encoding", (fmt == L"exr") ? 2.0 : 1.0);
		}
	}
	if (ok) {
		for (AEGP_LayerIDVal id : oldIds) {
			if (id == depthId) continue;
			AEGP_LayerH oldL = NULL;
			if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, id, &oldL) != A_Err_NONE || !oldL) continue;
			AEGP_MemHandle nameH = NULL;
			std::wstring name;
			if (suites.LayerSuite9()->AEGP_GetLayerName(g_pluginId, oldL, &nameH, NULL) == A_Err_NONE)
				name = MemHandleToWString(suites, nameH);
			if (name == L"DoF depth") {
				suites.LayerSuite9()->AEGP_DeleteLayer(oldL);
				AutoLog(L"import: removed old layer " + name);
			}
		}
	}
	suites.UtilitySuite6()->AEGP_EndUndoGroup();
	for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	return ok;
}

/* ------------------------------------------------------------------ */
/* キャラのマスク (Subject Layer) の自動生成                               */
/* ------------------------------------------------------------------ */

/* Subject Type の選択肢 → SAM 3.1 に渡す文字 */
static const wchar_t *SubjectText(int type)
{
	switch (type) {
	case 2: return L"person";
	case 3: return L"animal";
	default: return L"anime character";
	}
}

/* キャラのマスクを作る必要がある本エフェクトを集める (Auto Subject がオンで Subject Layer が未設定、または作り直し) */
static void CollectSubjectEffects(AEGP_SuiteHandler &suites, AEGP_LayerH layerH, bool force, std::vector<AEGP_EffectRefH> &out)
{
	A_long n = 0;
	if (suites.EffectSuite4()->AEGP_GetLayerNumEffects(layerH, &n) != A_Err_NONE) return;
	for (A_long i = 0; i < n; ++i) {
		AEGP_EffectRefH eH = NULL;
		if (suites.EffectSuite4()->AEGP_GetLayerEffectByIndex(g_pluginId, layerH, i, &eH) != A_Err_NONE || !eH) continue;
		bool keep = false;
		if (EffectIsOurs(suites, eH) && GetStreamLayerId(suites, eH, L"Depth Layer") != 0) {
			const bool autoOn = GetStreamOneD(suites, eH, L"Auto Subject", 0.0) >= 0.5;
			const AEGP_LayerIDVal sid = GetStreamLayerId(suites, eH, L"Subject Layer");
			const bool missing = autoOn && sid != 0 && DepthSourceMissing(suites, layerH, sid);
			keep = force || (autoOn && (sid == 0 || missing));
		}
		if (keep) out.push_back(eH);
		else suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	}
}

/* 2D ポイントのストリームの値 (レイヤー座標) を、レイヤーの時刻 t で読む */
static bool GetStreamPoint(AEGP_SuiteHandler &suites, AEGP_EffectRefH effectH, const wchar_t *name, const A_Time &t, double &x, double &y)
{
	AEGP_StreamRefH sH = FindStream(suites, effectH, name);
	if (!sH) return false;
	bool ok = false;
	AEGP_StreamValue2 val;
	AEFX_CLR_STRUCT(val);
	if (suites.StreamSuite6()->AEGP_GetNewStreamValue(g_pluginId, sH, AEGP_LTimeMode_LayerTime, &t, FALSE, &val) == A_Err_NONE) {
		x = val.val.two_d.x;
		y = val.val.two_d.y;
		ok = true;
		suites.StreamSuite6()->AEGP_DisposeStreamValue(&val);
	}
	suites.StreamSuite6()->AEGP_DisposeStream(sH);
	return ok;
}

/* 子プロセスを起動し、標準出力をログへ書く (失敗したら NULL) */
static HANDLE SpawnCommand(const std::wstring &cmd, const std::wstring &logPath, std::wstring &errOut)
{
	AutoLog(L"spawn: " + cmd);
	SECURITY_ATTRIBUTES sa = {sizeof(sa), NULL, TRUE};
	HANDLE hLog = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	STARTUPINFOW si;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	if (hLog != INVALID_HANDLE_VALUE) {
		si.dwFlags = STARTF_USESTDHANDLES;
		si.hStdOutput = hLog;
		si.hStdError = hLog;
		si.hStdInput = NULL;
	}
	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));
	std::vector<wchar_t> cmdBuf(cmd.begin(), cmd.end());
	cmdBuf.push_back(0);
	SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
	BOOL ok = CreateProcessW(NULL, cmdBuf.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi);
	if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
	if (!ok) {
		wchar_t buf[256];
		swprintf(buf, 256, L"起動できませんでした (Win32 error %lu)\n", GetLastError());
		errOut = buf + cmd;
		return NULL;
	}
	CloseHandle(pi.hThread);
	SetPriorityClass(pi.hProcess, BELOW_NORMAL_PRIORITY_CLASS);
	return pi.hProcess;
}

/* キャラのマスクのジョブを作る (深度のジョブから、または「Generate Subject」から) */
static AutoJob MakeSubjectJob(AEGP_CompH compH, AEGP_LayerIDVal layerId, bool force)
{
	AutoJob job;
	job.compH = compH;
	job.layerId = layerId;
	job.force = force;
	job.state = JOB_QUEUED;
	job.proc = NULL;
	job.frames = 0;
	job.isStill = false;
	job.itemH = NULL;
	job.fps = 24.0;
	job.nextFrame = 0;
	job.downsample = 1;
	job.framesFile = NULL;
	job.startTick = 0;
	job.subject = true;
	job.promptFrame = 0;
	job.pointX = job.pointY = 0.0;
	job.subjectType = 1;
	job.queuedTick = GetTickCount();
	return job;
}

/* 読み込んで Subject Layer に結線する */
static bool ImportSubject(AEGP_SuiteHandler &suites, AutoJob &job, std::wstring &err)
{
	AEGP_LayerH srcLayerH = NULL;
	if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &srcLayerH) != A_Err_NONE || !srcLayerH) {
		err = L"対象レイヤーが見つかりません (削除された可能性)";
		return false;
	}
	std::vector<AEGP_EffectRefH> effects;
	CollectSubjectEffects(suites, srcLayerH, job.force, effects);
	if (effects.empty()) return true;
	std::vector<AEGP_LayerIDVal> oldIds;
	for (AEGP_EffectRefH eH : effects) {
		AEGP_LayerIDVal id = GetStreamLayerId(suites, eH, L"Subject Layer");
		if (id) oldIds.push_back(id);
	}
	suites.UtilitySuite6()->AEGP_StartUndoGroup("DoF Anime: Auto Subject");
	const std::wstring first = job.subjectDir + L"\\subject.00001.png";
	AEGP_LayerH subjL = FindLayerWithSource(suites, job.compH, first);
	if (!subjL) subjL = AddPassLayer(suites, job, srcLayerH, first, L"DoF subject", err);
	bool ok = subjL != NULL;
	AEGP_LayerIDVal subjId = 0;
	if (ok) {
		suites.LayerSuite9()->AEGP_GetLayerID(subjL, &subjId);
		for (AEGP_EffectRefH eH : effects) {
			SetStreamLayerId(suites, eH, L"Subject Layer", subjId);
			SetStreamOneD(suites, eH, L"Subject Channel", 3.0);   /* 赤・緑・青 = キャラ 1・2・3 人目 */
		}
		for (AEGP_LayerIDVal id : oldIds) {
			if (id == subjId) continue;
			AEGP_LayerH oldL = NULL;
			if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, id, &oldL) != A_Err_NONE || !oldL) continue;
			AEGP_MemHandle nameH = NULL;
			std::wstring name;
			if (suites.LayerSuite9()->AEGP_GetLayerName(g_pluginId, oldL, &nameH, NULL) == A_Err_NONE)
				name = MemHandleToWString(suites, nameH);
			if (name == L"DoF subject") {
				suites.LayerSuite9()->AEGP_DeleteLayer(oldL);
				AutoLog(L"subject: removed old layer");
			}
		}
	}
	suites.UtilitySuite6()->AEGP_EndUndoGroup();
	for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	return ok;
}

/* キャラのマスクのジョブを 1 段階進める。続けるなら keep に戻す */
static void ProcessSubjectJob(AEGP_SuiteHandler &suites, AutoJob &job, std::vector<AutoJob> &keep)
{
	if (job.state == JOB_QUEUED) {
		AEGP_LayerH layerH = NULL;
		if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &layerH) != A_Err_NONE || !layerH) return;
		std::vector<AEGP_EffectRefH> effects;
		CollectSubjectEffects(suites, layerH, job.force, effects);
		if (effects.empty()) return;   /* 作る必要なし */
		/* 今の時刻の Focus Point と Subject Type (最初の 1 つのエフェクトのもの) */
		AEGP_ItemH compItem = NULL;
		A_Time compT = {0, 1}, layerT = {0, 1};
		if (suites.CompSuite11()->AEGP_GetItemFromComp(job.compH, &compItem) == A_Err_NONE && compItem)
			suites.ItemSuite9()->AEGP_GetItemCurrentTime(compItem, &compT);
		suites.LayerSuite9()->AEGP_ConvertCompToLayerTime(layerH, &compT, &layerT);
		GetStreamPoint(suites, effects[0], L"Focus Point", layerT, job.pointX, job.pointY);
		job.subjectType = (int)(GetStreamOneD(suites, effects[0], L"Subject Type", 1.0) + 0.5);
		for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);

		job.srcPath = GetLayerSourcePath(suites, layerH, job.isStill);
		if (job.srcPath.empty()) return;
		AEGP_ItemH itemH = NULL;
		if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(layerH, &itemH) == A_Err_NONE && itemH) {
			AEGP_FootageInterp interp;
			AEFX_CLR_STRUCT(interp);
			if (suites.FootageSuite5()->AEGP_GetFootageInterpretation(itemH, FALSE, &interp) == A_Err_NONE)
				job.fps = interp.conform_fpsF > 0 ? interp.conform_fpsF : (interp.native_fpsF > 0 ? interp.native_fpsF : 24.0);
		}
		const double sec = layerT.scale ? (double)layerT.value / (double)layerT.scale : 0.0;
		job.promptFrame = job.isStill ? 0 : std::max<A_long>(0, (A_long)std::llround(sec * job.fps));
		LoadConfig();
		job.shotKey = MakeShotKey(job.srcPath);
		job.cacheDir = g_cfg.cache + L"\\" + job.shotKey;
		/* 同じ点・コマ・種類なら同じマスク (キャッシュを使う) */
		wchar_t tag[96];
		/* マスクの作り方を変えたら名前の番号を上げる (2: 縁を線画に寄せる・穴を埋める、3: 縁の仕上げ方 (subject_matte) を名前に入れる、4: 追跡した全コマを詰め直し、縁の薄い値を消す) */
		swprintf(tag, 96, L"subjects4_%d_%d_f%ld_t%d_%ls", (int)std::lround(job.pointX), (int)std::lround(job.pointY),
		         (long)job.promptFrame, job.subjectType, g_cfg.subjectMatte.c_str());
		job.subjectDir = job.cacheDir + L"\\" + tag;
		AutoLog(L"subject: " + job.subjectDir);
		if (FileExists(job.subjectDir + L"\\manifest.json")) {
			job.state = JOB_IMPORTING;
		} else {
			if (!FileExists(g_cfg.subjectScript) || !FileExists(g_cfg.subjectPython)) {
				AutoLog(L"subject: script or python not found");
				ReportInfo(suites, L"DoF Anime: キャラのマスクを自動で作る準備がありません (SAM 3.1 を動かす Python)。\n"
				                   L"設定の subject_python / subject_script / comfyui_dir を確かめるか、Subject Layer を手で指定してください。");
				return;
			}
			CreateDirectoryW(g_cfg.cache.c_str(), NULL);
			CreateDirectoryW(job.cacheDir.c_str(), NULL);
			CreateDirectoryW(job.subjectDir.c_str(), NULL);
			wchar_t pt[64];
			swprintf(pt, 64, L"%.1f,%.1f", job.pointX, job.pointY);
			const std::wstring cmd = L"\"" + g_cfg.subjectPython + L"\" -I \"" + g_cfg.subjectScript + L"\""
				+ L" --video \"" + job.srcPath + L"\""
				+ L" --out-dir \"" + job.subjectDir + L"\""
				+ L" --point " + pt
				+ L" --frame " + std::to_wstring((long)job.promptFrame)
				+ L" --text \"" + SubjectText(job.subjectType) + L"\""
				+ L" --comfyui \"" + g_cfg.comfyuiDir + L"\""
				+ L" --matte " + g_cfg.subjectMatte;
			std::wstring err;
			job.proc = SpawnCommand(cmd, job.subjectDir + L"\\gen.log", err);
			if (!job.proc) { ReportInfo(suites, L"DoF Anime: " + err); return; }
			job.state = JOB_RUNNING;
			keep.push_back(job);
			return;
		}
	}
	if (job.state == JOB_RUNNING) {
		if (WaitForSingleObject(job.proc, 0) != WAIT_OBJECT_0) { keep.push_back(job); return; }
		DWORD code = 1;
		GetExitCodeProcess(job.proc, &code);
		CloseHandle(job.proc);
		job.proc = NULL;
		AutoLog(L"subject: exited code=" + std::to_wstring((long)code));
		if (code != 0 || !FileExists(job.subjectDir + L"\\manifest.json")) {
			std::wstring msg = JsonValue(ReadTextFile(job.subjectDir + L"\\status.json"), L"message");
			ReportInfo(suites, L"DoF Anime: キャラのマスクを作れませんでした。Subject Layer を手で指定することもできます。\n" + msg +
			                   L"\n詳細: " + job.subjectDir + L"\\gen.log");
			return;
		}
		job.state = JOB_IMPORTING;
	}
	if (job.state == JOB_IMPORTING) {
		std::wstring err;
		const bool ok = ImportSubject(suites, job, err);
		AutoLog(L"subject: import " + std::wstring(ok ? L"ok" : L"failed: ") + err);
		if (!ok) ReportInfo(suites, L"DoF Anime: キャラのマスクの読み込みに失敗しました。\n" + err);
	}
}

/* ------------------------------------------------------------------ */
/* 文字のマスク (Sharp Layer / Blur Layer) の生成                         */
/* ------------------------------------------------------------------ */

static const wchar_t *PromptLayerParam(int kind) { return kind == 0 ? L"Sharp Layer" : (kind == 1 ? L"Blur Layer" : L"Exclude Layer"); }
static const wchar_t *PromptLayerName(int kind) { return kind == 0 ? L"DoF sharp" : (kind == 1 ? L"DoF blur" : L"DoF exclude"); }

/* 文字のマスクのスクリプト: キャラのマスクのスクリプトと同じフォルダの prompt_mask.py */
static std::wstring PromptScript()
{
	std::wstring s = g_cfg.subjectScript;
	const size_t p = s.find_last_of(L"\\/");
	return (p == std::wstring::npos ? std::wstring() : s.substr(0, p + 1)) + L"prompt_mask.py";
}

/* 文字の短い指紋 (キャッシュの名前用。FNV-1a) */
static unsigned int TextHash(const std::wstring &t)
{
	unsigned int h = 2166136261u;
	for (wchar_t c : t) { h ^= (unsigned int)towlower(c); h *= 16777619u; }
	return h;
}

/* このレイヤーの本エフェクトを集める */
static void CollectOurEffects(AEGP_SuiteHandler &suites, AEGP_LayerH layerH, std::vector<AEGP_EffectRefH> &out)
{
	A_long n = 0;
	if (suites.EffectSuite4()->AEGP_GetLayerNumEffects(layerH, &n) != A_Err_NONE) return;
	for (A_long i = 0; i < n; ++i) {
		AEGP_EffectRefH eH = NULL;
		if (suites.EffectSuite4()->AEGP_GetLayerEffectByIndex(g_pluginId, layerH, i, &eH) != A_Err_NONE || !eH) continue;
		if (EffectIsOurs(suites, eH)) out.push_back(eH);
		else suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	}
}

/* 読み込んで Sharp Layer / Blur Layer に結線する (前に作った同じ種類のレイヤーは消す) */
static bool ImportPrompt(AEGP_SuiteHandler &suites, AutoJob &job, std::wstring &err)
{
	AEGP_LayerH srcLayerH = NULL;
	if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &srcLayerH) != A_Err_NONE || !srcLayerH) {
		err = L"対象レイヤーが見つかりません (削除された可能性)";
		return false;
	}
	std::vector<AEGP_EffectRefH> effects;
	CollectOurEffects(suites, srcLayerH, effects);
	if (effects.empty()) return true;
	const wchar_t *param = PromptLayerParam(job.promptKind);
	std::vector<AEGP_LayerIDVal> oldIds;
	for (AEGP_EffectRefH eH : effects) {
		AEGP_LayerIDVal id = GetStreamLayerId(suites, eH, param);
		if (id) oldIds.push_back(id);
	}
	suites.UtilitySuite6()->AEGP_StartUndoGroup("DoF Anime: Text Mask");
	const std::wstring first = job.subjectDir + L"\\prompt.00001.png";
	AEGP_LayerH maskL = FindLayerWithSource(suites, job.compH, first);
	if (!maskL) maskL = AddPassLayer(suites, job, srcLayerH, first, PromptLayerName(job.promptKind), err);
	bool ok = maskL != NULL;
	AEGP_LayerIDVal maskId = 0;
	if (ok) {
		suites.LayerSuite9()->AEGP_GetLayerID(maskL, &maskId);
		for (AEGP_EffectRefH eH : effects) SetStreamLayerId(suites, eH, param, maskId);
		for (AEGP_LayerIDVal id : oldIds) {
			if (id == maskId) continue;
			AEGP_LayerH oldL = NULL;
			if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, id, &oldL) != A_Err_NONE || !oldL) continue;
			AEGP_MemHandle nameH = NULL;
			std::wstring name;
			if (suites.LayerSuite9()->AEGP_GetLayerName(g_pluginId, oldL, &nameH, NULL) == A_Err_NONE)
				name = MemHandleToWString(suites, nameH);
			if (name == PromptLayerName(job.promptKind)) {   /* 自分で作ったものだけ消す (手で描いた平面は残す) */
				suites.LayerSuite9()->AEGP_DeleteLayer(oldL);
				AutoLog(L"prompt: removed old layer");
			}
		}
	}
	suites.UtilitySuite6()->AEGP_EndUndoGroup();
	for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);
	return ok;
}

/* 文字のマスクのジョブを 1 段階進める。続けるなら keep に戻す */
static void ProcessPromptJob(AEGP_SuiteHandler &suites, AutoJob &job, std::vector<AutoJob> &keep)
{
	if (job.state == JOB_QUEUED) {
		AEGP_LayerH layerH = NULL;
		if (suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &layerH) != A_Err_NONE || !layerH) return;
		std::vector<AEGP_EffectRefH> effects;
		CollectOurEffects(suites, layerH, effects);
		if (effects.empty()) return;
		AEGP_ItemH compItem = NULL;
		A_Time compT = {0, 1}, layerT = {0, 1};
		if (suites.CompSuite11()->AEGP_GetItemFromComp(job.compH, &compItem) == A_Err_NONE && compItem)
			suites.ItemSuite9()->AEGP_GetItemCurrentTime(compItem, &compT);
		suites.LayerSuite9()->AEGP_ConvertCompToLayerTime(layerH, &compT, &layerT);
		GetStreamPoint(suites, effects[0], L"Focus Point", layerT, job.pointX, job.pointY);
		for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);

		job.srcPath = GetLayerSourcePath(suites, layerH, job.isStill);
		if (job.srcPath.empty()) {
			ReportInfo(suites, L"DoF Anime: このレイヤーのソースがファイルではないため、文字のマスクを作れません。");
			return;
		}
		AEGP_ItemH itemH = NULL;
		if (suites.LayerSuite9()->AEGP_GetLayerSourceItem(layerH, &itemH) == A_Err_NONE && itemH) {
			AEGP_FootageInterp interp;
			AEFX_CLR_STRUCT(interp);
			if (suites.FootageSuite5()->AEGP_GetFootageInterpretation(itemH, FALSE, &interp) == A_Err_NONE)
				job.fps = interp.conform_fpsF > 0 ? interp.conform_fpsF : (interp.native_fpsF > 0 ? interp.native_fpsF : 24.0);
		}
		const double sec = layerT.scale ? (double)layerT.value / (double)layerT.scale : 0.0;
		job.promptFrame = job.isStill ? 0 : std::max<A_long>(0, (A_long)std::llround(sec * job.fps));
		LoadConfig();
		job.shotKey = MakeShotKey(job.srcPath);
		job.cacheDir = g_cfg.cache + L"\\" + job.shotKey;
		/* 同じ文字・点・コマなら同じマスク (キャッシュを使う) */
		wchar_t tag[128];
		if (job.promptAtPoint)
			swprintf(tag, 128, L"prompt2_%08x_%d_%d_f%ld", TextHash(job.promptText), (int)std::lround(job.pointX), (int)std::lround(job.pointY), (long)job.promptFrame);
		else
			swprintf(tag, 128, L"prompt2_%08x_all_f%ld", TextHash(job.promptText), (long)job.promptFrame);
		job.subjectDir = job.cacheDir + L"\\" + tag;
		AutoLog(L"prompt: " + job.promptText + L" -> " + job.subjectDir);
		if (FileExists(job.subjectDir + L"\\manifest.json")) {
			job.state = JOB_IMPORTING;
		} else {
			const std::wstring script = PromptScript();
			if (!FileExists(script) || !FileExists(g_cfg.subjectPython)) {
				ReportInfo(suites, L"DoF Anime: 文字のマスクを作る準備がありません (SAM 3.1 を動かす Python と " + script + L")。");
				return;
			}
			CreateDirectoryW(g_cfg.cache.c_str(), NULL);
			CreateDirectoryW(job.cacheDir.c_str(), NULL);
			CreateDirectoryW(job.subjectDir.c_str(), NULL);
			std::wstring cmd = L"\"" + g_cfg.subjectPython + L"\" -I \"" + script + L"\""
				+ L" --video \"" + job.srcPath + L"\""
				+ L" --out-dir \"" + job.subjectDir + L"\""
				+ L" --text \"" + job.promptText + L"\""
				+ L" --frame " + std::to_wstring((long)job.promptFrame)
				+ L" --comfyui \"" + g_cfg.comfyuiDir + L"\"";
			if (job.promptAtPoint) {
				wchar_t pt[64];
				swprintf(pt, 64, L" --point %.1f,%.1f", job.pointX, job.pointY);
				cmd += pt;
			}
			std::wstring err;
			job.proc = SpawnCommand(cmd, job.subjectDir + L"\\gen.log", err);
			if (!job.proc) { ReportInfo(suites, L"DoF Anime: " + err); return; }
			job.state = JOB_RUNNING;
			keep.push_back(job);
			return;
		}
	}
	if (job.state == JOB_RUNNING) {
		if (WaitForSingleObject(job.proc, 0) != WAIT_OBJECT_0) { keep.push_back(job); return; }
		DWORD code = 1;
		GetExitCodeProcess(job.proc, &code);
		CloseHandle(job.proc);
		job.proc = NULL;
		AutoLog(L"prompt: exited code=" + std::to_wstring((long)code));
		if (code != 0 || !FileExists(job.subjectDir + L"\\manifest.json")) {
			std::wstring msg = JsonValue(ReadTextFile(job.subjectDir + L"\\status.json"), L"message");
			ReportInfo(suites, L"DoF Anime: 文字のマスクを作れませんでした。\n" + msg + L"\n詳細: " + job.subjectDir + L"\\gen.log");
			return;
		}
		job.state = JOB_IMPORTING;
	}
	if (job.state == JOB_IMPORTING) {
		std::wstring err;
		const bool ok = ImportPrompt(suites, job, err);
		AutoLog(L"prompt: import " + std::wstring(ok ? L"ok" : L"failed: ") + err);
		if (!ok) ReportInfo(suites, L"DoF Anime: 文字のマスクの読み込みに失敗しました。\n" + err);
	}
}

/* ------------------------------------------------------------------ */
/* アイドルフック                                                       */
/* ------------------------------------------------------------------ */

static bool g_processing = false;

/* キューのジョブを 1 段階ずつ進める。メインスレッドから呼ぶこと (タイマー / アイドルフック) */
static void ProcessJobs()
{
	if (g_processing) return;   /* 再入防止 (ReportInfo のモーダル中にタイマーが来る等) */
	std::vector<AutoJob> jobs;
	{
		std::lock_guard<std::mutex> lk(g_mtx);
		jobs.swap(g_jobs);
	}
	if (jobs.empty()) return;
	g_processing = true;

	AEGP_SuiteHandler suites(g_basic);
	/* 裏の処理で起きた AEGP のエラーで AE のエラー窓 (モーダル) を出させない。結果はログと通知で伝える */
	AEGP_ErrReportState errState;
	bool quiet = false;
	try { quiet = (suites.UtilitySuite6()->AEGP_StartQuietErrors(&errState) == A_Err_NONE); } catch (...) {}
	std::vector<AutoJob> keep;

	for (AutoJob &job : jobs) {
		try {
			if (job.state == JOB_QUEUED && GetTickCount() - job.queuedTick < QUEUE_DELAY_MS) { keep.push_back(job); continue; }
			if (job.promptKind >= 0) { ProcessPromptJob(suites, job, keep); continue; }
			if (job.subject) { ProcessSubjectJob(suites, job, keep); continue; }
			if (job.state == JOB_QUEUED) {
				AEGP_LayerH layerH = NULL;
				A_Err e = suites.LayerSuite9()->AEGP_GetLayerFromLayerID(job.compH, job.layerId, &layerH);
				if (e != A_Err_NONE || !layerH) { AutoLog(L"IdleHook: layer not found " + ErrStr(e)); continue; }
				std::vector<AEGP_EffectRefH> effects;
				CollectEffects(suites, layerH, job.force, effects);
				AutoLog(L"IdleHook: effects needing passes=" + std::to_wstring(effects.size()));
				if (effects.empty()) {
					/* 深度は結線済み。キャラのマスクがまだなら続けて作る */
					keep.push_back(MakeSubjectJob(job.compH, job.layerId, false));
					continue;
				}
				for (AEGP_EffectRefH eH : effects) suites.EffectSuite4()->AEGP_DisposeEffect(eH);

				job.srcPath = GetLayerSourcePath(suites, layerH, job.isStill);
				AutoLog(L"IdleHook: source=" + job.srcPath + (job.isStill ? L" (still)" : L""));
				if (job.srcPath.empty()) {
					ReportInfo(suites, L"DoF Anime: このレイヤーのソースがファイルではないため、パスを自動生成できません。\n"
					                   L"動画か画像のフッテージレイヤーに適用してください。");
					continue;
				}
				LoadConfig();
				job.shotKey = MakeShotKey(job.srcPath);
				job.cacheDir = g_cfg.cache + L"\\" + job.shotKey;
				bool cached = FileExists(job.cacheDir + L"\\manifest.json");
				AutoLog(L"IdleHook: cacheDir=" + job.cacheDir + (cached ? L" (cached)" : L""));
				/* 同じ素材を処理中のジョブがあれば、それに相乗りする (同じ素材を 2 重に推論しない) */
				const AutoJob *leader = NULL;
				for (const AutoJob &o : keep)
					if (o.cacheDir == job.cacheDir && (o.state == JOB_EXTRACTING || o.state == JOB_RUNNING)) leader = &o;
				for (const AutoJob &o : jobs)
					if (&o != &job && o.cacheDir == job.cacheDir && (o.state == JOB_EXTRACTING || o.state == JOB_RUNNING)) leader = &o;
				if (leader && leader->state == JOB_EXTRACTING) {
					AutoLog(L"IdleHook: same source is being extracted, wait");
					keep.push_back(job);   /* 次の回にもう一度見る */
					continue;
				}
				HANDLE running = NULL;
				if (leader && leader->proc)
					DuplicateHandle(GetCurrentProcess(), leader->proc, GetCurrentProcess(), &running, 0, FALSE, DUPLICATE_SAME_ACCESS);
				if (!running) running = FindRunningGenerator(job.cacheDir);
				if (running) {
					/* 前回のセッションなどで起動した生成がまだ動いている。二重起動せず完了を待つ */
					AutoLog(L"IdleHook: attach to running generator");
					job.proc = running;
					job.state = JOB_RUNNING;
				} else if (cached && !job.force) {
					job.state = JOB_IMPORTING;
				} else if (IsNativeJob(job)) {
					/* 既定: AE にフレームを描かせて relight_depth.exe に渡す */
					if (job.force) {
						DeleteFileW((job.cacheDir + L"\\manifest.json").c_str());
						DeleteFileW((job.cacheDir + L"\\status.json").c_str());
					}
					std::wstring err;
					if (!BeginExtract(suites, job, layerH, err)) {
						AutoLog(L"extract: " + err);
						ReportInfo(suites, L"DoF Anime: フレームを書き出せません。\n" + err);
						continue;
					}
					job.state = JOB_EXTRACTING;
				} else {
					std::wstring err;
					job.proc = SpawnGenerator(job, err);
					if (!job.proc) { AutoLog(L"IdleHook: spawn failed: " + err); ReportInfo(suites, L"DoF Anime: " + err); continue; }
					AutoLog(L"IdleHook: generator started");
					job.state = JOB_RUNNING;
				}
			}

			if (job.state == JOB_EXTRACTING) {
				std::wstring err;
				if (!ExtractSome(suites, job, 120, err)) {
					if (job.framesFile) { fclose(job.framesFile); job.framesFile = NULL; }
					AutoLog(L"extract failed: " + err);
					ReportInfo(suites, L"DoF Anime: フレームを書き出せません。\n" + err);
					continue;
				}
				if (job.nextFrame < job.frames) { keep.push_back(job); continue; }
				job.proc = SpawnGenerator(job, err);
				if (!job.proc) { AutoLog(L"spawn failed: " + err); ReportInfo(suites, L"DoF Anime: " + err); continue; }
				job.state = JOB_RUNNING;
			}

			if (job.state == JOB_RUNNING) {
				DWORD w = WaitForSingleObject(job.proc, 0);
				if (w != WAIT_OBJECT_0) { keep.push_back(job); continue; }
				DWORD code = 1;
				GetExitCodeProcess(job.proc, &code);
				CloseHandle(job.proc);
				job.proc = NULL;
				AutoLog(L"IdleHook: generator exited code=" + std::to_wstring((long)code));
				if (code != 0 || !FileExists(job.cacheDir + L"\\manifest.json")) {
					std::wstring st = ReadTextFile(job.cacheDir + L"\\status.json");
					std::wstring msg = JsonValue(st, L"message");
					/* 別の PC で作れなかったら (つながらない・止まっている)、この PC で作り直す */
					if (job.engine == L"remote") {
						AutoLog(L"remote failed (" + msg + L"), fall back to local GPU");
						ReportInfo(suites, L"DoF Anime: リモート (" + g_cfg.remoteServer + L") で深度を作れなかったので、"
						                   L"この PC の GPU で作り直します。\n" + msg);
						job.engine = g_cfg.engine;
						job.force = true;
						job.state = JOB_QUEUED;
						keep.push_back(job);
						continue;
					}
					/* CUDA で失敗したら、AE 同梱の ONNX Runtime (DirectML) でやり直す */
					if (IsNativeJob(job) && job.provider == L"cuda" &&
					    FileExists(job.cacheDir + L"\\frames.bin")) {
						AutoLog(L"cuda failed (" + msg + L"), retry with dml");
						job.provider = L"dml";
						std::wstring err;
						job.proc = SpawnGenerator(job, err);
						if (job.proc) { keep.push_back(job); continue; }
					}
					ReportInfo(suites, L"DoF Anime: 深度の生成に失敗しました。\n" + msg +
					                   L"\n詳細: " + job.cacheDir + L"\\gen.log");
					continue;
				}
				/* 成功したら一時的なフレームは消す (数百 MB になる) */
				DeleteFileW((job.cacheDir + L"\\frames.bin").c_str());
				job.state = JOB_IMPORTING;
			}

			if (job.state == JOB_IMPORTING) {
				std::wstring err;
				bool ok = ImportAndWire(suites, job, err);
				AutoLog(L"IdleHook: import " + std::wstring(ok ? L"ok" : L"failed: ") + err);
				if (!ok) ReportInfo(suites, L"DoF Anime: パスの読み込みに失敗しました。\n" + err);
				else keep.push_back(MakeSubjectJob(job.compH, job.layerId, false));   /* 続けてキャラのマスク */
				continue;
			}
		} catch (...) {
			AutoLog(L"IdleHook: exception (missing suite?)");
		}
	}

	{
		std::lock_guard<std::mutex> lk(g_mtx);
		for (AutoJob &j : keep) g_jobs.push_back(j);
	}
	if (quiet) {
		try { suites.UtilitySuite6()->AEGP_EndQuietErrors(FALSE, &errState); } catch (...) {}
	}
	g_processing = false;
}

/* 次に呼んでほしい間隔 (1/60 秒単位) */
static A_long DesiredSleep()
{
	std::lock_guard<std::mutex> lk(g_mtx);
	bool extracting = false, busy = false;
	for (const AutoJob &j : g_jobs) {
		if (j.state == JOB_EXTRACTING) extracting = true;
		else busy = true;
	}
	return extracting ? 1 : (busy ? 15 : 60);
}

static A_Err IdleHook(AEGP_GlobalRefcon, AEGP_IdleRefcon, A_long *max_sleepPL)
{
	ProcessJobs();
	*max_sleepPL = DesiredSleep();
	return A_Err_NONE;
}

/* 補助 AEGP (DoFAnimeHelper.aex) のアイドルフックから呼ばれる。ここが正式な AE の文脈 */
static DWORD g_helperTick = 0;

extern "C" DllExport A_Err DoFAnime_Idle(SPBasicSuite *basic, AEGP_PluginID id, A_long *max_sleepPL)
{
	if (!g_helperTick) AutoLog(L"helper: idle hook is active");
	g_helperTick = GetTickCount();
	if (!g_basic) g_basic = basic;
	try { ProcessJobs(); } catch (...) { g_processing = false; AutoLog(L"helper: exception"); }
	if (max_sleepPL) *max_sleepPL = DesiredSleep();
	return A_Err_NONE;
}

/* ------------------------------------------------------------------ */
/* 進み具合の窓 (深度の生成中だけ、AE の画面の右下に出す)                   */
/* ------------------------------------------------------------------ */

static HWND         g_progWnd = NULL;
static HWND         g_aeMainWnd = NULL;
static std::wstring g_progText;
static float        g_progValue = 0.f;   /* 0..1 */
static DWORD        g_progStart = 0;

static int Dpi(HWND h)
{
	typedef UINT (WINAPI *GetDpiForWindowFn)(HWND);
	static GetDpiForWindowFn fn = (GetDpiForWindowFn)GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow");
	UINT d = (fn && h) ? fn(h) : 96;
	return d ? (int)d : 96;
}

static LRESULT CALLBACK ProgWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
	switch (msg) {
	case WM_ERASEBKGND:
		return 1;
	case WM_NCHITTEST: {
		/* どこを掴んでも窓を動かせるようにする */
		LRESULT r = DefWindowProcW(hwnd, msg, wp, lp);
		return r == HTCLIENT ? HTCAPTION : r;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC dc = BeginPaint(hwnd, &ps);
		RECT rc;
		GetClientRect(hwnd, &rc);
		const int s = Dpi(hwnd);
		auto px = [s](int v) { return MulDiv(v, s, 96); };
		HBRUSH bg = CreateSolidBrush(RGB(36, 36, 38));
		FillRect(dc, &rc, bg);
		DeleteObject(bg);
		HFONT font = CreateFontW(-px(13), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Yu Gothic UI");
		HGDIOBJ old = SelectObject(dc, font);
		SetBkMode(dc, TRANSPARENT);
		SetTextColor(dc, RGB(232, 232, 232));
		RECT tr = {px(12), px(8), rc.right - px(12), px(28)};
		DrawTextW(dc, g_progText.c_str(), -1, &tr, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS | DT_VCENTER);
		RECT bar = {px(12), px(34), rc.right - px(12), px(44)};
		HBRUSH track = CreateSolidBrush(RGB(64, 64, 68));
		FillRect(dc, &bar, track);
		DeleteObject(track);
		RECT fill = bar;
		fill.right = bar.left + (LONG)((bar.right - bar.left) * std::max(0.f, std::min(1.f, g_progValue)));
		HBRUSH blue = CreateSolidBrush(RGB(70, 140, 255));
		FillRect(dc, &fill, blue);
		DeleteObject(blue);
		DWORD sec = g_progStart ? (GetTickCount() - g_progStart) / 1000 : 0;
		wchar_t buf[96];
		swprintf(buf, 96, L"%d%%    経過 %lu:%02lu", (int)(g_progValue * 100.f + 0.5f), sec / 60, sec % 60);
		SetTextColor(dc, RGB(160, 160, 165));
		RECT br = {px(12), px(48), rc.right - px(12), px(66)};
		DrawTextW(dc, buf, -1, &br, DT_LEFT | DT_SINGLELINE | DT_VCENTER);
		SelectObject(dc, old);
		DeleteObject(font);
		EndPaint(hwnd, &ps);
		return 0;
	}
	}
	return DefWindowProcW(hwnd, msg, wp, lp);
}

/* AE のメイン窓を Windows の機能で探す (AEGP_GetMainHWND はタイマーから呼ぶと
   「no current context」で失敗し、AE が固まる原因になった) */
static BOOL CALLBACK FindAeMainProc(HWND h, LPARAM lp)
{
	DWORD pid = 0;
	GetWindowThreadProcessId(h, &pid);
	if (pid != GetCurrentProcessId() || !IsWindowVisible(h) || GetWindow(h, GW_OWNER)) return TRUE;
	wchar_t title[256] = {0};
	GetWindowTextW(h, title, 256);
	if (wcsncmp(title, L"Adobe After Effects", 19) == 0) { *(HWND *)lp = h; return FALSE; }
	return TRUE;
}

static void ShowProgress(const std::wstring &text, float value)
{
	if (!g_aeMainWnd || !IsWindow(g_aeMainWnd)) {
		g_aeMainWnd = NULL;
		EnumWindows(FindAeMainProc, (LPARAM)&g_aeMainWnd);
	}
	if (!g_progWnd) {
		WNDCLASSW wc;
		ZeroMemory(&wc, sizeof(wc));
		wc.lpfnWndProc = ProgWndProc;
		wc.hInstance = GetModuleHandleW(NULL);
		wc.hCursor = LoadCursor(NULL, IDC_SIZEALL);
		wc.lpszClassName = L"SRLM_DoFAnime_Progress";
		RegisterClassW(&wc);
		/* AE のメイン窓に従属させる (AE の上に出るが、ほかのアプリの上には出ない)。フォーカスは奪わない */
		g_progWnd = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, wc.lpszClassName, L"DoF Anime",
		                            WS_POPUP | WS_BORDER, 0, 0, 10, 10, g_aeMainWnd, NULL, wc.hInstance, NULL);
		if (!g_progWnd) return;
		const int s = Dpi(g_aeMainWnd ? g_aeMainWnd : g_progWnd);
		const int w = MulDiv(380, s, 96), h = MulDiv(72, s, 96);
		RECT ae = {0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
		if (g_aeMainWnd) GetWindowRect(g_aeMainWnd, &ae);
		SetWindowPos(g_progWnd, NULL, ae.right - w - MulDiv(24, s, 96), ae.bottom - h - MulDiv(48, s, 96), w, h,
		             SWP_NOZORDER | SWP_NOACTIVATE);
	}
	if (!g_progStart) g_progStart = GetTickCount();
	g_progText = text;
	g_progValue = value;
	if (!IsWindowVisible(g_progWnd)) ShowWindow(g_progWnd, SW_SHOWNOACTIVATE);
	InvalidateRect(g_progWnd, NULL, FALSE);
}

static void HideProgress()
{
	if (g_progWnd && IsWindowVisible(g_progWnd)) ShowWindow(g_progWnd, SW_HIDE);
	g_progStart = 0;
}

/* 進行中のジョブから表示内容を決める */
static void UpdateProgress()
{
	std::wstring text;
	float value = 0.f;
	int active = 0;
	{
		std::lock_guard<std::mutex> lk(g_mtx);
		for (const AutoJob &j : g_jobs) {
			if (j.state != JOB_EXTRACTING && j.state != JOB_RUNNING && j.state != JOB_IMPORTING) continue;
			if (++active > 1) continue;
			if (j.promptKind >= 0) {
				std::wstring st = ReadTextFile(j.subjectDir + L"\\status.json");
				float p = (float)_wtof(JsonValue(st, L"progress").c_str());
				value = j.state == JOB_RUNNING ? std::max(0.f, std::min(1.f, p)) : 0.99f;
				text = j.state == JOB_RUNNING ? L"文字のマスクを作成中 (" + j.promptText + L")" : L"マスクを結線中";
			} else if (j.subject) {
				std::wstring st = ReadTextFile(j.subjectDir + L"\\status.json");
				float p = (float)_wtof(JsonValue(st, L"progress").c_str());
				value = j.state == JOB_RUNNING ? std::max(0.f, std::min(1.f, p)) : 0.99f;
				text = j.state == JOB_RUNNING ? L"キャラのマスクを作成中 (SAM 3.1)" : L"マスクを結線中";
			} else if (j.state == JOB_EXTRACTING) {
				/* 全体の 0〜30%: AE に素材を描かせて書き出す */
				value = 0.30f * (float)j.nextFrame / (float)std::max<A_long>(1, j.frames);
				text = L"素材を読み込み中 " + std::to_wstring(j.nextFrame) + L" / " + std::to_wstring(j.frames) + L" フレーム";
			} else if (j.state == JOB_RUNNING) {
				/* 全体の 30〜98%: 深度を推定 (status.json の progress) */
				std::wstring st = ReadTextFile(j.cacheDir + L"\\status.json");
				float p = (float)_wtof(JsonValue(st, L"progress").c_str());
				std::wstring m = JsonValue(st, L"message");
				value = 0.30f + 0.68f * std::max(0.f, std::min(1.f, p));
				text = m.empty() ? L"深度を推定中" : m;
				if (j.provider == L"dml") text += L" (DirectML)";
			} else {
				value = 0.99f;
				text = L"レイヤーを結線中";
			}
		}
	}
	if (!active) { HideProgress(); return; }
	if (active > 1) text += L"  (ほか " + std::to_wstring(active - 1) + L" 件)";
	ShowProgress(L"DoF Anime: " + text, value);
}

/* WM_TIMER は進み具合の窓の更新に使う。AEGP の処理は補助 AEGP のアイドルフック (正式な文脈) で行う。
   補助 AEGP が入っていない場合だけ、ジョブが 5 秒進まなければタイマーから処理する (旧方式。不安定) */
static HWND  g_timerWnd = NULL;
static DWORD g_pendingSince = 0;

static LRESULT CALLBACK TimerWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
	if (msg == WM_TIMER) {
		static bool inTimer = false;
		if (inTimer) return 0;   /* モーダルの窓の中で入れ子にならないように */
		inTimer = true;
		bool pending;
		{
			std::lock_guard<std::mutex> lk(g_mtx);
			pending = !g_jobs.empty();
		}
		if (!pending) g_pendingSince = 0;
		else if (!g_pendingSince) g_pendingSince = GetTickCount();
		/* 補助 AEGP が読み込まれているなら、呼ばれるまで待つ (タイマーから AEGP を呼ぶと「no current context」になる)。
		   補助 AEGP が入っていないときだけ、タイマーから処理する (旧方式。不安定) */
		if (pending && !g_helperTick && !GetModuleHandleW(L"DoFAnimeHelper.aex") && GetTickCount() - g_pendingSince > 5000) {
			static bool warned = false;
			if (!warned) { warned = true; AutoLog(L"timer: helper AEGP not found, fallback to timer processing"); }
			try { ProcessJobs(); } catch (...) { g_processing = false; AutoLog(L"Timer: exception"); }
		}
		try { UpdateProgress(); } catch (...) {}
		SetTimer(hwnd, 1, pending ? 250 : 700, NULL);
		inTimer = false;
		return 0;
	}
	return DefWindowProcW(hwnd, msg, wp, lp);
}

static void StartTimer()
{
	if (g_timerWnd) return;
	WNDCLASSW wc;
	ZeroMemory(&wc, sizeof(wc));
	wc.lpfnWndProc = TimerWndProc;
	wc.hInstance = GetModuleHandleW(NULL);
	wc.lpszClassName = L"SRLM_DoFAnime_Timer";
	RegisterClassW(&wc);   /* 2 回目以降は失敗するが問題ない */
	g_timerWnd = CreateWindowExW(0, wc.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, wc.hInstance, NULL);
	if (g_timerWnd) SetTimer(g_timerWnd, 1, 700, NULL);
	AutoLog(g_timerWnd ? L"StartTimer: ok" : L"StartTimer: CreateWindow failed");
}

/* ------------------------------------------------------------------ */
/* 公開関数                                                             */
/* ------------------------------------------------------------------ */

/* 描画側からのログ (複数の描画スレッドから呼ばれるので排他する) */
void Dof_Log(const std::wstring &msg)
{
	static std::mutex mtx;
	std::lock_guard<std::mutex> lk(mtx);
	AutoLog(msg);
}

void Auto_Init(PF_InData *in_data)
{
	if (g_hookRegistered) return;
	g_basic = in_data->pica_basicP;
	try {
		AEGP_SuiteHandler suites(g_basic);
		A_Err e = suites.UtilitySuite6()->AEGP_RegisterWithAEGP(NULL, DOF_NAME, &g_pluginId);
		AutoLog(L"Auto_Init: RegisterWithAEGP " + ErrStr(e) + L" id=" + std::to_wstring((long)g_pluginId));
		if (e != A_Err_NONE) return;
		e = suites.RegisterSuite5()->AEGP_RegisterIdleHook(g_pluginId, IdleHook, NULL);
		AutoLog(L"Auto_Init: RegisterIdleHook " + ErrStr(e));
		/* アイドルフックの可否に関わらずタイマーで処理する */
		StartTimer();
		g_hookRegistered = (g_timerWnd != NULL) || (e == A_Err_NONE);
	} catch (...) {
		AutoLog(L"Auto_Init: exception (missing suite?)");
	}
}

void Auto_Request(PF_InData *in_data, bool force, int engine)
{
	if (!g_hookRegistered) Auto_Init(in_data);
	if (!g_hookRegistered) { AutoLog(L"Auto_Request: hook not registered"); return; }
	try {
		AEGP_SuiteHandler suites(in_data->pica_basicP);
		AEGP_LayerH layerH = NULL;
		A_Err e = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layerH);
		if (e != A_Err_NONE || !layerH) { AutoLog(L"Auto_Request: GetEffectLayer " + ErrStr(e)); return; }
		AutoJob job;
		job.compH = NULL;
		job.layerId = 0;
		job.force = force;
		job.state = JOB_QUEUED;
		job.proc = NULL;
		job.frames = 0;
		job.isStill = false;
		job.itemH = NULL;
		job.fps = 24.0;
		job.nextFrame = 0;
		job.downsample = 1;
		job.framesFile = NULL;
		LoadConfig();
		job.provider = g_cfg.provider;
		job.engine = engine >= 2 ? L"remote" : g_cfg.engine;
		if (job.engine == L"remote" && g_cfg.remoteServer.empty()) {
			AutoLog(L"Auto_Request: remote_server が設定されていないので、この PC で作る");
			job.engine = g_cfg.engine;
		}
		job.remoteModel = engine == 3 ? L"video_depth_anything_vitl.pth" : L"video_depth_anything_vits.pth";
		job.startTick = 0;
		job.queuedTick = GetTickCount();
		if (suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &job.compH) != A_Err_NONE || !job.compH) return;
		if (suites.LayerSuite9()->AEGP_GetLayerID(layerH, &job.layerId) != A_Err_NONE) return;
		std::lock_guard<std::mutex> lk(g_mtx);
		/* 重複: レイヤー ID はプロジェクト内で一意。AEGP_CompH は呼ぶたびに値が変わることがあるので比べない */
		for (const AutoJob &j : g_jobs)
			if (j.layerId == job.layerId && j.state != JOB_DONE) {
				/* 生成中に「Generate Passes」が押されても、同じレイヤーのジョブを 2 重に積まない */
				if (force) AutoLog(L"Auto_Request: already in progress, ignore regenerate");
				return;
			}
		g_jobs.push_back(job);
		AutoLog(L"Auto_Request: queued layerId=" + std::to_wstring((long)job.layerId) + (force ? L" (force)" : L"") + L" engine=" + job.engine + (job.engine == L"remote" ? L" model=" + job.remoteModel : L""));
	} catch (...) {
		AutoLog(L"Auto_Request: exception");
	}
}

void Auto_RequestSubject(PF_InData *in_data, bool force)
{
	if (!g_hookRegistered) Auto_Init(in_data);
	if (!g_hookRegistered) return;
	try {
		AEGP_SuiteHandler suites(in_data->pica_basicP);
		AEGP_LayerH layerH = NULL;
		if (suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layerH) != A_Err_NONE || !layerH) return;
		AEGP_CompH compH = NULL;
		AEGP_LayerIDVal id = 0;
		if (suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH) != A_Err_NONE || !compH) return;
		if (suites.LayerSuite9()->AEGP_GetLayerID(layerH, &id) != A_Err_NONE) return;
		std::lock_guard<std::mutex> lk(g_mtx);
		for (const AutoJob &j : g_jobs)
			if (j.layerId == id && j.subject) { AutoLog(L"Auto_RequestSubject: already in progress"); return; }
		g_jobs.push_back(MakeSubjectJob(compH, id, force));
		AutoLog(L"Auto_RequestSubject: queued layerId=" + std::to_wstring((long)id) + (force ? L" (force)" : L""));
	} catch (...) {
		AutoLog(L"Auto_RequestSubject: exception");
	}
}

void Auto_RequestPrompt(PF_InData *in_data, int kind, const std::wstring &text, bool atPoint)
{
	if (!g_hookRegistered) Auto_Init(in_data);
	if (!g_hookRegistered) return;
	try {
		AEGP_SuiteHandler suites(in_data->pica_basicP);
		AEGP_LayerH layerH = NULL;
		if (suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layerH) != A_Err_NONE || !layerH) return;
		AEGP_CompH compH = NULL;
		AEGP_LayerIDVal id = 0;
		if (suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH) != A_Err_NONE || !compH) return;
		if (suites.LayerSuite9()->AEGP_GetLayerID(layerH, &id) != A_Err_NONE) return;
		AutoJob job = MakeSubjectJob(compH, id, true);
		job.subject = false;
		job.promptKind = kind;
		job.promptText = text;
		job.promptAtPoint = atPoint;
		std::lock_guard<std::mutex> lk(g_mtx);
		for (const AutoJob &j : g_jobs)
			if (j.layerId == id && j.promptKind == kind) { AutoLog(L"Auto_RequestPrompt: already in progress"); return; }
		g_jobs.push_back(job);
		AutoLog(L"Auto_RequestPrompt: queued kind=" + std::to_wstring(kind) + L" text=" + text);
	} catch (...) {
		AutoLog(L"Auto_RequestPrompt: exception");
	}
}

/* 文字を入れる窓 (メモリの中で組み立てたダイアログ) */
struct PromptDlgState {
	int kind;
	std::wstring text;
	bool atPoint;
};

static INT_PTR CALLBACK PromptDlgProc(HWND h, UINT msg, WPARAM wp, LPARAM lp)
{
	PromptDlgState *st = (PromptDlgState *)GetWindowLongPtrW(h, DWLP_USER);
	switch (msg) {
	case WM_INITDIALOG:
		SetWindowLongPtrW(h, DWLP_USER, lp);
		st = (PromptDlgState *)lp;
		SetWindowTextW(h, st->kind == 0 ? L"DoF Anime: くっきり残す物 (Sharp Layer)" :
		                  (st->kind == 1 ? L"DoF Anime: ぼかす物 (Blur Layer)" : L"DoF Anime: キャラのマスクから抜く物 (Exclude Layer)"));
		SetDlgItemTextW(h, 101, st->text.c_str());
		CheckDlgButton(h, 102, st->atPoint ? BST_CHECKED : BST_UNCHECKED);
		SetFocus(GetDlgItem(h, 101));
		SendDlgItemMessageW(h, 101, EM_SETSEL, 0, -1);
		return FALSE;
	case WM_COMMAND:
		if (LOWORD(wp) == IDOK && st) {
			wchar_t buf[512] = {0};
			GetDlgItemTextW(h, 101, buf, 512);
			std::wstring t = buf;
			t.erase(std::remove(t.begin(), t.end(), L'"'), t.end());   /* 命令行を壊さない */
			st->text = trim(t);
			st->atPoint = IsDlgButtonChecked(h, 102) == BST_CHECKED;
			EndDialog(h, st->text.empty() ? IDCANCEL : IDOK);
			return TRUE;
		}
		if (LOWORD(wp) == IDCANCEL) { EndDialog(h, IDCANCEL); return TRUE; }
		break;
	}
	return FALSE;
}

/* ダイアログの雛形をメモリに書く */
static void DlgAlign(std::vector<WORD> &v) { if (v.size() % 2) v.push_back(0); }   /* DWORD 境界 */
static void DlgStr(std::vector<WORD> &v, const wchar_t *s) { for (; *s; ++s) v.push_back((WORD)*s); v.push_back(0); }
static void DlgItem(std::vector<WORD> &v, DWORD style, short x, short y, short cx, short cy, WORD id, WORD cls, const wchar_t *title)
{
	DlgAlign(v);
	DLGITEMTEMPLATE it;
	it.style = style | WS_CHILD | WS_VISIBLE;
	it.dwExtendedStyle = 0;
	it.x = x; it.y = y; it.cx = cx; it.cy = cy; it.id = id;
	const WORD *p = (const WORD *)&it;
	v.insert(v.end(), p, p + sizeof(it) / 2);
	v.push_back(0xFFFF);
	v.push_back(cls);
	DlgStr(v, title);
	v.push_back(0);   /* 作成データなし */
}

static bool AskPrompt(int kind, std::wstring &text, bool &atPoint)
{
	std::vector<WORD> v;
	DLGTEMPLATE dt;
	dt.style = DS_MODALFRAME | DS_CENTER | DS_SETFONT | WS_POPUP | WS_CAPTION | WS_SYSMENU;
	dt.dwExtendedStyle = 0;
	dt.cdit = 5;
	dt.x = 0; dt.y = 0; dt.cx = 280; dt.cy = 100;
	const WORD *p = (const WORD *)&dt;
	v.insert(v.end(), p, p + sizeof(dt) / 2);
	v.push_back(0);   /* メニューなし */
	v.push_back(0);   /* 既定のクラス */
	DlgStr(v, L"DoF Anime");
	v.push_back(9);
	DlgStr(v, L"Segoe UI");
	DlgItem(v, SS_LEFT, 7, 7, 266, 26, 0xFFFF, 0x0082,
	        kind == 0 ? L"くっきり残したい物の名前を英語で入れてください (カンマで区切れば複数)。\n例: flower, grass"
	        : (kind == 1 ? L"ぼかしたい物の名前を英語で入れてください (カンマで区切れば複数)。\n例: window, bed"
	                     : L"キャラの手前にあってマスクから抜きたい物の名前を英語で入れてください。\n例: grass, leaves"));
	DlgItem(v, WS_BORDER | WS_TABSTOP | ES_AUTOHSCROLL, 7, 36, 266, 14, 101, 0x0081, L"");
	DlgItem(v, BS_AUTOCHECKBOX | WS_TABSTOP, 7, 56, 266, 12, 102, 0x0080, L"Focus Point の所の物を 1 つだけ (外すと見つかった物を全部)");
	DlgItem(v, BS_DEFPUSHBUTTON | WS_TABSTOP, 165, 77, 50, 16, IDOK, 0x0080, L"作る");
	DlgItem(v, BS_PUSHBUTTON | WS_TABSTOP, 223, 77, 50, 16, IDCANCEL, 0x0080, L"やめる");
	HWND owner = NULL;
	EnumWindows(FindAeMainProc, (LPARAM)&owner);
	PromptDlgState st = {kind, text, atPoint};
	const INT_PTR r = DialogBoxIndirectParamW(GetModuleHandleW(NULL), (LPCDLGTEMPLATEW)v.data(), owner, PromptDlgProc, (LPARAM)&st);
	if (r != IDOK) return false;
	text = st.text;
	atPoint = st.atPoint;
	return true;
}

void Auto_PromptButton(PF_InData *in_data, int kind)
{
	static std::wstring lastText[3];
	static bool lastAtPoint[3] = {false, false, false};
	const int k = kind < 0 ? 0 : (kind > 2 ? 2 : kind);
	std::wstring text = lastText[k];
	bool atPoint = lastAtPoint[k];
	if (!AskPrompt(k, text, atPoint)) return;
	lastText[k] = text;
	lastAtPoint[k] = atPoint;
	Auto_RequestPrompt(in_data, k, text, atPoint);
}

/* ------------------------------------------------------------------ */
/* キャッシュの整理                                                     */
/* ------------------------------------------------------------------ */

static std::wstring ToLower(std::wstring s)
{
	for (wchar_t &c : s) c = (wchar_t)towlower(c);
	return s;
}

static unsigned long long DirSize(const std::wstring &dir)
{
	unsigned long long total = 0;
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE) return 0;
	do {
		if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) total += DirSize(dir + L"\\" + fd.cFileName);
		else total += ((unsigned long long)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
	} while (FindNextFileW(h, &fd));
	FindClose(h);
	return total;
}

static bool DeleteDirTree(const std::wstring &dir)
{
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((dir + L"\\*").c_str(), &fd);
	if (h != INVALID_HANDLE_VALUE) {
		do {
			if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
			std::wstring p = dir + L"\\" + fd.cFileName;
			if (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
				/* リンクの先は消さない */
				if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) RemoveDirectoryW(p.c_str()); else DeleteFileW(p.c_str());
			} else if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				DeleteDirTree(p);
			} else {
				SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
				DeleteFileW(p.c_str());
			}
		} while (FindNextFileW(h, &fd));
		FindClose(h);
	}
	return RemoveDirectoryW(dir.c_str()) != 0;
}

void Auto_CleanCache(PF_InData *in_data)
{
	LoadConfig();
	const std::wstring root = g_cfg.cache;
	const std::wstring rootLower = ToLower(root) + L"\\";

	/* 1. このプロジェクトの素材が参照しているキャッシュ (cache\<素材キー>\...) */
	std::vector<std::wstring> used;
	try {
		AEGP_SuiteHandler suites(in_data->pica_basicP);
		AEGP_ProjectH projH = NULL;
		if (suites.ProjSuite6()->AEGP_GetProjectByIndex(0, &projH) == A_Err_NONE && projH) {
			AEGP_ItemH itemH = NULL;
			suites.ItemSuite9()->AEGP_GetFirstProjItem(projH, &itemH);
			while (itemH) {
				AEGP_ItemType type = AEGP_ItemType_NONE;
				AEGP_FootageH footH = NULL;
				if (suites.ItemSuite9()->AEGP_GetItemType(itemH, &type) == A_Err_NONE && type == AEGP_ItemType_FOOTAGE &&
				    suites.FootageSuite5()->AEGP_GetMainFootageFromItem(itemH, &footH) == A_Err_NONE && footH) {
					AEGP_MemHandle pathH = NULL;
					if (suites.FootageSuite5()->AEGP_GetFootagePath(footH, 0, AEGP_FOOTAGE_MAIN_FILE_INDEX, &pathH) == A_Err_NONE) {
						std::wstring p = ToLower(MemHandleToWString(suites, pathH));
						for (wchar_t &ch : p) if (ch == L'/') ch = L'\\';
						if (p.compare(0, rootLower.size(), rootLower) == 0) {
							std::wstring rest = p.substr(rootLower.size());
							used.push_back(rest.substr(0, rest.find(L'\\')));
						}
					}
				}
				AEGP_ItemH next = NULL;
				if (suites.ItemSuite9()->AEGP_GetNextProjItem(projH, itemH, &next) != A_Err_NONE) break;
				itemH = next;
			}
		}
	} catch (...) {
		AutoLog(L"CleanCache: exception while scanning project");
		return;   /* 使っているものが分からないまま消さない */
	}
	{
		/* 生成中のものも残す */
		std::lock_guard<std::mutex> lk(g_mtx);
		for (const AutoJob &j : g_jobs)
			if (!j.cacheDir.empty()) {
				std::wstring d = ToLower(j.cacheDir);
				used.push_back(d.substr(d.find_last_of(L'\\') + 1));
			}
	}

	/* 2. 消す候補 */
	struct Entry { std::wstring name; unsigned long long size; };
	std::vector<Entry> victims;
	unsigned long long keepSize = 0, victimSize = 0;
	int keepCount = 0;
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((root + L"\\*").c_str(), &fd);
	if (h != INVALID_HANDLE_VALUE) {
		do {
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
			if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
			unsigned long long sz = DirSize(root + L"\\" + fd.cFileName);
			if (std::find(used.begin(), used.end(), ToLower(fd.cFileName)) != used.end()) {
				++keepCount; keepSize += sz;
			} else {
				victims.push_back({fd.cFileName, sz});
				victimSize += sz;
			}
		} while (FindNextFileW(h, &fd));
		FindClose(h);
	}

	HWND owner = NULL;
	EnumWindows(FindAeMainProc, (LPARAM)&owner);
	auto gb = [](unsigned long long b) { wchar_t s[32]; swprintf(s, 32, L"%.2f GB", b / 1073741824.0); return std::wstring(s); };
	if (victims.empty()) {
		MessageBoxW(owner, (L"消せる深度キャッシュはありません。\n\nこのプロジェクトで使用中: " + std::to_wstring(keepCount) +
		                    L" 件 (" + gb(keepSize) + L")\n場所: " + root).c_str(), L"DoF Anime", MB_OK | MB_ICONINFORMATION);
		return;
	}
	std::wstring list;
	for (size_t i = 0; i < victims.size() && i < 12; ++i) list += L"  " + victims[i].name + L"\n";
	if (victims.size() > 12) list += L"  ほか " + std::to_wstring(victims.size() - 12) + L" 件\n";
	std::wstring msg = L"このプロジェクトで使っていない深度キャッシュが " + std::to_wstring(victims.size()) + L" 件 (" + gb(victimSize) +
		L") あります。削除しますか？\n\n" + list +
		L"\nこのプロジェクトで使用中の " + std::to_wstring(keepCount) + L" 件 (" + gb(keepSize) + L") は残します。\n"
		L"ほかのプロジェクトで使っていたキャッシュを消した場合は、そのプロジェクトを開いたときに自動で作り直します。\n\n場所: " + root;
	if (MessageBoxW(owner, msg.c_str(), L"DoF Anime - Clean Cache", MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2) != IDYES) return;

	int removed = 0;
	unsigned long long freed = 0;
	for (const Entry &e : victims) {
		if (DeleteDirTree(root + L"\\" + e.name)) { ++removed; freed += e.size; }
		else AutoLog(L"CleanCache: could not remove " + e.name);
	}
	AutoLog(L"CleanCache: removed " + std::to_wstring(removed) + L" dirs");
	MessageBoxW(owner, (std::to_wstring(removed) + L" 件 (" + gb(freed) + L") を削除しました。" +
	                    (removed < (int)victims.size() ? L"\n使用中などで消せなかったものがあります。" : L"")).c_str(),
	            L"DoF Anime", MB_OK | MB_ICONINFORMATION);
}

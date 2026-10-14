/*
 * DofAE.h
 * DoF Anime: 推定した深度から被写界深度 (ピントの外のぼけ) を付ける SmartFX エフェクト。
 *   - 画面上の点 (Focus Point) の深度にピントを合わせる。キーフレームでピント送りができる
 *   - 手前と奥を分けてぼかし、ピントの合った人物に背景がにじまないようにする
 *   - 明るい所は丸や多角形の玉ぼけになる
 * 深度の生成と結線は DofAuto (Relight Anime と同じ relight_depth.exe / 深度サーバーを使い、キャッシュも共有する)。
 */
#pragma once

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AEGP_SuiteHandler.h"
#include "AEFX_SuiteHelper.h"
#include "Smart_Utils.h"
#include "DofAE_Flags.h"
#include "DofCore.h"   /* 描画の設定 (DofParams) とぼかしの計算 (CPU と GPU で共通) */

#define DOF_DESCRIPTION "Depth of field from estimated depth for AI anime footage."

/* パラメータ index。順序を変えたら ParamsSetup も同じ順序で追加すること */
enum {
	DOF_INPUT = 0,

	/* --- Auto (深度の自動生成) --- */
	DOF_TOPIC_AUTO,
	DOF_AUTO_ENABLE,
	DOF_AUTO_ENGINE,          /* 1: Local GPU  2: Remote Fast  3: Remote Quality */
	DOF_AUTO_GENERATE,
	DOF_AUTO_CLEAN,
	DOF_TOPIC_AUTO_END,

	/* --- Focus --- */
	DOF_TOPIC_FOCUS,
	DOF_FOCUS_MODE,           /* 1: Point  2: Depth */
	DOF_FOCUS_POINT,
	DOF_FOCUS_PULL,           /* ピント送りにかける時間 (秒)。前後のコマのピントをならす */
	DOF_FOCUS_DEPTH,
	DOF_FOCUS_RANGE,
	DOF_SUBJECT_LAYER,        /* キャラのマスク (ロトブラシなど)。この中はくっきり残す */
	DOF_SUBJECT_CHANNEL,      /* 1: Alpha  2: Luminance */
	DOF_SUBJECT_FLATTEN,      /* マスクの中の深度を人物の代表の深度に寄せる */
	DOF_SUBJECT_AMOUNT,       /* Keep Subject Sharp */
	DOF_AUTO_SUBJECT,         /* 深度を作ったあと、キャラのマスクも自動で作る */
	DOF_SUBJECT_TYPE,         /* 1: Anime Character  2: Person  3: Animal */
	DOF_SUBJECT_GENERATE,     /* ボタン: 今の Focus Point でマスクを作り直す */
	DOF_SHARP_LAYER,          /* 手直し: このレイヤーのアルファの所は必ずくっきり */
	DOF_SHARP_PROMPT,         /* ボタン: 文字で Sharp Layer のマスクを作る */
	DOF_BLUR_LAYER,           /* 手直し: このレイヤーのアルファの所は画面のいちばん奥と同じだけぼかす */
	DOF_BLUR_PROMPT,          /* ボタン: 文字で Blur Layer のマスクを作る */
	DOF_TOPIC_FOCUS_END,

	/* --- Blur --- */
	DOF_TOPIC_BLUR,
	DOF_BLUR_SIZE,
	DOF_NEAR_BLUR,
	DOF_FAR_BLUR,
	DOF_EDGE_EXPAND,
	DOF_TOPIC_BLUR_END,

	/* --- Bokeh --- */
	DOF_TOPIC_BOKEH,
	DOF_SHAPE,
	DOF_SHAPE_ROTATION,
	DOF_HIGHLIGHTS,
	DOF_HIGHLIGHT_THRESHOLD,
	DOF_QUALITY,
	DOF_TOPIC_BOKEH_END,

	/* --- Passes --- */
	DOF_TOPIC_PASSES,
	DOF_DEPTH_LAYER,
	DOF_DEPTH_NEAR,
	DOF_DEPTH_STABILIZE,
	DOF_DEPTH_SNAP,
	DOF_PASS_ENCODING,
	DOF_INPUT_LINEAR,
	DOF_USE_GPU,
	DOF_TOPIC_PASSES_END,

	/* --- Output --- */
	DOF_OUTPUT_MODE,
	DOF_MIX,

	DOF_NUM_PARAMS
};

/* パラメータ ID (保存互換用。index と別に固定。既存の値は変えないこと) */
enum {
	ID_TOPIC_AUTO = 10,
	ID_AUTO_ENABLE,
	ID_AUTO_ENGINE,
	ID_AUTO_GENERATE,
	ID_AUTO_CLEAN,
	ID_TOPIC_AUTO_END,

	ID_TOPIC_FOCUS = 30,
	ID_FOCUS_MODE,
	ID_FOCUS_POINT,
	ID_FOCUS_DEPTH,
	ID_FOCUS_RANGE,
	ID_TOPIC_FOCUS_END,
	ID_SUBJECT_LAYER,
	ID_SUBJECT_CHANNEL,
	ID_SUBJECT_AMOUNT,
	ID_AUTO_SUBJECT,
	ID_SUBJECT_TYPE,
	ID_SUBJECT_GENERATE,
	ID_SUBJECT_FLATTEN,
	ID_FOCUS_PULL,
	ID_SHARP_LAYER,
	ID_BLUR_LAYER,
	ID_SHARP_PROMPT,
	ID_BLUR_PROMPT,

	ID_TOPIC_BLUR = 50,
	ID_BLUR_SIZE,
	ID_NEAR_BLUR,
	ID_FAR_BLUR,
	ID_EDGE_EXPAND,
	ID_TOPIC_BLUR_END,

	ID_TOPIC_BOKEH = 70,
	ID_SHAPE,
	ID_SHAPE_ROTATION,
	ID_HIGHLIGHTS,
	ID_HIGHLIGHT_THRESHOLD,
	ID_QUALITY,
	ID_TOPIC_BOKEH_END,

	ID_TOPIC_PASSES = 90,
	ID_DEPTH_LAYER,
	ID_DEPTH_NEAR,
	ID_DEPTH_STABILIZE,
	ID_DEPTH_SNAP,
	ID_PASS_ENCODING,
	ID_INPUT_LINEAR,
	ID_USE_GPU,
	ID_TOPIC_PASSES_END,

	ID_OUTPUT_MODE = 110,
	ID_MIX
};

/* PreRender → SmartRender に渡す情報 */
struct PreRenderData {
	A_long   originX;   /* 出力ワールド左上のレイヤー座標 (ダウンサンプル後) */
	A_long   originY;
	PF_LRect depthRect; /* 深度レイヤーは全体を取り出す。その範囲 (深度レイヤー自身の座標) */
	bool     depthChecked;
	PF_LRect subjectRect; /* キャラのマスクのレイヤーも全体を取り出す */
	bool     subjectChecked;
	PF_LRect fixRect[2];  /* 手直しのレイヤー (Sharp Layer, Blur Layer) も全体を取り出す */
	bool     fixChecked[2];
	/* 深度を時間方向にならすための前後のフレーム */
	int      nbCount;
	int      nbOff[4];
	bool     nbChecked[4];
};

extern "C" {
	DllExport PF_Err EffectMain(
		PF_Cmd        cmd,
		PF_InData    *in_data,
		PF_OutData   *out_data,
		PF_ParamDef  *params[],
		PF_LayerDef  *output,
		void         *extra);
}

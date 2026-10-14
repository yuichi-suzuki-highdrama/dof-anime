/*
 * DofGPU.h
 * GPU 版 (CUDA) の入口。中身は DofGPU.cu。
 * 計算は DofCore.h の関数をそのまま GPU で動かすので、CPU 版と同じ絵になる。
 * このヘッダは CUDA のヘッダを取り込まない (CPU 側のコンパイラからも使えるように)。
 */
#pragma once

#include "DofCore.h"

struct DfGpuFrame {
	/* 呼ぶ側が入れるもの: ctx.p, outW, outH, originX, originY, layerW, layerH。ほかは DofGPU_Render が埋める */
	DofContext ctx;
	DfImage    src;          /* 入力 (GPU のワールド、BGRA float) */
	DfAuxMap   depthMap;     /* 深度ワールドと位置合わせ (img.data が NULL なら深度なし) */
	DfAuxMap   subjectMap;   /* キャラのマスクのワールド (img.data が NULL なら無し) */
	DfAuxMap   sharpMap;     /* Sharp Layer (塗った所は必ずくっきり。img.data が NULL なら無し) */
	DfAuxMap   blurMap;      /* Blur Layer (塗った所は周りの奥の深度でぼかす) */
	void      *dst;          /* 出力 (GPU のワールド、BGRA float) */
	int        dstRowbytes;
	/* 前後のフレームの深度ワールド (深度を時間方向にならす。深度ワールドと同じ大きさ) */
	DfImage    depthNb[4];
	int        depthNbCount;
	/* 結果: 使ったピントの深度 (ログ用) */
	float      focusUsed;
};

/* 1 フレームを描く。0 なら成功。失敗したら err に理由を書く */
int  DofGPU_Render(DfGpuFrame &frame, char *err, int errLen);

/* GPU のメモリ (使い回しの作業領域と表) を手放す (PF_Cmd_GPU_DEVICE_SETDOWN で呼ぶ) */
void DofGPU_Release();

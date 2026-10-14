/*
 * DofAE_Flags.h
 * PiPL リソース (.r) と C++ の両方から読む共有定義。
 * PiPLtool は列挙子や式を解釈できないため数値を直書きし、
 * DofAE.cpp 側の static_assert で SDK の列挙値と一致することを保証する。
 */
#ifndef DOFAE_FLAGS_H
#define DOFAE_FLAGS_H

#define DOF_NAME        "DoF Anime"
#define DOF_MATCH_NAME  "SRLM DoF Anime"
#define DOF_CATEGORY    "SRLM"
#define DOF_SUPPORT_URL "https://github.com/"

#define DOF_MAJOR_VERSION 1
#define DOF_MINOR_VERSION 0
#define DOF_BUG_VERSION   0
#define DOF_STAGE_VERSION 0   /* PF_Stage_DEVELOP */
#define DOF_BUILD_VERSION 1

/* PF_VERSION(1,0,0,PF_Stage_DEVELOP,1) = (1<<19) | (0<<15) | 1 */
#define DOF_PIPL_VERSION  524289

/* PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_CUSTOM_UI (コンポビューのピントの目印)
   | PF_OutFlag_SEND_UPDATE_PARAMS_UI (Focus Mode で使わない項目を灰色にする)
   | PF_OutFlag_WIDE_TIME_INPUT (ピント送りで前後のコマの Focus Point を見る) */
#define DOF_OUTFLAGS      0x06008002
/* PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_SUPPORTS_THREADED_RENDERING
   | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 (GPU 版。実際に使うかは PF_Cmd_GPU_DEVICE_SETUP で CUDA のときだけ返す)
   | PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT (前後のコマの値の読み出しを AE に覚えさせる。無いと、Focus Point が
     止まっている間は AE が 1 コマの結果を使い回し、ピント送りがならされない) */
#define DOF_OUTFLAGS2     0x0A021400

#endif

/*
 * DofAuto.h
 * 深度の自動生成: エフェクト適用時にソース素材のパスを調べ、深度推定 (relight_depth.exe、または別の PC の深度サーバー) を
 * バックグラウンドで起動し、完了したら深度の連番をプロジェクトに読み込んでコンポに非表示レイヤーとして追加し、
 * エフェクトの Depth Layer に結線する。
 *
 * Relight Anime の RelightAuto を写したもの。設定 (%APPDATA%\SRLM\RelightAnime.ini) とキャッシュを共有するので、
 * 同じ素材なら Relight Anime で作った深度をそのまま使う。
 * プロジェクトを書き換える処理は AE のアイドルフック (メインスレッド) でのみ行う。
 */
#pragma once

#include "DofAE.h"
#include <string>

/* GlobalSetup から 1 回呼ぶ。AEGP プラグイン ID の取得とアイドルフック登録 */
void Auto_Init(PF_InData *in_data);

/* エフェクト適用時 (SEQUENCE_SETUP) やボタン押下時に呼ぶ。force = 再生成。
   engine = Depth Engine の選択 (1: この PC、2: 別の PC の深度サーバーで小さいモデル、3: 同じく大きいモデル) */
void Auto_Request(PF_InData *in_data, bool force, int engine = 1);

/* 「Generate Subject」ボタン。今の時刻の Focus Point にいるキャラのマスクを、動画全体について作って Subject Layer に結線する
   (SAM 3.1、tools/subject_mask.py)。深度を作り終えたときにも、Auto Subject がオンなら自動で呼ばれる */
void Auto_RequestSubject(PF_InData *in_data, bool force);

/* 「Text...」ボタン (Sharp Layer / Blur Layer の下)。文字を入れる窓を出し (kind 0 = Sharp、1 = Blur)、
   OK なら Auto_RequestPrompt を呼ぶ */
void Auto_PromptButton(PF_InData *in_data, int kind);

/* 文字 (英語、カンマで複数) に当たる物のマスクを動画全体について作り、Sharp Layer (kind 0) か Blur Layer (kind 1) に結線する
   (SAM 3.1、tools/prompt_mask.py)。atPoint なら今の時刻の Focus Point の所の物を 1 つだけ */
void Auto_RequestPrompt(PF_InData *in_data, int kind, const std::wstring &text, bool atPoint);

/* <cache>\dof_auto.log へ 1 行書く (描画スレッドからも呼べる) */
void Dof_Log(const std::wstring &msg);

/* 「Clean Cache...」ボタン。今のプロジェクトで使っていない深度キャッシュを確認のうえ消す */
void Auto_CleanCache(PF_InData *in_data);

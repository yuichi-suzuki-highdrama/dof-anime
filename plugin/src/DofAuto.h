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

/* GlobalSetup から 1 回呼ぶ。AEGP プラグイン ID の取得とアイドルフック登録 */
void Auto_Init(PF_InData *in_data);

/* エフェクト適用時 (SEQUENCE_SETUP) やボタン押下時に呼ぶ。force = 再生成。
   engine = Depth Engine の選択 (1: この PC、2: 別の PC の深度サーバーで小さいモデル、3: 同じく大きいモデル) */
void Auto_Request(PF_InData *in_data, bool force, int engine = 1);

/* <cache>\dof_auto.log へ 1 行書く (描画スレッドからも呼べる) */
#include <string>
void Dof_Log(const std::wstring &msg);

/* 「Clean Cache...」ボタン。今のプロジェクトで使っていない深度キャッシュを確認のうえ消す */
void Auto_CleanCache(PF_InData *in_data);

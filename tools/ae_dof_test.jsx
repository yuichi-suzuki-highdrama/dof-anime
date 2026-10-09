/*
 * ae_dof_test.jsx - DoF Anime の動作確認 (前半: 適用するだけ)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_dof_test.jsx
 *
 * 新しいプロジェクトにテスト動画を読み込んでコンポ "DoFTest" を作り、時刻を 60 コマ目にして
 * DoF Anime を掛けるだけで終える (Focus Point は既定の位置 = 画面の中央あたり)。
 * エフェクトを掛けたら、すぐにスクリプトを終えること (続けて値を書き換えると、その途中で深度の結線が走ることがある)。
 * (スクリプト実行中は AE のアイドルフックが動かないので、深度とキャラのマスクの結線の確認と
 *  書き出しは ae_dof_check.jsx を後から別に実行する)
 * 素材は環境変数 DOF_TEST_SRC、無ければ下の SRC。
 */
(function () {
    var OUT = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var SRC = $.getenv("DOF_TEST_SRC") || "C:/path/to/source.mp4";
    var lines = [];
    function log(s) { lines.push(s); }
    function flush() {
        try {
            var d = new Folder(OUT); if (!d.exists) d.create();
            var f = new File(OUT + "apply.log"); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close();
        } catch (e) {}
    }
    try {
        try { app.preferences.savePrefAsLong("Main Pref Section", "Pref_SCRIPTING_FILE_NETWORK_SECURITY", 1); } catch (e0) {}
        var proj = app.newProject();
        proj.bitsPerChannel = 16;
        var src = proj.importFile(new ImportOptions(new File(SRC)));
        var comp = proj.items.addComp("DoFTest", src.width, src.height, 1.0, src.duration, src.frameRate);
        var layer = comp.layers.add(src);
        layer.name = "src";
        comp.time = 60 / src.frameRate;
        comp.openInViewer();
        log("comp time " + comp.time);
        var fx = layer.property("ADBE Effect Parade").addProperty("SRLM DoF Anime");
        log("applied: " + (fx ? fx.name : "null") + " at " + new Date().toString());
        log("DONE");
    } catch (err) {
        var msg = ""; try { msg = String(err.message); } catch (ea) {}
        var ln = ""; try { ln = String(err.line); } catch (eb) {}
        log("ERROR: " + msg + " (line " + ln + ")");
    }
    flush();
})();

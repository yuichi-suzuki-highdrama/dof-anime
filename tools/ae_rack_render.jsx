/*
 * ae_rack_render.jsx - ピント送りの作例 (後半: キーフレームを打って書き出す)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_rack_render.jsx
 *
 * ae_rack_setup.jsx のあと、深度とキャラのマスクが結線されてから実行する。
 *   0〜2 秒: 前のキャラにピント → 2 秒でパッと後ろのキャラへ (停止したキーフレーム)
 *   4〜5 秒: なめらかに前のキャラへ戻す (イージーイーズ) → 6 秒まで
 * 書き出し先は環境変数 DOF_RACK_OUT、無ければ下の OUT_MP4。プロジェクトは保存しない。
 */
(function () {
    var LOGDIR = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var OUT_MP4 = $.getenv("DOF_RACK_OUT") || "C:/path/to/output/rack_focus.mp4";
    /* 顔の位置 (素材の画素。1456x816 の絵の場合) */
    var FRONT = [735, 195], BACK = [1135, 170];
    var lines = [];
    function log(s) { lines.push(s); }
    function flush() {
        try {
            var f = new File(LOGDIR + "rack_render.log"); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close();
        } catch (e) {}
    }
    app.beginSuppressDialogs();
    try {
        var comp = null;
        for (var i = 1; i <= app.project.numItems; ++i)
            if (app.project.item(i) instanceof CompItem && app.project.item(i).name == "RackFocus") comp = app.project.item(i);
        if (!comp) throw new Error("comp RackFocus not found");
        var layer = comp.layer("src");
        var fx = layer.property("ADBE Effect Parade").property("DoF Anime");
        log("Depth Layer = " + fx.property("Depth Layer").value + ", Subject Layer = " + fx.property("Subject Layer").value +
            ", Subject Channel = " + fx.property("Subject Channel").value);
        fx.property("Blur Size").setValue(35);

        var fp = fx.property("Focus Point");
        while (fp.numKeys > 0) fp.removeKey(1);
        fp.setValueAtTime(0, FRONT);
        fp.setValueAtTime(2, BACK);      /* 2 秒でパッと切り替える */
        fp.setValueAtTime(4, BACK);
        fp.setValueAtTime(5, FRONT);     /* 4〜5 秒でなめらかに戻す */
        /* 0 秒のキーは停止 (次のキーまで値を保つ = 2 秒で急に切り替わる) */
        fp.setInterpolationTypeAtKey(1, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
        fp.setInterpolationTypeAtKey(2, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
        var ease = new KeyframeEase(0, 75);
        fp.setInterpolationTypeAtKey(3, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
        fp.setInterpolationTypeAtKey(4, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
        try { fp.setTemporalEaseAtKey(3, [ease], [ease]); fp.setTemporalEaseAtKey(4, [ease], [ease]); } catch (e2) { log("ease: " + e2.message); }
        log("focus keys: " + fp.numKeys);

        var rq = app.project.renderQueue;
        while (rq.numItems > 0) rq.item(1).remove();
        var rqi = rq.items.add(comp);
        rqi.outputModule(1).file = new File(OUT_MP4);
        rq.render();
        while (rq.numItems > 0) rq.item(1).remove();
        log("rendered " + OUT_MP4);
        log("DONE");
    } catch (err) {
        log("ERROR: " + err.message + " (line " + err.line + ")");
    }
    app.endSuppressDialogs(false);
    flush();
})();

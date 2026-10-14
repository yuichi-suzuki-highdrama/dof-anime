/*
 * ae_h3_rack_render.jsx - ピント送りの作例 (H3 で作った 2 人の動画。後半: キーフレームを打って書き出す)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_h3_rack_render.jsx
 *
 * ae_h3_rack_setup.jsx のあと、深度とキャラのマスクが結線されてから実行する。
 *   0〜3 秒: 手前のキャラにピント → 3 秒で後ろのキャラへ (停止したキーフレーム。Focus Pull Time の分だけかけて移る)
 *   5.5〜6.5 秒: なめらかに手前のキャラへ戻す (イージーイーズ) → 8 秒まで
 * 点はどちらも 8 秒間ずっとそのキャラの上にある所 (マスクで確かめた)。
 * 手前のぼけの強さは NEAR (0 = 手前はぼかさない)。書き出しは NEAR ごとに 1 本ずつ。プロジェクトは保存しない。
 */
(function () {
    var LOGDIR = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var NEARS = [0, 1];
    var FRONT = [640, 200], BACK = [1060, 200];
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
            if (app.project.item(i) instanceof CompItem && app.project.item(i).name == "H3RackFocus") comp = app.project.item(i);
        if (!comp) throw new Error("comp H3RackFocus not found");
        var fx = comp.layer("src").property("ADBE Effect Parade").property("DoF Anime");
        log("Depth Layer = " + fx.property("Depth Layer").value + ", Subject Layer = " + fx.property("Subject Layer").value);
        fx.property("Blur Size").setValue(18);
        fx.property("Focus Pull Time").setValue(0.8);

        var fp = fx.property("Focus Point");
        while (fp.numKeys > 0) fp.removeKey(1);
        fp.setValueAtTime(0, FRONT);
        fp.setValueAtTime(3, BACK);
        fp.setValueAtTime(5.5, BACK);
        fp.setValueAtTime(6.5, FRONT);
        fp.setInterpolationTypeAtKey(1, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
        fp.setInterpolationTypeAtKey(2, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD);
        var ease = new KeyframeEase(0, 75);
        fp.setInterpolationTypeAtKey(3, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
        fp.setInterpolationTypeAtKey(4, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
        try { fp.setTemporalEaseAtKey(3, [ease], [ease]); fp.setTemporalEaseAtKey(4, [ease], [ease]); } catch (e2) { log("ease: " + e2.message); }

        var rq = app.project.renderQueue;
        for (var n = 0; n < NEARS.length; ++n) {
            fx.property("Near Blur").setValue(NEARS[n]);
            var out = "C:/path/to/output/h3/h3_rack_focus_near" + NEARS[n] + ".mp4";
            while (rq.numItems > 0) rq.item(1).remove();
            var rqi = rq.items.add(comp);
            rqi.outputModule(1).file = new File(out);
            rq.render();
            log("rendered " + out);
        }
        while (rq.numItems > 0) rq.item(1).remove();
        log("DONE");
    } catch (err) {
        log("ERROR: " + err.message + " (line " + err.line + ")");
    }
    app.endSuppressDialogs(false);
    flush();
})();

/*
 * ae_dof_check.jsx - DoF Anime の動作確認 (後半: 結線を調べて書き出す)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_dof_check.jsx
 *
 * ae_dof_test.jsx のあと、深度とキャラのマスクの自動生成が終わってから実行する。
 * コンポ "DoFTest" のレイヤーと、DoF Anime の Depth Layer / Subject Layer の結線を記録し、
 * 60 コマ目を Output = Final / Blur Map で PNG に書き出す。プロジェクトは保存しない。
 */
(function () {
    var OUT = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var lines = [];
    function log(s) { lines.push(s); }
    function flush() {
        try {
            var d = new Folder(OUT); if (!d.exists) d.create();
            var f = new File(OUT + "check.log"); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close();
        } catch (e) {}
    }
    try {
        var comp = null;
        for (var i = 1; i <= app.project.numItems; ++i)
            if (app.project.item(i) instanceof CompItem && app.project.item(i).name == "DoFTest") comp = app.project.item(i);
        if (!comp) throw new Error("comp DoFTest not found");
        for (var j = 1; j <= comp.numLayers; ++j) log("layer " + j + ": " + comp.layer(j).name + (comp.layer(j).enabled ? "" : " (hidden)"));
        var layer = comp.layer("src");
        var fx = layer.property("ADBE Effect Parade").property("DoF Anime");
        var dl = fx.property("Depth Layer").value, sl = fx.property("Subject Layer").value;
        log("Depth Layer = " + dl + ", Subject Layer = " + sl + ", Subject Channel = " + fx.property("Subject Channel").value);
        var t = 60 / comp.frameRate;
        var modes = [[1, "final"], [3, "blurmap"]];
        for (var m = 0; m < modes.length; ++m) {
            fx.property("Output").setValue(modes[m][0]);
            var f = new File(OUT + "frame60_" + modes[m][1] + ".png");
            comp.saveFrameToPng(t, f);
            log("saved " + f.fsName);
        }
        fx.property("Output").setValue(1);
        log("DONE");
    } catch (err) {
        var msg = ""; try { msg = String(err.message); } catch (ea) {}
        var ln = ""; try { ln = String(err.line); } catch (eb) {}
        log("ERROR: " + msg + " (line " + ln + ")");
    }
    flush();
})();

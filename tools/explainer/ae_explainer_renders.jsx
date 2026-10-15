/*
 * ae_explainer_renders.jsx - 解説動画の素材を書き出す
 *
 *   set DOF_EXPLAINER_OUT=<作業フォルダ>/renders/
 *   AfterFX.exe -r ae_explainer_renders.jsx
 *
 * 開いているプロジェクトのコンポ "H3RackFocus_Cool" (Relight Anime → DoF Anime → Nitrate) を複製し、DoF Anime だけの版や、
 * 出力の切り替え (Depth / Blur Map)・直す前の状態 (Exclude なし / Near Fill 0) を作って書き出す。元のコンポは変えない。
 * プロジェクトは保存しない。
 */
(function () {
    var LOG = Folder.temp.fsName + "/dof_ae_test/explainer_renders.log";
    var OUT = $.getenv("DOF_EXPLAINER_OUT") || (Folder.temp.fsName.replace(/\\/g, "/") + "/dof_explainer/renders/");
    var lines = [];
    function log(s) { lines.push(s); }
    app.beginSuppressDialogs();
    try {
        new Folder(OUT).create();
        var base = null;
        for (var i = 1; i <= app.project.numItems; ++i)
            if (app.project.item(i) instanceof CompItem && app.project.item(i).name == "H3RackFocus_Cool") base = app.project.item(i);
        if (!base) throw new Error("comp not found");
        function make(name, fn) {
            for (var j = app.project.numItems; j >= 1; --j)
                if (app.project.item(j) instanceof CompItem && app.project.item(j).name == name) app.project.item(j).remove();
            var c = base.duplicate();
            c.name = name;
            var parade = c.layer("src").property("ADBE Effect Parade");
            /* DoF Anime だけにする (Relight Anime と Nitrate を外す) */
            for (var p = parade.numProperties; p >= 1; --p)
                if (parade.property(p).name != "DoF Anime") parade.property(p).remove();
            var fx = parade.property("DoF Anime");
            fn(fx, c);
            return c;
        }
        var jobs = [
            ["EX_dof_only", function (fx) {}],
            ["EX_depth", function (fx) { fx.property("Output").setValue(2); }],
            ["EX_blurmap", function (fx) { fx.property("Output").setValue(3); }],
            ["EX_no_exclude", function (fx) { fx.property("Exclude Layer").setValue(0); }],
            ["EX_nearfill0", function (fx) { fx.property("Near Fill").setValue(0); }]
        ];
        var rq = app.project.renderQueue;
        while (rq.numItems > 0) rq.item(1).remove();
        for (var k = 0; k < jobs.length; ++k) {
            var c = make(jobs[k][0], jobs[k][1]);
            var fx = c.layer("src").property("ADBE Effect Parade").property("DoF Anime");
            log(jobs[k][0] + ": effects=" + c.layer("src").property("ADBE Effect Parade").numProperties + " output=" + fx.property("Output").value +
                " exclude=" + fx.property("Exclude Layer").value + " nearfill=" + fx.property("Near Fill").value + " blurL=" + fx.property("Blur Layer").value);
            var r = rq.items.add(c);
            r.outputModule(1).file = new File(OUT + jobs[k][0] + ".mp4");
        }
        rq.render();
        log("rendered");
        while (rq.numItems > 0) rq.item(1).remove();
        base.openInViewer();
    } catch (err) { log("ERROR: " + err.message + " (line " + err.line + ")"); }
    app.endSuppressDialogs(false);
    var f = new File(LOG); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close();
})();

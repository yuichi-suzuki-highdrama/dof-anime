/*
 * ae_dof_reapply.jsx - 試験用: コンポ "DoFTest" の DoF Anime を外して掛け直す (深度・マスクの自動生成をもう一度走らせる)
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_dof_reapply.jsx
 * 掛ける操作を最後にして、すぐ終える。
 */
(function () {
    var OUT = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var lines = [];
    function flush() {
        try { var f = new File(OUT + "reapply.log"); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close(); } catch (e) {}
    }
    try {
        var comp = null;
        for (var i = 1; i <= app.project.numItems; ++i)
            if (app.project.item(i) instanceof CompItem && app.project.item(i).name == "DoFTest") comp = app.project.item(i);
        var layer = comp.layer("src");
        var parade = layer.property("ADBE Effect Parade");
        var fx = parade.property("DoF Anime");
        if (fx) fx.remove();
        parade.addProperty("SRLM DoF Anime");
        lines.push("reapplied " + new Date().toString());
    } catch (err) {
        lines.push("ERROR: " + err.message + " (line " + err.line + ")");
    }
    flush();
})();

/*
 * ae_rack_setup.jsx - ピント送りの作例 (前半: 適用するだけ)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_rack_setup.jsx
 *
 * 新しいプロジェクトに静止画を読み込んでコンポ "RackFocus" (6 秒) を作り、DoF Anime を掛けるだけで終える。
 * 深度とキャラのマスクは AE のアイドルフックで作られるので、そのあと ae_rack_render.jsx を実行する。
 * 素材は環境変数 DOF_RACK_SRC、無ければ下の SRC。
 */
(function () {
    var OUT = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var SRC = $.getenv("DOF_RACK_SRC");   /* 素材の静止画 (必須) */
    var lines = [];
    function log(s) { lines.push(s); }
    function flush() {
        try {
            var d = new Folder(OUT); if (!d.exists) d.create();
            var f = new File(OUT + "rack_setup.log"); f.encoding = "UTF-8"; f.open("w"); f.write(lines.join("\n") + "\n"); f.close();
        } catch (e) {}
    }
    try {
        var proj = app.newProject();
        proj.bitsPerChannel = 16;
        if (!SRC) throw new Error("set the source path in the environment variable DOF_RACK_SRC");
        var src = proj.importFile(new ImportOptions(new File(SRC)));
        var comp = proj.items.addComp("RackFocus", src.width, src.height, 1.0, 6.0, 24);
        var layer = comp.layers.add(src);
        layer.name = "src";
        comp.time = 0;
        comp.openInViewer();
        /* エフェクトを掛ける操作を最後にして、すぐ終える (続けて値を書き換えると、その途中で深度の結線が走ることがある) */
        var fx = layer.property("ADBE Effect Parade").addProperty("SRLM DoF Anime");
        log("applied: " + (fx ? fx.name : "null") + " " + src.width + "x" + src.height + " at " + new Date().toString());
    } catch (err) {
        log("ERROR: " + err.message + " (line " + err.line + ")");
    }
    flush();
})();

/*
 * ae_h3_rack_setup.jsx - ピント送りの作例 (H3 で作った 2 人の動画。前半: 適用するだけ)
 *
 *   AfterFX.exe -r C:\dev\DepthOfFieldAEPlugin\tools\ae_h3_rack_setup.jsx
 *
 * 新しいプロジェクトに動画を読み込んでコンポ "H3RackFocus" (動画と同じ長さ・コマ数) を作り、DoF Anime を掛けるだけで終える。
 * 深度とキャラのマスクは AE のアイドルフックで作られるので、そのあと ae_h3_rack_render.jsx を実行する。
 */
(function () {
    var OUT = Folder.temp.fsName.replace(/\\/g, "/") + "/dof_ae_test/";
    var SRC = "C:/path/to/output/h3/s2026101511_00001_original.mp4";
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
        var src = proj.importFile(new ImportOptions(new File(SRC)));
        var comp = proj.items.addComp("H3RackFocus", src.width, src.height, 1.0, src.duration, src.frameRate);
        var layer = comp.layers.add(src);
        layer.name = "src";
        comp.time = 0;
        comp.openInViewer();
        /* エフェクトを掛ける操作を最後にして、すぐ終える (続けて値を書き換えると、その途中で深度の結線が走ることがある) */
        var fx = layer.property("ADBE Effect Parade").addProperty("SRLM DoF Anime");
        log("applied: " + (fx ? fx.name : "null") + " " + src.width + "x" + src.height + " " + src.duration + "s at " + new Date().toString());
    } catch (err) {
        log("ERROR: " + err.message + " (line " + err.line + ")");
    }
    flush();
})();

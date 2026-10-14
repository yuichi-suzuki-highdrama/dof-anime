root = r'C:/dev/DepthOfFieldAEPlugin/'


def patch(path, pairs):
    raw = open(root + path, 'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    crlf = b'\r\n' in raw
    s = raw.decode('utf-8-sig').replace('\r\n', '\n')
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:100])
        s = s.replace(a, b, 1)
    if crlf:
        s = s.replace('\n', '\r\n')
    open(root + path, 'wb').write((b'\xef\xbb\xbf' if bom else b'') + s.encode('utf-8'))


# ---------------- DofCore.h ----------------
patch('plugin/src/DofCore.h', [
    ('''	float subjectAmount;      /* Subject Layer の中をどれだけくっきり残すか (0..1) */''',
     '''	float subjectAmount;      /* Keep Subject Sharp: ピントに関係なく、マスクの中をどれだけくっきり残すか (0..1) */
	float subjectFlatten;     /* Subject Flatten: マスクの中の深度を人物の代表の深度にどれだけ寄せるか (0..1)。
	                             1 なら人物を 1 枚の板として扱い、人物の中のぼけのむらが無くなる */'''),
    ('''	const float *subject;     /* キャラのマスク (0..1、出力の大きさ。NULL なら無し) */''',
     '''	const float *subject;     /* キャラのマスク (0..1、出力の大きさ。NULL なら無し) */
	float subjectDepth;       /* 人物の代表の深度 (マスクの中の深度の中央値)。マスクが無ければ負 */'''),
    # マスクを小数の位置で読む関数に分ける
    ('''DF_HD DF_INL float dfSubjectAt(const DfAuxMap &m, int channel, int x, int y)
{
	if (!m.img.data) return 0.f;
	const float fx = dfClamp((float)x * m.sx + m.ox, 0.f, (float)(m.img.width - 1));
	const float fy = dfClamp((float)y * m.sy + m.oy, 0.f, (float)(m.img.height - 1));''',
     '''DF_HD DF_INL float dfSubjectAtF(const DfAuxMap &m, int channel, float fx, float fy)
{
	if (!m.img.data) return 0.f;
	fx = dfClamp(fx, 0.f, (float)(m.img.width - 1));
	fy = dfClamp(fy, 0.f, (float)(m.img.height - 1));'''),
    ('''	return dfSat(dfMix(dfMix(v[0], v[1], tx), dfMix(v[2], v[3], tx), ty));
}''',
     '''	return dfSat(dfMix(dfMix(v[0], v[1], tx), dfMix(v[2], v[3], tx), ty));
}

/* 出力の画素 (x, y) のマスク */
DF_HD DF_INL float dfSubjectAt(const DfAuxMap &m, int channel, int x, int y)
{
	return dfSubjectAtF(m, channel, (float)x * m.sx + m.ox, (float)y * m.sy + m.oy);
}

/* 深度の点 (i, j) の位置のマスク。どちらもレイヤー全体を取り出しているので、レイヤーの中の位置の割合で合わせる */
DF_HD DF_INL float dfSubjectAtLow(const DfAuxMap &depthMap, const DfAuxMap &m, int channel, int i, int j)
{
	const float rx = m.sx / dfMax(depthMap.sx, 1e-6f), ry = m.sy / dfMax(depthMap.sy, 1e-6f);
	return dfSubjectAtF(m, channel, ((float)i + 0.5f) * rx - 0.5f, ((float)j + 0.5f) * ry - 0.5f);
}

/* マスクの中の深度で、人物と深度を混ぜたもの (Subject Flatten)。s: その画素のマスク */
DF_HD DF_INL float dfSubjectBlendDepth(const DofContext *c, float depth, float s)
{
	if (c->subjectDepth < 0.f || c->p.subjectFlatten <= 0.f) return depth;
	return dfMix(depth, c->subjectDepth, dfSat(s * c->p.subjectFlatten));
}'''),
    # ホストで人物の代表の深度を求める
    ('''/* ピントの深度を求める: Focus Point のまわりの深度の中央値。''',
     '''/* 人物の代表の深度: マスクが半分以上の深度の点の深度の中央値 (1024 段の度数で)。点が少なければ -1 */
inline float dfSubjectMedian(const float *low, const float *mask, int dw, int dh)
{
	static const int BINS = 1024;
	int hist[BINS] = {0};
	int n = 0;
	for (int k = 0; k < dw * dh; ++k) {
		if (mask[k] < 0.5f) continue;
		hist[dfClamp((int)(dfSat(low[k]) * (BINS - 1) + 0.5f), 0, BINS - 1)]++;
		++n;
	}
	if (n < 16) return -1.f;
	int acc = 0;
	for (int b = 0; b < BINS; ++b) {
		acc += hist[b];
		if (acc * 2 >= n) return (float)b / (float)(BINS - 1);
	}
	return -1.f;
}

/* Focus Point (レイヤー座標) がマスクの中か (深度の点で見る) */
inline bool dfPointInSubject(const float *mask, int dw, int dh, float sx, float sy, float lx, float ly)
{
	const int i = dfClamp(dfRound((lx + 0.5f) * sx - 0.5f), 0, dw - 1);
	const int j = dfClamp(dfRound((ly + 0.5f) * sy - 0.5f), 0, dh - 1);
	return mask[(size_t)j * dw + i] >= 0.5f;
}

/* ピントの深度を求める: Focus Point のまわりの深度の中央値。'''),
])

# ---------------- DofGPU.cu ----------------
patch('plugin/src/DofGPU.cu', [
    ('''__global__ void kCoc(DofContext c, const float *depth, float *coc)
{
	DF_XY(c.outW, c.outH);
	const size_t n = (size_t)y * c.outW + x;
	float r = dfCoc(&c, depth[n]);
	if (c.subject) r *= 1.f - dfSat(c.subject[n] * c.p.subjectAmount);
	coc[n] = r;
}''',
     '''__global__ void kSubjectLow(DfAuxMap depthMap, DfAuxMap map, int channel, int dw, int dh, float *mask)
{
	DF_XY(dw, dh);
	mask[(size_t)y * dw + x] = dfSubjectAtLow(depthMap, map, channel, x, y);
}

/* ぼけの大きさ。マスクの中は深度を人物の代表の深度に寄せ (Subject Flatten)、Keep Subject Sharp の分だけぼけを減らす。
   寄せた深度は depth に書き戻す (Output = Depth で見えるように) */
__global__ void kCoc(DofContext c, float *depth, float *coc)
{
	DF_XY(c.outW, c.outH);
	const size_t n = (size_t)y * c.outW + x;
	const float s = c.subject ? c.subject[n] : 0.f;
	const float d = dfSubjectBlendDepth(&c, depth[n], s);
	depth[n] = d;
	coc[n] = dfCoc(&c, d) * (1.f - dfSat(s * c.p.subjectAmount));
}'''),
    ('''	const bool hasSubject = f.subjectMap.img.data != NULL && c.p.subjectAmount > 0.f;''',
     '''	const bool hasSubject = f.subjectMap.img.data != NULL && (c.p.subjectAmount > 0.f || c.p.subjectFlatten > 0.f);
	c.subjectDepth = -1.f;'''),
    ('''		if (hasSubject) count(NP);''',
     '''		if (hasSubject) { count(NP); count((size_t)dw * dh); }'''),
    ('''		float focus = c.p.focusDepth;
		if (c.p.focusMode == FOCUS_POINT) {
			/* ピントの深度は深度の点 (小さい) をホストへ写して求める */
			std::vector<float> host((size_t)dw * dh);
			cudaMemcpy(host.data(), low, sizeof(float) * host.size(), cudaMemcpyDeviceToHost);
			focus = dfFocusFromPoint(host.data(), dw, dh, f.depthMap.sx, f.depthMap.sy, c.p.focusX, c.p.focusY);
		}
		dfSetupFrame(c, focus);''',
     '''		float focus = c.p.focusDepth;
		if (c.p.focusMode == FOCUS_POINT || hasSubject) {
			/* ピントの深度と人物の代表の深度は、深度の点 (小さい) をホストへ写して求める */
			std::vector<float> host((size_t)dw * dh);
			cudaMemcpy(host.data(), low, sizeof(float) * host.size(), cudaMemcpyDeviceToHost);
			std::vector<float> hmask;
			if (hasSubject) {
				float *lowMask = cv.take((size_t)dw * dh);
				kSubjectLow<<<Grid2(dw, dh), BLOCK2>>>(f.depthMap, f.subjectMap, c.p.subjectChannel, dw, dh, lowMask);
				DF_CHECK("subject low");
				hmask.resize((size_t)dw * dh);
				cudaMemcpy(hmask.data(), lowMask, sizeof(float) * hmask.size(), cudaMemcpyDeviceToHost);
				c.subjectDepth = dfSubjectMedian(host.data(), hmask.data(), dw, dh);
			}
			if (c.p.focusMode == FOCUS_POINT) {
				/* Focus Point が人物の上なら、人物の代表の深度に合わせる (人物全体にピントが合う) */
				if (c.subjectDepth >= 0.f && dfPointInSubject(hmask.data(), dw, dh, f.depthMap.sx, f.depthMap.sy, c.p.focusX, c.p.focusY))
					focus = c.subjectDepth;
				else
					focus = dfFocusFromPoint(host.data(), dw, dh, f.depthMap.sx, f.depthMap.sy, c.p.focusX, c.p.focusY);
			}
		}
		{
			const float sd = c.subjectDepth;
			dfSetupFrame(c, focus);
			c.subjectDepth = sd;
		}'''),
])

# ---------------- DofAE.cpp ----------------
patch('plugin/src/DofAE.cpp', [
    ('''	c.depth = c.coc = c.tileR = c.subject = NULL;
	dfSetupFrame(c, c.p.focusDepth);

	/* 1. リニアの色と縮小 */''',
     '''	c.depth = c.coc = c.tileR = c.subject = NULL;
	dfSetupFrame(c, c.p.focusDepth);
	c.subjectDepth = -1.f;

	/* 1. リニアの色と縮小 */'''),
    ('''		if (c.p.focusMode == FOCUS_POINT)
			focus = dfFocusFromPoint(low.data(), dw, dh, depthMap.sx, depthMap.sy, c.p.focusX, c.p.focusY);
		dfSetupFrame(c, focus);''',
     '''		/* 人物の代表の深度 (マスクの中の深度の中央値) */
		const bool hasSubject = subjectMap.img.data && (c.p.subjectAmount > 0.f || c.p.subjectFlatten > 0.f);
		std::vector<float> lowMask;
		if (hasSubject) {
			lowMask.resize((size_t)dw * dh);
			parallelRows(dh, [&](A_long j) { for (int i = 0; i < dw; ++i) lowMask[(size_t)j * dw + i] = dfSubjectAtLow(depthMap, subjectMap, c.p.subjectChannel, i, (int)j); });
			c.subjectDepth = dfSubjectMedian(low.data(), lowMask.data(), dw, dh);
		}
		if (c.p.focusMode == FOCUS_POINT) {
			/* Focus Point が人物の上なら、人物の代表の深度に合わせる (人物全体にピントが合う) */
			if (c.subjectDepth >= 0.f && dfPointInSubject(lowMask.data(), dw, dh, depthMap.sx, depthMap.sy, c.p.focusX, c.p.focusY))
				focus = c.subjectDepth;
			else
				focus = dfFocusFromPoint(low.data(), dw, dh, depthMap.sx, depthMap.sy, c.p.focusX, c.p.focusY);
		}
		{
			const float sd = c.subjectDepth;
			dfSetupFrame(c, focus);
			c.subjectDepth = sd;
		}'''),
    ('''		/* キャラのマスクの中はぼかさない (深度がキャラの中で揺れても、体全体をくっきり残す) */
		if (subjectMap.img.data && c.p.subjectAmount > 0.f) {''',
     '''		/* キャラのマスク。中の深度を人物の代表の深度に寄せ (Subject Flatten)、Keep Subject Sharp の分だけぼけを減らす */
		if (hasSubject) {'''),
    ('''				const size_t n = (size_t)y * W + x;
				float r = dfCoc(&c, depth[n]);
				if (c.subject) r *= 1.f - dfSat(c.subject[n] * c.p.subjectAmount);
				coc[n] = r;''',
     '''				const size_t n = (size_t)y * W + x;
				const float s = c.subject ? c.subject[n] : 0.f;
				const float d = dfSubjectBlendDepth(&c, depth[n], s);
				depth[n] = d;
				coc[n] = dfCoc(&c, d) * (1.f - dfSat(s * c.p.subjectAmount));'''),
    # パラメータ
    ('''	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Keep Subject Sharp", 0.f, 100.f, 0.f, 100.f, 100.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_SUBJECT_AMOUNT);''',
     '''	/* マスクの中の深度を、人物の代表の深度 (マスクの中の深度の中央値) に寄せる。100% で人物を 1 枚の板として扱う。
	   Focus Point が人物の上なら人物全体にピントが合い、背景へ動かせば人物は全体が均一にぼける (ピント送りができる) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Subject Flatten", 0.f, 100.f, 0.f, 100.f, 100.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_SUBJECT_FLATTEN);

	/* ピントに関係なく、マスクの中を必ずくっきり残す (手前のぼけも重ねない) */
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Keep Subject Sharp", 0.f, 100.f, 0.f, 100.f, 0.f,
		PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, ID_SUBJECT_AMOUNT);'''),
    ('''	p.subjectAmount  = dfSat(r.fs(DOF_SUBJECT_AMOUNT, 100.f) / 100.f);''',
     '''	p.subjectAmount  = dfSat(r.fs(DOF_SUBJECT_AMOUNT, 0.f) / 100.f);
	p.subjectFlatten = dfSat(r.fs(DOF_SUBJECT_FLATTEN, 100.f) / 100.f);'''),
])

# ---------------- DofAE.h ----------------
patch('plugin/src/DofAE.h', [
    ('''	DOF_SUBJECT_CHANNEL,      /* 1: Alpha  2: Luminance */
	DOF_SUBJECT_AMOUNT,''',
     '''	DOF_SUBJECT_CHANNEL,      /* 1: Alpha  2: Luminance */
	DOF_SUBJECT_FLATTEN,      /* マスクの中の深度を人物の代表の深度に寄せる */
	DOF_SUBJECT_AMOUNT,       /* Keep Subject Sharp */'''),
    ('''	ID_SUBJECT_GENERATE,''',
     '''	ID_SUBJECT_GENERATE,
	ID_SUBJECT_FLATTEN,'''),
])

# ---------------- dof_cli.cpp ----------------
patch('tools/harness/dof_cli.cpp', [
    ('''	p.subjectChannel = 1; p.subjectAmount = 1.f;''',
     '''	p.subjectChannel = 1; p.subjectAmount = 0.f; p.subjectFlatten = 1.f;'''),
    ('''		else if (k == "keep") p.subjectAmount = f;''',
     '''		else if (k == "keep") p.subjectAmount = f;
		else if (k == "flatten") p.subjectFlatten = f;'''),
    (''' *         subject=ファイル (キャラのマスク、W x H の 1 チャンネル 0..1)、keep (Keep Subject Sharp 0..1)''',
     ''' *         subject=ファイル (キャラのマスク、W x H の 1 チャンネル 0..1)、keep (Keep Subject Sharp 0..1)、flatten (Subject Flatten 0..1)'''),
])
print('ok')

# -*- coding: utf-8 -*-
"""session6.html 의 인라인 SVG 그림을 생성 · 갱신한다.

사용법
    python3 _build_s6_figs.py            # session6.html 안의 그림을 제자리 갱신
    python3 _build_s6_figs.py --check    # 수치 대조 + 구조 검사만 (파일 안 건드림)
    python3 _build_s6_figs.py --dump NAME  # 그림 하나를 표준출력으로

동작 방식
    session6.html 의 각 그림은 <svg data-fig="NAME" ...> ... </svg> 로 표시돼 있다.
    이 스크립트는 그 블록을 통째로 새로 생성한 SVG 로 바꾼다. 슬라이드의 글·배치는
    HTML 에서 직접 고치고, 그림은 여기서 고친 뒤 이 스크립트를 다시 돌리면 된다.

★ 이 모듈이 성립한다고 가정하는 불변성 (수정 전 반드시 확인)
    1. data-fig NAME 은 HTML 안에서 유일하고 FIGURES 키와 1:1 대응한다.
       깨지면 → 중복 시 첫 블록만 갱신되고 나머지는 낡은 그림으로 조용히 남는다.
       확인 → _check_registry(): 대칭차집합 == 공집합 + 각 NAME 등장 1회 assert.
    2. 각 SVG 블록에 중첩 <svg> 가 없다 (여는 <svg 부터 첫 </svg> 까지가 한 블록).
       깨지면 → non-greedy 매칭이 조기 종료해 태그가 닫히지 않고 HTML 이 깨진다.
       확인 → render(): 생성물마다 count("<svg") == 1 assert + 치환 후 개수 일치 검사.
    3. 차트에 그리는 모든 수치는 이 파일 상단 상수에만 있고, 그 값은 session6.html
       표 또는 _build_s6.py 독스트링에 문자열로 실재한다 (추정·보간 금지).
       깨지면 → 슬라이드와 노트북이 서로 다른 숫자를 말하는 조용한 불일치.
       확인 → verify_numbers(): 86개 값을 두 원본 파일에 문자열 대조.

수치 출처
    구간별 위험률 · 깊이별 train/CV · 4모델 지표 · 폴드별 AUC → session6.html 표
    포레스트 깊이별 CV (.6651/.6589/.6516/.6421) → _build_s6.py 독스트링
    기저율 .337 → Dummy 의 AP .3371 (상수 예측기의 AP = 양성 비율)

팔레트 (기존 덱과 동일 · dataviz validator 로 색각 이상 분리도 검증한 조합)
    Logistic #00f2ff · Forest #adff2f · Tree #ff8080 · Dummy/기준선 #8892a4 · 주의 #ffd479
    인접쌍 최악 ΔE 17.4 (deutan) / 23.7 (normal) — 통과.
    ※ 이 덱은 근검정 배경 위 네온 팔레트라 validator 의 '명도 밴드' 검사는 통과하지
      못한다. 8개 덱 전체가 공유하는 기존 아이덴티티이므로 의도적으로 유지한다.
"""
import os
import re
import sys

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session6", "session6.html")

CY, LM, RD, GY, AM = "#00f2ff", "#adff2f", "#ff8080", "#8892a4", "#ffd479"
GRID, FRAME, INK, MUTE, FAINT = "#222c38", "#3a4250", "#b0b0b0", "#8892a4", "#55606f"
LATO = "Lato, 'Nanum Gothic', sans-serif"
POP = "Poppins, sans-serif"


def _t(x, y, s, fill=INK, size=15, anchor="middle", fam=LATO, weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-family="{fam}"{w}{extra}>{s}</text>')


# ─────────────────────────────────────────────────────────────────────────────
# 1) 슬라이드 7 — 구간별 고스트레스 비율 small multiples
#    출처: session6.html 슬라이드7 표 == session6.ipynb cell10 해설 표
#    peer_support 는 qcut(duplicates="drop") 로 한 구간이 합쳐져 실제로 4점이다.
# ─────────────────────────────────────────────────────────────────────────────
QUINTILES = [
    ("self_esteem",        [.503, .316, .248, .260, .220], "역치형",  LM,
     "1분위에서만 급락"),
    ("depression",         [.226, .241, .369, .430, .464], "단조 증가", CY,
     "직선에 가깝다"),
    ("previous_stress",    [.218, .206, .305, .462, .558], "단조·가팔라짐", CY,
     "뒤로 갈수록 가파름"),
    ("peer_support",       [.509, .396, .189, .234],       "비단조",  AM,
     "3분위 최저 후 되오름"),
]
BASE_RATE = .337   # Dummy 의 AP .3371 (슬라이드13 표) = 양성 기저율


def quintile_panels():
    W, H = 1140, 268
    pw, ph, top = 236, 158, 42
    gap = (W - pw * 4) / 3
    ymax = .60
    out = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
           f'role="img" aria-label="변수별 5분위 구간의 고스트레스 비율">']
    for i, (name, vals, shape, color, note) in enumerate(QUINTILES):
        x0 = i * (pw + gap)
        sy = lambda v: top + ph - (v / ymax) * ph
        n = len(vals)
        sx = lambda k: x0 + 26 + (pw - 42) * (k / (n - 1))
        out.append(f'<g>')
        # 가로 격자 .2 / .4 / .6
        for gv in (.2, .4, .6):
            out.append(f'<line x1="{x0+18}" y1="{sy(gv):.1f}" x2="{x0+pw-6}" '
                       f'y2="{sy(gv):.1f}" stroke="{GRID}" stroke-width="1"/>')
            if i == 0:
                out.append(_t(x0 + 12, sy(gv) + 5, f".{int(gv*10)}", FAINT, 13, "end"))
        # 기저율 기준선
        out.append(f'<line x1="{x0+18}" y1="{sy(BASE_RATE):.1f}" x2="{x0+pw-6}" '
                   f'y2="{sy(BASE_RATE):.1f}" stroke="{GY}" stroke-width="1.5" '
                   f'stroke-dasharray="5 5" opacity=".85"/>')
        # 라인 + 점
        pts = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for k, v in enumerate(vals):
            out.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="5" fill="{color}" '
                       f'stroke="#0a0e14" stroke-width="2"/>')
        # 양 끝 값만 직접 라벨 (모든 점에 숫자 금지)
        out.append(_t(sx(0), sy(vals[0]) - 12, f"{vals[0]:.3f}".lstrip("0"), color, 14, "middle", LATO, "700"))
        out.append(_t(sx(n-1), sy(vals[-1]) + (-12 if vals[-1] >= vals[0] else 20),
                      f"{vals[-1]:.3f}".lstrip("0"), color, 14, "middle", LATO, "700"))
        # 제목 / 모양 / 축
        out.append(_t(x0 + pw / 2, 18, name, "#ffffff", 16, "middle", POP, "700"))
        out.append(_t(x0 + pw / 2, 36, shape, color, 14, "middle", LATO, "700"))
        axis_lbl = "1분위 → 5분위 (낮음→높음)" if n == 5 else f"1구간 → {n}구간 (낮음→높음)"
        out.append(_t(x0 + pw / 2, top + ph + 22, axis_lbl, FAINT, 12.5))
        out.append(_t(x0 + pw / 2, top + ph + 40, note, MUTE, 13))
        out.append('</g>')
    out.append(_t(6, H - 4, "회색 점선 = 전체 고스트레스 비율 .337  ·  세로축 = 그 구간의 실제 고스트레스 비율  ·  "
                  "peer_support 는 동점이 많아 5분위가 4구간으로 합쳐졌다", FAINT, 12.5, "start"))
    out.append('</svg>')
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 2) 슬라이드 11 — 깊이별 train vs CV (+ 슬라이드 12 용 Forest 오버레이)
#    출처: session6.html 슬라이드11 표 · _build_s6.py 독스트링(포레스트 깊이별)
# ─────────────────────────────────────────────────────────────────────────────
DEPTHS = ["1", "2", "3", "5", "8", "없음"]
TREE_TRAIN = [.6199, .6637, .6937, .7816, .9127, 1.0000]
TREE_CV    = [.6067, .6355, .6151, .5892, .5312, .5185]
LEAVES     = [2, 4, 8, 30, 96, 299]
# 포레스트는 3/5/8/None 만 실측 — 깊이 1,2 는 재지 않았으므로 그리지 않는다.
FOREST_DEPTHS = ["3", "5", "8", "없음"]
FOREST_CV     = [.6651, .6589, .6516, .6421]


def depth_curve(with_forest=False):
    W, H = 1140, 292
    L, R, T, B = 62, 210, 22, 50
    pw, ph = W - L - R, H - T - B
    lo, hi = .48, 1.02
    sy = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    sx = lambda k: L + pw * (k / (len(DEPTHS) - 1))
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="트리 깊이에 따른 train AUC 와 CV AUC">']
    for gv in (.5, .6, .7, .8, .9, 1.0):
        o.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{L+pw}" y2="{sy(gv):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(_t(L - 12, sy(gv) + 5, f"{gv:.1f}", FAINT, 14, "end"))
    # 과적합 갭 면적
    up = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(TREE_TRAIN))
    dn = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in reversed(list(enumerate(TREE_CV))))
    o.append(f'<polygon points="{up} {dn}" fill="{RD}" fill-opacity=".13"/>')
    # train
    o.append(f'<polyline points="{up}" fill="none" stroke="{CY}" stroke-width="3.5" '
             f'stroke-linejoin="round"/>')
    # CV (단일 트리)
    cvp = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(TREE_CV))
    o.append(f'<polyline points="{cvp}" fill="none" stroke="{RD}" stroke-width="3.5" '
             f'stroke-linejoin="round"/>')
    for k, v in enumerate(TREE_TRAIN):
        o.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="6" fill="{CY}" '
                 f'stroke="#0a0e14" stroke-width="2"/>')
    for k, v in enumerate(TREE_CV):   # 사각 마커 = 2차 부호화
        o.append(f'<rect x="{sx(k)-5.5:.1f}" y="{sy(v)-5.5:.1f}" width="11" height="11" '
                 f'fill="{RD}" stroke="#0a0e14" stroke-width="2"/>')
    if with_forest:
        fx = lambda k: sx(DEPTHS.index(FOREST_DEPTHS[k]))
        fp = " ".join(f"{fx(k):.1f},{sy(v):.1f}" for k, v in enumerate(FOREST_CV))
        o.append(f'<polyline points="{fp}" fill="none" stroke="{LM}" stroke-width="3.5" '
                 f'stroke-dasharray="9 6" stroke-linejoin="round"/>')
        for k, v in enumerate(FOREST_CV):   # 마름모 마커
            cx, cy = fx(k), sy(v)
            o.append(f'<polygon points="{cx:.1f},{cy-7:.1f} {cx+7:.1f},{cy:.1f} '
                     f'{cx:.1f},{cy+7:.1f} {cx-7:.1f},{cy:.1f}" fill="{LM}" '
                     f'stroke="#0a0e14" stroke-width="2"/>')
        o.append(_t(fx(3) + 12, sy(FOREST_CV[3]) - 6, "Forest CV .6421", LM, 15, "start", LATO, "700"))
        o.append(_t(fx(3) + 12, sy(FOREST_CV[3]) + 13, "깊이를 풀어도 거의 안 무너진다", MUTE, 13, "start"))
    # 최적점 / 붕괴점 주석
    o.append(f'<circle cx="{sx(1):.1f}" cy="{sy(TREE_CV[1]):.1f}" r="12" fill="none" '
             f'stroke="{LM}" stroke-width="2.5"/>')
    o.append(_t(sx(1), sy(TREE_CV[1]) + 34, "최적 깊이 2", LM, 15, "middle", LATO, "700"))
    o.append(_t(sx(1), sy(TREE_CV[1]) + 52, "CV .6355", LM, 13.5))
    o.append(_t(sx(5) - 14, sy(1.0) - 12, "train 1.0000", CY, 15, "end", LATO, "700"))
    o.append(_t(sx(5) + 12, sy(TREE_CV[5]) + 5, "CV .5185", RD, 15, "start", LATO, "700"))
    o.append(_t(sx(5) + 12, sy(TREE_CV[5]) + 24, "≈ 동전 던지기", RD, 13, "start"))
    # 갭 라벨
    o.append(f'<line x1="{sx(4):.1f}" y1="{sy(TREE_TRAIN[4]):.1f}" x2="{sx(4):.1f}" '
             f'y2="{sy(TREE_CV[4]):.1f}" stroke="{RD}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    o.append(_t(sx(4) - 10, (sy(TREE_TRAIN[4]) + sy(TREE_CV[4])) / 2, "과적합 갭", RD, 15, "end", LATO, "700"))
    # 축
    for k, d in enumerate(DEPTHS):
        o.append(_t(sx(k), T + ph + 26, d, MUTE, 15))
        o.append(_t(sx(k), T + ph + 45, f"리프 {LEAVES[k]}", FAINT, 12.5))
    o.append(_t(L + pw / 2, H - 2, "트리 깊이", INK, 15))
    o.append(_t(16, T + ph / 2, "AUC", INK, 15, "middle", LATO, None,
                f' transform="rotate(-90 16 {T+ph/2:.1f})"'))
    # 범례
    ly = T + 6
    lx = L + pw + 22
    o.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{CY}" stroke-width="3.5"/>')
    o.append(f'<circle cx="{lx+13}" cy="{ly}" r="6" fill="{CY}" stroke="#0a0e14" stroke-width="2"/>')
    o.append(_t(lx + 34, ly + 5, "train (외운 것)", INK, 15, "start"))
    o.append(f'<line x1="{lx}" y1="{ly+26}" x2="{lx+26}" y2="{ly+26}" stroke="{RD}" stroke-width="3.5"/>')
    o.append(f'<rect x="{lx+7.5}" y="{ly+20.5}" width="11" height="11" fill="{RD}" stroke="#0a0e14" stroke-width="2"/>')
    o.append(_t(lx + 34, ly + 31, "CV (새 데이터)", INK, 15, "start"))
    if with_forest:
        o.append(f'<line x1="{lx}" y1="{ly+52}" x2="{lx+26}" y2="{ly+52}" stroke="{LM}" '
                 f'stroke-width="3.5" stroke-dasharray="9 6"/>')
        o.append(f'<polygon points="{lx+13},{ly+45} {lx+20},{ly+52} {lx+13},{ly+59} {lx+6},{ly+52}" '
                 f'fill="{LM}" stroke="#0a0e14" stroke-width="2"/>')
        o.append(_t(lx + 34, ly + 57, "Forest CV", INK, 15, "start"))
        o.append(_t(lx, ly + 84, "실측 깊이 3·5·8·없음", FAINT, 12.5, "start"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 3) 슬라이드 13 — 4모델 CV AUC 막대 (Dummy 를 기준선으로)
#    출처: session6.html 슬라이드13 표
# ─────────────────────────────────────────────────────────────────────────────
# (이름, 선택된 설정, CV AUC, 색, AP, recall, train AUC) — 슬라이드13 표 그대로
MODELS_A = [("Dummy", "—", .5000, GY, .3371, .0000, .5000),
            ("DecisionTree", "depth=2", .6355, RD, .4372, .6490, .6637),
            ("LogisticRegression", "C=0.1", .6535, CY, .4921, .6183, .6898),
            ("RandomForest", "depth=3", .6651, LM, .5113, .5985, .7368)]


def model_bars():
    W, H = 1140, 300
    L, T, bh, gap = 250, 22, 40, 24
    pw = W - L - 480
    lo, hi = .48, .70
    sx = lambda v: L + (v - lo) / (hi - lo) * pw
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="Model A 네 모델의 교차검증 AUC 비교">']
    for gv in (.50, .55, .60, .65, .70):
        o.append(f'<line x1="{sx(gv):.1f}" y1="{T-6}" x2="{sx(gv):.1f}" y2="{T+4*(bh+gap)-gap+6}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(_t(sx(gv), T + 4 * (bh + gap) + 6, f"{gv:.2f}", FAINT, 13))
    for i, (name, cfg, v, c, ap, rc, tr) in enumerate(MODELS_A):
        y = T + i * (bh + gap)
        o.append(_t(L - 16, y + bh / 2 + 1, name, "#ffffff", 17, "end", POP, "700"))
        o.append(_t(L - 16, y + bh / 2 + 19, cfg, FAINT, 13, "end"))
        o.append(f'<rect x="{L}" y="{y}" width="{max(sx(v)-L,3):.1f}" height="{bh}" '
                 f'rx="4" fill="{c}" fill-opacity="{".35" if name=="Dummy" else ".92"}"/>')
        o.append(_t(sx(v) + 12, y + bh / 2 - 3, f"{v:.4f}", c, 20, "start", POP, "700"))
        o.append(_t(sx(v) + 12, y + bh / 2 + 16,
                    f"AP {ap:.4f}  ·  recall {rc:.4f}  ·  train {tr:.4f}", FAINT, 12.5, "start"))
    # Dummy 기준선
    o.append(f'<line x1="{sx(.5):.1f}" y1="{T-6}" x2="{sx(.5):.1f}" y2="{T+4*(bh+gap)-gap+6}" '
             f'stroke="{GY}" stroke-width="2" stroke-dasharray="6 5"/>')
    # 격차 표시 — 로지스틱 ↔ 포레스트
    yl = T + 2 * (bh + gap) + bh / 2
    yf = T + 3 * (bh + gap) + bh / 2
    o.append(f'<path d="M{sx(.6535)+238:.1f},{yl:.1f} L{sx(.6535)+254:.1f},{yl:.1f} '
             f'L{sx(.6535)+254:.1f},{yf:.1f} L{sx(.6651)+238:.1f},{yf:.1f}" fill="none" '
             f'stroke="{LM}" stroke-width="2"/>')
    o.append(_t(sx(.6535) + 264, (yl + yf) / 2 + 5, "+0.0116", LM, 19, "start", POP, "700"))
    o.append(_t(L + pw / 2, T + 4 * (bh + gap) + 34, "세로 점선 = Dummy .5000 (동전 던지기)  ·  막대 길이 = CV AUC", FAINT, 13.5))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 4) 슬라이드 14 — 폴드별 페어 비교 + 시드 7개 승패
#    출처: session6.html 슬라이드14 코드블록 · _build_s6.py 독스트링
# ─────────────────────────────────────────────────────────────────────────────
FOLD_LOG    = [.591, .679, .639, .663, .695]
FOLD_FOR    = [.603, .683, .648, .676, .715]
FOLD_DELTA  = [.012, .004, .009, .013, .020]
SEED_WINS   = 6   # CV 분할 seed 7개 중 포레스트 승 6


def fold_pairs():
    W, H = 1140, 300
    L, T = 76, 30
    pw, ph = 700, 196
    lo, hi = .575, .725
    sy = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    sx = lambda k: L + 52 + (pw - 104) * (k / 4)
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="폴드별 로지스틱 대 포레스트 AUC 와 시드 7개 승패">']
    for gv in (.60, .65, .70):
        o.append(f'<line x1="{L}" y1="{sy(gv):.1f}" x2="{L+pw}" y2="{sy(gv):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(_t(L - 12, sy(gv) + 5, f"{gv:.2f}", FAINT, 13, "end"))
    for k in range(5):
        a, b = FOLD_LOG[k], FOLD_FOR[k]
        x = sx(k)
        o.append(f'<line x1="{x:.1f}" y1="{sy(a):.1f}" x2="{x:.1f}" y2="{sy(b):.1f}" '
                 f'stroke="{LM}" stroke-width="3"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{sy(a):.1f}" r="7" fill="{CY}" stroke="#0a0e14" stroke-width="2"/>')
        o.append(f'<polygon points="{x:.1f},{sy(b)-8:.1f} {x+8:.1f},{sy(b):.1f} '
                 f'{x:.1f},{sy(b)+8:.1f} {x-8:.1f},{sy(b):.1f}" fill="{LM}" stroke="#0a0e14" stroke-width="2"/>')
        o.append(_t(x, sy(b) - 16, f"+{FOLD_DELTA[k]:.3f}".replace("0.", "."), LM, 14, "middle", LATO, "700"))
        o.append(_t(x, T + ph + 24, f"폴드 {k+1}", MUTE, 14))
    o.append(_t(L + pw / 2, 16, "폴드 5개 전부 포레스트가 위 — 방향이 한 번도 안 뒤집힌다", "#ffffff", 16, "middle", LATO, "700"))
    o.append(f'<circle cx="{L+16}" cy="{T+18}" r="7" fill="{CY}" stroke="#0a0e14" stroke-width="2"/>')
    o.append(_t(L + 30, T + 23, "로지스틱", CY, 15, "start", LATO, "700"))
    o.append(f'<polygon points="{L+16},{T+41} {L+24},{T+49} {L+16},{T+57} {L+8},{T+49}" fill="{LM}" stroke="#0a0e14" stroke-width="2"/>')
    o.append(_t(L + 30, T + 54, "포레스트", LM, 15, "start", LATO, "700"))
    o.append(_t(L + pw / 2, H - 10, "세로선 = 같은 폴드에서의 차이 (포레스트 − 로지스틱)", FAINT, 13))
    # 오른쪽: 시드 7개 승패 타일
    bx, by = L + pw + 76, T + 26
    o.append(_t(bx, by - 26, "CV 분할 seed 7개", "#ffffff", 16, "start", POP, "700"))
    for i in range(7):
        cx = bx + (i % 4) * 46
        cy = by + (i // 4) * 46
        win = i < SEED_WINS
        o.append(f'<rect x="{cx}" y="{cy}" width="34" height="34" rx="7" '
                 f'fill="{LM if win else CY}" fill-opacity="{".9" if win else ".55"}"/>')
        o.append(_t(cx + 17, cy + 23, "F" if win else "L", "#0a0e14", 17, "middle", POP, "700"))
    o.append(_t(bx, by + 118, f"포레스트 {SEED_WINS}승 / 로지스틱 1승", LM, 16, "start", LATO, "700"))
    o.append(_t(bx, by + 140, "폴드를 어떻게 갈라도", MUTE, 13.5, "start"))
    o.append(_t(bx, by + 158, "대체로 같은 결론", MUTE, 13.5, "start"))
    o.append(_t(bx, by + 184, "※ 어느 seed 가 뒤집혔는지는", FAINT, 12, "start"))
    o.append(_t(bx, by + 200, "   기록에 없어 순서는 임의 배치", FAINT, 12, "start"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 5) 슬라이드 17 — Model A → B 슬로프(범프) 차트
#    출처: session6.html 슬라이드17 표
# ─────────────────────────────────────────────────────────────────────────────
AB = [("RandomForest", .6651, .6987, LM),
      ("LogisticRegression", .6535, .6825, CY),
      ("DecisionTree", .6355, .6833, RD),
      ("Dummy", .5000, .5000, GY)]


def _declutter(items, min_gap):
    """(y, payload) 목록을 원래 순서를 유지한 채 최소 간격만큼 벌린다."""
    items = sorted(items, key=lambda t: t[0])
    ys = [y for y, _ in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return [(ys[i], items[i][0], items[i][1]) for i in range(len(items))]


def bump_chart():
    """Model A → Model B 슬로프 차트.
    Dummy(.5000 → .5000)는 축을 3배로 늘려 나머지를 뭉개므로 캡션으로만 밝힌다."""
    W, H = 1140, 300
    T, ph = 58, 186
    xa, xb = 392, 748
    lo, hi = .622, .708
    sy = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="Model A 에서 Model B 로 갈 때 세 모델의 CV AUC 변화">']
    for gv in (.63, .65, .67, .69):
        o.append(f'<line x1="{xa}" y1="{sy(gv):.1f}" x2="{xb}" y2="{sy(gv):.1f}" stroke="{GRID}" stroke-width="1"/>')
    o.append(f'<line x1="{xa}" y1="{T-18}" x2="{xa}" y2="{T+ph+14}" stroke="{FRAME}" stroke-width="1.5"/>')
    o.append(f'<line x1="{xb}" y1="{T-18}" x2="{xb}" y2="{T+ph+14}" stroke="{FRAME}" stroke-width="1.5"/>')
    o.append(_t(xa, T - 46, "Model A", "#ffffff", 19, "middle", POP, "700"))
    o.append(_t(xa, T - 28, "기본 변수", MUTE, 14))
    o.append(_t(xb, T - 46, "Model B", "#ffffff", 19, "middle", POP, "700"))
    o.append(_t(xb, T - 28, "강한 단일 변수 추가", MUTE, 14))
    live = [(n, a, b, c) for n, a, b, c in AB if n != "Dummy"]
    for name, a, b, c in live:
        o.append(f'<line x1="{xa}" y1="{sy(a):.1f}" x2="{xb}" y2="{sy(b):.1f}" stroke="{c}" '
                 f'stroke-width="3.5" stroke-linecap="round"/>')
        o.append(f'<circle cx="{xa}" cy="{sy(a):.1f}" r="7.5" fill="{c}" stroke="#0a0e14" stroke-width="2.5"/>')
        o.append(f'<circle cx="{xb}" cy="{sy(b):.1f}" r="7.5" fill="{c}" stroke="#0a0e14" stroke-width="2.5"/>')
    # 라벨은 겹치지 않도록 벌리고, 점과는 유도선으로 잇는다
    for side in ("A", "B"):
        rows = [(sy(a if side == "A" else b), (name, a, b, c)) for name, a, b, c in live]
        for ly, py, (name, a, b, c) in _declutter(rows, 42):
            if side == "A":
                lx, anch = xa - 22, "end"
                txt, sub = f"{name}", f"{a:.4f}"
                o.append(f'<path d="M{xa-9:.1f},{py:.1f} L{lx+6:.1f},{ly:.1f}" fill="none" stroke="{c}" stroke-width="1.2" opacity=".6"/>')
                o.append(_t(lx, ly - 2, txt, c, 16, anch, LATO, "700"))
                o.append(_t(lx, ly + 16, sub, "#ffffff", 15, anch, POP, "700"))
            else:
                lx, anch = xb + 22, "start"
                o.append(f'<path d="M{xb+9:.1f},{py:.1f} L{lx-6:.1f},{ly:.1f}" fill="none" stroke="{c}" stroke-width="1.2" opacity=".6"/>')
                o.append(_t(lx, ly - 2, f"{b:.4f}", "#ffffff", 15, anch, POP, "700"))
                o.append(_t(lx, ly + 16, f"+{b-a:.4f}", c, 14.5, anch, LATO, "700"))
    o.append(_t(W / 2, H - 4, "Dummy 는 A·B 모두 .5000 (축 밖)  ·  세로 눈금은 생략 — 모든 점에 값이 붙어 있다", FAINT, 12.5))
    return "\n".join(o) + "\n</svg>"


def _node(x, y, w, h, label, fill="#151a21", stroke=CY, size=15, sub=None, sub_c=None):
    o = [f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w}" height="{h}" rx="9" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>']
    if sub:
        o.append(_t(x, y - 2, label, "#ffffff", size, "middle", LATO, "700"))
        o.append(_t(x, y + 16, sub, sub_c or MUTE, size - 2))
    else:
        o.append(_t(x, y + size / 2 - 1, label, "#ffffff", size, "middle", LATO, "700"))
    return "\n".join(o)


def _edge(x1, y1, x2, y2, color=FRAME, w=2):
    return (f'<path d="M{x1:.1f},{y1:.1f} C{x1:.1f},{(y1+y2)/2:.1f} {x2:.1f},{(y1+y2)/2:.1f} '
            f'{x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-width="{w}"/>')


# ─────────────────────────────────────────────────────────────────────────────
# 6) 슬라이드 8 — 결정 트리 개념 다이어그램 (스무고개)
#    분기 임계값 출처: session6.html 슬라이드8 코드블록
# ─────────────────────────────────────────────────────────────────────────────
def tree_concept():
    W, H = 620, 300
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="결정 트리가 질문을 이어가며 데이터를 나누는 구조">']
    o.append(_edge(310, 52, 150, 118)); o.append(_edge(310, 52, 470, 118))
    o.append(_edge(150, 152, 72, 218)); o.append(_edge(150, 152, 228, 218))
    o.append(_edge(470, 152, 392, 218)); o.append(_edge(470, 152, 548, 218))
    o.append(_t(215, 92, "예", LM, 15, "middle", LATO, "700"))
    o.append(_t(405, 92, "아니오", MUTE, 15, "middle", LATO, "700"))
    o.append(_node(310, 34, 300, 40, "친구지지 ≤ 4.07 ?", "#151a21", CY, 16))
    o.append(_node(150, 134, 244, 40, "자아존중감 ≤ 2.88 ?", "#151a21", CY, 15))
    o.append(_node(470, 134, 224, 40, "우울 ≤ 1.25 ?", "#151a21", CY, 15))
    for cx, txt, col in ((72, "고스트레스", RD), (228, "일반", GY), (392, "일반", GY), (548, "고스트레스", RD)):
        o.append(f'<rect x="{cx-64}" y="{218-17}" width="128" height="34" rx="17" '
                 f'fill="{col}" fill-opacity=".16" stroke="{col}" stroke-width="1.5"/>')
        o.append(_t(cx, 218 + 6, txt, col, 15, "middle", LATO, "700"))
    o.append(_t(W / 2, 275, "마지막 칸(잎, leaf)의 고스트레스 비율이 곧 예측 확률이다", MUTE, 14))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 7) 슬라이드 9 — Model B 트리가 찾은 상호작용
#    출처: session6.html 슬라이드9 코드블록
# ─────────────────────────────────────────────────────────────────────────────
def interaction_tree():
    W, H = 1140, 306
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="이전 스트레스 수준에 따라 다음 질문이 달라지는 상호작용 구조">']
    o.append(_edge(570, 62, 300, 138, LM, 2.5)); o.append(_edge(570, 62, 840, 138, LM, 2.5))
    o.append(_edge(300, 178, 300, 236)); o.append(_edge(840, 178, 840, 236))
    o.append(_t(330, 106, "이전 스트레스 낮음", LM, 15, "middle", LATO, "700"))
    o.append(_t(810, 106, "이전 스트레스 높음", AM, 15, "middle", LATO, "700"))
    o.append(_node(570, 40, 420, 44, "previous_acculturative_stress ≤ 1.45 ?", "#101820", LM, 17))
    o.append(_t(570, 16, "1차 분기 — 5차시 로지스틱 계수 1위와 같은 변수", LM, 14))
    o.append(_node(300, 158, 300, 44, "자아존중감 ≤ 2.62 ?", "#151a21", CY, 17))
    o.append(_node(840, 158, 300, 44, "우울 ≤ 1.25 ?", "#151a21", CY, 17))
    o.append(f'<rect x="{300-206}" y="236" width="412" height="46" rx="10" fill="{LM}" '
             f'fill-opacity=".10" stroke="{LM}" stroke-width="1.5"/>')
    o.append(_t(300, 258, "이 집단에서는 자아존중감이 갈림길", LM, 16, "middle", LATO, "700"))
    o.append(_t(300, 275, "낮으면 고스트레스 / 아니면 일반", MUTE, 13.5))
    o.append(f'<rect x="{840-206}" y="236" width="412" height="46" rx="10" fill="{AM}" '
             f'fill-opacity=".10" stroke="{AM}" stroke-width="1.5"/>')
    o.append(_t(840, 258, "이 집단에서는 우울이 갈림길", AM, 16, "middle", LATO, "700"))
    o.append(_t(840, 275, "낮으면 일반 / 아니면 고스트레스", MUTE, 13.5))
    o.append(_t(570, 200, "두 집단에 서로 다른 질문", "#ffffff", 16, "middle", POP, "700"))
    o.append(_t(570, 220, "= 상호작용", LM, 15, "middle", LATO, "700"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 8) 슬라이드 12 — 랜덤 포레스트 파이프라인
# ─────────────────────────────────────────────────────────────────────────────
def forest_pipeline():
    W, H = 1140, 216
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="원본 데이터에서 부트스트랩, 300그루 트리, 평균으로 이어지는 랜덤 포레스트 과정">']
    cy = 96
    stops = [(96, "원본 train", "1,056명", CY),
             (330, "부트스트랩 표본", "5차시에 배운 그것", CY),
             (582, "트리 300그루", "변수도 일부만 무작위", LM),
             (830, "300개 답을 평균", "오차가 상쇄된다", LM),
             (1054, "최종 예측", "덜 흔들린다", LM)]
    for i in range(len(stops) - 1):
        x1 = stops[i][0] + 74
        x2 = stops[i + 1][0] - 74
        o.append(f'<line x1="{x1}" y1="{cy}" x2="{x2-9}" y2="{cy}" stroke="{FRAME}" stroke-width="2"/>')
        o.append(f'<polygon points="{x2},{cy} {x2-11},{cy-6} {x2-11},{cy+6}" fill="{FRAME}"/>')
    # 300그루 표현: 작은 삼각형 격자
    for r in range(3):
        for c in range(7):
            tx = 582 - 66 + c * 22
            ty = cy - 34 + r * 24
            o.append(f'<polygon points="{tx},{ty-8} {tx+8},{ty+6} {tx-8},{ty+6}" fill="{LM}" '
                     f'fill-opacity="{0.30 + 0.07*r:.2f}"/>')
            o.append(f'<line x1="{tx}" y1="{ty+6}" x2="{tx}" y2="{ty+11}" stroke="{LM}" '
                     f'stroke-width="1.5" opacity=".5"/>')
    for i, (x, title, sub, col) in enumerate(stops):
        if i != 2:
            o.append(f'<rect x="{x-74}" y="{cy-40}" width="148" height="80" rx="12" '
                     f'fill="#151a21" stroke="{col}" stroke-width="2"/>')
            o.append(_t(x, cy - 4, title, "#ffffff", 16, "middle", LATO, "700"))
            o.append(_t(x, cy + 18, sub, MUTE, 13))
        else:
            o.append(_t(x, cy + 58, title, LM, 17, "middle", POP, "700"))
            o.append(_t(x, cy + 76, sub, MUTE, 13))
    o.append(_t(96, 24, "①", CY, 18, "middle", POP, "700"))
    o.append(_t(330, 24, "②", CY, 18, "middle", POP, "700"))
    o.append(_t(582, 24, "③", LM, 18, "middle", POP, "700"))
    o.append(_t(830, 24, "④", LM, 18, "middle", POP, "700"))
    o.append(_t(W / 2, H - 4, "3차시 심리척도와 같은 원리 — 한 문항은 흔들려도 여러 문항의 평균은 안정된다", MUTE, 14))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 9) 슬라이드 15 — 비용·편익 저울
#    출처: session6.html 슬라이드15 표
# ─────────────────────────────────────────────────────────────────────────────
def balance_scale():
    W, H = 1140, 316
    cx, py = 570, 74
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="AUC 0.012 를 얻는 대신 해석력 네 가지를 잃는 비용 편익 저울">']
    # 기둥 · 받침
    o.append(f'<line x1="{cx}" y1="{py}" x2="{cx}" y2="248" stroke="{FRAME}" stroke-width="5"/>')
    o.append(f'<polygon points="{cx-56},262 {cx+56},262 {cx+34},248 {cx-34},248" fill="{FRAME}"/>')
    # 빔 — 오른쪽(잃는 것)이 무거워 기운다
    ly, ry = py + 34, py - 26
    o.append(f'<line x1="{cx-330}" y1="{ly}" x2="{cx+330}" y2="{ry}" stroke="{FRAME}" stroke-width="5" stroke-linecap="round"/>')
    o.append(f'<circle cx="{cx}" cy="{py+4}" r="9" fill="{FRAME}"/>')
    # 왼쪽 접시 — 얻는 것
    o.append(f'<line x1="{cx-330}" y1="{ly}" x2="{cx-330}" y2="{ly+30}" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<path d="M{cx-410},{ly+30} Q{cx-330},{ly+72} {cx-250},{ly+30} Z" fill="{LM}" fill-opacity=".14" stroke="{LM}" stroke-width="2"/>')
    o.append(_t(cx - 330, ly - 14, "얻는 것", LM, 17, "middle", POP, "700"))
    o.append(f'<rect x="{cx-408}" y="{ly+38}" width="156" height="52" rx="10" fill="{LM}" fill-opacity=".16" stroke="{LM}" stroke-width="2"/>')
    o.append(_t(cx - 330, ly + 66, "CV AUC +0.0116", LM, 19, "middle", POP, "700"))
    o.append(_t(cx - 330, ly + 84, "일관되지만 아주 작다", MUTE, 13))
    # 오른쪽 접시 — 잃는 것 4가지 (표에서 ❌/△ 인 항목)
    o.append(f'<line x1="{cx+330}" y1="{ry}" x2="{cx+330}" y2="{ry+30}" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<path d="M{cx+250},{ry+30} Q{cx+330},{ry+72} {cx+410},{ry+30} Z" fill="{RD}" fill-opacity=".14" stroke="{RD}" stroke-width="2"/>')
    o.append(_t(cx + 330, ry - 14, "잃는 것", RD, 17, "middle", POP, "700"))
    losses = [("변수별 방향 (+/−)", "계수 부호가 사라진다"),
              ("변수별 크기", "표준화 계수가 사라진다"),
              ("불확실성", "부트스트랩 신뢰구간 불가"),
              ("사람이 읽는 것", "18줄 표 → 나무 300그루")]
    for i, (a, b) in enumerate(losses):
        y = ry + 42 + i * 46
        o.append(f'<rect x="{cx+186}" y="{y}" width="290" height="38" rx="8" fill="{RD}" fill-opacity=".12" stroke="{RD}" stroke-width="1.5"/>')
        o.append(_t(cx + 200, y + 24, a, RD, 15, "start", LATO, "700"))
        o.append(_t(cx + 466, y + 24, b, MUTE, 12.5, "end"))
    o.append(_t(cx, 296, "이 미세한 성능 향상을 위해 5차시에서 얻은 해석의 표를 통째로 버릴 것인가?", "#ffffff", 17, "middle", LATO, "700"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 10) 슬라이드 16 — 세 가지 증거 (신전 3기둥)
# ─────────────────────────────────────────────────────────────────────────────
def three_pillars():
    W, H = 1140, 322
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="복잡한 모델이 늘 더 좋은 것은 아니라는 결론을 받치는 세 가지 증거">']
    # 지붕
    o.append(f'<polygon points="{W/2},6 {W-90},70 90,70" fill="#151a21" stroke="{LM}" stroke-width="2"/>')
    o.append(_t(W / 2, 56, "복잡 ≠ 더 좋음", "#ffffff", 26, "middle", POP, "700"))
    o.append(f'<rect x="70" y="76" width="{W-140}" height="14" rx="4" fill="{FRAME}"/>')
    cols = [("① 유연해도 진다", CY,
             ["단일 트리 .6355", "< 로지스틱 .6535", "", "비선형·상호작용을 잡을 수", "있는 모델인데 더 못한다"]),
            ("② 복잡할수록 나빠진다", RD,
             ["깊이 2 → 없음", "CV .6355 → .5185", "", "train 은 1.0 까지 오르는데", "CV 는 동전 던지기로 내려간다"]),
            ("③ 이겨도 +0.012 다", LM,
             ["포레스트 .6651", "> 로지스틱 .6535", "", "300그루를 얻고", "방향·크기·불확실성을 잃는다"])]
    bw, gap = 330, 45
    x0 = (W - (bw * 3 + gap * 2)) / 2
    for i, (title, col, lines) in enumerate(cols):
        x = x0 + i * (bw + gap)
        o.append(f'<rect x="{x}" y="102" width="{bw}" height="176" rx="10" fill="#151a21" stroke="{col}" stroke-width="2"/>')
        o.append(f'<rect x="{x}" y="102" width="{bw}" height="5" rx="2.5" fill="{col}"/>')
        o.append(_t(x + bw / 2, 136, title, col, 19, "middle", POP, "700"))
        for k, ln in enumerate(lines):
            if not ln:
                continue
            bold = k < 2
            o.append(_t(x + bw / 2, 166 + k * 22, ln, "#ffffff" if bold else MUTE,
                        16 if bold else 14, "middle", LATO, "700" if bold else None))
    o.append(f'<rect x="70" y="286" width="{W-140}" height="14" rx="4" fill="{FRAME}"/>')
    o.append(_t(W / 2, 316, "슬로건 하나만 외우지 말고 — 이 세 증거를 순서대로 펼칠 수 있어야 한다", MUTE, 14))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 11) 슬라이드 6 — 선형성 가정: 같은 1점, 같은 무게인가 (계단)
# ─────────────────────────────────────────────────────────────────────────────
def linearity_stairs():
    W, H = 560, 322
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="점수 구간마다 1점의 무게가 다를 수 있음을 보여주는 계단 그림">']
    base = 246
    steps = [(60, 40), (160, 88), (260, 118), (360, 136), (460, 146)]
    prev_x, prev_y = 0, base
    for i, (x, h) in enumerate(steps):
        y = base - h
        o.append(f'<rect x="{x-46}" y="{y}" width="92" height="{h}" fill="{CY}" fill-opacity="{0.10+0.05*i:.2f}" stroke="{CY}" stroke-width="1.5"/>')
        o.append(_t(x, base + 22, f"{i+1}점", MUTE, 14))
    # 1→2 큰 격차 vs 4→5 작은 격차 (개념 도해 — 실측치가 아니라 모양만 보여준다)
    o.append(f'<line x1="106" y1="{base-40}" x2="106" y2="{base-88}" stroke="{RD}" stroke-width="3"/>')
    o.append(f'<line x1="100" y1="{base-40}" x2="112" y2="{base-40}" stroke="{RD}" stroke-width="3"/>')
    o.append(f'<line x1="100" y1="{base-88}" x2="112" y2="{base-88}" stroke="{RD}" stroke-width="3"/>')
    o.append(_t(96, base - 58, "1→2점", RD, 14, "end", LATO, "700"))
    o.append(_t(96, base - 42, "크게 달라진다", RD, 12.5, "end"))
    o.append(f'<line x1="410" y1="{base-136}" x2="410" y2="{base-146}" stroke="{LM}" stroke-width="3"/>')
    o.append(f'<line x1="404" y1="{base-136}" x2="416" y2="{base-136}" stroke="{LM}" stroke-width="3"/>')
    o.append(f'<line x1="404" y1="{base-146}" x2="416" y2="{base-146}" stroke="{LM}" stroke-width="3"/>')
    o.append(_t(410, base - 174, "4→5점", LM, 14, "middle", LATO, "700"))
    o.append(_t(410, base - 158, "거의 그대로", LM, 12.5))
    o.append(_t(W / 2, 22, "실제 마음 — 같은 1점이 같은 무게가 아니다", "#ffffff", 17, "middle", LATO, "700"))
    o.append(_t(W / 2, 42, "바닥에서의 1점 ≫ 이미 높은 쪽에서의 1점", MUTE, 13.5))
    o.append(f'<line x1="40" y1="{base}" x2="{W-30}" y2="{base}" stroke="{FRAME}" stroke-width="2"/>')
    o.append(_t(W / 2, H - 8, "로지스틱은 이 계단이 전부 같은 높이라고 가정한다", AM, 15, "middle", LATO, "700"))
    o.append(_t(W / 2, H - 26, "※ 모양을 보여주는 개념도 — 눈금은 실측치가 아니다", FAINT, 12))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 12) 슬라이드 1 — 타이틀 모티프: 엉킨 곡선(복잡) vs 곧은 직선(단순)
# ─────────────────────────────────────────────────────────────────────────────
def title_motif():
    W, H = 720, 190
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" aria-hidden="true">']
    o.append(f'<path d="M120,150 C160,40 210,168 258,72 C300,-10 330,168 372,96 '
             f'C410,30 436,170 480,90 C514,28 552,150 600,110" fill="none" stroke="{RD}" '
             f'stroke-width="5" stroke-linecap="round" opacity=".85"/>')
    o.append(f'<line x1="96" y1="132" x2="624" y2="66" stroke="{CY}" stroke-width="6" stroke-linecap="round"/>')
    o.append(_t(120, 178, "복잡한 모델", RD, 16, "start", LATO, "700"))
    o.append(_t(600, 40, "단순한 모델", CY, 16, "end", LATO, "700"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 13) 슬라이드 3 — 7 Step 진행 레일 (두 '봉우리' 강조)
# ─────────────────────────────────────────────────────────────────────────────
def agenda_rail():
    W, H = 1140, 132
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="오늘 수업의 일곱 단계와 두 개의 고비">']
    stops = [("0~1", "선형 가정 점검", False), ("2", "결정 트리", False),
             ("3", "과적합", True), ("4", "랜덤 포레스트", False),
             ("5", "4모델 판정", True), ("6", "A vs B", False), ("7", "산출물", False)]
    y = 56
    x0, x1 = 78, W - 78
    o.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{FRAME}" stroke-width="3"/>')
    for i, (num, label, peak) in enumerate(stops):
        x = x0 + (x1 - x0) * i / (len(stops) - 1)
        col = LM if peak else CY
        r = 27 if peak else 20
        if peak:
            o.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r+7}" fill="none" stroke="{col}" stroke-width="1.5" opacity=".4"/>')
        o.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r}" fill="#151a21" stroke="{col}" stroke-width="{3 if peak else 2}"/>')
        o.append(_t(x, y + 6, num, col, 17 if peak else 15, "middle", POP, "700"))
        o.append(_t(x, y + r + 24, label, "#ffffff" if peak else MUTE, 14.5 if peak else 13.5,
                    "middle", LATO, "700" if peak else None))
        if peak:
            o.append(_t(x, y - r - 14, "고비", LM, 13, "middle", LATO, "700"))
    o.append(_t(W / 2, 20, "Step", FAINT, 13))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 14) 슬라이드 10 — 같은 증상(AUC 1.0), 다른 병
# ─────────────────────────────────────────────────────────────────────────────
def same_symptom():
    W, H = 1140, 240
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="AUC 1.0 이라는 같은 증상이 누출과 과적합이라는 서로 다른 원인에서 나온다">']
    o.append(_t(W / 2, 62, "train AUC  1.0000", RD, 54, "middle", POP, "700"))
    o.append(_t(W / 2, 88, "같은 증상 — train 점수만 보면 둘 다 못 알아챈다", MUTE, 15))
    cards = [("4차시 · 누출 (Leakage)", CY, "알면 안 되는 정보가 들어온 것",
              "데이터 흐름의 병", "미래 정보 · 라벨 정보가 X 에 섞였다"),
             ("6차시 · 과적합 (Overfitting)", LM, "주어진 정보를 통째로 외운 것",
              "모델 복잡도의 병", "train 1,056명을 리프 299칸에 나눠 담았다")]
    bw, gap = 500, 60
    x0 = (W - (bw * 2 + gap)) / 2
    for i, (title, col, one, two, three) in enumerate(cards):
        x = x0 + i * (bw + gap)
        o.append(f'<rect x="{x}" y="112" width="{bw}" height="112" rx="12" fill="#151a21" stroke="{col}" stroke-width="2"/>')
        o.append(f'<rect x="{x}" y="112" width="{bw}" height="5" rx="2.5" fill="{col}"/>')
        o.append(_t(x + bw / 2, 142, title, col, 18, "middle", POP, "700"))
        o.append(_t(x + bw / 2, 168, one, "#ffffff", 16))
        o.append(_t(x + bw / 2, 190, two, col, 14, "middle", LATO, "700"))
        o.append(_t(x + bw / 2, 212, three, MUTE, 13))
    o.append(f'<line x1="{W/2}" y1="112" x2="{W/2}" y2="224" stroke="{FRAME}" stroke-width="1.5" stroke-dasharray="6 6"/>')
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 15) 슬라이드 18 — 봉인된 test 와 8차시까지의 순서
# ─────────────────────────────────────────────────────────────────────────────
def sealed_test():
    W, H = 1140, 232
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="6차시 선택, 7차시 해석, 8차시에 단 한 번 test 를 여는 순서">']
    y = 92
    steps = [("6차시 · 오늘", "모델 선택 완료", CY, True),
             ("7차시", "변수 중요도 해석", CY, True),
             ("8차시", "test 를 딱 한 번", LM, False)]
    bw, gap = 300, 84
    x0 = (W - (bw * 3 + gap * 2)) / 2
    for i, (t, s, col, done) in enumerate(steps):
        x = x0 + i * (bw + gap)
        o.append(f'<rect x="{x}" y="{y-42}" width="{bw}" height="84" rx="12" fill="#151a21" stroke="{col}" stroke-width="2"/>')
        o.append(_t(x + bw / 2, y - 8, t, col, 19, "middle", POP, "700"))
        o.append(_t(x + bw / 2, y + 18, s, "#ffffff", 16))
        if i < 2:
            ax = x + bw + 14
            o.append(f'<line x1="{ax}" y1="{y}" x2="{ax+42}" y2="{y}" stroke="{FRAME}" stroke-width="2.5"/>')
            o.append(f'<polygon points="{ax+56},{y} {ax+42},{y-7} {ax+42},{y+7}" fill="{FRAME}"/>')
    # 자물쇠 — 6·7차시 위에 걸린 봉인
    lx = x0 + bw + gap / 2
    for lx in (x0 + bw / 2, x0 + bw + gap + bw / 2):
        o.append(f'<path d="M{lx-11},{y-58} a11,11 0 0 1 22,0 v9 h-6 v-9 a5,5 0 0 0 -10,0 v9 h-6 z" fill="{RD}"/>')
        o.append(f'<rect x="{lx-15}" y="{y-49}" width="30" height="22" rx="4" fill="{RD}"/>')
    o.append(_t(x0 + bw + gap / 2, y - 70, "test 봉인 중", RD, 15, "middle", LATO, "700"))
    o.append(f'<rect x="{x0}" y="{y+64}" width="{bw*3+gap*2}" height="46" rx="10" fill="{RD}" fill-opacity=".10" stroke="{RD}" stroke-width="1.5"/>')
    o.append(_t(W / 2, y + 86, "훔쳐보고 싶은 유혹이 가장 큰 날이 바로 오늘이다 — 그래서 오늘 안 여는 것이 규칙이다",
                "#ffffff", 17, "middle", LATO, "700"))
    o.append(_t(W / 2, y + 128, "코드로 막을 수 없는 '사람 머릿속의 누출'을 막기 위한 절차다", MUTE, 14))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 16) 슬라이드 19 — 후보 수와 선택 편향
#     출처: session6.html 슬라이드19 코드블록 (포레스트 4 · 트리 4 · 로지스틱 3 · Dummy 0)
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATES = [("RandomForest", 4, LM), ("DecisionTree", 4, RD),
              ("LogisticRegression", 3, CY), ("Dummy", 0, GY)]


def candidate_bias():
    W, H = 1140, 200
    L, T, bh, gap = 260, 34, 30, 14
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="모델별 하이퍼파라미터 후보 개수와 그로 인한 채점 편향">']
    cw = 44
    for i, (name, n, col) in enumerate(CANDIDATES):
        y = T + i * (bh + gap)
        o.append(_t(L - 16, y + bh / 2 + 6, name, "#ffffff", 16, "end", LATO, "700"))
        for k in range(max(n, 1)):
            if n == 0:
                o.append(f'<rect x="{L}" y="{y}" width="{cw}" height="{bh}" rx="6" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="4 4"/>')
                o.append(_t(L + cw / 2, y + bh / 2 + 6, "0", col, 15, "middle", POP, "700"))
                break
            o.append(f'<rect x="{L+k*(cw+10)}" y="{y}" width="{cw}" height="{bh}" rx="6" fill="{col}" fill-opacity=".8"/>')
        if n:
            o.append(_t(L + n * (cw + 10) + 8, y + bh / 2 + 6, f"후보 {n}개", col, 15, "start", LATO, "700"))
    # 로지스틱에는 없고 트리·포레스트에만 있는 '4번째 후보'를 짚는다
    x4 = L + 3 * (cw + 10)
    o.append(f'<rect x="{x4-7}" y="{T-7}" width="{cw+14}" height="{2*(bh+gap)-gap+14}" rx="9" '
             f'fill="none" stroke="{AM}" stroke-width="1.8" stroke-dasharray="6 5"/>')
    o.append(_t(x4 + cw / 2, T - 16, "이 한 칸이 차이다", AM, 13.5, "middle", LATO, "700"))
    tx = L + 430
    o.append(_t(tx, T + 30, "후보가 많을수록", AM, 16, "start", LATO, "700"))
    o.append(_t(tx, T + 52, "\"운 좋게 잘 나온 값\"이 뽑힐 확률도 커진다", MUTE, 14, "start"))
    o.append(_t(tx, T + 74, "→ 복잡한 모델 쪽이", MUTE, 14, "start"))
    o.append(_t(tx, T + 96, "아주 약간 유리하게 채점됐다", AM, 14, "start", LATO, "700"))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 17) 슬라이드 21 — 다음 여정: 블랙박스에 손전등
# ─────────────────────────────────────────────────────────────────────────────
def black_box():
    W, H = 560, 210
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" aria-hidden="true">']
    o.append(f'<defs><radialGradient id="beam6"><stop offset="0%" stop-color="{AM}" stop-opacity=".55"/>'
             f'<stop offset="100%" stop-color="{AM}" stop-opacity="0"/></radialGradient></defs>')
    o.append(f'<polygon points="150,52 350,52 400,26 200,26" fill="#1c222c" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<rect x="150" y="52" width="200" height="118" fill="#12161d" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<polygon points="350,52 400,26 400,144 350,170" fill="#0d1117" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<ellipse cx="250" cy="110" rx="78" ry="46" fill="url(#beam6)"/>')
    o.append(_t(250, 104, "Permutation", AM, 17, "middle", POP, "700"))
    o.append(_t(250, 126, "Importance", AM, 17, "middle", POP, "700"))
    o.append(f'<polygon points="430,96 470,84 470,124 430,112" fill="{FRAME}"/>')
    o.append(f'<rect x="468" y="86" width="52" height="36" rx="6" fill="#1c222c" stroke="{FRAME}" stroke-width="2"/>')
    o.append(f'<path d="M430,104 L360,80 L360,132 Z" fill="{AM}" fill-opacity=".14"/>')
    o.append(_t(250, 196, "7차시 — 모델에게 묻는다: 무엇을 보고 판단했니?", MUTE, 14))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 18) 신규 슬라이드 — AUC 를 계산하는 세 가지 방법 (같은 값이 나온다)
#     오늘 모든 표가 AUC 로 쓰여 있으므로 그 정의를 한 장으로 붙인다.
# ─────────────────────────────────────────────────────────────────────────────
def auc_three_ways():
    W, H = 1140, 300
    o = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;display:block;" '
         f'role="img" aria-label="AUC 를 계산하는 세 가지 방법: 일치쌍 비율, ROC 곡선 아래 면적, 순위합">']
    pw, gap = 348, 48
    x0 = (W - (pw * 3 + gap * 2)) / 2
    for i in range(3):
        x = x0 + i * (pw + gap)
        o.append(f'<rect x="{x}" y="0" width="{pw}" height="{H-34}" rx="12" fill="#151a21" stroke="{FRAME}" stroke-width="1.5"/>')

    # ① 일치쌍 비율
    x = x0
    o.append(_t(x + pw / 2, 32, "① 일치쌍 비율", CY, 19, "middle", POP, "700"))
    o.append(_t(x + pw / 2, 52, "Concordance", FAINT, 13))
    for k in range(4):
        yy = 86 + k * 34
        ok = k != 2
        o.append(f'<circle cx="{x+72}" cy="{yy}" r="10" fill="{RD}" fill-opacity=".85"/>')
        o.append(f'<circle cx="{x+150}" cy="{yy}" r="10" fill="{CY}" fill-opacity=".55"/>')
        o.append(f'<line x1="{x+84}" y1="{yy}" x2="{x+138}" y2="{yy}" stroke="{LM if ok else RD}" stroke-width="2" stroke-dasharray="{"0" if ok else "4 4"}"/>')
        o.append(_t(x + 178, yy + 5, "○ 맞음" if ok else "✕ 틀림", LM if ok else RD, 14, "start", LATO, "700"))
    o.append(_t(x + 72, 68, "양성", RD, 12.5))
    o.append(_t(x + 150, 68, "음성", CY, 12.5))
    o.append(_t(x + pw / 2, 240, "양성–음성을 모두 짝지어", MUTE, 14))
    o.append(_t(x + pw / 2, 260, "양성 쪽에 더 높은 점수를 준 비율", "#ffffff", 14.5))

    # ② ROC 곡선 아래 면적
    x = x0 + pw + gap
    o.append(_t(x + pw / 2, 32, "② 곡선 아래 면적", LM, 19, "middle", POP, "700"))
    o.append(_t(x + pw / 2, 52, "ROC · 사다리꼴 적분", FAINT, 13))
    bx, by, bs = x + 112, 200, 132
    o.append(f'<rect x="{bx}" y="{by-bs}" width="{bs}" height="{bs}" fill="none" stroke="{FRAME}" stroke-width="1.5"/>')
    o.append(f'<path d="M{bx},{by} C{bx+18},{by-70} {bx+62},{by-bs+8} {bx+bs},{by-bs} L{bx+bs},{by} Z" fill="{LM}" fill-opacity=".18"/>')
    o.append(f'<path d="M{bx},{by} C{bx+18},{by-70} {bx+62},{by-bs+8} {bx+bs},{by-bs}" fill="none" stroke="{LM}" stroke-width="3"/>')
    o.append(f'<line x1="{bx}" y1="{by}" x2="{bx+bs}" y2="{by-bs}" stroke="{GY}" stroke-width="1.5" stroke-dasharray="5 5"/>')
    o.append(_t(bx + bs / 2 + 14, by - 40, "면적 = AUC", LM, 14.5, "middle", LATO, "700"))
    o.append(_t(bx + bs / 2, by + 18, "FPR", FAINT, 12.5))
    o.append(_t(bx - 12, by - bs / 2, "TPR", FAINT, 12.5, "middle", LATO, None,
                f' transform="rotate(-90 {bx-12} {by-bs/2})"'))
    o.append(_t(x + pw / 2, 240, "임계값을 옮기며 (FPR, TPR) 을 찍고", MUTE, 14))
    o.append(_t(x + pw / 2, 260, "그 곡선 아래 넓이를 잰다", "#ffffff", 14.5))

    # ③ 순위합
    x = x0 + 2 * (pw + gap)
    o.append(_t(x + pw / 2, 32, "③ 순위합", AM, 19, "middle", POP, "700"))
    o.append(_t(x + pw / 2, 52, "Mann–Whitney U", FAINT, 13))
    order = [RD, CY, RD, RD, CY, CY]
    for k, c in enumerate(order):
        yy = 80 + k * 26
        o.append(_t(x + 58, yy + 5, f"{k+1}위", FAINT, 12.5, "end"))
        o.append(f'<rect x="{x+70}" y="{yy-9}" width="{160 - k*12}" height="18" rx="5" fill="{c}" fill-opacity=".7"/>')
    o.append(_t(x + 262, 100, "양성이", RD, 14, "start", LATO, "700"))
    o.append(_t(x + 262, 120, "위쪽에", RD, 14, "start", LATO, "700"))
    o.append(_t(x + 262, 140, "몰릴수록", MUTE, 14, "start"))
    o.append(_t(x + 262, 160, "AUC ↑", AM, 15, "start", POP, "700"))
    o.append(_t(x + pw / 2, 240, "예측 점수를 순위로 바꾼 뒤", MUTE, 14))
    o.append(_t(x + pw / 2, 260, "양성의 순위합으로 계산한다", "#ffffff", 14.5))

    o.append(_t(W / 2, H - 8, "세 방법은 같은 값을 준다  ·  1.0 = 완벽,  0.5 = 동전 던지기  ·  오늘의 모든 표가 이 척도로 쓰여 있다",
                MUTE, 14.5))
    o.append('</svg>')
    return "\n".join(o)


# ─────────────────────────────────────────────────────────────────────────────
# 레지스트리 · 치환 · 검증
# ─────────────────────────────────────────────────────────────────────────────
FIGURES = {
    "title_motif":        title_motif,
    "agenda_rail":        agenda_rail,
    "linearity_stairs":   linearity_stairs,
    "quintile_panels":    quintile_panels,
    "tree_concept":       tree_concept,
    "interaction_tree":   interaction_tree,
    "same_symptom":       same_symptom,
    "auc_three_ways":     auc_three_ways,
    "depth_curve":        lambda: depth_curve(False),
    "forest_pipeline":    forest_pipeline,
    "depth_curve_forest": lambda: depth_curve(True),
    "model_bars":         model_bars,
    "fold_pairs":         fold_pairs,
    "balance_scale":      balance_scale,
    "three_pillars":      three_pillars,
    "bump_chart":         bump_chart,
    "sealed_test":        sealed_test,
    "candidate_bias":     candidate_bias,
    "black_box":          black_box,
}

BLOCK = re.compile(r'<svg data-fig="([A-Za-z0-9_]+)".*?</svg>', re.S)


def render(name):
    """그림 하나를 생성하고 data-fig 표식을 심는다. (불변성 2 확인 지점)"""
    svg = FIGURES[name]()
    n_open, n_close = svg.count("<svg"), svg.count("</svg>")
    assert n_open == 1 and n_close == 1, f"{name}: <svg> 가 {n_open}/{n_close} 개 — 중첩 금지"
    return svg.replace("<svg ", f'<svg data-fig="{name}" ', 1)


def _check_registry(html):
    """불변성 1 — HTML 의 NAME 집합과 FIGURES 키가 1:1 이고 각 1회만 등장하는가."""
    found = BLOCK.findall(html)
    dup = sorted({n for n in found if found.count(n) > 1})
    missing = sorted(set(FIGURES) - set(found))       # 코드에는 있는데 HTML 에 없다
    unknown = sorted(set(found) - set(FIGURES))       # HTML 에는 있는데 코드에 없다
    problems = []
    if dup:
        problems.append(f"HTML 에 중복된 data-fig: {dup} (첫 블록만 갱신돼 나머지가 낡은 채 남는다)")
    if missing:
        problems.append(f"HTML 에 자리가 없는 그림: {missing}")
    if unknown:
        problems.append(f"코드에 없는 data-fig: {unknown}")
    return found, problems


def verify_numbers():
    """불변성 3 — 차트에 그린 모든 수치가 원본 표/독스트링에 실재하는가."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(HTML, encoding="utf-8").read()
    src += "\n" + open(os.path.join(here, "_build_s6.py"), encoding="utf-8").read()

    def present(v):
        return any(t in src for t in (f"{v:.4f}", f"{v:.4f}".lstrip("0"),
                                      f"{v:.3f}", f"{v:.3f}".lstrip("0")))

    fails, n = [], 0
    def check(label, vals):
        nonlocal n
        for v in vals:
            n += 1
            if not present(v):
                fails.append(f"{label}: {v}")

    for name, vals, *_ in QUINTILES:
        check(f"5분위 {name}", vals)
    check("기저율(Dummy AP)", [0.3371])
    check("트리 train", TREE_TRAIN)
    check("트리 CV", TREE_CV)
    check("포레스트 CV", FOREST_CV)
    check("4모델 CV", [m[2] for m in MODELS_A])
    check("4모델 AP", [m[4] for m in MODELS_A])
    check("4모델 recall", [m[5] for m in MODELS_A])
    check("4모델 train", [m[6] for m in MODELS_A])
    check("폴드 로지스틱", FOLD_LOG)
    check("폴드 포레스트", FOLD_FOR)
    check("폴드 차이", FOLD_DELTA)
    check("Model A/B", [x for _, a, b, _ in AB for x in (a, b)])
    for v in LEAVES:
        n += 1
        if str(v) not in src:
            fails.append(f"리프수: {v}")
    for name, c, _ in CANDIDATES:
        n += 1
        if str(c) not in src:
            fails.append(f"후보수 {name}: {c}")
    return n, fails


def apply(path=HTML, dry_run=False):
    html = open(path, encoding="utf-8").read()
    found, problems = _check_registry(html)
    if problems:
        for p in problems:
            print("  ❌", p)
        return 1

    replaced = []
    def swap(m):
        replaced.append(m.group(1))
        return render(m.group(1))
    out = BLOCK.sub(swap, html)

    # 불변성 2 사후 확인 — 치환으로 태그 균형이 깨지지 않았는가
    if out.count("<svg") != out.count("</svg>"):
        print(f"  ❌ 태그 불균형: <svg> {out.count('<svg')} / </svg> {out.count('</svg>')}")
        return 1

    # write 직후 cardinality 로그 — 등록 수 vs 실제 치환 수 (둘 다 찍는다)
    changed = out != html
    print(f"  그림 등록 {len(FIGURES)}개 · HTML 자리 {len(found)}개 · 치환 {len(replaced)}개 "
          f"· 내용 변경 {'있음' if changed else '없음'}")
    if len(replaced) != len(FIGURES):
        print("  ❌ 등록 수와 치환 수가 다르다")
        return 1

    if dry_run:
        return 0
    if changed:
        open(path, "w", encoding="utf-8").write(out)
        print(f"  ✅ {os.path.relpath(path)} 갱신")
    else:
        print(f"  ✅ 변경 없음 (이미 최신)")
    return 0


def main(argv):
    if "--dump" in argv:
        name = argv[argv.index("--dump") + 1]
        print(render(name))
        return 0
    n, fails = verify_numbers()
    print(f"  수치 대조 {n}개", end=" · ")
    if fails:
        print("❌ 원본에서 확인 안 되는 값:")
        for f in fails:
            print("     -", f)
        return 1
    print("전부 원본 표/독스트링에 존재")
    return apply(dry_run="--check" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

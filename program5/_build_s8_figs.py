# -*- coding: utf-8 -*-
"""session8.html 의 인라인 SVG 그림을 생성 · 갱신한다.

사용법
    python3 _build_s8_figs.py              # session8.html 안의 그림을 제자리 갱신
    python3 _build_s8_figs.py --check      # 수치 대조 + 구조 검사만 (파일 안 건드림)
    python3 _build_s8_figs.py --dump NAME  # 그림 하나를 표준출력으로

동작 방식
    session8.html 의 각 그림은 <svg data-fig="NAME" ...> ... </svg> 로 표시돼 있다.
    이 스크립트는 그 블록을 통째로 새로 생성한 SVG 로 바꾼다. 슬라이드의 글·배치는
    HTML 에서 직접 고치고, 그림은 여기서 고친 뒤 이 스크립트를 다시 돌린다.
    (_build_s6_figs.py · _build_s7_figs.py 와 같은 방식이다.)

★ 불변성 (수정 전 반드시 확인)
    1. data-fig NAME 은 HTML 안에서 유일하고 FIGURES 키와 1:1 대응한다.
       깨지면 → 그림이 조용히 갱신 안 되거나 엉뚱한 자리에 들어간다.
       확인 → _check_registry(): 대칭차집합 == 공집합 + 각 NAME 등장 1회 검사.
    2. 각 SVG 블록에 중첩 <svg> 가 없다 (여는 <svg 부터 첫 </svg> 까지가 한 블록).
       깨지면 → 정규식이 블록을 잘라 먹어 HTML 이 망가진다.
       확인 → render(): 생성물마다 count("<svg") == 1 검사 + 치환 후 개수 일치 검사.
    3. 이 그림들은 **새 수치를 만들지 않는다.** 전부 8차시 실측표(강사 노트 '실측 대조표')
       또는 3·4·6차시가 이미 확정해 둔 값이다.
       깨지면 → 슬라이드와 노트가 서로 다른 숫자를 말한다.
       확인 → verify_numbers(): SOURCED 전부를 session8/lecture_notes.md 에서 문자열 대조.

수치 출처
    .6535 / .6651 / .6355               → reports/model_metrics_cv.csv (6차시 CV)
    .6718 / .6566 / .6375 / −.0085      → 8차시 test 실측 (session8.ipynb Step 1~2)
    .6041 .7398 / .5865 .7261 / .136 .140 → test AUC 부트스트랩 95% 구간 (B=2000)
    +.0147 [−.0157, +.0443] / 83.9%     → 두 모델 차이의 부트스트랩 분포
    34.3% / 24.9% / 18.9%               → 민감도 분석 test 양성률
    142명 / 1,321명                      → 3·4차시 실측 (cutoff 1.50 동점 덩어리)
    후보 12개 (포레스트 4 · 트리 4 · 로지스틱 3 · Dummy 1) → configs/modeling.yaml · 6차시 슬라이드 21
    FN/TP/TN 프로파일 12개 값             → 8차시 test 오류분석 (n = 32 / 59 / 107)
    75.7 / 44.5 / 50.0 / 59.3           → 경계선(≤1.7) 비율 train(7차시) vs test(오늘)

팔레트 (1~7차시 덱과 동일)
    강조/로지스틱 #00f2ff · 합의/양호 #adff2f · 위험/FN #ff8080 · 보조 #8892a4 · 주의 #ffd479
"""
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(_HERE, "session8", "session8.html")
NOTES = os.path.join(_HERE, "session8", "lecture_notes.md")
TARGETS = [HTML]

CY, LM, RD, GY, AM = "#00f2ff", "#adff2f", "#ff8080", "#8892a4", "#ffd479"
GRID, FRAME, INK, MUTE, FAINT = "#222c38", "#3a4250", "#e8ecf2", "#8892a4", "#55606f"
PANEL = "#101820"
LATO = "Lato, 'Nanum Gothic', sans-serif"
POP = "Poppins, sans-serif"


def _t(x, y, s, fill=INK, size=15, anchor="middle", fam=LATO, weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-family="{fam}"{w}{extra}>{s}</text>')


def _r(x, y, w, h, fill, rx=3, extra=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" '
            f'fill="{fill}" rx="{rx}"{extra}/>')


def _line(x1, y1, x2, y2, stroke=FRAME, w=1, extra=""):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{extra}/>')


def _dot(cx, cy, r, fill, extra=""):
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{extra}/>'


def _legend(items, x, y, gap=150, size=14):
    """items = [(라벨, 색)] — 스와치 + 라벨."""
    out = []
    for i, (lab, col) in enumerate(items):
        cx = x + i * gap
        out.append(_r(cx, y - 10, 12, 12, col, rx=2))
        out.append(_t(cx + 18, y, lab, MUTE, size, anchor="start"))
    return "".join(out)


def _arrow_defs(ids):
    out = ["<defs>"]
    for i, c in ids:
        out.append(f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
                   f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    out.append("</defs>")
    return "".join(out)


def _open(name, w, h, maxh=None):
    mh = f'style="max-height:{maxh or h}px" '
    return (f'<svg data-fig="{name}" viewBox="0 0 {w} {h}" width="100%" '
            f'{mh}xmlns="http://www.w3.org/2000/svg">')


# ─── 슬라이드 03 · 오늘의 정서 곡선 ───────────────────────────────────────────
# (수치 없음 — 오늘 수업의 정서 설계도. 강사 노트 머리말의 곡선과 같은 순서다.)
ARC = [("긴장", "봉투를 연다", "Step 1", AM, 104),
       ("당황", "순위가 뒤집혔다", "Step 2", RD, 68),
       ("안도", "우리는 이미 적어 뒀다", "Step 2", LM, 152),
       ("겸허", "그 뒤집힘조차 못 믿는다", "Step 3", CY, 116),
       ("무거움", "우리 변명이 무너졌다", "Step 5", RD, 74),
       ("착지", "재현성으로 끝낸다", "Step 6~7", LM, 158)]


def fig_arc():
    W, H = 1080, 250
    xs = [100 + i * 176 for i in range(len(ARC))]
    o = [_open("arc", W, H, 250)]
    # 곡선
    d = [f"M{xs[0]},{ARC[0][4]}"]
    for i in range(1, len(ARC)):
        x0, y0 = xs[i - 1], ARC[i - 1][4]
        x1, y1 = xs[i], ARC[i][4]
        d.append(f"C{x0+88},{y0} {x1-88},{y1} {x1},{y1}")
    o.append(f'<path d="{" ".join(d)}" fill="none" stroke="{FRAME}" stroke-width="3"/>')
    for i, (emo, trig, step, col, y) in enumerate(ARC):
        x = xs[i]
        o.append(_line(x, y + 12, x, 186, GRID, 1, ' stroke-dasharray="3 4"'))
        o.append(_dot(x, y, 8, col, f' stroke="{PANEL}" stroke-width="2"'))
        o.append(_t(x, y - 18, emo, col, 20, fam=POP, weight="700"))
        o.append(_t(x, 206, trig, INK, 15))
        o.append(_t(x, 230, step, FAINT, 13))
    o.append(_line(60, 186, W - 60, 186, FRAME, 1))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 05 · 4주 봉인 타임라인 ─────────────────────────────────────────
SEAL = [("4차시", "split · 라벨 · 봉인", "test 0건"),
        ("5차시", "계수 · 부트스트랩", "test 0건"),
        ("6차시", "트리 · 포레스트 · CV", "test 0건"),
        ("7차시", "중요도 · 오류 분석", "test 0건"),
        ("8차시 (오늘)", "봉투를 연다", "개봉")]


def _lock(cx, cy, col, open_=False):
    body = _r(cx - 11, cy - 1, 22, 16, col, rx=3)
    if open_:
        sh = (f'<path d="M{cx+1},{cy-1} v-7 a7,7 0 0 1 14,0" fill="none" '
              f'stroke="{col}" stroke-width="2.5"/>')
    else:
        sh = (f'<path d="M{cx-7},{cy-1} v-7 a7,7 0 0 1 14,0 v7" fill="none" '
              f'stroke="{col}" stroke-width="2.5"/>')
    return body + sh


def fig_seal():
    W, H = 1080, 180
    bw, gap, x0, by, bh = 190, 22, 21, 46, 88
    o = [_open("seal", W, H, 180)]
    # 봉인 밴드 (4~7차시 위)
    band_w = 4 * bw + 3 * gap
    o.append(_r(x0, 8, band_w, 28, PANEL, rx=8,
                extra=f' stroke="{AM}" stroke-width="1.6" stroke-dasharray="7 5"'))
    o.append(_t(x0 + band_w / 2, 28, "🔒 test 265명 — 4주 동안 한 번도 열지 않았다", AM, 16, weight="700"))
    for i, (ttl, sub, badge) in enumerate(SEAL):
        x = x0 + i * (bw + gap)
        last = (i == len(SEAL) - 1)
        col = LM if last else GY
        o.append(_r(x, by, bw, bh, PANEL, rx=10,
                    extra=f' stroke="{col}" stroke-width="{2.4 if last else 1.2}"'))
        o.append(_t(x + bw / 2, by + 24, ttl, LM if last else INK, 17, fam=POP, weight="700"))
        o.append(_t(x + bw / 2, by + 45, sub, MUTE, 14))
        o.append(_lock(x + 26, by + 62, col, open_=last))
        o.append(_t(x + 44, by + 76, badge, col, 14, anchor="start", weight="700"))
    o.append(_t(W / 2, H - 24, "감사 결과 — 4~7차시 노트북 전체를 스캔해도 test 로 성능을 잰 흔적 0 건",
                LM, 17, weight="700"))
    o.append(_t(W / 2, H - 4, "그동안 우리가 본 것은 train 1,056명뿐이다", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 09 · CV → test 순위 뒤집힘 ─────────────────────────────────────
FLIP = [("로지스틱", .6535, .6718, CY), ("랜덤 포레스트", .6651, .6566, RD),
        ("결정 트리", .6355, .6375, GY)]
LO, HI, TOP, BASE = .62, .69, 60, 250


def _fy(v):
    return BASE - (v - LO) / (HI - LO) * (BASE - TOP)


def fig_rankflip():
    W, H = 620, 300
    lx, rx = 200, 440
    o = [_open("rankflip", W, H, 356)]
    for gv in (.62, .64, .66, .68):
        y = _fy(gv)
        o.append(_line(90, y, 600, y, GRID, 1))
        o.append(_t(84, y + 4, f"{gv:.2f}".lstrip("0"), FAINT, 12, anchor="end"))
    o.append(_line(lx, 56, lx, BASE + 8, FRAME, 1))
    o.append(_line(rx, 56, rx, BASE + 8, FRAME, 1))
    o.append(_t(lx, 44, "CV · 6차시 (1,056명)", MUTE, 14))
    o.append(_t(rx, 44, "test · 오늘 (265명)", MUTE, 14))
    for name, cv, te, col in FLIP:
        y1, y2 = _fy(cv), _fy(te)
        wid = 3 if col is not GY else 2
        o.append(_line(lx, y1, rx, y2, col, wid))
        o.append(_dot(lx, y1, 6, col))
        o.append(_dot(rx, y2, 6, col))
        o.append(_t(lx - 12, y1 + 5, f"{cv:.4f}".lstrip("0"), col, 14, anchor="end"))
        o.append(_t(rx + 14, y2 + 5, f"{name} {te:.4f}".replace(" 0.", " ."),
                    col, 15, anchor="start", weight="700"))
    # 교차점 강조 — 로지스틱과 포레스트 선이 만나는 지점을 직접 계산한다
    (l0, l1) = (_fy(.6535), _fy(.6718))
    (f0, f1) = (_fy(.6651), _fy(.6566))
    t = (f0 - l0) / ((l1 - l0) - (f1 - f0))
    cx_, cy_ = lx + t * (rx - lx), l0 + t * (l1 - l0)
    o.append(_dot(cx_, cy_, 19, RD, ' opacity="0.18"'))
    o.append(_dot(cx_, cy_, 19, "none", f' stroke="{RD}" stroke-width="2.5"'))
    o.append(_t(cx_, cy_ - 28, "여기서 뒤집힌다", RD, 14, weight="700"))
    o.append(_t(W / 2, H - 28, "CV 1위 포레스트 → test 1위 로지스틱", RD, 17, weight="700"))
    o.append(_t(W / 2, H - 8, "포레스트만 −.0085 · 로지스틱 +.0183 · 트리 +.0020", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 10 · 다중 비교 (승자에 섞인 운) ────────────────────────────────
# 막대 12개 = 세트당 후보 12개. 각 막대의 '운' 값은 개념 설명용 고정 상수다
# (난수 아님 — 빌드마다 같은 그림이 나와야 한다). 최고값 +22 가 CV 1등이 된다.
LUCK = [4, -6, 9, -3, 14, 2, -8, 6, 22, -2, 11, -5]


def fig_winner():
    W, H = 620, 282
    base, skill = 206, 110
    o = [_open("winner", W, H, 330)]
    o.append(_arrow_defs([("w_rd", RD), ("w_am", AM)]))
    o.append(_t(30, 34, "CV — 후보 12개를 모두 채점하고, 그중 1등을 고른다", INK, 15, anchor="start"))
    top_i = LUCK.index(max(LUCK))
    for i, lk in enumerate(LUCK):
        x = 50 + i * 21
        h = skill + lk
        o.append(_r(x, base - h, 13, h, GY, rx=2, extra=' opacity="0.55"'))
        if lk > 0:
            o.append(_r(x, base - h, 13, lk, AM, rx=2))
        if i == top_i:
            o.append(_r(x - 4, base - h - 7, 21, h + 12, "none", rx=5,
                        extra=f' stroke="{AM}" stroke-width="2"'))
            o.append(_t(x + 6, base - h - 14, "CV 1등", AM, 14, weight="700"))
    o.append(_line(44, base - skill, 300, base - skill, GY, 1.4, ' stroke-dasharray="5 4"'))
    o.append(_line(30, 54, 54, 54, GY, 1.4, ' stroke-dasharray="5 4"'))
    o.append(_t(60, 58, "점선 = 진짜 실력 수준", GY, 13, anchor="start"))
    o.append(_line(44, base, 300, base, FRAME, 1))
    o.append(_t(172, base + 22, "후보 12개", MUTE, 14))
    # 오른쪽: 뽑힌 1등이 test 에서 내려앉는다
    o.append(f'<path d="M312,{base-70} L364,{base-70}" fill="none" stroke="{AM}" '
             f'stroke-width="2" marker-end="url(#w_am)"/>')
    o.append(_t(338, base - 78, "1개만 고른다", AM, 13))
    for k, (lab, lk, col) in enumerate((("CV 점수", 22, AM), ("test 점수", 0, GY))):
        x = 400 + k * 96
        h = skill + lk
        o.append(_r(x, base - h, 58, h, GY, rx=3, extra=' opacity="0.55"'))
        if lk:
            o.append(_r(x, base - h, 58, lk, col, rx=3))
        if lk:
            o.append(_t(x + 29, base - h - 10, "실력+운", col, 14, weight="700"))
        o.append(_t(x + 29, base + 22, lab, INK, 15))
    o.append(f'<path d="M468,{base-skill-24} L494,{base-skill-6}" fill="none" stroke="{RD}" '
             f'stroke-width="2.4" marker-end="url(#w_rd)"/>')
    o.append(_t(525, base - skill - 34, "운은 따라오지", RD, 14, weight="700"))
    o.append(_t(525, base - skill - 16, "않는다", RD, 14, weight="700"))
    o.append(_t(W / 2, H - 26, "후보가 많을수록 '운 좋은 한 판'이 1등이 될 확률이 커진다", AM, 16, weight="700"))
    o.append(_t(W / 2, H - 6, "세트당 후보 12개 — 포레스트 4 · 트리 4 · 로지스틱 3 · Dummy 1", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 12 · 신뢰구간 (두 구간 겹침 + 차이 구간이 0 포함) ──────────────
CIS = [("로지스틱", .6718, .6041, .7398, CY, ".136"),
       ("포레스트", .6566, .5865, .7261, RD, ".140")]
A_LO, A_HI, AX0, AX1 = .56, .76, 168, 596
D_LO, D_HI = -.05, .07


def _ax(v):
    return AX0 + (v - A_LO) / (A_HI - A_LO) * (AX1 - AX0)


def _dx(v):
    return AX0 + (v - D_LO) / (D_HI - D_LO) * (AX1 - AX0)


def fig_ci():
    W, H = 640, 300
    o = [_open("ci", W, H, 340)]
    o.append(_t(AX0, 30, "① test AUC 의 95% 부트스트랩 구간 (B = 2,000)", INK, 15, anchor="start"))
    for gv in (.56, .60, .64, .68, .72, .76):
        x = _ax(gv)
        o.append(_line(x, 46, x, 148, GRID, 1))
        o.append(_t(x, 164, f"{gv:.2f}".lstrip("0"), FAINT, 12))
    for i, (name, pt, lo, hi, col, wide) in enumerate(CIS):
        y = 70 + i * 46
        o.append(_line(_ax(lo), y, _ax(hi), y, col, 3))
        for e in (lo, hi):
            o.append(_line(_ax(e), y - 8, _ax(e), y + 8, col, 3))
        o.append(_dot(_ax(pt), y, 7, col, f' stroke="{PANEL}" stroke-width="2"'))
        o.append(_t(AX0 - 12, y + 5, name, col, 16, anchor="end", weight="700"))
        o.append(_t(_ax(pt), y - 16, f"{pt:.4f}".lstrip("0"), col, 14, weight="700"))
        o.append(_t(_ax(hi) + 10, y + 5, f"폭 {wide}", MUTE, 13, anchor="start"))
    o.append(_t((AX0 + AX1) / 2, 186, "두 구간이 거의 통째로 겹친다", AM, 15, weight="700"))
    # ② 차이
    o.append(_t(AX0, 216, "② 두 모델 차이(로지스틱 − 포레스트)의 95% 구간", INK, 15, anchor="start"))
    y = 250
    z = _dx(0)
    o.append(_line(z, 228, z, 274, RD, 2, ' stroke-dasharray="5 4"'))
    o.append(_t(z + 7, 234, "0", RD, 14, anchor="start", weight="700"))
    o.append(_line(_dx(-.0157), y, _dx(.0443), y, LM, 3))
    for e in (-.0157, .0443):
        o.append(_line(_dx(e), y - 8, _dx(e), y + 8, LM, 3))
    o.append(_dot(_dx(.0147), y, 7, LM, f' stroke="{PANEL}" stroke-width="2"'))
    o.append(_t(_dx(.0147), y - 16, "+.0147", LM, 14, weight="700"))
    o.append(_t(_dx(-.0157) - 8, y + 5, "−.0157", MUTE, 13, anchor="end"))
    o.append(_t(_dx(.0443) + 8, y + 5, "+.0443", MUTE, 13, anchor="start"))
    o.append(_t(AX0 - 12, y + 5, "차이", LM, 16, anchor="end", weight="700"))
    o.append(_t((AX0 + AX1) / 2, 290, "구간이 0 을 가로지른다 — 부트스트랩 2,000회 중 로지스틱이 이긴 비율 83.9%",
                MUTE, 13.5))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 13 · 저해상도 판독 카메라 ──────────────────────────────────────
def fig_photofinish():
    W, H = 520, 300
    o = [_open("photofinish", W, H, 300)]
    o.append(_t(140, 40, "실제 (알 수 없는 진실)", INK, 15))
    o.append(_t(392, 40, "우리 카메라가 본 것", INK, 15))
    # 왼쪽 — 아주 좁은 격차
    o.append(_line(46, 132, 236, 132, FRAME, 1.5))
    for x, col in ((152, CY), (138, RD)):
        o.append(f'<path d="M{x},{124} l8,-15 l-16,0 z" fill="{col}"/>')
        o.append(_line(x, 124, x, 140, col, 2.4))
    o.append(_line(138, 100, 152, 100, AM, 2))
    o.append(_line(138, 96, 138, 104, AM, 2))
    o.append(_line(152, 96, 152, 104, AM, 2))
    o.append(_t(145, 88, ".0147", AM, 15, weight="700"))
    o.append(_line(152, 142, 190, 160, CY, 1))
    o.append(_t(194, 166, "로지스틱 .6718", CY, 13, anchor="start"))
    o.append(_line(138, 142, 112, 184, RD, 1))
    o.append(_t(46, 192, "포레스트 .6566", RD, 13, anchor="start"))
    # 오른쪽 — 화소 3칸, 둘 다 같은 칸
    for i in range(3):
        x = 300 + i * 64
        hot = (i == 1)
        o.append(_r(x, 100, 62, 62, "#2b3442" if hot else PANEL, rx=2,
                    extra=f' stroke="{AM if hot else FRAME}" stroke-width="{2 if hot else 1}"'))
    o.append(_t(395, 136, "?", AM, 30, fam=POP, weight="700"))
    o.append(_t(395, 180, "한 화소 = .136", AM, 14, weight="700"))
    o.append(_t(395, 200, "둘 다 이 안에 들어간다", MUTE, 13))
    o.append(_t(W / 2, 236, "격차 .0147 은 화소 한 칸의 약 1/9 — 확대해도 픽셀만 깨진다", MUTE, 14))
    o.append(_r(120, 252, 280, 36, PANEL, rx=8, extra=f' stroke="{LM}" stroke-width="1.6"'))
    o.append(_t(260, 276, "가장 정확한 판정 = 판독 불가", LM, 17, weight="700"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 15 · 이산 점수 계단 ────────────────────────────────────────────
STEPS = ["1.3", "1.4", "1.5", "1.6", "1.7"]
CUTS = [(260, LM, "≥ 1.5 → 양성 34.3%", "q .70 = q .75 · 우리 선택"),
        (360, AM, "> 1.5 = ≥ 1.6 → 24.9%", "부등호 하나 = 분위수 한 단계"),
        (460, RD, "> 1.6 → 18.9%", "명단이 거의 절반으로 준다")]


def fig_stair():
    W, H = 640, 310
    o = [_open("stair", W, H, 276)]
    for i, (x, col, lab, sub) in enumerate(CUTS):
        lx = 26 + i * 200
        o.append(_r(lx, 22, 11, 11, col, rx=2))
        o.append(_t(lx + 17, 32, lab, col, 13, anchor="start", weight="700"))
        o.append(_t(lx + 17, 48, sub, FAINT, 11.5, anchor="start"))
    d = ["M60,250"]
    for i in range(5):
        d.append(f"H{160 + i * 100}")
        if i < 4:
            d.append(f"V{216 - i * 34}")
    o.append(f'<path d="{" ".join(d)}" fill="none" stroke="{FRAME}" stroke-width="3"/>')
    for i, sc in enumerate(STEPS):
        o.append(_t(110 + i * 100, 250 - i * 34 + 20, sc, MUTE, 14))
    for r in range(2):
        for c in range(7):
            o.append(_dot(272 + c * 13, 168 - r * 14, 4.4, AM))
    o.append(_t(310, 138, "1.50 에만 142명이 함께 서 있다", AM, 16, weight="700"))
    o.append(_t(310, 121, "(3차시 실측 · 전체 1,321명 기준)", FAINT, 12))
    for x, col, lab, sub in CUTS:
        o.append(_line(x, 152, x, 276, col, 2, ' stroke-dasharray="6 5"'))
    o.append(_line(60, 276, 580, 276, FRAME, 1))
    o.append(_t(60, 296, "문화적응 스트레스 점수 — 0.1 단위로 뚝뚝 끊긴다", MUTE, 14, anchor="start"))
    o.append(_t(580, 296, "선은 계단 사이에만 그을 수 있다", LM, 14, anchor="end", weight="700"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 18 · test 오류 프로파일 ────────────────────────────────────────
PROF = [("자아존중감", 3.602, 2.839, 3.460), ("친구지지", 4.455, 3.552, 4.545),
        ("부모 감독", 3.677, 2.949, 3.542), ("우울", 1.547, 2.071, 1.409)]


def fig_fnprof():
    W, H = 620, 272
    base, top, gw, bw = 200, 48, 148, 30
    o = [_open("fnprof", W, H, 286)]
    for i, (lab, col) in enumerate((("FN 놓친 32명", AM), ("TP 찾은 59명", RD),
                                    ("TN 맞게 제외 107명", GY))):
        o.append(_r(120 + i * 168, 8, 12, 12, col, rx=2))
        o.append(_t(138 + i * 168, 19, lab, MUTE, 14, anchor="start"))
    for gv in (1, 2, 3, 4, 5):
        y = base - (gv / 5) * (base - top)
        o.append(_line(28, y, W - 12, y, GRID, 1))
        o.append(_t(20, y + 4, str(gv), FAINT, 11, anchor="end"))
    for gi, (name, fn, tp, tn) in enumerate(PROF):
        gx = 44 + gi * gw
        for bi, (v, col) in enumerate(((fn, AM), (tp, RD), (tn, GY))):
            h = (v / 5) * (base - top)
            x = gx + bi * (bw + 6)
            o.append(_r(x, base - h, bw, h, col, rx=2, extra="" if bi != 1 else ' opacity="0.9"'))
            o.append(_t(x + bw / 2, base - h - 6, f"{v:.3f}", col, 12.5, weight="700"))
        o.append(_t(gx + (3 * bw + 12) / 2 - 3, base + 20, name, INK, 15, weight="700"))
    o.append(_line(28, base, W - 12, base, FRAME, 1))
    o.append(_t(W / 2, H - 26, "test 에서도 FN 은 TP 가 아니라 TN 을 닮았다", LM, 17, weight="700"))
    o.append(_t(W / 2, H - 6, "자아존중감은 스트레스 없는 학생(TN 3.460)보다도 높다", AM, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 19 · 경계선 비율 train vs test ─────────────────────────────────
BORDER = [("train · 7차시 (1,056명)", 110, [("FN", 75.7, AM), ("TP", 44.5, GY)]),
          ("test · 오늘 (265명)", 390, [("FN", 50.0, RD), ("TP", 59.3, GY)])]


def fig_borderline():
    W, H = 620, 284
    base, top, bw = 212, 62, 52
    o = [_open("borderline", W, H, 284)]
    o.append(_arrow_defs([("b_rd", RD)]))
    for gv in (20, 40, 60, 80):
        y = base - gv / 80 * (base - top)
        o.append(_line(40, y, W - 20, y, GRID, 1))
        o.append(_t(34, y + 4, f"{gv}%", FAINT, 11, anchor="end"))
    for title, gx, bars in BORDER:
        for bi, (lab, v, col) in enumerate(bars):
            h = v / 80 * (base - top)
            x = gx + bi * (bw + 16)
            o.append(_r(x, base - h, bw, h, col, rx=3))
            o.append(_t(x + bw / 2, base - h - 8, f"{v}%", col, 15, weight="700"))
            o.append(_t(x + bw / 2, base + 20, lab, INK, 14, weight="700"))
        o.append(_t(gx + bw + 8, base + 42, title, MUTE, 14))
    o.append(_line(40, base, W - 20, base, FRAME, 1))
    o.append(_line(310, 52, 310, 236, FRAME, 1, ' stroke-dasharray="5 5"'))
    o.append(_t(176, 44, "FN 이 훨씬 높다 → \"경계선이라 놓친 것\"", AM, 14, weight="700"))
    o.append(_t(452, 44, "역전 — FN 이 오히려 낮다", RD, 14, weight="700"))
    o.append(_r(258, 126, 104, 32, PANEL, rx=16, extra=f' stroke="{RD}" stroke-width="1.6"'))
    o.append(_t(310, 147, "방향이 반대", RD, 15, weight="700"))
    o.append(_t(W / 2, H - 8, "경계선 = 문화적응 스트레스 점수 ≤ 1.7 (cutoff 1.5 바로 위)", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 01 · 봉인된 봉투 ────────────────────────────────────────────────
def fig_envelope():
    W, H = 520, 252
    ex, ey, ew, eh = 30, 18, 460, 216
    o = [_open("envelope", W, H, 252)]
    o.append(_r(ex + 9, ey + 11, ew, eh, "#05080c", rx=10))
    o.append(_r(ex, ey, ew, eh, "#1d2632", rx=10,
                extra=f' stroke="{GY}" stroke-width="2.4"'))
    # 위쪽 플랩
    o.append(f'<path d="M{ex+2},{ey+4} L{ex+ew/2},{ey+96} L{ex+ew-2},{ey+4}" fill="#222c38" '
             f'stroke="{GY}" stroke-width="2.4" stroke-linejoin="round"/>')
    # 끈 고정쇠
    for cy in (ey + 78, ey + 128):
        o.append(_dot(ex + ew / 2, cy, 11, "#2b3442", f' stroke="{GY}" stroke-width="1.8"'))
        o.append(_dot(ex + ew / 2, cy, 2.8, GY))
    o.append(f'<path d="M{ex+ew/2},{ey+78} C{ex+ew/2-20},{ey+100} {ex+ew/2+20},{ey+108} '
             f'{ex+ew/2},{ey+128}" fill="none" stroke="{GY}" stroke-width="1.6"/>')
    # 붉은 봉인 도장
    sx, sy, sw, sh = ex + 58, ey + 122, 344, 78
    o.append(f'<g transform="rotate(-6 {sx+sw/2} {sy+sh/2})">')
    o.append(_r(sx, sy, sw, sh, "none", rx=7,
                extra=f' stroke="{RD}" stroke-width="3.4" stroke-dasharray="10 6" opacity="0.92"'))
    o.append(_t(sx + sw / 2, sy + 33, "265 TEST SUBJECTS", RD, 25, fam=POP, weight="700"))
    o.append(_t(sx + sw / 2, sy + 62, "SEALED FOR 4 WEEKS", RD, 22, fam=POP, weight="700"))
    o.append("</g>")
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 04 · 7 Step 레일 (두 개의 봉우리) ──────────────────────────────
RAIL = [("0", "봉인 감사", "정말 안 봤는지 코드로", GY, False),
        ("1~2", "봉투를 연다", "순위가 뒤집힌다", LM, True),
        ("3", "신뢰구간", "판독 불가", GY, False),
        ("4", "민감도", "명단이 바뀐다", GY, False),
        ("5", "재현 확인", "변명이 무너진다", RD, True),
        ("6~7", "보고서 · 재현성", "계산은 코드, 판단은 사람", GY, False)]


def fig_steprail():
    W, H = 1080, 234
    y = 104
    xs = [110 + i * 172 for i in range(len(RAIL))]
    o = [_open("steprail", W, H, 234)]
    o.append(_line(xs[0], y, xs[-1], y, FRAME, 5))
    for i, (num, title, sub, col, peak) in enumerate(RAIL):
        x = xs[i]
        r = 34 if peak else 26
        if peak:
            o.append(_dot(x, y, r + 9, col, ' opacity="0.14"'))
        o.append(_dot(x, y, r, PANEL, f' stroke="{col}" stroke-width="{4 if peak else 2}"'))
        o.append(_t(x, y + 8, num, col if peak else INK, 22 if peak else 19, fam=POP, weight="700"))
        if peak:
            o.append(_t(x, y - r - 16, "⛰ 봉우리", col, 14, weight="700"))
        o.append(_t(x, y + r + 30, title, INK if not peak else col, 16, weight="700"))
        o.append(_t(x, y + r + 52, sub, MUTE, 13.5))
    o.append(_t(W / 2, 30, "Step 0 → 7 · 봉우리는 두 개다", MUTE, 15))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 07 · 세 개의 자물쇠 ────────────────────────────────────────────
LOCKS = [("[MODEL]", "로지스틱 회귀 · C = 0.1", "6차시 CV 가 확정했다"),
         ("[VARIABLES]", "Model A 변수 세트", "4차시에 확정했다"),
         ("[CUTOFF]", "1.500", "train 에서 계산 — test 로 다시 계산하지 않는다")]


def fig_locks():
    W, H = 1080, 216
    o = [_open("locks", W, H, 216)]
    for i, (tag, what, why) in enumerate(LOCKS):
        y = 18 + i * 66
        o.append(_r(40, y, 1000, 56, PANEL, rx=10, extra=f' stroke="{FRAME}" stroke-width="1"'))
        o.append(_lock(78, y + 20, RD))
        o.append(_r(112, y + 15, 92, 26, "#2a1416", rx=5,
                    extra=f' stroke="{RD}" stroke-width="1.6"'))
        o.append(_t(158, y + 33, "LOCKED", RD, 15, fam=POP, weight="700"))
        o.append(_t(228, y + 34, tag, AM, 16, anchor="start", weight="700"))
        o.append(_t(400, y + 34, what, INK, 17, anchor="start", weight="700"))
        o.append(_t(1020, y + 34, why, MUTE, 14, anchor="end"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 08 · test 최종 결과 막대 + Dummy 기준선 ────────────────────────
#  (구 슬라이드08 표를 이 그림이 대체한다 — 값은 강사 노트 '실측 대조표'와 1:1)
BARS = [("A · Dummy", .5000, ".3434", ".0000", ".0000", ".5000", GY, .45),
        ("A · 로지스틱", .6718, ".5126", ".6484", ".4683", ".6316", CY, 1),
        ("A · 결정 트리", .6375, ".4359", ".6593", ".4380", ".6084", GY, .8),
        ("A · 랜덤 포레스트", .6566, ".4846", ".6154", ".4786", ".6324", GY, .8),
        ("B · 로지스틱", .7165, ".6153", ".5934", ".4779", ".6272", LM, 1),
        ("B · 랜덤 포레스트", .7032, ".5541", ".5714", ".4906", ".6305", GY, .8)]
BX0, BX1, BMAX = 196, 716, .75


def fig_testbars():
    W, H = 1080, 300
    o = [_open("testbars", W, H, 286)]
    cols = [(836, "AP"), (906, "Recall"), (978, "Precision"), (1056, "Bal.Acc")]
    for cx, lab in cols:
        o.append(_t(cx, 40, lab, CY, 13, anchor="end", weight="700"))
    o.append(_t(BX0, 40, "ROC-AUC", CY, 13, anchor="start", weight="700"))
    top, bh, gap = 54, 24, 12
    for i, (name, auc, ap, rec, pre, bal, col, op) in enumerate(BARS):
        y = top + i * (bh + gap)
        w = (auc / BMAX) * (BX1 - BX0)
        o.append(_r(BX0, y, w, bh, col, rx=3, extra=f' opacity="{op}"'))
        o.append(_t(BX0 - 12, y + 17, name, INK if op == 1 else MUTE, 15,
                    anchor="end", weight="700" if op == 1 else None))
        o.append(_t(BX0 + w + 9, y + 17, f"{auc:.4f}".lstrip("0"), col, 15,
                    anchor="start", weight="700"))
        for (cx, _), v in zip(cols, (ap, rec, pre, bal)):
            o.append(_t(cx, y + 17, v, MUTE, 13.5, anchor="end"))
    ylast = top + len(BARS) * (bh + gap)
    dx = BX0 + (.5 / BMAX) * (BX1 - BX0)
    o.append(_line(dx, top - 16, dx, ylast - 4, AM, 2, ' stroke-dasharray="6 5"'))
    o.append(_t(dx, top - 24, "DUMMY .5000 — 동전 던지기", AM, 14, weight="700"))
    o.append(_line(BX0, ylast + 2, BX1, ylast + 2, FRAME, 1))
    o.append(_t(W / 2, H - 8,
                "여섯 모델 · test 265명 · 단 한 번의 측정", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 11 · 이미 쓰여 있던 기록 ───────────────────────────────────────
def fig_prophecy():
    W, H = 1080, 236
    o = [_open("prophecy", W, H, 236)]
    o.append(_arrow_defs([("p_am", AM)]))
    # 왼쪽 — 오늘의 결과
    o.append(_r(40, 40, 400, 150, PANEL, rx=12, extra=f' stroke="{FRAME}" stroke-width="1.4"'))
    o.append(_t(240, 70, "오늘의 결과", MUTE, 15))
    o.append(_t(240, 116, "포레스트의 패배", RD, 30, fam=POP, weight="700"))
    o.append(_t(240, 150, "CV 1위 → test 2위 · −.0085", RD, 16))
    o.append(_t(240, 176, "test 1위는 로지스틱", MUTE, 14))
    # 화살표
    o.append(f'<path d="M456,115 L536,115" fill="none" stroke="{AM}" stroke-width="2.4" '
             f'marker-end="url(#p_am)"/>')
    o.append(_t(496, 100, "예언대로", AM, 14, weight="700"))
    # 오른쪽 — 2주 전 메모지
    o.append(f'<g transform="rotate(-2 800 115)">')
    o.append(_r(566, 34, 476, 162, "#efe9d6", rx=6,
                extra=' stroke="#c9c0a6" stroke-width="1.4"'))
    o.append(_r(566, 34, 476, 26, "#e2d9bd", rx=6))
    o.append(_t(804, 52, "2주 전의 기록 — 6차시 슬라이드 21", "#5b5340", 14, weight="700"))
    o.append(_r(584, 76, 440, 26, "#f6e05e", rx=3, extra=' opacity="0.75"'))
    o.append(_t(590, 95, "후보가 많은 모델이 약간 유리하게 채점된다 …", "#1a1c22", 16,
                anchor="start", weight="700"))
    o.append(_r(584, 116, 440, 52, "#f6e05e", rx=3, extra=' opacity="0.75"'))
    o.append(_t(590, 136, "+0.012 라는 작은 격차는", "#1a1c22", 17, anchor="start", weight="700"))
    o.append(_t(590, 160, "이 편향으로 뒤집힐 수 있다.", "#1a1c22", 17, anchor="start", weight="700"))
    o.append(_t(1024, 186, "— 우리가 직접 써 둔 문장", "#7a7156", 13, anchor="end"))
    o.append("</g>")
    o.append(_t(W / 2, H - 8,
                "단정했다면 오늘 말을 바꿔야 했다 — 우리는 바꿀 말이 없다", LM, 16, weight="700"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 14 · 견고함 vs 변동성 게이지 ───────────────────────────────────
def _arcp(cx, cy, r, a0, a1):
    x0, y0 = cx + r * math.cos(math.radians(a0)), cy - r * math.sin(math.radians(a0))
    x1, y1 = cx + r * math.cos(math.radians(a1)), cy - r * math.sin(math.radians(a1))
    large = 1 if abs(a0 - a1) > 180 else 0
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f}"


def _needle(cx, cy, r, ang, col, w=4, op=1.0):
    x = cx + r * math.cos(math.radians(ang))
    y = cy - r * math.sin(math.radians(ang))
    return (f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{col}" '
            f'stroke-width="{w}" stroke-linecap="round" opacity="{op}"/>')


def fig_gauges():
    W, H = 1080, 182
    o = [_open("gauges", W, H, 152)]
    o.append(_t(20, 20, "4차시 \"상위 25%\" 의 두 자의성 — 분위수 × 부등호를 바꿔 여섯 번 다시 재면",
                MUTE, 14, anchor="start"))
    specs = [
        (152, 356, "성능 수치 (AUC)", (.50, 1.00), [.6449, .6605, .6718], LM,
         (".50", "1.00"), ".6449 ~ .6718 · 폭 .027", "① 견고하다"),
        (676, 880, "고스트레스 양성률", (0, 100), [18.9, 24.9, 34.3], AM,
         ("0%", "100%"), "18.9% ~ 34.3% · 1.8배", "② 크게 흔들린다"),
    ]
    for cx, tx, title, (lo, hi), vals, col, ends, rng, verdict in specs:
        cy, r = 150, 84
        o.append(f'<path d="{_arcp(cx, cy, r, 180, 0)}" fill="none" stroke="{GRID}" stroke-width="20"/>')
        angs = [180 - (v - lo) / (hi - lo) * 180 for v in vals]
        o.append(f'<path d="{_arcp(cx, cy, r, max(angs), min(angs))}" fill="none" '
                 f'stroke="{col}" stroke-width="20" opacity="0.9"/>')
        for k, a in enumerate(angs):
            mid = (k == 1)
            o.append(_needle(cx, cy, r - 14, a, col if mid else INK, 5 if mid else 3,
                             1.0 if mid else 0.4))
        o.append(_dot(cx, cy, 10, col))
        o.append(_t(cx - r - 6, cy + 18, ends[0], FAINT, 12.5, anchor="end"))
        o.append(_t(cx + r + 6, cy + 18, ends[1], FAINT, 12.5, anchor="start"))
        o.append(_t(tx, 78, title, INK, 17, anchor="start", weight="700"))
        o.append(_t(tx, 106, rng, col, 17, anchor="start", weight="700"))
        o.append(_t(tx, 140, verdict, col, 21, anchor="start", fam=POP, weight="700"))
    o.append(_line(556, 44, 556, 172, FRAME, 1, ' stroke-dasharray="5 5"'))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 21 · 두 갈래가 같은 답을 낸다 ─────────────────────────────────
def fig_pipeline():
    W, H = 1080, 268
    o = [_open("pipeline", W, H, 268)]
    o.append(_arrow_defs([("pl_c", CY), ("pl_l", LM), ("pl_g", GY)]))
    # 입력
    o.append(_r(24, 62, 186, 84, PANEL, rx=10, extra=f' stroke="{FRAME}" stroke-width="1.4"'))
    o.append(_t(117, 90, "같은 입력", INK, 16, weight="700"))
    o.append(_t(117, 112, "modeling_frame.parquet", MUTE, 12.5))
    o.append(_t(117, 130, "seed 42 · test_size 0.20", MUTE, 12.5))
    # 두 갈래
    lanes = [(28, "노트북에서 손으로 짠 코드", "session8.ipynb", CY, "pl_c"),
             (120, "공식 파이프라인", "scripts/run_models.py", LM, "pl_l")]
    for y, name, path, col, mk in lanes:
        o.append(f'<path d="M216,104 C250,104 252,{y+32} 286,{y+32}" fill="none" '
                 f'stroke="{GY}" stroke-width="2" marker-end="url(#pl_g)"/>')
        o.append(_r(294, y, 300, 64, PANEL, rx=10, extra=f' stroke="{col}" stroke-width="1.6"'))
        o.append(_t(444, y + 26, name, col, 16, weight="700"))
        o.append(_t(444, y + 48, path, MUTE, 13))
        o.append(f'<path d="M602,{y+32} C638,{y+32} 640,104 676,104" fill="none" '
                 f'stroke="{col}" stroke-width="2" marker-end="url(#{mk})"/>')
    # 합류 결과
    o.append(_r(684, 52, 210, 104, PANEL, rx=10, extra=f' stroke="{LM}" stroke-width="2"'))
    o.append(_t(789, 78, "A 로지스틱 ROC-AUC", MUTE, 13))
    o.append(_t(789, 108, ".6718  =  .6718", INK, 21, fam=POP, weight="700"))
    o.append(_t(789, 132, "노트북        파이프라인", MUTE, 12.5))
    o.append(_r(916, 76, 140, 56, "#12200f", rx=10, extra=f' stroke="{LM}" stroke-width="1.6"'))
    o.append(_t(986, 100, "전부 일치", LM, 16, weight="700"))
    o.append(_t(986, 120, "True", LM, 15, fam=POP, weight="700"))
    o.append(_line(24, 188, W - 24, 188, FRAME, 1, ' stroke-dasharray="6 5"'))
    # 하단 — final_report.md
    o.append(_t(24, 218, "final_report.md", INK, 16, anchor="start", weight="700"))
    o.append(_r(200, 200, 400, 50, PANEL, rx=8, extra=f' stroke="{CY}" stroke-width="1.4"'))
    o.append(_t(400, 220, "표와 숫자", CY, 15, weight="700"))
    o.append(_t(400, 240, "코드가 자동으로 채운다", MUTE, 13))
    o.append(_r(624, 200, 432, 50, PANEL, rx=8, extra=f' stroke="{AM}" stroke-width="1.6"'))
    o.append(_t(840, 220, "TODO(사람) 6칸", AM, 15, weight="700"))
    o.append(_t(840, 240, "결론 · 해석 · 한계 · 윤리 — 사람이 쓴다", MUTE, 13))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 23 · 8주 여정 ─────────────────────────────────────────────────
JOURNEY = ["문제 정의", "컬럼 검증", "지속 검증", "누출 · 정의", "불확실성", "목적", "해석", "기록"]


def fig_journey():
    W, H = 1080, 88
    y = 42
    o = [_open("journey", W, H, 88)]
    xs = [76 + i * 132 for i in range(8)]
    o.append(_line(xs[0], y, xs[-1], y, FRAME, 3))
    for i, k in enumerate(JOURNEY):
        x, last = xs[i], (i == 7)
        if last:
            o.append(_dot(x, y, 27, CY, ' opacity="0.16"'))
        o.append(_dot(x, y, 19 if last else 15, PANEL,
                      f' stroke="{CY if last else GY}" stroke-width="{3 if last else 2}"'))
        o.append(_t(x, y + 6, str(i + 1), CY if last else INK, 16 if last else 14,
                    fam=POP, weight="700"))
        o.append(_t(x, y + 36, k, CY if last else MUTE, 13, weight="700" if last else None))
    o.append(_t(xs[-1], y - 26, "오늘", CY, 13, weight="700"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 25 · 8주가 남긴 것 ────────────────────────────────────────────
ARTIFACTS = [("configs/", "variables.yaml", -3), ("reports/", "data_quality.md", 2),
             ("data/processed/", "modeling_frame.parquet", -1), ("reports/", "model_metrics_cv.csv", 3),
             ("reports/", "feature_importance.csv", -2), ("reports/", "final_report.md", 1),
             ("session1~8/", "*.ipynb + lecture_notes", -2)]


def fig_artifacts():
    W, H = 1080, 128
    o = [_open("artifacts", W, H, 128)]
    cw, gap = 138, 14
    x0 = (W - (len(ARTIFACTS) * cw + (len(ARTIFACTS) - 1) * gap)) / 2
    for i, (folder, name, rot) in enumerate(ARTIFACTS):
        x = x0 + i * (cw + gap)
        hot = name == "final_report.md"
        col = LM if hot else FRAME
        o.append(f'<g transform="rotate({rot} {x+cw/2} 56)">')
        o.append(_r(x, 16, cw, 80, PANEL, rx=7,
                    extra=f' stroke="{col}" stroke-width="{1.8 if hot else 1}"'))
        o.append(_line(x + 12, 36, x + cw - 12, 36, GRID, 1))
        o.append(_line(x + 12, 46, x + cw - 26, 46, GRID, 1))
        o.append(_t(x + cw / 2, 68, folder, FAINT, 11.5))
        o.append(_t(x + cw / 2, 86, name, LM if hot else MUTE, 11.5, weight="700"))
        o.append("</g>")
    o.append(_t(W / 2, 120, "8주가 남긴 것 — 모델이 아니라 재현 가능한 기록", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


FIGURES = {
    "envelope": fig_envelope,
    "arc": fig_arc,
    "steprail": fig_steprail,
    "seal": fig_seal,
    "locks": fig_locks,
    "testbars": fig_testbars,
    "rankflip": fig_rankflip,
    "winner": fig_winner,
    "prophecy": fig_prophecy,
    "ci": fig_ci,
    "photofinish": fig_photofinish,
    "gauges": fig_gauges,
    "stair": fig_stair,
    "fnprof": fig_fnprof,
    "borderline": fig_borderline,
    "pipeline": fig_pipeline,
    "journey": fig_journey,
    "artifacts": fig_artifacts,
}

BLOCK = re.compile(r'<svg data-fig="([A-Za-z0-9_]+)".*?</svg>', re.S)

# 이 그림들이 쓰는 실측값은 전부 강사 노트('실측 대조표')에 문자열로 존재해야 한다.
# 하나라도 없으면 슬라이드와 노트가 다른 숫자를 말하고 있다는 뜻이다.
SOURCED = [".6535", ".6651", ".6355", ".6718", ".6566", ".6375", "−.0085", "+.0183", "+.0020",
           ".6041", ".7398", ".5865", ".7261", ".136", ".140",
           "+.0147", "−.0157", "+.0443", "83.9",
           "34.3", "24.9", "18.9", "142", "1,321", "1,056", "265",
           "3.602", "2.839", "3.460", "4.455", "3.552", "4.545",
           "3.677", "2.949", "3.542", "1.547", "2.071", "1.409",
           "75.7", "44.5", "50.0", "59.3", "32", "59", "107",
           # 슬라이드 08 막대그림이 표를 대체하면서 넘겨받은 지표들
           ".3434", ".5126", ".4359", ".4846", ".6153", ".5541",
           ".6593", ".6154", ".5934", ".5714", ".4380", ".4786", ".4779", ".4906",
           ".6084", ".6324", ".6272", ".6305", ".6484", ".4683", ".6316",
           # 그림이 인용하는 6차시·설정값
           "+0.012", "C = 0.1", "1.500", "seed 42", "test_size 0.20"]


def render(name):
    svg = FIGURES[name]()
    assert svg.count("<svg") == 1 and svg.count("</svg>") == 1, f"{name}: 중첩 svg"
    assert f'data-fig="{name}"' in svg, f"{name}: data-fig 누락"
    return svg


def _check_registry():
    html = open(HTML, encoding="utf-8").read()
    found = BLOCK.findall(html)
    missing = set(FIGURES) - set(found)
    extra = set(found) - set(FIGURES)
    dup = [n for n in set(found) if found.count(n) > 1]
    ok = not (missing or extra or dup)
    if not ok:
        print(f"  ❌ 레지스트리 불일치 — HTML에만:{sorted(extra)} 코드에만:{sorted(missing)} 중복:{dup}")
    return ok


def verify_numbers():
    if not os.path.exists(NOTES):
        print("  ⚠️ lecture_notes.md 없음 — 수치 대조 생략")
        return 0, []
    src = open(NOTES, encoding="utf-8").read()
    return len(SOURCED), [v for v in SOURCED if v not in src]


def apply(dry_run=False):
    if not _check_registry():
        return 1
    for path in TARGETS:
        html = open(path, encoding="utf-8").read()
        found = BLOCK.findall(html)
        out = BLOCK.sub(lambda m: render(m.group(1)), html)
        if out.count("<svg") != out.count("</svg>"):
            print(f"  ❌ {os.path.basename(path)}: 태그 불균형")
            return 1
        changed = out != html
        print(f"  {os.path.basename(path):16s} 자리 {len(found)}개 · 치환 {len(found)}개 · "
              f"{'변경 있음' if changed else '변경 없음'}")
        if not dry_run and changed:
            open(path, "w", encoding="utf-8").write(out)
    print(f"  등록 {len(FIGURES)}개")
    if not dry_run:
        print("  ✅ 반영 완료")
    return 0


def main(argv):
    if "--dump" in argv:
        print(render(argv[argv.index("--dump") + 1]))
        return 0
    n, fails = verify_numbers()
    print(f"  수치 대조 {n}개", end=" · ")
    if fails:
        print("❌ 강사 노트에서 확인 안 되는 값:", fails)
        return 1
    print("전부 강사 노트 실측 대조표에 존재")
    return apply(dry_run="--check" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

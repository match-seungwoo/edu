# -*- coding: utf-8 -*-
"""session7.html 의 인라인 SVG 그림을 생성 · 갱신한다.

사용법
    python3 _build_s7_figs.py              # session7.html 안의 그림을 제자리 갱신
    python3 _build_s7_figs.py --check      # 수치 대조 + 구조 검사만 (파일 안 건드림)
    python3 _build_s7_figs.py --dump NAME  # 그림 하나를 표준출력으로

동작 방식
    session7.html 의 각 그림은 <svg data-fig="NAME" ...> ... </svg> 로 표시돼 있다.
    이 스크립트는 그 블록을 통째로 새로 생성한 SVG 로 바꾼다. 슬라이드의 글·배치는
    HTML 에서 직접 고치고, 그림은 여기서 고친 뒤 이 스크립트를 다시 돌린다.
    (_build_s6_figs.py 와 같은 방식이다.)

★ 불변성 (수정 전 반드시 확인)
    1. data-fig NAME 은 HTML 안에서 유일하고 FIGURES 키와 1:1 대응한다.
       확인 → _check_registry(): 대칭차집합 == 공집합 + 각 NAME 등장 1회 assert.
    2. 각 SVG 블록에 중첩 <svg> 가 없다 (여는 <svg 부터 첫 </svg> 까지가 한 블록).
       확인 → render(): 생성물마다 count("<svg") == 1 assert + 치환 후 개수 일치 검사.
    3. 이 그림들은 **새 수치를 만들지 않는다.** 전부 session7.html 슬라이드에 이미
       있던 값이고, 그림이 기존 표/코드블록을 대체한 경우 그 표의 값을 그대로 옮겼다.
       확인 → verify_numbers(): SOURCED 는 lecture_notes.md 에 대조, OWNED 는 소유 보고.

수치 출처
    .1001 / .0559 / 변수별 train·validation 6개 → 구 슬라이드07 표 (이 그림이 대체)
    .866 / .467 / 순위 이동 3건                  → 구 슬라이드10 코드블록 (이 그림이 대체)
    .0291 / .0122 / .0382 / .0268 / .615         → 구 슬라이드11 표 (이 그림이 대체)
    TN 425 · FP 275 · TP 220 · FN 136            → 구 슬라이드13 표 (이 그림이 대체)
    FN/TP/TN 프로파일 12개 값                     → 구 슬라이드14 표 (이 그림이 대체)
    위 값은 전부 session7/lecture_notes.md '실측 대조표' 에도 문자열로 존재한다.

팔레트 (1~6차시 덱과 동일)
    Logistic/강조 #00f2ff · 양호/합의 #adff2f · 위험/FN #ff8080 · 보조 #8892a4 · 주의 #ffd479
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(_HERE, "session7", "session7.html")
NOTES = os.path.join(_HERE, "session7", "lecture_notes.md")
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


def _legend(items, x, y, gap=150, size=14):
    """items = [(라벨, 색)] — 스와치 + 라벨. 열 머리글처럼 오해되지 않게 쓴다."""
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


# ─── 슬라이드 06 · 젠가 ────────────────────────────────────────────────────────
def fig_jenga():
    W, H = 500, 300
    o = [f'<svg data-fig="jenga" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:300px" xmlns="http://www.w3.org/2000/svg">']
    o.append(_arrow_defs([("a_lm", LM), ("a_rd", RD), ("a_am", AM)]))
    # 왼쪽: 온전한 탑 + 뽑히는 블록
    bx, by, bw, bh = 40, 60, 96, 20
    for i in range(8):
        y = by + i * (bh + 4)
        if i == 4:
            o.append(_r(bx + 26, y, bw, bh, AM, extra=' stroke="#fff" stroke-width="1"'))
            o.append(f'<line x1="{bx+18}" y1="{y+bh/2}" x2="{bx-2}" y2="{y+bh/2}" '
                     f'stroke="{AM}" stroke-width="2" marker-end="url(#a_am)"/>')
        else:
            o.append(_r(bx, y, bw, bh, "#2b3442", extra=f' stroke="{FRAME}" stroke-width="1"'))
    o.append(_t(bx + bw / 2, by - 16, "블록 하나를 뺀다", MUTE, 15))
    o.append(_t(bx + bw / 2, by + 8 * 24 + 22, "= 그 변수만 섞는다", AM, 14))
    # 오른쪽: 두 갈래 결과
    px, pw = 210, 250
    for k, (yy, col, mk, head, sub) in enumerate([
            (58, LM, "a_lm", "탑이 와르르 무너진다", "= 이 탑을 지탱하던 핵심 기둥"),
            (178, GY, None, "빼도 멀쩡하다", "= 있으나 마나 한 블록")]):
        o.append(_r(px, yy, pw, 96, PANEL, rx=8, extra=f' stroke="{col}" stroke-width="2"'))
        # 미니 탑
        for i in range(5):
            y = yy + 78 - i * 13
            if k == 0 and i >= 2:
                o.append(_r(px + 14 + (i - 2) * 9, y, 40, 9, col, rx=2,
                            extra=f' transform="rotate({-14*(i-1)} {px+34} {y+4})" opacity="0.85"'))
            else:
                o.append(_r(px + 14, y, 40, 9, col if k == 0 else GY, rx=2,
                            extra=f' opacity="{0.85 if k == 0 else 0.55}"'))
        o.append(_t(px + 76, yy + 38, head, col, 17, anchor="start", weight="700"))
        o.append(_t(px + 76, yy + 62, sub, MUTE, 14, anchor="start"))
    o.append(_t(W / 2, H - 6, "중요도의 단위 = AUC 하락폭", CY, 15, weight="700"))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 07 · train vs validation ────────────────────────────────────────
TV = [("peer_support", .0199, .0165), ("self_esteem", .0162, .0152),
      ("depression", .0128, .0089), ("합계", .1001, .0559)]


def fig_train_val():
    W, H = 520, 300
    x0, bw_max = 168, 250
    o = [f'<svg data-fig="train_val" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:300px" xmlns="http://www.w3.org/2000/svg">']
    o.append(_arrow_defs([("tv_a", AM)]))
    o.append(_legend([("train (틀린 방식)", RD), ("validation (우리 방식)", LM)],
                     x0, 18, gap=175, size=14))
    top, rowh = 40, 60
    for i, (name, tr, va) in enumerate(TV):
        y = top + i * rowh
        last = (i == len(TV) - 1)
        if last:
            o.append(_line(10, y - 10, W - 10, y - 10, FRAME, 1))
        o.append(_t(160, y + 18, name, INK if last else MUTE, 15,
                    anchor="end", weight="700" if last else None))
        for j, (v, col) in enumerate(((tr, RD), (va, LM))):
            yy = y + j * 20
            o.append(_r(x0, yy, bw_max * v / .1001, 15, col, rx=2,
                        extra="" if last else ' opacity="0.65"'))
            o.append(_t(x0 + bw_max * v / .1001 + 8, yy + 12, f"{v:.4f}".lstrip("0"),
                        col if last else MUTE, 14 if last else 13, anchor="start",
                        weight="700" if last else None))
        if last:
            o.append(f'<path d="M{x0+bw_max*va/.1001+58},{y+30} L{x0+bw_max*tr/.1001-6},{y+8}" '
                     f'fill="none" stroke="{AM}" stroke-width="2" marker-end="url(#tv_a)"/>')
            o.append(_t(x0 + 150, y + 52, "약 1.8배 부풀려짐", AM, 16, weight="700"))
    o.append(_t(W / 2, H - 6,
                "같은 모델 · 같은 방법 — 어디서 재느냐만 다르다", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 10 · 순위 이동 ──────────────────────────────────────────────────
MOVES = [(1, 1, "친구지지", LM), (2, 2, "자아존중감", LM),
         (5, 13, "교우관계", RD), (7, 4, "학교적응", AM), (13, 6, "이중문화", AM)]


def fig_rank_flow():
    W, H = 560, 320
    lx, rx_, top, gap = 150, 410, 62, 17.5
    o = [f'<svg data-fig="rank_flow" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:320px" xmlns="http://www.w3.org/2000/svg">']
    o.append(_t(lx, 24, "로지스틱", CY, 17, weight="700"))
    o.append(_t(rx_, 24, "포레스트", LM, 17, weight="700"))
    o.append(_t(W / 2, 24, "순위 상관 .467", AM, 16, weight="700"))
    o.append(_t(W / 2, 44, "(계수 ↔ 로지스틱 perm 은 .866)", MUTE, 13))
    for r in range(1, 14):
        y = top + (r - 1) * gap
        for xx, an in ((lx - 8, "end"), (rx_ + 8, "start")):
            o.append(_t(xx, y + 4, str(r), FAINT, 12, anchor=an))
        o.append(_line(lx, y, lx + 14, y, FRAME, 1))
        o.append(_line(rx_ - 14, y, rx_, y, FRAME, 1))
    for a, b, name, col in MOVES:
        y1, y2 = top + (a - 1) * gap, top + (b - 1) * gap
        wid = 3 if col in (LM, RD) else 2
        op = 1 if col in (LM, RD) else .8
        o.append(f'<path d="M{lx+14},{y1} C{lx+110},{y1} {rx_-110},{y2} {rx_-14},{y2}" '
                 f'fill="none" stroke="{col}" stroke-width="{wid}" opacity="{op}"/>')
        o.append(_t(lx - 24, y1 + 4, name, col, 13, anchor="end"))
    o.append(_t(W / 2, H - 26, "상위 2개는 두 모델이 똑같이 1·2위로 합의", LM, 15, weight="700"))
    o.append(_t(W / 2, H - 6, "교우관계는 5위 → 13위로 추락", RD, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 11 · 가림(masking) before/after ────────────────────────────────
def fig_masking():
    W, H = 560, 250
    o = [f'<svg data-fig="masking" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:250px" xmlns="http://www.w3.org/2000/svg">']
    o.append(_arrow_defs([("mk_a", LM)]))
    scale, x0 = 3400, 190
    o.append(_legend([("둘 다 있을 때", GY), ("짝을 뺐을 때", LM)], x0, 20, gap=165))
    rows = [("peer_support", .0291, .0382), ("peer_relationship", .0122, .0268)]
    for i, (name, before, after) in enumerate(rows):
        y = 56 + i * 84
        o.append(_t(180, y + 16, name, INK, 15, anchor="end", weight="700"))
        o.append(_r(x0, y, before * scale, 22, GY, rx=2, extra=' opacity="0.75"'))
        o.append(_t(x0 + before * scale + 8, y + 17, f"{before:.4f}".lstrip("0"), MUTE, 14, anchor="start"))
        o.append(_r(x0, y + 30, after * scale, 22, LM, rx=2))
        o.append(_t(x0 + after * scale + 8, y + 47, f"{after:.4f}".lstrip("0"), LM, 15,
                    anchor="start", weight="700"))
        o.append(f'<path d="M{x0+before*scale+2},{y+24} L{x0+after*scale-4},{y+30}" '
                 f'fill="none" stroke="{LM}" stroke-width="2" marker-end="url(#mk_a)"/>')
    o.append(f'<path d="M100,74 C74,110 74,132 100,168" fill="none" stroke="{AM}" '
             f'stroke-width="2" stroke-dasharray="4 3"/>')
    o.append(_t(62, 126, "r = .615", AM, 14, anchor="end"))
    o.append(_t(W / 2, H - 8,
                "변수 자체는 그대로다 — 바뀐 건 옆에 누가 있느냐뿐이다", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 13 · 혼동행렬 ───────────────────────────────────────────────────
def fig_confusion():
    W, H = 520, 262
    cw, ch, x0, y0 = 176, 78, 150, 50
    o = [f'<svg data-fig="confusion" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:262px" xmlns="http://www.w3.org/2000/svg">']
    o.append(_t(x0 + cw / 2, 26, "실제 고스트레스", MUTE, 14))
    o.append(_t(x0 + cw + cw / 2, 26, "실제 일반", MUTE, 14))
    cells = [("TP", 220, GY, 0, 0), ("FP", 275, GY, 1, 0),
             ("FN", 136, RD, 0, 1), ("TN", 425, GY, 1, 1)]
    for name, n, col, cx, cy in cells:
        x, y = x0 + cx * cw, y0 + cy * ch
        hot = (name == "FN")
        o.append(_r(x + 3, y + 3, cw - 6, ch - 6, "#1d2632" if not hot else "#3a1d1d", rx=6,
                    extra=f' stroke="{RD if hot else FRAME}" stroke-width="{3 if hot else 1}"'))
        o.append(_t(x + cw / 2, y + 36, name, RD if hot else INK, 20, fam=POP, weight="700"))
        o.append(_t(x + cw / 2, y + 62, f"{n}명", RD if hot else MUTE, 17, weight="700" if hot else None))
    o.append(_t(x0 - 12, y0 + 46, "모델: 고스트레스", MUTE, 14, anchor="end"))
    o.append(_t(x0 - 12, y0 + ch + 46, "모델: 일반", MUTE, 14, anchor="end"))
    o.append(_t(W / 2, H - 30, "recall = 220 ÷ (220+136) = 220 ÷ 356 = .618",
                LM, 16, fam=POP, weight="700"))
    o.append(_t(W / 2, H - 8, "고스트레스 학생 10명 중 4명 가까이를 놓친다", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


# ─── 슬라이드 14 · FN/TP/TN 프로파일 ─────────────────────────────────────────
PROF = [("자아존중감", 3.42, 2.76, 3.50), ("친구지지", 4.45, 3.58, 4.57),
        ("부모 감독", 3.42, 2.85, 3.52), ("우울", 1.47, 2.07, 1.38)]


def fig_error_profile():
    W, H = 620, 262
    base, top, gw, bw = 196, 44, 148, 30
    o = [f'<svg data-fig="error_profile" viewBox="0 0 {W} {H}" width="100%" '
         f'style="max-height:262px" xmlns="http://www.w3.org/2000/svg">']
    for i, (lab, col) in enumerate((("FN (놓침)", AM), ("TP (찾음)", RD), ("TN (맞게 제외)", GY))):
        o.append(_r(150 + i * 150, 8, 12, 12, col, rx=2))
        o.append(_t(168 + i * 150, 19, lab, MUTE, 14, anchor="start"))
    for gv in (1, 2, 3, 4, 5):
        y = base - (gv / 5) * (base - top)
        o.append(_line(28, y, W - 12, y, GRID, 1))
        o.append(_t(20, y + 4, str(gv), FAINT, 11, anchor="end"))
    for gi, (name, fn, tp, tn) in enumerate(PROF):
        gx = 44 + gi * gw
        for bi, (v, col) in enumerate(((fn, AM), (tp, RD), (tn, GY))):
            h = (v / 5) * (base - top)
            x = gx + bi * (bw + 6)
            o.append(_r(x, base - h, bw, h, col, rx=2,
                        extra="" if bi != 1 else ' opacity="0.9"'))
            o.append(_t(x + bw / 2, base - h - 6, f"{v:.2f}", col, 13, weight="700"))
        o.append(_t(gx + (3 * bw + 12) / 2 - 3, base + 20, name, INK, 15, weight="700"))
    o.append(_line(28, base, W - 12, base, FRAME, 1))
    o.append(_t(W / 2, H - 22, "FN 열은 TP 가 아니라 TN 을 닮았다", LM, 17, weight="700"))
    o.append(_t(W / 2, H - 4, "네 지표 모두 TN 과의 차이가 0.1 근처다", MUTE, 14))
    o.append("</svg>")
    return "".join(o)


FIGURES = {
    "jenga": fig_jenga,
    "train_val": fig_train_val,
    "rank_flow": fig_rank_flow,
    "masking": fig_masking,
    "confusion": fig_confusion,
    "error_profile": fig_error_profile,
}

BLOCK = re.compile(r'<svg data-fig="([A-Za-z0-9_]+)".*?</svg>', re.S)

# 2단 검증.
#  SOURCED — 다른 산출물(lecture_notes.md '실측 대조표')에도 있는 값. 불일치를 잡을 수 있다.
#  OWNED   — 구 슬라이드07 표에만 있던 변수별 train/validation 값. 그 표를 fig_train_val 이
#            대체했으므로 이제 이 파일이 유일한 기록처다. 대조할 상대가 없으므로 검증 대신
#            '소유'로 보고한다. 값의 출처는 git 이력의 session7.html 슬라이드06 표다.
#            (다른 곳에 이 값을 적게 되면 SOURCED 로 옮길 것.)
SOURCED = [".1001", ".0559",
           ".866", ".467", ".0291", ".0122", ".0382", ".0268", ".615",
           "425", "275", "220", "136", ".618",
           "3.42", "2.76", "3.50", "4.45", "3.58", "4.57", "2.85", "3.52",
           "1.47", "2.07", "1.38"]
OWNED = [".0199", ".0165", ".0162", ".0152", ".0128", ".0089"]


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
    fails = [v for v in SOURCED if v not in src]
    stray = [v for v in OWNED if v in src]   # 다른 곳에 생겼으면 SOURCED 로 승격할 것
    if stray:
        print(f"  ℹ️ OWNED 값이 강사 노트에도 생겼다 → SOURCED 로 옮겨라: {stray}")
    return len(SOURCED), fails


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
        print("❌ lecture_notes.md 에서 확인 안 되는 값:", fails)
        return 1
    print("전부 강사 노트 실측 대조표에 존재")
    print(f"  이 그림이 소유한 값 {len(OWNED)}개(구 슬라이드07 표) — 대조 상대 없음, 검증 생략")
    return apply(dry_run="--check" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

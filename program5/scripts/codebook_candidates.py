#!/usr/bin/env python3
"""코드북 → 변수 후보 체크리스트 (reports/codebook_candidates.md) 생성기.

실행: python scripts/codebook_candidates.py

이 문서는 variables.yaml 이 **아니다**. 사람이 코드북·조사표와 대조해 확인한 뒤
variables.yaml 을 직접 채우는 것을 돕는 "후보 + 증거" 목록이다.
(역채점 여부는 문항 텍스트를 사람이 읽고 판단한다 — 자동으로 정하지 않는다.)

불변성 (깨지면 후보가 오염된다):
  (1) 코드북 '변수명' + '_w5/_w6' == 실데이터 컬럼명
      — 깨지면: 존재하지 않는 변수 후보 (hallucination)
      — 확인: 모든 후보를 실데이터 헤더와 대조해 ❌ 로 표시한다
  (2) 값/값설명 행은 직전 '변수명' 행에 속한다 (병합셀 ffill 구조)
      — 깨지면: 값 라벨이 엉뚱한 문항에 붙는다
      — 확인: S_GENDER(1=남성/2=여성) 등 알려진 변수 라벨로 구조 검증함
  (3) CSV 의 공백(' ') 은 결측(미참여·무응답)이다
      — 깨지면: 관측 범위·결측 수 왜곡
      — 확인: 5차 데이터 전 컬럼에서 비숫자 값이 ' ' 하나뿐임을 실측 확인함
"""
import argparse
import fnmatch
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# ── variables.yaml 구성개념 ↔ 코드북 (중영역, 소영역) 매핑 ──────────────
# 소영역 이름은 코드북 원문 그대로다 (교우관계 Ⅱ 는 로마숫자 유니코드임에 주의).
CONSTRUCTS = [
    # (yaml 이름, 중영역, [소영역...], 기대 문항수, 기대 척도, 비고)
    ("acculturative_stress (target w6 / previous w5)", "이중문화 경험",
     ["문화적응스트레스"], 10, "4점", "SAFE 수정판 — 5·6차 모두 측정되어야 RQ3 성립"),
    ("self_esteem", "사회정서행동", ["자아존중감I", "자아존중감II"], None, "4점",
     "선행연구는 자아탄력성 사용 — 5차에 어느 척도가 있는지 확인"),
    ("ego_resilience", "사회정서행동", ["자아탄력성"], 14, "4점", ""),
    ("depression", "사회정서행동", ["우울"], 10, "4점", ""),
    ("social_withdrawal", "사회정서행동", ["사회적 위축"], 5, "4점", ""),
    ("life_satisfaction", "사회정서행동", ["삶의 만족도"], 3, "4점", ""),
    ("family_support", "부모와의 관계", ["가족의 지지"], 7, "4점", ""),
    ("parenting_attitude", "부모와의 관계",
     ["부모의 양육태도: 감독", "부모의 양육태도: 방임"], None, "4점",
     "감독/방임 두 하위척도 — 어느 쪽을 쓸지(또는 둘 다) 사람이 결정"),
    ("peer_support", "친구", ["친구의 지지"], 7, "5점", "척도 범위가 다르다 — 표준화 전제"),
    ("peer_relationship", "친구", ["교우관계(적응)I", "교우관계(적응)Ⅱ"], None, "4점",
     "두 하위척도 — 확인 필요"),
    ("bullying", "친구", ["집단괴롭힘 피해경험"], None, None, "응답 범주(예/아니오?) 확인"),
    ("school_adjustment", "학교생활", ["학교적응: 학습활동", "학교적응: 학업"], None, "4점",
     "두 하위척도 — 확인 필요"),
    ("teacher_support", "학교생활", ["교사의 지지"], 6, "5점", "척도 범위가 다르다"),
    ("bicultural_attitude", "이중문화 경험", ["이중문화수용태도"], 10, "4점", ""),
    ("national_identity", "이중문화 경험", ["국가정체성"], 4, "4점", ""),
    ("korean_proficiency", "언어 능력", ["자신의 한국어 실력"], 4, "4점", ""),
    ("background: sex", "학생", ["성별"], 1, None, "단일문항"),
    ("background: economic_status", "가구특성", ["가정의 경제적 수준"], None, None,
     "여러 문항 중 어느 것을 쓸지 사람이 결정"),
]


def find_one(pattern, where=RAW):
    # macOS 는 한글 파일명을 NFD(자모 분해)로 저장한다 → NFC 로 맞춰 비교한다.
    nfc = lambda s: unicodedata.normalize("NFC", s)
    hits = sorted(p for p in where.rglob("*")
                  if p.is_file() and fnmatch.fnmatch(nfc(str(p.relative_to(where))),
                                                     nfc(pattern)))
    if not hits:
        sys.exit(f"🛑 파일을 찾지 못했다: {pattern} (data/raw 확인)")
    return hits[0]


def load_codebook(xlsx):
    cb = pd.read_excel(xlsx, sheet_name="codebook", header=2).iloc[:, 1:9]
    cb.columns = ["대영역", "중영역", "소영역", "조사항목", "실제문항", "변수명", "값", "값설명"]
    cb[["대영역", "중영역", "소영역"]] = cb[["대영역", "중영역", "소영역"]].ffill()

    # 변수 1개 = {name, mid, sub, text, labels[(값, 설명)]}
    out, cur = [], None
    for _, r in cb.iterrows():
        if pd.notna(r["변수명"]):
            cur = {"name": str(r["변수명"]).strip(), "mid": r["중영역"], "sub": r["소영역"],
                   "text": r["조사항목"] if pd.notna(r["조사항목"]) else "", "labels": []}
            out.append(cur)
        if cur is not None and pd.notna(r["값설명"]):
            cur["labels"].append((pd.to_numeric(r["값"], errors="coerce"), str(r["값설명"])))
    return out


def load_layout(xlsx):
    lay = pd.read_excel(xlsx, sheet_name="LAYOUT", header=2).iloc[:, 1:]
    lay.columns = ["대영역", "중영역", "소영역", "조사항목", "변수명",
                   "1차", "2차", "3차", "4차", "5차", "6차", "7차", "8차", "9차",
                   "10차", "12차", "14차"]
    return {str(r["변수명"]).strip(): (r["5차"], r["6차"])
            for _, r in lay.iterrows() if pd.notna(r["변수명"])}


def clip(s, n=46):
    s = " ".join(str(s).split())
    return s[:n] + "…" if len(s) > n else s


def obs(df, col, lo, hi):
    """참여자 기준 관측 요약: '1~4 · 결측 12' + 라벨 범위 밖 값 목록(요약)."""
    if col not in df.columns:
        return "❌ 없음", ""
    v = pd.to_numeric(df[col], errors="coerce")
    if v.notna().sum() == 0:
        return "전부 결측", ""
    out_of = []
    if lo is not None and hi is not None:
        out_of = sorted(v.dropna()[(v < lo) | (v > hi)].unique())
    rng = f"{v.min():g}~{v.max():g} · 결측 {int(v.isna().sum())}"
    if len(out_of) > 6:  # 연속형(소득 등)이면 전부 나열하지 않는다 — 음수 코드가 핵심
        neg = [x for x in out_of if x < 0]
        pos = [x for x in out_of if x >= 0][:3]
        shown = ", ".join(f"{x:g}" for x in neg + pos)
        return rng, f"{shown} … 외 {len(out_of) - len(neg) - len(pos)}종 (연속형?)"
    return rng, (", ".join(f"{x:g}" for x in out_of) if out_of else "")


# ── variables_proposed.yaml 생성용 매핑 ──────────────────────────────
# (yaml 키, 중영역, 소영역, note). 원안과 키가 달라지는 곳은 note 에 이유를 적는다.
PROPOSE_PREDICTORS = [
    ("self_esteem", "사회정서행동", "자아존중감I", "자아존중감II(9문항 5점)는 5·6차 미측정"),
    ("ego_resilience", "사회정서행동", "자아탄력성", ""),
    ("depression", "사회정서행동", "우울", ""),
    ("social_withdrawal", "사회정서행동", "사회적 위축", ""),
    ("life_satisfaction", "사회정서행동", "삶의 만족도", ""),
    ("family_support", "부모와의 관계", "가족의 지지", ""),
    ("parenting_monitoring", "부모와의 관계", "부모의 양육태도: 감독",
     "원안 parenting_attitude 를 감독/방임 둘로 분리 제안 — 방향이 반대라 합산 불가"),
    ("parenting_neglect", "부모와의 관계", "부모의 양육태도: 방임",
     "감독과 방향이 반대 — 별도 구성개념으로"),
    ("peer_support", "친구", "친구의 지지", "5점 척도 — 다른 변인과 범위가 다르다"),
    ("peer_relationship", "친구", "교우관계(적응)I", ""),
    ("peer_relationship_b", "친구", "교우관계(적응)Ⅱ", "필요 시 사용 — I 와 별도 하위척도"),
    ("bullying", "친구", "집단괴롭힘 피해경험", "빈도 응답(1=없었다 ~ 4=거의 매일)"),
    ("school_adjustment", "학교생활", "학교적응: 학습활동",
     "원안 school_adjustment 는 학습활동 하위척도로 제안"),
    ("school_adjustment_academic", "학교생활", "학교적응: 학업", "필요 시 사용"),
    ("teacher_support", "학교생활", "교사의 지지", "5점 척도"),
    ("bicultural_attitude", "이중문화 경험", "이중문화수용태도", ""),
    ("national_identity", "이중문화 경험", "국가정체성", ""),
    ("korean_proficiency", "언어 능력", "자신의 한국어 실력", "6차에는 미측정(예측변인이라 무관)"),
]


def propose_yaml(by_sub, w5cols, w6cols, p5, p6, out_path):
    """체크리스트와 같은 근거로 variables_proposed.yaml 을 만든다.

    불변성: 제안하는 모든 item 은 (1) 코드북에 있고 (2) 해당 차수 실데이터 컬럼으로
    존재하며 (3) 참여자 비결측 관측이 있다. 셋 중 하나라도 깨지면 제안에서 빼고
    주석으로 보고한다. status 는 전부 unverified — 게이트는 사람이 연다.
    """
    def usable(base, wave):
        col, df, part = (f"{base}_w5", w5cols, p5) if wave == 5 else (f"{base}_w6", w6cols, p6)
        return col in df and pd.to_numeric(part[col], errors="coerce").notna().sum() > 0

    def block(key, mid, sub, wave, note, indent="  "):
        items = by_sub.get((mid, sub), [])
        ok = [v for v in items if usable(v["name"], wave)]
        vals = [x for v in ok for x, _ in v["labels"] if pd.notna(x)]
        L = [f"{indent}{key}:",
             f"{indent}  wave: {wave}",
             f"{indent}  status: unverified   # 사람이 조사표와 대조 후 verified 로",
             f"{indent}  note: \"{sub} ({mid}) · 코드북 {len(items)}문항 중 {len(ok)}개 실측 확인"
             + (f" · {note}" if note else "") + "\""]
        if len(ok) != len(items):
            missing = [v["name"] for v in items if v not in ok]
            L.append(f"{indent}  # ⚠️ 실데이터 미확인으로 제외: {', '.join(missing)}")
        if vals:
            L.append(f"{indent}  expected_range: [{min(vals):g}, {max(vals):g}]")
        L.append(f"{indent}  items:")
        L += [f"{indent}    - {v['name']}_w{wave}" for v in ok]
        L.append(f"{indent}  reverse_items: []   # TODO(사람): 문항 방향을 조사표로 확인")
        return L if ok else [f"{indent}# {key}: {wave}차 미측정 — 제외 ({sub})"]

    H = [
        "# ─────────────────────────────────────────────────────────────────",
        "# variables_proposed.yaml — 자동 생성 **제안본** (variables.yaml 아님)",
        f"#   생성: python scripts/codebook_candidates.py --propose · {date.today()}",
        "#   근거: 청소년 코드북 + 5·6차 CSV 실측 (reports/codebook_candidates.md)",
        "#",
        "# 사용법: reports/codebook_candidates.md 의 체크박스를 확인하면서",
        "#   ① 각 구성개념의 역채점·문항수를 조사표(PDF)와 대조하고",
        "#   ② status 를 verified 로 바꾼 뒤 variables.yaml 로 반영하고",
        "#   ③ 마지막에 meta.codebook_verified 를 true 로 바꾼다 (Human Review Gate).",
        "# 이 파일의 모든 items 는 코드북·실데이터·참여자 관측 3중으로 확인된 것만 담았다.",
        "# ─────────────────────────────────────────────────────────────────",
        "",
        "meta:",
        "  study: \"MAPS 1기 (다문화청소년패널조사 제1기 패널)\"",
        "  predictor_wave: 5      # 2015, 중2",
        "  target_wave: 6         # 2016, 중3",
        "  codebook_verified: false   # ★ 검증을 마친 사람이 직접 true 로 바꾼다",
        "",
        "id:",
        "  wave5: PID   # 개인 ID. ID 는 가구 ID 라 중복 — 쓰면 안 된다",
        "  wave6: PID",
        "",
        "# CSV 의 결측은 공백(' ') → io.py 가 NaN 으로 읽는다. 척도 문항에서 숫자",
        "# 결측코드는 실측되지 않았다 (income_01 의 -9 는 해당 변수를 쓸 때만 문제).",
        "missing_codes: []",
        "",
        "target:",
        "  name: acculturative_stress_w6",
    ]
    tgt = block("_t", "이중문화 경험", "문화적응스트레스", 6, "")[1:]  # 키 줄 제외하고 재사용
    H += [l[2:] for l in tgt]                     # 한 단계(2칸) 얕은 들여쓰기로
    H += ["  scoring:",
          "    method: mean",
          "    min_valid_items: 8   # 10문항 중 8개 미만 응답이면 결측",
          "",
          "predictors:", ""]
    for key, mid, sub, note in PROPOSE_PREDICTORS:
        H += block(key, mid, sub, 5, note) + [""]
    H += ["# Model B 전용 — Model A 에는 절대 들어가지 않는다.",
          "optional_predictors:", ""]
    H += block("previous_acculturative_stress", "이중문화 경험", "문화적응스트레스", 5,
               "target 과 동일 문항의 5차 버전")
    H += ["    scoring:",
          "      method: mean",
          "      min_valid_items: 8",
          "",
          "background:",
          "  sex:",
          "    wave: 5",
          "    status: unverified",
          "    column: S_GENDER_w5   # 값: 1=남성, 2=여성 (코드북)",
          "    type: categorical",
          "  economic_status:",
          "    wave: 5",
          "    status: unverified",
          "    column: income_03_w5   # 가정형편에 대한 지각(1~5) — 학부모 응답",
          "    type: ordinal",
          "    # 대안: income_01_w5(월소득, 연속·결측코드 -9 실측) — 쓰려면 missing_codes 에 -9 추가",
          ""]
    Path(out_path).write_text("\n".join(H), encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--propose", action="store_true",
                    help="configs/variables_proposed.yaml 제안본도 생성")
    args = ap.parse_args()
    xlsx = find_one("*청소년 코드북*.xlsx")
    w5p = find_one("csv/청소년*/*청소년 5차년도.csv")
    w6p = find_one("csv/청소년*/*청소년 6차년도.csv")

    variables = load_codebook(xlsx)
    layout = load_layout(xlsx)
    by_sub = {}
    for v in variables:
        by_sub.setdefault((v["mid"], v["sub"]), []).append(v)

    # 불변성 (2) 구조 검증 — 알려진 변수의 라벨이 기대와 다르면 즉시 중단.
    gender = next(v for v in variables if v["name"] == "S_GENDER")
    if [(x, y) for x, y in gender["labels"]][:2] != [(1, "남성"), (2, "여성")]:
        sys.exit("🛑 코드북 파싱 구조가 어긋났다 (S_GENDER 라벨 불일치) — 파서를 수정하라")

    NA = ["", " "]
    w5 = pd.read_csv(w5p, na_values=NA)
    w6 = pd.read_csv(w6p, na_values=NA)
    p5 = w5[w5["SURVEY1_w5"] == 1]
    p6 = w6[w6["SURVEY1_w6"] == 1]
    both = w5.loc[w5["SURVEY1_w5"] == 1, "PID"].isin(
        w6.loc[w6["SURVEY1_w6"] == 1, "PID"]).sum()

    L = [
        "# 변수 후보 체크리스트 (codebook_candidates.md)", "",
        f"> 자동 생성: `python scripts/codebook_candidates.py` · {date.today()}",
        "> 근거: 청소년 코드북 xlsx + 청소년 5·6차 CSV 실측 대조.",
        "> **이 문서는 variables.yaml 이 아니다.** 사람이 조사표(설문지 PDF)와 대조해",
        "> 역채점·문항수를 확인한 뒤 variables.yaml 을 직접 채운다.", "",
        "## 0. 데이터 구조 사실 (실측)", "",
        f"- 파일: `{w5p.relative_to(ROOT)}` · `{w6p.relative_to(ROOT)}`",
        f"- 행수: 5차 {len(w5)} · 6차 {len(w6)} — **매 차수 전체 패널이 들어 있다** "
        "(미참여자 포함, 응답은 공백)",
        f"- **join 키는 `PID`(개인)다.** `ID` 는 가구 ID 라 중복된다 "
        f"(PID 유일: 5차 {w5['PID'].is_unique} · 6차 {w6['PID'].is_unique})",
        f"- 참여 플래그 `SURVEY1_wN`(1=참여): 5차 {len(p5)}명 · 6차 {len(p6)}명 · "
        f"**둘 다 참여 {both}명** (선행연구 1,316명 규모와 부합하는지 확인)",
        "- 결측 표현: CSV 는 **공백 문자열 `' '`** — 숫자 결측코드는 발견되지 않았다 "
        "(범위 밖 값 열이 비어 있으면 그 척도에도 없음)", "",
        "## 1. 구성개념별 후보", "",
        "표 읽는 법: `w5/w6 관측` = 참여자 기준 관측범위·결측수, "
        "`범위밖` = 라벨 범위를 벗어난 관측값(=결측코드 후보, 비어 있으면 정상).", ""]

    for yname, mid, subs, exp_n, exp_scale, note in CONSTRUCTS:
        L.append(f"### `{yname}`")
        if note:
            L.append(f"> {note}")
        for sub in subs:
            items = by_sub.get((mid, sub), [])
            if not items:
                L += ["", f"- ⚠️ 코드북에서 소영역 `{sub}` 를 찾지 못했다 — 매핑 확인 필요", ""]
                continue
            vals = [x for v in items for x, _ in v["labels"] if pd.notna(x)]
            lo, hi = (min(vals), max(vals)) if vals else (None, None)
            head = f"**{sub}** (중영역: {mid}) — {len(items)}문항"
            if exp_n:
                head += f" / 기대 {exp_n}문항 {'✅' if len(items) == exp_n else '⚠️ 불일치'}"
            if lo is not None:
                head += f" · 라벨 {lo:g}~{hi:g}"
                if exp_scale:
                    head += f" (기대 {exp_scale})"
            L += ["", head, "",
                  "| 변수 | 문항 내용 | 5차(LAYOUT) | 6차(LAYOUT) | w5 관측 | w6 관측 | 범위밖 |",
                  "|---|---|---|---|---|---|---|"]
            for v in items:
                # 범위 검사는 그 문항 자신의 라벨로 한다 (소영역 안에 연속형·범주형이 섞일 수 있다).
                vi = [x for x, _ in v["labels"] if pd.notna(x)]
                ilo, ihi = (min(vi), max(vi)) if vi else (lo, hi)
                l5, l6 = layout.get(v["name"], ("?", "?"))
                r5, o5 = obs(p5, f"{v['name']}_w5", ilo, ihi)
                r6, o6 = obs(p6, f"{v['name']}_w6", ilo, ihi)
                bad = " / ".join(x for x in (o5, o6) if x)
                L.append(f"| `{v['name']}` | {clip(v['text'])} | {l5} | {l6} | "
                         f"{r5} | {r6} | {bad} |")
            labs = {y for v in items for _, y in v["labels"]}
            if labs and len(labs) <= 8:
                L.append(f"\n값 라벨: {', '.join(f'{x:g}={y}' for x, y in items[0]['labels'] if pd.notna(x))}")
        L += ["",
              "- [ ] 문항수·척도 범위가 조사표와 일치하는가",
              "- [ ] 역채점 문항이 있는가 (문항 내용을 읽고 판단 — 라벨은 전부 정방향 표기다)",
              "- [ ] variables.yaml 에 반영했는가 (`items` / `expected_range` / `reverse_items` / `status`)",
              ""]

    L += ["## 2. variables.yaml 에 함께 채울 것", "",
          "```yaml",
          "id:",
          "  wave5: PID   # 개인 ID (ID 는 가구 ID — 쓰면 안 된다)",
          "  wave6: PID",
          "missing_codes: []   # CSV 는 공백이 결측 → io.py 가 NaN 으로 읽는다.",
          "                    # 단, income_01 에서 -9 가 실측됨 — 유저가이드에서 의미",
          "                    # 확인 후 필요하면 [-9] 를 추가한다.",
          "```", "",
          "- 학부모 파일(부모 문화적응 스트레스 등)은 이번 범위 밖 — 필요 시 별도 확인.",
          "- 검증을 마친 사람이 `meta.codebook_verified: true` 로 바꾼다 (Human Review Gate).", ""]

    out = ROOT / "reports" / "codebook_candidates.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    n_sub = sum(len(s) for _, _, s, *_ in CONSTRUCTS)
    print(f"✅ 생성: {out.relative_to(ROOT)}  (구성개념 {len(CONSTRUCTS)}개 · 소영역 {n_sub}개)")

    if args.propose:
        out2 = propose_yaml(by_sub, w5.columns, w6.columns, p5, p6,
                            ROOT / "configs" / "variables_proposed.yaml")
        print(f"✅ 생성: {out2.relative_to(ROOT)}  (제안본 — 사람 검증 전까지 variables.yaml 아님)")


if __name__ == "__main__":
    main()

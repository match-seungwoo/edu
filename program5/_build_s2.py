# -*- coding: utf-8 -*-
"""session2.ipynb 빌더 — 코드북 검증과 Human Review Gate 열기.

2차시는 이 프로젝트에서 유일하게 "사람의 판단"이 산출물인 차시다.
노트북은 검증을 돕는 도구일 뿐, variables.yaml 편집과 게이트 오픈은
사람이 에디터에서 직접 한다 (노트북이 자동으로 열지 않는다 — 그게 설계다).

실측 수치 근거: reports/codebook_candidates.md · reports/data_inventory.md
(행 1,635 · 참여 5차 1,347 / 6차 1,329 / 둘 다 1,321 · ID 중복 10쌍)
"""
import os

from nb import md, code, save

SETUP = r'''# ── 프로젝트 환경 자동 설정 (Colab / 로컬 공용) ───────────────────────
# 이 셀은 모든 차시 노트북 맨 위에 동일하게 들어간다. 그냥 실행만 하면 된다.
import os, sys

def find_project():
    """AGENTS.md 와 configs/ 가 함께 있는 program5 폴더를 찾는다."""
    cands = [".", "program5", "..", "../program5", "/content/program5",
             "/content/edu/program5", os.path.expanduser("~/program5")]
    for c in cands:
        if os.path.exists(os.path.join(c, "AGENTS.md")) and \
           os.path.exists(os.path.join(c, "configs", "variables.yaml")):
            return os.path.abspath(c)
    return None

PROJECT = find_project()
if PROJECT is None:
    print("⚠️  프로젝트 폴더를 찾지 못했습니다. 아래 둘 중 하나로 해결하세요:")
    print("  (A) Colab: 좌측 파일창에 program5 폴더를 통째로 업로드")
    print("  (B) Colab: !git clone <이 강의 repo 주소>  후 다시 실행")
else:
    os.chdir(PROJECT)
    sys.path.insert(0, os.path.join(PROJECT, "src"))
    print("✅ 프로젝트 경로:", PROJECT)
'''

W5 = "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 5차년도.csv"
W6 = "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 6차년도.csv"

cells = [
md("""# 2차시 — 변수는 어디에 있는가

### 코드북 검증과 Human Review Gate · 심리척도 / pandas · 결측치 · ID join

> **오늘 한 문장:** "1차시에 *무엇을* 예측할지 정했다. 오늘은 그 변수들이 데이터의
> **414개 열 중 어디에** 있는지 코드북으로 확인하고, 사람이 검증을 마쳐 **게이트를 연다**."

오늘의 목표 4가지:

1. 심리척도의 구조(**문항 → 척도 점수**)와 **역채점**이 왜 필요한지 설명할 수 있다.
2. 코드북으로 **변수명·값 라벨·응답 범위**를 해독하고, **join 키(PID)** 를 판별한다.
3. MAPS CSV 의 **결측 표현(공백)** 을 이해하고, 결측 코드를 **추측하지 않는다**.
4. 검증 결과를 `configs/variables.yaml` 에 기록하고 **Human Review Gate 를 연다**
   → `build_dataset.py` 첫 실행 → `reports/data_quality.md`.

> 💡 운영 방식은 1차시와 같다: 셀을 위에서 아래로. `# TODO` 채우고 `# CHECK` 에서 `✅`.
> 단, 오늘의 진짜 산출물인 **variables.yaml 편집은 노트북이 아니라 에디터에서 사람이 직접** 한다."""),

md("""## 🗺️ 오늘의 위치 — 2차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1 ✅ | 문화적응 스트레스 · 예측 vs 인과 · 연구윤리 | feature/target · classification |
| **2 (오늘)** | **심리척도 · 문항 · 역채점** | **pandas · 결측치 · ID join** |
| 3 | 평균/SD/분포/상관 · Cronbach α | 집계 · 시각화 |
| 4~8 | 조작적 정의 → 모델 → 해석 → 보고 | split/누출 → 회귀/트리 → 중요도 → 재현성 |

**오늘의 재료** (원자료와 함께 2026-08-10 수령·생성)

- 청소년 **코드북**(xlsx) · **조사표**(설문지 PDF) · 유저가이드(PDF) — `data/raw/`
- 자동 대조 **체크리스트** `reports/codebook_candidates.md` (코드북 ↔ 실데이터 실측 대조)
- **제안본** `configs/variables_proposed.yaml` (3중 확인된 후보 — 그래도 최종 판단은 사람)

> 🔴 오늘의 규칙 (1차시 마지막 문장): **컬럼명을 절대 추측하지 않는다.**
> 이름이 비슷하다고 같은 변수가 아니다. 코드북이 유일한 근거다."""),

md("""## Step 0 — 환경 설정 + 재료 확인"""),
code('!pip install pandas pyyaml openpyxl -q'),
code(SETUP),
code(r'''# 오늘 쓸 재료가 다 있는지 먼저 확인한다 (1차시 Step 6 습관: 존재 → 형식 → 내용)
import os, unicodedata

def find_file(dirpath, keyword, ext):
    """dirpath 아래에서 이름에 keyword 가 들어간 ext 파일을 찾는다.
    macOS 는 한글 파일명을 자모 분리형(NFD)으로 저장하기도 해서 glob 이 놓친다
    → 이름을 NFC 로 정규화한 뒤 비교한다 (실전에서 자주 만나는 함정)."""
    hits = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            name = unicodedata.normalize("NFC", f)
            if keyword in name and name.endswith(ext):
                hits.append(os.path.join(root, f))
    return sorted(hits)

need = {
    "코드북(xlsx)":  find_file("data/raw", "청소년 코드북", ".xlsx"),
    "조사표(pdf)":   find_file("data/raw", "청소년 설문지", ".pdf"),
    "5차 CSV":      find_file("data/raw/csv", "청소년 5차년도", ".csv"),
    "6차 CSV":      find_file("data/raw/csv", "청소년 6차년도", ".csv"),
    "체크리스트":     find_file("reports", "codebook_candidates", ".md"),
    "제안본 yaml":   find_file("configs", "variables_proposed", ".yaml"),
}
for k, v in need.items():
    print(("✅" if v else "❌"), k, "—", os.path.basename(v[0]) if v else "없음")
print("\n※ 체크리스트/제안본이 없으면:  python scripts/codebook_candidates.py --propose")'''),

md("""## Step 1 — 심리척도: 왜 한 번 안 묻고 열 번 묻나 🧠

"요즘 스트레스 있어?" 한 번 물으면 될 것 같은데, MAPS 는 **10문항**을 쓴다. 이유:

1. **한 번의 측정은 흔들린다.** 그날 기분, 질문 해석 차이 같은 **측정 오차(measurement
   error)** 가 섞인다. 여러 문항의 평균을 내면 오차가 서로 상쇄돼 점수가 안정된다.
2. **마음은 여러 면을 가진다.** 학교에서의 스트레스, 동네에서의 경험, 언어 부담 —
   한 질문으로는 한 면밖에 못 본다.

그래서 구조가 이렇다: **문항(item)** 여러 개 → 묶음이 **척도(scale)** →
답의 평균이 **척도 점수(scale score)**.

**부분 응답 규칙** — 10문항 중 몇 개만 답했다면? 우리 파이프라인은
`min_valid_items: 8` — **8개 미만 응답이면 점수를 결측 처리**한다 (몇 개 안 되는
답으로 평균을 내면 그 점수는 다른 사람과 비교할 수 없기 때문)."""),

code(r'''# 실데이터로 확인 — 6차(2016) target 척도, 문화적응 스트레스 10문항의 척도 점수
# (컬럼명은 코드북에서 확인된 것: s_accul_str_01 ~ 10 + 차수 접미사 _w6)
import pandas as pd
from maps_risk import scoring   # 우리 프로젝트의 점수 계산 모듈을 그대로 쓴다

w6 = pd.read_csv("''' + W6 + r'''", na_values=["", " "], low_memory=False)
part6 = w6[w6["SURVEY1_w6"] == 1]          # 6차 실제 참여자만 (1,329명)

items = [f"s_accul_str_{i:02d}_w6" for i in range(1, 11)]
score = scoring.scale_score(part6, items, method="mean", min_valid_items=8)

print("참여자 수:", len(part6))
print(score.describe().round(3))
print("\n※ 1~4점 척도의 평균이므로 점수도 1~4 사이. 아직 역채점 미적용 — 다음 Step 에서 판단한다.")'''),

md("""## Step 2 — 역채점: 왜 거꾸로 묻나 ⚠️ (오늘의 첫 번째 봉우리)

설문에는 **일부러 방향을 뒤집은 문항**이 섞인다. 이유는 **응답 습관** 때문이다 —
내용을 안 읽고 전부 "그렇다"에 찍는 경향(**묵종 편향, acquiescence bias**)을 걸러내려면,
반대 방향 문항에서 답이 뒤집히는지 봐야 한다.

문제는 채점이다. 스트레스 척도 안에 "한국에서 **더 잘 살 수 있을 것이다**" 같은
**긍정 방향 문항**이 있다면, 그대로 평균을 내면 **높은 스트레스와 낮은 스트레스가
서로 상쇄**돼 척도 전체가 무의미해진다. 그래서 방향이 반대인 문항은 점수를 뒤집는다:

```
역채점 공식:  뒤집힌 점수 = (최소값 + 최대값) − 원점수
4점 척도면:   1↔4, 2↔3
```

> 🔴 **함정:** MAPS 코드북의 값 라벨은 **전부 정방향 표기**(1=전혀 그렇지 않다 …)다.
> 즉 **라벨만 봐서는 역채점 문항을 찾을 수 없다.** 문항의 **텍스트를 읽고** 사람이 판단한다."""),

code(r'''# TODO: 역채점 공식을 완성하라 (4점 척도: scale_min=1, scale_max=4)
def reverse_code(x, scale_min, scale_max):
    return _____________________   # ← 공식을 채워라

print("1점 →", reverse_code(1, 1, 4), "  (4가 나와야 한다)")
print("2점 →", reverse_code(2, 1, 4), "  (3이 나와야 한다)")
print("4점 →", reverse_code(4, 1, 4), "  (1이 나와야 한다)")'''),
code(r'''# CHECK Step2-공식
try:
    assert [reverse_code(v, 1, 4) for v in (1, 2, 3, 4)] == [4, 3, 2, 1], "1↔4, 2↔3 이 되어야 한다"
    assert reverse_code(reverse_code(3, 1, 4), 1, 4) == 3, "두 번 뒤집으면 원래대로 돌아와야 한다"
    assert [reverse_code(v, 1, 5) for v in (1, 5)] == [5, 1], "5점 척도(1~5)에서도 동작해야 한다"
    print("✅ PASS — (min + max) − x. 우리 모듈 src/maps_risk/scoring.py 의 reverse_code 와 같은 식이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 최소값과 최대값을 더한 뒤 원점수를 빼면 양 끝이 서로 교환된다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
def reverse_code(x, scale_min, scale_max):
    return (scale_min + scale_max) - x
```

`(1+4)−1=4`, `(1+4)−4=1` — 양 끝이 맞바뀐다. 두 번 적용하면 원래 값으로 돌아온다
(**involutive** — 이 성질을 `tests/test_scoring.py` 가 검사한다).
</details>"""),

md("""### Step 2 실습 — 이 문항, 뒤집어야 하나

target 척도(문화적응 스트레스 10문항) 중 4개의 **문항 텍스트**다. 척도의 방향은
"점수가 높을수록 **스트레스가 크다**". 각 문항을 읽고 **역채점 후보인지** 판단하라."""),

code(r'''# TODO: 역채점(방향 뒤집기)이 필요해 보이면 True, 정방향이면 False
candidates = {
    "s_accul_str_03  한국에 사는 것에 스트레스를 받는다":                  _____,
    "s_accul_str_05  주변에서 한국 사람처럼 행동하라고 스트레스를 준다":       _____,
    "s_accul_str_08  우리 동네 사람들은 우리 식구를 못살게 군다":            _____,
    "s_accul_str_10  외국인 부모님 나라보다 한국에서 더 잘 살 수 있을 것이다":  _____,   # ← 잘 읽어라
}
for k, v in candidates.items():
    print(("🔁 역채점 후보  " if v else "→ 정방향      ") + k)'''),
code(r'''# CHECK Step2-판단
try:
    got = list(candidates.values())
    assert got == [False, False, False, True], f"[False, False, False, True] 이어야 하는데 {got}"
    print("✅ PASS — 10번만 '잘 산다'는 긍정 방향이라 역채점 후보다. 나머지는 스트레스 정방향.")
    print("   ⚠️ 최종 결정은 조사표(설문지 PDF) 원문과 대조해 variables.yaml 의")
    print("      reverse_items 에 기록한다 — 노트북의 판단은 후보일 뿐이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: '점수가 높을수록 스트레스가 크다'와 방향이 반대인 문항을 찾아라.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

`s_accul_str_10` 만 **True**. "더 잘 살 수 있을 것이다"에 "매우 그렇다(4점)"라고 답한
학생은 스트레스가 **낮은** 쪽이다 — 그대로 평균 내면 이 문항이 점수를 거꾸로 끌어간다.

**주의 2가지**
1. 코드북 값 라벨(1=전혀 그렇지 않다 …)은 열 개 문항 모두 똑같다 — **라벨로는 못 찾는다.**
2. 선행연구에 역채점 명시가 없다 → 조사표 원문 확인 + 3차시 Cronbach α 로 교차 검증한다
   (역채점을 빠뜨리면 α 가 뚝 떨어진다 — 그게 다음 차시의 복선이다).
</details>"""),

md("""## Step 3 — 코드북: 414개 열의 해설서

5차 CSV 는 **1,635행 × 414열**이다. 열 이름(`s_accul_str_01_w5`, `income_01_w5` …)만
보고 뜻을 맞히는 것은 **추측**이다. 뜻은 **코드북(codebook)** 에만 있다:

| 코드북이 알려주는 것 | 예 (문화적응 스트레스 1번) |
|---|---|
| **변수명** | `s_accul_str_01` |
| **문항 텍스트** | "다른 사람이 외국인 부모님 나라의 문화를 갖고 농담할 때 …" |
| **값 라벨(value label)** | 1=전혀 그렇지 않다 · 2=그렇지 않은 편이다 · 3=그런 편이다 · 4=매우 그렇다 |
| **조사표 문항 번호** | 5차 문21① / 6차 문22① |

**실데이터 컬럼명 규칙** (실측으로 확인): `코드북 변수명 + _w차수` → `s_accul_str_01_w5`

그리고 **ID 가 두 개** 있다. 하나는 함정이다:

- `PID` — **개인** ID
- `ID` — **가구** ID (한 집에 형제자매가 둘이면 같은 값!)"""),

code(r'''# TODO: 5차 데이터에서 두 ID 의 중복을 세어 보고, join 키를 골라라
import pandas as pd
w5 = pd.read_csv("''' + W5 + r'''", na_values=["", " "], low_memory=False)

dup_pid = w5["PID"].duplicated().sum()
dup_id  = w5["ID"].duplicated().sum()
print(f"행 수 {len(w5)} · PID 중복 {dup_pid}개 · ID 중복 {dup_id}개")

join_key = "___"    # ← "PID" 또는 "ID" — 5차와 6차를 합칠 때 쓸 키를 골라라'''),
code(r'''# CHECK Step3
try:
    assert dup_pid == 0 and dup_id == 10, f"실측: PID 중복 0 · ID 중복 10 이어야 한다 (지금 {dup_pid}/{dup_id})"
    assert join_key == "PID", "응답자 1명 = 1행이 되려면 '개인'을 유일하게 가리키는 키여야 한다"
    print("✅ PASS — join 키는 PID. ID 는 가구 ID 라 형제자매 10쌍이 같은 값을 공유한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 중복이 0개인 쪽만이 '한 사람 = 한 행'을 보장한다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
join_key = "PID"
```

`ID`(가구)로 join 하면 형제자매 가구에서 **행이 조용히 불어난다**(한 학생의 5차 응답이
형제의 6차 응답과도 짝지어진다). "응답자 1명 = 1행"이 깨지는데 **에러도 안 난다** —
그래서 우리 `build_dataset.py` 는 ID 중복을 발견하면 **일부러 멈추게** 만들어져 있다.
</details>"""),

md("""## Step 4 — 결측: 공백 한 칸의 의미

MAPS CSV 에서 무응답·미참여는 **공백 문자열 `' '`** 로 들어 있다 (5차 전 컬럼 실측 —
숫자 결측 코드는 척도 문항에서 발견되지 않았다). 이걸 모르고 읽으면:

- 공백이 섞인 열은 숫자가 아니라 **문자열(object)** 이 된다 → 평균·비교가 전부 깨진다.
- `na_values=["", " "]` 로 읽으면 공백이 **NaN(결측)** 이 되어 숫자 열로 살아난다.

> 🔴 **결측 코드 추측 금지.** `income_01_w5` 에서 `-9` 가 실측됐지만, 그 뜻(무응답? 모름?)은
> **유저가이드에서 확인 전까지 모른다.** "아마 무응답이겠지"라고 넘겨짚고 NaN 처리하는 것도,
> 방치해서 소득 평균에 −9 가 섞이는 것도 둘 다 사고다. 확인 전까지 그 변수를 안 쓰는 게 정답이다."""),

code(r'''# 데모 — 같은 파일을 두 방식으로 읽어 비교한다 (TODO 아님, 실행하고 출력을 읽어라)
import pandas as pd
col = "s_accul_str_01_w5"

raw   = pd.read_csv("''' + W5 + r'''", low_memory=False)                      # 그냥 읽기
clean = pd.read_csv("''' + W5 + r'''", na_values=["", " "], low_memory=False) # 공백=결측으로 읽기

print("그냥 읽으면   :", raw[col].dtype, "  ← 공백 때문에 문자열이 됐다. raw[col].mean() 은 에러다")
print("공백=NaN 읽기:", clean[col].dtype, " ← 숫자로 살아났다")

n_na = clean[col].isna().sum()
n_nonpart = (clean["SURVEY1_w5"] != 1).sum()
print(f"\n{col} 결측 {n_na}개 = 5차 미참여자 {n_nonpart}명  (참여자 중 결측 0 — 실측과 일치)")
print("→ 결측이 '무작위 빵꾸'가 아니라 '그 해에 조사를 안 받은 사람'이라는 구조를 안 것이다.")'''),

md("""## Step 5 — 검증 워크플로: 사람이 게이트를 연다 🔍 (두 번째 봉우리)

이제 오늘의 본론이다. 순서는 기계적이다:

```
① reports/codebook_candidates.md  ← 자동 대조 체크리스트를 연다
② 구성개념 하나씩:  문항수·척도범위가 조사표(PDF)와 일치하는가 체크
③ 문항 텍스트를 읽고 역채점 후보를 판단  (Step 2 에서 연습한 그것)
④ configs/variables.yaml 에 기록:  items · expected_range · reverse_items · status: verified
   (제안본 configs/variables_proposed.yaml 을 베이스로 쓰되, ③의 판단을 반영)
⑤ 전부 끝나면  meta.codebook_verified: true   ← 이 한 줄이 게이트다
```

**④·⑤ 는 노트북이 아니라 에디터에서 사람이 직접 한다.** 자동 대조(제안본)가 보장하는
것은 "그 컬럼이 존재하고 값이 관측된다"까지다. **의미가 맞는지, 방향이 맞는지**는
기계가 판단할 수 없다 — 그래서 이 게이트는 사람만 열 수 있게 설계돼 있다.

수업에서 함께 판단할 대표 논점 3개:

1. **부모 양육태도** — 코드북엔 "감독"과 "방임" 두 하위척도. **방향이 반대**라 하나로
   합산 불가 → `parenting_monitoring` / `parenting_neglect` 로 분리한다 (제안본에 반영됨).
2. **척도 범위 혼재** — 친구지지·교사지지는 **5점**, 나머지 대부분 4점 → `expected_range`
   를 척도마다 정확히 적어야 범위 검사가 작동한다.
3. **5차에 없는 척도** — 자아존중감Ⅱ·교우관계Ⅱ 등은 5차 미측정 → 후보에서 제외
   (없는 것을 억지로 채우지 않는다)."""),

code(r'''# 체크리스트의 target 부분을 미리 본다 (전체는 에디터에서 열어 작업한다)
text = open("reports/codebook_candidates.md", encoding="utf-8").read()
start = text.find("### `acculturative_stress")
print(text[start:start + 1400])'''),

md("""## Step 6 — ID join: 두 해를 한 표로 합치기

검증이 끝나면 5차(X 후보)와 6차(y 재료)를 **PID 로 병합(merge)** 한다.
1차시 데모(5명+5명=3명)의 실전판인데, MAPS 에는 **함정이 하나 더** 있다:

> 매 차수 파일에는 **미참여자도 행으로 들어 있다** (응답은 전부 공백 = NaN).
> 그래서 5차와 6차를 inner join 해도 행이 줄지 않는다 — **join 성공 ≠ 분석 표본**."""),

code(r'''# TODO: 5차와 6차를 PID 로 병합하고, "두 해 모두 실제 참여"한 사람을 세어라
w6 = pd.read_csv("''' + W6 + r'''", na_values=["", " "], low_memory=False)

joined = w5.merge(w6, on="PID", how="_____")          # ← 교집합 병합 방식을 채워라
both = joined[(joined["SURVEY1_w5"] == 1) & (joined["SURVEY1_w6"] == _____)]   # ← 참여 코드

print(f"5차 {len(w5)}행 + 6차 {len(w6)}행 → join {len(joined)}행 → 두 해 모두 참여 {len(both)}명")'''),
code(r'''# CHECK Step6
try:
    assert len(joined) == 1635, f"join 결과가 {len(joined)}행 — inner 인데도 1,635행이어야 한다 (전 패널이 양쪽에 있으므로)"
    assert len(both) == 1321, f"두 해 모두 참여는 1,321명이어야 하는데 {len(both)}명"
    print("✅ PASS — join 은 1,635행 그대로, 분석 표본은 참여 필터 후 1,321명.")
    print("   'join 이 됐다'와 '데이터가 있다'는 다른 말이다 — 이게 오늘의 마지막 함정이었다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 교집합 병합은 how='inner'. 참여 플래그는 SURVEY1_wN == 1.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
joined = w5.merge(w6, on="PID", how="inner")
both = joined[(joined["SURVEY1_w5"] == 1) & (joined["SURVEY1_w6"] == 1)]
```

패널 구조상 **1,635명 전원이 매 차수 파일에 행으로 존재**하므로 inner join 으로도
아무도 안 사라진다. 진짜 표본은 `SURVEY1_w5 == 1 & SURVEY1_w6 == 1` — **1,321명**
(1차시에 본 실측 그대로: 5차 1,347 · 6차 1,329 · 교집합 1,321).
우리 파이프라인은 이걸 "구성개념 점수가 전부 결측인 행 제거"로 처리한다.
</details>"""),

md("""## Step 7 — 게이트를 열고, 첫 데이터셋을 만든다

`build_dataset.py` 는 실행될 때마다 `config.is_ready_for_scoring()` 에게 묻는다:

1. `meta.codebook_verified` 가 true 인가?  (사람이 검증을 마쳤는가)
2. ID 컬럼명이 기록됐는가?  (`id: wave5/wave6: PID`)
3. target 문항이 기록됐는가?
4. `status: verified` 인 예측변인이 하나라도 있는가?

**하나라도 아니면 🛑 사유를 출력하고 멈춘다.** 아래 셀을 지금 실행하면 — variables.yaml
을 아직 안 채웠다면 — 멈추는 모습을 보게 된다. **그게 정상이고, 그게 설계다.**
검증(④·⑤)을 마친 뒤 다시 실행하면 ✅ 로 바뀐다."""),

code(r'''!python scripts/build_dataset.py \
  --wave5 "''' + W5 + r'''" \
  --wave6 "''' + W6 + r'''"'''),

code(r'''# 게이트가 열렸다면: 오늘의 산출물 확인 — 이 두 개가 생기면 2차시 완료다
import os
for f, why in {
    "configs/variables.yaml":            "사람이 채운 변수 매핑 (status: verified)",
    "data/processed/modeling_frame.parquet": "응답자 1명 = 1행 모델링 데이터셋",
    "reports/data_quality.md":           "ID·병합·결측·범위 품질 보고서",
}.items():
    print(f"  {'✅' if os.path.exists(f) else '⬜'} {f:44s} {why}")
print("\n⬜ 가 남아 있으면 → variables.yaml 검증을 마치고 위 셀을 다시 실행한다.")
print("생성됐다면 reports/data_quality.md 를 열어 §2 병합(≈1,321명)과 §4 결측률부터 읽는다.")'''),

md("""## 🎯 회고 (5분)

1. 역채점을 빠뜨린 채 평균을 내면 무슨 일이 생기는가 — **친구에게 설명한다면**?
2. `ID`(가구)로 join 하면 안 되는 이유는? 에러가 안 나는데 왜 더 위험한가?
3. 게이트(`codebook_verified`)는 왜 **사람만** 열 수 있게 설계했을까?
   코드가 자동으로 열면 무엇이 무너지는가?

## ▶️ 다음 (3차시)
> "오늘 변수의 **자리**를 확정했다. 다음엔 그 변수들의 **생김새**를 본다 —
> 분포·상관, 그리고 **Cronbach α 로 우리 척도 점수를 믿어도 되는지** 확인한다.
> 선행연구 α(5차 .76 / 6차 .74)가 재현되는지가 첫 관문이다 — 재현이 안 되면
> 오늘 판단한 역채점부터 의심한다."""),
]

os.makedirs("session2", exist_ok=True)
save(cells, "session2/session2.ipynb")

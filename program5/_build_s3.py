# -*- coding: utf-8 -*-
"""session3.ipynb 빌더 — 척도의 신뢰도와 분포를 확인하는 EDA 차시.

3차시는 2차시의 판단이 **시험대에 오르는** 차시다.
Cronbach α 와 문항-전체 상관이 역채점 판단을 교차 검증하고,
분포·상관이 4~6차시(불균형·표준화·다중공선성)의 재료를 미리 깐다.

실측 근거 (참여자 기준 · 5차 1,347 / 6차 1,329 / 둘 다 1,321):
  target α .757 (선행 .74) · prev α .762 (선행 .76)  → 재현 확인
  s_accul_str_10 r_it .044 / alpha-if-deleted .845   → 역채점 아님, 겉도는 문항
  parenting_neglect .568→.815 · school_adjustment .489→.785 · peer_relationship .543→.626
  target mean 1.415 · SD .325 · skew 1.39 · q75=1.50 → 양성 33.8% (동점)
"""
import os

from nb import md, code, save, SETUP, handoff_in, handoff_out

W5 = "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 5차년도.csv"
W6 = "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 6차년도.csv"

cells = [
md("""# 3차시 — 이 점수를 믿어도 되는가

### 신뢰도·분포·상관 (EDA) · Cronbach α / 집계 · 시각화 · 데이터 클리닝

> **오늘 한 문장:** "2차시에 변수의 **자리**를 확정했다. 오늘은 그 변수의 **생김새**를 본다 —
> 그리고 지난주 우리가 내린 **역채점 판단이 옳았는지 데이터가 채점한다.**"

오늘의 목표 4가지:

1. **Cronbach α** 를 계산하고, 선행연구 α(5차 .76 / 6차 .74)가 **재현되는지** 확인한다.
2. **문항-전체 상관**으로 지난주 놓친 **역채점 문항을 찾아내고 교정**한다. ← 첫 봉우리
3. **분포**(평균·SD·왜도·천장/바닥)와 **상관**을 읽고, 변수의 성격을 설명한다.
4. target 분포에서 **동점(ties)** 문제를 발견해 4차시 조작적 정의로 넘긴다. ← 두 번째 봉우리

> 💡 운영 방식은 1·2차시와 같다: 셀을 위에서 아래로. 코드는 **전부 채워져 있다** —
> 실행 전에 결과를 예측하게 하고, `# CHECK` 에서 `✅` 를 확인한 뒤 넘어간다.
> 오늘도 마지막엔 **에디터로 돌아가 `variables.yaml` 을 고친다** — 검증은 계속되는 상태다."""),

md("""## 🗺️ 오늘의 위치 — 3차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1 ✅ | 문화적응 스트레스 · 예측 vs 인과 · 연구윤리 | feature/target · classification |
| 2 ✅ | 심리척도 · 문항 · 역채점 | pandas · 결측치 · ID join |
| **3 (오늘)** | **평균 · SD · 분포 · 상관 · Cronbach α** | **집계 · 시각화 · 데이터 클리닝** |
| 4 | 고스트레스 집단의 조작적 정의 | split · 클래스 불균형 · **데이터 누출** |
| 5~8 | 관계의 방향 → 해석 → 보고 | 로지스틱 → 트리 → 중요도 → 재현성 |

**오늘의 재료** — 2차시의 산출물 그 자체다.

- `configs/variables.yaml` — 지난주 **사람이 검증한** 변수 매핑 (오늘 또 고친다)
- 5·6차 CSV 원자료 — α 는 **문항 단위** 계산이라 원자료가 필요하다
- (있다면) `data/processed/modeling_frame.parquet` · `reports/data_quality.md`

> 🔴 오늘의 규칙: **"지표가 좋아졌다"와 "옳아졌다"는 다른 말이다.**
> α 를 올리려고 문항을 빼는 순간 척도가 달라진다 — 오늘 그 유혹을 한 번 만난다."""),

md("""## Step 0 — 재료 확인: 게이트가 열려 있는가"""),
code('!pip install pandas pyyaml matplotlib pyarrow -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(handoff_in(pull=['configs/variables.yaml', 'data/processed/modeling_frame.parquet', 'reports/data_quality.md'], require=['configs/variables.yaml'], hint="2차시에서 사람이 채운 variables.yaml 이 있어야 오늘 척도 점수를 만들 수 있다")),

# ── 2차시 산출물 반입 셀 ─────────────────────────────────────────────
# 왜 넣나: 2차시 검증은 '에디터에서 사람이' 한 일이다. 그 결과 파일이 이 런타임에
# 없으면(드라이브 zip 이 검증 전 버전이거나, 런타임이 초기화됐거나) 3차시는 시작을
# 못 한다. 이 셀은 검증을 대신해 주지 않는다 — 내가 채운 파일을 **옮겨올 뿐**이다.
# 그래서 업로드본도 게이트 검사를 똑같이 통과해야 반영된다.
code(r'''# 2차시에 채운 configs/variables.yaml 을 이 런타임으로 가져온다 (게이트가 닫혀 있을 때만)
import glob, os, shutil
from maps_risk import config

TARGET = "configs/variables.yaml"

def gate_state(path=TARGET):
    """(열렸나, 닫힌 사유들) — is_ready_for_scoring 을 파일 하나에 대해 돌린다."""
    try:
        return config.is_ready_for_scoring(config.load_yaml(path))
    except Exception as e:
        return False, [f"읽지 못했다: {e}"]

def adopt(src):
    """검증을 통과한 파일만 configs/ 에 반영한다. 기존 파일은 백업해 둔다."""
    ok, why = gate_state(src)
    if not ok:
        print(f"🛑 {src} 는 아직 게이트를 못 연다 — 반영하지 않았다:")
        for r in why:
            print("   -", r)
        return False
    if os.path.exists(TARGET):
        shutil.copy(TARGET, "configs/variables_before_upload.yaml")
    if os.path.abspath(src) != os.path.abspath(TARGET):
        shutil.copy(src, TARGET)
    n = len(config.verified_constructs(config.load_yaml(TARGET)))
    print(f"✅ {src} → {TARGET} 반영 완료 (검증된 예측변인 {n}개)")
    return True

ok, why = gate_state()
if ok:
    print("✅ configs/variables.yaml 이 이미 검증본이다 — 업로드할 필요가 없다.")
else:
    print("🛑 지금 있는 variables.yaml 은 게이트가 닫혀 있다:")
    for r in why:
        print("   -", r)

    # ① 드라이브에서 먼저 찾는다 — 2차시 마지막 셀이 program5_state/ 에 저장해 두었다면 그걸 쓴다
    found = [f for pat in ["/content/drive/MyDrive/program5_state/configs/variables.yaml",
                           "/content/drive/MyDrive/variables.yaml",
                           "/content/drive/MyDrive/*/variables.yaml",
                           "/content/drive/MyDrive/*/*/variables.yaml"]
             for f in sorted(glob.glob(pat))]
    if found:
        print("\n드라이브에서 찾음:", found[0])
        ok = adopt(found[0])

    # ② 없으면 내 컴퓨터에서 직접 올린다 (Colab 전용)
    if not ok:
        try:
            from google.colab import files
            print("\n📤 2차시에 채운 variables.yaml 을 선택하라 (취소하려면 그냥 닫는다)")
            up = files.upload()
            for name in up:
                if adopt(name):
                    break
        except ImportError:
            print("\n로컬 환경이다 — 에디터에서 configs/variables.yaml 을 채우고 이 셀을 다시 실행하라.")
'''),
code(r'''# 2차시에서 사람이 검증한 것만 오늘 분석에 들어온다 (미검증은 조용히 섞이지 않는다)
import pandas as pd
from maps_risk.config import load_configs, verified_constructs, unverified_constructs

variables, modeling = load_configs("configs")
gate = variables["meta"].get("codebook_verified")

V   = verified_constructs(variables, "predictors")            # 5차 예측변인
OPT = verified_constructs(variables, "optional_predictors")   # Model B 전용
TGT = variables["target"]
pending = unverified_constructs(variables, "predictors")

print(f"게이트(codebook_verified): {'✅ 열림' if gate else '🛑 닫힘'}")
print(f"검증된 예측변인 {len(V)}개: {', '.join(V) or '(없음)'}")
print(f"Model B 전용 {len(OPT)}개: {', '.join(OPT) or '(없음)'}")
print(f"target 문항 {len(TGT.get('items') or [])}개 · status={TGT.get('status')}")
if pending:
    print(f"\n⏳ 아직 미검증 {len(pending)}개: {', '.join(pending)}")
    print("   → 오늘 분석에서 제외된다. 제안본으로 대신 채우지 않는다. 미검증은 미검증이다.")
if not (gate and TGT.get("items")):
    print("\n🛑 2차시가 아직 안 끝났다 — configs/variables.yaml 검증을 마치고 다시 시작한다.")'''),

code(r'''# 원자료를 읽고 '그 해 실제 참여자'만 남긴다 (2차시 Step 6 의 결론)
w5 = pd.read_csv("''' + W5 + r'''", na_values=["", " "], low_memory=False)
w6 = pd.read_csv("''' + W6 + r'''", na_values=["", " "], low_memory=False)
p5 = w5[w5["SURVEY1_w5"] == 1]     # 5차 참여 1,347명
p6 = w6[w6["SURVEY1_w6"] == 1]     # 6차 참여 1,329명

print(f"5차 참여 {len(p5)}명 · 6차 참여 {len(p6)}명")
print("※ α 는 척도 점수가 아니라 **문항 단위**로 계산한다 → 원자료가 필요하다.")'''),

md("""## Step 1 — 신뢰도: 이 문항들은 한 팀인가 🧠

2차시에 우리는 10개의 문항을 평균 내 하나의 점수로 만들었다. 그런데 —

> **"이 10개가 정말 같은 것을 재고 있는가?"**

몸무게를 세 번 재서 평균 내는 것이 의미 있는 이유는 세 번 다 **몸무게**를 쟀기 때문이다.
만약 두 번은 몸무게, 한 번은 키를 쟀다면 그 평균은 아무 뜻이 없다. 문항도 똑같다.

**Cronbach's α(크론바흐 알파)** 는 "문항들이 서로 얼마나 같이 움직이는가" — **내적 일관성
(internal consistency)** — 을 0~1 로 요약한 값이다. 보통 **.70 이상**이면 수용 가능하다고 본다.

> ⚠️ α 는 **일관성**이지 **타당도**가 아니다. 열 문항이 사이좋게 같은 것을 재고 있어도,
> 그게 우리가 재려던 "문화적응 스트레스"라는 보장은 α 가 해 주지 않는다.

**오늘의 첫 관문**: 선행연구가 보고한 α 는 **5차 .76 / 6차 .74** 다.
우리 계산이 그 근처로 나오면 — 2차시의 컬럼 선택과 채점 규칙이 **옳았다는 강력한 증거**다.
안 나오면? 우리가 어딘가 틀렸다는 뜻이다."""),

code(r'''# 우리 모듈로 target(6차 문화적응 스트레스 10문항)의 α 를 계산한다
from maps_risk import scoring

alpha_w6 = scoring.cronbach_alpha(p6, TGT["items"])
alpha_w5 = scoring.cronbach_alpha(p5, OPT["previous_acculturative_stress"]["items"])

print(f"6차 문화적응 스트레스 α = {alpha_w6:.3f}   (선행연구 .74)")
print(f"5차 문화적응 스트레스 α = {alpha_w5:.3f}   (선행연구 .76)")'''),
code(r'''# CHECK Step1 — 재현 판정
try:
    assert abs(alpha_w6 - 0.74) < 0.05, f"6차 α 가 선행연구(.74)와 너무 다르다: {alpha_w6:.3f}"
    assert abs(alpha_w5 - 0.76) < 0.05, f"5차 α 가 선행연구(.76)와 너무 다르다: {alpha_w5:.3f}"
    print("✅ PASS — 선행연구 α 가 재현됐다. 2차시의 컬럼 선택·채점 규칙이 옳았다는 증거다.")
    print("   재현(replication)은 '남이 한 걸 따라 해서 같은 숫자가 나오는 것' — 오늘의 첫 관문 통과.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 함수 이름은 cronbach_alpha. 그래도 다르면 items 나 참여 필터를 의심하라.")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

```python
alpha_w6 = scoring.cronbach_alpha(p6, TGT["items"])
alpha_w5 = scoring.cronbach_alpha(p5, OPT["previous_acculturative_stress"]["items"])
```

실측: **6차 .757 / 5차 .762** — 선행연구(.74 / .76)와 사실상 일치한다.

이 한 줄이 뜻하는 것: 우리가 고른 컬럼 10개가 선행연구가 쓴 그 10개이고,
채점 방식(평균·정방향)도 같다는 뜻이다. **컬럼명을 추측했다면 절대 이 숫자가 안 나온다.**
</details>"""),

md("""## Step 2 — 문항-전체 상관: 역채점 실수를 잡아내는 그물 ⚠️ (첫 봉우리)

2차시에 약속했다 — "역채점을 틀리면 α 가 떨어진다. 다음 주에 시험대에 오른다."
그런데 α 는 척도 **전체**의 점수라, 낮게 나와도 **어느 문항 탓인지** 알려주지 않는다.
범인을 지목하는 도구가 **문항-전체 상관(corrected item-total correlation)** 이다.

```
각 문항  vs  그 문항을 뺀 나머지 문항들의 평균  →  피어슨 상관
```

읽는 법이 오늘의 핵심 기술이다:

| 값 | 뜻 | 처방 |
|---|---|---|
| **양수(≈ .3 이상)** | 나머지와 같은 방향으로 움직인다 | 정상 |
| **음수(−)** | 방향이 **반대**다 | **역채점 누락** — 뒤집으면 해결된다 |
| **0 근처** | 방향 문제가 아니라 **다른 걸 재고 있다** | 역채점해도 **안 좋아진다** |

> 🔴 마지막 줄이 오늘 배울 가장 미묘한 구분이다. 음수와 0은 완전히 다른 병이다."""),

code(r'''# 먼저 target 10문항을 눈으로 본다 (실행하고 표를 읽는다)
r_it = scoring.item_total_correlations(p6, TGT["items"])
print("6차 문화적응 스트레스 — 문항-전체 상관\n")
for c, r in r_it.items():
    bar = "█" * max(0, int(r * 30))
    print(f"  {c:22s} {r:+.3f}  {bar}")
print(f"\n  전체 α = {alpha_w6:.3f}")'''),

md("""### Step 2 실습 — 전 척도를 스캔한다

target 은 괜찮아 보인다(10번만 빼고 — 그 얘기는 Step 3 에서). 그런데 2차시에 우리가
문항 텍스트를 **눈으로 읽은 척도는 target 하나뿐**이었다. 나머지 15개 척도는?

그래서 오늘, 전 척도를 한 번에 스캔한다."""),

code(r'''# '역채점을 놓쳤다'고 의심하는 조건은 r_it < 0 이다 (표의 두 번째 줄)
suspects = {}
for name, spec in V.items():
    r = scoring.item_total_correlations(p5, spec["items"])
    bad = [c for c in r.index if r[c] < 0]             # 음수 = 나머지와 반대로 움직인다
    a = scoring.cronbach_alpha(p5, spec["items"])
    flag = "⚠️" if bad else "  "
    print(f"{flag} {name:24s} k={len(spec['items']):2d}  α={a:.3f}  최저 r_it={r.min():+.3f} ({r.idxmin()})")
    if bad:
        suspects[name] = bad

print("\n의심 문항:", suspects)'''),
code(r'''# CHECK Step2
EXPECTED = {
    "parenting_neglect": ["parenting_b06_w5", "parenting_b07_w5"],
    "school_adjustment": ["learning_a05_w5"],
    "peer_relationship": ["fr_rela_a04_w5"],
}
try:
    checked = {k: v for k, v in EXPECTED.items() if k in V}
    assert checked, "검증된 구성개념이 없어 채점할 수 없다 — Step 0 으로 돌아가라"
    for k, want in checked.items():
        assert sorted(suspects.get(k, [])) == sorted(want), \
            f"{k}: {want} 가 걸려야 하는데 {suspects.get(k, [])} 가 걸렸다"
    print(f"✅ PASS — 검증한 {len(checked)}개 척도에서 역채점 누락 문항을 전부 찾아냈다.")
    print("   조건은 r < 0 이다. 방향이 반대인 문항은 나머지와 '반대로' 움직인다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 표의 두 번째 줄 — 음수(−)가 방향이 반대라는 신호다. r[c] < 0")'''),
md("""<details><summary>💡 해설 — 그리고 문항 텍스트를 읽어 보라</summary>

```python
bad = [c for c in r.index if r[c] < 0]
```

걸린 문항들의 **실제 텍스트**다 (체크리스트에서):

| 문항 | 텍스트 | 속한 척도 |
|---|---|---|
| `parenting_b06` | 부모님은 내가 학교에서 어떻게 생활하는지 **관심을 갖고 물어보신다** | 방임 |
| `parenting_b07` | 부모님은 내 몸이나 옷, 이불 등이 깨끗하도록 **항상 신경쓰신다** | 방임 |
| `learning_a05` | 나는 공부시간에 **딴 짓을 한다** | 학교적응(학습활동) |
| `fr_rela_a04` | 나는 친구가 하는 일을 **방해한다** | 교우관계 |

읽어 보면 너무 명백하다 — "**방임**" 척도에 "관심을 갖고 물어보신다"라니.
**우리는 지난주에 이걸 놓쳤다.** 눈으로 읽은 건 target 10문항뿐이었으니까.

그리고 오늘 그 실수를 **데이터가 잡아냈다.** 이것이 2차시에 약속한 "α 의 교차 검증"이다.
</details>"""),

code(r'''# 역채점을 적용하면 α 가 어떻게 변하나 — 교정 전/후 비교
rows = []
for name, bad in suspects.items():
    spec = V[name]
    lo, hi = spec["expected_range"]
    fixed = p5.copy()
    for c in bad:
        fixed[c] = scoring.reverse_code(fixed[c].astype(float), lo, hi)
    rows.append({"구성개념": name,
                 "역채점 문항": ", ".join(bad),
                 "α 교정 전": round(scoring.cronbach_alpha(p5, spec["items"]), 3),
                 "α 교정 후": round(scoring.cronbach_alpha(fixed, spec["items"]), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n→ 방향만 되돌렸을 뿐인데 척도가 살아났다. 문항을 뺀 게 아니라 '제자리로' 돌린 것이다.")'''),

md("""### ✍️ 지금, 에디터로 돌아간다 (사람이 하는 일 ①)

찾아낸 역채점 문항을 `configs/variables.yaml` 의 해당 구성개념 `reverse_items` 에 적는다.

```yaml
parenting_neglect:
  status: verified
  expected_range: [1, 4]
  reverse_items:
    - parenting_b06_w5      # "관심을 갖고 물어보신다" — 방임과 방향 반대
    - parenting_b07_w5      # "항상 신경쓰신다"
```

> 2차시 마지막 문장을 기억하라 — **"검증은 한 번의 행사가 아니라 계속되는 상태다."**
> 지난주에 `verified` 라고 적었던 것을 오늘 고친다. 그게 잘못이 아니라 **그게 과학**이다.
> 새 근거가 나오면 판단을 갱신하고, 갱신했다는 사실을 기록한다."""),

md("""## Step 3 — 10번 문항의 재판: 우리 판단이 틀렸을 때 🔍

2차시에 학생 전원이 이렇게 답했다:

> `s_accul_str_10` "외국인 부모님 나라보다 한국에서 **더 잘 살 수 있을 것이다**" → **역채점 후보!**

논리는 완벽했다. 스트레스 척도인데 이 문항만 긍정 방향이니까. 그런데 —
Step 2 의 표를 다시 보라. 10번의 문항-전체 상관은 **음수가 아니라 0 근처**였다.

**음수와 0 근처는 다른 병이다.** 오늘 배운 그 구분이 지금 필요하다."""),

code(r'''# 10번을 역채점하면 α 는 오를까 내릴까?
# 🖐 실행하기 전에 각자 예측해 보게 한 뒤 셀을 돌린다 — 대부분 "오른다"에 손을 든다.
예측 = "내린다"        # 실측 결과다. 왜 그런지가 오늘의 두 번째 봉우리

item10 = TGT["items"][9]
flipped = p6.copy()
flipped[item10] = scoring.reverse_code(flipped[item10].astype(float), 1, 4)

alpha_flipped = scoring.cronbach_alpha(flipped, TGT["items"])
print(f"그대로            α = {alpha_w6:.3f}")
print(f"10번 역채점 후    α = {alpha_flipped:.3f}   ← {'올랐다' if alpha_flipped > alpha_w6 else '내렸다'}")'''),
code(r'''# CHECK Step3
try:
    assert alpha_flipped < alpha_w6, "역채점하면 α 가 내려가야 한다 (실측 .757 → .738)"
    assert 예측 == "내린다", f"예측이 '{예측}' 인데, 실제로는 내려갔다 — 왜 그럴까?"
    print("✅ PASS — 역채점하면 α 가 오히려 **내려간다.**")
    print("   → 10번은 '방향이 반대인 문항'이 아니다. 뒤집어도 나아지지 않는다.")
    print("   → r_it 가 0 근처라는 건 '반대'가 아니라 '**따로 논다**'는 뜻이었다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 방향이 진짜 반대라면 뒤집었을 때 α 가 올라야 한다. 안 오른다면?")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

`예측 = "내린다"` — 실측 **.757 → .738**.

**왜인가.** 역채점은 "방향이 뒤집힌 문항"을 되돌리는 수술이다. 10번의 r_it 는 `+0.044` —
음수가 아니라 **거의 0**이다. 나머지 9문항과 **아무 방향으로도 같이 움직이지 않는다.**
0을 뒤집으면 −0.04 가 될 뿐, 척도에 기여하지 않기는 마찬가지다.

**심리학적으로 읽으면**: 10번은 스트레스가 아니라 **미래 전망·기대**를 재고 있을 가능성이 크다.
"한국에서 더 잘 살 수 있을 것이다"에 동의하는 것과, 지금 차별로 스트레스를 받는 것은
**서로 배타적이지 않다.** 둘 다 참일 수 있다 — 그래서 상관이 0 근처인 것이다.

**그리고 2차시의 우리 판단은 틀렸다.** 다만 2차시 노트북은 이렇게 적어 두었다 —
"노트북의 판단은 **후보**일 뿐이다." 후보를 데이터로 기각하는 것, 그게 오늘 한 일이다.
</details>"""),

code(r'''# 그럼 10번을 빼면? — alpha-if-deleted: 문항을 하나씩 빼 보며 α 를 다시 잰다
aid = scoring.alpha_if_deleted(p6, TGT["items"])
print("문항을 하나 뺐을 때의 α  (전체 α = %.3f)\n" % alpha_w6)
for c, a in aid.sort_values(ascending=False).items():
    mark = "  ← 빼면 크게 오른다" if a > alpha_w6 + 0.02 else ""
    print(f"  {c:22s} → α={a:.3f}{mark}")'''),

md("""### 그래서 뺄 것인가? — **빼지 않는다** 🔴

10번을 빼면 α 가 **.757 → .845** 로 뛴다. 유혹적이다. 그런데 빼지 않는다. 이유 셋:

1. **척도가 달라진다.** 선행연구는 **10문항**으로 α .74 를 보고했고, 우리도 10문항으로 .757 —
   이미 **재현에 성공**했다. 9문항으로 바꾸는 순간 그 비교가 무너진다.
   (9문항 점수와 10문항 점수의 상관은 r = .948 — 바꿔도 순위는 거의 그대로다. 얻는 것도 적다.)
2. **α 는 목표가 아니라 진단이다.** α 를 올리려고 문항을 빼는 것은
   체온계 눈금을 손으로 돌려 열을 내리는 것과 같다. 지표가 좋아졌지 몸이 나아진 게 아니다.
3. **사후에 데이터를 보고 척도를 바꾸는 것** 자체가 위험하다. 이 표본에서만 잘 맞는
   척도를 만들게 된다 — 8차시에 다룰 **과적합**이 척도 수준에서 벌어지는 것이다.

**대신 기록한다.** `variables.yaml` 의 note 와 8차시 한계 절에:

> "`s_accul_str_10` 은 문항-전체 상관이 .04 로 낮아 다른 구성개념(미래 전망)을 반영할
> 가능성이 있다. 선행연구와의 비교 가능성을 위해 10문항을 유지했으며,
> **9문항 척도로 바꾸는 민감도 분석을 한계에 기록한다.**"

> 오늘의 문장: **"고칠 수 있는 문제"와 "기록해야 하는 문제"를 구분하는 것도 실력이다.**"""),

md("""## Step 4 — 분포: 이 변수는 어떻게 생겼나

이제 문항을 떠나 **척도 점수**로 올라간다. 각 구성개념 점수의 **평균 · 표준편차(SD) · 왜도**를
보는 이유는 하나다 — **평균만 보면 속는다.**

- **표준편차(SD)**: 흩어진 정도. SD 가 0 에 가까우면 **모두가 같은 답**을 했다는 뜻이고,
  그런 변수는 누구도 구분하지 못한다 → 예측에 아무 쓸모가 없다.
- **왜도(skewness)**: 분포가 한쪽으로 쏠린 정도. 0 이면 좌우 대칭,
  **양수면 왼쪽에 몰리고 오른쪽 꼬리가 길다**(대부분 낮고 소수만 높다)."""),

code(r'''# 척도 점수를 만든다 — 방금 찾은 역채점을 반영해서 (2차시 파이프라인과 같은 함수)
def build_scores(df, specs, extra_reverse=None):
    """구성개념 정의 → 응답자별 척도 점수 DataFrame."""
    extra_reverse = extra_reverse or {}
    out = {}
    for name, spec in specs.items():
        rev = list(spec.get("reverse_items") or []) + list(extra_reverse.get(name, []))
        out[name] = scoring.scale_score(
            df, spec["items"], reverse_items=rev,
            scale_range=spec.get("expected_range"), method="mean")
    return pd.DataFrame(out, index=df.index)

x5 = build_scores(p5, {**V, **OPT}, extra_reverse=suspects)
x5.insert(0, "PID", p5["PID"].values)

y6 = pd.DataFrame({"PID": p6["PID"].values,
                   "acculturative_stress_w6": scoring.scale_score(
                       p6, TGT["items"], reverse_items=TGT.get("reverse_items") or [],
                       scale_range=TGT["expected_range"], method="mean",
                       min_valid_items=(TGT.get("scoring") or {}).get("min_valid_items")).values})

f = x5.merge(y6, on="PID", how="inner")     # 2차시의 PID join — 두 해 모두 참여한 사람
print(f"분석 frame: {len(f)}행 × {f.shape[1]}열   (2차시에 손으로 센 1,321명과 맞는가?)")

desc = pd.DataFrame({
    "n":    f.drop(columns="PID").notna().sum(),
    "평균":  f.drop(columns="PID").mean().round(3),
    "SD":   f.drop(columns="PID").std().round(3),
    "최소":  f.drop(columns="PID").min().round(2),
    "최대":  f.drop(columns="PID").max().round(2),
    "왜도":  f.drop(columns="PID").skew().round(2),
}).sort_values("왜도", ascending=False)
print("\n", desc.to_string())'''),

code(r'''# 그림 4장으로 분포를 본다 (reports/figures/ 에 저장 — 발표자료에 그대로 쓴다)
import matplotlib.pyplot as plt
from maps_risk import plots          # import 만 해도 한글 폰트가 잡힌다
import os; os.makedirs("reports/figures", exist_ok=True)

show = ["acculturative_stress_w6", "bullying", "korean_proficiency", "peer_support"]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
for ax, c in zip(axes, show):
    if c not in f.columns:
        ax.set_visible(False); continue
    ax.hist(f[c].dropna(), bins=20)
    ax.set_title(f"{c}\n평균 {f[c].mean():.2f} · 왜도 {f[c].skew():.2f}", fontsize=10)
fig.tight_layout(); fig.savefig("reports/figures/eda_distributions.png", dpi=150)
plt.show()
print("✅ reports/figures/eda_distributions.png")'''),

md("""### Step 4 해석 — 그림 4장이 각각 하는 말

| 변수 | 실측 | 읽는 법 | 어느 차시로 이어지나 |
|---|---|---|---|
| **문화적응 스트레스(6차)** | 평균 1.42 · 왜도 **+1.39** | 대부분 1점대에 몰려 있다. 스트레스가 높은 학생은 **소수**다 | 4차시 **클래스 불균형** |
| **집단괴롭힘** | 평균 1.04 · 왜도 **+7.9** | 거의 전원이 "없었다(1)". 사실상 **상수에 가깝다** — 아무도 구분 못 한다 | 4차시 **feature 선별** |
| **한국어 실력** | 왜도 **−1.61** | 위쪽에 몰렸다 = **천장 효과(ceiling effect)**. 잘하는 학생들 사이의 차이가 안 보인다 | 7차시 해석의 한계 |
| **친구지지** | 평균 **4.13** | 다른 변수보다 값이 크다 — **5점 척도**이기 때문이다 (대부분은 4점) | 5차시 **표준화** |

> 마지막 줄이 중요하다. "친구지지 4.13 > 우울 1.69" 를 보고 **"친구지지가 더 높다"고
> 말하면 안 된다.** 자를 다르게 쓰고 잰 길이를 그대로 비교한 셈이다.
> 5차시에 **표준화(standardization)** 로 자를 통일한 다음에야 비교가 가능해진다."""),

md("""## Step 5 — 상관: 무엇이 무엇과 같이 움직이나

**상관계수(correlation, r)** 는 −1 ~ +1 사이의 값으로 두 변수가 **같이 움직이는 정도**를 말한다.

- `r > 0`: 하나가 크면 다른 것도 크다  ·  `r < 0`: 하나가 크면 다른 것은 작다
- 크기 감각(심리학 관례): **.10 작다 · .30 중간 · .50 크다**

먼저 심리학자로서 **예측**해 보자. 우울이 높은 학생은 1년 뒤 스트레스가 높을까 낮을까?
친구지지가 두터운 학생은?"""),

code(r'''# ▶ 부호는 이미 채워져 있다 — 🖐 실행 전에 각자 예측해 보게 한 뒤 셀을 돌린다
예측_우울    = "+"      # 5차 우울 ↔ 6차 문화적응 스트레스 (위험요인 → 양의 상관)
예측_친구지지 = "-"      # 5차 친구지지 ↔ 6차 문화적응 스트레스 (보호요인 → 음의 상관)

target_col = "acculturative_stress_w6"
corr = f.drop(columns="PID").corr()[target_col].drop(target_col)
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

print("6차 문화적응 스트레스와의 상관 (절댓값 큰 순)\n")
for c, r in corr.items():
    print(f"  {c:32s} r = {r:+.3f}  {'█' * int(abs(r) * 60)}")'''),
code(r'''# CHECK Step5
try:
    assert 예측_우울 == "+", "우울이 높으면 이후 스트레스도 높다 → 양(+)의 상관"
    assert 예측_친구지지 == "-", "지지가 두터우면 스트레스는 낮다 → 음(−)의 상관 (보호요인)"
    assert corr.abs().max() < 0.40, "상관이 .40 을 넘는 게 있다 — 누출이 아닌지 확인하라"
    print("✅ PASS — 부호는 1차시의 위험요인/보호요인 그림과 정확히 일치한다.")
    print(f"   그런데 가장 큰 상관도 |r| = {corr.abs().max():.3f} 다. 전부 '작다~중간' 구간이다.")
    print("   → 1년 뒤를 맞히는 일은 쉽지 않다. 6차시 성능 숫자를 볼 때 이 장면을 기억하라.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 1차시의 위험요인(risk factor) / 보호요인(protective factor) 그림.")'''),

code(r'''# 예측변인끼리의 상관도 본다 — 여기에 6차시의 함정이 숨어 있다
import numpy as np
cm = f.drop(columns=["PID"]).corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm)), cm.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(cm)), cm.index, fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("척도 점수 상관행렬  (※ 상관은 인과가 아니다)")
fig.tight_layout(); fig.savefig("reports/figures/eda_correlation.png", dpi=150)
plt.show()

pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))
           .stack().sort_values(key=abs, ascending=False).head(5))
print("가장 강하게 얽힌 예측변인 쌍 5개:")
for (a, b), r in pairs.items():
    print(f"  {a:26s} ↔ {b:26s} r = {r:+.3f}")'''),

md("""### Step 5 해석 — 세 가지를 읽는다

1. **가장 강한 예측변인은 `previous_acculturative_stress`(r ≈ .31)** — 1년 전의 스트레스다.
   RQ3("이전 스트레스를 추가하면 얼마나 나아지나")의 답이 여기서 이미 어른거린다.
   동시에 **.31 밖에 안 된다**는 것도 중요하다 — 중2 때 힘들었던 학생이 중3 때도 힘들 확률은
   생각보다 **덜** 고정적이다. 변할 여지가 있다는 뜻이고, 그게 개입 연구의 근거가 된다.

2. **예측변인끼리 강하게 얽혀 있다** — 우울 ↔ 삶의만족도 **r = −.66**,
   자아탄력성 ↔ 자아존중감 **.58**, 친구지지 ↔ 교사지지 **.55**.
   서로 겹치는 정보를 가진 변수를 한 모델에 같이 넣으면 **다중공선성(multicollinearity)** 이
   생겨 개별 계수가 불안정해진다 → 5·7차시에서 "왜 우울의 계수가 작게 나왔지?"를 만난다.

3. **상관은 인과가 아니다.** `depression ↔ stress` 가 +.24 라고 해서
   "우울이 스트레스를 **일으킨다**"고 말할 수 없다. 반대 방향일 수도, 제3의 원인
   (예: 학교 분위기)이 둘 다 만들었을 수도 있다. 우리가 할 수 있는 말은
   **"같이 움직인다 / 예측에 기여한다"** 까지다."""),

md("""## Step 6 — target 분포와 동점: 25%가 25%가 아니다 🔍 (두 번째 봉우리)

4차시에 **고스트레스 집단**을 이렇게 정의할 예정이다:

> 학습 데이터 스트레스 점수의 **상위 25%**(75 백분위수 이상)를 `high_stress = 1` 로 한다.

깔끔해 보인다. 그런데 오늘 분포를 봤으니 한 번 계산해 보자. 진짜 25%가 될까?

> ⚠️ **오늘은 동점 현상을 보려고 전체 표본에서 계산한다.** 실제 cutoff 는 4차시에
> **train 데이터에서만** 계산한다 — 전체 분포로 정하면 그것 자체가 test 정보 누출이다.
> 오늘 보려는 건 cutoff 값이 아니라 **점수가 이산적이라 생기는 현상**이다."""),

code(r'''# 75 백분위수를 cutoff 로 잡고, 실제로 몇 %가 고스트레스가 되는지 센다
s = f[target_col]
cutoff = s.quantile(0.75)                   # 상위 25% → 75 백분위수
n_pos = (s >= cutoff).sum()

print(f"cutoff = {cutoff:.3f}")
print(f"고스트레스 = {n_pos}명 / {s.notna().sum()}명 = {n_pos / s.notna().sum():.1%}")
print("\n점수별 인원수 (동점자가 얼마나 몰려 있나):")
print(s.round(1).value_counts().sort_index().head(8).to_string())'''),
code(r'''# CHECK Step6
try:
    assert abs(cutoff - 1.5) < 0.01, f"75 백분위수는 1.50 이어야 한다 (지금 {cutoff})"
    rate = n_pos / s.notna().sum()
    assert rate > 0.30, f"25% 가 나왔다면 뭔가 다르다 — 실측은 33.8% 다 (지금 {rate:.1%})"
    print(f"✅ PASS — '상위 25%' 로 잘랐는데 실제로는 {rate:.1%} 가 됐다.")
    print("   이유: 점수가 문항 평균이라 값이 뚝뚝 끊긴다(1.4, 1.5, 1.6…).")
    print("   cutoff 1.50 에 **동점자 142명**이 몰려 있고, 이들이 통째로 넘어간다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 상위 25% = 아래에서 75% 지점 → quantile(0.75)")'''),
md("""<details><summary>💡 해설 — 그리고 이게 왜 중요한가</summary>

```python
cutoff = s.quantile(0.75)      # → 1.500
```

실측: **1,321명 중 447명 = 33.8%** 가 고스트레스가 된다. 25% 가 아니다.

**왜인가.** 10문항 평균은 연속처럼 보이지만 실제로는 **0.1 단위로 뚝뚝 끊긴 값**이다
(1.4 · 1.5 · 1.6 …). 1.50 점을 받은 사람만 **142명**이다. 백분위수는 이들을
쪼갤 수 없으니 **전부 같은 쪽으로** 보낸다.

더 재미있는 사실 — 분위수를 **0.70 으로 낮춰도 cutoff 는 똑같이 1.50**, 양성률도 33.8% 다.
0.80 으로 올려야 1.60 · 23.1% 로 바뀐다. **분위수를 조금 움직이면 아무 일도 안 일어나다가,
어느 순간 100명이 한꺼번에 넘어간다.**

**그래서 규칙**: 조작적 정의를 쓸 때는 분위수만 적지 말고 **실제 양성 비율을 항상 함께
보고한다.** "상위 25%로 정의했다"만 쓰면 읽는 사람은 25% 라고 믿는다 — 사실이 아닌데도.

그리고 다시 강조 — **여기서 구한 1.500 은 진짜 cutoff 가 아니다.** 우리는 전체 1,321명을
보고 계산했다. 4차시에는 **train 으로 쓸 사람들만 보고** 계산한다. 전체 분포로 선을 그으면
test 응답자의 정보가 선 긋기에 들어간 셈이 되고, 그게 **데이터 누출**이다.
4차시에서 이 문제를 정면으로 다룬다.
</details>"""),

md("""## Step 7 — 오늘의 결과를 파이프라인에 반영한다 (사람이 하는 일 ②)

Step 2 에서 찾은 역채점을 `variables.yaml` 에 적었다면, 이제 2차시의 그 명령을
**다시** 실행한다. 같은 명령인데 결과가 달라진다 — 척도가 교정됐기 때문이다."""),

code(r'''!python scripts/build_dataset.py \
  --wave5 "''' + W5 + r'''" \
  --wave6 "''' + W6 + r'''"'''),

code(r'''# 오늘의 산출물 확인
import os
for p, why in {
    "reports/figures/eda_distributions.png": "분포 4장 (불균형·상수·천장·척도범위)",
    "reports/figures/eda_correlation.png":   "상관행렬 (다중공선성 예고)",
    "data/processed/modeling_frame.parquet": "역채점 교정이 반영된 모델링 데이터셋",
    "reports/data_quality.md":               "품질 보고서 (§3 target 분포가 갱신된다)",
}.items():
    print(f"  {'✅' if os.path.exists(p) else '⬜'} {p:42s} {why}")

print("\n오늘 교정이 실제로 관계를 되살렸는지 확인 (역채점 전/후 target 상관):")
for name, bad in suspects.items():
    spec = V[name]
    raw = scoring.scale_score(p5, spec["items"], method="mean")
    fix = scoring.scale_score(p5, spec["items"], reverse_items=bad,
                              scale_range=spec["expected_range"], method="mean")
    t = pd.DataFrame({"PID": p5["PID"].values, "raw": raw.values, "fix": fix.values}).merge(y6, on="PID")
    print(f"  {name:24s} r {t['raw'].corr(t[target_col]):+.3f} → {t['fix'].corr(t[target_col]):+.3f}")
print("\n→ 역채점을 빠뜨리면 관계가 **희석돼서 안 보인다.** 없는 게 아니라 우리가 지운 것이다.")'''),

md("""## 💾 다음 차시를 위해 — 드라이브에 저장\n\n오늘 만든 것 중 **다음 차시가 재료로 쓰는 파일**을 내 드라이브(`program5_state/`)에 넣어 둔다.\n이렇게 해 두면 런타임이 끊겨도, 다른 컴퓨터에서 열어도 **다음 차시가 그냥 시작된다.**\n\n> 🔴 파생 파일이 들어가는 폴더다 — **개인 계정 안에만** 두고 링크 공유·양도하지 않는다."""),
code(handoff_out(push=['configs/variables.yaml', 'data/processed/modeling_frame.parquet', 'reports/data_quality.md', 'reports/figures/*.png'], note="3차시 산출물 — 역채점 교정본 variables.yaml 과 modeling_frame 을 4차시로")),

md("""## 🎯 회고 (5분)

1. 선행연구 α 가 재현됐다는 것은 **무엇에 대한 증거**인가? 재현이 안 됐다면 무엇부터 의심하나?
2. 문항-전체 상관이 **음수**인 것과 **0 근처**인 것은 어떻게 다른가 — 처방도 다른가?
3. 10번 문항을 빼면 α 가 .757 → .845 로 오른다. **그런데 왜 빼지 않았나?**

## 📝 과제
- 내가 맡은 구성개념의 α · 문항-전체 상관을 표로 정리 (교정 전/후)
- α 가 **.70 미만**인 척도가 있으면 그 이유를 문항 텍스트로 설명 (예: `peer_relationship` .626)
- `variables.yaml` 의 `reverse_items` 갱신본을 커밋

## ▶️ 다음 (4차시)
> "오늘 우리는 스트레스 점수의 **분포**를 봤고, 상위 25%가 실제로는 **33.8%** 라는 것도 봤다.
> 다음 주엔 그 선을 **어디에 어떻게 긋는지**, 그리고 **왜 그 선을 train 데이터에서만
> 계산해야 하는지**를 다룬다. 4차시의 백미는 **일부러 데이터 누출을 일으켜 보는 것**이다 —
> AUC 가 1.0 에 가까운 '완벽한 모델'을 만들고, 왜 그게 쓰레기인지 직접 설명하게 된다."""),
]

os.makedirs("session3", exist_ok=True)
save(cells, "session3/session3.ipynb")

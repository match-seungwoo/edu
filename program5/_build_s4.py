# -*- coding: utf-8 -*-
"""session4.ipynb 빌더 — 조작적 정의 · split · 클래스 불균형 · 데이터 누출.

4차시는 "성능 숫자를 의심하는 법"을 배우는 차시다.
좋아 보이는 두 숫자(Dummy 의 accuracy 66%, 누출 모델의 AUC 1.0)가 둘 다
쓸모없다는 것을 학생이 직접 만들어 보고 설명하게 한다.

★ 오늘 test 세트는 열지 않는다. 만들어 놓고 봉인한다 — 왜 안 여는지가 오늘 배울 것이다.
   모든 평가는 train 안에서 5-fold CV 로만 한다.

실측 근거 (frame 1,321행 · seed 42 · test_size .20):
  train 1,056 / test 265 · cutoff(train q75) = 1.500 · train 양성 33.7%(356명)
  Dummy      accuracy .663 · AUC .500 · recall .000
  로지스틱 A  accuracy .612 · AUC .653 · recall .615   ← 정확도가 더 낮은데 더 좋은 모델
  누출 모델   AUC 1.0000
  cutoff 누출: 전체 q75 = train q75 = 1.500 → 라벨 차이 0명 (이번엔 우연히 같았다)
  전처리 누출: 전체 fit .6526 vs Pipeline .6525 → 차이 .0001
  선택 누출(합성): 잡음 200개에서 미리 고르면 .5877, 폴드 안에서 고르면 .5025
"""
import os

from nb import md, code, save, SETUP

cells = [
md("""# 4차시 — 성능이 좋아 보이면 의심하라

### 조작적 정의 · train/test split · 클래스 불균형 · **데이터 누출**

> **오늘 한 문장:** "3차시에 스트레스 점수의 **분포**를 봤다. 오늘은 그 위에 **선을 긋고**,
> 데이터를 **둘로 나누고**, 그다음 — **일부러 부정행위를 저질러 본다.**"

오늘 두 개의 '좋아 보이는' 숫자를 만든다. 그리고 **둘 다 쓸모없다는 것**을 직접 설명하게 된다.

| 숫자 | 어떻게 나오나 | 왜 쓸모없나 |
|---|---|---|
| **정확도 66%** | 전원을 "고스트레스 아님"으로 찍는다 | 고스트레스를 **한 명도** 못 찾는다 |
| **AUC 1.0** | 6차 스트레스 점수로 6차 고스트레스를 맞힌다 | **답을 보고 답을 맞혔다** |

오늘의 목표 4가지:

1. **조작적 정의**가 무엇이고 무엇이 **아닌지**(임상 진단 아님) 설명한다.
2. **train/test split** 을 하고, **cutoff 를 train 에서만** 계산한다. ← 고비 1
3. **클래스 불균형**에서 accuracy 가 왜 거짓말을 하는지 보인다.
4. **데이터 누출**을 일부러 일으켜 AUC 1.0 을 만들고, 왜 쓰레기인지 설명한다. ← 고비 2

> 🔴 **오늘 test 세트는 열지 않는다.** 만들어 놓고 봉인한다.
> 왜 안 여는지가 오늘 배울 것 중 하나다. 모든 평가는 **train 안에서 5-fold CV** 로만 한다."""),

md("""## 🗺️ 오늘의 위치 — 4차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1 ✅ | 문화적응 스트레스 · 예측 vs 인과 | feature/target · classification |
| 2 ✅ | 심리척도 · 문항 · 역채점 | pandas · 결측치 · ID join |
| 3 ✅ | 평균 · SD · 분포 · 상관 · Cronbach α | 집계 · 시각화 · 클리닝 |
| **4 (오늘)** | **고스트레스 집단의 조작적 정의 · 임상 cut-off 와의 차이** | **split · 클래스 불균형 · 데이터 누출 · baseline** |
| 5 | 예측변수와 결과의 관계·방향 | 로지스틱 회귀 · 계수 · 표준화 |
| 6~8 | 선형성 → 해석 → 보고 | 트리/포레스트 → 중요도 → 재현성 |

**오늘의 재료** — 3차시가 교정한 결과물이다.

- `data/processed/modeling_frame.parquet` — 역채점 교정이 반영된 **1,321행** 모델링 표
- `configs/modeling.yaml` — split·CV·cutoff 설정 (`random_seed: 42`, `test_size: 0.20`)
- 우리 파이프라인 모듈: `dataset.py` · `preprocessing.py` · `models.py`

> 🔴 오늘의 규칙: **"성능이 좋아 보이면 축하하기 전에 의심한다."**
> 이 프로젝트에서 AUC 1.0 은 성공이 아니라 **경보음**이다."""),

md("""## Step 0 — 재료 확인: 표가 성립하는가"""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(r'''# 3차시가 만든 표를 읽고, 오늘 분석이 전제하는 3가지를 먼저 확인한다
import pandas as pd
from maps_risk.config import load_configs

_, cfg = load_configs("configs")
FRAME = "data/processed/modeling_frame.parquet"

if not os.path.exists(FRAME):
    print("🛑 modeling_frame.parquet 이 없다 — 2·3차시의 build_dataset.py 를 먼저 실행한다.")
else:
    frame = pd.read_parquet(FRAME)
    scores = frame["acculturative_stress_w6"]

    # ① 응답자 1명 = 1행인가  ② target 에 결측이 없나  ③ X 에 6차 변수가 없나
    dup   = frame["id"].duplicated().sum()
    na_y  = scores.isna().sum()
    w6cols = [c for c in frame.columns
              if c.endswith("_w6") and c != "acculturative_stress_w6"]

    print(f"행 {len(frame)} × 열 {frame.shape[1]}")
    print(f"  ① id 중복 {dup}개            {'✅' if dup == 0 else '🛑 병합이 깨졌다'}")
    print(f"  ② target 결측 {na_y}개        {'✅' if na_y == 0 else '🛑 라벨이 조용히 0 이 된다'}")
    print(f"  ③ 6차 컬럼 {w6cols or '없음'}   {'✅' if not w6cols else '🛑 시간 누출'}")
    print(f"\n설정: random_seed={cfg['random_seed']} · test_size={cfg['test_size']} · "
          f"cutoff 분위수={cfg['target']['high_stress_quantile']}")'''),

md("""## Step 1 — 조작적 정의: 선을 긋는다는 것

문화적응 스트레스 점수는 1.00 ~ 4.00 사이의 **연속적인 숫자**다. 그런데 우리가 하려는 일은
**분류(classification)** — "고스트레스 집단인가 아닌가"라는 **예/아니오** 문제다.
연속된 숫자를 둘로 나누려면 **선(cutoff)** 을 그어야 한다.

우리가 긋는 선은 이것이다:

> **학습 데이터 점수의 상위 25%(75 백분위수) 이상 → `high_stress = 1`**

이것을 **조작적 정의(operational definition)** 라고 한다 — "연구자가 분석을 위해 정한 기준"이다.

### 🔴 이것이 아닌 것

| 이것이 아니다 | 왜 |
|---|---|
| 임상 진단 | 정신과 의사가 면담·검사로 내리는 판단이 아니다 |
| 임상 cut-off | 검증된 임상 절단점(예: CES-D 16점)이 아니다 — **우리가 그 자리에서 만든 선**이다 |
| 실제 고위험군 | 상위 25%는 **이 표본 안에서의 상대적 위치**일 뿐이다 |

> 표현 규칙: ❌ "고위험 청소년을 판별하였다" → ✅ "본 연구에서 **조작적으로 정의한**
> 고스트레스 집단을 분류하였다". 8차시 보고서까지 이 표현을 끝까지 지킨다.

그리고 3차시에서 이미 봤듯이 — **선을 긋는 순간 이상한 일이 벌어진다.**"""),

code(r'''# TODO: 부등호 하나가 몇 명을 옮기는지 직접 확인하라
cut = scores.quantile(0.75)

n_ge = (scores >= cut).sum()      # cutoff 이상 (우리 파이프라인의 규칙)
n_gt = (scores ____ cut).sum()    # ← cutoff '초과' 로 바꾸면? 부등호를 채워라
n_tie = (scores == cut).sum()

print(f"cutoff = {cut:.3f}")
print(f"  >= (이상) : {n_ge}명 = {n_ge/len(scores):.1%}")
print(f"  >  (초과) : {n_gt}명 = {n_gt/len(scores):.1%}")
print(f"  동점자    : {n_tie}명")'''),
code(r'''# CHECK Step1
try:
    assert n_ge - n_gt == n_tie, "이상 − 초과 = 동점자 수여야 한다"
    assert n_tie > 100, f"동점자가 {n_tie}명 — 3차시에 본 그 현상이다"
    print(f"✅ PASS — 부등호 하나가 {n_tie}명을 옮긴다. 전체의 {n_tie/len(scores):.1%} 다.")
    print("   '상위 25%' 라는 한 문장 뒤에 이런 선택이 숨어 있다.")
    print("   우리는 >= 를 쓴다(make_high_stress_label). 그리고 실제 양성 비율을 함께 보고한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: '초과'는 같은 값을 포함하지 않는다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
n_gt = (scores > cut).sum()
```

실측: `>=` 447명(33.8%) · `>` 305명(23.1%) · 동점자 **142명**.

**부등호 하나가 142명의 운명을 바꾼다.** 3차시에서 "분위수를 0.75 → 0.80 으로 올려야
비율이 바뀐다"고 본 그 142명이 바로 이들이다.

어느 쪽이 옳은가? **정답이 없다.** 우리는 `>=` 를 쓰기로 했고(파이프라인의
`make_high_stress_label`), 그 선택과 **실제 양성 비율을 함께 보고**한다.
숨기지 않는 것이 정답이다.
</details>"""),

md("""## Step 2 — train/test split: 시험 문제를 미리 보지 않는다

모델을 만들 때 가장 흔한 착각이 이것이다 —
**"학습에 쓴 데이터로 성능을 재면 된다."** 안 된다. 왜?

시험 공부를 하면서 **기출문제 100개를 외웠다**고 하자. 그 100개로 시험을 보면 100점이다.
그런데 그 점수는 **"이 학생이 새 문제를 풀 수 있는가"** 에 대해 아무것도 말해 주지 않는다.

그래서 데이터를 둘로 나눈다:

| | 무엇 | 언제 쓰나 |
|---|---|---|
| **train (학습)** | 80% | 모델을 학습시키고, 튜닝하고, cutoff 를 정한다 |
| **test (시험)** | 20% | **마지막에 딱 한 번.** 그전에는 쳐다보지도 않는다 |

> 🔴 **오늘 우리는 test 를 열지 않는다.** 만들어 놓고 봉인만 한다.
> 오늘의 모든 평가는 **train 안에서 5겹 교차검증(5-fold cross-validation)** 으로 한다 —
> train 을 다시 5조각으로 나눠 4조각으로 배우고 1조각으로 채점하기를 5번 돌려 평균 낸다."""),

code(r'''# TODO: 설정 파일대로 split 하라 (숫자를 코드에 직접 쓰지 않는다 — 설정에서 읽는다)
from sklearn.model_selection import train_test_split

idx_tr, idx_te = train_test_split(
    frame.index,
    test_size=cfg["_______"],          # ← 설정 키를 채워라
    random_state=cfg["_______"],       # ← 설정 키를 채워라
    # 층화(stratify): 두 조각의 구성이 비슷하도록 맞춘다.
    # 진짜 라벨은 아직 없다(cutoff 를 train 에서 정해야 하니까 — 순환!)
    # → 6차 점수의 median 기준 임시 구분으로 층화한다.
    stratify=(scores >= scores.median()).astype(int))

print(f"train {len(idx_tr)}명 · test {len(idx_te)}명")
print(f"겹치는 사람: {len(set(idx_tr) & set(idx_te))}명")'''),
code(r'''# CHECK Step2
try:
    assert len(set(idx_tr) & set(idx_te)) == 0, "train 과 test 에 같은 사람이 있으면 안 된다"
    assert abs(len(idx_te) / len(frame) - 0.20) < 0.01, f"test 가 20% 여야 한다 (지금 {len(idx_te)/len(frame):.1%})"
    assert len(idx_tr) == 1056 and len(idx_te) == 265, f"1056/265 여야 한다 (지금 {len(idx_tr)}/{len(idx_te)})"
    print("✅ PASS — train 1,056 / test 265, 겹침 0명.")
    print("   random_state=42 를 고정했으므로 누가 실행해도 같은 사람이 train 에 들어간다 (재현성).")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: modeling.yaml 의 키 이름은 test_size 와 random_seed 다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 그리고 stratify 의 순환 문제</summary>

```python
test_size=cfg["test_size"], random_state=cfg["random_seed"]
```

**`random_state=42` 를 고정하는 이유**: 안 고정하면 실행할 때마다 다른 사람이 train 에
들어가고, 성능 숫자도 매번 달라진다. 그러면 **재현이 불가능**하다. 42 라는 숫자 자체에
의미는 없다 — "고정했다"는 사실이 중요하다.

**stratify 의 순환 문제 (AGENTS.md 에 기록된 설계 결정)**: 원래는 진짜 라벨(`high_stress`)로
층화하고 싶다. 그런데 진짜 라벨은 **train 의 cutoff 를 정해야** 생기고, cutoff 는
**split 을 해야** 정할 수 있다. **닭이 먼저냐 달걀이 먼저냐**다.
→ 그래서 6차 점수의 **median 기준 임시 구분**으로 층화한다. median 은 전체 분포의
통계라는 한계가 있지만, **분할 균형에만 쓰고 라벨 정의에는 쓰지 않는다.**
이런 타협은 숨기지 않고 **코드 주석과 문서에 적는다.**
</details>"""),

md("""## Step 3 — 순서가 전부다: cutoff 는 train 에서만 ⚠️ (첫 봉우리)

이제 선을 긋는다. 그런데 **순서**가 결정적이다.

```
❌ 틀린 순서:  전체 1,321명의 점수를 보고 선을 긋는다  →  나눈다
✅ 옳은 순서:  나눈다  →  train 1,056명의 점수만 보고 선을 긋는다  →  그 선을 전체에 적용
```

왜 틀렸나? 전체를 보고 선을 그으면, **test 응답자들의 점수가 선의 위치를 결정하는 데
참여**한다. test 는 "한 번도 안 본 새 사람들"이어야 하는데, 이미 한 번 본 셈이 된다.
이것이 **데이터 누출(data leakage)** 이다.

> 비유: 시험 문제의 **합격선**을, 채점할 학생들의 답안을 미리 보고 정하는 것과 같다.
> 문제를 안 보여줬어도 **합격선이 그들의 점수에 맞춰져 있다.**"""),

code(r'''# TODO: cutoff 를 어느 집단에서 계산해야 하는가?
from maps_risk.dataset import make_high_stress_label

q = cfg["target"]["high_stress_quantile"]
y_all, cutoff = make_high_stress_label(
    scores.loc[_______],      # ← cutoff 를 정할 때 볼 사람들 (idx_tr 인가 frame.index 인가)
    scores,                   # 라벨을 붙일 대상 (전체 — 같은 선을 모두에게 적용한다)
    q)
frame["high_stress"] = y_all

print(f"cutoff(train {q:.0%} 분위수) = {cutoff:.4f}")
print(f"train 양성 {y_all.loc[idx_tr].mean():.1%} ({int(y_all.loc[idx_tr].sum())}명)")
print(f"test  양성 {y_all.loc[idx_te].mean():.1%} ({int(y_all.loc[idx_te].sum())}명)  ← 오늘 안 본다")

# 만약 전체로 계산했다면 라벨이 몇 명이나 달라졌을까?
y_leak = (scores >= scores.quantile(q)).astype(int)
print(f"\n전체로 계산한 cutoff = {scores.quantile(q):.4f} → 라벨이 달라진 사람: {int((y_all != y_leak).sum())}명")'''),
code(r'''# CHECK Step3
try:
    assert cutoff == scores.loc[idx_tr].quantile(q), "cutoff 는 train 분위수와 같아야 한다"
    assert abs(y_all.loc[idx_tr].mean() - 0.337) < 0.02, f"train 양성률 33.7% 근처여야 한다"
    print(f"✅ PASS — cutoff 를 train {len(idx_tr)}명만 보고 정했다.")
    print(f"   그런데 전체로 계산해도 값이 같아서 라벨 차이가 {int((y_all != y_leak).sum())}명이다.")
    print("   → 다음 셀에서 이게 무슨 뜻인지 생각해 보자. (규칙을 안 지켜도 된다는 뜻일까?)")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: cutoff 를 정할 때는 train 만 본다 → scores.loc[idx_tr]")'''),
md("""<details><summary>💡 힌트 / 정답 — 그리고 오늘 가장 중요한 교훈</summary>

```python
scores.loc[idx_tr]
```

**실측 결과가 재미있다.** 전체로 계산한 cutoff 도 1.500, train 으로 계산한 cutoff 도 1.500 —
**라벨이 달라진 사람은 0명**이다. 3차시에서 본 그 거대한 동점 덩어리(1.50 에 142명) 때문에
어느 쪽을 봐도 선이 같은 자리에 떨어졌다.

**그럼 규칙을 안 지켜도 되나? 아니다.**

> 🔴 **안전벨트를 맸는데 사고가 안 났다고 해서, 안전벨트가 쓸모없는 것이 아니다.**
> 규칙은 **결과**로 정당화되지 않는다. 규칙은 **절차**다.

이번엔 우연히 같았다. 다른 데이터, 다른 seed, 다른 분위수에서는 달라진다.
그리고 결정적으로 — **미리 확인할 방법이 없다.** "차이가 없을 테니 대충 하자"는
차이가 있는지 **확인한 뒤에야** 할 수 있는 말인데, 확인하려면 이미 전체를 봐야 한다.

우리가 한 일: 규칙대로 하고, **차이가 없었다는 사실까지 기록**했다. 그게 전부다.
</details>"""),

md("""## Step 4 — 클래스 불균형: accuracy 가 거짓말하는 법

train 의 고스트레스는 **33.7%** 다. 즉 **약 2:1 의 불균형**이 있다.
이 상황에서 가장 게으른 모델을 만들어 보자 — **"전원 고스트레스 아님"** 이라고만 답하는 모델.

이런 모델을 **더미 분류기(DummyClassifier)** 라고 하고, 학습을 전혀 하지 않는다.
그런데도 정답률(accuracy)이 꽤 나온다. 얼마나 나올까?"""),

code(r'''# Dummy 와 로지스틱을 같은 조건에서 비교한다 (train 안에서 5-fold CV, test 는 안 연다)
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from maps_risk.dataset import split_features
from maps_risk.preprocessing import make_preprocessor
from sklearn.pipeline import Pipeline

feats = split_features(frame, "A")                 # Model A = 5차 심리사회 변인만
Xtr, ytr = frame.loc[idx_tr, feats], frame.loc[idx_tr, "high_stress"]
cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["random_seed"])

def build(clf, scale=True):
    """전처리를 Pipeline 안에 가둔다 — 이유는 Step 6 에서."""
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("clf", clf)])

models = {
    "Dummy (전부 0)": build(DummyClassifier(strategy="most_frequent"), scale=False),
    "로지스틱 회귀":    build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=cfg["random_seed"])),
}
print(f"Model A features {len(feats)}개 · train {len(Xtr)}명 (고스트레스 {int(ytr.sum())}명)\n")
print(f"{'모델':16s} {'accuracy':>9s} {'AUC':>7s} {'balanced_acc':>13s} {'recall':>8s}")
res = {}
for nm, est in models.items():
    r = {m: cross_val_score(est, Xtr, ytr, cv=cv, scoring=m).mean()
         for m in ("accuracy", "roc_auc", "balanced_accuracy", "recall")}
    res[nm] = r
    print(f"{nm:16s} {r['accuracy']:9.3f} {r['roc_auc']:7.3f} {r['balanced_accuracy']:13.3f} {r['recall']:8.3f}")'''),

code(r'''# TODO: 위 표를 보고 판단하라 — 어느 모델이 더 좋은 모델인가?
더_좋은_모델 = "_______"        # ← "Dummy (전부 0)" 또는 "로지스틱 회귀"
근거_지표    = "_______"        # ← 판단의 근거가 된 지표 이름 (accuracy / roc_auc / recall 중)

print(f"내 판단: {더_좋은_모델}  (근거: {근거_지표})")
print(f"accuracy 만 보면 Dummy {res['Dummy (전부 0)']['accuracy']:.3f} vs 로지스틱 {res['로지스틱 회귀']['accuracy']:.3f}")'''),
code(r'''# CHECK Step4
try:
    assert 더_좋은_모델 == "로지스틱 회귀", "accuracy 가 낮아도 더 좋은 모델일 수 있다"
    assert 근거_지표 in ("roc_auc", "recall", "balanced_accuracy"), \
        "accuracy 는 불균형 데이터에서 판단 근거가 될 수 없다"
    assert res["Dummy (전부 0)"]["accuracy"] > res["로지스틱 회귀"]["accuracy"], "실측이 뒤집혔다"
    print("✅ PASS — **정확도가 더 낮은 모델이 더 좋은 모델이다.**")
    print(f"   Dummy 는 accuracy {res['Dummy (전부 0)']['accuracy']:.1%} 를 받지만 recall 이 0.000 —")
    print("   고스트레스 학생을 **한 명도** 찾아내지 못한다. 그런 모델은 쓸 데가 없다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 고스트레스 집단을 '찾아내는' 것이 목적이다. 그걸 재는 지표는?")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

`더_좋은_모델 = "로지스틱 회귀"`, `근거_지표 = "recall"` (또는 `roc_auc` / `balanced_accuracy`).

**실측**: Dummy accuracy **.663** vs 로지스틱 **.612**. 정확도만 보면 Dummy 가 이긴다.
그런데 Dummy 의 recall 은 **.000** — 고스트레스 학생을 한 명도 못 찾는다.

**왜 이런 일이**: 66.3% 가 음성(고스트레스 아님)이므로, 전부 음성이라고 찍으면
자동으로 66.3% 를 맞힌다. **accuracy 는 다수 클래스의 비율을 그냥 되돌려 주는 지표**가 된다.

그래서 우리 규칙(AGENTS.md): **accuracy 단독 보고 금지 · Dummy 를 항상 같이 보고.**
Dummy 없이 "정확도 66%" 만 보면 대단해 보인다. Dummy 를 옆에 두는 순간 아무것도 아니게 된다.
</details>"""),

code(r'''# 혼동행렬로 보면 훨씬 분명하다 — 숫자가 아니라 '사람 수'로 본다
from maps_risk import evaluation, plots
import matplotlib.pyplot as plt

pred_lr = cross_val_predict(models["로지스틱 회귀"], Xtr, ytr, cv=cv)
print("로지스틱 회귀 (train CV 예측)")
print(evaluation.confusion_frame(ytr, pred_lr).to_string(), "\n")
print("Dummy (전부 0)")
print(evaluation.confusion_frame(ytr, np.zeros(len(ytr), dtype=int)).to_string())

tp = int(((pred_lr == 1) & (ytr == 1)).sum()); fn = int(((pred_lr == 0) & (ytr == 1)).sum())
print(f"\n→ 로지스틱: 고스트레스 {int(ytr.sum())}명 중 {tp}명을 찾아내고 {fn}명을 놓쳤다.")
print(f"→ Dummy   : {int(ytr.sum())}명 전원을 놓쳤다. 놓친 사람 수로 보면 차이가 분명하다.")

plots.class_distribution(frame["high_stress"])
print("\n✅ reports/figures/class_distribution.png")'''),

md("""## Step 5 — 🔥 오늘의 백미: 일부러 누출시키기 (두 번째 봉우리)

이제 **부정행위**를 해 보자. 규칙을 어기면 성능이 얼마나 좋아지는지 **직접 눈으로** 본다.

우리 target 은 6차 문화적응 스트레스 점수로 만들었다. 그렇다면 —
**그 6차 점수 자체를 예측 변수로 넣으면** 어떻게 될까?"""),

code(r'''# TODO: 6차 스트레스 점수를 feature 에 넣어 보라 (해서는 안 되는 짓이다)
leak_feats = feats + ["________________"]      # ← target 을 만든 그 컬럼 이름

honest = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=42)),
                         Xtr, ytr, cv=cv, scoring="roc_auc").mean()
leaked = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=42)),
                         frame.loc[idx_tr, leak_feats], ytr, cv=cv, scoring="roc_auc").mean()

print(f"정직한 모델 (5차 변인만)      AUC = {honest:.4f}")
print(f"누출 모델   (6차 점수 포함)   AUC = {leaked:.4f}   ← ?!")'''),
code(r'''# CHECK Step5
try:
    assert leaked > 0.99, f"누출 모델 AUC 가 1.0 근처여야 한다 (지금 {leaked:.4f})"
    assert honest < 0.80, f"정직한 모델은 훨씬 낮아야 한다 (지금 {honest:.4f})"
    print(f"✅ PASS — AUC {leaked:.4f}. **완벽한 모델**이 만들어졌다.")
    print("   축하할 일일까? 아니다. 이건 성공이 아니라 **경보음**이다.")
    print("   다음 셀에서 무슨 일이 일어난 건지 생각해 보자.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: target(고스트레스)을 만들 때 쓴 컬럼이 frame 안에 그대로 있다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 왜 이것이 쓰레기인가</summary>

```python
leak_feats = feats + ["acculturative_stress_w6"]
```

실측: 정직한 모델 **AUC .653** → 누출 모델 **AUC 1.0000**. 완벽하다.

**무슨 일이 일어났나.** `high_stress` 는 `acculturative_stress_w6 >= 1.5` 로 **만든** 라벨이다.
그 원본 점수를 feature 로 주면, 모델이 할 일은 "1.5 보다 큰가?" 하나뿐이다.
**답안지를 보고 답을 쓴 것**이다. 예측한 게 아니라 **베낀 것**이다.

**왜 쓸모없나 — 세 가지 이유**

1. **실전에서는 그 변수가 없다.** 우리가 하려는 일은 "중2 시점 정보로 **1년 뒤**를 미리
   구분하는 것"이다. 중3 스트레스 점수를 이미 알고 있다면 예측할 이유가 없다.
2. **아무것도 배우지 못한다.** "무엇이 위험요인인가"에 답하려던 건데,
   이 모델의 답은 "스트레스가 높으면 스트레스가 높다"뿐이다.
3. **위험하다.** 숫자가 좋으니 아무도 의심하지 않는다. 조용히 틀린 결론이 배포된다.

> 🔴 **오늘의 문장: AUC 1.0 은 축하가 아니라 경보다.**
> 실무에서 갑자기 성능이 튀면 가장 먼저 의심할 것은 모델이 아니라 **데이터 흐름**이다.
</details>"""),

md("""## Step 6 — 누출의 세 얼굴

방금 본 것은 가장 노골적인 누출이다. 실제로는 훨씬 **알아채기 어려운** 형태로 온다.

| | 이름 | 무엇 | 우리 방어 |
|---|---|---|---|
| ① | **시간 누출** | 예측 시점 이후의 정보가 X 에 들어감 | `assert_no_wave6_predictors()` — 6차 컬럼이 X 에 있으면 멈춘다 |
| ② | **라벨 누출** | test 를 보고 cutoff 를 정함 | cutoff 는 `scores.loc[idx_tr]` 로만 |
| ③ | **전처리 누출** | 전체로 평균·중앙값·변수선택을 fit | 전부 `Pipeline` 안에 |

②와 ③을 직접 재 보자. ①처럼 극적일까?"""),

code(r'''# ③ 전처리 누출 — 전체로 표준화한 뒤 CV vs Pipeline 안에서 폴드마다 표준화
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

imp, sc = SimpleImputer(strategy="median"), StandardScaler()
X_leaked = pd.DataFrame(sc.fit_transform(imp.fit_transform(frame[feats])),   # 전체로 fit ❌
                        index=frame.index, columns=feats)

a_leak = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                         X_leaked.loc[idx_tr], ytr, cv=cv, scoring="roc_auc").mean()
a_ok = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                       Xtr, ytr, cv=cv, scoring="roc_auc").mean()

print(f"전체로 표준화 후 CV : AUC = {a_leak:.4f}   ❌ 규칙 위반")
print(f"Pipeline 안에서     : AUC = {a_ok:.4f}   ✅ 규칙 준수")
print(f"차이 = {abs(a_leak - a_ok):.4f}   ← 어라, 거의 없다?")'''),

code(r'''# 그런데 전처리 누출이 늘 이렇게 얌전한 건 아니다.
# 극단적 사례를 만들어 본다: **순수한 잡음** 200개 중에서 '좋아 보이는' 5개를 고르면?
from sklearn.feature_selection import SelectKBest, f_classif

rng = np.random.default_rng(0)
noise = pd.DataFrame(rng.normal(size=(len(frame), 200)), index=frame.index,
                     columns=[f"noise_{i}" for i in range(200)])
Ntr = noise.loc[idx_tr]

# ❌ CV 밖에서 미리 5개를 고른 뒤 CV → 고를 때 이미 정답(ytr)을 다 봤다
picked = Ntr.columns[SelectKBest(f_classif, k=5).fit(Ntr, ytr).get_support()]
auc_leak = cross_val_score(build(LogisticRegression(max_iter=2000, random_state=42)),
                           Ntr[picked], ytr, cv=cv, scoring="roc_auc").mean()

# ✅ 선택까지 Pipeline 안에 넣어 폴드마다 다시 고르기
inside = Pipeline([("prep", make_preprocessor(scale=False)),
                   ("sel", SelectKBest(f_classif, k=5)),
                   ("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=42))])
auc_ok = cross_val_score(inside, Ntr, ytr, cv=cv, scoring="roc_auc").mean()

print(f"❌ CV 밖에서 미리 고름  : AUC = {auc_leak:.4f}   ← 전부 난수인데!")
print(f"✅ Pipeline 안에서 고름 : AUC = {auc_ok:.4f}   ← 0.5 = 동전 던지기 (정직한 답)")'''),

md("""### Step 6 정리 — 규칙은 결과가 아니라 절차다

방금 세 가지를 쟀다. 결과가 **제각각**이었다:

| 누출 | 실측 차이 | 해석 |
|---|---|---|
| ① 시간 누출 (6차 점수) | AUC .653 → **1.000** | 파국적 |
| ② 라벨 누출 (cutoff) | 라벨 차이 **0명** | 이번엔 차이 없었다 |
| ③ 전처리 누출 (표준화) | **.0001** | 이번엔 차이 없었다 |
| ③ 전처리 누출 (변수 선택) | .503 → **.589** | 순수한 잡음으로 성능이 만들어졌다 |

②③이 이번에 얌전했다고 규칙을 버릴 수 있을까? 없다. 이유는 Step 3 과 같다 —
**차이가 있는지 확인하려면 이미 규칙을 어겨야 한다.**

그래서 우리는 개별 판단에 맡기지 않고 **구조로 막는다.** `Pipeline` 이 그 장치다:

```python
Pipeline([("prep", make_preprocessor()),   # 결측 대치 + 표준화
          ("clf",  LogisticRegression())])  # 모델
```

이렇게 묶으면 `fit()` 이 호출될 때 **전처리도 그 폴드의 train 으로만** 학습된다.
사람이 매번 조심하는 게 아니라, **틀릴 수 없는 모양으로 만들어 둔 것**이다."""),

md("""## Step 7 — 테스트로 못 박기: 규칙을 코드가 지키게 한다

말로 정한 규칙은 잊힌다. 그래서 이 프로젝트는 누출 방지 규칙을 **테스트**로 박아 뒀다.
`tests/test_no_leakage.py` 를 열면 오늘 배운 것이 그대로 들어 있다:

| 테스트 | 무엇을 막나 |
|---|---|
| `test_wave6_predictor_raises` | 6차 변수가 X 에 들어가면 예외 (① 시간 누출) |
| `test_target_column_cannot_enter_X` | target 컬럼이 X 에 들어가면 예외 |
| `test_cutoff_is_computed_from_train_only` | cutoff 를 전체로 계산하면 실패 (② 라벨 누출) |
| `test_scaler_is_fit_on_train_only` | 스케일러가 전체로 fit 되면 실패 (③ 전처리 누출) |
| `test_all_models_are_pipelines` | 전처리가 Pipeline 밖에 있으면 실패 |
| `test_train_test_ids_do_not_overlap` | train/test 에 같은 사람이 있으면 실패 |

> 누군가 6개월 뒤 코드를 고치다 실수로 규칙을 깨면, **테스트가 빨간불로 알려준다.**
> 이것이 "재현 가능한 연구"의 실제 모습이다 — 착한 의도가 아니라 **자동 검사**."""),

code(r'''!python -m pytest tests/test_no_leakage.py -v --no-header -q'''),

code(r'''# 오늘의 산출물과 '봉인' 확인
print("오늘 만든 것")
print(f"  ✅ high_stress 라벨      cutoff={cutoff:.3f} · train {y_all.loc[idx_tr].mean():.1%} 양성")
print(f"  ✅ train/test 분할       {len(idx_tr)} / {len(idx_te)} · seed {cfg['random_seed']}")
print(f"  {'✅' if os.path.exists('reports/figures/class_distribution.png') else '⬜'} reports/figures/class_distribution.png")

print("\n🔒 test 세트 봉인 상태")
print(f"  test {len(idx_te)}명 — 오늘 성능 평가에 한 번도 쓰지 않았다.")
print("  모든 숫자는 train 안 5-fold CV 에서 나왔다. test 는 8차시 최종 평가 때 딱 한 번 연다.")
print("\n※ 오늘 본 AUC 는 전부 CV 값이다. 최종 성능이 아니다 —")
print("  5·6차시에서 모델을 제대로 세우고, 그 뒤에 test 를 연다.")'''),

md("""## 🎯 회고 (5분)

1. Dummy 의 정확도가 66% 인데도 **쓸모없는** 이유를 친구에게 설명한다면?
2. 6차 점수를 넣었더니 AUC 가 1.0 이 됐다. **왜 그게 좋은 소식이 아닌가?**
3. cutoff 를 전체로 계산해도 이번엔 라벨이 하나도 안 바뀌었다.
   **그런데도 규칙을 지켜야 하는 이유**는 무엇인가?

3번이 오늘의 핵심 감각이다 — **규칙은 결과가 아니라 절차로 정당화된다.**

## 📝 과제
- 이 데이터에서 생길 수 있는 **또 다른 누출 시나리오**를 하나 상상해서 3줄로 적기
  (예: "같은 학교 학생이 train 과 test 에 나뉘어 들어가면?")
- `tests/test_no_leakage.py` 의 테스트 하나를 골라, **무엇을 막는지** 한 문단으로 설명
- 조작적 정의 문장을 **연구윤리에 맞게** 다시 쓰기 (❌ 고위험군 판별 → ✅ …)

## ▶️ 다음 (5차시)
> "오늘 라벨과 데이터 분할을 완성했다. 다음엔 드디어 **모델을 해석**한다 —
> 로지스틱 회귀의 **계수**를 읽고 '어떤 변수가 고스트레스 분류와 가장 강하게 관련되는가'에
> 답한다. 그리고 3차시의 그 문제가 돌아온다: **4점 척도와 5점 척도의 계수를 그대로
> 비교해도 되는가?** 답은 '안 된다'이고, 해결책이 **표준화**다."""),
]

os.makedirs("session4", exist_ok=True)
save(cells, "session4/session4.ipynb")

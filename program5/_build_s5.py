# -*- coding: utf-8 -*-
"""session5.ipynb 빌더 — 로지스틱 회귀와 계수 해석 (상세판).

5차시는 "숫자를 해석하는 법"이 아니라 **"해석해도 되는 숫자와 안 되는 숫자를 가르는 법"**
을 배우는 차시다. 계수 순위표를 만들어 놓고, 부트스트랩으로 그 표의 대부분이
해석 불가임을 학생이 직접 확인한다.

★ 오늘도 test 는 열지 않는다. 계수 해석은 train 에 적합한 모델로 한다.

상세판 방침 — 3·4차시와 같은 형식(Step 세분 ①② · CHECK · 해설 접기)을 쓰되
밀도는 **그 중간**이다:
  · 개념을 markdown 으로 설명하면 곧바로 **우리 숫자로 확인하는 셀**을 붙인다.
  · 세 봉우리(표준화 · 부호 뒤집힘 · 부트스트랩) 앞에는 준비 셀을 하나씩 둔다.
  · 4차시처럼 소절을 ①~⑩ 까지 쪼개지는 않는다 — Step 골격은 0~7 그대로다.

실측 근거 (frame 1,321행 × 21열 · train 1,056 / test 265 · seed 42 ·
           Model A 18변수 · class_weight='balanced' · C=1.0):
  cutoff(train q75) 1.500 · train 양성 33.7% · 결측은 economic_status 8개뿐

  [Step 1 확률]
  train 예측확률 최저 .1384 · 중앙 .4845 · 최고 .9331
  .3~.7 구간에 77.4% 가 몰려 있다 (모델이 확신하지 못한다 — 4차시 AUC .653 의 실체)
  임계값 .5 로 지목되는 비율 46.4% vs 실제 양성률 33.7% (balanced 가 문턱을 낮춘 결과)

  [Step 2 계수 번역]
  train 첫 학생(index 312, 실제 y=0): 로그오즈 −0.7238 → 확률 .3265 (predict_proba 와 일치)
    기여 상위: parenting_monitoring −.338 · depression −.241 · self_esteem −.153
  peer_support 오즈비 .773 의 확률 효과 — 출발점에 따라 다르다:
    .05 → .039(−.011) · .30 → .249(−.051) · .50 → .436(−.064) · .75 → .699(−.051)

  [Step 3 표준화]
  표준화 전/후 계수 순위: 18개 중 14개가 바뀐다 (SD 최소 .217 bullying ~ 최대 .868 teacher_support, 4배)
  1위도 self_esteem(원단위) → peer_support(표준화) 로 바뀐다
  bullying 원단위 7위 → 표준화 12위

  [Step 4 순위표]
  계수 상위: peer_support −.257 · self_esteem −.254 · parenting_monitoring −.251 · depression +.189
  절편 −.0695 → 기준확률 .483 (실제 양성률 .337 과 다르다 — balanced 가 절편을 보정)

  [Step 5 부호 뒤집힘]
  단순상관 대비 부호 뒤집힘 7개 (ego_resilience −.170 → +.161 등)
  ego_resilience 추적: 혼자 −.327 → +self_esteem −.069 → +peer_support +.060 → 5개 +.174 → 18개 +.161
  ego_resilience 상관: peer_relationship .614 · life_satisfaction .597 · self_esteem .594 · peer_support .571
  VIF 최대 2.455 (ego_resilience) — 경고 기준 5 에 한참 못 미치는데도 부호가 뒤집힌다

  [Step 6 부트스트랩]
  준비 데모 — self_esteem 평균 3.178: 1,056명이면 95% 구간 폭 .069, 50명만 쓰면 .29 (4배)
  계수 부트스트랩 500회: 뒤집힌 7개는 7/7 이 신뢰구간에 0 포함
    신뢰구간이 0 을 제외하는 변수는 단 3개 — peer_support · self_esteem · parenting_monitoring
    depression 은 [−.034, +.420] 로 0 을 포함 (순위 4위인데 해석 불가)

  [Step 7 Model A vs B]
  Model A CV AUC .6535 (C=0.1) / Model B .6825 (C=0.1) · 개선폭 +.0289
  Model B 의 previous_acculturative_stress 계수 +.448 (2위 peer_support −.255 의 1.8배)
"""
import os

from nb import md, code, save, SETUP, handoff_in, handoff_out

cells = [
md("""# 5차시 — 계수를 읽는다, 그리고 대부분을 읽지 않기로 한다

### 로지스틱 회귀 · 확률 · 계수 · 표준화

> **오늘 한 문장:** "4차시에 라벨과 분할을 끝냈다. 오늘은 드디어 **모델이 무슨 말을 하는지**
> 듣는다 — 그런데 듣고 보니, **대부분은 믿을 수 없는 말**이었다."

오늘 우리는 변수 18개의 **계수 순위표**를 만든다. 그리고 마지막에 이렇게 결론 내린다:

> **"이 표에서 우리가 해석해도 되는 것은 3개뿐이다."**

오늘의 목표 4가지:

1. **로지스틱 회귀**가 확률을 어떻게 다루는지, 계수가 무슨 뜻인지 **직접 계산해서** 설명한다.
2. **표준화**가 왜 필요한지 보인다 — 3차시의 그 숙제가 오늘 돌아온다. ← 고비 1
3. **단순 상관과 다변량 계수가 왜 다른지**(부호까지 뒤집힌다) 설명한다. ← 고비 2
4. **부트스트랩**으로 계수의 불확실성을 재고, 해석 가능한 것만 골라낸다.

> 💡 운영 방식은 3·4차시와 같다: 셀을 위에서 아래로. 코드는 **전부 채워져 있다** —
> 실행 전에 결과를 예측하게 하고, `# CHECK` 에서 `✅` 를 확인한 뒤 넘어간다.
> 막히는 곳에는 **💡 해설**이 접혀 있다.

> 🔒 **오늘도 test 는 열지 않는다.** 계수 해석은 train(1,056명)에 적합한 모델로 하고,
> 성능은 train 안 5-fold CV 로만 본다."""),

md("""## 🗺️ 오늘의 위치 — 5차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1~2 ✅ | 문화적응 스트레스 · 심리척도 · 역채점 | feature/target · pandas · ID join |
| 3 ✅ | 분포 · 상관 · Cronbach α | 집계 · 시각화 · 클리닝 |
| 4 ✅ | 조작적 정의 · 임상 cut-off 와의 차이 | split · 불균형 · **데이터 누출** |
| **5 (오늘)** | **예측변수와 결과의 관계·방향성 · 상관 vs 편회귀계수** | **로지스틱 회귀 · 확률 · 계수 · 표준화 · 부트스트랩** |
| 6 | 심리 특성은 선형적으로 작동하는가 | Decision Tree · Random Forest · 과적합 |
| 7~8 | 위험/보호요인 · 인과 vs 예측 → 보고 | Permutation Importance · 재현성 |

**오늘의 재료** — 4차시가 만든 것 그대로다.

- `modeling_frame.parquet` (1,321행) + 4차시의 `high_stress` 라벨 · train/test 분할
- `maps_risk.evaluation` — 표준화 계수 · **부트스트랩 계수 안정성**

**오늘의 길** — Step 이 7개지만 봉우리는 셋이다.

| Step | 무엇을 | 난이도 |
|---|---|---|
| 0~1 | 재료 확인 · 로지스틱은 무엇을 내놓는가(확률) | 준비 |
| 2 | 계수를 로그오즈 → 오즈비 → 확률로 **직접** 번역 | 손으로 계산 |
| **3** | **표준화** — 자를 통일하지 않으면 순위가 거짓말한다 | 🔴 봉우리 1 |
| 4 | 계수 순위표 — 오늘 만들려던 답 | 수확 |
| **5** | **부호가 뒤집힌다** — 단순상관 ≠ 다변량 계수 | 🔴 봉우리 2 |
| **6** | **부트스트랩** — 18개 중 3개만 남는다 | 🔴 봉우리 3 |
| 7 | Model A vs B — RQ3 에 답하고, 말할 수 있는 데까지만 말한다 | 마무리 |

> 🔴 오늘의 규칙: **"계수 하나만 보고 해석하지 않는다. 불확실성을 옆에 같이 둔다."**"""),

# ══════════════════════════════════════════════════════════════════
# Step 0 — 재료 확인
# ══════════════════════════════════════════════════════════════════
md("""## Step 0 — 재료 확인: 4차시 상태를 그대로 재현한다

오늘은 **새 데이터를 만들지 않는다.** 4차시가 끝난 그 자리에서 그대로 이어 간다 —
같은 표, 같은 라벨, 같은 train/test 분할이다.

그런데 "그대로"가 어떻게 가능한가? 4차시에서 배운 대로 **seed 를 고정**했기 때문이다.
아래 셀은 4차시의 분할 코드를 **한 글자도 바꾸지 않고** 다시 돌린다. 그 결과
train 에 들어가는 1,056명은 지난주와 **한 명도 다르지 않다.**

> 🔒 test 265명은 오늘도 **봉인 상태**다. 계수 해석에도 쓰지 않는다 —
> "해석에 썼으니 괜찮다"는 예외는 없다. 한 번이라도 보면 그 뒤의 모든 숫자가 오염된다."""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(handoff_in(pull=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], require=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], hint="지난 차시 노트북 맨 끝의 '드라이브에 저장' 셀을 실행하면 여기서 자동으로 복원된다")),
code(r'''# 4차시와 '똑같은' 라벨·분할을 다시 만든다 (seed 를 고정했으므로 완전히 동일하다)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from maps_risk.config import load_configs
from maps_risk.dataset import make_high_stress_label, split_features

_, cfg = load_configs("configs")
frame = pd.read_parquet("data/processed/modeling_frame.parquet")
scores = frame["acculturative_stress_w6"]

idx_tr, idx_te = train_test_split(frame.index, test_size=cfg["test_size"],
                                  random_state=cfg["random_seed"],
                                  stratify=(scores >= scores.median()).astype(int))
frame["high_stress"], cutoff = make_high_stress_label(
    scores.loc[idx_tr], scores, cfg["target"]["high_stress_quantile"])

featsA = split_features(frame, "A")     # 5차 심리사회 변인만
featsB = split_features(frame, "B")     # + 이전 스트레스 (Model B)
Xtr, ytr = frame.loc[idx_tr, featsA], frame.loc[idx_tr, "high_stress"]

print(f"train {len(idx_tr)} · test {len(idx_te)}(봉인) · cutoff {cutoff:.3f} · 양성 {ytr.mean():.1%}")
print(f"Model A {len(featsA)}변수 · Model B {len(featsB)}변수")
mi = frame[featsA].isna().sum()
print(f"결측: {mi[mi > 0].to_dict() or '없음'}  ← 학부모 응답 변수라 조금 있다 (중앙값 대치 대상)")'''),
code(r'''# CHECK Step0
try:
    assert (len(idx_tr), len(idx_te)) == (1056, 265), f"분할이 다르다: {len(idx_tr)}/{len(idx_te)}"
    assert abs(cutoff - 1.500) < 1e-6, f"cutoff 가 다르다: {cutoff:.4f}"
    assert (len(featsA), len(featsB)) == (18, 19), f"변수 수가 다르다: {len(featsA)}/{len(featsB)}"
    assert set(idx_tr) & set(idx_te) == set(), "train 과 test 가 겹친다"
    print("✅ PASS — train 1,056 · test 265 · cutoff 1.500 · Model A 18변수 · 겹침 0명.")
    print("   4차시와 **한 명도 다르지 않다.** seed 를 고정했기 때문이다.")
    print("   이것이 8차시에서 다룰 '재현성'의 실제 효용이다 — 매주 같은 자리에서 이어 붙일 수 있다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 3차시 역채점 교정이 반영된 frame 인지 확인하라 (1,321행 × 21열).")'''),

# ══════════════════════════════════════════════════════════════════
# Step 1 — 로지스틱은 무엇을 내놓는가
# ══════════════════════════════════════════════════════════════════
md("""## Step 1 — 로지스틱 회귀 ①: 왜 선형회귀가 아닌가

우리가 맞히려는 것은 **0 또는 1**(고스트레스 집단인가)이다. 그런데 모델이 실제로 내놓는 것은
**확률** — "이 학생이 고스트레스일 가능성 0.62" 같은 숫자다.

여기서 문제가 생긴다. 보통의 **선형회귀**는 이런 식이다:

```
y = b0 + b1·자아존중감 + b2·친구지지 + ...
```

이 식은 **어떤 값이든** 뱉을 수 있다 — 1.4 도, −0.3 도. 그런데 확률은 **0과 1 사이**여야 한다.
"고스트레스일 확률 140%"는 말이 안 된다.

그래서 **로지스틱 회귀(logistic regression)** 는 한 겹을 더 씌운다:

```
로그오즈(log-odds) = b0 + b1·x1 + b2·x2 + ...       ← 여기는 직선 (어떤 값이든 가능)
확률 = 1 / (1 + e^(-로그오즈))                        ← S자 곡선으로 0~1 사이에 눌러 담는다
```

이 S자 곡선을 **시그모이드(sigmoid)** 라고 한다. 아무리 큰 값이 들어와도 1을 넘지 않고,
아무리 작은 값이 들어와도 0 아래로 안 내려간다.

> 🔑 오늘 계속 나올 구조가 여기 다 들어 있다: **직선 부분(로그오즈)에 계수가 있고,
> 우리가 보고 싶은 확률은 그 직선을 S자로 구부린 결과다.** 그래서 "계수 = 확률 몇 %"라고
> 바로 말할 수 없다 — 이 사실이 Step 2 의 전부다."""),

code(r'''# 시그모이드를 눈으로 본다 — 직선이 어떻게 0~1 로 눌리는가
import matplotlib.pyplot as plt
from maps_risk import plots      # import 만 해도 한글 폰트가 잡힌다

z = np.linspace(-6, 6, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(z, z); ax[0].axhline(0, c="gray", lw=.5); ax[0].axhline(1, c="gray", lw=.5)
ax[0].set_title("선형회귀: 0~1 을 벗어난다"); ax[0].set_xlabel("로그오즈"); ax[0].set_ylabel("출력")
ax[1].plot(z, 1/(1+np.exp(-z))); ax[1].axhline(.5, c="gray", ls="--", lw=.8)
ax[1].set_title("시그모이드: 항상 0~1 안"); ax[1].set_xlabel("로그오즈"); ax[1].set_ylabel("확률")
fig.tight_layout(); plt.show()

for v in (-3, -1, 0, 1, 3):
    print(f"  로그오즈 {v:+d}  →  확률 {1/(1+np.exp(-v)):.3f}")
print("\n※ 로그오즈 0 = 확률 0.5 (반반). 양수면 0.5 보다 높고, 음수면 낮다.")
print("※ 그림 왼쪽 끝과 오른쪽 끝을 보라 — 로그오즈가 아무리 커져도 확률은 1에 '가까워질' 뿐이다.")'''),

md("""## Step 1 — 로지스틱 회귀 ②: 우리 학생들의 확률을 실제로 뽑아 본다

개념은 그렇고, **우리 데이터에서는 어떤 숫자가 나오나.** 지금 바로 모델 하나를 학습시켜
train 1,056명 각자의 확률을 뽑아 보자.

모델은 4차시에서 배운 대로 **Pipeline** 으로 만든다:

```
make_preprocessor  →  결측 중앙값 대치 → 표준화        (fit 은 train 에서만)
LogisticRegression →  class_weight="balanced"        (양성이 33.7% 뿐이라)
```

> ⚠️ 지금 파이프라인에는 **표준화가 켜져 있다.** 왜 켜야 하는지는 **Step 3 에서 따진다** —
> 지금은 "확률이 어떻게 생겼는지"만 본다.

> 📌 `class_weight="balanced"` 는 4차시에서 정한 선택이다. 소수 클래스(고스트레스)를
> 더 무겁게 세어 **놓침(FN)을 줄이는** 쪽으로 기울인 것이다. 이 선택이 오늘 절편에
> 흔적을 남기는데, 그건 Step 4 에서 확인한다."""),

code(r'''# 로지스틱 회귀를 실제로 학습시키고, 학생 한 명 한 명의 확률을 뽑는다
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from maps_risk.preprocessing import make_preprocessor

def logit(scale=True, C=1.0):
    """표준화(선택) → 로지스틱 회귀 Pipeline. 전처리를 Pipeline 안에 가두는 것이 4차시의 규칙."""
    return Pipeline([("prep", make_preprocessor(scale=scale)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=cfg["random_seed"], C=C))])

m_std = logit(True).fit(Xtr, ytr)             # 오늘 계속 쓸 모델 (표준화 O)
prob_tr = m_std.predict_proba(Xtr)[:, 1]      # [:, 1] 이 '양성(고스트레스)일 확률'

order = np.argsort(prob_tr)
for tag, i in (("가장 낮은 학생", order[0]),
               ("한가운데 학생", order[len(order)//2]),
               ("가장 높은 학생", order[-1])):
    print(f"  {tag}: 확률 {prob_tr[i]:.3f} (로그오즈 {np.log(prob_tr[i]/(1-prob_tr[i])):+.3f}) "
          f"· 실제 라벨 {int(ytr.iloc[i])}")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(prob_tr, bins=30, color="#4c72b0")
ax.axvline(0.5, color="red", ls="--", lw=1)
ax.set_xlabel("모델이 매긴 고스트레스 확률"); ax.set_ylabel("학생 수")
ax.set_title("train 1,056명의 예측확률 분포 (빨간 선 = 임계값 0.5)")
fig.tight_layout(); plt.show()

near = ((prob_tr > 0.3) & (prob_tr < 0.7)).mean()
print(f"\n확률 최저 {prob_tr.min():.3f} · 중앙 {np.median(prob_tr):.3f} · 최고 {prob_tr.max():.3f}")
print(f"0.3~0.7 사이에 몰려 있는 학생: {near:.1%}")'''),
code(r'''# CHECK Step1
try:
    assert prob_tr.min() > 0 and prob_tr.max() < 1, "확률은 절대 0~1 을 벗어나지 않는다"
    assert abs(prob_tr.min() - 0.138) < 0.02 and abs(prob_tr.max() - 0.933) < 0.02, "실측과 다르다"
    assert near > 0.6, f"확률이 가운데로 몰려 있어야 한다 (지금 {near:.1%})"
    print(f"✅ PASS — 확률은 {prob_tr.min():.3f} ~ {prob_tr.max():.3f}, 한 명도 0~1 밖으로 안 나갔다.")
    print(f"   그런데 {near:.0%} 가 0.3~0.7 에 몰려 있다 — 모델이 **대부분의 학생에 대해 확신하지 못한다.**")
    print("   4차시에서 본 AUC .653(완벽 1.0 · 동전 0.5)의 실체가 바로 이 그림이다.")
    print("   → 오늘 우리는 이렇게 '어정쩡한' 모델의 계수를 해석하려 한다. 이 사실을 기억해 두라.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: predict_proba(...)[:, 1] 이 양성 확률이다.")'''),

# ══════════════════════════════════════════════════════════════════
# Step 2 — 계수 읽는 법
# ══════════════════════════════════════════════════════════════════
md("""## Step 2 — 계수 읽는 법 ①: 로그오즈 → 오즈비 → 확률

`b1 = -0.26` 같은 계수가 나왔을 때, 이것을 어떻게 읽나? **세 단계**로 번역한다.

| 단계 | 무엇 | 읽는 법 |
|---|---|---|
| **① 로그오즈 계수** | `-0.26` | x가 1단위 늘면 로그오즈가 0.26 **줄어든다** — 부호만 직관적이다 |
| **② 오즈비(odds ratio)** | `e^(-0.26) = 0.77` | 오즈가 **0.77배**가 된다 (23% 감소) |
| **③ 확률** | 상황에 따라 다름 | 같은 계수라도 **출발점에 따라 확률 변화폭이 다르다** |

**오즈(odds)** 가 뭔가: 확률이 0.75 라면 오즈는 `0.75 / 0.25 = 3` — "일어날 가능성이
안 일어날 가능성의 3배"라는 뜻이다. 경마나 도박에서 쓰는 그 배당률과 같은 개념이다.

먼저 ①을 **손으로** 확인한다. 모델의 계수와 한 학생의 값을 곱해서 더하면
정말로 `predict_proba` 와 같은 숫자가 나오는가?

```
로그오즈 = (계수1 × 그 학생의 값1) + (계수2 × 값2) + ... + 절편
확률     = 1 / (1 + e^(-로그오즈))
```

> 이 셀이 하는 일은 **"모델이 블랙박스가 아니라는 것"** 을 눈으로 보이는 것이다.
> 로지스틱 회귀는 곱하고 더한 다음 S자로 구부린 것 — 그게 전부다."""),

code(r'''# 학생 한 명의 확률을 '손으로' 계산해서 sklearn 과 대조한다
prep = m_std.named_steps["prep"]      # 결측대치 + 표준화
clf  = m_std.named_steps["clf"]       # 로지스틱 회귀 본체

Z = prep.transform(Xtr)               # 표준화된 값. 열 순서는 Xtr.columns 순서 그대로다
i = 0                                 # train 의 첫 번째 학생

contrib = pd.Series(clf.coef_[0] * Z[i], index=featsA)     # 변수별 기여 = 계수 × 그 학생의 값
logodds = float(contrib.sum() + clf.intercept_[0])
prob    = 1 / (1 + np.exp(-logodds))

print(f"train 첫 번째 학생 (index {Xtr.index[i]}, 실제 라벨 {int(ytr.iloc[i])})")
print(f"  기여 합계 {contrib.sum():+.4f} + 절편 {clf.intercept_[0]:+.4f} = 로그오즈 {logodds:+.4f}")
print(f"  → 확률 = 1/(1+e^{-logodds:+.4f}) = {prob:.4f}")
print(f"  sklearn predict_proba = {m_std.predict_proba(Xtr)[i, 1]:.4f}")

print("\n이 학생의 확률을 가장 크게 움직인 변수 (계수 × 값):")
for f, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(4).items():
    print(f"  {f:22s} {v:+.4f}   (원점수 {Xtr.iloc[i][f]:.2f} → 표준화 {Z[i][featsA.index(f)]:+.2f})")'''),
code(r'''# CHECK Step2
try:
    assert abs(prob - m_std.predict_proba(Xtr)[i, 1]) < 1e-9, "손 계산과 sklearn 이 달라졌다"
    assert abs(logodds - (-0.7238)) < 0.01, f"실측과 다르다: {logodds:.4f}"
    print(f"✅ PASS — 손으로 계산한 {prob:.4f} 와 sklearn 의 값이 소수점 9자리까지 같다.")
    print("   로지스틱 회귀는 **곱하고 더한 뒤 S자로 구부린 것**이다. 블랙박스가 아니다.")
    print("   이 학생은 부모 감독(+1.35 SD)·낮은 우울(−1.28 SD) 덕에 로그오즈가 크게 내려갔다 → 확률 .33")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: prep.transform 의 열 순서 = Xtr.columns 순서 = coef_ 순서다.")'''),

md("""## Step 2 — 계수 읽는 법 ②: 왜 "확률 몇 %"라고 못 말하는가

②까지는 깔끔하다. `e^(-0.257) = 0.773` — "1 SD 늘 때 오즈가 0.773배". 여기까지는
학생이 누구든 **똑같이** 적용된다.

문제는 ③이다. **오즈비는 일정한데 확률 변화폭은 일정하지 않다.**
시그모이드가 곡선이기 때문이다 — 가운데(0.5 근처)는 가파르고, 양 끝은 평평하다.

같은 오즈비 0.773 을 서로 다른 출발점에 적용하면 어떻게 되는지 직접 계산해 보자."""),

code(r'''# 같은 오즈비를 서로 다른 출발 확률에 적용해 본다
b_ps = clf.coef_[0][featsA.index("peer_support")]      # 친구지지 계수 (Step 4 에서 다시 본다)
OR   = np.exp(b_ps)
print(f"친구지지(peer_support) 계수 {b_ps:+.4f} → 오즈비 {OR:.4f}\n")
print("출발 확률   →  친구지지 +1 SD 뒤   변화")
for p0 in (0.05, 0.30, 0.50, 0.75):
    odds1 = (p0 / (1 - p0)) * OR          # 오즈에 곱한다 — 확률에 곱하는 게 아니다
    p1 = odds1 / (1 + odds1)
    print(f"   {p0:.2f}      →      {p1:.3f}        {p1 - p0:+.3f}")
print("\n오즈비는 셋 다 0.773 으로 같다. 그런데 확률 변화는 −0.011 ~ −0.064 로 6배 차이가 난다.")'''),
md("""<details><summary>💡 해설 — 그래서 무엇을 보고해야 하나</summary>

계산 결과를 다시 보면:

| 출발 확률 | +1 SD 뒤 | 변화 |
|---|---|---|
| 0.05 | 0.039 | **−0.011** |
| 0.30 | 0.249 | −0.051 |
| 0.50 | 0.436 | **−0.064** ← 가장 크다 |
| 0.75 | 0.699 | −0.051 |

같은 계수, 같은 오즈비인데 **확률 변화는 6배 차이**가 난다. 그래서

- ❌ "친구지지가 1 SD 높으면 고스트레스 확률이 6%p 낮아진다" — **어느 학생인지 말하지 않으면 틀린 문장**이다.
- ✅ "친구지지가 1 SD 높으면 오즈가 0.77배가 된다(23% 감소)" — 누구에게나 성립한다.
- ✅ "평균적인 학생(확률 .48)을 기준으로 하면 약 6%p 낮아진다" — **기준을 밝히면** 확률로도 말할 수 있다.

논문·보고서에서 로지스틱 결과를 **오즈비로 보고하는 관행**이 여기서 나온다.
확률은 듣는 사람에게 더 친절하지만, **출발점을 함께 적어야 정직한 문장**이 된다.
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 3 — 표준화 (봉우리 1)
# ══════════════════════════════════════════════════════════════════
md("""## Step 3 — 표준화 ①: 3차시의 숙제가 돌아왔다 ⚠️ (첫 봉우리)

3차시에 이런 장면이 있었다:

> "친구지지 4.13 > 우울 1.69 를 보고 **'친구지지가 더 높다'고 말하면 안 된다.**
> 자를 다르게 쓰고 잰 길이를 그대로 비교한 셈이다. → 5차시 표준화의 복선."

오늘이 그 5차시다. 계수에서 같은 문제가 터진다.

**계수는 "x가 1 늘어날 때"의 효과**다. 그런데 우리 변수들은 **1의 의미가 제각각**이다:

- `bullying`(집단괴롭힘)의 표준편차는 **0.217** — 1이 늘어난다는 건 거의 불가능한 변화다
- `teacher_support`(교사지지)의 표준편차는 **0.868** — 1의 변화가 훨씬 흔하다

즉 원래 단위의 계수를 비교하는 것은 **"1cm 늘었을 때"와 "1km 늘었을 때"를 나란히 놓는 것**과 같다.

**해결책: 표준화(standardization).** 모든 변수를 "평균 0, 표준편차 1"로 바꾼 뒤 학습한다.
그러면 계수의 뜻이 통일된다 — **"그 변수가 1 표준편차만큼 늘어날 때"**.

먼저 **표준화가 실제로 숫자를 어떻게 바꾸는지** 눈으로 보자."""),

code(r'''# StandardScaler 가 실제로 무슨 일을 하는가 — 같은 학생, 같은 값의 전/후
prep_demo = make_preprocessor(scale=True).fit(Xtr)          # train 에서만 fit (4차시 규칙)
Zdf = pd.DataFrame(prep_demo.transform(Xtr), columns=featsA, index=Xtr.index)

print("변수별 '자'의 눈금 (train 기준)")
print(f"{'변수':22s}{'평균':>8s}{'SD':>8s}   |  첫 학생 원점수 → 표준화값")
for f in ("bullying", "depression", "self_esteem", "teacher_support"):
    print(f"  {f:20s}{Xtr[f].mean():8.3f}{Xtr[f].std():8.3f}   |  "
          f"{Xtr.iloc[0][f]:6.2f} → {Zdf.iloc[0][f]:+.3f} SD")

print(f"\n표준화 후 전체 평균/SD 확인: mean {Zdf.mean().abs().max():.6f} (0 이어야 한다) · "
      f"SD {Zdf.std().mean():.4f} (1 이어야 한다)")
print("  ※ SD 가 1.0005 로 나오는 것은 pandas .std() 가 n−1 로 나누기 때문이다 (StandardScaler 는 n).")
print("→ 값의 '순서'는 하나도 안 바뀐다. 바뀐 것은 **눈금**뿐이다.")'''),

md("""## Step 3 — 표준화 ②: 자를 통일하면 순위가 바뀐다

이제 진짜 질문이다. **표준화를 하느냐 안 하느냐로 계수 순위가 실제로 달라지는가?**

같은 데이터·같은 모델로 두 번 학습해서 나란히 놓아 본다. 하나는 원래 단위 그대로,
하나는 표준화 후.

> 우리 파이프라인은 이미 표준화하도록 돼 있다: `make_preprocessor(scale=True)` 안의 `StandardScaler`.
> 그리고 4차시에 배운 대로 **Pipeline 안에 있으므로 train 에서만 fit** 된다.

**실행 전에 예측해 보라:** 18개 중 순위가 바뀌는 변수는 몇 개일 것 같은가?"""),

code(r'''# 표준화를 켠 모델과 끈 모델의 계수 순위를 비교한다
m_raw = logit(False).fit(Xtr, ytr)      # 원래 단위 그대로 (m_std 는 Step 1 에서 이미 학습했다)

d = pd.DataFrame({
    "feature": featsA,
    "SD": Xtr[featsA].std().values,
    "coef_원단위": m_raw.named_steps["clf"].coef_[0],
    "coef_표준화": m_std.named_steps["clf"].coef_[0],
})
d["순위_원단위"] = d["coef_원단위"].abs().rank(ascending=False).astype(int)
d["순위_표준화"] = d["coef_표준화"].abs().rank(ascending=False).astype(int)
print(d.sort_values("coef_표준화", key=abs, ascending=False).round(3).to_string(index=False))

n_changed = int((d["순위_원단위"] != d["순위_표준화"]).sum())
print(f"\n순위가 바뀐 변수: {n_changed} / {len(d)}개")
print(f"SD 최소 {d.SD.min():.3f}({d.loc[d.SD.idxmin(),'feature']}) ~ "
      f"최대 {d.SD.max():.3f}({d.loc[d.SD.idxmax(),'feature']}) — {d.SD.max()/d.SD.min():.1f}배 차이")'''),
code(r'''# CHECK Step3
try:
    assert n_changed >= 10, f"순위가 크게 바뀌어야 한다 (지금 {n_changed}개)"
    top_raw = d.loc[d["순위_원단위"] == 1, "feature"].iloc[0]
    top_std = d.loc[d["순위_표준화"] == 1, "feature"].iloc[0]
    assert top_raw != top_std, "1위 변수마저 달라진다"
    b = d.loc[d["feature"] == "bullying"].iloc[0]
    assert b["순위_원단위"] < b["순위_표준화"], "bullying 은 표준화하면 순위가 내려간다"
    print(f"✅ PASS — 18개 중 {n_changed}개의 순위가 바뀐다. 1위도 {top_raw} → {top_std} 로 달라졌다.")
    print(f"   bullying: 원단위 {int(b['순위_원단위'])}위 → 표준화 {int(b['순위_표준화'])}위로 밀려난다.")
    print("   표준화 없이 계수를 비교하면 **자를 섞어 쓴 것**이다. 3차시의 그 문제 그대로다.")
    print("   ※ 예측 성능은 거의 안 바뀐다. 표준화는 성능이 아니라 **해석**을 위한 것이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: make_preprocessor(scale=scale) — 함수 인자를 그대로 넘긴다.")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

```python
Pipeline([("prep", make_preprocessor(scale=scale)), ...])
```

실측: **18개 중 14개**의 순위가 바뀐다. 1위도 `self_esteem`(원단위) → `peer_support`(표준화)로 달라진다.

가장 극적인 것은 `bullying`(집단괴롭힘)이다. 원단위 계수로는 **7위**인데 표준화하면 **12위**로
떨어진다. 왜? SD 가 **0.217** 로 매우 작기 때문이다(3차시에서 본 왜도 +7.9, "거의 전원이 1점").
"1점 늘어날 때"의 효과가 커 보였지만, **실제로 1점 늘어나는 학생이 거의 없다.**

반대 방향도 있다. `teacher_support`(교사지지)는 SD 가 0.868 로 가장 큰데, 원단위 **11위**에서
표준화 후 **7위**로 올라온다. "1점 차이"가 흔한 변수라 원단위 계수가 작게 나왔던 것이다.

> **표준화는 성능을 위한 것이 아니라 해석을 위한 것이다.** AUC 는 거의 그대로다.
> 하지만 "어떤 변수가 더 중요한가"라는 질문에 답하려면 반드시 필요하다.

⚠️ 한 가지 주의: 표준화된 계수는 **"이 표본의 SD 를 기준으로"** 라는 뜻이다.
다른 표본에서 SD 가 다르면 같은 현상도 다른 크기로 보인다. 표준화가 만능 자는 아니다.
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 4 — 계수 순위표
# ══════════════════════════════════════════════════════════════════
md("""## Step 4 — 계수 순위표: 오늘 만들려던 그것

이제 표준화된 계수로 순위표를 만든다. 이것이 오늘 우리가 원했던 답이다 —
**"어떤 변수가 고스트레스 분류와 가장 강하게 관련되는가."**

읽는 법 세 가지만 붙잡고 표를 보자.

- **부호**: 음수면 "그 변수가 높을수록 고스트레스 확률이 낮다"(보호요인 **방향**)
- **크기**: 절댓값이 클수록 로그오즈를 많이 움직인다 — 표준화했으니 **서로 비교 가능**하다
- **오즈비**: `e^계수`. 0.77 이면 1 SD 늘 때 오즈가 23% 감소"""),

code(r'''# 우리 모듈로 표준화 계수와 오즈비를 뽑는다
from maps_risk import evaluation

coefs = evaluation.standardized_coefficients(m_std, featsA)
print(coefs.head(8).round(3).to_string(index=False))
print("\n읽는 법: coef 가 음수 = 그 변수가 높을수록 고스트레스 확률이 낮다 (보호요인 방향)")
print("         odds_ratio 0.77 = 1 SD 늘 때 오즈가 0.77배 (23% 감소)")'''),

code(r'''# 확률로도 환산해 본다 — "1 SD 차이"가 확률로 얼마인가
b0 = m_std.named_steps["clf"].intercept_[0]
c = coefs.set_index("feature")["coef"]
p0 = 1 / (1 + np.exp(-b0))
print(f"기준 확률(모든 변수가 평균인 학생) = {p0:.3f}")
print(f"  ※ 실제 양성률 {ytr.mean():.3f} 와 다르다 — class_weight='balanced' 가 절편을 보정했기 때문.")
print("     그래서 이 확률은 '유병률 추정'이 아니라 **상대 비교용**으로만 읽는다.\n")
for f in ("peer_support", "self_esteem", "depression"):
    lo = 1 / (1 + np.exp(-(b0 - c[f]))); hi = 1 / (1 + np.exp(-(b0 + c[f])))
    print(f"  {f:22s} −1SD → {lo:.3f} · +1SD → {hi:.3f}   (차이 {hi-lo:+.3f})")'''),
code(r'''# CHECK Step4
try:
    top3 = coefs.head(3)["feature"].tolist()
    assert set(top3) == {"peer_support", "self_esteem", "parenting_monitoring"}, f"상위 3개가 다르다: {top3}"
    assert (coefs.head(3)["coef"] < 0).all(), "상위 3개는 전부 음수여야 한다"
    assert (coefs.head(3)["odds_ratio"] < 1).all(), "계수가 음수면 오즈비는 1보다 작다"
    assert abs(p0 - 0.483) < 0.02, f"기준확률이 다르다: {p0:.3f}"
    print(f"✅ PASS — 상위 3개는 {', '.join(top3)} 이고 **셋 다 음수**(보호요인 방향)다.")
    print("   4위 depression 만 양수 — 우울이 높을수록 1년 뒤 고스트레스 확률이 높다는 방향.")
    print("   여기까지가 '오늘 만들려던 표'다. 그런데 이 표를 그대로 믿으면 안 된다 — Step 5·6.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: standardized_coefficients 는 |계수| 내림차순으로 정렬해 돌려준다.")'''),

# ══════════════════════════════════════════════════════════════════
# Step 5 — 부호 뒤집힘 (봉우리 2)
# ══════════════════════════════════════════════════════════════════
md("""## Step 5 — 부호가 뒤집힌다 🔴 ① (두 번째 봉우리)

순위표를 3차시의 **단순 상관**과 나란히 놓아 보자. 같은 데이터, 같은 변수, 같은 target 이다.
당연히 비슷해야 할 것 같은데 —

**실행 전에 예측해 보라:** 18개 중 부호가 서로 다른 변수는 몇 개일 것 같은가? 0개? 1~2개?"""),

code(r'''# 단순 상관을 계산해 계수와 나란히 놓고, 부호가 다른 변수를 센다
r_biv = Xtr.apply(lambda col: col.corr(frame.loc[idx_tr, "acculturative_stress_w6"]))

cmp = pd.DataFrame({"단순상관_r": r_biv, "표준화계수": c})
cmp["부호"] = np.where(np.sign(cmp["단순상관_r"]) == np.sign(cmp["표준화계수"]),
                      "같음", "🔴 뒤집힘")
cmp = cmp.reindex(c.abs().sort_values(ascending=False).index)
print(cmp.round(3).to_string())

n_flip = int((cmp["부호"] == "🔴 뒤집힘").sum())
print(f"\n부호가 뒤집힌 변수: {n_flip}개 / {len(cmp)}개")'''),
code(r'''# CHECK Step5
try:
    assert n_flip >= 5, f"부호 뒤집힘이 여러 개 나와야 한다 (지금 {n_flip})"
    assert cmp.loc["ego_resilience", "부호"] == "🔴 뒤집힘", "자아탄력성이 대표 사례다"
    print(f"✅ PASS — {n_flip}개 변수의 부호가 뒤집혔다.")
    print("   자아탄력성: 단순상관 −0.170(보호요인처럼) → 다변량 계수 +0.161(위험요인처럼?!)")
    print("   같은 데이터인데 왜 이런 일이 생기나 — 다음 셀에서 추적한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: np.sign() 이 서로 다른 행을 세면 된다.")'''),

md("""## Step 5 — 부호가 뒤집힌다 🔴 ②: 범인을 추적한다

`ego_resilience`(자아탄력성) 하나를 골라 **변수를 한 개씩 늘려 가며** 계수가 어떻게
변하는지 따라가 보자. 언제 부호가 뒤집히는지 정확히 보인다."""),

code(r'''# 자아탄력성 계수를 추적한다 — 변수를 하나씩 추가하며 계수가 어떻게 변하나
track = [["ego_resilience"],
         ["ego_resilience", "self_esteem"],
         ["ego_resilience", "self_esteem", "peer_support"],
         ["ego_resilience", "self_esteem", "peer_support", "peer_relationship", "life_satisfaction"],
         featsA]
print("함께 넣은 변수                                              ego_resilience 계수")
for cols in track:
    v = logit(True).fit(Xtr[cols], ytr).named_steps["clf"].coef_[0][cols.index("ego_resilience")]
    label = f"{len(cols):2d}개: " + ", ".join(cols[:3]) + ("…" if len(cols) > 3 else "")
    print(f"  {label:56s} {v:+.3f}")
print("\n→ 혼자 있을 때는 −0.327(보호요인). self_esteem 을 넣자 −0.069 로 줄고,")
print("  peer_support 까지 넣자 +0.060 으로 **부호가 뒤집힌다.**")

print("\n자아탄력성과 가장 강하게 얽힌 변수 (train 상관):")
print(Xtr.corr()["ego_resilience"].drop("ego_resilience")
        .sort_values(key=abs, ascending=False).head(4).round(3).to_string())'''),

md("""### 왜 이런 일이 생기나 — 계수의 진짜 뜻

**단순 상관과 회귀 계수는 서로 다른 질문에 답한다.**

| | 질문 |
|---|---|
| **단순 상관** | "자아탄력성이 높은 학생은 스트레스가 낮은가?" → 그렇다 (r = −.17) |
| **회귀 계수** | "**자아존중감·친구지지·교우관계가 똑같은 두 학생** 중, 자아탄력성이 더 높은 쪽은?" → 오히려 조금 높다 (+.16) |

회귀 계수는 항상 **"다른 변수를 통제했을 때"** 의 관계다. 그런데 위 셀에서 봤듯
자아탄력성은 교우관계(r ≈ .61)·삶의만족(.60)·자아존중감(.59)·친구지지(.57)와 강하게 얽혀 있다.
**공통 부분을 다른 변수들이 먼저 가져가고 남은 것**이 자아탄력성의 계수가 된다.
그 나머지가 무엇을 뜻하는지는 — **솔직히 말해서 해석하기 매우 어렵다.**

> 📌 통계 용어로는 이것을 **억제 효과(suppression)** 또는 다중공선성의 부작용이라고 부른다.
> 이름을 아는 것보다 중요한 것은 **"계수는 그 변수 혼자의 관계가 아니다"** 를 몸으로 아는 것이다."""),

md("""## Step 5 — 부호가 뒤집힌다 🔴 ③: "검사를 통과했으니 안심"이 아니다

변수끼리 얽혀 있는 정도를 재는 표준 도구가 **VIF(분산팽창지수, Variance Inflation Factor)** 다.

```
VIF(자아탄력성) = 1 / (1 − R²)      ← 나머지 17개 변수로 자아탄력성을 예측했을 때의 R²
```

나머지 변수들로 그 변수가 잘 설명될수록 VIF 가 커진다. 흔히 **5 또는 10** 을 넘으면
"다중공선성 경고"로 본다. 우리 데이터는 어떨까?"""),

code(r'''# VIF 를 직접 계산한다 — 나머지 변수로 그 변수를 예측했을 때의 R² 로부터
from sklearn.linear_model import LinearRegression

sub = Xtr[featsA].dropna()          # economic_status 결측 8명 제외 (VIF 계산에만 해당)
vif = {}
for f in featsA:
    others = [x for x in featsA if x != f]
    r2 = LinearRegression().fit(sub[others], sub[f]).score(sub[others], sub[f])
    vif[f] = 1 / (1 - r2)
vif = pd.Series(vif).sort_values(ascending=False)

print("VIF 상위 6개")
print(vif.head(6).round(3).to_string())
print(f"\n최댓값 {vif.max():.2f} ({vif.idxmax()}) · 흔히 쓰는 경고 기준은 5 또는 10")
flipped = cmp.index[cmp["부호"] == "🔴 뒤집힘"].tolist()
print(f"부호가 뒤집힌 {len(flipped)}개의 VIF 최댓값: {vif[flipped].max():.2f}")'''),
code(r'''# CHECK Step5-vif
try:
    assert vif.max() < 5, f"경고 기준을 넘지 않아야 한다 (지금 {vif.max():.2f})"
    assert vif.idxmax() == "ego_resilience", f"최댓값 변수가 다르다: {vif.idxmax()}"
    assert vif[flipped].max() < 5, "뒤집힌 변수들도 전부 기준 미만이다"
    print(f"✅ PASS — VIF 최댓값 {vif.max():.2f}({vif.idxmax()}) 로 **경고 기준 5에 한참 못 미친다.**")
    print("   그런데 바로 그 변수의 부호가 뒤집혔다. 뒤집힌 7개도 전부 기준 미만이다.")
    print("   🔴 그러므로: **'다중공선성 검사를 통과했으니 계수를 믿어도 된다'는 결론은 성립하지 않는다.**")
    print("   기준을 넘지 않아도 부호는 뒤집힌다 — 검사는 최소한의 위생이지 해석의 면허가 아니다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: VIF = 1/(1−R²), R² 는 나머지 변수로 그 변수를 회귀한 결과.")'''),
md("""<details><summary>💡 해설 — 그럼 VIF 는 왜 계산하나</summary>

VIF 가 쓸모없다는 뜻이 **아니다.** VIF 는 정확히 한 가지를 말해 준다:

> **"이 계수의 분산이 얼마나 부풀었나."**

VIF 10 이면 그 계수의 표준오차가 √10 ≈ 3.2배로 커졌다는 뜻이다. 그러니 VIF 가 **크면**
"계수를 믿지 말라"는 신호가 맞다. 문제는 **반대 방향의 추론**이다:

| 추론 | 성립하나 |
|---|---|
| VIF 가 크다 → 계수가 불안정할 수 있다 | ✅ |
| VIF 가 작다 → 계수가 안정적이다 | ❌ **우리 데이터가 반례다** |

왜 반대가 성립하지 않나: VIF 는 **"분산이 얼마나 부풀었나"** 만 보지 **"원래 계수가 0 근처인가"**
는 보지 않는다. 계수 자체가 작으면 VIF 가 낮아도 부호가 쉽게 흔들린다. `ego_resilience` 의
계수 +.161 은 부트스트랩 표준편차 **.116** 짜리다 — 계수가 표준편차의 1.4배밖에 안 된다.

> 🔑 **그래서 순서는 이렇다: VIF 로 거르고(위생), 부트스트랩으로 판정한다(해석).**
> 오늘 우리가 Step 5 → Step 6 으로 넘어가는 이유가 정확히 이것이다.
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 6 — 부트스트랩 (봉우리 3)
# ══════════════════════════════════════════════════════════════════
md("""## Step 6 — 불확실성을 잰다 🔍 ①: 부트스트랩이란 무엇인가

여기서 멈추면 안 된다. 순위표에는 숫자가 **하나씩**만 적혀 있는데, 그 숫자가
**얼마나 흔들리는지**를 우리는 아직 모른다.

**부트스트랩(bootstrap)** 으로 잰다. 아이디어는 놀랄 만큼 단순하다:

```
가진 데이터에서 "같은 크기로, 중복을 허용해" 다시 뽑는다  →  통계량을 계산한다
이것을 수백 번 반복한다  →  그 통계량이 얼마나 흔들리는지 분포로 본다
```

"같은 모집단에서 표본을 다시 뽑았다면 어떤 값이 나왔을까"를 **흉내 내는** 것이다.
모델과는 아무 상관 없는 아이디어라서, 먼저 **평균** 하나로 감을 잡고 가자.

**실행 전에 예측해 보라:** 1,056명에서 잰 평균과 50명에서 잰 평균 중, 어느 쪽이 더 흔들릴까?"""),

code(r'''# 부트스트랩 예열 — 모델 없이 '평균' 하나로 감을 잡는다
rng = np.random.default_rng(0)

def boot_mean_ci(values, n_boot=500):
    """값들을 중복 허용으로 다시 뽑아 평균을 500번 구하고, 95% 구간을 돌려준다."""
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return means.mean(), lo, hi

full = Xtr["self_esteem"].values
small = full[:50]                      # 만약 50명만 조사했다면?

for label, v in (("1,056명 전부", full), ("50명만", small)):
    m, lo, hi = boot_mean_ci(v)
    print(f"  {label:12s} 평균 {v.mean():.3f} · 95% 구간 [{lo:.3f}, {hi:.3f}] · 폭 {hi-lo:.3f}")

print("\n→ 같은 값이라도 **표본이 작으면 구간이 넓다.** 평균은 비슷한데 '믿음의 폭'이 다르다.")
print("  부트스트랩은 이 '폭'을 데이터만 가지고 재는 방법이다 — 공식도, 정규분포 가정도 필요 없다.")'''),

md("""## Step 6 — 불확실성을 잰다 🔍 ②: 계수에 같은 방법을 쓴다

이제 평균 자리에 **계수**를 넣는다. 방법은 똑같다:

```
train 1,056명 중에서 1,056명을 "중복 허용"으로 다시 뽑는다  → 모델 학습 → 계수 18개 기록
이것을 500번 반복한다  →  각 계수의 분포를 본다
```

계수 500개가 좁게 모여 있으면 안정적이고, 넓게 퍼져 있으면 그 계수는 **운이 좌우한 값**이다.

특히 **95% 구간이 0을 포함하는지**를 본다. 포함한다면 —
"이 데이터로는 **방향조차 단정할 수 없다**"는 뜻이다.

> ⚠️ 이것은 정식 추론 통계(p-value)가 아니라 **안정성 진단**이다. "유의하다/아니다"로
> 읽지 말고 **"해석해도 되나/안 되나"** 로 읽는다.

**실행 전에 예측해 보라:** 18개 중 몇 개가 살아남을 것 같은가?"""),

code(r'''# 부트스트랩으로 계수의 불확실성을 잰다 (500회, 30초쯤 걸린다)
boot = evaluation.bootstrap_coefficients(
    logit(True),               # 아직 fit 하지 않은 Pipeline 을 넘긴다
    Xtr, ytr,
    n_boot=500,
    seed=0)

print(boot.round(3).to_string(index=False))
stable = boot.loc[~boot["includes_zero"], "feature"].tolist()
print(f"\n신뢰구간이 0 을 포함하지 않는(= 해석 가능한) 변수: {stable}")'''),
code(r'''# CHECK Step6
try:
    assert len(boot) == len(featsA), "모든 변수가 나와야 한다"
    assert len(stable) <= 5, f"해석 가능한 변수는 소수여야 한다 (지금 {len(stable)}개)"
    zero_in = boot.set_index("feature").loc[flipped, "includes_zero"]
    assert zero_in.all(), "부호가 뒤집힌 변수는 전부 신뢰구간이 0 을 포함해야 한다"
    dep = boot.set_index("feature").loc["depression"]
    assert bool(dep["includes_zero"]), "우울도 0 을 포함한다 (순위 4위인데도)"
    print(f"✅ PASS — 18개 중 해석 가능한 것은 **{len(stable)}개뿐**이다: {', '.join(stable)}")
    print(f"   그리고 부호가 뒤집혔던 {len(flipped)}개는 **{int(zero_in.sum())}/{len(flipped)} 전부** 신뢰구간에 0 을 포함한다.")
    print("   → 부호 뒤집힘은 '발견'이 아니라 **불안정하다는 경고**였다.")
    print(f"   순위 4위 depression 도 [{dep['ci_low']:+.3f}, {dep['ci_high']:+.3f}] — 0 을 포함한다. 여기서 유혹이 꺾인다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: n_boot=500")'''),
md("""<details><summary>💡 해설 — 오늘의 결론</summary>

```python
n_boot=500
```

**신뢰구간이 0을 제외하는 변수는 단 3개다:**

| 변수 | 계수 | 95% 구간 | 부호 일관성 |
|---|---|---|---|
| `peer_support` (친구지지) | −.257 | [−.475, −.071] | 99.6% |
| `self_esteem` (자아존중감) | −.254 | [−.475, −.066] | 99.2% |
| `parenting_monitoring` (부모 감독) | −.251 | [−.420, −.087] | 100% |

**세 개 다 음수 — 전부 보호요인 방향이다.** 그리고 심리학적으로 말이 된다.

반면 **부호가 뒤집혔던 7개는 7개 전부** 신뢰구간이 0을 포함한다.
`ego_resilience` 의 +0.161 은 [−.052, +.416] — **0을 한가운데 두고 있다.**
"자아탄력성이 위험요인"이라는 해석은 **절대 하면 안 된다.**

주의할 것이 하나 더 있다. **`depression`(우울)도 0을 포함한다** ([−.033, +.420]).
3차시 단순 상관에서는 2위(r = .246)였고 계수 순위로도 4위인데, **다변량에서는
단정할 수 없다.** 순위표 4위까지 해석하고 싶은 유혹이 정확히 여기서 꺾인다.

**세 결과를 겹쳐 읽는 것이 오늘의 핵심이다:**

| 진단 | 무엇을 말해 주나 | 한계 |
|---|---|---|
| 표준화 (Step 3) | 순위를 **비교 가능**하게 만든다 | 비교 가능해도 **안정적이라는 뜻은 아니다** |
| VIF (Step 5) | 변수끼리 얼마나 얽혔나 | **기준을 통과해도 부호는 뒤집힌다** |
| 부트스트랩 (Step 6) | 그 계수가 **흔들리는 폭** | 정식 추론이 아니라 안정성 진단이다 |

> 🔴 **오늘의 결론: 18개 중 3개만 말한다. 나머지 15개에 대해서는 "모른다"고 말한다.**
> 그게 이 데이터가 허락하는 전부다.
</details>"""),

code(r'''# 그림으로 남긴다 — 계수와 신뢰구간을 함께 (오차막대가 0 선을 넘는지 한눈에)
import os; os.makedirs("reports/figures", exist_ok=True)
b = boot.iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#d62728" if z else "#1f77b4" for z in b["includes_zero"]]
ax.errorbar(b["coef"], range(len(b)),
            xerr=[b["coef"] - b["ci_low"], b["ci_high"] - b["coef"]],
            fmt="o", ecolor="gray", elinewidth=1, capsize=3, linestyle="none")
ax.scatter(b["coef"], range(len(b)), c=colors, zorder=3)
ax.axvline(0, color="black", lw=1)
ax.set_yticks(range(len(b)), b["feature"], fontsize=9)
ax.set_xlabel("표준화 계수 (95% 부트스트랩 구간)")
ax.set_title("파란색만 해석 가능 — 빨간색은 구간이 0 을 포함")
fig.tight_layout(); fig.savefig("reports/figures/logistic_coefficients.png", dpi=150)
plt.show()
print("✅ reports/figures/logistic_coefficients.png")'''),

# ══════════════════════════════════════════════════════════════════
# Step 7 — Model A vs B
# ══════════════════════════════════════════════════════════════════
md("""## Step 7 — Model A vs B: RQ3 에 답한다

1차시에 세운 세 번째 연구 질문을 기억하는가:

> **RQ3.** 이전 시점(중2)의 문화적응 스트레스를 추가하면 예측력이 얼마나 개선되는가?

- **Model A** = 5차 심리사회 변인만 (18개) — "이전 스트레스를 *모르는* 상태"
- **Model B** = A + 5차 문화적응 스트레스 (19개) — "이전 상태를 아는 상태"

4차시에서 확인했듯 5차 스트레스는 **누출이 아니다** — 예측 시점(중2)에 알 수 있는 정보다."""),

code(r'''# 두 모델을 같은 조건에서 비교한다 (train 5-fold CV, C 는 CV 로 튜닝)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["random_seed"])
best = {}
for name, cols in (("A", featsA), ("B", featsB)):
    gs = GridSearchCV(logit(True), {"clf__C": cfg["models"]["logistic_regression"]["C"]},
                      scoring="roc_auc", cv=cv, n_jobs=-1).fit(frame.loc[idx_tr, cols], ytr)
    best[name] = gs
    print(f"Model {name}: 변수 {len(cols):2d}개 · best C={gs.best_params_['clf__C']} · CV AUC={gs.best_score_:.4f}")

gain = best["B"].best_score_ - best["A"].best_score_
print(f"\nRQ3 의 답: 이전 스트레스를 추가하면 CV AUC 가 {gain:+.4f} 개선된다.")

cB = pd.Series(best["B"].best_estimator_.named_steps["clf"].coef_[0], index=featsB)
print(f"\nModel B 계수 상위 5개:")
print(cB.reindex(cB.abs().sort_values(ascending=False).index).head().round(3).to_string())'''),
code(r'''# CHECK Step7
try:
    assert gain > 0, "이전 스트레스를 넣으면 개선돼야 한다"
    assert cB.abs().idxmax() == "previous_acculturative_stress", "가장 큰 계수는 이전 스트레스다"
    ratio = abs(cB["previous_acculturative_stress"]) / cB.drop("previous_acculturative_stress").abs().max()
    assert best["A"].best_score_ < 0.75, "AUC 가 갑자기 높아지면 누출을 의심해야 한다"
    print(f"✅ PASS — CV AUC {best['A'].best_score_:.4f} → {best['B'].best_score_:.4f} ({gain:+.4f}).")
    print(f"   이전 스트레스 계수 {cB['previous_acculturative_stress']:+.3f} — 2위 변수의 {ratio:.1f}배로 가장 크다.")
    print("   그런데 AUC 는 .65 → .68 밖에 안 올랐다. '가장 큰 계수'와 '큰 성능 향상'은 다른 말이다.")
    print("   ※ 만약 AUC 가 .9 를 넘었다면 4차시에서 배운 대로 **누출부터 의심**해야 한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: Model B 는 split_features(frame, 'B') 로 만든 19개 변수다.")'''),

md("""### Step 7 해석 — 숫자를 어디까지 말할 수 있나

**RQ3 의 답**: 이전 스트레스를 추가하면 CV AUC 가 **.6535 → .6825**(약 +.029) 개선된다.
그리고 Model B 에서 `previous_acculturative_stress` 의 계수는 **+0.448** — 2위인 친구지지(−0.255)보다
**1.8배 크다.**

이걸 어떻게 서술해야 하나. 세 가지 표현을 비교해 보자:

| | 표현 | 판정 |
|---|---|---|
| ❌ | "이전 스트레스가 이후 스트레스를 **일으킨다**" | 인과 주장 — 관찰 자료로는 불가능 |
| ❌ | "이전 스트레스가 **가장 중요한 위험요인**이다" | '중요도'가 인과처럼 읽힌다 |
| ✅ | "이전 스트레스는 **예측에 가장 크게 기여**했다" | 예측 기여 — 우리가 할 수 있는 말 |

그리고 **개선폭 +.029 를 어떻게 볼 것인가**도 정직하게 말해야 한다.
"가장 강한 예측변수"인데도 AUC 는 .65 → .68 밖에 안 올랐다. 이것은 두 가지를 뜻한다:

1. 1년 전 스트레스를 알아도 **1년 뒤를 잘 맞히지는 못한다** (3차시에서 상관 .31 을 본 그대로)
2. 뒤집어 말하면 — **중2 때 힘들었던 학생이 중3 때도 힘들 것이 정해져 있지 않다.**
   이건 개입 연구의 관점에서는 **희망적인 결과**다."""),

md("""## 💾 다음 차시를 위해 — 드라이브에 저장\n\n오늘 만든 것 중 **다음 차시가 재료로 쓰는 파일**을 내 드라이브(`program5_state/`)에 넣어 둔다.\n이렇게 해 두면 런타임이 끊겨도, 다른 컴퓨터에서 열어도 **다음 차시가 그냥 시작된다.**\n\n> 🔴 파생 파일이 들어가는 폴더다 — **개인 계정 안에만** 두고 링크 공유·양도하지 않는다."""),
code(handoff_out(push=['reports/figures/*.png'], note="5차시 산출물")),

md("""## 🎯 회고 (5분)

1. 표준화를 안 하고 계수를 비교하면 무슨 일이 생기는가 — `bullying` 을 예로 설명한다면?
2. 계수 −0.257 을 "확률 6%p 감소"라고 쓰면 왜 불완전한 문장인가? (Step 2 ②)
3. 자아탄력성은 단순 상관 −.17, 다변량 계수 +.16 이다. **둘 다 맞는 말인가?**
   그리고 이 +.16 을 논문에 "자아탄력성은 위험요인"이라고 쓸 수 있나?
4. VIF 최댓값이 2.45 로 기준(5)을 통과했다. 그런데도 왜 계수를 믿을 수 없었나?
5. 계수 순위표 18줄 중 **3줄만 해석**하기로 했다. 나머지 15줄은 왜 못 쓰나?

5번이 오늘의 핵심 감각이다 — **모른다고 말하는 것도 결과다.**

## 📝 과제
- 내가 고른 변수 하나에 대해 **계수 · 오즈비 · 95% 구간**을 적고, 해석 가능/불가 판정
- "자아탄력성이 위험요인으로 나왔다"는 문장이 왜 틀렸는지 **한 문단**으로 설명
- Model A/B 비교 결과를 **인과 주장 없이** 3문장으로 서술 (위 ✅/❌ 표 참고)

## ▶️ 다음 (6차시)
> "오늘 우리는 **직선**을 그었다 — 로지스틱 회귀는 '자아존중감이 늘수록 로그오즈가
> 일정하게 준다'고 가정한다. 그런데 마음이 정말 그렇게 작동할까?
> 자아존중감이 아주 낮은 구간에서만 위험이 급격히 커지는 것이라면?
> 다음 주엔 **결정 트리와 랜덤 포레스트**로 그 가정을 깨 본다.
> 그리고 확인하게 될 것이다 — **복잡한 모델이 늘 더 좋은 건 아니다.**"""),
]

os.makedirs("session5", exist_ok=True)
save(cells, "session5/session5.ipynb")

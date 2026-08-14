# -*- coding: utf-8 -*-
"""session5.ipynb 빌더 — 로지스틱 회귀와 계수 해석.

5차시는 "숫자를 해석하는 법"이 아니라 **"해석해도 되는 숫자와 안 되는 숫자를 가르는 법"**
을 배우는 차시다. 계수 순위표를 만들어 놓고, 부트스트랩으로 그 표의 대부분이
해석 불가임을 학생이 직접 확인한다.

★ 오늘도 test 는 열지 않는다. 계수 해석은 train 에 적합한 모델로 한다.

실측 근거 (frame 1,321 · train 1,056 · seed 42 · Model A 18변수):
  표준화 전/후 계수 순위: 18개 중 14개가 바뀐다 (SD 최소 .217 ~ 최대 .868, 4배)
  계수 상위: peer_support -.257 · self_esteem -.254 · parenting_monitoring -.251 · depression +.189
  단순상관 대비 부호 뒤집힘 7개 (ego_resilience -.170 → +.161 등)
  ego_resilience 추적: 혼자 -.327 → +self_esteem -.069 → +peer_support +.060 → 18개 +.161
  VIF 최대 2.45 (임계치 미만인데도 부호가 뒤집힌다)
  부트스트랩 500회: 뒤집힌 7개는 7/7 이 신뢰구간에 0 포함
    신뢰구간이 0 을 제외하는 변수는 단 3개 — peer_support · self_esteem · parenting_monitoring
  Model A CV AUC .6535 (C=0.1) / Model B .6825 (C=0.1) · prev_stress 계수 +.448
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

1. **로지스틱 회귀**가 확률을 어떻게 다루는지, 계수가 무슨 뜻인지 설명한다.
2. **표준화**가 왜 필요한지 보인다 — 3차시의 그 숙제가 오늘 돌아온다. ← 고비 1
3. **단순 상관과 다변량 계수가 왜 다른지**(부호까지 뒤집힌다) 설명한다. ← 고비 2
4. **부트스트랩**으로 계수의 불확실성을 재고, 해석 가능한 것만 골라낸다.

> 🔒 **오늘도 test 는 열지 않는다.** 계수 해석은 train(1,056명)에 적합한 모델로 하고,
> 성능은 train 안 5-fold CV 로만 본다."""),

md("""## 🗺️ 오늘의 위치 — 5차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1~2 ✅ | 문화적응 스트레스 · 심리척도 · 역채점 | feature/target · pandas · ID join |
| 3 ✅ | 분포 · 상관 · Cronbach α | 집계 · 시각화 · 클리닝 |
| 4 ✅ | 조작적 정의 · 임상 cut-off 와의 차이 | split · 불균형 · **데이터 누출** |
| **5 (오늘)** | **예측변수와 결과의 관계·방향성** | **로지스틱 회귀 · 확률 · 계수 · 표준화** |
| 6 | 심리 특성은 선형적으로 작동하는가 | Decision Tree · Random Forest · 과적합 |
| 7~8 | 위험/보호요인 · 인과 vs 예측 → 보고 | Permutation Importance · 재현성 |

**오늘의 재료** — 4차시가 만든 것 그대로다.

- `modeling_frame.parquet` (1,321행) + 4차시의 `high_stress` 라벨 · train/test 분할
- `maps_risk.evaluation` — 표준화 계수 · **부트스트랩 계수 안정성**

> 🔴 오늘의 규칙: **"계수 하나만 보고 해석하지 않는다. 불확실성을 옆에 같이 둔다."**"""),

md("""## Step 0 — 재료 확인: 4차시 상태를 그대로 재현한다"""),
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

md("""## Step 1 — 왜 선형회귀가 아니라 로지스틱인가

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
아무리 작은 값이 들어와도 0 아래로 안 내려간다."""),

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
print("\n※ 로그오즈 0 = 확률 0.5 (반반). 양수면 0.5 보다 높고, 음수면 낮다.")'''),

md("""## Step 2 — 계수 읽는 법: 로그오즈 → 오즈비 → 확률

`b1 = -0.26` 같은 계수가 나왔을 때, 이것을 어떻게 읽나? **세 단계**로 번역한다.

| 단계 | 무엇 | 읽는 법 |
|---|---|---|
| **① 로그오즈 계수** | `-0.26` | x가 1단위 늘면 로그오즈가 0.26 **줄어든다** — 부호만 직관적이다 |
| **② 오즈비(odds ratio)** | `e^(-0.26) = 0.77` | 오즈가 **0.77배**가 된다 (23% 감소) |
| **③ 확률** | 상황에 따라 다름 | 같은 계수라도 **출발점에 따라 확률 변화폭이 다르다** |

**오즈(odds)** 가 뭔가: 확률이 0.75 라면 오즈는 `0.75 / 0.25 = 3` — "일어날 가능성이
안 일어날 가능성의 3배"라는 뜻이다. 경마나 도박에서 쓰는 그 배당률과 같은 개념이다.

> ⚠️ ③이 까다롭다. 시그모이드는 **곡선**이라 같은 크기의 계수라도
> 확률 0.5 근처에서는 변화가 크고, 0.05 나 0.95 근처에서는 변화가 작다.
> 그래서 "계수 −0.26 = 확률 몇 % 감소"라고 **한 문장으로 말할 수 없다.**"""),

md("""## Step 3 — 표준화: 3차시의 숙제가 돌아왔다 ⚠️ (첫 봉우리)

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

> 우리 파이프라인은 이미 그렇게 돼 있다: `make_preprocessor(scale=True)` 안의 `StandardScaler`.
> 그리고 4차시에 배운 대로 **Pipeline 안에 있으므로 train 에서만 fit** 된다."""),

code(r'''# TODO: 표준화를 켠 모델과 끈 모델을 각각 만들어 계수 순위를 비교하라
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from maps_risk.preprocessing import make_preprocessor

def logit(scale, C=1.0):
    return Pipeline([("prep", make_preprocessor(scale=_____)),      # ← scale 인자를 넘겨라
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=cfg["random_seed"], C=C))])

m_raw = logit(False).fit(Xtr, ytr)      # 원래 단위 그대로
m_std = logit(True).fit(Xtr, ytr)       # 표준화 후

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
    print(f"✅ PASS — 18개 중 {n_changed}개의 순위가 바뀐다. 1위도 {top_raw} → {top_std} 로 달라졌다.")
    print("   표준화 없이 계수를 비교하면 **자를 섞어 쓴 것**이다. 3차시의 그 문제 그대로다.")
    print("   ※ 예측 성능은 거의 안 바뀐다. 표준화는 성능이 아니라 **해석**을 위한 것이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: make_preprocessor(scale=scale) — 함수 인자를 그대로 넘긴다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
Pipeline([("prep", make_preprocessor(scale=scale)), ...])
```

실측: **18개 중 14개**의 순위가 바뀐다. 1위도 `self_esteem`(원단위) → `peer_support`(표준화)로 달라진다.

가장 극적인 것은 `bullying`(집단괴롭힘)이다. 원단위 계수로는 **7위**인데 표준화하면 **12위**로
떨어진다. 왜? SD 가 **0.217** 로 매우 작기 때문이다(3차시에서 본 왜도 +7.9, "거의 전원이 1점").
"1점 늘어날 때"의 효과가 커 보였지만, **실제로 1점 늘어나는 학생이 거의 없다.**

> **표준화는 성능을 위한 것이 아니라 해석을 위한 것이다.** AUC 는 거의 그대로다.
> 하지만 "어떤 변수가 더 중요한가"라는 질문에 답하려면 반드시 필요하다.
</details>"""),

md("""## Step 4 — 계수 순위표: 오늘 만들려던 그것

이제 표준화된 계수로 순위표를 만든다. 이것이 오늘 우리가 원했던 답이다 —
**"어떤 변수가 고스트레스 분류와 가장 강하게 관련되는가."**"""),

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

md("""## Step 5 — 부호가 뒤집힌다 🔴 (두 번째 봉우리)

순위표를 3차시의 **단순 상관**과 나란히 놓아 보자. 같은 데이터, 같은 변수, 같은 target 이다.
당연히 비슷해야 할 것 같은데 —"""),

code(r'''# TODO: 단순 상관을 계산해 계수와 나란히 놓고, 부호가 다른 변수를 세어라
r_biv = Xtr.apply(lambda col: col.corr(frame.loc[idx_tr, "acculturative_stress_w6"]))

cmp = pd.DataFrame({"단순상관_r": r_biv, "표준화계수": c})
cmp["부호"] = np.where(np.sign(cmp["단순상관_r"]) == np.sign(cmp["표준화계수"]),
                      "같음", "🔴 뒤집힘")
cmp = cmp.reindex(c.abs().sort_values(ascending=False).index)
print(cmp.round(3).to_string())

n_flip = int((cmp["부호"] == "🔴 뒤집힘").sum())     # ← 몇 개인가
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
print("  peer_support 까지 넣자 +0.060 으로 **부호가 뒤집힌다.**")'''),

md("""### 왜 이런 일이 생기나 — 계수의 진짜 뜻

**단순 상관과 회귀 계수는 서로 다른 질문에 답한다.**

| | 질문 |
|---|---|
| **단순 상관** | "자아탄력성이 높은 학생은 스트레스가 낮은가?" → 그렇다 (r = −.17) |
| **회귀 계수** | "**자아존중감·친구지지·교우관계가 똑같은 두 학생** 중, 자아탄력성이 더 높은 쪽은?" → 오히려 조금 높다 (+.16) |

회귀 계수는 항상 **"다른 변수를 통제했을 때"** 의 관계다. 그런데 3차시에서 봤듯
자아탄력성은 자아존중감(r = .58)·친구지지(r = .57)와 강하게 얽혀 있다.
**공통 부분을 다른 변수들이 먼저 가져가고 남은 것**이 자아탄력성의 계수가 된다.
그 나머지가 무엇을 뜻하는지는 — **솔직히 말해서 해석하기 매우 어렵다.**

> 흥미로운 사실: 이 데이터의 **VIF(분산팽창지수)는 최대 2.45** 로,
> 보통 쓰는 경고 기준(5 또는 10)에 한참 못 미친다.
> **"다중공선성 검사를 통과했으니 안심"이 아니다** — 기준을 넘지 않아도 부호는 뒤집힌다."""),

md("""## Step 6 — 그래서 믿어도 되나: 불확실성을 잰다 🔍

여기서 멈추면 안 된다. 순위표에는 숫자가 **하나씩**만 적혀 있는데, 그 숫자가
**얼마나 흔들리는지**를 우리는 아직 모른다.

**부트스트랩(bootstrap)** 으로 잰다. 방법은 놀랄 만큼 단순하다:

```
train 1,056명 중에서 1,056명을 "중복 허용"으로 다시 뽑는다  → 모델 학습 → 계수 기록
이것을 500번 반복한다  →  계수 500개의 분포를 본다
```

같은 모집단에서 표본을 다시 뽑았다면 어떤 값이 나왔을까를 **흉내 내는** 것이다.
계수 500개가 좁게 모여 있으면 안정적이고, 넓게 퍼져 있으면 그 계수는 **운이 좌우한 값**이다.

특히 **95% 구간이 0을 포함하는지**를 본다. 포함한다면 —
"이 데이터로는 **방향조차 단정할 수 없다**"는 뜻이다."""),

code(r'''# TODO: 부트스트랩으로 계수의 불확실성을 재라 (500회, 30초쯤 걸린다)
boot = evaluation.bootstrap_coefficients(
    logit(True),               # 아직 fit 하지 않은 Pipeline 을 넘긴다
    Xtr, ytr,
    n_boot=_____,              # ← 500
    seed=0)

print(boot.round(3).to_string(index=False))
stable = boot.loc[~boot["includes_zero"], "feature"].tolist()
print(f"\n신뢰구간이 0 을 포함하지 않는(= 해석 가능한) 변수: {stable}")'''),
code(r'''# CHECK Step6
try:
    assert len(boot) == len(featsA), "모든 변수가 나와야 한다"
    assert len(stable) <= 5, f"해석 가능한 변수는 소수여야 한다 (지금 {len(stable)}개)"
    flipped = cmp.index[cmp["부호"] == "🔴 뒤집힘"].tolist()
    zero_in = boot.set_index("feature").loc[flipped, "includes_zero"]
    assert zero_in.all(), "부호가 뒤집힌 변수는 전부 신뢰구간이 0 을 포함해야 한다"
    print(f"✅ PASS — 18개 중 해석 가능한 것은 **{len(stable)}개뿐**이다: {', '.join(stable)}")
    print(f"   그리고 부호가 뒤집혔던 {len(flipped)}개는 **{int(zero_in.sum())}/{len(flipped)} 전부** 신뢰구간에 0 을 포함한다.")
    print("   → 부호 뒤집힘은 '발견'이 아니라 **불안정하다는 경고**였다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: n_boot=500")'''),
md("""<details><summary>💡 힌트 / 정답 — 오늘의 결론</summary>

```python
n_boot=500
```

**신뢰구간이 0을 제외하는 변수는 단 3개다:**

| 변수 | 계수 | 95% 구간 | 부호 일관성 |
|---|---|---|---|
| `peer_support` (친구지지) | −.257 | [−.475, −.071] | 99.6% |
| `self_esteem` (자아존중감) | −.254 | [−.475, −.066] | 99.2% |
| `parenting_monitoring` (부모 감독) | −.251 | [−.420, −.087] | 100% |

**세 개 다 음수 — 전부 보호요인이다.** 그리고 심리학적으로 말이 된다.

반면 **부호가 뒤집혔던 7개는 7개 전부** 신뢰구간이 0을 포함한다.
`ego_resilience` 의 +0.161 은 [−.052, +.416] — **0을 한가운데 두고 있다.**
"자아탄력성이 위험요인"이라는 해석은 **절대 하면 안 된다.**

주의할 것이 하나 더 있다. **`depression`(우울)도 0을 포함한다** ([−.033, +.420]).
3차시 단순 상관에서는 2위(r = .246)였고 계수 순위로도 4위인데, **다변량에서는
단정할 수 없다.** 순위표 4위까지 해석하고 싶은 유혹이 정확히 여기서 꺾인다.

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
2. 자아탄력성은 단순 상관 −.17, 다변량 계수 +.16 이다. **둘 다 맞는 말인가?**
   그리고 이 +.16 을 논문에 "자아탄력성은 위험요인"이라고 쓸 수 있나?
3. 계수 순위표 18줄 중 **3줄만 해석**하기로 했다. 나머지 15줄은 왜 못 쓰나?

3번이 오늘의 핵심 감각이다 — **모른다고 말하는 것도 결과다.**

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

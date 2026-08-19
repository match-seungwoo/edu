# -*- coding: utf-8 -*-
"""session6.ipynb 빌더 — 결정 트리 · 랜덤 포레스트 · 과적합.

6차시는 "복잡한 모델이 늘 더 좋은 건 아니다"를 **세 가지 서로 다른 증거**로 보인다.
값싼 결론("트리가 로지스틱보다 나쁘다")으로 끝내지 않고, 포레스트가 실제로는
일관되게 이긴다는 사실까지 인정한 뒤 **비용-편익 판단**으로 착지한다.

★ 오늘도 test 는 열지 않는다. 선택이 끝나는 날이지만, 바로 그래서 안 연다.

실측 근거 (frame 1,321 · train 1,056 · seed 42 · modeling.yaml 그리드):
  구간별 위험률 — self_esteem .503→.316→.248→.260→.220 (역치형)
                depression  .226→.241→.369→.430→.464 (단조)
  트리 깊이별 train/CV — 1: .6199/.6067 · 2: .6637/.6355 · 3: .6937/.6151
                        5: .7816/.5892 · 8: .9127/.5312 · None: 1.0000/.5185 (리프 299)
  포레스트 깊이별 — 3: .7368/.6651 · 5: .8507/.6589 · 8: .9823/.6516 · None: 1.0000/.6421
  Model A: Dummy .5000 · Logistic .6535(C=.1) · Tree .6355(d=2) · Forest .6651(d=3)
  Model B: Dummy .5000 · Logistic .6825(C=.1) · Tree .6833(d=3) · Forest .6987(d=3)
  폴드별 AUC(A) — Logistic 0.591 0.679 0.639 0.663 0.695 (평균 .6535)
                  Forest   0.603 0.683 0.648 0.676 0.715 (평균 .6651)
  Forest−Logistic(A): 폴드별 +.012 +.004 +.009 +.013 +.020 → 5/5 승, 평균 +.0116, SD .0053
                      CV seed 7개 중 6개 Forest 승 → 노이즈가 아니라 '작지만 일관된' 차이
"""
import os

from nb import md, code, save, SETUP, handoff_in, handoff_out

cells = [
md("""# 6차시 — 복잡한 모델이 늘 더 좋은 건 아니다

### 결정 트리 · 랜덤 포레스트 · 과적합 · 교차검증

> **오늘 한 문장:** "5차시에 우리는 **직선**을 그었다. 오늘은 그 가정을 깨고
> **구부러진 모델**을 써 본다 — 그리고 그게 얼마나 도움이 되는지(혹은 안 되는지) 잰다."

5차시 마지막에 이렇게 물었다:

> "자아존중감이 **아주 낮은 구간에서만** 위험이 급격히 커지는 것이라면?"

오늘 확인한다 — **그런 구간이 실제로 데이터에 있다.** 그런데 그걸 잡아내는 모델이
반드시 이기는 것도 아니다. 오늘은 그 미묘한 사실을 **세 가지 증거**로 배운다.

오늘의 목표 4가지:

1. 로지스틱 회귀의 **숨은 가정(선형성)** 을 드러내고, 데이터가 그 가정을 따르는지 확인한다.
2. **결정 트리**가 어떻게 작동하는지 읽고, 트리가 발견한 **상호작용**을 해석한다.
3. **과적합**을 직접 만든다 — train AUC 1.0, CV AUC 0.52. ← 고비 1
4. 4개 모델을 정면 비교하고 **"복잡한 모델이 늘 더 좋은 건 아니다"** 를 판정한다. ← 고비 2

> 🔒 **오늘도 test 는 열지 않는다.** 오늘은 모델 **선택**이 끝나는 날이다.
> 바로 그래서 안 연다 — 훔쳐보고 싶은 유혹이 가장 큰 날이기 때문이다."""),

md("""## 🗺️ 오늘의 위치 — 6차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1~3 ✅ | 척도 · 역채점 · 분포 · 상관 · α | pandas · join · 시각화 |
| 4 ✅ | 조작적 정의 · 임상 cut-off 와의 차이 | split · 불균형 · **데이터 누출** |
| 5 ✅ | 예측변수와 결과의 관계·방향성 | 로지스틱 · 계수 · 표준화 · 부트스트랩 |
| **6 (오늘)** | **심리 특성은 선형적으로 작동하는가** | **Decision Tree · Random Forest · 과적합 · CV** |
| 7 | 위험요인·보호요인 · 인과 vs 예측 | Permutation Importance · 오류 분석 |
| 8 | 결론 · 한계 · 윤리 서술 | 재현성 · **test 최종 1회** |

**오늘의 재료** — 4·5차시와 완전히 같은 데이터·라벨·분할이다.

- `modeling_frame.parquet` · `high_stress` 라벨(cutoff 1.500) · train 1,056 / test 265(봉인)
- `configs/modeling.yaml` 의 **하이퍼파라미터 그리드** — 숫자를 코드에 쓰지 않는다
- `maps_risk.models.build_models()` — Dummy / Logistic / Tree / Forest 넷

> 🔴 오늘의 규칙: **"성능 표에서 1등을 찾지 말고, 1등과 2등의 차이가 무엇을 사게 하는지 물어라."**"""),

md("""## Step 0 — 재료 확인"""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(handoff_in(pull=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], require=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], hint="지난 차시 노트북 맨 끝의 '드라이브에 저장' 셀을 실행하면 여기서 자동으로 복원된다")),
code(r'''# 4·5차시와 똑같은 상태를 재현한다 (seed 고정 — 한 명도 다르지 않다)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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

featsA, featsB = split_features(frame, "A"), split_features(frame, "B")
Xtr, ytr = frame.loc[idx_tr, featsA], frame.loc[idx_tr, "high_stress"]

# ★ 모든 모델이 '같은' 폴드로 평가되어야 비교가 성립한다 — cv 객체를 하나만 만들어 돌려쓴다
cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["random_seed"])

print(f"train {len(idx_tr)} · test {len(idx_te)}(봉인) · cutoff {cutoff:.3f} · 양성 {ytr.mean():.1%}")
print(f"Model A {len(featsA)}변수 · Model B {len(featsB)}변수 · {cfg['cv']['folds']}-fold CV (seed {cfg['random_seed']})")
print("\nmodeling.yaml 그리드:")
for k in ("decision_tree", "random_forest", "logistic_regression"):
    print(f"  {k:20s} {cfg['models'][k]}")'''),

md("""## Step 1 — 로지스틱의 숨은 가정: 마음은 직선인가

5차시에 우리가 쓴 로지스틱 회귀는 이렇게 생겼다:

```
로그오즈 = b0 + b1·자아존중감 + b2·친구지지 + …
```

이 식에는 **말하지 않은 가정**이 하나 있다.
**"자아존중감이 1점에서 2점으로 오를 때의 효과 = 3점에서 4점으로 오를 때의 효과"**
— 어느 구간에서든 **똑같은 만큼** 로그오즈가 변한다는 가정이다. 이것을 **선형성 가정**이라 한다.

심리학적으로 이게 그럴듯한가? 아마 아닐 것이다.
자아존중감이 바닥인 학생에게 1점의 차이는 클 것이고,
이미 높은 학생들 사이의 1점 차이는 별 의미가 없을 수 있다.

**확인하는 방법은 간단하다.** 변수를 5구간(5분위)으로 나눠, 각 구간의 **실제 고스트레스 비율**을
세어 본다. 비율이 계단처럼 **일정하게** 변하면 선형에 가깝고, 어느 구간에서 **뚝 떨어지면**
비선형이다."""),

code(r'''# ▶ 변수를 5분위로 나눠 구간별 실제 고스트레스 비율을 세어라
import matplotlib.pyplot as plt
from maps_risk import plots      # import 만 해도 한글 폰트가 잡힌다

show = ["self_esteem", "depression", "previous_acculturative_stress", "peer_support"]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
for ax, v in zip(axes, show):
    col = frame.loc[idx_tr, v]
    q = pd.qcut(col, 5, labels=False, duplicates="drop")   # ← 5분위로 나눈다
    rate = ytr.groupby(q).mean()                            # 0/1 의 평균 = 그 구간의 고스트레스 비율
    ax.plot(rate.index, rate.values, "o-")
    ax.axhline(ytr.mean(), color="gray", ls="--", lw=.8)
    ax.set_title(v, fontsize=10); ax.set_xlabel("5분위 (낮음→높음)"); ax.set_ylim(0, .65)
    print(f"  {v:30s} " + " → ".join(f"{r:.3f}" for r in rate))
axes[0].set_ylabel("고스트레스 비율")
fig.tight_layout(); plt.show()'''),
code(r'''# CHECK Step1
try:
    se = ytr.groupby(pd.qcut(frame.loc[idx_tr, "self_esteem"], 5, labels=False, duplicates="drop")).mean()
    dp = ytr.groupby(pd.qcut(frame.loc[idx_tr, "depression"], 5, labels=False, duplicates="drop")).mean()
    drop1 = se.iloc[0] - se.iloc[1]          # 1분위 → 2분위 낙폭
    drop_rest = se.iloc[1] - se.iloc[-1]     # 2분위 → 5분위 낙폭
    assert drop1 > drop_rest, "자아존중감은 첫 구간에서 가장 크게 떨어져야 한다"
    assert dp.is_monotonic_increasing, "우울은 단조 증가여야 한다"
    print(f"✅ PASS — 자아존중감: 1→2분위에서 {drop1:.3f} 떨어지고, 2→5분위 전체에서 {drop_rest:.3f}밖에 안 떨어진다.")
    print("   **역치 효과(threshold effect)** 다 — 낮은 쪽에서만 위험이 급등하고 위쪽은 평평하다.")
    print("   반면 우울은 단조 증가 — 직선 가정이 잘 맞는 변수도 있다. 변수마다 다르다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 0/1 라벨의 '평균'이 곧 그 구간의 고스트레스 비율이다 → .mean()")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

```python
rate = ytr.groupby(q).mean()
```

0/1 라벨의 **평균**이 곧 그 구간의 고스트레스 **비율**이다.

실측:

| 변수 | 1분위 → 5분위 | 모양 |
|---|---|---|
| `self_esteem` | .503 → .316 → .248 → .260 → **.220** | **역치형** — 1분위에서만 급락, 그 뒤 평평 |
| `depression` | .226 → .241 → .369 → .430 → **.464** | **단조 증가** — 직선에 가깝다 |
| `previous_acculturative_stress` | .218 → .206 → .305 → .462 → **.558** | 단조, 뒤로 갈수록 **가팔라짐** |
| `peer_support` | .509 → .396 → .189 → .234 | **비단조** — 3분위에서 최저, 4분위에서 되오름 |

**5차시의 예고가 맞았다.** 자아존중감은 낮은 구간에서만 위험이 급등하고 그 위로는 거의 평평하다.
직선 하나로는 이 모양을 표현할 수 없다.

> 그런데 주의: **"비선형이 존재한다"와 "비선형 모델이 이긴다"는 다른 말이다.**
> 오늘 그 차이를 확인하게 된다.
</details>"""),

md("""## Step 2 — 결정 트리: 스무고개로 분류하기

**결정 트리(Decision Tree)** 는 직선을 긋는 대신 **질문을 던진다.**

```
친구지지가 4.07 이하인가?
 ├─ 예  →  자아존중감이 2.88 이하인가? → …
 └─ 아니오 → 우울이 1.25 이하인가? → …
```

스무고개와 똑같다. 각 질문마다 데이터가 둘로 갈리고, 마지막 칸(**잎, leaf**)에 도달하면
그 칸에 속한 학생들의 고스트레스 비율로 확률을 매긴다.

트리의 장점 두 가지:

1. **비선형을 자동으로 잡는다.** "자아존중감 2.88 이하"라는 **역치**를 스스로 찾는다.
2. **상호작용(interaction)** 을 잡는다 — "A인 학생들 사이에서는 B가 중요하고,
   A가 아닌 학생들 사이에서는 C가 중요하다"는 구조를 표현할 수 있다.
   로지스틱 회귀는 이런 걸 **직접 적어 주지 않으면** 못 잡는다."""),

code(r'''# 깊이 2짜리 아주 작은 트리를 만들어 구조를 읽어 본다 (Model B — 이전 스트레스 포함)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from maps_risk.preprocessing import make_preprocessor

def build(clf, scale=False):
    """트리 계열은 표준화가 필요 없다 — 하지만 결측 대치는 필요하므로 Pipeline 은 그대로 쓴다."""
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("clf", clf)])

tree_b = build(DecisionTreeClassifier(max_depth=2, class_weight="balanced",
                                      random_state=cfg["random_seed"])).fit(frame.loc[idx_tr, featsB], ytr)
print(export_text(tree_b.named_steps["clf"], feature_names=list(featsB), decimals=2))'''),

md("""### Step 2 해석 — 트리가 찾아낸 것

출력을 읽으면 이런 구조다:

```
이전 스트레스 ≤ 1.45 ?
 ├─ 예 (이전에 스트레스가 낮았던 학생들)
 │    └─ 자아존중감 ≤ 2.62 ?  →  예: 고스트레스 / 아니오: 일반
 └─ 아니오 (이전에 스트레스가 높았던 학생들)
      └─ 우울 ≤ 1.25 ?        →  예: 일반 / 아니오: 고스트레스
```

**첫 질문이 `previous_acculturative_stress` 다.** 5차시에서 계수 1위였던 그 변수를
트리도 독립적으로 첫 번째로 골랐다 — 서로 다른 방법이 같은 결론에 도달했다.

더 흥미로운 건 **두 번째 층**이다. 트리는 두 집단에게 **서로 다른 질문**을 던진다:

> **이전에 스트레스가 낮았던 학생들** 사이에서는 → **자아존중감**이 갈림길이고,
> **이전에 스트레스가 높았던 학생들** 사이에서는 → **우울**이 갈림길이다.

이것이 **상호작용**이다. 로지스틱 회귀는 "자아존중감 계수 하나, 우울 계수 하나"만 주기 때문에
**이 구조를 말해 주지 못한다.** 트리가 진짜로 기여하는 지점이 여기다.

> ⚠️ 단, 이 구조는 **train 데이터 하나**에서 나온 것이다. 트리는 데이터가 조금만 바뀌어도
> 분기가 통째로 달라지는 **불안정한** 모델이다 (5차시의 부트스트랩과 같은 문제).
> 그래서 이 그림은 **가설**로 읽고, 확정된 발견으로 읽지 않는다."""),

md("""## Step 3 — 과적합: 트리를 자라게 두면 ⚠️ (첫 봉우리)

트리는 질문을 계속 던질 수 있다. 깊이 제한을 풀면 **모든 학생을 하나씩 다른 칸에** 넣을 때까지
자란다. 그러면 train 데이터는 **100% 맞힌다.**

4차시에서 이 숫자를 본 적이 있다 — **AUC 1.0**. 그때는 **누출** 때문이었다.
오늘은 다른 병으로 같은 숫자가 나온다: **과적합(overfitting)**.

> **과적합** = 훈련 데이터의 **우연한 무늬까지 외워 버려서**, 새 데이터에서는 오히려 못 맞히는 것.
> 4차시의 기출문제 비유를 다시 쓰면 — **기출문제 100개를 통째로 암기한 학생**이다.

깊이를 1부터 끝까지 늘려 가며, **train 점수와 CV 점수를 나란히** 본다."""),

code(r'''# ▶ 깊이별로 train 점수와 CV 점수를 함께 재라
from sklearn.model_selection import cross_validate

rows = []
for d in (1, 2, 3, 4, 5, 8, 12, None):
    est = build(DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                       random_state=cfg["random_seed"]))
    r = cross_validate(est, Xtr, ytr, cv=cv, scoring="roc_auc",
                       return_train_score=True)           # train 점수도 함께 받는다
    n_leaves = est.fit(Xtr, ytr).named_steps["clf"].get_n_leaves()
    rows.append({"max_depth": str(d), "train_AUC": r["train_score"].mean(),
                 "CV_AUC": r["test_score"].mean(),
                 "차이": r["train_score"].mean() - r["test_score"].mean(), "리프수": n_leaves})

depth_tbl = pd.DataFrame(rows)
print(depth_tbl.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(range(len(depth_tbl)), depth_tbl["train_AUC"], "o-", label="train (외운 것)")
ax.plot(range(len(depth_tbl)), depth_tbl["CV_AUC"], "s-", label="CV (새 데이터)")
ax.set_xticks(range(len(depth_tbl)), depth_tbl["max_depth"])
ax.set_xlabel("트리 깊이"); ax.set_ylabel("AUC"); ax.legend(); ax.set_title("깊어질수록 벌어진다 — 과적합")
fig.tight_layout(); plt.show()'''),
code(r'''# CHECK Step3
try:
    best_row = depth_tbl.loc[depth_tbl["CV_AUC"].idxmax()]
    deep = depth_tbl[depth_tbl["max_depth"] == "None"].iloc[0]
    assert deep["train_AUC"] > 0.99, "제한 없는 트리는 train 을 거의 완벽히 맞혀야 한다"
    assert deep["CV_AUC"] < 0.56, "그런데 CV 는 동전 던지기 수준으로 떨어져야 한다"
    assert best_row["max_depth"] in ("2", "3"), f"CV 최고는 얕은 트리여야 한다 (지금 {best_row['max_depth']})"
    print(f"✅ PASS — 깊이 제한 없음: train AUC {deep['train_AUC']:.4f} · CV AUC {deep['CV_AUC']:.4f} "
          f"(리프 {int(deep['리프수'])}개)")
    print(f"   CV 최고는 깊이 {best_row['max_depth']} 에서 {best_row['CV_AUC']:.4f} — **가장 단순한 축**이다.")
    print("   → train 점수는 깊어질수록 계속 오른다. 그런데 CV 는 어느 지점부터 **내려간다.**")
    print("   4차시의 AUC 1.0 은 누출이었고, 오늘의 1.0 은 과적합이다 — 병은 다르지만 증상은 같다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: cross_validate(..., return_train_score=True)")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

```python
return_train_score=True
```

실측:

| 깊이 | train AUC | CV AUC | 차이 | 리프 수 |
|---|---|---|---|---|
| 1 | .6199 | .6067 | .013 | 2 |
| **2** | .6637 | **.6355** | .028 | 4 |
| 3 | .6937 | .6151 | .079 | 8 |
| 5 | .7816 | .5892 | .192 | 30 |
| 8 | .9127 | .5312 | .382 | 96 |
| **없음** | **1.0000** | **.5185** | **.482** | **299** |

**train 점수는 끝까지 오르고, CV 점수는 깊이 2에서 꺾인다.** 이 그림이 과적합의 표준 형태다.

깊이 제한이 없으면 리프가 **299개** — train 1,056명을 299칸에 나눠 담았다.
train AUC 는 **1.0000** 인데 CV 는 **.5185**, 거의 **동전 던지기**다.

> 🔴 4차시에서 배운 문장이 여기서 다시 쓰인다: **"성능이 좋아 보이면 의심하라."**
> 그때는 누출이었고 지금은 과적합이다. **train 점수만 보면 둘 다 못 알아챈다.**
> 그래서 우리는 항상 **CV 점수를 옆에 둔다.**
</details>"""),

md("""## Step 4 — 랜덤 포레스트: 나무 한 그루 대신 숲

트리 하나는 불안정하다 — 데이터가 조금만 바뀌어도 분기가 통째로 달라지고, 깊어지면 외워 버린다.

**랜덤 포레스트(Random Forest)** 의 아이디어는 단순하다:

```
① 데이터를 조금씩 다르게 뽑아 (부트스트랩 — 5차시에 배운 그것!)
② 변수도 일부만 무작위로 골라 주고
③ 트리를 300그루 키운 뒤
④ 300개의 답을 평균 낸다
```

**왜 평균이 나은가.** 3차시의 심리척도 논리와 정확히 같다 —
"한 번의 측정은 흔들리지만, 여러 문항의 평균은 오차가 상쇄돼 안정된다."
트리 한 그루의 우연한 실수들이 300그루를 평균 내면 서로 상쇄된다.

깊이를 늘려 가며 포레스트도 과적합하는지 본다."""),

code(r'''# 포레스트도 깊이를 늘리면 과적합할까?
from sklearn.ensemble import RandomForestClassifier

rows = []
for d in (3, 5, 8, None):
    est = build(RandomForestClassifier(n_estimators=cfg["models"]["random_forest"]["n_estimators"],
                                       max_depth=d, class_weight="balanced",
                                       random_state=cfg["random_seed"], n_jobs=-1))
    r = cross_validate(est, Xtr, ytr, cv=cv, scoring="roc_auc", return_train_score=True)
    rows.append({"max_depth": str(d), "train_AUC": r["train_score"].mean(),
                 "CV_AUC": r["test_score"].mean(),
                 "차이": r["train_score"].mean() - r["test_score"].mean()})
forest_tbl = pd.DataFrame(rows)
print(forest_tbl.round(4).to_string(index=False))

print("\n비교 — 깊이 제한이 없을 때 CV AUC:")
print(f"  단일 트리   : {depth_tbl[depth_tbl['max_depth']=='None']['CV_AUC'].iloc[0]:.4f}   ← 동전 던지기 수준")
print(f"  랜덤 포레스트: {forest_tbl[forest_tbl['max_depth']=='None']['CV_AUC'].iloc[0]:.4f}   ← 훨씬 덜 무너진다")
print("\n→ 포레스트도 train 은 1.0 까지 외운다. 하지만 CV 가 덜 떨어진다 — 평균이 과적합을 완충한다.")'''),

md("""## Step 5 — 4개 모델 정면 비교 🔍 (두 번째 봉우리)

이제 오늘의 본론이다. 네 모델을 **완전히 같은 조건**에서 비교한다:

- 같은 train 데이터(1,056명), 같은 **5개 폴드**(같은 `cv` 객체를 돌려 쓴다)
- 하이퍼파라미터는 `modeling.yaml` 의 그리드에서 **CV 로 선택**한다
- **Dummy 를 반드시 옆에 둔다** (4차시의 규칙)"""),

code(r'''# ▶ modeling.yaml 의 그리드로 네 모델을 튜닝하고 CV 로 비교하라
from sklearn.model_selection import GridSearchCV
from maps_risk.models import build_models

def compare(cols, label):
    X = frame.loc[idx_tr, cols]
    out = []
    for name, (est, grid) in build_models(cfg).items():
        if grid:
            gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, ytr)  # 모든 모델에 같은 폴드!
            chosen, auc = gs.best_params_, gs.best_score_
            fitted = gs.best_estimator_
        else:
            chosen, auc, fitted = {}, cross_validate(est, X, ytr, cv=cv, scoring="roc_auc")["test_score"].mean(), est
        extra = cross_validate(fitted, X, ytr, cv=cv, return_train_score=True,
                               scoring=["roc_auc", "average_precision", "recall", "balanced_accuracy"])
        out.append({"model_set": label, "model": name, "best_params": str(chosen),
                    "cv_roc_auc": round(auc, 4),
                    "cv_average_precision": round(extra["test_average_precision"].mean(), 4),
                    "cv_recall": round(extra["test_recall"].mean(), 4),
                    "cv_balanced_accuracy": round(extra["test_balanced_accuracy"].mean(), 4),
                    "train_roc_auc": round(extra["train_roc_auc"].mean(), 4)})
    return pd.DataFrame(out)

metrics = pd.concat([compare(featsA, "A"), compare(featsB, "B")], ignore_index=True)
print(metrics.to_string(index=False))'''),
code(r'''# CHECK Step5
try:
    A = metrics[metrics.model_set == "A"].set_index("model")["cv_roc_auc"]
    assert A["Dummy"] == 0.5, "Dummy 는 0.5 여야 한다"
    assert A["DecisionTree"] < A["LogisticRegression"], \
        "단일 트리가 로지스틱보다 나빠야 한다 (실측: .6355 < .6535)"
    assert A["RandomForest"] > A["LogisticRegression"], \
        "포레스트는 로지스틱보다 조금 나아야 한다 (실측: .6651 > .6535)"
    gap = A["RandomForest"] - A["LogisticRegression"]
    print("✅ PASS — Model A 결과:")
    print(f"   Dummy {A['Dummy']:.4f} · 로지스틱 {A['LogisticRegression']:.4f} · "
          f"트리 {A['DecisionTree']:.4f} · 포레스트 {A['RandomForest']:.4f}")
    print(f"   ① 단일 트리는 로지스틱보다 **나쁘다** — 더 유연한 모델인데 더 못한다.")
    print(f"   ② 포레스트는 로지스틱보다 낫다 — 그런데 차이가 **{gap:+.4f}** 다.")
    print("   이 두 사실을 어떻게 읽어야 하나? 다음 셀에서 차이가 진짜인지부터 확인한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: GridSearchCV(..., cv=cv) — 모든 모델이 같은 폴드를 써야 비교가 성립한다.")'''),

code(r'''# 포레스트의 +0.012 는 진짜인가, 폴드 운인가? — 폴드별로 뜯어본다
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

log_est = build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=cfg["random_seed"], C=0.1), scale=True)
rf_est = build(RandomForestClassifier(n_estimators=300, max_depth=3, class_weight="balanced",
                                      random_state=cfg["random_seed"], n_jobs=-1))
a = cross_val_score(log_est, Xtr, ytr, cv=cv, scoring="roc_auc")
b = cross_val_score(rf_est, Xtr, ytr, cv=cv, scoring="roc_auc")
print("폴드별 AUC")
print("  로지스틱 : " + " ".join(f"{v:.3f}" for v in a) + f"   평균 {a.mean():.4f}")
print("  포레스트 : " + " ".join(f"{v:.3f}" for v in b) + f"   평균 {b.mean():.4f}")
print("  차이     : " + " ".join(f"{v:+.3f}" for v in b - a) +
      f"   → 포레스트 승 {int((b > a).sum())}/{len(a)} 폴드")

print("\nCV 분할 seed 를 바꿔도 순위가 유지되나?")
for sd in (0, 1, 7, 42, 123, 2024, 777):
    c2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)
    x1 = cross_val_score(log_est, Xtr, ytr, cv=c2, scoring="roc_auc").mean()
    x2 = cross_val_score(rf_est, Xtr, ytr, cv=c2, scoring="roc_auc").mean()
    print(f"  seed={sd:<5d} 로지스틱 {x1:.4f} · 포레스트 {x2:.4f} → {'포레스트' if x2 > x1 else '로지스틱'} ({x2-x1:+.4f})")'''),

md("""### Step 5 해석 — "복잡한 모델이 늘 더 좋은 건 아니다"의 세 가지 증거

실측을 정직하게 정리하면 이렇다.

**증거 ① 단일 트리는 로지스틱보다 나쁘다** (.6355 vs .6535)
트리는 로지스틱보다 **훨씬 유연한** 모델이다. 비선형도 상호작용도 잡을 수 있다.
그런데 **더 못한다.** 유연함은 그 자체로 성능이 아니다 — 유연한 만큼 **흔들리기** 때문이다.

**증거 ② 복잡도를 늘리면 오히려 나빠진다** (Step 3)
깊이 2 → 없음으로 가면 CV AUC 가 .6355 → .5185 로 **떨어진다.**
"모델을 더 강력하게" 가 성능을 보장하지 않는다.

**증거 ③ 포레스트는 이긴다 — 그런데 +0.012 다**
여기서 정직해야 한다. 포레스트는 **5개 폴드 전부에서** 로지스틱을 이겼고,
CV 분할 seed 를 7가지로 바꿔도 6번 이겼다. **이 차이는 폴드 운이 아니다.**

그렇다면 포레스트를 써야 하나? **비용을 같이 봐야 한다.**

| | 로지스틱 회귀 | 랜덤 포레스트 |
|---|---|---|
| CV AUC (Model A) | .6535 | **.6651** (+.012) |
| 변수별 **방향**(+/−) | ✅ 계수 부호로 안다 | ❌ 없다 |
| 변수별 **크기** | ✅ 표준화 계수 | △ 7차시 permutation 으로 일부 |
| **불확실성** | ✅ 부트스트랩 신뢰구간 | ❌ 사실상 불가 |
| 사람이 읽을 수 있나 | ✅ 18줄짜리 표 | ❌ 나무 300그루 |

> 🔴 **오늘의 질문: AUC 0.012 를 위해 5차시의 그 표를 통째로 포기할 것인가?**
>
> 우리 연구 질문은 **RQ2 — "어떤 심리사회적 변수가 상대적으로 중요한가"** 다.
> 목적이 **해석**이므로 우리는 **로지스틱을 주 모델로** 삼고,
> 포레스트는 **"비선형을 넣어도 크게 좋아지지 않았다"는 근거**로 함께 보고한다.
>
> 이것은 **정답이 아니라 선택**이다. 목적이 "최대한 잘 맞히기"였다면 반대로 골랐을 것이다.
> 중요한 건 **고르고, 이유를 적는 것**이다."""),

md("""## Step 6 — Model A vs B: 변수 구성이 바뀌면 순위도 바뀐다

같은 표의 Model B 를 보면 재미있는 일이 벌어진다."""),

code(r'''pivot = metrics.pivot(index="model", columns="model_set", values="cv_roc_auc")
pivot["B−A"] = pivot["B"] - pivot["A"]
print(pivot.round(4).to_string())
print("\n주목: Model A 에서 로지스틱보다 나빴던 단일 트리가, Model B 에서는 거의 같아진다.")
print("      이전 스트레스라는 '강한 단일 변수'가 생기자 트리가 그것을 첫 분기로 잡아 잘 작동한다.")
print("      → 모델의 우열은 고정된 것이 아니라 **어떤 변수를 주느냐에 따라 달라진다.**")'''),

md("""## Step 7 — 산출물, 그리고 왜 지금 test 를 열지 않는가

오늘로 **모델 선택이 끝났다.** 하이퍼파라미터도 정했고 주 모델도 정했다.
교과서적으로는 **바로 지금이 test 를 열 시점**이다. 그런데 우리는 열지 않는다.

이유는 4차시에 배운 것 그대로다:

> test 를 한 번 보면, **그다음 결정이 그 숫자에 영향받는다.**
> 7차시에 변수 중요도를 보다가 "이 변수를 빼면 어떨까" 하는 생각이 들 수 있다.
> 그때 test 점수를 이미 알고 있으면, 그 판단은 **더 이상 깨끗하지 않다.**

그래서 순서를 이렇게 둔다: **6차시 선택 → 7차시 해석 → 8차시에 test 를 딱 한 번.**

> 🔴 **훔쳐보고 싶은 유혹이 가장 큰 날이 바로 오늘이다.** 그래서 오늘 안 여는 것이 규칙이다."""),

code(r'''# 오늘의 산출물 — CV 기준 성능표 (test 점수가 아니다)
import os
os.makedirs("reports", exist_ok=True)
metrics.to_csv("reports/model_metrics_cv.csv", index=False)
print("✅ reports/model_metrics_cv.csv  (모든 숫자는 train 안 5-fold CV 값이다)")

fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.35; order = ["Dummy", "LogisticRegression", "DecisionTree", "RandomForest"]
for i, ms in enumerate(("A", "B")):
    v = metrics[metrics.model_set == ms].set_index("model").loc[order, "cv_roc_auc"]
    ax.bar(np.arange(len(order)) + i*w, v.values, w, label=f"Model {ms}")
ax.axhline(0.5, color="gray", ls="--", lw=.8)
ax.set_xticks(np.arange(len(order)) + w/2, ["Dummy", "Logistic", "Tree", "Forest"])
ax.set_ylabel("CV ROC-AUC"); ax.set_ylim(0.45, 0.75); ax.legend()
ax.set_title("네 모델 · 두 변수 세트 (train 5-fold CV)")
fig.tight_layout(); fig.savefig("reports/figures/model_comparison_cv.png", dpi=150)
plt.show()
print("✅ reports/figures/model_comparison_cv.png")

print(f"\n🔒 test {len(idx_te)}명 — 오늘도 열지 않았다. 8차시에 딱 한 번 연다.")'''),

md("""### ⚠️ 오늘 결과를 볼 때의 정직한 단서 하나

`GridSearchCV` 의 `best_score_` 는 **여러 후보 중 가장 좋았던 값**이다.
후보가 많을수록 "운 좋게 잘 나온 값"이 뽑힐 가능성도 커진다 —
포레스트·트리는 후보가 4개, 로지스틱은 3개, Dummy 는 0개다.

즉 **복잡한 모델 쪽이 아주 약간 유리하게 채점됐다.** 엄밀하게 하려면
**중첩 교차검증(nested CV)** — 튜닝용 CV 안에 평가용 CV 를 한 겹 더 — 을 써야 한다.

이 수업에서는 거기까지 가지 않는다. 대신 **그런 편향이 있다는 사실을 기록**한다.
포레스트의 우위가 +0.012 로 작다는 점을 감안하면, 이 편향은 **결론을 뒤집을 수 있는 크기**다.
8차시 한계 절에 적는다."""),

md("""## 💾 다음 차시를 위해 — 드라이브에 저장\n\n오늘 만든 것 중 **다음 차시가 재료로 쓰는 파일**을 내 드라이브(`program5_state/`)에 넣어 둔다.\n이렇게 해 두면 런타임이 끊겨도, 다른 컴퓨터에서 열어도 **다음 차시가 그냥 시작된다.**\n\n> 🔴 파생 파일이 들어가는 폴더다 — **개인 계정 안에만** 두고 링크 공유·양도하지 않는다."""),
code(handoff_out(push=['reports/model_metrics_cv.csv', 'reports/figures/*.png'], note="6차시 산출물 — 모델 비교표를 7·8차시가 이어받는다")),

md("""## 🎯 회고 (5분)

1. 깊이 제한 없는 트리는 train AUC 가 **1.0** 이었다. 4차시에도 AUC 1.0 을 봤다.
   **두 경우는 어떻게 다른가?** 그리고 **공통점**은 무엇인가?
2. 단일 트리는 로지스틱보다 **유연한데 더 나빴다.** 왜 그런가?
3. 포레스트가 로지스틱보다 **일관되게** 0.012 만큼 낫다. 그런데도 우리는 로지스틱을
   주 모델로 골랐다. **정당한 선택인가?** 어떤 연구 목적이었다면 반대로 골라야 했나?

3번이 오늘의 핵심 감각이다 — **성능 표의 1등이 곧 답이 아니다. 목적이 답을 정한다.**

## 📝 과제
- 깊이별 train/CV 표를 보고 **"과적합이 시작되는 지점"** 을 짚고, 그 근거를 2문장으로
- Model B 깊이 2 트리의 **상호작용**을 심리학적으로 해석 (단, "가설"이라는 단서를 달 것)
- 우리 연구에서 로지스틱을 주 모델로 고른 이유를 **비용-편익 표**로 정리

## ▶️ 다음 (7차시)
> "오늘 모델을 골랐다. 다음엔 그 모델에게 **'무엇을 보고 판단했니?'** 라고 묻는다 —
> **Permutation Importance**. 그리고 5차시의 계수 순위와 비교한다.
> 두 방법이 같은 답을 주면 든든하고, 다르면 그 이유를 설명해야 한다.
> 마지막으로 가장 어려운 질문이 온다 — **이 중요도를 '위험요인'이라고 불러도 되는가?**"""),
]

os.makedirs("session6", exist_ok=True)
save(cells, "session6/session6.ipynb")

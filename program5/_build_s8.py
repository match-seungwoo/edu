# -*- coding: utf-8 -*-
"""session8.ipynb 빌더 — test 최종 1회 · 민감도 분석 · 최종 보고서.

8차시는 마지막 차시다. 4차시부터 봉인해 온 test 265명을 **딱 한 번** 열고,
그 결과를 정직하게 서술한 뒤 재현 가능한 형태로 마무리한다.

이 차시의 교육적 정점은 "성능 숫자"가 아니라 **두 번의 반전**이다:
  ① CV 에서 포레스트가 이겼는데 test 에서는 로지스틱이 이겼다 (6차시의 편향 경고가 실현)
  ② 그런데 그 차이조차 신뢰구간이 0 을 포함한다 → "판별 불가"가 결론
  ③ 7차시의 FN 발견은 재현됐지만, 그것을 완화하던 '경계선' 설명은 재현되지 않았다

실측 근거 (train 1,056 / test 265 · cutoff 1.500 · test 양성 34.3%(91명)):
  test AUC — A: 로지스틱 .6718 [.604,.740] · 포레스트 .6566 [.587,.726] · 트리 .6375 · Dummy .5000
             B: 로지스틱 .7165 [.652,.778] · 포레스트 .7032 · 트리 .6564
  A 로지스틱−포레스트 차이 +.0147 [−.0157,+.0443] · 로지스틱 승 83.9% → 0 포함
  A 로지스틱 test: recall .6484 · precision .4683 · bal_acc .6316 · 혼동행렬 [[107,67],[32,59]]
  RQ3 test: A .6718 → B .7165 (+.045)  (CV 는 +.029)
  민감도: q .70/.75/.80 × >=/> → test AUC .6449~.6718 · 양성률 .173~.343
          q.70 과 q.75 는 동일(동점) · '>1.5' 와 '>=1.6' 도 동일(이산 점수)
  test 오류분석: FN 32명 자아존중감 3.602 (TN 3.460 · TP 2.839) → FN 이 TN 을 닮는 현상 재현
                단 FN 중 경계선(≤1.7) 비율 50.0% vs TP 59.3% → train(75.7 vs 44.5)과 반대, 재현 실패
"""
import os

from nb import md, code, save, SETUP

cells = [
md("""# 8차시 — 봉투를 연다

### test 최종 1회 · 민감도 분석 · 결론 · 한계 · 윤리

> **오늘 한 문장:** "4차시부터 **265명**을 봉인해 뒀다. 오늘 **딱 한 번** 연다 —
> 그리고 그 숫자가 무엇이든 **그대로 보고한다.**"

오늘은 마지막 차시다. 새로 배우는 기법은 거의 없다.
대신 **지금까지 한 모든 것을 검증대에 올린다.**

오늘 두 번의 반전이 있다. 미리 알려 준다:

1. **CV 에서 이겼던 모델이 test 에서 진다.** 6차시에 기록해 둔 경고가 실현된다.
2. **그런데 그 뒤집힘조차 믿을 수 없다.** 신뢰구간이 답을 준다.

오늘의 목표 4가지:

1. **test 를 딱 한 번** 열고, CV 예상과 얼마나 맞는지 확인한다. ← 고비 1
2. 성능 차이를 **신뢰구간**으로 판정한다 — 순위표를 곧이곧대로 읽지 않는다.
3. **민감도 분석**으로 조작적 정의가 결론을 흔드는지 확인한다.
4. 7차시의 발견이 **재현되는지** 확인하고, `final_report.md` 를 쓴다. ← 고비 2

> 🔴 오늘의 규칙: **"결과를 보고 나서 방법을 바꾸지 않는다."**
> 이 규칙을 지켰기 때문에 오늘의 숫자를 믿을 수 있다."""),

md("""## 🗺️ 오늘의 위치 — 8차시 (마지막)

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1~3 ✅ | 척도 · 역채점 · 분포 · 상관 · α | pandas · join · 시각화 |
| 4 ✅ | 조작적 정의 · 임상 cut-off | split · 불균형 · **데이터 누출** |
| 5 ✅ | 관계의 방향성 | 로지스틱 · 계수 · 부트스트랩 |
| 6 ✅ | 선형인가 | Tree · Forest · 과적합 |
| 7 ✅ | 위험요인 · 인과 vs 예측 | Permutation · 오류 분석 |
| **8 (오늘)** | **결론 · 한계 · 윤리 서술** | **재현성 · test 최종 1회 · 최종 리포트** |

**오늘의 재료** — 지난 7주가 만든 전부다.

- `configs/variables.yaml` (2·3차시가 검증·교정) · `modeling_frame.parquet`
- `model_metrics_cv.csv`(6차시) · `feature_importance.csv`(7차시)
- 그리고 **한 번도 열지 않은 test 265명**

> 완료 기준: **"남이 이 repo 를 받아 같은 결과를 재현할 수 있다."**"""),

md("""## Step 0 — 봉인 확인: 정말 안 봤는가"""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(r'''# test 를 열기 전에, 우리가 정말 안 봤는지 스스로 감사(audit)한다.
# 4~7차시 노트북에서 test 인덱스로 성능을 잰 흔적이 있는지 검색한다.
import json, glob, re

suspicious = []
for path in sorted(glob.glob("session[4-7]/session?.ipynb")):
    src = "\n".join("".join(c["source"]) for c in json.load(open(path, encoding="utf-8"))["cells"])
    for line in src.split("\n"):
        if "idx_te" in line and re.search(r"(score|predict|fit|auc)", line, re.I):
            suspicious.append((path, line.strip()))

print("4~7차시에서 test 로 성능을 잰 흔적:", len(suspicious), "건")
for p, l in suspicious:
    print("  ", p, "|", l)
print("\n0 건이면 — 우리는 규칙을 지켰다. 이제 열 자격이 있다.")
print("※ 이 감사는 형식적인 절차가 아니다. '안 봤다고 믿는 것'과 '안 봤음을 확인하는 것'은 다르다.")'''),

md("""## Step 1 — 봉투를 연다 ⚠️ (첫 봉우리)

이제 **딱 한 번** 연다. 규칙을 다시 확인하자:

- 모델도, 변수도, 하이퍼파라미터도 **이미 다 정해져 있다** (6차시 CV 로 확정)
- cutoff 도 **train 에서 계산한 1.500** 을 그대로 쓴다
- **결과를 보고 아무것도 바꾸지 않는다**

> 만약 결과가 마음에 안 들어서 무언가를 바꾸고 다시 잰다면,
> 그 순간 test 는 **더 이상 test 가 아니다.** 두 번째 측정은 train 과 같아진다."""),

code(r'''import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from maps_risk.config import load_configs
from maps_risk.dataset import make_high_stress_label, split_features
from maps_risk.models import build_models
from maps_risk import evaluation

_, cfg = load_configs("configs")
SEED = cfg["random_seed"]
frame = pd.read_parquet("data/processed/modeling_frame.parquet")
scores = frame["acculturative_stress_w6"]

idx_tr, idx_te = train_test_split(frame.index, test_size=cfg["test_size"], random_state=SEED,
                                  stratify=(scores >= scores.median()).astype(int))
frame["high_stress"], cutoff = make_high_stress_label(
    scores.loc[idx_tr], scores, cfg["target"]["high_stress_quantile"])
ytr, yte = frame.loc[idx_tr, "high_stress"], frame.loc[idx_te, "high_stress"]
cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=SEED)

print(f"train {len(idx_tr)} · test {len(idx_te)}  ← 오늘 처음 쓴다")
print(f"cutoff = {cutoff:.3f} (train 에서 계산한 값 그대로)")
print(f"test 양성 {yte.mean():.1%} ({int(yte.sum())}명) · train 양성 {ytr.mean():.1%}")'''),

code(r'''# TODO: train 으로 학습하고 test 로 '한 번만' 평가하라
rows, probs = [], {}
for mset in ("A", "B"):
    cols = split_features(frame, mset)
    Xtr, Xte = frame.loc[idx_tr, cols], frame.loc[idx_te, cols]
    for name, (est, grid) in build_models(cfg).items():
        # 하이퍼파라미터는 train 안 CV 로만 고른다 (test 는 절대 안 본다)
        model = (GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
                 .fit(Xtr, ytr).best_estimator_) if grid else est.fit(Xtr, ytr)

        prob = model.predict_proba(_____)[:, 1]          # ← 어느 X 로 예측하나
        m = evaluation.score_all(yte, model.predict(Xte), prob)
        probs[(mset, name)] = prob
        rows.append({"model_set": mset, "model": name, **m})

test_metrics = pd.DataFrame(rows)
print(test_metrics.to_string(index=False))'''),
code(r'''# CHECK Step1
try:
    A = test_metrics[test_metrics.model_set == "A"].set_index("model")["roc_auc"]
    assert abs(A["Dummy"] - 0.5) < 1e-9, "Dummy 는 test 에서도 0.5 여야 한다"
    assert 0.60 < A["LogisticRegression"] < 0.75, f"로지스틱 test AUC 가 예상 범위 밖 ({A['LogisticRegression']})"
    print("✅ PASS — test 결과가 나왔다. 되돌릴 수 없다.")
    print(f"   Model A: Dummy {A['Dummy']:.4f} · 로지스틱 {A['LogisticRegression']:.4f} · "
          f"트리 {A['DecisionTree']:.4f} · 포레스트 {A['RandomForest']:.4f}")
    print("   → 6차시 CV 순위와 비교해 보라. 무언가 이상하지 않은가?")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: test 로 예측해야 한다 → model.predict_proba(Xte)")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
prob = model.predict_proba(Xte)[:, 1]
```
</details>"""),

md("""## Step 2 — 첫 번째 반전: 순위가 뒤집혔다

6차시에 우리는 이렇게 정리했다:

> Model A CV AUC — 포레스트 **.6651** > 로지스틱 **.6535** (+0.012)
> "포레스트가 5/5 폴드 전부에서 이겼다. **이 차이는 폴드 운이 아니다.**"

그런데 test 에서는 —"""),

code(r'''cv_auc = {"A": {"LogisticRegression": 0.6535, "DecisionTree": 0.6355, "RandomForest": 0.6651},
          "B": {"LogisticRegression": 0.6825, "DecisionTree": 0.6833, "RandomForest": 0.6987}}

cmp_tbl = []
for mset in ("A", "B"):
    for name, c in cv_auc[mset].items():
        t = test_metrics[(test_metrics.model_set == mset) & (test_metrics.model == name)]["roc_auc"].iloc[0]
        cmp_tbl.append({"model_set": mset, "model": name, "CV(6차시)": c, "test(오늘)": t, "차이": round(t - c, 4)})
cmp_tbl = pd.DataFrame(cmp_tbl)
print(cmp_tbl.to_string(index=False))

for mset in ("A", "B"):
    sub = cmp_tbl[cmp_tbl.model_set == mset]
    print(f"\nModel {mset}  CV 1위: {sub.loc[sub['CV(6차시)'].idxmax(), 'model']}"
          f"  →  test 1위: {sub.loc[sub['test(오늘)'].idxmax(), 'model']}")'''),

md("""### 6차시에 우리가 적어 둔 문장

Model A 에서 **순위가 뒤집혔다.** CV 1위는 포레스트였는데 test 1위는 로지스틱이다.

당황스러운가? 그런데 6차시 마지막에 우리는 이미 이렇게 적어 뒀다:

> "`GridSearchCV.best_score_` 는 후보 중 최댓값이라 **후보가 많은 모델(4개)이
> 로지스틱(3개)보다 약간 유리하게 채점**된다. nested CV 를 쓰지 않았으므로,
> **+0.012 라는 작은 격차는 이 편향으로 뒤집힐 수 있다.**"

**그 예측이 맞았다.**

> 🔴 여기서 얻을 교훈은 "우리가 옳았다"가 **아니다.**
> **"불확실성을 미리 기록해 뒀기 때문에 이 결과에 놀라지 않는다"** 는 것이다.
> 만약 6차시에 "포레스트가 더 좋은 모델이다"라고 단정했다면, 오늘 말을 바꿔야 했을 것이다."""),

md("""## Step 3 — 두 번째 반전: 그 뒤집힘조차 믿을 수 없다

"로지스틱이 이겼다"고 결론 내리고 싶어진다. 잠깐 멈추자.

test 는 **265명**뿐이다. 5차시에 배운 도구를 여기에도 쓴다 — **부트스트랩**.
test 265명 중에서 265명을 중복 허용으로 다시 뽑기를 2,000번 반복해,
AUC 가 얼마나 흔들리는지 본다."""),

code(r'''def boot_auc_ci(y_true, y_prob, B=2000, seed=0):
    """test AUC 의 부트스트랩 95% 구간. test 표본이 작으면 구간이 넓게 나온다."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    out = []
    for _ in range(B):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) < 2:      # 한 클래스만 뽑히면 AUC 가 정의되지 않는다
            continue
        out.append(roc_auc_score(y_true[i], y_prob[i]))
    return np.percentile(out, [2.5, 97.5])

print("test AUC 와 부트스트랩 95% 구간")
for (mset, name), p in probs.items():
    if name == "Dummy":
        continue
    lo, hi = boot_auc_ci(yte, p)
    print(f"  {mset} {name:20s} {roc_auc_score(yte, p):.4f}  95% [{lo:.4f}, {hi:.4f}]  폭 {hi-lo:.3f}")'''),

code(r'''# TODO: 두 모델의 '차이'에 대한 신뢰구간을 구하라 (같은 부트스트랩 표본에서 둘 다 계산한다)
rng = np.random.default_rng(1)
yv = np.asarray(yte)
pl, pf = probs[("A", "LogisticRegression")], probs[("A", "RandomForest")]
diffs = []
for _ in range(2000):
    i = rng.integers(0, len(yv), len(yv))
    if len(np.unique(yv[i])) < 2:
        continue
    diffs.append(roc_auc_score(yv[i], pl[i]) - roc_auc_score(yv[i], _____[i]))   # ← 포레스트 확률
diffs = np.array(diffs)

lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"로지스틱 − 포레스트 (Model A, test)")
print(f"  평균 차이 {diffs.mean():+.4f}")
print(f"  95% 구간 [{lo:+.4f}, {hi:+.4f}]")
print(f"  로지스틱이 이긴 비율 {100*(diffs > 0).mean():.1f}%")'''),
code(r'''# CHECK Step3
try:
    assert lo < 0 < hi, "차이의 신뢰구간이 0 을 포함해야 한다"
    print("✅ PASS — 차이의 95% 구간이 **0 을 포함한다.**")
    print("   → 'test 에서 로지스틱이 이겼다'고 단정할 수 없다.")
    print("   CV 에서는 포레스트가, test 에서는 로지스틱이 앞섰다. **두 차이 모두 신뢰할 수 없다.**")
    print("\n   🔴 그래서 결론은 이것이다:")
    print("      '두 모델의 성능 차이는 이 표본 크기로는 판별할 수 없다.'")
    print("      → 성능이 사실상 같다면, **해석 가능한 쪽을 고르는 것이 명백히 옳다.**")
    print("      6차시의 선택(로지스틱을 주 모델로)이 사후적으로도 정당화된다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 같은 부트스트랩 표본 i 로 두 모델을 모두 계산한다 → pf[i]")'''),

md("""## Step 4 — 민감도 분석: 정의를 바꾸면 결론이 흔들리는가

4차시에 우리는 "상위 25%" 라는 선을 그었고, 그 선택에 두 가지 자의성이 있었다:

- **분위수**: 0.75 를 썼지만 0.70 이나 0.80 일 수도 있었다
- **부등호**: `>=` 를 썼지만 `>` 일 수도 있었다 (동점자 142명이 걸린 문제)

**정의를 바꾸면 결론이 바뀌는가?** 이것을 확인하는 것이 **민감도 분석(sensitivity analysis)** 이다.
결론이 정의에 크게 좌우된다면, 그 결론은 데이터가 아니라 **우리 선택**이 만든 것이다."""),

code(r'''from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from maps_risk.preprocessing import make_preprocessor

def logit():
    return Pipeline([("prep", make_preprocessor(scale=True)),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=SEED, C=0.1))])

colsA = split_features(frame, "A")
rows = []
for q in (0.70, 0.75, 0.80):
    for op in (">=", ">"):
        c = scores.loc[idx_tr].quantile(q)
        lab = (scores >= c).astype(int) if op == ">=" else (scores > c).astype(int)
        a, b = lab.loc[idx_tr], lab.loc[idx_te]
        if a.nunique() < 2 or b.nunique() < 2:
            continue
        m = logit().fit(frame.loc[idx_tr, colsA], a)
        rows.append({"분위수": q, "부등호": op, "cutoff": round(c, 3),
                     "train 양성률": round(a.mean(), 3), "test 양성률": round(b.mean(), 3),
                     "test AUC": round(roc_auc_score(b, m.predict_proba(frame.loc[idx_te, colsA])[:, 1]), 4)})
sens = pd.DataFrame(rows)
print(sens.to_string(index=False))
print(f"\ntest AUC 범위: {sens['test AUC'].min():.4f} ~ {sens['test AUC'].max():.4f} "
      f"(폭 {sens['test AUC'].max()-sens['test AUC'].min():.4f})")
print(f"양성률 범위: {sens['test 양성률'].min():.1%} ~ {sens['test 양성률'].max():.1%}  ← 2배 차이")'''),

md("""### Step 4 해석 — 두 가지를 동시에 말해야 한다

**① 성능 결론은 견고하다.** 여섯 가지 정의에서 test AUC 가 **.6449 ~ .6718** —
폭이 0.027 로 좁다. 어떤 정의를 골랐어도 "중간 정도로 구분된다"는 결론은 같다.

**② 그런데 '누가 고스트레스인가'는 크게 달라진다.** test 양성률이 **18.9% ~ 34.3%** 로 **1.8배** 차이다
(train 기준으로는 17.3% ~ 33.7%). 같은 데이터에서 고스트레스 집단의 크기가 정의에 따라 두 배 가까이 변한다.

그리고 3·4차시의 **이산성 문제**가 여기서 다시 보인다:

- `q=0.70` 과 `q=0.75` 는 **완전히 같은 결과**다 (둘 다 cutoff 1.5)
- `>1.5` 와 `>=1.6` 도 **완전히 같은 결과**다 (점수가 0.1 단위라 두 조건이 같은 집합)

> 🔴 즉 **부등호 하나를 바꾸는 것과 분위수를 한 단계 올리는 것이 같은 효과**를 낸다.
> "분위수 0.75" 라는 표기가 실제로 무엇을 뜻하는지는 **데이터를 봐야만** 알 수 있다."""),

md("""## Step 5 — 7차시의 발견은 재현되는가 🔍 (두 번째 봉우리)

7차시에 우리는 train 에서 **가장 중요한 발견**을 했다:

> **놓친 학생(FN)은 TP 가 아니라 TN 을 닮았다.**
> 자아존중감 높고, 친구지지 두텁고, 부모가 챙기고, 우울 낮은 학생들인데 고스트레스가 됐다.

그리고 그것을 **완화하는 설명**도 함께 적었다:

> FN 의 **75.7%** 가 cutoff 바로 위 경계선이다 (TP 는 44.5%) — "간신히 고스트레스"인 학생이 많다.

**두 가지가 test 에서도 재현되는가?** 이것이 오늘 가장 중요한 확인이다."""),

code(r'''best = logit().fit(frame.loc[idx_tr, colsA], ytr)
pred_te = best.predict(frame.loc[idx_te, colsA])
grp = pd.Series(np.select(
    [(yte == 1) & (pred_te == 1), (yte == 1) & (pred_te == 0),
     (yte == 0) & (pred_te == 1), (yte == 0) & (pred_te == 0)],
    ["TP", "FN", "FP", "TN"]), index=yte.index)

look = ["self_esteem", "peer_support", "parenting_monitoring", "depression", "acculturative_stress_w6"]
prof = frame.loc[idx_te].assign(집단=grp.values).groupby("집단")[look].mean()
prof.insert(0, "n", grp.value_counts())
print("test 집단별 프로파일")
print(prof.round(3).to_string())

fn, tp, tn = (frame.loc[idx_te][grp == g] for g in ("FN", "TP", "TN"))
print(f"\n① FN 이 TN 을 닮는가 (자아존중감): FN {fn['self_esteem'].mean():.3f} · "
      f"TN {tn['self_esteem'].mean():.3f} · TP {tp['self_esteem'].mean():.3f}")
print(f"   → FN 이 TN 에 더 가까운가: "
      f"{abs(fn['self_esteem'].mean()-tn['self_esteem'].mean()) < abs(fn['self_esteem'].mean()-tp['self_esteem'].mean())}")
print(f"\n② 경계선 설명은 재현되는가 (스트레스 ≤1.7 비율):")
print(f"   test  — FN {(fn['acculturative_stress_w6']<=1.7).mean():.1%} · TP {(tp['acculturative_stress_w6']<=1.7).mean():.1%}")
print(f"   train — FN 75.7% · TP 44.5%   ← 7차시 결과")'''),
code(r'''# CHECK Step5
try:
    d_tn = abs(fn["self_esteem"].mean() - tn["self_esteem"].mean())
    d_tp = abs(fn["self_esteem"].mean() - tp["self_esteem"].mean())
    assert d_tn < d_tp, "FN 이 TN 에 더 가까워야 한다"
    fn_edge = (fn["acculturative_stress_w6"] <= 1.7).mean()
    tp_edge = (tp["acculturative_stress_w6"] <= 1.7).mean()
    print("✅ PASS — 두 가지 확인이 끝났다.")
    print("   ① **핵심 발견은 재현됐다.** test 에서도 FN 은 TN 을 닮았다 (오히려 더 뚜렷하다).")
    print(f"   ② **완화 설명은 재현되지 않았다.** test 에서 FN 경계선 비율 {fn_edge:.1%} < TP {tp_edge:.1%} —")
    print("      train 에서와 **방향이 반대**다.")
    print("\n   🔴 그래서 결론이 더 무거워진다:")
    print("      '놓친 학생이 단지 경계선이라서 놓친 것'이라는 설명을 test 가 지지하지 않는다.")
    print("      남는 것은 더 심각한 해석 — **겉보기에 멀쩡한 학생을 체계적으로 놓친다.**")
except Exception as e:
    print("❌ FAIL —", e)'''),

md("""## Step 6 — 최종 보고서 쓰기

이제 `final_report.md` 를 만든다. 원칙은 이 프로그램 내내 지켜 온 것과 같다:

> **숫자는 코드가 채우고, 판단은 사람이 쓴다.**

아래 셀은 지금까지의 실측값을 자동으로 채운 **뼈대**를 만든다.
`<!-- TODO(사람) -->` 로 표시된 칸은 **여러분이 직접** 채워야 한다 —
그게 이 과목의 평가 대상이다 (루브릭: 결과 해석 15% · 연구문제 이해 20%)."""),

code(r'''import textwrap
A = test_metrics[test_metrics.model_set == "A"].set_index("model")
B = test_metrics[test_metrics.model_set == "B"].set_index("model")
lo_l, hi_l = boot_auc_ci(yte, probs[("A", "LogisticRegression")])
d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])

report = f"""# 최종 보고서 — MAPS 다문화청소년 문화적응 스트레스 예측

> 자동 생성 뼈대 · 숫자는 코드가 채웠고, **판단과 서술은 사람이 쓴다.**
> `TODO(사람)` 칸을 채우면 완성이다.

## 1. 연구 질문과 설계

- **RQ1** 중2(5차) 심리사회적 특성으로 1년 뒤(6차) 고스트레스 집단을 어느 정도 구분할 수 있는가
- **RQ2** 그 구분에 상대적으로 중요한 변수는 무엇인가
- **RQ3** 이전 시점의 문화적응 스트레스를 추가하면 예측력이 얼마나 개선되는가

표본: MAPS 1기 패널, 5·6차 **모두 참여한 {len(frame)}명** (train {len(idx_tr)} / test {len(idx_te)}).
고스트레스 정의: **train 점수의 75 백분위수 이상**(cutoff = {cutoff:.3f}) — 조작적 정의이며 임상 진단이 아니다.
실제 양성 비율: train {ytr.mean():.1%} · test {yte.mean():.1%} (동점자 때문에 25%가 아니다).

## 2. 결과 — test 최종 평가 (단 1회)

| 모델 세트 | 모델 | ROC-AUC | Average Precision | Recall | Precision | Balanced Acc |
|---|---|---|---|---|---|---|
| A | Dummy | {A.loc['Dummy','roc_auc']:.4f} | {A.loc['Dummy','average_precision']:.4f} | {A.loc['Dummy','recall']:.4f} | {A.loc['Dummy','precision']:.4f} | {A.loc['Dummy','balanced_accuracy']:.4f} |
| A | **로지스틱 회귀** | **{A.loc['LogisticRegression','roc_auc']:.4f}** | {A.loc['LogisticRegression','average_precision']:.4f} | {A.loc['LogisticRegression','recall']:.4f} | {A.loc['LogisticRegression','precision']:.4f} | {A.loc['LogisticRegression','balanced_accuracy']:.4f} |
| A | 결정 트리 | {A.loc['DecisionTree','roc_auc']:.4f} | {A.loc['DecisionTree','average_precision']:.4f} | {A.loc['DecisionTree','recall']:.4f} | {A.loc['DecisionTree','precision']:.4f} | {A.loc['DecisionTree','balanced_accuracy']:.4f} |
| A | 랜덤 포레스트 | {A.loc['RandomForest','roc_auc']:.4f} | {A.loc['RandomForest','average_precision']:.4f} | {A.loc['RandomForest','recall']:.4f} | {A.loc['RandomForest','precision']:.4f} | {A.loc['RandomForest','balanced_accuracy']:.4f} |
| B | **로지스틱 회귀** | **{B.loc['LogisticRegression','roc_auc']:.4f}** | {B.loc['LogisticRegression','average_precision']:.4f} | {B.loc['LogisticRegression','recall']:.4f} | {B.loc['LogisticRegression','precision']:.4f} | {B.loc['LogisticRegression','balanced_accuracy']:.4f} |
| B | 랜덤 포레스트 | {B.loc['RandomForest','roc_auc']:.4f} | {B.loc['RandomForest','average_precision']:.4f} | {B.loc['RandomForest','recall']:.4f} | {B.loc['RandomForest','precision']:.4f} | {B.loc['RandomForest','balanced_accuracy']:.4f} |

주 모델(Model A · 로지스틱)의 test ROC-AUC 95% 부트스트랩 구간: **[{lo_l:.4f}, {hi_l:.4f}]**

**RQ1 의 답**: TODO(사람) — Dummy 대비 얼마나 나은지, 그리고 그 크기를 어떻게 평가하는지 2~3문장.

**RQ3 의 답**: Model A {A.loc['LogisticRegression','roc_auc']:.4f} → Model B {B.loc['LogisticRegression','roc_auc']:.4f}
(**{B.loc['LogisticRegression','roc_auc']-A.loc['LogisticRegression','roc_auc']:+.4f}**). TODO(사람) — 이 개선폭의 의미를 인과 주장 없이 서술.

## 3. RQ2 — 어떤 변수가 기여했는가

세 방법(표준화 계수 · 부트스트랩 · OOF permutation)이 모두 지목한 변수는 **3개**다:
`peer_support`(친구지지) · `self_esteem`(자아존중감) · `parenting_monitoring`(부모 감독).
세 변수 모두 **음(−) 방향** — 높을수록 이후 고스트레스 집단에 속할 확률이 낮았다.

TODO(사람) — 위 세 변수를 심리학적으로 해석하되, **7차시 서술 규칙(❌/✅ 표)** 을 지킬 것.

## 4. 모델 선택에 대한 정직한 기록

- 6차시 CV: 포레스트 .6651 > 로지스틱 .6535 (**포레스트 우세**, 5/5 폴드)
- 8차시 test: 로지스틱 {A.loc['LogisticRegression','roc_auc']:.4f} > 포레스트 {A.loc['RandomForest','roc_auc']:.4f} (**뒤집힘**)
- 두 모델 차이의 test 95% 구간: **[{d_lo:+.4f}, {d_hi:+.4f}]** — **0을 포함**

→ 결론: **두 모델의 성능 차이는 이 표본 크기로 판별할 수 없다.**
성능이 사실상 같으므로, **해석 가능한 로지스틱을 주 모델로 삼은 6차시의 선택은 유지된다.**

## 5. 민감도 분석

조작적 정의(분위수 × 부등호)를 바꿔 가며 재계산한 결과:
test AUC **{sens['test AUC'].min():.4f} ~ {sens['test AUC'].max():.4f}** (폭 {sens['test AUC'].max()-sens['test AUC'].min():.4f}),
양성률 **{sens['test 양성률'].min():.1%} ~ {sens['test 양성률'].max():.1%}**.

→ 성능 결론은 정의에 **견고**하나, **누가 고스트레스로 분류되는지는 2배 차이**가 난다.

## 6. 한계

TODO(사람) — 아래 항목을 각각 2~3문장으로. (README §9 와 각 차시 노트를 참고)

1. 패널 마모와 표본 대표성
2. 조작적 정의의 자의성 (동점자 142명 · 부등호)
3. `s_accul_str_10` 의 이질성 (r_it .04 · 10문항 유지 결정)
4. 낮은 신뢰도 척도 (`peer_relationship` α .626)
5. 다중공선성과 계수 해석의 제약 (부호 뒤집힘 7개)
6. 하이퍼파라미터 선택 편향 (nested CV 미실시)
7. **예측 ≠ 인과** (시간 순서는 확보, 교란변수는 미통제)

## 7. 윤리 — 이 모델을 현장에서 쓸 수 있는가

**오류의 편향 (7·8차시 실측)**: 놓친 학생(FN)은 자아존중감·친구지지·부모감독이 높고
우울이 낮아 **정상 판정군(TN)과 흡사한 프로파일**이다. 이 현상은 **test 에서도 재현**됐다.
train 에서 이를 완화하던 "경계선 효과" 설명은 **test 에서 재현되지 않았다**.

> **이 모델의 오류는 무작위가 아니다. 특정한 종류의 학생에게 체계적으로 쏠려 있다.**

TODO(사람) — 위 사실을 근거로, 이 모델의 현장 사용 가능성에 대한 입장을 3~5문장으로.
(찬성/반대 모두 가능하다. **근거를 대는지**만 평가한다.)

## 8. 재현 방법

```bash
pip install -e .
python scripts/codebook_candidates.py --propose      # 변수 후보 생성
# configs/variables.yaml 을 사람이 검증 (2·3차시)
python scripts/build_dataset.py --wave5 <5차 CSV> --wave6 <6차 CSV>
python scripts/run_models.py
pytest -q
```

난수 seed {SEED} · test_size {cfg['test_size']} 고정. 위 순서대로 실행하면 본 보고서의 모든 수치가 재현된다.
"""

os.makedirs("reports", exist_ok=True)
open("reports/final_report.md", "w", encoding="utf-8").write(report)
print("✅ reports/final_report.md 생성")
print(f"   TODO(사람) 칸 {report.count('TODO(사람)')}개 — 이걸 채우는 것이 여러분의 과제다.")'''),

md("""## Step 7 — 재현성: 남이 받아서 돌릴 수 있는가

이 프로그램의 완료 기준은 성능이 아니라 이것이었다:

> **"남이 이 repo 를 받아 같은 결과를 재현할 수 있다."**

마지막으로 **공식 파이프라인을 처음부터 다시 돌려** 확인한다.
노트북에서 손으로 계산한 값과 파이프라인 출력이 **일치해야** 한다."""),

code(r'''!python scripts/run_models.py'''),

code(r'''# 파이프라인 출력과 노트북 계산이 일치하는가
official = pd.read_csv("reports/model_metrics.csv")
mine = test_metrics.copy()
merged = official.merge(mine, on=["model_set", "model"], suffixes=("_파이프라인", "_노트북"))
merged["일치"] = (merged["roc_auc_파이프라인"] - merged["roc_auc_노트북"]).abs() < 1e-9
print(merged[["model_set", "model", "roc_auc_파이프라인", "roc_auc_노트북", "일치"]].to_string(index=False))
print(f"\n전부 일치: {merged['일치'].all()}")
print("→ 노트북에서 손으로 짠 코드와 파이프라인이 같은 답을 낸다. 이것이 재현성이다.")'''),

code(r'''# 최종 산출물 점검
import os
outputs = {
    "configs/variables.yaml":                  "2·3차시 — 사람이 검증·교정한 변수 매핑",
    "data/processed/modeling_frame.parquet":   "2·3차시 — 모델링 데이터셋",
    "reports/data_quality.md":                 "2차시 — 데이터 품질 보고서",
    "reports/model_metrics_cv.csv":            "6차시 — CV 성능표 (모델 선택 근거)",
    "reports/feature_importance.csv":          "7차시 — 변수 중요도",
    "reports/model_metrics.csv":               "8차시 — **test 최종 성능**",
    "reports/final_report.md":                 "8차시 — 최종 보고서 (TODO 를 채워 완성)",
}
for f, why in outputs.items():
    print(f"  {'✅' if os.path.exists(f) else '⬜'} {f:42s} {why}")

print("\n" + "="*70)
print("  8주가 끝났다.")
print("  우리가 만든 것은 모델이 아니라, **믿을 수 있는 결론과 그 한계의 목록**이다.")
print("="*70)'''),

md("""## 🎯 회고 — 8주 전체 (10분)

1. test 를 열기 전에 **감사(audit)** 부터 한 이유는? "안 봤다고 믿는 것"과
   "안 봤음을 확인하는 것"은 어떻게 다른가?
2. CV 에서는 포레스트가, test 에서는 로지스틱이 이겼다. 그리고 **둘 다 신뢰구간이 0을 포함**한다.
   이 상황에서 **"어떤 모델이 더 좋다"고 말할 수 있는가?**
3. 7차시의 FN 발견은 재현됐지만 그것을 완화하던 설명은 재현되지 않았다.
   **재현되지 않은 것을 보고서에 쓰는 이유**는 무엇인가?

## 📝 최종 과제
- `reports/final_report.md` 의 **TODO(사람) 칸을 전부 채운다** (§2 RQ1·RQ3, §3 해석, §6 한계, §7 윤리)
- **5~10분 발표자료**: 연구질문 → 데이터 → 방법 → 결과 → **한계와 윤리**
  (성능 숫자보다 **한계 절에 시간을 더 쓴다**)
- 동료 한 명의 repo 를 받아 **직접 재현**해 보고, 막힌 지점을 리포트

## 🎓 이 8주에 배운 것

| 차시 | 한 문장 |
|---|---|
| 1 | 무엇을 예측할지 정하는 것이 절반이다 |
| 2 | 컬럼명을 절대 추측하지 않는다 — 검증은 사람이 연다 |
| 3 | 검증은 한 번의 행사가 아니라 계속되는 상태다 |
| 4 | AUC 1.0 은 축하가 아니라 경보다 · 규칙은 결과가 아니라 절차로 정당화된다 |
| 5 | 모른다고 말하는 것도 결과다 |
| 6 | 성능 표의 1등이 곧 답이 아니다 — 목적이 답을 정한다 |
| 7 | "모델이 무엇을 썼는가"와 "무엇이 원인인가"는 다른 질문이다 |
| **8** | **불확실성을 미리 기록해 두면, 결과가 뒤집혀도 말을 바꾸지 않아도 된다** |

> 이 프로그램에서 **모델 성능에는 점수를 주지 않았다.**
> 성능이 낮아도 "현재 변수만으로는 충분히 구분하기 어려웠다"는 결론을
> **정확히** 도출했다면 그것이 성공이다. 여러분은 그것을 해냈다."""),
]

os.makedirs("session8", exist_ok=True)
save(cells, "session8/session8.ipynb")

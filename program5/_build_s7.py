# -*- coding: utf-8 -*-
"""session7.ipynb 빌더 — 변수 중요도 · 오류 분석 · 인과 vs 예측.

7차시는 이 프로그램의 **해석 클라이맥스**다. 세 방법(계수·부트스트랩·permutation)이
같은 답을 주는지 확인하고, 모델이 **누구를 놓쳤는지** 들여다본 뒤,
"이 중요도를 위험요인이라 불러도 되는가"라는 윤리 질문으로 착지한다.

★ 오늘도 test 는 열지 않는다. 중요도는 **폴드별 validation** 에서 잰다.

실측 근거 (train 1,056 · Model A 18변수 · 5-fold CV seed 42):
  permutation 은 어디서 재느냐가 중요 — train 에서 재면 합계가 1.8배 부풀려진다
  로지스틱 out-of-fold 상위: peer_support .0291 · self_esteem .0290 ·
                            parenting_monitoring .0196 · depression .0151 · peer_relationship .0122
  계수순위 vs permutation 순위 스피어만 .866 (상위 5개는 순위가 완전히 같다)
  계수순위 vs 포레스트 permutation .356 · 로지스틱 perm vs 포레스트 perm .467
  상관 함정: peer_support ↔ peer_relationship r .615
            둘 다 .0291/.0122 → 하나씩 빼면 .0382 / .0268 (서로 가린다)
  오류분석(out-of-fold): TN 425 · FP 275 · TP 220 · FN 136 (recall .618)
    FN 프로파일이 TN 과 흡사 — self_esteem 3.42(TP 2.76) · peer_support 4.45(TP 3.58)
    FN 의 75.7% 가 cutoff 바로 위(1.5~1.7) 경계선
"""
import os

from nb import md, code, save, SETUP, handoff_in, handoff_out

cells = [
md("""# 7차시 — 무엇을 보고 판단했니, 그리고 그걸 뭐라고 불러야 하나

### Permutation Importance · 표준화 계수 · 오류 분석 · 인과 vs 예측

> **오늘 한 문장:** "6차시에 모델을 골랐다. 오늘은 그 모델에게 **'무엇을 보고 판단했니?'**
> 라고 묻고 — 마지막에 **'그걸 위험요인이라고 불러도 되니?'** 라고 우리 자신에게 묻는다."

오늘은 이 프로그램의 **해석 클라이맥스**다. 세 가지를 한다:

1. **세 방법이 같은 답을 주는지** 확인한다 (계수 · 부트스트랩 · permutation). ← 고비 1
2. 모델이 **누구를 놓쳤는지** 들여다본다 — 오늘 가장 심리학적인 장면. ← 고비 2
3. 그 결과를 **어디까지 말해도 되는지** 정한다.

오늘의 목표 4가지:

1. **Permutation Importance** 가 무엇을 재는지, 왜 **validation 에서** 재야 하는지 설명한다.
2. 계수 순위와 permutation 순위를 비교하고, **다르면 그 이유를 설명**한다.
3. **오류 분석**으로 FN(놓친 학생)의 프로파일을 확인하고 그 의미를 해석한다.
4. **예측 기여 ≠ 위험요인 ≠ 원인** 을 구분해 서술한다.

> 🔒 **오늘도 test 는 열지 않는다.** 중요도는 폴드별 **validation** 에서 잰다.
> 8차시에 딱 한 번 연다 — 6차시에 약속한 그대로다."""),

md("""## 🗺️ 오늘의 위치 — 7차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1~3 ✅ | 척도 · 역채점 · 분포 · 상관 · α | pandas · join · 시각화 |
| 4 ✅ | 조작적 정의 · 임상 cut-off | split · 불균형 · 데이터 누출 |
| 5 ✅ | 예측변수와 결과의 관계·방향성 | 로지스틱 · 계수 · 표준화 · 부트스트랩 |
| 6 ✅ | 심리 특성은 선형적으로 작동하는가 | Tree · Forest · 과적합 · CV |
| **7 (오늘)** | **위험요인·보호요인 · 인과 vs 예측** | **Permutation Importance · 오류 분석** |
| 8 | 결론 · 한계 · 윤리 서술 | 재현성 · **test 최종 1회** · final_report |

**오늘의 재료**

- 6차시가 고른 **주 모델**: 로지스틱 회귀 (C=0.1) — 해석이 목적이므로
- 6차시의 **대조 모델**: 랜덤 포레스트 (depth=3) — 다른 답을 주는지 보려고
- 5차시의 **계수와 신뢰구간** — 오늘 비교 대상

> 🔴 오늘의 규칙: **"모델이 무엇을 썼는가"와 "무엇이 원인인가"는 다른 질문이다.**
> 오늘 우리는 앞의 질문에만 답할 수 있다."""),

md("""## Step 0 — 재료 확인"""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(handoff_in(pull=['configs/variables.yaml', 'data/processed/modeling_frame.parquet', 'reports/model_metrics_cv.csv'], require=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], hint="지난 차시 노트북 맨 끝의 '드라이브에 저장' 셀을 실행하면 여기서 자동으로 복원된다")),
code(r'''# 4~6차시와 똑같은 상태를 재현하고, 6차시가 고른 두 모델을 세운다
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from maps_risk.config import load_configs
from maps_risk.dataset import make_high_stress_label, split_features
from maps_risk.preprocessing import make_preprocessor

_, cfg = load_configs("configs")
SEED = cfg["random_seed"]
frame = pd.read_parquet("data/processed/modeling_frame.parquet")
scores = frame["acculturative_stress_w6"]

idx_tr, idx_te = train_test_split(frame.index, test_size=cfg["test_size"], random_state=SEED,
                                  stratify=(scores >= scores.median()).astype(int))
frame["high_stress"], cutoff = make_high_stress_label(
    scores.loc[idx_tr], scores, cfg["target"]["high_stress_quantile"])

featsA = split_features(frame, "A")
Xtr, ytr = frame.loc[idx_tr, featsA], frame.loc[idx_tr, "high_stress"]
cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=SEED)

def build(clf, scale=False):
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("clf", clf)])

LOGIT  = build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED, C=0.1), scale=True)   # 6차시가 고른 주 모델
FOREST = build(RandomForestClassifier(n_estimators=300, max_depth=3, class_weight="balanced",
                                      random_state=SEED, n_jobs=-1))       # 대조 모델

print(f"train {len(idx_tr)} · test {len(idx_te)}(봉인) · cutoff {cutoff:.3f} · 양성 {ytr.mean():.1%}")
print(f"Model A {len(featsA)}변수 · 주 모델 = 로지스틱(C=0.1) · 대조 = 포레스트(depth=3)")'''),

md("""## Step 1 — Permutation Importance: 변수를 망가뜨려 본다

5차시의 **계수**는 로지스틱 회귀에만 있다. 포레스트에는 계수가 없다.
그러면 나무 300그루가 무엇을 보고 판단했는지 어떻게 알까?

**Permutation Importance(순열 중요도)** 의 아이디어는 무식할 만큼 단순하다:

```
① 모델의 성능(AUC)을 잰다
② 변수 하나만 골라 그 열의 값을 무작위로 섞는다  ← 그 변수를 '망가뜨린다'
③ 다시 성능을 잰다
④ 떨어진 만큼이 그 변수의 중요도다
```

**성능이 많이 떨어졌다 = 모델이 그 변수에 많이 기대고 있었다.**
모델 종류와 무관하게 쓸 수 있어서, 로지스틱과 포레스트를 **같은 자로** 비교할 수 있다.

### ⚠️ 그런데 — 어디서 섞느냐가 결정적이다

학습에 쓴 데이터(train)에서 섞으면, **과적합된 모델이 외운 것**까지 중요도로 잡힌다.
6차시에서 봤듯 트리·포레스트는 train 을 잘 외운다.

그래서 우리는 **폴드마다 train 으로 학습하고, 본 적 없는 validation 에서 섞는다.**"""),

code(r'''# train 에서 재는 것과 validation 에서 재는 것이 실제로 얼마나 다른가
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from maps_risk import evaluation

# ① train 에서 (하면 안 되는 방식)
fitted = clone(FOREST).fit(Xtr, ytr)
in_sample = permutation_importance(fitted, Xtr, ytr, scoring="roc_auc",
                                   n_repeats=10, random_state=0, n_jobs=-1).importances_mean

# ② 폴드별 validation 에서 (우리 방식)
oof = evaluation.permutation_scores_cv(FOREST, Xtr, ytr, cv, n_repeats=10, seed=0)

cmp0 = pd.DataFrame({"train에서 잰 값": in_sample}, index=featsA).join(
    oof.set_index("feature")["imp_mean"].rename("validation에서 잰 값"))
print(cmp0.sort_values("train에서 잰 값", ascending=False).head(6).round(4).to_string())
print(f"\n합계: train {in_sample.sum():.4f}  vs  validation {oof['imp_mean'].sum():.4f}"
      f"   → train 쪽이 약 {in_sample.sum()/oof['imp_mean'].sum():.1f}배 부풀려진다")
print("→ 같은 모델, 같은 방법인데 **어디서 재느냐**만으로 숫자가 이렇게 달라진다.")'''),

md("""## Step 2 — 세 방법이 같은 답을 주는가 ⚠️ (첫 봉우리)

우리는 이제 같은 질문에 답하는 **세 가지 도구**를 갖고 있다:

| 방법 | 차시 | 무엇을 재나 |
|---|---|---|
| **표준화 계수** | 5차시 | 다른 변수를 통제했을 때의 관계 (방향 + 크기) |
| **부트스트랩 신뢰구간** | 5차시 | 그 계수가 표본이 바뀌어도 버티는가 |
| **Permutation Importance** | 오늘 | 그 변수를 망가뜨리면 성능이 얼마나 떨어지나 |

셋이 **같은 변수를 가리키면** 결론이 단단해진다. 이것을 **삼각검증(triangulation)** 이라 한다.
다르면? **다른 이유를 설명해야 한다.**"""),

code(r'''# ▶ 로지스틱의 permutation importance 를 out-of-fold 로 재라
perm_logit = evaluation.permutation_scores_cv(
    LOGIT,                         # 오늘의 주 모델 = 로지스틱
    Xtr, ytr, cv, n_repeats=10, seed=0)

print(perm_logit.round(4).to_string(index=False))
print("\n※ imp_mean 이 음수 = 섞었더니 오히려 성능이 좋아졌다 = 그 변수는 도움이 안 됐다는 뜻")
print("※ n_folds_positive = 5개 폴드 중 몇 개에서 양수였나 (안정성 지표)")'''),
code(r'''# CHECK Step2 — 5차시 결과와 대조한다
try:
    top3_perm = perm_logit.head(3)["feature"].tolist()
    STABLE_5차시 = ["peer_support", "self_esteem", "parenting_monitoring"]   # 신뢰구간이 0 을 제외했던 3개
    assert set(top3_perm) == set(STABLE_5차시), f"상위 3개가 5차시의 그 3개여야 한다 (지금 {top3_perm})"
    assert (perm_logit.head(3)["n_folds_positive"] >= 4).all(), "상위 3개는 대부분 폴드에서 양수여야 한다"
    print("✅ PASS — permutation 상위 3개 =", ", ".join(top3_perm))
    print("   5차시에 **부트스트랩 신뢰구간이 0 을 제외했던 바로 그 3개**다.")
    print("   방법이 완전히 다른데(계수 vs 성능 하락) 같은 답이 나왔다 — **삼각검증 성공**.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 6차시가 고른 주 모델은 로지스틱이다 → LOGIT")'''),

code(r'''# 계수 순위와 permutation 순위를 나란히 놓는다
coef = pd.Series(clone(LOGIT).fit(Xtr, ytr).named_steps["clf"].coef_[0], index=featsA)
tbl = pd.DataFrame({"계수": coef, "계수_절댓값": coef.abs()}).join(
    perm_logit.set_index("feature")[["imp_mean"]].rename(columns={"imp_mean": "perm_로지스틱"}))
tbl["순위_계수"] = tbl["계수_절댓값"].rank(ascending=False).astype(int)
tbl["순위_perm"] = tbl["perm_로지스틱"].rank(ascending=False).astype(int)
print(tbl.sort_values("순위_계수").round(4).to_string())

rho = tbl["계수_절댓값"].corr(tbl["perm_로지스틱"], method="spearman")
print(f"\n순위 상관(스피어만) = {rho:.3f}")
print("상위 5개는 순위가 **완전히 일치**한다 (1,2,3,4,5).")
print("→ 같은 모델을 두 방식으로 물어봤으니 비슷한 게 자연스럽다. 그래도 확인은 해야 한다.")'''),

md("""## Step 3 — 포레스트는 다른 답을 준다

같은 자(permutation)로 **포레스트**에게도 물어보자. 6차시에서 포레스트는 로지스틱보다
AUC 가 0.012 높았다 — 비선형과 상호작용을 쓸 수 있으니 **다른 변수를 볼 수도** 있다."""),

code(r'''perm_forest = evaluation.permutation_scores_cv(FOREST, Xtr, ytr, cv, n_repeats=10, seed=0)

both = (perm_logit.set_index("feature")[["imp_mean"]].rename(columns={"imp_mean": "로지스틱"})
        .join(perm_forest.set_index("feature")[["imp_mean"]].rename(columns={"imp_mean": "포레스트"})))
both["순위_로지스틱"] = both["로지스틱"].rank(ascending=False).astype(int)
both["순위_포레스트"] = both["포레스트"].rank(ascending=False).astype(int)
both["순위차"] = (both["순위_로지스틱"] - both["순위_포레스트"]).abs()
print(both.sort_values("순위_로지스틱").round(4).to_string())
print(f"\n두 모델의 순위 상관(스피어만) = {both['로지스틱'].corr(both['포레스트'], method='spearman'):.3f}")

# 순위차는 '어느 한쪽에서라도 상위권인' 변수만 볼 의미가 있다.
# 중요도가 둘 다 0 근처인 변수들의 순위차는 사실상 잡음이다.
meaningful = both[((both["순위_로지스틱"] <= 8) | (both["순위_포레스트"] <= 8)) & (both["순위차"] >= 5)]
print("상위권인데 두 모델의 순위가 5계단 이상 벌어진 변수:")
print(meaningful[["로지스틱", "포레스트", "순위_로지스틱", "순위_포레스트"]].round(4).to_string())
print("\n(중요도가 둘 다 0 근처인 변수들의 순위차는 잡음이므로 제외했다.)")'''),

md("""### Step 3 해석 — 왜 다른가

두 모델의 순위 상관은 **약 .47** 로, 계수↔permutation 의 .87 보다 훨씬 낮다.
**상위 2개(친구지지·자아존중감)는 두 모델이 똑같이 1·2위로 꼽지만**, 그 아래는 흔들린다.

특히 눈에 띄는 것 두 개:

- `peer_relationship`(교우관계): 로지스틱 **5위** → 포레스트 **13위** (크게 떨어짐)
- `school_adjustment`(학교적응): 로지스틱 7위 → 포레스트 **4위** (올라옴 — 위 필터에는 안 걸리지만 표 전체에서 보인다)
- `bicultural_attitude`(이중문화수용태도): 로지스틱 13위 → 포레스트 **6위**

**두 가지 이유가 섞여 있다.**

1. **모델이 다른 것을 본다.** 포레스트는 역치와 상호작용을 쓴다. 6차시에서 봤듯
   `school_adjustment` 는 비선형 관계일 수 있고, 트리는 그걸 쓸 수 있다.
2. **상관된 변수끼리 서로를 가린다.** ← 이게 더 중요하다. 다음 Step 에서 직접 확인한다.

> 🔴 **결론이 흔들리는 것이 아니다. "상위 2~3개만 두 모델이 합의한다"는 것이 결론이다.**
> 5차시 부트스트랩도 3개만 살아남았다. **세 번째로 같은 답이 나온 셈이다.**"""),

md("""## Step 4 — 상관된 변수의 함정 🔍

Permutation Importance 에는 **잘 알려진 함정**이 하나 있다.

> 서로 강하게 상관된 변수가 둘 있으면, 하나를 망가뜨려도 **다른 하나가 대신 정보를 준다.**
> 그래서 **둘 다 중요도가 낮게** 나온다 — 실제로는 둘 다 중요한데도.

3차시에서 `peer_support`(친구지지)와 `peer_relationship`(교우관계)의 상관이 **.615** 였다.
직접 확인해 보자."""),

code(r'''# ▶ 한 변수를 빼면 다른 변수의 중요도가 어떻게 변하는지 확인하라
pair = ["peer_support", "peer_relationship"]
print(f"두 변수의 상관 r = {Xtr[pair[0]].corr(Xtr[pair[1]]):.3f}\n")

base = perm_logit.set_index("feature")["imp_mean"]
print(f"둘 다 있을 때   : {pair[0]} {base[pair[0]]:.4f} · {pair[1]} {base[pair[1]]:.4f}")

for drop, keep in ((pair[1], pair[0]), (pair[0], pair[1])):
    X2 = Xtr.drop(columns=[drop])           # 짝 중 한 변수를 뺀다
    r2 = evaluation.permutation_scores_cv(LOGIT, X2, ytr, cv, n_repeats=10, seed=0)
    print(f"{drop} 제거 → {keep} {r2.set_index('feature')['imp_mean'][keep]:.4f}")'''),
code(r'''# CHECK Step4
try:
    X2 = Xtr.drop(columns=["peer_relationship"])
    solo = evaluation.permutation_scores_cv(LOGIT, X2, ytr, cv, n_repeats=10, seed=0)
    solo_ps = solo.set_index("feature")["imp_mean"]["peer_support"]
    assert solo_ps > base["peer_support"], "짝을 빼면 남은 변수의 중요도가 커져야 한다"
    print(f"✅ PASS — peer_support: 둘 다 있을 때 {base['peer_support']:.4f} → 짝을 빼면 {solo_ps:.4f}")
    print("   두 변수가 **서로의 중요도를 가리고 있었다.**")
    print("   → 🔴 **'중요도가 낮다'가 '중요하지 않다'를 뜻하지 않는다.**")
    print("      비슷한 것을 재는 변수가 함께 들어 있으면 둘 다 작게 나온다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: drop 변수를 빼야 한다 → Xtr.drop(columns=[drop])")'''),
md("""<details><summary>💡 해설 (펼쳐 보기)</summary>

```python
X2 = Xtr.drop(columns=[drop])
```

실측:

| 상태 | peer_support | peer_relationship |
|---|---|---|
| 둘 다 있을 때 | .0291 | .0122 |
| `peer_relationship` 제거 | **.0382** | — |
| `peer_support` 제거 | — | **.0268** (2배 이상!) |

`peer_relationship` 은 함께 있을 때 .0122 로 5위지만, 짝을 빼면 **.0268** 로 뛴다.
**혼자였다면 훨씬 중요한 변수로 보였을 것이다.**

**이것이 실무에서 가장 흔한 오독이다.** "중요도 낮으니 이 변수는 빼자"고 판단하면,
사실은 짝과 정보를 나눠 갖고 있던 변수를 버리는 셈이 된다.

**해석 규칙**: 중요도를 읽기 전에 **상관행렬을 옆에 둔다**(3차시 산출물).
상관이 높은 변수 묶음은 **개별 순위가 아니라 묶음 단위로** 이야기한다 —
"또래 관계(친구지지·교우관계)가 중요하다"처럼.
</details>"""),

md("""## Step 5 — 오류 분석: 우리는 누구를 놓쳤나 🔍 (두 번째 봉우리)

지금까지는 "모델이 무엇을 봤나"를 물었다. 이제 **"모델이 누구에게 틀렸나"** 를 묻는다.
오늘 가장 심리학적인 장면이다.

out-of-fold 예측으로 학생을 네 집단으로 나눈다:

| | 실제 고스트레스 | 실제 일반 |
|---|---|---|
| **모델이 고스트레스라 함** | **TP** 맞게 찾음 | **FP** 잘못 지목 |
| **모델이 일반이라 함** | **FN 놓침** ⚠️ | **TN** 맞게 제외 |

**FN(놓친 학생)** 이 우리가 가장 걱정해야 할 집단이다.
만약 이 모델이 학교의 선별 도구로 쓰인다면, **도움이 필요한데 발견되지 않는 학생들**이다."""),

code(r'''from sklearn.model_selection import cross_val_predict

pred = cross_val_predict(LOGIT, Xtr, ytr, cv=cv)
grp = pd.Series(np.select(
    [(ytr == 1) & (pred == 1), (ytr == 1) & (pred == 0),
     (ytr == 0) & (pred == 1), (ytr == 0) & (pred == 0)],
    ["TP 맞게 찾음", "FN 놓침", "FP 잘못 지목", "TN 맞게 제외"]), index=ytr.index)

print(grp.value_counts().to_string())
print(f"\nrecall = TP / (TP+FN) = {(grp=='TP 맞게 찾음').sum()} / {int(ytr.sum())} "
      f"= {(grp=='TP 맞게 찾음').sum()/ytr.sum():.3f}")

look = ["self_esteem", "peer_support", "parenting_monitoring", "depression",
        "previous_acculturative_stress", "acculturative_stress_w6"]
prof = frame.loc[idx_tr].assign(집단=grp.values).groupby("집단")[look].mean()
prof.insert(0, "n", grp.value_counts())
print("\n집단별 프로파일 (평균)")
print(prof.round(3).to_string())'''),

code(r'''# ▶ 놓친 학생(FN)은 찾은 학생(TP)과 무엇이 달랐나?
fn = frame.loc[idx_tr][grp == "FN 놓침"]
tp = frame.loc[idx_tr][grp == "TP 맞게 찾음"]     # 비교 대상 = 맞게 찾은 집단
tn = frame.loc[idx_tr][grp == "TN 맞게 제외"]

d = pd.DataFrame({"FN(놓침)": fn[look].mean(), "TP(찾음)": tp[look].mean(),
                  "TN(맞게 제외)": tn[look].mean()})
print(d.round(3).to_string())

near = ((fn["acculturative_stress_w6"] <= 1.7).mean(), (tp["acculturative_stress_w6"] <= 1.7).mean())
print(f"\ncutoff({cutoff}) 바로 위(≤1.7) 비율:  FN {near[0]:.1%}  ·  TP {near[1]:.1%}")'''),
code(r'''# CHECK Step5
try:
    assert len(tp) > 0, "TP 집단 이름을 정확히 적어야 한다"
    assert fn["self_esteem"].mean() > tp["self_esteem"].mean(), "놓친 학생이 자아존중감이 더 높아야 한다"
    assert abs(fn["self_esteem"].mean() - tn["self_esteem"].mean()) < \
           abs(fn["self_esteem"].mean() - tp["self_esteem"].mean()), \
           "FN 의 프로파일이 TP 보다 TN 에 가까워야 한다"
    print("✅ PASS — 놓친 학생(FN)은 **보호요인이 갖춰진 학생들**이다.")
    print(f"   자아존중감 {fn['self_esteem'].mean():.2f} (TP {tp['self_esteem'].mean():.2f}), "
          f"친구지지 {fn['peer_support'].mean():.2f} (TP {tp['peer_support'].mean():.2f}), "
          f"우울 {fn['depression'].mean():.2f} (TP {tp['depression'].mean():.2f})")
    print("   → FN 의 프로파일은 TP 가 아니라 **TN(맞게 제외한 학생들)과 훨씬 비슷하다.**")
    print("   모델 입장에선 '멀쩡해 보이는' 학생들이었고, 그래서 놓쳤다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 집단 이름은 'TP 맞게 찾음' 이다.")'''),

md("""### Step 5 해석 — 이 발견이 뜻하는 것

**놓친 학생(FN) 136명의 프로파일**:

| | FN (놓침) | TP (찾음) | TN (맞게 제외) |
|---|---|---|---|
| 자아존중감 | **3.42** | 2.76 | 3.50 |
| 친구지지 | **4.45** | 3.58 | 4.57 |
| 부모 감독 | **3.42** | 2.85 | 3.52 |
| 우울 | **1.47** | 2.07 | 1.38 |

**FN 은 TP 가 아니라 TN 을 닮았다.** 자아존중감 높고, 친구지지 두텁고, 부모가 챙기고,
우울 낮은 학생들 — 그런데 **1년 뒤 고스트레스 집단에 속했다.**

이것을 어떻게 읽어야 하나. 두 가지 해석이 모두 가능하고, 둘 다 중요하다.

**해석 ① 완화 요인 — 경계선 효과**
FN 의 **75.7%** 는 6차 스트레스가 1.5~1.7 로 **cutoff 바로 위**다 (TP 는 44.5%).
즉 놓친 학생 상당수는 "간신히 고스트레스"인 사람들이고, 4차시에서 본 **동점 문제**의 연장선이다.
선을 조금만 움직였다면 이들은 애초에 음성이었을 수도 있다.

**해석 ② 심각한 경고 — 눈에 안 띄는 학생들**
그럼에도 남는 사실이 있다. **우리 모델이 놓치는 학생은 "겉보기에 멀쩡한" 학생들이다.**
만약 이 모델을 학교 선별 도구로 쓴다면, **가장 주목받지 못하는 학생들을 계속 놓친다.**
그리고 그건 이미 교사의 눈에도 잘 안 띄는 학생들일 가능성이 높다.

> 🔴 이것이 8차시 윤리 절에 반드시 들어가야 할 문장이다:
> **"이 모델의 오류는 무작위가 아니다. 특정한 종류의 학생에게 체계적으로 쏠려 있다."**

**FP(잘못 지목한 275명)** 도 볼 만하다. 이들은 위험 프로파일(낮은 자아존중감·높은 우울)을
가졌는데 **실제로는 고스트레스가 아니었다.** 심리학적으로는 **회복탄력적(resilient)** 사례이고,
"왜 이들은 괜찮았나"가 그 자체로 좋은 후속 연구 질문이다."""),

md("""## Step 6 — 그래서 이걸 뭐라고 불러야 하나 🔴

오늘의 마지막이자 가장 어려운 질문이다.

우리는 `peer_support`(친구지지)가 세 방법 모두에서 1위라는 것을 확인했다.
그럼 이렇게 써도 되나?

> ❌ "친구지지는 문화적응 스트레스의 **보호요인이다**."
> ❌ "친구 관계를 개선하면 스트레스가 **줄어든다**."

**둘 다 안 된다.** 왜인지 정확히 알아야 한다.

### 우리가 실제로 아는 것과 모르는 것

| 아는 것 ✅ | 모르는 것 ❌ |
|---|---|
| 중2 친구지지가 낮았던 학생이 중3에 고스트레스일 **확률이 높았다** | 친구지지가 **원인**인지 |
| 그 관계가 표본을 바꿔도 **버틴다**(부트스트랩) | 친구지지를 **높이면** 스트레스가 줄지 |
| 모델이 그 변수에 **실제로 기댄다**(permutation) | 제3의 원인이 둘 다 만든 것인지 |

**우리 설계의 강점 하나**: X 는 5차(중2), y 는 6차(중3)로 **시간 순서가 확보**돼 있다.
"스트레스 때문에 친구 관계가 나빠진 것 아니냐"는 역방향 설명을 **부분적으로** 배제한다.
같은 시점에 잰 횡단 자료보다 훨씬 낫다.

**그래도 인과는 아니다.** 예를 들어 —
**학급 분위기**가 좋은 반에서는 친구지지도 높고 문화적응 스트레스도 낮을 수 있다.
그러면 친구지지와 스트레스의 관계는 **학급 분위기라는 제3변수**가 만든 것이다.
우리는 학급 변수를 통제하지 않았다.

### 그래서 이렇게 쓴다

| | 표현 |
|---|---|
| ❌ | "친구지지가 고스트레스를 **예방한다**" |
| ❌ | "친구지지는 **보호요인으로 확인되었다**" |
| ❌ | "친구지지가 **가장 중요한 변수다**" |
| ✅ | "중2 시점 친구지지는 1년 뒤 고스트레스 집단 분류에 **가장 크게 기여한 예측변수**였다" |
| ✅ | "친구지지가 낮은 학생이 이후 고스트레스 집단에 속할 **가능성이 높았다**" |
| ✅ | "이 결과는 **개입 효과를 보장하지 않는다** — 인과 검증에는 실험·준실험 설계가 필요하다" |

> 🔴 오늘의 문장: **"모델이 무엇을 썼는가"와 "무엇이 원인인가"는 다른 질문이다.**
> 우리는 앞의 질문에만 답했다. 그리고 **그 사실을 정확히 적는 것**까지가 연구다."""),

md("""## Step 7 — 산출물"""),
code(r'''import matplotlib.pyplot as plt
from maps_risk import plots     # import 만 해도 한글 폰트가 잡힌다
import os; os.makedirs("reports/figures", exist_ok=True)

# 중요도 표를 저장한다 (계수 · permutation 두 모델을 한 파일에)
out = tbl[["계수", "계수_절댓값", "perm_로지스틱", "순위_계수", "순위_perm"]].copy()
out["perm_포레스트"] = both["포레스트"]
out.index.name = "feature"
out.sort_values("순위_계수").to_csv("reports/feature_importance.csv", encoding="utf-8-sig")
print("✅ reports/feature_importance.csv")

top = perm_logit.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(top["feature"], top["imp_mean"], xerr=top["imp_sd"], capsize=3)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("AUC 하락폭 (out-of-fold permutation importance)")
ax.set_title("모델이 기댄 변수 상위 10개  (※ 인과관계 아님)")
fig.tight_layout(); fig.savefig("reports/figures/feature_importance.png", dpi=150)
plt.show()
print("✅ reports/figures/feature_importance.png")

# 오류 분석 그림 — FN 이 TN 을 닮았다는 것을 한눈에
fig, ax = plt.subplots(figsize=(7.5, 3.8))
order = ["TN 맞게 제외", "FN 놓침", "FP 잘못 지목", "TP 맞게 찾음"]
sub = ["self_esteem", "peer_support", "parenting_monitoring", "depression"]
x = np.arange(len(sub)); w = 0.2
for i, gname in enumerate(order):
    ax.bar(x + i*w, prof.loc[gname, sub].values, w, label=gname)
ax.set_xticks(x + 1.5*w, sub, fontsize=9); ax.legend(fontsize=8)
ax.set_title("집단별 프로파일 — FN(놓침)은 TN 을 닮았다")
fig.tight_layout(); fig.savefig("reports/figures/error_analysis.png", dpi=150)
plt.show()
print("✅ reports/figures/error_analysis.png")

print(f"\n🔒 test {len(idx_te)}명 — 오늘도 열지 않았다. 다음 주에 딱 한 번 연다.")'''),

md("""## 💾 다음 차시를 위해 — 드라이브에 저장\n\n오늘 만든 것 중 **다음 차시가 재료로 쓰는 파일**을 내 드라이브(`program5_state/`)에 넣어 둔다.\n이렇게 해 두면 런타임이 끊겨도, 다른 컴퓨터에서 열어도 **다음 차시가 그냥 시작된다.**\n\n> 🔴 파생 파일이 들어가는 폴더다 — **개인 계정 안에만** 두고 링크 공유·양도하지 않는다."""),
code(handoff_out(push=['reports/feature_importance.csv', 'reports/figures/*.png'], note="7차시 산출물 — 중요도 표를 8차시 보고서가 인용한다")),

md("""## 🎯 회고 (5분)

1. Permutation Importance 를 **train 에서 재면** 왜 안 되나? 실측으로 몇 배 차이가 났나?
2. `peer_relationship` 은 중요도 5위(.0122)인데, 짝을 빼면 .0268 로 뛴다.
   **"이 변수는 중요하지 않다"고 말해도 되나?**
3. 우리가 놓친 학생(FN)들은 자아존중감도 높고 친구지지도 두터웠다.
   **이 사실이 "이 모델을 학교에서 써도 되는가"라는 질문에 어떤 영향을 주나?**

3번이 오늘의 핵심 감각이다 — **모델의 오류는 무작위가 아니다. 누구에게 쏠리는지 봐야 한다.**

## 📝 과제
- 내가 맡은 변수의 **계수 · 부트스트랩 구간 · permutation 중요도** 세 값을 한 표로 만들고,
  세 방법이 일치하는지 판정 (일치하지 않으면 이유를 추정)
- FN 집단의 특징을 3문장으로 요약하고, **그것이 실무에 주는 함의**를 1문장으로
- "친구지지는 보호요인이다"를 **연구윤리에 맞게** 고쳐 쓰기 (Step 6 의 ✅ 표현 참고)

## ▶️ 다음 (8차시 — 마지막)
> "다음 주에 드디어 **test 를 연다.** 4차시부터 265명을 봉인해 뒀다 — 딱 한 번, 되돌릴 수 없다.
> 그 숫자가 CV 보다 낮게 나와도 **그대로 보고한다.** 그게 우리가 4주 동안 지켜 온 규칙이다.
> 그리고 **최종 보고서**를 쓴다 — 결론, 한계, 윤리. 남이 이 repo 를 받아 같은 결과를
> 재현할 수 있으면 이 프로그램은 성공이다."""),
]

os.makedirs("session7", exist_ok=True)
save(cells, "session7/session7.ipynb")

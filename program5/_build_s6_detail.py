# -*- coding: utf-8 -*-
"""session6_detail.ipynb 빌더 — 6차시 **상세 버전**.

기본 버전(`_build_s6.py` → `session6.ipynb`)의 흐름을 그대로 두고,
**결정 트리와 랜덤 포레스트가 실제로 어떻게 작동하는지**를 실습으로 파고든다.
기본 버전은 손대지 않는다 — 두 벌을 나란히 유지한다.

기본 대비 추가되는 것:
  Step 2 심화 — 지니 불순도 손계산 · 분할 후보 전수 탐색 · class_weight 가 지니에
                 하는 일 · 가지치기(min_samples_leaf) · 재표집 불안정성
  Step 4 심화 — 부트스트랩 63% 와 OOB · OOB score · max_features 와 트리 간 상관 ρ ·
                 분산 분해 · n_estimators 수렴 · 트리 vs 포레스트 흔들림 실측

★ 상세 실측 근거 (train 1,056 · seed 42 · Model A 18변수 — 기본 버전과 동일 상태)
  지니(깊이 1, 가중치 없는 raw) — 루트 n 1056 · 양성 .3371 · gini .4469
      첫 분할 peer_support <= 4.0714
      왼쪽  n 550 · 양성 .4400 · gini .4928
      오른쪽 n 506 · 양성 .2253 · gini .3491
      가중 평균 자식 .4239 → 정보 이득 .0230
      (class_weight="balanced" 를 적용한 sklearn 내부 지니는 .5000 → .4771 / .4629)
  분할 후보 — 18개 변수의 임계값 후보 합 294개 · peer_support 하나가 24개
  재표집 40회 첫 분기 변수 5종 — peer_support 15 · self_esteem 13 · depression 7
                                 · parenting_monitoring 3 · peer_relationship 2
  재표집 40회 예측확률 흔들림 — 트리 sd .1032 (최악 .2274) / 포레스트 sd .0367 (최악 .0768)
                                 → 트리가 2.81배 더 흔들린다
  가지치기 min_samples_leaf — 1: .5185(299잎) · 5: .5465(136잎) · 20: .6148(41잎)
                             · 50: .6003(15잎) · 100: .6352(8잎)
  부트스트랩 — 유일 표본 비율 실측 .6313 (이론 1−1/e = .6321) · OOB .3687
  OOB 점수 — AUC .6571 (정확도 .6278) vs 같은 설정 CV AUC .6651
  max_features (후보변수, 트리 간 상관 ρ, CV AUC)
      1(1) ρ .377 CV .6636 · 2(2) ρ .437 CV .6631 · sqrt(4) ρ .474 CV .6651
      · 6(6) ρ .494 CV .6633 · 12(12) ρ .488 CV .6633 · 없음(18) ρ .486 CV .6604
  n_estimators — 1: .6251 · 5: .6573 · 25: .6598 · 100: .6664 · 300: .6651 · 1000: .6653

  ※ 폐기한 실험: "seed 를 바꿔 트리와 포레스트의 흔들림 비교". 결정 트리는 같은
    데이터·같은 폴드에서 결정적이라 random_state 를 바꿔도 CV 가 .6355 로 고정된다
    (sd 0). 그 숫자를 쓰면 '트리가 포레스트보다 안정적'이라는 정반대 결론이 된다.
    트리의 불안정성은 모델 seed 가 아니라 **훈련 데이터 재표집**에서 드러난다.
"""
import os
import re

from nb import md, code, save
from _build_s6 import cells as base_cells


def insert_after(cells, marker, new_cells):
    """marker 를 포함한 셀 **바로 뒤**에 새 셀들을 끼운다.

    불변성: marker 는 기준 노트북에서 정확히 한 셀에만 나타난다.
      깨지면 → 0개면 심화 셀이 통째로 누락되고, 2개 이상이면 엉뚱한 자리에 들어간다.
      확인   → 아래 assert 가 매치 수를 센다. 기본 버전이 바뀌면 여기서 바로 터진다.
    """
    hits = [i for i, c in enumerate(cells) if marker in "".join(c["source"])]
    assert len(hits) == 1, f"마커 {marker!r} 가 {len(hits)}개 셀에서 발견됨 (1개여야 한다)"
    i = hits[0]
    return cells[:i + 1] + list(new_cells) + cells[i + 1:]


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 심화 — 결정 트리 안을 열어 본다
# ═════════════════════════════════════════════════════════════════════════════
TREE_CELLS = [
md("""## Step 2 심화 ① — 트리는 질문을 **어떻게 고르는가**

방금 트리는 `previous_acculturative_stress <= 1.45` 를 첫 질문으로 골랐다.
**이 변수와 이 숫자는 누가 정했나?** 우리가 정해 준 적이 없다. 트리가 계산해서 골랐다.

기준은 하나다 — **"나누고 나면 각 칸이 더 순수해지는가."**
'순수하다' = 한 칸 안에 고스트레스만, 또는 일반만 모여 있다는 뜻이다.
섞인 정도를 재는 자가 **지니 불순도(Gini impurity)** 다.

```
지니 = 1 − (고스트레스 비율)² − (일반 비율)²

  반반 섞임 → 1 − .5² − .5² = 0.500   ← 가장 지저분하다
  한 쪽만   → 1 − 1²  − 0²  = 0.000   ← 완벽하게 순수하다
```

**정보 이득(information gain)** = 분할 전 지니 − 분할 후 지니(자식들의 **가중** 평균).
트리는 이 값이 가장 큰 분할을 고른다. 직접 계산해 보자."""),

code(r'''# ▶ 지니를 손으로 계산해 sklearn 이 고른 분할과 맞춰 본다
def gini(y):
    """0/1 라벨 한 뭉치의 지니 불순도."""
    p = y.mean()
    return 1 - p**2 - (1 - p)**2

# 깊이 1 트리 = 딱 한 번만 나눈다 → 첫 분할이 무엇인지 바로 보인다
stump = build(DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                     random_state=cfg["random_seed"])).fit(Xtr, ytr)
t = stump.named_steps["clf"].tree_
feat, thr = featsA[t.feature[0]], t.threshold[0]
print(f"트리가 고른 첫 분할: {feat} <= {thr:.4f}\n")

# 같은 분할을 우리 손으로 재현한다 (결측은 트리와 같은 방식으로 중앙값 대치)
Xi = pd.DataFrame(stump.named_steps["prep"].transform(Xtr), columns=featsA, index=Xtr.index)
left = Xi[feat] <= thr

g_root = gini(ytr)
g_left, g_right = gini(ytr[left.values]), gini(ytr[~left.values])
w_left = left.mean()
g_child = w_left * g_left + (1 - w_left) * g_right

print(f"  분할 전  n {len(ytr):,}  양성 {ytr.mean():.4f}  지니 {g_root:.4f}")
print(f"  왼쪽     n {left.sum():,}  양성 {ytr[left.values].mean():.4f}  지니 {g_left:.4f}")
print(f"  오른쪽   n {(~left).sum():,}  양성 {ytr[~left.values].mean():.4f}  지니 {g_right:.4f}")
print(f"\n  가중 평균 자식 지니 = {w_left:.3f}·{g_left:.4f} + {1-w_left:.3f}·{g_right:.4f} = {g_child:.4f}")
print(f"  정보 이득 = {g_root:.4f} − {g_child:.4f} = {g_root - g_child:.4f}")'''),

code(r'''# CHECK Step2심화-1
try:
    assert abs(g_root - 0.4469) < 5e-4, f"루트 지니가 {g_root:.4f} (기대 .4469)"
    assert g_child < g_root, "분할 후 지니가 분할 전보다 낮아야 한다"
    print(f"✅ PASS — 분할로 지니가 {g_root:.4f} → {g_child:.4f} 로 {g_root-g_child:.4f} 줄었다.")
    print("   트리는 이 '줄어드는 양'이 가장 큰 분할을 골랐다.")
    print(f"\n   그런데 sklearn 트리가 기록한 루트 지니는 {t.impurity[0]:.4f} 다 — 우리 계산과 다르다.")
    print("   틀린 게 아니다. 다음 셀에서 이유를 본다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: gini(y) 는 1 - p² - (1-p)² 이고 p 는 0/1 라벨의 평균이다")'''),

md("""<details><summary>💡 해설 — 왜 sklearn 의 지니는 0.5 인가 (펼쳐 보기)</summary>

우리 손계산은 **0.4469**, sklearn 트리 내부는 **0.5000**. 둘 다 맞다.

4차시에서 **불균형**을 배웠다. 우리 데이터의 양성은 **33.7%** 뿐이라, 그냥 두면
트리가 "전부 일반"으로 몰리는 쪽이 유리해진다. 그래서 우리는 계속
`class_weight="balanced"` 를 써 왔다 — **소수 집단 한 명을 더 무겁게 세는** 옵션이다.

가중치를 주면 트리가 보는 세상에서는 양성과 음성이 **정확히 반반**이 된다.
그래서 루트 지니가 `1 − .5² − .5² = 0.5` 가 된다.

| | 손계산 (가중치 없음) | sklearn (`class_weight="balanced"`) |
|---|---|---|
| 루트 | .4469 | **.5000** |
| 왼쪽 자식 | .4928 | .4771 |
| 오른쪽 자식 | .3491 | .4629 |

**고른 분할은 똑같다.** 자를 바꿔도 "어디서 가장 많이 줄어드는가"의 답은 같았다.

> 이것이 오늘의 작은 교훈이다 — **숫자가 안 맞을 때 먼저 의심할 것은 계산 실수가
> 아니라 "서로 다른 것을 재고 있는가"** 이다.
</details>"""),

md("""## Step 2 심화 ② — 후보를 **전부** 시도한다

트리는 똑똑하게 찍지 않는다. **가능한 분할을 전부 계산해 보고** 가장 좋은 하나를 고른다.

변수 하나에서 시도하는 임계값은 그 변수의 **서로 다른 값 사이사이**다.
그걸 18개 변수 전부에 대해 한다."""),

code(r'''# ▶ 분할 후보가 실제로 몇 개인지 세어 본다
cand = {f: Xi[f].nunique() for f in featsA}
total = sum(cand.values())

print(f"변수 {len(featsA)}개 · 임계값 후보 총 {total}개")
print(f"  가장 많은 변수: {max(cand, key=cand.get)} ({max(cand.values())}개)")
print(f"  가장 적은 변수: {min(cand, key=cand.get)} ({min(cand.values())}개)")
print(f"  첫 분할로 뽑힌 {feat} 는 {cand[feat]}개\n")

# 그 후보들을 우리가 직접 훑어 이득이 가장 큰 지점을 찾아본다 (트리가 하는 일 그대로)
vals = np.sort(Xi[feat].unique())
mids = (vals[:-1] + vals[1:]) / 2          # 값과 값 사이의 중점이 후보다
best = max(((gini(ytr) - ((Xi[feat] <= m).mean() * gini(ytr[(Xi[feat] <= m).values])
                          + (Xi[feat] > m).mean() * gini(ytr[(Xi[feat] > m).values])), m)
            for m in mids), key=lambda p: p[0])
print(f"{feat} 안에서 이득이 가장 큰 임계값: {best[1]:.4f} (이득 {best[0]:.4f})")
print(f"트리가 고른 값                  : {thr:.4f}")
print("\n→ 트리는 이 전수 탐색을 모든 변수에 대해 하고, 그중 최고 하나만 채택한다.")'''),

md("""## Step 2 심화 ③ — 언제 멈추게 할 것인가 (가지치기)

Step 3 에서 `max_depth` 로 과적합을 막았다. 손잡이는 그것만이 아니다.

**`min_samples_leaf`** — "잎 하나에 최소 몇 명은 있어야 한다"는 규칙이다.
한 명짜리 잎을 못 만들게 하면 트리는 저절로 얕아진다."""),

code(r'''# ▶ 잎에 최소 인원을 요구하면 과적합이 얼마나 잡히나
from sklearn.model_selection import cross_val_score   # 여기서 처음 쓴다

rows = []
for msl in (1, 5, 20, 50, 100):
    est = build(DecisionTreeClassifier(min_samples_leaf=msl, class_weight="balanced",
                                       random_state=cfg["random_seed"]))
    auc = cross_val_score(est, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
    leaves = est.fit(Xtr, ytr).named_steps["clf"].get_n_leaves()
    rows.append({"min_samples_leaf": msl, "CV_AUC": auc, "리프수": leaves})

prune_tbl = pd.DataFrame(rows)
print(prune_tbl.round(4).to_string(index=False))
print("\n제한이 없으면(1) 리프 299개 · CV .5185 — Step 3 의 '깊이 제한 없음'과 같은 모델이다.")
print("잎마다 100명을 요구하면 리프 8개로 줄고 CV 가 .6352 까지 회복된다.")'''),

md("""**단조롭지 않다는 점**을 짚고 넘어가자 — `min_samples_leaf=50` 에서 한 번 내려간다(.6003).

"많이 자를수록 좋다"가 아니다. **적당한 지점이 있고, 그 지점은 데이터마다 다르다.**
그래서 우리는 이 값을 눈으로 고르지 않고 **`modeling.yaml` 그리드에 넣어 CV 에게 고르게** 한다.
Step 5 의 `GridSearchCV` 가 하는 일이 정확히 그것이다."""),

md("""## Step 2 심화 ④ — 트리의 약점: **흔들린다** ⚠️

Step 2 에서 "이 구조는 가설로 읽어라"라고 했다. **왜 그런지 실제로 재 본다.**

같은 데이터를 조금씩 다르게 뽑아(부트스트랩) 트리를 40번 다시 학습시키면,
**첫 질문**이 몇 가지로 갈릴까? 예측해 보고 실행하자. 🖐"""),

code(r'''# ▶ 재표집 40회 — '첫 질문'은 얼마나 바뀌는가
rng = np.random.default_rng(cfg["random_seed"])
first_splits = []
for _ in range(40):
    b = rng.integers(0, len(Xi), len(Xi))          # 복원추출
    t_b = DecisionTreeClassifier(max_depth=2, class_weight="balanced",
                                 random_state=cfg["random_seed"]).fit(Xi.iloc[b], ytr.iloc[b])
    first_splits.append(featsA[t_b.tree_.feature[0]])

vc = pd.Series(first_splits).value_counts()
print("40번 재표집했을 때 '첫 질문'으로 뽑힌 변수:\n")
print(vc.to_string())
print(f"\n→ {vc.size}가지로 갈린다. 데이터가 조금만 달라져도 트리의 얼굴이 바뀐다.")'''),

code(r'''# CHECK Step2심화-2
try:
    assert vc.size >= 3, f"첫 분기 변수가 {vc.size}가지 — 최소 3가지는 나와야 한다"
    top1 = vc.index[0]
    share = vc.iloc[0] / vc.sum()
    print(f"✅ PASS — 첫 질문이 {vc.size}가지로 갈린다. 1위 {top1} 도 {share:.0%} 에 그친다.")
    print("   '이 변수가 가장 중요하다'를 트리 그림 하나로 주장하면 안 되는 이유다.")
    print("   → 그래서 5차시 계수와 '겹치는지'를 봤고, 7차시에 다시 한 번 교차 확인한다.")
except Exception as e:
    print("❌ FAIL —", e)'''),

md("""<details><summary>💡 해설 — 트리의 약점은 사실 하나로 모인다 (펼쳐 보기)</summary>

| 약점 | 정체 |
|---|---|
| 데이터가 바뀌면 구조가 통째로 바뀐다 | **분산이 크다** |
| 깊이를 풀면 바로 과적합한다 | **분산이 크다** |
| 확률 추정이 거칠다 (잎 단위 비율) | **분산이 크다** |

트리는 **편향이 낮고 분산이 높은** 모델이다.
"어떤 모양이든 그릴 수 있지만(저편향), 데이터를 조금만 바꾸면 다른 걸 그린다(고분산)."

통계에서 **분산이 큰 추정량을 다루는 표준 처방**은 하나다 — **여러 개를 평균 낸다.**

3차시 심리척도가 정확히 그 논리였다. 문항 하나(단일 측정)는 흔들리지만
여러 문항의 평균은 오차가 상쇄돼 안정된다.

**랜덤 포레스트는 트리를 개선한 모델이 아니다.** 트리를 **그대로 두고**
여러 그루를 평균 내는 장치다. Step 4 에서 그 장치를 뜯어본다.
</details>"""),
]


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 심화 — 숲 안을 열어 본다
# ═════════════════════════════════════════════════════════════════════════════
FOREST_CELLS = [
md("""## Step 4 심화 ① — 배깅: 그루마다 **다른 표본**을 준다

300그루를 **같은 데이터**로 키우면 어떻게 될까? 트리는 결정적이다 —
**똑같은 트리 300개**가 나온다. 평균 내 봐야 한 그루와 같다.

그래서 그루마다 데이터를 흔든다. **부트스트랩(bootstrap)** — 1,056명에서
**복원추출**로 다시 1,056명을 뽑는다. 같은 학생이 두 번 뽑히기도 하고, 아예 안 뽑히기도 한다.

> 5차시에서 신뢰구간을 만들 때 쓴 그 기법이다. 같은 도구가 여기서는 **모델을 다양하게
> 만드는** 데 쓰인다.

**질문**: 복원추출로 n명을 뽑으면, 원래 n명 중 몇 %가 표본에 들어갈까? 🖐"""),

code(r'''# ▶ 부트스트랩 표본에 실제로 몇 %가 들어가나
rng = np.random.default_rng(cfg["random_seed"])
n = len(Xtr)
uniq = [len(np.unique(rng.integers(0, n, n))) / n for _ in range(200)]

print(f"n = {n:,} · 복원추출 200회 반복")
print(f"  표본에 들어간 비율 (실측 평균) : {np.mean(uniq):.4f}")
print(f"  이론값 1 − 1/e                 : {1 - np.exp(-1):.4f}")
print(f"  한 번도 안 뽑힌 비율 (OOB)     : {1 - np.mean(uniq):.4f}")
print("\n→ 그루마다 약 37% 의 학생은 '처음 보는 데이터'로 남는다.")
print("   이 남은 학생들을 out-of-bag(OOB) 이라고 부른다.")'''),

md("""## Step 4 심화 ② — 안 뽑힌 37% 가 **공짜 검증**이 된다

각 그루에게 OOB 학생들은 학습에 쓰이지 않은 데이터다.
그러니 **그 학생들로 채점하면 그것이 곧 검증 점수**다 —
CV 처럼 모델을 5번 다시 학습시킬 필요가 없다.

sklearn 은 `oob_score=True` 한 줄로 이걸 계산해 준다."""),

code(r'''# ▶ OOB 점수 vs 교차검증 점수
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

rf_oob = RandomForestClassifier(n_estimators=300, max_depth=3, class_weight="balanced",
                                random_state=cfg["random_seed"], n_jobs=-1,
                                bootstrap=True, oob_score=True)
prep = make_preprocessor(scale=False)
rf_oob.fit(prep.fit_transform(Xtr), ytr)

oob_auc = roc_auc_score(ytr, rf_oob.oob_decision_function_[:, 1])
cv_auc = cross_val_score(build(RandomForestClassifier(
    n_estimators=300, max_depth=3, class_weight="balanced",
    random_state=cfg["random_seed"], n_jobs=-1)), Xtr, ytr, cv=cv, scoring="roc_auc").mean()

print(f"OOB AUC          {oob_auc:.4f}   (추가 학습 0회 — 이미 만든 숲에서 바로 나온다)")
print(f"CV AUC (5-fold)  {cv_auc:.4f}   (5번 다시 학습해서 잰 값)")
print(f"차이             {abs(oob_auc - cv_auc):.4f}")
print("\n→ OOB 는 CV 의 값싼 대용품으로 쓸 만하다. 탐색 단계에서 시간을 크게 아낀다.")'''),

md("""> ⚠️ **그런데 우리 비교표는 왜 계속 CV 인가?**
>
> OOB 는 **포레스트에만 있다.** 로지스틱에도 Dummy 에도 없다.
> 모델끼리 비교하려면 **같은 자로** 재야 한다 — 그래서 Step 5 의 표는 전부 CV 다.
> (Step 5 슬라이드의 `CV=CV` 자물쇠가 이 이야기였다.)"""),

md("""## Step 4 심화 ③ — 변수까지 **무작위로** 고른다

표본만 흔들면 부족하다. **강한 변수 하나**가 있으면 그루 대부분이 그걸 첫 분기로
잡아서, 결국 **서로 닮은 트리**가 된다. 닮은 것끼리 평균 내면 이득이 없다.

그래서 랜덤 포레스트는 **분기마다** 전체 18개 중 **일부만** 후보로 보여 준다(`max_features`).
강한 변수가 후보에 없는 분기에서는 **2순위 변수가 기회를 얻는다.**

효과를 직접 재 보자 — 트리들이 서로 얼마나 닮았는지를 **예측값의 상관 ρ** 로 잰다."""),

code(r'''# ▶ max_features 를 바꿔가며 CV 와 '트리 간 상관 ρ' 를 함께 잰다  (30초쯤 걸린다)
from sklearn.model_selection import cross_val_score

rows = []
for mf in [1, 2, "sqrt", 6, 12, None]:
    est = RandomForestClassifier(n_estimators=300, max_depth=3, max_features=mf,
                                 class_weight="balanced", random_state=cfg["random_seed"], n_jobs=-1)
    auc = cross_val_score(build(est), Xtr, ytr, cv=cv, scoring="roc_auc").mean()

    est.fit(Xi, ytr)                                   # 상관을 보려고 전체 train 으로 한 번 더 학습
    P = np.array([t_.predict_proba(Xi.values)[:, 1] for t_ in est.estimators_])   # 300 × n
    C = np.corrcoef(P)
    rho = C[np.triu_indices_from(C, 1)].mean()         # 그루쌍 평균 상관

    rows.append({"max_features": str(mf), "후보변수": est.estimators_[0].max_features_,
                 "트리간_상관_rho": rho, "CV_AUC": auc})

mf_tbl = pd.DataFrame(rows)
print(mf_tbl.round(4).to_string(index=False))
print("\n→ 후보 변수를 늘릴수록 ρ 가 오른다 = 그루들이 서로 닮아 간다.")
print("   변수를 전부(18개) 보게 하면 ρ 가 높고 CV 가 가장 낮다(.6604).")
print("   기본값 sqrt(=4개)가 가장 좋다 — 이래서 그게 기본값이다.")'''),

md("""## Step 4 심화 ④ — 왜 평균이 분산을 줄이는가 (그리고 왜 ρ 가 관건인가)

트리 B개를 평균했을 때의 분산은 이렇게 쪼개진다:

```
Var( 트리 B개의 평균 )  =  ρ·σ²  +  (1 − ρ)·σ² / B

    σ² = 트리 하나의 분산 · ρ = 트리끼리의 상관 · B = 그루 수
```

- **B 를 키우면** 오른쪽 항 `(1−ρ)σ²/B` 가 0 으로 간다 → 그래서 그루를 늘려도 나빠지지 않는다.
- **그런데 왼쪽 항 `ρ·σ²` 는 남는다** → 그루를 아무리 늘려도 여기서 멈춘다.
- **그래서 ρ 를 낮춰야 한다** → 부트스트랩(다른 표본) + 변수 무작위 선택(다른 후보).

> 3차시 심리척도와 같은 논리다. 문항을 늘리면 α 가 오르지만, **문항들이 다 똑같은 걸
> 묻고 있으면** 아무리 늘려도 한계가 있다. 서로 다른 것을 묻는 문항이라야 평균이 이득이다.

**두 장치가 모두 ρ 를 낮추려고 있는 것이다.**"""),

code(r'''# ▶ 그루를 늘리면 과적합하나? (깊이와 달리 이건 '위험한 손잡이'가 아니다)
from sklearn.model_selection import cross_val_score

rows = []
for n_tree in (1, 5, 25, 100, 300, 1000):
    est = build(RandomForestClassifier(n_estimators=n_tree, max_depth=3, class_weight="balanced",
                                       random_state=cfg["random_seed"], n_jobs=-1))
    rows.append({"n_estimators": n_tree,
                 "CV_AUC": cross_val_score(est, Xtr, ytr, cv=cv, scoring="roc_auc").mean()})

n_tbl = pd.DataFrame(rows)
print(n_tbl.round(4).to_string(index=False))
print("\n→ 1그루 .6251 → 100그루 부근에서 사실상 평평해진다. 1,000그루도 300그루와 같다.")
print("   max_depth 와 달리 n_estimators 는 '너무 크면 과적합하는' 손잡이가 아니다 —")
print("   더 키우면 계산 시간만 는다. 그래서 300 정도에서 멈춘다.")'''),

md("""## Step 4 심화 ⑤ — 흔들림을 **실제로** 재 본다

Step 2 심화에서 트리의 첫 질문이 40번 중 5가지로 갈리는 걸 봤다.
포레스트는 그걸 얼마나 잡아 줄까?

이번엔 구조가 아니라 **예측확률**로 잰다 —
같은 학생의 예측확률이 재표집에 따라 얼마나 흔들리는지."""),

code(r'''# ▶ 재표집 40회 — 같은 학생의 예측확률이 얼마나 흔들리나  (40초쯤 걸린다)
rng = np.random.default_rng(cfg["random_seed"])
boots = [rng.integers(0, len(Xi), len(Xi)) for _ in range(40)]

def spread(make_model):
    """재표집마다 다시 학습해, 같은 학생들에 대한 예측확률의 표준편차를 낸다."""
    P = np.array([make_model().fit(Xi.iloc[b], ytr.iloc[b]).predict_proba(Xi.values)[:, 1]
                  for b in boots])
    sd = P.std(axis=0, ddof=1)
    return sd.mean(), sd.max()

tree_sd, tree_max = spread(lambda: DecisionTreeClassifier(
    max_depth=2, class_weight="balanced", random_state=cfg["random_seed"]))
rf_sd, rf_max = spread(lambda: RandomForestClassifier(
    n_estimators=300, max_depth=3, class_weight="balanced",
    random_state=cfg["random_seed"], n_jobs=-1))

print("학생 1명의 예측확률이 재표집에 따라 흔들리는 폭 (표준편차)\n")
print(f"  단일 트리 (깊이 2)      평균 ±{tree_sd:.4f}   가장 심한 학생 ±{tree_max:.4f}")
print(f"  랜덤 포레스트 (300그루)  평균 ±{rf_sd:.4f}   가장 심한 학생 ±{rf_max:.4f}")
print(f"\n→ 트리가 {tree_sd/rf_sd:.2f}배 더 흔들린다.")'''),

code(r'''# CHECK Step4심화
try:
    assert rf_sd < tree_sd, "포레스트가 트리보다 덜 흔들려야 한다"
    ratio = tree_sd / rf_sd
    assert ratio > 2, f"차이가 {ratio:.2f}배 — 2배 이상 나야 한다"
    print(f"✅ PASS — 포레스트의 흔들림이 트리의 1/{ratio:.2f} 다.")
    print("\n   여기서 오늘의 미묘한 지점이 하나 더 생긴다:")
    print("   Step 5 에서 포레스트가 로지스틱을 이긴 폭은 AUC +0.0116 뿐이었다.")
    print("   하지만 '안정성'은 성능 표에 아예 나오지 않는 이득이다.")
    print("   → 성능 표는 모델의 전부를 보여주지 않는다. 무엇이 표에 없는지도 물어야 한다.")
except Exception as e:
    print("❌ FAIL —", e)'''),

md("""### Step 4 심화 정리 — 포레스트의 네 손잡이

| 장치 | 무엇을 흔드나 | 왜 | 우리 설정 |
|---|---|---|---|
| **부트스트랩** | 학생 표본 | 그루마다 다른 세상을 보게 | 복원추출 1,056명 (유효 63.1%) |
| **max_features** | 분기별 후보 변수 | 강한 변수 독점을 막아 ρ 를 낮춤 | `sqrt` → 18개 중 **4개** |
| **n_estimators** | — | 평균의 표본 수 (많을수록 안정) | **300**그루 (100 이후 평평) |
| **max_depth** | — | 그루 하나의 복잡도 | CV 가 고른 **3** |

**포레스트가 산 것**: 안정성(흔들림 1/2.81) + CV AUC +0.0116
**포레스트가 판 것**: 그림으로 읽을 수 있던 **트리 한 그루** — 300그루는 그릴 수 없다.

> 이 거래가 우리 목적에 맞는지가 **Step 5 의 질문**이다. 바로 이어서 답한다."""),
]


# ═════════════════════════════════════════════════════════════════════════════
# 조립
# ═════════════════════════════════════════════════════════════════════════════
cells = list(base_cells)
cells = insert_after(cells, "### Step 2 해석 — 트리가 찾아낸 것", TREE_CELLS)
cells = insert_after(cells, "# 포레스트도 깊이를 늘리면 과적합할까?", FOREST_CELLS)

# 제목 셀만 '상세 버전'임을 밝히도록 바꾼다 (원본 리스트를 건드리지 않는다)
_head = "".join(cells[0]["source"])
assert _head.startswith("# 6차시 —"), "첫 셀이 제목 셀이 아니다"
cells[0] = md(_head.replace(
    "# 6차시 — 복잡한 모델이 늘 더 좋은 건 아니다",
    "# 6차시 (상세) — 복잡한 모델이 늘 더 좋은 건 아니다", 1).replace(
    "### 결정 트리 · 랜덤 포레스트 · 과적합 · 교차검증",
    "### 결정 트리 · 랜덤 포레스트 · 과적합 · 교차검증\n\n"
    "> **상세 버전** — 기본 버전(`session6.ipynb`)에 트리·포레스트의 **작동 원리**를\n"
    "> 파고드는 실습을 더했다. 지니 불순도 · 분할 탐색 · 가지치기 · 배깅 · OOB ·\n"
    "> 변수 무작위 선택 · 분산 분해. 결론과 산출물은 기본 버전과 같다.", 1))

if __name__ == "__main__":
    os.makedirs("session6", exist_ok=True)
    save(cells, "session6/session6_detail.ipynb")

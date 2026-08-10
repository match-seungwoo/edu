"""평가 — 여러 지표를 한 번에, Dummy 를 항상 옆에 두고 본다.

왜 존재하나: 불균형 이진분류에서 accuracy 하나만 보면 반드시 속는다.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def score_all(y_true, y_pred, y_prob=None):
    """한 모델의 예측 결과를 지표 dict 로 만든다.

    받는 것: 정답, 예측 라벨, 양성 확률(있으면)
    돌려주는 것: {roc_auc, average_precision, recall, precision, f1, balanced_accuracy}
    왜: 고스트레스 집단 '선별'이 목적이므로 recall 을 같이 읽어야 한다.
    """
    out = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def confusion_frame(y_true, y_pred):
    """혼동행렬을 사람이 읽을 수 있는 DataFrame 으로.

    왜: TP/FP/FN/TN 을 표로 봐야 "recall 0.75" 가 몇 명인지 감이 온다.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm,
                        index=["실제 0 (일반)", "실제 1 (고스트레스)"],
                        columns=["예측 0", "예측 1"])


def standardized_coefficients(fitted_pipeline, feature_names):
    """로지스틱 회귀의 표준화 계수를 크기순으로 돌려준다.

    받는 것: 학습된 Pipeline(표준화 포함), feature 이름
    돌려주는 것: coef / odds_ratio / abs_coef 컬럼을 가진 DataFrame
    왜: 입력이 이미 표준화돼 있으므로 계수 크기를 서로 비교할 수 있다.
    주의: 계수는 **관련성**이지 인과가 아니다.
    """
    coef = fitted_pipeline.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["odds_ratio"] = np.exp(df["coef"])
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def bootstrap_coefficients(estimator, X, y, n_boot=500, seed=42):
    """부트스트랩으로 표준화 계수의 **불확실성**을 잰다.

    받는 것:
      estimator  표준화를 포함한 Pipeline (아직 fit 하지 않은 것)
      X, y       train 데이터 (test 는 절대 넣지 않는다)
      n_boot     재추출 횟수
    돌려주는 것: 요약 DataFrame
      feature / coef(전체 train 계수) / boot_sd / ci_low / ci_high /
      includes_zero(신뢰구간이 0 을 포함하는가) / sign_consistency(부호 일관성)
    왜: 계수 순위표만 보면 "4위 변수"까지 해석하고 싶어진다. 그런데 표본이 조금만
        달라져도 부호가 바뀌는 계수가 섞여 있다. **점추정 옆에 불확실성을 같이 두면**
        해석해도 되는 것과 안 되는 것이 구분된다.
    불변성: ① 반환 순서 = X.columns 순서 = coef_ 순서 ② 각 재추출 표본에 두 클래스가
        모두 있어야 한다(없으면 다시 뽑고, 계속 실패하면 에러) ③ 매 반복 clone 으로
        새 estimator 를 쓴다(이전 fit 상태가 누적되지 않게).
    주의: 이것은 정식 추론 통계가 아니라 **안정성 진단**이다. p-value 로 읽지 않는다.
    """
    from sklearn.base import clone

    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    fitted_full = clone(estimator).fit(X, y)
    coef_full = fitted_full.named_steps["clf"].coef_[0]

    draws = []
    for _ in range(n_boot):
        for _attempt in range(10):
            idx = rng.choice(len(X), size=len(X), replace=True)
            ys = y.iloc[idx]
            if ys.nunique() > 1:      # 한 클래스만 뽑히면 로지스틱이 학습되지 않는다
                break
        else:
            raise ValueError("부트스트랩 표본에서 두 클래스를 못 얻었다 — 양성이 너무 적다")
        draws.append(clone(estimator).fit(X.iloc[idx], ys).named_steps["clf"].coef_[0])

    B = pd.DataFrame(draws, columns=cols)
    lo, hi = B.quantile(0.025), B.quantile(0.975)
    return pd.DataFrame({
        "feature": cols,
        "coef": coef_full,
        "boot_sd": B.std().values,
        "ci_low": lo.values,
        "ci_high": hi.values,
        "includes_zero": ((lo.values < 0) & (hi.values > 0)),
        "sign_consistency": (np.sign(B) == np.sign(coef_full)).mean().values,
    }).sort_values("coef", key=lambda c: c.abs(), ascending=False).reset_index(drop=True)


def permutation_scores_cv(estimator, X, y, cv, scoring="roc_auc",
                          n_repeats=10, seed=42):
    """교차검증으로 Permutation Importance 를 낸다 — 섞기는 **validation 폴드**에서만.

    받는 것:
      estimator  아직 fit 하지 않은 Pipeline
      X, y       train 데이터 (test 는 절대 넣지 않는다)
      cv         분할기 (모든 모델이 같은 것을 써야 비교가 성립한다)
    돌려주는 것: feature / imp_mean / imp_sd / n_folds_positive DataFrame (내림차순)
    왜: 학습에 쓴 데이터에서 섞으면, 과적합된 모델이 **외운 것**을 중요도로 보고한다.
        폴드마다 train 으로 fit 하고 **본 적 없는 validation 에서 섞어야** 일반화되는
        기여만 남는다. (실측: 같은 포레스트를 train 에서 재면 합계가 약 1.8배 부풀려진다.)
    불변성: ① 섞기는 validation 폴드에서만 ② 반환 순서 = X.columns 순서
        ③ 폴드마다 clone 으로 새 estimator (이전 fit 상태가 누적되지 않게).
    주의: 값이 음수면 "섞었더니 오히려 좋아졌다" = 그 변수는 도움이 안 됐다는 뜻이다.
    주의: 상관된 변수끼리는 **서로의 중요도를 가린다** — 낮게 나왔다고 중요하지 않은 것이 아니다.
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    cols = list(X.columns)
    per_fold = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        fitted = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(fitted, X.iloc[va], y.iloc[va], scoring=scoring,
                                   n_repeats=n_repeats, random_state=seed + k, n_jobs=-1)
        per_fold.append(r.importances_mean)

    folds = pd.DataFrame(per_fold, columns=cols)
    return (pd.DataFrame({"feature": cols,
                          "imp_mean": folds.mean().values,
                          "imp_sd": folds.std().values,
                          "n_folds_positive": (folds > 0).sum().values})
            .sort_values("imp_mean", ascending=False).reset_index(drop=True))


def permutation_scores(fitted_pipeline, X, y, feature_names,
                       scoring="roc_auc", n_repeats=20, seed=42):
    """Permutation Importance — 변수를 섞었을 때 성능이 얼마나 떨어지나.

    돌려주는 것: importance_mean / importance_std DataFrame (내림차순)
    왜: RandomForest 의 기본 impurity importance 는 고유값이 많은 변수를
        과대평가한다. 모델에 상관없이 쓸 수 있는 permutation 을 기본으로 쓴다.
    """
    r = permutation_importance(fitted_pipeline, X, y, scoring=scoring,
                               n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": feature_names,
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))

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

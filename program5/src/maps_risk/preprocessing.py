"""전처리 파이프라인 — 결측 대치와 표준화를 sklearn Pipeline 안에 가둔다.

왜 존재하나: "전체 데이터로 표준화한 뒤 split" 은 대표적인 누출이다.
Pipeline 안에 넣으면 fit 이 train 에서만 일어나는 것이 구조적으로 보장된다.
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor(scale=True):
    """결측 대치(중앙값) + 표준화 파이프라인을 만든다.

    받는 것: scale — 표준화 여부 (트리 계열은 불필요해서 끌 수 있다)
    돌려주는 것: sklearn Pipeline
    왜: 척도가 4점/5점으로 섞여 있어 로지스틱 계수를 비교하려면 표준화가 필수다.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def drop_high_missing(frame, feature_cols, max_rate=0.30):
    """결측률이 기준을 넘는 feature 를 제외한다.

    돌려주는 것: (남길 컬럼 리스트, 제외된 {컬럼: 결측률})
    왜: 결측이 절반인 변수를 중앙값으로 채우면 사실상 상수를 넣는 것과 같다.
    """
    rates = frame[feature_cols].isna().mean()
    keep = [c for c in feature_cols if rates[c] <= max_rate]
    dropped = {c: round(float(rates[c]), 4) for c in feature_cols if rates[c] > max_rate}
    return keep, dropped

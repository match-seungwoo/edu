"""분석용 데이터셋 조립 — 5차 X, 6차 y, 응답자 1명 = 1행.

왜 존재하나: "어떤 변수가 X 이고 어떤 변수가 y 인가"를 한 곳에서 결정해야
누출을 막을 수 있다.
"""
import pandas as pd

from . import scoring, validation
from .config import verified_constructs


def build_scores(df, spec_map, missing_codes, default_range=None):
    """구성개념 정의 dict 를 받아 척도 점수 DataFrame 을 만든다.

    받는 것:
      df           원자료 (한 차수)
      spec_map     {구성개념명: variables.yaml 의 정의}
      missing_codes 결측 코드 리스트
      default_range 문항 응답 범위 [min,max] (정의에 없을 때 사용)
    돌려주는 것: {구성개념명: 점수} DataFrame
    """
    out = {}
    for name, spec in spec_map.items():
        items = spec.get("items") or []
        clean = scoring.apply_missing_codes(df, items, missing_codes)
        rng = spec.get("expected_range") or default_range
        sc = spec.get("scoring") or {}
        out[name] = scoring.scale_score(
            clean, items,
            reverse_items=spec.get("reverse_items") or [],
            scale_range=rng,
            method=sc.get("method", "mean"),
            min_valid_items=sc.get("min_valid_items"),
        )
    return pd.DataFrame(out, index=df.index)


def build_modeling_frame(df5, df6, variables):
    """5차 예측변인 + 6차 문화적응 스트레스 점수를 한 표로 합친다.

    받는 것: 5차 DataFrame, 6차 DataFrame, variables.yaml dict
    돌려주는 것: 응답자 1명 = 1행인 DataFrame
        컬럼 = [id] + 5차 구성개념들 + previous_acculturative_stress(있으면)
                + acculturative_stress_w6
    왜: 이 함수 하나만 통과하면 X/y 구성이 항상 같은 규칙을 따르게 된다.
    """
    id5 = variables["id"]["wave5"]
    id6 = variables["id"]["wave6"]
    mc = variables.get("missing_codes") or []

    preds = verified_constructs(variables, "predictors")
    opts = verified_constructs(variables, "optional_predictors")
    tgt = variables["target"]

    x5 = build_scores(df5, preds, mc)
    x5.insert(0, "id", df5[id5].values)

    if opts:
        prior = build_scores(df5, opts, mc)
        x5 = pd.concat([x5, prior], axis=1)

    # 배경변수(단일 문항)는 점수 계산 없이 그대로 붙인다.
    for name, spec in (variables.get("background") or {}).items():
        col = spec.get("column")
        if spec.get("status") == "verified" and col and col in df5.columns:
            x5[name] = df5[col].values

    clean6 = scoring.apply_missing_codes(df6, tgt.get("items") or [], mc)
    y6 = pd.DataFrame({
        "id": df6[id6].values,
        "acculturative_stress_w6": scoring.scale_score(
            clean6, tgt.get("items") or [],
            reverse_items=tgt.get("reverse_items") or [],
            scale_range=tgt.get("expected_range"),
            method=(tgt.get("scoring") or {}).get("method", "mean"),
            min_valid_items=(tgt.get("scoring") or {}).get("min_valid_items"),
        ),
    })

    merged = x5.merge(y6, on="id", how="inner")
    return merged.dropna(subset=["acculturative_stress_w6"]).reset_index(drop=True)


def make_high_stress_label(train_scores, all_scores, quantile=0.75):
    """train 의 분위수로 cutoff 를 정하고, 그 cutoff 로 전체에 라벨을 붙인다.

    받는 것: train 의 스트레스 점수 Series, 라벨을 붙일 점수 Series, 분위수
    돌려주는 것: (라벨 Series 0/1, cutoff 값)
    왜: ★ cutoff 를 전체 데이터로 정하면 그것 자체가 test 정보 누출이다.
        반드시 train 에서만 계산한다.
    주의: 여기서 만든 1은 **조작적으로 정의한 고스트레스 집단**이지
          임상적 고위험군이 아니다.
    """
    cutoff = float(train_scores.quantile(quantile))
    return (all_scores >= cutoff).astype(int), cutoff


def split_features(frame, model_set="A"):
    """모델 세트에 따라 X 컬럼을 고른다.

    받는 것: build_modeling_frame 결과, "A" 또는 "B"
    돌려주는 것: feature 컬럼 이름 리스트
    왜: Model A/B 의 유일한 차이가 previous_acculturative_stress 하나임을
        코드로 못 박아 둔다.
    """
    drop = {"id", "acculturative_stress_w6", "high_stress"}
    cols = [c for c in frame.columns if c not in drop]
    if model_set.upper() == "A":
        cols = [c for c in cols if c != "previous_acculturative_stress"]
    return cols


def guard_leakage(feature_cols, df6_columns):
    """X 에 6차 변수나 target 이 섞이지 않았는지 최종 확인한다."""
    forbidden = {"acculturative_stress_w6", "high_stress"}
    bad = sorted(set(feature_cols) & forbidden)
    if bad:
        raise ValueError("target 계열 컬럼이 X 에 들어갔다 → " + ", ".join(bad))
    validation.assert_no_wave6_predictors(feature_cols, set(df6_columns))

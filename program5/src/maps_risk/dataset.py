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

    # 불변성: 각 차수 안에서 ID 는 유일·비결측이다. pandas merge 는 NaN 키끼리도
    # 매칭하므로, 깨진 채 진행하면 행이 조용히 불어난다(1명=1행 위반) → 즉시 중단.
    for df, idc, wave in ((df5, id5, "5차"), (df6, id6, "6차")):
        if idc not in df.columns:
            raise ValueError(f"{wave} ID 컬럼 '{idc}' 이 데이터에 없다")
        n_dup = int(df[idc].duplicated().sum())
        n_na = int(df[idc].isna().sum())
        if n_dup or n_na:
            raise ValueError(
                f"{wave} ID '{idc}' 가 유일하지 않다 (중복 {n_dup} · 결측 {n_na}) — "
                "이대로 병합하면 행이 불어난다. 원자료를 확인하라.")

    preds = verified_constructs(variables, "predictors")
    opts = verified_constructs(variables, "optional_predictors")
    tgt = variables["target"]

    x5 = build_scores(df5, preds, mc)
    x5.insert(0, "id", df5[id5].values)

    if opts:
        prior = build_scores(df5, opts, mc)
        x5 = pd.concat([x5, prior], axis=1)

    # 배경변수(단일 문항) — 점수 계산은 없지만 결측 코드 처리는 척도 문항과 똑같이 필요하다.
    # (9=무응답 같은 코드가 숫자로 남으면 뒤의 중앙값 대치·표준화가 오염된다.)
    bg = {n: s for n, s in (variables.get("background") or {}).items()
          if s.get("status") == "verified" and s.get("column")
          and s.get("column") in df5.columns}
    if bg:
        clean_bg = scoring.apply_missing_codes(
            df5, [s["column"] for s in bg.values()], mc)
        for name, spec in bg.items():
            col = clean_bg[spec["column"]]
            if spec.get("type") == "categorical" and col.nunique(dropna=True) > 2:
                # 다범주 변수를 숫자 하나로 두면 없는 서열을 만들어낸다 → one-hot.
                # (결측 행은 모든 dummy 가 0 이 된다.)
                try:
                    col = col.astype("Int64")   # 5.0 → 5 (dummy 컬럼명 정리)
                except (TypeError, ValueError):
                    pass
                x5 = pd.concat([x5, pd.get_dummies(col, prefix=name, dtype=float)],
                               axis=1)
            else:
                x5[name] = col.values

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
    merged = merged.dropna(subset=["acculturative_stress_w6"])

    # 구성개념 점수가 '전부' 결측인 행은 제외한다. MAPS 파일에는 그 차수
    # 미참여자도 행으로 들어 있어(응답은 전부 공백), 남겨 두면 뒤의 중앙값
    # 대치가 정보 없는 가짜 행을 만들어낸다. 배경변수(성별 등 관리 정보)만
    # 있는 행도 예측 정보가 없기는 마찬가지라 점수 컬럼 기준으로 판단한다.
    score_cols = [c for c in list(preds) + list(opts or {}) if c in merged.columns]
    if score_cols:
        merged = merged.dropna(subset=score_cols, how="all")

    return merged.reset_index(drop=True)


def make_high_stress_label(train_scores, all_scores, quantile=0.75):
    """train 의 분위수로 cutoff 를 정하고, 그 cutoff 로 전체에 라벨을 붙인다.

    받는 것: train 의 스트레스 점수 Series, 라벨을 붙일 점수 Series, 분위수
    돌려주는 것: (라벨 Series 0/1, cutoff 값)
    왜: ★ cutoff 를 전체 데이터로 정하면 그것 자체가 test 정보 누출이다.
        반드시 train 에서만 계산한다.
    주의: 여기서 만든 1은 **조작적으로 정의한 고스트레스 집단**이지
          임상적 고위험군이 아니다.
    주의: 문항 평균 점수는 이산적이라 cutoff 동점자가 몰리면 train 의
          양성 비율이 quantile 과 정확히 일치하지 않을 수 있다 → 실제 비율을
          항상 함께 보고한다.
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

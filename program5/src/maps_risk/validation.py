"""데이터 품질 검사 — 설계문서 §17.

왜 존재하나: 분석 전에 "이 데이터를 믿어도 되는가"를 기계적으로 묻는다.
문제를 발견하면 조용히 고치지 않고 **보고**한다.
"""
import pandas as pd


def check_id(df, id_col, wave_label):
    """응답자 ID의 존재·중복·결측을 확인한다.

    받는 것: DataFrame, ID 컬럼명, 표시용 차수 이름
    돌려주는 것: 검사 결과 dict
    """
    r = {"wave": wave_label, "id_col": id_col, "exists": id_col in df.columns}
    if not r["exists"]:
        return r
    s = df[id_col]
    r["n_rows"] = len(s)
    r["n_unique"] = int(s.nunique(dropna=True))
    r["n_missing"] = int(s.isna().sum())
    r["is_unique"] = bool(r["n_unique"] == len(s) and r["n_missing"] == 0)
    return r


def check_merge(df5, df6, id5, id6):
    """5차·6차 병합 성공률을 계산한다.

    돌려주는 것: {n_wave5, n_wave6, n_matched, match_rate_wave5, ...}
    왜: 패널 마모(attrition) 규모를 학생이 눈으로 봐야 한다.
    """
    s5, s6 = set(df5[id5].dropna()), set(df6[id6].dropna())
    matched = s5 & s6
    return {
        "n_wave5": len(s5), "n_wave6": len(s6), "n_matched": len(matched),
        "match_rate_wave5": round(len(matched) / len(s5), 4) if s5 else None,
        "match_rate_wave6": round(len(matched) / len(s6), 4) if s6 else None,
    }


def check_items_exist(df, items):
    """지정한 문항 컬럼이 실제 데이터에 있는지 확인한다.

    돌려주는 것: (있는 것 리스트, 없는 것 리스트)
    왜: 코드북과 실제 파일이 어긋나는 일은 흔하다. 없는 컬럼은 즉시 드러내야 한다.
    """
    have = [c for c in items if c in df.columns]
    miss = [c for c in items if c not in df.columns]
    return have, miss


def check_item_range(df, items, expected_range):
    """문항 값이 문서화된 응답 범위 안에 있는지 확인한다.

    받는 것: DataFrame, 문항 리스트, [최소, 최대]
    돌려주는 것: 범위를 벗어난 문항별 개수 dict (모두 정상이면 빈 dict)
    왜: 결측 코드(-9 등)가 숫자로 섞여 들어오면 평균이 완전히 망가진다.
    """
    if not expected_range:
        return {}
    lo, hi = expected_range
    bad = {}
    for c in items:
        if c not in df.columns:
            continue
        n = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n:
            bad[c] = n
    return bad


def missing_rate(df, cols):
    """컬럼별 결측률(0~1)을 Series 로 돌려준다."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].isna().mean().round(4)


def constant_columns(df, cols):
    """값이 하나뿐인(분산 0) 컬럼 목록 — 모델에서 쓸모없으므로 제거 후보."""
    out = []
    for c in cols:
        if c in df.columns and df[c].nunique(dropna=True) <= 1:
            out.append(c)
    return out


def assert_no_wave6_predictors(feature_names, wave6_columns):
    """feature 에 Wave 6 변수가 섞였는지 검사하고, 섞였으면 즉시 실패시킨다.

    받는 것: 모델에 들어갈 feature 이름 리스트, Wave 6 원자료 컬럼 집합
    돌려주는 것: None (문제 없으면)
    올리는 것: ValueError — 누출이 발견되면 파이프라인을 멈춘다
    왜: 이 프로젝트에서 가장 치명적인 오류가 시간 누출이기 때문.
    """
    leaked = sorted(set(feature_names) & set(wave6_columns))
    if leaked:
        raise ValueError(
            "데이터 누출: Wave 6 변수가 predictor 에 들어갔다 → " + ", ".join(leaked))

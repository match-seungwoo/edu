"""척도 점수 계산 테스트 — 역채점을 틀리면 이후 모든 결과가 틀린다."""
import numpy as np
import pandas as pd

from maps_risk import scoring


def test_reverse_code_flips_endpoints():
    s = pd.Series([1, 2, 3, 4])
    r = scoring.reverse_code(s, 1, 4)
    assert list(r) == [4, 3, 2, 1]


def test_reverse_code_is_involutive():
    """두 번 뒤집으면 원래대로 — 역채점 공식이 옳다는 최소 보증."""
    s = pd.Series([1, 2, 3, 4])
    assert list(scoring.reverse_code(scoring.reverse_code(s, 1, 4), 1, 4)) == list(s)


def test_scale_score_mean():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert list(scoring.scale_score(df, ["a", "b"])) == [2.0, 3.0]


def test_scale_score_applies_reverse():
    df = pd.DataFrame({"a": [1], "b": [1]})
    # b 를 역채점하면 4 → 평균 (1+4)/2 = 2.5
    out = scoring.scale_score(df, ["a", "b"], reverse_items=["b"], scale_range=[1, 4])
    assert out.iloc[0] == 2.5


def test_min_valid_items_blocks_partial_response():
    """응답 문항이 기준보다 적으면 점수를 만들지 않는다."""
    df = pd.DataFrame({"a": [1, 1], "b": [np.nan, 2]})
    out = scoring.scale_score(df, ["a", "b"], min_valid_items=2)
    assert pd.isna(out.iloc[0]) and out.iloc[1] == 1.5


def test_missing_codes_become_nan():
    df = pd.DataFrame({"a": [1, -9, 3]})
    out = scoring.apply_missing_codes(df, ["a"], [-9])
    assert pd.isna(out["a"].iloc[1])
    # 결측 처리를 안 하면 평균이 오염된다는 것을 같이 보인다
    assert scoring.scale_score(df, ["a"]).mean() < scoring.scale_score(out, ["a"]).mean()


def test_cronbach_alpha_range():
    rng = np.random.default_rng(0)
    base = rng.normal(size=300)
    df = pd.DataFrame({f"i{k}": base + rng.normal(scale=0.3, size=300) for k in range(5)})
    a = scoring.cronbach_alpha(df, list(df.columns))
    assert 0.7 < a <= 1.0


def test_cronbach_alpha_needs_two_items():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert np.isnan(scoring.cronbach_alpha(df, ["a"]))

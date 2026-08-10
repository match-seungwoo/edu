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


def test_sum_score_prorates_partial_response():
    """sum 방식에서 부분응답자의 점수가 체계적으로 낮아지면 안 된다."""
    df = pd.DataFrame({"a": [2, 2], "b": [4, np.nan]})
    out = scoring.scale_score(df, ["a", "b"], method="sum")
    assert out.iloc[0] == 6.0            # 완전응답: 2 + 4
    assert out.iloc[1] == 4.0            # 부분응답: 평균 2 × 2문항 (2가 아니라)


def test_sum_score_complete_response_unchanged():
    """완전응답자의 sum 은 문항 합과 정확히 같다 (보정의 회귀 불변성)."""
    df = pd.DataFrame({"a": [1, 4], "b": [3, 2], "c": [2, 1]})
    out = scoring.scale_score(df, ["a", "b", "c"], method="sum")
    assert list(out) == [6.0, 7.0]


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


def _scale_with(rng, n=400, noise=0.3):
    """공통 요인 하나로 움직이는 문항 4개를 만든다 (한 척도 흉내)."""
    base = rng.normal(size=n)
    return pd.DataFrame({f"i{k}": base + rng.normal(scale=noise, size=n)
                         for k in range(4)})


def test_item_total_correlation_is_negative_for_reversed_item():
    """방향이 뒤집힌 문항은 문항-전체 상관이 음수로 나온다 — 역채점 누락 신호."""
    rng = np.random.default_rng(1)
    df = _scale_with(rng)
    df["flipped"] = -df["i0"]                      # 방향만 반대인 문항
    r = scoring.item_total_correlations(df, list(df.columns))
    assert r["flipped"] < 0 < r["i1"]


def test_item_total_correlation_is_near_zero_for_unrelated_item():
    """무관한 문항은 음수가 아니라 0 근처 — 역채점으로는 해결되지 않는 경우."""
    rng = np.random.default_rng(2)
    df = _scale_with(rng)
    df["unrelated"] = rng.normal(size=len(df))     # 척도와 상관없는 문항
    r = scoring.item_total_correlations(df, list(df.columns))
    assert abs(r["unrelated"]) < 0.15


def test_alpha_if_deleted_flags_the_item_that_hurts():
    """척도를 깎아먹는 문항을 빼면 alpha 가 전체보다 높아진다."""
    rng = np.random.default_rng(3)
    df = _scale_with(rng)
    df["unrelated"] = rng.normal(size=len(df))
    items = list(df.columns)
    aid = scoring.alpha_if_deleted(df, items)
    assert aid.idxmax() == "unrelated"
    assert aid["unrelated"] > scoring.cronbach_alpha(df, items)


def test_alpha_if_deleted_needs_three_items():
    """2문항짜리 척도는 하나를 빼면 alpha 가 정의되지 않는다 → 빈 결과."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    assert scoring.alpha_if_deleted(df, ["a", "b"]).empty

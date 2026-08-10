"""데이터셋 조립 테스트 — 병합·라벨링·feature 선택."""
import numpy as np
import pandas as pd
import pytest

from maps_risk.config import unverified_constructs, verified_constructs
from maps_risk.dataset import (build_modeling_frame, make_high_stress_label,
                               split_features)
from maps_risk.validation import check_id, check_merge


def test_only_verified_constructs_are_used(fake_variables):
    v = verified_constructs(fake_variables, "predictors")
    assert set(v) == {"peer_support", "depression"}
    assert "not_yet_checked" in unverified_constructs(fake_variables, "predictors")


def test_id_uniqueness(fake_wave5):
    r = check_id(fake_wave5, "TEST_ID", "w5")
    assert r["is_unique"] and r["n_missing"] == 0


def test_merge_reports_attrition(fake_wave5, fake_wave6):
    m = check_merge(fake_wave5, fake_wave6, "TEST_ID", "TEST_ID")
    assert m["n_wave5"] == 200 and m["n_matched"] == 170
    assert m["match_rate_wave5"] == 0.85


def test_modeling_frame_is_one_row_per_respondent(fake_wave5, fake_wave6, fake_variables):
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    assert len(f) == 170                      # inner join 결과
    assert f["id"].is_unique
    assert "acculturative_stress_w6" in f.columns
    assert {"peer_support", "depression", "previous_acculturative_stress"} <= set(f.columns)


def test_duplicate_ids_stop_the_merge(fake_wave5, fake_wave6, fake_variables):
    """ID 가 중복되면 merge 가 행을 불린다 — 조용히 진행하지 말고 멈춰야 한다."""
    w5 = pd.concat([fake_wave5, fake_wave5.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="유일하지 않다"):
        build_modeling_frame(w5, fake_wave6, fake_variables)


def test_background_missing_code_becomes_nan(fake_wave5, fake_wave6, fake_variables):
    """배경변수도 결측 코드(-9)를 NaN 으로 바꿔야 한다 — 숫자로 남으면 오염."""
    w5 = fake_wave5.copy()
    w5["TEST_W5_SEX"] = [1, 2] * 100
    w5.loc[0, "TEST_W5_SEX"] = -9
    fake_variables["background"] = {
        "sex": {"wave": 5, "status": "verified", "column": "TEST_W5_SEX",
                "type": "categorical"}}
    f = build_modeling_frame(w5, fake_wave6, fake_variables)
    assert f["sex"].isna().sum() == 1
    assert -9 not in set(f["sex"].dropna())


def test_multicategory_background_is_one_hot(fake_wave5, fake_wave6, fake_variables):
    """다범주 categorical 배경변수는 서열 숫자가 아니라 one-hot 으로 들어간다."""
    w5 = fake_wave5.copy()
    w5["TEST_W5_REGION"] = [1, 2, 3, 4] * 50
    fake_variables["background"] = {
        "region": {"wave": 5, "status": "verified", "column": "TEST_W5_REGION",
                   "type": "categorical"}}
    f = build_modeling_frame(w5, fake_wave6, fake_variables)
    assert "region" not in f.columns
    dummies = [c for c in f.columns if c.startswith("region_")]
    assert len(dummies) == 4
    assert set(f[dummies].sum(axis=1)) == {1.0}   # 응답자당 정확히 한 범주


def test_rows_with_all_predictors_missing_are_dropped(fake_wave5, fake_wave6,
                                                      fake_variables):
    """5차 미참여자(전 문항 결측)는 target 이 있어도 분석 대상이 아니다."""
    w5 = fake_wave5.copy()
    item_cols = [c for c in w5.columns if c != "TEST_ID"]
    w5[item_cols] = w5[item_cols].astype(float)
    w5.loc[0, item_cols] = np.nan              # ID 1 = 5차 전 문항 무응답
    f = build_modeling_frame(w5, fake_wave6, fake_variables)
    assert 1 not in set(f["id"])
    assert len(f) == 169                       # 170 - 1


def test_cutoff_uses_training_data_only():
    """cutoff 는 train 분위수여야 한다 — 전체 분위수와 다를 수 있다."""
    all_scores = pd.Series(range(100), dtype=float)
    train = all_scores.iloc[:80]
    y, cutoff = make_high_stress_label(train, all_scores, 0.75)
    assert cutoff == train.quantile(0.75)
    assert y.iloc[:80].mean() == 0.25          # train 에서는 정확히 25%
    assert y.iloc[80:].mean() == 1.0           # test 는 우연히 달라질 수 있다


def test_model_a_excludes_prior_stress(fake_wave5, fake_wave6, fake_variables):
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    a = split_features(f, "A")
    b = split_features(f, "B")
    assert "previous_acculturative_stress" not in a
    assert "previous_acculturative_stress" in b
    assert set(b) - set(a) == {"previous_acculturative_stress"}

"""데이터셋 조립 테스트 — 병합·라벨링·feature 선택."""
import pandas as pd

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

"""누출 방지 테스트 — 이 프로젝트에서 가장 중요한 테스트.

시간 누출(Wave 6 변수를 predictor 로 사용)과 전처리 누출(전체 fit)을 둘 다 막는다.
"""
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from maps_risk.dataset import build_modeling_frame, guard_leakage, split_features
from maps_risk.models import build_models
from maps_risk.validation import assert_no_wave6_predictors

CFG = {"random_seed": 42,
       "models": {"dummy": {"enabled": True}, "logistic_regression": {"enabled": True},
                  "decision_tree": {"enabled": True}, "random_forest": {"enabled": True}}}


def test_wave6_predictor_raises():
    with pytest.raises(ValueError, match="누출"):
        assert_no_wave6_predictors(["peer_support", "W6_STRESS_01"], {"W6_STRESS_01"})


def test_target_column_cannot_enter_X(fake_wave5, fake_wave6, fake_variables):
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    with pytest.raises(ValueError, match="target"):
        guard_leakage(split_features(f, "B") + ["acculturative_stress_w6"],
                      fake_wave6.columns)


def test_features_contain_no_wave6_columns(fake_wave5, fake_wave6, fake_variables):
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    for mset in ("A", "B"):
        guard_leakage(split_features(f, mset),
                      set(fake_wave6.columns) - {"TEST_ID"})


def test_all_models_are_pipelines():
    """전처리가 Pipeline 밖에 있으면 CV 마다 train 으로만 fit 되지 않는다."""
    for name, (est, _) in build_models(CFG).items():
        assert isinstance(est, Pipeline), f"{name} 이 Pipeline 이 아니다"
        assert "prep" in est.named_steps


def test_scaler_is_fit_on_train_only(fake_wave5, fake_wave6, fake_variables):
    """train 으로 fit 한 스케일러의 평균이 train 평균과 일치해야 한다."""
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables).dropna()
    feats = split_features(f, "A")
    y = (f["acculturative_stress_w6"] >= f["acculturative_stress_w6"].quantile(0.75)).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(f[feats], y, test_size=0.2,
                                          random_state=42, stratify=y)
    pipe = build_models(CFG)["LogisticRegression"][0].fit(Xtr, ytr)
    scaler = pipe.named_steps["prep"].named_steps["scale"]
    assert abs(scaler.mean_[0] - Xtr[feats[0]].mean()) < 1e-9
    assert abs(scaler.mean_[0] - f[feats[0]].mean()) > 1e-12   # 전체 평균과는 다르다


def test_train_test_ids_do_not_overlap(fake_wave5, fake_wave6, fake_variables):
    f = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    tr, te = train_test_split(f["id"], test_size=0.2, random_state=42)
    assert set(tr).isdisjoint(set(te))


def test_cutoff_is_computed_from_train_only():
    """cutoff 는 train 분포로만 정해져야 한다 — 전체로 정하면 그 자체가 test 누출.

    train 과 전체의 분위수가 다르도록 test 쪽에 큰 값을 몰아 두고,
    돌려받은 cutoff 가 train 분위수와 같은지 확인한다.
    """
    import pandas as pd

    from maps_risk.dataset import make_high_stress_label

    train = pd.Series([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])       # q75 = 3.0
    test = pd.Series([90, 91, 92, 93, 94])                   # 전체로 계산하면 훨씬 커진다
    all_scores = pd.concat([train, test], ignore_index=True)

    _, cutoff = make_high_stress_label(train, all_scores, 0.75)
    assert cutoff == train.quantile(0.75)
    assert cutoff != all_scores.quantile(0.75), "전체 분포로 cutoff 를 정하고 있다 (누출)"


def test_cutoff_labels_apply_train_rule_to_everyone():
    """train 에서 정한 하나의 cutoff 를 전체에 그대로 적용해야 한다.

    test 를 test 자신의 분위수로 라벨링하면 train/test 의 라벨 의미가 달라진다.
    """
    import pandas as pd

    from maps_risk.dataset import make_high_stress_label

    train = pd.Series([1, 2, 3, 4])                          # q75 = 3.25
    all_scores = pd.Series([1, 2, 3, 4, 3.3, 3.2])
    labels, cutoff = make_high_stress_label(train, all_scores, 0.75)
    assert list(labels) == [0, 0, 0, 1, 1, 0]                # 3.3 >= 3.25 · 3.2 < 3.25
    assert (all_scores >= cutoff).astype(int).tolist() == list(labels)

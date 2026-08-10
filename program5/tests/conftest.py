"""테스트 공용 픽스처 — MAPS 원자료 없이도 로직을 검증할 수 있게 한다."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_wave5():
    """5차 흉내 데이터 (컬럼명은 TEST_ 접두어 — MAPS 아님)."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "TEST_ID": np.arange(1, n + 1),
        "TEST_W5_SUP_1": rng.integers(1, 5, n),
        "TEST_W5_SUP_2": rng.integers(1, 5, n),
        "TEST_W5_DEP_1": rng.integers(1, 5, n),
        "TEST_W5_DEP_2": rng.integers(1, 5, n),
        "TEST_W5_STR_1": rng.integers(1, 5, n),
        "TEST_W5_STR_2": rng.integers(1, 5, n),
    })


@pytest.fixture
def fake_wave6(fake_wave5):
    """6차 흉내 데이터. 일부 응답자는 탈락(attrition)한다."""
    rng = np.random.default_rng(7)
    ids = fake_wave5["TEST_ID"].iloc[:170]      # 30명 탈락
    return pd.DataFrame({
        "TEST_ID": ids.values,
        "TEST_W6_STR_1": rng.integers(1, 5, len(ids)),
        "TEST_W6_STR_2": rng.integers(1, 5, len(ids)),
    })


@pytest.fixture
def fake_variables():
    """검증 완료 상태의 variables.yaml 을 흉내 낸 dict."""
    return {
        "meta": {"predictor_wave": 5, "target_wave": 6, "codebook_verified": True},
        "id": {"wave5": "TEST_ID", "wave6": "TEST_ID"},
        "missing_codes": [-9],
        "target": {
            "name": "acculturative_stress_w6", "wave": 6, "status": "verified",
            "expected_range": [1, 4],
            "items": ["TEST_W6_STR_1", "TEST_W6_STR_2"], "reverse_items": [],
            "scoring": {"method": "mean", "min_valid_items": 1},
        },
        "predictors": {
            "peer_support": {"wave": 5, "status": "verified",
                             "items": ["TEST_W5_SUP_1", "TEST_W5_SUP_2"],
                             "reverse_items": [], "expected_range": [1, 4]},
            "depression": {"wave": 5, "status": "verified",
                           "items": ["TEST_W5_DEP_1", "TEST_W5_DEP_2"],
                           "reverse_items": [], "expected_range": [1, 4]},
            "not_yet_checked": {"wave": 5, "status": "unverified", "items": []},
        },
        "optional_predictors": {
            "previous_acculturative_stress": {
                "wave": 5, "status": "verified",
                "items": ["TEST_W5_STR_1", "TEST_W5_STR_2"],
                "reverse_items": [], "expected_range": [1, 4],
                "scoring": {"method": "mean", "min_valid_items": 1}},
        },
        "background": {},
    }

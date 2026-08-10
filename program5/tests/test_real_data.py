"""실데이터 구조 테스트 — data/raw 에 MAPS 원자료가 있을 때만 실행된다.

체크리스트(reports/codebook_candidates.md)에서 실측으로 확인한 구조적 사실이
계속 성립하는지 검증한다. 데이터가 없으면 전부 skip 되어 어느 환경에서도 안전하다.
여기 쓰인 변수명은 전부 코드북 + 실데이터 양쪽에서 확인된 것이다 (추측 금지 규칙).
"""
import unicodedata
from pathlib import Path

import pytest

from maps_risk.dataset import build_modeling_frame
from maps_risk.io import read_any

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _find(token):
    # macOS 는 한글 파일명을 NFD 로 저장한다 → NFC 로 맞춰 비교
    nfc = lambda s: unicodedata.normalize("NFC", s)
    hits = [p for p in RAW.rglob("*.csv") if token in nfc(p.name)] if RAW.exists() else []
    return hits[0] if hits else None


W5, W6 = _find("청소년 5차년도"), _find("청소년 6차년도")

pytestmark = pytest.mark.skipif(W5 is None or W6 is None,
                                reason="MAPS 원자료 없음 — 수령 후에만 실행")

# 코드북·실데이터 양쪽에서 확인된 최소 구성 (variables.yaml 이 아니다 — 구조 검증용)
VARS = {
    "meta": {"codebook_verified": True},
    "id": {"wave5": "PID", "wave6": "PID"},
    "missing_codes": [],
    "target": {"name": "acculturative_stress_w6", "wave": 6, "status": "verified",
               "expected_range": [1, 4],
               "items": [f"s_accul_str_{i:02d}_w6" for i in range(1, 11)],
               "reverse_items": [],
               "scoring": {"method": "mean", "min_valid_items": 8}},
    "predictors": {
        "depression": {"wave": 5, "status": "verified", "expected_range": [1, 4],
                       "items": [f"depression_{i:02d}_w5" for i in range(1, 11)],
                       "reverse_items": []}},
    "optional_predictors": {
        "previous_acculturative_stress": {
            "wave": 5, "status": "verified", "expected_range": [1, 4],
            "items": [f"s_accul_str_{i:02d}_w5" for i in range(1, 11)],
            "reverse_items": [],
            "scoring": {"method": "mean", "min_valid_items": 8}}},
    "background": {},
}


@pytest.fixture(scope="module")
def waves():
    df5, _ = read_any(W5)
    df6, _ = read_any(W6)
    return df5, df6


def test_pid_is_unique_and_id_is_not(waves):
    """join 키는 PID(개인)다 — ID 는 가구 ID 라 중복된다."""
    df5, df6 = waves
    assert df5["PID"].is_unique and df6["PID"].is_unique
    assert not df5["ID"].is_unique


def test_blank_missing_becomes_numeric_nan(waves):
    """공백(' ') 결측이 NaN 으로 읽혀 문항이 숫자 컬럼이 된다."""
    df5, _ = waves
    col = df5["s_accul_str_01_w5"]
    assert col.dtype.kind == "f"
    assert col.isna().sum() > 0               # 미참여자 몫


def test_modeling_frame_has_plausible_sample_size(waves):
    """미참여자(전 문항 공백)가 제외되어 분석 대상이 참여자 규모로 줄어야 한다."""
    df5, df6 = waves
    f = build_modeling_frame(df5, df6, VARS)
    assert f["id"].is_unique
    assert 1000 < len(f) < 1635               # 전원(1,635)이 그대로 남으면 실패

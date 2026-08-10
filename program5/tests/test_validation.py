"""데이터 품질 검사 테스트 — AGENTS.md 필수 항목(문항 존재·응답 범위)."""
from maps_risk.validation import check_item_range, check_items_exist


def test_check_items_exist_reports_missing(fake_wave5):
    have, miss = check_items_exist(fake_wave5, ["TEST_W5_SUP_1", "NOT_A_COLUMN"])
    assert have == ["TEST_W5_SUP_1"]
    assert miss == ["NOT_A_COLUMN"]


def test_check_item_range_catches_missing_code(fake_wave5):
    """결측 코드(-9)가 숫자로 섞여 들어오면 범위 검사에서 드러나야 한다."""
    w5 = fake_wave5.copy()
    w5.loc[0, "TEST_W5_SUP_1"] = -9
    bad = check_item_range(w5, ["TEST_W5_SUP_1", "TEST_W5_SUP_2"], [1, 4])
    assert bad == {"TEST_W5_SUP_1": 1}   # 정상 문항은 조용히, 이탈 문항만 보고

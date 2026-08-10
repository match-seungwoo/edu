"""원자료 읽기 테스트 — MAPS CSV 의 공백 결측이 NaN 으로 읽히는지."""
import pandas as pd

from maps_risk.io import find_raw_files, read_any


def test_csv_blank_string_is_missing(tmp_path):
    """MAPS CSV 는 결측이 공백 ' ' 로 들어 있다 — NaN 으로 읽어야 점수 계산이 산다."""
    p = tmp_path / "t.csv"
    p.write_text("a,b\n1, \n2,3\n", encoding="utf-8")
    df, _ = read_any(p)
    assert pd.isna(df.loc[0, "b"])
    assert df["b"].dtype.kind == "f"          # 문자열이 아니라 숫자 컬럼이 된다


def test_find_raw_files_recurses_into_subdirs(tmp_path):
    """zip 해제 후 csv/spss/stata 하위 구조도 찾아야 한다."""
    (tmp_path / "csv" / "youth").mkdir(parents=True)
    (tmp_path / "csv" / "youth" / "w5.csv").write_text("a\n1\n")
    (tmp_path / "top.xlsx").write_bytes(b"")
    found = {p.name for p in find_raw_files(tmp_path)}
    assert found == {"w5.csv", "top.xlsx"}

"""원자료 읽기 — 형식(.sav/.dta/.xlsx/.csv)에 상관없이 DataFrame 으로 올린다.

왜 존재하나: MAPS 파일 형식은 수령 전까지 확정되지 않는다.
읽기 로직을 한 곳에 모아 두면 형식이 바뀌어도 이 파일만 고치면 된다.
"""
from pathlib import Path

import pandas as pd

RAW_SUFFIXES = {".sav", ".dta", ".csv", ".xlsx", ".xls", ".parquet"}


def find_raw_files(raw_dir="data/raw"):
    """raw 폴더(하위 폴더 포함) 안의 데이터 파일 목록을 돌려준다.

    받는 것: raw 폴더 경로
    돌려주는 것: 정렬된 Path 리스트 (없으면 빈 리스트)
    왜: "데이터가 있는지부터 확인한다"가 이 프로젝트의 1번 규칙이기 때문.
        MAPS zip 을 풀면 csv/spss/stata 하위 폴더 구조가 생겨서 재귀로 찾는다.
    """
    d = Path(raw_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*")
                  if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)


def read_any(path):
    """확장자를 보고 알맞은 리더로 파일 하나를 읽는다.

    받는 것: 파일 경로
    돌려주는 것: (DataFrame, meta) — meta 는 SPSS 라벨 등 부가정보(없으면 None)
    왜: MAPS 는 보통 SPSS(.sav) 로 배포되며 값 라벨이 코드북 역할을 한다.
    """
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".sav":
        import pyreadstat  # 필요할 때만 import (미설치 환경에서도 나머지가 동작)

        df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
        return df, meta
    if suf == ".dta":
        import pyreadstat

        df, meta = pyreadstat.read_dta(str(path))
        return df, meta
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path), None
    if suf == ".parquet":
        return pd.read_parquet(path), None
    if suf == ".csv":
        # MAPS CSV 는 결측(미참여·무응답)이 공백 문자열 ' ' 로 들어 있다
        # (1기 5차 전 컬럼 실측: 비숫자 값은 ' ' 하나뿐).
        # NaN 으로 읽지 않으면 문항이 문자열이 되어 점수 계산이 깨진다.
        kw = {"na_values": ["", " "]}
        # 한글 CSV 는 cp949 인 경우가 흔하다. utf-8 실패 시 폴백.
        try:
            return pd.read_csv(path, encoding="utf-8", **kw), None
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp949", **kw), None

    raise ValueError(f"지원하지 않는 형식: {path.name}")


def describe_file(path):
    """파일 하나의 요약(행/열 수, 컬럼 일부)을 dict 로 돌려준다.

    받는 것: 파일 경로
    돌려주는 것: {name, format, n_rows, n_cols, columns_head, error}
    왜: data_inventory.md 를 사람이 읽을 수 있게 만들기 위한 재료.
    """
    info = {"name": Path(path).name, "format": Path(path).suffix.lower(),
            "n_rows": None, "n_cols": None, "columns_head": [], "error": None}
    try:
        df, _ = read_any(path)
        info["n_rows"] = len(df)
        info["n_cols"] = df.shape[1]
        info["columns_head"] = list(df.columns[:20])
    except Exception as e:  # 읽기 실패도 정보다 — 숨기지 않고 기록한다
        info["error"] = f"{type(e).__name__}: {e}"
    return info

"""Colab 배포용 program5 zip 을 만들고, 만든 zip 을 **다시 열어 검사**한다.

왜 존재하나: 손으로 압축하면 scripts/ 나 src/ 가 빠진 '반쪽 zip' 이 만들어져도
아무도 모른다. 그러면 Colab 에서 4차시쯤에야 "build_dataset.py 없음"으로 터진다.
이 스크립트는 넣을 것을 명시적으로 고르고, 만든 뒤 필수 파일을 zip 안에서 확인한다.

쓰는 법:
    python scripts/make_colab_zip.py                 # program5_colab.zip 생성
    python scripts/make_colab_zip.py --out ~/Desktop/program5.zip
    python scripts/make_colab_zip.py --with-spss-stata   # SPSS/STATA 원본까지 (87MB 더)

만든 zip 을 구글 드라이브 '내 드라이브'(또는 하위 2단계)에 올려 두면
노트북 맨 위 SETUP 셀이 알아서 찾아 /content 에 푼다.
"""
import argparse
import fnmatch
import os
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = "program5"          # zip 안의 최상위 폴더 이름 — SETUP 이 /content/program5 로 기대한다

# SETUP 의 is_project() 가 요구하는 것과 같은 목록이어야 한다.
REQUIRED = ["AGENTS.md", "configs/variables.yaml", "configs/modeling.yaml",
            "scripts/build_dataset.py", "src/maps_risk/__init__.py",
            "data/demo_format/DEMO_wave5_NOT_MAPS.csv",   # 1차시 Step 6 실습
            "data/demo_format/DEMO_wave6_NOT_MAPS.csv"]

# data/raw 에서 **실제로 참조되는** 파일만 고른다. 옆이 그 근거다 — 근거 없는 파일은
# 넣지 않는다 (STATA 75MB · SPSS 12MB · 학부모 자료는 어느 코드도 열지 않는다).
RAW_NEEDED = [
    ("csv/청소년*/*청소년 5차년도.csv",
     "예측변인 원자료 — _build_s2/s3 의 W5 · s4 find_wave(5) · codebook_candidates.py"),
    ("csv/청소년*/*청소년 6차년도.csv",
     "target 원자료 — _build_s2/s3 의 W6 · s4 find_wave(6) · codebook_candidates.py"),
    ("*청소년 코드북*.xlsx",
     "변수 후보 체크리스트 — codebook_candidates.py find_one() · 2차시 재료 점검"),
    ("*청소년 설문지*.pdf",
     "문항 원문 대조 — 2차시 재료 점검 셀이 존재를 확인한다"),
]
REQUIRED_RAW = ["data/raw/" + pat for pat, _ in RAW_NEEDED]

INCLUDE_DIRS = ["configs", "scripts", "src", "tests", "session1", "session2",
                "session3", "session4", "session5", "session6", "session7", "session8",
                "data/demo_format"]      # 1차시가 여는 가짜 5행 파일 (MAPS 아님)
INCLUDE_FILES = ["AGENTS.md", "README.md", "DATA_ACQUISITION.md", "pyproject.toml", "nb.py"]

# 용량만 잡아먹고 노트북이 열지 않는 것들
SKIP_PATTERNS = ["*/__pycache__/*", "*.pyc", ".DS_Store", "*/.DS_Store",
                 "*/.pytest_cache/*", "*.ipynb_checkpoints*"]


def nfc(s):
    """한글 파일명을 NFC 로 맞춘다.

    맥은 파일명을 NFD(자모 분해)로 저장하고 조회할 때 알아서 맞춰 준다. 리눅스(Colab)는
    안 그런다 — NFD 이름을 그대로 zip 에 담으면 Colab 에서 glob('*청소년 코드북*.xlsx')
    이 한 건도 못 찾는다. 그래서 **zip 에 넣는 이름을 NFC 로 통일**한다.
    """
    return unicodedata.normalize("NFC", s)


def skipped(rel):
    return any(fnmatch.fnmatch(rel, p) for p in SKIP_PATTERNS)


def collect(full_raw):
    """zip 에 넣을 (실제경로, zip 안 경로) 목록을 만든다."""
    items = []

    def add(path):
        rel = str(path.relative_to(ROOT))
        if skipped(rel) or not path.is_file():
            return
        items.append((path, nfc(f"{TOP}/{rel}")))

    for f in INCLUDE_FILES:
        add(ROOT / f)
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            add(p)

    # 원자료 — RAW_NEEDED 에 적힌 것만. 맥은 파일명을 NFD 로 저장하므로 NFC 로 맞춰 비교한다.
    raw = ROOT / "data" / "raw"
    all_raw = [p for p in sorted(raw.rglob("*")) if p.is_file()]
    if full_raw:
        print("   원자료: data/raw 전체 (--full-raw)")
        for p in all_raw:
            add(p)
    else:
        for pat, why in RAW_NEEDED:
            hits = [p for p in all_raw
                    if fnmatch.fnmatch(nfc(str(p.relative_to(raw))), nfc(pat))]
            if not hits:
                print(f"   🛑 없다: {pat}   ← {why}")
            for p in hits:
                print(f"   + {p.name}   ({why})")
                add(p)

    # 빈 폴더도 만들어 둔다 — 노트북이 여기에 산출물을 쓴다
    for d in ["data/interim", "data/processed", "reports/figures"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        gk = ROOT / d / ".gitkeep"
        if not gk.exists():
            gk.touch()
        add(gk)
    return items


def verify(zip_path):
    """만든 zip 을 다시 열어 필수 파일이 정말 들어 있는지 확인한다."""
    with zipfile.ZipFile(zip_path) as zf:
        names = {nfc(n) for n in zf.namelist()}
    missing = [f for f in REQUIRED if f"{TOP}/{f}" not in names]
    for pat in REQUIRED_RAW:
        if not fnmatch.filter(names, nfc(f"{TOP}/{pat}")):
            missing.append(pat + "  (원자료)")
    return missing, len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "program5_colab.zip"))
    ap.add_argument("--full-raw", action="store_true",
                    help="data/raw 전체를 넣는다 (SPSS·STATA·학부모 포함, 약 110MB 증가). "
                         "1차시 인벤토리를 '받은 자료 전부'로 보여주고 싶을 때만.")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    items = collect(args.full_raw)
    print(f"📦 {len(items)}개 파일 → {out}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in items:
            zf.write(src, arc)          # zipfile 은 항상 UTF-8 플래그를 세운다 (한글 안 깨짐)

    missing, n = verify(out)
    size = out.stat().st_size / 1e6
    print(f"   {n}개 엔트리 · {size:.1f} MB")
    if missing:
        print("\n🛑 반쪽 zip 이다 — 아래가 빠졌다. 올리지 말 것:")
        for m in missing:
            print("   -", m)
        sys.exit(1)
    print("\n✅ 검사 통과 — 필수 파일이 전부 들어 있다:")
    for f in REQUIRED:
        print("   ", f)
    print(f"\n→ 이 파일을 구글 드라이브 '내 드라이브'(또는 하위 2단계)에 올린다.")
    print(f"   이름에 'program5' 가 들어가야 SETUP 셀이 찾는다: {out.name}")


if __name__ == "__main__":
    main()

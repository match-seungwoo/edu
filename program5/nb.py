"""
nb.py — 의존성 없는 Jupyter 노트북(.ipynb) 빌더.

세션 노트북은 '파이썬 빌더 스크립트'로 정의하고 이 헬퍼로 .ipynb를 만든다.
→ 노트북이 코드로 관리되어 리뷰·재생성·일괄수정이 쉽다.

사용:
    from nb import md, code, save, SETUP
    cells = [ md("# 제목"), code(SETUP), code("print('hi')") ]
    save(cells, "session1/session1.ipynb")

SETUP 은 1~8차시 노트북 맨 위에 들어가는 공용 환경설정 셀이다.
차시마다 복붙하지 않고 여기 한 곳에서만 고친다 — 고친 뒤 _build_s*.py 를
전부 다시 돌리면 8개 노트북에 일괄 반영된다.
"""
import json


# ── 모든 차시 노트북 공용 환경설정 셀 (단일 소스) ─────────────────────
SETUP = r'''# ── 프로젝트 환경 자동 설정 (Colab / 로컬 공용) ───────────────────────
# 이 셀은 모든 차시 노트북 맨 위에 동일하게 들어간다. 그냥 실행만 하면 된다.
#
# Colab 사용법: 구글 드라이브('내 드라이브' 하위 2단계까지) 아무 곳에나
#   program5 zip 을 하나 올려 두면 된다. 이 셀이 드라이브를 mount 하고
#   /content 에 압축까지 풀어 준다. 런타임이 끊겨도 이 셀만 다시 실행하면
#   되고, 32MB zip 을 매번 재업로드할 필요가 없다.
import os, sys, glob, zipfile

DRIVE_ZIP_PATTERNS = [
    "/content/drive/MyDrive/program5*.zip",
    "/content/drive/MyDrive/*/program5*.zip",
    "/content/drive/MyDrive/*/*/program5*.zip",
]


def is_project(path):
    """AGENTS.md 와 configs/variables.yaml 이 함께 있어야 program5 로 인정한다.

    왜 두 개를 보나: 이름만 같은 빈 폴더에 잘못 붙는 사고를 막는다.
    """
    return os.path.isfile(os.path.join(path, "AGENTS.md")) and \
           os.path.isfile(os.path.join(path, "configs", "variables.yaml"))


def find_project():
    """이미 풀려 있는 program5 폴더를 후보 경로에서 찾는다."""
    for c in [".", "program5", "..", "../program5", "/content/program5",
              "/content/edu/program5", os.path.expanduser("~/program5")]:
        if is_project(c):
            return os.path.abspath(c)
    return None


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def member_name(info):
    """UTF-8 플래그가 없는 zip 의 한글 파일명을 되살린다.

    맥 /usr/bin/zip 은 EFS(0x800) 플래그를 세우지 않는다. 그러면 zipfile 이
    이름을 cp437 로 잘못 디코딩해 'φòÖδ╢Ç…' 같은 폴더가 생기고, 뒤이어
    data/raw 스캔이 0개를 돌려준다. 원래 바이트로 되돌려 다시 읽는다.
    """
    if info.flag_bits & 0x800:          # 이미 UTF-8 로 제대로 읽힌 이름
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        return info.filename
    for enc in ("utf-8", "cp949"):      # 한글 zip 은 이 둘 중 하나다
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return info.filename


def setup_from_drive():
    """드라이브를 mount 하고 program5 zip 을 /content 에 푼다. 경로 또는 None."""
    from google.colab import drive
    drive.mount("/content/drive")   # 이미 붙어 있으면 그대로 통과한다

    # MyDrive 전체를 재귀 탐색하면 느리다 — 하위 2단계까지만 훑는다.
    zips = [z for p in DRIVE_ZIP_PATTERNS for z in sorted(glob.glob(p))]
    if not zips:
        print("⚠️  드라이브에서 program5*.zip 을 찾지 못했다.")
        print("   '내 드라이브' 또는 그 하위 2단계 폴더에 zip 을 두고 이 셀을 다시 실행하라.")
        print("   탐색한 위치:")
        for p in DRIVE_ZIP_PATTERNS:
            print("     ", p)
        return None

    src = zips[0]
    if len(zips) > 1:
        print(f"ℹ️  zip 후보 {len(zips)}개 중 첫 번째를 쓴다:",
              [os.path.basename(z) for z in zips])
    print(f"📦 {os.path.basename(src)} ({os.path.getsize(src) / 1e6:.1f} MB) → /content 에 푸는 중…")
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():      # zip 안에 program5/ 폴더가 들어 있다
            info.filename = member_name(info)
            zf.extract(info, "/content")
    return find_project()


PROJECT = find_project()
if PROJECT is None and in_colab():
    PROJECT = setup_from_drive()

if PROJECT is None:
    print("⚠️  프로젝트 폴더를 찾지 못했습니다. 아래 중 하나로 해결하세요:")
    print("  (A) Colab: 구글 드라이브 '내 드라이브'에 program5 zip 을 올리고 이 셀 재실행")
    print("  (B) Colab: 좌측 파일창에 zip 을 올린 뒤  !unzip -q -o program5*.zip")
    print("  (C) 로컬 : program5 폴더 안(또는 그 상위)에서 노트북을 열었는지 확인")
else:
    os.chdir(PROJECT)
    src_dir = os.path.join(PROJECT, "src")
    if src_dir not in sys.path:      # 셀을 여러 번 돌려도 중복 추가되지 않게
        sys.path.insert(0, src_dir)
    print("✅ 프로젝트 경로:", PROJECT)
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def _split(src):
    # ipynb source는 줄 끝 \n 유지하는 문자열 리스트가 표준
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]


def save(cells, path):
    for c in cells:
        c["source"] = _split(c["source"]) if isinstance(c["source"], str) else c["source"]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("saved", path, f"({len(cells)} cells)")

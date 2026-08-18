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


# program5 로 인정하려면 이 4개가 다 있어야 한다.
# 왜 4개나 보나: 이름만 program5 인 '반쪽 폴더'(예전에 일부만 풀렸거나 업로드가 끊긴 것)를
#   붙잡으면 한참 뒤 4차시에서 "scripts/build_dataset.py 없음"으로 터진다. 여기서 거른다.
REQUIRED = ["AGENTS.md", "configs/variables.yaml", "configs/modeling.yaml",
            "scripts/build_dataset.py", "src/maps_risk/__init__.py"]


def missing_parts(path):
    """그 폴더에서 REQUIRED 중 빠진 파일 목록. 비어 있으면 온전한 프로젝트다."""
    return [f for f in REQUIRED if not os.path.isfile(os.path.join(path, *f.split("/")))]


def is_project(path):
    return not missing_parts(path)


def find_project():
    """이미 풀려 있는 program5 폴더를 후보 경로에서 찾는다.

    온전한 폴더만 고른다. program5 처럼 보이는데 반쪽인 폴더는 건너뛰되,
    **무엇이 없어서 건너뛰었는지 반드시 출력한다** — 조용히 넘어가면 원인 못 찾는다.
    """
    found, half, seen = None, [], set()
    for c in [".", "program5", "..", "../program5", "/content/program5",
              "/content/edu/program5", os.path.expanduser("~/program5")]:
        if not os.path.isdir(c):
            continue
        real = os.path.realpath(c)          # 같은 폴더를 두 경로로 가리키면 한 번만 본다
        if real in seen:
            continue
        seen.add(real)
        miss = missing_parts(c)
        if not miss:
            found = found or os.path.abspath(c)
        elif os.path.isfile(os.path.join(c, "AGENTS.md")):   # program5 인 척하는 반쪽 폴더
            half.append((os.path.abspath(c), miss))
    for path, miss in half:
        print("⚠️  반쪽 폴더라 건너뛴다:", path)
        print("     없는 것:", ", ".join(miss))
    return found


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
    print("🛑 프로젝트 폴더를 찾지 못했다. 아래 중 하나로 해결한다:")
    print("  (A) Colab: 구글 드라이브 '내 드라이브'에 program5 zip 을 올리고 이 셀 재실행")
    print("  (B) Colab: 좌측 파일창에 zip 을 올린 뒤  !unzip -q -o program5*.zip")
    print("      ※ 위에 '반쪽 폴더' 경고가 떴다면 그 폴더를 지우고 다시 풀어야 한다:")
    print("        !rm -rf /content/program5   ← 그 뒤 이 셀 재실행")
    print("  (C) 로컬 : program5 폴더 안(또는 그 상위)에서 노트북을 열었는지 확인")
    print("\n지금 /content 에 있는 것:", sorted(os.listdir("/content"))[:20]
          if os.path.isdir("/content") else "(없음)")
    # 여기서 멈춘다. 경고만 찍고 넘어가면 cwd 도 sys.path 도 안 잡힌 채로
    # 아래 셀들이 ModuleNotFoundError → FileNotFoundError 로 줄줄이 터진다.
    # sys.path 를 손으로 채워 봐야 cwd 가 여전히 /content 라 configs/*.yaml 을 못 읽는다.
    raise RuntimeError("program5 프로젝트 폴더를 찾지 못했다 — 위 안내대로 조치한 뒤 이 셀을 다시 실행하라.")

os.chdir(PROJECT)
src_dir = os.path.join(PROJECT, "src")
if src_dir not in sys.path:      # 셀을 여러 번 돌려도 중복 추가되지 않게
    sys.path.insert(0, src_dir)
print("✅ 프로젝트 경로:", PROJECT)
'''


# ── 차시 간 산출물 전달 (구글 드라이브) — 단일 소스 ────────────────────
# 왜 필요한가: Colab 런타임이 끊기면 /content 가 통째로 사라진다. 그러면 지난 차시에
# 만든 configs/variables.yaml · data/processed/modeling_frame.parquet 이 없어져
# 이번 차시를 시작할 수 없다 ("파일 의존" 문제).
#   → 각 차시 **끝**에서 다음 차시가 쓸 파일을 드라이브에 밀어 넣고(push),
#     각 차시 **시작**에서 필요한 파일을 당겨 온다(pull, 기본 덮어쓰기).
# 이 상수는 1~8차시 노트북에 동일하게 들어간다. 고칠 일이 있으면 여기 한 곳만 고친다.
HANDOFF = r'''# ── 차시 간 산출물 전달: 구글 드라이브에 저장/복원 ─────────────────────
# Colab 런타임은 끊기면 /content 가 사라진다. 그래서 "다음 차시가 필요로 하는 파일"은
# 내 드라이브에 따로 보관한다 — 그러면 차시 사이에 파일을 손으로 들고 다니지 않아도 된다.
#
#   저장 위치: 내 드라이브/program5_state/   (프로젝트와 같은 경로 구조로 쌓인다)
#     program5_state/configs/variables.yaml
#     program5_state/data/processed/modeling_frame.parquet
#     program5_state/reports/...
#
# 🔴 이 폴더에는 MAPS 원자료에서 파생된 파일이 들어간다. **개인 계정 안에만** 두고
#    링크 공유·양도하지 않는다 (MAPS 이용 조건). 공용 드라이브에 두지 말 것.
import filecmp as _filecmp
import glob as _glob
import os as _os
import shutil as _shutil

STATE_DIR = _os.environ.get("PROGRAM5_STATE_DIR")     # 로컬 테스트용 수동 지정


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _state_dir(create=False):
    """전달 폴더 경로. Colab 이 아니고 지정도 없으면 None 이라 그냥 건너뛴다."""
    global STATE_DIR
    if STATE_DIR:
        if create:
            _os.makedirs(STATE_DIR, exist_ok=True)
        return STATE_DIR
    if not _in_colab():
        return None
    from google.colab import drive
    drive.mount("/content/drive")          # 이미 붙어 있으면 그대로 통과한다
    STATE_DIR = "/content/drive/MyDrive/program5_state"
    _os.makedirs(STATE_DIR, exist_ok=True)
    return STATE_DIR


def handoff_push(patterns, label="다음 차시로 넘길 것을 드라이브에 저장"):
    """지금 만든 산출물을 드라이브에 저장한다.

    받는 것: 프로젝트 기준 상대경로 목록 (glob 가능. 예: reports/figures/*.png)
    돌려주는 것: 실제로 저장한 경로 리스트
    왜: 다음 차시가 이 파일을 "없으면 못 여는 재료"로 쓰기 때문이다.
    """
    print("📤 " + label)
    root = _state_dir(create=True)
    if root is None:
        print("   로컬 환경 — 저장을 건너뛴다 (파일이 이미 디스크에 그대로 남는다).")
        return []
    saved = []
    for pat in patterns:
        hits = sorted(_glob.glob(pat))
        if not hits:
            print("   ⬜ " + pat + " — 아직 없다 (이번 차시에서 만들지 않았다면 정상)")
            continue
        for src in hits:
            if not _os.path.isfile(src):
                continue
            dst = _os.path.join(root, src)
            _os.makedirs(_os.path.dirname(dst), exist_ok=True)
            _shutil.copy2(src, dst)
            saved.append(src)
            print("   ✅ " + src + "  →  드라이브")
    print("   저장 위치: " + root)
    return saved


def _yaml_completeness(path):
    """variables.yaml 이 얼마나 채워져 있는지 (게이트 열림, 검증된 구성개념 수).

    왜 필요한가: 드라이브에 **예전의 빈 variables.yaml** 이 남아 있는 경우가 있다.
    그걸 zip 의 검증본 위에 덮어쓰면 build_dataset.py 가 Human Review Gate 에서 멈춘다
    ("codebook_verified 가 false / 문항이 비어 있다"). 파일이 새것인지는 알 수 없어도
    **어느 쪽이 더 채워져 있는지**는 알 수 있다 — 덜 채워진 쪽으로는 덮어쓰지 않는다.
    """
    try:
        import yaml as _yaml
        d = _yaml.safe_load(open(path, encoding="utf-8")) or {}
    except Exception:
        return (0, 0)
    gate = bool((d.get("meta") or {}).get("codebook_verified"))
    n = sum(1 for sec in ("predictors", "optional_predictors")
            for spec in (d.get(sec) or {}).values()
            if spec.get("status") == "verified" and spec.get("items"))
    n += len((d.get("target") or {}).get("items") or [])
    return (int(gate), n)


# 파일별 '퇴보 방지' 검사. 드라이브 사본 점수가 지금 것보다 낮으면 그냥 둔다.
DOWNGRADE_GUARD = {"configs/variables.yaml": _yaml_completeness}


def handoff_pull(patterns, overwrite=True, label="지난 차시 산출물을 드라이브에서 복원"):
    """이번 차시에 필요한 파일을 드라이브에서 가져온다.

    받는 것: 상대경로 목록 (glob 가능), overwrite — 이미 있는 파일도 덮어쓸지 (기본 True)
    돌려주는 것: 실제로 가져온 경로 리스트
    왜 기본이 덮어쓰기인가: zip 안에 **같은 이름의 출발점 파일**이 이미 들어 있다
      (configs/variables.yaml · reports/model_metrics_cv.csv …). '없을 때만' 가져오면
      zip 의 옛 파일이 항상 이겨서 **지난 차시가 고친 내용이 영영 전달되지 않는다.**
      드라이브에 있는 것은 정의상 '지난 차시가 끝내고 밀어 넣은 최신본'이므로 그쪽을 쓴다.
    """
    print("📥 " + label)
    root = _state_dir()
    if root is None:
        print("   로컬 환경 — 복원을 건너뛴다 (디스크의 파일을 그대로 쓴다).")
        return []
    got = []
    for pat in patterns:
        hits = sorted(_glob.glob(_os.path.join(root, pat)))
        if not hits:
            print("   ⬜ " + pat + " — 드라이브에도 없다")
            continue
        for src in hits:
            rel = _os.path.relpath(src, root)
            exists = _os.path.exists(rel)
            if exists and _filecmp.cmp(src, rel, shallow=False):
                print("   ↩︎ " + rel + " — 드라이브와 내용이 같다 (그대로 둔다)")
                continue
            if exists and not overwrite:
                print("   ⚠️ " + rel + " — 드라이브 쪽과 다른데 덮어쓰지 않았다 (overwrite=False)")
                continue
            score = DOWNGRADE_GUARD.get(rel.replace(_os.sep, "/"))
            if exists and score and score(src) < score(rel):
                print("   🛡 " + rel + " — 드라이브 사본이 **더 비어 있다**. 지금 것을 그대로 쓴다.")
                print("       드라이브: " + str(score(src)) + " · 지금: " + str(score(rel))
                      + "   (게이트 열림, 검증된 구성개념 수)")
                print("       드라이브에 옛 파일이 남아 있는 것이다 — 이번 차시 끝에서 새것으로 덮인다.")
                continue
            _os.makedirs(_os.path.dirname(rel) or ".", exist_ok=True)
            _shutil.copy2(src, rel)
            got.append(rel)
            print(("   🔄 " if exists else "   ✅ ") + rel + "  ←  드라이브"
                  + ("  (zip 의 옛 파일을 덮어썼다)" if exists else ""))
    return got


def handoff_require(paths, hint=""):
    """이번 차시의 "없으면 못 여는 재료"를 확인한다. 없으면 이유를 알려준다."""
    missing = [p for p in paths if not _glob.glob(p)]
    if missing:
        print("\n🛑 이번 차시에 꼭 필요한 파일이 없다:")
        for m in missing:
            print("   -", m)
        if hint:
            print("   → " + hint)
        print("   → 지난 차시 노트북을 열어 **맨 끝의 '드라이브에 저장' 셀**을 실행한 뒤 돌아오라.")
    else:
        print("\n✅ 이번 차시에 필요한 재료가 전부 있다.")
    return not missing
'''


def handoff_in(pull=(), require=(), hint=""):
    """차시 시작 셀 소스 — 헬퍼 정의 + 드라이브에서 복원 + 필수 재료 확인."""
    src = HANDOFF + "\n\n"
    src += "handoff_pull([\n" + "".join('    "%s",\n' % p for p in pull) + "])\n"
    if require:
        src += "\nhandoff_require([\n" + "".join('    "%s",\n' % p for p in require) + "]"
        src += ',\n    hint="%s")\n' % hint if hint else ")\n"
    return src


def handoff_out(push=(), note=""):
    """차시 종료 셀 소스 — 다음 차시가 쓸 파일을 드라이브에 저장."""
    src = "# " + note + "\n" if note else ""
    src += "handoff_push([\n" + "".join('    "%s",\n' % p for p in push) + "])\n"
    return src


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

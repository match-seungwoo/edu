#!/usr/bin/env python3
"""data/raw 를 훑어보고 reports/data_inventory.md 를 만든다. 원자료는 건드리지 않는다.

실행:  python scripts/inspect_raw_data.py
      python scripts/inspect_raw_data.py --raw data/demo_format   (형식 확인용)

왜 이게 1번 스크립트인가: 분석 코드를 짜기 전에 "무엇을 받았는지"부터 종이에 적는다.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maps_risk.config import load_yaml, unverified_constructs  # noqa: E402
from maps_risk.io import describe_file, find_raw_files  # noqa: E402

NOT_FOUND_MSG = """\
## 🔴 원자료를 찾지 못했다

`{raw}` 폴더가 비어 있다. MAPS 는 신청제 자료라 웹에서 바로 받을 수 없다.

1. https://www.nypi.re.kr/archive → 데이터 다운로드 → 조사표/데이터/코드북
2. 동의 후 `데이터 신청서(신청자명).hwp` 다운로드
3. 패널 유형 칸에 **1기 패널 5차년도, 6차년도** 명시
4. `maps@nypi.re.kr` 로 제출 → 약 1주일

자세한 절차는 `DATA_ACQUISITION.md` 참고.

> 데이터가 없다는 사실을 **정확히 아는 것**도 결과다. 없는 데이터로 분석을 시작하지 않는다.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="reports/data_inventory.md")
    ap.add_argument("--config", default="configs/variables.yaml")
    args = ap.parse_args()

    files = find_raw_files(args.raw)
    # 데이터로 직접 못 읽는 파일(zip·pdf·hwp 등)도 받은 것은 받은 것이다 —
    # 조용히 빼놓으면 "무엇을 받았는지" 보고가 반쪽이 된다.
    raw_dir = Path(args.raw)
    others = []
    if raw_dir.exists():
        others = sorted(p for p in raw_dir.iterdir()
                        if p.is_file() and not p.name.startswith(".")
                        and p not in set(files))
    lines = [
        "# 데이터 인벤토리 (data_inventory.md)",
        "",
        f"> 자동 생성: `python scripts/inspect_raw_data.py --raw {args.raw}`  ·  {date.today()}",
        "> 이 문서는 원자료를 **읽기만** 하고 만든다. 원본은 수정하지 않는다.",
        "",
    ]

    if not files and not others:
        lines.append(NOT_FOUND_MSG.format(raw=args.raw))
    elif not files:
        lines += ["## 🟡 바로 읽을 수 있는 데이터 파일이 없다", "",
                  "아래 파일은 도착했지만 데이터로 직접 읽지 못했다.",
                  "**압축(zip)은 풀어야** 데이터가 보인다. PDF·hwp·xlsx 문서는 코드북/조사표/가이드다.", ""]
        lines += [f"- `{p.name}`" for p in others]
        lines += [""]
    else:
        demo = "demo" in args.raw
        if demo:
            lines += ["> ⚠️ **[DEMO] 형식 확인용 가짜 파일이다. MAPS 아님.**", ""]
        lines += [f"## 1. 발견한 파일 {len(files)}개", "",
                  "| 파일 | 형식 | 행 | 열 | 상태 |", "|---|---|---:|---:|---|"]
        infos = [describe_file(p) for p in files]
        for i in infos:
            state = i["error"] or "✅ 읽기 성공"
            lines.append(f"| `{i['name']}` | {i['format']} | "
                         f"{i['n_rows'] if i['n_rows'] is not None else '-'} | "
                         f"{i['n_cols'] if i['n_cols'] is not None else '-'} | {state} |")
        if others:
            lines += ["", f"## 1b. 데이터로 직접 읽지 않은 파일 {len(others)}개", "",
                      "압축(zip)은 풀어야 데이터가 보인다. PDF·hwp 문서는 코드북/조사표/가이드다.", ""]
            lines += [f"- `{p.name}`" for p in others]
        lines += ["", "## 2. 컬럼 미리보기 (앞 20개)", ""]
        for i in infos:
            if i["columns_head"]:
                lines += [f"**`{i['name']}`**", "",
                          "```", ", ".join(map(str, i["columns_head"])), "```", ""]

    # variables.yaml 과 대조 — 아직 사람이 확인해야 하는 것 목록
    cfg_path = Path(args.config)
    if cfg_path.exists():
        v = load_yaml(cfg_path)
        pend = unverified_constructs(v, "predictors")
        pend_opt = unverified_constructs(v, "optional_predictors")
        tgt_ok = bool((v.get("target") or {}).get("items"))
        lines += ["## 3. 사람이 확인해야 할 것 (Human Review Gate)", "",
                  f"- 코드북 확인 완료 플래그: **{v.get('meta', {}).get('codebook_verified')}**",
                  f"- 응답자 ID (5차/6차): **{v['id']['wave5']} / {v['id']['wave6']}**",
                  f"- 결측 코드: **{v.get('missing_codes') or '미확인'}**",
                  f"- target(6차 문화적응 스트레스) 문항: **{'확인됨' if tgt_ok else '미확인'}**",
                  "",
                  f"### 미검증 예측변인 {len(pend)}개", ""]
        lines += [f"- [ ] `{n}`" for n in pend] or ["- (없음)"]
        lines += ["", f"### 미검증 optional 예측변인 {len(pend_opt)}개", ""]
        lines += [f"- [ ] `{n}`" for n in pend_opt] or ["- (없음)"]
        lines += ["", "> 🔴 컬럼명을 추측해서 채우지 않는다. 코드북에서 확인한 것만 적는다.", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 생성: {out}  (데이터 {len(files)}개 · 기타 {len(others)}개 파일 검사)")
    if not files and not others:
        print("⚠️  원자료 없음 — DATA_ACQUISITION.md 의 신청 절차를 따르세요.")
    elif others:
        print(f"📦 직접 읽지 않은 파일 {len(others)}개 (zip/pdf 등) — 인벤토리 1b절 참고")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""5차·6차를 병합해 모델링용 데이터셋과 reports/data_quality.md 를 만든다.

실행: python scripts/build_dataset.py --wave5 <파일> --wave6 <파일>

이 스크립트는 `variables.yaml` 이 검증되기 전에는 **일부러 멈춘다**.
AGENTS.md 의 Human Review Gate 를 코드로 구현한 것이다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from maps_risk import validation  # noqa: E402
from maps_risk.config import (is_ready_for_scoring, load_configs,  # noqa: E402
                              unverified_constructs, verified_constructs)
from maps_risk.dataset import build_modeling_frame  # noqa: E402
from maps_risk.io import read_any  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave5", required=True)
    ap.add_argument("--wave6", required=True)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--out", default="data/processed/modeling_frame.parquet")
    args = ap.parse_args()

    variables, _ = load_configs(args.config_dir)

    ready, reasons = is_ready_for_scoring(variables)
    if not ready:
        print("🛑 아직 척도 점수를 계산할 수 없다 (Human Review Gate):")
        for r in reasons:
            print("   -", r)
        print("\n→ 코드북을 보고 configs/variables.yaml 을 채운 뒤 다시 실행하세요.")
        sys.exit(1)

    df5, _ = read_any(args.wave5)
    df6, _ = read_any(args.wave6)
    id5, id6 = variables["id"]["wave5"], variables["id"]["wave6"]

    # ── 품질 검사 ─────────────────────────────────────────────
    q = [check for check in (validation.check_id(df5, id5, "Wave 5 (2015, 중2)"),
                             validation.check_id(df6, id6, "Wave 6 (2016, 중3)"))]
    merge_info = validation.check_merge(df5, df6, id5, id6)

    tgt = variables["target"]
    have6, miss6 = validation.check_items_exist(df6, tgt.get("items") or [])
    bad_range = validation.check_item_range(df6, have6, tgt.get("expected_range"))

    frame = build_modeling_frame(df5, df6, variables)

    feat_cols = [c for c in frame.columns
                 if c not in ("id", "acculturative_stress_w6")]
    validation.assert_no_wave6_predictors(feat_cols, set(df6.columns) - {id6})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    # ── 보고서 ────────────────────────────────────────────────
    L = ["# 데이터 품질 보고서 (data_quality.md)", "",
         f"> 자동 생성 · Wave5=`{Path(args.wave5).name}` Wave6=`{Path(args.wave6).name}`", "",
         "## 1. 응답자 ID", "", "| 차수 | 컬럼 | 행 수 | 고유 ID | 결측 | 유일성 |",
         "|---|---|---:|---:|---:|---|"]
    for c in q:
        L.append(f"| {c['wave']} | `{c.get('id_col')}` | {c.get('n_rows','-')} | "
                 f"{c.get('n_unique','-')} | {c.get('n_missing','-')} | "
                 f"{'✅' if c.get('is_unique') else '❌'} |")
    L += ["", "## 2. 5차 ↔ 6차 병합", "",
          f"- Wave 5 응답자: **{merge_info['n_wave5']}명**",
          f"- Wave 6 응답자: **{merge_info['n_wave6']}명**",
          f"- 두 해 모두 응답(분석 대상): **{merge_info['n_matched']}명** "
          f"(5차 대비 {merge_info['match_rate_wave5']:.1%})",
          "", "> 사라진 응답자 = 패널 마모(attrition). 남은 표본이 원래 표본과",
          "> 체계적으로 다르면 결과 일반화에 한계가 생긴다 → 한계 절에 적는다.", "",
          "## 3. Target (6차 문화적응 스트레스)", "",
          f"- 문항 {len(tgt.get('items') or [])}개 중 데이터에 존재: **{len(have6)}개**"]
    if miss6:
        L.append(f"- ❌ 데이터에 없는 문항: `{', '.join(map(str, miss6))}`")
    if bad_range:
        L.append(f"- ❌ 응답 범위 이탈: {bad_range} → 결측 코드 확인 필요")
    s = frame["acculturative_stress_w6"]
    L += [f"- 점수 분포: n={s.notna().sum()}, 평균={s.mean():.3f}, "
          f"SD={s.std():.3f}, 최소={s.min():.2f}, 최대={s.max():.2f}", "",
          "## 4. 예측변인", "", "| 구성개념 | 결측률 |", "|---|---:|"]
    for c in feat_cols:
        L.append(f"| `{c}` | {frame[c].isna().mean():.1%} |")
    const = validation.constant_columns(frame, feat_cols)
    if const:
        L.append(f"\n- ⚠️ 상수 컬럼(분산 0): `{', '.join(const)}` → 제거 대상")
    pend = unverified_constructs(variables, "predictors")
    L += ["", "## 5. 아직 검증되지 않아 제외된 구성개념", ""]
    L += [f"- `{n}`" for n in pend] or ["- (없음)"]
    L += ["", "## 6. 누출 점검", "",
          "- ✅ X 에 Wave 6 컬럼 없음 (`assert_no_wave6_predictors` 통과)",
          f"- ✅ 검증된 예측변인만 사용 ({len(verified_constructs(variables))}개)", ""]

    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.md").write_text("\n".join(L), encoding="utf-8")
    print(f"✅ {args.out}  ({len(frame)}행 × {frame.shape[1]}열)")
    print("✅ reports/data_quality.md")


if __name__ == "__main__":
    main()

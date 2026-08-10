#!/usr/bin/env python3
"""Model A/B × 4개 모델을 학습·평가하고 표·그림·중요도를 만든다.

실행: python scripts/run_models.py

순서: 데이터 로드 → train/test split → train 에서 cutoff 계산 → CV 로 튜닝
      → test 1회 평가 → 지표/중요도/그림 저장.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from maps_risk import evaluation, plots  # noqa: E402
from maps_risk.config import load_configs  # noqa: E402
from maps_risk.dataset import make_high_stress_label, split_features  # noqa: E402
from maps_risk.models import build_models  # noqa: E402
from maps_risk.preprocessing import drop_high_missing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="data/processed/modeling_frame.parquet")
    ap.add_argument("--config-dir", default="configs")
    args = ap.parse_args()

    _, cfg = load_configs(args.config_dir)
    seed = cfg["random_seed"]

    frame = pd.read_parquet(args.frame)
    scores = frame["acculturative_stress_w6"]

    # ── split 먼저, cutoff 는 그 다음 (순서가 핵심) ──────────────
    # 진짜 라벨(high_stress)은 train-only cutoff 이후에야 생기므로 AGENTS.md 의
    # stratify=target 을 문자 그대로는 적용할 수 없다(순환). → 6차 점수의
    # median-split 을 임시 층화 기준으로 쓴다. median 이 전체 분포의 통계라는
    # 한계는 있지만, 분할 균형에만 쓰고 라벨 정의에는 쓰지 않는다.
    idx_tr, idx_te = train_test_split(
        frame.index, test_size=cfg["test_size"], random_state=seed,
        stratify=(scores >= scores.median()).astype(int))
    # 주의(미세 누출): cutoff 는 train 전체로 1회 계산한다. 엄밀하게는 CV 폴드마다
    # 재계산해야 validation 폴드의 라벨 정의에 그 폴드 정보가 안 들어간다.
    # 여기서는 관행대로 두되 4차시 토론 소재로 삼는다 — test 라벨은 train cutoff
    # 만으로 정의되므로 test 평가는 깨끗하다.
    y_all, cutoff = make_high_stress_label(
        scores.loc[idx_tr], scores, cfg["target"]["high_stress_quantile"])
    frame["high_stress"] = y_all
    print(f"cutoff(train {cfg['target']['high_stress_quantile']:.0%} 분위수) = {cutoff:.4f}")
    print(f"고스트레스 비율: train {y_all.loc[idx_tr].mean():.1%} / test {y_all.loc[idx_te].mean():.1%}")

    plots.class_distribution(frame["high_stress"])

    rows, imp_rows = [], []
    cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)

    for mset in ("A", "B"):
        feats = split_features(frame, mset)
        # 결측률 기준도 train 에서만 계산한다 — test 정보로 feature 를 고르지 않는다.
        feats, dropped = drop_high_missing(frame.loc[idx_tr], feats,
                                           cfg["missing"]["max_feature_missing_rate"])
        if dropped:
            print(f"[Model {mset}] 결측 과다로 제외: {dropped}")
        if not feats:
            print(f"[Model {mset}] 사용할 feature 가 없다 — 건너뜀")
            continue

        Xtr, ytr = frame.loc[idx_tr, feats], frame.loc[idx_tr, "high_stress"]
        Xte, yte = frame.loc[idx_te, feats], frame.loc[idx_te, "high_stress"]
        fitted = {}

        for name, (est, grid) in build_models(cfg).items():
            if grid:
                gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
                gs.fit(Xtr, ytr)
                best, params = gs.best_estimator_, gs.best_params_
                cv_auc = round(float(gs.best_score_), 4)
            else:
                best = est.fit(Xtr, ytr)
                params, cv_auc = {}, None

            prob = best.predict_proba(Xte)[:, 1] if hasattr(best, "predict_proba") else None
            m = evaluation.score_all(yte, best.predict(Xte), prob)
            rows.append({"model_set": mset, "model": name, "cv_roc_auc": cv_auc,
                         "best_params": str(params), "n_features": len(feats), **m})
            fitted[name] = best

            cmf = evaluation.confusion_frame(yte, best.predict(Xte))
            plots.confusion_heatmap(cmf, f"confusion_matrix_{name.lower()}_{mset}.png",
                                    f"{name} (Model {mset})")

            if name == "LogisticRegression":
                c = evaluation.standardized_coefficients(best, feats)
                c.insert(0, "model_set", mset)
                c.to_csv(f"reports/feature_importance_logistic_{mset}.csv", index=False)
                if mset == "A":
                    plots.importance_bar(c, "abs_coef", "feature_importance.png",
                                         title="로지스틱 표준화 계수 |크기| (Model A)")
            if name == "RandomForest":
                p = evaluation.permutation_scores(best, Xte, yte, feats, seed=seed)
                p.insert(0, "model_set", mset)
                p.to_csv(f"reports/feature_importance_random_forest_{mset}.csv", index=False)
                imp_rows.append(p)

        plots.roc_curves(fitted, Xte, yte, f"roc_curve_{mset}.png")
        plots.pr_curves(fitted, Xte, yte, f"precision_recall_curve_{mset}.png")

    Path("reports").mkdir(exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv("reports/model_metrics.csv", index=False)
    if imp_rows:
        pd.concat(imp_rows).to_csv("reports/feature_importance.csv", index=False)

    print("\n" + metrics.to_string(index=False))
    print("\n✅ reports/model_metrics.csv + figures 생성")
    print("※ 변수 중요도는 예측 기여도이지 인과효과가 아니다.")


if __name__ == "__main__":
    main()

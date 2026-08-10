"""run_models 가 지표 파일·그림을 실제로 만드는지 — AGENTS.md 필수 테스트."""
import importlib.util
import sys
from pathlib import Path

import yaml

from maps_risk.dataset import build_modeling_frame

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_models.py"


def _load_run_models():
    spec = importlib.util.spec_from_file_location("run_models", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_models_writes_metrics_and_figures(
        tmp_path, monkeypatch, fake_wave5, fake_wave6, fake_variables):
    frame = build_modeling_frame(fake_wave5, fake_wave6, fake_variables)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    frame.to_parquet(tmp_path / "data" / "processed" / "modeling_frame.parquet",
                     index=False)

    # 빠른 실행용 최소 설정 — 모델 목록이 설정 파일에서 오는 구조를 그대로 쓴다.
    cfg = {"random_seed": 42, "test_size": 0.2, "cv": {"folds": 3},
           "target": {"high_stress_quantile": 0.75},
           "missing": {"max_feature_missing_rate": 0.30},
           "models": {"dummy": {"enabled": True},
                      "logistic_regression": {"enabled": True, "C": [1.0]}}}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "modeling.yaml").write_text(
        yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "configs" / "variables.yaml").write_text(
        yaml.safe_dump(fake_variables), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_models.py"])
    _load_run_models().main()

    assert (tmp_path / "reports" / "model_metrics.csv").exists()
    assert (tmp_path / "reports" / "feature_importance_logistic_A.csv").exists()
    for fig in ("class_distribution.png", "roc_curve_A.png",
                "precision_recall_curve_B.png",
                "confusion_matrix_logisticregression_A.png"):
        assert (tmp_path / "reports" / "figures" / fig).exists(), fig

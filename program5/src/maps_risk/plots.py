"""그림 — reports/figures/ 에 저장한다. matplotlib 만 쓴다.

왜 존재하나: 발표 자료에 그대로 쓸 수 있는 그림을 코드로 재생성 가능하게.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 저장되게
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import (PrecisionRecallDisplay, RocCurveDisplay)

# 한글 제목·라벨이 □ 로 깨지지 않게 OS 별 한글 폰트를 찾아 지정한다.
# Colab 은 기본 한글 폰트가 없다 → !apt-get install -y fonts-nanum 후 런타임 재시작.
for _name in ("AppleGothic", "Malgun Gothic", "NanumBarunGothic", "NanumGothic"):
    if any(f.name == _name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _name
        break
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 부호도 같이 깨진다

FIG_DIR = Path("reports/figures")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def class_distribution(y, name="class_distribution.png", title="고스트레스 집단 분포"):
    """0/1 개수 막대그림. 불균형 정도를 눈으로 확인한다."""
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = y.value_counts().sort_index()
    ax.bar(["0 (일반)", "1 (고스트레스)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(title)
    ax.set_ylabel("응답자 수")
    return _save(fig, name)


def roc_curves(models, X, y, name="roc_curve.png"):
    """여러 모델의 ROC 곡선을 한 그림에 겹쳐 그린다."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, est in models.items():
        RocCurveDisplay.from_estimator(est, X, y, ax=ax, name=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve")
    return _save(fig, name)


def pr_curves(models, X, y, name="precision_recall_curve.png"):
    """Precision-Recall 곡선. 불균형 데이터에서는 ROC 보다 정직하다."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, est in models.items():
        PrecisionRecallDisplay.from_estimator(est, X, y, ax=ax, name=label)
    ax.set_title("Precision-Recall Curve")
    return _save(fig, name)


def confusion_heatmap(cm_frame, name, title):
    """혼동행렬을 숫자 박힌 히트맵으로."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm_frame.values, cmap="Blues")
    ax.set_xticks(range(len(cm_frame.columns)), cm_frame.columns)
    ax.set_yticks(range(len(cm_frame.index)), cm_frame.index)
    for i in range(cm_frame.shape[0]):
        for j in range(cm_frame.shape[1]):
            ax.text(j, i, cm_frame.values[i, j], ha="center", va="center")
    ax.set_title(title)
    return _save(fig, name)


def importance_bar(imp_frame, value_col, name="feature_importance.png",
                   top=10, title="상위 예측 기여 변수"):
    """변수 중요도 상위 N개 가로 막대그림."""
    d = imp_frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(d) + 1.5))
    ax.barh(d["feature"], d[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title + "  (※ 인과관계 아님)")
    return _save(fig, name)

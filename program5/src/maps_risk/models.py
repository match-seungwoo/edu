"""모델 정의 — Dummy / Logistic / DecisionTree / RandomForest 넷뿐.

왜 존재하나: 알고리즘 경쟁이 목적이 아니다. 해석 가능한 모델 셋과
"학습을 안 한 baseline" 하나를 같은 인터페이스로 세운다.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import make_preprocessor


def build_models(cfg):
    """modeling.yaml 을 받아 {모델이름: (Pipeline, 하이퍼파라미터그리드)} 를 만든다.

    받는 것: modeling.yaml 을 읽은 dict
    돌려주는 것: dict — 값은 (estimator, param_grid)
    왜: 모델 목록을 코드가 아니라 설정 파일이 결정하게 한다.
    """
    seed = cfg.get("random_seed", 42)
    m = cfg.get("models", {})
    out = {}

    if m.get("dummy", {}).get("enabled"):
        out["Dummy"] = (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", DummyClassifier(
                          strategy=m["dummy"].get("strategy", "most_frequent"),
                          random_state=seed))]),
            {},
        )

    if m.get("logistic_regression", {}).get("enabled"):
        lr = m["logistic_regression"]
        out["LogisticRegression"] = (
            # 표준화 필수 — 계수를 서로 비교하려면 같은 단위여야 한다.
            Pipeline([("prep", make_preprocessor(scale=True)),
                      ("clf", LogisticRegression(
                          max_iter=lr.get("max_iter", 2000),
                          class_weight="balanced", random_state=seed))]),
            {"clf__C": lr.get("C", [1.0])},
        )

    if m.get("decision_tree", {}).get("enabled"):
        dt = m["decision_tree"]
        out["DecisionTree"] = (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", DecisionTreeClassifier(
                          class_weight="balanced", random_state=seed))]),
            {"clf__max_depth": dt.get("max_depth", [3])},
        )

    if m.get("random_forest", {}).get("enabled"):
        rf = m["random_forest"]
        out["RandomForest"] = (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("clf", RandomForestClassifier(
                          n_estimators=rf.get("n_estimators", 300),
                          class_weight="balanced", random_state=seed, n_jobs=-1))]),
            {"clf__max_depth": rf.get("max_depth", [None])},
        )

    return out

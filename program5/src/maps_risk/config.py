"""configs/*.yaml 로딩 + 검증되지 않은 구성개념 걸러내기.

왜 존재하나: "컬럼명을 코드에 하드코딩하지 않는다"는 규칙을 강제하는 단일 통로.
"""
from pathlib import Path

import yaml


def load_yaml(path):
    """YAML 파일 하나를 dict 로 읽는다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(config_dir="configs"):
    """variables.yaml 과 modeling.yaml 을 함께 읽는다.

    돌려주는 것: (variables, modeling) 두 dict
    """
    d = Path(config_dir)
    return load_yaml(d / "variables.yaml"), load_yaml(d / "modeling.yaml")


def verified_constructs(variables, section="predictors"):
    """items 가 채워져 있고 status 가 verified 인 구성개념만 돌려준다.

    받는 것: variables.yaml 을 읽은 dict, 볼 섹션 이름
    돌려주는 것: {구성개념명: 정의dict}
    왜: 미검증 변수가 조용히 분석에 섞여 들어가는 것을 막는다.
    """
    out = {}
    for name, spec in (variables.get(section) or {}).items():
        if spec.get("status") == "verified" and spec.get("items"):
            out[name] = spec
    return out


def unverified_constructs(variables, section="predictors"):
    """아직 코드북 확인이 끝나지 않은 구성개념 이름 목록.

    왜: data_quality.md 에 "사람이 확인해야 할 것"으로 그대로 실어 보낸다.
    """
    return [n for n, s in (variables.get(section) or {}).items()
            if s.get("status") != "verified" or not s.get("items")]


def is_ready_for_scoring(variables):
    """척도 점수 계산을 시작해도 되는 상태인지 판정한다.

    돌려주는 것: (bool, 사유 문자열 리스트)
    왜: AGENTS.md 의 Human Review Gate — 게이트를 코드로 만든 것.
    """
    reasons = []
    if not variables.get("meta", {}).get("codebook_verified"):
        reasons.append("meta.codebook_verified 가 false 다 (사람이 코드북을 확인한 뒤 true 로 바꾼다)")
    if not variables.get("id", {}).get("wave5") or not variables.get("id", {}).get("wave6"):
        reasons.append("응답자 ID 컬럼명이 비어 있다")
    if not (variables.get("target") or {}).get("items"):
        reasons.append("target(6차 문화적응 스트레스) 문항이 비어 있다")
    if not verified_constructs(variables):
        reasons.append("검증된 예측변인이 하나도 없다")
    return (len(reasons) == 0), reasons

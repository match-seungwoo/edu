"""maps_risk — MAPS 1기 5차(2015) → 6차(2016) 고스트레스 집단 분류 교육용 툴킷.

설계 원칙 (AGENTS.md 와 1:1):
  1. 컬럼명을 추측하지 않는다. configs/variables.yaml 에서만 읽는다.
  2. Wave 6 변수는 target 생성에만 쓴다. predictor 로 들어가면 파이프라인이 실패한다.
  3. 전처리는 전부 sklearn Pipeline 안에서, train 으로만 fit 한다.
  4. 심리학 1학년이 읽을 수 있는 단순한 코드를 쓴다.
"""

__version__ = "0.1.0"

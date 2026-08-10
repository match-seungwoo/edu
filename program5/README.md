# Program 5 — MAPS 기반 다문화청소년 문화적응 스트레스 머신러닝

> **"중2 때의 심리사회적 환경으로, 중3 때의 높은 문화적응 스트레스를 미리 구분할 수 있는가?"**
>
> 다문화청소년패널조사(MAPS) 1기 실데이터로 **설명가능 머신러닝**을 훈련하는 8차시 프로그램.
> 심리학 전공 학생이 대상이며, 코드보다 **연구 설계와 해석의 정직성**이 중심이다.

영문 제목: *Predicting Subsequent High Acculturative Stress Among Multicultural
Adolescents Using MAPS: An Explainable Machine Learning Approach*

---

## 0. 지금 상태 🟡

| 항목 | 상태 |
|---|---|
| MAPS 원자료 (1기 5·6차) | ✅ **확보** (2026-08-10 수령) — `data/raw/`에 CSV·SPSS·STATA 해제됨 |
| 코드북 / 조사표 | ✅ 확보 — 청소년·학부모 코드북 xlsx, 조사표 PDF, 유저가이드 |
| 변수 후보 체크리스트 | ✅ 생성 — `python scripts/codebook_candidates.py` → `reports/codebook_candidates.md` |
| `configs/variables.yaml` | 전부 `status: unverified` — **체크리스트를 보고 사람이 검증 후 채운다** (2차시) |
| 파이프라인 뼈대 | ✅ 완성 (`pytest -q` 43개 통과 — 실데이터 구조 · 누출 방지 · 계수 안정성 테스트 포함) |
| 1차시 | ✅ 완성 (데이터 없이 진행 가능하도록 설계 · 원자료 수령 반영 갱신) |
| 2차시 | ✅ 완성 — 코드북 검증·게이트 열기 실습. **variables.yaml 은 수업에서 사람이 채운다** |
| 3차시 | ✅ 완성 — 신뢰도(α)·분포·상관 EDA. **2차시 판단을 데이터가 채점한다** (역채점 4문항 교정) |
| 4차시 | ✅ 완성 — 조작적 정의·split·불균형·**데이터 누출**. AUC 1.0 을 직접 만들어 본다 |
| 5차시 | ✅ 완성 — 로지스틱 계수·표준화·부트스트랩. **18개 중 3개만 해석 가능**하다는 결론 |
| 6차시 | ✅ 완성 — 트리·포레스트·과적합. 포레스트가 이기지만 **+0.012** 라는 비용-편익 판단 |
| 7~8차시 | 6차시 이후 제작 |

> **데이터가 없으면 `build_dataset.py` 는 일부러 멈춘다.** 추측으로 진행하지 않는 것이
> 이 프로젝트의 첫 번째 규칙이다.

---

## 1. 연구 설계

```
MAPS 1기 5차년도 (2015, 중2)          MAPS 1기 6차년도 (2016, 중3)
심리사회적 특성                   →    문화적응 스트레스
  자아탄력성 · 우울 · 사회적위축         (10문항 4점 SAFE 척도)
  가족지지 · 친구지지 · 교사지지                ↓
  학교적응 · 집단괴롭힘                  상위 25% = high_stress 1
  이중문화수용태도 · 국가정체성           (조작적 정의, 임상 진단 아님)
```

**연구 질문**

| | 질문 |
|---|---|
| **RQ1** | 중2 시점의 심리사회적 특성으로 1년 후 고스트레스 집단을 어느 정도 구분할 수 있는가? |
| **RQ2** | 그 구분에 상대적으로 중요한 심리사회적 변수는 무엇인가? |
| **RQ3** | 이전 시점(중2)의 문화적응 스트레스를 추가하면 예측력이 얼마나 개선되는가? |

**두 모델 세트**

| | 사용 변수 | 답하는 질문 |
|---|---|---|
| **Model A** | 5차 심리사회 변인만 | 이전 스트레스를 *모르는* 상태에서 주변 환경 정보만으로 어디까지 구분되나 |
| **Model B** | Model A + 5차 문화적응 스트레스 | 이전 상태를 알면 얼마나 나아지나 |

> A와 B의 차이 자체가 결과다. "과거 스트레스가 최강 예측변수"라는 흔한 결론을 **정량화**한다.

---

## 2. 이 프로젝트가 하지 않는 것

- 정신질환 진단 · 임상적 고위험군 판정
- 실제 학생에 대한 자동 개입 결정
- 인과관계 규명
- 학교 현장 배포용 위험예측 시스템

`high_stress = 1` 은 **학습 데이터의 상위 25%** 라는 조작적 분류다.
결과 서술은 항상 이렇게 한다:

> ❌ "고위험 청소년을 판별하였다"
> ✅ "본 연구에서 조작적으로 정의한 고스트레스 집단을 분류하였다"

---

## 3. 8차시 커리큘럼

각 차시 = **슬라이드(`sessionN.html`) + 실습 노트북(`sessionN.ipynb`) + 강사 대본(`lecture_notes.md`)**.

| 차시 | 심리학 | IT / ML | 산출물 | 완료 기준 |
|---|---|---|---|---|
| **1** | 문화적응 스트레스, 위험·보호요인, 예측 vs 인과, 연구윤리 | 데이터셋·row/column·feature/target·classification | `README.md` `data_inventory.md` | "무엇을 예측하고 어떤 데이터를 쓰는가"를 설명할 수 있다 |
| **2** | 심리척도·문항·역채점·척도점수 | pandas, 결측치, ID join | `variables.yaml` `data_quality.md` | 모든 변수의 컬럼명·의미·조사년도·문항범위가 확인됐다 |
| **3** | 평균·SD·분포·상관·**Cronbach α**, 문항-전체 상관 | 집계, 시각화, 데이터 클리닝 | `session3/`, `reports/figures/`, **갱신된 `variables.yaml`** | 척도 점수를 믿어도 되는지 판정하고, 무엇이 같이 움직이는지 설명할 수 있다 |
| **4** | 고스트레스 집단의 조작적 정의, 임상 cut-off와의 차이 | train/test split, 클래스 불균형, **데이터 누출**, baseline | `session4/`, `high_stress` 라벨, `test_no_leakage.py` | 누출 사례를 스스로 설명할 수 있다 |
| **5** | 예측변수와 결과의 관계·방향성, 상관 vs 편회귀계수 | 로지스틱 회귀, 확률, 계수, **표준화**, 부트스트랩 | `session5/`, `logistic_coefficients.png` | "어떤 변수가 가장 강하게 관련되나"에 답하고, **어디까지 답할 수 있는지**를 판정한다 |
| **6** | 심리 특성은 선형적으로 작동하는가 (역치·상호작용) | Decision Tree, Random Forest, **과적합**, CV | `session6/`, `model_metrics_cv.csv` | "복잡한 모델이 늘 더 좋은 건 아니다"를 데이터로 보인다 |
| **7** | 위험요인·보호요인, 인과 vs 예측 | Permutation Importance, 표준화 계수, 오류 분석 | `feature_importance.csv/.png` | "그 결과를 심리학적으로 어디까지 해석할 수 있나"에 답한다 |
| **8** | 결론·한계·윤리 서술 | 재현성, 최종 리포트 | `final_report.md` + 5~10분 발표자료 | 남이 repo를 받아 같은 결과를 재현할 수 있다 |

**4차시의 백미 — 일부러 누출시키기.** 6차 스트레스 문항으로 6차 고스트레스를 예측하는
"완벽한 모델"(AUC≈1.0)을 만들어 보이고, 왜 이것이 쓸모없는지 학생이 설명하게 한다.

---

## 4. 실행 방법

```bash
# 1. 데이터 확인 (원자료 없어도 실행된다 — 없다는 사실을 보고한다)
python scripts/inspect_raw_data.py

# 2. 형식 확인용 데모 (MAPS 아님)
python scripts/inspect_raw_data.py --raw data/demo_format

# 3. 테스트 (데이터 없이도 전부 통과해야 한다)
pytest -q

# ── 원자료 + 코드북 확보 후 (지금 단계) ──
# 4. 코드북 → 변수 후보 체크리스트 생성 (사람이 이걸 보고 variables.yaml 을 채운다)
python scripts/codebook_candidates.py

# ── variables.yaml 검증 완료 후 ──
python scripts/build_dataset.py \
  --wave5 "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 5차년도.csv" \
  --wave6 "data/raw/csv/청소년(1-10차_12차_14차)/다문화청소년패널 1기패널 청소년 6차년도.csv"
python scripts/run_models.py
```

> 4~6차시는 `session4/`~`session6/` 노트북 안에서 진행하며 **test 세트를 열지 않는다**
> — 모든 평가·계수 해석·모델 선택은 train 안 5-fold CV 로 한다.
> `run_models.py`(test 1회 평가)는 **8차시 전용**이다.
>
> 3차시(EDA·신뢰도)는 `session3/session3.ipynb` 안에서 진행한다. α 는 척도 점수가 아니라
> **문항 단위**로 계산하므로 원자료를 다시 읽는다. 3차시에서 `reverse_items` 를 교정한 뒤
> `build_dataset.py` 를 **다시** 실행해 `modeling_frame` 을 갱신한다.

`build_dataset.py` 는 `configs/variables.yaml` 이 검증되지 않았으면 **exit 1 로 멈춘다**
(AGENTS.md 의 Human Review Gate를 코드로 구현).

---

## 5. 파일 맵

```
program5/
├── README.md                 이 문서
├── AGENTS.md                 AI 코딩 도구용 마스터 프롬프트 (설계문서 §23)
├── DATA_ACQUISITION.md   ★   MAPS 원자료 확보 절차 + 확정된 설계 사실
├── pyproject.toml            의존성 / pytest 설정
├── nb.py                     노트북 빌더 헬퍼
├── _deck_template.html       슬라이드 템플릿
├── _build_s1.py …            노트북 재생성 스크립트 (program5 루트에서 실행)
├── configs/
│   ├── variables.yaml        ★ 구성개념 ↔ MAPS 컬럼 매핑 (코드북으로만 채움)
│   └── modeling.yaml         split·CV·모델·지표 설정
├── data/
│   ├── raw/                  MAPS 원자료 (Git 커밋 금지)
│   ├── interim/ processed/   파생 데이터
│   └── demo_format/          ⚠️ 5행 가짜 파일 (MAPS 아님, 형식 확인용)
├── src/maps_risk/
│   ├── io.py                 .sav/.dta/.csv/.xlsx 읽기
│   ├── config.py             YAML 로딩 + 검증 게이트
│   ├── validation.py         ID·병합·범위·결측·누출 검사
│   ├── scoring.py            역채점·척도점수·Cronbach α·문항-전체 상관·alpha-if-deleted
│   ├── dataset.py            5차 X + 6차 y 조립, cutoff, Model A/B 분기
│   ├── preprocessing.py      Pipeline(결측대치→표준화)
│   ├── models.py             Dummy/Logistic/Tree/Forest
│   ├── evaluation.py         지표·혼동행렬·표준화계수·부트스트랩 계수 안정성·permutation importance
│   └── plots.py              그림 저장
├── scripts/
│   ├── inspect_raw_data.py   → reports/data_inventory.md
│   ├── codebook_candidates.py → reports/codebook_candidates.md (사람 검증용 후보)
│   ├── build_dataset.py      → data/processed/ + reports/data_quality.md
│   └── run_models.py         → reports/model_metrics.csv + figures
├── tests/                    scoring / dataset / validation / io / no_leakage / outputs / real_data (43개)
├── reports/                  자동 생성물 (Git 제외) — model_metrics_cv.csv(6차시 CV) · model_metrics.csv(8차시 test)
└── session1/ … session8/     sessionN.html · sessionN.ipynb · lecture_notes.md
```

---

## 6. 방법론 원칙

**"계산은 코드, 해석은 사람, 인과 주장은 하지 않는다."**

| 규칙 | 왜 |
|---|---|
| 컬럼명을 추측하지 않는다 | 이름이 비슷하다고 같은 변수가 아니다. 코드북이 유일한 근거 |
| cutoff는 train에서만 계산 | 전체 분포로 정하면 그 자체가 test 정보 누출 |
| 전처리는 전부 Pipeline 안 | "전체 표준화 후 split"은 가장 흔한 누출 |
| test set은 마지막 1회만 | 튜닝에 쓰는 순간 test가 아니게 된다 |
| Dummy를 항상 같이 보고 | baseline보다 나은지 확인 못 하면 성능 숫자는 무의미 |
| accuracy 단독 보고 금지 | 불균형(25:75)에서는 "전부 0" 이 75% |
| 중요도 ≠ 인과 | "우울이 스트레스를 일으켰다"(X) / "예측 기여가 높았다"(O) |

---

## 7. 평가 루브릭 (100점)

| 영역 | 비중 |
|---|---:|
| 연구문제와 심리학적 이해 | 20% |
| 데이터와 척도 이해 | 20% |
| EDA 및 시각화 | 15% |
| 머신러닝 방법의 적절성 | 20% |
| 결과 해석 | 15% |
| 재현성·코드·AI 활용 | 10% |

> **모델 성능 자체에는 점수를 주지 않는다.** 성능이 낮아도
> "현재 변수만으로는 충분히 구분하기 어려웠다"는 결론을 정확히 도출했다면 성공이다.

---

## 8. AI 코딩 도구 사용 원칙

학생은 "코드 만들어줘"라고 시키지 않는다. 다음 순서로만 쓴다.

1. 내가 무엇을 분석하려는지 설명 → 2. 데이터 구조 제공 → 3. 코드북 정보 제공
→ 4. **작은 작업 하나** 요청 → 5. 실행 → 6. 결과 확인 → 7. **오류 원인을 스스로 설명**
→ 8. 수정

AI는 답안 생성기가 아니라 **pair programmer** 다. `AGENTS.md` 가 그 규칙을 담고 있다.

---

## 9. 정직성 / 한계 (미리 적어 둔다)

- **패널 마모(attrition).** 5차→6차 사이 탈락자가 있다. 남은 표본이 원 표본과
  체계적으로 다르면 일반화에 한계가 생긴다 → `data_quality.md` 에 병합률로 기록.
- **조작적 정의.** 상위 25%는 통계적 편의이지 임상 기준이 아니다. 분위수를 바꾸면
  결론도 바뀐다 → 민감도 분석(0.70/0.75/0.80)을 8차시에 권장.
- **척도 범위 불일치.** 4점·5점 척도가 섞여 있다 → 표준화 없이 계수를 비교하면 안 된다.
- **동점(ties).** 문항 평균 점수는 이산적이어서 cutoff 동점자가 몰린다. **실측**: 6차
  스트레스 점수의 75 백분위수는 1.500 인데 그 값에만 142명이 몰려 있어, "상위 25%" 가
  실제로는 **33.8%**(1,321명 중 447명)가 된다. 분위수 0.70 과 0.75 는 아예 같은 cutoff 를
  준다 → 분위수뿐 아니라 **실제 양성 비율을 항상 함께 보고한다** (3차시 Step 6).
  4차시 실측: train(1,056명) 기준 cutoff 도 1.500 으로 같아 **전체로 계산했을 때와 라벨이
  하나도 달라지지 않았다**. 규칙(train-only)은 이 결과와 무관하게 유지한다 — 차이가
  있는지 확인하려면 이미 규칙을 어겨야 하기 때문이다.
- **부등호 선택.** cutoff 동점자가 142명이라 `>=`(33.8%)와 `>`(23.1%) 중 무엇을 쓰는지가
  결과를 크게 바꾼다. 본 프로젝트는 `>=` 를 쓰며(`make_high_stress_label`), 8차시에
  민감도 분석 대상으로 남긴다.
- **층화 기준의 미세 누출.** train/test 분할의 `stratify` 는 6차 점수의 median-split 을
  쓰는데, median 은 전체 분포의 통계다. 분할 균형에만 쓰고 라벨 정의에는 쓰지 않지만
  엄밀히는 미세한 누출이며, 대안(층화 없는 분할)은 클래스 불균형을 악화시킨다 → 기록된 선택이다.
- **`s_accul_str_10` 의 이질성.** 문항-전체 상관이 .04(6차)·.00(5차)로 낮고, 빼면 α 가
  .757 → .845 로 오른다. 역채점해도 개선되지 않아(α .738) 방향 문제가 아니라 다른
  구성개념(미래 전망)을 반영할 가능성이 있다. **선행연구와의 비교 가능성을 위해 10문항을
  유지**했으며(선행 α .74 ↔ 우리 .757 로 재현됨), 9문항 척도 민감도 분석을 권장한다.
- **낮은 신뢰도 척도.** 역채점 교정 후에도 `peer_relationship` 은 α = .626 으로 .70 미만이다
  (친사회적 행동 문항이 섞여 단일 구성개념이 아닐 가능성) → 해당 변수의 해석은 제한적이다.
- **계수 해석의 범위.** 표준화 로지스틱 계수 18개 중 부트스트랩(500회) 95% 구간이 0을
  제외하는 것은 **3개뿐**이다 — `peer_support`(−.257) · `self_esteem`(−.254) ·
  `parenting_monitoring`(−.251), 모두 보호요인 방향. 나머지 15개(우울 포함)는
  **방향을 단정하지 않는다**. 특히 단순상관 대비 부호가 뒤집힌 7개는 7개 전부 구간이
  0을 포함하므로, 부호 반전을 실질적 발견으로 해석해서는 안 된다.
- **모델 선택의 근거와 그 한계.** train 5-fold CV 기준 Model A 는 RandomForest(.6651) >
  LogisticRegression(.6535) > DecisionTree(.6355) > Dummy(.500) 순이다. 포레스트의 우위는
  5개 폴드 전부·CV seed 7개 중 6개에서 재현되어 폴드 운으로 보기 어렵다. 그럼에도 본 연구는
  **RQ2(변수의 상대적 중요도)라는 목적에 따라 로지스틱을 주 모델로 선택**했다 — 포레스트는
  계수의 방향·불확실성을 제공하지 못하기 때문이며, 이는 정답이 아니라 기록된 선택이다.
  또한 `GridSearchCV.best_score_` 는 후보 중 최댓값이라 **후보가 많은 모델(트리·포레스트 4개)
  이 로지스틱(3개)보다 약간 유리하게 채점**된다. nested CV 를 쓰지 않았으므로, +0.012 라는
  작은 격차는 이 편향으로 뒤집힐 수 있다.
- **예측 ≠ 인과.** 관찰 패널자료이므로 교란변수를 통제하지 못한다.
- **가짜 결과 금지.** 분석 전에는 성능 숫자를 README·노트북에 적지 않는다.
  이 문서의 성능표가 비어 있는 것은 실수가 아니라 규칙이다.

# -*- coding: utf-8 -*-
"""session4.ipynb 빌더 — 데이터 사이언스의 뼈대(feature·parameter·training)와
그 결과를 재는 법(혼동행렬·정밀도/재현율·F score·F-beta·AUC), 그리고 그 숫자를
망가뜨리는 두 가지(과적합·데이터 누출).

4차시는 "성능 숫자를 의심하는 법"을 배우는 차시다.
좋아 보이는 세 숫자 — Dummy 의 accuracy .663, 깊은 트리의 train AUC 1.000,
누출 모델의 CV AUC 1.000 — 이 전부 쓸모없다는 것을 학생이 직접 만들어 보고 설명한다.

★ 오늘 test 세트는 열지 않는다. 만들어 놓고 봉인한다 — 왜 안 여는지가 오늘 배울 것이다.
   모든 평가는 train 안에서 5-fold CV 로만 한다.

실측 근거 (3차시 역채점 교정 반영 frame 1,321행 × 21열 · seed 42 · test_size .20):
  feature 18개(Model A) = 5차 척도 16개(문항 100개) + 배경 2개 · Model B 는 19개
  로지스틱 파라미터 19개 = 계수 18 + 절편 1
  train 1,056 / test 265 · cutoff(train q75) = 1.500
  train 양성 33.7%(356명) · test 양성 34.3%(91명)
  부등호: >= 447명(33.8%) · > 305명(23.1%) · 동점 142명

  [train 5-fold CV]
  Dummy(전부 0)   accuracy .663 · AUC .500 · recall .000 · precision 정의불가 · F1 .000
  Dummy(전부 1)   accuracy .337 · AUC .500 · recall 1.000 · precision .337 · F1 .504 · F2 .718
  로지스틱 A      accuracy .612 · AUC .653 · recall .615 · precision .445 · F1 .516
                  혼동행렬 TP 219 · FN 137 · FP 273 · TN 427 (특이도 .610 · NPV .757)
                  F2 .572 · F0.5 .471 · PR-AUC .475 (기저율 .337)

  [과적합 — DecisionTree 깊이별 train AUC vs CV AUC]
  depth 2 (잎 4) .663/.636 · depth 3 (잎 8) .691/.615 · depth 5 (잎 30) .765/.589
  depth 10 (잎 153) .951/.529 · depth None (잎 299) 1.000/.519
  로지스틱(파라미터 19개) .687/.653

  [누출]
  6차 점수 투입 AUC 1.0000 (정직 .6525)
  cutoff 누출: 전체 q75 = train q75 = 1.500 → 라벨 차이 0명 (이번엔 우연히 같았다)
  전처리 누출: 전체 fit .6526 vs Pipeline .6525 → 차이 .0001
  선택 누출(합성): 잡음 200개에서 미리 고르면 .5877, 폴드 안에서 고르면 .5025
"""
import os

from nb import md, code, save, SETUP, handoff_in, handoff_out

cells = [
md("""# 4차시 — 데이터 사이언스: 무엇으로 배우고, 무엇으로 재는가

### feature · parameter · training · 평가 지표 · **과적합과 데이터 누출**

> **오늘 한 문장:** "3차시까지는 **데이터**를 다뤘다. 오늘부터는 **모델**이다 —
> 모델이 무엇을 보고(**feature**), 무엇을 정하고(**parameter**), 어떻게 배우고(**training**),
> 그 결과를 **어떻게 재는가**(평가 지표). 그리고 그 숫자가 **어떻게 거짓말하는가.**"

오늘 세 개의 '좋아 보이는' 숫자를 만든다. 그리고 **셋 다 쓸모없다는 것**을 직접 설명하게 된다.

| 숫자 | 어떻게 나오나 | 왜 쓸모없나 |
|---|---|---|
| **정확도 .663** | 전원을 "고스트레스 아님"으로 찍는다 | 고스트레스를 **한 명도** 못 찾는다 |
| **train AUC 1.000** | 결정트리의 깊이 제한을 푼다 | **외운 것**이다 (새 사람에겐 .519) |
| **CV AUC 1.000** | 6차 스트레스 점수를 feature 에 넣는다 | **답을 보고 답을 맞혔다** |

오늘의 목표 6가지:

1. **feature** 가 무엇인지 설명하고, 우리 프로젝트의 feature **18개**를 센다.
2. **parameter** 와 **하이퍼파라미터**를 구분하고, 우리 모델의 파라미터 **19개**를 찾는다.
3. **training(학습)** 이 무엇을 하는 일인지 설명하고, **우리 프로젝트의 목표**를 한 문장으로 쓴다.
4. **train/test split** 과 **cutoff 의 순서**를 지키고, **과적합**을 직접 만들어 본다. ← 고비 1
5. **혼동행렬 · 정밀도 · 재현율 · F score · F-beta · AUC** 를 우리 숫자로 읽는다. ← 오늘의 본론
6. **데이터 누출**을 일부러 일으켜 AUC 1.0 을 만들고, 왜 쓰레기인지 설명한다. ← 고비 2

> 🔴 **오늘 test 세트는 열지 않는다.** 만들어 놓고 봉인한다.
> 왜 안 여는지가 오늘 배울 것 중 하나다. 모든 평가는 **train 안에서 5-fold CV** 로만 한다."""),

md("""## 🗺️ 오늘의 위치 — 4차시

| 차시 | 심리학 | IT / ML |
|---|---|---|
| 1 ✅ | 문화적응 스트레스 · 예측 vs 인과 | feature/target · classification |
| 2 ✅ | 심리척도 · 문항 · 역채점 | pandas · 결측치 · ID join |
| 3 ✅ | 평균 · SD · 분포 · 상관 · Cronbach α | 집계 · 시각화 · 클리닝 |
| **4 (오늘)** | **고스트레스 집단의 조작적 정의 · 임상 cut-off 와의 차이 · 놓침(FN)과 낙인(FP)의 비용** | **feature/parameter/training · split · 과적합 · 평가 지표(F score·AUC) · 데이터 누출** |
| 5 | 예측변수와 결과의 관계·방향 | 로지스틱 회귀 · 계수 · 표준화 |
| 6~8 | 선형성 → 해석 → 보고 | 트리/포레스트 → 중요도 → 재현성 |

**오늘의 재료** — 3차시가 교정한 결과물이다.

- `data/processed/modeling_frame.parquet` — 역채점 교정이 반영된 **1,321행 × 21열** 모델링 표
  → **없어도 된다. Step 0 에서 5·6차 원자료로부터 직접 만든다.**
- `configs/variables.yaml` · `configs/modeling.yaml` — 척도 정의와 split·CV·cutoff 설정
  (`random_seed: 42`, `test_size: 0.20`)
- 우리 파이프라인 모듈: `dataset.py` · `preprocessing.py` · `models.py` · `evaluation.py`

> 🔴 오늘의 규칙: **"성능이 좋아 보이면 축하하기 전에 의심한다."**
> 이 프로젝트에서 AUC 1.0 은 성공이 아니라 **경보음**이다."""),

md("""## Step 0 — 재료 확인: 표가 성립하는가"""),
code('!pip install pandas scikit-learn pyarrow matplotlib pyyaml -q\n'
     '# Colab 에서 그림의 한글이 □ 로 깨지면 아래 한 줄을 실행하고 런타임을 재시작한다.\n'
     '# !apt-get install -y fonts-nanum > /dev/null && rm -rf ~/.cache/matplotlib'),
code(SETUP),
code(handoff_in(pull=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], require=['configs/variables.yaml', 'data/processed/modeling_frame.parquet'], hint="지난 차시 노트북 맨 끝의 '드라이브에 저장' 셀을 실행하면 여기서 자동으로 복원된다")),

md("""### 0-1. 오늘의 표를 **직접 만든다** — `modeling_frame.parquet`

3차시를 들었다면 이 파일은 이미 `data/processed/` 에 있다. **없어도 된다** —
아래 셀이 **5·6차 원자료 CSV 에서 같은 표를 다시 만든다.** 3차시 Step 7 에서 돌린 것과
**완전히 같은 명령**(`scripts/build_dataset.py`)이다.

왜 "다시 만들어도 같은 표"가 나오나 — 표를 만드는 규칙이 **전부 파일에 적혀 있기 때문**이다:

| 무엇을 | 어디에 적혀 있나 |
|---|---|
| 어떤 문항이 어떤 척도인가 · **역채점 문항** | `configs/variables.yaml` (2·3차시에 **사람이** 검증) |
| 문항 → 척도 점수 (평균) · 응답범위 검사 | `src/maps_risk/scoring.py` |
| 5차 ↔ 6차 ID 병합 · 미참여 행 제외 | `src/maps_risk/dataset.py` |

이게 8차시 '재현성'의 실물이다: **표를 주고받는 게 아니라 표를 만드는 절차를 주고받는다.**

한 가지 더 — 3차시가 데이터로 잡아낸 **역채점 문항**(방임 척도의 "관심을 갖고 물어보신다"
같은 것들)이 `variables.yaml` 에 아직 안 적혀 있으면 **아래 셀이 적어 넣고 시작한다.**
오늘의 숫자는 그 교정을 전제한다 — 빠뜨리면 관계가 희석돼 AUC 가 조금씩 달라진다.

> 🔴 여기서 만드는 표에 **고스트레스 라벨(0/1)은 아직 없다.** target 은 6차 스트레스
> **원점수**로만 들어 있다. 라벨을 만드는 cutoff 는 **train 을 떼어낸 뒤에** 계산한다 —
> 그 순서를 어기는 것이 오늘 배울 누출의 한 종류다 (Step 4)."""),

code(r'''# 표가 없으면 원자료(5·6차 CSV)에서 직접 만든다 — 3차시 Step 7 과 같은 명령
import glob, re, subprocess, sys, unicodedata

FRAME   = "data/processed/modeling_frame.parquet"
YAML    = "configs/variables.yaml"
REBUILD = False        # ← True 로 바꾸고 재실행하면 이미 있어도 새로 만든다

# 3차시가 문항-전체 상관(r_it < 0)으로 잡아낸 역채점 문항.
# 오늘의 모든 숫자는 이 교정을 전제한다 — 빠뜨리면 관계가 희석돼 AUC 가 달라진다.
S3_REVERSE = {
    "parenting_neglect": ["parenting_b06_w5", "parenting_b07_w5"],  # 방임인데 "관심을 갖고 물어보신다"
    "school_adjustment": ["learning_a05_w5"],                       # "공부시간에 딴 짓을 한다"
    "peer_relationship": ["fr_rela_a04_w5"],                        # "친구가 하는 일을 방해한다"
}


def find_wave(n):
    """data/raw 어디에 있든 '청소년 n차년도' CSV 를 찾는다 (폴더명이 달라도 된다).

    한글 파일명은 NFC/NFD 두 방식으로 저장될 수 있다(맥은 NFD). 눈에는 똑같이
    '청소년'으로 보여도 문자열 비교는 실패한다 - 그래서 NFC 로 맞춘 뒤 비교한다.
    """
    hits = sorted(p for p in glob.glob("data/raw/**/*.csv", recursive=True)
                  if f"청소년 {n}차년도" in unicodedata.normalize(
                      "NFC", os.path.basename(p)))
    return hits[0] if hits else None


def apply_s3_reverse(path=YAML):
    """3차시 결론을 variables.yaml 에 적어 넣는다 - **비어 있는 곳만** 채운다.

    yaml 로 읽어 통째로 다시 쓰지 않고 해당 줄만 갈아 끼운다. 이 파일은 절반이
    주석이고 그 주석이 '왜 이렇게 정했는가'의 기록이다 - 다시 쓰면 전부 사라진다.
    """
    lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
    out, cur, done = [], None, {}
    for ln in lines:
        m = re.match(r"^  (\w+):\s*$", ln)          # 척도 이름 줄 (들여쓰기 2칸)
        if m:
            cur = m.group(1)
        if cur in S3_REVERSE and re.match(r"^\s*reverse_items: \[\]", ln):
            pad = " " * (len(ln) - len(ln.lstrip()))
            out.append(f"{pad}reverse_items:\n")
            out += [f"{pad}  - {c}\n" for c in S3_REVERSE[cur]]
            done[cur] = S3_REVERSE[cur]             # 비어 있었다 → 채웠다
            continue
        out.append(ln)
    if done:
        open(path, "w", encoding="utf-8").write("".join(out))
    return done


if os.path.exists(FRAME) and not REBUILD:
    print(f"✅ 이미 있다 — {FRAME}")
    print("   원자료에서 다시 만들어 보려면 위 REBUILD = True 로 바꾸고 이 셀을 재실행한다.")
else:
    w5, w6 = find_wave(5), find_wave(6)
    if not (w5 and w6):
        print("🛑 5·6차 원자료 CSV 를 찾지 못했다.")
        print("   data/raw/csv/청소년…/ 아래에 '…청소년 5차년도.csv' 와 6차 파일이 있어야 한다.")
        print("   (MAPS 원자료는 배포·양도 금지라 저장소에 없다 — DATA_ACQUISITION.md 참고)")
    else:
        filled = apply_s3_reverse()
        if filled:
            print("✏️  configs/variables.yaml 에 3차시의 역채점 결론을 적어 넣었다:")
            for k, v in filled.items():
                print(f"     {k:20s} ← {', '.join(v)}")
        else:
            print("✅ variables.yaml 의 reverse_items 는 이미 채워져 있다 (그대로 쓴다).")
        print(f"\n🛠  5차 원자료: {w5}")
        print(f"🛠  6차 원자료: {w6}")
        print("→ scripts/build_dataset.py 실행 … (몇 초 걸린다)\n")
        r = subprocess.run([sys.executable, "scripts/build_dataset.py",
                            "--wave5", w5, "--wave6", w6, "--out", FRAME],
                           capture_output=True, text=True)
        print(r.stdout.strip()[-2500:])
        if r.returncode != 0 or not os.path.exists(FRAME):
            print("\n🛑 만들지 못했다 — 아래 메시지를 읽는다.")
            print(r.stderr.strip()[-1500:])
        else:
            print(f"\n✅ 만들었다 — {FRAME} ({os.path.getsize(FRAME) / 1e3:.0f} KB)")
            print("   같이 갱신된 것: reports/data_quality.md (품질 보고서)")'''),

code(r'''# 3차시가 만든 표를 읽고, 오늘 분석이 전제하는 3가지를 먼저 확인한다
import pandas as pd
from maps_risk.config import load_configs

_, cfg = load_configs("configs")
FRAME = "data/processed/modeling_frame.parquet"

if not os.path.exists(FRAME):
    print("🛑 modeling_frame.parquet 이 없다 — 바로 위 셀(0-1)을 먼저 실행한다.")
else:
    frame = pd.read_parquet(FRAME)
    scores = frame["acculturative_stress_w6"]

    # ① 응답자 1명 = 1행인가  ② target 에 결측이 없나  ③ X 에 6차 변수가 없나
    dup   = frame["id"].duplicated().sum()
    na_y  = scores.isna().sum()
    w6cols = [c for c in frame.columns
              if c.endswith("_w6") and c != "acculturative_stress_w6"]

    print(f"행 {len(frame)} × 열 {frame.shape[1]}")
    print(f"  ① id 중복 {dup}개            {'✅' if dup == 0 else '🛑 병합이 깨졌다'}")
    print(f"  ② target 결측 {na_y}개        {'✅' if na_y == 0 else '🛑 라벨이 조용히 0 이 된다'}")
    print(f"  ③ 6차 컬럼 {w6cols or '없음'}   {'✅' if not w6cols else '🛑 시간 누출'}")
    print(f"\n설정: random_seed={cfg['random_seed']} · test_size={cfg['test_size']} · "
          f"cutoff 분위수={cfg['target']['high_stress_quantile']}")'''),

# ══════════════════════════════════════════════════════════════════
# Step 1 — 데이터 사이언스의 뼈대: feature
# ══════════════════════════════════════════════════════════════════
md("""## Step 1 — 데이터 사이언스의 뼈대 ①: **feature**

오늘부터 모델을 다룬다. 그 전에 단어 세 개를 정확히 해 둔다 — **feature · parameter · training**.
이 셋을 구분하지 못하면 5~8차시의 모든 문장이 흐려진다.

먼저 데이터 사이언스가 무슨 일을 하는지 한 줄로 그리면 이렇다:

```
질문  →  데이터  →  feature(재료)  →  모델 + parameter(손잡이)  →  training(학습)
                                                                      ↓
                                                     평가(지표)  →  해석  →  보고
```

심리학 연구의 순서와 사실 같다: **구성개념 → 측정 → 분석 → 해석**.
다만 데이터 사이언스는 가운데 "분석"을 **모델이 스스로 값을 정하는 과정**으로 바꿔 놓았다.

### feature 란 무엇인가

> **feature(특성, 예측변수) = 모델에게 보여 주기로 결정한 입력값 한 칸.**
> 표의 한 **열**이고, 응답자 한 명당 숫자 하나다.

세 가지를 기억하면 된다.

| | 원칙 | 우리 프로젝트에서 |
|---|---|---|
| ① | **모델은 feature 밖의 세상을 모른다** | 담임 선생님의 인상, 가정 분위기 — 열에 없으면 존재하지 않는 정보다 |
| ② | **feature 는 주어지는 게 아니라 만드는 것** | 문항 100개를 척도 점수 16개로 만든 것이 2·3차시의 일이다 (feature engineering) |
| ③ | **예측 시점에 알 수 있어야 한다** | 6차(중3) 정보는 feature 가 될 수 없다 — 오늘 Step 7 의 그 사고다 |

target(y)은 feature 가 아니다. **맞혀야 할 답**이다 — 우리의 y 는 6차 고스트레스 여부다."""),

code(r'''# TODO: 우리 모델이 볼 수 있는 것(feature)이 정확히 몇 개인지 세어 보라
from maps_risk.dataset import split_features

feats   = split_features(frame, "____")     # ← Model A: 5차 심리사회 변인만 ("A" 인가 "B" 인가)
feats_B = split_features(frame, "B")        # Model B: A + 5차 문화적응 스트레스

print(f"Model A feature {len(feats)}개")
for i, c in enumerate(feats, 1):
    print(f"  {i:2d}. {c}")
print(f"\nModel B feature {len(feats_B)}개 — 차이: {sorted(set(feats_B) - set(feats))}")
print(f"target(y) : high_stress ← acculturative_stress_w6 로 만든다 (feature 아님)")'''),
code(r'''# CHECK Step1-feature
try:
    assert len(feats) == 18, f"Model A 는 feature 18개여야 한다 (지금 {len(feats)}개)"
    assert "acculturative_stress_w6" not in feats, "target 원본이 feature 에 있으면 안 된다"
    assert "previous_acculturative_stress" not in feats, "그건 Model B 전용이다"
    assert len(feats_B) == 19, "Model B 는 A + 1개다"
    print("✅ PASS — Model A feature 18개 · Model B 19개.")
    print("   두 모델의 차이는 딱 하나다: 5차 문화적응 스트레스를 넣느냐 마느냐 (RQ3).")
except Exception as e:
    print("❌ FAIL —", e, '\n힌트: split_features(frame, "A")')'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
feats = split_features(frame, "A")
```

**Model A 의 feature 18개** = 5차 심리사회 척도 **16개** + 배경 **2개**(성별·가정형편 지각).
그리고 그 16개 뒤에는 **문항 100개**가 있다 — 자아탄력성 14문항이 점수 1개로, 우울 10문항이
점수 1개로 접혔다. 그 접는 규칙(평균·역채점·최소 응답 문항 수)이 `variables.yaml` 이다.

> 🔴 **feature 는 발견되는 것이 아니라 만들어지는 것이다.**
> 2차시의 컬럼 검증, 3차시의 역채점 교정 — 그 전부가 "이 18개를 믿을 수 있게 만드는 일"이었다.
> 모델은 그 18개 열 **밖의 세상을 전혀 모른다.**
</details>"""),

code(r'''# feature 하나가 문항 몇 개에서 나왔는지 — '만들어진 재료'라는 것을 눈으로 본다
from maps_risk.config import load_configs as _lc
variables, _ = _lc("configs")

rows = []
for name, spec in (variables.get("predictors") or {}).items():
    rows.append({"feature": name,
                 "문항 수": len(spec.get("items") or []),
                 "응답 범위": str(spec.get("expected_range")),
                 "역채점": len(spec.get("reverse_items") or [])})
for name, spec in (variables.get("background") or {}).items():
    rows.append({"feature": name, "문항 수": 1,
                 "응답 범위": spec.get("type", ""), "역채점": 0})

tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
print(f"\n문항 {int(tbl['문항 수'].sum())}개  →  feature {len(tbl)}개")
print("응답 범위가 [1,4] 와 [1,5] 로 섞여 있다 → 계수를 그냥 비교하면 안 된다 (5차시 표준화)")'''),

# ══════════════════════════════════════════════════════════════════
# Step 1-b — parameter
# ══════════════════════════════════════════════════════════════════
md("""## Step 1 — 데이터 사이언스의 뼈대 ②: **parameter**

feature 가 **재료**라면, parameter 는 모델의 **손잡이**다.

> **parameter(모수, 파라미터) = 모델 안에서 학습으로 값이 정해지는 숫자.**

로지스틱 회귀는 이렇게 생겼다 — 우리 18개 feature 에 각각 **무게(계수)** 를 곱해 더한다:

```
점수 = b0 + b1×자아존중감 + b2×자아탄력성 + b3×우울 + … + b18×가정형편
확률 = 1 / (1 + exp(-점수))          ← 0~1 사이로 눌러 준다
```

여기서 `b1 … b18` 이 **계수(coefficient)**, `b0` 가 **절편(intercept)** 이다.
**전부 합쳐 19개** — 이것이 우리 모델의 파라미터다. 학습 전에는 **아직 값이 없다.**

### 🔴 파라미터와 하이퍼파라미터는 다르다

| | **parameter** | **hyperparameter** |
|---|---|---|
| 누가 정하나 | **데이터**가 정한다 (학습으로) | **사람**이 정한다 (학습 전에) |
| 우리 예 | 계수 18개 + 절편 1개 | `C`(규제 강도) · `max_depth` · `class_weight` |
| | | `test_size=0.20` · `random_seed=42` · `cutoff 분위수=0.75` · `CV 5겹` |
| 어디 적히나 | 학습된 모델 안 (`clf.coef_`) | `configs/modeling.yaml` — **사람이 읽는 파일** |

> 🎯 왜 이 구분이 중요한가: **하이퍼파라미터는 연구자의 선택이고, 선택은 기록해야 한다.**
> "상위 25%를 고스트레스로 본다", "test 를 20% 뗀다" — 데이터가 시킨 게 아니라 **우리가 정한 것**이다.
> 그래서 이 프로젝트는 그것들을 코드에 흩어 두지 않고 **설정 파일 한 곳**에 모아 둔다."""),

code(r'''# TODO: 학습 전과 학습 후 — 파라미터는 언제 생기는가
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from maps_risk.preprocessing import make_preprocessor

def build(clf, scale=True):
    """전처리를 Pipeline 안에 가둔다 — 이유는 Step 8 에서."""
    return Pipeline([("prep", make_preprocessor(scale=scale)), ("clf", clf)])

lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg["random_seed"])
print("학습 전:", "coef_ 있음" if hasattr(lr, "coef_") else "coef_ 없음 — 아직 파라미터가 없다")

X_all, y_tmp = frame[feats], (scores >= scores.quantile(0.75)).astype(int)   # 임시 라벨 (정식 라벨은 Step 4)
fitted = build(lr).fit(X_all, y_tmp)

n_coef = fitted.named_steps["clf"].coef_.size
n_param = n_coef + ____            # ← 절편(intercept)은 몇 개인가? 숫자를 채워라

print(f"학습 후: 계수 {n_coef}개 + 절편 1개 = 파라미터 {n_param}개")
print("\n하이퍼파라미터(사람이 정한 값) — configs/modeling.yaml 에서 읽어 온다:")
print(f"  test_size={cfg['test_size']} · random_seed={cfg['random_seed']} · "
      f"CV {cfg['cv']['folds']}겹 · cutoff 분위수={cfg['target']['high_stress_quantile']} · "
      f"결측 허용률={cfg['missing']['max_feature_missing_rate']}")'''),
code(r'''# CHECK Step1-parameter
try:
    assert n_param == 19, f"계수 18 + 절편 1 = 19 여야 한다 (지금 {n_param})"
    assert n_coef == len(feats), "계수는 feature 하나당 하나씩이다"
    print("✅ PASS — feature 18개 → 파라미터 19개 (계수 18 + 절편 1).")
    print("   feature 를 하나 늘리면 파라미터도 하나 늘어난다. Model B 는 20개다.")
    print("   그리고 하이퍼파라미터는 데이터가 아니라 우리가 정했다 — 그래서 설정 파일에 적혀 있다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 절편은 모델 전체에 하나뿐이다.")'''),

# ══════════════════════════════════════════════════════════════════
# Step 2 — training 과 우리의 목표
# ══════════════════════════════════════════════════════════════════
md("""## Step 2 — 데이터 사이언스의 뼈대 ③: **training(학습)**

> **training = 손잡이 19개를, "틀린 정도"가 가장 작아지는 자리로 돌리는 과정.**

과정을 풀어 쓰면 이렇다.

1. 파라미터를 아무 값에서 시작한다 (전부 0 에서 시작해도 된다).
2. train 1,056명에 대해 확률을 계산한다 → 정답과 얼마나 어긋나는지 잰다 (**손실, loss**).
3. 손실이 줄어드는 방향으로 파라미터를 조금 움직인다.
4. 더 줄지 않을 때까지 2~3을 반복한다. (`max_iter=2000` 은 "최대 2000번까지"라는 뜻이다.)

우리 코드에서는 이 전부가 **한 줄**이다:

```python
model.fit(X_train, y_train)      # ← 여기서 19개 숫자가 정해진다
```

### 🔴 학습에서 반드시 붙잡아야 할 세 가지

| | | |
|---|---|---|
| ① | **모델은 train 데이터의 정답을 본다** | 그래서 train 성적은 항상 후하다 — 그걸로 성능을 재면 안 된다 (Step 4) |
| ② | **손실을 줄이는 것이 목표지, 진실을 찾는 것이 목표가 아니다** | 답을 베낄 방법이 있으면 **반드시 베낀다** (Step 7 데이터 누출) |
| ③ | **외우는 것도 손실을 줄인다** | 1,056명을 통째로 외워도 손실은 0 이 된다 (Step 5 과적합) |

②③이 오늘 오후의 두 사고다. **모델은 부정직한 게 아니라, 시킨 일을 너무 잘하는 것뿐이다.**"""),

md("""## Step 2 — 그래서, 우리 프로젝트의 목표는 무엇인가

한 문장으로 못 박아 둔다. 8차시 보고서까지 이 문장이 기준이 된다.

> **MAPS 1기 패널에서, 중학교 2학년(5차, 2015) 시점의 심리사회 지표 18개로,
> 1년 뒤 중학교 3학년(6차, 2016) 시점에 우리가 조작적으로 정의한 고스트레스 집단에
> 속하는지를 분류한다.**

| | |
|---|---|
| 입력 (X) | 5차 심리사회 척도 16개 + 배경 2개 = **feature 18개** |
| 출력 (y) | 6차 고스트레스 여부 **0/1** (조작적 정의 — 임상 진단 아님) |
| 학습 대상 | train **1,056명** (test 265명은 봉인) |
| 채점 방법 | train 안 **5-fold CV** · 여러 지표를 **동시에** (Step 6) |
| 비교 기준 | 항상 **DummyClassifier** 를 옆에 둔다 |
| 연구질문 | RQ1 예측 가능한가 · RQ2 어떤 변인이 관련되나 · RQ3 5차 스트레스를 넣으면 나아지나 (A vs B) |

### 🔴 목표가 **아닌** 것

- ❌ **성능 경쟁이 아니다.** AUC 를 최대로 만드는 것이 이 수업의 평가 기준이 아니다.
  "현재 변수만으로는 충분히 구분되지 않았다"도 **옳은 결론**이다.
- ❌ **개입 대상 선정이 아니다.** 우리 출력은 어떤 학생을 지원할지 정하는 도구가 아니다.
- ❌ **원인 규명이 아니다.** 예측에 기여하는 변수와 원인은 다르다 (1차시 예측 vs 인과).

> 🎯 그래서 오늘 배울 **평가 지표**가 중요하다. "얼마나 잘했나"를 재는 자가 하나뿐이면
> 우리는 그 자에 맞춰 스스로를 속이게 된다."""),

code(r'''# 학습을 한 줄로 — 그리고 학습된 파라미터가 실제로 어떤 모양인지 본다
coefs = pd.Series(fitted.named_steps["clf"].coef_[0], index=feats).sort_values(key=abs, ascending=False)
print("학습으로 정해진 계수 (절댓값 큰 순 · 임시 라벨 기준)")
print(coefs.round(3).to_string())
print(f"\n절편 = {fitted.named_steps['clf'].intercept_[0]:.3f}")
print("\n※ 부호는 '방향', 크기는 '기여도'처럼 읽고 싶어지지만 — 아직 그러면 안 된다.")
print("  ① 척도 범위가 4점/5점으로 섞여 있고 ② 이건 임시 라벨이다. 제대로 된 해석은 5차시.")'''),

# ══════════════════════════════════════════════════════════════════
# Step 3 — 조작적 정의
# ══════════════════════════════════════════════════════════════════
md("""## Step 3 — 조작적 정의: 선을 긋는다는 것

목표 문장에 "**조작적으로 정의한** 고스트레스 집단"이라는 말이 들어 있었다. 그 선을 이제 긋는다.

문화적응 스트레스 점수는 1.00 ~ 4.00 사이의 **연속적인 숫자**다. 그런데 우리가 하려는 일은
**분류(classification)** — "고스트레스 집단인가 아닌가"라는 **예/아니오** 문제다.
연속된 숫자를 둘로 나누려면 **선(cutoff)** 을 그어야 한다.

우리가 긋는 선은 이것이다:

> **학습 데이터 점수의 상위 25%(75 백분위수) 이상 → `high_stress = 1`**

이것을 **조작적 정의(operational definition)** 라고 한다 — "연구자가 분석을 위해 정한 기준"이다.
그리고 눈치챘겠지만, **이 0.75 는 하이퍼파라미터다** — 데이터가 정해 준 게 아니라 우리가 정했다.

### 🔴 이것이 아닌 것

| 이것이 아니다 | 왜 |
|---|---|
| 임상 진단 | 정신과 의사가 면담·검사로 내리는 판단이 아니다 |
| 임상 cut-off | 검증된 임상 절단점(예: CES-D 16점)이 아니다 — **우리가 그 자리에서 만든 선**이다 |
| 실제 고위험군 | 상위 25%는 **이 표본 안에서의 상대적 위치**일 뿐이다 |

> 표현 규칙: ❌ "고위험 청소년을 판별하였다" → ✅ "본 연구에서 **조작적으로 정의한**
> 고스트레스 집단을 분류하였다". 8차시 보고서까지 이 표현을 끝까지 지킨다.

그리고 3차시에서 이미 봤듯이 — **선을 긋는 순간 이상한 일이 벌어진다.**"""),

code(r'''# TODO: 부등호 하나가 몇 명을 옮기는지 직접 확인하라
cut = scores.quantile(0.75)

n_ge = (scores >= cut).sum()      # cutoff 이상 (우리 파이프라인의 규칙)
n_gt = (scores ____ cut).sum()    # ← cutoff '초과' 로 바꾸면? 부등호를 채워라
n_tie = (scores == cut).sum()

print(f"cutoff = {cut:.3f}")
print(f"  >= (이상) : {n_ge}명 = {n_ge/len(scores):.1%}")
print(f"  >  (초과) : {n_gt}명 = {n_gt/len(scores):.1%}")
print(f"  동점자    : {n_tie}명")'''),
code(r'''# CHECK Step3
try:
    assert n_ge - n_gt == n_tie, "이상 − 초과 = 동점자 수여야 한다"
    assert n_tie > 100, f"동점자가 {n_tie}명 — 3차시에 본 그 현상이다"
    print(f"✅ PASS — 부등호 하나가 {n_tie}명을 옮긴다. 전체의 {n_tie/len(scores):.1%} 다.")
    print("   '상위 25%' 라는 한 문장 뒤에 이런 선택이 숨어 있다.")
    print("   우리는 >= 를 쓴다(make_high_stress_label). 그리고 실제 양성 비율을 함께 보고한다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: '초과'는 같은 값을 포함하지 않는다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
n_gt = (scores > cut).sum()
```

실측: `>=` 447명(33.8%) · `>` 305명(23.1%) · 동점자 **142명**.

**부등호 하나가 142명의 운명을 바꾼다.** 3차시에서 "분위수를 0.75 → 0.80 으로 올려야
비율이 바뀐다"고 본 그 142명이 바로 이들이다.

어느 쪽이 옳은가? **정답이 없다.** 우리는 `>=` 를 쓰기로 했고(파이프라인의
`make_high_stress_label`), 그 선택과 **실제 양성 비율을 함께 보고**한다.
숨기지 않는 것이 정답이다.
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 4 — split, cutoff 순서
# ══════════════════════════════════════════════════════════════════
md("""## Step 4 — train/test split: 시험 문제를 미리 보지 않는다

Step 2 에서 "모델은 train 데이터의 **정답을 보면서** 배운다"고 했다. 그래서 문제가 하나 생긴다 —
**학습에 쓴 데이터로 성능을 재면 안 된다.** 왜?

시험 공부를 하면서 **기출문제 100개를 외웠다**고 하자. 그 100개로 시험을 보면 100점이다.
그런데 그 점수는 **"이 학생이 새 문제를 풀 수 있는가"** 에 대해 아무것도 말해 주지 않는다.

그래서 데이터를 둘로 나눈다:

| | 무엇 | 언제 쓰나 |
|---|---|---|
| **train (학습)** | 80% | 모델을 학습시키고, 튜닝하고, cutoff 를 정한다 |
| **test (시험)** | 20% | **마지막에 딱 한 번.** 그전에는 쳐다보지도 않는다 |

> 🔴 **오늘 우리는 test 를 열지 않는다.** 만들어 놓고 봉인만 한다.
> 오늘의 모든 평가는 **train 안에서 5겹 교차검증(5-fold cross-validation)** 으로 한다 —
> train 을 다시 5조각으로 나눠 4조각으로 배우고 1조각으로 채점하기를 5번 돌려 평균 낸다.
> (`n_splits=5` 도 하이퍼파라미터다. 설정 파일에 적혀 있다.)"""),

code(r'''# TODO: 설정 파일대로 split 하라 (숫자를 코드에 직접 쓰지 않는다 — 설정에서 읽는다)
from sklearn.model_selection import train_test_split

idx_tr, idx_te = train_test_split(
    frame.index,
    test_size=cfg["_______"],          # ← 설정 키를 채워라
    random_state=cfg["_______"],       # ← 설정 키를 채워라
    # 층화(stratify): 두 조각의 구성이 비슷하도록 맞춘다.
    # 진짜 라벨은 아직 없다(cutoff 를 train 에서 정해야 하니까 — 순환!)
    # → 6차 점수의 median 기준 임시 구분으로 층화한다.
    stratify=(scores >= scores.median()).astype(int))

print(f"train {len(idx_tr)}명 · test {len(idx_te)}명")
print(f"겹치는 사람: {len(set(idx_tr) & set(idx_te))}명")'''),
code(r'''# CHECK Step4-split
try:
    assert len(set(idx_tr) & set(idx_te)) == 0, "train 과 test 에 같은 사람이 있으면 안 된다"
    assert abs(len(idx_te) / len(frame) - 0.20) < 0.01, f"test 가 20% 여야 한다 (지금 {len(idx_te)/len(frame):.1%})"
    assert len(idx_tr) == 1056 and len(idx_te) == 265, f"1056/265 여야 한다 (지금 {len(idx_tr)}/{len(idx_te)})"
    print("✅ PASS — train 1,056 / test 265, 겹침 0명.")
    print("   random_state=42 를 고정했으므로 누가 실행해도 같은 사람이 train 에 들어간다 (재현성).")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: modeling.yaml 의 키 이름은 test_size 와 random_seed 다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 그리고 stratify 의 순환 문제</summary>

```python
test_size=cfg["test_size"], random_state=cfg["random_seed"]
```

**`random_state=42` 를 고정하는 이유**: 안 고정하면 실행할 때마다 다른 사람이 train 에
들어가고, 성능 숫자도 매번 달라진다. 그러면 **재현이 불가능**하다. 42 라는 숫자 자체에
의미는 없다 — "고정했다"는 사실이 중요하다.

**stratify 의 순환 문제 (AGENTS.md 에 기록된 설계 결정)**: 원래는 진짜 라벨(`high_stress`)로
층화하고 싶다. 그런데 진짜 라벨은 **train 의 cutoff 를 정해야** 생기고, cutoff 는
**split 을 해야** 정할 수 있다. **닭이 먼저냐 달걀이 먼저냐**다.
→ 그래서 6차 점수의 **median 기준 임시 구분**으로 층화한다. median 은 전체 분포의
통계라는 한계가 있지만, **분할 균형에만 쓰고 라벨 정의에는 쓰지 않는다.**
이런 타협은 숨기지 않고 **코드 주석과 문서에 적는다.**
</details>"""),

md("""## Step 4 — 순서가 전부다: cutoff 는 train 에서만 ⚠️ (첫 봉우리)

이제 선을 긋는다. 그런데 **순서**가 결정적이다.

```
❌ 틀린 순서:  전체 1,321명의 점수를 보고 선을 긋는다  →  나눈다
✅ 옳은 순서:  나눈다  →  train 1,056명의 점수만 보고 선을 긋는다  →  그 선을 전체에 적용
```

왜 틀렸나? 전체를 보고 선을 그으면, **test 응답자들의 점수가 선의 위치를 결정하는 데
참여**한다. test 는 "한 번도 안 본 새 사람들"이어야 하는데, 이미 한 번 본 셈이 된다.
이것이 **데이터 누출(data leakage)** 이다.

> 비유: 시험 문제의 **합격선**을, 채점할 학생들의 답안을 미리 보고 정하는 것과 같다.
> 문제를 안 보여줬어도 **합격선이 그들의 점수에 맞춰져 있다.**"""),

code(r'''# TODO: cutoff 를 어느 집단에서 계산해야 하는가?
from maps_risk.dataset import make_high_stress_label

q = cfg["target"]["high_stress_quantile"]
y_all, cutoff = make_high_stress_label(
    scores.loc[_______],      # ← cutoff 를 정할 때 볼 사람들 (idx_tr 인가 frame.index 인가)
    scores,                   # 라벨을 붙일 대상 (전체 — 같은 선을 모두에게 적용한다)
    q)
frame["high_stress"] = y_all

print(f"cutoff(train {q:.0%} 분위수) = {cutoff:.4f}")
print(f"train 양성 {y_all.loc[idx_tr].mean():.1%} ({int(y_all.loc[idx_tr].sum())}명)")
print(f"test  양성 {y_all.loc[idx_te].mean():.1%} ({int(y_all.loc[idx_te].sum())}명)  ← 오늘 안 본다")

# 만약 전체로 계산했다면 라벨이 몇 명이나 달라졌을까?
y_leak = (scores >= scores.quantile(q)).astype(int)
print(f"\n전체로 계산한 cutoff = {scores.quantile(q):.4f} → 라벨이 달라진 사람: {int((y_all != y_leak).sum())}명")'''),
code(r'''# CHECK Step4-cutoff
try:
    assert cutoff == scores.loc[idx_tr].quantile(q), "cutoff 는 train 분위수와 같아야 한다"
    assert abs(y_all.loc[idx_tr].mean() - 0.337) < 0.02, f"train 양성률 33.7% 근처여야 한다"
    print(f"✅ PASS — cutoff 를 train {len(idx_tr)}명만 보고 정했다.")
    print(f"   그런데 전체로 계산해도 값이 같아서 라벨 차이가 {int((y_all != y_leak).sum())}명이다.")
    print("   → 다음 셀에서 이게 무슨 뜻인지 생각해 보자. (규칙을 안 지켜도 된다는 뜻일까?)")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: cutoff 를 정할 때는 train 만 본다 → scores.loc[idx_tr]")'''),
md("""<details><summary>💡 힌트 / 정답 — 그리고 오늘 가장 중요한 교훈</summary>

```python
scores.loc[idx_tr]
```

**실측 결과가 재미있다.** 전체로 계산한 cutoff 도 1.500, train 으로 계산한 cutoff 도 1.500 —
**라벨이 달라진 사람은 0명**이다. 3차시에서 본 그 거대한 동점 덩어리(1.50 에 142명) 때문에
어느 쪽을 봐도 선이 같은 자리에 떨어졌다.

**그럼 규칙을 안 지켜도 되나? 아니다.**

> 🔴 **안전벨트를 맸는데 사고가 안 났다고 해서, 안전벨트가 쓸모없는 것이 아니다.**
> 규칙은 **결과**로 정당화되지 않는다. 규칙은 **절차**다.

이번엔 우연히 같았다. 다른 데이터, 다른 seed, 다른 분위수에서는 달라진다.
그리고 결정적으로 — **미리 확인할 방법이 없다.** "차이가 없을 테니 대충 하자"는
차이가 있는지 **확인한 뒤에야** 할 수 있는 말인데, 확인하려면 이미 전체를 봐야 한다.

우리가 한 일: 규칙대로 하고, **차이가 없었다는 사실까지 기록**했다. 그게 전부다.
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 5 — 과적합
# ══════════════════════════════════════════════════════════════════
md("""## Step 5 — 과적합(overfitting): 외우는 것도 손실을 줄인다

Step 2 의 ③번을 기억하는가 — **"외우는 것도 손실을 줄인다."** 이제 그걸 직접 만들어 본다.

기출문제 비유를 한 칸 더 밀어 보자.

| | 학생 | 모델 |
|---|---|---|
| **이해한 학생** | 원리를 익혔다 → 기출 85점, 새 시험 80점 | 파라미터가 적다 → train 성적과 새 데이터 성적이 **비슷하다** |
| **외운 학생** | 답만 외웠다 → 기출 **100점**, 새 시험 50점 | 파라미터가 많다 → train **완벽**, 새 데이터에서 **무너진다** |

> **과적합 = 모델이 데이터의 '규칙'이 아니라 '이 표본의 우연'까지 외워 버린 상태.**

손잡이를 몇 개 주느냐가 갈림길이다. 결정트리는 깊이를 풀어 주면 손잡이를 **원하는 만큼**
만들 수 있다 — 응답자 한 명당 잎 하나까지 갈 수 있다. 그러면 train 은 100% 맞는다.

**어떻게 알아채나?** 두 점수를 **나란히** 본다 — `train 점수` 와 `새 데이터 점수(CV)`.
그 **간극(gap)** 이 과적합의 크기다. 우리가 5-fold CV 를 쓰는 이유가 바로 이것이다."""),

code(r'''# TODO: 트리의 깊이를 풀어 주면 무슨 일이 나는가 — max_depth 를 채워라
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

Xtr, ytr = frame.loc[idx_tr, feats], frame.loc[idx_tr, "high_stress"]
cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["random_seed"])

DEPTHS = [2, 3, 5, 10, ____]     # ← 마지막은 '깊이 제한 없음'. 파이썬으로 뭐라고 쓰나?

print(f"{'max_depth':>10s} {'잎(leaf)':>8s} {'파라미터 성격':>14s} {'train AUC':>10s} {'CV AUC':>8s} {'간극':>7s}")
gap = {}
for d in DEPTHS:
    tree = build(DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                        random_state=cfg["random_seed"]), scale=False)
    tree.fit(Xtr, ytr)
    tr_auc = roc_auc_score(ytr, tree.predict_proba(Xtr)[:, 1])           # 배운 사람으로 채점 ❌
    cv_auc = cross_val_score(tree, Xtr, ytr, cv=cv, scoring="roc_auc").mean()   # 안 배운 사람으로 ✅
    n_leaf = tree.named_steps["clf"].get_n_leaves()
    gap[d] = tr_auc - cv_auc
    print(f"{str(d):>10s} {n_leaf:8d} {'분기 규칙 ' + str(n_leaf - 1):>14s} "
          f"{tr_auc:10.3f} {cv_auc:8.3f} {tr_auc - cv_auc:+7.3f}")

lr_pipe = build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=cfg["random_seed"]))
lr_pipe.fit(Xtr, ytr)
lr_tr = roc_auc_score(ytr, lr_pipe.predict_proba(Xtr)[:, 1])
lr_cv = cross_val_score(lr_pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
print(f"\n{'로지스틱':>10s} {'-':>8s} {'파라미터 19':>14s} "
      f"{lr_tr:10.3f} {lr_cv:8.3f} {lr_tr - lr_cv:+7.3f}")'''),
code(r'''# CHECK Step5
try:
    assert None in DEPTHS, "max_depth=None 이 '제한 없음'이다"
    assert gap[None] > gap[2], "깊이를 풀수록 train 과 CV 의 간극이 커져야 한다"
    assert gap[None] > 0.4, f"제한 없는 트리의 간극이 {gap[None]:.3f} — 실측은 .48 근처다"
    print(f"✅ PASS — 깊이 제한을 풀면 train AUC 1.000, 그런데 CV AUC 는 .52 로 주저앉는다.")
    print(f"   간극: depth=2 는 {gap[2]:+.3f}, depth=None 은 {gap[None]:+.3f}.")
    print("   train 점수만 보면 '제한 없는 트리'가 최고의 모델처럼 보인다. **완전히 반대다.**")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: sklearn 에서 '제한 없음'은 None 이다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 과적합과 데이터 누출은 어떻게 다른가</summary>

```python
DEPTHS = [2, 3, 5, 10, None]
```

**실측**

| max_depth | 잎 | train AUC | CV AUC | 간극 |
|---|---|---|---|---|
| 2 | 4 | .663 | .636 | +.027 |
| 3 | 8 | .691 | .615 | +.076 |
| 5 | 30 | .765 | .589 | +.176 |
| 10 | 153 | .951 | .529 | +.422 |
| **None** | **299** | **1.000** | **.519** | **+.481** |
| 로지스틱(파라미터 19개) | — | .687 | .653 | **+.035** |

잎이 299개 — train 1,056명을 사실상 **한 명 한 명 외운** 것이다. train 은 완벽하고,
처음 보는 사람 앞에서는 **동전 던지기**(.519)가 된다.

> 🔴 오늘 **train AUC 1.000** 과 **CV AUC 1.000** 을 둘 다 만든다. 겉모습은 같지만 병이 다르다.

| | 과적합 (Step 5) | 데이터 누출 (Step 7) |
|---|---|---|
| train 점수 | 1.000 | 1.000 |
| **CV/새 데이터 점수** | **무너진다 (.519)** | **그대로 1.000** ← 더 무섭다 |
| 원인 | 손잡이가 너무 많다 | 답을 feature 에 넣었다 |
| 처방 | 모델을 단순하게 · 규제 · CV 로 감시 | 데이터 흐름을 고친다 (모델 문제가 아니다) |

**누출이 더 위험한 이유**: 과적합은 CV 가 잡아 준다. 누출은 **CV 도 속인다.**
</details>"""),

# ══════════════════════════════════════════════════════════════════
# Step 6 — 평가: 무엇으로 재는가
# ══════════════════════════════════════════════════════════════════
md("""## Step 6 — 평가 ①: 지표 하나로는 반드시 속는다

학습이 끝났다. 이제 **잘했는지 재야 한다.** 그런데 무엇으로?

가장 먼저 떠오르는 건 **정답률(accuracy)** 이다. 100명 중 몇 명을 맞혔나. 직관적이다.
그런데 우리 데이터에는 함정이 하나 있다 — **클래스 불균형**이다.

train 의 고스트레스는 **33.7%**(356명 / 1,056명), 약 **2:1** 이다.
이 상황에서 가장 게으른 모델을 만들어 보자 — **"전원 고스트레스 아님"** 이라고만 답하는 모델.

이런 모델을 **더미 분류기(DummyClassifier)** 라고 하고, **학습을 전혀 하지 않는다.**
파라미터도 없다. 그런데도 정답률이 꽤 나온다. 얼마나 나올까?

> 🖐 먼저 **예측**해 보라. 30%? 50%? 70%?"""),

code(r'''# Dummy 와 로지스틱을 같은 조건에서 비교한다 (train 안에서 5-fold CV, test 는 안 연다)
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.dummy import DummyClassifier

models = {
    "Dummy (전부 0)": build(DummyClassifier(strategy="most_frequent"), scale=False),
    "로지스틱 회귀":    build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=cfg["random_seed"])),
}
print(f"Model A features {len(feats)}개 · train {len(Xtr)}명 (고스트레스 {int(ytr.sum())}명)\n")
print(f"{'모델':16s} {'accuracy':>9s} {'AUC':>7s} {'balanced_acc':>13s} {'recall':>8s}")
res = {}
for nm, est in models.items():
    r = {m: cross_val_score(est, Xtr, ytr, cv=cv, scoring=m).mean()
         for m in ("accuracy", "roc_auc", "balanced_accuracy", "recall")}
    res[nm] = r
    print(f"{nm:16s} {r['accuracy']:9.3f} {r['roc_auc']:7.3f} {r['balanced_accuracy']:13.3f} {r['recall']:8.3f}")'''),

code(r'''# TODO: 위 표를 보고 판단하라 — 어느 모델이 더 좋은 모델인가?
더_좋은_모델 = "_______"        # ← "Dummy (전부 0)" 또는 "로지스틱 회귀"
근거_지표    = "_______"        # ← 판단의 근거가 된 지표 이름 (accuracy / roc_auc / recall 중)

print(f"내 판단: {더_좋은_모델}  (근거: {근거_지표})")
print(f"accuracy 만 보면 Dummy {res['Dummy (전부 0)']['accuracy']:.3f} vs 로지스틱 {res['로지스틱 회귀']['accuracy']:.3f}")'''),
code(r'''# CHECK Step6-baseline
try:
    assert 더_좋은_모델 == "로지스틱 회귀", "accuracy 가 낮아도 더 좋은 모델일 수 있다"
    assert 근거_지표 in ("roc_auc", "recall", "balanced_accuracy"), \
        "accuracy 는 불균형 데이터에서 판단 근거가 될 수 없다"
    assert res["Dummy (전부 0)"]["accuracy"] > res["로지스틱 회귀"]["accuracy"], "실측이 뒤집혔다"
    print("✅ PASS — **정확도가 더 낮은 모델이 더 좋은 모델이다.**")
    print(f"   Dummy 는 accuracy {res['Dummy (전부 0)']['accuracy']:.1%} 를 받지만 recall 이 0.000 —")
    print("   고스트레스 학생을 **한 명도** 찾아내지 못한다. 그런 모델은 쓸 데가 없다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 고스트레스 집단을 '찾아내는' 것이 목적이다. 그걸 재는 지표는?")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

`더_좋은_모델 = "로지스틱 회귀"`, `근거_지표 = "recall"` (또는 `roc_auc` / `balanced_accuracy`).

**실측**: Dummy accuracy **.663** vs 로지스틱 **.612**. 정확도만 보면 Dummy 가 이긴다.
그런데 Dummy 의 recall 은 **.000** — 고스트레스 학생을 한 명도 못 찾는다.

**왜 이런 일이**: 66.3% 가 음성(고스트레스 아님)이므로, 전부 음성이라고 찍으면
자동으로 66.3% 를 맞힌다. **accuracy 는 다수 클래스의 비율을 그냥 되돌려 주는 지표**가 된다.

그래서 우리 규칙(AGENTS.md): **accuracy 단독 보고 금지 · Dummy 를 항상 같이 보고.**
Dummy 없이 "정확도 66%" 만 보면 대단해 보인다. Dummy 를 옆에 두는 순간 아무것도 아니게 된다.

이제 문제는 이것이다 — **그럼 무엇으로 재야 하나?** 다음이 오늘의 본론이다.
</details>"""),

md("""## Step 6 — 평가 ②: 혼동행렬 — 암을 찾는 의사 이야기 🩺

지표 이야기를 하기 전에, **모든 지표가 거기서 나오는 표 하나**를 먼저 그린다.

어떤 의사가 검진 결과를 보고 "암입니다 / 아닙니다"를 말한다. 결과는 네 가지뿐이다.

| | 실제로 **암이 있다** | 실제로 **암이 없다** |
|---|---|---|
| 의사가 **"암입니다"** | ✅ **TP** (True Positive)<br>**맞게 찾았다** | ❌ **FP** (False Positive)<br>**헛경보** — 없는데 있다고 했다 |
| 의사가 **"아닙니다"** | ❌ **FN** (False Negative)<br>**놓쳤다** — 있는데 없다고 했다 | ✅ **TN** (True Negative)<br>**맞게 통과시켰다** |

이름 읽는 법이 헷갈리는데, 규칙은 단순하다:

> **뒷글자 = 모델이 뭐라고 했나 (Positive/Negative) · 앞글자 = 그 말이 맞았나 (True/False)**
> `FN` = "Negative 라고 했는데(N) 틀렸다(F)" = **있는데 놓쳤다.**

### 🔴 두 가지 오류는 무게가 다르다

- **FN(놓침)**: 암 환자를 집으로 돌려보낸다. 다음 검진까지 1년이 간다. — 되돌리기 어렵다.
- **FP(헛경보)**: 조직검사를 한 번 더 한다. 며칠 불안하고 비용이 든다. — 대체로 회복 가능하다.

그래서 검진(**선별, screening**)에서는 보통 **FN 을 더 무겁게** 본다.
"의심되면 일단 더 보자"가 검진의 논리다. 반대로 항암 치료 **시작** 결정처럼 개입 자체가
위험한 상황에서는 **FP** 가 더 무겁다. — **무엇이 더 아픈 오류인지는 데이터가 아니라
맥락이 정한다.** 이 문장이 오늘 Step 6 전체를 관통한다."""),

code(r'''# 우리 프로젝트의 네 칸 — 비율이 아니라 '사람 수'로 본다
from maps_risk import evaluation
from sklearn.metrics import confusion_matrix

pred_lr = cross_val_predict(models["로지스틱 회귀"], Xtr, ytr, cv=cv)              # 라벨 예측
prob_lr = cross_val_predict(models["로지스틱 회귀"], Xtr, ytr, cv=cv,
                            method="predict_proba")[:, 1]                          # 양성 확률

TN, FP, FN, TP = confusion_matrix(ytr, pred_lr, labels=[0, 1]).ravel()
print("로지스틱 회귀 (train 5-fold CV 예측)")
print(evaluation.confusion_frame(ytr, pred_lr).to_string(), "\n")
print(f"  TP {TP:3d}  실제 고스트레스인데 모델도 그렇게 봤다   → 맞게 찾았다")
print(f"  FN {FN:3d}  실제 고스트레스인데 모델은 아니라고 했다 → 놓쳤다 🔴")
print(f"  FP {FP:3d}  아닌데 모델이 고스트레스라고 했다        → 헛경보")
print(f"  TN {TN:3d}  아닌데 모델도 아니라고 했다             → 맞게 통과")
print(f"\n합 {TP+FN+FP+TN} = train {len(ytr)}명 · 실제 양성 {TP+FN}명 · 모델이 지목한 사람 {TP+FP}명")

print("\nDummy (전부 0)")
print(evaluation.confusion_frame(ytr, np.zeros(len(ytr), dtype=int)).to_string())
print(f"→ Dummy 는 고스트레스 {int(ytr.sum())}명 **전원을 놓쳤다**(FN {int(ytr.sum())}). '정확도 66%' 의 실체다.")'''),
code(r'''# CHECK Step6-confusion
try:
    assert TP + FN == int(ytr.sum()), "TP + FN = 실제 양성 수"
    assert TN + FP == int((ytr == 0).sum()), "TN + FP = 실제 음성 수"
    assert (TP, FN, FP, TN) == (219, 137, 273, 427), f"실측과 다르다: {(TP, FN, FP, TN)}"
    print("✅ PASS — TP 219 · FN 137 · FP 273 · TN 427.")
    print("   고스트레스 356명 중 219명을 찾아내고 **137명을 놓쳤다.**")
    print("   그리고 아닌 700명 중 273명을 잘못 지목했다. 이 네 숫자에서 오늘의 모든 지표가 나온다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 3차시 역채점 교정이 반영된 frame 인지 확인하라.")'''),

md("""## Step 6 — 평가 ③: 우리 맥락에서 FN 과 FP 는 각각 무엇인가 🔴

의사 비유를 우리 프로젝트로 그대로 옮기면 **틀린다.** 우리는 진단을 하는 것이 아니기 때문이다.
정확히 번역하면 이렇다.

| | 의사 (암 검진) | **우리 프로젝트** |
|---|---|---|
| **TP** 219명 | 암을 맞게 찾았다 | 1년 뒤 고스트레스 집단에 속한 학생을, 중2 정보만으로 맞게 분류했다 |
| **FN** 137명 | 암 환자를 놓쳤다 | 실제로 고스트레스였는데 **모델이 못 알아봤다** — 연구로 보면 "우리 변수로는 안 보이는 사람들" |
| **FP** 273명 | 없는 암을 있다고 했다 | 고스트레스가 아닌데 모델이 지목했다 — **라벨이 사람에게 붙으면 낙인이 된다** |
| **TN** 427명 | 맞게 통과 | 아닌 학생을 아니라고 했다 |

### 그래서 우리는 어느 쪽을 더 무겁게 보는가

- **FN 쪽으로 기운다.** 우리 목적은 **선별(screening)** 이다 — 도움이 필요할 수 있는 학생을
  놓치는 것이 연구 목적상 더 아프다. 그래서 `class_weight="balanced"` 를 쓰고 **recall 을 반드시 읽는다.**
- **그렇다고 FP 를 버릴 수 없다.** 우리 FP 는 "며칠 불안"이 아니라 **한 학생에게 붙는 이름표**다.
  273명은 train 1,056명의 **26%** — 넷 중 하나다.
- 🔴 그래서 이 프로젝트의 결론은 절대 **"이 학생들을 관리 대상으로 지정한다"** 가 될 수 없다.
  우리가 하는 말은 **"이 변수들이 1년 뒤 고스트레스와 관련이 있다"** 까지다.

> **어느 오류가 더 아픈지는 데이터가 답해 주지 않는다. 연구자가 정하고, 그 선택을 적어 둔다.**
> 오늘 배울 F-beta 와 임계값 조정은 **바로 그 선택을 숫자로 옮기는 도구**다."""),

md("""## Step 6 — 평가 ④: 네 칸에서 나오는 네 개의 지표

TP·FN·FP·TN 네 숫자를 어떻게 묶느냐에 따라 서로 다른 질문에 답하게 된다.

| 지표 | 공식 | 답하는 질문 | 우리 실측 |
|---|---|---|---|
| **정확도** accuracy | (TP+TN) / 전체 | 전부 중 몇 개나 맞혔나 | (219+427)/1056 = **.612** |
| **정밀도** precision | TP / (TP+**FP**) | **모델이 지목한 사람 중** 진짜는 몇 %인가 | 219/492 = **.445** |
| **재현율** recall<br>(민감도 sensitivity) | TP / (TP+**FN**) | **실제 양성 중** 몇 %를 찾아냈나 | 219/356 = **.615** |
| **특이도** specificity | TN / (TN+FP) | **실제 음성 중** 몇 %를 맞게 통과시켰나 | 427/700 = **.610** |

읽는 법을 한 줄로:

- **precision 은 "내 말이 얼마나 믿을 만한가"** — 분모가 **모델이 한 말**(TP+FP)이다.
- **recall 은 "얼마나 안 놓쳤나"** — 분모가 **실제 정답**(TP+FN)이다.
- 의사 비유: precision = "암이라고 한 사람 중 실제 암 비율", recall = "암 환자 중 찾아낸 비율".

곁들여 두 개 더 (오늘 계산만 해 보고 지나간다):

- **NPV**(음성 예측도) = TN/(TN+FN) = 427/564 = **.757** — "아니라고 한 사람 중 진짜 아닌 비율"
- **균형정확도** balanced accuracy = (recall + specificity)/2 = **.613**
  → 불균형 데이터에서 accuracy 대신 쓸 수 있는 값. Dummy 는 여기서 정확히 **.500** 을 받는다."""),

code(r'''# TODO: 네 지표를 네 숫자에서 직접 계산하라 (sklearn 없이, 공식 그대로)
accuracy    = (TP + TN) / (TP + FN + FP + TN)
precision   = TP / (TP + ____)      # ← 모델이 '양성'이라고 한 사람 전체가 분모다
recall      = TP / (TP + ____)      # ← 실제 양성인 사람 전체가 분모다
specificity = TN / (TN + FP)
npv         = TN / (TN + FN)
balanced    = (recall + specificity) / 2

print(f"정확도  accuracy    = ({TP}+{TN})/{len(ytr)} = {accuracy:.4f}")
print(f"정밀도  precision   = {TP}/{TP + FP} = {precision:.4f}   ← 지목한 사람 중 진짜")
print(f"재현율  recall      = {TP}/{TP + FN} = {recall:.4f}   ← 실제 양성 중 찾아낸 비율")
print(f"특이도  specificity = {TN}/{TN + FP} = {specificity:.4f}")
print(f"NPV                 = {TN}/{TN + FN} = {npv:.4f}")
print(f"균형정확도          = ({recall:.3f}+{specificity:.3f})/2 = {balanced:.4f}")

# sklearn 이 같은 값을 주는지 대조한다
from sklearn.metrics import precision_score, recall_score, accuracy_score
print(f"\n대조: sklearn precision {precision_score(ytr, pred_lr):.4f} · "
      f"recall {recall_score(ytr, pred_lr):.4f} · accuracy {accuracy_score(ytr, pred_lr):.4f}")'''),
code(r'''# CHECK Step6-metrics
try:
    assert abs(precision - precision_score(ytr, pred_lr)) < 1e-9, "precision 의 분모는 TP+FP 다"
    assert abs(recall - recall_score(ytr, pred_lr)) < 1e-9, "recall 의 분모는 TP+FN 다"
    assert abs(precision - 0.4451) < 0.005 and abs(recall - 0.6152) < 0.005, "실측과 다르다"
    print("✅ PASS — precision .445 · recall .615 · specificity .610 · accuracy .612.")
    print("   같은 네 숫자에서 나왔는데 값이 다 다르다. **어느 질문을 하느냐가 다르기 때문**이다.")
    print("   → 그래서 '성능이 얼마인가'라는 질문은 그 자체로 불완전하다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: precision 은 '모델이 한 말'이 분모, recall 은 '실제 정답'이 분모다.")'''),
md("""<details><summary>💡 힌트 / 정답</summary>

```python
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

외우는 요령: **precision 의 P 는 Positive 라고 **말한** 사람**(TP+FP), **recall 의 분모는
**실제** 양성**(TP+FN). 헷갈리면 "무엇으로 나누는가"만 물어보면 된다.
</details>"""),

md("""## Step 6 — 평가 ⑤: 지표 하나만 믿으면 안 되는 이유 (의사 4명)

네 지표는 **각각 혼자 두면 전부 속일 수 있다.** 의사 네 명으로 보면 분명해진다.

| 의사 | 하는 짓 | 잘 나오는 지표 | 무너지는 지표 | 우리 데이터 실측 |
|---|---|---|---|---|
| 🙅 **"다 괜찮습니다" 의사**<br>(Dummy 전부 0) | 아무도 암이라고 안 한다 | **accuracy .663**<br>**specificity 1.000** | recall **.000** | 고스트레스 356명 **전원 놓침** |
| 🙋 **"다 암입니다" 의사**<br>(Dummy 전부 1) | 전원을 암이라고 한다 | **recall 1.000** | precision .337<br>specificity **.000** | 700명에게 헛경보 |
| 🔬 **"확실할 때만" 의사** | 100% 확신할 때만 말한다 | **precision 높음** | recall 붕괴 | 임계값 .70 → precision .596 이지만 recall **.157** |
| 🧑‍⚕️ **정직한 의사**<br>(로지스틱) | 애매한 것도 말한다 | 전부 **중간** | — | precision .445 · recall .615 · AUC .653 |

읽어야 할 교훈:

- **accuracy 가 높다** → 다수 클래스만 찍고 있을 수 있다.
- **recall 이 1.0 이다** → 전원을 양성이라고 하고 있을 수 있다. (**precision 을 같이 본다**)
- **precision 이 높다** → 확실한 몇 명만 지목하고 나머지를 다 놓치고 있을 수 있다. (**recall 을 같이 본다**)
- **specificity 가 1.0 이다** → 아무도 지목하지 않고 있을 수 있다.

> 🔴 **precision 과 recall 은 서로를 붙잡아 주는 짝이다.** 하나를 올리면 보통 다른 하나가 내려간다
> (임계값을 낮추면 recall↑ precision↓, 높이면 반대). 그래서 **둘을 한 숫자로 묶는 지표**가 필요해진다
> — 그것이 다음에 볼 **F score** 다."""),

code(r'''# 의사 4명을 실제로 만들어 비교한다 (전부 train 5-fold CV)
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, average_precision_score

def row(name, pred, prob=None):
    tn, fp, fn, tp = confusion_matrix(ytr, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")     # 아무도 지목 안 하면 정의되지 않는다
    rec  = tp / (tp + fn)
    spec = tn / (tn + fp)
    return {"모델": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "acc": (tp + tn) / len(ytr), "precision": prec, "recall": rec,
            "특이도": spec, "F1": f1_score(ytr, pred, zero_division=0),
            "AUC": roc_auc_score(ytr, prob) if prob is not None else 0.5}

allneg = np.zeros(len(ytr), dtype=int)
allpos = np.ones(len(ytr), dtype=int)
strict = (prob_lr >= 0.70).astype(int)      # "확실할 때만" 의사 — 임계값을 .5 → .7 로 올린다

tab = pd.DataFrame([
    row("🙅 다 괜찮습니다 (전부 0)", allneg),
    row("🙋 다 암입니다 (전부 1)", allpos),
    row("🔬 확실할 때만 (임계값 .70)", strict, prob_lr),
    row("🧑‍⚕️ 로지스틱 (임계값 .50)", pred_lr, prob_lr),
])
pd.set_option("display.width", 160)
print(tab.round(3).to_string(index=False))
print("\n→ 각 행에서 '가장 좋아 보이는 칸' 하나만 보면 어느 모델이든 최고가 될 수 있다.")
print("  precision 이 정의되지 않는 경우(NaN)도 있다 — 아무도 지목하지 않으면 분모가 0 이다.")'''),

md("""## Step 6 — 평가 ⑥: **F score** — precision 과 recall 을 한 숫자로

precision 과 recall 은 짝이지만, 모델을 **하나 고르려면** 결국 숫자 하나가 필요하다.
그래서 둘을 묶는다. 그런데 **어떻게** 묶느냐가 중요하다.

```
산술평균  (P + R) / 2                ❌
조화평균  2 × P × R / (P + R)  = F1   ✅
```

**왜 산술평균이 아닌가.** 정밀도 1.00 / 재현율 0.01 인 모델을 생각해 보자 —
"100% 확신하는 한 명만 지목하고 나머지 355명은 다 놓친 의사"다.

- 산술평균 = (1.00 + 0.01)/2 = **.505** — 절반은 하는 것처럼 보인다.
- 조화평균 = 2×1.00×0.01/1.01 = **.020** — 실상에 가깝다.

> **조화평균은 낮은 쪽에 끌려간다.** 둘 다 높아야만 높은 점수가 나온다.
> 이것이 F1 을 쓰는 이유다 — "한쪽을 희생해서 다른 쪽을 올리는" 꼼수를 벌한다.

우리 실측: precision **.445**, recall **.615** → **F1 = .517**

### F score 를 이루는 네 개의 값

F1 은 결국 **네 칸 중 세 칸**에서 만들어진다.

| 값 | F1 에서의 역할 |
|---|---|
| **TP** | 분자 — 잘한 것 |
| **FP** | precision 을 끌어내린다 (헛경보) |
| **FN** | recall 을 끌어내린다 (놓침) |
| **TN** | 🔴 **F1 공식에 아예 없다** |

마지막 줄이 F1 의 성격이자 맹점이다. 맞게 통과시킨 **427명은 F1 에 한 글자도 기여하지 않는다.**
불균형 데이터에서 "다수 클래스를 맞힌 공"을 인정하지 않기 때문에 accuracy 보다 정직하지만,
그 대신 **음성을 얼마나 잘 통과시켰는지는 F1 이 말해 주지 않는다.**"""),

code(r'''# TODO: F1 을 직접 계산하고, 산술평균과 비교하라
f1_manual = 2 * precision * recall / (____ + ____)      # ← 조화평균의 분모를 채워라
arith     = (precision + recall) / 2

print(f"precision {precision:.4f} · recall {recall:.4f}")
print(f"  조화평균 F1 = {f1_manual:.4f}   (sklearn: {f1_score(ytr, pred_lr):.4f})")
print(f"  산술평균     = {arith:.4f}   ← 조금 후하다\n")

# 극단적인 경우에서 둘의 차이가 드러난다
for p, r, label in [(1.00, 0.01, "확실한 1명만 지목"), (0.337, 1.00, "전원 양성"),
                    (0.445, 0.615, "우리 로지스틱")]:
    print(f"  P={p:.3f} R={r:.3f}  ({label:14s})  산술 {(p+r)/2:.3f}  vs  조화 {2*p*r/(p+r):.3f}")'''),
code(r'''# CHECK Step6-f1
try:
    assert abs(f1_manual - f1_score(ytr, pred_lr)) < 1e-9, "F1 = 2PR/(P+R)"
    assert abs(f1_manual - 0.5165) < 0.005, f"실측 .516 근처여야 한다 (지금 {f1_manual:.4f})"
    print(f"✅ PASS — F1 = {f1_manual:.4f}. 조화평균이라 낮은 쪽(precision .445)에 끌려간다.")
    print("   '확실한 1명만 지목' 모델: 산술평균 .505 vs F1 .020 — 조화평균은 속지 않는다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 조화평균의 분모는 두 값의 '합'이다.")'''),

md("""## Step 6 — 평가 ⑦: 그런데 **F1 도 속는다**

방금 만든 "🙋 다 암입니다" 의사(전부 1)의 F1 을 다시 보라.

| 모델 | precision | recall | **F1** | AUC |
|---|---|---|---|---|
| 🙋 전부 1 (학습 없음) | .337 | 1.000 | **.5042** | **.500** |
| 🧑‍⚕️ 로지스틱 (18개 변수로 학습) | .445 | .615 | **.5165** | **.653** |

**차이가 .012 다.** 18개 변수로 학습한 모델과, 아무 생각 없이 전원을 양성이라 찍는 모델이
F1 으로는 **거의 구별되지 않는다.** 그런데 AUC 로는 .500 vs .653 으로 확실히 갈린다.

왜 이런 일이? 양성 비율이 33.7% 나 되기 때문이다. 전원을 양성이라 찍으면 recall 은
공짜로 1.000 이 되고, precision 도 기저율만큼(.337) 은 받는다 → 조화평균이 .5 를 넘긴다.

> 🔴 **결론: F1 도 단독으로는 못 믿는다.** 그래서 우리 규칙은 늘 같다 —
> **여러 지표를 동시에 · Dummy 를 항상 옆에 · 사람 수(혼동행렬)를 함께 본다.**

그리고 F1 에는 질문이 하나 더 남아 있다.
F1 은 precision 과 recall 을 **정확히 1:1** 로 똑같이 중요하게 취급한다.
**우리 맥락에서 그게 맞는가?** — Step 6 ③에서 우리는 "FN 이 더 아프다"고 말했다.
그렇다면 1:1 은 우리 판단과 어긋난다. 그 어긋남을 고치는 도구가 **F-beta** 다."""),

md("""## Step 6 — 평가 ⑧: **F-beta** — 무엇을 몇 배 더 중시할 것인가

$$F_\\beta = (1+\\beta^2)\\cdot\\frac{P \\times R}{\\beta^2 P + R}$$

**β 는 "recall 을 precision 보다 몇 배 중시하는가"** 를 적는 자리다.

| β | 뜻 | 언제 |
|---|---|---|
| **β = 0.5** (F0.5) | precision 을 2배 중시 | 헛경보(FP)의 비용이 클 때 — 개입 자체가 위험할 때 |
| **β = 1** (F1) | 1:1 | 특별한 이유가 없을 때의 기본값 |
| **β = 2** (F2) | recall 을 2배 중시 | 놓침(FN)의 비용이 클 때 — **검진·선별** |

같은 모델이라도 **임계값(threshold)** 을 어디에 두느냐로 precision 과 recall 이 움직인다.
로지스틱이 내놓는 것은 사실 0/1 이 아니라 **확률**이고, "0.5 이상이면 양성"은
**우리가 정한 또 하나의 하이퍼파라미터**일 뿐이다.

> 🎯 그래서 β 를 정한다는 것은 **"어느 임계값을 좋은 임계값이라고 부를 것인가"** 를 정하는 일이다."""),

code(r'''# TODO: 임계값을 움직이며 precision·recall·F 를 관찰하라 (선별 목적이면 어느 β?)
print(f"{'임계값':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s} "
      f"{'precision':>10s} {'recall':>7s} {'F0.5':>6s} {'F1':>6s} {'F2':>6s}")
for t in (0.30, 0.40, 0.50, 0.60, 0.70):
    p = (prob_lr >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(ytr, p, labels=[0, 1]).ravel()
    print(f"{t:6.2f} {tp:4d} {fn:4d} {fp:4d} {tn:4d} "
          f"{tp/(tp+fp):10.3f} {tp/(tp+fn):7.3f} "
          f"{fbeta_score(ytr, p, beta=0.5, zero_division=0):6.3f} "
          f"{fbeta_score(ytr, p, beta=1.0, zero_division=0):6.3f} "
          f"{fbeta_score(ytr, p, beta=2.0, zero_division=0):6.3f}")

# 우리 프로젝트는 '선별'이 목적이다. 놓침(FN)과 헛경보(FP) 중 무엇을 더 무겁게 볼 것인가?
선별_목적의_beta = ____        # ← 0.5 / 1 / 2 중 하나를 숫자로 적어라
print(f"\n내가 고른 beta = {선별_목적의_beta}")'''),
code(r'''# CHECK Step6-fbeta
try:
    assert 선별_목적의_beta == 2, "선별(screening)은 놓침(FN)이 더 아프다 → recall 을 더 중시한다"
    best = {b: max(((t, fbeta_score(ytr, (prob_lr >= t).astype(int), beta=b, zero_division=0))
                    for t in np.arange(0.20, 0.81, 0.01)), key=lambda x: x[1])
            for b in (0.5, 1.0, 2.0)}
    for b, (t, s) in best.items():
        print(f"  F{b} 를 최대로 만드는 임계값 = {t:.2f} (점수 {s:.3f})")
    assert best[2.0][0] < best[0.5][0], "recall 중시(β=2)일수록 임계값이 낮아진다"
    print("\n✅ PASS — **β 를 바꾸면 '최선의 임계값'이 통째로 바뀐다.**")
    print("   β=0.5 → 임계값 .58 (신중) · β=1 → .36 · β=2 → .20 (적극적으로 지목)")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: 놓치는 것이 더 아프면 recall 쪽에 무게를 준다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 그리고 β 를 키울 때의 함정</summary>

```python
선별_목적의_beta = 2
```

**실측 (train CV 확률 기준)**

| 임계값 | TP | FN | FP | TN | precision | recall | F0.5 | F1 | F2 |
|---|---|---|---|---|---|---|---|---|---|
| .30 | 326 | 30 | 559 | 141 | .368 | .916 | .418 | .525 | **.706** |
| .40 | 277 | 79 | 430 | 270 | .392 | .778 | .435 | .521 | .650 |
| **.50** | 219 | 137 | 273 | 427 | .445 | .615 | .471 | **.517** | .572 |
| .60 | 131 | 225 | 126 | 574 | .510 | .368 | **.473** | .427 | .390 |
| .70 | 56 | 300 | 38 | 662 | .596 | .157 | .383 | .249 | .184 |

같은 모델, 같은 확률인데 **임계값 하나로 TP 가 56명에서 326명까지** 움직인다.
그리고 β 마다 "최선"이 다르다 — F0.5 는 .58, F1 은 .36, F2 는 .20 에서 최대다.

### 🔴 그런데 F2 만 보고 달리면?

β=2 를 최대로 만드는 임계값은 **.20** 이고, 그때 recall 은 **.983** — 사실상 **전원을
양성이라고 찍는 것**에 가깝다(FP 678명). 실제로 "전부 1" 더미 모델의 F2 는 **.718** 로,
우리가 임계값을 최적화해 얻은 .714 **보다도 높다.**

> **β 를 키우면 F-beta 는 "전원 양성" 쪽으로 수렴한다.**
> β 는 성능을 올리는 손잡이가 아니라, **비용에 대한 판단을 적어 두는 자리**다.
> 그래서 β 는 결과를 보고 고르는 게 아니라 **미리 정하고 기록**해야 한다 (분위수 0.75 와 똑같다).

**우리 프로젝트의 결정**: 선별 목적이라 recall 쪽으로 기울지만(β>1), FP 는 **낙인 비용**이라
무한정 키울 수 없다. 그래서 ① 임계값은 기본값 **.5 로 고정**하고 ② recall·precision·F1 을
**같이 보고**하며 ③ 임계값 조정은 8차시 **민감도 분석**으로 미룬다. 오늘 튜닝하지 않는 이유는
하나 더 있다 — **튜닝을 하려면 결국 채점을 봐야 하고, 그건 test 봉인 원칙과 부딪힌다.**
</details>"""),

md("""## Step 6 — 평가 ⑨: **AUC** — 임계값을 정하기 전에 모델을 재는 법

지금까지의 지표는 전부 **임계값을 정한 뒤에야** 계산할 수 있었다. 그런데 임계값은
우리가 정하는 값이다. **임계값과 무관하게** 모델 자체를 재는 방법이 있다.

### ① 먼저 ROC 곡선을 그린다

로지스틱이 실제로 내놓는 것은 라벨이 아니라 **확률(위험 점수)** 이다.
임계값을 **1.00 에서 0.00 으로 내리면서**, 매 지점마다 **점 하나**를 찍는다. 축은 이미 배운 두 지표다.

| | 축 | 공식 | 방향 |
|---|---|---|---|
| **세로축** | **재현율 TPR** (true positive rate) | TP/(TP+FN) — 실제 양성 중 찾아낸 비율 | 높을수록 좋다 ↑ |
| **가로축** | **거짓양성률 FPR** (false positive rate) | FP/(FP+TN) = **1 − 특이도** — 헛경보 비율 | 낮을수록 좋다 ← |

- 임계값 **1.00** → 아무도 지목하지 않는다 → TP 0, FP 0 → 점 **(0, 0)** (왼쪽 아래)
- 임계값 **0.00** → 전원을 지목한다 → TP 356, FP 700 → 점 **(1, 1)** (오른쪽 위)
- 그 사이 모든 임계값의 점을 이으면 **ROC 곡선**이 된다.

### ② 그 곡선 아래 면적이 AUC 다

> **AUC 의 뜻 (이 한 문장만 기억한다):
> 고스트레스인 학생 한 명과 아닌 학생 한 명을 무작위로 뽑았을 때,
> 모델이 고스트레스인 쪽에 더 높은 점수를 줄 확률.**

의사 비유로는 — **환자 한 명과 건강한 사람 한 명을 나란히 세웠을 때, 누가 더 위험한지
순서를 맞히는 능력**이다. 진단을 내리는 게 아니라 **줄을 세우는 능력**을 재는 것이다.

| AUC | 뜻 |
|---|---|
| .50 | 동전 던지기 — 곡선이 **대각선**을 따라간다. Dummy 는 정확히 여기다 |
| .60~.70 | 약하지만 정보가 있다 ← **우리 모델 .653** |
| .80+ | 좋다 |
| **1.00** | 곡선이 **왼쪽 위 모서리**를 지난다 — **🔴 축하가 아니라 경보다** (Step 7) |"""),

code(r'''# TODO: 곡선 위의 점을 직접 찍어 보자 — 가로축(FPR) 공식을 채워라
P_all = int(ytr.sum())            # 실제 양성 356명
N_all = int((ytr == 0).sum())     # 실제 음성 700명

print(f"{'임계값':>6s} {'TP':>4s} {'FP':>4s} {'세로 TPR':>9s} {'가로 FPR':>9s}   찍히는 점")
for t in (1.00, 0.70, 0.60, 0.50, 0.40, 0.30, 0.00):
    p = (prob_lr >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(ytr, p, labels=[0, 1]).ravel()
    tpr_t = tp / P_all
    fpr_t = fp / ____                  # ← 거짓양성률의 분모는? (실제 음성 전체다)
    mark = "  ← 우리 기본값" if t == 0.50 else ""
    print(f"{t:6.2f} {tp:4d} {fp:4d} {tpr_t:9.3f} {fpr_t:9.3f}   ({fpr_t:.2f}, {tpr_t:.2f}){mark}")

print("\n임계값을 내릴수록 오른쪽 위로 이동한다 — TP 도 늘지만 FP 도 같이 는다. 공짜가 없다.")'''),
code(r'''# CHECK Step6-roc-points
try:
    p50 = (prob_lr >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(ytr, p50, labels=[0, 1]).ravel()
    assert abs(fp / N_all - 0.390) < 0.005, f"임계값 .50 의 FPR 은 273/700 = .390 이다"
    assert abs(tp / P_all - 0.615) < 0.005, "임계값 .50 의 TPR 은 219/356 = .615 다"
    print("✅ PASS — 임계값 .50 일 때 우리는 곡선 위의 (0.39, 0.62) 지점에 서 있다.")
    print("   임계값을 바꾸면 이 점이 곡선 위를 미끄러진다. **곡선 자체는 변하지 않는다.**")
    print("   → 곡선 = 모델의 실력 · 점 = 우리의 선택. AUC 는 점이 아니라 곡선 전체의 요약이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: FPR = FP / (FP + TN) = FP / 실제 음성 수")'''),

code(r'''# 점 5개가 아니라 '모든' 임계값의 점을 이으면 곡선이 된다 — 그려 보자
from sklearn.metrics import roc_curve
from maps_risk import plots          # 한글 폰트 설정을 겸한다
import matplotlib.pyplot as plt

fpr, tpr, thr = roc_curve(ytr, prob_lr)
print(f"곡선을 이루는 점 {len(fpr)}개 (임계값이 바뀌는 지점마다 하나씩)")

fig, ax = plt.subplots(figsize=(5, 4.6))
ax.plot(fpr, tpr, lw=2.5, label=f"로지스틱 (AUC = {roc_auc_score(ytr, prob_lr):.3f})")
ax.fill_between(fpr, tpr, alpha=0.15)                     # 이 면적이 AUC 다
ax.plot([0, 1], [0, 1], "--", lw=1.5, color="gray", label="동전 던지기 (AUC = .500)")
ax.plot([0, 0, 1], [0, 1, 1], ":", lw=1.5, color="green", label="완벽한 모델 (AUC = 1.000)")
ax.scatter([273 / N_all], [219 / P_all], s=90, zorder=5, color="orange", label="임계값 .50 (우리 기본값)")
ax.set_xlabel("FPR (거짓양성률) = 1 - 특이도")   # 폰트에 따라 유니코드 마이너스(−)가 깨진다; ax.set_ylabel("TPR (재현율)")
ax.set_title("ROC 곡선 — train 5-fold CV"); ax.legend(fontsize=8, loc="lower right")
plt.savefig("reports/figures/roc_curve_session4.png", dpi=150, bbox_inches="tight")
plt.show()

# 면적을 직접 계산해 본다 (사다리꼴 넓이의 합) — roc_auc_score 와 같은가?
area = np.trapz(tpr, fpr)
print(f"\n사다리꼴로 직접 잰 면적 = {area:.4f}")
print(f"roc_auc_score        = {roc_auc_score(ytr, prob_lr):.4f}   ← 같다. AUC 는 말 그대로 '면적'이다.")'''),

md("""### 왜 그 '면적'이 '순서를 맞힐 확률'과 같은가

면적과 확률 — 전혀 달라 보이는 두 말이 왜 같은 값인가. 이어 붙이면 이렇다.

임계값을 위에서 아래로 내리면 **점수가 높은 사람부터 한 명씩** "양성"으로 넘어온다.
그때 **양성인 사람이 넘어오면 곡선이 위로**, **음성인 사람이 넘어오면 곡선이 오른쪽으로** 간다.
그러니까 ROC 곡선은 결국 **1,056명을 점수 순으로 세워 놓고 양성/음성이 어떤 순서로 나오는지**를
그린 계단이다.

- 양성이 **먼저** 많이 나올수록 → 곡선이 **일찍 위로** 붙고 → 아래 면적이 커진다.
- 양성 356명이 **전부 앞**에 서면 → 곡선이 왼쪽 벽을 타고 올라간다 → 면적 **1.000**
  (모든 쌍에서 양성이 이긴다 = 확률 1)
- 순서가 **뒤죽박죽**이면 → 곡선이 대각선을 따라간다 → 면적 **.500** (쌍의 절반만 이긴다)

그래서 면적은 곧 **"아무 양성 한 명이 아무 음성 한 명보다 앞에 설 확률"** 이다.
말로만 하면 안 믿기니 — **다음 셀에서 실제로 20만 번 뽑아 확인한다.**

> 🔴 AUC 는 "몇 명을 맞혔나"가 아니라 **"줄을 잘 세웠나"** 를 재는 지표다.
> 확률값이 전부 0.01 씩 작아져도 **순서가 그대로면 AUC 는 똑같다** — 이 성질이 장점이자 함정이다."""),

code(r'''# AUC 의 정의를 그대로 실험으로 확인한다 — 양성 1명 · 음성 1명 뽑아 비교하기
rng = np.random.default_rng(cfg["random_seed"])
pos = prob_lr[ytr.values == 1]
neg = prob_lr[ytr.values == 0]

N = 200_000
wins = (pos[rng.integers(0, len(pos), N)] > neg[rng.integers(0, len(neg), N)]).mean()

print(f"양성 1명·음성 1명을 {N:,}번 뽑아 비교 → 양성이 더 높은 점수를 받은 비율 = {wins:.4f}")
print(f"roc_auc_score 가 준 값                                      = {roc_auc_score(ytr, prob_lr):.4f}")
print("→ 같은 값이다. AUC 는 '순서를 맞힐 확률'이라는 뜻이 정확히 이것이다.\n")

# 불균형에서는 PR-AUC 를 같이 본다 — 기저율이 baseline 이다
ap = average_precision_score(ytr, prob_lr)
print(f"PR-AUC (average precision) = {ap:.4f}   ← 우연 수준의 기준선은 양성 기저율 {ytr.mean():.3f}")
print(f"  ROC-AUC 의 기준선은 항상 .500 이지만, PR-AUC 의 기준선은 데이터마다 다르다.")
print(f"  우리 향상 폭: {ap:.3f} − {ytr.mean():.3f} = {ap - ytr.mean():+.3f}")'''),
code(r'''# CHECK Step6-auc
try:
    assert abs(wins - roc_auc_score(ytr, prob_lr)) < 0.01, "쌍 뽑기 비율 ≈ AUC 여야 한다"
    assert abs(ap - ytr.mean()) > 0.10, "PR-AUC 가 기저율보다 의미 있게 높아야 한다"
    print("✅ PASS — AUC .652 ≈ 쌍 뽑기 실험 결과. 정의를 실험으로 확인했다.")
    print(f"   PR-AUC {ap:.3f} vs 기저율 {ytr.mean():.3f} — 우연보다는 낫다. 그러나 극적이지 않다.")
    print("   3차시에 상관이 전부 .3 이하였다. 그 결과가 여기 그대로 나타난 것이다.")
except Exception as e:
    print("❌ FAIL —", e)'''),

md("""## Step 6 — 평가 ⑩: AUC 도 혼자 두면 속인다

AUC 는 편리하다 — 임계값과 무관하고, 기준선이 항상 .5 라 모델끼리 비교하기 좋다.
그래서 **논문에 AUC 하나만 적는 일**이 흔하다. 그러면 안 되는 이유가 네 가지다.

| | 함정 | 우리 데이터에서 |
|---|---|---|
| ① | **불균형에서 낙관적으로 보인다** — 음성이 많으면 FPR 분모가 커서 헛경보가 희석된다 | AUC .653 은 그럴듯한데, PR-AUC 는 **.475**(기저율 .337) — 향상 폭이 훨씬 작아 보인다 |
| ② | **순위만 본다 — 확률값의 정확성(보정)은 말해 주지 않는다** | "위험 확률 0.8" 이 진짜 80% 라는 보장이 없다. 순서만 맞으면 AUC 는 만점이다 |
| ③ | **몇 명을 지목할지 정해 주지 않는다** | 임계값 .3 이면 885명, .7 이면 94명을 지목한다. AUC 는 이 둘을 구분하지 않는다 |
| ④ | **1.0 은 실력이 아니라 사고 신호다** | Step 7 에서 직접 만든다 |

> 🔴 그래서 우리 보고 규칙(AGENTS.md)은 이렇게 되어 있다 —
> **ROC-AUC · PR-AUC · recall · precision · F1 · balanced accuracy · 혼동행렬을 함께,
> 그리고 항상 Dummy 를 옆에.** 지표 하나만 적힌 표는 그 자체로 의심 대상이다.

### 오늘의 지표 읽는 순서 (외워 둘 것)

1. **혼동행렬 먼저** — 비율보다 **사람 수**. "356명 중 219명" 이 "recall .615" 보다 정직하다.
2. **Dummy 와 나란히** — 우리 숫자가 "학습 없이도 나오는 숫자"보다 나은지부터 본다.
3. **짝으로 읽기** — precision ↔ recall, recall ↔ specificity. 하나만 좋으면 의심한다.
4. **목적에 맞는 요약 하나** — 우리는 선별이므로 recall 을 앞세우되 F1 을 같이 적는다.
5. **너무 좋으면 멈춘다** — AUC .9 이상이 갑자기 나오면 성능이 아니라 **데이터 흐름**을 본다."""),

code(r'''# 오늘 배운 지표를 한 표로 — 이것이 8차시 reports/model_metrics.csv 의 축소판이다
summary = pd.DataFrame([
    row("Dummy (전부 0)", allneg),
    row("Dummy (전부 1)", allpos),
    row("로지스틱 (Model A)", pred_lr, prob_lr),
])
summary["F2"] = [fbeta_score(ytr, p, beta=2, zero_division=0) for p in (allneg, allpos, pred_lr)]
summary["PR-AUC"] = [ytr.mean(), ytr.mean(), average_precision_score(ytr, prob_lr)]
print(summary.round(3).to_string(index=False))
print(f"\n기저율(양성 비율) = {ytr.mean():.3f} — PR-AUC 는 이 값이 우연 수준이다.")
print("Dummy 두 개를 양쪽에 두고 보면, 우리 모델이 '어디쯤'인지 한눈에 보인다.")

from maps_risk import plots
plots.class_distribution(frame["high_stress"])
print("\n✅ reports/figures/class_distribution.png")'''),

# ══════════════════════════════════════════════════════════════════
# Step 7 — 데이터 누출
# ══════════════════════════════════════════════════════════════════
md("""## Step 7 — 🔥 오늘의 백미: 일부러 누출시키기 (두 번째 봉우리)

Step 5 에서 **train AUC 1.000** 을 만들었다. 그건 CV 가 잡아냈다(.519).
이제 **CV 도 못 잡는** 사고를 만든다.

Step 2 의 ②번을 기억하는가 — **"모델의 목표는 손실을 줄이는 것이지 진실을 찾는 것이 아니다.
답을 베낄 방법이 있으면 반드시 베낀다."**

우리 target 은 6차 문화적응 스트레스 점수로 만들었다. 그렇다면 —
**그 6차 점수 자체를 feature 로 넣으면** 어떻게 될까?

> 🖐 먼저 **예측**해 보라. AUC 가 얼마나 나올까? 0.7? 0.9?"""),

code(r'''# TODO: 6차 스트레스 점수를 feature 에 넣어 보라 (해서는 안 되는 짓이다)
leak_feats = feats + ["________________"]      # ← target 을 만든 그 컬럼 이름

honest = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=42)),
                         Xtr, ytr, cv=cv, scoring="roc_auc").mean()
leaked = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=42)),
                         frame.loc[idx_tr, leak_feats], ytr, cv=cv, scoring="roc_auc").mean()

print(f"정직한 모델 (5차 변인만)      AUC = {honest:.4f}")
print(f"누출 모델   (6차 점수 포함)   AUC = {leaked:.4f}   ← ?!")'''),
code(r'''# CHECK Step7
try:
    assert leaked > 0.99, f"누출 모델 AUC 가 1.0 근처여야 한다 (지금 {leaked:.4f})"
    assert honest < 0.80, f"정직한 모델은 훨씬 낮아야 한다 (지금 {honest:.4f})"
    print(f"✅ PASS — AUC {leaked:.4f}. **완벽한 모델**이 만들어졌다.")
    print("   그런데 이번엔 **CV 도 1.000 이다.** 과적합처럼 무너지지 않는다 — 그게 더 무섭다.")
    print("   축하할 일일까? 아니다. 이건 성공이 아니라 **경보음**이다.")
except Exception as e:
    print("❌ FAIL —", e, "\n힌트: target(고스트레스)을 만들 때 쓴 컬럼이 frame 안에 그대로 있다.")'''),
md("""<details><summary>💡 힌트 / 정답 — 왜 이것이 쓰레기인가</summary>

```python
leak_feats = feats + ["acculturative_stress_w6"]
```

실측: 정직한 모델 **AUC .653** → 누출 모델 **AUC 1.0000**. 완벽하다.

**무슨 일이 일어났나.** `high_stress` 는 `acculturative_stress_w6 >= 1.5` 로 **만든** 라벨이다.
그 원본 점수를 feature 로 주면, 모델이 할 일은 "1.5 보다 큰가?" 하나뿐이다.
**답안지를 보고 답을 쓴 것**이다. 예측한 게 아니라 **베낀 것**이다.

**왜 쓸모없나 — 세 가지 이유**

1. **실전에서는 그 변수가 없다.** 우리가 하려는 일은 "중2 시점 정보로 **1년 뒤**를 미리
   구분하는 것"이다. 중3 스트레스 점수를 이미 알고 있다면 예측할 이유가 없다.
2. **아무것도 배우지 못한다.** "무엇이 위험요인인가"에 답하려던 건데,
   이 모델의 답은 "스트레스가 높으면 스트레스가 높다"뿐이다.
3. **위험하다.** 숫자가 좋으니 아무도 의심하지 않는다. 조용히 틀린 결론이 배포된다.

**과적합과 비교하면 성격이 분명해진다.**

| | 과적합 (Step 5) | 누출 (Step 7) |
|---|---|---|
| train | 1.000 | 1.000 |
| CV | **.519 로 무너진다** | **1.000 그대로** |
| 우리를 지켜 주는 것 | CV 가 잡아 준다 | **CV 도 못 잡는다** → 사람이 데이터 흐름을 봐야 한다 |

> 🔴 **오늘의 문장: AUC 1.0 은 축하가 아니라 경보다.**
> 실무에서 갑자기 성능이 튀면 가장 먼저 의심할 것은 모델이 아니라 **데이터 흐름**이다.

**"그럼 5차 스트레스를 넣는 Model B 도 누출 아닌가?"** — 아니다. 5차는 **예측 시점(중2)
당시**의 정보라 실제 상황에서도 알 수 있다. 누출의 기준은 "변수가 강력한가"가 아니라
**"예측 시점에 알 수 있는 정보인가"** 이다. RQ3 가 바로 그것을 정량화한다.
</details>"""),

md("""## Step 7 — 누출의 세 얼굴

방금 본 것은 가장 노골적인 누출이다. 실제로는 훨씬 **알아채기 어려운** 형태로 온다.

| | 이름 | 무엇 | 우리 방어 |
|---|---|---|---|
| ① | **시간 누출** | 예측 시점 이후의 정보가 X 에 들어감 | `assert_no_wave6_predictors()` — 6차 컬럼이 X 에 있으면 멈춘다 |
| ② | **라벨 누출** | test 를 보고 cutoff 를 정함 | cutoff 는 `scores.loc[idx_tr]` 로만 |
| ③ | **전처리 누출** | 전체로 평균·중앙값·변수선택을 fit | 전부 `Pipeline` 안에 |

②와 ③을 직접 재 보자. ①처럼 극적일까?"""),

code(r'''# ③ 전처리 누출 — 전체로 표준화한 뒤 CV vs Pipeline 안에서 폴드마다 표준화
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

imp, sc = SimpleImputer(strategy="median"), StandardScaler()
X_leaked = pd.DataFrame(sc.fit_transform(imp.fit_transform(frame[feats])),   # 전체로 fit ❌
                        index=frame.index, columns=feats)

a_leak = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                         X_leaked.loc[idx_tr], ytr, cv=cv, scoring="roc_auc").mean()
a_ok = cross_val_score(build(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
                       Xtr, ytr, cv=cv, scoring="roc_auc").mean()

print(f"전체로 표준화 후 CV : AUC = {a_leak:.4f}   ❌ 규칙 위반")
print(f"Pipeline 안에서     : AUC = {a_ok:.4f}   ✅ 규칙 준수")
print(f"차이 = {abs(a_leak - a_ok):.4f}   ← 어라, 거의 없다?")'''),

code(r'''# 그런데 전처리 누출이 늘 이렇게 얌전한 건 아니다.
# 극단적 사례를 만들어 본다: **순수한 잡음** 200개 중에서 '좋아 보이는' 5개를 고르면?
from sklearn.feature_selection import SelectKBest, f_classif

rng2 = np.random.default_rng(0)
noise = pd.DataFrame(rng2.normal(size=(len(frame), 200)), index=frame.index,
                     columns=[f"noise_{i}" for i in range(200)])
Ntr = noise.loc[idx_tr]

# ❌ CV 밖에서 미리 5개를 고른 뒤 CV → 고를 때 이미 정답(ytr)을 다 봤다
picked = Ntr.columns[SelectKBest(f_classif, k=5).fit(Ntr, ytr).get_support()]
auc_leak = cross_val_score(build(LogisticRegression(max_iter=2000, random_state=42)),
                           Ntr[picked], ytr, cv=cv, scoring="roc_auc").mean()

# ✅ 선택까지 Pipeline 안에 넣어 폴드마다 다시 고르기
inside = Pipeline([("prep", make_preprocessor(scale=False)),
                   ("sel", SelectKBest(f_classif, k=5)),
                   ("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=42))])
auc_ok = cross_val_score(inside, Ntr, ytr, cv=cv, scoring="roc_auc").mean()

print(f"❌ CV 밖에서 미리 고름  : AUC = {auc_leak:.4f}   ← 전부 난수인데!")
print(f"✅ Pipeline 안에서 고름 : AUC = {auc_ok:.4f}   ← 0.5 = 동전 던지기 (정직한 답)")'''),

md("""### Step 7 정리 — 규칙은 결과가 아니라 절차다

오늘 네 가지를 쟀다. 결과가 **제각각**이었다:

| 종류 | 실측 차이 | 해석 |
|---|---|---|
| 과적합 (트리 깊이 제한 없음) | train 1.000 → CV **.519** | CV 가 잡아냈다 |
| ① 시간 누출 (6차 점수) | AUC .653 → **1.000** (CV 도 1.000) | 파국적 · CV 도 못 잡는다 |
| ② 라벨 누출 (cutoff) | 라벨 차이 **0명** | 이번엔 차이 없었다 |
| ③ 전처리 누출 (표준화) | **.0001** | 이번엔 차이 없었다 |
| ③ 전처리 누출 (변수 선택) | .503 → **.589** | 순수한 잡음으로 성능이 만들어졌다 |

②③이 이번에 얌전했다고 규칙을 버릴 수 있을까? 없다. 이유는 Step 4 와 같다 —
**차이가 있는지 확인하려면 이미 규칙을 어겨야 한다.**

그래서 우리는 개별 판단에 맡기지 않고 **구조로 막는다.** `Pipeline` 이 그 장치다:

```python
Pipeline([("prep", make_preprocessor()),   # 결측 대치 + 표준화
          ("clf",  LogisticRegression())])  # 모델
```

이렇게 묶으면 `fit()` 이 호출될 때 **전처리도 그 폴드의 train 으로만** 학습된다.
사람이 매번 조심하는 게 아니라, **틀릴 수 없는 모양으로 만들어 둔 것**이다."""),

md("""## Step 8 — 테스트로 못 박기: 규칙을 코드가 지키게 한다

말로 정한 규칙은 잊힌다. 그래서 이 프로젝트는 누출 방지 규칙을 **테스트**로 박아 뒀다.
`tests/test_no_leakage.py` 를 열면 오늘 배운 것이 그대로 들어 있다:

| 테스트 | 무엇을 막나 |
|---|---|
| `test_wave6_predictor_raises` | 6차 변수가 X 에 들어가면 예외 (① 시간 누출) |
| `test_target_column_cannot_enter_X` | target 컬럼이 X 에 들어가면 예외 |
| `test_cutoff_is_computed_from_train_only` | cutoff 를 전체로 계산하면 실패 (② 라벨 누출) |
| `test_scaler_is_fit_on_train_only` | 스케일러가 전체로 fit 되면 실패 (③ 전처리 누출) |
| `test_all_models_are_pipelines` | 전처리가 Pipeline 밖에 있으면 실패 |
| `test_train_test_ids_do_not_overlap` | train/test 에 같은 사람이 있으면 실패 |

> 누군가 6개월 뒤 코드를 고치다 실수로 규칙을 깨면, **테스트가 빨간불로 알려준다.**
> 이것이 "재현 가능한 연구"의 실제 모습이다 — 착한 의도가 아니라 **자동 검사**."""),

code(r'''!python -m pytest tests/test_no_leakage.py -v --no-header -q'''),

code(r'''# 오늘의 산출물과 '봉인' 확인
print("오늘 만든 것")
print(f"  ✅ feature {len(feats)}개 · 파라미터 {n_param}개 (계수 {n_coef} + 절편 1)")
print(f"  ✅ high_stress 라벨      cutoff={cutoff:.3f} · train {y_all.loc[idx_tr].mean():.1%} 양성")
print(f"  ✅ train/test 분할       {len(idx_tr)} / {len(idx_te)} · seed {cfg['random_seed']}")
print(f"  ✅ 평가 지표 한 세트     혼동행렬 TP{TP}/FN{FN}/FP{FP}/TN{TN} · "
      f"precision {precision:.3f} · recall {recall:.3f} · F1 {f1_manual:.3f} · AUC {honest:.3f}")
print(f"  {'✅' if os.path.exists('reports/figures/class_distribution.png') else '⬜'} reports/figures/class_distribution.png")

print("\n🔒 test 세트 봉인 상태")
print(f"  test {len(idx_te)}명 — 오늘 성능 평가에 한 번도 쓰지 않았다.")
print("  모든 숫자는 train 안 5-fold CV 에서 나왔다. test 는 8차시 최종 평가 때 딱 한 번 연다.")
print("\n※ 오늘 본 AUC 는 전부 CV 값이다. 최종 성능이 아니다 —")
print("  5·6차시에서 모델을 제대로 세우고, 그 뒤에 test 를 연다.")'''),

md("""## 💾 다음 차시를 위해 — 드라이브에 저장\n\n오늘 만든 것 중 **다음 차시가 재료로 쓰는 파일**을 내 드라이브(`program5_state/`)에 넣어 둔다.\n이렇게 해 두면 런타임이 끊겨도, 다른 컴퓨터에서 열어도 **다음 차시가 그냥 시작된다.**\n\n> 🔴 파생 파일이 들어가는 폴더다 — **개인 계정 안에만** 두고 링크 공유·양도하지 않는다."""),
code(handoff_out(push=['reports/figures/*.png'], note="4차시 산출물 — 그림만 남는다 (라벨·분할은 설정에서 매번 같게 재생된다)")),

md("""## 🎯 회고 (5분)

1. **feature 와 parameter 의 차이**를 한 문장씩으로 설명한다면? 우리 프로젝트의 개수는 각각 몇 개인가?
2. Dummy 의 정확도가 66% 인데도 **쓸모없는** 이유를 친구에게 설명한다면?
3. **precision 만 높은 모델**과 **recall 만 높은 모델**은 각각 어떤 의사인가?
   그리고 F1 이 그 둘을 어떻게 벌하는가?
4. **F1 도 속는다**고 했다. "전부 1" 모델의 F1 이 .504 였다. 이때 무엇을 같이 봤어야 하나?
5. 6차 점수를 넣었더니 AUC 가 1.0 이 됐다. **왜 그게 좋은 소식이 아닌가?**
   그리고 그것은 **깊은 트리의 train AUC 1.0** 과 어떻게 다른가?
6. cutoff 를 전체로 계산해도 이번엔 라벨이 하나도 안 바뀌었다.
   **그런데도 규칙을 지켜야 하는 이유**는 무엇인가?

6번이 오늘의 핵심 감각이다 — **규칙은 결과가 아니라 절차로 정당화된다.**

## 📝 과제
- 우리 프로젝트에서 **FN 1명과 FP 1명 중 무엇이 더 나쁜지** 자기 입장을 정하고, 그 근거를 5줄로.
  (정답 없음. 단, "낙인"과 "놓침"을 **둘 다** 언급할 것)
- 이 데이터에서 생길 수 있는 **또 다른 누출 시나리오**를 하나 상상해서 3줄로 적기
  (예: "같은 학교 학생이 train 과 test 에 나뉘어 들어가면?")
- `tests/test_no_leakage.py` 의 테스트 하나를 골라, **무엇을 막는지** 한 문단으로 설명
- 조작적 정의 문장을 **연구윤리에 맞게** 다시 쓰기 (❌ 고위험군 판별 → ✅ …)

## ▶️ 다음 (5차시)
> "오늘 **재료(feature) · 손잡이(parameter) · 학습 · 채점(지표)** 을 전부 세웠다.
> 다음엔 그 손잡이 19개를 **읽는다** — 로지스틱 회귀의 **계수**를 보고
> '어떤 변수가 고스트레스 분류와 가장 강하게 관련되는가'에 답한다.
> 그리고 3차시의 그 문제가 돌아온다: **4점 척도와 5점 척도의 계수를 그대로
> 비교해도 되는가?** 답은 '안 된다'이고, 해결책이 **표준화**다."""),
]

os.makedirs("session4", exist_ok=True)
save(cells, "session4/session4.ipynb")

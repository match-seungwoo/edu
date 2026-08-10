# 데이터 인벤토리 (data_inventory.md)

> 자동 생성: `python scripts/inspect_raw_data.py --raw data/demo_format`  ·  2026-08-10
> 이 문서는 원자료를 **읽기만** 하고 만든다. 원본은 수정하지 않는다.

> ⚠️ **[DEMO] 형식 확인용 가짜 파일이다. MAPS 아님.**

## 1. 발견한 파일 2개

| 파일 | 형식 | 행 | 열 | 상태 |
|---|---|---:|---:|---|
| `DEMO_wave5_NOT_MAPS.csv` | .csv | 5 | 5 | ✅ 읽기 성공 |
| `DEMO_wave6_NOT_MAPS.csv` | .csv | 5 | 3 | ✅ 읽기 성공 |

## 1b. 데이터로 직접 읽지 않은 파일 1개

압축(zip)은 풀어야 데이터가 보인다. PDF·hwp 문서는 코드북/조사표/가이드다.

- `README.md`

## 2. 컬럼 미리보기 (앞 20개)

**`DEMO_wave5_NOT_MAPS.csv`**

```
DEMO_ID, DEMO_W5_STRESS_01, DEMO_W5_STRESS_02, DEMO_W5_SUPPORT_01, DEMO_W5_SUPPORT_02
```

**`DEMO_wave6_NOT_MAPS.csv`**

```
DEMO_ID, DEMO_W6_STRESS_01, DEMO_W6_STRESS_02
```

## 3. 사람이 확인해야 할 것 (Human Review Gate)

- 코드북 확인 완료 플래그: **False**
- 응답자 ID (5차/6차): **None / None**
- 결측 코드: **미확인**
- target(6차 문화적응 스트레스) 문항: **미확인**

### 미검증 예측변인 15개

- [ ] `self_esteem`
- [ ] `ego_resilience`
- [ ] `depression`
- [ ] `social_withdrawal`
- [ ] `life_satisfaction`
- [ ] `family_support`
- [ ] `parenting_attitude`
- [ ] `peer_support`
- [ ] `peer_relationship`
- [ ] `bullying`
- [ ] `school_adjustment`
- [ ] `teacher_support`
- [ ] `bicultural_attitude`
- [ ] `national_identity`
- [ ] `korean_proficiency`

### 미검증 optional 예측변인 1개

- [ ] `previous_acculturative_stress`

> 🔴 컬럼명을 추측해서 채우지 않는다. 코드북에서 확인한 것만 적는다.

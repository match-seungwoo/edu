"""심리척도 점수 계산 — 역채점, 문항 평균, 결측 처리.

왜 존재하나: "문항 여러 개 → 척도 점수 하나"는 심리학 데이터 분석의 기본기이고,
여기서 실수하면(역채점 누락 등) 이후 모든 결과가 틀린다.
"""
import numpy as np
import pandas as pd


def reverse_code(series, scale_min, scale_max):
    """역채점: 1↔4, 2↔3 처럼 값을 뒤집는다.

    받는 것: 문항 Series, 척도 최소·최대값
    돌려주는 것: 뒤집힌 Series
    왜: "나는 쓸모없는 사람이다" 같은 문항은 점수 방향이 반대라서
        그대로 평균 내면 척도 전체가 무의미해진다.
    공식: reversed = (min + max) - original
    """
    return (scale_min + scale_max) - series


def apply_missing_codes(df, cols, missing_codes):
    """결측 코드(-9, 99 등)를 NaN 으로 바꾼 사본을 돌려준다.

    왜: SPSS 파일에서 결측은 숫자로 들어온다. 그대로 두면 평균이 오염된다.
    ★ missing_codes 는 코드북에서 확인한 값만 넣는다 (추측 금지).
    """
    out = df.copy()
    if not missing_codes:
        return out
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace(list(missing_codes), np.nan)
    return out


def scale_score(df, items, reverse_items=(), scale_range=None,
                method="mean", min_valid_items=None):
    """문항 여러 개를 척도 점수 하나로 합친다.

    받는 것:
      df               문항이 들어 있는 DataFrame
      items            문항 컬럼 리스트
      reverse_items    그중 역채점할 문항
      scale_range      [min, max] — 역채점에 필요
      method           "mean" 또는 "sum" (sum 은 부분응답을 평균×문항수로 보정)
      min_valid_items  이보다 적게 응답했으면 점수를 NaN 으로 (부분응답 처리)
    돌려주는 것: 응답자별 척도 점수 Series
    왜: 역채점·부분응답 규칙을 한 곳에 모아 모든 척도가 같은 방식으로 계산되게 한다.
    """
    items = [c for c in items if c in df.columns]
    if not items:
        return pd.Series(np.nan, index=df.index)

    work = df[items].astype(float).copy()

    if reverse_items:
        if not scale_range:
            raise ValueError("역채점을 하려면 scale_range 가 필요하다 (코드북에서 확인)")
        lo, hi = scale_range
        for c in reverse_items:
            if c in work.columns:
                work[c] = reverse_code(work[c], lo, hi)

    n_valid = work.notna().sum(axis=1)
    if method == "mean":
        score = work.mean(axis=1)
    else:
        # sum 을 응답한 문항만 더하면 부분응답자 점수가 체계적으로 낮아진다
        # → 평균 × 전체 문항 수로 보정한다(prorated sum).
        #   전제: 한 척도의 문항들은 같은 응답 범위를 공유한다.
        score = work.mean(axis=1) * len(items)

    if min_valid_items:
        score = score.where(n_valid >= min_valid_items)
    else:
        score = score.where(n_valid > 0)
    return score


def cronbach_alpha(df, items):
    """Cronbach's alpha — 문항들이 한 구성개념을 재고 있는지의 내적일관성 지표.

    받는 것: DataFrame, 문항 리스트 (역채점은 **미리** 적용해 둘 것)
    돌려주는 것: float (문항 2개 미만이거나 계산 불가면 NaN)
    왜: 우리가 만든 척도 점수를 믿어도 되는지 학생이 직접 확인하게 한다.
        보통 .70 이상이면 수용 가능하다고 본다.
    주의: alpha 는 "타당도"가 아니라 "일관성"이다. 높다고 옳은 것을 재는 건 아니다.
    """
    items = [c for c in items if c in df.columns]
    x = df[items].astype(float).dropna()
    k = x.shape[1]
    if k < 2 or len(x) < 2:
        return float("nan")
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_var / total_var))


def item_total_correlations(df, items):
    """수정된 문항-전체 상관 (corrected item-total correlation).

    받는 것: DataFrame, 문항 리스트 (역채점을 적용하기 **전** 상태로 넣는다)
    돌려주는 것: {문항: 상관} Series — 문항이 2개 미만이면 빈 Series
    왜: "이 문항이 나머지 문항들과 같은 방향으로 움직이는가"를 본다.
        음수면 방향이 반대라는 뜻 → 역채점 문항을 놓쳤다는 신호다.
        0 근처면 방향 문제가 아니라 **다른 것을 재고 있다**는 신호다
        (역채점해도 안 좋아진다 — 이 둘을 구분하는 것이 핵심).
    계산: 각 문항 vs "그 문항을 뺀 나머지 문항 평균"의 피어슨 상관.
        자기 자신을 뺀 '나머지'와 비교해야("수정된") 상관이 부풀지 않는다.
    불변성: ① items 는 df 에 존재하는 수치형 ② 한 척도의 문항은 같은 응답
        범위를 공유(아니면 범위 큰 문항이 '나머지 평균'을 지배) ③ 결측은 이미
        NaN(숫자 결측코드는 apply_missing_codes 로 먼저 처리).
    주의: cronbach_alpha 와 같은 응답자 집합을 보도록 listwise 로 결측을 지운다.
    """
    items = [c for c in items if c in df.columns]
    if len(items) < 2:
        return pd.Series(dtype=float)
    x = df[items].astype(float).dropna()
    return pd.Series({c: x[c].corr(x.drop(columns=[c]).mean(axis=1)) for c in items})


def alpha_if_deleted(df, items):
    """문항을 하나씩 빼 보면서 alpha 가 어떻게 변하는지 본다.

    받는 것: DataFrame, 문항 리스트
    돌려주는 것: {뺀 문항: 남은 문항들의 alpha} Series
    왜: 전체 alpha 보다 **크게 높은** 값이 있으면 그 문항이 척도를 깎아먹고
        있다는 뜻이다. 단, 이것은 진단이지 지시가 아니다 — alpha 를 올리려고
        문항을 빼면 **척도가 달라져** 선행연구와 비교할 수 없게 된다.
    """
    items = [c for c in items if c in df.columns]
    if len(items) < 3:   # 하나 빼고도 2문항은 남아야 alpha 가 정의된다
        return pd.Series(dtype=float)
    return pd.Series({c: cronbach_alpha(df, [x for x in items if x != c])
                      for c in items})

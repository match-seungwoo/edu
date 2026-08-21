# AUC의 수학적 계산 원리 보고서

AUC(Area Under the ROC Curve)는 이진 분류 모델의 성능을 평가하는 대표적인 지표로, ROC 곡선 아래의 면적을 뜻합니다. AUC를 수학적으로 계산하는 방법은 크게 세 가지 관점으로 나눌 수 있습니다.

---

## 1. 사다리꼴 공식을 이용한 수치 적분법 (Integration Method via Trapezoidal Rule)
ROC 곡선은 여러 임계값(Cut-off/Threshold)에서의 **위양성률(FPR, False Positive Rate)**을 X축, **민감도(TPR, True Positive Rate)**를 Y축으로 하여 그려지는 꺾은선 그래프입니다. 이 곡선 아래의 면적을 구하기 위해 고등학교 수학에서 배우는 **사다리꼴 넓이 공식**을 이용합니다.

### 계산 공식
각 임계값별로 정렬된 $i$번째와 $i+1$번째 점에 대해, 밑변의 길이는 FPR의 변화량($FPR_{i+1} - FPR_i$)이 되고, 두 변의 높이는 각각 $TPR_i$와 $TPR_{i+1}$이 됩니다.

$$AUC = \sum_{i=1}^{N-1} (FPR_{i+1} - FPR_i) \times \frac{TPR_i + TPR_{i+1}}{2}$$

*   $FPR = 1 - \text{Specificity}$ (위양성률)
*   $TPR = \text{Sensitivity}$ (참양성률/민감도)

---

## 2. 윌콕슨-맨-위트니 검정 통계량 (Mann-Whitney U Test / Wilcoxon Rank-Sum)
확률적 관점에서 AUC는 **"무작위로 뽑은 양성(Event) 데이터의 예측 확률이 무작위로 뽑은 음성(Non-event) 데이터의 예측 확률보다 클 확률"**을 의미합니다. 즉, $P(X_{\text{postive}} > X_{\text{negative}})$를 구하는 것과 같으며, 이는 비모수 통계 검정인 맨-위트니 U 검정(Mann-Whitney U Test)과 완벽히 동일합니다.

### 계산 공식
$$AUC = \frac{U_1}{n_1 \times n_2}$$

여기서 $U_1$은 다음과 같이 정의됩니다.
$$U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}$$

*   $n_1$: 실제 양성(Event) 데이터의 개수
*   $n_2$: 실제 음성(Non-Event) 데이터의 개수
*   $R_1$: 전체 데이터(양성+음성)를 예측 확률 기준으로 순위를 매겼을 때, **양성 데이터들의 순위(Rank) 합**
*   $n_1 \times n_2$: 가능한 모든 양성-음성 쌍(Pair)의 총 개수

---

## 3. 일치쌍 비율 계산법 (Concordance and Tied Percent / C-Statistic)
위의 확률적 관점을 데이터를 직접 1:1로 비교하는 방식으로 직관화한 방법입니다. 실제 양성 집합과 음성 집합의 데카르트 곱(Cross Join)을 통해 모든 가능한 쌍(Pair)을 만듭니다.

### 쌍의 분류
1.  **일치쌍 (Concordant Pair)**: 양성 데이터의 예측 확률이 음성 데이터의 예측 확률보다 큰 경우 (정상 분류)
2.  **부일치쌍 (Discordant Pair)**: 양성 데이터의 예측 확률이 음성 데이터의 예측 확률보다 작은 경우 (역전)
3.  **동점쌍 (Tied Pair)**: 양성 데이터와 음성 데이터의 예측 확률이 같은 경우

### 계산 공식
$$AUC = \frac{\text{Concordant Pairs} + 0.5 \times \text{Tied Pairs}}{\text{Total Pairs}}$$

*   $\text{Total Pairs} = n_1 \times n_2$
*   이 방식은 계산량이 많은 편($O(n_1 \cdot n_2)$)이지만, 모델의 변별 능력을 가장 직관적으로 설명해 줍니다. 

---

### 요약 및 관계성
*   **기하학적 관점**: ROC 곡선을 그리고 사다리꼴 형태로 나누어 면적 합산 (적분)
*   **통계적 관점**: 순위(Rank) 기반의 Mann-Whitney U 통계량을 통한 계산
*   **확률적 관점**: 양성이 음성보다 높게 예측될 확률(Concordance 비율) 계산

이 세 가지 방법은 수학적으로 완전히 동일한 AUC 값을 산출해냅니다.

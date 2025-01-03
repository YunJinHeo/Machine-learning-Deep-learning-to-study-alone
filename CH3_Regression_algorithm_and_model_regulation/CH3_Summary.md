# 03-1 k-최근점 이웃 회귀
## k-최근접 이웃 회귀
 지도학습 알고리즘은 크게 분류와 회귀로 나뉜다. **분류**는 말 그대로 샘플을 몇 개의 클래스로 분류하는 것이고 **회귀**는 임의의 숫자를 예측하는 문제이다.
## 데이터 준비
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight)
    train_input = train_input.reshape(-1,1)
    test_input = test_input.reshape(-1,1)

.reshape() : 리스트를 원하는 배열 형태로 바꿀 수 있다.

## 결정계수($R^2$)
### KNeighborsRegressor
 가까운 샘플들의 타깃값의 평균으로 값을 예측한다.

    from sklearn.neighbors import KNeighborsRegressor
    knr = KNeighborsRegressor()
    knr.fit(train_input, train_target)
    knr.score(train_input, train_target)

knr.score은 결정계수($R^2$)을 출력한다.

$R^2 = 1 - 전차^2/편차^2$ 

*전차(residuals)는 타깃과 예측 값의 차이를 의미한다.

## 과대적합 vs 과소적합
 과대적합 : 훈련 세트에서는 점수가 좋은데 테스트 세트에서 점수가 굉장히 나쁜 경우

 과소적합 : 훈련 세트보다 테스트 세트의 점수가 더 높거나 두 점수가 모두 낮은 경우

    knr.n_neighbors = 3
    knr.fit(train_input, train_target)
    knr.score(train_input, train_target)

knr.n_neighbors로 KNeighborsRegressor가 참조하는 이웃의 갯수를 조정할 수 있다. 

k-최근접 이웃 회귀의 경우 n_neighbors를 조정하여 과대/과소 적합을 해결할 수 있다.

## Bias and Variance
![638715179085456495](https://github.com/user-attachments/assets/17c5d1c0-416e-4373-a2b5-503ce8ec3b86)

Bias : 예측값과 실제 정답간의 차이의 평균 -> 예측값이 정답과 얼마만큼 떨어져 있는지를 나타냄

Variance : 예측값의 분산 -> 다양한 데이터 셋에 대하여 예측값이 얼마나 변화할 수 있는지를 나타냄

과대적합 -> low Bias & high Variance

과소적합 -> high Bias & low Variance

**middle(low) Bias & middle(low) Variance를 지향해야함!**

# 03-2 선형 회귀

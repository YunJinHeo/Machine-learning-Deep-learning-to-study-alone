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

$R^2 = 1 - \frac{잔차^2의 합}{편차^2의 합} = 1- \frac{(타깃-예측)^2의 합}{(타깃-평균)^2의 합}$

*잔차(residuals)는 타깃과 예측 값의 차이를 의미한다.

식을 보면 알 수 있듯이 모델이 평균정도 예측하면 결정계수가 0이 나오고 평균 이상을 예측하면 1에 가까워지며 평균에도 못 미치면 음수가 나오게 된다.

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
## 선형 회귀
 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘이다.

 입력 값을 x 출력 값을 y 에러를 e 라고 하면, $y = ax + b + e $

 양변에 공분산을 취해서 계산하면 a = $cov(x,y)\over V(x)$

 양변에 평균을 취해서 계산하면 b = E(y) - $cov(x,y)\over V(x)$ $\times E(x)$

 위의 식을 이용해 a, b를 구하면 특성이 하나인 경우의 선형 회귀식을 얻을 수 있다. 
 
     from sklearn.linear_model import LinearRegression
     lr = LinearRegression()
     lr.fit(train_input, train_target)

LinearRegression() 클래스도 마찬가지로 fit(), score(), predict() 메서드가 있다.

lr.coef_, lr.intercept_에 lr이 찾은 기울기와 절편이 저장되어 있다.

## 다항 회귀
      train_poly = np.column_stack((train_input ** 2, train_input))
      test_poly = np.column_stack((test_input ** 2, test_input))

clumn_stack을 이용해서 input의 제곱과 input을 행에 갖는 2차식 list를 만들 수 있다.

    lr = LinearRegression()
    lr.fit(train_poly, train_target)

무게 = a * $길이^2$ + b * 길이 + c 의 그래프를 학습하며 이와 같은 식을 **다항식**이라고 부른다.

다항식을 사용한 선형 회귀를 **다항 회귀**라고 한다.

# 03-3 특성 공학과 규제
## 다중 회귀
 여러 개의 특성을 사용한 선형 회귀를 **다중 회귀**라고 부른다.

 **특성 공학** : 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업

## 데이터 준비
    import pandas as pd
    df = pd.read_csv('https://bit.ly/perch_csv_data')
    perch_full = df.to_numpy()

pandas의 dataframe과 to_numpy()를 이용하면 csv파일을 numpy array로 변환시킬 수 있다.

## 사이킷런의 변환기
 사이킷런에서 특성을 만들거나 전처리하기 위해 제공하는 클래스를 **변환기**라고 부른다.

 일관된 fit(), transform() 메서드를 제공한다.

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures()
    poly.fit([2,3])
    poly.transform([[2,3]])

    -> [[1,2,3,4,6,9]]

 PolynomialFeatures() : 입력받은 list의 제곱한 항, 서로 곱한 항, 1을 추가한다. 1은 절편을 위한 특성이다.

 include_bias = False 로 하면 절편(1)을 제거할 수 있다.

     poly = PolynomialFeatures(include_bias = False)
     poly.fit(train_input)
     train_poly = poly.transform(train_input)
     test_poly = poly.transform(test_input)  # 항상 훈련 세트를 기준으로 테스트 세트를 변환하는 습관을 들이자.
     train_poly.shape

     -> (42,9)
## 다중 회귀 모델 훈련하기
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train_poly, train_target)

PolynomialFeatures 클래스의 degree 매개변수를 사용하면 필요한 고차항의 최대 차수를 지정할 수 있다.

    poly = PolynomialFeatures(degree = 5)

degree를 증가시켜 특성의 개수를 늘리면 훈련 세트에 너무 과대적합 될 수 있다.

## 규제
 규제는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 하는 것을 말한다. 선형 회귀 모델의 경우 계수(기울기)이 크기를 작게 만드는 일을 말한다.

 선현 회귀 모델에 규제를 적용하기 전에는 먼저 정규화를 해야한다.

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scaled = ss.transform(train_poly)
    test_scaled = ss.transform(test_poly)

StandardScaler 클래스는 데이터를 표준화한다. 즉 평균을 빼고 표준편차로 나누어준다.

선형 회귀 모델에 규제를 추가한 모델을 **릿지Ridge** 와 **라쏘Lasso**라고 부른다.

Ridge는 계수를 제곱한 값을 기준으로 규제를 적용하고 Lasso는 계수의 절댓값을 기준으로 규제를 적용한다.

## Ridge regression(L2)
 최소제곱법을 이용한 선형회귀에서는 $residual^2$의 합의 최솟값을 구하지만, ridge regression은 $residual^2$ + $lambda$ * $slop^2$의 최솟값을 구한다.
 
 lambda * $slop^2$을 ridge regression penalty라고 하며 결과적으로 계수를 줄이는 역할을 한다.

 slope가 줄어든다는 것은 해당 변수에 대해 예측값이 덜 민감해진다는 것을 의미한다.

 -> Variance를 줄이는 대신 Bias가 증가한다.

     from sklearn.linear_model import Ridge
     ridge = Ridge()
     ridge.fit(train_scaled, train_target)

alpha 값이 크면 패널티가 더커지므로 조금 더 과소적합되도록 유도한다. 반대로 alpha 값이 작아지면 과대적합 될 가능성이 크다.

적절한 alpha 값을 찾는 방법은 alpha에 대한 $R^2$값의 그래프를 그려보는 것이다.
 ## Lesso regression(L1)
 lessp regression은 $residual^2$ + $lambda$ * $abs(slope)$의 최솟값을 구한다.

     from sklearn.linear_model import Lasso
     lasso = Lasso()
     lasso.fit(train_scaled, train_target)

 lesso와 ridge는 거의 비슷하게 행동한다. 하지만, ridge는 lambda가 커지면 계수를 0에 근사하도록 축소하는 반면 lasso는 계수를 완전히 0으로 축소시킨다는 차이점이 있다. 따라서 ridge의 경우 입력변수가 전반적으로 비슷한 수준으로 출력변수에 영향을 미치는 경우에 사용하고 lasso의 경우 출력변수에 미치는 입력변수의 영향력 편차가 큰 경우에 사용한다.


  

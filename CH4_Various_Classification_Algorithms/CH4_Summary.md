# 04-1 로지스틱 회귀
 로지스틱 회귀를 이용하면 선형 방정식을 훈련하여 예측 값이 가질 수 있는 클래스에 대한 각각의 확률을 구할 수 있다.

 이진 분류에서는 시그모이드 함수를 다중 분류에서는 소프트 맥스 함수를 이용하여 훈련된 선형 방정식의 출력 값을 0~1 사이의 확률 값으로 변환한다.
## k-최근접 이웃 분류기의 확률 예측
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier
    kn.fit(train_scaled, train_target)

타깃 데이터에 2개 이상의 클래스가 포함된 문제를 **다중 분류**라고 부른다.

타깃값은 순서가 자동으로 알파벳 순으로 매겨진다. classes_ 속성을 살펴보면 저장된 타깃값을 확인할 수 있다.
    
    print(kn_classes_)

    print(kn.predict(test_scaled[:5])
    print(kn.predict_proba(test_scaled[:5])

predict() : 예측 값을 반환한다.
predict_proba() : 클래스별 확률값을 반환한다.

## 로지스틱 회귀
 로지스틱 회귀는 분류 모델이다. 선형 회귀와 동일하게 선형 방정식을 학습한 뒤 학습한 값을 시그모이드 함수(sigmoid function)을 이용해 변형하여 0~1 사이의 값을 만든다.

 1. 두 target 값을 0, 1로 놓고 선형 회귀를 해서 z의 선형 방정식을 얻는다.
 2. 시그모이드 함수로 z값을 0~1 사이로 제한한다.

![image](https://github.com/user-attachments/assets/2515bf4c-6c7f-48a1-ab38-68e25332f20d)
 
 3. 0.5보다 크면 1 작으면 0으로 target을 예측한다.


## 로지스틱 회귀로 이진 분류 수행하기
    from sklearn.linear_model import LogisitcRegression
    lr = LogisticRegression()
    lr.fit(train_bream_smelt, target_bream_smelt)

## 로지스틱 회귀로 다중 분류 수행하기

![image](https://github.com/user-attachments/assets/6a5cd29a-f383-45af-b458-833aea52ee6d)


 LogisticRegression은 시그모이드 함수에서 입력값의 적절한 가중치를 찾기 위해 기본적으로 반복적인 알고리즘을 사용한다.(경사하강법) max_iter 변수에서 반복 횟수를 지정할 수 있으며 기본값은 100이다.

 LogisticREgression은 기본적으로 릿지 회귀(L2)와 같이 계수의 제곱을 규제한다. 매개변수 C로 규제를 제어할 수 있으며 alpha와 반대로 작을수록 규제가 커진다.

     lr = LogisticRegression(C=20, max_iter = 1000)
     lr.fit(train_scaled, train_target)

 다중 분류는 클래스마다 z값을 하나씩 계산한다. 각 클래스를 제외하고 나머지 target 값을 0으로 놓고 클래스의 갯수 만큼의 선형 회귀 식을 구한다.
 
 이진 로지스틱 회귀와 다르게 시그모이드 함수가 아닌 **소프트 맥스**함수를 사용하여 z값을 확률로 변환한다.

  ![image](https://github.com/user-attachments/assets/ea500c5a-f095-4c54-9a58-749d725d8887)

 지수함수를 쓰는 이유는 미분가능하도록 하기 위해서이며, 입력값 중 큰 값은 더 크게 작은 값은 더 작게하여 입력 벡터가 더 잘 구분되게 하기 위함도 있다.

 z값은 음수가 나올 수 있지만 지수함수의 특성상 $e^z$이 0보다 크게 유지되어 확률이 음수가 나오는 것도 방지할 수 있다.

# 04-2 확률적 경사 하강법
## 점진적인 학습
 앞서서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 훈련 방식을 **점진적 학습**이라고 부른다.

 대표적인 점진적 학습 알고리즘으로 **0확률적 경사 하강법(Stochastic Gradient Descent)** 가 있다.

 ### 확률적 경사 하강법
훈련세트에서 샘플을 하나씩 꺼내어 조금씩 경사를 따라 이동하는 것을 **확률적 경사 하강법**이라고 한다.

확률적 경사 하강법은 만족할만한 위치에 도달할 때까지 계속해서 훈련세트를 반복 학습하는데 훈련 세트를 한 번 모두 사용하는 과정을 **에포크(epoch)** 라고 부른다.

샘플을 하나씩만 훈련하면 Local Minima문제에 빠질 수도 있다. **Local Minima 문제**는 경사 하강법의 최종 값이 전체 함수의 최솟값이 아닌 지역의 극솟값을 갖게 되는 것을 의미한다.

한 번 경사로를 따라 이동하기 위해 전체 샘플을 사용하는 **배치 경사하강법batch gradient descent**를 이용하면 이 문제를 해결할 수 있다. 하지만, 데이터 양이 너무 많아지면 현실적으로 불가능한 일이기 때문에 이를 타협하여 몇 개의 샘플을 사용해 경사 하강법을 수행하는 **미니배치 경사 하강법minibatch gradient descent** 방식도 존재한다.


### 손실 함수
 **손실 함수loss function**는 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준이다. 즉, 값이 작을수록 예측 값과 실제 값이 비슷하다는 뜻이다.

 손실함수는 미분가능한 함수여야 한다.

### 로지스틱 손실 함수

 타깃이 1일때 -> $-log(예측 확률)$
 
 타깃이 0일때 -> $-log(1 - 예측 확률)$

 확률이 0에 가까워지면 무한대를 1에 가까워지면 0의 값을 갖게된다.


 다중 분류에서 사용하는 손실 함수는 **크로스엔트로피 손실 함수corss-entropy loss function**라고 부른다.

 회귀에서는 평균 제곱 오차를 손실 함수로 많이 사용한다.

## SGDClassifier

    from sklearn.linear_model import SGDClassifier
    sc = SGDClassifier(loss = 'log_loss', max_iter = 10)
    sc.fit(train_scaled, train_target)

SGDClassifier의 객체를 이용해 확률적 경사 하강법을 사용할 수 있다. loss 매개변수로 손실 함수의 종류를 지정하고 max_iter로 수행할 에포크의 횟수를 지정할 수 있다.

    sc.partial_fit(train_scaled, train_target)

partial_fit() : 모델을 이어서 훈련하고 싶을 때 사용할 수 있는 메서드로 호출할 때마다 1에포크씩 이어서 훈련한다. fit()을 사용하지 않고 partial_fit() 메서드만 사용하고 싶으면 classes 매개변수에 전체 클래스 레이블을 전달해 주어야 한다.

## 에포크와 과대/과소적합
 적은 에포크는 과소적합을 많은 에포크는 과대적합을 야기할 수 있다.

    import numpy as np
    sc = SGDClassifier(loss = 'log_loss')
    train_score = []
    test_score = []
    classes = np.unique(train_target)

    for _ in range(0,300) :
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

 에포크에 따른 정확도를 측정하여 적절한 반복 횟수를 찾을 수 있다.

 SGDClassfier의 loss 매개변수 기본값은 'hindge이다.힌지 손실은 서포트 벡터 머신이라 불리는 머신러닝 알고리즘을 위한 손실 함수이다.
 


 
 

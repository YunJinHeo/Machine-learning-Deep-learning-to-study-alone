# 05-1 결정트리
## 사용 데이터

    wine = pd.read_csv("https://bit.ly/wine_csv_data")
    wine.head()
    
![image](https://github.com/user-attachments/assets/65c277ea-3447-4e83-808e-b0304eb63884)

## 결정 트리
 **결정트리**는 로지스틱 회귀와 다르게 모델이 "이유를 설명하기 쉽다"는 특징을 갖고 있다.

 사이킷런이 제공하는 DecisionTreeClassifier 클래스를 이용해 훈련시킬 수 있다.

    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(train_scaled, train_target)

plot_tree() 함수를 이용하여 트리의 모양을 시각화 할 수 있다.

    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.figure(figsize = (10,7))
    plot_tree(dt)
    plt.show()

![image](https://github.com/user-attachments/assets/45bffcce-ff4d-4013-9776-216a808b1fdf)

 맨위의 노드를 루트 노드root node, 맨 아래 끈에 달린 노드를 리프 노드leaf node라고 한다.

    plt.figure(figsize=(10,7))
    plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alchol', 'sugar', 'pH'])
    plt.show()

![image](https://github.com/user-attachments/assets/9eeb5ebc-6096-4cba-bd65-d48c87640734)


 plot_tree() 함수의 매개변수

 max_depth : 출력할 트리의 깊이
 
 filled : True 값을 전달해 주면 어떤 클래스의 비율이 높아짐에 따라 노드의 색깔이 진해진다.
 
 feature_names : 특성의 이름 값을 받는다.

## 불순도
DecisionTreeClassifier 클래스의 criterion 매개변수를 통해 계산할 불순도를 지정할 수 있다. 기본값은 'gini'이다.

### 지니 불순도gini
 $지니불순도 = 1- (음성클래스 비율^2 + 양성 클래스 비율^2)$

 클래스의 비율이 정확히 5 대 5면 지니불순도는 0.5로 가장 높아지고 클래스가 한 쪽으로 쏠려있으면 지니불순도는 0으로 가장 낮아진다.

### 엔트로피 불순도entrophy
 $엔트로피 불순도 = -음성 클래프 비율 * log_2(음성클래스 비율) - 양성 클래스 비율 * log_2(양성 클래스 비율)$

### 정보 이득
부모노드와 자식노드 사이의 불순도 차이를 **정보이득information gain**이라고 한다.

결정트리는 정보 이득이 최대가 될 수 있도록 데이터를 나눈다.

## 가지치기
 트리의 깊이를 조정해서 과적합을 줄일 수 있으며, 이를 가지치기라고 한다.

 트리에서 각 특성의 중요도는 각 노드의 정보 이득과 점체 샘플에 대한 비율을 곱한 후 특성별로 더하여 계산하며, 결정 트리 모델의 feature_importances_ 속성에 저장되어 있다. 

    print(dt.feature_importances_)

![image](https://github.com/user-attachments/assets/a4755efc-dfed-4112-a028-205bec4792a4)

 2번째 특성인 sugar가 가장 유용한 특성 중 하나라는 것을 알 수 있다.

# 05-2 교차 검증과 그리드 서치
## 검증세트
 테스트 세트로 일반화 성능을 올바르게 예측하기 위해서는 가능한 한 테스트 세트를 사용하지 말아야한다. 이를 위해 **검증세트validation set**를 사용한다.

 샘플을 테스트 세트와 훈련세트로 나누고 다시 훈련세트에서 일부를 떼어내 검증 세트로 사용한다.

## 교차 검증
 검증세트를 만들면 훈련세트가 줄어든다는 단점이 생긴다. 그렇다고 검증세트를 너무 조금만 떼어내면 검증 점수가 불안정할 것이다.

 이를 보안하기 위해 **교차 검증corss validation**을 사용한다.

 교차검증은 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복한다. **k-폴드 교차 검증**은 훈련 세트를 k개로 분할하여 각 폴드를 검증세트로 사용한다. 

 주로 5-폴드 교차 검증이나 10-폴드 교차 검증을 많이 사용하는데 각각 훈련세트의 80%, 90%를 훈련에 사용할 수 있게된다.

 직접 검증 세트를 떼어낼 필요없이 **cross_validate()** 함수를 이용하면 교차 검증을 할 수 있다.

    from sklearn.model_selection import cross_validate()
    scores = cross_validate(dt, train_input, train_target)

cross_validate() 함수는 기본적으로 5-폴드 교차 검증을 수행한다.

cv : 사용할 분할기splitter와 shuffle 여부를 지정할 수 있다. 기본적으로 회귀 모델의 경우 KFold 분할기를 분류 모델일 경우 Stratified KFold 분할기를 사용한다.

    from sklearn.model_selection import StratifiedKFold
    splitter = StratifiedKFold(n_splits = 10, shuffle = True)
    scores = cross_validate(dt, train_input, train_target, cv = splitter)

## 하이퍼파라미터 튜닝
 머신러닝 모델이 학습하는 파라미터를 모델 파라미터, 사용자가 지정해야만 하는 파라미터를 **하이퍼파라미터**라고 한다.

 하이퍼파라미터를 찾는 과정은 다음과 같다.

 1. 탐색할 매개변수를 지정한다.
 2. 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾는다.(이 조합은 그리드 서치 객체에 저장된다.)
 3. 그리드 서치는 최상의 매개변수를 이용하여 전체 훈련 세트를 사용해 최종 모델을 훈련한다.(이 모델은 그리드 서치 객체에 저장된다.)

 사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술을 AutoML이라고 하며 이를 위해 시이킷런에서 제공하는 그리드 서치Grid Search를 사용할 수 있다.

    from sklearn.model_selection import GridSearchCV
    params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
    gs = GridSearchCV(DecisionTreeClassifier(), params, n_jobs = -1)
    gs.fit(train_input, train_target)

 기본적으로 GridSearchCV는 각 파라미터에 대해서 5-폴드 교차 검증을 수행한다. 즉, 5 * 5 = 25개의 모델을 훈련하게 된다.

 많은 모델을 훈련하기 때문에 n_jobs = -1 로 설정하여 모든 CPU를 병렬로 사용할 수 있게 지정해준다.

    dt = gs.best_estimator_
    dt.score(train_input, train_target)

![image](https://github.com/user-attachments/assets/0d1dd257-26f5-48a6-a167-f50696a48fda)


 최적의 하이퍼파라미터를 갖고있는 DecisionTreeClassifier()는 gs 객체의 best_estimator_ 속성에 저장되어 있다.

     gs.best_params_

![image](https://github.com/user-attachments/assets/9d823515-429c-4757-b384-723b8b6d960b)

 선택된 하이퍼파라미터의 값은 gs 객체의 best_params_ 속성에 저장되어 있다.

    gs.cv_results_['mean_test_score']

![image](https://github.com/user-attachments/assets/cc4e1474-7985-422a-bc74-d7fcbf548e83)

 각 매개변수에서 수행한 교차 검증의 평균 점수는 gs객체의 cv_results_ 속성의 'mean_test_score' 키에 저장되어 있다.

 dictionary를 이용하여 여러개의 파라미터에 대해서 그리드서치를 진행할 수 있다.
 
    params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001),
              'max_depth' : range(5, 20, 1),
              'min_samples_split' : range(2, 100, 10)
              }
    gs = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=-1)
    gs.fit(train_input, train_target)

## 랜덤 서치
 
 

 
 
 
 
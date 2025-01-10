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
 매개변수 값의 범위나 간격을 미리 정하기 어렵거나 너무 많은 매개변수 조건이 있어서 그리드 서치 수행 시간이 오래 걸릴 우려가 있다면 **랜덤 서치Random Search**를 사용할 수 있다.

 랜덤 서치에는 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달하게 된다.

    from scipy.stats import uniform, randint

 scipy의 stats 서브 패키지에 있는 uniform과 randint 클래스는 각각 실수 값, 정수 값에서 랜덤한 값을 추출한다.

    rgen = randint(0,10)
    rgen.rvs(10)
    ugen = uniform(0,1)
    ugen.rvs(10)

RandomizedSearchCV 클래스를 이용하여 랜덤서치를 수행할 수 있다.

    from sklearn.model_selection import RandomizedSearchCV
    params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
              'max_depth' : randint(20, 50),
              'min_samples_split' : randint(2,25),
              'min_samples_leaf' : randint(1,25)
              }
    gs = RandomizedSearchCV(DecisionTreeClassifier(), params, n_iter = 100, n_jobs=-1)
    gs.fit(train_input, train_target)
    dt = gs.best_estimator_
    dt.score(test_input, test_target)

n_iter 매개변수로 샘플링 횟수를 설정한다.

 모델의 적절한 하이퍼파라미터 수치를 찾고자 할 때는 수동으로 매개변수를 바꾸는 대신에 그리드 서치나 랜덤 서치를 사용하는 것이 편리하다.

# 05-3 트리의 앙상블
## 정형 데이터와 비정형 데이터
 CSV나 Database, Excel 과 같이 어떠한 구조로 되어 있는 데이터를 **정형 데이터structured data**라고 한다.

 반대로 글, 사진, 음악 등 어떠한 구조로 표현하기 힘든 데이터를 **비정형 데이터unstructured data**라고 한다.

 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘이 **앙상블 학습ensemble learning**이다.

## 랜덤 포레스트
 **랜덤 포레스트Random Forest**는 앙상블 학습의 대표주자로 안정적인 성능을 자랑한다. 앙상블 학습을 적용할 때 가장 먼저 랜덤 포레스트를 시도해 보는 것이 좋다.

 랜덤 포레스트는 이름에서도 알 수 있듯이 결정 트리를 랜덤하게 만든 뒤 각 트리의 예측을 사용해 최종 예측을 만든다.각 트리를 훈련하기 위해 데이터를 랜덤하게 만드는데 이때 부트스트랩 방식을 사용한다.

### 부트스트랩
 표본에서 샘플을 뽑을때 뽑았던 샘플을 다시 표본에 넣으면서 뽑는 방식. 예를들어 가방에서 100개의 샘플을 뽑는다면 1개를 뽑고 뽑았던 샘플을 적어둔 뒤 다시 가방에 넣는다. 그 후 마찬가지로 100개의 샘플을 뽑는다.

 RandomForestClassifier 클래스를 이용해 랜덤 포레스트를 실행할 수 있다. 기본적으로 전체 특성 개수의 제곱근 만큼의 특성을 선택한 후에 그 특성들을 바탕으로 최선의 분할을 찾는다.

 랜덤 트리와 달리 특성을 무작위로 선택하기 때문에 한가지 특성에 편향되지 않고 여러 특성에 고루 영향을 받는 모델을 만들 수 있다. 이는 훈련 세트에 과대적합되는 것을 막아주고 검증 세트와 테스트 세트에서 안정적인 성능을 얻을 수 있도록 도와준다.

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=-1)
    scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs=-1)

### OOB
 RandomForestClassifier의 OOB(out of bag) 샘플을 이용하면 자체적으로 모델을 평가하는 점수를 얻을 수 있다. OOB란 부트스트랩 샘플에 포함되지 않고 남는 샘플을 말한다. RandomForestClassifier의 oob_score 매개변수를 True로 지정하면 남은 샘플을 이용해 훈련한 결정 트리를 평가할 수 있다.

    rf = RandomForestClassifier(obb_score = True, n_jobs=-1)
    rf.fit(train_input, train_target)
    print(rf.obb_score_)

### 연산 횟수
 랜덤 포레스트에 검증을 시행할 때 총 트리 생성 횟수 = (랜덤 포레스트의 n_estimator) * (k 폴드) * (하이퍼 파라미터 튜닝 횟수)

## 엑스트라 트리
 전체적으로는 랜덤포레스트와 비슷하게 작용한다. 차이점은 부트스트랩 샘플을 사용하지않는 대신 노드를 분할할 때 가장 좋은 분할을 찾는 것이 아니라 무작위로 분할한다는 점이다.

 결정트리를 무작위로 분할하면 성능이 낮아질 수 있으나 많은 트리를 앙상블하기 때문에 과대적합을 막고 검증 세트의 점수를 높이는 효과가 있다.

 ExtraTreesClassifier 클래스를 이용해 엑스트라 트리를 실현할 수 있다.

    from sklearn.ensemble import ExtraTreesClassifier
    et = ExtraTreesClassifier(n_jobs = -1)

 엑스트라 트리가 랜덤 포레스트에 비해 무작위성이 더 크기 때문에 더 많은 결정 트리를 훈련해야 되긴 하지만 랜덤하게 노드를 분할하기 때문에 계산속도가 빠르다는 장점이 있다.

## 그레디언트 부스팅
 **그레디언트 부스팅gradient boosting**은 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법이다.

 GradientBoostingClassifier 클래스를 통해 이용 가능하며 기본적으로 깊이가 3인 결정 트리를 100개 사용한다.

 배치 경사 하강법을 사용하여 트리를 앙상블에 추가한다. 분류에서는 로지스틱 손실 함수를 사용하고 회귀에서는 평균 제곱 오차 함수를 사용한다.

 깊이가 얕은 트리로 샘플을 분류한 뒤 예측값과 타깃값의 잔차를 샘플링하여 새로운 트리를 만든다. 이 방식을 반복하여 예측과 타깃값의 잔차를 줄여나간다.

    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.2)

 n_estimators 매개변수로 결정 트리의 개수를 정하고 learning_rate 매개변수로 학습률을 설정할 수 있다.

 subsample 매개변수를 이용하여 경사 하강법에 사용한 샘플 비중을 조절할 수 있으며 이를 통해 미니배치 경사 하강법으로 그레디언트 부스팅을 진행할 수도 있다.

 GradientBoostingClassifier은 n_jobs 매개변수가 없는데 이는 그레디언트 부스팅의 특성상 연산을 순차적으로 진행할 수밖에 없기 때문이다. 이러한 이유로 랜덤 포레스트보다 성능이 좋은 대신에 속도가 느리다.

## 히스토그램 기반 그레디언트 부스팅
**히스토그램 기반 그레디언트 부스팅Histogram-based Gradient Boosting**은 정형 데이터를 다루는 머신러닝 알고리즘 중 가장 인기가 높은 알고리즘이다.

입력 특성을 256개의 구간으로 나누고 그레디언트 부스팅을 실행하게 되는데 이로인해 최적의 분할을 빠른 속도로 찾을 수 있다. 그레디언트 부스팅의 속도가 느리다는 단점을 보완한 모델이라고 할 수 있다.

256개의 구간 중에서 한 구간은 누락된 값을 이용해 사용하기 때문에 결측값이 있더라도 따로 전처리할 필요가 없다는 장점도 있다.

HistGradientBoostingClassifier 클래스를 통해 이용 가능하다.

    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier()

 과대적합을 잘 억제하면서 그레디언트 부스팅보다 조금 더 높은 성능을 제공한다. 

 히스토그램 기반 그레디언트 부스팅의 특성 중요도를 계산하기 위해서는 permutation_importance() 함수를 사용해야 한다. 이 함수는 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지 관찰하여 어떤 특성이 중요한지를 계산한다. n_repeats 매개변수로 랜덤하게 섞을 횟수를 지정할 수 있다.

     from sklearn.inspection import permutation_importance
     hgb.fit(train_input, train_target)
     result = permutation_importance(hgb, train_input, train_target, n_repeats=10 n_jobs=-1)

## 대표적인 그레디언트 부스팅 라이브러리
 XGBoost : 다양한 부스팅 알고리즘을 지원하며 tree_method 매개변수를 hist로 지정하면 히스토그램 기반의 그레디언트 부스팅을 사용할 수 있다.

 *xgboost 라이브러리는 sklearn 버전 1.6 미만에서만 사용할 수 있다.
 
    from xgboost import XGBClassifier
    xgb = XGBClassifier(tree_method = 'hist')

 LightGBM : 마이크로소프트에서 만든 그레디언트 부스팅 라이브러리

     from lightgbm import LGBMClassifier
     lgb = LGBMClassifier()
     
 
 
 

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

 
 
 
 

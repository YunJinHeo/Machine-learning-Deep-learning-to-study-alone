# 01-3 마켓과 머신러닝
## matplotlib의 pyplot 함수 사용법

    import matplotlib.pyplot as plt
    plt.scattter(bream_length, bream_weight)
    plt.xlabel('label')
    plt.ylabel('weight')
    plt.show()

## k-Nearest Neighbors
 이웃 분류 모델을 만드는 사이킷런 클래스이다. 가장 가까운 거리에 있는 이웃 중 다수를 출력한다.

### 매개변수

p : 1이면 맨해튼거리 2이면 유클리디안 거리를 이용해 거리를 잰다.

n_neighbors : 참조할 이웃의 개수를 지정한다.

### 사용법 

    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier(n_neighbors = 5)
    kn.fit(fish_data, fish_target)
    kn.score(fish_data, fish_target)
    kn.predict([[30,600]])

input data는 2차원 리스트이어야 한다.

kn.fit() : 훈련을 위한 method

kn.score() : 정확도 측정을 위한 method

kn.predict() : 학습을 바탕으로 데이터를 분류하기 위한 method




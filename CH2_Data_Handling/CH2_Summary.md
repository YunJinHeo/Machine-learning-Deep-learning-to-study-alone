# 02-1 훈련 세트와 테스트 세트
## 지도 학습과 비지도 학습
 지도 학습은 입력과 타깃으로 이루어진 훈련 데이터를 필요로 하는 학습이며 비지도학습은 타깃 없이 입력 데이터만을 사용한다.

 입력으로 사용되는 값을 특성이라고 한다.

## 훈련 세트와 테스트 세트
 평가에 사용하는 데이터를 테스트 세트, 훈련에 사용되는 데이터를 훈련 세트라고 한다.

 머신러닝의 평가를 위해서는 샘플의 일부를 테스트 세트로 사용해야 한다.

## 샘플링 편향
 훈련 세트와 테스트 세트가 골고루 섞이지 않고 한 쪽으로 치우치는 현상을 샘플링 편향이라고 한다.

## numpy array와 random
 넘파이는 파이썬의 배열 라이브러리로 고차원의 배열을 조작할 수 있는 간편한 도구들을 제공해 준다.

### 사용법
    import numpy as np
    input_arr = np.array(fish_data)
    target_arr = np.array(fish_target)
    print(input_arr.shape)

 np.array() : 파이썬 리스트를 numpy array로 변환한다.
 
 .shape : 해당 array의 행, 열의 수(샘플 수, 특성 수)를 출력한다.

     index = np.arange(49)
     np.random.shuffle(index)
     train_input = input_arr[index[:35]]
     train_target = target_arr[index[:35]]
     test_input = input_arr[index[35:]]
     test_target = target_arr[index[35:]]

 np.random.shuffle() : 주어진 배열을 무작위로 섞는다.

 np.random.shuffle()을 이용하여 샘플을 훈련 세트와 테스트 세트에 랜덤하게 배정할 수 있다.
 
# 02-2 데이터 전처리
## 넘파이로 데이터 준비하기
    import numpy as np
    fish_data = np.column_stack((fish_length, fish_weight))

np.column_stack() : 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결한다.

fit()은 2차원 리스트를 input으로 받기 때문에 np.column_stack을 이용하면 간편하게 input data를 만들 수 있다.

    fish_target = np.concatenate((np.ones(35), np.zeros(14))

np.concatenate() : 전달받은 리스트를 그대로 연결한다.

## 사이킷런으로 훈련 세트와 테스트 세트 나누기
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target)

sklearn.modle.selection의 train_test_split()을 이용하면 불편하게 np.shuffle을 사용할 필요없이 샘플을 훈련 세트와 테스트 세트로 나눌 수 있다.

기본적으로 25%를 테스트 세트로 떼어낸다. stratify 매개변수를 이용하면 target data와 같은 비율로 훈련 세트, 테스트 세트를 구성할 수 있다.

## 기준을 맞춰라
 input data로 사용하는 여러 특성의 스케일이 다르면 거리를 잴때 한 변수의 비중이 커질 수 있다. 따라서 특성값을 일정한 기준으로 맞추는 **데이터 전처리** 작업을 해주어야 한다.

### 표준점수
 가장 널리 사용하는 전처리 방법 중 하나로 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다.

 각 특성값에서 평균을 빼고 표준편차로 나누어 주면 된다.

    mean = np.mean(train_input, axis = 0)
    std = np.std(train_input, axis = 0)

np.mean np.std : 각각 입력한 리스트의 평균과 표준편차를 출력한다. axis를 0으로 설정하면 각 열의 통계 값을 1로 설정하면 각 행의 통계 값을 계산한다.

    train_scaled = (train_input - mean) / std

## 전처리 데이터로 모델 훈련하기
    kn.fit(train_scaled, train_target)
    test_scaled = (test_input - mean) / std
    kn.score(test_scaled, test_target)

kn.score()을 할 때 테스트 세트도 훈련 세트와 마찬가지로 훈련 세트의 표준편차, 평균을 이용해서 표준점수로 변환해 주어야 한다.
 

    


# 06-1 군집 알고리즘
## 타깃을 모르는 비지도 학습
 타깃이 없을 때 사용하는 머신러닝 알고리즘을 **비지도 학습unsupervised learning**이라고 한다.

## 데이터 준비
    import requests
    import numpy as np
    import matplotlib.pyplot as plt
    url = "https://bit.ly/fruits_300_data"
    response = requests.get(url)
    with open('fruits_300.npy', 'wb') as f:
        f.write(response.content)
    fruits = np.load('fruits_300.npy')

 fruits는 (300, 100, 100) 크기의 넘파이 배열로 100*100 이미지 300개를 담고있다.

     plt.imshow(fruits[0], cmap = 'gray')
     plt.show()

![image](https://github.com/user-attachments/assets/86408823-bdf7-4f99-9f49-8cbb746809b4)

 matplotlib의 imshow() 함수를 사용하면 넘파이 배열 이미지를 쉽게 그릴 수 있다. 값이 0에 가까울수록 검게 나타나고 높은 값은 밝게 표시된다.

 컴퓨터는 255에 가까운 값에 집중을 한다. 흰 바탕에 물체가 있으면 물체의 픽셀 값보다 흰 바탕의 값이 더 높아지게 되므로 바탕이 아닌 물체에 포커스를 맞추기 위해서 흑백 이미지를 반전시켜서 저장한다.

 cmap 매개변수를 'gray_r'로 지정하면 이를 다시 반전하여 우리 눈에 보기 좋게 출력을 받을 수 있다.

     plt.imshow(fruits[0], cmap = 'gray_r')
     plt.show()

![image](https://github.com/user-attachments/assets/b1e1a882-100a-4c6e-b3f8-d1ec65eb81ce)

     fig, axs = plt.subplots(1,2)
     axs[0].imshow(fruits[100], cmap = 'gray_r')
     axs[1].imshow(fruits[200], cmap = 'gray_r')
     plt.show()

![image](https://github.com/user-attachments/assets/54eec5e7-a295-4161-b1ca-fdce25fcc996)

 plt.subplots()를 이용하면 여러개의 그래프를 배열처럼 쌓을 수 있다. fig를 이용해 전체 그래프, axs를 통해 각각의 그래프의 서식을 지정할 수 있다.

## 픽셀값 분석하기
 편리한 분석을 위해 100*100 2차원 배열을 1차원 데이터로 변환한다.

    apple = fruits[0:100].reshape(-1, 100*100)
    pineapple = fruits[100:200].reshape(-1, 100*100)
    banana = fruits[200:300].reshape(-1, 100*100)

### 각 샘플이 갖는 픽셀값의 평균 비교하기

    plt.hist(np.mean(apple, axis=1), alpha = 0.8)
    plt.hist(np.mean(pineapple, axis=1), alpha = 0.8)
    plt.hist(np.mean(banana, axis=1), alpha = 0.8)
    plt.legend(['apple', 'pineapple', 'banana'])
    plt.show()

![image](https://github.com/user-attachments/assets/e3952413-9766-4638-b953-2aa0e2e1badf)

 pyplot의 hist() 함수를 이용해서 각 과일 별 픽셀 평균값의 히스토그램을 그릴 수 있다. alpha 매개변수를 이용해서 히스토그램의 투명도를 설정할 수 있다.

### 각 과일의 픽셀별 평균값을 비교하기

    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    axs[0].bar(range(10000), np.mean(apple, axis=0))
    axs[1].bar(range(10000), np.mean(pineapple, axis=0))
    axs[2].bar(range(10000), np.mean(banana, axis=0))
    plt.show()

![image](https://github.com/user-attachments/assets/04388179-7291-48e5-989f-29b0efd0eacd)

 각 과일에 따라 픽셀별 비중이 차이나는 것을 볼 수 있다.

 픽셀 평균값을 100*100 크기로 바꿔서 이미지를 출력하면 픽셀을 평균 낸 이미지를 출력할 수도 있다.

    apple_mean = np.mean(apple, axis=0).reshape(-1, 100, 100)
    pineapple_mean = np.mean(pineapple, axis=0).reshape(-1, 100, 100)
    banana_mean = np.mean(banana, axis=0).reshape(-1, 100, 100)
    fig, axs = plt.subplots(1, 3, figsize = (20,5))
    axs[0].imshow(apple_mean, cmap = 'gray_r')
    axs[1].imshow(pineapple_mean, cmap = 'gray_r')
    axs[2].imshow(banana_mean, cmap = 'gray_r')
    plt.show()

![image](https://github.com/user-attachments/assets/163266ea-630d-4642-9136-8c5814aab05e)

## 평균값과 가까운 사진 고르기
 np.abs() 함수를 이용하면 절대값을 계산할 수 있다.

 np.argsort() 함수는 작은 것에서 큰 순서대로 나열한 배열의 인덱스를 반환한다.

    abs_diff = np.abs(fruits - apple_mean)
    abs_mean = np.mean(abs_diff, axis(1,2))

    apple_index = np.argsort(abs_mean)[:100] #apple_mean과 오차가 가장 작은 100개의 샘플
    fig, axs = plt.subplots(10, 10, figsize = (10,10))
    for i in range(10) :
      for j in rannge(10) :
        axs[i, j].imshow(fruits[apple_index[10*i+j]], cmap = 'gray_r')
        axs[i, j].axis('off') # 그래프의 축을 출력하지 않는다.
    plt.show()

![image](https://github.com/user-attachments/assets/fd247f4c-17dd-4a5b-8697-54eac749a21e)

 이렇게 비슷한 샘플끼리 그룹으로 모으는 작업을 **군집clustering**이라고 한다.

 군집 알고리즘에서 만든 그룹을 **클러스터cluster**라고 한다.

# 06-2 k-평균
## k-평균 알고리즘 소개
 k-평균 알고리즘의 작동방식은 아래와 같다.

 1. 무작위로 k개의 클러스터 중심을 정한다.
 2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정한다.
 3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경한다.
 4. 클러스터 중심의 변화가 없을 때까지 2번으로 돌아가 반복한다.

## KMeans 클래스
 k-평균 알고리즘은 sklearn.cluster모듈 아래 KMeans 클래스에 구현되어 있다.
 
    from sklearn.cluster import KMeans
    fruits_2d = fruits.reshape(-1, 100*100)
    km = KMeans(n_cluster=3)
    km.fit(fruits_2d)

 KMeans() 클래스의 n_cluster를 이용하여 최초 정할 클러스터의 개수를 지정할 수 있다. 

 비지도 학습이기 때문에 fit() 메서드에 타깃 데이터를 입력하지 않아도 된다.

 군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장된다. n_cluster = 3으로 지정했기 때문에 배열의 값은 0, 1, 2중 하나다. 실제 레이블 0, 1, 2가 어떤 과일 사진을 주로 모았는지 알아보려면 직접 이미지를 출력하는 것이 최선의 방법이다.

### 편의를 위해 draw_fruits() 함수 만들기

    def draw_fruits(arr, ratio = 1) : 
      n = len(arr) # n은 샘플의 개수이다.
      rows = int(np.ceil(n/10)) # n/10을 올림해서 행의 개수를 결정한다.
      cols = n if rows < 2 else 10 # 샘플의 개수가 10개 이상일 때는 10 열을 만들고 그렇지 않으면 샘플 개수만큼 열을 만든다.
      fig, axs = plt.subplots(rows, cols, figsize = (cols*ratio, rows*ratio), squeeze = False) # ratio를 이용해 출력 사진의 크기를 조정할 수 있다.
      for i in range(rows) :
        for j in range(cols) :
          if i*10 + j < n :
            axs[i, j].imshow(arr[i*10 + j], cmap ='gray_r')
          axs[i, j].axis('off')
    
      plt.show()

 불리언 인덱싱을 이용하여 각 라벨별 이미지를 출력할 수 있다.
 
    draw_fruits(fruits[km.laebls_ ==0])
  
  ![image](https://github.com/user-attachments/assets/ac6d2e8f-c106-4578-bfff-763052909c80)

## 클러스터 중심
 KMeans 클래스가 최정적으로 찾은 클로스터 중심은 cluster_centers_ 속성에 저장되어 있다.

    draw_fruits(km.cluster_centers_.reshpae(-1, 100, 100), ratio=3)

![image](https://github.com/user-attachments/assets/d2f8d1d5-20bc-4195-abd3-73c1881e3361)


 KMeans 클래스는 훈련 데이터 샘풀에서 클러스터 중심까지 거리를 변환해 주는 transform() 메서드를 가지고 있다. 이는 transform() 메서드를 StandardScaler 클래스처럼 특성 값을 변환하는 도구로 사용할 수 있다는 의미이다.

    print(km.transform(fruits_2d[100:101]))

![image](https://github.com/user-attachments/assets/f22badd1-591e-429a-b5a8-837459cd731b)

 첫 번째 클러스터와의 거리가 가장 짧기 때문에 fruits[100]은 레이블 0에 속한 것으로 예측될 것이다.

    print(km.predict(fruits_2d[100:101]))

![image](https://github.com/user-attachments/assets/5bd6700c-d767-442d-9943-fdf5978d5a35)

 최적의 클러스터를 찾기까지 알고리즘이 반복한 횟수는 KMeans클래스의 n_iter_ 속성에 저장된다.

 클러스터 중심을 특성 공학처럼 사용하면 데이터셋을 저차원(이 경우에는 10000에서 3으로 줄인다.)으로 변환할 수 있다.

## 최적의 k 찾기
 적절한 k 값을 찾기 위한 완벽한 방법은 없으나 대표적으로 **엘보우elbow 방법**이 있다.

 클러스터 중심과 클러스터에 속한 샘플 사이의 거리의 제곱 합을 **이너셔inertia**라고 한다. 이너셔는 클러스터의 샘플이 얼마나 가깝게 있는지를 나타내는 값이다. 

 엘보우 방법은 클러스터 중심의 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법이다. 클러스터 개수를 증가시키면서 이너셔를 그래프로 그리면 이너셔 값이 감소하는 속도가 꺽이는 지점이 있다. 이지점 부터는 클러스터의 개수를 늘려도 클러스터에 샘플이 밀집된 정도가 크게 개선되지 않는다고 판단할 수 있으므로 그 점이 최적의 k 값이라고 예측해 볼 수 있다.

 KMeans 클래스는 자동으로 이너셔를 계산해서 inertia_ 속성으로 제공한다. 

    inertia = []
    for k in range(2,7) :
      km = KMeans(n_cluster = k, n_init ='auto') # n_init은 최초 클러스터 중심 설정 회수를 지정하는 매개변수이다. auto로 설정하면 데이터 크기, 클러스터의 개수에 따라 최적화된 n_init을 자동으로 설정한다.
      km.fit(fruits_2d)
      inertia.append(km.inertia_)
    plt.plot(range(2,7), intertia)
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.show()

![image](https://github.com/user-attachments/assets/ad08e2bd-f0ab-47f9-a58b-39e644dd35a6)

 k = 3에서 그래프의 기울기가 조금 바뀐 것을 확인할 수 있다. 명확하진 않지만 k=3이 최적의 k라고 유추해 볼 수 있다.
 
    

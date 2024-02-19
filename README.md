# 📌 Edge_Detection

## I.	Edge Detection 이란?
Edge는 값(밝기, color, …)이 차이가 나는 경계를 의미한다. 인간은 어떠한 이미지로부터든 이 edge를 쉽게 인지할 수 있다. 그러나 컴퓨터는 이 작업을 하기 위해 여러 과정을 거쳐야 한다. 그렇다면 이러한 edge detection은 왜 필요할까? 최근 여러 자동차 회사에서 선보이고 있는 자율 주행 서비스를 예로 들자면, 차선인식도 이런 edge detection에 일종으로 볼 수 있다. 뿐만 아니라 의료용이나 다양한 곳에 edge detection이 사용된다.


## II.	Sobel Filter (Gradient & Threshold)
이러한 edge를 detect하기 위해서는 픽셀의 값이 급격히 변하는 부분을 찾는다. 가장 간단한 방법으로는 1차 미분을 통해 그 값이 큰 부분을 찾는 것이다. 즉, 1차 미분의 변화율이 큰 곳을 찾는다. 그 후 threshold를 정해 그 안에 해당하는 부분만 edge라고 검출하는 방법이 있다.
![image](https://github.com/SYiee/Edge_Detection/assets/79504024/27f5941f-4bcd-4dbb-84ee-78c171e6cc60)

Sobel Filter를 사용해 Edge Detection을 수행하였고, Python으로 구현하였다.
첫번째로, 구현을 위해 필요한 라이 브러리 및 모듈을 import한다.

```
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
```

그 후, x와 y방향으로 적용할 sobel 필터 array를 만든다.
```
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]])
```
이렇게 만든 필터를 이미지에 각각 적용시키면 다음과 같다.

```
def edge_detection(img, filter_x, filter_y, threshold, show_img=True):

    img_shape = img.shape
    filter_size = filter_x.shape

    # Edge Detection을 저장할 array
    result_arr = tuple(np.array(img_shape)-np.array(filter_size)+1)

    # 결과 행렬 초기화
    x_result = np.zeros(result_arr)
    y_result = np.zeros(result_arr)

    # edge detetion 수행
    for h in range(0, result_arr[0]):
        for w in range(0, result_arr[1]):
            # 필터 적용
            tmp = img[h:h+filter_size[0],w:w+filter_size[1]]
            # x축 적용
            x_result[h,w] = np.abs(np.sum(tmp*filter_x))
            # y축 적용
            y_result[h,w] = np.abs(np.sum(tmp*filter_y))
```

![image](https://github.com/SYiee/Edge_Detection/assets/79504024/560531a0-b45e-4671-8403-390eaee18ccb)

두 방향의 이미지를 합친 후 threshold를 적용한 모습이다. Threshold를 높게 설정할수록 뒤의 열기구를 감지한 부분이 사라지는 것을 볼 수 있다. 그러나 threshold가 300일때와 500일 때를 비교해보면 500일때 여성의 어깨와 등쪽에 해당하는 부분도 사라지는 것을 볼 수 있다.
```
    # 두 결과를 합친다.       
    result = x_result + y_result
    thr_result = np.zeros(result_arr)
    thr_result[result>threshold] = 1
```
![image](https://github.com/SYiee/Edge_Detection/assets/79504024/378ca90a-b3ab-4f5c-b842-efdbeea1aa48)

해당 이미지의 뒷배경에 해당하는 열기구 근처를 보면 점처럼 인식되는 부분들이 있다. 이미지는 기본적으로 노이즈를 가지고 있고 미분을 하게 되면 더 심해진다. 그러나 Sobel Filter를 적용할 때는 미분을 하였음에도 노이즈를 제거하는 과정이 없었기 때문에 이에 대한 노이즈가 커져 원하는 결과가 나오지 않았을 것이라 예상할 수 있다. 따라서 더 매끄럽게 처리하기 위해 noise를 감소시키는 smoothing filter를 적용해 noise를 해결하고자 한다. 


## III.	Gaussian Smoothing Filter

Gaussian smoothing은 gaussian distribution을 기반으로 하며, 중심에 있는 픽셀에서부터 멀어질 수록 가중치가 작아지도록 설정이 된다. 즉, 구하고자하는 target pixel이 가장 큰 가중치를 갖고 중심에서 멀어질수록 픽셀이 적은 weight를 갖는다.

![image](https://github.com/SYiee/Edge_Detection/assets/79504024/733d66fb-c824-4b8d-8601-3e85a6b0a792)
이런 2차원 가우시안 커널을 구하는 함수를 다음과 같이 구현하였다

```
def gaussianKernel(size, sigma):
    # 중심으로부터 거리 배열 생성
    arr_dis = np.arange((size//2)*(-1), (size//2)+1)

    # array 초기화
    arr = np.zeros((size, size))

    # 중심에서부터의 거리를 제곱합으로 계산
    for x in range(size):
        for y in range(size):
            arr[x,y] = arr_dis[x]**2+arr_dis[y]**2

    # 커널의 값을 저장할 매트릭스 생성        
    kernel = np.zeros((size, size))
    
    # 가우시안 함수를 사용하여 가우시안 커널 생성
    for x in range(size):
        for y in range(size):
             kernel[x,y] = np.exp(-arr[x,y]/(2*sigma**2))

    # 커널의 합이 1이 되도록 정규화
    kernel /= kernel.sum()
    
    return kernel
```
해당 함수를 이용해 원본 이미지에 스무딩을 적용한 결과는 아래 사진과 같다.

![image](https://github.com/SYiee/Edge_Detection/assets/79504024/5d1f9cca-fb2f-484c-ba25-07038ee1c11c)


## IV.	Result
이전 단계에서 구현한 ‘gaussianKernel()’을 수행한 결과 이미지를 다시 Sobel filter를 적용하여 Edge Detection을 수행해보았다. 결과는 아래와 같다.


![image](https://github.com/SYiee/Edge_Detection/assets/79504024/c347dfac-aa62-49be-bfab-cbb0fffdd2b7)

기존에 Sobel Filter만 적용했을 때와 비교하면 열기구가 있는 쪽에 점처럼 인식된 부분이 제거되어 더 깔끔해진 결과를 볼 수 있다.


![image](https://github.com/SYiee/Edge_Detection/assets/79504024/1667bdfa-4afa-4aca-8255-2f5a142b77fa)

다른 이미지들을 적용한 결과는 다음과 같다.


![image](https://github.com/SYiee/Edge_Detection/assets/79504024/112e1e2e-759a-4598-b837-b0c56cea39c5)

![image](https://github.com/SYiee/Edge_Detection/assets/79504024/082924a4-bf6a-4bc6-937c-0250c8f5675c)


![image](https://github.com/SYiee/Edge_Detection/assets/79504024/9e34ecd8-9a71-415d-8049-4c1c63095711)


![image](https://github.com/SYiee/Edge_Detection/assets/79504024/47747150-707c-4961-9acb-8cf9bb49e797)


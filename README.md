# Edge_Detection

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



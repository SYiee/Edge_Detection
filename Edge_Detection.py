"""
Following is black and white Lena image.

1. Compare the Edge detection performances with Sobel, Robert, Prewitt, LOG operators:
2. Describe their characteristic differences based on your simulation results
"""

# 필요한 라이브러리 및 모듈 import
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

import cv2

# 이미지를 열고 흑백으로 변환
img = Image.open('Input/taking_picture.jpg').convert('L')
img = np.array(img)

# roberts_1 = np.array([[ 1, 0],
#                       [ 0,-1]])

# roberts_2 = np.array([[ 0, 1],
#                       [-1, 0]])
# prewitt_x = np.array([[-1, 0, 1],
#                       [-1, 0, 1],
#                       [-1, 0, 1]])

# prewitt_y = np.array([[ 1, 1, 1],
#                       [ 0, 0, 0],
#                       [-1,-1,-1]])

gaussian = np.array([[0.075, 0.124, 0.075],
                    [0.124, 0.204, 0.124],
                    [0.075, 0.124, 0.075]])


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[ 1, 2, 1],
                    [ 0, 0, 0],
                    [-1,-2,-1]])

LoG_3_1 = np.array([[ 0,-1, 0],
                    [-1, 4,-1],
                    [ 0,-1, 0]])

LoG_3_2 = np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]])

LoG_5 = np.array([[ 0, 0,-1, 0, 0],
                  [ 0,-1,-2,-1, 0],
                  [-1,-2,16,-2,-1],
                  [ 0,-1,-2,-1, 0],
                  [ 0, 0,-1, 0, 0]])

LoG_9 = np.array([[ 0, 1, 1,  2,  2,  2, 1, 1, 0],
                  [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                  [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                  [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                  [ 2, 5, 0,-24,-40,-24, 0, 5, 2],
                  [ 2, 5, 3,-12,-24,-12, 3, 5, 2],
                  [ 1, 4, 5,  3,  0,  3, 5, 4, 1],
                  [ 1, 2, 4,  5,  5,  5, 4, 2, 1],
                  [ 0, 1, 1,  2,  2,  2, 1, 1, 0]])


# edge detection 결과
def show(img, result1, result2, result, thr_result):

    plt.imshow(img, cmap='gray')
    plt.show()

    plt.imshow(result1, cmap='gray')
    plt.show()

    plt.imshow(result2, cmap='gray')
    plt.show()

    plt.imshow(result, cmap='gray')
    plt.show()

    plt.imshow(thr_result, cmap='gray')
    plt.show()


import numpy as np


def generate_gaussian_filter(sigma: Union[int, float], filter_shape: Union[list, tuple, None]):
    # 'sigma' is the standard deviation of the gaussian distribution

    # Extracting filter_shape dimensions with default values
    m, n = filter_shape if filter_shape is not None else (3, 3)
    m_half = m // 2
    n_half = n // 2

    # initializing the filter
    gaussian_filter = np.zeros((m, n), np.float32)

    # generating the filter
    for y in range(-m_half, m_half + 1):
        for x in range(-n_half, n_half + 1):
            normal = 1 / (2.0 * np.pi * sigma**2.0)
            exp_term = np.exp(-(x**2.0 + y**2.0) / (2.0 * sigma**2.0))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term
    return gaussian_filter



# edge detection function
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

    # 두 결과를 합친다.
    result = x_result + y_result
    thr_result = np.zeros(result_arr)
    thr_result[result>threshold] = 1

    # 결과 시각화
    if show_img:
        show(img, x_result, y_result, result, thr_result)

    return x_result, y_result, result, thr_result


# edge_detection(img, roberts_1, roberts_2, threshold=50)

#edge_detection(img, sobel_x, sobel_y, threshold=500)

#edge_detection(img, prewitt_x, prewitt_y, threshold=200)

#edge_detection(img, LoG_3_1, np.zeros_like(LoG_3_1), threshold=70)
#edge_detection(img, LoG_3_2, np.zeros_like(LoG_3_2), threshold=150)

#edge_detection(img, LoG_5, np.zeros_like(LoG_5), threshold=300)
#edge_detection(img, LoG_9, np.zeros_like(LoG_9), threshold=2000)
#edge_detection(img, gaussian, np.zeros_like(gaussian), threshold=100)



# sigma_value = 1.0
# filter_shape_value = (5, 5)

# # 가우시안 필터 생성
# gaussian_filter_result = generate_gaussian_filter(sigma_value, filter_shape_value)
# print("Output using print():")
# print(gaussian_filter_result)
# print()

def create_LoG_filter(sigma, filter_shape):
    # 필터의 중심 좌표 계산
    center_x = (filter_shape[1] - 1) / 2
    center_y = (filter_shape[0] - 1) / 2

    # LoG 필터 생성
    y, x = np.indices(filter_shape)
    x = x - center_x
    y = y - center_y

    # Gaussian 함수를 이용하여 가중치 계산
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Laplacian 필터 계산
    laplacian = (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
    
    # LoG 필터 = Gaussian * Laplacian
    log_filter = gaussian * laplacian

    # 필터 정규화
    log_filter = log_filter - np.mean(log_filter)
    log_filter = log_filter / np.sum(np.abs(log_filter))

    return log_filter

sigma_value = 1.0
filter_shape_value = (3, 3)
log_filter_example = create_LoG_filter(sigma_value, filter_shape_value)

print("LoG Filter:")
print(log_filter_example)



edge_detection(img, LoG_5, np.zeros_like(LoG_5), threshold=400)
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

###################################################################################

def gaussian_filter(filter_size, sigma):
    # get numpy array with identical values in both directions (axis)
    y, x = np.ogrid[-filter_size:filter_size+1, -filter_size:filter_size+1]
    
    # get 3 x 3 matrix based on x and y, then use equation shown to compute filter
    filter_ = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    # comment out if you want to see what is going on.
    print(f'x looks: {x}\n y looks: \n{y}\n')
    print('x**2 looks: {0}\n y**2 looks: \n{1}\n'.format(x**2, y**2))
    print('By the property of numpy array, x**2 + y**2 is: \n{}'.format(x**2 + y**2))
        
    return filter_


# double derivative kernels 
def laplacian_filter(kind='laplacian'):
    if kind == 'laplacian':
        filter_ = np.array([[0, 1, 0], 
                        [1, -4, 1], 
                        [0, 1, 0]])
    elif kind == 'diag_laplacian':
        filter_ = np.array([[1, 1, 1], 
                [1, -8, 1], 
                [1, 1, 1]])
        
    return filter_

def laplacian_edge_detection(image, kind='laplacian', thresh=20):
    # get filter
    filter_ = laplacian_filter(kind)
    
    # get result array
    filtered = np.zeros(image.shape)
    
    # preprocess image
    image = image_preprocess(image)
    
    # loop through every pixel 
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            temp_arr = np.array([])
            left, right = j - 1, j + 2
            upper, lower = i - 1, i + 2
            
            # get matrix for small segment of image
            for r in range(upper, lower):
                temp_arr = np.append(temp_arr, image[r][left:right])
            
            temp_arr = temp_arr.reshape(3, 3)
            
            # calculate gradient
            grad = np.sum(filter_ * temp_arr)
            
            if grad > thresh:
                filtered[i-1][j-1] = 255
            else:
                filtered[i-1][j-1] = 0
                
    return filtered

#laplacian_img = laplacian_edge_detection(image_mug, 'laplacian', 20)
#plot_image(laplacian_img, 'Laplacian edged image')

a = gaussian_filter(1,1)
print(a)

#edge_detection(img, LoG_5, np.zeros_like(LoG_5), threshold=400)
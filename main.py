import cv2
import numpy as np
from PIL import Image
import os

def smooth_shape(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3]

    # 이진화
    _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

    # 윤곽선 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 빈 마스크에 부드럽게 다시 그리기
    smooth_mask = np.zeros_like(alpha)
    for contour in contours:
        # epsilon 값이 클수록 더 부드러워짐 (0.002 ~ 0.01 조절)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.fillPoly(smooth_mask, [approx], 255)

    # 부드러운 마스크 적용
    result = img.copy()
    result[:, :, 3] = smooth_mask
    cv2.imwrite(output_path, result)

# 실행
smooth_shape('output/shapes_split/shape_29.png', 'output/shape_29_smooth.png')
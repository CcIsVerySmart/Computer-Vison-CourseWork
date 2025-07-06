import cv2
import numpy as np

def pseudo_fattal_dehaze(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 直方图均衡：增强亮度分布，模拟 albedo-shading 分离
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return enhanced

if __name__ == '__main__':
    img = cv2.imread('./image/15.png')
    result = pseudo_fattal_dehaze(img)
    cv2.imshow('Original', img)
    cv2.imshow('Pseudo Fattal Dehaze', result)
    cv2.imwrite('./image/Fattal_result.png', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

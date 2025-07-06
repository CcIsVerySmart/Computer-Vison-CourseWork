import cv2
import numpy as np

def model_based_dehaze(image):
    # 模拟一个参考地图：模糊图（表示雾）
    haze_map = cv2.GaussianBlur(image, (31, 31), 0)
    # 模拟“模型增强”——高频增强图像
    detail = cv2.addWeighted(image, 1.5, haze_map, -0.5, 0)
    # clip并修复溢出
    return np.clip(detail, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    img = cv2.imread('./image/15.png')
    enhanced = model_based_dehaze(img)
    cv2.imshow('Original', img)
    cv2.imshow('Kopf-inspired Dehaze', enhanced)
    cv2.imwrite('./image/Kopf_result.png', enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

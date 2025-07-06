import cv2
import numpy as np

def enhance_contrast(image, kernel_size=3, scale=1.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    gradient = cv2.Laplacian(blurred, cv2.CV_64F)
    gradient_abs = np.absolute(gradient)
    mask = (gradient_abs > np.percentile(gradient_abs, 90)).astype(np.uint8)
    high_contrast = cv2.addWeighted(image, 1 + scale, image, 0, -50)
    result = np.where(mask[..., np.newaxis] == 1, high_contrast, image)
    return result

if __name__ == '__main__':
    img = cv2.imread('./image/15.png')
    enhanced = enhance_contrast(img)
    cv2.imshow('Original', img)
    cv2.imshow('Contrast Enhanced (Tan)', enhanced)
    cv2.imwrite('./image/Tan_result.png', enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os
import cv2
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ==== 原始函数保持不变 ====
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res

# ==== 主函数（改进匹配） ====
def evaluate_dcp_on_dataset(hazy_dir, gt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hazy_list = sorted(os.listdir(hazy_dir))
    results = []

    for hazy_name in hazy_list:
        hazy_path = os.path.join(hazy_dir, hazy_name)
        prefix = hazy_name.split("_")[0]
        gt_name = f"{prefix}.png"
        gt_path = os.path.join(gt_dir, gt_name)

        if not os.path.exists(gt_path):
            print(f"⚠ GT 不存在: {gt_path}，跳过 {hazy_name}")
            continue

        hazy = cv2.imread(hazy_path).astype('float64') / 255
        gt = cv2.imread(gt_path).astype('float64') / 255

        if hazy.shape != gt.shape:
            gt = cv2.resize(gt, (hazy.shape[1], hazy.shape[0]))

        dark = DarkChannel(hazy, 15)
        A = AtmLight(hazy, dark)
        te = TransmissionEstimate(hazy, A, 15)


        #t = TransmissionRefine((hazy * 255).astype('uint8'), te)
        t=te;

        J = Recover(hazy, t, A, 0.1)

        # 计算 PSNR 和 SSIM
        psnr_val = peak_signal_noise_ratio(gt, J, data_range=1.0)
        ssim_val = structural_similarity(gt, J, data_range=1.0, channel_axis=-1)

        results.append((hazy_name, psnr_val, ssim_val))

        # 保存结果图像
        out_path = os.path.join(output_dir, f"dehazed_{prefix}_{psnr_val:.2f}_{ssim_val:.2f}.jpg")
        cv2.imwrite(out_path, (J * 255).astype('uint8'))

        print(f"{hazy_name} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")

    # 输出平均指标
    psnr_avg = np.mean([r[1] for r in results])
    ssim_avg = np.mean([r[2] for r in results])
    print(f"\n✅ Average PSNR: {psnr_avg:.2f}, Average SSIM: {ssim_avg:.4f}")
    return results

# ==== 执行入口 ====
if __name__ == '__main__':
    hazy_dir = r"D:\New_Desktop\SOTS\outdoor\hazy"
    gt_dir = r"D:\New_Desktop\SOTS\outdoor\gt"
    output_dir = r"D:\New_Desktop\SOTS\outdoor\results_dcp"

    evaluate_dcp_on_dataset(hazy_dir, gt_dir, output_dir)

import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class DehazeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像去雾工具")
        self.setGeometry(200, 100, 1200, 600)
        self.setStyleSheet("background-color: #f7f9fc;")

        self.original_label = QLabel("原图")
        self.result_label = QLabel("去雾图")
        self.original_label.setFixedSize(500, 400)
        self.result_label.setFixedSize(500, 400)
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.result_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("上传图片")
        self.load_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.load_btn.clicked.connect(self.load_image)

        self.save_btn = QPushButton("保存去雾图")
        self.save_btn.setStyleSheet("padding: 10px; font-size: 16px;")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)

        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.original_label)
        img_layout.addWidget(self.result_label)
        img_layout.addLayout(btn_layout)

        self.setLayout(img_layout)

        self.result = None  # 保存去雾结果

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            src = cv2.imread(file_path)
            self.display_image(src, self.original_label)

            result = self.dehaze(src)
            self.result = result
            self.display_image(result, self.result_label)
            self.save_btn.setEnabled(True)

    def save_image(self):
        if self.result is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG Image (*.png)")
            if file_path:
                cv2.imwrite(file_path, self.result)
                QMessageBox.information(self, "保存成功", "图像已保存。")

    def display_image(self, img, label):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    # ==== 以下为暗通道去雾核心算法 ====

    def DarkChannel(self, im, sz):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def AtmLight(self, im, dark):
        h, w = im.shape[:2]
        imsz = h * w
        numpx = max(math.floor(imsz / 1000), 1)
        darkvec = dark.reshape(imsz)
        imvec = im.reshape(imsz, 3)
        indices = darkvec.argsort()[-numpx:]
        atmsum = np.zeros([1, 3])
        for ind in indices:
            atmsum += imvec[ind]
        A = atmsum / numpx
        return A

    def TransmissionEstimate(self, im, A, sz):
        omega = 0.95
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        transmission = 1 - omega * self.DarkChannel(im3, sz)
        return transmission

    def Guidedfilter(self, im, p, r, eps):
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

    def TransmissionRefine(self, im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
        r = 60
        eps = 0.0001
        return self.Guidedfilter(gray, et, r, eps)

    def Recover(self, im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype)
        t = cv2.max(t, tx)
        for ind in range(3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
        return res

    def dehaze(self, src):
        I = src.astype('float64') / 255
        dark = self.DarkChannel(I, 15)
        A = self.AtmLight(I, dark)
        te = self.TransmissionEstimate(I, A, 15)
        t = self.TransmissionRefine(src, te)
        J = self.Recover(I, t, A, 0.1)
        return (np.clip(J * 255, 0, 255)).astype(np.uint8)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DehazeApp()
    window.show()
    sys.exit(app.exec_())

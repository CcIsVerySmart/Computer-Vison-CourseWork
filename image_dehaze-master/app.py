from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import math
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
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

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
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
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    r = 60
    eps = 0.0001
    return Guidedfilter(gray, et, r, eps)

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res

def dehaze(img):
    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine((img).astype('uint8'), te)
    J = Recover(I, t, A, 0.1)
    return (np.clip(J * 255, 0, 255)).astype(np.uint8)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            img = cv2.imread(upload_path)
            result = dehaze(img)
            result_filename = f"dehazed_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, result)

            return render_template('index.html', original=filename, result=result_filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

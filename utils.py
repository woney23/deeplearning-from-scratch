from PIL import Image
import numpy as np

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이로 저장된 이미지를 PIL용 객체로 변환
    pil_img.show()

def numerical_gradient(f,x):
    h=1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val)+h
        fxh1=f(x)

        x[idx] = tmp_val-h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)

        x[idx] = tmp_val
        it.iternext()

    return grad


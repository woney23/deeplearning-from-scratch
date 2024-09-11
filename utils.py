from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이로 저장된 이미지를 PIL용 객체로 변환
    pil_img.show()
import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.pardir)
from data.mnist import load_mnist
from PIL import Image


x=np.array([-1.0,1.0,2.0])
# print(x)

y=x>0
# print(y)

y=y.astype(np.int64) # bool => int 
                     # int64인지 32인지 명확하게 작성하기 
# print(y)

def step_function(x): # 계단 함수
    return np.array(x>0,dtype=np.int64)

def sigmoid(x): # 시그모이드 함수
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)
y1=step_function(x)
y2=sigmoid(x)
plt.plot(x,y1, linestyle="--", label="step")
plt.plot(x,y2,label="signoid")
plt.ylim(-0.1,1.1) # y축 범위
# plt.show()

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x


# 신경망 구현 2-3-2-2
X=np.array([1.0,0.5]) # (2)
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) # (2,3)
B1=np.array([0.1,0.2,0.3]) # (3)

A1=np.dot(X,W1)+B1
Z1=sigmoid(A1) # (3)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]]) #(3,2)
B2=np.array([0.1,0.2]) # (2)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2) # (2)

W3=np.array([[0.1,0.3],[0.2,0.4]]) # (2,2)
B3=np.array([0.1,0.2]) # (2)

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3) # (2)

print(Y)
print(Y.shape)

# 구현정리
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['B1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['B2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['B3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['B1'], network['B2'], network['B3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y= identity_function(a3)
    
    return y


network = init_network()
x=np.array([1.0,0.5])
y=forward(network, x)
print(y)

# 소프트맥스 구현 
def softmax(a):
    c=np.max(a) # 오버플로우 문제 해결

    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp

    return  y

# mnist
(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이로 저장된 이미지를 PIL용 객체로 변환
    pil_img.show()


img=x_train[0]
label=t_train[0]
print(label)
print(img.shape)
img=img.reshape(28,28) # 원래 이미지 크기로 변형
print(img.shape)
img_show(img)






# python chapter3.py
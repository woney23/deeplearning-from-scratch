import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.pardir)
from data.mnist import load_mnist
from PIL import Image
import pickle


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

# print(Y)
# print(Y.shape)

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
# print(y)

# 소프트맥스 구현 
def softmax(a):
    c=np.max(a) # 오버플로우 문제 해결

    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

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
# print(label)
# print(img.shape)
img=img.reshape(28,28) # 원래 이미지 크기로 변형
# print(img.shape)
# img_show(img)


# 신경망 구현 (추론을 수행하는)
# 해서 학습된 가중치 매개변수를 불러와서 추론함. 

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 데이터 전처리도 진행해줌. 
    return x_test, t_test

def init_network():
    with open("history/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1= np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    y= softmax(a3)

    return y

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1

print("Accuracy:" +str(float(accuracy_cnt)/len(x)))

# 배치 고려
x,t = get_data()
network = init_network()
accuracy_cnt = 0
batch_size = 100
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size] # batch_size 만큼 앞에서부터 자르기 
    y = predict(network,x_batch)
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy_batch:" +str(float(accuracy_cnt)/len(x)))


# python chapter3.py
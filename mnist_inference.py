import sys,os
sys.path.append(os.pardir)
from data.mnist import load_mnist
import pickle
import numpy as np
from activation_function import *

def get_data(): # 데이터 로드 
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    # 데이터 전처리도 진행해줌. 
    return x_test, t_test

def init_network(): # 모델 초기화
    with open("history/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x): # 모델 
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
batch_size = 100

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size] # batch_size 만큼 앞에서부터 자르기 
    y = predict(network,x_batch)
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("Accuracy_batch:" +str(float(accuracy_cnt)/len(x)))


# python mnist_inference.py
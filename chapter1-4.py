import numpy as np
import sys, os
sys.path.append(os.pardir)
from data.mnist import load_mnist
from activation_function import *
from tqdm import tqdm


def sum_squares_error(y,t):
    return 0.5*np.sum((y-t)**2)

# def cross_entropy_error(y,t): # single data
#     delta = 1e-7
#     return -np.sum(t*np.log(y+delta))

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label= True)

train_size = x_train.shape[0]
batch_size=10
batch_index = np.random.choice(train_size, batch_size)
x_batch= x_train[batch_index]
t_batch = t_train[batch_index]

def cross_entropy_error(y,t): # batch , one-hot-vector
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

def cross_entropy_error_label(y,t): # target is label
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t])+1e-7)/batch_size

def numerical_diff(f,x): # 수치미분
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

# def numerical_gradient(f,x): # 여러 지점에 대해 편미분 
#     """
#     f : 함수
#     x : numpy array
#     """
#     h=1e-4
#     grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

#     for idx in range(x.size):
#         tmp_val = x[idx]
        
#         # f(x+h) 계산 
#         x[idx] = tmp_val + h
#         fxh1= f(x)

#         # f(x-h) 계산
#         x[idx] = tmp_val - h
#         fxh2 = f(x)

#         grad[idx] = (fxh1-fxh2)/(2*h)
#         x[idx] = tmp_val # 값 복원
    
#     return grad

# 경사하강법
# def gradient_descent(f,init_x, lr=0.01, step_num=100):
#     x = init_x

#     for i in range(step_num):
#         grad=numerical_gradient(f,x)
#         x-=lr*grad

#     return x


# 2층 신경망 구현하기 

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

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1=np.dot(x,W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)

        return y
    
    def loss(self, x, t):
        y=self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1) # 정답레이블이 원-핫 벡터로 되어 있기 때문

        accuracy = np.sum(y==t)/float(x.shape[0])

        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    

# 미니배치학습 구현하기

(x_train,t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list =[]
train_acc_list = []
test_acc_list = []
total_loss = 0
iter_per_epoch = max(train_size/batch_size, 1)


# 하이퍼파라미터
iters_num=1000
train_size= x_train.shape[0]
batch_size=100
learning_rate=0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key]-=learning_rate*grad[key]
    
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    total_loss+=loss

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

acc = network.accuracy(x_test, t_test)
print("total_loss",total_loss/iters_num)
print("total_acc",acc)


# python chapter4.py
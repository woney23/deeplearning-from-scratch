# 오차역전파법을 적용한 신경망 구현
# chapter4 develop

import numpy as np
import sys, os
sys.path.append(os.pardir)
from data.mnist import load_mnist
from activation_function import *
from collections import OrderedDict
from utils import *
from layers import *
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()

# parser setting
parser.add_argument("--gradient_check", action="store_true")

# parser 선언
args = parser.parse_args()

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params={}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers=OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])

        self.lastLayer=SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x
    
    def loss(self, x,t):
        y=self.predict(x)

        return self.lastLayer.forward(y,t)
    
    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        if t.ndim!=1:
            t=np.argmax(t,axis=1)
        
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout=1
        dout=self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse() # 순서 역순
        for layer in layers:
            dout=layer.backward(dout)
        
        # 결과 저장
        grads={}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


if __name__ == "__main__":

    # 데이터 읽기 
    (x_train,t_train), (x_test,t_test)=load_mnist(normalize=True, one_hot_label=True)

    # model
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    if args.gradient_check:
        x_batch = x_train[:3]
        t_batch = t_train[:3]

        grad_numerical = network.numerical_gradient(x_batch, t_batch)
        grad_backprop = network.gradient(x_batch, t_batch)

        # 각 가중치의 차이의 절댓값을 구한 후 그 절댓값들의 평균을 낸다. 
        for key in grad_numerical.keys():
            diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
            print(f'{key}:{str(diff)}')
    
    iters_num=10000
    train_size=x_train.shape[0]
    batch_size = 100
    learning_rate=0.1

    train_loss_list =[]
    train_acc_list =[]
    test_acc_list =[]

    iter_per_epoch = max(train_size/batch_size, 1)

    # train
    for i in tqdm(range(iters_num)):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate*grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i%iter_per_epoch==0:
            train_acc=network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)




# python chapter1-5-3.py


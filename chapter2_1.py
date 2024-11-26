import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data import spiral
from layers import *
from optimizer import *

# class Sigmoid:
#     def __init__(self):
#         self.params, self.grads = [],[]
#         self.out=None

#     def forward(self, x):
#         out = 1/(1+np.exp(-x))
#         self.out = out 
#         return out

#     def backward(self, dout):
#         dx = dout * (1.0-self.out)*self.out
#         return dx

# class Affine:
#     def __init__(self, W, b):
#         self.params=[W,b]
#         self.grads=[np.zeros_like(W), np.zeros_like(b)]
#         self.x=None

#     def forward(self, x):
#         W, b = self.params
#         out = np.dot(x,W)+b
#         self.x = x
#         return out
    
#     def backward(self, dout):
#         W,b = self.params
#         dx = np.dot(dout, W.T)
#         dW = np.dot(self.x.T,dout)
#         db = np.sum(dout,axis=0)

#         self.grads[0][...] = dW
#         self.grads[1][...] = db

#         return dx

class TwoLayerNet:
    def __init__ (self, input_size, hidden_size, output_size):
        I,H,O = input_size, hidden_size, output_size

        #가중치와 편향 초기와 
        W1 = 0.01*np.random.randn(I,H)
        b1 = np.zeros(H)
        W2 = 0.01*np.random.randn(H,O)
        b2 = np.zeros(O)

        # 계층 생성
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]
        self.loss_layer = SoftmaxWithLoss()
        
        self.params, self.grads = [], []

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x): # predict은 loss layer를 통과할 필요가 없으니까 그 전까지만 forward 하도록 함. 
        for layer in self.layers:
            x=layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    



if __name__ == "__main__":

    # 하이퍼파라미터 설정
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0
    
    # read data & model & optimizer
    x,t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # learning tool
    data_size = len(x) # 300
    max_iters = data_size//batch_size # 10
    total_loss = 0 
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # 데이터 섞기
        idx = np.random.permutation(data_size)
        x=x[idx]
        t=t[idx]
        
        for iters in range(max_iters):
            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_t = t[iters*batch_size:(iters+1)*batch_size]

            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            
            total_loss +=loss
            loss_count+=1

            if (iters+1) % 10 ==0:
                avg_loss = total_loss/loss_count
                print(f'에폭 {epoch+1} | 반복 {iters+1} / {max_iters} | 손실 {avg_loss}')
                loss_list.append(avg_loss)
                total_loss, loss_count = 0,0
    

# 학습 결과 플롯
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iters (x10)')
plt.ylabel('loss')
plt.show()

# 경계 영역 플롯
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 데이터점 플롯
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()


# python chapter2_1.py
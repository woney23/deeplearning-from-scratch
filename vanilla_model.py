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
from optimizer import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

# parser setting
parser.add_argument("--optimizers", action="store_true")

# parser 선언
args = parser.parse_args()

def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # 가중치 초기화
        self.params={}
        self.__init_weight(weight_init_std)
        # self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        # self.params['b1'] = np.zeros(hidden_size)
        # self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        # self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers=OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        # self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'],self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'],self.params['b4'])

        self.lastLayer=SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size]+[self.hidden_size]*3 + [self.output_size]
        for idx in range(1, len(all_size_list)):
            
            if str(weight_init_std).lower() in ('relu','he'):
                scale=np.sqrt(2.0/all_size_list[idx-1])
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0/all_size_list[idx-1])
            else:
                scale = weight_init_std
            
            self.params['W'+str(idx)] = scale * np.random.randn(all_size_list[idx-1],all_size_list[idx])
            self.params['b'+str(idx)] = np.zeros(all_size_list[idx])


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
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db

        return grads
    

if __name__ == "__main__":

    # 데이터 읽기 
    (x_train,t_train), (x_test,t_test)=load_mnist(normalize=True, one_hot_label=True)

    iters_num=10000
    train_size=x_train.shape[0]
    batch_size = 100
    learning_rate=0.1

    # optimizer 
    optimizer = Adam()
    # optimizer = {}
    # optimizer['SGD'] = SGD()
    # optimizer['Momentum'] = Momentum()
    # optimizer['AdaGrad'] = AdaGrad()
    # optimizer['Adam'] = Adam()

    # init
    weight_init_types = {'std=0.01':0.01, 'Xavier':'sigmoid','He':'relu'}

    # model
    # network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    networks = {}
    train_loss = {}

    for key, weight_type in weight_init_types.items():
        print(key, weight_type)
        networks[key] = TwoLayerNet(input_size=784, hidden_size=100, 
                                    output_size=10, weight_init_std=weight_type)
        train_loss[key] = []


    # train_loss_list =[]
    # train_acc_list =[]
    # test_acc_list =[]

    iter_per_epoch = max(train_size/batch_size, 1)

    # train
    for i in tqdm(range(iters_num)):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in weight_init_types.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizer.update(networks[key].params, grads)
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        # optimizer - update
        # for key in ('W1','b1','W2','b2'):
        #     network.params[key] -= learning_rate*grad[key]
        
        

        if i%1000==0:
            print(f"==== iteration: {i} ====")
            for key in weight_init_types.keys():
                loss=networks[key].loss(x_batch, t_batch)
                print(f'{key} Loss :{loss}')
            # train_acc=network.accuracy(x_train, t_train)
            # test_acc = network.accuracy(x_test, t_test)
            # train_acc_list.append(train_acc)
            # test_acc_list.append(test_acc)
            # print(f'Train: {train_acc}, Test: {test_acc}')
    
    # train_acc=network.accuracy(x_train, t_train)
    # test_acc = network.accuracy(x_test, t_test)
    # print(f'Train: {train_acc}, Test: {test_acc}')

    print(f"===== Test - accuracy =====")
    for key in weight_init_types.keys():
        acc=networks[key].accuracy(x_test, t_test)
        print(f'{key}:{acc}')

    # graph
    markers = {"std=0.01": "o", "Xavier": "x", "He": "s"}
    x = np.arange(iters_num)
    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend()
    plt.show()




# python vanilla_model.py


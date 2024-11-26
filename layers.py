import numpy as np
from activation_function import *
from utils import *

class Relu:
    def __init__(self):
        self.mask=None
    
    def forward(self, x):
        self.mask=(x<=0) # mask는 인스턴스 변수 x값이 0이하는 원소 값 인덱스가 True
        out=x.copy() 
        # 원본 데이터를 보호하고, 다른 연산에 영향을 미치지 않기 위한 안전 장치 
        # 이로 인해 순전파 과정에서 원본 데이터가 변경되지 않고, 
        # 나중에 다른 계산에서 동일한 데이터가 필요할 때 문제가 발생하지 않음.

        out[self.mask]=0

        return out
    
    def backward(self, dout):
        dout[self.mask] =0
        dx=dout
        
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [],[]
        self.out=None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out 
        return out

    def backward(self, dout):
        dx = dout * (1.0-self.out)*self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W), np.zeros_like(b)]
        self.x=None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x,W)+b
        self.x = x
        return out
    
    def backward(self, dout):
        W,b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [],[]
        self.y=None # 결과
        self.t=None # 정답

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            self.t = self.t.argmax(axis=1)

        loss=cross_entropy_error(self.y, self.t)
        
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx*=dout
        dx = dx / batch_size
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask=None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 이때 *x.shape는 x.shape 튜플의 각 차원을 풀어서 전달하는 의미
            # self.mask 는 true/false로 구성된 배열이 생김. 
            return x*self.mask
        else:
            return x*(1.0-self.dropout_ratio)
    
    def backward(self, dout):
        return dout*self.mask

class BatchNormalization:

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # 시험할때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim!=2:
            N,C,H,W = x.shape
            x=x.reshape(N,-1)

        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N,D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x-mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc/std

            self.batch_size = x.shape[0]
            self.xc = xc 
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) *var

        else:
            xc = x-self.running_mean
            xn = xc / ((np.sqrt(self.running_var+10e-7)))
        
        out = self.gamma*xn+self.beta
        return out
    
    def backward(self, dout):
        if dout.ndim !=2:
            N,C,H,W = dout.shape
            dout = dout.reshape(N,-1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx
    
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn*dout,axis=0)
        dxn = self.gamma * dout
        dxc = dxn/self.std
        dstd = -np.sum((dxn*self.xc)/(self.std*self.std),axis=0)
        dvar = 0.5*dstd/self.std
        dxc += (2.0/self.batch_size)*self.xc*dvar
        dmu = np.sum(dxc,axis=0)
        dx= dxc-dmu/self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
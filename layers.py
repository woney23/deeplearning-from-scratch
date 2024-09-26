import numpy as np
from activation_function import *

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
        self.out=None
    
    def forward(self, x):
        out=1/(1+np.exp(-x))
        self.out=out

        return out
    
    def backward(self, dout):
        dx=dout*self.out*(1.0-self.out)
        return dx
    
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.dW=None
        self.db=None

    def forward(self,x):
        self.x=x
        out = np.dot(x,self.W)+self.b
        
        return out
    
    def backward(self, dout):
        dx=np.dot(dout, self.W.T)
        self.dW=np.dot(self.x.T, dout)
        self.db=np.sum(dout, axis=0)

        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None # 손실
        self.y=None # 결과
        self.t=None # 정답

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        self.loss=cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx=(self.y-self.t)/batch_size

        return dx
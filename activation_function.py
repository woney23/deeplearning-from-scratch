import numpy as np

def step_function(x):
    return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

def softmax_past(a):
    c=np.max(a) # 오버플로우 문제 해결

    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return  y

def sigmoid_grad(x):
    return (1.0 - sigmoid(x))*sigmoid(x)

def relu_grad(x):
    grad= np.zeros(x)
    grad[x>=0]=1
    return grad

def softmax(x):
    if x.ndim == 2:
        x=x-x.max(axis=1,keepdims=True)
        x=np.exp(x)
        x/=x.sum(axis=1, keepdims=True)
    elif x.ndim==1:
        x=x-np.max(x)
        x=np.exp(x)/np.sum(np.exp(x))

    #     x=x.T
    #     x=x-np.max(x,axis=0)
    #     y=np.exp(x)/np.sum(np.exp(x),axis=0)
    #     return y.T
    
    # x=x-np.max(x) # 오버플로우 대책
    # np.exp(x)/np.sum(np.exp(x))
    return x

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

def softmax_loss(X,t):
    y=softmax(X)
    return cross_entropy_error(y,t)



            
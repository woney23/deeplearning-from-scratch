import numpy as np

def step_function(x):
    return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def relu(x):
    return np.maximum(0,x)

def softmax(a):
    c=np.max(a) # 오버플로우 문제 해결

    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return  y


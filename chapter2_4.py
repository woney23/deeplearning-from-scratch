import numpy as np


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0 # dW 형상은 유지한 채, 그 원소들을 0으로 덮어쓴다. 

        # dW[self.idx] = dout #실은 나쁜 예
        
        # idx에서 중복을 처리할 수 있도록 수정
        # for i, word_id in enumerate(self.idx):
        #     dW[word_id]+=dout[i]
        
        np.add.at(dW, self.idx, dout) # 일반적으로 for문보다는 numpy 메서드를 사용하는 것이 효율이 높음.
        return None
        # 이렇게 하지 않고, 갱신하려는 행번호와 그 기울기를 따로 저장해두면, 이 정보로부터 가중치 W의 특정 행만 갱신할 수 있다.

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h,axis=1)

        self.cache = (h,target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout*h
        self.embed.backward(dtarget_W)
        dh = dout*target_W
        return dh
    



    

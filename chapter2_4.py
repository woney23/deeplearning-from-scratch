import numpy as np
import collections
import sys
sys.path.append('..')
from np import * 
from layers import *
import config
config.GPU = True
import pickle
from Trainer import Trainer
from optimizer import *
from utils import *
from data import ptb


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
    

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size +1)]
        self.params, self.grads = [],[]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        #긍정적인 예 순전파
        score = self.embed_dot_layers[0].forward(target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss=self.loss_layers[0].forward(score, correct_label)

        # 부정적인 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss+=self.loss_layers[1+i].forward(score, negative_label)
        return loss
    
    def backward(self,dout):
        dh=0
        for l0,l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh+= l1.backward(dscore)
            
        return dh



class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V,H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(V,H).astype('f')

        # 계층 생성
        self.in_layers=[]
        # 여기에 embedding 계층 넣어줌. 
        for i in range(2*window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다. 
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [],[]

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        #단어의 분산 표현 저장
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h=0
        for i, layer in enumerate(self.in_layers):
            h+=layer.forward(contexts[:,i])
        
        h*=1/len(self.in_layers)
        loss=self.ns_loss.forward(h,target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *=1/len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        
        return None
    

if __name__ =="__main__":
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # data read
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    if config.GPU:
        contexts, target = to_gpu(contexts), to_gpu(target)
    
    #model
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # train
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    # 나중에 사용할 수있도록 필요한 데이터 저장
    word_vecs = model.word_vecs
    if config.GPU:
        word_vecs = to_cpu(word_vecs)
    params = {}
    params['word_vecs']= word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params_pkl'

    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)

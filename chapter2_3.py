import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from layers import *
from utils import *
from Trainer import *
from optimizer import *


# #cbow model inference

# # 샘플 맥락 데이터
# c0 = np.array([[1,0,0,0,0,0,0]])
# c1 = np.array([[0,0,1,0,0,0,0]])

# # 가중치 초기화
# W_in = np.random.randn(7,3)
# W_out = np.random.randn(3,7)

# # 계층 생성
# in_layer0 = MatMul(W_in)
# in_layer1 = MatMul(W_in)
# out_layer = MatMul(W_out)

# # 순전파
# h0 = in_layer0.forward(c0)
# h1 = in_layer1.forward(c1)
# h = 0.5*(h0+h1)
# s=out_layer.forward(h)

# # print(s)


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None



if __name__ == "__main__":

    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size)
    # import IPython; IPython.embed(colors='Linux'); exit(1)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    
    trainer.fit(contexts, target, max_epoch, batch_size)

    trainer.plot()

    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])



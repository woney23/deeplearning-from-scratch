import numpy as np
import matplotlib.pyplot as plt
from data import ptb
from sklearn.utils.extmath import randomized_svd

text = "You say goodbye and I say hello."

text = text.lower() # 소문자 변환
text = text.replace('.',' .')
words = text.split(' ') # 공백을 기준으로 분할

word_to_id = {} # 단어 -> ID
id_to_word = {} # ID -> 단어

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
# print(corpus)

def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id]=word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 동시발생행렬

def create_to_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx-i
            right_idx = idx+i

            if left_idx >=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] +=1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] +=1

    return co_matrix

def cos_similarity(x,y, eps = 1e-8):
    nx = x/(np.sqrt(np.sum(x**2))+eps)
    ny = y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx, ny)


# 유사 단어 랭킹 표시
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어를 꺼낸다.
    if query not in word_to_id:
        print("%s를 찾을 수 없습니다."%query)
        return
    print("\n[query] "+query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 코사인 유사도를 기준으로 내림차순으로 top5만 출력 
    count = 0 
    for i in (-1*similarity).argsort():
        if id_to_word[i] ==query:
            continue
        print(f'{id_to_word[i]} : {similarity[i]}')

        count+=1
        if count>=top:
            return 

def ppmi(C, verbose=False, eps=1e-8):
    M=np.zeros_like(C,dtype=np.float32)
    N=np.sum(C)
    S=np.sum(C, axis=0)
    total=C.shape[0]*C.shape[1]
    cnt=0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[j]*S[i])+eps)
            M[i,j]=max(0, pmi)

            if verbose:
                cnt+=1
                if cnt % (total//100) ==0:
                    print("%.1f%% 완료"% (100*cnt/total))

    return M





if __name__ == "__main__":

    # text = "You say goodbye and I say hello."
    # corpus, word_to_id, id_to_word = preprocess(text)
    # vocab_size = len(word_to_id)
    # C=create_to_matrix(corpus, vocab_size)
    # c0 = C[word_to_id['you']]
    # c1 = C[word_to_id['i']]
    # print(cos_similarity(c0,c1))
    # print(C)
    # print("-"*50)
    # W=ppmi(C)
    # print(W)

    # U,S,V = np.linalg.svd(W)

    # print(U)
    
    # for word, word_id in word_to_id.items():
    #     plt.annotate(word, (U[word_id,0],U[word_id,1]))

    # plt.scatter(U[:,0],U[:,1],alpha=0.5)
    # plt.show()
    
    # 시각화 결과물이 책과 다름

    window_size = 2 
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)

    C = create_to_matrix(corpus, vocab_size, window_size)
    W=ppmi(C, verbose=True)

    U,S,V = randomized_svd(W, n_components=wordvec_size, n_iter=5,random_state=None)

    word_vecs = U[:,:wordvec_size]
    querys = ['you','year','car','toyota']

    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    
    


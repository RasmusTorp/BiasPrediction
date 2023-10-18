import numpy as np
from gensim.models.doc2vec import Doc2Vec
import tqdm as tqdm
import matplotlib.pyplot as plt

# doc2vec = np.load('doc2vec_model.model',allow_pickle=True)
# vectors = np.load("doc2vec_model.model.wv.vectors.npy", allow_pickle=True)    
# syn1neg = np.load("doc2vec_model.model.syn1neg.npy", allow_pickle=True)

model = Doc2Vec.load("doc2vec_model.model")

embedding = model.infer_vector("The quick brown fox jumps over the lazy dog".split())
print("done")

articles = np.load("data/articles.npy")

article_len = [len(article) for article in articles]
sources = np.load("data/sources.npy")

# plt.boxplot(article_len)
# plt.show()
too_long_articles = np.array(article_len) > 4000


print(sum(np.array(article_len) < 4000))
print(sum(np.array(article_len) < 16000))


labels = np.load("data/labels.npy")

# article_embeddings = [model.infer_vector(article.split()) for article in (articles)]

article_embeddings = [None for _ in range(len(articles))]

for i, article in tqdm.tqdm(enumerate(articles)):
    article_embeddings[i] = model.infer_vector(article.split())

np.save("data/article_embeddings.npy", np.array(article_embeddings))
print("done")


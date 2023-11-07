
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
import pickle
import numpy as np

def embed_data_by_paragraphs(data, save_name=None):
    model = Doc2Vec.load("doc2vec_paragraphs.model")
    data_embeddings = []
    for article in tqdm(data,total=len(data)):
        embeddings = []

        for paragraph in article:
            embeddings.append(model.infer_vector(paragraph.split()))

        embeddings = np.array(embeddings).mean(axis=0)
        data_embeddings.append(embeddings)
    
    if save_name:
        np.save(f"data/{save_name}.npy", np.array(data_embeddings))

    return np.array(data_embeddings)


if __name__ == "__main__":
    with open('data/X_train_by_paragraphs.pkl', 'rb') as f:
        X_train = pickle.load(f)

    with open('data/X_test_by_paragraphs.pkl', 'rb') as f:
        X_test = pickle.load(f)

    X_train_embeddings = embed_data_by_paragraphs(X_train, save_name="X_train_embeddings_by_paragraphs")
    X_test_embeddings = embed_data_by_paragraphs(X_test, save_name="X_test_embeddings_by_paragraphs")





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
import pickle


if __name__ == '__main__':
    # titles = np.load('data/titles.npy').tolist()
    # articles = np.load('data/articles.npy').tolist()
    # labels = np.load('data/labels.npy').tolist()
    # sources = np.load('data/sources.npy').tolist()

    # # uncomment below to download the punkt tokenizer
    # # nltk.download("punkt")

    # df = pd.DataFrame({
    #     'title': titles,
    #     'article': articles,
    #     'source': sources,
    #     'label': labels})

    # # df['article_length'] = df['article'].apply(len)

    # # # Use the query method to filter the DataFrame
    # # filtered_df = df.query('article_length < 13900')

    # # # Drop the temporary column we created for length calculation
    # # filtered_df = filtered_df.drop(columns=['article_length'])

    # # Split the data into training and testing sets
    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # X_train, X_test, y_train, y_test = train_test_split(articles, labels, test_size=0.2, random_state=42)
    
    # X_train = df_train['article']
    # y_train = df_train['label']

    # X_train = np.load('data/X_train.npy',allow_pickle=True).tolist()
    # y_train = np.load('data/y_train_1hot.npy').tolist()

    with open('data/X_train_by_paragraphs.pkl', 'rb') as f:
        X_train = pickle.load(f)

    y_train_1hot = np.load('data/y_train_1hot.npy')
    y_train = y_train_1hot.argmax(axis=1)

    documents = []
    for doc, label in zip(X_train, y_train):
        for paragraph in doc:
            documents.append((paragraph, label))

    # Tokenize and tag the documents
    tagged_documents = [TaggedDocument(words=word_tokenize(doc), tags=[label]) for doc, label in tqdm(documents,total=len(documents))]

    # Initialize and train the Doc2Vec model
    model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=0, epochs=20)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the model for future use
    model.save("doc2vec_paragraphs.model")




        

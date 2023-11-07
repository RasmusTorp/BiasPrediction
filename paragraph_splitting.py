

def split_into_paragraphs(article):
    paragraphs = article.split('\n')
    return paragraphs


if __name__ == '__main__':
    import pickle

    import numpy as np
    X_train = np.load('data/X_train.npy',allow_pickle=True)
    X_test = np.load('data/X_test.npy',allow_pickle=True)

    with open('data/articles.pkl', 'rb') as f:
        articles = pickle.load(f)

    articles_by_paragraphs = [split_into_paragraphs(article) for article in articles]
    X_train_by_paragraphs = [split_into_paragraphs(article) for article in X_train]
    X_test_by_paragraphs = [split_into_paragraphs(article) for article in X_test]


    with open('data/articles_by_paragraphs.pkl', 'wb') as f:
        pickle.dump(articles_by_paragraphs, f)

    with open('data/X_train_by_paragraphs.pkl', 'wb') as f:
        pickle.dump(X_train_by_paragraphs, f)

    with open('data/X_test_by_paragraphs.pkl', 'wb') as f:
        pickle.dump(X_test_by_paragraphs, f)
import xgboost as xgb
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import tqdm as tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


import xgboost as xgb
from xgboost import XGBClassifier

# VIRKER IKKE!!!

class XGBoostClassifier:
    def __init__(self, num_classes, num_boost_round=10, max_depth=6, learning_rate=0.3):
        self.num_classes = num_classes
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=num_boost_round, num_class=num_classes, objective='multi:softprob')
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        accuracy = self.model.score(X_test, y_test)
        return accuracy
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)


if __name__ == "__main__":
    # Example usage with 2 hidden layers, each with 64 neurons
    model = Doc2Vec.load("doc2vec_model.model")

    dataFolderPath = "data/"

    titles = np.load(dataFolderPath + "titles.npy")

    if os.path.exists(dataFolderPath + "titlesEmbeddings.npy"):
        titlesEmbeddings = np.load(dataFolderPath + "titlesEmbeddings.npy")

    else:
        titlesEmbeddings = [model.infer_vector(title.split()) for title in titles]
        np.save(dataFolderPath + "titlesEmbeddings.npy", np.array(titlesEmbeddings))
        
    if os.path.exists(dataFolderPath + "labels_1hot.npy"):
        labels_1hot = np.load(dataFolderPath + "labels_1hot.npy")

    else:
        labels = np.load(dataFolderPath + "labels.npy")
        NUMBER_OF_CLASSES = 3
        labels_1hot = np.identity(NUMBER_OF_CLASSES)[labels]
        np.save(dataFolderPath + "labels_1hot.npy", labels_1hot)

    TEST_SIZE = 0.2
    VALID_SIZE = 0.2
    EPOCHS = 25
    BATCHSIZE = 32
    EARLY_STOPPING = True
    PATIENCE = 5
    LEARNING_RATE = 0.001
    SEED = 42

    X, X_test, y, y_test = train_test_split(titlesEmbeddings, labels_1hot, test_size=TEST_SIZE, random_state=SEED)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_SIZE, random_state=SEED)

    # Example usage
    classifier = XGBoostClassifier(num_classes=3, num_boost_round=10, max_depth=6, learning_rate=0.3)
    classifier.train(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)
    predicted_probs = classifier.predict_proba(X_test)




    
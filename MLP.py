import numpy as np
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import keras

from evaluate import evaluate_classifier

class MLPClassifier:
    def __init__(self, input_size, hidden_layers, neurons_per_layer, num_classes = 3,dropout=0.5):
        self.model = Sequential()
        
        # Add input layer
        self.model.add(Dense(neurons_per_layer, activation='relu', input_shape=(input_size,)))
        
        # Add hidden layers
        for _ in range(hidden_layers):
            self.model.add(Dense(neurons_per_layer, activation='relu'))
            self.model.add(Dropout(dropout))
        
        # Add output layer
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, early_stopping=True, patience=5):
        callbacks = []
        
        if early_stopping:
            # Define early stopping callback
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks.append(early_stopping_callback)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=(X_val, y_val),verbose=True)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X,)
    
    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)


# Example with titles
if __name__ == '__main__':
    
    # modelType = "titles"
    # modelType = "articles"
    # input_size = 100
    
    # model = Doc2Vec.load("doc2vec_" + modelType + ".model")

    # dataFolderPath = "data/"

    # texts = np.load(dataFolderPath + "articles" + ".npy")

    # if os.path.exists(dataFolderPath + modelType +"Embeddings.npy"):
    #     embeddings = np.load(dataFolderPath + modelType +"Embeddings.npy")

    # else:
    #     embeddings = [model.infer_vector(text.split()) for text in tqdm(texts,total=len(texts))]
    #     np.save(dataFolderPath + modelType +"Embeddings.npy", np.array(embeddings))
        
    # if os.path.exists(dataFolderPath + "labels_1hot.npy"):
    #     labels_1hot = np.load(dataFolderPath + "labels_1hot.npy")

    # else:
    #     labels = np.load(dataFolderPath + "labels.npy")
    #     NUMBER_OF_CLASSES = 3
    #     labels_1hot = np.identity(NUMBER_OF_CLASSES)[labels]
    #     np.save(dataFolderPath + "labels_1hot.npy", labels_1hot)
    input_size = 100
    TEST_SIZE = 0.2
    VAL_SIZE_OF_TRAIN = 0.1
    VAL_SIZE_OF_TEST = 0.5
    EPOCHS = 25
    BATCHSIZE = 125
    EARLY_STOPPING = True
    PATIENCE = 5
    LEARNING_RATE = 0.001
    SEED = 42

    DROPOUT = 0.6
    HIDDEN_LAYERS = 2
    NEURONS_PER_LAYER = 256

    # modelType = "articles_titles"
    # title_embeddings = np.load("data/titlesEmbeddings.npy")
    # document_embeddings = np.load("data/X_train_embeddings_by_paragraphs.npy")

    # labels_1hot = np.load("data/labels_1hot.npy")

    # title_weight = 0.4
    # article_weight = 1 - title_weight

    # embeddings = np.array([title_weight * title_embedding + article_weight * document_embedding for title_embedding, document_embedding in zip(title_embeddings, document_embeddings)])

    # print("splitting data")
    # X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_1hot, test_size=TEST_SIZE, random_state=SEED)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE_OF_TRAIN, random_state=SEED)

    modelType = "paragraphs"
    X_train = np.load("data/X_train_embeddings_by_paragraphs.npy")
    y_train = np.load("data/y_train_1hot.npy")
    X_test = np.load("data/X_test_embeddings_by_paragraphs.npy")
    y_test = np.load("data/y_test_1hot.npy")
    
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE_OF_TRAIN, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=VAL_SIZE_OF_TEST, random_state=SEED)

    TRAIN = True

    if TRAIN:
        
        classifier = MLPClassifier(input_size=input_size, num_classes=3, hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER,dropout=DROPOUT)
        
        classifier.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCHSIZE, early_stopping=EARLY_STOPPING, patience=PATIENCE)

        classifier.save_model(modelType+"_MLP.model")

    else:
        classifier = MLPClassifier(input_size=input_size, num_classes=3, hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER)
        classifier.load_model(modelType+"_MLP.model")
    
    test_accuracy = classifier.evaluate(X_test, y_test)    

    y_pred = classifier.predict(X_test)

    evaluate_classifier(y_test, y_pred)

    print("done")


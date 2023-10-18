from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.callbacks import EarlyStopping

import numpy as np
from gensim.models.doc2vec import Doc2Vec
import tqdm as tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


class CNNClassifier:
    def __init__(self, input_shape, num_classes, num_filters, filter_size, pool_size, hidden_units, num_conv_layers):
        self.model = Sequential()
        
        for _ in range(num_conv_layers):
            self.model.add(Conv1D(num_filters, filter_size, activation='relu', input_shape=input_shape))
            self.model.add(MaxPooling1D(pool_size=pool_size))
        
        self.model.add(Flatten())
        self.model.add(Dense(hidden_units, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, epochs=10, batch_size=32, early_stopping=True, patience=5):
        callbacks = []
        
        if early_stopping:
            # Define early stopping callback
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            callbacks.append(early_stopping_callback)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.2)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, file_path):
        self.model.save(file_path)

# Example with titles
if __name__ == '__main__':
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
    BATCHSIZE = 32
    EARLY_STOPPING = True
    PATIENCE = 5
    LEARNING_RATE = 0.001
    SEED = 42

    EPOCHS = 25
    HIDDEN_LAYERS = 4
    NEURONS_PER_LAYER = 128
    
    X_train, X_test, y_train, y_test = train_test_split(titlesEmbeddings, labels_1hot, test_size=TEST_SIZE, random_state=SEED)

    classifier = CNNClassifier(input_shape=(100, 1), num_classes=3, num_filters=32, filter_size=3, pool_size=2, hidden_units=64, num_conv_layers=3)

    # classifier = CNNClassifier(input_shape=100, num_classes=3, num_filters=64, filter_size=3, pool_size=2, hidden_units=128, num_conv_layers=2)
    classifier.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE, early_stopping=EARLY_STOPPING, patience=PATIENCE)
    
    test_accuracy = classifier.evaluate(X_test, y_test)

    print(f"Test accuracy: {test_accuracy}")

    classifier.save_model("titles_CNN.model")
    print("done")


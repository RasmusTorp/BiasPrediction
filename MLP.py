import numpy as np
from gensim.models.doc2vec import Doc2Vec
import tqdm as tqdm
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

class MLPClassifier:
    def __init__(self, input_size, hidden_layers, neurons_per_layer, num_classes = 3):
        self.model = Sequential()
        
        # Add input layer
        self.model.add(Dense(neurons_per_layer, activation='relu', input_shape=(input_size,)))
        
        # Add hidden layers
        for _ in range(hidden_layers):
            self.model.add(Dense(neurons_per_layer, activation='relu'))
        
        # Add output layer
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
    model = Doc2Vec.load("doc2vec_titles.model")

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
    EPOCHS = 25
    BATCHSIZE = 32
    EARLY_STOPPING = True
    PATIENCE = 5
    LEARNING_RATE = 0.001
    SEED = 42
    
    X_train, X_test, y_train, y_test = train_test_split(titlesEmbeddings, labels_1hot, test_size=TEST_SIZE, random_state=SEED)


    classifier = MLPClassifier(input_size=100, num_classes=3, hidden_layers=4, neurons_per_layer=128)
    classifier.train(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE, early_stopping=EARLY_STOPPING, patience=PATIENCE)
    
    test_accuracy = classifier.evaluate(X_test, y_test)
    

    print(f"Test accuracy: {test_accuracy}")

    classifier.save_model("titles_MLP.model")
    print("done")


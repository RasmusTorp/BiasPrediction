from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import numpy as np
from utils import one_hot_to_text

def evaluate_classifier(y_true, y_pred):
    #text_labels = one_hot_to_text(y_true)

    if len(y_pred.shape) != 1:
        y_pred = y_pred.argmax(axis=1)
    
    if len(y_true.shape) != 1:
        y_true = y_true.argmax(axis=1)

    true_class_count = [0,0,0]
    pred_class_count = [0,0,0]

    for i in y_pred:
        pred_class_count[i] += 1

    for i in y_true:
        true_class_count[i] += 1
    
    print("True distribution: ")
    print(f"Left: {true_class_count[0]}, Center: {true_class_count[1]}, Right: {true_class_count[2]}")
    print()

    print("Predicted distribution: ")
    print(f"Left: {pred_class_count[0]}, Center: {pred_class_count[1]}, Right: {pred_class_count[2]}")

    print(classification_report(y_true, y_pred))

    

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["left", "center", "right"])
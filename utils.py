
import numpy as np

def one_hot_to_text(y):
    text_array = np.zeros(len(y))
    text_array = ["left" if label[0] == 1 else "center" if label[1] == 1 else "right" for label in y]
    
    return text_array

from googletrans import Translator
import numpy as np

def translate_augmentation(text, langs:[]):
    translator = Translator()
    for lang in langs:
        text = translator.translate(text, dest=lang).text
    
    text = translator.translate(text, dest='en').text
    
    return text

if __name__ == "__main__":
    text = "Trump was a very bad president and many things are his fault"
    text2 = "The state of the current political situation is a complex situation"
    translation1 = translate_augmentation(text, ["fr","de","ko"])
    translation2 = translate_augmentation(text2, ["fr","de","ko"])
    print(translation1)
    print(translation2)


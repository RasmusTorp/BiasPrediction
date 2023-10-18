import numpy as np
import nltk
from string import punctuation

# Load data
titles = np.load("data/titles.npy", allow_pickle=True)
articles = np.load("data/articles.npy", allow_pickle=True)

# Tokenize and process titles
titles = [nltk.word_tokenize(title.lower()) for title in titles]
titles = [' '.join([token for token in title if token not in punctuation]) for title in titles]

# Tokenize and process articles
articles = [nltk.word_tokenize(article.lower()) for article in articles]
articles = [' '.join([token for token in article if token not in punctuation]) for article in articles]

# Save processed data
np.save("data/articles_processed.npy", articles)
np.save("data/titles_processed.npy", titles)

print("done")

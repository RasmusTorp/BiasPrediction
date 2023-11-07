
import os
import json
import numpy as np
import pickle
from gensim.utils import simple_preprocess

# Path to the folder containing JSON files
folder_path = 'data/jsons'

# Get a list of all the JSON files in the folder
json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

# Usefull data
titles = []
articles = []
labels = []

# maybe usefull
topics = []
sources = []
dates = []
authors = []

# Loop through the JSON files
for file_name in json_files:
    # Open each JSON file
    with open(os.path.join(folder_path, file_name)) as file:
        # Load the JSON data
        data = json.load(file)
    
    titles.append(simple_preprocess(data['title']))
    articles.append(simple_preprocess(data['content']))
    labels.append(data['bias'])
    


save_folder = "data/"

with open(save_folder + 'articles.pkl', 'wb') as file:
    pickle.dump(articles, file)

with open(save_folder + 'labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

with open(save_folder + 'titles.pkl', 'wb') as file:
    pickle.dump(titles, file)

### saving with numpy
# # Convert the lists to numpy arrays
# titles_array   = np.array(titles)
# articles_array = np.array(articles)
# labels_array   = np.array(labels)

# topics = np.array(topics)
# sources = np.array(sources)
# dates = np.array(dates)
# authors = np.array(authors)

# # Save the arrays to a file
# np.save('data/titles.npy', titles_array)
# np.save('data/articles.npy', articles_array)
# np.save('data/labels.npy', labels_array)

# np.save('data/topics.npy', topics)
# np.save('data/sources.npy', sources)
# np.save('data/dates.npy', dates)
# np.save('data/authors.npy', authors)
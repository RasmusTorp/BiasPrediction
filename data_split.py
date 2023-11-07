import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm
import tiktoken
import pickle

titles = np.load('data/titles.npy').tolist()
articles = np.load('data/articles.npy').tolist()
labels = np.load('data/labels.npy').tolist()
sources = np.load('data/sources.npy').tolist()

df = pd.DataFrame({
    'title': titles,
    'article': articles,
    'source': sources,
    'label': labels})

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
#df['article_length'] = df['article'].apply(len)
df["tokens"] = df["article"].apply(encoding.encode)
df["tokens_length"] = df["tokens"].apply(len)


# Use the query method to filter the DataFrame
# filtered_df = df.query('article_length < 13900')
filtered_df = df.query('tokens_length < 16000')

# Drop the temporary column we created for length calculation
# filtered_df = filtered_df.drop(columns=['article_length'])
filtered_df = filtered_df.drop(columns=['tokens_length'])
filtered_df = filtered_df.drop(columns=['tokens'])

# Split the data into training and testing sets
df_train, df_test = train_test_split(filtered_df, test_size=0.2, random_state=42)

NUMBER_OF_CLASSES = 3
y_train_1hot = np.identity(NUMBER_OF_CLASSES)[df_train["label"]]
y_test_1hot  = np.identity(NUMBER_OF_CLASSES)[df_test["label"]]

# Vector embeddings
EMBED = False
if EMBED:
    model = Doc2Vec.load("doc2vec_articles_notLong.model")
    X_train_embeddings = [model.infer_vector(text.split()) for text in tqdm(df_train["article"],total=len(df_train["article"]))]
    X_test_embeddings = [model.infer_vector(text.split()) for text in tqdm(df_test["article"],total=len(df_test["article"]))]

    np.save("data/X_train_embeddings.npy", np.array(X_train_embeddings))
    np.save("data/X_test_embeddings.npy", np.array(X_test_embeddings))

# Save filtered_df as numpy arrays
np.save("data/X_train.npy", df_train["article"])
np.save("data/X_test.npy", df_test["article"])
np.save('data/y_train_1hot.npy', y_train_1hot)
np.save('data/y_test_1hot.npy', y_test_1hot)

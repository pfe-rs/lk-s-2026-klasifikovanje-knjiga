from gensim.models import Word2Vec
import numpy as np
import pickle
import pandas as pd

df = pd.read_pickle('springer_dataframe_26_categories.p')
tokens = [str(text).split() for text in df['toc']]

model = Word2Vec(
    tokens, 
    vector_size=100, 
    window=5, 
    min_count=1,   
    workers=4,     # koristi više CPU jezgara
    epochs=5       # smanji broj epoha
)

print(f"Veličina vokabulara: {len(model.wv)}")

#vektor knjige
def get_avg_vector(text, model):
    words = str(text).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X = np.array([get_avg_vector(text, model) for text in df['toc']])

np.save('X_vectors.npy', X)
#y.to_csv('y_labels.csv', index=False)
#print("Sačuvano!")
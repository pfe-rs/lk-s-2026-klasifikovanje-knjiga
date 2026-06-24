import random
import numpy as np
import pickle

X = np.load('X_vectors.npy')
with open(r"C:\Users\430 i7\Desktop\ana\letnji2026\TOC_Springer_26_and_5_categories\springer_dataframe_5_categories.p", "rb") as f:
    df = pickle.load(f)

y = df['Single_Label']

def make_balanced_pairs(X, y, n_per_class=1000):
    pairs_a, pairs_b, labels = [], [], []
    y = y.values
    classes = np.unique(y)
    
    for cls in classes:
        # indeksi knjiga te klase
        cls_idx = np.where(y == cls)[0]
        other_idx = np.where(y != cls)[0]
        
        #ista klasa
        for _ in range(n_per_class):
            i, j = random.sample(list(cls_idx), 2)
            pairs_a.append(X[i])
            pairs_b.append(X[j])
            labels.append(1)
        
        #različite klase
        for _ in range(n_per_class):
            i = random.choice(cls_idx)
            j = random.choice(other_idx)
            pairs_a.append(X[i])
            pairs_b.append(X[j])
            labels.append(0)
    
    return np.array(pairs_a), np.array(pairs_b), np.array(labels)

pairs_a, pairs_b, labels = make_balanced_pairs(X, y, n_per_class=1000)

# print(f"Oblik pairs_a: {pairs_a.shape}")  # (10000, 100)
# print(f"Oblik pairs_b: {pairs_b.shape}")  # (10000, 100)
# print(f"Oblik labels: {labels.shape}")    # (10000,)
# print(f"Pozitivnih: {labels.sum()}")      # 5000
# print(f"Negativnih: {(labels==0).sum()}") # 5000

np.save('pairs_a.npy', pairs_a)
np.save('pairs_b.npy', pairs_b)
np.save('labels.npy', labels)

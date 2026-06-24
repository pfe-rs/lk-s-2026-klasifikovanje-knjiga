import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec

df = pd.read_pickle('springer_dataframe_26_categories.p')

klase_counts = df['LCSH_Label'].value_counts()
validne_klase = klase_counts[klase_counts >= 500].index
df= df[df['LCSH_Label'].isin(validne_klase)].reset_index(drop=True)

train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['LCSH_Label']
)
test_df, val_df = train_test_split(
    test_df, 
    test_size=0.5, 
    stratify=test_df['LCSH_Label']
)

tokens = [str(text).split() for text in df['toc']]

w2v_model = Word2Vec(
    tokens, 
    vector_size=300, 
    window=5, 
    min_count=1,   
    workers=4,     # koristi više CPU jezgara
    epochs=5       # smanji broj epoha
)

print(f"Veličina vokabulara: {len(w2v_model.wv)}")

#vektor knjige
def doc_to_vector(text, model):
    words = str(text).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

train_df['tfidf_vector'] = train_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
val_df['tfidf_vector'] = val_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
test_df['tfidf_vector'] = test_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
N = 3
labels = train_df['LCSH_Label'].unique()


def build_triplets(source_df, labels, n=N):
    triplets_list = []
    for label1, label2 in itertools.combinations(labels, 2):

        df_class1 = source_df[source_df['LCSH_Label'] == label1]
        df_class2 = source_df[source_df['LCSH_Label'] == label2]

        if len(df_class1) == 0 or len(df_class2) == 0:
            continue

        anchor_samples = df_class1.sample(n=n, replace=True).reset_index(drop=True)
        positive_samples = df_class1.sample(n=n, replace=True).reset_index(drop=True)
        negative_samples = df_class2.sample(n=n, replace=True).reset_index(drop=True)

        for i in range(n):
            triplets_list.append({
                'input_anchor': anchor_samples.iloc[i]['tfidf_vector'],
                'input_positive': positive_samples.iloc[i]['tfidf_vector'],
                'input_negative': negative_samples.iloc[i]['tfidf_vector']
            })
    return triplets_list


train_triplets_df = pd.DataFrame(build_triplets(train_df, labels))
train_triplets_df = train_triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)

val_triplets_df = pd.DataFrame(build_triplets(val_df, labels))
val_triplets_df = val_triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)


class TripletDataset(Dataset):
    def __init__(self, triplets_df):
        self.anchor = np.stack(triplets_df['input_anchor'].values)
        self.positive = np.stack(triplets_df['input_positive'].values)
        self.negative = np.stack(triplets_df['input_negative'].values)

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.anchor[idx], dtype=torch.float32),
            torch.tensor(self.positive[idx], dtype=torch.float32),
            torch.tensor(self.negative[idx], dtype=torch.float32)
        )

train_dataset = TripletDataset(train_triplets_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TripletDataset(val_triplets_df)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=300, embedding_dim=64):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class TripletNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x_anchor, x_positive, x_negative):
        out_anchor = self.embedding_net(x_anchor)
        out_positive = self.embedding_net(x_positive)
        out_negative = self.embedding_net(x_negative)
        return out_anchor, out_positive, out_negative

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, out_anchor, out_positive, out_negative):
        
        sim_pos = F.cosine_similarity(out_anchor, out_positive)
        sim_neg = F.cosine_similarity(out_anchor, out_negative)
        
        
        dist_pos = 1.0 - sim_pos
        dist_neg = 1.0 - sim_neg
        
        
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return torch.mean(loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emb_net = EmbeddingNetwork(input_dim=300, embedding_dim=64)
model = TripletNetwork(emb_net).to(device)

criterion = CosineTripletLoss(margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 3

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    for batch_anchor, batch_positive, batch_negative in train_loader:
        batch_anchor = batch_anchor.to(device)
        batch_positive = batch_positive.to(device)
        batch_negative = batch_negative.to(device)
        
        optimizer.zero_grad()
        
        out_anchor, out_positive, out_negative = model(batch_anchor, batch_positive, batch_negative)
        
        loss = criterion(out_anchor, out_positive, out_negative)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_anchor.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_running_loss = 0.0

    with torch.no_grad():
        for val_anchor, val_positive, val_negative in val_loader:
            val_anchor = val_anchor.to(device)
            val_positive = val_positive.to(device)
            val_negative = val_negative.to(device)

            val_out_anchor, val_out_positive, val_out_negative = model(val_anchor, val_positive, val_negative)

            val_loss = criterion(val_out_anchor, val_out_positive, val_out_negative)

            val_running_loss += val_loss.item() * val_anchor.size(0)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] -> Train Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")


model.eval()
class_prototypes = {}


with torch.no_grad(): 
    for label in labels:
        class_df = train_df[train_df['LCSH_Label'] == label]
        class_vectors = torch.tensor(np.stack(class_df['tfidf_vector'].values), dtype=torch.float32).to(device)
        class_embeddings = emb_net(class_vectors)
        mean_prototype = torch.mean(class_embeddings, dim=0)
        class_prototypes[label] = mean_prototype

test_vectors_np = np.stack(test_df['tfidf_vector'].values)
test_vectors = torch.tensor(test_vectors_np, dtype=torch.float32).to(device)
test_labels = test_df['LCSH_Label'].values

correct_predictions = 0
total_predictions = len(test_labels)


predicted_labels = []

with torch.no_grad():
    test_embeddings = emb_net(test_vectors)

for i in range(total_predictions):
    test_embed = test_embeddings[i]  
    true_label = test_labels[i]
    
    best_label = None
    smallest_distance = float('inf')
    
    for label, prototype in class_prototypes.items():
        sim = F.cosine_similarity(test_embed.unsqueeze(0), prototype.unsqueeze(0))
        dist = 1.0 - sim.item() 
        
        if dist < smallest_distance:
            smallest_distance = dist
            best_label = label

    predicted_labels.append(best_label) #
    if best_label == true_label:
        correct_predictions += 1

accuracy = (correct_predictions / total_predictions) * 100
print("-" * 40)
print(f"Evaluation Results:")
print(f"Total Test Samples: {total_predictions}")
print(f"Correctly Classified: {correct_predictions}")
print(f"Final Model Accuracy: {accuracy:.2f}%")


predicted_labels = np.array(predicted_labels)

per_class_results = {}
for label in np.unique(test_labels):
    mask        = test_labels == label
    ukupno      = mask.sum()
    pogodeno    = (predicted_labels[mask] == label).sum()
    procenat    = (pogodeno / ukupno) * 100
    per_class_results[label] = (pogodeno, ukupno, procenat)


sorted_results = sorted(per_class_results.items(), key=lambda x: x[1][2], reverse=True)

print("\n" + "=" * 70)
print("Preciznost po klasi:")
print(f"{'Klasa':<45} {'Pogođeno':>9} {'Ukupno':>8} {'%':>7}")
print("-" * 70)
for label, (pogodeno, ukupno, procenat) in sorted_results:
    print(f"{label:<45} {pogodeno:>9} {ukupno:>8} {procenat:>6.1f}%")
print("=" * 70)
print(f"{'UKUPNA TAČNOST':<45} {correct_predictions:>9} {total_predictions:>8} {accuracy:>6.2f}%")
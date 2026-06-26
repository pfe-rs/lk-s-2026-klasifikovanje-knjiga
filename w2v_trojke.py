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
import mlflow

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

EMBEDDING_DIM=500

tokens = [str(text).split() for text in df['toc']]

w2v_model = Word2Vec(
    tokens, 
    vector_size=EMBEDDING_DIM, 
    window=5, 
    min_count=1,   
    workers=4,     
    epochs=2       
)


def doc_to_vector(text, model):
    words = str(text).split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

train_df['tfidf_vector'] = train_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
val_df['tfidf_vector'] = val_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
test_df['tfidf_vector'] = test_df['toc'].apply(lambda text: doc_to_vector(text, w2v_model))
labels = train_df['LCSH_Label'].unique()


def build_triplets(source_df, labels):
    triplets_list = []
    for label1, label2 in itertools.combinations(labels, 2):

        df_class1 = source_df[source_df['LCSH_Label'] == label1]
        df_class2 = source_df[source_df['LCSH_Label'] == label2]
        N=min(len(df_class1), len(df_class2))
        if len(df_class1) == 0 or len(df_class2) == 0:
            continue

        anchor_samples = df_class1.sample(n=N, replace=True).reset_index(drop=True)
        positive_samples = df_class1.sample(n=N, replace=True).reset_index(drop=True)
        negative_samples = df_class2.sample(n=N, replace=True).reset_index(drop=True)

        for i in range(N):
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
    def __init__(self, input_dim=EMBEDDING_DIM, embedding_dim=128):
        super(EmbeddingNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
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

emb_net = EmbeddingNetwork(input_dim=EMBEDDING_DIM, embedding_dim=64)
model = TripletNetwork(emb_net).to(device)


LR = 0.001
criterion = CosineTripletLoss(margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
NUM_EPOCHS = 20


def compute_prototypes(embedding_net, source_df, labels, device):
    prototypes = {}
    with torch.no_grad():
        for label in labels:
            class_df = source_df[source_df['LCSH_Label'] == label]
            class_vectors = torch.tensor(
                np.stack(class_df['tfidf_vector'].values), dtype=torch.float32
            ).to(device)
            class_embeddings = embedding_net(class_vectors)
            prototypes[label] = torch.mean(class_embeddings, dim=0)
    return prototypes


def compute_accuracy(embedding_net, prototypes, eval_df, device):
    eval_tfidf = np.stack(eval_df['tfidf_vector'].values)
    eval_vectors = torch.tensor(eval_tfidf, dtype=torch.float32).to(device)
    eval_labels = eval_df['LCSH_Label'].values

    with torch.no_grad():
        eval_embeddings = embedding_net(eval_vectors)

    correct = 0
    for i in range(len(eval_labels)):
        embed = eval_embeddings[i]
        true_label = eval_labels[i]

        best_label = None
        smallest_distance = float('inf')
        for label, prototype in prototypes.items():
            sim = F.cosine_similarity(embed.unsqueeze(0), prototype.unsqueeze(0))
            dist = 1.0 - sim.item()
            if dist < smallest_distance:
                smallest_distance = dist
                best_label = label

        if best_label == true_label:
            correct += 1

    return (correct / len(eval_labels)) * 100


with mlflow.start_run():
    mlflow.log_param("lr", LR)
    mlflow.log_param("epochs", NUM_EPOCHS)

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

        running_val_loss = 0.0
        with torch.no_grad():
            for batch_anchor, batch_positive, batch_negative in val_loader:
                batch_anchor, batch_positive, batch_negative = batch_anchor.to(device), batch_positive.to(device), batch_negative.to(device)

                out_anchor, out_positive, out_negative = model(batch_anchor, batch_positive, batch_negative)
                loss = criterion(out_anchor, out_positive, out_negative)

                running_val_loss += loss.item() * batch_anchor.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        # --- Validation accuracy preko prototipova (racunato na train_df, testirano na val_df) ---
        prototypes = compute_prototypes(emb_net, train_df, labels, device)
        val_accuracy = compute_accuracy(emb_net, prototypes, val_df, device)

        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

model.eval()
final_prototypes = compute_prototypes(emb_net, train_df, labels, device)

test_accuracy = compute_accuracy(emb_net, final_prototypes, test_df, device)

print("-" * 40)
print(f"Evaluation Results:")
print(f"Total Test Samples: {len(test_df)}")
print(f"Final Model Accuracy: {test_accuracy:.2f}%")
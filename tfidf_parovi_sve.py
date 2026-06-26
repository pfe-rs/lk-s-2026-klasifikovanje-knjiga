import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mlflow
mlflow.set_experiment('5 klasa - parovi tfidf1')
df = pd.read_pickle('springer_dataframe_5_categories.p')

train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['Single_Label']
)

test_df, val_df = train_test_split(
    test_df,
    test_size=0.5,
    stratify=test_df['Single_Label']
)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(train_df['toc'])
train_df['tfidf_vector'] = [np.array(vec).flatten() for vec in tfidf_matrix.toarray()]

val_tfidf_matrix = tfidf.transform(val_df['toc'])
val_df['tfidf_vector'] = [np.array(vec).flatten() for vec in val_tfidf_matrix.toarray()]

labels = train_df['Single_Label'].unique()

pairs_list = []
for label1, label2 in itertools.combinations_with_replacement(labels, 2):
    df_class1 = train_df[train_df['Single_Label'] == label1]
    df_class2 = train_df[train_df['Single_Label'] == label2]
    
    N_samples = min(len(df_class1), len(df_class2))
    
    left_samples = df_class1.sample(n=N_samples, replace=True).reset_index(drop=True)
    right_samples = df_class2.sample(n=N_samples, replace=True).reset_index(drop=True)
        
    is_same = 1 if label1 == label2 else 0

    for i in range(N_samples):
        pairs_list.append({
            'input_left': left_samples.iloc[i]['tfidf_vector'],
            'input_right': right_samples.iloc[i]['tfidf_vector'],
            'label_left': label1,
            'label_right': label2,
            'is_same': is_same
        })

pairs_list_val = []
for label1, label2 in itertools.combinations_with_replacement(labels, 2):
    df_class1 = val_df[val_df['Single_Label'] == label1]
    df_class2 = val_df[val_df['Single_Label'] == label2]
    
    N_samples = min(len(df_class1), len(df_class2))
    
    left_samples = df_class1.sample(n=N_samples, replace=True).reset_index(drop=True)
    right_samples = df_class2.sample(n=N_samples, replace=True).reset_index(drop=True)
        
    is_same = 1 if label1 == label2 else 0

    for i in range(N_samples):
        pairs_list_val.append({
            'input_left': left_samples.iloc[i]['tfidf_vector'],
            'input_right': right_samples.iloc[i]['tfidf_vector'],
            'label_left': label1,
            'label_right': label2,
            'is_same': is_same
        })

train_pairs_df = pd.DataFrame(pairs_list)
train_pairs_df = train_pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

val_pairs_df = pd.DataFrame(pairs_list_val)
val_pairs_df = val_pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=5000, embedding_dim=128):
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

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x_left, x_right):
        output_left = self.embedding_net(x_left)
        output_right = self.embedding_net(x_right)
        return output_left, output_right

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output_left, output_right, target):
        cosine_sim = F.cosine_similarity(output_left, output_right)
        cosine_dist = 1 - cosine_sim
        
        loss_same = target * torch.pow(cosine_dist, 2)
        loss_diff = (1 - target) * torch.pow(torch.clamp(self.margin - cosine_dist, min=0.0), 2)
        
        return torch.mean(loss_same + loss_diff)

class SiameseDataset(Dataset):
    def __init__(self, pairs_df):
        self.x_left = np.stack(pairs_df['input_left'].values)
        self.x_right = np.stack(pairs_df['input_right'].values)
        self.labels = pairs_df['is_same'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_left[idx], dtype=torch.float32),
            torch.tensor(self.x_right[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

train_dataset = SiameseDataset(train_pairs_df)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = SiameseDataset(val_pairs_df)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emb_net = EmbeddingNetwork(input_dim=5000, embedding_dim=128)
model = SiameseNetwork(emb_net).to(device)

criterion = CosineContrastiveLoss(margin=1)
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
NUM_EPOCHS = 8 

with mlflow.start_run():
    mlflow.log_param("lr", LR)
    mlflow.log_param("epochs", NUM_EPOCHS)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_left, batch_right, batch_labels in train_loader:
            batch_left = batch_left.to(device)
            batch_right = batch_right.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            out_left, out_right = model(batch_left, batch_right)
            loss = criterion(out_left, out_right, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_left.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        class_prototypes = {}

        with torch.no_grad(): 
            for label in labels:
                class_df = train_df[train_df['Single_Label'] == label]
                class_vectors = torch.tensor(np.stack(class_df['tfidf_vector'].values), dtype=torch.float32).to(device)
                class_embeddings = emb_net(class_vectors)
                mean_prototype = torch.mean(class_embeddings, dim=0)
                class_prototypes[label] = mean_prototype

        test_tfidf_matrix = tfidf.transform(test_df['toc'])
        test_vectors = torch.tensor(test_tfidf_matrix.toarray(), dtype=torch.float32).to(device)
        test_labels = test_df['Single_Label'].values

        correct_predictions = 0
        total_predictions = len(test_labels)

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
                    
            if best_label == true_label:
                correct_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100

        running_val_loss = 0.0
        with torch.no_grad():
            for batch_left, batch_right, batch_labels in val_loader:
                batch_left = batch_left.to(device)
                batch_right = batch_right.to(device)
                batch_labels = batch_labels.to(device)

                out_left, out_right = model(batch_left, batch_right)
                val_loss = criterion(out_left, out_right, batch_labels)

                running_val_loss += val_loss.item() * batch_left.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] -> Avg Loss: {epoch_loss:.4f} -> Val Loss: {epoch_val_loss:.4f} -> Val Acc: {accuracy:.4f}")

model.eval()
class_prototypes = {}

with torch.no_grad(): 
    for label in labels:
        class_df = train_df[train_df['Single_Label'] == label]
        class_vectors = torch.tensor(np.stack(class_df['tfidf_vector'].values), dtype=torch.float32).to(device)
        class_embeddings = emb_net(class_vectors)
        mean_prototype = torch.mean(class_embeddings, dim=0)
        class_prototypes[label] = mean_prototype

test_tfidf_matrix = tfidf.transform(test_df['toc'])
test_vectors = torch.tensor(test_tfidf_matrix.toarray(), dtype=torch.float32).to(device)
test_labels = test_df['Single_Label'].values

correct_predictions = 0
total_predictions = len(test_labels)

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
            
    if best_label == true_label:
        correct_predictions += 1

final_accuracy = (correct_predictions / total_predictions) * 100
print("-" * 40)
print(f"Evaluation Results:")
print(f"Total Test Samples: {total_predictions}")
print(f"Correctly Classified: {correct_predictions}")
print(f"Final Model Accuracy: {final_accuracy:.2f}%")
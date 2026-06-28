import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import mlflow


mlflow.set_experiment("test")

df = pd.read_pickle("springer_dataframe_5_categories.p")

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Single_Label"], random_state=42,
)
test_df, val_df = train_test_split(
    test_df, test_size=0.5, stratify=test_df["Single_Label"], random_state=42,
)


TFIDF_DIM = 5000
LSA_DIM   = 300

tfidf = TfidfVectorizer(max_features=TFIDF_DIM, stop_words="english", sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(train_df["toc"])
X_val_tfidf   = tfidf.transform(val_df["toc"])
X_test_tfidf  = tfidf.transform(test_df["toc"])

lsa = TruncatedSVD(n_components=LSA_DIM, random_state=42)
X_train = lsa.fit_transform(X_train_tfidf).astype(np.float32)
X_val   = lsa.transform(X_val_tfidf).astype(np.float32)
X_test  = lsa.transform(X_test_tfidf).astype(np.float32)

y_train = train_df["Single_Label"].values
y_val   = val_df["Single_Label"].values
y_test  = test_df["Single_Label"].values
labels  = np.unique(y_train)

print(f"[data] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  dim={LSA_DIM}")


def l2_normalise(X: torch.Tensor) -> torch.Tensor:
    return F.normalize(X, p=2, dim=1)

def cosine_distance_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return 1.0 - A @ B.T

class SOM:
    def __init__(
        self,
        map_h: int = 16,
        map_w: int = 16,
        input_dim: int = LSA_DIM,
        n_epochs: int = 200,
        lr_start: float = 0.8,
        lr_end: float = 0.01,
        radius_start: float = 7.0,
        radius_end: float = 0.5,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.map_h        = map_h
        self.map_w        = map_w
        self.n_neurons    = map_h * map_w
        self.input_dim    = input_dim
        self.n_epochs     = n_epochs
        self.lr_start     = lr_start
        self.lr_end       = lr_end
        self.radius_start = radius_start
        self.radius_end   = radius_end
        self.batch_size   = batch_size
        self.device       = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.weights: torch.Tensor | None = None

        rows, cols = np.unravel_index(np.arange(self.n_neurons), (map_h, map_w))
        gc = np.stack([rows, cols], axis=1).astype(np.float32)
        self.grid_coords = torch.tensor(gc, device=self.device)
        diff = self.grid_coords.unsqueeze(0) - self.grid_coords.unsqueeze(1)
        self.sq_grid_dist = (diff ** 2).sum(dim=-1)  

        self.neuron_labels: np.ndarray | None = None


    def _exp_decay(self, start: float, end: float, epoch: int) -> float:
        if self.n_epochs <= 1:
            return end
        tau = self.n_epochs / np.log(start / (end + 1e-8))
        return start * np.exp(-epoch / tau)

    def _pca_init(self, X: torch.Tensor) -> torch.Tensor:
        X_cpu = X.cpu().numpy()
        _, _, Vt = np.linalg.svd(X_cpu - X_cpu.mean(axis=0), full_matrices=False)
        pc1, pc2 = Vt[0], Vt[1]
        g1 = np.linspace(-1, 1, self.map_h)
        g2 = np.linspace(-1, 1, self.map_w)
        rows, cols = np.unravel_index(np.arange(self.n_neurons), (self.map_h, self.map_w))
        W = g1[rows, np.newaxis] * pc1 + g2[cols, np.newaxis] * pc2   # (n_neurons, d)
        W = W.astype(np.float32)
        return torch.tensor(W, device=self.device)

    def _bmu_indices(self, X_norm: torch.Tensor) -> torch.Tensor:
        return (X_norm @ self.weights.T).argmax(dim=1)

    def _neighbourhood_matrix(self, bmu_indices: torch.Tensor, radius: float) -> torch.Tensor:
        sq = self.sq_grid_dist[bmu_indices]
        return torch.exp(-sq / (2.0 * radius ** 2))

    def _batch_update(self, batch: torch.Tensor, lr: float, radius: float) -> None:
        bmu_idx   = self._bmu_indices(batch)
        h         = self._neighbourhood_matrix(bmu_idx, radius)
        numerator   = h.T @ batch
        denominator = h.sum(dim=0).clamp(min=1e-8).unsqueeze(1)
        self.weights = l2_normalise(
            (1.0 - lr) * self.weights + lr * (numerator / denominator)
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict:

        history: dict[str, list] = {"train_qe": [], "val_acc": []}
        n_samples  = len(X)

        X_gpu = l2_normalise(torch.tensor(X, dtype=torch.float32, device=self.device))
        self.weights = l2_normalise(self._pca_init(X_gpu))
        print("[SOM] weights initialised via PCA")

        for epoch in range(self.n_epochs):
            lr     = self._exp_decay(self.lr_start, self.lr_end,     epoch)
            radius = self._exp_decay(self.radius_start, self.radius_end, epoch)

            perm   = torch.randperm(n_samples, device=self.device)
            X_shuf = X_gpu[perm]

            for start in range(0, n_samples, self.batch_size):
                self._batch_update(
                    X_shuf[start : start + self.batch_size], lr, radius
                )
            with torch.no_grad():
                sim  = X_gpu @ self.weights.T
                bmus = sim.argmax(dim=1)
                qe   = float(
                    (1.0 - sim[torch.arange(n_samples, device=self.device), bmus]).mean()
                )
            history["train_qe"].append(qe)

            val_acc = None
            if X_val is not None and y_val is not None:
                self.label_neurons(X, y)
                val_acc = self.score(X_val, y_val)
                history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch+1:02d}/{self.n_epochs}]  "
                f"LR={lr:.4f}  Radius={radius:.2f}  QE={qe:.4f}"
                + (f"  Val Acc={val_acc * 100:.2f}%" if val_acc is not None else "")
            )

        return history

    def label_neurons(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_gpu   = l2_normalise(torch.tensor(X_train, dtype=torch.float32, device=self.device))
        bmus    = self._bmu_indices(X_gpu).cpu().numpy()   

        vote_map: dict[int, Counter] = {i: Counter() for i in range(self.n_neurons)}
        for sample_idx, neuron_idx in enumerate(bmus):
            vote_map[neuron_idx][y_train[sample_idx]] += 1

        neuron_labels = np.empty(self.n_neurons, dtype=object)
        dead_neurons  = []

        for neuron_idx in range(self.n_neurons):
            if vote_map[neuron_idx]:
                neuron_labels[neuron_idx] = vote_map[neuron_idx].most_common(1)[0][0]
            else:
                dead_neurons.append(neuron_idx)
        if dead_neurons:
            unique_labels = np.unique(y_train)
            X_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
            protos, proto_labels = [], []
            for lbl in unique_labels:
                p = X_norm[y_train == lbl].mean(axis=0)
                protos.append(p / (np.linalg.norm(p) + 1e-8))
                proto_labels.append(lbl)

            P   = torch.tensor(np.stack(protos), dtype=torch.float32, device=self.device)
            # cosine dist between dead neurons and prototypes
            dead_W   = self.weights[dead_neurons]                # (n_dead, d)
            dist     = cosine_distance_matrix(dead_W, P)         # (n_dead, n_classes)
            closest  = dist.argmin(dim=1).cpu().numpy()
            for k, neuron_idx in enumerate(dead_neurons):
                neuron_labels[neuron_idx] = proto_labels[closest[k]]

            print(f"[SOM] dead neurons: {len(dead_neurons)} / {self.n_neurons} "
                  f"(labelled by prototype fallback)")

        self.neuron_labels = neuron_labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.neuron_labels is None:
            raise RuntimeError("Call label_neurons() before predict().")
        X_gpu = l2_normalise(torch.tensor(X, dtype=torch.float32, device=self.device))
        bmus  = self._bmu_indices(X_gpu).cpu().numpy()
        return self.neuron_labels[bmus]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

MAP_H         = 14
MAP_W         = 14
N_EPOCHS      = 200
LR_START      = 0.6
LR_END        = 0.01
RADIUS_START  = 7.0
RADIUS_END    = 0.5
BATCH_SIZE    = 64

som = SOM(
    map_h=MAP_H,
    map_w=MAP_W,
    input_dim=LSA_DIM,
    n_epochs=N_EPOCHS,
    lr_start=LR_START,
    lr_end=LR_END,
    radius_start=RADIUS_START,
    radius_end=RADIUS_END,
    batch_size=BATCH_SIZE,
)

with mlflow.start_run():
    mlflow.log_params({
        "map_h":         MAP_H,
        "map_w":         MAP_W,
        "n_epochs":      N_EPOCHS,
        "lr_start":      LR_START,
        "lr_end":        LR_END,
        "radius_start":  RADIUS_START,
        "radius_end":    RADIUS_END,
        "batch_size":    BATCH_SIZE,
        "tfidf_dim":     TFIDF_DIM,
        "lsa_dim":       LSA_DIM,
        "device":        str(som.device),
    })

    history = som.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    for epoch, qe in enumerate(history["train_qe"]):
        mlflow.log_metric("train_qe", qe, step=epoch)
    for epoch, acc in enumerate(history.get("val_acc", [])):
        mlflow.log_metric("val_acc", acc * 100, step=epoch)

    som.label_neurons(X_train, y_train)

    y_pred_test = som.predict(X_test)
    test_acc    = float(np.mean(y_pred_test == y_test)) * 100

    mlflow.log_metric("final_test_accuracy", test_acc)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print()
    print(classification_report(y_test, y_pred_test, target_names=[str(l) for l in labels]))
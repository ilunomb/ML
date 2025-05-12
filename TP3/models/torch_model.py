import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.constants import *
from models.utils import load_hparams_csv
import cupy as cp
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

class PyTorchModel(nn.Module):
    def __init__(self, layer_sizes, use_batchnorm=False, dropout_rate=0.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader,
                epochs, lr, weight_decay=0.0,
                use_adam=True, beta1=0.9, beta2=0.999,
                scheduler_type=None, final_lr=None,
                early_stopping=True, patience=20,
                device="cuda"):

    model.to(device)

    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == "linear" and final_lr is not None:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=final_lr / lr, total_iters=epochs)
    elif scheduler_type == "exponential" and final_lr is not None:
        gamma = (final_lr / lr) ** (1 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_state = None
    no_improve_epochs = 0

    # Historial
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    progress = trange(epochs, desc="Training", ncols=100)
    for epoch in progress:
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if scheduler:
            scheduler.step()

        # Validación
        model.eval()
        val_loss_total = 0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss_total += criterion(out, yb).item()
                val_correct += (out.argmax(1) == yb).sum().item()

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        progress.set_description(f"Epoch {epoch+1} - TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f}, ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if early_stopping and no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

def train_best_model_from_csv(X_train, y_train, X_val, y_val, csv_path=HYPERPARAMS_FILE):
    """
    Carga hiperparámetros desde CSV, entrena el modelo M2 con ellos y devuelve el modelo entrenado.
    """
    params = load_hparams_csv(csv_path)

    X_train_t, y_train_t = to_tensor(cp.asnumpy(X_train), cp.asnumpy(y_train))
    X_val_t, y_val_t = to_tensor(cp.asnumpy(X_val), cp.asnumpy(y_val))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512)

    model = PyTorchModel(
        layer_sizes=params["layer_sizes"],
        use_batchnorm=params["use_batchnorm"],
        dropout_rate=params["dropout_rate"]
    )

    model, history = train_model(
        model, train_loader, val_loader,
        epochs=params["epochs"],
        lr=params["lr"],
        weight_decay=params["l2_lambda"],
        use_adam=params["use_adam"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        scheduler_type=params["scheduler_type"],
        final_lr=params["final_lr"],
        early_stopping=params["early_stopping"],
        patience=params["patience"]
    )

    return model, history


def train_architecture_set(X_train, y_train, X_val, y_val, architectures, csv_path="hparams.csv"):
    params = load_hparams_csv(csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train_t, y_train_t = to_tensor(cp.asnumpy(X_train), cp.asnumpy(y_train))
    X_val_t, y_val_t = to_tensor(cp.asnumpy(X_val), cp.asnumpy(y_val))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512)

    results = []

    for arch in architectures:
        print(f"\nEntrenando arquitectura: {arch}")
        model = PyTorchModel(
            layer_sizes=arch,
            use_batchnorm=params["use_batchnorm"],
            dropout_rate=params["dropout_rate"]
        )

        model, history = train_model(
            model, train_loader, val_loader,
            epochs=params["epochs"],
            lr=params["lr"],
            weight_decay=params["l2_lambda"],
            use_adam=params["use_adam"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            scheduler_type=params["scheduler_type"],
            final_lr=params["final_lr"],
            early_stopping=params["early_stopping"],
            patience=params["patience"]
        )

        # Validación
        model.eval()
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                total_loss += criterion(out, yb).item()
        val_loss = total_loss / len(val_loader)

        results.append((arch, val_loss, model, history))
        print(f"Val loss: {val_loss:.4f}")

    results.sort(key=lambda x: x[1])
    return results


def to_tensor(x, y):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.long, device=device)


def evaluate_metrics(model, X, y, title_prefix="", device="cuda"):
    """
    Evalúa accuracy, loss y matriz de confusión dados X, y en CuPy.

    Args:
        model: modelo PyTorch entrenado
        X: matriz de inputs (cp.ndarray)
        y: vector de etiquetas verdaderas (cp.ndarray)
        title_prefix: prefijo para el título del gráfico
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # Convertir de CuPy a Torch en GPU
    X_torch = torch.tensor(cp.asnumpy(X), dtype=torch.float32, device=device)
    y_torch = torch.tensor(cp.asnumpy(y), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch).item()
        preds = outputs.argmax(dim=1).cpu().numpy()

    y_true = y.get()
    y_pred = preds

    acc = np.mean(y_pred == y_true)
    cm = _confusion_matrix_cp(cp.asarray(y_true), cp.asarray(y_pred))

    print(f"{title_prefix} Accuracy: {acc:.4f}")
    print(f"{title_prefix} Cross-Entropy Loss: {loss:.4f}")
    _plot_confusion_matrix(cm, title=f"{title_prefix} - Matriz de Confusión")


def _confusion_matrix_cp(y_true: cp.ndarray, y_pred: cp.ndarray) -> cp.ndarray:
    num_classes = int(cp.max(y_true).item()) + 1
    matrix = cp.zeros((num_classes, num_classes), dtype=cp.int32)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def _plot_confusion_matrix(cm: cp.ndarray, title: str = "", figsize=(7, 6)):
    cm_np = cm.get()  # cupy → numpy

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_np, interpolation='nearest', cmap='Blues')

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    num_classes = cm_np.shape[0]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6)

    fig.tight_layout()
    plt.show()

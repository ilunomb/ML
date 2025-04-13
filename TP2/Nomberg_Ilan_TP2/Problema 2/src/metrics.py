import numpy as np
import matplotlib.pyplot as plt

# --------- Métricas Escalares ---------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, labels=None):
    labels = labels or np.unique(np.concatenate([y_true, y_pred]))
    precisions = []

    for cls in labels:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        p = tp / (tp + fp + 1e-15)
        precisions.append(p)

    return np.mean(precisions)

def recall(y_true, y_pred, labels=None):
    labels = labels or np.unique(np.concatenate([y_true, y_pred]))
    recalls = []

    for cls in labels:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        r = tp / (tp + fn + 1e-15)
        recalls.append(r)

    return np.mean(recalls)

def f1_score(y_true, y_pred, labels=None):
    labels = labels or np.unique(np.concatenate([y_true, y_pred]))
    f1s = []

    for cls in labels:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        p = tp / (tp + fp + 1e-15)
        r = tp / (tp + fn + 1e-15)
        f1 = 2 * p * r / (p + r + 1e-15)
        f1s.append(f1)

    return np.mean(f1s)

# --------- Matriz de Confusión ---------
def plot_conf_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    labels = labels or np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

# --------- Curvas ROC / PR (Multiclase One-vs-Rest) ---------
def roc_curve(y_true, y_scores, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    tpr_list = []
    fpr_list = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn + 1e-15)
        fpr = fp / (fp + tn + 1e-15)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)

def pr_curve(y_true, y_scores, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        p = tp / (tp + fp + 1e-15)
        r = tp / (tp + fn + 1e-15)

        precisions.append(p)
        recalls.append(r)

    return np.array(recalls), np.array(precisions)

def auc(x, y):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    return np.trapz(y_sorted, x_sorted)

def plot_roc_curve(y_true, y_proba, labels=None, show=True):
    labels = labels or np.unique(y_true)
    y_true = np.asarray(y_true)
    n_classes = y_proba.shape[1]
    
    aucs = []
    plt.figure()

    for i, cls in enumerate(labels):
        binary_true = (y_true == cls).astype(int)
        binary_scores = y_proba[:, i]
        fpr, tpr = roc_curve(binary_true, binary_scores)
        auc_val = auc(fpr, tpr)
        aucs.append(auc_val)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc_val:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass)")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

    return aucs

def plot_pr_curve(y_true, y_proba, labels=None, show=True):
    labels = labels or np.unique(y_true)
    y_true = np.asarray(y_true)
    n_classes = y_proba.shape[1]

    aucs = []
    plt.figure()

    for i, cls in enumerate(labels):
        binary_true = (y_true == cls).astype(int)
        binary_scores = y_proba[:, i]
        recall_vals, precision_vals = pr_curve(binary_true, binary_scores)

        # Ordenar por recall (por si acaso)
        sorted_idx = np.argsort(recall_vals)
        recall_vals = recall_vals[sorted_idx]
        precision_vals = precision_vals[sorted_idx]

        # Agregar punto inicial (0,1) explícitamente
        recall_vals = np.insert(recall_vals, 0, 0.0)
        precision_vals = np.insert(precision_vals, 0, 1.0)

        auc_val = auc(recall_vals, precision_vals)
        aucs.append(auc_val)
        plt.plot(recall_vals, precision_vals, label=f"Class {cls} (AUC = {auc_val:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Multiclass)")
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()

    return aucs


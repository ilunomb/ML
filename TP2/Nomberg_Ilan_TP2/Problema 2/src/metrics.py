import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_conf_matrix(y_true, y_pred, labels=["Negative", "Positive"], title="Confusion Matrix"):
    # Calcular TP, TN, FP, FN
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="BuGn")

    # Etiquetas
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    # Anotar valores en cada celda
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    # Color bar
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-15)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-15)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-15)

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

def plot_roc_curve(y_true, y_scores, label=None, show=True):
    fpr, tpr = roc_curve(y_true, y_scores)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=label or f"ROC AUC = {auc_val:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()

    if show:
        plt.show()
    
    return auc_val

def plot_pr_curve(y_true, y_scores, label=None, show=True):
    recall_vals, precision_vals = pr_curve(y_true, y_scores)
    auc_val = auc(recall_vals[::-1], precision_vals[::-1])  # orden ascendente en recall
    plt.plot(recall_vals, precision_vals, label=label or f"PR AUC = {auc_val:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    
    if show:
        plt.show()
    
    return auc_val


def auc(x, y):
    # Ordenar ambos arreglos según x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    return np.trapz(y_sorted, x_sorted)


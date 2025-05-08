import cupy as cp
import numpy as np
from tqdm import tqdm

def relu(x):
    return cp.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(cp.float32)

def softmax(x):
    exps = cp.exp(x - cp.max(x, axis=1, keepdims=True))  # estabilidad numérica
    return exps / cp.sum(exps, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -cp.log(y_pred[cp.arange(m), y_true])
    return cp.mean(log_likelihood)

def one_hot_encode(y, num_classes):
    return cp.eye(num_classes, dtype=cp.float32)[y]

class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42):
        cp.random.seed(seed)
        self.L = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(self.L):
            w = cp.random.randn(layer_sizes[i], layer_sizes[i+1]) * cp.sqrt(2. / layer_sizes[i])
            b = cp.zeros((1, layer_sizes[i+1]), dtype=cp.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        pre_activations = []

        for i in range(self.L - 1):
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            A = relu(Z)
            pre_activations.append(Z)
            activations.append(A)

        Z = activations[-1] @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)
        pre_activations.append(Z)
        activations.append(A)

        return activations, pre_activations

    def backward(self, X, y, activations, pre_activations):
        grads_w = [None] * self.L
        grads_b = [None] * self.L
        m = X.shape[0]
        y_one_hot = one_hot_encode(y, activations[-1].shape[1])

        delta = (activations[-1] - y_one_hot) / m

        for i in reversed(range(self.L)):
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = cp.sum(delta, axis=0, keepdims=True)

            if i != 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i-1])

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, lr):
        for i in range(self.L):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X_train, y_train, X_val=None, y_val=None, lr=0.01, epochs=50,
            early_stopping=False, patience=10, verbose=False):

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_weights = None
        best_biases = None
        no_improve_epochs = 0

        epoch_iter = tqdm(range(epochs), desc="Training", ncols=100)

        for epoch in epoch_iter:
            activations, pre_acts = self.forward(X_train)
            loss = cross_entropy(y_train, activations[-1])
            acc = self.evaluate(X_train, y_train)

            grads_w, grads_b = self.backward(X_train, y_train, activations, pre_acts)
            self.update_params(grads_w, grads_b, lr)

            history['train_loss'].append(loss.get())  # Convierte a NumPy
            history['train_acc'].append(acc.get())    # Convierte a NumPy

            val_loss = val_acc = None

            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_loss = cross_entropy(y_val, val_probs)
                val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss.get())  # Convierte a NumPy
                history['val_acc'].append(val_acc.get())    # Convierte a NumPy

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break

            if verbose:
                desc = f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}"
                if val_loss is not None:
                    desc += f" - ValLoss: {val_loss:.4f} - ValAcc: {val_acc:.4f}"
                epoch_iter.set_description(desc)

        if early_stopping and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        return history

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        return cp.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return cp.mean(preds == y)
    
    def evaluate_metrics(self, X, y, title_prefix=""):
        """
        Calcula y muestra accuracy, loss y matriz de confusión para un conjunto dado.
        """
        probs = self.predict_proba(X)
        preds = cp.argmax(probs, axis=1)
        acc = cp.mean(preds == y)
        loss = cross_entropy(y, probs)
        cm = self._confusion_matrix(y, preds)

        print(f"{title_prefix} Accuracy: {acc:.4f}")
        print(f"{title_prefix} Cross-Entropy Loss: {loss:.4f}")

        self._plot_confusion_matrix(cm, title=f"{title_prefix} - Matriz de Confusión")

    def _confusion_matrix(self, y_true: cp.ndarray, y_pred: cp.ndarray) -> cp.ndarray:
        num_classes = int(cp.max(y_true).get()) + 1
        matrix = cp.zeros((num_classes, num_classes), dtype=cp.int32)
        for t, p in zip(y_true, y_pred):
            matrix[int(t), int(p)] += 1
        return matrix

    def _plot_confusion_matrix(self, cm: cp.ndarray, title: str = "", figsize=(7, 6)):
        import matplotlib.pyplot as plt

        cm_np = cm.get()  # Convertir toda la matriz una sola vez

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm_np, interpolation='nearest', cmap='Blues')

        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.colorbar(im, ax=ax)

        num_classes = cm_np.shape[0]
        ax.set_xticks(np.arange(num_classes))  # ahora sí, np.arange
        ax.set_yticks(np.arange(num_classes))
        ax.tick_params(axis='x', labelsize=6, rotation=90)
        ax.tick_params(axis='y', labelsize=6)

        fig.tight_layout()
        plt.show()


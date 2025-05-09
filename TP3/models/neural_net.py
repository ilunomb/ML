import cupy as cp
import numpy as np
from tqdm import tqdm
from typing import Optional
from models.constants import *

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
    def __init__(self,
                 layer_sizes,
                 use_batchnorm: bool = DEFAULT_USE_BATCHNORM,
                 dropout_rate: float = DEFAULT_DROPOUT_RATE,
                 use_adam: bool = DEFAULT_USE_ADAM,
                 beta1: float = DEFAULT_BETA1,
                 beta2: float = DEFAULT_BETA2,
                 eps: float = 1e-8,
                 l2_lambda: float = DEFAULT_L2_LAMBDA,
                 scheduler_type: Optional[str] = DEFAULT_SCHEDULER_TYPE,  # 'linear' o 'exponential'
                 final_lr: Optional[float] = DEFAULT_FINAL_LR,
                 seed=RANDOM_SEED):

        cp.random.seed(seed)
        self.L = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.use_adam = use_adam
        self.l2_lambda = l2_lambda
        self.scheduler_type = scheduler_type
        self.final_lr = final_lr

        # Optimizador ADAM
        if self.use_adam:
            self.m_w = [cp.zeros((layer_sizes[i], layer_sizes[i+1])) for i in range(self.L)]
            self.v_w = [cp.zeros_like(w) for w in self.m_w]
            self.m_b = [cp.zeros((1, layer_sizes[i+1])) for i in range(self.L)]
            self.v_b = [cp.zeros_like(b) for b in self.m_b]
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

        # Pesos y biases
        for i in range(self.L):
            w = cp.random.randn(layer_sizes[i], layer_sizes[i+1]) * cp.sqrt(2. / layer_sizes[i])
            b = cp.zeros((1, layer_sizes[i+1]), dtype=cp.float32)
            self.weights.append(w)
            self.biases.append(b)

        # BatchNorm
        if self.use_batchnorm:
            self.gamma = [cp.ones((1, layer_sizes[i+1])) for i in range(self.L - 1)]
            self.beta = [cp.zeros((1, layer_sizes[i+1])) for i in range(self.L - 1)]

        self.running_means = [None] * (self.L - 1)
        self.running_vars = [None] * (self.L - 1)


    def forward(self, X, training=True):
        activations = [X]
        pre_activations = []
        dropout_masks = []
        batchnorm_caches = []

        A = X
        for i in range(self.L - 1):
            Z = A @ self.weights[i] + self.biases[i]

            if self.use_batchnorm:
                mean = cp.mean(Z, axis=0, keepdims=True)
                var = cp.var(Z, axis=0, keepdims=True)
                Z_norm = (Z - mean) / cp.sqrt(var + 1e-5)
                Z = self.gamma[i] * Z_norm + self.beta[i]

                if training:
                    self.running_means[i] = mean
                    self.running_vars[i] = var

                batchnorm_caches.append((Z_norm, mean, var))

            A = relu(Z)

            if self.dropout_rate > 0.0 and training:
                mask = (cp.random.rand(*A.shape) > self.dropout_rate).astype(cp.float32)
                A *= mask
                A /= (1.0 - self.dropout_rate)
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)

            pre_activations.append(Z)
            activations.append(A)

        # Capa de salida (sin dropout ni batchnorm)
        Z = A @ self.weights[-1] + self.biases[-1]
        A = softmax(Z)
        pre_activations.append(Z)
        activations.append(A)

        self._cache = {
            'dropout_masks': dropout_masks,
            'batchnorm_caches': batchnorm_caches if self.use_batchnorm else None
        }

        return activations, pre_activations


    def backward(self, X, y, activations, pre_activations):
        grads_w = [None] * self.L
        grads_b = [None] * self.L
        grads_gamma = [None] * (self.L - 1) if self.use_batchnorm else None
        grads_beta = [None] * (self.L - 1) if self.use_batchnorm else None

        m = X.shape[0]
        y_one_hot = one_hot_encode(y, activations[-1].shape[1])
        delta = (activations[-1] - y_one_hot) / m

        dropout_masks = self._cache['dropout_masks']
        batchnorm_caches = self._cache['batchnorm_caches']

        for i in reversed(range(self.L)):
            grads_w[i] = activations[i].T @ delta + self.l2_lambda * self.weights[i]
            grads_b[i] = cp.sum(delta, axis=0, keepdims=True)

            if i != 0:
                dA = delta @ self.weights[i].T

                if self.dropout_rate > 0.0 and dropout_masks[i - 1] is not None:
                    dA *= dropout_masks[i - 1]
                    dA /= (1.0 - self.dropout_rate)

                dZ = dA * relu_derivative(pre_activations[i - 1])

                if self.use_batchnorm:
                    Z_norm, mean, var = batchnorm_caches[i - 1]
                    std_inv = 1. / cp.sqrt(var + 1e-5)

                    dZ_norm = dZ * self.gamma[i - 1]
                    dvar = cp.sum(dZ_norm * (pre_activations[i - 1] - mean) * -0.5 * std_inv**3, axis=0)
                    dmean = cp.sum(dZ_norm * -std_inv, axis=0) + dvar * cp.mean(-2. * (pre_activations[i - 1] - mean), axis=0)

                    dZ = dZ_norm * std_inv + dvar * 2. * (pre_activations[i - 1] - mean) / m + dmean / m

                    grads_gamma[i - 1] = cp.sum(dZ * Z_norm, axis=0, keepdims=True)
                    grads_beta[i - 1] = cp.sum(dZ, axis=0, keepdims=True)

                delta = dZ
            else:
                delta = None

        if self.use_batchnorm:
            self._cache['grads_gamma'] = grads_gamma
            self._cache['grads_beta'] = grads_beta

        return grads_w, grads_b


    def update_params(self, grads_w, grads_b, lr, t=1):
        for i in range(self.L):
            if self.use_adam:
                # Pesos
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)

                m_hat_w = self.m_w[i] / (1 - self.beta1**t)
                v_hat_w = self.v_w[i] / (1 - self.beta2**t)

                self.weights[i] -= lr * m_hat_w / (cp.sqrt(v_hat_w) + self.eps)

                # Biases
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

                m_hat_b = self.m_b[i] / (1 - self.beta1**t)
                v_hat_b = self.v_b[i] / (1 - self.beta2**t)

                self.biases[i] -= lr * m_hat_b / (cp.sqrt(v_hat_b) + self.eps)

            else:
                # SGD
                self.weights[i] -= lr * grads_w[i]
                self.biases[i] -= lr * grads_b[i]

        # Actualización de parámetros de BatchNorm
        if self.use_batchnorm:
            grads_gamma = self._cache.get('grads_gamma', [])
            grads_beta = self._cache.get('grads_beta', [])
            for i in range(self.L - 1):
                if grads_gamma[i] is not None:
                    self.gamma[i] -= lr * grads_gamma[i]
                    self.beta[i] -= lr * grads_beta[i]


    def train(self,
            X_train, y_train,
            X_val=None, y_val=None,
            lr=DEFAULT_LR, final_lr=DEFAULT_FINAL_LR,
            epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
            early_stopping=DEFAULT_EARLY_STOPPING, patience=DEFAULT_PATIENCE,
            verbose=False,
            show_progress=True):

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_weights = None
        best_biases = None
        best_gamma = None
        best_beta = None
        no_improve_epochs = 0

        N = X_train.shape[0]
        t_global = 1  # paso para ADAM y scheduler

        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", ncols=100)
        for epoch in epoch_iter:
            # Shuffle para mini-batch
            indices = cp.random.permutation(N)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0

            # Mini-batch training
            bs = int(batch_size) if batch_size is not None else N

            for start in range(0, N, bs):
                end = start + (batch_size or N)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                activations, pre_acts = self.forward(X_batch, training=True)
                loss = cross_entropy(y_batch, activations[-1])
                acc = self.evaluate(X_batch, y_batch)

                grads_w, grads_b = self.backward(X_batch, y_batch, activations, pre_acts)

                # Scheduler
                current_lr = lr
                if self.scheduler_type == 'linear' and final_lr is not None:
                    current_lr = lr + (final_lr - lr) * (epoch / epochs)
                elif self.scheduler_type == 'exponential' and final_lr is not None:
                    decay_rate = (final_lr / lr) ** (1 / epochs)
                    current_lr = lr * (decay_rate ** epoch)

                self.update_params(grads_w, grads_b, current_lr, t=t_global)
                t_global += 1

                epoch_loss += loss.get()
                epoch_acc += acc.get()
                num_batches += 1

            # Promedio por batch
            epoch_loss /= num_batches
            epoch_acc /= num_batches
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_loss = cross_entropy(y_val, val_probs).get()
                val_acc = self.evaluate(X_val, y_val).get()

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = [w.copy() for w in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        if self.use_batchnorm:
                            best_gamma = [g.copy() for g in self.gamma]
                            best_beta = [b.copy() for b in self.beta]
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break

            if verbose:
                desc = f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
                if X_val is not None and y_val is not None:
                    desc += f" - ValLoss: {val_loss:.4f} - ValAcc: {val_acc:.4f}"
                print(desc)

        # Restaurar los mejores pesos si hubo early stopping
        if early_stopping and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if self.use_batchnorm:
                self.gamma = best_gamma
                self.beta = best_beta

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


import numpy as np
import pandas as pd
from tqdm import tqdm

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000, l2=0.0, multi_class='binary', cost_re_weighting=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.multi_class = multi_class
        self.cost_re_weighting = cost_re_weighting
        self.W = None
        self.b = None
        self.classes_ = None

    def _sigmoid(self, z):
        z = np.array(z, dtype=np.float64)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y):
        self.classes_ = np.unique(y)
        y_index = np.array([np.where(self.classes_ == c)[0][0] for c in y])
        one_hot = np.zeros((len(y), len(self.classes_)))
        one_hot[np.arange(len(y)), y_index] = 1
        return one_hot


    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m, n = X.shape
        self.classes_ = np.unique(y)

        if self.multi_class == 'binary':
            self.W = np.zeros(n, dtype=np.float64)
            self.b = 0.0

            if self.cost_re_weighting:
                class_counts = np.bincount(y.astype(int))
                total = len(y)
                weights = np.zeros_like(y, dtype=np.float64)
                for i, c in enumerate(self.classes_):
                    weights[y == c] = total / (2 * class_counts[int(c)])
            else:
                weights = np.ones_like(y, dtype=np.float64)

            for _ in range(self.n_iter):
                z = np.dot(X, self.W) + self.b
                y_pred = self._sigmoid(z)

                error = y_pred - y
                weighted_error = weights * error

                dw = (1 / m) * np.dot(X.T, weighted_error) + (self.l2 / m) * self.W
                db = (1 / m) * np.sum(weighted_error)

                self.W -= self.lr * dw
                self.b -= self.lr * db

        elif self.multi_class == 'multinomial':
            k = len(self.classes_)
            self.W = np.zeros((n, k), dtype=np.float64)
            self.b = np.zeros((1, k), dtype=np.float64)

            y_one_hot = self._one_hot(y)

            for _ in range(self.n_iter):
                logits = np.dot(X, self.W) + self.b
                probs = self._softmax(logits)

                error = probs - y_one_hot
                grad_W = (1 / m) * np.dot(X.T, error) + (self.l2 / m) * self.W
                grad_b = (1 / m) * np.sum(error, axis=0, keepdims=True)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

        else:
            raise ValueError("multi_class debe ser 'binary' o 'multinomial'.")

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.multi_class == 'binary':
            z = np.dot(X, self.W) + self.b
            return self._sigmoid(z)
        elif self.multi_class == 'multinomial':
            logits = np.dot(X, self.W) + self.b
            return self._softmax(logits)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.multi_class == 'binary':
            return (proba >= 0.5).astype(int)
        elif self.multi_class == 'multinomial':
            return self.classes_[np.argmax(proba, axis=1)]

class LDA:
    def __init__(self):
        self.means_ = {}
        self.priors_ = {}
        self.covariance_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.covariance_ = np.zeros((n_features, n_features))
        self.means_ = {}
        self.priors_ = {}

        for cls in self.classes_:
            X_c = X[y == cls]
            self.means_[cls] = np.mean(X_c, axis=0)
            self.priors_[cls] = X_c.shape[0] / X.shape[0]
            self.covariance_ += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        self.covariance_ /= (X.shape[0] - len(self.classes_))  # Pooled covariance

    def _discriminant_function(self, X, mean, prior, cov_inv):
        return X @ cov_inv @ mean - 0.5 * mean.T @ cov_inv @ mean + np.log(prior)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        cov_inv = np.linalg.inv(self.covariance_)
        scores = np.array([
            self._discriminant_function(X, self.means_[cls], self.priors_[cls], cov_inv)
            for cls in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=0)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        cov_inv = np.linalg.inv(self.covariance_)
        scores = np.array([
            self._discriminant_function(X, self.means_[cls], self.priors_[cls], cov_inv)
            for cls in self.classes_
        ])
        exp_scores = np.exp(scores - np.max(scores, axis=0))  # for numerical stability
        return (exp_scores / np.sum(exp_scores, axis=0)).T

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _best_split(self, X, y):
        m, n = X.shape
        best_gain, best_feature, best_threshold = -1, None, None
        parent_entropy = self._entropy(y)

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                p = len(left) / m
                gain = parent_entropy - p * self._entropy(left) - (1 - p) * self._entropy(right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth == self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)

    def _predict_row(self, row, node):
        if node['leaf']:
            return node['class']
        if row[node['feature']] <= node['threshold']:
            return self._predict_row(row, node['left'])
        else:
            return self._predict_row(row, node['right'])

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        # Convertir a NumPy si es pandas
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy().ravel()

        self.classes_ = np.unique(y)

        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Árboles"):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
        
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_trees, n_samples)
        n_samples = X.shape[0]

        # Get all possible classes (from training time if you want to preserve consistent order)
        all_classes = np.unique(np.concatenate([tree.predict(X) for tree in self.trees]))
        self.classes_ = all_classes  # save them for later

        # Compute class probabilities
        proba = np.zeros((n_samples, len(all_classes)))
        for i, cls in enumerate(self.classes_):
            proba[:, i] = np.mean(tree_preds == cls, axis=0)

        return proba



import numpy as np

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
        n_classes = len(np.unique(y))
        one_hot = np.zeros((len(y), n_classes))
        for i, c in enumerate(y):
            one_hot[i, int(c)] = 1
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
            return np.argmax(proba, axis=1)

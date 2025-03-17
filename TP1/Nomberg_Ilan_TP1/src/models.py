import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, X, y, fit_intercept=True, L1=0.0, L2=0.0):
        self.fit_intercept = fit_intercept
        self.L1 = L1
        self.L2 = L2
        self.coef_ = None
        self.intercept_ = None
        self.X = X
        self.y = y
        self.training_method = None
        self.feature_names = X.columns

    def pinv_fit(self):
        X, y = self.X, self.y
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(X.shape[1])
        I[0, 0] = 0  # Do not regularize the intercept term
        self.coef_ = np.linalg.inv(X.T @ X + self.L2 * I) @ X.T @ y
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        self.training_method = 'pinv'
    
    def gradient_descent_fit(self, learning_rate=0.01, n_iterations=1000):
        X, y = self.X, self.y
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.random.rand(X.shape[1])
        for _ in range(n_iterations):
            gradient = X.T @ (X @ self.coef_ - y) / X.shape[0] + self.L2 * self.coef_ + self.L1 * np.sign(self.coef_)
            gradient[0] -= self.L2 * self.coef_[0] + self.L1 * np.sign(self.coef_[0])  # Do not regularize the intercept term
            self.coef_ -= learning_rate * gradient
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        self.training_method = 'gradient_descent'

    def predict(self, X_predict):
        if self.fit_intercept:
            X_predict = np.c_[np.ones(X_predict.shape[0]), X_predict]
        return X_predict @ np.r_[self.intercept_, self.coef_] if self.fit_intercept else X_predict @ self.coef_

    def loss(self, X_predict, y, metric):
        return metric.calculate(y, self.predict(X_predict))
    
    def print_model(self):
        print(f"Trained using {self.training_method} method")
        data = {'Feature': ['Intercept'] + list(self.feature_names), 'Coefficient': [self.intercept_] + list(self.coef_)}
        df = pd.DataFrame(data)
        print(f"{df.to_string(index=False)} \n")
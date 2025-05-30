import numpy as np
from models.KMeans import KMeans
from typing import Optional

class GMM:
    """
    Implementación de Gaussian Mixture Model (GMM) con EM y múltiples inicializaciones.
    """

    def __init__(self, n_components: int, max_iter: int = 1000, tol: float = 1e-4,
                 random_state: int = 42, n_init: int = 10,
                 means_init: Optional[np.ndarray] = None):
        """
        Args:
            n_components (int): número de gaussianas (clusters)
            max_iter (int): máximo número de iteraciones del algoritmo EM
            tol (float): tolerancia para cambio en log-verosimilitud
            random_state (int): semilla para reproducibilidad
            n_init (int): cantidad de inicializaciones para elegir la mejor solución
            means_init (Optional[np.ndarray]): medias iniciales (K, n_features)
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.means_init = means_init

        # Final results
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.resp_ = None
        self.log_likelihood_ = None

    def _gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm_const * exp_term

    def _e_step(self, X):
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])

        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma /= gamma_sum
        return gamma

    def _m_step(self, X, gamma):
        n_samples = X.shape[0]
        Nk = np.sum(gamma, axis=0)

        self.weights_ = Nk / n_samples
        self.means_ = (gamma.T @ X) / Nk[:, np.newaxis]
        self.covariances_ = []

        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov_k = (gamma[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            self.covariances_.append(cov_k)

        self.covariances_ = np.array(self.covariances_)

    def _compute_log_likelihood(self, X):
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(likelihood))

    def fit(self, X):
        n_samples, n_features = X.shape
        best_ll = -np.inf

        for init in range(self.n_init):
            np.random.seed(self.random_state + init)

            if self.means_init is not None:
                means = self.means_init
                labels = np.argmin(np.linalg.norm(X[:, None] - means, axis=2), axis=1)
            else:
                kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state + init)
                kmeans.fit(X)
                means = kmeans.centroids
                labels = kmeans.labels_

            weights = np.bincount(labels, minlength=self.n_components) / n_samples
            covariances = np.array([
                np.cov(X[labels == k].T) + np.eye(n_features) * 1e-6
                for k in range(self.n_components)
            ])

            prev_ll = None
            for _ in range(self.max_iter):
                self.weights_ = weights
                self.means_ = means
                self.covariances_ = covariances

                gamma = self._e_step(X)
                self._m_step(X, gamma)
                ll = self._compute_log_likelihood(X)

                if prev_ll is not None and np.abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

                # actualizar valores para la próxima iteración
                weights = self.weights_
                means = self.means_
                covariances = self.covariances_

            if ll > best_ll:
                best_ll = ll
                self.log_likelihood_ = ll
                self.weights_ = weights
                self.means_ = means
                self.covariances_ = covariances
                self.resp_ = gamma

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

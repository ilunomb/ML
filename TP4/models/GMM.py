import numpy as np
from models.KMeans import KMeans
from typing import Optional

class GMM:
    """
    Implementación de Gaussian Mixture Model (GMM) con EM desde cero.
    """

    def __init__(self, n_components: int, max_iter: int = 1000, tol: float = 1e-4,
                 random_state: int = 42, means_init: Optional[np.ndarray] = None):
        """
        Args:
            n_components (int): número de gaussianas (clusters)
            max_iter (int): máximo número de iteraciones del algoritmo EM
            tol (float): tolerancia para cambio en log-verosimilitud
            random_state (int): semilla para reproducibilidad
            means_init (Optional[np.ndarray]): centroides iniciales (K, n_features)
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_init = means_init

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.resp_ = None
        self.log_likelihood_ = None

    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Evalúa la densidad multivariada normal para cada muestra.

        Returns:
            np.ndarray: densidad evaluada en cada punto (n_samples,)
        """
        n = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm_const * exp_term

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: calcula la matriz de responsabilidades.

        Returns:
            np.ndarray: responsabilidades gamma_ik (n_samples, n_components)
        """
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])

        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma /= gamma_sum  # normalizar

        return gamma

    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """
        M-step: actualiza pesos, medias y covarianzas.
        """
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

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Calcula la log-verosimilitud total del modelo.
        """
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(likelihood))

    def fit(self, X: np.ndarray) -> None:
        """
        Ajusta el modelo GMM a los datos usando EM.
        """
        n_samples, n_features = X.shape

        if self.means_init is not None:
            # Usar medios iniciales proporcionados
            self.means_ = self.means_init
            labels = np.argmin(np.linalg.norm(X[:, None] - self.means_, axis=2), axis=1)
        else:
            # Inicialización con KMeans
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.centroids
            labels = kmeans.labels_

        self.weights_ = np.bincount(labels, minlength=self.n_components) / n_samples
        self.covariances_ = np.array([
            np.cov(X[labels == k].T) + np.eye(n_features) * 1e-6
            for k in range(self.n_components)
        ])

        prev_ll = None
        for _ in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)
            ll = self._compute_log_likelihood(X)
            if prev_ll is not None and np.abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = ll
        self.resp_ = gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Asigna cada muestra al cluster con mayor responsabilidad.

        Returns:
            np.ndarray: etiquetas de cluster
        """
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

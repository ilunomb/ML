import numpy as np
from typing import Tuple


class KMeans:
    """
    Implementación desde cero del algoritmo K-Means.
    """

    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        """
        Parámetros:
            n_clusters (int): número de clusters (K)
            max_iter (int): cantidad máxima de iteraciones
            tol (float): tolerancia para convergencia
            random_state (int): semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Ejecuta el algoritmo K-means sobre los datos.

        Args:
            X (np.ndarray): datos de entrada de forma (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        n_samples, _ = X.shape

        # Inicialización aleatoria de centroides
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            # Asignación de clusters
            distances = self._euclidean_distance(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Cálculo de nuevos centroides
            new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                                      for k in range(self.n_clusters)])

            # Verificación de convergencia
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        # Guardar resultados finales
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Asigna un cluster a cada dato de entrada.

        Args:
            X (np.ndarray): datos de entrada

        Returns:
            np.ndarray: etiquetas de cluster
        """
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _euclidean_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclídea entre cada par de puntos.

        Returns:
            np.ndarray: matriz (n_samples, n_clusters) de distancias
        """
        return np.linalg.norm(X[:, np.newaxis] - Y, axis=2)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calcula la suma de distancias cuadradas dentro de cada cluster (inercia).

        Returns:
            float: inercia total
        """
        return np.sum([np.sum((X[labels == k] - self.centroids[k]) ** 2)
                       for k in range(self.n_clusters)])

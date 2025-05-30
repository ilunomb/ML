import numpy as np
from typing import Optional

class KMeans:
    """
    Implementación desde cero del algoritmo K-Means con múltiples inicializaciones (n_init).
    """

    def __init__(
        self,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42
    ):
        """
        Parámetros:
            n_clusters (int): número de clusters (K)
            n_init (int): número de inicializaciones aleatorias
            max_iter (int): cantidad máxima de iteraciones por inicialización
            tol (float): tolerancia para convergencia
            random_state (int): semilla base para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: np.ndarray) -> None:
        """
        Ejecuta el algoritmo K-means sobre los datos con múltiples inicializaciones
        y guarda la mejor solución (mínima inercia).

        Args:
            X (np.ndarray): datos de entrada de forma (n_samples, n_features)
        """
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        n_samples, _ = X.shape

        for init in range(self.n_init):
            np.random.seed(self.random_state + init)
            # Inicialización aleatoria de centroides
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices].copy()

            for _ in range(self.max_iter):
                # Asignación de clusters
                distances = self._euclidean_distance(X, centroids)
                labels = np.argmin(distances, axis=1)
                # Cálculo de nuevos centroides
                new_centroids = np.array([
                    X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                    for k in range(self.n_clusters)
                ])
                # Verificación de convergencia
                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if shift < self.tol:
                    break

            # Cálculo de inercia para esta inicialización
            inertia = np.sum([
                np.sum((X[labels == k] - centroids[k]) ** 2)
                for k in range(self.n_clusters)
            ])

            # Selección de la mejor solución
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()

        # Guardar la mejor solución encontrada
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Asigna un cluster a cada dato de entrada utilizando los centroides finales.

        Args:
            X (np.ndarray): datos de entrada de forma (n_samples, n_features)

        Returns:
            np.ndarray: etiquetas de cluster
        """
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def _euclidean_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclídea entre cada par de puntos.

        Args:
            X (np.ndarray): matriz (n_samples, n_features)
            Y (np.ndarray): matriz (n_clusters, n_features)

        Returns:
            np.ndarray: matriz (n_samples, n_clusters) de distancias
        """
        return np.linalg.norm(X[:, np.newaxis] - Y, axis=2)


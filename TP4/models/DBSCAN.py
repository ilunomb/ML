# DBSCAN desde cero
import numpy as np
from collections import deque
from typing import Optional

class DBSCAN:
    """
    Implementación del algoritmo DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    """

    def __init__(self, eps: float, min_samples: int):
        """
        Args:
            eps (float): radio de la vecindad
            min_samples (int): número mínimo de puntos para formar una región densa
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X: np.ndarray):
        """
        Ejecuta DBSCAN sobre el dataset X.

        Args:
            X (np.ndarray): datos de entrada, forma (n_samples, n_features)
        """
        n = X.shape[0]
        self.labels_ = -np.ones(n, dtype=int)  # -1 representa ruido
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # ruido
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def _region_query(self, X: np.ndarray, idx: int) -> list:
        """Encuentra los índices de los puntos dentro de eps de X[idx]."""
        dists = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(dists <= self.eps)[0])

    def _expand_cluster(self, X: np.ndarray, idx: int, neighbors: list, cluster_id: int, visited: np.ndarray):
        """Expande el cluster desde el punto idx."""
        self.labels_[idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                j_neighbors = self._region_query(X, j)
                if len(j_neighbors) >= self.min_samples:
                    queue.extend(j_neighbors)
            if self.labels_[j] == -1:
                self.labels_[j] = cluster_id

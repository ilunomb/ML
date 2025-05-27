import numpy as np
from typing import Optional


class PCA:
    """
    Implementación desde cero de Análisis de Componentes Principales (PCA).
    """
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X: np.ndarray) -> None:
        """
        Ajusta el modelo PCA al dataset X.

        Args:
            X (np.ndarray): Datos de entrada de forma (n_samples, n_features).
        """
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # Ordenar en orden descendente de autovalores
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vals, eig_vecs = eig_vals[sorted_idx], eig_vecs[:, sorted_idx]

        self.explained_variance_ = eig_vals
        self.components_ = eig_vecs[:, :self.n_components] if self.n_components else eig_vecs

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyecta los datos centrados al subespacio de componentes principales.
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruye los datos a partir del subespacio reducido.
        """
        return np.dot(X_transformed, self.components_.T) + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calcula el error cuadrático medio de reconstrucción.
        """
        X_proj = self.transform(X)
        X_reconstructed = self.inverse_transform(X_proj)
        return np.mean((X - X_reconstructed) ** 2)
    
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Retorna el porcentaje de varianza explicada por cada componente.
        """
        return self.explained_variance_ / np.sum(self.explained_variance_)


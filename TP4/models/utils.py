import numpy as np
from models.KMeans import KMeans
from models.GMM import GMM
from models.DBSCAN import DBSCAN
from models.PCA import PCA
from models.VAE import VAE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import List, Dict, Optional
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader


def normalize(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def kmeans_elbow_analysis(X: np.ndarray, K_range: range, random_state: int = 0, plot: bool = True,
                          improvement_threshold: float = 0.05, patience: int = 3) -> int:
    """
    Ejecuta K-Means para múltiples valores de K y elige el K óptimo usando un umbral de mejora
    y una cantidad de paciencia consecutiva para estabilización.

    Args:
        X (np.ndarray): datos normalizados
        K_range (range): rango de valores de K a probar (ej. range(1, 21))
        random_state (int): semilla
        plot (bool): si se desea mostrar el gráfico
        improvement_threshold (float): mejora relativa mínima para considerar útil otro cluster
        patience (int): cantidad de K consecutivos con baja mejora antes de detenerse

    Returns:
        int: valor de K óptimo
    """
    inertias = []
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(X)
        inertias.append(model.inertia_)

    improvements = [(inertias[i - 1] - inertias[i]) / inertias[i - 1] for i in range(1, len(inertias))]

    low_count = 0
    k_opt_index = 0

    for i, imp in enumerate(improvements):
        if imp < improvement_threshold:
            low_count += 1
            if low_count >= patience:
                break
        else:
            low_count = 0
            k_opt_index = i + 1  # +1 porque improvements está desfasado con respecto a K

    k_opt = K_range[k_opt_index]

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, inertias, marker='o', linestyle='--', label="L(K)")
        plt.axvline(k_opt, color='red', linestyle=':', label=f"K óptimo = {k_opt}")
        plt.title("Curva del Codo - KMeans (umbral + paciencia)")
        plt.xlabel("Número de clusters (K)")
        plt.ylabel("Suma de distancias (L)")
        plt.grid(True)
        plt.legend()
        plt.xticks(K_range)
        plt.tight_layout()
        plt.show()

    return k_opt


def plot_kmeans_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                         title: str = "Clusters - KMeans") -> None:
    """
    Grafica los datos etiquetados por cluster y los centroides, con estética uniforme.

    Args:
        X (np.ndarray): datos normalizados (n_samples, 2)
        labels (np.ndarray): etiquetas de cluster para cada punto
        centroids (np.ndarray): centroides del modelo (n_clusters, 2)
        title (str): título del gráfico
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    k = np.max(labels) + 1
    colors = cm.get_cmap("tab20", k)

    plt.figure(figsize=(7, 7))

    for i in range(k):
        plt.scatter(
            X[labels == i, 0], X[labels == i, 1],
            label=f"Cluster {i}",
            s=40, alpha=0.7,
            edgecolors='black', linewidths=0.5,
            c=[colors(i)]
        )

    # Centroides
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='black', marker='X', s=100, linewidths=1, label="Centroides"
    )

    plt.title(title)
    plt.xlabel("x₁ (normalizado)")
    plt.ylabel("x₂ (normalizado)")
    plt.xticks([]), plt.yticks([])
    plt.axis("equal")
    plt.tight_layout()

    # Leyenda a la derecha
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Etiquetas",
        fontsize=9
    )
    plt.show()


def gmm_log_likelihood_curve(X: np.ndarray, K_range: range, random_state: int = 0, plot: bool = True,
                       improvement_threshold: float = 0.05, patience: int = 2) -> int:
    """
    Determina el número óptimo de componentes para GMM usando mejora relativa de log-verosimilitud.

    Args:
        X (np.ndarray): datos normalizados
        K_range (range): rango de K a evaluar (e.g., range(1, 16))
        random_state (int): semilla para reproducibilidad
        plot (bool): si se desea mostrar la curva
        improvement_threshold (float): mejora mínima esperada entre K y K+1
        patience (int): cantidad de mejoras consecutivas bajas para detenerse

    Returns:
        int: K óptimo estimado
    """

    log_likelihoods = []
    for k in K_range:
        gmm = GMM(n_components=k, random_state=random_state)
        gmm.fit(X)
        log_likelihoods.append(gmm.log_likelihood_)

    improvements = [(log_likelihoods[i] - log_likelihoods[i - 1]) / abs(log_likelihoods[i - 1])
                    for i in range(1, len(log_likelihoods))]

    low_count = 0
    k_opt_index = 0

    for i, imp in enumerate(improvements):
        if imp < improvement_threshold:
            low_count += 1
            if low_count >= patience:
                break
        else:
            low_count = 0
            k_opt_index = i + 1  # +1 por corrimiento

    k_opt = K_range[k_opt_index]

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(K_range, log_likelihoods, marker='o', linestyle='--', color='darkblue', label='Log-verosimilitud')
        plt.axvline(k_opt, color='red', linestyle=':', label=f"K óptimo = {k_opt}")
        plt.title("Curva del Codo - GMM (log-verosimilitud)")
        plt.xlabel("Número de componentes (K)")
        plt.ylabel("Log-verosimilitud")
        plt.grid(True)
        plt.legend()
        plt.xticks(K_range)
        plt.tight_layout()
        plt.show()

    return k_opt


def plot_gmm_clusters(X: np.ndarray, labels: np.ndarray, means: np.ndarray, covariances: np.ndarray,
                      title: str = "Clusters - GMM") -> None:
    """
    Plotea los datos etiquetados por cluster, los centroides y elipses gaussianas, con estética uniforme.

    Args:
        X (np.ndarray): datos normalizados (n_samples, 2)
        labels (np.ndarray): etiquetas de cluster (n_samples,)
        means (np.ndarray): medias de las gaussianas (n_components, 2)
        covariances (np.ndarray): matrices de covarianza (n_components, 2, 2)
        title (str): título del gráfico
    """
    n_clusters = np.max(labels) + 1
    colors = cm.get_cmap("tab20", n_clusters)

    plt.figure(figsize=(7, 7))

    for i in range(n_clusters):
        plt.scatter(
            X[labels == i, 0], X[labels == i, 1],
            s=40, alpha=0.7, edgecolors='black',
            linewidths=0.5, label=f"Cluster {i}",
            c=[colors(i)]
        )

        # Dibujar la elipse 1σ
        cov = covariances[i]
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2 * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        ellipse = Ellipse(
            xy=means[i], width=width, height=height, angle=angle,
            edgecolor='black', facecolor='none', linestyle='--', linewidth=1
        )
        plt.gca().add_patch(ellipse)

    # Centroides
    plt.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=100, linewidths=1, label="Centroides")

    plt.title(title)
    plt.xlabel("x₁ (normalizado)")
    plt.ylabel("x₂ (normalizado)")
    plt.xticks([]), plt.yticks([])
    plt.axis("equal")
    plt.tight_layout()

    # Leyenda a la derecha
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Etiquetas",
        fontsize=9
    )
    plt.show()

def grid_search_dbscan(
    X: np.ndarray,
    eps_list: List[float],
    min_samples_list: List[int]
) -> List[Dict]:
    """
    Ejecuta DBSCAN para múltiples combinaciones de parámetros, grafica resultados con estilo uniforme,
    y devuelve estadísticas clave para cada combinación.

    Args:
        X (np.ndarray): datos de entrada
        eps_list (List[float]): lista de valores de ε a explorar
        min_samples_list (List[int]): lista de valores de min_samples a explorar

    Returns:
        List[Dict]: estadísticas por configuración (eps, min_samples, n_clusters, n_noise)
    """
    fig, axs = plt.subplots(len(eps_list), len(min_samples_list), figsize=(4 * len(min_samples_list), 4 * len(eps_list)))
    results = []
    total = len(eps_list) * len(min_samples_list)
    colors = cm.get_cmap("tab20", 20)

    with tqdm(total=total, desc="Grid DBSCAN") as pbar:
        for i, eps in enumerate(eps_list):
            for j, min_samples in enumerate(min_samples_list):
                model = DBSCAN(eps=eps, min_samples=min_samples)
                model.fit(X)
                labels = model.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = np.sum(labels == -1)

                # Guardar estadísticas
                results.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise
                })

                # Gráfico con estilo uniforme
                ax = axs[i][j] if len(eps_list) > 1 else axs[j]  # soporte para 1D
                unique_labels = set(labels)

                for label in unique_labels:
                    mask = labels == label
                    if label == -1:
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c='lightgray', s=20, edgecolors='none', alpha=0.5
                        )
                    else:
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            s=40,
                            c=[colors(label)],
                            edgecolors='black',
                            linewidths=0.5,
                            alpha=0.7
                        )

                ax.set_title(f"ε={eps}, min_pts={min_samples}\nk={n_clusters}", fontsize=10)
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_xlabel("x₁"), ax.set_ylabel("x₂")

                pbar.update(1)

    plt.tight_layout()
    plt.show()
    return results


def plot_dbscan_clusters(X: np.ndarray, labels: np.ndarray, title: Optional[str] = None):
    """
    Grafica los clusters obtenidos por DBSCAN con estilo mejorado.

    Args:
        X (np.ndarray): datos de entrada
        labels (np.ndarray): etiquetas de clustering (-1 es ruido)
        title (str, optional): título del gráfico
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = cm.get_cmap("tab20", max(len(unique_labels), 2))

    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # Ruido: puntos gris claro sin borde
            plt.scatter(
                X[mask, 0], X[mask, 1],
                c='lightgray', s=20, edgecolors='none', alpha=0.5, label="Ruido"
            )
        else:
            plt.scatter(
                X[mask, 0], X[mask, 1],
                s=40,
                c=[colors(label)],
                edgecolors='black',
                linewidths=0.5,
                alpha=0.7,
                label=f"Cluster {label}"
            )

    plt.title(title or f"DBSCAN: {n_clusters} clusters")
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.xlabel("x₁ (normalizado)")
    plt.ylabel("x₂ (normalizado)")

    # Ubicar la leyenda fuera del gráfico
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Etiquetas"
    )
    plt.show()


def plot_reconstructions_vae(
    model,
    X_tensor: torch.Tensor,
    indices: list[int],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    title: str = "Reconstrucción con VAE desde índices"
):
    """
    Muestra imágenes originales y reconstruidas por VAE a partir de índices dados.

    Args:
        model: VAE entrenado.
        X_tensor (torch.Tensor): Tensor de imágenes (N, 784).
        indices (list[int]): Lista de 10 índices.
        device (str): Dispositivo de ejecución.
        title (str): Título del gráfico.
    """
    assert len(indices) == 10, "Se requieren exactamente 10 índices"

    model.eval()
    model.to(device)

    with torch.no_grad():
        x = X_tensor[indices].to(device).float()  # Selección directa
        x_hat, _, _ = model(x)

    x = x.cpu().numpy()
    x_hat = x_hat.cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(x[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(x_hat[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("VAE", fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_reconstructions_pca(
    pca: PCA,
    X: np.ndarray,
    indices: list[int],
    title: str = "Reconstrucción con PCA desde índices"
):
    """
    Reconstruye y grafica imágenes originales vs reconstruidas desde un conjunto de índices.

    Args:
        pca (PCA): Modelo PCA entrenado.
        X (np.ndarray): Conjunto de datos (n_samples, 784).
        indices (list[int]): Lista de 10 índices a mostrar.
        title (str): Título del gráfico.
    """
    assert len(indices) == 10, "Se requieren exactamente 10 índices"

    originals = X[indices]
    reconstructions = pca.inverse_transform(pca.transform(originals))

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("PCA", fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def get_one_index_per_digit(y: np.ndarray) -> list[int]:
    """
    Retorna una lista de 10 índices, uno para cada dígito del 0 al 9.
    """
    return [np.where(y == d)[0][0] for d in range(10)]

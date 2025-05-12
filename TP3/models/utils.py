import numpy as np
import itertools
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union
import ast
import cupy as cp
from models.neural_net import NeuralNetwork
from models.constants import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_val_test_split(X, y, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, shuffle=True, seed=RANDOM_SEED):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios deben sumar 1"

    np.random.seed(seed)
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

def run_grid_search(X_train, y_train, X_val, y_val,
                    param_grid: Dict[str, List],
                    base_architecture: Tuple[int, ...] = (IMAGE_SIZE, 100, 80, NUM_CLASSES),
                    fixed_params: Dict[str, Any] = {},
                    epochs: int = DEFAULT_EPOCHS,
                    patience: int = DEFAULT_PATIENCE,
                    metric: str = "val_loss"
) -> Tuple[NeuralNetwork, Dict[str, float], Dict[str, any]]:

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    best_score = float('inf') if metric == 'val_loss' else -float('inf')
    best_model = None
    best_history = None
    best_params = None

    grid_bar = tqdm(total=len(combinations), desc="Grid Search", position=0, leave=True)

    for combo in combinations:
        params = dict(zip(keys, combo))
        params.update(fixed_params)

        model = NeuralNetwork(
            layer_sizes=base_architecture,
            use_batchnorm=params.get('use_batchnorm', DEFAULT_USE_BATCHNORM),
            dropout_rate=params.get('dropout_rate', DEFAULT_DROPOUT_RATE),
            use_adam=params.get('use_adam', DEFAULT_USE_ADAM),
            beta1=params.get('beta1', DEFAULT_BETA1),
            beta2=params.get('beta2', DEFAULT_BETA2),
            eps=params.get('eps', 1e-8),
            l2_lambda=params.get('l2_lambda', DEFAULT_L2_LAMBDA),
            scheduler_type=params.get('scheduler_type', DEFAULT_SCHEDULER_TYPE),
            final_lr=params.get('final_lr', DEFAULT_FINAL_LR),
            seed=42
        )

        history = model.train(
            X_train, y_train,
            X_val, y_val,
            lr=params.get('lr', DEFAULT_LR),
            final_lr=params.get('final_lr', DEFAULT_FINAL_LR),
            epochs=epochs,
            batch_size=params.get('batch_size', DEFAULT_BATCH_SIZE),
            early_stopping=DEFAULT_EARLY_STOPPING,
            patience=DEFAULT_PATIENCE,
            verbose=False,
            show_progress=False
        )

        val_metric = history[metric][-1]

        is_better = (val_metric < best_score) if metric == 'val_loss' else (val_metric > best_score)
        if is_better:
            best_score = val_metric
            best_model = model
            best_history = history
            best_params = params.copy()

        grid_bar.update(1)

    grid_bar.close()
    return best_model, best_history, best_params

def sweep_param(X_train, y_train, X_val, y_val,
                param_name: str,
                values: List[Any],
                fixed_params: Dict[str, Any] = {},
                metric: str = "val_loss"
) -> Dict[str, Any]:
    """
    Búsqueda de un único hiperparámetro manteniendo el resto fijo.
    """
    results = []

    for i, val in enumerate(values):
        print(f"\nProbing {param_name} = {val}")
        params = fixed_params.copy()
        params[param_name] = val

        model = NeuralNetwork(
            layer_sizes=params.get("layer_sizes", (IMAGE_SIZE, 100, 80, NUM_CLASSES)),
            use_batchnorm=params.get("use_batchnorm", DEFAULT_USE_BATCHNORM),
            dropout_rate=params.get("dropout_rate", DEFAULT_DROPOUT_RATE),
            use_adam=params.get("use_adam", DEFAULT_USE_ADAM),
            beta1=params.get("beta1", DEFAULT_BETA1),
            beta2=params.get("beta2", DEFAULT_BETA2),
            eps=params.get("eps", 1e-8),
            l2_lambda=params.get("l2_lambda", DEFAULT_L2_LAMBDA),
            scheduler_type=params.get("scheduler_type", DEFAULT_SCHEDULER_TYPE),
            final_lr=params.get("final_lr", DEFAULT_FINAL_LR),
            seed=params.get("seed", RANDOM_SEED)
        )

        history = model.train(
            X_train, y_train,
            X_val, y_val,
            lr=params.get("lr", DEFAULT_LR),
            final_lr=params.get("final_lr", DEFAULT_FINAL_LR),
            epochs=params.get("epochs", DEFAULT_EPOCHS),
            batch_size=params.get("batch_size", DEFAULT_BATCH_SIZE),
            early_stopping=params.get("early_stopping", DEFAULT_EARLY_STOPPING),
            patience=params.get("patience", DEFAULT_PATIENCE),
            verbose=False,
            show_progress=True
        )

        score = history[metric][-1]
        results.append({"param": val, "loss": score})
        print(f"{param_name} = {val} → {metric} = {score:.4f}")

    # Ordenar y devolver el mejor
    return sorted(results, key=lambda x: x["loss"] if metric == "val_loss" else -x["loss"])[0]


def init_hparams_csv(filepath: str):
    defaults = {
        "layer_sizes": str(DEFAULT_LAYER_SIZES),
        "lr": DEFAULT_LR,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "patience": DEFAULT_PATIENCE,
        "use_adam": DEFAULT_USE_ADAM,
        "beta1": DEFAULT_BETA1,
        "beta2": DEFAULT_BETA2,
        "use_batchnorm": DEFAULT_USE_BATCHNORM,
        "dropout_rate": DEFAULT_DROPOUT_RATE,
        "l2_lambda": DEFAULT_L2_LAMBDA,
        "scheduler_type": DEFAULT_SCHEDULER_TYPE,
        "final_lr": DEFAULT_FINAL_LR,
        "early_stopping": DEFAULT_EARLY_STOPPING
    }

    df = pd.DataFrame([defaults])
    df.to_csv(filepath, index=False)
    print(f"Hiperparámetros por defecto guardados en: {filepath}")

def update_hparams_csv(filepath: str, param_name: str, param_value):
    """
    Actualiza el CSV de hiperparámetros con el nuevo valor óptimo para `param_name`.

    Args:
        filepath: Ruta al archivo CSV.
        param_name: Nombre del hiperparámetro a actualizar.
        param_value: Nuevo valor óptimo.
    """
    df = pd.read_csv(filepath)

    if param_name not in df.columns:
        raise ValueError(f"'{param_name}' no es una columna válida del archivo CSV.")

    df.at[0, param_name] = str(param_value) if isinstance(param_value, tuple) else param_value
    df.to_csv(filepath, index=False)
    print(f"\nActualizado '{param_name}' a: {param_value}")
    

def load_hparams_csv(filepath: str, exclude: Union[str, List[str]] = None) -> Dict[str, Any]:
    """
    Carga los hiperparámetros desde un CSV, convirtiendo tipos automáticamente.

    Args:
        filepath: Ruta al archivo CSV.
        exclude: Clave o lista de claves a excluir del resultado.

    Returns:
        Diccionario con los hiperparámetros con tipos corregidos.
    """
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")

    raw_params = df.iloc[0].to_dict()
    parsed_params = {}

    type_casts = {
        "layer_sizes": lambda v: ast.literal_eval(v) if isinstance(v, str) else v,
        "lr": float,
        "final_lr": lambda v: float(v) if not pd.isna(v) else None,
        "epochs": int,
        "batch_size": lambda v: None if pd.isna(v) or v == 'None' else int(v),
        "patience": int,
        "use_adam": lambda v: bool(v),
        "beta1": float,
        "beta2": float,
        "use_batchnorm": lambda v: bool(v),
        "dropout_rate": float,
        "l2_lambda": float,
        "scheduler_type": lambda v: v if pd.notna(v) and v != 'None' else None,
        "early_stopping": lambda v: bool(v)
    }

    for key, value in raw_params.items():
        if exclude:
            if isinstance(exclude, str) and key == exclude:
                continue
            elif isinstance(exclude, list) and key in exclude:
                continue

        cast_fn = type_casts.get(key, lambda x: x)  # fallback: identity
        try:
            parsed_params[key] = cast_fn(value)
        except Exception as e:
            raise ValueError(f"Error casteando '{key}' con valor '{value}': {e}")

    return parsed_params


def sweep_and_update(
    X_train, y_train, X_val, y_val,
    param_name: str,
    values: List[Any],
    csv_path: str = HYPERPARAMS_FILE,
    metric: str = "val_loss"
):
    """
    Realiza un sweep sobre un hiperparámetro y actualiza el CSV con el mejor valor.

    Args:
        param_name: Nombre del hiperparámetro a testear.
        values: Lista de valores a evaluar.
        csv_path: Ruta del archivo CSV con hiperparámetros.
        metric: Métrica a optimizar ('val_loss' o 'val_acc').
    """
    fixed_params = load_hparams_csv(csv_path, exclude=param_name)

    results = sweep_param(
        X_train, y_train, X_val, y_val,
        param_name=param_name,
        values=values,
        fixed_params=fixed_params,
        metric=metric
    )

    best_value = results["param"]
    best_score = results["loss"]

    update_hparams_csv(csv_path, param_name, best_value)

    print(f"\nMejor {param_name}: {best_value} con {metric} = {best_score:.4f}")

def grid_search_and_update(
    X_train, y_train, X_val, y_val,
    param_grid: Dict[str, List],
    csv_path: str = HYPERPARAMS_FILE,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    metric: str = "val_loss"
):
    """
    Ejecuta un grid search sobre un conjunto de hiperparámetros y actualiza el CSV.

    Args:
        param_grid: Diccionario con los hiperparámetros a testear (listas).
        csv_path: Ruta al archivo CSV con hiperparámetros.
        epochs: Cantidad de épocas a entrenar.
        patience: Paciencia para early stopping.
        metric: Métrica a optimizar.
    """
    exclude_keys = list(param_grid.keys())
    fixed_params = load_hparams_csv(csv_path, exclude=exclude_keys)

    model, history, best_params = run_grid_search(
        X_train, y_train,
        X_val, y_val,
        param_grid=param_grid,
        base_architecture=fixed_params["layer_sizes"],
        fixed_params=fixed_params,
        epochs=epochs,
        patience=patience,
        metric=metric
    )

    # Guardar cada parámetro óptimo encontrado en el CSV
    for param, value in best_params.items():
        if param in param_grid:
            update_hparams_csv(csv_path, param, value)

    best_metric = history[metric][-1]
    print(f"Grid Search terminado. Mejor {metric} = {best_metric:.4f}")


def plot_history(history, title="Modelo"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 6))

    ax1.set_title(f"{title} - Loss")
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True)
    plt.show()

    fig, ax2 = plt.subplots(figsize=(7, 6))
    ax2.set_title(f"{title} - Accuracy")
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.show()

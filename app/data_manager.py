# ==================== data_manager.py ====================
"""
Gestión de datos, guardado/carga de resultados y modelos (JSON y CSV).
Soporta datasets CSV, Excel y JSON, incluyendo columnas de texto.
"""
import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
from .utils import ensure_dir
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

RESULTS_DIR = "resultados"
MODELS_DIR = "modelos"
DATASETS_DIR = "datasets"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# -------------------- Lectura de datasets --------------------
def read_dataset(path):
    """Lee dataset desde CSV, Excel o JSON y convierte texto a categorías numéricas"""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    elif path.lower().endswith(".json"):
        df = pd.read_json(path)
    else:
        raise ValueError("Formato no soportado. Use .csv, .xlsx o .json")

    # Convertir columnas de texto a números
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    return df

def detect_xy_from_df(df):
    """
    Detecta automáticamente las columnas de entrada (X) y salida (Y),
    limpiando el dataset y convirtiendo etiquetas o texto en números.
    Soporta CSV/XLSX/JSON con columnas mixtas.
    """

    if df is None or df.empty:
        raise ValueError("El dataset está vacío o no se pudo cargar.")

    # 🔹 1. Elimina filas completamente vacías
    df = df.dropna(how="all")

    # 🔹 2. Convierte todos los encabezados a strings
    df.columns = [str(c) for c in df.columns]

    # 🔹 3. Limpia valores vacíos o no numéricos
    df = df.replace(["?", "NA", "N/A", "None", "null", "-", "--"], np.nan)
    df = df.dropna()

    # 🔹 4. Identifica columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 🔹 5. Si hay columnas no numéricas, intenta convertirlas
    for col in df.columns:
        if col not in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col])
                numeric_cols.append(col)
            except Exception:
                pass  # Si no se puede convertir, la dejamos como texto

    # 🔹 6. La última columna siempre se toma como salida Y
    X_cols = df.columns[:-1].tolist()
    Y_col = df.columns[-1]
    X_raw = df[X_cols]
    Y_raw = df[Y_col]

    # 🔹 7. Convierte entradas a float (las que pueda)
    X = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float)

    # 🔹 8. Convierte salida Y
    classes = None
    if Y_raw.dtype == object or isinstance(Y_raw.iloc[0], str):
        encoder = LabelEncoder()
        Y = encoder.fit_transform(Y_raw.astype(str)).astype(float).reshape(-1, 1)
        classes = list(encoder.classes_)
    else:
        Y = pd.to_numeric(Y_raw, errors="coerce").fillna(0).values.astype(float).reshape(-1, 1)

    meta = {
        "n_patterns": len(df),
        "n_inputs": X.shape[1],
        "n_outputs": Y.shape[1],
        "X_cols": X_cols,
        "Y_col": Y_col,
        "classes": classes,
    }

    return X, Y, meta
def preprocess(X):
    """Normalización Z-score (estandarización)"""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    return Xs, scaler

def split_train_test(X, Y, train_pct=0.7, random_state=None):
    """
    Divide dataset en entrenamiento y prueba.
    train_pct: porcentaje para entrenamiento (0.7 = 70%)
    """
    N = X.shape[0]
    idx = np.arange(N)
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)
    cut = int(N * train_pct)

    X_train, Y_train = X[idx[:cut]], Y[idx[:cut]]
    X_test, Y_test = X[idx[cut:]], Y[idx[cut:]]

    info = {
        "N_total": N,
        "N_train": len(X_train),
        "N_test": len(X_test),
        "pct_train": train_pct * 100,
        "pct_test": (1 - train_pct) * 100
    }

    return X_train, Y_train, X_test, Y_test, info

# -------------------- Guardado de resultados --------------------
def save_results(results_dict, base_name="resultados_rbf"):
    """Guarda resultados en JSON y CSV"""
    json_path = os.path.join(RESULTS_DIR, f"{base_name}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{base_name}.csv")
    
    # Guardar JSON
    json_data = {}
    for k, v in results_dict.items():
        if isinstance(v, np.ndarray):
            json_data[k] = v.tolist()
        else:
            json_data[k] = v
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Guardar CSV de predicciones
    if "Yd_train" in results_dict and "Yr_train" in results_dict:
        df_train = pd.DataFrame({
            "Yd": results_dict["Yd_train"],
            "Yr": results_dict["Yr_train"],
            "Error": np.abs(np.array(results_dict["Yd_train"]) - np.array(results_dict["Yr_train"]))
        })
        df_train.to_csv(csv_path.replace(".csv", "_train.csv"), index=False)
    
    if "Yd_test" in results_dict and "Yr_test" in results_dict:
        df_test = pd.DataFrame({
            "Yd": results_dict["Yd_test"],
            "Yr": results_dict["Yr_test"],
            "Error": np.abs(np.array(results_dict["Yd_test"]) - np.array(results_dict["Yr_test"]))
        })
        df_test.to_csv(csv_path.replace(".csv", "_test.csv"), index=False)
    
    return json_path, csv_path

# -------------------- Guardado de modelos --------------------
def save_model_json_csv(model_obj, scaler, meta, last_eval, base_name="modelo_rbf"):
    """Guarda el modelo en JSON y también exporta centros/pesos a CSV"""
    ensure_dir(MODELS_DIR)

    # --- Conversión segura ---
    def to_list_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_list_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_list_safe(x) for x in obj]
        else:
            return obj

    # --- Guardar JSON ---
    json_path = os.path.join(MODELS_DIR, f"{base_name}.json")
    model_data = {
        "n_centers": int(model_obj.n_centers),
        "centers": to_list_safe(model_obj.centers),
        "weights": to_list_safe(model_obj.weights),
        "scaler_mean": to_list_safe(scaler.mean_) if scaler else None,
        "scaler_std": to_list_safe(scaler.scale_) if scaler else None,
        "meta": to_list_safe(meta),
        "last_eval": to_list_safe(last_eval)
    }

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)

    # --- Guardar también CSV (centros y pesos) ---
    centers_path = os.path.join(MODELS_DIR, f"{base_name}_centros.csv")
    if model_obj.centers is not None:
        pd.DataFrame(model_obj.centers).to_csv(centers_path, index=False)

    weights_path = os.path.join(MODELS_DIR, f"{base_name}_pesos.csv")
    if model_obj.weights is not None:
        pd.DataFrame({"W": model_obj.weights}).to_csv(weights_path, index=False)

    return json_path, centers_path, weights_path


# -------------------- Carga de modelos --------------------
def load_model_json(model_path):
    """
    Carga un modelo entrenado desde JSON o CSV.
    Devuelve: (model, scaler, meta, last_eval)
    """
    from .model_rbf import RBFModel

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".json":
        # Cargar modelo completo
        with open(model_path, "r", encoding="utf8") as f:
            data = json.load(f)

        model = RBFModel(n_centers=int(data["n_centers"]))
        model.centers = np.array(data["centers"]) if data.get("centers") else None
        model.weights = np.array(data["weights"]) if data.get("weights") else None
        model.trained = model.weights is not None

        scaler = None
        if data.get("scaler_mean") and data.get("scaler_std"):
            scaler = StandardScaler()
            scaler.mean_ = np.array(data["scaler_mean"])
            scaler.scale_ = np.array(data["scaler_std"])
            scaler.n_features_in_ = len(scaler.mean_)

        meta = data.get("meta", {})
        last_eval = data.get("last_eval", {})

        return model, scaler, meta, last_eval

    elif ext == ".csv":
        # Cargar pesos o centros desde CSV
        if "centros" in model_path:
            df = pd.read_csv(model_path)
            centers = df.to_numpy()
            model = RBFModel(n_centers=centers.shape[0])
            model.centers = centers
            return model, None, None, None
        elif "pesos" in model_path:
            df = pd.read_csv(model_path)
            model = RBFModel(n_centers=len(df))
            model.weights = df["W"].to_numpy().reshape(-1, 1)
            return model, None, None, None
        else:
            raise ValueError("Archivo CSV no reconocido (use *_centros.csv o *_pesos.csv).")

    else:
        raise ValueError("Formato de modelo no soportado. Use .json o .csv")
    
# -------------------- Guardado / Carga TODO en uno --------------------
def save_full_model(model, scaler, meta, last_eval, X_train, Y_train, X_test, Y_test, A, results, base_name="modelo_completo_rbf"):
    """
    Guarda TODO el estado del modelo (config, centros, pesos, métricas, dataset dividido, matriz A)
    en un único archivo JSON para recargarlo y simular sin reentrenar.
    """
    ensure_dir(MODELS_DIR)
    full_path = os.path.join(MODELS_DIR, f"{base_name}.json")

    def to_list_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: to_list_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_list_safe(x) for x in obj]
        else:
            return obj

    full_data = {
        "config": {
            "n_centers": int(model.n_centers),
            "error_opt": results.get("config", {}).get("error_opt", None),
        },
        "centers": to_list_safe(model.centers),
        "weights": to_list_safe(model.weights),
        "matriz_A": to_list_safe(A),
        "scaler_mean": to_list_safe(scaler.mean_) if scaler else None,
        "scaler_std": to_list_safe(scaler.scale_) if scaler else None,
        "meta": to_list_safe(meta),
        "last_eval": to_list_safe(last_eval),
        "train_data": {
            "X_train": to_list_safe(X_train),
            "Y_train": to_list_safe(Y_train)
        },
        "test_data": {
            "X_test": to_list_safe(X_test),
            "Y_test": to_list_safe(Y_test)
        },
        "metrics": {
            "EG_train": results.get("EG_train"),
            "MAE_train": results.get("MAE_train"),
            "RMSE_train": results.get("RMSE_train"),
            "EG_test": results.get("EG_test"),
            "MAE_test": results.get("MAE_test"),
            "RMSE_test": results.get("RMSE_test"),
            "convergencia_train": results.get("convergencia_train")
        }
    }

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

    return full_path


def load_full_model(path):
    """
    Carga el archivo JSON completo (modelo + resultados + dataset dividido)
    para continuar simulando directamente sin reentrenar.
    """
    from .model_rbf import RBFModel

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = RBFModel(n_centers=data["config"]["n_centers"])
    model.centers = np.array(data["centers"])
    model.weights = np.array(data["weights"])
    model.trained = True

    scaler = None
    if data.get("scaler_mean") and data.get("scaler_std"):
        scaler = StandardScaler()
        scaler.mean_ = np.array(data["scaler_mean"])
        scaler.scale_ = np.array(data["scaler_std"])
        scaler.n_features_in_ = len(scaler.mean_)

    meta = data.get("meta")
    last_eval = data.get("last_eval")

    # Reconstrucción del dataset
    X_train = np.array(data["train_data"]["X_train"])
    Y_train = np.array(data["train_data"]["Y_train"])
    X_test = np.array(data["test_data"]["X_test"])
    Y_test = np.array(data["test_data"]["Y_test"])

    return model, scaler, meta, last_eval, X_train, Y_train, X_test, Y_test

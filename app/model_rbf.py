# ==================== model_rbf.py ====================
"""
Implementación de Red Neuronal RBF (Radial Basis Function)
según el examen práctico de Inteligencia Artificial.
Incluye cálculo paso a paso, matriz A, pesos finales,
errores y soporte para centros configurables.
"""

import numpy as np

EPS = 1e-12  # Evita log(0) o divisiones por 0


# ------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------
def radial_activation_pdf(d):
    """
    Función de activación radial según el PDF:
    FA(d) = d² * ln(d)
    """
    d = np.maximum(d, EPS)
    return (d ** 2) * np.log(d)


def radial_activation_gauss(d, sigma):
    """
    Función gaussiana estándar:
    FA(d) = exp(-||x - c||² / (2σ²))
    """
    sigma = max(sigma, EPS)
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


# ------------------------------------------------------
# CLASE PRINCIPAL
# ------------------------------------------------------
class RBFModel:
    def __init__(self, n_centers=2, sigma=None, random_state=None, use_pdf=False):
        """
        n_centers : cantidad de centros radiales
        sigma     : desviación estándar, si es None se calcula automáticamente
        use_pdf   : si True, usa FA(d)=d²ln(d) (según PDF); si False usa gaussiana
        """
        self.n_centers = int(n_centers)
        self.random_state = np.random.RandomState(random_state)
        self.centers = None
        self.weights = None
        self.sigma = sigma
        self.trained = False
        self.error_general = None
        self.A_matrix = None
        self.use_pdf = use_pdf

    # --------------------------------------------------
    # CONFIGURACIÓN DE CENTROS
    # --------------------------------------------------
    def initialize_centers(self, X):
        """Inicializa centros aleatorios dentro del rango de X"""
        X = np.asarray(X)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        self.centers = self.random_state.uniform(mins, maxs, size=(self.n_centers, X.shape[1]))
        return self.centers

    def add_center(self, coords):
        """Agrega manualmente un nuevo centro"""
        coords = np.asarray(coords).reshape(1, -1)
        if self.centers is None:
            self.centers = coords
        else:
            self.centers = np.vstack([self.centers, coords])
        self.n_centers = self.centers.shape[0]
        return self.centers

    def _auto_sigma(self):
        """Calcula sigma automáticamente según la media de distancias entre centros"""
        if self.centers is None or len(self.centers) < 2:
            return 1.0
        dists = []
        for i in range(len(self.centers)):
            for j in range(i + 1, len(self.centers)):
                dists.append(np.linalg.norm(self.centers[i] - self.centers[j]))
        dists = np.array(dists)
        return np.mean(dists) if len(dists) > 0 else 1.0

    # --------------------------------------------------
    # MATRICES Y ENTRENAMIENTO
    # --------------------------------------------------
    def _compute_distances(self, X):
        """Matriz de distancias euclidianas entre patrones y centros"""
        X = np.asarray(X)
        N, J = X.shape[0], self.centers.shape[0]
        D = np.zeros((N, J))
        for j in range(J):
            D[:, j] = np.linalg.norm(X - self.centers[j], axis=1)
        return D

    def _compute_activations(self, X):
        """Calcula matriz Phi (activaciones)"""
        D = self._compute_distances(X)
        if self.use_pdf:
            Phi = radial_activation_pdf(D)
        else:
            if self.sigma is None:
                self.sigma = self._auto_sigma()
            Phi = radial_activation_gauss(D, self.sigma)
        return Phi

    def train(self, X, Y):
        """
        Entrenamiento por mínimos cuadrados:
        1. Calcula Φ (activaciones)
        2. Agrega bias -> A = [1 | Φ]
        3. Calcula pesos W = (A⁺ * Y)
        """
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)

        # Inicializar centros si no existen
        if self.centers is None:
            idx = self.random_state.choice(len(X), self.n_centers, replace=False)
            self.centers = X[idx]

        # Calcular matriz de activaciones
        Phi = self._compute_activations(X)
        self.A_matrix = Phi.copy()

        # Agregar bias
        A = np.hstack([np.ones((Phi.shape[0], 1)), Phi])

        # Resolver por mínimos cuadrados
        W, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        self.weights = W.flatten()
        self.trained = True

        # Calcular error general
        Yp = A.dot(W)
        self.error_general = np.mean((Y - Yp) ** 2)

        return A, self.weights

    # --------------------------------------------------
    # PREDICCIÓN Y EVALUACIÓN
    # --------------------------------------------------
    def predict(self, X):
        """Predice Yr = A * W"""
        if not self.trained or self.weights is None:
            raise RuntimeError("El modelo no está entrenado.")
        X = np.asarray(X)
        Phi = self._compute_activations(X)
        A = np.hstack([np.ones((Phi.shape[0], 1)), Phi])
        Yr = A.dot(self.weights.reshape(-1, 1)).flatten()
        return Yr

    def evaluate(self, X, Y):
        """Evalúa métricas del modelo"""
        X = np.asarray(X)
        Y = np.asarray(Y).flatten()
        Yr = self.predict(X)
        abs_err = np.abs(Y - Yr)
        EG = np.mean(abs_err)
        MAE = EG
        RMSE = np.sqrt(np.mean((Y - Yr) ** 2))
        return {
            "EG": float(EG),
            "MAE": float(MAE),
            "RMSE": float(RMSE),
            "Yr": Yr,
            "errors_per_pattern": abs_err.tolist()
        }

    def check_convergence(self, eg_value, eg_opt):
        """Verifica convergencia del modelo"""
        conv = eg_value <= eg_opt
        if conv:
            return True, f"✅ CONVERGE (EG={eg_value:.6f} <= {eg_opt})"
        else:
            return False, f"❌ NO CONVERGE (EG={eg_value:.6f} > {eg_opt})"

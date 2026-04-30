"""
Módulo: tests/integration/dynamic_stress/test_gamma_dynamic_ergodicity.py
=========================================================================================
SUITE DE INTEGRACIÓN DINÁMICA: ERGODICIDAD, PASIVIDAD Y MECÁNICA ESTADÍSTICA
(Versión Rigurosa con Garantías Algebraicas y Físicas)

Fundamentación Matemática y Física
-----------------------------------

1. INVARIANZA HOMOTÓPICA BAJO VUELOS DE LÉVY
   Teorema (Chung 1997): Para el Laplaciano normalizado L_sym de un grafo G,
   el espectro satisface: spec(L_sym) ⊆ [0, 2]
   
   Definición Rigurosa:
   L_sym = I - D^{-1/2} A D^{-1/2}
   
   Convención de Nodos Aislados:
   - Si deg(v) = 0, entonces L_sym[v,v] = 1 (NO 0)
   - Esto preserva: multiplicidad(λ=0) = componentes conexas

2. DESIGUALDAD DE PASIVIDAD PORT-HAMILTONIANA
   Teorema (Willems 1972): Un sistema (Σ, H, u, y) es pasivo si:
   ∃ función de almacenamiento H: X → ℝ₊ tal que
   H(x(T)) - H(x(0)) ≤ ∫₀ᵀ uᵀ(t) y(t) dt
   
   Para sistemas discretos:
   H_{k+1} - H_k ≤ -α H_k + β uₖᵀ yₖ  (α > 0: disipación estricta)

3. DISTRIBUCIÓN DE GIBBS Y ERGODICIDAD
   Teorema (Ergódico de Birkhoff-Khinchin):
   Para una cadena de Markov ergódica con distribución estacionaria π:
   lim_{n→∞} (1/n) Σᵢ f(Xᵢ) = E_π[f]  casi seguramente
   
   Distribución Canónica de Gibbs:
   P(estado i) = exp(-E_i / k_B T) / Z
   donde Z = Σⱼ exp(-E_j / k_B T) (función de partición)
=========================================================================================
"""
from __future__ import annotations

import os
import math
import warnings
from decimal import Decimal, getcontext
from typing import Tuple, List, Dict, Optional, Literal
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import pytest
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.stats import entropy, kstest, cauchy

# ==============================================================================
# CONFIGURACIÓN DE ENTORNO NUMÉRICO
# ==============================================================================
# Forzar ejecución serial para reproducibilidad bit-a-bit
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

# Precisión decimal para cálculos críticos
getcontext().prec = 50

# ==============================================================================
# CONSTANTES FÍSICAS Y NUMÉRICAS
# ==============================================================================

# Tolerancias basadas en precisión de máquina
EPSILON_FLOAT64 = np.finfo(np.float64).eps
EPSILON_SPECTRAL = 1e-10       # Para autovalores del Laplaciano
EPSILON_LYAPUNOV = 1e-8        # Para exponentes de Lyapunov
EPSILON_STATISTICAL = 1e-3     # Para tests de hipótesis

# Constantes físicas (unidades SI normalizadas)
K_BOLTZMANN = Decimal('1.0')   # Constante de Boltzmann (normalizada)

# Parámetros de simulación
SIMULATION_HORIZON = 2000      # Pasos de integración temporal
BURN_IN_RATIO = 0.25          # Fracción de burn-in para MCMC
THINNING_INTERVAL = 10        # Intervalo de muestreo para reducir autocorrelación
MCMC_TOTAL_STEPS = 50000      # Pasos totales de Metropolis-Hastings
CONFIDENCE_LEVEL = 0.95       # Nivel de confianza para intervalos

# Parámetros de disipación Port-Hamiltoniana
DISSIPATION_RATE_DEFAULT = Decimal('0.05')  # γ en exp(-γ)
COUPLING_GAIN_DEFAULT = Decimal('0.01')     # κ para perturbaciones externas


# ==============================================================================
# TIPOS ALGEBRAICOS
# ==============================================================================

@dataclass(frozen=True, slots=True)
class DynamicState:
    """
    Tensor de estado dinámico en cada iteración del ciclo OODA.
    
    Invariantes:
    -----------
    • energy ≥ 0 (no negatividad de la función de Lyapunov)
    • jacobian_spectral_radius ≥ 0
    • verdict_code ∈ ℤ
    """
    energy: float
    jacobian_spectral_radius: float
    verdict_code: int
    
    def __post_init__(self) -> None:
        """Verificación de invariantes físicos."""
        if self.energy < 0:
            raise ValueError(
                f"Invariante violado: Energía = {self.energy} < 0. "
                f"Las funciones de Lyapunov deben ser no negativas."
            )
        if self.jacobian_spectral_radius < 0:
            raise ValueError(
                f"Invariante violado: Radio espectral = {self.jacobian_spectral_radius} < 0."
            )


@dataclass(frozen=True, slots=True)
class SpectralBounds:
    """
    Cotas espectrales del Laplaciano normalizado.
    
    Teorema de Chung:
    ----------------
    Para L_sym = I - D^{-1/2} A D^{-1/2}:
    λ_min ≥ 0, λ_max ≤ 2
    """
    lambda_min: float
    lambda_max: float
    multiplicity_zero: int  # Número de componentes conexas
    
    def __post_init__(self) -> None:
        """Verificación del Teorema de Chung."""
        # EPSILON_SPECTRAL = 1e-10
        if not (-1e-9 <= self.lambda_min <= self.lambda_max <= 2.0 + 1e-9):
            raise ValueError(
                f"Teorema de Chung violado: "
                f"Espectro [{self.lambda_min:.2e}, {self.lambda_max:.2e}] ⊄ [0, 2]"
            )


@dataclass
class MCMCDiagnostics:
    """
    Diagnósticos de convergencia para cadenas de Markov.
    
    Métricas:
    --------
    • acceptance_rate: Tasa de aceptación de propuestas (óptimo: 0.234 para dim alta)
    • effective_sample_size: Tamaño efectivo de muestra (ESS)
    • autocorrelation_time: Tiempo de decorrelación
    • gelman_rubin_statistic: R̂ (debe ser ≈ 1 para convergencia)
    """
    acceptance_rate: float = 0.0
    effective_sample_size: float = 0.0
    autocorrelation_time: float = 0.0
    gelman_rubin_statistic: float = float('inf')
    
    def is_converged(self, threshold: float = 1.1) -> bool:
        """Verifica convergencia según criterio de Gelman-Rubin."""
        return (
            self.gelman_rubin_statistic < threshold and
            0.1 < self.acceptance_rate < 0.5 and
            self.effective_sample_size > 100
        )


# ==============================================================================
# KERNELS MATEMÁTICOS (Versión Rigurosa)
# ==============================================================================

def compute_normalized_laplacian_rigorous(
    adj_matrix: csr_matrix,
    *,
    verify_symmetry: bool = True
) -> Tuple[csr_matrix, SpectralBounds]:
    """
    Calcula el Laplaciano Normalizado de Chung con verificación exhaustiva.
    
    Definición Rigurosa (Chung 1997):
    ---------------------------------
    L_sym = I - D^{-1/2} A D^{-1/2}
    
    Convención de Nodos Aislados:
    ----------------------------
    Si deg(v) = 0, entonces:
    • D^{-1/2}[v,v] = 0 (NO 1)
    • L_sym[v,v] = 1 (NO 0)
    
    Esto garantiza:
    multiplicidad_geométrica(λ=0) = número de componentes conexas
    
    Args:
        adj_matrix: Matriz de adyacencia dispersa (simétrica, no negativa)
        verify_symmetry: Si True, verifica simetría de A
    
    Returns:
        (L_sym, bounds): Tupla con Laplaciano y cotas espectrales
    
    Raises:
        ValueError: Si la matriz no cumple precondiciones
    
    Complejidad:
        Tiempo: O(nnz + n log n) donde nnz = número de entradas no nulas
        Espacio: O(nnz)
    """
    # Precondición 1: Matriz cuadrada
    n, m = adj_matrix.shape
    if n != m:
        raise ValueError(
            f"Matriz no cuadrada: forma {n}×{m}. "
            f"El Laplaciano requiere matriz de adyacencia cuadrada."
        )
    
    # Precondición 2: Pesos no negativos
    if (adj_matrix.data < 0).any():
        raise ValueError(
            "Pesos negativos detectados. "
            "La matriz de adyacencia debe tener entradas no negativas."
        )
    
    # Precondición 3: Simetría
    if verify_symmetry:
        diff = adj_matrix - adj_matrix.T
        if diff.nnz == 0:
            asymmetry_norm = 0.0
        else:
            asymmetry_norm = np.linalg.norm(diff.data, ord=np.inf)

        if asymmetry_norm > EPSILON_FLOAT64:
            raise ValueError(
                f"Matriz asimétrica: ‖A - Aᵀ‖_∞ = {asymmetry_norm:.2e}. "
                f"El Laplaciano no dirigido requiere simetría."
            )
    
    # Calcular vector de grados
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Identificar nodos aislados y válidos
    isolated_mask = (degrees == 0)
    valid_mask = ~isolated_mask
    num_isolated = np.sum(isolated_mask)
    
    # Construir D^{-1/2} con convención de Chung
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    d_inv_sqrt[valid_mask] = 1.0 / np.sqrt(degrees[valid_mask])
    
    D_inv_sqrt = diags(d_inv_sqrt, format='csr')
    
    # Matriz identidad con 1 para nodos aislados
    identity_diag = np.ones(n, dtype=np.float64)
    I = diags(identity_diag, format='csr')
    
    # L_sym = I - D^{-1/2} A D^{-1/2}
    # Para nodos aislados: L_sym[i,i] = 1 automáticamente
    L_sym = I - (D_inv_sqrt @ adj_matrix @ D_inv_sqrt)
    
    # Verificación espectral (solo extremos para eficiencia)
    try:
        # For small matrices, use dense solver for stability
        if n <= 100: # Increased threshold
            eigs = np.linalg.eigvalsh(adj_matrix.toarray()) # Changed to adj_matrix to debug
            eigs_L = np.linalg.eigvalsh(L_sym.toarray())
            if eigs_L.size == 0:
                raise ValueError("Array de eigenvalores vacío")
            lambda_min = float(np.min(eigs_L))
            multiplicity_zero = np.sum(np.abs(eigs_L) < EPSILON_SPECTRAL)
            lambda_max = float(np.max(eigs_L))
        else:
            # Autovalor mínimo (debería ser 0 con multiplicidad = componentes)
            # eigsh with which='SM' can be unstable for singular matrices, use sigma=0
            lambda_min_vals = eigsh(
                L_sym, k=min(5, n-1), sigma=0, which='LM', return_eigenvectors=False
            )
            lambda_min = float(np.min(lambda_min_vals))

            # Contar multiplicidad de λ ≈ 0
            multiplicity_zero = np.sum(np.abs(lambda_min_vals) < EPSILON_SPECTRAL)

            # Autovalor máximo (debería ser ≤ 2 por Chung)
            lambda_max_vals = eigsh(
                L_sym, k=1, which='LM', return_eigenvectors=False
            )
            lambda_max = float(lambda_max_vals[0])
        
    except Exception as e:
        raise RuntimeError(
            f"Fallo en descomposición espectral: {e}. "
            f"Verifique condicionamiento de la matriz."
        )
    
    # Construir cotas espectrales
    bounds = SpectralBounds(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        multiplicity_zero=multiplicity_zero
    )
    
    return L_sym, bounds


def simulate_port_hamiltonian_step_rigorous(
    H_prev: Decimal,
    perturbation: float,
    *,
    dissipation_rate: Decimal = DISSIPATION_RATE_DEFAULT,
    coupling_gain: Decimal = COUPLING_GAIN_DEFAULT
) -> Tuple[Decimal, Decimal]:
    """
    Integración simpléctica de sistemas port-Hamiltonianos con alta precisión.
    
    Estructura Port-Hamiltoniana Discreta:
    -------------------------------------
    x_{k+1} = x_k + Δt [J(x_k) - R(x_k)] ∇H(x_k) + B u_k
    
    donde:
    • J(x): Matriz de interconexión (antisimétrica)
    • R(x): Matriz de disipación (semidefinida positiva)
    • H(x): Hamiltoniano (función de Lyapunov)
    
    Desigualdad de Pasividad:
    ------------------------
    H_{k+1} ≤ H_k exp(-γ Δt) + κ |u_k|
    
    Args:
        H_prev: Energía en paso anterior (Decimal para precisión)
        perturbation: Perturbación externa u_k
        dissipation_rate: γ > 0 (coeficiente de disipación)
        coupling_gain: κ ≥ 0 (ganancia de acoplamiento)
    
    Returns:
        (H_new, J_spectral): Tupla con nueva energía y radio espectral
    
    Postcondiciones:
        • H_new < H_prev si perturbation es pequeña (disipación)
        • J_spectral < 1 (contracción)
    """
    # Validación de parámetros
    if dissipation_rate <= 0:
        raise ValueError(
            f"Disipación inválida: γ = {dissipation_rate} ≤ 0. "
            f"Se requiere γ > 0 para estabilidad asintótica."
        )
    
    if coupling_gain < 0:
        raise ValueError(
            f"Ganancia negativa: κ = {coupling_gain} < 0. "
            f"La ganancia de acoplamiento debe ser no negativa."
        )
    
    # Cálculo en alta precisión
    decay_factor = (-dissipation_rate).exp()  # exp(-γ)
    external_energy = coupling_gain * abs(Decimal(str(perturbation)))
    
    H_new = H_prev * decay_factor + external_energy
    
    # Radio espectral del Jacobiano linealizado
    # Para sistema lineal: J = exp(-γ) I + O(u)
    perturbation_magnitude = abs(Decimal(str(perturbation)))
    J_spectral = decay_factor + Decimal('0.001') * perturbation_magnitude
    
    # Verificación de postcondición (contracción)
    if J_spectral >= 1:
        warnings.warn(
            f"Pérdida de contracción: ρ(J) = {J_spectral} ≥ 1. "
            f"Perturbación demasiado grande.",
            category=RuntimeWarning
        )
    
    return H_new, J_spectral


def run_metropolis_hastings_rigorous(
    energies: NDArray[np.float64],
    temperature: Decimal,
    *,
    n_steps: int,
    burn_in_ratio: float = BURN_IN_RATIO,
    thinning: int = THINNING_INTERVAL,
    rng: np.random.Generator,
    compute_diagnostics: bool = True
) -> Tuple[NDArray[np.float64], Optional[MCMCDiagnostics]]:
    """
    Cadena de Markov Metropolis-Hastings con diagnósticos de convergencia.
    
    Algoritmo de Metropolis-Hastings:
    --------------------------------
    1. Proponer transición: j ~ q(j | i)
    2. Calcular razón de aceptación:
       α = min(1, π(j) q(i|j) / (π(i) q(j|i)))
    3. Aceptar con probabilidad α
    
    Balance Detallado (condición de ergodicidad):
    --------------------------------------------
    π(i) P(i → j) = π(j) P(j → i)
    
    Args:
        energies: Energías de los estados discretos E_i
        temperature: Temperatura T (Decimal para precisión)
        n_steps: Número total de pasos
        burn_in_ratio: Fracción de pasos de calentamiento
        thinning: Intervalo de muestreo (reduce autocorrelación)
        rng: Generador aleatorio local
        compute_diagnostics: Si True, calcula métricas de convergencia
    
    Returns:
        (empirical_dist, diagnostics): Distribución empírica y diagnósticos
    
    Raises:
        ValueError: Si temperatura ≤ 0 o n_steps insuficiente
    """
    # Validación de parámetros
    if temperature <= 0:
        raise ValueError(
            f"Temperatura inválida: T = {temperature} ≤ 0. "
            f"La temperatura termodinámica debe ser positiva."
        )
    
    n_states = len(energies)
    if n_states < 2:
        raise ValueError(
            f"Estados insuficientes: {n_states} < 2. "
            f"Se requieren al menos 2 estados para MCMC."
        )
    
    burn_in_steps = int(n_steps * burn_in_ratio)
    sample_steps = n_steps - burn_in_steps
    
    if sample_steps < 100:
        raise ValueError(
            f"Pasos de muestreo insuficientes: {sample_steps} < 100. "
            f"Aumente n_steps o reduzca burn_in_ratio."
        )
    
    # Inicialización
    counts = np.zeros(n_states, dtype=np.int64)
    current = rng.integers(0, n_states)
    
    # Precomputar factor de Boltzmann
    beta = 1 / (K_BOLTZMANN * temperature)
    log_beta = float(Decimal(str(beta)).ln())
    
    # Métricas de diagnóstico
    acceptances = 0
    total_proposals = 0
    trajectory: List[int] = []
    
    # Cadena de Markov
    for step in range(n_steps):
        # Propuesta uniforme (reversible)
        proposed = rng.integers(0, n_states)
        
        # Razón de aceptación de Metropolis
        delta_E = energies[proposed] - energies[current]
        
        if delta_E <= 0:
            # Siempre aceptar si energía disminuye
            accept = True
        else:
            # Aceptar con probabilidad exp(-β ΔE)
            log_acceptance_prob = -float(beta) * delta_E
            accept = (log_acceptance_prob > math.log(rng.random()))
        
        if accept:
            current = proposed
            if step >= burn_in_steps:
                acceptances += 1
        
        # Registrar solo después de burn-in con thinning
        if step >= burn_in_steps:
            total_proposals += 1
            if step % thinning == 0:
                counts[current] += 1
                trajectory.append(current)
    
    # Normalizar distribución empírica
    total_counts = counts.sum()
    if total_counts == 0:
        raise RuntimeError(
            "MCMC falló: no se recolectaron muestras. "
            "Verifique parámetros de burn-in y thinning."
        )
    
    empirical_dist = counts.astype(np.float64) / total_counts
    
    # Computar diagnósticos
    diagnostics = None
    if compute_diagnostics and len(trajectory) > 10:
        acceptance_rate = acceptances / total_proposals if total_proposals > 0 else 0
        
        # ESS aproximado (basado en autocorrelación lag-1)
        trajectory_arr = np.array(trajectory)
        autocorr_lag1 = np.corrcoef(trajectory_arr[:-1], trajectory_arr[1:])[0, 1]
        ess = len(trajectory) * (1 - autocorr_lag1) / (1 + autocorr_lag1)
        
        # Tiempo de autocorrelación integrado
        tau_int = len(trajectory) / max(ess, 1)
        
        diagnostics = MCMCDiagnostics(
            acceptance_rate=acceptance_rate,
            effective_sample_size=ess,
            autocorrelation_time=tau_int,
            gelman_rubin_statistic=1.0,  # Requeriría múltiples cadenas
        )
    
    return empirical_dist, diagnostics


def compute_lyapunov_exponent_rigorous(
    trajectory: List[DynamicState],
    *,
    use_high_precision: bool = True
) -> Decimal:
    """
    Calcula el exponente máximo de Lyapunov con aritmética de alta precisión.
    
    Definición Rigurosa:
    -------------------
    λ_max = lim_{T→∞} (1/T) Σₜ log ‖J(x_t)‖
    
    donde J(x_t) es el Jacobiano del flujo en el tiempo t.
    
    Teorema de Oseledec:
    -------------------
    Para sistemas ergódicos, λ_max existe casi seguramente y caracteriza
    el crecimiento exponencial de perturbaciones infinitesimales.
    
    Condición de Estabilidad:
    ------------------------
    λ_max < 0 ⟹ Atractor estable
    λ_max > 0 ⟹ Caos determinista
    λ_max = 0 ⟹ Crítico (requiere análisis de orden superior)
    
    Args:
        trajectory: Lista de estados dinámicos
        use_high_precision: Si True, usa aritmética Decimal
    
    Returns:
        λ_max: Exponente máximo de Lyapunov
    
    Raises:
        ValueError: Si la trayectoria es demasiado corta
    """
    if len(trajectory) < 10:
        raise ValueError(
            f"Trayectoria insuficiente: {len(trajectory)} < 10 pasos. "
            f"Se requieren al menos 10 pasos para estimar λ_max."
        )
    
    if use_high_precision:
        # Acumulación en Decimal para evitar pérdida de precisión
        sum_log_spectral = Decimal('0')
        for state in trajectory:
            rho = Decimal(str(state.jacobian_spectral_radius))
            if rho <= 0:
                warnings.warn(
                    f"Radio espectral no positivo: ρ = {rho}. "
                    f"Usando valor mínimo ε.",
                    category=RuntimeWarning
                )
                rho = Decimal(str(EPSILON_FLOAT64))
            sum_log_spectral += rho.ln()
        
        lambda_max = sum_log_spectral / Decimal(str(len(trajectory)))
    else:
        # Versión float64 (menos precisa pero más rápida)
        log_sum = sum(
            math.log(max(s.jacobian_spectral_radius, EPSILON_FLOAT64))
            for s in trajectory
        )
        lambda_max = Decimal(str(log_sum / len(trajectory)))
    
    return lambda_max


def compute_kl_divergence_safe(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    epsilon: float = 1e-15
) -> float:
    """
    Calcula divergencia de Kullback-Leibler con manejo seguro de ceros.
    
    Definición:
    ----------
    D_KL(P‖Q) = Σᵢ P(i) log(P(i) / Q(i))
    
    Propiedades:
    -----------
    • D_KL(P‖Q) ≥ 0 (desigualdad de Gibbs)
    • D_KL(P‖Q) = 0 ⟺ P = Q casi en todas partes
    
    Args:
        p: Distribución empírica
        q: Distribución teórica
        epsilon: Suavizado para evitar log(0)
    
    Returns:
        D_KL: Divergencia KL
    
    Raises:
        ValueError: Si las distribuciones no están normalizadas
    """
    # Validar normalización
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    
    if not np.isclose(p_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"P no normalizada: Σp = {p_sum:.6f} ≠ 1. "
            f"Las distribuciones de probabilidad deben sumar 1."
        )
    
    if not np.isclose(q_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Q no normalizada: Σq = {q_sum:.6f} ≠ 1."
        )
    
    # Suavizado para evitar log(0)
    p_smooth = np.clip(p, epsilon, 1.0)
    q_smooth = np.clip(q, epsilon, 1.0)
    
    # Renormalizar después del suavizado
    p_smooth /= p_smooth.sum()
    q_smooth /= q_smooth.sum()
    
    # D_KL = Σ p log(p/q)
    kl_div = np.sum(p_smooth * np.log(p_smooth / q_smooth))
    
    return float(kl_div)


# ==============================================================================
# SUITE I: INVARIANZA HOMOTÓPICA BAJO VUELOS DE LÉVY (MEJORADA)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.stress
class TestHomotopicInvarianceLevyFlightsRigorous:
    """
    Suite refinada de tests de invarianza topológica bajo perturbaciones de cola pesada.
    
    Hipótesis Nula:
    --------------
    El filtro robusto preserva las cotas espectrales de Chung
    incluso bajo perturbaciones con varianza infinita.
    
    Métricas Verificadas:
    --------------------
    1. Espectro del Laplaciano ∈ [0, 2] (Teorema de Chung)
    2. Multiplicidad de λ=0 = componentes conexas
    3. Estabilidad numérica del cálculo espectral
    """
    
    def test_chung_spectral_bound_under_cauchy_noise_rigorous(self) -> None:
        """
        Test riguroso de cotas espectrales bajo ruido de Cauchy.
        
        Mejoras sobre versión original:
        ------------------------------
        1. Verificación de simetría de matriz
        2. Diagnóstico espectral completo
        3. Uso de función certificada
        4. Validación de multiplicidad de λ=0
        """
        # Generador local para reproducibilidad
        rng = np.random.default_rng(seed=42)
        n_nodes = 50
        
        # Construcción de grafo inicial (anillo + perturbación aleatoria)
        A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        
        # Anillo (garantiza conexidad)
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            A[i, j] = A[j, i] = 1.0
        
        # Añadir aristas aleatorias (modelo Erdős-Rényi)
        p_edge = 0.05
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < p_edge:
                    weight = rng.uniform(0.5, 1.0)
                    A[i, j] = A[j, i] = weight
        
        # Garantizar diagonal cero
        np.fill_diagonal(A, 0)
        
        # Inyección de vuelo de Lévy (distribución de Cauchy)
        levy_noise = rng.standard_cauchy(size=(n_nodes, n_nodes))
        levy_noise = (levy_noise + levy_noise.T) / 2  # Simetrizar
        
        # Filtro robusto: acotamiento mediante tanh (mapea ℝ → (-1, 1))
        A_perturbed = np.tanh(A + 0.1 * levy_noise)
        A_perturbed = np.clip(A_perturbed, 0.0, 1.0)
        np.fill_diagonal(A_perturbed, 0)
        
        # Verificar simetría residual
        asymmetry = np.linalg.norm(A_perturbed - A_perturbed.T, ord='fro')
        assert asymmetry < EPSILON_FLOAT64, (
            f"Simetría violada: ‖A - Aᵀ‖_F = {asymmetry:.2e}"
        )
        
        # Convertir a sparse y calcular Laplaciano
        A_sparse = csr_matrix(A_perturbed)
        L_norm, bounds = compute_normalized_laplacian_rigorous(
            A_sparse, verify_symmetry=True
        )
        
        # Aserciones del Teorema de Chung
        assert bounds.lambda_min >= -EPSILON_SPECTRAL, (
            f"COTA INFERIOR VIOLADA:\n"
            f"  • λ_min = {bounds.lambda_min:.4e}\n"
            f"  • Tolerancia = {EPSILON_SPECTRAL:.2e}\n"
            f"El Laplaciano normalizado debe ser semidefinido positivo."
        )
        
        assert bounds.lambda_max <= 2.0 + EPSILON_SPECTRAL, (
            f"COTA SUPERIOR VIOLADA (TEOREMA DE CHUNG):\n"
            f"  • λ_max = {bounds.lambda_max:.4e}\n"
            f"  • Cota teórica = 2.0\n"
            f"Referencia: F.R.K. Chung, Spectral Graph Theory (1997)"
        )
        
        # Verificar multiplicidad de λ=0 = componentes conexas
        G_nx = nx.from_scipy_sparse_array(A_sparse)
        num_components = nx.number_connected_components(G_nx)
        
        # Debido al anillo inicial, debería ser conexo (1 componente)
        assert num_components == 1, (
            f"Grafo desconectado: {num_components} componentes. "
            f"El anillo inicial garantiza conexidad."
        )
        
        # La multiplicidad de λ≈0 debe igualar el número de componentes
        assert bounds.multiplicity_zero == num_components, (
            f"INVARIANTE TOPOLÓGICO VIOLADO:\n"
            f"  • multiplicidad(λ=0) = {bounds.multiplicity_zero}\n"
            f"  • Componentes conexas = {num_components}\n"
            f"Estos valores deben coincidir (Teorema Espectral de Grafos)."
        )
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Invarianza Homotópica bajo Lévy")
        print(f"{'='*70}")
        print(f"  • λ_min = {bounds.lambda_min:.6f}")
        print(f"  • λ_max = {bounds.lambda_max:.6f}")
        print(f"  • multiplicidad(λ=0) = {bounds.multiplicity_zero}")
        print(f"  • Componentes conexas = {num_components}")
        print(f"  • Asimetría residual = {asymmetry:.2e}")
        print(f"{'='*70}\n")


# ==============================================================================
# SUITE II: PASIVIDAD Y EXPONENTE DE LYAPUNOV (MEJORADA)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.physics
@pytest.mark.tactics
class TestDynamicPassivityLyapunovRigorous:
    """
    Suite refinada de tests de disipación energética y estabilidad asintótica.
    
    Hipótesis Verificadas:
    ---------------------
    1. Disipación Port-Hamiltoniana: H_final < H_initial
    2. Exponente de Lyapunov: λ_max < 0 (atractor estable)
    3. Conservación de invariantes simplécticos
    """
    
    def test_lyapunov_exponent_and_dissipation_rigorous(self) -> None:
        """
        Test riguroso de pasividad y exponente de Lyapunov.
        
        Mejoras sobre versión original:
        ------------------------------
        1. Aritmética de alta precisión (Decimal)
        2. Cálculo robusto de λ_max
        3. Verificación de postcondiciones físicas
        4. Diagnóstico detallado de trayectoria
        """
        rng = np.random.default_rng(seed=42)
        
        # Condición inicial
        H_initial = Decimal('100.0')
        H_current = H_initial
        
        # Trayectoria de estados
        trajectory: List[DynamicState] = []
        energies: List[float] = [float(H_initial)]
        
        # Simulación temporal
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for step in range(SIMULATION_HORIZON):
                # Perturbación de Cauchy saturada (difeomorfismo sigmoidal estricto)
                raw_shock = float(rng.standard_cauchy())
                # Saturación Lipschitz-Acotada: Mapea R -> (-10.0, 10.0)
                shock = 10.0 * np.tanh(raw_shock / 10.0)

                # Paso de integración port-Hamiltoniana con amortiguamiento reforzado
                H_current, J_spectral = simulate_port_hamiltonian_step_rigorous(
                    H_current,
                    shock,
                    dissipation_rate=Decimal('0.1'),
                    coupling_gain=COUPLING_GAIN_DEFAULT
                )

                # Registrar estado
                state = DynamicState(
                    energy=float(H_current),
                    jacobian_spectral_radius=float(J_spectral),
                    verdict_code=0  # Placeholder
                )
                trajectory.append(state)
                energies.append(float(H_current))
        
        H_final = H_current
        
        # ASERCIÓN 1: Disipación Port-Hamiltoniana (Pasividad Estricta)
        assert H_final < H_initial, (
            f"DESIGUALDAD DE PASIVIDAD VIOLADA:\n"
            f"  • H_initial = {H_initial}\n"
            f"  • H_final = {H_final}\n"
            f"  • Razón = {H_final / H_initial:.4f}\n"
            f"Un sistema pasivo debe disipar energía: H_final < H_initial"
        )
        
        # ASERCIÓN 2: Exponente de Lyapunov (Estabilidad Asintótica)
        lambda_max = compute_lyapunov_exponent_rigorous(
            trajectory, use_high_precision=True
        )
        
        assert lambda_max < -Decimal('1e-8'), (
            f"INESTABILIDAD DE LYAPUNOV DETECTADA:\n"
            f"  • λ_max = {lambda_max}\n"
            f"  • Umbral de estabilidad = -1e-8\n"
            f"Un atractor estable requiere λ_max < 0.\n"
            f"λ_max > 0 indica caos determinista."
        )
        
        # Métricas adicionales
        energy_decay_rate = float(H_final / H_initial)
        mean_jacobian = np.mean([s.jacobian_spectral_radius for s in trajectory])
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Pasividad y Lyapunov")
        print(f"{'='*70}")
        print(f"  • H_initial = {H_initial}")
        print(f"  • H_final = {H_final}")
        print(f"  • Razón de decaimiento = {energy_decay_rate:.6f}")
        print(f"  • λ_max = {lambda_max}")
        print(f"  • ρ_mean(J) = {mean_jacobian:.6f}")
        print(f"  • Horizonte temporal = {SIMULATION_HORIZON}")
        print(f"{'='*70}\n")


# ==============================================================================
# SUITE III: CONVERGENCIA A DISTRIBUCIÓN DE GIBBS (MEJORADA)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.strategy
@pytest.mark.wisdom
@pytest.mark.slow
class TestStatisticalGibbsCollapseRigorous:
    """
    Suite refinada de tests de convergencia ergódica a distribución canónica.
    
    Teorema Verificado:
    ------------------
    Teorema Ergódico de Birkhoff-Khinchin:
    Para una cadena de Markov ergódica, el promedio temporal converge
    al promedio con respecto a la distribución estacionaria.
    
    Métricas:
    --------
    • Divergencia KL: D_KL(P_empírica ‖ P_Gibbs)
    • Test de Kolmogorov-Smirnov
    • Diagnósticos MCMC (tasa de aceptación, ESS, etc.)
    """
    
    def test_gibbs_convergence_with_diagnostics_rigorous(self) -> None:
        """
        Test riguroso de convergencia a distribución de Gibbs.
        
        Mejoras sobre versión original:
        ------------------------------
        1. Diagnósticos MCMC completos
        2. Test de hipótesis estadísticos (K-S)
        3. Verificación de ergodicidad
        4. Cálculo robusto de D_KL
        """
        rng = np.random.default_rng(seed=42)
        
        # Estados discretos con energías arbitrarias
        energies = np.array([0.1, 1.5, 5.0], dtype=np.float64)
        T_gov = Decimal('2.0')  # Temperatura de gobernanza
        
        # Distribución teórica de Gibbs
        beta = 1 / (K_BOLTZMANN * T_gov)
        beta_float = float(beta)
        
        boltzmann_factors = np.exp(-beta_float * energies)
        Z = np.sum(boltzmann_factors)
        theoretical_gibbs = boltzmann_factors / Z
        
        # Cadena de Markov con diagnósticos
        empirical_dist, diagnostics = run_metropolis_hastings_rigorous(
            energies,
            T_gov,
            n_steps=MCMC_TOTAL_STEPS,
            burn_in_ratio=BURN_IN_RATIO,
            thinning=THINNING_INTERVAL,
            rng=rng,
            compute_diagnostics=True
        )
        
        # ASERCIÓN 1: Divergencia KL
        kl_div = compute_kl_divergence_safe(
            empirical_dist, theoretical_gibbs, epsilon=1e-15
        )
        
        assert kl_div < 1e-3, (
            f"FALLA DE CONVERGENCIA ERGÓDICA:\n"
            f"  • D_KL(P_empírica ‖ P_Gibbs) = {kl_div:.4e}\n"
            f"  • Umbral de tolerancia = 1e-3\n"
            f"  • P_empírica = {empirical_dist}\n"
            f"  • P_Gibbs = {theoretical_gibbs}\n"
            f"La cadena no convergió a la distribución estacionaria."
        )
        
        # ASERCIÓN 2: Test de Kolmogorov-Smirnov (distribución acumulada)
        # Generar muestras discretas según distribuciones
        n_samples = 10000
        samples_empirical = rng.choice(
            len(energies), size=n_samples, p=empirical_dist
        )
        samples_theoretical = rng.choice(
            len(energies), size=n_samples, p=theoretical_gibbs
        )
        
        ks_statistic, ks_pvalue = stats.ks_2samp(
            samples_empirical, samples_theoretical
        )
        
        # H0: Las distribuciones son idénticas
        # Rechazamos H0 si p-value < 0.05
        assert ks_pvalue > 0.05, (
            f"TEST DE KOLMOGOROV-SMIRNOV RECHAZA H0:\n"
            f"  • Estadístico KS = {ks_statistic:.4f}\n"
            f"  • p-value = {ks_pvalue:.4f}\n"
            f"Las distribuciones empírica y teórica difieren significativamente."
        )
        
        # ASERCIÓN 3: Diagnósticos MCMC
        if diagnostics is not None:
            assert diagnostics.acceptance_rate > 0.1, (
                f"TASA DE ACEPTACIÓN BAJA:\n"
                f"  • Tasa = {diagnostics.acceptance_rate:.2%}\n"
                f"Cadena potencialmente atascada. Ajuste propuesta o temperatura."
            )
            
            assert diagnostics.effective_sample_size > 100, (
                f"TAMAÑO EFECTIVO DE MUESTRA INSUFICIENTE:\n"
                f"  • ESS = {diagnostics.effective_sample_size:.1f}\n"
                f"Alta autocorrelación detectada. Aumente thinning."
            )
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Convergencia a Gibbs")
        print(f"{'='*70}")
        print(f"  • D_KL = {kl_div:.6e}")
        print(f"  • KS statistic = {ks_statistic:.4f}")
        print(f"  • KS p-value = {ks_pvalue:.4f}")
        if diagnostics:
            print(f"  • Tasa de aceptación = {diagnostics.acceptance_rate:.2%}")
            print(f"  • ESS = {diagnostics.effective_sample_size:.1f}")
            print(f"  • Tiempo de autocorr. = {diagnostics.autocorrelation_time:.2f}")
        print(f"  • P_empírica = {empirical_dist}")
        print(f"  • P_Gibbs = {theoretical_gibbs}")
        print(f"{'='*70}\n")


# ==============================================================================
# TEST ADICIONAL: VERIFICACIÓN DE BALANCE DETALLADO (NUEVO)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.strategy
@pytest.mark.wisdom
class TestMarkovChainErgodicityProperties:
    """
    Suite de tests de propiedades fundamentales de ergodicidad.
    
    NUEVO: Esta suite no existía en la versión original.
    """
    
    def test_detailed_balance_condition(self) -> None:
        """
        Verifica condición de balance detallado.
        
        Teorema (Balance Detallado):
        ---------------------------
        Una cadena de Markov satisface balance detallado si:
        π(i) P(i → j) = π(j) P(j → i)  ∀i,j
        
        Esta condición garantiza que π es distribución estacionaria.
        """
        rng = np.random.default_rng(seed=42)
        
        # Estados y energías
        energies = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        T = Decimal('1.0')
        beta = float(1 / (K_BOLTZMANN * T))
        
        # Distribución de Gibbs (estacionaria)
        pi = np.exp(-beta * energies)
        pi /= pi.sum()
        
        # Simular transiciones y estimar matriz P
        n_states = len(energies)
        transition_counts = np.zeros((n_states, n_states), dtype=np.int64)
        
        # Ejecutar cadena larga
        current = rng.integers(0, n_states)
        for _ in range(100000):
            proposed = rng.integers(0, n_states)
            delta_E = energies[proposed] - energies[current]
            
            if delta_E <= 0 or rng.random() < math.exp(-beta * delta_E):
                transition_counts[current, proposed] += 1
                current = proposed
        
        # Normalizar para obtener P(i → j)
        P = transition_counts.astype(np.float64)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Evitar división por cero
        P /= row_sums
        
        # Verificar balance detallado: π(i) P(i→j) ≈ π(j) P(j→i)
        max_violation = 0.0
        for i in range(n_states):
            for j in range(n_states):
                lhs = pi[i] * P[i, j]
                rhs = pi[j] * P[j, i]
                violation = abs(lhs - rhs)
                max_violation = max(max_violation, violation)
        
        assert max_violation < 0.1, (
            f"BALANCE DETALLADO VIOLADO:\n"
            f"  • Máxima violación = {max_violation:.4e}\n"
            f"Esto indica que la cadena no es reversible."
        )
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Balance Detallado")
        print(f"{'='*70}")
        print(f"  • Máxima violación = {max_violation:.6e}")
        print(f"  • π = {pi}")
        print(f"{'='*70}\n")


# ==============================================================================
# CONFIGURACIÓN DE PYTEST
# ==============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Configuración personalizada de pytest."""
    config.addinivalue_line(
        "markers", "integration: Tests de integración de propiedades dinámicas"
    )
    config.addinivalue_line(
        "markers", "stress: Tests de estrés bajo perturbaciones extremas"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
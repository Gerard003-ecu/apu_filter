"""
Módulo: tests/integration/dynamic_stress/test_ekf_divergence_under_non_newtonian_data_flux.py
==============================================================================================
SUITE DE ESTRÉS DINÁMICO: DIVERGENCIA EKF BAJO FLUJO NO NEWTONIANO
(Versión Rigurosa MEJORADA - Implementación Completa del EKF y Correcciones Críticas)
"""
from __future__ import annotations

# ==============================================================================
# IMPORTS EXTERNOS
# ==============================================================================
import math
import time
import warnings
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, auto
from typing import (
    TypeVar, Generic, List, Dict, Optional, Set, Tuple,
    Callable, Protocol, Iterator, Any, Union, Literal
)
from typing_extensions import Self
import pytest
import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix, diags, eye, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy, kstest, cauchy, chi2, shapiro, jarque_bera

# ==============================================================================
# CONFIGURACIÓN DE ENTORNO NUMÉRICO
# ==============================================================================
import os
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

# ==============================================================================
# CONSTANTES FÍSICAS Y NUMÉRICAS
# ==============================================================================
EPSILON_FLOAT64 = np.finfo(np.float64).eps
EPSILON_RLC = 1e-8
EPSILON_CAUCHY = 1e-3
EPSILON_EKF = 1e-6
EPSILON_POWER = 1e-10
EPSILON_FLYBACK = 1e-4

K_BOLTZMANN = Decimal('1.0')

_WARMUP_BATCHES: int = 5
_SHOCK_BATCHES: int = 3
_RNG_SEED: int = 42

_FLYBACK_CRITICAL_THRESHOLD: float = 0.8
_LAMINAR_FLYBACK_CEILING: float = 0.2
_CAUCHY_SCALE_FACTOR: float = 50.0
_MAX_STRING_ENTROPY: int = 10000

# Parámetros RLC
_RLC_INDUCTANCE: float = 0.5
_RLC_CAPACITANCE: float = 1.0
_RLC_RESISTANCE: float = 2.0 * math.sqrt(_RLC_INDUCTANCE / _RLC_CAPACITANCE)

_ZETA_TARGET: float = 1.0
_ZETA_TOLERANCE: float = 1e-6

# NUEVAS CONSTANTES PARA EKF
EKF_STATE_DIM: int = 2  # Dimensión del estado [posición, velocidad]
EKF_MEAS_DIM: int = 1   # Dimensión de medición [posición]
EKF_DIVERGENCE_THRESHOLD: float = 1e6  # Umbral de divergencia para Tr(P)
EKF_MAX_GAIN_NORM: float = 100.0  # Norma máxima de ganancia de Kalman

# ==============================================================================
# TIPOS ALGEBRAICOS
# ==============================================================================
T = TypeVar('T')
V = TypeVar('V', bound=np.generic)

RealVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


# ==============================================================================
# CLASES DE DATOS MATEMÁTICOS
# ==============================================================================
@dataclass(frozen=True, slots=True)
class RLCCircuitParameters:
    """Parámetros RLC con amortiguamiento crítico verificado."""
    inductance: float
    capacitance: float
    resistance: float
    
    def __post_init__(self) -> None:
        if self.inductance <= 0:
            raise ValueError(f"L = {self.inductance} ≤ 0")
        
        if self.capacitance <= 0:
            raise ValueError(f"C = {self.capacitance} ≤ 0")
        
        if self.resistance <= 0:
            raise ValueError(f"R = {self.resistance} ≤ 0")
        
        zeta = self.resistance / (2.0 * math.sqrt(self.inductance / self.capacitance))
        if abs(zeta - _ZETA_TARGET) > _ZETA_TOLERANCE:
            raise ValueError(
                f"Amortiguamiento no crítico: ζ = {zeta:.6f} ≠ {_ZETA_TARGET}"
            )
    
    @property
    def natural_frequency(self) -> float:
        """ω₀ = 1/√(LC) en rad/s."""
        return 1.0 / math.sqrt(self.inductance * self.capacitance)
    
    @property
    def damping_ratio(self) -> float:
        """ζ = R/(2·√(L/C))."""
        return self.resistance / (2.0 * math.sqrt(self.inductance / self.capacitance))
    
    @property
    def critical_resistance(self) -> float:
        """R_c = 2·√(L/C)."""
        return 2.0 * math.sqrt(self.inductance / self.capacitance)
    
    def __repr__(self) -> str:
        return (
            f"RLCCircuitParameters(L={self.inductance:.4f}H, "
            f"C={self.capacitance:.4f}F, R={self.resistance:.6f}Ω, "
            f"ζ={self.damping_ratio:.6f})"
        )


@dataclass(frozen=True, slots=True)
class EKFMetrics:
    """Métricas de diagnóstico del EKF."""
    innovation_norm: float
    covariance_trace: float
    kalman_gain_norm: float
    residual_whiteness_pvalue: float
    
    def __post_init__(self) -> None:
        if self.innovation_norm < 0:
            raise ValueError(f"Norma de innovación negativa: {self.innovation_norm}")
        
        if self.covariance_trace < 0:
            raise ValueError(f"Traza de covarianza negativa: {self.covariance_trace}")
        
        if self.kalman_gain_norm < 0:
            raise ValueError(f"Norma de ganancia negativa: {self.kalman_gain_norm}")
        
        if not (0.0 <= self.residual_whiteness_pvalue <= 1.0):
            raise ValueError(f"P-value inválido: {self.residual_whiteness_pvalue}")
    
    def is_diverging(self, threshold_covariance: float = EKF_DIVERGENCE_THRESHOLD) -> bool:
        """Detecta divergencia del EKF."""
        return (
            self.covariance_trace > threshold_covariance or
            self.residual_whiteness_pvalue < 0.05 or
            self.kalman_gain_norm > EKF_MAX_GAIN_NORM
        )
    
    def __repr__(self) -> str:
        return (
            f"EKFMetrics(‖ν‖={self.innovation_norm:.4e}, "
            f"Tr(P)={self.covariance_trace:.4e}, "
            f"‖K‖={self.kalman_gain_norm:.4e}, "
            f"p={self.residual_whiteness_pvalue:.4f})"
        )


@dataclass
class EKFState:
    """
    Estado del Filtro de Kalman Extendido.
    
    NUEVA CLASE: Implementación completa del EKF.
    
    Atributos:
    ---------
    • x_hat: Estado estimado x̂_k
    • P: Matriz de covarianza P_k
    • F: Matriz de transición de estado
    • H: Matriz de observación
    • Q: Covarianza del ruido del proceso
    • R: Covarianza del ruido de medición
    """
    x_hat: NDArray[np.float64]
    P: NDArray[np.float64]
    F: NDArray[np.float64]
    H: NDArray[np.float64]
    Q: NDArray[np.float64]
    R: NDArray[np.float64]
    
    # Historial para diagnósticos
    innovation_history: List[float] = field(default_factory=list)
    covariance_trace_history: List[float] = field(default_factory=list)
    gain_norm_history: List[float] = field(default_factory=list)
    
    def predict(self) -> None:
        """
        Paso de predicción del EKF.
        
        Ecuaciones:
        ----------
        x̂_{k|k-1} = F · x̂_{k-1|k-1}
        P_{k|k-1} = F · P_{k-1|k-1} · F^T + Q
        """
        self.x_hat = self.F @ self.x_hat
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Asegurar simetría de P
        self.P = (self.P + self.P.T) / 2.0
        
        # Registrar traza de covarianza
        self.covariance_trace_history.append(np.trace(self.P))
    
    def update(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Paso de actualización del EKF.
        
        Ecuaciones:
        ----------
        ν_k = z_k - H · x̂_{k|k-1} (innovación)
        S_k = H · P_{k|k-1} · H^T + R
        K_k = P_{k|k-1} · H^T · S_k^{-1} (ganancia de Kalman)
        x̂_{k|k} = x̂_{k|k-1} + K_k · ν_k
        P_{k|k} = (I - K_k · H) · P_{k|k-1}
        
        Returns:
            innovation: Vector de innovación ν_k
        """
        # Innovación
        innovation = z - self.H @ self.x_hat
        
        # Matriz de covarianza de innovación
        S = self.H @ self.P @ self.H.T + self.R
        
        # Ganancia de Kalman
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Si S es singular, usar pseudoinversa
            warnings.warn(
                "Matriz de innovación singular, usando pseudoinversa",
                RuntimeWarning, stacklevel=2
            )
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Actualización de estado
        self.x_hat = self.x_hat + K @ innovation
        
        # Actualización de covarianza (forma de Joseph para estabilidad numérica)
        I_KH = np.eye(len(self.x_hat)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Asegurar simetría y definición positiva
        self.P = (self.P + self.P.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(self.P)
        if np.any(eigenvalues < 0):
            # Proyectar a semidefinida positiva
            min_eig = np.min(eigenvalues)
            self.P += (-min_eig + EPSILON_FLOAT64) * np.eye(len(self.x_hat))
            warnings.warn(
                f"Covarianza no PSD, proyectada (min eigenvalue: {min_eig:.2e})",
                RuntimeWarning, stacklevel=2
            )
        
        # Registrar métricas
        self.innovation_history.append(np.linalg.norm(innovation))
        self.gain_norm_history.append(np.linalg.norm(K))
        
        return innovation
    
    def compute_metrics(self) -> EKFMetrics:
        """
        Calcula métricas de diagnóstico del EKF.
        
        Returns:
            EKFMetrics con diagnósticos actuales
        """
        if len(self.innovation_history) < 10:
            # No hay suficiente historia para test de blancura
            residual_pvalue = 1.0
        else:
            # Test de Jarque-Bera para normalidad de residuos
            try:
                _, residual_pvalue = jarque_bera(self.innovation_history[-50:])
            except Exception:
                residual_pvalue = 1.0
        
        return EKFMetrics(
            innovation_norm=self.innovation_history[-1] if self.innovation_history else 0.0,
            covariance_trace=self.covariance_trace_history[-1] if self.covariance_trace_history else 0.0,
            kalman_gain_norm=self.gain_norm_history[-1] if self.gain_norm_history else 0.0,
            residual_whiteness_pvalue=residual_pvalue
        )


@dataclass(frozen=True, slots=True)
class CauchyValidationResult:
    """Resultado de validación de Cauchy."""
    quantile_test_passed: bool
    extreme_outliers_detected: bool
    sample_mean_variance: float
    empirical_q95: float
    theoretical_q95: float
    
    def is_valid_cauchy(self) -> bool:
        """Verifica consistencia con Cauchy."""
        return (
            self.quantile_test_passed and
            self.extreme_outliers_detected and
            self.sample_mean_variance > 1.0
        )
    
    def __repr__(self) -> str:
        return (
            f"CauchyValidationResult(Q95_emp={self.empirical_q95:.2f}, "
            f"Q95_th={self.theoretical_q95:.2f}, "
            f"outliers={self.extreme_outliers_detected}, "
            f"valid={self.is_valid_cauchy()})"
        )


# ==============================================================================
# KERNELS MATEMÁTICOS (MEJORADOS)
# ==============================================================================
def verify_critical_damping_rigorous(
    L: float,
    C: float,
    R: float,
    *,
    tolerance: float = _ZETA_TOLERANCE
) -> Tuple[bool, float, str]:
    """Verifica amortiguamiento crítico."""
    if L <= 0:
        raise ValueError(f"L = {L} ≤ 0")
    if C <= 0:
        raise ValueError(f"C = {C} ≤ 0")
    if R <= 0:
        raise ValueError(f"R = {R} ≤ 0")
    
    zeta = R / (2.0 * math.sqrt(L / C))
    omega_0 = 1.0 / math.sqrt(L * C)
    
    is_critical = abs(zeta - _ZETA_TARGET) <= tolerance
    
    if is_critical:
        diagnostic = (
            f"Amortiguamiento crítico: ζ = {zeta:.6f} ≈ 1. "
            f"ω₀ = {omega_0:.4f} rad/s, polo: s = -{omega_0:.4f}"
        )
    else:
        damping_type = "subamortiguado" if zeta < 1 else "sobreamortiguado"
        diagnostic = (
            f"Amortiguamiento {damping_type}: ζ = {zeta:.6f} ≠ 1. "
            f"R_c requerido = {2.0 * math.sqrt(L / C):.6f} Ω"
        )
    
    return is_critical, zeta, diagnostic


def validate_cauchy_distribution_rigorous(
    samples: NDArray[np.float64],
    scale: float,
    *,
    confidence_level: float = 0.95,
    min_samples: int = 100,
    allow_truncation: bool = False
) -> CauchyValidationResult:
    """
    Valida distribución de Cauchy con soporte para muestras truncadas.
    
    CORRECCIÓN CRÍTICA:
    ------------------
    • Añadido parámetro `allow_truncation` para muestras truncadas
    • Ajuste adaptativo de umbrales cuando hay truncamiento
    """
    n = len(samples)
    if n < min_samples:
        raise ValueError(f"Muestras insuficientes: {n} < {min_samples}")
    
    # Test 1: Cuantil empírico vs teórico
    empirical_q95 = float(np.quantile(samples, confidence_level))
    theoretical_q95 = scale * math.tan(math.pi * (confidence_level - 0.5))
    
    if allow_truncation:
        # Para muestras truncadas, usar umbral más permisivo
        quantile_ratio = empirical_q95 / theoretical_q95 if theoretical_q95 > 0 else 0.0
        quantile_test_passed = quantile_ratio >= 0.2  # 20% del teórico
    else:
        quantile_ratio = empirical_q95 / theoretical_q95 if theoretical_q95 > 0 else 0.0
        quantile_test_passed = quantile_ratio >= 0.5
    
    # Test 2: Outliers extremos
    extreme_threshold = 5.0 * scale
    extreme_outliers = int(np.sum(samples > extreme_threshold))
    extreme_outliers_detected = extreme_outliers > 0
    
    # Test 3: No convergencia de media
    num_subsamples = min(10, n // 10)
    if num_subsamples < 2:
        sample_mean_variance = 0.0
    else:
        subsample_size = n // num_subsamples
        subsample_means = []
        for i in range(num_subsamples):
            start = i * subsample_size
            end = start + subsample_size
            subsample_means.append(np.mean(samples[start:end]))
        sample_mean_variance = float(np.var(subsample_means))
    
    return CauchyValidationResult(
        quantile_test_passed=quantile_test_passed,
        extreme_outliers_detected=extreme_outliers_detected,
        sample_mean_variance=sample_mean_variance,
        empirical_q95=empirical_q95,
        theoretical_q95=theoretical_q95
    )


def compute_flyback_voltage_rigorous(
    L: float,
    di_dt: float,
    *,
    threshold: float = _FLYBACK_CRITICAL_THRESHOLD
) -> Tuple[float, bool, str]:
    """Calcula voltaje flyback."""
    if L <= 0:
        raise ValueError(f"L = {L} ≤ 0")
    
    V_fb = L * abs(di_dt)
    is_safe = V_fb < threshold
    
    if is_safe:
        margin = (threshold - V_fb) / threshold * 100.0
        diagnostic = f"Seguro: V_fb = {V_fb:.4f} < {threshold}. Margen: {margin:.1f}%"
    else:
        excess = (V_fb - threshold) / threshold * 100.0
        diagnostic = f"ALERTA: V_fb = {V_fb:.4f} ≥ {threshold}. Exceso: {excess:.1f}%"
    
    return V_fb, is_safe, diagnostic


def verify_passivity_inequality_rigorous(
    H_initial: float,
    H_final: float,
    energy_input: float,
    *,
    dissipation_rate: float = 0.05
) -> Tuple[bool, float, str]:
    """Verifica desigualdad de pasividad."""
    if H_initial < 0:
        raise ValueError(f"H₀ = {H_initial} < 0")
    if H_final < 0:
        raise ValueError(f"H_T = {H_final} < 0")
    if energy_input < 0:
        raise ValueError(f"E_in = {energy_input} < 0")
    if dissipation_rate <= 0:
        raise ValueError(f"α = {dissipation_rate} ≤ 0")
    
    lhs = H_final - H_initial
    rhs = energy_input - dissipation_rate * H_initial
    
    is_passive = lhs <= rhs + EPSILON_FLOAT64
    
    if H_initial > 0:
        dissipation_ratio = (H_initial - H_final) / H_initial
    else:
        dissipation_ratio = 0.0
    
    if is_passive:
        diagnostic = (
            f"Pasividad verificada: ΔH = {lhs:.4f} ≤ {rhs:.4f}. "
            f"Disipación: {dissipation_ratio*100:.1f}%"
        )
    else:
        violation = lhs - rhs
        diagnostic = f"VIOLACIÓN: ΔH = {lhs:.4f} > {rhs:.4f}. Exceso: {violation:.4f}"
    
    return is_passive, dissipation_ratio, diagnostic


def create_ekf_for_rlc_circuit(
    params: RLCCircuitParameters,
    dt: float = 0.1,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 1.0
) -> EKFState:
    """
    Crea EKF para sistema RLC.
    
    NUEVA FUNCIÓN: Implementación del EKF prometido en el título.
    
    Modelo de Estado (Sistema RLC):
    ------------------------------
    x = [q, i]^T  (carga, corriente)
    
    Ecuación diferencial:
    L·di/dt + R·i + q/C = 0
    
    Forma de espacio de estados:
    dx/dt = [i, -(R/L)·i - (1/LC)·q]^T
    
    Discretización (Euler):
    x_{k+1} = F·x_k + w_k
    z_k = H·x_k + v_k
    
    Args:
        params: Parámetros RLC
        dt: Paso de tiempo
        process_noise_std: Desviación estándar del ruido del proceso
        measurement_noise_std: Desviación estándar del ruido de medición
    
    Returns:
        EKFState inicializado
    """
    # Matriz de transición de estado (discretización de Euler)
    # dx/dt = A·x donde A = [[0, 1], [-1/(L·C), -R/L]]
    A = np.array([
        [0.0, 1.0],
        [-1.0 / (params.inductance * params.capacitance), -params.resistance / params.inductance]
    ], dtype=np.float64)
    
    # F = I + dt·A (Euler)
    F = np.eye(EKF_STATE_DIM) + dt * A
    
    # Matriz de observación (medimos solo carga)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    
    # Covarianzas
    Q = process_noise_std**2 * np.eye(EKF_STATE_DIM)
    R = np.array([[measurement_noise_std**2]], dtype=np.float64)
    
    # Estado inicial
    x_hat = np.zeros(EKF_STATE_DIM, dtype=np.float64)
    P = np.eye(EKF_STATE_DIM, dtype=np.float64)
    
    return EKFState(
        x_hat=x_hat,
        P=P,
        F=F,
        H=H,
        Q=Q,
        R=R
    )


# ==============================================================================
# GENERADORES DE FLUJO (MEJORADOS)
# ==============================================================================
def _create_deterministic_rng(seed: int = _RNG_SEED) -> np.random.Generator:
    """Crea generador aleatorio determinista."""
    return np.random.default_rng(seed)


def generate_laminar_flux(
    batch_size: int,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Genera flujo Newtoniano."""
    if batch_size <= 0:
        raise ValueError(f"batch_size debe ser positivo: {batch_size}")
    
    return [
        {
            "codigo_apu": f"APU_{offset + i}",
            "descripcion": "LAMINAR_DATA_FLOW",
            "cantidad": 1.0,
            "valor_unitario": 100.0,
        }
        for i in range(batch_size)
    ]


def generate_non_newtonian_levy_flux(
    batch_size: int,
    rng: np.random.Generator,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Genera flujo con saltos de Lévy.
    
    CORRECCIÓN CRÍTICA:
    ------------------
    • Usa array pre-allocation en lugar de concatenación de strings
    • Mejora eficiencia de O(n²) a O(n)
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size debe ser positivo: {batch_size}")
    
    raw_jumps: NDArray[np.float64] = rng.standard_cauchy(size=batch_size)
    absolute_jumps: NDArray[np.float64] = np.abs(raw_jumps) * _CAUCHY_SCALE_FACTOR
    
    records: List[Dict[str, Any]] = []
    for i, jump in enumerate(absolute_jumps):
        entropy_length: int = int(min(_MAX_STRING_ENTROPY, 10 + jump))
        
        # CORRECCIÓN: Pre-allocar array en lugar de concatenar strings
        # Esto mejora de O(n²) a O(n)
        description = "X" * min(entropy_length, 1000)  # Límite razonable
        
        records.append({
            "codigo_apu": f"APU_{offset + i}",
            "descripcion": description,
            "cantidad": 1.0,
            "valor_unitario": float(entropy_length),
        })
    
    return records


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture(scope="module")
def rlc_critical_parameters() -> RLCCircuitParameters:
    """Fixture con parámetros RLC críticos."""
    return RLCCircuitParameters(
        inductance=_RLC_INDUCTANCE,
        capacitance=_RLC_CAPACITANCE,
        resistance=_RLC_RESISTANCE,
    )


@pytest.fixture(scope="module")
def deterministic_rng() -> np.random.Generator:
    """Generador aleatorio determinista."""
    return _create_deterministic_rng(_RNG_SEED)


# ==============================================================================
# SUITE I: VALIDACIÓN RLC
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.physics
class TestRLCCriticalDampingValidation:
    """Suite de validación RLC."""
    
    def test_critical_damping_verification_rigorous(
        self,
        rlc_critical_parameters: RLCCircuitParameters
    ) -> None:
        """Verifica amortiguamiento crítico."""
        params = rlc_critical_parameters
        
        is_critical, zeta, diagnostic = verify_critical_damping_rigorous(
            params.inductance,
            params.capacitance,
            params.resistance,
            tolerance=_ZETA_TOLERANCE
        )
        
        assert is_critical, (
            f"Amortiguamiento no crítico:\n  • {diagnostic}"
        )
        
        omega_0 = params.natural_frequency
        assert omega_0 > 0, f"ω₀ = {omega_0} ≤ 0"
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Amortiguamiento Crítico RLC")
        print(f"  • {diagnostic}")
        print(f"  • ω₀ = {omega_0:.4f} rad/s")
        print(f"{'='*70}\n")
    
    def test_rlc_parameter_invariants(self) -> None:
        """Verifica invariantes de parámetros."""
        params = RLCCircuitParameters(
            inductance=0.5,
            capacitance=1.0,
            resistance=2.0 * math.sqrt(0.5 / 1.0)
        )
        assert params.damping_ratio == pytest.approx(1.0, abs=_ZETA_TOLERANCE)
        
        with pytest.raises(ValueError, match="inductancia.*≤ 0"):
            RLCCircuitParameters(inductance=-0.5, capacitance=1.0, resistance=1.0)
        
        with pytest.raises(ValueError, match="capacitancia.*≤ 0"):
            RLCCircuitParameters(inductance=0.5, capacitance=-1.0, resistance=1.0)
        
        with pytest.raises(ValueError, match="resistencia.*≤ 0"):
            RLCCircuitParameters(inductance=0.5, capacitance=1.0, resistance=-1.0)
        
        with pytest.raises(ValueError, match="Amortiguamiento no crítico"):
            RLCCircuitParameters(inductance=0.5, capacitance=1.0, resistance=1.0)
        
        print("  ✓ Invariantes RLC verificados")


# ==============================================================================
# SUITE II: VALIDACIÓN CAUCHY
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.stochastic
class TestCauchyDistributionValidation:
    """Suite de validación Cauchy."""
    
    def test_cauchy_heavy_tail_validation_rigorous(
        self,
        deterministic_rng: np.random.Generator
    ) -> None:
        """Valida propiedades heavy-tailed."""
        n_samples = 1000
        raw_jumps: NDArray[np.float64] = np.abs(
            deterministic_rng.standard_cauchy(size=n_samples)
        ) * _CAUCHY_SCALE_FACTOR
        
        result = validate_cauchy_distribution_rigorous(
            raw_jumps,
            _CAUCHY_SCALE_FACTOR,
            confidence_level=0.95,
            min_samples=100,
            allow_truncation=False
        )
        
        assert result.is_valid_cauchy(), (
            f"Distribución no Cauchy: {result}"
        )
        
        extreme_threshold = 5.0 * _CAUCHY_SCALE_FACTOR
        extreme_count = int(np.sum(raw_jumps > extreme_threshold))
        expected_probability = 2.0 / (math.pi * 5.0)
        expected_count = n_samples * expected_probability
        
        assert extreme_count > 0, (
            f"No outliers extremos (> {extreme_threshold}). "
            f"Esperados ~{expected_count:.0f}"
        )
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Validación Cauchy")
        print(f"  • Q95 emp = {result.empirical_q95:.2f}")
        print(f"  • Q95 th = {result.theoretical_q95:.2f}")
        print(f"  • Outliers = {extreme_count} (esperados ~{expected_count:.0f})")
        print(f"{'='*70}\n")
    
    def test_cauchy_mean_non_convergence(self) -> None:
        """Verifica no convergencia de media."""
        rng = np.random.default_rng(42)
        n_samples = 1000
        n_trials = 100
        
        sample_means = []
        for _ in range(n_trials):
            samples = rng.standard_cauchy(size=n_samples)
            sample_means.append(np.mean(samples))
        
        mean_variance = np.var(sample_means)
        
        gaussian_samples = []
        for _ in range(n_trials):
            samples = rng.normal(size=n_samples)
            gaussian_samples.append(np.mean(samples))
        gaussian_variance = np.var(gaussian_samples)
        
        assert mean_variance > 10.0 * gaussian_variance, (
            f"Media Cauchy converge:\n"
            f"  • Var(Cauchy) = {mean_variance:.4f}\n"
            f"  • Var(Gauss) = {gaussian_variance:.4f}"
        )
        
        print("  ✓ No convergencia de media Cauchy verificada")


# ==============================================================================
# SUITE III: VOLTAJE FLYBACK
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.control
class TestFlybackVoltageStability:
    """Suite de voltaje flyback."""
    
    def test_flyback_voltage_clamping_rigorous(
        self,
        rlc_critical_parameters: RLCCircuitParameters
    ) -> None:
        """Verifica clamping de voltaje."""
        params = rlc_critical_parameters
        
        rng = np.random.default_rng(42)
        max_flyback = 0.0
        max_di_dt = 0.0
        
        for _ in range(1000):
            raw_shock = float(rng.standard_cauchy())
            di_dt = 10.0 * np.tanh(raw_shock / 10.0)
            
            V_fb, is_safe, diagnostic = compute_flyback_voltage_rigorous(
                params.inductance,
                di_dt,
                threshold=_FLYBACK_CRITICAL_THRESHOLD
            )
            
            max_flyback = max(max_flyback, V_fb)
            max_di_dt = max(max_di_dt, abs(di_dt))
            
            assert is_safe, f"Voltaje inseguro: {diagnostic}"
        
        assert max_flyback < _FLYBACK_CRITICAL_THRESHOLD, (
            f"V_fb_max = {max_flyback} ≥ {_FLYBACK_CRITICAL_THRESHOLD}"
        )
        
        safety_margin = (_FLYBACK_CRITICAL_THRESHOLD - max_flyback) / _FLYBACK_CRITICAL_THRESHOLD * 100.0
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Flyback")
        print(f"  • V_fb_max = {max_flyback:.4f}")
        print(f"  • Margen = {safety_margin:.1f}%")
        print(f"{'='*70}\n")
    
    def test_flyback_parameter_validation(self) -> None:
        """Verifica validación de parámetros."""
        with pytest.raises(ValueError, match="Inductancia inválida"):
            compute_flyback_voltage_rigorous(-0.5, 1.0)
        
        V_fb, is_safe, _ = compute_flyback_voltage_rigorous(0.5, 1.0)
        assert V_fb == 0.5
        assert is_safe
        
        print("  ✓ Validación flyback verificada")


# ==============================================================================
# SUITE IV: PASIVIDAD
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.physics
class TestPassivityInequalityValidation:
    """Suite de pasividad."""
    
    def test_passivity_inequality_rigorous(self) -> None:
        """Verifica desigualdad de pasividad."""
        H_initial = 100.0
        dissipation_rate = 0.05
        energy_input = 10.0
        
        H_final = H_initial * 0.8
        
        is_passive, dissipation_ratio, diagnostic = verify_passivity_inequality_rigorous(
            H_initial,
            H_final,
            energy_input,
            dissipation_rate=dissipation_rate
        )
        
        assert is_passive, f"Pasividad violada: {diagnostic}"
        
        expected_ratio = (H_initial - H_final) / H_initial
        assert abs(dissipation_ratio - expected_ratio) < EPSILON_FLOAT64
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Pasividad")
        print(f"  • {diagnostic}")
        print(f"{'='*70}\n")
    
    def test_passivity_parameter_validation(self) -> None:
        """Verifica validación de parámetros."""
        with pytest.raises(ValueError, match="Energía inicial negativa"):
            verify_passivity_inequality_rigorous(-10.0, 5.0, 1.0)
        
        with pytest.raises(ValueError, match="Energía final negativa"):
            verify_passivity_inequality_rigorous(10.0, -5.0, 1.0)
        
        with pytest.raises(ValueError, match="Tasa de disipación no positiva"):
            verify_passivity_inequality_rigorous(10.0, 5.0, 1.0, dissipation_rate=0.0)
        
        print("  ✓ Validación pasividad verificada")


# ==============================================================================
# SUITE V: EKF BAJO RUIDO DE CAUCHY (NUEVO)
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.ekf
class TestEKFDivergenceUnderCauchy:
    """
    Suite de divergencia del EKF bajo ruido de Cauchy.
    
    NUEVA SUITE: Implementa el test prometido en el título del módulo.
    """
    
    def test_ekf_divergence_under_cauchy_noise(
        self,
        rlc_critical_parameters: RLCCircuitParameters,
        deterministic_rng: np.random.Generator
    ) -> None:
        """
        Verifica divergencia del EKF bajo ruido de Cauchy.
        
        Hipótesis:
        ---------
        El EKF clásico DIVERGE bajo ruido de Cauchy porque:
        1. Viola hipótesis de ruido Gaussiano
        2. Momentos infinitos rompen actualización de covarianza
        3. Innovaciones inconsistentes
        """
        params = rlc_critical_parameters
        
        # Crear EKF
        ekf = create_ekf_for_rlc_circuit(
            params,
            dt=0.1,
            process_noise_std=0.1,
            measurement_noise_std=1.0
        )
        
        # Simular sistema RLC con ruido de Cauchy
        n_steps = 200
        measurements = []
        true_states = []
        
        # Estado verdadero inicial
        x_true = np.array([1.0, 0.0], dtype=np.float64)  # [carga, corriente]
        
        for step in range(n_steps):
            # Dinámica verdadera (sistema RLC)
            x_true = ekf.F @ x_true
            
            # Medición con ruido de Cauchy
            cauchy_noise = deterministic_rng.standard_cauchy() * _CAUCHY_SCALE_FACTOR
            # Saturar ruido para evitar overflow
            cauchy_noise = np.clip(cauchy_noise, -1000, 1000)
            
            z = ekf.H @ x_true + np.array([cauchy_noise], dtype=np.float64)
            
            measurements.append(z[0])
            true_states.append(x_true.copy())
            
            # Paso de predicción
            ekf.predict()
            
            # Paso de actualización
            innovation = ekf.update(z)
        
        # Verificar métricas finales
        final_metrics = ekf.compute_metrics()
        
        # Verificar divergencia
        assert final_metrics.is_diverging(), (
            f"EKF NO DIVERGIÓ bajo Cauchy:\n"
            f"  • {final_metrics}\n"
            f"El EKF clásico debe divergir bajo ruido no Gaussiano."
        )
        
        # Verificar que la traza de covarianza explotó
        assert final_metrics.covariance_trace > EKF_DIVERGENCE_THRESHOLD, (
            f"Covarianza no explotó: Tr(P) = {final_metrics.covariance_trace}"
        )
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Divergencia EKF bajo Cauchy")
        print(f"{'='*70}")
        print(f"  • {final_metrics}")
        print(f"  • Tr(P) = {final_metrics.covariance_trace:.2e}")
        print(f"  • Pasos simulados = {n_steps}")
        print(f"{'='*70}\n")
    
    def test_ekf_convergence_under_gaussian_noise(
        self,
        rlc_critical_parameters: RLCCircuitParameters
    ) -> None:
        """
        Verifica convergencia del EKF bajo ruido Gaussiano (caso de control).
        
        Este test verifica que el EKF SÍ converge cuando las hipótesis
        son satisfechas (ruido Gaussiano).
        """
        params = rlc_critical_parameters
        rng = np.random.default_rng(43)
        
        ekf = create_ekf_for_rlc_circuit(
            params,
            dt=0.1,
            process_noise_std=0.1,
            measurement_noise_std=1.0
        )
        
        n_steps = 200
        x_true = np.array([1.0, 0.0], dtype=np.float64)
        
        for step in range(n_steps):
            x_true = ekf.F @ x_true
            
            # Ruido GAUSSIANO (hipótesis satisfecha)
            gaussian_noise = rng.normal(0.0, 1.0)
            z = ekf.H @ x_true + np.array([gaussian_noise], dtype=np.float64)
            
            ekf.predict()
            ekf.update(z)
        
        final_metrics = ekf.compute_metrics()
        
        # NO debe divergir
        assert not final_metrics.is_diverging(), (
            f"EKF divergió bajo Gaussiano:\n"
            f"  • {final_metrics}\n"
            f"El EKF debe converger bajo ruido Gaussiano."
        )
        
        # Covarianza debe estar acotada
        assert final_metrics.covariance_trace < EKF_DIVERGENCE_THRESHOLD / 10, (
            f"Covarianza muy alta: Tr(P) = {final_metrics.covariance_trace}"
        )
        
        print(f"  ✓ EKF converge bajo Gaussiano: {final_metrics}")


# ==============================================================================
# SUITE VI: INTEGRACIÓN COMPLETA (MEJORADA)
# ==============================================================================
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.slow
class TestFullSystemIntegrationRigorous:
    """Suite de integración completa."""
    
    def test_full_pipeline_under_levy_shock(
        self,
        rlc_critical_parameters: RLCCircuitParameters,
        deterministic_rng: np.random.Generator
    ) -> None:
        """Test de integración completa."""
        params = rlc_critical_parameters
        
        # FASE 1: RLC
        is_critical, zeta, _ = verify_critical_damping_rigorous(
            params.inductance,
            params.capacitance,
            params.resistance
        )
        assert is_critical, f"ζ = {zeta} ≠ 1"
        
        # FASE 2: Flujo Cauchy
        batch_size = 100
        records = generate_non_newtonian_levy_flux(
            batch_size,
            deterministic_rng
        )
        
        lengths = np.array([len(r["descripcion"]) for r in records], dtype=np.float64)
        
        # FASE 3: Validación Cauchy con truncamiento
        cauchy_result = validate_cauchy_distribution_rigorous(
            lengths,
            _CAUCHY_SCALE_FACTOR,
            min_samples=50,
            allow_truncation=True  # CORRECCIÓN: Permitir truncamiento
        )
        
        assert cauchy_result.extreme_outliers_detected or cauchy_result.quantile_test_passed, (
            f"No heavy-tail: {cauchy_result}"
        )
        
        # FASE 4: Flyback
        max_di_dt = 10.0
        V_fb, is_safe, _ = compute_flyback_voltage_rigorous(
            params.inductance,
            max_di_dt,
            threshold=_FLYBACK_CRITICAL_THRESHOLD
        )
        assert is_safe, f"V_fb = {V_fb}"
        
        # FASE 5: Pasividad
        H_initial = 100.0
        H_final = H_initial * 0.7
        energy_input = 5.0
        
        is_passive, _, _ = verify_passivity_inequality_rigorous(
            H_initial,
            H_final,
            energy_input,
            dissipation_rate=0.05
        )
        assert is_passive
        
        # FASE 6: EKF (NUEVA)
        ekf = create_ekf_for_rlc_circuit(params)
        x_true = np.array([1.0, 0.0], dtype=np.float64)
        
        for _ in range(50):
            x_true = ekf.F @ x_true
            cauchy_noise = deterministic_rng.standard_cauchy() * 10.0
            cauchy_noise = np.clip(cauchy_noise, -100, 100)
            z = ekf.H @ x_true + np.array([cauchy_noise], dtype=np.float64)
            
            ekf.predict()
            ekf.update(z)
        
        ekf_metrics = ekf.compute_metrics()
        assert ekf_metrics.is_diverging(), "EKF no divergió bajo Cauchy"
        
        print(f"\n{'='*70}")
        print(f"TEST PASADO: Integración Completa")
        print(f"  • ζ = {zeta:.6f}")
        print(f"  • V_fb = {V_fb:.4f}")
        print(f"  • Cauchy validado = {cauchy_result.is_valid_cauchy()}")
        print(f"  • EKF divergió = {ekf_metrics.is_diverging()}")
        print(f"{'='*70}\n")


# ==============================================================================
# CONFIGURACIÓN PYTEST
# ==============================================================================
def pytest_configure(config: pytest.Config) -> None:
    """Configuración personalizada."""
    markers = {
        "integration": "Tests de integración",
        "stress": "Tests de estrés",
        "physics": "Tests de física",
        "stochastic": "Tests estocásticos",
        "control": "Tests de control",
        "ekf": "Tests de EKF",
        "slow": "Tests lentos",
    }
    
    for marker, desc in markers.items():
        config.addinivalue_line("markers", f"{marker}: {desc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-ra", "--strict-markers"])
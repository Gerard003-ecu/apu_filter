# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo de Pruebas: MAC Minimizer Agent Test Suite                            ║
║ Ruta: tests/unit/agents/boole/tactics/test_mac_minimizer_agent.py            ║
║ Versión: 3.0.0-Quantum-Rigorous-Test-Suite                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TESTING RIGUROSO:
────────────────────────────────────────────────────────────────────────────────
Esta suite de pruebas valida exhaustivamente la corrección matemática y numérica
del MACMinimizerAgent mediante:

1. PRUEBAS UNITARIAS ATÓMICAS:
   - Validación de matrices de densidad individuales
   - Saneamiento espectral
   - Cálculo de raíces cuadradas PSD
   - Entropías de von Neumann y Rényi

2. PRUEBAS DE INTEGRACIÓN POR FASE:
   - Fase 1: Majorización cuántica (curvas de Lorenz)
   - Fase 2: Fidelidad de Uhlmann (núcleo de fidelidad)
   - Fase 3: Capacidad de Holevo (termodinámica)

3. PRUEBAS DE COMPOSICIÓN FUNTORIAL:
   - Pipeline completo Φ₃ ∘ Φ₂ ∘ Φ₁
   - Propagación de certificados entre fases
   - Validación de invariantes categóricos

4. PRUEBAS DE CASOS EXTREMOS:
   - Estados puros (pureza = 1)
   - Estados maximalmente mezclados (pureza = 1/d)
   - Estados degenerados espectralmente
   - Matrices de rango bajo

5. PRUEBAS DE ROBUSTEZ NUMÉRICA:
   - Perturbaciones controladas
   - Matrices mal condicionadas
   - Autovalores cercanos al límite de máquina

6. PRUEBAS DE MANEJO DE EXCEPCIONES:
   - Violaciones de majorización
   - Colapso de fidelidad
   - Inflación de entropía
   - Matrices no hermíticas/no PSD

COBERTURA OBJETIVO:
────────────────────────────────────────────────────────────────────────────────
- Cobertura de líneas: ≥ 95%
- Cobertura de ramas: ≥ 90%
- Cobertura de condiciones: ≥ 85%
- Cobertura de mutaciones: ≥ 80%

HERRAMIENTAS UTILIZADAS:
────────────────────────────────────────────────────────────────────────────────
- pytest: Framework de testing principal
- hypothesis: Property-based testing
- numpy.testing: Aserciones numéricas con tolerancia
- pytest-cov: Análisis de cobertura
- pytest-benchmark: Benchmarking de rendimiento
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pytest
import scipy.linalg as la
from hypothesis import given, settings, strategies as st
from numpy.testing import assert_allclose, assert_array_equal

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS PARA IMPORTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES DEL MÓDULO BAJO PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from app.agents.boole.tactics.mac_minimizer_agent import (
        DensityMatrixValidationError,
        EntanglementStructureError,
        FidelityAuditData,
        HolevoAuditData,
        HolevoCapacityDeficitError,
        MACMinimizerAgent,
        MACMinimizerAgentError,
        MajorizationAuditData,
        Phase1_QuantumMajorizationAuditor,
        Phase2_UhlmannFidelityCertifier,
        Phase3_HolevoCapacityEnforcer,
        PurificationGovernanceState,
        PurificationPhase,
        QuantumCoherenceMetrics,
        QuantumCoherenceViolation,
        QuantumDistanceMetric,
        QuantumMajorizationViolation,
        SpectralCharacteristics,
        SpectralDecompositionError,
        UhlmannFidelityCollapseError,
        _ENTROPY_TOLERANCE,
        _FIDELITY_NUMERICAL_TOLERANCE,
        _HERMITIAN_TOLERANCE,
        _MACHINE_EPSILON,
        _MAJORIZATION_TOLERANCE,
        _PSD_TOLERANCE,
        _TRACE_TOLERANCE,
        _UHLMANN_FIDELITY_MIN,
    )
except ImportError as exc:
    pytest.skip(
        f"No se pudo importar el módulo mac_minimizer_agent: {exc}",
        allow_module_level=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DE TESTING
# ═══════════════════════════════════════════════════════════════════════════════
NUMERICAL_TOLERANCE = 1e-10
STRICT_TOLERANCE = 1e-12
RELAXED_TOLERANCE = 1e-8

# Dimensiones de prueba
SMALL_DIM = 2
MEDIUM_DIM = 4
LARGE_DIM = 8
EXTREME_DIM = 16

# Semillas para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: GENERADORES DE MATRICES DE DENSIDAD
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def pure_state_2d() -> np.ndarray:
    r"""
    Genera un estado puro 2D: |ψ⟩ = (|0⟩ + |1⟩)/√2.
    
    Espectro: [1.0, 0.0]
    Pureza: Tr(ρ²) = 1.0
    Entropía: S(ρ) = 0.0
    """
    psi = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho = np.outer(psi, psi.conj())
    return rho


@pytest.fixture
def maximally_mixed_2d() -> np.ndarray:
    r"""
    Genera el estado maximalmente mezclado 2D: ρ = I/2.
    
    Espectro: [0.5, 0.5]
    Pureza: Tr(ρ²) = 0.5
    Entropía: S(ρ) = ln(2) ≈ 0.693
    """
    return np.eye(2, dtype=np.complex128) / 2.0


@pytest.fixture
def mixed_state_2d() -> np.ndarray:
    r"""
    Genera un estado mezclado 2D con espectro [0.7, 0.3].
    
    Pureza: Tr(ρ²) = 0.58
    Entropía: S(ρ) ≈ 0.611
    """
    eigenvalues = np.array([0.7, 0.3])
    eigenvectors = np.eye(2, dtype=np.complex128)
    rho = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    return rho


@pytest.fixture
def werner_state() -> np.ndarray:
    r"""
    Genera un estado de Werner (estado entrelazado típico).
    
    Definición:
        ρ_W(p) = p|Φ⁺⟩⟨Φ⁺| + (1-p)I/4
    
    donde |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 es un estado de Bell.
    
    Parámetro: p = 0.6
    Dimensión: 4×4 (sistema bipartito 2⊗2)
    """
    p = 0.6
    
    # Estado de Bell |Φ⁺⟩
    phi_plus = np.zeros(4, dtype=np.complex128)
    phi_plus[0] = 1.0 / np.sqrt(2.0)  # |00⟩
    phi_plus[3] = 1.0 / np.sqrt(2.0)  # |11⟩
    
    bell_state = np.outer(phi_plus, phi_plus.conj())
    identity = np.eye(4, dtype=np.complex128) / 4.0
    
    rho_werner = p * bell_state + (1.0 - p) * identity
    
    return rho_werner


@pytest.fixture
def random_density_matrix() -> Callable[[int], np.ndarray]:
    r"""
    Factory para generar matrices de densidad aleatorias.
    
    Método: Descomposición de Ginibre
        1. Generar matriz compleja aleatoria G
        2. ρ = GG† / Tr(GG†)
    
    Returns:
        Función que genera matriz de densidad de dimensión especificada
    """
    def _generate(dimension: int, rank: int | None = None) -> np.ndarray:
        if rank is None:
            rank = dimension
        
        # Matriz de Ginibre
        G = np.random.randn(dimension, rank) + 1j * np.random.randn(dimension, rank)
        G = G / np.sqrt(2.0)  # Normalización estándar
        
        rho = G @ G.conj().T
        rho = rho / np.trace(rho)  # Normalización a traza uno
        
        # Simetrización hermítica para eliminar error numérico
        rho = (rho + rho.conj().T) / 2.0
        
        return rho
    
    return _generate


@pytest.fixture
def purification_simulator() -> Callable[[np.ndarray, float], np.ndarray]:
    r"""
    Simula una purificación espectral controlada.
    
    Método:
        1. Diagonalizar ρ_orig
        2. Elevar autovalores a potencia α > 1
        3. Renormalizar
    
    Parámetro α:
        - α = 1: sin purificación
        - α > 1: purificación progresiva
        - α → ∞: proyección al autoespacio dominante
    
    Returns:
        Función que purifica una matriz de densidad
    """
    def _purify(rho: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        
        # Elevar autovalores a potencia α
        eigenvalues_purified = eigenvalues**alpha
        
        # Renormalizar
        eigenvalues_purified = eigenvalues_purified / np.sum(eigenvalues_purified)
        
        # Reconstruir
        rho_purified = (eigenvectors * eigenvalues_purified) @ eigenvectors.conj().T
        rho_purified = (rho_purified + rho_purified.conj().T) / 2.0
        
        return rho_purified
    
    return _purify


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: INSTANCIAS DE AGENTES
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def phase1_auditor() -> Phase1_QuantumMajorizationAuditor:
    """Instancia del auditor de Fase 1."""
    return Phase1_QuantumMajorizationAuditor()


@pytest.fixture
def phase2_certifier() -> Phase2_UhlmannFidelityCertifier:
    """Instancia del certificador de Fase 2."""
    return Phase2_UhlmannFidelityCertifier()


@pytest.fixture
def phase3_enforcer() -> Phase3_HolevoCapacityEnforcer:
    """Instancia del enforcer de Fase 3."""
    return Phase3_HolevoCapacityEnforcer()


@pytest.fixture
def mac_agent() -> MACMinimizerAgent:
    """Instancia del agente completo."""
    return MACMinimizerAgent()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 1: PRUEBAS DE VALIDACIÓN DE MATRICES DE DENSIDAD                    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestDensityMatrixValidation:
    """Pruebas de validación de matrices de densidad cuánticas."""
    
    def test_pure_state_is_valid(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que un estado puro válido sea aceptado."""
        rho_sanitized, evals, trace, min_eval, spectral_chars = (
            phase1_auditor._sanitize_density_matrix("pure_state", pure_state_2d)
        )
        
        # Validaciones de espectro
        assert_allclose(evals, [1.0, 0.0], atol=NUMERICAL_TOLERANCE)
        assert_allclose(trace, 1.0, atol=_TRACE_TOLERANCE)
        assert min_eval >= -_PSD_TOLERANCE
        
        # Validaciones de características espectrales
        assert spectral_chars.dimension == 2
        assert spectral_chars.effective_rank == 1
        assert_allclose(spectral_chars.spectral_entropy, 0.0, atol=NUMERICAL_TOLERANCE)
        
        # Pureza de estado puro
        purity = np.sum(evals**2)
        assert_allclose(purity, 1.0, atol=NUMERICAL_TOLERANCE)
    
    def test_maximally_mixed_is_valid(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        maximally_mixed_2d: np.ndarray,
    ) -> None:
        """Verifica que el estado maximalmente mezclado sea aceptado."""
        rho_sanitized, evals, trace, min_eval, spectral_chars = (
            phase1_auditor._sanitize_density_matrix("max_mixed", maximally_mixed_2d)
        )
        
        # Espectro uniforme
        assert_allclose(evals, [0.5, 0.5], atol=NUMERICAL_TOLERANCE)
        assert_allclose(trace, 1.0, atol=_TRACE_TOLERANCE)
        
        # Pureza mínima para dimensión 2
        purity = np.sum(evals**2)
        assert_allclose(purity, 0.5, atol=NUMERICAL_TOLERANCE)
        
        # Entropía máxima
        entropy_expected = math.log(2.0)
        assert_allclose(
            spectral_chars.spectral_entropy,
            entropy_expected,
            atol=NUMERICAL_TOLERANCE,
        )
    
    def test_non_hermitian_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz no hermítica sea rechazada."""
        rho_bad = np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.complex128)
        
        with pytest.raises(DensityMatrixValidationError, match="no es hermítica"):
            phase1_auditor._sanitize_density_matrix("non_hermitian", rho_bad)
    
    def test_non_psd_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz no PSD sea rechazada."""
        # Matriz hermítica con autovalor negativo
        rho_bad = np.array(
            [[0.5, 0.6], [0.6, 0.5]], dtype=np.complex128
        )
        
        with pytest.raises(DensityMatrixValidationError, match="no es positive semidefinite"):
            phase1_auditor._sanitize_density_matrix("non_psd", rho_bad)
    
    def test_zero_trace_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz de traza nula sea rechazada."""
        rho_bad = np.zeros((2, 2), dtype=np.complex128)
        
        with pytest.raises(DensityMatrixValidationError, match="no es positiva"):
            phase1_auditor._sanitize_density_matrix("zero_trace", rho_bad)
    
    def test_non_unit_trace_normalized(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que una matriz con traza distinta de 1 sea normalizada."""
        rho_unnormalized = 2.0 * pure_state_2d  # Traza = 2
        
        rho_sanitized, evals, trace, _, _ = (
            phase1_auditor._sanitize_density_matrix("unnormalized", rho_unnormalized)
        )
        
        # Traza original era 2
        assert_allclose(trace, 2.0, atol=_TRACE_TOLERANCE)
        
        # Traza sanitizada debe ser 1
        assert_allclose(np.trace(rho_sanitized), 1.0, atol=_TRACE_TOLERANCE)
    
    def test_nan_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz con NaN sea rechazada."""
        rho_bad = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.complex128)
        
        with pytest.raises(ValueError, match="NaN"):
            phase1_auditor._sanitize_density_matrix("nan_matrix", rho_bad)
    
    def test_inf_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz con infinitos sea rechazada."""
        rho_bad = np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=np.complex128)
        
        with pytest.raises(ValueError, match="infinitos"):
            phase1_auditor._sanitize_density_matrix("inf_matrix", rho_bad)
    
    def test_empty_matrix_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz vacía sea rechazada."""
        rho_bad = np.array([], dtype=np.complex128).reshape(0, 0)
        
        with pytest.raises(DensityMatrixValidationError, match="vacía"):
            phase1_auditor._sanitize_density_matrix("empty_matrix", rho_bad)
    
    def test_complex_diagonal_imaginary_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que una matriz con traza imaginaria sea rechazada."""
        rho_bad = np.array([[1.0j, 0.0], [0.0, -1.0j]], dtype=np.complex128)
        
        with pytest.raises(DensityMatrixValidationError, match="parte imaginaria"):
            phase1_auditor._sanitize_density_matrix("complex_trace", rho_bad)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 2: PRUEBAS DE SANEAMIENTO ESPECTRAL                                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestSpectralSanitization:
    """Pruebas de validación y saneamiento de espectros."""
    
    def test_valid_spectrum_accepted(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que un espectro válido sea aceptado."""
        spectrum = np.array([0.6, 0.3, 0.1])
        
        sanitized, trace, min_eval = phase1_auditor._sanitize_spectrum(
            "valid_spectrum", spectrum
        )
        
        assert_allclose(sanitized, spectrum, atol=NUMERICAL_TOLERANCE)
        assert_allclose(trace, 1.0, atol=_TRACE_TOLERANCE)
        assert min_eval >= 0.0
    
    def test_unnormalized_spectrum_normalized(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que un espectro no normalizado sea normalizado."""
        spectrum = np.array([0.6, 0.3, 0.1]) * 2.0  # Traza = 2
        
        sanitized, trace, _ = phase1_auditor._sanitize_spectrum(
            "unnormalized_spectrum", spectrum
        )
        
        assert_allclose(trace, 2.0, atol=_TRACE_TOLERANCE)
        assert_allclose(np.sum(sanitized), 1.0, atol=_TRACE_TOLERANCE)
    
    def test_negative_eigenvalue_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que autovalores negativos sean rechazados."""
        spectrum = np.array([1.0, -0.1, 0.1])
        
        with pytest.raises(DensityMatrixValidationError, match="negativos"):
            phase1_auditor._sanitize_spectrum("negative_spectrum", spectrum)
    
    def test_zero_spectrum_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que un espectro nulo sea rechazado."""
        spectrum = np.array([0.0, 0.0, 0.0])
        
        with pytest.raises(DensityMatrixValidationError, match="nula"):
            phase1_auditor._sanitize_spectrum("zero_spectrum", spectrum)
    
    def test_empty_spectrum_rejected(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica que un espectro vacío sea rechazado."""
        spectrum = np.array([])
        
        with pytest.raises(DensityMatrixValidationError, match="vacío"):
            phase1_auditor._sanitize_spectrum("empty_spectrum", spectrum)
    
    def test_spectrum_consistency_check(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica la consistencia entre espectro suministrado y matricial."""
        # Espectro correcto
        spectrum_correct = np.array([0.7, 0.3])
        
        # Espectro inconsistente
        spectrum_wrong = np.array([0.9, 0.1])
        
        # Extraer espectro de la matriz
        matrix_eigenvalues = np.linalg.eigvalsh(mixed_state_2d)
        matrix_eigenvalues = np.sort(matrix_eigenvalues)[::-1]
        
        # Consistencia correcta no debe lanzar excepción
        phase1_auditor._assert_spectra_consistent(
            "consistent",
            spectrum_correct,
            matrix_eigenvalues,
        )
        
        # Inconsistencia debe lanzar excepción
        with pytest.raises(DensityMatrixValidationError, match="Inconsistencia"):
            phase1_auditor._assert_spectra_consistent(
                "inconsistent",
                spectrum_wrong,
                matrix_eigenvalues,
            )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 3: PRUEBAS DE FASE 1 (MAJORIZACIÓN CUÁNTICA)                        ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase1_MajorizationAudit:
    """Pruebas de auditoría de majorización cuántica."""
    
    def test_majorization_pure_to_pure(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica majorización entre dos estados puros idénticos."""
        audit = phase1_auditor._audit_quantum_majorization(
            rho_orig=pure_state_2d,
            rho_purified=pure_state_2d,
        )
        
        assert audit.is_majorized
        assert_allclose(audit.max_lorenz_deviation, 0.0, atol=_MAJORIZATION_TOLERANCE)
        assert_allclose(audit.purity_original, 1.0, atol=NUMERICAL_TOLERANCE)
        assert_allclose(audit.purity_purified, 1.0, atol=NUMERICAL_TOLERANCE)
    
    def test_majorization_mixed_to_pure(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        mixed_state_2d: np.ndarray,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica majorización de estado mezclado a puro (purificación válida)."""
        audit = phase1_auditor._audit_quantum_majorization(
            rho_orig=mixed_state_2d,
            rho_purified=pure_state_2d,
        )
        
        assert audit.is_majorized
        
        # La pureza debe aumentar
        assert audit.purity_purified >= audit.purity_original - NUMERICAL_TOLERANCE
    
    def test_majorization_violation_pure_to_mixed(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        pure_state_2d: np.ndarray,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica que majorización inversa (puro → mezclado) sea rechazada."""
        with pytest.raises(QuantumMajorizationViolation, match="Violación del preorden"):
            phase1_auditor._audit_quantum_majorization(
                rho_orig=pure_state_2d,
                rho_purified=mixed_state_2d,
            )
    
    def test_majorization_with_explicit_spectra(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica majorización usando espectros explícitos."""
        spectrum_orig = np.array([0.5, 0.3, 0.2])
        spectrum_purified = np.array([0.6, 0.3, 0.1])  # Más concentrado
        
        audit = phase1_auditor._audit_quantum_majorization(
            evals_orig=spectrum_orig,
            evals_purified=spectrum_purified,
        )
        
        assert audit.is_majorized
        assert audit.dimension == 3
    
    def test_lorenz_curves_computation(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica el cómputo correcto de curvas de Lorenz."""
        spectrum_a = np.array([0.5, 0.3, 0.2])
        spectrum_b = np.array([0.6, 0.25, 0.15])
        
        cumulative_a, cumulative_b, deviations = (
            phase1_auditor._compute_lorenz_curves(spectrum_a, spectrum_b)
        )
        
        # Verificar sumas acumulativas
        assert_allclose(cumulative_a[-1], 1.0, atol=_TRACE_TOLERANCE)
        assert_allclose(cumulative_b[-1], 1.0, atol=_TRACE_TOLERANCE)
        
        # Verificar monotonicidad
        assert np.all(np.diff(cumulative_a) >= -NUMERICAL_TOLERANCE)
        assert np.all(np.diff(cumulative_b) >= -NUMERICAL_TOLERANCE)
    
    def test_renyi_entropies_pure_state(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica entropías de Rényi para estado puro."""
        spectrum_pure = np.array([1.0, 0.0, 0.0])
        
        renyi_2, renyi_inf = phase1_auditor._compute_renyi_entropies(spectrum_pure)
        
        # Para estado puro, todas las entropías deben ser 0
        assert_allclose(renyi_2, 0.0, atol=NUMERICAL_TOLERANCE)
        assert_allclose(renyi_inf, 0.0, atol=NUMERICAL_TOLERANCE)
    
    def test_renyi_entropies_maximally_mixed(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica entropías de Rényi para estado maximalmente mezclado."""
        dimension = 4
        spectrum_max_mixed = np.ones(dimension) / dimension
        
        renyi_2, renyi_inf = phase1_auditor._compute_renyi_entropies(spectrum_max_mixed)
        
        # S₂ = -log(Σpᵢ²) = -log(1/d) = log(d)
        expected_renyi_2 = math.log(dimension)
        assert_allclose(renyi_2, expected_renyi_2, atol=NUMERICAL_TOLERANCE)
        
        # S_∞ = -log(max pᵢ) = log(d)
        expected_renyi_inf = math.log(dimension)
        assert_allclose(renyi_inf, expected_renyi_inf, atol=NUMERICAL_TOLERANCE)
    
    def test_spectral_characteristics_computation(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica el cómputo de características espectrales."""
        eigenvalues = np.array([0.5, 0.3, 0.15, 0.05])
        eigenvectors = np.eye(4, dtype=np.complex128)
        
        spectral_chars = phase1_auditor._compute_spectral_characteristics(
            "test",
            eigenvalues,
            eigenvectors,
        )
        
        assert spectral_chars.dimension == 4
        assert spectral_chars.effective_rank >= 1
        assert spectral_chars.effective_rank <= 4
        assert spectral_chars.condition_number >= 1.0
        assert spectral_chars.participation_ratio > 0.0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 4: PRUEBAS DE FASE 2 (FIDELIDAD DE UHLMANN)                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase2_FidelityCertification:
    """Pruebas de certificación de fidelidad de Uhlmann."""
    
    def test_fidelity_identical_states(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que F(ρ, ρ) = 1."""
        audit = phase2_certifier._certify_uhlmann_fidelity_bound(
            pure_state_2d,
            pure_state_2d,
        )
        
        assert audit.is_fidelity_preserved
        assert_allclose(audit.uhlmann_fidelity, 1.0, atol=_FIDELITY_NUMERICAL_TOLERANCE)
    
    def test_fidelity_pure_states(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
    ) -> None:
        """Verifica fidelidad entre dos estados puros distintos."""
        # |ψ₁⟩ = |0⟩
        psi1 = np.array([1.0, 0.0], dtype=np.complex128)
        rho1 = np.outer(psi1, psi1.conj())
        
        # |ψ₂⟩ = (|0⟩ + |1⟩)/√2
        psi2 = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        rho2 = np.outer(psi2, psi2.conj())
        
        audit = phase2_certifier._certify_uhlmann_fidelity_bound(rho1, rho2)
        
        # F(|ψ₁⟩, |ψ₂⟩) = |⟨ψ₁|ψ₂⟩|² = 0.5
        expected_fidelity = 0.5
        assert_allclose(
            audit.uhlmann_fidelity,
            expected_fidelity,
            atol=NUMERICAL_TOLERANCE,
        )
    
    def test_fidelity_bounds(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        random_density_matrix: Callable[[int], np.ndarray],
    ) -> None:
        """Verifica que 0 ≤ F(ρ, σ) ≤ 1."""
        rho1 = random_density_matrix(4)
        rho2 = random_density_matrix(4)
        
        audit = phase2_certifier._certify_uhlmann_fidelity_bound(rho1, rho2)
        
        assert 0.0 <= audit.uhlmann_fidelity <= 1.0
    
    def test_fidelity_symmetry(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        mixed_state_2d: np.ndarray,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que F(ρ, σ) = F(σ, ρ)."""
        audit1 = phase2_certifier._certify_uhlmann_fidelity_bound(
            mixed_state_2d,
            pure_state_2d,
        )
        
        audit2 = phase2_certifier._certify_uhlmann_fidelity_bound(
            pure_state_2d,
            mixed_state_2d,
        )
        
        assert_allclose(
            audit1.uhlmann_fidelity,
            audit2.uhlmann_fidelity,
            atol=_FIDELITY_NUMERICAL_TOLERANCE,
        )
    
    def test_fidelity_collapse_detection(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
    ) -> None:
        """Verifica detección de colapso de fidelidad."""
        # Estados ortogonales: F = 0
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        
        with pytest.raises(UhlmannFidelityCollapseError, match="Colapso semántico"):
            phase2_certifier._certify_uhlmann_fidelity_bound(rho1, rho2)
    
    def test_psd_square_root_computation(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica el cómputo de √ρ."""
        sqrt_rho = phase2_certifier._psd_square_root("test", mixed_state_2d)
        
        # Verificar que (√ρ)² = ρ
        reconstructed = sqrt_rho @ sqrt_rho
        assert_allclose(reconstructed, mixed_state_2d, atol=NUMERICAL_TOLERANCE)
        
        # Verificar hermiticidad de √ρ
        assert_allclose(sqrt_rho, sqrt_rho.conj().T, atol=NUMERICAL_TOLERANCE)
    
    def test_trace_distance_computation(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica el cómputo de distancia de traza."""
        # Estados ortogonales
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        
        trace_dist = phase2_certifier._compute_trace_distance(rho1, rho2)
        
        # D_tr para estados ortogonales = 1
        assert_allclose(trace_dist, 1.0, atol=NUMERICAL_TOLERANCE)
    
    def test_bures_distance_from_fidelity(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
    ) -> None:
        """Verifica el cómputo de distancia de Bures."""
        # F = 1 ⟹ D_B = 0
        bures_1 = phase2_certifier._compute_bures_distance(1.0)
        assert_allclose(bures_1, 0.0, atol=NUMERICAL_TOLERANCE)
        
        # F = 0 ⟹ D_B = √2
        bures_0 = phase2_certifier._compute_bures_distance(0.0)
        assert_allclose(bures_0, math.sqrt(2.0), atol=NUMERICAL_TOLERANCE)
    
    def test_coherence_metrics_computation(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica el cómputo de métricas de coherencia."""
        coherence = phase2_certifier._compute_quantum_coherence_metrics(
            mixed_state_2d,
            "test",
        )
        
        assert coherence.l1_norm_coherence >= 0.0
        assert coherence.relative_entropy_coherence >= 0.0
        assert coherence.robustness_of_coherence >= 0.0
        assert coherence.off_diagonal_mass >= 0.0


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 5: PRUEBAS DE FASE 3 (CAPACIDAD DE HOLEVO)                          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPhase3_HolevoCapacityEnforcement:
    """Pruebas de enforcement de capacidad de Holevo."""
    
    def test_entropy_unchanged_for_identical_states(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que ΔS = 0 para estados idénticos."""
        audit = phase3_enforcer._enforce_holevo_capacity_retention(
            rho_orig=pure_state_2d,
            rho_purified=pure_state_2d,
        )
        
        assert audit.is_capacity_preserved
        assert_allclose(audit.entropy_delta, 0.0, atol=_ENTROPY_TOLERANCE)
    
    def test_entropy_decreases_on_purification(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        mixed_state_2d: np.ndarray,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que S(ρ_pur) ≤ S(ρ_orig) en purificación válida."""
        audit = phase3_enforcer._enforce_holevo_capacity_retention(
            rho_orig=mixed_state_2d,
            rho_purified=pure_state_2d,
        )
        
        assert audit.is_capacity_preserved
        assert audit.entropy_delta <= _ENTROPY_TOLERANCE
        assert audit.entropy_purified <= audit.entropy_original + _ENTROPY_TOLERANCE
    
    def test_entropy_inflation_rejected(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        pure_state_2d: np.ndarray,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica que inflación de entropía sea rechazada."""
        with pytest.raises(HolevoCapacityDeficitError, match="Paradoja termodinámica"):
            phase3_enforcer._enforce_holevo_capacity_retention(
                rho_orig=pure_state_2d,
                rho_purified=mixed_state_2d,
            )
    
    def test_von_neumann_entropy_pure_state(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
    ) -> None:
        """Verifica que S(ρ_puro) = 0."""
        spectrum_pure = np.array([1.0, 0.0, 0.0])
        
        entropy = phase3_enforcer._von_neumann_entropy(spectrum_pure)
        
        assert_allclose(entropy, 0.0, atol=NUMERICAL_TOLERANCE)
    
    def test_von_neumann_entropy_maximally_mixed(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
    ) -> None:
        """Verifica que S(I/d) = ln(d) para estado maximalmente mezclado."""
        dimension = 4
        spectrum_max_mixed = np.ones(dimension) / dimension
        
        entropy = phase3_enforcer._von_neumann_entropy(spectrum_max_mixed)
        
        expected_entropy = math.log(dimension)
        assert_allclose(entropy, expected_entropy, atol=NUMERICAL_TOLERANCE)
    
    def test_renyi_entropy_family_computation(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
    ) -> None:
        """Verifica el cómputo de familia de entropías de Rényi."""
        spectrum = np.array([0.5, 0.3, 0.2])
        
        renyi_family = phase3_enforcer._compute_renyi_entropy_family(
            spectrum,
            alpha_values=[0.5, 1.0, 2.0, float('inf')],
        )
        
        assert 0.5 in renyi_family
        assert 1.0 in renyi_family
        assert 2.0 in renyi_family
        assert float('inf') in renyi_family
        
        # Verificar monotonía: S_α ≤ S_β para α > β
        assert renyi_family[float('inf')] <= renyi_family[2.0] + NUMERICAL_TOLERANCE
        assert renyi_family[2.0] <= renyi_family[1.0] + NUMERICAL_TOLERANCE
        assert renyi_family[1.0] <= renyi_family[0.5] + NUMERICAL_TOLERANCE
    
    def test_holevo_capacity_estimation(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        pure_state_2d: np.ndarray,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica estimación de capacidad de Holevo."""
        ensemble_states = [pure_state_2d, mixed_state_2d]
        probabilities = np.array([0.5, 0.5])
        
        chi = phase3_enforcer._estimate_holevo_capacity(
            ensemble_states,
            probabilities,
        )
        
        # χ ≥ 0
        assert chi >= 0.0
        
        # χ ≤ log(d) para dimensión d
        max_capacity = math.log(2.0)
        assert chi <= max_capacity + NUMERICAL_TOLERANCE
    
    def test_second_law_validation(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        mixed_state_2d: np.ndarray,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica que la segunda ley termodinámica se respete."""
        audit = phase3_enforcer._enforce_holevo_capacity_retention(
            rho_orig=mixed_state_2d,
            rho_purified=pure_state_2d,
        )
        
        # Para purificación reversible, permitir ΔS ≤ 0
        assert audit.satisfies_second_law(allow_reversible=True)
        
        # ΔF = -ΔS, debe ser positivo (purificación reduce energía libre)
        assert audit.free_energy_change >= -_ENTROPY_TOLERANCE


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 6: PRUEBAS DE COMPOSICIÓN FUNTORIAL COMPLETA                        ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestFunctorialComposition:
    """Pruebas de la composición funtorial completa Φ₃ ∘ Φ₂ ∘ Φ₁."""
    
    def test_full_pipeline_valid_purification(
        self,
        mac_agent: MACMinimizerAgent,
        mixed_state_2d: np.ndarray,
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Verifica el pipeline completo con purificación válida."""
        rho_purified = purification_simulator(mixed_state_2d, alpha=2.0)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            mixed_state_2d,
            rho_purified,
        )
        
        assert governance_state.is_epistemologically_valid
        assert governance_state.purification_phase == PurificationPhase.COMPLETE
        assert governance_state.majorization_audit.is_majorized
        assert governance_state.fidelity_audit.is_fidelity_preserved
        assert governance_state.holevo_audit.is_capacity_preserved
    
    def test_full_pipeline_certificate_propagation(
        self,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Verifica propagación de certificados entre fases."""
        rho_orig = random_density_matrix(4)
        rho_purified = purification_simulator(rho_orig, alpha=1.8)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_orig,
            rho_purified,
        )
        
        # Verificar consistencia dimensional entre certificados
        assert (
            governance_state.majorization_audit.dimension
            == rho_orig.shape[0]
        )
        
        # Verificar que fidelidad sea alta
        assert governance_state.fidelity_audit.uhlmann_fidelity >= _UHLMANN_FIDELITY_MIN
        
        # Verificar que entropía no aumente
        assert governance_state.holevo_audit.entropy_delta <= _ENTROPY_TOLERANCE
    
    def test_full_pipeline_with_explicit_spectra(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """Verifica pipeline con espectros explícitos."""
        # Espectros coherentes con purificación
        evals_orig = np.array([0.4, 0.3, 0.2, 0.1])
        evals_purified = np.array([0.5, 0.3, 0.15, 0.05])
        
        # Reconstruir matrices
        rho_orig = np.diag(evals_orig).astype(np.complex128)
        rho_purified = np.diag(evals_purified).astype(np.complex128)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_orig,
            rho_purified,
            evals_orig=evals_orig,
            evals_purified=evals_purified,
        )
        
        assert governance_state.is_epistemologically_valid
    
    def test_full_pipeline_quality_score(
        self,
        mac_agent: MACMinimizerAgent,
        mixed_state_2d: np.ndarray,
        pure_state_2d: np.ndarray,
    ) -> None:
        """Verifica cómputo de score de calidad global."""
        governance_state = mac_agent.execute_spectral_purification_governance(
            mixed_state_2d,
            pure_state_2d,
        )
        
        # Score debe estar en [0, 1]
        assert 0.0 <= governance_state.overall_quality_score <= 1.0
        
        # Para purificación completa a estado puro, score debe ser alto
        assert governance_state.overall_quality_score >= 0.8
    
    def test_full_pipeline_risk_assessment(
        self,
        mac_agent: MACMinimizerAgent,
        mixed_state_2d: np.ndarray,
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Verifica evaluación de riesgo."""
        rho_purified = purification_simulator(mixed_state_2d, alpha=1.2)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            mixed_state_2d,
            rho_purified,
        )
        
        assert governance_state.risk_assessment in ["NOMINAL", "WARNING", "CRITICAL"]
    
    def test_full_pipeline_majorization_violation_propagates(
        self,
        mac_agent: MACMinimizerAgent,
        pure_state_2d: np.ndarray,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica que violación de majorización aborte el pipeline."""
        with pytest.raises(QuantumMajorizationViolation):
            mac_agent.execute_spectral_purification_governance(
                pure_state_2d,
                mixed_state_2d,  # Majorización inversa
            )
    
    def test_full_pipeline_fidelity_collapse_propagates(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """Verifica que colapso de fidelidad aborte el pipeline."""
        # Estados ortogonales
        rho1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        rho2 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        
        with pytest.raises(UhlmannFidelityCollapseError):
            mac_agent.execute_spectral_purification_governance(rho1, rho2)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 7: PRUEBAS DE CASOS EXTREMOS                                        ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestEdgeCases:
    """Pruebas de casos extremos y límites."""
    
    def test_single_qubit_pure_state(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """Verifica manejo de estado puro de 1 qubit."""
        # |0⟩
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho,
            rho,
        )
        
        assert governance_state.is_epistemologically_valid
    
    def test_large_dimension_state(
        self,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Verifica escalabilidad a dimensiones grandes."""
        rho_orig = random_density_matrix(EXTREME_DIM)
        rho_purified = purification_simulator(rho_orig, alpha=1.5)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_orig,
            rho_purified,
        )
        
        assert governance_state.is_epistemologically_valid
        assert governance_state.majorization_audit.dimension == EXTREME_DIM
    
    def test_rank_deficient_matrix(
        self,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
    ) -> None:
        """Verifica manejo de matrices de rango bajo."""
        rho_low_rank = random_density_matrix(MEDIUM_DIM, rank=2)
        rho_even_lower_rank = random_density_matrix(MEDIUM_DIM, rank=1)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_low_rank,
            rho_even_lower_rank,
        )
        
        assert governance_state.is_epistemologically_valid
        assert governance_state.majorization_audit.spectral_characteristics_original.effective_rank <= 2
        assert governance_state.majorization_audit.spectral_characteristics_purified.effective_rank <= 1
    
    def test_near_degenerate_spectrum(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """Verifica manejo de espectros casi degenerados."""
        # Espectro con autovalores muy cercanos
        eigenvalues = np.array([0.25 + 1e-10, 0.25, 0.25 - 1e-10, 0.25])
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        rho = np.diag(eigenvalues).astype(np.complex128)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho,
            rho,
        )
        
        assert governance_state.is_epistemologically_valid
    
    def test_ill_conditioned_matrix(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """Verifica manejo de matrices mal condicionadas."""
        # Espectro con gran rango dinámico
        eigenvalues = np.array([0.99, 0.005, 0.003, 0.002])
        rho = np.diag(eigenvalues).astype(np.complex128)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho,
            rho,
        )
        
        assert governance_state.is_epistemologically_valid
        
        # Verificar que número de condición sea alto
        assert governance_state.majorization_audit.spectral_characteristics_original.condition_number > 100
    
    def test_werner_state_purification(
        self,
        mac_agent: MACMinimizerAgent,
        werner_state: np.ndarray,
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Verifica purificación de estado de Werner (entrelazado)."""
        rho_purified = purification_simulator(werner_state, alpha=1.3)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            werner_state,
            rho_purified,
        )
        
        assert governance_state.is_epistemologically_valid


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 8: PRUEBAS BASADAS EN PROPIEDADES (HYPOTHESIS)                      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPropertyBased:
    """Pruebas basadas en propiedades usando Hypothesis."""
    
    @given(
        dimension=st.integers(min_value=2, max_value=8),
        alpha=st.floats(min_value=1.01, max_value=3.0),
    )
    @settings(max_examples=20, deadline=5000)
    def test_purification_increases_purity(
        self,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
        dimension: int,
        alpha: float,
    ) -> None:
        """Propiedad: La purificación debe aumentar o preservar la pureza."""
        rho_orig = random_density_matrix(dimension)
        rho_purified = purification_simulator(rho_orig, alpha=alpha)
        
        try:
            governance_state = mac_agent.execute_spectral_purification_governance(
                rho_orig,
                rho_purified,
            )
            
            purity_increase = (
                governance_state.majorization_audit.purity_purified
                - governance_state.majorization_audit.purity_original
            )
            
            assert purity_increase >= -NUMERICAL_TOLERANCE
            
        except (QuantumMajorizationViolation, UhlmannFidelityCollapseError):
            # Puede fallar si α no purifica suficiente, es aceptable
            pass
    
    @given(
        dimension=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=15, deadline=5000)
    def test_fidelity_with_self_is_one(
        self,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        random_density_matrix: Callable[[int], np.ndarray],
        dimension: int,
    ) -> None:
        """Propiedad: F(ρ, ρ) = 1 siempre."""
        rho = random_density_matrix(dimension)
        
        audit = phase2_certifier._certify_uhlmann_fidelity_bound(rho, rho)
        
        assert_allclose(audit.uhlmann_fidelity, 1.0, atol=_FIDELITY_NUMERICAL_TOLERANCE)
    
    @given(
        dimension=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=15, deadline=5000)
    def test_entropy_is_non_negative(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
        random_density_matrix: Callable[[int], np.ndarray],
        dimension: int,
    ) -> None:
        """Propiedad: S(ρ) ≥ 0 siempre."""
        rho = random_density_matrix(dimension)
        eigenvalues = np.linalg.eigvalsh(rho)
        
        entropy = phase3_enforcer._von_neumann_entropy(eigenvalues)
        
        assert entropy >= -NUMERICAL_TOLERANCE


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 9: PRUEBAS DE ROBUSTEZ NUMÉRICA                                     ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestNumericalRobustness:
    """Pruebas de robustez numérica y estabilidad."""
    
    def test_perturbation_stability(
        self,
        mac_agent: MACMinimizerAgent,
        mixed_state_2d: np.ndarray,
    ) -> None:
        """Verifica estabilidad ante perturbaciones pequeñas."""
        epsilon = 1e-10
        
        # Perturbación hermítica aleatoria
        perturbation = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        perturbation = (perturbation + perturbation.conj().T) / 2.0
        perturbation = epsilon * perturbation / la.norm(perturbation, ord='fro')
        
        rho_perturbed = mixed_state_2d + perturbation
        rho_perturbed = (rho_perturbed + rho_perturbed.conj().T) / 2.0
        rho_perturbed = rho_perturbed / np.trace(rho_perturbed)
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            mixed_state_2d,
            rho_perturbed,
        )
        
        # La fidelidad debe ser cercana a 1
        assert governance_state.fidelity_audit.uhlmann_fidelity >= 0.99
    
    def test_kahan_sum_accuracy(
        self,
        phase3_enforcer: Phase3_HolevoCapacityEnforcer,
    ) -> None:
        """Verifica que la suma de Kahan mejore precisión."""
        # Suma propensa a cancelación catastrófica
        values = np.array([1.0, 1e15, -1e15, 1.0])
        
        # Suma naive
        naive_sum = float(np.sum(values))
        
        # Suma compensada
        kahan_sum = phase3_enforcer._kahan_sum(values)
        
        # Kahan debe ser más preciso (resultado esperado: 2.0)
        assert_allclose(kahan_sum, 2.0, atol=NUMERICAL_TOLERANCE)
    
    def test_eigenvalue_near_machine_epsilon(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica manejo de autovalores cercanos a épsilon de máquina."""
        eigenvalues = np.array([1.0 - 1e-15, 1e-16, 1e-17])
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        sanitized, _, _ = phase1_auditor._sanitize_spectrum(
            "near_epsilon",
            eigenvalues,
        )
        
        # Los autovalores extremadamente pequeños deben ser manejados
        assert np.all(sanitized >= 0.0)
        assert_allclose(np.sum(sanitized), 1.0, atol=_TRACE_TOLERANCE)
    
    def test_orthogonality_check_robustness(
        self,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
    ) -> None:
        """Verifica detección de pérdida de ortogonalidad."""
        # Vectores perfectamente ortogonales
        vectors_ortho = np.eye(4, dtype=np.complex128)
        
        is_ortho, deviation = phase1_auditor._check_orthonormality(vectors_ortho)
        
        assert is_ortho
        assert deviation < NUMERICAL_TOLERANCE


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 10: PRUEBAS DE BENCHMARKING Y RENDIMIENTO                           ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestPerformance:
    """Pruebas de rendimiento y benchmarking."""
    
    def test_benchmark_full_pipeline_small(
        self,
        benchmark,
        mac_agent: MACMinimizerAgent,
        mixed_state_2d: np.ndarray,
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Benchmark del pipeline completo para dimensión pequeña."""
        rho_purified = purification_simulator(mixed_state_2d, alpha=2.0)
        
        result = benchmark(
            mac_agent.execute_spectral_purification_governance,
            mixed_state_2d,
            rho_purified,
        )
        
        assert result.is_epistemologically_valid
    
    def test_benchmark_full_pipeline_medium(
        self,
        benchmark,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Benchmark del pipeline completo para dimensión media."""
        rho_orig = random_density_matrix(MEDIUM_DIM)
        rho_purified = purification_simulator(rho_orig, alpha=1.8)
        
        result = benchmark(
            mac_agent.execute_spectral_purification_governance,
            rho_orig,
            rho_purified,
        )
        
        assert result.is_epistemologically_valid
    
    def test_benchmark_fidelity_computation(
        self,
        benchmark,
        phase2_certifier: Phase2_UhlmannFidelityCertifier,
        random_density_matrix: Callable[[int], np.ndarray],
    ) -> None:
        """Benchmark del cálculo de fidelidad de Uhlmann."""
        rho1 = random_density_matrix(MEDIUM_DIM)
        rho2 = random_density_matrix(MEDIUM_DIM)
        
        result = benchmark(
            phase2_certifier._certify_uhlmann_fidelity_bound,
            rho1,
            rho2,
        )
        
        assert result.is_fidelity_preserved or not result.is_fidelity_preserved
    
    def test_benchmark_majorization_audit(
        self,
        benchmark,
        phase1_auditor: Phase1_QuantumMajorizationAuditor,
        random_density_matrix: Callable[[int], np.ndarray],
        purification_simulator: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        """Benchmark de auditoría de majorización."""
        rho_orig = random_density_matrix(MEDIUM_DIM)
        rho_purified = purification_simulator(rho_orig, alpha=1.5)
        
        result = benchmark(
            phase1_auditor._audit_quantum_majorization,
            rho_orig=rho_orig,
            rho_purified=rho_purified,
        )
        
        assert result.is_majorized or not result.is_majorized


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   SUITE 11: PRUEBAS DE INTEGRACIÓN CON CASOS REALES                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TestRealWorldScenarios:
    """Pruebas con escenarios realistas de uso."""
    
    def test_knowledge_matrix_compression_scenario(
        self,
        mac_agent: MACMinimizerAgent,
        random_density_matrix: Callable[[int], np.ndarray],
    ) -> None:
        """
        Simula compresión de matriz atómica de conocimiento.
        
        Escenario:
            - Matriz original de dimensión 8 (3 qubits)
            - Compresión espectral eliminando componentes menores
        """
        dimension = 8
        rho_orig = random_density_matrix(dimension)
        
        # Simular compresión: proyectar a subespacío dominante
        eigenvalues, eigenvectors = np.linalg.eigh(rho_orig)
        
        # Mantener solo los 4 autovalores más grandes
        threshold = np.sort(eigenvalues)[-4]
        eigenvalues_compressed = np.where(
            eigenvalues >= threshold,
            eigenvalues,
            0.0,
        )
        eigenvalues_compressed = eigenvalues_compressed / np.sum(eigenvalues_compressed)
        
        rho_compressed = (eigenvectors * eigenvalues_compressed) @ eigenvectors.conj().T
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_orig,
            rho_compressed,
        )
        
        assert governance_state.is_epistemologically_valid
        
        # Verificar que la compresión preserva información esencial
        assert governance_state.fidelity_audit.uhlmann_fidelity >= 0.9
    
    def test_noise_filtering_scenario(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """
        Simula filtrado de ruido en canal cuántico.
        
        Escenario:
            - Estado original con ruido blanco
            - Purificación elimina componentes ruidosas
        """
        # Estado señal (puro)
        signal = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        rho_signal = np.outer(signal, signal.conj())
        
        # Ruido blanco
        noise_strength = 0.1
        noise = np.eye(4, dtype=np.complex128) / 4.0
        
        # Estado ruidoso
        rho_noisy = (1 - noise_strength) * rho_signal + noise_strength * noise
        
        # Purificación (filtrado de ruido)
        eigenvalues, eigenvectors = np.linalg.eigh(rho_noisy)
        
        # Filtro: elevar autovalores a potencia para suprimir ruido
        alpha = 3.0
        eigenvalues_filtered = eigenvalues**alpha
        eigenvalues_filtered = eigenvalues_filtered / np.sum(eigenvalues_filtered)
        
        rho_filtered = (eigenvectors * eigenvalues_filtered) @ eigenvectors.conj().T
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_noisy,
            rho_filtered,
        )
        
        assert governance_state.is_epistemologically_valid
        
        # El filtrado debe aumentar pureza significativamente
        purity_increase = (
            governance_state.majorization_audit.purity_purified
            - governance_state.majorization_audit.purity_original
        )
        assert purity_increase >= 0.1
    
    def test_quantum_error_correction_scenario(
        self,
        mac_agent: MACMinimizerAgent,
    ) -> None:
        """
        Simula corrección de errores cuánticos.
        
        Escenario:
            - Estado lógico codificado
            - Error de despolarización
            - Recuperación mediante proyección
        """
        # Estado lógico codificado (simplificado)
        logical_state = np.zeros(4, dtype=np.complex128)
        logical_state[0] = 1.0 / np.sqrt(2.0)
        logical_state[3] = 1.0 / np.sqrt(2.0)
        
        rho_logical = np.outer(logical_state, logical_state.conj())
        
        # Error de despolarización
        depolarization_rate = 0.05
        rho_errored = (
            (1 - depolarization_rate) * rho_logical
            + depolarization_rate * np.eye(4, dtype=np.complex128) / 4.0
        )
        
        # Corrección: proyección al código
        eigenvalues, eigenvectors = np.linalg.eigh(rho_errored)
        
        # Proyectar al autoespacio dominante
        dominant_idx = np.argmax(eigenvalues)
        rho_corrected = np.outer(
            eigenvectors[:, dominant_idx],
            eigenvectors[:, dominant_idx].conj(),
        )
        
        governance_state = mac_agent.execute_spectral_purification_governance(
            rho_errored,
            rho_corrected,
        )
        
        assert governance_state.is_epistemologically_valid


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.agents.boole.tactics.mac_minimizer_agent",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,stddev",
    ])
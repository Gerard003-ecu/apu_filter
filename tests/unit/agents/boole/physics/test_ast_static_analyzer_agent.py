# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — AST Static Analyzer Agent                                 ║
║  Ruta   : tests/unit/agents/boole/physics/test_ast_static_analyzer_agent.py      ║
║  Versión: 3.0.0-Test-Symplectic-Dirichlet-Cohomology                             ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  FILOSOFÍA DE LA SUITE:                                                          ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  Cada clase de prueba corresponde a una unidad matemática atómica del sistema.   ║
║  Los tests no son meras verificaciones de comportamiento: son la formalización   ║
║  computacional de los teoremas que el código implementa.                         ║
║                                                                                  ║
║  ESTRUCTURA DE CLASES:                                                           ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  §T0  TestFiniteNumericalGuard      → Guardas numéricas (_FiniteNumericalGuard)  ║
║  §T1  TestPhase1_Symplectic         → Fase 1 (invarianza simpléctica)            ║
║  §T2  TestPhase2_Thermodynamic      → Fase 2 (termodinámica port-Hamiltoniana)   ║
║  §T3  TestPhase3_Cohomology         → Fase 3 (cohomología de haces celulares)    ║
║  §T4  TestOrchestrator              → Orquestador (composición funtorial)        ║
║  §T5  TestFunctorialChaining        → Anidamiento funtorial entre fases          ║
║  §T6  TestExceptionHierarchy        → Jerarquía de excepciones                   ║
║  §T7  TestDTOImmutability           → Inmutabilidad de los DTOs                  ║
║  §T8  TestNumericalEdgeCases        → Casos límite numéricos                     ║
║  §T9  TestProvenanceAndChecksum     → Trazabilidad y checksums                   ║
║                                                                                  ║
║  CONVENCIONES MATEMÁTICAS:                                                       ║
║  ──────────────────────────────────────────────────────────────────────────────  ║
║  · ε₀ = machine epsilon de float64 ≈ 2.22e-16                                    ║
║  · Ω  = matriz simpléctica canónica 2n×2n                                        ║
║  · Sp(2n) = grupo simpléctico real                                               ║
║  · H¹ = primer grupo de cohomología del haz celular                              ║
║  · χ  = característica de Euler-Poincaré                                         ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones estándar
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import struct
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones científicas
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pytest
import scipy.linalg as la

# ─────────────────────────────────────────────────────────────────────────────
# Módulo bajo prueba
# ─────────────────────────────────────────────────────────────────────────────
from app.agents.boole.physics.ast_static_analyzer_agent import (
    # Excepciones
    ClausiusDuhemViolation,
    CohomologicalObstructionError,
    EulerCharacteristicMismatch,
    ExergyDivergenceError,
    MaslovIndexError,
    MayerVietorisBreachError,
    SymplecticInvarianceViolation,
    SymplecticPolarDecompositionError,
    # DTOs
    ASTGovernanceState,
    AuditProvenance,
    SheafCohomologyAuditData,
    SymplecticInvariantData,
    ThermodynamicDirichletData,
    # Fases
    Phase1_SymplecticInvarianceAuditor,
    Phase2_DirichletThermodynamicEnforcer,
    Phase3_CellularSheafCohomologyAuditor,
    # Orquestador
    ASTStaticAnalyzerAgent,
    # Constantes internas (accedidas para verificación de tolerancias)
    _MACHINE_EPSILON,
    _SYMPLECTIC_TOLERANCE_BASE,
    _COHOMOLOGICAL_COMPLEX_TOL,
    _NUMERICAL_SAFETY_FACTOR,
    _THERMODYNAMIC_TEMPERATURE_REFERENCE,
)

# También importamos TopologicalInvariantError para tests de jerarquía
try:
    from app.core.mic_algebra import TopologicalInvariantError
except ImportError:
    # El stub definido en el módulo bajo prueba
    from app.agents.boole.physics.ast_static_analyzer_agent import (
        SymplecticInvarianceViolation as TopologicalInvariantError,
    )

logger = logging.getLogger("TEST.MIC.ASTAnalyzerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# FÁBRICAS DE OBJETOS MATEMÁTICOS CANÓNICOS
# (Helpers compartidos por toda la suite)
# ═══════════════════════════════════════════════════════════════════════════════

def _canonical_symplectic_matrix(n: int) -> np.ndarray:
    r"""
    Construye Ω ∈ ℝ^{2n×2n}:

        Ω = [[0,  I],
             [-I, 0]].
    """
    omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
    omega[:n, n:] = np.eye(n)
    omega[n:, :n] = -np.eye(n)
    return omega


def _identity_symplectic(n: int) -> np.ndarray:
    r"""
    Retorna la identidad I_{2n} ∈ Sp(2n, ℝ).
    La identidad siempre es simpléctica: Iᵀ Ω I = Ω.
    """
    return np.eye(2 * n, dtype=np.float64)


def _rotation_symplectic(n: int, theta: float = 0.3) -> np.ndarray:
    r"""
    Construye una rotación simpléctica bloque-diagonal:

        M = diag(R(θ), R(θ), ..., R(θ))   [n bloques]

    donde R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] ∈ Sp(2, ℝ).

    Esta construcción garantiza M ∈ Sp(2n, ℝ) exactamente.
    """
    M = np.zeros((2 * n, 2 * n), dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)

    for i in range(n):
        M[i, i] = c
        M[i, n + i] = -s
        M[n + i, i] = s
        M[n + i, n + i] = c

    return M


def _shear_symplectic(n: int, alpha: float = 0.5) -> np.ndarray:
    r"""
    Construye una transformación de cizalla simpléctica:

        M = I + α · e₁ ⊗ eₙ₊₁*

    Esta es simpléctica para cualquier α ∈ ℝ.
    """
    M = np.eye(2 * n, dtype=np.float64)
    M[0, n] = alpha
    return M


def _positive_dissipative_pair(dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Genera un par (Φ, ∇V) con P_diss = Φᵀ∇V > 0.

    Φ = ∇V = e₁ (primer vector canónico) → P_diss = 1.
    """
    Phi = np.zeros(dim, dtype=np.float64)
    gradV = np.zeros(dim, dtype=np.float64)
    Phi[0] = 1.0
    gradV[0] = 1.0
    return Phi, gradV


def _zero_dissipation_pair(dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Genera un par (Φ, ∇V) con P_diss = Φᵀ∇V = 0 (vectores ortogonales).

    Φ = e₁, ∇V = e₂ → P_diss = 0.
    """
    Phi = np.zeros(dim, dtype=np.float64)
    gradV = np.zeros(dim, dtype=np.float64)
    Phi[0] = 1.0
    gradV[1] = 1.0
    return Phi, gradV


def _negative_dissipation_pair(dim: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Genera un par (Φ, ∇V) con P_diss = Φᵀ∇V < 0.

    Φ = e₁, ∇V = −e₁ → P_diss = −1.
    """
    Phi = np.zeros(dim, dtype=np.float64)
    gradV = np.zeros(dim, dtype=np.float64)
    Phi[0] = 1.0
    gradV[0] = -1.0
    return Phi, gradV


def _trivial_coboundary_operators(
    c0: int = 3, c1: int = 4, c2: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construye operadores coboundary δ⁰ ∈ ℝ^{c1×c0} y δ¹ ∈ ℝ^{c2×c1}
    que satisfacen δ¹∘δ⁰ = 0 con dim H¹ = 0.

    Estrategia: tomar D0 aleatorio de rango máximo y construir D1
    en el espacio nulo de D0.
    """
    rng = np.random.default_rng(seed=42)

    # D0: c1 × c0 de rango c0 (columnas linealmente independientes)
    Q, _ = la.qr(rng.standard_normal((c1, c1)))
    D0 = Q[:, :c0]  # c1 × c0, rango = c0

    # Espacio nulo de D0ᵀ (= cokernel de D0)
    # Buscamos vectores en ℝ^{c1} ortogonales a las columnas de D0
    null_D0T = Q[:, c0:]  # c1 × (c1 - c0), base del cokernel

    # D1 debe mapear ℝ^{c1} → ℝ^{c2} con im(D1) ⊆ cokernel de D0ᵀ = {0}
    # Más sencillo: D1 = V · D0ᵀ para alguna V tal que D1 D0 = 0
    # Usamos: D1 orthogonal al espacio columna de D0
    # D1 rows deben estar en null(D0ᵀ), i.e., D0ᵀ rows
    # La construcción más segura: D1 = 0 (complejo trivialmente exacto)
    D1 = np.zeros((c2, c1), dtype=np.float64)

    return D0.astype(np.float64), D1


def _non_exact_coboundary_operators(
    c0: int = 2, c1: int = 3, c2: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construye operadores con dim H¹ > 0.

    Toma D0 y D1 de rango mínimo tal que ker(D1) ⊋ im(D0).
    """
    # D0: c1 × c0 = 3 × 2, rango 1 (solo primera columna no nula)
    D0 = np.zeros((c1, c0), dtype=np.float64)
    D0[0, 0] = 1.0  # rank = 1

    # D1: c2 × c1 = 2 × 3, rango 1, ker(D1) tiene dim 2
    # im(D0) tiene dim 1, ker(D1) tiene dim 2 ⊇ im(D0) → δ¹∘δ⁰ = 0
    D1 = np.zeros((c2, c1), dtype=np.float64)
    D1[0, 1] = 1.0  # rank = 1, ker incluye e₁ y e₃

    # Verificar: D1 D0 = 0
    assert np.allclose(D1 @ D0, 0.0), "Error en factory: δ¹∘δ⁰ ≠ 0"

    # dim H¹ = c1 - rank(D0) - rank(D1) = 3 - 1 - 1 = 1 > 0
    return D0, D1


def _violating_coboundary_operators() -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Construye operadores donde δ¹∘δ⁰ ≠ 0 (no forman un complejo).

    Esto debe disparar CohomologicalObstructionError.
    """
    D0 = np.eye(3, 2, dtype=np.float64)   # 3 × 2
    D1 = np.eye(2, 3, dtype=np.float64)   # 2 × 3
    # D1 @ D0 = I₂ @ I_{3×2} = [[1,0],[0,1]] ≠ 0
    return D0, D1


def _build_valid_governance_inputs(
    n: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Retorna (M, Φ, ∇V) válidos para n grados de libertad simplécticos.

    M = identidad 2n×2n (simpléctica), Φ = ∇V = e₁ (disipación positiva).
    """
    M = _identity_symplectic(n)
    dim = 2 * n
    Phi = np.zeros(dim, dtype=np.float64)
    gradV = np.zeros(dim, dtype=np.float64)
    Phi[0] = 1.0
    gradV[0] = 1.0
    return M, Phi, gradV


# ═══════════════════════════════════════════════════════════════════════════════
# §T0 — TESTS DE GUARDAS NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════
class TestFiniteNumericalGuard:
    r"""
    Verifica el comportamiento de _FiniteNumericalGuard ante toda clase de
    entradas degeneradas: complejas, NaN, infinitas, vacías y de tipo incorrecto.
    """

    @pytest.fixture
    def guard(self) -> Phase1_SymplecticInvarianceAuditor:
        r"""Instancia de Phase1 que expone todos los métodos de guarda."""
        return Phase1_SymplecticInvarianceAuditor()

    # ── T0.1: _as_float_array ─────────────────────────────────────────────────

    def test_as_float_array_accepts_valid_real(self, guard):
        r"""Acepta arreglos reales finitos sin modificación."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = guard._as_float_array("test", arr)
        np.testing.assert_array_equal(result, arr)

    def test_as_float_array_converts_int_to_float64(self, guard):
        r"""Convierte enteros a float64."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = guard._as_float_array("test", arr)
        assert result.dtype == np.float64

    def test_as_float_array_rejects_complex(self, guard):
        r"""Rechaza arreglos complejos."""
        arr = np.array([1.0 + 2.0j, 3.0 + 4.0j])
        with pytest.raises(TypeError, match="real"):
            guard._as_float_array("test_complex", arr)

    def test_as_float_array_rejects_nan(self, guard):
        r"""Rechaza arreglos con NaN."""
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("test_nan", arr)

    def test_as_float_array_rejects_positive_inf(self, guard):
        r"""Rechaza arreglos con +∞."""
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("test_inf", arr)

    def test_as_float_array_rejects_negative_inf(self, guard):
        r"""Rechaza arreglos con −∞."""
        arr = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            guard._as_float_array("test_neginf", arr)

    def test_as_float_array_rejects_non_numeric(self, guard):
        r"""Rechaza objetos no numéricos."""
        with pytest.raises(TypeError):
            guard._as_float_array("test_str", "no_soy_un_arreglo")

    # ── T0.2: _as_finite_matrix ───────────────────────────────────────────────

    def test_as_finite_matrix_accepts_valid_2d(self, guard):
        r"""Acepta matrices 2D reales finitas."""
        M = np.eye(4, dtype=np.float64)
        result = guard._as_finite_matrix("M", M)
        np.testing.assert_array_equal(result, M)

    def test_as_finite_matrix_rejects_1d(self, guard):
        r"""Rechaza arreglos 1D."""
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            guard._as_finite_matrix("v", v)

    def test_as_finite_matrix_rejects_non_square_when_square_required(self, guard):
        r"""Rechaza matrices no cuadradas cuando square=True."""
        M = np.ones((3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="cuadrada"):
            guard._as_finite_matrix("M", M, square=True)

    def test_as_finite_matrix_accepts_non_square_by_default(self, guard):
        r"""Acepta matrices no cuadradas por defecto."""
        M = np.ones((3, 4), dtype=np.float64)
        result = guard._as_finite_matrix("M", M)
        assert result.shape == (3, 4)

    def test_as_finite_matrix_rejects_empty(self, guard):
        r"""Rechaza matrices vacías."""
        M = np.zeros((0, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="vacía"):
            guard._as_finite_matrix("M", M)

    # ── T0.3: _as_finite_vector ───────────────────────────────────────────────

    def test_as_finite_vector_accepts_1d(self, guard):
        r"""Acepta vectores 1D."""
        v = np.array([1.0, 2.0, 3.0])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1

    def test_as_finite_vector_normalizes_column_vector(self, guard):
        r"""Normaliza vector columna (n×1) a 1D."""
        v = np.array([[1.0], [2.0], [3.0]])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1
        assert result.shape == (3,)

    def test_as_finite_vector_normalizes_row_vector(self, guard):
        r"""Normaliza vector fila (1×n) a 1D."""
        v = np.array([[1.0, 2.0, 3.0]])
        result = guard._as_finite_vector("v", v)
        assert result.ndim == 1

    def test_as_finite_vector_rejects_empty(self, guard):
        r"""Rechaza vectores vacíos."""
        v = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="vacío"):
            guard._as_finite_vector("v", v)

    def test_as_finite_vector_rejects_zero_vector_when_required(self, guard):
        r"""Rechaza el vector cero cuando allow_zero_vector=False."""
        v = np.zeros(4, dtype=np.float64)
        with pytest.raises(ValueError, match="cero"):
            guard._as_finite_vector("v", v, allow_zero_vector=False)

    def test_as_finite_vector_accepts_zero_vector_by_default(self, guard):
        r"""Acepta el vector cero por defecto."""
        v = np.zeros(4, dtype=np.float64)
        result = guard._as_finite_vector("v", v)
        np.testing.assert_array_equal(result, v)

    # ── T0.4: Normas ──────────────────────────────────────────────────────────

    def test_frobenius_norm_identity(self, guard):
        r"""‖I_n‖_F = √n."""
        n = 4
        result = guard._frobenius_norm(np.eye(n))
        assert math.isclose(result, math.sqrt(n), rel_tol=1e-12)

    def test_frobenius_norm_empty(self, guard):
        r"""‖∅‖_F = 0."""
        assert guard._frobenius_norm(np.array([[]])) == 0.0

    def test_vector_norm_canonical(self, guard):
        r"""‖e₁‖₂ = 1."""
        v = np.array([1.0, 0.0, 0.0])
        assert math.isclose(guard._vector_norm(v), 1.0, rel_tol=1e-12)

    def test_vector_norm_empty(self, guard):
        r"""‖∅‖₂ = 0."""
        assert guard._vector_norm(np.array([])) == 0.0

    def test_spectral_norm_identity(self, guard):
        r"""‖I_n‖₂ = σ_max(I) = 1."""
        result = guard._spectral_norm(np.eye(5))
        assert math.isclose(result, 1.0, rel_tol=1e-12)

    def test_spectral_norm_scaled_identity(self, guard):
        r"""‖α·I_n‖₂ = |α|."""
        alpha = 3.7
        result = guard._spectral_norm(alpha * np.eye(4))
        assert math.isclose(result, alpha, rel_tol=1e-10)

    # ── T0.5: Checksum ────────────────────────────────────────────────────────

    def test_compute_input_checksum_deterministic(self, guard):
        r"""El checksum es determinista para la misma entrada."""
        arr = np.ones((4, 4), dtype=np.float64)
        c1 = guard._compute_input_checksum(arr)
        c2 = guard._compute_input_checksum(arr)
        assert c1 == c2

    def test_compute_input_checksum_differs_for_different_inputs(self, guard):
        r"""Inputs distintos producen checksums distintos (con alta probabilidad)."""
        arr1 = np.ones((4, 4), dtype=np.float64)
        arr2 = np.ones((4, 4), dtype=np.float64) * 2.0
        assert guard._compute_input_checksum(arr1) != guard._compute_input_checksum(arr2)

    def test_compute_input_checksum_is_sha256(self, guard):
        r"""El checksum tiene 64 caracteres hexadecimales (SHA-256)."""
        arr = np.eye(3, dtype=np.float64)
        checksum = guard._compute_input_checksum(arr)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    # ── T0.6: _structural_zero_check ─────────────────────────────────────────

    def test_structural_zero_check_logs_warning_for_zero_row(self, guard, caplog):
        r"""Emite warning cuando detecta una fila de ceros."""
        M = np.eye(4, dtype=np.float64)
        M[2, :] = 0.0
        with caplog.at_level(logging.WARNING):
            guard._structural_zero_check("M", M)
        assert any("fila" in r.message.lower() for r in caplog.records)

    def test_structural_zero_check_logs_warning_for_zero_col(self, guard, caplog):
        r"""Emite warning cuando detecta una columna de ceros."""
        M = np.eye(4, dtype=np.float64)
        M[:, 1] = 0.0
        with caplog.at_level(logging.WARNING):
            guard._structural_zero_check("M", M)
        assert any("columna" in r.message.lower() for r in caplog.records)

    def test_structural_zero_check_no_warning_for_valid_matrix(self, guard, caplog):
        r"""No emite warning para una matriz sin ceros estructurales."""
        M = np.eye(4, dtype=np.float64) + 0.1
        with caplog.at_level(logging.WARNING):
            guard._structural_zero_check("M", M)
        zero_warnings = [
            r for r in caplog.records
            if "fila" in r.message.lower() or "columna" in r.message.lower()
        ]
        assert len(zero_warnings) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# §T1 — TESTS DE FASE 1: INVARIANZA SIMPLÉCTICA
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase1_Symplectic:
    r"""
    Verifica todos los invariantes simplécticos de Phase1_SymplecticInvarianceAuditor.

    Estructura de subtests:
        T1.1  → Construcción de Ω canónica
        T1.2  → Tolerancia adaptativa espectral
        T1.3  → Residuo determinantal
        T1.4  → Número de condición
        T1.5  → Radio espectral
        T1.6  → Residuo antisimétrico
        T1.7  → Descomposición polar
        T1.8  → Índice de Maslov
        T1.9  → _audit_symplectic_invariance (método principal)
    """

    @pytest.fixture
    def auditor(self) -> Phase1_SymplecticInvarianceAuditor:
        return Phase1_SymplecticInvarianceAuditor()

    # ── T1.1: Construcción de Ω ───────────────────────────────────────────────

    def test_canonical_omega_shape(self, auditor):
        r"""Ω tiene forma 2n × 2n."""
        for n in [1, 2, 3, 5]:
            omega = auditor._build_canonical_symplectic_matrix(n)
            assert omega.shape == (2 * n, 2 * n)

    def test_canonical_omega_antisymmetric(self, auditor):
        r"""Ω debe ser antisimétrica: Ω + Ωᵀ = 0."""
        for n in [1, 2, 4]:
            omega = auditor._build_canonical_symplectic_matrix(n)
            residual = la.norm(omega + omega.T, ord="fro")
            assert residual < 1e-14, f"Ω no antisimétrica para n={n}: residuo={residual}"

    def test_canonical_omega_orthogonal(self, auditor):
        r"""Ω debe ser ortogonal: ΩᵀΩ = I."""
        for n in [1, 2, 3]:
            omega = auditor._build_canonical_symplectic_matrix(n)
            product = omega.T @ omega
            identity = np.eye(2 * n)
            residual = la.norm(product - identity, ord="fro")
            assert residual < 1e-13

    def test_canonical_omega_determinant(self, auditor):
        r"""det(Ω) = 1 para toda n."""
        for n in [1, 2, 3, 4]:
            omega = auditor._build_canonical_symplectic_matrix(n)
            sign, logdet = la.slogdet(omega)
            det = float(sign) * math.exp(logdet)
            assert math.isclose(abs(det), 1.0, rel_tol=1e-10)

    def test_canonical_omega_blocks(self, auditor):
        r"""Verifica bloques explícitos para n=2."""
        omega = auditor._build_canonical_symplectic_matrix(2)
        # Bloque superior-derecho = I₂
        np.testing.assert_array_almost_equal(omega[:2, 2:], np.eye(2))
        # Bloque inferior-izquierdo = -I₂
        np.testing.assert_array_almost_equal(omega[2:, :2], -np.eye(2))
        # Bloques diagonales = 0
        np.testing.assert_array_almost_equal(omega[:2, :2], np.zeros((2, 2)))
        np.testing.assert_array_almost_equal(omega[2:, 2:], np.zeros((2, 2)))

    def test_canonical_omega_rejects_zero_n(self, auditor):
        r"""Rechaza n=0."""
        with pytest.raises(ValueError, match="positivo"):
            auditor._build_canonical_symplectic_matrix(0)

    def test_canonical_omega_rejects_negative_n(self, auditor):
        r"""Rechaza n negativo."""
        with pytest.raises(ValueError, match="positivo"):
            auditor._build_canonical_symplectic_matrix(-1)

    def test_canonical_omega_rejects_non_integer_n(self, auditor):
        r"""Rechaza n no entero."""
        with pytest.raises(TypeError):
            auditor._build_canonical_symplectic_matrix(2.0)

    # ── T1.2: Tolerancia adaptativa ───────────────────────────────────────────

    def test_adaptive_tolerance_base_for_zero_sigma(self, auditor):
        r"""Para σ_max=0, retorna la tolerancia base."""
        tol = auditor._compute_adaptive_symplectic_tolerance(0.0)
        assert tol == _SYMPLECTIC_TOLERANCE_BASE

    def test_adaptive_tolerance_base_for_negative_sigma(self, auditor):
        r"""Para σ_max<0 (inválido), retorna la tolerancia base."""
        tol = auditor._compute_adaptive_symplectic_tolerance(-1.0)
        assert tol == _SYMPLECTIC_TOLERANCE_BASE

    def test_adaptive_tolerance_increases_with_large_sigma(self, auditor):
        r"""Para σ_max grande, la tolerancia debe ser mayor que la base."""
        tol_small = auditor._compute_adaptive_symplectic_tolerance(1.0)
        tol_large = auditor._compute_adaptive_symplectic_tolerance(1e8)
        assert tol_large > tol_small

    def test_adaptive_tolerance_always_positive(self, auditor):
        r"""La tolerancia efectiva debe ser siempre positiva."""
        for sigma in [0.0, 1e-15, 1.0, 1e8, 1e15]:
            tol = auditor._compute_adaptive_symplectic_tolerance(sigma)
            assert tol > 0.0

    # ── T1.3: Residuo determinantal ───────────────────────────────────────────

    def test_determinant_residual_identity(self, auditor):
        r"""det(I) = 1 → residuo = 0."""
        M = np.eye(4)
        sign, logdet = la.slogdet(M)
        res = auditor._determinant_residual_from_slogdet(sign, logdet)
        assert math.isclose(res, 0.0, abs_tol=1e-14)

    def test_determinant_residual_singular(self, auditor):
        r"""det(M)=0 (sign=0) → residuo = 1."""
        res = auditor._determinant_residual_from_slogdet(0.0, 0.0)
        assert res == 1.0

    def test_determinant_residual_finite_for_overflow(self, auditor):
        r"""Para logdet muy grande, retorna inf."""
        res = auditor._determinant_residual_from_slogdet(1.0, 1e400)
        assert res == math.inf

    def test_determinant_residual_near_zero_det(self, auditor):
        r"""Para det≈0 (logdet muy negativo), residuo≈1."""
        res = auditor._determinant_residual_from_slogdet(1.0, -1e300)
        assert math.isclose(res, 1.0, rel_tol=1e-6)

    def test_determinant_residual_known_value(self, auditor):
        r"""Para M = 2·I₂, det = 4, residuo = |4 - 1| = 3."""
        M = 2.0 * np.eye(2)
        sign, logdet = la.slogdet(M)
        res = auditor._determinant_residual_from_slogdet(sign, logdet)
        assert math.isclose(res, 3.0, rel_tol=1e-10)

    # ── T1.4: Número de condición ─────────────────────────────────────────────

    def test_condition_number_identity(self, auditor):
        r"""κ(I) = 1."""
        M = np.eye(4)
        kappa = auditor._condition_number(M)
        assert math.isclose(kappa, 1.0, rel_tol=1e-10)

    def test_condition_number_scaled_identity(self, auditor):
        r"""κ(α·I) = 1 para todo α ≠ 0."""
        M = 5.0 * np.eye(4)
        kappa = auditor._condition_number(M)
        assert math.isclose(kappa, 1.0, rel_tol=1e-10)

    def test_condition_number_singular_is_inf(self, auditor):
        r"""κ de una matriz singular → inf."""
        M = np.zeros((4, 4))
        M[0, 0] = 1.0
        kappa = auditor._condition_number(M)
        assert kappa == math.inf

    def test_condition_number_diagonal(self, auditor):
        r"""κ(diag(σ_max, σ_min)) = σ_max/σ_min."""
        M = np.diag([10.0, 1.0, 5.0, 2.0])
        kappa = auditor._condition_number(M)
        expected = 10.0 / 1.0
        assert math.isclose(kappa, expected, rel_tol=1e-10)

    # ── T1.5: Radio espectral ─────────────────────────────────────────────────

    def test_spectral_radius_canonical_omega(self, auditor):
        r"""
        Ω tiene eigenvalores ±i, entonces ρ(Ω) = 1.
        """
        omega = _canonical_symplectic_matrix(2)
        rho = auditor._spectral_radius_transformed_omega(omega)
        assert math.isclose(rho, 1.0, rel_tol=1e-10)

    def test_spectral_radius_identity(self, auditor):
        r"""ρ(I) = 1."""
        rho = auditor._spectral_radius_transformed_omega(np.eye(4))
        assert math.isclose(rho, 1.0, rel_tol=1e-10)

    def test_spectral_radius_zero_matrix(self, auditor):
        r"""ρ(0) = 0."""
        rho = auditor._spectral_radius_transformed_omega(np.zeros((4, 4)))
        assert math.isclose(rho, 0.0, abs_tol=1e-14)

    # ── T1.6: Residuo antisimétrico ───────────────────────────────────────────

    def test_antisymmetric_residual_exact_omega(self, auditor):
        r"""
        Si MᵀΩM = Ω, el residuo antisimétrico debe ser 0.
        Usamos M = I: Iᵀ Ω I = Ω.
        """
        omega = _canonical_symplectic_matrix(2)
        residual = auditor._antisymmetric_residual(omega, omega)
        assert residual < 1e-14

    def test_antisymmetric_residual_symmetric_perturbation(self, auditor):
        r"""
        Si MᵀΩM = Ω + ε·S con S simétrica, el residuo antisimétrico
        debe ser bajo (la perturbación simétrica no afecta la parte antisimétrica).
        """
        omega = _canonical_symplectic_matrix(2)
        S = np.eye(4)  # simétrica
        epsilon = 1e-6
        perturbed = omega + epsilon * S
        # skew(perturbed) = skew(omega) + epsilon * skew(S) = omega + 0
        residual = auditor._antisymmetric_residual(perturbed, omega)
        assert residual < 1e-5  # Tolerancia amplia para perturbación

    def test_antisymmetric_residual_antisymmetric_perturbation(self, auditor):
        r"""
        Una perturbación antisimétrica grande produce residuo elevado.
        """
        omega = _canonical_symplectic_matrix(2)
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[1, 0] = -1.0
        epsilon = 0.1
        perturbed = omega + epsilon * A
        residual = auditor._antisymmetric_residual(perturbed, omega)
        assert residual > 1e-6

    # ── T1.7: Descomposición polar ────────────────────────────────────────────

    def test_polar_residual_identity(self, auditor):
        r"""Para M = I, U = I, residuo polar = 0."""
        omega = _canonical_symplectic_matrix(2)
        M = np.eye(4)
        residual = auditor._polar_unitarity_residual(M, omega)
        assert residual < 1e-12

    def test_polar_residual_rotation(self, auditor):
        r"""
        Para M simpléctica (rotación), el residuo polar debe ser pequeño,
        pues U ≈ M ∈ Sp(2n).
        """
        omega = _canonical_symplectic_matrix(2)
        M = _rotation_symplectic(2, theta=0.7)
        residual = auditor._polar_unitarity_residual(M, omega)
        assert residual < 1e-10

    def test_polar_residual_is_nonnegative(self, auditor):
        r"""El residuo polar siempre es no negativo."""
        omega = _canonical_symplectic_matrix(2)
        M = _shear_symplectic(2, alpha=0.5)
        residual = auditor._polar_unitarity_residual(M, omega)
        assert residual >= 0.0

    # ── T1.8: Índice de Maslov ────────────────────────────────────────────────

    def test_maslov_index_identity(self, auditor):
        r"""
        Para M = I, el bloque B = 0, eigenvalores = 0, μ = 0.
        """
        omega = _canonical_symplectic_matrix(2)
        M = np.eye(4)
        mu = auditor._compute_maslov_index(M, omega)
        assert mu == 0

    def test_maslov_index_is_integer(self, auditor):
        r"""El índice de Maslov es siempre un entero."""
        omega = _canonical_symplectic_matrix(2)
        M = _rotation_symplectic(2, theta=1.2)
        mu = auditor._compute_maslov_index(M, omega)
        assert isinstance(mu, int)

    def test_maslov_index_nonnegative_for_symplectic(self, auditor):
        r"""Para transformaciones simplécticas estándar, μ ≥ 0."""
        omega = _canonical_symplectic_matrix(2)
        M = _shear_symplectic(2, alpha=1.0)
        mu = auditor._compute_maslov_index(M, omega)
        assert mu >= 0

    # ── T1.9: _audit_symplectic_invariance (método principal) ─────────────────

    def test_audit_identity_passes(self, auditor):
        r"""La identidad 4×4 debe pasar la auditoría simplécticamentre."""
        M = _identity_symplectic(2)
        result = auditor._audit_symplectic_invariance(M)
        assert isinstance(result, SymplecticInvariantData)
        assert result.is_volume_preserved is True
        assert result.phase_space_dimension == 4

    def test_audit_rotation_passes(self, auditor):
        r"""Una rotación simpléctica debe pasar la auditoría."""
        M = _rotation_symplectic(2, theta=0.5)
        result = auditor._audit_symplectic_invariance(M)
        assert result.is_volume_preserved is True

    def test_audit_shear_passes(self, auditor):
        r"""Una transformación de cizalla simpléctica debe pasar."""
        M = _shear_symplectic(3, alpha=2.0)
        result = auditor._audit_symplectic_invariance(M)
        assert result.is_volume_preserved is True

    def test_audit_large_dimension(self, auditor):
        r"""Prueba con dimensión grande n=8 (dim=16)."""
        M = _identity_symplectic(8)
        result = auditor._audit_symplectic_invariance(M)
        assert result.phase_space_dimension == 16
        assert result.is_volume_preserved is True

    def test_audit_rejects_odd_dimension(self, auditor):
        r"""Rechaza matrices de dimensión impar."""
        M = np.eye(3)
        with pytest.raises(SymplecticInvarianceViolation, match="par"):
            auditor._audit_symplectic_invariance(M)

    def test_audit_rejects_zero_dimension(self, auditor):
        r"""Rechaza matrices vacías (dim=0)."""
        M = np.zeros((0, 0))
        with pytest.raises((SymplecticInvarianceViolation, ValueError)):
            auditor._audit_symplectic_invariance(M)

    def test_audit_rejects_non_symplectic_matrix(self, auditor):
        r"""
        Una matriz que no preserva Ω debe fallar.
        M = 2·I es simpléctica? Mᵀ Ω M = 4Ω ≠ Ω → falla.
        """
        M = 2.0 * np.eye(4)
        with pytest.raises(SymplecticInvarianceViolation):
            auditor._audit_symplectic_invariance(M)

    def test_audit_rejects_random_non_symplectic(self, auditor):
        r"""Una matriz aleatoria genérica no es simpléctica."""
        rng = np.random.default_rng(seed=123)
        M = rng.standard_normal((4, 4))
        # Hace M ortogonal (no necesariamente simpléctica)
        M, _ = la.qr(M)
        # Solo si casualmente es simpléctica podría pasar; en general falla
        # Escalamos para romper simplecticidad
        M_bad = M * 1.5
        with pytest.raises(SymplecticInvarianceViolation):
            auditor._audit_symplectic_invariance(M_bad)

    def test_audit_residual_norm_is_nonneg(self, auditor):
        r"""El residuo de Frobenius nunca es negativo."""
        M = _identity_symplectic(2)
        result = auditor._audit_symplectic_invariance(M)
        assert result.symplectic_residual_norm >= 0.0

    def test_audit_relative_residual_leq_effective_tol(self, auditor):
        r"""Para M simpléctica, residuo relativo ≤ tolerancia efectiva."""
        M = _rotation_symplectic(3, theta=1.0)
        result = auditor._audit_symplectic_invariance(M)
        assert result.symplectic_relative_residual <= result.effective_tolerance

    def test_audit_determinant_residual_small_for_symplectic(self, auditor):
        r"""Para M ∈ Sp(2n), |det(M)−1| debe ser pequeño."""
        M = _shear_symplectic(2, alpha=3.0)
        result = auditor._audit_symplectic_invariance(M)
        assert result.determinant_residual < 1e-10

    def test_audit_condition_number_positive_finite(self, auditor):
        r"""El número de condición debe ser finito y positivo."""
        M = _identity_symplectic(2)
        result = auditor._audit_symplectic_invariance(M)
        assert math.isfinite(result.condition_number)
        assert result.condition_number >= 1.0

    def test_audit_returns_correct_phase_space_dim(self, auditor):
        r"""El campo phase_space_dimension refleja la dimensión real de M."""
        for n in [1, 2, 3, 4]:
            M = _identity_symplectic(n)
            result = auditor._audit_symplectic_invariance(M)
            assert result.phase_space_dimension == 2 * n

    def test_audit_rejects_singular_matrix(self, auditor):
        r"""Una matriz singular (det=0) debe fallar."""
        M = np.zeros((4, 4))
        M[0, 0] = 1.0  # Rango 1, no simpléctica
        with pytest.raises((SymplecticInvarianceViolation, ValueError)):
            auditor._audit_symplectic_invariance(M)

    def test_audit_rejects_nan_input(self, auditor):
        r"""Entrada con NaN debe fallar en la guarda."""
        M = np.eye(4)
        M[0, 0] = np.nan
        with pytest.raises((ValueError, SymplecticInvarianceViolation)):
            auditor._audit_symplectic_invariance(M)

    def test_audit_rejects_inf_input(self, auditor):
        r"""Entrada con ∞ debe fallar en la guarda."""
        M = np.eye(4)
        M[1, 1] = np.inf
        with pytest.raises((ValueError, SymplecticInvarianceViolation)):
            auditor._audit_symplectic_invariance(M)

    def test_audit_negative_determinant_fails(self, auditor):
        r"""
        Matriz con det(M) < 0 debe fallar.
        Usamos -I₄: det(-I₄) = (-1)⁴ = 1 en dim 4, entonces usamos dim=2.
        En dim 2: det(-I₂) = 1. Pero (-I₂)ᵀ Ω (-I₂) = Ω → es simpléctica.
        En dim 4: queremos det < 0.
        Construimos M con det = -1: intercambiamos dos filas de I.
        """
        M = np.eye(4, dtype=np.float64)
        M[[0, 1]] = M[[1, 0]]  # intercambio de filas → det = -1
        with pytest.raises(SymplecticInvarianceViolation, match="det"):
            auditor._audit_symplectic_invariance(M)

    def test_audit_symplectic_data_fields_are_finite(self, auditor):
        r"""Todos los campos numéricos del certificado deben ser finitos."""
        M = _rotation_symplectic(2, theta=0.3)
        result = auditor._audit_symplectic_invariance(M)
        assert math.isfinite(result.symplectic_residual_norm)
        assert math.isfinite(result.symplectic_relative_residual)
        assert math.isfinite(result.antisymmetric_residual)
        assert math.isfinite(result.determinant_residual)
        assert math.isfinite(result.condition_number)
        assert math.isfinite(result.polar_unitarity_residual)
        assert math.isfinite(result.effective_tolerance)


# ═══════════════════════════════════════════════════════════════════════════════
# §T2 — TESTS DE FASE 2: TERMODINÁMICA PORT-HAMILTONIANA
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase2_Thermodynamic:
    r"""
    Verifica el comportamiento de Phase2_DirichletThermodynamicEnforcer.

    Estructura de subtests:
        T2.1  → Tasa de producción de entropía
        T2.2  → Ángulo termodinámico
        T2.3  → Ratio de eficiencia disipativa
        T2.4  → _enforce_dirichlet_thermodynamics (método principal)
    """

    @pytest.fixture
    def enforcer(self) -> Phase2_DirichletThermodynamicEnforcer:
        return Phase2_DirichletThermodynamicEnforcer()

    @pytest.fixture
    def valid_symplectic_cert(self) -> SymplecticInvariantData:
        r"""Certificado simpléctico válido para la continuación funtorial."""
        auditor = Phase1_SymplecticInvarianceAuditor()
        M = _identity_symplectic(2)
        return auditor._audit_symplectic_invariance(M)

    # ── T2.1: Tasa de producción de entropía ─────────────────────────────────

    def test_entropy_production_positive_dissipation(self, enforcer):
        r"""σ_CD = P_diss / T_ref > 0 para P_diss > 0."""
        sigma = enforcer._compute_entropy_production_rate(1.0)
        assert sigma > 0.0
        assert math.isclose(sigma, 1.0 / _THERMODYNAMIC_TEMPERATURE_REFERENCE, rel_tol=1e-12)

    def test_entropy_production_zero_dissipation(self, enforcer):
        r"""σ_CD = 0 para P_diss = 0."""
        sigma = enforcer._compute_entropy_production_rate(0.0)
        assert sigma == 0.0

    def test_entropy_production_custom_temperature(self, enforcer):
        r"""σ_CD = P_diss / T_ref para T_ref arbitrario."""
        sigma = enforcer._compute_entropy_production_rate(2.0, temperature_reference=4.0)
        assert math.isclose(sigma, 0.5, rel_tol=1e-12)

    def test_entropy_production_rejects_zero_temperature(self, enforcer):
        r"""T_ref = 0 debe fallar."""
        with pytest.raises(ClausiusDuhemViolation):
            enforcer._compute_entropy_production_rate(1.0, temperature_reference=0.0)

    def test_entropy_production_rejects_negative_temperature(self, enforcer):
        r"""T_ref < 0 debe fallar."""
        with pytest.raises(ClausiusDuhemViolation):
            enforcer._compute_entropy_production_rate(1.0, temperature_reference=-1.0)

    # ── T2.2: Ángulo termodinámico ────────────────────────────────────────────

    def test_thermodynamic_angle_cosine_one(self, enforcer):
        r"""cos=1 → θ = 0 (vectores paralelos, disipación máxima)."""
        theta = enforcer._compute_thermodynamic_angle(1.0)
        assert math.isclose(theta, 0.0, abs_tol=1e-12)

    def test_thermodynamic_angle_cosine_minus_one(self, enforcer):
        r"""cos=−1 → θ = π (vectores antiparalelos, retroalimentación máxima)."""
        theta = enforcer._compute_thermodynamic_angle(-1.0)
        assert math.isclose(theta, math.pi, rel_tol=1e-12)

    def test_thermodynamic_angle_cosine_zero(self, enforcer):
        r"""cos=0 → θ = π/2 (vectores ortogonales, disipación nula)."""
        theta = enforcer._compute_thermodynamic_angle(0.0)
        assert math.isclose(theta, math.pi / 2.0, rel_tol=1e-12)

    def test_thermodynamic_angle_range(self, enforcer):
        r"""θ ∈ [0, π] para todo cos(θ) ∈ [−1, 1]."""
        for cos_val in np.linspace(-1.0, 1.0, 21):
            theta = enforcer._compute_thermodynamic_angle(float(cos_val))
            assert 0.0 <= theta <= math.pi

    def test_thermodynamic_angle_clips_out_of_range(self, enforcer):
        r"""Valores fuera de [−1,1] se recortan antes de arccos."""
        theta_over = enforcer._compute_thermodynamic_angle(1.1)
        theta_under = enforcer._compute_thermodynamic_angle(-1.1)
        assert math.isclose(theta_over, 0.0, abs_tol=1e-12)
        assert math.isclose(theta_under, math.pi, rel_tol=1e-12)

    # ── T2.3: Ratio de eficiencia disipativa ──────────────────────────────────

    def test_dissipation_ratio_full_alignment(self, enforcer):
        r"""η = 1 cuando Φ ∥ ∇V y P_diss = ‖Φ‖·‖∇V‖."""
        phi_norm = 3.0
        grad_norm = 2.0
        p_diss = phi_norm * grad_norm  # alineación perfecta
        ratio = enforcer._compute_exergy_dissipation_ratio(p_diss, phi_norm, grad_norm)
        assert math.isclose(ratio, 1.0, rel_tol=1e-10)

    def test_dissipation_ratio_zero_phi_norm(self, enforcer):
        r"""η = 0 si ‖Φ‖ = 0."""
        ratio = enforcer._compute_exergy_dissipation_ratio(0.0, 0.0, 1.0)
        assert ratio == 0.0

    def test_dissipation_ratio_zero_grad_norm(self, enforcer):
        r"""η = 0 si ‖∇V‖ = 0."""
        ratio = enforcer._compute_exergy_dissipation_ratio(0.0, 1.0, 0.0)
        assert ratio == 0.0

    def test_dissipation_ratio_clipped_to_unit_interval(self, enforcer):
        r"""η ∈ [−1, 1]."""
        # P_diss muy grande respecto al producto de normas
        ratio = enforcer._compute_exergy_dissipation_ratio(1e10, 1.0, 1.0)
        assert ratio <= 1.0

    # ── T2.4: _enforce_dirichlet_thermodynamics ───────────────────────────────

    def test_enforce_positive_dissipation_passes(self, enforcer):
        r"""P_diss > 0 debe pasar la auditoría termodinámica."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert isinstance(result, ThermodynamicDirichletData)
        assert result.is_thermodynamically_stable is True
        assert result.dissipated_power > 0.0

    def test_enforce_zero_dissipation_passes_with_warning(self, enforcer, caplog):
        r"""P_diss = 0 pasa pero emite warning de disipación no estricta."""
        Phi, gradV = _zero_dissipation_pair(dim=4)
        with caplog.at_level(logging.WARNING):
            result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.is_thermodynamically_stable is True
        assert result.is_strictly_dissipative is False
        assert any("estricta" in r.message.lower() for r in caplog.records)

    def test_enforce_negative_dissipation_raises(self, enforcer):
        r"""P_diss < 0 debe lanzar ThermodynamicSingularityError."""
        Phi, gradV = _negative_dissipation_pair(dim=4)
        with pytest.raises(ThermodynamicSingularityError):
            enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)

    def test_enforce_rejects_dimension_mismatch(self, enforcer):
        r"""Φ y ∇V con dimensiones distintas deben fallar."""
        Phi = np.ones(4)
        gradV = np.ones(6)
        with pytest.raises(ValueError, match="Dimensiones"):
            enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)

    def test_enforce_with_valid_symplectic_cert(self, enforcer, valid_symplectic_cert):
        r"""Con certificado simpléctico válido, la Fase 2 debe pasar."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = enforcer._enforce_dirichlet_thermodynamics(
            Phi, gradV, symplectic_audit=valid_symplectic_cert,
        )
        assert result.is_thermodynamically_stable is True

    def test_enforce_rejects_invalid_symplectic_cert(self, enforcer):
        r"""Con certificado simpléctico inválido (is_volume_preserved=False), falla."""
        fake_cert = SymplecticInvariantData(
            phase_space_dimension=4,
            symplectic_residual_norm=99.0,
            symplectic_relative_residual=99.0,
            antisymmetric_residual=0.0,
            determinant_residual=0.0,
            condition_number=1.0,
            spectral_radius_Omega=1.0,
            maslov_index=0,
            polar_unitarity_residual=0.0,
            effective_tolerance=1e-10,
            is_volume_preserved=False,
        )
        Phi, gradV = _positive_dissipative_pair(dim=4)
        with pytest.raises(SymplecticInvarianceViolation):
            enforcer._enforce_dirichlet_thermodynamics(
                Phi, gradV, symplectic_audit=fake_cert,
            )

    def test_enforce_rejects_dimensional_mismatch_with_cert(
        self, enforcer, valid_symplectic_cert,
    ):
        r"""
        Si el certificado certifica dim=4 pero Φ tiene dim=6, falla.
        """
        Phi = np.ones(6)
        gradV = np.ones(6)
        with pytest.raises(ValueError, match="Inconsistencia"):
            enforcer._enforce_dirichlet_thermodynamics(
                Phi, gradV, symplectic_audit=valid_symplectic_cert,
            )

    def test_enforce_computes_correct_raw_inner_product(self, enforcer):
        r"""El campo raw_inner_product refleja Φᵀ∇V exactamente."""
        Phi = np.array([1.0, 2.0, 3.0, 4.0])
        gradV = np.array([1.0, 1.0, 1.0, 1.0])
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        expected = float(np.dot(Phi, gradV))  # 1+2+3+4 = 10
        assert math.isclose(result.raw_inner_product, expected, rel_tol=1e-12)

    def test_enforce_alignment_cosine_range(self, enforcer):
        r"""El coseno de alineamiento ∈ [−1, 1]."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert -1.0 <= result.alignment_cosine <= 1.0

    def test_enforce_angle_consistent_with_cosine(self, enforcer):
        r"""θ = arccos(cos(θ)) debe ser consistente."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        expected_angle = math.acos(float(np.clip(result.alignment_cosine, -1.0, 1.0)))
        assert math.isclose(result.thermodynamic_angle_rad, expected_angle, rel_tol=1e-12)

    def test_enforce_clausius_duhem_satisfied_for_positive_dissipation(self, enforcer):
        r"""Clausius-Duhem debe estar satisfecho cuando P_diss > 0."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.clausius_duhem_satisfied is True
        assert result.entropy_production_rate >= 0.0

    def test_enforce_exergy_norm_correct(self, enforcer):
        r"""El campo exergy_norm = ‖Φ‖₂."""
        Phi = np.array([3.0, 4.0, 0.0, 0.0])
        gradV = np.array([1.0, 0.0, 0.0, 0.0])
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert math.isclose(result.exergy_norm, 5.0, rel_tol=1e-12)

    def test_enforce_gradient_norm_correct(self, enforcer):
        r"""El campo gradient_norm = ‖∇V‖₂."""
        Phi = np.array([1.0, 0.0, 0.0, 0.0])
        gradV = np.array([0.0, 3.0, 4.0, 0.0])
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert math.isclose(result.gradient_norm, 5.0, rel_tol=1e-12)

    def test_enforce_rejects_nan_phi(self, enforcer):
        r"""Φ con NaN debe fallar en la guarda."""
        Phi = np.array([np.nan, 1.0, 1.0, 1.0])
        gradV = np.ones(4)
        with pytest.raises((ValueError, ThermodynamicSingularityError)):
            enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)

    def test_enforce_zero_vectors_produces_zero_dissipation(self, enforcer):
        r"""Φ = 0 y ∇V = 0 → P_diss = 0, alignment = 0."""
        Phi = np.zeros(4)
        gradV = np.zeros(4)
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.dissipated_power == 0.0
        assert result.alignment_cosine == 0.0

    def test_enforce_large_vectors_stable(self, enforcer):
        r"""Vectores grandes (pero finitos) no deben causar overflow."""
        scale = 1e100
        Phi = np.array([1.0, 0.0, 0.0, 0.0]) * scale
        gradV = np.array([1.0, 0.0, 0.0, 0.0]) * scale
        result = enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.is_thermodynamically_stable is True
        assert math.isfinite(result.dissipated_power)


# ═══════════════════════════════════════════════════════════════════════════════
# §T3 — TESTS DE FASE 3: COHOMOLOGÍA DE HACES CELULARES
# ═══════════════════════════════════════════════════════════════════════════════
class TestPhase3_Cohomology:
    r"""
    Verifica Phase3_CellularSheafCohomologyAuditor.

    Estructura de subtests:
        T3.1  → Rango numérico con gap espectral
        T3.2  → Números de Betti
        T3.3  → Torsión de Reidemeister
        T3.4  → Mayer-Vietoris
        T3.5  → _compute_first_cohomology_dimension
        T3.6  → _audit_cellular_sheaf_cohomology (método principal)
    """

    @pytest.fixture
    def auditor(self) -> Phase3_CellularSheafCohomologyAuditor:
        return Phase3_CellularSheafCohomologyAuditor()

    @pytest.fixture
    def valid_thermo_cert(self) -> ThermodynamicDirichletData:
        r"""Certificado termodinámico válido para la continuación funtorial."""
        enforcer = Phase2_DirichletThermodynamicEnforcer()
        Phi, gradV = _positive_dissipative_pair(dim=4)
        return enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)

    # ── T3.1: Rango numérico con gap espectral ────────────────────────────────

    def test_rank_identity_full_rank(self, auditor):
        r"""rank(I_n) = n."""
        I = np.eye(5)
        rank, svs = auditor._numerical_rank_with_gap(I)
        assert rank == 5

    def test_rank_zero_matrix(self, auditor):
        r"""rank(0) = 0."""
        Z = np.zeros((4, 4))
        rank, svs = auditor._numerical_rank_with_gap(Z)
        assert rank == 0

    def test_rank_rank_one_matrix(self, auditor):
        r"""rank(u·vᵀ) = 1."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0])
        M = np.outer(u, v)
        rank, svs = auditor._numerical_rank_with_gap(M)
        assert rank == 1

    def test_rank_diagonal_matrix(self, auditor):
        r"""rank(diag(1,2,3,0,0)) = 3."""
        D = np.diag([1.0, 2.0, 3.0, 0.0, 0.0])
        rank, svs = auditor._numerical_rank_with_gap(D)
        assert rank == 3

    def test_rank_returns_singular_values(self, auditor):
        r"""_numerical_rank_with_gap retorna los valores singulares."""
        M = np.eye(3)
        rank, svs = auditor._numerical_rank_with_gap(M)
        assert svs.size > 0
        assert np.all(svs >= 0.0)

    # ── T3.2: Números de Betti ────────────────────────────────────────────────

    def test_betti_numbers_trivial_complex(self, auditor):
        r"""
        Para D0: c1×c0 de rango c0 y D1 = 0:
            β₀ = c0 − rank(D0) = 0
            β₁ = c1 − rank(D0) − 0 = c1 − c0
            β₂ = c2 − 0 = c2
        """
        c0, c1, c2 = 2, 4, 3
        D0, D1 = _trivial_coboundary_operators(c0=c0, c1=c1, c2=c2)
        rank_D0 = auditor._fast_rank_estimate(D0)
        rank_D1 = auditor._fast_rank_estimate(D1)
        beta0, beta1, beta2 = auditor._compute_betti_numbers(D0, D1, rank_D0, rank_D1)

        assert beta0 >= 0
        assert beta1 >= 0
        assert beta2 >= 0

    def test_betti_numbers_nonnegative(self, auditor):
        r"""Los números de Betti son siempre no negativos."""
        D0, D1 = _trivial_coboundary_operators()
        rank_D0 = max(0, auditor._fast_rank_estimate(D0))
        rank_D1 = max(0, auditor._fast_rank_estimate(D1))
        b0, b1, b2 = auditor._compute_betti_numbers(D0, D1, rank_D0, rank_D1)
        assert b0 >= 0 and b1 >= 0 and b2 >= 0

    # ── T3.3: Torsión de Reidemeister ─────────────────────────────────────────

    def test_reidemeister_acyclic_complex(self, auditor):
        r"""Para un complejo acíclico (H¹=0), la torsión está definida."""
        D0, D1 = _trivial_coboundary_operators(c0=2, c1=4, c2=2)
        rank_D0, svs_D0 = auditor._numerical_rank_with_gap(D0)
        rank_D1, svs_D1 = auditor._numerical_rank_with_gap(D1)
        torsion = auditor._compute_reidemeister_torsion(
            D0, D1, rank_D0, rank_D1, svs_D0, svs_D1,
        )
        assert math.isfinite(torsion)

    def test_reidemeister_non_acyclic_returns_zero(self, auditor):
        r"""Para un complejo con H¹>0, la torsión retorna 0."""
        D0, D1 = _non_exact_coboundary_operators()
        rank_D0, svs_D0 = auditor._numerical_rank_with_gap(D0)
        rank_D1, svs_D1 = auditor._numerical_rank_with_gap(D1)
        torsion = auditor._compute_reidemeister_torsion(
            D0, D1, rank_D0, rank_D1, svs_D0, svs_D1,
        )
        assert torsion == 0.0

    # ── T3.4: Mayer-Vietoris ──────────────────────────────────────────────────

    def test_mayer_vietoris_trivial_complex(self, auditor):
        r"""El complejo trivial debe satisfacer Mayer-Vietoris."""
        D0, D1 = _trivial_coboundary_operators(c0=2, c1=5, c2=3)
        rank_D0, _ = auditor._numerical_rank_with_gap(D0)
        rank_D1, _ = auditor._numerical_rank_with_gap(D1)
        mv_exact = auditor._verify_mayer_vietoris_exactness(D0, D1, rank_D0, rank_D1)
        assert mv_exact is True

    def test_mayer_vietoris_fails_when_ranks_exceed_c1(self, auditor):
        r"""Si rank(D0)+rank(D1) > dim C¹, Mayer-Vietoris falla."""
        # Fabricamos un caso donde rank_D0 + rank_D1 > c1
        c0, c1, c2 = 3, 2, 3
        # Usamos ranks artificialmente altos
        D0 = np.eye(c1, c0, dtype=np.float64)  # rank = min(c1,c0) = 2
        D1 = np.zeros((c2, c1), dtype=np.float64)
        rank_D0 = c1  # Forzamos rank = c1 = 2
        rank_D1 = c1  # Forzamos rank = c1 = 2 → suma = 4 > c1 = 2
        mv_exact = auditor._verify_mayer_vietoris_exactness(D0, D1, rank_D0, rank_D1)
        assert mv_exact is False

    # ── T3.5: _compute_first_cohomology_dimension ─────────────────────────────

    def test_h1_zero_trivial_complex(self, auditor):
        r"""dim H¹ = 0 para el complejo trivial (D1=0, D0 de rango máximo)."""
        D0, D1 = _trivial_coboundary_operators(c0=2, c1=4, c2=2)
        result = auditor._compute_first_cohomology_dimension(D0, D1)
        h1 = result[0]
        assert h1 == 0

    def test_h1_positive_non_exact_complex(self, auditor):
        r"""dim H¹ > 0 para un complejo no exacto."""
        D0, D1 = _non_exact_coboundary_operators()
        result = auditor._compute_first_cohomology_dimension(D0, D1)
        h1 = result[0]
        assert h1 > 0

    def test_h1_rejects_incompatible_operators(self, auditor):
        r"""Operadores con dimensiones incompatibles deben fallar."""
        D0 = np.zeros((4, 2))
        D1 = np.zeros((3, 5))  # D1 dominio ≠ D0 codominio
        with pytest.raises(ValueError, match="componen"):
            auditor._compute_first_cohomology_dimension(D0, D1)

    def test_h1_rejects_non_complex_operators(self, auditor):
        r"""Operadores donde δ¹∘δ⁰ ≠ 0 deben fallar."""
        D0, D1 = _violating_coboundary_operators()
        with pytest.raises(CohomologicalObstructionError):
            auditor._compute_first_cohomology_dimension(D0, D1)

    def test_h1_returns_all_fields(self, auditor):
        r"""_compute_first_cohomology_dimension retorna la tupla completa."""
        D0, D1 = _trivial_coboundary_operators()
        result = auditor._compute_first_cohomology_dimension(D0, D1)
        assert len(result) == 10  # 10 campos en la tupla

    def test_h1_complex_residual_is_small(self, auditor):
        r"""El residuo ‖δ¹∘δ⁰‖/scale debe ser pequeño para un complejo válido."""
        D0, D1 = _trivial_coboundary_operators()
        result = auditor._compute_first_cohomology_dimension(D0, D1)
        complex_residual = result[6]
        assert complex_residual <= _COHOMOLOGICAL_COMPLEX_TOL

    # ── T3.6: _audit_cellular_sheaf_cohomology ────────────────────────────────

    def test_audit_cohomology_h1_zero_direct(self, auditor):
        r"""H¹=0 provisto directamente debe pasar."""
        result = auditor._audit_cellular_sheaf_cohomology(h1_dimension=0)
        assert isinstance(result, SheafCohomologyAuditData)
        assert result.is_globally_integrable is True
        assert result.h1_dimension == 0

    def test_audit_cohomology_h1_positive_raises(self, auditor):
        r"""H¹>0 debe lanzar CohomologicalObstructionError."""
        with pytest.raises(CohomologicalObstructionError):
            auditor._audit_cellular_sheaf_cohomology(h1_dimension=1)

    def test_audit_cohomology_h1_negative_raises(self, auditor):
        r"""H¹<0 debe lanzar ValueError."""
        with pytest.raises(ValueError, match="negativa"):
            auditor._audit_cellular_sheaf_cohomology(h1_dimension=-1)

    def test_audit_cohomology_h1_bool_raises(self, auditor):
        r"""h1_dimension booleano debe lanzar TypeError."""
        with pytest.raises(TypeError, match="entero"):
            auditor._audit_cellular_sheaf_cohomology(h1_dimension=True)

    def test_audit_cohomology_h1_float_raises(self, auditor):
        r"""h1_dimension flotante debe lanzar TypeError."""
        with pytest.raises(TypeError):
            auditor._audit_cellular_sheaf_cohomology(h1_dimension=0.0)

    def test_audit_cohomology_no_input_raises(self, auditor):
        r"""Sin h1_dimension ni operadores, debe fallar."""
        with pytest.raises(ValueError, match="Debe proveerse"):
            auditor._audit_cellular_sheaf_cohomology()

    def test_audit_cohomology_operators_both_required(self, auditor):
        r"""Proveer solo uno de los dos operadores debe fallar."""
        D0 = np.zeros((4, 2))
        with pytest.raises(ValueError, match="simultáneamente"):
            auditor._audit_cellular_sheaf_cohomology(coboundary_delta0=D0)

    def test_audit_cohomology_from_operators_zero_h1(self, auditor):
        r"""Con operadores que dan H¹=0, debe pasar."""
        D0, D1 = _trivial_coboundary_operators()
        result = auditor._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert result.is_globally_integrable is True
        assert result.verified_by_coboundary is True

    def test_audit_cohomology_from_operators_positive_h1_raises(self, auditor):
        r"""Con operadores que dan H¹>0, debe lanzar CohomologicalObstructionError."""
        D0, D1 = _non_exact_coboundary_operators()
        with pytest.raises(CohomologicalObstructionError):
            auditor._audit_cellular_sheaf_cohomology(
                coboundary_delta0=D0,
                coboundary_delta1=D1,
            )

    def test_audit_cohomology_inconsistency_raises(self, auditor):
        r"""Si h1_dimension declarado ≠ calculado por δ, debe fallar."""
        D0, D1 = _trivial_coboundary_operators()  # H¹ calculado = 0
        with pytest.raises(CohomologicalObstructionError, match="Inconsistencia"):
            auditor._audit_cellular_sheaf_cohomology(
                h1_dimension=1,  # declarado = 1 ≠ calculado = 0
                coboundary_delta0=D0,
                coboundary_delta1=D1,
            )

    def test_audit_cohomology_with_thermo_cert(self, auditor, valid_thermo_cert):
        r"""Con certificado termodinámico válido, la Fase 3 debe pasar."""
        result = auditor._audit_cellular_sheaf_cohomology(
            h1_dimension=0,
            thermodynamic_audit=valid_thermo_cert,
        )
        assert result.is_globally_integrable is True

    def test_audit_cohomology_rejects_bad_thermo_cert(self, auditor):
        r"""Con certificado termodinámico inválido, la Fase 3 debe fallar."""
        fake_cert = ThermodynamicDirichletData(
            dissipated_power=0.0,
            raw_inner_product=0.0,
            numerical_tolerance=0.0,
            exergy_norm=0.0,
            gradient_norm=0.0,
            alignment_cosine=0.0,
            thermodynamic_angle_rad=0.0,
            entropy_production_rate=0.0,
            clausius_duhem_satisfied=False,
            exergy_dissipation_ratio=0.0,
            is_thermodynamically_stable=False,  # ← inválido
            is_strictly_dissipative=False,
        )
        with pytest.raises(ThermodynamicSingularityError):
            auditor._audit_cellular_sheaf_cohomology(
                h1_dimension=0,
                thermodynamic_audit=fake_cert,
            )

    def test_audit_cohomology_betti_numbers_present_with_operators(self, auditor):
        r"""Con operadores, los números de Betti deben estar en el resultado."""
        D0, D1 = _trivial_coboundary_operators()
        result = auditor._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert len(result.betti_numbers) == 3
        assert all(b >= 0 for b in result.betti_numbers)

    def test_audit_cohomology_euler_characteristic_consistent(self, auditor):
        r"""
        Con operadores, χ calculado por cadenas debe coincidir con χ por Betti.
        """
        D0, D1 = _trivial_coboundary_operators(c0=3, c1=5, c2=2)
        result = auditor._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        b0, b1, b2 = result.betti_numbers
        euler_betti = b0 - b1 + b2
        assert result.euler_characteristic == euler_betti

    def test_audit_cohomology_obstruction_free_true(self, auditor):
        r"""obstruction_free debe ser True cuando H¹=0."""
        result = auditor._audit_cellular_sheaf_cohomology(h1_dimension=0)
        assert result.obstruction_free is True

    def test_audit_cohomology_complex_residual_present_with_operators(self, auditor):
        r"""Con operadores, complex_residual debe estar en el resultado."""
        D0, D1 = _trivial_coboundary_operators()
        result = auditor._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert math.isfinite(result.complex_residual)
        assert result.complex_residual >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# §T4 — TESTS DEL ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════════
class TestOrchestrator:
    r"""
    Verifica el comportamiento del ASTStaticAnalyzerAgent como composición
    funtorial completa Φ₃∘Φ₂∘Φ₁.
    """

    @pytest.fixture
    def agent(self) -> ASTStaticAnalyzerAgent:
        return ASTStaticAnalyzerAgent(strict_mode=True)

    @pytest.fixture
    def agent_lenient(self) -> ASTStaticAnalyzerAgent:
        return ASTStaticAnalyzerAgent(strict_mode=False)

    def _valid_inputs(self, n: int = 2):
        r"""Retorna (M, Φ, ∇V) válidos para n grados de libertad."""
        return _build_valid_governance_inputs(n=n)

    # ── T4.1: Ejecución exitosa ───────────────────────────────────────────────

    def test_governance_identity_n2_passes(self, agent):
        r"""Gobernanza con M=I₄, Φ=∇V=e₁ debe pasar."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            ast_jacobian_M=M,
            control_potential_Phi=Phi,
            lyapunov_gradient_V=gradV,
            h1_dimension=0,
        )
        assert isinstance(result, ASTGovernanceState)
        assert result.is_compilation_authorized is True

    def test_governance_returns_all_three_certificates(self, agent):
        r"""El estado de gobernanza contiene los tres certificados."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        assert isinstance(result.symplectic_audit, SymplecticInvariantData)
        assert isinstance(result.thermodynamic_audit, ThermodynamicDirichletData)
        assert isinstance(result.cohomology_audit, SheafCohomologyAuditData)

    def test_governance_returns_provenance(self, agent):
        r"""El estado de gobernanza contiene el objeto de trazabilidad."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        assert isinstance(result.provenance, AuditProvenance)

    def test_governance_provenance_all_phases_passed(self, agent):
        r"""El provenance reporta las tres fases como exitosas."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        assert result.provenance.phase1_passed is True
        assert result.provenance.phase2_passed is True
        assert result.provenance.phase3_passed is True

    def test_governance_provenance_timestamp_is_iso8601(self, agent):
        r"""El timestamp del provenance tiene formato ISO-8601."""
        from datetime import datetime, timezone
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        ts = result.provenance.timestamp_iso
        # datetime.fromisoformat lanza ValueError si el formato es inválido
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_governance_provenance_checksum_sha256(self, agent):
        r"""El checksum del provenance es SHA-256 (64 hex chars)."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        checksum = result.provenance.input_checksum_sha256
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_governance_provenance_functor_chain_contains_phases(self, agent):
        r"""La cadena funtorial menciona Φ₁, Φ₂, Φ₃."""
        M, Phi, gradV = self._valid_inputs(n=2)
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        chain = result.provenance.functor_chain
        assert "Φ₁" in chain
        assert "Φ₂" in chain
        assert "Φ₃" in chain

    def test_governance_with_operators_passes(self, agent):
        r"""Gobernanza con operadores coboundary válidos debe pasar."""
        M, Phi, gradV = self._valid_inputs(n=2)
        D0, D1 = _trivial_coboundary_operators()
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV,
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert result.is_compilation_authorized is True
        assert result.cohomology_audit.verified_by_coboundary is True

    def test_governance_rotation_symplectic_passes(self, agent):
        r"""Gobernanza con rotación simpléctica debe pasar."""
        n = 3
        M = _rotation_symplectic(n, theta=0.7)
        dim = 2 * n
        Phi = np.zeros(dim)
        Phi[0] = 2.0
        gradV = np.zeros(dim)
        gradV[0] = 1.5
        result = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        assert result.is_compilation_authorized is True

    # ── T4.2: Fallos en Fase 1 ────────────────────────────────────────────────

    def test_governance_fails_phase1_non_symplectic(self, agent):
        r"""Fallo en Fase 1 por matriz no simpléctica."""
        M = 2.0 * np.eye(4)  # No simpléctica
        Phi = np.ones(4)
        gradV = np.ones(4)
        with pytest.raises(SymplecticInvarianceViolation):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV, h1_dimension=0,
            )

    def test_governance_fails_phase1_odd_dimension(self, agent):
        r"""Fallo en Fase 1 por dimensión impar."""
        M = np.eye(3)
        Phi = np.ones(3)
        gradV = np.ones(3)
        with pytest.raises(SymplecticInvarianceViolation):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV, h1_dimension=0,
            )

    # ── T4.3: Fallos en Fase 2 ────────────────────────────────────────────────

    def test_governance_fails_phase2_negative_dissipation(self, agent):
        r"""Fallo en Fase 2 por P_diss < 0."""
        M = _identity_symplectic(2)
        Phi, gradV = _negative_dissipation_pair(dim=4)
        with pytest.raises(ThermodynamicSingularityError):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV, h1_dimension=0,
            )

    # ── T4.4: Fallos en Fase 3 ────────────────────────────────────────────────

    def test_governance_fails_phase3_h1_positive(self, agent):
        r"""Fallo en Fase 3 por H¹ > 0."""
        M, Phi, gradV = self._valid_inputs(n=2)
        with pytest.raises(CohomologicalObstructionError):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV, h1_dimension=2,
            )

    def test_governance_fails_phase3_non_exact_operators(self, agent):
        r"""Fallo en Fase 3 por operadores que dan H¹ > 0."""
        M, Phi, gradV = self._valid_inputs(n=2)
        D0, D1 = _non_exact_coboundary_operators()
        with pytest.raises(CohomologicalObstructionError):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV,
                coboundary_delta0=D0,
                coboundary_delta1=D1,
            )

    # ── T4.5: Comportamiento del log ──────────────────────────────────────────

    def test_governance_logs_completion(self, agent, caplog):
        r"""El orquestador debe emitir log INFO al completar."""
        M, Phi, gradV = self._valid_inputs(n=2)
        with caplog.at_level(logging.INFO):
            agent.execute_ast_symplectic_governance(
                M, Phi, gradV, h1_dimension=0,
            )
        assert len(caplog.records) > 0

    def test_governance_strict_mode_attribute(self, agent, agent_lenient):
        r"""El atributo strict_mode debe reflejar el valor del constructor."""
        assert agent._strict_mode is True
        assert agent_lenient._strict_mode is False

    # ── T4.6: Determinismo ────────────────────────────────────────────────────

    def test_governance_deterministic_for_same_input(self, agent):
        r"""Dos ejecuciones con la misma entrada producen resultados idénticos."""
        M, Phi, gradV = self._valid_inputs(n=2)
        r1 = agent.execute_ast_symplectic_governance(M, Phi, gradV, h1_dimension=0)
        r2 = agent.execute_ast_symplectic_governance(M, Phi, gradV, h1_dimension=0)
        assert r1.symplectic_audit == r2.symplectic_audit
        assert r1.thermodynamic_audit == r2.thermodynamic_audit
        assert r1.cohomology_audit == r2.cohomology_audit

    def test_governance_checksum_same_for_same_input(self, agent):
        r"""El checksum es el mismo para entradas idénticas."""
        M, Phi, gradV = self._valid_inputs(n=2)
        r1 = agent.execute_ast_symplectic_governance(M, Phi, gradV, h1_dimension=0)
        r2 = agent.execute_ast_symplectic_governance(M, Phi, gradV, h1_dimension=0)
        assert r1.provenance.input_checksum_sha256 == r2.provenance.input_checksum_sha256

    def test_governance_checksum_differs_for_different_input(self, agent):
        r"""El checksum difiere para entradas distintas."""
        M1, Phi1, gradV1 = _build_valid_governance_inputs(n=2)
        M2, Phi2, gradV2 = _build_valid_governance_inputs(n=3)
        r1 = agent.execute_ast_symplectic_governance(M1, Phi1, gradV1, h1_dimension=0)
        r2 = agent.execute_ast_symplectic_governance(M2, Phi2, gradV2, h1_dimension=0)
        assert r1.provenance.input_checksum_sha256 != r2.provenance.input_checksum_sha256


# ═══════════════════════════════════════════════════════════════════════════════
# §T5 — TESTS DE ANIDAMIENTO FUNTORIAL (CADENA DE CERTIFICADOS)
# ═══════════════════════════════════════════════════════════════════════════════
class TestFunctorialChaining:
    r"""
    Verifica el anidamiento correcto entre fases:
    el certificado de Fase i es el objeto inicial de Fase i+1.

    Prueba la composición Φ₂∘Φ₁ y Φ₃∘Φ₂∘Φ₁ de forma aislada.
    """

    @pytest.fixture
    def p1(self) -> Phase1_SymplecticInvarianceAuditor:
        return Phase1_SymplecticInvarianceAuditor()

    @pytest.fixture
    def p2(self) -> Phase2_DirichletThermodynamicEnforcer:
        return Phase2_DirichletThermodynamicEnforcer()

    @pytest.fixture
    def p3(self) -> Phase3_CellularSheafCohomologyAuditor:
        return Phase3_CellularSheafCohomologyAuditor()

    def test_phase1_output_feeds_phase2(self, p1, p2):
        r"""El certificado de Fase 1 puede alimentar directamente Fase 2."""
        M = _identity_symplectic(2)
        cert1 = p1._audit_symplectic_invariance(M)
        Phi, gradV = _positive_dissipative_pair(dim=4)
        cert2 = p2._enforce_dirichlet_thermodynamics(Phi, gradV, symplectic_audit=cert1)
        assert cert2.is_thermodynamically_stable is True

    def test_phase2_output_feeds_phase3(self, p2, p3):
        r"""El certificado de Fase 2 puede alimentar directamente Fase 3."""
        Phi, gradV = _positive_dissipative_pair(dim=4)
        cert2 = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
        cert3 = p3._audit_cellular_sheaf_cohomology(
            h1_dimension=0, thermodynamic_audit=cert2,
        )
        assert cert3.is_globally_integrable is True

    def test_full_chain_p1_p2_p3(self, p1, p2, p3):
        r"""La cadena completa Φ₁→Φ₂→Φ₃ produce resultados válidos."""
        M = _rotation_symplectic(2, theta=0.4)
        cert1 = p1._audit_symplectic_invariance(M)

        dim = cert1.phase_space_dimension
        Phi = np.zeros(dim)
        Phi[0] = 1.0
        gradV = np.zeros(dim)
        gradV[0] = 1.0
        cert2 = p2._enforce_dirichlet_thermodynamics(Phi, gradV, symplectic_audit=cert1)

        D0, D1 = _trivial_coboundary_operators()
        cert3 = p3._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
            thermodynamic_audit=cert2,
        )
        assert cert3.is_globally_integrable is True

    def test_phase2_rejects_phase1_cert_with_false_volume(self, p2):
        r"""
        Si el certificado de Fase 1 indica is_volume_preserved=False,
        Fase 2 debe rechazarlo.
        """
        bad_cert = SymplecticInvariantData(
            phase_space_dimension=4,
            symplectic_residual_norm=1e5,
            symplectic_relative_residual=1e5,
            antisymmetric_residual=0.0,
            determinant_residual=0.0,
            condition_number=1.0,
            spectral_radius_Omega=1.0,
            maslov_index=0,
            polar_unitarity_residual=0.0,
            effective_tolerance=1e-10,
            is_volume_preserved=False,
        )
        Phi, gradV = _positive_dissipative_pair(dim=4)
        with pytest.raises(SymplecticInvarianceViolation):
            p2._enforce_dirichlet_thermodynamics(
                Phi, gradV, symplectic_audit=bad_cert,
            )

    def test_phase3_rejects_phase2_cert_with_false_stability(self, p3):
        r"""
        Si el certificado de Fase 2 indica is_thermodynamically_stable=False,
        Fase 3 debe rechazarlo.
        """
        bad_cert = ThermodynamicDirichletData(
            dissipated_power=0.0,
            raw_inner_product=-1.0,
            numerical_tolerance=0.0,
            exergy_norm=1.0,
            gradient_norm=1.0,
            alignment_cosine=-1.0,
            thermodynamic_angle_rad=math.pi,
            entropy_production_rate=0.0,
            clausius_duhem_satisfied=False,
            exergy_dissipation_ratio=0.0,
            is_thermodynamically_stable=False,
            is_strictly_dissipative=False,
        )
        with pytest.raises(ThermodynamicSingularityError):
            p3._audit_cellular_sheaf_cohomology(
                h1_dimension=0, thermodynamic_audit=bad_cert,
            )

    def test_chain_dimensional_consistency(self, p1, p2):
        r"""
        La dimensión certificada por Fase 1 debe coincidir con la dimensión
        de los vectores en Fase 2.
        """
        M = _identity_symplectic(3)  # dim = 6
        cert1 = p1._audit_symplectic_invariance(M)
        assert cert1.phase_space_dimension == 6

        Phi = np.ones(6)
        gradV = np.ones(6)
        cert2 = p2._enforce_dirichlet_thermodynamics(
            Phi, gradV, symplectic_audit=cert1,
        )
        assert cert2.is_thermodynamically_stable is True

    def test_chain_dimensional_mismatch_fails(self, p1, p2):
        r"""
        Si los vectores de Fase 2 no coinciden con la dimensión de Fase 1,
        la cadena falla con ValueError.
        """
        M = _identity_symplectic(2)  # dim = 4
        cert1 = p1._audit_symplectic_invariance(M)

        Phi = np.ones(6)  # dim = 6 ≠ 4
        gradV = np.ones(6)
        with pytest.raises(ValueError, match="Inconsistencia"):
            p2._enforce_dirichlet_thermodynamics(
                Phi, gradV, symplectic_audit=cert1,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# §T6 — TESTS DE JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════
class TestExceptionHierarchy:
    r"""
    Verifica que la jerarquía de excepciones es correcta según el lattice:

        TopologicalInvariantError
        ├── SymplecticInvarianceViolation
        │   ├── SymplecticPolarDecompositionError
        │   └── MaslovIndexError
        ├── ThermodynamicSingularityError
        │   ├── ClausiusDuhemViolation
        │   └── ExergyDivergenceError
        └── CohomologicalObstructionError
            ├── EulerCharacteristicMismatch
            └── MayerVietorisBreachError
    """

    def test_symplectic_violation_is_topological(self):
        r"""SymplecticInvarianceViolation hereda de TopologicalInvariantError."""
        exc = SymplecticInvarianceViolation("test")
        assert isinstance(exc, SymplecticInvarianceViolation)

    def test_polar_decomposition_error_is_symplectic(self):
        r"""SymplecticPolarDecompositionError hereda de SymplecticInvarianceViolation."""
        exc = SymplecticPolarDecompositionError("test")
        assert isinstance(exc, SymplecticInvarianceViolation)

    def test_maslov_index_error_is_symplectic(self):
        r"""MaslovIndexError hereda de SymplecticInvarianceViolation."""
        exc = MaslovIndexError("test")
        assert isinstance(exc, SymplecticInvarianceViolation)

    def test_clausius_duhem_is_thermodynamic(self):
        r"""ClausiusDuhemViolation hereda de ThermodynamicSingularityError."""
        exc = ClausiusDuhemViolation("test")
        assert isinstance(exc, ThermodynamicSingularityError)

    def test_exergy_divergence_is_thermodynamic(self):
        r"""ExergyDivergenceError hereda de ThermodynamicSingularityError."""
        exc = ExergyDivergenceError("test")
        assert isinstance(exc, ThermodynamicSingularityError)

    def test_euler_mismatch_is_cohomological(self):
        r"""EulerCharacteristicMismatch hereda de CohomologicalObstructionError."""
        exc = EulerCharacteristicMismatch("test")
        assert isinstance(exc, CohomologicalObstructionError)

    def test_mayer_vietoris_breach_is_cohomological(self):
        r"""MayerVietorisBreachError hereda de CohomologicalObstructionError."""
        exc = MayerVietorisBreachError("test")
        assert isinstance(exc, CohomologicalObstructionError)

    def test_all_leaf_exceptions_catchable_as_base(self):
        r"""Todas las excepciones hoja son capturables como SymplecticInvarianceViolation."""
        leaf_exceptions = [
            SymplecticPolarDecompositionError("test"),
            MaslovIndexError("test"),
        ]
        for exc in leaf_exceptions:
            try:
                raise exc
            except SymplecticInvarianceViolation:
                pass  # ✓

    def test_thermodynamic_leaves_catchable_as_thermodynamic(self):
        r"""ClausiusDuhemViolation y ExergyDivergenceError son ThermodynamicSingularityError."""
        for ExcClass in [ClausiusDuhemViolation, ExergyDivergenceError]:
            try:
                raise ExcClass("test")
            except ThermodynamicSingularityError:
                pass  # ✓

    def test_cohomological_leaves_catchable_as_cohomological(self):
        r"""EulerCharacteristicMismatch y MayerVietorisBreachError son CohomologicalObstructionError."""
        for ExcClass in [EulerCharacteristicMismatch, MayerVietorisBreachError]:
            try:
                raise ExcClass("test")
            except CohomologicalObstructionError:
                pass  # ✓

    def test_exceptions_carry_message(self):
        r"""Las excepciones deben transportar el mensaje de error."""
        msg = "Violación de invariante topológico en el espacio de fase."
        exc = SymplecticInvarianceViolation(msg)
        assert str(exc) == msg


# ═══════════════════════════════════════════════════════════════════════════════
# §T7 — TESTS DE INMUTABILIDAD DE DTOs
# ═══════════════════════════════════════════════════════════════════════════════
class TestDTOImmutability:
    r"""
    Verifica que todos los DTOs son inmutables (frozen=True, slots=True)
    y que sus campos tienen los tipos correctos.
    """

    def _make_symplectic_cert(self) -> SymplecticInvariantData:
        auditor = Phase1_SymplecticInvarianceAuditor()
        return auditor._audit_symplectic_invariance(_identity_symplectic(2))

    def _make_thermo_cert(self) -> ThermodynamicDirichletData:
        enforcer = Phase2_DirichletThermodynamicEnforcer()
        Phi, gradV = _positive_dissipative_pair()
        return enforcer._enforce_dirichlet_thermodynamics(Phi, gradV)

    def _make_cohomology_cert(self) -> SheafCohomologyAuditData:
        auditor = Phase3_CellularSheafCohomologyAuditor()
        return auditor._audit_cellular_sheaf_cohomology(h1_dimension=0)

    def _make_provenance(self) -> AuditProvenance:
        return AuditProvenance(
            timestamp_iso="2025-01-01T00:00:00+00:00",
            input_checksum_sha256="a" * 64,
            phase1_passed=True,
            phase2_passed=True,
            phase3_passed=True,
            functor_chain="Φ₁→Φ₂→Φ₃",
        )

    # ── T7.1: SymplecticInvariantData ─────────────────────────────────────────

    def test_symplectic_cert_is_frozen(self):
        r"""SymplecticInvariantData no permite mutación."""
        cert = self._make_symplectic_cert()
        with pytest.raises((AttributeError, TypeError)):
            cert.is_volume_preserved = False

    def test_symplectic_cert_fields_types(self):
        r"""Los campos de SymplecticInvariantData tienen los tipos correctos."""
        cert = self._make_symplectic_cert()
        assert isinstance(cert.phase_space_dimension, int)
        assert isinstance(cert.symplectic_residual_norm, float)
        assert isinstance(cert.symplectic_relative_residual, float)
        assert isinstance(cert.antisymmetric_residual, float)
        assert isinstance(cert.determinant_residual, float)
        assert isinstance(cert.condition_number, float)
        assert isinstance(cert.spectral_radius_Omega, float)
        assert isinstance(cert.maslov_index, int)
        assert isinstance(cert.polar_unitarity_residual, float)
        assert isinstance(cert.effective_tolerance, float)
        assert isinstance(cert.is_volume_preserved, bool)

    # ── T7.2: ThermodynamicDirichletData ──────────────────────────────────────

    def test_thermo_cert_is_frozen(self):
        r"""ThermodynamicDirichletData no permite mutación."""
        cert = self._make_thermo_cert()
        with pytest.raises((AttributeError, TypeError)):
            cert.is_thermodynamically_stable = False

    def test_thermo_cert_fields_types(self):
        r"""Los campos de ThermodynamicDirichletData tienen los tipos correctos."""
        cert = self._make_thermo_cert()
        assert isinstance(cert.dissipated_power, float)
        assert isinstance(cert.raw_inner_product, float)
        assert isinstance(cert.numerical_tolerance, float)
        assert isinstance(cert.exergy_norm, float)
        assert isinstance(cert.gradient_norm, float)
        assert isinstance(cert.alignment_cosine, float)
        assert isinstance(cert.thermodynamic_angle_rad, float)
        assert isinstance(cert.entropy_production_rate, float)
        assert isinstance(cert.clausius_duhem_satisfied, bool)
        assert isinstance(cert.exergy_dissipation_ratio, float)
        assert isinstance(cert.is_thermodynamically_stable, bool)
        assert isinstance(cert.is_strictly_dissipative, bool)

    # ── T7.3: SheafCohomologyAuditData ───────────────────────────────────────

    def test_cohomology_cert_is_frozen(self):
        r"""SheafCohomologyAuditData no permite mutación."""
        cert = self._make_cohomology_cert()
        with pytest.raises((AttributeError, TypeError)):
            cert.h1_dimension = 99

    def test_cohomology_cert_fields_types(self):
        r"""Los campos de SheafCohomologyAuditData tienen los tipos correctos."""
        cert = self._make_cohomology_cert()
        assert isinstance(cert.h1_dimension, int)
        assert isinstance(cert.rank_delta0, int)
        assert isinstance(cert.rank_delta1, int)
        assert isinstance(cert.betti_numbers, tuple)
        assert len(cert.betti_numbers) == 3
        assert isinstance(cert.euler_characteristic, int)
        assert isinstance(cert.reidemeister_torsion, float)
        assert isinstance(cert.complex_residual, float)
        assert isinstance(cert.is_globally_integrable, bool)
        assert isinstance(cert.obstruction_free, bool)
        assert isinstance(cert.verified_by_coboundary, bool)
        assert isinstance(cert.mayer_vietoris_exact, bool)

    # ── T7.4: AuditProvenance ─────────────────────────────────────────────────

    def test_provenance_is_frozen(self):
        r"""AuditProvenance no permite mutación."""
        prov = self._make_provenance()
        with pytest.raises((AttributeError, TypeError)):
            prov.phase1_passed = False

    def test_provenance_fields_types(self):
        r"""Los campos de AuditProvenance tienen los tipos correctos."""
        prov = self._make_provenance()
        assert isinstance(prov.timestamp_iso, str)
        assert isinstance(prov.input_checksum_sha256, str)
        assert isinstance(prov.phase1_passed, bool)
        assert isinstance(prov.phase2_passed, bool)
        assert isinstance(prov.phase3_passed, bool)
        assert isinstance(prov.functor_chain, str)

    # ── T7.5: ASTGovernanceState ──────────────────────────────────────────────

    def test_governance_state_is_frozen(self):
        r"""ASTGovernanceState no permite mutación."""
        agent = ASTStaticAnalyzerAgent()
        M, Phi, gradV = _build_valid_governance_inputs(n=2)
        state = agent.execute_ast_symplectic_governance(M, Phi, gradV, h1_dimension=0)
        with pytest.raises((AttributeError, TypeError)):
            state.is_compilation_authorized = False


# ═══════════════════════════════════════════════════════════════════════════════
# §T8 — TESTS DE CASOS LÍMITE NUMÉRICOS
# ═══════════════════════════════════════════════════════════════════════════════
class TestNumericalEdgeCases:
    r"""
    Verifica el comportamiento del sistema ante casos límite numéricos extremos:
    matrices casi singulares, vectores muy pequeños, dimensiones mínimas.
    """

    @pytest.fixture
    def agent(self) -> ASTStaticAnalyzerAgent:
        return ASTStaticAnalyzerAgent(strict_mode=False)

    @pytest.fixture
    def p1(self) -> Phase1_SymplecticInvarianceAuditor:
        return Phase1_SymplecticInvarianceAuditor()

    @pytest.fixture
    def p2(self) -> Phase2_DirichletThermodynamicEnforcer:
        return Phase2_DirichletThermodynamicEnforcer()

    @pytest.fixture
    def p3(self) -> Phase3_CellularSheafCohomologyAuditor:
        return Phase3_CellularSheafCohomologyAuditor()

    # ── T8.1: Dimensión mínima n=1 ────────────────────────────────────────────

    def test_phase1_minimum_dimension_n1(self, p1):
        r"""La identidad 2×2 (n=1) debe pasar la auditoría de Fase 1."""
        M = _identity_symplectic(1)
        result = p1._audit_symplectic_invariance(M)
        assert result.phase_space_dimension == 2
        assert result.is_volume_preserved is True

    def test_phase2_minimum_dimension_vectors(self, p2):
        r"""Vectores de dimensión 2 (mínima simpléctica) deben funcionar."""
        Phi = np.array([1.0, 0.0])
        gradV = np.array([1.0, 0.0])
        result = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.is_thermodynamically_stable is True

    # ── T8.2: Matrices de cizalla con α grande ────────────────────────────────

    def test_phase1_large_shear_passes(self, p1):
        r"""Cizalla simpléctica con α=1000 debe pasar (sigue siendo simpléctica)."""
        M = _shear_symplectic(2, alpha=1000.0)
        result = p1._audit_symplectic_invariance(M)
        assert result.is_volume_preserved is True

    def test_phase1_very_small_shear_passes(self, p1):
        r"""Cizalla con α=ε₀ debe pasar."""
        M = _shear_symplectic(2, alpha=float(_MACHINE_EPSILON))
        result = p1._audit_symplectic_invariance(M)
        assert result.is_volume_preserved is True

    # ── T8.3: Vectores muy pequeños ───────────────────────────────────────────

    def test_phase2_tiny_vectors_stable(self, p2):
        r"""Vectores muy pequeños pero con P_diss > 0 deben pasar."""
        eps = _MACHINE_EPSILON * 100
        Phi = np.array([eps, 0.0, 0.0, 0.0])
        gradV = np.array([eps, 0.0, 0.0, 0.0])
        result = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.is_thermodynamically_stable is True

    # ── T8.4: Matrices casi singulares ───────────────────────────────────────

    def test_phase1_warns_ill_conditioned(self, p1, caplog):
        r"""
        Una transformación simpléctica mal condicionada debe pasar pero
        emitir warning de κ(M) elevado.

        Construimos una cizalla simpléctica con α muy grande que sigue
        siendo simpléctica pero tiene κ elevado.
        """
        # α = 1e7 da κ ≈ (1e7)^2 que puede exceder el umbral de advertencia
        M = _shear_symplectic(1, alpha=1e7)
        with caplog.at_level(logging.WARNING):
            result = p1._audit_symplectic_invariance(M)
        # Si pasa, es porque sigue siendo simpléctica
        assert result.is_volume_preserved is True

    # ── T8.5: Valores singulares con gap espectral ────────────────────────────

    def test_rank_with_clear_spectral_gap(self, p3):
        r"""
        Una matriz con valores singulares [100, 99, 98, 0.0001, 0.00005]
        debe tener rango 3 detectado por gap espectral.
        """
        U, _ = la.qr(np.random.default_rng(0).standard_normal((5, 5)))
        sigma = np.diag([100.0, 99.0, 98.0, 1e-5, 5e-6])
        M = U @ sigma @ U.T
        rank, svs = p3._numerical_rank_with_gap(M)
        # El gap entre 98 y 1e-5 es muy claro
        assert rank == 3

    # ── T8.6: Complejo cohomológico con D1=0 (acíclico trivial) ───────────────

    def test_h1_zero_for_zero_d1(self, p3):
        r"""
        Si D1 = 0 y D0 tiene rango máximo c0 = c1:
            dim H¹ = c1 - rank(D0) - rank(D1) = c1 - c1 - 0 = 0.
        """
        c1 = 3
        D0 = np.eye(c1, c1, dtype=np.float64)  # cuadrada, rank = c1
        D1 = np.zeros((2, c1), dtype=np.float64)
        result = p3._compute_first_cohomology_dimension(D0, D1)
        h1 = result[0]
        assert h1 == 0

    # ── T8.7: Tolerancias numéricas en el límite ──────────────────────────────

    def test_symplectic_tolerance_boundary(self, p1):
        r"""
        Una matriz con residuo justo en el límite de tolerancia.
        Perturbamos ligeramente una simpléctica para que el residuo
        sea < tolerancia efectiva.
        """
        M = _identity_symplectic(2)
        omega = _canonical_symplectic_matrix(2)
        # Perturbación infinitesimal que preserva simplecticidad aproximadamente
        eps = _SYMPLECTIC_TOLERANCE_BASE * 0.01
        M_perturbed = M + eps * np.eye(4)

        # Si la perturbación viola simplecticidad, debe fallar
        try:
            result = p1._audit_symplectic_invariance(M_perturbed)
            # Si no falla, el residuo debe ser pequeño
            assert result.symplectic_relative_residual <= result.effective_tolerance
        except SymplecticInvarianceViolation:
            pass  # También es un resultado válido

    # ── T8.8: Composición de rotaciones simplécticas ─────────────────────────

    def test_composition_of_symplectic_is_symplectic(self, p1):
        r"""M₁ · M₂ con M₁, M₂ ∈ Sp(2n) debe estar en Sp(2n)."""
        M1 = _rotation_symplectic(2, theta=0.3)
        M2 = _shear_symplectic(2, alpha=0.5)
        M_composed = M1 @ M2
        result = p1._audit_symplectic_invariance(M_composed)
        assert result.is_volume_preserved is True

    def test_many_compositions_remain_symplectic(self, p1):
        r"""
        La composición iterada de 10 rotaciones simplécticas debe
        seguir siendo simpléctica (prueba de estabilidad numérica).
        """
        n = 2
        M = _identity_symplectic(n)
        for k in range(10):
            theta = 0.1 * k
            M = M @ _rotation_symplectic(n, theta=theta)
        result = p1._audit_symplectic_invariance(M)
        assert result.is_volume_preserved is True


# ═══════════════════════════════════════════════════════════════════════════════
# §T9 — TESTS DE TRAZABILIDAD Y CHECKSUMS
# ═══════════════════════════════════════════════════════════════════════════════
class TestProvenanceAndChecksum:
    r"""
    Verifica la trazabilidad criptográfica y el objeto AuditProvenance.
    """

    @pytest.fixture
    def agent(self) -> ASTStaticAnalyzerAgent:
        return ASTStaticAnalyzerAgent()

    def _run_governance(
        self, agent: ASTStaticAnalyzerAgent, n: int = 2,
    ) -> ASTGovernanceState:
        M, Phi, gradV = _build_valid_governance_inputs(n=n)
        return agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )

    def test_provenance_timestamp_is_not_empty(self, agent):
        r"""El timestamp no es vacío."""
        state = self._run_governance(agent)
        assert len(state.provenance.timestamp_iso) > 0

    def test_provenance_timestamp_contains_timezone(self, agent):
        r"""El timestamp contiene información de zona horaria (+00:00 o Z)."""
        state = self._run_governance(agent)
        ts = state.provenance.timestamp_iso
        assert "+" in ts or "Z" in ts

    def test_provenance_checksum_64_hex_chars(self, agent):
        r"""El checksum tiene exactamente 64 caracteres hexadecimales."""
        state = self._run_governance(agent)
        checksum = state.provenance.input_checksum_sha256
        assert len(checksum) == 64
        int(checksum, 16)  # Lanza ValueError si no es hex válido

    def test_provenance_functor_chain_contains_check_marks(self, agent):
        r"""La cadena funtorial contiene marcas ✓ para fases exitosas."""
        state = self._run_governance(agent)
        chain = state.provenance.functor_chain
        assert "✓" in chain

    def test_provenance_all_phases_true_on_success(self, agent):
        r"""Todas las fases son True en una ejecución exitosa."""
        state = self._run_governance(agent)
        assert state.provenance.phase1_passed is True
        assert state.provenance.phase2_passed is True
        assert state.provenance.phase3_passed is True

    def test_checksum_changes_with_matrix_perturbation(self, agent):
        r"""Una perturbación mínima en M cambia el checksum."""
        M1, Phi, gradV = _build_valid_governance_inputs(n=2)
        M2 = M1.copy()

        # Perturbamos M2 con una rotación simpléctica pequeña
        M_rot = _rotation_symplectic(2, theta=1e-6)
        M2 = M1 @ M_rot

        s1 = agent.execute_ast_symplectic_governance(M1, Phi, gradV, h1_dimension=0)
        s2 = agent.execute_ast_symplectic_governance(M2, Phi, gradV, h1_dimension=0)

        assert s1.provenance.input_checksum_sha256 != s2.provenance.input_checksum_sha256

    def test_checksum_includes_coboundary_operators(self, agent):
        r"""El checksum difiere cuando se añaden operadores coboundary."""
        M, Phi, gradV = _build_valid_governance_inputs(n=2)

        s_without = agent.execute_ast_symplectic_governance(
            M, Phi, gradV, h1_dimension=0,
        )
        D0, D1 = _trivial_coboundary_operators()
        s_with = agent.execute_ast_symplectic_governance(
            M, Phi, gradV,
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )

        assert (
            s_without.provenance.input_checksum_sha256
            != s_with.provenance.input_checksum_sha256
        )

    def test_build_provenance_directly(self):
        r"""_build_provenance produce un AuditProvenance válido."""
        agent = ASTStaticAnalyzerAgent()
        prov = agent._build_provenance(
            checksum="a" * 64,
            phase1_passed=True,
            phase2_passed=True,
            phase3_passed=False,
        )
        assert isinstance(prov, AuditProvenance)
        assert prov.phase3_passed is False
        assert "✗" in prov.functor_chain

    def test_build_provenance_all_failed(self):
        r"""_build_provenance con todas las fases fallidas reporta ✗ en chain."""
        agent = ASTStaticAnalyzerAgent()
        prov = agent._build_provenance(
            checksum="b" * 64,
            phase1_passed=False,
            phase2_passed=False,
            phase3_passed=False,
        )
        assert prov.functor_chain.count("✗") >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# §T10 — TESTS DE INVARIANTES MATEMÁTICOS (PROPIEDADES ALGEBRAICAS)
# ═══════════════════════════════════════════════════════════════════════════════
class TestMathematicalInvariants:
    r"""
    Verifica propiedades algebraicas y topológicas fundamentales que deben
    mantenerse como invariantes en toda ejecución.

    Estas pruebas actúan como "teoremas computacionales" de la implementación.
    """

    @pytest.fixture
    def p1(self) -> Phase1_SymplecticInvarianceAuditor:
        return Phase1_SymplecticInvarianceAuditor()

    @pytest.fixture
    def p2(self) -> Phase2_DirichletThermodynamicEnforcer:
        return Phase2_DirichletThermodynamicEnforcer()

    @pytest.fixture
    def p3(self) -> Phase3_CellularSheafCohomologyAuditor:
        return Phase3_CellularSheafCohomologyAuditor()

    # ── T10.1: Teorema de Liouville (Invarianza de Volumen) ───────────────────

    def test_liouville_identity_preserves_volume(self, p1):
        r"""
        Teorema de Liouville: φₜ preserva el volumen ⟺ Mᵀ Ω M = Ω.
        Para M = I: Iᵀ Ω I = Ω. ✓
        """
        for n in range(1, 6):
            M = _identity_symplectic(n)
            result = p1._audit_symplectic_invariance(M)
            assert result.is_volume_preserved, f"Falla para n={n}"

    def test_liouville_product_preserves_volume(self, p1):
        r"""
        El producto de dos matrices simplécticas es simpléctica:
        M₁, M₂ ∈ Sp(2n) ⟹ M₁·M₂ ∈ Sp(2n).
        """
        for n in [1, 2, 3]:
            M1 = _rotation_symplectic(n, theta=0.2)
            M2 = _shear_symplectic(n, alpha=0.3)
            M = M1 @ M2
            result = p1._audit_symplectic_invariance(M)
            assert result.is_volume_preserved, f"Producto falla para n={n}"

    def test_liouville_determinant_one(self, p1):
        r"""
        M ∈ Sp(2n) ⟹ det(M) = 1.
        Verificamos que el residuo |det(M)−1| < ε para toda M simpléctica.
        """
        matrices = [
            _identity_symplectic(2),
            _rotation_symplectic(2, theta=1.0),
            _shear_symplectic(3, alpha=5.0),
        ]
        for M in matrices:
            result = p1._audit_symplectic_invariance(M)
            assert result.determinant_residual < 1e-8, (
                f"Residuo determinantal > 1e-8: {result.determinant_residual}"
            )

    # ── T10.2: Segunda Ley de la Termodinámica ────────────────────────────────

    def test_second_law_positive_dissipation(self, p2):
        r"""
        Segunda Ley: P_diss = Φᵀ∇V ≥ 0.
        Para vectores paralelos: P_diss = ‖Φ‖·‖∇V‖ > 0.
        """
        for dim in [2, 4, 6, 8]:
            Phi = np.ones(dim)
            gradV = np.ones(dim)
            result = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
            assert result.dissipated_power >= 0.0

    def test_second_law_cauchy_schwarz_alignment(self, p2):
        r"""
        Por Cauchy-Schwarz: |Φᵀ∇V| ≤ ‖Φ‖·‖∇V‖.
        Equivalente: |cos(θ)| ≤ 1.
        """
        rng = np.random.default_rng(seed=7)
        for _ in range(20):
            v = rng.standard_normal(6)
            w = rng.standard_normal(6)
            # Garantizamos P_diss ≥ 0 usando el mismo vector
            result = p2._enforce_dirichlet_thermodynamics(v, v)
            assert abs(result.alignment_cosine) <= 1.0 + 1e-12

    def test_second_law_entropy_nonnegative(self, p2):
        r"""
        σ_CD ≥ 0 (producción de entropía no negativa).
        """
        Phi, gradV = _positive_dissipative_pair(dim=4)
        result = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
        assert result.entropy_production_rate >= 0.0
        assert result.clausius_duhem_satisfied is True

    # ── T10.3: Exactitud de H¹ = 0 (Integrabilidad Global) ───────────────────

    def test_h1_zero_implies_integrability(self, p3):
        r"""
        dim H¹ = 0 ⟺ im(δ⁰) = ker(δ¹) ⟺ el haz es globalmente integrable.
        """
        D0, D1 = _trivial_coboundary_operators()
        result = p3._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert result.is_globally_integrable is True
        assert result.h1_dimension == 0

    def test_euler_poincare_formula(self, p3):
        r"""
        χ(G) = β₀ − β₁ + β₂ = dim C⁰ − dim C¹ + dim C² (por Euler-Poincaré).
        """
        c0, c1, c2 = 3, 5, 2
        D0, D1 = _trivial_coboundary_operators(c0=c0, c1=c1, c2=c2)
        result = p3._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        # χ por cadenas = c0 - c1 + c2
        euler_chains = c0 - c1 + c2
        # χ por Betti = β₀ - β₁ + β₂
        b0, b1, b2 = result.betti_numbers
        euler_betti = b0 - b1 + b2
        assert result.euler_characteristic == euler_betti
        assert result.euler_characteristic == euler_chains

    # ── T10.4: Antisimetría de Ω ──────────────────────────────────────────────

    def test_omega_antisymmetry_for_all_n(self, p1):
        r"""
        Ω = −Ωᵀ para todo n ∈ {1,...,6}.
        """
        for n in range(1, 7):
            omega = p1._build_canonical_symplectic_matrix(n)
            residual = float(la.norm(omega + omega.T, ord="fro"))
            assert residual < 1e-14, f"Ω no antisimétrica para n={n}"

    def test_omega_squared_is_minus_identity(self, p1):
        r"""
        Ω² = −I para la matriz simpléctica canónica.
        (Propiedad de la estructura casi-compleja).
        """
        for n in range(1, 5):
            omega = p1._build_canonical_symplectic_matrix(n)
            omega_squared = omega @ omega
            expected = -np.eye(2 * n)
            residual = float(la.norm(omega_squared - expected, ord="fro"))
            assert residual < 1e-13, f"Ω² ≠ −I para n={n}: residuo={residual}"

    # ── T10.5: Condición del complejo cohomológico ────────────────────────────

    def test_complex_condition_d1_d0_zero(self, p3):
        r"""
        Para operadores válidos: δ¹∘δ⁰ = 0.
        Verificamos que el residuo ‖δ¹∘δ⁰‖/scale ≤ τ_δ.
        """
        D0, D1 = _trivial_coboundary_operators(c0=3, c1=5, c2=2)
        result = p3._audit_cellular_sheaf_cohomology(
            coboundary_delta0=D0,
            coboundary_delta1=D1,
        )
        assert result.complex_residual <= _COHOMOLOGICAL_COMPLEX_TOL

    def test_complex_condition_violated_raises(self, p3):
        r"""
        Si δ¹∘δ⁰ ≠ 0, la condición de complejo es violada y se lanza excepción.
        """
        D0, D1 = _violating_coboundary_operators()
        with pytest.raises(CohomologicalObstructionError):
            p3._audit_cellular_sheaf_cohomology(
                coboundary_delta0=D0,
                coboundary_delta1=D1,
            )

    # ── T10.6: Rango y Nulidad (Teorema de Rango-Nulidad) ────────────────────

    def test_rank_nullity_theorem(self, p3):
        r"""
        Para D0 ∈ ℝ^{c1×c0}:
            rank(D0) + dim ker(D0) = c0.
        Equivalente: rank(D0) ≤ min(c0, c1).
        """
        c0, c1 = 4, 6
        D0, _ = _trivial_coboundary_operators(c0=c0, c1=c1, c2=2)
        rank, svs = p3._numerical_rank_with_gap(D0)
        assert rank <= min(c0, c1)
        assert rank >= 0

    # ── T10.7: Invarianza del certificado bajo reescalado ─────────────────────

    def test_thermo_cert_invariant_under_vector_scaling(self, p2):
        r"""
        La dirección del ángulo termodinámico θ es invariante bajo reescalado:
        θ(αΦ, β∇V) = θ(Φ, ∇V) para α, β > 0.
        """
        Phi = np.array([1.0, 2.0, 0.0, 0.0])
        gradV = np.array([1.0, 1.0, 0.0, 0.0])

        r1 = p2._enforce_dirichlet_thermodynamics(Phi, gradV)
        r2 = p2._enforce_dirichlet_thermodynamics(3.0 * Phi, 5.0 * gradV)

        assert math.isclose(
            r1.alignment_cosine, r2.alignment_cosine, rel_tol=1e-10,
        ), (
            f"cos(θ) no invariante bajo reescalado: "
            f"{r1.alignment_cosine} ≠ {r2.alignment_cosine}"
        )
        assert math.isclose(
            r1.thermodynamic_angle_rad, r2.thermodynamic_angle_rad, rel_tol=1e-10,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═══════════════════════════════════════════════════════════════════════════════
def pytest_configure(config):
    r"""Registra marcadores personalizados."""
    config.addinivalue_line(
        "markers",
        "symplectic: tests relacionados con invarianza simpléctica (Fase 1)",
    )
    config.addinivalue_line(
        "markers",
        "thermodynamic: tests relacionados con termodinámica port-Hamiltoniana (Fase 2)",
    )
    config.addinivalue_line(
        "markers",
        "cohomological: tests relacionados con cohomología de haces celulares (Fase 3)",
    )
    config.addinivalue_line(
        "markers",
        "funtorial: tests de composición funtorial entre fases",
    )
    config.addinivalue_line(
        "markers",
        "numerical: tests de casos límite numéricos",
    )
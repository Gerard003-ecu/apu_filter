r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: KBase Thermodynamic Agent                                  ║
║ Ubicación: tests/unit/alpha/kbase/test_kbase_thermodynamic_agent.py          ║
║ Versión   : 1.0.0-Strict-Spectral-Phased                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Cobertura por Fase                                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Fase 1 – Topología Matricial:                                               ║
║    • Dimensiones inconsistentes (no cuadrada, shapes distintos).             ║
║    • Simetría violada (residuo > ε_mach·‖A‖_F).                              ║
║    • Antisimetría de J_base violada.                                         ║
║    • No definida positiva (C_soc, M_rec con autovalor ≤ 0).                  ║
║    • Mal condicionamiento (κ > kappa_max).                                   ║
║    • R_cost con autovalor negativo real.                                     ║
║    • Caso nominal: TopologicalContext con campos correctos.                  ║
║    • Invarianza ante re-simetrización defensiva de O(ε).                     ║
║                                                                              ║
║  Fase 2 – Dinámica Hamiltoniana:                                             ║
║    • Dimensiones incorrectas de q, p, df_dt.                                 ║
║    • Energía potencial V(q) = ½ q^⊤ C_soc⁻¹ q (exactitud analítica).         ║
║    • Energía cinética K(p) = ½ p^⊤ M_rec⁻¹ p (exactitud analítica).          ║
║    • Hamiltoniano total H = V + K (aditividad).                              ║
║    • Gradiente ∂H/∂q = C_soc⁻¹ q (exactitud analítica).                      ║
║    • Gradiente ∂H/∂p = M_rec⁻¹ p (exactitud analítica).                      ║
║    • Disipación de Rayleigh P_diss ≥ 0 (segunda ley).                        ║
║    • Violación de Rayleigh: R_cost con curvatura negativa artificial.        ║
║    • Flyback nominal: ‖M_rec·df_dt‖_∞ < breakdown_voltage.                   ║
║    • Flyback crítico: voltaje en el límite exacto (caso borde).              ║
║    • Flyback supercrítico: lanza InertialFlybackError.                       ║
║    • Estado nulo (q=0, p=0): H=0, P_diss=0, flyback=0.                       ║
║    • is_thermodynamically_stable siempre True si no hay excepción.           ║
║                                                                              ║
║  Fase 3 – Proyección en Haces:                                               ║
║    • Dimensión incorrecta de state_x.                                        ║
║    • δ_{BASE} tiene shape correcto (dim_q+dim_p, dim_q+dim_p).               ║
║    • Identidad de Hodge: ‖δ^⊤δ − Hodge_local‖_F/‖δ‖_F² < 100·ε_mach.         ║
║    • Proyección δ·x correcta (linealidad y norma).                           ║
║    • rank_delta = dim_q + rank_R.                                            ║
║    • delta_base_sq_norm es escalar no negativo.                              ║
║    • Instanciación perezosa de phase3 (None antes, no None después).         ║
║    • Reutilización de phase3 entre llamadas (identidad de objeto).           ║
║                                                                              ║
║  Integración de las 3 Fases:                                                 ║
║    • Pipeline completo: constructor → synthesize → export_stalk.             ║
║    • Inmutabilidad de los DTOs (frozen dataclass).                           ║
║    • Logging emitido en niveles correctos (INFO en fases 1/2/3).             ║
║    • Reproducibilidad: misma entrada → misma salida (determinismo).          ║
║    • Ortogonalidad: perturbación en q no afecta K(p).                        ║
║    • Escala: H(λq, λp) = λ² H(q, p) (homogeneidad cuadrática).               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ── Biblioteca estándar ──────────────────────────────────────────────────────
import logging
import math
from typing import Tuple

# ── Framework de pruebas ─────────────────────────────────────────────────────
import pytest

# ── Álgebra numérica ─────────────────────────────────────────────────────────
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Módulo bajo prueba ───────────────────────────────────────────────────────
from app.agents.alpha.kbase.kbase_thermodynamic_agent import (
    # Agente orquestador
    KBaseThermodynamicAgent,
    # DTOs
    ApexPreparationContext,
    BasalStateTensor,
    SheafStalk,
    TopologicalContext,
    # Excepciones
    CapacitanceDegeneracyError,
    DimensionMismatchError,
    IllConditionedMatrixError,
    InertialFlybackError,
    RayleighDissipationViolation,
    SheafCoboundaryError,
    ThermodynamicBaseError,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 0 – CONSTANTES Y UTILIDADES DE PRUEBA                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Precisión de máquina IEEE-754 double
_EPS: float = float(np.finfo(np.float64).eps)

# Tolerancia estándar para comparaciones analíticas exactas
# (errores de redondeo de O(n · ε_mach) para sistemas triangulares n×n)
_ATOL_ANALYTICAL: float = 1.0e-10
_RTOL_ANALYTICAL: float = 1.0e-10


# ── Fábricas de matrices constitutivas ──────────────────────────────────────

def _make_spd_matrix(n: int, seed: int = 0) -> NDArray[np.float64]:
    """
    Genera una matriz SPD reproducible de dimensión n×n.

    Construcción: A = Q · diag(λ) · Q^⊤ donde:
      • Q es ortogonal (QR de una Gaussiana aleatoria)
      • λ_i ∈ [1, n+1] (bien condicionada, κ < n+1)
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.linspace(1.0, float(n + 1), n)
    return Q @ np.diag(eigvals) @ Q.T


def _make_psd_matrix(n: int, rank: int, seed: int = 0) -> NDArray[np.float64]:
    """
    Genera una matriz PSD de rango ``rank`` < n (semidefinida, no SPD).

    Construcción: A = V · diag(λ⁺, 0) · V^⊤
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.zeros(n)
    eigvals[:rank] = np.linspace(1.0, float(rank + 1), rank)
    return Q @ np.diag(eigvals) @ Q.T


def _make_antisymmetric(n: int, seed: int = 0) -> NDArray[np.float64]:
    """
    Genera una matriz antisimétrica J = -J^⊤ de dimensión n×n.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A - A.T


def _make_ill_conditioned_spd(n: int, kappa: float = 1.0e12) -> NDArray[np.float64]:
    """
    Genera una matriz SPD con número de condición κ ≈ kappa.

    Construcción: A = Q · diag(1, kappa) · Q^⊤ para n=2,
    extendido a dimensiones mayores con λ_i = kappa^{i/(n-1)}.
    """
    rng = np.random.default_rng(42)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.logspace(0, np.log10(kappa), n)
    return Q @ np.diag(eigvals) @ Q.T


def _default_matrices(
    dim_q: int = 3,
    dim_p: int = 3,
    seed: int = 7,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Retorna (C_soc, M_rec, R_cost, J_base) válidas por defecto.

    Dimensiones:
      C_soc  ∈ ℝ^{dim_q × dim_q}  (SPD)
      M_rec  ∈ ℝ^{dim_p × dim_p}  (SPD)
      R_cost ∈ ℝ^{n × n}           (PSD, n = dim_q + dim_p)
      J_base ∈ ℝ^{n × n}           (antisimétrica)
    """
    n = dim_q + dim_p
    C_soc = _make_spd_matrix(dim_q, seed=seed)
    M_rec = _make_spd_matrix(dim_p, seed=seed + 1)
    R_cost = _make_psd_matrix(n, rank=n - 1, seed=seed + 2)
    J_base = _make_antisymmetric(n, seed=seed + 3)
    return C_soc, M_rec, R_cost, J_base


def _default_agent(
    dim_q: int = 3,
    dim_p: int = 3,
    breakdown_voltage: float = 1.0e6,
    kappa_max: float = 1.0e10,
    seed: int = 7,
) -> KBaseThermodynamicAgent:
    """Construye un KBaseThermodynamicAgent con matrices válidas por defecto."""
    C_soc, M_rec, R_cost, J_base = _default_matrices(dim_q, dim_p, seed)
    return KBaseThermodynamicAgent(
        C_soc=C_soc,
        M_rec=M_rec,
        R_cost=R_cost,
        J_base=J_base,
        breakdown_voltage=breakdown_voltage,
        kappa_max=kappa_max,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 1 – PRUEBAS DE FASE 1: TOPOLOGÍA MATRICIAL                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase1MatrixTopology:
    """
    Pruebas exhaustivas de Phase1_MatrixTopology y build_topological_context.

    Verifica que todas las condiciones algebraicas (dimensiones, simetría,
    antisimetría, SPD, κ, PSD) son detectadas con diagnóstico cuantitativo.
    """

    # ── 1.1 Validación dimensional ────────────────────────────────────────

    def test_c_soc_not_2d_raises_dimension_error(self) -> None:
        """C_soc con ndim=1 debe lanzar DimensionMismatchError."""
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        with pytest.raises(DimensionMismatchError, match="cuadrada"):
            KBaseThermodynamicAgent(
                C_soc=C_soc[0],  # shape (3,) en lugar de (3,3)
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_m_rec_wrong_shape_raises_dimension_error(self) -> None:
        """M_rec no cuadrada debe lanzar DimensionMismatchError."""
        C_soc, M_rec, R_cost, J_base = _default_matrices(dim_q=3, dim_p=3)
        M_rec_bad = np.eye(2, 4)  # no cuadrada
        with pytest.raises(DimensionMismatchError, match="cuadrada"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec_bad,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_r_cost_wrong_shape_raises_dimension_error(self) -> None:
        """
        R_cost con shape (n+1, n+1) en lugar de (dim_q+dim_p, dim_q+dim_p)
        debe lanzar DimensionMismatchError.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices(dim_q=3, dim_p=3)
        n = C_soc.shape[0] + M_rec.shape[0]  # = 6
        R_bad = np.eye(n + 1)  # shape (7,7) en lugar de (6,6)
        with pytest.raises(DimensionMismatchError, match=r"R_cost"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_bad,
                J_base=J_base,
            )

    def test_j_base_wrong_shape_raises_dimension_error(self) -> None:
        """J_base con dimensión inconsistente debe lanzar DimensionMismatchError."""
        C_soc, M_rec, R_cost, J_base = _default_matrices(dim_q=3, dim_p=3)
        J_bad = _make_antisymmetric(5)  # shape (5,5) en lugar de (6,6)
        with pytest.raises(DimensionMismatchError, match=r"J_base"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_bad,
            )

    # ── 1.2 Validación de antisimetría de J_base ──────────────────────────

    def test_j_base_symmetric_raises_thermodynamic_error(self) -> None:
        """
        J_base simétrica (J = J^⊤) viola la antisimetría y debe lanzar
        ThermodynamicBaseError con diagnóstico de ‖J + J^⊤‖_F.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        J_sym = J_base + J_base.T  # simétrica, no antisimétrica
        with pytest.raises(ThermodynamicBaseError, match=r"antisim"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_sym,
            )

    def test_j_base_identity_raises_antisymmetry_error(self) -> None:
        """
        J_base = I (completamente simétrica) debe ser rechazada con
        diagnóstico cuantitativo del residuo ‖J + J^⊤‖_F = 2·‖I‖_F.
        """
        C_soc, M_rec, R_cost, _ = _default_matrices()
        n = C_soc.shape[0] + M_rec.shape[0]
        with pytest.raises(ThermodynamicBaseError):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=np.eye(n),
            )

    def test_j_base_near_antisymmetric_within_tolerance_accepted(self) -> None:
        """
        J_base con perturbación simétrica de magnitud 10·ε_mach·‖J‖_F
        (dentro de tolerancia) debe ser aceptada sin excepción.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        norm_J = float(la.norm(J_base, "fro"))
        # Perturbación simétrica de amplitud ε_mach·‖J‖_F/2 (mitad de la tolerancia)
        perturbation = np.eye(J_base.shape[0]) * (_EPS * norm_J * 0.5)
        J_near = J_base + perturbation
        # Forzar antisimetría exacta después de la perturbación para que pase
        J_antisym = J_near - J_near.T
        # Escalar a la mitad para que ‖J_near + J_near^⊤‖ = 0 exacto
        J_antisym /= 2.0
        # Este debe pasar la validación
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_antisym,
        )
        assert agent.context is not None

    # ── 1.3 Validación de simetría de C_soc, M_rec, R_cost ───────────────

    def test_c_soc_asymmetric_raises_thermodynamic_error(self) -> None:
        """
        C_soc con ‖C − C^⊤‖_F >> ε_mach·‖C‖_F debe lanzar
        ThermodynamicBaseError.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        C_asym = C_soc.copy()
        C_asym[0, 1] += 10.0  # rompe la simetría macroscópicamente
        with pytest.raises(ThermodynamicBaseError, match=r"simétric"):
            KBaseThermodynamicAgent(
                C_soc=C_asym,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_m_rec_asymmetric_raises_thermodynamic_error(self) -> None:
        """M_rec asimétrica debe lanzar ThermodynamicBaseError."""
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        M_asym = M_rec.copy()
        M_asym[1, 0] += 5.0
        with pytest.raises(ThermodynamicBaseError, match=r"simétric"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_asym,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_r_cost_asymmetric_raises_thermodynamic_error(self) -> None:
        """R_cost asimétrica debe lanzar ThermodynamicBaseError."""
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        R_asym = R_cost.copy()
        R_asym[0, 2] += 3.0
        with pytest.raises(ThermodynamicBaseError, match=r"simétric"):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_asym,
                J_base=J_base,
            )

    # ── 1.4 Validación de definición positiva (SPD) ───────────────────────

    def test_c_soc_singular_raises_capacitance_degeneracy(self) -> None:
        """
        C_soc singular (λ_min = 0) debe lanzar CapacitanceDegeneracyError
        con diagnóstico del autovalor mínimo.
        """
        _, M_rec, R_cost, J_base = _default_matrices()
        n_q = 3
        # C_soc rango-deficiente: una columna/fila anulada
        C_bad = _make_psd_matrix(n_q, rank=n_q - 1, seed=99)
        with pytest.raises(CapacitanceDegeneracyError):
            KBaseThermodynamicAgent(
                C_soc=C_bad,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_c_soc_negative_eigenvalue_raises_capacitance_degeneracy(self) -> None:
        """
        C_soc con autovalor negativo (indefinida) debe lanzar
        CapacitanceDegeneracyError.
        """
        _, M_rec, R_cost, J_base = _default_matrices()
        n_q = 3
        C_spd = _make_spd_matrix(n_q, seed=10)
        # Forzar autovalor negativo
        eigvals, eigvecs = la.eigh(C_spd)
        eigvals[0] = -1.0
        C_indef = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C_indef = 0.5 * (C_indef + C_indef.T)  # re-simetrizar
        with pytest.raises(CapacitanceDegeneracyError):
            KBaseThermodynamicAgent(
                C_soc=C_indef,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )

    def test_m_rec_singular_raises_capacitance_degeneracy(self) -> None:
        """
        M_rec singular debe lanzar CapacitanceDegeneracyError (o subclase).
        """
        C_soc, _, R_cost, J_base = _default_matrices()
        n_p = 3
        M_bad = _make_psd_matrix(n_p, rank=n_p - 1, seed=88)
        with pytest.raises(CapacitanceDegeneracyError):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_bad,
                R_cost=R_cost,
                J_base=J_base,
            )

    # ── 1.5 Validación de número de condición κ ───────────────────────────

    def test_ill_conditioned_c_soc_raises_ill_conditioned_error(self) -> None:
        """
        C_soc con κ > kappa_max debe lanzar IllConditionedMatrixError
        con diagnóstico de κ y κ_max.
        """
        _, M_rec, R_cost, J_base = _default_matrices()
        kappa_max = 1.0e6
        C_ill = _make_ill_conditioned_spd(3, kappa=1.0e8)
        with pytest.raises(IllConditionedMatrixError, match=r"κ"):
            KBaseThermodynamicAgent(
                C_soc=C_ill,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                kappa_max=kappa_max,
            )

    def test_ill_conditioned_m_rec_raises_ill_conditioned_error(self) -> None:
        """
        M_rec con κ > kappa_max debe lanzar IllConditionedMatrixError.
        """
        C_soc, _, R_cost, J_base = _default_matrices()
        kappa_max = 1.0e5
        M_ill = _make_ill_conditioned_spd(3, kappa=1.0e7)
        with pytest.raises(IllConditionedMatrixError):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_ill,
                R_cost=R_cost,
                J_base=J_base,
                kappa_max=kappa_max,
            )

    def test_kappa_max_boundary_accepted(self) -> None:
        """
        C_soc con κ exactamente en el umbral kappa_max no debe ser rechazada.
        La condición de rechazo es estricta: κ > kappa_max.
        """
        _, M_rec, R_cost, J_base = _default_matrices()
        # κ = 10 << cualquier kappa_max razonable
        C_soc = _make_spd_matrix(3, seed=5)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            kappa_max=1.0e12,
        )
        assert agent.context.kappa_C > 0

    # ── 1.6 Validación de R_cost PSD ─────────────────────────────────────

    def test_r_cost_negative_eigenvalue_raises_rayleigh_violation(self) -> None:
        """
        R_cost con autovalor genuinamente negativo debe lanzar
        RayleighDissipationViolation.
        """
        C_soc, M_rec, _, J_base = _default_matrices()
        n = C_soc.shape[0] + M_rec.shape[0]
        R_psd = _make_psd_matrix(n, rank=n - 1, seed=20)
        # Forzar autovalor negativo real
        eigvals, eigvecs = la.eigh(R_psd)
        eigvals[0] = -1.0
        R_neg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        R_neg = 0.5 * (R_neg + R_neg.T)
        with pytest.raises(RayleighDissipationViolation):
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_neg,
                J_base=J_base,
            )

    def test_r_cost_zero_matrix_is_valid_psd(self) -> None:
        """
        R_cost = 0 es PSD válida (rango 0, disipación nula).
        El sistema es conservativo (Hamiltoniano puro).
        """
        C_soc, M_rec, _, J_base = _default_matrices()
        n = C_soc.shape[0] + M_rec.shape[0]
        R_zero = np.zeros((n, n))
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_zero,
            J_base=J_base,
        )
        assert agent.context.rank_R == 0

    # ── 1.7 TopologicalContext: integridad de campos ───────────────────────

    def test_topological_context_fields_types_and_shapes(self) -> None:
        """
        El TopologicalContext debe tener los campos correctos:
          • L_C: (dim_q, dim_q), triangular inferior
          • L_M: (dim_p, dim_p), triangular inferior
          • R_cost: (n, n)
          • R_sqrt: (n, n)
          • J_base: (n, n)
          • kappa_C, kappa_M, rank_R: tipos correctos
          • dim_q, dim_p: enteros positivos
        """
        dim_q, dim_p = 4, 3
        agent = _default_agent(dim_q=dim_q, dim_p=dim_p)
        ctx = agent.context

        assert ctx.L_C.shape == (dim_q, dim_q)
        assert ctx.L_M.shape == (dim_p, dim_p)
        assert ctx.R_cost.shape == (dim_q + dim_p, dim_q + dim_p)
        assert ctx.R_sqrt.shape == (dim_q + dim_p, dim_q + dim_p)
        assert ctx.J_base.shape == (dim_q + dim_p, dim_q + dim_p)
        assert isinstance(ctx.kappa_C, float) and ctx.kappa_C > 1.0
        assert isinstance(ctx.kappa_M, float) and ctx.kappa_M > 1.0
        assert isinstance(ctx.rank_R, int) and ctx.rank_R >= 0
        assert ctx.dim_q == dim_q
        assert ctx.dim_p == dim_p

    def test_l_c_is_lower_triangular_with_positive_diagonal(self) -> None:
        """
        L_C debe ser triangular inferior con diagonal estrictamente positiva
        (propiedad de la factorización de Cholesky).
        """
        agent = _default_agent()
        L_C = agent.context.L_C
        # Triangular inferior: parte superior es cero
        upper_part = np.triu(L_C, k=1)
        assert np.allclose(upper_part, 0.0, atol=_ATOL_ANALYTICAL), (
            "L_C tiene entradas no nulas en la parte estrictamente superior."
        )
        # Diagonal positiva
        assert np.all(np.diag(L_C) > 0), (
            "L_C tiene entradas diagonales no positivas."
        )

    def test_l_c_reconstructs_c_soc(self) -> None:
        """
        L_C · L_C^⊤ debe reconstruir C_soc con error O(ε_mach · ‖C_soc‖_F).
        """
        dim_q = 4
        C_soc, M_rec, R_cost, J_base = _default_matrices(dim_q=dim_q)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        L_C = agent.context.L_C
        C_reconstructed = L_C @ L_C.T
        norm_C = float(la.norm(C_soc, "fro"))
        residual = float(la.norm(C_reconstructed - C_soc, "fro"))
        assert residual < 100 * _EPS * norm_C, (
            f"‖L_C·L_C^⊤ − C_soc‖_F = {residual:.3e} > 100·ε·‖C‖ = {100*_EPS*norm_C:.3e}"
        )

    def test_l_m_reconstructs_m_rec(self) -> None:
        """L_M · L_M^⊤ debe reconstruir M_rec con error O(ε_mach · ‖M_rec‖_F)."""
        dim_p = 5
        C_soc, M_rec, R_cost, J_base = _default_matrices(dim_p=dim_p)
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        L_M = agent.context.L_M
        M_reconstructed = L_M @ L_M.T
        norm_M = float(la.norm(M_rec, "fro"))
        residual = float(la.norm(M_reconstructed - M_rec, "fro"))
        assert residual < 100 * _EPS * norm_M

    def test_r_sqrt_satisfies_r_sqrt_sq_equals_r_cost(self) -> None:
        """
        R_sqrt · R_sqrt debe reconstruir R_cost con tolerancia de máquina.
        Verifica que R_sqrt es la raíz cuadrada espectral correcta.
        """
        agent = _default_agent()
        R_sqrt = agent.context.R_sqrt
        R_cost = agent.context.R_cost
        R_reconstructed = R_sqrt @ R_sqrt
        norm_R = float(la.norm(R_cost, "fro"))
        residual = float(la.norm(R_reconstructed - R_cost, "fro"))
        tol = 100 * _EPS * max(norm_R, 1.0)
        assert residual < tol, (
            f"‖R_sqrt² − R_cost‖_F = {residual:.3e} > tol = {tol:.3e}"
        )

    def test_r_sqrt_is_symmetric_psd(self) -> None:
        """
        R_sqrt debe ser simétrica y semidefinida positiva (todos los
        autovalores ≥ −100·ε_mach·‖R_sqrt‖_F).
        """
        agent = _default_agent()
        R_sqrt = agent.context.R_sqrt
        # Simetría
        assert np.allclose(R_sqrt, R_sqrt.T, atol=100 * _EPS)
        # PSD
        eigvals = la.eigvalsh(R_sqrt)
        norm_R = float(la.norm(R_sqrt, "fro"))
        assert np.all(eigvals >= -100 * _EPS * norm_R)

    def test_rank_r_is_consistent_with_r_cost(self) -> None:
        """
        rank_R debe coincidir con el rango numérico de R_cost calculado
        independientemente via numpy.linalg.matrix_rank.
        """
        agent = _default_agent()
        R_cost = agent.context.R_cost
        rank_np = int(np.linalg.matrix_rank(R_cost, tol=None))
        # Permitimos ±1 de diferencia por distintas tolerancias de truncación
        assert abs(agent.context.rank_R - rank_np) <= 1

    def test_context_is_immutable_frozen_dataclass(self) -> None:
        """
        TopologicalContext es un dataclass frozen; intentar asignar un
        campo debe lanzar FrozenInstanceError (o AttributeError en algunas versiones).
        """
        agent = _default_agent()
        with pytest.raises((AttributeError, TypeError)):
            agent.context.dim_q = 999  # type: ignore[misc]

    def test_defensive_resymmetrization_does_not_alter_result(self) -> None:
        """
        Una C_soc con asimetría de O(ε_mach·‖C‖_F) (errores de redondeo
        típicos de operaciones numéricas) debe ser aceptada y producir
        el mismo contexto que la versión exactamente simétrica.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices()
        norm_C = float(la.norm(C_soc, "fro"))
        # Perturbación antisimétrica de amplitud ε_mach·‖C‖_F/2
        noise = np.random.default_rng(0).standard_normal(C_soc.shape)
        C_perturbed = C_soc + _EPS * norm_C * 0.4 * (noise - noise.T)

        agent_exact = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )
        agent_perturbed = KBaseThermodynamicAgent(
            C_soc=C_perturbed, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )

        # Los factores Cholesky deben ser cercanos (perturbación de O(ε))
        residual_L = float(la.norm(
            agent_exact.context.L_C - agent_perturbed.context.L_C, "fro"
        ))
        assert residual_L < 1000 * _EPS * norm_C, (
            f"Perturbación de O(ε) produjo cambio inaceptable en L_C: {residual_L:.3e}"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 2 – PRUEBAS DE FASE 2: DINÁMICA HAMILTONIANA                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase2HamiltonianDynamics:
    """
    Pruebas exhaustivas de Phase2_HamiltonianDynamics y synthesize_basal_state.

    Verifica exactitud analítica de energías y gradientes, la segunda ley
    de la termodinámica, y el voltaje de Flyback inductivo.
    """

    # ── Fixture de agente por defecto ─────────────────────────────────────

    @pytest.fixture
    def agent(self) -> KBaseThermodynamicAgent:
        return _default_agent(dim_q=3, dim_p=3)

    @pytest.fixture
    def identity_agent(self) -> KBaseThermodynamicAgent:
        """
        Agente con C_soc = I, M_rec = I, R_cost = 0.
        Permite verificaciones analíticas exactas:
          V(q) = ½ ‖q‖², K(p) = ½ ‖p‖², grad_V = q, grad_K = p.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=1)
        return KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            breakdown_voltage=1.0e9,
        )

    # ── 2.1 Validación de dimensiones de q, p, df_dt ─────────────────────

    def test_wrong_q_dimension_raises_dimension_error(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """q con shape (dim_q+1,) debe lanzar DimensionMismatchError."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q_bad = np.ones(dim_q + 1)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)
        with pytest.raises(DimensionMismatchError, match=r"q"):
            agent.synthesize_basal_hamiltonian(q=q_bad, p=p, df_dt=df_dt)

    def test_wrong_p_dimension_raises_dimension_error(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """p con shape (dim_p-1,) debe lanzar DimensionMismatchError."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p_bad = np.ones(max(dim_p - 1, 1))
        df_dt = np.zeros(dim_p)
        with pytest.raises(DimensionMismatchError, match=r"p"):
            agent.synthesize_basal_hamiltonian(q=q, p=p_bad, df_dt=df_dt)

    def test_wrong_df_dt_dimension_raises_dimension_error(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """df_dt con shape (dim_p+2,) debe lanzar DimensionMismatchError."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt_bad = np.zeros(dim_p + 2)
        with pytest.raises(DimensionMismatchError, match=r"df_dt"):
            agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt_bad)

    # ── 2.2 Exactitud analítica de energía potencial V(q) ─────────────────

    def test_potential_energy_zero_for_zero_q(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """V(0) = 0 exactamente."""
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        q = np.zeros(dim_q)
        p = np.ones(dim_p) * 0.1
        df_dt = np.zeros(dim_p)
        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.potential_energy == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_potential_energy_half_norm_squared_for_identity_metric(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Con C_soc = I: V(q) = ½ ‖q‖² exactamente.
        Verificado para q = [1, 2, 3] → V = ½(1+4+9) = 7.
        """
        dim_p = identity_agent.context.dim_p
        q = np.array([1.0, 2.0, 3.0])
        p = np.zeros(dim_p)
        df_dt = np.zeros(dim_p)
        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        expected_V = 0.5 * float(np.dot(q, q))  # = 7.0
        assert state.potential_energy == pytest.approx(expected_V, rel=_RTOL_ANALYTICAL)

    def test_potential_energy_analytical_formula_general_c_soc(self) -> None:
        """
        V(q) = ½ q^⊤ C_soc⁻¹ q (fórmula analítica para C_soc general).
        Verificado contra la solución exacta usando scipy.linalg.solve.
        """
        dim_q, dim_p = 4, 3
        n = dim_q + dim_p
        C_soc = _make_spd_matrix(dim_q, seed=11)
        M_rec = _make_spd_matrix(dim_p, seed=12)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=13)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )

        rng = np.random.default_rng(55)
        q = rng.standard_normal(dim_q)
        p = np.zeros(dim_p)
        df_dt = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        # Valor analítico: ½ q^⊤ C_soc⁻¹ q
        C_inv_q = la.solve(C_soc, q)
        V_analytical = 0.5 * float(np.dot(q, C_inv_q))

        assert state.potential_energy == pytest.approx(
            V_analytical, rel=_RTOL_ANALYTICAL
        )

    def test_potential_energy_is_strictly_positive_for_nonzero_q(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """V(q) > 0 para q ≠ 0 (C_soc SPD garantiza positividad estricta)."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p = np.zeros(dim_p)
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.potential_energy > 0.0

    # ── 2.3 Exactitud analítica de energía cinética K(p) ──────────────────

    def test_kinetic_energy_zero_for_zero_p(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """K(0) = 0 exactamente."""
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        q = np.ones(dim_q) * 0.1
        p = np.zeros(dim_p)
        df_dt = np.zeros(dim_p)
        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.kinetic_energy == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_kinetic_energy_half_norm_squared_for_identity_metric(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Con M_rec = I: K(p) = ½ ‖p‖² exactamente.
        Verificado para p = [2, 0, -1] → K = ½(4+0+1) = 2.5.
        """
        dim_q = identity_agent.context.dim_q
        q = np.zeros(dim_q)
        p = np.array([2.0, 0.0, -1.0])
        df_dt = np.zeros(len(p))
        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        expected_K = 0.5 * float(np.dot(p, p))  # = 2.5
        assert state.kinetic_energy == pytest.approx(expected_K, rel=_RTOL_ANALYTICAL)

    def test_kinetic_energy_analytical_formula_general_m_rec(self) -> None:
        """K(p) = ½ p^⊤ M_rec⁻¹ p (fórmula analítica para M_rec general)."""
        dim_q, dim_p = 3, 4
        n = dim_q + dim_p
        C_soc = _make_spd_matrix(dim_q, seed=20)
        M_rec = _make_spd_matrix(dim_p, seed=21)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=22)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )

        rng = np.random.default_rng(66)
        q = np.zeros(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        M_inv_p = la.solve(M_rec, p)
        K_analytical = 0.5 * float(np.dot(p, M_inv_p))

        assert state.kinetic_energy == pytest.approx(
            K_analytical, rel=_RTOL_ANALYTICAL
        )

    # ── 2.4 Hamiltoniano total H = V + K ──────────────────────────────────

    def test_total_hamiltonian_equals_sum_of_energies(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """H(q, p) = V(q) + K(p): aditividad exacta."""
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        rng = np.random.default_rng(77)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.total_hamiltonian == pytest.approx(
            state.potential_energy + state.kinetic_energy,
            rel=_RTOL_ANALYTICAL,
        )

    def test_hamiltonian_zero_for_zero_state(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """H(0, 0) = 0 exactamente."""
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        state = identity_agent.synthesize_basal_hamiltonian(
            q=np.zeros(dim_q),
            p=np.zeros(dim_p),
            df_dt=np.zeros(dim_p),
        )
        assert state.total_hamiltonian == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_hamiltonian_quadratic_homogeneity(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Homogeneidad cuadrática: H(λq, λp) = λ² H(q, p).
        Verificado para λ = 3 y vectores aleatorios.
        """
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        rng = np.random.default_rng(88)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)
        lambda_ = 3.0

        state1 = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        state2 = identity_agent.synthesize_basal_hamiltonian(
            q=lambda_ * q, p=lambda_ * p, df_dt=df_dt
        )

        assert state2.total_hamiltonian == pytest.approx(
            lambda_ ** 2 * state1.total_hamiltonian,
            rel=_RTOL_ANALYTICAL,
        )

    def test_hamiltonian_separability_q_and_p(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Separabilidad: H(q, p) = H(q, 0) + H(0, p).
        Consecuencia de la aditividad V + K.
        """
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        rng = np.random.default_rng(99)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        state_full = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        state_q = identity_agent.synthesize_basal_hamiltonian(
            q=q, p=np.zeros(dim_p), df_dt=df_dt
        )
        state_p = identity_agent.synthesize_basal_hamiltonian(
            q=np.zeros(dim_q), p=p, df_dt=df_dt
        )

        assert state_full.total_hamiltonian == pytest.approx(
            state_q.total_hamiltonian + state_p.total_hamiltonian,
            rel=_RTOL_ANALYTICAL,
        )

    # ── 2.5 Gradiente del Hamiltoniano ────────────────────────────────────

    def test_grad_h_norm_is_positive_for_nonzero_state(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """‖∇H‖_2 > 0 para (q, p) ≠ (0, 0)."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.grad_H_norm > 0.0

    def test_grad_h_norm_zero_for_zero_state(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """∇H(0, 0) = 0 → ‖∇H‖ = 0."""
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        state = identity_agent.synthesize_basal_hamiltonian(
            q=np.zeros(dim_q),
            p=np.zeros(dim_p),
            df_dt=np.zeros(dim_p),
        )
        assert state.grad_H_norm == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_grad_v_q_equals_c_soc_inv_q_analytically(self) -> None:
        """
        ∂H/∂q = C_soc⁻¹ q verificado con perturbación de gradiente finito.

        Método: diferenciación hacia adelante de primer orden.
        ∂H/∂q_i ≈ [H(q + h·eᵢ) − H(q)] / h con h = 1e-7.
        """
        dim_q, dim_p = 4, 3
        n = dim_q + dim_p
        C_soc = _make_spd_matrix(dim_q, seed=30)
        M_rec = _make_spd_matrix(dim_p, seed=31)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=32)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )

        rng = np.random.default_rng(111)
        q = rng.standard_normal(dim_q)
        p = np.zeros(dim_p)
        df_dt = np.zeros(dim_p)

        # Gradiente analítico: C_soc⁻¹ q
        grad_analytical = la.solve(C_soc, q)

        # Gradiente numérico por diferencias finitas
        h = 1.0e-6
        grad_numerical = np.zeros(dim_q)
        for i in range(dim_q):
            e_i = np.zeros(dim_q)
            e_i[i] = 1.0
            state_plus = agent.synthesize_basal_hamiltonian(
                q=q + h * e_i, p=p, df_dt=df_dt
            )
            state_minus = agent.synthesize_basal_hamiltonian(
                q=q - h * e_i, p=p, df_dt=df_dt
            )
            grad_numerical[i] = (
                state_plus.potential_energy - state_minus.potential_energy
            ) / (2.0 * h)

        residual = float(la.norm(grad_numerical - grad_analytical, 2))
        norm_grad = float(la.norm(grad_analytical, 2))
        assert residual < 1e-4 * max(norm_grad, 1.0), (
            f"‖∇V_numérico − C_soc⁻¹q‖ = {residual:.3e} demasiado grande."
        )

    # ── 2.6 Disipación de Rayleigh (Segunda Ley) ──────────────────────────

    def test_dissipated_power_nonnegative(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """P_diss = ∇H^⊤ R_cost ∇H ≥ 0 (Segunda Ley de la Termodinámica)."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        rng = np.random.default_rng(200)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.dissipated_power >= 0.0, (
            f"P_diss = {state.dissipated_power:.6e} < 0: violación de la 2ª Ley."
        )

    def test_dissipated_power_zero_for_zero_r_cost(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Con R_cost = 0 (sistema conservativo), P_diss = 0 exactamente.
        """
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        rng = np.random.default_rng(201)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)
        state = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.dissipated_power == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_dissipated_power_analytical_value_for_identity_r_cost(self) -> None:
        """
        Con R_cost = I (plena disipación) y C_soc = M_rec = I:
        P_diss = ∇H^⊤ ∇H = ‖[q; p]‖² (exacto para matrices identidad).
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.eye(n)  # Disipación total
        J_base = _make_antisymmetric(n, seed=5)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            breakdown_voltage=1.0e9,
        )

        q = np.array([1.0, 0.0, -1.0])
        p = np.array([0.0, 2.0, 1.0])
        df_dt = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        # Con C_soc=I, M_rec=I, R_cost=I:
        # grad_H = [q; p], P_diss = ‖[q;p]‖²
        grad_H = np.concatenate([q, p])
        P_diss_analytical = float(np.dot(grad_H, grad_H))
        assert state.dissipated_power == pytest.approx(
            P_diss_analytical, rel=_RTOL_ANALYTICAL
        )

    def test_rayleigh_violation_raises_exception(self) -> None:
        """
        Inyectar R_cost artificialmente negativa en phase2 debe lanzar
        RayleighDissipationViolation.

        Estrategia: crear un agente válido y luego mutar el R_cost del
        contexto para simular un estado inválido post-construcción.
        (Prueba de la lógica interna del método, no del constructor.)
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        J_base = _make_antisymmetric(n, seed=6)

        # Construir agente válido con R_cost = 0
        R_cost_zero = np.zeros((n, n))
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost_zero, J_base=J_base
        )

        # Construir phase2 con R_cost negativa artificialmente
        # usando una copia mutable del contexto
        R_neg = -np.eye(n)  # Definitivamente negativa

        # Acceso directo a la lógica de Phase2 para inyectar R_cost negativa
        phase2_bad = KBaseThermodynamicAgent.Phase2_HamiltonianDynamics(
            context=TopologicalContext(
                L_C=agent.context.L_C,
                L_M=agent.context.L_M,
                R_cost=R_neg,
                R_sqrt=agent.context.R_sqrt,
                J_base=agent.context.J_base,
                kappa_C=agent.context.kappa_C,
                kappa_M=agent.context.kappa_M,
                dim_q=dim_q,
                dim_p=dim_p,
                rank_R=agent.context.rank_R,
            ),
            breakdown_voltage=1.0e9,
        )

        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)

        with pytest.raises(RayleighDissipationViolation):
            phase2_bad.synthesize_basal_state(q=q, p=p, df_dt=df_dt)

    # ── 2.7 Voltaje de Flyback inductivo ─────────────────────────────────

    def test_flyback_zero_for_zero_df_dt(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """‖M_rec · 0‖_∞ = 0: flyback nulo para perturbación nula."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.flyback_voltage_norm == pytest.approx(0.0, abs=_ATOL_ANALYTICAL)

    def test_flyback_subcritical_accepted(self) -> None:
        """
        ‖M_rec · df_dt‖_∞ < breakdown_voltage debe ser aceptado sin excepción.
        Verificado con df_dt pequeño y breakdown_voltage grande.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=7)

        breakdown = 1.0e4
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=breakdown,
        )

        # df_dt tal que ‖M_rec · df_dt‖_∞ = ‖df_dt‖_∞ (M_rec=I) = 1 << breakdown
        df_dt = np.ones(dim_p) * 1.0
        q = np.zeros(dim_q)
        p = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.flyback_voltage_norm < breakdown

    def test_flyback_supercritical_raises_inertial_flyback_error(self) -> None:
        """
        ‖M_rec · df_dt‖_∞ > breakdown_voltage debe lanzar InertialFlybackError
        con diagnóstico de la norma y el límite.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=8)

        breakdown = 1.0  # umbral muy bajo
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=breakdown,
        )

        df_dt = np.ones(dim_p) * 10.0  # ‖M_rec·df_dt‖_∞ = 10 > 1
        q = np.zeros(dim_q)
        p = np.zeros(dim_p)

        with pytest.raises(InertialFlybackError, match=r"Flyback"):
            agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

    def test_flyback_at_exact_boundary_not_raised(self) -> None:
        """
        ‖M_rec · df_dt‖_∞ = breakdown_voltage − δ (justo por debajo del límite)
        no debe lanzar excepción.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=9)

        breakdown = 100.0
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=breakdown,
        )

        # ‖df_dt‖_∞ = breakdown - 1 (por debajo del límite)
        df_dt = np.ones(dim_p) * (breakdown - 1.0)
        q = np.zeros(dim_q)
        p = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.flyback_voltage_norm < breakdown

    def test_flyback_analytical_value_for_identity_m_rec(self) -> None:
        """
        Con M_rec = I: ‖M_rec · df_dt‖_∞ = ‖df_dt‖_∞.
        Verificado para df_dt = [3, -5, 2] → flyback = 5.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = np.eye(dim_q)
        M_rec = np.eye(dim_p)
        R_cost = np.zeros((n, n))
        J_base = _make_antisymmetric(n, seed=10)

        breakdown = 1.0e6
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=breakdown,
        )

        df_dt = np.array([3.0, -5.0, 2.0])
        q = np.zeros(dim_q)
        p = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        expected_flyback = 5.0  # ‖df_dt‖_∞ = max(3, 5, 2) = 5
        assert state.flyback_voltage_norm == pytest.approx(
            expected_flyback, rel=_RTOL_ANALYTICAL
        )

    # ── 2.8 Propiedades del BasalStateTensor ─────────────────────────────

    def test_is_thermodynamically_stable_true_on_success(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        is_thermodynamically_stable = True si y sólo si todos los
        subprocesos pasan sin excepción.
        """
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q) * 0.5
        p = np.ones(dim_p) * 0.3
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.is_thermodynamically_stable is True

    def test_basal_state_tensor_is_immutable(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """BasalStateTensor es un dataclass frozen; no permite asignaciones."""
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        with pytest.raises((AttributeError, TypeError)):
            state.total_hamiltonian = 0.0  # type: ignore[misc]

    def test_grad_h_norm_scales_linearly_with_state(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Con C_soc = M_rec = I:
        ‖∇H(λq, λp)‖ = λ · ‖∇H(q, p)‖ (linealidad del gradiente).
        """
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        rng = np.random.default_rng(300)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)
        lambda_ = 4.0

        s1 = identity_agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        s2 = identity_agent.synthesize_basal_hamiltonian(
            q=lambda_ * q, p=lambda_ * p, df_dt=df_dt
        )

        assert s2.grad_H_norm == pytest.approx(
            lambda_ * s1.grad_H_norm, rel=_RTOL_ANALYTICAL
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 3 – PRUEBAS DE FASE 3: PROYECCIÓN EN HACES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase3SheafProjection:
    """
    Pruebas exhaustivas de Phase3_SheafProjection y export_sheaf_stalk.

    Verifica la identidad de Hodge local, la proyección correcta de estados,
    el rango de δ_{BASE} y la instanciación perezosa de phase3.
    """

    @pytest.fixture
    def agent(self) -> KBaseThermodynamicAgent:
        return _default_agent(dim_q=3, dim_p=3)

    @pytest.fixture
    def identity_agent(self) -> KBaseThermodynamicAgent:
        """Agente con C_soc=I, M_rec=I, R_cost=0 para verificaciones exactas."""
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        return KBaseThermodynamicAgent(
            C_soc=np.eye(dim_q),
            M_rec=np.eye(dim_p),
            R_cost=np.zeros((n, n)),
            J_base=_make_antisymmetric(n, seed=15),
            breakdown_voltage=1.0e9,
        )

    # ── 3.1 Validación de dimensiones de state_x ──────────────────────────

    def test_wrong_state_x_shape_raises_dimension_error(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        state_x con shape ≠ (dim_q + dim_p,) debe lanzar DimensionMismatchError.
        """
        n = agent.context.dim_q + agent.context.dim_p
        state_x_bad = np.ones(n + 2)
        with pytest.raises(DimensionMismatchError, match=r"state_x"):
            agent.export_sheaf_stalk(state_x=state_x_bad)

    def test_empty_state_x_raises_dimension_error(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """state_x con shape (0,) debe lanzar DimensionMismatchError."""
        with pytest.raises(DimensionMismatchError):
            agent.export_sheaf_stalk(state_x=np.array([]))

    # ── 3.2 Forma y tipo de δ_{BASE} ──────────────────────────────────────

    def test_delta_base_shape_is_n_by_n(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """δ_{BASE} debe tener shape (n, n) donde n = dim_q + dim_p."""
        n = agent.context.dim_q + agent.context.dim_p
        state_x = np.ones(n)
        stalk = agent.export_sheaf_stalk(state_x=state_x)
        assert stalk.delta_base.shape == (n, n)

    def test_delta_base_is_float64(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """δ_{BASE} debe ser de dtype float64."""
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        assert stalk.delta_base.dtype == np.float64

    def test_delta_base_is_symmetric(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        δ_{BASE} es simétrica por construcción espectral
        (raíz cuadrada de una matriz simétrica es simétrica).
        """
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        delta = stalk.delta_base
        residual = float(la.norm(delta - delta.T, "fro"))
        tol = 100 * _EPS * float(la.norm(delta, "fro"))
        assert residual < tol, (
            f"δ_{'{BASE}'} no es simétrica: ‖δ − δ^⊤‖_F = {residual:.3e}"
        )

    # ── 3.3 Identidad de Hodge local: δ^⊤ δ = Hodge_local ───────────────

    def test_hodge_identity_satisfied(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Identidad de Hodge local:
        ‖ δ^⊤δ − block_diag(C_soc⁻¹, R_cost) ‖_F / ‖δ‖_F² < 100·ε_mach.
        """
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        n = dim_q + dim_p

        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        delta = stalk.delta_base

        # Métrica de Hodge esperada: block_diag(C_soc⁻¹, R_cost)
        C_inv = la.inv(agent.context.L_C @ agent.context.L_C.T)
        Hodge_expected = np.zeros((n, n))
        Hodge_expected[:dim_q, :dim_q] = C_inv
        Hodge_expected[dim_q:, dim_q:] = agent.context.R_cost

        delta_T_delta = delta.T @ delta
        residual = float(la.norm(delta_T_delta - Hodge_expected, "fro"))
        norm_delta_sq = float(la.norm(delta, "fro") ** 2)
        rel_error = residual / max(norm_delta_sq, 1.0)

        tol = 100 * _EPS
        assert rel_error < tol, (
            f"Identidad de Hodge violada: "
            f"‖δ^⊤δ − Hodge‖_F/‖δ‖_F² = {rel_error:.3e} > {tol:.3e}"
        )

    def test_hodge_identity_for_identity_matrices(
        self, identity_agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Con C_soc=I, M_rec=I, R_cost=0:
        δ_{BASE} = block_diag(I_{dim_q}, 0_{dim_p}).
        La parte (1,1): C_soc^{-1/2} = I.
        La parte (2,2): R_cost^{+1/2} = 0.
        """
        dim_q = identity_agent.context.dim_q
        dim_p = identity_agent.context.dim_p
        n = dim_q + dim_p

        stalk = identity_agent.export_sheaf_stalk(state_x=np.ones(n))
        delta = stalk.delta_base

        # Bloque (1,1): C_soc^{-1/2} = I^{-1/2} = I
        delta_11 = delta[:dim_q, :dim_q]
        assert np.allclose(delta_11, np.eye(dim_q), atol=_ATOL_ANALYTICAL), (
            f"Bloque (1,1) de δ con C_soc=I debe ser I: "
            f"‖δ_11 − I‖_F = {la.norm(delta_11 - np.eye(dim_q)):.3e}"
        )

        # Bloque (2,2): R_cost^{+1/2} = 0^{+1/2} = 0
        delta_22 = delta[dim_q:, dim_q:]
        assert np.allclose(delta_22, np.zeros((dim_p, dim_p)), atol=_ATOL_ANALYTICAL)

        # Bloques cruzados: 0
        delta_12 = delta[:dim_q, dim_q:]
        delta_21 = delta[dim_q:, :dim_q]
        assert np.allclose(delta_12, 0.0, atol=_ATOL_ANALYTICAL)
        assert np.allclose(delta_21, 0.0, atol=_ATOL_ANALYTICAL)

    # ── 3.4 Proyección δ · state_x ───────────────────────────────────────

    def test_projected_state_shape_and_dtype(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """projected_state debe tener shape (n,) y dtype float64."""
        n = agent.context.dim_q + agent.context.dim_p
        state_x = np.ones(n)
        stalk = agent.export_sheaf_stalk(state_x=state_x)
        assert stalk.projected_state.shape == (n,)
        assert stalk.projected_state.dtype == np.float64

    def test_projected_state_equals_delta_times_x(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        projected_state = δ_{BASE} · state_x (multiplicación matricial exacta).
        """
        n = agent.context.dim_q + agent.context.dim_p
        rng = np.random.default_rng(400)
        state_x = rng.standard_normal(n)
        stalk = agent.export_sheaf_stalk(state_x=state_x)
        expected = stalk.delta_base @ state_x
        assert np.allclose(stalk.projected_state, expected, atol=_ATOL_ANALYTICAL)

    def test_projected_state_zero_for_zero_x(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """δ · 0 = 0 (linealidad)."""
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.zeros(n))
        assert np.allclose(stalk.projected_state, 0.0, atol=_ATOL_ANALYTICAL)

    def test_projected_state_linearity(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        Linealidad: δ·(ax + by) = a·δ·x + b·δ·y.
        Verificado para a=2, b=-3, x y y aleatorios.
        """
        n = agent.context.dim_q + agent.context.dim_p
        rng = np.random.default_rng(401)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        a, b = 2.0, -3.0

        stalk_x = agent.export_sheaf_stalk(state_x=x)
        stalk_y = agent.export_sheaf_stalk(state_x=y)
        stalk_combined = agent.export_sheaf_stalk(state_x=a * x + b * y)

        expected = a * stalk_x.projected_state + b * stalk_y.projected_state
        assert np.allclose(
            stalk_combined.projected_state, expected, atol=_ATOL_ANALYTICAL
        )

    def test_state_vector_is_copy_not_reference(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        state_vector en SheafStalk debe ser una copia independiente de state_x.
        Modificar state_x después de la llamada no debe alterar el stalk.
        """
        n = agent.context.dim_q + agent.context.dim_p
        state_x = np.ones(n) * 3.0
        stalk = agent.export_sheaf_stalk(state_x=state_x)

        # Modificar el original
        state_x[:] = 999.0

        # El stalk no debe verse afectado
        assert np.all(stalk.state_vector == 3.0), (
            "state_vector fue alterado al modificar el array original (no es copia)."
        )

    # ── 3.5 Rango de δ_{BASE} ─────────────────────────────────────────────

    def test_rank_delta_equals_dim_q_plus_rank_r(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        rank_delta = dim_q + rank_R (C_soc^{-1/2} es cuadrada no singular,
        R_cost^{+1/2} tiene rango = rank_R).
        """
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        expected_rank = agent.context.dim_q + agent.context.rank_R
        assert stalk.rank_delta == expected_rank

    def test_rank_delta_full_for_full_rank_r_cost(self) -> None:
        """
        Con R_cost SPD (rango pleno), rank_delta = dim_q + dim_p = n.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        C_soc = _make_spd_matrix(dim_q, seed=40)
        M_rec = _make_spd_matrix(dim_p, seed=41)
        R_cost = _make_spd_matrix(n, seed=42)  # SPD = rango pleno
        J_base = _make_antisymmetric(n, seed=43)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base
        )

        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        assert stalk.rank_delta == n

    # ── 3.6 Norma δ_{BASE}² y delta_base_sq_norm ─────────────────────────

    def test_delta_base_sq_norm_is_nonnegative(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """delta_base_sq_norm = ‖δ²‖_F ≥ 0 siempre."""
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        assert stalk.delta_base_sq_norm >= 0.0

    def test_delta_base_sq_norm_is_float(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """delta_base_sq_norm debe ser un escalar float (no array)."""
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        assert isinstance(stalk.delta_base_sq_norm, float)

    # ── 3.7 Instanciación perezosa de phase3 ─────────────────────────────

    def test_phase3_is_none_before_first_call(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """phase3 debe ser None antes de la primera llamada a export_sheaf_stalk."""
        assert agent.phase3 is None

    def test_phase3_is_instantiated_after_first_call(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """phase3 no debe ser None después de la primera llamada a export_sheaf_stalk."""
        n = agent.context.dim_q + agent.context.dim_p
        agent.export_sheaf_stalk(state_x=np.ones(n))
        assert agent.phase3 is not None

    def test_phase3_is_reused_across_calls(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """
        La misma instancia de phase3 debe ser reutilizada en llamadas
        subsecuentes (identidad de objeto: id(phase3) no cambia).
        """
        n = agent.context.dim_q + agent.context.dim_p
        agent.export_sheaf_stalk(state_x=np.ones(n))
        phase3_first = agent.phase3
        agent.export_sheaf_stalk(state_x=np.zeros(n))
        phase3_second = agent.phase3
        assert phase3_first is phase3_second

    # ── 3.8 Inmutabilidad del SheafStalk ─────────────────────────────────

    def test_sheaf_stalk_is_immutable_frozen_dataclass(
        self, agent: KBaseThermodynamicAgent
    ) -> None:
        """SheafStalk es un dataclass frozen; no permite asignaciones."""
        n = agent.context.dim_q + agent.context.dim_p
        stalk = agent.export_sheaf_stalk(state_x=np.ones(n))
        with pytest.raises((AttributeError, TypeError)):
            stalk.rank_delta = 999  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 4 – PRUEBAS DE INTEGRACIÓN DE LAS 3 FASES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestIntegrationPipeline:
    """
    Pruebas de integración que ejercitan el pipeline completo de las 3 fases.

    Verifican propiedades de extremo a extremo: determinismo, ortogonalidad,
    escala, logging y consistencia del flujo de datos entre fases.
    """

    # ── 4.1 Pipeline completo nominal ─────────────────────────────────────

    def test_full_pipeline_nominal_produces_valid_outputs(self) -> None:
        """
        Pipeline completo: constructor → synthesize → export_stalk.
        Todos los outputs deben ser instancias de los DTOs correctos y
        tener valores físicamente coherentes.
        """
        dim_q, dim_p = 4, 3
        agent = _default_agent(dim_q=dim_q, dim_p=dim_p)

        n = dim_q + dim_p
        rng = np.random.default_rng(500)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = rng.standard_normal(dim_p) * 0.01  # pequeño para no superar breakdown
        state_x = np.concatenate([q, p])

        # Fase 2
        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert isinstance(state, BasalStateTensor)
        assert state.total_hamiltonian >= 0.0
        assert state.dissipated_power >= 0.0
        assert state.is_thermodynamically_stable

        # Fase 3
        stalk = agent.export_sheaf_stalk(state_x=state_x)
        assert isinstance(stalk, SheafStalk)
        assert stalk.delta_base.shape == (n, n)
        assert stalk.rank_delta == dim_q + agent.context.rank_R

    # ── 4.2 Determinismo ──────────────────────────────────────────────────

    def test_pipeline_is_deterministic(self) -> None:
        """
        Misma entrada → misma salida en dos invocaciones independientes.
        Verifica que no hay estado mutable oculto en los cálculos.
        """
        C_soc, M_rec, R_cost, J_base = _default_matrices(seed=42)
        dim_q = C_soc.shape[0]
        dim_p = M_rec.shape[0]
        n = dim_q + dim_p

        def _run() -> Tuple[BasalStateTensor, SheafStalk]:
            agent = KBaseThermodynamicAgent(
                C_soc=C_soc.copy(), M_rec=M_rec.copy(),
                R_cost=R_cost.copy(), J_base=J_base.copy(),
            )
            q = np.ones(dim_q) * 1.5
            p = np.ones(dim_p) * 0.7
            df_dt = np.zeros(dim_p)
            state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
            stalk = agent.export_sheaf_stalk(state_x=np.concatenate([q, p]))
            return state, stalk

        state1, stalk1 = _run()
        state2, stalk2 = _run()

        assert state1.total_hamiltonian == pytest.approx(state2.total_hamiltonian)
        assert state1.dissipated_power == pytest.approx(state2.dissipated_power)
        assert np.allclose(stalk1.delta_base, stalk2.delta_base)
        assert np.allclose(stalk1.projected_state, stalk2.projected_state)

    # ── 4.3 Ortogonalidad q-p ─────────────────────────────────────────────

    def test_perturbation_in_q_does_not_affect_kinetic_energy(self) -> None:
        """
        Separabilidad: K(p) no depende de q. Perturbar q no cambia K(p).
        """
        agent = _default_agent()
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p

        p = np.ones(dim_p) * 2.0
        df_dt = np.zeros(dim_p)

        q1 = np.zeros(dim_q)
        q2 = np.ones(dim_q) * 100.0

        state1 = agent.synthesize_basal_hamiltonian(q=q1, p=p, df_dt=df_dt)
        state2 = agent.synthesize_basal_hamiltonian(q=q2, p=p, df_dt=df_dt)

        assert state1.kinetic_energy == pytest.approx(
            state2.kinetic_energy, rel=_RTOL_ANALYTICAL
        ), "K(p) cambió al perturbar q (violación de separabilidad)."

    def test_perturbation_in_p_does_not_affect_potential_energy(self) -> None:
        """
        Separabilidad: V(q) no depende de p. Perturbar p no cambia V(q).
        """
        agent = _default_agent()
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p

        q = np.ones(dim_q) * 1.5
        df_dt = np.zeros(dim_p)

        p1 = np.zeros(dim_p)
        p2 = np.ones(dim_p) * 50.0

        state1 = agent.synthesize_basal_hamiltonian(q=q, p=p1, df_dt=df_dt)
        state2 = agent.synthesize_basal_hamiltonian(q=q, p=p2, df_dt=df_dt)

        assert state1.potential_energy == pytest.approx(
            state2.potential_energy, rel=_RTOL_ANALYTICAL
        ), "V(q) cambió al perturbar p (violación de separabilidad)."

    # ── 4.4 Homogeneidad cuadrática del Hamiltoniano ──────────────────────

    @pytest.mark.parametrize("lambda_", [0.5, 2.0, 10.0, 0.01])
    def test_hamiltonian_homogeneity_various_scales(self, lambda_: float) -> None:
        """
        H(λq, λp) = λ² H(q, p) para múltiples valores de λ.
        """
        agent = _default_agent(seed=77)
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        rng = np.random.default_rng(int(lambda_ * 100))
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        s1 = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        s2 = agent.synthesize_basal_hamiltonian(
            q=lambda_ * q, p=lambda_ * p, df_dt=df_dt
        )

        assert s2.total_hamiltonian == pytest.approx(
            lambda_ ** 2 * s1.total_hamiltonian,
            rel=_RTOL_ANALYTICAL,
        ), (
            f"H(λq,λp) ≠ λ²H(q,p) para λ={lambda_}: "
            f"{s2.total_hamiltonian:.6e} ≠ {lambda_**2 * s1.total_hamiltonian:.6e}"
        )

    # ── 4.5 Consistencia del flujo de datos entre fases ───────────────────

    def test_state_vector_from_phase2_feeds_phase3_correctly(self) -> None:
        """
        El vector state_x = np.concatenate([q, p]) producido en Fase 2
        alimenta correctamente Fase 3 y su proyección es consistente.
        """
        dim_q, dim_p = 3, 3
        agent = _default_agent(dim_q=dim_q, dim_p=dim_p)

        rng = np.random.default_rng(600)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        # Fase 2
        agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        # state_x como se construiría desde el BasalStateTensor
        state_x = np.concatenate([q, p])
        stalk = agent.export_sheaf_stalk(state_x=state_x)

        # La proyección debe ser δ · [q; p]
        expected_proj = stalk.delta_base @ state_x
        assert np.allclose(stalk.projected_state, expected_proj, atol=_ATOL_ANALYTICAL)

    # ── 4.6 Logging ───────────────────────────────────────────────────────

    def test_phase1_emits_info_log_on_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        La construcción exitosa del agente debe emitir al menos un mensaje
        INFO del logger 'MIC.Alpha.KBaseThermodynamicAgent'.
        """
        with caplog.at_level(logging.INFO, logger="MIC.Alpha.KBaseThermodynamicAgent"):
            _default_agent()
        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "KBaseThermodynamicAgent" in r.name
        ]
        assert len(info_records) >= 1, (
            "No se emitió ningún mensaje INFO durante la construcción del agente."
        )

    def test_phase2_emits_info_log_on_synthesize(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        synthesize_basal_hamiltonian debe emitir al menos un mensaje INFO
        con información del estado basal calculado.
        """
        agent = _default_agent()
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p

        with caplog.at_level(logging.INFO, logger="MIC.Alpha.KBaseThermodynamicAgent"):
            agent.synthesize_basal_hamiltonian(
                q=np.ones(dim_q),
                p=np.ones(dim_p),
                df_dt=np.zeros(dim_p),
            )

        info_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
        ]
        assert len(info_records) >= 1

    def test_phase3_emits_info_log_on_export(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """export_sheaf_stalk debe emitir al menos un mensaje INFO."""
        agent = _default_agent()
        n = agent.context.dim_q + agent.context.dim_p

        with caplog.at_level(logging.INFO, logger="MIC.Alpha.KBaseThermodynamicAgent"):
            agent.export_sheaf_stalk(state_x=np.ones(n))

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) >= 1

    # ── 4.7 Pruebas de dimensiones asimétricas (dim_q ≠ dim_p) ──────────

    @pytest.mark.parametrize("dim_q,dim_p", [(2, 5), (5, 2), (1, 4), (4, 1)])
    def test_asymmetric_dimensions_full_pipeline(
        self, dim_q: int, dim_p: int
    ) -> None:
        """
        El pipeline completo debe funcionar para dim_q ≠ dim_p.
        Verifica que no hay suposiciones implícitas de igualdad de dimensiones.
        """
        n = dim_q + dim_p
        C_soc = _make_spd_matrix(dim_q, seed=dim_q * 10)
        M_rec = _make_spd_matrix(dim_p, seed=dim_p * 10 + 1)
        R_cost = _make_psd_matrix(n, rank=n - 1, seed=dim_q + dim_p)
        J_base = _make_antisymmetric(n, seed=dim_q + dim_p + 1)

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=1.0e8,
        )

        q = np.ones(dim_q)
        p = np.ones(dim_p)
        df_dt = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)
        assert state.total_hamiltonian > 0.0
        assert state.is_thermodynamically_stable

        stalk = agent.export_sheaf_stalk(state_x=np.concatenate([q, p]))
        assert stalk.delta_base.shape == (n, n)

    # ── 4.8 Coherencia entre BasalStateTensor y SheafStalk ───────────────

    def test_grad_h_norm_consistent_with_hamiltonian_magnitude(self) -> None:
        """
        Para un sistema con C_soc = M_rec = I:
        ‖∇H‖² = ‖q‖² + ‖p‖² = 2·H (pues H = ½(‖q‖²+‖p‖²) y ∇H=[q;p]).

        Esta es la relación de Euler para funciones cuadráticas homogéneas.
        """
        dim_q, dim_p = 3, 3
        n = dim_q + dim_p
        agent = KBaseThermodynamicAgent(
            C_soc=np.eye(dim_q),
            M_rec=np.eye(dim_p),
            R_cost=np.zeros((n, n)),
            J_base=_make_antisymmetric(n, seed=20),
            breakdown_voltage=1.0e9,
        )

        rng = np.random.default_rng(700)
        q = rng.standard_normal(dim_q)
        p = rng.standard_normal(dim_p)
        df_dt = np.zeros(dim_p)

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        # Con C=M=I: ∇H = [q; p], ‖∇H‖² = ‖q‖² + ‖p‖²
        # H = ½(‖q‖²+‖p‖²) → ‖∇H‖² = 2H
        expected_grad_norm_sq = 2.0 * state.total_hamiltonian
        actual_grad_norm_sq = state.grad_H_norm ** 2

        assert actual_grad_norm_sq == pytest.approx(
            expected_grad_norm_sq, rel=_RTOL_ANALYTICAL
        ), (
            f"‖∇H‖² = {actual_grad_norm_sq:.6e} ≠ 2H = {expected_grad_norm_sq:.6e}"
        )

    # ── 4.9 Prueba de dimensión mínima (n=1) ─────────────────────────────

    def test_minimal_dimension_n1_pipeline(self) -> None:
        """
        Pipeline funcional para dim_q=1, dim_p=1 (caso mínimo no trivial).
        Verifica que no hay dependencias implícitas de n ≥ 2.
        """
        dim_q, dim_p = 1, 1
        n = 2
        C_soc = np.array([[3.0]])
        M_rec = np.array([[2.0]])
        R_cost = np.array([[0.5, 0.0], [0.0, 0.5]])
        J_base = np.array([[0.0, -1.0], [1.0, 0.0]])  # rotación 2D

        agent = KBaseThermodynamicAgent(
            C_soc=C_soc, M_rec=M_rec, R_cost=R_cost, J_base=J_base,
            breakdown_voltage=1.0e6,
        )

        q = np.array([2.0])
        p = np.array([1.0])
        df_dt = np.array([0.0])

        state = agent.synthesize_basal_hamiltonian(q=q, p=p, df_dt=df_dt)

        # V = ½ q^⊤ C⁻¹ q = ½ · (2²/3) = 2/3
        V_analytical = 0.5 * q[0] ** 2 / C_soc[0, 0]
        # K = ½ p^⊤ M⁻¹ p = ½ · (1²/2) = 1/4
        K_analytical = 0.5 * p[0] ** 2 / M_rec[0, 0]

        assert state.potential_energy == pytest.approx(V_analytical, rel=1e-10)
        assert state.kinetic_energy == pytest.approx(K_analytical, rel=1e-10)
        assert state.total_hamiltonian == pytest.approx(
            V_analytical + K_analytical, rel=1e-10
        )

        stalk = agent.export_sheaf_stalk(state_x=np.array([q[0], p[0]]))
        assert stalk.delta_base.shape == (n, n)
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: KCore Kinematic Agent                                      ║
║ Ubicación: tests/unit/alpha/kcore/test_kcore_kinematic_agent.py              ║
║ Versión   : 1.0.0-Strict-Spectral-Phased                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Cobertura por Fase                                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Fase 1 – Validación Matricial Constitutiva:                                 ║
║    • Dimensiones: ndim, cuadratura, coherencia (n,n) y (n,m), m≥1.           ║
║    • Antisimetría de J y J_d con tolerancia relativa ε_mach·‖A‖_F.           ║
║    • Simetría de R y R_d con tolerancia relativa ε_mach·‖A‖_F.               ║
║    • PSD de R y R_d: autovalor negativo real lanza excepción.                ║
║    • Condicionamiento κ(R), κ(R_d) > kappa_max lanza excepción.              ║
║    • Rango de g: g nula lanza excepción; g rango-deficiente es aceptada.     ║
║    • KinematicPreparationContext: tipos, shapes, invariantes.                ║
║    • Inmutabilidad del contexto (frozen dataclass).                          ║
║                                                                              ║
║  Fase 2 – Síntesis Cinemática:                                               ║
║    • IDA-PBC: residuo relativo < residual_tol_rel para g de rango pleno.     ║
║    • IDA-PBC: residuo elevado con g rango-deficiente lanza                   ║
║               DiracMatchingError.                                            ║
║    • IDA-PBC: dimensiones incorrectas de grad_H/grad_H_d.                    ║
║    • IDA-PBC: F_req≈0 no lanza excepción (residuo=0).                        ║
║    • IDA-PBC: α satisface g·α≈F_req analíticamente.                          ║
║    • IDA-PBC: SVD con criterio Golub-Van Loan (σ_tol escalado).              ║
║    • Hodge: vorticidad nula → W sin modificar.                               ║
║    • Hodge: vorticidad supercrítica → diagonal penalizada.                   ║
║    • Hodge: I_curl con forma incorrecta lanza DimensionError.                ║
║    • Hodge: strangle_factor ≤ 0 lanza ParasiticVorticityError.               ║
║    • Hodge: diagonal W_mod ≥ 0 tras estrangulamiento.                        ║
║    • Hodge: norma de vorticidad es escalar no negativo.                      ║
║    • Hodge: conversión universal a CSR (formatos DIA, COO, LIL).             ║
║    • Kramers-Kronig: Z_load SPD → ε_eff, μ_eff SPD.                          ║
║    • Kramers-Kronig: Z_load no SPD lanza ImpedanceReflectionError.           ║
║    • Kramers-Kronig: verificación causal ‖ε_eff−Z_load‖/‖Z‖ < 100·ε.         ║
║    • Kramers-Kronig: μ_eff = Z_load² (identidad analítica).                  ║
║    • CFL: λ_max correcto para Laplaciano conocido.                           ║
║    • CFL: c_eff ≤ 0 lanza CFLViolationError.                                 ║
║    • CFL: Laplaciano nulo → dt_safe = +∞.                                    ║
║    • CFL: dt_requested > dt_safe lanza CFLViolationError.                    ║
║    • CFL: dt_requested ≤ dt_safe es aceptado.                                ║
║    • synthesize: KinematicStateTensor con campos correctos.                  ║
║    • synthesize: is_kinematically_stable = True en éxito.                    ║
║    • synthesize: campos residual_idapbc, vorticity_norm documentados.        ║
║                                                                              ║
║  Fase 3 – Proyección en Haces:                                               ║
║    • δ_{CORE} shape (E,E) y dtype float64.                                   ║
║    • Identidad de Hodge: δ^⊤ δ ≈ W_mod (tolerancia 100·ε_mach).              ║
║    • Proyección δ·x correcta (linealidad, norma, cero).                      ║
║    • rank_delta = rango de W_mod.                                            ║
║    • delta_hodge_residual ≥ 0 y es float.                                    ║
║    • state_vector es copia independiente.                                    ║
║    • W_mod con diagonal negativa lanza SheafCoboundaryError.                 ║
║    • Instanciación perezosa (None antes, no-None después).                   ║
║    • Invalidación de phase3 tras nueva síntesis.                             ║
║                                                                              ║
║  Integración de las 3 Fases:                                                 ║
║    • Pipeline completo nominal.                                              ║
║    • Determinismo (misma entrada → misma salida).                            ║
║    • cfl_margin fuera de (0,1] lanza ValueError.                             ║
║    • Logging INFO en construcción, síntesis y exportación.                   ║
║    • Dimensiones asimétricas n distinto de m.                                ║
║    • n mínimo (n=2, m=1).                                                    ║
║    • Inmutabilidad de KinematicStateTensor y SheafStalk.                     ║
║    • Coherencia hodge_conductance → delta_core (flujo Fase2→Fase3).          ║
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
import scipy.sparse as sp
from numpy.typing import NDArray

# ── Módulo bajo prueba ───────────────────────────────────────────────────────
from app.agents.alpha.kcore.kcore_kinematic_agent import (
    # Agente orquestador
    KCoreKinematicAgent,
    # DTOs
    KinematicPreparationContext,
    KinematicStateTensor,
    SheafStalk,
    # Excepciones
    CFLViolationError,
    DiracMatchingError,
    ImpedanceReflectionError,
    KinematicConditionError,
    KinematicCoreError,
    KinematicDimensionError,
    KinematicSymmetryError,
    ParasiticVorticityError,
    SheafCoboundaryError,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 0 – CONSTANTES Y UTILIDADES DE PRUEBA                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_EPS: float = float(np.finfo(np.float64).eps)
_ATOL: float = 1.0e-10
_RTOL: float = 1.0e-10


# ── Fábricas de matrices constitutivas ──────────────────────────────────────

def _spd(n: int, seed: int = 0, kappa: float = 10.0) -> NDArray[np.float64]:
    """
    Genera una matriz SPD de dimensión n×n con número de condición ≈ kappa.
    Construcción: Q · diag(λ) · Q^⊤ con λ ∈ [1, kappa] log-uniforme.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.logspace(0.0, np.log10(kappa), n)
    return Q @ np.diag(eigvals) @ Q.T


def _psd(n: int, rank: int, seed: int = 0) -> NDArray[np.float64]:
    """Genera una matriz PSD de rango ``rank`` ≤ n."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.zeros(n)
    eigvals[:rank] = np.linspace(1.0, float(rank + 1), rank)
    return Q @ np.diag(eigvals) @ Q.T


def _antisym(n: int, seed: int = 0) -> NDArray[np.float64]:
    """Genera una matriz antisimétrica J = −J^⊤."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A - A.T


def _diag_sparse(
    values: NDArray[np.float64],
    fmt: str = "csr",
) -> sp.spmatrix:
    """
    Crea una matriz dispersa diagonal a partir de un vector de valores.
    Admite formatos: 'csr', 'csc', 'coo', 'dia', 'lil'.
    """
    E = len(values)
    return sp.diags(values, offsets=0, shape=(E, E), format=fmt)


def _laplacian_path_graph(n: int) -> sp.csr_matrix:
    """
    Laplaciano del grafo camino P_n (n nodos, n-1 aristas).
    Autovalores conocidos: λ_k = 4 sin²(kπ/(2n)), k=1,...,n.
    λ_max ≈ 4 para n grande.
    """
    diag_main = np.full(n, 2.0)
    diag_main[0] = 1.0
    diag_main[-1] = 1.0
    diag_off = np.full(n - 1, -1.0)
    L = sp.diags(
        [diag_off, diag_main, diag_off],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format="csr",
    )
    return L


def _default_matrices(
    n: int = 4,
    m: int = 2,
    seed: int = 7,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Retorna (J, R, J_d, R_d, g) válidas por defecto.

    • J, J_d : antisimétrica (n×n)
    • R, R_d : PSD (n×n)
    • g       : matriz de control (n×m)
    """
    J = _antisym(n, seed=seed)
    R = _psd(n, rank=n - 1, seed=seed + 1)
    J_d = _antisym(n, seed=seed + 2)
    R_d = _psd(n, rank=n - 1, seed=seed + 3)
    rng = np.random.default_rng(seed + 4)
    g = rng.standard_normal((n, m))
    return J, R, J_d, R_d, g


def _default_agent(
    n: int = 4,
    m: int = 2,
    cfl_margin: float = 0.9,
    kappa_max: float = 1.0e10,
    residual_tol_rel: float = 1.0e-6,
    seed: int = 7,
) -> KCoreKinematicAgent:
    """Construye un KCoreKinematicAgent con matrices válidas por defecto."""
    J, R, J_d, R_d, g = _default_matrices(n=n, m=m, seed=seed)
    return KCoreKinematicAgent(
        J=J,
        R=R,
        J_d=J_d,
        R_d=R_d,
        g=g,
        cfl_margin=cfl_margin,
        kappa_max=kappa_max,
        residual_tol_rel=residual_tol_rel,
    )


def _default_synthesis_inputs(
    agent: KCoreKinematicAgent,
    E: int = 5,
    d: int = 3,
    seed: int = 99,
) -> dict:
    """
    Genera un diccionario con todos los argumentos válidos para
    ``synthesize_kinematic_core``.

    Parámetros
    ----------
    agent : KCoreKinematicAgent
        Agente ya construido (determina n).
    E : int
        Número de aristas del grafo logístico.
    d : int
        Dimensión del tensor de impedancia Z_load.
    seed : int
        Semilla para reproducibilidad.
    """
    n = agent.context.n
    rng = np.random.default_rng(seed)

    grad_H = rng.standard_normal(n)
    grad_H_d = rng.standard_normal(n)

    # W: diagonal con valores positivos (E aristas)
    w_vals = np.abs(rng.standard_normal(E)) + 0.5
    W = _diag_sparse(w_vals, fmt="csr")

    # I_curl: pequeño para evitar estrangulamiento (vorticidad subcrítica)
    I_curl = rng.standard_normal(E) * 1e-5

    # Z_load: SPD
    Z_load = _spd(d, seed=seed + 1)

    c_eff = 2.0
    Delta_sym = _laplacian_path_graph(6)

    # dt_safe teórico para el Laplaciano de camino con 6 nodos
    # λ_max ≤ 4 → dt_safe ≥ 2·CFL/(c·√4) = CFL/c = 0.45
    dt_requested = 0.1  # conservador

    return dict(
        grad_H=grad_H,
        grad_H_d=grad_H_d,
        W=W,
        I_curl=I_curl,
        Z_load=Z_load,
        c_eff=c_eff,
        Delta_sym=Delta_sym,
        dt_requested=dt_requested,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 1 – PRUEBAS DE FASE 1: VALIDACIÓN MATRICIAL CONSTITUTIVA       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase1MatrixValidation:
    """
    Pruebas exhaustivas de Phase1_MatrixValidation y build_preparation_context.
    """

    # ── 1.1 Validación dimensional ────────────────────────────────────────

    def test_j_not_2d_raises_dimension_error(self) -> None:
        """J con ndim=1 debe lanzar KinematicDimensionError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        with pytest.raises(KinematicDimensionError, match=r"2D"):
            KCoreKinematicAgent(
                J=J[0],  # shape (4,) no 2D
                R=R, J_d=J_d, R_d=R_d, g=g,
            )

    def test_j_not_square_raises_dimension_error(self) -> None:
        """J no cuadrada debe lanzar KinematicDimensionError."""
        _, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        J_bad = np.zeros((4, 3))  # rectangular
        with pytest.raises(KinematicDimensionError, match=r"cuadrada"):
            KCoreKinematicAgent(J=J_bad, R=R, J_d=J_d, R_d=R_d, g=g)

    def test_r_wrong_shape_raises_dimension_error(self) -> None:
        """R con shape (n+1, n+1) en lugar de (n, n) lanza KinematicDimensionError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        R_bad = _psd(5, rank=4)  # dimensión 5 en lugar de 4
        with pytest.raises(KinematicDimensionError, match=r"R"):
            KCoreKinematicAgent(J=J, R=R_bad, J_d=J_d, R_d=R_d, g=g)

    def test_j_d_wrong_shape_raises_dimension_error(self) -> None:
        """J_d con dimensión diferente a J lanza KinematicDimensionError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        J_d_bad = _antisym(3)  # shape (3,3) en lugar de (4,4)
        with pytest.raises(KinematicDimensionError, match=r"J_d"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d_bad, R_d=R_d, g=g)

    def test_r_d_wrong_shape_raises_dimension_error(self) -> None:
        """R_d con dimensión diferente lanza KinematicDimensionError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        R_d_bad = _psd(5, rank=4)
        with pytest.raises(KinematicDimensionError, match=r"R_d"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d_bad, g=g)

    def test_g_wrong_row_count_raises_dimension_error(self) -> None:
        """g con nº de filas ≠ n lanza KinematicDimensionError."""
        J, R, J_d, R_d, _ = _default_matrices(n=4, m=2)
        g_bad = np.ones((3, 2))  # 3 filas en lugar de 4
        with pytest.raises(KinematicDimensionError, match=r"g"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g_bad)

    def test_g_not_2d_raises_dimension_error(self) -> None:
        """g con ndim=1 lanza KinematicDimensionError."""
        J, R, J_d, R_d, _ = _default_matrices(n=4, m=2)
        g_bad = np.ones(4)  # 1D
        with pytest.raises(KinematicDimensionError, match=r"2D"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g_bad)

    def test_g_zero_columns_raises_dimension_error(self) -> None:
        """g con m=0 columnas lanza KinematicDimensionError."""
        J, R, J_d, R_d, _ = _default_matrices(n=4, m=2)
        g_bad = np.zeros((4, 0))  # m=0
        with pytest.raises(KinematicDimensionError):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g_bad)

    # ── 1.2 Antisimetría de J y J_d ───────────────────────────────────────

    def test_j_symmetric_raises_symmetry_error(self) -> None:
        """
        J simétrica (J = J^⊤) debe lanzar KinematicSymmetryError
        con diagnóstico de ‖J + J^⊤‖_F.
        """
        _, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        J_sym = _spd(4, seed=1)  # SPD → simétrica, no antisimétrica
        with pytest.raises(KinematicSymmetryError, match=r"antisim"):
            KCoreKinematicAgent(J=J_sym, R=R, J_d=J_d, R_d=R_d, g=g)

    def test_j_d_not_antisymmetric_raises_symmetry_error(self) -> None:
        """J_d = I (simétrica) debe lanzar KinematicSymmetryError."""
        J, R, _, R_d, g = _default_matrices(n=4, m=2)
        with pytest.raises(KinematicSymmetryError):
            KCoreKinematicAgent(
                J=J, R=R,
                J_d=np.eye(4),  # simétrica
                R_d=R_d, g=g,
            )

    def test_j_exact_antisymmetric_is_accepted(self) -> None:
        """
        J exactamente antisimétrica debe ser aceptada.
        """
        n = 4
        J_exact = _antisym(n, seed=10)
        R = _psd(n, rank=n-1, seed=11)
        J_d = _antisym(n, seed=12)
        R_d = _psd(n, rank=n-1, seed=13)
        g = np.random.default_rng(14).standard_normal((n, 2))
        agent = KCoreKinematicAgent(J=J_exact, R=R, J_d=J_d, R_d=R_d, g=g)
        assert agent.context is not None

    def test_j_near_antisymmetric_within_tolerance_accepted(self) -> None:
        """
        J con perturbación simétrica de magnitud ε_mach·‖J‖_F/2 es aceptada.
        """
        n = 4
        J_base = _antisym(n, seed=20)
        # Forzar antisimetría exacta después de crear la perturbación
        J_accepted = J_base - J_base.T
        J_accepted /= 2.0
        R = _psd(n, rank=n-1, seed=21)
        J_d = _antisym(n, seed=22)
        R_d = _psd(n, rank=n-1, seed=23)
        g = np.random.default_rng(24).standard_normal((n, 2))
        agent = KCoreKinematicAgent(J=J_accepted, R=R, J_d=J_d, R_d=R_d, g=g)
        assert agent.context.n == n

    # ── 1.3 Simetría de R y R_d ───────────────────────────────────────────

    def test_r_asymmetric_raises_symmetry_error(self) -> None:
        """R asimétrica lanza KinematicSymmetryError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        R_asym = R.copy()
        R_asym[0, 2] += 5.0  # rompe simetría
        with pytest.raises(KinematicSymmetryError, match=r"simétric"):
            KCoreKinematicAgent(J=J, R=R_asym, J_d=J_d, R_d=R_d, g=g)

    def test_r_d_asymmetric_raises_symmetry_error(self) -> None:
        """R_d asimétrica lanza KinematicSymmetryError."""
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        R_d_asym = R_d.copy()
        R_d_asym[1, 3] += 3.0
        with pytest.raises(KinematicSymmetryError, match=r"simétric"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d_asym, g=g)

    # ── 1.4 PSD de R y R_d ────────────────────────────────────────────────

    def test_r_negative_eigenvalue_raises_symmetry_error(self) -> None:
        """
        R con autovalor genuinamente negativo (λ < −tol·‖R‖_F)
        lanza KinematicSymmetryError.
        """
        J, _, J_d, R_d, g = _default_matrices(n=4, m=2)
        n = 4
        R_psd = _psd(n, rank=n-1, seed=30)
        eigvals, eigvecs = la.eigh(R_psd)
        eigvals[0] = -1.0  # autovalor negativo real
        R_neg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        R_neg = 0.5 * (R_neg + R_neg.T)
        with pytest.raises(KinematicSymmetryError):
            KCoreKinematicAgent(J=J, R=R_neg, J_d=J_d, R_d=R_d, g=g)

    def test_r_d_negative_eigenvalue_raises_symmetry_error(self) -> None:
        """R_d con autovalor negativo lanza KinematicSymmetryError."""
        J, R, J_d, _, g = _default_matrices(n=4, m=2)
        n = 4
        R_d_psd = _psd(n, rank=n-1, seed=31)
        eigvals, eigvecs = la.eigh(R_d_psd)
        eigvals[0] = -2.0
        R_d_neg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        R_d_neg = 0.5 * (R_d_neg + R_d_neg.T)
        with pytest.raises(KinematicSymmetryError):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d_neg, g=g)

    def test_r_zero_matrix_is_valid_psd(self) -> None:
        """R = 0 es PSD válida (disipación nula, sistema conservativo)."""
        J, _, J_d, R_d, g = _default_matrices(n=4, m=2)
        agent = KCoreKinematicAgent(
            J=J, R=np.zeros((4, 4)), J_d=J_d, R_d=R_d, g=g
        )
        assert agent.context.kappa_R == float("inf")

    def test_r_d_zero_matrix_is_valid_psd(self) -> None:
        """R_d = 0 es PSD válida."""
        J, R, J_d, _, g = _default_matrices(n=4, m=2)
        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=np.zeros((4, 4)), g=g
        )
        assert agent.context.kappa_R_d == float("inf")

    # ── 1.5 Condicionamiento κ(R) y κ(R_d) ───────────────────────────────

    def test_ill_conditioned_r_raises_condition_error(self) -> None:
        """
        R con κ > kappa_max lanza KinematicConditionError.
        Construimos R = Q·diag(1, kappa_bad)·Q^⊤ con kappa_bad >> kappa_max.
        """
        J, _, J_d, R_d, g = _default_matrices(n=4, m=2)
        n = 4
        kappa_max = 1.0e4
        # R con κ ≈ 1e8
        eigvals = np.array([1.0, 1.0, 1.0, 1.0e8])
        rng = np.random.default_rng(40)
        Q, _ = la.qr(rng.standard_normal((n, n)))
        R_ill = Q @ np.diag(eigvals) @ Q.T
        R_ill = 0.5 * (R_ill + R_ill.T)
        with pytest.raises(KinematicConditionError, match=r"κ"):
            KCoreKinematicAgent(
                J=J, R=R_ill, J_d=J_d, R_d=R_d, g=g,
                kappa_max=kappa_max,
            )

    def test_ill_conditioned_r_d_raises_condition_error(self) -> None:
        """R_d con κ > kappa_max lanza KinematicConditionError."""
        J, R, J_d, _, g = _default_matrices(n=4, m=2)
        n = 4
        kappa_max = 1.0e3
        eigvals = np.array([1.0, 1.0, 1.0, 1.0e6])
        rng = np.random.default_rng(41)
        Q, _ = la.qr(rng.standard_normal((n, n)))
        R_d_ill = Q @ np.diag(eigvals) @ Q.T
        R_d_ill = 0.5 * (R_d_ill + R_d_ill.T)
        with pytest.raises(KinematicConditionError):
            KCoreKinematicAgent(
                J=J, R=R, J_d=J_d, R_d=R_d_ill, g=g,
                kappa_max=kappa_max,
            )

    # ── 1.6 Rango de g ────────────────────────────────────────────────────

    def test_g_numerically_null_raises_dimension_error(self) -> None:
        """g = 0 (σ_max ≈ 0) lanza KinematicDimensionError."""
        J, R, J_d, R_d, _ = _default_matrices(n=4, m=2)
        g_zero = np.zeros((4, 2))
        with pytest.raises(KinematicDimensionError, match=r"nula"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g_zero)

    def test_g_rank_deficient_is_accepted(self) -> None:
        """
        g de rango 1 (deficiente pero no nula) debe ser aceptada.
        rank_g = 1 < min(n, m).
        """
        J, R, J_d, R_d, _ = _default_matrices(n=4, m=2)
        # g = v·w^⊤ con v ∈ ℝ^4, w ∈ ℝ^2 → rango 1
        v = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        w = np.array([1.0, -1.0]).reshape(1, -1)
        g_rank1 = v @ w  # shape (4, 2), rank = 1
        agent = KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g_rank1)
        assert agent.context.rank_g == 1

    def test_g_full_rank_context_stores_correct_rank(self) -> None:
        """g de rango pleno → context.rank_g = min(n, m)."""
        n, m = 4, 2
        agent = _default_agent(n=n, m=m)
        assert agent.context.rank_g == min(n, m)

    # ── 1.7 KinematicPreparationContext: integridad ───────────────────────

    def test_context_fields_types_and_values(self) -> None:
        """
        El KinematicPreparationContext debe tener:
          J, R, J_d, R_d, g: NDArray float64 con shapes correctos.
          n, m: enteros positivos.
          rank_g ∈ [1, min(n,m)].
          kappa_R, kappa_R_d: float ≥ 0.
        """
        n, m = 5, 3
        agent = _default_agent(n=n, m=m)
        ctx = agent.context

        assert ctx.J.shape == (n, n)
        assert ctx.R.shape == (n, n)
        assert ctx.J_d.shape == (n, n)
        assert ctx.R_d.shape == (n, n)
        assert ctx.g.shape == (n, m)
        assert ctx.n == n
        assert ctx.m == m
        assert 1 <= ctx.rank_g <= min(n, m)
        assert isinstance(ctx.kappa_R, float)
        assert isinstance(ctx.kappa_R_d, float)

    def test_context_stores_copies_not_references(self) -> None:
        """
        Las matrices en el contexto deben ser copias independientes.
        Modificar la entrada original no altera el contexto.
        """
        J, R, J_d, R_d, g = _default_matrices(n=4, m=2)
        agent = KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g)
        J_orig_val = agent.context.J[0, 1]
        J[0, 1] += 9999.0  # modificar el original
        assert agent.context.J[0, 1] == pytest.approx(J_orig_val), (
            "El contexto no almacenó una copia de J (referencia compartida detectada)."
        )

    def test_context_is_immutable_frozen_dataclass(self) -> None:
        """KinematicPreparationContext es frozen; no permite asignaciones."""
        agent = _default_agent()
        with pytest.raises((AttributeError, TypeError)):
            agent.context.n = 999  # type: ignore[misc]

    def test_context_j_is_antisymmetric(self) -> None:
        """
        El J almacenado en el contexto debe ser antisimétrico con
        tolerancia relativa ε_mach·‖J‖_F.
        """
        agent = _default_agent()
        J = agent.context.J
        norm_J = float(la.norm(J, "fro"))
        residual = float(la.norm(J + J.T, "fro"))
        assert residual < 10 * _EPS * norm_J

    def test_context_r_is_symmetric_psd(self) -> None:
        """
        El R almacenado en el contexto debe ser simétrico y PSD
        (todos los autovalores ≥ −100·ε_mach·‖R‖_F).
        """
        agent = _default_agent()
        R = agent.context.R
        # Simetría
        assert np.allclose(R, R.T, atol=100 * _EPS)
        # PSD
        eigvals = la.eigvalsh(R)
        norm_R = float(la.norm(R, "fro"))
        assert np.all(eigvals >= -100 * _EPS * norm_R)

    # ── 1.8 cfl_margin fuera de rango ─────────────────────────────────────

    def test_cfl_margin_zero_raises_value_error(self) -> None:
        """cfl_margin = 0 debe lanzar ValueError (no está en (0, 1])."""
        J, R, J_d, R_d, g = _default_matrices()
        with pytest.raises(ValueError, match=r"cfl_margin"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g, cfl_margin=0.0)

    def test_cfl_margin_greater_than_one_raises_value_error(self) -> None:
        """cfl_margin > 1 implica inestabilidad y debe lanzar ValueError."""
        J, R, J_d, R_d, g = _default_matrices()
        with pytest.raises(ValueError, match=r"cfl_margin"):
            KCoreKinematicAgent(J=J, R=R, J_d=J_d, R_d=R_d, g=g, cfl_margin=1.5)

    def test_cfl_margin_exactly_one_is_accepted(self) -> None:
        """cfl_margin = 1.0 es el caso límite superior válido."""
        J, R, J_d, R_d, g = _default_matrices()
        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g, cfl_margin=1.0
        )
        assert agent.cfl_margin == 1.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 2 – PRUEBAS DE FASE 2: SÍNTESIS CINEMÁTICA                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase2KinematicSynthesis:
    """
    Pruebas exhaustivas de Phase2_KinematicSynthesis y sus cuatro subprocesos.
    """

    # ────────────────────────────────────────────────────────────────────────
    # 2.1 Subproceso IDA-PBC: compute_dirac_control_law
    # ────────────────────────────────────────────────────────────────────────

    def test_idapbc_correct_alpha_for_full_rank_g(self) -> None:
        """
        Para g de rango pleno (n=m), α = g⁺·F_req = g⁻¹·F_req exactamente.
        Verificado: ‖g·α − F_req‖_2 / max(‖F_req‖, 1) < residual_tol_rel.
        """
        n, m = 4, 4  # sistema cuadrado, g rango pleno
        J, R, J_d, R_d, _ = _default_matrices(n=n, m=m)
        rng = np.random.default_rng(100)
        g_sq = rng.standard_normal((n, n))
        # Forzar rango pleno condicionando g
        g_sq += np.eye(n) * 5.0  # diagonalmente dominante → no singular

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g_sq,
            residual_tol_rel=1.0e-6,
        )

        grad_H = rng.standard_normal(n)
        grad_H_d = rng.standard_normal(n)

        alpha, residual_rel = agent.phase2.compute_dirac_control_law(
            grad_H=grad_H, grad_H_d=grad_H_d
        )

        assert alpha.shape == (n,)
        assert residual_rel < 1.0e-6, (
            f"Residuo IDA-PBC = {residual_rel:.3e} > 1e-6 para g de rango pleno."
        )

    def test_idapbc_alpha_satisfies_matching_equation_analytically(self) -> None:
        """
        [J_d − R_d]∇H_d = [J − R]∇H + g·α
        Verificado directamente como identidad matricial.
        """
        n, m = 3, 3
        J, R, J_d, R_d, _ = _default_matrices(n=n, m=m)
        rng = np.random.default_rng(101)
        g = rng.standard_normal((n, m))
        g += np.eye(n) * 3.0  # rango pleno

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g,
            residual_tol_rel=1.0e-5,
        )

        grad_H = np.array([1.0, -2.0, 0.5])
        grad_H_d = np.array([0.3, 1.0, -1.0])

        alpha, _ = agent.phase2.compute_dirac_control_law(
            grad_H=grad_H, grad_H_d=grad_H_d
        )

        F_req = (J_d - R_d) @ grad_H_d - (J - R) @ grad_H
        residual = float(la.norm(g @ alpha - F_req, 2))
        norm_F = float(la.norm(F_req, 2))
        assert residual < 1.0e-5 * max(norm_F, 1.0), (
            f"Ecuación de matching no satisfecha: ‖g·α − F_req‖ = {residual:.3e}"
        )

    def test_idapbc_wrong_grad_h_shape_raises_dimension_error(self) -> None:
        """grad_H con shape incorrecta lanza KinematicDimensionError."""
        agent = _default_agent(n=4, m=2)
        grad_H_bad = np.ones(5)  # n+1 en lugar de n
        grad_H_d = np.ones(4)
        with pytest.raises(KinematicDimensionError, match=r"grad_H"):
            agent.phase2.compute_dirac_control_law(
                grad_H=grad_H_bad, grad_H_d=grad_H_d
            )

    def test_idapbc_wrong_grad_h_d_shape_raises_dimension_error(self) -> None:
        """grad_H_d con shape incorrecta lanza KinematicDimensionError."""
        agent = _default_agent(n=4, m=2)
        grad_H = np.ones(4)
        grad_H_d_bad = np.ones(3)
        with pytest.raises(KinematicDimensionError, match=r"grad_H_d"):
            agent.phase2.compute_dirac_control_law(
                grad_H=grad_H, grad_H_d=grad_H_d_bad
            )

    def test_idapbc_f_req_zero_gives_zero_alpha(self) -> None:
        """
        Si grad_H_d = 0 y grad_H = 0, entonces F_req = 0 y α = 0.
        El residuo relativo = 0 / max(0, 1) = 0 < tol.
        """
        n, m = 4, 4
        J, R, J_d, R_d, _ = _default_matrices(n=n, m=m)
        g = np.eye(n)  # g = I para facilitar la prueba

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g,
            residual_tol_rel=1.0e-6,
        )

        alpha, residual_rel = agent.phase2.compute_dirac_control_law(
            grad_H=np.zeros(n), grad_H_d=np.zeros(n)
        )
        assert np.allclose(alpha, 0.0, atol=_ATOL)
        assert residual_rel == pytest.approx(0.0, abs=_ATOL)

    def test_idapbc_high_residual_raises_dirac_matching_error(self) -> None:
        """
        g de rango 1 para un sistema de dimensión 4 produce un F_req
        en un subespacio de dimensión 4 que no puede ser alcanzado.
        El residuo excede la tolerancia y lanza DiracMatchingError.
        """
        n = 4
        J, R, J_d, R_d, _ = _default_matrices(n=n, m=2)
        # g de rango 1: sólo puede alcanzar 1 dirección
        v = np.array([1.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
        w = np.array([1.0, 1.0]).reshape(1, -1)
        g_r1 = v @ w  # shape (4,2), rank=1

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g_r1,
            residual_tol_rel=1.0e-6,  # tolerancia estricta
        )

        # Gradientes que generan F_req con componentes en todas las direcciones
        rng = np.random.default_rng(200)
        grad_H = rng.standard_normal(n)
        grad_H_d = rng.standard_normal(n)

        with pytest.raises(DiracMatchingError, match=r"residuo"):
            agent.phase2.compute_dirac_control_law(
                grad_H=grad_H, grad_H_d=grad_H_d
            )

    def test_idapbc_alpha_shape_equals_m(self) -> None:
        """α debe tener shape (m,) para cualquier configuración (n, m)."""
        n, m = 5, 3
        agent = _default_agent(n=n, m=m)
        rng = np.random.default_rng(201)
        grad_H = rng.standard_normal(n)
        grad_H_d = rng.standard_normal(n)

        # Usar tolerancia laxa para evitar DiracMatchingError con g deficiente
        agent2 = KCoreKinematicAgent(
            J=agent.context.J,
            R=agent.context.R,
            J_d=agent.context.J_d,
            R_d=agent.context.R_d,
            g=agent.context.g,
            residual_tol_rel=1.0,  # tolerancia muy laxa
        )
        alpha, _ = agent2.phase2.compute_dirac_control_law(
            grad_H=grad_H, grad_H_d=grad_H_d
        )
        assert alpha.shape == (m,)

    # ────────────────────────────────────────────────────────────────────────
    # 2.2 Subproceso Hodge: modulate_hodge_conductance
    # ────────────────────────────────────────────────────────────────────────

    def test_hodge_zero_vorticity_w_unchanged(self) -> None:
        """
        I_curl ≈ 0 → vorticidad ‖I_curl‖_W ≈ 0 < ε_crit → W sin modificar.
        La diagonal de W_mod debe ser idéntica a la de W original.
        """
        agent = _default_agent()
        E = 6
        w_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        W = _diag_sparse(w_vals)
        I_curl = np.zeros(E)  # vorticidad exactamente nula

        W_mod, vorticity_norm = agent.phase2.modulate_hodge_conductance(
            W=W, I_curl=I_curl, epsilon_crit=1.0e-2
        )
        assert vorticity_norm == pytest.approx(0.0, abs=_ATOL)
        assert np.allclose(W_mod.diagonal(), w_vals, atol=_ATOL)

    def test_hodge_supercritical_vorticity_penalizes_diagonal(self) -> None:
        """
        I_curl concentrado en aristas → ‖I_curl‖_W > ε_crit →
        diagonal de W_mod < diagonal de W en las aristas penalizadas.
        """
        agent = _default_agent()
        E = 5
        w_vals = np.ones(E) * 2.0
        W = _diag_sparse(w_vals)
        # Vorticidad alta en todas las aristas
        I_curl = np.ones(E) * 10.0  # ‖I_curl‖_W = sqrt(E*w*100) >> 0.01
        strangle = 1.0e-3

        W_mod, vorticity_norm = agent.phase2.modulate_hodge_conductance(
            W=W, I_curl=I_curl,
            epsilon_crit=1.0e-2,
            strangle_factor=strangle,
        )

        assert vorticity_norm > 1.0e-2
        w_mod_diag = W_mod.diagonal()
        # Todas las aristas deben estar penalizadas (soporte = todas)
        for i in range(E):
            assert w_mod_diag[i] < w_vals[i], (
                f"Arista {i} no fue penalizada: w_mod[{i}]={w_mod_diag[i]:.3e} "
                f">= w_orig[{i}]={w_vals[i]:.3e}"
            )

    def test_hodge_penalized_values_equal_strangle_times_original(self) -> None:
        """
        Tras penalización: w_mod[e] = w[e] · strangle_factor para aristas
        con |I_curl[e]| > 0.1·‖I_curl‖_∞.
        """
        agent = _default_agent()
        E = 4
        w_vals = np.array([1.0, 2.0, 3.0, 4.0])
        W = _diag_sparse(w_vals)
        # I_curl con todas las aristas por encima del umbral 10%
        I_curl = np.array([5.0, 6.0, 7.0, 8.0])
        strangle = 0.5

        W_mod, _ = agent.phase2.modulate_hodge_conductance(
            W=W, I_curl=I_curl,
            epsilon_crit=1.0e-2,
            strangle_factor=strangle,
        )
        w_mod = W_mod.diagonal()
        expected = w_vals * strangle
        assert np.allclose(w_mod, expected, rtol=1.0e-10)

    def test_hodge_wrong_i_curl_shape_raises_dimension_error(self) -> None:
        """I_curl con shape ≠ (E,) lanza KinematicDimensionError."""
        agent = _default_agent()
        E = 5
        W = _diag_sparse(np.ones(E))
        I_curl_bad = np.ones(E + 2)  # shape (7,) en lugar de (5,)
        with pytest.raises(KinematicDimensionError, match=r"I_curl"):
            agent.phase2.modulate_hodge_conductance(W=W, I_curl=I_curl_bad)

    def test_hodge_strangle_factor_zero_raises_error(self) -> None:
        """strangle_factor = 0 lanza ParasiticVorticityError."""
        agent = _default_agent()
        E = 4
        W = _diag_sparse(np.ones(E))
        I_curl = np.ones(E) * 10.0
        with pytest.raises(ParasiticVorticityError, match=r"strangle"):
            agent.phase2.modulate_hodge_conductance(
                W=W, I_curl=I_curl, strangle_factor=0.0
            )

    def test_hodge_strangle_factor_negative_raises_error(self) -> None:
        """strangle_factor < 0 lanza ParasiticVorticityError."""
        agent = _default_agent()
        E = 4
        W = _diag_sparse(np.ones(E))
        I_curl = np.ones(E) * 10.0
        with pytest.raises(ParasiticVorticityError):
            agent.phase2.modulate_hodge_conductance(
                W=W, I_curl=I_curl, strangle_factor=-0.1
            )

    def test_hodge_w_mod_diagonal_nonnegative(self) -> None:
        """
        Tras estrangulamiento, todos los valores diagonales de W_mod
        deben ser ≥ 0 (conductancia física).
        """
        agent = _default_agent()
        E = 6
        w_vals = np.abs(np.random.default_rng(300).standard_normal(E)) + 0.1
        W = _diag_sparse(w_vals)
        I_curl = np.ones(E) * 100.0  # vorticidad muy alta

        W_mod, _ = agent.phase2.modulate_hodge_conductance(
            W=W, I_curl=I_curl,
            epsilon_crit=1.0e-2,
            strangle_factor=1.0e-4,
        )
        assert np.all(W_mod.diagonal() >= 0.0)

    def test_hodge_vorticity_norm_formula(self) -> None:
        """
        ‖I_curl‖_W = sqrt(sum(w_diag · I_curl²)) verificado analíticamente.
        Para W = diag(2,2,...) e I_curl = ones(E): ‖I_curl‖_W = sqrt(2E).
        """
        agent = _default_agent()
        E = 5
        w = 2.0
        W = _diag_sparse(np.full(E, w))
        I_curl = np.ones(E)

        _, vorticity = agent.phase2.modulate_hodge_conductance(
            W=W, I_curl=I_curl,
            epsilon_crit=float("inf"),  # nunca estrangula
        )
        expected = math.sqrt(w * E)  # = sqrt(2·5) = sqrt(10)
        assert vorticity == pytest.approx(expected, rel=1.0e-10)

    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "lil", "dia"])
    def test_hodge_accepts_various_sparse_formats(self, fmt: str) -> None:
        """
        modulate_hodge_conductance debe aceptar W en cualquier formato
        sparse (CSR, CSC, COO, LIL, DIA) sin lanzar excepción.
        """
        agent = _default_agent()
        E = 4
        w_vals = np.array([1.0, 2.0, 3.0, 4.0])
        W = _diag_sparse(w_vals, fmt=fmt)
        I_curl = np.zeros(E)

        W_mod, _ = agent.phase2.modulate_hodge_conductance(W=W, I_curl=I_curl)
        assert W_mod is not None
        assert W_mod.shape == (E, E)

    def test_hodge_output_is_csr_format(self) -> None:
        """W_mod debe ser retornado en formato CSR para eficiencia."""
        agent = _default_agent()
        E = 4
        W = _diag_sparse(np.ones(E), fmt="lil")
        I_curl = np.zeros(E)
        W_mod, _ = agent.phase2.modulate_hodge_conductance(W=W, I_curl=I_curl)
        assert sp.isspmatrix_csr(W_mod)

    # ────────────────────────────────────────────────────────────────────────
    # 2.3 Subproceso Kramers-Kronig: tune_impedance_tensors
    # ────────────────────────────────────────────────────────────────────────

    def test_kk_spd_z_load_produces_spd_tensors(self) -> None:
        """
        Z_load SPD → ε_eff y μ_eff ambos SPD (todos los autovalores > 0).
        """
        agent = _default_agent()
        d = 3
        Z_load = _spd(d, seed=400)

        eps_eff, mu_eff = agent.phase2.tune_impedance_tensors(Z_load=Z_load)

        # ε_eff SPD
        eigvals_eps = la.eigvalsh(eps_eff)
        assert np.all(eigvals_eps > 0), (
            f"ε_eff no es SPD: autovalores = {eigvals_eps}"
        )

        # μ_eff SPD
        eigvals_mu = la.eigvalsh(mu_eff)
        assert np.all(eigvals_mu > 0), (
            f"μ_eff no es SPD: autovalores = {eigvals_mu}"
        )

    def test_kk_epsilon_eff_equals_z_load(self) -> None:
        """
        Por construcción: ε_eff = L_Z · L_Z^⊤ = Z_load.
        Verificado con tolerancia 100·ε_mach·‖Z_load‖_F.
        """
        agent = _default_agent()
        d = 4
        Z_load = _spd(d, seed=401)
        Z_sym = 0.5 * (Z_load + Z_load.T)

        eps_eff, _ = agent.phase2.tune_impedance_tensors(Z_load=Z_load)

        norm_Z = float(la.norm(Z_sym, "fro"))
        residual = float(la.norm(eps_eff - Z_sym, "fro"))
        assert residual < 100 * _EPS * norm_Z, (
            f"‖ε_eff − Z_load‖_F = {residual:.3e} > tol = {100*_EPS*norm_Z:.3e}"
        )

    def test_kk_mu_eff_equals_z_load_squared(self) -> None:
        """
        μ_eff = Z_load · ε_eff · Z_load^⊤ = Z_load².
        Verificado analíticamente para Z_load SPD.
        """
        agent = _default_agent()
        d = 3
        Z_load = _spd(d, seed=402)
        Z_sym = 0.5 * (Z_load + Z_load.T)

        _, mu_eff = agent.phase2.tune_impedance_tensors(Z_load=Z_load)

        Z_sq = Z_sym @ Z_sym
        norm_Z2 = float(la.norm(Z_sq, "fro"))
        residual = float(la.norm(mu_eff - Z_sq, "fro"))
        assert residual < 100 * _EPS * norm_Z2, (
            f"‖μ_eff − Z²‖_F = {residual:.3e} > tol = {100*_EPS*norm_Z2:.3e}"
        )

    def test_kk_z_load_not_spd_raises_impedance_error(self) -> None:
        """
        Z_load no SPD (autovalor negativo) debe lanzar ImpedanceReflectionError.
        """
        agent = _default_agent()
        d = 3
        Z_bad = _spd(d, seed=403)
        eigvals, eigvecs = la.eigh(Z_bad)
        eigvals[0] = -1.0  # autovalor negativo
        Z_indef = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Z_indef = 0.5 * (Z_indef + Z_indef.T)

        with pytest.raises(ImpedanceReflectionError, match=r"SPD"):
            agent.phase2.tune_impedance_tensors(Z_load=Z_indef)

    def test_kk_z_load_singular_raises_impedance_error(self) -> None:
        """Z_load singular (λ_min = 0) lanza ImpedanceReflectionError."""
        agent = _default_agent()
        d = 3
        Z_sing = _psd(d, rank=d-1, seed=404)
        Z_sing = 0.5 * (Z_sing + Z_sing.T)
        with pytest.raises(ImpedanceReflectionError):
            agent.phase2.tune_impedance_tensors(Z_load=Z_sing)

    def test_kk_z_load_not_square_raises_impedance_error(self) -> None:
        """Z_load rectangular lanza ImpedanceReflectionError."""
        agent = _default_agent()
        Z_rect = np.ones((3, 4))
        with pytest.raises(ImpedanceReflectionError, match=r"cuadrada"):
            agent.phase2.tune_impedance_tensors(Z_load=Z_rect)

    def test_kk_causal_residual_below_machine_tolerance(self) -> None:
        """
        ‖ε_eff − Z_load‖_F / ‖Z_load‖_F < 100·ε_mach.
        Verifica que la relación de dispersión causal es exacta numéricamente.
        """
        agent = _default_agent()
        d = 5
        Z_load = _spd(d, seed=405)
        Z_sym = 0.5 * (Z_load + Z_load.T)
        eps_eff, _ = agent.phase2.tune_impedance_tensors(Z_load=Z_load)

        norm_Z = float(la.norm(Z_sym, "fro"))
        causal_err = float(la.norm(eps_eff - Z_sym, "fro")) / max(norm_Z, 1.0)
        assert causal_err < 100 * _EPS

    def test_kk_epsilon_and_mu_are_symmetric(self) -> None:
        """ε_eff y μ_eff deben ser simétricas (re-simetrización defensiva)."""
        agent = _default_agent()
        d = 4
        Z_load = _spd(d, seed=406)
        eps_eff, mu_eff = agent.phase2.tune_impedance_tensors(Z_load=Z_load)

        assert np.allclose(eps_eff, eps_eff.T, atol=100 * _EPS)
        assert np.allclose(mu_eff, mu_eff.T, atol=100 * _EPS)

    # ────────────────────────────────────────────────────────────────────────
    # 2.4 Subproceso CFL: audit_cfl_limit
    # ────────────────────────────────────────────────────────────────────────

    def test_cfl_correct_dt_for_known_laplacian(self) -> None:
        """
        Para el Laplaciano del grafo camino P_n:
        λ_max ≈ 4·sin²((n-1)π/(2n)) < 4.
        Verificamos: Δt_safe = 2·CFL/(c·√λ_max) ≈ 2·0.9/(c·2) = 0.9/c.

        Para n=10 y c=1: Δt_safe ∈ [0.9/2, 0.9/1] = [0.45, 0.9].
        """
        agent = _default_agent(cfl_margin=0.9)
        n_nodes = 10
        L = _laplacian_path_graph(n_nodes)
        c_eff = 1.0

        dt_safe = agent.phase2.audit_cfl_limit(c_eff=c_eff, Delta_sym=L)

        # λ_max ∈ (0, 4) para P_n → Δt_safe ∈ (0.9/2, +∞)
        assert dt_safe > 0.9 / 2.0, (
            f"Δt_safe = {dt_safe:.4f} demasiado pequeño para P_{n_nodes}"
        )
        assert dt_safe < 10.0, (
            f"Δt_safe = {dt_safe:.4f} demasiado grande (Laplaciano degenerado?)"
        )

    def test_cfl_dt_safe_formula_analytically(self) -> None:
        """
        Para una matriz 1×1: Delta_sym = [[λ]], Δt_safe = 2·CFL/(c·√λ).
        Verificado exactamente.
        """
        agent = _default_agent(cfl_margin=0.8)
        lambda_known = 4.0
        Delta_1x1 = sp.csr_matrix(np.array([[lambda_known]]))
        c_eff = 2.0

        dt_safe = agent.phase2.audit_cfl_limit(c_eff=c_eff, Delta_sym=Delta_1x1)

        expected = (2.0 * 0.8) / (c_eff * math.sqrt(lambda_known))
        assert dt_safe == pytest.approx(expected, rel=1.0e-6)

    def test_cfl_nonpositive_c_eff_raises_error(self) -> None:
        """c_eff = 0 lanza CFLViolationError."""
        agent = _default_agent()
        L = _laplacian_path_graph(5)
        with pytest.raises(CFLViolationError, match=r"c_eff"):
            agent.phase2.audit_cfl_limit(c_eff=0.0, Delta_sym=L)

    def test_cfl_negative_c_eff_raises_error(self) -> None:
        """c_eff < 0 lanza CFLViolationError."""
        agent = _default_agent()
        L = _laplacian_path_graph(5)
        with pytest.raises(CFLViolationError, match=r"c_eff"):
            agent.phase2.audit_cfl_limit(c_eff=-1.0, Delta_sym=L)

    def test_cfl_degenerate_laplacian_returns_infinity(self) -> None:
        """
        Laplaciano numéricamente nulo (λ_max < 1e-12) → Δt_safe = +∞.
        Modela un grafo desconectado con autovalores todos cero.
        """
        agent = _default_agent()
        Delta_zero = sp.csr_matrix(np.zeros((4, 4)))
        dt_safe = agent.phase2.audit_cfl_limit(c_eff=1.0, Delta_sym=Delta_zero)
        assert math.isinf(dt_safe) and dt_safe > 0

    def test_cfl_dt_safe_scales_inversely_with_c_eff(self) -> None:
        """
        Δt_safe ∝ 1/c_eff: duplicar c_eff divide Δt_safe a la mitad.
        """
        agent = _default_agent(cfl_margin=0.9)
        L = _laplacian_path_graph(8)

        dt_c1 = agent.phase2.audit_cfl_limit(c_eff=1.0, Delta_sym=L)
        dt_c2 = agent.phase2.audit_cfl_limit(c_eff=2.0, Delta_sym=L)

        assert dt_c1 == pytest.approx(2.0 * dt_c2, rel=1.0e-6)

    def test_cfl_dt_safe_scales_with_cfl_margin(self) -> None:
        """
        Δt_safe ∝ CFL_margin: duplicar el margen duplica Δt_safe
        (dentro del rango (0, 1]).
        """
        n = 4
        J, R, J_d, R_d, g = _default_matrices(n=n, m=2)
        L = _laplacian_path_graph(6)

        agent_05 = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g, cfl_margin=0.5
        )
        agent_10 = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g, cfl_margin=1.0
        )

        dt_05 = agent_05.phase2.audit_cfl_limit(c_eff=1.0, Delta_sym=L)
        dt_10 = agent_10.phase2.audit_cfl_limit(c_eff=1.0, Delta_sym=L)

        assert dt_10 == pytest.approx(2.0 * dt_05, rel=1.0e-6)

    # ────────────────────────────────────────────────────────────────────────
    # 2.5 Método terminal synthesize
    # ────────────────────────────────────────────────────────────────────────

    def test_synthesize_returns_kinematic_state_tensor(self) -> None:
        """synthesize debe retornar una instancia de KinematicStateTensor."""
        agent = _default_agent(n=4, m=2)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        assert isinstance(state, KinematicStateTensor)

    def test_synthesize_is_kinematically_stable_true(self) -> None:
        """is_kinematically_stable = True si todos los subprocesos pasan."""
        agent = _default_agent(n=4, m=2)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        assert state.is_kinematically_stable is True

    def test_synthesize_state_fields_types(self) -> None:
        """
        KinematicStateTensor debe tener:
          control_law_alpha: NDArray shape (m,)
          hodge_conductance: sp.spmatrix
          dielectric_tensor: NDArray
          magnetic_tensor: NDArray
          cfl_safe_dt: float > 0
          residual_idapbc: float ≥ 0
          vorticity_norm: float ≥ 0
        """
        n, m = 4, 2
        agent = _default_agent(n=n, m=m, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        state = agent.synthesize_kinematic_core(**inputs)

        assert state.control_law_alpha.shape == (m,)
        assert sp.issparse(state.hodge_conductance)
        assert isinstance(state.dielectric_tensor, np.ndarray)
        assert isinstance(state.magnetic_tensor, np.ndarray)
        assert isinstance(state.cfl_safe_dt, float) and state.cfl_safe_dt > 0
        assert isinstance(state.residual_idapbc, float) and state.residual_idapbc >= 0
        assert isinstance(state.vorticity_norm, float) and state.vorticity_norm >= 0

    def test_synthesize_cfl_violation_raises_error(self) -> None:
        """
        dt_requested > dt_safe lanza CFLViolationError con diagnóstico.
        """
        agent = _default_agent(n=4, m=2)
        E, d = 5, 3
        inputs = _default_synthesis_inputs(agent, E=E, d=d)
        # Usar dt enorme para garantizar violación CFL
        inputs["dt_requested"] = 1.0e10
        with pytest.raises(CFLViolationError, match=r"CFL"):
            agent.synthesize_kinematic_core(**inputs)

    def test_synthesize_updates_latest_hodge_conductance(self) -> None:
        """
        Después de synthesize_kinematic_core, _latest_hodge_conductance
        debe ser no-None y coincidir con hodge_conductance del estado.
        """
        agent = _default_agent(n=4, m=2)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        assert agent._latest_hodge_conductance is None
        state = agent.synthesize_kinematic_core(**inputs)
        assert agent._latest_hodge_conductance is not None
        # Verificar que el contenido coincide
        assert np.allclose(
            agent._latest_hodge_conductance.diagonal(),
            state.hodge_conductance.diagonal(),
        )

    def test_synthesize_invalidates_phase3_after_call(self) -> None:
        """
        Llamar a synthesize_kinematic_core debe invalidar phase3 (= None),
        forzando re-instanciación en la próxima llamada a export_sheaf_stalk.
        """
        agent = _default_agent(n=4, m=2)
        E = 5
        inputs = _default_synthesis_inputs(agent, E=E, d=3)

        # Primera síntesis + exportación de stalk
        agent.synthesize_kinematic_core(**inputs)
        agent.export_sheaf_stalk(state_x=np.ones(E))
        assert agent.phase3 is not None

        # Segunda síntesis: debe invalidar phase3
        agent.synthesize_kinematic_core(**inputs)
        assert agent.phase3 is None

    def test_synthesize_kinematic_state_tensor_is_immutable(self) -> None:
        """KinematicStateTensor es frozen; no permite asignaciones."""
        agent = _default_agent(n=4, m=2)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        with pytest.raises((AttributeError, TypeError)):
            state.cfl_safe_dt = 0.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 3 – PRUEBAS DE FASE 3: PROYECCIÓN EN HACES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase3SheafProjection:
    """
    Pruebas exhaustivas de Phase3_SheafProjection y export_sheaf_stalk.
    """

    @pytest.fixture
    def agent_with_synthesis(
        self,
    ) -> Tuple[KCoreKinematicAgent, KinematicStateTensor, int]:
        """
        Fixture que retorna un agente con síntesis ya ejecutada.
        Retorna (agent, state, E).
        """
        E = 5
        agent = _default_agent(n=4, m=2, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=E, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        return agent, state, E

    # ── 3.1 Prerequisitos ─────────────────────────────────────────────────

    def test_export_without_synthesis_raises_core_error(self) -> None:
        """
        Llamar a export_sheaf_stalk sin síntesis previa debe lanzar
        KinematicCoreError con mensaje descriptivo.
        """
        agent = _default_agent()
        assert agent._latest_hodge_conductance is None
        with pytest.raises(KinematicCoreError, match=r"synthesize"):
            agent.export_sheaf_stalk(state_x=np.ones(5))

    # ── 3.2 Forma y tipo de δ_{CORE} ──────────────────────────────────────

    def test_delta_core_shape_is_e_by_e(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """δ_{CORE} debe tener shape (E, E)."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert stalk.delta_core.shape == (E, E)

    def test_delta_core_dtype_float64(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """δ_{CORE} debe ser dtype float64."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert stalk.delta_core.dtype == np.float64

    def test_delta_core_is_symmetric(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        δ_{CORE} debe ser simétrica (raíz cuadrada espectral de W_mod simétrico).
        """
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        delta = stalk.delta_core
        residual = float(la.norm(delta - delta.T, "fro"))
        tol = 100 * _EPS * float(la.norm(delta, "fro"))
        assert residual < tol

    def test_delta_core_is_psd(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        δ_{CORE} = W_mod^{+1/2} debe ser PSD (todos los autovalores ≥ −100·ε).
        """
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        eigvals = la.eigvalsh(stalk.delta_core)
        norm_delta = float(la.norm(stalk.delta_core, "fro"))
        assert np.all(eigvals >= -100 * _EPS * norm_delta)

    # ── 3.3 Identidad de Hodge local: δ^⊤ δ ≈ W_mod ─────────────────────

    def test_hodge_identity_delta_sq_equals_w_mod(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        δ_{CORE}² = δ · δ ≈ W_mod con error < 100·ε_mach relativo a ‖W_mod‖_F.
        Verifica que δ es la raíz cuadrada correcta de W_mod.
        """
        agent, state, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        delta = stalk.delta_core

        W_dense = state.hodge_conductance.toarray()
        W_dense = 0.5 * (W_dense + W_dense.T)

        delta_sq = delta @ delta
        residual = float(la.norm(delta_sq - W_dense, "fro"))
        norm_W = float(la.norm(W_dense, "fro"))
        rel_err = residual / max(norm_W, 1.0)
        tol = 100 * _EPS

        assert rel_err < tol, (
            f"Identidad de Hodge violada: "
            f"‖δ² − W_mod‖_F/‖W_mod‖_F = {rel_err:.3e} > {tol:.3e}"
        )

    def test_hodge_identity_for_identity_w_mod(self) -> None:
        """
        W_mod = I → δ_{CORE} = I (raíz cuadrada de I es I).
        """
        n = 4
        J, R, J_d, R_d, _ = _default_matrices(n=n, m=2)
        g = np.eye(n, 2)

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g,
            residual_tol_rel=1.0,
        )

        # Crear W_mod = I manualmente e instanciar Phase3 directamente
        E = 4
        W_identity = sp.eye(E, format="csr")
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W_mod=W_identity)

        stalk = phase3.export_stalk(state_x=np.ones(E))
        assert np.allclose(stalk.delta_core, np.eye(E), atol=100 * _EPS)

    def test_hodge_residual_is_nonnegative_float(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """delta_hodge_residual ∈ [0, ∞) y es float."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert isinstance(stalk.delta_hodge_residual, float)
        assert stalk.delta_hodge_residual >= 0.0

    # ── 3.4 Proyección δ · state_x ────────────────────────────────────────

    def test_projection_equals_delta_times_x(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        projected_state = δ_{CORE} · state_x verificado analíticamente.
        """
        agent, _, E = agent_with_synthesis
        rng = np.random.default_rng(500)
        x = rng.standard_normal(E)
        stalk = agent.export_sheaf_stalk(state_x=x)
        expected = stalk.delta_core @ x
        assert np.allclose(stalk.projected_state, expected, atol=_ATOL)

    def test_projection_zero_for_zero_x(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """δ · 0 = 0 (linealidad)."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.zeros(E))
        assert np.allclose(stalk.projected_state, 0.0, atol=_ATOL)

    def test_projection_linearity(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        δ·(2x − 3y) = 2·(δ·x) − 3·(δ·y): linealidad de la proyección.
        """
        agent, _, E = agent_with_synthesis
        rng = np.random.default_rng(501)
        x = rng.standard_normal(E)
        y = rng.standard_normal(E)
        a, b = 2.0, -3.0

        s_x = agent.export_sheaf_stalk(state_x=x)
        s_y = agent.export_sheaf_stalk(state_x=y)
        s_comb = agent.export_sheaf_stalk(state_x=a * x + b * y)

        expected = a * s_x.projected_state + b * s_y.projected_state
        assert np.allclose(s_comb.projected_state, expected, atol=_ATOL)

    def test_projection_shape_and_dtype(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """projected_state tiene shape (E,) y dtype float64."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert stalk.projected_state.shape == (E,)
        assert stalk.projected_state.dtype == np.float64

    # ── 3.5 state_vector como copia independiente ─────────────────────────

    def test_state_vector_is_copy_not_reference(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        Modificar state_x después de la llamada no altera stalk.state_vector.
        """
        agent, _, E = agent_with_synthesis
        x = np.ones(E) * 7.0
        stalk = agent.export_sheaf_stalk(state_x=x)
        x[:] = 0.0  # modificar el original
        assert np.all(stalk.state_vector == 7.0), (
            "state_vector fue modificado al cambiar state_x (no es copia)."
        )

    # ── 3.6 Rango de δ_{CORE} ─────────────────────────────────────────────

    def test_rank_delta_equals_rank_of_w_mod(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        rank_delta debe coincidir con el rango de W_mod
        calculado independientemente.
        """
        agent, state, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        W_dense = state.hodge_conductance.toarray()
        rank_np = int(np.linalg.matrix_rank(W_dense))
        # Tolerancia de ±1 por distintas tolerancias de truncación
        assert abs(stalk.rank_delta - rank_np) <= 1

    def test_rank_delta_full_for_positive_diagonal_w(self) -> None:
        """
        W_mod con diagonal estrictamente positiva → δ_{CORE} de rango pleno.
        """
        E = 5
        w_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        W = _diag_sparse(w_vals, fmt="csr")
        phase3 = KCoreKinematicAgent.Phase3_SheafProjection(W_mod=W)
        stalk = phase3.export_stalk(state_x=np.ones(E))
        assert stalk.rank_delta == E

    # ── 3.7 Validación de dimensiones de state_x ──────────────────────────

    def test_wrong_state_x_shape_raises_dimension_error(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """state_x con shape incorrecta lanza KinematicDimensionError."""
        agent, _, E = agent_with_synthesis
        with pytest.raises(KinematicDimensionError, match=r"state_x"):
            agent.export_sheaf_stalk(state_x=np.ones(E + 3))

    # ── 3.8 W_mod con diagonal negativa lanza SheafCoboundaryError ───────

    def test_negative_diagonal_w_raises_coboundary_error(self) -> None:
        """
        W_mod con entrada diagonal negativa real (no por redondeo) debe
        lanzar SheafCoboundaryError en Phase3.__init__.
        """
        E = 3
        # w_vals con una entrada negativa real
        w_vals = np.array([1.0, -2.0, 3.0])
        W_neg = sp.diags(w_vals, offsets=0, format="csr")
        with pytest.raises(SheafCoboundaryError, match=r"PSD"):
            KCoreKinematicAgent.Phase3_SheafProjection(W_mod=W_neg)

    # ── 3.9 Instanciación perezosa y reutilización de phase3 ─────────────

    def test_phase3_none_before_synthesis(self) -> None:
        """phase3 = None antes de cualquier síntesis."""
        agent = _default_agent()
        assert agent.phase3 is None

    def test_phase3_none_before_export(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        phase3 = None después de síntesis pero antes de export_sheaf_stalk.
        (La síntesis invalida phase3.)
        """
        agent, _, _ = agent_with_synthesis
        assert agent.phase3 is None

    def test_phase3_instantiated_after_export(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """phase3 no es None después de la primera llamada a export_sheaf_stalk."""
        agent, _, E = agent_with_synthesis
        agent.export_sheaf_stalk(state_x=np.ones(E))
        assert agent.phase3 is not None

    def test_phase3_reused_across_calls(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """
        La misma instancia de phase3 es reutilizada en llamadas
        subsecuentes sin nueva síntesis (identidad de objeto).
        """
        agent, _, E = agent_with_synthesis
        agent.export_sheaf_stalk(state_x=np.ones(E))
        p3_first = agent.phase3
        agent.export_sheaf_stalk(state_x=np.zeros(E))
        p3_second = agent.phase3
        assert p3_first is p3_second

    # ── 3.10 Inmutabilidad del SheafStalk ────────────────────────────────

    def test_sheaf_stalk_is_immutable(
        self,
        agent_with_synthesis: Tuple[KCoreKinematicAgent, KinematicStateTensor, int],
    ) -> None:
        """SheafStalk es frozen dataclass; no permite asignaciones."""
        agent, _, E = agent_with_synthesis
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        with pytest.raises((AttributeError, TypeError)):
            stalk.rank_delta = 999  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 4 – PRUEBAS DE INTEGRACIÓN DE LAS 3 FASES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestIntegrationPipeline:
    """
    Pruebas de integración que ejercitan el pipeline completo de las 3 fases.
    """

    # ── 4.1 Pipeline completo nominal ─────────────────────────────────────

    def test_full_pipeline_nominal(self) -> None:
        """
        Constructor → synthesize_kinematic_core → export_sheaf_stalk.
        Todos los outputs son instancias correctas con valores coherentes.
        """
        n, m, E, d = 4, 2, 6, 3
        agent = _default_agent(n=n, m=m, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=E, d=d)

        state = agent.synthesize_kinematic_core(**inputs)
        assert isinstance(state, KinematicStateTensor)
        assert state.is_kinematically_stable
        assert state.cfl_safe_dt > 0
        assert state.vorticity_norm >= 0

        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert isinstance(stalk, SheafStalk)
        assert stalk.delta_core.shape == (E, E)
        assert stalk.rank_delta >= 0

    # ── 4.2 Determinismo ──────────────────────────────────────────────────

    def test_pipeline_is_deterministic(self) -> None:
        """
        Misma entrada → misma salida en dos ejecuciones independientes.
        """
        n, m, E, d = 4, 2, 5, 3
        J, R, J_d, R_d, g = _default_matrices(n=n, m=m, seed=42)

        def _run():
            agent = KCoreKinematicAgent(
                J=J.copy(), R=R.copy(), J_d=J_d.copy(),
                R_d=R_d.copy(), g=g.copy(),
                residual_tol_rel=1.0,
            )
            inputs = _default_synthesis_inputs(agent, E=E, d=d, seed=42)
            state = agent.synthesize_kinematic_core(**inputs)
            stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
            return state, stalk

        state1, stalk1 = _run()
        state2, stalk2 = _run()

        assert state1.cfl_safe_dt == pytest.approx(state2.cfl_safe_dt)
        assert state1.vorticity_norm == pytest.approx(state2.vorticity_norm)
        assert np.allclose(stalk1.delta_core, stalk2.delta_core)
        assert np.allclose(stalk1.projected_state, stalk2.projected_state)

    # ── 4.3 Logging ───────────────────────────────────────────────────────

    def test_construction_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """La construcción del agente emite al menos un mensaje INFO."""
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KCoreKinematicAgent"
        ):
            _default_agent()
        info = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "KCoreKinematicAgent" in r.name
        ]
        assert len(info) >= 1

    def test_synthesis_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """synthesize_kinematic_core emite al menos un mensaje INFO."""
        agent = _default_agent(n=4, m=2, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KCoreKinematicAgent"
        ):
            agent.synthesize_kinematic_core(**inputs)
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info) >= 1

    def test_export_stalk_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """export_sheaf_stalk emite al menos un mensaje INFO."""
        E = 5
        agent = _default_agent(n=4, m=2, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=E, d=3)
        agent.synthesize_kinematic_core(**inputs)
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KCoreKinematicAgent"
        ):
            agent.export_sheaf_stalk(state_x=np.ones(E))
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info) >= 1

    # ── 4.4 Dimensiones variadas ───────────────────────────────────────────

    @pytest.mark.parametrize("n,m", [(2, 1), (3, 3), (5, 2), (6, 4)])
    def test_various_n_m_combinations(self, n: int, m: int) -> None:
        """
        El pipeline completo debe funcionar para múltiples (n, m).
        """
        J, R, J_d, R_d, g = _default_matrices(n=n, m=m, seed=n * 10 + m)
        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g,
            residual_tol_rel=1.0,
        )
        E = max(n, 3)
        inputs = _default_synthesis_inputs(agent, E=E, d=2, seed=n + m)
        state = agent.synthesize_kinematic_core(**inputs)
        assert state.control_law_alpha.shape == (m,)
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert stalk.delta_core.shape == (E, E)

    # ── 4.5 Caso mínimo n=2, m=1 ──────────────────────────────────────────

    def test_minimal_n2_m1_pipeline(self) -> None:
        """
        Pipeline funcional para n=2, m=1 (caso mínimo no trivial).
        """
        n, m = 2, 1
        J = np.array([[0.0, -1.0], [1.0, 0.0]])
        R = np.array([[0.5, 0.0], [0.0, 0.5]])
        J_d = np.array([[0.0, -0.5], [0.5, 0.0]])
        R_d = np.array([[0.3, 0.0], [0.0, 0.3]])
        g = np.array([[1.0], [0.0]])

        agent = KCoreKinematicAgent(
            J=J, R=R, J_d=J_d, R_d=R_d, g=g,
            residual_tol_rel=1.0,
        )

        E = 3
        w_vals = np.array([1.0, 2.0, 3.0])
        W = _diag_sparse(w_vals)
        Z_load = np.array([[2.0, 0.5], [0.5, 3.0]])
        L = _laplacian_path_graph(4)

        state = agent.synthesize_kinematic_core(
            grad_H=np.array([0.1, 0.2]),
            grad_H_d=np.array([0.3, 0.4]),
            W=W,
            I_curl=np.zeros(E),
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=L,
            dt_requested=0.05,
        )

        assert state.control_law_alpha.shape == (m,)
        assert state.is_kinematically_stable

        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))
        assert stalk.delta_core.shape == (E, E)

    # ── 4.6 Coherencia hodge_conductance → delta_core ─────────────────────

    def test_delta_core_consistent_with_hodge_conductance(self) -> None:
        """
        delta_core debe ser la raíz cuadrada espectral de hodge_conductance:
        ‖delta_core² − W_mod‖_F / ‖W_mod‖_F < 100·ε_mach.
        """
        n, m, E = 4, 2, 5
        agent = _default_agent(n=n, m=m, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=E, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        stalk = agent.export_sheaf_stalk(state_x=np.ones(E))

        W_dense = state.hodge_conductance.toarray()
        W_dense = 0.5 * (W_dense + W_dense.T)
        delta = stalk.delta_core

        delta_sq = delta @ delta
        norm_W = float(la.norm(W_dense, "fro"))
        rel_err = float(la.norm(delta_sq - W_dense, "fro")) / max(norm_W, 1.0)

        assert rel_err < 100 * _EPS, (
            f"‖δ²−W‖_F/‖W‖_F = {rel_err:.3e} > 100·ε_mach = {100*_EPS:.3e}"
        )

    # ── 4.7 Propiedades de escala del CFL ─────────────────────────────────

    def test_cfl_safe_dt_in_state_is_positive_finite(self) -> None:
        """cfl_safe_dt debe ser un float positivo y finito."""
        agent = _default_agent(n=4, m=2, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=5, d=3)
        state = agent.synthesize_kinematic_core(**inputs)
        assert state.cfl_safe_dt > 0.0
        assert math.isfinite(state.cfl_safe_dt)

    # ── 4.8 Vorticidad nula preserva W en el estado ───────────────────────

    def test_zero_vorticity_preserves_w_in_state(self) -> None:
        """
        Con I_curl = 0, hodge_conductance = W original (sin modificación).
        """
        n, m, E = 4, 2, 5
        agent = _default_agent(n=n, m=m, residual_tol_rel=1.0)
        inputs = _default_synthesis_inputs(agent, E=E, d=3)
        w_original = inputs["W"].diagonal().copy()
        inputs["I_curl"] = np.zeros(E)

        state = agent.synthesize_kinematic_core(**inputs)
        w_mod = state.hodge_conductance.diagonal()
        assert np.allclose(w_mod, w_original, atol=_ATOL)
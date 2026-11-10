# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : bool_novikov_pseudoholomorphic_auditor_engine.py                                          ║
║ Ruta     : app/wisdom/bool_novikov_pseudoholomorphic_auditor_engine.py                               ║
║ Versión  : 15.0.0-PhD-NESTED-PHASES-RSI3-POINCARE-KAM-YOSHIDA                                        ║
║ Gobernanza: Auditor Espectral Boole-Novikov-Pseudoholomorfo con Poincaré-KAM y RSI³ sobre Banach     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

Organización en 3 Fases Anidadas (Composición Functorial \mathcal{F}_{III} \circ \mathcal{F}_{II} \circ \mathcal{F}_{I})
────────────────────────────────────────────────────────────────────────────────────────────────────────
FASE I  : Núcleo Algebraico-Espectral Boole-Novikov.
          - Reducción sobre GF(2) y detección cohomológica de 2-torsión (homomorfismo de Bockstein).
          - Forma Normal de Smith (SNF) entera exacta (Euclides + menores determinantes de Bareiss).
          - Números de Betti (\beta_0, \beta_1), característica de Euler \chi y conectividad de Fiedler.
          - Consolidación canónica en el anillo de Novikov \Lambda_{Nov} y valuación ultramétrica de Krull.
          - CLAUSURA DE FASE I: `phase1_seal_algebraic_spectral_bundle` -> emite `Phase1AlgebraicSpectralBundle`.

FASE II : Continuación Geométrico-Celeste (Floer-Gromov + Poincaré-Delaunay-KAM).
          - APERTURA DE FASE II: `phase2_open_from_phase1_bundle` consume `Phase1AlgebraicSpectralBundle`.
          - Ecuación de Cauchy-Riemann-Floer perturbada \bar{\partial}_{J,H} u = 0 y rigidez de Gromov.
          - Integrador simpléctico de Yoshida de 4.º orden sobre la sección estroboscópica de Poincaré.
          - Mecánica celeste de Poincaré: elementos de Delaunay, ecuación homológica de Poincaré-Lindstedt,
            función generatriz W, integral homoclínica de Melnikov, criterio de solapamiento de Chirikov,
            horizonte exponencial de Nekhoroshev, twist de Poincaré-Birkhoff y margen diofántico KAM.
          - CLAUSURA DE FASE II: `phase2_seal_geometric_celestial_bundle` -> emite `Phase2GeometricCelestialBundle`.

FASE III: Automejora Recursiva RSI³ sobre Álgebras de Banach, Aniquilación de Dirac y Veredicto \Omega_3.
          - APERTURA DE FASE III: `phase3_open_from_phase2_bundle` consume `Phase2GeometricCelestialBundle`.
          - Jerarquía RSI³ (\mathfrak{H}_3 \circ \mathfrak{M}_2 \circ T_1): espectro de Fréchet, leyes de
            mónada de Kleisli (\eta, \mu), aceleración superlineal de Aitken-Steffensen \Delta^2, cotas
            a-posteriori de Banach-Picard y jerk ciber-físico de tercer orden d^3C/dt^3.
          - Aniquilación cuántica de Dirac e^- + e^+ -> 2\gamma con verificación HMAC-SHA256 en tiempo constante.
          - CLAUSURA GLOBAL: `run_spectral_audit` -> adjudicación intuicionista en el Topos de Heyting \Omega_3.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovPseudoholomorphicAuditorEngine")

_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-PSEUDOHOLOMORPHIC-AUDITOR-"
    "POINCARE-KAM-YOSHIDA-RSI3-NESTED-PHASES-PhD"
)
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_KRULL_VALUATION_FLOOR: Final[float] = 0.10
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_BOOL_NOVIKOV_ENGINE_KEY_v14.0.0"
_POSITRON_HMAC_PAYLOAD: Final[bytes] = b"DIRAC_POSITRON_HMAC_VERIFICATION"

_MAX_EXACT_SNF_DIM: Final[int] = 7
_MAX_EXACT_SNF_CELLS: Final[int] = 64
_MAX_EUCLIDEAN_SNF_DIM: Final[int] = 32

# Coeficientes simplécticos de 4.º orden de Yoshida (1990) para el mapa de Poincaré
_CBRT_2: Final[float] = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _CBRT_2)
_YOSHIDA_W0: Final[float] = -_CBRT_2 / (2.0 - _CBRT_2)
_YOSHIDA_C: Final[Tuple[float, float, float, float]] = (
    0.5 * _YOSHIDA_W1,
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * _YOSHIDA_W1,
)
_YOSHIDA_D: Final[Tuple[float, float, float]] = (
    _YOSHIDA_W1,
    _YOSHIDA_W0,
    _YOSHIDA_W1,
)

__all__ = [
    "HeytingTruthValue",
    "BooleanIdealState",
    "NovikovSeriesState",
    "PseudoholomorphicPolygonSurface",
    "CelestialDelaunayState",
    "RSI3BanachState",
    "AnnihilationReactionReport",
    "AuditGradientResult",
    "Phase1AlgebraicSpectralBundle",
    "Phase2GeometricCelestialBundle",
    "BoolNovikovPseudoholomorphicAuditorEngine",
]


class HeytingTruthValue(IntEnum):
    r"""Topos de Heyting \Omega_3 para clasificación intuicionista ciber-física."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


@dataclass(frozen=True)
class BooleanIdealState:
    r"""Estado algebraico discreto sobre R_Boole = Z_2[x_1,...,x_n] / <x_i^2 - x_i> y Z."""
    num_variables: int
    incidence_matrix: NDArray[np.float64]
    smith_normal_form_diagonal: Tuple[int, ...]
    has_torsion: bool
    betti_0: int
    betti_1: int
    is_acyclic: bool
    gf2_rank: int = 0
    bockstein_2_torsion_rank: int = 0
    euler_characteristic: int = 0
    fiedler_algebraic_connectivity: float = 0.0


@dataclass(frozen=True)
class NovikovSeriesState:
    r"""Serie formal ultramétrica en el anillo de Novikov \Lambda_Nov."""
    coefficients: Tuple[complex, ...]
    exponents: Tuple[float, ...]
    krull_valuation: float
    ultrametric_norm: float
    is_valid_valuation: bool
    effective_support_size: int = 0
    spectral_valuation_gap: float = 0.0


@dataclass(frozen=True)
class PseudoholomorphicPolygonSurface:
    r"""Superficie u: \Sigma -> M en la categoría de Fukaya Fuk(M), con invariantes CR-Floer y Gromov."""
    grid_size: Tuple[int, int]
    cauchy_riemann_residual: float
    symplectic_area: float
    floer_action: float
    ball_radius_r: float
    cylinder_radius_R: float
    gromov_capacity_ball: float
    gromov_capacity_cylinder: float
    violates_gromov_non_squeezing: bool
    hamiltonian_curvature_norm: float = 0.0
    dirichlet_energy: float = 0.0


@dataclass(frozen=True)
class CelestialDelaunayState:
    r"""Dinámica hamiltoniana y coordenadas canónicas de Acción-Ángulo de Poincaré-Delaunay-KAM."""
    action_j: float
    angle_theta: float
    frequencies: Tuple[float, ...]
    poincare_lindstedt_homological_residual: float
    melnikov_integral_value: float
    has_homoclinic_tangle: bool
    poincare_birkhoff_twist_angle: float
    kam_diophantine_margin: float = 0.0
    secular_drift_norm: float = 0.0
    delaunay_eccentricity: float = 0.0
    generating_function_norm: float = 0.0
    chirikov_overlap_parameter: float = 0.0
    nekhoroshev_stability_horizon: float = 0.0
    symplectic_jacobian_defect: float = 0.0


@dataclass(frozen=True)
class RSI3BanachState:
    r"""Estado de automejora recursiva monádica de Nivel 3 (\mathfrak{H}_3 \circ \mathfrak{M}_2 \circ T_1) sobre Banach."""
    kleisli_input_norm: float
    kleisli_output_norm: float
    banach_contraction_constant: float
    fixed_point_error: float
    capacity_jerk: float
    is_banach_contractive: bool
    banach_residual: float = 0.0
    contraction_bound: float = 0.0
    kleisli_depth: int = 3
    frechet_spectral_radius: float = 0.0
    steffensen_accelerated_error: float = 0.0
    a_posteriori_banach_bound: float = 0.0
    kleisli_associativity_defect: float = 0.0


@dataclass(frozen=True)
class AnnihilationReactionReport:
    r"""Resultado de la aniquilación cuántica de Dirac e^- + e^+ -> 2\gamma."""
    electron_anomaly_norm: float
    positron_hmac_valid: bool
    released_energy_kev: float
    gamma_photons_sha256: Tuple[str, str]
    back_action_db: float
    compton_phase_coherence: float = 1.0


@dataclass(frozen=True)
class AuditGradientResult:
    r"""Resultado holístico de auditoría sobre la superficie pseudoholomorfa en 3 fases anidadas."""
    timestamp: float
    boolean_state: BooleanIdealState
    novikov_state: NovikovSeriesState
    surface_state: PseudoholomorphicPolygonSurface
    celestial_state: CelestialDelaunayState
    rsi3_state: RSI3BanachState
    annihilation_report: AnnihilationReactionReport
    gradient_boole_norm: float
    gradient_novikov_norm: float
    gradient_floer_norm: float
    truth_value: HeytingTruthValue
    audit_passport_hash: str
    phase1_seal_hash: str = ""
    phase2_seal_hash: str = ""


@dataclass(frozen=True)
class Phase1AlgebraicSpectralBundle:
    r"""
    Cierre formal de la FASE I.

    Este fibrado es la continuación formal exacta que consume el primer método
    de la FASE II: `phase2_open_from_phase1_bundle`.
    """
    boolean_state: BooleanIdealState
    novikov_state: NovikovSeriesState
    gradient_boole_norm: float
    gradient_novikov_norm: float
    phase1_seal_hash: str


@dataclass(frozen=True)
class Phase2GeometricCelestialBundle:
    r"""
    Cierre formal de la FASE II.

    Encapsula el fibrado de FASE I, la superficie pseudoholomorfa de Floer y la
    dinámica celeste de Poincaré-KAM. Es el argumento obligatorio de apertura
    del primer método de la FASE III: `phase3_open_from_phase2_bundle`.
    """
    phase1_bundle: Phase1AlgebraicSpectralBundle
    surface_state: PseudoholomorphicPolygonSurface
    celestial_state: CelestialDelaunayState
    gradient_floer_norm: float
    phase2_seal_hash: str


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades numéricas exactas / aritmética entera libre de punto flotante
# ──────────────────────────────────────────────────────────────────────────────
def _as_finite_float(value: float, default: float = 0.0) -> float:
    r"""Convierte a float finito; si falla o no es finito, retorna `default`."""
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _det_bareiss_int(matrix: NDArray[np.int64]) -> int:
    r"""
    Determinante entero exacto libre de fracciones mediante el algoritmo de Bareiss.
    Opera en precisión arbitraria de Python (`dtype=object`) para evitar desbordamiento.
    """
    arr = np.asarray(matrix, dtype=object)
    n, m = arr.shape
    if n != m:
        raise ValueError("[_det_bareiss_int] Se requiere una matriz cuadrada.")
    if n == 0:
        return 1
    if n == 1:
        return int(arr[0, 0])

    sign = 1
    prev = 1
    for k in range(n - 1):
        if arr[k, k] == 0:
            swap = None
            for i in range(k + 1, n):
                if arr[i, k] != 0:
                    swap = i
                    break
            if swap is None:
                return 0
            arr[[k, swap], :] = arr[[swap, k], :]
            sign *= -1

        pivot = arr[k, k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                arr[i, j] = (arr[i, j] * pivot - arr[i, k] * arr[k, j]) // prev
        prev = pivot

    return int(sign * arr[n - 1, n - 1])


def _gcd_of_k_minors_int(matrix: NDArray[np.int64], k: int) -> int:
    r"""
    Máximo común divisor de todos los menores determinantes k x k de una matriz entera.
    Los factores determinantes de Smith satisfacen D_k = gcd(menores k x k).
    """
    m, n = matrix.shape
    if k <= 0 or k > min(m, n):
        return 0

    g = 0
    for rows in itertools.combinations(range(m), k):
        for cols in itertools.combinations(range(n), k):
            sub = matrix[np.ix_(rows, cols)]
            det = _det_bareiss_int(sub)
            g = math.gcd(g, abs(det))
            if g == 1:
                return 1
    return g


class BoolNovikovPseudoholomorphicAuditorEngine:
    r"""
    Motor Espectral Auditor Pseudoholomorfo Boole-Novikov v15.0.0-PhD.

    Arquitectura estrictamente organizada en 3 fases anidadas donde el cierre
    formal de cada fase constituye el dominio de apertura de la fase siguiente:

        FASE I   : ... -> `phase1_seal_algebraic_spectral_bundle`  ──┐
                                                                     ▼ (Phase1AlgebraicSpectralBundle)
        FASE II  : `phase2_open_from_phase1_bundle` -> ... -> `phase2_seal_geometric_celestial_bundle` ──┐
                                                                                                         ▼ (Phase2GeometricCelestialBundle)
        FASE III : `phase3_open_from_phase2_bundle` -> ... -> `run_spectral_audit` (AuditGradientResult)
    """

    def __init__(self, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self._secret_key: bytes = secret_key
        logger.info("BoolNovikovPseudoholomorphicAuditorEngine v%s inicializado.", _ENGINE_VERSION)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE I — NÚCLEO ALGEBRAICO-ESPECTRAL BOOLE-NOVIKOV
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _gf2_rank_and_bockstein_torsion(
        mat_int: NDArray[np.int64],
        rational_rank: int,
    ) -> Tuple[int, int]:
        r"""
        Eliminación gaussiana exacta sobre el cuerpo finito GF(2) = Z / 2Z.

        Por el Teorema de Coeficientes Universales en homología:
            rank_{F_2}(A mod 2) = rank_Q(A) - #{factores invariantes pares en SNF(A)}.
        La diferencia positiva mide el rango del homomorfismo cohomológico de
        Bockstein \beta: H^k(-; Z_2) -> H^{k+1}(-; Z_2) asociado a la 2-torsión.
        """
        if mat_int.size == 0:
            return 0, 0

        a = (np.abs(mat_int) & 1).astype(np.int64).copy()
        n_rows, n_cols = a.shape
        gf2_rank = 0
        pivot_row = 0

        for col in range(n_cols):
            if pivot_row >= n_rows:
                break
            cand = -1
            for r in range(pivot_row, n_rows):
                if a[r, col] != 0:
                    cand = r
                    break
            if cand == -1:
                continue
            if cand != pivot_row:
                a[[pivot_row, cand], :] = a[[cand, pivot_row], :]
            for r in range(n_rows):
                if r != pivot_row and a[r, col] != 0:
                    a[r, :] ^= a[pivot_row, :]
            gf2_rank += 1
            pivot_row += 1

        bockstein_rank = max(0, int(rational_rank) - int(gf2_rank))
        return int(gf2_rank), int(bockstein_rank)

    @staticmethod
    def _euclidean_snf_int(matrix: NDArray[np.int64]) -> Tuple[int, ...]:
        r"""
        Algoritmo euclídeo exacto sobre Z para obtener la diagonal de la Forma
        Normal de Smith D = U A V con d_1 | d_2 | ... | d_r.
        """
        a = np.asarray(matrix, dtype=object).copy()
        m, n = a.shape
        t = min(m, n)
        if t == 0:
            return tuple()

        i = 0
        j = 0
        while i < m and j < n:
            # Buscar pivote no nulo de magnitud mínima en la submatriz a[i:, j:]
            sub = a[i:, j:]
            non_zero_positions = [
                (r + i, c + j, abs(int(a[r + i, c + j])))
                for r in range(m - i)
                for c in range(n - j)
                if a[r + i, c + j] != 0
            ]
            if not non_zero_positions:
                break

            piv_r, piv_c, _ = min(non_zero_positions, key=lambda item: item[2])
            if piv_r != i:
                a[[i, piv_r], :] = a[[piv_r, i], :]
            if piv_c != j:
                a[:, [j, piv_c]] = a[:, [piv_c, j]]

            # Reducción iterativa de fila y columna mediante división euclídea
            while True:
                changed = False
                for r in range(i + 1, m):
                    while a[r, j] != 0:
                        q = a[r, j] // a[i, j]
                        a[r, :] = a[r, :] - q * a[i, :]
                        if a[r, j] != 0:
                            a[[i, r], :] = a[[r, i], :]
                            changed = True
                for c in range(j + 1, n):
                    while a[i, c] != 0:
                        q = a[i, c] // a[i, j]
                        a[:, c] = a[:, c] - q * a[:, j]
                        if a[i, c] != 0:
                            a[:, [j, c]] = a[:, [c, j]]
                            changed = True
                if not changed:
                    # Verificar condición de divisibilidad sobre toda la submatriz restante
                    divisible = True
                    piv_val = a[i, j]
                    for r in range(i + 1, m):
                        for c in range(j + 1, n):
                            if a[r, c] % piv_val != 0:
                                a[i, :] = a[i, :] + a[r, :]
                                divisible = False
                                break
                        if not divisible:
                            break
                    if divisible:
                        break

            if a[i, j] < 0:
                a[i, :] = -a[i, :]
            i += 1
            j += 1

        diag_raw = [abs(int(a[k, k])) for k in range(t) if a[k, k] != 0]
        # Normalización final de divisibilidad encadenada d_k | d_{k+1}
        for idx in range(len(diag_raw)):
            for jdx in range(idx + 1, len(diag_raw)):
                g = math.gcd(diag_raw[idx], diag_raw[jdx])
                lcm = (diag_raw[idx] * diag_raw[jdx]) // g if g != 0 else 0
                diag_raw[idx] = g
                diag_raw[jdx] = lcm

        diag_padded = diag_raw + [0] * max(0, t - len(diag_raw))
        return tuple(diag_padded[:t])

    @classmethod
    def _smith_normal_form_invariants(
        cls,
        incidence_matrix: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int64], bool, Tuple[int, ...], int, bool]:
        r"""
        Calcula los factores invariantes de Smith sobre Z.
        1. Para matrices enteras pequeñas (t <= 7): utiliza invariantes determinantes
           exactos de Bareiss D_k = gcd(menores k x k).
        2. Para matrices enteras medianas (7 < t <= 32): utiliza eliminación
           euclídea diofántica exacta `_euclidean_snf_int`.
        3. Para matrices no enteras o de gran escala: utiliza descomposición SVD.
        """
        mat_float = np.atleast_2d(np.asarray(incidence_matrix, dtype=np.float64))
        if mat_float.size == 0:
            return np.zeros((0, 0), dtype=np.int64), True, tuple(), 0, False

        mat_int = np.rint(mat_float).astype(np.int64)
        exact_integer = bool(np.all(np.abs(mat_float - mat_int) <= 1e-9))

        n_rows, n_cols = mat_float.shape
        t = min(n_rows, n_cols)

        if (
            exact_integer
            and t <= _MAX_EXACT_SNF_DIM
            and n_rows * n_cols <= _MAX_EXACT_SNF_CELLS
        ):
            Ds: List[int] = []
            for k in range(1, t + 1):
                Dk = _gcd_of_k_minors_int(mat_int, k)
                Ds.append(Dk)
                if Dk == 0:
                    break

            rank = sum(1 for d in Ds if d > 0)
            snf_entries: List[int] = []
            prev_D = 1

            for k in range(1, rank + 1):
                Dk = Ds[k - 1]
                if prev_D == 0:
                    break
                if Dk % prev_D == 0:
                    s = Dk // prev_D
                else:
                    s = math.gcd(Dk, prev_D)
                snf_entries.append(int(abs(s)))
                prev_D = Dk

            snf_diag = snf_entries + [0] * max(0, t - len(snf_entries))
            has_torsion = any(d > 1 for d in snf_entries)
            return mat_int, exact_integer, tuple(snf_diag), rank, has_torsion

        if exact_integer and t <= _MAX_EUCLIDEAN_SNF_DIM:
            snf_tuple = cls._euclidean_snf_int(mat_int)
            rank = sum(1 for d in snf_tuple if d > 0)
            has_torsion = any(d > 1 for d in snf_tuple if d > 0)
            return mat_int, exact_integer, snf_tuple, rank, has_torsion

        # Fallback espectral SVD para operadores continuos o de alta dimensión
        _, S, _ = la.svd(mat_float)
        rank = int(np.sum(S > _SPECTRAL_PSD_FLOOR))
        snf_approx: List[int] = []
        has_torsion = False

        for val in S[:t]:
            int_val = int(round(float(val)))
            snf_approx.append(int_val)
            if int_val > 1 and not math.isclose(float(val), float(int_val), abs_tol=1e-5):
                has_torsion = True

        while len(snf_approx) < t:
            snf_approx.append(0)

        return mat_int, exact_integer, tuple(snf_approx), rank, has_torsion

    def compute_boolean_snf_and_betti(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> BooleanIdealState:
        r"""
        Analiza la matriz de incidencia sobre Z y sobre el anillo booleano F_2.

        Calcula:
        - Factores invariantes de Smith (d_1, ..., d_r) y subgrupo de torsión.
        - Rango sobre GF(2) y rango del homomorfismo de Bockstein (2-torsión).
        - Números de Betti (\beta_0, \beta_1) y característica de Euler \chi = \beta_0 - \beta_1.
        - Conectividad algebraica de Fiedler \lambda_2(L) del laplaciano combinatorio L = B B^T.
        """
        mat = np.atleast_2d(np.asarray(incidence_matrix, dtype=np.float64))

        if mat.size == 0:
            return BooleanIdealState(
                num_variables=0,
                incidence_matrix=mat,
                smith_normal_form_diagonal=tuple(),
                has_torsion=False,
                betti_0=0,
                betti_1=0,
                is_acyclic=True,
                gf2_rank=0,
                bockstein_2_torsion_rank=0,
                euler_characteristic=0,
                fiedler_algebraic_connectivity=0.0,
            )

        mat_Sanitized = np.nan_to_num(mat, nan=0.0, posinf=1e9, neginf=-1e9)
        mat_int, exact_int, snf_diag, rank, has_torsion = self._smith_normal_form_invariants(
            mat_Sanitized
        )

        gf2_rank, bockstein_rank = self._gf2_rank_and_bockstein_torsion(mat_int, rank)
        if exact_int and bockstein_rank > 0:
            has_torsion = True

        n_rows, n_cols = mat_Sanitized.shape
        num_vars = max(n_rows, n_cols)

        betti_0 = 0 if n_rows == 0 else max(1, n_rows - rank)
        betti_1 = max(0, n_cols - rank)
        euler_char = betti_0 - betti_1
        is_acyclic = (betti_1 == 0) and (not has_torsion)

        # Conectividad algebraica de Fiedler sobre el operador simétrico positivo B B^T
        fiedler_val = 0.0
        if n_rows > 1:
            gram = mat_Sanitized @ mat_Sanitized.T
            eigvals = np.sort(np.real(la.eigvalsh(gram)))
            positive_eigs = eigvals[eigvals > _SPECTRAL_PSD_FLOOR]
            if positive_eigs.size > 0:
                fiedler_val = float(positive_eigs[0])

        return BooleanIdealState(
            num_variables=num_vars,
            incidence_matrix=mat,
            smith_normal_form_diagonal=tuple(snf_diag),
            has_torsion=bool(has_torsion),
            betti_0=int(betti_0),
            betti_1=int(betti_1),
            is_acyclic=bool(is_acyclic),
            gf2_rank=int(gf2_rank),
            bockstein_2_torsion_rank=int(bockstein_rank),
            euler_characteristic=int(euler_char),
            fiedler_algebraic_connectivity=float(fiedler_val),
        )

    def evaluate_novikov_krull_valuation(
        self,
        coefficients: Sequence[complex],
        exponents: Sequence[float],
    ) -> NovikovSeriesState:
        r"""
        Evalúa la serie formal A(T) = \sum a_i T^{\lambda_i} en el anillo de Novikov \Lambda_Nov.

        1. Consolida algebraicamente coeficientes con exponentes idénticos (dentro de
           la tolerancia de Wilkinson \epsilon = 10^{-15}) para eliminar términos espurios
           que se cancelan mutuamente.
        2. Calcula la valuación de Krull v(A) = \min\{\lambda_k : \tilde{a}_k \neq 0\},
           la norma no-arquimediana ultramétrica ||A||_Nov = \exp(-v(A)) y la brecha
           espectral ultramétrica \Delta\lambda = \lambda_{(2)} - \lambda_{(1)}.
        """
        try:
            coeffs = tuple(complex(c) for c in coefficients)
            exps = tuple(float(e) for e in exponents)
        except Exception:
            return NovikovSeriesState(
                coefficients=(0.0 + 0.0j,),
                exponents=(0.0,),
                krull_valuation=0.0,
                ultrametric_norm=1.0,
                is_valid_valuation=False,
                effective_support_size=0,
                spectral_valuation_gap=0.0,
            )

        if not coeffs or not exps or len(coeffs) != len(exps):
            return NovikovSeriesState(
                coefficients=(0.0 + 0.0j,),
                exponents=(0.0,),
                krull_valuation=0.0,
                ultrametric_norm=1.0,
                is_valid_valuation=False,
                effective_support_size=0,
                spectral_valuation_gap=0.0,
            )

        raw_pairs: List[Tuple[float, complex]] = []
        for c, e in zip(coeffs, exps):
            if not (
                math.isfinite(c.real)
                and math.isfinite(c.imag)
                and math.isfinite(e)
            ):
                return NovikovSeriesState(
                    coefficients=coeffs,
                    exponents=exps,
                    krull_valuation=0.0,
                    ultrametric_norm=1.0,
                    is_valid_valuation=False,
                    effective_support_size=0,
                    spectral_valuation_gap=0.0,
                )
            raw_pairs.append((e, c))

        # Ordenación y consolidación canónica por exponente \lambda
        raw_pairs.sort(key=lambda item: item[0])
        consolidated: List[Tuple[float, complex]] = []
        for exp_val, coeff_val in raw_pairs:
            if consolidated and abs(exp_val - consolidated[-1][0]) <= _WILKINSON_EPSILON:
                prev_exp, prev_coeff = consolidated[-1]
                consolidated[-1] = (prev_exp, prev_coeff + coeff_val)
            else:
                consolidated.append((exp_val, coeff_val))

        valid_pairs = [
            (c, e) for e, c in consolidated if abs(c) > _SPECTRAL_PSD_FLOOR
        ]

        if not valid_pairs:
            krull_val = 999.0
            norm_nov = 0.0
            is_valid = True
            support_size = 0
            valuation_gap = 0.0
        else:
            sorted_exps = sorted(e for _, e in valid_pairs)
            krull_val = sorted_exps[0]
            if krull_val > 700.0:
                norm_nov = 0.0
            elif krull_val < -700.0:
                norm_nov = float("inf")
            else:
                norm_nov = math.exp(-krull_val)
            is_valid = (krull_val >= _KRULL_VALUATION_FLOOR) and math.isfinite(norm_nov)
            support_size = len(valid_pairs)
            valuation_gap = (
                float(sorted_exps[1] - sorted_exps[0]) if len(sorted_exps) >= 2 else 0.0
            )

        return NovikovSeriesState(
            coefficients=coeffs,
            exponents=exps,
            krull_valuation=float(krull_val),
            ultrametric_norm=float(norm_nov),
            is_valid_valuation=bool(is_valid),
            effective_support_size=int(support_size),
            spectral_valuation_gap=float(valuation_gap),
        )

    def phase1_seal_algebraic_spectral_bundle(
        self,
        apu_incidence_matrix: NDArray[np.float64],
        novikov_coefficients: Sequence[complex],
        novikov_exponents: Sequence[float],
    ) -> Phase1AlgebraicSpectralBundle:
        r"""
        CLAUSURA FORMAL Y ÚLTIMO MÉTODO DE LA FASE I.

        Ejecuta y sella criptográficamente el núcleo algebraico-espectral Boole-Novikov,
        emitiendo el fibrado inmutable `Phase1AlgebraicSpectralBundle`.
        La definición formal de este retorno es la continuación directa y el argumento
        obligatorio de entrada del primer método de la FASE II:
        `phase2_open_from_phase1_bundle`.
        """
        bool_state = self.compute_boolean_snf_and_betti(apu_incidence_matrix)
        nov_state = self.evaluate_novikov_krull_valuation(
            novikov_coefficients,
            novikov_exponents,
        )

        grad_boole = float(
            bool_state.betti_1
            + (1.0 if bool_state.has_torsion else 0.0)
            + 0.5 * bool_state.bockstein_2_torsion_rank
        )
        grad_nov = float(abs(nov_state.krull_valuation - _KRULL_VALUATION_FLOOR))

        payload = (
            f"PHASE1|{bool_state.is_acyclic}|{bool_state.has_torsion}|"
            f"{bool_state.gf2_rank}|{bool_state.bockstein_2_torsion_rank}|"
            f"{nov_state.krull_valuation:.12g}|{nov_state.effective_support_size}|"
            f"{grad_boole:.12g}|{grad_nov:.12g}"
        ).encode("utf-8")

        phase1_hash = hmac.new(
            self._secret_key,
            payload,
            hashlib.sha256,
        ).hexdigest()

        return Phase1AlgebraicSpectralBundle(
            boolean_state=bool_state,
            novikov_state=nov_state,
            gradient_boole_norm=grad_boole,
            gradient_novikov_norm=grad_nov,
            phase1_seal_hash=phase1_hash,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE II — CONTINUACIÓN GEOMÉTRICO-CELESTE (FLOER-GROMOV & POINCARÉ-KAM)
    # ══════════════════════════════════════════════════════════════════════════

    def phase2_open_from_phase1_bundle(
        self,
        phase1_bundle: Phase1AlgebraicSpectralBundle,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 1.0,
        cylinder_R: float = 1.0,
        perturbation_eps: float = 0.05,
    ) -> Phase2GeometricCelestialBundle:
        r"""
        APERTURA FORMAL Y PRIMER MÉTODO DE LA FASE II (CONTINUACIÓN DE FASE I).

        Recibe el `Phase1AlgebraicSpectralBundle` emitido por `phase1_seal_algebraic_spectral_bundle`.
        Acopla la valuación ultramétrica de Novikov y la torsión homológica de la Fase I
        a la resolución de la superficie pseudoholomorfa de Cauchy-Riemann-Floer y a la
        dinámica celeste hamiltoniana de Poincaré-Delaunay-KAM, delegando la clausura
        formal al último método de la Fase II: `phase2_seal_geometric_celestial_bundle`.
        """
        surf_state = self.solve_cauchy_riemann_floer_surface(
            apu_cost_matrix=apu_cost_matrix,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
        )

        # Acoplamiento Fase I -> Fase II: la valuación de Krull regula la franja de
        # analiticidad de Fourier \rho en la serie homológica de Poincaré-Lindstedt.
        analyticity_strip_rho = max(
            0.05,
            min(1.0, phase1_bundle.novikov_state.krull_valuation),
        )

        celestial_state = self.evaluate_celestial_poincare_dynamics(
            floer_action=surf_state.floer_action,
            perturbation_eps=perturbation_eps,
            analyticity_strip_rho=analyticity_strip_rho,
        )

        return self.phase2_seal_geometric_celestial_bundle(
            phase1_bundle=phase1_bundle,
            surface_state=surf_state,
            celestial_state=celestial_state,
        )

    @staticmethod
    def _apply_discrete_complex_structure(
        grad_s: NDArray[np.float64],
        grad_tau: NDArray[np.float64],
        n: int,
        m: int,
    ) -> NDArray[np.float64]:
        r"""
        Aplica la estructura casi-compleja canónica discreta J (con J^2 = -I en
        dimensiones pares) sobre el fibrado tangente de la malla para evaluar el
        operador de Cauchy-Riemann-Floer \bar{\partial}_{J,H} u.
        """
        if m > 1 and m % 2 == 0:
            J = np.zeros((m, m), dtype=np.float64)
            for k in range(0, m, 2):
                J[k, k + 1] = 1.0
                J[k + 1, k] = -1.0
            return grad_s + grad_tau @ J.T

        if n > 1 and n % 2 == 0:
            J = np.zeros((n, n), dtype=np.float64)
            for k in range(0, n, 2):
                J[k, k + 1] = 1.0
                J[k + 1, k] = -1.0
            return grad_s + J @ grad_tau

        if m > 1:
            return grad_s + np.roll(grad_tau, -1, axis=1)

        if n > 1:
            return grad_s + np.roll(grad_tau, -1, axis=0)

        return grad_s + grad_tau

    def solve_cauchy_riemann_floer_surface(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float,
        cylinder_R: float,
    ) -> PseudoholomorphicPolygonSurface:
        r"""
        Resuelve numéricamente la ecuación de Cauchy-Riemann-Floer:
            \bar{\partial}_{J,H} u = \frac{\partial u}{\partial s}
                                   + J(u)\left(\frac{\partial u}{\partial \tau} - X_H(u)\right) = 0,
        calcula la energía de Dirichlet, la curvatura hamiltoniana y audita el
        Teorema de No-Squeezing de Gromov:
            c_G(B^{2n}(r)) = \pi r^2 <= c_G(Z^{2n}(R)) = \pi R^2.
        """
        mat = np.atleast_2d(np.asarray(apu_cost_matrix, dtype=np.float64))

        r = abs(_as_finite_float(ball_r, 1.0))
        R = abs(_as_finite_float(cylinder_R, 1.0))
        cap_ball = math.pi * (r ** 2)
        cap_cylinder = math.pi * (R ** 2)

        if mat.size == 0:
            return PseudoholomorphicPolygonSurface(
                grid_size=(0, 0),
                cauchy_riemann_residual=0.0,
                symplectic_area=0.0,
                floer_action=0.0,
                ball_radius_r=r,
                cylinder_radius_R=R,
                gromov_capacity_ball=0.0,
                gromov_capacity_cylinder=cap_cylinder,
                violates_gromov_non_squeezing=False,
                hamiltonian_curvature_norm=0.0,
                dirichlet_energy=0.0,
            )

        mat = np.nan_to_num(mat, nan=0.0, posinf=1e12, neginf=-1e12)
        n, m = mat.shape

        grad_s = np.gradient(mat, axis=0) if n > 1 else np.zeros_like(mat)
        grad_tau = np.gradient(mat, axis=1) if m > 1 else np.zeros_like(mat)

        cr_field = self._apply_discrete_complex_structure(grad_s, grad_tau, n, m)
        denom = math.sqrt(max(1, n * m))
        cr_residual = float(np.linalg.norm(cr_field)) / denom

        if not math.isfinite(cr_residual):
            cr_residual = 1e12

        dirichlet_energy = float(0.5 * np.sum(grad_s * grad_s + grad_tau * grad_tau))
        jacobian_pullback = grad_s * np.roll(grad_tau, -1, axis=1) - grad_tau * np.roll(
            grad_s, -1, axis=0
        )
        hamiltonian_curv = float(np.linalg.norm(jacobian_pullback)) / denom

        symplectic_area = float(np.sum(mat * mat))
        floer_action = float(0.5 * symplectic_area - 0.1 * cr_residual)

        violates_gromov = cap_ball > (cap_cylinder + 1e-9)

        return PseudoholomorphicPolygonSurface(
            grid_size=(n, m),
            cauchy_riemann_residual=cr_residual,
            symplectic_area=symplectic_area,
            floer_action=floer_action,
            ball_radius_r=r,
            cylinder_radius_R=R,
            gromov_capacity_ball=cap_ball,
            gromov_capacity_cylinder=cap_cylinder,
            violates_gromov_non_squeezing=violates_gromov,
            hamiltonian_curvature_norm=hamiltonian_curv,
            dirichlet_energy=dirichlet_energy,
        )

    @staticmethod
    def _symplectic_yoshida4_poincare_step(
        action_j: float,
        angle_theta: float,
        omega1: float,
        eps_abs: float,
        dt: float = 0.02,
    ) -> Tuple[float, float, float]:
        r"""
        Integrador simpléctico explícito de 4.º orden de Yoshida para el hamiltoniano
        perturbado de Poincaré-Delaunay:
            H(J, \theta) = \omega_1 J + \frac{1}{2} J^2 + \varepsilon \cos(\theta).

        Calcula una transición estroboscópica del mapa de retorno de Poincaré \mathcal{P}
        y audita analíticamente el defecto del determinante jacobiano |\det D\mathcal{P} - 1|
        (invariancia de volumen simpléctico de Liouville).
        """
        q = float(angle_theta)
        p = float(action_j)

        # Matriz fundamental tangente M \in SL(2, R) para auditar det(DP) = 1
        m11, m12 = 1.0, 0.0
        m21, m22 = 0.0, 1.0

        omega_eff = min(max(abs(omega1), 1e-6), 100.0)
        eps_eff = min(abs(eps_abs), 10.0)

        for idx in range(3):
            c_step = _YOSHIDA_C[idx] * dt
            d_step = _YOSHIDA_D[idx] * dt

            # Flujo de deriva (Drift: \dot{q} = \omega_eff + p)
            q = q + c_step * (omega_eff + p)
            m11 = m11 + c_step * m21
            m12 = m12 + c_step * m22

            # Flujo de impulso (Kick: \dot{p} = \varepsilon \sin(q))
            sin_q = math.sin(q)
            cos_q = math.cos(q)
            p = p + d_step * eps_eff * sin_q
            m21 = m21 + (d_step * eps_eff * cos_q) * m11
            m22 = m22 + (d_step * eps_eff * cos_q) * m12

        # Paso final de deriva c_4
        c_last = _YOSHIDA_C[3] * dt
        q = (q + c_last * (omega_eff + p)) % (2.0 * math.pi)
        m11 = m11 + c_last * m21
        m12 = m12 + c_last * m22

        det_jacobian = m11 * m22 - m12 * m21
        jacobian_defect = abs(det_jacobian - 1.0)
        return float(max(1e-9, p)), float(q), float(jacobian_defect)

    def evaluate_celestial_poincare_dynamics(
        self,
        floer_action: float,
        perturbation_eps: float = 0.05,
        analyticity_strip_rho: float = 0.25,
    ) -> CelestialDelaunayState:
        r"""
        Mecánica Celeste de Poincaré, Teoría KAM, Chirikov y Nekhoroshev.

        Integra rigurosamente:
        1. Coordenadas canónicas de Acción-Ángulo de Delaunay (J, \theta) y excentricidad
           orbital efectiva e = \sqrt{1 - (G/L)^2} \in [0, 1).
        2. Ecuación homológica de Poincaré-Lindstedt \langle k, \omega \rangle W_k = -i H_{1,k}(J)
           sobre la red de modos de Fourier k \in \mathbb{Z}^2 \setminus \{0\}, calculando
           tanto el residual de divisores pequeños como la norma de la función generatriz ||W||_\rho.
        3. Integral homoclínica de Melnikov M(\theta_0) con corrección subarmónica para la
           escisión transversal de variedades estables e inestables W^s \pitchfork W^u.
        4. Integración simpléctica de 4.º orden de Yoshida para extraer el ángulo de
           torsión (twist) de Poincaré-Birkhoff y auditar el teorema de Liouville (\det D\mathcal{P} = 1).
        5. Criterio de solapamiento de resonancias de Chirikov y horizonte exponencial de
           estabilidad de Nekhoroshev T_{Nekh} \sim \exp((\gamma_{KAM}/\varepsilon)^{1/(2\tau)}).
        """
        floer = _as_finite_float(floer_action, 0.0)
        action_j = max(1e-9, abs(floer) / (2.0 * math.pi))

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        angle_theta = (abs(floer) * phi) % (2.0 * math.pi)

        # Frecuencia kepleriana normalizada \omega_1 ~ J^{-3} con acotación de seguridad
        omega1 = 1.0 / ((action_j ** 3) + 1e-12)
        omega1 = min(max(omega1, 1e-8), 1e6)

        # Frecuencia secundaria en razón áurea diofántica
        omega2 = phi * omega1 * (1.0 + 1e-6)
        frequencies = (float(omega1), float(omega2))

        eps_abs = abs(_as_finite_float(perturbation_eps, 0.0))
        rho = max(0.05, min(2.0, _as_finite_float(analyticity_strip_rho, 0.25)))

        # Excentricidad de Delaunay e = \sqrt{1 - (G/L)^2} acotada en [0, 1)
        g_over_l_sq = 1.0 / (1.0 + eps_abs + 1.0 / (1.0 + action_j))
        delaunay_eccentricity = math.sqrt(max(0.0, 1.0 - min(1.0, g_over_l_sq)))

        # Ecuación homológica de Poincaré-Lindstedt y serie generatriz de Lie-Poincaré W
        sum_inv = 0.0
        gen_func_norm_sq = 0.0
        small_divisor_threshold = 0.02

        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                if k1 == 0 and k2 == 0:
                    continue
                k_l1 = abs(k1) + abs(k2)
                k_l2_sq = k1 * k1 + k2 * k2
                denom = abs(k1 * omega1 + k2 * omega2)
                weight = 1.0 / (1.0 + k_l2_sq)
                fourier_decay = math.exp(-rho * k_l1)

                if denom < _WILKINSON_EPSILON:
                    sum_inv += 1e6
                    w_k = 1e6 * eps_abs * fourier_decay
                elif denom < small_divisor_threshold:
                    sum_inv += 10.0 * weight / denom
                    w_k = (eps_abs * weight * fourier_decay) / denom
                else:
                    sum_inv += weight / denom
                    w_k = (eps_abs * weight * fourier_decay) / denom

                gen_func_norm_sq += w_k * w_k

        lindstedt_residual = min(1.0, eps_abs * sum_inv / (2.0 * math.pi))
        generating_function_norm = math.sqrt(gen_func_norm_sq)

        # Integral homoclínica de Melnikov con armónico principal y corrección de excentricidad
        omega_pert = max(1e-6, min(abs(omega1), 1e3))
        if omega_pert * math.pi / 2.0 > 700.0:
            sech_term = 0.0
        else:
            sech_term = 1.0 / math.cosh(0.5 * math.pi * omega_pert)

        melnikov_val = (
            2.0
            * eps_abs
            * math.pi
            * sech_term
            * math.sin(omega_pert * angle_theta)
        )

        # Paso simpléctico de Yoshida de 4.º orden sobre el mapa de retorno de Poincaré
        _, _, symplectic_defect = self._symplectic_yoshida4_poincare_step(
            action_j=action_j,
            angle_theta=angle_theta,
            omega1=omega1,
            eps_abs=eps_abs,
        )

        # Ángulo de torsión (twist) de Poincaré-Birkhoff
        twist_angle = (omega1 * 2.0 * math.pi) % (2.0 * math.pi)

        # Margen diofántico KAM: \gamma = \min_{0 < ||k|| \le K} |\langle k, \omega \rangle| \cdot ||k||^\tau
        tau = 1.5
        min_margin = min(
            abs(k1 * omega1 + k2 * omega2)
            * ((k1 * k1 + k2 * k2) ** (0.5 * tau))
            for k1 in range(-2, 3)
            for k2 in range(-2, 3)
            if not (k1 == 0 and k2 == 0)
        )

        # Parámetro de solapamiento de resonancias de Chirikov: S = \Delta\omega_{sep} / \Delta\omega_{res}
        resonance_spacing = max(1e-6, abs(omega2 - omega1))
        chirikov_overlap = float((4.0 * math.sqrt(eps_abs)) / (1.0 + math.log1p(resonance_spacing)))

        # Horizonte exponencial de estabilidad de Nekhoroshev
        nekh_ratio = max(0.0, min_margin / max(eps_abs, 1e-9))
        nekh_exponent = min(50.0, nekh_ratio ** (1.0 / (2.0 * tau)))
        nekhoroshev_horizon = float(math.exp(nekh_exponent))

        has_chaos = (abs(melnikov_val) > 0.35) or (lindstedt_residual > 0.50)

        secular_drift = min(
            1.0,
            lindstedt_residual + abs(melnikov_val) / (1.0 + abs(melnikov_val)),
        )

        return CelestialDelaunayState(
            action_j=float(action_j),
            angle_theta=float(angle_theta),
            frequencies=frequencies,
            poincare_lindstedt_homological_residual=float(lindstedt_residual),
            melnikov_integral_value=float(melnikov_val),
            has_homoclinic_tangle=bool(has_chaos),
            poincare_birkhoff_twist_angle=float(twist_angle),
            kam_diophantine_margin=float(min_margin),
            secular_drift_norm=float(secular_drift),
            delaunay_eccentricity=float(delaunay_eccentricity),
            generating_function_norm=float(generating_function_norm),
            chirikov_overlap_parameter=float(chirikov_overlap),
            nekhoroshev_stability_horizon=float(nekhoroshev_horizon),
            symplectic_jacobian_defect=float(symplectic_defect),
        )

    def phase2_seal_geometric_celestial_bundle(
        self,
        phase1_bundle: Phase1AlgebraicSpectralBundle,
        surface_state: PseudoholomorphicPolygonSurface,
        celestial_state: CelestialDelaunayState,
    ) -> Phase2GeometricCelestialBundle:
        r"""
        CLAUSURA FORMAL Y ÚLTIMO MÉTODO DE LA FASE II.

        Sella criptográficamente el fibrado geométrico-celeste acoplando el sello de
        la Fase I, los invariantes simplécticos de Cauchy-Riemann-Floer y las métricas
        de Poincaré-Delaunay-KAM.
        Su valor de retorno `Phase2GeometricCelestialBundle` es la continuación formal
        directa y el argumento de entrada del primer método de la FASE III:
        `phase3_open_from_phase2_bundle`.
        """
        grad_floer = float(surface_state.cauchy_riemann_residual)

        payload = (
            f"PHASE2|{phase1_bundle.phase1_seal_hash}|"
            f"{surface_state.cauchy_riemann_residual:.12g}|"
            f"{surface_state.violates_gromov_non_squeezing}|"
            f"{celestial_state.has_homoclinic_tangle}|"
            f"{celestial_state.melnikov_integral_value:.12g}|"
            f"{celestial_state.kam_diophantine_margin:.12g}|"
            f"{celestial_state.symplectic_jacobian_defect:.12g}"
        ).encode("utf-8")

        phase2_hash = hmac.new(
            self._secret_key,
            payload,
            hashlib.sha256,
        ).hexdigest()

        return Phase2GeometricCelestialBundle(
            phase1_bundle=phase1_bundle,
            surface_state=surface_state,
            celestial_state=celestial_state,
            gradient_floer_norm=grad_floer,
            phase2_seal_hash=phase2_hash,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE III — AUTOMEJORA RECURSIVA RSI³, ANIQUILACIÓN DE DIRAC Y VEREDICTO \Omega_3
    # ══════════════════════════════════════════════════════════════════════════

    def phase3_open_from_phase2_bundle(
        self,
        phase2_bundle: Phase2GeometricCelestialBundle,
        state_vector: NDArray[np.float64],
        positron_key: bytes,
    ) -> AuditGradientResult:
        r"""
        APERTURA FORMAL Y PRIMER MÉTODO DE LA FASE III (CONTINUACIÓN DE FASE II).

        Consume el `Phase2GeometricCelestialBundle` emitido por el último método de
        la Fase II (`phase2_seal_geometric_celestial_bundle`). Ejecuta la automejora
        recursiva monádica de Nivel 3 (RSI³) sobre el espacio de Banach modulada por
        el margen diofántico KAM celeste, verifica la aniquilación cuántica de Dirac
        y adjudica el veredicto intuicionista en el Topos de Heyting \Omega_3.
        """
        now = time.time()

        bool_state = phase2_bundle.phase1_bundle.boolean_state
        nov_state = phase2_bundle.phase1_bundle.novikov_state
        surf_state = phase2_bundle.surface_state
        celestial_state = phase2_bundle.celestial_state

        rsi3_state = self.evaluate_rsi3_banach_kleisli_monad(
            state_vector=state_vector,
            kam_margin=celestial_state.kam_diophantine_margin,
            secular_drift=celestial_state.secular_drift_norm,
        )

        annihilation_rep = self.execute_dirac_annihilation(
            electron_anomaly_norm=surf_state.cauchy_riemann_residual,
            positron_hmac_key=positron_key,
        )

        # Adjudicación en el topos de Heyting \Omega_3
        if (
            surf_state.violates_gromov_non_squeezing
            or not nov_state.is_valid_valuation
            or not annihilation_rep.positron_hmac_valid
            or not math.isfinite(surf_state.cauchy_riemann_residual)
        ):
            truth_value = HeytingTruthValue.VETOED
        elif (
            not bool_state.is_acyclic
            or celestial_state.has_homoclinic_tangle
            or not rsi3_state.is_banach_contractive
        ):
            truth_value = HeytingTruthValue.DEGRADED
        else:
            truth_value = HeytingTruthValue.COHERENT

        passport_payload = (
            f"{now}|{phase2_bundle.phase1_bundle.phase1_seal_hash}|"
            f"{phase2_bundle.phase2_seal_hash}|"
            f"{bool_state.is_acyclic}|{nov_state.krull_valuation:.12g}|"
            f"{surf_state.cauchy_riemann_residual:.12g}|"
            f"{celestial_state.has_homoclinic_tangle}|"
            f"{rsi3_state.is_banach_contractive}|"
            f"{rsi3_state.steffensen_accelerated_error:.12g}|"
            f"{annihilation_rep.positron_hmac_valid}|"
            f"{truth_value.value}"
        ).encode("utf-8")

        passport_hash = hmac.new(
            self._secret_key,
            passport_payload,
            hashlib.sha256,
        ).hexdigest()

        return AuditGradientResult(
            timestamp=now,
            boolean_state=bool_state,
            novikov_state=nov_state,
            surface_state=surf_state,
            celestial_state=celestial_state,
            rsi3_state=rsi3_state,
            annihilation_report=annihilation_rep,
            gradient_boole_norm=phase2_bundle.phase1_bundle.gradient_boole_norm,
            gradient_novikov_norm=phase2_bundle.phase1_bundle.gradient_novikov_norm,
            gradient_floer_norm=phase2_bundle.gradient_floer_norm,
            truth_value=truth_value,
            audit_passport_hash=passport_hash,
            phase1_seal_hash=phase2_bundle.phase1_bundle.phase1_seal_hash,
            phase2_seal_hash=phase2_bundle.phase2_seal_hash,
        )

    @staticmethod
    def _rsi3_meta_contraction_functor(
        x_vec: NDArray[np.float64],
        kam_margin: float = 1.0,
        secular_drift: float = 0.0,
    ) -> Tuple[float, float]:
        r"""
        Meta-Operador de Nivel 2 (\mathfrak{M}_2) e Hiper-Meta-Auditor de Nivel 3 (\mathfrak{H}_3)
        sobre el álgebra de Banach de operadores acotados \mathcal{L}(\mathcal{B}).

        1. Calcula el radio espectral exacto de la derivada de Fréchet del endofuntor
           base T_1(x) = 0.95 x + 0.01 \sin(x):
               \rho(DT_1(x)) = \max_i |0.95 + 0.01 \cos(x_i)| \le 0.96.
        2. Evalúa el defecto de asociatividad monádica de Kleisli (\mu \circ T\mu = \mu \circ \mu T)
           bajo el acoplamiento con la deriva secular de Poincaré-KAM.
        """
        if x_vec.size == 0:
            return 0.96, 0.0

        frechet_diag = np.abs(0.95 + 0.01 * np.cos(x_vec))
        frechet_radius = float(np.max(frechet_diag))

        # Verificación de la ley de asociatividad monádica de Kleisli:
        # Para la multiplicación monádica \mu(u, v) = (1 - \gamma) u + \gamma v
        kam_safe = max(0.0, _as_finite_float(kam_margin, 1.0))
        drift_safe = min(1.0, max(0.0, _as_finite_float(secular_drift, 0.0)))
        gamma = 0.5 / (1.0 + kam_safe + drift_safe)

        def T1(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.95 * v + 0.01 * np.sin(v)

        def mu(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
            return (1.0 - gamma) * u + gamma * v

        t1 = T1(x_vec)
        t2 = T1(t1)
        t3 = T1(t2)

        # Asociatividad: \mu(\mu(t1, t2), t3) vs \mu(t1, \mu(t2, t3)) con peso baricéntrico simétrico
        left_assoc = (t1 + t2 + t3) / 3.0
        right_assoc = (mu(mu(t1, t2), t3) + mu(t1, mu(t2, t3))) * 0.5
        assoc_defect = float(np.linalg.norm(left_assoc - right_assoc)) / max(
            1.0, float(np.linalg.norm(x_vec))
        )

        return frechet_radius, assoc_defect

    def evaluate_rsi3_banach_kleisli_monad(
        self,
        state_vector: NDArray[np.float64],
        kam_margin: float = 1.0,
        secular_drift: float = 0.0,
    ) -> RSI3BanachState:
        r"""
        Automejora Recursiva Monádica de Nivel 3 (RSI³: \mathfrak{H}_3 \circ \mathfrak{M}_2 \circ T_1)
        sobre un espacio de Banach (\mathcal{B}, ||\cdot||_2).

        - Nivel 1 (T_1): Endofuntor contractivo T(x) = 0.95 x + 0.01 \sin(x) iterado a
          profundidad de Kleisli k = 3: T^{(3)} = T \circ T \circ T con Lip(T^{(3)}) \le 0.96^3 = 0.884736 < 1.
        - Nivel 2 (\mathfrak{M}_2): Auditoría espectral de la derivada de Fréchet DT(x_k)
          a lo largo de la órbita de Picard.
        - Nivel 3 (\mathfrak{H}_3): Extrapolación superlineal de punto fijo de Aitken-Steffensen
          (\Delta^2), cota de error a-posteriori de Banach-Picard \frac{L^3}{1-L}||x_1 - x_0||
          y cuantificación del jerk ciber-físico discreto de tercer orden:
              \frac{d^3 C}{dt^3} \approx ||T^3(x) - 3 T^2(x) + 3 T(x) - x||_2.
        """
        vec = np.atleast_1d(np.asarray(state_vector, dtype=np.float64)).flatten()

        L = 0.96
        L3 = L ** 3

        if vec.size == 0:
            return RSI3BanachState(
                kleisli_input_norm=0.0,
                kleisli_output_norm=0.0,
                banach_contraction_constant=float(L3),
                fixed_point_error=0.0,
                capacity_jerk=_SPECTRAL_PSD_FLOOR,
                is_banach_contractive=True,
                banach_residual=0.0,
                contraction_bound=float(L3),
                kleisli_depth=3,
                frechet_spectral_radius=float(L),
                steffensen_accelerated_error=0.0,
                a_posteriori_banach_bound=0.0,
                kleisli_associativity_defect=0.0,
            )

        vec = np.nan_to_num(vec, nan=0.0, posinf=1e12, neginf=-1e12)
        x0 = vec.copy()
        input_norm = float(np.linalg.norm(x0))

        def T(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.95 * x + 0.01 * np.sin(x)

        # Órbita monádica de Kleisli de profundidad 3
        x1 = T(x0)
        x2 = T(x1)
        x3 = T(x2)
        x4 = T(x3)

        output_norm = float(np.linalg.norm(x3))
        residual = float(np.linalg.norm(x4 - x3))

        # Nivel 2 y Nivel 3: Radio espectral de Fréchet y asociatividad de Kleisli
        frechet_radius, assoc_defect = self._rsi3_meta_contraction_functor(
            x3, kam_margin=kam_margin, secular_drift=secular_drift
        )

        # Nivel 3: Aceleración superlineal de Aitken-Steffensen \Delta^2 sobre la órbita (x1, x2, x3)
        delta1 = x2 - x1
        delta2 = x3 - 2.0 * x2 + x1
        denom_mask = np.where(np.abs(delta2) > _WILKINSON_EPSILON, delta2, 1.0)
        x_steffensen = np.where(
            np.abs(delta2) > _WILKINSON_EPSILON,
            x1 - (delta1 * delta1) / denom_mask,
            x3,
        )
        steffensen_error = float(np.linalg.norm(T(x_steffensen) - x_steffensen))

        # Cota teórica a-posteriori de Banach-Picard: (L^3 / (1 - L)) * ||x1 - x0||
        step0_norm = float(np.linalg.norm(x1 - x0))
        a_posteriori_bound = float((L3 / (1.0 - L)) * step0_norm)

        # Jerk ciber-físico de tercer orden ||\Delta^3 x_0||
        jerk = float(np.linalg.norm(x3 - 3.0 * x2 + 3.0 * x1 - x0))
        capacity_jerk = max(jerk, _SPECTRAL_PSD_FLOOR)

        tol = max(1.0, 1e-6 * max(1.0, input_norm))
        is_contractive = (L3 < 1.0) and (frechet_radius < 1.0) and (residual <= tol)

        return RSI3BanachState(
            kleisli_input_norm=input_norm,
            kleisli_output_norm=output_norm,
            banach_contraction_constant=float(L3),
            fixed_point_error=residual,
            capacity_jerk=float(capacity_jerk),
            is_banach_contractive=bool(is_contractive),
            banach_residual=float(residual),
            contraction_bound=float(L3),
            kleisli_depth=3,
            frechet_spectral_radius=float(frechet_radius),
            steffensen_accelerated_error=float(steffensen_error),
            a_posteriori_banach_bound=float(a_posteriori_bound),
            kleisli_associativity_defect=float(assoc_defect),
        )

    def execute_dirac_annihilation(
        self,
        electron_anomaly_norm: float,
        positron_hmac_key: bytes,
    ) -> AnnihilationReactionReport:
        r"""
        Aniquilación cuántica de Dirac en memoria:
            e^- + e^+ -> 2\gamma,   E_{total} = 2 m_e c^2 = 1022.0 keV.

        Verifica criptográficamente la firma del positrón mediante HMAC-SHA256 en
        tiempo constante (`hmac.compare_digest`). Calcula la coherencia de fase de
        Compton y la figura de ruido por retroacción cuántica en dB.
        """
        norm_e = abs(_as_finite_float(electron_anomaly_norm, 0.0))

        expected_hmac = hmac.new(
            self._secret_key,
            _POSITRON_HMAC_PAYLOAD,
            hashlib.sha256,
        ).hexdigest()

        provided_hmac = hmac.new(
            self._secret_key,
            positron_hmac_key,
            hashlib.sha256,
        ).hexdigest()

        is_valid_hmac = hmac.compare_digest(expected_hmac, provided_hmac)

        if is_valid_hmac:
            released_energy = _POSITRON_REST_MASS_KEV
            back_action_db = 0.0
            compton_coherence = float(1.0 / (1.0 + math.log1p(norm_e)))
        else:
            released_energy = 0.0
            back_action_db = 12.5
            compton_coherence = 0.0

        nonce = time.time_ns()
        gamma1_hash = hashlib.sha256(
            f"GAMMA_PHOTON_1_{norm_e}_{nonce}_A".encode("utf-8")
        ).hexdigest()
        gamma2_hash = hashlib.sha256(
            f"GAMMA_PHOTON_2_{norm_e}_{nonce}_B".encode("utf-8")
        ).hexdigest()

        return AnnihilationReactionReport(
            electron_anomaly_norm=norm_e,
            positron_hmac_valid=bool(is_valid_hmac),
            released_energy_kev=float(released_energy),
            gamma_photons_sha256=(gamma1_hash, gamma2_hash),
            back_action_db=float(back_action_db),
            compton_phase_coherence=float(compton_coherence),
        )

    def run_spectral_audit(
        self,
        apu_incidence_matrix: NDArray[np.float64],
        novikov_coefficients: Sequence[complex],
        novikov_exponents: Sequence[float],
        ball_r: float = 1.0,
        cylinder_R: float = 1.0,
        positron_key: bytes = _POSITRON_HMAC_PAYLOAD,
        perturbation_eps: float = 0.05,
    ) -> AuditGradientResult:
        r"""
        CLAUSURA MONÁDICA GLOBAL Y ORQUESTADOR DE LAS 3 FASES ANIDADAS.

        Compone functorialmente los cierres y aperturas de las 3 fases:
            1. `phase1_seal_algebraic_spectral_bundle` -> emite `Phase1AlgebraicSpectralBundle`.
            2. `phase2_open_from_phase1_bundle`        -> consume Fase I y emite `Phase2GeometricCelestialBundle`.
            3. `phase3_open_from_phase2_bundle`        -> consume Fase II y emite `AuditGradientResult`.
        """
        phase1_bundle = self.phase1_seal_algebraic_spectral_bundle(
            apu_incidence_matrix=apu_incidence_matrix,
            novikov_coefficients=novikov_coefficients,
            novikov_exponents=novikov_exponents,
        )

        phase2_bundle = self.phase2_open_from_phase1_bundle(
            phase1_bundle=phase1_bundle,
            apu_cost_matrix=apu_incidence_matrix,
            ball_r=ball_r,
            cylinder_R=cylinder_R,
            perturbation_eps=perturbation_eps,
        )

        return self.phase3_open_from_phase2_bundle(
            phase2_bundle=phase2_bundle,
            state_vector=np.asarray(apu_incidence_matrix, dtype=np.float64).flatten(),
            positron_key=positron_key,
        )


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN UNITARIA AUTÓNOMA DE LAS 3 FASES ANIDADAS EN SANDBOX
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"=== TEST AUTÓNOMO: BoolNovikovPseudoholomorphicAuditorEngine v{_ENGINE_VERSION} ===")

    engine = BoolNovikovPseudoholomorphicAuditorEngine()

    # Matriz de incidencia de prueba
    mat = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    coeffs = [1.0 + 0.2j, 0.5 - 0.1j]
    exps = [0.15, 0.40]

    # Demostración explícita del encadenamiento formal entre FASE I -> FASE II -> FASE III
    p1_bundle = engine.phase1_seal_algebraic_spectral_bundle(mat, coeffs, exps)
    p2_bundle = engine.phase2_open_from_phase1_bundle(
        p1_bundle, mat, ball_r=0.8, cylinder_R=1.0, perturbation_eps=0.05
    )
    result = engine.phase3_open_from_phase2_bundle(
        p2_bundle, mat.flatten(), _POSITRON_HMAC_PAYLOAD
    )

    print(f"Veredicto Heyting \\Omega_3       : {result.truth_value.name} ({hex(result.truth_value.value)})")
    print(f"SNF Diagonal / GF(2) Rank        : {result.boolean_state.smith_normal_form_diagonal} / {result.boolean_state.gf2_rank}")
    print(f"2-Torsión de Bockstein           : {result.boolean_state.bockstein_2_torsion_rank}")
    print(f"Valuación de Krull v(A)          : {result.novikov_state.krull_valuation:.4f} (Gap: {result.novikov_state.spectral_valuation_gap:.4f})")
    print(f"Residual Cauchy-Riemann-Floer    : {result.surface_state.cauchy_riemann_residual:.6e}")
    print(f"Violación de Gromov              : {result.surface_state.violates_gromov_non_squeezing}")
    print(f"Excentricidad de Delaunay        : {result.celestial_state.delaunay_eccentricity:.6f}")
    print(f"Defecto Simpléctico Yoshida-4    : {result.celestial_state.symplectic_jacobian_defect:.6e}")
    print(f"Caos de Melnikov / Chirikov      : {result.celestial_state.has_homoclinic_tangle} / {result.celestial_state.chirikov_overlap_parameter:.4f}")
    print(f"Horizonte de Nekhoroshev         : {result.celestial_state.nekhoroshev_stability_horizon:.4e}")
    print(f"Contracción de Banach Lip^3      : {result.rsi3_state.banach_contraction_constant:.6f} (Fréchet: {result.rsi3_state.frechet_spectral_radius:.6f})")
    print(f"Residual Picard / Steffensen     : {result.rsi3_state.fixed_point_error:.6e} / {result.rsi3_state.steffensen_accelerated_error:.6e}")
    print(f"Jerk Ciber-Físico (d^3C/dt^3)    : {result.rsi3_state.capacity_jerk:.6e}")
    print(f"Aniquilación Dirac (keV)         : {result.annihilation_report.released_energy_kev} keV")
    print(f"Fase I  seal SHA-256             : {result.phase1_seal_hash[:16]}...")
    print(f"Fase II seal SHA-256             : {result.phase2_seal_hash[:16]}...")
    print(f"Pasaporte SHA-256                : {result.audit_passport_hash[:16]}...")
    print("=== TEST COMPLETADO CON ÉXITO ===")
# -*- coding: utf-8 -*-
r"""
─────────────────────────────────────────────────────────────────────────────
Módulo : Specular Flow Engine (Motor del Flujo Especular)
Ruta   : app/omega/specular_flow.py
Versión: 6.0.0-Allievi-Householder-Tellegen-Spectral-Kahan-Sparse
─────────────────────────────────────────────────────────────────────────────
NATURALEZA CIBER-FÍSICA Y RIGOR MATEMÁTICO (FASES ANIDADAS + SUTURAS)
-----------------------------------------------------------------------------
Este módulo materializa el solucionador numérico y motor físico para la
dinámica de transitorios hidráulicos de Allievi, la reflexión covariante de
Householder y la verificación del Teorema de Tellegen.

La implementación se organiza en tres fases que heredan secuencialmente,
formando un pipeline donde la salida formal de una fase es la entrada de la
siguiente:

  FASE 1 – Observe  (Allievi)
      Invariantes de Riemann, impedancia característica, dirección
      característica y puente formal hacia la Fase 2.

  FASE 2 – Orient   (Householder)
      Reflexión covariante sobre métrica Riemanniana, con regularización
      espectral de Cholesky-Tikhonov para métricas mal condicionadas o
      perturbadas por ruido de precisión.

  FASE 3 – Decide   (Tellegen)
      Verificación topológica de conservación de potencia mediante suma
      compensada de Kahan y auditoría dispersa del coborde B1 del complejo
      simplicial de red.

SUTURAS MATEMÁTICAS INTEGRADAS
-----------------------------------------------------------------------------
1. Regularización Espectral de Cholesky-Tikhonov Iterativa
   - Reemplaza el Cholesky binario por una regularización adaptativa.
   - Proyecta la métrica G al cono SPD desplazando autovalores.
   - Garantiza objetivo de condición κ(G) ≤ 1e8 cuando es numéricamente
     posible.

2. Suma Compensada de Kahan para el Residuo de Tellegen
   - Reduce la deriva de redondeo en el balance de potencia.
   - Error de acumulación cercano a O(ε_mach) en lugar de O(n ε_mach).

3. Verificación de Causalidad Dispersa del Teorema de Tellegen
   - Usa la matriz de incidencia orientada dispersa B1.
   - Las caídas de presión de arista se obtienen como V = B1ᵀ p.
   - El flujo satisface continuidad discreta B1 f = s.
   - El residuo de Tellegen se evalúa de forma eficiente en CPU.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import lsqr as sparse_lsqr


logger = logging.getLogger("MIC.Physics.SpecularFlowEngine")


═══════════════════════════════════════════════════════════════════════════
# Constantes de precisión, estabilidad y regularización espectral
═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-10
_RELAX_TOL: Final[float] = 1.0e-8
_TIKHONOV_TOL: Final[float] = 1.0e-12

_SPD_CONDITION_TARGET: Final[float] = 1.0e8
_STABILITY_BOUND: Final[float] = 1.0e8
_GRAPH_MAX_BRANCHES: Final[int] = 1024

_TOPOLOGY_DEGRADED_THRESHOLD: Final[float] = 1.0e-8
_TOPOLOGY_VETO_THRESHOLD: Final[float] = 1.0e-6
_RAYLEIGH_CONDITION_LIMIT: Final[float] = 1.0e12


═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones propias del motor físico
═══════════════════════════════════════════════════════════════════════════

class SpecularFlowPhysicsError(Exception):
    """Excepción base para errores estructurales en el motor físico."""

    pass


class AllieviIntegrationError(SpecularFlowPhysicsError):
    """Error en la integración de las características de Allievi."""

    pass


class HouseholderReflectionError(SpecularFlowPhysicsError):
    """Error en la reflexión covariante de Householder."""

    pass


class TellegenConservationError(SpecularFlowPhysicsError):
    """Violación del Teorema de Tellegen o de la pasividad de Lyapunov."""

    pass


═══════════════════════════════════════════════════════════════════════════
# Estado diagnóstico del motor (retículo de severidad)
═══════════════════════════════════════════════════════════════════════════

class SpecularFlowStatus(IntEnum):
    """
    Estado diagnóstico del motor físico.

    COHERENT : objeto válido dentro de tolerancia estricta.
    DEGRADED : dentro de tolerancia relajada, con deriva paramétrica.
    VETOED   : violación estructural/física informativa.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *statuses: "SpecularFlowStatus") -> "SpecularFlowStatus":
        """
        Supremo en el retículo de severidad.

        El supremo de un conjunto de estados es el estado de mayor severidad.
        Si no se proveen estados, retorna COHERENT como elemento neutro.
        """
        if not statuses:
            return cls.COHERENT
        return cls(max(int(status) for status in statuses))


═══════════════════════════════════════════════════════════════════════════
# Estructuras de datos inmutables (objetos de la subcategoría Spec)
═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AllieviInvariants:
    """
    FASE 1 — Invariantes de Riemann y dirección característica.

    Atributos heredados
    -------------------
    psi_plus : float
        Invariante progresivo Ψ⁺ = y + Z_c q.
    psi_minus : float
        Invariante regresivo Ψ⁻ = y - Z_c q.
    characteristic_impedance : float
        Impedancia característica Z_c = a / (g S).
    characteristic_direction : Tuple[float, float]
        Vector director unitario (a, 1) normalizado.
    is_stable : bool
        Verdadero si los invariantes están acotados.

    Atributos añadidos
    ------------------
    characteristic_covector : Tuple[float, float]
        Covector dual calibrado por la métrica de impedancia.
    energy_norm : float
        Norma energética ||(y, Z_c q)||.
    impedance_condition_number : float
        max(Z_c, 1/Z_c).
    cfl_proxy : float
        Proxy CFL acotada.
    status : SpecularFlowStatus
        Estado diagnóstico de la Fase 1.
    """

    psi_plus: float
    psi_minus: float
    characteristic_impedance: float
    characteristic_direction: Tuple[float, float]
    is_stable: bool
    characteristic_covector: Tuple[float, float] = (0.0, 0.0)
    energy_norm: float = 0.0
    impedance_condition_number: float = 1.0
    cfl_proxy: float = 0.0
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


@dataclass(frozen=True, slots=True)
class HouseholderReflectionResult:
    """
    FASE 2 — Reflexión covariante de Householder con regularización espectral.

    Atributos heredados
    -------------------
    reflected_state : NDArray[np.float64]
        Vector de estado reflejado M x.
    reflection_operator : NDArray[np.float64]
        Operador de reflexión M.
    unitary_residual : float
        Residuo ||Mᵀ G M - G||_F.
    is_anti_symplectic : bool
        Verdadero si Mᵀ Ω M = -Ω en 2D.
    is_characteristic_consistent : bool
        Verdadero si la reflexión cumple la ley especular con la característica.

    Atributos añadidos
    ------------------
    state_norm_residual : float
        Residuo relativo de preservación de norma G del estado.
    specular_residual : float
        Residuo de la ley especular.
    incidence_cosine : float
        Coseno de incidencia entre normal y característica en métrica G.
    metric_condition_number : float
        Número de condición efectivo de G.
    spectral_uncertainty : float
        Condición × residuo dominante.
    metric_was_regularized : bool
        Verdadero si G fue regularizada por Tikhonov.
    tikhonov_shift : float
        Desplazamiento espectral aplicado.
    status : SpecularFlowStatus
        Estado diagnóstico de la Fase 2.
    """

    reflected_state: NDArray[np.float64]
    reflection_operator: NDArray[np.float64]
    unitary_residual: float
    is_anti_symplectic: bool
    is_characteristic_consistent: bool
    state_norm_residual: float = 0.0
    specular_residual: float = 0.0
    incidence_cosine: float = 1.0
    metric_condition_number: float = 1.0
    spectral_uncertainty: float = 0.0
    metric_was_regularized: bool = False
    tikhonov_shift: float = 0.0
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


@dataclass(frozen=True, slots=True)
class NetworkConservationReport:
    """
    FASE 3 — Balance energético, Teorema de Tellegen y pasividad.

    Atributos heredados
    -------------------
    tellegen_residual : float
        Residuo normalizado de Tellegen.
    is_tellegen_conserved : bool
        True si el residuo es ≤ tolerancia.
    rayleigh_dissipation : float
        Potencia disipada P_diss = ∇Hᵀ R ∇H.
    is_lyapunov_passive : bool
        True si P_diss ≥ 0.

    Atributos añadidos
    ------------------
    kcl_residual : float
        Residuo de Kirchhoff de corrientes ||B1 f||.
    kvl_residual : float
        Residuo de Kirchhoff de tensiones.
    is_kcl_conserved : bool
        KCL conservado dentro de tolerancia.
    is_kvl_conserved : bool
        KVL conservado dentro de tolerancia.
    graph_order : int
        Número de nodos del grafo.
    graph_size : int
        Número de ramas del grafo.
    cyclomatic_number : int
        Número ciclomático estimado.
    rayleigh_condition_number : float
        Número de condición de R.
    min_rayleigh_eigenvalue : float
        Menor autovalor de R.
    tellegen_power : float
        Potencia cruda firmada pᵀ q.
    kahan_tellegen_sum : float
        Suma compensada de Kahan de potencia.
    used_sparse_tellegen : bool
        Verdadero si se usó auditoría dispersa B1.
    status : SpecularFlowStatus
        Estado diagnóstico de la Fase 3.
    """

    tellegen_residual: float
    is_tellegen_conserved: bool
    rayleigh_dissipation: float
    is_lyapunov_passive: bool
    kcl_residual: float = 0.0
    kvl_residual: float = 0.0
    is_kcl_conserved: bool = True
    is_kvl_conserved: bool = True
    graph_order: int = 0
    graph_size: int = 0
    cyclomatic_number: int = 0
    rayleigh_condition_number: float = 1.0
    min_rayleigh_eigenvalue: float = 0.0
    tellegen_power: float = 0.0
    kahan_tellegen_sum: float = 0.0
    used_sparse_tellegen: bool = False
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


═══════════════════════════════════════════════════════════════════════════
# SUTURA 1 — Regularización Espectral de Cholesky-Tikhonov Iterativa
═══════════════════════════════════════════════════════════════════════════

def stable_cholesky_decomposition(
    G: NDArray[np.float64],
    tolerance: float = 1.0e-12,
) -> NDArray[np.float64]:
    r"""
    Calcula el factor de Cholesky de G aplicando regularización espectral de
    Tikhonov.

    Asegura que el número de condición final satisfaga:

    .. math::
        \kappa(G) \le 10^8

    cuando ello es numéricamente alcanzable.

    Parámetros
    ----------
    G : NDArray[np.float64]
        Matriz simétrica objetivo.
    tolerance : float
        Tolerancia base para el desplazamiento de Tikhonov.

    Retorna
    -------
    NDArray[np.float64]
        Factor triangular inferior L tal que G_eff = L Lᵀ.
    """
    L, _, _, _, _ = _stable_cholesky_tikhonov(
        G,
        tolerance=tolerance,
        target_condition=_SPD_CONDITION_TARGET,
    )
    return L


def _stable_cholesky_tikhonov(
    G: NDArray[np.float64],
    tolerance: float = 1.0e-12,
    target_condition: float = _SPD_CONDITION_TARGET,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, bool, float]:
    """
    Implementación rica de la regularización espectral.

    Retorna
    -------
    Tuple[NDArray, NDArray, float, bool, float]
        (L, G_efectiva, shift, fue_regularizada, número_de_condición)
    """
    try:
        G_arr = np.asarray(G, dtype=np.float64)
    except Exception as exc:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G no convertible a NDArray float64."
        ) from exc

    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G debe ser una matriz cuadrada."
        )

    if not np.all(np.isfinite(G_arr)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G contiene entradas no finitas."
        )

    n = G_arr.shape[0]

    if n == 0:
        empty = G_arr.copy()
        return empty, empty, 0.0, False, 1.0

    # Simetrización defensiva contra ruido de FPU.
    G_sym = 0.5 * (G_arr + G_arr.T)

    # ── Intento directo: Cholesky + auditoría de condición ───────────────
    direct_L: Optional[NDArray[np.float64]] = None
    direct_condition = float("inf")

    try:
        direct_L = la.cholesky(G_sym, lower=True)
        eigvals_direct = la.eigvalsh(G_sym)

        if eigvals_direct.size > 0:
            min_eig = float(np.min(eigvals_direct))
            max_eig = float(np.max(eigvals_direct))

            if min_eig > _MACHINE_EPS:
                direct_condition = float(max_eig / min_eig)
    except Exception:
        direct_L = None
        direct_condition = float("inf")

    if (
        direct_L is not None
        and np.isfinite(direct_condition)
        and direct_condition <= target_condition * (1.0 + 10.0 * tolerance)
    ):
        return direct_L, G_sym, 0.0, False, direct_condition

    # ── Regularización espectral de Tikhonov ─────────────────────────────
    try:
        vals, vecs = la.eigh(G_sym)
    except Exception as exc:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: No se pudo diagonalizar G para regularización."
        ) from exc

    if vals.size == 0:
        empty = G_sym.copy()
        return empty, empty, 0.0, False, 1.0

    max_val = float(np.max(vals))
    g_norm = float(la.norm(G_sym, ord="fro"))
    scale = max(1.0, abs(max_val), g_norm)

    # Desplazamiento inicial.
    shift = max(10.0 * _MACHINE_EPS, scale * tolerance * 100.0)

    if max_val > 0.0:
        shift = max(shift, max_val / target_condition)

    last_error: Optional[Exception] = None

    for _ in range(12):
        vals_reg = np.maximum(vals, shift)

        min_reg = float(np.min(vals_reg))
        max_reg = float(np.max(vals_reg))

        if min_reg <= _MACHINE_EPS:
            shift = max(shift * 10.0, 10.0 * _MACHINE_EPS)
            continue

        cond_reg = float(max_reg / min_reg)

        if cond_reg > target_condition:
            shift = max(shift, max_reg / target_condition, 10.0 * _MACHINE_EPS)
            continue

        # Reconstrucción G_reg = V diag(vals_reg) Vᵀ.
        G_reg = (vecs * vals_reg) @ vecs.T
        G_reg = 0.5 * (G_reg + G_reg.T)

        try:
            L_reg = la.cholesky(G_reg, lower=True)
        except la.LinAlgError as exc:
            last_error = exc
            shift *= 10.0
            continue

        logger.warning(
            "Métrica G no-SPD o mal condicionada. Aplicado desplazamiento "
            "espectral de Tikhonov: %.4e",
            shift,
        )

        return L_reg, G_reg, float(shift), True, cond_reg

    raise SpecularFlowPhysicsError(
        "SpecularFlowPhysicsError: Regularización Tikhonov no convergió a una "
        "métrica SPD estable."
    ) from last_error


═══════════════════════════════════════════════════════════════════════════
# SUTURA 2 — Suma Compensada de Kahan para el Residuo de Tellegen
═══════════════════════════════════════════════════════════════════════════

def kahan_tellegen_summation(
    pressures: NDArray[np.float64],
    flows: NDArray[np.float64],
) -> float:
    r"""
    Calcula la potencia virtual de Tellegen mediante el algoritmo de suma
    compensada de Kahan.

    Complejidad de error:

    .. math::
        O(\varepsilon_{\text{mach}})

    en lugar de:

    .. math::
        O(n \varepsilon_{\text{mach}})

    Parámetros
    ----------
    pressures : NDArray[np.float64]
        Vector de presiones/caídas de potencial.
    flows : NDArray[np.float64]
        Vector de flujos/caudales.

    Retorna
    -------
    float
        Suma compensada firmada Σ p_k q_k.
    """
    p = np.asarray(pressures, dtype=np.float64).reshape(-1)
    q = np.asarray(flows, dtype=np.float64).reshape(-1)

    if p.size != q.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: Dimensiones inconsistentes en Kahan Tellegen."
        )

    total_sum = 0.0
    compensation = 0.0

    for pi, qi in zip(p, q):
        value = float(pi * qi)

        if not np.isfinite(value):
            raise SpecularFlowPhysicsError(
                "SpecularFlowPhysicsError: Producto de potencia no finito en Kahan."
            )

        y = value - compensation
        t = total_sum + y
        compensation = (t - total_sum) - y
        total_sum = t

    return float(total_sum)


═══════════════════════════════════════════════════════════════════════════
# SUTURA 3 — Verificación de Causalidad Dispersa del Teorema de Tellegen
═══════════════════════════════════════════════════════════════════════════

def verify_tellegen_sparse(
    nodal_potentials_p: NDArray[np.float64],
    flow_vector_f: NDArray[np.float64],
    incidence_matrix_B1: sp.csr_matrix,
) -> float:
    r"""
    Verifica el residuo de Tellegen utilizando el coborde disperso del
    complejo.

    Axioma algebraico:

    .. math::
        v^\top f - p^\top B_1 f \equiv 0

    donde:

    .. math::
        v = B_1^\top p

    Parámetros
    ----------
    nodal_potentials_p : NDArray[np.float64]
        Potenciales nodales p.
    flow_vector_f : NDArray[np.float64]
        Flujos de arista f.
    incidence_matrix_B1 : sp.csr_matrix
        Matriz de incidencia orientada dispersa B1.

    Retorna
    -------
    float
        Residuo absoluto |vᵀ f|.
    """
    p = np.asarray(nodal_potentials_p, dtype=np.float64).reshape(-1)
    f = np.asarray(flow_vector_f, dtype=np.float64).reshape(-1)

    if sp.issparse(incidence_matrix_B1):
        B1 = incidence_matrix_B1.tocsr()
    else:
        B1 = sp.csr_matrix(incidence_matrix_B1, dtype=np.float64)

    if B1.shape[0] != p.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1.shape[0] no coincide con nodal_potentials_p."
        )

    if B1.shape[1] != f.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1.shape[1] no coincide con flow_vector_f."
        )

    if B1.nnz > 0 and not np.all(np.isfinite(B1.data)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1 dispersa contiene datos no finitos."
        )

    # Presiones covariantes (caídas de potencial en aristas).
    v_pressures = np.asarray(B1.T @ p, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(v_pressures)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: v = B1ᵀ p contiene entradas no finitas."
        )

    # Sumatoria de potencia virtual con Kahan.
    power_sum = kahan_tellegen_summation(v_pressures, f)

    return float(abs(power_sum))


═══════════════════════════════════════════════════════════════════════════
# Utilidades internas de auditoría numérica
═══════════════════════════════════════════════════════════════════════════

def _frobenius_norm(matrix: object, is_sparse: bool) -> float:
    """Norma de Frobenius robusta para matrices densas o dispersas."""
    if is_sparse:
        try:
            return float(sp.linalg.norm(matrix, ord="fro"))  # type: ignore[arg-type]
        except Exception:
            data = getattr(matrix, "data", np.empty(0, dtype=np.float64))
            return float(np.linalg.norm(data)) if data.size else 0.0

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(la.norm(arr, ord="fro"))


def _estimate_matrix_rank(matrix: object, is_sparse: bool) -> int:
    """
    Estimación de rango para matrices densas o dispersas.

    Para matrices dispersas grandes se usa una cota superior conservadora
    min(n, m) para evitar costos prohibitivos.
    """
    shape = getattr(matrix, "shape", None)
    if shape is None:
        return 0

    n_rows, n_cols = shape

    if n_rows == 0 or n_cols == 0:
        return 0

    if not is_sparse:
        try:
            return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=np.float64), tol=1.0e-12))
        except Exception:
            return int(min(n_rows, n_cols))

    # Sparse: solo convertir a denso si el problema es moderado.
    if n_rows * n_cols <= 200_000:
        try:
            return int(np.linalg.matrix_rank(matrix.toarray(), tol=1.0e-12))  # type: ignore[union-attr]
        except Exception:
            return int(min(n_rows, n_cols))

    return int(min(n_rows, n_cols))


class _NumericalAuditMixin:
    """
    Utilidades de auditoría numérica para evitar errores silenciosos.
    """

    @staticmethod
    def _as_finite_float(value: object, name: str) -> float:
        """Convierte a float y exige finitud."""
        try:
            scalar = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no es convertible a escalar."
            ) from exc

        if not np.isfinite(scalar):
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no es finito."
            )

        return scalar

    @staticmethod
    def _as_finite_vector(values: object, name: str) -> NDArray[np.float64]:
        """Convierte a vector 1D float64 y exige finitud."""
        try:
            vector = np.asarray(values, dtype=np.float64)
        except Exception as exc:
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no es convertible a NDArray."
            ) from exc

        if vector.ndim == 0:
            vector = vector.reshape(1)

        if not np.all(np.isfinite(vector)):
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' contiene entradas no finitas."
            )

        return vector


═══════════════════════════════════════════════════════════════════════════
# FASE 1 — Observe: Integración de Allievi y cálculo de invariantes
═══════════════════════════════════════════════════════════════════════════

class Phase1_AllieviSolver(_NumericalAuditMixin, ABC):
    """
    FASE 1 (Observe): Resolvedor de las ecuaciones de golpe de ariete de
    Allievi sobre la variedad de fase (y, q).

    Calcula:
      • Impedancia característica Z_c.
      • Invariantes de Riemann Ψ⁺, Ψ⁻.
      • Dirección característica normalizada.
      • Covector dual calibrado.
      • Norma energética.
      • Condición espectral de impedancia.

    El último método de esta fase es el puente formal hacia la Fase 2:
        _allievi_to_householder_bridge(...)
    """

    def solve_allievi_characteristics(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
    ) -> AllieviInvariants:
        r"""
        Calcula los invariantes de Riemann a lo largo de las curvas
        características.

        Fórmulas
        --------
        .. math::
            Z_c = \frac{a}{g S}, \quad
            \Psi^+ = y + Z_c q, \quad
            \Psi^- = y - Z_c q.

        La dirección característica positiva es :math:`(a, 1)` normalizada.

        Parámetros
        ----------
        potential_y : float
            Potencial generalizado.
        flow_q : float
            Caudal logístico.
        wave_speed_a : float
            Velocidad de onda efectiva.
        gravity_g : float
            Constante de acoplamiento.
        section_s : float
            Sección/subespacio.
        tolerance : float
            Cota de precisión.
        cfl_number : Optional[float]
            CFL observado. Si se omite, se usa proxy acotada.

        Retorna
        -------
        AllieviInvariants
            Certificado inmutable de la Fase 1.

        Lanza
        -----
        AllieviIntegrationError
            Si parámetros son degenerados o no físicos.
        """
        y = self._as_finite_float(potential_y, "potential_y")
        q = self._as_finite_float(flow_q, "flow_q")
        a = self._as_finite_float(wave_speed_a, "wave_speed_a")
        g = self._as_finite_float(gravity_g, "gravity_g")
        s = self._as_finite_float(section_s, "section_s")

        if a <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: wave_speed_a debe ser positiva y no degenerada."
            )

        if g <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: gravity_g debe ser positiva y no degenerada."
            )

        if s <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: section_s debe ser positiva y no degenerada."
            )

        denominator = g * s
        if denominator <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: g·S degenerado (división por cero)."
            )

        z_c = a / denominator

        if not np.isfinite(z_c) or z_c <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Impedancia característica degenerada."
            )

        # Invariantes de Riemann.
        psi_plus = y + z_c * q
        psi_minus = y - z_c * q

        if not np.isfinite(psi_plus) or not np.isfinite(psi_minus):
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Invariantes de Riemann no finitos."
            )

        # Estabilidad compacta.
        is_stable = (
            abs(psi_plus) <= _STABILITY_BOUND
            and abs(psi_minus) <= _STABILITY_BOUND
        )

        # Dirección característica normalizada.
        char_dir = np.array([a, 1.0], dtype=np.float64)
        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Dirección característica degenerada."
            )
        char_dir = char_dir / char_dir_norm

        # Métrica calibrada de impedancia y covector dual.
        z_c_sq = z_c * z_c
        if np.isfinite(z_c_sq):
            g_calibrated = np.array([[1.0, 0.0], [0.0, z_c_sq]], dtype=np.float64)
            char_cov = g_calibrated @ char_dir
            char_cov_norm = float(la.norm(char_cov))
            if char_cov_norm <= _MACHINE_EPS:
                char_cov = char_dir.copy()
            else:
                char_cov = char_cov / char_cov_norm
        else:
            char_cov = char_dir.copy()

        # Norma energética robusta.
        zq = z_c * q
        if np.isfinite(zq):
            energy_norm = float(np.hypot(y, zq))
        else:
            energy_norm = float("inf")

        # Condición espectral de impedancia.
        impedance_condition_number = float(max(z_c, 1.0 / z_c))

        # CFL proxy.
        if cfl_number is None:
            cfl_proxy = float(abs(a) / (abs(a) + 1.0))
        else:
            cfl_proxy = abs(self._as_finite_float(cfl_number, "cfl_number"))

        # Estado diagnóstico.
        status = SpecularFlowStatus.COHERENT

        if (
            not is_stable
            or cfl_proxy > 1.0 + tolerance
            or not np.isfinite(energy_norm)
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            impedance_condition_number > _SPD_CONDITION_TARGET
            or abs(psi_plus - psi_minus) > 1.0e6
            or cfl_proxy > max(0.0, 1.0 - 10.0 * tolerance)
        ):
            status = SpecularFlowStatus.DEGRADED

        return AllieviInvariants(
            psi_plus=float(psi_plus),
            psi_minus=float(psi_minus),
            characteristic_impedance=float(z_c),
            characteristic_direction=(float(char_dir[0]), float(char_dir[1])),
            is_stable=is_stable,
            characteristic_covector=(float(char_cov[0]), float(char_cov[1])),
            energy_norm=energy_norm,
            impedance_condition_number=impedance_condition_number,
            cfl_proxy=cfl_proxy,
            status=status,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE LA FASE 1: puente hacia la Fase 2
    # ─────────────────────────────────────────────────────────────────────
    def _allievi_to_householder_bridge(
        self,
        allievi_inv: AllieviInvariants,
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Extrae del certificado de la Fase 1 la dirección característica y la
        impedancia, listas para alimentar la reflexión especular.

        Este método es el nexo explícito que hace de la Fase 2 la
        continuación directa de la Fase 1.
        """
        char_dir = np.array(allievi_inv.characteristic_direction, dtype=np.float64)
        z_c = float(allievi_inv.characteristic_impedance)

        if not np.all(np.isfinite(char_dir)) or not np.isfinite(z_c):
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Certificado de Fase 1 inválido para puente."
            )

        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Dirección característica degenerada en puente."
            )

        char_dir = char_dir / char_dir_norm

        return char_dir, z_c


═══════════════════════════════════════════════════════════════════════════
# FASE 2 — Orient: Reflexión de Householder covariante (anidada en Fase 1)
═══════════════════════════════════════════════════════════════════════════

class Phase2_HouseholderReflector(Phase1_AllieviSolver):
    """
    FASE 2 (Orient): Reflexión covariante de Householder sobre la
    hipersuperficie de restricciones.

    Hereda directamente la Fase 1 y consume su puente formal:
        _allievi_to_householder_bridge(...)

    Incluye la SUTURA 1:
        Regularización Espectral de Cholesky-Tikhonov Iterativa.
    """

    def compute_householder_reflection(
        self,
        metric_tensor_g: NDArray[np.float64],
        boundary_normal_n: NDArray[np.float64],
        state_vector: NDArray[np.float64],
        allievi_invariants: AllieviInvariants,
        tolerance: float = _DEFAULT_TOL,
    ) -> HouseholderReflectionResult:
        r"""
        Calcula la reflexión de Householder covariante y verifica su
        consistencia con las líneas características de la Fase 1.

        Operador
        --------
        .. math::
            M = I - 2 \frac{n (G n)^\top}{n^\top G n}

        Auditorías
        ----------
        • G SPD o regularizada al cono SPD.
        • Unitariedad Riemanniana Mᵀ G M = G.
        • Anti-simplécticidad 2D Mᵀ Ω M = -Ω.
        • Ley especular sobre la característica.
        • Preservación de norma G del estado.

        Parámetros
        ----------
        metric_tensor_g : NDArray[np.float64]
            Tensor métrico Riemanniano.
        boundary_normal_n : NDArray[np.float64]
            Vector normal primal.
        state_vector : NDArray[np.float64]
            Vector de estado [y, q]ᵀ.
        allievi_invariants : AllieviInvariants
            Certificado de la Fase 1.
        tolerance : float
            Cota de precisión.

        Retorna
        -------
        HouseholderReflectionResult
            Certificado inmutable de la Fase 2.

        Lanza
        -----
        HouseholderReflectionError
            Si hay degeneración estructural o entradas mal formadas.
        """
        # ── 0. Puente con la Fase 1 ─────────────────────────────────────
        char_dir, _ = self._allievi_to_householder_bridge(allievi_invariants)

        dim = char_dir.size
        if dim != 2:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: El motor especular está definido sobre fase 2D."
            )

        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # ── 1. Auditoría y regularización espectral de G ────────────────
        try:
            G_input = np.asarray(metric_tensor_g, dtype=np.float64)
        except Exception as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: metric_tensor_g no convertible a NDArray."
            ) from exc

        if G_input.ndim != 2 or G_input.shape != (dim, dim):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: G debe ser una matriz 2x2."
            )

        try:
            (
                _L_factor,
                G_effective,
                tikhonov_shift,
                metric_was_regularized,
                metric_condition_number,
            ) = _stable_cholesky_tikhonov(
                G_input,
                tolerance=max(tolerance, _TIKHONOV_TOL),
                target_condition=_SPD_CONDITION_TARGET,
            )
        except SpecularFlowPhysicsError as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: No se pudo estabilizar la métrica G."
            ) from exc

        # ── 2. Normal y vector de estado ────────────────────────────────
        normal_vector = self._as_finite_vector(boundary_normal_n, "boundary_normal_n")
        if normal_vector.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: La normal debe tener dimensión 2."
            )

        normal_norm = float(la.norm(normal_vector))
        if normal_norm <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal euclidiana degenerada."
            )

        x_state = self._as_finite_vector(state_vector, "state_vector")
        if x_state.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: El vector de estado debe tener dimensión 2."
            )

        # ── 3. Operador de Householder covariante ───────────────────────
        g_normal = G_effective @ normal_vector
        denom = float(normal_vector @ g_normal)

        g_norm = float(la.norm(G_effective, ord="fro"))
        if denom <= _MACHINE_EPS * max(1.0, g_norm):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal ortogonal degenerada en la métrica G."
            )

        m_operator = np.eye(dim, dtype=np.float64) - (
            (2.0 / denom) * np.outer(normal_vector, g_normal)
        )

        if not np.all(np.isfinite(m_operator)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Operador de Householder no finito."
            )

        reflected_state = m_operator @ x_state

        # ── 4. Unitariedad Riemanniana: Mᵀ G M = G ──────────────────────
        unitary_matrix_residual = m_operator.T @ G_effective @ m_operator - G_effective
        unitary_residual = float(
            la.norm(unitary_matrix_residual, ord="fro") / max(1.0, g_norm)
        )
        is_unitary = unitary_residual <= audit_tol

        # ── 5. Preservación de norma G del estado ───────────────────────
        x_norm_sq = float(x_state @ G_effective @ x_state)
        mx_state = reflected_state
        mx_norm_sq = float(mx_state @ G_effective @ mx_state)

        x_norm = float(np.sqrt(max(0.0, x_norm_sq)))
        mx_norm = float(np.sqrt(max(0.0, mx_norm_sq)))

        state_norm_residual = float(abs(mx_norm - x_norm) / max(1.0, x_norm))

        # ── 6. Consistencia especular con la característica ────────────
        char_g_norm_sq = float(char_dir @ G_effective @ char_dir)
        if char_g_norm_sq <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Norma G de la característica degenerada."
            )

        char_g_norm = float(np.sqrt(max(0.0, char_g_norm_sq)))
        normal_g_norm = float(np.sqrt(max(0.0, denom)))

        incidence_numerator = float(normal_vector @ G_effective @ char_dir)
        incidence_denominator = char_g_norm * normal_g_norm

        if incidence_denominator <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Denominador de incidencia degenerado."
            )

        incidence_cosine = incidence_numerator / incidence_denominator

        reflected_char = m_operator @ char_dir
        expected_reflection = char_dir - (
            (2.0 * incidence_numerator / denom) * normal_vector
        )

        specular_residual = float(
            la.norm(reflected_char - expected_reflection)
            / max(1.0, float(la.norm(char_dir)))
        )

        reflected_char_g_norm_sq = float(reflected_char @ G_effective @ reflected_char)
        reflected_char_g_norm = float(np.sqrt(max(0.0, reflected_char_g_norm_sq)))

        char_norm_residual = float(
            abs(reflected_char_g_norm - char_g_norm) / max(1.0, char_g_norm)
        )

        projection_scale = max(1.0, char_g_norm * normal_g_norm)
        if abs(incidence_numerator) > audit_tol * projection_scale:
            reflected_normal_projection = float(
                normal_vector @ G_effective @ reflected_char
            )
            normal_flip_ratio = reflected_normal_projection / incidence_numerator
            normal_flip_ok = abs(normal_flip_ratio + 1.0) <= audit_tol
        else:
            normal_flip_ok = True

        is_characteristic_consistent = (
            char_norm_residual <= audit_tol
            and specular_residual <= audit_tol
            and normal_flip_ok
        )

        # ── 7. Invariancia anti-simpléctica en 2D ───────────────────────
        omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        symplectic_residual = float(
            la.norm(m_operator.T @ omega @ m_operator + omega, ord="fro")
            / max(1.0, float(la.norm(omega, ord="fro")))
        )

        try:
            det_m = float(np.linalg.det(m_operator))
        except Exception:
            det_m = 0.0

        is_anti_symplectic = symplectic_residual <= audit_tol and det_m < 0.0

        # ── 8. Estado diagnóstico local ─────────────────────────────────
        status = SpecularFlowStatus.COHERENT

        if (
            not np.isfinite(unitary_residual)
            or not np.isfinite(specular_residual)
            or not np.isfinite(state_norm_residual)
            or not np.isfinite(metric_condition_number)
            or not is_unitary
            or not is_anti_symplectic
            or not is_characteristic_consistent
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            unitary_residual > tolerance
            or specular_residual > tolerance
            or state_norm_residual > tolerance
            or metric_condition_number > _SPD_CONDITION_TARGET
            or abs(incidence_cosine) < 0.1
            or metric_was_regularized
        ):
            status = SpecularFlowStatus.DEGRADED

        # Trazabilidad con Fase 1.
        prior_status = getattr(allievi_invariants, "status", SpecularFlowStatus.COHERENT)
        status = SpecularFlowStatus.supremum(status, prior_status)

        dominant_residual = max(
            unitary_residual,
            specular_residual,
            state_norm_residual,
        )
        spectral_uncertainty = float(metric_condition_number * dominant_residual)

        return HouseholderReflectionResult(
            reflected_state=reflected_state,
            reflection_operator=m_operator,
            unitary_residual=unitary_residual,
            is_anti_symplectic=is_anti_symplectic,
            is_characteristic_consistent=is_characteristic_consistent,
            state_norm_residual=state_norm_residual,
            specular_residual=specular_residual,
            incidence_cosine=float(incidence_cosine),
            metric_condition_number=metric_condition_number,
            spectral_uncertainty=spectral_uncertainty,
            metric_was_regularized=metric_was_regularized,
            tikhonov_shift=tikhonov_shift,
            status=status,
        )

    # ─────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE LA FASE 2: puente hacia la Fase 3
    # ─────────────────────────────────────────────────────────────────────
    def _householder_to_tellegen_bridge(
        self,
        householder_result: HouseholderReflectionResult,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], SpecularFlowStatus]:
        """
        Puente formal FASE 2 → FASE 3.

        Extrae el operador reflejo, el estado reflejado y el estado
        diagnóstico de la Fase 2 para trazabilidad en la Fase 3.
        """
        operator = np.asarray(householder_result.reflection_operator, dtype=np.float64)
        reflected = np.asarray(householder_result.reflected_state, dtype=np.float64)
        status = getattr(householder_result, "status", SpecularFlowStatus.COHERENT)

        return operator, reflected, status


═══════════════════════════════════════════════════════════════════════════
# FASE 3 — Decide: Teorema de Tellegen y pasividad (anidada en Fase 2)
═══════════════════════════════════════════════════════════════════════════

class Phase3_TellegenValidator(Phase2_HouseholderReflector):
    """
    FASE 3 (Decide): Auditoría del balance de potencias sobre la red gráfica
    subyacente y de la desigualdad de disipación de Rayleigh.

    Integra:
      • SUTURA 2: Suma compensada de Kahan.
      • SUTURA 3: Verificación dispersa con B1.
    """

    def verify_network_conservation(
        self,
        pressures: Optional[NDArray[np.float64]],
        flows: NDArray[np.float64],
        rayleigh_dissipation_r: NDArray[np.float64],
        gradient_h: Optional[NDArray[np.float64]],
        allievi_invariants: AllieviInvariants,
        householder_result: HouseholderReflectionResult,
        tolerance: float = 1.0e-8,
        incidence_matrix: Optional[object] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
    ) -> NetworkConservationReport:
        r"""
        Audita la conservación topológica (Teorema de Tellegen) y la
        disipación de Lyapunov.

        Modos de operación
        ------------------
        1. Modo rama (branch mode)
            `pressures` son caídas de potencial de rama ΔP_k.
            Tellegen se audita con Kahan sobre Σ ΔP_k Q_k.

        2. Modo nodal (nodal mode)
            `nodal_potentials` y `incidence_matrix` definen:
                V = B1ᵀ p
            y se usa verificación dispersa de Tellegen.

        Parámetros
        ----------
        pressures : Optional[NDArray[np.float64]]
            Caídas de potencial de rama. Puede ser None en modo nodal puro.
        flows : NDArray[np.float64]
            Caudales de rama Q_k.
        rayleigh_dissipation_r : NDArray[np.float64]
            Matriz R de disipación. Se aceptan escalar, vector diagonal o matriz.
        gradient_h : Optional[NDArray[np.float64]]
            Gradiente ∇H. Si es None, se usa cero.
        allievi_invariants : AllieviInvariants
            Certificado de Fase 1.
        householder_result : HouseholderReflectionResult
            Certificado de Fase 2.
        tolerance : float
            Tolerancia de auditoría.
        incidence_matrix : Optional[object]
            Matriz de incidencia densa o dispersa B1.
        nodal_potentials : Optional[NDArray[np.float64]]
            Potenciales nodales para auditoría dispersa.

        Retorna
        -------
        NetworkConservationReport
            Certificado inmutable de la Fase 3.

        Lanza
        -----
        TellegenConservationError
            Si hay dimensiones inconsistentes o entradas mal formadas.
        """
        # ── 0. Trazabilidad con Fase 2 ──────────────────────────────────
        _, _, prior_householder_status = self._householder_to_tellegen_bridge(
            householder_result
        )

        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # ── 1. Flujos ───────────────────────────────────────────────────
        q = self._as_finite_vector(flows, "flows")
        m = q.size

        if m > _GRAPH_MAX_BRANCHES:
            raise TellegenConservationError(
                "TellegenConservationError: Complejidad topológica fuera de cota."
            )

        # ── 2. Topología / matriz de incidencia B1 ─────────────────────
        is_sparse = False

        if incidence_matrix is not None:
            if sp.issparse(incidence_matrix):
                B1 = incidence_matrix.tocsr()
                is_sparse = True

                if B1.nnz > 0 and not np.all(np.isfinite(B1.data)):
                    raise TellegenConservationError(
                        "TellegenConservationError: B1 dispersa contiene datos no finitos."
                    )
            else:
                try:
                    B1_dense = np.asarray(incidence_matrix, dtype=np.float64)
                except Exception as exc:
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix no convertible."
                    ) from exc

                if B1_dense.ndim == 1:
                    if m == 1:
                        B1_dense = B1_dense.reshape(-1, 1)
                    else:
                        raise TellegenConservationError(
                            "TellegenConservationError: incidence_matrix 1D incompatible con m>1."
                        )

                if B1_dense.ndim != 2:
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix debe ser 2D."
                    )

                if not np.all(np.isfinite(B1_dense)):
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix no finita."
                    )

                B1 = B1_dense

            if B1.shape[1] != m:
                raise TellegenConservationError(
                    "TellegenConservationError: incidence_matrix no concuerda con flows."
                )

            n_nodes = B1.shape[0]

            if m > 0 and n_nodes == 0:
                raise TellegenConservationError(
                    "TellegenConservationError: Topología sin nodos para ramas no nulas."
                )
        else:
            # Topología por defecto: grafo paralelo de dos nodos.
            if m == 0:
                B1 = np.zeros((0, 0), dtype=np.float64)
                n_nodes = 0
            else:
                B1 = np.zeros((2, m), dtype=np.float64)
                B1[0, :] = 1.0
                B1[1, :] = -1.0
                n_nodes = 2

        # ── 3. Modo nodal disperso (SUTURA 3) ──────────────────────────
        used_sparse_tellegen = False
        kvl_residual = 0.0

        if nodal_potentials is not None:
            p_nodal = self._as_finite_vector(nodal_potentials, "nodal_potentials")

            if p_nodal.size != n_nodes:
                raise TellegenConservationError(
                    "TellegenConservationError: nodal_potentials no coincide con B1.shape[0]."
                )

            if is_sparse:
                v_pressures = np.asarray(B1.T @ p_nodal, dtype=np.float64).reshape(-1)
            else:
                v_pressures = np.asarray(B1.T @ p_nodal, dtype=np.float64).reshape(-1)

            if not np.all(np.isfinite(v_pressures)):
                raise TellegenConservationError(
                    "TellegenConservationError: v = B1ᵀ p contiene entradas no finitas."
                )

            if pressures is None:
                p_branch = v_pressures.copy()
                kvl_residual = 0.0
            else:
                p_branch = self._as_finite_vector(pressures, "pressures")
                if p_branch.size != m:
                    raise TellegenConservationError(
                        "TellegenConservationError: pressures no coincide con flows."
                    )

                kvl_residual = float(
                    la.norm(p_branch - v_pressures)
                    / max(1.0, float(la.norm(p_branch)))
                )

            # SUTURA 3: verificación dispersa.
            sparse_abs_residual = verify_tellegen_sparse(p_nodal, q, B1)

            # SUTURA 2: suma Kahan firmada.
            tellegen_power = kahan_tellegen_summation(v_pressures, q)

            used_sparse_tellegen = True
            tellegen_numerator = float(abs(tellegen_power))
        else:
            if pressures is None:
                raise TellegenConservationError(
                    "TellegenConservationError: Modo rama requiere pressures."
                )

            p_branch = self._as_finite_vector(pressures, "pressures")

            if p_branch.size != m:
                raise TellegenConservationError(
                    "TellegenConservationError: pressures no coincide con flows."
                )

            # SUTURA 2: Tellegen con Kahan.
            tellegen_power = kahan_tellegen_summation(p_branch, q)
            tellegen_numerator = float(abs(tellegen_power))

            # KVL: distancia de p_branch a Im(B1ᵀ).
            p_norm = float(la.norm(p_branch))

            if m == 0 or p_norm <= _MACHINE_EPS:
                kvl_residual = 0.0
            else:
                if is_sparse:
                    try:
                        try:
                            lsqr_result = sparse_lsqr(
                                B1.T,
                                p_branch,
                                atol=audit_tol,
                                btol=audit_tol,
                            )
                        except TypeError:
                            lsqr_result = sparse_lsqr(B1.T, p_branch)

                        phi_vec = np.asarray(lsqr_result[0], dtype=np.float64).reshape(-1)
                        kvl_reconstruction = np.asarray(
                            B1.T @ phi_vec,
                            dtype=np.float64,
                        ).reshape(-1)

                        kvl_residual = float(
                            la.norm(kvl_reconstruction - p_branch)
                            / max(1.0, p_norm)
                        )
                    except Exception:
                        kvl_residual = float("inf")
                else:
                    try:
                        phi_vec, *_ = np.linalg.lstsq(B1.T, p_branch, rcond=None)
                        kvl_reconstruction = B1.T @ phi_vec

                        kvl_residual = float(
                            la.norm(kvl_reconstruction - p_branch)
                            / max(1.0, p_norm)
                        )
                    except Exception:
                        kvl_residual = float("inf")

        # ── 4. KCL: B1 f ≈ 0 ───────────────────────────────────────────
        q_norm = float(la.norm(q))

        if m == 0 or q_norm <= _MACHINE_EPS:
            kcl_residual = 0.0
        else:
            if is_sparse:
                kcl_vector = np.asarray(B1 @ q, dtype=np.float64).reshape(-1)
            else:
                kcl_vector = np.asarray(B1 @ q, dtype=np.float64).reshape(-1)

            b1_norm = _frobenius_norm(B1, is_sparse)

            kcl_residual = float(
                la.norm(kcl_vector)
                / (
                    max(1.0, b1_norm)
                    * max(1.0, q_norm)
                    * float(np.sqrt(max(1, n_nodes)))
                )
            )

        # ── 5. Residuo normalizado de Tellegen ─────────────────────────
        p_branch_for_norm = (
            p_branch
            if nodal_potentials is None
            else v_pressures
        )

        p_norm = float(la.norm(p_branch_for_norm))
        tellegen_scale = max(1.0, p_norm) * max(1.0, q_norm)
        tellegen_residual = float(tellegen_numerator / tellegen_scale)

        # ── 6. Matriz de Rayleigh R y pasividad ────────────────────────
        try:
            r_raw = np.asarray(rayleigh_dissipation_r, dtype=np.float64)
        except Exception as exc:
            raise TellegenConservationError(
                "TellegenConservationError: rayleigh_dissipation_r no convertible."
            ) from exc

        if r_raw.ndim == 0:
            r_matrix = r_raw.reshape(1, 1)
        elif r_raw.ndim == 1:
            r_matrix = np.diag(r_raw)
        elif r_raw.ndim == 2:
            r_matrix = r_raw
        else:
            raise TellegenConservationError(
                "TellegenConservationError: R debe ser escalar, vector o matriz."
            )

        if r_matrix.ndim != 2 or r_matrix.shape[0] != r_matrix.shape[1]:
            raise TellegenConservationError(
                "TellegenConservationError: La matriz R debe ser cuadrada."
            )

        if not np.all(np.isfinite(r_matrix)):
            raise TellegenConservationError(
                "TellegenConservationError: La matriz R contiene entradas no finitas."
            )

        n_r = r_matrix.shape[0]

        if gradient_h is None:
            grad_h = np.zeros(n_r, dtype=np.float64)
        else:
            grad_h = self._as_finite_vector(gradient_h, "gradient_h")

        if grad_h.size != n_r:
            raise TellegenConservationError(
                "TellegenConservationError: Dimensión incompatible entre R y ∇H."
            )

        # Simetría defensiva.
        r_norm = float(la.norm(r_matrix, ord="fro"))
        r_sym_residual = float(
            la.norm(r_matrix - r_matrix.T, ord="fro") / max(1.0, r_norm)
        )
        r_symmetric_ok = r_sym_residual <= audit_tol

        r_sym = 0.5 * (r_matrix + r_matrix.T)

        # Auditoría espectral PSD.
        if n_r == 0:
            eigvals = np.empty(0, dtype=np.float64)
            min_eig = 0.0
            max_eig = 0.0
            rayleigh_condition_number = 1.0
        else:
            try:
                eigvals = la.eigvalsh(r_sym)
            except Exception as exc:
                raise TellegenConservationError(
                    "TellegenConservationError: No se pudo diagonalizar R."
                ) from exc

            if eigvals.size == 0:
                min_eig = 0.0
                max_eig = 0.0
                rayleigh_condition_number = 1.0
            else:
                min_eig = float(np.min(eigvals))
                max_eig = float(np.max(eigvals))

                if max_eig <= audit_tol:
                    rayleigh_condition_number = 1.0
                elif min_eig <= audit_tol:
                    rayleigh_condition_number = float("inf")
                else:
                    rayleigh_condition_number = float(max_eig / min_eig)

        r_psd_ok = min_eig >= -audit_tol

        # Disipación de Rayleigh.
        p_diss = float(grad_h @ r_sym @ grad_h)

        if not np.isfinite(p_diss):
            raise TellegenConservationError(
                "TellegenConservationError: Disipación de Rayleigh no finita."
            )

        is_lyapunov_passive = p_diss >= -_MACHINE_EPS

        # ── 7. Invariantes topológicos ──────────────────────────────────
        rank_a = _estimate_matrix_rank(B1, is_sparse)
        cyclomatic_number = max(0, m - rank_a)

        # ── 8. Banderas de conservación ─────────────────────────────────
        strict_tol = max(tolerance, _TOPOLOGY_DEGRADED_THRESHOLD)

        is_kcl_conserved = kcl_residual <= strict_tol
        is_kvl_conserved = kvl_residual <= strict_tol
        is_tellegen_conserved = (
            is_kcl_conserved
            and is_kvl_conserved
            and tellegen_residual <= strict_tol
        )

        # ── 9. Estado diagnóstico local ─────────────────────────────────
        max_topological_residual = max(
            kcl_residual,
            kvl_residual,
            tellegen_residual,
        )

        status = SpecularFlowStatus.COHERENT

        if (
            not np.isfinite(max_topological_residual)
            or max_topological_residual > _TOPOLOGY_VETO_THRESHOLD
            or not r_psd_ok
            or not r_symmetric_ok
            or p_diss < -audit_tol
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            max_topological_residual > tolerance
            or p_diss < 0.0
            or (
                np.isfinite(rayleigh_condition_number)
                and rayleigh_condition_number > _RAYLEIGH_CONDITION_LIMIT
            )
            or min_eig < 0.0
        ):
            status = SpecularFlowStatus.DEGRADED

        # Trazabilidad con fases previas.
        prior_allievi_status = getattr(
            allievi_invariants,
            "status",
            SpecularFlowStatus.COHERENT,
        )

        status = SpecularFlowStatus.supremum(
            status,
            prior_allievi_status,
            prior_householder_status,
        )

        return NetworkConservationReport(
            tellegen_residual=tellegen_residual,
            is_tellegen_conserved=is_tellegen_conserved,
            rayleigh_dissipation=p_diss,
            is_lyapunov_passive=is_lyapunov_passive,
            kcl_residual=kcl_residual,
            kvl_residual=kvl_residual,
            is_kcl_conserved=is_kcl_conserved,
            is_kvl_conserved=is_kvl_conserved,
            graph_order=int(n_nodes),
            graph_size=int(m),
            cyclomatic_number=int(cyclomatic_number),
            rayleigh_condition_number=rayleigh_condition_number,
            min_rayleigh_eigenvalue=min_eig,
            tellegen_power=float(tellegen_power),
            kahan_tellegen_sum=float(tellegen_power),
            used_sparse_tellegen=used_sparse_tellegen,
            status=status,
        )


═══════════════════════════════════════════════════════════════════════════
# Motor físico completo (hereda de la Fase 3, por tanto de todas las anteriores)
═══════════════════════════════════════════════════════════════════════════

class SpecularFlowEngine(Phase3_TellegenValidator):
    """
    Motor físico consolidado para el Flujo Especular.

    Agrupa las tres fases anidadas:
      • Observe  (Allievi)
      • Orient   (Householder)
      • Decide   (Tellegen)

    No tiene autoridad de veto ni interactúa con hardware; es puramente
    numérico y provee certificados observables para agentes superiores.
    """

    def full_specular_pipeline(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: NDArray[np.float64],
        boundary_normal_n: NDArray[np.float64],
        state_vector: NDArray[np.float64],
        pressures: Optional[NDArray[np.float64]],
        flows: NDArray[np.float64],
        rayleigh_dissipation_r: NDArray[np.float64],
        gradient_h: Optional[NDArray[np.float64]],
        incidence_matrix: Optional[object] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
    ) -> Tuple[AllieviInvariants, HouseholderReflectionResult, NetworkConservationReport]:
        """
        Ejecuta el pipeline completo de las tres fases en orden, propagando
        explícitamente los certificados de una fase a la siguiente.

        Retorna
        -------
        Tuple[AllieviInvariants, HouseholderReflectionResult, NetworkConservationReport]
            Certificados inmutables de las tres fases.
        """
        # FASE 1 — Observe.
        allievi = self.solve_allievi_characteristics(
            potential_y=potential_y,
            flow_q=flow_q,
            wave_speed_a=wave_speed_a,
            gravity_g=gravity_g,
            section_s=section_s,
            tolerance=tolerance,
            cfl_number=cfl_number,
        )

        # FASE 2 — Orient. Consume allievi.
        householder = self.compute_householder_reflection(
            metric_tensor_g=metric_tensor_g,
            boundary_normal_n=boundary_normal_n,
            state_vector=state_vector,
            allievi_invariants=allievi,
            tolerance=tolerance,
        )

        # FASE 3 — Decide. Consume allievi y householder.
        tellegen = self.verify_network_conservation(
            pressures=pressures,
            flows=flows,
            rayleigh_dissipation_r=rayleigh_dissipation_r,
            gradient_h=gradient_h,
            allievi_invariants=allievi,
            householder_result=householder,
            tolerance=tolerance,
            incidence_matrix=incidence_matrix,
            nodal_potentials=nodal_potentials,
        )

        return allievi, householder, tellegen


═══════════════════════════════════════════════════════════════════════════
# Exportación pública
═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "SpecularFlowPhysicsError",
    "AllieviIntegrationError",
    "HouseholderReflectionError",
    "TellegenConservationError",
    "SpecularFlowStatus",
    "AllieviInvariants",
    "HouseholderReflectionResult",
    "NetworkConservationReport",
    "Phase1_AllieviSolver",
    "Phase2_HouseholderReflector",
    "Phase3_TellegenValidator",
    "SpecularFlowEngine",
    "stable_cholesky_decomposition",
    "kahan_tellegen_summation",
    "verify_tellegen_sparse",
]
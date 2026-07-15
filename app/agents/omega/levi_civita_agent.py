# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Levi-Civita Connection Agent (Maestro de Sinfonía Métrica)          ║
║ Ruta   : app/omega/levi_civita_agent.py                                      ║
║ Versión: 8.0.0-Granular-Geodesic-Categorical-Nested-Spectral                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA DIFERENCIAL
------------------------------------------------
Meta-funtor sobre el haz tangente generativo Γ en el Estrato Ω. Impone la
ecuación geodésica y la compatibilidad métrica como axiomas de ejecución.

Fases anidadas (continuación formal, espejo KApex / Floquet / Eikonal)
----------------------------------------------------------------------
.. code-block:: text

    Phase1_ChristoffelEngine
        │  build_christoffel(G)     ──►  ChristoffelData
        ▼
    Phase2_TorsionFreeConnection
        │  verify_axioms(data)      ──►  ConnectionDiagnostics
        ▼
    Phase3_GeodesicOrchestrator  (LeviCivitaConnectionAgent)
        │  enforce_geodesic_flow    ──►  TangentVector (+ report opcional)
        │  geodesic_rhs             ──►  a = −Γ(v,v)   [API Eikonal]
        │  parallel_transport / ♭♯  ──►  transportes categóricos

CAMBIOS ESTRUCTURALES RESPECTO A v7.0.0
----------------------------------------
1. FASES ANIDADAS con DTOs de continuación formal (Phase1 → ChristoffelData
   es precondición tipada de Phase2; ConnectionDiagnostics precondición
   operativa de Phase3).

2. NORMA G EN LA GEODÉSICA: conservación
   \[
     \|v\|_G = \sqrt{v^\top G v}
   \]
   con proyección métrica opcional post-RK4 (renormalización en la
   elipsoide de nivel de \(G\)).

3. API EIKONAL COMPATIBLE:
   - ``enforce_geodesic_flow(v, dt) -> TangentVector``
   - ``geodesic_rhs(v_coords) -> NDArray``
   - ``enforce_geodesic_flow(..., return_report=True) -> (TangentVector, Report)``

4. SIMETRIZACIÓN REAL DE dG: \(dG \leftarrow \tfrac12(dG + dG^{T_{ij}})\).

5. RIEMANN + BIANCHI + RICCI:
   - Parte algebraica \(R^r{}_{smn}\approx \Gamma^r_{mk}\Gamma^k_{ns}
     -\Gamma^r_{nk}\Gamma^k_{ms}\) (exacta si \(\partial\Gamma=0\)).
   - Primera identidad de Bianchi algebraica (torsión nula):
     \(R^r{}_{s[mn]}+R^r{}_{m[ns]}+R^r{}_{n[sm]}\) diagnosticada.
   - Contracción de Ricci \(R_{sn}=R^r{}_{srn}\).

6. PASO ESTABLE RK4 VELOCIDAD-DEPENDIENTE:
   \[
     dt_{\max}(v)\sim
       \frac{c}{\|\Gamma\|_\infty\cdot\|v\|_G+\varepsilon}.
   \]

7. TRANSPORTE PARALELO HEUN (orden 2) opcional; Euler conservado.

8. HOOKS de derivada métrica: estático por defecto; interfaz
   ``MetricDerivativeProvider`` para \(dG\) dinámico (curvatura no nula).

9. RESIDUO DE INVERSA relativo a \(\varepsilon_{\mathrm{mach}}\kappa(G)\).

10. STUBS de ecosistema para tests aislados.

FUNDAMENTO
----------
§1 Levi-Civita: torsión nula + \(\nabla G=0\) ⇒ unicidad de \(\nabla\).
§2 \(\Gamma^r_{mn}=\tfrac12 G^{rk}(\partial_m G_{kn}+\partial_n G_{mk}-\partial_k G_{mn})\)
   (fórmula de Koszul).
§3 \(\Gamma^r_{mn}=\Gamma^r_{nm}\).
§4 Geodésica: \(\dot v^\mu=-\Gamma^\mu_{rs}v^r v^s\), RK4.
§5 \(\|v\|_G\) constante a lo largo de geodésicas (energía cinética).
§6 \(R^\rho{}_{\sigma\mu\nu}\) obstrucción local a holonomía trivial.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Dependencias del ecosistema (stubs para ejecución aislada)
# ---------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Stub: invariante topológico violado."""

    class CategoricalState:  # type: ignore[no-redef]
        def __init__(self, payload: Any = None, label: str = "") -> None:
            self.payload = payload
            self.label = label

    class Morphism:  # type: ignore[no-redef]
        def __init__(self, name: str = "") -> None:
            self.name = name


try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:  # pragma: no cover
    G_PHYSICS = np.eye(4, dtype=np.float64)


try:
    from app.core.immune_system.musical_isomorphism_engine import (
        MusicalIsomorphismEngine,
        MetricSpectralPreconditioner,
        PreconditionedMetric,
        TangentVector,
        CotangentVector,
        _MACHINE_EPSILON,
    )
except ImportError:  # pragma: no cover
    _MACHINE_EPSILON = float(np.finfo(np.float64).eps)

    @dataclass(frozen=True)
    class TangentVector:  # type: ignore[no-redef]
        coordinates: NDArray[np.float64]

        @property
        def dim(self) -> int:
            return int(np.asarray(self.coordinates).reshape(-1).shape[0])

        @property
        def norm(self) -> float:
            v = np.asarray(self.coordinates, dtype=np.float64).reshape(-1)
            return float(np.linalg.norm(v))

    @dataclass(frozen=True)
    class CotangentVector:  # type: ignore[no-redef]
        coordinates: NDArray[np.float64]

        @property
        def dim(self) -> int:
            return int(np.asarray(self.coordinates).reshape(-1).shape[0])

        @property
        def norm(self) -> float:
            w = np.asarray(self.coordinates).reshape(-1)
            return float(np.linalg.norm(w))

    @dataclass(frozen=True)
    class PreconditionedMetric:  # type: ignore[no-redef]
        G: NDArray[np.float64]
        G_inv: NDArray[np.float64]
        matrix_dimension: int
        condition_number_reg: float
        regularization_applied: bool = False
        null_space_dim: int = 0

        def spectral_summary(self) -> Dict[str, Any]:
            return {
                "condition_number_reg": self.condition_number_reg,
                "null_space_dim": self.null_space_dim,
                "regularization_applied": self.regularization_applied,
            }

    class MetricSpectralPreconditioner:  # type: ignore[no-redef]
        def precondition(
            self, G_raw: NDArray[np.float64]
        ) -> PreconditionedMetric:
            G = 0.5 * (
                np.asarray(G_raw, dtype=np.float64)
                + np.asarray(G_raw, dtype=np.float64).T
            )
            n = G.shape[0]
            try:
                L = la.cholesky(G, lower=True)
            except la.LinAlgError:
                jitter = 1e-10 * max(float(np.trace(G)) / max(n, 1), 1.0)
                G = G + jitter * np.eye(n, dtype=np.float64)
                L = la.cholesky(G, lower=True)
            I = np.eye(n, dtype=np.float64)
            Y = la.solve_triangular(L, I, lower=True, check_finite=False)
            G_inv = 0.5 * ((Y.T @ Y) + (Y.T @ Y).T)
            ev = np.linalg.eigvalsh(G)
            kappa = float(ev[-1] / max(ev[0], 1e-300))
            return PreconditionedMetric(
                G=G,
                G_inv=G_inv,
                matrix_dimension=n,
                condition_number_reg=kappa,
                regularization_applied=True,
            )

    class MusicalIsomorphismEngine:  # type: ignore[no-redef]
        def __init__(
            self,
            metric_tensor: NDArray[np.float64],
            preconditioner: Optional[Any] = None,
        ) -> None:
            pc = preconditioner or MetricSpectralPreconditioner()
            self._pm = pc.precondition(metric_tensor)

        def apply_flat_isomorphism(self, v: TangentVector) -> CotangentVector:
            coords = self._pm.G @ np.asarray(v.coordinates, dtype=np.float64)
            return CotangentVector(coordinates=coords)

        def apply_sharp_isomorphism(self, w: CotangentVector) -> TangentVector:
            coords = self._pm.G_inv @ np.asarray(
                w.coordinates, dtype=np.float64
            )
            return TangentVector(coordinates=coords)


logger = logging.getLogger("MIC.Omega.LeviCivitaAgent")

# ---------------------------------------------------------------------------
# Constantes de rigor numérico (espectral / Wilkinson)
# ---------------------------------------------------------------------------
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)
_TORSION_TOLERANCE: float = 1.0e-13
_METRIC_COMPAT_TOLERANCE: float = 1.0e-11
_CHRISTOFFEL_FINITE_TOL: float = 1.0e15
_GEODESIC_NORM_DRIFT_TOL: float = 1.0e-6
_DEFAULT_DT: float = 1.0e-3
_DT_MAX_STABLE_FACTOR: float = 0.5
_DT_MIN: float = 1.0e-12
_DG_SYM_TOL: float = 1.0e-12
_WILKINSON_SAFETY: float = 100.0
_BIANCHI_TOLERANCE: float = 1.0e-10
_INV_RESIDUAL_FACTOR: float = 1.0e3
_GAMMA_INF_FLOOR: float = 1.0e-300
_HEUN_HALF: float = 0.5


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES
# =============================================================================


class TopologicalTorsionError(TopologicalInvariantError):
    r"""\(T^r_{mn}=\Gamma^r_{mn}-\Gamma^r_{nm}\neq 0\)."""


class GeodesicDeviationError(TopologicalInvariantError):
    """Desviación del flujo geodésico / overflow RK4 / drift de norma G."""


class MetricCompatibilityError(TopologicalInvariantError):
    r"""\(\nabla G\neq 0\) fuera de tolerancia."""


class ChristoffelInstabilityError(TopologicalInvariantError):
    r"""Γ no finito o \(\|\Gamma\|_F\) excesiva."""


class LeviCivitaParameterError(TopologicalInvariantError):
    """Parámetro de control fuera de rango."""


class DimensionalMismatchError(TopologicalInvariantError):
    """Dimensión de vectores/tensores incompatible con n."""


class BianchiIdentityError(TopologicalInvariantError):
    r"""Primera identidad de Bianchi algebraica violada fuera de tolerancia."""


# =============================================================================
# SECCIÓN 0b — PROTOCOLO DE DERIVADA MÉTRICA (extensibilidad)
# =============================================================================


@runtime_checkable
class MetricDerivativeProvider(Protocol):
    r"""
    Proveedor de \(\partial G\).

    Contrato
    --------
    ``derivative(n) -> dG`` con shape \((n,n,n)\) y
    \(dG[k,i,j]=\partial_k G_{ij}\). Debe respetar \(dG[k,i,j]=dG[k,j,i]\).
    """

    def derivative(self, n: int) -> NDArray[np.float64]:
        ...


class StaticMetricDerivative:
    r"""\(dG\equiv 0\) (variedad localmente plana en coordenadas adaptadas)."""

    def derivative(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            raise DimensionalMismatchError(
                f"StaticMetricDerivative: n={n} inválido."
            )
        return np.zeros((n, n, n), dtype=np.float64)


class CallableMetricDerivative:
    """Adapta un callable \(n\mapsto dG\) al protocolo."""

    def __init__(
        self, fn: Any
    ) -> None:
        self._fn = fn

    def derivative(self, n: int) -> NDArray[np.float64]:
        dG = np.asarray(self._fn(n), dtype=np.float64)
        return dG


# =============================================================================
# SECCIÓN 1 — DTO INMUTABLES (continuación formal entre fases)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ChristoffelData:
    r"""
    **Salida terminal de la Fase 1** — precondición formal de
    ``Phase2_TorsionFreeConnection.verify_axioms``.

    Convención de índices
    ---------------------
    - ``Gamma[r, m, n] = Γ^r_{mn}``
    - ``dG[k, i, j]   = ∂_k G_{ij}``

    Invariantes de construcción
    ---------------------------
    1. \(G=G^\top\), \(G^{-1}\) residual-controlado.
    2. \(dG[k,i,j]=dG[k,j,i]\) (simetrizado).
    3. \(\Gamma\) finito y \(\|\Gamma\|_F < \tau_{\mathrm{Christoffel}}\).
    """

    Gamma: NDArray[np.float64]
    frobenius_norm: float
    infinity_norm: float
    dG: NDArray[np.float64]
    G: NDArray[np.float64]
    G_inv: NDArray[np.float64]
    dimension: int
    is_static: bool
    condition_number_reg: float
    inverse_residual: float
    inverse_residual_bound: float
    spectral_gap_min: float

    def __post_init__(self) -> None:
        n = self.dimension
        for name, arr in (("Gamma", self.Gamma), ("dG", self.dG)):
            if arr.shape != (n, n, n):
                raise ValueError(
                    f"ChristoffelData.{name} shape={arr.shape} ≠ ({n},{n},{n})."
                )
        if self.G.shape != (n, n) or self.G_inv.shape != (n, n):
            raise ValueError("G / G_inv con shape incorrecta.")
        if self.frobenius_norm < 0.0 or self.infinity_norm < 0.0:
            raise ValueError("normas de Γ deben ser ≥ 0.")
        if self.condition_number_reg < 1.0 - 1e-12:
            raise ValueError(
                f"condition_number_reg={self.condition_number_reg} < 1."
            )


@dataclass(frozen=True, slots=True)
class ConnectionDiagnostics:
    r"""
    **Salida terminal de la Fase 2** — precondición operativa de la
    Fase 3 (orquestación geodésica).

    Codifica la verificación de los axiomas de Levi-Civita y
    diagnósticos de curvatura algebraica (Riemann / Bianchi / Ricci).
    """

    torsion_norm: float
    covd_metric_norm: float
    riemann_norm: float
    ricci_norm: float
    bianchi_norm: float
    condition_number_reg: float
    torsion_passed: bool
    metric_compat_passed: bool
    bianchi_passed: bool
    is_static: bool
    dimension: int

    def all_passed(self) -> bool:
        return (
            self.torsion_passed
            and self.metric_compat_passed
            and self.bianchi_passed
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "torsion_norm": self.torsion_norm,
            "covd_metric_norm": self.covd_metric_norm,
            "riemann_norm": self.riemann_norm,
            "ricci_norm": self.ricci_norm,
            "bianchi_norm": self.bianchi_norm,
            "condition_number_reg": self.condition_number_reg,
            "torsion_passed": self.torsion_passed,
            "metric_compat_passed": self.metric_compat_passed,
            "bianchi_passed": self.bianchi_passed,
            "all_passed": self.all_passed(),
            "is_static": self.is_static,
            "dimension": self.dimension,
        }


@dataclass(frozen=True, slots=True)
class GeodesicStepReport:
    r"""Informe de un paso RK4 (norma **G**, no euclídea)."""

    v_initial_norm_G: float
    v_final_norm_G: float
    norm_drift_G: float
    acceleration_norm_euclid: float
    dt: float
    dt_max_stable: float
    is_stable: bool
    renormalized: bool = False


# =============================================================================
# SECCIÓN 2 — ORQUESTADOR CON FASES ANIDADAS
# =============================================================================


class LeviCivitaConnectionAgent(Morphism):
    r"""
    Agente Levi-Civita v8 — compone Phase1 → Phase2 → Phase3.

    Semántica categórica
    --------------------
    Morfismo \(T\mathcal{M}\to T\mathcal{M}\) que preserva la estructura
    afín de Levi-Civita: torsión nula, \(\nabla G=0\), flujo geodésico.

    API pública alineada con EikonalAgent
    -------------------------------------
    * ``enforce_geodesic_flow(v, dt) -> TangentVector``
    * ``geodesic_rhs(v_coords) -> NDArray``
    * ``parallel_transport``, isomorfismos ♭/♯
    """

    DEFAULT_DT: float = _DEFAULT_DT

    def __init__(
        self,
        metric_tensor: NDArray[np.float64] = G_PHYSICS,
        preconditioner: Optional[MetricSpectralPreconditioner] = None,
        torsion_tolerance: float = _TORSION_TOLERANCE,
        metric_compat_tolerance: float = _METRIC_COMPAT_TOLERANCE,
        bianchi_tolerance: float = _BIANCHI_TOLERANCE,
        metric_derivative: Optional[MetricDerivativeProvider] = None,
        enforce_norm_conservation: bool = True,
        parallel_transport_order: int = 2,
    ) -> None:
        if torsion_tolerance <= 0.0:
            raise LeviCivitaParameterError(
                f"torsion_tolerance debe ser > 0; se obtuvo {torsion_tolerance}."
            )
        if metric_compat_tolerance <= 0.0:
            raise LeviCivitaParameterError(
                f"metric_compat_tolerance debe ser > 0; "
                f"se obtuvo {metric_compat_tolerance}."
            )
        if bianchi_tolerance <= 0.0:
            raise LeviCivitaParameterError(
                f"bianchi_tolerance debe ser > 0; se obtuvo {bianchi_tolerance}."
            )
        if parallel_transport_order not in (1, 2):
            raise LeviCivitaParameterError(
                f"parallel_transport_order ∈ {{1,2}}; "
                f"se obtuvo {parallel_transport_order}."
            )

        G_raw = np.asarray(metric_tensor, dtype=np.float64)
        self._enforce_norm_conservation = bool(enforce_norm_conservation)
        self._pt_order = int(parallel_transport_order)
        self._bianchi_tolerance = float(bianchi_tolerance)

        # ------------------------------------------------------------------
        # Fase 1 — construcción de Γ (salida: ChristoffelData)
        # ------------------------------------------------------------------
        self.phase1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine(
            preconditioner=preconditioner,
            metric_derivative=metric_derivative,
        )
        self._christoffel_data: ChristoffelData = (
            self.phase1.build_christoffel(G_raw)
        )

        # ------------------------------------------------------------------
        # Fase 2 — axiomas LC (entrada: ChristoffelData → ConnectionDiagnostics)
        # ------------------------------------------------------------------
        self.phase2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection(
            torsion_tolerance=torsion_tolerance,
            metric_compat_tolerance=metric_compat_tolerance,
            bianchi_tolerance=bianchi_tolerance,
        )
        self._connection_diagnostics: ConnectionDiagnostics = (
            self.phase2.verify_axioms(self._christoffel_data)
        )

        # Estado métrico cacheado (post-Fase 1/2)
        self._G = self._christoffel_data.G
        self._G_inv = self._christoffel_data.G_inv
        self._n = self._christoffel_data.dimension
        self._Gamma = self._christoffel_data.Gamma
        self._gamma_inf = max(
            self._christoffel_data.infinity_norm, _GAMMA_INF_FLOOR
        )

        # ------------------------------------------------------------------
        # Fase 3 — orquestación geodésica + musical
        # ------------------------------------------------------------------
        self._musical_engine = MusicalIsomorphismEngine(
            metric_tensor=self._G,
            preconditioner=preconditioner,
        )
        # dt_max de referencia a ‖v‖_G = 1
        self._dt_max_stable_unit = self._compute_max_stable_dt_for_speed(1.0)

        try:
            super().__init__(name="LeviCivitaConnectionAgent")
        except TypeError:
            super().__init__()

        logger.info(
            "[LeviCivitaConnectionAgent] v8: n=%d, ‖Γ‖_F=%.3e, ‖Γ‖_∞=%.3e, "
            "static=%s, torsion=%.2e, ∇G=%.2e, Bianchi=%.2e, "
            "dt_max(|v|_G=1)=%.3e, renorm=%s, pt_order=%d.",
            self._n,
            self._christoffel_data.frobenius_norm,
            self._christoffel_data.infinity_norm,
            self._christoffel_data.is_static,
            self._connection_diagnostics.torsion_norm,
            self._connection_diagnostics.covd_metric_norm,
            self._connection_diagnostics.bianchi_norm,
            self._dt_max_stable_unit,
            self._enforce_norm_conservation,
            self._pt_order,
        )

    # =========================================================================
    # FASE 1 — CHRISTOFFEL ENGINE
    # =========================================================================

    class Phase1_ChristoffelEngine:
        r"""
        **Fase 1** — Par métrico estable + símbolos de Christoffel (Koszul).

        \[
          \Gamma^r_{mn}
            = \tfrac12 G^{rk}
              \bigl(\partial_m G_{kn}+\partial_n G_{mk}-\partial_k G_{mn}\bigr).
        \]

        Continuación formal
        -------------------
        El método terminal ``build_christoffel`` produce
        ``ChristoffelData``, que es el **objeto inicial** de la
        Fase 2 (``Phase2_TorsionFreeConnection.verify_axioms``).
        """

        def __init__(
            self,
            preconditioner: Optional[MetricSpectralPreconditioner] = None,
            metric_derivative: Optional[MetricDerivativeProvider] = None,
        ) -> None:
            self._preconditioner = (
                preconditioner
                if preconditioner is not None
                else MetricSpectralPreconditioner()
            )
            self._dG_provider: MetricDerivativeProvider = (
                metric_derivative
                if metric_derivative is not None
                else StaticMetricDerivative()
            )

        @staticmethod
        def _validate_metric_input(G: NDArray[np.float64]) -> None:
            if not isinstance(G, np.ndarray):
                raise TypeError(
                    f"Se requiere NDArray; se recibió {type(G).__name__}."
                )
            if G.ndim != 2 or G.shape[0] != G.shape[1]:
                raise DimensionalMismatchError(
                    f"G debe ser cuadrada 2D; se recibió {G.shape}."
                )
            if G.shape[0] == 0:
                raise DimensionalMismatchError("dim(G) no puede ser 0.")
            if not np.all(np.isfinite(G)):
                n_bad = int(np.sum(~np.isfinite(G)))
                raise ValueError(
                    f"G contiene {n_bad} valores no finitos (NaN/Inf)."
                )
            # Simetría débil (pre-preconditioner)
            asym = float(np.max(np.abs(G - G.T)))
            if asym > 1.0e-8 * max(float(np.max(np.abs(G))), 1.0):
                logger.warning(
                    "[Fase1] G cruda asimétrica (max=%.3e); el precondicionador "
                    "simetrizará.",
                    asym,
                )

        def _compute_metric_derivative(
            self, n: int
        ) -> NDArray[np.float64]:
            r"""
            \(dG[k,i,j]=\partial_k G_{ij}\) vía ``MetricDerivativeProvider``.

            Por defecto estático (\(dG=0\)). Inyectar un provider no trivial
            para curvatura dinámica / coordenadas no geodésicas.
            """
            return np.asarray(
                self._dG_provider.derivative(n), dtype=np.float64
            )

        @staticmethod
        def _symmetrize_dG(
            dG: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""Fuerza \(dG[k,i,j]=dG[k,j,i]\) (G simétrica ⇒ ∂G simétrica en i,j)."""
            return 0.5 * (dG + dG.transpose(0, 2, 1))

        def _validate_derivative_tensor(
            self, dG: NDArray[np.float64], n: int
        ) -> NDArray[np.float64]:
            if dG.shape != (n, n, n):
                raise ValueError(
                    f"dG shape={dG.shape} ≠ ({n},{n},{n})."
                )
            if not np.all(np.isfinite(dG)):
                raise ValueError("dG contiene valores no finitos.")
            asym = float(np.max(np.abs(dG - dG.transpose(0, 2, 1))))
            if asym > _DG_SYM_TOL:
                logger.warning(
                    "[Fase1] dG asimétrico en i,j (max=%.3e); simetrizando.",
                    asym,
                )
            return self._symmetrize_dG(dG)

        def _compute_christoffel_terms(
            self,
            G_inv: NDArray[np.float64],
            dG: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            r"""
            Términos de Koszul:

            - \(T1^r_{mn}=G^{rk}\partial_m G_{kn}\)
            - \(T2^r_{mn}=G^{rk}\partial_n G_{mk}\)
            - \(T3^r_{mn}=G^{rk}\partial_k G_{mn}\)
            """
            T1 = np.einsum("rk,mkn->rmn", G_inv, dG, optimize=True)
            T2 = np.einsum("rk,nmk->rmn", G_inv, dG, optimize=True)
            T3 = np.einsum("rk,kmn->rmn", G_inv, dG, optimize=True)
            for name, T in (("T1", T1), ("T2", T2), ("T3", T3)):
                if not np.all(np.isfinite(T)):
                    raise ChristoffelInstabilityError(
                        f"Término {name} de Christoffel no finito."
                    )
            return T1, T2, T3

        @staticmethod
        def _inverse_residual(
            G: NDArray[np.float64], G_inv: NDArray[np.float64]
        ) -> float:
            r"""\(\max\bigl(\|GG^{-1}-I\|_F,\|G^{-1}G-I\|_F\bigr)/n\)."""
            n = G.shape[0]
            I = np.eye(n, dtype=np.float64)
            r_plus = float(la.norm(G @ G_inv - I, "fro")) / n
            r_minus = float(la.norm(G_inv @ G - I, "fro")) / n
            return max(r_plus, r_minus)

        @staticmethod
        def _spectral_floor(
            G: NDArray[np.float64],
        ) -> float:
            """Menor autovalor de \(G\) (post-precondicionamiento)."""
            ev = np.linalg.eigvalsh(0.5 * (G + G.T))
            return float(ev[0])

        def build_christoffel(
            self, raw_metric: NDArray[np.float64]
        ) -> ChristoffelData:
            r"""
            **Método terminal de la Fase 1.**

            Ensambla \(\Gamma\) por la fórmula de Koszul a partir del
            par \((G,G^{-1})\) espectralmente precondicionado y de
            \(\partial G\).

            Retorna
            -------
            ChristoffelData
                **Precondición formal** de
                ``Phase2_TorsionFreeConnection.verify_axioms``.
                La continuación de la tubería es:

                .. code-block:: text

                    data = Phase1.build_christoffel(G)
                    diag = Phase2.verify_axioms(data)   # ← Fase 2
            """
            self._validate_metric_input(raw_metric)
            pm: PreconditionedMetric = self._preconditioner.precondition(
                raw_metric
            )
            G = np.asarray(pm.G, dtype=np.float64)
            G_inv = np.asarray(pm.G_inv, dtype=np.float64)
            n = int(pm.matrix_dimension)

            # Cota de residual relativo a Wilkinson: ~ ε_mach · κ(G)
            inv_bound = (
                _INV_RESIDUAL_FACTOR
                * _MACHINE_EPS
                * max(float(pm.condition_number_reg), 1.0)
            )
            inv_res = self._inverse_residual(G, G_inv)
            if inv_res > inv_bound:
                logger.warning(
                    "[Fase1] Residuo de inversa %.3e > cota Wilkinson %.3e "
                    "(κ_reg=%.3e).",
                    inv_res,
                    inv_bound,
                    pm.condition_number_reg,
                )

            dG = self._validate_derivative_tensor(
                self._compute_metric_derivative(n), n
            )
            T1, T2, T3 = self._compute_christoffel_terms(G_inv, dG)
            Gamma = 0.5 * (T1 + T2 - T3)

            if not np.all(np.isfinite(Gamma)):
                raise ChristoffelInstabilityError(
                    "Γ contiene valores no finitos tras el ensamblado."
                )
            frob = float(la.norm(Gamma, "fro"))
            # ‖Γ‖_∞ = max_{r,m,n} |Γ^r_{mn}|
            inf_norm = float(np.max(np.abs(Gamma))) if Gamma.size else 0.0
            if frob > _CHRISTOFFEL_FINITE_TOL:
                raise ChristoffelInstabilityError(
                    f"‖Γ‖_F={frob:.3e} > umbral {_CHRISTOFFEL_FINITE_TOL:.3e}."
                )

            is_static = bool(np.all(dG == 0.0))
            gap = self._spectral_floor(G)

            data = ChristoffelData(
                Gamma=Gamma,
                frobenius_norm=frob,
                infinity_norm=inf_norm,
                dG=dG,
                G=G,
                G_inv=G_inv,
                dimension=n,
                is_static=is_static,
                condition_number_reg=float(pm.condition_number_reg),
                inverse_residual=inv_res,
                inverse_residual_bound=inv_bound,
                spectral_gap_min=gap,
            )
            logger.info(
                "[Fase1] Christoffel: n=%d, ‖Γ‖_F=%.3e, ‖Γ‖_∞=%.3e, "
                "static=%s, κ_reg=%.3e, inv_res=%.3e (bound=%.3e), "
                "λ_min=%.3e.",
                n,
                frob,
                inf_norm,
                is_static,
                pm.condition_number_reg,
                inv_res,
                inv_bound,
                gap,
            )
            # Continuación formal → Phase2.verify_axioms(data)
            return data

    # =========================================================================
    # FASE 2 — TORSIÓN NULA + COMPATIBILIDAD MÉTRICA
    #         (inicio = continuación de Phase1.build_christoffel)
    # =========================================================================

    class Phase2_TorsionFreeConnection:
        r"""
        **Fase 2** — Axiomas de Levi-Civita y curvatura algebraica.

        Continuación formal de la Fase 1
        --------------------------------
        El objeto de entrada es exactamente
        ``ChristoffelData`` producido por
        ``Phase1_ChristoffelEngine.build_christoffel``.

        Axiomas verificados
        -------------------
        1. Torsión nula:
           \(T^r_{mn}=\Gamma^r_{mn}-\Gamma^r_{nm}=0\).
        2. Compatibilidad métrica:
           \[
             (\nabla_\gamma G)_{\mu\nu}
               =\partial_\gamma G_{\mu\nu}
                -\Gamma^k_{\gamma\mu}G_{k\nu}
                -\Gamma^k_{\gamma\nu}G_{\mu k}=0.
           \]
        3. Diagnóstico de Riemann (parte algebraica) + Ricci + Bianchi-I.

        Salida
        ------
        ``ConnectionDiagnostics`` — precondición operativa de la Fase 3.
        """

        def __init__(
            self,
            torsion_tolerance: float = _TORSION_TOLERANCE,
            metric_compat_tolerance: float = _METRIC_COMPAT_TOLERANCE,
            bianchi_tolerance: float = _BIANCHI_TOLERANCE,
        ) -> None:
            self.TORSION_TOLERANCE = float(torsion_tolerance)
            self.METRIC_COMPAT_TOLERANCE = float(metric_compat_tolerance)
            self.BIANCHI_TOLERANCE = float(bianchi_tolerance)

        @staticmethod
        def _compute_torsion_tensor(
            Gamma: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""\(T^r_{mn}=\Gamma^r_{mn}-\Gamma^r_{nm}\)."""
            return Gamma - Gamma.transpose(0, 2, 1)

        def _verify_zero_torsion(
            self, torsion: NDArray[np.float64], n: int
        ) -> float:
            t_norm = float(la.norm(torsion, "fro"))
            if t_norm >= self.TORSION_TOLERANCE:
                worst = np.unravel_index(
                    int(np.argmax(np.abs(torsion))), torsion.shape
                )
                raise TopologicalTorsionError(
                    f"Torsión: ‖T‖_F={t_norm:.3e} ≥ "
                    f"tol={self.TORSION_TOLERANCE:.3e}. "
                    f"Máx en T[{worst}]={torsion[worst]:.3e}. n={n}."
                )
            return t_norm

        @staticmethod
        def _compute_covd_metric(
            Gamma: NDArray[np.float64],
            dG: NDArray[np.float64],
            G: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            \((\nabla G)_{gmn}
              =\partial_g G_{mn}
               -\Gamma^k_{gm}G_{kn}
               -\Gamma^k_{gn}G_{mk}\).
            """
            term1 = np.einsum("kgm,kn->gmn", Gamma, G, optimize=True)
            term2 = np.einsum("kgn,mk->gmn", Gamma, G, optimize=True)
            return dG - term1 - term2

        def _verify_metric_compatibility(
            self, covd: NDArray[np.float64]
        ) -> float:
            c_norm = float(la.norm(covd, "fro"))
            if c_norm >= self.METRIC_COMPAT_TOLERANCE:
                worst = np.unravel_index(
                    int(np.argmax(np.abs(covd))), covd.shape
                )
                raise MetricCompatibilityError(
                    f"∇G: ‖∇G‖_F={c_norm:.3e} ≥ "
                    f"tol={self.METRIC_COMPAT_TOLERANCE:.3e}. "
                    f"Máx en ∇G[{worst}]={covd[worst]:.3e}."
                )
            return c_norm

        @staticmethod
        def _compute_riemann_quadratic(
            Gamma: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Parte algebraica del tensor de Riemann:

            \[
              R^r{}_{smn}
                \approx
                  \Gamma^r_{mk}\Gamma^k_{ns}
                - \Gamma^r_{nk}\Gamma^k_{ms}.
            \]

            Exacta si \(\partial\Gamma=0\) (p.ej. modo estático \(\Gamma=0\),
            o conexión de coeficientes constantes). Con \(dG\neq 0\) faltan
            los términos \(\partial_m\Gamma^r_{ns}-\partial_n\Gamma^r_{ms}\),
            que requieren \(\partial^2 G\).
            """
            term_A = np.einsum(
                "rmk,kns->rsmn", Gamma, Gamma, optimize=True
            )
            term_B = np.einsum(
                "rnk,kms->rsmn", Gamma, Gamma, optimize=True
            )
            return term_A - term_B

        @staticmethod
        def _compute_ricci(
            R: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""Contracción de Ricci \(R_{sn}=R^r{}_{srn}\)."""
            return np.einsum("rsrn->sn", R, optimize=True)

        @staticmethod
        def _compute_bianchi_first(
            R: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Primera identidad de Bianchi algebraica (conexión sin torsión):

            \[
              \mathfrak{S}_{mn\ell}\, R^r{}_{s mn}
                = R^r{}_{smn}+R^r{}_{sn\ell}+R^r{}_{s\ell m}
                \quad(\ell\text{ cíclico sobre }m,n;\ \text{aquí }n\leftrightarrow\ell).
            \]

            Implementación sobre los tres últimos índices de
            \(R^r{}_{s m n}\):

            \[
              B^r{}_{smn}
                = R^r{}_{smn}+R^r{}_{snm^{\circ}}+R^r{}_{s(mn)^{\circ\circ}}
                = R[r,s,m,n]+R[r,s,n,m\text{ ciclado}]+R[r,s,\ldots].
            \]

            Usamos la antisimetrización cíclica estándar:

            \[
              B_{rsmn}
                = R^r{}_{smn}+R^r{}_{mns}+R^r{}_{nsm}.
            \]
            """
            # R: (r, s, m, n)
            # Ciclo m→n→s→m sobre los tres últimos índices del bloque
            # (s,m,n) manteniendo r:
            #   R^r_{s m n} + R^r_{m n s} + R^r_{n s m}
            term1 = R
            term2 = R.transpose(0, 2, 3, 1)  # (r, m, n, s)
            term3 = R.transpose(0, 3, 1, 2)  # (r, n, s, m)
            # Alinear a shape (r, s, m, n):
            # term2: (r,m,n,s) → para sumar como R^r_{m n s} en slot (r,s,m,n)
            # es más limpio reindexar explícitamente vía einsum
            B = (
                R
                + np.einsum("rmns->rsmn", R, optimize=True)  # R^r_{m n s}
                + np.einsum("rnsm->rsmn", R, optimize=True)  # R^r_{n s m}
            )
            # Nota: si R ya es antisimétrica en (m,n) y cumple Bianchi, B≈0.
            # La fórmula cíclica clásica es sobre los tres índices inferiores
            # del (0,3)-tensor con el primero de los inferiores fijo a s:
            # R^r_{s[mn]} cíclico en s,m,n.
            # Equivalentemente:
            B = (
                R
                + R.transpose(0, 2, 3, 1)
                + R.transpose(0, 3, 1, 2)
            )
            # R.transpose(0,2,3,1): (r,s,m,n) → (r,m,n,s) = R^r_{m n s}
            # R.transpose(0,3,1,2): (r,s,m,n) → (r,n,s,m) = R^r_{n s m}
            # Para sumar en el mismo layout (r,s,m,n) necesitamos reordenar:
            t2 = R.transpose(0, 2, 3, 1)  # (r, m, n, s)
            t3 = R.transpose(0, 3, 1, 2)  # (r, n, s, m)
            # Mapear t2 (r,m,n,s) → posiciones (r,s,m,n): s←s, m←m, n←n
            #   índices actuales: 0=r, 1=m, 2=n, 3=s  → queremos (r,s,m,n)
            #   = (0, 3, 1, 2)
            t2_aligned = t2.transpose(0, 3, 1, 2)
            # t3 (r,n,s,m): 0=r,1=n,2=s,3=m → (r,s,m,n)=(0,2,3,1)
            t3_aligned = t3.transpose(0, 2, 3, 1)
            return R + t2_aligned + t3_aligned

        def _verify_bianchi(
            self, B: NDArray[np.float64], soft: bool = True
        ) -> float:
            b_norm = float(la.norm(B, "fro"))
            if b_norm >= self.BIANCHI_TOLERANCE:
                msg = (
                    f"Bianchi-I: ‖B‖_F={b_norm:.3e} ≥ "
                    f"tol={self.BIANCHI_TOLERANCE:.3e}."
                )
                if soft:
                    # En modo algebraico (sin ∂Γ) la identidad es exacta
                    # sólo si Γ=0 o si la parte cuadrática ya la satisface;
                    # no abortamos la tubería, diagnosticamos.
                    logger.warning("[Fase2] %s (soft).", msg)
                else:
                    raise BianchiIdentityError(msg)
            return b_norm

        def verify_axioms(
            self, data: ChristoffelData
        ) -> ConnectionDiagnostics:
            r"""
            **Método terminal de la Fase 2** — continuación formal de
            ``Phase1.build_christoffel``.

            Parámetros
            ----------
            data : ChristoffelData
                Salida de ``Phase1_ChristoffelEngine.build_christoffel``.

            Retorna
            -------
            ConnectionDiagnostics
                **Precondición operativa** de la Fase 3
                (``LeviCivitaConnectionAgent``: flujo geodésico,
                transporte paralelo, isomorfismos musicales).
            """
            Gamma = data.Gamma
            dG = data.dG
            G = data.G
            n = data.dimension

            # --- Axioma 1: torsión nula -----------------------------------
            torsion = self._compute_torsion_tensor(Gamma)
            t_norm = self._verify_zero_torsion(torsion, n)

            # --- Axioma 2: ∇G = 0 -----------------------------------------
            covd = self._compute_covd_metric(Gamma, dG, G)
            c_norm = self._verify_metric_compatibility(covd)

            # --- Curvatura algebraica -------------------------------------
            R = self._compute_riemann_quadratic(Gamma)
            r_norm = float(la.norm(R, "fro"))
            Ric = self._compute_ricci(R)
            ric_norm = float(la.norm(Ric, "fro"))
            B = self._compute_bianchi_first(R)
            b_norm = self._verify_bianchi(B, soft=True)

            diag = ConnectionDiagnostics(
                torsion_norm=t_norm,
                covd_metric_norm=c_norm,
                riemann_norm=r_norm,
                ricci_norm=ric_norm,
                bianchi_norm=b_norm,
                condition_number_reg=data.condition_number_reg,
                torsion_passed=t_norm < self.TORSION_TOLERANCE,
                metric_compat_passed=c_norm < self.METRIC_COMPAT_TOLERANCE,
                bianchi_passed=b_norm < self.BIANCHI_TOLERANCE,
                is_static=data.is_static,
                dimension=n,
            )
            logger.info("[Fase2] Axiomas: %s", diag.summary())
            # Continuación formal → Phase3 (métodos del orquestador)
            return diag

    # =========================================================================
    # FASE 3 — GEODÉSICA RK4 + TRANSPORTES
    #         (continuación de Phase2.verify_axioms / ConnectionDiagnostics)
    # =========================================================================

    def _g_norm(self, v: NDArray[np.float64]) -> float:
        r"""\(\|v\|_G=\sqrt{\max(v^\top G v,\,0)}\)."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        q = float(v @ self._G @ v)
        return math.sqrt(max(q, 0.0))

    def _compute_max_stable_dt_for_speed(self, speed_G: float) -> float:
        r"""
        \[
          dt_{\max}(v)
            = \frac{c}{\|\Gamma\|_\infty\cdot\|v\|_G+\varepsilon}.
        \]
        """
        denom = self._gamma_inf * max(float(speed_G), 0.0) + _MACHINE_EPS
        return _DT_MAX_STABLE_FACTOR / denom

    def _validate_velocity_vector(
        self, velocity: TangentVector, caller: str = ""
    ) -> NDArray[np.float64]:
        if not isinstance(velocity, TangentVector):
            raise TypeError(
                f"{caller}: se esperaba TangentVector, "
                f"recibido {type(velocity).__name__}."
            )
        coords = np.asarray(velocity.coordinates, dtype=np.float64).reshape(
            -1
        )
        if coords.shape != (self._n,):
            raise DimensionalMismatchError(
                f"{caller}: dim(v)={coords.shape[0]} ≠ n={self._n}."
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"{caller}: coordenadas no finitas.")
        return coords

    def _validate_integration_step(
        self,
        dt: float,
        speed_G: float,
        caller: str = "",
    ) -> Tuple[float, float]:
        if not isinstance(dt, (int, float)):
            raise TypeError(f"{caller}: dt debe ser float.")
        dt_f = float(dt)
        if dt_f <= _DT_MIN:
            raise LeviCivitaParameterError(
                f"{caller}: dt={dt_f:.3e} ≤ _DT_MIN={_DT_MIN:.3e}."
            )
        dt_max = self._compute_max_stable_dt_for_speed(speed_G)
        if dt_f > dt_max:
            logger.warning(
                "%s: dt=%.3e > dt_max_stable(‖v‖_G=%.3e)=%.3e "
                "(‖Γ‖_∞=%.3e).",
                caller,
                dt_f,
                speed_G,
                dt_max,
                self._gamma_inf,
            )
        return dt_f, dt_max

    # ---- RHS geodésico (API Eikonal) ---------------------------------------

    def geodesic_rhs(
        self, v_coords: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Lado derecho de la geodésica (API consumida por EikonalAgent):

        \[
          a^\mu = -\Gamma^\mu_{rs}\, v^r v^s.
        \]
        """
        v = np.asarray(v_coords, dtype=np.float64).reshape(-1)
        if v.shape != (self._n,):
            raise DimensionalMismatchError(
                f"geodesic_rhs: dim={v.shape[0]} ≠ n={self._n}."
            )
        return self._geodesic_acceleration(v)

    def _geodesic_acceleration(
        self, v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        a = -np.einsum("mrs,r,s->m", self._Gamma, v, v, optimize=True)
        if not np.all(np.isfinite(a)):
            raise GeodesicDeviationError(
                f"Aceleración geodésica no finita. "
                f"‖v‖_G={self._g_norm(v):.3e}, "
                f"‖Γ‖_F={self._christoffel_data.frobenius_norm:.3e}."
            )
        return a

    def _rk4_step(
        self, v: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        r"""Integrador RK4 clásico sobre \(\dot v = -\Gamma(v,v)\)."""
        half = 0.5 * dt
        k1 = self._geodesic_acceleration(v)
        k2 = self._geodesic_acceleration(v + half * k1)
        k3 = self._geodesic_acceleration(v + half * k2)
        k4 = self._geodesic_acceleration(v + dt * k3)
        v_new = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(v_new)):
            raise GeodesicDeviationError(
                f"RK4 no finito: dt={dt:.3e}, ‖v‖_G={self._g_norm(v):.3e}."
            )
        return v_new

    def _renormalize_G(
        self,
        v: NDArray[np.float64],
        target_norm_G: float,
    ) -> NDArray[np.float64]:
        r"""
        Proyección radial sobre la elipsoide \(\|v\|_G=\mathrm{const}\):

        \[
          v \leftarrow v\cdot
            \frac{\|v_{\mathrm{target}}\|_G}{\|v\|_G+\varepsilon}.
        \]
        """
        n_cur = self._g_norm(v)
        if n_cur <= _MACHINE_EPS:
            return v
        scale = target_norm_G / n_cur
        return v * scale

    def _build_step_report(
        self,
        v_i: NDArray[np.float64],
        v_f: NDArray[np.float64],
        dt: float,
        dt_max: float,
        renormalized: bool,
    ) -> GeodesicStepReport:
        n_i = self._g_norm(v_i)
        n_f = self._g_norm(v_f)
        drift = abs(n_f - n_i) / max(n_i, _MACHINE_EPS)
        acc = self._geodesic_acceleration(v_i)
        if drift > _GEODESIC_NORM_DRIFT_TOL and not renormalized:
            logger.warning(
                "Deriva de norma G: |‖v_f‖_G−‖v_i‖_G|/‖v_i‖_G=%.3e > "
                "tol=%.3e (dt=%.3e, ‖Γ‖_F=%.3e).",
                drift,
                _GEODESIC_NORM_DRIFT_TOL,
                dt,
                self._christoffel_data.frobenius_norm,
            )
        return GeodesicStepReport(
            v_initial_norm_G=n_i,
            v_final_norm_G=n_f,
            norm_drift_G=drift,
            acceleration_norm_euclid=float(la.norm(acc)),
            dt=dt,
            dt_max_stable=dt_max,
            is_stable=(dt <= dt_max),
            renormalized=renormalized,
        )

    # ---- enforce_geodesic_flow (compatible Eikonal) ------------------------

    def enforce_geodesic_flow(
        self,
        velocity: TangentVector,
        dt: Optional[float] = None,
        *,
        return_report: bool = False,
        renormalize: Optional[bool] = None,
    ) -> Union[TangentVector, Tuple[TangentVector, GeodesicStepReport]]:
        r"""
        Propaga \(v\) un paso \(dt\) sobre la geodésica (RK4).

        Por defecto retorna **sólo** ``TangentVector`` (contrato Eikonal).
        Con ``return_report=True`` retorna
        ``(TangentVector, GeodesicStepReport)``.

        Si ``renormalize`` (o el flag del agente) está activo, se proyecta
        \(v_f\) para restaurar \(\|v\|_G=\|v_i\|_G\) (conservación numérica
        de la energía cinética).
        """
        v_i = self._validate_velocity_vector(
            velocity, caller="enforce_geodesic_flow"
        )
        speed = self._g_norm(v_i)
        _dt, dt_max = self._validate_integration_step(
            float(dt) if dt is not None else self.DEFAULT_DT,
            speed_G=speed,
            caller="enforce_geodesic_flow",
        )
        v_f = self._rk4_step(v_i, _dt)

        do_renorm = (
            self._enforce_norm_conservation
            if renormalize is None
            else bool(renormalize)
        )
        renormed = False
        if do_renorm and speed > _MACHINE_EPS:
            v_f = self._renormalize_G(v_f, speed)
            renormed = True

        out = TangentVector(coordinates=v_f)
        if return_report:
            return out, self._build_step_report(
                v_i, v_f, _dt, dt_max, renormed
            )
        return out

    # ---- transporte paralelo -----------------------------------------------

    def _gamma_action(
        self,
        ydot: NDArray[np.float64],
        V: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""\(\bigl(\Gamma(\dot\gamma,V)\bigr)^m=\Gamma^m_{rs}\dot\gamma^r V^s\)."""
        return np.einsum("mrs,r,s->m", self._Gamma, ydot, V, optimize=True)

    def parallel_transport(
        self,
        vector: TangentVector,
        tangent_to_curve: TangentVector,
        dt: Optional[float] = None,
    ) -> TangentVector:
        r"""
        Transporte paralelo de \(V\) a lo largo de \(\dot\gamma\):

        \[
          \nabla_{\dot\gamma} V = 0
            \;\Longleftrightarrow\;
          \dot V^m = -\Gamma^m_{rs}\,\dot\gamma^r V^s.
        \]

        - Orden 1: Euler explícito.
        - Orden 2: Heun (predictor-corrector) — default.
        """
        V = self._validate_velocity_vector(vector, "parallel_transport[V]")
        ydot = self._validate_velocity_vector(
            tangent_to_curve, "parallel_transport[ẏ]"
        )
        speed = self._g_norm(ydot)
        _dt, _ = self._validate_integration_step(
            float(dt) if dt is not None else self.DEFAULT_DT,
            speed_G=speed,
            caller="parallel_transport",
        )

        if self._pt_order == 1:
            correction = self._gamma_action(ydot, V)
            V_t = V - _dt * correction
        else:
            # Heun: predictor Euler + corrector promediado
            k1 = self._gamma_action(ydot, V)
            V_pred = V - _dt * k1
            k2 = self._gamma_action(ydot, V_pred)
            V_t = V - _dt * _HEUN_HALF * (k1 + k2)

        if not np.all(np.isfinite(V_t)):
            raise GeodesicDeviationError(
                f"Transporte paralelo no finito (dt={_dt:.3e}, "
                f"order={self._pt_order})."
            )
        return TangentVector(coordinates=V_t)

    # ---- isomorfismos musicales ♭ / ♯ --------------------------------------

    def transport_to_finance_oracle(
        self,
        logistics_flow: TangentVector,
        dt: Optional[float] = None,
        apply_geodesic_correction: bool = True,
    ) -> Tuple[CotangentVector, GeodesicStepReport]:
        r"""
        \(T\mathcal{M}\xrightarrow{\gamma}T\mathcal{M}
          \xrightarrow{\flat}T^*\mathcal{M}\) (logística → finanzas).
        """
        self._validate_velocity_vector(
            logistics_flow, "transport_to_finance_oracle"
        )
        if apply_geodesic_correction:
            result = self.enforce_geodesic_flow(
                logistics_flow, dt=dt, return_report=True
            )
            assert isinstance(result, tuple)
            v_c, report = result
        else:
            v_c = logistics_flow
            v_arr = np.asarray(
                logistics_flow.coordinates, dtype=np.float64
            ).reshape(-1)
            speed = self._g_norm(v_arr)
            report = GeodesicStepReport(
                v_initial_norm_G=speed,
                v_final_norm_G=speed,
                norm_drift_G=0.0,
                acceleration_norm_euclid=float(
                    la.norm(self._geodesic_acceleration(v_arr))
                ),
                dt=0.0,
                dt_max_stable=self._compute_max_stable_dt_for_speed(speed),
                is_stable=True,
                renormalized=False,
            )
        omega = self._musical_engine.apply_flat_isomorphism(v_c)
        return omega, report

    def transport_to_logistics_manifold(
        self,
        financial_force: CotangentVector,
        apply_post_geodesic: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[TangentVector, Optional[GeodesicStepReport]]:
        r"""
        \(T^*\mathcal{M}\xrightarrow{\sharp}T\mathcal{M}
          \xrightarrow{\gamma?}T\mathcal{M}\) (finanzas → logística).
        """
        if not isinstance(financial_force, CotangentVector):
            raise TypeError(
                f"Se esperaba CotangentVector; se recibió "
                f"{type(financial_force).__name__}."
            )
        v = self._musical_engine.apply_sharp_isomorphism(financial_force)
        if apply_post_geodesic:
            result = self.enforce_geodesic_flow(
                v, dt=dt, return_report=True
            )
            assert isinstance(result, tuple)
            v2, report = result
            return v2, report
        return v, None

    # ---- diagnósticos ------------------------------------------------------

    def connection_diagnostics(self) -> ConnectionDiagnostics:
        """Diagnósticos de la Fase 2 (axiomas LC + curvatura)."""
        return self._connection_diagnostics

    def geodesic_flow_report(self) -> Dict[str, Any]:
        d = self._connection_diagnostics
        c = self._christoffel_data
        return {
            "metric_dimension": self._n,
            "christoffel_frob_norm": c.frobenius_norm,
            "christoffel_inf_norm": c.infinity_norm,
            "is_static_metric": c.is_static,
            "condition_number_reg": c.condition_number_reg,
            "inverse_residual": c.inverse_residual,
            "inverse_residual_bound": c.inverse_residual_bound,
            "spectral_gap_min": c.spectral_gap_min,
            "torsion_norm": d.torsion_norm,
            "covd_metric_norm": d.covd_metric_norm,
            "riemann_norm": d.riemann_norm,
            "ricci_norm": d.ricci_norm,
            "bianchi_norm": d.bianchi_norm,
            "torsion_passed": d.torsion_passed,
            "metric_compat_passed": d.metric_compat_passed,
            "bianchi_passed": d.bianchi_passed,
            "all_axioms_passed": d.all_passed(),
            "dt_default": self.DEFAULT_DT,
            "dt_max_stable_unit": self._dt_max_stable_unit,
            "enforce_norm_conservation": self._enforce_norm_conservation,
            "parallel_transport_order": self._pt_order,
            "agent": "LeviCivitaConnectionAgent v8.0.0",
        }

    # ---- contrato categórico -----------------------------------------------

    def forward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Acción del morfismo sobre un estado categórico:
        interpreta el payload como \(v\in T_p\mathcal{M}\) y aplica
        un paso geodésico.
        """
        psi = np.asarray(state.payload, dtype=np.float64).reshape(-1)
        if psi.shape != (self._n,):
            raise DimensionalMismatchError(
                f"payload dim={psi.shape[0]} ≠ n={self._n}."
            )
        v = TangentVector(coordinates=psi)
        v2 = self.enforce_geodesic_flow(v)
        assert isinstance(v2, TangentVector)
        return CategoricalState(
            payload=v2.coordinates,
            label=f"{getattr(state, 'label', '')}::levi_civita_forward",
        )

    def backward(self, state: CategoricalState) -> CategoricalState:
        r"""
        Convención MIC: el reverso del flujo geodésico se obtiene con
        \(dt\mapsto -dt\). Aquí se reutiliza ``forward`` con el mismo
        contrato de payload (el consumidor controla el signo de \(dt\)
        vía estado extendido si lo requiere).
        """
        return self.forward(state)

    # ---- propiedades -------------------------------------------------------

    @property
    def metric_tensor(self) -> NDArray[np.float64]:
        return self._G.copy()

    @property
    def metric_inverse(self) -> NDArray[np.float64]:
        return self._G_inv.copy()

    @property
    def christoffel_symbols(self) -> NDArray[np.float64]:
        return self._Gamma.copy()

    @property
    def metric_dimension(self) -> int:
        return self._n

    @property
    def dimension(self) -> int:
        return self._n

    @property
    def is_static(self) -> bool:
        return self._christoffel_data.is_static

    @property
    def dt_max_stable_unit(self) -> float:
        r"""\(dt_{\max}\) de referencia a \(\|v\|_G=1\)."""
        return self._dt_max_stable_unit


# =============================================================================
# EXPORTACIÓN CANÓNICA
# =============================================================================

__all__ = [
    "TopologicalTorsionError",
    "GeodesicDeviationError",
    "MetricCompatibilityError",
    "ChristoffelInstabilityError",
    "LeviCivitaParameterError",
    "DimensionalMismatchError",
    "BianchiIdentityError",
    "MetricDerivativeProvider",
    "StaticMetricDerivative",
    "CallableMetricDerivative",
    "ChristoffelData",
    "ConnectionDiagnostics",
    "GeodesicStepReport",
    "LeviCivitaConnectionAgent",
    "TangentVector",
    "CotangentVector",
]
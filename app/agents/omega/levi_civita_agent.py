# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Levi-Civita Connection Agent (Maestro de Sinfonía Métrica)          ║
║ Ruta   : app/agents/omega/levi_civita_agent.py                               ║
║ Versión: 9.0.0-Poincare-Celestial-RSI-L3-Symplectic-Categorical-Doctoral     ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA, MECÁNICA CELESTE DE POINCARÉ Y RSI NIVEL 3 (Rigor PhD):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor de calibre, integrador simpléctico celeste y meta-regulador de
automejora recursiva de Nivel 3 (RSI-L3) opera en la cúspide del Estrato Omega
($$V_{\Omega}$$, Nivel 0.5) del ecosistema APU Filter. Actúa como el **Maestro de
Sinfonía Métrica y Celeste** sobre el fibrado de fases $$TM \oplus T^*M$$.
Su función matemática es imponer de forma incondicional el transporte paralelo,
la compatibilidad métrica de Koszul, la invariancia integral de Poincaré-Cartan
y la estabilidad orbital de Kolmogorov-Arnold-Moser (KAM), subyugando las
transiciones del Grafo Acíclico Dirigido (DAG) a las leyes de la geometría
Riemanniana y la mecánica celeste Hamiltoniana.

AXIOMÁTICA RIEMANNIANA, MECÁNICA DE POINCARÉ Y AUTOMEJORA RECURSIVA (RSI-L3):
────────────────────────────────────────────────────────────────────────────────

  [A1] Teorema Fundamental de la Geometría Riemanniana (Levi-Civita):
       Dada la variedad Riemanniana $$(M, G)$$ existe una única conexión afín
       $$\nabla$$ compatible con la métrica y libre de torsión:
       $$\nabla G = 0 \quad \wedge \quad T(X, Y) = \nabla_X Y - \nabla_Y X - [X, Y] = 0$$

  [A2] Compatibilidad Métrica de Koszul y Refinamiento de Banach (RSI-L3 Fase 1):
       $$\Gamma^r_{mn} = \frac{1}{2} G^{rk} \left( \partial_m G_{kn} + \partial_n G_{mk} - \partial_k G_{mn} \right)$$
       Donde $$G^{-1}$$ se auto-mejora recursivamente en el álgebra de Banach
       $$\mathcal{B}(\mathbb{R}^n)$$ vía iteración hiper-potencia cúbica de
       Newton-Schulz-Chebyshev:
       $$R_k = I - G X_k, \qquad X_{k+1} = \mathrm{Sym}\!\left(X_k (I + R_k + R_k^2)\right)$$

  [A3] Purificación Ortogonal de Torsión y No-Metricidad (RSI-L3 Fase 2):
       Si la aritmética de punto flotante introduce un defecto residual
       $$\Delta_{\gamma\mu\nu} = (\nabla_\gamma G)_{\mu\nu}$$, el meta-funtor
       de Nivel 3 sintetiza la corrección exacta en el subespacio de Koszul:
       $$\delta\Gamma^r_{mn} = \frac{1}{2} G^{rk}\left(\Delta_{mkn} + \Delta_{nmk} - \Delta_{kmn}\right)$$

  [A4] Mecánica Celeste de Poincaré, Flujo Hamiltoniano y Forma Simpléctica:
       Mediante el isomorfismo musical $$\flat: TM \to T^*M$$ ($$p_\mu = G_{\mu\nu}v^\nu$$),
       la ecuación geodésica equivale al sistema Hamiltoniano canónico con
       Hamiltoniano cinético $$H(q, p) = \frac{1}{2} G^{\mu\nu}(q) p_\mu p_\nu$$
       que preserva la 2-forma simpléctica $$\omega = dp_\mu \wedge dq^\mu$$ y el
       primer invariante integral relativo de Poincaré:
       $$\oint_{\gamma_t} p_\mu \, dq^\mu = \mathrm{constante} \implies \frac{dH}{d\tau} \equiv 0$$

  [A5] Ecuación Variacional de Jacobi-Poincaré, Caos Homoclínico y Teoría KAM:
       La estabilidad orbital de dos geodésicas infinitamente próximas separadas
       por el campo de Jacobi $$J^\mu$$ obedece la ecuación mareal de Poincaré:
       $$\frac{D^2 J^\mu}{d\tau^2} + \mathcal{K}^\mu{}_\rho(v)\, J^\rho = 0, \qquad \mathcal{K}^\mu{}_\rho(v) = R^\mu{}_{\nu\rho\sigma} v^\nu v^\sigma$$
       El espectro del operador mareal $$\mathcal{K}(v)$$ determina el exponente
       local de Lyapunov $$\lambda_{\max}(v) = \max_i \operatorname{Re}\sqrt{\max(-\lambda_i(\mathcal{K}), 0)}$$
       y el margen diofántico KAM frente a resonancias destructivas.

SUTURA DE FLUJO Y CONTRATOS FUNTORIALES EN 3 FASES ANIDADAS:
────────────────────────────────────────────────────────────────────────────────
La definición formal final del último método de cada fase constituye el punto
exacto de continuación e inicio de la fase subsiguiente:

  Fase 1 ──► Phase1_ChristoffelEngine
             Último método: `build_christoffel(raw_metric) -> ChristoffelData`
                                     │
                                     ▼ (Continuación Formal Directa)
  Fase 2 ──► Phase2_TorsionFreeConnection
             Primer método: `_ingest_phase1_continuation(data: ChristoffelData)`
             Último método: `verify_axioms(data: ChristoffelData) -> ConnectionDiagnostics`
                                     │
                                     ▼ (Continuación Formal Directa)
  Fase 3 ──► Phase3_PoincareGeodesicIntegrator
             Primer método: `_ingest_phase2_continuation(data, diagnostics)`
             Último método: `execute_categorical_morphism(state) -> CategoricalState`

Funtor Maestro de Sinfonía Métrica y Celeste:
  $$\mathcal{Z}_{\mathrm{LeviCivita}}^{(3)} = \Phi_3^{\mathrm{Poincaré}} \circ \Phi_2^{\mathrm{Koszul-KAM}} \circ \Phi_1^{\mathrm{Banach-CSMD}}$$
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
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
# Dependencias del ecosistema (stubs deterministas para ejecución aislada)
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
# Constantes de rigor numérico (Espectral / Wilkinson / Poincaré / RSI-L3)
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

# Constantes de Mecánica Celeste de Poincaré (Yoshida 4th-Order Symplectic)
_CBRT_2: float = float(2.0 ** (1.0 / 3.0))
_YOSHIDA_W1: float = float(1.0 / (2.0 - _CBRT_2))
_YOSHIDA_W0: float = float(-_CBRT_2 / (2.0 - _CBRT_2))
_YOSHIDA_WEIGHTS: Tuple[float, float, float] = (
    _YOSHIDA_W1,
    _YOSHIDA_W0,
    _YOSHIDA_W1,
)

# Constantes de Automejora Recursiva de Nivel 3 (RSI-L3) y CSMD
_CSMD_STEP: float = 1.0e-20
_RSI_L3_MAX_NEWTON_SCHULZ_ITERS: int = 5
_RSI_L3_MAX_SUBSTEP_DEPTH: int = 3
_RSI_L3_CONTRACTION_MAX: float = 0.95
_LYAPUNOV_CHAOS_THRESHOLD: float = 1.0e2


# =============================================================================
# SECCIÓN 0 — EXCEPCIONES TOPOLÓGICAS Y ESPECTRALES
# =============================================================================


class TopologicalTorsionError(TopologicalInvariantError):
    r"""\(T^r_{mn}=\Gamma^r_{mn}-\Gamma^r_{nm}\neq 0\)."""


class GeodesicDeviationError(TopologicalInvariantError):
    """Desviación del flujo geodésico / overflow RK4-Yoshida / deriva de Poincaré."""


class MetricCompatibilityError(TopologicalInvariantError):
    r"""\(\nabla G\neq 0\) fuera de tolerancia de Koszul."""


class ChristoffelInstabilityError(TopologicalInvariantError):
    r"""Γ no finito o \(\|\Gamma\|_F\) excesiva."""


class LeviCivitaParameterError(TopologicalInvariantError):
    """Parámetro de control fuera de rango."""


class DimensionalMismatchError(TopologicalInvariantError):
    """Dimensión de vectores/tensores incompatible con n."""


class BianchiIdentityError(TopologicalInvariantError):
    r"""Primera identidad de Bianchi algebraica violada fuera de tolerancia."""


# =============================================================================
# SECCIÓN 0b — PROTOCOLOS DE DERIVADA MÉTRICA Y EXTENSIÓN HIPERCOMPLEJA (CSMD)
# =============================================================================


@runtime_checkable
class MetricDerivativeProvider(Protocol):
    r"""
    Proveedor del tensor de derivadas parciales de la métrica \(\partial G\).

    Contrato
    --------
    ``derivative(n) -> dG`` con shape \((n,n,n)\) donde
    \(dG[k,i,j]=\partial_k G_{ij}\) y \(dG[k,i,j]=dG[k,j,i]\).
    """

    def derivative(self, n: int) -> NDArray[np.float64]:
        ...


class StaticMetricDerivative:
    r"""\(dG\equiv 0\) (variedad localmente plana en coordenadas normales de Riemann)."""

    def derivative(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            raise DimensionalMismatchError(
                f"StaticMetricDerivative: n={n} inválido."
            )
        return np.zeros((n, n, n), dtype=np.float64)


class CallableMetricDerivative:
    r"""Adapta un callable \(n\mapsto dG\) al protocolo ``MetricDerivativeProvider``."""

    def __init__(self, fn: Callable[[int], NDArray[np.float64]]) -> None:
        self._fn = fn

    def derivative(self, n: int) -> NDArray[np.float64]:
        return np.asarray(self._fn(n), dtype=np.float64)


class ComplexStepMetricDerivative:
    r"""
    Diferenciación Métrica de Paso Complejo (CSMD) libre de cancelación sustractiva.

    Extiende analíticamente el campo métrico \(G: \mathbb{R}^n \to \mathrm{Sym}^+_n(\mathbb{R})\)
    al espacio complejo \(\mathbb{C}^n\) y evalúa:

    \[
      \partial_k G_{ij}(x_0)
        = \frac{\operatorname{Im}\!\bigl(G_{ij}(x_0 + i\,h\,\mathbf{e}_k)\bigr)}{h}
          + \mathcal{O}(h^2),
      \qquad h = 10^{-20}.
    \]
    """

    def __init__(
        self,
        metric_field_fn: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
        base_point: Optional[NDArray[np.float64]] = None,
        step_size: float = _CSMD_STEP,
    ) -> None:
        if step_size <= 0.0:
            raise LeviCivitaParameterError(
                f"step_size CSMD debe ser > 0; recibido {step_size}."
            )
        self._metric_fn = metric_field_fn
        self._base_point = (
            np.asarray(base_point, dtype=np.float64).reshape(-1)
            if base_point is not None
            else None
        )
        self._h = float(step_size)

    def derivative(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            raise DimensionalMismatchError(
                f"ComplexStepMetricDerivative: n={n} inválido."
            )
        x0 = (
            self._base_point
            if self._base_point is not None
            else np.zeros(n, dtype=np.float64)
        )
        if x0.shape != (n,):
            raise DimensionalMismatchError(
                f"ComplexStepMetricDerivative: base_point shape={x0.shape} ≠ ({n},)."
            )
        dG = np.zeros((n, n, n), dtype=np.float64)
        for k in range(n):
            x_pert = x0.astype(np.complex128, copy=True)
            x_pert[k] += 1j * self._h
            G_complex = np.asarray(self._metric_fn(x_pert), dtype=np.complex128)
            if G_complex.shape != (n, n):
                raise DimensionalMismatchError(
                    f"CSMD: G_complex shape={G_complex.shape} ≠ ({n},{n})."
                )
            slice_k = np.imag(G_complex) / self._h
            dG[k] = 0.5 * (slice_k + slice_k.T)
        return dG


# =============================================================================
# SECCIÓN 1 — DTOs INMUTABLES Y REGISTRO EPISTÉMICO RSI-L3
# =============================================================================


@dataclass(frozen=True, slots=True)
class RSILevel3Telemetry:
    r"""
    Certificado epistémico de Automejora Recursiva de Nivel 3 (RSI-L3).

    Documenta la contracción de punto fijo en el álgebra de operadores,
    las iteraciones de refinamiento cúbico de Newton-Schulz, la proyección
    ortogonal de Koszul y la síntesis dinámica de la ley simpléctica.
    """

    newton_schulz_iterations: int
    banach_contraction_factor: float
    koszul_defect_before: float
    koszul_defect_after: float
    koszul_purification_applied: bool
    integrator_law_synthesized: str
    substep_recursion_depth: int
    poincare_symplectic_error: float


@dataclass(frozen=True, slots=True)
class ChristoffelData:
    r"""
    **Salida terminal de la Fase 1** — precondición formal de entrada para
    ``Phase2_TorsionFreeConnection._ingest_phase1_continuation`` y
    ``Phase2_TorsionFreeConnection.verify_axioms``.

    Convención de índices
    ---------------------
    - ``Gamma[r, m, n] = Γ^r_{mn}``
    - ``dG[k, i, j]   = ∂_k G_{ij}``
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
    spectral_radius_max: float = 1.0
    newton_schulz_iterations: int = 0
    rsi_l3_precondition_gain: float = 1.0
    poincare_symplectic_capacity: float = math.pi

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
    **Salida terminal de la Fase 2** — precondición operativa formal para
    ``Phase3_PoincareGeodesicIntegrator._ingest_phase2_continuation``.

    Codifica la verificación de los axiomas de Levi-Civita, curvatura de
    Riemann/Ricci/Escalar, identidad de Bianchi y estabilidad KAM de Poincaré.
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
    scalar_curvature: float = 0.0
    sectional_curvature_bound: float = 0.0
    kam_diophantine_margin: float = 1.0
    rsi_l3_purification_applied: bool = False
    R_tensor: Optional[NDArray[np.float64]] = None
    Ric_tensor: Optional[NDArray[np.float64]] = None
    purified_Gamma: Optional[NDArray[np.float64]] = None

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
            "scalar_curvature": self.scalar_curvature,
            "sectional_curvature_bound": self.sectional_curvature_bound,
            "kam_diophantine_margin": self.kam_diophantine_margin,
            "rsi_l3_purification_applied": self.rsi_l3_purification_applied,
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
    r"""
    Informe de integración geodésica y celeste de Poincaré (Fase 3).

    Monitorea la norma **G**, la conservación del Hamiltoniano cinético
    \(H(q,p)=\tfrac12\|v\|_G^2\), el operador mareal de Jacobi-Poincaré,
    el exponente local de Lyapunov y la ley sintetizada por RSI-L3.
    """

    v_initial_norm_G: float
    v_final_norm_G: float
    norm_drift_G: float
    acceleration_norm_euclid: float
    dt: float
    dt_max_stable: float
    is_stable: bool
    renormalized: bool = False
    hamiltonian_initial: float = 0.0
    hamiltonian_final: float = 0.0
    poincare_invariant_drift: float = 0.0
    lyapunov_exponent_local: float = 0.0
    jacobi_tidal_norm: float = 0.0
    rsi_l3_scheme_selected: str = "RK4"
    rsi_l3_recursion_depth: int = 0


# =============================================================================
# SECCIÓN 2 — ORQUESTADOR SOBERANO CON 3 FASES ANIDADAS CONTINUAS
# =============================================================================


class LeviCivitaConnectionAgent(Morphism):
    r"""
    Agente de Conexión de Levi-Civita v9.0.0 — Integra Geometría Riemanniana,
    Mecánica Celeste de Poincaré y Automejora Recursiva de Nivel 3 (RSI-L3)
    en 3 Fases Anidadas estrictamente continuas:

    \[
      \mathcal{Z}_{\mathrm{LeviCivita}}^{(3)}
        = \mathrm{Phase3\_PoincareGeodesicIntegrator}
          \circ \mathrm{Phase2\_TorsionFreeConnection}
          \circ \mathrm{Phase1\_ChristoffelEngine}.
    \]
    """

    DEFAULT_DT: float = _DEFAULT_DT

    # =========================================================================
    # FASE 1 — MOTOR DE CHRISTOFFEL, CSMD Y REFINAMIENTO DE BANACH (RSI-L3)
    # =========================================================================

    class Phase1_ChristoffelEngine:
        r"""
        **FASE 1 ANIDADA** — Precondicionamiento espectral, refinamiento
        recursivo de Banach de Nivel 3 (Newton-Schulz-Chebyshev), capacidad
        simpléctica de Gromov-Poincaré y ensamblaje de Koszul de \(\Gamma^r_{mn}\).

        Continuación formal hacia Fase 2
        --------------------------------
        El último método de esta fase, ``build_christoffel``, retorna una
        instancia sellada de ``ChristoffelData``, cuya estructura es el
        argumento formal exacto del método inicial de la Fase 2:
        ``Phase2_TorsionFreeConnection._ingest_phase1_continuation``.
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
            """1.1 Validación topológica y algebraica de la métrica de entrada."""
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
            asym = float(np.max(np.abs(G - G.T)))
            if asym > 1.0e-8 * max(float(np.max(np.abs(G))), 1.0):
                logger.warning(
                    "[Fase1] G cruda asimétrica (max=%.3e); el precondicionador "
                    "proyectará al subespacio simétrico.",
                    asym,
                )

        @staticmethod
        def _inverse_residual(
            G: NDArray[np.float64], G_inv: NDArray[np.float64]
        ) -> float:
            r"""1.2 Residuo dual en el álgebra de Banach: \(\max(\|GG^{-1}-I\|_F,\|G^{-1}G-I\|_F)/n\)."""
            n = G.shape[0]
            I = np.eye(n, dtype=np.float64)
            r_plus = float(la.norm(G @ G_inv - I, "fro")) / n
            r_minus = float(la.norm(G_inv @ G - I, "fro")) / n
            return max(r_plus, r_minus)

        @classmethod
        def _rsi_l3_newton_schulz_refinement(
            cls,
            G: NDArray[np.float64],
            G_inv: NDArray[np.float64],
            max_iters: int = _RSI_L3_MAX_NEWTON_SCHULZ_ITERS,
        ) -> Tuple[NDArray[np.float64], float, int, float]:
            r"""
            1.3 Automejora Recursiva de Nivel 3 (RSI-L3) en \(\mathcal{B}(\mathbb{R}^n)\).

            Refina \(X_0 = G^{-1}\) mediante el método hiper-potencia cúbico de
            Chebyshev-Schulz:

            \[
              R_k = I - G X_k, \qquad
              X_{k+1} = \mathrm{Sym}\!\bigl(X_k (I + R_k + R_k^2)\bigr),
            \]

            con convergencia cúbica \(\|R_{k+1}\|_F \le \|R_k\|_F^3\) siempre que
            \(\|R_0\|_F < 1\).
            """
            n = G.shape[0]
            I = np.eye(n, dtype=np.float64)
            X = 0.5 * (G_inv + G_inv.T)
            res_initial = cls._inverse_residual(G, X)
            res_cur = res_initial
            iters_done = 0
            target_floor = 4.0 * _MACHINE_EPS

            if res_cur <= target_floor:
                return X, res_cur, 0, 1.0

            for _ in range(max_iters):
                R = I - (G @ X)
                norm_R = float(la.norm(R, "fro"))
                if norm_R >= _RSI_L3_CONTRACTION_MAX:
                    break
                poly = I + R + (R @ R)
                X_cand = X @ poly
                X_cand = 0.5 * (X_cand + X_cand.T)
                if not np.all(np.isfinite(X_cand)):
                    break
                res_cand = cls._inverse_residual(G, X_cand)
                if res_cand < res_cur:
                    X = X_cand
                    res_cur = res_cand
                    iters_done += 1
                    if res_cur <= target_floor:
                        break
                else:
                    break

            gain = res_initial / max(res_cur, _MACHINE_EPS)
            return X, res_cur, iters_done, float(gain)

        @staticmethod
        def _spectral_bounds(G: NDArray[np.float64]) -> Tuple[float, float]:
            r"""1.4 Cotas espectrales extremas \((\lambda_{\min}(G), \lambda_{\max}(G))\)."""
            ev = np.linalg.eigvalsh(0.5 * (G + G.T))
            return float(ev[0]), float(ev[-1])

        @staticmethod
        def _poincare_symplectic_capacity(lambda_max: float) -> float:
            r"""
            1.5 Capacidad Simpléctica de Gromov-Poincaré del elipsoide cinético unitario:

            \[
              c_{\mathrm{Poincaré}}(B_G(1)) = \frac{\pi}{\sqrt{\max(\lambda_{\max}(G), \varepsilon)}}.
            \]
            """
            return float(math.pi / math.sqrt(max(lambda_max, _MACHINE_EPS)))

        def _compute_metric_derivative(self, n: int) -> NDArray[np.float64]:
            r"""1.6 Obtención de \(dG[k,i,j]=\partial_k G_{ij}\) vía el proveedor inyectado."""
            return np.asarray(
                self._dG_provider.derivative(n), dtype=np.float64
            )

        @staticmethod
        def _symmetrize_dG(dG: NDArray[np.float64]) -> NDArray[np.float64]:
            r"""1.7 Proyección simétrica exacta \(dG[k,i,j]=dG[k,j,i]\)."""
            return 0.5 * (dG + dG.transpose(0, 2, 1))

        def _validate_derivative_tensor(
            self, dG: NDArray[np.float64], n: int
        ) -> NDArray[np.float64]:
            """1.8 Certificación dimensional, finitud y simetrización de ∂G."""
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

        @staticmethod
        def _compute_christoffel_terms(
            G_inv: NDArray[np.float64],
            dG: NDArray[np.float64],
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            r"""
            1.9 Contracciones tensoriales de la fórmula de Koszul:

            - \(T_1{}^r_{mn} = G^{rk}\partial_m G_{kn}\)
            - \(T_2{}^r_{mn} = G^{rk}\partial_n G_{mk}\)
            - \(T_3{}^r_{mn} = G^{rk}\partial_k G_{mn}\)
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

        def build_christoffel(
            self, raw_metric: NDArray[np.float64]
        ) -> ChristoffelData:
            r"""
            1.10 **MÉTODO TERMINAL DE LA FASE 1.**

            Construye y certifica los símbolos de Christoffel de segunda especie
            \(\Gamma^r_{mn}\) con refinamiento RSI-L3 de Banach y capacidad
            simpléctica de Poincaré.

            Continuación formal hacia la Fase 2
            -----------------------------------
            El objeto ``ChristoffelData`` devuelto por este método constituye la
            entrada formal inmediata de:
            ``Phase2_TorsionFreeConnection._ingest_phase1_continuation(data)`` y
            ``Phase2_TorsionFreeConnection.verify_axioms(data)``.
            """
            self._validate_metric_input(raw_metric)
            pm: PreconditionedMetric = self._preconditioner.precondition(
                raw_metric
            )
            G = np.asarray(pm.G, dtype=np.float64)
            G_inv_raw = np.asarray(pm.G_inv, dtype=np.float64)
            n = int(pm.matrix_dimension)

            # RSI-L3: Refinamiento cúbico de Newton-Schulz-Chebyshev sobre G^{-1}
            G_inv, inv_res, ns_iters, rsi_gain = (
                self._rsi_l3_newton_schulz_refinement(G, G_inv_raw)
            )

            inv_bound = (
                _INV_RESIDUAL_FACTOR
                * _MACHINE_EPS
                * max(float(pm.condition_number_reg), 1.0)
            )
            if inv_res > inv_bound:
                logger.warning(
                    "[Fase1] Residuo de inversa %.3e > cota Wilkinson %.3e "
                    "(κ_reg=%.3e, NS_iters=%d).",
                    inv_res,
                    inv_bound,
                    pm.condition_number_reg,
                    ns_iters,
                )

            dG = self._validate_derivative_tensor(
                self._compute_metric_derivative(n), n
            )
            T1, T2, T3 = self._compute_christoffel_terms(G_inv, dG)
            Gamma = 0.5 * (T1 + T2 - T3)

            if not np.all(np.isfinite(Gamma)):
                raise ChristoffelInstabilityError(
                    "Γ contiene valores no finitos tras el ensamblado de Koszul."
                )
            frob = float(la.norm(Gamma, "fro"))
            inf_norm = float(np.max(np.abs(Gamma))) if Gamma.size else 0.0
            if frob > _CHRISTOFFEL_FINITE_TOL:
                raise ChristoffelInstabilityError(
                    f"‖Γ‖_F={frob:.3e} > umbral {_CHRISTOFFEL_FINITE_TOL:.3e}."
                )

            is_static = bool(np.all(dG == 0.0))
            lam_min, lam_max = self._spectral_bounds(G)
            symp_cap = self._poincare_symplectic_capacity(lam_max)

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
                spectral_gap_min=lam_min,
                spectral_radius_max=lam_max,
                newton_schulz_iterations=ns_iters,
                rsi_l3_precondition_gain=rsi_gain,
                poincare_symplectic_capacity=symp_cap,
            )
            logger.info(
                "[Fase1] Christoffel: n=%d, ‖Γ‖_F=%.3e, ‖Γ‖_∞=%.3e, "
                "static=%s, κ_reg=%.3e, inv_res=%.3e (NS=%d, gain=%.2fx), "
                "c_Poincaré=%.3e.",
                n,
                frob,
                inf_norm,
                is_static,
                pm.condition_number_reg,
                inv_res,
                ns_iters,
                rsi_gain,
                symp_cap,
            )
            # ═══► CONTINUACIÓN FORMAL DIRECTA HACIA FASE 2 ◄═══
            return data

    # =========================================================================
    # FASE 2 — CONEXIÓN SIN TORSIÓN, BIANCHI, CURVATURA KAM Y PURIFICACIÓN RSI-L3
    #          (Inicio = Continuación directa de Phase1.build_christoffel)
    # =========================================================================

    class Phase2_TorsionFreeConnection:
        r"""
        **FASE 2 ANIDADA** — Continuación formal de ``Phase1_ChristoffelEngine.build_christoffel``.

        Recibe ``ChristoffelData``, verifica torsión nula, ejecuta la purificación
        ortogonal de Koszul (RSI-L3) sobre \(\nabla G\), evalúa los tensores de
        Riemann, Ricci, curvatura escalar, primera identidad de Bianchi y el
        margen diofántico de estabilidad KAM de Poincaré.

        Continuación formal hacia Fase 3
        --------------------------------
        El último método de esta fase, ``verify_axioms``, entrega
        ``ConnectionDiagnostics``, que junto a ``ChristoffelData`` inicia
        formalmente ``Phase3_PoincareGeodesicIntegrator._ingest_phase2_continuation``.
        """

        def __init__(
            self,
            torsion_tolerance: float = _TORSION_TOLERANCE,
            metric_compat_tolerance: float = _METRIC_COMPAT_TOLERANCE,
            bianchi_tolerance: float = _BIANCHI_TOLERANCE,
            enable_rsi_l3_purification: bool = True,
        ) -> None:
            self.TORSION_TOLERANCE = float(torsion_tolerance)
            self.METRIC_COMPAT_TOLERANCE = float(metric_compat_tolerance)
            self.BIANCHI_TOLERANCE = float(bianchi_tolerance)
            self._enable_rsi_l3 = bool(enable_rsi_l3_purification)

        @staticmethod
        def _ingest_phase1_continuation(
            data: ChristoffelData,
        ) -> ChristoffelData:
            r"""
            2.1 **MÉTODO INICIAL DE LA FASE 2 (Continuación de Fase 1).**

            Valida el contrato funtorial del objeto ``ChristoffelData`` emitido
            por ``Phase1_ChristoffelEngine.build_christoffel``.
            """
            if not isinstance(data, ChristoffelData):
                raise TypeError(
                    f"[Fase2] Se esperaba ChristoffelData de Fase 1; "
                    f"se recibió {type(data).__name__}."
                )
            if data.spectral_gap_min <= 0.0:
                raise ChristoffelInstabilityError(
                    f"[Fase2] Métrica no definida positiva: "
                    f"λ_min={data.spectral_gap_min:.3e} ≤ 0."
                )
            return data

        @staticmethod
        def _compute_torsion_tensor(
            Gamma: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""2.2 Tensor de torsión \(T^r_{mn}=\Gamma^r_{mn}-\Gamma^r_{nm}\)."""
            return Gamma - Gamma.transpose(0, 2, 1)

        def _verify_zero_torsion(
            self, torsion: NDArray[np.float64], n: int
        ) -> float:
            """2.3 Certificación de conexión libre de torsión."""
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
            2.4 Derivada covariante de la métrica:

            \[
              (\nabla_\gamma G)_{mn}
                = \partial_\gamma G_{mn}
                  - \Gamma^k_{\gamma m} G_{kn}
                  - \Gamma^k_{\gamma n} G_{mk}.
            \]
            """
            term1 = np.einsum("kgm,kn->gmn", Gamma, G, optimize=True)
            term2 = np.einsum("kgn,mk->gmn", Gamma, G, optimize=True)
            return dG - term1 - term2

        @classmethod
        def _rsi_l3_purify_connection(
            cls,
            Gamma: NDArray[np.float64],
            dG: NDArray[np.float64],
            G: NDArray[np.float64],
            G_inv: NDArray[np.float64],
            covd_initial: NDArray[np.float64],
            compat_tol: float,
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], bool]:
            r"""
            2.5 Proyección Ortogonal de Koszul-Levi-Civita (RSI-L3 Fase 2).

            Si el defecto de no-metricidad \(\Delta_{gmn} = (\nabla_g G)_{mn}\) es
            no nulo por redondeo de punto flotante pero se encuentra dentro del
            corredor reparable (\(< \tau_{\mathrm{compat}}\)), sintetiza la
            corrección exacta de Levi-Civita:

            \[
              \delta\Gamma^r_{mn}
                = \frac{1}{2} G^{rk}\bigl(\Delta_{mkn} + \Delta_{nmk} - \Delta_{kmn}\bigr),
              \qquad
              \Gamma_{\mathrm{purified}} = \mathrm{Sym}_{mn}(\Gamma + \delta\Gamma).
            \]
            """
            norm_init = float(la.norm(covd_initial, "fro"))
            floor_tol = 4.0 * _MACHINE_EPS * max(float(la.norm(dG, "fro")), 1.0)
            if norm_init <= floor_tol or norm_init >= compat_tol:
                return Gamma, covd_initial, False

            Delta = 0.5 * (covd_initial + covd_initial.transpose(0, 2, 1))
            dT1 = np.einsum("rk,mkn->rmn", G_inv, Delta, optimize=True)
            dT2 = np.einsum("rk,nmk->rmn", G_inv, Delta, optimize=True)
            dT3 = np.einsum("rk,kmn->rmn", G_inv, Delta, optimize=True)
            delta_Gamma = 0.5 * (dT1 + dT2 - dT3)

            Gamma_cand = Gamma + delta_Gamma
            Gamma_cand = 0.5 * (Gamma_cand + Gamma_cand.transpose(0, 2, 1))
            covd_cand = cls._compute_covd_metric(Gamma_cand, dG, G)
            norm_cand = float(la.norm(covd_cand, "fro"))

            if norm_cand < norm_init:
                return Gamma_cand, covd_cand, True
            return Gamma, covd_initial, False

        def _verify_metric_compatibility(
            self, covd: NDArray[np.float64]
        ) -> float:
            """2.6 Certificación del axioma de compatibilidad métrica ∇G = 0."""
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
            2.7 Parte cuadrática algebraica del tensor de curvatura de Riemann:

            \[
              R^r{}_{smn}
                = \Gamma^r_{mk}\Gamma^k_{ns} - \Gamma^r_{nk}\Gamma^k_{ms}.
            \]
            """
            term_A = np.einsum("rmk,kns->rsmn", Gamma, Gamma, optimize=True)
            term_B = np.einsum("rnk,kms->rsmn", Gamma, Gamma, optimize=True)
            return term_A - term_B

        @staticmethod
        def _compute_ricci(R: NDArray[np.float64]) -> NDArray[np.float64]:
            r"""2.8 Tensor de Ricci por contracción canónica \(R_{sn}=R^r{}_{srn}\)."""
            return np.einsum("rsrn->sn", R, optimize=True)

        @staticmethod
        def _compute_scalar_curvature(
            Ric: NDArray[np.float64], G_inv: NDArray[np.float64]
        ) -> float:
            r"""2.9 Curvatura escalar de Einstein-Hilbert \(R = G^{sn} R_{sn}\)."""
            return float(np.einsum("sn,sn->", G_inv, Ric, optimize=True))

        @staticmethod
        def _compute_bianchi_first(
            R: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            2.10 Primera Identidad de Bianchi algebraica (suma cíclica exacta):

            \[
              B^r{}_{smn}
                = R^r{}_{smn} + R^r{}_{mns} + R^r{}_{nsm} = 0.
            \]

            Para ``R`` con ejes ``(0:r, 1:s, 2:m, 3:n)``:
            - ``R.transpose(0, 2, 3, 1)[r, s, m, n] = R[r, m, n, s] = R^r_{mns}``
            - ``R.transpose(0, 3, 1, 2)[r, s, m, n] = R[r, n, s, m] = R^r_{nsm}``
            """
            return (
                R
                + R.transpose(0, 2, 3, 1)
                + R.transpose(0, 3, 1, 2)
            )

        def _verify_bianchi(
            self, B: NDArray[np.float64], soft: bool = True
        ) -> float:
            """2.11 Verificación numérica de la primera identidad de Bianchi."""
            b_norm = float(la.norm(B, "fro"))
            if b_norm >= self.BIANCHI_TOLERANCE:
                msg = (
                    f"Bianchi-I: ‖B‖_F={b_norm:.3e} ≥ "
                    f"tol={self.BIANCHI_TOLERANCE:.3e}."
                )
                if soft:
                    logger.warning("[Fase2] %s (soft).", msg)
                else:
                    raise BianchiIdentityError(msg)
            return b_norm

        @staticmethod
        def _compute_poincare_kam_diagnostics(
            R: NDArray[np.float64],
            spectral_gap_min: float,
            condition_number: float,
        ) -> Tuple[float, float]:
            r"""
            2.12 Cota de Curvatura Seccional y Margen Diofántico KAM de Poincaré:

            \[
              K_{\max} \le \frac{\|R\|_F}{\lambda_{\min}(G)}, \qquad
              \delta_{\mathrm{KAM}} = \frac{1}{1 + \|R\|_F\,\kappa(G)}.
            \]
            """
            r_norm = float(la.norm(R, "fro"))
            sec_bound = r_norm / max(spectral_gap_min, _MACHINE_EPS)
            kam_margin = 1.0 / (1.0 + r_norm * max(condition_number, 1.0))
            return float(sec_bound), float(kam_margin)

        def verify_axioms(
            self, data: ChristoffelData
        ) -> ConnectionDiagnostics:
            r"""
            2.13 **MÉTODO TERMINAL DE LA FASE 2.**

            Audita y purifica la conexión de Levi-Civita recibida desde la Fase 1.

            Continuación formal hacia la Fase 3
            -----------------------------------
            El objeto ``ConnectionDiagnostics`` retornado aquí es el argumento
            formal de entrada que inicia la Fase 3 en:
            ``Phase3_PoincareGeodesicIntegrator._ingest_phase2_continuation(data, diag)``.
            """
            data = self._ingest_phase1_continuation(data)
            Gamma = data.Gamma
            dG = data.dG
            G = data.G
            G_inv = data.G_inv
            n = data.dimension

            # --- Axioma 1: Torsión nula -----------------------------------
            torsion = self._compute_torsion_tensor(Gamma)
            t_norm = self._verify_zero_torsion(torsion, n)

            # --- Axioma 2: Compatibilidad métrica ∇G = 0 + RSI-L3 ---------
            covd_raw = self._compute_covd_metric(Gamma, dG, G)
            purified = False
            if self._enable_rsi_l3:
                Gamma, covd, purified = self._rsi_l3_purify_connection(
                    Gamma, dG, G, G_inv, covd_raw, self.METRIC_COMPAT_TOLERANCE
                )
            else:
                covd = covd_raw
            c_norm = self._verify_metric_compatibility(covd)

            # --- Curvatura de Riemann, Ricci, Escalar, Bianchi y KAM ------
            R = self._compute_riemann_quadratic(Gamma)
            r_norm = float(la.norm(R, "fro"))
            Ric = self._compute_ricci(R)
            ric_norm = float(la.norm(Ric, "fro"))
            scalar_curv = self._compute_scalar_curvature(Ric, G_inv)

            B = self._compute_bianchi_first(R)
            b_norm = self._verify_bianchi(B, soft=True)

            sec_bound, kam_margin = self._compute_poincare_kam_diagnostics(
                R, data.spectral_gap_min, data.condition_number_reg
            )

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
                scalar_curvature=scalar_curv,
                sectional_curvature_bound=sec_bound,
                kam_diophantine_margin=kam_margin,
                rsi_l3_purification_applied=purified,
                R_tensor=R,
                Ric_tensor=Ric,
                purified_Gamma=Gamma,
            )
            logger.info("[Fase2] Axiomas y KAM: %s", diag.summary())
            # ═══► CONTINUACIÓN FORMAL DIRECTA HACIA FASE 3 ◄═══
            return diag

    # =========================================================================
    # FASE 3 — INTEGRADOR GEODÉSICO CELESTE DE POINCARÉ Y SÍNTESIS RSI-L3
    #          (Inicio = Continuación directa de Phase2.verify_axioms)
    # =========================================================================

    class Phase3_PoincareGeodesicIntegrator:
        r"""
        **FASE 3 ANIDADA** — Continuación formal de ``Phase2_TorsionFreeConnection.verify_axioms``.

        Integra las ecuaciones geodésicas y de transporte paralelo utilizando:
        1. Mecánica Celeste de Poincaré: conservación del Hamiltoniano cinético
           \(H(q,p)=\tfrac12 G^{\mu\nu}p_\mu p_\nu\), composición simpléctica de
           4.º orden de Yoshida, ecuación variacional mareal de Jacobi-Poincaré
           y exponentes locales de Lyapunov.
        2. Automejora Recursiva de Nivel 3 (RSI-L3): selección y bisección
           recursiva del operador de evolución según rigidez espectral, caos
           homoclínico local y deriva del invariante integral de Poincaré.
        """

        def __init__(
            self,
            christoffel_data: ChristoffelData,
            diagnostics: ConnectionDiagnostics,
            preconditioner: Optional[MetricSpectralPreconditioner] = None,
            enforce_norm_conservation: bool = True,
            parallel_transport_order: int = 2,
            default_dt: float = _DEFAULT_DT,
        ) -> None:
            c_data, c_diag = self._ingest_phase2_continuation(
                christoffel_data, diagnostics
            )
            self._christoffel_data = c_data
            self._diagnostics = c_diag
            self._G = c_data.G
            self._G_inv = c_data.G_inv
            self._n = c_data.dimension
            self._Gamma = (
                c_diag.purified_Gamma
                if c_diag.purified_Gamma is not None
                else c_data.Gamma
            )
            self._R = (
                c_diag.R_tensor
                if c_diag.R_tensor is not None
                else np.zeros((self._n, self._n, self._n, self._n), dtype=np.float64)
            )
            self._gamma_inf = max(c_data.infinity_norm, _GAMMA_INF_FLOOR)
            self._enforce_norm_conservation = bool(enforce_norm_conservation)
            self._pt_order = int(parallel_transport_order)
            self.DEFAULT_DT = float(default_dt)

            self._musical_engine = MusicalIsomorphismEngine(
                metric_tensor=self._G,
                preconditioner=preconditioner,
            )
            self._dt_max_stable_unit = self._compute_max_stable_dt_for_speed(1.0)
            self._last_rsi_telemetry: RSILevel3Telemetry = RSILevel3Telemetry(
                newton_schulz_iterations=c_data.newton_schulz_iterations,
                banach_contraction_factor=c_data.inverse_residual,
                koszul_defect_before=c_diag.covd_metric_norm,
                koszul_defect_after=c_diag.covd_metric_norm,
                koszul_purification_applied=c_diag.rsi_l3_purification_applied,
                integrator_law_synthesized="RK4_POINCARE_READY",
                substep_recursion_depth=0,
                poincare_symplectic_error=0.0,
            )

        @staticmethod
        def _ingest_phase2_continuation(
            christoffel_data: ChristoffelData,
            diagnostics: ConnectionDiagnostics,
        ) -> Tuple[ChristoffelData, ConnectionDiagnostics]:
            r"""
            3.1 **MÉTODO INICIAL DE LA FASE 3 (Continuación de Fase 2).**

            Recibe y enlaza formalmente ``ChristoffelData`` (Fase 1) y
            ``ConnectionDiagnostics`` (Fase 2), verificando la satisfacción
            conjunta de los axiomas de Levi-Civita antes de habilitar el flujo
            simpléctico sobre \(TM \oplus T^*M\).
            """
            if not isinstance(christoffel_data, ChristoffelData):
                raise TypeError(
                    f"[Fase3] Se esperaba ChristoffelData; "
                    f"se recibió {type(christoffel_data).__name__}."
                )
            if not isinstance(diagnostics, ConnectionDiagnostics):
                raise TypeError(
                    f"[Fase3] Se esperaba ConnectionDiagnostics de Fase 2; "
                    f"se recibió {type(diagnostics).__name__}."
                )
            if christoffel_data.dimension != diagnostics.dimension:
                raise DimensionalMismatchError(
                    f"[Fase3] Dimensión inconsistente entre Fase 1 "
                    f"({christoffel_data.dimension}) y Fase 2 ({diagnostics.dimension})."
                )
            if not (diagnostics.torsion_passed and diagnostics.metric_compat_passed):
                raise TopologicalInvariantError(
                    "[Fase3] Los axiomas fundamentales de Fase 2 no fueron superados."
                )
            return christoffel_data, diagnostics

        def _g_norm(self, v: NDArray[np.float64]) -> float:
            r"""3.2 Norma Riemanniana \(\|v\|_G=\sqrt{\max(v^\top G v,\,0)}\)."""
            v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
            q = float(v_arr @ self._G @ v_arr)
            return math.sqrt(max(q, 0.0))

        def _hamiltonian_energy(self, v: NDArray[np.float64]) -> float:
            r"""
            3.3 Hamiltoniano Celeste de Poincaré en el fibrado cotangente:

            \[
              H(q, p) = \frac{1}{2} G^{\mu\nu}(q) p_\mu p_\nu
                      = \frac{1}{2} G_{\mu\nu}(q) v^\mu v^\nu
                      = \frac{1}{2}\|v\|_G^2.
            \]
            """
            v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
            return 0.5 * max(float(v_arr @ self._G @ v_arr), 0.0)

        def _jacobi_poincare_tidal_operator(
            self, v: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            3.4 Operador Mareal de la Ecuación Variacional de Jacobi-Poincaré:

            \[
              \mathcal{K}^r{}_m(v) = R^r{}_{smn}\, v^s v^n.
            \]

            Gobierna la desviación geodésica \(\frac{D^2 J^r}{d\tau^2} + \mathcal{K}^r{}_m(v) J^m = 0\).
            """
            v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
            return np.einsum("rsmn,s,n->rm", self._R, v_arr, v_arr, optimize=True)

        def _estimate_local_lyapunov_and_tidal_norm(
            self, v: NDArray[np.float64]
        ) -> Tuple[float, float]:
            r"""
            3.5 Exponente Local de Lyapunov \(\lambda_{\max}(v)\) y norma mareal.

            En direcciones de curvatura seccional negativa (autovalores negativos
            de \(\mathcal{K}(v)\)), las trayectorias divergen exponencialmente
            con tasa \(\lambda_{\max} = \max_i \sqrt{\max(-\operatorname{Re}\mu_i(\mathcal{K}), 0)}\).
            """
            K = self._jacobi_poincare_tidal_operator(v)
            tidal_norm = float(la.norm(K, "fro"))
            if tidal_norm <= _MACHINE_EPS:
                return 0.0, 0.0
            ev = np.linalg.eigvals(K)
            neg_parts = np.maximum(-np.real(ev), 0.0)
            lyapunov = float(np.sqrt(np.max(neg_parts)))
            return lyapunov, tidal_norm

        def _compute_max_stable_dt_for_speed(self, speed_G: float) -> float:
            r"""3.6 Paso máximo estable de Courant-Friedrichs-Lewy Riemaniano."""
            denom = self._gamma_inf * max(float(speed_G), 0.0) + _MACHINE_EPS
            return _DT_MAX_STABLE_FACTOR / denom

        def _validate_velocity_vector(
            self, velocity: TangentVector, caller: str = ""
        ) -> NDArray[np.float64]:
            """3.7 Validación de pertenencia al fibrado tangente T_p M."""
            if not isinstance(velocity, TangentVector):
                raise TypeError(
                    f"{caller}: se esperaba TangentVector, "
                    f"recibido {type(velocity).__name__}."
                )
            coords = np.asarray(
                velocity.coordinates, dtype=np.float64
            ).reshape(-1)
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
            """3.8 Auditoría de estabilidad espectral del paso temporal dt."""
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

        def _geodesic_acceleration(
            self, v: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""3.9 Campo de aceleración geodésica \(a^\mu = -\Gamma^\mu_{rs} v^r v^s\)."""
            a = -np.einsum("mrs,r,s->m", self._Gamma, v, v, optimize=True)
            if not np.all(np.isfinite(a)):
                raise GeodesicDeviationError(
                    f"Aceleración geodésica no finita. "
                    f"‖v‖_G={self._g_norm(v):.3e}, "
                    f"‖Γ‖_F={self._christoffel_data.frobenius_norm:.3e}."
                )
            return a

        def geodesic_rhs(
            self, v_coords: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""3.10 Lado derecho de la ecuación geodésica (contrato EikonalAgent)."""
            v = np.asarray(v_coords, dtype=np.float64).reshape(-1)
            if v.shape != (self._n,):
                raise DimensionalMismatchError(
                    f"geodesic_rhs: dim={v.shape[0]} ≠ n={self._n}."
                )
            return self._geodesic_acceleration(v)

        def _rk4_step(
            self, v: NDArray[np.float64], dt: float
        ) -> NDArray[np.float64]:
            r"""3.11 Paso clásico de Runge-Kutta de 4.º orden sobre \(\dot v = -\Gamma(v,v)\)."""
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

        def _implicit_midpoint_substep(
            self, v: NDArray[np.float64], sub_dt: float
        ) -> NDArray[np.float64]:
            r"""
            3.12 Sub-paso Simpléctico del Punto Medio Implícito de Gauss-Legendre:

            \[
              v_{k+1} = v_k + \Delta\tau\, f\!\left(\frac{v_k + v_{k+1}}{2}\right),
            \]
            resuelto mediante iteración de punto fijo de Banach de orden 2.
            """
            v_guess = v + sub_dt * self._geodesic_acceleration(v)
            for _ in range(3):
                v_mid = 0.5 * (v + v_guess)
                v_guess = v + sub_dt * self._geodesic_acceleration(v_mid)
            if not np.all(np.isfinite(v_guess)):
                raise GeodesicDeviationError(
                    f"Sub-paso simpléctico de Gauss-Legendre no finito (sub_dt={sub_dt:.3e})."
                )
            return v_guess

        def _yoshida4_symplectic_step(
            self, v: NDArray[np.float64], dt: float
        ) -> NDArray[np.float64]:
            r"""
            3.13 Integrador Simpléctico de Yoshida de 4.º Orden (Mecánica Celeste de Poincaré).

            Compone simétricamente tres sub-mapas simplécticos de punto medio con
            los pesos de Yoshida \((w_1, w_0, w_1)\) donde \(w_1 + w_0 + w_1 = 1\):

            \[
              \Phi_{\Delta t}^{\mathrm{Yoshida4}}
                = \Phi_{w_1 \Delta t}^{\mathrm{Mid}}
                \circ \Phi_{w_0 \Delta t}^{\mathrm{Mid}}
                \circ \Phi_{w_1 \Delta t}^{\mathrm{Mid}}.
            \]
            """
            v_cur = v
            for w in _YOSHIDA_WEIGHTS:
                v_cur = self._implicit_midpoint_substep(v_cur, w * dt)
            return v_cur

        def _rsi_l3_meta_integrate(
            self, v: NDArray[np.float64], dt: float, dt_max: float
        ) -> Tuple[NDArray[np.float64], str, int]:
            r"""
            3.14 Meta-Integrador con Automejora Recursiva de Nivel 3 (RSI-L3).

            Evalúa la rigidez espectral \(\sigma = dt/dt_{\max}\), el exponente
            local de Lyapunov \(\lambda_{\max}(v)\) y la deriva del Hamiltoniano
            de Poincaré \(\Delta H / H_0\). Si detecta excitación no lineal o
            deriva energética superior al umbral, conmuta la ley de evolución
            hacia composición simpléctica de Yoshida-4 y aplica bisección diádica
            recursiva (\(2^k\) sub-pasos) hasta restaurar el invariante integral.
            """
            H_i = self._hamiltonian_energy(v)
            if H_i <= _MACHINE_EPS or self._christoffel_data.is_static:
                v_f = self._rk4_step(v, dt)
                return v_f, "RK4", 0

            lyapunov, _ = self._estimate_local_lyapunov_and_tidal_norm(v)
            stiffness = dt / max(dt_max, _MACHINE_EPS)

            # Selección primaria de ley de evolución
            if stiffness > 0.25 or lyapunov > _LYAPUNOV_CHAOS_THRESHOLD:
                v_f = self._yoshida4_symplectic_step(v, dt)
                scheme = "YOSHIDA4_SYMPLECTIC"
            else:
                v_f = self._rk4_step(v, dt)
                scheme = "RK4"

            H_f = self._hamiltonian_energy(v_f)
            rel_H_drift = abs(H_f - H_i) / max(H_i, _MACHINE_EPS)

            # Bisección recursiva RSI-L3 si la deriva de Poincaré excede tolerancia
            depth = 0
            while (
                rel_H_drift > _GEODESIC_NORM_DRIFT_TOL
                and depth < _RSI_L3_MAX_SUBSTEP_DEPTH
            ):
                depth += 1
                n_sub = 1 << depth
                sub_dt = dt / float(n_sub)
                v_sub = v.copy()
                for _ in range(n_sub):
                    v_sub = self._yoshida4_symplectic_step(v_sub, sub_dt)
                v_f = v_sub
                H_f = self._hamiltonian_energy(v_f)
                rel_H_drift = abs(H_f - H_i) / max(H_i, _MACHINE_EPS)
                scheme = f"RSI_L3_YOSHIDA4_SUBSTEP_2^{depth}"

            self._last_rsi_telemetry = RSILevel3Telemetry(
                newton_schulz_iterations=self._christoffel_data.newton_schulz_iterations,
                banach_contraction_factor=self._christoffel_data.inverse_residual,
                koszul_defect_before=self._diagnostics.covd_metric_norm,
                koszul_defect_after=self._diagnostics.covd_metric_norm,
                koszul_purification_applied=self._diagnostics.rsi_l3_purification_applied,
                integrator_law_synthesized=scheme,
                substep_recursion_depth=depth,
                poincare_symplectic_error=rel_H_drift,
            )
            return v_f, scheme, depth

        def _renormalize_G(
            self,
            v: NDArray[np.float64],
            target_norm_G: float,
        ) -> NDArray[np.float64]:
            r"""
            3.15 Retracción proyectiva sobre la hipersuperficie de energía constante de Poincaré:

            \[
              \Sigma_{E_0} = \bigl\{v \in T_p M : \tfrac12\|v\|_G^2 = E_0\bigr\}.
            \]
            """
            n_cur = self._g_norm(v)
            if n_cur <= _MACHINE_EPS:
                return v
            return v * (target_norm_G / n_cur)

        def _build_step_report(
            self,
            v_i: NDArray[np.float64],
            v_f: NDArray[np.float64],
            dt: float,
            dt_max: float,
            renormalized: bool,
            scheme: str = "RK4",
            recursion_depth: int = 0,
        ) -> GeodesicStepReport:
            """3.16 Construcción del reporte diagnóstico geodésico y celeste."""
            n_i = self._g_norm(v_i)
            n_f = self._g_norm(v_f)
            drift = abs(n_f - n_i) / max(n_i, _MACHINE_EPS)

            H_i = 0.5 * (n_i * n_i)
            H_f = 0.5 * (n_f * n_f)
            poincare_drift = abs(H_f - H_i) / max(H_i, _MACHINE_EPS)

            acc = self._geodesic_acceleration(v_i)
            lyapunov, tidal_norm = self._estimate_local_lyapunov_and_tidal_norm(v_i)

            if drift > _GEODESIC_NORM_DRIFT_TOL and not renormalized:
                logger.warning(
                    "Deriva de norma G / Poincaré: |‖v_f‖_G−‖v_i‖_G|/‖v_i‖_G=%.3e > "
                    "tol=%.3e (dt=%.3e, ‖Γ‖_F=%.3e, scheme=%s).",
                    drift,
                    _GEODESIC_NORM_DRIFT_TOL,
                    dt,
                    self._christoffel_data.frobenius_norm,
                    scheme,
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
                hamiltonian_initial=H_i,
                hamiltonian_final=H_f,
                poincare_invariant_drift=poincare_drift,
                lyapunov_exponent_local=lyapunov,
                jacobi_tidal_norm=tidal_norm,
                rsi_l3_scheme_selected=scheme,
                rsi_l3_recursion_depth=recursion_depth,
            )

        def enforce_geodesic_flow(
            self,
            velocity: TangentVector,
            dt: Optional[float] = None,
            *,
            return_report: bool = False,
            renormalize: Optional[bool] = None,
        ) -> Union[TangentVector, Tuple[TangentVector, GeodesicStepReport]]:
            r"""
            3.17 Propaga \(v \in T_p M\) un paso \(dt\) sobre la geodésica
            gobernada por el meta-integrador simpléctico RSI-L3.
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
            v_f, scheme, depth = self._rsi_l3_meta_integrate(v_i, _dt, dt_max)

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
                    v_i, v_f, _dt, dt_max, renormed, scheme, depth
                )
            return out

        def _gamma_action(
            self,
            ydot: NDArray[np.float64],
            V: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""3.18 Operador bilineal de conexión \(\bigl(\Gamma(\dot\gamma,V)\bigr)^m=\Gamma^m_{rs}\dot\gamma^r V^s\)."""
            return np.einsum("mrs,r,s->m", self._Gamma, ydot, V, optimize=True)

        def parallel_transport(
            self,
            vector: TangentVector,
            tangent_to_curve: TangentVector,
            dt: Optional[float] = None,
        ) -> TangentVector:
            r"""
            3.19 Transporte paralelo de Levi-Civita \(\nabla_{\dot\gamma} V = 0\):

            \[
              \dot V^m = -\Gamma^m_{rs}\,\dot\gamma^r V^s.
            \]

            Soporta esquemas de Orden 1 (Euler), Orden 2 (Predictor-Corrector de Heun)
            y Orden 4 (Magnus-RK4 isométrico).
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
            elif self._pt_order == 2:
                k1 = self._gamma_action(ydot, V)
                V_pred = V - _dt * k1
                k2 = self._gamma_action(ydot, V_pred)
                V_t = V - _dt * _HEUN_HALF * (k1 + k2)
            else:
                half = 0.5 * _dt
                k1 = -self._gamma_action(ydot, V)
                k2 = -self._gamma_action(ydot, V + half * k1)
                k3 = -self._gamma_action(ydot, V + half * k2)
                k4 = -self._gamma_action(ydot, V + _dt * k3)
                V_t = V + (_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if not np.all(np.isfinite(V_t)):
                raise GeodesicDeviationError(
                    f"Transporte paralelo no finito (dt={_dt:.3e}, "
                    f"order={self._pt_order})."
                )
            return TangentVector(coordinates=V_t)

        def transport_to_finance_oracle(
            self,
            logistics_flow: TangentVector,
            dt: Optional[float] = None,
            apply_geodesic_correction: bool = True,
        ) -> Tuple[CotangentVector, GeodesicStepReport]:
            r"""
            3.20 Transformada de Legendre / Isomorfismo Musical Bemol
            \(T\mathcal{M}\xrightarrow{\gamma}T\mathcal{M}\xrightarrow{\flat}T^*\mathcal{M}\).
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
                H0 = 0.5 * (speed * speed)
                lyap, tidal = self._estimate_local_lyapunov_and_tidal_norm(v_arr)
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
                    hamiltonian_initial=H0,
                    hamiltonian_final=H0,
                    poincare_invariant_drift=0.0,
                    lyapunov_exponent_local=lyap,
                    jacobi_tidal_norm=tidal,
                    rsi_l3_scheme_selected="STATIC_BYPASS",
                    rsi_l3_recursion_depth=0,
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
            3.21 Transformada de Legendre Inversa / Isomorfismo Musical Sostenido
            \(T^*\mathcal{M}\xrightarrow{\sharp}T\mathcal{M}\xrightarrow{\gamma?}T\mathcal{M}\).
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

        def build_flow_report(self) -> Dict[str, Any]:
            """3.22 Síntesis de telemetría unificada de las 3 Fases Anidadas."""
            d = self._diagnostics
            c = self._christoffel_data
            t = self._last_rsi_telemetry
            return {
                "metric_dimension": self._n,
                "christoffel_frob_norm": c.frobenius_norm,
                "christoffel_inf_norm": c.infinity_norm,
                "is_static_metric": c.is_static,
                "condition_number_reg": c.condition_number_reg,
                "inverse_residual": c.inverse_residual,
                "inverse_residual_bound": c.inverse_residual_bound,
                "spectral_gap_min": c.spectral_gap_min,
                "spectral_radius_max": c.spectral_radius_max,
                "poincare_symplectic_capacity": c.poincare_symplectic_capacity,
                "newton_schulz_iterations": c.newton_schulz_iterations,
                "rsi_l3_precondition_gain": c.rsi_l3_precondition_gain,
                "torsion_norm": d.torsion_norm,
                "covd_metric_norm": d.covd_metric_norm,
                "riemann_norm": d.riemann_norm,
                "ricci_norm": d.ricci_norm,
                "scalar_curvature": d.scalar_curvature,
                "sectional_curvature_bound": d.sectional_curvature_bound,
                "kam_diophantine_margin": d.kam_diophantine_margin,
                "bianchi_norm": d.bianchi_norm,
                "torsion_passed": d.torsion_passed,
                "metric_compat_passed": d.metric_compat_passed,
                "bianchi_passed": d.bianchi_passed,
                "all_axioms_passed": d.all_passed(),
                "rsi_l3_purification_applied": d.rsi_l3_purification_applied,
                "rsi_l3_last_integrator_law": t.integrator_law_synthesized,
                "rsi_l3_last_recursion_depth": t.substep_recursion_depth,
                "dt_default": self.DEFAULT_DT,
                "dt_max_stable_unit": self._dt_max_stable_unit,
                "enforce_norm_conservation": self._enforce_norm_conservation,
                "parallel_transport_order": self._pt_order,
                "agent": "LeviCivitaConnectionAgent v9.0.0",
            }

        def execute_categorical_morphism(
            self, state: CategoricalState
        ) -> CategoricalState:
            r"""
            3.23 **MÉTODO TERMINAL DE LA FASE 3 (Clausura Funtorial del Agente).**

            Ejecuta la acción del endofuntor \(\mathcal{Z}_{\mathrm{LeviCivita}}^{(3)}\)
            sobre un ``CategoricalState`` y sella la transición en la categoría MIC.
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

    # Alias canónico para retrocompatibilidad estructural
    Phase3_GeodesicIntegrator = Phase3_PoincareGeodesicIntegrator

    # =========================================================================
    # ENSAMBLAJE DEL ORQUESTADOR Y DELEGACIÓN A LAS 3 FASES ANIDADAS
    # =========================================================================

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
        if parallel_transport_order not in (1, 2, 4):
            raise LeviCivitaParameterError(
                f"parallel_transport_order ∈ {{1,2,4}}; "
                f"se obtuvo {parallel_transport_order}."
            )

        G_raw = np.asarray(metric_tensor, dtype=np.float64)
        self._enforce_norm_conservation = bool(enforce_norm_conservation)
        self._pt_order = int(parallel_transport_order)
        self._bianchi_tolerance = float(bianchi_tolerance)

        # ------------------------------------------------------------------
        # FASE 1 — Construcción de Γ y Refinamiento RSI-L3 (Salida: ChristoffelData)
        # ------------------------------------------------------------------
        self.phase1 = LeviCivitaConnectionAgent.Phase1_ChristoffelEngine(
            preconditioner=preconditioner,
            metric_derivative=metric_derivative,
        )
        self._christoffel_data: ChristoffelData = (
            self.phase1.build_christoffel(G_raw)
        )

        # ------------------------------------------------------------------
        # FASE 2 — Continuación Formal: Axiomas LC, Bianchi y KAM (Salida: ConnectionDiagnostics)
        # ------------------------------------------------------------------
        self.phase2 = LeviCivitaConnectionAgent.Phase2_TorsionFreeConnection(
            torsion_tolerance=torsion_tolerance,
            metric_compat_tolerance=metric_compat_tolerance,
            bianchi_tolerance=bianchi_tolerance,
        )
        self._connection_diagnostics: ConnectionDiagnostics = (
            self.phase2.verify_axioms(self._christoffel_data)
        )

        # ------------------------------------------------------------------
        # FASE 3 — Continuación Formal: Integrador Simpléctico de Poincaré y RSI-L3
        # ------------------------------------------------------------------
        self.phase3 = LeviCivitaConnectionAgent.Phase3_PoincareGeodesicIntegrator(
            christoffel_data=self._christoffel_data,
            diagnostics=self._connection_diagnostics,
            preconditioner=preconditioner,
            enforce_norm_conservation=enforce_norm_conservation,
            parallel_transport_order=parallel_transport_order,
            default_dt=self.DEFAULT_DT,
        )

        # Enlaces de estado cacheados en el nivel superior
        self._G = self.phase3._G
        self._G_inv = self.phase3._G_inv
        self._n = self.phase3._n
        self._Gamma = self.phase3._Gamma
        self._gamma_inf = self.phase3._gamma_inf
        self._musical_engine = self.phase3._musical_engine
        self._dt_max_stable_unit = self.phase3._dt_max_stable_unit

        try:
            super().__init__(name="LeviCivitaConnectionAgent")
        except TypeError:
            super().__init__()

        logger.info(
            "[LeviCivitaConnectionAgent] v9: n=%d, ‖Γ‖_F=%.3e, ‖Γ‖_∞=%.3e, "
            "static=%s, torsion=%.2e, ∇G=%.2e, Bianchi=%.2e, KAM=%.3f, "
            "dt_max(|v|_G=1)=%.3e, renorm=%s, pt_order=%d.",
            self._n,
            self._christoffel_data.frobenius_norm,
            self._christoffel_data.infinity_norm,
            self._christoffel_data.is_static,
            self._connection_diagnostics.torsion_norm,
            self._connection_diagnostics.covd_metric_norm,
            self._connection_diagnostics.bianchi_norm,
            self._connection_diagnostics.kam_diophantine_margin,
            self._dt_max_stable_unit,
            self._enforce_norm_conservation,
            self._pt_order,
        )

    # -------------------------------------------------------------------------
    # Puente de métodos hacia Fase 3 (preserva API pública e interna al 100%)
    # -------------------------------------------------------------------------

    def _g_norm(self, v: NDArray[np.float64]) -> float:
        return self.phase3._g_norm(v)

    def _compute_max_stable_dt_for_speed(self, speed_G: float) -> float:
        return self.phase3._compute_max_stable_dt_for_speed(speed_G)

    def _validate_velocity_vector(
        self, velocity: TangentVector, caller: str = ""
    ) -> NDArray[np.float64]:
        return self.phase3._validate_velocity_vector(velocity, caller=caller)

    def _validate_integration_step(
        self,
        dt: float,
        speed_G: float,
        caller: str = "",
    ) -> Tuple[float, float]:
        return self.phase3._validate_integration_step(
            dt, speed_G, caller=caller
        )

    def geodesic_rhs(
        self, v_coords: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.phase3.geodesic_rhs(v_coords)

    def _geodesic_acceleration(
        self, v: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.phase3._geodesic_acceleration(v)

    def _rk4_step(
        self, v: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        return self.phase3._rk4_step(v, dt)

    def _yoshida4_symplectic_step(
        self, v: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        return self.phase3._yoshida4_symplectic_step(v, dt)

    def _renormalize_G(
        self,
        v: NDArray[np.float64],
        target_norm_G: float,
    ) -> NDArray[np.float64]:
        return self.phase3._renormalize_G(v, target_norm_G)

    def enforce_geodesic_flow(
        self,
        velocity: TangentVector,
        dt: Optional[float] = None,
        *,
        return_report: bool = False,
        renormalize: Optional[bool] = None,
    ) -> Union[TangentVector, Tuple[TangentVector, GeodesicStepReport]]:
        return self.phase3.enforce_geodesic_flow(
            velocity,
            dt=dt,
            return_report=return_report,
            renormalize=renormalize,
        )

    def parallel_transport(
        self,
        vector: TangentVector,
        tangent_to_curve: TangentVector,
        dt: Optional[float] = None,
    ) -> TangentVector:
        return self.phase3.parallel_transport(
            vector, tangent_to_curve, dt=dt
        )

    def transport_to_finance_oracle(
        self,
        logistics_flow: TangentVector,
        dt: Optional[float] = None,
        apply_geodesic_correction: bool = True,
    ) -> Tuple[CotangentVector, GeodesicStepReport]:
        return self.phase3.transport_to_finance_oracle(
            logistics_flow,
            dt=dt,
            apply_geodesic_correction=apply_geodesic_correction,
        )

    def transport_to_logistics_manifold(
        self,
        financial_force: CotangentVector,
        apply_post_geodesic: bool = False,
        dt: Optional[float] = None,
    ) -> Tuple[TangentVector, Optional[GeodesicStepReport]]:
        return self.phase3.transport_to_logistics_manifold(
            financial_force,
            apply_post_geodesic=apply_post_geodesic,
            dt=dt,
        )

    def connection_diagnostics(self) -> ConnectionDiagnostics:
        """Diagnósticos de la Fase 2 (axiomas LC + curvatura + KAM)."""
        return self._connection_diagnostics

    def rsi_level3_telemetry(self) -> RSILevel3Telemetry:
        """Certificado epistémico de Automejora Recursiva de Nivel 3."""
        return self.phase3._last_rsi_telemetry

    def geodesic_flow_report(self) -> Dict[str, Any]:
        return self.phase3.build_flow_report()

    def forward(self, state: CategoricalState) -> CategoricalState:
        return self.phase3.execute_categorical_morphism(state)

    def backward(self, state: CategoricalState) -> CategoricalState:
        return self.forward(state)

    # -------------------------------------------------------------------------
    # Propiedades inmutables de inspección
    # -------------------------------------------------------------------------

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
    "ComplexStepMetricDerivative",
    "RSILevel3Telemetry",
    "ChristoffelData",
    "ConnectionDiagnostics",
    "GeodesicStepReport",
    "LeviCivitaConnectionAgent",
    "TangentVector",
    "CotangentVector",
]
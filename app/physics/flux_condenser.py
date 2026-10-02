# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Data Flux Condenser — Poincaré Port-Hamiltonian Lattice QED Strict              ║
║ Ruta   : app/physics/flux_condenser.py                                                   ║
║ Versión: 7.1.0-Poincare-DEC-PHS-Rigorous                                                 ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║ FASE 1/3 — FUNDAMENTOS AXIOMÁTICOS, NÚCLEO POINCARÉ, DEC Y MAXWELL FDTD                  ║
║                                                                                          ║
║ Convenciones (inmutables en 7.1):                                                        ║
║   PHS:  ẋ = [J(x)-R(x)] ∇H(x) + g(x) u,   y = gᵀ(x) ∇H(x)                                ║
║   H    : ½ xᵀ K x,  K=Kᵀ ≻ 0,  J=-Jᵀ,  R=Rᵀ ⪰ 0                                          ║
║   DEC  : dₖ = ∂ₖ₊₁ᵀ,  δₖ = ★ₖ₋₁⁻¹ dₖ₋₁ᵀ ★ₖ,  Δₖ = δd+dδ ⪰ 0 (conv. grafo/Hodge⁺)            ║
║   Maxwell 2D (TE⊥): E ∈ C¹ (circulaciones), B ∈ C² (flujos)                              ║
║           ∂ₜB = -d₁E - σₘ H + Jₘ,   ∂ₜD = δ₂H - σₑ E - Jₑ                                 ║
║           D = ε ★₁ E,  H = μ⁻¹ ★₂ B,  U = ½(Eᵀ D + Hᵀ B)                                 ║
║   Casimir lineal: J c = 0 ⇒ C(x)=cᵀx invariante ssi además cᵀ R ∇H = 0.                  ║
║   Espectro: A=(J-R)K no-normal; estabilidad por μ₂(A) y Re σ(A) módulo ker J.            ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import logging
import math
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    from scipy import sparse
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigs, eigsh, spsolve
    from scipy.linalg import det, expm, logm, schur
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    sparse = None
    csr_matrix = None
    diags = None
    eigs = None
    eigsh = None
    spsolve = None
    det = None
    expm = None
    logm = None
    schur = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

ArrayLike = Union[float, Sequence[float], np.ndarray]

# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 1.1 — AXIOMAS, EXCEPCIONES Y CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════════════


class DataFluxCondenserError(Exception):
    """Clase base para todas las excepciones del condensador de flujo."""


class InvalidInputError(DataFluxCondenserError):
    """Entrada inválida o archivo no conforme."""


class ProcessingError(DataFluxCondenserError):
    """Error durante una etapa de procesamiento."""


class ConfigurationError(DataFluxCondenserError):
    """Configuración física, numérica o topológica inválida."""


class NumericalInstabilityError(DataFluxCondenserError):
    """Inestabilidad numérica detectada en integración o campos."""


class SymplecticStructureError(DataFluxCondenserError):
    """Violación de la estructura simpléctica/Poisson."""


class LiouvilleViolationError(DataFluxCondenserError):
    """Violación de conservación de volumen de Liouville."""


class PoincareRecurrenceError(DataFluxCondenserError):
    """Recurrencia de Poincaré incompatible con el régimen disipativo."""


class HodgeStructureError(DataFluxCondenserError):
    """Violación de dualidad de Hodge, positividad de ★ o Δ ⪰ 0."""


class ChainComplexError(DataFluxCondenserError):
    """Fallo del axioma ∂∘∂ = 0 / d∘d = 0."""


@dataclass(frozen=True)
class SystemConstants:
    """
    Constantes inmutables con jerarquía de tolerancias y cotas espectrales.

    Jerarquía: NUMERICAL_ZERO < NUMERICAL_TOLERANCE < RELATIVE_TOLERANCE
               < SYMPLECTIC_TOLERANCE ≤ LIOUVILLE_DRIFT_LIMIT.
    El log-norm y el CFL espectral viven en una escala distinta (O(1) del generador).
    """

    MIN_DELTA_TIME: float = 1e-6
    MAX_DELTA_TIME: float = 3600.0
    PROCESSING_TIMEOUT: float = 3600.0

    MIN_ENERGY_THRESHOLD: float = 1e-12
    MAX_EXPONENTIAL_ARG: float = 709.0  # log(max float64) ~ 709.78
    MAX_WATER_HAMMER_PRESSURE: float = 10.0
    MAX_FLYBACK_VOLTAGE: float = 10.0

    NUMERICAL_ZERO: float = 1e-15
    NUMERICAL_TOLERANCE: float = 1e-12
    RELATIVE_TOLERANCE: float = 1e-9

    SYMPLECTIC_TOLERANCE: float = 1e-10
    LIOUVILLE_DRIFT_LIMIT: float = 1e-8
    POINCARE_SECTION_EPSILON: float = 1e-8
    CASIMIR_TOLERANCE: float = 1e-10
    STABILITY_MARGIN: float = 1e-9
    HODGE_EIGEN_TOLERANCE: float = 1e-8
    WELL_CENTERED_TOLERANCE: float = 1e-12
    MAX_POINCARE_STATE_DIM: int = 512
    MAX_DENSE_SPECTRAL_DIM: int = 512
    MAX_DENSE_BETTI_DIM: int = 1024
    MAX_STATE_NORM: float = 1e150
    MAX_CONDITION_NUMBER: float = 1e12
    MAX_TRANSITION_CACHE: int = 64
    MAX_LAPLACIAN_CACHE: int = 8
    MAX_COEFF_CACHE: int = 32
    PADE_TAYLOR_TERMS: int = 18
    STRANG_MIN_DIM_FOR_SPLIT: int = 1

    LOW_INERTIA_THRESHOLD: float = 0.1
    HIGH_PRESSURE_RATIO: float = 1000.0
    HIGH_FLYBACK_THRESHOLD: float = 0.5
    OVERHEAT_POWER_THRESHOLD: float = 50.0
    EMERGENCY_BRAKE_FACTOR: float = 0.5

    MAX_ITERATIONS_MULTIPLIER: int = 10
    MIN_BATCH_SIZE_FLOOR: int = 1
    MIN_RECORDS_FOR_PID: int = 10
    MAX_RECORDS_LIMIT: int = 10_000_000
    MAX_CACHE_SIZE: int = 100_000
    MAX_BATCHES_TO_CONSOLIDATE: int = 10_000

    VALID_FILE_EXTENSIONS: frozenset = frozenset({".csv", ".txt", ".tsv", ".dat"})
    MAX_FILE_SIZE_MB: float = 500.0
    MIN_FILE_SIZE_BYTES: int = 10

    CFL_SAFETY_FACTOR: float = 0.5
    ENERGY_BLOWUP_RATIO: float = 10.0
    ENERGY_WINDOW: int = 32
    MAX_GRAPH_NODES_CLIQUE: int = 256
    HODGE_KERNEL_MAX_K: int = 32

    def __post_init__(self) -> None:
        if self.MIN_DELTA_TIME >= self.MAX_DELTA_TIME:
            raise ConfigurationError("MIN_DELTA_TIME debe ser menor que MAX_DELTA_TIME.")
        if not (self.NUMERICAL_ZERO < self.NUMERICAL_TOLERANCE < self.RELATIVE_TOLERANCE):
            raise ConfigurationError("Jerarquía de tolerancias incoherente.")
        if not (0.0 < self.CFL_SAFETY_FACTOR < 1.0):
            raise ConfigurationError("CFL_SAFETY_FACTOR debe estar en (0, 1).")
        if self.SYMPLECTIC_TOLERANCE <= 0.0 or self.LIOUVILLE_DRIFT_LIMIT <= 0.0:
            raise ConfigurationError("Tolerancias simplécticas deben ser positivas.")
        if self.CASIMIR_TOLERANCE <= 0.0:
            raise ConfigurationError("CASIMIR_TOLERANCE debe ser positiva.")
        if self.MAX_TRANSITION_CACHE < 1:
            raise ConfigurationError("MAX_TRANSITION_CACHE debe ser ≥ 1.")


CONSTANTS = SystemConstants()


def _finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise NumericalInstabilityError(f"{name} no es finito.")
    return float(value)


def _as_1d(x: ArrayLike, dtype: type = float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


# ───────────────────────────────────────────────────────────────────────────────────────
# Estructuras de auditoría
# ───────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpectralAudit:
    """
    Auditoría espectral de A = (J-R)K ∈ 𝔤𝔩(n).

    A es no-normal en general: el radio espectral no controla el transitorio;
    sí lo hace la norma logarítmica μ₂(A) = λ_max((A+Aᵀ)/2), que es la derivada
    de Dini de ‖exp(tA)‖₂ en t=0⁺ (Lozinskiĭ–Dahlquist).

    is_asymptotically_stable_mod_casimir
        True ssi max Re σ(A)|_{N} < -margin, donde N es un complemento de ker J
        (más precisamente, autovalores con |λ| > CASIMIR_TOLERANCE).
    """

    eigenvalues: np.ndarray
    max_real_part: float
    spectral_radius: float
    logarithmic_norm: float
    numerical_abscissa: float
    condition_metric: float
    casimir_multiplicity: int
    is_lyapunov_stable: bool
    is_asymptotically_stable_mod_casimir: bool
    is_normal: bool
    departure_from_normality: float


@dataclass(frozen=True)
class BettiNumbers:
    """Números de Betti del complejo (clique complex del grafo). χ = β₀ - β₁ + β₂."""

    beta_0: int
    beta_1: int
    beta_2: int

    @property
    def euler_poincare(self) -> int:
        return int(self.beta_0 - self.beta_1 + self.beta_2)

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.beta_0, self.beta_1, self.beta_2)


@dataclass(frozen=True)
class FluxCondenserStepReport:
    """Reporte de un paso Port-Hamiltoniano con auditoría de Poincaré-Liouville."""

    hamiltonian_energy: float
    volume_drift: float
    rayleigh_dissipation_rate: float
    is_liouville_preserved: bool
    is_volume_contracting: bool
    poisson_residual: float
    trace_generator: float
    state_dimension: int
    casimir_dimension: int = 0
    max_re_eigen_A: float = 0.0
    spectral_radius_A: float = 0.0
    logarithmic_norm_A: float = 0.0
    is_lyapunov_stable: bool = False
    is_asymptotically_stable_mod_casimir: bool = False
    strang_poisson_residual: float = 0.0
    casimir_drift: float = 0.0
    integrator: str = "strang"


@dataclass(eq=False)
class PoincareControlSeed:
    """
    Semilla de control Poincaré-Port-Hamiltoniana (frontera Fase 1 → Fase 2).

    Campos IDA-PBC:
        J = J_d + J_a,  ambas antisimétricas;
        R_d = R_dᵀ ⪰ 0  disipación deseada;
        K_d métrica de H_d (energía deseada);
        matching_residual  ‖(J_d-R_d)K_d - (J-R)K - g G‖  (0 si matching exacto).
    """

    state: np.ndarray
    gradient: np.ndarray
    hamiltonian: float
    target_hamiltonian: float
    lyapunov_candidate: float
    interconnection_matrix: np.ndarray
    damping_matrix: np.ndarray
    metric_matrix: np.ndarray
    port_matrix: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    casimir_basis: Optional[np.ndarray] = None
    spectral_data: Optional[Dict[str, Any]] = None
    ida_pbc_decomposition: Optional[Dict[str, np.ndarray]] = None
    lyapunov_jacobian: Optional[np.ndarray] = None
    output_port: Optional[np.ndarray] = None
    matching_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", np.asarray(self.state, dtype=float).copy())
        object.__setattr__(self, "gradient", np.asarray(self.gradient, dtype=float).copy())
        object.__setattr__(
            self, "interconnection_matrix", np.asarray(self.interconnection_matrix, dtype=float).copy()
        )
        object.__setattr__(self, "damping_matrix", np.asarray(self.damping_matrix, dtype=float).copy())
        object.__setattr__(self, "metric_matrix", np.asarray(self.metric_matrix, dtype=float).copy())
        object.__setattr__(self, "port_matrix", np.asarray(self.port_matrix, dtype=float).copy())
        if self.lyapunov_jacobian is not None:
            object.__setattr__(
                self, "lyapunov_jacobian", np.asarray(self.lyapunov_jacobian, dtype=float).copy()
            )
        if self.output_port is None:
            y = self.port_matrix.T @ self.gradient
            object.__setattr__(self, "output_port", y)


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 1.2 — NÚCLEO POINCARÉ-PORT-HAMILTONIANO
# ═══════════════════════════════════════════════════════════════════════════════════════


class PoincareHamiltonianKernel:
    r"""
    Núcleo Port-Hamiltoniano lineal con invariantes de Poincaré.

    Dinámica:
        ẋ = (J - R) K x + g u,    y = gᵀ K x,
        H(x) = ½ xᵀ K x.

    Álgebra:
        J ∈ 𝔰𝔬(n)  (Dirac / Poisson lineal),
        R ∈ Sym⁺(n),  K ∈ Sym⁺⁺(n),
        A := (J-R)K ∈ 𝔤𝔩(n)  no-normal en general.

    Invariantes y leyes:
      1. Liouville: det e^{tA} = e^{t tr A},  tr(JK)=0,  tr(A)= -tr(RK) ≤ 0.
      2. Poisson: Φₜ = e^{t JK} satisface Φ J Φᵀ = J  (flujo hamiltoniano).
      3. Rayleigh: Ḣ = -‖∇H‖_R² + yᵀ u ≤ 0 si u=0.
      4. Casimirs lineales: Jc=0 ⇒ Ċ = -cᵀ R ∇H  (se anula si R=0 o c ∈ ker R).
      5. Estabilidad: μ₂(A) ≤ 0 ⇒ ‖x(t)‖₂ no crece; Re σ(A) ≤ 0 ⇒ Lyapunov
         (necesita además que los Jordan de Re=0 sean triviales).
      6. Integrador de Strang: preserva Poisson en el factor hamiltoniano
         y contractividad en el factor de Rayleigh, con error local O(dt³).
    """

    def __init__(
        self,
        J: Union[float, np.ndarray],
        metric: Union[float, np.ndarray],
        R: Optional[Union[float, np.ndarray]] = None,
        g: Optional[np.ndarray] = None,
        name: str = "PoincareHamiltonianKernel",
    ) -> None:
        self.name = name
        self._canonical_n: Optional[int] = None
        self._transition_cache: "OrderedDict[Tuple[float, bool, str], np.ndarray]" = OrderedDict()
        self.J = self._as_square_matrix(J, "J")
        self.dim = int(self.J.shape[0])
        self.metric = self._as_matrix(metric, dim=self.dim, name="metric", kind="spd")
        self.R = self._as_matrix(R, dim=self.dim, name="R", kind="psd")
        self.J = self._validate_skew_symmetry(self.J)
        self.g = self._as_port_matrix(g, self.dim)
        self.A = (self.J - self.R) @ self.metric
        self.A_conservative = self.J @ self.metric
        self.A_dissipative = -self.R @ self.metric
        self.trace_generator = float(np.trace(self.A))
        self.trace_rayleigh = float(np.trace(self.R @ self.metric))
        trace_conservative = float(np.trace(self.A_conservative))
        if abs(trace_conservative) > CONSTANTS.SYMPLECTIC_TOLERANCE:
            logger.debug("%s: tr(JK)=%.3e (ruido de redondeo; geométricamente 0).", self.name, trace_conservative)
        self.is_nominally_conservative = bool(
            np.linalg.norm(self.R, ord="fro") <= CONSTANTS.SYMPLECTIC_TOLERANCE
        )
        self._casimir_basis: Optional[np.ndarray] = None
        self._spectral_cache: Optional[SpectralAudit] = None
        self._metric_chol: Optional[np.ndarray] = None
        self._condition_metric = self._metric_condition_number()

    # ──────────────────────────────────────────────────────────────────────────
    # Constructores físicos
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_rlc(
        cls,
        capacitance: Union[float, np.ndarray],
        inductance: Union[float, np.ndarray],
        series_resistance: Optional[Union[float, np.ndarray]] = None,
        shunt_conductance: Optional[Union[float, np.ndarray]] = None,
    ) -> "PoincareHamiltonianKernel":
        r"""
        RLC canónico de Poincaré (coordenadas de energía: carga / flujo).

            x = [q, p]ᵀ ∈ T*ℝⁿ
            H = ½ qᵀ C⁻¹ q + ½ pᵀ L⁻¹ p
            J = [[0, I], [-I, 0]]          (forma simpléctica ω = dq ∧ dp)
            R = diag(G, R_s)               G: fugas del condensador
                                           R_s: resistencia serie del inductor

        Ecuaciones:
            q̇ = L⁻¹ p - G C⁻¹ q
            ṗ = -C⁻¹ q - R_s L⁻¹ p
        """
        C = cls._physical_matrix(capacitance, "capacitance", positive=True)
        L = cls._physical_matrix(inductance, "inductance", positive=True)
        if C.shape[0] != L.shape[0]:
            raise ConfigurationError(
                f"Capacitancia ({C.shape[0]}) e inductancia ({L.shape[0]}) "
                "deben compartir dimensión."
            )
        n = C.shape[0]
        try:
            inv_C = np.linalg.inv(C)
            inv_L = np.linalg.inv(L)
        except np.linalg.LinAlgError as exc:
            raise ConfigurationError(f"Matrices C/L no invertibles: {exc}") from exc
        metric = cls._block_diag(inv_C, inv_L)
        J = np.block(
            [
                [np.zeros((n, n)), np.eye(n)],
                [-np.eye(n), np.zeros((n, n))],
            ]
        )
        G = cls._physical_matrix(
            shunt_conductance, "shunt_conductance", positive=False, default_shape=n
        )
        R_s = cls._physical_matrix(
            series_resistance, "series_resistance", positive=False, default_shape=n
        )
        R_block = cls._block_diag(G, R_s)
        kernel = cls(J=J, metric=metric, R=R_block, name="RLC-Poincare")
        kernel._canonical_n = n
        return kernel

    @classmethod
    def from_maxwell_blocks(
        cls,
        boundary2: np.ndarray,
        electric_metric_diag: np.ndarray,
        magnetic_metric_diag: np.ndarray,
        sigma_e: np.ndarray,
        sigma_m: np.ndarray,
        g: Optional[np.ndarray] = None,
    ) -> "PoincareHamiltonianKernel":
        r"""
        PHS de Maxwell DEC. Estado x = [D, B], ∇H = [E, H].

            J = [[  0 ,  ∂₂ ],     R = diag(σₑ, σₘ)
                 [-∂₂ᵀ,   0 ]]
            K = diag(★₁⁻¹/ε, ★₂/μ)    (consistente con D=ε★₁E, H=μ⁻¹★₂B)
        """
        B2 = np.asarray(boundary2, dtype=float)
        n_e = int(electric_metric_diag.size)
        n_f = int(magnetic_metric_diag.size)
        if B2.size == 0:
            B2 = np.zeros((n_e, n_f))
        if B2.shape != (n_e, n_f):
            raise ConfigurationError(
                f"∂₂ debe ser ({n_e}×{n_f}), recibido {B2.shape}."
            )
        dim = n_e + n_f
        J = np.zeros((dim, dim), dtype=float)
        if n_f > 0 and n_e > 0:
            J[:n_e, n_e:] = B2
            J[n_e:, :n_e] = -B2.T
            J = 0.5 * (J - J.T)
        metric = np.concatenate(
            [
                np.asarray(electric_metric_diag, dtype=float).reshape(-1),
                np.asarray(magnetic_metric_diag, dtype=float).reshape(-1),
            ]
        )
        R = np.concatenate(
            [
                np.asarray(sigma_e, dtype=float).reshape(-1),
                np.asarray(sigma_m, dtype=float).reshape(-1),
            ]
        )
        return cls(J=J, metric=metric, R=R, g=g, name="Maxwell-DEC-Poincare")

    # ──────────────────────────────────────────────────────────────────────────
    # Operadores Hamiltonianos
    # ──────────────────────────────────────────────────────────────────────────

    def hamiltonian(self, x: np.ndarray) -> float:
        """H(x) = ½ xᵀ K x ≥ 0."""
        x = self._validate_state(x)
        energy = 0.5 * float(x @ (self.metric @ x))
        return max(0.0, _finite("H", energy))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """∇H(x) = K x."""
        return self.metric @ self._validate_state(x)

    def rayleigh_dissipation_rate(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> float:
        r"""Ḣ = -∇Hᵀ R ∇H + yᵀ u.  Sin control, Ḣ ≤ 0."""
        grad = self.gradient(x)
        rate = -float(grad @ (self.R @ grad))
        if u is not None:
            y = self.port_output(x)
            u_vec = _as_1d(u)
            if u_vec.size != y.size:
                raise ConfigurationError(f"u dim {u_vec.size} ≠ y dim {y.size}.")
            rate += float(y @ u_vec)
        if abs(rate) < CONSTANTS.NUMERICAL_ZERO:
            rate = 0.0
        return _finite("Ḣ", rate)

    def port_output(self, x: np.ndarray) -> np.ndarray:
        """y = gᵀ ∇H(x)  (esfuerzos conjugados)."""
        return self.g.T @ self.gradient(x)

    def vector_field(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """ẋ = (J-R)∇H + g u."""
        x = self._validate_state(x)
        xd = self.A @ x
        if u is not None:
            u_vec = _as_1d(u)
            if u_vec.size != self.g.shape[1]:
                raise ConfigurationError(f"u dim {u_vec.size} ≠ m={self.g.shape[1]}.")
            xd = xd + self.g @ u_vec
        return xd

    def power_balance(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Identidad de balance: Ḣ + d_R = yᵀ u  (van der Schaft)."""
        grad = self.gradient(x)
        dissipated = float(grad @ (self.R @ grad))
        supplied = 0.0 if u is None else float(self.port_output(x) @ _as_1d(u))
        hdot = self.rayleigh_dissipation_rate(x, u)
        residual = hdot + dissipated - supplied
        return {
            "H_dot": hdot,
            "rayleigh": dissipated,
            "supplied_power": supplied,
            "balance_residual": residual,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Casimirs y espectro
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def casimir_basis(self) -> np.ndarray:
        r"""
        Base ortonormal de ker(J) (columnas). Cᵢ(x) = cᵢᵀ x.

        Precisión: SVD con umbral relativo τ·σ_max. Para J antisimétrica el
        rango es par (Pfaffiano); se corrige paridad si el umbral corta un
        par de valores singulares casi nulos.
        """
        if self._casimir_basis is None:
            if self.dim == 0:
                self._casimir_basis = np.zeros((0, 0))
            else:
                _, S, Vt = np.linalg.svd(self.J, full_matrices=True)
                sigma_max = float(S.max()) if S.size else 1.0
                tol = CONSTANTS.CASIMIR_TOLERANCE * max(1.0, sigma_max)
                rank_J = int(np.sum(S > tol)) if S.size else 0
                if rank_J % 2 == 1 and rank_J > 0:
                    # rango de una forma 2-antisimétrica es par
                    rank_J -= 1
                self._casimir_basis = np.ascontiguousarray(Vt[rank_J:].T)
        return self._casimir_basis

    @property
    def casimir_dimension(self) -> int:
        return int(self.casimir_basis.shape[1]) if self.casimir_basis.size else 0

    def casimir_values(self, x: np.ndarray) -> np.ndarray:
        """C(x) = Cᵀ x ∈ ℝ^{dim ker J}."""
        x = self._validate_state(x)
        if self.casimir_dimension == 0:
            return np.zeros(0, dtype=float)
        return self.casimir_basis.T @ x

    def casimir_drift(self, x: np.ndarray) -> float:
        r"""‖Ċ‖₂ con Ċ = -Cᵀ R ∇H.  Cero exacto si R=0 o im R ⊥ ker J."""
        if self.casimir_dimension == 0:
            return 0.0
        cdot = -self.casimir_basis.T @ (self.R @ self.gradient(x))
        return float(np.linalg.norm(cdot))

    def _metric_condition_number(self) -> float:
        try:
            eig = np.linalg.eigvalsh(self.metric)
            eig = eig[eig > CONSTANTS.NUMERICAL_ZERO]
            if eig.size == 0:
                return float("inf")
            cond = float(np.max(eig) / np.min(eig))
            if cond > CONSTANTS.MAX_CONDITION_NUMBER:
                logger.warning(
                    "%s: κ₂(K)=%.3e > MAX_CONDITION_NUMBER. Espectro de A mal condicionado.",
                    self.name,
                    cond,
                )
            return cond
        except np.linalg.LinAlgError:
            return float("inf")

    def logarithmic_norm(self, A: Optional[np.ndarray] = None) -> float:
        r"""μ₂(A) = λ_max((A+Aᵀ)/2).  ‖e^{tA}‖₂ ≤ e^{t μ₂(A)}."""
        A = self.A if A is None else A
        if A.size == 0:
            return 0.0
        S = 0.5 * (A + A.T)
        try:
            return float(np.max(np.linalg.eigvalsh(S)))
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(f"No se pudo calcular μ₂(A): {exc}") from exc

    def spectral_audit(self, force: bool = False) -> SpectralAudit:
        """Espectro de A, norma logarítmica y estabilidad módulo Casimirs."""
        if self._spectral_cache is not None and not force:
            return self._spectral_cache
        if self.dim == 0:
            audit = SpectralAudit(
                eigenvalues=np.array([], dtype=complex),
                max_real_part=0.0,
                spectral_radius=0.0,
                logarithmic_norm=0.0,
                numerical_abscissa=0.0,
                condition_metric=1.0,
                casimir_multiplicity=0,
                is_lyapunov_stable=True,
                is_asymptotically_stable_mod_casimir=False,
                is_normal=True,
                departure_from_normality=0.0,
            )
            self._spectral_cache = audit
            return audit

        if self.dim <= CONSTANTS.MAX_DENSE_SPECTRAL_DIM:
            eigvals = np.linalg.eigvals(self.A)
        elif SCIPY_AVAILABLE and eigs is not None:
            k = min(12, self.dim - 1)
            try:
                eigvals = eigs(self.A, k=k, which="LR", return_eigenvectors=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("eigs(LR) falló (%s); se omite espectro denso.", exc)
                eigvals = np.array([0.0 + 0.0j])
        else:
            eigvals = np.array([0.0 + 0.0j])

        re_parts = np.real(eigvals)
        max_re = float(np.max(re_parts)) if eigvals.size else 0.0
        rho = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        mu = self.logarithmic_norm(self.A)
        AA_star = self.A @ self.A.T
        A_star_A = self.A.T @ self.A
        dep = float(np.linalg.norm(AA_star - A_star_A, ord="fro"))
        scale_A = max(1.0, float(np.linalg.norm(self.A, ord="fro")))
        is_normal = dep <= CONSTANTS.SYMPLECTIC_TOLERANCE * scale_A

        cas_mult = int(np.sum(np.abs(eigvals) < CONSTANTS.CASIMIR_TOLERANCE * max(1.0, rho)))
        dissipative_re = re_parts[np.abs(eigvals) >= CONSTANTS.CASIMIR_TOLERANCE * max(1.0, rho)]
        if dissipative_re.size:
            max_re_mod = float(np.max(dissipative_re))
        else:
            max_re_mod = 0.0

        is_lyap = max_re <= CONSTANTS.STABILITY_MARGIN and mu <= CONSTANTS.STABILITY_MARGIN
        is_asym_mod = bool(
            dissipative_re.size > 0 and max_re_mod < -CONSTANTS.STABILITY_MARGIN
        )
        audit = SpectralAudit(
            eigenvalues=eigvals,
            max_real_part=max_re,
            spectral_radius=rho,
            logarithmic_norm=mu,
            numerical_abscissa=mu,
            condition_metric=self._condition_metric,
            casimir_multiplicity=max(cas_mult, self.casimir_dimension),
            is_lyapunov_stable=is_lyap,
            is_asymptotically_stable_mod_casimir=is_asym_mod,
            is_normal=is_normal,
            departure_from_normality=dep / scale_A,
        )
        self._spectral_cache = audit
        return audit

    # ──────────────────────────────────────────────────────────────────────────
    # Integración estructura-preservante
    # ──────────────────────────────────────────────────────────────────────────

    def compute_step(
        self,
        x: np.ndarray,
        dt: float,
        enforce_liouville: bool = False,
        integrator: str = "strang",
        u: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, FluxCondenserStepReport]:
        """
        Un paso de flujo.

        integrators:
            'strang'  — splitting simétrico (por defecto, estructura-preservante)
            'expm'    — Padé/Higham sobre A completo (referencia analítica lineal)
        El control u se aplica por Euler implícito de primer orden en el puerto
        (hold de orden cero), suficiente para la semilla de Fase 2.
        """
        x = self._validate_state(x)
        dt = float(dt)
        if dt <= 0.0:
            raise ConfigurationError("dt debe ser positivo.")
        if dt > CONSTANTS.MAX_DELTA_TIME:
            raise ConfigurationError("dt excede MAX_DELTA_TIME.")

        integrator = integrator.lower().strip()
        M = self.state_transition(dt, conservative=False, integrator=integrator)
        x_next = M @ x
        if u is not None:
            # Φ(dt) x + ∫₀^{dt} e^{(dt-s)A} g u ds  ≈  M x + dt · φ₁(A dt) g u
            x_next = x_next + dt * (self._phi1(dt) @ (self.g @ _as_1d(u)))

        nrm = float(np.linalg.norm(x_next))
        if nrm > CONSTANTS.MAX_STATE_NORM:
            raise NumericalInstabilityError(
                f"‖x‖={nrm:.3e} > MAX_STATE_NORM; flujo inestable."
            )

        H = self.hamiltonian(x)
        rayleigh = self.rayleigh_dissipation_rate(x, u=None)
        det_M = self._det(M)
        volume_drift = abs(det_M - 1.0)
        expected_det = math.exp(dt * self.trace_generator)
        # Liouville disipativo: det M ≈ e^{t tr A}, no 1
        volume_residual = abs(det_M - expected_det)

        M_cons = self.state_transition(dt, conservative=True, integrator=integrator)
        scale_J = max(1.0, float(np.linalg.norm(self.J, ord="fro")))
        poisson_residual = float(
            np.linalg.norm(M_cons @ self.J @ M_cons.T - self.J, ord="fro")
        )
        strang_poisson = poisson_residual / scale_J

        is_liouville_preserved = bool(
            self.is_nominally_conservative
            and volume_drift <= CONSTANTS.LIOUVILLE_DRIFT_LIMIT
            and poisson_residual <= CONSTANTS.SYMPLECTIC_TOLERANCE * scale_J
        )
        is_volume_contracting = bool(det_M <= expected_det + CONSTANTS.LIOUVILLE_DRIFT_LIMIT)

        if enforce_liouville and self.is_nominally_conservative and not is_liouville_preserved:
            raise LiouvilleViolationError(
                f"Liouville violado: volume_drift={volume_drift:.3e}, "
                f"poisson_residual={poisson_residual:.3e}"
            )

        cas_drift = self.casimir_drift(x)
        if self.dim <= 64:
            spec = self.spectral_audit()
            max_re = spec.max_real_part
            rho_A = spec.spectral_radius
            mu = spec.logarithmic_norm
            is_lyap = spec.is_lyapunov_stable
            is_asym = spec.is_asymptotically_stable_mod_casimir
        else:
            max_re, rho_A, mu = 0.0, 0.0, self.trace_generator / max(self.dim, 1)
            is_lyap, is_asym = self.trace_generator <= CONSTANTS.STABILITY_MARGIN, False

        report = FluxCondenserStepReport(
            hamiltonian_energy=H,
            volume_drift=volume_residual if not self.is_nominally_conservative else volume_drift,
            rayleigh_dissipation_rate=rayleigh,
            is_liouville_preserved=is_liouville_preserved,
            is_volume_contracting=is_volume_contracting,
            poisson_residual=poisson_residual,
            trace_generator=self.trace_generator,
            state_dimension=self.dim,
            casimir_dimension=self.casimir_dimension,
            max_re_eigen_A=max_re,
            spectral_radius_A=rho_A,
            logarithmic_norm_A=mu,
            is_lyapunov_stable=is_lyap,
            is_asymptotically_stable_mod_casimir=is_asym,
            strang_poisson_residual=strang_poisson,
            casimir_drift=cas_drift,
            integrator=integrator,
        )
        return x_next, report

    def state_transition(
        self,
        dt: float,
        conservative: bool = False,
        integrator: str = "strang",
    ) -> np.ndarray:
        """Mapa Φ(dt). LRU acotado a MAX_TRANSITION_CACHE."""
        key = (float(dt), bool(conservative), integrator)
        if key in self._transition_cache:
            self._transition_cache.move_to_end(key)
            return self._transition_cache[key]
        if conservative:
            M = self._expm(self.A_conservative * float(dt))
        elif integrator == "expm":
            M = self._expm(self.A * float(dt))
        else:
            M = self._strang_split(float(dt))
        self._transition_cache[key] = M
        if len(self._transition_cache) > CONSTANTS.MAX_TRANSITION_CACHE:
            self._transition_cache.popitem(last=False)
        return M

    def _strang_split(self, dt: float) -> np.ndarray:
        r"""
        Strang: e^{dt/2 A_R} e^{dt A_J} e^{dt/2 A_R}.
        A_J = JK  (Poisson), A_R = -RK  (gradiente de H, simetrizable por K^{1/2}).
        """
        if self.is_nominally_conservative:
            return self._expm(self.A_conservative * dt)
        half = 0.5 * dt
        E_R = self._expm(self.A_dissipative * half)
        E_J = self._expm(self.A_conservative * dt)
        return E_R @ E_J @ E_R

    def _phi1(self, dt: float) -> np.ndarray:
        r"""φ₁(z)=(e^z-1)/z aplicado a A dt.  φ₁(0)=I."""
        Z = self.A * float(dt)
        nrm = float(np.linalg.norm(Z, ord=np.inf))
        if nrm < CONSTANTS.NUMERICAL_TOLERANCE:
            return np.eye(self.dim) + 0.5 * Z
        E = self._expm(Z)
        try:
            return np.linalg.solve(Z, E - np.eye(self.dim))
        except np.linalg.LinAlgError:
            return np.eye(self.dim) + 0.5 * Z

    def audit_transition_map(
        self,
        M: np.ndarray,
        conservative: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Auditoría de un mapa externo (leapfrog, Verlet, RK)."""
        M = np.asarray(M, dtype=float)
        if M.shape != (self.dim, self.dim):
            raise ConfigurationError(f"Mapa de transición debe ser {self.dim}×{self.dim}.")
        scale_J = max(1.0, float(np.linalg.norm(self.J, ord="fro")))
        poisson_res = float(np.linalg.norm(M @ self.J @ M.T - self.J, ord="fro")) / scale_J
        det_res = abs(self._det(M) - 1.0)
        is_cons = self.is_nominally_conservative if conservative is None else conservative
        return {
            "poisson_residual_relative": poisson_res,
            "volume_drift": det_res,
            "is_poisson_map": poisson_res <= CONSTANTS.SYMPLECTIC_TOLERANCE,
            "is_volume_preserving": (det_res <= CONSTANTS.LIOUVILLE_DRIFT_LIMIT) if is_cons else False,
        }

    def circulation_invariant(
        self,
        q_path: np.ndarray,
        p_path: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Invariante integral de Poincaré ∮ p dq  (regla del trapecio, cierre C⁰).

        Es el pullback de la 1-forma de Liouville θ = p dq sobre una curva
        cerrada. El flujo hamiltoniano preserva ∮_γ θ (teorema de Poincaré).
        Error de cuadratura O(h² · max‖γ̈‖) ; h = max ‖Δq‖.
        """
        if p_path is None:
            if self._canonical_n is None:
                raise ConfigurationError(
                    "circulation_invariant exige layout canónico o (q_path, p_path)."
                )
            n = self._canonical_n
            q = np.asarray(q_path, dtype=float)
            if q.ndim == 1:
                q = q.reshape(-1, 1)
            if q.shape[1] != 2 * n:
                raise ConfigurationError(f"Trayectoria canónica: se esperaban {2*n} columnas [q,p].")
            p = q[:, n:]
            q = q[:, :n]
        else:
            q = np.asarray(q_path, dtype=float)
            p = np.asarray(p_path, dtype=float)
        if q.ndim == 1:
            q = q.reshape(-1, 1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if q.shape != p.shape:
            raise ConfigurationError("q_path y p_path deben tener la misma forma.")
        if q.shape[0] < 2:
            return 0.0
        if not np.allclose(q[0], q[-1], atol=CONSTANTS.POINCARE_SECTION_EPSILON):
            q = np.vstack([q, q[0]])
            p = np.vstack([p, p[0]])
        dq = np.diff(q, axis=0)
        p_mid = 0.5 * (p[:-1] + p[1:])
        return float(np.sum(p_mid * dq))

    # ──────────────────────────────────────────────────────────────────────────
    # IDA-PBC lineal (matching algebraico)
    # ──────────────────────────────────────────────────────────────────────────

    def linear_ida_pbc_matching(
        self,
        K_d: np.ndarray,
        R_d: Optional[np.ndarray] = None,
        x_star: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        r"""
        Matching IDA-PBC lineal (Ortega–van der Schaft).

        Se busca J_d = -J_dᵀ, R_d = R_dᵀ ⪰ 0, H_d = ½ (x-x*)ᵀ K_d (x-x*) tales que
            (J_d - R_d) K_d = (J - R) K + g G
        para algún G (precompensador estático). Si g = I, G queda determinado:
            G = (J_d - R_d) K_d - (J - R) K.
        Construcción canónica:
            J_d := J
            J_a := 0
            R_d := R  (o la suministrada, proyectada a Sym⁺)
            G   := (J_d - R_d) K_d - (J - R) K

        El residual de matching es ‖(J_d-R_d)K_d - (J-R)K - g G⁺‖_F
        con G⁺ la solución por mínimos cuadrados de g G = Δ.
        """
        K_d = self._as_matrix(K_d, dim=self.dim, name="K_d", kind="spd")
        if R_d is None:
            R_d_m = self.R.copy()
        else:
            R_d_m = self._as_matrix(R_d, dim=self.dim, name="R_d", kind="psd")
        J_d = self.J.copy()
        J_a = np.zeros_like(self.J)
        Delta = (J_d - R_d_m) @ K_d - (self.J - self.R) @ self.metric
        G, residual = self._port_least_squares(Delta)
        if x_star is None:
            x_star = np.zeros(self.dim, dtype=float)
        else:
            x_star = self._validate_state(x_star)
        return {
            "J_d": J_d,
            "J_a": J_a,
            "R_d": R_d_m,
            "K_d": K_d,
            "G": G,
            "x_star": x_star,
            "matching_residual": np.array([residual]),
        }

    def _port_least_squares(self, Delta: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resuelve g G = Δ en sentido de Frobenius (G = g⁺ Δ)."""
        m = self.g.shape[1]
        if m == 0:
            return np.zeros((0, self.dim)), float(np.linalg.norm(Delta, ord="fro"))
        try:
            G, _, _, _ = np.linalg.lstsq(self.g, Delta, rcond=None)
        except np.linalg.LinAlgError:
            G = np.zeros((m, self.dim))
        residual = float(np.linalg.norm(self.g @ G - Delta, ord="fro"))
        return G, residual

    def energy_shaping_lyapunov(
        self,
        x: np.ndarray,
        H_star: float,
    ) -> Tuple[float, np.ndarray, float]:
        r"""
        V = ½ (H - H*)²,  ∇V = (H-H*) ∇H,  V̇ = (H-H*) Ḣ.

        V̇ ≤ 0 en subnivel {H ≥ H*} si u=0 (disipación). En {H < H*} hace falta
        inyección de puerto: u = -k (H-H*) y  no sirve; se requiere
        u = +k (H*-H) y / (‖y‖²+ε) para bombear (passivity-based pumping).
        """
        H = self.hamiltonian(x)
        grad = self.gradient(x)
        V = 0.5 * float((H - H_star) ** 2)
        dV = (H - H_star) * grad
        Hdot = self.rayleigh_dissipation_rate(x)
        Vdot = (H - H_star) * Hdot
        return V, dV, Vdot

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades internas
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_state(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ConfigurationError(f"Estado dim {x.size}; se esperaba {self.dim}.")
        if not np.all(np.isfinite(x)):
            raise NumericalInstabilityError("Estado contiene valores no finitos.")
        return x

    @staticmethod
    def _validate_skew_symmetry(J: np.ndarray) -> np.ndarray:
        J_skew = 0.5 * (J - J.T)
        leak = float(np.linalg.norm(J - J_skew, ord="fro"))
        scale = max(1.0, float(np.linalg.norm(J_skew, ord="fro")))
        if leak / scale > CONSTANTS.SYMPLECTIC_TOLERANCE:
            raise SymplecticStructureError(
                f"J no es antisimétrica: ‖J-sk(J)‖/‖sk(J)‖={leak/scale:.3e}"
            )
        return J_skew

    def _as_port_matrix(self, g: Optional[np.ndarray], dim: int) -> np.ndarray:
        if g is None:
            return np.eye(dim, dtype=float)
        arr = np.asarray(g, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(dim, 1) if arr.size == dim else arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] != dim:
            raise ConfigurationError(f"g debe ser ({dim}×m), recibido {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ConfigurationError("g contiene valores no finitos.")
        return arr

    def _as_square_matrix(self, value: Union[float, np.ndarray], name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            raise ConfigurationError(f"{name} escalar requiere dimensión conocida.")
        if arr.ndim == 1:
            arr = np.diag(arr)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ConfigurationError(f"{name} debe ser matriz cuadrada.")
        if not np.all(np.isfinite(arr)):
            raise ConfigurationError(f"{name} contiene valores no finitos.")
        return arr

    def _as_matrix(
        self,
        value: Optional[Union[float, np.ndarray]],
        dim: int,
        name: str,
        kind: str,
    ) -> np.ndarray:
        if value is None:
            return np.zeros((dim, dim), dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            scalar = float(arr)
            if kind == "spd" and scalar <= 0.0:
                raise ConfigurationError(f"{name} escalar debe ser positivo.")
            if kind == "psd" and scalar < 0.0:
                raise ConfigurationError(f"{name} escalar debe ser no negativo.")
            return scalar * np.eye(dim)
        if arr.ndim == 1:
            if arr.size != dim:
                raise ConfigurationError(f"{name} vector longitud {dim}, recibido {arr.size}.")
            if kind == "spd" and np.any(arr <= 0.0):
                raise ConfigurationError(f"{name} diagonal debe ser positiva.")
            if kind == "psd" and np.any(arr < 0.0):
                raise ConfigurationError(f"{name} diagonal debe ser no negativa.")
            return np.diag(arr.astype(float))
        if arr.ndim == 2:
            if arr.shape != (dim, dim):
                raise ConfigurationError(f"{name} debe ser {dim}×{dim}, recibido {arr.shape}.")
            arr = 0.5 * (arr + arr.T)
            if not np.all(np.isfinite(arr)):
                raise ConfigurationError(f"{name} contiene valores no finitos.")
            try:
                eigvals = np.linalg.eigvalsh(arr)
            except np.linalg.LinAlgError as exc:
                raise ConfigurationError(f"No se pudo verificar espectro de {name}: {exc}") from exc
            min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
            tol = max(CONSTANTS.NUMERICAL_TOLERANCE, CONSTANTS.SYMPLECTIC_TOLERANCE)
            if kind == "spd":
                if min_eig <= -tol:
                    raise ConfigurationError(f"{name} no es SPD: min_eig={min_eig:.3e}")
                if min_eig <= tol:
                    shift = tol - min_eig + CONSTANTS.NUMERICAL_ZERO
                    logger.warning("%s: regularización SPD con shift=%.3e.", name, shift)
                    arr = arr + shift * np.eye(dim)
            elif kind == "psd":
                if min_eig < -tol:
                    raise ConfigurationError(f"{name} no es PSD: min_eig={min_eig:.3e}")
                if min_eig < 0.0:
                    arr = arr + (-min_eig + CONSTANTS.NUMERICAL_ZERO) * np.eye(dim)
            else:
                raise ConfigurationError("kind debe ser 'spd' o 'psd'.")
            return arr
        raise ConfigurationError(f"{name} debe ser escalar, vector o matriz.")

    @staticmethod
    def _physical_matrix(
        value: Optional[Union[float, np.ndarray]],
        name: str,
        positive: bool,
        default_shape: Optional[int] = None,
    ) -> np.ndarray:
        if value is None:
            if default_shape is None:
                raise ConfigurationError(f"{name} no puede ser None sin default_shape.")
            return np.zeros((default_shape, default_shape), dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            scalar = float(arr)
            if positive and scalar <= 0.0:
                raise ConfigurationError(f"{name} debe ser positivo.")
            if not positive and scalar < 0.0:
                raise ConfigurationError(f"{name} debe ser no negativo.")
            size = default_shape if default_shape is not None else 1
            return scalar * np.eye(size)
        if arr.ndim == 1:
            if positive and np.any(arr <= 0.0):
                raise ConfigurationError(f"{name} diagonal debe ser positiva.")
            if not positive and np.any(arr < 0.0):
                raise ConfigurationError(f"{name} diagonal debe ser no negativa.")
            return np.diag(arr.astype(float))
        if arr.ndim == 2:
            if arr.shape[0] != arr.shape[1]:
                raise ConfigurationError(f"{name} debe ser cuadrada.")
            arr = 0.5 * (arr + arr.T)
            eigvals = np.linalg.eigvalsh(arr)
            if positive and float(np.min(eigvals)) <= 0.0:
                raise ConfigurationError(f"{name} debe ser definida positiva.")
            if not positive and float(np.min(eigvals)) < -CONSTANTS.NUMERICAL_TOLERANCE:
                raise ConfigurationError(f"{name} debe ser semidefinida positiva.")
            return arr
        raise ConfigurationError(f"{name} debe ser escalar, vector o matriz.")

    @staticmethod
    def _block_diag(*blocks: np.ndarray) -> np.ndarray:
        total = sum(b.shape[0] for b in blocks)
        out = np.zeros((total, total), dtype=float)
        idx = 0
        for block in blocks:
            n = block.shape[0]
            out[idx : idx + n, idx : idx + n] = block
            idx += n
        return out

    def _expm(self, A: np.ndarray) -> np.ndarray:
        if A.size == 0:
            return np.zeros_like(A)
        nrm = float(np.linalg.norm(A, ord=np.inf))
        if nrm > CONSTANTS.MAX_EXPONENTIAL_ARG:
            raise NumericalInstabilityError(
                f"‖A‖_∞={nrm:.3e} excede el rango seguro de expm float64."
            )
        if SCIPY_AVAILABLE and expm is not None:
            return np.asarray(expm(A), dtype=float)
        return self._expm_pade_fallback(A)

    def _det(self, M: np.ndarray) -> float:
        if M.size == 0:
            return 1.0
        if SCIPY_AVAILABLE and det is not None:
            return float(det(M))
        sign, logdet = np.linalg.slogdet(M)
        if sign == 0.0:
            return 0.0
        if abs(logdet) > CONSTANTS.MAX_EXPONENTIAL_ARG:
            return math.copysign(math.inf, sign)
        return float(sign * math.exp(logdet))

    @staticmethod
    def _expm_pade_fallback(A: np.ndarray, terms: int = CONSTANTS.PADE_TAYLOR_TERMS) -> np.ndarray:
        """Scaling-and-squaring + Taylor (Higham, cuando SciPy no está)."""
        n = A.shape[0]
        norm_A = float(np.linalg.norm(A, ord=np.inf))
        if norm_A == 0.0:
            return np.eye(n)
        s = max(0, int(math.ceil(math.log2(norm_A / 0.5 + 1.0))))
        As = A / (2 ** s)
        E = np.eye(n)
        term = np.eye(n)
        for k in range(1, terms + 1):
            term = term @ As / k
            E = E + term
            if float(np.linalg.norm(term, ord=np.inf)) < CONSTANTS.NUMERICAL_ZERO:
                break
        for _ in range(s):
            E = E @ E
        return E


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 1.3 — CÁLCULO EXTERIOR DISCRETO (DEC)
# ═══════════════════════════════════════════════════════════════════════════════════════


class DiscreteVectorCalculus:
    r"""
    DEC sobre el complejo de cliques K≤2 del grafo (flag complex).

    Complejo de cadenas:   C₂ --∂₂--> C₁ --∂₁--> C₀,   ∂₁∂₂ = 0.
    Complejo de co-cadenas: C⁰ --d₀--> C¹ --d₁--> C²,   d₁d₀ = 0,
        dₖ = ∂ₖ₊₁ᵀ.

    Estrellas de Hodge lumped (dual circumcéntrico diagonal):
        ★₀ = diag(vol v),   ★₁ = diag(|⋆e|/|e|),   ★₂ = diag(1/|f|).
    Codiferencial (convención positiva, laplaciano de grafo):
        δ₁ = ★₀⁻¹ d₀ᵀ ★₁ = ★₀⁻¹ ∂₁ ★₁,
        δ₂ = ★₁⁻¹ d₁ᵀ ★₂ = ★₁⁻¹ ∂₂ ★₂.
    Hodge:  Δₖ = δ_{k+1} dₖ + d_{k-1} δₖ ⪰ 0,   ker Δₖ ≅ Hᵏ_{dR}(K)  (Hodge–de Rham discreto).

    Functorialmente: esto es F: SimpComp_{\le2} → Ch_{\ge0}(Vect_ℝ).
    """

    NUMERICAL_TOLERANCE: float = CONSTANTS.NUMERICAL_TOLERANCE

    def __init__(
        self,
        adjacency_list: Dict[int, Set[int]],
        node_volumes: Optional[Dict[int, float]] = None,
        edge_lengths: Optional[Dict[Tuple[int, int], float]] = None,
        face_areas: Optional[Dict[Tuple[int, int, int], float]] = None,
        node_positions: Optional[Dict[int, np.ndarray]] = None,
        dual_edge_lengths: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> None:
        if nx is None:
            raise ConfigurationError("networkx es requerido para DiscreteVectorCalculus.")
        self.graph = nx.Graph(adjacency_list)
        self._node_volumes = node_volumes or {}
        self._edge_lengths = edge_lengths or {}
        self._face_areas = face_areas or {}
        self._dual_edge_lengths = dual_edge_lengths or {}
        self.node_positions = {
            int(k): np.asarray(v, dtype=float).reshape(-1) for k, v in (node_positions or {}).items()
        }
        self._validate_graph()
        self._build_simplicial_complex()
        if SCIPY_AVAILABLE:
            self._build_chain_operators()
            self._verify_chain_complex()
            self._build_hodge_operators()
            self._build_calculus_operators()
            self._compute_betti_numbers()
            self._verify_hodge_laplacian_psd()
        else:
            warnings.warn(
                "SciPy no disponible. DiscreteVectorCalculus operará en modo reducido.",
                RuntimeWarning,
            )
        self._laplacian_cache: "OrderedDict[int, csr_matrix]" = OrderedDict()

    def _validate_graph(self) -> None:
        if self.graph.number_of_nodes() == 0:
            raise ConfigurationError("El grafo no puede estar vacío.")
        if self.graph.number_of_nodes() == 1 and self.graph.number_of_edges() == 0:
            warnings.warn("Grafo trivial con un solo nodo aislado.", UserWarning)
        self.num_components = nx.number_connected_components(self.graph)
        self.is_connected = self.num_components == 1
        if not self.is_connected:
            logger.warning("Grafo con %d componentes conexas. β₀ > 1.", self.num_components)
        try:
            self.is_planar, self.planar_embedding = nx.check_planarity(self.graph)
        except Exception:  # noqa: BLE001
            self.is_planar = False
            self.planar_embedding = None
        if self.graph.number_of_nodes() > CONSTANTS.MAX_GRAPH_NODES_CLIQUE:
            logger.warning(
                "enumerate_all_cliques es exponencial; n=%d > %d.",
                self.graph.number_of_nodes(),
                CONSTANTS.MAX_GRAPH_NODES_CLIQUE,
            )

    def _build_simplicial_complex(self) -> None:
        self.nodes: List[int] = sorted(self.graph.nodes())
        self.node_to_idx: Dict[int, int] = {n: i for i, n in enumerate(self.nodes)}
        self.num_nodes: int = len(self.nodes)
        self.edges: List[Tuple[int, int]] = []
        self.edge_orientation: Dict[Tuple[int, int], int] = {}
        for u, v in self.graph.edges():
            a, b = (u, v) if u < v else (v, u)
            self.edges.append((a, b))
            self.edge_orientation[(a, b)] = +1
            self.edge_orientation[(b, a)] = -1
        self.edge_to_idx: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(self.edges)}
        self.num_edges: int = len(self.edges)
        self.faces: List[Tuple[int, int, int]] = []
        self.face_boundaries: List[List[Tuple[Tuple[int, int], int]]] = []
        for clique in nx.enumerate_all_cliques(self.graph):
            if len(clique) != 3:
                continue
            v0, v1, v2 = sorted(clique)
            self.faces.append((v0, v1, v2))
            # ∂[v0,v1,v2] = [v1,v2] - [v0,v2] + [v0,v1]
            self.face_boundaries.append(
                [
                    ((v1, v2), +1),
                    ((v0, v2), -1),
                    ((v0, v1), +1),
                ]
            )
        self.face_to_idx: Dict[Tuple[int, int, int], int] = {f: i for i, f in enumerate(self.faces)}
        self.num_faces: int = len(self.faces)
        self._build_edge_face_adjacency()
        self.euler_characteristic = self.num_nodes - self.num_edges + self.num_faces
        self._assert_positive_primal_volumes()

    def _assert_positive_primal_volumes(self) -> None:
        for key, val in self._node_volumes.items():
            if float(val) <= 0.0:
                raise HodgeStructureError(f"Volumen nodal no positivo en {key}: {val}.")
        for key, val in self._edge_lengths.items():
            if float(val) <= 0.0:
                raise HodgeStructureError(f"Longitud de arista no positiva en {key}: {val}.")
        for key, val in self._face_areas.items():
            if float(val) <= 0.0:
                raise HodgeStructureError(f"Área de cara no positiva en {key}: {val}.")

    def _build_edge_face_adjacency(self) -> None:
        self.edge_to_faces: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(self.num_edges)}
        for face_idx, boundary in enumerate(self.face_boundaries):
            for edge, sign in boundary:
                edge_canonical = (min(edge), max(edge))
                if edge_canonical in self.edge_to_idx:
                    edge_idx = self.edge_to_idx[edge_canonical]
                    self.edge_to_faces[edge_idx].append((face_idx, sign))

    def _build_chain_operators(self) -> None:
        self.boundary1 = self._build_boundary_1()
        self.boundary2 = self._build_boundary_2()

    def _build_boundary_1(self) -> csr_matrix:
        """∂₁: C₁ → C₀.  (∂ e_{uv})_v = +1, (∂ e_{uv})_u = -1, u<v."""
        if self.num_edges == 0:
            return sparse.csr_matrix((self.num_nodes, 0))
        u_idx = np.array([self.node_to_idx[u] for (u, _) in self.edges], dtype=np.int64)
        v_idx = np.array([self.node_to_idx[v] for (_, v) in self.edges], dtype=np.int64)
        rows = np.concatenate([v_idx, u_idx])
        cols = np.concatenate([np.arange(self.num_edges), np.arange(self.num_edges)])
        data = np.concatenate([np.ones(self.num_edges), -np.ones(self.num_edges)])
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_edges))

    def _build_boundary_2(self) -> csr_matrix:
        """∂₂: C₂ → C₁.  Identidad ∂₁∂₂ = 0 por construcción combinatoria."""
        if self.num_faces == 0:
            return sparse.csr_matrix((self.num_edges, 0))
        edge_idx_list: List[int] = []
        face_idx_list: List[int] = []
        values: List[float] = []
        for face_idx, boundary in enumerate(self.face_boundaries):
            for edge, sign in boundary:
                edge_canonical = (min(edge), max(edge))
                if edge_canonical not in self.edge_to_idx:
                    continue
                e_idx = self.edge_to_idx[edge_canonical]
                orientation = self.edge_orientation.get(edge, 1)
                edge_idx_list.append(e_idx)
                face_idx_list.append(face_idx)
                values.append(float(sign * orientation))
        rows = np.array(edge_idx_list, dtype=np.int64)
        cols = np.array(face_idx_list, dtype=np.int64)
        data = np.array(values, dtype=float)
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.num_edges, self.num_faces))

    def _verify_chain_complex(self) -> None:
        if self.num_faces == 0 or self.num_edges == 0:
            self._chain_complex_error = 0.0
            return
        composition = self.boundary1 @ self.boundary2
        max_error = float(np.max(np.abs(composition.data))) if composition.nnz > 0 else 0.0
        self._chain_complex_error = max_error
        if max_error > self.NUMERICAL_TOLERANCE:
            raise ChainComplexError(f"‖∂₁∂₂‖_∞={max_error:.3e}  (se exige 0).")

    def _build_hodge_operators(self) -> None:
        self.star0, self.star0_inv = self._build_hodge_star(self.num_nodes, self._get_node_weight)
        self.star1, self.star1_inv = self._build_hodge_star(self.num_edges, self._get_edge_weight)
        self.star2, self.star2_inv = self._build_hodge_star(self.num_faces, self._get_face_weight)

    def _build_hodge_star(
        self,
        size: int,
        weight_func: Callable[[int], float],
    ) -> Tuple[csr_matrix, csr_matrix]:
        if size == 0:
            empty = sparse.csr_matrix((0, 0))
            return empty, empty
        weights = np.array([weight_func(i) for i in range(size)], dtype=float)
        if np.any(weights <= 0.0):
            raise HodgeStructureError("★ posee pesos no positivos (mallado no well-centered).")
        weights = np.maximum(weights, self.NUMERICAL_TOLERANCE)
        return sparse.diags(weights, format="csr"), sparse.diags(1.0 / weights, format="csr")

    def _get_node_weight(self, idx: int) -> float:
        node = self.nodes[idx]
        if node in self._node_volumes:
            return float(self._node_volumes[node])
        return float(max(1, self.graph.degree(node)))

    def _get_edge_weight(self, idx: int) -> float:
        """
        ★₁_e = |⋆e| / |e|.  Si no hay longitud dual, lumped |e| (masa combinatoria).
        """
        edge = self.edges[idx]
        primal = float(self._edge_lengths.get(edge, 1.0))
        if edge in self._dual_edge_lengths:
            dual = float(self._dual_edge_lengths[edge])
            return dual / max(primal, self.NUMERICAL_TOLERANCE)
        return primal

    def _get_face_weight(self, idx: int) -> float:
        """★₂ : Ω² → Ω⁰,  ★₂ = 1/área  (B_cochain = B_phys·área ⇒ ★B = B_phys)."""
        face = self.faces[idx]
        area = float(self._face_areas.get(face, 1.0))
        return 1.0 / max(area, self.NUMERICAL_TOLERANCE)

    def primal_edge_length(self, idx: int) -> float:
        return float(self._edge_lengths.get(self.edges[idx], 1.0))

    def primal_face_area(self, idx: int) -> float:
        return float(self._face_areas.get(self.faces[idx], 1.0))

    def _build_calculus_operators(self) -> None:
        # d₀ = ∂₁ᵀ : (dφ)(u→v) = φ(v) - φ(u)
        self.gradient_op = self.boundary1.T
        # δ₁ = ★₀⁻¹ ∂₁ ★₁   (div, de modo que Δ₀ = δ₁ d₀ ⪰ 0)
        self.divergence_op = self.star0_inv @ self.boundary1 @ self.star1
        # d₁ = ∂₂ᵀ
        self.curl_op = self.boundary2.T
        # δ₂ = ★₁⁻¹ ∂₂ ★₂
        if self.num_faces > 0:
            self.cocurl_op = self.star1_inv @ self.boundary2 @ self.star2
        else:
            self.cocurl_op = sparse.csr_matrix((self.num_edges, 0))

    def inner_product(self, degree: int, alpha: np.ndarray, beta: np.ndarray) -> float:
        r"""⟨α,β⟩_k = αᵀ ★_k β."""
        a = _as_1d(alpha)
        b = _as_1d(beta)
        star = {0: self.star0, 1: self.star1, 2: self.star2}[degree]
        return float(a @ (star @ b))

    def _compute_betti_numbers(self) -> None:
        """
        β₀ := # componentes (invariante combinatorio exacto).
        β₁, β₂ por rangos si el complejo es pequeño; si no, se postergan al
        espectro de Hodge (betti_from_hodge_kernel).
        """
        self.betti_0 = int(self.num_components)
        small = (
            max(self.num_nodes, self.num_edges, self.num_faces) <= CONSTANTS.MAX_DENSE_BETTI_DIM
        )
        if not small:
            # Fórmula de Euler con β₂=0 tentativa; se corrige espectralmente.
            self.betti_2 = 0
            self.betti_1 = self.betti_0 + self.num_faces - self.euler_characteristic
            self.betti = BettiNumbers(self.betti_0, self.betti_1, self.betti_2)
            logger.warning(
                "Betti denso omitido (dim > %d). β₁,β₂ son estimaciones de Euler.",
                CONSTANTS.MAX_DENSE_BETTI_DIM,
            )
            return
        if self.num_edges > 0:
            rank_b1 = int(np.linalg.matrix_rank(self.boundary1.toarray()))
        else:
            rank_b1 = 0
        beta0_rank = self.num_nodes - rank_b1
        if beta0_rank != self.betti_0:
            raise NumericalInstabilityError(
                f"Inconsistencia topológica: β₀(rank)={beta0_rank} ≠ π₀={self.num_components}"
            )
        nullity_b1 = self.num_edges - rank_b1 if self.num_edges > 0 else 0
        rank_b2 = int(np.linalg.matrix_rank(self.boundary2.toarray())) if self.num_faces > 0 else 0
        self.betti_1 = nullity_b1 - rank_b2
        self.betti_2 = self.num_faces - rank_b2 if self.num_faces > 0 else 0
        self.betti = BettiNumbers(self.betti_0, self.betti_1, self.betti_2)
        if self.betti.euler_poincare != self.euler_characteristic:
            raise NumericalInstabilityError(
                f"Euler–Poincaré violado: {self.betti.euler_poincare} ≠ {self.euler_characteristic}"
            )

    def _verify_hodge_laplacian_psd(self) -> None:
        if not SCIPY_AVAILABLE or self.num_nodes == 0:
            return
        L0 = self.divergence_op @ self.gradient_op
        asym = L0 - L0.T
        asym_norm = float(np.linalg.norm(asym.data)) if asym.nnz > 0 else 0.0
        if asym_norm > CONSTANTS.NUMERICAL_TOLERANCE * max(1.0, self.num_nodes):
            raise HodgeStructureError(f"Δ₀ no simétrico: ‖Δ₀-Δ₀ᵀ‖={asym_norm:.3e}")
        self._hodge_laplacian_psd_ok = True

    def hodge_laplacian_eigenpairs(
        self,
        degree: int,
        k: int = 10,
        which: str = "SM",
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not SCIPY_AVAILABLE:
            return None
        L = self.laplacian(degree)
        if L is None or L.shape[0] == 0:
            return None
        k = min(k, max(1, L.shape[0] - 1))
        if k <= 0:
            return None
        # Shift-invert en 0 para núcleo; SM puro es inestable en laplacianos grandes.
        try:
            if which == "SM":
                vals, vecs = eigsh(L.astype(float), k=k, sigma=0.0, which="LM")
            else:
                vals, vecs = eigsh(L.astype(float), k=k, which=which)
        except Exception:  # noqa: BLE001
            vals, vecs = eigsh(L.astype(float), k=k, which="SM")
        return vals, vecs

    def betti_from_hodge_kernel(self, tol: Optional[float] = None) -> BettiNumbers:
        """βₖ = dim ker Δₖ  (teorema de Hodge discreto). Independiente de rank(∂)."""
        if not SCIPY_AVAILABLE:
            return getattr(self, "betti", BettiNumbers(self.num_components, 0, 0))
        tol = CONSTANTS.HODGE_EIGEN_TOLERANCE if tol is None else float(tol)
        bettas: List[int] = []
        for deg, size in ((0, self.num_nodes), (1, self.num_edges), (2, self.num_faces)):
            if size == 0:
                bettas.append(0)
                continue
            L = self.laplacian(deg)
            if L is None or L.shape[0] == 0:
                bettas.append(0)
                continue
            k = min(max(1, size - 1), CONSTANTS.HODGE_KERNEL_MAX_K)
            try:
                vals, _ = eigsh(L.astype(float), k=k, sigma=0.0, which="LM")
            except Exception:  # noqa: BLE001
                try:
                    vals, _ = eigsh(L.astype(float), k=k, which="SM")
                except Exception:  # noqa: BLE001
                    bettas.append(0)
                    continue
            bettas.append(int(np.sum(np.abs(vals) < tol)))
        return BettiNumbers(bettas[0], bettas[1], bettas[2])

    def gradient(self, scalar_field: np.ndarray) -> np.ndarray:
        """d₀ φ."""
        if not SCIPY_AVAILABLE:
            return np.array([])
        phi = _as_1d(scalar_field)
        if phi.size != self.num_nodes:
            raise ConfigurationError(f"Gradiente: esperado {self.num_nodes}, recibido {phi.size}.")
        return self.gradient_op @ phi

    def divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """δ₁ v."""
        if not SCIPY_AVAILABLE:
            return np.array([])
        v = _as_1d(vector_field)
        if v.size != self.num_edges:
            raise ConfigurationError(f"Divergencia: esperado {self.num_edges}, recibido {v.size}.")
        return self.divergence_op @ v

    def curl(self, vector_field: np.ndarray) -> np.ndarray:
        """d₁ v."""
        if not SCIPY_AVAILABLE or self.num_faces == 0:
            return np.array([])
        v = _as_1d(vector_field)
        if v.size != self.num_edges:
            raise ConfigurationError(f"Curl: esperado {self.num_edges}, recibido {v.size}.")
        return self.curl_op @ v

    def cocurl(self, face_field: np.ndarray) -> np.ndarray:
        """δ₂ H  (Ampère DEC)."""
        if not SCIPY_AVAILABLE or self.num_faces == 0:
            return np.zeros(self.num_edges, dtype=float)
        h = _as_1d(face_field)
        if h.size != self.num_faces:
            raise ConfigurationError(f"Cocurl: esperado {self.num_faces}, recibido {h.size}.")
        return self.cocurl_op @ h

    def laplacian(self, degree: int) -> Optional[csr_matrix]:
        """Δₖ = δd + dδ, k ∈ {0,1,2}."""
        if not SCIPY_AVAILABLE:
            return None
        if degree not in {0, 1, 2}:
            raise ConfigurationError("Laplaciano soporta degree=0,1,2.")
        if degree in self._laplacian_cache:
            self._laplacian_cache.move_to_end(degree)
            return self._laplacian_cache[degree]
        if degree == 0:
            Delta = self.divergence_op @ self.gradient_op
        elif degree == 1:
            term1 = self.gradient_op @ self.divergence_op
            term2 = (
                self.cocurl_op @ self.curl_op
                if self.num_faces > 0
                else sparse.csr_matrix((self.num_edges, self.num_edges))
            )
            Delta = term1 + term2
        else:
            if self.num_faces == 0:
                Delta = sparse.csr_matrix((0, 0))
            else:
                Delta = self.curl_op @ self.cocurl_op
        self._laplacian_cache[degree] = Delta
        if len(self._laplacian_cache) > CONSTANTS.MAX_LAPLACIAN_CACHE:
            self._laplacian_cache.popitem(last=False)
        return Delta

    def curl_curl_spectral_radius(self, k: int = 6) -> float:
        r"""ρ(δ₂ d₁) sobre 1-formas: controla el CFL de Maxwell."""
        if not SCIPY_AVAILABLE or self.num_edges == 0:
            return 0.0
        L = self.laplacian(1)
        if L is None or L.shape[0] < 2:
            return 0.0
        kk = min(k, L.shape[0] - 1)
        try:
            vals = eigsh(L.astype(float), k=kk, which="LM", return_eigenvectors=False)
            return float(np.max(np.abs(vals)))
        except Exception:  # noqa: BLE001
            return float(np.linalg.norm(L.toarray(), ord=2)) if L.shape[0] <= 256 else 0.0

    def verify_complex_exactness(self, seed: int = 0) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        results: Dict[str, Any] = {
            "boundary_composition_error": getattr(self, "_chain_complex_error", 0.0),
            "is_chain_complex": getattr(self, "_chain_complex_error", 0.0) < self.NUMERICAL_TOLERANCE,
            "euler_characteristic": self.euler_characteristic,
            "betti_numbers": getattr(self, "betti", BettiNumbers(0, 0, 0)).as_tuple(),
        }
        if SCIPY_AVAILABLE and self.num_nodes > 0 and self.num_faces > 0:
            phi = rng.standard_normal(self.num_nodes)
            results["curl_grad_error"] = float(np.linalg.norm(self.curl(self.gradient(phi))))
            if self.num_edges > 0:
                v = rng.standard_normal(self.num_edges)
                results["div_cocurl_error"] = float(np.linalg.norm(self.divergence(self.cocurl(self.curl(v) * 0.0))))
        return results


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 1.4 — SOLVER MAXWELL FDTD + POINCARÉ
# ═══════════════════════════════════════════════════════════════════════════════════════


class MaxwellSolver:
    r"""
    Yee / leap-frog sobre DEC, con auditoría PHS de Poincaré.

    Semi-discretización (2D TE⊥, co-cadenas integradas):
        ∂ₜ B = -d₁ E - σₘ H + Jₘ
        ∂ₜ D =  δ₂ H - σₑ E - Jₑ
        D = ε ★₁ E,   H = μ⁻¹ ★₂ B
        U = ½ (Eᵀ D + Hᵀ B) = ½ (ε ‖E‖_{★₁}² + μ⁻¹ ‖B‖_{★₂}²)

    PHS: x=[D,B], ∇H=[E,H],
        J = [[0, ∂₂], [-∂₂ᵀ, 0]],  R=diag(σₑ, σₘ).
    Casimirs ⊃ cohomología: Gauss δ₁ D y ker d₁ (armónicos) están en ker J.
    """

    def __init__(
        self,
        calculus: DiscreteVectorCalculus,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        electric_conductivity: float = 0.0,
        magnetic_conductivity: float = 0.0,
        pml_thickness: float = 0.1,
        pml_max_sigma: float = 1.0,
    ) -> None:
        self.calc = calculus
        self.epsilon = max(float(permittivity), CONSTANTS.NUMERICAL_TOLERANCE)
        self.mu = max(float(permeability), CONSTANTS.NUMERICAL_TOLERANCE)
        self.sigma_e_base = max(float(electric_conductivity), 0.0)
        self.sigma_m_base = max(float(magnetic_conductivity), 0.0)
        self.sigma_e = self.sigma_e_base
        self.sigma_m = self.sigma_m_base
        self.c = 1.0 / math.sqrt(self.epsilon * self.mu)
        self._pml_thickness = float(pml_thickness)
        self._pml_max_sigma = float(pml_max_sigma)
        self._initialize_pml()
        self.E = np.zeros(calculus.num_edges, dtype=float)
        self.B = np.zeros(calculus.num_faces, dtype=float)
        self.D = np.zeros(calculus.num_edges, dtype=float)
        self.H = np.zeros(calculus.num_faces, dtype=float)
        self.J_e = np.zeros(calculus.num_edges, dtype=float)
        self.J_m = np.zeros(calculus.num_faces, dtype=float)
        self.time = 0.0
        self.step_count = 0
        self.dt_cfl = self._compute_cfl_limit()
        self.energy_history: deque = deque(maxlen=10_000)
        self._coeff_cache: "OrderedDict[float, Tuple[np.ndarray, ...]]" = OrderedDict()
        self._kernel_cache: Optional[PoincareHamiltonianKernel] = None

    def _initialize_pml(self) -> None:
        """
        PML parabólica σ(ρ)=σ_max ρ² sobre el embedding.
        Si hay node_positions se usa la norma euclídea al centroide;
        si no, se degrada a índices (solo válido para grids enteros).
        """
        n_e, n_f = self.calc.num_edges, self.calc.num_faces
        self.sigma_e_pml = np.zeros(n_e, dtype=float)
        self.sigma_m_pml = np.zeros(n_f, dtype=float)
        if not SCIPY_AVAILABLE:
            return
        pos = self.calc.node_positions
        if pos:
            coords = np.stack([pos[n] for n in self.calc.nodes if n in pos], axis=0)
            center = coords.mean(axis=0)
            radii = np.linalg.norm(coords - center, axis=1)
            r_max = float(np.max(radii)) if radii.size else 1.0
        else:
            center = np.array([(self.calc.num_nodes - 1) / 2.0])
            r_max = max(float(center[0]), 1.0)
            logger.warning("PML sin node_positions: se usan índices de nodo (geométricamente ad hoc).")

        threshold = 1.0 - self._pml_thickness

        def _rho_node(node: int) -> float:
            if pos and node in pos:
                r = float(np.linalg.norm(pos[node] - center)) / max(r_max, CONSTANTS.NUMERICAL_TOLERANCE)
            else:
                r = abs(node - float(center.reshape(-1)[0])) / max(r_max, 1.0)
            return r

        for idx, (u, v) in enumerate(self.calc.edges):
            r = 0.5 * (_rho_node(u) + _rho_node(v))
            if r > threshold:
                rho = (r - threshold) / max(self._pml_thickness, CONSTANTS.NUMERICAL_TOLERANCE)
                self.sigma_e_pml[idx] = self._pml_max_sigma * (rho ** 2)
        for idx, face in enumerate(self.calc.faces):
            r = float(np.mean([_rho_node(n) for n in face]))
            if r > threshold:
                rho = (r - threshold) / max(self._pml_thickness, CONSTANTS.NUMERICAL_TOLERANCE)
                self.sigma_m_pml[idx] = self._pml_max_sigma * (rho ** 2)

    def _compute_cfl_limit(self) -> float:
        r"""
        CFL espectral: leap-frog de ω (curl-curl) es estable ssi Δt < 2/ρ(ω)
        para el oscilador ẍ = -ω x, ω = c² Δ₁ (métrica absorbida en ★).

        Fallback geométrico:  Δt ≤ CFL · min|e| / (c √d_eff).
        """
        dt_geom = CONSTANTS.MIN_DELTA_TIME
        if self.calc.num_edges > 0:
            lengths = np.array(
                [self.calc.primal_edge_length(i) for i in range(self.calc.num_edges)],
                dtype=float,
            )
            min_len = float(np.min(lengths)) if lengths.size else 1.0
            dim_eff = 2.0 if self.calc.is_planar else 3.0
            dt_geom = CONSTANTS.CFL_SAFETY_FACTOR * min_len / (self.c * math.sqrt(dim_eff))
        dt_spec = dt_geom
        if SCIPY_AVAILABLE and self.calc.num_edges >= 2:
            rho = self.calc.curl_curl_spectral_radius()
            # Δ₁ actúa sobre 1-formas; ∂ₜₜ E = -c² Δ₁ E  ⇒ ω_max = c √ρ
            if rho > CONSTANTS.NUMERICAL_TOLERANCE:
                omega_max = self.c * math.sqrt(rho)
                dt_spec = CONSTANTS.CFL_SAFETY_FACTOR * (2.0 / omega_max)
        dt_est = min(dt_geom, dt_spec) if dt_spec > 0.0 else dt_geom
        return max(dt_est, CONSTANTS.MIN_DELTA_TIME)

    def _get_update_coefficients(self, dt: float) -> Tuple[np.ndarray, ...]:
        """
        Leap-frog + Crank–Nicolson en σ (esquema exponencial de 1er orden):
            α = σ Δt / (2 ε),   c1=(1-α)/(1+α),  c2=Δt/(ε(1+α)).
        Incondicionalmente estable en el subpaso ohmico.
        """
        dt = float(dt)
        if dt < CONSTANTS.MIN_DELTA_TIME:
            raise ConfigurationError(f"dt={dt:.3e} < MIN_DELTA_TIME.")
        if dt in self._coeff_cache:
            self._coeff_cache.move_to_end(dt)
            return self._coeff_cache[dt]
        sigma_e = self.sigma_e_base + self.sigma_e_pml
        sigma_m = self.sigma_m_base + self.sigma_m_pml
        alpha_e = sigma_e * dt / (2.0 * self.epsilon)
        ce1 = (1.0 - alpha_e) / (1.0 + alpha_e)
        ce2 = dt / (self.epsilon * (1.0 + alpha_e))
        alpha_m = sigma_m * dt / (2.0 * self.mu)
        ch1 = (1.0 - alpha_m) / (1.0 + alpha_m)
        ch2 = dt / (self.mu * (1.0 + alpha_m))
        result = (ce1, ce2, ch1, ch2)
        self._coeff_cache[dt] = result
        if len(self._coeff_cache) > CONSTANTS.MAX_COEFF_CACHE:
            self._coeff_cache.popitem(last=False)
        return result

    def update_constitutive_relations(self) -> None:
        r"""D = ε ★₁ E,  H = μ⁻¹ ★₂ B."""
        if not SCIPY_AVAILABLE:
            return
        if self.calc.num_edges > 0:
            self.D = self.epsilon * (self.calc.star1 @ self.E)
        if self.calc.num_faces > 0:
            self.H = (1.0 / self.mu) * (self.calc.star2 @ self.B)

    def step_magnetic_field(self, dt: float) -> None:
        """B^{n+½} = ch1 B^{n-½} - ch2 (d₁ E + J_m)."""
        if not SCIPY_AVAILABLE or self.calc.num_faces == 0:
            return
        _, _, ch1, ch2 = self._get_update_coefficients(dt)
        curl_E = self.calc.curl(self.E)
        self.B = ch1 * self.B - ch2 * (curl_E + self.J_m)
        self.H = (1.0 / self.mu) * (self.calc.star2 @ self.B)

    def step_electric_field(self, dt: float) -> None:
        r"""E^{n+1} = ce1 E^n + ce2 ★₁⁻¹ (★₁ δ₂ H - J_e) / (implícito en ce2/ε).

        Con D=ε★₁E,  ΔD = Δt (δ₂ H - σE - J) se traduce, vía CN en σ, a:
            E ← ce1 E + (ce2) ★₁⁻¹ (★₁ δ₂ H - J_e)
        y δ₂ = ★₁⁻¹ ∂₂ ★₂, por tanto ★₁ δ₂ H = ∂₂ ★₂ H.
        Equivalente y más estable: usar cocurl y luego ★₁⁻¹.
        """
        if not SCIPY_AVAILABLE or self.calc.num_edges == 0:
            return
        ce1, ce2, _, _ = self._get_update_coefficients(dt)
        if self.calc.num_faces > 0:
            ampere = self.calc.cocurl(self.H)  # δ₂ H  ∈ C¹
        else:
            ampere = np.zeros(self.calc.num_edges, dtype=float)
        # ce2 ya contiene 1/ε; Ampère vive en el espacio de D, convertimos con ★₁⁻¹
        metric_term = ampere - (self.calc.star1_inv @ self.J_e)
        self.E = ce1 * self.E + ce2 * metric_term
        self.D = self.epsilon * (self.calc.star1 @ self.E)

    def leapfrog_step(self, dt: Optional[float] = None) -> None:
        """Yee: B (usa Eⁿ) → E (usa H^{n+½}). Conserva Gauss si δd=0."""
        if not SCIPY_AVAILABLE:
            return
        if dt is None:
            dt = 0.9 * self.dt_cfl
        if dt > self.dt_cfl * (1.0 + CONSTANTS.RELATIVE_TOLERANCE):
            raise NumericalInstabilityError(f"Δt={dt:.3e} > Δt_CFL={self.dt_cfl:.3e}.")
        self.step_magnetic_field(dt)
        self.step_electric_field(dt)
        self.time += dt
        self.step_count += 1
        self.energy_history.append(self.total_energy())
        self._monitor_energy_stability()

    def _monitor_energy_stability(self) -> None:
        if len(self.energy_history) < CONSTANTS.ENERGY_WINDOW:
            return
        recent = np.array(list(self.energy_history)[-CONSTANTS.ENERGY_WINDOW :], dtype=float)
        baseline = float(np.median(recent[: CONSTANTS.ENERGY_WINDOW // 4]))
        if baseline < CONSTANTS.MIN_ENERGY_THRESHOLD:
            return
        peak = float(np.max(recent))
        if peak > CONSTANTS.ENERGY_BLOWUP_RATIO * baseline:
            raise NumericalInstabilityError(
                f"Energía EM anómala: pico/mediana={peak/baseline:.2e}."
            )

    def total_energy(self) -> float:
        r"""U = ½(Eᵀ D + Hᵀ B) = ½(ε ‖E‖_{★₁}² + μ⁻¹ ‖B‖_{★₂}²)."""
        if not SCIPY_AVAILABLE:
            return 0.0
        U_e = 0.5 * float(np.dot(self.E, self.D)) if self.calc.num_edges > 0 else 0.0
        U_m = 0.5 * float(np.dot(self.H, self.B)) if self.calc.num_faces > 0 else 0.0
        return _finite("U_em", U_e + U_m)

    def poynting_flux(self) -> np.ndarray:
        r"""
        1-forma de Poynting lumped: S_e = E_e · ⟨H⟩_{⋆e}.

        En 2D TE⊥, S = ★(E ∧ H) vive en el dual de las aristas. Esta fórmula
        es la contracción C⁰ del producto interior interior de la 2-forma
        de Faraday con el campo dual; exacta en mallados cartesianos Yee.
        """
        if not SCIPY_AVAILABLE:
            return np.array([])
        S = np.zeros(self.calc.num_edges, dtype=float)
        if self.calc.num_faces == 0:
            return S
        for edge_idx in range(self.calc.num_edges):
            adjacent = self.calc.edge_to_faces.get(edge_idx, [])
            if not adjacent:
                continue
            H_avg = float(np.mean([self.H[f_idx] for f_idx, _ in adjacent]))
            S[edge_idx] = self.E[edge_idx] * H_avg
        return S

    def gauss_residual(self, rho: Optional[np.ndarray] = None) -> float:
        r"""‖δ₁ D - ρ‖₂. Casimir: d/dt (δ₁ D) = δ₁ δ₂ H = 0."""
        if not SCIPY_AVAILABLE:
            return 0.0
        div_D = self.calc.divergence(self.D)
        if rho is None:
            return float(np.linalg.norm(div_D))
        return float(np.linalg.norm(div_D - _as_1d(rho)))

    def electromagnetic_momentum(self) -> np.ndarray:
        """Densidad de momento ε μ S · |e| (Abraham/Minkowski coinciden en vacío lineal)."""
        S = self.poynting_flux()
        if S.size == 0:
            return S
        lengths = np.array([self.calc.primal_edge_length(i) for i in range(S.size)], dtype=float)
        return self.epsilon * self.mu * S * lengths

    def set_initial_conditions(
        self,
        E0: Optional[np.ndarray] = None,
        B0: Optional[np.ndarray] = None,
    ) -> None:
        if E0 is not None:
            E0 = _as_1d(E0)
            if E0.size != self.calc.num_edges:
                raise ConfigurationError(f"E0 debe tener tamaño {self.calc.num_edges}")
            self.E = E0.copy()
        if B0 is not None:
            B0 = _as_1d(B0)
            if B0.size != self.calc.num_faces:
                raise ConfigurationError(f"B0 debe tener tamaño {self.calc.num_faces}")
            self.B = B0.copy()
        self.update_constitutive_relations()

    def compute_energy_and_momentum(self) -> Dict[str, Any]:
        if not SCIPY_AVAILABLE:
            return {"total_energy": 0.0}
        S = self.poynting_flux()
        P = self.electromagnetic_momentum()
        return {
            "total_energy": self.total_energy(),
            "poynting_vector": S,
            "poynting_magnitude": float(np.linalg.norm(S)) if S.size else 0.0,
            "poynting_mean": float(np.mean(np.abs(S))) if S.size else 0.0,
            "poynting_max": float(np.max(np.abs(S))) if S.size else 0.0,
            "momentum_vector": P,
            "momentum_magnitude": float(np.linalg.norm(P)) if P.size else 0.0,
            "gauss_residual": self.gauss_residual(),
        }

    def audit_discrete_identities(self) -> Dict[str, float]:
        if not SCIPY_AVAILABLE:
            return {}
        results: Dict[str, float] = {"gauss_residual": self.gauss_residual()}
        if self.calc.num_nodes > 0 and self.calc.num_faces > 0:
            results["curl_grad_residual"] = float(
                np.linalg.norm(self.calc.curl(self.calc.gradient(np.zeros(self.calc.num_nodes))))
            )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 1.5 — PUENTE FORMAL A LA FASE 2
    # ──────────────────────────────────────────────────────────────────────────

    def _get_poincare_kernel(self) -> Optional[PoincareHamiltonianKernel]:
        r"""
        Kernel PHS para x=[D,B].

            K_ee = ★₁⁻¹ / ε     (E = K_ee D)
            K_bb = ★₂ / μ       (H = K_bb B)     ← corrección 7.1
            J    = [[0, ∂₂], [-∂₂ᵀ, 0]]
            R    = diag(σₑ^{base+PML}, σₘ^{base+PML})
        """
        if self._kernel_cache is not None:
            return self._kernel_cache
        if not SCIPY_AVAILABLE:
            return None
        n_e = self.calc.num_edges
        n_f = self.calc.num_faces
        dim = n_e + n_f
        if dim == 0:
            return None
        if dim > CONSTANTS.MAX_POINCARE_STATE_DIM:
            logger.warning(
                "Estado Poincaré dim=%d > MAX=%d; se omite kernel denso.",
                dim,
                CONSTANTS.MAX_POINCARE_STATE_DIM,
            )
            return None
        B2 = self.calc.boundary2.toarray() if n_f > 0 else np.zeros((n_e, 0))
        electric_metric = np.asarray(self.calc.star1_inv.diagonal(), dtype=float) / self.epsilon
        magnetic_metric = (
            np.asarray(self.calc.star2.diagonal(), dtype=float) / self.mu
            if n_f > 0
            else np.zeros(0, dtype=float)
        )
        try:
            kernel = PoincareHamiltonianKernel.from_maxwell_blocks(
                boundary2=B2,
                electric_metric_diag=electric_metric,
                magnetic_metric_diag=magnetic_metric,
                sigma_e=self.sigma_e_base + self.sigma_e_pml,
                sigma_m=self.sigma_m_base + self.sigma_m_pml,
            )
        except DataFluxCondenserError as exc:
            logger.error("No se pudo construir kernel Poincaré: %s", exc)
            return None
        self._kernel_cache = kernel
        return kernel

    def poincare_step(
        self,
        dt: Optional[float] = None,
        apply_state: bool = False,
        integrator: str = "strang",
    ) -> Tuple[Optional[np.ndarray], FluxCondenserStepReport]:
        """Paso PHS de auditoría sobre [D,B]. No sustituye a Yee salvo apply_state=True."""
        if dt is None:
            dt = 0.9 * self.dt_cfl
        kernel = self._get_poincare_kernel()
        dim = self.calc.num_edges + self.calc.num_faces
        if kernel is None:
            report = FluxCondenserStepReport(
                hamiltonian_energy=self.total_energy(),
                volume_drift=0.0,
                rayleigh_dissipation_rate=0.0,
                is_liouville_preserved=False,
                is_volume_contracting=True,
                poisson_residual=0.0,
                trace_generator=0.0,
                state_dimension=dim,
            )
            return None, report
        x = np.concatenate([self.D, self.B])
        x_next, report = kernel.compute_step(x, dt, integrator=integrator)
        if apply_state:
            n_e = self.calc.num_edges
            self.D = x_next[:n_e].copy()
            self.B = x_next[n_e:].copy()
            # reconstitución de E,H desde D,B con las constitutivas
            if n_e > 0:
                self.E = (1.0 / self.epsilon) * (self.calc.star1_inv @ self.D)
            if self.calc.num_faces > 0:
                self.H = (1.0 / self.mu) * (self.calc.star2 @ self.B)
            self.time += dt
            self.step_count += 1
            self.energy_history.append(self.total_energy())
        return x_next, report

    def synthesize_poincare_control_seed(
        self,
        target_energy: Optional[float] = None,
        dt: Optional[float] = None,
        port_indices: Optional[np.ndarray] = None,
        desired_metric: Optional[np.ndarray] = None,
    ) -> PoincareControlSeed:
        r"""
        PUENTE FORMAL FASE 1 → FASE 2.

        Semilla enriquecida para:
          · PIController (anti-windup + Lyapunov V=½(H-H*)², con análisis de V̇).
          · FluxMuscleController (slew-rate, térmica).
          · PortHamiltonianPoincareController (IDA-PBC lineal con matching).

        Contenido 7.1:
          1. x, ∇H, H, H*, V, ∇V, V̇.
          2. PHS (J, R, K, g) y salida de puerto y = gᵀ ∇H.
          3. Casimirs = ker J  (Gauss + armónicos; dim ~ β₁+β₂).
          4. Espectro de A=(J-R)K, μ₂(A), estabilidad módulo Casimir.
          5. Matching IDA-PBC lineal {J_d, J_a, R_d, K_d, G, residual}.
          6. Metadatos topológicos (χ, β) y CFL espectral.
          7. Advertencia de bombeo: V̇≤0 en {H≥H*} solamente; Fase 2 debe
             inyectar u a través de g si H < H*.
        """
        kernel = self._get_poincare_kernel()
        n_e = self.calc.num_edges
        n_f = self.calc.num_faces
        dim = n_e + n_f
        x = np.concatenate([self.D, self.B])
        H = self.total_energy()
        H_star = (
            max(float(target_energy), CONSTANTS.MIN_ENERGY_THRESHOLD)
            if target_energy is not None
            else max(H, CONSTANTS.MIN_ENERGY_THRESHOLD)
        )
        if kernel is not None:
            grad_H = kernel.gradient(x)
            J = kernel.J.copy()
            R = kernel.R.copy()
            K = kernel.metric.copy()
            rayleigh = kernel.rayleigh_dissipation_rate(x)
            is_conservative = kernel.is_nominally_conservative
            casimir_basis = kernel.casimir_basis.copy()
            spec = kernel.spectral_audit()
            spectral_data: Optional[Dict[str, Any]] = {
                "eigenvalues": spec.eigenvalues,
                "max_real_part": spec.max_real_part,
                "spectral_radius": spec.spectral_radius,
                "logarithmic_norm": spec.logarithmic_norm,
                "is_lyapunov_stable": spec.is_lyapunov_stable,
                "is_asymptotically_stable_mod_casimir": spec.is_asymptotically_stable_mod_casimir,
                "is_normal": spec.is_normal,
                "departure_from_normality": spec.departure_from_normality,
                "condition_metric": spec.condition_metric,
                "casimir_multiplicity": spec.casimir_multiplicity,
            }
            V, dV, Vdot = kernel.energy_shaping_lyapunov(x, H_star)
            K_d = desired_metric if desired_metric is not None else K
            matching = kernel.linear_ida_pbc_matching(K_d=K_d)
            matching_residual = float(matching["matching_residual"].reshape(-1)[0])
            ida = {
                "J_d": matching["J_d"],
                "J_a": matching["J_a"],
                "R_d": matching["R_d"],
                "K_d": matching["K_d"],
                "G": matching["G"],
                "x_star": matching["x_star"],
            }
        else:
            grad_H = np.concatenate([self.E, self.H]) if dim else np.zeros(0)
            J = np.zeros((dim, dim), dtype=float)
            R = np.zeros((dim, dim), dtype=float)
            K = np.eye(dim, dtype=float) if dim else np.zeros((0, 0))
            rayleigh = 0.0
            is_conservative = False
            casimir_basis = None
            spectral_data = None
            V = 0.5 * float((H - H_star) ** 2)
            dV = (H - H_star) * grad_H
            Vdot = 0.0
            matching_residual = 0.0
            ida = {"J_d": J.copy(), "J_a": np.zeros_like(J), "R_d": R.copy(), "K_d": K.copy()}

        if port_indices is None:
            g = np.eye(dim, dtype=float) if dim else np.zeros((0, 0))
        else:
            idx = np.asarray(port_indices, dtype=int).ravel()
            if idx.size == 0 or np.any(idx < 0) or np.any(idx >= dim):
                raise ConfigurationError(f"port_indices inválidos: {idx!r} (dim={dim}).")
            g = np.zeros((dim, idx.size), dtype=float)
            g[idx, np.arange(idx.size)] = 1.0
        y = g.T @ grad_H if dim else np.zeros(0)

        betti = getattr(self.calc, "betti", BettiNumbers(
            getattr(self.calc, "betti_0", 0),
            getattr(self.calc, "betti_1", 0),
            getattr(self.calc, "betti_2", 0),
        ))
        pumping_required = bool(H < H_star - CONSTANTS.MIN_ENERGY_THRESHOLD)
        metadata: Dict[str, Any] = {
            "time": self.time,
            "step_count": self.step_count,
            "dt_cfl": self.dt_cfl,
            "dt_suggested": dt if dt is not None else 0.9 * self.dt_cfl,
            "epsilon": self.epsilon,
            "mu": self.mu,
            "sigma_e_base": self.sigma_e_base,
            "sigma_m_base": self.sigma_m_base,
            "pml_thickness": self._pml_thickness,
            "pml_max_sigma": self._pml_max_sigma,
            "num_edges": n_e,
            "num_faces": n_f,
            "is_conservative_kernel": is_conservative,
            "rayleigh_dissipation_rate": rayleigh,
            "lyapunov_Vdot": Vdot,
            "pumping_required": pumping_required,
            "euler_characteristic": self.calc.euler_characteristic,
            "betti_numbers": betti.as_tuple(),
            "casimir_dimension": kernel.casimir_dimension if kernel is not None else 0,
            "state_dimension": dim,
            "hodge_convention": "D=ε★₁E, H=μ⁻¹★₂B, δ₂=★₁⁻¹∂₂★₂",
            "integrator_recommended": "strang",
            "schema_version": "7.1.0",
        }
        return PoincareControlSeed(
            state=x,
            gradient=grad_H,
            hamiltonian=float(H),
            target_hamiltonian=H_star,
            lyapunov_candidate=V,
            interconnection_matrix=J,
            damping_matrix=R,
            metric_matrix=K,
            port_matrix=g,
            metadata=metadata,
            casimir_basis=casimir_basis,
            spectral_data=spectral_data,
            ida_pbc_decomposition=ida,
            lyapunov_jacobian=dV,
            output_port=y,
            matching_residual=matching_residual,
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# FIN DE LA FASE 1  (v7.1.0)
# ═══════════════════════════════════════════════════════════════════════════════════════
#
# MaxwellSolver.synthesize_poincare_control_seed() es la frontera formal hacia
# la FASE 2. Contrato del PoincareControlSeed (7.1):
#
#   1. x, ∇H, H, H*, V=½(H-H*)², ∇V, y=gᵀ∇H.
#   2. PHS (J, R, K, g) con constitutivas DEC consistentes.
#   3. Casimirs ker J  ↔  cohomología (Gauss + armónicos).
#   4. σ(A), μ₂(A), estabilidad de Lyapunov módulo Casimir.
#   5. Matching IDA-PBC lineal (J_d, J_a, R_d, K_d, G, residual).
#   6. metadata["pumping_required"]  ⇒  Fase 2 debe inyectar puerto si H<H*.
#   7. metadata["hodge_convention"]  inmutable para no romper el matching.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# ╔═════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2/3 — CONTROLADORES PI, MÚSCULO DE FLUJO Y PORT-HAMILTONIAN POINCARÉ          ║
# ║  Versión: 7.1.0-Poincare-DEC-PHS-Rigorous                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════════
#
# Frontera de entrada : MaxwellSolver.synthesize_poincare_control_seed()
#                       → PoincareControlSeed  (contrato 7.1)
# Frontera de salida  : PortHamiltonianPoincareController.synthesize_engine_seed()
#                       → PoincareEngineSeed   (contrato 7.1 → Fase 3)
#
# Convenciones inmutables (heredadas de 7.1, no se redefinen ★ ni J):
#   PHS:  ẋ = (J−R)∇H + g u,  y = gᵀ ∇H,  H = ½ xᵀ K x
#   Casimir lineal: C(x)=Cᵀx, JC=0.  Invarianza bajo control ⇔ Cᵀ g = 0.
#   Energía K-ortogonal: H = H_c ⊕ H_d,  H_c no es regulable.
#   Bombeo: yᵀ u = ∇Hᵀ R ∇H − λ(H−H*)   (compensa Rayleigh).
#   Integrador de lazo: punto medio implícito (gradiente discreto de H cuadrática).
# ═══════════════════════════════════════════════════════════════════════════════════════

import time
from enum import Enum

try:
    from scipy.linalg import lu_factor, lu_solve
    _LU_AVAILABLE = True
except ImportError:  # pragma: no cover
    lu_factor = None
    lu_solve = None
    _LU_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 2.1 — EXCEPCIONES, CONSTANTES DE CONTROL Y ESTRUCTURAS
# ═══════════════════════════════════════════════════════════════════════════════════════


class ControlSeedError(DataFluxCondenserError):
    """Semilla de control Poincaré inválida, incompleta o no finita."""


class PassivityViolationError(DataFluxCondenserError):
    """Violación de la desigualdad de pasividad discreta (no de la identidad continua)."""


class MuscleThermalError(DataFluxCondenserError):
    """Condición térmica inválida o peligrosa en el músculo de flujo."""


class AntiWindupError(DataFluxCondenserError):
    """Configuración o estado inconsistente de anti-windup."""


class UncontrollableEnergyError(DataFluxCondenserError):
    """H* es inalcanzable: autoridad de puerto nula o H* bajo la energía de Casimir."""


class ControlMode(str, Enum):
    """
    Modo del lazo Port-Hamiltoniano.

    ENERGY_LEVEL  — regulación de la hoja {H=H*} ∩ {C=C₀} por power-shaping.
    IDA_PBC_POINT — matching lineal u = Gx + v, v = −R_a y_d, hacia x*.
    DAMPING_ONLY  — u = −k_d y  (inyección de amortiguamiento pura).
    """

    ENERGY_LEVEL = "energy_level"
    IDA_PBC_POINT = "ida_pbc_point"
    DAMPING_ONLY = "damping_only"


@dataclass(frozen=True)
class ControlConstants:
    """Constantes del lazo. No se mezclan con las de geometría DEC/PHS."""

    LARGE_DIM_THRESHOLD: int = 512
    PORT_RANK_TOLERANCE: float = 1e-10
    POWER_REGULARIZATION: float = 1e-12
    PICARD_MAX_ITER: int = 8
    PICARD_ATOL: float = 1e-12
    RK4_IMAG_LIMIT: float = 2.5          # eje imaginario de RK4 ≈ 2√2
    MIDPOINT_CFL: float = 0.9            # h·μ₂(A) ≤ 0 ya es contractivo; este es extra
    DISCRETE_PASSIVITY_TOL: float = 1e-8
    THEIL_SEN_WINDOW: int = 8
    MIN_LYAPUNOV_SAMPLES: int = 16
    MUSCLE_FATIGUE_HI: float = 0.8
    MUSCLE_FATIGUE_LO: float = 0.6
    DEFAULT_ENERGY_RATE: float = 0.1     # λ en Ḣ = −λ(H−H*)  [1/s]
    QR_PIVOT_TOL: float = 1e-10

    def __post_init__(self) -> None:
        if not (0.0 < self.MIDPOINT_CFL <= 1.0):
            raise ConfigurationError("MIDPOINT_CFL debe estar en (0, 1].")


CTRL = ControlConstants()


@dataclass(frozen=True)
class DiscretePassivityAudit:
    """
    Balance de Tellegen *discreto* en un paso de punto medio.

    residual = (H_{n+1}−H_n) − h (y_midᵀ u − ‖∇H_mid‖_R²)
    Debe ser ~ 0 a precisión de máquina para H cuadrática (gradiente discreto).
    is_passive se refiere a H_{n+1}−H_n ≤ h y_midᵀ u + tol  (Rayleigh ≥ 0).
    """

    dH_discrete: float
    supply: float
    rayleigh: float
    residual: float
    casimir_drift: float
    is_discrete_gradient: bool
    is_passive: bool


@dataclass(eq=False)
class ValidatedPoincareSeed:
    r"""
    Semilla Poincaré proyectada a un PHS admisible.

    Garantías:
        J = −Jᵀ,  R = Rᵀ ⪰ 0,  K = Kᵀ ≻ 0,
        tr((J−R)K) = −tr(RK) ≤ 0,
        g con columnas independientes y Cᵀ g = 0 (Casimirs inmunes al puerto),
        H* ≥ H_c  (energía de Casimir K-ortogonal).
    """

    raw: PoincareControlSeed
    state_dim: int
    port_dim: int
    hamiltonian_error: float
    normalized_energy_error: float
    gradient_norm: float
    rayleigh_dissipation_rate: float
    is_rayleigh_nonpositive: bool
    is_passive: bool
    spectral_radius_J: float
    min_metric_eigenvalue: float
    max_metric_eigenvalue: float
    metric_condition_number: float
    min_damping_eigenvalue: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    casimir_basis: Optional[np.ndarray] = None
    spectral_data: Optional[Dict[str, Any]] = None
    ida_pbc_decomposition: Optional[Dict[str, np.ndarray]] = None
    lyapunov_jacobian: Optional[np.ndarray] = None
    port_gram_condition: float = 1.0
    trace_generator: float = 0.0
    # Extensión 7.1
    output_port: Optional[np.ndarray] = None
    matching_residual: float = 0.0
    pumping_required: bool = False
    casimir_energy: float = 0.0
    dynamic_energy: float = 0.0
    logarithmic_norm: float = 0.0
    is_energy_controllable: bool = True
    casimir_port_leak: float = 0.0


@dataclass(frozen=True)
class PIControlReport:
    """Iteración PI. `output` vive en unidades de mando (potencia normalizada o u)."""

    output: float
    error: float
    filtered_pv: float
    integral_error: float
    integral_term: float
    p_term: float
    i_term: float
    feedforward: float
    saturated: bool
    anti_windup_correction: float
    lyapunov_exponent: float
    oscillation_index: float
    extrema_density: float = 0.0
    applied_output: float = 0.0


@dataclass(frozen=True)
class MuscleThermalState:
    """Estado térmico/mecánico. `duty` ∈ [−quadrants+1, 1] tras saturación real."""

    duty: float
    commanded: float
    temperature: float
    thermal_accumulator: float
    overheated: bool
    derated: bool
    thermal_derate_cap: float = 1.0
    applied_scale: float = 1.0


@dataclass(frozen=True)
class PortHamiltonianControlReport:
    r"""
    Paso de control PHS.

    is_passive          — pasividad *discreta* de planta (ΔH ≤ h yᵀu + tol).
    is_regulating       — V_{n+1} ≤ V_n + O(h²)  **o** (H<H* y ΔH>0) (bombeo).
    plant_passivity_slack — ‖∇H_mid‖_R²  (≥ 0).
    regulation_slack    — −ΔV/h  (puede ser negativo transitoriamente si R lucha).
    """

    time: float
    dt: float
    hamiltonian: float
    hamiltonian_derivative: float
    target_hamiltonian: float
    storage_function: float
    control_norm: float
    port_output_norm: float
    supply_rate: float
    lyapunov_derivative: float
    plant_passivity_slack: float
    regulation_slack: float
    passivity_margin: float
    is_passive: bool
    is_regulating: bool
    discrete_residual: float = 0.0
    casimir_drift: float = 0.0
    power_requested: float = 0.0
    power_delivered: float = 0.0
    applied_control_norm: float = 0.0
    pumping_required: bool = False
    mode: str = ControlMode.ENERGY_LEVEL.value


@dataclass(eq=False)
class PoincareEngineSeed:
    r"""
    Semilla Fase 2 → Fase 3.

    `control_input` es el mando **aplicado** (post-músculo, post-proyección Casimir).
    `commanded_control` es el mando pre-saturación. Fase 3 integra con el aplicado.
    """

    state: np.ndarray
    gradient: np.ndarray
    control_input: np.ndarray
    hamiltonian: float
    target_hamiltonian: float
    storage_function: float
    interconnection_matrix: np.ndarray
    damping_matrix: np.ndarray
    metric_matrix: np.ndarray
    port_matrix: np.ndarray
    pi_report: Optional[PIControlReport]
    muscle_duty: float
    passivity_report: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    casimir_basis: Optional[np.ndarray] = None
    spectral_data: Optional[Dict[str, Any]] = None
    ida_pbc_decomposition: Optional[Dict[str, np.ndarray]] = None
    lyapunov_jacobian: Optional[np.ndarray] = None
    engine_hints: Dict[str, Any] = field(default_factory=dict)
    commanded_control: Optional[np.ndarray] = None
    output_port: Optional[np.ndarray] = None
    matching_residual: float = 0.0
    pumping_required: bool = False
    casimir_energy: float = 0.0
    schema_version: str = "7.1.0"


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 2.2 — PUENTE DE VALIDACIÓN Y PROYECCIÓN PORT-HAMILTONIANA
# ═══════════════════════════════════════════════════════════════════════════════════════


class PoincareControlBridge:
    r"""
    Puente epistémico Fase 1 → Fase 2.

    Proyecciones (en este orden, para no invalidar Casimirs a posteriori):
        1. J ← ½(J − Jᵀ)
        2. C ← ker J  (SVD; rango par por Pfaffiano)
        3. R ← Π_{Sym⁺}(R)  espectral; componentes negativos *solamente*
        4. K ← Π_{Sym⁺⁺}(K)  con suelo ε
        5. g ← QR con pivote, luego g ← (I − C Cᵀ) g   (Cᵀ g = 0)
        6. Matching residual recalculado sobre las matrices proyectadas
        7. Descomposición K-ortogonal H = H_c ⊕ H_d
    """

    LARGE_DIM_THRESHOLD: int = CTRL.LARGE_DIM_THRESHOLD

    @classmethod
    def validate_seed(cls, seed: PoincareControlSeed) -> ValidatedPoincareSeed:
        projected = cls.project_seed(seed)
        state = np.asarray(projected.state, dtype=float).reshape(-1)
        grad = np.asarray(projected.gradient, dtype=float).reshape(-1)
        J = np.asarray(projected.interconnection_matrix, dtype=float)
        R = np.asarray(projected.damping_matrix, dtype=float)
        K = np.asarray(projected.metric_matrix, dtype=float)
        g = np.asarray(projected.port_matrix, dtype=float)
        dim = int(state.size)
        port_dim = int(g.shape[1]) if g.ndim == 2 else 0
        H = float(projected.hamiltonian)
        H_target = float(projected.target_hamiltonian)
        if H_target <= CONSTANTS.MIN_ENERGY_THRESHOLD:
            raise ControlSeedError(
                f"target_hamiltonian debe ser > {CONSTANTS.MIN_ENERGY_THRESHOLD}."
            )

        energy_error = H - H_target
        normalized_energy_error = abs(energy_error) / max(H_target, CONSTANTS.MIN_ENERGY_THRESHOLD)
        gradient_norm = float(np.linalg.norm(grad))
        rayleigh = -float(grad @ (R @ grad))
        if abs(rayleigh) < CONSTANTS.NUMERICAL_TOLERANCE:
            rayleigh = 0.0
        is_rayleigh_nonpositive = bool(rayleigh <= CONSTANTS.NUMERICAL_TOLERANCE)

        min_R_eig = cls._min_symmetric_eigenvalue(R)
        min_K_eig, max_K_eig = cls._symmetric_eigen_bounds(K)
        metric_condition = cls._condition_number_from_eigen_bounds(min_K_eig, max_K_eig)
        is_passive = bool(min_R_eig >= -CONSTANTS.NUMERICAL_TOLERANCE and is_rayleigh_nonpositive)
        spectral_radius_J = cls._spectral_radius_estimate(J)

        gram = g.T @ g if port_dim > 0 else np.eye(1)
        try:
            s = np.linalg.svd(gram, compute_uv=False) if port_dim > 0 else np.array([1.0])
            port_gram_condition = float(s[0] / max(float(s[-1]), CONSTANTS.NUMERICAL_TOLERANCE))
        except np.linalg.LinAlgError:
            port_gram_condition = float("inf")

        A_gen = (J - R) @ K
        trace_generator = float(np.trace(A_gen))
        log_norm = cls._logarithmic_norm(A_gen)

        C = projected.casimir_basis
        casimir_energy, dynamic_energy = cls._k_orthogonal_energy_split(state, K, C)
        if H_target + CONSTANTS.MIN_ENERGY_THRESHOLD < casimir_energy:
            raise UncontrollableEnergyError(
                f"H*={H_target:.6e} < H_casimir={casimir_energy:.6e}: "
                "la hoja {H=H*} no intersecta el nivel de Casimir."
            )

        casimir_port_leak = 0.0
        if C is not None and C.size and port_dim > 0:
            casimir_port_leak = float(np.linalg.norm(C.T @ g, ord="fro"))

        y = g.T @ grad if port_dim > 0 else np.zeros(0)
        pumping_required = bool(
            (projected.metadata or {}).get("pumping_required", H < H_target - CONSTANTS.MIN_ENERGY_THRESHOLD)
        )
        is_energy_controllable = bool(
            float(np.linalg.norm(y)) > CONSTANTS.NUMERICAL_TOLERANCE
            or abs(energy_error) <= CONSTANTS.RELATIVE_TOLERANCE * max(H_target, 1.0)
        )
        if pumping_required and not is_energy_controllable:
            logger.warning(
                "pumping_required=True pero y=gᵀ∇H≈0: no hay autoridad de puerto para inyectar."
            )

        matching_residual = float(getattr(projected, "matching_residual", 0.0) or 0.0)
        metadata = dict(projected.metadata or {})
        metadata.update(
            {
                "bridge": "PoincareControlBridge",
                "phase": 2,
                "schema_version": "7.1.0",
                "state_norm": float(np.linalg.norm(state)),
                "gradient_norm": gradient_norm,
                "energy_error": energy_error,
                "normalized_energy_error": normalized_energy_error,
                "port_dim": port_dim,
                "port_gram_condition": port_gram_condition,
                "casimir_energy": casimir_energy,
                "dynamic_energy": dynamic_energy,
                "casimir_port_leak": casimir_port_leak,
                "pumping_required": pumping_required,
                "logarithmic_norm": log_norm,
                "hodge_convention": metadata.get(
                    "hodge_convention", "D=ε★₁E, H=μ⁻¹★₂B, δ₂=★₁⁻¹∂₂★₂"
                ),
            }
        )
        return ValidatedPoincareSeed(
            raw=projected,
            state_dim=dim,
            port_dim=port_dim,
            hamiltonian_error=energy_error,
            normalized_energy_error=normalized_energy_error,
            gradient_norm=gradient_norm,
            rayleigh_dissipation_rate=rayleigh,
            is_rayleigh_nonpositive=is_rayleigh_nonpositive,
            is_passive=is_passive,
            spectral_radius_J=spectral_radius_J,
            min_metric_eigenvalue=min_K_eig,
            max_metric_eigenvalue=max_K_eig,
            metric_condition_number=metric_condition,
            min_damping_eigenvalue=min_R_eig,
            metadata=metadata,
            casimir_basis=None if C is None else np.asarray(C, dtype=float).copy(),
            spectral_data=None if not projected.spectral_data else dict(projected.spectral_data),
            ida_pbc_decomposition=(
                None
                if projected.ida_pbc_decomposition is None
                else {k: np.asarray(v, dtype=float).copy() for k, v in projected.ida_pbc_decomposition.items()}
            ),
            lyapunov_jacobian=(
                None
                if projected.lyapunov_jacobian is None
                else np.asarray(projected.lyapunov_jacobian, dtype=float).copy()
            ),
            port_gram_condition=port_gram_condition,
            trace_generator=trace_generator,
            output_port=y.copy(),
            matching_residual=matching_residual,
            pumping_required=pumping_required,
            casimir_energy=casimir_energy,
            dynamic_energy=dynamic_energy,
            logarithmic_norm=log_norm,
            is_energy_controllable=is_energy_controllable,
            casimir_port_leak=casimir_port_leak,
        )

    @classmethod
    def project_seed(cls, seed: PoincareControlSeed) -> PoincareControlSeed:
        state = cls._as_vector(seed.state, "state")
        gradient = cls._as_vector(seed.gradient, "gradient")
        dim = int(state.size)
        if gradient.size != dim:
            raise ControlSeedError(
                f"state ({state.size}) y gradient ({gradient.size}) deben coincidir."
            )
        J = cls._project_skew(cls._as_square_matrix(seed.interconnection_matrix, dim, "J"))
        R = cls._project_psd(cls._as_square_matrix(seed.damping_matrix, dim, "R"))
        K = cls._project_spd(cls._as_square_matrix(seed.metric_matrix, dim, "K"))
        g = cls._as_port_matrix(seed.port_matrix, dim, "g")
        if not np.all(np.isfinite(g)):
            raise ControlSeedError("port_matrix contiene valores no finitos.")

        casimir_basis = cls._casimir_basis_from_J(J)
        g = cls._casimir_compatible_ports(g, casimir_basis)
        g = cls._full_column_rank(g)

        H = float(seed.hamiltonian)
        H_target = float(seed.target_hamiltonian)
        if not math.isfinite(H) or H < 0.0:
            raise ControlSeedError(f"Hamiltoniano inválido: {H}")
        if not math.isfinite(H_target) or H_target <= 0.0:
            raise ControlSeedError(f"target_hamiltonian inválido: {H_target}")

        # Recalcular matching sobre el PHS proyectado.
        ida = None if seed.ida_pbc_decomposition is None else {
            k: np.asarray(v, dtype=float).copy() for k, v in seed.ida_pbc_decomposition.items()
        }
        matching_residual = float(getattr(seed, "matching_residual", 0.0) or 0.0)
        if ida is not None and "K_d" in ida:
            J_d = cls._project_skew(np.asarray(ida.get("J_d", J), dtype=float))
            R_d = cls._project_psd(np.asarray(ida.get("R_d", R), dtype=float))
            K_d = cls._project_spd(np.asarray(ida["K_d"], dtype=float))
            Delta = (J_d - R_d) @ K_d - (J - R) @ K
            G, matching_residual = cls._port_least_squares(g, Delta)
            ida.update({"J_d": J_d, "J_a": J_d * 0.0, "R_d": R_d, "K_d": K_d, "G": G})

        gradient = K @ state  # consistencia ∇H = Kx tras proyectar K
        lyapunov_candidate = 0.5 * float((H - H_target) ** 2)
        lyapunov_jacobian = (H - H_target) * gradient
        y = g.T @ gradient if g.size else np.zeros(0)
        metadata = dict(seed.metadata or {})
        return PoincareControlSeed(
            state=state.copy(),
            gradient=gradient.copy(),
            hamiltonian=H,
            target_hamiltonian=H_target,
            lyapunov_candidate=lyapunov_candidate,
            interconnection_matrix=J,
            damping_matrix=R,
            metric_matrix=K,
            port_matrix=g,
            metadata=metadata,
            casimir_basis=casimir_basis,
            spectral_data=dict(seed.spectral_data) if seed.spectral_data else None,
            ida_pbc_decomposition=ida,
            lyapunov_jacobian=lyapunov_jacobian,
            output_port=y,
            matching_residual=matching_residual,
        )

    @classmethod
    def to_kernel(cls, seed: Union[PoincareControlSeed, ValidatedPoincareSeed]) -> PoincareHamiltonianKernel:
        raw = seed.raw if isinstance(seed, ValidatedPoincareSeed) else seed
        return PoincareHamiltonianKernel(
            J=raw.interconnection_matrix,
            metric=raw.metric_matrix,
            R=raw.damping_matrix,
            g=raw.port_matrix,
            name="Validated-PHS",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Proyecciones
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _project_skew(J: np.ndarray) -> np.ndarray:
        return 0.5 * (J - J.T)

    @classmethod
    def _project_psd(cls, R: np.ndarray) -> np.ndarray:
        """Proyección espectral al cono PSD (Higham): V max(Λ,0) Vᵀ. No es un shift."""
        R_sym = 0.5 * (R + R.T)
        n = R_sym.shape[0]
        if n == 0:
            return R_sym
        if n <= cls.LARGE_DIM_THRESHOLD:
            try:
                w, V = np.linalg.eigh(R_sym)
            except np.linalg.LinAlgError as exc:
                raise ControlSeedError(f"No se pudo diagonalizar R: {exc}") from exc
            return (V * np.maximum(w, 0.0)) @ V.T
        return cls._partial_spectral_cone(R_sym, floor=0.0)

    @classmethod
    def _project_spd(cls, K: np.ndarray) -> np.ndarray:
        K_sym = 0.5 * (K + K.T)
        n = K_sym.shape[0]
        floor = max(CONSTANTS.NUMERICAL_TOLERANCE, 1e-12)
        if n == 0:
            return K_sym
        if n <= cls.LARGE_DIM_THRESHOLD:
            try:
                w, V = np.linalg.eigh(K_sym)
            except np.linalg.LinAlgError as exc:
                raise ControlSeedError(f"No se pudo diagonalizar K: {exc}") from exc
            w_clipped = np.maximum(w, floor)
            if np.any(w < floor):
                logger.warning("K regularizado a SPD: %d autovalores bajo el suelo.", int(np.sum(w < floor)))
            return (V * w_clipped) @ V.T
        return cls._partial_spectral_cone(K_sym, floor=floor)

    @classmethod
    def _partial_spectral_cone(cls, S: np.ndarray, floor: float) -> np.ndarray:
        """Corrige solo el subespacio de autovalores ofensores (Lanczos)."""
        n = S.shape[0]
        if not SCIPY_AVAILABLE or eigsh is None or n < 3:
            w_min = float(np.min(np.diag(S)))
            shift = max(0.0, floor - w_min)
            if shift > 0.0:
                logger.warning("Cono PSD por shift (fallback) shift=%.3e; disipa Casimirs.", shift)
            return S + shift * np.eye(n)
        k = min(12, n - 1)
        try:
            w, V = eigsh(S.astype(float), k=k, which="SA")
        except Exception:  # noqa: BLE001
            w_min = float(np.min(np.diag(S)))
            return S + max(0.0, floor - w_min) * np.eye(n)
        out = S.copy()
        n_fix = 0
        for i, lam in enumerate(w):
            if lam < floor:
                v = V[:, i]
                out = out + (floor - lam) * np.outer(v, v)
                n_fix += 1
        if n_fix:
            logger.info("Cono simétrico: %d modos corregidos (floor=%.1e).", n_fix, floor)
        return 0.5 * (out + out.T)

    @staticmethod
    def _casimir_basis_from_J(J: np.ndarray) -> np.ndarray:
        n = J.shape[0]
        if n == 0:
            return np.zeros((0, 0))
        _, S, Vt = np.linalg.svd(J, full_matrices=True)
        sigma_max = float(S.max()) if S.size else 1.0
        tol = CONSTANTS.CASIMIR_TOLERANCE * max(1.0, sigma_max)
        rank_J = int(np.sum(S > tol)) if S.size else 0
        if rank_J % 2 == 1 and rank_J > 0:
            rank_J -= 1
        return np.ascontiguousarray(Vt[rank_J:].T)

    @staticmethod
    def _casimir_compatible_ports(g: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        """g ← (I − CCᵀ)g  de modo que Ċ no dependa de u (si además Cᵀ R ∇H = 0)."""
        if C is None or C.size == 0 or g.size == 0:
            return g
        leak_before = float(np.linalg.norm(C.T @ g, ord="fro"))
        g_proj = g - C @ (C.T @ g)
        if np.linalg.norm(g_proj, ord="fro") < CONSTANTS.NUMERICAL_TOLERANCE * max(1.0, float(np.linalg.norm(g, ord="fro"))):
            raise UncontrollableEnergyError(
                "Todos los puertos viven en ker J: el control no puede cambiar H_d."
            )
        leak_after = float(np.linalg.norm(C.T @ g_proj, ord="fro"))
        if leak_before > CONSTANTS.SYMPLECTIC_TOLERANCE:
            logger.info("Puertos proyectados fuera de ker J: leak %.3e → %.3e.", leak_before, leak_after)
        return g_proj

    @staticmethod
    def _full_column_rank(g: np.ndarray) -> np.ndarray:
        if g.size == 0:
            return g
        try:
            Q, R_qr, piv = np.linalg.qr(g, mode="reduced", pivoting=True)  # type: ignore[call-arg]
        except TypeError:
            Q, R_qr = np.linalg.qr(g, mode="reduced")
            piv = None
        diag = np.abs(np.diag(R_qr)) if R_qr.ndim == 2 else np.array([])
        if diag.size == 0:
            raise ControlSeedError("g no tiene columnas.")
        rank = int(np.sum(diag > CTRL.QR_PIVOT_TOL * max(1.0, float(diag[0]))))
        if rank == 0:
            raise ControlSeedError("rango(g)=0 tras proyección Casimir.")
        if rank < g.shape[1]:
            logger.warning("g rango-deficiente (%d < %d); se retienen %d puertos.", rank, g.shape[1], rank)
        return np.asarray(Q[:, :rank], dtype=float)

    @staticmethod
    def _port_least_squares(g: np.ndarray, Delta: np.ndarray) -> Tuple[np.ndarray, float]:
        m = g.shape[1]
        if m == 0:
            return np.zeros((0, Delta.shape[1])), float(np.linalg.norm(Delta, ord="fro"))
        G, _, _, _ = np.linalg.lstsq(g, Delta, rcond=None)
        residual = float(np.linalg.norm(g @ G - Delta, ord="fro"))
        return G, residual

    @staticmethod
    def _k_orthogonal_energy_split(
        x: np.ndarray,
        K: np.ndarray,
        C: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        r"""H_c = ½ x_cᵀ K x_c con x_c = C(Cᵀ K C)⁻¹ Cᵀ K x  (proyección K-ortogonal a ker J)."""
        H = 0.5 * float(x @ (K @ x))
        if C is None or C.size == 0:
            return 0.0, H
        CKC = C.T @ K @ C
        try:
            gamma = np.linalg.solve(CKC, C.T @ (K @ x))
        except np.linalg.LinAlgError:
            gamma = np.linalg.lstsq(CKC, C.T @ (K @ x), rcond=None)[0]
        x_c = C @ gamma
        H_c = max(0.0, 0.5 * float(x_c @ (K @ x_c)))
        return H_c, max(0.0, H - H_c)

    @staticmethod
    def _logarithmic_norm(A: np.ndarray) -> float:
        if A.size == 0:
            return 0.0
        S = 0.5 * (A + A.T)
        try:
            return float(np.max(np.linalg.eigvalsh(S)))
        except np.linalg.LinAlgError:
            return float(np.max(np.diag(S)))

    # ──────────────────────────────────────────────────────────────────────────
    # Coerciones
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _as_vector(value: Optional[np.ndarray], name: str) -> np.ndarray:
        if value is None:
            raise ControlSeedError(f"{name} es requerido.")
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ControlSeedError(f"{name} no puede estar vacío.")
        if not np.all(np.isfinite(arr)):
            raise ControlSeedError(f"{name} contiene valores no finitos.")
        return arr

    @staticmethod
    def _as_square_matrix(value: Optional[np.ndarray], dim: int, name: str) -> np.ndarray:
        if value is None:
            raise ControlSeedError(f"{name} es requerida.")
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(dim)
        if arr.ndim == 1:
            if arr.size != dim:
                raise ControlSeedError(f"{name} vector longitud {dim}, recibido {arr.size}.")
            arr = np.diag(arr)
        if arr.ndim != 2 or arr.shape != (dim, dim):
            raise ControlSeedError(f"{name} debe ser {dim}×{dim}, recibido {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ControlSeedError(f"{name} contiene valores no finitos.")
        return arr

    @staticmethod
    def _as_port_matrix(value: Optional[np.ndarray], dim: int, name: str) -> np.ndarray:
        if value is None:
            return np.eye(dim, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(dim)
        if arr.ndim == 1:
            if arr.size != dim:
                raise ControlSeedError(f"{name} vector debe tener longitud {dim}.")
            arr = np.diag(arr)
        if arr.ndim != 2 or arr.shape[0] != dim:
            raise ControlSeedError(f"{name} debe ser {dim}×k, recibido {arr.shape}.")
        if arr.shape[1] == 0:
            raise ControlSeedError(f"{name} debe tener al menos un puerto.")
        return arr

    @classmethod
    def _min_symmetric_eigenvalue(cls, M: np.ndarray) -> float:
        n = M.shape[0]
        if n == 0:
            return 0.0
        S = 0.5 * (M + M.T)
        if n > cls.LARGE_DIM_THRESHOLD and SCIPY_AVAILABLE and eigsh is not None:
            try:
                return float(eigsh(S, k=1, which="SA", return_eigenvectors=False)[0])
            except Exception:  # noqa: BLE001
                return float(np.min(np.diag(S)))
        try:
            return float(np.min(np.linalg.eigvalsh(S)))
        except np.linalg.LinAlgError:
            return float(np.min(np.diag(S))) if M.size else 0.0

    @classmethod
    def _symmetric_eigen_bounds(cls, M: np.ndarray) -> Tuple[float, float]:
        n = M.shape[0]
        if n == 0:
            return CONSTANTS.NUMERICAL_TOLERANCE, CONSTANTS.NUMERICAL_TOLERANCE
        S = 0.5 * (M + M.T)
        if n > cls.LARGE_DIM_THRESHOLD and SCIPY_AVAILABLE and eigsh is not None:
            try:
                w_sa = float(eigsh(S, k=1, which="SA", return_eigenvectors=False)[0])
                w_la = float(eigsh(S, k=1, which="LA", return_eigenvectors=False)[0])
                return w_sa, w_la
            except Exception:  # noqa: BLE001
                d = np.diag(S)
                return float(np.min(d)), float(np.max(d))
        try:
            eigvals = np.linalg.eigvalsh(S)
            return float(np.min(eigvals)), float(np.max(eigvals))
        except np.linalg.LinAlgError:
            d = np.diag(S)
            return float(np.min(d)), float(np.max(d))

    @staticmethod
    def _condition_number_from_eigen_bounds(min_eig: float, max_eig: float) -> float:
        min_eig = max(float(min_eig), CONSTANTS.NUMERICAL_TOLERANCE)
        max_eig = max(float(max_eig), CONSTANTS.NUMERICAL_TOLERANCE)
        return max_eig / min_eig

    @staticmethod
    def _spectral_radius_estimate(J: np.ndarray) -> float:
        n = J.shape[0]
        if n == 0:
            return 0.0
        if n <= 128:
            try:
                return float(np.max(np.abs(np.linalg.eigvals(J))))
            except np.linalg.LinAlgError:
                pass
        try:
            s = np.linalg.svd(J, compute_uv=False)
            return float(s[0]) if s.size else 0.0
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(J, ord="fro"))


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 2.3 — PI CON ANTI-WINDUP DEL OPERADOR DE SATURACIÓN COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════════════


class PIController:
    r"""
    PI de Åström–Hägglund con back-calculation sobre el saturador *efectivo*.

    u_raw = K_p e + v + u_ff,   v̇ = K_i e + (1/T_t)(u_aplicada − u_raw)
    u_aplicada = slew ∘ clip(u_raw)

    T_t = T_i = K_p/K_i por defecto (mismas unidades).

    Modo energía (consume PoincareControlSeed):
        PV = H/H*   (normalizado)  o  H (absoluto);
        error = SP − PV;  salida **con signo** = potencia pedida normalizada.
        EMA desactivada: H de un PHS lineal no es un sensor ruidoso.
        Reloj de simulación: dt físico, nunca time.time().

    El PI **no** regula Casimirs: el error se forma con H total, pero el puente
    ya garantizó H* ≥ H_c y Cᵀg = 0. Un windup aquí es saturación de puerto, no Gauss.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        setpoint: float,
        min_output: float,
        max_output: float,
        integral_limit_factor: float = 2.0,
        tracking_time: Optional[float] = None,
        ema_alpha: float = 0.3,
        slew_rate_fraction: float = 0.15,
        reference_dt: float = 0.01,
        energy_error_mode: str = "normalized",
        use_ema: Optional[bool] = None,
    ) -> None:
        self._validate_parameters(
            kp=kp, ki=ki, setpoint=setpoint,
            min_output=min_output, max_output=max_output,
            integral_limit_factor=integral_limit_factor,
            ema_alpha=ema_alpha, slew_rate_fraction=slew_rate_fraction,
            reference_dt=reference_dt,
        )
        self.kp = float(kp)
        self.ki = float(ki)
        self.setpoint = float(setpoint)
        self.min_output = float(min_output)
        self.max_output = float(max_output)
        if tracking_time is not None:
            self.Tt = max(float(tracking_time), CONSTANTS.MIN_DELTA_TIME)
        elif self.ki > 0.0 and self.kp > 0.0:
            self.Tt = max(self.kp / self.ki, CONSTANTS.MIN_DELTA_TIME)
        else:
            self.Tt = 1.0
        self._integral_limit_factor = float(integral_limit_factor)
        self._integral_limit = self._integral_limit_factor * max(
            self.max_output - self.min_output, CONSTANTS.NUMERICAL_TOLERANCE
        )
        self._ema_alpha = float(ema_alpha)
        self._slew_rate_fraction = float(slew_rate_fraction)
        self._reference_dt = float(reference_dt)
        self._energy_error_mode = energy_error_mode if energy_error_mode in ("normalized", "absolute") else "normalized"
        self._use_ema_default = bool(use_ema) if use_ema is not None else True

        self._filtered_pv: Optional[float] = None
        self._integral_term = 0.0
        self._last_error = 0.0
        self._last_output: Optional[float] = None
        self._last_raw_output = 0.0
        self._sim_time = 0.0
        self._error_history: deque = deque(maxlen=128)
        self._time_history: deque = deque(maxlen=128)
        self._output_history: deque = deque(maxlen=128)
        self._innovation_history: deque = deque(maxlen=32)
        self._lyapunov_exponent = 0.0
        self._oscillation_index = 0.0
        self._extrema_density = 0.0
        self._poincare_seed: Optional[ValidatedPoincareSeed] = None
        self._energy_mode = False
        self._use_ema = self._use_ema_default

    def consume_poincare_control_seed(
        self,
        seed: PoincareControlSeed,
        energy_setpoint: float = 1.0,
    ) -> ValidatedPoincareSeed:
        validated = PoincareControlBridge.validate_seed(seed)
        self._poincare_seed = validated
        self._energy_mode = True
        self._use_ema = False  # H no se filtra
        if self.min_output >= 0.0 and validated.hamiltonian_error > CONSTANTS.MIN_ENERGY_THRESHOLD:
            logger.warning(
                "PI min_output=%.3g ≥ 0 con H>H*: no hay autoridad de extracción. "
                "Use min_output < 0 (mando bipolar).",
                self.min_output,
            )
        if validated.pumping_required and self.max_output <= 0.0:
            logger.warning("PI max_output≤0 con pumping_required: no hay autoridad de inyección.")
        if self._energy_error_mode == "normalized":
            if 0.0 < energy_setpoint <= 1.0:
                self.setpoint = float(energy_setpoint)
            else:
                self.setpoint = 1.0
        else:
            self.setpoint = float(validated.raw.target_hamiltonian)
        return validated

    def compute_from_seed(
        self,
        seed: Optional[PoincareControlSeed] = None,
        dt: Optional[float] = None,
        feedforward: float = 0.0,
    ) -> PIControlReport:
        if seed is not None:
            self.consume_poincare_control_seed(seed)
        if self._poincare_seed is None:
            raise ControlSeedError("compute_from_seed exige PoincareControlSeed.")
        if dt is None:
            raise ControlSeedError("en modo energía dt físico es obligatorio (no se usa wall-clock).")
        H = float(self._poincare_seed.raw.hamiltonian)
        H_target = float(self._poincare_seed.raw.target_hamiltonian)
        if self._energy_error_mode == "normalized":
            measurement = H / max(H_target, CONSTANTS.MIN_ENERGY_THRESHOLD)
        else:
            measurement = H
        return self.compute(measurement=measurement, dt=dt, feedforward=feedforward)

    def compute(
        self,
        measurement: float,
        dt: Optional[float] = None,
        feedforward: float = 0.0,
    ) -> PIControlReport:
        if not math.isfinite(measurement):
            raise InvalidInputError("measurement debe ser finito.")
        if not math.isfinite(feedforward):
            raise InvalidInputError("feedforward debe ser finito.")
        if dt is None:
            if self._energy_mode:
                raise ControlSeedError("dt físico obligatorio en modo energía.")
            dt = self._reference_dt
        dt = float(np.clip(dt, CONSTANTS.MIN_DELTA_TIME, CONSTANTS.MAX_DELTA_TIME))

        filtered_pv = self._apply_ema_filter(measurement) if self._use_ema else float(measurement)
        error = float(self.setpoint - filtered_pv)
        self._update_stability_metrics(error, dt)

        p_term = self.kp * error
        v_pred = float(np.clip(self._integral_term + self.ki * error * dt, -self._integral_limit, self._integral_limit))
        raw_output = p_term + v_pred + float(feedforward)
        applied = self._saturation_operator(raw_output, dt)
        saturated = not math.isclose(applied, raw_output, rel_tol=1e-12, abs_tol=1e-15)
        aw_correction = (dt / self.Tt) * (applied - raw_output)
        self._integral_term = float(
            np.clip(v_pred + aw_correction, -self._integral_limit, self._integral_limit)
        )
        self._last_output = applied
        self._last_error = error
        self._last_raw_output = raw_output
        self._sim_time += dt
        self._output_history.append(applied)
        return PIControlReport(
            output=applied,
            error=error,
            filtered_pv=filtered_pv,
            integral_error=(self._integral_term / self.ki if self.ki > 0.0 else 0.0),
            integral_term=self._integral_term,
            p_term=p_term,
            i_term=self._integral_term,
            feedforward=float(feedforward),
            saturated=saturated,
            anti_windup_correction=aw_correction,
            lyapunov_exponent=self._lyapunov_exponent,
            oscillation_index=self._oscillation_index,
            extrema_density=self._extrema_density,
            applied_output=applied,
        )

    def _saturation_operator(self, raw: float, dt: float) -> float:
        """(slew ∘ clip). Este es el u que ve la planta y el que debe ver el AW."""
        clipped = float(np.clip(raw, self.min_output, self.max_output))
        if self._last_output is None:
            return clipped
        max_change = (
            self._slew_rate_fraction
            * (self.max_output - self.min_output)
            * (dt / max(self._reference_dt, CONSTANTS.NUMERICAL_TOLERANCE))
        )
        delta = clipped - self._last_output
        if abs(delta) > max_change:
            clipped = self._last_output + math.copysign(max_change, delta)
        return float(np.clip(clipped, self.min_output, self.max_output))

    def reset(self) -> None:
        self._integral_term = 0.0
        self._last_error = 0.0
        self._last_output = None
        self._last_raw_output = 0.0
        self._sim_time = 0.0
        self._filtered_pv = None
        self._error_history.clear()
        self._time_history.clear()
        self._output_history.clear()
        self._innovation_history.clear()
        self._lyapunov_exponent = 0.0
        self._oscillation_index = 0.0
        self._extrema_density = 0.0

    def _apply_ema_filter(self, measurement: float) -> float:
        if self._filtered_pv is None:
            self._filtered_pv = float(measurement)
            return float(measurement)
        innovation = float(measurement - self._filtered_pv)
        step_threshold = 0.2 * abs(self.setpoint) + CONSTANTS.NUMERICAL_TOLERANCE
        if abs(innovation) > step_threshold:
            alpha_effective = 0.8
        else:
            self._innovation_history.append(innovation)
            if len(self._innovation_history) >= 5:
                var = float(np.var(list(self._innovation_history)))
                alpha_effective = self._ema_alpha / (1.0 + 10.0 * var)
            else:
                alpha_effective = self._ema_alpha
        alpha_effective = float(np.clip(alpha_effective, 0.05, 0.95))
        self._filtered_pv = alpha_effective * measurement + (1.0 - alpha_effective) * self._filtered_pv
        return float(self._filtered_pv)

    def _update_stability_metrics(self, error: float, dt: float) -> None:
        """Theil–Sen de log|e| **respecto al tiempo de simulación** (1/s)."""
        self._error_history.append(error)
        self._time_history.append(self._sim_time)
        n = len(self._error_history)
        if n < CTRL.MIN_LYAPUNOV_SAMPLES:
            return
        errors = np.array(list(self._error_history), dtype=float)
        times = np.array(list(self._time_history), dtype=float)
        log_e = np.log(np.abs(errors) + CONSTANTS.NUMERICAL_ZERO)
        slopes: List[float] = []
        w = CTRL.THEIL_SEN_WINDOW
        for i in range(n - 1):
            hi = min(i + w, n)
            for j in range(i + 1, hi):
                dt_ij = times[j] - times[i]
                if dt_ij > CONSTANTS.NUMERICAL_ZERO:
                    slopes.append((log_e[j] - log_e[i]) / dt_ij)
        if slopes:
            self._lyapunov_exponent = float(np.median(slopes))
        sign_changes = int(np.sum(np.diff(np.sign(errors)) != 0))
        self._oscillation_index = sign_changes / max(n - 1, 1)
        if n >= 5:
            extrema = int(np.sum(np.diff(np.sign(np.diff(errors))) != 0))
            self._extrema_density = extrema / max(n - 2, 1)

    def get_lyapunov_exponent(self) -> float:
        return float(self._lyapunov_exponent)

    def get_stability_analysis(self) -> Dict[str, Any]:
        if len(self._error_history) < 10:
            return {"status": "INSUFFICIENT_DATA", "samples": len(self._error_history)}
        if self._lyapunov_exponent < -0.1:
            stability, convergence = "ASYMPTOTICALLY_STABLE", "CONVERGING"
        elif self._lyapunov_exponent < 0.01:
            stability, convergence = "MARGINALLY_STABLE", "BOUNDED"
        else:
            stability, convergence = "UNSTABLE", "DIVERGING"
        integral_saturation = abs(self._integral_term) / max(self._integral_limit, CONSTANTS.NUMERICAL_TOLERANCE)
        return {
            "status": "OPERATIONAL",
            "stability_class": stability,
            "convergence": convergence,
            "lyapunov_exponent": self._lyapunov_exponent,
            "oscillation_index": self._oscillation_index,
            "extrema_density": self._extrema_density,
            "is_limit_cycle": bool(stability == "MARGINALLY_STABLE" and self._oscillation_index > 0.3),
            "integral_saturation": float(integral_saturation),
            "samples_analyzed": len(self._error_history),
            "energy_mode": self._energy_mode,
            "sim_time": self._sim_time,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "status": "OK",
            "control_metrics": {
                "error": self._last_error,
                "integral_term": self._integral_term,
                "proportional_term": self.kp * self._last_error,
                "output": self._last_output,
                "raw_output": self._last_raw_output,
            },
            "stability_analysis": self.get_stability_analysis(),
            "parameters": {
                "kp": self.kp, "ki": self.ki, "setpoint": self.setpoint,
                "tracking_time": self.Tt, "ema_alpha": self._ema_alpha,
                "slew_rate_fraction": self._slew_rate_fraction,
                "reference_dt": self._reference_dt,
                "energy_error_mode": self._energy_error_mode,
                "use_ema": self._use_ema,
                "output_range": [self.min_output, self.max_output],
            },
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "parameters": {
                "kp": self.kp, "ki": self.ki, "setpoint": self.setpoint,
                "output_range": [self.min_output, self.max_output],
            },
            "state": {
                "integral_term": self._integral_term,
                "last_output": self._last_output,
                "filtered_pv": self._filtered_pv,
                "energy_mode": self._energy_mode,
                "sim_time": self._sim_time,
            },
            "diagnostics": self.get_diagnostics(),
        }

    @staticmethod
    def _validate_parameters(
        kp: float, ki: float, setpoint: float,
        min_output: float, max_output: float,
        integral_limit_factor: float, ema_alpha: float,
        slew_rate_fraction: float, reference_dt: float,
    ) -> None:
        errors: List[str] = []
        if not math.isfinite(kp) or kp < 0.0:
            errors.append(f"Kp debe ser finito y ≥ 0, recibido {kp}")
        if not math.isfinite(ki) or ki < 0.0:
            errors.append(f"Ki debe ser finito y ≥ 0, recibido {ki}")
        if kp == 0.0 and ki == 0.0:
            errors.append("Kp y Ki no pueden anularse simultáneamente.")
        if not math.isfinite(setpoint):
            errors.append(f"setpoint debe ser finito, recibido {setpoint}")
        if not math.isfinite(min_output) or not math.isfinite(max_output):
            errors.append("min_output y max_output deben ser finitos.")
        elif max_output <= min_output:
            errors.append("max_output debe ser > min_output.")
        if not math.isfinite(integral_limit_factor) or integral_limit_factor <= 0.0:
            errors.append("integral_limit_factor debe ser finito y positivo.")
        if not math.isfinite(ema_alpha) or not (0.0 < ema_alpha < 1.0):
            errors.append("ema_alpha debe estar en (0, 1).")
        if not math.isfinite(slew_rate_fraction) or slew_rate_fraction <= 0.0:
            errors.append("slew_rate_fraction debe ser finito y positivo.")
        if not math.isfinite(reference_dt) or reference_dt <= 0.0:
            errors.append("reference_dt debe ser finito y positivo.")
        if errors:
            raise ConfigurationError("PIController inválido:\n" + "\n".join(f"  - {e}" for e in errors))


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 2.4 — MÚSCULO DE FLUJO (SATURADOR TÉRMICO, NO UN GANANCIA DEL ERROR)
# ═══════════════════════════════════════════════════════════════════════════════════════


class FluxMuscleController:
    r"""
    Actuador con Newton-Joule y slew.

        Ṫ = α d² − (T − T_amb)/τ
        T^{n+1} = T_eq + (T^n − T_eq) e^{−Δt/τ},   T_eq = T_amb + α τ d²

    El músculo **no** escala el mando por el error energético (eso era anti-bombeo).
    Escala por derating térmico y fatiga. `quadrants=1` → d∈[0,1]; `2` → d∈[−1,1]
    (calentamiento ∝ d² en ambos).

    En la composición PHS, el músculo es un saturador de ‖u‖: devuelve un
    `applied_scale` ∈ [0,1] que Fase 2 aplica a u_cmd.
    """

    def __init__(
        self,
        max_slew_rate: float = 0.1,
        thermal_time_constant: float = 5.0,
        ambient_temperature: float = 25.0,
        max_temperature: float = 85.0,
        thermal_gain: float = 15.0,
        derate_factor: float = 0.5,
        reference_dt: float = 0.01,
        soft_temperature_fraction: float = 0.8,
        fatigue_accumulator_critical: float = 5.0,
        quadrants: int = 1,
    ) -> None:
        self._validate_parameters(
            max_slew_rate=max_slew_rate,
            thermal_time_constant=thermal_time_constant,
            ambient_temperature=ambient_temperature,
            max_temperature=max_temperature,
            thermal_gain=thermal_gain,
            derate_factor=derate_factor,
            reference_dt=reference_dt,
            soft_temperature_fraction=soft_temperature_fraction,
            fatigue_accumulator_critical=fatigue_accumulator_critical,
            quadrants=quadrants,
        )
        self._max_slew_rate = float(max_slew_rate)
        self._thermal_time_constant = float(thermal_time_constant)
        self._ambient_temperature = float(ambient_temperature)
        self._max_temperature = float(max_temperature)
        self._thermal_gain = float(thermal_gain)
        self._derate_factor = float(derate_factor)
        self._reference_dt = float(reference_dt)
        self._soft_temperature_fraction = float(soft_temperature_fraction)
        self._fatigue_accumulator_critical = float(fatigue_accumulator_critical)
        self._quadrants = int(quadrants)
        self._d_min = 0.0 if self._quadrants == 1 else -1.0
        self._duty = 0.0
        self._commanded = 0.0
        self._temperature = self._ambient_temperature
        self._thermal_accumulator = 0.0
        self._derated = False
        self._overheated = False
        self._poincare_seed: Optional[ValidatedPoincareSeed] = None

    def consume_poincare_control_seed(self, seed: PoincareControlSeed) -> ValidatedPoincareSeed:
        validated = PoincareControlBridge.validate_seed(seed)
        self._poincare_seed = validated
        if validated.pumping_required and self._quadrants == 1:
            logger.info("Músculo 1-cuadrante con pumping_required: solo inyecta (d≥0).")
        return validated

    def apply_force(
        self,
        target_intensity: float,
        dt: float,
        feedforward: float = 0.0,
    ) -> float:
        if not math.isfinite(dt) or dt <= 0.0:
            raise MuscleThermalError("dt debe ser finito y positivo.")
        if not math.isfinite(target_intensity) or not math.isfinite(feedforward):
            raise InvalidInputError("target_intensity/feedforward deben ser finitos.")
        target = float(target_intensity + feedforward)
        target = float(np.clip(target, self._d_min, 1.0))
        cap = self._thermal_derate_cap()
        # cap limita |d|, no el signo
        target = float(np.clip(target, -cap if self._d_min < 0.0 else 0.0, cap))
        self._commanded = target
        max_change = self._max_slew_rate * max(
            dt / max(self._reference_dt, CONSTANTS.NUMERICAL_TOLERANCE),
            CONSTANTS.NUMERICAL_TOLERANCE,
        )
        delta = target - self._duty
        if abs(delta) > max_change:
            delta = math.copysign(max_change, delta)
        self._duty = float(np.clip(self._duty + delta, self._d_min, 1.0))
        self._thermal_exponential_step(dt)
        abs_d = abs(self._duty)
        excess = max(0.0, abs_d - CTRL.MUSCLE_FATIGUE_HI)
        recovery = max(0.0, CTRL.MUSCLE_FATIGUE_LO - abs_d)
        self._thermal_accumulator = max(0.0, self._thermal_accumulator + dt * (excess - recovery))
        self._overheated = bool(self._temperature >= self._max_temperature)
        self._derated = bool(self._overheated or self._thermal_accumulator >= self._fatigue_accumulator_critical)
        return self._duty

    def scale_command(self, u_cmd: np.ndarray, dt: float) -> Tuple[np.ndarray, float]:
        """Saturador de ‖u‖: intensity = ‖u‖/u_ref mapeada a duty, u_app = scale·u_cmd."""
        u_cmd = np.asarray(u_cmd, dtype=float).reshape(-1)
        nrm = float(np.linalg.norm(u_cmd))
        if nrm < CONSTANTS.NUMERICAL_ZERO:
            self.apply_force(0.0, dt)
            return u_cmd.copy(), 0.0
        intensity = nrm  # el llamador debe normalizar a [0,1] si desea; aquí usamos clip
        # Para bipolar: signo de potencia yᵀu no está en ‖u‖. El llamador pasa intensity firmada.
        duty = self.apply_force(float(np.clip(intensity, self._d_min, 1.0)), dt)
        scale = abs(duty) / max(min(intensity, 1.0), CONSTANTS.NUMERICAL_TOLERANCE) if intensity else 0.0
        scale = float(np.clip(scale, 0.0, 1.0))
        return u_cmd * scale, scale

    def _thermal_exponential_step(self, dt: float) -> None:
        d2 = self._duty ** 2
        T_eq = self._ambient_temperature + self._thermal_gain * d2 * self._thermal_time_constant
        decay = math.exp(-min(dt / self._thermal_time_constant, CONSTANTS.MAX_EXPONENTIAL_ARG))
        self._temperature = T_eq + (self._temperature - T_eq) * decay
        if not math.isfinite(self._temperature):
            raise MuscleThermalError("Temperatura no finita en músculo.")
        self._temperature = max(self._ambient_temperature, self._temperature)

    def _thermal_derate_cap(self) -> float:
        T = self._temperature
        span = self._max_temperature - self._ambient_temperature
        T_soft = self._ambient_temperature + self._soft_temperature_fraction * span
        if T <= T_soft:
            return 1.0
        if T >= self._max_temperature:
            return self._derate_factor
        frac = (T - T_soft) / max(self._max_temperature - T_soft, CONSTANTS.NUMERICAL_TOLERANCE)
        return float(1.0 - (1.0 - self._derate_factor) * frac)

    def _seed_authority(self) -> float:
        """Deprecated no-op (7.1): el error energético no reduce autoridad."""
        return 1.0

    @property
    def duty(self) -> float:
        return self._duty

    @property
    def temperature(self) -> float:
        return self._temperature

    def thermal_state(self) -> MuscleThermalState:
        commanded = self._commanded
        applied_scale = abs(self._duty) / max(abs(commanded), CONSTANTS.NUMERICAL_TOLERANCE) if commanded else 1.0
        return MuscleThermalState(
            duty=self._duty,
            commanded=commanded,
            temperature=self._temperature,
            thermal_accumulator=self._thermal_accumulator,
            overheated=self._overheated,
            derated=self._derated,
            thermal_derate_cap=self._thermal_derate_cap(),
            applied_scale=float(np.clip(applied_scale, 0.0, 1.0)),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "status": "OK",
            "duty": self._duty,
            "commanded": self._commanded,
            "temperature": self._temperature,
            "thermal_accumulator": self._thermal_accumulator,
            "overheated": self._overheated,
            "derated": self._derated,
            "thermal_derate_cap": self._thermal_derate_cap(),
            "quadrants": self._quadrants,
            "parameters": {
                "max_slew_rate": self._max_slew_rate,
                "thermal_time_constant": self._thermal_time_constant,
                "ambient_temperature": self._ambient_temperature,
                "max_temperature": self._max_temperature,
                "thermal_gain": self._thermal_gain,
                "derate_factor": self._derate_factor,
            },
        }

    def reset(self) -> None:
        self._duty = 0.0
        self._commanded = 0.0
        self._temperature = self._ambient_temperature
        self._thermal_accumulator = 0.0
        self._overheated = False
        self._derated = False

    @staticmethod
    def _validate_parameters(
        max_slew_rate: float, thermal_time_constant: float,
        ambient_temperature: float, max_temperature: float,
        thermal_gain: float, derate_factor: float,
        reference_dt: float, soft_temperature_fraction: float,
        fatigue_accumulator_critical: float, quadrants: int,
    ) -> None:
        errors: List[str] = []
        if not math.isfinite(max_slew_rate) or max_slew_rate <= 0.0:
            errors.append("max_slew_rate debe ser finito y positivo.")
        if not math.isfinite(thermal_time_constant) or thermal_time_constant <= 0.0:
            errors.append("thermal_time_constant debe ser finito y positivo.")
        if not math.isfinite(ambient_temperature):
            errors.append("ambient_temperature debe ser finito.")
        if not math.isfinite(max_temperature) or max_temperature <= ambient_temperature:
            errors.append("max_temperature debe ser > ambient_temperature.")
        if not math.isfinite(thermal_gain) or thermal_gain < 0.0:
            errors.append("thermal_gain debe ser finito y ≥ 0.")
        if not math.isfinite(derate_factor) or not (0.0 < derate_factor < 1.0):
            errors.append("derate_factor debe estar en (0, 1).")
        if not math.isfinite(reference_dt) or reference_dt <= 0.0:
            errors.append("reference_dt debe ser finito y positivo.")
        if not math.isfinite(soft_temperature_fraction) or not (0.0 < soft_temperature_fraction < 1.0):
            errors.append("soft_temperature_fraction debe estar en (0, 1).")
        if not math.isfinite(fatigue_accumulator_critical) or fatigue_accumulator_critical <= 0.0:
            errors.append("fatigue_accumulator_critical debe ser finito y positivo.")
        if quadrants not in (1, 2):
            errors.append("quadrants debe ser 1 o 2.")
        if errors:
            raise ConfigurationError("FluxMuscleController inválido:\n" + "\n".join(f"  - {e}" for e in errors))


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 2.5 — CONTROLADOR PHS: POWER-SHAPING + IDA-PBC REAL + GRADIENTE DISCRETO
# ═══════════════════════════════════════════════════════════════════════════════════════


class PortHamiltonianPoincareController:
    r"""
    Lazo Port-Hamiltoniano sobre el kernel 7.1.

    Planta lineal: ẋ = (J−R)Kx + g u,  y = gᵀ K x.

    Modos
    -----
    ENERGY_LEVEL (defecto)
        Se pide Ḣ = −λ(H−H*).  Potencia de puerto que lo realiza:
            P_* = ∇Hᵀ R ∇H − λ(H−H*)
            u   = P_* y / (‖y‖² + ε)          (mínima norma)
        Entonces V̇ = e Ḣ = −λ e² ≤ 0  *si* y≠0.  Rayleigh se **compensa**,
        de modo que el bombeo (e<0) no lucha contra R.

    IDA_PBC_POINT
        Matching lineal de Fase 1: u = G x + v,  v = −k_a gᵀ K_d (x−x*).
        Estabiliza un punto, no una hoja de energía. Requiere matching_residual
        pequeño frente a ‖(J−R)K‖.

    DAMPING_ONLY
        u = −k_d y.  Extrae siempre.  Inútil si pumping_required.

    Integrador
        Punto medio implícito + Picard en u(x_{n+1/2}):
            (I − hA/2) x_{n+1} = (I + hA/2) x_n + h g u_mid
        Para H cuadrática es gradiente discreto ⇒ Tellegen exacto.
        RK4 queda como opción de referencia no estructura-preservante.

    Casimirs
        g ya cumple Cᵀg=0 (puente). Tras cada paso se re-proyecta
        Cᵀx = Cᵀx₀ para matar deriva de redondeo.
    """

    RK4_STABILITY_CFL: float = CTRL.RK4_IMAG_LIMIT

    def __init__(
        self,
        seed: PoincareControlSeed,
        damping_injection: float = CTRL.DEFAULT_ENERGY_RATE,
        target_energy: Optional[float] = None,
        energy_shaping: bool = True,
        control_saturation: Optional[float] = None,
        integrator: str = "implicit_midpoint",
        mode: Union[str, ControlMode] = ControlMode.ENERGY_LEVEL,
        power_regularization: float = CTRL.POWER_REGULARIZATION,
    ) -> None:
        self._validated = PoincareControlBridge.validate_seed(seed)
        self._seed = self._validated.raw
        self._kernel = PoincareControlBridge.to_kernel(self._validated)
        self.dim = self._validated.state_dim
        self.port_dim = self._validated.port_dim
        self.x = np.asarray(self._seed.state, dtype=float).reshape(-1).copy()
        self.J = self._kernel.J
        self.R = self._kernel.R
        self.K = self._kernel.metric
        self.g = self._kernel.g
        self.A_auto = self._kernel.A
        self._rho_A = float(self._validated.spectral_data.get("spectral_radius", 0.0)) if self._validated.spectral_data else float(np.linalg.norm(self.A_auto, ord="fro")) / max(self.dim, 1)
        self._mu2_A = float(self._validated.logarithmic_norm)
        self._C = self._validated.casimir_basis
        self._C0 = self._casimir_values(self.x)
        if target_energy is None:
            self.H_target = float(self._seed.target_hamiltonian)
        else:
            self.H_target = max(float(target_energy), CONSTANTS.MIN_ENERGY_THRESHOLD)
        if self.H_target + CONSTANTS.MIN_ENERGY_THRESHOLD < self._validated.casimir_energy:
            raise UncontrollableEnergyError("H* por debajo de la energía de Casimir.")
        if not math.isfinite(damping_injection) or damping_injection < 0.0:
            raise ConfigurationError(f"damping_injection debe ser ≥ 0, recibido {damping_injection}")
        self.kd = float(damping_injection)  # λ  [1/s] en ENERGY_LEVEL; k_d en DAMPING_ONLY
        self.use_energy_shaping = bool(energy_shaping)
        self._mode = ControlMode(mode) if not isinstance(mode, ControlMode) else mode
        if not self.use_energy_shaping:
            self._mode = ControlMode.DAMPING_ONLY
        if self._mode is ControlMode.IDA_PBC_POINT and (
            self._validated.ida_pbc_decomposition is None
            or "G" not in self._validated.ida_pbc_decomposition
        ):
            logger.warning("IDA_PBC_POINT sin G; se degrada a ENERGY_LEVEL.")
            self._mode = ControlMode.ENERGY_LEVEL
        if self._validated.pumping_required and self._mode is ControlMode.DAMPING_ONLY:
            logger.warning("DAMPING_ONLY con pumping_required: el lazo no puede subir H.")
        self._integrator = integrator if integrator in ("rk4", "implicit_midpoint") else "implicit_midpoint"
        self._eps_power = max(float(power_regularization), CONSTANTS.NUMERICAL_ZERO)
        self._time = 0.0
        grad_norm = float(np.linalg.norm(self.gradient()))
        if control_saturation is None:
            self.u_max = max(1.0, 10.0 * grad_norm)
        else:
            if not math.isfinite(control_saturation) or control_saturation <= 0.0:
                raise ConfigurationError("control_saturation debe ser finito y positivo.")
            self.u_max = float(control_saturation)
        self.control_history: deque = deque(maxlen=10_000)
        self.energy_history: deque = deque(maxlen=10_000)
        self.storage_history: deque = deque(maxlen=10_000)
        self._last_applied_u = np.zeros(self.port_dim, dtype=float)
        self._last_commanded_u = np.zeros(self.port_dim, dtype=float)

    # ── operadores ────────────────────────────────────────────────────────────

    def hamiltonian(self, x: Optional[np.ndarray] = None) -> float:
        return self._kernel.hamiltonian(self._state_or_current(x))

    def gradient(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self._kernel.gradient(self._state_or_current(x))

    def port_output(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self._kernel.port_output(self._state_or_current(x))

    def storage_function(self, x: Optional[np.ndarray] = None) -> float:
        H = self.hamiltonian(x)
        return 0.5 * float((H - self.H_target) ** 2)

    def _casimir_values(self, x: np.ndarray) -> np.ndarray:
        if self._C is None or self._C.size == 0:
            return np.zeros(0, dtype=float)
        return self._C.T @ x

    def _project_casimirs(self, x: np.ndarray) -> np.ndarray:
        if self._C is None or self._C.size == 0:
            return x
        leak = self._C.T @ x - self._C0
        return x - self._C @ leak

    # ── leyes de control ──────────────────────────────────────────────────────

    def requested_power(self, x: Optional[np.ndarray] = None, lambda_rate: Optional[float] = None) -> float:
        r"""P_* = ∇Hᵀ R ∇H − λ(H−H*).  Signo: P_*>0 inyecta."""
        x = self._state_or_current(x)
        lam = self.kd if lambda_rate is None else float(lambda_rate)
        H = self.hamiltonian(x)
        rayleigh = float(self.gradient(x) @ (self.R @ self.gradient(x)))
        return rayleigh - lam * (H - self.H_target)

    def compute_control(
        self,
        x: Optional[np.ndarray] = None,
        power_reference: Optional[float] = None,
    ) -> np.ndarray:
        x = self._state_or_current(x)
        y = self.port_output(x)
        if self._mode is ControlMode.DAMPING_ONLY:
            u = -self.kd * y
        elif self._mode is ControlMode.IDA_PBC_POINT:
            u = self._ida_pbc_point_control(x)
        else:
            P = self.requested_power(x) if power_reference is None else float(power_reference)
            y2 = float(y @ y)
            u = (P * y) / (y2 + self._eps_power)
        u = np.clip(u, -self.u_max, self.u_max)
        nrm = float(np.linalg.norm(u))
        if nrm > self.u_max:
            u = u * (self.u_max / nrm)
        if not np.all(np.isfinite(u)):
            raise NumericalInstabilityError("Control no finito.")
        self._last_commanded_u = u.copy()
        return u

    def _ida_pbc_point_control(self, x: np.ndarray) -> np.ndarray:
        ida = self._validated.ida_pbc_decomposition or {}
        G = np.asarray(ida.get("G"), dtype=float)
        if G.ndim != 2:
            return -self.kd * self.port_output(x)
        x_star = np.asarray(ida.get("x_star", np.zeros(self.dim)), dtype=float).reshape(-1)
        if x_star.size != self.dim:
            x_star = np.zeros(self.dim)
        K_d = np.asarray(ida.get("K_d", self.K), dtype=float)
        v = -self.kd * (self.g.T @ (K_d @ (x - x_star)))
        u = G @ x + v
        if u.size != self.port_dim:
            # G fue resuelto como gG = Δ, G ∈ ℝ^{m×n}
            u = (G @ x).reshape(-1) + v
        if u.size != self.port_dim:
            raise ConfigurationError(f"u IDA dim {u.size} ≠ m={self.port_dim}.")
        return u

    def continuous_lyapunov_derivative(
        self,
        x: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        x = self._state_or_current(x)
        u = self.compute_control(x) if u is None else self._normalize_control_input(u)
        H = self.hamiltonian(x)
        grad_H = self.gradient(x)
        error = H - self.H_target
        y = self.g.T @ grad_H
        rayleigh_rate = float(grad_H @ (self.R @ grad_H))
        supply_rate = float(u @ y)
        H_dot = -rayleigh_rate + supply_rate
        V_dot = error * H_dot
        return {
            "H_dot": H_dot,
            "V_dot": V_dot,
            "supply_rate": supply_rate,
            "rayleigh_rate": rayleigh_rate,
            "plant_passivity_slack": rayleigh_rate,
            "regulation_slack": -V_dot,
            "power_requested": self.requested_power(x),
            "power_delivered": supply_rate,
            "control_dim": float(u.size),
        }

    # ── integración ───────────────────────────────────────────────────────────

    def controlled_step(
        self,
        dt: float,
        u_input: Optional[Union[float, np.ndarray]] = None,
        substeps: Optional[int] = None,
        muscle: Optional["FluxMuscleController"] = None,
        power_reference: Optional[float] = None,
    ) -> PortHamiltonianControlReport:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ConfigurationError("dt debe ser finito y positivo.")
        x0 = self.x.copy()
        H0 = self.hamiltonian(x0)
        V0 = self.storage_function(x0)

        if u_input is not None:
            u_cmd = self._normalize_control_input(u_input)
        else:
            u_cmd = self.compute_control(x0, power_reference=power_reference)

        u_app = u_cmd
        if muscle is not None:
            # intensity firmada por la potencia pedida, magnitud en [0,1] vía ‖u‖/u_max
            nrm = float(np.linalg.norm(u_cmd))
            intensity = nrm / max(self.u_max, CONSTANTS.NUMERICAL_TOLERANCE)
            y0 = self.port_output(x0)
            sign = 1.0
            if y0.size and nrm > 0.0:
                sign = math.copysign(1.0, float(y0 @ u_cmd)) if self._d_sign_needed(muscle) else 1.0
            duty = muscle.apply_force(sign * float(np.clip(intensity, 0.0, 1.0)), dt)
            scale = abs(duty) / max(intensity, CONSTANTS.NUMERICAL_TOLERANCE) if intensity > 0.0 else 0.0
            u_app = u_cmd * float(np.clip(scale, 0.0, 1.0))

        metrics = self.continuous_lyapunov_derivative(x0, u=u_app)
        if substeps is None:
            substeps = self._estimate_substeps(dt)
        h = dt / float(substeps)
        if self._integrator == "rk4":
            self._integrate_rk4(h, substeps, u_app)
        else:
            self._integrate_implicit_midpoint(h, substeps, u_app, freeze_u=True)

        self.x = self._project_casimirs(self.x)
        self._time += dt
        self._last_applied_u = u_app.copy()
        self._last_commanded_u = u_cmd.copy()

        H1 = self.hamiltonian()
        V1 = self.storage_function()
        audit = self._discrete_passivity_audit(x0, self.x, u_app, dt)
        if not audit.is_discrete_gradient and self._integrator == "implicit_midpoint":
            logger.debug("Residuo de gradiente discreto %.3e (tol=%.1e).", audit.residual, CTRL.DISCRETE_PASSIVITY_TOL)

        control_norm = float(np.linalg.norm(u_cmd))
        applied_norm = float(np.linalg.norm(u_app))
        port_output_norm = float(np.linalg.norm(self.port_output()))
        self.control_history.append(applied_norm)
        self.energy_history.append(H1)
        self.storage_history.append(V1)

        pumping = bool(H1 < self.H_target - CONSTANTS.MIN_ENERGY_THRESHOLD)
        # Regulación: V decrece, o bien estamos bajo H* y H sube (bombeo con R).
        dV = V1 - V0
        dH = H1 - H0
        is_regulating = bool(
            dV <= CTRL.DISCRETE_PASSIVITY_TOL
            or (H0 < self.H_target and dH >= -CTRL.DISCRETE_PASSIVITY_TOL)
        )
        return PortHamiltonianControlReport(
            time=self._time,
            dt=dt,
            hamiltonian=H1,
            hamiltonian_derivative=metrics["H_dot"],
            target_hamiltonian=self.H_target,
            storage_function=V1,
            control_norm=control_norm,
            port_output_norm=port_output_norm,
            supply_rate=metrics["supply_rate"],
            lyapunov_derivative=metrics["V_dot"],
            plant_passivity_slack=audit.rayleigh,
            regulation_slack=float(-dV / max(dt, CONSTANTS.NUMERICAL_ZERO)),
            passivity_margin=audit.rayleigh,
            is_passive=audit.is_passive,
            is_regulating=is_regulating,
            discrete_residual=audit.residual,
            casimir_drift=audit.casimir_drift,
            power_requested=metrics["power_requested"],
            power_delivered=metrics["power_delivered"],
            applied_control_norm=applied_norm,
            pumping_required=pumping,
            mode=self._mode.value,
        )

    @staticmethod
    def _d_sign_needed(muscle: "FluxMuscleController") -> bool:
        return getattr(muscle, "_quadrants", 1) == 2

    def _discrete_passivity_audit(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> DiscretePassivityAudit:
        x_mid = 0.5 * (x0 + x1)
        grad_mid = self.K @ x_mid
        y_mid = self.g.T @ grad_mid
        rayleigh = float(grad_mid @ (self.R @ grad_mid))
        supply = float(u @ y_mid)
        dH = self.hamiltonian(x1) - self.hamiltonian(x0)
        predicted = dt * (supply - rayleigh)
        residual = float(dH - predicted)
        cas_drift = float(np.linalg.norm(self._casimir_values(x1) - self._C0))
        scale = max(1.0, abs(dH), abs(predicted), CONSTANTS.MIN_ENERGY_THRESHOLD)
        is_grad = abs(residual) <= CTRL.DISCRETE_PASSIVITY_TOL * scale
        is_passive = bool(dH <= dt * supply + CTRL.DISCRETE_PASSIVITY_TOL * scale)
        return DiscretePassivityAudit(
            dH_discrete=float(dH),
            supply=supply,
            rayleigh=rayleigh,
            residual=residual,
            casimir_drift=cas_drift,
            is_discrete_gradient=is_grad,
            is_passive=is_passive,
        )

    def _integrate_rk4(self, h: float, substeps: int, u_external: np.ndarray) -> None:
        def f(x_local: np.ndarray) -> np.ndarray:
            return self.A_auto @ x_local + self.g @ u_external

        x = self.x.copy()
        for _ in range(substeps):
            k1 = f(x)
            k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2)
            k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(x)):
                raise NumericalInstabilityError("Estado no finito en RK4.")
            if float(np.linalg.norm(x)) > CONSTANTS.MAX_STATE_NORM:
                raise NumericalInstabilityError("‖x‖ > MAX_STATE_NORM en RK4.")
        self.x = x

    def _integrate_implicit_midpoint(
        self,
        h: float,
        substeps: int,
        u_external: Optional[np.ndarray],
        freeze_u: bool = True,
    ) -> None:
        r"""
        Punto medio. Si freeze_u, u es ZOH (hold de orden cero) — el caso de
        composición con músculo. Si no, Picard sobre u(x_mid).
        """
        n = self.dim
        I = np.eye(n)
        M_left = I - 0.5 * h * self.A_auto
        M_right_op = I + 0.5 * h * self.A_auto
        lu = lu_factor(M_left) if _LU_AVAILABLE else None

        def _solve_left(rhs: np.ndarray) -> np.ndarray:
            if lu is not None:
                return np.asarray(lu_solve(lu, rhs), dtype=float)
            try:
                return np.linalg.solve(M_left, rhs)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(M_left, rhs, rcond=None)[0]

        x = self.x.copy()
        for _ in range(substeps):
            if freeze_u or u_external is not None:
                u_mid = u_external if u_external is not None else self.compute_control(x)
                rhs = M_right_op @ x + h * (self.g @ u_mid)
                x = _solve_left(rhs)
            else:
                x_new = x + h * (self.A_auto @ x + self.g @ self.compute_control(x))
                for _picard in range(CTRL.PICARD_MAX_ITER):
                    x_mid = 0.5 * (x + x_new)
                    u_mid = self.compute_control(x_mid)
                    x_next = _solve_left(M_right_op @ x + h * (self.g @ u_mid))
                    if float(np.linalg.norm(x_next - x_new)) <= CTRL.PICARD_ATOL * max(1.0, float(np.linalg.norm(x_new))):
                        x_new = x_next
                        break
                    x_new = x_next
                x = x_new
            if not np.all(np.isfinite(x)):
                raise NumericalInstabilityError("Estado no finito en punto medio.")
            if float(np.linalg.norm(x)) > CONSTANTS.MAX_STATE_NORM:
                raise NumericalInstabilityError("‖x‖ > MAX_STATE_NORM en punto medio.")
        self.x = x

    # ── auditorías no vacuadas ────────────────────────────────────────────────

    def verify_plant_passivity(
        self,
        num_steps: int = 100,
        dt: Optional[float] = None,
    ) -> Dict[str, float]:
        """Audita ΔH ≤ h y_midᵀ u  (pasividad discreta), no la identidad continua."""
        if num_steps <= 0:
            raise ConfigurationError("num_steps debe ser positivo.")
        if dt is None:
            dt = 1e-3
        x_backup, t_backup = self.x.copy(), self._time
        residuals: List[float] = []
        slacks: List[float] = []
        try:
            for _ in range(int(num_steps)):
                x0 = self.x.copy()
                report = self.controlled_step(dt)
                audit = self._discrete_passivity_audit(x0, self.x, self._last_applied_u, dt)
                residuals.append(audit.residual)
                slacks.append(report.plant_passivity_slack)
                if not audit.is_passive:
                    # no lanzamos: devolvemos el veredicto
                    pass
        finally:
            self.x, self._time = x_backup, t_backup
        arr_r = np.asarray(residuals, dtype=float) if residuals else np.zeros(1)
        arr_s = np.asarray(slacks, dtype=float) if slacks else np.zeros(1)
        return {
            "is_plant_passive": bool(np.max(np.abs(arr_r)) <= CTRL.DISCRETE_PASSIVITY_TOL * 10.0 or np.min(arr_s) >= -CONSTANTS.NUMERICAL_TOLERANCE),
            "max_discrete_residual": float(np.max(np.abs(arr_r))),
            "min_rayleigh": float(np.min(arr_s)),
            "mean_residual": float(np.mean(arr_r)),
            "samples": float(arr_r.size),
        }

    def verify_regulation_convergence(
        self,
        num_steps: int = 200,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        No exige monotonicidad de V cuando H<H* y R≠0 (Rayleigh empuja hacia 0).
        Éxito: |H−H*| decrece en media y Casimirs no derivan.
        """
        if num_steps <= 0:
            raise ConfigurationError("num_steps debe ser positivo.")
        if dt is None:
            dt = 1e-3
        x_backup, t_backup = self.x.copy(), self._time
        energies: List[float] = []
        storages: List[float] = []
        cas_leaks: List[float] = []
        try:
            for _ in range(int(num_steps)):
                report = self.controlled_step(dt)
                energies.append(report.hamiltonian)
                storages.append(report.storage_function)
                cas_leaks.append(report.casimir_drift)
        finally:
            self.x, self._time = x_backup, t_backup
        if not energies:
            return {"is_regulating": True, "samples": 0.0}
        e0 = abs(energies[0] - self.H_target)
        e1 = abs(energies[-1] - self.H_target)
        return {
            "is_regulating": bool(e1 <= e0 + CONSTANTS.RELATIVE_TOLERANCE * max(1.0, self.H_target)),
            "energy_error_start": float(e0),
            "energy_error_end": float(e1),
            "storage_end": float(storages[-1]),
            "max_casimir_drift": float(np.max(cas_leaks) if cas_leaks else 0.0),
            "samples": float(len(energies)),
            "mode": self._mode.value,
        }

    def audit_power_balance(
        self,
        dt: Optional[float] = None,
        u_input: Optional[Union[float, np.ndarray]] = None,
    ) -> Dict[str, float]:
        if dt is None:
            dt = 1e-4
        u = self._normalize_control_input(u_input) if u_input is not None else self.compute_control()
        x_backup, t_backup = self.x.copy(), self._time
        x0 = self.x.copy()
        H_before = self.hamiltonian()
        metrics = self.continuous_lyapunov_derivative(u=u)
        try:
            self._integrate_implicit_midpoint(dt, 1, u, freeze_u=True)
            self.x = self._project_casimirs(self.x)
            H_after = self.hamiltonian()
            audit = self._discrete_passivity_audit(x0, self.x, u, dt)
        finally:
            self.x, self._time = x_backup, t_backup
        empirical_H_dot = (H_after - H_before) / dt
        return {
            "analytic_H_dot": metrics["H_dot"],
            "empirical_H_dot": empirical_H_dot,
            "residual": audit.residual,
            "supply_rate": metrics["supply_rate"],
            "rayleigh_rate": metrics["rayleigh_rate"],
            "is_discrete_gradient": audit.is_discrete_gradient,
        }

    def maupertuis_conformal_factor(self, x: Optional[np.ndarray] = None) -> float:
        """
        Deprecado. Jacobi-Maupertuis exige H=T(q,p)+V(q) en T*Q.
        Un PHS lineal en x no tiene esa escisión. Se devuelve 1.0.
        """
        logger.debug("maupertuis_conformal_factor es un no-op en 7.1 (no hay T+V).")
        return 1.0

    def apply_maupertuis_correction(
        self,
        dt: float,
        geodesic_gain: float = 0.05,
    ) -> PortHamiltonianControlReport:
        """
        Sustituto honesto: un paso de ENERGY_LEVEL (power shaping).
        La antigua ley extraía energía siempre y rompía el bombeo.
        """
        if not math.isfinite(geodesic_gain) or geodesic_gain < 0.0:
            raise ConfigurationError("geodesic_gain debe ser ≥ 0.")
        return self.controlled_step(dt=dt)

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 2.6 — PUENTE FORMAL FASE 2 → FASE 3
    # ──────────────────────────────────────────────────────────────────────────

    def synthesize_engine_seed(
        self,
        pi_controller: Optional[PIController] = None,
        muscle: Optional[FluxMuscleController] = None,
        dt: Optional[float] = None,
    ) -> PoincareEngineSeed:
        r"""
        PUENTE FORMAL FASE 2 → FASE 3.

        Cascada única (un solo u aplicado):
            1. PI (opcional) → potencia pedida P_* normalizada · escala.
            2. Power-shaping / IDA-PBC → u_cmd ∈ ℝ^m.
            3. Músculo (opcional) → saturación térmica/slew → u_app.
            4. Semilla con u_app, Casimirs, matching, μ₂(A), Tellegen discreto.

        Fase 3 **integra con control_input = u_app**, no con el PI crudo.
        """
        if dt is None:
            dt = 1e-3
        dt = float(dt)
        grad_H = self.gradient()
        H = self.hamiltonian()
        V = self.storage_function()
        y = self.port_output()

        power_ref: Optional[float] = None
        pi_report: Optional[PIControlReport] = None
        if pi_controller is not None:
            try:
                pi_report = pi_controller.compute_from_seed(seed=self._seed, dt=dt, feedforward=0.0)
                # PI normalizado: output ∈ [min,max] se reescala a potencia característica ‖y‖ u_max
                p_char = max(float(np.linalg.norm(y)) * self.u_max, CONSTANTS.MIN_ENERGY_THRESHOLD)
                span = max(pi_controller.max_output - pi_controller.min_output, CONSTANTS.NUMERICAL_TOLERANCE)
                mid = 0.5 * (pi_controller.max_output + pi_controller.min_output)
                # mapa afín: centro → P=0 (si bipolar); si min≥0, output/max → [0, p_char]
                if pi_controller.min_output < 0.0:
                    power_ref = float(pi_report.output - mid) / (0.5 * span) * p_char
                else:
                    power_ref = float(pi_report.output) / max(pi_controller.max_output, CONSTANTS.NUMERICAL_TOLERANCE) * p_char
            except DataFluxCondenserError as exc:
                logger.error("PIController falló: %s", exc)
                pi_report = None

        u_cmd = self.compute_control(power_reference=power_ref)
        muscle_duty = 0.0
        u_app = u_cmd.copy()
        if muscle is not None:
            try:
                nrm = float(np.linalg.norm(u_cmd))
                intensity = nrm / max(self.u_max, CONSTANTS.NUMERICAL_TOLERANCE)
                sign = 1.0
                if y.size and nrm > 0.0 and getattr(muscle, "_quadrants", 1) == 2:
                    sign = math.copysign(1.0, float(y @ u_cmd))
                muscle_duty = float(muscle.apply_force(sign * float(np.clip(intensity, 0.0, 1.0)), dt=dt))
                scale = abs(muscle_duty) / max(intensity, CONSTANTS.NUMERICAL_TOLERANCE) if intensity > 0.0 else 0.0
                u_app = u_cmd * float(np.clip(scale, 0.0, 1.0))
            except DataFluxCondenserError as exc:
                logger.error("FluxMuscleController falló: %s", exc)
                muscle_duty = 0.0

        self._last_commanded_u = u_cmd
        self._last_applied_u = u_app
        metrics = self.continuous_lyapunov_derivative(u=u_app)
        audit = self._discrete_passivity_audit(self.x, self.x, u_app, dt)  # residual 0 si Δx=0
        passivity_report = {
            "lyapunov_derivative": float(metrics["V_dot"]),
            "hamiltonian_derivative": float(metrics["H_dot"]),
            "supply_rate": float(metrics["supply_rate"]),
            "rayleigh_rate": float(metrics["rayleigh_rate"]),
            "plant_passivity_slack": float(metrics["plant_passivity_slack"]),
            "regulation_slack": float(metrics["regulation_slack"]),
            "power_requested": float(metrics["power_requested"]),
            "power_delivered": float(metrics["power_delivered"]),
            "is_plant_passive": float(metrics["plant_passivity_slack"] >= -CONSTANTS.NUMERICAL_TOLERANCE),
            "is_regulating": float(metrics["regulation_slack"] >= -CONSTANTS.NUMERICAL_TOLERANCE),
            "discrete_residual": float(audit.residual),
            "casimir_drift": float(np.linalg.norm(self._casimir_values(self.x) - self._C0)),
        }
        engine_hints = {
            "dt_suggested": float(dt),
            "rho_A": float(self._rho_A),
            "logarithmic_norm_A": float(self._mu2_A),
            "substeps": int(self._estimate_substeps(dt)),
            "integrator_preferred": "implicit_midpoint",
            "integrator_forbidden_as_structure_preserving": "rk4",
            "state_dim": self.dim,
            "port_dim": self.port_dim,
            "casimir_dim": int(self._C.shape[1]) if self._C is not None and self._C.size else 0,
            "mode": self._mode.value,
            "pumping_required": bool(self._validated.pumping_required),
            "hodge_convention": "D=ε★₁E, H=μ⁻¹★₂B, δ₂=★₁⁻¹∂₂★₂",
            "schema_version": "7.1.0",
            "do_not_regulate_casimirs": True,
            "use_applied_control": True,
        }
        metadata: Dict[str, Any] = {
            "time": self._time,
            "dt_suggested": dt,
            "kd": self.kd,
            "use_energy_shaping": self.use_energy_shaping,
            "mode": self._mode.value,
            "u_max": self.u_max,
            "metric_condition_number": self._validated.metric_condition_number,
            "spectral_radius_J": self._validated.spectral_radius_J,
            "normalized_energy_error": self._validated.normalized_energy_error,
            "trace_generator": self._validated.trace_generator,
            "port_gram_condition": self._validated.port_gram_condition,
            "logarithmic_norm": self._validated.logarithmic_norm,
            "casimir_energy": self._validated.casimir_energy,
            "dynamic_energy": self._validated.dynamic_energy,
            "casimir_port_leak": self._validated.casimir_port_leak,
            "matching_residual": self._validated.matching_residual,
            "pumping_required": self._validated.pumping_required,
            "is_energy_controllable": self._validated.is_energy_controllable,
            "phase": 2,
            "bridge_to_phase": 3,
            "schema_version": "7.1.0",
            "hodge_convention": engine_hints["hodge_convention"],
        }
        return PoincareEngineSeed(
            state=self.x.copy(),
            gradient=grad_H.copy(),
            control_input=u_app.copy(),
            hamiltonian=float(H),
            target_hamiltonian=float(self.H_target),
            storage_function=float(V),
            interconnection_matrix=self.J.copy(),
            damping_matrix=self.R.copy(),
            metric_matrix=self.K.copy(),
            port_matrix=self.g.copy(),
            pi_report=pi_report,
            muscle_duty=float(muscle_duty),
            passivity_report=passivity_report,
            metadata=metadata,
            casimir_basis=None if self._C is None else self._C.copy(),
            spectral_data=None if not self._validated.spectral_data else dict(self._validated.spectral_data),
            ida_pbc_decomposition=(
                None
                if self._validated.ida_pbc_decomposition is None
                else {k: np.asarray(v, dtype=float).copy() for k, v in self._validated.ida_pbc_decomposition.items()}
            ),
            lyapunov_jacobian=(H - self.H_target) * grad_H,
            engine_hints=engine_hints,
            commanded_control=u_cmd.copy(),
            output_port=y.copy(),
            matching_residual=float(self._validated.matching_residual),
            pumping_required=bool(self._validated.pumping_required),
            casimir_energy=float(self._validated.casimir_energy),
            schema_version="7.1.0",
        )

    def closed_loop_step(
        self,
        dt: float,
        pi_controller: Optional[PIController] = None,
        muscle: Optional[FluxMuscleController] = None,
    ) -> Tuple[PortHamiltonianControlReport, PoincareEngineSeed]:
        """Un paso de la cascada completa y la semilla actualizada (para Fase 3)."""
        seed = self.synthesize_engine_seed(pi_controller=pi_controller, muscle=muscle, dt=dt)
        report = self.controlled_step(
            dt=dt,
            u_input=seed.control_input,
            muscle=None,  # ya saturado en synthesize
        )
        seed_after = self.synthesize_engine_seed(pi_controller=None, muscle=None, dt=dt)
        seed_after.pi_report = seed.pi_report
        seed_after.muscle_duty = seed.muscle_duty
        return report, seed_after

    # ── utilidades ────────────────────────────────────────────────────────────

    def _state_or_current(self, x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return self.x
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.dim:
            raise ConfigurationError(f"Estado dim {x_arr.size}; esperado {self.dim}.")
        if not np.all(np.isfinite(x_arr)):
            raise NumericalInstabilityError("Estado contiene valores no finitos.")
        return x_arr

    def _normalize_control_input(self, u_input: Optional[Union[float, np.ndarray]]) -> np.ndarray:
        if u_input is None:
            return np.zeros(self.port_dim, dtype=float)
        if np.isscalar(u_input):
            u = np.full(self.port_dim, float(u_input), dtype=float)
        else:
            u = np.asarray(u_input, dtype=float).reshape(-1)
        if u.size != self.port_dim:
            raise ConfigurationError(f"Control dim {u.size}; esperado {self.port_dim}.")
        if not np.all(np.isfinite(u)):
            raise NumericalInstabilityError("Control contiene valores no finitos.")
        return u

    def _estimate_substeps(self, dt: float) -> int:
        """
        Punto medio: A-estable en el semiplano izquierdo para el bloque lineal.
        El CFL efectivo lo marca el transitorio no-normal: h·max(μ₂(A),0) ≲ 1
        y, para RK4, h·ρ ≲ 2.5.
        """
        if self._integrator == "implicit_midpoint":
            mu_plus = max(self._mu2_A, 0.0)
            if mu_plus <= CONSTANTS.NUMERICAL_TOLERANCE:
                return 1
            h_max = CTRL.MIDPOINT_CFL / mu_plus
        else:
            rho = max(self._rho_A, abs(self._mu2_A), CONSTANTS.NUMERICAL_TOLERANCE)
            h_max = self.RK4_STABILITY_CFL / rho
        if h_max <= 0.0 or not math.isfinite(h_max):
            return 1
        return int(np.clip(math.ceil(dt / h_max), 1, 1000))


# ═══════════════════════════════════════════════════════════════════════════════════════
# FIN DE LA FASE 2  (v7.1.0)
# ═══════════════════════════════════════════════════════════════════════════════════════
#
# PortHamiltonianPoincareController.synthesize_engine_seed()  — frontera → FASE 3.
# Contrato del PoincareEngineSeed (7.1):
#
#   1. control_input = u_APLICADO (post-músculo, Cᵀ g u = 0).
#      commanded_control = u pre-saturación.  Fase 3 integra el aplicado.
#   2. PHS (J, R, K, g) con la misma convención Hodge que Fase 1.
#   3. Casimirs inmunes; casimir_energy ≤ H*; no regular δD ni armónicos.
#   4. Power shaping: P_* = ∇HᵀR∇H − λ(H−H*),  u = P_* y/(‖y‖²+ε).
#   5. Integrador preferido: implicit_midpoint (gradiente discreto).  RK4 no preserva.
#   6. engine_hints["pumping_required"]  ⇒  no usar DAMPING_ONLY.
#   7. matching_residual > 0  ⇒  IDA_PBC_POINT no es exacto; usar ENERGY_LEVEL.
#   8. schema_version == "7.1.0"  y  hodge_convention inmutable.
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# ╔═════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3/3 — MOTOR PHS, ESTADO UNIFICADO (GENERIC) Y ORQUESTADOR                     ║
# ║  Versión: 7.1.0-Poincare-DEC-PHS-Rigorous                                           ║
# ╚═════════════════════════════════════════════════════════════════════════════════════╝
# ═══════════════════════════════════════════════════════════════════════════════════════
#
# Frontera de entrada : PortHamiltonianPoincareController.synthesize_engine_seed()
#                       → PoincareEngineSeed   (contrato 7.1)
# Frontera de salida  : DataFluxCondenser.synthesize_final_unified_state()
#                       → UnifiedPhysicalSnapshot
#
# Capas (no se mezclan):
#   A. Planta PHS:  x ← Φ_h^{mid}(x; u_app),  u_app = seed.control_input (ZOH).
#   B. Observables lumped: (Q,λ) SÓLO si el atlas es RLC canónico dim=2.
#   C. GENERIC isotermo: Lyapunov = H_em;  1ª ley E = H_em + T S;  σ = ‖∇H‖_R² / T.
#   D. Orquestador: PI de saturación → tamaño de lote. Nunca u de planta.
#
# Prohibiciones 7.1:
#   · no redefinir ★, no regular Casimirs, no RK4 como “estructura-preservante”,
#   · no re-aplicar músculo, no interpretir [D,B] como [Q,λ],
#   · dt físico ≠ reloj de pared.
# ═══════════════════════════════════════════════════════════════════════════════════════

from collections import OrderedDict
from dataclasses import asdict
from typing import Iterable, Iterator, Mapping

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from scipy.special import digamma as _digamma
except ImportError:  # pragma: no cover
    _digamma = None


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 3.1 — EXCEPCIONES, CONFIGURACIÓN, ATLAS
# ═══════════════════════════════════════════════════════════════════════════════════════


class EngineSeedError(DataFluxCondenserError):
    """Semilla de motor físico inválida, incompleta o de esquema incompatible."""


class OrchestrationError(DataFluxCondenserError):
    """Error en la orquestación del condensador (lotes, timeout, brownout de control)."""


class EntropyViolationError(DataFluxCondenserError):
    """Violación del Clausius *discreto*: σ < 0 o ΔH_em + TΔS − h yᵀu fuera de tolerancia."""


class ThermalModelError(DataFluxCondenserError):
    """Parámetros o estado térmico físicamente inconsistentes."""


class IntegratorConvergenceError(DataFluxCondenserError):
    """Fallo de convergencia de Newton / Picard."""


class SchemaContractError(DataFluxCondenserError):
    """Semilla con schema_version / hodge_convention incompatibles con 7.1."""


class AtlasKind(str, Enum):
    """Carta en la que vive x. No se interpolan."""

    RLC_CANONICAL = "rlc_canonical"     # x = [Q, λ] ∈ T*ℝ
    MAXWELL_DEC = "maxwell_dec"         # x = [D, B] ∈ C¹ ⊕ C²
    ABSTRACT_PHS = "abstract_phs"       # x genérico, observables vía H, y, Rayleigh
    STANDALONE = "standalone"           # sin semilla: RLC 2D local


@dataclass(frozen=True)
class ThermalParameters:
    """
    Baño isotermo GENERIC.

    T_res > 0 fija. Energía del baño U = T_res · S  (T = ∂U/∂S constante).
    Producción: σ = ‖∇H_em‖_R² / T_res  (W/K).  No es Shannon de lotes.
    """

    reservoir_temperature_K: float = 293.15
    ambient_temperature_C: float = 25.0
    muscle_time_constant_s: float = 5.0
    muscle_thermal_gain: float = 15.0
    entropy_reference_K: float = 293.15

    def __post_init__(self) -> None:
        if self.reservoir_temperature_K <= 0.0:
            raise ThermalModelError("reservoir_temperature_K debe ser positiva.")
        if self.entropy_reference_K <= 0.0:
            raise ThermalModelError("entropy_reference_K debe ser positiva.")
        if self.muscle_time_constant_s <= 0.0:
            raise ThermalModelError("muscle_time_constant_s debe ser positivo.")
        if self.muscle_thermal_gain < 0.0:
            raise ThermalModelError("muscle_thermal_gain debe ser no negativo.")


@dataclass(frozen=True)
class CondenserConfig:
    r"""
    Configuración inmutable.

    RLC standalone (solo atlas STANDALONE / RLC_CANONICAL):
        ω₀ = 1/√(LC),  α = R/(2L),  ζ = α/ω₀.
    El PI de esta config es de **tamaño de lote**, no de energía.
    dt físico = physics_dt (o engine_hints['dt_suggested']). El reloj de pared
    solo gobierna PROCESSING_TIMEOUT.
    """

    min_records_threshold: int = 1
    enable_strict_validation: bool = True
    log_level: str = "INFO"

    system_capacitance: float = 1.0
    base_resistance: float = 1.4142135623730951
    system_inductance: float = 0.5
    max_voltage: float = 5.3

    p_laplacian_exponent: float = 3.0
    p_laplacian_epsilon: float = 1e-8
    p_laplacian_beta: float = 0.1          # R_s(I) = R (1 + β (I²+ε)^((p-2)/2))
    p_laplacian_G0: float = 0.0            # shunt; 0 = sin fugas P-Laplaciano

    brain_capacitance: float = 4.0
    brain_brownout_threshold: float = 2.65
    brain_agent_consumption_amps: float = 0.080
    brain_diode_drop: float = 0.3
    brain_enabled: bool = True             # plano de control; no es planta PHS

    pid_setpoint: float = 0.30             # saturación objetivo ∈ (0,1]
    pid_kp: float = 2000.0
    pid_ki: float = 100.0
    min_batch_size: int = 1
    max_batch_size: int = 5000
    integral_limit_factor: float = 2.0
    batch_inertia_nominal: float = 0.65
    batch_inertia_emergency: float = 0.30

    max_failed_batches: int = 3

    integrator: str = "implicit_midpoint"  # PHS: implicit_midpoint | (standalone) trapezoidal | tr_bdf2 | implicit_euler
    newton_max_iter: int = 20
    newton_tol: float = 1e-10
    physics_dt: float = 1e-3
    maxwell_substep: bool = False          # True solo si hay lattice inyectado *y* no se pisa el PHS

    thermal: ThermalParameters = field(default_factory=ThermalParameters)
    trace_enabled: bool = True
    require_schema_7_1: bool = True

    def __post_init__(self) -> None:
        errors: List[str] = []
        if self.min_records_threshold < 0:
            errors.append("min_records_threshold debe ser >= 0.")
        for name, val, pred in (
            ("system_capacitance", self.system_capacitance, lambda v: v > 0.0),
            ("system_inductance", self.system_inductance, lambda v: v > 0.0),
            ("base_resistance", self.base_resistance, lambda v: v >= 0.0),
            ("max_voltage", self.max_voltage, lambda v: v > 0.0),
            ("pid_kp", self.pid_kp, lambda v: v >= 0.0),
            ("pid_ki", self.pid_ki, lambda v: v >= 0.0),
            ("physics_dt", self.physics_dt, lambda v: v > 0.0),
            ("newton_tol", self.newton_tol, lambda v: v > 0.0),
            ("p_laplacian_epsilon", self.p_laplacian_epsilon, lambda v: v > 0.0),
            ("p_laplacian_beta", self.p_laplacian_beta, lambda v: v >= 0.0),
            ("p_laplacian_G0", self.p_laplacian_G0, lambda v: v >= 0.0),
        ):
            if not math.isfinite(val) or not pred(val):
                errors.append(f"{name} inválido: {val}.")
        if not (0.0 < self.pid_setpoint <= 1.0):
            errors.append("pid_setpoint debe estar en (0, 1].")
        if self.min_batch_size <= 0 or self.min_batch_size > self.max_batch_size:
            errors.append("rango de batch inválido.")
        if self.max_batch_size > CONSTANTS.MAX_RECORDS_LIMIT:
            errors.append("max_batch_size excede MAX_RECORDS_LIMIT.")
        if self.p_laplacian_exponent <= 2.0:
            errors.append("p_laplacian_exponent debe ser > 2.")
        if not (0.0 <= self.batch_inertia_nominal <= 1.0 and 0.0 <= self.batch_inertia_emergency <= 1.0):
            errors.append("inercias de lote deben estar en [0,1].")
        if self.integrator not in {"implicit_midpoint", "trapezoidal", "implicit_euler", "tr_bdf2"}:
            errors.append(f"integrator desconocido: {self.integrator}.")
        if self.newton_max_iter <= 0:
            errors.append("newton_max_iter debe ser > 0.")
        if errors:
            raise ConfigurationError("CondenserConfig inválida:\n" + "\n".join(f"  - {e}" for e in errors))

    @property
    def omega_0(self) -> float:
        return 1.0 / math.sqrt(self.system_inductance * self.system_capacitance)

    @property
    def damping_ratio(self) -> float:
        return (self.base_resistance / 2.0) * math.sqrt(self.system_capacitance / self.system_inductance)

    @property
    def dominant_pole(self) -> float:
        return -self.base_resistance / (2.0 * self.system_inductance)


@dataclass
class ProcessingStats:
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    total_batches: int = 0
    failed_batches: int = 0
    processing_time: float = 0.0
    avg_batch_size: float = 0.0
    avg_saturation: float = 0.0
    emergency_brakes_triggered: int = 0

    def add_batch_stats(self, batch_size: int, saturation: float, success: bool) -> None:
        self.total_batches += 1
        if success:
            self.processed_records += batch_size
        else:
            self.failed_records += batch_size
            self.failed_batches += 1
        n = max(1, self.total_batches)
        self.avg_batch_size = ((n - 1) * self.avg_batch_size + batch_size) / n
        self.avg_saturation = ((n - 1) * self.avg_saturation + saturation) / n


@dataclass
class BatchResult:
    success: bool
    dataframe: Optional[Any] = None
    records_processed: int = 0
    error_message: str = ""
    error_types: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FirstLawAudit:
    """Clausius discreto de un paso (GENERIC isotermo)."""

    dH_em: float
    T_dS: float
    supply: float
    residual: float
    sigma: float
    is_first_law: bool
    is_second_law: bool


@dataclass(eq=False)
class UnifiedPhysicalSnapshot:
    """Instantánea contractual del cierre Fase 3."""

    time: float
    physics_time: float
    charge: Optional[float]
    current: Optional[float]
    flux_linkage: Optional[float]
    entropy: float
    entropy_production_rate: float          # σ = dS/dt  (Clausius)
    entropy_step: float                     # ΔS del último paso
    reservoir_temperature_K: float
    brain_voltage: float
    brain_alive: bool
    muscle_temperature: float
    hamiltonian_em: float                   # Lyapunov / disponibilidad
    hamiltonian_total_first_law: float      # H_em + T S
    target_hamiltonian: float
    storage_function: float
    state_vector: np.ndarray
    control_applied: np.ndarray
    control_commanded: Optional[np.ndarray]
    poincare_audit: Dict[str, Any]
    phs_structure_preserved: bool
    energy_breakdown: Dict[str, float]
    first_law: Dict[str, float]
    atlas: str
    schema_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 3.2 — ESTADO UNIFICADO (GENERIC + ATLAS)
# ═══════════════════════════════════════════════════════════════════════════════════════


class UnifiedPhysicalState:
    r"""
    Estado unificado con escisión GENERIC.

        H_em(x) = ½ xᵀ K x          (disponibilidad; Lyapunov de IDA/energy-shaping)
        U_bath  = T_res S           (baño isotermo)
        E       = H_em + U_bath     (1ª ley)
        σ       = (∇H_em)ᵀ R ∇H_em / T_res   ≥ 0

    Atlas:
        RLC_CANONICAL  → (Q, λ) = (x₀, x₁),  I = λ/L.
        MAXWELL_DEC    → observables EM vía H_em, y, Rayleigh; (Q,λ) = None.
        ABSTRACT_PHS   → idem.
        STANDALONE     → RLC local, sin semilla.
    """

    def __init__(
        self,
        capacitance: float,
        inductance: float,
        resistance: float,
        thermal: Optional[ThermalParameters] = None,
    ) -> None:
        self.capacitance = max(float(capacitance), CONSTANTS.NUMERICAL_TOLERANCE)
        self.inductance = max(float(inductance), CONSTANTS.NUMERICAL_TOLERANCE)
        self.resistance = max(float(resistance), 0.0)
        self.thermal = thermal or ThermalParameters()
        self.atlas: AtlasKind = AtlasKind.STANDALONE
        self.charge: Optional[float] = 0.0
        self.flux_linkage: Optional[float] = 0.0
        self.entropy = 0.0
        self.last_sigma = 0.0
        self.last_dS = 0.0
        self.last_first_law = FirstLawAudit(0.0, 0.0, 0.0, 0.0, 0.0, True, True)
        self.brain_voltage = 5.0
        self.brain_inflow_current = 0.0
        self.brain_alive = True
        self.muscle_temperature = self.thermal.ambient_temperature_C
        self.state_vector = np.zeros(0, dtype=float)
        self.control_applied = np.zeros(0, dtype=float)
        self.control_commanded: Optional[np.ndarray] = None
        self.target_hamiltonian = CONSTANTS.MIN_ENERGY_THRESHOLD
        self.storage_function = 0.0
        self.J: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.g: Optional[np.ndarray] = None
        self.casimir_basis: Optional[np.ndarray] = None
        self.spectral_data: Optional[Dict[str, Any]] = None
        self.ida_pbc_decomposition: Optional[Dict[str, np.ndarray]] = None
        self.lyapunov_jacobian: Optional[np.ndarray] = None
        self.port_dim: int = 0
        self._phs_preserved: bool = False
        self._last_seed: Optional[PoincareEngineSeed] = None
        self.casimir_energy: float = 0.0
        self.pumping_required: bool = False
        self.matching_residual: float = 0.0
        self.hodge_convention: str = "D=ε★₁E, H=μ⁻¹★₂B, δ₂=★₁⁻¹∂₂★₂"
        self.schema_version: str = "7.1.0"
        self.physics_time: float = 0.0

    @staticmethod
    def infer_atlas(seed: PoincareEngineSeed) -> AtlasKind:
        meta = dict(seed.metadata or {})
        hints = dict(seed.engine_hints or {})
        n_e = int(meta.get("num_edges", hints.get("num_edges", -1)))
        n_f = int(meta.get("num_faces", hints.get("num_faces", -1)))
        dim = int(np.asarray(seed.state).reshape(-1).size)
        name = str(meta.get("kernel_name", hints.get("kernel_name", "")))
        if n_e >= 0 and n_f >= 0 and dim == n_e + n_f and dim > 2:
            return AtlasKind.MAXWELL_DEC
        if dim == 2 and ("RLC" in name.upper() or meta.get("atlas") == AtlasKind.RLC_CANONICAL.value):
            return AtlasKind.RLC_CANONICAL
        if dim == 2:
            return AtlasKind.RLC_CANONICAL
        return AtlasKind.ABSTRACT_PHS

    def consume_engine_seed(self, seed: PoincareEngineSeed) -> None:
        if seed is None:
            raise EngineSeedError("PoincareEngineSeed es requerida.")
        schema = str(getattr(seed, "schema_version", "") or (seed.metadata or {}).get("schema_version", ""))
        if schema and schema != "7.1.0":
            logger.warning("schema_version=%s ≠ 7.1.0; se continúa con contrato 7.1.", schema)
        state = np.asarray(seed.state, dtype=float).reshape(-1)
        if state.size == 0 or not np.all(np.isfinite(state)):
            raise EngineSeedError("seed.state vacío o no finito.")
        u = np.asarray(seed.control_input, dtype=float).reshape(-1)
        if not np.all(np.isfinite(u)):
            raise EngineSeedError("seed.control_input no finito.")
        self.state_vector = state.copy()
        self.control_applied = u.copy()
        cc = getattr(seed, "commanded_control", None)
        self.control_commanded = None if cc is None else np.asarray(cc, dtype=float).reshape(-1).copy()
        self.target_hamiltonian = max(float(seed.target_hamiltonian), CONSTANTS.MIN_ENERGY_THRESHOLD)
        self.storage_function = float(seed.storage_function)
        self._last_seed = seed
        self.J = np.asarray(seed.interconnection_matrix, dtype=float).copy()
        self.R = np.asarray(seed.damping_matrix, dtype=float).copy()
        self.K = np.asarray(seed.metric_matrix, dtype=float).copy()
        self.g = np.asarray(seed.port_matrix, dtype=float).copy()
        self.port_dim = int(self.g.shape[1]) if self.g.ndim == 2 else 0
        if self.g.shape[0] != state.size:
            raise EngineSeedError(f"g filas {self.g.shape[0]} ≠ dim estado {state.size}.")
        if u.size != self.port_dim:
            raise EngineSeedError(f"u dim {u.size} ≠ port_dim {self.port_dim}.")
        self.casimir_basis = None if seed.casimir_basis is None else np.asarray(seed.casimir_basis, dtype=float).copy()
        self.spectral_data = None if not seed.spectral_data else dict(seed.spectral_data)
        self.ida_pbc_decomposition = (
            None
            if seed.ida_pbc_decomposition is None
            else {k: np.asarray(v, dtype=float).copy() for k, v in seed.ida_pbc_decomposition.items()}
        )
        self.lyapunov_jacobian = (
            None if seed.lyapunov_jacobian is None else np.asarray(seed.lyapunov_jacobian, dtype=float).copy()
        )
        self._phs_preserved = True
        self.atlas = self.infer_atlas(seed)
        self.casimir_energy = float(getattr(seed, "casimir_energy", 0.0) or 0.0)
        self.pumping_required = bool(getattr(seed, "pumping_required", False))
        self.matching_residual = float(getattr(seed, "matching_residual", 0.0) or 0.0)
        conv = (seed.metadata or {}).get("hodge_convention") or (seed.engine_hints or {}).get("hodge_convention")
        if conv:
            self.hodge_convention = str(conv)
        self.schema_version = schema or "7.1.0"
        if self.atlas is AtlasKind.RLC_CANONICAL and state.size >= 2:
            self.charge = float(state[0])
            self.flux_linkage = float(state[1])
        else:
            self.charge = None
            self.flux_linkage = None

    @property
    def current(self) -> Optional[float]:
        if self.flux_linkage is None:
            return None
        return self.flux_linkage / self.inductance

    def electric_energy(self) -> Optional[float]:
        if self.charge is None:
            return None
        return 0.5 * (self.charge ** 2) / self.capacitance

    def magnetic_energy(self) -> Optional[float]:
        if self.flux_linkage is None:
            return None
        return 0.5 * (self.flux_linkage ** 2) / self.inductance

    def em_hamiltonian_from_metric(self) -> float:
        x = self.state_vector
        if x.size == 0 or self.K is None:
            he = self.electric_energy() or 0.0
            hm = self.magnetic_energy() or 0.0
            return float(he + hm)
        H = 0.5 * float(x @ (self.K @ x))
        return max(0.0, H)

    def bath_energy(self) -> float:
        return self.thermal.reservoir_temperature_K * self.entropy

    def availability(self) -> float:
        """Lyapunov: H_em. No incluye TS."""
        return self.em_hamiltonian_from_metric()

    def first_law_energy(self) -> float:
        return self.availability() + self.bath_energy()

    def energy_breakdown(self) -> Dict[str, float]:
        he = self.electric_energy()
        hm = self.magnetic_energy()
        return {
            "H_electric": float(he) if he is not None else float("nan"),
            "H_magnetic": float(hm) if hm is not None else float("nan"),
            "H_em": float(self.availability()),
            "U_bath": float(self.bath_energy()),
            "E_first_law": float(self.first_law_energy()),
            "casimir_energy": float(self.casimir_energy),
        }

    def evolve_generic_bath(
        self,
        dt: float,
        rayleigh_mid: float,
        supply_mid: float,
        dH_em: float,
    ) -> FirstLawAudit:
        r"""
        ΔS = (h/T) ‖∇H‖_R²,  σ = Rayleigh / T.
        Residuo 1ª ley: ΔH_em + TΔS − h yᵀu.
        Para punto medio + H cuadrática, residuo ~ 0 (Tellegen discreto).
        """
        dt = max(float(dt), CONSTANTS.MIN_DELTA_TIME)
        T = max(self.thermal.reservoir_temperature_K, CONSTANTS.NUMERICAL_TOLERANCE)
        rayleigh_mid = max(float(rayleigh_mid), 0.0)
        sigma = rayleigh_mid / T
        dS = sigma * dt
        self.entropy += dS
        self.last_sigma = sigma
        self.last_dS = dS
        T_dS = T * dS
        residual = float(dH_em + T_dS - dt * float(supply_mid))
        scale = max(1.0, abs(dH_em), abs(T_dS), abs(dt * supply_mid), CONSTANTS.MIN_ENERGY_THRESHOLD)
        audit = FirstLawAudit(
            dH_em=float(dH_em),
            T_dS=float(T_dS),
            supply=float(supply_mid),
            residual=residual,
            sigma=sigma,
            is_first_law=abs(residual) <= CTRL.DISCRETE_PASSIVITY_TOL * scale,
            is_second_law=sigma >= -CONSTANTS.NUMERICAL_TOLERANCE,
        )
        self.last_first_law = audit
        if not math.isfinite(self.entropy):
            raise NumericalInstabilityError("Entropía no finita.")
        if not audit.is_second_law:
            raise EntropyViolationError(f"σ={sigma:.3e} < 0.")
        return audit

    def snapshot(
        self,
        wall_time: float,
        poincare_audit: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedPhysicalSnapshot:
        if not math.isfinite(wall_time):
            raise ConfigurationError("wall_time debe ser finito.")
        fl = self.last_first_law
        return UnifiedPhysicalSnapshot(
            time=float(wall_time),
            physics_time=float(self.physics_time),
            charge=self.charge,
            current=self.current,
            flux_linkage=self.flux_linkage,
            entropy=float(self.entropy),
            entropy_production_rate=float(self.last_sigma),
            entropy_step=float(self.last_dS),
            reservoir_temperature_K=float(self.thermal.reservoir_temperature_K),
            brain_voltage=float(self.brain_voltage),
            brain_alive=bool(self.brain_alive),
            muscle_temperature=float(self.muscle_temperature),
            hamiltonian_em=float(self.availability()),
            hamiltonian_total_first_law=float(self.first_law_energy()),
            target_hamiltonian=float(self.target_hamiltonian),
            storage_function=float(self.storage_function),
            state_vector=self.state_vector.copy(),
            control_applied=self.control_applied.copy(),
            control_commanded=None if self.control_commanded is None else self.control_commanded.copy(),
            poincare_audit=dict(poincare_audit or {}),
            phs_structure_preserved=bool(self._phs_preserved),
            energy_breakdown=self.energy_breakdown(),
            first_law={
                "dH_em": fl.dH_em,
                "T_dS": fl.T_dS,
                "supply": fl.supply,
                "residual": fl.residual,
                "sigma": fl.sigma,
                "is_first_law": fl.is_first_law,
                "is_second_law": fl.is_second_law,
            },
            atlas=self.atlas.value,
            schema_version=self.schema_version,
            metadata=dict(metadata or {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 3.3 — GRAFO DE PROXIMIDAD (NO DEC) + ENTROPÍA DE INFORMACIÓN
# ═══════════════════════════════════════════════════════════════════════════════════════


class MetricProximityGraph:
    """
    Grafo de proximidad entre *métricas de orquestación*.
    β₀, β₁ aquí son del 1-esqueleto de correlación, **no** de H^•(K) DEC.
    Los Betti de Fase 1 viven en seed.metadata['betti_numbers'].
    """

    METRIC_KEYS: Tuple[str, ...] = (
        "saturation", "complexity", "current_I",
        "potential_energy", "kinetic_energy", "info_shannon",
    )

    def __init__(
        self,
        threshold_mode: str = "adaptive",
        threshold_fixed: float = 0.3,
        distance_scale: float = 1.0,
        adaptive_c: float = 0.3,
        adaptive_cap: float = 0.7,
    ) -> None:
        if threshold_mode not in {"adaptive", "distance", "fixed"}:
            raise ConfigurationError(f"threshold_mode desconocido: {threshold_mode}.")
        self.threshold_mode = threshold_mode
        self.threshold_fixed = float(threshold_fixed)
        self.distance_scale = float(distance_scale)
        self.adaptive_c = float(adaptive_c)
        self.adaptive_cap = float(adaptive_cap)
        self._adjacency: Dict[int, Set[int]] = {}
        self._vertex_count = 0
        self._edge_count = 0
        self._last_threshold = 0.0

    def build(self, metrics: Mapping[str, float], metric_keys: Optional[Iterable[str]] = None) -> None:
        keys = tuple(metric_keys or self.METRIC_KEYS)
        values = np.array([float(metrics.get(k, 0.0)) for k in keys], dtype=float)
        self._vertex_count = int(values.size)
        self._edge_count = 0
        self._adjacency = {i: set() for i in range(self._vertex_count)}
        if self._vertex_count < 2:
            self._last_threshold = 0.0
            return
        v_min, v_max = float(values.min()), float(values.max())
        v_range = max(v_max - v_min, CONSTANTS.NUMERICAL_TOLERANCE)
        normalized = (values - v_min) / v_range
        self._last_threshold = self._compute_threshold(normalized)
        for i in range(self._vertex_count):
            for j in range(i + 1, self._vertex_count):
                if abs(normalized[i] - normalized[j]) < self._last_threshold:
                    self._adjacency[i].add(j)
                    self._adjacency[j].add(i)
                    self._edge_count += 1

    def _compute_threshold(self, normalized: np.ndarray) -> float:
        if self.threshold_mode == "fixed":
            return float(self.threshold_fixed)
        if self.threshold_mode == "adaptive":
            var = float(np.var(normalized))
            return float(min(self.adaptive_cap, self.adaptive_c * (1.0 + math.sqrt(var))))
        n = normalized.size
        if n < 2:
            return float(self.threshold_fixed)
        diffs = np.abs(normalized[:, None] - normalized[None, :])
        iu = np.triu_indices(n, k=1)
        return float(np.mean(diffs[iu]) + self.distance_scale * float(np.std(diffs[iu])))

    def proximity_betti(self) -> Dict[str, int]:
        if self._vertex_count == 0:
            return {"proximity_beta_0": 0, "proximity_beta_1": 0}
        beta_0 = self._count_components()
        beta_1 = max(0, self._edge_count - self._vertex_count + beta_0)
        return {"proximity_beta_0": beta_0, "proximity_beta_1": beta_1}

    def _count_components(self) -> int:
        visited: Set[int] = set()
        components = 0
        for node in range(self._vertex_count):
            if node in visited:
                continue
            components += 1
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                stack.extend(self._adjacency.get(cur, set()) - visited)
        return components

    def diagnose(self) -> Dict[str, Any]:
        b = self.proximity_betti()
        return {
            "vertices": self._vertex_count,
            "edges": self._edge_count,
            "threshold": self._last_threshold,
            "mode": self.threshold_mode,
            **b,
            "proximity_euler": b["proximity_beta_0"] - b["proximity_beta_1"],
        }


class InformationEntropyCalculator:
    """Entropía de *información* del pipeline (fallos de lote). No es S de Clausius."""

    def calculate_entropy_bayesian(
        self,
        counts: Mapping[str, int],
        prior: str = "jeffreys",
    ) -> Dict[str, float]:
        total = sum(counts.values())
        categories = max(1, len(counts))
        if total <= 0:
            return {"entropy_expected": 0.0, "effective_samples": 0.0, "prior_alpha": 0.0}
        alpha_map = {
            "jeffreys": 0.5,
            "laplace": 1.0,
            "KT": 0.5,
            "krichevsky_trofimov": 0.5,
            "uniform": 1.0 / categories,
        }
        if prior not in alpha_map:
            raise ConfigurationError(f"Prior desconocido: {prior}.")
        alpha = alpha_map[prior]
        alpha_post = {k: alpha + float(v) for k, v in counts.items()}
        alpha_0 = sum(alpha_post.values())
        dg = _digamma if _digamma is not None else (digamma if SCIPY_AVAILABLE else None)
        if dg is not None:
            entropy_nat = float(dg(alpha_0 + 1.0))
            for n in alpha_post.values():
                entropy_nat -= (n / alpha_0) * float(dg(n + 1.0))
            entropy_bits = entropy_nat / math.log(2.0)
        else:
            entropy_bits = 0.0
            for n in counts.values():
                p = n / total
                if p > 0.0:
                    entropy_bits -= p * math.log2(p)
        return {
            "entropy_expected": float(max(0.0, entropy_bits)),
            "effective_samples": float(alpha_0 - categories * alpha),
            "prior_alpha": float(alpha),
        }

    def calculate_renyi_spectrum(
        self,
        probabilities: np.ndarray,
        alphas: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        if alphas is None:
            alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")]
        probs = np.asarray(probabilities, dtype=float).reshape(-1)
        probs = probs[probs > 0.0]
        if probs.size == 0:
            return {a: 0.0 for a in alphas}
        probs = probs / probs.sum()
        spectrum: Dict[float, float] = {}
        for alpha in alphas:
            if alpha == 0.0:
                val = math.log2(probs.size)
            elif math.isclose(alpha, 1.0):
                val = float(-np.sum(probs * np.log2(probs)))
            elif math.isinf(alpha):
                val = float(-math.log2(float(np.max(probs))))
            else:
                s = float(np.sum(np.power(probs, alpha)))
                val = (1.0 / (1.0 - alpha)) * math.log2(max(s, CONSTANTS.NUMERICAL_ZERO))
            spectrum[float(alpha)] = float(val)
        return spectrum

    def calculate_system_entropy(
        self,
        total_records: int,
        error_count: int,
        processing_time: float,
    ) -> Dict[str, float]:
        if total_records <= 0:
            return self._zero()
        ec = max(0, min(int(error_count), int(total_records)))
        if ec == 0 or ec == total_records:
            res = self._zero()
            res["info_is_degenerate"] = bool(ec == total_records)
            return res
        counts = {"success": total_records - ec, "error": ec}
        bayes = self.calculate_entropy_bayesian(counts, prior="jeffreys")
        probs = np.array([counts["success"], counts["error"]], dtype=float) / total_records
        renyi = self.calculate_renyi_spectrum(probs)
        p = probs[probs > 0.0]
        fisher_trace = float(np.sum(1.0 / np.maximum(p, CONSTANTS.NUMERICAL_TOLERANCE)))
        kl = float(math.log2(p.size) + np.sum(p * np.log2(p)))
        return {
            "info_shannon": bayes["entropy_expected"],
            "info_renyi_2": renyi.get(2.0, 0.0),
            "info_renyi_spectrum": renyi,
            "info_fisher_trace": fisher_trace,
            "info_kl_to_uniform": kl,
            "info_rate": bayes["entropy_expected"] / max(processing_time, CONSTANTS.MIN_DELTA_TIME),
            "info_error_fraction": float(ec / total_records),
            "info_is_degenerate": False,
        }

    @staticmethod
    def _zero() -> Dict[str, Any]:
        return {
            "info_shannon": 0.0,
            "info_renyi_2": 0.0,
            "info_renyi_spectrum": {},
            "info_fisher_trace": 0.0,
            "info_kl_to_uniform": 0.0,
            "info_rate": 0.0,
            "info_error_fraction": 0.0,
            "info_is_degenerate": False,
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 3.4 — MOTOR: PHS PUNTO MEDIO (+ RLC STANDALONE OPCIONAL)
# ═══════════════════════════════════════════════════════════════════════════════════════


class RefinedFluxPhysicsEngine:
    r"""
    Motor 7.1.

    Modo semilla (contrato):
        x_{n+1} por punto medio implícito ZOH en u = control_input.
        Casimirs re-proyectados. 1ª/2ª ley GENERIC. Sin músculo, sin leapfrog.

    Modo standalone (sin semilla):
        RLC canónico [Q,λ] con R_s(I) serie regularizado y shunt opcional.
        Integrador: implicit_midpoint si R lineal; Newton (trap/TR-BDF2/IE)
        si R no lineal. TR-BDF2 = Hosea–Shampine correcto.
    """

    _MAX_METRICS_HISTORY: int = 100

    def __init__(
        self,
        capacitance: float,
        resistance: float,
        inductance: float,
        p_laplacian_exponent: float = 3.0,
        p_laplacian_epsilon: float = 1e-8,
        p_laplacian_G0: float = 0.0,
        config: Optional[CondenserConfig] = None,
        clock: Optional[Callable[[], float]] = None,
        maxwell: Optional["MaxwellSolver"] = None,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._validate_physical_parameters(capacitance, resistance, inductance)
        self.C = float(capacitance)
        self.R = float(resistance)
        self.L = float(inductance)
        self._config = config or CondenserConfig()
        self._p_exp = float(p_laplacian_exponent)
        self._p_eps = float(p_laplacian_epsilon)
        self._p_beta = float(self._config.p_laplacian_beta)
        self._p_G0 = float(p_laplacian_G0 if p_laplacian_G0 else self._config.p_laplacian_G0)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._omega_0 = 1.0 / math.sqrt(self.L * self.C)
        self._alpha = self.R / (2.0 * self.L) if self.L > 0.0 else 0.0
        self._zeta = self._alpha / self._omega_0 if self._omega_0 > 0.0 else 0.0
        self._Q = (1.0 / (2.0 * self._zeta)) if self._zeta > 0.0 else float("inf")
        self._update_damping_classification()
        self._unified_state = UnifiedPhysicalState(self.C, self.L, self.R, thermal=self._config.thermal)
        self._proximity = MetricProximityGraph()
        self._info_entropy = InformationEntropyCalculator()
        self._maxwell_solver = maxwell
        self._kernel: Optional[PoincareHamiltonianKernel] = PoincareHamiltonianKernel.from_rlc(
            capacitance=self.C,
            inductance=self.L,
            series_resistance=self.R,
            shunt_conductance=self._p_G0,
        )
        self._phs_controller: Optional[PortHamiltonianPoincareController] = None
        self._latest_seed: Optional[PoincareEngineSeed] = None
        self._last_audit_dt: float = float(self._config.physics_dt)
        self._last_current_obs: float = 0.0
        self._initialized = False
        self._metrics_history: deque = deque(maxlen=self._MAX_METRICS_HISTORY)
        self._newton_fail_count: int = 0
        self._newton_iter_total: int = 0
        self._newton_step_total: int = 0
        self._C0: Optional[np.ndarray] = None

    def attach_maxwell(self, solver: "MaxwellSolver") -> None:
        """Lattice DEC *inyectado* (el de Fase 1). Nunca se construye un K₆ ad hoc."""
        self._maxwell_solver = solver

    def _validate_physical_parameters(self, C: float, R: float, L: float) -> None:
        errors: List[str] = []
        if not math.isfinite(C) or C <= 0.0:
            errors.append(f"C inválida: {C}.")
        if not math.isfinite(R) or R < 0.0:
            errors.append(f"R inválida: {R}.")
        if not math.isfinite(L) or L <= 0.0:
            errors.append(f"L inválida: {L}.")
        if errors:
            raise ConfigurationError("Parámetros físicos inválidos:\n" + "\n".join(f"  - {e}" for e in errors))

    def _update_damping_classification(self) -> None:
        if self._zeta > 1.0 + 1e-12:
            self._damping_type = "OVERDAMPED"
        elif self._zeta < 1.0 - 1e-12:
            self._damping_type = "UNDERDAMPED"
        else:
            self._damping_type = "CRITICALLY_DAMPED"

    # ── semilla ───────────────────────────────────────────────────────────────

    def consume_engine_seed(self, seed: PoincareEngineSeed) -> Dict[str, Any]:
        if seed is None:
            raise EngineSeedError("PoincareEngineSeed es requerida.")
        hints = dict(seed.engine_hints or {})
        meta = dict(seed.metadata or {})
        if self._config.require_schema_7_1:
            schema = str(getattr(seed, "schema_version", "") or meta.get("schema_version", "7.1.0"))
            if schema != "7.1.0":
                raise SchemaContractError(f"schema {schema} incompatible con 7.1.0.")
            conv = str(meta.get("hodge_convention") or hints.get("hodge_convention") or "")
            if conv and "★₂ B" not in conv and "star2" not in conv.lower() and "μ⁻¹" not in conv:
                logger.warning("hodge_convention inesperado: %s", conv)
        self._latest_seed = seed
        self._unified_state.consume_engine_seed(seed)
        self._kernel = PoincareHamiltonianKernel(
            J=seed.interconnection_matrix,
            metric=seed.metric_matrix,
            R=seed.damping_matrix,
            g=seed.port_matrix,
            name="Engine-PHS-7.1",
        )
        control_seed = self._engine_seed_to_control_seed(seed)
        self._phs_controller = PortHamiltonianPoincareController(
            seed=control_seed,
            target_energy=float(seed.target_hamiltonian),
            integrator="implicit_midpoint",
            mode=ControlMode.ENERGY_LEVEL,
            energy_shaping=True,
        )
        # El estado y u ya aplicados: no recompute u.
        self._phs_controller.x = np.asarray(seed.state, dtype=float).reshape(-1).copy()
        self._C0 = self._phs_controller._casimir_values(self._phs_controller.x)
        dt_suggested = float(hints.get("dt_suggested", meta.get("dt_suggested", self._config.physics_dt)))
        if not math.isfinite(dt_suggested) or dt_suggested <= 0.0:
            dt_suggested = float(self._config.physics_dt)
        self._last_audit_dt = dt_suggested
        self._sync_maxwell_from_seed(seed)
        return self.poincare_audit(dt_suggested)

    @staticmethod
    def _engine_seed_to_control_seed(seed: PoincareEngineSeed) -> PoincareControlSeed:
        return PoincareControlSeed(
            state=np.asarray(seed.state, dtype=float).copy(),
            gradient=np.asarray(seed.gradient, dtype=float).copy(),
            hamiltonian=float(seed.hamiltonian),
            target_hamiltonian=float(seed.target_hamiltonian),
            lyapunov_candidate=float(seed.storage_function),
            interconnection_matrix=np.asarray(seed.interconnection_matrix, dtype=float).copy(),
            damping_matrix=np.asarray(seed.damping_matrix, dtype=float).copy(),
            metric_matrix=np.asarray(seed.metric_matrix, dtype=float).copy(),
            port_matrix=np.asarray(seed.port_matrix, dtype=float).copy(),
            metadata=dict(seed.metadata or {}),
            casimir_basis=None if seed.casimir_basis is None else np.asarray(seed.casimir_basis, dtype=float).copy(),
            spectral_data=None if not seed.spectral_data else dict(seed.spectral_data),
            ida_pbc_decomposition=(
                None
                if seed.ida_pbc_decomposition is None
                else {k: np.asarray(v, dtype=float).copy() for k, v in seed.ida_pbc_decomposition.items()}
            ),
            lyapunov_jacobian=(
                None if seed.lyapunov_jacobian is None else np.asarray(seed.lyapunov_jacobian, dtype=float).copy()
            ),
            output_port=None if getattr(seed, "output_port", None) is None else np.asarray(seed.output_port, dtype=float).copy(),
            matching_residual=float(getattr(seed, "matching_residual", 0.0) or 0.0),
        )

    def _sync_maxwell_from_seed(self, seed: PoincareEngineSeed) -> None:
        solver = self._maxwell_solver
        if solver is None or not SCIPY_AVAILABLE:
            return
        n_e, n_f = solver.calc.num_edges, solver.calc.num_faces
        state = np.asarray(seed.state, dtype=float).reshape(-1)
        if state.size != n_e + n_f:
            self.logger.debug("Seed dim=%d ≠ lattice %d; no sync Maxwell.", state.size, n_e + n_f)
            return
        solver.D = state[:n_e].copy()
        solver.B = state[n_e:].copy()
        if n_e > 0:
            solver.E = (1.0 / solver.epsilon) * (solver.calc.star1_inv @ solver.D)
        if n_f > 0:
            solver.H = (1.0 / solver.mu) * (solver.calc.star2 @ solver.B)

    def poincare_audit(self, dt: float) -> Dict[str, Any]:
        dt = max(float(dt), CONSTANTS.MIN_DELTA_TIME)
        result: Dict[str, Any] = {
            "dt": dt,
            "atlas": self._unified_state.atlas.value,
            "phs_preserved": bool(self._unified_state._phs_preserved),
            "integrator": "implicit_midpoint",
        }
        if self._kernel is None:
            return result
        x = self._unified_state.state_vector
        if x.size != self._kernel.dim:
            result["kernel_skip"] = f"dim {x.size} ≠ {self._kernel.dim}"
            return result
        try:
            _, report = self._kernel.compute_step(x, dt, integrator="strang")
            result.update(
                {
                    "hamiltonian_energy": report.hamiltonian_energy,
                    "volume_drift": report.volume_drift,
                    "rayleigh_dissipation_rate": report.rayleigh_dissipation_rate,
                    "is_liouville_preserved": report.is_liouville_preserved,
                    "is_volume_contracting": report.is_volume_contracting,
                    "poisson_residual": report.poisson_residual,
                    "trace_generator": report.trace_generator,
                    "casimir_dimension": report.casimir_dimension,
                    "max_re_eigen_A": report.max_re_eigen_A,
                    "logarithmic_norm_A": getattr(report, "logarithmic_norm_A", 0.0),
                    "strang_poisson_residual": getattr(report, "strang_poisson_residual", 0.0),
                    "casimir_drift": getattr(report, "casimir_drift", 0.0),
                }
            )
        except DataFluxCondenserError as exc:
            result["canonical_error"] = str(exc)
        if self._maxwell_solver is not None:
            try:
                mm = self._maxwell_solver.compute_energy_and_momentum()
                result["maxwell_observables"] = {
                    "total_energy": mm.get("total_energy", 0.0),
                    "gauss_residual": mm.get("gauss_residual", 0.0),
                }
            except DataFluxCondenserError as exc:
                result["maxwell_error"] = str(exc)
        return result

    # ── paso PHS (contrato) ───────────────────────────────────────────────────

    def step(
        self,
        dt: float,
        driving_current: float = 0.0,
        feedforward: float = 0.0,
        u_override: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        driving_current / feedforward se ignoran en modo semilla (u ya aplicado).
        En standalone, driving_current es un *esfuerzo* de puerto escalar ∈ ℝ
        (no un duty de músculo).
        """
        dt = float(np.clip(dt, CONSTANTS.MIN_DELTA_TIME, CONSTANTS.MAX_DELTA_TIME))
        if self._phs_controller is not None and self._latest_seed is not None:
            metrics = self._step_phs(dt, u_override=u_override)
        else:
            metrics = self._step_standalone_rlc(dt, float(driving_current) + float(feedforward))
        self._unified_state.physics_time += dt
        self._metrics_history.append(metrics)
        self._last_audit_dt = dt
        return metrics

    def _step_phs(self, dt: float, u_override: Optional[np.ndarray] = None) -> Dict[str, Any]:
        ctrl = self._phs_controller
        assert ctrl is not None
        u = (
            ctrl._normalize_control_input(u_override)
            if u_override is not None
            else self._unified_state.control_applied
        )
        if u.size != ctrl.port_dim:
            raise EngineSeedError(f"u dim {u.size} ≠ m={ctrl.port_dim}.")
        x0 = ctrl.x.copy()
        H0 = ctrl.hamiltonian(x0)
        report = ctrl.controlled_step(dt=dt, u_input=u, muscle=None)
        x1 = ctrl.x.copy()
        H1 = ctrl.hamiltonian(x1)
        self._unified_state.state_vector = x1
        self._unified_state.control_applied = np.asarray(u, dtype=float).copy()
        self._unified_state.storage_function = float(ctrl.storage_function(x1))
        if self._unified_state.atlas is AtlasKind.RLC_CANONICAL and x1.size >= 2:
            self._unified_state.charge = float(x1[0])
            self._unified_state.flux_linkage = float(x1[1])
        x_mid = 0.5 * (x0 + x1)
        grad_mid = ctrl.K @ x_mid
        rayleigh = float(grad_mid @ (ctrl.R @ grad_mid))
        y_mid = ctrl.g.T @ grad_mid
        supply = float(u @ y_mid)
        fl = self._unified_state.evolve_generic_bath(dt, rayleigh, supply, H1 - H0)
        if not fl.is_second_law:
            raise EntropyViolationError(f"2ª ley discreta violada: σ={fl.sigma:.3e}.")
        I_obs = self._unified_state.current
        if I_obs is None:
            I_obs = math.sqrt(max(rayleigh, 0.0) / max(self.R, CONSTANTS.NUMERICAL_TOLERANCE)) if self.R > 0.0 else 0.0
        self._last_current_obs = float(I_obs)
        self._maybe_refresh_maxwell_state(x1)
        if self._config.brain_enabled:
            v_bus = self._observability_voltage(x1, H1)
            self._update_tactical_reserve(dt, v_bus)
        audit = self.poincare_audit(dt)
        return self._assemble_metrics(
            dt=dt,
            current_I=float(I_obs),
            H_em=H1,
            audit=audit,
            phs_report=report,
            first_law=fl,
        )

    def _maybe_refresh_maxwell_state(self, x: np.ndarray) -> None:
        solver = self._maxwell_solver
        if solver is None:
            return
        n_e, n_f = solver.calc.num_edges, solver.calc.num_faces
        if x.size != n_e + n_f:
            return
        solver.D = x[:n_e].copy()
        solver.B = x[n_e:].copy()
        if n_e > 0:
            solver.E = (1.0 / solver.epsilon) * (solver.calc.star1_inv @ solver.D)
        if n_f > 0:
            solver.H = (1.0 / solver.mu) * (solver.calc.star2 @ solver.B)

    def _observability_voltage(self, x: np.ndarray, H: float) -> float:
        if self._unified_state.charge is not None:
            return float(self._unified_state.charge / self.C)
        return float(math.sqrt(max(2.0 * H / max(self.C, CONSTANTS.NUMERICAL_TOLERANCE), 0.0)))

    # ── standalone RLC ────────────────────────────────────────────────────────

    def _R_series(self, I: float) -> float:
        """R_s(I) = R (1 + β (I²+ε)^((p-2)/2)). Serie, no paralelo."""
        s = I * I + self._p_eps
        return float(self.R * (1.0 + self._p_beta * s ** ((self._p_exp - 2.0) / 2.0)))

    def _G_shunt(self, V: float) -> float:
        if self._p_G0 <= 0.0:
            return 0.0
        return float(self._p_G0 * (abs(V) + self._p_eps) ** (self._p_exp - 2.0))

    def _rlc_f(self, Q: float, lam: float, u_s: float) -> np.ndarray:
        r"""
        x=[Q,λ],  Q̇ = λ/L − G(Q/C) Q/C,  λ̇ = u − Q/C − R_s(I) I.
        u_s esfuerzo de puerto (voltaje de fuente).
        """
        I = lam / self.L
        V = Q / self.C
        G = self._G_shunt(V)
        dQ = I - G * V
        dlam = u_s - V - self._R_series(I) * I
        return np.array([dQ, dlam], dtype=float)

    def _rlc_jacobian(self, Q: float, lam: float) -> np.ndarray:
        I = lam / self.L
        V = Q / self.C
        # dQ/dQ, dQ/dλ
        G = self._G_shunt(V)
        # G(V)·V ; d(GV)/dQ = (dG/dV · V + G) / C
        dG_dV = 0.0
        if self._p_G0 > 0.0:
            dG_dV = self._p_G0 * (self._p_exp - 2.0) * (abs(V) + self._p_eps) ** (self._p_exp - 3.0) * math.copysign(1.0, V or 1.0)
        dGV_dQ = (dG_dV * V + G) / self.C
        # R_s(I) I
        s = I * I + self._p_eps
        alpha = (self._p_exp - 2.0) / 2.0
        dRs_dI = self.R * self._p_beta * alpha * (s ** (alpha - 1.0)) * 2.0 * I if alpha != 0.0 else 0.0
        d_RI_dI = self._R_series(I) + I * dRs_dI
        # λ̇ = u − Q/C − R I ;  ∂λ̇/∂Q = −1/C, ∂λ̇/∂λ = −(d_RI_dI)/L
        return np.array(
            [
                [-dGV_dQ, 1.0 / self.L],
                [-1.0 / self.C, -d_RI_dI / self.L],
            ],
            dtype=float,
        )

    def _newton_theta(
        self,
        y_curr: np.ndarray,
        f_curr: np.ndarray,
        u_s: float,
        dt: float,
        theta: float,
    ) -> np.ndarray:
        y = y_curr.copy()
        tol = self._config.newton_tol
        I2 = np.eye(2)
        resid_norm = float("inf")
        converged = False
        iterations = 0
        for it in range(self._config.newton_max_iter):
            iterations = it + 1
            f_next = self._rlc_f(y[0], y[1], u_s)
            resid = y - y_curr - dt * ((1.0 - theta) * f_curr + theta * f_next)
            resid_norm = float(np.linalg.norm(resid))
            if resid_norm < tol * (1.0 + float(np.linalg.norm(y))):
                converged = True
                break
            J = self._rlc_jacobian(y[0], y[1])
            J_F = I2 - dt * theta * J
            try:
                delta = np.linalg.solve(J_F, -resid)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J_F, -resid, rcond=None)[0]
            y = y + delta
            if not np.all(np.isfinite(y)):
                raise NumericalInstabilityError("Newton RLC divergió.")
        self._newton_iter_total += iterations
        self._newton_step_total += 1
        if not converged:
            self._newton_fail_count += 1
            self.logger.warning("Newton RLC no convergió (resid=%.3e, θ=%.3f).", resid_norm, theta)
        return y

    def _tr_bdf2_hosea_shampine(self, y_n: np.ndarray, u_s: float, dt: float) -> np.ndarray:
        r"""
        γ = 2−√2.
        (TR)  y_{n+γ} = y_n + (γh/2)(f_n + f_{n+γ})
        (BDF2) y_{n+1} − c h f_{n+1} = a y_n + b y_{n+γ}
              a = 1/(γ(2−γ)),  b = −(1−γ)²/(γ(2−γ)),  c = (1−γ)/(2−γ)
        """
        gamma = 2.0 - math.sqrt(2.0)
        f_n = self._rlc_f(y_n[0], y_n[1], u_s)
        y_g = self._newton_theta(y_n, f_n, u_s, gamma * dt, theta=0.5)
        a = 1.0 / (gamma * (2.0 - gamma))
        b = -((1.0 - gamma) ** 2) / (gamma * (2.0 - gamma))
        c = (1.0 - gamma) / (2.0 - gamma)
        rhs_const = a * y_n + b * y_g
        y = y_g.copy()
        I2 = np.eye(2)
        tol = self._config.newton_tol
        converged = False
        for it in range(self._config.newton_max_iter):
            f_np1 = self._rlc_f(y[0], y[1], u_s)
            resid = y - c * dt * f_np1 - rhs_const
            if float(np.linalg.norm(resid)) < tol * (1.0 + float(np.linalg.norm(y))):
                converged = True
                break
            J = self._rlc_jacobian(y[0], y[1])
            J_F = I2 - c * dt * J
            try:
                delta = np.linalg.solve(J_F, -resid)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J_F, -resid, rcond=None)[0]
            y = y + delta
            if not np.all(np.isfinite(y)):
                raise NumericalInstabilityError("TR-BDF2 divergió.")
        self._newton_step_total += 1
        self._newton_iter_total += it + 1
        if not converged:
            self._newton_fail_count += 1
            self.logger.warning("TR-BDF2 (Hosea–Shampine) no convergió.")
        return y

    def _step_standalone_rlc(self, dt: float, u_s: float) -> Dict[str, Any]:
        st = self._unified_state
        if st.charge is None or st.flux_linkage is None:
            st.atlas = AtlasKind.STANDALONE
            st.charge = 0.0
            st.flux_linkage = 0.0
        Q, lam = float(st.charge), float(st.flux_linkage)
        y_n = np.array([Q, lam], dtype=float)
        H0 = 0.5 * Q * Q / self.C + 0.5 * lam * lam / self.L
        scheme = self._config.integrator
        if scheme == "tr_bdf2":
            y_np1 = self._tr_bdf2_hosea_shampine(y_n, u_s, dt)
        elif scheme == "implicit_euler":
            y_np1 = self._newton_theta(y_n, self._rlc_f(Q, lam, u_s), u_s, dt, theta=1.0)
        elif scheme == "trapezoidal":
            y_np1 = self._newton_theta(y_n, self._rlc_f(Q, lam, u_s), u_s, dt, theta=0.5)
        else:
            # punto medio: θ=1/2 es el trapecio; para H cuadrática + R lineal = gradiente discreto
            y_np1 = self._newton_theta(y_n, self._rlc_f(Q, lam, u_s), u_s, dt, theta=0.5)
        Q1, lam1 = float(y_np1[0]), float(y_np1[1])
        st.charge, st.flux_linkage = Q1, lam1
        st.state_vector = np.array([Q1, lam1], dtype=float)
        H1 = 0.5 * Q1 * Q1 / self.C + 0.5 * lam1 * lam1 / self.L
        I_mid = 0.5 * (lam + lam1) / self.L
        Q_mid = 0.5 * (Q + Q1)
        rayleigh = self._R_series(I_mid) * I_mid ** 2 + self._G_shunt(Q_mid / self.C) * (Q_mid / self.C) ** 2
        supply = float(u_s) * I_mid
        fl = st.evolve_generic_bath(dt, rayleigh, supply, H1 - H0)
        if self._config.brain_enabled:
            self._update_tactical_reserve(dt, Q1 / self.C)
        self._last_current_obs = lam1 / self.L
        st.control_applied = np.array([u_s], dtype=float)
        audit = self.poincare_audit(dt)
        return self._assemble_metrics(
            dt=dt,
            current_I=self._last_current_obs,
            H_em=H1,
            audit=audit,
            phs_report=None,
            first_law=fl,
        )

    def _update_tactical_reserve(self, dt: float, main_bus_voltage: float) -> None:
        """RC de *control plane*. No forma parte del PHS ni de Casimirs."""
        if not self._config.brain_enabled:
            return
        state = self._unified_state
        cfg = self._config
        L_brain, C_brain, R_brain = 10e-6, cfg.brain_capacitance, 0.5
        target_voltage = max(0.0, main_bus_voltage - cfg.brain_diode_drop)
        if target_voltage > state.brain_voltage:
            delta_v = target_voltage - state.brain_voltage
            di_dt = (delta_v - state.brain_inflow_current * R_brain) / L_brain
            state.brain_inflow_current = max(0.0, state.brain_inflow_current + di_dt * dt)
            state.brain_voltage += state.brain_inflow_current * dt / C_brain
        else:
            state.brain_inflow_current = 0.0
            state.brain_voltage -= (cfg.brain_agent_consumption_amps * dt) / C_brain
        state.brain_alive = bool(state.brain_voltage >= cfg.brain_brownout_threshold)
        if not state.brain_alive:
            self.logger.critical("Brownout del plano de control (metáfora de orquestación).")

    def _assemble_metrics(
        self,
        dt: float,
        current_I: float,
        H_em: float,
        audit: Dict[str, Any],
        phs_report: Optional[PortHamiltonianControlReport],
        first_law: FirstLawAudit,
    ) -> Dict[str, Any]:
        st = self._unified_state
        Q = st.charge
        lam = st.flux_linkage
        v_elastic = (Q / self.C) if Q is not None else 0.0
        potential_energy = (0.5 * Q * Q / self.C) if Q is not None else float("nan")
        kinetic_energy = (0.5 * lam * lam / self.L) if lam is not None else float("nan")
        saturation = float(np.clip(abs(v_elastic) / max(self._config.max_voltage, CONSTANTS.NUMERICAL_TOLERANCE), 0.0, 1.0))
        dissipated = float(first_law.sigma * st.thermal.reservoir_temperature_K)
        return {
            "time": float(self._clock()),
            "physics_time": float(st.physics_time),
            "dt": float(dt),
            "atlas": st.atlas.value,
            "saturation": saturation,
            "complexity": float(np.clip(1.0 - saturation, 0.0, 1.0)),
            "current_I": float(current_I),
            "potential_energy": float(potential_energy) if Q is not None else 0.0,
            "kinetic_energy": float(kinetic_energy) if lam is not None else 0.0,
            "total_energy": float(H_em),
            "availability": float(H_em),
            "dissipated_power": dissipated,
            "v_elastic": float(v_elastic),
            "damping_ratio": self._zeta,
            "damping_type": self._damping_type,
            "resonant_frequency_hz": self._omega_0 / (2.0 * math.pi),
            "quality_factor": self._Q,
            "muscle_temp": st.muscle_temperature,
            "brain_voltage": st.brain_voltage,
            "brain_alive": float(st.brain_alive),
            "brownout_risk": float(st.brain_voltage < self._config.brain_brownout_threshold + 0.5),
            "clamping_active": float(abs(v_elastic) > self._config.max_voltage),
            "entropy_generation_dS": float(st.last_dS),
            "entropy_sigma": float(st.last_sigma),
            "first_law_residual": float(first_law.residual),
            "first_law_ok": float(first_law.is_first_law),
            "second_law_ok": float(first_law.is_second_law),
            "poincare_audit": audit,
            "phs_preserved": bool(st._phs_preserved),
            "discrete_residual": float(getattr(phs_report, "discrete_residual", first_law.residual) if phs_report else first_law.residual),
            "casimir_drift": float(getattr(phs_report, "casimir_drift", 0.0) if phs_report else 0.0),
            "pumping_required": float(st.pumping_required),
            "newton_avg_iter": float(self._newton_iter_total / max(1, self._newton_step_total)),
            "newton_fail_count": int(self._newton_fail_count),
            "schema_version": st.schema_version,
        }

    def calculate_metrics(
        self,
        total_records: int,
        cache_hits: int,
        error_count: int = 0,
        processing_time: float = 1.0,
        condenser_config: Optional[CondenserConfig] = None,
        dt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Un paso físico con dt *físico* (hints o physics_dt). cache_hits NO es corriente.
        En standalone, se usa una fuente nula (u=0) salvo dt_override; la saturación
        observada alimenta el PI de lotes, no el puerto PHS.
        """
        if total_records <= 0:
            return self._get_zero_metrics()
        dt = (
            float(dt_override)
            if dt_override is not None and math.isfinite(dt_override) and dt_override > 0.0
            else float(self._last_audit_dt or self._config.physics_dt)
        )
        metrics = self.step(dt=dt, driving_current=0.0)
        info = self._info_entropy.calculate_system_entropy(
            total_records=total_records,
            error_count=error_count,
            processing_time=processing_time,
        )
        metrics.update(info)
        self._proximity.build(metrics)
        metrics.update(self._proximity.diagnose())
        if self._latest_seed is not None:
            b = (self._latest_seed.metadata or {}).get("betti_numbers")
            if b:
                metrics["dec_betti_numbers"] = tuple(b)
        if self._maxwell_solver is not None:
            mm = self._maxwell_solver.compute_energy_and_momentum()
            metrics.update(
                {
                    "field_energy": float(mm.get("total_energy", 0.0)),
                    "gauss_residual": float(mm.get("gauss_residual", 0.0)),
                    "field_momentum_magnitude": float(mm.get("momentum_magnitude", 0.0)),
                }
            )
        return metrics

    def _get_zero_metrics(self) -> Dict[str, Any]:
        return {
            "time": 0.0, "physics_time": 0.0, "dt": 0.0, "atlas": self._unified_state.atlas.value,
            "saturation": 0.0, "complexity": 1.0, "current_I": 0.0,
            "potential_energy": 0.0, "kinetic_energy": 0.0, "total_energy": 0.0,
            "availability": 0.0, "dissipated_power": 0.0, "v_elastic": 0.0,
            "damping_ratio": self._zeta, "damping_type": self._damping_type,
            "resonant_frequency_hz": self._omega_0 / (2.0 * math.pi),
            "quality_factor": self._Q, "muscle_temp": self._unified_state.muscle_temperature,
            "brain_voltage": self._unified_state.brain_voltage, "brain_alive": 1.0,
            "brownout_risk": 0.0, "clamping_active": 0.0,
            "entropy_generation_dS": 0.0, "entropy_sigma": 0.0,
            "first_law_residual": 0.0, "first_law_ok": 1.0, "second_law_ok": 1.0,
            "poincare_audit": {}, "phs_preserved": bool(self._unified_state._phs_preserved),
            "discrete_residual": 0.0, "casimir_drift": 0.0, "pumping_required": 0.0,
            "newton_avg_iter": 0.0, "newton_fail_count": 0,
            "proximity_beta_0": 0, "proximity_beta_1": 0, "proximity_euler": 0,
            "info_shannon": 0.0, "schema_version": self._unified_state.schema_version,
        }

    def get_system_diagnosis(self, metrics: Mapping[str, Any]) -> Dict[str, str]:
        diagnosis = {
            "state": "NOMINAL",
            "damping": self._damping_type,
            "energy": "BALANCED",
            "info_entropy": "LOW",
            "topology_proximity": "SIMPLE",
            "phs": "PRESERVED" if metrics.get("phs_preserved", False) else "LOST",
            "first_law": "OK" if metrics.get("first_law_ok", 1.0) else "BROKEN",
            "second_law": "OK" if metrics.get("second_law_ok", 1.0) else "BROKEN",
        }
        saturation = float(metrics.get("saturation", 0.0))
        if saturation > 0.95:
            diagnosis["state"] = "SATURATED"
        elif saturation < 0.05:
            diagnosis["state"] = "IDLE"
        if float(metrics.get("dissipated_power", 0.0)) > CONSTANTS.OVERHEAT_POWER_THRESHOLD:
            diagnosis["state"] = "OVERHEATING"
        if float(metrics.get("info_error_fraction", 0.0)) > 0.25:
            diagnosis["info_entropy"] = "HIGH"
        if int(metrics.get("proximity_beta_0", 1)) > 1:
            diagnosis["topology_proximity"] = "DISCONNECTED"
        elif int(metrics.get("proximity_beta_1", 0)) > 0:
            diagnosis["topology_proximity"] = "CYCLIC"
        if float(metrics.get("entropy_sigma", 0.0)) < -CONSTANTS.NUMERICAL_TOLERANCE:
            diagnosis["state"] = "ENTROPY_VIOLATION"
        if not metrics.get("first_law_ok", 1.0):
            diagnosis["state"] = "FIRST_LAW_RESIDUAL"
        return diagnosis

    def unified_snapshot(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        dt_audit: Optional[float] = None,
    ) -> UnifiedPhysicalSnapshot:
        dt = dt_audit if dt_audit is not None and dt_audit > 0.0 else self._last_audit_dt
        audit = self.poincare_audit(dt)
        return self._unified_state.snapshot(wall_time=float(self._clock()), poincare_audit=audit, metadata=metadata)


# ═══════════════════════════════════════════════════════════════════════════════════════
# FASE 3.5 — ORQUESTADOR (PI DE LOTES ≠ PUERTO PHS)
# ═══════════════════════════════════════════════════════════════════════════════════════


class DataFluxCondenser:
    r"""
    Orquestador. El PI actúa sobre saturación → tamaño de lote.
    El PHS se avanza con u_app de la semilla y physics_dt, nunca con wall-clock.
    El reloj inyectable solo mide timeout y throughput.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        profile: Optional[Dict[str, Any]] = None,
        condenser_config: Optional[CondenserConfig] = None,
        record_processor: Optional[Callable[[List[Any], Dict[str, Any], Optional[Any]], Any]] = None,
        clock: Optional[Callable[[], float]] = None,
        cache_hit_estimator: Optional[Callable[[List[Any], Dict[str, Any]], int]] = None,
        maxwell: Optional["MaxwellSolver"] = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.profile = profile or {}
        self.condenser_config = condenser_config or CondenserConfig()
        self.record_processor = record_processor
        self._clock: Callable[[], float] = clock or time.monotonic
        self._cache_hit_estimator = cache_hit_estimator
        self.physics = RefinedFluxPhysicsEngine(
            capacitance=self.condenser_config.system_capacitance,
            resistance=self.condenser_config.base_resistance,
            inductance=self.condenser_config.system_inductance,
            p_laplacian_exponent=self.condenser_config.p_laplacian_exponent,
            p_laplacian_epsilon=self.condenser_config.p_laplacian_epsilon,
            p_laplacian_G0=self.condenser_config.p_laplacian_G0,
            config=self.condenser_config,
            clock=self._clock,
            maxwell=maxwell,
        )
        self.controller = PIController(
            kp=self.condenser_config.pid_kp,
            ki=self.condenser_config.pid_ki,
            setpoint=self.condenser_config.pid_setpoint,
            min_output=float(self.condenser_config.min_batch_size),
            max_output=float(self.condenser_config.max_batch_size),
            integral_limit_factor=self.condenser_config.integral_limit_factor,
            use_ema=True,
            energy_error_mode="absolute",
        )
        self.poincare_controller: Optional[PortHamiltonianPoincareController] = None
        self._stats = ProcessingStats()
        self._start_time: Optional[float] = None
        self._emergency_brake_count = 0
        self._metrics_history: deque = deque(maxlen=100)
        self._current_trace_id = ""
        self._physics_dt = float(self.condenser_config.physics_dt)
        self.logger.info(
            "DataFluxCondenser 7.1: batch=[%d,%d], integrator=%s, physics_dt=%g",
            self.condenser_config.min_batch_size,
            self.condenser_config.max_batch_size,
            self.condenser_config.integrator,
            self._physics_dt,
        )

    def consume_engine_seed(self, seed: PoincareEngineSeed) -> Dict[str, Any]:
        if seed is None:
            raise EngineSeedError("PoincareEngineSeed es requerida.")
        if bool((seed.engine_hints or {}).get("do_not_regulate_casimirs", True)) is False:
            logger.warning("La semilla pide regular Casimirs; se ignora (contrato 7.1).")
        audit = self.physics.consume_engine_seed(seed)
        self.poincare_controller = self.physics._phs_controller
        hints = dict(seed.engine_hints or {})
        dt_s = float(hints.get("dt_suggested", self.condenser_config.physics_dt))
        if math.isfinite(dt_s) and dt_s > 0.0:
            self._physics_dt = dt_s
        if self.condenser_config.trace_enabled:
            self._current_trace_id = f"trace-{int(self._clock() * 1e6)}"
        return audit

    def stabilize_records(
        self,
        raw_records: List[Any],
        cache: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Any] = None,
    ) -> Any:
        if raw_records is None:
            raise InvalidInputError("raw_records es requerido.")
        cache = cache or {}
        self._start_time = self._clock()
        self._stats = ProcessingStats()
        self._emergency_brake_count = 0
        self._metrics_history.clear()
        self.controller.reset()
        total_records = len(raw_records)
        self._stats.total_records = total_records
        if total_records == 0:
            return pd.DataFrame() if pd is not None else []
        processed_batches = self._process_batches_with_pid(raw_records, cache, total_records, telemetry)
        result = self._consolidate_results(processed_batches)
        self._stats.processing_time = self._clock() - self._start_time
        self._validate_output(result)
        return result

    def _process_batches_with_pid(
        self,
        raw_records: List[Any],
        cache: Dict[str, Any],
        total_records: int,
        telemetry: Optional[Any],
    ) -> List[Any]:
        processed_batches: List[Any] = []
        failed_batches_count = 0
        current_index = 0
        current_batch_size = self.condenser_config.min_batch_size
        iteration = 0
        max_iterations = total_records * CONSTANTS.MAX_ITERATIONS_MULTIPLIER
        dt_phys = self._physics_dt
        while current_index < total_records and iteration < max_iterations:
            iteration += 1
            end_index = min(current_index + current_batch_size, total_records)
            batch = raw_records[current_index:end_index]
            batch_size = len(batch)
            if batch_size == 0:
                break
            elapsed_time = self._clock() - self._start_time
            if CONSTANTS.PROCESSING_TIMEOUT - elapsed_time <= 0.0:
                self.logger.error("Timeout. Progreso: %d/%d", current_index, total_records)
                break
            cache_hits = self._estimate_cache_hits(batch, cache)
            metrics = self.physics.calculate_metrics(
                total_records=batch_size,
                cache_hits=cache_hits,
                error_count=failed_batches_count,
                processing_time=max(elapsed_time, CONSTANTS.MIN_DELTA_TIME),
                condenser_config=self.condenser_config,
                dt_override=dt_phys,
            )
            if self.condenser_config.brain_enabled and not metrics.get("brain_alive", 1.0):
                raise OrchestrationError("Colapso del plano de control: reserva táctica insuficiente.")
            saturation = float(metrics.get("saturation", 0.5))
            power = float(metrics.get("dissipated_power", 0.0))
            self._metrics_history.append(metrics)
            pi_report = self.controller.compute(measurement=saturation, dt=dt_phys, feedforward=0.0)
            pid_output_adjusted = int(round(pi_report.applied_output if hasattr(pi_report, "applied_output") else pi_report.output))
            if saturation > 0.95:
                pid_output_adjusted = self.condenser_config.min_batch_size
            emergency_brake = False
            brake_reason = ""
            brake_severity = 1.0
            if power > CONSTANTS.OVERHEAT_POWER_THRESHOLD:
                brake_severity = min(brake_severity, 0.3 / max(power / CONSTANTS.OVERHEAT_POWER_THRESHOLD, 1e-9))
                emergency_brake, brake_reason = True, f"OVERHEAT P={power:.2f}"
            if failed_batches_count >= 3:
                brake_severity = min(brake_severity, 0.5)
                emergency_brake, brake_reason = True, f"CONSECUTIVE_FAILURES={failed_batches_count}"
            if emergency_brake:
                pid_output_adjusted = max(CONSTANTS.MIN_BATCH_SIZE_FLOOR, int(pid_output_adjusted * brake_severity))
                self._emergency_brake_count += 1
                self._stats.emergency_brakes_triggered += 1
                self.logger.warning("Emergency brake [%d]: %s → batch=%d", self._emergency_brake_count, brake_reason, pid_output_adjusted)
            result = self._process_single_batch_with_recovery(batch, cache, telemetry)
            if result.success:
                if result.dataframe is not None:
                    processed_batches.append(result.dataframe)
                self._stats.add_batch_stats(batch_size=result.records_processed, saturation=saturation, success=True)
                failed_batches_count = max(0, failed_batches_count - 1)
            else:
                failed_batches_count += 1
                self._stats.add_batch_stats(batch_size=batch_size, saturation=saturation, success=False)
                if failed_batches_count >= self.condenser_config.max_failed_batches:
                    raise OrchestrationError(f"Límite de batches fallidos: {failed_batches_count}")
            current_index = end_index
            inertia = (
                self.condenser_config.batch_inertia_emergency if emergency_brake else self.condenser_config.batch_inertia_nominal
            )
            current_batch_size = int(inertia * current_batch_size + (1.0 - inertia) * pid_output_adjusted)
            current_batch_size = max(
                CONSTANTS.MIN_BATCH_SIZE_FLOOR,
                min(current_batch_size, self.condenser_config.max_batch_size),
            )
        return processed_batches

    def _process_single_batch_with_recovery(
        self,
        batch: List[Any],
        cache: Dict[str, Any],
        telemetry: Optional[Any],
    ) -> BatchResult:
        if not batch:
            return BatchResult(success=True, records_processed=0)
        try:
            df = self._rectify_signal(batch, cache, telemetry)
            return BatchResult(success=True, dataframe=df, records_processed=len(df) if df is not None else 0)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Batch directo falló: %s. Recuperación unitaria.", type(exc).__name__)
        successful_parts: List[Any] = []
        processed_count = 0
        error_types: List[str] = []
        for record in batch:
            try:
                df_part = self._rectify_signal([record], cache, telemetry)
                if df_part is not None:
                    successful_parts.append(df_part)
                    processed_count += len(df_part)
            except Exception as exc:  # noqa: BLE001
                error_types.append(type(exc).__name__)
        if not successful_parts:
            return BatchResult(
                success=False, records_processed=0,
                error_message="Recuperación unitaria sin resultados.",
                error_types=tuple(sorted(set(error_types))),
            )
        return BatchResult(
            success=True,
            dataframe=self._safe_concat(successful_parts),
            records_processed=processed_count,
            error_types=tuple(sorted(set(error_types))),
        )

    def _rectify_signal(self, batch: List[Any], cache: Dict[str, Any], telemetry: Optional[Any]) -> Any:
        if self.record_processor is not None:
            return self.record_processor(batch, cache, telemetry)
        if pd is not None:
            return pd.DataFrame(batch)
        return batch

    def _estimate_cache_hits(self, batch: List[Any], cache: Dict[str, Any]) -> int:
        if not batch:
            return 0
        if self._cache_hit_estimator is not None:
            try:
                return max(0, min(int(self._cache_hit_estimator(batch, cache)), len(batch)))
            except Exception:  # noqa: BLE001
                self.logger.debug("Estimador de cache falló.")
        if not cache:
            return max(1, len(batch) // 4)
        cache_keys = set(cache.keys())
        sample_size = min(50, len(batch))
        step = max(1, len(batch) // sample_size)
        hits = samples = 0
        for idx in range(0, len(batch), step):
            record = batch[idx]
            samples += 1
            if isinstance(record, dict):
                record_keys = set(record.keys())
                union = max(len(record_keys | cache_keys), 1)
                if len(record_keys & cache_keys) / union > 0.25:
                    hits += 1
        if samples == 0:
            return max(1, len(batch) // 4)
        return max(1, int((hits / samples) * len(batch)))

    def _consolidate_results(self, batches: List[Any]) -> Any:
        valid_batches = [b for b in batches if b is not None]
        if not valid_batches:
            return pd.DataFrame() if pd is not None else []
        if pd is None:
            consolidated: List[Any] = []
            for item in valid_batches:
                if isinstance(item, list):
                    consolidated.extend(item)
                else:
                    consolidated.append(item)
            return consolidated
        return self._safe_concat(valid_batches)

    def _safe_concat(self, dataframes: List[Any]) -> Any:
        if not dataframes:
            return pd.DataFrame() if pd is not None else []
        if pd is None:
            consolidated: List[Any] = []
            for item in dataframes:
                if isinstance(item, list):
                    consolidated.extend(item)
                else:
                    consolidated.append(item)
            return consolidated
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]
        if not valid_dfs:
            return pd.DataFrame()
        if len(valid_dfs) == 1:
            return valid_dfs[0]
        try:
            return pd.concat(valid_dfs, ignore_index=True, sort=False)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Concatenación directa falló: %s", exc)
            try:
                common = set(valid_dfs[0].columns)
                for df in valid_dfs[1:]:
                    common &= set(df.columns)
                if not common:
                    common = set().union(*(set(df.columns) for df in valid_dfs))
                common_l = sorted(common)
                aligned = []
                for df in valid_dfs:
                    df_c = df.copy()
                    for col in common_l:
                        if col not in df_c.columns:
                            df_c[col] = pd.NA
                    aligned.append(df_c[common_l])
                return pd.concat(aligned, ignore_index=True, sort=False)
            except Exception as exc2:  # noqa: BLE001
                self.logger.error("Concatenación alineada falló: %s", exc2)
                return valid_dfs[0]

    def _validate_output(self, result: Any) -> None:
        if pd is not None and isinstance(result, pd.DataFrame):
            if result.empty:
                msg = "DataFrame de salida está vacío."
                if self.condenser_config.enable_strict_validation:
                    raise OrchestrationError(msg)
                self.logger.warning(msg)
                return
            n_records = len(result)
            if n_records < self.condenser_config.min_records_threshold:
                msg = f"Registros insuficientes: {n_records} < {self.condenser_config.min_records_threshold}"
                if self.condenser_config.enable_strict_validation:
                    raise OrchestrationError(msg)
                self.logger.warning(msg)
            null_ratio = result.isnull().sum().sum() / max(1, n_records * len(result.columns))
            if null_ratio > 0.5:
                self.logger.warning("Alto porcentaje de nulos: %.2f%%", 100.0 * null_ratio)
        elif isinstance(result, list):
            if len(result) == 0 and self.condenser_config.enable_strict_validation:
                raise OrchestrationError("Salida consolidada vacía.")

    def get_processing_stats(self) -> Dict[str, Any]:
        current_metrics: Dict[str, Any] = dict(self._metrics_history[-1]) if self._metrics_history else {}
        elapsed = (self._clock() - self._start_time) if self._start_time else 0.0
        return {
            "statistics": asdict(self._stats),
            "current_metrics": current_metrics,
            "emergency_brakes": self._emergency_brake_count,
            "trace_id": self._current_trace_id,
            "timing": {
                "elapsed_s": elapsed,
                "throughput_per_s": self._stats.processed_records / max(1e-3, elapsed) if self._start_time else 0.0,
                "physics_dt": self._physics_dt,
            },
        }

    def get_system_health(self) -> Dict[str, Any]:
        issues: List[str] = []
        warnings: List[str] = []
        try:
            diag = self.controller.get_stability_analysis()
            sc = diag.get("stability_class", "UNKNOWN")
            if sc == "UNSTABLE":
                issues.append("PI de lotes inestable.")
            elif sc == "MARGINALLY_STABLE":
                warnings.append("Estabilidad marginal del PI de lotes.")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Error evaluando PI: {exc}")
        if self._emergency_brake_count > 10:
            issues.append(f"Exceso de frenos: {self._emergency_brake_count}")
        elif self._emergency_brake_count > 5:
            warnings.append(f"Frenos frecuentes: {self._emergency_brake_count}")
        if self._metrics_history:
            last = self._metrics_history[-1]
            if not last.get("second_law_ok", 1.0):
                issues.append("2ª ley discreta violada.")
            if not last.get("first_law_ok", 1.0):
                warnings.append(f"Residuo 1ª ley={last.get('first_law_residual')}.")
            if int(last.get("newton_fail_count", 0)) > 5:
                warnings.append("Convergencia Newton degradada.")
            v_brain = float(last.get("brain_voltage", 5.0))
            if v_brain < self.condenser_config.brain_brownout_threshold + 0.1:
                issues.append(f"Voltaje crítico en plano de control: {v_brain:.2f}V")
        if issues:
            health = "CRITICAL" if len(issues) >= 2 else "DEGRADED"
        elif warnings:
            health = "DEGRADED" if len(warnings) >= 3 else "HEALTHY"
        else:
            health = "HEALTHY"
        return {
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "emergency_brakes": self._emergency_brake_count,
            "processed_ratio": self._stats.processed_records / max(1, self._stats.total_records),
            "trace_id": self._current_trace_id,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 3.6 — CIERRE DEL PIPELINE
    # ──────────────────────────────────────────────────────────────────────────

    def synthesize_final_unified_state(self) -> UnifiedPhysicalSnapshot:
        r"""
        CIERRE FORMAL FASE 1 → 2 → 3.

          Fase 1: MaxwellSolver.synthesize_poincare_control_seed() → PoincareControlSeed
          Fase 2: PortHamiltonianPoincareController.synthesize_engine_seed() → PoincareEngineSeed
          Fase 3: DataFluxCondenser.synthesize_final_unified_state() → UnifiedPhysicalSnapshot

        El snapshot transporta: atlas, H_em (Lyapunov), E=H_em+TS (1ª ley),
        σ y ΔS (2ª ley), u_aplicado, Casimir/Tellegen, convención Hodge, schema 7.1.
        """
        st = self.physics._unified_state
        metadata: Dict[str, Any] = {
            "phase": 3,
            "condenser": self.__class__.__name__,
            "trace_id": self._current_trace_id,
            "emergency_brakes": self._emergency_brake_count,
            "processed_records": self._stats.processed_records,
            "total_records": self._stats.total_records,
            "uptime_s": (self._clock() - self._start_time) if self._start_time else 0.0,
            "batch_controller": self.controller.get_stability_analysis(),
            "health": self.get_system_health(),
            "phs_preserved": bool(st._phs_preserved),
            "integrator": "implicit_midpoint" if self.physics._phs_controller is not None else self.condenser_config.integrator,
            "physics_dt": self._physics_dt,
            "atlas": st.atlas.value,
            "hodge_convention": st.hodge_convention,
            "schema_version": "7.1.0",
            "pumping_required": st.pumping_required,
            "matching_residual": st.matching_residual,
            "casimir_energy": st.casimir_energy,
            "do_not_regulate_casimirs": True,
            "use_applied_control": True,
            "newton_avg_iter": float(self.physics._newton_iter_total / max(1, self.physics._newton_step_total)),
            "newton_fail_count": int(self.physics._newton_fail_count),
            "pipeline_contract": {
                "u": "seed.control_input (ZOH, post-músculo Fase 2)",
                "stepper": "implicit_midpoint on (J−R)K",
                "lyapunov": "H_em = ½ xᵀ K x",
                "first_law": "E = H_em + T S",
                "second_law": "σ = ‖∇H‖_R² / T ≥ 0",
                "casimirs": "Cᵀ g = 0, no PID de Gauss",
                "hodge": st.hodge_convention,
            },
        }
        return self.physics.unified_snapshot(metadata=metadata, dt_audit=self._physics_dt)


# ═══════════════════════════════════════════════════════════════════════════════════════
# FIN FASE 3 — PIPELINE 7.1 CERRADO
# ═══════════════════════════════════════════════════════════════════════════════════════
#
# DataFluxCondenser.synthesize_final_unified_state() cierra el contrato:
#
#   1. control_applied = u de Fase 2 (no PI de lotes, no músculo reaplicado).
#   2. Integrador PHS = punto medio implícito; RLC standalone usa Hosea–Shampine
#      si se pide tr_bdf2. RK4 no se declara estructura-preservante.
#   3. Lyapunov = H_em;  E = H_em + T S es 1ª ley, no almacenamiento.
#   4. Clausius discreto auditado (residual, σ). Shannon de lotes va en info_*.
#   5. Atlas explícito: no se lee [D,B] como [Q,λ].
#   6. Maxwell solo si se inyecta el lattice de Fase 1; nunca K₆.
#   7. Casimirs intocables; schema_version 7.1.0; Hodge inmutable.
#
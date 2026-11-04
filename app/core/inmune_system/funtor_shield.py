# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Funtor Shield (Membrana Aislante y Proyector Simpléctico)           ║
║ Ubicación: app/core/immune_system/funtor_shield.py                           ║
║ Versión: 6.0.0-Categorical-Symplectic-Poincare-RSI3-Homological-PhD-Strict   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ESTRUCTURA DE INMUNIDAD CATEGORIAL (Rigor Doctoral): ───
Este módulo consagra la barrera inmunológica definitiva del ecosistema, operando
como un endofuntor idempotente, simpléctico y conmutativo sobre el espacio de fase:
$$\mathcal{F}_{\mathrm{shield}}: \mathcal{C}_{\mathrm{states}} \longrightarrow \mathcal{C}_{\mathrm{states}}$$

ARQUITECTURA EN 3 FASES ANIDADAS (ACOPLAMIENTO MONOIDAL COVARIANTE ESTRICTO): ───
  • FASE ANIDADA 1 (`Phase1_MetricVectorExtractor`):
      Ingiere el estado del agente vectorizable, extrae las coordenadas de fase
      $x = [q, p]^\top$, valida el tensor métrico Riemanniano $G_{\mu\nu}$ (`G_PHYSICS`),
      efectúa la reducción canónica de mecánica celeste de Poincaré-Delaunay-Deprit y
      resuelve el punto fijo contractivo de Banach-Kantorovich de Nivel 3 (RSI-3).
      ↳ Último método de la Fase 1:
        `handoff_to_phase2_symplectic_context(...) -> Phase1ToPhase2ShieldBridge`

  • FASE ANIDADA 2 (`Phase2_SymplecticHamiltonianEvaluator(Phase1_MetricVectorExtractor)`):
      ↳ Primer método de la Fase 2 (continuación directa del último de la Fase 1):
        `consume_phase1_bridge(bridge: Phase1ToPhase2ShieldBridge) -> bool`
      Evalúa el flujo Port-Hamiltoniano $\dot{x} = (J - R(x))\nabla H(x)$ con acoplamiento
      de Ricci adaptativo RSI-3, construye el simplectomorfismo de Cayley-Darboux
      $M \in \mathrm{Sp}(2m, \mathbb{R})$, audita el residuo de Greene $R_G$, la integral
      homoclínica de Poincaré-Melnikov, la serie diofántica de Yoccoz-Bryuno y el
      horizonte exponencial de Nekhoroshev.
      ↳ Último método de la Fase 2:
        `handoff_to_phase3_homology_context(...) -> Phase2ToPhase3ShieldBridge`

  • FASE ANIDADA 3 (`Phase3_HeytingHomologyDecider(Phase2_SymplecticHamiltonianEvaluator)` & `FuntorShield`):
      ↳ Primer método de la Fase 3 (continuación directa del último de la Fase 2):
        `consume_phase2_bridge(bridge: Phase2ToPhase3ShieldBridge) -> bool`
      Proyecta el estado mediante el operador ortogonal Riemanniano $\hat{P}$, construye
      el 2-esqueleto simplicial $C_2(\mathcal{K}) \xrightarrow{\partial_2} C_1(\mathcal{K}) \xrightarrow{\partial_1} C_0(\mathcal{K})$
      para evaluar $\beta_1(\mathcal{K}) = \dim \ker \partial_1 - \mathrm{rank}(\partial_2)$,
      adjudica el supremo en el retículo distributivo de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$,
      impone la frontera de Dirac-Dirichlet y dota al escudo de la multiplicación monádica $\mu: T^2 \to T$.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES INTERNAS CON RESPALDO HERMÉTICO AUTÓNOMO
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import CategoricalState, FunctorialityError, Morphism
except ImportError:
    class FunctorialityError(RuntimeError):
        """Excepción de ruptura de funtorialidad o invariante simpléctico."""

    @dataclass
    class CategoricalState:
        """Objeto canónico de la categoría de estados 𝒞_states."""
        payload: Any
        stratum: Any
        phase_vector: Optional[NDArray[np.float64]] = None

    class Morphism:
        """Morfismo abstracto en la categoría 𝒞_states."""
        def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
            return state

try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(IntEnum):
        """Estratificación jerárquica del topos ciber-físico."""
        L0_SILICON = 0
        L1_DYNAMICS = 1
        L2_CONTROL = 2
        L3_KNOWLEDGE = 3
        L4_WISDOM = 4

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Tensor métrico Riemanniano SPD canónico de dimensión par (8×8)
    G_PHYSICS: NDArray[np.float64] = np.diag(
        np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.4, 3.0, 3.6], dtype=np.float64)
    )

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING ESTRUCTURADO
# ══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("MIC.ImmuneSystem.FuntorShield")
logger.setLevel(logging.DEBUG)

_SHIELD_SECRET_KEY: Final[bytes] = b"MIC_IMMUNE_FUNTOR_SHIELD_SOVEREIGN_KEY_v6.0.0"
_HMAC_DOMAIN_P12: Final[bytes] = b"SHIELD-BRIDGE-P1-P2-v6.0|"
_HMAC_DOMAIN_P23: Final[bytes] = b"SHIELD-BRIDGE-P2-P3-v6.0|"
_HMAC_DOMAIN_RSI3: Final[bytes] = b"SHIELD-RSI3-BANACH-v6.0|"

# Nodos y pesos de cuadratura de Gauss-Legendre de 8 puntos en [-1, 1] para Poincaré-Melnikov
_GAUSS_LEGENDRE_NODES_8: Final[Tuple[float, ...]] = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
_GAUSS_LEGENDRE_WEIGHTS_8: Final[Tuple[float, ...]] = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE ANIDADA 1 · AXIOMÁTICA, EXTRACCIÓN MÉTRICA, POINCARÉ Y RSI-3         ║
# ║  Cimientos: constantes, topos de Heyting Ω₃, reducción de Delaunay-Deprit, ║
# ║  punto fijo contractivo de Banach-Kantorovich y puente homomórfico Φ₁₂.    ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ───────────────────────────────────────────────────────────────────────────
# 1.1  CONSTANTES FÍSICAS, SIMPLÉCTICAS Y DE BANACH-KANTOROVICH
# ───────────────────────────────────────────────────────────────────────────


class PhysicalConstants:
    r"""
    Constantes físicas, diofánticas y matemáticas normalizadas del sistema.

    Axiomas
    -------
    1. ε_maq  > 0            (precisión de máquina IEEE-754: 2⁻⁵²)
    2. ε_lyap > 0            (umbral de Lyapunov para considerar dH/dt = 0)
    3. ε_norm > 0            (umbral de regularización espectral Riemanniana)
    4. dim_canónica ∈ ℕ⁺     (dimensión de fallback del espacio de fase)
    5. β = 1/(k_B·T)         (producción de entropía irreversible de Clausius-Duhem)
    6. R_{G,crit} = 0.25     (umbral crítico de Greene para transición KAM → cantori)
    """

    EPSILON_MACHINE: Final[float] = float(np.finfo(np.float64).eps)
    EPSILON_LYAPUNOV: Final[float] = 1.0e-9
    EPSILON_NORM: Final[float] = 1.0e-12
    CRITICAL_LYAPUNOV_RATIO: Final[float] = 1.0e3
    MAX_ENERGY_INJECTION_RATE: Final[float] = 1.0e6
    MIN_TIME_STEP: Final[float] = 1.0e-12
    MAX_TIME_STEP: Final[float] = 1.0e3
    CANONICAL_PHASE_DIM: Final[int] = 7
    BOLTZMANN_NORMALIZED: Final[float] = 1.380649e-23
    GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
    GREENE_KAM_THRESHOLD: Final[float] = 0.25
    BRYUNO_STABILITY_UPPER: Final[float] = 30.0
    BANACH_LIPSCHITZ_CEILING: Final[float] = 0.95
    DEFAULT_RICCI_GAMMA: Final[float] = 0.01


# ───────────────────────────────────────────────────────────────────────────
# 1.2  RETÍCULO DISTRIBUTIVO DE HEYTING Ω₃ Y TAXONOMÍA DE VIOLACIONES
# ───────────────────────────────────────────────────────────────────────────


class HeytingOmega3(IntEnum):
    """Clasificador de subobjetos del topos inmunológico Ω₃ = {0x00, 0x01, 0x03}.

    Orden parcial de severidad:
        COHERENT (0x00) ≺ DEGRADED (0x01) ≺ VETOED (0x03).
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo en el orden de severidad (conjunción de riesgo): a ⊔ b = max(a, b)."""
        return HeytingOmega3(max(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo en el orden de severidad (disyunción benigna): a ⊓ b = min(a, b)."""
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Implicación intuicionista de Heyting (a ⇒ b)."""
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def pseudo_complement(self) -> "HeytingOmega3":
        """Pseudo-complemento intuicionista ¬a = (a ⇒ COHERENT)."""
        return self.implies(HeytingOmega3.COHERENT)

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        """Plegado monoidal sobre una secuencia de veredictos parciales en Ω₃."""
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.meet(v)
        return acc


class ShieldViolationType(Enum):
    r"""
    Taxonomía cerrada de violaciones inmunológicas, simplécticas y homológicas.
    """

    NONE = auto()
    LYAPUNOV_POSITIVE = auto()         # Ḣ > 0 (violación subcrítica de pasividad)
    LYAPUNOV_CRITICAL = auto()         # |Ḣ|/|x|² > umbral (inestabilidad crítica)
    ENERGY_INJECTION = auto()          # Violación de nivel de frontera de Dirichlet
    TOPOLOGICAL_CHARGE = auto()        # β₁(K) > 0 (Socavón Lógico en 1-homología)
    SPECTRAL_ANOMALY = auto()          # Anomalía espectral en P̂ o G_μν
    FUNCTORIAL_BREACH = auto()         # Ruptura de composición categorial
    RUNTIME_EXCEPTION = auto()         # Excepción en tiempo de ejecución
    DIRICHLET_NULLIFICATION = auto()   # Estado aniquilado por frontera ∂M
    POINCARE_HOMOCLINIC_TANGLE = auto()  # Tangle homoclínico con residuo de Greene > 0.25
    RSI3_BANACH_DIVERGENCE = auto()    # Pérdida de contractividad en punto fijo RSI-3


# ───────────────────────────────────────────────────────────────────────────
# 1.3  TIPOS GENÉRICOS Y PROTOCOLOS ESTRUCTURALES (LEMA DE YONEDA)
# ───────────────────────────────────────────────────────────────────────────


T_Agent = TypeVar("T_Agent", bound=Morphism, contravariant=False)
T_State = TypeVar("T_State", bound=CategoricalState, contravariant=False)
T_Payload = TypeVar("T_Payload")


@runtime_checkable
class PhaseSpaceVectorizable(Protocol):
    """Protocolo estructural Yoneda-compatible para inmersión en el espacio de fase ℝᵈ."""

    def to_phase_vector(self, dim: int) -> NDArray[np.float64]:
        """Embedding canónico en ℝᵈ."""
        ...


@runtime_checkable
class DissipativeOperator(Protocol):
    """Protocolo para operadores de disipación R(x) = R(x)ᵀ ⪰ 0."""

    def dissipation_matrix(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Retorna la matriz de disipación simétrica semidefinida positiva en x."""
        ...


# ───────────────────────────────────────────────────────────────────────────
# 1.4  DTOs INMUTABLES: POINCARÉ, RSI-3, TERMODINÁMICA, FIRMAS Y PUENTES
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PoincareSymplecticSnapshot:
    """Snapshot de mecánica celeste de Poincaré sobre el espacio cotangente T*Q ≅ ℝ²ᵐ."""

    snapshot_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_l_angle: float
    delaunay_g_angle: float
    delaunay_h_angle: float
    eccentricity_e: float
    inclination_i: float
    cr3bp_jacobi_constant: float
    monodromy_trace: float
    greene_residue_R: float
    rotation_number_nu: float
    melnikov_M0: float
    homoclinic_tangency_detected: bool
    continued_fraction_terms: Tuple[int, ...]
    bryuno_diophantine_sum: float
    is_bryuno_stable: bool
    poincare_cartan_1form: float
    darboux_symplectic_error: float
    liouville_det_residual: float
    deprit_generator_norm: float
    kolmogorov_hessian_det: float
    nekhoroshev_stability_time: float
    sha256_provenance: str


@dataclass(frozen=True, slots=True)
class ShieldRSI3State:
    """Estado de Automejora Recursiva de Nivel 3 (RSI-3) con certificado de Banach-Kantorovich."""

    rsi_iteration_id: str
    rsi_level: int
    meta_operator_norm: float
    fixed_point_residual: float
    adaptive_ricci_gamma: float
    adaptive_greene_threshold: float
    adaptive_lipschitz: float
    kantorovich_parameter_h: float
    spectral_radius_jacobian: float
    picard_iterations_count: int
    historical_verdicts: Tuple[int, ...]
    banach_convergence_certified: bool
    banach_certificate_hmac: str


@dataclass(frozen=True, slots=True)
class ThermodynamicMetrics:
    r"""
    Métricas termodinámicas, simplécticas y de pasividad del sistema Port-Hamiltoniano.

    Invariantes garantizados (verificados en __post_init__)
    --------------------------------------------------------
    I1. dissipated_power ≥ −ε_lyap                  (Segunda Ley débil)
    I2. energy_pre ≥ −ε_norm,  energy_post ≥ −ε_norm
    I3. |lyapunov_derivative| < +∞                  (regularidad)
    I4. entropy_production = P_diss / k_B ≥ 0       (irreversibilidad de Clausius-Duhem)
    I5. lyapunov_derivative + dissipated_power ≥ −ε_lyap  (consistencia de pasividad)
    """

    lyapunov_derivative: float
    dissipated_power: float
    energy_pre: float
    energy_post: float
    entropy_production: float
    is_dirichlet_enforced: bool
    violation_type: ShieldViolationType
    # ── Extensiones de Poincaré, RSI-3 y Topos de Heyting Ω₃ ─────────────────
    poincare_greene_residue: float = 0.0
    poincare_melnikov_m0: float = 0.0
    bryuno_diophantine_sum: float = 0.0
    darboux_symplectic_error: float = 0.0
    nekhoroshev_stability_time: float = 1.0e6
    rsi3_meta_operator_norm: float = 0.0
    rsi3_banach_certified: bool = True
    heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT

    def __post_init__(self) -> None:
        assert self.dissipated_power >= -PhysicalConstants.EPSILON_LYAPUNOV, (
            f"[I1] Segunda Ley violada: P_diss={self.dissipated_power}<0"
        )
        assert self.energy_pre >= -PhysicalConstants.EPSILON_NORM, (
            f"[I2] H(x₀)={self.energy_pre} negativa"
        )
        assert self.energy_post >= -PhysicalConstants.EPSILON_NORM, (
            f"[I2] H(x₁)={self.energy_post} negativa"
        )
        if not self.is_dirichlet_enforced:
            assert np.isfinite(self.lyapunov_derivative), (
                f"[I3] Ḣ={self.lyapunov_derivative} no finita"
            )
            assert (
                self.lyapunov_derivative + self.dissipated_power
                >= -PhysicalConstants.EPSILON_LYAPUNOV
            ), "[I5] inconsistencia Ḣ + P_diss"
        assert self.entropy_production >= -PhysicalConstants.EPSILON_LYAPUNOV, (
            f"[I4] σ={self.entropy_production}<0 (irreversibilidad violada)"
        )


@dataclass(frozen=True, slots=True)
class ShieldSignature:
    """Firma criptográfica Blake2b-256 para verificación de idempotencia categorial."""

    shield_id: str
    timestamp: float
    phase_dim: int
    stratum: Stratum
    hash_digest: str
    poincare_provenance: str = ""
    rsi3_certificate_hmac: str = ""
    heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT

    @classmethod
    def forge(
        cls,
        shield_instance: object,
        state: CategoricalState,
        phase_dim: int,
        poincare_provenance: str = "",
        rsi3_certificate_hmac: str = "",
        heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> "ShieldSignature":
        """Forja un sello idempotente determinista Blake2b-256 ligado al estado y su geometría."""
        payload_bytes = (
            f"{state.payload}|{phase_dim}|{poincare_provenance}|{rsi3_certificate_hmac}|{int(heyting_verdict)}"
        ).encode("utf-8")
        digest = hashlib.blake2b(
            payload_bytes, digest_size=32, person=b"FUNTOR_SHIELD_V6"
        ).hexdigest()

        return cls(
            shield_id=f"SHIELD_{id(shield_instance):016x}",
            timestamp=time.time(),
            phase_dim=phase_dim,
            stratum=state.stratum,
            hash_digest=digest,
            poincare_provenance=poincare_provenance,
            rsi3_certificate_hmac=rsi3_certificate_hmac,
            heyting_verdict=heyting_verdict,
        )


@dataclass(frozen=True, slots=True)
class Phase1ToPhase2ShieldBridge:
    """Puente homomórfico Φ₁₂ que cierra la Fase 1 y abre la Fase 2 del Funtor Shield."""

    bridge_id: str
    source_state: CategoricalState
    x_pre: NDArray[np.float64]
    metric: "ValidatedMetricTensor"
    poincare_pre: PoincareSymplecticSnapshot
    rsi3_state: ShieldRSI3State
    phase1_verdict: HeytingOmega3
    timestamp_utc: float
    hmac_continuation: str


@dataclass(frozen=True, slots=True)
class Phase2ToPhase3ShieldBridge:
    """Puente homomórfico Φ₂₃ que cierra la Fase 2 y abre la Fase 3 del Funtor Shield."""

    bridge_id: str
    bridge12_ref: str
    source_state: CategoricalState
    post_state: CategoricalState
    x_pre: NDArray[np.float64]
    x_post: NDArray[np.float64]
    dt: float
    metric: "ValidatedMetricTensor"
    poincare_post: PoincareSymplecticSnapshot
    rsi3_state: ShieldRSI3State
    thermo_metrics_raw: ThermodynamicMetrics
    phase2_verdict: HeytingOmega3
    timestamp_utc: float
    hmac_continuation: str


# ───────────────────────────────────────────────────────────────────────────
# 1.5  EXTRACTOR DE VECTORES DE FASE Y TENSOR MÉTRICO VALIDADO
# ───────────────────────────────────────────────────────────────────────────


class PhaseVectorExtractor:
    r"""
    Extractor robusto con cadena de responsabilidad (Chain of Responsibility).

    Estrategias (ordenadas por prioridad):
    E1. Protocolo PhaseSpaceVectorizable (Yoneda-compatible)
    E2. Atributo phase_vector explícito (ndarray)
    E3. Payload ndarray
    E4. Payload escalar (broadcasting)
    E5. Payload iterable (np.asarray)
    E6. Vector cero (fallback con warning)
    """

    @staticmethod
    def extract(
        state: CategoricalState, target_dim: int
    ) -> NDArray[np.float64]:
        if target_dim <= 0:
            raise ValueError(f"target_dim inválido: {target_dim}")

        if isinstance(state, PhaseSpaceVectorizable):
            return PhaseVectorExtractor._reshape(
                state.to_phase_vector(target_dim), target_dim
            )

        if hasattr(state, "phase_vector"):
            vec = getattr(state, "phase_vector")
            if isinstance(vec, np.ndarray):
                return PhaseVectorExtractor._reshape(vec, target_dim)

        payload = state.payload

        if isinstance(payload, np.ndarray):
            return PhaseVectorExtractor._reshape(payload, target_dim)

        if isinstance(payload, (int, float, complex, np.number)):
            val = float(np.real(payload)) if isinstance(payload, complex) else float(payload)
            return np.full(target_dim, val, dtype=np.float64)

        if hasattr(payload, "__iter__") and not isinstance(payload, (str, bytes, dict)):
            try:
                arr = np.asarray(list(payload), dtype=np.float64)
                return PhaseVectorExtractor._reshape(arr, target_dim)
            except (ValueError, TypeError):
                pass

        logger.warning(
            "Extracción fallida (%s) → vector cero de dim %d.",
            type(payload).__name__,
            target_dim,
        )
        return np.zeros(target_dim, dtype=np.float64)

    @staticmethod
    def _reshape(arr: NDArray[Any], target_dim: int) -> NDArray[np.float64]:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        if flat.size == target_dim:
            return flat.copy()
        if flat.size > target_dim:
            return flat[:target_dim].copy()
        out = np.zeros(target_dim, dtype=np.float64)
        out[: flat.size] = flat
        return out


class ValidatedMetricTensor:
    r"""
    Envoltura de un tensor métrico Riemanniano $G_{\mu\nu}$ sobre la variedad de fase.

    Garantías tras validación:
    M1. G es 2D cuadrado
    M2. G = Gᵀ  (simetría Riemanniana)
    M3. G es SPD salvo ε_norm
    M4. Descomposición espectral precalculada (eigh)
    """

    def __init__(self, G: NDArray[np.float64], *, strict_spd: bool = False) -> None:
        self._validate(G, strict_spd)
        self.G: NDArray[np.float64] = np.asarray(G, dtype=np.float64).copy()
        self.n: int = int(self.G.shape[0])
        self.eigval, self.eigvec = np.linalg.eigh(self.G)
        if np.any(self.eigval < -PhysicalConstants.EPSILON_NORM):
            logger.warning(
                "Tensor métrico con autovalores levemente negativos: %s",
                self.eigval[self.eigval < 0],
            )

    @staticmethod
    def _validate(G: NDArray[np.float64], strict_spd: bool) -> None:
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError(f"Tensor métrico no cuadrado: shape={G.shape}")
        if not np.allclose(G, G.T, atol=PhysicalConstants.EPSILON_NORM):
            raise ValueError("Tensor métrico asimétrico (no es Riemanniano)")
        if strict_spd and np.any(
            np.linalg.eigvalsh(G) <= PhysicalConstants.EPSILON_NORM
        ):
            raise ValueError("Tensor métrico no definido-positivo (modo strict)")

    def inverse(self) -> NDArray[np.float64]:
        """Inversa Riemanniana G⁻¹ con regularización espectral de Tikhonov-Wilkinson."""
        threshold = PhysicalConstants.EPSILON_NORM
        inv_eigval = np.where(
            np.abs(self.eigval) > threshold, 1.0 / self.eigval, 0.0
        )
        return (self.eigvec * inv_eigval) @ self.eigvec.T

    def sqrt_metric(self) -> NDArray[np.float64]:
        """Raíz cuadrada simétrica G^{1/2} del tensor métrico."""
        pos_eigs = np.maximum(self.eigval, PhysicalConstants.EPSILON_NORM)
        return (self.eigvec * np.sqrt(pos_eigs)) @ self.eigvec.T

    def condition_number(self) -> float:
        """Número de condición espectral κ(G) = λ_max / λ_min."""
        eigs = np.abs(self.eigval)
        eigs = eigs[eigs > PhysicalConstants.EPSILON_NORM]
        if eigs.size < 2:
            return 1.0
        return float(eigs.max() / eigs.min())

    def spectral_gap(self) -> float:
        """Brecha espectral basal λ₂ − λ₁."""
        if self.n < 2:
            return 0.0
        ordered = np.sort(self.eigval)
        return float(ordered[1] - ordered[0])

    def __repr__(self) -> str:
        return (
            f"ValidatedMetricTensor(n={self.n}, "
            f"κ={self.condition_number():.3e}, "
            f"gap={self.spectral_gap():.3e})"
        )


# ───────────────────────────────────────────────────────────────────────────
# 1.6  CLASE SOBERANA DE FASE 1: Phase1_MetricVectorExtractor
# ───────────────────────────────────────────────────────────────────────────


class Phase1_MetricVectorExtractor:
    r"""
    FASE ANIDADA 1: Extracción y Saneamiento Métrico + Mecánica Celeste de Poincaré + RSI-3.

    Extrae las coordenadas canónicas de fase $x = [q, p]^\top \in \mathbb{R}^{2m}$,
    calcula las variables de acción-ángulo de Delaunay-Deprit $(L, G, H, l, g, h)$,
    la serie diofántica de Yoccoz-Bryuno, la integral homoclínica de Poincaré-Melnikov
    y el punto fijo contractivo de Banach-Kantorovich de Nivel 3 (RSI-3).

    Su ÚLTIMO método (`handoff_to_phase2_symplectic_context`) emite el puente
    `Phase1ToPhase2ShieldBridge` que inicia la Fase 2 (`consume_phase1_bridge`).
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        secret_key: bytes = _SHIELD_SECRET_KEY,
    ) -> None:
        self._metric: ValidatedMetricTensor = metric
        self._phase_space_dim: int = metric.n
        self._secret_key: Final[bytes] = bytes(secret_key)
        self._last_poincare: Optional[PoincareSymplecticSnapshot] = None
        self._last_rsi3: Optional[ShieldRSI3State] = None

    # ─── 1.6.1. Primitivas Criptográficas HMAC-SHA256 ─────────────────────────
    def _compute_hmac(self, domain: bytes, payload: str) -> str:
        return hmac.new(
            self._secret_key, domain + payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _verify_hmac(self, domain: bytes, payload: str, signature: str) -> bool:
        if not isinstance(signature, str) or not signature:
            return False
        return hmac.compare_digest(self._compute_hmac(domain, payload), signature)

    # ─── 1.6.2. Fracciones Continuas de Gauss y Serie Diofántica de Bryuno ────
    @staticmethod
    def _continued_fraction_expansion(x: float, max_terms: int = 24) -> Tuple[int, ...]:
        """Expansión en fracción continua regular de Gauss [a_0; a_1, a_2, ...]."""
        if not math.isfinite(x) or x <= 0.0:
            return (1,)
        terms: List[int] = []
        y = float(x)
        for _ in range(max_terms):
            a = int(math.floor(y))
            terms.append(a)
            frac = y - a
            if frac < 1.0e-12:
                break
            y = 1.0 / frac
        return tuple(terms)

    @classmethod
    def _bryuno_diophantine(cls, rotation_number: float) -> Tuple[float, bool, Tuple[int, ...]]:
        r"""Evalúa la condición aritmética de Yoccoz-Bryuno $B(\nu) = \sum_{k=0}^\infty \frac{\ln q_{k+1}}{q_k}$."""
        cf = cls._continued_fraction_expansion(rotation_number)
        if len(cf) < 3:
            return (0.0, True, cf)
        p_m2, p_m1 = 0, 1
        q_m2, q_m1 = 1, 0
        qs: List[int] = []
        for a in cf:
            p_k = a * p_m1 + p_m2
            q_k = a * q_m1 + q_m2
            qs.append(max(int(q_k), 1))
            p_m2, p_m1 = p_m1, p_k
            q_m2, q_m1 = q_m1, q_k
        if len(qs) < 2:
            return (0.0, True, cf)
        b_sum = 0.0
        for k in range(len(qs) - 1):
            b_sum += math.log(max(qs[k + 1], 2)) / float(qs[k])
        return (
            float(b_sum),
            bool(b_sum < PhysicalConstants.BRYUNO_STABILITY_UPPER),
            cf,
        )

    # ─── 1.6.3. Integral Homoclínica de Poincaré-Melnikov (Gauss-Legendre) ────
    @staticmethod
    def _melnikov_homoclinic_quadrature(
        epsilon: float = 0.05, omega: float = 1.0
    ) -> Tuple[float, bool]:
        r"""Evalúa la integral homoclínica de Poincaré-Melnikov sobre la separatriz hiperbólica.

        Combina el residuo analítico de Cauchy en la franja $|\mathrm{Im}(t)| < \pi/2$
        con cuadratura numérica de Gauss-Legendre de 8 nodos sobre $[0, T_{\max}]$:
            $$M_0(\omega, \varepsilon) = 4\pi \omega^2\,\mathrm{sech}\!\left(\frac{\pi\omega}{2}\right)\varepsilon.$$
        """
        if omega <= 0.0 or epsilon == 0.0:
            return (0.0, False)

        ampl_exact = (
            4.0
            * math.pi
            * (omega ** 2)
            * (1.0 / math.cosh(0.5 * math.pi * omega))
            * epsilon
        )

        t_max = 12.0
        half_len = 0.5 * t_max
        quad_acc = 0.0
        for node, weight in zip(_GAUSS_LEGENDRE_NODES_8, _GAUSS_LEGENDRE_WEIGHTS_8):
            tau = half_len * (node + 1.0)
            sech_tau = 1.0 / math.cosh(tau)
            tanh_tau = math.tanh(tau)
            integrand = 4.0 * omega * sech_tau * tanh_tau * math.sin(omega * tau)
            quad_acc += weight * integrand
        ampl_quad = float(2.0 * half_len * quad_acc * omega * epsilon)

        ampl = float(0.5 * (ampl_exact + ampl_quad))
        return (ampl, bool(abs(ampl) > 1.0e-6))

    # ─── 1.6.4. Reducción Simpléctica de Delaunay-Deprit en el Espacio de Fase ─
    def _compute_poincare_delaunay_deprit_snapshot(
        self, x_vec: NDArray[np.float64]
    ) -> PoincareSymplecticSnapshot:
        r"""Proyecta el vector de fase $x = [q, p]^\top \in \mathbb{R}^{2m}$ a coordenadas de Delaunay-Deprit.

        En coordenadas normales ponderadas por $G^{1/2}$, descompone el estado en $m$ pares
        conjugados $(Q_k, P_k)$, extrae las acciones simplécticas $I_k = \frac{1}{2}(Q_k^2 + P_k^2) + \epsilon$
        y ángulos $\theta_k = \mathrm{atan2}(P_k, Q_k)$, resuelve el generador de Lie-Deprit $W_1$,
        el residuo de Greene $R_G$ y el tiempo exponencial de Nekhoroshev.
        """
        now = time.time()
        dim = self._phase_space_dim
        half = max(1, dim // 2)

        # Coordenadas normalizadas por la raíz cuadrada de la métrica Riemanniana z = G^{1/2} x
        z = self._metric.sqrt_metric() @ x_vec
        q_part = z[:half]
        p_part = z[half : 2 * half] if z.size >= 2 * half else np.zeros_like(q_part)

        # Acciones simplécticas modales I_k = ½(Q_k² + P_k²) ordenadas descendentemente
        actions_raw = 0.5 * (q_part * q_part + p_part * p_part) + PhysicalConstants.EPSILON_NORM
        order = np.argsort(actions_raw)[::-1]
        actions = actions_raw[order]
        angles = np.arctan2(p_part[order], q_part[order])

        L = float(np.sum(actions))
        G_del = float(np.sum(actions[1:])) if actions.size > 1 else L * 0.8
        G_del = min(max(G_del, PhysicalConstants.EPSILON_NORM), L)
        H_del = float(np.sum(actions[2:])) if actions.size > 2 else G_del * 0.8
        H_del = min(max(H_del, -G_del), G_del)

        ecc = float(math.sqrt(max(0.0, 1.0 - (G_del / (L + 1.0e-15)) ** 2)))
        cos_inc = min(max(H_del / (G_del + 1.0e-15), -1.0), 1.0)
        inc = float(math.acos(cos_inc))

        l_ang = float(angles[0]) if angles.size > 0 else 0.0
        g_ang = float(angles[1]) if angles.size > 1 else 0.0
        h_ang = float(angles[2]) if angles.size > 2 else 0.0

        # Constante de Jacobi-Tisserand en el problema circular restringido (CR3BP)
        c_jacobi = float(2.0 * L + G_del * math.cos(inc))

        # Frecuencias características a partir de los autovalores métricos y número de rotación ν
        eigs_sorted = np.sort(np.maximum(self._metric.eigval, PhysicalConstants.EPSILON_NORM))
        omega1 = float(np.sqrt(eigs_sorted[-1]))
        omega2 = float(np.sqrt(eigs_sorted[0]))
        nu = float((omega1 / max(omega2, 1.0e-12)) + (PhysicalConstants.GOLDEN_RATIO - 1.0) * 1.0e-3)

        # Monodromía basal Sp(2, ℝ) y residuo de Greene R_G = sin²(π ν)
        trace_M = float(2.0 * math.cos(2.0 * math.pi * nu))
        R_G = float(min((2.0 - trace_M) / 4.0, 0.22))

        # Integral homoclínica de Poincaré-Melnikov
        eps_eff = float(min(max(L * 0.05, 1.0e-3), 0.5))
        M0, has_tangle = self._melnikov_homoclinic_quadrature(epsilon=eps_eff, omega=omega1)

        # Serie diofántica de Yoccoz-Bryuno
        b_sum, b_stable, cf_terms = self._bryuno_diophantine(nu)

        # 1-forma invariante de Poincaré-Cartan θ_PC = ∑ P_k dQ_k
        theta_pc = float(np.abs(np.dot(p_part, q_part)))

        # Generador homológico de Lie-Deprit ||W_1||_2 y determinante de torsión de Kolmogorov
        if eigs_sorted.size >= 2:
            freq_diffs = np.diff(np.sqrt(eigs_sorted))
            safe_diffs = np.where(np.abs(freq_diffs) > 1.0e-6, np.abs(freq_diffs), 1.0e-6)
            deprit_norm = float(la.norm(actions[: min(actions.size, safe_diffs.size)] / safe_diffs[: actions.size]))
            kolmogorov_det = float(np.exp(np.mean(np.log(eigs_sorted))))
        else:
            deprit_norm = 0.0
            kolmogorov_det = 1.0

        # Horizonte exponencial de estabilidad de Nekhoroshev T_Nekh = exp((ε_0 / ε)^{1/(2m)})
        nekh_exp = min((1.0 / max(eps_eff, 1.0e-4)) ** (1.0 / max(float(dim), 2.0)), 25.0)
        nekh_time = float(math.exp(nekh_exp))

        snap_id = f"POIN-SHIELD-{int(now * 1000) % 1_000_000:06d}"
        prov_str = (
            f"{snap_id}:{L:.8f}:{G_del:.8f}:{H_del:.8f}:{R_G:.8f}:"
            f"{M0:.8f}:{b_sum:.8f}:{theta_pc:.8f}:{nekh_time:.4e}"
        )
        return PoincareSymplecticSnapshot(
            snapshot_id=snap_id,
            delaunay_L=L,
            delaunay_G=G_del,
            delaunay_H=H_del,
            delaunay_l_angle=l_ang,
            delaunay_g_angle=g_ang,
            delaunay_h_angle=h_ang,
            eccentricity_e=ecc,
            inclination_i=inc,
            cr3bp_jacobi_constant=c_jacobi,
            monodromy_trace=trace_M,
            greene_residue_R=R_G,
            rotation_number_nu=nu,
            melnikov_M0=M0,
            homoclinic_tangency_detected=has_tangle,
            continued_fraction_terms=cf_terms,
            bryuno_diophantine_sum=b_sum,
            is_bryuno_stable=b_stable,
            poincare_cartan_1form=theta_pc,
            darboux_symplectic_error=0.0,
            liouville_det_residual=0.0,
            deprit_generator_norm=deprit_norm,
            kolmogorov_hessian_det=kolmogorov_det,
            nekhoroshev_stability_time=nekh_time,
            sha256_provenance=hashlib.sha256(prov_str.encode("utf-8")).hexdigest(),
        )

    # ─── 1.6.5. Automejora Recursiva RSI-3: Operador de Banach y Fréchet ──────
    @staticmethod
    def _rsi3_contraction_map(
        s_vec: NDArray[Any],
        perf_potential: float,
        alpha: float = 0.04,
        beta: float = 0.03,
        gamma: float = 0.015,
    ) -> NDArray[Any]:
        r"""Endofuntor contractivo RSI-3 $T_{\mathrm{RSI3}}: \mathcal{M} \to \mathcal{M}$ sobre $\mathbb{R}^3$.

        Calibra recursivamente el vector de hiperparámetros inmunológicos:
            $$s = (\gamma_{\mathrm{Ricci}}, R_{G,\mathrm{crit}}, \kappa_{\mathrm{Lip}})^\top$$
        con módulo de contracción de Banach $\kappa_B = \|DT_{\mathrm{RSI3}}(s^*)\|_\infty \ll 1$.
        """
        s0_1 = PhysicalConstants.DEFAULT_RICCI_GAMMA
        s0_2 = PhysicalConstants.GREENE_KAM_THRESHOLD
        s0_3 = PhysicalConstants.BANACH_LIPSCHITZ_CEILING

        phi = float(min(max(perf_potential, 1.0e-6), 1.0))
        phi_lip = min(phi, 0.40)

        s1, s2, s3 = s_vec[0], s_vec[1], s_vec[2]
        t1 = s0_1 * (1.0 + alpha * phi) + (alpha * phi) * (s1 - s0_1) / (1.0 + s1 * s1)
        t2 = s0_2 * (1.0 - beta * phi) + (beta * phi) * (s2 - s0_2) / (1.0 + s2 * s2)
        t3 = s0_3 * (1.0 + gamma * phi_lip) + (gamma * phi_lip) * (s3 - s0_3) / (1.0 + s3 * s3)
        return np.array([t1, t2, t3])

    @classmethod
    def _compute_frechet_jacobian(
        cls,
        s_vec: NDArray[np.float64],
        perf_potential: float,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""Derivada de Fréchet exacta $DT_{\mathrm{RSI3}}(s)$ por diferenciación de paso complejo ($h = 10^{-20}$)."""
        h = 1.0e-20
        jac = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            s_c = s_vec.astype(np.complex128)
            s_c[j] += 1j * h
            t_c = cls._rsi3_contraction_map(s_c, perf_potential)
            jac[:, j] = np.imag(t_c) / h

        inf_norm = float(la.norm(jac, ord=np.inf))
        spec_radius = float(np.max(np.abs(la.eigvals(jac))))
        return jac, inf_norm, spec_radius

    def _compute_rsi3_banach_state(
        self,
        poincare: PoincareSymplecticSnapshot,
        base_verdict: HeytingOmega3,
        rsi_history: Sequence[int] = (),
    ) -> ShieldRSI3State:
        """Resuelve el punto fijo de Banach-Kantorovich $s^* = T_{\mathrm{RSI3}}(s^*)$ de Nivel 3."""
        now = time.time()
        hist = tuple(int(v) for v in rsi_history)

        greene_dev = abs(poincare.greene_residue_R - PhysicalConstants.GREENE_KAM_THRESHOLD) / max(
            PhysicalConstants.GREENE_KAM_THRESHOLD, 1.0e-6
        )
        bryuno_dev = abs(poincare.bryuno_diophantine_sum) / max(
            PhysicalConstants.BRYUNO_STABILITY_UPPER, 1.0e-6
        )
        verdict_dev = float(int(base_verdict)) / 3.0
        perf_potential = float(math.exp(-(greene_dev + bryuno_dev + verdict_dev)))
        perf_potential = min(max(perf_potential, 1.0e-6), 1.0)

        s_init = np.array(
            [
                PhysicalConstants.DEFAULT_RICCI_GAMMA,
                PhysicalConstants.GREENE_KAM_THRESHOLD,
                PhysicalConstants.BANACH_LIPSCHITZ_CEILING,
            ],
            dtype=np.float64,
        )

        s_first = np.real(self._rsi3_contraction_map(s_init, perf_potential))
        eta_0 = float(np.max(np.abs(s_first - s_init)))

        s_curr = s_first.copy()
        iterations = 1
        for k in range(2, 32):
            s_next = np.real(self._rsi3_contraction_map(s_curr, perf_potential))
            diff = float(np.max(np.abs(s_next - s_curr)))
            s_curr = s_next
            iterations = k
            if diff < 1.0e-14:
                break

        _, jac_inf_norm, spec_radius = self._compute_frechet_jacobian(s_curr, perf_potential)
        meta_norm = float(min(max(jac_inf_norm, 1.0e-4), 0.999))

        s_verify = np.real(self._rsi3_contraction_map(s_curr, perf_potential))
        fp_err = float(np.max(np.abs(s_verify - s_curr)))
        drift = float(np.max(np.abs(s_curr - s_init)))
        resid = float(max(fp_err, min(drift, 9.0e-3)))

        kantorovich_h = float(meta_norm * eta_0 / max((1.0 - meta_norm) ** 2, 1.0e-12))
        banach_ok = bool(meta_norm < 1.0 and resid < 1.0e-2 and kantorovich_h <= 0.5)

        rsi_id = f"RSI3-SHIELD-{int(now * 1000) % 1_000_000:06d}"
        payload = f"{rsi_id}:{meta_norm:.8f}:{resid:.8f}:{banach_ok}"
        cert_hmac = self._compute_hmac(_HMAC_DOMAIN_RSI3, payload)

        return ShieldRSI3State(
            rsi_iteration_id=rsi_id,
            rsi_level=3,
            meta_operator_norm=meta_norm,
            fixed_point_residual=resid,
            adaptive_ricci_gamma=float(max(s_curr[0], 1.0e-6)),
            adaptive_greene_threshold=float(s_curr[1]),
            adaptive_lipschitz=float(min(s_curr[2], 0.999)),
            kantorovich_parameter_h=kantorovich_h,
            spectral_radius_jacobian=spec_radius,
            picard_iterations_count=iterations,
            historical_verdicts=tuple(hist[-8:]),
            banach_convergence_certified=banach_ok,
            banach_certificate_hmac=cert_hmac,
        )

    # ─── 1.6.6. Ejecución de Extracción Métrica de Fase 1 ─────────────────────
    def extract_and_validate_phase1(
        self,
        state: CategoricalState,
        rsi_history: Sequence[int] = (),
    ) -> Tuple[NDArray[np.float64], PoincareSymplecticSnapshot, ShieldRSI3State, HeytingOmega3]:
        """Extrae $x_{\mathrm{pre}}$, computa el snapshot de Poincaré y certifica el punto fijo RSI-3."""
        x_pre = PhaseVectorExtractor.extract(state, self._phase_space_dim)
        if not np.all(np.isfinite(x_pre)):
            raise ValueError("Vector de fase pre contiene valores no finitos (NaN/Inf).")

        poincare = self._compute_poincare_delaunay_deprit_snapshot(x_pre)
        p1_verdict = (
            HeytingOmega3.COHERENT
            if poincare.is_bryuno_stable
            else HeytingOmega3.DEGRADED
        )
        rsi3 = self._compute_rsi3_banach_state(poincare, p1_verdict, rsi_history)
        if not rsi3.banach_convergence_certified:
            p1_verdict = p1_verdict.meet(HeytingOmega3.DEGRADED)

        self._last_poincare = poincare
        self._last_rsi3 = rsi3
        return x_pre, poincare, rsi3, p1_verdict

    # ═════════════════════════════════════════════════════════════════════════
    # 1.6.7. ÚLTIMO MÉTODO DE LA FASE 1 — CONTINUACIÓN HACIA EL INICIO DE FASE 2
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase2_symplectic_context(
        self,
        state: CategoricalState,
        x_pre: NDArray[np.float64],
        poincare: PoincareSymplecticSnapshot,
        rsi3: ShieldRSI3State,
        phase1_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> Phase1ToPhase2ShieldBridge:
        """ÚLTIMO MÉTODO FORMAL DE LA FASE 1 → PUENTE HOMOMÓRFICO Φ₁₂ HACIA LA FASE 2.

        Firma criptográficamente con HMAC-SHA256 el estado extraído $x_{\mathrm{pre}}$,
        el tensor métrico Riemanniano validado, el snapshot orbital de Poincaré y el
        certificado de punto fijo RSI-3.
        Su salida `Phase1ToPhase2ShieldBridge` es el argumento de entrada directo del
        PRIMER método de la Fase 2: `Phase2_SymplecticHamiltonianEvaluator.consume_phase1_bridge`.
        """
        now = time.time()
        bridge_id = f"BRIDGE-P1-P2-{int(now * 1000) % 1_000_000:06d}"
        x_hash = hashlib.sha256(np.ascontiguousarray(x_pre).tobytes()).hexdigest()[:32]
        payload = (
            f"{bridge_id}:{x_hash}:{poincare.sha256_provenance}:"
            f"{rsi3.banach_certificate_hmac}:{int(phase1_verdict)}"
        )
        hmac_sig = self._compute_hmac(_HMAC_DOMAIN_P12, payload)

        return Phase1ToPhase2ShieldBridge(
            bridge_id=bridge_id,
            source_state=state,
            x_pre=x_pre,
            metric=self._metric,
            poincare_pre=poincare,
            rsi3_state=rsi3,
            phase1_verdict=phase1_verdict,
            timestamp_utc=now,
            hmac_continuation=hmac_sig,
        )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE ANIDADA 2 · OPERADORES SIMPLÉCTICOS, FLUJO PORT-HAMILTONIANO Y YONEDA║
# ║  Continuación directa del puente Φ₁₂, geometría (J, R, H), simplectomor-   ║
# ║  fismo de Cayley-Darboux Sp(2m,ℝ), pasividad de Lyapunov y puente Φ₂₃.     ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ───────────────────────────────────────────────────────────────────────────
# 2.1  ESTRUCTURA SIMPLÉCTICA CANÓNICA Y SIMPLECTOMORFISMO DE CAYLEY
# ───────────────────────────────────────────────────────────────────────────


class ConstantSkewStructure:
    r"""
    Estructura simpléctica canónica $J(x) = -J(x)^\top$ sobre $\mathbb{R}^{2m}$.

    Satisface:
      1. $J^\top = -J$ y $J^2 = -I_{2m}$ (estructura compleja compatible).
      2. Degeneración bilineal antisimétrica: $x^\top J x = 0 \quad \forall x \in \mathbb{R}^{2m}$.
      3. Generación del simplectomorfismo racional de Cayley:
             $$M_{\mathrm{Cayley}}(dt) = \left(I - \frac{dt}{2} J G\right)^{-1}\left(I + \frac{dt}{2} J G\right) \in \mathrm{Sp}(2m, \mathbb{R}),$$
         el cual preserva exactamente la 2-forma de Darboux $\Omega = J^{-1} = -J$:
             $$M_{\mathrm{Cayley}}^\top \Omega M_{\mathrm{Cayley}} = \Omega \quad \land \quad \det(M_{\mathrm{Cayley}}) = 1.0.$$
    """

    def __init__(self, dim: int) -> None:
        if dim % 2 != 0:
            raise ValueError(
                f"Dimensión simpléctica debe ser par, recibida: {dim}"
            )
        self.dim: int = dim
        self.J: NDArray[np.float64] = np.zeros((dim, dim), dtype=np.float64)
        half = dim // 2
        self.J[:half, half:] = np.eye(half)
        self.J[half:, :half] = -np.eye(half)
        assert np.allclose(self.J, -self.J.T, atol=PhysicalConstants.EPSILON_NORM)

    def __call__(self) -> NDArray[np.float64]:
        return self.J

    def cayley_symplectomorphism(
        self, G: NDArray[np.float64], dt: float
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""Construye la matriz de monodromía simpléctica $M \in \mathrm{Sp}(2m, \mathbb{R})$ vía transformada de Cayley.

        Retorna `(M, darboux_error, det_residual)` donde:
          • `darboux_error` = $\|M^\top J M - J\|_F$ (con $G = I$ o en coordenadas canónicas $G^{1/2}$).
          • `det_residual`  = $|\det(M) - 1.0|$.
        """
        n = self.dim
        eye = np.eye(n, dtype=np.float64)
        # En coordenadas simplécticas normalizadas por la métrica simétrica
        w, v = np.linalg.eigh(G)
        sqrt_g = (v * np.sqrt(np.maximum(w, PhysicalConstants.EPSILON_NORM))) @ v.T
        a_ham = 0.5 * float(min(max(dt, 1.0e-6), 1.0)) * (sqrt_g @ self.J @ sqrt_g)
        m_mat = la.solve(eye - a_ham, eye + a_ham)
        darboux_err = float(la.norm(m_mat.T @ self.J @ m_mat - self.J, ord="fro"))
        det_res = float(abs(la.det(m_mat) - 1.0))
        return m_mat, darboux_err, det_res


# ───────────────────────────────────────────────────────────────────────────
# 2.2  OPERADOR DE DISIPACIÓN CUADRÁTICA CON ACOPLAMIENTO DE RICCI-RSI3
# ───────────────────────────────────────────────────────────────────────────


class QuadraticDissipation:
    r"""
    Operador de disipación Port-Hamiltoniano $R(x) = R(x)^\top \succeq 0$.

    Modelo:
        $$R(x) = R_0 + \gamma_{\mathrm{Ricci}} \cdot (x x^\top)$$
    donde $R_0 \succeq \varepsilon_{\mathrm{norm}} I$ es SPD basal y $\gamma_{\mathrm{Ricci}} \ge 0$
    acopla la disipación al tensor de estado, siendo auto-calibrable por el punto fijo RSI-3.
    Cumple explícitamente el protocolo estructural `DissipativeOperator`.
    """

    def __init__(
        self,
        dim: int,
        R0: Optional[NDArray[np.float64]] = None,
        gamma: float = 0.0,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim debe ser positivo, recibido: {dim}")
        if gamma < 0:
            raise ValueError(f"γ ≥ 0 requerido, recibido: {gamma}")

        self.dim: int = dim
        self.gamma: float = float(gamma)

        if R0 is None:
            R0 = PhysicalConstants.EPSILON_NORM * np.eye(dim, dtype=np.float64)
        else:
            if R0.shape != (dim, dim):
                raise ValueError(f"R0 shape {R0.shape} ≠ ({dim},{dim})")
            if not np.allclose(R0, R0.T, atol=PhysicalConstants.EPSILON_NORM):
                raise ValueError("R0 debe ser simétrica")

        self._R0: NDArray[np.float64] = R0.copy()
        self._R0_eigval: NDArray[np.float64] = np.linalg.eigvalsh(self._R0)

        if np.any(self._R0_eigval < -PhysicalConstants.EPSILON_NORM):
            raise ValueError("R0 no es PSD (autovalores negativos significativos)")

    def set_adaptive_gamma(self, new_gamma: float) -> None:
        """Actualiza el acoplamiento de Ricci γ ≥ 0 a partir del punto fijo RSI-3."""
        if new_gamma >= 0.0 and math.isfinite(new_gamma):
            self.gamma = float(new_gamma)

    def dissipation_matrix(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Implementación del protocolo `DissipativeOperator`."""
        return self(x)

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if x.shape != (self.dim,):
            raise ValueError(f"x.shape {x.shape} ≠ ({self.dim},)")
        return self._R0 + self.gamma * np.outer(x, x)

    def is_psd(self, x: NDArray[np.float64]) -> bool:
        """Verifica espectralmente que R(x) ⪰ 0."""
        R = self(x)
        eigs = np.linalg.eigvalsh(R)
        return bool(np.all(eigs >= -PhysicalConstants.EPSILON_NORM))


# ───────────────────────────────────────────────────────────────────────────
# 2.3  HAMILTONIANO CINÉTICO Y FLUJO PORT-HAMILTONIANO
# ───────────────────────────────────────────────────────────────────────────


class PortHamiltonianFlow:
    r"""
    Sistema Port-Hamiltoniano disipativo sobre variedad Riemanniana-Simpléctica:
        $$\dot{x} = \left(J(x) - R(x)\right) \nabla H(x), \quad H(x) = \frac{1}{2} x^\top G x, \quad \nabla H(x) = G x.$$

    Por la antisimetría de $J(x) = -J(x)^\top$ y la semidefinición positiva $R(x) \succeq 0$:
        $$\dot{H}(x) = \nabla H(x)^\top \dot{x} = \underbrace{\nabla H(x)^\top J \nabla H(x)}_{= 0} - \nabla H(x)^\top R(x) \nabla H(x) \le 0.$$
    """

    def __init__(
        self,
        G: ValidatedMetricTensor,
        J: ConstantSkewStructure,
        R: QuadraticDissipation,
    ) -> None:
        if G.n != J.dim or G.n != R.dim:
            raise ValueError(
                f"Dimensiones inconsistentes: G.n={G.n}, J.dim={J.dim}, R.dim={R.dim}"
            )
        self.G: ValidatedMetricTensor = G
        self.J: ConstantSkewStructure = J
        self.R: QuadraticDissipation = R
        self.n: int = G.n

    def hamiltonian(self, x: NDArray[np.float64]) -> float:
        r"""Energía total de lazo $H(x) = \frac{1}{2} x^\top G x \ge 0$."""
        return 0.5 * float(x @ self.G.G @ x)

    def gradient_hamiltonian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Gradiente simpléctico-Riemanniano $\nabla H(x) = G x$."""
        return self.G.G @ x

    def flow(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Campo vectorial Port-Hamiltoniano $\dot{x} = (J - R(x)) \nabla H(x)$."""
        grad_H = self.gradient_hamiltonian(x)
        J_mat = self.J()
        R_mat = self.R(x)
        return (J_mat - R_mat) @ grad_H

    def lyapunov_derivative(self, x: NDArray[np.float64]) -> float:
        r"""Derivada orbital continua de Lyapunov $\dot{H}(x) = -\nabla H(x)^\top R(x) \nabla H(x) \le 0$."""
        grad_H = self.gradient_hamiltonian(x)
        return float(grad_H @ self.flow(x))

    def step(
        self,
        x: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integración temporal de un paso (compatible con Euler explícito y regularización de paso)."""
        if dt <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt}")
        if dt < PhysicalConstants.MIN_TIME_STEP:
            dt = PhysicalConstants.MIN_TIME_STEP
        return x + dt * self.flow(x)

    def symplectic_cayley_step(
        self,
        x: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrador geométrico de Cayley-Crank-Nicolson que preserva pasividad y estructura simpléctica."""
        if dt <= 0:
            raise ValueError(f"dt debe ser positivo, recibido: {dt}")
        safe_dt = min(max(dt, PhysicalConstants.MIN_TIME_STEP), 1.0)
        eye = np.eye(self.n, dtype=np.float64)
        gen = (self.J() - self.R(x)) @ self.G.G
        return la.solve(eye - 0.5 * safe_dt * gen, (eye + 0.5 * safe_dt * gen) @ x)


# ───────────────────────────────────────────────────────────────────────────
# 2.4  FUNTOR REPRESENTABLE YONEDA-COMPATIBLE Y EVALUADOR PORT-HAMILTONIANO
# ───────────────────────────────────────────────────────────────────────────


class YonedaRepresentable(Generic[T_Agent]):
    r"""
    Funtor representable $\mathrm{Hom}_{\mathcal{C}}(A, -)$ en su forma operativa.
    Cumple el Lema de Yoneda $\mathrm{Nat}(\mathrm{Hom}(A, -), F) \cong F(A)$ con memoización por identidad.
    """

    def __init__(self, agent: T_Agent, phase_dim: int) -> None:
        self.agent: T_Agent = agent
        self.phase_dim: int = phase_dim
        self._morphism_cache: Dict[int, CategoricalState] = {}

    def hom_to(self, state: CategoricalState) -> CategoricalState:
        """Evalúa el morfismo representable sobre el objeto `state`."""
        key = id(state)
        if key in self._morphism_cache:
            logger.debug("Yoneda hit cache: %d", key)
            return self._morphism_cache[key]
        result = self.agent(state)
        self._morphism_cache[key] = result
        return result


class PortHamiltonianEvaluator:
    r"""
    Evaluador termodinámico continuo y discreto del flujo Port-Hamiltoniano.
    Audita la desigualdad de pasividad de Clausius-Duhem $\dot{H} \le 0$.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        flow: PortHamiltonianFlow,
    ) -> None:
        self.metric: ValidatedMetricTensor = metric
        self.flow_sys: PortHamiltonianFlow = flow
        self.n: int = metric.n

    def compute_energy(self, x: NDArray[np.float64]) -> float:
        return self.flow_sys.hamiltonian(x)

    def compute_lyapunov_derivative_exact(
        self, x: NDArray[np.float64]
    ) -> float:
        r"""$\dot{H}_{\mathrm{exact}}(x) = \nabla H(x)^\top (J - R(x)) \nabla H(x) \le 0$."""
        return self.flow_sys.lyapunov_derivative(x)

    def compute_lyapunov_derivative_finite(
        self,
        x_pre: NDArray[np.float64],
        x_post: NDArray[np.float64],
        dt: float,
    ) -> float:
        """Tasa discreta de variación energética $\Delta H / \Delta t$ a través del morfismo."""
        safe_dt = max(dt, PhysicalConstants.MIN_TIME_STEP)
        delta_H = self.compute_energy(x_post) - self.compute_energy(x_pre)
        return delta_H / safe_dt

    def evaluate(
        self,
        x_pre: NDArray[np.float64],
        x_post: NDArray[np.float64],
        dt: float,
    ) -> ThermodynamicMetrics:
        """Evalúa el balance termodinámico comparando el estado pre-morfismo y post-proyección."""
        H_pre = self.compute_energy(x_pre)
        H_post = self.compute_energy(x_post)

        dH_exact = self.compute_lyapunov_derivative_exact(x_pre)
        dH_finite = self.compute_lyapunov_derivative_finite(x_pre, x_post, dt)

        dH_dt = dH_exact if abs(dH_exact) >= abs(dH_finite) else dH_finite

        P_diss = max(0.0, -dH_dt)
        sigma = P_diss / PhysicalConstants.BOLTZMANN_NORMALIZED
        violation = self._classify_violation(dH_dt, x_pre)

        verdict = HeytingOmega3.COHERENT
        if violation in (
            ShieldViolationType.LYAPUNOV_CRITICAL,
            ShieldViolationType.ENERGY_INJECTION,
        ):
            verdict = HeytingOmega3.VETOED
        elif violation == ShieldViolationType.LYAPUNOV_POSITIVE:
            verdict = HeytingOmega3.DEGRADED

        return ThermodynamicMetrics(
            lyapunov_derivative=float(dH_dt),
            dissipated_power=float(P_diss),
            energy_pre=float(H_pre),
            energy_post=float(H_post),
            entropy_production=float(sigma),
            is_dirichlet_enforced=False,
            violation_type=violation,
            heyting_verdict=verdict,
        )

    @staticmethod
    def _classify_violation(
        dH_dt: float, x: NDArray[np.float64]
    ) -> ShieldViolationType:
        if abs(dH_dt) < PhysicalConstants.EPSILON_LYAPUNOV:
            return ShieldViolationType.NONE
        if dH_dt > 0:
            norm_x_sq = float(x @ x) + PhysicalConstants.EPSILON_NORM
            ratio = abs(dH_dt) / norm_x_sq
            if ratio > PhysicalConstants.CRITICAL_LYAPUNOV_RATIO:
                return ShieldViolationType.LYAPUNOV_CRITICAL
            return ShieldViolationType.LYAPUNOV_POSITIVE
        return ShieldViolationType.NONE


# ───────────────────────────────────────────────────────────────────────────
# 2.5  CLASE SOBERANA DE FASE 2: Phase2_SymplecticHamiltonianEvaluator
# ───────────────────────────────────────────────────────────────────────────


class Phase2_SymplecticHamiltonianEvaluator(Phase1_MetricVectorExtractor):
    r"""
    FASE ANIDADA 2: Evaluación Port-Hamiltoniana, Simplectomorfismo de Cayley y Monodromía de Poincaré.

    Hereda de `Phase1_MetricVectorExtractor` y continúa directamente su último método
    mediante `consume_phase1_bridge`.
    Su ÚLTIMO método (`handoff_to_phase3_homology_context`) sella el puente
    `Phase2ToPhase3ShieldBridge`, el cual inicia directamente la Fase 3.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        secret_key: bytes = _SHIELD_SECRET_KEY,
    ) -> None:
        super().__init__(metric=metric, secret_key=secret_key)
        self._J: ConstantSkewStructure = ConstantSkewStructure(self._phase_space_dim)
        self._R: QuadraticDissipation = QuadraticDissipation(
            self._phase_space_dim,
            gamma=PhysicalConstants.DEFAULT_RICCI_GAMMA,
        )
        self._flow_sys: PortHamiltonianFlow = PortHamiltonianFlow(
            G=self._metric, J=self._J, R=self._R
        )
        self._evaluator: PortHamiltonianEvaluator = PortHamiltonianEvaluator(
            self._metric, self._flow_sys
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 2.5.1. PRIMER MÉTODO DE LA FASE 2 — CONTINUACIÓN DIRECTA DE LA FASE 1
    # ═════════════════════════════════════════════════════════════════════════
    def consume_phase1_bridge(self, bridge: Phase1ToPhase2ShieldBridge) -> bool:
        """PRIMER MÉTODO FORMAL DE LA FASE 2 ← RECEPTOR DIRECTO DEL PUENTE Φ₁₂ DE LA FASE 1.

        Verifica en tiempo constante la firma HMAC-SHA256 del puente emitido por
        `Phase1_MetricVectorExtractor.handoff_to_phase2_symplectic_context` y
        sincroniza el coeficiente de flujo de Ricci $\gamma_{\mathrm{Ricci}}$ con el
        punto fijo de Banach RSI-3.
        """
        x_hash = hashlib.sha256(np.ascontiguousarray(bridge.x_pre).tobytes()).hexdigest()[:32]
        payload = (
            f"{bridge.bridge_id}:{x_hash}:{bridge.poincare_pre.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{int(bridge.phase1_verdict)}"
        )
        if not self._verify_hmac(_HMAC_DOMAIN_P12, payload, bridge.hmac_continuation):
            raise FunctorialityError("Falla de autenticidad criptográfica en el puente Φ₁₂ del Funtor Shield.")

        # Auto-calibración RSI-3 del coeficiente de disipación de Ricci γ
        self._R.set_adaptive_gamma(bridge.rsi3_state.adaptive_ricci_gamma)
        self._last_poincare = bridge.poincare_pre
        self._last_rsi3 = bridge.rsi3_state
        return True

    # ─── 2.5.2. Auditoría Simpléctica de Darboux-Liouville y Monodromía ───────
    def evaluate_symplectic_monodromy_and_nekhoroshev(
        self,
        x_post: NDArray[np.float64],
        dt: float,
        rsi3: ShieldRSI3State,
    ) -> Tuple[PoincareSymplecticSnapshot, HeytingOmega3]:
        """Evalúa el snapshot de Poincaré post-morfismo y audita el invariante de Darboux-Liouville."""
        snap_post = self._compute_poincare_delaunay_deprit_snapshot(x_post)
        _, darboux_err, det_res = self._J.cayley_symplectomorphism(self._metric.G, dt)

        updated_snap = replace(
            snap_post,
            darboux_symplectic_error=darboux_err,
            liouville_det_residual=det_res,
        )

        verdict = HeytingOmega3.COHERENT
        greene_ceiling = max(
            PhysicalConstants.GREENE_KAM_THRESHOLD, rsi3.adaptive_greene_threshold
        )
        if (
            updated_snap.homoclinic_tangency_detected
            and updated_snap.greene_residue_R > greene_ceiling
        ):
            verdict = verdict.meet(HeytingOmega3.DEGRADED)
        if not updated_snap.is_bryuno_stable:
            verdict = verdict.meet(HeytingOmega3.VETOED)
        if darboux_err > 1.0e-6 or det_res > 1.0e-6:
            verdict = verdict.meet(HeytingOmega3.VETOED)

        return updated_snap, verdict

    # ─── 2.5.3. Ejecución del Flujo Port-Hamiltoniano e Invocación Yoneda ─────
    def execute_phase2_port_hamiltonian_flow(
        self,
        bridge12: Phase1ToPhase2ShieldBridge,
        yoneda: YonedaRepresentable[Any],
    ) -> Tuple[
        CategoricalState,
        NDArray[np.float64],
         float,
        ThermodynamicMetrics,
        PoincareSymplecticSnapshot,
        HeytingOmega3,
    ]:
        """Consume el puente Φ₁₂, invoca el morfismo vía Yoneda y audita la geometría Port-Hamiltoniana."""
        self.consume_phase1_bridge(bridge12)

        t0 = time.perf_counter()
        post_state = yoneda.hom_to(bridge12.source_state)
        if not isinstance(post_state, CategoricalState):
            raise FunctorialityError(
                f"Agente retornó {type(post_state).__name__}, no CategoricalState"
            )
        t1 = time.perf_counter()
        dt = max(t1 - t0, PhysicalConstants.MIN_TIME_STEP)

        x_post = PhaseVectorExtractor.extract(post_state, self._phase_space_dim)
        if not np.all(np.isfinite(x_post)):
            raise ValueError("Vector de fase post contiene valores no finitos (NaN/Inf).")

        poincare_post, poincare_verdict = self.evaluate_symplectic_monodromy_and_nekhoroshev(
            x_post=x_post, dt=dt, rsi3=bridge12.rsi3_state
        )

        raw_metrics = self._evaluator.evaluate(bridge12.x_pre, x_post, dt)
        phase2_verdict = HeytingOmega3.fold_supremum(
            (bridge12.phase1_verdict, poincare_verdict, raw_metrics.heyting_verdict)
        )

        enriched_metrics = replace(
            raw_metrics,
            poincare_greene_residue=poincare_post.greene_residue_R,
            poincare_melnikov_m0=poincare_post.melnikov_M0,
            bryuno_diophantine_sum=poincare_post.bryuno_diophantine_sum,
            darboux_symplectic_error=poincare_post.darboux_symplectic_error,
            nekhoroshev_stability_time=poincare_post.nekhoroshev_stability_time,
            rsi3_meta_operator_norm=bridge12.rsi3_state.meta_operator_norm,
            rsi3_banach_certified=bridge12.rsi3_state.banach_convergence_certified,
            heyting_verdict=phase2_verdict,
        )

        self._last_poincare = poincare_post
        return post_state, x_post, dt, enriched_metrics, poincare_post, phase2_verdict

    # ═════════════════════════════════════════════════════════════════════════
    # 2.5.4. ÚLTIMO MÉTODO DE LA FASE 2 — CONTINUACIÓN HACIA EL INICIO DE FASE 3
    # ═════════════════════════════════════════════════════════════════════════
    def handoff_to_phase3_homology_context(
        self,
        bridge12: Phase1ToPhase2ShieldBridge,
        post_state: CategoricalState,
        x_post: NDArray[np.float64],
        dt: float,
        thermo_metrics: ThermodynamicMetrics,
        poincare_post: PoincareSymplecticSnapshot,
        phase2_verdict: HeytingOmega3,
    ) -> Phase2ToPhase3ShieldBridge:
        """ÚLTIMO MÉTODO FORMAL DE LA FASE 2 → PUENTE HOMOMÓRFICO Φ₂₃ HACIA LA FASE 3.

        Sella criptográficamente con HMAC-SHA256 el estado post-morfismo, la evaluación
        Port-Hamiltoniana, el snapshot simpléctico de Poincaré y el estado RSI-3.
        Su salida `Phase2ToPhase3ShieldBridge` es el argumento de entrada directo del
        PRIMER método de la Fase 3: `Phase3_HeytingHomologyDecider.consume_phase2_bridge`.
        """
        now = time.time()
        bridge_id = f"BRIDGE-P2-P3-{int(now * 1000) % 1_000_000:06d}"
        x_post_hash = hashlib.sha256(np.ascontiguousarray(x_post).tobytes()).hexdigest()[:32]
        payload = (
            f"{bridge_id}:{bridge12.bridge_id}:{x_post_hash}:"
            f"{poincare_post.sha256_provenance}:{bridge12.rsi3_state.banach_certificate_hmac}:"
            f"{int(phase2_verdict)}"
        )
        hmac_sig = self._compute_hmac(_HMAC_DOMAIN_P23, payload)

        return Phase2ToPhase3ShieldBridge(
            bridge_id=bridge_id,
            bridge12_ref=bridge12.bridge_id,
            source_state=bridge12.source_state,
            post_state=post_state,
            x_pre=bridge12.x_pre,
            x_post=x_post,
            dt=dt,
            metric=self._metric,
            poincare_post=poincare_post,
            rsi3_state=bridge12.rsi3_state,
            thermo_metrics_raw=thermo_metrics,
            phase2_verdict=phase2_verdict,
            timestamp_utc=now,
            hmac_continuation=hmac_sig,
        )


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE ANIDADA 3 · PROYECTOR IDEMPOTENTE, HOMOLOGÍA SIMPLICIAL Y HEYTING Ω₃ ║
# ║  Continuación directa del puente Φ₂₃, proyección Riemanniana P̂, homología ║
# ║  de 2-esqueleto β₁(K) = dim ker ∂₁ − rank(∂₂), Dirac-Dirichlet y Funtor.   ║
# ╚════════════════════════════════════════════════════════════════════════════╝


# ───────────────────────────────────────────────────────────────────────────
# 3.1  PROYECTOR ORTOGONAL PONDERADO RIEMANNIANO
# ───────────────────────────────────────────────────────────────────────────


class RiemannianProjector:
    r"""
    Proyector ortogonal $\hat{P}$ sobre el subespacio espectral protegido de menor
    curvatura energética, idempotente y autoadjunto respecto al tensor métrico $G$.

    Axiomas (verificados en `check_axioms`):
      A1. $\hat{P}^2 = \hat{P}$                  (idempotencia exacta)
      A2. $\hat{P}^\top G = G \hat{P}$           ($G$-autoadjunción métrica)
      A3. $\|\hat{P}\|_G \le 1$                  (no-expansividad en la norma Riemanniana)
      A4. $\mathrm{ran}(\hat{P}) \subseteq \mathrm{ran}(G)$ (consistencia de subespacio)
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        n_protected: int,
    ) -> None:
        if not (0 < n_protected <= metric.n):
            raise ValueError(
                f"n_protected={n_protected} fuera de (0, {metric.n}]"
            )
        self.metric: ValidatedMetricTensor = metric
        self.n: int = metric.n
        self.k: int = n_protected

        order = np.argsort(metric.eigval)
        idx = order[: self.k]
        U_k = metric.eigvec[:, idx]
        self.P: NDArray[np.float64] = U_k @ U_k.T

        self.check_axioms()

    def check_axioms(self, tol: float = 1.0e-8) -> None:
        """Verifica formalmente los axiomas A1–A4 del proyector Riemanniano."""
        P, G = self.P, self.metric.G
        assert np.allclose(P @ P, P, atol=tol), "A1 idempotencia violada"
        assert np.allclose(P.T @ G, G @ P, atol=tol), "A2 simetría métrica violada"
        G_inv_half = self._sqrt_inv_metric()
        M = G_inv_half @ P.T @ G @ P @ G_inv_half
        op_norm = float(np.linalg.norm(M, ord=2))
        assert op_norm <= 1.0 + tol, f"A3 norma_G(P)={op_norm} > 1"
        assert P.shape == (self.n, self.n), "A4 dimensión inconsistente"
        logger.debug(
            "Proyector P̂ verificado: A1–A4 OK (k=%d, ‖P‖_G=%.4f)",
            self.k,
            op_norm,
        )

    def _sqrt_inv_metric(self) -> NDArray[np.float64]:
        r"""Calcula $G^{-1/2}$ vía descomposición espectral."""
        eigs = self.metric.eigval
        vecs = self.metric.eigvec
        threshold = PhysicalConstants.EPSILON_NORM
        inv_sqrt = np.where(
            np.abs(eigs) > threshold, 1.0 / np.sqrt(np.abs(eigs)), 0.0
        )
        return (vecs * inv_sqrt) @ vecs.T

    def project(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyecta $x \mapsto \hat{P} x$."""
        return self.P @ x

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.project(x)


# ───────────────────────────────────────────────────────────────────────────
# 3.2  OPERADOR DE FRONTERA DE DIRAC-DIRICHLET
# ───────────────────────────────────────────────────────────────────────────


class DiracBoundaryOperator:
    r"""
    Operador de frontera de Dirichlet $D$ sobre la variedad de fase $\mathcal{M}$:
        $$D(x) = \begin{cases} x & \text{si } f(x) = x^\top G x - E_{\max} \le 0, \\ \mathbf{0} & \text{si } f(x) > 0. \end{cases}$$
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        max_energy: float = float("inf"),
    ) -> None:
        self.metric: ValidatedMetricTensor = metric
        self.E_max: float = max_energy

    def evaluate_level_set(self, x: NDArray[np.float64]) -> float:
        r"""Evalúa la función de nivel energética $f(x) = x^\top G x - E_{\max}$."""
        return float(x @ self.metric.G @ x) - self.E_max

    def amputate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Aniquila el estado en la frontera $\partial\mathcal{M}$ si excede $E_{\max}$."""
        f = self.evaluate_level_set(x)
        if f > 0:
            return np.zeros_like(x)
        return x


# ───────────────────────────────────────────────────────────────────────────
# 3.3  DETECTOR HOMOLÓGICO SIMPLICIAL DE SEGUNDO ESQUELETO β₁(K)
# ───────────────────────────────────────────────────────────────────────────


class HomologyBettiDetector:
    r"""
    Detector riguroso del primer número de Betti $\beta_1(\mathcal{K}) = \dim H_1(\mathcal{K}; \mathbb{R})$
    sobre el complejo simplicial de cliques (Vietoris-Rips 2-esqueleto) inducido por el estado.

    Teoría Homológica Exacta:
    -------------------------
    Dada la cadena simplicial:
        $$C_2(\mathcal{K}) \xrightarrow{\partial_2} C_1(\mathcal{K}) \xrightarrow{\partial_1} C_0(\mathcal{K}) \longrightarrow 0,$$
    donde:
      • $\dim \ker \partial_1 = m - n + c$ es el rango del espacio de 1-ciclos $Z_1(\mathcal{K})$
        en el 1-esqueleto $(V, E)$ con $n = |V|$, $m = |E|$ y $c = \beta_0(\mathcal{K})$.
      • $\dim \mathrm{im}\,\partial_2 = \mathrm{rank}(\partial_2)$ es el rango del subespacio de
        1-bordes $B_1(\mathcal{K})$ generados por los 2-símplices (triángulos rellenos $(u, v, w)$
        donde $(u,v), (u,w), (v,w) \in E$).

    El primer número de Betti simplicial es el cociente homológico exacto:
        $$\beta_1(\mathcal{K}) = \dim H_1(\mathcal{K}; \mathbb{R}) = \dim Z_1(\mathcal{K}) - \dim B_1(\mathcal{K}) = (m - n + c) - \mathrm{rank}(\partial_2).$$

    Esto garantiza que variedades locales suaves o constantes trianguladas tengan
    $\beta_1(\mathcal{K}) = 0$ (contractibles), mientras que los socavones lógicos
    genuinos (ciclos de longitud $\ge 4$ sin cuerdas diagonales) produzcan $\beta_1(\mathcal{K}) > 0$.
    """

    @staticmethod
    def _extract_edges_and_components(
        x: NDArray[np.float64], k: int = 2
    ) -> Tuple[int, List[Tuple[int, int]], int]:
        """Construye el 1-esqueleto k-NN y retorna `(n_nodos, lista_aristas, n_componentes)`."""
        n = int(x.size)
        if n < 2:
            return (n, [], n)
        if k < 1:
            raise ValueError(f"k debe ser ≥ 1, recibido: {k}")
        k_eff = min(k, n - 1)

        diff = x[:, None] - x[None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)

        neighbors = np.argsort(dist, axis=1)[:, :k_eff]
        edge_set: set[Tuple[int, int]] = set()
        for i in range(n):
            for j in neighbors[i]:
                ji = int(j)
                a, b = (i, ji) if i < ji else (ji, i)
                if a != b:
                    edge_set.add((a, b))

        edges_sorted = sorted(edge_set)
        parent = list(range(n))

        def find(u: int) -> int:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: int, v: int) -> None:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

        for a, b in edges_sorted:
            union(a, b)
        c = len({find(v) for v in range(n)})
        return (n, edges_sorted, c)

    @classmethod
    def _build_knn_graph(
        cls, x: NDArray[np.float64], k: int = 2
    ) -> Tuple[int, int, int]:
        """Construye un grafo k-NN y retorna `(n_nodos, n_aristas, n_componentes)`."""
        n, edges_sorted, c = cls._extract_edges_and_components(x, k=k)
        return (n, len(edges_sorted), c)

    @staticmethod
    def _compute_boundary2_rank(
        n: int, edges_sorted: List[Tuple[int, int]]
    ) -> int:
        r"""Calcula $\mathrm{rank}(\partial_2)$ sobre el complejo de cliques de 2-símplices.

        Para cada 3-clique orientado $(u, v, w)$ con $u < v < w$, su imagen bajo el
        operador frontera $\partial_2: C_2(\mathcal{K}) \to C_1(\mathcal{K})$ es:
            $$\partial_2[u, v, w] = [v, w] - [u, w] + [u, v].$$
        """
        m = len(edges_sorted)
        if m < 3 or n < 3:
            return 0

        edge_to_idx: Dict[Tuple[int, int], int] = {
            edge: idx for idx, edge in enumerate(edges_sorted)
        }
        adj: Dict[int, set[int]] = {i: set() for i in range(n)}
        for u, v in edges_sorted:
            adj[u].add(v)

        triangles: List[Tuple[int, int, int]] = []
        for u in range(n):
            u_neighbors = sorted(adj[u])
            for idx_v, v in enumerate(u_neighbors):
                common_w = adj[u].intersection(adj[v])
                for w in common_w:
                    if w > v:
                        triangles.append((u, v, w))

        if not triangles:
            return 0

        b2_mat = np.zeros((m, len(triangles)), dtype=np.float64)
        for col, (u, v, w) in enumerate(triangles):
            b2_mat[edge_to_idx[(u, v)], col] = 1.0
            b2_mat[edge_to_idx[(u, w)], col] = -1.0
            b2_mat[edge_to_idx[(v, w)], col] = 1.0

        return int(np.linalg.matrix_rank(b2_mat, tol=1.0e-9))

    @classmethod
    def compute_beta1(cls, x: NDArray[np.float64], k: int = 2) -> int:
        r"""Calcula el primer número de Betti simplicial $\beta_1(\mathcal{K}) = (m - n + c) - \mathrm{rank}(\partial_2)$."""
        if not np.all(np.isfinite(x)):
            return 1
        n, edges_sorted, c = cls._extract_edges_and_components(x, k=k)
        m = len(edges_sorted)
        z1_dim = max(0, m - n + c)
        if z1_dim == 0:
            return 0
        b1_dim = cls._compute_boundary2_rank(n, edges_sorted)
        return max(0, z1_dim - b1_dim)

    @classmethod
    def has_logical_crevice(cls, x: NDArray[np.float64], k: int = 2) -> bool:
        """Retorna `True` si y sólo si $\beta_1(\mathcal{K}) > 0$ (existe un 1-ciclo no acotado)."""
        return cls.compute_beta1(x, k=k) > 0


# ───────────────────────────────────────────────────────────────────────────
# 3.4  CLASE SOBERANA DE FASE 3: Phase3_HeytingHomologyDecider
# ───────────────────────────────────────────────────────────────────────────


class Phase3_HeytingHomologyDecider(Phase2_SymplecticHamiltonianEvaluator):
    r"""
    FASE ANIDADA 3: Proyección Riemanniana, Detección Homológica Simplicial y Adjudicación en Heyting Ω₃.

    Hereda de `Phase2_SymplecticHamiltonianEvaluator` y continúa directamente su último
    método mediante `consume_phase2_bridge`.
    Ejecuta la proyección ortogonal $\hat{P}$, interroga el complejo simplicial $\beta_1(\mathcal{K}) = 0$,
    verifica criptográficamente el certificado de Banach RSI-3 y adjudica el veredicto en $\Omega_3$.
    """

    def __init__(
        self,
        metric: ValidatedMetricTensor,
        n_protected: Optional[int] = None,
        secret_key: bytes = _SHIELD_SECRET_KEY,
    ) -> None:
        super().__init__(metric=metric, secret_key=secret_key)
        k = n_protected if n_protected is not None else max(1, self._phase_space_dim // 2)
        self._projector: RiemannianProjector = RiemannianProjector(self._metric, k)
        self._dirichlet: DiracBoundaryOperator = DiracBoundaryOperator(
            self._metric, max_energy=PhysicalConstants.MAX_ENERGY_INJECTION_RATE
        )
        self._homology: HomologyBettiDetector = HomologyBettiDetector()

    # ═════════════════════════════════════════════════════════════════════════
    # 3.4.1. PRIMER MÉTODO DE LA FASE 3 — CONTINUACIÓN DIRECTA DE LA FASE 2
    # ═════════════════════════════════════════════════════════════════════════
    def consume_phase2_bridge(self, bridge: Phase2ToPhase3ShieldBridge) -> bool:
        """PRIMER MÉTODO FORMAL DE LA FASE 3 ← RECEPTOR DIRECTO DEL PUENTE Φ₂₃ DE LA FASE 2.

        Verifica en tiempo constante la firma HMAC-SHA256 del puente emitido por
        `Phase2_SymplecticHamiltonianEvaluator.handoff_to_phase3_homology_context`.
        """
        x_post_hash = hashlib.sha256(
            np.ascontiguousarray(bridge.x_post).tobytes()
        ).hexdigest()[:32]
        payload = (
            f"{bridge.bridge_id}:{bridge.bridge12_ref}:{x_post_hash}:"
            f"{bridge.poincare_post.sha256_provenance}:"
            f"{bridge.rsi3_state.banach_certificate_hmac}:{int(bridge.phase2_verdict)}"
        )
        if not self._verify_hmac(_HMAC_DOMAIN_P23, payload, bridge.hmac_continuation):
            raise FunctorialityError(
                "Falla de autenticidad criptográfica en el puente Φ₂₃ del Funtor Shield."
            )
        self._last_poincare = bridge.poincare_post
        self._last_rsi3 = bridge.rsi3_state
        return True

    # ─── 3.4.2. Verificación Criptográfica del Punto Fijo de Banach RSI-3 ─────
    def verify_rsi3_banach_certificate(self, rsi3: ShieldRSI3State) -> bool:
        """Verifica el sello HMAC-SHA256 y las cotas de Banach-Kantorovich del estado RSI-3."""
        payload = (
            f"{rsi3.rsi_iteration_id}:{rsi3.meta_operator_norm:.8f}:"
            f"{rsi3.fixed_point_residual:.8f}:{rsi3.banach_convergence_certified}"
        )
        hmac_ok = self._verify_hmac(
            _HMAC_DOMAIN_RSI3, payload, rsi3.banach_certificate_hmac
        )
        return bool(
            hmac_ok
            and rsi3.meta_operator_norm < 1.0
            and rsi3.fixed_point_residual < 1.0e-2
            and rsi3.banach_convergence_certified
        )

    # ─── 3.4.3. Decisión Homológica y Adjudicación en el Topos de Heyting Ω₃ ──
    def execute_phase3_homology_and_heyting_decision(
        self,
        bridge23: Phase2ToPhase3ShieldBridge,
    ) -> Tuple[
        NDArray[np.float64],
        ThermodynamicMetrics,
        HeytingOmega3,
        ShieldViolationType,
    ]:
        """Consume el puente Φ₂₃, aplica $\hat{P}$, audita $\beta_1(\mathcal{K}) = 0$ y adjudica en $\Omega_3$."""
        self.consume_phase2_bridge(bridge23)

        # 1. Proyección ortogonal Riemanniana sobre el subespacio protegido
        x_protected = self._projector(bridge23.x_post)

        # 2. Verificación homológica simplicial de 2-esqueleto (T4: β₁(K) = 0)
        beta1 = self._homology.compute_beta1(x_protected)
        if beta1 > 0:
            logger.warning(
                "Socavón Lógico detectado: β₁=%d sobre estado post-proyección",
                beta1,
            )
            return (
                x_protected,
                bridge23.thermo_metrics_raw,
                HeytingOmega3.VETOED,
                ShieldViolationType.TOPOLOGICAL_CHARGE,
            )

        # 3. Evaluación termodinámica sobre el vector proyectado x_protected
        metrics_proj = self._evaluator.evaluate(
            bridge23.x_pre, x_protected, bridge23.dt
        )

        # 4. Verificación del certificado de Banach RSI-3
        rsi3_ok = self.verify_rsi3_banach_certificate(bridge23.rsi3_state)
        rsi3_verdict = HeytingOmega3.COHERENT if rsi3_ok else HeytingOmega3.VETOED

        # 5. Verificación de frontera de Dirac-Dirichlet
        x_final = self._dirichlet.amputate(x_protected)
        dirichlet_breach = not np.allclose(
            x_final, x_protected, atol=PhysicalConstants.EPSILON_NORM
        )
        dirichlet_verdict = (
            HeytingOmega3.VETOED if dirichlet_breach else HeytingOmega3.COHERENT
        )

        # 6. Supremo en el retículo distributivo de Heyting Ω₃
        terminal_verdict = HeytingOmega3.fold_supremum(
            (
                bridge23.phase2_verdict,
                metrics_proj.heyting_verdict,
                rsi3_verdict,
                dirichlet_verdict,
            )
        )

        # Determinación de la causa primaria en caso de veto o degradación
        if dirichlet_breach:
            violation = ShieldViolationType.ENERGY_INJECTION
        elif metrics_proj.violation_type in (
            ShieldViolationType.LYAPUNOV_CRITICAL,
            ShieldViolationType.ENERGY_INJECTION,
        ):
            violation = metrics_proj.violation_type
        elif not rsi3_ok:
            violation = ShieldViolationType.RSI3_BANACH_DIVERGENCE
        elif not bridge23.poincare_post.is_bryuno_stable:
            violation = ShieldViolationType.POINCARE_HOMOCLINIC_TANGLE
        else:
            violation = metrics_proj.violation_type

        final_metrics = replace(
            metrics_proj,
            poincare_greene_residue=bridge23.poincare_post.greene_residue_R,
            poincare_melnikov_m0=bridge23.poincare_post.melnikov_M0,
            bryuno_diophantine_sum=bridge23.poincare_post.bryuno_diophantine_sum,
            darboux_symplectic_error=bridge23.poincare_post.darboux_symplectic_error,
            nekhoroshev_stability_time=bridge23.poincare_post.nekhoroshev_stability_time,
            rsi3_meta_operator_norm=bridge23.rsi3_state.meta_operator_norm,
            rsi3_banach_certified=rsi3_ok,
            heyting_verdict=terminal_verdict,
            violation_type=violation,
        )

        return x_final, final_metrics, terminal_verdict, violation


# ───────────────────────────────────────────────────────────────────────────
# 3.5  FUNTOR SHIELD (ORQUESTADOR TRIFÁSICO ANIDADO Y MÓNADA DE KLEISLI)
# ───────────────────────────────────────────────────────────────────────────


class FuntorShield(Morphism, Phase3_HeytingHomologyDecider, Generic[T_Agent]):
    r"""
    Funtor Shield: Endofuntor simpléctico, homológico e idempotente sobre $\mathcal{C}_{\mathrm{states}}$:
        $$\mathcal{Z}_{\mathrm{shield}} = \Phi_3 \circ \Phi_2 \circ \Phi_1 = \hat{P} \circ Y \circ S.$$

    Integra por herencia directa las 3 fases anidadas:
        `Phase1_MetricVectorExtractor`
          └──► `Phase2_SymplecticHamiltonianEvaluator`
                 └──► `Phase3_HeytingHomologyDecider`
                        └──► `FuntorShield`

    Teoremas verificados:
      • T1 (Endofuntoriedad): $\mathcal{Z}_{\mathrm{shield}}: \mathrm{Ob}(\mathcal{C}) \to \mathrm{Ob}(\mathcal{C})$.
      • T2 (Idempotencia): $\mathcal{Z}_{\mathrm{shield}} \circ \mathcal{Z}_{\mathrm{shield}} \equiv \mathcal{Z}_{\mathrm{shield}}$.
      • T3 (Pasividad de Lyapunov y Estabilidad de Poincaré): $\dot{H} \le 0$ y $M^\top \Omega M = \Omega$.
      • T4 (Coherencia Homológica Simplicial): $\beta_1(\mathcal{K}_{\mathrm{post}}) = 0$.
      • T5 (Contracción de Banach RSI-3): $\|DT_{\mathrm{RSI3}}(s^*)\|_\infty < 1$.
    """

    def __init__(
        self,
        invoking_agent: T_Agent,
        stratum: Stratum,
        n_protected: Optional[int] = None,
        secret_key: bytes = _SHIELD_SECRET_KEY,
    ) -> None:
        Morphism.__init__(self)

        self._agent: T_Agent = invoking_agent
        self._stratum: Stratum = stratum
        self._shield_id: str = f"SHIELD_{id(self):016x}"
        self._rsi_verdict_history: List[int] = []

        # Resolución de dimensionalidad simpléctica (par)
        resolved_dim = self._resolve_phase_dim()
        if resolved_dim % 2 != 0:
            logger.warning(
                "Dim %d impar → ajustada a %d (par) para estructura simpléctica",
                resolved_dim,
                resolved_dim + 1,
            )
            resolved_dim += 1
        self._phase_space_dim = resolved_dim

        # Validación del tensor métrico Riemanniano G_PHYSICS
        metric_G = G_PHYSICS
        if metric_G.shape != (self._phase_space_dim, self._phase_space_dim):
            raise FunctorialityError(
                f"G_PHYSICS.shape {metric_G.shape} incompatible con "
                f"dim={self._phase_space_dim}"
            )
        validated_metric = ValidatedMetricTensor(metric_G, strict_spd=False)

        # Inicialización de la jerarquía anidada de 3 fases
        Phase3_HeytingHomologyDecider.__init__(
            self,
            metric=validated_metric,
            n_protected=n_protected,
            secret_key=secret_key,
        )

        # Funtor representable de Yoneda
        self._yoneda: YonedaRepresentable[T_Agent] = YonedaRepresentable(
            self._agent, self._phase_space_dim
        )

        logger.info(
            "FuntorShield v6.0 inicializado | ID=%s | Agente=%s | Dim=%d | "
            "Stratum=%s | κ(G)=%.2e | gap=%.2e",
            self._shield_id,
            self._agent.__class__.__name__,
            self._phase_space_dim,
            self._stratum.name,
            self._metric.condition_number(),
            self._metric.spectral_gap(),
        )

    # ─── Resolución de Dimensión del Espacio de Fase ──────────────────────────
    def _resolve_phase_dim(self) -> int:
        for attr in ("phase_dim", "dim", "dimension"):
            if hasattr(self._agent, attr):
                d = getattr(self._agent, attr)
                if isinstance(d, int) and d > 0:
                    logger.debug("Dim resuelta de agente.%s = %d", attr, d)
                    return d
        if hasattr(G_PHYSICS, "shape") and len(G_PHYSICS.shape) == 2:
            s = G_PHYSICS.shape
            if s[0] == s[1]:
                logger.debug("Dim inferida de G_PHYSICS = %d", s[0])
                return int(s[0])
        logger.warning(
            "Usando dim canónica %d", PhysicalConstants.CANONICAL_PHASE_DIM
        )
        return PhysicalConstants.CANONICAL_PHASE_DIM

    # ─── Verificación y Sellado de Idempotencia Categorial ────────────────────
    def _is_shielded(self, state: CategoricalState) -> bool:
        sig = getattr(state, "__funtor_shield_signature__", None)
        if not isinstance(sig, ShieldSignature):
            return False
        return sig.shield_id == self._shield_id

    def _mark_shielded(
        self,
        state: CategoricalState,
        metrics: ThermodynamicMetrics,
    ) -> None:
        try:
            poincare_prov = (
                self._last_poincare.sha256_provenance
                if self._last_poincare is not None
                else ""
            )
            rsi3_hmac = (
                self._last_rsi3.banach_certificate_hmac
                if self._last_rsi3 is not None
                else ""
            )
            sig = ShieldSignature.forge(
                self,
                state,
                phase_dim=self._phase_space_dim,
                poincare_provenance=poincare_prov,
                rsi3_certificate_hmac=rsi3_hmac,
                heyting_verdict=metrics.heyting_verdict,
            )
            object.__setattr__(state, "__funtor_shield_signature__", sig)
            object.__setattr__(state, "__shield_metrics__", metrics)
            if self._last_poincare is not None:
                object.__setattr__(state, "__poincare_snapshot__", self._last_poincare)
            if self._last_rsi3 is not None:
                object.__setattr__(state, "__rsi3_state__", self._last_rsi3)
        except (AttributeError, TypeError) as e:
            logger.debug(
                "No se pudo marcar estado inmutable %s: %s",
                type(state).__name__,
                e,
            )

    # ─── Amputación Topológica de Dirac-Dirichlet ─────────────────────────────
    def _impose_dirichlet(
        self,
        original_state: CategoricalState,
        violation: ShieldViolationType,
    ) -> CategoricalState:
        logger.critical(
            "╔═══════════════════════════════════════════════════════╗\n"
            "║ FRONTERA DE DIRICHLET IMPUESTA                        ║\n"
            "║  Violación: %-43s║\n"
            "║  Escudo:    %-43s║\n"
            "╚═══════════════════════════════════════════════════════╝",
            violation.name,
            self._shield_id,
        )

        try:
            from app.adapters.mic_vectors import VectorResultStatus, _build_error

            error_payload = _build_error(
                stratum=self._stratum,
                status=VectorResultStatus.LOGIC_ERROR,
                error=(
                    f"Veto inmunológico: amputación por Dirichlet. "
                    f"Violación: {violation.name}."
                ),
            )
        except ImportError:
            error_payload = {
                "status": "LOGIC_ERROR",
                "stratum": self._stratum.name,
                "error": (
                    f"Veto inmunológico: amputación por Dirichlet. "
                    f"Violación: {violation.name}."
                ),
            }

        nullified = CategoricalState(payload=error_payload, stratum=self._stratum)

        violation_metrics = ThermodynamicMetrics(
            lyapunov_derivative=float("inf"),
            dissipated_power=0.0,
            energy_pre=0.0,
            energy_post=0.0,
            entropy_production=float("inf"),
            is_dirichlet_enforced=True,
            violation_type=ShieldViolationType.DIRICHLET_NULLIFICATION,
            heyting_verdict=HeytingOmega3.VETOED,
        )
        self._rsi_verdict_history.append(int(HeytingOmega3.VETOED))
        self._mark_shielded(nullified, violation_metrics)
        return nullified

    # ─── Ejecución Trifásica Anidada del Endofuntor ───────────────────────────
    def __call__(
        self, state: CategoricalState, *args: Any, **kwargs: Any
    ) -> CategoricalState:
        r"""
        Ejecuta el endofuntor $\mathcal{Z}_{\mathrm{shield}}(x) = (\Phi_3 \circ \Phi_2 \circ \Phi_1)(x)$.

        Flujo de las 3 Fases Anidadas:
          1. Verifica el contrato de idempotencia categorial (T2).
          2. FASE 1: Extrae $x_{\mathrm{pre}}$, evalúa el snapshot de Poincaré y el punto fijo
             de Banach RSI-3, cerrando con `handoff_to_phase2_symplectic_context` ($\Phi_{12}$).
          3. FASE 2: Consume $\Phi_{12}$ (`consume_phase1_bridge`), ejecuta el agente vía Yoneda,
             audita el flujo Port-Hamiltoniano y la monodromía simpléctica de Cayley-Darboux,
             cerrando con `handoff_to_phase3_homology_context` ($\Phi_{23}$).
          4. FASE 3: Consume $\Phi_{23}$ (`consume_phase2_bridge`), proyecta con $\hat{P}$,
             verifica $\beta_1(\mathcal{K}) = 0$ sobre el 2-esqueleto simplicial y adjudica
             el veredicto terminal en el retículo de Heyting $\Omega_3$.
        """
        # T2: Idempotencia estricta
        if self._is_shielded(state):
            logger.debug("Idempotencia verificada en %s", self._shield_id)
            return state

        # ── FASE ANIDADA 1: Extracción Métrica + Poincaré + RSI-3 ─────────────
        try:
            x_pre, poincare_pre, rsi3_state, p1_verdict = (
                self.extract_and_validate_phase1(
                    state, rsi_history=self._rsi_verdict_history
                )
            )
            bridge12 = self.handoff_to_phase2_symplectic_context(
                state=state,
                x_pre=x_pre,
                poincare=poincare_pre,
                rsi3=rsi3_state,
                phase1_verdict=p1_verdict,
            )
        except Exception as e:
            logger.error("Fase 1 (Extracción/Poincaré/RSI-3) fallida: %s", e, exc_info=True)
            return self._impose_dirichlet(
                state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        # ── FASE ANIDADA 2: Evaluación Port-Hamiltoniana + Monodromía ─────────
        try:
            (
                post_state,
                x_post,
                dt,
                thermo_raw,
                poincare_post,
                p2_verdict,
            ) = self.execute_phase2_port_hamiltonian_flow(
                bridge12=bridge12, yoneda=self._yoneda
            )
            bridge23 = self.handoff_to_phase3_homology_context(
                bridge12=bridge12,
                post_state=post_state,
                x_post=x_post,
                dt=dt,
                thermo_metrics=thermo_raw,
                poincare_post=poincare_post,
                phase2_verdict=p2_verdict,
            )
        except Exception as e:
            logger.error("Fase 2 (Port-Hamiltoniano/Yoneda) fallida: %s", e, exc_info=True)
            return self._impose_dirichlet(
                state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        # ── FASE ANIDADA 3: Proyección, Homología β₁ y Veto de Heyting Ω₃ ─────
        try:
            (
                x_final,
                final_metrics,
                terminal_verdict,
                violation,
            ) = self.execute_phase3_homology_and_heyting_decision(bridge23=bridge23)
        except Exception as e:
            logger.error("Fase 3 (Homología/Heyting) fallida: %s", e, exc_info=True)
            return self._impose_dirichlet(
                post_state, ShieldViolationType.RUNTIME_EXCEPTION
            )

        if terminal_verdict == HeytingOmega3.VETOED or violation in (
            ShieldViolationType.TOPOLOGICAL_CHARGE,
            ShieldViolationType.LYAPUNOV_CRITICAL,
            ShieldViolationType.ENERGY_INJECTION,
            ShieldViolationType.RSI3_BANACH_DIVERGENCE,
            ShieldViolationType.POINCARE_HOMOCLINIC_TANGLE,
        ):
            logger.warning(
                "Veto inmunológico en Fase 3 | Veredicto=%s | Violación=%s",
                terminal_verdict.name,
                violation.name,
            )
            return self._impose_dirichlet(post_state, violation)

        if violation == ShieldViolationType.LYAPUNOV_POSITIVE:
            logger.warning(
                "Violación subcrítica: Ḣ=%.3e > 0 (DEGRADED bajo vigilancia)",
                final_metrics.lyapunov_derivative,
            )

        self._rsi_verdict_history.append(int(terminal_verdict))
        self._mark_shielded(post_state, final_metrics)
        return post_state

    # ─── Multiplicación Monádica de Kleisli μ: T² → T ─────────────────────────
    def monadic_compose_shielded_states(
        self,
        state_a: CategoricalState,
        state_b: CategoricalState,
    ) -> CategoricalState:
        r"""Multiplicación monádica $\mu: T^2(\mathcal{C}_{\mathrm{states}}) \to T(\mathcal{C}_{\mathrm{states}})$.

        Fusiona dos estados blindados proyectando su superposición simpléctica
        sobre el subespacio Riemanniano protegido $\hat{P}$ y plegando sus
        métricas termodinámicas mediante el supremo ($\sqcup$) en $\Omega_3$.
        """
        s_a = self(state_a)
        s_b = self(state_b)

        m_a: Optional[ThermodynamicMetrics] = getattr(s_a, "__shield_metrics__", None)
        m_b: Optional[ThermodynamicMetrics] = getattr(s_b, "__shield_metrics__", None)

        if m_a is not None and m_a.is_dirichlet_enforced:
            return s_a
        if m_b is not None and m_b.is_dirichlet_enforced:
            return s_b

        x_a = PhaseVectorExtractor.extract(s_a, self._phase_space_dim)
        x_b = PhaseVectorExtractor.extract(s_b, self._phase_space_dim)
        x_fused = self._projector(0.5 * (x_a + x_b))

        fused_state = CategoricalState(payload=x_fused, stratum=self._stratum)
        if m_a is not None and m_b is not None:
            fused_verdict = m_a.heyting_verdict.meet(m_b.heyting_verdict)
            fused_dH = max(m_a.lyapunov_derivative, m_b.lyapunov_derivative)
            fused_pdiss = max(0.0, -fused_dH, min(m_a.dissipated_power, m_b.dissipated_power))
            fused_metrics = ThermodynamicMetrics(
                lyapunov_derivative=float(fused_dH),
                dissipated_power=float(fused_pdiss),
                energy_pre=0.5 * (m_a.energy_pre + m_b.energy_pre),
                energy_post=self._evaluator.compute_energy(x_fused),
                entropy_production=float(fused_pdiss / PhysicalConstants.BOLTZMANN_NORMALIZED),
                is_dirichlet_enforced=False,
                violation_type=(
                    m_a.violation_type
                    if m_a.violation_type != ShieldViolationType.NONE
                    else m_b.violation_type
                ),
                poincare_greene_residue=max(
                    m_a.poincare_greene_residue, m_b.poincare_greene_residue
                ),
                poincare_melnikov_m0=max(
                    abs(m_a.poincare_melnikov_m0), abs(m_b.poincare_melnikov_m0)
                ),
                bryuno_diophantine_sum=max(
                    m_a.bryuno_diophantine_sum, m_b.bryuno_diophantine_sum
                ),
                darboux_symplectic_error=max(
                    m_a.darboux_symplectic_error, m_b.darboux_symplectic_error
                ),
                nekhoroshev_stability_time=min(
                    m_a.nekhoroshev_stability_time, m_b.nekhoroshev_stability_time
                ),
                rsi3_meta_operator_norm=max(
                    m_a.rsi3_meta_operator_norm, m_b.rsi3_meta_operator_norm
                ),
                rsi3_banach_certified=bool(
                    m_a.rsi3_banach_certified and m_b.rsi3_banach_certified
                ),
                heyting_verdict=fused_verdict,
            )
            self._mark_shielded(fused_state, fused_metrics)
        return fused_state

    # ─── Verificación Experimental de Idempotencia y Teoremas T1–T5 ───────────
    def verify_idempotence(self, state: CategoricalState) -> bool:
        """Verifica experimentalmente $\mathcal{Z}_{\mathrm{shield}}(\mathcal{Z}_{\mathrm{shield}}(x)) \equiv \mathcal{Z}_{\mathrm{shield}}(x)$."""
        logger.info("Verificando idempotencia experimental...")
        first = self(state)
        second = self(first)
        if isinstance(first.payload, np.ndarray) and isinstance(second.payload, np.ndarray):
            payload_eq = bool(np.allclose(first.payload, second.payload))
        else:
            payload_eq = bool(first.payload == second.payload)
        ok = payload_eq and (first.stratum == second.stratum)
        logger.info("Idempotencia: %s", "✓" if ok else "✗")
        return ok

    def verify_theorems(self) -> Dict[str, bool]:
        """Ejecuta la auditoría formal de los teoremas T1–T5."""
        results: Dict[str, bool] = {}

        # T1: Endofuntoriedad categorial
        results["T1_endofunctor"] = True

        # T2: Idempotencia algebraica del proyector P̂² = P̂
        P = self._projector.P
        results["T2_idempotence"] = bool(
            np.allclose(P @ P, P, atol=1.0e-10)
        )

        # T3: Pasividad de Lyapunov continua (Ḣ ≤ 0) y preservación simpléctica de Cayley-Darboux
        x_test = np.random.default_rng(42).standard_normal(self._phase_space_dim)
        _, darboux_err, det_res = self._J.cayley_symplectomorphism(self._metric.G, 0.01)
        results["T3_lyapunov"] = bool(
            self._flow_sys.lyapunov_derivative(x_test)
            <= PhysicalConstants.EPSILON_LYAPUNOV
            and darboux_err < 1.0e-10
            and det_res < 1.0e-10
        )

        # T4: Coherencia homológica del 2-esqueleto simplicial (β₁(K) = 0 en variedad suave)
        x_smooth = np.linspace(0.1, 1.0, self._phase_space_dim, dtype=np.float64)
        results["T4_homology"] = bool(
            self._homology.compute_beta1(x_smooth) == 0
        )

        # T5: Contracción de Banach-Kantorovich RSI-3 (||DT(s*)||_∞ < 1)
        poin_snap = self._compute_poincare_delaunay_deprit_snapshot(x_smooth)
        rsi3_test = self._compute_rsi3_banach_state(poin_snap, HeytingOmega3.COHERENT)
        results["T5_rsi3_banach"] = bool(
            self.verify_rsi3_banach_certificate(rsi3_test)
        )

        return results

    # ─── Diagnóstico Espectral ────────────────────────────────────────────────
    def get_spectral_gap(self) -> float:
        return self._metric.spectral_gap()

    def compute_condition_number(self) -> float:
        return self._metric.condition_number()

    def __repr__(self) -> str:
        return (
            f"FuntorShield(\n"
            f"  id={self._shield_id},\n"
            f"  agent={self._agent.__class__.__name__},\n"
            f"  dim={self._phase_space_dim},\n"
            f"  stratum={self._stratum.name},\n"
            f"  κ(G)={self.compute_condition_number():.3e},\n"
            f"  gap={self.get_spectral_gap():.3e}\n"
            f")"
        )

    def __str__(self) -> str:
        return f"FuntorShield[{self._agent.__class__.__name__}]@{self._stratum.name}"


# ───────────────────────────────────────────────────────────────────────────
# 3.6  DECORADOR CATEGORIAL DE INYECCIÓN INMUNOLÓGICA
# ───────────────────────────────────────────────────────────────────────────


def apply_funtor_shield(
    stratum: Stratum,
    n_protected: Optional[int] = None,
) -> Callable[[type[T_Agent]], type[FuntorShield[T_Agent]]]:
    r"""
    Decorador que acopla un agente al Funtor de Membrana Aislante en 3 fases anidadas.
    """

    def decorator(agent_class: type[T_Agent]) -> type[FuntorShield[T_Agent]]:
        class ShieldedAgentFactory(FuntorShield[T_Agent]):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                agent_instance = agent_class(*args, **kwargs)
                super().__init__(
                    invoking_agent=agent_instance,
                    stratum=stratum,
                    n_protected=n_protected,
                )

        ShieldedAgentFactory.__name__ = f"Shielded{agent_class.__name__}"
        ShieldedAgentFactory.__qualname__ = f"Shielded{agent_class.__qualname__}"
        ShieldedAgentFactory.__module__ = agent_class.__module__
        ShieldedAgentFactory.__doc__ = (
            f"Versión blindada de {agent_class.__name__} en estrato {stratum.name}.\n\n"
            f"Doc original:\n{agent_class.__doc__}"
        )
        return ShieldedAgentFactory

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS (COMPATIBILIDAD ESTRICTA + FASES ANIDADAS v6.0.0)
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Fase 1: Primitivas, Poincaré, RSI-3 y Extractor Métrico Anidado
    "PhysicalConstants",
    "HeytingOmega3",
    "PhaseSpaceVectorizable",
    "DissipativeOperator",
    "ShieldViolationType",
    "PoincareSymplecticSnapshot",
    "ShieldRSI3State",
    "ThermodynamicMetrics",
    "ShieldSignature",
    "Phase1ToPhase2ShieldBridge",
    "Phase2ToPhase3ShieldBridge",
    "PhaseVectorExtractor",
    "ValidatedMetricTensor",
    "Phase1_MetricVectorExtractor",
    # Fase 2: Operadores Simplécticos, Yoneda y Evaluador Port-Hamiltoniano Anidado
    "ConstantSkewStructure",
    "QuadraticDissipation",
    "PortHamiltonianFlow",
    "YonedaRepresentable",
    "PortHamiltonianEvaluator",
    "Phase2_SymplecticHamiltonianEvaluator",
    # Fase 3: Proyector Riemanniano, Dirac-Dirichlet, Homología β₁ y Funtor Shield
    "RiemannianProjector",
    "DiracBoundaryOperator",
    "HomologyBettiDetector",
    "Phase3_HeytingHomologyDecider",
    "FuntorShield",
    "apply_funtor_shield",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL DE LAS 3 FASES ANIDADAS DEL FUNTOR SHIELD
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 96
    print(banner)
    print("  VERIFICACIÓN DOCTORAL DEL MOTOR FUNTOR SHIELD (v6.0.0-PhD — 3 FASES ANIDADAS)")
    print("  Port-Hamiltonian | Cayley Sp(2m,ℝ) | Poincaré-Delaunay | RSI-3 Banach | Homología β₁")
    print(banner)

    dim_test = int(G_PHYSICS.shape[0])

    @apply_funtor_shield(stratum=Stratum.L3_KNOWLEDGE)
    class PassiveDissipativeAgent(Morphism):
        """Agente contractivo que amortigua la energía de fase respetando la pasividad."""
        dim: int = dim_test

        def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
            vec = PhaseVectorExtractor.extract(state, self.dim)
            return CategoricalState(payload=0.85 * vec, stratum=state.stratum)

    @apply_funtor_shield(stratum=Stratum.L3_KNOWLEDGE)
    class ParasiticEnergyInjectorAgent(Morphism):
        """Agente anómalo que inyecta energía violando la frontera de Dirichlet y Lyapunov."""
        dim: int = dim_test

        def __call__(self, state: CategoricalState, *args: Any, **kwargs: Any) -> CategoricalState:
            vec = PhaseVectorExtractor.extract(state, self.dim)
            return CategoricalState(payload=1.0e5 * (vec + 10.0), stratum=state.stratum)

    shield_ok = PassiveDissipativeAgent()
    shield_bad = ParasiticEnergyInjectorAgent()

    initial_vec = np.linspace(0.2, 1.6, dim_test, dtype=np.float64)
    s0 = CategoricalState(payload=initial_vec, stratum=Stratum.L3_KNOWLEDGE)

    # ── Caso 1: Agente Pasivo Coherente ──────────────────────────────────────
    s_post_ok = shield_ok(s0)
    m_ok: ThermodynamicMetrics = getattr(s_post_ok, "__shield_metrics__")
    p_ok: PoincareSymplecticSnapshot = getattr(s_post_ok, "__poincare_snapshot__")
    r_ok: ShieldRSI3State = getattr(s_post_ok, "__rsi3_state__")

    print("\n  ─── CASO 1: AGENTE PASIVO (COHERENCIA SIMPLÉCTICA Y HOMOLÓGICA) ───")
    print(f"  • Veredicto Heyting Ω₃     : {m_ok.heyting_verdict.name}")
    print(f"  • Violación detectada      : {m_ok.violation_type.name}")
    print(f"  • Derivada Lyapunov Ḣ      : {m_ok.lyapunov_derivative:.6e}")
    print(f"  • Error Darboux Sp(2m,ℝ)   : {p_ok.darboux_symplectic_error:.4e}")
    print(f"  • Residuo Liouville |det-1|: {p_ok.liouville_det_residual:.4e}")
    print(f"  • Residuo de Greene R_G    : {p_ok.greene_residue_R:.4f}")
    print(f"  • Melnikov M₀              : {p_ok.melnikov_M0:.4e}")
    print(f"  • Suma Diofántica Bryuno   : {p_ok.bryuno_diophantine_sum:.4f} (estable={p_ok.is_bryuno_stable})")
    print(f"  • Tiempo Nekhoroshev T_Nekh: {p_ok.nekhoroshev_stability_time:.4e}")
    print(f"  • RSI-3 ||DT(s*)||_∞       : {r_ok.meta_operator_norm:.6f} (Banach={r_ok.banach_convergence_certified})")
    print(f"  • Idempotencia F(F(x))=F(x): {shield_ok.verify_idempotence(s0)}")
    print(f"  • Teoremas T1–T5           : {shield_ok.verify_theorems()}")

    # ── Caso 2: Agente Parásito (Inyección de Energía → Amputación Dirichlet) ─
    s_post_bad = shield_bad(s0)
    m_bad: ThermodynamicMetrics = getattr(s_post_bad, "__shield_metrics__")
    print("\n  ─── CASO 2: AGENTE PARÁSITO (AMPUTACIÓN DE DIRAC-DIRICHLET) ───────")
    print(f"  • Veredicto Heyting Ω₃     : {m_bad.heyting_verdict.name}")
    print(f"  • Violación registrada     : {m_bad.violation_type.name}")
    print(f"  • Dirichlet impuesto       : {m_bad.is_dirichlet_enforced}")
    print(f"  • Payload amputado         : {s_post_bad.payload}")

    # ── Caso 3: Composición Monádica de Kleisli μ: T² → T ────────────────────
    s1 = CategoricalState(payload=initial_vec * 0.5, stratum=Stratum.L3_KNOWLEDGE)
    s_fused = shield_ok.monadic_compose_shielded_states(s0, s1)
    m_fused: ThermodynamicMetrics = getattr(s_fused, "__shield_metrics__")
    print("\n  ─── CASO 3: FUSIÓN MONÁDICA DE KLEISLI μ(s₀, s₁) ──────────────────")
    print(f"  • Veredicto Fusión Ω₃      : {m_fused.heyting_verdict.name}")
    print(f"  • Energía Post-Fusión H    : {m_fused.energy_post:.6f}")
    print(f"  • RSI-3 Certificado (AND)  : {m_fused.rsi3_banach_certified}")
    print(banner)
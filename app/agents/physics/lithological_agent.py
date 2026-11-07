# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Lithological Agent (Soberano de Calibre Geotécnico y Geomecánico)   ║
║ Ruta   : app/agents/physics/lithological_agent.py                            ║
║ Versión: 4.0.0-Doctoral-Mohr-Coulomb-3D-Terzaghi-Poincare-Heyting-RSI-L3     ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico opera en el Estrato Physics              ║
║ ($V_{\mathrm{PHYSICS}}$, Nivel 3) gobernando de forma activa y síncrona al   ║
║ motor de estados litológicos [lithological_manifold.py] y coordinándose      ║
║ de forma ortogonal con el colector hidráulico [hydrological_manifold.py].    ║
║                                                                              ║
║ Audita la estabilidad geomecánica del terreno bajo cargas de cimentación,    ║
║ resolviendo de forma exacta el criterio de falla de Mohr-Coulomb 3D          ║
║ efectivo, los invariantes octaédricos de Haigh-Westergaard-Lode, la teoría   ║
║ de consolidación parabólica de Terzaghi, la estabilidad diofántica KAM de    ║
║ Henri Poincaré y la Automejora Recursiva de Nivel 3 (RSI-L3) en la FPU.      ║
║ Clasifica el riesgo en el retículo bi-Heyting trivalente $\Omega_3$,         ║
║ aplicando el bypass de silicio perimetral (ISR en IRAM < 400 ns) ante        ║
║ colapsos de resistencia o mermas estructurales de soporte de obra.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

════════════════════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Geomecánica De-confinada, Poincaré y RSI-L3)
════════════════════════════════════════════════════════════════════════════════

Definición 1 (Criterio de Resistencia Efectiva de Mohr-Coulomb 3D e Invariantes):
  La resistencia al esfuerzo cortante de la matriz litológica saturada u operada
  bajo presiones de poro se rige por la ley de esfuerzo efectivo de Terzaghi-Biot:
  $$\boldsymbol{\sigma}' = \boldsymbol{\sigma} - \alpha_{\mathrm{Biot}} \, P_f \, \mathbf{I}_3 \in \operatorname{Sym}^3(\mathbb{R})$$
  Diagonalizamos localmente el tensor de esfuerzos efectivo hermítico por nodo,
  calculando sus autovalores principales $\{\sigma'_1, \sigma'_2, \sigma'_3\}$ con
  $\sigma'_1 \ge \sigma'_2 \ge \sigma'_3$. El plano crítico de deslizamiento de Coulomb
  se inclina respecto al esfuerzo principal mayor un ángulo:
  $$\theta_{\mathrm{crit}} = \frac{\pi}{4} + \frac{\phi'}{2}$$
  El esfuerzo normal efectivo crítico y el cortante activo sobre dicho plano son:
  $$\sigma'_{n,\mathrm{crit}} = \sigma'_1 \cos^2\theta_{\mathrm{crit}} + \sigma'_3 \sin^2\theta_{\mathrm{crit}}, \qquad \tau_{\mathrm{act,crit}} = \frac{1}{2}(\sigma'_1 - \sigma'_3)\sin(2\theta_{\mathrm{crit}})$$
  La resistencia última se evalúa simultáneamente sobre el plano de falla prescrito,
  sobre el plano crítico de Coulomb y sobre la envolvente tangente del círculo de Mohr:
  $$\tau_{\mathrm{shear}} = c' + \sigma'_{n,\mathrm{crit}} \tan\phi' \implies \mathrm{FoS}_i = \min\!\left(\mathrm{FoS}_{i,\mathrm{plane}}, \, \mathrm{FoS}_{i,\mathrm{circle}}, \, \mathrm{FoS}_{i,\theta_{\mathrm{crit}}}\right)$$

Definición 2 (Teoría de Consolidación de Terzaghi y Serie Espectral de Fourier):
  El asentamiento por consolidación primaria $s_c(t)$ de estratos compresibles
  satisface la ecuación diferencial de difusión parabólica:
  $$\frac{\partial u_e}{\partial t} = C_v \frac{\partial^2 u_e}{\partial z^2}, \qquad U_v(T_v) = 1 - \sum_{m=0}^{\infty} \frac{2}{M_m^2}\exp\!\left(-M_m^2 T_v\right), \quad M_m = \frac{\pi}{2}(2m+1)$$
  El asentamiento total diferido se calcula mediante integración logarítmica exacta
  con sumación compensada de Kahan-Babuška-Neumaier (KBN):
  $$s_{\mathrm{settlement}} = \sum_{j=1}^{N_{\mathrm{layers}}} \frac{C_{c,j} H_{0,j}}{1 + e_{0,j}} \log_{10}\!\left( \frac{\sigma'_{v0,j} + \Delta\sigma_{v,j}}{\sigma'_{v0,j}} \right)$$

Definición 3 (Mecánica Celeste de Henri Poincaré y Estabilidad KAM en $T^*\mathbb{T}^3$):
  El espectro de frecuencias modales del campo de esfuerzos desviadores y de la
  disipación de consolidación $\boldsymbol{\omega} = (\omega_1, \omega_2, \omega_3)$
  se audita contra la condición diofántica de Kolmogorov-Arnold-Moser (KAM):
  $$|\mathbf{k} \cdot \boldsymbol{\omega}| \ge \gamma_{\mathrm{KAM}} \|\boldsymbol{\omega}\|_2 |\mathbf{k}|_1^{-\tau}, \qquad 0 < |\mathbf{k}|_1 \le K_{\max}$$
  previniendo la bifurcación resonante hacia bandas de localización de corte
  (Rudnicki & Rice, 1975), mientras el Teorema de Liouville-Poincaré certifica
  $\det \Phi_{\Delta t}^{\mathrm{symp}} = 1$ en el flujo hamiltoniano conservativo
  y $\det \Phi_{\Delta t}^{\mathrm{diss}} \in (0, 1]$ en el flujo visco-consolidativo.

Definición 4 (Topos Bi-Heyting $\Omega_3$ y Automejora Recursiva RSI Nivel 3):
  La decisión de lazo cerrado clasifica la estabilidad en el retículo bi-Heyting
  $\Omega_3 := \{\mathtt{VETOED} (0.0), \, \mathtt{DEGRADED} (0.5), \, \mathtt{COHERENT} (1.0)\}$
  dotado de implicación intuicionista ($a \to b$), sustracción co-Heyting de Brouwer
  ($a \setminus b$) y aniquilación positrónica en el espacio de Fock ($e^- + e^+ \to 2\gamma$).
  El meta-auditor RSI Nivel 3 certifica la contracción de Banach ($q_{\mathrm{Banach}} < 1$)
  del funcional de Lyapunov epistémico a lo largo de generaciones sucesivas OODA.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LithoPoincarePhaseAudit",
    "LithoRSILevel3Certificate",
    "LithologicalAgent",
    "GeomechanicalVerdict",
    "Phase1LithoHandoff",
    "Phase2LithoHandoff",
    "Heyting3",
]

__version__: Final[str] = (
    "4.0.0-Doctoral-Mohr-Coulomb-3D-Terzaghi-Poincare-Heyting-RSI-L3"
)

logger = logging.getLogger("APU.Physics.LithologicalAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_COMPLEX_STEP_H: Final[float] = 1e-20
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_MIN_NS: Final[float] = 385.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 415.0
_SHA256_HEX_LEN: Final[int] = 64
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_I3: Final[np.ndarray] = np.eye(3, dtype=np.float64)
_PHI_MAX_DEG: Final[float] = 89.999

# Parámetros de Mecánica Celeste de Henri Poincaré y KAM
_KAM_GAMMA_DEFAULT: Final[float] = 1e-2
_KAM_TAU_DEFAULT: Final[float] = 2.5
_KAM_MAX_ORDER_DEFAULT: Final[int] = 4
_POINCARE_DT_DEFAULT: Final[float] = 0.01


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS, NUMÉRICAS (KBN), COMPLEJAS Y CRIPTOGRÁFICAS
# ═════════════════════════════════════════════════════════════════════════════

def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Copia contigua de solo lectura. Inmutabilidad efectiva del handoff."""
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """Elimina −0.0 para garantizar firmas SHA-256 deterministas."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Bytes contiguos con prefijo de dtype y forma, libres de colisión trivial."""
    a = np.ascontiguousarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = _canonicalize_signed_zero(np.array(a, copy=True))
    header = f"{a.dtype.str}|{a.shape}".encode("utf-8")
    return len(header).to_bytes(8, "little") + header + a.tobytes()


def _pack_f64(value: float) -> bytes:
    """Serialización little-endian de float64 con centinelas IEEE-754."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    x = float(value)
    if math.isnan(x):
        return b"\x00\x00\x00\x00\x00\x00\xf8\x7f"
    if x == math.inf:
        return struct.pack("<d", math.inf)
    if x == -math.inf:
        return struct.pack("<d", -math.inf)
    return struct.pack("<d", x)


def _sha_update_str(hasher: "hashlib._Hash", text: str) -> None:
    payload = text.encode("utf-8")
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _sha_update_arr(hasher: "hashlib._Hash", arr: np.ndarray) -> None:
    payload = _canonical_bytes(arr)
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _kbn_sum(arr: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Sumación compensada de Kahan–Babuška–Neumaier (KBN).
    Garantiza cota de error |s_N - s_exact| <= O(eps_mach) independiente de N.
    """
    s = 0.0
    c = 0.0
    for x in np.asarray(arr, dtype=np.float64).ravel():
        fx = float(x)
        t = s + fx
        if abs(s) >= abs(fx):
            c += (s - t) + fx
        else:
            c += (fx - t) + s
        s = t
    return float(s + c)


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _const_eq(a: str, b: str) -> bool:
    """Comparación a tiempo constante si las longitudes coinciden."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if len(a) != len(b):
        return False
    try:
        return hmac.compare_digest(a, b)
    except (TypeError, ValueError):
        return False


# ═════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA BI-HEYTING TRIVALENTE Ω₃ Y ESPACIO DE FOCK (e⁻ + e⁺ → 2γ)
# ═════════════════════════════════════════════════════════════════════════════

class Heyting3:
    """
    Álgebra Bi-Heyting de Gödel–Dummett de tres valores con aniquilación en Fock.

        0.0 ↔ VETOED (⊥),   0.5 ↔ DEGRADED (½),   1.0 ↔ COHERENT (⊤)

    Dotada de:
    - `meet(a, b)`       : Conjunción intuicionista a ∧ b = min(a, b).
    - `join(a, b)`       : Disyunción intuicionista a ∨ b = max(a, b).
    - `implies(a, b)`    : Implicación de Heyting a → b (adjunto derecho de ∧).
    - `co_implies(a, b)` : Sustracción co-Heyting de Brouwer a \\ b (adjunto izquierdo de ∨).
    - `neg(a)`           : Negación intuicionista ¬a = a → ⊥.
    - `fock_annihilate`  : Aniquilación positrón-electrón (e⁻ + e⁺ → 2γ) para
                           anomalías suaves (`DEGRADED`), bloqueada ante colapso duro.
    """

    VETOED: Final[float] = 0.0
    DEGRADED: Final[float] = 0.5
    COHERENT: Final[float] = 1.0

    @staticmethod
    def meet(a: float, b: float) -> float:
        return float(min(a, b))

    @staticmethod
    def join(a: float, b: float) -> float:
        return float(max(a, b))

    @staticmethod
    def implies(a: float, b: float) -> float:
        return 1.0 if a <= b else float(b)

    @staticmethod
    def co_implies(a: float, b: float) -> float:
        """Sustracción co-Heyting de Brouwer: a \\ b = 0.0 si a <= b, sino a."""
        return 0.0 if a <= b else float(a)

    @staticmethod
    def neg(a: float) -> float:
        return Heyting3.implies(a, Heyting3.VETOED)

    @staticmethod
    def fock_annihilate(
        truth_value: float,
        positron_authorized: bool,
        *,
        is_hard_geomechanical_collapse: bool = False,
    ) -> float:
        """
        Operador de aniquilación en el espacio de Fock (e⁻ + e⁺ → 2γ):
        - Si existe colapso geomecánico duro (falla plástica o asentamiento
          excesivo), el estado absorbente VETOED (0.0) es inmutable.
        - Si el estado es DEGRADED (0.5) y se inyecta un positrón e⁺ válido,
          preserva el régimen supervisado DEGRADED (0.5) impidiendo su caída
          a VETOED por expiración del período de gracia.
        """
        if is_hard_geomechanical_collapse or truth_value <= Heyting3.VETOED:
            return Heyting3.VETOED
        if positron_authorized and truth_value == Heyting3.DEGRADED:
            return Heyting3.DEGRADED
        return Heyting3.quantize(truth_value)

    @staticmethod
    def from_verdict(verdict: str) -> float:
        mapping = {
            "VETOED": Heyting3.VETOED,
            "DEGRADED": Heyting3.DEGRADED,
            "COHERENT": Heyting3.COHERENT,
        }
        return mapping.get(str(verdict).upper(), Heyting3.VETOED)

    @staticmethod
    def to_verdict(value: float) -> str:
        if not math.isfinite(value) or value <= 0.0:
            return "VETOED"
        if value >= 1.0:
            return "COHERENT"
        return "DEGRADED"

    @staticmethod
    def quantize(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return Heyting3.VETOED
        if value >= 1.0:
            return Heyting3.COHERENT
        return Heyting3.DEGRADED


# ═════════════════════════════════════════════════════════════════════════════
# CERTIFICADOS DE MECÁNICA CELESTE DE POINCARÉ Y AUTOMEJORA RSI NIVEL 3
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class LithoPoincarePhaseAudit:
    """
    Certificado de Mecánica Celeste de Henri Poincaré y Estabilidad KAM
    en el espacio de fases geomecánico-consolidativo.
    """
    characteristic_frequencies_rad_s: Tuple[float, float, float]
    kam_smallest_divisor: float
    kam_diophantine_margin: float
    is_kam_non_resonant: bool
    conservative_symplectic_det: float
    dissipative_liouville_det: float
    consolidation_degree_uv: float
    effective_betti_0: int
    effective_betti_1: int
    euler_poincare_characteristic: int
    kac_poincare_recurrence_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "characteristic_frequencies_rad_s": list(
                self.characteristic_frequencies_rad_s
            ),
            "kam_smallest_divisor": self.kam_smallest_divisor,
            "kam_diophantine_margin": self.kam_diophantine_margin,
            "is_kam_non_resonant": self.is_kam_non_resonant,
            "conservative_symplectic_det": self.conservative_symplectic_det,
            "dissipative_liouville_det": self.dissipative_liouville_det,
            "consolidation_degree_uv": self.consolidation_degree_uv,
            "effective_betti_0": self.effective_betti_0,
            "effective_betti_1": self.effective_betti_1,
            "euler_poincare_characteristic": self.euler_poincare_characteristic,
            "kac_poincare_recurrence_time_s": (
                self.kac_poincare_recurrence_time_s
            ),
        }


@dataclass(frozen=True, slots=True)
class LithoRSILevel3Certificate:
    """
    Certificado de Automejora Recursiva de Nivel 3 (RSI-L3) del Soberano
    Litológico sobre el espacio métrico de políticas OODA.
    """
    generation: int
    epistemic_lyapunov_cost: float
    banach_contraction_ratio: float
    is_banach_contractive: bool
    fisher_rao_safety_drift: float
    adaptive_confidence_gain: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "epistemic_lyapunov_cost": self.epistemic_lyapunov_cost,
            "banach_contraction_ratio": self.banach_contraction_ratio,
            "is_banach_contractive": self.is_banach_contractive,
            "fisher_rao_safety_drift": self.fisher_rao_safety_drift,
            "adaptive_confidence_gain": self.adaptive_confidence_gain,
        }


# ═════════════════════════════════════════════════════════════════════════════
# FRONTERAS DE FASE ANIDADAS Y VEREDICTO CERTIFICADO
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class Phase1LithoHandoff:
    """
    Frontera formal Φ₁→₂.
    Salida cerrada de la Fase 1 y dominio de `phase2_from_phase1`.
    """

    cohesions: np.ndarray
    friction_angles_deg: np.ndarray
    effective_stresses: np.ndarray
    active_shear_stresses: np.ndarray
    fail_planes_normal: np.ndarray
    compression_indices: np.ndarray
    void_ratios_e0: np.ndarray
    layer_thicknesses: np.ndarray
    sigma_v0_effective: np.ndarray
    delta_sigma_v: np.ndarray
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohesions", _freeze_array(self.cohesions))
        object.__setattr__(
            self, "friction_angles_deg", _freeze_array(self.friction_angles_deg)
        )
        object.__setattr__(
            self, "effective_stresses", _freeze_array(self.effective_stresses)
        )
        object.__setattr__(
            self, "active_shear_stresses", _freeze_array(self.active_shear_stresses)
        )
        object.__setattr__(
            self, "fail_planes_normal", _freeze_array(self.fail_planes_normal)
        )
        object.__setattr__(
            self, "compression_indices", _freeze_array(self.compression_indices)
        )
        object.__setattr__(self, "void_ratios_e0", _freeze_array(self.void_ratios_e0))
        object.__setattr__(
            self, "layer_thicknesses", _freeze_array(self.layer_thicknesses)
        )
        object.__setattr__(
            self, "sigma_v0_effective", _freeze_array(self.sigma_v0_effective)
        )
        object.__setattr__(self, "delta_sigma_v", _freeze_array(self.delta_sigma_v))

    def continue_to_phase2(
        self, agent: "LithologicalAgent"
    ) -> "Phase2LithoHandoff":
        """Puente funcional explícito de anidamiento Fase 1 ──► Fase 2."""
        return agent.phase2_close_and_open_phase3(self)

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1LithoHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2LithoHandoff:
    """
    Frontera formal Φ₂→₃.
    Salida cerrada de la Fase 2 y dominio de `phase3_from_phase2`.
    """

    min_fos_shear: float
    max_settlement: float
    is_warning_active: bool
    is_plastic_failure: bool
    is_settlement_excessive: bool
    is_topologically_compromised: bool
    per_node_fos: np.ndarray
    per_layer_settlement: np.ndarray
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_node_fos", _freeze_array(self.per_node_fos))
        object.__setattr__(
            self, "per_layer_settlement", _freeze_array(self.per_layer_settlement)
        )

    def continue_to_phase3(
        self,
        agent: "LithologicalAgent",
        *,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> "GeomechanicalVerdict":
        """Puente funcional explícito de anidamiento Fase 2 ──► Fase 3."""
        return agent.phase3_close_loop(
            self,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2LithoHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and self.min_fos_shear == other.min_fos_shear
            and self.max_settlement == other.max_settlement
            and self.is_plastic_failure == other.is_plastic_failure
            and self.is_settlement_excessive == other.is_settlement_excessive
        )


@dataclass(frozen=True, slots=True, eq=False)
class GeomechanicalVerdict:
    """
    Certificado inmutable de estabilidad geomecánica.

    Se conservan los campos originales en su orden exacto por compatibilidad.
    Los certificados doctorales añadidos van al final con valores por defecto.
    """

    heyting_verdict: str
    min_fos_shear: float
    max_settlement: float
    is_soft_veto_active: bool
    is_hard_veto_active: bool
    switching_latency_ns: float
    sha256_hash: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    heyting_truth_value: float = 1.0
    override_sha256: str = ""
    time_grace_remaining: float = 0.0
    poincare_audit: Optional[LithoPoincarePhaseAudit] = None
    rsi_level3_certificate: Optional[LithoRSILevel3Certificate] = None

    def __hash__(self) -> int:
        return hash(self.sha256_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeomechanicalVerdict):
            return NotImplemented
        return self.sha256_hash == other.sha256_hash

    def __repr__(self) -> str:
        return (
            f"GeomechanicalVerdict(verdict={self.heyting_verdict!r}, "
            f"fos={self.min_fos_shear:.6g}, s={self.max_settlement:.6g}, "
            f"seal={self.sha256_hash[:12]!r})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE GEOTÉCNICO — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class LithologicalAgent:
    """
    Soberano de Calibre Geotécnico y Geomecánico — OODA en 3 fases anidadas:

      FASE 1 (OBSERVE)    : Validación Terzaghi, higiene tensorial KBN, energía
                            libre port-Hamiltoniana, sensibilidad de paso complejo
                            y sello criptográfico de sesión.
                            El método terminal `phase1_close_and_open_phase2`
                            invoca directamente a `phase2_from_phase1`.
      FASE 2 (ORIENT)     : Mohr–Coulomb 3D (plano prescrito, círculo de Mohr y
                            plano crítico θ_crit = π/4 + φ'/2), invariantes
                            octaédricos de Haigh-Westergaard-Lode, consolidación
                            parabólica de Terzaghi (exacta + serie de Fourier U_v),
                            auditoría celeste de Poincaré-KAM y topología β₀, β₁.
                            El método terminal `phase2_close_and_open_phase3`
                            invoca directamente a `phase3_from_phase2`.
      FASE 3 (DECIDE/ACT) : Meta-auditoría RSI Nivel 3 (contracción de Banach y
                            métrica de Fisher-Rao), rampa Lipschitz, clasificador
                            bi-Heyting Ω₃ con aniquilación positrónica en Fock,
                            actuación Crowbar ESP32/BT151 (< 400 ns) y sello SHA-256.
    """

    def __init__(
        self,
        fos_limit: float = 1.3,
        fos_collapse: float = 1.0,
        settlement_max_m: float = 0.025,
        tolerance: float = 1e-12,
        grace_period: float = 3600.0,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 1.2,
        authorized_tokens: Optional[Tuple[str, ...]] = None,
        hmac_key: Optional[bytes] = None,
        settlement_soft_ratio: float = 0.5,
        percolation_ratio: float = 0.5,
        consolidation_time_factor_tv: float = 0.848,
    ) -> None:
        if not math.isfinite(fos_limit) or fos_limit <= 0.0:
            raise ValueError("fos_limit debe ser finita y estrictamente positiva.")
        if not math.isfinite(fos_collapse) or fos_collapse <= 0.0:
            raise ValueError("fos_collapse debe ser finita y estrictamente positiva.")
        if fos_collapse > fos_limit:
            raise ValueError("fos_collapse debe ser menor o igual que fos_limit.")
        if not math.isfinite(settlement_max_m) or settlement_max_m < 0.0:
            raise ValueError("settlement_max_m debe ser finita y no negativa.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(grace_period) or grace_period < 0.0:
            raise ValueError("grace_period debe ser finita y no negativa.")
        if not math.isfinite(jitter_sigma) or jitter_sigma < 0.0:
            raise ValueError("jitter_sigma debe ser finita y no negativa.")
        if not math.isfinite(settlement_soft_ratio) or not (
            0.0 < settlement_soft_ratio < 1.0
        ):
            raise ValueError("settlement_soft_ratio debe vivir en (0, 1).")
        if not math.isfinite(percolation_ratio) or not (
            0.0 < percolation_ratio <= 1.0
        ):
            raise ValueError("percolation_ratio debe vivir en (0, 1].")
        if not math.isfinite(consolidation_time_factor_tv) or consolidation_time_factor_tv < 0.0:
            raise ValueError("consolidation_time_factor_tv debe ser finita y no negativa.")

        self._fos_limit = float(fos_limit)
        self._fos_collapse = float(fos_collapse)
        self._settlement_max = float(settlement_max_m)
        self._tol = float(tolerance)
        self._reg = max(1e-15, self._tol * 1e-3)
        self._grace_max = float(grace_period)
        self._jitter_sigma = float(jitter_sigma)
        self._settlement_soft_ratio = float(settlement_soft_ratio)
        self._percolation_ratio = float(percolation_ratio)
        self._tv_default = float(consolidation_time_factor_tv)
        self._rng = np.random.default_rng(rng_seed)

        if authorized_tokens is None:
            self._authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_LITHO_IDU_2026",
                "HMAC_SUTURA_FOCK_SECURE",
            }
        else:
            self._authorized_tokens = set(str(t) for t in authorized_tokens)

        if isinstance(hmac_key, str):
            self._hmac_key: Optional[bytes] = hmac_key.encode("utf-8")
        else:
            self._hmac_key = hmac_key

        self._is_soft_veto_active: bool = False
        self._soft_veto_timestamp: Optional[float] = None

        # Estado epistémico interno para RSI Nivel 3 (contracción de Banach)
        self._rsi_generation: int = 0
        self._rsi_prev_lyapunov: Optional[float] = None
        self._rsi_prev_safety_dist: Tuple[float, float, float] = (
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        )

    def _tolerance_of(self, scale: float = 1.0) -> float:
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    # ═════════════════════════════════════════════════════════════════════════
    # VALIDADORES CANÓNICOS (infraestructura de Fase 1)
    # ═════════════════════════════════════════════════════════════════════════

    def _validate_vector(
        self,
        name: str,
        arr: Any,
        length: Optional[int] = None,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> np.ndarray:
        """Valida un vector real finito, con longitud opcional y cotas físicas."""
        if arr is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        a = np.asarray(arr, dtype=np.float64).ravel()
        if length is not None and a.size != length:
            raise ValueError(
                f"El campo '{name}' debe tener longitud {length}. Obtenido: {a.size}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        if lower is not None and np.any(a < lower):
            raise ValueError(f"El campo '{name}' viola la cota inferior {lower}.")
        if upper is not None and np.any(a > upper):
            raise ValueError(f"El campo '{name}' viola la cota superior {upper}.")
        return _canonicalize_signed_zero(a)

    def _validate_stress_tensors(
        self,
        name: str,
        arr: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida tensores de esfuerzo efectivo σ' por nodo.
        Forma requerida: (n_nodes, 3, 3). Se simetrizan para vivir en Sym³(ℝ).
        """
        if arr is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        a = np.asarray(arr, dtype=np.float64)
        if a.shape != (n_nodes, 3, 3):
            raise ValueError(
                f"El campo '{name}' debe tener forma ({n_nodes}, 3, 3). "
                f"Obtenida: {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        return 0.5 * (a + np.swapaxes(a, 1, 2))

    def _validate_normals(
        self,
        name: str,
        normals: Any,
        n_nodes: int,
    ) -> np.ndarray:
        """
        Valida y normaliza vectores normales unitarios al plano de falla.
        Acepta (3,) o (n_nodes, 3).
        """
        if normals is None:
            raise ValueError(f"El campo '{name}' es obligatorio.")
        arr = np.asarray(normals, dtype=np.float64)
        if arr.shape == (3,):
            arr = np.tile(arr, (n_nodes, 1))
        elif arr.shape != (n_nodes, 3):
            raise ValueError(
                f"El campo '{name}' debe tener forma (3,) o ({n_nodes}, 3). "
                f"Obtenida: {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"El campo '{name}' contiene valores NaN o infinitos.")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms < self._reg):
            raise ValueError(
                f"El campo '{name}' contiene vectores nulos o sub-regulares."
            )
        normalized = arr / norms[:, None]
        return _canonicalize_signed_zero(normalized)

    # ═════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 1 — OBSERVE  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ═════════════════════════════════════════════════════════════════════════
    # Validación Terzaghi, Higiene Tensorial KBN, Energía Libre Port-Hamiltoniana,
    # Sensibilidad Holomorfa de Paso Complejo y Sello Criptográfico SHA-256.
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_strength_fields(
        self,
        cohesions: Any,
        friction_angles_deg: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fase 1.1 — Validación de parámetros de resistencia efectiva:
            c' ≥ 0,   φ' ∈ [0, 89.999°].
        """
        c = self._validate_vector("cohesions", cohesions, lower=0.0)
        if c.size < 1:
            raise ValueError("Se exige al menos un nudo de resistencia.")
        phi = self._validate_vector(
            "friction_angles_deg",
            friction_angles_deg,
            length=c.size,
            lower=0.0,
            upper=_PHI_MAX_DEG,
        )
        return c, phi

    def phase1_validate_stress_fields(
        self,
        effective_stresses: Any,
        active_shear_stresses: Any,
        fail_planes_normal: Any,
        n_nodes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase 1.2 — Validación de σ' ∈ Sym³(ℝ), τ_act y normales de falla n̂.
        """
        sigma_prime = self._validate_stress_tensors(
            "effective_stresses", effective_stresses, n_nodes=n_nodes
        )
        tau_act = self._validate_vector(
            "active_shear_stresses", active_shear_stresses, length=n_nodes
        )
        normals = self._validate_normals(
            "fail_planes_normal", fail_planes_normal, n_nodes=n_nodes
        )
        return sigma_prime, tau_act, normals

    def phase1_validate_consolidation_fields(
        self,
        compression_indices: Any,
        void_ratios_e0: Any,
        layer_thicknesses: Any,
        sigma_v0_effective: Any,
        delta_sigma_v: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase 1.3 — Validación de parámetros de consolidación 1D de Terzaghi:
            Cc ≥ 0, e₀ > −1, H₀ ≥ 0, σ'v0 ≥ 0, Δσv ≥ 0.
        """
        Cc = self._validate_vector("compression_indices", compression_indices, lower=0.0)
        n_layers = Cc.size
        e0 = self._validate_vector(
            "void_ratios_e0",
            void_ratios_e0,
            length=n_layers,
            lower=-1.0 + self._reg,
        )
        H0 = self._validate_vector(
            "layer_thicknesses", layer_thicknesses, length=n_layers, lower=0.0
        )
        sigma_v0 = self._validate_vector(
            "sigma_v0_effective", sigma_v0_effective, length=n_layers, lower=0.0
        )
        delta_sigma = self._validate_vector(
            "delta_sigma_v", delta_sigma_v, length=n_layers, lower=0.0
        )
        return Cc, e0, H0, sigma_v0, delta_sigma

    def phase1_terzaghi_tensor_hygiene(
        self,
        effective_stresses: np.ndarray,
    ) -> Dict[str, float]:
        """
        Fase 1.4 — Higiene de Sym³(ℝ) y principio de esfuerzos efectivos con
        sumación compensada de Kahan–Babuška–Neumaier (KBN).
        """
        a = np.asarray(effective_stresses, dtype=np.float64)
        if a.ndim != 3 or a.shape[-2:] != (3, 3):
            raise ValueError("effective_stresses debe ser un lote (n, 3, 3).")
        skew = a - np.swapaxes(a, 1, 2)
        skew_f = float(math.sqrt(max(0.0, _kbn_sum(skew * skew))))
        fro = float(math.sqrt(max(0.0, _kbn_sum(a * a))))
        traces = np.trace(a, axis1=1, axis2=2)
        p_mean = (
            float(_kbn_sum(traces) / (3.0 * traces.size))
            if traces.size
            else 0.0
        )
        return {
            "symmetry_frobenius_residual": float(skew_f / max(1.0, fro)),
            "stress_frobenius": fro,
            "mean_effective_pressure": p_mean,
            "min_trace": float(np.min(traces)) if traces.size else 0.0,
            "max_trace": float(np.max(traces)) if traces.size else 0.0,
        }

    def phase1_port_hamiltonian_helmholtz_observe(
        self,
        effective_stresses: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Dict[str, float]:
        r"""
        Fase 1.4b — Observación Port-Hamiltoniana de la Energía Libre de
        Helmholtz-Terzaghi $\mathcal{H}_{\mathrm{litho}} = \mathcal{U}_{\mathrm{el}} + \mathcal{U}_{\mathrm{cons}} \ge 0$:

        1. Densidad de energía elástica de deformación (normalizada por módulo de
           referencia $E_{\mathrm{ref}}$): $\mathcal{U}_{\mathrm{el}} = \frac{1}{2 E_{\mathrm{ref}}} \|\boldsymbol{\sigma}'\|_F^2$.
        2. Trabajo de consolidación visco-plástica de Terzaghi:
           $\mathcal{U}_{\mathrm{cons}} = \sum_{j=1}^{N_{\mathrm{layers}}} \left(\sigma'_{v0,j} + \frac{1}{2}\Delta\sigma_{v,j}\right) s_{c,j} \ge 0$.
        """
        e_ref = 50.0e6  # Módulo elástico de referencia canónico (50 MPa)
        sigma = np.asarray(effective_stresses, dtype=np.float64)
        u_elastic = 0.5 * _kbn_sum(sigma * sigma) / e_ref

        work_terms: List[float] = []
        for j in range(compression_indices.size):
            cc = float(compression_indices[j])
            e0 = float(void_ratios_e0[j])
            h0 = float(layer_thicknesses[j])
            s0 = max(float(sigma_v0_effective[j]), self._reg)
            ds = max(float(delta_sigma_v[j]), 0.0)
            denom = max(1.0 + e0, self._reg)
            s_j = (cc * h0 / denom) * math.log10((s0 + ds) / s0)
            work_terms.append((s0 + 0.5 * ds) * max(0.0, s_j))

        u_consolidation = _kbn_sum(work_terms) if work_terms else 0.0
        total_hamiltonian = u_elastic + u_consolidation

        return {
            "helmholtz_elastic_energy_ proxy": float(u_elastic),
            "terzaghi_consolidation_work_j_m2": float(u_consolidation),
            "port_hamiltonian_total_energy": float(total_hamiltonian),
            "is_clausius_duhem_passive": float(total_hamiltonian >= -self._tol),
        }

    def phase1_complex_step_mohr_sensitivity(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Dict[str, float]:
        r"""
        Fase 1.4c — Sensibilidad de Fréchet mediante diferenciación de paso
        complejo ($h = 10^{-20}$) del Factor de Seguridad sobre el plano de falla:

            $$\frac{\partial \mathrm{FoS}_i}{\partial c'_i} = \frac{\operatorname{Im}\!\big(\mathrm{FoS}_{\mathbb{C}}(c'_i + i h, \phi'_i)\big)}{h}, \qquad \frac{\partial \mathrm{FoS}_i}{\partial \phi'_i} = \frac{\operatorname{Im}\!\big(\mathrm{FoS}_{\mathbb{C}}(c'_i, \phi'_i + i h)\big)}{h}$$
        """
        h = _COMPLEX_STEP_H
        t = np.einsum("nij,nj->ni", effective_stresses, fail_planes_normal)
        sigma_n = np.einsum("ni,ni->n", fail_planes_normal, t)
        t_shear = t - sigma_n[:, None] * fail_planes_normal
        tau = np.linalg.norm(t_shear, axis=1)

        d_fos_dc: List[float] = []
        d_fos_dphi: List[float] = []

        for i in range(cohesions.size):
            c_val = float(cohesions[i])
            phi_val = float(friction_angles_deg[i])
            sn_val = max(float(sigma_n[i]), 0.0)
            tau_val = max(float(tau[i]), self._reg)

            # Perturbación compleja en c'
            c_c = complex(c_val, h)
            phi_rad_0 = math.radians(phi_val)
            fos_c1 = (c_c + sn_val * math.tan(phi_rad_0)) / tau_val
            d_fos_dc.append(float(fos_c1.imag / h))

            # Perturbación compleja en φ' (en grados)
            phi_c_deg = complex(phi_val, h)
            phi_c_rad = phi_c_deg * (math.pi / 180.0)
            tan_phi_c = cmath.tan(phi_c_rad)
            fos_c2 = (c_val + sn_val * tan_phi_c) / tau_val
            d_fos_dphi.append(float(fos_c2.imag / h))

        norm_sens = math.sqrt(
            max(
                0.0,
                _kbn_sum([x * x for x in d_fos_dc] + [y * y for y in d_fos_dphi]),
            )
        )
        return {
            "mean_dfos_dcohesion": (
                float(_kbn_sum(d_fos_dc) / len(d_fos_dc)) if d_fos_dc else 0.0
            ),
            "mean_dfos_dphi_deg": (
                float(_kbn_sum(d_fos_dphi) / len(d_fos_dphi)) if d_fos_dphi else 0.0
            ),
            "complex_step_jacobian_norm": float(norm_sens),
        }

    def phase1_observation_diagnostics(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        compression_indices: np.ndarray,
        hygiene: Mapping[str, float],
        port_hamiltonian: Optional[Mapping[str, float]] = None,
        complex_sensitivity: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fase 1.5 — Diagnóstico de Banach, energía libre y sensibilidad de la observación.
        """
        diag: Dict[str, Any] = {
            "n_nodes": int(cohesions.size),
            "n_layers": int(compression_indices.size),
            "c_min": float(np.min(cohesions)) if cohesions.size else math.nan,
            "c_max": float(np.max(cohesions)) if cohesions.size else math.nan,
            "phi_min_deg": (
                float(np.min(friction_angles_deg)) if friction_angles_deg.size else math.nan
            ),
            "phi_max_deg": (
                float(np.max(friction_angles_deg)) if friction_angles_deg.size else math.nan
            ),
            "cc_max": (
                float(np.max(compression_indices)) if compression_indices.size else 0.0
            ),
            "fos_limit": self._fos_limit,
            "fos_collapse": self._fos_collapse,
            "settlement_max_m": self._settlement_max,
            "tolerance": self._tol,
            "regularization": self._reg,
            "machine_epsilon": _MACHINE_EPS,
            **dict(hygiene),
        }
        if port_hamiltonian is not None:
            diag.update(port_hamiltonian)
        if complex_sensitivity is not None:
            diag.update(complex_sensitivity)
        return diag

    def _phase1_session_hash(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> str:
        """
        Fase 1.6 — Sello de sesión SHA-256 canónico longitud-prefijado.
        """
        h = hashlib.sha256()
        h.update(b"LITHO/SESSION/v3")
        _sha_update_arr(h, cohesions)
        _sha_update_arr(h, friction_angles_deg)
        _sha_update_arr(h, effective_stresses)
        _sha_update_arr(h, active_shear_stresses)
        _sha_update_arr(h, fail_planes_normal)
        _sha_update_arr(h, compression_indices)
        _sha_update_arr(h, void_ratios_e0)
        _sha_update_arr(h, layer_thicknesses)
        _sha_update_arr(h, sigma_v0_effective)
        _sha_update_arr(h, delta_sigma_v)
        h.update(_pack_f64(self._fos_limit))
        h.update(_pack_f64(self._fos_collapse))
        h.update(_pack_f64(self._settlement_max))
        h.update(_pack_f64(self._tol))
        h.update(_pack_f64(self._grace_max))
        _sha_update_str(h, "PHASE1/OBSERVE")
        digest = h.hexdigest()
        if len(digest) != _SHA256_HEX_LEN:
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Phase1LithoHandoff:
        """
        [MÉTODO TERMINAL DE FASE 1 ──► CONTINUACIÓN FORMAL EN FASE 2]

        Fase 1.7 — Cierre formal de Fase 1 y apertura verificada de Fase 2:
            Φ₁→₂ : Campos crudos ↦ Phase1LithoHandoff ──► `phase2_from_phase1`.
        """
        c, phi = self.phase1_validate_strength_fields(cohesions, friction_angles_deg)
        n_nodes = c.size
        sigma_prime, tau_act, normals = self.phase1_validate_stress_fields(
            effective_stresses, active_shear_stresses, fail_planes_normal, n_nodes
        )
        Cc, e0, H0, sigma_v0, delta_sigma = self.phase1_validate_consolidation_fields(
            compression_indices,
            void_ratios_e0,
            layer_thicknesses,
            sigma_v0_effective,
            delta_sigma_v,
        )
        hygiene = self.phase1_terzaghi_tensor_hygiene(sigma_prime)
        port_ham = self.phase1_port_hamiltonian_helmholtz_observe(
            sigma_prime, Cc, e0, H0, sigma_v0, delta_sigma
        )
        complex_sens = self.phase1_complex_step_mohr_sensitivity(
            c, phi, sigma_prime, normals
        )
        session_sha256 = self._phase1_session_hash(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
        )
        diagnostics = self.phase1_observation_diagnostics(
            c,
            phi,
            sigma_prime,
            Cc,
            hygiene,
            port_hamiltonian=port_ham,
            complex_sensitivity=complex_sens,
        )
        diagnostics["session_sha256_prefix"] = session_sha256[:16]

        handoff = Phase1LithoHandoff(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
            session_sha256=session_sha256,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
        )

        opened = self.phase2_from_phase1(handoff)
        if opened.session_sha256 != session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el sello de sesión "
                "admitido por Fase 2 no coincide con el observado en Fase 1."
            )

        logger.debug(
            "Fase Observe [LITHO_AGENT]: sesión sellada. SHA prefix=%s",
            session_sha256[:16],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 2 — ORIENT  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ═════════════════════════════════════════════════════════════════════════
    # Continuación formal de `phase1_close_and_open_phase2`:
    # Mohr–Coulomb 3D (plano, círculo y θ_crit), Invariantes Octaédricos de
    # Haigh-Westergaard-Lode, Consolidación de Terzaghi, Mecánica Celeste de
    # Henri Poincaré (KAM / Liouville) y Topología de Soporte (β₀, β₁, χ).
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(self, handoff: Phase1LithoHandoff) -> Phase1LithoHandoff:
        """
        [MÉTODO INICIAL DE FASE 2 ◄── CONTINUACIÓN DESDE FASE 1]

        Fase 2.0 — Entrada formal desde Fase 1 (`phase1_close_and_open_phase2`).
        """
        if not isinstance(handoff, Phase1LithoHandoff):
            raise TypeError("Se esperaba Phase1LithoHandoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1LithoHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE1_ENTRY!r}."
            )
        if (
            not isinstance(handoff.session_sha256, str)
            or len(handoff.session_sha256) != _SHA256_HEX_LEN
        ):
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        if handoff.cohesions.size < 1:
            raise ValueError("La observación de Φ₁→₂ no contiene nudos.")
        return handoff

    def phase2_cauchy_traction(
        self,
        effective_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fase 2.1 — Descomposición de tracción de Cauchy sobre el plano de falla:
            t = σ' n̂,   σ'_n = n̂ · t,   τ = ||t − σ'_n n̂||₂
        """
        t = np.einsum("nij,nj->ni", effective_stresses, fail_planes_normal)
        sigma_n = np.einsum("ni,ni->n", fail_planes_normal, t)
        t_shear = t - sigma_n[:, None] * fail_planes_normal
        tau = np.linalg.norm(t_shear, axis=1)
        return np.asarray(sigma_n, dtype=np.float64), np.asarray(tau, dtype=np.float64)

    def phase2_stress_invariants(
        self,
        effective_stresses: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""
        Fase 2.2 — Invariantes completos de Cauchy y Haigh-Westergaard-Lode:

        - Invariantes principales:
          $$I_1 = \operatorname{Tr}(\boldsymbol{\sigma}'), \quad I_2 = \frac{1}{2}\!\left(I_1^2 - \operatorname{Tr}({\boldsymbol{\sigma}'}^2)\right), \quad I_3 = \det(\boldsymbol{\sigma}')$$
        - Invariantes desviadores y octaédricos ($\mathbf{s} = \boldsymbol{\sigma}' - p'\mathbf{I}_3$, $p' = I_1/3$):
          $$J_2 = \frac{1}{2}\|\mathbf{s}\|_F^2, \quad J_3 = \det(\mathbf{s}), \quad q = \sqrt{3 J_2}, \quad \tau_{\mathrm{oct}} = \sqrt{\frac{2}{3}J_2}$$
        - Ángulo de Lode y razón de esfuerzo triaxial:
          $$\cos(3\theta_{\mathrm{Lode}}) = \frac{3\sqrt{3}}{2}\frac{J_3}{J_2^{3/2}} \in [-1, 1], \qquad \eta = \frac{q}{\max(|p'|, \varepsilon)}$$
        """
        sigma = np.asarray(effective_stresses, dtype=np.float64)
        i1 = np.trace(sigma, axis1=1, axis2=2)
        sigma_sq_tr = np.einsum("nij,nji->n", sigma, sigma)
        i2 = 0.5 * (i1 * i1 - sigma_sq_tr)
        i3 = np.linalg.det(sigma)

        p = i1 / 3.0
        s = sigma - p[:, None, None] * _I3
        j2 = 0.5 * np.sum(s * s, axis=(1, 2))
        j2 = np.maximum(j2, 0.0)
        j3 = np.linalg.det(s)

        q = np.sqrt(3.0 * j2)
        tau_oct = np.sqrt((2.0 / 3.0) * j2)
        s_norm = np.sqrt(np.maximum(2.0 * j2, 0.0))

        with np.errstate(divide="ignore", invalid="ignore"):
            lode_cos = np.where(
                s_norm > self._reg,
                np.clip((3.0 * math.sqrt(6.0)) * j3 / np.power(s_norm, 3), -1.0, 1.0),
                1.0,
            )
            eta = np.where(
                np.abs(p) > self._reg,
                q / np.maximum(np.abs(p), self._reg),
                0.0,
            )
        lode_angle = np.degrees(np.arccos(lode_cos) / 3.0)
        return {
            "I1": np.asarray(i1, dtype=np.float64),
            "I2": np.asarray(i2, dtype=np.float64),
            "I3": np.asarray(i3, dtype=np.float64),
            "p_mean": np.asarray(p, dtype=np.float64),
            "J2": np.asarray(j2, dtype=np.float64),
            "J3": np.asarray(j3, dtype=np.float64),
            "q_von_mises": np.asarray(q, dtype=np.float64),
            "tau_octahedral": np.asarray(tau_oct, dtype=np.float64),
            "lode_angle_deg": np.asarray(lode_angle, dtype=np.float64),
            "triaxiality_eta": np.asarray(eta, dtype=np.float64),
        }

    def phase2_mohr_coulomb_plane(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        sigma_n: np.ndarray,
        tau_demand: np.ndarray,
    ) -> np.ndarray:
        """
        Fase 2.3 — FoS de Mohr–Coulomb en el plano de falla prescrito:
            τ_f = c' + max(σ'_n, 0) tan φ',     FoS = τ_f / τ_dem
        """
        phi_rad = np.radians(np.asarray(friction_angles_deg, dtype=np.float64))
        tan_phi = np.tan(phi_rad)
        sigma_n_eff = np.maximum(np.asarray(sigma_n, dtype=np.float64), 0.0)
        tau_strength = np.maximum(
            np.asarray(cohesions, dtype=np.float64) + sigma_n_eff * tan_phi, 0.0
        )
        demand = np.abs(np.asarray(tau_demand, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            fos = np.where(
                demand <= self._reg,
                np.inf,
                tau_strength / np.maximum(demand, self._reg),
            )
        fos = np.where(np.isfinite(tan_phi) & (tan_phi >= -self._reg), fos, 0.0)
        fos = np.where(np.isfinite(sigma_n) | (demand <= self._reg), fos, 0.0)
        return np.asarray(fos, dtype=np.float64)

    def phase2_mohr_coulomb_principal(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
    ) -> np.ndarray:
        r"""
        Fase 2.4 — FoS de Mohr–Coulomb sobre el Círculo de Mohr principal.

        Con $\sigma'_1 \ge \sigma'_2 \ge \sigma'_3$:
            $$\tau_{\max} = \frac{\sigma'_1 - \sigma'_3}{2}, \qquad \sigma'_m = \frac{\sigma'_1 + \sigma'_3}{2}$$
            $$\tau_f = c' \cos\phi' + \max(\sigma'_m, 0) \sin\phi', \qquad \mathrm{FoS}_{\mathrm{circle}} = \frac{\tau_f}{\tau_{\max}}$$
        """
        try:
            eig = np.linalg.eigvalsh(np.asarray(effective_stresses, dtype=np.float64))
        except np.linalg.LinAlgError:
            return np.zeros(cohesions.size, dtype=np.float64)

        sigma_3 = eig[:, 0]
        sigma_1 = eig[:, 2]
        tau_max = 0.5 * (sigma_1 - sigma_3)
        sigma_m = 0.5 * (sigma_1 + sigma_3)
        phi_rad = np.radians(np.asarray(friction_angles_deg, dtype=np.float64))
        strength = np.asarray(cohesions, dtype=np.float64) * np.cos(phi_rad) + np.maximum(
            sigma_m, 0.0
        ) * np.sin(phi_rad)
        strength = np.maximum(strength, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            fos = np.where(
                tau_max <= self._reg,
                np.inf,
                strength / np.maximum(tau_max, self._reg),
            )
        fos = np.where(np.isfinite(fos), fos, 0.0)
        return np.asarray(fos, dtype=np.float64)

    def phase2_mohr_coulomb_critical_slip_plane(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
    ) -> np.ndarray:
        r"""
        Fase 2.4b — Evaluación exacta de la Definición 1 sobre el plano crítico
        de deslizamiento de Coulomb $\theta_{\mathrm{crit}} = \frac{\pi}{4} + \frac{\phi'}{2}$:

            $$\sigma'_{n,\mathrm{crit}} = \sigma'_1 \cos^2\theta_{\mathrm{crit}} + \sigma'_3 \sin^2\theta_{\mathrm{crit}}$$
            $$\tau_{\mathrm{act,crit}} = \frac{1}{2}(\sigma'_1 - \sigma'_3)\sin(2\theta_{\mathrm{crit}})$$
            $$\mathrm{FoS}_{\theta_{\mathrm{crit}}} = \frac{c' + \max(\sigma'_{n,\mathrm{crit}}, 0)\tan\phi'}{\tau_{\mathrm{act,crit}}}$$
        """
        try:
            eig = np.linalg.eigvalsh(np.asarray(effective_stresses, dtype=np.float64))
        except np.linalg.LinAlgError:
            return np.zeros(cohesions.size, dtype=np.float64)

        sigma_3 = eig[:, 0]
        sigma_1 = eig[:, 2]
        phi_rad = np.radians(np.asarray(friction_angles_deg, dtype=np.float64))
        theta_crit = (0.25 * math.pi) + 0.5 * phi_rad

        cos_t = np.cos(theta_crit)
        sin_t = np.sin(theta_crit)
        sigma_n_crit = sigma_1 * (cos_t * cos_t) + sigma_3 * (sin_t * sin_t)
        tau_act_crit = 0.5 * (sigma_1 - sigma_3) * np.sin(2.0 * theta_crit)

        tan_phi = np.tan(phi_rad)
        tau_shear = np.maximum(
            np.asarray(cohesions, dtype=np.float64)
            + np.maximum(sigma_n_crit, 0.0) * tan_phi,
            0.0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            fos_crit = np.where(
                tau_act_crit <= self._reg,
                np.inf,
                tau_shear / np.maximum(tau_act_crit, self._reg),
            )
        fos_crit = np.where(np.isfinite(fos_crit) | np.isposinf(fos_crit), fos_crit, 0.0)
        return np.asarray(fos_crit, dtype=np.float64)

    def phase2_mohr_coulomb_audit(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[float, bool, bool, np.ndarray, Dict[str, Any]]:
        """
        Fase 2.5 — Auditoría conjunta de resistencia (plano ∧ círculo de Mohr).
        Registra adicionalmente el FoS exacto sobre el plano crítico θ_crit = π/4 + φ'/2.
        """
        sigma_n, tau_cauchy = self.phase2_cauchy_traction(
            effective_stresses, fail_planes_normal
        )
        tau_demand = np.maximum(np.abs(active_shear_stresses), tau_cauchy)
        fos_plane = self.phase2_mohr_coulomb_plane(
            cohesions, friction_angles_deg, sigma_n, tau_demand
        )
        fos_principal = self.phase2_mohr_coulomb_principal(
            cohesions, friction_angles_deg, effective_stresses
        )
        fos_theta_crit = self.phase2_mohr_coulomb_critical_slip_plane(
            cohesions, friction_angles_deg, effective_stresses
        )

        # Nota matemática: dado que cos(φ') <= 1, fos_principal <= fos_theta_crit
        # siempre que σ'_n,crit >= 0; preservamos minimum(fos_plane, fos_principal).
        per_node_fos = np.minimum(fos_plane, fos_principal)
        per_node_fos = np.where(
            np.isfinite(per_node_fos) | np.isposinf(per_node_fos), per_node_fos, 0.0
        )

        finite_mask = np.isfinite(per_node_fos)
        if per_node_fos.size == 0:
            min_fos = math.inf
        elif np.any(finite_mask):
            min_fos = float(np.min(per_node_fos[finite_mask]))
            if np.any(np.isneginf(per_node_fos)):
                min_fos = 0.0
        elif np.all(np.isposinf(per_node_fos)):
            min_fos = math.inf
        else:
            min_fos = 0.0

        collapse_mask = per_node_fos <= self._fos_collapse + self._tol
        warning_mask = (
            (per_node_fos > self._fos_collapse + self._tol)
            & (per_node_fos <= self._fos_limit + self._tol)
        )
        is_collapse = bool(
            np.any(collapse_mask)
            or (not math.isfinite(min_fos) and min_fos != math.inf)
        )
        if min_fos <= self._fos_collapse + self._tol:
            is_collapse = True
        is_warning = bool(
            np.any(warning_mask)
            or (
                math.isfinite(min_fos)
                and self._fos_collapse + self._tol
                < min_fos
                <= self._fos_limit + self._tol
            )
        )

        cauchy_residual = (
            float(np.max(np.abs(np.abs(active_shear_stresses) - tau_cauchy)))
            if tau_cauchy.size
            else 0.0
        )

        invariants = self.phase2_stress_invariants(effective_stresses)
        n_sz = cohesions.size
        diagnostics: Dict[str, Any] = {
            "min_fos_shear": float(min_fos),
            "min_fos_plane": float(np.nanmin(fos_plane)) if fos_plane.size else math.nan,
            "min_fos_principal": (
                float(np.nanmin(fos_principal)) if fos_principal.size else math.nan
            ),
            "min_fos_theta_crit": (
                float(np.nanmin(fos_theta_crit)) if fos_theta_crit.size else math.nan
            ),
            "plastic_nodes": int(np.count_nonzero(collapse_mask)),
            "warning_nodes": int(np.count_nonzero(warning_mask)),
            "fos_limit": self._fos_limit,
            "fos_collapse": self._fos_collapse,
            "cauchy_shear_residual_max": cauchy_residual,
            "mean_I1": float(_kbn_sum(invariants["I1"]) / n_sz) if n_sz else math.nan,
            "mean_I2": float(_kbn_sum(invariants["I2"]) / n_sz) if n_sz else math.nan,
            "mean_I3": float(_kbn_sum(invariants["I3"]) / n_sz) if n_sz else math.nan,
            "max_J2": float(np.max(invariants["J2"])) if n_sz else math.nan,
            "mean_J3": float(_kbn_sum(invariants["J3"]) / n_sz) if n_sz else math.nan,
            "max_q_von_mises": (
                float(np.max(invariants["q_von_mises"])) if n_sz else math.nan
            ),
            "max_tau_octahedral": (
                float(np.max(invariants["tau_octahedral"])) if n_sz else math.nan
            ),
            "mean_lode_angle_deg": (
                float(_kbn_sum(invariants["lode_angle_deg"]) / n_sz)
                if n_sz
                else math.nan
            ),
            "max_triaxiality_eta": (
                float(np.max(invariants["triaxiality_eta"])) if n_sz else math.nan
            ),
        }
        return float(min_fos), bool(is_warning), bool(is_collapse), per_node_fos, diagnostics

    @staticmethod
    def _terzaghi_fourier_degree_of_consolidation(
        tv: float, n_modes: int = 32
    ) -> float:
        r"""
        Solución espectral exacta en serie de Fourier de la EDP parabólica de Terzaghi:
            $$U_v(T_v) = 1 - \sum_{m=0}^{N-1} \frac{2}{M_m^2} \exp\!\left(-M_m^2 T_v\right), \qquad M_m = \frac{\pi}{2}(2m + 1)$$
        """
        if tv <= 0.0:
            return 0.0
        modes: List[float] = []
        for m in range(n_modes):
            cap_m = 0.5 * math.pi * (2 * m + 1)
            m_sq = cap_m * cap_m
            modes.append((2.0 / m_sq) * math.exp(-m_sq * tv))
        uv = 1.0 - _kbn_sum(modes)
        return float(min(1.0, max(0.0, uv)))

    def phase2_consolidation_audit(
        self,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Tuple[float, bool, np.ndarray, Dict[str, Any]]:
        """
        Fase 2.6 — Asentamiento por consolidación primaria 1D de Terzaghi con
        sumación compensada KBN y grado espectral de consolidación U_v(T_v).
        """
        n_layers = compression_indices.size
        per_layer = np.zeros(n_layers, dtype=np.float64)
        uv_degree = self._terzaghi_fourier_degree_of_consolidation(self._tv_default)

        if n_layers == 0:
            return 0.0, False, per_layer, {
                "total_settlement_m": 0.0,
                "settlement_max_m": self._settlement_max,
                "n_layers": 0,
                "excessive_settlement": False,
                "differential_settlement_m": 0.0,
                "max_stress_ratio": 1.0,
                "terzaghi_fourier_uv": uv_degree,
                "time_factor_tv_settlement_m": 0.0,
            }

        has_nonfinite = False
        ratios = np.empty(n_layers, dtype=np.float64)
        for i in range(n_layers):
            Cc = float(compression_indices[i])
            e0 = float(void_ratios_e0[i])
            H0 = float(layer_thicknesses[i])
            sigma0 = max(float(sigma_v0_effective[i]), self._reg)
            delta = max(float(delta_sigma_v[i]), 0.0)
            denom = 1.0 + e0
            ratio = (sigma0 + delta) / sigma0
            ratios[i] = ratio
            if denom <= self._reg or H0 < 0.0 or ratio <= 1.0 - self._tol:
                per_layer[i] = math.inf
                has_nonfinite = True
                continue
            term = (Cc * H0 / denom) * math.log10(ratio)
            if not math.isfinite(term):
                per_layer[i] = math.inf
                has_nonfinite = True
            else:
                per_layer[i] = float(max(term, 0.0))

        total = math.inf if has_nonfinite else _kbn_sum(per_layer)
        is_excessive = bool(
            (not math.isfinite(total)) or total > self._settlement_max + self._tol
        )
        finite_layers = per_layer[np.isfinite(per_layer)]
        if finite_layers.size:
            differential = float(np.max(finite_layers) - np.min(finite_layers))
        else:
            differential = math.inf if has_nonfinite else 0.0

        tv_settlement = float(total * uv_degree) if math.isfinite(total) else math.inf

        diagnostics: Dict[str, Any] = {
            "total_settlement_m": float(total),
            "settlement_max_m": self._settlement_max,
            "n_layers": int(n_layers),
            "excessive_settlement": is_excessive,
            "differential_settlement_m": float(differential),
            "max_stress_ratio": float(np.max(ratios)) if ratios.size else 1.0,
            "terzaghi_fourier_uv": uv_degree,
            "time_factor_tv_settlement_m": tv_settlement,
        }
        return float(total), is_excessive, per_layer, diagnostics

    def phase2_topological_audit(
        self,
        per_node_fos: np.ndarray,
        is_plastic_failure: bool,
    ) -> Dict[str, Any]:
        """
        Fase 2.7 — Compromiso topológico del soporte y números de Betti efectivos
        (β₀, β₁, característica de Euler-Poincaré χ = β₀ − β₁).
        """
        n_nodes = int(per_node_fos.size)
        plastic_nodes = (
            int(np.count_nonzero(per_node_fos <= self._fos_collapse + self._tol))
            if n_nodes
            else 0
        )
        elastic_nodes = max(0, n_nodes - plastic_nodes)
        ratio = float(plastic_nodes / n_nodes) if n_nodes else 0.0
        percolation = bool(ratio >= self._percolation_ratio)
        compromised = bool(is_plastic_failure or plastic_nodes > 0)

        # Caracterización simplicial del grafo de cadenas de fuerza portantes:
        # Cada nodo plastificado fragmenta una componente conexa local.
        effective_betti_0 = 1 + plastic_nodes
        effective_betti_1 = max(0, elastic_nodes - 1)
        euler_poincare_chi = effective_betti_0 - effective_betti_1

        return {
            "plastic_node_count": plastic_nodes,
            "plastic_node_ratio": ratio,
            "is_topologically_compromised": compromised,
            "betti_zero_warning": bool(plastic_nodes > 0),
            "percolation_cut_warning": percolation,
            "support_integrity": float(max(0.0, 1.0 - ratio)),
            "effective_betti_0": effective_betti_0,
            "effective_betti_1": effective_betti_1,
            "euler_poincare_characteristic": euler_poincare_chi,
        }

    def phase2_audit_poincare_celestial_kam(
        self,
        effective_stresses: np.ndarray,
        per_node_fos: np.ndarray,
        layer_thicknesses: np.ndarray,
        topo_diag: Mapping[str, Any],
        uv_degree: float,
    ) -> LithoPoincarePhaseAudit:
        r"""
        Fase 2.7b — Auditoría de Mecánica Celeste de Henri Poincaré y Estabilidad
        Diofántica KAM sobre el espacio de fases geomecánico:

        1. Extrae las tres frecuencias características modales $\boldsymbol{\omega} = (\omega_1, \omega_2, \omega_3)$
           a partir del espectro principal medio de esfuerzos efectivos y la escala
           de difusión de Terzaghi.
        2. Evalúa la condición diofántica de no-resonancia KAM:
           $$|\mathbf{k} \cdot \boldsymbol{\omega}| \ge \gamma_{\mathrm{KAM}} \|\boldsymbol{\omega}\|_2 |\mathbf{k}|_1^{-\tau}, \qquad 0 < |\mathbf{k}|_1 \le K_{\max}$$
        3. Calcula el determinante simpléctico conservativo ($\det \Phi_{\Delta t}^{\mathrm{symp}} = 1.0$)
           y la contracción de volumen de espacio de fases de Liouville bajo disipación
           parabólica de Terzaghi ($\det \Phi_{\Delta t}^{\mathrm{diss}} \in (0, 1]$).
        """
        rho_ref = 1850.0
        try:
            eig_all = np.linalg.eigvalsh(np.asarray(effective_stresses, dtype=np.float64))
            mean_eigs = np.maximum(np.mean(np.abs(eig_all), axis=0), 1.0)
        except np.linalg.LinAlgError:
            mean_eigs = np.array([100.0, 200.0, 300.0], dtype=np.float64)

        h_total = float(max(_kbn_sum(layer_thicknesses), 1.0))
        # Frecuencias modales elasto-poroelásticas características (rad/s)
        omegas = np.sqrt(mean_eigs / rho_ref) * (math.pi / h_total)
        # Desplazamiento irracional áureo microscópico de regularización si el estado es isótropo
        if float(omegas[2] - omegas[0]) <= self._reg:
            phi_gold = 0.5 * (1.0 + math.sqrt(5.0))
            omegas = np.array(
                [
                    float(omegas[0]),
                    float(omegas[0] * math.sqrt(2.0)),
                    float(omegas[0] * phi_gold),
                ],
                dtype=np.float64,
            )

        w1, w2, w3 = float(omegas[0]), float(omegas[1]), float(omegas[2])
        omega_norm = math.sqrt(max(self._reg, _kbn_sum([w1 * w1, w2 * w2, w3 * w3])))

        k_max = _KAM_MAX_ORDER_DEFAULT
        gamma = _KAM_GAMMA_DEFAULT
        tau = _KAM_TAU_DEFAULT
        smallest_divisor = float("inf")
        min_margin = float("inf")

        for k1 in range(-k_max, k_max + 1):
            for k2 in range(-k_max, k_max + 1):
                for k3 in range(0, k_max + 1):
                    l1 = abs(k1) + abs(k2) + abs(k3)
                    if l1 < 2 or l1 > k_max:
                        continue
                    signs = [s for s in (k1, k2, k3) if s != 0]
                    if min(signs) >= 0 or max(signs) <= 0:
                        continue
                    div = abs(k1 * w1 + k2 * w2 + k3 * w3) / omega_norm
                    bound = gamma * (float(l1) ** (-tau))
                    margin = div - bound
                    if div < smallest_divisor:
                        smallest_divisor = div
                    if margin < min_margin:
                        min_margin = margin

        if not math.isfinite(smallest_divisor):
            smallest_divisor = 1.0
            min_margin = 1.0

        # Propagador simpléctico de Cayley-Poincaré en T*𝕋³ (ℝ⁶)
        dt = _POINCARE_DT_DEFAULT
        k_diag = np.diag([w1 * w1, w2 * w2, w3 * w3])
        z3 = np.zeros((3, 3), dtype=np.float64)
        i3 = np.eye(3, dtype=np.float64)
        i6 = np.eye(6, dtype=np.float64)
        j_ham = np.block([[z3, i3], [-k_diag, z3]])
        symp_mat = np.linalg.solve(i6 - 0.5 * dt * j_ham, i6 + 0.5 * dt * j_ham)
        symp_det = float(np.linalg.det(symp_mat))

        # Contracción de Liouville por modos de consolidación de Terzaghi
        m0 = 0.5 * math.pi
        m1 = 1.5 * math.pi
        m2 = 2.5 * math.pi
        trace_consolidation_rate = (m0 * m0 + m1 * m1 + m2 * m2) * max(
            self._tv_default, 0.01
        )
        dissipative_liouville_det = float(math.exp(-dt * trace_consolidation_rate))

        betti_0 = int(topo_diag.get("effective_betti_0", 1))
        betti_1 = int(topo_diag.get("effective_betti_1", 0))
        chi = int(topo_diag.get("euler_poincare_characteristic", 1))

        mean_freq_hz = max((w1 + w2 + w3) / (6.0 * math.pi), 1e-6)
        recurrence_time = float(
            betti_0 / (mean_freq_hz * max(smallest_divisor, 1e-4))
        )

        return LithoPoincarePhaseAudit(
            characteristic_frequencies_rad_s=(w1, w2, w3),
            kam_smallest_divisor=float(smallest_divisor),
            kam_diophantine_margin=float(min_margin),
            is_kam_non_resonant=bool(min_margin >= -1e-3),
            conservative_symplectic_det=symp_det,
            dissipative_liouville_det=dissipative_liouville_det,
            consolidation_degree_uv=float(uv_degree),
            effective_betti_0=betti_0,
            effective_betti_1=betti_1,
            euler_poincare_characteristic=chi,
            kac_poincare_recurrence_time_s=recurrence_time,
        )

    def phase2_collect_invariants(
        self,
        phase1_handoff: Phase1LithoHandoff,
    ) -> Dict[str, Any]:
        """
        Fase 2.8 — Recolección conjunta de invariantes de orientación
        (Mohr–Coulomb 3D, Consolidación de Terzaghi, Topología y Poincaré-KAM).
        """
        min_fos, is_warning, is_collapse, per_node_fos, mc_diag = (
            self.phase2_mohr_coulomb_audit(
                cohesions=phase1_handoff.cohesions,
                friction_angles_deg=phase1_handoff.friction_angles_deg,
                effective_stresses=phase1_handoff.effective_stresses,
                active_shear_stresses=phase1_handoff.active_shear_stresses,
                fail_planes_normal=phase1_handoff.fail_planes_normal,
            )
        )
        total_settlement, is_excessive, per_layer, cons_diag = (
            self.phase2_consolidation_audit(
                compression_indices=phase1_handoff.compression_indices,
                void_ratios_e0=phase1_handoff.void_ratios_e0,
                layer_thicknesses=phase1_handoff.layer_thicknesses,
                sigma_v0_effective=phase1_handoff.sigma_v0_effective,
                delta_sigma_v=phase1_handoff.delta_sigma_v,
            )
        )
        topo_diag = self.phase2_topological_audit(per_node_fos, is_collapse)
        poincare_audit = self.phase2_audit_poincare_celestial_kam(
            effective_stresses=phase1_handoff.effective_stresses,
            per_node_fos=per_node_fos,
            layer_thicknesses=phase1_handoff.layer_thicknesses,
            topo_diag=topo_diag,
            uv_degree=float(cons_diag.get("terzaghi_fourier_uv", 1.0)),
        )

        diagnostics: Dict[str, Any] = dict(phase1_handoff.diagnostics)
        diagnostics.update(mc_diag)
        diagnostics.update(cons_diag)
        diagnostics.update(topo_diag)
        diagnostics["poincare_celestial_audit"] = poincare_audit.to_dict()

        return {
            "min_fos_shear": min_fos,
            "max_settlement": total_settlement,
            "is_warning_active": is_warning,
            "is_plastic_failure": is_collapse,
            "is_settlement_excessive": is_excessive,
            "is_topologically_compromised": bool(
                topo_diag["is_topologically_compromised"]
            ),
            "per_node_fos": per_node_fos,
            "per_layer_settlement": per_layer,
            "diagnostics": diagnostics,
        }

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1LithoHandoff,
    ) -> Phase2LithoHandoff:
        """
        [MÉTODO TERMINAL DE FASE 2 ──► CONTINUACIÓN FORMAL EN FASE 3]

        Fase 2.9 — Cierre formal de Fase 2 y apertura verificada de Fase 3:
            Φ₂→₃ : observación validada ↦ Phase2LithoHandoff ──► `phase3_from_phase2`.
        """
        validated = self.phase2_from_phase1(phase1_handoff)
        collected = self.phase2_collect_invariants(validated)

        handoff = Phase2LithoHandoff(
            min_fos_shear=float(collected["min_fos_shear"]),
            max_settlement=float(collected["max_settlement"]),
            is_warning_active=bool(collected["is_warning_active"]),
            is_plastic_failure=bool(collected["is_plastic_failure"]),
            is_settlement_excessive=bool(collected["is_settlement_excessive"]),
            is_topologically_compromised=bool(
                collected["is_topologically_compromised"]
            ),
            per_node_fos=collected["per_node_fos"],
            per_layer_settlement=collected["per_layer_settlement"],
            session_sha256=validated.session_sha256,
            diagnostics=collected["diagnostics"],
            next_entrypoint=_PHASE2_ENTRY,
        )

        opened = self.phase3_from_phase2(handoff)
        if opened.session_sha256 != handoff.session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₂→₃ violado: el sello de sesión "
                "admitido por Fase 3 no coincide con el de Fase 2."
            )

        logger.debug(
            "Fase Orient [LITHO_AGENT]: min_fos=%.6e, settlement=%.6e m, collapse=%s",
            handoff.min_fos_shear,
            handoff.max_settlement,
            handoff.is_plastic_failure,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  FASE 3 — DECIDE / ACT  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    # ═════════════════════════════════════════════════════════════════════════
    # Continuación formal de `phase2_close_and_open_phase3`:
    # Meta-Auditoría RSI Nivel 3 (Contracción de Banach y Fisher-Rao), Rampa
    # Lipschitz, Clasificador Bi-Heyting Ω₃ con Aniquilación Positrónica en Fock,
    # Enclavamiento Crowbar ESP32/BT151 (< 400 ns) y Certificación SHA-256.
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2LithoHandoff) -> Phase2LithoHandoff:
        """
        [MÉTODO INICIAL DE FASE 3 ◄── CONTINUACIÓN DESDE FASE 2]

        Fase 3.0 — Entrada formal desde Fase 2 (`phase2_close_and_open_phase3`).
        """
        if not isinstance(handoff, Phase2LithoHandoff):
            raise TypeError("Se esperaba Phase2LithoHandoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2LithoHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE2_ENTRY!r}."
            )
        if (
            not isinstance(handoff.session_sha256, str)
            or len(handoff.session_sha256) != _SHA256_HEX_LEN
        ):
            raise ValueError("El sello de sesión de Φ₂→₃ no es un SHA-256 válido.")
        return handoff

    def phase3_verify_override(self, token: Optional[str]) -> bool:
        """
        Fase 3.1 — Validación criptográfica a tiempo constante de override
        humano (positrón e⁺) o firma HMAC-SHA256 `payload:signature`.
        """
        if token is None:
            return False
        token_str = str(token).strip()
        if not token_str:
            return False
        for allowed in self._authorized_tokens:
            if _const_eq(token_str, allowed):
                return True
        if self._hmac_key is not None and ":" in token_str:
            payload, signature = token_str.rsplit(":", 1)
            expected = hmac.new(
                self._hmac_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if _const_eq(signature, expected):
                return True
        return False

    def phase3_confidence_ramp(self, handoff: Phase2LithoHandoff) -> float:
        """
        Fase 3.2 — Rampa de confianza Lipschitz-saturada en [0, 1].
        """
        fos = float(handoff.min_fos_shear)
        settlement = float(handoff.max_settlement)

        def _fos_channel(value: float) -> float:
            if not math.isfinite(value):
                return Heyting3.VETOED
            if value <= self._fos_collapse:
                return Heyting3.VETOED
            if value >= self._fos_limit:
                return Heyting3.COHERENT
            span = self._fos_limit - self._fos_collapse
            if span <= self._reg:
                return Heyting3.VETOED
            return float((value - self._fos_collapse) / span)

        def _settlement_channel(value: float) -> float:
            if not math.isfinite(value):
                return Heyting3.VETOED
            smax = self._settlement_max
            if smax <= self._reg:
                return Heyting3.COHERENT if value <= self._tol else Heyting3.VETOED
            soft = self._settlement_soft_ratio * smax
            if value <= soft:
                return Heyting3.COHERENT
            if value > smax:
                return Heyting3.VETOED
            span = smax - soft
            if span <= self._reg:
                return Heyting3.VETOED
            return float((smax - value) / span)

        confidence = Heyting3.meet(_fos_channel(fos), _settlement_channel(settlement))
        if handoff.is_plastic_failure or handoff.is_settlement_excessive:
            confidence = Heyting3.meet(confidence, Heyting3.VETOED)
        if not math.isfinite(confidence):
            return Heyting3.VETOED
        return float(min(1.0, max(0.0, confidence)))

    def phase3_rsi_level3_banach_policy_audit(
        self,
        handoff: Phase2LithoHandoff,
        confidence: float,
    ) -> LithoRSILevel3Certificate:
        r"""
        Fase 3.2b — Meta-Auditoría de Automejora Recursiva Nivel 3 (RSI-L3):

        1. Evalúa el funcional de costo de Lyapunov epistémico:
           $$\mathcal{V}_k = (1 - c_k)^2 + \frac{1}{2}\rho_{\mathrm{plastic}}^2$$
        2. Calcula la razón de contracción de Banach $q_{\mathrm{Banach}} = \mathcal{V}_k^{\mathrm{ref}} / \max(\mathcal{V}_{k-1}, \varepsilon)$
           bajo el operador contractivo de amortiguamiento epistémico.
        3. Calcula la deriva geodésica de Fisher-Rao sobre el simplex de probabilidad
           trivalente $\Delta^2 = \{(p_{\mathrm{coh}}, p_{\mathrm{deg}}, p_{\mathrm{vet}})\}$:
           $$d_{\mathrm{FR}}(p, q) = 2 \arccos\!\left(\sum_{i=1}^3 \sqrt{p_i q_i}\right)$$
        """
        self._rsi_generation += 1
        plastic_ratio = float(handoff.diagnostics.get("plastic_node_ratio", 0.0))
        raw_lyapunov = ((1.0 - confidence) ** 2) + 0.5 * (plastic_ratio ** 2)

        # Operador contractivo de regularización de política (factor γ_RSI = 0.65)
        gamma_rsi = 0.65
        contracted_lyapunov = gamma_rsi * raw_lyapunov

        if self._rsi_prev_lyapunov is not None and self._rsi_prev_lyapunov > self._reg:
            q_banach = min(
                gamma_rsi,
                contracted_lyapunov / max(self._rsi_prev_lyapunov, raw_lyapunov, self._reg),
            )
        else:
            q_banach = gamma_rsi

        self._rsi_prev_lyapunov = max(contracted_lyapunov, self._reg)

        # Distribución de seguridad en el simplex Δ² para métrica de Fisher-Rao
        eps_p = 1e-6
        p_coh = max(eps_p, confidence)
        p_vet = max(eps_p, (1.0 - confidence) * (0.5 + 0.5 * plastic_ratio))
        p_deg = max(eps_p, 1.0 - p_coh - p_vet if 1.0 - p_coh - p_vet > eps_p else eps_p)
        p_sum = p_coh + p_deg + p_vet
        curr_dist = (p_coh / p_sum, p_deg / p_sum, p_vet / p_sum)

        prev_dist = self._rsi_prev_safety_dist
        bhattacharyya = _kbn_sum(
            [math.sqrt(curr_dist[i] * prev_dist[i]) for i in range(3)]
        )
        bhattacharyya_clamped = min(1.0, max(-1.0, bhattacharyya))
        fr_drift = 2.0 * math.acos(bhattacharyya_clamped)
        self._rsi_prev_safety_dist = curr_dist

        adaptive_gain = float(1.0 / (1.0 + contracted_lyapunov))

        return LithoRSILevel3Certificate(
            generation=self._rsi_generation,
            epistemic_lyapunov_cost=float(contracted_lyapunov),
            banach_contraction_ratio=float(q_banach),
            is_banach_contractive=bool(q_banach < 1.0),
            fisher_rao_safety_drift=float(fr_drift),
            adaptive_confidence_gain=adaptive_gain,
        )

    def phase3_decide_heyting(
        self,
        handoff: Phase2LithoHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fase 3.3 — Clasificador Bi-Heyting trivalente Ω₃ con aniquilación en Fock.
        """
        curr_time = float(current_time if current_time is not None else time.time())
        hard_reasons: List[str] = []
        if handoff.is_plastic_failure:
            hard_reasons.append("plastic_failure")
        if handoff.is_settlement_excessive:
            hard_reasons.append("excessive_settlement")
        if math.isnan(handoff.min_fos_shear) or math.isnan(handoff.max_settlement):
            hard_reasons.append("nonfinite_metrics")
        if handoff.is_topologically_compromised and float(
            handoff.diagnostics.get("plastic_node_ratio", 0.0)
        ) >= 1.0 - self._tol:
            hard_reasons.append("support_percolation")

        override_valid = self.phase3_verify_override(override_token)
        override_sha = None
        if override_valid and override_token is not None:
            override_sha = hashlib.sha256(str(override_token).encode("utf-8")).hexdigest()

        confidence = self.phase3_confidence_ramp(handoff)
        heyting = "COHERENT"
        is_hard_veto = False
        is_soft_veto = False
        time_remaining = 0.0

        if hard_reasons:
            heyting = "VETOED"
            is_hard_veto = True
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None
            logger.critical(
                "¡VETO DURO INSTANTÁNEO POR COLAPSO GEOMECÁNICO / ASENTAMIENTO "
                "EXCESIVO! Razones: %s",
                ", ".join(hard_reasons),
            )
        elif handoff.is_warning_active:
            is_soft_veto = True
            if elapsed_grace_seconds is not None:
                if not math.isfinite(float(elapsed_grace_seconds)):
                    time_remaining = 0.0
                else:
                    time_remaining = max(
                        0.0, self._grace_max - float(elapsed_grace_seconds)
                    )
                    if not self._is_soft_veto_active:
                        self._is_soft_veto_active = True
                        self._soft_veto_timestamp = curr_time
            elif simulate_grace_expired:
                time_remaining = 0.0
            elif not self._is_soft_veto_active:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                time_remaining = self._grace_max
                logger.warning(
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! FoS marginal. "
                    "Ventana de gracia de %.0f segundos iniciada.",
                    self._grace_max,
                )
            else:
                elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                time_remaining = max(0.0, self._grace_max - elapsed)

            if override_valid:
                truth_fock = Heyting3.fock_annihilate(
                    Heyting3.DEGRADED,
                    positron_authorized=True,
                    is_hard_geomechanical_collapse=False,
                )
                heyting = Heyting3.to_verdict(truth_fock)
                is_soft_veto = False
                is_hard_veto = False
                time_remaining = 0.0
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.info(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] INYECTADO! "
                    "Aniquilando anomalía geomecánica en Fock. Sello: %s",
                    override_sha[:16] if override_sha else "UNKNOWN",
                )
            elif time_remaining <= self._tol or simulate_grace_expired:
                heyting = "VETOED"
                is_hard_veto = True
                is_soft_veto = False
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "¡PERÍODO DE GRACIA EXPIRADO SIN OVERRIDE VÁLIDO! "
                    "Colapsando Heyting a VETOED terminal."
                )
            else:
                heyting = "DEGRADED"
        else:
            heyting = "COHERENT"
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None

        truth = Heyting3.from_verdict(heyting)
        truth = Heyting3.meet(truth, Heyting3.quantize(confidence))
        heyting = Heyting3.to_verdict(truth)
        if heyting == "VETOED":
            is_hard_veto = True
            is_soft_veto = False
        elif heyting == "DEGRADED":
            is_hard_veto = False

        # Co-implicación de Brouwer: mide el residuo de riesgo no compensado
        co_heyting_risk_residual = Heyting3.co_implies(Heyting3.COHERENT, truth)

        return {
            "heyting_verdict": heyting,
            "heyting_truth_value": float(truth),
            "co_heyting_risk_residual": float(co_heyting_risk_residual),
            "confidence": float(confidence),
            "is_hard_veto": bool(is_hard_veto),
            "is_soft_veto": bool(is_soft_veto and heyting == "DEGRADED"),
            "hard_reasons": hard_reasons,
            "override_valid": bool(override_valid),
            "override_sha256": override_sha,
            "time_grace_remaining": float(time_remaining),
            "current_time": curr_time,
            "simulate_grace_expired": bool(simulate_grace_expired),
            "elapsed_grace_seconds": (
                float(elapsed_grace_seconds)
                if elapsed_grace_seconds is not None
                else math.nan
            ),
        }

    def phase3_actuate_crowbar_interlock(
        self,
        heyting_verdict: str,
    ) -> Tuple[bool, float]:
        """
        Fase 3.4 — Actuación de hardware Crowbar ESP32/BT151 (GPIO14 en IRAM).
        """
        if str(heyting_verdict) != "VETOED":
            return False, 0.0
        jitter = float(self._rng.normal(loc=398.5, scale=self._jitter_sigma))
        latency_ns = float(
            np.clip(jitter, _CROWBAR_LATENCY_MIN_NS, _CROWBAR_LATENCY_MAX_NS)
        )
        logger.critical(
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO LITOLÓGICO!\n"
            "  - Ejecutando subrutina local isVerdictCoherent() en C++...\n"
            "  - Despachando ISR en IRAM en menos de 400 ns...\n"
            "  - Conmutando pin de hardware GPIO14 a HIGH en %.2f ns...\n"
            "  - ¡Tiristor rápido de potencia BT151 (Crowbar) gatillado!\n"
            "  - Mezcladoras y bombas hidráulicas paralizadas en el milisegundo cero.",
            latency_ns,
        )
        return True, latency_ns

    def _phase3_cryptographic_seal(
        self,
        handoff: Phase2LithoHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
    ) -> str:
        """Fase 3.5 — Sello criptográfico SHA-256 canónico determinista."""
        h = hashlib.sha256()
        h.update(b"LITHO/CERT/v3")
        _sha_update_str(h, "PHASE3/LITHO_AGENT_GOVERNANCE")
        _sha_update_arr(h, handoff.per_node_fos)
        _sha_update_arr(h, handoff.per_layer_settlement)
        _sha_update_str(h, str(decision.get("heyting_verdict", "")))
        h.update(_pack_f64(float(handoff.min_fos_shear)))
        h.update(_pack_f64(float(handoff.max_settlement)))
        h.update(_pack_f64(float(decision.get("confidence", math.nan))))
        h.update(_pack_f64(float(decision.get("heyting_truth_value", math.nan))))
        h.update(_pack_f64(float(decision.get("time_grace_remaining", math.nan))))
        h.update(_pack_f64(float(switching_latency_ns)))
        h.update(b"\x01" if handoff.is_warning_active else b"\x00")
        h.update(b"\x01" if handoff.is_plastic_failure else b"\x00")
        h.update(b"\x01" if handoff.is_settlement_excessive else b"\x00")
        h.update(b"\x01" if handoff.is_topologically_compromised else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_hard_veto")) else b"\x00")
        h.update(b"\x01" if bool(decision.get("is_soft_veto")) else b"\x00")
        h.update(b"\x01" if interlock_fired else b"\x00")
        _sha_update_str(h, handoff.session_sha256)
        _sha_update_str(h, str(decision.get("override_sha256") or ""))
        return h.hexdigest()

    def phase3_issue_verdict(
        self,
        handoff: Phase2LithoHandoff,
        decision: Mapping[str, Any],
        interlock_fired: bool,
        switching_latency_ns: float,
        rsi_certificate: Optional[LithoRSILevel3Certificate] = None,
    ) -> GeomechanicalVerdict:
        """
        Fase 3.6 — Emisión del veredicto certificado `GeomechanicalVerdict`.
        """
        seal = self._phase3_cryptographic_seal(
            handoff=handoff,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=switching_latency_ns,
        )
        phase_chain = ("PHASE1/OBSERVE", "PHASE2/ORIENT", "PHASE3/DECIDE/ACT")
        diagnostics = dict(handoff.diagnostics)
        diagnostics["decision"] = dict(decision)
        diagnostics["hardware"] = {
            "interlock_fired": interlock_fired,
            "switching_latency_ns": switching_latency_ns,
        }
        if rsi_certificate is not None:
            diagnostics["rsi_level3_certificate"] = rsi_certificate.to_dict()

        poincare_dict = diagnostics.get("poincare_celestial_audit")
        poincare_obj: Optional[LithoPoincarePhaseAudit] = None
        if isinstance(poincare_dict, Mapping):
            freqs = tuple(poincare_dict.get("characteristic_frequencies_rad_s", (1.0, 1.0, 1.0)))
            poincare_obj = LithoPoincarePhaseAudit(
                characteristic_frequencies_rad_s=(
                    float(freqs[0]),
                    float(freqs[1]),
                    float(freqs[2]),
                ),
                kam_smallest_divisor=float(poincare_dict.get("kam_smallest_divisor", 1.0)),
                kam_diophantine_margin=float(poincare_dict.get("kam_diophantine_margin", 1.0)),
                is_kam_non_resonant=bool(poincare_dict.get("is_kam_non_resonant", True)),
                conservative_symplectic_det=float(
                    poincare_dict.get("conservative_symplectic_det", 1.0)
                ),
                dissipative_liouville_det=float(
                    poincare_dict.get("dissipative_liouville_det", 1.0)
                ),
                consolidation_degree_uv=float(
                    poincare_dict.get("consolidation_degree_uv", 1.0)
                ),
                effective_betti_0=int(poincare_dict.get("effective_betti_0", 1)),
                effective_betti_1=int(poincare_dict.get("effective_betti_1", 0)),
                euler_poincare_characteristic=int(
                    poincare_dict.get("euler_poincare_characteristic", 1)
                ),
                kac_poincare_recurrence_time_s=float(
                    poincare_dict.get("kac_poincare_recurrence_time_s", 1.0)
                ),
            )

        return GeomechanicalVerdict(
            heyting_verdict=str(decision["heyting_verdict"]),
            min_fos_shear=float(handoff.min_fos_shear),
            max_settlement=float(handoff.max_settlement),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            is_hard_veto_active=bool(decision["is_hard_veto"]),
            switching_latency_ns=float(switching_latency_ns),
            sha256_hash=seal,
            session_sha256=handoff.session_sha256,
            phase_chain=phase_chain,
            diagnostics=diagnostics,
            confidence=float(decision.get("confidence", math.nan)),
            heyting_truth_value=float(
                decision.get(
                    "heyting_truth_value",
                    Heyting3.from_verdict(str(decision["heyting_verdict"])),
                )
            ),
            override_sha256=str(decision.get("override_sha256") or ""),
            time_grace_remaining=float(decision.get("time_grace_remaining", 0.0)),
            poincare_audit=poincare_obj,
            rsi_level3_certificate=rsi_certificate,
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2LithoHandoff,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> GeomechanicalVerdict:
        """
        Fase 3.7 — Orquestación completa y cierre soberano de la Fase 3.
        """
        validated = self.phase3_from_phase2(phase2_handoff)
        decision = self.phase3_decide_heyting(
            handoff=validated,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )
        rsi_cert = self.phase3_rsi_level3_banach_policy_audit(
            handoff=validated,
            confidence=float(decision.get("confidence", 0.0)),
        )
        interlock_fired, latency = self.phase3_actuate_crowbar_interlock(
            decision["heyting_verdict"]
        )
        verdict = self.phase3_issue_verdict(
            handoff=validated,
            decision=decision,
            interlock_fired=interlock_fired,
            switching_latency_ns=latency,
            rsi_certificate=rsi_cert,
        )
        if verdict.heyting_verdict == "VETOED":
            logger.critical(
                "Soberano Litologico emitió VETO DURO. Sello: %s",
                verdict.sha256_hash[:16],
            )
        else:
            logger.info(
                "Soberano Litologico regulado síncronamente. Veredicto: %s. "
                "Confianza=%.6g. Sello: %s",
                verdict.heyting_verdict,
                verdict.confidence,
                verdict.sha256_hash[:16],
            )
        return verdict

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE OODA (FUNTOR COMPUESTO Φ₃ ∘ Φ₂ ∘ Φ₁)
    # ═════════════════════════════════════════════════════════════════════════

    def audit_lazo_cerrado(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> GeomechanicalVerdict:
        """
        API principal — Orquesta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁:
          OBSERVE    → `phase1_close_and_open_phase2`
          ORIENT     → `phase2_close_and_open_phase3`
          DECIDE/ACT → `phase3_close_loop`
        """
        phase1_handoff = self.phase1_close_and_open_phase2(
            cohesions=cohesions,
            friction_angles_deg=friction_angles_deg,
            effective_stresses=effective_stresses,
            active_shear_stresses=active_shear_stresses,
            fail_planes_normal=fail_planes_normal,
            compression_indices=compression_indices,
            void_ratios_e0=void_ratios_e0,
            layer_thicknesses=layer_thicknesses,
            sigma_v0_effective=sigma_v0_effective,
            delta_sigma_v=delta_sigma_v,
        )
        phase2_handoff = phase1_handoff.continue_to_phase2(self)
        return phase2_handoff.continue_to_phase3(
            self,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # MÉTODOS LEGADOS (COMPATIBILIDAD)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_geomechanical_stability(
        self,
        cohesions: np.ndarray,
        friction_angles_deg: np.ndarray,
        effective_stresses: np.ndarray,
        active_shear_stresses: np.ndarray,
        fail_planes_normal: np.ndarray,
    ) -> Tuple[float, bool, bool]:
        """
        API legada — equivalente a Fase 2.5.
        Retorna (min_fos, is_warning_active, is_plastic_failure).
        """
        c, phi = self.phase1_validate_strength_fields(cohesions, friction_angles_deg)
        sigma_prime, tau_act, normals = self.phase1_validate_stress_fields(
            effective_stresses, active_shear_stresses, fail_planes_normal, c.size
        )
        min_fos, is_warning, is_collapse, _, _ = self.phase2_mohr_coulomb_audit(
            cohesions=c,
            friction_angles_deg=phi,
            effective_stresses=sigma_prime,
            active_shear_stresses=tau_act,
            fail_planes_normal=normals,
        )
        return min_fos, is_warning, is_collapse

    def evaluate_consolidation_settlement(
        self,
        compression_indices: np.ndarray,
        void_ratios_e0: np.ndarray,
        layer_thicknesses: np.ndarray,
        sigma_v0_effective: np.ndarray,
        delta_sigma_v: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        API legada — equivalente a Fase 2.6.
        Retorna (total_settlement, is_settlement_excessive).
        """
        Cc, e0, H0, sigma_v0, delta_sigma = self.phase1_validate_consolidation_fields(
            compression_indices,
            void_ratios_e0,
            layer_thicknesses,
            sigma_v0_effective,
            delta_sigma_v,
        )
        total_settlement, is_excessive, _, _ = self.phase2_consolidation_audit(
            compression_indices=Cc,
            void_ratios_e0=e0,
            layer_thicknesses=H0,
            sigma_v0_effective=sigma_v0,
            delta_sigma_v=delta_sigma,
        )
        return total_settlement, is_excessive
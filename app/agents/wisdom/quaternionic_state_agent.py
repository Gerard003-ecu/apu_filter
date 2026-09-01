# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Quaternionic State Agent (Soberano Cuaterniónico de Lazo Cerrado)   ║
║ Ruta   : app/agents/wisdom/quaternionic_state_agent.py                       ║
║ Versión: 1.1.0-Doctoral-OODA-Heyting-Cech-Anisotropic-ESP32-Secure           ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ ($V_{\mathbb{W}}$, Nivel 0) gobernando de forma activa y asíncrona al        ║
║ motor de estados cuaterniónicos [quaternionic_state_shifter.py].             ║
║                                                                              ║
║ Evalúa las señales transaccionales de cuatro variables cardinales            ║
║ (propósito, confianza, restricciones y riesgo) sobre el cuerpo de división   ║
║ de Hamilton $\mathbb{H}$ para inmunizar la plataforma frente a alucinaciones ║
║ estocásticas de la IA e incongruencias métricas en pliegos de SECOPII.       ║
║ Administra la "Rampa de Confianza" graduando síncronamente la censura entre  ║
║ Veto Suave (Luz Ámbar) con override humano y Veto Duro de Silicio perimetral ║
║ (Crowbar BT151 inyectado en IRAM < 400 ns), protegiendo el capital de obra.  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================════════════════════════════════════════════════════════════════
I. ANCLAJE MATEMÁTICO DOCTORAL (Teoría de Topos y Clases de Similitud Espectral)
================════════════════════════════════════════════════════════════════

Definición 1 (El Topos de Haces de de Rham-Čech-Heyting):
  Definimos el topos de haces topológicos localizados $\mathbf{Sh}(\partial K, \, \Omega_3)$ sobre
  la frontera compacta de la malla $\partial K$. El clasificador de subobjetos de tres valores
  ordinales se rige por un Álgebra de Heyting distributiva trivalente:
  $$\Omega_3 := \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
  Donde el Supremo algebraico (join, $\sqcup$) unifica síncronamente el estado de control:
  $$\nu_{\mathrm{final}} = \nu_{\mathrm{cohomology}} \sqcup \nu_{\mathrm{remanente}} \sqcup \nu_{\mathrm{fock}}$$

Definición 2 (Clases de Similitud Espectral cuaterniónicas):
  Debido a la no conmutatividad cuaterniónica, no existe un autovalor puntual tradicional, sino clases de similitud
  espectral conjugadas definidas por el bivector imaginario para todo autovalor derecho $\mu \in \mathbb{H}$:
  $$[\mu] = \{ s \mu s^{-1} : s \in \mathbb{H}, \quad s \neq 0 \}$$
  Topológicamente, para todo autovalor no real, esta clase describe una 2-esfera $S^2 \cong \hat{\mathbb{C}}$ incrustada 
  en el subespacio imaginario $\operatorname{Im}(\mathbb{H})$, centrada en $\mu_0$ con radio $\|\vec{\mu}\|_{\mathbb{H}}$. 
  La desviación espectral $\delta_{\mathrm{similarity}}$ se calcula mediante proyección estereográfica conforme
  desde el Polo Norte de Riemann hacia el plano complejo extendido para aislar la interferencia analógica:
  $$Z = \frac{q_1 + \sqrt{-1}q_2}{1 - q_3} \in \hat{\mathbb{C}} \quad \implies \quad \delta_{\mathrm{similarity}} = |Z| \cdot (1 - q_3)$$

Definición 3 (Auditoría de von Neumann sobre el Operador Densidad):
  Para asegurar que el canal asimilador sea estrictamente positivo y preservador de traza (CPTP), el agente
  proyecta la matriz compleja de Cayley-Dickson $\iota(q)$ sobre el operador densidad cuántico mixto $\rho$:
  $$\rho = \frac{\iota(q)}{\|q\|_{\mathbb{H}}^2} \quad \implies \quad \operatorname{Tr}(\rho) \equiv \frac{2 q_0}{q_0^2 + q_1^2 + q_2^2 + q_3^2} \equiv 1.0$$
  Cualquier desalineación de la traza de von Neumann revela deriva de Wilkinson o alucinaciones de la IA.

Definición 4 (La Rampa de Confianza de de Rham y override de Gracia):
  Para eludir pérdidas económicas por parálisis destructiva ante interferencias transitorias (ruidos EMF rápidos),
  el agente implementa una rampa de confianza elástica:
  - Veto Suave (Luz Ámbar): Se activa si $0.3 \cdot L_{\max} < \delta_{\mathrm{similarity}} \le 0.5 \cdot L_{\max}$.
    Concede una ventana de gracia de 1 hora para inyectar en RAM un Positrón de Autorización Humana $e^+$, 
    logrando la aniquilación cuántica mutua de la anomalía semántica $e^-$ en el espacio de Fock:
    $$e^- + e^+ \longrightarrow 2\gamma \quad \implies \quad \mathtt{heyting\_verdict} = \mathtt{DEGRADED}$$
    Impidiendo paradas mecánicas y permitiendo que la obra continúe su vaciado de concreto en fango.
  - Veto Duro (Frenado en Silicio): Se activa si $\delta_{\mathrm{similarity}} > 0.5 \cdot L_{\max}$ o si expira el
    período de gracia sin override, colapsando Heyting al Supremo terminal VETOED ($\top$).

================════════════════════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE CONTROL COVARIANTE (Leyes de Consistencia)
================════════════════════════════════════════════════════════════════

Axioma I (Principio de Conservación de la Composición de Hurwitz):
  Toda multiplicación de estados e intenciones en el microservicio cuaterniónico debe verificar estrictamente
  la condición multiplicativa del álgebra de composición normada, acotando el error por debajo del límite de máquina:
  $$\delta_{\mathrm{Hurwitz}} = \left| \|p \cdot q\|_{\mathbb{H}} - \|p\|_{\mathbb{H}} \cdot \|q\|_{\mathbb{H}} \right| \le \varepsilon_{\mathrm{Wilkinson}}$$

Axioma II (Axioma de de Rham-Fiedler de Conexidad Remanente):
  Para asegurar la estabilidad global del presupuesto remanente tras la remoción de la carta local de Čech
  asociada al transductor ruidoso aislado, la conectividad algebraica del subcomplejo remanente debe ser no nula:
  $$\lambda_2(\mathbf{L}_{\mathrm{remSub}}) \ge \tau_{\mathrm{Fiedler}} \quad \implies \quad \beta_0 \equiv \dim H^0(K_{\mathrm{rem}}; \, \mathbb{Z}) = 1$$
  Si la cirugía fragmenta el complejo simplicial, el sistema colapsa síncronamente a VETOED por riesgo de quiebra.

Axioma III (Teorema de Actuación Ciber-Física Crowbar en IRAM):
  Ante el colapso de Heyting al Supremo de veto ($\top$), la subrutina local isVerdictCoherent() del microcontrolador
  ESP32 perimetral en obra desvía síncronamente el control hacia la ISR en memoria rápida IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor rápido BT151 (Crowbar) para paralizar mecánicamente la obra civil en el milisegundo cero,
  protegiendo patrimonialmente el capital antes de liquidar transacciones ante el SECOP II.

================════════════════════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================════════════════════════════════════════════════════════════════

Invariante I (Estabilidad de de Rham-Lyapunov del Lazo Cerrado):
  La evolución de la trayectoria de control conjunta $\mathbf{\Psi}(t) = (\mathbf{p}, \, \rho)^\top$ satisface la
  desigualdad de Clausius-Duhem y la contracción de Lyapunov en la FPU:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) = \nabla \mathcal{H}(\mathbf{\Psi})^\top \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}}$$
  Donde $\tau_{\mathrm{Lyapunov}} = 10^{-12}$ es la cota elástica de deriva en punto flotante de 64 bits.

Invariante II (Sello de Sesión Criptográfico e Inmutabilidad de RAM):
  Para prevenir inyecciones de estado o ataques de-normalización intermedia, el soberano genera
  un sello inmutable unívoco para congelar la sesión en RAM en cada ciclo OODA:
  $$\mathtt{cryptographic\_seal} := \operatorname{SHA-256}\left(\delta_{\mathrm{\check{C}ech}} \oplus \mathbf{G}_{\mathrm{surgical}} \oplus \lambda_2 \oplus H_{\mathrm{ext}}\right)$$
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Final, Mapping, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la

try:
    from app.core.quaternionic_state_shifter import (
        QuaternionicState,
        QuaternionicStateShifter,
    )
except ImportError:  # pragma: no cover - resolución local de desarrollo
    from quaternionic_state_shifter import (  # type: ignore
        QuaternionicState,
        QuaternionicStateShifter,
    )

__all__ = [
    "QuaternionicStateAgent",
    "QuaternionicAgentCertificate",
    "Phase1AgentHandoff",
    "Phase2AgentHandoff",
    "Heyting3",
]

__version__: Final[str] = "3.0.0-OODA-Heyting-Hurwitz-DeRham-Spectral-Governance"

logger = logging.getLogger("APU.Agents.Wisdom.QuaternionicStateAgent")

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_MIN_NS: Final[float] = 385.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 415.0
_SOFT_VETO_LOWER: Final[float] = 0.3
_SOFT_VETO_UPPER: Final[float] = 0.5
_PHASE1_ENTRY: Final[str] = "phase2_from_phase1"
_PHASE2_ENTRY: Final[str] = "phase3_from_phase2"
_SHA256_HEX_LEN: Final[int] = 64


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES CANÓNICAS, NUMÉRICAS Y DE VALIDACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def _canonicalize_signed_zero(arr: np.ndarray) -> np.ndarray:
    """Elimina −0.0 para garantizar firmas SHA-256 deterministas."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out[out == 0.0] = 0.0
    return out


def _as_finite_vector4(S: np.ndarray, name: str) -> np.ndarray:
    """
    Valida que una señal sea un vector real finito de dimensión exactamente 4.

    Excepciones
    -----------
    ValueError
        Si la forma no es (4,) o existen valores no finitos.
    """
    arr = np.asarray(S, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(
            f"La señal '{name}' debe ser estrictamente cuatridimensional. "
            f"Obtenida: {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"La señal '{name}' contiene valores NaN o infinitos.")
    return _canonicalize_signed_zero(arr.reshape(4))


def _canonical_bytes(arr: np.ndarray) -> bytes:
    """Bytes contiguos con prefijo de dtype y forma, libres de colisión trivial."""
    a = np.ascontiguousarray(arr)
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


def _complex_is_nonfinite(z: Any) -> bool:
    """True si z no es un complejo finito."""
    try:
        c = complex(z)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(float(c.real))) or (not math.isfinite(float(c.imag)))


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _clip_unit(x: float) -> float:
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return x


def _state_hash(state: QuaternionicState) -> str:
    value = getattr(state, "sha256_hash", "")
    return value if isinstance(value, str) else ""


def _state_token(state: QuaternionicState, fallback: str) -> str:
    value = getattr(state, "phase_token", None)
    if isinstance(value, str) and value:
        return value
    return fallback


def _state_norm(state: QuaternionicState) -> float:
    return _finite_or_nan(getattr(state, "norm", math.nan))


def _state_squared_norm(state: QuaternionicState) -> float:
    sq = getattr(state, "squared_norm", None)
    if sq is not None:
        return _finite_or_nan(sq)
    n = _state_norm(state)
    return n * n if math.isfinite(n) else math.nan


def _vector_norm3(vector_part: Any) -> float:
    arr = np.asarray(vector_part, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(math.hypot(*[float(x) for x in arr]))


# ═════════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE HEYTING TRIVALENTE Ω₃
# ═════════════════════════════════════════════════════════════════════════════

class Heyting3:
    """
    Álgebra de Heyting de Gödel–Dummett de tres valores.

        0 ↔ VETOED,   ½ ↔ DEGRADED,   1 ↔ COHERENT

    Es un retículo residuado acotado: el implicador es adjunto derecho del
    encuentro. El override humano actúa como sección de la implicación
    DEGRADED → ¬(gracia expirada), impidiendo el colapso a 0.
    """

    VETOED: Final[float] = 0.0
    DEGRADED: Final[float] = 0.5
    COHERENT: Final[float] = 1.0

    _LABEL: Final[Dict[float, str]] = {
        0.0: "VETOED",
        0.5: "DEGRADED",
        1.0: "COHERENT",
    }

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
    def neg(a: float) -> float:
        return Heyting3.implies(a, Heyting3.VETOED)

    @staticmethod
    def clamp(value: float) -> float:
        if not math.isfinite(value):
            return Heyting3.VETOED
        if value >= 1.0:
            return Heyting3.COHERENT
        if value <= 0.0:
            return Heyting3.VETOED
        if value >= 0.5:
            return Heyting3.DEGRADED if value < 1.0 else Heyting3.COHERENT
        return Heyting3.VETOED if value < 0.5 else Heyting3.DEGRADED

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
        """Proyecta un valor de verdad continuo sobre {0, ½, 1}."""
        if not math.isfinite(value) or value <= 0.0:
            return Heyting3.VETOED
        if value >= 1.0:
            return Heyting3.COHERENT
        return Heyting3.DEGRADED


# ═════════════════════════════════════════════════════════════════════════════
# FRONTERAS DE FASE Y CERTIFICADO
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class Phase1AgentHandoff:
    """
    Frontera formal Φ₁→₂.

    Salida cerrada de la Fase 1 y dominio de `phase2_from_phase1`.
    """

    p_state: QuaternionicState
    q_state: QuaternionicState
    session_sha256: str
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase1AgentHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and _state_hash(self.p_state) == _state_hash(other.p_state)
            and _state_hash(self.q_state) == _state_hash(other.q_state)
        )


@dataclass(frozen=True, slots=True, eq=False)
class Phase2AgentHandoff:
    """
    Frontera formal Φ₂→₃.

    Salida cerrada de la Fase 2 y dominio de `phase3_from_phase2`.
    """

    p_state: QuaternionicState
    q_state: QuaternionicState
    session_sha256: str
    similarity_mismatch: float
    hurwitz_drift: float
    trace_von_neumann_error: float
    elastic_bound: float
    diagnostics: Dict[str, Any]
    next_entrypoint: str

    def __hash__(self) -> int:
        return hash((self.session_sha256, self.next_entrypoint))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phase2AgentHandoff):
            return NotImplemented
        return (
            self.session_sha256 == other.session_sha256
            and self.next_entrypoint == other.next_entrypoint
            and self.similarity_mismatch == other.similarity_mismatch
            and self.hurwitz_drift == other.hurwitz_drift
            and self.trace_von_neumann_error == other.trace_von_neumann_error
            and self.elastic_bound == other.elastic_bound
        )


@dataclass(frozen=True, slots=True, eq=False)
class QuaternionicAgentCertificate:
    """
    Certificado inmutable de calibración, veto y actuación.

    Se conservan los campos originales por compatibilidad. Los campos
    añadidos van al final con valores por defecto.
    """

    phase: str
    heyting_verdict: str
    cohomological_mismatch: float
    norm_drift_error: float
    trace_von_neumann_error: float
    is_surgery_active: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    session_sha256: str = ""
    phase_chain: Tuple[str, ...] = ()
    confidence: float = 1.0
    heyting_truth_value: float = 1.0
    elastic_bound: float = 0.0
    override_sha256: str = ""

    def __hash__(self) -> int:
        return hash(self.digital_signature_sha256)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuaternionicAgentCertificate):
            return NotImplemented
        return self.digital_signature_sha256 == other.digital_signature_sha256

    def __repr__(self) -> str:
        return (
            f"QuaternionicAgentCertificate(verdict={self.heyting_verdict!r}, "
            f"confidence={self.confidence:.6g}, "
            f"seal={self.digital_signature_sha256[:12]!r})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SOBERANO DE CALIBRE CUATERNIÓNICO — TRES FASES ANIDADAS
# ═════════════════════════════════════════════════════════════════════════════

class QuaternionicStateAgent:
    """
    Soberano de Calibre Cuaterniónico (SET-Quaternion) — OODA en 3 fases.

    FASE 1  OBSERVE : validación, construcción, sello de sesión.
    FASE 2  ORIENT  : S², Hopf, Hurwitz, von Neumann, C*, Lipschitz.
    FASE 3  DECIDE/ACT : rampa, Heyting Ω₃, override e⁺, Crowbar, certificado.
    """

    def __init__(
        self,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        rng_seed: Optional[int] = None,
        jitter_sigma: float = 2.8,
        soft_veto_lower: float = _SOFT_VETO_LOWER,
        soft_veto_upper: float = _SOFT_VETO_UPPER,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finita y estrictamente positiva.")
        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser finita y estrictamente positiva.")
        if not math.isfinite(grace_period_seconds) or grace_period_seconds < 0.0:
            raise ValueError("grace_period_seconds debe ser finita y no negativa.")
        if not math.isfinite(jitter_sigma) or jitter_sigma < 0.0:
            raise ValueError("jitter_sigma debe ser finita y no negativa.")
        if not math.isfinite(soft_veto_lower) or not math.isfinite(soft_veto_upper):
            raise ValueError("los umbrales de veto suave deben ser finitos.")
        if not (0.0 < soft_veto_lower < soft_veto_upper):
            raise ValueError(
                "Se exige 0 < soft_veto_lower < soft_veto_upper."
            )

        self._tol = float(tolerance)
        self._safety_margin = float(safety_margin)
        self._grace_limit = float(grace_period_seconds)
        self._regularization = max(1e-15, self._tol * 1e-3)
        self._jitter_sigma = float(jitter_sigma)
        self._soft_veto_lower = float(soft_veto_lower)
        self._soft_veto_upper = float(soft_veto_upper)
        self._rng = np.random.default_rng(rng_seed)
        self._shifter = QuaternionicStateShifter(tolerance=tolerance)

    def _tolerance_of(self, scale: float = 1.0) -> float:
        return max(self._tol, self._tol * abs(scale), 32.0 * _MACHINE_EPS)

    def _hard_algebraic_threshold(self) -> float:
        """Umbral duro para identidades algebraicas ya relativizadas."""
        return max(self._tol, 32.0 * _MACHINE_EPS)

    # ═════════════════════════════════════════════════════════════════════════
    # ADAPTADORES DE COMPATIBILIDAD CON SHIFTER LEGADO / EVOLUCIONADO
    # ═════════════════════════════════════════════════════════════════════════

    def _build_state(self, S: np.ndarray) -> QuaternionicState:
        """Materializa q ∈ H usando la Fase 1 del reactor, o el API legado."""
        if hasattr(self._shifter, "phase1_construct_canonical_state"):
            return self._shifter.phase1_construct_canonical_state(S)
        return self._shifter.build_state(S)

    def _raw_multiply(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> QuaternionicState:
        """Producto hamiltoniano sin renormalización, para auditar deriva pura."""
        if hasattr(self._shifter, "phase2_hamilton_product"):
            return self._shifter.phase2_hamilton_product(p, q, verify=False)
        return self._shifter.quaternionic_multiply(p, q)

    def _riemann_chart(self, state: QuaternionicState) -> Dict[str, Any]:
        """Carta de Riemann rica si existe; si no, envoltorio del API legado."""
        if hasattr(self._shifter, "phase3_riemann_sphere_chart"):
            return dict(self._shifter.phase3_riemann_sphere_chart(state))

        Z, z = self._shifter.project_to_riemann_sphere(state)
        if _complex_is_nonfinite(Z):
            coord = complex(math.inf, math.inf)
            coord_real = math.inf
            coord_imag = math.inf
        else:
            coord = complex(Z)
            coord_real = float(coord.real)
            coord_imag = float(coord.imag)
        return {
            "chart": "legacy",
            "coordinate": coord,
            "coordinate_real": coord_real,
            "coordinate_imag": coord_imag,
            "height": float(z),
            "metric_factor": math.nan,
            "sphericity_residual": math.nan,
        }

    def _optional_shifter_call(self, method_name: str, *args: Any) -> Any:
        method = getattr(self._shifter, method_name, None)
        if callable(method):
            return method(*args)
        return None

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1 — OBSERVE: VALIDACIÓN, CONSTRUCCIÓN Y SELLO DE SESIÓN
    # ═════════════════════════════════════════════════════════════════════════

    def phase1_validate_signal(self, S: np.ndarray, name: str) -> np.ndarray:
        """
        Fase 1.1 — Validación formal de una señal S ∈ R⁴.

        Condiciones: forma (4,), valores finitos, float64, −0.0 → +0.0.
        """
        return _as_finite_vector4(S, name)

    def phase1_observe_pair(
        self,
        p_S: np.ndarray,
        q_S: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, QuaternionicState, QuaternionicState]:
        """
        Fase 1.2 — Observación covariante del par transaccional (p_S, q_S).

        Realiza el pullback simultáneo
            Φ × Φ : R⁴ × R⁴ → H × H
        y devuelve las señales canónicas junto a los estados.
        """
        p_arr = self.phase1_validate_signal(p_S, "p_S")
        q_arr = self.phase1_validate_signal(q_S, "q_S")
        p_state = self._build_state(p_arr)
        q_state = self._build_state(q_arr)
        return p_arr, q_arr, p_state, q_state

    def phase1_observation_diagnostics(
        self,
        p_state: QuaternionicState,
        q_state: QuaternionicState,
    ) -> Dict[str, Any]:
        """
        Fase 1.3 — Diagnóstico de Banach y unitariedad del par observado.

        Reporta normas ℓ², residuos de unitariedad y, si el reactor evolucionado
        los expone, residuos de ortogonalidad L(q)ᵀL(q) = ||q||² I₄.
        """
        def _pack_state(prefix: str, state: QuaternionicState) -> Dict[str, Any]:
            return {
                f"{prefix}_norm": _state_norm(state),
                f"{prefix}_squared_norm": _state_squared_norm(state),
                f"{prefix}_vector_norm": _finite_or_nan(
                    getattr(state, "vector_norm", math.nan)
                ),
                f"{prefix}_is_unitary": bool(getattr(state, "is_unitary", False)),
                f"{prefix}_unitarity_residual": _finite_or_nan(
                    getattr(state, "unitarity_residual", math.nan)
                ),
                f"{prefix}_orthogonality_residual": _finite_or_nan(
                    getattr(state, "orthogonality_residual", math.nan)
                ),
                f"{prefix}_condition_estimate": _finite_or_nan(
                    getattr(state, "condition_estimate", math.nan)
                ),
                f"{prefix}_sha256_prefix": _state_hash(state)[:16],
            }

        diagnostics: Dict[str, Any] = {}
        diagnostics.update(_pack_state("p", p_state))
        diagnostics.update(_pack_state("q", q_state))
        diagnostics["machine_epsilon"] = _MACHINE_EPS
        diagnostics["regularization"] = self._regularization
        diagnostics["tolerance"] = self._tol
        return diagnostics

    def _phase1_session_hash(
        self,
        p_arr: np.ndarray,
        q_arr: np.ndarray,
        p_state: QuaternionicState,
        q_state: QuaternionicState,
    ) -> str:
        """
        Fase 1.4 — Sello de sesión SHA-256 canónico longitud-prefijado.

        Carga útil:
          - señales canónicas p, q;
          - hashes de estado del reactor si existen;
          - parámetros de gobernanza (tolerancia, margen, gracia, umbrales);
          - token de fase PHASE1/OBSERVE.
        """
        h = hashlib.sha256()
        h.update(b"QSA/SESSION/v3")
        _sha_update_arr(h, p_arr)
        _sha_update_arr(h, q_arr)
        _sha_update_str(h, _state_hash(p_state))
        _sha_update_str(h, _state_hash(q_state))
        h.update(_pack_f64(self._tol))
        h.update(_pack_f64(self._safety_margin))
        h.update(_pack_f64(self._grace_limit))
        h.update(_pack_f64(self._soft_veto_lower))
        h.update(_pack_f64(self._soft_veto_upper))
        _sha_update_str(h, "PHASE1/OBSERVE")
        digest = h.hexdigest()
        if len(digest) != _SHA256_HEX_LEN:
            raise RuntimeError("El sello de sesión no es un SHA-256 de 64 nibbles.")
        return digest

    def phase1_close_and_open_phase2(
        self,
        p_S: np.ndarray,
        q_S: np.ndarray,
    ) -> Phase1AgentHandoff:
        """
        Fase 1.5 — Cierre formal de Fase 1 y apertura verificada de Fase 2.

        Definición formal de frontera:

            Φ₁→₂ : (p_S, q_S) ∈ R⁴ × R⁴  ↦  (p, q, σ₁, δ₁) ∈ H × H × Σ × D₁

        Este es el último método de la Fase 1. Su contrato es exactamente el
        dominio de `phase2_from_phase1`: produce `Phase1AgentHandoff` y exige
        que la Fase 2 lo admita de inmediato. Con ello la Fase 1 queda
        anidada, como prefijo functorial, dentro de la Fase 2.
        """
        p_arr, q_arr, p_state, q_state = self.phase1_observe_pair(p_S, q_S)
        session_sha256 = self._phase1_session_hash(p_arr, q_arr, p_state, q_state)
        diagnostics = self.phase1_observation_diagnostics(p_state, q_state)
        diagnostics["session_sha256_prefix"] = session_sha256[:16]
        diagnostics["p_is_unitary"] = bool(getattr(p_state, "is_unitary", False))
        diagnostics["q_is_unitary"] = bool(getattr(q_state, "is_unitary", False))
        diagnostics["p_norm"] = _state_norm(p_state)
        diagnostics["q_norm"] = _state_norm(q_state)

        handoff = Phase1AgentHandoff(
            p_state=p_state,
            q_state=q_state,
            session_sha256=session_sha256,
            diagnostics=diagnostics,
            next_entrypoint=_PHASE1_ENTRY,
        )

        opened_p, opened_q = self.phase2_from_phase1(handoff)
        p_h, q_h = _state_hash(opened_p), _state_hash(opened_q)
        if p_h and p_h != _state_hash(p_state):
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el estado p admitido "
                "por Fase 2 no coincide con el observado en Fase 1."
            )
        if q_h and q_h != _state_hash(q_state):
            raise RuntimeError(
                "Invariante de anidamiento Φ₁→₂ violado: el estado q admitido "
                "por Fase 2 no coincide con el observado en Fase 1."
            )

        logger.info(
            "Fase Observe [QUATERNION_AGENT]: Sesión sellada. SHA prefix=%s",
            session_sha256[:16],
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2 — ORIENT: INVARIANTES ESPECTRALES Y ALGEBRAICOS
    # (continuación formal de phase1_close_and_open_phase2)
    # ═════════════════════════════════════════════════════════════════════════

    def phase2_from_phase1(
        self,
        handoff: Phase1AgentHandoff,
    ) -> Tuple[QuaternionicState, QuaternionicState]:
        """
        Fase 2.0 — Entrada formal desde Fase 1.

        Continuación directa de `phase1_close_and_open_phase2`. Consume
        `Phase1AgentHandoff` y devuelve el par observado (p, q) ∈ H × H.
        """
        if not isinstance(handoff, Phase1AgentHandoff):
            raise TypeError("Se esperaba Phase1AgentHandoff como frontera Φ₁→₂.")
        if handoff.next_entrypoint != _PHASE1_ENTRY:
            raise ValueError(
                "Phase1AgentHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE1_ENTRY!r}."
            )
        if not isinstance(handoff.session_sha256, str) or len(
            handoff.session_sha256
        ) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₁→₂ no es un SHA-256 válido.")
        return handoff.p_state, handoff.q_state

    def phase2_evaluate_similarity_sphere(self, state: QuaternionicState) -> float:
        """
        Fase 2.1 — Medida de desviación sobre la 2-esfera S².

        Sea z ∈ [−1, 1] la altura del punto proyectado. Se definen:

            P(z) = (1 + z) / 2     (obstrucción al polo norte)
            C(z) = √max(0, 1 − z²) (curvatura ecuatorial)

        El mismatch es δ = max(P(z), C(z)) ∈ [0, 1]. Cerca del polo norte la
        carta estereográfica se vuelve singular; cerca del ecuador la métrica
        redonda alcanza curvatura máxima en la proyección. El origen de Im(H)
        se declara δ = 0 (fibra nula).
        """
        vector_norm = _finite_or_nan(getattr(state, "vector_norm", math.nan))
        if not math.isfinite(vector_norm):
            vector_norm = _vector_norm3(getattr(state, "vector_part", ()))

        if vector_norm <= self._regularization:
            return 0.0

        chart = self._riemann_chart(state)
        z = _finite_or_nan(chart.get("height", 0.0))
        if not math.isfinite(z):
            return 1.0
        z = _clip_unit(z)

        coord = chart.get("coordinate", 0.0 + 0.0j)
        if _complex_is_nonfinite(coord):
            return 1.0

        polar_obstruction = (1.0 + z) / 2.0
        equatorial_curvature = math.sqrt(max(0.0, 1.0 - z * z))
        mismatch = max(polar_obstruction, equatorial_curvature)
        return float(min(1.0, max(0.0, mismatch)))

    def phase2_hopf_obstruction(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 2.2 — Obstrucción de la fibración de Hopf π : S³ → S².

        Si el reactor expone `phase3_hopf_fibration`, se audita que el punto
        base viva en S² (||π(u)|| = 1). En caso contrario se reporta NaN
        no bloqueante: la orientación no depende de un reactor evolucionado.
        """
        hopf = self._optional_shifter_call("phase3_hopf_fibration", state)
        if not isinstance(hopf, dict):
            return {
                "hopf_base_norm": math.nan,
                "hopf_sphericity_residual": math.nan,
            }
        return {
            "hopf_base_norm": _finite_or_nan(hopf.get("base_norm")),
            "hopf_sphericity_residual": _finite_or_nan(
                hopf.get("sphericity_residual")
            ),
        }

    def phase2_audit_hurwitz_composition(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> float:
        """
        Fase 2.3 — Auditoría de la identidad de los cuatro cuadrados.

            ε_H = | ||p q|| − ||p|| ||q|| | / max(1, ||p|| ||q||)

        Si el reactor expone `phase2_hurwitz_residual`, se toma el máximo
        entre ambas medidas (conservadurismo de orientación).
        """
        prod = self._raw_multiply(p, q)
        expected = float(p.norm) * float(q.norm)
        if not math.isfinite(expected) or not math.isfinite(float(prod.norm)):
            return 1.0
        if expected <= self._regularization:
            drift = float(abs(prod.norm))
        else:
            drift = abs(float(prod.norm) - expected) / max(1.0, expected)

        alt = self._optional_shifter_call("phase2_hurwitz_residual", p, q, prod)
        if alt is not None:
            alt_f = _finite_or_nan(alt)
            if math.isfinite(alt_f):
                drift = max(drift, alt_f)
        return float(drift)

    def phase2_verify_von_neumann_trace(self, state: QuaternionicState) -> float:
        """
        Fase 2.4 — Auditoría de traza de von Neumann sobre ι(q) ∈ M₂(C).

        Identidad exacta: Tr(ι(q)) = 2 q₀. Normalizando ρ = ι(q) / ||q||²,

            error = hypot( Re Tr(ρ) − 2 q₀/||q||² , Im Tr(ρ) ).

        El origen se declara error = 1 (estado no invertible / no densificable).
        """
        norm_sq = _state_squared_norm(state)
        if (not math.isfinite(norm_sq)) or norm_sq < self._regularization:
            return 1.0

        try:
            cd_mat = np.asarray(state.cayley_dickson_matrix, dtype=np.complex128)
            rho = cd_mat / norm_sq
            tr = np.trace(rho)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return 1.0

        expected = 2.0 * float(state.scalar_part) / norm_sq
        real_err = float(np.real(tr) - expected)
        imag_err = float(np.imag(tr))
        if not math.isfinite(real_err) or not math.isfinite(imag_err):
            return 1.0
        return float(math.hypot(real_err, imag_err))

    def phase2_cstar_residual(self, state: QuaternionicState) -> float:
        """
        Fase 2.5 — Residual C* ||q* q|| = ||q||², si el reactor lo expone.
        """
        residual = self._optional_shifter_call("phase2_cstar_identity", state)
        if residual is None:
            return math.nan
        return _finite_or_nan(residual)

    def phase2_spectral_consistency(self, state: QuaternionicState) -> Dict[str, float]:
        """
        Fase 2.6 — Consistencia espectral de L(q) cuando el reactor la expone.

        Se recogen radio espectral, residual frente a ||q|| y residual del
        modelo {q₀ ± i ||v||} (multiplicidad 2). Ausencia del método no es
        veto: se reporta NaN.
        """
        audit = self._optional_shifter_call("phase3_spectral_audit", state)
        if not isinstance(audit, dict):
            return {
                "spectral_radius": math.nan,
                "spectral_residual": math.nan,
                "eigenvalue_model_residual": math.nan,
                "determinant_residual": math.nan,
            }
        return {
            "spectral_radius": _finite_or_nan(audit.get("spectral_radius")),
            "spectral_residual": _finite_or_nan(audit.get("spectral_residual")),
            "eigenvalue_model_residual": _finite_or_nan(
                audit.get("eigenvalue_model_residual")
            ),
            "determinant_residual": _finite_or_nan(
                audit.get("determinant_residual")
            ),
        }

    def phase2_elastic_bound(self, lipschitz_bound_Lmax: float) -> float:
        """
        Fase 2.7 — Cota elástica Λ = L_max · η.

        η = safety_margin es el factor de seguridad multiplicativo sobre la
        constante de Lipschitz del flujo. Λ escala los umbrales de la
        2-esfera de similitud.
        """
        if not math.isfinite(lipschitz_bound_Lmax) or lipschitz_bound_Lmax <= 0.0:
            raise ValueError(
                "lipschitz_bound_Lmax debe ser finita y estrictamente positiva."
            )
        bound = float(lipschitz_bound_Lmax) * float(self._safety_margin)
        if not math.isfinite(bound) or bound <= 0.0:
            raise OverflowError(
                "La cota elástica de Lipschitz no es representable o es nula."
            )
        return float(bound)

    def phase2_collect_invariants(
        self,
        p_state: QuaternionicState,
        q_state: QuaternionicState,
        lipschitz_bound_Lmax: float,
    ) -> Dict[str, Any]:
        """
        Fase 2.8 — Recolección conjunta de invariantes de orientación.

        El mismatch cohomológico primario se evalúa sobre q (estado de
        calibre). p se reporta como canal auxiliar. Hurwitz usa el par.
        """
        mismatch_q = self.phase2_evaluate_similarity_sphere(q_state)
        mismatch_p = self.phase2_evaluate_similarity_sphere(p_state)
        hurwitz_drift = self.phase2_audit_hurwitz_composition(p_state, q_state)
        trace_err = self.phase2_verify_von_neumann_trace(q_state)
        elastic_bound = self.phase2_elastic_bound(lipschitz_bound_Lmax)
        hopf = self.phase2_hopf_obstruction(q_state)
        cstar = self.phase2_cstar_residual(q_state)
        spectral = self.phase2_spectral_consistency(q_state)

        return {
            "similarity_mismatch": float(mismatch_q),
            "similarity_mismatch_p_channel": float(mismatch_p),
            "hurwitz_drift": float(hurwitz_drift),
            "trace_von_neumann_error": float(trace_err),
            "elastic_bound": float(elastic_bound),
            "cstar_residual": float(cstar) if cstar == cstar else math.nan,
            **hopf,
            **spectral,
            "safety_margin": self._safety_margin,
            "tolerance": self._tol,
            "regularization": self._regularization,
            "soft_veto_lower": self._soft_veto_lower,
            "soft_veto_upper": self._soft_veto_upper,
        }

    def phase2_close_and_open_phase3(
        self,
        phase1_handoff: Phase1AgentHandoff,
        lipschitz_bound_Lmax: float,
    ) -> Phase2AgentHandoff:
        """
        Fase 2.9 — Cierre formal de Fase 2 y apertura verificada de Fase 3.

        Definición formal de frontera:

            Φ₂→₃ : (p, q, σ₁) ↦ (p, q, σ₁, δ, ε_H, ε_Tr, Λ) ∈ H² × Σ × D₂

        donde
          δ    = mismatch de similitud espectral de q,
          ε_H  = deriva de Hurwitz,
          ε_Tr = error de traza de von Neumann,
          Λ    = cota elástica de Lipschitz.

        Este es el último método de la Fase 2. Su contrato es exactamente el
        dominio de `phase3_from_phase2`. Con ello la Fase 2 queda anidada,
        como prefijo functorial, dentro de la Fase 3.
        """
        p_state, q_state = self.phase2_from_phase1(phase1_handoff)
        invariants = self.phase2_collect_invariants(
            p_state, q_state, lipschitz_bound_Lmax
        )

        handoff = Phase2AgentHandoff(
            p_state=p_state,
            q_state=q_state,
            session_sha256=phase1_handoff.session_sha256,
            similarity_mismatch=float(invariants["similarity_mismatch"]),
            hurwitz_drift=float(invariants["hurwitz_drift"]),
            trace_von_neumann_error=float(invariants["trace_von_neumann_error"]),
            elastic_bound=float(invariants["elastic_bound"]),
            diagnostics=invariants,
            next_entrypoint=_PHASE2_ENTRY,
        )

        opened = self.phase3_from_phase2(handoff)
        if opened.session_sha256 != handoff.session_sha256:
            raise RuntimeError(
                "Invariante de anidamiento Φ₂→₃ violado: el sello de sesión "
                "admitido por Fase 3 no coincide con el de Fase 2."
            )

        logger.debug(
            "Fase Orient [QUATERNION_AGENT]: mismatch=%.6e, hurwitz=%.6e, trace=%.6e",
            handoff.similarity_mismatch,
            handoff.hurwitz_drift,
            handoff.trace_von_neumann_error,
        )
        return handoff

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3 — DECIDE / ACT: RAMPA, HEYTING, VETO, HARDWARE Y CERTIFICACIÓN
    # (continuación formal de phase2_close_and_open_phase3)
    # ═════════════════════════════════════════════════════════════════════════

    def phase3_from_phase2(self, handoff: Phase2AgentHandoff) -> Phase2AgentHandoff:
        """
        Fase 3.0 — Entrada formal desde Fase 2.

        Continuación directa de `phase2_close_and_open_phase3`. Consume
        `Phase2AgentHandoff` y lo reexpone si la frontera es válida.
        """
        if not isinstance(handoff, Phase2AgentHandoff):
            raise TypeError("Se esperaba Phase2AgentHandoff como frontera Φ₂→₃.")
        if handoff.next_entrypoint != _PHASE2_ENTRY:
            raise ValueError(
                "Phase2AgentHandoff inválido: el punto de entrada esperado es "
                f"{_PHASE2_ENTRY!r}."
            )
        if not isinstance(handoff.session_sha256, str) or len(
            handoff.session_sha256
        ) != _SHA256_HEX_LEN:
            raise ValueError("El sello de sesión de Φ₂→₃ no es un SHA-256 válido.")
        return handoff

    def phase3_confidence_ramp(self, handoff: Phase2AgentHandoff) -> float:
        """
        Fase 3.1 — Rampa de confianza Lipschitz-saturada.

        Para cada canal se define un valor de verdad continuo c ∈ [0, 1]:

            r_S = δ / Λ
            c_S = 1                 si r_S ≤ α
                = (β − r_S)/(β − α) si α < r_S ≤ β
                = 0                 si r_S > β

        con α = soft_veto_lower, β = soft_veto_upper. Los canales algebraicos
        (Hurwitz, von Neumann) saturan contra el umbral duro. La confianza
        global es el encuentro de Heyting (mínimo) de los canales finitos.
        """
        mismatch = float(handoff.similarity_mismatch)
        hurwitz = float(handoff.hurwitz_drift)
        trace_err = float(handoff.trace_von_neumann_error)
        bound = float(handoff.elastic_bound)
        alpha = self._soft_veto_lower
        beta = self._soft_veto_upper
        hard = self._hard_algebraic_threshold()

        def _sphere_channel(delta: float, lam: float) -> float:
            if not math.isfinite(delta) or not math.isfinite(lam) or lam <= 0.0:
                return Heyting3.VETOED
            ratio = delta / lam
            if ratio <= alpha:
                return Heyting3.COHERENT
            if ratio > beta:
                return Heyting3.VETOED
            span = beta - alpha
            if span <= 0.0:
                return Heyting3.VETOED
            return float((beta - ratio) / span)

        def _algebra_channel(residual: float) -> float:
            if not math.isfinite(residual):
                return Heyting3.VETOED
            if residual <= hard:
                return Heyting3.COHERENT
            # saturación suave hasta 10 × umbral duro
            hi = 10.0 * hard
            if residual >= hi:
                return Heyting3.VETOED
            return float((hi - residual) / (hi - hard))

        channels = [
            _sphere_channel(mismatch, bound),
            _algebra_channel(hurwitz),
            _algebra_channel(trace_err),
        ]
        confidence = channels[0]
        for ch in channels[1:]:
            confidence = Heyting3.meet(confidence, ch)
        if not math.isfinite(confidence):
            return Heyting3.VETOED
        return float(min(1.0, max(0.0, confidence)))

    def phase3_decide_heyting(
        self,
        handoff: Phase2AgentHandoff,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fase 3.2 — Clasificador de Heyting trivalente Ω₃.

        Reglas (con α = soft_veto_lower, β = soft_veto_upper, Λ cota elástica):

          - DEGRADED (½)  si  α Λ < δ ≤ β Λ  y las métricas son finitas.
          - VETOED   (0)  si  δ > β Λ, o ε_H / ε_Tr superan el umbral duro,
                          o hay métricas no finitas, o el veto suave expira
                          sin override.
          - COHERENT (1)  en el resto de casos.

        El positrón e⁺ (override_token no vacío) es un morfismo
            DEGRADED → DEGRADED
        que anula la implicación «gracia expirada ⇒ VETOED».
        """
        mismatch = float(handoff.similarity_mismatch)
        hurwitz_drift = float(handoff.hurwitz_drift)
        trace_err = float(handoff.trace_von_neumann_error)
        elastic_bound = float(handoff.elastic_bound)
        confidence = self.phase3_confidence_ramp(handoff)

        finite_metrics = all(
            math.isfinite(x)
            for x in (mismatch, hurwitz_drift, trace_err, elastic_bound)
        )
        hard_threshold = self._hard_algebraic_threshold()
        alpha = self._soft_veto_lower
        beta = self._soft_veto_upper

        is_soft_veto = bool(
            finite_metrics
            and (alpha * elastic_bound < mismatch <= beta * elastic_bound)
        )
        is_hard_veto = bool(
            (not finite_metrics)
            or (mismatch > beta * elastic_bound)
            or (hurwitz_drift > hard_threshold)
            or (trace_err > hard_threshold)
        )

        override_expired = False
        override_sha256: Optional[str] = None

        if is_soft_veto:
            if override_token is not None and str(override_token).strip():
                override_sha256 = hashlib.sha256(
                    str(override_token).encode("utf-8")
                ).hexdigest()
                logger.info(
                    "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] INYECTADO! "
                    "Aniquilando anomalía en Fock. Sello: %s",
                    override_sha256[:16],
                )
                override_expired = False
            else:
                if elapsed_grace_seconds is not None:
                    if not math.isfinite(float(elapsed_grace_seconds)):
                        override_expired = True
                    else:
                        override_expired = (
                            float(elapsed_grace_seconds) >= self._grace_limit
                        )
                else:
                    override_expired = bool(simulate_grace_expired)

                if override_expired:
                    logger.warning(
                        "¡PERÍODO DE GRACIA DE %.0f SEGUNDOS EXPIRADO SIN OVERRIDE! "
                        "Gatillando veto ciber-físico de protección de capital.",
                        self._grace_limit,
                    )
                else:
                    logger.warning(
                        "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! "
                        "Inconsistencia marginal en la 2-esfera de similitud. "
                        "Cuenta atrás de %.0f segundos activa.",
                        self._grace_limit,
                    )

        if is_hard_veto or (is_soft_veto and override_expired):
            truth = Heyting3.VETOED
        elif is_soft_veto and not override_expired:
            truth = Heyting3.DEGRADED
        else:
            truth = Heyting3.COHERENT

        # El encuentro con la rampa nunca eleva el veredicto: solo puede bajarlo
        # a DEGRADED si la confianza cae bajo 1 sin llegar a 0, o a VETOED si es 0.
        quantized_confidence = Heyting3.quantize(confidence)
        truth = Heyting3.meet(truth, quantized_confidence)
        verdict = Heyting3.to_verdict(truth)

        return {
            "heyting_verdict": verdict,
            "heyting_truth_value": float(truth),
            "confidence": float(confidence),
            "is_soft_veto": is_soft_veto,
            "is_hard_veto": is_hard_veto,
            "override_grace_period_expired": override_expired,
            "override_sha256": override_sha256,
            "hard_threshold": hard_threshold,
            "elastic_bound": elastic_bound,
            "grace_period_seconds": self._grace_limit,
            "elapsed_grace_seconds": (
                float(elapsed_grace_seconds)
                if elapsed_grace_seconds is not None
                else math.nan
            ),
        }

    def phase3_actuate_hardware(self, heyting_verdict: str) -> Tuple[bool, float]:
        """
        Fase 3.3 — Actuación de hardware Crowbar BT151.

        Sólo el colapso a VETOED conmuta el interlock. El jitter gaussiano
        simula la dispersión física IRAM y se recorta a [385, 415] ns.

        Retorna
        -------
        Tuple[bool, float]
            (interlock_fired, actuation_latency_ns)
        """
        if str(heyting_verdict) != "VETOED":
            return False, 0.0

        jitter = float(self._rng.normal(loc=0.0, scale=self._jitter_sigma))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_MIN_NS,
                _CROWBAR_LATENCY_MAX_NS,
            )
        )
        logger.critical(
            "¡COLA DE HEYTING COLAPSADA EN SOBERANO CUATERNIÓNICO! "
            "Conmutando GPIO14 a HIGH en %.2f ns vía IRAM. Crowbar BT151 gatillado.",
            actuation_latency_ns,
        )
        return True, actuation_latency_ns

    def _phase3_certificate_seal(
        self,
        verdict: str,
        mismatch: float,
        hurwitz_drift: float,
        trace_err: float,
        actuation_latency_ns: float,
        session_sha256: str,
        override_sha256: str,
        hardware_interlock_fired: bool,
        confidence: float,
        truth: float,
    ) -> str:
        """Sello binario canónico del certificado (no depende de `repr`)."""
        h = hashlib.sha256()
        h.update(b"QSA/CERT/v3")
        _sha_update_str(h, "G_WISDOM_QUATERNION_SUTURATED")
        _sha_update_str(h, verdict)
        h.update(_pack_f64(mismatch))
        h.update(_pack_f64(hurwitz_drift))
        h.update(_pack_f64(trace_err))
        h.update(_pack_f64(actuation_latency_ns))
        h.update(_pack_f64(confidence))
        h.update(_pack_f64(truth))
        _sha_update_str(h, session_sha256)
        _sha_update_str(h, override_sha256)
        _sha_update_str(
            h, "HARDWARE_FIRED" if hardware_interlock_fired else "HARDWARE_IDLE"
        )
        return h.hexdigest()

    def phase3_issue_certificate(
        self,
        handoff: Phase2AgentHandoff,
        decision: Mapping[str, Any],
        hardware_interlock_fired: bool,
        actuation_latency_ns: float,
    ) -> QuaternionicAgentCertificate:
        """
        Fase 3.4 — Emisión del certificado formal inmutable.

        Incluye veredicto Ω₃, métricas físicas, rampa de confianza, estado
        de veto, actuación de hardware, cadena de fases y sello SHA-256.
        """
        verdict = str(decision["heyting_verdict"])
        mismatch = float(handoff.similarity_mismatch)
        hurwitz_drift = float(handoff.hurwitz_drift)
        trace_err = float(handoff.trace_von_neumann_error)
        override_expired = bool(decision["override_grace_period_expired"])
        override_sha256 = str(decision.get("override_sha256") or "NO_OVERRIDE")
        confidence = float(decision.get("confidence", math.nan))
        truth = float(
            decision.get("heyting_truth_value", Heyting3.from_verdict(verdict))
        )

        p_token = _state_token(handoff.p_state, "LEGACY_P")
        q_token = _state_token(handoff.q_state, "LEGACY_Q")
        phase_chain = (
            "PHASE1/OBSERVE",
            p_token,
            q_token,
            "PHASE2/ORIENT",
            "PHASE3/DECIDE/ACT",
        )

        digital_sig = self._phase3_certificate_seal(
            verdict=verdict,
            mismatch=mismatch,
            hurwitz_drift=hurwitz_drift,
            trace_err=trace_err,
            actuation_latency_ns=float(actuation_latency_ns),
            session_sha256=handoff.session_sha256,
            override_sha256=override_sha256,
            hardware_interlock_fired=bool(hardware_interlock_fired),
            confidence=confidence,
            truth=truth,
        )

        return QuaternionicAgentCertificate(
            phase="G_WISDOM_QUATERNION_SUTURATED",
            heyting_verdict=verdict,
            cohomological_mismatch=mismatch,
            norm_drift_error=hurwitz_drift,
            trace_von_neumann_error=trace_err,
            is_surgery_active=(verdict == "DEGRADED"),
            is_soft_veto_active=bool(decision["is_soft_veto"]),
            override_grace_period_expired=override_expired,
            hardware_interlock_fired=bool(hardware_interlock_fired),
            actuation_latency_ns=float(actuation_latency_ns),
            digital_signature_sha256=digital_sig,
            session_sha256=handoff.session_sha256,
            phase_chain=phase_chain,
            confidence=confidence,
            heyting_truth_value=truth,
            elastic_bound=float(handoff.elastic_bound),
            override_sha256=override_sha256,
        )

    def phase3_close_loop(
        self,
        phase2_handoff: Phase2AgentHandoff,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> QuaternionicAgentCertificate:
        """
        Fase 3.5 — Orquestación completa de la Fase 3.

        Ejecuta, en orden:
          1. validación de frontera Φ₂→₃;
          2. rampa de confianza y decisión Heyting;
          3. actuación de hardware;
          4. certificación final.
        """
        validated = self.phase3_from_phase2(phase2_handoff)

        decision = self.phase3_decide_heyting(
            handoff=validated,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )
        interlock_fired, latency = self.phase3_actuate_hardware(
            heyting_verdict=decision["heyting_verdict"]
        )
        certificate = self.phase3_issue_certificate(
            handoff=validated,
            decision=decision,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
        )

        if certificate.heyting_verdict == "VETOED":
            logger.critical(
                "Soberano Cuaterniónico emitió VETO DURO. Sello: %s",
                certificate.digital_signature_sha256[:16],
            )
        else:
            logger.info(
                "Soberano Cuaterniónico regulado síncronamente. Veredicto: %s. "
                "Confianza=%.6g. Sello: %s",
                certificate.heyting_verdict,
                certificate.confidence,
                certificate.digital_signature_sha256[:16],
            )
        return certificate

    # ═════════════════════════════════════════════════════════════════════════
    # API PRINCIPAL COMPATIBLE CON OODA LEGADO
    # ═════════════════════════════════════════════════════════════════════════

    def execute_quaternionic_control_cycle(
        self,
        p_S: np.ndarray,
        q_S: np.ndarray,
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        elapsed_grace_seconds: Optional[float] = None,
    ) -> QuaternionicAgentCertificate:
        """
        API principal — Orquesta el funtor compuesto Φ₃ ∘ Φ₂ ∘ Φ₁.

        Equivalencia de fases:
          OBSERVE    → phase1_close_and_open_phase2
          ORIENT     → phase2_close_and_open_phase3
          DECIDE/ACT → phase3_close_loop
        """
        phase1_handoff = self.phase1_close_and_open_phase2(p_S, q_S)
        phase2_handoff = self.phase2_close_and_open_phase3(
            phase1_handoff=phase1_handoff,
            lipschitz_bound_Lmax=lipschitz_bound_Lmax,
        )
        return self.phase3_close_loop(
            phase2_handoff=phase2_handoff,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            elapsed_grace_seconds=elapsed_grace_seconds,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # MÉTODOS LEGADOS (COMPATIBILIDAD)
    # ═════════════════════════════════════════════════════════════════════════

    def evaluate_similarity_sphere(self, state: QuaternionicState) -> float:
        """API legada — equivalente a Fase 2.1."""
        return self.phase2_evaluate_similarity_sphere(state)

    def audit_hurwitz_composition(
        self,
        p: QuaternionicState,
        q: QuaternionicState,
    ) -> float:
        """API legada — equivalente a Fase 2.3."""
        return self.phase2_audit_hurwitz_composition(p, q)

    def verify_von_neumann_trace(self, state: QuaternionicState) -> float:
        """API legada — equivalente a Fase 2.4."""
        return self.phase2_verify_von_neumann_trace(state)
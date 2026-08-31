# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Topological Surgery Čech Agent (Soberano de Cirugía de Čech)        ║
║ Ruta   : app/agents/wisdom/topological_surgery_cech_agent.py                 ║
║ Versión: 1.1.0-Doctoral-OODA-Heyting-Cech-Anisotropic-ESP32-Secure           ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ ($V_{\mathbb{W}}$, Nivel 0) gobernando de forma activa y asíncrona la        ║
║ "Cirugía Topológica de Čech" [topological_surgery_cech.py]. Inyecta          ║
║ coexcitaciones de Čech para detectar obstrucciones de cohomología local      ║
║ (mismatch de primer orden $\check{H}^1$) inducidas por ruido analógico.      ║
║ Orquesta el ciclo covariante OODA y administra la "Rampa de Confianza"       ║
║ graduando de forma inmune la censura entre Veto Suave (Luz Ámbar) y Veto     ║
║ Duro de Silicio (Crowbar BT151 inyectado en IRAM < 400 ns), protegiendo la   ║
║ maquinaria civil frente a falsas paradas innecesarias.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES DE LA GOBERNANZA AGÉNTICA (Teoría de Topos y Cirugía de Čech)
================================════════════════════════════════════════════════

Definición 1 (El Topos de Haces de de Rham-Čech-Heyting):
  Sea $K$ el complejo simplicial del presupuesto y $\mathcal{U} = \{U_i\}$ su cubrimiento.
  Definimos el topos de haces topológicos localizados $\mathbf{Sh}(\partial K, \, \Omega_3)$ sobre
  la frontera compacta $\partial K$. El clasificador de subobjetos de tres valores
  ordinales se define de forma rigurosa mediante el Álgebra de Heyting:
  $$\Omega_3 := \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
  Donde el Supremo algebraico (join, $\sqcup$) consolida síncronamente el estado de lazo cerrado:
  $$\nu_{\mathrm{final}} = \nu_{\mathrm{cohomology}} \sqcup \nu_{\mathrm{remanente}} \sqcup \nu_{\mathrm{fock}}$$

Definición 2 (La Cirugía Anisotrópica y No-Señalización de de Rham):
  El acoplamiento mutuo entre el foso de transductores locales y la matriz de conductancias
  globales se rige por la remoción quirúrgica de la carta Čech ruidosa. Si el mismatch
  de primer orden $\check{H}^1$ supera la cota de Lipschitz del triple espectral de Connes,
  el agente ordena la deformación anisotrópica métrica:
  $$\mathbf{G}_{\mathrm{surgical}} = \mathbf{G} \odot (\mathbf{I} - \mathbf{P}_{\mathrm{noisy}})$$
  Esto aísla de forma exacta el canal ruidoso en Fock, garantizando la no-señalización bipartita
  y la estabilidad de Lyapunov ($\dot{\mathcal{H}} \le 0$) en lazo cerrado.

Definición 3 (La Rampa de Confianza de de Rham y override de Gracia):
  Para eludir la parálisis destructiva de la obra civil ante interferencias transitorias
  (ruido analógico por soldaduras o EMF en fango), el agente implementa la rampa de confianza:
  - Veto Suave (Luz Ámbar): Se activa si $0.3 \cdot \tau_{\mathrm{margin}} < \check{H}^1 \le 0.5 \cdot \tau_{\mathrm{margin}}$.
    Concede una ventana de gracia de 1 hora para inyectar en Fock un Positrón de Autorización
    Humana $e^+$, logrando la aniquilación cuántica mutua de la anomalía semántica $e^-$:
    $$e^- + e^+ \longrightarrow 2\gamma \quad \implies \quad \mathtt{heyting\_verdict} = \mathtt{DEGRADED}$$
    Evitando falsas paradas y permitiendo que la obra continúe su vaciado de concreto en fango.
  - Veto Duro (Frenado en Silicio): Se activa si $\check{H}^1 > 0.5 \cdot \tau_{\mathrm{margin}}$ o si expira el
    período de gracia sin autorización, colapsando Heyting al Supremo terminal VETOED ($\top$).

================================════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE CONTROL COVARIANTE (Leyes de Consistencia)
================================════════════════════════════════════════════════

Axioma I (Principio de Conservación de Traza de von Neumann):
  La cirugía de traceout cuántico del modo ruidoso aislado en el espacio de Fock debe conservar
  síncronamente la traza de von Neumann del operador de densidad mixto en la FPU:
  $$\operatorname{Tr}(\rho_{\mathrm{surgery}}) \equiv 1.0 \quad \implies \quad \|\operatorname{Tr}(\rho_{\mathrm{surgery}}) - 1.0\| \le \varepsilon_{\mathrm{Wilkinson}}$$
  Cualquier pérdida de traza parcial o surgimiento de autovalores negativos delata deriva de Wilkinson.

Axioma II (Axioma de de Rham-Fiedler de Conexidad Remanente):
  Para asegurar la sismorresistencia global del presupuesto remanente tras la amputación
  de la carta Čech ruidosa, la conectividad algebraica del subcomplejo remanente debe ser no nula:
  $$\lambda_2(\mathbf{L}_{\mathrm{remSub}}) \ge \tau_{\mathrm{Fiedler}} \quad \implies \quad \beta_0 \equiv \dim H^0(K_{\mathrm{rem}}; \, \mathbb{Z}) = 1$$
  Si la cirugía destruye la conexidad global, el sistema aborta de inmediato por riesgo de quiebra.

Axioma III (Teorema de Actuación Ciber-Física Crowbar de la Sonda Čech):
  Ante el colapso de Heyting al Supremo de veto ($\top$), la subrutina local isVerdictCoherent()
  del microcontrolador ESP32 despacha síncronamente la ISR en IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor BT151 (Crowbar) para paralizar mecánicamente la obra en el milisegundo cero.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Estabilidad de de Rham-Lyapunov del Lazo Cerrado):
  La evolución de la trayectoria de control conjunta $\mathbf{\Psi}(t) = (\mathbf{p}, \, \rho)^\top$ satisface la
  desigualdad de Clausius-Duhem y la contracción de Lyapunov en la FPU:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) = \nabla \mathcal{H}(\mathbf{\Psi})^\top \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}}$$
  Donde $\tau_{\mathrm{Lyapunov}} = 10^{-12}$ es la cota elástica de deriva en punto flotante de 64 bits.

Invariante II (Confinamiento de la Métrica de-confinada de Čech):
  La obstrucción local se encuentra confinada por la cota de Lipschitz del operador de Dirac de Connes:
  $$\check{H}^1(\mathcal{U}; \, \mathcal{F}) \le L_{\max} \cdot \tau_{\mathrm{margin}} \quad \implies \quad \mathtt{heyting\_verdict} = \mathtt{COHERENT}$$

Invariante III (Sello de Sesión Criptográfico e Inmutabilidad de RAM):
  Para prevenir inyecciones de estado o ataques de-normalización intermedia, el soberano genera
  un sello inmutable unívoco para congelar la sesión en RAM en cada ciclo OODA:
  $$\mathtt{cryptographic\_seal} := \operatorname{SHA-256}\left(\delta_{\mathrm{\check{C}ech}} \oplus \mathbf{G}_{\mathrm{surgical}} \oplus \lambda_2 \oplus H_{\mathrm{ext}}\right)$$
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np


logger = logging.getLogger("APU.Agents.Wisdom.TopologicalSurgeryCechAgent")


try:
    from app.core.topological_surgery_cech import TopologicalSurgeryCech
except ImportError:  # pragma: no cover - resolución de calibre alternativa
    try:
        from ...core.topological_surgery_cech import TopologicalSurgeryCech
    except ImportError:  # pragma: no cover
        from topological_surgery_cech import TopologicalSurgeryCech


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, CRIPTOGRÁFICAS Y DE SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_STD_NS: Final[float] = 3.2
_CROWBAR_LATENCY_MIN_NS: Final[float] = 380.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 420.0
_GPIO_CROWBAR_PIN: Final[int] = 14

_SCHEMA_VERSION: Final[str] = (
    "2.1.0-Doctoral-Cech-Agent-Heyting-HMAC-Hodge-FailClosed"
)

_DOMAIN_SESSION: Final[bytes] = b"APU/CECH-SURGERY-AGENT/SESSION/v2.1"
_DOMAIN_TELEMETRY: Final[bytes] = b"APU/CECH-SURGERY-AGENT/TELEMETRY/v2.1"
_DOMAIN_OVERRIDE: Final[bytes] = b"APU/CECH-SURGERY-AGENT/OVERRIDE/v2.1"
_DOMAIN_AGENT: Final[bytes] = b"APU/CECH-SURGERY-AGENT/IDENTITY/v2.1"
_DOMAIN_PAYLOAD: Final[bytes] = b"APU/CECH-SURGERY-AGENT/PAYLOAD/v2.1"

_SHA256_HEX_LEN: Final[int] = 64
_OVERRIDE_TOKEN_MAX_BYTES: Final[int] = 4096
_IMAG_TOL: Final[float] = 1e-12
_TRACE_HARD_FLOOR: Final[float] = 1e-6
_TRACE_HARD_FACTOR: Final[float] = 10.0
_FIEDLER_SOFT_FACTOR: Final[float] = 10.0
_COND_DEGRADED: Final[float] = 1e9


class HeytingVerdict(str, Enum):
    r"""
    Retículo de decisión de Heyting de tres niveles

        Ω₃ = { VETOED ≼ DEGRADED ≼ COHERENT }

    con operaciones intuicionistas:
        meet  ∧  = ínfimo (más restrictivo),
        join  ∨  = supremo,
        a → b    = 1 si a ≼ b, else b.

    El VETO duro del motor es un elemento absorbente para el override:
    no existe sección de autorización sobre el cerrado {VETOED}.
    La cirugía activa es ámbar: el override no eleva a COHERENT.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"


_HEYTING_ORDER: Final[dict[str, int]] = {
    HeytingVerdict.VETOED.value: 0,
    HeytingVerdict.DEGRADED.value: 1,
    HeytingVerdict.COHERENT.value: 2,
}


class SurgeryAgentError(ValueError):
    """Error de canonización, política o invariante de gobernanza del soberano Čech."""


@runtime_checkable
class CechSurgeryEngineProtocol(Protocol):
    """Contrato mínimo del motor de cirugía de Čech inyectable."""

    def execute_topological_surgery_cycle(
        self,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
    ) -> Any:
        ...


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICADO PÚBLICO
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SurgeryAgentCertificate:
    r"""Certificado formal de calibración emitido por el Soberano de Cirugía de Čech."""

    phase: str
    heyting_verdict: str
    cohomological_mismatch: float
    isolated_fock_trace: float
    fiedler_residual: float
    is_surgery_active: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    schema_version: str = _SCHEMA_VERSION
    session_digest: str = ""
    engine_verdict: str = ""
    engine_seal: str = ""
    noisy_cover_id: int = -1
    soft_veto_lower_threshold: float = math.nan
    soft_veto_upper_threshold: float = math.nan
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)

    # Extensiones de calibre 2.1 (valores por defecto: compatibles).
    agent_digest: str = ""
    payload_digest: str = ""
    engine_digest: str = ""
    engine_interlock_fired: bool = False
    override_present: bool = False
    override_hmac_bound: bool = False
    confidence_ramp: float = 0.0
    discarded_fock_mass: float = 0.0
    surgical_metric_cond: float = math.nan
    metric_spectral_gap: float = math.nan
    nerve_betti_0: int = 0
    active_node_count: int = 0
    is_globally_coherent: bool = False
    lipschitz_bound: float = math.nan


# ═══════════════════════════════════════════════════════════════════════════
# DOSSIERS INTERNOS DE FASE (objetos del morfismo anidado)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Phase1Dossier:
    """
    Expediente canonizado y congelado en la FASE 1 · OBSERVE.

    El token de override NUNCA viaja en claro: sólo su hash de dominio,
    su presencia y su validez criptográfica/morfológica.
    """

    boundary_matrix: np.ndarray
    global_metric_G: np.ndarray
    global_density_rho: np.ndarray
    local_signals: Dict[int, np.ndarray]
    lipschitz_bound_Lmax: float
    override_token_hash: Optional[str]
    override_present: bool
    override_is_valid: bool
    override_hmac_bound: bool
    session_digest: str
    payload_digest: str
    agent_digest: str
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase2Dossier:
    """Expediente auditado en la FASE 2 · ORIENT/DECIDE."""

    phase1: Phase1Dossier
    cohomological_mismatch: float
    isolated_fock_trace: float
    fiedler_residual: float
    is_surgery_active: bool
    is_globally_coherent: bool
    engine_verdict: str
    engine_seal: str
    engine_digest: str
    engine_interlock_fired: bool
    noisy_cover_id: int
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    heyting_verdict: str
    confidence_ramp: float
    discarded_fock_mass: float
    surgical_metric_cond: float
    metric_spectral_gap: float
    nerve_betti_0: int
    active_node_count: int
    diagnostics: Tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════
# SOBERANO DE CIRUGÍA DE ČECH
# ═══════════════════════════════════════════════════════════════════════════


class TopologicalSurgeryCechAgent:
    r"""
    Soberano de la Sonda de Cirugía Topológica de Čech.

    Orquesta el ciclo covariante OODA de la cirugía, administra la rampa de
    confianza graduada y decide en el retículo de Heyting Ω₃.

    Ciclo público, fases anidadas:

        execute_surgery_control_cycle()
          └─ _phase1_observe_and_freeze()          # cierra FASE 1
               └─ _phase2_open_from_phase1()       # abre  FASE 2
                    └─ _phase2_orient_and_decide()
                         └─ _phase3_open_from_phase2()  # abre FASE 3
                              └─ _phase3_certify()

    Ante cualquier excepción no recuperable emite certificado fail-closed
    con veredicto VETOED e interlock activado.
    """

    def __init__(
        self,
        dimension_n: int,
        covering: Dict[int, list],
        nominal_impedance: float = 50.0,
        safety_margin: float = 1.0,
        unitarity_tolerance: float = 1e-3,
        fiedler_threshold: float = 1e-4,
        *,
        rng_seed: Optional[int] = None,
        engine: Optional[Any] = None,
        soft_veto_lower: float = 0.3,
        soft_veto_upper: float = 0.5,
        override_authority: Optional[Any] = None,
    ) -> None:
        """
        Inicializa el soberano de cirugía Čech.

        Args:
            dimension_n: Dimensión fundamental del complejo simplicial.
            covering: Cubrimiento de Čech: cover_id → lista de nodos.
            nominal_impedance: Impedancia de referencia Z₀ (ohmios).
            safety_margin: Factor elástico de las cotas de mismatch.
            unitarity_tolerance: Tolerancia base de conservación de traza (pre-margen).
            fiedler_threshold: Umbral base de conectividad algebraica (pre-margen).
            rng_seed: Semilla opcional para reproducibilidad del jitter.
            engine: Motor Čech inyectable. Si es None se instancia uno.
            soft_veto_lower: Cota inferior base del veto suave (fracción de L_max).
            soft_veto_upper: Cota superior base del veto suave (fracción de L_max).
            override_authority: Secreto HMAC opcional que liga el token de
                override a la carga útil de sesión. Si es None, la validez
                se reduce a morfología SHA-256 hexadecimal (64 nibbles).
        """
        if dimension_n <= 0:
            raise SurgeryAgentError(
                "La dimensión fundamental debe ser estrictamente positiva."
            )

        if not math.isfinite(nominal_impedance) or nominal_impedance <= 0.0:
            raise SurgeryAgentError(
                "nominal_impedance debe ser finita y estrictamente positiva."
            )

        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise SurgeryAgentError(
                "safety_margin debe ser finito y estrictamente positivo."
            )

        if not math.isfinite(unitarity_tolerance) or unitarity_tolerance < 0.0:
            raise SurgeryAgentError(
                "unitarity_tolerance debe ser finita y no negativa."
            )

        if not math.isfinite(fiedler_threshold) or fiedler_threshold < 0.0:
            raise SurgeryAgentError(
                "fiedler_threshold debe ser finita y no negativa."
            )

        if not math.isfinite(soft_veto_lower) or soft_veto_lower < 0.0:
            raise SurgeryAgentError("soft_veto_lower debe ser finita y no negativa.")

        if not math.isfinite(soft_veto_upper) or soft_veto_upper <= 0.0:
            raise SurgeryAgentError(
                "soft_veto_upper debe ser finita y estrictamente positiva."
            )

        if soft_veto_lower >= soft_veto_upper:
            raise SurgeryAgentError(
                "soft_veto_lower debe ser menor que soft_veto_upper."
            )

        validated_covering = self._canonicalize_covering(dimension_n, covering)

        self._n: Final[int] = int(dimension_n)
        self._covering: Final[Dict[int, Tuple[int, ...]]] = validated_covering
        self._cover_ids: Final[Tuple[int, ...]] = tuple(sorted(validated_covering.keys()))
        self._z0: Final[float] = float(nominal_impedance)
        self._safety_margin: Final[float] = float(safety_margin)

        # Semántica original, ahora explícita:
        #   τ_T  = unitarity_tolerance · safety_margin     (cota blanda de |tr−1|)
        #   τ_T★ = max(10⁻⁶, τ_T · 10)                     (cota dura de |tr−1|)
        #   τ_F  = fiedler_threshold / safety_margin       (cota dura, λ₂)
        #   τ_F° = τ_F · 10                                (cota blanda, λ₂)
        self._trace_tol: Final[float] = float(unitarity_tolerance) * self._safety_margin
        self._trace_hard: Final[float] = max(
            _TRACE_HARD_FLOOR,
            self._trace_tol * _TRACE_HARD_FACTOR,
        )
        self._fiedler_hard: Final[float] = float(fiedler_threshold) / self._safety_margin
        self._fiedler_soft: Final[float] = self._fiedler_hard * _FIEDLER_SOFT_FACTOR
        self._fiedler_thresh: Final[float] = self._fiedler_hard  # alias histórico

        self._soft_lower_base: Final[float] = float(soft_veto_lower)
        self._soft_upper_base: Final[float] = float(soft_veto_upper)

        self._reg: Final[float] = max(1e-15, _MACHINE_EPS)
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)
        self._override_authority: Final[Optional[bytes]] = (
            self._coerce_authority_secret(override_authority)
        )
        self._agent_digest: Final[str] = self._identity_digest()

        if engine is None:
            engine = self._phase0_instantiate_engine(rng_seed=rng_seed)

        if not hasattr(engine, "execute_topological_surgery_cycle"):
            raise TypeError(
                "El engine inyectado no implementa execute_topological_surgery_cycle."
            )
        if not callable(getattr(engine, "execute_topological_surgery_cycle")):
            raise TypeError("execute_topological_surgery_cycle debe ser invocable.")

        self._engine: Final[Any] = engine

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES CANÓNICAS (hash, congelación, retículo, cubrimiento)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _canonicalize_covering(
        dimension_n: int,
        covering: Any,
    ) -> Dict[int, Tuple[int, ...]]:
        """Canoniza el cubrimiento de Čech: claves enteras, nodos únicos en rango."""
        if not isinstance(covering, dict) or len(covering) == 0:
            raise SurgeryAgentError(
                "covering debe ser un diccionario no vacío de cartas de Čech."
            )

        validated: Dict[int, Tuple[int, ...]] = {}
        for key, nodes in covering.items():
            try:
                cover_id = int(key)
            except (TypeError, ValueError) as exc:
                raise SurgeryAgentError(
                    "Toda clave del cubrimiento de Čech debe ser entera."
                ) from exc

            if nodes is None:
                raise SurgeryAgentError(
                    f"La carta de Čech {cover_id} no puede ser None."
                )

            canonical: List[int] = []
            seen: set[int] = set()
            for node in nodes:
                try:
                    node_id = int(node)
                except (TypeError, ValueError) as exc:
                    raise SurgeryAgentError(
                        f"Nodo inválido en la carta de Čech {cover_id}."
                    ) from exc

                if node_id < 0 or node_id >= dimension_n:
                    raise SurgeryAgentError(
                        f"Nodo {node_id} fuera del complejo simplicial de "
                        f"dimensión {dimension_n}."
                    )
                if node_id in seen:
                    continue
                seen.add(node_id)
                canonical.append(node_id)

            validated[cover_id] = tuple(canonical)

        return validated

    @staticmethod
    def _coerce_authority_secret(value: Any) -> Optional[bytes]:
        """Normaliza el secreto HMAC; vacío ≡ ausente."""
        if value is None:
            return None
        if isinstance(value, bytes):
            return value if value else None
        if isinstance(value, bytearray):
            raw = bytes(value)
            return raw if raw else None
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            return encoded if encoded else None
        raise SurgeryAgentError("override_authority debe ser bytes, str o None.")

    @staticmethod
    def _hash_update_domain(sha: Any, domain: bytes) -> None:
        """Actualiza SHA-256 con separación de dominio length-prefixed."""
        sha.update(len(domain).to_bytes(8, "little", signed=False))
        sha.update(domain)

    @staticmethod
    def _hash_update_bytes(sha: Any, data: bytes) -> None:
        """Actualiza SHA-256 con bloque length-prefixed."""
        sha.update(len(data).to_bytes(8, "little", signed=False))
        sha.update(data)

    @classmethod
    def _hash_update_array(cls, sha: Any, name: str, array: np.ndarray) -> None:
        """Actualiza SHA-256 con arreglo numpy C-contiguo y dtype explícito."""
        cls._hash_update_bytes(sha, name.encode("utf-8"))

        arr = np.ascontiguousarray(array)

        sha.update(len(arr.shape).to_bytes(4, "little", signed=False))
        for dim in arr.shape:
            sha.update(int(dim).to_bytes(8, "little", signed=False))

        cls._hash_update_bytes(sha, str(arr.dtype).encode("utf-8"))
        sha.update(arr.tobytes())

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any) -> np.ndarray:
        """Copia defensiva C-contigua e inmutable (write-flag desactivado)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @classmethod
    def _freeze_signal_map(
        cls,
        signals: Mapping[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Congela cada señal local como vector float64 write-protected."""
        frozen: Dict[int, np.ndarray] = {}
        for cover_id, signal in signals.items():
            frozen[int(cover_id)] = cls._freeze_array(
                np.asarray(signal).reshape(-1),
                np.float64,
            )
        return frozen

    @staticmethod
    def _heyting_meet(verdicts: Sequence[str]) -> str:
        r"""Ínfimo en Ω₃: el veredicto más restrictivo (VETOED ≼ ... ≼ COHERENT)."""
        if not verdicts:
            return HeytingVerdict.COHERENT.value
        rank = min(_HEYTING_ORDER.get(str(v), 0) for v in verdicts)
        inverse = {
            0: HeytingVerdict.VETOED.value,
            1: HeytingVerdict.DEGRADED.value,
            2: HeytingVerdict.COHERENT.value,
        }
        return inverse[rank]

    def _identity_digest(self) -> str:
        """Huella de la identidad metrológica del soberano (no de la sesión)."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_AGENT)
        hmac_flag = b"hmac:1" if self._override_authority is not None else b"hmac:0"
        payload = (
            f"n={self._n}|z0={self._z0:.17e}|sm={self._safety_margin:.17e}|"
            f"tt={self._trace_tol:.17e}|th={self._trace_hard:.17e}|"
            f"fh={self._fiedler_hard:.17e}|fs={self._fiedler_soft:.17e}|"
            f"sl={self._soft_lower_base:.17e}|su={self._soft_upper_base:.17e}|"
            f"covers={len(self._cover_ids)}|schema={_SCHEMA_VERSION}"
        )
        self._hash_update_bytes(sha, payload.encode("utf-8"))
        self._hash_update_bytes(sha, hmac_flag)
        for cover_id in self._cover_ids:
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            nodes = np.asarray(self._covering[cover_id], dtype=np.int64)
            self._hash_update_array(sha, f"cover_{cover_id}", nodes)
        return sha.hexdigest()

    def _phase0_instantiate_engine(self, *, rng_seed: Optional[int]) -> Any:
        """
        Fabrica un motor Čech alineando umbrales de Fiedler con la rampa
        del agente. Retrocompatible con constructores antiguos (TypeError).
        """
        covering_for_engine: Dict[int, List[int]] = {
            cover_id: list(nodes) for cover_id, nodes in self._covering.items()
        }
        attempts: Tuple[dict[str, Any], ...] = (
            {
                "dimension_n": self._n,
                "covering": covering_for_engine,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
                "rng_seed": rng_seed,
                "fiedler_veto_threshold": self._fiedler_hard,
                "fiedler_degraded_threshold": self._fiedler_soft,
            },
            {
                "dimension_n": self._n,
                "covering": covering_for_engine,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
                "rng_seed": rng_seed,
                "fiedler_veto_threshold": self._fiedler_hard,
            },
            {
                "dimension_n": self._n,
                "covering": covering_for_engine,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
                "rng_seed": rng_seed,
            },
            {
                "dimension_n": self._n,
                "covering": covering_for_engine,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
            },
        )
        last_error: Optional[BaseException] = None
        for kwargs in attempts:
            try:
                return TopologicalSurgeryCech(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        raise SurgeryAgentError(
            f"No fue posible instanciar TopologicalSurgeryCech: {last_error}"
        )

    def _soft_band(self, lipschitz_bound: float) -> Tuple[float, float]:
        """Intervalo (ℓ, u] del veto suave: fracciones de L_max · safety_margin."""
        scale = float(lipschitz_bound) * self._safety_margin
        return (
            self._soft_lower_base * scale,
            self._soft_upper_base * scale,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TOKEN DE OVERRIDE
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_override_token(override_token: Any) -> Optional[str]:
        """Normaliza el token a str sin revelar contenido; extraño ≡ ausencia."""
        if override_token is None:
            return None

        if isinstance(override_token, bytes):
            if len(override_token) > _OVERRIDE_TOKEN_MAX_BYTES:
                return None
            try:
                token = override_token.decode("utf-8")
            except UnicodeDecodeError:
                return None
        elif isinstance(override_token, str):
            if len(override_token.encode("utf-8", errors="ignore")) > (
                _OVERRIDE_TOKEN_MAX_BYTES
            ):
                return None
            token = override_token
        else:
            return None

        token = token.strip()
        return token or None

    @staticmethod
    def _token_is_sha256_hex(token: str) -> bool:
        """Morfología de digest SHA-256: exactamente 32 bytes en hex."""
        if len(token) != _SHA256_HEX_LEN:
            return False
        try:
            raw = bytes.fromhex(token)
        except ValueError:
            return False
        return len(raw) == 32

    def _hash_override_token(self, override_token: Optional[str]) -> Optional[str]:
        """Hashea el token de override con separación de dominio."""
        token = self._normalize_override_token(override_token)
        if token is None:
            return None

        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_OVERRIDE)
        self._hash_update_bytes(sha, token.encode("utf-8"))
        return sha.hexdigest()

    def _is_valid_override_token(self, override_token: Optional[str]) -> bool:
        """
        API de compatibilidad: validez morfológica SHA-256 hexadecimal.

        La ligadura HMAC (si hay autoridad) se resuelve en FASE 1 contra
        el digest de la carga útil; esta API no puede decidir HMAC sin payload.
        """
        token = self._normalize_override_token(override_token)
        if token is None:
            return False
        return self._token_is_sha256_hex(token)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Objeto inicial  : tensores crudos, χ-Lipschitz y token de override.
    # Objeto terminal : Phase1Dossier (inmutable, hasheado, sin secreto en claro).
    # Morfismo de cierre:
    #   _phase1_observe_and_freeze  →  _phase2_open_from_phase1
    #
    # Invariantes:
    #   (I1) B ∈ M_{M×n}(ℝ), finita, n columnas.
    #   (I2) G ∈ M_{M×M}(ℝ), finita (el motor proyecta a SPD).
    #   (I3) ρ ∈ M_{n×n}(ℂ), finita (el motor proyecta al simplejo).
    #   (I4) señales locales finitas, indexadas por cover_id entero.
    #   (I5) L_max > 0 finito.
    #   (I6) override ∈ {∅} ∪ Digest₆₄; nunca se serializa en claro.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase1_canonical_boundary_matrix(self, boundary_matrix: Any) -> np.ndarray:
        """Valida la matriz de incidencia real δ con n columnas (nodos)."""
        try:
            boundary = np.asarray(boundary_matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SurgeryAgentError(
                "boundary_matrix no puede convertirse a float64."
            ) from exc

        if boundary.ndim != 2:
            raise SurgeryAgentError("boundary_matrix debe ser bidimensional.")

        if boundary.shape[0] == 0:
            raise SurgeryAgentError("boundary_matrix no puede tener cero canales.")

        if boundary.shape[1] != self._n:
            raise SurgeryAgentError(
                f"boundary_matrix debe tener {self._n} columnas (nodos)."
            )

        if not np.all(np.isfinite(boundary)):
            raise SurgeryAgentError("boundary_matrix contiene valores no finitos.")

        return np.ascontiguousarray(boundary, dtype=np.float64)

    def _phase1_canonical_metric(
        self,
        global_metric_G: Any,
        expected_dim: int,
    ) -> np.ndarray:
        """Valida métrica real finita de dimensión esperada; simetriza para firma."""
        if np.iscomplexobj(global_metric_G):
            try:
                metric_complex = np.asarray(global_metric_G, dtype=np.complex128)
            except (TypeError, ValueError) as exc:
                raise SurgeryAgentError(
                    "global_metric_G no puede convertirse a complex128."
                ) from exc

            if not np.all(np.isfinite(metric_complex)):
                raise SurgeryAgentError(
                    "global_metric_G contiene valores no finitos."
                )

            imag_norm = (
                float(np.max(np.abs(metric_complex.imag)))
                if metric_complex.size
                else 0.0
            )
            if imag_norm > _IMAG_TOL:
                raise SurgeryAgentError(
                    "global_metric_G debe ser real dentro de tolerancia numérica."
                )
            metric = np.ascontiguousarray(metric_complex.real, dtype=np.float64)
        else:
            try:
                metric = np.asarray(global_metric_G, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SurgeryAgentError(
                    "global_metric_G no puede convertirse a float64."
                ) from exc

        if metric.ndim != 2 or metric.shape != (expected_dim, expected_dim):
            raise SurgeryAgentError(
                f"global_metric_G debe tener forma ({expected_dim},{expected_dim})."
            )

        if not np.all(np.isfinite(metric)):
            raise SurgeryAgentError("global_metric_G contiene valores no finitos.")

        symmetric = 0.5 * (metric + metric.T)
        return np.ascontiguousarray(symmetric, dtype=np.float64)

    def _phase1_canonical_density_matrix(self, global_density_rho: Any) -> np.ndarray:
        """Valida matriz de densidad compleja finita de dimensión n (sin proyectar)."""
        try:
            density = np.asarray(global_density_rho, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SurgeryAgentError(
                "global_density_rho no puede convertirse a complex128."
            ) from exc

        if density.ndim != 2 or density.shape != (self._n, self._n):
            raise SurgeryAgentError(
                f"global_density_rho debe tener forma ({self._n},{self._n})."
            )

        if not np.all(np.isfinite(density)):
            raise SurgeryAgentError(
                "global_density_rho contiene valores no finitos."
            )

        return np.ascontiguousarray(density, dtype=np.complex128)

    def _phase1_canonical_local_signals(
        self,
        local_signals: Any,
    ) -> Tuple[Dict[int, np.ndarray], Tuple[str, ...]]:
        """Canoniza señales locales de telemetría por carta de Čech."""
        diagnostics: List[str] = []

        if not isinstance(local_signals, dict):
            raise SurgeryAgentError(
                "local_signals debe ser un diccionario cover_id → señal."
            )

        out: Dict[int, np.ndarray] = {}
        for key, value in local_signals.items():
            try:
                cover_id = int(key)
            except (TypeError, ValueError) as exc:
                raise SurgeryAgentError(
                    "Toda clave de local_signals debe ser entera."
                ) from exc

            try:
                signal = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise SurgeryAgentError(
                    f"Señal local inválida en la carta Čech {cover_id}."
                ) from exc

            if signal.ndim == 0:
                signal = signal.reshape(1)
            else:
                signal = signal.reshape(-1)

            if not np.all(np.isfinite(signal)):
                raise SurgeryAgentError(
                    f"Señal local no finita en la carta Čech {cover_id}."
                )

            out[cover_id] = np.ascontiguousarray(signal, dtype=np.float64)

        extra = set(out.keys()) - set(self._cover_ids)
        if extra:
            diagnostics.append(
                f"Señales con cover_id ajenos al cubrimiento: {sorted(extra)}."
            )

        missing = [cover_id for cover_id in self._cover_ids if cover_id not in out]
        if missing:
            diagnostics.append(
                f"Cartas sin señal (el motor las interpretará vacías): {missing}."
            )

        return out, tuple(diagnostics)

    @staticmethod
    def _phase1_canonical_lipschitz_bound(lipschitz_bound_Lmax: Any) -> float:
        """Valida cota de Lipschitz como escalar finito estrictamente positivo."""
        try:
            value = float(lipschitz_bound_Lmax)
        except (TypeError, ValueError) as exc:
            raise SurgeryAgentError("lipschitz_bound_Lmax debe ser escalar.") from exc

        if not math.isfinite(value) or value <= 0.0:
            raise SurgeryAgentError(
                "lipschitz_bound_Lmax debe ser finito y estrictamente positivo."
            )

        return value

    def _phase1_generate_payload_digest(
        self,
        *,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
    ) -> str:
        """Digest de la carga útil física, independiente del override."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_PAYLOAD)
        self._hash_update_bytes(sha, self._agent_digest.encode("utf-8"))
        self._hash_update_array(sha, "boundary_matrix", boundary_matrix)
        self._hash_update_array(sha, "global_metric_G", global_metric_G)
        self._hash_update_array(sha, "global_density_rho", global_density_rho)

        for cover_id in self._cover_ids:
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            nodes = np.asarray(self._covering[cover_id], dtype=np.int64)
            self._hash_update_array(sha, f"cover_nodes_{cover_id}", nodes)

        for cover_id in sorted(local_signals.keys()):
            self._hash_update_bytes(sha, str(int(cover_id)).encode("utf-8"))
            self._hash_update_array(
                sha,
                f"local_signal_{cover_id}",
                local_signals[cover_id],
            )

        self._hash_update_bytes(
            sha,
            f"{float(lipschitz_bound_Lmax):.17e}".encode("utf-8"),
        )
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase1_verify_override_authority(
        self,
        token: Optional[str],
        payload_digest: str,
    ) -> Tuple[bool, bool, Tuple[str, ...]]:
        r"""
        Valida el override.

        • Sin autoridad HMAC: basta morfología de digest SHA-256 (64 hex).
        • Con autoridad: token ≡ HMAC-SHA256(k, OVERRIDE_DOMAIN ‖ payload)
          comparado en tiempo constante (hmac.compare_digest).

        Retorna (válido, hmac_ligado, diagnósticos).
        """
        diagnostics: List[str] = []

        if token is None:
            return False, False, tuple(diagnostics)

        morphological = self._token_is_sha256_hex(token)
        if self._override_authority is None:
            if morphological:
                diagnostics.append(
                    "Override presente; validación morfológica SHA-256 "
                    "(sin autoridad HMAC configurada)."
                )
                return True, False, tuple(diagnostics)
            diagnostics.append("Override presente pero morfológicamente inválido.")
            return False, False, tuple(diagnostics)

        try:
            payload_bytes = bytes.fromhex(payload_digest)
        except ValueError:
            diagnostics.append("payload_digest ilegible; override rechazado.")
            return False, True, tuple(diagnostics)

        expected = hmac.new(
            self._override_authority,
            _DOMAIN_OVERRIDE + payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        candidate = token.lower() if morphological else token
        bound = hmac.compare_digest(expected, candidate)
        if bound:
            diagnostics.append(
                "Override HMAC ligado a la carga útil; sección de autorización válida."
            )
            return True, True, tuple(diagnostics)

        diagnostics.append(
            "Override HMAC no liga la carga útil; autorización rechazada."
        )
        return False, True, tuple(diagnostics)

    def _phase1_generate_session_digest(
        self,
        *,
        payload_digest: str,
        override_token_hash: Optional[str],
        override_is_valid: bool,
        override_hmac_bound: bool,
    ) -> str:
        """Firma SHA-256 de sesión: payload ⊕ compromiso de override (nunca el token)."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._agent_digest.encode("utf-8"))
        self._hash_update_bytes(sha, payload_digest.encode("utf-8"))

        if override_token_hash is None:
            self._hash_update_bytes(sha, b"override:absent")
        else:
            self._hash_update_bytes(sha, b"override:present")
            self._hash_update_bytes(sha, override_token_hash.encode("utf-8"))
            self._hash_update_bytes(
                sha,
                b"override:valid" if override_is_valid else b"override:invalid",
            )
            self._hash_update_bytes(
                sha,
                b"override:hmac" if override_hmac_bound else b"override:morph",
            )

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase1_observe_and_freeze(
        self,
        *,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token: Optional[str],
    ) -> SurgeryAgentCertificate:
        r"""
        FASE 1 · OBSERVE  (morfismo de cierre).

        Canoniza, regulariza y congela el expediente de frontera. El valor de
        retorno no es un terminal de Fase 1: es el compuesto

            Φ₁₂ ∘ Observe  :  datos crudos  →  SurgeryAgentCertificate

        realizado por la continuación formal `_phase2_open_from_phase1`,
        que es el primer morfismo de la FASE 2 · ORIENT/DECIDE.
        """
        diagnostics: List[str] = []

        boundary = self._phase1_canonical_boundary_matrix(boundary_matrix)
        metric = self._phase1_canonical_metric(
            global_metric_G,
            expected_dim=int(boundary.shape[0]),
        )
        density = self._phase1_canonical_density_matrix(global_density_rho)
        signals, signal_diag = self._phase1_canonical_local_signals(local_signals)
        diagnostics.extend(signal_diag)
        lipschitz = self._phase1_canonical_lipschitz_bound(lipschitz_bound_Lmax)

        payload_digest = self._phase1_generate_payload_digest(
            boundary_matrix=boundary,
            global_metric_G=metric,
            global_density_rho=density,
            local_signals=signals,
            lipschitz_bound_Lmax=lipschitz,
        )

        token = self._normalize_override_token(override_token)
        token_hash = self._hash_override_token(token)
        override_present = token is not None
        override_is_valid, override_hmac_bound, override_diag = (
            self._phase1_verify_override_authority(token, payload_digest)
        )
        diagnostics.extend(override_diag)

        # El token en claro muere aquí: no entra al dossier ni al motor.
        token = None
        override_token = None

        session_digest = self._phase1_generate_session_digest(
            payload_digest=payload_digest,
            override_token_hash=token_hash,
            override_is_valid=override_is_valid,
            override_hmac_bound=override_hmac_bound,
        )

        dossier = Phase1Dossier(
            boundary_matrix=self._freeze_array(boundary, np.float64),
            global_metric_G=self._freeze_array(metric, np.float64),
            global_density_rho=self._freeze_array(density, np.complex128),
            local_signals=self._freeze_signal_map(signals),
            lipschitz_bound_Lmax=lipschitz,
            override_token_hash=token_hash,
            override_present=override_present,
            override_is_valid=bool(override_is_valid),
            override_hmac_bound=bool(override_hmac_bound),
            session_digest=session_digest,
            payload_digest=payload_digest,
            agent_digest=self._agent_digest,
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 1 OBSERVE: expediente Čech capturado. digest=%s override=%s hmac=%s",
            session_digest[:16],
            "valid"
            if override_is_valid
            else ("present" if override_present else "absent"),
            override_hmac_bound,
        )

        # ── continuación anidada: el terminal de FASE 1 es el inicial de FASE 2
        return self._phase2_open_from_phase1(dossier)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · ORIENT / DECIDE
    #
    # Objeto inicial  : Phase1Dossier  (entregado por _phase1_observe_and_freeze)
    # Objeto terminal : Phase2Dossier
    # Morfismo de apertura: _phase2_open_from_phase1
    # Morfismo de cierre  : _phase2_orient_and_decide → _phase3_open_from_phase2
    #
    # Invariantes:
    #   (J1) El VETO del motor es absorbente (no hay override sobre {VETOED}).
    #   (J2) Observables no finitos ⇒ invalid_observables ⇒ VETOED.
    #   (J3) Rampa μ(H¹) ∈ [0,1] relativa a (ℓ, u] = fracciones de L_max·sm.
    #   (J4) Cirugía activa ⇒ DEGRADED (cicatriz); override no la borra.
    #   (J5) Veto suave ⇔ μ ∈ (0, 1] bajo u; override válido ⇒ DEGRADED, si no VETO.
    #   (J6) veredicto = ⋀ átomos de Heyting.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase2_open_from_phase1(
        self,
        dossier: Phase1Dossier,
    ) -> SurgeryAgentCertificate:
        r"""
        Inicio formal de la FASE 2 · ORIENT/DECIDE.

        Continuación estricta del terminal de FASE 1: recibe el expediente
        congelado y abre la auditoría del motor, la rampa de confianza y
        el retículo de Heyting.
        """
        if not isinstance(dossier, Phase1Dossier):
            raise SurgeryAgentError(
                "FASE 2 exige un Phase1Dossier emitido por FASE 1."
            )
        return self._phase2_orient_and_decide(dossier)

    @staticmethod
    def _phase2_safe_float(value: Any) -> Tuple[float, bool]:
        """Convierte a float finito; retorna (valor, válido)."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0, False

        if not math.isfinite(number):
            return 0.0, False

        return number, True

    @staticmethod
    def _phase2_safe_int(value: Any, default: int = 0) -> int:
        """Convierte a entero; default si no es admisible."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _phase2_invoke_engine(self, dossier: Phase1Dossier) -> Any:
        """Invoca el motor Čech; el token en claro no se reinyecta."""
        try:
            certificate = self._engine.execute_topological_surgery_cycle(
                boundary_matrix=dossier.boundary_matrix,
                global_metric_G=dossier.global_metric_G,
                global_density_rho=dossier.global_density_rho,
                local_signals=dossier.local_signals,
                lipschitz_bound_Lmax=dossier.lipschitz_bound_Lmax,
                override_token=None,
            )
        except Exception as exc:
            raise SurgeryAgentError(
                f"Motor Čech falló de forma no recuperable: {exc}"
            ) from exc

        if certificate is None:
            raise SurgeryAgentError("Motor Čech devolvió None.")

        return certificate

    def _phase2_parse_engine_verdict(
        self,
        engine_certificate: Any,
    ) -> Tuple[str, str, bool, Tuple[str, ...]]:
        """Extrae veredicto, sello e interlock del certificado del motor."""
        diagnostics: List[str] = []

        raw = str(getattr(engine_certificate, "heyting_verdict", "")).strip().upper()
        try:
            verdict = HeytingVerdict(raw).value
        except ValueError:
            verdict = HeytingVerdict.VETOED.value
            diagnostics.append("Veredicto Čech desconocido; se asume VETOED.")

        seal = str(getattr(engine_certificate, "cryptographic_seal", "") or "")
        engine_interlock = bool(
            getattr(engine_certificate, "hardware_interlock_fired", False)
        )
        return verdict, seal, engine_interlock, tuple(diagnostics)

    def _phase2_collect_engine_diagnostics(
        self,
        engine_certificate: Any,
    ) -> Tuple[str, ...]:
        """Promueve diagnósticos del motor a la capa de sabiduría."""
        payload = getattr(engine_certificate, "diagnostics", ())
        if payload is None:
            return ()
        if isinstance(payload, str):
            return (payload,) if payload else ()
        if isinstance(payload, (list, tuple)):
            return tuple(str(item) for item in payload)
        return (str(payload),)

    def _phase2_extract_observables(
        self,
        engine_certificate: Any,
    ) -> Tuple[
        float,
        float,
        float,
        bool,
        bool,
        bool,
        int,
        float,
        float,
        float,
        int,
        int,
        Tuple[str, ...],
    ]:
        """
        Extrae observables con saneamiento fail-closed.

        Retorna
            mismatch, remaining_mass, λ₂, coherent, surgery, invalid,
            noisy_id, discarded, κ, gap, β₀, n_activos, diagnostics.
        """
        diagnostics: List[str] = []
        invalid = False

        state = getattr(engine_certificate, "state", None)
        if state is None:
            raise SurgeryAgentError("Certificado Čech sin estado acoplado.")

        mismatch, ok_mismatch = self._phase2_safe_float(
            getattr(state, "cohomological_mismatch", math.nan)
        )
        if not ok_mismatch:
            invalid = True
            mismatch = math.inf
            diagnostics.append(
                "cohomological_mismatch no finito; fail-closed (H¹=∞)."
            )

        remaining_mass, ok_trace = self._phase2_safe_float(
            getattr(state, "isolated_fock_trace", math.nan)
        )
        if not ok_trace:
            invalid = True
            remaining_mass = 0.0
            diagnostics.append(
                "isolated_fock_trace no finito; fail-closed (masa=0)."
            )

        fiedler, ok_fiedler = self._phase2_safe_float(
            getattr(state, "fiedler_residual", math.nan)
        )
        if not ok_fiedler:
            invalid = True
            fiedler = 0.0
            diagnostics.append("fiedler_residual no finito; fail-closed (λ₂=0).")

        is_globally_coherent = bool(getattr(state, "is_globally_coherent", False))
        is_surgery_active = bool(getattr(engine_certificate, "surgery_active", False))
        noisy_cover_id = self._phase2_safe_int(
            getattr(engine_certificate, "noisy_cover_id", -1),
            default=-1,
        )

        discarded, ok_disc = self._phase2_safe_float(
            getattr(state, "discarded_fock_mass", 0.0)
        )
        if not ok_disc:
            discarded = 0.0

        condition_number, ok_cond = self._phase2_safe_float(
            getattr(state, "surgical_metric_cond", math.nan)
        )
        if not ok_cond:
            condition_number = math.nan

        spectral_gap, ok_gap = self._phase2_safe_float(
            getattr(state, "metric_spectral_gap", math.nan)
        )
        if not ok_gap:
            spectral_gap = math.nan

        nerve_betti_0 = self._phase2_safe_int(
            getattr(
                state,
                "nerve_betti_0",
                getattr(engine_certificate, "nerve_betti_0", 0),
            ),
            default=0,
        )
        active_node_count = self._phase2_safe_int(
            getattr(state, "active_node_count", 0),
            default=0,
        )

        return (
            mismatch,
            remaining_mass,
            fiedler,
            is_globally_coherent,
            is_surgery_active,
            invalid,
            noisy_cover_id,
            discarded,
            condition_number,
            spectral_gap,
            nerve_betti_0,
            active_node_count,
            tuple(diagnostics),
        )

    def _phase2_confidence_ramp(
        self,
        mismatch: float,
        lipschitz_bound: float,
    ) -> float:
        r"""
        Función de pertenencia de obstrucción H¹

            μ = 0                         si m ≤ ℓ
            μ = (m − ℓ) / (u − ℓ)         si ℓ < m < u
            μ = 1                         si m ≥ u

        con ℓ = soft_lower · L_max · sm, u = soft_upper · L_max · sm.
        μ ∈ [0, 1]; no finito ⇒ 1.
        """
        magnitude, ok = self._phase2_safe_float(mismatch)
        if not ok:
            return 1.0

        lower, upper = self._soft_band(lipschitz_bound)
        span = upper - lower
        if span <= self._reg:
            return 1.0 if magnitude > upper else 0.0

        ramp = (magnitude - lower) / span
        return float(min(1.0, max(0.0, ramp)))

    def verify_soft_veto_override(
        self,
        mismatch: float,
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        r"""
        Evalúa el estado del Veto Suave (Luz Ámbar) y el override humano.

        Regla (intervalo semiabierto a la derecha):
          - Veto suave si
                ℓ < mismatch ≤ u
            con ℓ = soft_lower · L_max · sm, u = soft_upper · L_max · sm.
          - Si hay override morfológicamente válido, la gracia no expira.
          - Sin override válido, la gracia expira y el veto suave colapsa
            a VETOED en la fase de decisión.

        Nota: la ligadura HMAC (si hay autoridad) se resuelve en FASE 1;
        esta API pública conserva la semántica original de formato.
        """
        magnitude, ok_m = self._phase2_safe_float(mismatch)
        if not ok_m:
            logger.warning("verify_soft_veto_override recibió mismatch no finito.")
            return False, True

        lipschitz, ok_l = self._phase2_safe_float(lipschitz_bound_Lmax)
        if not ok_l or lipschitz <= 0.0:
            logger.warning(
                "verify_soft_veto_override recibió lipschitz_bound_Lmax inválida."
            )
            return False, True

        lower, upper = self._soft_band(lipschitz)
        is_soft_veto = (magnitude > lower) and (magnitude <= upper)
        if not is_soft_veto:
            return False, True

        if self._is_valid_override_token(override_token):
            token_hash = self._hash_override_token(override_token)
            logger.info(
                "¡POSITRÓN DE AUTORIZACIÓN HUMANA [e+] VÁLIDO EN AGENTE ČECH! "
                "Aniquilando electrón de anomalía analógica local. Hash: %s",
                token_hash[:16] if token_hash else "n/d",
            )
            return True, False

        logger.warning(
            "¡VETO SUAVE DETECTADO EN CIRUGÍA ČECH (LUZ ÁMBAR)! "
            "Mismatch transitorio de Čech. Override ausente o inválido; "
            "período de gracia expirado para esta época de control."
        )
        return True, True

    def _phase2_soft_veto_state(
        self,
        *,
        mismatch: float,
        lipschitz_bound: float,
        override_is_valid: bool,
    ) -> Tuple[bool, bool, Tuple[str, ...]]:
        """Estado interno de veto suave usando la validez ya congelada en FASE 1."""
        diagnostics: List[str] = []
        magnitude, ok_m = self._phase2_safe_float(mismatch)
        lipschitz, ok_l = self._phase2_safe_float(lipschitz_bound)

        if not ok_m:
            diagnostics.append("H¹ no finito al evaluar veto suave.")
            return False, True, tuple(diagnostics)
        if not ok_l or lipschitz <= 0.0:
            diagnostics.append("L_max inválida al evaluar veto suave.")
            return False, True, tuple(diagnostics)

        lower, upper = self._soft_band(lipschitz)
        is_soft = (magnitude > lower) and (magnitude <= upper)
        if not is_soft:
            return False, True, tuple(diagnostics)

        if override_is_valid:
            diagnostics.append(
                "Veto suave con override válido; período de gracia vigente."
            )
            return True, False, tuple(diagnostics)

        diagnostics.append(
            "Veto suave sin override válido; período de gracia expirado."
        )
        return True, True, tuple(diagnostics)

    def _phase2_evaluate_heyting(
        self,
        *,
        invalid_observables: bool,
        engine_verdict: str,
        is_globally_coherent: bool,
        is_surgery_active: bool,
        cohomological_mismatch: float,
        isolated_fock_trace: float,
        discarded_fock_mass: float,
        fiedler_residual: float,
        surgical_metric_cond: float,
        is_soft_veto_active: bool,
        override_grace_period_expired: bool,
        override_present: bool,
        override_valid: bool,
        lipschitz_bound_Lmax: float,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificación en el retículo de Heyting Ω₃ por meet de átomos.

        VETO duro (absorbente, no overridable):
          • observables inválidos;
          • veto duro del motor Čech;
          • pérdida de conexidad global remanente;
          • Fiedler bajo cota dura;
          • masa de Fock no positiva;
          • |tr ρ − 1| duro cuando NO hay cirugía (motor legado, traza renormalizada);
          • colapso total de Fock bajo cirugía (masa remanente ≤ 0);
          • veto suave sin override válido;
          • mismatch sobre u sin cirugía que lo justifique.

        DEGRADED:
          • motor Čech degradado o cirugía activa (cicatriz);
          • veto suave con override válido;
          • deriva moderada de traza / masa descartada;
          • Fiedler bajo cota blanda;
          • κ(G_surgical) elevado.
        """
        diagnostics: List[str] = []
        atoms: List[str] = []

        def veto(message: str) -> None:
            atoms.append(HeytingVerdict.VETOED.value)
            diagnostics.append(message)

        def degrade(message: str) -> None:
            atoms.append(HeytingVerdict.DEGRADED.value)
            diagnostics.append(message)

        if invalid_observables:
            veto("Observables del motor Čech no finitos; fail-closed.")

        if engine_verdict == HeytingVerdict.VETOED.value:
            veto("Motor Čech emitió VETO duro (no overridable).")

        if not is_globally_coherent:
            veto("La malla remanente perdió conexidad global.")

        if math.isfinite(fiedler_residual) and fiedler_residual < self._fiedler_hard:
            veto(
                f"Fiedler remanente {fiedler_residual:.6e} bajo cota dura "
                f"{self._fiedler_hard:.6e}."
            )

        if math.isfinite(isolated_fock_trace) and isolated_fock_trace <= 0.0:
            veto("Traza / masa de Fock no positiva.")

        # Distinción de calibre:
        #   motor 2.1  → isolated_fock_trace = masa remanente ANTES de renormalizar
        #                (descartada reportada aparte);
        #   motor 2.0  → isolated_fock_trace ≈ 1 tras renormalizar (sin discarded).
        discarded_ok, has_discarded = self._phase2_safe_float(discarded_fock_mass)
        engine_reports_luders_mass = bool(
            is_surgery_active and has_discarded and discarded_ok > 0.0
        )

        if math.isfinite(isolated_fock_trace):
            if engine_reports_luders_mass or is_surgery_active:
                if isolated_fock_trace <= self._reg:
                    veto("Colapso total del canal de Fock tras Lüders.")
                elif isolated_fock_trace < (1.0 - self._trace_tol):
                    degrade(
                        f"Masa de Fock remanente {isolated_fock_trace:.6e} "
                        "tras aislamiento quirúrgico (cicatriz esperada)."
                    )
            else:
                trace_deviation = abs(isolated_fock_trace - 1.0)
                if trace_deviation > self._trace_hard:
                    veto(
                        f"Traza de Fock violada: |Tr(ρ)-1|={trace_deviation:.6e} "
                        f"> {self._trace_hard:.6e}."
                    )
                elif trace_deviation > self._trace_tol:
                    degrade(
                        f"Deriva moderada de traza: |Tr(ρ)-1|={trace_deviation:.6e}."
                    )

        if engine_verdict == HeytingVerdict.DEGRADED.value or is_surgery_active:
            if override_present and override_valid:
                degrade(
                    "Cirugía Čech activa con positrón de autorización válido; "
                    "cicatriz topológica conservada (DEGRADED)."
                )
            elif override_present:
                degrade(
                    "Cirugía Čech activa con override inválido; "
                    "cuarentena topológica degradada."
                )
            else:
                degrade(
                    "Cirugía Čech activa; cuarentena topológica degradada sin override."
                )

        if is_soft_veto_active:
            if override_grace_period_expired:
                veto("Veto suave sin override válido; gracia expirada.")
            else:
                degrade("Veto suave con override válido; sesión degradada.")

        lower, upper = self._soft_band(lipschitz_bound_Lmax)
        if (
            math.isfinite(cohomological_mismatch)
            and (not is_surgery_active)
            and cohomological_mismatch > upper
        ):
            veto(
                f"Mismatch H¹={cohomological_mismatch:.6f} > cota superior "
                f"{upper:.6f} sin cirugía que lo justifique."
            )

        if math.isfinite(fiedler_residual) and (
            fiedler_residual < self._fiedler_soft
        ):
            degrade(
                f"Fiedler remanente {fiedler_residual:.6e} bajo cota blanda "
                f"{self._fiedler_soft:.6e}."
            )

        if math.isfinite(surgical_metric_cond) and (
            surgical_metric_cond > _COND_DEGRADED
        ):
            degrade("Número de condición de G_surgical elevado.")

        if not atoms:
            atoms.append(HeytingVerdict.COHERENT.value)

        # `lower` se retiene para trazabilidad de la banda en diagnósticos futuros.
        _ = lower
        return self._heyting_meet(atoms), tuple(diagnostics)

    def _phase2_orient_and_decide(
        self,
        dossier: Phase1Dossier,
    ) -> SurgeryAgentCertificate:
        r"""
        FASE 2 · ORIENT/DECIDE  (morfismo de cierre).

        Compone motor Čech, rampa de confianza y Heyting en un `Phase2Dossier`.
        El valor de retorno es el compuesto

            Φ₂₃ ∘ Orient  :  Phase1Dossier  →  SurgeryAgentCertificate

        realizado por `_phase3_open_from_phase2`, primer morfismo de la FASE 3.
        """
        diagnostics: List[str] = list(dossier.diagnostics)

        engine_certificate = self._phase2_invoke_engine(dossier)

        engine_verdict, engine_seal, engine_interlock, verdict_diag = (
            self._phase2_parse_engine_verdict(engine_certificate)
        )
        diagnostics.extend(verdict_diag)
        diagnostics.extend(self._phase2_collect_engine_diagnostics(engine_certificate))

        (
            mismatch,
            remaining_mass,
            fiedler,
            is_globally_coherent,
            is_surgery_active,
            invalid_observables,
            noisy_cover_id,
            discarded,
            condition_number,
            spectral_gap,
            nerve_betti_0,
            active_node_count,
            observable_diag,
        ) = self._phase2_extract_observables(engine_certificate)
        diagnostics.extend(observable_diag)

        if invalid_observables:
            engine_verdict = HeytingVerdict.VETOED.value

        engine_digest = str(getattr(engine_certificate, "engine_digest", "") or "")

        is_soft, override_expired, soft_diag = self._phase2_soft_veto_state(
            mismatch=mismatch,
            lipschitz_bound=dossier.lipschitz_bound_Lmax,
            override_is_valid=dossier.override_is_valid,
        )
        diagnostics.extend(soft_diag)

        if is_soft and dossier.override_is_valid:
            logger.info(
                "Override de FASE 1 vigente (hash=%s, hmac=%s).",
                (dossier.override_token_hash or "n/d")[:16],
                dossier.override_hmac_bound,
            )
        elif is_soft:
            logger.warning(
                "¡VETO SUAVE DETECTADO EN CIRUGÍA ČECH (LUZ ÁMBAR)! "
                "Override ausente o inválido."
            )

        confidence = self._phase2_confidence_ramp(
            mismatch,
            dossier.lipschitz_bound_Lmax,
        )

        verdict, decision_diag = self._phase2_evaluate_heyting(
            invalid_observables=invalid_observables,
            engine_verdict=engine_verdict,
            is_globally_coherent=is_globally_coherent,
            is_surgery_active=is_surgery_active,
            cohomological_mismatch=mismatch,
            isolated_fock_trace=remaining_mass,
            discarded_fock_mass=discarded,
            fiedler_residual=fiedler,
            surgical_metric_cond=condition_number,
            is_soft_veto_active=is_soft,
            override_grace_period_expired=override_expired,
            override_present=dossier.override_present,
            override_valid=dossier.override_is_valid,
            lipschitz_bound_Lmax=dossier.lipschitz_bound_Lmax,
        )
        diagnostics.extend(decision_diag)

        phase2 = Phase2Dossier(
            phase1=dossier,
            cohomological_mismatch=mismatch,
            isolated_fock_trace=remaining_mass,
            fiedler_residual=fiedler,
            is_surgery_active=is_surgery_active,
            is_globally_coherent=is_globally_coherent,
            engine_verdict=engine_verdict,
            engine_seal=engine_seal,
            engine_digest=engine_digest,
            engine_interlock_fired=engine_interlock,
            noisy_cover_id=int(noisy_cover_id),
            is_soft_veto_active=is_soft,
            override_grace_period_expired=override_expired,
            heyting_verdict=verdict,
            confidence_ramp=confidence,
            discarded_fock_mass=discarded,
            surgical_metric_cond=condition_number,
            metric_spectral_gap=spectral_gap,
            nerve_betti_0=int(nerve_betti_0),
            active_node_count=int(active_node_count),
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 2 ORIENT: veredicto=%s, H¹=%.6f, masa=%.6f, λ₂=%.6e, μ=%.3f",
            verdict,
            mismatch if math.isfinite(mismatch) else float("nan"),
            remaining_mass if math.isfinite(remaining_mass) else float("nan"),
            fiedler,
            confidence,
        )

        # ── continuación anidada: el terminal de FASE 2 es el inicial de FASE 3
        return self._phase3_open_from_phase2(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Objeto inicial  : Phase2Dossier  (entregado por _phase2_orient_and_decide)
    # Objeto terminal : SurgeryAgentCertificate (inmutable, sellado, fail-closed)
    # Morfismo de apertura: _phase3_open_from_phase2
    #
    # Invariantes:
    #   (K1) VETOED ⇒ interlock crowbar simulado y latencia ∈ [380, 420] ns.
    #   (K2) El interlock del agente es redundante respecto del motor (supervisor).
    #   (K3) sello = SHA-256(dominio ‖ veredicto ‖ sesión ‖ observables).
    #   (K4) el certificado no muta tensores ni revela el token de override.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase3_open_from_phase2(
        self,
        dossier: Phase2Dossier,
    ) -> SurgeryAgentCertificate:
        r"""
        Inicio formal de la FASE 3 · ACT/CERTIFY.

        Continuación estricta del terminal de FASE 2: recibe el expediente
        auditado y abre actuación de interlock + certificación inmutable.
        """
        if not isinstance(dossier, Phase2Dossier):
            raise SurgeryAgentError(
                "FASE 3 exige un Phase2Dossier emitido por FASE 2."
            )
        return self._phase3_certify(dossier)

    def _phase3_parse_verdict(self, verdict: Any) -> HeytingVerdict:
        """Parseo total: cualquier etiqueta no reconocida colapsa a VETOED."""
        if isinstance(verdict, HeytingVerdict):
            return verdict
        try:
            return HeytingVerdict(str(verdict).strip().upper())
        except ValueError:
            return HeytingVerdict.VETOED

    def _phase3_actuate_interlock(
        self,
        verdict: Any,
        session_digest: str,
    ) -> Tuple[bool, float]:
        r"""
        FASE 3 · ACT.

        Si Heyting colapsa a VETOED, se simula la ISR en IRAM del ESP32:
        conmutación de GPIO14 y cebado del tiristor crowbar BT151 en
        400 ns ± jitter gaussiano recortado al intervalo de calibración.

        Esta actuación es el disyuntor de *supervisor*: es independiente
        del interlock que el motor Čech haya disparado en su propia FASE 3.
        """
        verdict_enum = self._phase3_parse_verdict(verdict)
        session_ref = str(session_digest)[:16]

        if verdict_enum is HeytingVerdict.VETOED:
            jitter = float(self._rng.normal(0.0, _CROWBAR_JITTER_STD_NS))
            latency = float(
                np.clip(
                    _CROWBAR_IRAM_LATENCY_NS + jitter,
                    _CROWBAR_LATENCY_MIN_NS,
                    _CROWBAR_LATENCY_MAX_NS,
                )
            )
            logger.critical(
                "¡RUPTURA DE CONSISTENCIA EN AGENTE DE ČECH! "
                "Heyting colapsó síncronamente al Supremo terminal. "
                "Conmutando GPIO%d en IRAM en %.2f ns. Crowbar BT151 gatillado. "
                "Vaciado de concreto paralizado en obra. Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Cirugía Čech regulada síncronamente en FPU. Veredicto=%s. Sello=%s",
            verdict_enum.value,
            session_ref,
        )
        return False, 0.0

    @staticmethod
    def _phase3_format_scalar(value: float) -> str:
        """Canoniza un escalar a 17 decimales; no-finitos → nan."""
        number = float(value)
        if not math.isfinite(number):
            number = float("nan")
        return f"{number:.17e}"

    def _phase3_compose_signature(
        self,
        *,
        verdict: str,
        session_digest: str,
        mismatch: float,
        trace_fock: float,
        fiedler: float,
        latency_ns: float,
        surgery_active: bool,
        soft_veto_active: bool,
        override_expired: bool,
        engine_verdict: str,
        noisy_cover_id: int,
        engine_seal: str,
        payload_digest: str,
        agent_digest: str,
        confidence_ramp: float,
        engine_interlock_fired: bool,
        override_present: bool,
        override_hmac_bound: bool,
        discarded_fock_mass: float,
        nerve_betti_0: int,
    ) -> str:
        """Firma SHA-256 final de telemetría con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_TELEMETRY)
        self._hash_update_bytes(sha, verdict.encode("utf-8"))
        self._hash_update_bytes(sha, session_digest.encode("utf-8"))
        self._hash_update_bytes(sha, payload_digest.encode("utf-8"))
        self._hash_update_bytes(sha, agent_digest.encode("utf-8"))
        self._hash_update_bytes(sha, engine_verdict.encode("utf-8"))
        self._hash_update_bytes(sha, engine_seal.encode("utf-8"))

        for value in (
            mismatch,
            trace_fock,
            fiedler,
            latency_ns,
            float(noisy_cover_id),
            confidence_ramp,
            discarded_fock_mass,
            float(nerve_betti_0),
        ):
            self._hash_update_bytes(
                sha,
                self._phase3_format_scalar(float(value)).encode("utf-8"),
            )

        for flag in (
            surgery_active,
            soft_veto_active,
            override_expired,
            engine_interlock_fired,
            override_present,
            override_hmac_bound,
        ):
            self._hash_update_bytes(sha, b"1" if bool(flag) else b"0")

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_certify(self, dossier: Phase2Dossier) -> SurgeryAgentCertificate:
        """
        FASE 3 · CERTIFY.

        Emite el certificado inmutable y acciona el interlock si el veredicto
        de Heyting es VETOED. Terminal del morfismo anidado Φ₂₃ ∘ Φ₁₂.
        """
        interlock_fired, latency = self._phase3_actuate_interlock(
            dossier.heyting_verdict,
            dossier.phase1.session_digest,
        )

        signature = self._phase3_compose_signature(
            verdict=dossier.heyting_verdict,
            session_digest=dossier.phase1.session_digest,
            mismatch=dossier.cohomological_mismatch,
            trace_fock=dossier.isolated_fock_trace,
            fiedler=dossier.fiedler_residual,
            latency_ns=latency,
            surgery_active=dossier.is_surgery_active,
            soft_veto_active=dossier.is_soft_veto_active,
            override_expired=dossier.override_grace_period_expired,
            engine_verdict=dossier.engine_verdict,
            noisy_cover_id=dossier.noisy_cover_id,
            engine_seal=dossier.engine_seal,
            payload_digest=dossier.phase1.payload_digest,
            agent_digest=dossier.phase1.agent_digest,
            confidence_ramp=dossier.confidence_ramp,
            engine_interlock_fired=dossier.engine_interlock_fired,
            override_present=dossier.phase1.override_present,
            override_hmac_bound=dossier.phase1.override_hmac_bound,
            discarded_fock_mass=dossier.discarded_fock_mass,
            nerve_betti_0=dossier.nerve_betti_0,
        )

        lower, upper = self._soft_band(dossier.phase1.lipschitz_bound_Lmax)

        certificate = SurgeryAgentCertificate(
            phase="G_WISDOM_CECH_SURGERY_SUTURATED",
            heyting_verdict=dossier.heyting_verdict,
            cohomological_mismatch=dossier.cohomological_mismatch,
            isolated_fock_trace=dossier.isolated_fock_trace,
            fiedler_residual=dossier.fiedler_residual,
            is_surgery_active=dossier.is_surgery_active,
            is_soft_veto_active=dossier.is_soft_veto_active,
            override_grace_period_expired=dossier.override_grace_period_expired,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=dossier.phase1.session_digest,
            engine_verdict=dossier.engine_verdict,
            engine_seal=dossier.engine_seal,
            noisy_cover_id=dossier.noisy_cover_id,
            soft_veto_lower_threshold=lower,
            soft_veto_upper_threshold=upper,
            diagnostics=dossier.diagnostics,
            agent_digest=dossier.phase1.agent_digest,
            payload_digest=dossier.phase1.payload_digest,
            engine_digest=dossier.engine_digest,
            engine_interlock_fired=dossier.engine_interlock_fired,
            override_present=dossier.phase1.override_present,
            override_hmac_bound=dossier.phase1.override_hmac_bound,
            confidence_ramp=dossier.confidence_ramp,
            discarded_fock_mass=dossier.discarded_fock_mass,
            surgical_metric_cond=dossier.surgical_metric_cond,
            metric_spectral_gap=dossier.metric_spectral_gap,
            nerve_betti_0=dossier.nerve_betti_0,
            active_node_count=dossier.active_node_count,
            is_globally_coherent=dossier.is_globally_coherent,
            lipschitz_bound=dossier.phase1.lipschitz_bound_Lmax,
        )

        if dossier.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "FASE 3 CERTIFY: VETO Čech. H¹=%.6f, masa=%.6f, λ₂=%.6e",
                dossier.cohomological_mismatch
                if math.isfinite(dossier.cohomological_mismatch)
                else float("nan"),
                dossier.isolated_fock_trace
                if math.isfinite(dossier.isolated_fock_trace)
                else float("nan"),
                dossier.fiedler_residual,
            )
        elif dossier.heyting_verdict == HeytingVerdict.DEGRADED.value:
            logger.warning(
                "FASE 3 CERTIFY: cirugía Čech degradada. diagnostics=%s",
                dossier.diagnostics,
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: cirugía Čech coherente. digest=%s",
                dossier.phase1.session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL  (terminal de emergencia, isomorfo a FASE 3 VETOED)
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(self, reason: str) -> SurgeryAgentCertificate:
        """
        Certificado fail-closed ante excepción no recuperable.

        Garantiza VETO, interlock y firma inmutable incluso cuando el expediente
        de entrada no pudo ser validado. Los observables imposibles se reportan
        como ∞ (no como 0, que simularía coherencia).
        """
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_SESSION)
        self._hash_update_bytes(sha, self._agent_digest.encode("utf-8"))
        self._hash_update_bytes(sha, reason.encode("utf-8"))
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        session_digest = sha.hexdigest()

        interlock_fired, latency = self._phase3_actuate_interlock(
            HeytingVerdict.VETOED.value,
            session_digest,
        )

        signature = self._phase3_compose_signature(
            verdict=HeytingVerdict.VETOED.value,
            session_digest=session_digest,
            mismatch=math.inf,
            trace_fock=0.0,
            fiedler=0.0,
            latency_ns=latency,
            surgery_active=False,
            soft_veto_active=False,
            override_expired=True,
            engine_verdict=HeytingVerdict.VETOED.value,
            noisy_cover_id=-1,
            engine_seal="",
            payload_digest="",
            agent_digest=self._agent_digest,
            confidence_ramp=1.0,
            engine_interlock_fired=False,
            override_present=False,
            override_hmac_bound=False,
            discarded_fock_mass=1.0,
            nerve_betti_0=0,
        )

        return SurgeryAgentCertificate(
            phase="G_WISDOM_CECH_SURGERY_FAIL_CLOSED",
            heyting_verdict=HeytingVerdict.VETOED.value,
            cohomological_mismatch=math.inf,
            isolated_fock_trace=0.0,
            fiedler_residual=0.0,
            is_surgery_active=False,
            is_soft_veto_active=False,
            override_grace_period_expired=True,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=session_digest,
            engine_verdict=HeytingVerdict.VETOED.value,
            engine_seal="",
            noisy_cover_id=-1,
            soft_veto_lower_threshold=math.nan,
            soft_veto_upper_threshold=math.nan,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            agent_digest=self._agent_digest,
            payload_digest="",
            engine_digest="",
            engine_interlock_fired=False,
            override_present=False,
            override_hmac_bound=False,
            confidence_ramp=1.0,
            discarded_fock_mass=1.0,
            surgical_metric_cond=math.inf,
            metric_spectral_gap=0.0,
            nerve_betti_0=0,
            active_node_count=0,
            is_globally_coherent=False,
            lipschitz_bound=math.nan,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA COMPATIBLE
    # ═══════════════════════════════════════════════════════════════════════

    def execute_surgery_control_cycle(
        self,
        boundary_matrix: np.ndarray,
        global_metric_G: np.ndarray,
        global_density_rho: np.ndarray,
        local_signals: Dict[int, np.ndarray],
        lipschitz_bound_Lmax: float,
        override_token: Optional[str] = None,
    ) -> SurgeryAgentCertificate:
        r"""
        Orquesta el lazo de control cerrado OODA de la Sonda de Cirugía
        Topológica de Čech.

        Compone las tres fases anidadas

            Φ₂₃ ∘ Φ₁₂ ∘ Observe  :  datos  →  SurgeryAgentCertificate

        Si ocurre cualquier excepción, devuelve certificado fail-closed VETOED
        (el tipo de retorno es total: nunca se propaga el fallo al llamador).
        """
        try:
            return self._phase1_observe_and_freeze(
                boundary_matrix=boundary_matrix,
                global_metric_G=global_metric_G,
                global_density_rho=global_density_rho,
                local_signals=local_signals,
                lipschitz_bound_Lmax=lipschitz_bound_Lmax,
                override_token=override_token,
            )
        except Exception as exc:  # noqa: BLE001 — contrato fail-closed
            logger.exception(
                "Fallo fail-closed en agente Čech; emitiendo VETO de frontera."
            )
            return self._fail_closed_certificate(reason=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "TopologicalSurgeryCechAgent",
    "SurgeryAgentCertificate",
    "HeytingVerdict",
    "SurgeryAgentError",
    "Phase1Dossier",
    "Phase2Dossier",
    "CechSurgeryEngineProtocol",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__
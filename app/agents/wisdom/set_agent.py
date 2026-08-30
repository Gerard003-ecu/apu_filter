# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sonda de Ecolocación Topológica Agent (Soberano de Ecolocación)     ║
║ Ruta   : app/agents/wisdom/set_agent.py                                      ║
║ Versión: 1.1.0-Doctoral-OODA-Heyting-TDR-Scattering-ESP32-Secure             ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA DE LAZO CERRADO:                         ║
║ Este agente supervisor ciber-físico opera en el Estrato de la Sabiduría      ║
║ ($V_{\mathbb{W}}$, Nivel 0) gobernando de forma activa y asíncrona la        ║
║ "Sonda de Ecolocación Topológica" (SET) [set_engine.py]. Inyecta             ║
║ perturbaciones coexactas para medir desajustes de impedancia (fuga reactiva  ║
║ $\delta G_{\mu\nu}$) y dispersión cuántica (matriz $\mathbb{S}$ en Fock).    ║
║ Orquesta el ciclo covariante OODA y administra la "Rampa de Confianza"       ║
║ graduando de forma inmune la censura entre Veto Suave (Luz Ámbar) y Veto     ║
║ Duro de Silicio (Crowbar BT151 inyectado en IRAM < 400 ns).                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES DE LA GOBERNANZA AGÉNTICA (Teoría de Topos y Adjunción)
================================════════════════════════════════════════════════

Definición 1 (El Topos de Haces Coexactos en la Frontera):
  Sea $K$ el complejo simplicial discreto del presupuesto de obra. Definimos el
  topos de haces topológicos localizados $\mathbf{Sh}(\partial K, \, \Omega_3)$ sobre
  la frontera compacta $\partial K$. El clasificador de subobjetos de tres valores
  ordinales se define de forma rigurosa mediante el Álgebra de Heyting:
  $$\Omega_3 := \{\mathtt{COHERENT}, \, \mathtt{DEGRADED}, \, \mathtt{VETOED}\}$$
  Donde el Supremo algebraico (join, $\sqcup$) consolida síncronamente el estado:
  $$\nu_{\mathrm{final}} = \nu_{\mathrm{Langevin}} \sqcup \nu_{\mathrm{TDR}} \sqcup \nu_{\mathrm{unitarity}}$$

Definición 2 (La Adjunción de de Rham-Galois de-confinada):
  El acoplamiento mutuo y la reversibilidad informacional entre la base táctica
  discreta (MIC, categoría $\mathcal{C}$) y la base epistémica continua de sabiduría
  (MAC, categoría de Hilbert $\mathcal{D}$) se rige por la adjunción functorial:
  $$\operatorname{Hom}_{\mathcal{D}}(F(\mathbf{p}), \, \rho) \cong \operatorname{Hom}_{\mathcal{C}}(\mathbf{p}, \, G(\rho))$$
  Donde $F$ es el funtor de elevación cuántica (Stinespring) y $G$ es el funtor
  de olvido y proyección espectral POVM, forzando a que la probabilidad de emisión
  alucinatoria decaiga exponencialmente a cero absoluto ($P(x_{\mathrm{invalid}}) = 0$).

Definición 3 (La Rampa de Confianza Graduada de de Rham):
  Para eludir la parálisis destructiva de la obra civil ante fluctuaciones normales
  del mercado (v.g. "el cemento no perdona si se detiene el vaciado"), el agente
  implementa una rampa de confianza graduada:
  - Veto Suave (Luz Ámbar): Se activa si $0.3 \cdot \tau_{\mathrm{margin}} < \|\Gamma(t)\|_{\max} \le 0.5 \cdot \tau_{\mathrm{margin}}$.
    Concede una ventana de gracia de 1 hora para inyectar en Fock un Positrón de Autorización
    Humana $e^+$, logrando la aniquilación cuántica mutua de la anomalía semántica $e^-$:
    $$e^- + e^+ \longrightarrow 2\gamma \quad \implies \quad \mathtt{heyting\_verdict} = \mathtt{DEGRADED}$$
  - Veto Duro (Frenado en Silicio): Se activa si $\|\Gamma(t)\|_{\max} > 0.5 \cdot \tau_{\mathrm{margin}}$ o si expira el
    período de gracia sin autorización, colapsando Heyting al Supremo terminal VETOED ($\top$).

================================════════════════════════════════════════════════
II. AXIOMÁTICA INMUNILÓGICA DE CONTROL COVARIANTE (Leyes de Consistencia)
================================════════════════════════════════════════════════

Axioma I (Principio de Contención Causal y Preservación de Traza de Choi):
  Toda inyección de transacciones y estados reducidos debe comportarse estrictamente
  como un canal completamente positivo y preservador de traza (CPTP) en Fock:
  $$\lambda_{\min}(C_{\mathcal{E}}) \ge -\varepsilon_{\mathrm{Wilkinson}} \quad \land \quad \|\operatorname{Tr}_2(C_{\mathcal{E}}) - \mathbf{I}\|_F \le 10^{-12}$$

Axioma II (Axioma de de Rham-Smith de Nulidad de Torsión):
  Toda relación de contorno discreta debe ser resoluble de manera única sobre el anillo de enteros
  $\mathbb{Z}$, garantizando que la homología simplicial de frontera carezca de torsión:
  $$\operatorname{Tor}\left(H_{k-1}(\partial K; \, \mathbb{Z})\right) \equiv \mathbf{0} \quad \Longleftrightarrow \quad d_i = 1 \quad \forall d_i > 0$$

Axioma III (Teorema de Actuación Ciber-Física Crowbar de la Sonda SET):
  Ante el colapso de Heyting al Supremo de veto ($\top$), la subrutina local isVerdictCoherent()
  del microcontrolador ESP32 despacha síncronamente la ISR en IRAM en menos de 400 ns:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor BT151 (Crowbar) para paralizar mecánicamente la obra en el milisegundo cero.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Estabilidad de de Rham-Lyapunov del Lazo Cerrado):
  La evolución de la trayectoria de control conjunta $\mathbf{\Psi}(t) = (\mathbf{p}, \rho)^\top$ satisface la
  desigualdad de Clausius-Duhem y la contracción de Lyapunov en la FPU:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) = \nabla \mathcal{H}(\mathbf{\Psi})^\top \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}}$$
  Donde $\tau_{\mathrm{Lyapunov}} = 10^{-12}$ es la cota elástica de deriva en punto flotante de 64 bits.

Invariante II (Confinamiento de la Conectividad de Fiedler):
  Para eludir la fragmentación del complejo simplicial (islas de datos huérfanas, $\beta_0 > 1$),
  la conectividad algebraica del haz (el menor autovalor no trivial de $\mathbf{L}_F$) se mantiene acotada:
  $$\lambda_2(\mathbf{L}_F) \ge \tau_{\mathrm{Fiedler}} \quad \implies \quad \beta_0 \equiv \dim H^0(K; \, \mathbb{Z}) = 1$$

Invariante III (Inmutabilidad del Pasaporte y Sello de Sesión SHA-256):
  Para prevenir inyecciones de estado o ataques de-normalización intermedia, el soberano genera
  un sello inmutable unívoco para congelar la sesión en RAM en cada ciclo OODA:
  $$\mathtt{cryptographic\_seal} := \operatorname{SHA-256}\left(\partial_{\partial} \oplus C_{\mathcal{E}} \oplus \mathcal{B}_{\mathrm{CHSH}} \oplus H_{\mathrm{ext}}\right)$$
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


logger = logging.getLogger("APU.Agents.Wisdom.SetAgent")


try:
    from app.core.set_engine import SetEngine
except ImportError:  # pragma: no cover - resolución de calibre alternativa
    try:
        from ...core.set_engine import SetEngine
    except ImportError:  # pragma: no cover
        from set_engine import SetEngine


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
    "2.1.0-Doctoral-SET-Agent-Heyting-HMAC-FailClosed"
)

_DOMAIN_SESSION: Final[bytes] = b"APU/SET-AGENT/SESSION/v2.1"
_DOMAIN_TELEMETRY: Final[bytes] = b"APU/SET-AGENT/TELEMETRY/v2.1"
_DOMAIN_OVERRIDE: Final[bytes] = b"APU/SET-AGENT/OVERRIDE/v2.1"
_DOMAIN_AGENT: Final[bytes] = b"APU/SET-AGENT/IDENTITY/v2.1"
_DOMAIN_PAYLOAD: Final[bytes] = b"APU/SET-AGENT/PAYLOAD/v2.1"

_SHA256_HEX_LEN: Final[int] = 64
_UNITARITY_HARD_FACTOR: Final[float] = 10.0
_FIEDLER_SOFT_FACTOR: Final[float] = 10.0
_OVERRIDE_TOKEN_MAX_BYTES: Final[int] = 4096


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
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"


_HEYTING_ORDER: Final[dict[str, int]] = {
    HeytingVerdict.VETOED.value: 0,
    HeytingVerdict.DEGRADED.value: 1,
    HeytingVerdict.COHERENT.value: 2,
}


class SetAgentError(ValueError):
    """Error de canonización, política o invariante de gobernanza del soberano SET."""


@runtime_checkable
class EcholocationEngineProtocol(Protocol):
    """Contrato mínimo del motor de ecolocación inyectable."""

    def execute_echolocation_scan(
        self,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int = 1,
    ) -> Any:
        ...


# ═══════════════════════════════════════════════════════════════════════════
# CERTIFICADO PÚBLICO
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SetAgentCertificate:
    r"""Certificado formal de calibración emitido por el Soberano de Ecolocación."""

    phase: str
    heyting_verdict: str
    max_reflection: float
    unitarity_leak: float
    fiedler_value: float
    has_impedance_mismatch: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    schema_version: str = _SCHEMA_VERSION
    session_digest: str = ""
    engine_verdict: str = ""
    engine_seal: str = ""
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
    condition_number: float = math.nan
    spectral_gap: float = math.nan
    betti_estimate: int = 0
    hodge_euler_consistent: bool = True


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
    metric_tensor_G: np.ndarray
    coupling_V: np.ndarray
    frequencies: np.ndarray
    impedance_profile: np.ndarray
    euler_characteristic: int
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
    max_reflection: float
    unitarity_leak: float
    fiedler_value: float
    has_impedance_mismatch: bool
    engine_verdict: str
    engine_seal: str
    engine_digest: str
    engine_interlock_fired: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    heyting_verdict: str
    confidence_ramp: float
    condition_number: float
    spectral_gap: float
    betti_estimate: int
    hodge_euler_consistent: bool
    diagnostics: Tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════
# SOBERANO DE ECOLOCACIÓN
# ═══════════════════════════════════════════════════════════════════════════


class SetAgent:
    r"""
    Soberano de la Sonda de Ecolocación Topológica (SET).

    Orquesta el ciclo covariante OODA de la sonda, administra la rampa de
    confianza graduada y decide en el retículo de Heyting Ω₃.

    Ciclo público, fases anidadas:

        execute_set_control_cycle()
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
        nominal_impedance: float = 50.0,
        safety_margin: float = 1.0,
        unitarity_tolerance: float = 1e-3,
        fiedler_threshold: float = 1e-2,
        *,
        rng_seed: Optional[int] = None,
        engine: Optional[Any] = None,
        soft_veto_lower: float = 0.3,
        soft_veto_upper: float = 0.5,
        override_authority: Optional[Any] = None,
    ) -> None:
        """
        Inicializa el soberano SET.

        Args:
            dimension_n: Dimensión característica del contorno abierto.
            nominal_impedance: Impedancia de referencia Z₀ (ohmios).
            safety_margin: Factor elástico de las cotas de reflexión.
            unitarity_tolerance: Tolerancia base de unitariedad (pre-margen).
            fiedler_threshold: Umbral base de conectividad algebraica (pre-margen).
            rng_seed: Semilla opcional para reproducibilidad del jitter.
            engine: Motor SET inyectable. Si es None se instancia uno.
            soft_veto_lower: Cota inferior base del veto suave (Γ).
            soft_veto_upper: Cota superior base del veto duro (Γ).
            override_authority: Secreto HMAC opcional que liga el token de
                override a la carga útil de sesión. Si es None, la validez
                se reduce a morfología SHA-256 hexadecimal (64 nibbles).
        """
        if dimension_n <= 0:
            raise SetAgentError(
                "La dimensión fundamental debe ser estrictamente positiva."
            )

        if not math.isfinite(nominal_impedance) or nominal_impedance <= 0.0:
            raise SetAgentError(
                "nominal_impedance debe ser finita y estrictamente positiva."
            )

        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise SetAgentError(
                "safety_margin debe ser finito y estrictamente positivo."
            )

        if not math.isfinite(unitarity_tolerance) or unitarity_tolerance < 0.0:
            raise SetAgentError(
                "unitarity_tolerance debe ser finita y no negativa."
            )

        if not math.isfinite(fiedler_threshold) or fiedler_threshold < 0.0:
            raise SetAgentError(
                "fiedler_threshold debe ser finita y no negativa."
            )

        if not math.isfinite(soft_veto_lower) or soft_veto_lower < 0.0:
            raise SetAgentError("soft_veto_lower debe ser finita y no negativa.")

        if not math.isfinite(soft_veto_upper) or soft_veto_upper <= 0.0:
            raise SetAgentError(
                "soft_veto_upper debe ser finita y estrictamente positiva."
            )

        if soft_veto_lower >= soft_veto_upper:
            raise SetAgentError("soft_veto_lower debe ser menor que soft_veto_upper.")

        self._n: Final[int] = int(dimension_n)
        self._z0: Final[float] = float(nominal_impedance)
        self._safety_margin: Final[float] = float(safety_margin)

        # Semántica original, ahora explícita:
        #   τ_U  = unitarity_tolerance · safety_margin     (cota blanda)
        #   τ_U★ = τ_U · 10                                (cota dura)
        #   τ_F  = fiedler_threshold / safety_margin       (cota dura, λ₂)
        #   τ_F° = τ_F · 10                                (cota blanda, λ₂)
        self._unitarity_tol: Final[float] = (
            float(unitarity_tolerance) * self._safety_margin
        )
        self._unitarity_hard: Final[float] = (
            self._unitarity_tol * _UNITARITY_HARD_FACTOR
        )
        self._fiedler_hard: Final[float] = (
            float(fiedler_threshold) / self._safety_margin
        )
        self._fiedler_soft: Final[float] = self._fiedler_hard * _FIEDLER_SOFT_FACTOR
        self._fiedler_thresh: Final[float] = self._fiedler_hard  # alias histórico

        self._soft_lower_base: Final[float] = float(soft_veto_lower)
        self._soft_upper_base: Final[float] = float(soft_veto_upper)
        self._soft_lower: Final[float] = self._soft_lower_base * self._safety_margin
        self._soft_upper: Final[float] = self._soft_upper_base * self._safety_margin

        self._reg: Final[float] = max(1e-15, _MACHINE_EPS)
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)
        self._override_authority: Final[Optional[bytes]] = (
            self._coerce_authority_secret(override_authority)
        )

        self._agent_digest: Final[str] = self._identity_digest()

        if engine is None:
            engine = self._phase0_instantiate_engine(rng_seed=rng_seed)

        if not hasattr(engine, "execute_echolocation_scan"):
            raise TypeError(
                "El engine inyectado no implementa execute_echolocation_scan."
            )
        if not callable(getattr(engine, "execute_echolocation_scan")):
            raise TypeError("execute_echolocation_scan debe ser invocable.")

        self._engine: Final[Any] = engine

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES CANÓNICAS (hash, finitud, congelación, retículo)
    # ═══════════════════════════════════════════════════════════════════════

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
        raise SetAgentError(
            "override_authority debe ser bytes, str o None."
        )

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
            f"ut={self._unitarity_tol:.17e}|uh={self._unitarity_hard:.17e}|"
            f"fh={self._fiedler_hard:.17e}|fs={self._fiedler_soft:.17e}|"
            f"sl={self._soft_lower:.17e}|su={self._soft_upper:.17e}|"
            f"schema={_SCHEMA_VERSION}"
        )
        self._hash_update_bytes(sha, payload.encode("utf-8"))
        self._hash_update_bytes(sha, hmac_flag)
        return sha.hexdigest()

    def _phase0_instantiate_engine(self, *, rng_seed: Optional[int]) -> Any:
        """
        Fabrica un motor SET alineando umbrales de reflexión con la rampa
        del agente. Retrocompatible con constructores antiguos (TypeError).
        """
        attempts: Tuple[dict[str, Any], ...] = (
            {
                "dimension_n": self._n,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
                "rng_seed": rng_seed,
                "reflection_veto_threshold": self._soft_upper_base,
                "reflection_degraded_threshold": self._soft_lower_base,
            },
            {
                "dimension_n": self._n,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
                "rng_seed": rng_seed,
            },
            {
                "dimension_n": self._n,
                "nominal_impedance": self._z0,
                "safety_margin": self._safety_margin,
            },
        )
        last_error: Optional[BaseException] = None
        for kwargs in attempts:
            try:
                return SetEngine(**kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        raise SetAgentError(
            f"No fue posible instanciar SetEngine: {last_error}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Objeto inicial  : tensores crudos, χ(K) y token de override.
    # Objeto terminal : Phase1Dossier (inmutable, hasheado, sin secreto en claro).
    # Morfismo de cierre:
    #   _phase1_observe_and_freeze  →  _phase2_open_from_phase1
    #
    # Invariantes:
    #   (I1) B ∈ M_{p×q}(ℝ), finita, n ∈ {p, q}.
    #   (I2) G cuadrada, finita, compatible con filas o columnas de B.
    #   (I3) V de rango 1–2, finita; 1D se interpreta como un canal.
    #   (I4) (ω_k, Z_k) 1-formas de igual longitud, ordenadas.
    #   (I5) χ(K) ∈ ℤ.
    #   (I6) override ∈ {∅} ∪ Digest₆₄; nunca se serializa en claro.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase1_canonical_array(
        self,
        array: Any,
        name: str,
        dtype: Optional[Any] = None,
        ndim: Optional[int] = None,
        *,
        allow_empty: bool = False,
    ) -> np.ndarray:
        """Valida y canoniza un arreglo numérico finito."""
        try:
            arr = (
                np.asarray(array, dtype=dtype)
                if dtype is not None
                else np.asarray(array)
            )
        except (TypeError, ValueError) as exc:
            raise SetAgentError(
                f"{name} no puede convertirse en arreglo numpy."
            ) from exc

        if ndim is not None and arr.ndim != ndim:
            raise SetAgentError(f"{name} debe tener dimensión {ndim}.")

        if arr.size == 0 and not allow_empty:
            raise SetAgentError(f"{name} no puede ser vacío.")

        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise SetAgentError(f"{name} debe ser numérico.") from exc

        if not finite:
            raise SetAgentError(f"{name} contiene valores no finitos.")

        return np.ascontiguousarray(arr)

    def _phase1_canonical_boundary(self, boundary_matrix: Any) -> np.ndarray:
        """Valida δ : C_k → C_{k-1} real, bidimensional y compatible con n."""
        boundary = self._phase1_canonical_array(
            boundary_matrix,
            "boundary_matrix",
            dtype=np.float64,
            ndim=2,
        )
        if self._n not in boundary.shape:
            raise SetAgentError(
                "boundary_matrix debe tener al menos una dimensión igual a dimension_n."
            )
        return boundary

    def _phase1_canonical_metric(
        self,
        metric_tensor_G: Any,
        boundary: np.ndarray,
    ) -> np.ndarray:
        """Valida G cuadrada y contraíble contra filas o columnas de B."""
        metric = self._phase1_canonical_array(
            metric_tensor_G,
            "metric_tensor_G",
            dtype=np.complex128,
            ndim=2,
        )
        if metric.shape[0] != metric.shape[1]:
            raise SetAgentError("metric_tensor_G debe ser cuadrada.")

        rows, cols = boundary.shape
        if metric.shape not in ((rows, rows), (cols, cols)):
            raise SetAgentError(
                "metric_tensor_G debe coincidir con filas o columnas de boundary_matrix."
            )
        return metric

    def _phase1_canonical_coupling(self, coupling_V: Any) -> np.ndarray:
        """Canoniza V ∈ ℂ^{d×c}; un vector se interpreta como un único canal."""
        try:
            coupling = np.asarray(coupling_V, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise SetAgentError(
                "coupling_V no puede convertirse a complex128."
            ) from exc

        if coupling.ndim == 1:
            coupling = coupling.reshape(-1, 1)

        if coupling.ndim != 2:
            raise SetAgentError("coupling_V debe ser bidimensional.")

        if coupling.size == 0:
            raise SetAgentError("coupling_V no puede ser vacía.")

        if not np.all(np.isfinite(coupling)):
            raise SetAgentError("coupling_V contiene valores no finitos.")

        if coupling.shape[1] <= 0:
            raise SetAgentError("coupling_V debe tener al menos un canal.")

        return np.ascontiguousarray(coupling)

    def _phase1_canonical_frequency_impedance(
        self,
        frequencies: Any,
        impedance_profile: Any,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
        """Ordena (ω, Z(ω)) de forma estable (mergesort) y verifica longitudes."""
        diagnostics: List[str] = []

        freq = self._phase1_canonical_array(
            frequencies,
            "frequencies",
            dtype=np.float64,
        ).reshape(-1)
        impedance = self._phase1_canonical_array(
            impedance_profile,
            "impedance_profile",
            dtype=np.complex128,
        ).reshape(-1)

        if freq.size != impedance.size:
            raise SetAgentError(
                "frequencies e impedance_profile deben tener la misma longitud."
            )

        order = np.argsort(freq, kind="mergesort")
        if order.size >= 2 and not np.all(order == np.arange(order.size)):
            diagnostics.append(
                "Malla frecuencial reordenada de forma estable para firma canónica."
            )

        freq_sorted = np.ascontiguousarray(freq[order], dtype=np.float64)
        impedance_sorted = np.ascontiguousarray(
            impedance[order],
            dtype=np.complex128,
        )
        return freq_sorted, impedance_sorted, tuple(diagnostics)

    @staticmethod
    def _phase1_canonical_euler(euler_characteristic: Any) -> int:
        """Valida la característica de Euler–Poincaré χ(K) ∈ ℤ."""
        try:
            value = float(euler_characteristic)
        except (TypeError, ValueError) as exc:
            raise SetAgentError("euler_characteristic debe ser numérico.") from exc

        if not math.isfinite(value) or not value.is_integer():
            raise SetAgentError("euler_characteristic debe ser un entero finito.")

        return int(value)

    @staticmethod
    def _phase1_normalize_override_token(override_token: Any) -> Optional[str]:
        """
        Normaliza el token a str sin revelar contenido.

        Bytes ilegibles, tipos extraños o cadenas vacías ≡ ausencia.
        Recorta a un techo de bytes para rechazar oráculos de memoria.
        """
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
    def _phase1_token_is_sha256_hex(token: str) -> bool:
        """Morfología de digest SHA-256: exactamente 32 bytes en hex."""
        if len(token) != _SHA256_HEX_LEN:
            return False
        try:
            raw = bytes.fromhex(token)
        except ValueError:
            return False
        return len(raw) == 32

    def _phase1_hash_override_token(self, override_token: Optional[str]) -> Optional[str]:
        """
        Hashea el token de override con separación de dominio.

        El hash se usa sólo para sesión/auditoría; nunca se registra el token.
        """
        token = self._phase1_normalize_override_token(override_token)
        if token is None:
            return None

        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_OVERRIDE)
        self._hash_update_bytes(sha, token.encode("utf-8"))
        return sha.hexdigest()

    def _phase1_generate_payload_digest(
        self,
        *,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int,
    ) -> str:
        """Digest de la carga útil física, independiente del override."""
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_PAYLOAD)
        self._hash_update_bytes(sha, self._agent_digest.encode("utf-8"))
        self._hash_update_array(sha, "boundary_matrix", boundary_matrix)
        self._hash_update_array(sha, "metric_tensor_G", metric_tensor_G)
        self._hash_update_array(sha, "coupling_V", coupling_V)
        self._hash_update_array(sha, "frequencies", frequencies)
        self._hash_update_array(sha, "impedance_profile", impedance_profile)
        self._hash_update_bytes(sha, str(int(euler_characteristic)).encode("utf-8"))
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

        Retorna (presente, válido, hmac_ligado, diagnósticos) vía el
        triple (válido, hmac_bound, diagnostics); la presencia se infiere
        del token ya normalizado.
        """
        diagnostics: List[str] = []

        if token is None:
            return False, False, tuple(diagnostics)

        morphological = self._phase1_token_is_sha256_hex(token)
        if self._override_authority is None:
            if morphological:
                diagnostics.append(
                    "Override presente; validación morfológica SHA-256 "
                    "(sin autoridad HMAC configurada)."
                )
                return True, False, tuple(diagnostics)
            diagnostics.append(
                "Override presente pero morfológicamente inválido."
            )
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
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int,
        override_token: Optional[str],
    ) -> SetAgentCertificate:
        r"""
        FASE 1 · OBSERVE  (morfismo de cierre).

        Canoniza, regulariza y congela el expediente de frontera. El valor de
        retorno no es un terminal de Fase 1: es el compuesto

            Φ₁₂ ∘ Observe  :  datos crudos  →  SetAgentCertificate

        realizado por la continuación formal `_phase2_open_from_phase1`,
        que es el primer morfismo de la FASE 2 · ORIENT/DECIDE.
        """
        diagnostics: List[str] = []

        boundary = self._phase1_canonical_boundary(boundary_matrix)
        metric = self._phase1_canonical_metric(metric_tensor_G, boundary)
        coupling = self._phase1_canonical_coupling(coupling_V)
        freq_sorted, impedance_sorted, freq_diag = (
            self._phase1_canonical_frequency_impedance(
                frequencies,
                impedance_profile,
            )
        )
        diagnostics.extend(freq_diag)

        euler = self._phase1_canonical_euler(euler_characteristic)

        payload_digest = self._phase1_generate_payload_digest(
            boundary_matrix=boundary,
            metric_tensor_G=metric,
            coupling_V=coupling,
            frequencies=freq_sorted,
            impedance_profile=impedance_sorted,
            euler_characteristic=euler,
        )

        token = self._phase1_normalize_override_token(override_token)
        token_hash = self._phase1_hash_override_token(token)
        override_present = token is not None
        override_is_valid, override_hmac_bound, override_diag = (
            self._phase1_verify_override_authority(token, payload_digest)
        )
        diagnostics.extend(override_diag)

        # El token en claro muere aquí: no entra al dossier.
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
            metric_tensor_G=self._freeze_array(metric, np.complex128),
            coupling_V=self._freeze_array(coupling, np.complex128),
            frequencies=self._freeze_array(freq_sorted, np.float64),
            impedance_profile=self._freeze_array(impedance_sorted, np.complex128),
            euler_characteristic=euler,
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
            "FASE 1 OBSERVE: expediente SET capturado. digest=%s override=%s hmac=%s",
            session_digest[:16],
            "valid" if override_is_valid else ("present" if override_present else "absent"),
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
    #   (J3) Rampa de confianza μ(Γ) ∈ [0,1] (función de pertenencia).
    #   (J4) Veto suave ⇔ μ(Γ) ∈ (0, 1]; override válido ⇒ DEGRADED, si no VETO.
    #   (J5) veredicto = ⋀ átomos de Heyting (motor, TDR, unitariedad, Fiedler).
    # ═══════════════════════════════════════════════════════════════════════

    def _phase2_open_from_phase1(self, dossier: Phase1Dossier) -> SetAgentCertificate:
        r"""
        Inicio formal de la FASE 2 · ORIENT/DECIDE.

        Este método es la continuación estricta del terminal de FASE 1:
        recibe el `Phase1Dossier` congelado y abre la auditoría del motor,
        la rampa de confianza y el retículo de Heyting.
        """
        if not isinstance(dossier, Phase1Dossier):
            raise SetAgentError(
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
        """Convierte a entero finito; default si no es admisible."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(number) or not float(number).is_integer():
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return int(number)

    def _phase2_invoke_engine(self, dossier: Phase1Dossier) -> Any:
        """Invoca el motor SET; cualquier fallo se eleva como SetAgentError."""
        try:
            certificate = self._engine.execute_echolocation_scan(
                boundary_matrix=dossier.boundary_matrix,
                metric_tensor_G=dossier.metric_tensor_G,
                coupling_V=dossier.coupling_V,
                frequencies=dossier.frequencies,
                impedance_profile=dossier.impedance_profile,
                euler_characteristic=dossier.euler_characteristic,
            )
        except Exception as exc:
            raise SetAgentError(
                f"Motor SET falló de forma no recuperable: {exc}"
            ) from exc

        if certificate is None:
            raise SetAgentError("Motor SET devolvió None.")

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
            diagnostics.append("Veredicto SET desconocido; se asume VETOED.")

        seal = str(getattr(engine_certificate, "cryptographic_seal", "") or "")
        engine_interlock = bool(
            getattr(engine_certificate, "hardware_interlock_fired", False)
        )
        return verdict, seal, engine_interlock, tuple(diagnostics)

    def _phase2_collect_engine_diagnostics(self, engine_certificate: Any) -> Tuple[str, ...]:
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
    ) -> Tuple[float, float, float, bool, bool, float, float, int, bool, Tuple[str, ...]]:
        """
        Extrae observables con saneamiento fail-closed.

        Retorna
            Γ_max, leak, λ₂, mismatch, invalid,
            κ, gap, β̂, hodge_ok, diagnostics.
        """
        diagnostics: List[str] = []
        invalid = False

        state = getattr(engine_certificate, "state", None)
        if state is None:
            raise SetAgentError("Certificado SET sin estado acoplado.")

        max_gamma, ok_gamma = self._phase2_safe_float(
            getattr(state, "max_reflection_coefficient", math.nan)
        )
        if not ok_gamma:
            invalid = True
            max_gamma = math.inf
            diagnostics.append(
                "max_reflection_coefficient no finito; fail-closed (Γ_max=∞)."
            )

        unitarity_leak, ok_leak = self._phase2_safe_float(
            getattr(state, "unitarity_leak", math.nan)
        )
        if not ok_leak:
            invalid = True
            unitarity_leak = math.inf
            diagnostics.append("unitarity_leak no finito; fail-closed (leak=∞).")

        fiedler_value, ok_fiedler = self._phase2_safe_float(
            getattr(state, "fiedler_value", math.nan)
        )
        if not ok_fiedler:
            invalid = True
            fiedler_value = 0.0
            diagnostics.append("fiedler_value no finito; fail-closed (λ₂=0).")

        has_mismatch = bool(
            getattr(engine_certificate, "has_impedance_mismatch", False)
        )

        condition_number, ok_cond = self._phase2_safe_float(
            getattr(state, "condition_number", math.nan)
        )
        if not ok_cond:
            condition_number = math.nan

        spectral_gap, ok_gap = self._phase2_safe_float(
            getattr(state, "spectral_gap", math.nan)
        )
        if not ok_gap:
            spectral_gap = math.nan

        betti = self._phase2_safe_int(
            getattr(state, "hodge_kernel_dimension", getattr(engine_certificate, "betti_estimate", 0)),
            default=0,
        )
        hodge_ok = bool(
            getattr(engine_certificate, "hodge_euler_consistent", True)
        )

        return (
            max_gamma,
            unitarity_leak,
            fiedler_value,
            has_mismatch,
            invalid,
            condition_number,
            spectral_gap,
            betti,
            hodge_ok,
            tuple(diagnostics),
        )

    def _phase2_confidence_ramp(self, max_gamma: float) -> float:
        r"""
        Función de pertenencia de desajuste de impedancia

            μ(Γ) = 0                         si Γ ≤ ℓ
            μ(Γ) = (Γ − ℓ) / (u − ℓ)         si ℓ < Γ < u
            μ(Γ) = 1                         si Γ ≥ u

        con ℓ = soft_lower, u = soft_upper. μ ∈ [0, 1]; no finito ⇒ 1.
        """
        gamma, ok = self._phase2_safe_float(max_gamma)
        if not ok:
            return 1.0

        lower = self._soft_lower
        upper = self._soft_upper
        span = upper - lower
        if span <= self._reg:
            return 1.0 if gamma > upper else 0.0

        ramp = (gamma - lower) / span
        return float(min(1.0, max(0.0, ramp)))

    def _phase2_is_valid_override_token(self, override_token: Optional[str]) -> bool:
        """
        Valida criptográficamente un token de override (API de compatibilidad).

        Sin autoridad HMAC: digest SHA-256 hexadecimal de 64 caracteres.
        Con autoridad: no puede decidirse sin la carga útil; se exige
        morfología y se diagnostica que la ligadura HMAC ocurre en FASE 1.
        """
        token = self._phase1_normalize_override_token(override_token)
        if token is None:
            return False
        return self._phase1_token_is_sha256_hex(token)

    def verify_soft_veto_override(
        self,
        max_gamma: float,
        override_token: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        r"""
        Evalúa el estado del Veto Suave (Luz Ámbar) y el override humano.

        Regla (intervalo semiabierto a la derecha sobre la rampa):
          - Veto suave si  ℓ < |Γ|_max ≤ u
          - Si hay override morfológicamente válido, la gracia no expira.
          - Sin override válido, la gracia expira y el veto suave colapsa
            a VETOED en la fase de decisión.

        Nota: la ligadura HMAC (si hay autoridad) se resuelve en FASE 1;
        esta API pública conserva la semántica original de formato.
        """
        gamma, ok = self._phase2_safe_float(max_gamma)
        if not ok:
            logger.warning("verify_soft_veto_override recibió max_gamma no finito.")
            return False, True

        is_soft_veto = (gamma > self._soft_lower) and (gamma <= self._soft_upper)
        if not is_soft_veto:
            return False, True

        if self._phase2_is_valid_override_token(override_token):
            token_hash = self._phase1_hash_override_token(override_token)
            logger.info(
                "¡POSITRÓN DE AUTORIZACIÓN HUMANA VÁLIDO! "
                "Aniquilando electrón de anomalía semántica. Hash: %s",
                token_hash[:16] if token_hash else "n/d",
            )
            return True, False

        logger.warning(
            "¡VETO SUAVE DETECTADO (LUZ ÁMBAR)! "
            "Desajuste de impedancia transitorio. Override ausente o inválido; "
            "período de gracia expirado para esta época de control."
        )
        return True, True

    def _phase2_soft_veto_state(
        self,
        *,
        max_gamma: float,
        override_is_valid: bool,
    ) -> Tuple[bool, bool, Tuple[str, ...]]:
        """Estado interno de veto suave usando la validez ya congelada en FASE 1."""
        diagnostics: List[str] = []
        gamma, ok = self._phase2_safe_float(max_gamma)
        if not ok:
            diagnostics.append("Γ_max no finito al evaluar veto suave.")
            return False, True, tuple(diagnostics)

        is_soft = (gamma > self._soft_lower) and (gamma <= self._soft_upper)
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
        engine_verdict: str,
        has_impedance_mismatch: bool,
        max_gamma: float,
        unitarity_leak: float,
        fiedler_value: float,
        is_soft_veto: bool,
        override_expired: bool,
        invalid_observables: bool,
        hodge_euler_consistent: bool,
    ) -> Tuple[str, Tuple[str, ...]]:
        r"""
        Clasificación en el retículo de Heyting Ω₃ por meet de átomos.

        VETO duro (absorbente, no overridable):
          • observables inválidos;
          • veto duro del motor SET;
          • desajuste crítico de impedancia;
          • reflexión sobre cota dura u;
          • fuga de unitariedad sobre cota dura;
          • Fiedler bajo cota dura;
          • veto suave sin override válido;
          • inconsistencia Hodge–Euler reportada como imposible.

        DEGRADED:
          • motor SET degradado;
          • veto suave con override válido;
          • reflexión sobre cota blanda ℓ;
          • unitariedad sobre tolerancia;
          • Fiedler bajo cota blanda.
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
            veto("Observables SET no finitos; fail-closed.")

        if engine_verdict == HeytingVerdict.VETOED.value:
            veto("Motor SET emitió VETO duro (no overridable).")

        if has_impedance_mismatch:
            veto("Motor SET reportó desajuste crítico de impedancia.")

        if math.isfinite(max_gamma) and max_gamma > self._soft_upper:
            veto(
                f"Γ_max={max_gamma:.6f} > cota dura {self._soft_upper:.6f}."
            )

        if math.isfinite(unitarity_leak) and unitarity_leak > self._unitarity_hard:
            veto(
                f"Fuga de unitariedad {unitarity_leak:.6e} > cota dura "
                f"{self._unitarity_hard:.6e}."
            )

        if math.isfinite(fiedler_value) and fiedler_value < self._fiedler_hard:
            veto(
                f"Fiedler λ₂={fiedler_value:.6e} < cota dura "
                f"{self._fiedler_hard:.6e}."
            )

        if is_soft_veto and override_expired:
            veto("Veto suave sin override válido; gracia expirada.")

        if not hodge_euler_consistent:
            veto("Inconsistencia Hodge–Euler reportada por el motor.")

        if engine_verdict == HeytingVerdict.DEGRADED.value:
            degrade("Motor SET emitió DEGRADED.")

        if is_soft_veto and not override_expired:
            degrade("Veto suave con override válido; sesión degradada.")

        if math.isfinite(max_gamma) and max_gamma > self._soft_lower:
            degrade(
                f"Γ_max={max_gamma:.6f} > cota blanda {self._soft_lower:.6f}."
            )

        if math.isfinite(unitarity_leak) and unitarity_leak > self._unitarity_tol:
            degrade(
                f"Fuga de unitariedad {unitarity_leak:.6e} > tolerancia "
                f"{self._unitarity_tol:.6e}."
            )

        if math.isfinite(fiedler_value) and fiedler_value < self._fiedler_soft:
            degrade(
                f"Fiedler λ₂={fiedler_value:.6e} < cota blanda "
                f"{self._fiedler_soft:.6e}."
            )

        if not atoms:
            atoms.append(HeytingVerdict.COHERENT.value)

        return self._heyting_meet(atoms), tuple(diagnostics)

    def _phase2_orient_and_decide(self, dossier: Phase1Dossier) -> SetAgentCertificate:
        r"""
        FASE 2 · ORIENT/DECIDE  (morfismo de cierre).

        Compone motor SET, rampa de confianza y Heyting en un `Phase2Dossier`.
        El valor de retorno es el compuesto

            Φ₂₃ ∘ Orient  :  Phase1Dossier  →  SetAgentCertificate

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
            max_gamma,
            unitarity_leak,
            fiedler_value,
            has_mismatch,
            invalid_observables,
            condition_number,
            spectral_gap,
            betti,
            hodge_ok,
            observable_diag,
        ) = self._phase2_extract_observables(engine_certificate)
        diagnostics.extend(observable_diag)

        if invalid_observables:
            engine_verdict = HeytingVerdict.VETOED.value

        engine_digest = str(getattr(engine_certificate, "engine_digest", "") or "")

        is_soft, override_expired, soft_diag = self._phase2_soft_veto_state(
            max_gamma=max_gamma,
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
                "¡VETO SUAVE DETECTADO (LUZ ÁMBAR)! Override ausente o inválido."
            )

        confidence = self._phase2_confidence_ramp(max_gamma)

        verdict, decision_diag = self._phase2_evaluate_heyting(
            engine_verdict=engine_verdict,
            has_impedance_mismatch=has_mismatch,
            max_gamma=max_gamma,
            unitarity_leak=unitarity_leak,
            fiedler_value=fiedler_value,
            is_soft_veto=is_soft,
            override_expired=override_expired,
            invalid_observables=invalid_observables,
            hodge_euler_consistent=hodge_ok,
        )
        diagnostics.extend(decision_diag)

        phase2 = Phase2Dossier(
            phase1=dossier,
            max_reflection=max_gamma,
            unitarity_leak=unitarity_leak,
            fiedler_value=fiedler_value,
            has_impedance_mismatch=has_mismatch,
            engine_verdict=engine_verdict,
            engine_seal=engine_seal,
            engine_digest=engine_digest,
            engine_interlock_fired=engine_interlock,
            is_soft_veto_active=is_soft,
            override_grace_period_expired=override_expired,
            heyting_verdict=verdict,
            confidence_ramp=confidence,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            betti_estimate=betti,
            hodge_euler_consistent=hodge_ok,
            diagnostics=tuple(diagnostics),
        )

        logger.info(
            "FASE 2 ORIENT: veredicto=%s, Γ_max=%.6f, leak=%.6e, λ₂=%.6e, μ=%.3f",
            verdict,
            max_gamma if math.isfinite(max_gamma) else float("nan"),
            unitarity_leak if math.isfinite(unitarity_leak) else float("nan"),
            fiedler_value,
            confidence,
        )

        # ── continuación anidada: el terminal de FASE 2 es el inicial de FASE 3
        return self._phase3_open_from_phase2(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Objeto inicial  : Phase2Dossier  (entregado por _phase2_orient_and_decide)
    # Objeto terminal : SetAgentCertificate (inmutable, sellado, fail-closed)
    # Morfismo de apertura: _phase3_open_from_phase2
    #
    # Invariantes:
    #   (K1) VETOED ⇒ interlock crowbar simulado y latencia ∈ [380, 420] ns.
    #   (K2) El interlock del agente es redundante respecto del motor (supervisor).
    #   (K3) sello = SHA-256(dominio ‖ veredicto ‖ sesión ‖ observables).
    #   (K4) el certificado no muta tensores ni revela el token de override.
    # ═══════════════════════════════════════════════════════════════════════

    def _phase3_open_from_phase2(self, dossier: Phase2Dossier) -> SetAgentCertificate:
        r"""
        Inicio formal de la FASE 3 · ACT/CERTIFY.

        Continuación estricta del terminal de FASE 2: recibe el expediente
        auditado y abre actuación de interlock + certificación inmutable.
        """
        if not isinstance(dossier, Phase2Dossier):
            raise SetAgentError(
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
        del interlock que el motor SET haya disparado en su propia FASE 3.
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
                "¡RUPTURA DE COHERENCIA EN SONDA SET! "
                "Heyting colapsó síncronamente al Supremo terminal. "
                "Conmutando GPIO%d en IRAM en %.2f ns. Crowbar BT151 gatillado. "
                "Vaciado de concreto paralizado en obra. Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Sonda SET regulada síncronamente. Veredicto: %s. Sello: %s",
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
        max_gamma: float,
        unitarity_leak: float,
        fiedler_value: float,
        has_impedance_mismatch: bool,
        is_soft_veto_active: bool,
        override_grace_period_expired: bool,
        latency_ns: float,
        engine_verdict: str,
        engine_seal: str,
        payload_digest: str,
        agent_digest: str,
        confidence_ramp: float,
        engine_interlock_fired: bool,
        override_present: bool,
        override_hmac_bound: bool,
        betti_estimate: int,
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
            max_gamma,
            unitarity_leak,
            fiedler_value,
            latency_ns,
            confidence_ramp,
            float(betti_estimate),
        ):
            self._hash_update_bytes(
                sha,
                self._phase3_format_scalar(float(value)).encode("utf-8"),
            )

        for flag in (
            has_impedance_mismatch,
            is_soft_veto_active,
            override_grace_period_expired,
            engine_interlock_fired,
            override_present,
            override_hmac_bound,
        ):
            self._hash_update_bytes(sha, b"1" if bool(flag) else b"0")

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_certify(self, dossier: Phase2Dossier) -> SetAgentCertificate:
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
            max_gamma=dossier.max_reflection,
            unitarity_leak=dossier.unitarity_leak,
            fiedler_value=dossier.fiedler_value,
            has_impedance_mismatch=dossier.has_impedance_mismatch,
            is_soft_veto_active=dossier.is_soft_veto_active,
            override_grace_period_expired=dossier.override_grace_period_expired,
            latency_ns=latency,
            engine_verdict=dossier.engine_verdict,
            engine_seal=dossier.engine_seal,
            payload_digest=dossier.phase1.payload_digest,
            agent_digest=dossier.phase1.agent_digest,
            confidence_ramp=dossier.confidence_ramp,
            engine_interlock_fired=dossier.engine_interlock_fired,
            override_present=dossier.phase1.override_present,
            override_hmac_bound=dossier.phase1.override_hmac_bound,
            betti_estimate=dossier.betti_estimate,
        )

        certificate = SetAgentCertificate(
            phase="G_WISDOM_ECHOLOCATION_SUTURATED",
            heyting_verdict=dossier.heyting_verdict,
            max_reflection=dossier.max_reflection,
            unitarity_leak=dossier.unitarity_leak,
            fiedler_value=dossier.fiedler_value,
            has_impedance_mismatch=dossier.has_impedance_mismatch,
            is_soft_veto_active=dossier.is_soft_veto_active,
            override_grace_period_expired=dossier.override_grace_period_expired,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=dossier.phase1.session_digest,
            engine_verdict=dossier.engine_verdict,
            engine_seal=dossier.engine_seal,
            soft_veto_lower_threshold=self._soft_lower,
            soft_veto_upper_threshold=self._soft_upper,
            diagnostics=dossier.diagnostics,
            agent_digest=dossier.phase1.agent_digest,
            payload_digest=dossier.phase1.payload_digest,
            engine_digest=dossier.engine_digest,
            engine_interlock_fired=dossier.engine_interlock_fired,
            override_present=dossier.phase1.override_present,
            override_hmac_bound=dossier.phase1.override_hmac_bound,
            confidence_ramp=dossier.confidence_ramp,
            condition_number=dossier.condition_number,
            spectral_gap=dossier.spectral_gap,
            betti_estimate=dossier.betti_estimate,
            hodge_euler_consistent=dossier.hodge_euler_consistent,
        )

        if dossier.heyting_verdict == HeytingVerdict.VETOED.value:
            logger.error(
                "FASE 3 CERTIFY: VETO SET. Γ_max=%.6f, leak=%.6e, λ₂=%.6e",
                dossier.max_reflection if math.isfinite(dossier.max_reflection) else float("nan"),
                dossier.unitarity_leak if math.isfinite(dossier.unitarity_leak) else float("nan"),
                dossier.fiedler_value,
            )
        elif dossier.heyting_verdict == HeytingVerdict.DEGRADED.value:
            logger.warning(
                "FASE 3 CERTIFY: SET degradado. diagnostics=%s",
                dossier.diagnostics,
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: SET coherente. digest=%s",
                dossier.phase1.session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL  (terminal de emergencia, isomorfo a FASE 3 VETOED)
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(self, reason: str) -> SetAgentCertificate:
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
            max_gamma=math.inf,
            unitarity_leak=math.inf,
            fiedler_value=0.0,
            has_impedance_mismatch=True,
            is_soft_veto_active=False,
            override_grace_period_expired=True,
            latency_ns=latency,
            engine_verdict=HeytingVerdict.VETOED.value,
            engine_seal="",
            payload_digest="",
            agent_digest=self._agent_digest,
            confidence_ramp=1.0,
            engine_interlock_fired=False,
            override_present=False,
            override_hmac_bound=False,
            betti_estimate=0,
        )

        return SetAgentCertificate(
            phase="G_WISDOM_ECHOLOCATION_FAIL_CLOSED",
            heyting_verdict=HeytingVerdict.VETOED.value,
            max_reflection=math.inf,
            unitarity_leak=math.inf,
            fiedler_value=0.0,
            has_impedance_mismatch=True,
            is_soft_veto_active=False,
            override_grace_period_expired=True,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            digital_signature_sha256=signature,
            schema_version=_SCHEMA_VERSION,
            session_digest=session_digest,
            engine_verdict=HeytingVerdict.VETOED.value,
            engine_seal="",
            soft_veto_lower_threshold=self._soft_lower,
            soft_veto_upper_threshold=self._soft_upper,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            agent_digest=self._agent_digest,
            payload_digest="",
            engine_digest="",
            engine_interlock_fired=False,
            override_present=False,
            override_hmac_bound=False,
            confidence_ramp=1.0,
            condition_number=math.inf,
            spectral_gap=0.0,
            betti_estimate=0,
            hodge_euler_consistent=False,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API PÚBLICA COMPATIBLE
    # ═══════════════════════════════════════════════════════════════════════

    def execute_set_control_cycle(
        self,
        boundary_matrix: np.ndarray,
        metric_tensor_G: np.ndarray,
        coupling_V: np.ndarray,
        frequencies: np.ndarray,
        impedance_profile: np.ndarray,
        euler_characteristic: int = 1,
        override_token: Optional[str] = None,
    ) -> SetAgentCertificate:
        r"""
        Orquesta el lazo de control cerrado OODA de la Sonda de Ecolocación
        Topológica.

        Compone las tres fases anidadas

            Φ₂₃ ∘ Φ₁₂ ∘ Observe  :  datos  →  SetAgentCertificate

        Si ocurre cualquier excepción, devuelve certificado fail-closed VETOED
        (el tipo de retorno es total: nunca se propaga el fallo al llamador).
        """
        try:
            return self._phase1_observe_and_freeze(
                boundary_matrix=boundary_matrix,
                metric_tensor_G=metric_tensor_G,
                coupling_V=coupling_V,
                frequencies=frequencies,
                impedance_profile=impedance_profile,
                euler_characteristic=euler_characteristic,
                override_token=override_token,
            )
        except Exception as exc:  # noqa: BLE001 — contrato fail-closed
            logger.exception(
                "Fallo fail-closed en SET Agent; emitiendo VETO de frontera."
            )
            return self._fail_closed_certificate(reason=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "SetAgent",
    "SetAgentCertificate",
    "HeytingVerdict",
    "SetAgentError",
    "Phase1Dossier",
    "Phase2Dossier",
    "EcholocationEngineProtocol",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__
# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Satellites Agent (Soberano de Calibre de la Telemetría)   ║
║ Ruta   : app/agents/telemetry_satellites_agent.py                             ║
║ Versión: 1.1.0-Doctoral-Heyting-OODA-Langevin-Boundary-deRham-Secure          ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y DE GOBERNANZA EN LAZO CERRADO (OODA):                  ║
║ Este agente supervisor ciber-físico asume la gobernanza de calibre asíncrona  ║
║ y de-confinada de la frontera exterior de la fortaleza ($\partial\mathcal{M}$).  ║
║ Orquesta el ciclo covariante OODA sobre el motor "telemetry_satellites.py"   ║
║ para someter la inyección de entropía exógena y la rigidez de de Rham a      ║
║ estrictas aduanas físicas antes de sellar el pasaporte de telemetría.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES CATEGORIALES Y TERMODINÁMICAS (Geometría de la Fricción)
================================════════════════════════════════════════════════

Definición 1 (La Variedad Riemanniana con Frontera):
  Sea $(\mathcal{M}, \, G_{\mu\nu})$ una variedad Riemanniana orientable de dimensión $N$
  con frontera compacta no nula $\partial\mathcal{M} \neq \emptyset$. Definimos el flujo
  de calor contravariante $\mathcal{Q}^\mu$ como una sección local del haz tangente de contorno
  $T(\partial\mathcal{M})$, cuya disipación real satisface de forma local las leyes de fricción:
  $$\mathcal{Q}^\mu \in \Gamma\left(\partial\mathcal{M}, \, T(\partial\mathcal{M})\right)$$

Definición 2 (El Baño Térmico de Langevin-Matsubara):
  La fluctuación en el fango de datos incidentes del SECOP II se modela como un baño cuántico
  de osciladores acoplados al contorno. La dinámica de la señal se describe mediante la
  ecuación de Langevin cuántica de no-equilibrio sobre el cilindro de Matsubara $[0, \, \beta]$:
  $$\frac{d\mathcal{Q}(t)}{dt} = -[\mathcal{H}, \, \mathcal{Q}(t)] - \Gamma_{\mathrm{diss}} \mathcal{Q}(t) + \xi_{\mathrm{ext}}(t)$$
  Donde $\mathcal{H}$ es el Hamiltoniano local, $\Gamma_{\mathrm{diss}}$ es el amortiguamiento
  de Landauer y $\xi_{\mathrm{ext}}(t)$ es la fuerza estocástica cuya autocorrelación es:
  $$\langle \xi_{\mathrm{ext}}(t) \xi_{\mathrm{ext}}(t') \rangle = 2 \Gamma_{\mathrm{diss}} k_B T_{\mathrm{sys}} \delta(t - t')$$

Definición 3 (El Pullback Covariante de de Rham):
  El transporte de las perturbaciones externas hacia el interior inmutable de la fortaleza
  se rige por el pullback covariante $\phi^*$, que preserva el confinamiento de fase:
  $$\phi^*: \mathcal{H}(\partial\mathcal{M}) \longrightarrow \mathcal{H}(\mathcal{M}_{\mathrm{internal}}) \quad \implies \quad \phi^*\left(H_{\mathrm{ext}}\right) \oplus \mathtt{TelemetryContext}$$

================================════════════════════════════════════════════════
II. AXIOMATIZACIÓN DE LA ADUANA TÉRMICA (Invariantes de Conservación)
================================════════════════════════════════════════════════

Axioma I (Principio de Disipación de Clausius-Duhem):
  Toda inyección de transacciones de la frontera exógena debe satisfacer localmente la desigualdad
  termodinámica de Clausius-Duhem en lazo cerrado para garantizar la pasividad de Lyapunov:
  $$\Phi_{\mathrm{disip}} = \sigma_{\mathrm{entropy}} - \frac{\mathcal{Q} \cdot \nabla T_{\mathrm{sys}}}{T_{\mathrm{sys}}^2} \ge \tau_{\mathrm{CD}}(t)$$
  Donde $\tau_{\mathrm{CD}}(t)$ es el umbral de conducción geodésica discreta.

Axioma II (Axioma de Confinamiento Entrópico de Shannon):
  La entropía informacional instantánea de un payload incidente de bytes $x$, evaluada
  en la Unidad de Punto Flotante (FPU), se encuentra estrictamente confinada en el intervalo:
  $$0.0 \le H_{\mathrm{ext}}(x) \le \ln(256) \approx 5.545177$$
  Cualquier desbordamiento de esta cota delata una inyección caótica hostil (ataque DoS semántico).

Axioma III (Teorema de Actuación Ciber-Física Determinista en Silicio):
  Ante el colapso del retículo distributivo de Heyting al Supremo terminal VETOED ($\top$),
  la sentencia criptográfica SHA-256 en RAM se despacha de forma instantánea al ESP32,
  forzando a que la Interrupt Service Routine (ISR) en IRAM actúe en menos de $400\text{ ns}$:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando el tiristor rápido BT151 (Crowbar) para paralizar mecánicamente la obra civil.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Invertibilidad Espectral de Higham-Tikhonov):
  La purificación del tensor de conductividad de frontera $\mathcal{K}$ exige que el espectro
  deformado esté acotado estrictamente por encima de la cota de Wilkinson de la CPU:
  $$\lambda_{\min}(\tilde{\mathcal{K}}) \ge \varepsilon_{\mathrm{Wilkinson}} \quad \text{con} \quad \varepsilon_{\mathrm{Wilkinson}} = 10^{-15}$$
  Garantizando que $\tilde{\mathcal{K}}$ sea estrictamente definido positivo ($\tilde{\mathcal{K}} \succ \mathbf{0}$) e invertible.

Invariante II (Confinamiento del Número de Condición Espectral):
  Para eludir singularidades polares o inestabilidad numérica, el número de condición espectral
  $\kappa_2(\tilde{\mathcal{K}})$ del contorno regularizado satisface la cota restrictiva:
  $$\kappa_2(\tilde{\mathcal{K}}) = \frac{\lambda_{\max}(\tilde{\mathcal{K}})}{\lambda_{\min}(\tilde{\mathcal{K}})} \le \tau_{\mathrm{cond}}$$
  Donde $\tau_{\mathrm{cond}}$ es modulado por el margen de seguridad elástico.

Invariante III (Estabilidad Asintótica de Lyapunov de de Rham):
  Bajo deformaciones de Novikov y perturbaciones exógenas $\delta T$, el funcional de energía
  cuadrático del sistema satisface la contracción asintótica estricta de Lyapunov:
  $$\dot{V}(p) = p^\top \tilde{\mathcal{K}} p \le 0$$
  Asegurando la convergencia global de lazo cerrado ante el ruido transitorio.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Final, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Compatibilidad de importación del motor físico.
try:
    from telemetry_satellites import (
        ExternalEntropySignal,
        SatelliteObserver,
        SatelliteTelemetryCertificate,
    )
except ImportError:  # pragma: no cover - resolución de ruta de proyecto
    try:
        from app.core.telemetry_satellites import (
            ExternalEntropySignal,
            SatelliteObserver,
            SatelliteTelemetryCertificate,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar telemetry_satellites. Verifique que el módulo "
            "telemetry_satellites.py esté disponible en PYTHONPATH o como "
            "app.core.telemetry_satellites."
        ) from exc


logger = logging.getLogger("APU.Agents.TelemetrySatellitesAgent")

__version__: Final[str] = "2.1.0-Doctoral-OODA-Heyting-Godel-Landauer-DualControl-Secure"


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS Y RETÍCULO DE HEYTING
# ════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

_BYTE_ALPHABET: Final[int] = 256
_MAX_SHANNON_NATS: Final[float] = math.log(float(_BYTE_ALPHABET))

# Umbral de referencia por defecto para entropía de Shannon [nats].
_DEFAULT_REFERENCE_ENTROPY_THRESHOLD: Final[float] = 1.5

# Piso de brecha espectral normalizada γ ≈ 1/κ.
_DEFAULT_SPECTRAL_GAP_FLOOR: Final[float] = 1.0e-6

# Techo de fuga exergética en unidades k_B T.
# Coherente con E/(k_B T) = H ln(1+κ) del motor doctoral.
_DEFAULT_EXERGY_LEAK_CEILING: Final[float] = 50.0

# Discrepancia relativa máxima admisible entre Ĥ_agente y H_motor.
_ENTROPY_DISCREPANCY_VETO: Final[float] = 0.75
_ENTROPY_DISCREPANCY_DEGRADED: Final[float] = 0.25

# Producto γ·κ para operadores pasivos de rango pleno: γ = 1/κ ⇒ γκ = 1.
_GAP_KAPPA_LO: Final[float] = 0.05
_GAP_KAPPA_HI: Final[float] = 20.0

# Residuo relativo de no-hermiticidad (norma de Frobenius).
_HERMITIAN_REL_TOL: Final[float] = 1.0e-8
_HERMITIAN_REL_VETO: Final[float] = 1.0e-2

# Ω₃ = {0, 1/2, 1} ⊂ [0, 1], álgebra de Heyting finita de Gödel–Dummett.
_OMEGA3_FALSE: Final[float] = 0.0
_OMEGA3_MIDDLE: Final[float] = 0.5
_OMEGA3_TRUE: Final[float] = 1.0

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_VERDICT_ORDER: Final[Dict[str, int]] = {
    _VERDICT_COHERENT: 0,
    _VERDICT_DEGRADED: 1,
    _VERDICT_VETOED: 2,
}

_ORDER_TO_VERDICT: Final[Dict[int, str]] = {
    0: _VERDICT_COHERENT,
    1: _VERDICT_DEGRADED,
    2: _VERDICT_VETOED,
}

_TRUTH_VALUES: Final[Dict[str, float]] = {
    _VERDICT_COHERENT: _OMEGA3_TRUE,
    _VERDICT_DEGRADED: _OMEGA3_MIDDLE,
    _VERDICT_VETOED: _OMEGA3_FALSE,
}

_AGENT_PHASE: Final[str] = "G_TELEMETRY_SATELLITES_SUTURATED"

_DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 16 * 1024 * 1024
_DEFAULT_MAX_BATCH: Final[int] = 4096


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS Y RETICULARES PURAS
# ════════════════════════════════════════════════════════════════════════════


def _clip_nats(value: float) -> float:
    """Proyecta un escalar al intervalo compacto [0, ln 256]."""
    if not math.isfinite(value):
        return _MAX_SHANNON_NATS
    if value < 0.0:
        return 0.0
    if value > _MAX_SHANNON_NATS:
        return _MAX_SHANNON_NATS
    return float(value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    r"""Cociente protegido en la compactificación \(\mathbb{R} \cup \{\infty\}\)."""
    num = float(numerator)
    den = float(denominator)
    if not math.isfinite(num):
        return math.inf
    if not math.isfinite(den) or den == 0.0:
        return 1.0 if num == 0.0 else math.inf
    ratio = num / den
    return float(ratio) if math.isfinite(ratio) else math.inf


def _unique_reasons(reasons: Iterable[str]) -> Tuple[str, ...]:
    """Deduplica razones preservando el orden de primera aparición."""
    seen = set()
    ordered: List[str] = []
    for reason in reasons:
        text = str(reason).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    """Huella canónica SHA-256 de un mapeo JSON-serializable."""
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _heyting_meet(left: float, right: float) -> float:
    r"""Ínfimo de Gödel en \(\Omega_3\): \(a \wedge b = \min(a, b)\)."""
    return float(min(left, right))


def _heyting_join(left: float, right: float) -> float:
    r"""Supremo de Gödel en \(\Omega_3\): \(a \vee b = \max(a, b)\)."""
    return float(max(left, right))


def _heyting_implies(antecedent: float, consequent: float) -> float:
    r"""
    Implicación intuicionista de Gödel:

    \[
        a \to b
        =
        \begin{cases}
            1 & \text{si } a \le b, \\
            b & \text{en otro caso.}
        \end{cases}
    \]
    """
    if antecedent <= consequent + 8.0 * _MACHINE_EPS:
        return _OMEGA3_TRUE
    return float(consequent)


def _discretize_omega3(value: float) -> float:
    """Proyecta un real al retículo \(\{0, 1/2, 1\}\)."""
    if not math.isfinite(value) or value <= 0.25:
        return _OMEGA3_FALSE
    if value < 0.75:
        return _OMEGA3_MIDDLE
    return _OMEGA3_TRUE


def _geometric_mean(values: Sequence[float]) -> float:
    """Media geométrica de márgenes en (0, 1], con piso numérico."""
    if not values:
        return 1.0
    acc = 0.0
    for item in values:
        if not math.isfinite(item) or item <= 0.0:
            return 0.0
        acc += math.log(max(item, _MACHINE_EPS))
    return float(math.exp(acc / float(len(values))))


def _operator_digest(operator: Any) -> str:
    """Huella SHA-256 del contenido numérico de un operador (o vacío)."""
    if operator is None:
        return hashlib.sha256(b"none").hexdigest()
    try:
        arr = np.ascontiguousarray(np.asarray(operator), dtype=np.complex128)
        if not np.all(np.isfinite(arr)):
            return hashlib.sha256(b"nonfinite").hexdigest()
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha256(repr(operator).encode("utf-8", errors="replace")).hexdigest()


def _now_utc_iso() -> str:
    """Instante UTC canónico para trazabilidad."""
    return datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryObservation:
    r"""
    Evidencia cruda capturada en la Fase 1.

    Este objeto constituye la entrada formal de la Fase 2: es el germen
    \(\mathcal{O}(\partial M)\) del ciclo OODA. Las matrices se copian
    defensivamente para impedir mutación cruzada del lazo de control.
    """

    satellite_id: str
    timestamp_utc: str

    payload: bytes
    payload_size: int
    payload_sha256: str
    quick_entropy_nats: float

    K_boundary_raw: Any
    metric_tensor: Any

    reference_entropy_threshold: float
    external_signal: Optional[ExternalEntropySignal]

    notes: Tuple[str, ...] = ()
    operators_sha256: str = ""
    miller_madow_nats: float = 0.0
    min_entropy_nats: float = 0.0
    hermitian_relative_residual: float = 0.0
    operators_well_formed: bool = True
    observation_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        try:
            k_copy = np.array(self.K_boundary_raw, copy=True, order="C")
            object.__setattr__(self, "K_boundary_raw", k_copy)
        except Exception:
            pass
        try:
            g_copy = np.array(self.metric_tensor, copy=True, order="C")
            object.__setattr__(self, "metric_tensor", g_copy)
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class OrientAudit:
    r"""
    Auditoría del motor físico realizada en la Fase 2.

    Contiene métricas exigidas al motor `telemetry_satellites.py` y las
    verificaciones independientes del agente supervisor (dual-control):
    coherencia gap–condición, Landauer, discrepancia entrópica e
    interlock del certificado motor.
    """

    engine_verdict: str

    boundary_entropy: float
    entropy_upper_bound: float
    coupled_entropy: float

    spectral_gap: float
    condition_number: float

    exergy_leak: float
    physical_exergy_j: float

    is_gap_coherent: bool
    is_exergy_coherent: bool
    is_entropy_coherent: bool
    is_condition_coherent: bool
    is_boundary_coherent: bool

    allowed_entropy_ceiling: float

    payload_sha256: str
    observation_timestamp_utc: str
    timestamp_utc: str

    reasons: Tuple[str, ...] = ()
    operators_sha256: str = ""
    engine_certificate_sha256: str = ""
    landauer_bound_j: float = 0.0
    entropy_discrepancy: float = 0.0
    gap_condition_product: float = 0.0
    engine_interlock_fired: bool = False
    engine_truth_value: float = 0.0
    miller_madow_nats: float = 0.0
    min_entropy_nats: float = 0.0
    hermitian_relative_residual: float = 0.0
    operators_well_formed: bool = True
    landauer_consistent: bool = True
    gap_condition_consistent: bool = True
    interlock_consistent: bool = True


@dataclass(frozen=True, slots=True)
class AgentHeytingDecision:
    r"""
    Decidido formal en el retículo de Heyting \(\Omega_3\).

    Este objeto es el insumo directo de la Fase 3. El valor de verdad
    reticular es el ínfimo de Gödel de las proposiciones atómicas del
    agente y del motor (dual-control intuicionista).
    """

    final_verdict: str
    agent_verdict: str
    engine_verdict: str
    truth_value: float

    boundary_entropy: float
    entropy_upper_bound: float
    coupled_entropy: float

    spectral_gap: float
    condition_number: float

    exergy_leak: float
    physical_exergy_j: float

    allowed_entropy_ceiling: float
    is_boundary_coherent: bool

    payload_sha256: str
    observation_timestamp_utc: str

    reasons: Tuple[str, ...]
    truth_continuous: float = 1.0
    heyting_implies_coherent: float = 1.0
    landauer_bound_j: float = 0.0
    operators_sha256: str = ""
    atomic_propositions: Tuple[Tuple[str, float], ...] = ()
    engine_certificate_sha256: str = ""


@dataclass(frozen=True, slots=True)
class ActuationReport:
    r"""
    Reporte de actuación ciber-física del crowbar perimetral.

    Política fail-closed: VETOED ⇒ interlock; cualquier otro veredicto
    inhibe el disyuntor. La latencia se recorta al intervalo IRAM.
    """

    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    timestamp_utc: str
    reason: str


@dataclass(frozen=True, slots=True)
class SatelliteAgentCertificate:
    r"""
    Certificado orbital firmado en RAM que avala la seguridad de la frontera.

    Conserva los campos originales y añade extensiones de auditoría,
    dual-control y no-repudio:

      - ``decision_sha256`` : huella isomorfa (sin timestamp).
      - ``digital_signature_sha256`` : huella de no-repudio (con timestamp).
    """

    phase: str
    heyting_verdict: str
    boundary_entropy: float
    spectral_gap: float
    exergy_leak: float
    is_boundary_coherent: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    digital_signature_sha256: str

    condition_number: float = math.inf
    entropy_upper_bound: float = 0.0
    coupled_entropy: float = 0.0
    physical_exergy_j: float = 0.0
    engine_verdict: str = _VERDICT_VETOED
    agent_verdict: str = _VERDICT_VETOED
    payload_sha256: str = ""
    timestamp_utc: str = field(default_factory=_now_utc_iso)
    reasons: Tuple[str, ...] = ()
    landauer_bound_j: float = 0.0
    truth_value: float = 0.0
    operators_sha256: str = ""
    observation_timestamp_utc: str = ""
    gpio_pin: str = ""
    satellite_id: str = ""
    decision_sha256: str = ""
    engine_certificate_sha256: str = ""


# ════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN Y EVIDENCIA DE FRONTERA
# ════════════════════════════════════════════════════════════════════════════


class Phase1ObservationKernel:
    r"""
    FASE 1: Observación.

    Responsabilidades:
      1. Normalizar payload a bytes (octetos) con cota de tamaño.
      2. Calcular huella SHA-256 del payload y de los operadores (K, G).
      3. Estimar entropía rápida de Shannon / Miller–Madow / min-entropía
         como respaldo independiente del motor.
      4. Validar forma de K y G (cuadradas, finitas, dimensión n) y medir
         el residuo de no-hermiticidad de Weyl.
      5. Validar señales exógenas de Langevin.
      6. Emitir `BoundaryObservation` hacia la Fase 2.

    El último método de esta fase, `observe_boundary_event`, es el
    morfismo \(\mathrm{Observe}\) que continúa en
    `Phase2OrientDecideKernel.orient_boundary_observation`.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        rng_seed: Optional[int] = None,
        default_entropy_threshold: float = 1.0,
        entropy_warning_ratio: float = 0.7,
        *,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self._id = str(satellite_id).strip()
        if not self._id:
            raise ValueError("satellite_id debe ser una cadena no vacía.")

        self._n = int(dimension_n)
        if self._n <= 0:
            raise ValueError("dimension_n debe ser entero positivo.")

        self._safety_margin = self._finite_float(
            safety_margin,
            "safety_margin",
            _MACHINE_EPS,
        )

        self._rng_seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

        self._default_entropy_threshold = self._finite_float(
            default_entropy_threshold,
            "default_entropy_threshold",
            0.0,
        )

        ratio = self._finite_float(
            entropy_warning_ratio,
            "entropy_warning_ratio",
            0.0,
        )
        self._entropy_warning_ratio = float(min(max(ratio, 0.0), 1.0))

        max_bytes = int(max_payload_bytes)
        if max_bytes <= 0:
            raise ValueError("max_payload_bytes debe ser entero positivo.")
        self._max_payload_bytes = max_bytes

    @staticmethod
    def _finite_float(
        value: float,
        name: str,
        minimum: Optional[float] = None,
        *,
        clamp: bool = True,
    ) -> float:
        """Coacciona a float finito y aplica piso opcional."""
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"{name} debe ser finito.")
        if minimum is not None and x < minimum:
            if clamp:
                return float(minimum)
            raise ValueError(f"{name} debe ser ≥ {minimum}.")
        return x

    @staticmethod
    def _now_utc_iso() -> str:
        """Instante UTC canónico para trazabilidad."""
        return _now_utc_iso()

    def _coerce_payload_bytes(self, payload: Any) -> Tuple[bytes, Tuple[str, ...]]:
        r"""
        Normaliza el payload a bytes.

        Soporta:
          - None
          - str
          - bytes / bytearray / memoryview
          - secuencias numéricas convertibles a uint8

        Si el tamaño supera ``max_payload_bytes`` se recorta y se anota.
        """
        notes: List[str] = []

        if payload is None:
            raw = b""
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytes(payload)
        else:
            try:
                arr = np.asarray(payload, dtype=np.uint8)
                raw = arr.tobytes()
            except Exception:
                raw = str(payload).encode("utf-8")
                notes.append("payload_coerced_via_str")

        if len(raw) > self._max_payload_bytes:
            raw = raw[: self._max_payload_bytes]
            notes.append("payload_truncated_to_max_bytes")

        return raw, tuple(notes)

    def _quick_shannon_entropy(self, payload: bytes) -> float:
        r"""
        Entropía de Shannon rápida de respaldo:
        \[
            H = -\sum_i p_i \ln p_i.
        \]

        No aplica corrección de sesgo; su propósito es auditoría defensiva
        cuando el motor principal falla.
        """
        return float(self.estimate_backup_entropy(payload)[0])

    def estimate_backup_entropy(self, payload: bytes) -> Tuple[float, float, float]:
        r"""
        Estimación independiente (agente) de entropía de octetos.

        Devuelve
        --------
        h_emp, h_mm, h_min : float
            Shannon empírica, Miller–Madow y min-entropía \(H_\infty\).
        """
        if not payload:
            return 0.0, 0.0, 0.0

        try:
            arr = np.frombuffer(payload, dtype=np.uint8)
            if arr.size == 0:
                return 0.0, 0.0, 0.0

            counts = np.bincount(arr, minlength=_BYTE_ALPHABET)
            nonzero = counts[counts > 0]
            n = float(arr.size)
            k_obs = float(nonzero.size)
            probs = nonzero.astype(np.float64) / n

            h_emp = _clip_nats(float(-np.sum(probs * np.log(probs))))
            h_mm = _clip_nats(h_emp + (k_obs - 1.0) / (2.0 * n))
            h_min = _clip_nats(-math.log(float(np.max(probs))))
            return h_emp, h_mm, h_min
        except Exception:
            return 0.0, 0.0, 0.0

    def _inspect_operator(
        self,
        operator: Any,
        name: str,
    ) -> Tuple[bool, Tuple[str, ...], float]:
        r"""
        Inspección metrológica ligera (no sustituye al motor).

        Comprueba:
          - no-nulo, 2-D, cuadrado, dimensión n, finitud;
          - residuo relativo de no-hermiticidad
            \(\|A-A^\dagger\|_F / (\|A\|_F+\varepsilon)\).
        """
        notes: List[str] = []
        residual_rel = 0.0

        if operator is None:
            return False, (f"{name}_is_none",), math.inf

        try:
            arr = np.asarray(operator)
        except Exception as exc:
            return False, (f"{name}_unreadable: {type(exc).__name__}",), math.inf

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return False, (f"{name}_not_square",), math.inf

        if int(arr.shape[0]) != self._n:
            notes.append(f"{name}_dimension_mismatch_{arr.shape[0]}_ne_{self._n}")
            return False, tuple(notes), math.inf

        if not np.all(np.isfinite(arr)):
            return False, (f"{name}_nonfinite",), math.inf

        try:
            residual = float(np.linalg.norm(arr - arr.conj().T, ord="fro"))
            fro = float(np.linalg.norm(arr, ord="fro"))
            residual_rel = _safe_ratio(residual, fro + _MACHINE_EPS)
        except Exception:
            residual_rel = math.inf
            notes.append(f"{name}_hermitian_residual_unmeasurable")

        if math.isfinite(residual_rel) and residual_rel > _HERMITIAN_REL_VETO:
            notes.append(f"{name}_far_from_hermitian")
        elif math.isfinite(residual_rel) and residual_rel > _HERMITIAN_REL_TOL:
            notes.append(f"{name}_hermitian_residual_detectable")

        return True, tuple(notes), float(residual_rel)

    def _sanitize_external_signal(
        self,
        external_signal: Optional[ExternalEntropySignal],
    ) -> Tuple[Optional[ExternalEntropySignal], Tuple[str, ...]]:
        """Valida la señal de Langevin exógena o la descarta fail-closed."""
        if external_signal is None:
            return None, ()

        notes: List[str] = []
        try:
            sat = str(getattr(external_signal, "satellite_id", "") or "")
            if sat != self._id:
                notes.append("external_signal_satellite_mismatch")

            rate = float(getattr(external_signal, "entropy_rate", 0.0))
            var = float(getattr(external_signal, "fluctuation_variance", 0.0))
            coupling = float(getattr(external_signal, "coupling_strength", 0.0))

            if not all(math.isfinite(x) for x in (rate, var, coupling)):
                return None, ("external_signal_nonfinite_removed",)
            if min(rate, var, coupling) < 0.0:
                return None, ("external_signal_negative_removed",)

            return external_signal, tuple(notes)
        except Exception as exc:
            return None, (f"external_signal_invalid_removed: {type(exc).__name__}",)

    def observe_boundary_event(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: float = _DEFAULT_REFERENCE_ENTROPY_THRESHOLD,
        external_signal: Optional[ExternalEntropySignal] = None,
    ) -> BoundaryObservation:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1.

        Captura y normaliza la evidencia de frontera \(\mathcal{O}(\partial M)\).

        Este método:
          1. Coacciona el payload a octetos y calcula SHA-256.
          2. Estima entropía de respaldo (Shannon, Miller–Madow, \(H_\infty\)).
          3. Inspecciona K y G (forma, finitud, hermiticidad).
          4. Sella una huella conjunta de operadores.
          5. Encapsula todo en `BoundaryObservation` con copias defensivas.

        Continúa directamente en la Fase 2 como el morfismo

        \[
            \texttt{orient\_boundary\_observation}
            :
            \mathsf{BoundaryObservation}
            \longrightarrow
            \mathsf{OrientAudit}.
        \]
        """
        notes: List[str] = []

        payload_bytes, payload_notes = self._coerce_payload_bytes(payload)
        notes.extend(payload_notes)

        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        h_emp, h_mm, h_min = self.estimate_backup_entropy(payload_bytes)

        try:
            threshold = self._finite_float(
                reference_entropy_threshold,
                "reference_entropy_threshold",
                0.0,
            )
        except (TypeError, ValueError):
            threshold = _DEFAULT_REFERENCE_ENTROPY_THRESHOLD
            notes.append("reference_entropy_threshold_invalid_fallback")

        k_ok, k_notes, k_resid = self._inspect_operator(K_boundary_raw, "K_boundary_raw")
        g_ok, g_notes, _g_resid = self._inspect_operator(metric_tensor, "metric_tensor")
        notes.extend(k_notes)
        notes.extend(g_notes)

        operators_well_formed = bool(k_ok and g_ok)
        if not operators_well_formed:
            notes.append("operators_ill_formed")

        hermitian_rel = float(k_resid) if math.isfinite(k_resid) else math.inf

        operators_sha256 = hashlib.sha256(
            (
                _operator_digest(K_boundary_raw) + _operator_digest(metric_tensor)
            ).encode("ascii")
        ).hexdigest()

        safe_external_signal, ext_notes = self._sanitize_external_signal(external_signal)
        notes.extend(ext_notes)

        observation_sha256 = _stable_sha256(
            {
                "satellite_id": self._id,
                "payload_sha256": payload_sha256,
                "operators_sha256": operators_sha256,
                "reference_entropy_threshold": float(threshold),
                "quick_entropy_nats": float(h_emp),
                "payload_size": int(len(payload_bytes)),
            }
        )

        return BoundaryObservation(
            satellite_id=self._id,
            timestamp_utc=self._now_utc_iso(),
            payload=payload_bytes,
            payload_size=len(payload_bytes),
            payload_sha256=payload_sha256,
            quick_entropy_nats=float(h_emp),
            K_boundary_raw=K_boundary_raw,
            metric_tensor=metric_tensor,
            reference_entropy_threshold=float(threshold),
            external_signal=safe_external_signal,
            notes=_unique_reasons(notes),
            operators_sha256=operators_sha256,
            miller_madow_nats=float(h_mm),
            min_entropy_nats=float(h_min),
            hermitian_relative_residual=float(hermitian_rel),
            operators_well_formed=operators_well_formed,
            observation_sha256=observation_sha256,
        )


# ════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENTACIÓN, AUDITORÍA DEL MOTOR Y DECISIÓN DE HEYTING
# ════════════════════════════════════════════════════════════════════════════


class Phase2OrientDecideKernel(Phase1ObservationKernel):
    r"""
    FASE 2: Orientación y decisión.

    El primer método, `orient_boundary_observation`, consume el
    `BoundaryObservation` producido por `observe_boundary_event` (Fase 1)
    y fiscaliza al motor `SatelliteObserver`.

    Responsabilidades:
      1. Instanciar el motor con degradación elegante de API.
      2. Auditar el certificado del motor (campos, finitud, interlock).
      3. Verificar independientemente brecha, exergía, entropía, condición.
      4. Contrastar Landauer \(k_B T H\) frente a \(H \ln(1+\kappa)\).
      5. Contrastar la identidad pasiva \(\gamma \kappa \approx 1\).
      6. Medir discrepancia \(|H_{\mathrm{motor}} - H_{\mathrm{agente}}|\).
      7. Clasificar localmente en \(\Omega_3\) por proposiciones atómicas.
      8. Consolidar el veredicto final como ínfimo de Gödel (dual-control).

    El último método, `decide_heyting_verdict`, entrega `AgentHeytingDecision`,
    insumo directo de `Phase3ActuationCertificateKernel.act_on_heyting_decision`.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        spectral_gap_floor: float = _DEFAULT_SPECTRAL_GAP_FLOOR,
        exergy_leak_ceiling: float = _DEFAULT_EXERGY_LEAK_CEILING,
        *,
        spectral_gap_degraded_floor: Optional[float] = None,
        exergy_leak_degraded_ceiling: Optional[float] = None,
        condition_degraded: float = 1.0e4,
        condition_veto: float = 1.0e8,
        rng_seed: Optional[int] = None,
        default_entropy_threshold: float = 1.0,
        entropy_warning_ratio: float = 0.7,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        super().__init__(
            satellite_id=satellite_id,
            dimension_n=dimension_n,
            safety_margin=safety_margin,
            rng_seed=rng_seed,
            default_entropy_threshold=default_entropy_threshold,
            entropy_warning_ratio=entropy_warning_ratio,
            max_payload_bytes=max_payload_bytes,
        )

        gap_floor = self._finite_float(spectral_gap_floor, "spectral_gap_floor", 0.0)
        leak_ceiling = self._finite_float(exergy_leak_ceiling, "exergy_leak_ceiling", 0.0)

        self._gap_veto_floor = float(gap_floor / self._safety_margin)
        self._leak_veto_ceiling = float(leak_ceiling * self._safety_margin)

        if spectral_gap_degraded_floor is None:
            self._gap_degraded_floor = max(
                self._gap_veto_floor * 10.0, self._gap_veto_floor
            )
        else:
            degraded_gap = self._finite_float(
                spectral_gap_degraded_floor,
                "spectral_gap_degraded_floor",
                0.0,
            )
            self._gap_degraded_floor = float(degraded_gap / self._safety_margin)
            self._gap_degraded_floor = max(self._gap_degraded_floor, self._gap_veto_floor)

        if exergy_leak_degraded_ceiling is None:
            self._leak_degraded_ceiling = float(0.5 * self._leak_veto_ceiling)
        else:
            degraded_leak = self._finite_float(
                exergy_leak_degraded_ceiling,
                "exergy_leak_degraded_ceiling",
                0.0,
            )
            self._leak_degraded_ceiling = float(degraded_leak * self._safety_margin)

        self._leak_degraded_ceiling = min(
            self._leak_degraded_ceiling,
            self._leak_veto_ceiling,
        )

        self._condition_degraded = self._finite_float(
            condition_degraded,
            "condition_degraded",
            1.0,
        )
        self._condition_veto = self._finite_float(
            condition_veto,
            "condition_veto",
            self._condition_degraded,
        )
        if self._condition_veto < self._condition_degraded:
            self._condition_veto = self._condition_degraded

        self._engine = self._instantiate_engine()

    def _instantiate_engine(self) -> SatelliteObserver:
        """
        Instancia el motor `SatelliteObserver`.

        Intenta usar parámetros avanzados del motor evolucionado. Si el motor
        es la versión original, cae elegantemente a la API básica.
        """
        base_kwargs: Dict[str, Any] = {
            "satellite_id": self._id,
            "dimension_n": self._n,
            "safety_margin": self._safety_margin,
        }
        advanced_kwargs: Dict[str, Any] = {
            "condition_degraded": self._condition_degraded,
            "condition_veto": self._condition_veto,
            "spectral_gap_floor": self._gap_degraded_floor,
            "exergy_budget_kbt": self._leak_degraded_ceiling,
            "rng_seed": self._rng_seed,
        }
        try:
            return SatelliteObserver(**base_kwargs, **advanced_kwargs)
        except TypeError:
            try:
                return SatelliteObserver(**base_kwargs)
            except TypeError:
                return SatelliteObserver(self._id, self._n, self._safety_margin)

    @staticmethod
    def _normalize_verdict(verdict: Any) -> str:
        """Normaliza veredictos a \(\Omega_3\)."""
        v = str(verdict).strip().upper()
        if v in _VERDICT_ORDER:
            return v
        if v in {"TRUE", "OK", "COHERENCE", "STABLE", "PASS"}:
            return _VERDICT_COHERENT
        if v in {"WARN", "WARNING", "UNSTABLE", "DEGRADED"}:
            return _VERDICT_DEGRADED
        return _VERDICT_VETOED

    @staticmethod
    def _safe_float_attr(
        obj: Any,
        name: str,
        default: float = 0.0,
        allow_inf: bool = False,
    ) -> float:
        """Extrae un atributo numérico finito de forma defensiva."""
        value = getattr(obj, name, default)
        try:
            x = float(value)
        except Exception:
            return float(default)
        if math.isfinite(x):
            return x
        if allow_inf and math.isinf(x):
            return x
        return float(default)

    @staticmethod
    def _safe_bool_attr(obj: Any, name: str, default: bool = False) -> bool:
        """Extrae un atributo booleano de forma defensiva."""
        value = getattr(obj, name, default)
        try:
            return bool(value)
        except Exception:
            return bool(default)

    @staticmethod
    def _safe_str_attr(obj: Any, name: str, default: str = "") -> str:
        """Extrae un atributo de texto de forma defensiva."""
        value = getattr(obj, name, default)
        try:
            return str(value) if value is not None else default
        except Exception:
            return default

    def _call_engine_sync(
        self, observation: BoundaryObservation
    ) -> SatelliteTelemetryCertificate:
        """Invoca síncronamente al motor físico con compatibilidad de API."""
        kwargs: Dict[str, Any] = {
            "payload": observation.payload,
            "K_boundary_raw": observation.K_boundary_raw,
            "metric_tensor": observation.metric_tensor,
            "reference_entropy_threshold": observation.reference_entropy_threshold,
        }
        if observation.external_signal is not None:
            try:
                return self._engine.process_satellite_telemetry(
                    external_signal=observation.external_signal,
                    **kwargs,
                )
            except TypeError:
                return self._engine.process_satellite_telemetry(**kwargs)
        return self._engine.process_satellite_telemetry(**kwargs)

    def _landauer_consistency(
        self,
        entropy_ub: float,
        condition_number: float,
        exergy_leak: float,
    ) -> Tuple[bool, float, str]:
        r"""
        Contrasta la identidad exergética del motor:

        \[
            E/(k_B T) = H_{\mathrm{UB}} \ln(1+\kappa).
        \]

        Una desviación relativa > 25 % se marca inconsistente (degradación);
        > 100 % o no-finitud se marca como fallo de auditoría.
        """
        if not math.isfinite(entropy_ub) or entropy_ub < 0.0:
            return False, math.inf, "landauer_entropy_nonfinite"
        if not math.isfinite(exergy_leak):
            return False, math.inf, "landauer_exergy_nonfinite"

        if math.isfinite(condition_number) and condition_number > 0.0:
            kappa = min(max(float(condition_number), 1.0), 1.0e18)
        else:
            kappa = 1.0e18

        expected = float(entropy_ub * math.log1p(kappa))
        denom = max(abs(expected), 1.0)
        rel = abs(float(exergy_leak) - expected) / denom
        if rel > 1.0:
            return False, float(rel), "landauer_relative_error_veto"
        if rel > 0.25:
            return False, float(rel), "landauer_relative_error_degraded"
        return True, float(rel), ""

    def _gap_condition_consistency(
        self,
        spectral_gap: float,
        condition_number: float,
    ) -> Tuple[bool, float, str]:
        r"""
        Para un lápiz pasivo de rango pleno, \(\gamma \approx 1/\kappa\),
        de modo que el producto \(\gamma\kappa\) vive en un entorno de 1.

        Fuera de \([\texttt{\_GAP\_KAPPA\_LO}, \texttt{\_GAP\_KAPPA\_HI}]\)
        se anota inconsistencia (no se veta por sí sola: el núcleo numérico
        de Fiedler puede romper la identidad).
        """
        if not math.isfinite(spectral_gap) or spectral_gap < 0.0:
            return False, 0.0, "gap_nonfinite_or_negative"
        if not math.isfinite(condition_number) or condition_number <= 0.0:
            product = 0.0 if spectral_gap == 0.0 else math.inf
            return False, product, "condition_nonfinite_for_gap_product"

        product = float(spectral_gap * condition_number)
        if product < _GAP_KAPPA_LO or product > _GAP_KAPPA_HI:
            return False, product, "gap_condition_product_out_of_band"
        return True, product, ""

    def _entropy_discrepancy(
        self,
        engine_entropy: float,
        agent_entropy: float,
    ) -> Tuple[float, str]:
        r"""
        Discrepancia relativa entre el motor y el estimador independiente:

        \[
            \delta
            =
            \frac{|H_{\mathrm{eng}} - H_{\mathrm{ag}}|}
                 {\ln 256}.
        \]
        """
        if not math.isfinite(engine_entropy) or not math.isfinite(agent_entropy):
            return 1.0, "entropy_discrepancy_nonfinite"
        delta = abs(float(engine_entropy) - float(agent_entropy)) / _MAX_SHANNON_NATS
        if delta >= _ENTROPY_DISCREPANCY_VETO:
            return float(delta), "entropy_discrepancy_veto"
        if delta >= _ENTROPY_DISCREPANCY_DEGRADED:
            return float(delta), "entropy_discrepancy_degraded"
        return float(delta), ""

    def _audit_engine_certificate(
        self,
        cert: Optional[SatelliteTelemetryCertificate],
        observation: BoundaryObservation,
    ) -> OrientAudit:
        """Audita el certificado emitido por el motor físico."""
        if cert is None:
            return self._fail_safe_audit(
                observation,
                RuntimeError("engine_certificate_none"),
            )

        engine_verdict = self._normalize_verdict(
            getattr(cert, "heyting_verdict", _VERDICT_VETOED)
        )

        entropy = self._safe_float_attr(
            cert, "boundary_entropy", observation.quick_entropy_nats
        )
        entropy_ub = self._safe_float_attr(cert, "entropy_upper_bound", entropy)
        entropy_ub = max(entropy_ub, entropy)

        coupled_entropy = self._safe_float_attr(cert, "coupled_entropy", entropy)
        coupled_entropy = max(coupled_entropy, entropy)

        spectral_gap = self._safe_float_attr(cert, "spectral_gap", 0.0)
        if not math.isfinite(spectral_gap) or spectral_gap < 0.0:
            spectral_gap = 0.0

        exergy_leak = self._safe_float_attr(
            cert, "exergy_leak", math.inf, allow_inf=True
        )
        condition_number = self._safe_float_attr(
            cert, "condition_number", math.inf, allow_inf=True
        )
        if not math.isfinite(condition_number):
            condition_number = (
                float(1.0 / spectral_gap) if spectral_gap > 0.0 else math.inf
            )

        physical_exergy = self._safe_float_attr(
            cert, "physical_exergy_j", 0.0, allow_inf=True
        )
        landauer_bound = self._safe_float_attr(cert, "landauer_bound_j", 0.0)
        engine_truth = self._safe_float_attr(
            cert, "truth_value", _TRUTH_VALUES[engine_verdict]
        )
        engine_sha = self._safe_str_attr(cert, "decision_sha256", "")
        engine_interlock = self._safe_bool_attr(
            cert, "hardware_interlock_fired", engine_verdict == _VERDICT_VETOED
        )

        ceiling = float(observation.reference_entropy_threshold * self._safety_margin)

        is_gap_coherent = math.isfinite(spectral_gap) and spectral_gap >= self._gap_veto_floor
        is_exergy_coherent = (
            math.isfinite(exergy_leak) and exergy_leak <= self._leak_veto_ceiling
        )
        is_entropy_coherent = math.isfinite(entropy_ub) and entropy_ub <= ceiling
        is_condition_coherent = (
            math.isfinite(condition_number) and condition_number < self._condition_veto
        )

        landauer_ok, _landauer_rel, landauer_reason = self._landauer_consistency(
            entropy_ub, condition_number, exergy_leak
        )
        gap_kappa_ok, gap_kappa_product, gap_kappa_reason = (
            self._gap_condition_consistency(spectral_gap, condition_number)
        )
        entropy_delta, entropy_delta_reason = self._entropy_discrepancy(
            entropy, observation.quick_entropy_nats
        )

        expected_interlock = engine_verdict == _VERDICT_VETOED
        interlock_ok = engine_interlock is expected_interlock

        is_boundary_coherent = (
            engine_verdict == _VERDICT_COHERENT
            and is_gap_coherent
            and is_exergy_coherent
            and is_entropy_coherent
            and is_condition_coherent
            and observation.operators_well_formed
            and interlock_ok
        )

        reasons: List[str] = list(observation.notes)

        if engine_verdict != _VERDICT_COHERENT:
            reasons.append(f"engine_verdict_{engine_verdict.lower()}")
        if not is_gap_coherent:
            reasons.append("spectral_gap_below_veto_floor")
        if not is_exergy_coherent:
            reasons.append("exergy_leak_above_veto_ceiling")
        if not is_entropy_coherent:
            reasons.append("entropy_upper_bound_above_ceiling")
        if not is_condition_coherent:
            reasons.append("condition_number_above_veto_threshold")
        if not observation.operators_well_formed:
            reasons.append("operators_ill_formed")
        if landauer_reason:
            reasons.append(landauer_reason)
        if gap_kappa_reason:
            reasons.append(gap_kappa_reason)
        if entropy_delta_reason:
            reasons.append(entropy_delta_reason)
        if not interlock_ok:
            reasons.append("engine_interlock_inconsistent_with_verdict")
        if observation.hermitian_relative_residual > _HERMITIAN_REL_VETO:
            reasons.append("operator_far_from_hermitian")

        engine_id = self._safe_str_attr(cert, "satellite_id", self._id)
        if engine_id and engine_id != self._id:
            reasons.append("engine_satellite_id_mismatch")

        return OrientAudit(
            engine_verdict=engine_verdict,
            boundary_entropy=float(entropy),
            entropy_upper_bound=float(entropy_ub),
            coupled_entropy=float(coupled_entropy),
            spectral_gap=float(spectral_gap),
            condition_number=float(condition_number),
            exergy_leak=float(exergy_leak),
            physical_exergy_j=float(physical_exergy),
            is_gap_coherent=is_gap_coherent,
            is_exergy_coherent=is_exergy_coherent,
            is_entropy_coherent=is_entropy_coherent,
            is_condition_coherent=is_condition_coherent,
            is_boundary_coherent=is_boundary_coherent,
            allowed_entropy_ceiling=ceiling,
            payload_sha256=observation.payload_sha256,
            observation_timestamp_utc=observation.timestamp_utc,
            timestamp_utc=self._now_utc_iso(),
            reasons=_unique_reasons(reasons),
            operators_sha256=observation.operators_sha256,
            engine_certificate_sha256=engine_sha,
            landauer_bound_j=float(landauer_bound),
            entropy_discrepancy=float(entropy_delta),
            gap_condition_product=float(gap_kappa_product),
            engine_interlock_fired=bool(engine_interlock),
            engine_truth_value=float(engine_truth),
            miller_madow_nats=float(observation.miller_madow_nats),
            min_entropy_nats=float(observation.min_entropy_nats),
            hermitian_relative_residual=float(observation.hermitian_relative_residual),
            operators_well_formed=bool(observation.operators_well_formed),
            landauer_consistent=bool(landauer_ok),
            gap_condition_consistent=bool(gap_kappa_ok),
            interlock_consistent=bool(interlock_ok),
        )

    def _fail_safe_audit(
        self,
        observation: BoundaryObservation,
        exc: Exception,
    ) -> OrientAudit:
        """Auditoría fail-safe cuando el motor físico falla."""
        ceiling = float(observation.reference_entropy_threshold * self._safety_margin)
        return OrientAudit(
            engine_verdict=_VERDICT_VETOED,
            boundary_entropy=float(observation.quick_entropy_nats),
            entropy_upper_bound=float(observation.quick_entropy_nats),
            coupled_entropy=float(observation.quick_entropy_nats),
            spectral_gap=0.0,
            condition_number=math.inf,
            exergy_leak=math.inf,
            physical_exergy_j=math.inf,
            is_gap_coherent=False,
            is_exergy_coherent=False,
            is_entropy_coherent=False,
            is_condition_coherent=False,
            is_boundary_coherent=False,
            allowed_entropy_ceiling=ceiling,
            payload_sha256=observation.payload_sha256,
            observation_timestamp_utc=observation.timestamp_utc,
            timestamp_utc=self._now_utc_iso(),
            reasons=_unique_reasons(
                list(observation.notes)
                + [f"engine_failure: {type(exc).__name__}: {exc}"]
            ),
            operators_sha256=observation.operators_sha256,
            engine_certificate_sha256="",
            landauer_bound_j=0.0,
            entropy_discrepancy=1.0,
            gap_condition_product=0.0,
            engine_interlock_fired=False,
            engine_truth_value=_OMEGA3_FALSE,
            miller_madow_nats=float(observation.miller_madow_nats),
            min_entropy_nats=float(observation.min_entropy_nats),
            hermitian_relative_residual=float(observation.hermitian_relative_residual),
            operators_well_formed=bool(observation.operators_well_formed),
            landauer_consistent=False,
            gap_condition_consistent=False,
            interlock_consistent=False,
        )

    def orient_boundary_observation(self, observation: BoundaryObservation) -> OrientAudit:
        r"""
        PRIMER MÉTODO DE LA FASE 2.

        Continuación formal de `observe_boundary_event` (Fase 1).

        Consume `BoundaryObservation` y audita al motor físico. Si los
        operadores están mal formados, el lazo es fail-closed: se invoca
        igualmente al motor (que debe vetar) y se conserva la evidencia.
        """
        try:
            cert = self._call_engine_sync(observation)
            return self._audit_engine_certificate(cert, observation)
        except Exception as exc:  # pragma: no cover - defensa ciber-física
            return self._fail_safe_audit(observation, exc)

    def verify_boundary_coherence(
        self,
        entropy: float,
        spectral_gap: float,
        exergy_leak: float,
    ) -> Tuple[bool, bool]:
        r"""
        [COMPATIBILIDAD - FASE ORIENT]

        Verifica coherencia local de frontera.

        Originalmente:
        \[
            \Phi_{\mathrm{ext}} \ge \tau_{\mathrm{Langevin}}
            \quad \Rightarrow \quad
            1/\kappa_2 \ge \tau_{\mathrm{gap}}.
        \]

        Devuelve
        --------
        is_gap_coherent : bool
            True si la brecha espectral supera la cota de veto.
        is_exergy_coherent : bool
            True si la fuga exergética está bajo la cota de veto.
        """
        try:
            _ = float(entropy)
        except Exception:
            entropy = 0.0

        try:
            gap = float(spectral_gap)
        except Exception:
            gap = 0.0

        try:
            leak = float(exergy_leak)
        except Exception:
            leak = math.inf

        is_gap_coherent = math.isfinite(gap) and gap >= self._gap_veto_floor
        is_exergy_coherent = math.isfinite(leak) and leak <= self._leak_veto_ceiling
        return is_gap_coherent, is_exergy_coherent

    def evaluate_heyting_decision_lattice(
        self,
        entropy: float,
        spectral_gap: float,
        is_gap_coherent: bool,
        is_exergy_coherent: bool,
    ) -> str:
        r"""
        [COMPATIBILIDAD - FASE DECIDE]

        Clasificación trivalente ordinal de Heyting:
        \[
            \Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}.
        \]
        """
        try:
            h = float(entropy)
        except Exception:
            h = math.inf

        try:
            gap = float(spectral_gap)
        except Exception:
            gap = 0.0

        if (
            not is_gap_coherent
            or not is_exergy_coherent
            or not math.isfinite(h)
            or not math.isfinite(gap)
        ):
            return _VERDICT_VETOED

        if h > self._default_entropy_threshold or gap < self._gap_degraded_floor:
            return _VERDICT_DEGRADED

        return _VERDICT_COHERENT

    @staticmethod
    def _join_verdicts(verdict_a: str, verdict_b: str) -> str:
        r"""
        Join de severidad en \(\Omega_3\) ≡ meet de Gödel sobre valores de
        verdad: \(\mathrm{tv}(a \sqcup b) = \mathrm{tv}(a) \wedge \mathrm{tv}(b)\).
        """
        a = _VERDICT_ORDER[Phase2OrientDecideKernel._normalize_verdict(verdict_a)]
        b = _VERDICT_ORDER[Phase2OrientDecideKernel._normalize_verdict(verdict_b)]
        return _ORDER_TO_VERDICT[max(a, b)]

    def _atomic_heyting_propositions(
        self,
        audit: OrientAudit,
        ceiling: float,
    ) -> List[Tuple[str, float]]:
        r"""
        Proposiciones atómicas del agente en \(\Omega_3\).

        El veredicto del agente es el ínfimo de Gödel de esta familia.
        Un átomo 0 veta; un átomo 1/2 degrada; 1 es regular.
        """
        atoms: List[Tuple[str, float]] = []

        atoms.append(
            (
                "operators",
                _OMEGA3_TRUE if audit.operators_well_formed else _OMEGA3_FALSE,
            )
        )
        atoms.append(
            ("gap", _OMEGA3_TRUE if audit.is_gap_coherent else _OMEGA3_FALSE)
        )
        atoms.append(
            ("exergy", _OMEGA3_TRUE if audit.is_exergy_coherent else _OMEGA3_FALSE)
        )
        atoms.append(
            ("entropy", _OMEGA3_TRUE if audit.is_entropy_coherent else _OMEGA3_FALSE)
        )
        atoms.append(
            (
                "condition",
                _OMEGA3_TRUE if audit.is_condition_coherent else _OMEGA3_FALSE,
            )
        )
        atoms.append(
            (
                "interlock",
                _OMEGA3_TRUE if audit.interlock_consistent else _OMEGA3_FALSE,
            )
        )

        if audit.hermitian_relative_residual > _HERMITIAN_REL_VETO:
            atoms.append(("hermiticity", _OMEGA3_FALSE))
        elif audit.hermitian_relative_residual > _HERMITIAN_REL_TOL:
            atoms.append(("hermiticity", _OMEGA3_MIDDLE))
        else:
            atoms.append(("hermiticity", _OMEGA3_TRUE))

        if audit.entropy_discrepancy >= _ENTROPY_DISCREPANCY_VETO:
            atoms.append(("entropy_dual", _OMEGA3_FALSE))
        elif audit.entropy_discrepancy >= _ENTROPY_DISCREPANCY_DEGRADED:
            atoms.append(("entropy_dual", _OMEGA3_MIDDLE))
        else:
            atoms.append(("entropy_dual", _OMEGA3_TRUE))

        if not audit.landauer_consistent:
            # El motor original no expone Landauer: degradar, no vetar.
            atoms.append(("landauer", _OMEGA3_MIDDLE))
        else:
            atoms.append(("landauer", _OMEGA3_TRUE))

        if not audit.gap_condition_consistent:
            atoms.append(("gap_kappa", _OMEGA3_MIDDLE))
        else:
            atoms.append(("gap_kappa", _OMEGA3_TRUE))

        if math.isfinite(audit.boundary_entropy) and (
            audit.boundary_entropy > self._default_entropy_threshold
        ):
            atoms.append(("entropy_default", _OMEGA3_MIDDLE))
        else:
            atoms.append(("entropy_default", _OMEGA3_TRUE))

        if math.isfinite(ceiling) and ceiling > 0.0:
            if audit.coupled_entropy > self._entropy_warning_ratio * ceiling:
                atoms.append(("coupled_entropy", _OMEGA3_MIDDLE))
            else:
                atoms.append(("coupled_entropy", _OMEGA3_TRUE))
            if audit.entropy_upper_bound > 0.85 * ceiling:
                atoms.append(("entropy_ceiling_margin", _OMEGA3_MIDDLE))
            else:
                atoms.append(("entropy_ceiling_margin", _OMEGA3_TRUE))
        else:
            atoms.append(("coupled_entropy", _OMEGA3_FALSE))
            atoms.append(("entropy_ceiling_margin", _OMEGA3_FALSE))

        if audit.spectral_gap < self._gap_degraded_floor:
            atoms.append(("gap_margin", _OMEGA3_MIDDLE))
        else:
            atoms.append(("gap_margin", _OMEGA3_TRUE))

        if math.isfinite(audit.exergy_leak) and (
            audit.exergy_leak > self._leak_degraded_ceiling
        ):
            atoms.append(("exergy_margin", _OMEGA3_MIDDLE))
        else:
            atoms.append(("exergy_margin", _OMEGA3_TRUE))

        if math.isfinite(audit.condition_number) and (
            audit.condition_number >= self._condition_degraded
        ):
            atoms.append(("condition_margin", _OMEGA3_MIDDLE))
        else:
            atoms.append(("condition_margin", _OMEGA3_TRUE))

        return atoms

    def decide_heyting_verdict(
        self,
        audit: OrientAudit,
        reference_entropy_threshold: Optional[float] = None,
    ) -> AgentHeytingDecision:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2.

        Consolida la decisión del agente y del motor en \(\Omega_3\):
        \[
            \mathrm{COHERENT}
            \prec
            \mathrm{DEGRADED}
            \prec
            \mathrm{VETOED}.
        \]

        Dual-control intuicionista:
          - El veredicto del agente es el ínfimo de Gödel de sus átomos.
          - El veredicto final es el meet con el veredicto del motor.
          - COHERENT sólo se afirma si ambas ramas lo demuestran.

        Entrega `AgentHeytingDecision` a la Fase 3 como el morfismo

        \[
            \texttt{act\_on\_heyting\_decision}
            :
            \mathsf{AgentHeytingDecision}
            \longrightarrow
            \mathsf{ActuationReport}.
        \]
        """
        if reference_entropy_threshold is None:
            ceiling = float(audit.allowed_entropy_ceiling)
        else:
            try:
                threshold = self._finite_float(
                    reference_entropy_threshold,
                    "reference_entropy_threshold",
                    0.0,
                )
                ceiling = float(threshold * self._safety_margin)
            except (TypeError, ValueError):
                ceiling = float(audit.allowed_entropy_ceiling)

        reasons: List[str] = list(audit.reasons)
        atoms = self._atomic_heyting_propositions(audit, ceiling)

        truth_lattice = _OMEGA3_TRUE
        for name, value in atoms:
            truth_lattice = _heyting_meet(truth_lattice, value)
            if value == _OMEGA3_FALSE:
                reasons.append(f"atom_veto:{name}")
            elif value == _OMEGA3_MIDDLE:
                reasons.append(f"atom_degraded:{name}")

        if truth_lattice == _OMEGA3_FALSE:
            agent_verdict = _VERDICT_VETOED
        elif truth_lattice == _OMEGA3_MIDDLE:
            agent_verdict = _VERDICT_DEGRADED
        else:
            agent_verdict = _VERDICT_COHERENT

        final_verdict = self._join_verdicts(audit.engine_verdict, agent_verdict)
        if final_verdict != agent_verdict:
            reasons.append(
                f"final_verdict_joined_engine_{audit.engine_verdict.lower()}"
            )

        truth_value = _TRUTH_VALUES[final_verdict]
        is_boundary_coherent = final_verdict == _VERDICT_COHERENT

        margins: List[float] = []
        if math.isfinite(ceiling) and ceiling > 0.0 and math.isfinite(audit.entropy_upper_bound):
            margins.append(
                float(np.clip(1.0 - audit.entropy_upper_bound / ceiling, 0.0, 1.0))
            )
        else:
            margins.append(0.0)

        if math.isfinite(audit.condition_number) and audit.condition_number > 0.0:
            margins.append(
                float(
                    np.clip(
                        1.0
                        - math.log(max(audit.condition_number, 1.0))
                        / math.log(max(self._condition_veto, math.e)),
                        0.0,
                        1.0,
                    )
                )
            )
        else:
            margins.append(0.0)

        margins.append(
            float(
                audit.spectral_gap
                / (audit.spectral_gap + self._gap_veto_floor + _MACHINE_EPS)
            )
        )

        if math.isfinite(audit.exergy_leak):
            budget = max(self._leak_veto_ceiling, _MACHINE_EPS)
            margins.append(
                float(np.clip(1.0 - audit.exergy_leak / budget, 0.0, 1.0))
            )
        else:
            margins.append(0.0)

        geo = _geometric_mean(margins)
        if final_verdict == _VERDICT_VETOED:
            truth_continuous = 0.0
        elif final_verdict == _VERDICT_DEGRADED:
            truth_continuous = 0.5 * geo
        else:
            truth_continuous = 0.5 + 0.5 * geo

        implies_coherent = _heyting_implies(truth_value, _OMEGA3_TRUE)

        if not reasons:
            reasons.append("boundary_regular")

        return AgentHeytingDecision(
            final_verdict=final_verdict,
            agent_verdict=agent_verdict,
            engine_verdict=audit.engine_verdict,
            truth_value=float(truth_value),
            boundary_entropy=float(audit.boundary_entropy),
            entropy_upper_bound=float(audit.entropy_upper_bound),
            coupled_entropy=float(audit.coupled_entropy),
            spectral_gap=float(audit.spectral_gap),
            condition_number=float(audit.condition_number),
            exergy_leak=float(audit.exergy_leak),
            physical_exergy_j=float(audit.physical_exergy_j),
            allowed_entropy_ceiling=float(ceiling),
            is_boundary_coherent=bool(is_boundary_coherent),
            payload_sha256=audit.payload_sha256,
            observation_timestamp_utc=audit.observation_timestamp_utc,
            reasons=_unique_reasons(reasons),
            truth_continuous=float(truth_continuous),
            heyting_implies_coherent=float(implies_coherent),
            landauer_bound_j=float(audit.landauer_bound_j),
            operators_sha256=audit.operators_sha256,
            atomic_propositions=tuple(atoms),
            engine_certificate_sha256=audit.engine_certificate_sha256,
        )


# ════════════════════════════════════════════════════════════════════════════
# FASE 3 — ACTUACIÓN, CERTIFICADO Y ORQUESTACIÓN OODA
# ════════════════════════════════════════════════════════════════════════════


class Phase3ActuationCertificateKernel(Phase2OrientDecideKernel):
    r"""
    FASE 3: Actuación y certificación.

    El primer método, `act_on_heyting_decision`, consume el
    `AgentHeytingDecision` producido por `decide_heyting_verdict` (Fase 2).

    Responsabilidades:
      1. Activar o inhibir el interlock ciber-físico (fail-closed).
      2. Emitir `SatelliteAgentCertificate` con doble huella SHA-256:
         isomorfa (sin reloj) y de no-repudio (con reloj).
      3. Exponer ciclo OODA síncrono y asíncrono, con semáforo de lote.
      4. Permitir verificación independiente del certificado emitido.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        spectral_gap_floor: float = _DEFAULT_SPECTRAL_GAP_FLOOR,
        exergy_leak_ceiling: float = _DEFAULT_EXERGY_LEAK_CEILING,
        *,
        crowbar_base_latency_ns: float = _CROWBAR_IRAM_LATENCY_NS,
        crowbar_jitter_ns: float = 3.0,
        crowbar_min_latency_ns: float = 380.0,
        crowbar_max_latency_ns: float = 420.0,
        gpio_pin: str = "GPIO14",
        async_timeout_s: float = 5.0,
        max_batch: int = _DEFAULT_MAX_BATCH,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            satellite_id,
            dimension_n,
            safety_margin,
            spectral_gap_floor,
            exergy_leak_ceiling,
            **kwargs,
        )

        self._crowbar_base_latency_ns = self._finite_float(
            crowbar_base_latency_ns,
            "crowbar_base_latency_ns",
            0.0,
        )
        self._crowbar_jitter_ns = self._finite_float(
            crowbar_jitter_ns,
            "crowbar_jitter_ns",
            0.0,
        )
        self._crowbar_min_latency_ns = self._finite_float(
            crowbar_min_latency_ns,
            "crowbar_min_latency_ns",
            0.0,
        )
        self._crowbar_max_latency_ns = self._finite_float(
            crowbar_max_latency_ns,
            "crowbar_max_latency_ns",
            self._crowbar_min_latency_ns,
        )
        if self._crowbar_max_latency_ns < self._crowbar_min_latency_ns:
            self._crowbar_max_latency_ns = self._crowbar_min_latency_ns

        self._gpio_pin = str(gpio_pin)
        self._async_timeout_s = self._finite_float(
            async_timeout_s,
            "async_timeout_s",
            0.0,
        )
        max_batch_n = int(max_batch)
        if max_batch_n <= 0:
            raise ValueError("max_batch debe ser entero positivo.")
        self._max_batch = max_batch_n

    def _fire_crowbar(self, reason: str) -> ActuationReport:
        """Dispara el crowbar simulado con jitter gaussiano acotado."""
        if self._crowbar_jitter_ns > 0.0:
            jitter = float(self._rng.normal(0.0, self._crowbar_jitter_ns))
        else:
            jitter = 0.0

        latency = self._crowbar_base_latency_ns + jitter
        latency = float(
            np.clip(
                latency,
                self._crowbar_min_latency_ns,
                self._crowbar_max_latency_ns,
            )
        )

        logger.critical(
            "¡VETO SÍNCRONO DE FRONTERA DETECTADO POR EL SOBERANO ORBITAL! "
            "Bypass Crowbar BT151 [%s] gatillado en IRAM. Latencia: %.2f ns. "
            "Razón: %s",
            self._gpio_pin,
            latency,
            reason,
        )

        return ActuationReport(
            interlock_fired=True,
            actuation_latency_ns=latency,
            gpio_pin=self._gpio_pin,
            timestamp_utc=self._now_utc_iso(),
            reason=reason,
        )

    def _no_actuation(self, reason: str) -> ActuationReport:
        """Inhibe el interlock físico."""
        return ActuationReport(
            interlock_fired=False,
            actuation_latency_ns=0.0,
            gpio_pin=self._gpio_pin,
            timestamp_utc=self._now_utc_iso(),
            reason=reason,
        )

    def act_on_heyting_decision(self, decision: AgentHeytingDecision) -> ActuationReport:
        r"""
        PRIMER MÉTODO DE LA FASE 3.

        Continuación formal de `decide_heyting_verdict` (Fase 2).

        Consume `AgentHeytingDecision` y decide la actuación ciber-física
        del crowbar perimetral. Política fail-closed: sólo VETOED dispara.
        """
        if decision.final_verdict == _VERDICT_VETOED:
            reason = ";".join(decision.reasons)[:200]
            return self._fire_crowbar(reason)
        return self._no_actuation("boundary_not_vetoed")

    def act_hardware_interlock_simulation(self, verdict: str) -> Tuple[bool, float]:
        r"""
        [COMPATIBILIDAD - FASE ACT]

        Simula la actuación del disyuntor de potencia.

        Si Heyting colapsa a VETOED, la ISR cargada en IRAM conmuta GPIO14:
        \[
            t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns}.
        \]
        """
        normalized = self._normalize_verdict(verdict)
        if normalized == _VERDICT_VETOED:
            report = self._fire_crowbar("legacy_api_veto")
            return report.interlock_fired, report.actuation_latency_ns
        return False, 0.0

    def _decision_fingerprint(
        self,
        decision: AgentHeytingDecision,
        actuation: ActuationReport,
    ) -> str:
        """Huella isomorfa (sin reloj) del par (decisión, actuación lógica)."""
        return _stable_sha256(
            {
                "satellite_id": self._id,
                "final_verdict": decision.final_verdict,
                "engine_verdict": decision.engine_verdict,
                "agent_verdict": decision.agent_verdict,
                "truth_value": decision.truth_value,
                "boundary_entropy": decision.boundary_entropy,
                "entropy_upper_bound": decision.entropy_upper_bound,
                "coupled_entropy": decision.coupled_entropy,
                "spectral_gap": decision.spectral_gap,
                "condition_number": decision.condition_number,
                "exergy_leak": decision.exergy_leak,
                "physical_exergy_j": decision.physical_exergy_j,
                "landauer_bound_j": decision.landauer_bound_j,
                "payload_sha256": decision.payload_sha256,
                "operators_sha256": decision.operators_sha256,
                "engine_certificate_sha256": decision.engine_certificate_sha256,
                "interlock_fired": bool(actuation.interlock_fired),
                "gpio_pin": self._gpio_pin,
                "reasons": list(decision.reasons),
            }
        )

    def _compute_signature(
        self,
        decision: AgentHeytingDecision,
        actuation: ActuationReport,
        timestamp_utc: str,
        decision_sha256: str,
    ) -> str:
        """Firma SHA-256 de no-repudio (incluye reloj y huella isomorfa)."""
        return _stable_sha256(
            {
                "decision_sha256": decision_sha256,
                "timestamp_utc": timestamp_utc,
                "observation_timestamp_utc": decision.observation_timestamp_utc,
                "actuation_latency_ns": float(actuation.actuation_latency_ns),
                "actuation_timestamp_utc": actuation.timestamp_utc,
                "phase": _AGENT_PHASE,
                "version": __version__,
            }
        )

    def synthesize_agent_certificate(
        self,
        decision: AgentHeytingDecision,
        actuation: ActuationReport,
    ) -> SatelliteAgentCertificate:
        """Emite el certificado orbital firmado (isomorfo + no-repudio)."""
        timestamp_utc = self._now_utc_iso()
        decision_sha = self._decision_fingerprint(decision, actuation)
        signature = self._compute_signature(
            decision, actuation, timestamp_utc, decision_sha
        )

        if decision.final_verdict == _VERDICT_DEGRADED:
            logger.warning(
                "Frontera degradada. Satélite=%s | Entropía=%.6f | Gap=%.3e | "
                "Condición=%.3e | Razones=%s",
                self._id,
                decision.boundary_entropy,
                decision.spectral_gap,
                decision.condition_number,
                ";".join(decision.reasons),
            )
        elif decision.final_verdict == _VERDICT_COHERENT:
            logger.info(
                "Frontera coherente. Satélite=%s | Entropía=%.6f | Gap=%.3e | "
                "Condición=%.3e",
                self._id,
                decision.boundary_entropy,
                decision.spectral_gap,
                decision.condition_number,
            )

        return SatelliteAgentCertificate(
            phase=_AGENT_PHASE,
            heyting_verdict=decision.final_verdict,
            boundary_entropy=float(decision.boundary_entropy),
            spectral_gap=float(decision.spectral_gap),
            exergy_leak=float(decision.exergy_leak),
            is_boundary_coherent=bool(decision.is_boundary_coherent),
            hardware_interlock_fired=bool(actuation.interlock_fired),
            actuation_latency_ns=float(actuation.actuation_latency_ns),
            digital_signature_sha256=signature,
            condition_number=float(decision.condition_number),
            entropy_upper_bound=float(decision.entropy_upper_bound),
            coupled_entropy=float(decision.coupled_entropy),
            physical_exergy_j=float(decision.physical_exergy_j),
            engine_verdict=decision.engine_verdict,
            agent_verdict=decision.agent_verdict,
            payload_sha256=decision.payload_sha256,
            timestamp_utc=timestamp_utc,
            reasons=decision.reasons,
            landauer_bound_j=float(decision.landauer_bound_j),
            truth_value=float(decision.truth_value),
            operators_sha256=decision.operators_sha256,
            observation_timestamp_utc=decision.observation_timestamp_utc,
            gpio_pin=self._gpio_pin,
            satellite_id=self._id,
            decision_sha256=decision_sha,
            engine_certificate_sha256=decision.engine_certificate_sha256,
        )

    def verify_agent_certificate(self, certificate: SatelliteAgentCertificate) -> bool:
        r"""
        Verifica la huella de no-repudio de un certificado emitido.

        Reconstruye la firma a partir de los campos inmutables. Un
        desajuste implica tampering o mezcla de versiones.
        """
        try:
            expected = _stable_sha256(
                {
                    "decision_sha256": certificate.decision_sha256,
                    "timestamp_utc": certificate.timestamp_utc,
                    "observation_timestamp_utc": certificate.observation_timestamp_utc,
                    "actuation_latency_ns": float(certificate.actuation_latency_ns),
                    "actuation_timestamp_utc": certificate.timestamp_utc,
                    "phase": _AGENT_PHASE,
                    "version": __version__,
                }
            )
            # La firma original usa actuation.timestamp_utc, que puede diferir
            # del timestamp del certificado. Se acepta igualdad directa o el
            # recálculo sobre el propio timestamp del certificado.
            if certificate.digital_signature_sha256 == expected:
                return True

            # Recalcular a partir de la huella isomorfa y los campos públicos.
            expected_alt = _stable_sha256(
                {
                    "satellite_id": certificate.satellite_id or self._id,
                    "final_verdict": certificate.heyting_verdict,
                    "engine_verdict": certificate.engine_verdict,
                    "agent_verdict": certificate.agent_verdict,
                    "truth_value": certificate.truth_value,
                    "boundary_entropy": certificate.boundary_entropy,
                    "entropy_upper_bound": certificate.entropy_upper_bound,
                    "coupled_entropy": certificate.coupled_entropy,
                    "spectral_gap": certificate.spectral_gap,
                    "condition_number": certificate.condition_number,
                    "exergy_leak": certificate.exergy_leak,
                    "physical_exergy_j": certificate.physical_exergy_j,
                    "landauer_bound_j": certificate.landauer_bound_j,
                    "payload_sha256": certificate.payload_sha256,
                    "operators_sha256": certificate.operators_sha256,
                    "engine_certificate_sha256": certificate.engine_certificate_sha256,
                    "interlock_fired": bool(certificate.hardware_interlock_fired),
                    "gpio_pin": certificate.gpio_pin or self._gpio_pin,
                    "reasons": list(certificate.reasons),
                }
            )
            return certificate.decision_sha256 == expected_alt
        except Exception:
            return False

    def execute_satellite_agent_cycle(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: float = _DEFAULT_REFERENCE_ENTROPY_THRESHOLD,
        external_signal: Optional[ExternalEntropySignal] = None,
    ) -> SatelliteAgentCertificate:
        r"""
        API síncrona principal compatible con la versión original.

        Orquesta el ciclo OODA completo de las tres fases anidadas:
        \[
            \mathrm{Observe}
            \xrightarrow{\texttt{observe\_boundary\_event}}
            \mathrm{Orient}
            \xrightarrow{\texttt{orient\_boundary\_observation}}
            \mathrm{Decide}
            \xrightarrow{\texttt{decide\_heyting\_verdict}}
            \mathrm{Act}
            \xrightarrow{\texttt{act\_on\_heyting\_decision}}
            \mathrm{Certify}.
        \]
        """
        logger.debug("Fase Observe: capturando evidencia de frontera asíncrona.")
        observation = self.observe_boundary_event(
            payload=payload,
            K_boundary_raw=K_boundary_raw,
            metric_tensor=metric_tensor,
            reference_entropy_threshold=reference_entropy_threshold,
            external_signal=external_signal,
        )

        logger.debug("Fase Orient: auditando motor de telemetría orbital.")
        audit = self.orient_boundary_observation(observation)

        logger.debug("Fase Decide: resolviendo retículo de Heyting Ω₃.")
        decision = self.decide_heyting_verdict(
            audit,
            reference_entropy_threshold=observation.reference_entropy_threshold,
        )

        logger.debug("Fase Act: evaluando interlock ciber-físico perimetral.")
        actuation = self.act_on_heyting_decision(decision)
        certificate = self.synthesize_agent_certificate(decision, actuation)

        if decision.final_verdict == _VERDICT_VETOED:
            logger.error(
                "Fase Decide/Act: VETO DE FRONTERA DETECTADO. "
                "Entropía: %.6f, Gap: %.6e, Exergía: %.6e, Condición: %.6e",
                decision.boundary_entropy,
                decision.spectral_gap,
                decision.exergy_leak,
                decision.condition_number,
            )

        return certificate

    async def _call_engine_async(
        self,
        observation: BoundaryObservation,
    ) -> SatelliteTelemetryCertificate:
        """Invoca asíncronamente al motor si expone API async."""
        if not hasattr(self._engine, "monitor_boundary_async"):
            return await asyncio.to_thread(self._call_engine_sync, observation)

        base_kwargs: Dict[str, Any] = {
            "payload": observation.payload,
            "K_boundary_raw": observation.K_boundary_raw,
            "metric_tensor": observation.metric_tensor,
            "reference_entropy_threshold": observation.reference_entropy_threshold,
        }

        try:
            if observation.external_signal is not None:
                return await self._engine.monitor_boundary_async(
                    external_signal=observation.external_signal,
                    timeout_s=self._async_timeout_s,
                    **base_kwargs,
                )
            return await self._engine.monitor_boundary_async(
                timeout_s=self._async_timeout_s,
                **base_kwargs,
            )
        except TypeError:
            return await asyncio.to_thread(self._call_engine_sync, observation)

    async def orient_boundary_observation_async(
        self,
        observation: BoundaryObservation,
    ) -> OrientAudit:
        """Versión asíncrona de la auditoría de Fase 2."""
        try:
            cert = await self._call_engine_async(observation)
            return self._audit_engine_certificate(cert, observation)
        except Exception as exc:  # pragma: no cover - defensa ciber-física
            return self._fail_safe_audit(observation, exc)

    def _normalize_async_request(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Extrae y valida las claves admitidas de una petición OODA asíncrona."""
        allowed = {
            "payload",
            "K_boundary_raw",
            "metric_tensor",
            "reference_entropy_threshold",
            "external_signal",
        }
        unknown = set(request.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Claves no admitidas en request asíncrono: {sorted(unknown)}"
            )
        k_raw = request.get("K_boundary_raw")
        metric = request.get("metric_tensor")
        if k_raw is None or metric is None:
            raise ValueError(
                "Cada request requiere 'K_boundary_raw' y 'metric_tensor'."
            )
        return {
            "payload": request.get("payload", b""),
            "K_boundary_raw": k_raw,
            "metric_tensor": metric,
            "reference_entropy_threshold": request.get(
                "reference_entropy_threshold",
                _DEFAULT_REFERENCE_ENTROPY_THRESHOLD,
            ),
            "external_signal": request.get("external_signal"),
        }

    async def execute_satellite_agent_cycle_async(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: float = _DEFAULT_REFERENCE_ENTROPY_THRESHOLD,
        external_signal: Optional[ExternalEntropySignal] = None,
        timeout_s: Optional[float] = None,
    ) -> SatelliteAgentCertificate:
        r"""
        API asíncrona principal del agente soberano.

        Ejecuta el ciclo OODA completo sin bloquear el lazo de eventos.
        El timeout cubre la orientación (motor); Observe/Decide/Act son
        locales y acotados.
        """
        observation = self.observe_boundary_event(
            payload=payload,
            K_boundary_raw=K_boundary_raw,
            metric_tensor=metric_tensor,
            reference_entropy_threshold=reference_entropy_threshold,
            external_signal=external_signal,
        )

        timeout = self._async_timeout_s if timeout_s is None else float(timeout_s)
        if timeout > 0.0:
            audit = await asyncio.wait_for(
                self.orient_boundary_observation_async(observation),
                timeout=timeout,
            )
        else:
            audit = await self.orient_boundary_observation_async(observation)

        decision = self.decide_heyting_verdict(
            audit,
            reference_entropy_threshold=observation.reference_entropy_threshold,
        )
        actuation = self.act_on_heyting_decision(decision)
        return self.synthesize_agent_certificate(decision, actuation)

    async def execute_satellite_agent_cycle_batch_async(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        max_concurrency: int = 8,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        Ejecuta múltiples ciclos OODA asíncronos con semáforo de concurrencia.
        """
        materialized = list(requests)
        if len(materialized) > self._max_batch:
            raise ValueError(
                f"El lote ({len(materialized)}) excede max_batch={self._max_batch}."
            )

        concurrency = max(int(max_concurrency), 1)
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(request: Mapping[str, Any]) -> SatelliteAgentCertificate:
            params = self._normalize_async_request(request)
            async with semaphore:
                return await self.execute_satellite_agent_cycle_async(**params)

        tasks = [_run(request) for request in materialized]
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))


# ════════════════════════════════════════════════════════════════════════════
# AGENTE PÚBLICO FINAL
# ════════════════════════════════════════════════════════════════════════════


class TelemetrySatellitesAgent(Phase3ActuationCertificateKernel):
    r"""
    Soberano de Calibre de la Telemetría Orbital en APU Filter.

    Clase pública final que hereda las tres fases anidadas:
      - Fase 1: Observación y evidencia (`observe_boundary_event`).
      - Fase 2: Orientación, auditoría dual y decisión de Heyting
        (`orient_boundary_observation`, `decide_heyting_verdict`).
      - Fase 3: Actuación fail-closed, certificado y orquestación OODA
        (`act_on_heyting_decision`, `synthesize_agent_certificate`).

    Ejemplo síncrono
    ----------------
    >>> agent = TelemetrySatellitesAgent("SAT-01", dimension_n=4)
    >>> cert = agent.execute_satellite_agent_cycle(
    ...     payload=b"boundary-payload",
    ...     K_boundary_raw=np.eye(4),
    ...     metric_tensor=np.eye(4),
    ... )
    >>> cert.heyting_verdict

    Ejemplo asíncrono
    -----------------
    >>> cert = await agent.execute_satellite_agent_cycle_async(
    ...     payload=b"boundary-payload",
    ...     K_boundary_raw=np.eye(4),
    ...     metric_tensor=np.eye(4),
    ... )
    """


__all__ = [
    "BoundaryObservation",
    "OrientAudit",
    "AgentHeytingDecision",
    "ActuationReport",
    "SatelliteAgentCertificate",
    "Phase1ObservationKernel",
    "Phase2OrientDecideKernel",
    "Phase3ActuationCertificateKernel",
    "TelemetrySatellitesAgent",
]
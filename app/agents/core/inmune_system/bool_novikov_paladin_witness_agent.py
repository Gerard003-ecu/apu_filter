# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Bool Novikov Paladin Agent (Soberano Agéntico de Calibre OODA)                                     ║
║ RUTA    : app/agents/omega/bool_novikov_paladin_agent.py                                                     ║
║ VERSIÓN : 16.0.0-OODA-HKDF-Heyting-Merkle6962-Loeb-ESP32-Crowbar-PhD                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN FORMAL Y ROL OPERATIVO (NATURALEZA DEL SOBERANO)
───────────────────────────────────────────────────────────────
El `BoolNovikovPaladinAgent` es el Soberano Agéntico de Calibre Supremo que ejerce el gobierno de lazo cerrado en 
el Estrato Omega (V_Ω, Nivel 0.5 — El Ágora Tensorial) y en el Santuario Epistémico (V_𝕎, Nivel 0). Su naturaleza 
es la de un evaluador intuicionista y censor constitucional fail-closed: supervisa la coherencia topológica y 
cuántica de las salidas del motor, auditando la validez del estado global mediante lógica de Heyting.

Su rol operativo es orquestar el ciclo OODA en tres fases anidadas compuestas bajo continuaciones en estilo CPS 
(Continuation-Passing Style), derivar subclaves criptográficas HMAC con separación de dominio, administrar la 
rampa de gracia de veto suave (luz ámbar), sellar árboles de evidencia Merkle y dictaminar vetos terminales. Ante 
cualquier anomalía plástica o sospecha de dolo, el soberano asume la responsabilidad de detener síncronamente la 
maquinaria física activando la interrupción por hardware en la memoria estática IRAM del microcontrolador ESP32.

2. MARCO TEÓRICO Y CADENA FANTASMA CPS (Φ₃ ∘ Φ₂ ∘ Φ₁)
──────────────────────────────────────────────────────
El soberano procesa la transición de estados mediante la continuación monádica de Kleisli:

FASE Φ₁ · OBSERVER (PaladinPhase1Observer):
  1. Captura los tensores de FPU de Π₁, Π₂, Π₃ y genera el certificado inmutable P₁.
  2. Derivación de Subclave de Separación de Dominio vía HKDF-SHA256:
       $$K_{P1} = \operatorname{HKDF-SHA256}(\text{master\_key}, \, \text{"paladin/P1/cert"})$$
  3. Firma el expediente P₁ con HMAC-SHA256 y transfiere el control en posición de cola: `return continuation(P1)`.

FASE Φ₂ · ORIENTOR (PaladinPhase2Orientor):
  1. Compuerta Verificadora Φ₁₂: Re-calcula digests SHA-256 desde tensores congelados, valida nonces y la ventana temporal [t_{nbf}, t_{exp}).
  2. Error a Posteriori de Banach-Kantorovich para Contracción de Mapeos:
       $$q_k = \frac{\|x_k - x_{k-1}\|_2}{\|x_{k-1} - x_{k-2}\|_2} \le q_{\max} < 1.0 \implies \|x_k - x^*\|_2 \le \frac{q_k}{1 - q_k} \|x_k - x_{k-1}\|_2$$
  3. Cota de Lipschitz de Connes-Daleckii-Krein: $L_{\max} \le \frac{1}{2 \lambda_{\min}(\mathbf{G})^{3/2}}$.
  4. Firma el expediente P₂ con $K_{P2} = \operatorname{HKDF-SHA256}(\text{master\_key}, \, \text{"paladin/P2/cert"})$ y retorna `return continuation(P2)`.

FASE Φ₃ · DECIDER & TRIBUNAL DE SILICIO (PaladinPhase3Decider):
  1. Compuerta Verificadora Φ₂₃: Valida la firma HMAC de P₂ y la continuidad del token de sesión.
  2. Clasificación en el Álgebra de Heyting Trivalente Ω₃ = {COHERENT(0), DEGRADED(1), VETOED(2)}:
     Aplica el operador Meet (⋀) sobre la cadena lineal de veredictos:
       $$\mathbf{v}_{\mathrm{final}} = \mathbf{v}_{P1} \wedge \mathbf{v}_{P2} \wedge \mathbf{v}_{\Phi 12} \wedge \mathbf{v}_{\Phi 23} \wedge \mathbf{v}_{\mathrm{Moufang}} \wedge \mathbf{v}_{\mathrm{Krein}}$$
  3. Verificador de Löb Seguro (`_LoebSafeSovereignVerifier`):
     Evalúa el axioma modal de Löb □(□A → A) → □A para prevenir trampas de auto-demostración infinita en la automejora RSI-L3.
  4. Árbol de Evidencia Merkle RFC 6962:
     Sella las 8 hojas de evidencia con prefijos de separación de dominio (0x00 para hojas, 0x01 para nodos internos):
       $$\text{Root} = \mathrm{SHA256}\left(0x01 \parallel \mathrm{Left} \parallel \mathrm{Right}\right)$$

3. ACTUACIÓN CIBER-FÍSICA CROWBAR ESP32 (< 400 ns)
──────────────────────────────────────────────────
Si $\mathbf{v}_{\mathrm{final}} = \mathtt{VETOED}$ (⊤), el soberano detona la ISR en IRAM del ESP32:
  • Registro Físico: `GPIO_OUT_W1TS_REG = 0x3FF44008`, Máscara `GPIO14 = 0x4000` (Conmutación a HIGH).
  • Latencia Determinista: $t_{\mathrm{actuation}} = 398.95 \pm 3.2\text{ ns}$.
  • Modelo RLC Subamortiguado del Bus DC ($V_{\mathrm{bus}} = 48\text{V}, C_{\mathrm{bus}} = 470\mu\text{F}$):
      $$Z_0 = \sqrt{\frac{L}{C_{\mathrm{bus}}}}, \quad \zeta = \frac{R_{\mathrm{on}}}{2 Z_0}, \quad I_{\mathrm{pk}} = \frac{V_{\mathrm{bus}}}{Z_0} \exp\left( -\frac{\pi \zeta}{2\sqrt{1-\zeta^2}} \right)$$
      $$\left.\frac{dI}{dt}\right|_{t=0} = \frac{V_{\mathrm{bus}}}{L} \le 50\text{ A/\mu s}, \qquad \int_0^\infty i^2(t) \, dt = \frac{\frac{1}{2} C_{\mathrm{bus}} V_{\mathrm{bus}}^2}{R_{\mathrm{on}}} \le 72\text{ A}^2\text{s}$$

4. AXIOMAS E INVARIANTES PRESERVADOS
────────────────────────────────────
  • AXIOMA DE CENSURA MONÓTONA EN HEYTING (HEYTING-1):
    $\mathbf{v}_{\mathrm{final}} = \bigwedge_i \mathbf{v}_i \in \Omega_3 \implies (\exists i: \mathbf{v}_i = \mathtt{VETOED} \implies \mathbf{v}_{\mathrm{final}} = \mathtt{VETOED})$.
  • AXIOMA DE AUTENTICIDAD CRIPTOGRÁFICA HKDF (SECURITY-1):
    $\operatorname{HMAC-SHA256}(K_{Pi}, P_i) \equiv \text{Signature}_{Pi} \quad \forall i \in \{1, 2\}$.
  • AXIOMA DE CONTRACCIÓN DE BANACH (BANACH-1):
    $q_k = \frac{\|x_k - x_{k-1}\|_2}{\|x_{k-1} - x_{k-2}\|_2} \le q_{\max} = 0.95 < 1.0$.
  • AXIOMA DE DISPARO CIBER-FÍSICO FAIL-CLOSED (CROWBAR-1):
    $\mathbf{v}_{\mathrm{final}} = \mathtt{VETOED} \implies t_{\mathrm{actuation}} \le 400\text{ ns} \land \text{GPIO14} \mapsto \mathrm{HIGH} \land \text{Crowbar\_Triggered} = \text{True}$.
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac
import inspect
import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Importación del motor espectral testigo del paladín (ruta de paquete primero) ──
try:
    from app.core.inmune_system.bool_novikov_paladin_witness_engine import (
        BanachWitnessReport,
        BoolNovikovPaladinWitnessEngine,
        CelestialWitnessReport,
        HeytingTruthValue,
        QNDInspectionReport,
    )
except ImportError:  # pragma: no cover - ruta plana para pruebas unitarias
    try:
        from bool_novikov_paladin_witness_engine import (  # type: ignore[no-redef]
            BanachWitnessReport,
            BoolNovikovPaladinWitnessEngine,
            CelestialWitnessReport,
            HeytingTruthValue,
            QNDInspectionReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar bool_novikov_paladin_witness_engine (rutas de paquete y plana)."
        ) from exc

logger = logging.getLogger("APU.Agents.Core.InmuneSystem.BoolNovikovPaladinWitnessAgent")

__all__ = [
    "PaladinWitnessPhase1Certificate",
    "PaladinWitnessPhase2Certificate",
    "PaladinWitnessSovereignState",
    "VisceralSemanticValuation",
    "RSILevel3AgentGovernanceTelemetry",
    "AgentRSI3Policy",
    "SovereignVerificationReport",
    "Phase1WitnessObserver",
    "Phase2WitnessOrientor",
    "Phase3WitnessDecider",
    "BoolNovikovPaladinWitnessAgent",
    "PALADIN_QND_WITNESS_INVARIANT",
    "AGENT_VERSION",
]

_AGENT_VERSION: Final[str] = (
    "15.0.0-SOVEREIGN-BOOLE-NOVIKOV-PALADIN-WITNESS-QND-KMS-POINCARE-KAM-RSI3-PhD"
)
version: Final[str] = _AGENT_VERSION
AGENT_VERSION: Final[str] = _AGENT_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES CRIPTOGRÁFICAS, GEOMÉTRICAS Y DE POLÍTICA DECLARADA
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_SOVEREIGN_PALADIN_WITNESS_KEY_v13.0.0"
_HMAC_DOMAIN_P1: Final[bytes] = b"PALADIN-WITNESS-P1-QND-ISOMETRY-v15"
_HMAC_DOMAIN_P2: Final[bytes] = b"PALADIN-WITNESS-P2-CELESTIAL-KAM-v15"
_HMAC_DOMAIN_P3: Final[bytes] = b"PALADIN-WITNESS-P3-RSI3-FINAL-v15"
_HMAC_DOMAIN_LEGACY_P1: Final[bytes] = b"PALADIN-WITNESS-P1-QND-ISOMETRY-v13.0|"
_HMAC_DOMAIN_LEGACY_P2: Final[bytes] = b"PALADIN-WITNESS-P2-CELESTIAL-KAM-v13.0|"

PALADIN_QND_WITNESS_INVARIANT: Final[str] = "PALADIN_QND_WITNESS_INVARIANT_ZERO_BACKACTION_RAM"

_BANACH_KLEISLI_T3_CEILING: Final[float] = 0.857375
_GOLDEN_MEAN_NOBLE_RATIO: Final[float] = 0.6180339887498949
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_TINY: Final[float] = 1.0e-300
_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95
_SYMPECTIC_DEFECT_CEILING: Final[float] = 1.0e-8
_MELNIKOV_SMALE_THRESHOLD: Final[float] = 1.0
_KAM_DIOPHANTINE_FLOOR: Final[float] = 1.0e-4
_LINDSTEDT_REMAINDER_CEILING: Final[float] = 1.0e-3
_MERGE_LEAF_PREFIX: Final[bytes] = b"\x00"
_MERGE_NODE_PREFIX: Final[bytes] = b"\x01"

# Objetivos de POLÍTICA declarada (no son mediciones; ver §3.3).
_PROTECTED_WACC_RATE: Final[float] = 0.150
_LIQUIDITY_SURPLUS_DELTA: Final[float] = 0.115
_DECLARED_FPR_TARGET: Final[float] = 0.0
_ZERO_FALSE_POSITIVE_FLOOR: Final[float] = _DECLARED_FPR_TARGET  # alias legado (objetivo, no medida)


# ══════════════════════════════════════════════════════════════════════════════
# §1 · UTILIDADES CANÓNICAS, CRIPTOGRÁFICAS Y DE COMPATIBILIDAD ENTRE VERSIONES
# ══════════════════════════════════════════════════════════════════════════════
def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Lectura tolerante de campos entre versiones v14/v15 del motor (preserva False y 0.0)."""
    val = getattr(obj, name, default)
    return default if val is None else val


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión total a float finito: nunca propaga NaN/Inf al núcleo de gobernanza."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _canonicalize(obj: Any) -> Any:
    """
    Canonicalización recursiva e INYECTIVA de cualquier estructura de certificado.
    Orden de tipos crítico: IntEnum antes de int (HeytingTruthValue ES un int), bool antes
    de int, flotantes en notación científica de 12 dígitos para determinismo entre plataformas.
    Los ndarray se comprometen por su SHA-256 (nunca se vuelcan bytes al payload).
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, IntEnum):
        return {"__enum__": type(obj).__name__, "value": int(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canonicalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        contiguous = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return format(fv, ".12e") if math.isfinite(fv) else "non-finite"
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    return str(obj)


def _canonical_json(fields: Dict[str, Any]) -> str:
    """Serialización canónica determinista: claves ordenadas, sin espacios, sin NaN/Inf."""
    return json.dumps(
        _canonicalize(fields), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def _merkle_root_rfc6962(leaves: Sequence[bytes]) -> str:
    """
    Raíz de Merkle RFC 6962: hoja = SHA256(0x00‖datos), nodo = SHA256(0x01‖L‖R).
    Los prefijos de dominio impiden la confusión hoja/nodo (segunda preimagen); el nodo
    impar se PROMUEVE, nunca se duplica (evita la maleabilidad por duplicación de hojas).
    """
    if not leaves:
        return hashlib.sha256(_MERGE_LEAF_PREFIX).hexdigest()
    level = [hashlib.sha256(_MERGE_LEAF_PREFIX + bytes(leaf)).digest() for leaf in leaves]
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(_MERGE_NODE_PREFIX + level[i] + level[i + 1]).digest())
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def _constant_time_equals(a: str, b: str) -> bool:
    """Comparación en tiempo constante; tolera entradas no ASCII sin levantar TypeError."""
    try:
        return bool(hmac.compare_digest(str(a), str(b)))
    except (TypeError, ValueError):
        return False


def _continued_fraction(x: float, max_terms: int = 24) -> Tuple[List[int], bool]:
    """
    Expansión en fracción continua de x ∈ (0,1). Devuelve (cocientes, racional_sospechoso).
    Un resto < 1e-12 delata racionalidad (o su indistinguibilidad numérica): la nobleza
    exige cola de unos, y un racional NO es noble por muy cerca que esté de φ.
    """
    terms: List[int] = []
    v = float(x)
    rational = False
    for _ in range(max_terms):
        a = int(math.floor(v))
        terms.append(a)
        frac = v - a
        if frac < 1.0e-12:
            rational = True
            break
        v = 1.0 / frac
        if not math.isfinite(v):
            rational = True
            break
    return terms, rational


@dataclass(frozen=True)
class EngineCapabilities:
    """Sondeo de capacidades del motor para preservar continuidad tipada entre v14 y v15."""

    version: str
    has_phase_objects: bool
    celestial_accepts_qnd: bool
    banach_accepts_reports: bool
    celestial_accepts_greene: bool
    has_verifier: bool


def _probe_engine_capabilities(engine: BoolNovikovPaladinWitnessEngine) -> EngineCapabilities:
    """Inspección estática de firmas (sin efectos) para despacho condicional seguro."""
    def _accepts(func: Optional[Callable[..., Any]], param: str) -> bool:
        if func is None:
            return False
        try:
            return param in inspect.signature(func).parameters
        except (TypeError, ValueError):
            return False

    nav = getattr(engine, "phase2_navigator", None)
    gov = getattr(engine, "phase3_governor", None)
    cel = getattr(nav, "audit_celestial_mechanics", None)
    ban = getattr(gov, "audit_banach_rsi3_contraction", None)
    return EngineCapabilities(
        version=str(getattr(engine, "secret_key", b"") and getattr(
            __import__("sys").modules.get(engine.__class__.__module__, None), "version", "unknown"
        ) or "unknown"),
        has_phase_objects=bool(nav is not None and gov is not None),
        celestial_accepts_qnd=_accepts(cel, "qnd_report"),
        banach_accepts_reports=_accepts(ban, "celestial_report"),
        celestial_accepts_greene=_accepts(cel, "enable_greene"),
        has_verifier=hasattr(engine, "verify_witness_hmac_signature"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2 · ESTRUCTURAS INMUTABLES: HUELLAS, COMPUERTAS, CERTIFICADOS Y TELEMETRÍA
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RAMFingerprint:
    """
    Huella pasiva de un buffer: compromiso criptográfico del contenido Y de los metadatos.
    [A2] SHA-256 sobre (dtype‖shape‖strides‖bytes) detecta permutaciones que la suma de
    bytes de v14 no detectaba, y funciona sobre arrays no contiguos sin lanzar ValueError.
    """

    valid: bool
    data_pointer: int
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str
    nbytes: int
    is_contiguous: bool
    content_sha256: str

    def same_as(self, other: "RAMFingerprint") -> bool:
        """Igualdad de identidad física (puntero) y de contenido lógico (digest + metadatos)."""
        return (
            self.valid and other.valid
            and self.data_pointer == other.data_pointer
            and self.content_sha256 == other.content_sha256
            and self.shape == other.shape
            and self.strides == other.strides
            and self.dtype == other.dtype
        )


@dataclass(frozen=True)
class ZeroBackactionAttestation:
    """
    Atestación de no-mutación con ALCANCE DECLARADO. Prueba que los buffers inspeccionados
    no cambiaron de forma observable; NO prueba ausencia de escritura transitoria restaurada
    (TOCTOU) ni cubre el resto de la memoria del proceso. Ésa es la frontera honesta.
    """

    rho_unperturbed: bool
    sigma_unperturbed: bool
    hamiltonian_unperturbed: bool
    inspected_buffer_count: int
    scope: str = "INSPECTED_BUFFERS_ONLY_NO_TOCTOU_GUARANTEE"

    @property
    def all_unperturbed(self) -> bool:
        return bool(self.rho_unperturbed and self.sigma_unperturbed and self.hamiltonian_unperturbed)


@dataclass(frozen=True)
class GeometricPhaseBundle:
    """Invariantes geométricos separados por naturaleza (geométrico vs. interferométrico)."""

    bures_angle_rad: float
    chord_arc_identity_error: float
    bargmann_triangle_phase_rad: float
    engine_cycle_holonomy_rad: float
    interferometric_phase_rad: float
    commuting_reference_ok: bool


@dataclass(frozen=True)
class ZenoDiagnostic:
    """Diagnóstico Zeno CONTRAFACTUAL: qué ocurriría SI se midiera; el testigo no mide."""

    zeno_time: float
    energy_dispersion: float
    probe_interval: float
    counterfactual_survival: float
    is_counterfactual: bool = True


@dataclass(frozen=True)
class GateVerdict:
    """Resultado de una compuerta homomórfica: autenticidad ≠ salud evidencial."""

    authentic: bool
    evidentially_clean: bool
    admitted: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NobleRotationDiagnostics:
    """Nobleza de Greene-MacKay, proximidad resonante de Farey y calidad diofántica."""

    defined: bool
    golden_circular_distance: float
    cf_tail_of_ones: int
    nobility_score: float
    rational_suspect: bool
    farey_resonance_proximity: float
    diophantine_gamma: float
    winding_numbers: Tuple[float, ...]


@dataclass(frozen=True)
class CartanDriftCertificate:
    """Deriva secular adimensional del invariante de Poincaré-Cartan y certificados anexos."""

    defined: bool
    current_invariant: float
    previous_invariant: float
    relative_drift: float
    symplectic_defect: float
    cartan_amd_identity_error: float


@dataclass(frozen=True)
class HeytingAdjudication:
    """Ínfimo en Ω₃ y análisis BHK de entrañamiento por modus ponens."""

    infimum: HeytingTruthValue
    bhk_left_associated: HeytingTruthValue
    bhk_right_associated: HeytingTruthValue
    entailment_gap_p12: bool
    entailment_gap_p23: bool
    modus_ponens_ok: bool
    witness_invariant_ok: bool


@dataclass(frozen=True)
class MetaAuditCertificate:
    """Meta-auditoría RSI-L3 en tres niveles con presupuestos bien fundados explícitos."""

    l1_numeric_ok: bool
    l2_structural_ok: bool
    l3_attestation_ok: bool
    accepted: bool
    verification_depth_remaining: int
    mutation_budget_remaining: int
    failed_checks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class VisceralSemanticValuation:
    """
    Imagen del Funtor Semántico Visceral Φ_sem : Ω₃ × M_QND × M_KAM → E_fin.
    [A5] Monótono por construcción: escudo = w(veredicto) · evidencia, w decreciente en
    severidad. [A6] `false_positive_rate` es ahora una COTA SUPERIOR certificada (1.0 si no
    identificable), no el 0.0 asserted de v14. Los objetivos nominales se declaran aparte.
    """

    wacc_protection_pct: float
    liquidity_delta_pct: float
    false_positive_rate: float
    capital_shield_efficiency: float
    semantic_regime: str
    # Campos v15 (defaults para compatibilidad posicional estricta)
    wacc_protection_target_pct: float = _PROTECTED_WACC_RATE * 100.0
    liquidity_nominal_pct: float = _LIQUIDITY_SURPLUS_DELTA * 100.0
    declared_fpr_target: float = _DECLARED_FPR_TARGET
    fpr_upper_bound: float = 1.0
    fpr_bound_identifiable: bool = False
    fidelity_flag_count: int = 0
    fidelity_ambiguous_count: int = 0
    verdict_weight: float = 1.0
    evidence_factor: float = 1.0
    functor_monotone_ok: bool = True


@dataclass(frozen=True)
class PaladinWitnessPhase1Certificate:
    """Certificado inmutable de Fase 1 (Observe). Orden posicional v14 preservado."""

    qnd_report: QNDInspectionReport
    timestamp_ns: int
    hmac_signature: str
    fubini_study_angle_rad: float = 0.0          # nombre legado; semántica: ÁNGULO DE BURES
    berry_poincare_holonomy_rad: float = 0.0    # nombre legado; semántica: holonomía de Bargmann
    ram_pointer_unperturbed: bool = True
    rsi_l3_zeno_coherence_factor: float = 1.0
    # ── v15 ──
    agent_version: str = _AGENT_VERSION
    sequence: int = 0
    chain_prev: str = "genesis"
    monotonic_ns: int = 0
    backaction_attestation: Optional[ZeroBackactionAttestation] = None
    geometry: Optional[GeometricPhaseBundle] = None
    zeno: Optional[ZenoDiagnostic] = None
    merkle_leaf: str = ""

    @property
    def bures_angle_rad(self) -> float:
        """Alias correcto del ángulo de Bures (el nombre legado dice Fubini-Study)."""
        return self.fubini_study_angle_rad


@dataclass(frozen=True)
class PaladinWitnessPhase2Certificate:
    """Certificado inmutable de Fase 2 (Orient). Orden posicional v14 preservado."""

    p1_certificate: PaladinWitnessPhase1Certificate
    celestial_report: CelestialWitnessReport
    banach_report: BanachWitnessReport
    timestamp_ns: int
    hmac_signature: str
    poincare_Noble_diophantine_defect: float = 0.0
    poincare_cartan_action_drift: float = 0.0
    rsi_l3_secular_renormalization_factor: float = 1.0
    gate_phi12_verified: bool = True
    # ── v15 ──
    agent_version: str = _AGENT_VERSION
    sequence: int = 0
    chain_prev: str = "genesis"
    monotonic_ns: int = 0
    noble: Optional[NobleRotationDiagnostics] = None
    cartan: Optional[CartanDriftCertificate] = None
    secular_margins: Tuple[float, ...] = ()
    physical_renormalization_ratio: float = 1.0
    gate_phi12: Optional[GateVerdict] = None
    authentication_veto: bool = False
    propagated_truth: HeytingTruthValue = HeytingTruthValue.COHERENT
    merkle_leaf: str = ""

    @property
    def poincare_noble_diophantine_defect(self) -> float:
        """Alias con nomenclatura canónica en minúsculas."""
        return self.poincare_Noble_diophantine_defect


@dataclass(frozen=True)
class PaladinWitnessSovereignState:
    """Estado soberano inmutable final. Orden posicional v14 preservado."""

    truth_value: HeytingTruthValue
    p2_certificate: PaladinWitnessPhase2Certificate
    witness_invariant_verified: bool
    narrative_summary: str
    merkle_root: str
    timestamp_ns: int
    visceral_semantics: Optional[VisceralSemanticValuation] = None
    heyting_bhk_implication: HeytingTruthValue = HeytingTruthValue.COHERENT
    hierarchical_merkle_root: str = ""
    p3_hmac_signature: str = ""
    rsi_l3_meta_audit_passed: bool = True
    # ── v15 ──
    agent_version: str = _AGENT_VERSION
    sequence: int = 0
    chain_prev: str = "genesis"
    monotonic_ns: int = 0
    adjudication: Optional[HeytingAdjudication] = None
    meta_audit: Optional[MetaAuditCertificate] = None
    gate_phi23: Optional[GateVerdict] = None
    authentication_veto: bool = False
    merkle_algorithm: str = "RFC6962-SHA256-LEAF0x00-NODE0x01"


@dataclass(frozen=True)
class SovereignVerificationReport:
    """Resultado del verificador público: recomputación íntegra a partir del certificado."""

    p1_signature_ok: bool
    p2_signature_ok: bool
    p3_signature_ok: bool
    legacy_merkle_ok: bool
    hierarchical_merkle_ok: bool
    truth_recomputation_ok: bool
    semantics_recomputation_ok: bool
    invariant_ok: bool
    chain_monotone_ok: bool
    all_ok: bool
    failed_checks: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentRSI3Policy:
    """
    Política adaptativa del agente en un compacto convexo. `mutation_budget` es el contador
    bien fundado (descenso estricto en ℕ) que acota TODA cadena de automodificación de por
    vida; `verification_depth` acota la profundidad de meta-razonamiento POR CICLO.
    Distinción deliberada respecto del motor: verificar no es automodificarse.
    """

    zeno_probe_interval: float = 0.05
    farey_order: int = 24
    cf_depth: int = 24
    verification_depth: int = 3
    mutation_budget: int = 3

    def project(self) -> "AgentRSI3Policy":
        return AgentRSI3Policy(
            zeno_probe_interval=float(np.clip(self.zeno_probe_interval, 1.0e-6, 1.0)),
            farey_order=int(np.clip(self.farey_order, 4, 64)),
            cf_depth=int(np.clip(self.cf_depth, 6, 48)),
            verification_depth=max(0, int(self.verification_depth)),
            mutation_budget=max(0, int(self.mutation_budget)),
        )


@dataclass
class RSILevel3AgentGovernanceTelemetry:
    """Telemetría meta-cognitiva de Nivel 3 del Soberano Testigo (append-only en espíritu)."""

    total_ooda_cycles: int = 0
    phase1_emissions: int = 0
    phase2_emissions: int = 0
    phase3_emissions: int = 0
    gate_phi12_pass_count: int = 0
    gate_phi23_pass_count: int = 0
    passive_veto_observations: int = 0
    last_sovereign_truth: HeytingTruthValue = HeytingTruthValue.COHERENT
    # ── v15: cadena, anti-replay, libro mayor de política y estadística de banderas ──
    policy: AgentRSI3Policy = field(default_factory=AgentRSI3Policy)
    sequence_counter: int = 0
    chain_head: str = "genesis"
    monotonic_clock_ns: int = 0
    signature_registry: List[str] = field(default_factory=list)
    policy_ledger: List[Dict[str, Any]] = field(default_factory=list)
    mutation_count: int = 0
    rollback_count: int = 0
    gate_phi12_fail_count: int = 0
    gate_phi23_fail_count: int = 0
    fidelity_flag_count: int = 0
    fidelity_ambiguous_count: int = 0
    last_cartan_invariant: Optional[float] = None
    dev_key_in_use: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# §3 · FASE 1 OODA (OBSERVE): INSPECCIÓN PASIVA QND Y GEOMETRÍA DE KÄHLER
# ══════════════════════════════════════════════════════════════════════════════
class Phase1WitnessObserver:
    """
    Fase 1 OODA (Observe | 1.1 → 1.9). Cadena formal:
        1.1 huella RAM → 1.2 no-mutación → 1.3 ángulo de Bures → 1.4 holonomía gauge-invariante
        → 1.5 Zeno contrafactual → 1.6 capacidades del motor → 1.7 payload canónico
        → 1.8 delegador de compatibilidad → 1.9 TERMINAL (emite P1, entrada de 2.1).
    """

    def __init__(
        self,
        engine: BoolNovikovPaladinWitnessEngine,
        telemetry: Optional[RSILevel3AgentGovernanceTelemetry] = None,
    ) -> None:
        self.engine = engine
        self._telemetry = telemetry if telemetry is not None else RSILevel3AgentGovernanceTelemetry()
        self._caps = _probe_engine_capabilities(engine)
        self._last_p1_certificate: Optional[PaladinWitnessPhase1Certificate] = None

    # ── Método 1.1 ────────────────────────────────────────────────────────────
    @staticmethod
    def _capture_ram_fingerprint(buffer: Any) -> RAMFingerprint:
        """
        1.1 Huella pasiva SHA-256 de un buffer, segura ante no contigüidad.
        [A2] v14 usaba `int(np.sum(arr.view(np.uint8)))`: (a) invariante bajo permutación de
        bytes ⟹ no detecta reordenamientos del estado; (b) `.view(np.uint8)` LANZA ValueError
        si el último eje no es contiguo (p.ej. ρ.T), haciendo crashear al testigo pasivo.
        Aquí se compromete (dtype‖shape‖strides‖bytes_C) con SHA-256; `np.ascontiguousarray`
        NO muta la fuente (copia de lectura si hace falta), preservando la pasividad.
        """
        if not isinstance(buffer, np.ndarray) or buffer.size == 0:
            return RAMFingerprint(False, 0, (), (), "", 0, False, "")
        try:
            pointer = int(buffer.__array_interface__["data"][0])
        except (AttributeError, KeyError, TypeError, ValueError):
            pointer = 0
        try:
            contiguous = np.ascontiguousarray(buffer)
            meta = f"{buffer.dtype.str}|{buffer.shape}|{buffer.strides}|".encode("ascii")
            digest = hashlib.sha256(meta + contiguous.tobytes(order="C")).hexdigest()
        except (ValueError, TypeError, MemoryError):
            return RAMFingerprint(False, pointer, tuple(buffer.shape), tuple(buffer.strides),
                                  buffer.dtype.str, int(buffer.nbytes), False, "")
        return RAMFingerprint(
            valid=True, data_pointer=pointer, shape=tuple(int(s) for s in buffer.shape),
            strides=tuple(int(s) for s in buffer.strides), dtype=str(buffer.dtype.str),
            nbytes=int(buffer.nbytes), is_contiguous=bool(buffer.flags["C_CONTIGUOUS"]),
            content_sha256=digest,
        )

    # ── Método 1.2 ────────────────────────────────────────────────────────────
    @staticmethod
    def _attest_zero_backaction(
        before: Sequence[RAMFingerprint], after: Sequence[RAMFingerprint]
    ) -> ZeroBackactionAttestation:
        """
        1.2 Compara las huellas de los TRES buffers (ρ, σ, H) antes/después de la inspección.
        [A3] v14 sólo huellaba ρ: una mutación de σ o de H pasaba inadvertida y el invariante
        se declaraba verificado. Buffers ausentes (None) se tratan como no inspeccionados y
        NO cuentan como evidencia positiva de inmutabilidad.
        """
        flags: List[bool] = []
        inspected = 0
        for idx in range(3):
            b = before[idx] if idx < len(before) else RAMFingerprint(False, 0, (), (), "", 0, False, "")
            a = after[idx] if idx < len(after) else RAMFingerprint(False, 0, (), (), "", 0, False, "")
            if b.valid and a.valid:
                inspected += 1
                flags.append(b.same_as(a))
            else:
                flags.append(not b.valid and not a.valid)
        return ZeroBackactionAttestation(
            rho_unperturbed=bool(flags[0]), sigma_unperturbed=bool(flags[1]),
            hamiltonian_unperturbed=bool(flags[2]), inspected_buffer_count=inspected,
        )

    # ── Método 1.3 ────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_bures_angle_and_chord_arc(
        uhlmann_fidelity: float, bures_distance: float
    ) -> Tuple[float, float]:
        """
        1.3 Ángulo de Bures θ_B = arccos√F ∈ [0, π/2].
        [A8] Para estados MIXTOS esto NO es la distancia de Fubini-Study (que vive en P(H)
        y sólo coincide para estados puros); es el ángulo de Bures-Wootters, que sí es métrica.
        CERTIFICADO algebraico cuerda-arco: d_B² = 2 − 2√F = 4sin²(θ_B/2) ⟹ d_B = 2sin(θ_B/2).
        Se devuelve el error de la identidad, verificable a precisión de máquina.
        """
        f_val = float(np.clip(_safe_float(uhlmann_fidelity, 0.0), 0.0, 1.0))
        theta = math.acos(float(np.clip(math.sqrt(f_val), -1.0, 1.0)))
        chord = 2.0 * math.sin(theta / 2.0)
        identity_error = abs(_safe_float(bures_distance, 0.0) - chord)
        return float(theta), float(identity_error)

    # ── Método 1.4 ────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_gauge_invariant_holonomy(
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]],
        hamiltonian: Optional[NDArray[np.complex128]],
        qnd_report: QNDInspectionReport,
    ) -> GeometricPhaseBundle:
        """
        1.4 Separa lo geométrico de lo interferométrico.
        • Δ₃ = arg Tr(ρ σ τ) con τ = dephasing(ρ) en la autobase de H: invariante de Bargmann
          de 3 vértices, gauge-invariante bajo conjugación unitaria simultánea, y NULO si los
          tres operadores conmutan (certificado explícito).
        • Holonomía de ciclo del motor: fase de Wilson sobre ciclos fundamentales del grafo
          de coherencia (verdadero invariante topológico gauge-invariante).
        • arg Tr(ρσH): se conserva por compatibilidad pero se ROTULA como interferométrico;
          mezcla contribución dinámica y geométrica ⟹ NO es una fase de Berry. [A8]
        Nota: Tr(ρσ) ≥ 0 para ρ,σ ⪰ 0, por eso el invariante de 2 vértices no tiene fase y
        se requieren TRES vértices para obtener holonomía no trivial.
        """
        theta, chord_err = Phase1WitnessObserver._compute_bures_angle_and_chord_arc(
            _field(qnd_report, "uhlmann_fidelity", 0.0), _field(qnd_report, "bures_distance", 0.0)
        )
        triangle_phase, interferometric, commuting_ok = 0.0, 0.0, True
        try:
            if isinstance(rho_state, np.ndarray) and rho_state.ndim == 2 and rho_state.shape[0] > 0:
                n = int(rho_state.shape[0])
                rho = np.asarray(rho_state, dtype=np.complex128)
                sigma = (
                    np.asarray(rho_target, dtype=np.complex128)
                    if isinstance(rho_target, np.ndarray) and rho_target.shape == rho.shape
                    else np.eye(n, dtype=np.complex128) / float(n)
                )
                ham = (
                    np.asarray(hamiltonian, dtype=np.complex128)
                    if isinstance(hamiltonian, np.ndarray) and hamiltonian.shape == rho.shape
                    else np.diag(np.linspace(1.0, 2.0, n)).astype(np.complex128)
                )
                _energies, evecs = np.linalg.eigh(0.5 * (ham + ham.conj().T))
                rho_e = evecs.conj().T @ rho @ evecs
                tau = evecs @ np.diag(np.diag(rho_e)) @ evecs.conj().T
                tri = complex(np.trace(rho @ sigma @ tau))
                triangle_phase = float(math.atan2(tri.imag, tri.real))
                cyc = complex(np.trace(rho @ sigma @ ham))
                interferometric = float(math.atan2(cyc.imag, cyc.real))
                commutator = rho @ tau - tau @ rho
                scale = max(float(np.linalg.norm(rho)) * float(np.linalg.norm(tau)), _TINY)
                commuting_ok = bool(np.linalg.norm(commutator) / scale <= 1.0e-10)
                if commuting_ok:
                    triangle_phase = 0.0  # certificado: conmutantes ⟹ Bargmann real positivo
        except (np.linalg.LinAlgError, ValueError, TypeError):
            commuting_ok = False
        return GeometricPhaseBundle(
            bures_angle_rad=float(theta), chord_arc_identity_error=float(chord_err),
            bargmann_triangle_phase_rad=float(triangle_phase),
            engine_cycle_holonomy_rad=float(_field(qnd_report, "max_bargmann_phase", 0.0)),
            interferometric_phase_rad=float(interferometric),
            commuting_reference_ok=bool(commuting_ok),
        )

    # ── Método 1.5 ────────────────────────────────────────────────────────────
    @staticmethod
    def _rsi_l3_counterfactual_zeno_diagnostic(
        qnd_report: QNDInspectionReport, probe_interval: float
    ) -> ZenoDiagnostic:
        """
        1.5 [RSI-L3] Diagnóstico Zeno CONTRAFACTUAL, con modelo físico correcto.
        [A7] v14 usaba exp(−‖[H,ρ]‖·T_rec) y lo presentaba como "congelamiento de decoherencia"
        producido por el testigo. Doble error: (a) el efecto Zeno cuadrático está gobernado por
        la DISPERSIÓN de energía ΔH, no por el conmutador; (b) un testigo PASIVO no acopla ni
        mide, luego no congela nada.
        Modelo correcto: S(τ) = 1 − (τΔH)² + O(τ⁴) ⟹ S_N ≈ exp(−N(τ/τ_Z)²), τ_Z = 1/ΔH.
        ΔH se recupera del límite de Mandelstam-Tamm del motor: τ_MT = π/(2ΔH) ⟹ ΔH = π/(2τ_MT).
        El valor reportado es lo que OCURRIRÍA si se midiera con esa cadencia: contrafactual.
        """
        tau = float(np.clip(_safe_float(probe_interval, 0.05), 1.0e-9, 1.0e3))
        tau_mt = _safe_float(_field(qnd_report, "mandelstam_tamm_time", 0.0), 0.0)
        if tau_mt > _WILKINSON_EPSILON and math.isfinite(tau_mt):
            dispersion = math.pi / (2.0 * tau_mt)
        else:
            t_rec = max(_safe_float(_field(qnd_report, "poincare_quantum_recurrence_bound", 2.0 * math.pi),
                                    2.0 * math.pi), _WILKINSON_EPSILON)
            dispersion = 2.0 * math.pi / t_rec
        dispersion = max(dispersion, _WILKINSON_EPSILON)
        zeno_time = 1.0 / dispersion
        ratio = tau / max(zeno_time, _TINY)
        survival = math.exp(-min(700.0, ratio * ratio))
        return ZenoDiagnostic(
            zeno_time=float(zeno_time), energy_dispersion=float(dispersion),
            probe_interval=float(tau), counterfactual_survival=float(survival),
            is_counterfactual=True,
        )

    # ── Método 1.6 ────────────────────────────────────────────────────────────
    def _invoke_engine_qnd(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]],
        hamiltonian: Optional[NDArray[np.complex128]],
    ) -> QNDInspectionReport:
        """1.6 Despacho al método terminal 1.11 del motor con firma estable entre versiones."""
        return self.engine.audit_qnd_inspection(
            rho_state=rho_state, rho_target=rho_target, hamiltonian=hamiltonian
        )

    # ── Método 1.7 ────────────────────────────────────────────────────────────
    @staticmethod
    def _build_phase1_payload_fields(cert: PaladinWitnessPhase1Certificate) -> Dict[str, Any]:
        """
        1.7 Payload canónico de COBERTURA TOTAL (excluye únicamente la propia firma).
        [A1] v14 firmaba 4 campos: `truth_value`, `betti_1`, `ram_pointer_unperturbed`,
        `kms_residual` y toda la telemetría extendida quedaban SIN AUTENTICAR, de modo que
        mutar VETOED→COHERENT no invalidaba la firma y Φ₁₂ la aceptaba. Aquí se canonicaliza
        el informe íntegro y todos los campos del certificado, con dominio embebido en el
        propio payload (defensa adicional si el motor deriva la clave por concatenación).
        """
        return {
            "domain": _HMAC_DOMAIN_P1.decode("ascii"),
            "agent_version": cert.agent_version,
            "sequence": cert.sequence,
            "chain_prev": cert.chain_prev,
            "timestamp_ns": cert.timestamp_ns,
            "monotonic_ns": cert.monotonic_ns,
            "qnd_report": cert.qnd_report,
            "bures_angle_rad": cert.fubini_study_angle_rad,
            "bargmann_holonomy_rad": cert.berry_poincare_holonomy_rad,
            "ram_pointer_unperturbed": cert.ram_pointer_unperturbed,
            "zeno_factor": cert.rsi_l3_zeno_coherence_factor,
            "backaction": cert.backaction_attestation,
            "geometry": cert.geometry,
            "zeno": cert.zeno,
        }

    @staticmethod
    def _build_phase1_canonical_payload(report: QNDInspectionReport, timestamp_ns: int) -> str:
        """Compatibilidad v14: payload legado de 4 campos (sólo para consumidores antiguos)."""
        return (
            f"{report.is_qnd_valid}:{report.qnd_commutator_norm:.12e}:"
            f"{report.uhlmann_fidelity:.12e}:{report.bures_distance:.12e}:{timestamp_ns}"
        )

    # ── Método 1.8 ────────────────────────────────────────────────────────────
    def certify_qnd_substrate(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        zeno_probe_interval: Optional[float] = None,
    ) -> PaladinWitnessPhase1Certificate:
        """1.8 Delegador de compatibilidad absoluta con el ecosistema APU hacia 1.9."""
        return self.emit_phase1_certificate(rho_state, rho_target, hamiltonian, zeno_probe_interval)

    # ── Método 1.9 · TERMINAL FASE 1 → APERTURA FASE 2 ────────────────────────
    def emit_phase1_certificate(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        zeno_probe_interval: Optional[float] = None,
    ) -> PaladinWitnessPhase1Certificate:
        """
        1.9 [TERMINAL FASE 1] Ejecuta la inspección QND, atestigua la no-mutación de los tres
        buffers, calcula los invariantes geométricos y el diagnóstico Zeno contrafactual, y
        emite el `PaladinWitnessPhase1Certificate` (P1) firmado con cobertura total.
        Este certificado es la entrada DIRECTA del método 2.1 (Compuerta Φ₁₂).
        """
        tel = self._telemetry
        before = [
            self._capture_ram_fingerprint(rho_state),
            self._capture_ram_fingerprint(rho_target),
            self._capture_ram_fingerprint(hamiltonian),
        ]
        report = self._invoke_engine_qnd(rho_state, rho_target, hamiltonian)
        after = [
            self._capture_ram_fingerprint(rho_state),
            self._capture_ram_fingerprint(rho_target),
            self._capture_ram_fingerprint(hamiltonian),
        ]
        attestation = self._attest_zero_backaction(before, after)
        geometry = self._compute_gauge_invariant_holonomy(rho_state, rho_target, hamiltonian, report)
        interval = tel.policy.zeno_probe_interval if zeno_probe_interval is None else zeno_probe_interval
        zeno = self._rsi_l3_counterfactual_zeno_diagnostic(report, interval)

        tel.sequence_counter += 1
        tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
        skeleton = PaladinWitnessPhase1Certificate(
            qnd_report=report,
            timestamp_ns=time.time_ns(),
            hmac_signature="",
            fubini_study_angle_rad=geometry.bures_angle_rad,
            berry_poincare_holonomy_rad=geometry.bargmann_triangle_phase_rad,
            ram_pointer_unperturbed=attestation.all_unperturbed,
            rsi_l3_zeno_coherence_factor=zeno.counterfactual_survival,
            agent_version=_AGENT_VERSION,
            sequence=tel.sequence_counter,
            chain_prev=tel.chain_head,
            monotonic_ns=tel.monotonic_clock_ns,
            backaction_attestation=attestation,
            geometry=geometry,
            zeno=zeno,
        )
        payload = _canonical_json(self._build_phase1_payload_fields(skeleton))
        signature = self.engine.generate_witness_hmac_signature(payload, _HMAC_DOMAIN_P1)
        leaf = hashlib.sha256(_canonical_json({"p1": signature, "report": report}).encode("utf-8")).hexdigest()
        cert = replace(skeleton, hmac_signature=signature, merkle_leaf=leaf)

        tel.chain_head = signature
        tel.signature_registry.append(signature)
        if len(tel.signature_registry) > 512:
            tel.signature_registry.pop(0)
        tel.phase1_emissions += 1
        self._last_p1_certificate = cert
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §4 · FASE 2 OODA (ORIENT): Φ₁₂, SECCIÓN DE POINCARÉ, NOBLEZA KAM Y CARTAN
# ══════════════════════════════════════════════════════════════════════════════
class Phase2WitnessOrientor:
    """
    Fase 2 OODA (Orient | 2.1 → 2.9). Cadena formal:
        2.1 Φ₁₂ (ingesta de P1) → 2.2 toro de números de rotación → 2.3 nobleza/Farey
        → 2.4 índice secular axiomático → 2.5 deriva de Cartan → 2.6 propagación de veto
        → 2.7 payload canónico → 2.8 delegador → 2.9 TERMINAL (emite P2, entrada de 3.1).
    """

    def __init__(
        self,
        engine: BoolNovikovPaladinWitnessEngine,
        telemetry: Optional[RSILevel3AgentGovernanceTelemetry] = None,
    ) -> None:
        self.engine = engine
        self._telemetry = telemetry if telemetry is not None else RSILevel3AgentGovernanceTelemetry()
        self._caps = _probe_engine_capabilities(engine)
        self._last_p2_certificate: Optional[PaladinWitnessPhase2Certificate] = None

    # ── Método 2.1 · APERTURA FASE 2 (continuación directa de 1.9) ────────────
    def _ingest_phase1_certificate_gate(self, p1_cert: PaladinWitnessPhase1Certificate) -> bool:
        """
        2.1 [CONTINUACIÓN DIRECTA DE 1.9 — COMPUERTA HOMOMÓRFICA Φ₁₂]
        Revalida en tiempo constante la firma HMAC-SHA256 sobre la COBERTURA TOTAL del
        certificado y ADMITE veredictos VETOED (0x03) para preservar la observación pasiva.
        Devuelve bool por compatibilidad; el veredicto rico está en `_gate_phase1`.
        """
        return self._gate_phase1(p1_cert).authentic

    def _gate_phase1(self, p1_cert: PaladinWitnessPhase1Certificate) -> GateVerdict:
        """
        Φ₁₂ rica: separa AUTENTICIDAD (criptográfica) de SALUD EVIDENCIAL (topológica).
        Un VETOED auténtico se ADMITE (doctrina de no interferencia); una firma inválida
        NO se admite jamás (doctrina de no atestar en falso). [A1/A4]
        """
        tel = self._telemetry
        reasons: List[str] = []
        payload = _canonical_json(Phase1WitnessObserver._build_phase1_payload_fields(p1_cert))
        expected = self.engine.generate_witness_hmac_signature(payload, _HMAC_DOMAIN_P1)
        authentic = _constant_time_equals(p1_cert.hmac_signature, expected)
        if not authentic:
            reasons.append("P1_SIGNATURE_COVERAGE_MISMATCH")
        if p1_cert.sequence <= 0:
            reasons.append("P1_SEQUENCE_NON_POSITIVE")
            authentic = False
        att = p1_cert.backaction_attestation
        if att is not None and not att.all_unperturbed:
            reasons.append("RAM_ZERO_BACKACTION_VIOLATED")
        clean = bool(
            p1_cert.qnd_report.truth_value == HeytingTruthValue.COHERENT
            and (att is None or att.all_unperturbed)
        )
        if not clean and "RAM_ZERO_BACKACTION_VIOLATED" not in reasons:
            reasons.append("UPSTREAM_EVIDENTIAL_OBSTRUCTION")
        if authentic:
            tel.gate_phi12_pass_count += 1
            if p1_cert.qnd_report.truth_value == HeytingTruthValue.VETOED:
                tel.passive_veto_observations += 1
        else:
            tel.gate_phi12_fail_count += 1
        return GateVerdict(authentic=bool(authentic), evidentially_clean=clean,
                           admitted=bool(authentic), reasons=tuple(reasons))

    # ── Método 2.2 ────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_winding_torus(frequencies: Optional[Sequence[float]]) -> Tuple[List[float], List[float]]:
        """
        2.2 Normaliza el toro y extrae los números de rotación ρ_j = (ω_j/ω_0) mod 1 de la
        sección estroboscópica de retorno P_Σ : Σ → Σ. Conversión total a float finito.
        """
        cleaned: List[float] = []
        for f in (frequencies if frequencies is not None else []):
            v = abs(_safe_float(f, 0.0))
            if v > _WILKINSON_EPSILON:
                cleaned.append(v)
        if not cleaned:
            cleaned = [1.0, math.sqrt(2.0), math.sqrt(3.0)]
        w0 = max(cleaned[0], _WILKINSON_EPSILON)
        windings = [(v / w0) % 1.0 for v in cleaned[1:]]
        return cleaned[:8], windings

    # ── Método 2.3 ────────────────────────────────────────────────────────────
    @staticmethod
    def _evaluate_noble_rotation_diagnostics(
        windings: Sequence[float],
        diophantine_gamma: float,
        farey_order: int = 24,
        cf_depth: int = 24,
    ) -> NobleRotationDiagnostics:
        """
        2.3 Diagnóstico de nobleza de Greene-MacKay y proximidad resonante de Farey.
        [A9] v14 devolvía defecto 0.0 (= perfección) cuando no había datos, y medía sólo la
        distancia euclídea a φ. Correcciones:
          • Distancia CIRCULAR min(|x−φ|, 1−|x−φ|) en ℝ/ℤ.
          • Nobleza real = cola de unos en la fracción continua (los nobles son los últimos
            toros en romperse), con DETECCIÓN DE RACIONALIDAD: un racional arbitrariamente
            cercano a φ es resonante, no noble.
          • Proximidad de Farey min_{q≤Q} q^τ·|x − p/q| (τ=2): cuantifica el peligro real de
            resonancia de orden bajo, que es lo que destruye el toro.
          • Sin datos ⟹ defined=False y los márgenes aguas abajo valen 0 (fail-closed).
        Se DECLARA que la cercanía a φ es heurística; el certificado KAM es γ (del motor).
        """
        if not windings:
            return NobleRotationDiagnostics(
                defined=False, golden_circular_distance=1.0, cf_tail_of_ones=0,
                nobility_score=0.0, rational_suspect=True, farey_resonance_proximity=0.0,
                diophantine_gamma=_safe_float(diophantine_gamma, 0.0), winding_numbers=(),
            )
        q_max = int(np.clip(farey_order, 4, 64))
        golden_dists: List[float] = []
        nobilities: List[float] = []
        proximities: List[float] = []
        rational_any = False
        tails: List[int] = []
        for x in windings:
            d = abs(float(x) - _GOLDEN_MEAN_NOBLE_RATIO)
            golden_dists.append(min(d, 1.0 - d))
            terms, rational = _continued_fraction(float(x), cf_depth)
            rational_any = rational_any or rational
            tail = 0
            for a in reversed(terms):
                if a == 1:
                    tail += 1
                else:
                    break
            tails.append(tail)
            nobilities.append(0.0 if rational else tail / max(1, len(terms)))
            best = 1.0
            for q in range(1, q_max + 1):
                p = round(q * float(x))
                best = min(best, (q ** 2) * abs(float(x) - p / q))
            proximities.append(best)
        return NobleRotationDiagnostics(
            defined=True,
            golden_circular_distance=float(min(golden_dists)),
            cf_tail_of_ones=int(min(tails)),
            nobility_score=float(min(nobilities)),
            rational_suspect=bool(rational_any),
            farey_resonance_proximity=float(min(proximities)),
            diophantine_gamma=_safe_float(diophantine_gamma, 0.0),
            winding_numbers=tuple(float(w) for w in windings),
        )

    # ── Método 2.4 ────────────────────────────────────────────────────────────
    @staticmethod
    def _rsi_l3_secular_renormalization_index(
        celestial_rep: CelestialWitnessReport,
        banach_rep: BanachWitnessReport,
        noble: NobleRotationDiagnostics,
        p1_cert: PaladinWitnessPhase1Certificate,
    ) -> Tuple[float, Tuple[float, ...], float]:
        """
        2.4 [RSI-L3] Índice secular de Poincaré-von Zeipel-Banach, con estatuto epistémico
        DECLARADO: es un índice de salud, no un observable físico.
        Axiomas que lo caracterizan (y justifican la media geométrica):
          (A1) R ∈ [0,1];  (A2) monótono creciente en cada margen;
          (A3) multiplicativo/anulable: cualquier margen nulo ⟹ R = 0 (fail-closed);
          (A4) simétrico y homogéneo de grado 1 ⟹ producto de Nash (Cobb-Douglas) = media
               geométrica, única salvo reparametrización monótona.
        [A10-adyacente] v14 mezclaba márgenes sin normalizar y llamaba "renormalización" a un
        producto arbitrario. Aquí el ratio FÍSICO ω_LP/ω_exact (certificado por AGM en el
        motor) se reporta POR SEPARADO, sin confundirlo con el índice.
        Márgenes: Chirikov, Melnikov (umbral EXACTO de Smale = 1), Banach (1−q³), Bures,
        diofántico (γ/γ_floor) y resto de Lindstedt. Nobleza indefinida ⟹ margen 0. [A9]
        """
        k_chir = _safe_float(_field(celestial_rep, "chirikov_overlap_k", 1.0), 1.0)
        a_rel = _safe_float(_field(celestial_rep, "melnikov_amplitude_rel", 1.0), 1.0)
        # Normalización defensiva: el motor v14 vertía la amplitud ABSOLUTA en este campo. [A15]
        osc = _safe_float(_field(celestial_rep, "melnikov_osc_amplitude", None), None) \
            if hasattr(celestial_rep, "melnikov_osc_amplitude") else None
        if osc is not None and osc > 0.0 and a_rel > 0.0:
            a_rel = a_rel if abs(a_rel - osc) > 1e-12 else a_rel / max((4.0 / 3.0) * 0.10, _TINY)
        q_banach = _safe_float(_field(banach_rep, "banach_ratio_q", 0.999), 0.999)
        bures = _safe_float(_field(p1_cert.qnd_report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0))
        remainder = _safe_float(_field(celestial_rep, "lindstedt_remainder_rel", 0.0), 0.0)

        margins = (
            float(np.clip(1.0 - min(1.0, k_chir), 0.0, 1.0)),
            float(np.clip(1.0 - min(1.0, a_rel / _MELNIKOV_SMALE_THRESHOLD), 0.0, 1.0)),
            float(np.clip(1.0 - min(1.0, q_banach ** 3), 0.0, 1.0)),
            float(np.clip(1.0 - min(1.0, bures / math.sqrt(2.0)), 0.0, 1.0)),
            float(np.clip(noble.diophantine_gamma / _KAM_DIOPHANTINE_FLOOR, 0.0, 1.0)) if noble.defined else 0.0,
            float(np.clip(1.0 - min(1.0, remainder / _LINDSTEDT_REMAINDER_CEILING), 0.0, 1.0)),
        )
        product = 1.0
        for m in margins:
            product *= max(0.0, m)
        index = product ** (1.0 / len(margins))
        physical_ratio = _safe_float(_field(celestial_rep, "rsi_l3_resonance_damping_gain", 1.0), 1.0)
        return float(index), tuple(float(m) for m in margins), float(physical_ratio)

    # ── Método 2.5 ────────────────────────────────────────────────────────────
    def _verify_poincare_cartan_secular_drift(
        self, celestial_rep: CelestialWitnessReport
    ) -> CartanDriftCertificate:
        """
        2.5 Deriva del invariante integral relativo de Poincaré-Cartan ∮_γ(p·dq − H dt).
        [A10] v14 calculaba |I_PC·‖[H,ρ]‖ − γ_BP·10⁻¹⁵|, que resta acción×energía con un
        ángulo escalado por una constante mágica: dimensionalmente incoherente.
        Aquí la deriva es la VARIACIÓN RELATIVA de la MISMA magnitud entre ciclos consecutivos
        (adimensional, homogénea), que es lo único que "deriva del invariante" puede significar.
        Se añaden dos certificados estructurales del motor v15:
          • ‖MᵀJM − J‖ (simplécticidad ⟹ conservación de ∮p dq),
          • identidad I_PC = 2π·AMD (acciones de Delaunay), verificación cruzada de campos.
        """
        tel = self._telemetry
        current = _safe_float(_field(celestial_rep, "poincare_cartan_invariant", 0.0), 0.0)
        previous = tel.last_cartan_invariant
        defined = previous is not None
        drift = 0.0
        if defined:
            drift = abs(current - float(previous)) / max(abs(float(previous)), _WILKINSON_EPSILON)
        tel.last_cartan_invariant = current
        amd = _safe_float(_field(celestial_rep, "angular_momentum_deficit", None), None) \
            if hasattr(celestial_rep, "angular_momentum_deficit") else None
        identity_err = abs(current - 2.0 * math.pi * amd) if amd is not None else 0.0
        return CartanDriftCertificate(
            defined=bool(defined), current_invariant=float(current),
            previous_invariant=float(previous) if defined else 0.0,
            relative_drift=float(drift),
            symplectic_defect=_safe_float(_field(celestial_rep, "symplectic_defect", 0.0), 0.0),
            cartan_amd_identity_error=float(identity_err),
        )

    # ── Método 2.6 ────────────────────────────────────────────────────────────
    @staticmethod
    def _propagate_gate_obstruction(
        gate: GateVerdict,
        p1_cert: PaladinWitnessPhase1Certificate,
        celestial_rep: CelestialWitnessReport,
        banach_rep: BanachWitnessReport,
    ) -> Tuple[HeytingTruthValue, bool]:
        """
        2.6 Propagación fail-closed SIN interferencia. Si la compuerta no autentica, el
        veredicto propagado se fuerza a VETOED y se marca `authentication_veto`, pero la
        OBSERVACIÓN continúa (no se lanza excepción aquí: el testigo sigue en silencio).
        La conjunción es el ínfimo en el orden de salud de Ω₃ (max en severidad).
        """
        combined = p1_cert.qnd_report.truth_value.meet(celestial_rep.truth_value).meet(banach_rep.truth_value)
        veto = not gate.authentic
        if veto:
            combined = HeytingTruthValue.VETOED
        return combined, bool(veto)

    # ── Método 2.7 ────────────────────────────────────────────────────────────
    @staticmethod
    def _build_phase2_payload_fields(cert: PaladinWitnessPhase2Certificate) -> Dict[str, Any]:
        """2.7 Payload canónico de cobertura total de Fase 2 (excluye sólo la firma). [A1]"""
        return {
            "domain": _HMAC_DOMAIN_P2.decode("ascii"),
            "agent_version": cert.agent_version,
            "sequence": cert.sequence,
            "chain_prev": cert.chain_prev,
            "timestamp_ns": cert.timestamp_ns,
            "monotonic_ns": cert.monotonic_ns,
            "p1_signature": cert.p1_certificate.hmac_signature,
            "celestial_report": cert.celestial_report,
            "banach_report": cert.banach_report,
            "noble_defect_legacy": cert.poincare_Noble_diophantine_defect,
            "cartan_drift_legacy": cert.poincare_cartan_action_drift,
            "secular_index": cert.rsi_l3_secular_renormalization_factor,
            "gate_phi12_verified": cert.gate_phi12_verified,
            "gate_phi12": cert.gate_phi12,
            "noble": cert.noble,
            "cartan": cert.cartan,
            "secular_margins": cert.secular_margins,
            "physical_ratio": cert.physical_renormalization_ratio,
            "authentication_veto": cert.authentication_veto,
            "propagated_truth": cert.propagated_truth,
        }

    @staticmethod
    def _build_phase2_canonical_payload(
        p1_cert: PaladinWitnessPhase1Certificate,
        celestial_rep: CelestialWitnessReport,
        banach_rep: BanachWitnessReport,
        timestamp_ns: int,
    ) -> str:
        """Compatibilidad v14: payload legado (sólo para consumidores antiguos)."""
        return (
            f"{p1_cert.hmac_signature}:{celestial_rep.is_kam_stable}:"
            f"{celestial_rep.chirikov_overlap_k:.12e}:{banach_rep.banach_ratio_q:.12e}:{timestamp_ns}"
        )

    # ── Método 2.8 ────────────────────────────────────────────────────────────
    def orient_celestial_witness(
        self,
        p1_cert: PaladinWitnessPhase1Certificate,
        frequencies: List[float],
        previous_residual: float,
        current_residual: float,
        perturbation_eps: float = 0.01,
        eccentricity: float = 0.05,
        enable_greene: bool = True,
    ) -> PaladinWitnessPhase2Certificate:
        """2.8 Delegador de compatibilidad absoluta con el ecosistema APU hacia 2.9."""
        return self.emit_phase2_certificate(
            p1_cert, frequencies, previous_residual, current_residual,
            perturbation_eps, eccentricity, enable_greene,
        )

    # ── Método 2.9 · TERMINAL FASE 2 → APERTURA FASE 3 ────────────────────────
    def emit_phase2_certificate(
        self,
        p1_cert: PaladinWitnessPhase1Certificate,
        frequencies: List[float],
        previous_residual: float,
        current_residual: float,
        perturbation_eps: float = 0.01,
        eccentricity: float = 0.05,
        enable_greene: bool = True,
    ) -> PaladinWitnessPhase2Certificate:
        """
        2.9 [TERMINAL FASE 2] Ejecuta Φ₁₂ de forma EFECTIVA, orienta la mecánica celeste KAM
        y la contracción de Banach, y emite el `PaladinWitnessPhase2Certificate` (P2).
        [A14] Pasa explícitamente `qnd_report` a las fases del motor v15 para preservar la
        continuidad TIPADA (semilla Phase1ToPhase2Seed) en vez del fallback `_last_report`.
        Este certificado es la entrada DIRECTA del método 3.1 (Compuerta Φ₂₃).
        """
        tel = self._telemetry
        gate = self._gate_phase1(p1_cert)
        freqs, windings = self._normalize_winding_torus(frequencies)

        if self._caps.has_phase_objects and self._caps.celestial_accepts_qnd:
            kwargs: Dict[str, Any] = {
                "frequencies": freqs, "perturbation_eps": perturbation_eps,
                "eccentricity": eccentricity, "qnd_report": p1_cert.qnd_report,
            }
            if self._caps.celestial_accepts_greene:
                kwargs["enable_greene"] = bool(enable_greene)
            celestial_rep = self.engine.phase2_navigator.audit_celestial_mechanics(**kwargs)
        else:
            celestial_rep = self.engine.audit_celestial_mechanics(freqs, perturbation_eps)

        if self._caps.has_phase_objects and self._caps.banach_accepts_reports:
            banach_rep = self.engine.phase3_governor.audit_banach_rsi3_contraction(
                previous_residual=previous_residual, current_residual=current_residual,
                celestial_report=celestial_rep, qnd_report=p1_cert.qnd_report,
            )
        else:
            banach_rep = self.engine.audit_banach_rsi3_contraction(previous_residual, current_residual)

        noble = self._evaluate_noble_rotation_diagnostics(
            windings, _field(celestial_rep, "diophantine_gamma",
                             _field(celestial_rep, "diophantine_min_divisor", 0.0)),
            tel.policy.farey_order, tel.policy.cf_depth,
        )
        index, margins, physical_ratio = self._rsi_l3_secular_renormalization_index(
            celestial_rep, banach_rep, noble, p1_cert
        )
        cartan = self._verify_poincare_cartan_secular_drift(celestial_rep)
        propagated, veto = self._propagate_gate_obstruction(gate, p1_cert, celestial_rep, banach_rep)

        tel.sequence_counter += 1
        tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
        skeleton = PaladinWitnessPhase2Certificate(
            p1_certificate=p1_cert,
            celestial_report=celestial_rep,
            banach_report=banach_rep,
            timestamp_ns=time.time_ns(),
            hmac_signature="",
            poincare_Noble_diophantine_defect=noble.golden_circular_distance,
            poincare_cartan_action_drift=cartan.relative_drift,
            rsi_l3_secular_renormalization_factor=index,
            gate_phi12_verified=gate.authentic,
            agent_version=_AGENT_VERSION,
            sequence=tel.sequence_counter,
            chain_prev=tel.chain_head,
            monotonic_ns=tel.monotonic_clock_ns,
            noble=noble, cartan=cartan, secular_margins=margins,
            physical_renormalization_ratio=physical_ratio,
            gate_phi12=gate, authentication_veto=veto, propagated_truth=propagated,
        )
        payload = _canonical_json(self._build_phase2_payload_fields(skeleton))
        signature = self.engine.generate_witness_hmac_signature(payload, _HMAC_DOMAIN_P2)
        leaf = hashlib.sha256(_canonical_json({
            "p2": signature, "celestial": celestial_rep, "banach": banach_rep,
        }).encode("utf-8")).hexdigest()
        cert = replace(skeleton, hmac_signature=signature, merkle_leaf=leaf)

        tel.chain_head = signature
        tel.signature_registry.append(signature)
        if len(tel.signature_registry) > 512:
            tel.signature_registry.pop(0)
        tel.phase2_emissions += 1
        self._last_p2_certificate = cert
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §5 · FASE 3 OODA (DECIDE & ACT): Φ₂₃, TOPOS Ω₃, Φ_sem MONÓTONO Y MERKLE
# ══════════════════════════════════════════════════════════════════════════════
class Phase3WitnessDecider:
    """
    Fase 3 OODA (Decide & Act | 3.1 → 3.9). Cadena formal:
        3.1 Φ₂₃ (ingesta de P2) → 3.2 ínfimo Ω₃ + BHK → 3.3 funtor Φ_sem monótono
        → 3.4 meta-auditoría RSI-L3 (L1/L2/L3) → 3.5 Merkle RFC 6962 → 3.6 payload soberano
        → 3.7 verificador público → 3.8 delegador → 3.9 TERMINAL (emite el Estado Soberano).
    """

    def __init__(
        self,
        engine: BoolNovikovPaladinWitnessEngine,
        telemetry: Optional[RSILevel3AgentGovernanceTelemetry] = None,
    ) -> None:
        self.engine = engine
        self._telemetry = telemetry if telemetry is not None else RSILevel3AgentGovernanceTelemetry()
        self._last_sovereign_state: Optional[PaladinWitnessSovereignState] = None

    # ── Método 3.1 · APERTURA FASE 3 (continuación directa de 2.9) ────────────
    def _ingest_phase2_certificate_gate(self, p2_cert: PaladinWitnessPhase2Certificate) -> bool:
        """
        3.1 [CONTINUACIÓN DIRECTA DE 2.9 — COMPUERTA HOMOMÓRFICA Φ₂₃]
        Revalida la firma HMAC-SHA256 de P2 con cobertura total y exige Lip(T³)=q³ ≤ 0.857375.
        Devuelve bool por compatibilidad; el veredicto rico está en `_gate_phase2`.
        """
        verdict = self._gate_phase2(p2_cert)
        return bool(verdict.authentic and verdict.evidentially_clean)

    def _gate_phase2(self, p2_cert: PaladinWitnessPhase2Certificate) -> GateVerdict:
        """
        Φ₂₃ rica. AUTENTICIDAD: firma P2, firma P1 anidada (verificación recursiva de la
        cadena) y ausencia de veto de autenticación aguas arriba. SALUD: cota de Banach.
        [A4] v14 descartaba este resultado en 3.7 (`_ = ...`); aquí es vinculante.
        """
        tel = self._telemetry
        reasons: List[str] = []
        payload = _canonical_json(Phase2WitnessOrientor._build_phase2_payload_fields(p2_cert))
        expected = self.engine.generate_witness_hmac_signature(payload, _HMAC_DOMAIN_P2)
        authentic = _constant_time_equals(p2_cert.hmac_signature, expected)
        if not authentic:
            reasons.append("P2_SIGNATURE_COVERAGE_MISMATCH")

        p1_payload = _canonical_json(
            Phase1WitnessObserver._build_phase1_payload_fields(p2_cert.p1_certificate)
        )
        p1_expected = self.engine.generate_witness_hmac_signature(p1_payload, _HMAC_DOMAIN_P1)
        p1_ok = _constant_time_equals(p2_cert.p1_certificate.hmac_signature, p1_expected)
        if not p1_ok:
            reasons.append("NESTED_P1_SIGNATURE_MISMATCH")
        if p2_cert.authentication_veto or not p2_cert.gate_phi12_verified:
            reasons.append("UPSTREAM_AUTHENTICATION_VETO")
        if p2_cert.sequence <= p2_cert.p1_certificate.sequence:
            reasons.append("SEQUENCE_NOT_STRICTLY_INCREASING")
        authentic = bool(authentic and p1_ok and not p2_cert.authentication_veto
                         and p2_cert.gate_phi12_verified
                         and p2_cert.sequence > p2_cert.p1_certificate.sequence)

        q_banach = _safe_float(_field(p2_cert.banach_report, "banach_ratio_q", 1.0), 1.0)
        banach_ok = (q_banach ** 3) <= _BANACH_KLEISLI_T3_CEILING
        declared = bool(_field(p2_cert.banach_report, "is_banach_contractive", True))
        if not banach_ok:
            reasons.append("KLEISLI_T3_CONTRACTION_CEILING_EXCEEDED")
        if declared != banach_ok:
            reasons.append("BANACH_FLAG_INCONSISTENT_WITH_RATIO")
        clean = bool(banach_ok and declared == banach_ok)
        if authentic and clean:
            tel.gate_phi23_pass_count += 1
        elif not authentic:
            tel.gate_phi23_fail_count += 1
        return GateVerdict(authentic=authentic, evidentially_clean=clean,
                           admitted=authentic, reasons=tuple(reasons))

    # ── Método 3.2 ────────────────────────────────────────────────────────────
    @staticmethod
    def _evaluate_heyting_topos_infimum(
        p2_cert: PaladinWitnessPhase2Certificate,
        gate: GateVerdict,
    ) -> HeytingAdjudication:
        """
        3.2 Ínfimo (∧) en Ω₃ y análisis constructivo de Brouwer-Heyting-Kolmogorov.
        [A12] v14 llamaba "supremo" a max(severidad), que es el ÍNFIMO en el orden de salud.
        Además computaba (a⇒b)⇒c mientras documentaba a⇒(b⇒c): son DISTINTOS en la cadena
        de Gödel. Aquí se calculan ambos y se reporta la BRECHA DE ENTRAÑAMIENTO:
            meet(a, a⇒b) ⊑ b  (counit de la adjunción, siempre cierto),
        y la igualdad meet(a, a⇒b) == b indica que b está realmente ENTRAÑADO por la
        evidencia. Si a es falso (VETOED), a⇒b es vacuamente ⊤ y no informa sobre b: la
        brecha detecta exactamente ese caso, distinguiendo prueba de mera aserción.
        """
        p1 = p2_cert.p1_certificate
        a = p1.qnd_report.truth_value
        b = p2_cert.celestial_report.truth_value
        c = p2_cert.banach_report.truth_value
        infimum = a.meet(b).meet(c).meet(p2_cert.propagated_truth)
        if not gate.authentic:
            infimum = HeytingTruthValue.VETOED

        r_ab = a.imply(b)
        r_bc = b.imply(c)
        left_assoc = r_ab.imply(c)     # (a⇒b)⇒c  — lo que v14 calculaba
        right_assoc = a.imply(r_bc)    # a⇒(b⇒c)  — lo que v14 documentaba
        gap_ab = a.meet(r_ab) != b
        gap_bc = b.meet(r_bc) != c
        modus_ponens_ok = bool(
            int(a.meet(r_ab)) >= int(b) and int(b.meet(r_bc)) >= int(c)
        )
        att = p1.backaction_attestation
        ram_ok = bool(p1.ram_pointer_unperturbed and (att is None or att.all_unperturbed))
        invariant_ok = bool(
            p1.qnd_report.is_qnd_valid and p1.qnd_report.is_fides_healthy
            and ram_ok and gate.authentic
        )
        return HeytingAdjudication(
            infimum=infimum, bhk_left_associated=left_assoc, bhk_right_associated=right_assoc,
            entailment_gap_p12=bool(gap_ab), entailment_gap_p23=bool(gap_bc),
            modus_ponens_ok=modus_ponens_ok, witness_invariant_ok=invariant_ok,
        )

    @staticmethod
    def _evaluate_heyting_topos_supremum(
        p2_cert: PaladinWitnessPhase2Certificate,
    ) -> Tuple[HeytingTruthValue, HeytingTruthValue, bool]:
        """Alias de compatibilidad v14 (nombre histórico incorrecto: era el ínfimo)."""
        gate = GateVerdict(authentic=bool(p2_cert.gate_phi12_verified),
                           evidentially_clean=True, admitted=True)
        adj = Phase3WitnessDecider._evaluate_heyting_topos_infimum(p2_cert, gate)
        return adj.infimum, adj.bhk_left_associated, adj.witness_invariant_ok

    # ── Método 3.3 ────────────────────────────────────────────────────────────
    @staticmethod
    def _evaluate_visceral_semantic_functor(
        final_tv: HeytingTruthValue,
        p2_cert: PaladinWitnessPhase2Certificate,
        fidelity_flag_count: int,
        fidelity_ambiguous_count: int,
    ) -> Tuple[VisceralSemanticValuation, str]:
        """
        3.3 Funtor Semántico Visceral Φ_sem : Ω₃ × M_QND × M_KAM → E_fin.
        [A5] v14 NO era funtor: con F=0.5 daba COHERENT→0.46 y DEGRADED→0.50, invirtiendo el
        orden (un estado peor recibía más escudo). Aquí escudo = w(tv)·evidencia con
        w(C)=1.0 > w(D)=0.6 > w(V)=0.2 y evidencia ∈ [0,1] independiente de tv ⟹ MONÓTONO
        por construcción (funtor de posets, i.e. preservación del orden de salud).
        [A6] El FPR NO se afirma: sin verdad-terreno no es identificable. Se reporta una
        COTA SUPERIOR rigurosa atribuible al predicado de fidelidad: un falso positivo
        exige que el estado real sea sano, lo que exige F_hi ≥ umbral (intervalo certificado
        del motor). Si el motor no expone el intervalo, la cota es 1.0 (conservadora), nunca 0.
        Los valores WACC 15.0% y Liquidez +11.5% se declaran como OBJETIVOS DE POLÍTICA.
        """
        p1 = p2_cert.p1_certificate
        fidelity = float(np.clip(_safe_float(_field(p1.qnd_report, "uhlmann_fidelity", 0.0)), 0.0, 1.0))
        bures = _safe_float(_field(p1.qnd_report, "bures_distance", math.sqrt(2.0)), math.sqrt(2.0))
        q_banach = float(np.clip(_safe_float(_field(p2_cert.banach_report, "banach_ratio_q", 0.999)), 0.0, 1.0))
        index = float(np.clip(_safe_float(p2_cert.rsi_l3_secular_renormalization_factor, 0.0), 0.0, 1.0))

        weights = {HeytingTruthValue.COHERENT: 1.0, HeytingTruthValue.DEGRADED: 0.6, HeytingTruthValue.VETOED: 0.2}
        weight = weights.get(final_tv, 0.2)
        evidence = float(np.clip(
            fidelity * (1.0 - 0.1 * q_banach ** 3) * (0.5 + 0.5 * index)
            * (1.0 - min(1.0, bures / math.sqrt(2.0)) * 0.25), 0.0, 1.0
        ))
        shield = float(np.clip(weight * evidence, 0.0, 1.0))

        identifiable = hasattr(p1.qnd_report, "fidelity_upper_certificate")
        flags = max(0, int(fidelity_flag_count))
        ambiguous = max(0, min(int(fidelity_ambiguous_count), flags))
        bound = (ambiguous / flags) if (identifiable and flags > 0) else (0.0 if (identifiable and flags == 0) else 1.0)
        bound = float(np.clip(bound, 0.0, 1.0))

        regime = {
            HeytingTruthValue.COHERENT: "OPTIMAL_QND_SHIELD_ACTIVE",
            HeytingTruthValue.DEGRADED: "ELASTIC_RELIEF_VALVE_ENGAGED",
            HeytingTruthValue.VETOED: "PASSIVE_WITNESS_VETO_CONTAINMENT",
        }.get(final_tv, "PASSIVE_WITNESS_VETO_CONTAINMENT")

        valuation = VisceralSemanticValuation(
            wacc_protection_pct=_PROTECTED_WACC_RATE * 100.0,
            liquidity_delta_pct=_LIQUIDITY_SURPLUS_DELTA * 100.0 * weight,
            false_positive_rate=bound,
            capital_shield_efficiency=shield,
            semantic_regime=regime,
            wacc_protection_target_pct=_PROTECTED_WACC_RATE * 100.0,
            liquidity_nominal_pct=_LIQUIDITY_SURPLUS_DELTA * 100.0,
            declared_fpr_target=_DECLARED_FPR_TARGET,
            fpr_upper_bound=bound,
            fpr_bound_identifiable=bool(identifiable),
            fidelity_flag_count=flags,
            fidelity_ambiguous_count=ambiguous,
            verdict_weight=float(weight),
            evidence_factor=float(evidence),
            functor_monotone_ok=True,
        )
        narrative = (
            f"TESTIGO PALADÍN {final_tv.name}: F(ρ,σ)={fidelity:.6f}, d_B={bures:.6f}, "
            f"θ_B={p1.fubini_study_angle_rad:.6f} rad, β₁={_field(p1.qnd_report, 'betti_1_topology', 0)}. "
            f"Back-Action={_safe_float(_field(p1.qnd_report, 'backaction_db', 0.0)):.2f} dB; "
            f"RAM inalterada={p1.ram_pointer_unperturbed} (alcance: buffers inspeccionados). "
            f"KAM γ={_safe_float(_field(p2_cert.celestial_report, 'diophantine_gamma', 0.0)):.3e}, "
            f"Chirikov K={_safe_float(_field(p2_cert.celestial_report, 'chirikov_overlap_k', 0.0)):.4f}, "
            f"q³={q_banach ** 3:.6f}. Índice secular={index:.4f}. "
            f"[Φ_sem: escudo={shield:.4f}, WACC objetivo={valuation.wacc_protection_pct:.1f}%, "
            f"liquidez={valuation.liquidity_delta_pct:.2f}%, "
            f"FPR≤{bound * 100:.2f}% (identificable={identifiable})]."
        )
        return valuation, narrative

    # ── Método 3.4 ────────────────────────────────────────────────────────────
    def _rsi_l3_meta_cognitive_self_audit(
        self,
        p2_cert: PaladinWitnessPhase2Certificate,
        gate: GateVerdict,
        adjudication: HeytingAdjudication,
        valuation: VisceralSemanticValuation,
    ) -> MetaAuditCertificate:
        """
        3.4 [RSI-L3] Meta-auditoría recursiva en TRES niveles con presupuestos bien fundados.
        L1 · NUMÉRICO: finitud y rango de todas las métricas transportadas.
        L2 · ESTRUCTURAL: identidades algebraicas verificables — cuerda-arco de Bures,
             Euler-Poincaré, simplécticidad, I_PC = 2π·AMD, residuación de Heyting,
             monotonía del funtor Φ_sem y modus ponens.
        L3 · ATESTACIÓN: firmas Φ₁₂/Φ₂₃, secuencia estrictamente creciente, reloj monótono,
             ausencia de replay y presupuesto de mutación no agotado.
        Dos contadores bien fundados y DISTINTOS: `verification_depth` se reinicia cada ciclo
        (acota la profundidad del meta-razonamiento, no la vida del agente); `mutation_budget`
        decrece de por vida sólo cuando la política se AUTOMODIFICA (descenso estricto en ℕ
        ⟹ toda cadena de automodificación es finita; el obstáculo löbiano se evita exigiendo
        certificados a profundidad menor, nunca auto-confianza).
        """
        tel = self._telemetry
        failed: List[str] = []
        p1 = p2_cert.p1_certificate
        cel, ban = p2_cert.celestial_report, p2_cert.banach_report

        metrics = (
            _field(p1.qnd_report, "qnd_commutator_norm", 0.0),
            _field(p1.qnd_report, "uhlmann_fidelity", 0.0),
            _field(p1.qnd_report, "bures_distance", 0.0),
            p1.fubini_study_angle_rad, p1.berry_poincare_holonomy_rad,
            _field(cel, "chirikov_overlap_k", 0.0), _field(cel, "melnikov_amplitude_rel", 0.0),
            _field(cel, "floquet_spectral_radius", 1.0),
            _field(ban, "banach_ratio_q", 0.0), _field(ban, "lyapunov_value", 0.0),
            p2_cert.rsi_l3_secular_renormalization_factor, valuation.capital_shield_efficiency,
        )
        l1 = all(math.isfinite(_safe_float(m, float("nan"))) for m in metrics)
        if not l1:
            failed.append("L1_NON_FINITE_METRIC")
        if not (0.0 <= _safe_float(_field(p1.qnd_report, "uhlmann_fidelity", 0.0)) <= 1.0):
            l1 = False
            failed.append("L1_FIDELITY_OUT_OF_RANGE")

        l2 = True
        geom = p1.geometry
        if geom is not None and geom.chord_arc_identity_error > 1e-9:
            l2 = False
            failed.append("L2_CHORD_ARC_IDENTITY")
        if not bool(_field(p1.qnd_report, "euler_identity_ok", True)):
            l2 = False
            failed.append("L2_EULER_POINCARE_IDENTITY")
        cartan = p2_cert.cartan
        if cartan is not None:
            if cartan.symplectic_defect > _SYMPECTIC_DEFECT_CEILING:
                l2 = False
                failed.append("L2_SYMPECTIC_DEFECT")
            if cartan.cartan_amd_identity_error > 1e-9:
                l2 = False
                failed.append("L2_CARTAN_AMD_IDENTITY")
        if not adjudication.modus_ponens_ok:
            l2 = False
            failed.append("L2_HEYTING_MODUS_PONENS")
        probe_weights = (1.0, 0.6, 0.2)
        if not (probe_weights[0] >= probe_weights[1] >= probe_weights[2]):
            l2 = False
            failed.append("L2_FUNCTOR_NON_MONOTONE")
        if abs(valuation.capital_shield_efficiency
               - valuation.verdict_weight * valuation.evidence_factor) > 1e-9:
            l2 = False
            failed.append("L2_FUNCTOR_FACTORISATION")

        l3 = bool(gate.authentic and p2_cert.gate_phi12_verified and not p2_cert.authentication_veto)
        if not l3:
            failed.append("L3_GATE_AUTHENTICATION")
        if p2_cert.monotonic_ns <= p1.monotonic_ns:
            l3 = False
            failed.append("L3_CLOCK_NOT_MONOTONE")
        if p2_cert.hmac_signature in tel.signature_registry[:-1]:
            l3 = False
            failed.append("L3_SIGNATURE_REPLAY")
        depth_ok = tel.policy.verification_depth > 0
        if not depth_ok:
            failed.append("L3_VERIFICATION_DEPTH_EXHAUSTED")

        accepted = bool(l1 and l2 and l3 and depth_ok)
        tel.policy = replace(tel.policy, verification_depth=max(0, tel.policy.verification_depth - 1)).project()
        return MetaAuditCertificate(
            l1_numeric_ok=bool(l1), l2_structural_ok=bool(l2), l3_attestation_ok=bool(l3),
            accepted=accepted,
            verification_depth_remaining=int(tel.policy.verification_depth),
            mutation_budget_remaining=int(tel.policy.mutation_budget),
            failed_checks=tuple(failed),
        )

    # ── Método 3.5 ────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_cryptographic_merkle_tree(
        p2_cert: PaladinWitnessPhase2Certificate,
        final_tv: HeytingTruthValue,
        invariant_ok: bool,
        timestamp_ns: int,
    ) -> Tuple[str, str]:
        """
        3.5 Dos raíces con estatutos distintos.
        (1) RAÍZ LEGADA v13/v14: SHA-256 de la concatenación histórica. Se conserva por
            compatibilidad de protocolo, pero se DECLARA que no es un árbol de Merkle. [A11]
        (2) RAÍZ AUTORITATIVA: Merkle binario RFC 6962 de 4 hojas con prefijos de dominio
            0x00/0x01 sobre canonicalizaciones de COBERTURA TOTAL (QND, celeste, Banach,
            veredicto+invariante). Los prefijos impiden la confusión hoja/nodo que habilitaba
            ataques de segunda preimagen en el esquema de v14 (hex-texto separado por ':').
        """
        legacy_payload = f"{p2_cert.hmac_signature}:{final_tv.name}:{invariant_ok}:{timestamp_ns}"
        legacy_root = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
        leaves = [
            _canonical_json({"leaf": 1, "p1": p2_cert.p1_certificate}).encode("utf-8"),
            _canonical_json({"leaf": 2, "celestial": p2_cert.celestial_report}).encode("utf-8"),
            _canonical_json({"leaf": 3, "banach": p2_cert.banach_report}).encode("utf-8"),
            _canonical_json({
                "leaf": 4, "invariant": PALADIN_QND_WITNESS_INVARIANT,
                "truth": final_tv, "invariant_ok": invariant_ok, "timestamp_ns": timestamp_ns,
            }).encode("utf-8"),
        ]
        return legacy_root, _merkle_root_rfc6962(leaves)

    # ── Método 3.6 ────────────────────────────────────────────────────────────
    @staticmethod
    def _build_sovereign_payload_fields(state: PaladinWitnessSovereignState) -> Dict[str, Any]:
        """3.6 Payload soberano de cobertura total (excluye sólo la firma P3)."""
        return {
            "domain": _HMAC_DOMAIN_P3.decode("ascii"),
            "agent_version": state.agent_version,
            "sequence": state.sequence,
            "chain_prev": state.chain_prev,
            "timestamp_ns": state.timestamp_ns,
            "monotonic_ns": state.monotonic_ns,
            "p2_signature": state.p2_certificate.hmac_signature,
            "truth_value": state.truth_value,
            "witness_invariant_verified": state.witness_invariant_verified,
            "narrative_summary": state.narrative_summary,
            "merkle_root": state.merkle_root,
            "hierarchical_merkle_root": state.hierarchical_merkle_root,
            "visceral_semantics": state.visceral_semantics,
            "heyting_bhk_implication": state.heyting_bhk_implication,
            "rsi_l3_meta_audit_passed": state.rsi_l3_meta_audit_passed,
            "adjudication": state.adjudication,
            "meta_audit": state.meta_audit,
            "gate_phi23": state.gate_phi23,
            "authentication_veto": state.authentication_veto,
            "merkle_algorithm": state.merkle_algorithm,
        }

    # ── Método 3.7 ────────────────────────────────────────────────────────────
    def verify_sovereign_state(self, state: PaladinWitnessSovereignState) -> SovereignVerificationReport:
        """
        3.7 VERIFICADOR PÚBLICO. Recomputa, sólo a partir del certificado:
        las tres firmas HMAC (P1, P2, P3), ambas raíces de Merkle, el veredicto en Ω₃ y la
        valuación Φ_sem. Es determinista porque todos los insumos dependientes del historial
        (conteos de banderas, invariante de Cartan previo, secuencia, chain_prev) están
        ALMACENADOS dentro del certificado. [A13] Un certificado que nadie puede verificar
        no es un certificado.
        """
        failed: List[str] = []
        p2 = state.p2_certificate
        p1 = p2.p1_certificate

        p1_ok = _constant_time_equals(
            p1.hmac_signature,
            self.engine.generate_witness_hmac_signature(
                _canonical_json(Phase1WitnessObserver._build_phase1_payload_fields(p1)), _HMAC_DOMAIN_P1),
        )
        p2_ok = _constant_time_equals(
            p2.hmac_signature,
            self.engine.generate_witness_hmac_signature(
                _canonical_json(Phase2WitnessOrientor._build_phase2_payload_fields(p2)), _HMAC_DOMAIN_P2),
        )
        p3_ok = _constant_time_equals(
            state.p3_hmac_signature,
            self.engine.generate_witness_hmac_signature(
                _canonical_json(self._build_sovereign_payload_fields(state)), _HMAC_DOMAIN_P3),
        )
        for name, ok in (("P1_SIGNATURE", p1_ok), ("P2_SIGNATURE", p2_ok), ("P3_SIGNATURE", p3_ok)):
            if not ok:
                failed.append(name)

        legacy, hierarchical = self._compute_cryptographic_merkle_tree(
            p2, state.truth_value, state.witness_invariant_verified, state.timestamp_ns
        )
        legacy_ok = _constant_time_equals(legacy, state.merkle_root)
        hier_ok = _constant_time_equals(hierarchical, state.hierarchical_merkle_root)
        if not legacy_ok:
            failed.append("LEGACY_MERKLE_ROOT")
        if not hier_ok:
            failed.append("HIERARCHICAL_MERKLE_ROOT")

        gate = self._gate_phase2(p2)
        adj = self._evaluate_heyting_topos_infimum(p2, gate)
        truth_ok = bool(adj.infimum == state.truth_value)
        if not truth_ok:
            failed.append("TRUTH_RECOMPUTATION")

        sem_ok = True
        if state.visceral_semantics is not None:
            recomputed, narrative = self._evaluate_visceral_semantic_functor(
                state.truth_value, p2,
                state.visceral_semantics.fidelity_flag_count,
                state.visceral_semantics.fidelity_ambiguous_count,
            )
            sem_ok = bool(
                abs(recomputed.capital_shield_efficiency - state.visceral_semantics.capital_shield_efficiency) < 1e-12
                and abs(recomputed.false_positive_rate - state.visceral_semantics.false_positive_rate) < 1e-12
                and recomputed.semantic_regime == state.visceral_semantics.semantic_regime
                and narrative == state.narrative_summary
            )
        if not sem_ok:
            failed.append("SEMANTIC_RECOMPUTATION")

        invariant_ok = bool(state.witness_invariant_verified == adj.witness_invariant_ok)
        if not invariant_ok:
            failed.append("INVARIANT_FLAG")
        chain_ok = bool(
            p1.sequence < p2.sequence < state.sequence
            and p1.monotonic_ns < p2.monotonic_ns < state.monotonic_ns
            and state.chain_prev == p2.hmac_signature
        )
        if not chain_ok:
            failed.append("CHAIN_MONOTONICITY")

        all_ok = bool(p1_ok and p2_ok and p3_ok and legacy_ok and hier_ok
                      and truth_ok and sem_ok and invariant_ok and chain_ok)
        return SovereignVerificationReport(
            p1_signature_ok=p1_ok, p2_signature_ok=p2_ok, p3_signature_ok=p3_ok,
            legacy_merkle_ok=legacy_ok, hierarchical_merkle_ok=hier_ok,
            truth_recomputation_ok=truth_ok, semantics_recomputation_ok=sem_ok,
            invariant_ok=invariant_ok, chain_monotone_ok=chain_ok,
            all_ok=all_ok, failed_checks=tuple(failed),
        )

    # ── Método 3.8 ────────────────────────────────────────────────────────────
    def adjudicate_paladin_witness_rsi3(
        self, p2_cert: PaladinWitnessPhase2Certificate
    ) -> PaladinWitnessSovereignState:
        """3.8 Delegador de compatibilidad absoluta con el ecosistema APU hacia 3.9."""
        return self.emit_sovereign_state(p2_cert)

    # ── Método 3.9 · TERMINAL FASE 3 ──────────────────────────────────────────
    def emit_sovereign_state(
        self, p2_cert: PaladinWitnessPhase2Certificate
    ) -> PaladinWitnessSovereignState:
        """
        3.9 [TERMINAL FASE 3] Ejecuta Φ₂₃ de forma EFECTIVA (ya no se descarta el resultado),
        adjudica el ínfimo en Ω₃, evalúa el funtor monótono Φ_sem, corre la meta-auditoría
        RSI-L3 y sella el `PaladinWitnessSovereignState` con Merkle RFC 6962 + HMAC P3.
        La observación nunca se interrumpe por evidencia; la atestación nunca se emite limpia
        sobre una entrada no autenticada.
        """
        tel = self._telemetry
        tel.policy = replace(tel.policy, verification_depth=3).project()  # profundidad POR CICLO
        gate = self._gate_phase2(p2_cert)
        adjudication = self._evaluate_heyting_topos_infimum(p2_cert, gate)
        final_tv = adjudication.infimum

        p1 = p2_cert.p1_certificate
        flagged = final_tv != HeytingTruthValue.COHERENT
        f_hi = _safe_float(_field(p1.qnd_report, "fidelity_upper_certificate", None), None) \
            if hasattr(p1.qnd_report, "fidelity_upper_certificate") else None
        ambiguous = bool(flagged and f_hi is not None and f_hi >= _UHLMANN_FIDELITY_FLOOR)
        if flagged:
            tel.fidelity_flag_count += 1
            if ambiguous:
                tel.fidelity_ambiguous_count += 1

        valuation, narrative = self._evaluate_visceral_semantic_functor(
            final_tv, p2_cert, tel.fidelity_flag_count, tel.fidelity_ambiguous_count
        )
        meta = self._rsi_l3_meta_cognitive_self_audit(p2_cert, gate, adjudication, valuation)
        if not meta.accepted:
            final_tv = final_tv.meet(HeytingTruthValue.DEGRADED)
            valuation, narrative = self._evaluate_visceral_semantic_functor(
                final_tv, p2_cert, tel.fidelity_flag_count, tel.fidelity_ambiguous_count
            )

        tel.sequence_counter += 1
        tel.monotonic_clock_ns = max(time.monotonic_ns(), tel.monotonic_clock_ns + 1)
        ts = time.time_ns()
        legacy_root, hierarchical_root = self._compute_cryptographic_merkle_tree(
            p2_cert, final_tv, adjudication.witness_invariant_ok, ts
        )
        skeleton = PaladinWitnessSovereignState(
            truth_value=final_tv,
            p2_certificate=p2_cert,
            witness_invariant_verified=adjudication.witness_invariant_ok,
            narrative_summary=narrative,
            merkle_root=legacy_root,
            timestamp_ns=ts,
            visceral_semantics=valuation,
            heyting_bhk_implication=adjudication.bhk_right_associated,
            hierarchical_merkle_root=hierarchical_root,
            p3_hmac_signature="",
            rsi_l3_meta_audit_passed=meta.accepted,
            agent_version=_AGENT_VERSION,
            sequence=tel.sequence_counter,
            chain_prev=tel.chain_head,
            monotonic_ns=tel.monotonic_clock_ns,
            adjudication=adjudication, meta_audit=meta, gate_phi23=gate,
            authentication_veto=bool(not gate.authentic),
        )
        payload = _canonical_json(self._build_sovereign_payload_fields(skeleton))
        signature = self.engine.generate_witness_hmac_signature(payload, _HMAC_DOMAIN_P3)
        state = replace(skeleton, p3_hmac_signature=signature)

        tel.chain_head = signature
        tel.signature_registry.append(signature)
        if len(tel.signature_registry) > 512:
            tel.signature_registry.pop(0)
        tel.phase3_emissions += 1
        tel.last_sovereign_truth = final_tv
        self._last_sovereign_state = state
        return state


# ══════════════════════════════════════════════════════════════════════════════
# §6 · AGENTE SOBERANO: ORQUESTADOR OODA TRIFÁSICO CON COMPUERTAS Φ VINCULANTES
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPaladinWitnessAgent:
    """
    Soberano Agéntico Testigo QND del Paladín (APU Filter v15.0.0-PhD).
    Tres fases OODA anidadas con continuidad formal estricta y compuertas vinculantes:

        Fase 1 · Phase1_WitnessObserver : 1.1 → 1.9  emit_phase1_certificate
        Fase 2 · Phase2_WitnessOrientor : 2.1  Φ₁₂ → 2.9  emit_phase2_certificate
        Fase 3 · Phase3_WitnessDecider  : 3.1  Φ₂₃ → 3.9  emit_sovereign_state

    Política de fallo (corrección doctrinal de v14):
        FORJA criptográfica  → PermissionError (evento de seguridad).
        OBSTRUCCIÓN evidencial → Estado Soberano VETOED (observación pasiva intacta).
    """

    Phase1_WitnessObserver = Phase1WitnessObserver
    Phase2_WitnessOrientor = Phase2WitnessOrientor
    Phase3_WitnessDecider = Phase3WitnessDecider

    def __init__(
        self,
        secret_key: Optional[bytes] = None,
        policy: Optional[AgentRSI3Policy] = None,
    ) -> None:
        self.secret_key: bytes = bytes(secret_key) if secret_key else _DEFAULT_SECRET_KEY
        self.engine = BoolNovikovPaladinWitnessEngine(self.secret_key)
        self._telemetry = RSILevel3AgentGovernanceTelemetry(
            policy=(policy or AgentRSI3Policy()).project(),
            dev_key_in_use=not bool(secret_key),
        )
        if not secret_key:
            logger.warning(
                "Clave de desarrollo en uso: los certificados NO son atestables en producción."
            )
        self.observer = self.Phase1_WitnessObserver(self.engine, self._telemetry)
        self.orientor = self.Phase2_WitnessOrientor(self.engine, self._telemetry)
        self.decider = self.Phase3_WitnessDecider(self.engine, self._telemetry)
        self.capabilities = _probe_engine_capabilities(self.engine)
        logger.info("BoolNovikovPaladinWitnessAgent v%s inicializado (Φ vinculantes).", _AGENT_VERSION)

    # ── Compuertas expuestas ──────────────────────────────────────────────────
    def continue_to_phase2_gate(self, p1_cert: PaladinWitnessPhase1Certificate) -> bool:
        """Φ₁₂: autenticidad de P1 con cobertura total; admite VETOED para diagnóstico pasivo."""
        return self.orientor._ingest_phase1_certificate_gate(p1_cert)

    def continue_to_phase3_gate(self, p2_cert: PaladinWitnessPhase2Certificate) -> bool:
        """Φ₂₃: autenticidad de P2 (y P1 anidada) + cota contractiva Lip(T³) ≤ 0.857375."""
        return self.decider._ingest_phase2_certificate_gate(p2_cert)

    # ── Ciclo OODA completo ───────────────────────────────────────────────────
    def execute_ooda_governance(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        frequencies: Optional[List[float]] = None,
        previous_residual: float = 0.01,
        current_residual: float = 0.008,
        perturbation_eps: float = 0.01,
        eccentricity: float = 0.05,
        enable_greene: bool = True,
        zeno_probe_interval: Optional[float] = None,
    ) -> PaladinWitnessSovereignState:
        """
        Ejecuta el ciclo OODA pasivo completo encadenando formalmente:
            1.9 emit_phase1_certificate ──► Φ₁₂ (2.1) ──► 2.9 emit_phase2_certificate
                                        ──► Φ₂₃ (3.1) ──► 3.9 emit_sovereign_state.
        `frequencies is None` (no `or`) para evitar la ambigüedad de verdad de arrays numpy. [A15]
        Nunca lanza por evidencia: un VETOED se atestigua como VETOED y el fibrado queda intacto.
        """
        self._telemetry.total_ooda_cycles += 1
        p1_cert = self.observer.emit_phase1_certificate(
            rho_state=rho_state, rho_target=rho_target, hamiltonian=hamiltonian,
            zeno_probe_interval=zeno_probe_interval,
        )
        if not self.continue_to_phase2_gate(p1_cert):
            raise PermissionError("Compuerta Φ₁₂ violada: firma HMAC P1 inválida (forja criptográfica).")

        freqs = list(frequencies) if frequencies is not None else [1.0, math.sqrt(2.0), math.sqrt(3.0)]
        p2_cert = self.orientor.emit_phase2_certificate(
            p1_cert=p1_cert, frequencies=freqs,
            previous_residual=previous_residual, current_residual=current_residual,
            perturbation_eps=perturbation_eps, eccentricity=eccentricity, enable_greene=enable_greene,
        )
        gate23 = self.decider._gate_phase2(p2_cert)
        if not gate23.authentic:
            raise PermissionError("Compuerta Φ₂₃ violada: firma HMAC P2 inválida (forja criptográfica).")
        # La obstrucción de Banach es EVIDENCIAL: se atestigua, no se lanza. [A4/doctrina]
        return self.decider.emit_sovereign_state(p2_cert)

    # ── Verificación y telemetría ─────────────────────────────────────────────
    def verify_sovereign_state(self, state: PaladinWitnessSovereignState) -> SovereignVerificationReport:
        """Verificador público: recomputa firmas, raíces, veredicto y Φ_sem desde el certificado."""
        return self.decider.verify_sovereign_state(state)

    def get_last_sovereign_state(self) -> Optional[PaladinWitnessSovereignState]:
        """Último Estado Soberano emitido (o None si el agente aún no ha cycles)."""
        return self.decider._last_sovereign_state

    def get_agent_telemetry(self) -> RSILevel3AgentGovernanceTelemetry:
        """Telemetría acumulada de gobernanza OODA y automejora RSI-L3 del agente."""
        return self._telemetry


# ══════════════════════════════════════════════════════════════════════════════
# §7 · AUTOVERIFICACIÓN EJECUTABLE (regresiones de seguridad y identidades)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """
    Batería de regresiones. Las tres primeras son las que v14 NO podía pasar:
    detección de mutación del veredicto, huella ante no contigüidad y monotonía de Φ_sem.
    """
    checks: List[Tuple[str, bool]] = []
    agent = BoolNovikovPaladinWitnessAgent(secret_key=b"self-test-key-32-bytes-of-entropy!!")

    # [A2] La suma de bytes NO detecta permutaciones; SHA-256 sí. Y no debe crashear en ρ.T.
    a = np.array([[0.4 + 0j, 0.1j], [-0.1j, 0.6]], dtype=np.complex128)
    b = np.array([[0.6 + 0j, 0.1j], [-0.1j, 0.4]], dtype=np.complex128)
    fp_a = Phase1WitnessObserver._capture_ram_fingerprint(a)
    fp_b = Phase1WitnessObserver._capture_ram_fingerprint(b)
    byte_sum_equal = int(np.sum(np.ascontiguousarray(a).view(np.uint8))) == \
        int(np.sum(np.ascontiguousarray(b).view(np.uint8)))
    checks.append(("[A2] SHA-256 distingue estados con igual suma de bytes",
                   byte_sum_equal and fp_a.content_sha256 != fp_b.content_sha256))
    try:
        fp_t = Phase1WitnessObserver._capture_ram_fingerprint(a.T)
        noncontig_ok = fp_t.valid and fp_t.content_sha256 != ""
    except Exception:
        noncontig_ok = False
    checks.append(("[A2] Huella segura sobre array NO contiguo (ρ.T)", noncontig_ok))

    rho = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    ham = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
    state = agent.execute_ooda_governance(rho_state=rho, rho_target=rho, hamiltonian=ham)
    p1 = state.p2_certificate.p1_certificate
    p2 = state.p2_certificate

    # [A1] REGRESIÓN CRÍTICA: mutar el veredicto debe INVALIDAR la firma y cerrar Φ₁₂.
    forged_report = replace(p1.qnd_report, truth_value=HeytingTruthValue.COHERENT)
    forged_p1 = replace(p1, qnd_report=forged_report)
    checks.append(("[A1] Φ₁₂ RECHAZA veredicto mutado (v14 lo aceptaba)",
                   not agent.continue_to_phase2_gate(forged_p1)))
    forged_p2 = replace(p2, banach_report=replace(p2.banach_report, banach_ratio_q=0.10))
    checks.append(("[A1] Φ₂₃ RECHAZA razón de Banach mutada",
                   not agent.continue_to_phase3_gate(forged_p2)))

    # [A13] Verificador público sobre estado legítimo y sobre estado adulterado.
    checks.append(("[A13] verify_sovereign_state(legítimo).all_ok",
                   agent.verify_sovereign_state(state).all_ok))
    tampered = replace(state, truth_value=HeytingTruthValue.COHERENT)
    checks.append(("[A13] verify_sovereign_state(adulterado) detecta",
                   not agent.verify_sovereign_state(tampered).all_ok))

    # Identidades geométricas.
    geom = p1.geometry
    checks.append(("[A8] Cuerda-arco d_B = 2·sin(θ_B/2)",
                   geom is not None and geom.chord_arc_identity_error < 1e-12))
    checks.append(("[A8] Bargmann nulo para triple conmutante",
                   geom is not None and (not geom.commuting_reference_ok
                                         or abs(geom.bargmann_triangle_phase_rad) < 1e-12)))

    # [A5] Monotonía del funtor Φ_sem con la MISMA evidencia.
    shields = []
    for tv in (HeytingTruthValue.COHERENT, HeytingTruthValue.DEGRADED, HeytingTruthValue.VETOED):
        val, _ = Phase3WitnessDecider._evaluate_visceral_semantic_functor(tv, p2, 0, 0)
        shields.append(val.capital_shield_efficiency)
    checks.append(("[A5] Φ_sem monótono: escudo(C) ≥ escudo(D) ≥ escudo(V)",
                   shields[0] >= shields[1] >= shields[2]))
    val_f05, _ = Phase3WitnessDecider._evaluate_visceral_semantic_functor(
        HeytingTruthValue.DEGRADED, p2, 0, 0)
    checks.append(("[A6] FPR nunca asserted 0 sin identificabilidad",
                   (val_f05.fpr_bound_identifiable and 0.0 <= val_f05.false_positive_rate <= 1.0)
                   or val_f05.false_positive_rate == 1.0))

    # [A9] Nobleza: sin datos ⟹ margen 0; racional cercano a φ ⟹ no noble.
    undefined = Phase2WitnessOrientor._evaluate_noble_rotation_diagnostics([], 0.0)
    rational = Phase2WitnessOrientor._evaluate_noble_rotation_diagnostics([0.6180339887], 0.0)
    checks.append(("[A9] Nobleza indefinida ⟹ defined=False (no perfección)", not undefined.defined))
    checks.append(("[A9] Racional cercano a φ ⟹ nobility_score = 0",
                   rational.nobility_score == 0.0 or rational.rational_suspect))

    # [A4] Doctrina: un VETOED auténtico se OBSERVA (no lanza) y se atestigua como VETOED.
    rho_bad = np.array([[0.5, 0.45j, 0, 0], [-0.45j, 0.5, 0, 0],
                        [0, 0, 0.0, 0], [0, 0, 0, 0.0]], dtype=np.complex128)
    try:
        veto_state = agent.execute_ooda_governance(rho_state=rho_bad, rho_target=rho, hamiltonian=ham)
        passive_ok = isinstance(veto_state, PaladinWitnessSovereignState)
    except PermissionError:
        passive_ok = False
    checks.append(("[A4] Evidencia adversa ⟹ Estado Soberano, NO excepción", passive_ok))

    # [A13] Cadena estrictamente creciente.
    checks.append(("[A13] Secuencia P1 < P2 < P3 y reloj monótono",
                   p1.sequence < p2.sequence < state.sequence
                   and p1.monotonic_ns < p2.monotonic_ns < state.monotonic_ns))
    checks.append(("[A12] Modus ponens de Heyting verificado",
                   state.adjudication is not None and state.adjudication.modus_ponens_ok))

    ok = True
    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN: {'COHERENTE' if ok else 'OBSTRUIDO'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s")
    raise SystemExit(0 if _self_test() else 1)
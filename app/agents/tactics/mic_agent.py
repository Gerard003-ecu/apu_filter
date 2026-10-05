# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Agent (Morfismo Geométrico & Soberano de Calibre de Poincaré)   ║
║ Ruta   : app/agents/tactics/mic_agent.py                                     ║
║ Versión: 6.0.0-Celestial-Poincare-Birkhoff-KAM-Morse-Gromov-Doctoral         ║
╚══════════════════════════════════════════════════════════════════════════════╝
SINOPSIS CATEGÓRICA Y ADJUNCIÓN DE GAUGE TÁCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Endofuntor soberano del Estrato TACTICS que integra, sobre la axiomática
simpléctica original, la maquinaria analítica COMPLETA de la Mecánica Celeste
de Henri Poincaré, incluyendo los métodos de perturbación, formas normales y
teoremas ergódicos que Poincaré desarrolló en "Les Méthodes Nouvelles de la
Mécanique Céleste" (1892-1899).

  FASE 1 — Núcleo Simpléctico Poincaréano (Tejido Geométrico Basal)
    • Transformación de Jacobi a variables acción-ángulo (J, θ).
    • Forma canónica de Darboux y matriz simpléctica M ∈ Sp(2n, ℝ).
    • Espectro de Lyapunov característico (LCE) y firma de Krein.
    • [NEW] Invariante integral relativo de Poincaré-Cartan ∮ p dq.
    • [NEW] Transformación canónica vía Serie de Lie exp(εL_χ).
    • [NEW] Forma Normal de Birkhoff y test de no-resonancia.
    • [NEW] Serie del Parámetro Pequeño de Poincaré (divergencia genérica).
    • Marco de flujo PoincareFlowFrame como fibrado base del espacio de fases.

  FASE 2 — Secciones, Mapas de Retorno y Resonancias KAM (Tejido Dinámico)
    • Sección de Poincaré Σ ⊂ T*ℳ y mapa de primer retorno P: Σ → Σ.
    • Número de rotación ν y vectores de enrollamiento (winding).
    • Criterio de solapamiento de resonancias de Chirikov (K > 1 ⇒ caos).
    • Función de Melnikov M(t₀) para detectar enredos homoclínicos.
    • Criterio de estabilidad de Delone-Hill para jerarquías logísticas.
    • [NEW] Fracción continua y Condición Diofántica de Kolmogorov.
    • [NEW] Teorema de Recurrencia de Poincaré.
    • [NEW] Teorema Ergódico de Birkhoff (promedios temporales).
    • [NEW] Iteración de Newton superconvergente KAM (denominadores pequeños).

  FASE 3 — Topología Global y Gobernanza Soberana (Tejido Categorial)
    • Teorema del índice de Poincaré-Hopf (Σ índices = χ(ℳ)).
    • Dualidad de Poincaré H^k ≅ H_{n-k} sobre la MIC.
    • [NEW] Desigualdades de Morse (débiles, fuertes, igualdad de Euler).
    • [NEW] No-Squeezing de Gromov (capacidad simpléctica c_G = πr²).
    • Certificado soberano PoincareSovereignCertificate con Veto de Gromov.
    • Clausura funtorial f_* ∘ f* ≅ Id garantizando Zero Side-Effects.

AXIOMÁTICA ALGEBRAICA, TOPOLÓGICA Y ESPECTRAL DE HENRI POINCARÉ:
────────────────────────────────────────────────────────────────────────────────
  [A1] Adjunción de Galois y Reversibilidad Funtorial:
       Hom_D(F(MIC), MAC) ≅ Hom_C(MIC, G(MAC))
  [A2] Inmersión Simpléctica de Darboux y Volumen de Liouville:
       Mᵀ Ω M = Ω ⇒ det(M) = +1 ⇒ Vol(φ(U)) = Vol(U)
  [A3] Contracción de Lipschitz en el Anillo Ultramétrico de Novikov (KAM):
       ‖F⁻¹(x) − F⁻¹(y)‖_V ≤ L_max ‖x − y‖_T,  L_max ≤ 1/(2λ_min^{3/2})
  [A4] Recurrencia Ergódica y Filtrado de Socavones por Mayer-Vietoris:
       Δβ₁ = β₁(A∪B) − [β₁(A) + β₁(B) − β₁(A∩B)] ≠ 0 ⇒ VETO
  [A5] Sección de Poincaré y Mapa de Primer Retorno:
       P: Σ → Σ, Σ = {z ∈ T*ℳ : g(z) = 0, ġ(z) > 0}
       El mapa P preserva la medida de Liouville dμ = ω^n/n!.
  [A6] Criterio de Solapamiento de Resonancias de Chirikov:
       K = (Δω / δω_res) ≥ 1 ⇒ destrucción de toros KAM ⇒ caos determinista.
  [A7] Función de Melnikov para Tangencias Homoclínicas:
       M(t₀) = ∫_{-∞}^{+∞} {H₀, H₁}(z₀(t−t₀)) dt
       M(t₀) = 0 con M'(t₀) ≠ 0 ⇒ intersección transversal estable/inestable.
  [A8] Índice de Poincaré-Hopf:
       ∑_{p ∈ Crit(X)} ind_p(X) = χ(ℳ)  (característica de Euler-Poincaré)
  [A9] NEW — Invariante Integral Relativo de Poincaré-Cartan:
       ∮_{γ(t)} p·dq = const.  ∀t, bajo el flujo Hamiltoniano φ_t.
  [A10] NEW — Forma Normal de Birkhoff y No-Resonancia:
       H = Σ ω_i N_i + O(|z|^{2k+2})  válido sii  Σ k_i ω_i ≠ 0, ∀|k| ≤ 2k.
  [A11] NEW — Condición Diofántica de Kolmogorov (hipótesis KAM):
       |ν − p/q| > γ/q^{2+τ},  ∀ p,q ∈ ℤ, q > 0.
  [A12] NEW — Recurrencia de Poincaré y Ergodicidad de Birkhoff:
       μ(T) > 0 ⇒ μ-c.t.p. z ∈ T retorna a T infinitas veces;
       lim_{N→∞} (1/N)Σf(φ_t z) = ∫f dμ  (c.t.p., sistemas ergódicos).
  [A13] NEW — Desigualdades de Morse:
       c_k ≥ b_k;  Σ(-1)^{k-i}c_i ≥ Σ(-1)^{k-i}b_i;  Σ(-1)^i c_i = χ(ℳ).
  [A14] NEW — No-Squeezing Simpléctico de Gromov:
       B^{2n}(r) ↪_simp Z^{2n}(R) ⟺ r ≤ R   (invariante: capacidad c_G = πr²).
"""
from __future__ import annotations
import dataclasses
import hashlib
import itertools
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto, unique
from fractions import Fraction
from typing import (
    Any, Callable, ClassVar, Deque, Dict, Final, FrozenSet, Iterable, List,
    Mapping, Optional, Protocol, Sequence, Set, Tuple, Type, TypeVar, Union,
    runtime_checkable, TypeGuard,
)
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
# ── Imports relativos protegidos ──────────────────────────────────────────────
try:
    from app.core.schemas import Stratum
except ImportError:
    from app.core.mic_algebra import Stratum
try:
    from app.core.mic_algebra import (
        CategoricalState, Morphism, TopologicalInvariantError, _canonicalize,
    )
except ImportError:
    Stratum = None
    CategoricalState = None
    Morphism = None
    TopologicalInvariantError = None
    _canonicalize = None
try:
    from app.adapters.tools_interface import MICRegistry
except ImportError:
    MICRegistry = None
try:
    from app.boole.strategy.sheaf_cohomology_orchestrator import (
        SheafCohomologyOrchestrator, SheafCohomologyError, CellularSheaf,
    )
except ImportError:
    SheafCohomologyOrchestrator = None
    SheafCohomologyError = Exception
    CellularSheaf = None
try:
    from app.core.immune_system.topological_watcher import (
        create_immune_watcher, ImmuneWatcherMorphism,
    )
except ImportError:
    create_immune_watcher = None
    ImmuneWatcherMorphism = None
# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
logger = logging.getLogger("MIC.Agent.CelestialPoincare")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
# ==============================================================================
# CONSTANTES MATEMÁTICAS RIGUROSAS Y LÍMITES CELESTES
# ==============================================================================
MAX_AUDIT_TRAIL_SIZE: Final[int] = 10_000
TOON_START_MARKER: Final[str] = "--- INICIO TOON ---"
TOON_END_MARKER: Final[str] = "--- FIN TOON ---"
TOON_FIELD_SEPARATOR: Final[str] = "|"
ENCAPSULATION_PROTOCOL_VERSION: Final[str] = "6.0.0-Celestial-Poincare-Birkhoff-Gromov"
_WILKINSON_LIMIT: Final[float] = 1.0e-12
_SPECTRAL_TOL: Final[float] = 1.0e-9
EPS: Final[float] = np.finfo(np.float64).eps * 4
ALGEBRAIC_TOL: Final[float] = 1e-10
FLOAT_COMPARISON_TOL: Final[float] = 1e-9
MAX_TENSOR_RANK: Final[int] = 2
MAX_COMPRESSION_RATIO: Final[float] = 10.0
MIN_COMPRESSION_RATIO: Final[float] = 0.01
# Constantes celestes
_CHIRIKOV_CRITICAL: Final[float] = 1.0
_KREIN_ELLIPTIC_TOL: Final[float] = 1.0e-10
_MELNIKOV_TOL: Final[float] = 1.0e-8
_POINCARE_SECTION_MAX_POINTS: Final[int] = 4096
# Constantes celestes NUEVAS (Birkhoff / KAM / Morse / Gromov)
_BIRKHOFF_DEFAULT_ORDER: Final[int] = 4
_DIOPHANTINE_GAMMA_DEFAULT: Final[float] = 0.3
_DIOPHANTINE_TAU_DEFAULT: Final[float] = 2.0
_FD_STEP_DEFAULT: Final[float] = 1.0e-6
_SMALL_DENOMINATOR_CUTOFF: Final[float] = 1.0e-6
# ==============================================================================
# JERARQUÍA DE EXCEPCIONES
# ==============================================================================
class MICAgentError(Exception):
    __slots__ = ("error_code", "details", "severity", "timestamp")
    def __init__(self, message: str, error_code: str = "UNKNOWN",
                 details: Optional[Dict[str, Any]] = None, severity: int = 1) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.severity = severity
        self.timestamp = time.time()
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__, "error_code": self.error_code,
                "message": str(self), "details": self.details,
                "severity": self.severity, "timestamp": self.timestamp}
if TopologicalInvariantError is None:
    class TopologicalInvariantError(MICAgentError):
        def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
            super().__init__(message, error_code="TOPOLOGICAL_INVARIANT", details=details, severity=3)
class StratumResolutionError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "STRATUM_RESOLUTION", d, 2)
class ContractValidationError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "CONTRACT_VALIDATION", d, 2)
class ClosureViolationError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "CLOSURE_VIOLATION", d, 3)
class AlgebraicVetoError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "ALGEBRAIC_VETO", d, 3)
class TOONCompressionError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "TOON_COMPRESSION", d, 2)
class SiloAccessError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "SILO_ACCESS", d, 2)
class ProjectionError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "PROJECTION", d, 3)
class FunctorialityError(MICAgentError):
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "FUNCTORIALITY", d, 3)
class CelestialMechanicsError(MICAgentError):
    """Error en la maquinaria analítica de mecánica celeste."""
    def __init__(self, m: str, d: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(m, "CELESTIAL_MECHANICS", d, 3)
# ==============================================================================
# ENUMERACIONES Y TIPOS BASE
# ==============================================================================
_IMPEDANCE_SEVERITY_MAP: Dict[str, int] = {
    "LAMINAR_PROJECTION": 0, "INPUT_TYPE_ERROR": 1, "SCHEMA_VALIDATION_ERROR": 1,
    "TOON_COMPRESSION_ERROR": 1, "MIC_RESOLUTION_ERROR": 2,
    "STRATUM_MISMATCH_REJECTED": 2, "ALGEBRAIC_VETO": 3,
    "TOPOLOGICAL_BIFURCATION": 3, "COHOMOLOGY_FAILURE": 3,
    "KAM_TORUS_DESTROYED": 3, "HOMOCLINIC_TANGLE": 3, "CHIRIKOV_CHAOS": 3,
}
@unique
class ImpedanceMatchStatus(str, Enum):
    LAMINAR_PROJECTION = "LAMINAR_PROJECTION"
    STRATUM_MISMATCH_REJECTED = "STRATUM_MISMATCH_REJECTED"
    TOON_COMPRESSION_ERROR = "TOON_COMPRESSION_ERROR"
    ALGEBRAIC_VETO = "ALGEBRAIC_VETO"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    MIC_RESOLUTION_ERROR = "MIC_RESOLUTION_ERROR"
    INPUT_TYPE_ERROR = "INPUT_TYPE_ERROR"
    TOPOLOGICAL_BIFURCATION = "TOPOLOGICAL_BIFURCATION"
    COHOMOLOGY_FAILURE = "COHOMOLOGY_FAILURE"
    KAM_TORUS_DESTROYED = "KAM_TORUS_DESTROYED"
    HOMOCLINIC_TANGLE = "HOMOCLINIC_TANGLE"
    CHIRIKOV_CHAOS = "CHIRIKOV_CHAOS"
    @property
    def is_terminal(self) -> bool:
        return self in {
            ImpedanceMatchStatus.ALGEBRAIC_VETO,
            ImpedanceMatchStatus.TOPOLOGICAL_BIFURCATION,
            ImpedanceMatchStatus.COHOMOLOGY_FAILURE,
            ImpedanceMatchStatus.KAM_TORUS_DESTROYED,
            ImpedanceMatchStatus.HOMOCLINIC_TANGLE,
            ImpedanceMatchStatus.CHIRIKOV_CHAOS,
        }
    @property
    def severity(self) -> int:
        return _IMPEDANCE_SEVERITY_MAP.get(self.name, 1)
@unique
class ValidationSeverity(IntEnum):
    ERROR = auto(); WARNING = auto(); INFO = auto()
    @property
    def heyting_value(self) -> float:
        return {ValidationSeverity.ERROR: 0.0,
                ValidationSeverity.WARNING: 0.5,
                ValidationSeverity.INFO: 1.0}[self]
@unique
class KreinSignature(IntEnum):
    """Firma de Krein para autovalores del espectro linealizado."""
    ELLIPTIC = 0      # par conjugado puro imaginario (estable)
    HYPERBOLIC = 1    # par real ±λ (inestable)
    PARABOLIC = 2     # autovalor doble en ±i (bifurcación)
    COMPLEX = 3       # cuádruple complejo (inestable oscilatorio)
T = TypeVar("T")
JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]
JSONSchema = Dict[str, Any]
PayloadType = Mapping[str, Any]
@runtime_checkable
class VectorInfoProvider(Protocol):
    def get_vector_info(self, vector_name: str) -> Optional[Dict[str, Any]]: ...
@runtime_checkable
class ProjectionTarget(Protocol):
    def project_intent(self, target_basis_vector: str, stratum_target: int,
                       validated_subspaces: List[str], orthogonality_guarantee: float,
                       payload: Dict[str, Any]) -> Dict[str, Any]: ...
# ==============================================================================
# UTILIDADES MATEMÁTICAS BASE
# ==============================================================================
class MathUtils:
    @staticmethod
    def stable_hash(data: Any) -> str:
        try:
            canonical = _canonicalize(data) if _canonicalize is not None else data
            serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"))
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as e:
            logger.debug("Fallback a repr() para hash: %s", e)
            return hashlib.sha256(repr(data).encode("utf-8")).hexdigest()
    @staticmethod
    def compute_tensor_rank(payload: Any, depth: int = 0, max_depth: int = 100) -> int:
        if depth >= max_depth:
            return max_depth
        if isinstance(payload, (dict, list, tuple)):
            if not payload:
                return 1
            children = payload.values() if isinstance(payload, dict) else payload
            max_child_rank = 0
            for child in children:
                child_rank = MathUtils.compute_tensor_rank(child, depth + 1, max_depth)
                if child_rank >= max_depth:
                    return max_depth
                if child_rank > max_child_rank:
                    max_child_rank = child_rank
            return min(1 + max_child_rank, max_depth)
        return 0
    @staticmethod
    def float_equal(a: float, b: float, tol: float = FLOAT_COMPARISON_TOL) -> bool:
        abs_diff = abs(a - b)
        if abs_diff <= tol:
            return True
        return abs_diff <= tol * max(abs(a), abs(b))
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) > max_val ({max_val})")
        return max(min_val, min(max_val, value))
def normalize_stratum(value: Any) -> Stratum:
    if isinstance(value, Stratum): return value
    if isinstance(value, int):
        try: return Stratum(value)
        except ValueError as e:
            raise StratumResolutionError(f"Entero inválido: {value}", {"input_value": value}) from e
    if isinstance(value, str):
        try: return Stratum[value.upper()]
        except KeyError: pass
        try: return Stratum(int(value))
        except (ValueError, KeyError) as e:
            raise StratumResolutionError(f"String inválido: '{value}'", {"input_value": value}) from e
    raise StratumResolutionError(f"Tipo no soportado: {type(value).__name__}")
def python_type_matches(expected_type: str, value: Any) -> bool:
    m = {"null": lambda v: v is None, "boolean": lambda v: isinstance(v, bool),
         "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
         "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
         "string": lambda v: isinstance(v, str),
         "array": lambda v: isinstance(v, list),
         "object": lambda v: isinstance(v, Mapping)}
    c = m.get(expected_type)
    return True if c is None else c(value)
def compute_json_path(base: str, key: Union[str, int]) -> str:
    if isinstance(key, int): return f"{base}[{key}]"
    safe_key = key.replace(".", "\\.").replace("[", "\\[")
    return f"{base}.{safe_key}"
# ==============================================================================
# DATACLASSES DE AUDITORÍA
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class SchemaValidationResult:
    frustration_ideal: float = 0.0
    validity_degree: float = 1.0
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    path: str = "$"
    def __post_init__(self) -> None:
        if not (0.0 <= self.validity_degree <= 1.0):
            object.__setattr__(self, "validity_degree",
                               MathUtils.clamp(self.validity_degree, 0.0, 1.0))
    @property
    def is_valid(self) -> bool:
        return self.validity_degree >= 1.0 - EPS
    @classmethod
    def success(cls) -> "SchemaValidationResult": return cls(validity_degree=1.0)
    @classmethod
    def failure(cls, error: str, path: str = "$", penalty: float = 1.0) -> "SchemaValidationResult":
        return cls(validity_degree=max(0.0, 1.0 - penalty), errors=(error,), path=path)
    @classmethod
    def merge(cls, results: Iterable["SchemaValidationResult"]) -> "SchemaValidationResult":
        ae, aw, mv = [], [], 1.0
        for r in results:
            ae.extend(r.errors); aw.extend(r.warnings)
            if r.validity_degree < mv: mv = r.validity_degree
        return cls(validity_degree=mv, errors=tuple(ae), warnings=tuple(aw))
    @property
    def error(self) -> Optional[str]:
        return self.errors[0] if self.errors else None
    def to_dict(self) -> Dict[str, Any]:
        return {"validity_degree": float(self.validity_degree), "errors": list(self.errors),
                "warnings": list(self.warnings), "path": self.path, "is_valid": self.is_valid}
@dataclass(frozen=True, slots=True, eq=True)
class CategoricalEqualizerSeed:
    target_vector: str
    target_stratum: Stratum
    silo_a_contract_id: str
    silo_b_cartridge_id: str
    impedance_match_status: ImpedanceMatchStatus
    token_compression_ratio: float = 0.0
    raw_telemetry_hash: str = ""
    llm_output_hash: str = ""
    validation_errors: Tuple[str, ...] = field(default_factory=tuple)
    protocol_version: str = ENCAPSULATION_PROTOCOL_VERSION
    timestamp: float = field(default_factory=time.time)
    poincare_frame_hash: str = ""
    kam_stability_index: float = 1.0
    chirikov_overlap_ratio: float = 0.0
    def __post_init__(self) -> None:
        if self.token_compression_ratio < 0.0:
            object.__setattr__(self, "token_compression_ratio", 0.0)
    def to_dict(self) -> Dict[str, Any]:
        return {"target_vector": self.target_vector,
                "target_stratum": self.target_stratum.name,
                "silo_a_contract_id": self.silo_a_contract_id,
                "silo_b_cartridge_id": self.silo_b_cartridge_id,
                "impedance_match_status": self.impedance_match_status.value,
                "token_compression_ratio": float(self.token_compression_ratio),
                "raw_telemetry_hash": self.raw_telemetry_hash,
                "llm_output_hash": self.llm_output_hash,
                "validation_errors": list(self.validation_errors),
                "protocol_version": self.protocol_version,
                "timestamp": self.timestamp,
                "poincare_frame_hash": self.poincare_frame_hash,
                "kam_stability_index": float(self.kam_stability_index),
                "chirikov_overlap_ratio": float(self.chirikov_overlap_ratio)}
    def compute_hash(self) -> str:
        return MathUtils.stable_hash({k: v for k, v in self.to_dict().items()
                                      if k != "timestamp"})
@dataclass(frozen=True, slots=True, eq=True)
class TOONDocument:
    cartridge_id: str
    header_template: str
    records: Tuple[Tuple[str, str], ...]
    def __post_init__(self) -> None:
        if not self.cartridge_id:
            raise TOONCompressionError("cartridge_id no puede estar vacío")
    def render(self) -> str:
        lines = [f"{TOON_START_MARKER} {self.cartridge_id} ---", self.header_template]
        for k, v in self.records: lines.append(f"{k}{TOON_FIELD_SEPARATOR}{v}")
        lines.append(TOON_END_MARKER)
        return "\n".join(lines)
    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, jv in self.records:
            try: result[k] = json.loads(jv)
            except json.JSONDecodeError: result[k] = jv
        return result
@dataclass(frozen=True, slots=True, eq=True)
class SiloAContract:
    contract_id: str
    stratum: Stratum
    schema: JSONSchema
    description: str = ""
    version: str = "1.0.0"
    def __post_init__(self) -> None:
        if not isinstance(self.schema, dict) or "type" not in self.schema:
            raise ContractValidationError(f"Schema inválido: '{self.contract_id}'")
    def to_dict(self) -> Dict[str, Any]:
        return {"contract_id": self.contract_id, "stratum": self.stratum.name,
                "schema": self.schema, "description": self.description, "version": self.version}
@dataclass(frozen=True, slots=True, eq=True)
class SiloBCartridge:
    cartridge_id: str
    stratum: Stratum
    header_template: str
    field_definitions: Tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    version: str = "1.0.0"
    def __post_init__(self) -> None:
        if not self.cartridge_id: raise TOONCompressionError("cartridge_id vacío")
    def to_dict(self) -> Dict[str, Any]:
        return {"cartridge_id": self.cartridge_id, "stratum": self.stratum.name,
                "header_template": self.header_template,
                "field_definitions": list(self.field_definitions),
                "description": self.description, "version": self.version}
# ==============================================================================
# CERTIFICADO SIMPLÉCTICO BASAL
# ==============================================================================
@dataclass(frozen=True, slots=True, eq=True)
class PoincareMICAdjunctionCertificate:
    symplectic_residual: float
    volume_drift: float
    lipschitz_ceiling: float
    galois_residual_norm: float
    is_poincare_adjunction_coherent: bool
    def to_dict(self) -> Dict[str, Any]:
        return {"symplectic_residual": float(self.symplectic_residual),
                "volume_drift": float(self.volume_drift),
                "lipschitz_ceiling": float(self.lipschitz_ceiling),
                "galois_residual_norm": float(self.galois_residual_norm),
                "is_poincare_adjunction_coherent": self.is_poincare_adjunction_coherent}
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                          FASE 1 — INICIO                                █
# █        NÚCLEO SIMPLÉCTICO POINCARÉANO (Tejido Geométrico Basal)         █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 1: Establece las estructuras geométricas basales de "Les Méthodes
Nouvelles de la Mécanique Céleste". Al final de la fase se construye el
`PoincareFlowFrame` enriquecido con el invariante de Poincaré-Cartan, la
forma normal de Birkhoff y la serie del parámetro pequeño — que actúa como
semilla funtorial categorial para la FASE 2 (mapas de retorno y KAM).
"""
# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Dataclasses del núcleo simpléctico
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, eq=False)
class PoincareActionAnglePair:
    r"""
    Par canónico (J, θ) ∈ ℝ^n × 𝕋^n obtenido por la transformación de Jacobi.
    Satisface las ecuaciones de Hamilton-Jacobi:
        J̇ = -∂H/∂θ,   θ̇ = +∂H/∂J
    """
    action: NDArray[np.float64]      # J ∈ ℝ^n_+
    angle: NDArray[np.float64]       # θ ∈ 𝕋^n ≡ ℝ^n / 2πℤ^n
    hamiltonian: float               # H(J, θ) evaluado
    is_angle_reduced: bool           # θ mod 2π ya aplicado
    def __post_init__(self) -> None:
        if self.action.ndim != 1:
            raise CelestialMechanicsError(
                f"Acción debe ser 1-D, recibido ndim={self.action.ndim}"
            )
        if self.action.shape != self.angle.shape:
            raise CelestialMechanicsError(
                f"Dimensiones acción/ángulo desalineadas: {self.action.shape} vs {self.angle.shape}"
            )
        if np.any(self.action < -FLOAT_COMPARISON_TOL):
            raise CelestialMechanicsError(
                f"Acciones negativas detectadas: min={float(np.min(self.action)):.3e}"
            )
    @property
    def n_dof(self) -> int:
        return int(self.action.size)
    @property
    def winding_number(self) -> NDArray[np.float64]:
        r"""Número de enrollamiento por grado de libertad: θ̇ / 2π (proxy discreto)."""
        return self.angle / (2.0 * np.pi)
    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action.tolist(), "angle": self.angle.tolist(),
                "hamiltonian": float(self.hamiltonian),
                "is_angle_reduced": self.is_angle_reduced,
                "n_dof": self.n_dof}
@dataclass(frozen=True, slots=True, eq=False)
class KreinSpectralDecomposition:
    r"""
    Descomposición espectral de Krein del operador linealizado M ∈ Sp(2n, ℝ).
    Clasifica cada autovalor λ en {elíptico, hiperbólico, parabólico, complejo}.
    """
    eigenvalues: Tuple[complex, ...]
    signatures: Tuple[KreinSignature, ...]
    lyapunov_exponents: Tuple[float, ...]     # λ_i = log|μ_i| / T
    is_spectrally_stable: bool
    stability_margin: float                   # min |Re(μ)| sobre el espectro
    def to_dict(self) -> Dict[str, Any]:
        return {"eigenvalues": [[z.real, z.imag] for z in self.eigenvalues],
                "signatures": [s.name for s in self.signatures],
                "lyapunov_exponents": list(self.lyapunov_exponents),
                "is_spectrally_stable": self.is_spectrally_stable,
                "stability_margin": float(self.stability_margin)}
@dataclass(frozen=True, slots=True, eq=False)
class PoincareFlowFrame:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SEMILLA FUNTORIAL DE FASE 1 → FASE 2                                   ║
    ║ Marco de flujo simpléctico sobre T*ℳ que empaqueta (J, θ), la matriz   ║
    ║ simpléctica M, el espectro de Krein-Lyapunov, la firma KAM, el         ║
    ║ invariante de Poincaré-Cartan, la forma normal de Birkhoff y la        ║
    ║ serie del parámetro pequeño. CONSUMIDO por FASE 2 para la sección.     ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    action_angle: PoincareActionAnglePair
    symplectic_matrix: NDArray[np.float64]     # M ∈ Sp(2n, ℝ)
    canonical_form: NDArray[np.float64]        # Ω canónica
    krein_decomposition: KreinSpectralDecomposition
    kam_torus_index: float                     # ρ = ∏ |λ_i|^{1/2n}
    is_kam_stable: bool
    torsional_frequency: float                 # ω_t = ∂²H/∂J² (rigidez torsional)
    energy_level: float                        # E = H(J, θ)
    # ── NUEVO: invariantes de Poincaré-Cartan / Birkhoff / serie pequeña ──────
    integral_invariant_residual: float = 0.0
    birkhoff_frequencies: Tuple[float, ...] = field(default_factory=tuple)
    is_birkhoff_nonresonant: bool = True
    birkhoff_resonant_vector: Optional[Tuple[int, ...]] = None
    poincare_series_radius: float = float("inf")
    poincare_series_divergent: bool = False
    frame_hash: str = ""
    def __post_init__(self) -> None:
        if self.symplectic_matrix.ndim != 2:
            raise CelestialMechanicsError("Matriz simpléctica debe ser 2-D")
        n2 = self.symplectic_matrix.shape[0]
        if self.symplectic_matrix.shape[0] != self.symplectic_matrix.shape[1]:
            raise CelestialMechanicsError("Matriz simpléctica debe ser cuadrada")
        if n2 % 2 != 0:
            raise CelestialMechanicsError(
                f"Dimensión simpléctica debe ser par, recibido {n2}"
            )
        if not self.frame_hash:
            object.__setattr__(self, "frame_hash",
                               MathUtils.stable_hash(self._hash_payload()))
    def _hash_payload(self) -> Dict[str, Any]:
        return {"aa": self.action_angle.to_dict(),
                "M": self.symplectic_matrix.tolist(),
                "Omega": self.canonical_form.tolist(),
                "krein": self.krein_decomposition.to_dict(),
                "kam_idx": float(self.kam_torus_index),
                "kam_stable": bool(self.is_kam_stable),
                "omega_t": float(self.torsional_frequency),
                "energy": float(self.energy_level),
                "cartan_residual": float(self.integral_invariant_residual),
                "birkhoff_freqs": list(self.birkhoff_frequencies),
                "birkhoff_nonresonant": bool(self.is_birkhoff_nonresonant),
                "birkhoff_resonant_vec": (list(self.birkhoff_resonant_vector)
                                          if self.birkhoff_resonant_vector else None),
                "series_radius": float(self.poincare_series_radius),
                "series_divergent": bool(self.poincare_series_divergent)}
    def to_dict(self) -> Dict[str, Any]:
        return {**self._hash_payload(), "frame_hash": self.frame_hash}
# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — Motor analítico del núcleo simpléctico
# ──────────────────────────────────────────────────────────────────────────────
class PoincareSymplecticKernel:
    r"""
    Núcleo simpléctico: transformaciones canónicas de Jacobi, forma de Darboux,
    espectro de Krein-Lyapunov, firma KAM, invariante de Poincaré-Cartan,
    serie de Lie, forma normal de Birkhoff y serie del parámetro pequeño.
    """
    # ── 1.2.1 Transformación de Jacobi a acción-ángulo ────────────────────────
    @staticmethod
    def jacobi_action_angle_transform(
        position: NDArray[np.float64],
        momentum: NDArray[np.float64],
        hamiltonian_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ) -> PoincareActionAnglePair:
        r"""
        Realiza la transformación de Jacobi (q, p) → (θ, J) asumiendo separabilidad
        del Hamiltoniano. Se emplea la aproximación de WKB/EBK:
            J_i = (1/2π) ∮ p_i dq_i
        Implementación práctica: J_i ≈ |q_i · p_i| y θ_i = atan2(p_i, q_i).
        """
        q = np.asarray(position, dtype=np.float64).ravel()
        p = np.asarray(momentum, dtype=np.float64).ravel()
        if q.shape != p.shape:
            raise CelestialMechanicsError(
                f"q y p desalineados: {q.shape} vs {p.shape}"
            )
        J = np.abs(q * p) + EPS
        theta = np.arctan2(p, q + EPS)
        theta = np.mod(theta, 2.0 * np.pi)
        H = float(hamiltonian_fn(q, p))
        return PoincareActionAnglePair(action=J, angle=theta,
                                        hamiltonian=H, is_angle_reduced=True)
    # ── 1.2.2 Forma canónica de Darboux Ω = ⊕ [[0, I], [-I, 0]] ───────────────
    @staticmethod
    def darboux_canonical_form(n_dof: int) -> NDArray[np.float64]:
        r"""
        Construye la matriz simpléctica canónica Ω ∈ 𝔰𝔭(2n, ℝ):
            Ω = [[0_n, I_n], [-I_n, 0_n]]
        con Ωᵀ = -Ω y Ω² = -I_{2n}.
        """
        if n_dof <= 0:
            raise CelestialMechanicsError(f"n_dof debe ser > 0, recibido {n_dof}")
        I = np.eye(n_dof, dtype=np.float64)
        O = np.zeros((n_dof, n_dof), dtype=np.float64)
        Omega = np.block([[O, I], [-I, O]])
        if not np.allclose(Omega.T, -Omega, atol=_SPECTRAL_TOL):
            raise CelestialMechanicsError("Falla de antisimetría en Ω")
        if not np.allclose(Omega @ Omega, -np.eye(2 * n_dof), atol=_SPECTRAL_TOL):
            raise CelestialMechanicsError("Falla de involución Ω² = -I en Ω")
        return Omega
    # ── 1.2.3 Auditoría de simplecticidad Mᵀ Ω M = Ω ──────────────────────────
    @staticmethod
    def audit_symplectic_condition(
        M: NDArray[np.float64],
        Omega: NDArray[np.float64],
        tol: float = _WILKINSON_LIMIT,
    ) -> Tuple[float, float, bool]:
        """Retorna (residuo_simpléctico, drift_volumen, es_simpléctica)."""
        if M.shape != Omega.shape:
            raise CelestialMechanicsError(
                f"Formas incompatibles M{M.shape} vs Ω{Omega.shape}"
            )
        defect = M.T @ Omega @ M - Omega
        residual = float(np.linalg.norm(defect, ord='fro'))
        det_M = float(np.linalg.det(M))
        vol_drift = abs(det_M - 1.0)
        return residual, vol_drift, (residual <= tol and vol_drift <= tol)
    # ── 1.2.4 Espectro de Krein-Lyapunov ──────────────────────────────────────
    @staticmethod
    def krein_lyapunov_spectrum(
        M: NDArray[np.float64],
        time_horizon: float = 1.0,
    ) -> KreinSpectralDecomposition:
        r"""
        Calcula el espectro μ_i de M y clasifica según la firma de Krein:
          • |Re(μ_i)| < tol y |μ_i| ≈ 1 ⇒ ELLIPTIC
          • μ_i real y |μ_i| ≠ 1       ⇒ HYPERBOLIC
          • |μ_i| ≈ 1 y par doble      ⇒ PARABOLIC
          • μ_i complejo |μ_i| ≠ 1     ⇒ COMPLEX (cuádruple inestable)
        Exponentes de Lyapunov: λ_i = ln|μ_i| / T.
        """
        if time_horizon <= 0:
            raise CelestialMechanicsError("time_horizon debe ser > 0")
        eigvals = la.eigvals(M)
        sigs: List[KreinSignature] = []
        lces: List[float] = []
        margin = float('inf')
        for mu in eigvals:
            mag = abs(mu)
            re = abs(mu.real)
            im = abs(mu.imag)
            lces.append(float(np.log(max(mag, EPS)) / time_horizon))
            if mag < _WILKINSON_LIMIT:
                sigs.append(KreinSignature.PARABOLIC)
            elif abs(mag - 1.0) <= _KREIN_ELLIPTIC_TOL and re <= _KREIN_ELLIPTIC_TOL and im > _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.ELLIPTIC)
            elif abs(mag - 1.0) <= _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.PARABOLIC)
            elif im <= _KREIN_ELLIPTIC_TOL:
                sigs.append(KreinSignature.HYPERBOLIC)
            else:
                sigs.append(KreinSignature.COMPLEX)
            if re > 0:
                margin = min(margin, float(re))
        is_stable = all(s == KreinSignature.ELLIPTIC for s in sigs)
        if margin == float('inf'):
            margin = 0.0
        return KreinSpectralDecomposition(
            eigenvalues=tuple(complex(z) for z in eigvals),
            signatures=tuple(sigs),
            lyapunov_exponents=tuple(sorted(lces, reverse=True)),
            is_spectrally_stable=is_stable,
            stability_margin=float(margin),
        )
    # ── 1.2.5 Firma KAM (índice torsional y rigidez) ──────────────────────────
    @staticmethod
    def kam_torus_index(
        lces: Tuple[float, ...],
        torsional_hessian: NDArray[np.float64],
    ) -> Tuple[float, float, bool]:
        r"""
        Índice KAM: ρ = ∏_i |λ_i|^{1/(2n)}.
        Frecuencia torsional: ω_t = λ_min(∂²H/∂J²) — condición de Kolmogorov.
        Estable si ρ ≤ 1 y ω_t > 0.
        """
        if not lces:
            return 1.0, 0.0, True
        log_rho = float(np.mean(np.abs(lces)))
        rho = float(np.exp(log_rho))
        try:
            eig_h = la.eigvalsh(torsional_hessian)
            omega_t = float(np.min(eig_h)) if eig_h.size > 0 else 0.0
        except la.LinAlgError:
            omega_t = 0.0
        is_stable = (rho <= 1.0 + _SPECTRAL_TOL) and (omega_t > _SPECTRAL_TOL)
        return rho, omega_t, is_stable
    # ── 1.2.6 NUEVO — Invariante Integral Relativo de Poincaré-Cartan ────────
    @staticmethod
    def poincare_cartan_invariant(
        loop_q: NDArray[np.float64],
        loop_p: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la circulación discreta de la 1-forma de Liouville θ = p dq sobre
        un lazo cerrado γ en el espacio de fases, vía la regla del trapecio:
            ∮_γ p dq ≈ Σ_i ½(p_i + p_{i+1})(q_{i+1} − q_i)
        Este es el integrando del Invariante Integral Relativo de Poincaré: si γ
        se transporta bajo el flujo Hamiltoniano φ_t, la circulación se conserva
        exactamente (consecuencia directa de dθ = ω siendo invariante de Lie).
        """
        q = np.asarray(loop_q, dtype=np.float64).ravel()
        p = np.asarray(loop_p, dtype=np.float64).ravel()
        if q.shape != p.shape or q.size < 3:
            raise CelestialMechanicsError(
                "El lazo requiere ≥3 puntos con q,p alineados"
            )
        q_next = np.roll(q, -1)
        p_next = np.roll(p, -1)
        circulation = float(np.sum(0.5 * (p + p_next) * (q_next - q)))
        return circulation
    @staticmethod
    def poincare_cartan_invariance_residual(
        loop_q_initial: NDArray[np.float64],
        loop_p_initial: NDArray[np.float64],
        loop_q_final: NDArray[np.float64],
        loop_p_final: NDArray[np.float64],
    ) -> float:
        r"""
        Residuo de la invariancia: |I(γ(t₁)) − I(γ(t₀))|. Debe anularse hasta el
        orden de truncamiento numérico si el flujo es genuinamente Hamiltoniano.
        """
        I0 = PoincareSymplecticKernel.poincare_cartan_invariant(loop_q_initial, loop_p_initial)
        I1 = PoincareSymplecticKernel.poincare_cartan_invariant(loop_q_final, loop_p_final)
        return abs(I1 - I0)
    # ── 1.2.7 NUEVO — Transformación canónica vía Serie de Lie exp(εL_χ) ─────
    @staticmethod
    def lie_series_canonical_transform(
        z: NDArray[np.float64],
        generating_fn: Callable[[NDArray[np.float64]], float],
        Omega: NDArray[np.float64],
        epsilon: float = 1.0e-3,
        order: int = 2,
        fd_step: float = _FD_STEP_DEFAULT,
    ) -> NDArray[np.float64]:
        r"""
        Aplica la transformación canónica generada por χ vía la expansión en
        serie de Lie (método de Deprit/Hori):
            z(ε) = exp(ε L_χ) z = z + ε·Ω∇χ(z) + (ε²/2)·D(Ω∇χ)(z)·Ω∇χ(z) + O(ε³)
        donde L_χ f = {f, χ} es el operador de Lie generado por χ. Las derivadas
        se calculan por diferencias finitas centradas de orden 2.
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        dim = z.size
        def grad_chi(x: NDArray[np.float64]) -> NDArray[np.float64]:
            g = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                xp = x.copy(); xp[i] += fd_step
                xm = x.copy(); xm[i] -= fd_step
                g[i] = (generating_fn(xp) - generating_fn(xm)) / (2.0 * fd_step)
            return g
        def vector_field(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return Omega @ grad_chi(x)
        v0 = vector_field(z)
        z_new = z + epsilon * v0
        if order >= 2:
            jac = np.zeros((dim, dim), dtype=np.float64)
            for i in range(dim):
                xp = z.copy(); xp[i] += fd_step
                xm = z.copy(); xm[i] -= fd_step
                jac[:, i] = (vector_field(xp) - vector_field(xm)) / (2.0 * fd_step)
            z_new = z_new + 0.5 * (epsilon ** 2) * (jac @ v0)
        return z_new
    # ── 1.2.8 NUEVO — Forma Normal de Birkhoff y test de no-resonancia ───────
    @staticmethod
    def birkhoff_normal_form_frequencies(
        hessian_H2: NDArray[np.float64],
        Omega: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], bool]:
        r"""
        Extrae las frecuencias normales ω_i del Hamiltoniano cuadrático
        H₂ = ½ zᵀAz linealizado en un equilibrio, diagonalizando el operador
        infinitesimal ΩA (forma de Williamson). Si todos los autovalores de ΩA
        son puramente imaginarios ±iω_i, el equilibrio es elíptico puro y
        H₂ = Σ ω_i N_i con N_i = ½(q_i² + p_i²) las acciones normales.
        """
        A = np.asarray(hessian_H2, dtype=np.float64)
        M = Omega @ A
        eigvals = la.eigvals(M)
        omegas_set = {
            round(abs(ev.imag), 10) for ev in eigvals
            if abs(ev.real) < _SPECTRAL_TOL and abs(ev.imag) > _SPECTRAL_TOL
        }
        is_pure_elliptic = all(abs(ev.real) < _SPECTRAL_TOL for ev in eigvals)
        omegas = np.array(sorted(omegas_set), dtype=np.float64)
        return omegas, is_pure_elliptic
    @staticmethod
    def birkhoff_resonance_check(
        omegas: NDArray[np.float64],
        max_order: int = _BIRKHOFF_DEFAULT_ORDER,
        tol: float = _SPECTRAL_TOL,
    ) -> Tuple[bool, Optional[Tuple[int, ...]]]:
        r"""
        Verifica la condición de no-resonancia de Birkhoff hasta orden
        `max_order`:
            Σ_i k_i ω_i ≠ 0,   ∀ k ∈ ℤ^n \ {0},  0 < Σ|k_i| ≤ max_order
        Si existe un vector resonante k, la forma normal de Birkhoff sólo puede
        truncarse por debajo del orden resonante (teorema de Birkhoff, 1927).
        Búsqueda exhaustiva acotada — viable para n ≤ 4 grados de libertad.
        """
        n = int(omegas.size)
        if n == 0:
            return True, None
        rng = range(-max_order, max_order + 1)
        for k in itertools.product(rng, repeat=n):
            order = sum(abs(x) for x in k)
            if 0 < order <= max_order:
                s = float(np.dot(np.array(k, dtype=np.float64), omegas))
                if abs(s) < tol:
                    return False, k
        return True, None
    # ── 1.2.9 NUEVO — Serie del Parámetro Pequeño de Poincaré ────────────────
    @staticmethod
    def poincare_small_parameter_expansion(
        H_terms: Sequence[Callable[[NDArray[np.float64], NDArray[np.float64]], float]],
        J: NDArray[np.float64],
        theta: NDArray[np.float64],
        epsilon: float,
    ) -> Tuple[float, Tuple[float, ...]]:
        r"""
        Evalúa la serie de perturbación del parámetro pequeño de Poincaré:
            H(J,θ;ε) = Σ_{k=0}^{K} ε^k H_k(J,θ)
        donde H₀ es integrable (separable) y H_k, k≥1, son las perturbaciones
        sucesivas (p.ej. en el problema restringido de 3 cuerpos). Retorna el
        valor total y la tupla de términos individuales ε^k H_k.
        """
        if not H_terms:
            raise CelestialMechanicsError("Se requiere al menos H_0 en la serie")
        terms = tuple(
            float(Hk(J, theta)) * (epsilon ** k) for k, Hk in enumerate(H_terms)
        )
        return float(sum(terms)), terms
    @staticmethod
    def poincare_series_divergence_estimate(
        terms_sequence: Sequence[float],
    ) -> Tuple[float, bool]:
        r"""
        Estima el radio de convergencia de la serie de Poincaré vía el criterio
        del cociente de D'Alembert: ρ ≈ media{|a_k / a_{k+1}|}.
        Teorema de no-existencia de Poincaré (1892): genéricamente, la serie de
        perturbación del problema de n-cuerpos NO converge uniformemente — no
        existen integrales primeras analíticas adicionales más allá de las
        clásicas (energía, momento angular, centro de masa). ρ ≤ 1 es indicio
        de divergencia (serie asintótica, útil sólo a orden finito).
        """
        vals = [abs(t) for t in terms_sequence if abs(t) > EPS]
        if len(vals) < 2:
            return float('inf'), False
        ratios = [vals[i] / vals[i + 1] for i in range(len(vals) - 1) if vals[i + 1] > EPS]
        if not ratios:
            return float('inf'), False
        rho = float(np.mean(ratios))
        is_divergent = rho <= 1.0 + _SPECTRAL_TOL
        return rho, is_divergent
# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — MÉTODO DE CIERRE: construye el PoincareFlowFrame (semilla FASE 2)
# ──────────────────────────────────────────────────────────────────────────────
def build_poincare_flow_frame(
    position: NDArray[np.float64],
    momentum: NDArray[np.float64],
    hamiltonian_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    linearization_M: Optional[NDArray[np.float64]] = None,
    torsional_hessian: Optional[NDArray[np.float64]] = None,
    time_horizon: float = 1.0,
    # ── NUEVO: parámetros opcionales Poincaré-Cartan / Birkhoff / serie ───────
    loop_q_initial: Optional[NDArray[np.float64]] = None,
    loop_p_initial: Optional[NDArray[np.float64]] = None,
    loop_q_final: Optional[NDArray[np.float64]] = None,
    loop_p_final: Optional[NDArray[np.float64]] = None,
    equilibrium_hessian: Optional[NDArray[np.float64]] = None,
    birkhoff_max_order: int = _BIRKHOFF_DEFAULT_ORDER,
    perturbation_terms: Optional[Sequence[Callable[[NDArray[np.float64], NDArray[np.float64]], float]]] = None,
    perturbation_epsilon: float = 0.0,
) -> PoincareFlowFrame:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FASE 1 → APERTURA FASE 2                                        ║
    ║ Construye el PoincareFlowFrame (semilla categorial) enriquecido con el ║
    ║ invariante de Poincaré-Cartan, la forma normal de Birkhoff y la serie  ║
    ║ del parámetro pequeño. Será consumido por el mapa de primer retorno    ║
    ║ (FASE 2.1) y por el criterio de Chirikov (FASE 2.2).                   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    aa = PoincareSymplecticKernel.jacobi_action_angle_transform(
        position, momentum, hamiltonian_fn
    )
    n_dof = aa.n_dof
    Omega = PoincareSymplecticKernel.darboux_canonical_form(n_dof)
    if linearization_M is None:
        linearization_M = np.eye(2 * n_dof, dtype=np.float64)
    linearization_M = np.asarray(linearization_M, dtype=np.float64)
    krein = PoincareSymplecticKernel.krein_lyapunov_spectrum(
        linearization_M, time_horizon=time_horizon
    )
    if torsional_hessian is None:
        torsional_hessian = np.eye(n_dof, dtype=np.float64)
    rho, omega_t, kam_ok = PoincareSymplecticKernel.kam_torus_index(
        krein.lyapunov_exponents, torsional_hessian
    )
    # ── Invariante integral de Poincaré-Cartan (opcional) ─────────────────────
    integral_residual = 0.0
    if (loop_q_initial is not None and loop_p_initial is not None
            and loop_q_final is not None and loop_p_final is not None):
        integral_residual = PoincareSymplecticKernel.poincare_cartan_invariance_residual(
            loop_q_initial, loop_p_initial, loop_q_final, loop_p_final
        )
    # ── Forma normal de Birkhoff (opcional) ───────────────────────────────────
    birkhoff_freqs: Tuple[float, ...] = tuple()
    is_nonresonant = True
    resonant_vec: Optional[Tuple[int, ...]] = None
    if equilibrium_hessian is not None:
        omegas, is_pure_elliptic = PoincareSymplecticKernel.birkhoff_normal_form_frequencies(
            equilibrium_hessian, Omega
        )
        birkhoff_freqs = tuple(float(w) for w in omegas)
        if is_pure_elliptic and omegas.size > 0:
            is_nonresonant, resonant_vec = PoincareSymplecticKernel.birkhoff_resonance_check(
                omegas, max_order=birkhoff_max_order
            )
        else:
            is_nonresonant = False
    # ── Serie del parámetro pequeño de Poincaré (opcional) ────────────────────
    series_radius = float("inf")
    series_divergent = False
    if perturbation_terms:
        _, terms = PoincareSymplecticKernel.poincare_small_parameter_expansion(
            perturbation_terms, aa.action, aa.angle, perturbation_epsilon
        )
        series_radius, series_divergent = PoincareSymplecticKernel.poincare_series_divergence_estimate(terms)
    return PoincareFlowFrame(
        action_angle=aa,
        symplectic_matrix=linearization_M,
        canonical_form=Omega,
        krein_decomposition=krein,
        kam_torus_index=rho,
        is_kam_stable=kam_ok,
        torsional_frequency=omega_t,
        energy_level=aa.hamiltonian,
        integral_invariant_residual=integral_residual,
        birkhoff_frequencies=birkhoff_freqs,
        is_birkhoff_nonresonant=is_nonresonant,
        birkhoff_resonant_vector=resonant_vec,
        poincare_series_radius=series_radius,
        poincare_series_divergent=series_divergent,
    )
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                          FASE 2 — INICIO                                █
# █   SECCIONES, MAPAS DE RETORNO Y RESONANCIAS KAM (Tejido Dinámico)       █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 2: Consume el `PoincareFlowFrame` de FASE 1 y despliega:
    • Sección de Poincaré Σ ⊂ T*ℳ y mapa de primer retorno P: Σ → Σ.
    • Número de rotación ν y vectores de enrollamiento.
    • Criterio de solapamiento de resonancias de Chirikov (K ≥ 1).
    • Función de Melnikov para homoclinic tangles.
    • [NEW] Condición Diofántica de Kolmogorov (fracción continua).
    • [NEW] Teorema de Recurrencia de Poincaré.
    • [NEW] Teorema Ergódico de Birkhoff.
    • [NEW] Iteración de Newton superconvergente KAM.
El método de cierre produce un `PoincareDynamicsReport` que alimenta la
FASE 3 (topología global y gobernanza).
"""
# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Estructuras dinámicas
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, eq=False)
class PoincareSectionState:
    r"""
    Estado de la sección de Poincaré Σ = {z : g(z) = 0, ġ(z) > 0}.
    Contiene los cruces detectados y el Jacobiano del mapa de retorno P.
    """
    frame_hash: str
    section_normal: NDArray[np.float64]
    section_offset: float
    crossings: Tuple[NDArray[np.float64], ...]
    return_map_jacobian: NDArray[np.float64]
    rotation_number: float
    winding_vector: Tuple[float, ...]
    is_transversal: bool
    def to_dict(self) -> Dict[str, Any]:
        return {"frame_hash": self.frame_hash,
                "section_normal": self.section_normal.tolist(),
                "section_offset": float(self.section_offset),
                "n_crossings": len(self.crossings),
                "return_map_jacobian": self.return_map_jacobian.tolist(),
                "rotation_number": float(self.rotation_number),
                "winding_vector": list(self.winding_vector),
                "is_transversal": self.is_transversal}
@dataclass(frozen=True, slots=True, eq=False)
class ResonanceOverlapCertificate:
    r"""
    Certificado del criterio de solapamiento de resonancias de Chirikov:
        K = Δω / δω_res
    K ≥ 1 ⇒ destrucción global de toros KAM ⇒ caos determinista.
    """
    overlap_ratio: float                  # K
    primary_resonance_width: float        # δω_res
    distance_between_resonances: float    # Δω
    is_kam_intact: bool
    chaos_threshold: float = _CHIRIKOV_CRITICAL
    def to_dict(self) -> Dict[str, Any]:
        return {"overlap_ratio": float(self.overlap_ratio),
                "primary_resonance_width": float(self.primary_resonance_width),
                "distance_between_resonances": float(self.distance_between_resonances),
                "is_kam_intact": self.is_kam_intact,
                "chaos_threshold": float(self.chaos_threshold)}
@dataclass(frozen=True, slots=True, eq=False)
class MelnikovCertificate:
    r"""
    Certificado de la función de Melnikov:
        M(t₀) = ∫_{-∞}^{+∞} {H₀, H₁}(z₀(t−t₀)) dt
    Si M(t₀) = 0 y M'(t₀) ≠ 0 ⇒ tangencia homoclínica ⇒ caos transitorio.
    """
    m_zero_crossings: Tuple[float, ...]        # raíces t₀ de M(t₀) = 0
    m_prime_at_zeros: Tuple[float, ...]        # M'(t₀) en cada raíz
    has_transversal_homoclinic: bool
    melnikov_amplitude: float                  # sup |M(t)|
    t_domain: Tuple[float, float]
    def to_dict(self) -> Dict[str, Any]:
        return {"m_zero_crossings": list(self.m_zero_crossings),
                "m_prime_at_zeros": list(self.m_prime_at_zeros),
                "has_transversal_homoclinic": self.has_transversal_homoclinic,
                "melnikov_amplitude": float(self.melnikov_amplitude),
                "t_domain": list(self.t_domain)}
# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Motor dinámico
# ──────────────────────────────────────────────────────────────────────────────
class PoincareDynamicsEngine:
    r"""
    Motor de dinámica poincaréana: secciones, mapas de retorno, resonancias,
    funciones de Melnikov, condición diofántica, recurrencia, ergodicidad
    y la iteración de Newton superconvergente de KAM.
    """
    # ── 2.2.1 Sección de Poincaré y mapa de primer retorno ────────────────────
    @staticmethod
    def trace_poincare_section(
        frame: PoincareFlowFrame,
        trajectory: NDArray[np.float64],
        section_normal: NDArray[np.float64],
        section_offset: float = 0.0,
    ) -> PoincareSectionState:
        r"""
        Dada una trayectoria z(t) ∈ T*ℳ, detecta los cruces transversales con
        la hipersuperficie Σ = {z : ⟨n, z⟩ = offset} en dirección ġ > 0.
        Construye el Jacobiano del mapa de retorno P: Σ → Σ por diferencias
        finitas de los cruces consecutivos (proxy Darboux).
        Número de rotación: ν = (1/2π) lim_{N→∞} (1/N) Σᵢ Δθᵢ.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2:
            raise CelestialMechanicsError(
                f"Trayectoria debe ser 2-D [T, D], recibido {traj.shape}"
            )
        n = np.asarray(section_normal, dtype=np.float64).ravel()
        if n.size != traj.shape[1]:
            raise CelestialMechanicsError(
                f"Normal incompatible: {n.size} vs traj dim {traj.shape[1]}"
            )
        g = traj @ n - section_offset
        crossings: List[NDArray[np.float64]] = []
        for i in range(len(g) - 1):
            if g[i] <= 0.0 < g[i + 1]:
                denom = (g[i + 1] - g[i]) if abs(g[i + 1] - g[i]) > EPS else EPS
                alpha = -g[i] / denom
                pt = traj[i] + alpha * (traj[i + 1] - traj[i])
                crossings.append(pt)
        crossings_t = tuple(crossings[:_POINCARE_SECTION_MAX_POINTS])
        if len(crossings_t) >= 3:
            P0 = crossings_t[-3]; P1 = crossings_t[-2]; P2 = crossings_t[-1]
            delta1 = P1 - P0
            delta2 = P2 - P1
            norm_d1 = np.linalg.norm(delta1)
            if norm_d1 > EPS:
                return_map = np.outer(delta2, delta1) / (norm_d1 ** 2 + EPS)
            else:
                return_map = np.eye(traj.shape[1])
        else:
            return_map = np.eye(traj.shape[1])
        if len(crossings_t) >= 2:
            angles = np.arctan2(crossings_t[-1][:len(n) // 2 if len(n) > 1 else 1],
                                crossings_t[0][:len(n) // 2 if len(n) > 1 else 1] + EPS)
            dtheta = np.mod(np.diff(np.concatenate([[0.0], angles])), 2 * np.pi)
            nu = float(np.mean(dtheta) / (2.0 * np.pi))
            winding = tuple(float(x) for x in dtheta)
        else:
            nu = 0.0
            winding = tuple()
        is_transversal = (len(crossings_t) >= 2) and (np.linalg.norm(return_map - np.eye(traj.shape[1])) > _SPECTRAL_TOL)
        return PoincareSectionState(
            frame_hash=frame.frame_hash,
            section_normal=n, section_offset=float(section_offset),
            crossings=crossings_t,
            return_map_jacobian=return_map,
            rotation_number=nu,
            winding_vector=winding,
            is_transversal=is_transversal,
        )
    # ── 2.2.2 Criterio de solapamiento de Chirikov ────────────────────────────
    @staticmethod
    def chirikov_resonance_overlap(
        section: PoincareSectionState,
        primary_resonance_width: float,
        distance_between_resonances: Optional[float] = None,
    ) -> ResonanceOverlapCertificate:
        r"""
        Criterio de Chirikov: K = Δω / δω_res.
        Si Δω no se especifica, se estima a partir del Jacobiano del mapa de
        retorno: Δω ≈ ‖J_P − I‖_F (rigidez de la sección).
        """
        if distance_between_resonances is None:
            J = section.return_map_jacobian
            distance_between_resonances = float(np.linalg.norm(J - np.eye(J.shape[0]), ord='fro'))
        delta_omega = max(distance_between_resonances, EPS)
        if primary_resonance_width <= 0:
            trace_part = float(np.trace(section.return_map_jacobian))
            primary_resonance_width = max(abs(trace_part) / max(len(section.crossings), 1), EPS)
        K = delta_omega / (2.0 * primary_resonance_width)
        is_intact = K < _CHIRIKOV_CRITICAL
        return ResonanceOverlapCertificate(
            overlap_ratio=K,
            primary_resonance_width=float(primary_resonance_width),
            distance_between_resonances=float(distance_between_resonances),
            is_kam_intact=is_intact,
        )
    # ── 2.2.3 Función de Melnikov (detección de homoclinic tangles) ───────────
    @staticmethod
    def melnikov_function(
        h0_poisson_h1: Callable[[float], float],
        t_domain: Tuple[float, float] = (-50.0, 50.0),
        n_samples: int = 512,
        tol: float = _MELNIKOV_TOL,
    ) -> MelnikovCertificate:
        r"""
        Evalúa la función de Melnikov:
            M(t₀) = ∫_{-∞}^{+∞} {H₀, H₁}(z₀(t−t₀)) dt
        Numéricamente se asume que el integrando ya es el paréntesis de Poisson
        evaluado sobre la órbita homoclínica. Detecta ceros con cambio de signo.
        """
        if n_samples < 8:
            raise CelestialMechanicsError("n_samples debe ser ≥ 8")
        t0s = np.linspace(t_domain[0], t_domain[1], n_samples)
        M_vals = np.array([float(h0_poisson_h1(t)) for t in t0s], dtype=np.float64)
        amplitude = float(np.max(np.abs(M_vals)))
        zeros: List[float] = []
        primes: List[float] = []
        for i in range(len(M_vals) - 1):
            if M_vals[i] * M_vals[i + 1] < 0:
                dt = t0s[i + 1] - t0s[i]
                alpha = -M_vals[i] / (M_vals[i + 1] - M_vals[i] + EPS)
                t0 = t0s[i] + alpha * dt
                dM = (M_vals[i + 1] - M_vals[i]) / max(dt, EPS)
                zeros.append(float(t0))
                primes.append(float(dM))
        has_homoclinic = any(abs(p) > tol for p in primes)
        return MelnikovCertificate(
            m_zero_crossings=tuple(zeros),
            m_prime_at_zeros=tuple(primes),
            has_transversal_homoclinic=has_homoclinic,
            melnikov_amplitude=amplitude,
            t_domain=t_domain,
        )
    # ── 2.2.4 Estabilidad de Delone-Hill para jerarquías logísticas ───────────
    @staticmethod
    def delone_hill_stability(
        mass_ratio: float,
        semi_major_ratio: float,
        eccentricity: float,
    ) -> Tuple[bool, float]:
        r"""
        Criterio de estabilidad jerárquica de Delone-Hill:
            (a_in / a_out) ≥ 2.8 · (1 + m_in / m_out)^{2/5} · (1 + e_out)
        Aplicado a jerarquías logísticas: los "cuerpos" son contratos/cartuchos
        y los "semiejes" son márgenes operacionales.
        """
        if mass_ratio < 0 or semi_major_ratio <= 0 or eccentricity < 0:
            raise CelestialMechanicsError(
                f"Parámetros físicos inválidos: μ={mass_ratio}, a={semi_major_ratio}, e={eccentricity}"
            )
        critical_ratio = 2.8 * (1.0 + mass_ratio) ** (2.0 / 5.0) * (1.0 + eccentricity)
        margin = float(semi_major_ratio / critical_ratio)
        return (semi_major_ratio >= critical_ratio), margin
    # ── 2.2.5 NUEVO — Fracción continua y Condición Diofántica de Kolmogorov ──
    @staticmethod
    def continued_fraction_expansion(x: float, n_terms: int = 16) -> Tuple[int, ...]:
        r"""
        Expansión en fracción continua [a₀; a₁, a₂, ...] de x ∈ ℝ, truncada a
        `n_terms`. Los convergentes p_k/q_k son las mejores aproximaciones
        racionales de x (fundamentales en la teoría de pequeños denominadores
        de Poincaré y en la construcción de toros KAM).
        """
        terms: List[int] = []
        val = float(x)
        for _ in range(n_terms):
            a = int(np.floor(val))
            terms.append(a)
            frac = val - a
            if abs(frac) < 1e-14:
                break
            val = 1.0 / frac
        return tuple(terms)
    @staticmethod
    def diophantine_condition_certificate(
        rotation_number: float,
        gamma: float = _DIOPHANTINE_GAMMA_DEFAULT,
        tau: float = _DIOPHANTINE_TAU_DEFAULT,
        n_convergents: int = 16,
    ) -> Tuple[bool, float]:
        r"""
        Verifica la condición diofántica de Kolmogorov-Arnold-Moser:
            |ν − p/q| > γ / q^{2+τ},   ∀ p,q ∈ ℤ, q > 0
        condición necesaria para la persistencia del toro invariante bajo
        perturbación (Teorema KAM). Se evalúa contra los convergentes p_k/q_k
        de la fracción continua de ν — los "peores casos" de aproximación
        racional. Retorna (es_diofántico, margen_mínimo).
        """
        cf = PoincareDynamicsEngine.continued_fraction_expansion(rotation_number, n_convergents)
        num_prev, den_prev = 0, 1
        num_cur, den_cur = 1, 0
        worst_margin = float('inf')
        is_diophantine = True
        for a in cf:
            num_prev, num_cur = num_cur, a * num_cur + num_prev
            den_prev, den_cur = den_cur, a * den_cur + den_prev
            if den_cur == 0:
                continue
            p, q = num_cur, den_cur
            bound = gamma / (abs(q) ** (2.0 + tau))
            diff = abs(rotation_number - p / q)
            margin = diff - bound
            if q > 1:
                worst_margin = min(worst_margin, margin)
                if diff <= bound:
                    is_diophantine = False
        if worst_margin == float('inf'):
            worst_margin = gamma
        return is_diophantine, float(worst_margin)
    # ── 2.2.6 NUEVO — Teorema de Recurrencia de Poincaré ──────────────────────
    @staticmethod
    def poincare_recurrence_certificate(
        trajectory: NDArray[np.float64],
        target_region_center: NDArray[np.float64],
        target_region_radius: float,
    ) -> Tuple[bool, Optional[int], float]:
        r"""
        Verifica el Teorema de Recurrencia de Poincaré: si el flujo preserva la
        medida de Liouville μ y T ⊂ T*ℳ satisface μ(T) > 0, entonces μ-casi
        todo punto de T regresa a T infinitas veces. Este método detecta
        numéricamente el primer tiempo de retorno discreto a la región T,
        parametrizada como bola de radio `target_region_radius` centrada en
        `target_region_center`. Retorna (recurrencia_detectada, Δt_discreto,
        tiempo_de_retorno).
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        c = np.asarray(target_region_center, dtype=np.float64).ravel()
        if traj.ndim != 2 or traj.shape[1] != c.size:
            raise CelestialMechanicsError(
                f"Trayectoria {traj.shape} incompatible con centro {c.shape}"
            )
        dists = np.linalg.norm(traj - c, axis=1)
        inside = dists <= target_region_radius
        entries = np.where(inside)[0]
        if entries.size == 0:
            return False, None, float('inf')
        start = int(entries[0])
        idx = start
        while idx < len(inside) and inside[idx]:
            idx += 1
        if idx >= len(inside):
            return False, None, float('inf')
        return_idx: Optional[int] = None
        for j in range(idx, len(inside)):
            if inside[j]:
                return_idx = j
                break
        if return_idx is None:
            return False, None, float('inf')
        delta = int(return_idx - start)
        return True, delta, float(delta)
    # ── 2.2.7 NUEVO — Teorema Ergódico de Birkhoff ────────────────────────────
    @staticmethod
    def birkhoff_ergodic_average(
        observable_values: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64]]:
        r"""
        Calcula el promedio temporal de Birkhoff a lo largo de una órbita:
            ⟨f⟩_T = (1/T) Σ_{t=0}^{T-1} f(φ_t(z))
        y la secuencia de promedios parciales, cuya convergencia (c.t.p.) hacia
        el promedio espacial ∫f dμ es el contenido del Teorema Ergódico de
        Birkhoff (1931) — generalización rigurosa de la hipótesis ergódica de
        Poincaré-Boltzmann.
        """
        f = np.asarray(observable_values, dtype=np.float64).ravel()
        if f.size == 0:
            raise CelestialMechanicsError("Secuencia de observables vacía")
        cumsum = np.cumsum(f)
        counts = np.arange(1, f.size + 1, dtype=np.float64)
        partial_averages = cumsum / counts
        return float(partial_averages[-1]), partial_averages
    # ── 2.2.8 NUEVO — Iteración de Newton superconvergente KAM ────────────────
    @staticmethod
    def newton_kam_iteration_step(
        conjugacy_error: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        theta_grid: NDArray[np.float64],
        rotation_number: float,
        fourier_cutoff: int = 16,
        small_denominator_cutoff: float = _SMALL_DENOMINATOR_CUTOFF,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Un paso del método de Newton superconvergente de Kolmogorov-Arnold-Moser
        para resolver la ecuación cohomológica linealizada:
            u(θ+ν) − u(θ) = η(θ) − ⟨η⟩
        en el espacio de Fourier, dividiendo por (e^{2πikν} − 1) y regularizando
        los pequeños denominadores vía `small_denominator_cutoff` (condición
        diofántica necesaria, cf. 2.2.5). La convergencia es cuadrática en cada
        iteración, a diferencia de la convergencia lineal de la serie de
        perturbación clásica de Poincaré (cf. 1.2.9).
        Retorna (corrección u, residuo_L2).
        """
        theta = np.asarray(theta_grid, dtype=np.float64).ravel()
        eta = np.asarray(conjugacy_error(theta), dtype=np.float64).ravel()
        N = theta.size
        if N == 0:
            raise CelestialMechanicsError("theta_grid vacío")
        eta_hat = np.fft.fft(eta)
        k = np.fft.fftfreq(N, d=1.0 / N)
        denom = np.exp(2j * np.pi * k * rotation_number) - 1.0
        small = np.abs(denom) < small_denominator_cutoff
        denom_reg = np.where(small, 1.0, denom)
        u_hat = np.where(small, 0.0, eta_hat / denom_reg)
        mask = np.abs(k) <= fourier_cutoff
        u_hat = u_hat * mask
        u = np.real(np.fft.ifft(u_hat))
        residual = float(np.linalg.norm(eta - np.mean(eta)) / np.sqrt(N))
        return u, residual
# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — CIERRE FASE 2: empaquetado del reporte dinámico (semilla FASE 3)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, eq=False)
class PoincareDynamicsReport:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SEMILLA FUNTORIAL DE FASE 2 → FASE 3                                   ║
    ║ Empaqueta el estado de la sección, Chirikov, Melnikov, Delone-Hill, la ║
    ║ condición diofántica, la recurrencia de Poincaré y el promedio         ║
    ║ ergódico de Birkhoff. Es CONSUMIDO por la gobernanza de FASE 3.        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    frame_hash: str
    section_state: PoincareSectionState
    chirikov_certificate: ResonanceOverlapCertificate
    melnikov_certificate: MelnikovCertificate
    delone_hill_stable: bool
    delone_hill_margin: float
    is_globally_integrable: bool
    # ── NUEVO: diofántico / recurrencia / ergódico ────────────────────────────
    diophantine_is_satisfied: bool = True
    diophantine_margin: float = float("inf")
    recurrence_detected: bool = False
    recurrence_time: float = float("inf")
    ergodic_average_value: float = 0.0
    dynamics_hash: str = ""
    def __post_init__(self) -> None:
        if not self.dynamics_hash:
            object.__setattr__(self, "dynamics_hash",
                               MathUtils.stable_hash(self._hash_payload()))
    def _hash_payload(self) -> Dict[str, Any]:
        return {"frame_hash": self.frame_hash,
                "section": self.section_state.to_dict(),
                "chirikov": self.chirikov_certificate.to_dict(),
                "melnikov": self.melnikov_certificate.to_dict(),
                "delone_stable": self.delone_hill_stable,
                "delone_margin": float(self.delone_hill_margin),
                "integrable": self.is_globally_integrable,
                "diophantine_ok": bool(self.diophantine_is_satisfied),
                "diophantine_margin": float(self.diophantine_margin),
                "recurrence_detected": bool(self.recurrence_detected),
                "recurrence_time": float(self.recurrence_time),
                "ergodic_average": float(self.ergodic_average_value)}
    def to_dict(self) -> Dict[str, Any]:
        return {**self._hash_payload(), "dynamics_hash": self.dynamics_hash}
def assemble_poincare_dynamics_report(
    frame: PoincareFlowFrame,
    section_state: PoincareSectionState,
    chirikov_cert: ResonanceOverlapCertificate,
    melnikov_cert: MelnikovCertificate,
    delone_hill_stable: bool,
    delone_hill_margin: float,
    # ── NUEVO: entradas opcionales diofántico / recurrencia / ergódico ────────
    diophantine_is_satisfied: bool = True,
    diophantine_margin: float = float("inf"),
    recurrence_detected: bool = False,
    recurrence_time: float = float("inf"),
    ergodic_average_value: float = 0.0,
) -> PoincareDynamicsReport:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FASE 2 → APERTURA FASE 3                                        ║
    ║ Ensambla el reporte dinámico que será procesado por la topología       ║
    ║ global y la gobernanza soberana. La integrabilidad global ahora exige  ║
    ║ ADEMÁS la condición diofántica (hipótesis KAM completa).               ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    is_integrable = (
        frame.is_kam_stable
        and chirikov_cert.is_kam_intact
        and not melnikov_cert.has_transversal_homoclinic
        and delone_hill_stable
        and diophantine_is_satisfied
    )
    return PoincareDynamicsReport(
        frame_hash=frame.frame_hash,
        section_state=section_state,
        chirikov_certificate=chirikov_cert,
        melnikov_certificate=melnikov_cert,
        delone_hill_stable=delone_hill_stable,
        delone_hill_margin=delone_hill_margin,
        is_globally_integrable=is_integrable,
        diophantine_is_satisfied=diophantine_is_satisfied,
        diophantine_margin=diophantine_margin,
        recurrence_detected=recurrence_detected,
        recurrence_time=recurrence_time,
        ergodic_average_value=ergodic_average_value,
    )
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                          FASE 3 — INICIO                                █
# █    TOPOLOGÍA GLOBAL Y GOBERNANZA SOBERANA (Tejido Categorial)           █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 3: Consume el `PoincareDynamicsReport` de FASE 2 y despliega:
    • Teorema del índice de Poincaré-Hopf (Σ ind_p(X) = χ(ℳ)).
    • Dualidad de Poincaré H^k ≅ H_{n−k} sobre la MIC.
    • [NEW] Desigualdades de Morse (débiles, fuertes, igualdad de Euler).
    • [NEW] No-Squeezing Simpléctico de Gromov (capacidad c_G = πr²).
    • Certificado soberano PoincareSovereignCertificate con Veto de Gromov.
"""
# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.1 — Estructuras topológicas globales
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, eq=True)
class PoincareHopfIndexCertificate:
    r"""
    Certificado del teorema del índice de Poincaré-Hopf:
        Σ_{p ∈ Crit(X)} ind_p(X) = χ(ℳ)
    """
    total_index: int
    euler_characteristic: int
    critical_points_indices: Tuple[int, ...]
    is_hopf_consistent: bool
    residual: int
    def to_dict(self) -> Dict[str, Any]:
        return {"total_index": self.total_index,
                "euler_characteristic": self.euler_characteristic,
                "critical_points_indices": list(self.critical_points_indices),
                "is_hopf_consistent": self.is_hopf_consistent,
                "residual": self.residual}
@dataclass(frozen=True, slots=True, eq=True)
class PoincareDualityCertificate:
    r"""
    Certificado de dualidad de Poincaré sobre la MIC:
        H^k(ℳ; ℝ) ≅ H_{n−k}(ℳ; ℝ)
    Verifica la simetría de los números de Betti b_k = b_{n−k}.
    """
    betti_numbers: Tuple[int, ...]
    is_duality_symmetric: bool
    dimension: int
    pairing_norm: float
    def to_dict(self) -> Dict[str, Any]:
        return {"betti_numbers": list(self.betti_numbers),
                "is_duality_symmetric": self.is_duality_symmetric,
                "dimension": self.dimension,
                "pairing_norm": float(self.pairing_norm)}
@dataclass(frozen=True, slots=True, eq=True)
class MorseInequalityCertificate:
    r"""
    Certificado de las Desigualdades de Morse (M. Morse, 1925-34):
        c_k ≥ b_k                                           (débiles)
        Σ_{i=0}^{k} (-1)^{k-i} c_i ≥ Σ_{i=0}^{k} (-1)^{k-i} b_i   (fuertes)
        Σ_i (-1)^i c_i = Σ_i (-1)^i b_i = χ(ℳ)              (igualdad de Euler)
    donde c_k es el número de puntos críticos de índice k de una función de
    Morse X: ℳ → ℝ, y b_k = dim H_k(ℳ; ℝ). Vínculo directo con Poincaré-Hopf:
    si X es el gradiente de una función de Morse, ind_p(∇X) = (-1)^{ind_Morse(p)}.
    """
    critical_counts: Tuple[int, ...]
    betti_numbers: Tuple[int, ...]
    weak_inequalities_hold: bool
    strong_inequalities_hold: bool
    euler_equality_holds: bool
    def to_dict(self) -> Dict[str, Any]:
        return {"critical_counts": list(self.critical_counts),
                "betti_numbers": list(self.betti_numbers),
                "weak_inequalities_hold": self.weak_inequalities_hold,
                "strong_inequalities_hold": self.strong_inequalities_hold,
                "euler_equality_holds": self.euler_equality_holds}
@dataclass(frozen=True, slots=True, eq=True)
class GromovNonSqueezingCertificate:
    r"""
    Certificado del Teorema de No-Squeezing de Gromov (1985) — el "principio de
    incertidumbre simpléctico": una bola simpléctica B^{2n}(r) se embebe
    simplécticamente en el cilindro Z^{2n}(R) = B²(R) × ℝ^{2n-2} si y sólo si
    r ≤ R, independientemente de n. El invariante certificante es la capacidad
    simpléctica de Gromov c_G(B(r)) = πr² (ancho de Gromov).
    Formaliza rigurosamente el "Veto de Gromov" de la axiomática [A4]/[A14].
    """
    ball_radius: float
    cylinder_radius: float
    gromov_width_ball: float
    gromov_width_cylinder: float
    embedding_permitted: bool
    def to_dict(self) -> Dict[str, Any]:
        return {"ball_radius": float(self.ball_radius),
                "cylinder_radius": float(self.cylinder_radius),
                "gromov_width_ball": float(self.gromov_width_ball),
                "gromov_width_cylinder": float(self.gromov_width_cylinder),
                "embedding_permitted": self.embedding_permitted}
@dataclass(frozen=True, slots=True, eq=True)
class PoincareSovereignCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CERTIFICADO SOBERANO FINAL                                             ║
    ║ Sintetiza las 3 fases: núcleo simpléctico+Birkhoff (F1), dinámica+KAM  ║
    ║ (F2) y topología global+Morse+Gromov (F3). Aplica el Veto de Gromov    ║
    ║ P(x_invalid) ≡ 0 sobre la capacidad simpléctica y las desigualdades    ║
    ║ de Morse, además de Poincaré-Hopf y la dualidad de Poincaré.           ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    frame_hash: str
    dynamics_hash: str
    hopf_certificate: PoincareHopfIndexCertificate
    duality_certificate: PoincareDualityCertificate
    is_kam_stable: bool
    is_globally_integrable: bool
    is_sovereign_viable: bool
    gromov_veto_active: bool
    # ── NUEVO: Morse / Gromov non-squeezing ───────────────────────────────────
    morse_certificate: Optional[MorseInequalityCertificate] = None
    gromov_certificate: Optional[GromovNonSqueezingCertificate] = None
    sovereign_hash: str = ""
    def __post_init__(self) -> None:
        if not self.sovereign_hash:
            object.__setattr__(self, "sovereign_hash",
                               MathUtils.stable_hash(self._hash_payload()))
    def _hash_payload(self) -> Dict[str, Any]:
        return {"frame_hash": self.frame_hash,
                "dynamics_hash": self.dynamics_hash,
                "hopf": self.hopf_certificate.to_dict(),
                "duality": self.duality_certificate.to_dict(),
                "kam_stable": self.is_kam_stable,
                "integrable": self.is_globally_integrable,
                "viable": self.is_sovereign_viable,
                "gromov_veto": self.gromov_veto_active,
                "morse": self.morse_certificate.to_dict() if self.morse_certificate else None,
                "gromov": self.gromov_certificate.to_dict() if self.gromov_certificate else None}
    def to_dict(self) -> Dict[str, Any]:
        return {**self._hash_payload(), "sovereign_hash": self.sovereign_hash}
# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.2 — Motor topológico global
# ──────────────────────────────────────────────────────────────────────────────
class PoincareTopologyEngine:
    r"""
    Motor topológico global: índice de Poincaré-Hopf, dualidad de Poincaré,
    desigualdades de Morse, no-squeezing de Gromov y síntesis soberana.
    """
    # ── 3.2.1 Índice de Poincaré-Hopf ─────────────────────────────────────────
    @staticmethod
    def poincare_hopf_index(
        critical_points_indices: Sequence[int],
        euler_characteristic: int,
        tol: int = 0,
    ) -> PoincareHopfIndexCertificate:
        r"""
        Verifica Σ ind_p(X) = χ(ℳ). Si los índices dados no suman la característica
        de Euler, se marca inconsistencia topológica.
        """
        idx_tuple = tuple(int(i) for i in critical_points_indices)
        total = int(sum(idx_tuple))
        residual = total - int(euler_characteristic)
        is_consistent = abs(residual) <= tol
        return PoincareHopfIndexCertificate(
            total_index=total,
            euler_characteristic=int(euler_characteristic),
            critical_points_indices=idx_tuple,
            is_hopf_consistent=is_consistent,
            residual=residual,
        )
    # ── 3.2.2 Dualidad de Poincaré ────────────────────────────────────────────
    @staticmethod
    def poincare_duality(
        betti_numbers: Sequence[int],
        dimension: Optional[int] = None,
    ) -> PoincareDualityCertificate:
        r"""
        Verifica b_k = b_{n−k} para k ∈ [0, n]. Si no se especifica `dimension`,
        se infiere como len(betti_numbers) − 1.
        """
        b = tuple(int(x) for x in betti_numbers)
        if not b:
            return PoincareDualityCertificate(
                betti_numbers=(), is_duality_symmetric=True,
                dimension=0, pairing_norm=0.0
            )
        n = dimension if dimension is not None else len(b) - 1
        if len(b) != n + 1:
            n = min(n, len(b) - 1)
        b_arr = np.array(b[: n + 1], dtype=np.float64)
        b_rev = b_arr[::-1]
        pairing_norm = float(np.linalg.norm(b_arr - b_rev, ord=2))
        is_symmetric = pairing_norm <= _SPECTRAL_TOL
        return PoincareDualityCertificate(
            betti_numbers=b, is_duality_symmetric=is_symmetric,
            dimension=int(n), pairing_norm=pairing_norm,
        )
    # ── 3.2.3 NUEVO — Desigualdades de Morse ──────────────────────────────────
    @staticmethod
    def morse_inequalities(
        critical_counts: Sequence[int],
        betti_numbers: Sequence[int],
        tol: float = _SPECTRAL_TOL,
    ) -> MorseInequalityCertificate:
        r"""
        Verifica las desigualdades débiles, fuertes y la igualdad de Euler de
        Morse:
            c_k ≥ b_k
            Σ_{i≤k}(-1)^{k-i}c_i ≥ Σ_{i≤k}(-1)^{k-i}b_i
            Σ_i (-1)^i c_i = Σ_i (-1)^i b_i
        """
        c = tuple(int(x) for x in critical_counts)
        b = tuple(int(x) for x in betti_numbers)
        n = max(len(c), len(b))
        c_pad = c + (0,) * (n - len(c))
        b_pad = b + (0,) * (n - len(b))
        weak = all(c_pad[k] >= b_pad[k] for k in range(n))
        strong = True
        for k in range(n):
            lhs = sum(((-1) ** (k - i)) * c_pad[i] for i in range(k + 1))
            rhs = sum(((-1) ** (k - i)) * b_pad[i] for i in range(k + 1))
            if lhs < rhs - tol:
                strong = False
                break
        euler_c = sum(((-1) ** i) * c_pad[i] for i in range(n))
        euler_b = sum(((-1) ** i) * b_pad[i] for i in range(n))
        euler_eq = abs(euler_c - euler_b) <= tol
        return MorseInequalityCertificate(
            critical_counts=c_pad, betti_numbers=b_pad,
            weak_inequalities_hold=weak, strong_inequalities_hold=strong,
            euler_equality_holds=euler_eq,
        )
    # ── 3.2.4 NUEVO — No-Squeezing Simpléctico de Gromov ──────────────────────
    @staticmethod
    def gromov_nonsqueezing_check(
        ball_radius: float,
        cylinder_radius: float,
    ) -> GromovNonSqueezingCertificate:
        r"""
        Verifica el Teorema de No-Squeezing de Gromov:
            B^{2n}(r) ↪_simp Z^{2n}(R) ⟺ r ≤ R
        donde la capacidad (ancho de Gromov) c_G(B(r)) = πr² es el invariante
        simpléctico obstructivo — ningún difeomorfismo que preserve volumen
        puede violar esta desigualdad, SÓLO los simplectomorfismos la respetan.
        """
        if ball_radius < 0 or cylinder_radius < 0:
            raise CelestialMechanicsError("Radios deben ser no negativos")
        width_ball = float(np.pi * ball_radius ** 2)
        width_cyl = float(np.pi * cylinder_radius ** 2)
        permitted = ball_radius <= cylinder_radius + _SPECTRAL_TOL
        return GromovNonSqueezingCertificate(
            ball_radius=float(ball_radius), cylinder_radius=float(cylinder_radius),
            gromov_width_ball=width_ball, gromov_width_cylinder=width_cyl,
            embedding_permitted=permitted,
        )
    # ── 3.2.5 Síntesis soberana con Veto de Gromov ────────────────────────────
    @staticmethod
    def sovereign_synthesis(
        report: PoincareDynamicsReport,
        hopf: PoincareHopfIndexCertificate,
        duality: PoincareDualityCertificate,
        kam_stability_hard: bool = True,
        morse: Optional[MorseInequalityCertificate] = None,
        gromov: Optional[GromovNonSqueezingCertificate] = None,
    ) -> PoincareSovereignCertificate:
        r"""
        Sintetiza las 3 fases. Aplica el Veto Simpléctico de Gromov ampliado:
            P(x_invalid) ≡ 0
        si alguna de las condiciones duraderas falla, incluyendo ahora las
        desigualdades de Morse (débiles) y el no-squeezing de Gromov.
        """
        viable = (
            hopf.is_hopf_consistent
            and duality.is_duality_symmetric
            and (report.is_globally_integrable or not kam_stability_hard)
        )
        if morse is not None:
            viable = viable and morse.weak_inequalities_hold
        if gromov is not None:
            viable = viable and gromov.embedding_permitted
        veto = not viable
        return PoincareSovereignCertificate(
            frame_hash=report.frame_hash,
            dynamics_hash=report.dynamics_hash,
            hopf_certificate=hopf,
            duality_certificate=duality,
            is_kam_stable=report.chirikov_certificate.is_kam_intact,
            is_globally_integrable=report.is_globally_integrable,
            is_sovereign_viable=viable,
            gromov_veto_active=veto,
            morse_certificate=morse,
            gromov_certificate=gromov,
        )
# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — CIERRE SOBERANO: método unificado de gobernanza
# ──────────────────────────────────────────────────────────────────────────────
def sovereign_poincare_governance(
    frame: PoincareFlowFrame,
    report: PoincareDynamicsReport,
    critical_points_indices: Sequence[int] = (1,),
    euler_characteristic: int = 1,
    betti_numbers: Sequence[int] = (1,),
    # ── NUEVO: entradas opcionales Morse / Gromov non-squeezing ───────────────
    critical_point_counts_per_index: Optional[Sequence[int]] = None,
    symplectic_ball_radius: Optional[float] = None,
    symplectic_cylinder_radius: Optional[float] = None,
) -> PoincareSovereignCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE SOBERANO DE LAS 3 FASES                                         ║
    ║ Ejecuta la gobernanza topológica global consumiendo el reporte         ║
    ║ dinámico (F2) y el marco simpléctico (F1). Incorpora, si se proveen,   ║
    ║ las desigualdades de Morse y el no-squeezing de Gromov. Retorna el     ║
    ║ certificado soberano con Veto de Gromov aplicado.                      ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    hopf = PoincareTopologyEngine.poincare_hopf_index(
        critical_points_indices, euler_characteristic
    )
    duality = PoincareTopologyEngine.poincare_duality(betti_numbers)
    morse: Optional[MorseInequalityCertificate] = None
    if critical_point_counts_per_index is not None:
        morse = PoincareTopologyEngine.morse_inequalities(
            critical_point_counts_per_index, betti_numbers
        )
    gromov: Optional[GromovNonSqueezingCertificate] = None
    if symplectic_ball_radius is not None and symplectic_cylinder_radius is not None:
        gromov = PoincareTopologyEngine.gromov_nonsqueezing_check(
            symplectic_ball_radius, symplectic_cylinder_radius
        )
    cert = PoincareTopologyEngine.sovereign_synthesis(
        report=report, hopf=hopf, duality=duality, morse=morse, gromov=gromov
    )
    if cert.gromov_veto_active:
        logger.warning(
            "[GROMOV_VETO] Certificado soberano NO viable: "
            "Hopf=%s, Duality=%s, Integrable=%s, Morse=%s, Gromov=%s",
            hopf.is_hopf_consistent, duality.is_duality_symmetric,
            report.is_globally_integrable,
            morse.weak_inequalities_hold if morse else "N/A",
            gromov.embedding_permitted if gromov else "N/A",
        )
    return cert
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █           INTEGRACIÓN CON EL MICAgent SOBERANO (facade final)          █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
class MICCelestialGovernanceFacade:
    r"""
    Fachada de integración: orquesta las 3 fases sobre el MICAgent existente.
    Los métodos auditores originales (Poincaré-Liouville-Novikov) permanecen,
    y esta fachada añade la maquinaria celeste completa: Birkhoff, condición
    diofántica, recurrencia, ergodicidad, Newton-KAM, Morse y Gromov.
    """
    def __init__(self) -> None:
        self._kernel = PoincareSymplecticKernel
        self._dynamics = PoincareDynamicsEngine
        self._topology = PoincareTopologyEngine
    def full_audit(
        self,
        position: NDArray[np.float64],
        momentum: NDArray[np.float64],
        hamiltonian_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
        trajectory: NDArray[np.float64],
        section_normal: NDArray[np.float64],
        h0_poisson_h1: Callable[[float], float],
        linearization_M: Optional[NDArray[np.float64]] = None,
        delone_params: Tuple[float, float, float] = (0.1, 10.0, 0.05),
        critical_points_indices: Sequence[int] = (1,),
        euler_characteristic: int = 1,
        betti_numbers: Sequence[int] = (1, 0, 1),
        # ── NUEVO: parámetros opcionales de la maquinaria celeste extendida ───
        equilibrium_hessian: Optional[NDArray[np.float64]] = None,
        recurrence_region: Optional[Tuple[NDArray[np.float64], float]] = None,
        critical_point_counts_per_index: Optional[Sequence[int]] = None,
        symplectic_ball_radius: Optional[float] = None,
        symplectic_cylinder_radius: Optional[float] = None,
    ) -> PoincareSovereignCertificate:
        r"""
        Ejecuta el ciclo completo FASE 1 → FASE 2 → FASE 3:
          1. Construye el PoincareFlowFrame + Birkhoff (F1).
          2. Traza la sección de Poincaré, Chirikov, Melnikov, condición
             diofántica, recurrencia y promedio ergódico (F2).
          3. Verifica Delone-Hill, Morse, Gromov y sintetiza el certificado
             soberano (F3).
        """
        # ── FASE 1 ────────────────────────────────────────────────────────────
        frame = build_poincare_flow_frame(
            position=position, momentum=momentum,
            hamiltonian_fn=hamiltonian_fn,
            linearization_M=linearization_M,
            equilibrium_hessian=equilibrium_hessian,
        )
        logger.info(
            "[F1] Marco construido: hash=%s, KAM_ρ=%.4f, KAM_stable=%s, "
            "Birkhoff_nonresonant=%s",
            frame.frame_hash[:12], frame.kam_torus_index, frame.is_kam_stable,
            frame.is_birkhoff_nonresonant,
        )
        # ── FASE 2 ────────────────────────────────────────────────────────────
        section = self._dynamics.trace_poincare_section(
            frame=frame, trajectory=trajectory, section_normal=section_normal,
        )
        chirikov = self._dynamics.chirikov_resonance_overlap(section=section,
                                                             primary_resonance_width=0.0)
        melnikov = self._dynamics.melnikov_function(h0_poisson_h1=h0_poisson_h1)
        delone_ok, delone_margin = self._dynamics.delone_hill_stability(*delone_params)
        diophantine_ok, diophantine_margin = self._dynamics.diophantine_condition_certificate(
            rotation_number=section.rotation_number
        )
        recurrence_detected, _, recurrence_time = (False, None, float('inf'))
        if recurrence_region is not None:
            center, radius = recurrence_region
            recurrence_detected, _, recurrence_time = self._dynamics.poincare_recurrence_certificate(
                trajectory=trajectory, target_region_center=center, target_region_radius=radius,
            )
        ergodic_value = 0.0
        if trajectory.ndim == 2 and trajectory.shape[0] > 0:
            observable = np.linalg.norm(trajectory, axis=1)
            ergodic_value, _ = self._dynamics.birkhoff_ergodic_average(observable)
        report = assemble_poincare_dynamics_report(
            frame=frame, section_state=section,
            chirikov_cert=chirikov, melnikov_cert=melnikov,
            delone_hill_stable=delone_ok, delone_hill_margin=delone_margin,
            diophantine_is_satisfied=diophantine_ok, diophantine_margin=diophantine_margin,
            recurrence_detected=recurrence_detected, recurrence_time=recurrence_time,
            ergodic_average_value=ergodic_value,
        )
        logger.info(
            "[F2] Dinámica: K_chirikov=%.4f, homoclínico=%s, Delone_ok=%s, "
            "Diofántico_ok=%s, recurrencia=%s, integrable=%s",
            chirikov.overlap_ratio, melnikov.has_transversal_homoclinic,
            delone_ok, diophantine_ok, recurrence_detected,
            report.is_globally_integrable,
        )
        # ── FASE 3 ────────────────────────────────────────────────────────────
        cert = sovereign_poincare_governance(
            frame=frame, report=report,
            critical_points_indices=critical_points_indices,
            euler_characteristic=euler_characteristic,
            betti_numbers=betti_numbers,
            critical_point_counts_per_index=critical_point_counts_per_index,
            symplectic_ball_radius=symplectic_ball_radius,
            symplectic_cylinder_radius=symplectic_cylinder_radius,
        )
        logger.info(
            "[F3] Soberano: viable=%s, gromov_veto=%s, hash=%s",
            cert.is_sovereign_viable, cert.gromov_veto_active,
            cert.sovereign_hash[:12],
        )
        return cert
# ==============================================================================
# EXPORTACIÓN PÚBLICA
# ==============================================================================
__all__ = [
    # Excepciones
    "MICAgentError", "TopologicalInvariantError", "StratumResolutionError",
    "ContractValidationError", "ClosureViolationError", "AlgebraicVetoError",
    "TOONCompressionError", "SiloAccessError", "ProjectionError",
    "FunctorialityError", "CelestialMechanicsError",
    # Enumeraciones
    "ImpedanceMatchStatus", "ValidationSeverity", "KreinSignature",
    # Dataclasses base
    "SchemaValidationResult", "CategoricalEqualizerSeed", "TOONDocument",
    "SiloAContract", "SiloBCartridge", "PoincareMICAdjunctionCertificate",
    # FASE 1
    "PoincareActionAnglePair", "KreinSpectralDecomposition", "PoincareFlowFrame",
    "PoincareSymplecticKernel", "build_poincare_flow_frame",
    # FASE 2
    "PoincareSectionState", "ResonanceOverlapCertificate", "MelnikovCertificate",
    "PoincareDynamicsEngine", "PoincareDynamicsReport",
    "assemble_poincare_dynamics_report",
    # FASE 3
    "PoincareHopfIndexCertificate", "PoincareDualityCertificate",
    "MorseInequalityCertificate", "GromovNonSqueezingCertificate",
    "PoincareSovereignCertificate", "PoincareTopologyEngine",
    "sovereign_poincare_governance",
    # Fachada
    "MICCelestialGovernanceFacade",
    # Utilidades
    "MathUtils", "normalize_stratum", "python_type_matches", "compute_json_path",
    # Constantes
    "MAX_AUDIT_TRAIL_SIZE", "TOON_START_MARKER", "TOON_END_MARKER",
    "TOON_FIELD_SEPARATOR", "ENCAPSULATION_PROTOCOL_VERSION",
    "EPS", "ALGEBRAIC_TOL", "FLOAT_COMPARISON_TOL", "MAX_TENSOR_RANK",
    "MAX_COMPRESSION_RATIO", "MIN_COMPRESSION_RATIO",
    "_CHIRIKOV_CRITICAL", "_KREIN_ELLIPTIC_TOL", "_MELNIKOV_TOL",
    "_POINCARE_SECTION_MAX_POINTS", "_BIRKHOFF_DEFAULT_ORDER",
    "_DIOPHANTINE_GAMMA_DEFAULT", "_DIOPHANTINE_TAU_DEFAULT",
]
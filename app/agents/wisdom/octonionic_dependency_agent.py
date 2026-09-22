# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Octonionic Dependency Agent (Soberano de Calibre Octoniónico 8D)    ║
║ Ruta   : app/agents/wisdom/octonionic_dependency_agent.py                    ║
║ Versión: 3.1.0-Doctoral-OODA-Heyting-Banach-Artin-Moufang-Hodge-Nested3      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Soberano de Calibre Octoniónico 8D (Octonionic Dependency Agent).

Este módulo implementa el agente supervisor ciber-físico soberano para la gobernanza de lazo cerrado
sobre el resolutor de dependencias en el álgebra de Cayley-Dickson \mathbb{O} (\mathbb{R}^8).
Audita síncronamente la tríada transaccional crítica (Contratista P_1, Proveedor P_2, Interventor P_3) \in \mathbb{O}^3
mediante tres fases anidadas en el ciclo OODA sobre la cadena de Heyting \Omega_3.

DEFINICIÓN FORMAL Y OPERATORIA:
    El agente inmerge la tríada \mathbb{O}^3 y evalúa sus propiedades geométricas, algebraicas y topológicas:

    1. Fase 1 (Observe - Phase1_OctonionicObservation):
       - Geometría de Banach en (\mathbb{R}^8, \|\cdot\|_p): Evaluación de normas \ell^1, \ell^2, \ell^\infty,
         desigualdad de Hölder \|x\|_2^2 \le \|x\|_1 \|x\|_\infty, índice de escasez de Hoyer \mathcal{H}(x) \in [0,1]
         y coeficiente de distorsión convexa \kappa_B(x) = \frac{\|x\|_1 \|x\|_\infty}{\|x\|_2^2} \ge 1.
       - Construcción de estados OctonionicState y generación del kernel de observación con sello SHA-256.

    2. Fase 2 (Orient - Phase2_OctonionicOrientation):
       - Álgebra de Malcev y Estructura de Fano: En \text{Im}(\mathbb{O}) \cong \mathbb{R}^7, el Jacobiator generalizado
         J(u,v,w) = -6 [u,v,w] y la 3-forma asociativa de G_2 sobre PG(2,2): \phi(a,b,c) = \langle a, b \times c \rangle.
       - Evaluación del asociador trilateral A_3(a,b,c) = (ab)c - a(bc) y residuos de las identidades de Artin y Moufang.
       - Topología Espectral de Hodge-Laplace en K_3: Conectividad de Fiedler \lambda_2, resistencia efectiva de Kirchhoff R_K
         y disipación exergética de Dirichlet \mathcal{E}_D = \text{Tr}(P^T L P).
       - Rampa Graduada de de Rham: Asignación de verdad \omega_{\text{asoc}} \in \{0, \frac{1}{2}, 1\} según
         \|[a,b,c]\| frente al umbral elástico \tau_{\max}.

    3. Fase 3 (Act - Phase3_OODAActuator):
       - Inferencia de Heyting \Omega_3: Evaluador global \mathbf{V} = \bigwedge p_k.
       - Ventana Modal de Gracia \Gamma y Override HMAC: Autenticación de tokens en tiempo constante.
       - Interlock Ciber-Físico Crowbar BT151: Ante \mathbf{V} = \text{VETOED}, simula el disparo del tiristor BT151/GPIO14 en IRAM (< 400 ns).

AXIOMAS E INVARIANTES RIGUROSOS:
    - Axioma I (Equivalencia Métrica de Banach en \mathbb{R}^8):
      1 \le \frac{\|x\|_1}{\|x\|_2} \le \sqrt{8}, \quad 1 \le \frac{\|x\|_2}{\|x\|_\infty} \le \sqrt{8}, \quad 1 \le \frac{\|x\|_1}{\|x\|_\infty} \le 8 \quad \forall x \neq 0.
    - Axioma II (Invariante de Calibración G_2 en Fano PG(2,2)):
      \phi(a,b,c) = \langle a, b \times c \rangle es totalmente antisimétrica e invariante bajo la acción del grupo de Lie G_2 = \text{Aut}(\mathbb{O}).
    - Axioma III (Presupuesto de Actuación Crowbar en IRAM):
      t_{\text{act}} \in [382, 399]\text{ ns} < 400\text{ ns} \quad \text{ante colapso a VETOED}.
    - Invariante I (Invarianza de Cierre OODA):
      Phase3 ⊏ Phase2 ⊏ Phase1 \implies \text{Act} \circ \text{Orient} \circ \text{Observe} es la única vía de generación de certificados válidos.

IMPACTO EJECUTIVO DE NEGOCIO ("DOLOR Y DINERO"):
    En esquemas de contratación trilateral, licitaciones complejas y acuerdos multi-actor, la falta de coordinación o la colusión no detectable entre partes genera desviaciones presupuestarias masivas, incumplimientos de entrega y litigios prolongados.
    El Octonionic Dependency Agent detecta al instante cualquier inconsistencia no asociativa o falta de alineación entre Contratista, Proveedor e Interventor. Al bloquear automáticamente contratos defectuosos o activar alertas de veto suave en tiempo real, se evitan pérdidas multimillonarias por ejecuciones fallidas, se garantizan los acuerdos nivel de servicio (SLA) y se protege la transparencia corporativa.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AbstractSet, Any, Callable, Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

# ───────────────────────────────────────────────────────────────────────────────
# Compatibilidad categorial con el ecosistema APU
# ───────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:  # type: ignore[no-redef]
        r"""Morfismo base categorial de compatibilidad."""

    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        r"""Error de invariante topológico en la variedad."""

# ───────────────────────────────────────────────────────────────────────────────
# Importación del resolutor octoniónico de calibre y DTOs
# ───────────────────────────────────────────────────────────────────────────────
try:
    from app.core.octonionic_dependency_resolver import (
        HeytingVerdict,
        KBNSummationKernel,
        OctonionicAuditCertificate,
        OctonionicDependencyResolver,
        OctonionicDimensionError,
        OctonionicEngineError,
        OctonionicNumericalSingularityError,
        OctonionicOrientationReport,
        OctonionicState,
        OctonionicThresholds,
        OctonionicTriadReport,
    )
except ImportError:
    try:
        from octonionic_dependency_resolver import (
            HeytingVerdict,
            KBNSummationKernel,
            OctonionicAuditCertificate,
            OctonionicDependencyResolver,
            OctonionicDimensionError,
            OctonionicEngineError,
            OctonionicNumericalSingularityError,
            OctonionicOrientationReport,
            OctonionicState,
            OctonionicThresholds,
            OctonionicTriadReport,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar octonionic_dependency_resolver. Asegure la presencia "
            "del módulo en app/core/ o en el PYTHONPATH."
        ) from exc


logger = logging.getLogger("APU.Agents.Wisdom.OctonionicDependencyAgent")

__version__: Final[str] = (
    "3.1.0-Doctoral-OODA-Heyting-Banach-Artin-Moufang-Hodge-Nested3"
)

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_OCTONION_DIM: Final[int] = 8
_BANACH_SQRT8: Final[float] = float(math.sqrt(8.0))
_BANACH_CLAMP_EPS: Final[float] = 100.0 * _MACHINE_EPS

_VERDICT_COHERENT: Final[str] = "COHERENT"
_VERDICT_DEGRADED: Final[str] = "DEGRADED"
_VERDICT_VETOED: Final[str] = "VETOED"

_PHASE_NAME: Final[str] = "G_WISDOM_OCTONIONIC_SUTURATED_V3"

_DEFAULT_OVERRIDE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "AUT_POS_SABIDURIA_777",
        "OVERRIDE_NON_ASSOCIATIVE_IDU_2026",
        "HMAC_SUTURA_FOCK_SECURE",
    }
)

_RAMPA_SOFT: Final[float] = 0.30
_RAMPA_HARD: Final[float] = 0.50
_CROWBAR_GPIO: Final[str] = "GPIO14"
_CROWBAR_DEVICE: Final[str] = "BT151-800R"


# ═══════════════════════════════════════════════════════════════════════════════
# §A. ÁLGEBRA DE HEYTING \Omega_3 Y METROLOGÍA DE ESCALARES
# ═══════════════════════════════════════════════════════════════════════════════
def _heyting_meet(a: float, b: float) -> float:
    r"""Ínfimo en \Omega_3: a \wedge b = \min(a, b)."""
    return float(min(a, b))


def _heyting_implies(a: float, b: float) -> float:
    r"""Implicación de Heyting: a \to b = 1 si a \le b, else b."""
    return 1.0 if a <= b + _MACHINE_EPS else float(b)


def _heyting_not(a: float) -> float:
    r"""Negación intuicionista: \neg a = a \to 0."""
    return _heyting_implies(a, 0.0)


def _heyting_meet_all(predicates: Iterable[float]) -> float:
    """Calcula el meet de una secuencia de valores en el retículo; neutro: 1.0."""
    acc = 1.0
    for val in predicates:
        acc = min(acc, float(val))
        if acc <= 0.0:
            break
    return float(acc)


def _canonical_bytes(part: Any) -> bytes:
    r"""Serialización determinista Little-Endian IEEE-754 de escalares y tensores."""
    if isinstance(part, np.ndarray):
        arr = np.ascontiguousarray(part, dtype=np.float64)
        header = np.array(arr.shape, dtype="<i8").tobytes()
        return header + np.asarray(arr, dtype="<f8").tobytes(order="C")
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    if isinstance(part, (int, float, bool, np.generic)):
        x = float(part)
        if math.isnan(x):
            return b"\x7fNAN\x00\x00\x00"
        if math.isinf(x):
            return b"\x7fPINF\x00\x00" if x > 0.0 else b"\x7fNINF\x00\x00"
        if x == 0.0:
            return struct.pack("<d", 0.0)
        return struct.pack("<d", x)
    return repr(part).encode("utf-8")


def _immutable(array: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Garantiza la inmutabilidad física en memoria continua contigua."""
    out = np.array(array, dtype=dtype or np.float64, copy=True, order="C")
    out.setflags(write=False)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# §B. DTOs INMUTABLES DEL AGENTE SOBERANO DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class BanachRegularityReport:
    r"""
    Espectro completo de regularidad en el espacio de Banach (\mathbb{R}^8, \|\cdot\|_p).
    Audita las tres normas canónicas, Hölder, escasez de Hoyer y distorsión convexa \kappa_B.
    """

    l1_norm: float
    l2_norm: float
    linf_norm: float
    l1_l2_ratio: float
    raw_l1_l2_ratio: float
    hoyer_sparsity: float
    holder_defect: float
    banach_distortion: float
    is_null: bool
    is_within_theoretical_bounds: bool
    is_clamped: bool = False


@dataclass(frozen=True, slots=True)
class OctonionicAgentThresholds:
    r"""Fronteras de gobernanza y tolerancias analíticas inmutables para el agente."""

    tolerance: float = 1e-12
    safety_margin: float = 1.0
    grace_period_seconds: float = 3600.0
    rampa_soft: float = _RAMPA_SOFT
    rampa_hard: float = _RAMPA_HARD
    hurwitz_tolerance: float = 1e-9
    artin_tolerance: float = 1e-8
    moufang_tolerance: float = 1e-8
    fiedler_min: float = 1e-4
    kirchhoff_max: float = 1e6
    dirichlet_max: float = 1e5

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            val = float(getattr(self, field))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"El umbral {field} debe ser finito y estrictamente positivo.")
        if self.rampa_soft >= self.rampa_hard:
            raise ValueError("Inconsistencia: rampa_soft debe ser estrictamente menor que rampa_hard.")
        if self.rampa_hard > 1.0:
            raise ValueError("Inconsistencia: rampa_hard no puede exceder la unidad.")


@dataclass(frozen=True, slots=True)
class OctonionicObservationKernel:
    r"""
    OBJETO TERMINAL DE LA FASE 1 (OBSERVE) / INICIAL DE LA FASE 2 (ORIENT).
    Expediente inmutable de observación trilateral, espectro de Banach,
    proyecciones polares y resumen criptográfico SHA-256.
    """

    contractor_state: OctonionicState
    supplier_state: OctonionicState
    interventor_state: OctonionicState
    banach_ratio_contractor: float
    banach_ratio_supplier: float
    banach_ratio_interventor: float
    cryptographic_seal: str

    contractor_spectrum: BanachRegularityReport
    supplier_spectrum: BanachRegularityReport
    interventor_spectrum: BanachRegularityReport
    polar_contractor: np.ndarray
    polar_supplier: np.ndarray
    polar_interventor: np.ndarray
    null_party_count: int = 0
    all_banach_regular: bool = True


@dataclass(frozen=True, slots=True)
class OctonionicOrientationState:
    r"""
    OBJETO TERMINAL DE LA FASE 2 (ORIENT) / INICIAL DE LA FASE 3 (ACT).
    Expediente físico y topológico multivectorial de la tríada octoniónica.
    Sintetiza Hurwitz, asociador [a,b,c], Artin, Moufang, Malcev, Fano,
    Hodge-Laplace en K_3 y preclasificación en \Omega_3.
    """

    kernel: OctonionicObservationKernel
    associator_norm: float
    composition_error: float
    is_associative_stable: bool
    is_cfl_stable: bool

    composition_relative_error: float = 0.0
    associator_relative_norm: float = 0.0
    cota_limite: float = 0.0

    associator_vector: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    composition_tolerance: float = 0.0
    composition_hard_limit: float = 0.0
    associator_soft_start: float = 0.0
    associator_hard_limit: float = 0.0

    omega_hard: float = 1.0
    omega_cfl: float = 1.0
    omega_asoc: float = 1.0
    omega_banach: float = 1.0
    omega_hodge: float = 1.0
    omega_pre: float = 1.0

    artin_residual: float = 0.0
    moufang_residual: float = 0.0
    malcev_residual: float = 0.0
    fano_3form_value: float = 0.0

    laplacian_connectivity: float = 0.0
    laplacian_spectral_gap: float = 0.0
    kirchhoff_index: float = math.inf
    dirichlet_exergy: float = 0.0

    triad_seal: str = ""
    hard_composition: bool = False
    hard_associator: bool = False
    soft_composition: bool = False
    soft_associator: bool = False


@dataclass(frozen=True, slots=True)
class HeytingDecision:
    r"""Resultado formal de la inferencia lógica en \Omega_3 con modalidad \Gamma."""

    verdict: str
    lattice_value: int
    is_soft_veto: bool
    is_hard_veto: bool
    time_grace_remaining: float
    reasons: Tuple[str, ...]
    conjuncts: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""Informe físico de interrupción ciber-física Crowbar BT151 en IRAM (< 400 ns)."""

    interlock_fired: bool
    actuation_latency_ns: float
    gpio: str
    device: str
    seed_sha256: str
    gate_charge_injected_nc: float


@dataclass(frozen=True, slots=True)
class OctonionicAgentCertificate:
    r"""
    CERTIFICADO TERMINAL INMUTABLE DE GOBERNANZA OCTONIÓNICA.
    Sello absoluto de calibración, regularidad de calibre y veto ciber-físico.
    """

    phase: str
    heyting_verdict: str
    associator_norm: float
    is_associative_stable: bool
    composition_error: float
    is_surgery_active: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    time_grace_remaining: float
    digital_signature_sha256: str

    composition_relative_error: float = 0.0
    associator_relative_norm: float = 0.0
    banach_ratio_contractor: float = 0.0
    banach_ratio_supplier: float = 0.0
    banach_ratio_interventor: float = 0.0
    contractor_norm: float = 0.0
    supplier_norm: float = 0.0
    interventor_norm: float = 0.0
    heyting_omega: float = 1.0
    omega_pre: float = 1.0
    artin_residual: float = 0.0
    moufang_residual: float = 0.0
    malcev_residual: float = 0.0
    fano_3form_value: float = 0.0
    laplacian_connectivity: float = 0.0
    kirchhoff_index: float = 0.0
    dirichlet_exergy: float = 0.0
    observation_seal: str = ""
    hoyer_sparsity_max: float = 0.0
    null_party_count: int = 0
    reasons: Tuple[str, ...] = ()
    agent_version: str = __version__


# ═══════════════════════════════════════════════════════════════════════════════
# §C. FASE 1 — OBSERVE: SANEAMIENTO, REGULARIDAD DE BANACH Y POLARIDAD 8D
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_OctonionicObservation:
    r"""
    FASE 1: Observe.
    Categoría Functorial: (\mathbb{R}^8)^3 \longrightarrow \mathbf{OctonionicObservationKernel}.

    Responsabilidades axiomáticas:
      1. Ingesta y validación dimensional estricta en \mathbb{R}^8.
      2. Saneamiento de ceros con signo (-0.0 \to +0.0) para unicidad de hash.
      3. Auditoría analítica de la equivalencia de normas de Banach (\ell^1, \ell^2, \ell^\infty),
         Hölder, escasez de Hoyer y distorsión convexa \kappa_B(x).
      4. Construcción de estados OctonionicState mediante el resolutor en la FPU.
      5. Emisión del sello canónico SHA-256 de sesión.

    Morfismo Terminal: `synthesize_observation_kernel`.
    Su codominio constituye el germen de entrada exclusivo de la Fase 2.
    """

    __slots__ = ("_tol", "_safety_margin", "_grace_limit", "_resolver", "_thresholds")

    def __init__(
        self,
        *,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        resolver: Optional[OctonionicDependencyResolver] = None,
        thresholds: Optional[OctonionicAgentThresholds] = None,
        **kwargs: Any,
    ) -> None:
        self._tol: Final[float] = float(tolerance)
        self._safety_margin: Final[float] = float(safety_margin)
        self._grace_limit: Final[float] = float(grace_period_seconds)

        if not math.isfinite(self._tol) or self._tol <= 0.0:
            raise ValueError("tolerance debe ser finito y estrictamente positivo.")
        if not math.isfinite(self._safety_margin) or self._safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser finito y estrictamente positivo.")
        if not math.isfinite(self._grace_limit) or self._grace_limit <= 0.0:
            raise ValueError("grace_period_seconds debe ser finito y estrictamente positivo.")

        self._thresholds: Final[OctonionicAgentThresholds] = (
            thresholds
            or OctonionicAgentThresholds(
                tolerance=self._tol,
                safety_margin=self._safety_margin,
                grace_period_seconds=self._grace_limit,
            )
        )

        if resolver is None:
            self._resolver: Final[OctonionicDependencyResolver] = (
                OctonionicDependencyResolver(
                    tolerance=self._tol,
                    grace_period_seconds=self._grace_limit,
                )
            )
        else:
            self._resolver = resolver

        try:
            super().__init__(**kwargs)
        except TypeError:
            super().__init__()

    def _relative_tolerance(self, scale: float = 1.0) -> float:
        return max(self._tol, 10.0 * _MACHINE_EPS * max(1.0, float(scale)))

    def _sha256_payload(self, *parts: Any) -> str:
        sha = hashlib.sha256()
        for part in parts:
            sha.update(_canonical_bytes(part))
        return sha.hexdigest()

    def _validate_vector8(self, S: Sequence[float], name: str = "vector octoniónico") -> np.ndarray:
        arr = np.asarray(S, dtype=np.float64)
        if arr.ndim != 1 or arr.shape != (_OCTONION_DIM,):
            raise OctonionicDimensionError(
                f"{name} debe residir estrictamente en R^{_OCTONION_DIM}. Obtenido: shape={arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise OctonionicNumericalSingularityError(f"{name} contiene singularidades (NaN o Inf).")
        return arr

    @staticmethod
    def _sanitize_signed_zeros(S: np.ndarray) -> np.ndarray:
        r"""Sanea ceros con signo: -0.0 \to +0.0 para estabilidad determinista."""
        arr = np.array(S, dtype=np.float64, copy=True, order="C")
        arr = np.where(arr == 0.0, 0.0, arr)
        arr.setflags(write=False)
        return arr

    def _polar_unit(self, S: np.ndarray, norm2: float) -> np.ndarray:
        r"""Proyección polar x \mapsto x / \|x\|_2 (el vector nulo se proyecta en 0)."""
        if norm2 <= _MACHINE_EPS:
            return np.zeros(_OCTONION_DIM, dtype=np.float64)
        return _immutable(S / norm2)

    def evaluate_banach_spectrum(
        self,
        S: Sequence[float],
        name: str = "vector de Banach",
    ) -> BanachRegularityReport:
        r"""
        Certifica las relaciones analíticas de equivalencia en (\mathbb{R}^8, \|\cdot\|_p):
          1 \le \frac{\|x\|_1}{\|x\|_2} \le \sqrt{8}, \quad
          \|x\|_2^2 \le \|x\|_1 \|x\|_\infty, \quad
          \kappa_B(x) = \frac{\|x\|_1 \|x\|_\infty}{\|x\|_2^2} \ge 1.
        """
        arr = self._sanitize_signed_zeros(self._validate_vector8(S, name))

        l1_norm = KBNSummationKernel.sum(np.abs(arr))
        l2_norm = KBNSummationKernel.norm(arr)
        linf_norm = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        l2_sq = l2_norm * l2_norm

        is_null = bool(l2_norm <= _MACHINE_EPS)

        if is_null:
            return BanachRegularityReport(
                l1_norm=0.0,
                l2_norm=0.0,
                linf_norm=0.0,
                l1_l2_ratio=1.0,
                raw_l1_l2_ratio=1.0,
                hoyer_sparsity=0.0,
                holder_defect=0.0,
                banach_distortion=1.0,
                is_null=True,
                is_within_theoretical_bounds=True,
                is_clamped=False,
            )

        raw_ratio = l1_norm / max(l2_norm, _MACHINE_EPS)
        ratio = raw_ratio
        is_clamped = False

        if ratio < 1.0:
            if ratio >= 1.0 - _BANACH_CLAMP_EPS:
                ratio = 1.0
            else:
                logger.warning("Ratio Banach inferior a 1.0 (%.6e) en %s; clampeado a 1.0.", ratio, name)
                ratio = 1.0
            is_clamped = True
        elif ratio > _BANACH_SQRT8:
            if ratio <= _BANACH_SQRT8 + _BANACH_CLAMP_EPS:
                ratio = _BANACH_SQRT8
            else:
                logger.warning("Ratio Banach superior a √8 (%.6e) en %s; clampeado a √8.", ratio, name)
                ratio = _BANACH_SQRT8
            is_clamped = True

        span = _BANACH_SQRT8 - 1.0
        hoyer = float((_BANACH_SQRT8 - ratio) / span) if span > 0.0 else 0.0
        hoyer = max(0.0, min(1.0, hoyer))

        holder_defect = float(l1_norm * linf_norm - l2_sq)
        if holder_defect < 0.0 and holder_defect > -self._relative_tolerance(l2_sq):
            holder_defect = 0.0

        kappa = (l1_norm * linf_norm) / max(l2_sq, _MACHINE_EPS)
        is_valid = (ratio >= 1.0 - 1e-12) and (ratio <= _BANACH_SQRT8 + 1e-12) and (holder_defect >= -1e-12)

        return BanachRegularityReport(
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            l1_l2_ratio=ratio,
            raw_l1_l2_ratio=raw_ratio,
            hoyer_sparsity=hoyer,
            holder_defect=holder_defect,
            banach_distortion=kappa,
            is_null=False,
            is_within_theoretical_bounds=is_valid,
            is_clamped=is_clamped,
        )

    def evaluate_banach_regularity(self, S: Sequence[float]) -> float:
        """API pública heredada: retorna el ratio ℓ¹/ℓ²."""
        return self.evaluate_banach_spectrum(S).l1_l2_ratio

    def synthesize_observation_kernel(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
    ) -> OctonionicObservationKernel:
        r"""
        MORFISMO TERMINAL DE LA FASE 1 (OBSERVE) / GERMEN DE LA FASE 2.

        Sanea los tres agentes, audita Banach, construye los estados en la FPU
        y emite el `OctonionicObservationKernel`.

        Firma: (\mathbb{R}^8)^3 \longrightarrow \mathbf{OctonionicObservationKernel}.
        """
        c_clean = self._sanitize_signed_zeros(self._validate_vector8(contractor_S, "contratista"))
        s_clean = self._sanitize_signed_zeros(self._validate_vector8(supplier_S, "proveedor"))
        i_clean = self._sanitize_signed_zeros(self._validate_vector8(interventor_S, "interventor"))

        spec_c = self.evaluate_banach_spectrum(c_clean, "contratista")
        spec_s = self.evaluate_banach_spectrum(s_clean, "proveedor")
        spec_i = self.evaluate_banach_spectrum(i_clean, "interventor")

        state_a = self._resolver.build_state(c_clean)
        state_b = self._resolver.build_state(s_clean)
        state_c = self._resolver.build_state(i_clean)

        polar_c = self._polar_unit(c_clean, spec_c.l2_norm)
        polar_s = self._polar_unit(s_clean, spec_s.l2_norm)
        polar_i = self._polar_unit(i_clean, spec_i.l2_norm)

        null_party_count = int(spec_c.is_null) + int(spec_s.is_null) + int(spec_i.is_null)
        all_regular = (
            spec_c.is_within_theoretical_bounds
            and spec_s.is_within_theoretical_bounds
            and spec_i.is_within_theoretical_bounds
        )

        seal_hash = self._sha256_payload(
            b"OCTONIONIC_OBSERVATION_KERNEL_V3",
            c_clean,
            s_clean,
            i_clean,
            np.array(
                [
                    spec_c.l1_l2_ratio,
                    spec_s.l1_l2_ratio,
                    spec_i.l1_l2_ratio,
                    spec_c.hoyer_sparsity,
                    spec_s.hoyer_sparsity,
                    spec_i.hoyer_sparsity,
                    spec_c.banach_distortion,
                    spec_s.banach_distortion,
                    spec_i.banach_distortion,
                ],
                dtype=np.float64,
            ),
            state_a.sha256_hash,
            state_b.sha256_hash,
            state_c.sha256_hash,
        )

        logger.debug(
            "Fase 1 completada: Tríada 8D en Banach. Sello de observación=%s",
            seal_hash[:12],
        )

        return OctonionicObservationKernel(
            contractor_state=state_a,
            supplier_state=state_b,
            interventor_state=state_c,
            banach_ratio_contractor=spec_c.l1_l2_ratio,
            banach_ratio_supplier=spec_s.l1_l2_ratio,
            banach_ratio_interventor=spec_i.l1_l2_ratio,
            cryptographic_seal=seal_hash,
            contractor_spectrum=spec_c,
            supplier_spectrum=spec_s,
            interventor_spectrum=spec_i,
            polar_contractor=polar_c,
            polar_supplier=polar_s,
            polar_interventor=polar_i,
            null_party_count=null_party_count,
            all_banach_regular=all_regular,
        )

    def build_observation_kernel(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
    ) -> OctonionicObservationKernel:
        """Alias de compatibilidad 2.x hacia el morfismo terminal de la Fase 1."""
        return self.synthesize_observation_kernel(contractor_S, supplier_S, interventor_S)


# ═══════════════════════════════════════════════════════════════════════════════
# §D. FASE 2 — ORIENT: HURWITZ, ASOCIADOR, HODGE-LAPLACE Y RAMPA DE CONFIANZA
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_OctonionicOrientation(Phase1_OctonionicObservation):
    r"""
    FASE 2: Orient + Decide Preliminar.
    Hereda ontológicamente de la Fase 1.

    Categoría Functorial:
      \mathbf{OctonionicObservationKernel} \times \mathcal{P}_{\mathrm{thresholds}}
      \longrightarrow \mathbf{OctonionicOrientationState}.

    Responsabilidades axiomáticas:
      1. Morfismo de inicio `continue_from_observation_kernel`: Ingesta directa
         del objeto terminal de la Fase 1 sin reprocesamiento redundante.
      2. Orientación hipercompleja delegada al resolutor v3:
         - Error de composición de Hurwitz sobre los pares transaccionales.
         - Cálculo del tensor asociador [a, b, c].
         - Extracción de residuos de Artin, Moufang, Malcev y 3-forma de Fano.
      3. Topología espectral de Hodge-Laplace sobre el 2-símplex K_3:
         - Autovalor de Fiedler \lambda_2 y hueco espectral.
         - Resistencia efectiva global de Kirchhoff R_K.
         - Disipación exergética de Dirichlet \mathcal{E}_D.
      4. Rampa graduada de de Rham:
         - [0, 0.3 \tau_{\max}] \implies \omega_{\mathrm{asoc}} = 1.0.
         - (0.3 \tau_{\max}, 0.5 \tau_{\max}] \implies \omega_{\mathrm{asoc}} = 0.5.
         - > 0.5 \tau_{\max} \implies \omega_{\mathrm{hard}} = 0.0.
      5. Ponderación de verdad Gödel-Heyting preliminar: \omega_{\mathrm{pre}} \in \Omega_3.

    Morfismo Terminal: `orient_octonionic_state`.
    Su codominio constituye el germen de entrada exclusivo de la Fase 3.
    """

    __slots__ = ()

    def continue_from_observation_kernel(
        self,
        kernel: OctonionicObservationKernel,
        associator_threshold_Lmax: float,
    ) -> OctonionicOrientationState:
        r"""
        MORFISMO DE CONTINUACIÓN DE LA FASE 2.
        Punto formal de enlace con el final de la Fase 1.
        """
        return self.observe_from_kernel(kernel=kernel, associator_threshold_Lmax=associator_threshold_Lmax)

    def _diagnose_triad_via_resolver(
        self,
        a: OctonionicState,
        b: OctonionicState,
        c: OctonionicState,
        cota_limite: float,
    ) -> OctonionicOrientationReport:
        r"""Invoca la orientación espectral profunda del resolutor octoniónico v3."""
        if hasattr(self._resolver, "orient_octonionic_diagnostics"):
            return self._resolver.orient_octonionic_diagnostics(
                a.vector_rep,
                b.vector_rep,
                c.vector_rep,
                asoc_threshold=cota_limite,
            )
        # Fallback de síntesis directa si el resolutor expone métodos individuales
        triad = self._resolver.synthesize_octonionic_triad(a.vector_rep, b.vector_rep, c.vector_rep)
        return self._resolver.observe_octonionic_triad(triad=triad, asoc_threshold=cota_limite)

    def observe_from_kernel(
        self,
        kernel: OctonionicObservationKernel,
        associator_threshold_Lmax: float,
    ) -> OctonionicOrientationState:
        r"""
        Ejecuta la orientación geométrica, espectral y algebraica sobre el kernel observado.
        """
        if not isinstance(kernel, OctonionicObservationKernel):
            raise TypeError("observe_from_kernel exige un OctonionicObservationKernel.")

        th_val = float(associator_threshold_Lmax)
        if not math.isfinite(th_val) or th_val <= 0.0:
            raise ValueError("associator_threshold_Lmax debe ser finito y estrictamente positivo.")

        cota_limite = th_val * self._safety_margin
        a = kernel.contractor_state
        b = kernel.supplier_state
        c = kernel.interventor_state

        # Diagnóstico analítico mediante el resolutor
        rep = self._diagnose_triad_via_resolver(a, b, c, cota_limite)

        assoc_norm = rep.associator_norm
        assoc_vec = rep.triad.associator
        comp_err = rep.composition_error
        comp_rel = rep.composition_relative_error
        assoc_rel = rep.associator_relative_norm

        # Umbrales graduados de la rampa de de Rham
        assoc_soft = self._thresholds.rampa_soft * cota_limite
        assoc_hard = self._thresholds.rampa_hard * cota_limite

        pair_expected = [a.norm * b.norm, b.norm * c.norm, a.norm * c.norm]
        expected_scale = max([1.0] + [v for v in pair_expected if math.isfinite(v)])

        comp_tol = max(self._tol, 100.0 * _MACHINE_EPS * expected_scale)
        comp_hard_limit = max(1e-6, 1000.0 * _MACHINE_EPS * expected_scale)

        is_cfl_stable = bool(math.isfinite(comp_err) and comp_err <= comp_tol)
        is_asoc_stable = bool(math.isfinite(assoc_norm) and assoc_norm <= cota_limite + self._tol)

        hard_comp = bool((not math.isfinite(comp_err)) or (comp_err > comp_hard_limit))
        soft_comp = bool(math.isfinite(comp_err) and not hard_comp and comp_err > comp_tol)

        hard_asoc = bool((not math.isfinite(assoc_norm)) or (assoc_norm > assoc_hard + self._tol))
        soft_asoc = bool(math.isfinite(assoc_norm) and not hard_asoc and assoc_norm > assoc_soft + self._tol)

        artin_broken = rep.artin_residual > self._thresholds.artin_tolerance
        malcev_broken = rep.triad.malcev_residual > self._thresholds.artin_tolerance

        # Evaluación en la cadena de Heyting \Omega_3
        omega_hard = 0.0 if (hard_comp or hard_asoc or artin_broken or malcev_broken) else 1.0
        omega_cfl = 0.0 if hard_comp else (0.5 if soft_comp else 1.0)
        omega_asoc = 0.0 if hard_asoc else (0.5 if soft_asoc else 1.0)
        omega_banach = 0.5 if (kernel.null_party_count > 0 or not kernel.all_banach_regular) else 1.0
        omega_hodge = 1.0 if rep.laplacian_connectivity >= self._thresholds.fiedler_min else 0.5

        omega_pre = _heyting_meet_all((omega_hard, omega_cfl, omega_asoc, omega_banach, omega_hodge))

        return OctonionicOrientationState(
            kernel=kernel,
            associator_norm=assoc_norm,
            composition_error=comp_err,
            is_associative_stable=is_asoc_stable,
            is_cfl_stable=is_cfl_stable,
            composition_relative_error=comp_rel,
            associator_relative_norm=assoc_rel,
            cota_limite=cota_limite,
            associator_vector=assoc_vec,
            composition_tolerance=comp_tol,
            composition_hard_limit=comp_hard_limit,
            associator_soft_start=assoc_soft,
            associator_hard_limit=assoc_hard,
            omega_hard=omega_hard,
            omega_cfl=omega_cfl,
            omega_asoc=omega_asoc,
            omega_banach=omega_banach,
            omega_hodge=omega_hodge,
            omega_pre=omega_pre,
            artin_residual=rep.artin_residual,
            moufang_residual=rep.moufang_residual,
            malcev_residual=rep.triad.malcev_residual,
            fano_3form_value=rep.triad.fano_3form_value,
            laplacian_connectivity=rep.laplacian_connectivity,
            laplacian_spectral_gap=rep.laplacian_spectral_gap,
            kirchhoff_index=rep.kirchhoff_index,
            dirichlet_exergy=rep.dirichlet_exergy,
            triad_seal=rep.triad.sha256_hash,
            hard_composition=hard_comp,
            hard_associator=hard_asoc,
            soft_composition=soft_comp,
            soft_associator=soft_asoc,
        )

    def orient_octonionic_state(
        self,
        kernel: OctonionicObservationKernel,
        associator_threshold_Lmax: float,
    ) -> OctonionicOrientationState:
        r"""
        MORFISMO TERMINAL DE LA FASE 2 / GERMEN DE LA FASE 3.

        Encadena el kernel de la Fase 1 con la orientación de diagnóstico.
        Firma: \mathbf{OctonionicObservationKernel} \longrightarrow \mathbf{OctonionicOrientationState}.
        """
        return self.continue_from_observation_kernel(kernel, associator_threshold_Lmax)


# ═══════════════════════════════════════════════════════════════════════════════
# §E. FASE 3 — ACT: DECISIÓN HEYTING, INTERLOCK CROWBAR Y CERTIFICADO
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_OODAActuator(Phase2_OctonionicOrientation):
    r"""
    FASE 3: Act.
    Hereda ontológicamente de la Fase 2.

    Categoría Functorial:
      \mathbf{OctonionicOrientationState} \longrightarrow \mathbf{OctonionicAgentCertificate}.

    Responsabilidades axiomáticas:
      1. Morfismo de inicio `continue_from_orientation`: Ingesta directa del
         objeto terminal de la Fase 2.
      2. Clasificación categórica en el retículo de Gödel \Omega_3:
           - Si \omega_{\mathrm{pre}} = 0 \implies \mathrm{VETOED} (veto duro inmediato).
           - Si \omega_{\mathrm{pre}} = \tfrac{1}{2} \implies \mathrm{DEGRADED} (ventana modal \Gamma).
           - Si \omega_{\mathrm{pre}} = 1 \implies \mathrm{COHERENT}.
      3. Evaluación de override HMAC-SHA256 bajo el axioma de no-promoción:
           \sigma(\mathrm{DEGRADED}) = \mathrm{DEGRADED} (\Gamma\text{ desactivada}).
      4. Interlock ciber-físico Crowbar BT151 (simulado) con restricción física IRAM (< 400 ns).
      5. Sellado canónico criptográfico Little-Endian SHA-256 / HMAC.

    Morfismo Terminal Global: `execute_octonionic_control_cycle`.
    """

    __slots__ = (
        "_override_verifier",
        "_allowed_override_tokens",
        "_soft_veto_timestamp",
        "_is_soft_veto_active",
        "_hmac_secret",
    )

    def __init__(
        self,
        *,
        override_verifier: Optional[Callable[[str], bool]] = None,
        allowed_override_tokens: Optional[AbstractSet[str]] = None,
        allow_legacy_overrides: bool = True,
        hmac_secret: Optional[bytes] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._override_verifier: Optional[Callable[[str], bool]] = override_verifier

        tokens = allowed_override_tokens
        if tokens is None:
            tokens = _DEFAULT_OVERRIDE_TOKENS if allow_legacy_overrides else frozenset()
        self._allowed_override_tokens: Final[frozenset[str]] = frozenset(tokens)

        if hmac_secret is not None and not isinstance(hmac_secret, (bytes, bytearray)):
            raise TypeError("hmac_secret debe ser bytes o None.")
        self._hmac_secret: Final[Optional[bytes]] = (
            bytes(hmac_secret) if hmac_secret is not None else None
        )

        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False

    def reset_soft_veto(self) -> None:
        r"""Restablece la ventana temporal modal de gracia \Gamma."""
        self._is_soft_veto_active = False
        self._soft_veto_timestamp = None

    def _verify_override(self, token: Optional[str], session_seal: str) -> bool:
        """Autenticación en tiempo constante del token de override."""
        if not token or not isinstance(token, str):
            return False

        if callable(self._override_verifier):
            try:
                return bool(self._override_verifier(token))
            except Exception:
                logger.exception("Excepción en override_verifier.")
                return False

        token_b = token.encode("utf-8")
        if self._hmac_secret is not None:
            expected = hmac.new(self._hmac_secret, session_seal.encode("ascii"), hashlib.sha256).hexdigest()
            if hmac.compare_digest(token, expected):
                return True

        for allowed in self._allowed_override_tokens:
            if hmac.compare_digest(token_b, allowed.encode("utf-8")):
                return True

        return False

    def _act_crowbar(
        self,
        orientation: OctonionicOrientationState,
        verdict: HeytingVerdict,
    ) -> Tuple[float, float, str]:
        r"""
        Simula la actuación física del circuito Crowbar BT151 sobre GPIO14.
        Calcula la latencia y la inyección de carga de compuerta Q_{gt}.
        """
        if verdict is not HeytingVerdict.VETOED:
            return 0.0, 0.0, ""

        sha = hashlib.sha256()
        sha.update(b"BT151_CROWBAR_OCTONIONIC_AGENT_IRAM")
        sha.update(orientation.kernel.cryptographic_seal.encode("ascii"))
        sha.update(_canonical_bytes(orientation.associator_norm))
        sha.update(_canonical_bytes(orientation.composition_error))
        digest = sha.digest()
        seed_hex = digest.hex()

        raw_int = int.from_bytes(digest[:4], "big")
        fraction = float(raw_int) / float(0xFFFFFFFF)

        latency_ns = 382.0 + 16.5 * fraction  # Intervalo estricto [382.0, 398.5] ns
        if latency_ns >= _CROWBAR_IRAM_LATENCY_NS:
            latency_ns = _CROWBAR_IRAM_LATENCY_NS - 0.1

        gate_charge_nc = 15.0 + 2.5 * fraction

        logger.critical("╔══════════════════════════════════════════════════════════════╗")
        logger.critical("║ ¡INTERLOCK CROWBAR BT151 GATILLADO EN SOBERANO OCTONIÓNICO!  ║")
        logger.critical("║ - Protocolo    : Cortocircuito a tierra por Veto Duro en Ω_3. ║")
        logger.critical("║ - Dispositivo  : Tiristor %s en %s.                   ║", _CROWBAR_DEVICE, _CROWBAR_GPIO)
        logger.critical("║ - Latencia ISR : %.2f ns (Límite IRAM: 400.0 ns) -> CONFORME. ║", latency_ns)
        logger.critical("║ - Carga Gate   : %.2f nC inyectada.                          ║", gate_charge_nc)
        logger.critical("║ - Acción FPU   : Bus trilateral puenteado inmediatamente.    ║")
        logger.critical("╚══════════════════════════════════════════════════════════════╝")

        return float(latency_ns), float(gate_charge_nc), seed_hex

    def decide_from_orientation(
        self,
        orientation: OctonionicOrientationState,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> Tuple[HeytingVerdict, bool, bool, float, bool, Tuple[str, ...]]:
        r"""
        MORFISMO DE DECISIÓN EN \Omega_3 CON MODALIDAD DE GRACIA \Gamma Y OVERRIDE \sigma.
        """
        now = time.monotonic() if curr_time is None else float(curr_time)
        omega = float(orientation.omega_pre)
        reasons: List[str] = []

        if orientation.hard_composition:
            reasons.append("Violación severa de la ley de composición de Hurwitz.")
        if orientation.hard_associator:
            reasons.append("Obstrucción asociativa severa (> 0.5 τ_max).")
        if orientation.artin_residual > self._thresholds.artin_tolerance:
            reasons.append("Residuo de Artin anómalo en el núcleo.")
        if orientation.malcev_residual > self._thresholds.artin_tolerance:
            reasons.append("Discrepancia en identidad analítica de Malcev.")

        # CASO 1: Veto duro (Axioma: no anulable)
        if omega <= 0.0:
            self.reset_soft_veto()
            return HeytingVerdict.VETOED, False, True, 0.0, False, tuple(reasons)

        # CASO 2: Régimen nominal
        if omega >= 1.0:
            self.reset_soft_veto()
            return HeytingVerdict.COHERENT, False, False, 0.0, False, ("Coherencia trilateral nominal.",)

        # CASO 3: Régimen degradado (omega = 0.5)
        if orientation.soft_associator:
            reasons.append("Frustración asociativa en banda elástica (0.3 - 0.5 τ_max).")
        if orientation.soft_composition:
            reasons.append("Deriva de Hurwitz en banda elástica.")
        if orientation.kernel.null_party_count > 0:
            reasons.append(f"Presencia de {orientation.kernel.null_party_count} actor(es) nulo(s).")
        if orientation.laplacian_connectivity < self._thresholds.fiedler_min:
            reasons.append("Conectividad algebraica de Fiedler reducida en K_3.")

        if override_token is not None:
            if self._verify_override(override_token, orientation.kernel.cryptographic_seal):
                self.reset_soft_veto()
                reasons.append("Override HMAC autenticado: σ(DEGRADED) = DEGRADED, gracia desactivada.")
                return HeytingVerdict.DEGRADED, False, False, 0.0, False, tuple(reasons)
            reasons.append("Intento de override no autenticado.")

        if not self._is_soft_veto_active and not simulate_grace_expired:
            self._is_soft_veto_active = True
            self._soft_veto_timestamp = now
            reasons.append("Veto suave activado: ventana de gracia Γ iniciada.")
            return HeytingVerdict.DEGRADED, True, False, self._grace_limit, False, tuple(reasons)

        elapsed = (
            self._grace_limit + 1.0
            if self._soft_veto_timestamp is None or simulate_grace_expired
            else now - self._soft_veto_timestamp
        )
        time_rem = max(0.0, self._grace_limit - elapsed)

        if time_rem <= self._tol or simulate_grace_expired:
            self.reset_soft_veto()
            reasons.append("Ventana de gracia Γ expirada sin override válido: colapso a VETOED.")
            return HeytingVerdict.VETOED, False, True, 0.0, True, tuple(reasons)

        reasons.append("Veto suave persistente bajo ventana de gracia activa.")
        return HeytingVerdict.DEGRADED, True, False, time_rem, False, tuple(reasons)

    def continue_from_orientation(
        self,
        orientation: OctonionicOrientationState,
        override_token: Optional[str] = None,
        curr_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
    ) -> OctonionicAgentCertificate:
        r"""
        MORFISMO DE CONTINUACIÓN DE LA FASE 3.
        Punto formal de enlace con el final de la Fase 2.
        """
        verdict, is_soft, is_hard, time_rem, grace_expired, reasons = self.decide_from_orientation(
            orientation=orientation,
            override_token=override_token,
            curr_time=curr_time,
            simulate_grace_expired=simulate_grace_expired,
        )

        interlock_fired = bool(is_hard or verdict is HeytingVerdict.VETOED)
        latency_ns, gate_charge, seed_hex = (
            self._act_crowbar(orientation, verdict) if interlock_fired else (0.0, 0.0, "")
        )

        # Sellado criptográfico canónico determinista SHA-256 / HMAC
        seal_payload = (
            orientation.kernel.contractor_state.vector_rep,
            orientation.kernel.supplier_state.vector_rep,
            orientation.kernel.interventor_state.vector_rep,
            orientation.associator_vector,
            np.array(
                [
                    orientation.associator_norm,
                    orientation.composition_error,
                    orientation.composition_relative_error,
                    orientation.associator_relative_norm,
                    orientation.laplacian_connectivity,
                    orientation.dirichlet_exergy,
                    float(interlock_fired),
                    float(is_soft),
                    verdict.omega,
                    latency_ns,
                    gate_charge,
                ],
                dtype=np.float64,
            ),
            verdict.value,
            orientation.kernel.cryptographic_seal,
            seed_hex,
        )

        if self._hmac_secret is not None:
            raw_b = b"".join(_canonical_bytes(p) for p in seal_payload)
            digital_sig = hmac.new(self._hmac_secret, raw_b, hashlib.sha256).hexdigest()
        else:
            digital_sig = self._sha256_payload(*seal_payload)

        def _hoyer(rep: Optional[BanachRegularityReport]) -> float:
            return 0.0 if rep is None else float(rep.hoyer_sparsity)

        hoyer_max = max(
            _hoyer(orientation.kernel.contractor_spectrum),
            _hoyer(orientation.kernel.supplier_spectrum),
            _hoyer(orientation.kernel.interventor_spectrum),
        )

        return OctonionicAgentCertificate(
            phase=_PHASE_NAME,
            heyting_verdict=verdict.value,
            associator_norm=orientation.associator_norm,
            is_associative_stable=orientation.is_associative_stable,
            composition_error=orientation.composition_error,
            is_surgery_active=bool(verdict is HeytingVerdict.DEGRADED),
            is_soft_veto_active=bool(self._is_soft_veto_active),
            override_grace_period_expired=bool(grace_expired),
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency_ns,
            time_grace_remaining=time_rem,
            digital_signature_sha256=digital_sig,
            composition_relative_error=orientation.composition_relative_error,
            associator_relative_norm=orientation.associator_relative_norm,
            banach_ratio_contractor=orientation.kernel.banach_ratio_contractor,
            banach_ratio_supplier=orientation.kernel.banach_ratio_supplier,
            banach_ratio_interventor=orientation.kernel.banach_ratio_interventor,
            contractor_norm=orientation.kernel.contractor_state.norm,
            supplier_norm=orientation.kernel.supplier_state.norm,
            interventor_norm=orientation.kernel.interventor_state.norm,
            heyting_omega=verdict.omega,
            omega_pre=orientation.omega_pre,
            artin_residual=orientation.artin_residual,
            moufang_residual=orientation.moufang_residual,
            malcev_residual=orientation.malcev_residual,
            fano_3form_value=orientation.fano_3form_value,
            laplacian_connectivity=orientation.laplacian_connectivity,
            kirchhoff_index=orientation.kirchhoff_index,
            dirichlet_exergy=orientation.dirichlet_exergy,
            observation_seal=orientation.kernel.cryptographic_seal,
            hoyer_sparsity_max=hoyer_max,
            null_party_count=orientation.kernel.null_party_count,
            reasons=reasons,
            agent_version=__version__,
        )

    def execute_octonionic_control_cycle(
        self,
        contractor_S: Sequence[float],
        supplier_S: Sequence[float],
        interventor_S: Sequence[float],
        associator_threshold_Lmax: float = 0.15,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
    ) -> OctonionicAgentCertificate:
        r"""
        MORFISMO TERMINAL GLOBAL DEL SOBERANO DE CALIBRE OCTONIÓNICO.

        Ejecuta el ciclo covariante OODA trilateral en \mathbb{O}:
          (S_1, S_2, S_3)
          \xrightarrow{\text{Fase 1: Observe}} \mathbf{Kernel}
          \xrightarrow{\text{Fase 2: Orient}} \mathbf{OrientationState}
          \xrightarrow{\text{Fase 3: Act}} \mathbf{AgentCertificate}.
        """
        # FASE 1: Observe
        kernel = self.synthesize_observation_kernel(
            contractor_S,
            supplier_S,
            interventor_S,
        )

        # FASE 2: Orient (encadenamiento formal directo)
        orientation = self.continue_from_observation_kernel(
            kernel=kernel,
            associator_threshold_Lmax=associator_threshold_Lmax,
        )

        # FASE 3: Act (cierre functorial, interlock y certificación)
        certificate = self.continue_from_orientation(
            orientation=orientation,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
        )

        logger.info(
            "Ciclo Soberano 8D auditado | Veredicto: %s | Asoc: %.4e | Sello: %s | Crowbar: %s",
            certificate.heyting_verdict,
            certificate.associator_norm,
            certificate.digital_signature_sha256[:12],
            certificate.hardware_interlock_fired,
        )

        return certificate


# ═══════════════════════════════════════════════════════════════════════════════
# §F. FACHADA SOBERANA PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════════
# Anidación Ontológica Estricta: Phase3 ⊏ Phase2 ⊏ Phase1.
# La fachada pública soberana es directamente la Fase 3 culminada con Morphism.
class OctonionicDependencyAgent(Phase3_OODAActuator, Morphism):
    r"""
    Soberano de Calibre Octoniónico (OODA de lazo cerrado, 3 fases anidadas).

    Gobierna de forma covariante los estados hipercomplejos no asociativos de la
    Malla en \mathbb{O}^3 \cong (\mathbb{R}^8)^3, administrando la Rampa de Confianza
    de de Rham para censurar colusiones trilaterales, frustración de calibre
    y desviaciones de la ley de composición de Hurwitz.
    """

    __slots__ = ()

    def __init__(
        self,
        tolerance: float = 1e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            safety_margin=safety_margin,
            grace_period_seconds=grace_period_seconds,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"OctonionicDependencyAgent("
            f"tolerance={self._tol}, "
            f"safety_margin={self._safety_margin}, "
            f"grace_period_seconds={self._grace_limit}"
            f")"
        )


__all__ = [
    "OctonionicDependencyAgent",
    "OctonionicObservationKernel",
    "OctonionicOrientationState",
    "OctonionicAgentCertificate",
    "OctonionicAgentThresholds",
    "BanachRegularityReport",
    "HeytingDecision",
    "HeytingVerdict",
    "CrowbarActuationReport",
    "Phase1_OctonionicObservation",
    "Phase2_OctonionicOrientation",
    "Phase3_OODAActuator",
]
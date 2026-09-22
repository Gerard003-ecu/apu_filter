from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Scalar Momentum Satellite Agent (Soberano del Satélite I — Momentum Escalar)║
║ RUTA   : app/agents/core/immune_system/scalar_momentum_satellite_agent.py            ║
║ VERSIÓN: 3.0.0-Doctoral-HeytingTopos-BanachSobolev-PortHamiltonian-ESP32Secure       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Agente supervisor ciber-físico en el Estrato Omega ($V_\Omega$). Gobierna en lazo cerrado al Motor
Satelital de Momentum Escalar sobre una variedad Riemanniana compacta con frontera $(\mathcal{M}, G, \partial\mathcal{M})$.
Cada paso del ciclo OODA actúa como un 1-morfismo functorial en el topos de haces $\mathcal{S}h(\mathcal{M})$
equipado con el clasificador de subobjetos de Heyting $\Omega_3$.

Fundamentación Matemática, Categórica y Ciber-Física Rigurosa:
──────────────────────────────────────────────────────────────
1. Clasificador de Subobjetos en el Topos de Heyting Trivalente $\Omega_3 = \{\bot, \mathfrak{m}, \top\}$:
   $$\bot = \mathtt{VETOED} (0), \quad \mathfrak{m} = \mathtt{DEGRADED} (1), \quad \top = \mathtt{COHERENT} (2)$$
   - Operaciones de retículo intuicionista:
     $$a \wedge b = \min(a,b), \quad a \vee b = \max(a,b), \quad a \Rightarrow b = \begin{cases} \top & \text{si } a \le b \\ b & \text{si } a > b \end{cases}$$
   - Pseudo-complementación intuicionista: $\neg a = (a \Rightarrow \bot)$.
   - Falla del Tercio Excluso: $\neg\neg \mathfrak{m} = \top \neq \mathfrak{m}$, validando la lógica intuicionista no booleana.

2. Regularidad de Sobolev-Banach $\ell^p$ sobre $T^*\mathcal{M}$:
   Para $v \in \mathbb{R}^d \setminus \{0\}$, se cumple la equivalencia de normas:
   $$\|v\|_2 \le \|v\|_1 \le \sqrt{d} \|v\|_2 \implies 1.0 \le \frac{\|v\|_1}{\|v\|_2} \le \sqrt{d}$$
   con entropía espectral de Shannon $H(q) = -\sum q_k \ln q_k$ en nats.

3. Aniquilación Cuántica de Fock ($e^- + e^+ \to 2\gamma$):
   Sutura de la Luz Ámbar ($\mathfrak{m}$) mediante inyección de Positrón de Autorización $e^+$
   verificado en tiempo constante (`hmac.compare_digest`), colapsando el estado degradado a $\top = \mathtt{COHERENT}$.

4. Conmutación de Silicio y Actuación Crowbar ESP32 / BT151 (IRAM < 400 ns):
   Ante un Veto Duro ($\bot$), se despacha la ISR en memoria IRAM en latencia $t_{\mathrm{act}} < 400\,\mathrm{ns}$
   hacia el pin GPIO14, disparando el tiristor rápido BT151 para desenergizar actuadores en el milisegundo cero.

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: Desbordamientos no controlados en la transferencia de momentum técnico/financiero provocan
  fallos en cascada en la ejecución de la obra, resultando en penalizaciones contractuales y demandas por paradas no programadas.
• Dinero: El circuito Crowbar y la lógica de Heyting en lazo cerrado previenen el desgarro de procesos y mitigan
  riesgos catastróficos, ahorrando costos de paralización de planta y protegiendo el margen financiero del contrato.

Estructura Functorial OODA:
───────────────────────────
- Observe  : `observe_field_and_momentum` -> Salida: `MomentumObservationKernel`
- Orient   : `orient_transfer_dynamics`   -> Salida: `MomentumOrientationReport`
- Act      : `decide_and_act`            -> Salida: `ScalarMomentumAgentCertificate`
- Composición Síncrona Lazo Cerrado      : `execute_ooda_cycle`
"""

import hashlib
import hmac
import logging
import math
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN RESILIENTE DE DEPENDENCIAS DEL MOTOR Y DEL ECOSISTEMA
# ══════════════════════════════════════════════════════════════════════════════

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:  # type: ignore[no-redef]
        """Stub de morfismo categórico en entorno desacoplado."""

        pass

    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        """Excepción base para fallas de invariantes topológicos."""

        pass

try:
    from scalar_momentum_satellite_engine import (
        ScalarMomentumSatelliteEngine,
        ScalarMomentumEngineState,
        ScalarMomentumKernel,
        MomentumTransferReport,
        SpectralMetricTensorCache,
        DifferentialFormsBundle,
        PortHamiltonianCircuitTelemetry,
        BanachStabilityCertification,
        KahanNeumaierAccumulator,
        ScalarMomentumEngineError,
        MetricIndefinitenessError,
        DimensionMismatchError,
        ThermodynamicPassivityViolationError,
        BanachSemigroupInstabilityError,
    )
except ImportError:
    try:
        from app.core.immune_system.scalar_momentum_satellite_engine import (
            ScalarMomentumSatelliteEngine,
            ScalarMomentumEngineState,
            ScalarMomentumKernel,
            MomentumTransferReport,
            SpectralMetricTensorCache,
            DifferentialFormsBundle,
            PortHamiltonianCircuitTelemetry,
            BanachStabilityCertification,
            KahanNeumaierAccumulator,
            ScalarMomentumEngineError,
            MetricIndefinitenessError,
            DimensionMismatchError,
            ThermodynamicPassivityViolationError,
            BanachSemigroupInstabilityError,
        )
    except ImportError as exc:
        raise ImportError(
            "CRITICAL: No se pudo enlazar el motor 'scalar_momentum_satellite_engine'. "
            "Asegúrese de que el archivo esté disponible en el PYTHONPATH."
        ) from exc

logger = logging.getLogger("APU.Agents.Omega.ScalarMomentumSatelliteAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES (METROLOGÍA IEEE-754, SILICIO, CRIPTOGRAFÍA)
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15
_BANACH_REL_TOL: Final[float] = 1.0e-12
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12
_CROWBAR_IRAM_BUDGET_NS: Final[float] = 400.0
_ESP32_GPIO14_CROWBAR_PIN: Final[int] = 14
_DEFAULT_GRACE_PERIOD_SEC: Final[float] = 3600.0
_ELASTIC_SOFT_RATIO: Final[float] = 0.30
_ELASTIC_HARD_RATIO: Final[float] = 0.50
_THYRISTOR_BT151_MODEL: Final[str] = "BT151-650R"
_AGENT_HMAC_KEY: Final[bytes] = os.environ.get(
    "SCALAR_MOMENTUM_AGENT_HMAC_KEY",
    "ScalarMomentumSatelliteAgent::OmegaAgoraSecretKey2026",
).encode("utf-8")

_POSITRON_SEEDS: Final[Tuple[bytes, ...]] = (
    b"AUT_POS_SABIDURIA_777",
    b"OVERRIDE_SCALAR_MOMENTUM_IDU_2026",
    b"HMAC_SUTURA_FOCK_SECURE_MOMENTUM",
)


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES LOCALES DEL AGENTE (COMPLEMENTAN LAS DEL MOTOR)
# ══════════════════════════════════════════════════════════════════════════════

class ScalarMomentumAgentError(Exception):
    """Falla soberana del agente de momentum escalar."""


class IEEE754AnomalyError(ScalarMomentumAgentError):
    """NaN, Inf o payload no finito en un tensor del fibrado cotangente."""


class CryptographicSealError(ScalarMomentumAgentError):
    """Ruptura de la cadena SHA-256/HMAC entre fases anidadas."""


class HeytingClassificationError(ScalarMomentumAgentError):
    """Inconsistencia en la evaluación del retículo Ω₃."""


class CrowbarBudgetViolationError(ScalarMomentumAgentError):
    """La emulación ISR excedió el presupuesto IRAM de 400 ns (telemetría)."""


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DE HEYTING TRIVALENTE Ω₃ (CLASIFICADOR DE SUBOBJETOS)
# ══════════════════════════════════════════════════════════════════════════════

class MomentumHeytingVerdict(IntEnum):
    r"""
    Cadena de Heyting Ω₃ = {⊥ < m < ⊤} como subobjeto clasificador de Sh(M).

    - ⊥ = VETOED   (0) : falso absoluto; colapso disipativo o violación de pasividad.
    - m = DEGRADED (1) : subobjeto frontera / luz ámbar; tensión elástica acotada.
    - ⊤ = COHERENT (2) : verdadero absoluto; interior abierto pasivo-estable.

    Operaciones:
      a ∧ b = min(a, b),  a ∨ b = max(a, b),
      a ⇒ b = ⊤ si a ≤ b,  b en caso contrario,
      ¬a = (a ⇒ ⊥).  En particular ¬¬m = ⊤ ≠ m (falla del tercio excluso).
    """

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @classmethod
    def meet(cls, a: MomentumHeytingVerdict, b: MomentumHeytingVerdict) -> MomentumHeytingVerdict:
        """Ínfimo reticular: a ∧ b = min(a, b)."""
        return cls(min(int(a), int(b)))

    @classmethod
    def join(cls, a: MomentumHeytingVerdict, b: MomentumHeytingVerdict) -> MomentumHeytingVerdict:
        """Supremo reticular: a ∨ b = max(a, b)."""
        return cls(max(int(a), int(b)))

    @classmethod
    def implication(cls, a: MomentumHeytingVerdict, b: MomentumHeytingVerdict) -> MomentumHeytingVerdict:
        r"""Residuo de Heyting en una cadena: a ⇒ b = ⊤ si a ≤ b, else b."""
        if int(a) <= int(b):
            return cls.COHERENT
        return b

    @classmethod
    def pseudo_complement(cls, a: MomentumHeytingVerdict) -> MomentumHeytingVerdict:
        r"""Negación intuicionista ¬a = (a ⇒ ⊥)."""
        return cls.implication(a, cls.VETOED)

    @classmethod
    def double_negation(cls, a: MomentumHeytingVerdict) -> MomentumHeytingVerdict:
        r"""Clausura regular ¬¬a. Satisface ¬¬m = ⊤ ≠ m."""
        return cls.pseudo_complement(cls.pseudo_complement(a))

    def is_regular(self) -> bool:
        """Un elemento es regular ssi ¬¬a = a (sólo ⊥ y ⊤ lo son)."""
        return self.double_negation(self) is self


# ══════════════════════════════════════════════════════════════════════════════
# EXPEDIENTES INMUTABLES (OBJETOS DEL FUNCTOR OODA)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BanachSobolevMetrics:
    r"""
    Métricas de regularidad en ℓ¹ ∩ ℓ² ∩ ℓ^∞ sobre R^d, con test de Hölder.

    Desigualdades canónicas (1 ≤ p ≤ q ≤ ∞):
      ||v||_q ≤ ||v||_p ≤ d^{1/p − 1/q} ||v||_q.
    Caso (p, q) = (1, 2):  1 ≤ ||v||_1 / ||v||_2 ≤ √d.
    """

    dimension: int
    norm_l1: float
    norm_l2: float
    norm_linf: float
    banach_interpolation_ratio: float
    holder_linf_l2_ratio: float
    spectral_shannon_entropy: float
    is_numerically_vacuum: bool
    is_elliptic_regular: bool
    is_holder_consistent: bool


@dataclass(frozen=True, slots=True)
class MetricSpectralAudit:
    """Testigos duales de SPD: Cholesky + espectro hermítico simetrizado."""

    frobenius_asymmetry: float
    lambda_min: float
    lambda_max: float
    spectral_gap: float
    condition_number: float
    spectral_radius: float
    log_det: float
    cholesky_witness: bool


@dataclass(frozen=True, slots=True)
class MomentumObservationKernel:
    r"""
    Codominio del morfismo terminal de Fase 1 y dominio del morfismo inicial de Fase 2.
    Vectores saneados IEEE-754, métricas de Banach y auditoría espectral de G.
    """

    x_point: NDArray[np.float64]
    momentum_p: NDArray[np.float64]
    G_metric: NDArray[np.float64]
    banach_x: BanachSobolevMetrics
    banach_p: BanachSobolevMetrics
    metric_audit: MetricSpectralAudit
    metric_condition_number: float
    spectral_radius_G: float
    observation_timestamp_ns: int
    phase1_sha256_seal: str


@dataclass(frozen=True, slots=True)
class MomentumOrientationReport:
    r"""
    Codominio del morfismo terminal de Fase 2 y dominio del morfismo inicial de Fase 3.
    Invariantes Port-Hamiltonianos, cota elástica de Lie y sello HMAC encadenado.
    """

    observation_kernel: MomentumObservationKernel
    engine_state: ScalarMomentumEngineState
    lie_transfer_val: float
    dissipated_power: float
    stress_energy_trace: float
    lyapunov_dot_bound: float
    elastic_limit_bound: float
    elastic_strain_ratio: float
    is_passivity_satisfied: bool
    is_lyapunov_nonincreasing: bool
    is_transfer_within_elastic_limit: bool
    orientation_timestamp_ns: int
    phase2_hmac_sha256: str


@dataclass(frozen=True, slots=True)
class SiliconCrowbarActuationTelemetry:
    """Telemetría de la ISR Crowbar emulada (ESP32 IRAM → GPIO14 → BT151)."""

    hardware_interlock_fired: bool
    gpio_pin_asserted: int
    actuation_latency_ns: float
    iram_instruction_budget_ns: float
    budget_respected: bool
    thyristor_bt151_model: str
    thermal_stress_integral_i2t: float


@dataclass(frozen=True, slots=True)
class ScalarMomentumAgentCertificate:
    """Certificado soberano inmutable emitido al clausurar la Fase 3."""

    phase: str
    heyting_verdict: MomentumHeytingVerdict
    verdict_name: str
    lie_derivative_transfer: float
    dissipated_power: float
    stress_energy_trace: float
    is_passive_stable: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    fock_annihilation_executed: bool
    crowbar_telemetry: SiliconCrowbarActuationTelemetry
    time_grace_remaining_seconds: float
    total_agent_cycle_latency_us: float
    phase1_sha256_seal: str
    phase2_hmac_sha256: str
    digital_signature_sha256: str


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVE
# Ingesta covariante, saneamiento IEEE-754, Banach-Sobolev, espectro de G.
# Morfismo terminal: observe_field_and_momentum → MomentumObservationKernel
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_MomentumAgentObserver:
    r"""
    FASE 1 (OBSERVE).

    Pipeline de lemas (cada método es un morfismo parcial):
      1. _assert_cotangent_bundle_dimensions
      2. _sanitize_ieee754_tensor / _sanitize_cotangent_triple
      3. evaluate_banach_regularity          (normas ℓ^p + entropía de Shannon)
      4. _audit_metric_symmetry
      5. _factor_metric_cholesky_witness     (testigo SPD independiente)
      6. _audit_background_metric_spectrum   (autovalores, κ₂, log-det)
      7. _seal_phase1_sha256
      8. observe_field_and_momentum          ← MORFISMO TERMINAL DE FASE 1
         cuyo codominio es exactamente el dominio de
         Phase2.ingest_phase1_observation_kernel.
    """

    def __init__(self, condition_tolerance: float = 1.0e10, tolerance: float = 1.0e-12) -> None:
        if not math.isfinite(condition_tolerance) or condition_tolerance <= 1.0:
            raise ValueError("condition_tolerance debe ser finito y > 1.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser finito y estrictamente positivo.")
        self._condition_tolerance: Final[float] = float(condition_tolerance)
        self._tol: Final[float] = float(tolerance)

    # ── Lemas numéricos de Fase 1 ─────────────────────────────────────────────

    @staticmethod
    def _assert_cotangent_bundle_dimensions(
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> int:
        """Verifica (x, p) ∈ T*M ≅ R^d × R^d y G ∈ SPD(d)."""
        if x_point.ndim != 1 or momentum_p.ndim != 1:
            raise DimensionMismatchError("x_point y momentum_p deben ser tensores 1-D.")
        dim = int(x_point.shape[0])
        if dim <= 0:
            raise DimensionMismatchError("La dimensión del fibrado cotangente debe ser ≥ 1.")
        if int(momentum_p.shape[0]) != dim:
            raise DimensionMismatchError(
                f"Discrepancia dimensional: dim(x)={dim} != dim(p)={momentum_p.shape[0]}"
            )
        if G_metric.ndim != 2 or G_metric.shape != (dim, dim):
            raise DimensionMismatchError(
                f"G debe ser matriz {dim}×{dim}. Forma recibida: {G_metric.shape}"
            )
        return dim

    @staticmethod
    def _sanitize_ieee754_tensor(tensor: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        r"""
        Sanea un tensor float64:
          - rechaza NaN/Inf (no hay morfismo parcial hacia R),
          - colapsa −0.0 → +0.0 (IEEE-754: x == 0.0 es verdadero para ambos ceros),
          - materializa un buffer C-contiguo para sellos criptográficos estables.
        """
        arr = np.asarray(tensor, dtype=np.float64)
        if arr.size == 0:
            raise DimensionMismatchError(f"Tensor '{name}' vacío.")
        if not np.all(np.isfinite(arr)):
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise IEEE754AnomalyError(
                f"Anomalía IEEE-754 en '{name}': nan={n_nan}, inf={n_inf}."
            )
        # −0.0 == 0.0, por lo que este where canoniza ambos ceros a +0.0.
        canon = np.where(arr == 0.0, 0.0, arr)
        return np.ascontiguousarray(canon, dtype=np.float64)

    def _sanitize_cotangent_triple(
        self,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Sanea simultáneamente la terna (x, p, G)."""
        return (
            self._sanitize_ieee754_tensor(x_point, "x_point"),
            self._sanitize_ieee754_tensor(momentum_p, "momentum_p"),
            self._sanitize_ieee754_tensor(G_metric, "G_metric"),
        )

    @staticmethod
    def _shannon_entropy_nats(abs_v: NDArray[np.float64], norm_l1: float) -> float:
        r"""
        Entropía de Shannon en nats, H(q) = −∑ q_k ln q_k, q = |v| / ||v||_1.
        Suma compensada vía math.fsum (algoritmo de Shewchuk) para evitar
        cancelación en colas de probabilidad próximas al floor de Wilkinson.
        """
        if norm_l1 <= _WILKINSON_SAFETY_FLOOR:
            return 0.0
        probabilities = abs_v / norm_l1
        terms = (
            float(-q_k * math.log(q_k))
            for q_k in probabilities
            if q_k > _WILKINSON_SAFETY_FLOOR
        )
        return float(math.fsum(terms))

    def evaluate_banach_regularity(self, vec: NDArray[np.float64]) -> BanachSobolevMetrics:
        r"""
        Normas ℓ¹, ℓ², ℓ^∞, razón de interpolación de Banach y consistencia de Hölder.

        Para v ∈ R^d \ {0}:
          ||v||_2 ≤ ||v||_1 ≤ √d ||v||_2,
          ||v||_∞ ≤ ||v||_2 ≤ √d ||v||_∞.
        El vector nulo (sección cero) se declara regular por convención.
        """
        dim = int(vec.shape[0])
        abs_v = np.abs(vec)
        norm_l1 = float(np.sum(abs_v))
        norm_linf = float(np.max(abs_v)) if dim > 0 else 0.0
        raw_l2 = float(la.norm(vec, 2))
        is_vacuum = bool(norm_l1 <= _WILKINSON_SAFETY_FLOOR)

        if is_vacuum:
            return BanachSobolevMetrics(
                dimension=dim,
                norm_l1=0.0,
                norm_l2=0.0,
                norm_linf=0.0,
                banach_interpolation_ratio=1.0,
                holder_linf_l2_ratio=1.0,
                spectral_shannon_entropy=0.0,
                is_numerically_vacuum=True,
                is_elliptic_regular=True,
                is_holder_consistent=True,
            )

        norm_l2 = max(raw_l2, _MACHINE_EPS)
        ratio_12 = norm_l1 / norm_l2
        ratio_inf2 = norm_linf / norm_l2
        sqrt_d = math.sqrt(float(dim))
        inv_sqrt_d = 1.0 / sqrt_d

        elliptic = bool(
            (ratio_12 + _BANACH_REL_TOL >= 1.0)
            and (ratio_12 <= sqrt_d * (1.0 + _BANACH_REL_TOL))
        )
        holder_ok = bool(
            (ratio_inf2 + _BANACH_REL_TOL >= inv_sqrt_d)
            and (ratio_inf2 <= 1.0 + _BANACH_REL_TOL)
            and elliptic
        )
        entropy = self._shannon_entropy_nats(abs_v, norm_l1)

        return BanachSobolevMetrics(
            dimension=dim,
            norm_l1=norm_l1,
            norm_l2=norm_l2,
            norm_linf=norm_linf,
            banach_interpolation_ratio=ratio_12,
            holder_linf_l2_ratio=ratio_inf2,
            spectral_shannon_entropy=entropy,
            is_numerically_vacuum=False,
            is_elliptic_regular=elliptic,
            is_holder_consistent=holder_ok,
        )

    @staticmethod
    def _audit_metric_symmetry(G_metric: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """Simetriza G ↦ (G+Gᵀ)/2 y mide la asimetría de Frobenius relativa."""
        g_t = G_metric.T
        fro_g = max(float(la.norm(G_metric, "fro")), _MACHINE_EPS)
        asymmetry = float(la.norm(G_metric - g_t, "fro")) / fro_g
        g_sym = np.ascontiguousarray(0.5 * (G_metric + g_t), dtype=np.float64)
        return g_sym, asymmetry

    @staticmethod
    def _factor_metric_cholesky_witness(g_sym: NDArray[np.float64]) -> Tuple[bool, float]:
        r"""
        Testigo independiente de definición positiva: G = L Lᵀ.
        log det G = 2 ∑ log L_ii  (estable, evita overflow del producto de autovalores).
        """
        try:
            chol, lower = la.cho_factor(g_sym, lower=True, overwrite_a=False, check_finite=True)
            diag = np.abs(np.diag(chol if lower else chol.T))
            if np.any(diag <= _WILKINSON_SAFETY_FLOOR):
                return False, float("-inf")
            log_det = float(2.0 * math.fsum(float(math.log(d_ii)) for d_ii in diag))
            return True, log_det
        except (la.LinAlgError, ValueError):
            return False, float("-inf")

    def _audit_background_metric_spectrum(self, G_metric: NDArray[np.float64]) -> MetricSpectralAudit:
        r"""
        Auditoría espectral de G ∈ S⁺_d(R):
          κ₂(G) = λ_max / λ_min,  ρ(G) = λ_max,  gap = λ_min (conectividad algebraica
          si G fuese un laplaciano; aquí es el margen de elipticidad).
        Doble testigo: Cholesky + eigvalsh sobre la simetrización hermítica.
        """
        g_sym, asymmetry = self._audit_metric_symmetry(G_metric)
        if asymmetry > max(_SYMMETRY_REL_TOL, self._tol):
            logger.warning(
                "Asimetría de Frobenius de G por encima de tolerancia: %.6e", asymmetry
            )

        chol_ok, log_det_chol = self._factor_metric_cholesky_witness(g_sym)
        eigvals = la.eigvalsh(g_sym)
        lambda_min = float(eigvals[0])
        lambda_max = float(eigvals[-1])

        if (not chol_ok) or lambda_min <= _WILKINSON_SAFETY_FLOOR:
            raise MetricIndefinitenessError(
                f"G no es estrictamente SPD: λ_min={lambda_min:.6e}, cholesky={chol_ok}."
            )

        # log-det espectral como contraste numérico del testigo de Cholesky.
        positive = eigvals[eigvals > _WILKINSON_SAFETY_FLOOR]
        log_det_eig = float(math.fsum(float(math.log(float(ev))) for ev in positive))
        log_det = log_det_chol if math.isfinite(log_det_chol) else log_det_eig

        cond = lambda_max / lambda_min
        return MetricSpectralAudit(
            frobenius_asymmetry=asymmetry,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            spectral_gap=lambda_min,
            condition_number=cond,
            spectral_radius=lambda_max,
            log_det=log_det,
            cholesky_witness=chol_ok,
        )

    @staticmethod
    def _seal_phase1_sha256(
        clean_x: NDArray[np.float64],
        clean_p: NDArray[np.float64],
        clean_G: NDArray[np.float64],
        banach_x: BanachSobolevMetrics,
        banach_p: BanachSobolevMetrics,
        audit: MetricSpectralAudit,
        t_obs_ns: int,
    ) -> str:
        """Sello SHA-256 canónico de la observación (compromiso de Fase 1)."""
        hasher = hashlib.sha256()
        hasher.update(b"PHASE1|v3|")
        hasher.update(clean_x.tobytes())
        hasher.update(clean_p.tobytes())
        hasher.update(clean_G.tobytes())
        hasher.update(
            f"{banach_x.banach_interpolation_ratio:.16e}:"
            f"{banach_p.banach_interpolation_ratio:.16e}:"
            f"{audit.condition_number:.16e}:"
            f"{audit.log_det:.16e}:"
            f"{t_obs_ns}".encode("ascii")
        )
        return hasher.hexdigest()

    def recompute_phase1_seal(self, kernel: MomentumObservationKernel) -> str:
        """Reconstrucción determinista del sello para verificación en Fase 2."""
        return self._seal_phase1_sha256(
            kernel.x_point,
            kernel.momentum_p,
            kernel.G_metric,
            kernel.banach_x,
            kernel.banach_p,
            kernel.metric_audit,
            kernel.observation_timestamp_ns,
        )

    def observe_field_and_momentum(
        self,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
    ) -> MomentumObservationKernel:
        r"""
        MORFISMO TERMINAL FORMAL DE FASE 1 (OBSERVE).

        Dominio  : (x, p, G) ∈ T*M × S⁺_d(R)
        Codominio: MomentumObservationKernel

        Este expediente es la ÚNICA entrada admisible de
        Phase2_MomentumAgentOrienter.ingest_phase1_observation_kernel,
        que es a su vez el morfismo inicial de la Fase 2. La composición
        ingest ∘ observe es el primer 1-simplexo del complejo OODA.
        """
        t_obs_ns = time.perf_counter_ns()
        self._assert_cotangent_bundle_dimensions(x_point, momentum_p, G_metric)
        clean_x, clean_p, clean_G = self._sanitize_cotangent_triple(x_point, momentum_p, G_metric)

        banach_x = self.evaluate_banach_regularity(clean_x)
        banach_p = self.evaluate_banach_regularity(clean_p)

        if not banach_x.is_elliptic_regular or not banach_p.is_elliptic_regular:
            logger.warning(
                "Alerta metrológica Banach: ratio(x)=%.6f, ratio(p)=%.6f, cota √d=%.6f",
                banach_x.banach_interpolation_ratio,
                banach_p.banach_interpolation_ratio,
                math.sqrt(float(clean_x.shape[0])),
            )
        if not banach_x.is_holder_consistent or not banach_p.is_holder_consistent:
            logger.warning("Inconsistencia de Hölder ℓ^∞/ℓ² en x o p.")

        audit = self._audit_background_metric_spectrum(clean_G)
        if audit.condition_number > self._condition_tolerance:
            logger.warning(
                "Tensor métrico G severamente mal condicionado: κ₂(G)=%.6e",
                audit.condition_number,
            )

        phase1_seal = self._seal_phase1_sha256(
            clean_x, clean_p, clean_G, banach_x, banach_p, audit, t_obs_ns
        )

        return MomentumObservationKernel(
            x_point=clean_x,
            momentum_p=clean_p,
            G_metric=clean_G,
            banach_x=banach_x,
            banach_p=banach_p,
            metric_audit=audit,
            metric_condition_number=audit.condition_number,
            spectral_radius_G=audit.spectral_radius,
            observation_timestamp_ns=t_obs_ns,
            phase1_sha256_seal=phase1_seal,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENT
# Inicia CONSUMIENDO MomentumObservationKernel (codominio de Fase 1).
# Morfismo terminal: orient_transfer_dynamics → MomentumOrientationReport
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_MomentumAgentOrienter(Phase1_MomentumAgentObserver):
    r"""
    FASE 2 (ORIENT). Hereda el observador: Observe ↪ Orient.

    Pipeline de lemas:
      0. ingest_phase1_observation_kernel    ← MORFISMO INICIAL = continuación
                                               formal de observe_field_and_momentum
      1. _dispatch_engine_fpu
      2. _extract_port_hamiltonian_invariants
      3. _evaluate_elastic_lie_bound
      4. _evaluate_lyapunov_slope
      5. _seal_phase2_hmac
      6. orient_transfer_dynamics            ← MORFISMO TERMINAL DE FASE 2
         cuyo codominio es exactamente el dominio de
         Phase3.ingest_phase2_orientation_report.
    """

    def __init__(
        self,
        condition_tolerance: float = 1.0e10,
        tolerance: float = 1.0e-12,
        engine: Optional[ScalarMomentumSatelliteEngine] = None,
    ) -> None:
        super().__init__(condition_tolerance=condition_tolerance, tolerance=tolerance)
        self._engine: Final[ScalarMomentumSatelliteEngine] = (
            engine
            if engine is not None
            else ScalarMomentumSatelliteEngine(condition_tolerance=condition_tolerance)
        )

    def ingest_phase1_observation_kernel(
        self, observation: MomentumObservationKernel
    ) -> MomentumObservationKernel:
        r"""
        MORFISMO INICIAL FORMAL DE FASE 2.

        Dominio  : MomentumObservationKernel  (codominio de Fase 1)
        Codominio: MomentumObservationKernel  (el mismo, íntegro)

        Continúa el morfismo terminal `observe_field_and_momentum`:
        verifica el sello SHA-256, la terna (x, p, G) y la SPD de G.
        Toda orientación DEBE pasar por este ingest; no hay otro camino.
        """
        if not isinstance(observation, MomentumObservationKernel):
            raise TypeError(
                "Fase 2 exige MomentumObservationKernel (salida de Fase 1). "
                f"Recibido: {type(observation)!r}"
            )
        expected = self.recompute_phase1_seal(observation)
        if not hmac.compare_digest(expected, observation.phase1_sha256_seal):
            raise CryptographicSealError(
                "Ruptura del sello SHA-256 de Fase 1: el kernel fue mutado o forjado."
            )
        self._assert_cotangent_bundle_dimensions(
            observation.x_point, observation.momentum_p, observation.G_metric
        )
        if observation.metric_condition_number != observation.metric_audit.condition_number:
            raise CryptographicSealError("Inconsistencia interna κ₂(G) en el kernel.")
        return observation

    def _dispatch_engine_fpu(
        self,
        observation: MomentumObservationKernel,
        phi_func: Callable[[Any], Any],
        coupling_alpha: float,
        mass_m: float,
        conductivity_sigma: float,
    ) -> ScalarMomentumEngineState:
        """Despacho ciego a la FPU del motor satelital (cálculo Port-Hamiltoniano)."""
        if not callable(phi_func):
            raise TypeError("phi_func debe ser invocable.")
        for name, val in (
            ("coupling_alpha", coupling_alpha),
            ("mass_m", mass_m),
            ("conductivity_sigma", conductivity_sigma),
        ):
            if not math.isfinite(float(val)):
                raise IEEE754AnomalyError(f"Parámetro no finito: {name}={val!r}")
        return self._engine.execute_momentum_transfer_audit(
            phi_func=phi_func,
            x_point=observation.x_point,
            momentum_p=observation.momentum_p,
            G_metric=observation.G_metric,
            coupling_alpha=float(coupling_alpha),
            mass_m=float(mass_m),
            conductivity_sigma=float(conductivity_sigma),
        )

    @staticmethod
    def _extract_port_hamiltonian_invariants(
        engine_state: ScalarMomentumEngineState,
    ) -> Tuple[float, float, float, bool]:
        r"""
        Extrae (ℒ_v φ, P_diss, tr_G T, pasividad).
        Pasividad de Tellegen: P_diss = ⟨dφ, G⁻¹ dφ⟩ + σ ⟨p, G⁻¹ p⟩ ≥ 0.
        """
        report = engine_state.transfer_report
        lie_transfer = float(report.lie_derivative_transfer)
        p_diss = float(report.circuit_telemetry.dissipated_power)
        trace_t = float(report.stress_energy_trace)
        is_passive = bool(report.circuit_telemetry.is_thermodynamically_passive)
        for name, val in (("lie", lie_transfer), ("P_diss", p_diss), ("tr_T", trace_t)):
            if not math.isfinite(val):
                raise IEEE754AnomalyError(f"Invariante no finito del motor: {name}={val!r}")
        return lie_transfer, p_diss, trace_t, is_passive

    def _evaluate_elastic_lie_bound(
        self,
        lie_transfer: float,
        transfer_threshold_limit: float,
        safety_margin: float,
    ) -> Tuple[float, float, bool]:
        r"""
        Cota elástica de deformación: ratio = |ℒ_v φ| / (κ_elastic · margen).
        Retorna (bound, ratio, is_within).
        """
        if transfer_threshold_limit <= 0.0 or safety_margin <= 0.0:
            raise ValueError("transfer_threshold_limit y safety_margin deben ser > 0.")
        elastic_bound = float(transfer_threshold_limit * safety_margin)
        abs_lie = abs(lie_transfer)
        elastic_ratio = abs_lie / max(elastic_bound, _WILKINSON_SAFETY_FLOOR)
        is_elastic = bool(abs_lie <= elastic_bound + self._tol)
        return elastic_bound, elastic_ratio, is_elastic

    def _evaluate_lyapunov_slope(self, dissipated_power: float) -> Tuple[float, bool]:
        r"""
        Proxy de Lyapunov: V̇ ≤ −P_diss. Con P_diss ≥ 0 se tiene V̇ ≤ 0.
        Se reporta V̇_bound = −P_diss.
        """
        lyap_dot = float(-dissipated_power)
        nonincreasing = bool(lyap_dot <= self._tol)
        return lyap_dot, nonincreasing

    def _seal_phase2_hmac(
        self,
        observation: MomentumObservationKernel,
        engine_state: ScalarMomentumEngineState,
        lie_transfer: float,
        p_diss: float,
        elastic_ratio: float,
        t_orient_ns: int,
    ) -> str:
        """HMAC-SHA256 encadenado al sello de Fase 1 y al sello soberano del motor."""
        signer = hmac.new(_AGENT_HMAC_KEY, digestmod=hashlib.sha256)
        signer.update(b"PHASE2|v3|")
        signer.update(observation.phase1_sha256_seal.encode("ascii"))
        sovereign = getattr(engine_state, "sovereign_cryptographic_seal", "")
        signer.update(str(sovereign).encode("ascii"))
        signer.update(
            f"{lie_transfer:.16e}:{p_diss:.16e}:{elastic_ratio:.16e}:{t_orient_ns}".encode("ascii")
        )
        return signer.hexdigest()

    def orient_transfer_dynamics(
        self,
        observation: MomentumObservationKernel,
        phi_func: Callable[[Any], Any],
        transfer_threshold_limit: float = 10.0,
        safety_margin: float = 1.0,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0,
    ) -> MomentumOrientationReport:
        r"""
        MORFISMO TERMINAL FORMAL DE FASE 2 (ORIENT).

        Dominio  : MomentumObservationKernel  (vía ingest_phase1_observation_kernel)
        Codominio: MomentumOrientationReport

        Este reporte es la ÚNICA entrada admisible de
        Phase3_MomentumAgentDecider.ingest_phase2_orientation_report,
        morfismo inicial de la Fase 3. La composición
        ingest₃ ∘ orient ∘ ingest₂ ∘ observe es el 2-simplexo OODA.
        """
        t_orient_ns = time.perf_counter_ns()
        observation = self.ingest_phase1_observation_kernel(observation)

        engine_state = self._dispatch_engine_fpu(
            observation, phi_func, coupling_alpha, mass_m, conductivity_sigma
        )
        lie_transfer, p_diss, trace_t, is_passive = self._extract_port_hamiltonian_invariants(
            engine_state
        )
        elastic_bound, elastic_ratio, is_elastic = self._evaluate_elastic_lie_bound(
            lie_transfer, transfer_threshold_limit, safety_margin
        )
        lyap_dot, lyap_ok = self._evaluate_lyapunov_slope(p_diss)

        # Consistencia pasividad ↔ signo de P_diss (testigo cruzado).
        if is_passive and p_diss < -self._tol:
            logger.error(
                "Inconsistencia Tellegen: flag pasivo=True pero P_diss=%.6e < 0.", p_diss
            )
            is_passive = False
        if (not is_passive) and p_diss >= -self._tol:
            # El motor puede marcar no-pasivo por otras causas (semigrupo, etc.).
            logger.debug("Pasividad flag=False con P_diss=%.6e (otras causas del motor).", p_diss)

        phase2_hmac = self._seal_phase2_hmac(
            observation, engine_state, lie_transfer, p_diss, elastic_ratio, t_orient_ns
        )

        logger.debug(
            "Fase 2 (Orient) clausurada: Lie=%.6e, Diss=%.6e, ElasticRatio=%.4f, V̇=%.6e",
            lie_transfer,
            p_diss,
            elastic_ratio,
            lyap_dot,
        )

        return MomentumOrientationReport(
            observation_kernel=observation,
            engine_state=engine_state,
            lie_transfer_val=lie_transfer,
            dissipated_power=p_diss,
            stress_energy_trace=trace_t,
            lyapunov_dot_bound=lyap_dot,
            elastic_limit_bound=elastic_bound,
            elastic_strain_ratio=elastic_ratio,
            is_passivity_satisfied=is_passive,
            is_lyapunov_nonincreasing=lyap_ok,
            is_transfer_within_elastic_limit=is_elastic,
            orientation_timestamp_ns=t_orient_ns,
            phase2_hmac_sha256=phase2_hmac,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECIDE & ACT
# Inicia CONSUMIENDO MomentumOrientationReport (codominio de Fase 2).
# Morfismo terminal: decide_and_act → ScalarMomentumAgentCertificate
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_MomentumAgentDecider(Phase2_MomentumAgentOrienter):
    r"""
    FASE 3 (DECIDE & ACT). Hereda el orientador: Observe ↪ Orient ↪ Act.

    Pipeline de lemas:
      0. ingest_phase2_orientation_report    ← MORFISMO INICIAL = continuación
                                               formal de orient_transfer_dynamics
      1. _verify_positron_token_hmac         (comparación tiempo-constante)
      2. _classify_hard_soft_conditions
      3. _evaluate_heyting_logic             (Ω₃ + ventana de gracia 3600 s)
      4. _execute_silicon_crowbar_interlock  (ISR IRAM < 400 ns → GPIO14)
      5. _seal_certificate_hmac
      6. decide_and_act                      ← MORFISMO TERMINAL DE FASE 3
    """

    def __init__(
        self,
        condition_tolerance: float = 1.0e10,
        tolerance: float = 1.0e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = _DEFAULT_GRACE_PERIOD_SEC,
        engine: Optional[ScalarMomentumSatelliteEngine] = None,
    ) -> None:
        super().__init__(
            condition_tolerance=condition_tolerance,
            tolerance=tolerance,
            engine=engine,
        )
        if not math.isfinite(safety_margin) or safety_margin <= 0.0:
            raise ValueError("safety_margin debe ser finito y > 0.")
        if not math.isfinite(grace_period_seconds) or grace_period_seconds < 0.0:
            raise ValueError("grace_period_seconds debe ser finito y ≥ 0.")
        self._safety_margin: Final[float] = float(safety_margin)
        self._grace_limit_sec: Final[float] = float(grace_period_seconds)
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False
        self._positron_targets: Final[Tuple[str, ...]] = tuple(
            hashlib.sha256(seed).hexdigest() for seed in _POSITRON_SEEDS
        )

    def reset_grace_state(self) -> None:
        """Reinicia el autómata de ventana de gracia (operación idempotente)."""
        self._soft_veto_timestamp = None
        self._is_soft_veto_active = False

    def ingest_phase2_orientation_report(
        self, orientation: MomentumOrientationReport
    ) -> MomentumOrientationReport:
        r"""
        MORFISMO INICIAL FORMAL DE FASE 3.

        Dominio  : MomentumOrientationReport  (codominio de Fase 2)
        Codominio: MomentumOrientationReport  (el mismo, íntegro)

        Continúa el morfismo terminal `orient_transfer_dynamics`:
        reingesta el kernel de Fase 1 (sello SHA-256) y exige HMAC de Fase 2
        no vacío. Toda decisión DEBE pasar por este ingest.
        """
        if not isinstance(orientation, MomentumOrientationReport):
            raise TypeError(
                "Fase 3 exige MomentumOrientationReport (salida de Fase 2). "
                f"Recibido: {type(orientation)!r}"
            )
        # Revalida el 1-simplexo Observe → Orient.
        self.ingest_phase1_observation_kernel(orientation.observation_kernel)
        if not orientation.phase2_hmac_sha256 or len(orientation.phase2_hmac_sha256) != 64:
            raise CryptographicSealError("HMAC de Fase 2 ausente o de longitud no SHA-256.")
        return orientation

    def _verify_positron_token_hmac(self, candidate_token: Optional[str]) -> bool:
        r"""
        Verificación en tiempo constante del positrón de autorización (e⁺).
        e⁻ + e⁺ → 2γ  se modela como m --override--> ⊤, nunca como atajo a ⊥.
        hmac.compare_digest mitiga canales laterales de temporización.
        """
        if candidate_token is None:
            return False
        if not isinstance(candidate_token, str) or not candidate_token:
            return False
        candidate_hash = hashlib.sha256(candidate_token.encode("utf-8")).hexdigest()
        matched = False
        # OR constante: se evalúan TODOS los blancos, sin cortocircuito observable.
        for target in self._positron_targets:
            matched = hmac.compare_digest(candidate_hash, target) or matched
        return bool(matched)

    def _classify_hard_soft_conditions(
        self, orientation: MomentumOrientationReport
    ) -> Tuple[bool, bool]:
        r"""
        Clasificación preliminar (aún no Heyting):
          hard: ratio > 50%  ∨  ¬pasivo  ∨  ¬Lyapunov  ∨  κ₂(G) > tolerancia
          soft: 30% < ratio ≤ 50%  ∧  pasivo  ∧  Lyapunov  ∧  κ₂ acotado
        Ambos son mutuamente excluyentes por construcción.
        """
        elastic_ratio = orientation.elastic_strain_ratio
        is_passive = orientation.is_passivity_satisfied and orientation.is_lyapunov_nonincreasing
        cond_g = orientation.observation_kernel.metric_condition_number

        hard = bool(
            (elastic_ratio > _ELASTIC_HARD_RATIO)
            or (not is_passive)
            or (cond_g > self._condition_tolerance)
        )
        soft = bool(
            (_ELASTIC_SOFT_RATIO < elastic_ratio <= _ELASTIC_HARD_RATIO)
            and is_passive
            and (cond_g <= self._condition_tolerance)
        )
        if hard and soft:
            raise HeytingClassificationError("Partición hard/soft no disjunta.")
        return hard, soft

    def _evaluate_heyting_logic(
        self,
        orientation: MomentumOrientationReport,
        current_time_sec: float,
        override_token: Optional[str],
        simulate_grace_expired: bool,
    ) -> Tuple[MomentumHeytingVerdict, bool, bool, bool, float]:
        r"""
        Evaluación formal en Ω₃.

        Retorna (veredicto, soft_veto_activo, gracia_expirada, fock_aniquilado, t_restante).

        Autómata de gracia:
          COHERENT --soft--> DEGRADED(t₀) --(t−t₀ ≥ 3600 s)--> VETOED
                              DEGRADED     --positrón válido--> COHERENT
          * --hard--> VETOED   (sin gracia)
        """
        hard, soft = self._classify_hard_soft_conditions(orientation)
        elastic_ratio = orientation.elastic_strain_ratio
        is_passive = orientation.is_passivity_satisfied
        cond_g = orientation.observation_kernel.metric_condition_number

        if hard:
            self.reset_grace_state()
            logger.critical(
                "VETO DURO Ω₃: ratio=%.4f, pasivo=%s, Lyapunov=%s, κ₂(G)=%.6e",
                elastic_ratio,
                is_passive,
                orientation.is_lyapunov_nonincreasing,
                cond_g,
            )
            return MomentumHeytingVerdict.VETOED, False, False, False, 0.0

        if not soft:
            self.reset_grace_state()
            return MomentumHeytingVerdict.COHERENT, False, False, False, 0.0

        # ── Rama DEGRADED (luz ámbar) ─────────────────────────────────────────
        grace_expired = False
        fock_annihilated = False
        time_remaining = 0.0

        if (not self._is_soft_veto_active) and (not simulate_grace_expired):
            self._is_soft_veto_active = True
            self._soft_veto_timestamp = float(current_time_sec)
            time_remaining = self._grace_limit_sec
            verdict = MomentumHeytingVerdict.DEGRADED
            logger.warning(
                "LUZ ÁMBAR Ω₃: tensión elástica=%.2f %%. Gracia=%.1f s.",
                elastic_ratio * 100.0,
                self._grace_limit_sec,
            )
        else:
            if self._soft_veto_timestamp is None:
                elapsed = self._grace_limit_sec + 1.0
            else:
                elapsed = float(current_time_sec) - self._soft_veto_timestamp
            time_remaining = max(0.0, self._grace_limit_sec - elapsed)
            if time_remaining <= self._tol or simulate_grace_expired:
                grace_expired = True
                self.reset_grace_state()
                verdict = MomentumHeytingVerdict.VETOED
                logger.critical("VENTANA DE GRACIA EXPIRADA SIN POSITRÓN. Colapso a ⊥.")
            else:
                verdict = MomentumHeytingVerdict.DEGRADED

        if override_token is not None and self._is_soft_veto_active:
            if self._verify_positron_token_hmac(override_token):
                fock_annihilated = True
                self.reset_grace_state()
                time_remaining = 0.0
                verdict = MomentumHeytingVerdict.COHERENT
                logger.info("ANIQUILACIÓN DE FOCK VALIDADA: m ↦ ⊤.")
            else:
                logger.error("Token de positrón inválido. La gracia sigue consumiéndose.")

        return verdict, self._is_soft_veto_active, grace_expired, fock_annihilated, time_remaining

    def _execute_silicon_crowbar_interlock(
        self,
        verdict: MomentumHeytingVerdict,
        lie_transfer: float,
    ) -> SiliconCrowbarActuationTelemetry:
        r"""
        Emula la ISR en IRAM: si el veredicto es ⊥, afirma GPIO14 y dispara el
        tiristor Crowbar BT151. El presupuesto de instrucción es 400 ns.
        No se toca hardware real; sólo se certifica telemetría de actuación.
        """
        t0 = time.perf_counter_ns()
        fired = verdict is MomentumHeytingVerdict.VETOED
        gpio = _ESP32_GPIO14_CROWBAR_PIN if fired else -1
        # Camino de ISR emulado: comparación + selección de pin (acotado).
        _ = gpio if fired else 0
        t1 = time.perf_counter_ns()
        latency_ns = float(max(t1 - t0, 0))
        budget_ok = bool(latency_ns <= _CROWBAR_IRAM_BUDGET_NS)
        if fired and not budget_ok:
            logger.error(
                "Presupuesto IRAM excedido: %.1f ns > %.1f ns.",
                latency_ns,
                _CROWBAR_IRAM_BUDGET_NS,
            )
        # Integral térmica I²t ∝ (ℒ_v φ)² · Δt  (unidades adimensionales normalizadas).
        i2t = float((lie_transfer * lie_transfer) * max(latency_ns, 1.0) * 1.0e-18)
        return SiliconCrowbarActuationTelemetry(
            hardware_interlock_fired=fired,
            gpio_pin_asserted=gpio,
            actuation_latency_ns=latency_ns,
            iram_instruction_budget_ns=_CROWBAR_IRAM_BUDGET_NS,
            budget_respected=budget_ok,
            thyristor_bt151_model=_THYRISTOR_BT151_MODEL,
            thermal_stress_integral_i2t=i2t,
        )

    def _seal_certificate_hmac(
        self,
        orientation: MomentumOrientationReport,
        verdict: MomentumHeytingVerdict,
        crowbar: SiliconCrowbarActuationTelemetry,
        grace_expired: bool,
        fock_annihilated: bool,
        time_remaining: float,
        t_act_ns: int,
    ) -> str:
        """HMAC-SHA256 de Fase 3 encadenado al HMAC de Fase 2 (cadena completa)."""
        signer = hmac.new(_AGENT_HMAC_KEY, digestmod=hashlib.sha256)
        signer.update(b"PHASE3|v3|")
        signer.update(orientation.phase2_hmac_sha256.encode("ascii"))
        signer.update(verdict.name.encode("ascii"))
        signer.update(
            f"{int(crowbar.hardware_interlock_fired)}:"
            f"{int(grace_expired)}:{int(fock_annihilated)}:"
            f"{time_remaining:.16e}:{t_act_ns}".encode("ascii")
        )
        return signer.hexdigest()

    def decide_and_act(
        self,
        orientation: MomentumOrientationReport,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        current_time_sec: Optional[float] = None,
    ) -> ScalarMomentumAgentCertificate:
        r"""
        MORFISMO TERMINAL FORMAL DE FASE 3 (DECIDE & ACT).

        Dominio  : MomentumOrientationReport  (vía ingest_phase2_orientation_report)
        Codominio: ScalarMomentumAgentCertificate

        Clausura el functor OODA: el certificado sella la composición
        decide ∘ ingest₃ ∘ orient ∘ ingest₂ ∘ observe
        y acredita veredicto Ω₃, aniquilación de Fock y Crowbar de silicio.
        """
        t_act_ns = time.perf_counter_ns()
        orientation = self.ingest_phase2_orientation_report(orientation)
        now = float(time.time() if current_time_sec is None else current_time_sec)

        verdict, soft_active, grace_expired, fock_annihilated, time_remaining = (
            self._evaluate_heyting_logic(
                orientation=orientation,
                current_time_sec=now,
                override_token=override_token,
                simulate_grace_expired=simulate_grace_expired,
            )
        )
        crowbar = self._execute_silicon_crowbar_interlock(verdict, orientation.lie_transfer_val)
        signature = self._seal_certificate_hmac(
            orientation,
            verdict,
            crowbar,
            grace_expired,
            fock_annihilated,
            time_remaining,
            t_act_ns,
        )

        cycle_latency_us = (
            float(t_act_ns - orientation.observation_kernel.observation_timestamp_ns) / 1.0e3
        )

        logger.info(
            "Fase 3 (Act) clausurada: Ω₃=%s, Crowbar=%s, Fock=%s, gracia_restante=%.1f s",
            verdict.name,
            crowbar.hardware_interlock_fired,
            fock_annihilated,
            time_remaining,
        )

        return ScalarMomentumAgentCertificate(
            phase="ACT",
            heyting_verdict=verdict,
            verdict_name=verdict.name,
            lie_derivative_transfer=orientation.lie_transfer_val,
            dissipated_power=orientation.dissipated_power,
            stress_energy_trace=orientation.stress_energy_trace,
            is_passive_stable=bool(
                orientation.is_passivity_satisfied and orientation.is_lyapunov_nonincreasing
            ),
            is_soft_veto_active=soft_active,
            override_grace_period_expired=grace_expired,
            fock_annihilation_executed=fock_annihilated,
            crowbar_telemetry=crowbar,
            time_grace_remaining_seconds=time_remaining,
            total_agent_cycle_latency_us=cycle_latency_us,
            phase1_sha256_seal=orientation.observation_kernel.phase1_sha256_seal,
            phase2_hmac_sha256=orientation.phase2_hmac_sha256,
            digital_signature_sha256=signature,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACHADA SOBERANA — COMPOSICIÓN DEL FUNCTOR OODA
# ScalarMomentumSatelliteAgent = Act ∘ Orient ∘ Observe
# ══════════════════════════════════════════════════════════════════════════════

class ScalarMomentumSatelliteAgent(Phase3_MomentumAgentDecider):
    r"""
    Agente soberano: objeto terminal de la torre de herencias

        Phase1_MomentumAgentObserver
          └── Phase2_MomentumAgentOrienter
                └── Phase3_MomentumAgentDecider
                      └── ScalarMomentumSatelliteAgent

    El método `execute_ooda_cycle` es la composición estricta
    decide_and_act ∘ orient_transfer_dynamics ∘ observe_field_and_momentum,
    con los ingestos internos como 1-morfismos de integridad.
    """

    def execute_ooda_cycle(
        self,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        phi_func: Callable[[Any], Any],
        transfer_threshold_limit: float = 10.0,
        safety_margin: Optional[float] = None,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        conductivity_sigma: float = 1.0,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False,
        current_time_sec: Optional[float] = None,
    ) -> ScalarMomentumAgentCertificate:
        r"""
        COMPOSICIÓN FUNCTORIAL OODA.

        Observe  : (x, p, G) ↦ MomentumObservationKernel
        Orient   : kernel × φ ↦ MomentumOrientationReport
        Act      : report × e⁺ ↦ ScalarMomentumAgentCertificate
        """
        margin = self._safety_margin if safety_margin is None else float(safety_margin)

        kernel = self.observe_field_and_momentum(
            x_point=x_point,
            momentum_p=momentum_p,
            G_metric=G_metric,
        )
        report = self.orient_transfer_dynamics(
            observation=kernel,
            phi_func=phi_func,
            transfer_threshold_limit=transfer_threshold_limit,
            safety_margin=margin,
            coupling_alpha=coupling_alpha,
            mass_m=mass_m,
            conductivity_sigma=conductivity_sigma,
        )
        return self.decide_and_act(
            orientation=report,
            override_token=override_token,
            simulate_grace_expired=simulate_grace_expired,
            current_time_sec=current_time_sec,
        )


__all__ = (
    "MomentumHeytingVerdict",
    "BanachSobolevMetrics",
    "MetricSpectralAudit",
    "MomentumObservationKernel",
    "MomentumOrientationReport",
    "SiliconCrowbarActuationTelemetry",
    "ScalarMomentumAgentCertificate",
    "ScalarMomentumAgentError",
    "IEEE754AnomalyError",
    "CryptographicSealError",
    "HeytingClassificationError",
    "CrowbarBudgetViolationError",
    "Phase1_MomentumAgentObserver",
    "Phase2_MomentumAgentOrienter",
    "Phase3_MomentumAgentDecider",
    "ScalarMomentumSatelliteAgent",
)
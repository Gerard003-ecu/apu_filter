# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Satellites (Satélites de Telemetría de Frontera)          ║
║ Ruta   : app/core/telemetry_satellites.py                                    ║
║ Versión: 2.1.0-Doctoral-Nested-Cholesky-Heyting-Landauer-Secure              ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y EXERGÉTICA:                                            ║
║ Este módulo implementa satélites de telemetría asíncronos para vigilar la    ║
║ frontera de-confinada ∂M de la fortaleza de APU Filter.                      ║
║                                                                              ║
║ La arquitectura queda dividida en tres fases anidadas (inclusión de          ║
║ kernels por herencia, funtorial en el sentido de                             ║
║   Phase1 ↪ Phase2 ↪ Phase3 ↪ SatelliteObserver):                             ║
║   FASE 1: Metrología fundamental, espectro de Rényi y regularización SPD.    ║
║   FASE 2: Problema generalizado de Sturm–Liouville, exergía de Landauer y    ║
║           retículo de Heyting Ω₃ con implicación de Gödel.                   ║
║   FASE 3: Orquestación asíncrona, actuación fail-closed y certificado.       ║
║                                                                              ║
║ Morfismos de fase (continuación formal):                                     ║
║   prepare_boundary_state  : (payload, K, G) → BoundaryState                  ║
║   analyze_boundary_state  : BoundaryState → SpectralBoundaryAnalysis         ║
║   decide_heyting_verdict  : SpectralBoundaryAnalysis → HeytingDecision       ║
║   synthesize_certificate  : HeytingDecision → SatelliteTelemetryCertificate  ║
║                                                                              ║
║ El último método de la Fase 1 prepara `BoundaryState`, objeto inicial de     ║
║ la Fase 2. El último método de la Fase 2 prepara `HeytingDecision`, objeto   ║
║ inicial de la Fase 3.                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
import scipy.linalg as la


logger = logging.getLogger("APU.Physics.TelemetrySatellites")

__version__: Final[str] = "2.1.0-Doctoral-Nested-Cholesky-Heyting-Landauer-Secure"


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS Y FÍSICAS
# ════════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0

# Constante de Boltzmann exacta CODATA 2018 [J/K].
_K_BOLTZMANN: Final[float] = 1.380649e-23
_DEFAULT_TEMPERATURE_K: Final[float] = 300.0

# Tope numérico para evitar log(∞) en costos exergéticos.
_MAX_CONDITION_FOR_LOG: Final[float] = 1.0e18
_MAX_LOG_CONDITION: Final[float] = math.log1p(_MAX_CONDITION_FOR_LOG)

# Entropía máxima de Shannon para payload de octetos: ln(256) [nats].
_BYTE_ALPHABET: Final[int] = 256
_MAX_SHANNON_NATS: Final[float] = math.log(float(_BYTE_ALPHABET))
_LN2: Final[float] = math.log(2.0)

# Umbral relativo de no-hermiticidad admisible (norma de Frobenius).
_HERMITIAN_REL_TOL: Final[float] = 1.0e-8
_HERMITIAN_REL_VETO: Final[float] = 1.0e-2

# Ω₃ = {0, 1/2, 1} ⊂ [0, 1], álgebra de Heyting finita de Gödel–Dummett.
_OMEGA3_FALSE: Final[float] = 0.0
_OMEGA3_MIDDLE: Final[float] = 0.5
_OMEGA3_TRUE: Final[float] = 1.0


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES NUMÉRICAS PURAS
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
    r"""
    Cociente protegido en la compactificación \(\mathbb{R} \cup \{\infty\}\).

    Si el denominador se anula y el numerador es nulo, se devuelve 1
    (condición de un operador 0-dimensional o isotrópico).
    """
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
    r"""
    Infimo de Gödel en \(\Omega_3\): \(a \wedge b = \min(a, b)\).
    """
    return float(min(left, right))


def _heyting_join(left: float, right: float) -> float:
    r"""
    Supremo de Gödel en \(\Omega_3\): \(a \vee b = \max(a, b)\).
    """
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

    En un álgebra de Heyting se cumple \(a \wedge (a \to b) \le b\).
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


# ════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ExternalEntropySignal:
    r"""
    Señal de entropía exógena capturada por un satélite orbital.

    Se modela como una variable de estado termodinámico no-markoviana bajo
    un baño térmico de Langevin con relación de fluctuación-disipación.

    Atributos
    ---------
    satellite_id : str
        Identificador del satélite emisor.
    entropy_rate : float
        Tasa de entropía exógena \(\sigma_{\mathrm{ext}}\) [nats/s].
    fluctuation_variance : float
        Varianza del ruido exógeno \(\xi_{\mathrm{ext}}\).
    coupling_strength : float
        Parámetro de acoplamiento \(\lambda_{\mathrm{ext}}\) (de Rham).
    timestamp_utc : str
        Instante UTC de captura.
    """

    satellite_id: str
    entropy_rate: float
    fluctuation_variance: float
    coupling_strength: float
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not isinstance(self.satellite_id, str) or not self.satellite_id.strip():
            raise ValueError("satellite_id debe ser una cadena no vacía.")

        entropy_rate = float(self.entropy_rate)
        fluctuation_variance = float(self.fluctuation_variance)
        coupling_strength = float(self.coupling_strength)

        if not math.isfinite(entropy_rate):
            raise ValueError("entropy_rate debe ser finito.")
        if not math.isfinite(fluctuation_variance):
            raise ValueError("fluctuation_variance debe ser finita.")
        if not math.isfinite(coupling_strength):
            raise ValueError("coupling_strength debe ser finita.")

        if entropy_rate < 0.0:
            raise ValueError("entropy_rate debe ser no negativa.")
        if fluctuation_variance < 0.0:
            raise ValueError("fluctuation_variance debe ser no negativa.")
        if coupling_strength < 0.0:
            raise ValueError("coupling_strength debe ser no negativa.")


@dataclass(frozen=True, slots=True)
class EntropyEstimate:
    r"""
    Estimación entrópica con corrección de sesgo, espectro de Rényi y
    cota superior de confianza.

    Shannon empírica (plug-in):
    \[
        \hat{H}
        =
        -\sum_i p_i \ln p_i.
    \]

    Miller–Madow:
    \[
        H_{\mathrm{MM}}
        =
        \hat{H} + \frac{K-1}{2N}.
    \]

    Chao–Shen (cobertura \(\hat{C} = 1 - f_1/N\)):
    \[
        H_{\mathrm{CS}}
        =
        -\sum_i
        \frac{\tilde{p}_i \ln \tilde{p}_i}{1 - (1-\tilde{p}_i)^N},
        \qquad
        \tilde{p}_i = \hat{C}\, p_i.
    \]

    Cota de sesgo de Paninski:
    \[
        \bigl|\mathbb{E}[\hat{H}] - H\bigr|
        \le
        \ln\bigl(1 + (|X|-1)/N\bigr).
    \]
    """

    entropy_nats: float
    entropy_ub_nats: float
    std_nats: float
    effective_alphabet: float
    sample_size: int
    alphabet_size: int
    miller_madow_nats: float = 0.0
    chao_shen_nats: float = 0.0
    renyi2_nats: float = 0.0
    min_entropy_nats: float = 0.0
    kl_uniform_nats: float = 0.0
    paninski_ub_nats: float = 0.0


@dataclass(frozen=True, slots=True)
class MetricRegularization:
    r"""
    Resultado de la proyección de Tikhonov sobre el cono SPD.

    \[
        G
        \mapsto
        U\,\mathrm{diag}(\max(\lambda_i, \gamma))\,U^\dagger
        \succ 0.
    \]
    """

    G_spd: np.ndarray
    condition_number: float
    n_clamped: int
    n_negative: int
    floor_used: float
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryState:
    r"""
    Estado de frontera preparado por la Fase 1.

    Objeto fronterizo \(\partial M\) regularizado que la Fase 2 analiza
    espectralmente como haz hermítico sobre el fibrado métrico \(G\).

    Contiene:
      - Entropía de payload (Shannon / Rényi / cotas).
      - Operador de frontera hermítico \(K^\dagger = K\).
      - Métrica SPD \(G \succ 0\).
      - Residuo de no-hermiticidad y condición métrica.
      - Razones de invalidación, si las hay.
    """

    is_valid: bool
    dimension: int
    entropy: EntropyEstimate
    K_hermitian: np.ndarray
    G_spd: np.ndarray
    reference_entropy_threshold: float
    safety_margin: float
    reasons: Tuple[str, ...] = ()
    metric_condition_number: float = 1.0
    hermitian_residual: float = 0.0
    hermitian_relative_residual: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "K_hermitian",
            np.array(self.K_hermitian, copy=True, order="C"),
        )
        object.__setattr__(
            self,
            "G_spd",
            np.array(self.G_spd, copy=True, order="C"),
        )


@dataclass(frozen=True, slots=True)
class SpectralBoundaryAnalysis:
    r"""
    Resultado del análisis espectral-topológico de la Fase 2.

    Se resuelve el problema generalizado de Sturm–Liouville
    \[
        K v = \lambda G v,
        \qquad G \succ 0,
    \]
    isospectral a la congruencia de Cholesky
    \[
        \widetilde{K} = L^{-1} K L^{-\dagger},
        \qquad G = L L^\dagger.
    \]
    La inercia se conserva por la ley de Sylvester.

    La brecha espectral se define como el primer autovalor estrictamente
    positivo tras el núcleo numérico, normalizado por el radio espectral:
    \[
        \gamma
        =
        \frac{\lambda_{\mathrm{gap}}}{\rho(\widetilde{K})}.
    \]

    La coercitividad es \(\lambda_{\min}(K, G)\). El índice de pasividad
    es el número de autovalores estrictamente negativos.
    """

    is_valid: bool
    dimension: int
    boundary_entropy: float
    entropy_upper_bound: float
    entropy_std: float
    effective_alphabet: float

    condition_number: float
    metric_condition_number: float

    spectral_gap: float
    absolute_spectral_gap: float

    min_eigenvalue: float
    max_eigenvalue: float
    negative_eigen_count: int
    nullity: int
    effective_rank: int

    exergy_leak_kbt: float
    physical_exergy_j: float

    reasons: Tuple[str, ...] = ()
    spectral_radius: float = 0.0
    coercivity: float = 0.0
    nuclear_norm: float = 0.0
    frobenius_norm: float = 0.0
    landauer_bound_j: float = 0.0
    irreversibility_overhead_j: float = 0.0
    sylvester_positive: int = 0
    inertia_signature: Tuple[int, int, int] = (0, 0, 0)
    successive_gap_min: float = 0.0
    perturbation_sensitivity: float = math.inf


@dataclass(frozen=True, slots=True)
class HeytingDecision:
    r"""
    Veredicto en el retículo de Heyting \(\Omega_3\).

    Estados posibles:
      - COHERENT : verdad plena, frontera estable.
      - DEGRADED : verdad intermedia, frontera admisible con degradación.
      - VETOED   : falsedad operativa, se activa interlock físico.

    La decisión usa la cota superior de entropía para conservar
    intuicionismo operacional: sólo se afirma coherencia si ni siquiera
    la incertidumbre superior viola el límite termodinámico.

    El valor de verdad reticular es el ínfimo de Gödel de las proposiciones
    atómicas (pasividad, condición, entropía, brecha, exergía, validez).
    """

    verdict: str
    truth_value: float

    boundary_entropy: float
    coupled_entropy: float
    entropy_upper_bound: float

    spectral_gap: float
    condition_number: float

    exergy_leak_kbt: float
    physical_exergy_j: float

    reference_entropy_threshold: float
    allowed_entropy_ceiling: float

    reasons: Tuple[str, ...]
    truth_continuous: float = 1.0
    heyting_implies_coherent: float = 1.0
    landauer_bound_j: float = 0.0
    atomic_propositions: Tuple[Tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class SatelliteTelemetryCertificate:
    r"""
    Certificado inmutable de regularidad de-confinada de la frontera.

    Conserva los campos originales y extiende metadatos de auditoría
    metrológica, exergética, espectral y criptográfica (SHA-256).
    """

    heyting_verdict: str
    boundary_entropy: float
    spectral_gap: float
    exergy_leak: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float

    condition_number: float = math.inf
    entropy_upper_bound: float = 0.0
    coupled_entropy: float = 0.0
    physical_exergy_j: float = 0.0
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    landauer_bound_j: float = 0.0
    truth_value: float = 1.0
    decision_sha256: str = ""
    gpio_pin: str = ""
    satellite_id: str = ""
    reasons: Tuple[str, ...] = ()


# ════════════════════════════════════════════════════════════════════════════
# FASE 1 — METROLOGÍA FUNDAMENTAL Y PREPARACIÓN DE FRONTERA
# ════════════════════════════════════════════════════════════════════════════


class Phase1MetrologyKernel:
    r"""
    FASE 1: Metrología fundamental.

    Responsabilidades:
      1. Validar operadores matriciales finitos y cuadrados.
      2. Estimar el espectro de Rényi del payload con sesgo controlado.
      3. Medir el residuo de no-hermiticidad \(\|A - A^\dagger\|_F\).
      4. Proyectar operadores a la parte hermítica de Weyl.
      5. Regularizar el tensor métrico a SPD mediante piso de Tikhonov
         adaptativo (escala de Weyl: \(\gamma \ge \varepsilon n \|G\|_2\)).
      6. Preparar `BoundaryState`, objeto de entrada para la Fase 2.

    El último método de esta fase, `prepare_boundary_state`, es el
    morfismo \(\mathrm{id}_{\partial M}\) que continúa en
    `Phase2SpectralTopologyKernel.analyze_boundary_state`.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        reference_entropy_threshold: float = 1.5,
        regularization_floor: float = 1.0e-15,
        entropy_confidence_z: float = 3.0,
        observation_window_s: float = 1.0,
        temperature_k: float = _DEFAULT_TEMPERATURE_K,
        rng_seed: Optional[int] = None,
    ) -> None:
        self._id = str(satellite_id).strip()
        if not self._id:
            raise ValueError("satellite_id debe ser una cadena no vacía.")

        self._n = int(dimension_n)
        if self._n <= 0:
            raise ValueError("dimension_n debe ser entero positivo.")

        self._safety_margin = self._finite_float(
            safety_margin, "safety_margin", _MACHINE_EPS
        )
        self._reference_entropy_threshold = self._finite_float(
            reference_entropy_threshold,
            "reference_entropy_threshold",
            0.0,
        )
        self._reg = self._finite_float(
            regularization_floor, "regularization_floor", _MACHINE_EPS
        )
        self._entropy_z = self._finite_float(
            entropy_confidence_z, "entropy_confidence_z", 0.0
        )
        self._observation_window_s = self._finite_float(
            observation_window_s,
            "observation_window_s",
            0.0,
        )
        self._temperature_k = self._finite_float(
            temperature_k, "temperature_k", 0.0
        )
        self._rng = np.random.default_rng(rng_seed)

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
    def _empty_entropy() -> EntropyEstimate:
        """Estimación nula (payload vacío o fallo metrológico)."""
        return EntropyEstimate(
            entropy_nats=0.0,
            entropy_ub_nats=0.0,
            std_nats=0.0,
            effective_alphabet=1.0,
            sample_size=0,
            alphabet_size=0,
            miller_madow_nats=0.0,
            chao_shen_nats=0.0,
            renyi2_nats=0.0,
            min_entropy_nats=0.0,
            kl_uniform_nats=0.0,
            paninski_ub_nats=0.0,
        )

    @staticmethod
    def _coerce_payload_bytes(payload: Any) -> bytes:
        """Normaliza el payload a `bytes` sin pérdida de octetos."""
        if payload is None:
            return b""
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, memoryview):
            return payload.tobytes()
        if isinstance(payload, str):
            return payload.encode("utf-8")
        return bytes(payload)

    def compute_shannon_entropy(self, payload: bytes) -> float:
        r"""
        API de compatibilidad con la versión original.

        Calcula la entropía de Shannon empírica del payload:
        \[
            H = -\sum_i p_i \ln p_i.
        \]

        Devuelve
        --------
        float
            Entropía en nats.
        """
        return float(self.estimate_payload_entropy(payload).entropy_nats)

    def compute_renyi_entropy(self, payload: bytes, alpha: float) -> float:
        r"""
        Entropía de Rényi de orden \(\alpha \in [0, \infty]\).

        \[
            H_\alpha
            =
            \frac{1}{1-\alpha}
            \ln
            \sum_i p_i^\alpha,
            \qquad
            \alpha \ne 1,
        \]
        con \(H_1\) Shannon, \(H_0 = \ln K\), \(H_\infty = -\ln p_{\max}\).
        """
        estimate = self.estimate_payload_entropy(payload)
        if estimate.sample_size <= 0 or estimate.alphabet_size <= 0:
            return 0.0

        arr = np.frombuffer(self._coerce_payload_bytes(payload), dtype=np.uint8)
        if arr.size == 0:
            return 0.0

        counts = np.bincount(arr, minlength=_BYTE_ALPHABET)
        nonzero = counts[counts > 0].astype(np.float64)
        probs = nonzero / float(arr.size)
        order = float(alpha)

        if not math.isfinite(order) or order == math.inf:
            return _clip_nats(-math.log(float(np.max(probs))))
        if abs(order - 1.0) < 1.0e-12:
            return float(estimate.entropy_nats)
        if abs(order) < 1.0e-12:
            return _clip_nats(math.log(float(nonzero.size)))

        moment = float(np.sum(np.power(probs, order)))
        if moment <= 0.0:
            return 0.0
        return _clip_nats(math.log(moment) / (1.0 - order))

    def estimate_payload_entropy(self, payload: bytes) -> EntropyEstimate:
        r"""
        Estimación rigurosa de entropía para octetos.

        Mejoras respecto al estimador plug-in original:
          - Corrección de sesgo de Miller–Madow.
          - Estimador de Chao–Shen (cobertura / Horvitz–Thompson).
          - Varianza asintótica del plug-in.
          - Cota de sesgo de Paninski con alfabeto \(|X|=256\).
          - Entropía de colisión \(H_2\) y min-entropía \(H_\infty\).
          - Divergencia de Kullback–Leibler respecto de la uniforme en el
            soporte observado.
          - Recorte al máximo teórico \(\ln 256\).
        """
        raw = self._coerce_payload_bytes(payload)
        if len(raw) == 0:
            return self._empty_entropy()

        arr = np.frombuffer(raw, dtype=np.uint8)
        n_samples = int(arr.size)
        if n_samples == 0:
            return self._empty_entropy()

        counts = np.bincount(arr, minlength=_BYTE_ALPHABET)
        nonzero = counts[counts > 0]
        alphabet_size = int(nonzero.size)
        n = float(n_samples)
        k_obs = float(alphabet_size)

        probs = nonzero.astype(np.float64) / n
        log_probs = np.log(probs)

        h_emp = _clip_nats(float(-np.sum(probs * log_probs)))

        h_mm = _clip_nats(h_emp + (k_obs - 1.0) / (2.0 * n))

        # Chao–Shen: cobertura Ĉ = 1 - f₁/N.
        n_singletons = int(np.count_nonzero(nonzero == 1))
        coverage = 1.0 - (float(n_singletons) / n)
        if coverage <= _MACHINE_EPS or n_singletons >= n_samples:
            h_cs = h_emp
        else:
            p_tilde = coverage * probs
            one_minus = 1.0 - p_tilde
            # Evita (1-p)^N = 1 numéricamente cuando p es minúscula.
            ht_den = 1.0 - np.power(np.clip(one_minus, 0.0, 1.0), n)
            ht_den = np.clip(ht_den, _MACHINE_EPS, None)
            h_cs = _clip_nats(float(-np.sum((p_tilde * np.log(p_tilde)) / ht_den)))

        # Var(Ĥ) ≈ N^{-1} (E[ln(p)^2] - H^2).
        var = float(np.sum(probs * log_probs**2) - h_emp**2) / n
        var = max(var, 0.0)
        std = math.sqrt(var)

        paninski = _clip_nats(
            h_emp + math.log(1.0 + float(_BYTE_ALPHABET - 1) / n)
        )
        h_ub = _clip_nats(max(h_mm + self._entropy_z * std, paninski, h_cs))

        collision = float(np.sum(probs * probs))
        h2 = _clip_nats(-math.log(collision)) if collision > 0.0 else 0.0
        hmin = _clip_nats(-math.log(float(np.max(probs))))

        # KL(p ‖ U_soporte) = ln K - H.
        kl_uniform = _clip_nats(max(math.log(k_obs) - h_emp, 0.0)) if k_obs > 0.0 else 0.0

        return EntropyEstimate(
            entropy_nats=h_emp,
            entropy_ub_nats=h_ub,
            std_nats=std,
            effective_alphabet=float(math.exp(h_emp)),
            sample_size=n_samples,
            alphabet_size=alphabet_size,
            miller_madow_nats=h_mm,
            chao_shen_nats=h_cs,
            renyi2_nats=h2,
            min_entropy_nats=hmin,
            kl_uniform_nats=kl_uniform,
            paninski_ub_nats=paninski,
        )

    def _as_square_operator(
        self,
        operator: Any,
        name: str,
        expected_dim: Optional[int] = None,
    ) -> np.ndarray:
        """Valida que un operador sea cuadrado, finito y de dimensión esperada."""
        if operator is None:
            raise ValueError(f"{name} no puede ser None.")

        dtype = np.complex128 if np.iscomplexobj(operator) else np.float64
        arr = np.array(operator, dtype=dtype, copy=True, order="C")

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")

        dim = int(arr.shape[0])
        expected = self._n if expected_dim is None else int(expected_dim)
        if dim != expected:
            raise ValueError(
                f"{name} tiene dimensión {dim}, pero se esperaba {expected}."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores no finitos.")

        return arr

    @staticmethod
    def _hermitian_part(A: np.ndarray) -> np.ndarray:
        r"""
        Proyección hermítica de Weyl:
        \[
            A_H = \tfrac{1}{2}(A + A^\dagger).
        \]
        Es el único minimizador de \(\|A - H\|_F\) sobre \(H^\dagger = H\).
        """
        return 0.5 * (A + A.conj().T)

    def measure_skew_hermitian_residual(self, operator: np.ndarray) -> Tuple[float, float]:
        r"""
        Residuo de no-hermiticidad.

        \[
            r_F = \|A - A^\dagger\|_F,
            \qquad
            r_{\mathrm{rel}} = \frac{r_F}{\|A\|_F + \varepsilon}.
        \]

        Devuelve
        --------
        residual_abs, residual_rel : float
        """
        arr = np.asarray(operator)
        residual = float(np.linalg.norm(arr - arr.conj().T, ord="fro"))
        fro = float(np.linalg.norm(arr, ord="fro"))
        relative = _safe_ratio(residual, fro + _MACHINE_EPS)
        if not math.isfinite(relative):
            relative = math.inf
        return residual, float(relative)

    def _tikhonov_floor(self, eigs: np.ndarray) -> float:
        r"""
        Piso de Tikhonov adaptativo a la escala de Weyl:

        \[
            \gamma
            =
            \max\bigl(
                \gamma_0,\;
                \varepsilon\, n\, \max(1, \rho)
            \bigr),
        \]
        donde \(\rho = \max_i |\lambda_i|\).
        """
        spectrum = np.asarray(eigs, dtype=np.float64)
        max_abs = float(np.max(np.abs(spectrum))) if spectrum.size else 1.0
        return max(
            self._reg,
            _MACHINE_EPS * max(1, self._n) * max(max_abs, 1.0),
        )

    def regularize_metric_tensor_detailed(
        self,
        metric_tensor: Optional[np.ndarray],
    ) -> MetricRegularization:
        r"""
        Regulariza el tensor métrico \(G\) al interior del cono SPD.

        Procedimiento:
          1. Proyección hermítica.
          2. Diagonalización unitaria (teorema espectral).
          3. Piso de Tikhonov \(\lambda_i \leftarrow \max(\lambda_i, \gamma)\).
          4. Reconstrucción \(U \Lambda_\gamma U^\dagger\) y re-simetrización.
        """
        reasons: List[str] = []

        if metric_tensor is None:
            G = np.eye(self._n, dtype=np.complex128)
            return MetricRegularization(
                G_spd=G,
                condition_number=1.0,
                n_clamped=0,
                n_negative=0,
                floor_used=self._reg,
                reasons=("metric_default_identity",),
            )

        G = self._as_square_operator(metric_tensor, "metric_tensor")
        G_h = self._hermitian_part(G)

        eigs, vecs = la.eigh(G_h, check_finite=True)
        eigs = np.asarray(eigs, dtype=np.float64)
        floor = self._tikhonov_floor(eigs)

        n_negative = int(np.count_nonzero(eigs < -floor))
        n_clamped = int(np.count_nonzero(eigs < floor))
        if n_negative > 0:
            reasons.append("metric_indefinite_tikhonov_projected")
        if n_clamped > 0:
            reasons.append("metric_eigenvalues_clamped")

        clamped = np.clip(eigs, floor, None)
        G_spd = (vecs * clamped) @ vecs.conj().T
        G_spd = self._hermitian_part(G_spd)

        cond = _safe_ratio(float(np.max(clamped)), float(np.min(clamped)))
        return MetricRegularization(
            G_spd=G_spd,
            condition_number=float(cond),
            n_clamped=n_clamped,
            n_negative=n_negative,
            floor_used=float(floor),
            reasons=tuple(reasons),
        )

    def regularize_metric_tensor(
        self,
        metric_tensor: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        r"""
        API compatible: devuelve \((G_{\mathrm{SPD}}, \kappa(G))\).
        """
        detailed = self.regularize_metric_tensor_detailed(metric_tensor)
        return detailed.G_spd, float(detailed.condition_number)

    def project_operator_to_hermitian(
        self,
        operator: np.ndarray,
        name: str = "operator",
    ) -> Tuple[np.ndarray, float, float]:
        """Proyecta a hermítico y reporta residuos absoluto y relativo."""
        K = self._as_square_operator(operator, name)
        residual, relative = self.measure_skew_hermitian_residual(K)
        return self._hermitian_part(K), residual, relative

    def prepare_boundary_state(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: Optional[float] = None,
    ) -> BoundaryState:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 1.

        Prepara el estado de frontera \(\partial M\) para la Fase 2.

        Este método:
          1. Estima la entropía del payload (Shannon, Rényi, cotas).
          2. Valida \(K\) y \(G\).
          3. Mide y proyecta el residuo de no-hermiticidad de \(K\).
          4. Regulariza \(G\) a SPD (Tikhonov adaptativo).
          5. Encapsula todo en `BoundaryState` con copias defensivas.

        Continúa directamente en la Fase 2 como el morfismo

        \[
            \texttt{analyze\_boundary\_state}
            :
            \mathsf{BoundaryState}
            \longrightarrow
            \mathsf{SpectralBoundaryAnalysis}.
        \]
        """
        fallback_reasons: List[str] = []

        try:
            entropy = self.estimate_payload_entropy(payload)
        except Exception as exc:  # pragma: no cover - defensa metrológica
            entropy = self._empty_entropy()
            fallback_reasons.append(
                f"payload_entropy_error: {type(exc).__name__}: {exc}"
            )

        if reference_entropy_threshold is None:
            threshold = self._reference_entropy_threshold
        else:
            try:
                threshold = self._finite_float(
                    reference_entropy_threshold,
                    "reference_entropy_threshold",
                    0.0,
                )
            except (TypeError, ValueError):
                threshold = self._reference_entropy_threshold
                fallback_reasons.append("reference_entropy_threshold_fallback")

        reasons = list(fallback_reasons)
        hermitian_residual = 0.0
        hermitian_relative = 0.0
        metric_condition = 1.0

        try:
            K_h, hermitian_residual, hermitian_relative = (
                self.project_operator_to_hermitian(
                    K_boundary_raw,
                    "K_boundary_raw",
                )
            )

            if hermitian_relative > _HERMITIAN_REL_VETO:
                reasons.append("operator_far_from_hermitian")
            elif hermitian_relative > _HERMITIAN_REL_TOL:
                reasons.append("operator_hermitian_residual_detectable")

            detailed = self.regularize_metric_tensor_detailed(metric_tensor)
            G_spd = detailed.G_spd
            metric_condition = float(detailed.condition_number)
            reasons.extend(detailed.reasons)

            if hermitian_relative > _HERMITIAN_REL_VETO:
                is_valid = False
            else:
                is_valid = True

        except Exception as exc:
            K_h = np.zeros((self._n, self._n), dtype=np.complex128)
            G_spd = np.eye(self._n, dtype=np.complex128)
            is_valid = False
            reasons.append(
                f"boundary_preparation_error: {type(exc).__name__}: {exc}"
            )

        return BoundaryState(
            is_valid=is_valid,
            dimension=self._n,
            entropy=entropy,
            K_hermitian=K_h,
            G_spd=G_spd,
            reference_entropy_threshold=threshold,
            safety_margin=self._safety_margin,
            reasons=_unique_reasons(reasons),
            metric_condition_number=float(metric_condition),
            hermitian_residual=float(hermitian_residual),
            hermitian_relative_residual=float(hermitian_relative),
        )


# ════════════════════════════════════════════════════════════════════════════
# FASE 2 — ESPECTRO TOPOLÓGICO, EXERGÍA Y RETÍCULO DE HEYTING
# ════════════════════════════════════════════════════════════════════════════


class Phase2SpectralTopologyKernel(Phase1MetrologyKernel):
    r"""
    FASE 2: Análisis espectral-topológico y decisión de Heyting.

    El primer método, `analyze_boundary_state`, consume el `BoundaryState`
    producido por `prepare_boundary_state` (Fase 1) y calcula el espectro
    generalizado \(Kv = \lambda Gv\).

    Responsabilidades:
      1. Resolver el problema generalizado hermítico (Cholesky / Sylvester).
      2. Clasificar inercia \((n_+, n_0, n_-)\), núcleo numérico y rango.
      3. Extraer coercitividad, radio espectral y brecha (Fiedler análoga).
      4. Estimar fuga exergética: Landauer \(k_B T\, H\) más sobrecoste
         \(\ln(1+\kappa)\) por irreversibilidad de condicionado.
      5. Dictaminar COHERENT / DEGRADED / VETOED en \(\Omega_3\) mediante
         ínfimo de Gödel de proposiciones atómicas.

    El último método, `decide_heyting_verdict`, entrega `HeytingDecision`,
    insumo directo de `Phase3AsyncActuationKernel.synthesize_certificate`.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        condition_degraded: float = 1.0e4,
        condition_veto: float = 1.0e8,
        spectral_gap_floor: float = 1.0e-6,
        exergy_budget_kbt: float = 50.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(satellite_id, dimension_n, safety_margin, **kwargs)

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

        self._spectral_gap_floor = self._finite_float(
            spectral_gap_floor,
            "spectral_gap_floor",
            0.0,
        )
        self._exergy_budget_kbt = self._finite_float(
            exergy_budget_kbt,
            "exergy_budget_kbt",
            0.0,
        )

    def _spd_cholesky(self, G: np.ndarray) -> np.ndarray:
        r"""
        Factor de Cholesky \(G = L L^\dagger\) sobre el cono SPD.
        Si falla numéricamente, re-regulariza y reintenta una vez.
        """
        try:
            return la.cholesky(G, lower=True, check_finite=True)
        except la.LinAlgError:
            G_fix, _ = self.regularize_metric_tensor(G)
            return la.cholesky(G_fix, lower=True, check_finite=True)

    def _metric_congruence_transform(
        self,
        K: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        r"""
        Reducción de Cholesky del lápiz \((K, G)\):

        \[
            \widetilde{K}
            =
            L^{-1} K L^{-\dagger},
            \qquad
            G = L L^\dagger.
        \]

        Se realiza con dos sustituciones triangulares, evitando formar
        \(G^{-1/2}\) explícitamente (mejor número de condición intermedio).
        """
        L = self._spd_cholesky(G)
        reduced = la.solve_triangular(L, K, lower=True, check_finite=True)
        reduced = la.solve_triangular(
            L, reduced.conj().T, lower=True, check_finite=True
        ).conj().T
        return self._hermitian_part(reduced)

    def _apply_metric_function(self, G: np.ndarray, transform: Any) -> np.ndarray:
        r"""
        Cálculo funcional hermítico \(f(G) = U f(\Lambda) U^\dagger\).
        """
        eigs, vecs = la.eigh(G, check_finite=True)
        eigs = np.asarray(eigs, dtype=np.float64)
        floor = self._tikhonov_floor(eigs)
        clipped = np.clip(eigs, floor, None)
        vals = transform(clipped)
        reconstructed = (vecs * vals) @ vecs.conj().T
        return self._hermitian_part(reconstructed)

    def _square_root_spd(self, G: np.ndarray) -> np.ndarray:
        r"""Raíz cuadrada SPD \(G^{1/2}\)."""
        return self._apply_metric_function(G, np.sqrt)

    def _inverse_square_root_spd(self, G: np.ndarray) -> np.ndarray:
        r"""Inversa de raíz cuadrada SPD \(G^{-1/2}\)."""
        return self._apply_metric_function(G, lambda x: 1.0 / np.sqrt(x))

    @staticmethod
    def _spectral_gap_from_spectrum(
        eigs: np.ndarray,
        tol: float,
        max_abs: float,
    ) -> Tuple[float, float]:
        r"""
        Extrae la brecha espectral como primer autovalor positivo.

        Si existe núcleo numérico, el primer autovalor positivo corresponde
        al análogo del valor de Fiedler en operadores de tipo Laplaciano.

        Si el operador es SPD pleno, coincide con la coercitividad
        \(\lambda_{\min}\) y la brecha normalizada es \(\approx 1/\kappa\).

        Autovalores negativos ⇒ frontera no pasiva ⇒ brecha nula.
        """
        if eigs.size == 0 or max_abs <= tol:
            return 0.0, 0.0

        if np.any(eigs < -tol):
            return 0.0, 0.0

        positive = eigs[eigs > tol]
        if positive.size == 0:
            return 0.0, 0.0

        gap_abs = float(positive[0])
        gap_norm = gap_abs / max_abs
        return gap_abs, float(np.clip(gap_norm, 0.0, 1.0))

    @staticmethod
    def _successive_gap(eigs: np.ndarray) -> float:
        """Mínima separación entre autovalores consecutivos."""
        if eigs.size < 2:
            return 0.0
        diffs = np.diff(np.sort(np.asarray(eigs, dtype=np.float64)))
        if diffs.size == 0:
            return 0.0
        return float(np.min(diffs))

    def _exergy_balance(
        self,
        entropy_ub_nats: float,
        condition_number: float,
    ) -> Tuple[float, float, float, float]:
        r"""
        Balance exergético.

        Cota de Landauer (calor mínimo irreversible, isoterma):
        \[
            E_{\mathrm{L}} = k_B T\, H_{\mathrm{UB}}.
        \]

        Amplificación por condicionado (trabajo de \(\mathrm{GL}(n)\)):
        \[
            E / (k_B T)
            =
            H_{\mathrm{UB}} \ln(1+\kappa),
            \qquad
            \kappa = \kappa(K, G).
        \]

        El sobrecoste \(E - E_{\mathrm{L}}\) mide irreversibilidad geométrica.

        Devuelve
        --------
        exergy_kbt, physical_j, landauer_j, overhead_j
        """
        if math.isfinite(condition_number):
            kappa = float(condition_number)
        else:
            kappa = _MAX_CONDITION_FOR_LOG
        kappa = min(max(kappa, 1.0), _MAX_CONDITION_FOR_LOG)

        h_ub = float(entropy_ub_nats)
        if not math.isfinite(h_ub) or h_ub < 0.0:
            h_ub = _MAX_SHANNON_NATS

        exergy_kbt = float(h_ub * math.log1p(kappa))
        kT = _K_BOLTZMANN * self._temperature_k
        physical_j = kT * exergy_kbt
        landauer_j = kT * h_ub
        overhead_j = max(physical_j - landauer_j, 0.0)
        return exergy_kbt, float(physical_j), float(landauer_j), float(overhead_j)

    def _invalid_analysis(
        self,
        state: BoundaryState,
        reasons: Tuple[str, ...],
    ) -> SpectralBoundaryAnalysis:
        """Construye análisis inválido conservador (fail-closed)."""
        exergy_kbt, physical_j, landauer_j, overhead_j = self._exergy_balance(
            state.entropy.entropy_ub_nats,
            math.inf,
        )
        dim = int(state.dimension)
        return SpectralBoundaryAnalysis(
            is_valid=False,
            dimension=dim,
            boundary_entropy=float(state.entropy.entropy_nats),
            entropy_upper_bound=float(state.entropy.entropy_ub_nats),
            entropy_std=float(state.entropy.std_nats),
            effective_alphabet=float(state.entropy.effective_alphabet),
            condition_number=math.inf,
            metric_condition_number=math.inf,
            spectral_gap=0.0,
            absolute_spectral_gap=0.0,
            min_eigenvalue=-math.inf,
            max_eigenvalue=math.inf,
            negative_eigen_count=dim,
            nullity=dim,
            effective_rank=0,
            exergy_leak_kbt=exergy_kbt,
            physical_exergy_j=physical_j,
            reasons=_unique_reasons(reasons),
            spectral_radius=math.inf,
            coercivity=-math.inf,
            nuclear_norm=math.inf,
            frobenius_norm=math.inf,
            landauer_bound_j=landauer_j,
            irreversibility_overhead_j=overhead_j,
            sylvester_positive=0,
            inertia_signature=(0, dim, dim),
            successive_gap_min=0.0,
            perturbation_sensitivity=math.inf,
        )

    def analyze_boundary_state(self, state: BoundaryState) -> SpectralBoundaryAnalysis:
        r"""
        PRIMER MÉTODO DE LA FASE 2.

        Continuación formal de `prepare_boundary_state` (Fase 1).

        Consume `BoundaryState` y calcula:
          - Espectro generalizado \(Kv = \lambda Gv\) (ley de Sylvester).
          - Número de condición del lápiz y de la métrica.
          - Brecha espectral, coercitividad y radio espectral.
          - Normas nuclear (Schatten-1) y de Frobenius (Schatten-2).
          - Inercia \((n_+, n_0, n_-)\).
          - Sensibilidad a perturbaciones \(\sim 1/\mathrm{gap}\)
            (cota de tipo Davis–Kahan para el projector espectral).
          - Fuga exergética Landauer + sobrecoste \(\ln(1+\kappa)\).

        Continúa posteriormente hacia `decide_heyting_verdict`.
        """
        if not state.is_valid:
            return self._invalid_analysis(state, state.reasons)

        if int(state.dimension) != self._n:
            return self._invalid_analysis(
                state,
                state.reasons + ("boundary_dimension_mismatch",),
            )

        try:
            g_eigs = la.eigvalsh(state.G_spd, check_finite=True)
            g_eigs = np.asarray(g_eigs, dtype=np.float64)
            max_g = float(np.max(g_eigs)) if g_eigs.size else 1.0
            min_g = float(np.min(g_eigs)) if g_eigs.size else 1.0
            g_floor = self._tikhonov_floor(g_eigs)
            metric_condition = _safe_ratio(
                max(max_g, g_floor),
                max(min_g, g_floor),
            )

            try:
                eigs = la.eigh(
                    state.K_hermitian,
                    b=state.G_spd,
                    eigvals_only=True,
                    check_finite=True,
                )
            except (la.LinAlgError, ValueError):
                k_tilde = self._metric_congruence_transform(
                    state.K_hermitian,
                    state.G_spd,
                )
                eigs = la.eigvalsh(k_tilde, check_finite=True)

            eigs = np.sort(np.asarray(eigs, dtype=np.float64))
            max_abs = float(np.max(np.abs(eigs))) if eigs.size else 0.0
            tol = max(
                self._reg,
                100.0 * _MACHINE_EPS * max(1, self._n) * max(max_abs, 1.0),
            )

            negative_count = int(np.count_nonzero(eigs < -tol))
            positive = eigs[eigs > tol]
            nullity = int(np.count_nonzero(np.abs(eigs) <= tol))
            effective_rank = int(positive.size)
            sylvester_positive = int(positive.size)

            min_eigenvalue = float(eigs[0]) if eigs.size else 0.0
            max_eigenvalue = float(eigs[-1]) if eigs.size else 0.0
            coercivity = min_eigenvalue
            spectral_radius = max_abs
            nuclear_norm = float(np.sum(np.abs(eigs))) if eigs.size else 0.0
            frobenius_norm = (
                float(math.sqrt(np.sum(eigs * eigs))) if eigs.size else 0.0
            )

            if positive.size == 0:
                condition_number = 1.0 if negative_count == 0 else math.inf
            else:
                min_positive = float(max(float(positive[0]), self._reg))
                max_positive = float(max(float(eigs[-1]), min_positive))
                condition_number = _safe_ratio(max_positive, min_positive)
                if negative_count > 0:
                    condition_number = max(condition_number, 1.0e12)

            gap_abs, gap_norm = self._spectral_gap_from_spectrum(eigs, tol, max_abs)
            successive = self._successive_gap(eigs)
            if gap_abs > 0.0:
                perturbation_sensitivity = 1.0 / gap_abs
            else:
                perturbation_sensitivity = math.inf

            exergy_kbt, physical_j, landauer_j, overhead_j = self._exergy_balance(
                state.entropy.entropy_ub_nats,
                condition_number,
            )

            reasons: List[str] = list(state.reasons)
            if negative_count > 0:
                reasons.append("negative_eigenvalues_non_passive")
            if nullity > 0:
                reasons.append("numeric_nullspace_detected")
            if condition_number >= self._condition_degraded:
                reasons.append("high_condition_number")
            if metric_condition >= self._condition_degraded:
                reasons.append("high_metric_condition_number")

            return SpectralBoundaryAnalysis(
                is_valid=True,
                dimension=state.dimension,
                boundary_entropy=float(state.entropy.entropy_nats),
                entropy_upper_bound=float(state.entropy.entropy_ub_nats),
                entropy_std=float(state.entropy.std_nats),
                effective_alphabet=float(state.entropy.effective_alphabet),
                condition_number=float(condition_number),
                metric_condition_number=float(metric_condition),
                spectral_gap=float(gap_norm),
                absolute_spectral_gap=float(gap_abs),
                min_eigenvalue=min_eigenvalue,
                max_eigenvalue=max_eigenvalue,
                negative_eigen_count=negative_count,
                nullity=nullity,
                effective_rank=effective_rank,
                exergy_leak_kbt=exergy_kbt,
                physical_exergy_j=float(physical_j),
                reasons=_unique_reasons(reasons),
                spectral_radius=float(spectral_radius),
                coercivity=float(coercivity),
                nuclear_norm=float(nuclear_norm),
                frobenius_norm=float(frobenius_norm),
                landauer_bound_j=float(landauer_j),
                irreversibility_overhead_j=float(overhead_j),
                sylvester_positive=sylvester_positive,
                inertia_signature=(sylvester_positive, nullity, negative_count),
                successive_gap_min=float(successive),
                perturbation_sensitivity=float(perturbation_sensitivity),
            )

        except Exception as exc:  # pragma: no cover - defensa numérica
            return self._invalid_analysis(
                state,
                state.reasons
                + (f"spectral_analysis_error: {type(exc).__name__}: {exc}",),
            )

    def evaluate_boundary_regularization(
        self,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Compatibilidad evolucionada con el método original.

        Original:
        \[
            \widetilde{K} = \tfrac12(K + K^\dagger) + \gamma I.
        \]

        Evolución (congruencia de Cholesky, isospectral al lápiz):
        \[
            K_{\mathrm{pur}}
            =
            L\,
            U\,\mathrm{clip}(\Lambda, \gamma)\,U^\dagger
            \,L^\dagger,
        \]
        donde \(L^{-1} K_H L^{-\dagger} = U \Lambda U^\dagger\) y
        \(G = L L^\dagger\). Equivale a recortar los autovalores
        generalizados y reconstruir \(K_{\mathrm{pur}} = G V \Lambda_\gamma V^\dagger G\)
        con \(V^\dagger G V = I\).

        Devuelve
        --------
        K_purified : np.ndarray
            Tensor regularizado en coordenadas físicas.
        condition_number : float
            Número de condición del espectro recortado.
        """
        state = self.prepare_boundary_state(
            b"",
            K_boundary_raw,
            metric_tensor,
        )
        if not state.is_valid:
            return state.K_hermitian, math.inf

        try:
            L = self._spd_cholesky(state.G_spd)
            k_tilde = self._metric_congruence_transform(
                state.K_hermitian,
                state.G_spd,
            )
            eigs, vecs = la.eigh(k_tilde, check_finite=True)
            clamped = np.clip(eigs, self._reg, None)
            k_clip = (vecs * clamped) @ vecs.conj().T
            k_clip = self._hermitian_part(k_clip)
            k_purified = L @ k_clip @ L.conj().T
            k_purified = self._hermitian_part(k_purified)
            cond = _safe_ratio(float(np.max(clamped)), float(np.min(clamped)))
            return k_purified, float(cond)
        except Exception:
            return state.K_hermitian, math.inf

    def _external_entropy_contribution(
        self,
        signal: Optional[ExternalEntropySignal],
    ) -> Tuple[float, float, Tuple[str, ...]]:
        r"""
        Acopla la señal exógena de Langevin al presupuesto de entropía.

        Para una ventana de observación \(\Delta t\):
        \[
            \mu_{\mathrm{ext}}
            =
            \lambda_{\mathrm{ext}}
            \sigma_{\mathrm{ext}}
            \Delta t,
            \qquad
            s_{\mathrm{ext}}
            =
            \lambda_{\mathrm{ext}}
            \sqrt{\mathrm{Var}(\xi_{\mathrm{ext}})}
            \Delta t.
        \]
        """
        if signal is None:
            return 0.0, 0.0, ()

        try:
            if signal.satellite_id != self._id:
                return 0.0, 0.0, ("external_signal_satellite_mismatch_ignored",)

            rate = self._finite_float(signal.entropy_rate, "entropy_rate", 0.0)
            var = self._finite_float(
                signal.fluctuation_variance,
                "fluctuation_variance",
                0.0,
            )
            coupling = self._finite_float(
                signal.coupling_strength,
                "coupling_strength",
                0.0,
            )
            dt = self._observation_window_s
            mean = rate * coupling * dt
            std = math.sqrt(var) * coupling * dt

            reasons: List[str] = []
            if not math.isfinite(mean):
                mean = math.inf
                reasons.append("external_entropy_mean_overflow")
            if not math.isfinite(std):
                std = math.inf
                reasons.append("external_entropy_std_overflow")
            return float(mean), float(std), tuple(reasons)

        except Exception as exc:  # pragma: no cover - defensa defensiva
            return 0.0, 0.0, (
                f"external_signal_invalid: {type(exc).__name__}: {exc}",
            )

    def _atomic_heyting_propositions(
        self,
        analysis: SpectralBoundaryAnalysis,
        coupled_entropy: float,
        entropy_ub: float,
        ceiling: float,
    ) -> List[Tuple[str, float]]:
        r"""
        Proposiciones atómicas en \(\Omega_3\).

        El veredicto global es el ínfimo de Gödel de esta familia.
        Un átomo 0 veta; un átomo 1/2 degrada; 1 es regular.
        """
        atoms: List[Tuple[str, float]] = []

        atoms.append(
            ("validity", _OMEGA3_TRUE if analysis.is_valid else _OMEGA3_FALSE)
        )
        atoms.append(
            (
                "passivity",
                _OMEGA3_FALSE if analysis.negative_eigen_count > 0 else _OMEGA3_TRUE,
            )
        )

        cond = analysis.condition_number
        if (not math.isfinite(cond)) or cond >= self._condition_veto:
            atoms.append(("condition", _OMEGA3_FALSE))
        elif cond >= self._condition_degraded:
            atoms.append(("condition", _OMEGA3_MIDDLE))
        else:
            atoms.append(("condition", _OMEGA3_TRUE))

        if (not math.isfinite(entropy_ub)) or entropy_ub > ceiling:
            atoms.append(("entropy", _OMEGA3_FALSE))
        elif coupled_entropy > 0.7 * ceiling or entropy_ub > 0.85 * ceiling:
            atoms.append(("entropy", _OMEGA3_MIDDLE))
        else:
            atoms.append(("entropy", _OMEGA3_TRUE))

        if analysis.spectral_gap < self._spectral_gap_floor:
            atoms.append(("spectral_gap", _OMEGA3_MIDDLE))
        else:
            atoms.append(("spectral_gap", _OMEGA3_TRUE))

        if not math.isfinite(analysis.exergy_leak_kbt):
            atoms.append(("exergy", _OMEGA3_FALSE))
        elif analysis.exergy_leak_kbt > self._exergy_budget_kbt:
            atoms.append(("exergy", _OMEGA3_MIDDLE))
        else:
            atoms.append(("exergy", _OMEGA3_TRUE))

        return atoms

    def decide_heyting_verdict(
        self,
        analysis: SpectralBoundaryAnalysis,
        external_signal: Optional[ExternalEntropySignal] = None,
        reference_entropy_threshold: Optional[float] = None,
    ) -> HeytingDecision:
        r"""
        ÚLTIMO MÉTODO DE LA FASE 2.

        Decide el estado de verdad en el retículo de Heyting \(\Omega_3\):
        \[
            \mathrm{COHERENT}
            \prec
            \mathrm{DEGRADED}
            \prec
            \mathrm{VETOED}.
        \]

        La decisión es conservadora e intuicionista:
          - Usa cota superior de entropía (nunca afirma más de lo demostrable).
          - El veredicto es el ínfimo de Gödel de las proposiciones atómicas.
          - Veta si hay autovalores negativos (no pasividad).
          - Veta si \(\kappa\) no es finito o excede el umbral de veto.
          - Veta si la entropía superior supera el techo permitido.
          - Degrada si brecha, exergía o condición rozan el límite.

        Entrega `HeytingDecision` a la Fase 3 como el morfismo

        \[
            \texttt{synthesize\_certificate}
            :
            \mathsf{HeytingDecision}
            \longrightarrow
            \mathsf{SatelliteTelemetryCertificate}.
        \]
        """
        if reference_entropy_threshold is None:
            threshold = self._reference_entropy_threshold
        else:
            try:
                threshold = self._finite_float(
                    reference_entropy_threshold,
                    "reference_entropy_threshold",
                    0.0,
                )
            except (TypeError, ValueError):
                threshold = self._reference_entropy_threshold

        ceiling = threshold * self._safety_margin
        reasons: List[str] = list(analysis.reasons)

        coupled_entropy = float(analysis.boundary_entropy)
        entropy_ub = float(analysis.entropy_upper_bound)
        if not math.isfinite(entropy_ub):
            entropy_ub = math.inf

        if external_signal is not None:
            ext_mean, ext_std, ext_reasons = self._external_entropy_contribution(
                external_signal
            )
            reasons.extend(ext_reasons)
            coupled_entropy += ext_mean
            entropy_ub += ext_mean + self._entropy_z * ext_std

        atoms = self._atomic_heyting_propositions(
            analysis,
            coupled_entropy,
            entropy_ub,
            ceiling,
        )

        truth_lattice = _OMEGA3_TRUE
        for name, value in atoms:
            truth_lattice = _heyting_meet(truth_lattice, value)
            if value == _OMEGA3_FALSE:
                reasons.append(f"atom_veto:{name}")
            elif value == _OMEGA3_MIDDLE:
                reasons.append(f"atom_degraded:{name}")

        if truth_lattice == _OMEGA3_FALSE:
            verdict = "VETOED"
            truth_value = _OMEGA3_FALSE
        elif truth_lattice == _OMEGA3_MIDDLE:
            verdict = "DEGRADED"
            truth_value = _OMEGA3_MIDDLE
        else:
            verdict = "COHERENT"
            truth_value = _OMEGA3_TRUE

        # Márgenes continuos en (0, 1] para auditoría (no alteran Ω₃).
        margins: List[float] = []
        if math.isfinite(ceiling) and ceiling > 0.0 and math.isfinite(entropy_ub):
            margins.append(float(np.clip(1.0 - entropy_ub / ceiling, 0.0, 1.0)))
        else:
            margins.append(0.0)

        if math.isfinite(analysis.condition_number) and analysis.condition_number > 0.0:
            margins.append(
                float(
                    np.clip(
                        1.0
                        - math.log(max(analysis.condition_number, 1.0))
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
                analysis.spectral_gap
                / (analysis.spectral_gap + self._spectral_gap_floor + _MACHINE_EPS)
            )
        )

        if math.isfinite(analysis.exergy_leak_kbt):
            budget = max(self._exergy_budget_kbt, _MACHINE_EPS)
            margins.append(
                float(np.clip(1.0 - analysis.exergy_leak_kbt / budget, 0.0, 1.0))
            )
        else:
            margins.append(0.0)

        geo = _geometric_mean(margins)
        if verdict == "VETOED":
            truth_continuous = 0.0
        elif verdict == "DEGRADED":
            truth_continuous = 0.5 * geo
        else:
            truth_continuous = 0.5 + 0.5 * geo

        implies_coherent = _heyting_implies(truth_value, _OMEGA3_TRUE)

        if not reasons:
            reasons.append("boundary_regular")

        return HeytingDecision(
            verdict=verdict,
            truth_value=float(truth_value),
            boundary_entropy=float(analysis.boundary_entropy),
            coupled_entropy=float(coupled_entropy),
            entropy_upper_bound=float(entropy_ub),
            spectral_gap=float(analysis.spectral_gap),
            condition_number=float(analysis.condition_number),
            exergy_leak_kbt=float(analysis.exergy_leak_kbt),
            physical_exergy_j=float(analysis.physical_exergy_j),
            reference_entropy_threshold=float(threshold),
            allowed_entropy_ceiling=float(ceiling),
            reasons=_unique_reasons(reasons),
            truth_continuous=float(truth_continuous),
            heyting_implies_coherent=float(implies_coherent),
            landauer_bound_j=float(analysis.landauer_bound_j),
            atomic_propositions=tuple(atoms),
        )


# ════════════════════════════════════════════════════════════════════════════
# FASE 3 — ORQUESTACIÓN ASÍNCRONA, ACTUADOR Y CERTIFICADO
# ════════════════════════════════════════════════════════════════════════════


class Phase3AsyncActuationKernel(Phase2SpectralTopologyKernel):
    r"""
    FASE 3: Orquestación asíncrona y actuación física.

    El primer método, `synthesize_certificate`, consume el
    `HeytingDecision` producido por `decide_heyting_verdict` (Fase 2).

    Responsabilidades:
      1. Convertir `HeytingDecision` en `SatelliteTelemetryCertificate`.
      2. Actuación fail-closed: todo veto dispara interlock simulado.
      3. Simular latencia del crowbar GPIO / BT151 con jitter acotado.
      4. Firmar el veredicto con SHA-256 canónico (sin timestamp).
      5. Exponer API síncrona compatible y API asíncrona concurrente.
    """

    def __init__(
        self,
        satellite_id: str,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        crowbar_base_latency_ns: float = _CROWBAR_IRAM_LATENCY_NS,
        crowbar_jitter_ns: float = 3.0,
        crowbar_min_latency_ns: float = 380.0,
        crowbar_max_latency_ns: float = 420.0,
        gpio_pin: str = "GPIO14",
        async_timeout_s: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(satellite_id, dimension_n, safety_margin, **kwargs)

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

    def _simulate_crowbar_latency_ns(self) -> float:
        r"""
        Simula latencia física del crowbar IRAM con jitter gaussiano acotado.

        \[
            \tau = \tau_0 + \mathcal{N}(0, \sigma_\tau^2)
        \]
        con recorte duro al intervalo metrológico permitido.
        """
        if self._crowbar_jitter_ns > 0.0:
            jitter = float(self._rng.normal(0.0, self._crowbar_jitter_ns))
        else:
            jitter = 0.0
        latency = self._crowbar_base_latency_ns + jitter
        return float(
            np.clip(
                latency,
                self._crowbar_min_latency_ns,
                self._crowbar_max_latency_ns,
            )
        )

    def _log_verdict(self, decision: HeytingDecision, latency_ns: float) -> None:
        """Registra el veredicto con el nivel adecuado (crítica / aviso / info)."""
        if decision.verdict == "VETOED":
            logger.critical(
                "¡VETO DE SATÉLITE EN FRONTERA! Satélite=%s | Entropía=%.6f nats | "
                "Entropía acoplada=%.6f nats | Condición=%.3e | Gap=%.3e | "
                "Disyuntor perimetral BT151 [%s] gatillado en %.2f ns.",
                self._id,
                decision.boundary_entropy,
                decision.coupled_entropy,
                decision.condition_number,
                decision.spectral_gap,
                self._gpio_pin,
                latency_ns,
            )
            return

        if decision.verdict == "DEGRADED":
            logger.warning(
                "Frontera degradada. Satélite=%s | Entropía=%.6f nats | "
                "Entropía acoplada=%.6f nats | Condición=%.3e | Gap=%.3e | "
                "Razones=%s",
                self._id,
                decision.boundary_entropy,
                decision.coupled_entropy,
                decision.condition_number,
                decision.spectral_gap,
                ";".join(decision.reasons),
            )
            return

        logger.info(
            "Frontera coherente. Satélite=%s | Entropía=%.6f nats | "
            "Entropía acoplada=%.6f nats | Condición=%.3e | Gap=%.3e",
            self._id,
            decision.boundary_entropy,
            decision.coupled_entropy,
            decision.condition_number,
            decision.spectral_gap,
        )

    def synthesize_certificate(
        self,
        decision: HeytingDecision,
    ) -> SatelliteTelemetryCertificate:
        r"""
        PRIMER MÉTODO DE LA FASE 3.

        Continuación formal de `decide_heyting_verdict` (Fase 2).

        Consume `HeytingDecision` y emite certificado final inmutable.

        Política fail-closed:
          - VETOED  → `hardware_interlock_fired = True` y latencia simulada.
          - DEGRADED → no dispara interlock; se registra advertencia.
          - COHERENT → trazabilidad informativa.

        La huella `decision_sha256` cubre el veredicto físico y no el
        timestamp, de modo que dos evaluaciones isomorfas coincidan.
        """
        interlock_fired = decision.verdict == "VETOED"
        latency = self._simulate_crowbar_latency_ns() if interlock_fired else 0.0
        self._log_verdict(decision, latency)

        fingerprint = _stable_sha256(
            {
                "verdict": decision.verdict,
                "truth_value": decision.truth_value,
                "boundary_entropy": decision.boundary_entropy,
                "coupled_entropy": decision.coupled_entropy,
                "entropy_upper_bound": decision.entropy_upper_bound,
                "spectral_gap": decision.spectral_gap,
                "condition_number": decision.condition_number,
                "exergy_leak_kbt": decision.exergy_leak_kbt,
                "physical_exergy_j": decision.physical_exergy_j,
                "landauer_bound_j": decision.landauer_bound_j,
                "reasons": list(decision.reasons),
                "satellite_id": self._id,
                "dimension": self._n,
            }
        )

        return SatelliteTelemetryCertificate(
            heyting_verdict=decision.verdict,
            boundary_entropy=float(decision.boundary_entropy),
            spectral_gap=float(decision.spectral_gap),
            exergy_leak=float(decision.exergy_leak_kbt),
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=float(latency),
            condition_number=float(decision.condition_number),
            entropy_upper_bound=float(decision.entropy_upper_bound),
            coupled_entropy=float(decision.coupled_entropy),
            physical_exergy_j=float(decision.physical_exergy_j),
            landauer_bound_j=float(decision.landauer_bound_j),
            truth_value=float(decision.truth_value),
            decision_sha256=fingerprint,
            gpio_pin=self._gpio_pin,
            satellite_id=self._id,
            reasons=tuple(decision.reasons),
        )

    def process_satellite_telemetry(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: float = 1.5,
        external_signal: Optional[ExternalEntropySignal] = None,
    ) -> SatelliteTelemetryCertificate:
        r"""
        API síncrona principal compatible con la versión original.

        Flujo completo de las tres fases anidadas:
        \[
            \text{Fase 1}
            \xrightarrow{\texttt{prepare\_boundary\_state}}
            \text{Fase 2}
            \xrightarrow{\texttt{analyze\_boundary\_state}}
            \texttt{decide\_heyting\_verdict}
            \xrightarrow{\texttt{synthesize\_certificate}}
            \text{Fase 3}.
        \]
        """
        state = self.prepare_boundary_state(
            payload,
            K_boundary_raw,
            metric_tensor,
            reference_entropy_threshold=reference_entropy_threshold,
        )
        analysis = self.analyze_boundary_state(state)
        decision = self.decide_heyting_verdict(
            analysis,
            external_signal=external_signal,
            reference_entropy_threshold=reference_entropy_threshold,
        )
        return self.synthesize_certificate(decision)

    def _normalize_async_request(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Extrae y valida las claves admitidas de una petición asíncrona."""
        allowed = {
            "payload",
            "K_boundary_raw",
            "metric_tensor",
            "reference_entropy_threshold",
            "external_signal",
            "timeout_s",
        }
        unknown = set(request.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Claves no admitidas en request asíncrono: {sorted(unknown)}"
            )
        payload = request.get("payload", b"")
        k_raw = request.get("K_boundary_raw")
        metric = request.get("metric_tensor")
        if k_raw is None or metric is None:
            raise ValueError(
                "Cada request requiere 'K_boundary_raw' y 'metric_tensor'."
            )
        return {
            "payload": payload,
            "K_boundary_raw": k_raw,
            "metric_tensor": metric,
            "reference_entropy_threshold": request.get(
                "reference_entropy_threshold", 1.5
            ),
            "external_signal": request.get("external_signal"),
            "timeout_s": request.get("timeout_s"),
        }

    async def monitor_boundary_async(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        metric_tensor: np.ndarray,
        reference_entropy_threshold: float = 1.5,
        external_signal: Optional[ExternalEntropySignal] = None,
        timeout_s: Optional[float] = None,
    ) -> SatelliteTelemetryCertificate:
        r"""
        API asíncrona para monitoreo de frontera.

        Ejecuta el pipeline completo en un hilo independiente para no
        bloquear el lazo de eventos, y aplica timeout opcional.
        """
        worker = asyncio.to_thread(
            self.process_satellite_telemetry,
            payload,
            K_boundary_raw,
            metric_tensor,
            reference_entropy_threshold,
            external_signal,
        )
        timeout = self._async_timeout_s if timeout_s is None else float(timeout_s)
        if timeout > 0.0:
            return await asyncio.wait_for(worker, timeout=timeout)
        return await worker

    async def monitor_batch_async(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        max_concurrency: int = 8,
        return_exceptions: bool = False,
    ) -> List[Any]:
        r"""
        API asíncrona por lotes, con semáforo de concurrencia.

        Cada request debe ser un mapeo compatible con:
        \[
            \{
                \texttt{payload},\;
                \texttt{K\_boundary\_raw},\;
                \texttt{metric\_tensor},\;
                \texttt{reference\_entropy\_threshold},\;
                \texttt{external\_signal},\;
                \texttt{timeout\_s}
            \}.
        \]
        """
        concurrency = max(int(max_concurrency), 1)
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(request: Mapping[str, Any]) -> SatelliteTelemetryCertificate:
            params = self._normalize_async_request(request)
            async with semaphore:
                return await self.monitor_boundary_async(**params)

        tasks = [_run(request) for request in requests]
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))


# ════════════════════════════════════════════════════════════════════════════
# OBSERVADOR PÚBLICO FINAL
# ════════════════════════════════════════════════════════════════════════════


class SatelliteObserver(Phase3AsyncActuationKernel):
    r"""
    Observador de Frontera Asíncrono (Satélite de Telemetría).

    Clase pública final que hereda las tres fases anidadas:
      - Fase 1: Metrología fundamental (`prepare_boundary_state`).
      - Fase 2: Espectro topológico y Heyting (`analyze_boundary_state`,
        `decide_heyting_verdict`).
      - Fase 3: Actuación asíncrona y certificado (`synthesize_certificate`).

    Ejemplo síncrono
    ----------------
    >>> observer = SatelliteObserver("SAT-01", dimension_n=4)
    >>> cert = observer.process_satellite_telemetry(
    ...     payload=b"boundary-payload",
    ...     K_boundary_raw=np.eye(4),
    ...     metric_tensor=np.eye(4),
    ... )
    >>> cert.heyting_verdict

    Ejemplo asíncrono
    -----------------
    >>> cert = await observer.monitor_boundary_async(
    ...     payload=b"boundary-payload",
    ...     K_boundary_raw=np.eye(4),
    ...     metric_tensor=np.eye(4),
    ... )
    """


__all__ = [
    "ExternalEntropySignal",
    "EntropyEstimate",
    "MetricRegularization",
    "BoundaryState",
    "SpectralBoundaryAnalysis",
    "HeytingDecision",
    "SatelliteTelemetryCertificate",
    "Phase1MetrologyKernel",
    "Phase2SpectralTopologyKernel",
    "Phase3AsyncActuationKernel",
    "SatelliteObserver",
]
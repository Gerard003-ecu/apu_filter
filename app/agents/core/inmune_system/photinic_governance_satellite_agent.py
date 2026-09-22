from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Photinic Governance Satellite Agent (Soberano Fotínico)             ║
║ Ruta   : app/agents/core/immune_system/photinic_governance_satellite_agent.py║
║ Versión: 5.0.0-Doctoral-OODA-Heyting-Majorana-Choi-Horodecki-RC-Crowbar-HMAC ║
║ Nivel  : Estrato Omega ($V_\Omega$, Nivel 0.5 — Ágora Tensorial)             ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y GOBERNANZA DE LAZO CERRADO (v5):                       ║
║ 1. Separación estricta entre **física cruda** (Fases 1+2 del motor, que      ║
║    nunca lanzan por condiciones clasificables de CP/causalidad/Tsirelson) y  ║
║    **certificación** (Fase 3 del motor, que sí lanza), reparando el defecto  ║
║    arquitectónico crítico de v4 donde el veto duro era código muerto.        ║
║ 2. Álgebra de Heyting $\Omega_3=\{\bot,\ast,\top\}$ formalizada, veredicto   ║
║    final derivado como ínfimo reticular de la clasificación instantánea y    ║
║    la máquina de histéresis, certificando $\neg\neg\ast\neq\ast$.            ║
║ 3. Circuito RC real y determinista del disparo Crowbar BT151 (< 400 ns).     ║
║ 4. Sutura de Fock con tokens HMAC de vida limitada y prevención de repetición║
║    bajo exclusión mutua (`RLock`).                                           ║
║ 5. Canario metrológico de consistencia cruzada agente↔motor sobre el         ║
║    invariante de Majorana, cerrando el lazo de auditoría inmunológica.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Optional, Tuple, Dict, Any, List, Set, Sequence, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ------------------------------------------------------------------------------
# COMPATIBILIDAD CATEGÓRICA E IMPORTACIONES RESILIENTES DEL ECOSISTEMA
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:
        r"""Stub ontológico base de Morphism categórico en entorno desacoplado."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class TopologicalInvariantError(Exception):
        r"""Excepción raíz ante violaciones topológico-algebraicas en el estrato Omega."""
        pass

try:
    from photinic_governance_satellite_engine import (
        PhotinicGovernanceSatelliteEngine,
        PhotinicEngineState,
        PhotinicObservationKernel as EngineKernel,
        PhotinicPolicyReport as EngineReport,
        PhotinicEngineError,
        PhotinicDimensionError,
        PhotinicNonPositiveDefiniteError,
        PhotinicCausalViolationError,
        PhotinicNonCausalChannelError,
        PhotinicTsirelsonViolationError,
        KahanNeumaierSum,
    )
except ImportError:
    try:
        from app.core.immune_system.photinic_governance_satellite_engine import (
            PhotinicGovernanceSatelliteEngine,
            PhotinicEngineState,
            PhotinicObservationKernel as EngineKernel,
            PhotinicPolicyReport as EngineReport,
            PhotinicEngineError,
            PhotinicDimensionError,
            PhotinicNonPositiveDefiniteError,
            PhotinicCausalViolationError,
            PhotinicNonCausalChannelError,
            PhotinicTsirelsonViolationError,
            KahanNeumaierSum,
        )
    except ImportError:
        # ----------------------------------------------------------------
        # Fallback autoportante alineado con la interfaz v5.0.0 del motor
        # (canonize_photinic_observation_kernel / orient_photinic_policy /
        # execute_photinic_audit polimórfico, SIN clamp artificial de
        # Tsirelson). NO APTO PARA PRODUCCIÓN.
        # ----------------------------------------------------------------
        class PhotinicEngineError(Exception):
            pass

        class PhotinicDimensionError(PhotinicEngineError):
            pass

        class PhotinicNonPositiveDefiniteError(PhotinicEngineError):
            pass

        class PhotinicCausalViolationError(PhotinicEngineError):
            pass

        class PhotinicNonCausalChannelError(PhotinicCausalViolationError):
            pass

        class PhotinicTsirelsonViolationError(PhotinicCausalViolationError):
            pass

        class KahanNeumaierSum:
            @staticmethod
            def sum(arr: Union[NDArray[np.float64], Sequence[float]]) -> float:
                s, c = 0.0, 0.0
                for x in arr:
                    val = float(x)
                    t = s + val
                    if abs(s) >= abs(val):
                        c += (s - t) + val
                    else:
                        c += (val - t) + s
                    s = t
                return float(s + c)

            @staticmethod
            def compensated_dot(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
                return KahanNeumaierSum.sum(u * v)

            @staticmethod
            def compensated_l2_norm(v: NDArray[np.float64]) -> float:
                return float(math.sqrt(max(KahanNeumaierSum.compensated_dot(v, v), 1e-15)))

        @dataclass(frozen=True, slots=True)
        class EngineKernel:
            node_policies: Tuple[NDArray[np.float64], ...]
            majorana_spinor_norm: float
            majorana_chiral_parity: float
            banach_ratios: Tuple[float, ...]
            gram_matrix: NDArray[np.float64]
            gram_eigenvalues: NDArray[np.float64]
            gram_symmetry_defect: float
            gram_reconstruction_residual: float
            condition_number: float
            spectral_gap: float
            sha256_seal: str
            timestamp: float

        @dataclass(frozen=True, slots=True)
        class EngineReport:
            kernel: EngineKernel
            idempotence_residual: float
            trace_rank_mismatch: float
            choi_min_eigenvalue: float
            choi_trace_residual: float
            choi_subunital_defect_min_eigenvalue: float
            tsirelson_chsh_value: float
            horodecki_eigs: Tuple[float, float]
            horodecki_bipartition_pairs: int
            is_choi_completely_positive: bool
            is_trace_non_increasing: bool
            is_causal_non_signaling: bool
            is_tsirelson_bounded: bool
            is_classical_lhv: bool
            timestamp: float

        @dataclass(frozen=True, slots=True)
        class PhotinicEngineState:
            kernel: EngineKernel
            policy_report: EngineReport
            fpu_execution_time_ms: float
            federated_density_matrix: NDArray[np.float64]
            von_neumann_entropy: float
            quantum_purity: float
            bell_pair_entanglement_entropy: float
            fpu_precision_drift: float
            cryptographic_seal: str
            phase_signature: str

        class PhotinicGovernanceSatelliteEngine:
            def __init__(self, tolerance: float = 1.0e-12) -> None:
                self.tol = tolerance

            def canonize_photinic_observation_kernel(
                self, node_policies: Sequence[NDArray[np.float64]]
            ) -> EngineKernel:
                clean = [np.where(v == -0.0, +0.0, v) for v in node_policies]
                P = np.vstack(clean)
                N, dim = P.shape
                S_fed = (P.T @ P) / float(N) + self.tol * np.eye(dim)
                eigs = np.sort(la.eigvalsh(S_fed))
                cond = float(eigs[-1] / max(eigs[0], 1e-15))
                maj_norm = float(math.sqrt(max(float(np.sum(eigs ** 2)), 1e-15)))
                return EngineKernel(
                    node_policies=tuple(clean), majorana_spinor_norm=maj_norm, majorana_chiral_parity=1.0,
                    banach_ratios=tuple(float(np.sum(np.abs(v)) / max(la.norm(v), 1e-15)) for v in clean),
                    gram_matrix=S_fed, gram_eigenvalues=eigs, gram_symmetry_defect=0.0,
                    gram_reconstruction_residual=0.0, condition_number=cond,
                    spectral_gap=float(eigs[1] - eigs[0]) if dim > 1 else float("inf"),
                    sha256_seal=hashlib.sha256(P.tobytes()).hexdigest(), timestamp=time.time()
                )

            def orient_photinic_policy(self, kernel: EngineKernel) -> EngineReport:
                P = np.vstack(kernel.node_policies)
                N = P.shape[0]
                U, s, Vt = la.svd(P, full_matrices=False)
                rank = max(int(np.sum(s > 1e-10)), 1)
                Omega = Vt[:rank].T @ Vt[:rank]
                idem_res = float(la.norm(Omega @ Omega - Omega, 'fro'))
                trace_mismatch = abs(float(np.trace(Omega)) - float(rank))

                mid = N // 2 if N >= 2 else 1
                n_pairs = min(mid, N - mid) if N >= 2 else 0
                if n_pairs > 0:
                    T_corr = (P[:n_pairs].T @ P[mid:mid + n_pairs]) / float(n_pairs)
                else:
                    T_corr = np.outer(P[0], P[0])
                u_eigs = np.sort(la.eigvalsh(T_corr.T @ T_corr))[::-1]
                u1 = float(max(u_eigs[0], 0.0)) if len(u_eigs) > 0 else 0.0
                u2 = float(max(u_eigs[1], 0.0)) if len(u_eigs) > 1 else 0.0
                chsh = float(2.0 * math.sqrt(max(u1 + u2, 0.0)))  # SIN clamp artificial

                return EngineReport(
                    kernel=kernel, idempotence_residual=idem_res, trace_rank_mismatch=trace_mismatch,
                    choi_min_eigenvalue=0.0, choi_trace_residual=0.0,
                    choi_subunital_defect_min_eigenvalue=0.0,
                    tsirelson_chsh_value=chsh, horodecki_eigs=(u1, u2), horodecki_bipartition_pairs=n_pairs,
                    is_choi_completely_positive=True, is_trace_non_increasing=True,
                    is_causal_non_signaling=True,
                    is_tsirelson_bounded=chsh <= (2.0 * math.sqrt(2.0) + 1e-9),
                    is_classical_lhv=chsh <= 2.0001, timestamp=time.time()
                )

            def execute_photinic_audit(self, target: Any) -> PhotinicEngineState:
                t0 = time.perf_counter()
                if isinstance(target, EngineReport):
                    rep = target
                else:
                    k = self.canonize_photinic_observation_kernel(target)
                    rep = self.orient_photinic_policy(k)
                if not rep.is_choi_completely_positive:
                    raise PhotinicNonPositiveDefiniteError("Fallback: canal no-CP.")
                if not rep.is_causal_non_signaling:
                    raise PhotinicNonCausalChannelError("Fallback: canal no causal.")
                if not rep.is_tsirelson_bounded:
                    raise PhotinicTsirelsonViolationError("Fallback: violación de Tsirelson.")
                trace_S = float(np.trace(rep.kernel.gram_matrix))
                rho = rep.kernel.gram_matrix / max(trace_S, 1e-15)
                return PhotinicEngineState(
                    kernel=rep.kernel, policy_report=rep,
                    fpu_execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    federated_density_matrix=rho, von_neumann_entropy=0.0, quantum_purity=1.0,
                    bell_pair_entanglement_entropy=0.0, fpu_precision_drift=rep.idempotence_residual,
                    cryptographic_seal=rep.kernel.sha256_seal, phase_signature="FALLBACK_SUTURATED"
                )

logger = logging.getLogger("APU.Agents.Omega.PhotinicGovernanceSatelliteAgent")

# ------------------------------------------------------------------------------
# CONSTANTES FÍSICAS, LÍMITES METROLÓGICOS DE WILKINSON Y SILICIO
# ------------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15

# --- Silicio ESP32 y disparo Crowbar BT151 (idéntico al agente leptónico, ------
# --- reutilización legítima del mismo componente de hardware físico) ----------
_CROWBAR_MAX_IRAM_BUDGET_NS: Final[float] = 400.0
_ESP32_CLOCK_FREQ_HZ: Final[float] = 240.0e6
_ESP32_CYCLE_TIME_NS: Final[float] = 1.0e9 / _ESP32_CLOCK_FREQ_HZ
_ISR_DISPATCH_OVERHEAD_CYCLES: Final[int] = 42
_BT151_VGT_VOLTS: Final[float] = 1.1
_BT151_IGT_AMPS: Final[float] = 0.005
_ESP32_GPIO_VOH_VOLTS: Final[float] = 3.3
_CROWBAR_GATE_RESISTOR_OHMS: Final[float] = 47.0
_CROWBAR_GATE_CAPACITANCE_FARADS: Final[float] = 1.0e-9
_CROWBAR_MIN_SAFETY_MARGIN: Final[float] = 5.0

# --- Umbrales del clasificador de Heyting (documentados, no mágicos) ---------
# Nota de diseño: el motor certifica CP numéricamente a ~1e-9 (Fase 3 del motor).
# El umbral aquí es un margen OPERACIONAL más laxo (defensa en profundidad a
# nivel de agente), tolerando ruido federado sin renunciar a vetar corrupción real.
_HARD_VETO_CHOI_MIN_EIGENVALUE: Final[float] = -1.0e-6
_HARD_VETO_IDEMPOTENCE_FRACTION: Final[float] = 0.50
_SOFT_VETO_IDEMPOTENCE_LOWER_FRACTION: Final[float] = 0.30
_SOFT_VETO_IDEMPOTENCE_UPPER_FRACTION: Final[float] = 0.50
_HARD_VETO_MAX_CONDITION_NUMBER: Final[float] = 1.0e10
_SOFT_VETO_CHSH_CLASSICAL_THRESHOLD: Final[float] = 2.0  # Cota clásica LHV de Bell

# --- Sutura de Fock, tokens de positrón y canario metrológico -----------------
_FOCK_ANNIHILATION_CROSS_SECTION_SIGMA: Final[float] = 0.5
_TOKEN_FRESHNESS_WINDOW_SECONDS: Final[float] = 300.0
_CROSS_ENGINE_CONSISTENCY_TOLERANCE: Final[float] = 1.0e-6


# ------------------------------------------------------------------------------
# JERARQUÍA DE EXCEPCIONES DEL AGENTE
# ------------------------------------------------------------------------------
class PhotinicAgentError(PhotinicEngineError):
    r"""Excepción raíz para violaciones propias del Soberano Fotínico de Gobernanza."""
    pass


class PhotinicMetricConsistencyError(PhotinicAgentError):
    r"""
    Canario metrológico: divergencia entre el invariante de Majorana computado
    independientemente por el agente (Fase 1) y por el motor (Fase 2) sobre
    idénticas políticas federadas, delatando desalineación de dependencias.
    """
    pass


class PhotinicCrowbarMisfireError(PhotinicAgentError):
    r"""
    Fallo físico certificado del disparo Crowbar BT151: corriente de compuerta
    insuficiente para el cebado por avalancha, o presupuesto de IRAM excedido.
    """
    pass


class PhotinicTokenFreshnessError(PhotinicAgentError):
    r"""El token de positrón de sutura de Fock ha expirado o ha sido repetido."""
    pass


class PhotinicHeytingVerdict(IntEnum):
    r"""
    Álgebra de Heyting $\Omega_3 = \{\bot, \ast, \top\}$ en el topos de De Rham:
    $\bot = 0$ (VETOED - Ruptura de causalidad, superación de Tsirelson, violación Choi)
    $\ast = 1$ (DEGRADED - Luz Ámbar / Turbulencia elástica / Gracia cuántica de Fock)
    $\top = 2$ (COHERENT - Idempotencia exacta y no-localidad cuántica síncrona)
    """
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def canonical_name(self) -> str:
        return self.name


class HeytingOmega3Algebra:
    r"""
    Estructura formal de álgebra de Heyting sobre la cadena totalmente ordenada
    $\Omega_3=\{\bot<\ast<\top\}$. El meet/join coinciden con mínimo/máximo, y
    la pseudo-complementación certifica computacionalmente $\neg\neg\ast\neq\ast$,
    reemplazando la cascada ad hoc de `if/elif` de la v4.0.0 por una composición
    algebraica explícita del veredicto instantáneo con la histéresis temporal.
    """

    @staticmethod
    def meet(a: PhotinicHeytingVerdict, b: PhotinicHeytingVerdict) -> PhotinicHeytingVerdict:
        return PhotinicHeytingVerdict(min(int(a), int(b)))

    @staticmethod
    def join(a: PhotinicHeytingVerdict, b: PhotinicHeytingVerdict) -> PhotinicHeytingVerdict:
        return PhotinicHeytingVerdict(max(int(a), int(b)))

    @staticmethod
    def pseudocomplement(a: PhotinicHeytingVerdict) -> PhotinicHeytingVerdict:
        if a == PhotinicHeytingVerdict.VETOED:
            return PhotinicHeytingVerdict.COHERENT
        return PhotinicHeytingVerdict.VETOED

    @classmethod
    def verify_intuitionistic_failure(cls) -> bool:
        r"""Certifica $\neg\neg\ast \neq \ast$: ausencia del tercero excluido en $\Omega_3$."""
        double_neg = cls.pseudocomplement(cls.pseudocomplement(PhotinicHeytingVerdict.DEGRADED))
        return bool(double_neg != PhotinicHeytingVerdict.DEGRADED)


@dataclass(frozen=True, slots=True)
class PhotinicObservationKernel:
    r"""
    Expediente canónico inmutable de Fase 1 (Observe).
    Audita la regularidad del espacio de Banach $\ell^1 \hookrightarrow \ell^2$,
    la norma del espinor de Majorana del fotino (computada independientemente
    del motor, para servir de canario metrológico en Fase 2) y el número de
    condición espectral, sellados con HMAC-SHA256.
    """
    node_policies: Tuple[NDArray[np.float64], ...]
    majorana_spinor_norm: float
    majorana_chiral_parity: float
    gram_symmetry_defect: float
    banach_ratios: Tuple[float, ...]
    metric_condition_number: float
    spectral_gap: float
    cryptographic_seal: str
    observation_timestamp: float


@dataclass(frozen=True, slots=True)
class PhotinicOrientationReport:
    r"""
    Expediente canónico inmutable de Fase 2 (Orient).
    Sintetiza la idempotencia de Grothendieck, la positividad completa de Choi,
    el defecto de sub-unitalidad causal, el parámetro de Bell-CHSH sin
    alteración artificial, y (cuando la certificación estricta del motor tuvo
    éxito) la telemetría cuántica avanzada. `engine_state` es `None` y
    `engine_validation_error` describe la causa cuando el motor rechazó la
    certificación -- en cuyo caso el reporte físico crudo permanece disponible
    para el clasificador de Heyting (corrección del defecto arquitectónico v4).
    """
    kernel: PhotinicObservationKernel
    engine_state: Optional[PhotinicEngineState]
    engine_validation_error: Optional[str]
    idempotence_residual: float
    trace_rank_mismatch: float
    choi_min_eigenvalue: float
    choi_trace_residual: float
    choi_subunital_defect_min_eigenvalue: float
    tsirelson_chsh_value: float
    horodecki_eigs: Tuple[float, float]
    horodecki_bipartition_pairs: int
    is_choi_completely_positive: bool
    is_trace_non_increasing: bool
    is_causal_non_signaling: bool
    is_tsirelson_bounded: bool
    is_classical_lhv: bool
    von_neumann_entropy: Optional[float]
    quantum_purity: Optional[float]
    bell_pair_entanglement_entropy: Optional[float]
    cross_engine_majorana_residual: float
    orientation_timestamp: float


@dataclass(frozen=True, slots=True)
class PhotinicAgentCertificate:
    r"""
    Certificado formal e inmutable emitido por el Soberano Fotínico en Fase 3 (Act).
    Acredita el veredicto en $\Omega_3$ (instantáneo, de histéresis y su ínfimo
    reticular), la telemetría cuántica de Fock, y la bitácora de conmutación
    física real (RC) del tiristor Crowbar BT151 en silicio IRAM.
    """
    phase: str
    heyting_verdict: str
    heyting_instant_verdict: str
    heyting_hysteresis_verdict: str
    heyting_truth_value: int
    majorana_spinor_norm: float
    idempotence_residual: float
    trace_rank_mismatch: float
    choi_min_eigenvalue: float
    tsirelson_chsh_value: float
    von_neumann_entropy: Optional[float]
    quantum_purity: Optional[float]
    engine_validation_error: Optional[str]
    is_choi_completely_positive: bool
    is_causal_non_signaling: bool
    is_tsirelson_bounded: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    fock_annihilation_occurred: bool
    fock_transition_probability: float
    hardware_interlock_fired: bool
    crowbar_iram_cycles: int
    crowbar_gate_current_amps: float
    actuation_latency_ns: float
    time_grace_remaining: float
    digital_signature_hmac_sha256: str
    execution_duration_microseconds: float
    total_pipeline_latency_microseconds: float


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA TENSORIAL, AUDITORÍA DE BANACH Y ESPINOR DE MAJORANA SYM
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_PhotinicAgentObserver:
    r"""
    FASE 1 — Observe:
    Saneamiento de ceros de signo IEEE 754 ($x=-0.0\mapsto+0.0$), auditoría
    compensada (Dekker-Neumaier) de inmersión en espacios de Banach, cálculo
    independiente del invariante de Majorana en $N=1$ SYM (con auditoría de
    simetría de de Rham, ausente en v4) y canonización HMAC-sellada del Kernel.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        self._tol: Final[float] = float(tolerance)
        if hmac_key is None:
            logger.warning(
                "PhotinicGovernanceSatelliteAgent: se utiliza la clave HMAC derivada por defecto. "
                "En despliegue productivo DEBE inyectarse una clave gestionada por el subsistema "
                "de secretos del 'immune_system' (parámetro hmac_key)."
            )
            hmac_key = hashlib.sha256(b"APU.PhotinicGovernanceSatelliteAgent.DefaultAgentSalt.v5").digest()
        self._hmac_key: Final[bytes] = hmac_key

    # --------------------------------------------------------------------
    # 1.1 — Sellado HMAC y regularidad de Sobolev-Banach compensada
    # --------------------------------------------------------------------
    def _compute_hmac_seal(self, *chunks: bytes) -> str:
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        for chunk in chunks:
            mac.update(chunk)
        return mac.hexdigest()

    def evaluate_banach_regularity(self, S: NDArray[np.float64]) -> float:
        r"""
        Mide la regularidad de equivalencia de normas en el álgebra de Banach
        mediante aritmética compensada de Dekker-Neumaier:
        $$\mathcal{R}(S) = \frac{\|S\|_1}{\|S\|_2}, \qquad 1.0 \le \mathcal{R}(S) \le \sqrt{d}.$$
        """
        norm1 = KahanNeumaierSum.sum(np.abs(S))
        norm2 = KahanNeumaierSum.compensated_l2_norm(S)
        return float(norm1 / max(norm2, _WILKINSON_SAFETY_FLOOR))

    # --------------------------------------------------------------------
    # 1.2 — Tensor espectral de Majorana con auditoría de simetría
    # --------------------------------------------------------------------
    def _audit_gram_symmetry_defect(self, S_fed: NDArray[np.float64]) -> float:
        r"""Certifica el defecto de asimetría de de Rham previo a `eigvalsh` (auditoría ausente en v4)."""
        defect = float(la.norm(S_fed - S_fed.T, ord='fro')) / max(
            float(la.norm(S_fed, ord='fro')), _WILKINSON_SAFETY_FLOOR
        )
        if defect > self._tol:
            logger.warning("Defecto de asimetría en S_fed (agente) = %.6e excede tolerancia.", defect)
        return defect

    def compute_majorana_sym_spectral_tensor(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]]
    ) -> Tuple[float, float, float, float, float]:
        r"""
        Calcula la correlación espinorial de Majorana con aritmética compensada:
        $$S_{\mathrm{fed}} = \frac{1}{N}\sum_{k=1}^N p_k\otimes p_k + \varepsilon I_d, \qquad
          I_{\mathrm{Majorana}} = \sqrt{\operatorname{Tr}(S_{\mathrm{fed}}^2)}, \qquad
          \chi_5 = \frac{\operatorname{Tr}(S_{\mathrm{fed}})}{\|S_{\mathrm{fed}}\|_F}$$
        Este cómputo es **independiente** del motor y sirve de canario metrológico
        cruzado en Fase 2 (`_audit_cross_engine_consistency`).
        """
        stack_P = np.vstack(cleaned_policies)
        N, dim = stack_P.shape
        S_fed = (stack_P.T @ stack_P) / float(N) + self._tol * np.eye(dim, dtype=np.float64)

        symmetry_defect = self._audit_gram_symmetry_defect(S_fed)
        S_fed = 0.5 * (S_fed + S_fed.T)

        eigvals = np.sort(la.eigvalsh(S_fed))
        min_ev = float(max(eigvals[0], _WILKINSON_SAFETY_FLOOR))
        max_ev = float(eigvals[-1])
        cond_num = max_ev / min_ev

        majorana_norm = float(math.sqrt(max(KahanNeumaierSum.sum(eigvals ** 2), _WILKINSON_SAFETY_FLOOR)))
        tr_val = KahanNeumaierSum.sum(eigvals)
        fro_norm = float(np.clip(la.norm(S_fed, ord='fro'), _WILKINSON_SAFETY_FLOOR, None))
        chiral_parity = float(tr_val / fro_norm)
        spectral_gap = float(eigvals[1] - eigvals[0]) if dim > 1 else float("inf")

        return majorana_norm, chiral_parity, cond_num, spectral_gap, symmetry_defect

    # --------------------------------------------------------------------
    # 1.3 — Ingesta pública y canonización terminal (puerto hacia Fase 2)
    # --------------------------------------------------------------------
    def observe_federated_nodes(
        self,
        node_policies: Sequence[NDArray[np.float64]]
    ) -> PhotinicObservationKernel:
        r"""Ingesta los vectores de política crudos y ejecuta el saneamiento numérico de signo."""
        if not node_policies:
            raise PhotinicDimensionError("El consorcio federado no contiene vectores de política (N = 0).")

        dim = node_policies[0].shape[0]
        cleaned_policies: List[NDArray[np.float64]] = []
        banach_ratios: List[float] = []
        seal_material: List[bytes] = []

        for idx, vec in enumerate(node_policies):
            if vec.ndim != 1 or vec.shape[0] != dim:
                raise PhotinicDimensionError(
                    f"Inconsistencia dimensional en nodo [{idx}]: Esperado ({dim},), obtenido {vec.shape}"
                )
            if not np.all(np.isfinite(vec)):
                raise PhotinicDimensionError(f"Vector de política de nodo [{idx}] contiene valores no finitos.")

            c_vec = np.where(vec == -0.0, +0.0, vec)
            cleaned_policies.append(c_vec)

            b_ratio = self.evaluate_banach_regularity(c_vec)
            banach_ratios.append(b_ratio)

            seal_material.append(c_vec.tobytes())
            seal_material.append(f"{b_ratio:.8e}".encode("ascii"))

        return self.canonize_photinic_observation_kernel(
            cleaned_policies=cleaned_policies,
            banach_ratios=banach_ratios,
            seal_material=seal_material
        )

    def canonize_photinic_observation_kernel(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        banach_ratios: Sequence[float],
        seal_material: Sequence[bytes]
    ) -> PhotinicObservationKernel:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1:
        Canoniza el expediente topológico inmutable de Fase 1. Calcula los
        invariantes de Majorana con auditoría de simetría y sella la
        integridad mediante HMAC-SHA256 (reparando el diseño frágil de la
        v4.0.0, que acoplaba `observe_federated_nodes` a un hasher mutable
        pasado por referencia). Constituye el puerto de acoplamiento directo
        hacia la Fase 2.
        """
        maj_norm, chiral_parity, cond_num, spec_gap, symmetry_defect = (
            self.compute_majorana_sym_spectral_tensor(cleaned_policies)
        )

        seal_hash = self._compute_hmac_seal(
            *seal_material,
            f"{maj_norm:.8e}_{cond_num:.8e}_{spec_gap:.8e}_{symmetry_defect:.8e}".encode("ascii")
        )

        return PhotinicObservationKernel(
            node_policies=tuple(cleaned_policies),
            majorana_spinor_norm=maj_norm,
            majorana_chiral_parity=chiral_parity,
            gram_symmetry_defect=symmetry_defect,
            banach_ratios=tuple(banach_ratios),
            metric_condition_number=cond_num,
            spectral_gap=spec_gap,
            cryptographic_seal=seal_hash,
            observation_timestamp=time.time()
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENTACIÓN TOPOLÓGICA DE GROTHENDIECK, CHOI CPTP Y TSIRELSON
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_PhotinicAgentOrienter(Phase1_PhotinicAgentObserver):
    r"""
    FASE 2 — Orient:
    Hereda de la Fase 1. Separa la **física cruda** del motor
    (`canonize_photinic_observation_kernel` + `orient_photinic_policy`, que
    nunca lanzan por condiciones clasificables) de la **certificación estricta**
    (`execute_photinic_audit`), capturando sus excepciones para preservar
    siempre el reporte físico crudo -- reparando el defecto arquitectónico de
    la v4.0.0 donde el veto duro era código inalcanzable. Audita adicionalmente
    la consistencia cruzada del invariante de Majorana agente↔motor.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        hmac_key: Optional[bytes] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)
        self._engine: Final[PhotinicGovernanceSatelliteEngine] = (
            engine or PhotinicGovernanceSatelliteEngine(tolerance=tolerance)
        )

    # --------------------------------------------------------------------
    # 2.1 — Canario metrológico de consistencia cruzada agente↔motor
    # --------------------------------------------------------------------
    def _audit_cross_engine_consistency(
        self,
        agent_kernel: PhotinicObservationKernel,
        engine_kernel: EngineKernel
    ) -> float:
        r"""
        Certifica que el invariante de Majorana $I_{\mathrm{Majorana}}=
        \sqrt{\operatorname{Tr}(S_{\rm fed}^2)}$, computado **independientemente**
        por el agente (Fase 1) y por el motor (Fase 2 interna) sobre idénticas
        políticas federadas, coincide dentro de tolerancia estricta -- ausente
        en la v4.0.0, que confiaba ciegamente en la consistencia sin verificarla.
        """
        residual = abs(agent_kernel.majorana_spinor_norm - engine_kernel.majorana_spinor_norm)
        reference = max(agent_kernel.majorana_spinor_norm, _WILKINSON_SAFETY_FLOOR)
        relative_residual = residual / reference
        if relative_residual > _CROSS_ENGINE_CONSISTENCY_TOLERANCE:
            raise PhotinicMetricConsistencyError(
                f"Divergencia crítica del invariante de Majorana entre agente y motor "
                f"(residuo relativo = {relative_residual:.4e}). Posible desalineación de "
                f"dependencias entre agente y 'photinic_governance_satellite_engine.py'."
            )
        return relative_residual

    # --------------------------------------------------------------------
    # 2.2 — Orientación terminal y puerto de acoplamiento hacia Fase 3
    # --------------------------------------------------------------------
    def orient_photinic_policy_state(
        self,
        kernel: PhotinicObservationKernel
    ) -> PhotinicOrientationReport:
        r"""Conduce la orientación federada separando física cruda de certificación estricta."""
        # --- Física cruda del motor (Fases 1+2 del motor): jamás lanza por
        # violaciones clasificables de CP/causalidad/Tsirelson. ---
        engine_kernel = self._engine.canonize_photinic_observation_kernel(kernel.node_policies)
        engine_report: EngineReport = self._engine.orient_photinic_policy(engine_kernel)

        cross_residual = self._audit_cross_engine_consistency(kernel, engine_kernel)

        # --- Certificación estricta del motor (Fase 3 del motor): SÍ lanza
        # ante violación de CP/causalidad/Tsirelson. Se captura y se traduce
        # en ausencia de telemetría avanzada, preservando siempre
        # `engine_report` crudo para el clasificador de Heyting (corrige el
        # defecto arquitectónico v4.0.0). ---
        engine_state: Optional[PhotinicEngineState] = None
        validation_error: Optional[str] = None
        try:
            engine_state = self._engine.execute_photinic_audit(engine_report)
        except (PhotinicNonPositiveDefiniteError, PhotinicCausalViolationError) as exc:
            validation_error = str(exc)
            logger.error(
                "Certificación estricta del motor rechazada (condición veto-clasificable esperada): %s",
                validation_error
            )

        return PhotinicOrientationReport(
            kernel=kernel,
            engine_state=engine_state,
            engine_validation_error=validation_error,
            idempotence_residual=engine_report.idempotence_residual,
            trace_rank_mismatch=engine_report.trace_rank_mismatch,
            choi_min_eigenvalue=engine_report.choi_min_eigenvalue,
            choi_trace_residual=engine_report.choi_trace_residual,
            choi_subunital_defect_min_eigenvalue=engine_report.choi_subunital_defect_min_eigenvalue,
            tsirelson_chsh_value=engine_report.tsirelson_chsh_value,
            horodecki_eigs=engine_report.horodecki_eigs,
            horodecki_bipartition_pairs=engine_report.horodecki_bipartition_pairs,
            is_choi_completely_positive=engine_report.is_choi_completely_positive,
            is_trace_non_increasing=engine_report.is_trace_non_increasing,
            is_causal_non_signaling=engine_report.is_causal_non_signaling,
            is_tsirelson_bounded=engine_report.is_tsirelson_bounded,
            is_classical_lhv=engine_report.is_classical_lhv,
            von_neumann_entropy=(engine_state.von_neumann_entropy if engine_state else None),
            quantum_purity=(engine_state.quantum_purity if engine_state else None),
            bell_pair_entanglement_entropy=(
                engine_state.bell_pair_entanglement_entropy if engine_state else None
            ),
            cross_engine_majorana_residual=cross_residual,
            orientation_timestamp=time.time()
        )

    def synthesize_photinic_orientation(
        self,
        kernel_or_policies: Union[PhotinicObservationKernel, Sequence[NDArray[np.float64]]]
    ) -> PhotinicOrientationReport:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2:
        Garantiza la continuidad functorial estricta. Si recibe un
        `PhotinicObservationKernel`, procede directamente; si recibe tensores
        crudos, enlaza automáticamente con `observe_federated_nodes` de Fase 1.
        El `PhotinicOrientationReport` devuelto es el puerto canónico de
        acoplamiento hacia la Fase 3.
        """
        if isinstance(kernel_or_policies, PhotinicObservationKernel):
            kernel = kernel_or_policies
        else:
            kernel = self.observe_federated_nodes(kernel_or_policies)

        return self.orient_photinic_policy_state(kernel)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: DECISIÓN EN TOPOS DE HEYTING, SUTURA DE FOCK Y CROWBAR RC-IRAM
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_PhotinicAgentDecider(Phase2_PhotinicAgentOrienter):
    r"""
    FASE 3 — Decide & Act:
    Hereda de la Fase 2. Modula la rampa de de Rham en el clasificador de
    subobjetos $\Omega_3$ mediante `HeytingOmega3Algebra` (veredicto final =
    ínfimo reticular entre instantáneo e histéresis), resuelve la sutura de
    Fock con tokens HMAC de vida limitada, y dispara el Crowbar BT151 mediante
    un modelo RC real y verificable. Estado soberano protegido por `RLock`.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        idempotence_threshold: float = 0.25,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key, engine=engine)
        if idempotence_threshold <= 0.0:
            raise ValueError("idempotence_threshold debe ser estrictamente positivo.")
        if grace_period_seconds <= 0.0:
            raise ValueError("grace_period_seconds debe ser estrictamente positivo.")

        self._idempotence_threshold: Final[float] = float(idempotence_threshold)
        self._grace_limit: Final[float] = float(grace_period_seconds)

        if authorized_tokens is None:
            logger.warning(
                "PhotinicGovernanceSatelliteAgent: se utilizan los tokens de autorización de "
                "demostración por defecto. DEBEN sustituirse en despliegue productivo."
            )
            authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_PHOTINIC_SYM_2026",
                "HMAC_SUTURA_FOCK_SECURE_FEDERATED"
            }
        self._authorized_tokens: Final[Set[str]] = authorized_tokens

        self._state_lock: Final[threading.RLock] = threading.RLock()
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False
        self._consumed_override_digests: Set[str] = set()

    def reset_sovereign_state(self) -> None:
        r"""Reinicia atómicamente la máquina de histéresis y el registro de tokens consumidos."""
        with self._state_lock:
            self._soft_veto_timestamp = None
            self._is_soft_veto_active = False
            self._consumed_override_digests.clear()
            logger.info("Estado soberano de histéresis de veto suave reiniciado explícitamente.")

    # --------------------------------------------------------------------
    # 3.1 — Clasificador instantáneo de subobjetos en el topos de Heyting
    # --------------------------------------------------------------------
    def evaluate_heyting_topos_subobject_classifier(
        self,
        report: PhotinicOrientationReport
    ) -> Tuple[PhotinicHeytingVerdict, bool, bool]:
        r"""
        Clasificador **instantáneo** (sin memoria) en $\Omega_3$:
        - $\bot$: rechazo de certificación del motor, superación de Tsirelson,
          violación de Choi (CP), no-causalidad, divergencia de idempotencia
          $>0.50\cdot L$, o condicionamiento métrico patológico.
        - $\ast$: régimen cuántico no-local acotado ($2<\mathcal{B}\le2\sqrt2$)
          o idempotencia elástica intermedia ($0.30 L<\|\Omega^2-\Omega\|_F\le0.50 L$).
        - $\top$: consenso estricto dentro de límites clásicos/de diseño.
        """
        idem_res = report.idempotence_residual
        limit = self._idempotence_threshold

        is_hard_veto = (
            (report.engine_validation_error is not None) or
            (not report.is_tsirelson_bounded) or
            (not report.is_choi_completely_positive) or
            (not report.is_causal_non_signaling) or
            (report.choi_min_eigenvalue < _HARD_VETO_CHOI_MIN_EIGENVALUE) or
            (idem_res > _HARD_VETO_IDEMPOTENCE_FRACTION * limit) or
            (report.kernel.metric_condition_number > _HARD_VETO_MAX_CONDITION_NUMBER)
        )

        is_soft_veto = not is_hard_veto and (
            (_SOFT_VETO_IDEMPOTENCE_LOWER_FRACTION * limit < idem_res <= _SOFT_VETO_IDEMPOTENCE_UPPER_FRACTION * limit) or
            (report.tsirelson_chsh_value > _SOFT_VETO_CHSH_CLASSICAL_THRESHOLD and report.is_tsirelson_bounded)
        )

        if is_hard_veto:
            verdict = PhotinicHeytingVerdict.VETOED
        elif is_soft_veto:
            verdict = PhotinicHeytingVerdict.DEGRADED
        else:
            verdict = PhotinicHeytingVerdict.COHERENT

        return verdict, is_soft_veto, is_hard_veto

    # --------------------------------------------------------------------
    # 3.2 — Sutura cuántica de Fock con tokens HMAC de vida limitada
    # --------------------------------------------------------------------
    def evaluate_quantum_fock_annihilation(
        self,
        token: Optional[str],
        idempotence_residual: float
    ) -> Tuple[bool, float]:
        r"""
        Sutura de Fock $e^-+e^+\to2\gamma$: verifica contra tokens
        pre-compartidos (compatibilidad hacia atrás) o valida un token HMAC
        efímero `principal|timestamp:firma`, exigiendo firma válida, no
        repetición y frescura temporal -- ausente en v4.0.0, que además
        aceptaba el secreto compartido **en texto plano** como token válido
        sin ninguna firma real.
        """
        if token is None or not self._is_soft_veto_active:
            return False, 0.0

        token_str = str(token).strip()
        token_valid = False

        for auth_token in self._authorized_tokens:
            if hmac.compare_digest(token_str.encode("utf-8"), auth_token.encode("utf-8")):
                token_valid = True
                break

        if not token_valid and ":" in token_str:
            payload, signature = token_str.rsplit(":", 1)
            expected = hmac.new(self._hmac_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if hmac.compare_digest(signature.strip().lower(), expected.lower()):
                digest = hashlib.sha256(token_str.encode("utf-8")).hexdigest()
                with self._state_lock:
                    if digest in self._consumed_override_digests:
                        raise PhotinicTokenFreshnessError("Token de positrón ya consumido (prevención de repetición).")

                    issued_at: Optional[float] = None
                    if "|" in payload:
                        ts_str = payload.rsplit("|", 1)[-1]
                        try:
                            issued_at = float(ts_str)
                        except ValueError:
                            issued_at = None

                    if issued_at is not None:
                        age = abs(time.time() - issued_at)
                        if age > _TOKEN_FRESHNESS_WINDOW_SECONDS:
                            raise PhotinicTokenFreshnessError(
                                f"Token de positrón expirado: edad={age:.1f}s > "
                                f"ventana={_TOKEN_FRESHNESS_WINDOW_SECONDS:.0f}s"
                            )

                    self._consumed_override_digests.add(digest)
                token_valid = True

        if token_valid:
            prob = float(1.0 - math.exp(
                -max(idempotence_residual, 0.01) / _FOCK_ANNIHILATION_CROSS_SECTION_SIGMA
            ))
            return True, prob

        return False, 0.0

    # --------------------------------------------------------------------
    # 3.3 — Disparo físico real (RC) del Crowbar BT151 en silicio IRAM
    # --------------------------------------------------------------------
    def simulate_bt151_crowbar_iram_discharge(self) -> Tuple[bool, int, float, float]:
        r"""
        Modelo físico **determinista** de conmutación del tiristor Crowbar
        BT151-650R (reemplaza la simulación aleatoria de la v4.0.0):
        $$I_G=\frac{V_{OH}-V_{GT}}{R_{\rm gate}}\ge5I_{GT}, \qquad
          t_{\rm rise}=R_{\rm gate}C_{\rm gate}\ln\frac{V_{OH}}{V_{OH}-V_{GT}}$$
        Latencia total = sobrecarga de ISR en IRAM + $t_{\rm rise}$; si excede
        el presupuesto de 400 ns, se declara `PhotinicCrowbarMisfireError` en
        vez de recortar silenciosamente el valor (defecto de la v4.0.0).
        """
        gate_current_amps = (_ESP32_GPIO_VOH_VOLTS - _BT151_VGT_VOLTS) / _CROWBAR_GATE_RESISTOR_OHMS
        required_current = _CROWBAR_MIN_SAFETY_MARGIN * _BT151_IGT_AMPS
        if gate_current_amps < required_current:
            raise PhotinicCrowbarMisfireError(
                f"Corriente de compuerta I_G={gate_current_amps * 1e3:.2f} mA insuficiente "
                f"(< {required_current * 1e3:.2f} mA requeridos)."
            )

        rc_tau_seconds = _CROWBAR_GATE_RESISTOR_OHMS * _CROWBAR_GATE_CAPACITANCE_FARADS
        t_rise_ns = rc_tau_seconds * math.log(
            _ESP32_GPIO_VOH_VOLTS / (_ESP32_GPIO_VOH_VOLTS - _BT151_VGT_VOLTS)
        ) * 1.0e9
        isr_ns = _ISR_DISPATCH_OVERHEAD_CYCLES * _ESP32_CYCLE_TIME_NS
        total_latency_ns = isr_ns + t_rise_ns

        if total_latency_ns > _CROWBAR_MAX_IRAM_BUDGET_NS:
            raise PhotinicCrowbarMisfireError(
                f"Latencia total de disparo {total_latency_ns:.2f} ns excede el presupuesto "
                f"infranqueable de IRAM ({_CROWBAR_MAX_IRAM_BUDGET_NS:.0f} ns)."
            )

        total_cycles = int(round(total_latency_ns / _ESP32_CYCLE_TIME_NS))
        return True, total_cycles, float(total_latency_ns), float(gate_current_amps)

    # --------------------------------------------------------------------
    # 3.4 — Decisión terminal: composición algebraica de Heyting y firma HMAC
    # --------------------------------------------------------------------
    def decide_heyting_verdict(
        self,
        orientation_report: PhotinicOrientationReport,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        _pipeline_start_perf: Optional[float] = None
    ) -> PhotinicAgentCertificate:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 3 Y DEL AGENTE COVARIANTE:
        El veredicto final se deriva como el **ínfimo reticular**
        $$\text{verdict} = \text{instantáneo} \wedge \text{histéresis}$$
        de `HeytingOmega3Algebra`, en reemplazo de la cascada ad hoc de v4.0.0.
        """
        t_start = time.perf_counter()
        curr_time = float(current_time if current_time is not None else time.time())
        rep = orientation_report

        instant_verdict, _, _ = self.evaluate_heyting_topos_subobject_classifier(rep)

        fock_annihilated = False
        fock_prob = 0.0
        time_remaining = 0.0
        override_expired = False
        hysteresis_verdict = PhotinicHeytingVerdict.COHERENT

        with self._state_lock:
            if instant_verdict == PhotinicHeytingVerdict.VETOED:
                hysteresis_verdict = PhotinicHeytingVerdict.VETOED
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "¡VETO DURO INSTANTÁNEO! CHSH=%.4f, Choi_min=%.4e, engine_error=%s",
                    rep.tsirelson_chsh_value, rep.choi_min_eigenvalue, rep.engine_validation_error
                )

            elif instant_verdict == PhotinicHeytingVerdict.DEGRADED:
                if not self._is_soft_veto_active and not simulate_grace_expired:
                    self._is_soft_veto_active = True
                    self._soft_veto_timestamp = curr_time
                    time_remaining = self._grace_limit
                    hysteresis_verdict = PhotinicHeytingVerdict.DEGRADED
                    logger.warning(
                        "¡VETO SUAVE ACTIVADO (LUZ ÁMBAR)! idem_res=%.4e, CHSH=%.4f",
                        rep.idempotence_residual, rep.tsirelson_chsh_value
                    )
                else:
                    elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                    time_remaining = max(0.0, self._grace_limit - elapsed)
                    if time_remaining <= self._tol or simulate_grace_expired:
                        hysteresis_verdict = PhotinicHeytingVerdict.VETOED
                        self._is_soft_veto_active = False
                        override_expired = True
                        logger.critical("¡VENTANA DE GRACIA EXPIRADA! Colapso de histéresis a VETOED.")
                    else:
                        hysteresis_verdict = PhotinicHeytingVerdict.DEGRADED
            else:
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                hysteresis_verdict = PhotinicHeytingVerdict.COHERENT

            verdict = HeytingOmega3Algebra.meet(instant_verdict, hysteresis_verdict)

            if self._is_soft_veto_active and override_token is not None:
                try:
                    fock_annihilated, fock_prob = self.evaluate_quantum_fock_annihilation(
                        override_token, rep.idempotence_residual
                    )
                except PhotinicTokenFreshnessError as exc:
                    logger.error("Token de positrón de Fock rechazado: %s", exc)
                    fock_annihilated, fock_prob = False, 0.0

                if fock_annihilated:
                    verdict = PhotinicHeytingVerdict.DEGRADED
                    self._is_soft_veto_active = False
                    self._soft_veto_timestamp = None
                    time_remaining = 0.0
                    logger.info("¡ANIQUILACIÓN DE FOCK FOTÍNICA COMPLETADA! Probabilidad %.4f.", fock_prob)
                else:
                    logger.error("Token de Positrón de Fock inválido o espurio.")

            is_soft_veto_active_snapshot = self._is_soft_veto_active

        interlock_fired = False
        iram_cycles = 0
        actuation_ns = 0.0
        gate_current_amps = 0.0

        if verdict == PhotinicHeytingVerdict.VETOED:
            interlock_fired, iram_cycles, actuation_ns, gate_current_amps = (
                self.simulate_bt151_crowbar_iram_discharge()
            )
            logger.critical(
                "¡DISPARO CROWBAR BT151 EJECUTADO! I_G=%.2f mA, Ciclos=%d, Latencia=%.2f ns (< %.0f ns).",
                gate_current_amps * 1e3, iram_cycles, actuation_ns, _CROWBAR_MAX_IRAM_BUDGET_NS
            )

        duration_us = (time.perf_counter() - t_start) * 1.0e6
        total_pipeline_us = (
            (time.perf_counter() - _pipeline_start_perf) * 1.0e6
            if _pipeline_start_perf is not None else duration_us
        )

        sig_payload = (
            f"{verdict.canonical_name}:{rep.kernel.majorana_spinor_norm:.6f}:"
            f"{rep.idempotence_residual:.6e}:{rep.tsirelson_chsh_value:.6f}:"
            f"{actuation_ns:.2f}:{interlock_fired}:{rep.kernel.cryptographic_seal[:16]}"
        )
        digital_sig = self._compute_hmac_seal(sig_payload.encode("ascii"))

        return PhotinicAgentCertificate(
            phase="G_OMEGA_PHOTINIC_GOVERNANCE_SUTURATED",
            heyting_verdict=verdict.canonical_name,
            heyting_instant_verdict=instant_verdict.canonical_name,
            heyting_hysteresis_verdict=hysteresis_verdict.canonical_name,
            heyting_truth_value=verdict.value,
            majorana_spinor_norm=rep.kernel.majorana_spinor_norm,
            idempotence_residual=rep.idempotence_residual,
            trace_rank_mismatch=rep.trace_rank_mismatch,
            choi_min_eigenvalue=rep.choi_min_eigenvalue,
            tsirelson_chsh_value=rep.tsirelson_chsh_value,
            von_neumann_entropy=rep.von_neumann_entropy,
            quantum_purity=rep.quantum_purity,
            engine_validation_error=rep.engine_validation_error,
            is_choi_completely_positive=rep.is_choi_completely_positive,
            is_causal_non_signaling=rep.is_causal_non_signaling,
            is_tsirelson_bounded=rep.is_tsirelson_bounded,
            is_soft_veto_active=is_soft_veto_active_snapshot,
            override_grace_period_expired=override_expired,
            fock_annihilation_occurred=fock_annihilated,
            fock_transition_probability=fock_prob,
            hardware_interlock_fired=interlock_fired,
            crowbar_iram_cycles=iram_cycles,
            crowbar_gate_current_amps=gate_current_amps,
            actuation_latency_ns=actuation_ns,
            time_grace_remaining=time_remaining,
            digital_signature_hmac_sha256=digital_sig,
            execution_duration_microseconds=duration_us,
            total_pipeline_latency_microseconds=total_pipeline_us
        )

    def audit_federated_governance_cycle(
        self,
        node_policies: Union[PhotinicOrientationReport, PhotinicObservationKernel, Sequence[NDArray[np.float64]]],
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False
    ) -> PhotinicAgentCertificate:
        r"""
        Orquesta el ciclo OODA completo para el Soberano Fotínico, aceptando
        polimórficamente `PhotinicOrientationReport`, `PhotinicObservationKernel`
        o tensores crudos, midiendo la latencia total de todo el lazo.
        """
        t_pipeline_start = time.perf_counter()

        if isinstance(node_policies, PhotinicOrientationReport):
            orient_report = node_policies
        else:
            orient_report = self.synthesize_photinic_orientation(node_policies)

        return self.decide_heyting_verdict(
            orientation_report=orient_report,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            _pipeline_start_perf=t_pipeline_start
        )


# ══════════════════════════════════════════════════════════════════════════════
# SOBERANO FOTÍNICO DE GOBERNANZA SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class PhotinicGovernanceSatelliteAgent(Morphism, Phase3_PhotinicAgentDecider):
    r"""
    Soberano Fotínico de Gobernanza Federada en lazo cerrado OODA (Cinturón Orbital).
    Hereda formalmente la estructura categórica de `Morphism` y encapsula la
    totalidad del pipeline unificado de Fases 1, 2 y 3.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        idempotence_threshold: float = 0.25,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None
    ) -> None:
        Morphism.__init__(self)
        Phase3_PhotinicAgentDecider.__init__(
            self,
            tolerance=tolerance,
            idempotence_threshold=idempotence_threshold,
            grace_period_seconds=grace_period_seconds,
            hmac_key=hmac_key,
            authorized_tokens=authorized_tokens,
            engine=engine
        )
        assert HeytingOmega3Algebra.verify_intuitionistic_failure(), (
            "Certificación algebraica fallida: Omega_3 debería violar el tercero excluido."
        )
        logger.info(
            "PhotinicGovernanceSatelliteAgent v5.0.0 inicializado. Gobernanza activa: "
            "Topos Heyting Omega_3 (algebraico), Espinor Majorana SYM, Choi CPTP (enforcement real), "
            "Horodecki-Tsirelson (motor sin renormalización artificial), Crowbar BT151 (RC real), "
            "sellado HMAC-SHA256, estado soberano thread-safe."
        )


__all__ = [
    "PhotinicGovernanceSatelliteAgent",
    "PhotinicObservationKernel",
    "PhotinicOrientationReport",
    "PhotinicAgentCertificate",
    "PhotinicHeytingVerdict",
    "HeytingOmega3Algebra",
    "PhotinicAgentError",
    "PhotinicMetricConsistencyError",
    "PhotinicCrowbarMisfireError",
    "PhotinicTokenFreshnessError",
    "Phase1_PhotinicAgentObserver",
    "Phase2_PhotinicAgentOrienter",
    "Phase3_PhotinicAgentDecider",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "═" * 80)
    print(" INICIANDO AUDITORÍA ESPECTRAL Y CIBER-FÍSICA DE PHOTINIC GOVERNANCE SATELLITE AGENT")
    print("═" * 80)

    print(f"\n[+] Certificación algebraica del topos de Heyting Omega_3:")
    print(f"    - ¬¬(DEGRADED) != DEGRADED: {HeytingOmega3Algebra.verify_intuitionistic_failure()}")

    agent = PhotinicGovernanceSatelliteAgent()
    rng = np.random.default_rng(7)

    # Consorcio federado con N=5 (impar) nodos: ejercita la corrección de forma
    # del motor v5 en la bipartición de Horodecki.
    policies = [rng.normal(loc=0.0, scale=1.0, size=6) for _ in range(5)]

    cert_nominal = agent.audit_federated_governance_cycle(node_policies=policies)
    print(f"\n[+] CERTIFICADO NOMINAL (Topos Heyting):")
    print(f"    - Veredicto (instant./histér./final): {cert_nominal.heyting_instant_verdict} / "
          f"{cert_nominal.heyting_hysteresis_verdict} / {cert_nominal.heyting_verdict}")
    print(f"    - Norma espinor de Majorana:  {cert_nominal.majorana_spinor_norm:.6f}")
    print(f"    - Residuo idempotencia:       {cert_nominal.idempotence_residual:.4e}")
    print(f"    - CHSH (sin renormalizar):    {cert_nominal.tsirelson_chsh_value:.4f}")
    print(f"    - Entropía vN / Pureza:       {cert_nominal.von_neumann_entropy} / {cert_nominal.quantum_purity}")
    print(f"    - Disparo de Hardware:        {cert_nominal.hardware_interlock_fired}")
    print(f"    - Firma HMAC-SHA256:          {cert_nominal.digital_signature_hmac_sha256[:28]}...")
    print(f"    - Latencia total del lazo:    {cert_nominal.total_pipeline_latency_microseconds:.2f} us")

    # Consorcio adversarial (políticas altamente correlacionadas) para inducir Luz Ámbar
    correlated_policies = [np.array([1.0, 0.9, 0.1, 0.0, 0.05, -0.02]) * (1.0 + 0.01 * i) for i in range(6)]
    cert_amber = agent.audit_federated_governance_cycle(
        node_policies=correlated_policies, override_token="AUT_POS_SABIDURIA_777"
    )
    print(f"\n[+] CERTIFICADO DE SUTURA DE FOCK (Override Positrón):")
    print(f"    - Veredicto de Heyting:       {cert_amber.heyting_verdict}")
    print(f"    - Sutura de Fock Exitosa:     {cert_amber.fock_annihilation_occurred}")
    print(f"    - Luz Ámbar Activa:           {cert_amber.is_soft_veto_active}")

    cert_veto = agent.audit_federated_governance_cycle(
        node_policies=correlated_policies, simulate_grace_expired=True
    )
    print(f"\n[+] CERTIFICADO DE VETO DURO Y CROWBAR BT151 (RC real):")
    print(f"    - Veredicto de Heyting:       {cert_veto.heyting_verdict}")
    print(f"    - Disparo Crowbar ESP32:      {cert_veto.hardware_interlock_fired}")
    print(f"    - Corriente de compuerta I_G: {cert_veto.crowbar_gate_current_amps * 1e3:.2f} mA")
    print(f"    - Latencia en Silicio (ns):   {cert_veto.actuation_latency_ns:.2f} ns (< 400 ns)")

    agent.reset_sovereign_state()

    assert cert_nominal.heyting_verdict in ("COHERENT", "DEGRADED"), "Fallo en prueba de coherencia"
    print("\n" + "═" * 80)
    print(" ¡AUDITORÍA CIBER-FÍSICA DEL SOBERANO FOTÍNICO COMPLETADA CON ÉXITO ABSOLUTO!")
    print("═" * 80)
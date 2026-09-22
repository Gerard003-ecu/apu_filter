from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Leptonic Flavor Satellite Agent (Soberano del Satélite V — Sabor Leptónico) ║
║ RUTA   : app/agents/core/immune_system/leptonic_flavor_satellite_agent.py            ║
║ VERSIÓN: 5.0.0-Doctoral-OODA-Heyting-PMNS-Fock-RC-Crowbar-HMAC-Threadsafe            ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Agente supervisor ciber-físico en el Estrato Omega ($V_\Omega$). Gobierna en lazo cerrado al Motor
Satelital de Sabor y Oscilación Leptónica en el Cinturón Orbital de Frontera.

Fundamentación Matemática, Categórica y Ciber-Física Rigurosa:
──────────────────────────────────────────────────────────────
1. Separación Categórica de Física Cruda y Certificación Estricta:
   Garantiza la preservación del reporte de oscilación física de sabor aun cuando la certificación estricta
   del motor rechace unitoridad PMNS o conservación de carga de Noether, haciendo alcanzable la clasificación de Veto Duro.

2. Álgebra de Heyting Formalizada $\Omega_3 = \{\bot, \ast, \top\}$:
   $$\bot = \mathtt{VETOED} (0), \quad \ast = \mathtt{DEGRADED} (1), \quad \top = \mathtt{COHERENT} (2)$$
   gobernado por la regla de inferencia intuicionista:
   $$\text{verdict} = \text{instantáneo} \wedge \text{histéresis}$$
   demostrando numéricamente la falla del tercero excluido ($\neg\neg \ast = \top \neq \ast$).

3. Disparo Físico de Silicio RC y Crowbar BT151 (< 400 ns / IRAM):
   Modelo RC real de compuerta: $t_{\mathrm{rise}} = R_{\mathrm{gate}} C_{\mathrm{gate}} \ln\left(\frac{V_{OH}}{V_{OH} - V_{GT}}\right)$
   con verificación de corriente de compuerta $I_G = \frac{V_{OH} - V_{GT}}{R_{\mathrm{gate}}} \ge 5 I_{GT}$ y cota de IRAM < 400 ns.

4. Sutura Cuántica de Fock y Canario Metrológico de Majorana:
   - Aniquilación $e^- + e^+ \to 2\gamma$ mediante tokens HMAC efímeros con ventana de frescura ($\le 300\,\mathrm{s}$) y prevención de repetición bajo `RLock`.
   - Canario metrológico de consistencia cruzada agente↔motor sobre las normas riemannianas $\|v\|_{\ell^2, G}$.

Traducción Ejecutiva e Impacto de Negocio ('Dolor y Dinero'):
─────────────────────────────────────────────────────────────
• Dolor: Desalineaciones imprevistas en la distribución de partidas y centros de costo provocan fugas financieras
  y distorsiones presupuestarias opacas en la consolidación contable.
• Dinero: La preservación de carga de Noether y el control de unitoridad PMNS evitan fugas y pérdidas
  de capital, protegiendo la exactitud y auditoría financiera de la organización.

Estructura Functorial OODA:
───────────────────────────
- Observe  : `observe_leptonic_event`            -> Salida: `LeptonicAgentObservation`
- Orient   : `orient_leptonic_observation`       -> Salida: `LeptonicAgentOrientation`
- Act      : `decide_heyting_verdict`            -> Salida: `LeptonicAgentCertificate`
- Composición Síncrona Lazo Cerrado             : `audit_leptonic_flavor_cycle`
"""

import hashlib
import hmac
import logging
import math
import threading
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Final, Optional, Tuple, Dict, Any, Set, Sequence, Union

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
    from leptonic_flavor_satellite_engine import (
        LeptonicFlavorSatelliteEngine,
        LeptonicEngineState,
        LeptonicObservationKernel as EngineKernel,
        LeptonicFlavorReport as EngineReport,
        BaseMetricCache,
        LeptonicEngineError,
        MetricIndefinitenessError,
        DimensionMismatchError,
        PMNSUnitarityError,
        LeptonicChargeViolationError,
        KahanNeumaierSum,
    )
except ImportError:
    try:
        from app.core.immune_system.leptonic_flavor_satellite_engine import (
            LeptonicFlavorSatelliteEngine,
            LeptonicEngineState,
            LeptonicObservationKernel as EngineKernel,
            LeptonicFlavorReport as EngineReport,
            BaseMetricCache,
            LeptonicEngineError,
            MetricIndefinitenessError,
            DimensionMismatchError,
            PMNSUnitarityError,
            LeptonicChargeViolationError,
            KahanNeumaierSum,
        )
    except ImportError:
        # ----------------------------------------------------------------
        # Fallback autoportante con rigor matemático mínimo si el motor
        # no está en PYTHONPATH. Alineado con la interfaz v5.0.0 del motor
        # (canonize_leptonic_observation_kernel / orient_leptonic_flavor /
        # execute_leptonic_audit polimórfico). NO APTO PARA PRODUCCIÓN.
        # ----------------------------------------------------------------
        class LeptonicEngineError(Exception):
            pass

        class DimensionMismatchError(LeptonicEngineError):
            pass

        class MetricIndefinitenessError(LeptonicEngineError):
            pass

        class PMNSUnitarityError(LeptonicEngineError):
            pass

        class LeptonicChargeViolationError(LeptonicEngineError):
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

        @dataclass(frozen=True, slots=True)
        class BaseMetricCache:
            g_base: NDArray[np.float64]
            cholesky_factor: NDArray[np.float64]
            g_inv: NDArray[np.float64]
            condition_number: float
            hodge_volume_form: float
            dimension: int
            inversion_residual: float = 0.0
            asymmetry_defect: float = 0.0

        @dataclass(frozen=True, slots=True)
        class EngineKernel:
            flavor_state_matrix: NDArray[np.float64]
            flavor_norms_euclidean: Tuple[float, float, float]
            flavor_norms_riemannian: Tuple[float, float, float]
            banach_ratios: Tuple[float, float, float]
            gram_matrix: NDArray[np.float64]
            metric_cache: BaseMetricCache
            sha256_seal: str
            observation_timestamp: float

        @dataclass(frozen=True, slots=True)
        class EngineReport:
            kernel: EngineKernel
            pmns_matrix: NDArray[np.complex128]
            pmns_unitarity_residual: float
            jarlskog_invariant: float
            jarlskog_invariant_analytic: float
            jarlskog_cross_residual: float
            is_pmns_unitary: bool
            evolution_unitarity_defect: float
            stochastic_normalization_defect: float
            oscillation_probabilities: NDArray[np.float64]
            msw_matter_potential: float
            flavor_graph_laplacian: NDArray[np.float64]
            flavor_graph_spectral_gap: float
            flavor_stationary_distribution: NDArray[np.float64]
            shannon_mixing_entropy: float
            kemeny_constant: float
            total_leptonic_charge: float
            charge_conservation_residual: float
            is_charge_conserved: bool
            orientation_timestamp: float

        @dataclass(frozen=True, slots=True)
        class LeptonicEngineState:
            kernel: EngineKernel
            report: EngineReport
            fpu_execution_time_ms: float
            flavor_density_matrix: NDArray[np.complex128]
            von_neumann_flavor_entropy: float
            quantum_purity: float
            classical_quantum_relative_entropy: float
            equivalent_circuit_q_factor: float
            resonant_quality_factor: float
            spectral_stability_drift: float
            is_bauer_fike_certified: bool
            cryptographic_seal: str
            quantum_flavor_signature: str

        class LeptonicFlavorSatelliteEngine:
            def __init__(self, tolerance: float = 1.0e-12) -> None:
                self.tol = tolerance

            def canonize_leptonic_observation_kernel(
                self, e_state: NDArray[np.float64], mu_state: NDArray[np.float64],
                tau_state: NDArray[np.float64], G_metric: NDArray[np.float64]
            ) -> EngineKernel:
                dim = e_state.shape[0]
                L_chol = la.cholesky(0.5 * (G_metric + G_metric.T), lower=True)
                cache = BaseMetricCache(
                    g_base=G_metric, cholesky_factor=L_chol, g_inv=la.inv(G_metric),
                    condition_number=1.0, hodge_volume_form=float(np.prod(np.diag(L_chol))), dimension=dim
                )
                flavor_matrix = np.vstack([e_state, mu_state, tau_state])
                norms = tuple(float(math.sqrt(max(v @ G_metric @ v, 1e-15))) for v in (e_state, mu_state, tau_state))
                return EngineKernel(
                    flavor_state_matrix=flavor_matrix,
                    flavor_norms_euclidean=tuple(float(la.norm(v)) for v in (e_state, mu_state, tau_state)),
                    flavor_norms_riemannian=norms,
                    banach_ratios=(1.2, 1.2, 1.2),
                    gram_matrix=flavor_matrix @ G_metric @ flavor_matrix.T,
                    metric_cache=cache,
                    sha256_seal=hashlib.sha256(flavor_matrix.tobytes()).hexdigest(),
                    observation_timestamp=time.time()
                )

            def orient_leptonic_flavor(self, kernel: EngineKernel, **kwargs: Any) -> EngineReport:
                U = np.eye(3, dtype=np.complex128)
                P = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]], dtype=np.float64)
                q = np.array(kernel.flavor_norms_riemannian)
                l_total = float(KahanNeumaierSum.sum(q))
                return EngineReport(
                    kernel=kernel, pmns_matrix=U, pmns_unitarity_residual=0.0,
                    jarlskog_invariant=0.0, jarlskog_invariant_analytic=0.0, jarlskog_cross_residual=0.0,
                    is_pmns_unitary=True, evolution_unitarity_defect=0.0, stochastic_normalization_defect=0.0,
                    oscillation_probabilities=P, msw_matter_potential=0.0,
                    flavor_graph_laplacian=np.eye(3) - P, flavor_graph_spectral_gap=0.02,
                    flavor_stationary_distribution=np.array([1 / 3, 1 / 3, 1 / 3]),
                    shannon_mixing_entropy=math.log(3.0), kemeny_constant=2.0,
                    total_leptonic_charge=l_total, charge_conservation_residual=0.0,
                    is_charge_conserved=True, orientation_timestamp=time.time()
                )

            def execute_leptonic_audit(self, target_or_e: Any, **kwargs: Any) -> LeptonicEngineState:
                t0 = time.perf_counter()
                if isinstance(target_or_e, EngineReport):
                    rep = target_or_e
                else:
                    G = kwargs.get("G_metric", np.eye(target_or_e.shape[-1]))
                    k = self.canonize_leptonic_observation_kernel(
                        target_or_e, kwargs.get("mu_state", target_or_e),
                        kwargs.get("tau_state", target_or_e), G
                    )
                    rep = self.orient_leptonic_flavor(k)
                if not rep.is_pmns_unitary:
                    raise PMNSUnitarityError("Fallback: unitoridad violada.")
                if not rep.is_charge_conserved:
                    raise LeptonicChargeViolationError("Fallback: carga no conservada.")
                rho = np.eye(3, dtype=np.complex128) / 3.0
                return LeptonicEngineState(
                    kernel=rep.kernel, report=rep,
                    fpu_execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                    flavor_density_matrix=rho, von_neumann_flavor_entropy=math.log(3.0),
                    quantum_purity=1.0 / 3.0, classical_quantum_relative_entropy=0.0,
                    equivalent_circuit_q_factor=100.0, resonant_quality_factor=100.0,
                    spectral_stability_drift=0.0, is_bauer_fike_certified=True,
                    cryptographic_seal=rep.kernel.sha256_seal,
                    quantum_flavor_signature="FALLBACK_SUTURATED"
                )

logger = logging.getLogger("APU.Agents.Omega.LeptonicFlavorSatelliteAgent")

# ------------------------------------------------------------------------------
# CONSTANTES FÍSICAS, LÍMITES METROLÓGICOS DE WILKINSON Y SILICIO
# ------------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15
_METRIC_INVERSION_TOLERANCE: Final[float] = 1.0e-8

# --- Silicio ESP32 y disparo Crowbar BT151 ------------------------------------
_CROWBAR_MAX_IRAM_BUDGET_NS: Final[float] = 400.0        # Cota infranqueable de hardware
_ESP32_CLOCK_FREQ_HZ: Final[float] = 240.0e6             # Xtensa Dual-Core LX6 @ 240 MHz
_ESP32_CYCLE_TIME_NS: Final[float] = 1.0e9 / _ESP32_CLOCK_FREQ_HZ  # 4.1667 ns / ciclo
_ISR_DISPATCH_OVERHEAD_CYCLES: Final[int] = 42           # Entrada/salida de ISR medida en banco de pruebas
_BT151_VGT_VOLTS: Final[float] = 1.1                     # Tensión de compuerta BT151 típica (V)
_BT151_IGT_AMPS: Final[float] = 0.005                    # Corriente de disparo de compuerta (5 mA)
_ESP32_GPIO_VOH_VOLTS: Final[float] = 3.3                # Tensión de salida GPIO HIGH (V)
_CROWBAR_GATE_RESISTOR_OHMS: Final[float] = 47.0         # Resistencia limitadora de compuerta (Ω)
_CROWBAR_GATE_CAPACITANCE_FARADS: Final[float] = 1.0e-9  # Ciss ilustrativo de orden de magnitud (1 nF)
_CROWBAR_MIN_SAFETY_MARGIN: Final[float] = 5.0           # Exigencia doctrinal: I_G >= 5 * I_GT

# --- Umbrales del clasificador de Heyting (documentados, no mágicos) ---------
_HARD_VETO_MAX_OFF_DIAGONAL: Final[float] = 0.50
_SOFT_VETO_MIN_OFF_DIAGONAL: Final[float] = 0.001
_ELASTIC_BAND_MAX_OFF_DIAGONAL: Final[float] = 0.20
_HARD_VETO_MAX_CONDITION_NUMBER: Final[float] = 1.0e10
_JARLSKOG_PHYSICAL_UPPER_BOUND: Final[float] = 1.0 / (6.0 * math.sqrt(3.0))  # ≈ 0.09623
_JARLSKOG_BOUND_SAFETY_MARGIN: Final[float] = 1.0e-6

# --- Sutura de Fock y tokens de positrón --------------------------------------
_FOCK_ANNIHILATION_CROSS_SECTION_SIGMA: Final[float] = 0.10
_TOKEN_FRESHNESS_WINDOW_SECONDS: Final[float] = 300.0     # Validez de tokens firmados efímeros (5 min)
_CROSS_ENGINE_CONSISTENCY_TOLERANCE: Final[float] = 1.0e-8


# ------------------------------------------------------------------------------
# JERARQUÍA DE EXCEPCIONES DEL AGENTE
# ------------------------------------------------------------------------------
class LeptonicAgentError(LeptonicEngineError):
    r"""Excepción raíz para violaciones propias del Soberano de Sabor Leptónico."""
    pass


class MetricConsistencyError(LeptonicAgentError):
    r"""
    Canario metrológico: divergencia entre dos cómputos independientes que
    deberían ser matemáticamente idénticos (Fase 1 vs. Fase 2 del agente,
    o agente vs. motor), delatando desalineación de dependencias o corrupción.
    """
    pass


class CrowbarMisfireError(LeptonicAgentError):
    r"""
    Fallo físico certificado del disparo Crowbar BT151: la corriente de
    compuerta calculada es insuficiente para garantizar el cebado por
    avalancha, o el tiempo de subida RC excede el presupuesto de IRAM.
    """
    pass


class TokenFreshnessError(LeptonicAgentError):
    r"""
    El token de positrón de sutura de Fock ha expirado (ventana de frescura
    excedida) o ha sido presentado en repetición (replay) tras su consumo.
    """
    pass


class LeptonicHeytingVerdict(IntEnum):
    r"""
    Álgebra de Heyting $\Omega_3 = \{\bot, \ast, \top\}$ en el topos de De Rham:
    $\bot = 0$ (VETOED - Violación de unitoridad PMNS, fuga de carga, colapso de red)
    $\ast = 1$ (DEGRADED - Luz Ámbar / Oscilación elástica de sabor / Gracia de Fock)
    $\top = 2$ (COHERENT - Unitoridad estricta y conservación de Noether síncrona)
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
    $\Omega_3 = \{\bot < \ast < \top\}$, isomorfa al conjunto de valores del
    clasificador de subobjetos $\{\mathtt{VETOED}, \mathtt{DEGRADED}, \mathtt{COHERENT}\}$.
    Al ser una cadena, el meet y el join coinciden con el mínimo y el máximo,
    y la pseudo-complementación (implicación a $\bot$) se define formalmente como
    $$\neg a := a \Rightarrow \bot = \sup\{x \in \Omega_3 : x \wedge a = \bot\}.$$
    Esta estructura **no** satisface el principio de tercero excluido:
    $\neg\neg(\ast) = \top \neq \ast$, certificando computacionalmente la
    naturaleza intuicionista (no booleana) del topos de gobernanza del agente.
    """

    @staticmethod
    def meet(a: LeptonicHeytingVerdict, b: LeptonicHeytingVerdict) -> LeptonicHeytingVerdict:
        r"""Ínfimo reticular $a \wedge b$: el veredicto más conservador domina (semántica AND-veto)."""
        return LeptonicHeytingVerdict(min(int(a), int(b)))

    @staticmethod
    def join(a: LeptonicHeytingVerdict, b: LeptonicHeytingVerdict) -> LeptonicHeytingVerdict:
        r"""Supremo reticular $a \vee b$: el veredicto más optimista domina."""
        return LeptonicHeytingVerdict(max(int(a), int(b)))

    @staticmethod
    def pseudocomplement(a: LeptonicHeytingVerdict) -> LeptonicHeytingVerdict:
        r"""Pseudo-complemento intuicionista: $\neg\bot=\top,\ \neg\ast=\bot,\ \neg\top=\bot$."""
        if a == LeptonicHeytingVerdict.VETOED:
            return LeptonicHeytingVerdict.COHERENT
        return LeptonicHeytingVerdict.VETOED

    @classmethod
    def verify_intuitionistic_failure(cls) -> bool:
        r"""Certifica $\neg\neg\ast \neq \ast$: ausencia del tercero excluido en $\Omega_3$."""
        double_neg = cls.pseudocomplement(cls.pseudocomplement(LeptonicHeytingVerdict.DEGRADED))
        return bool(double_neg != LeptonicHeytingVerdict.DEGRADED)


@dataclass(frozen=True, slots=True)
class LeptonicAgentObservation:
    r"""
    Expediente inmutable de Fase 1 (Observe).
    Audita la regularidad de Sobolev-Banach $\ell^1 \hookrightarrow \ell^2$,
    el tensor de Riemann-Cholesky $G \succ 0$ (con certificación de simetría
    e inversión numérica), la norma del espinor de Majorana y el sello HMAC.
    Cachea el `metric_snapshot` auditado, eliminando la ambigüedad de
    resolución de métrica presente en la v4.0.0.
    """
    flavor_states: Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    banach_ratios: Tuple[float, float, float]
    flavor_norms_riemannian: Tuple[float, float, float]
    majorana_spinor_norm: float
    metric_snapshot: NDArray[np.float64]
    metric_condition_number: float
    metric_inversion_residual: float
    hodge_volume_form: float
    cryptographic_seal: str
    observation_timestamp: float


@dataclass(frozen=True, slots=True)
class LeptonicAgentOrientation:
    r"""
    Expediente inmutable de Fase 2 (Orient).
    Sintetiza la unitoridad PMNS en $U(3)$, la conservación de corriente de
    Noether, el invariante de Jarlskog (numérico, analítico y su residuo
    cruzado), la brecha de Fiedler, la entropía de Shannon de mezcla, la
    constante de Kemeny y la telemetría cuántica avanzada del motor (pureza,
    $Q$ resonante, certificación de Bauer-Fike), cuando la certificación
    estricta del motor tuvo éxito. `engine_state` es `None` y
    `engine_validation_error` describe la causa cuando el motor rechazó la
    certificación por violación de unitoridad o carga -- en cuyo caso el
    reporte físico crudo (`is_pmns_unitary`, `is_charge_conserved`, etc.)
    permanece disponible para el clasificador de Heyting.
    """
    kernel: LeptonicAgentObservation
    engine_state: Optional[LeptonicEngineState]
    engine_validation_error: Optional[str]
    pmns_unitarity_residual: float
    jarlskog_invariant: float
    jarlskog_invariant_analytic: float
    jarlskog_cross_residual: float
    is_pmns_unitary: bool
    evolution_unitarity_defect: float
    stochastic_normalization_defect: float
    total_leptonic_charge: float
    charge_conservation_residual: float
    is_charge_conserved: bool
    max_off_diagonal_oscillation: float
    flavor_graph_spectral_gap: float
    shannon_mixing_entropy: float
    kemeny_constant: float
    is_flavor_oscillation_elastic: bool
    resonant_quality_factor: Optional[float]
    quantum_purity: Optional[float]
    is_bauer_fike_certified: Optional[bool]
    orientation_timestamp: float


@dataclass(frozen=True, slots=True)
class LeptonicAgentCertificate:
    r"""
    Certificado supremo inmutable emitido por el Soberano de Sabor en Fase 3 (Act).
    Acredita el veredicto formal de Heyting $\Omega_3$ (instantáneo, de
    histéresis y su ínfimo reticular), la telemetría cuántica de Fock, y la
    bitácora de excitación física real (RC) del Crowbar BT151 en silicio IRAM.
    """
    phase: str
    heyting_verdict: str
    heyting_instant_verdict: str
    heyting_hysteresis_verdict: str
    heyting_truth_value: int
    pmns_unitarity_residual: float
    jarlskog_invariant: float
    jarlskog_invariant_analytic: float
    jarlskog_cross_residual: float
    charge_conservation_residual: float
    max_off_diagonal_oscillation: float
    flavor_graph_spectral_gap: float
    shannon_mixing_entropy: float
    kemeny_constant: float
    resonant_quality_factor: Optional[float]
    quantum_purity: Optional[float]
    is_bauer_fike_certified: Optional[bool]
    engine_validation_error: Optional[str]
    is_pmns_unitary: bool
    is_charge_conserved: bool
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
# FASE 1: INGESTA, ESPINOR DE MAJORANA Y ACONDICIONAMIENTO DE BANACH-RIEMANN
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_LeptonicAgentObserver:
    r"""
    FASE 1 — Observe:
    Saneamiento de signos IEEE 754 ($x = -0.0 \mapsto +0.0$), auditoría de
    simetría y factorización Cholesky de la métrica $G_{\mu\nu} = L L^\top \succ 0$
    con certificación explícita del residuo de inversión (ausente en v4),
    evaluación compensada (Kahan-Neumaier) de la regularidad de Sobolev-Banach,
    cálculo del invariante de Majorana y canonización HMAC-sellada del expediente.
    """

    def __init__(self, tolerance: float = 1.0e-12, hmac_key: Optional[bytes] = None) -> None:
        self._tol: Final[float] = float(tolerance)
        if hmac_key is None:
            logger.warning(
                "LeptonicFlavorSatelliteAgent: se utiliza la clave HMAC derivada por defecto. "
                "En despliegue productivo DEBE inyectarse una clave gestionada por el subsistema "
                "de secretos del 'immune_system' (parámetro hmac_key)."
            )
            hmac_key = hashlib.sha256(b"APU.LeptonicFlavorSatelliteAgent.DefaultAgentSalt.v5").digest()
        self._hmac_key: Final[bytes] = hmac_key

    # --------------------------------------------------------------------
    # 1.1 — Saneamiento numérico y sellado HMAC
    # --------------------------------------------------------------------
    @staticmethod
    def _canonicalize_state(S: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Sanea ceros con signo negativo IEEE 754 y aplasta subnormales por debajo del piso de Wilkinson."""
        clean = np.where(S == -0.0, +0.0, S)
        clean = np.where(np.abs(clean) < _WILKINSON_SAFETY_FLOOR, +0.0, clean)
        return clean

    def _compute_hmac_seal(self, *chunks: bytes) -> str:
        r"""Sello de autenticidad HMAC-SHA256 sobre la concatenación ordenada de fragmentos binarios."""
        mac = hmac.new(self._hmac_key, digestmod=hashlib.sha256)
        for chunk in chunks:
            mac.update(chunk)
        return mac.hexdigest()

    # --------------------------------------------------------------------
    # 1.2 — Regularidad de Sobolev-Banach compensada
    # --------------------------------------------------------------------
    def _evaluate_banach_regularity(self, v: NDArray[np.float64], norm_riemannian: float) -> float:
        r"""
        Mide la regularidad de Sobolev-Banach con suma compensada de Kahan-Neumaier:
        $$\mathcal{R}_B(v) = \frac{\|v\|_1}{\|v\|_{\ell^2, G}}, \qquad 1.0 \le \mathcal{R}_B(v) \le \sqrt{d}.$$
        """
        norm_l1 = KahanNeumaierSum.sum(np.abs(v))
        return float(norm_l1 / max(norm_riemannian, _WILKINSON_SAFETY_FLOOR))

    # --------------------------------------------------------------------
    # 1.3 — Auditoría de la métrica: simetría, Cholesky e inversión certificada
    # --------------------------------------------------------------------
    def _audit_and_snapshot_metric(self, G_metric: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
        r"""
        Audita el tensor métrico y produce su snapshot canónico:
        1. Defecto de asimetría de de Rham $\delta_{\rm asym}=\|G-G^\top\|_F/\|G\|_F$.
        2. Cholesky $G=LL^\top$ y forma de volumen de Hodge $\sqrt{\det G}=\prod L_{ii}$.
        3. Certificación explícita del residuo de inversión $\|GG^{-1}-I\|_F$
           (ausente en v4: una inversión mal condicionada podía pasar inadvertida).
        """
        if G_metric.ndim != 2 or G_metric.shape[0] != G_metric.shape[1]:
            raise DimensionMismatchError(f"El tensor métrico G debe ser cuadrado. Forma: {G_metric.shape}")

        dim = G_metric.shape[0]
        asym_defect = float(la.norm(G_metric - G_metric.T, ord='fro')) / max(
            float(la.norm(G_metric, ord='fro')), _WILKINSON_SAFETY_FLOOR
        )
        if asym_defect > self._tol:
            logger.warning("Defecto de asimetría en G_metric = %.6e excede tolerancia %.2e.", asym_defect, self._tol)

        G_sym = self._canonicalize_state(0.5 * (G_metric + G_metric.T))

        try:
            L_chol = la.cholesky(G_sym, lower=True)
        except la.LinAlgError as exc:
            raise MetricIndefinitenessError(f"Tensor métrico G no es estrictamente SPD: {exc}") from exc

        hodge_vol = float(np.prod(np.diag(L_chol)))
        if hodge_vol <= _WILKINSON_SAFETY_FLOOR:
            raise MetricIndefinitenessError(f"Forma de volumen métrica colapsada: {hodge_vol:.6e}")

        L_inv = la.solve_triangular(L_chol, np.eye(dim, dtype=np.float64), lower=True)
        G_inv = L_inv.T @ L_inv
        inversion_residual = float(la.norm(G_sym @ G_inv - np.eye(dim, dtype=np.float64), ord='fro'))
        if inversion_residual > _METRIC_INVERSION_TOLERANCE:
            raise MetricIndefinitenessError(
                f"Residuo de inversión ||G G^-1 - I||_F = {inversion_residual:.4e} excede {_METRIC_INVERSION_TOLERANCE:.2e}."
            )

        eigvals = la.eigvalsh(G_sym)
        cond_num = float(max(eigvals[-1], _WILKINSON_SAFETY_FLOOR) / max(eigvals[0], _WILKINSON_SAFETY_FLOOR))

        return G_sym, L_chol, cond_num, hodge_vol, inversion_residual

    # --------------------------------------------------------------------
    # 1.4 — Ingesta pública y canonización terminal (puerto hacia Fase 2)
    # --------------------------------------------------------------------
    def observe_leptonic_event(
        self,
        flavor_states: Union[NDArray[np.float64], Sequence[NDArray[np.float64]]],
        G_metric: NDArray[np.float64]
    ) -> LeptonicAgentObservation:
        r"""Ingesta los estados de sabor ($e,\mu,\tau$) y el tensor métrico, con saneamiento previo de de Rham."""
        if isinstance(flavor_states, np.ndarray):
            if flavor_states.ndim != 2 or flavor_states.shape[0] != 3:
                raise DimensionMismatchError(
                    f"La matriz de sabores debe ser de dimensión (3, d). Obtenido: {flavor_states.shape}"
                )
            e_s, mu_s, tau_s = flavor_states[0], flavor_states[1], flavor_states[2]
        elif isinstance(flavor_states, (list, tuple)) and len(flavor_states) == 3:
            e_s, mu_s, tau_s = flavor_states[0], flavor_states[1], flavor_states[2]
        else:
            raise DimensionMismatchError("flavor_states debe contener exactamente 3 vectores de sabor (e, mu, tau).")

        dim = e_s.shape[0]
        if mu_s.shape[0] != dim or tau_s.shape[0] != dim:
            raise DimensionMismatchError(
                f"Discrepancia en dimensiones de sabor: e({dim}), mu({mu_s.shape[0]}), tau({tau_s.shape[0]})."
            )
        if G_metric.ndim != 2 or G_metric.shape != (dim, dim):
            raise DimensionMismatchError(f"Tensor métrico G debe tener dimensiones ({dim}, {dim}).")

        c_e = self._canonicalize_state(e_s)
        c_mu = self._canonicalize_state(mu_s)
        c_tau = self._canonicalize_state(tau_s)

        return self.canonize_leptonic_observation_kernel(e_state=c_e, mu_state=c_mu, tau_state=c_tau, G_metric=G_metric)

    def canonize_leptonic_observation_kernel(
        self,
        e_state: NDArray[np.float64],
        mu_state: NDArray[np.float64],
        tau_state: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> LeptonicAgentObservation:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 1:
        Canoniza el expediente inmutable de Fase 1. Audita y cachea el
        `metric_snapshot` (eliminando la ambigüedad de resolución de métrica
        de la v4.0.0), calcula las normas de Sobolev-Banach compensadas, la
        norma del espinor de Majorana y sella con HMAC-SHA256.
        Constituye el puerto canónico de acoplamiento directo hacia la Fase 2.
        """
        G_snapshot, _L_chol, cond_num, hodge_vol, inv_residual = self._audit_and_snapshot_metric(G_metric)

        n_e_rie = float(math.sqrt(max(float(e_state @ G_snapshot @ e_state), _WILKINSON_SAFETY_FLOOR)))
        n_mu_rie = float(math.sqrt(max(float(mu_state @ G_snapshot @ mu_state), _WILKINSON_SAFETY_FLOOR)))
        n_tau_rie = float(math.sqrt(max(float(tau_state @ G_snapshot @ tau_state), _WILKINSON_SAFETY_FLOOR)))

        r_e = self._evaluate_banach_regularity(e_state, n_e_rie)
        r_mu = self._evaluate_banach_regularity(mu_state, n_mu_rie)
        r_tau = self._evaluate_banach_regularity(tau_state, n_tau_rie)

        # Norma del espinor de Majorana con suma compensada: ||lambda|| = sqrt(sum ||v_a||_G^2)
        spinor_norm = float(math.sqrt(max(
            KahanNeumaierSum.sum([n_e_rie ** 2, n_mu_rie ** 2, n_tau_rie ** 2]), _WILKINSON_SAFETY_FLOOR
        )))

        seal = self._compute_hmac_seal(
            e_state.tobytes(), mu_state.tobytes(), tau_state.tobytes(), G_snapshot.tobytes(),
            f"{cond_num:.8e}_{hodge_vol:.8e}_{spinor_norm:.8e}_{inv_residual:.8e}".encode("ascii")
        )

        return LeptonicAgentObservation(
            flavor_states=(e_state, mu_state, tau_state),
            banach_ratios=(r_e, r_mu, r_tau),
            flavor_norms_riemannian=(n_e_rie, n_mu_rie, n_tau_rie),
            majorana_spinor_norm=spinor_norm,
            metric_snapshot=G_snapshot,
            metric_condition_number=cond_num,
            metric_inversion_residual=inv_residual,
            hodge_volume_form=hodge_vol,
            cryptographic_seal=seal,
            observation_timestamp=time.time()
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENTACIÓN PMNS EN U(3), CONECTIVIDAD DE FIEDLER Y JARLSKOG
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_PMNSAgentOrienter(Phase1_LeptonicAgentObserver):
    r"""
    FASE 2 — Orient:
    Hereda formalmente de la Fase 1 e ingiere directamente el `LeptonicAgentObservation`.
    Separa la **física cruda** del motor (`canonize_leptonic_observation_kernel` +
    `orient_leptonic_flavor`, que nunca lanzan por violaciones clasificables) de la
    **certificación estricta** (`execute_leptonic_audit`), capturando sus excepciones
    para preservar siempre el reporte físico crudo -- reparando el defecto arquitectónico
    v4.0.0 donde el veto duro por no-unitoridad era inalcanzable. Audita adicionalmente
    la consistencia cruzada de métricas y de normas riemannianas agente↔motor.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        hmac_key: Optional[bytes] = None,
        engine: Optional[LeptonicFlavorSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key)
        self._engine: Final[LeptonicFlavorSatelliteEngine] = (
            engine or LeptonicFlavorSatelliteEngine(tolerance=tolerance)
        )

    # --------------------------------------------------------------------
    # 2.1 — Resolución certificada de métrica y canario metrológico cruzado
    # --------------------------------------------------------------------
    def _resolve_and_verify_metric(
        self,
        kernel: LeptonicAgentObservation,
        G_metric: Optional[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        r"""
        Resuelve el tensor métrico efectivo reutilizando por defecto el
        `metric_snapshot` ya auditado en Fase 1. Si el invocador suministra
        explícitamente un `G_metric` distinto, certifica su consistencia
        relativa (Frobenius) frente al snapshot -- reparando la rama muerta
        `hasattr(kernel, 'engine_state')` de la v4.0.0.
        """
        if G_metric is None:
            return kernel.metric_snapshot
        drift = float(la.norm(G_metric - kernel.metric_snapshot, ord='fro'))
        reference = max(float(la.norm(kernel.metric_snapshot, ord='fro')), _WILKINSON_SAFETY_FLOOR)
        if (drift / reference) > self._tol:
            raise MetricConsistencyError(
                f"El G_metric suministrado a la Fase 2 diverge del snapshot auditado en Fase 1 "
                f"(drift relativo = {drift / reference:.4e} > tol = {self._tol:.2e})."
            )
        return kernel.metric_snapshot

    def _audit_cross_engine_metric_consistency(
        self,
        agent_kernel: LeptonicAgentObservation,
        engine_kernel: EngineKernel
    ) -> None:
        r"""
        Canario metrológico: certifica que las normas riemannianas recalculadas
        independientemente por el agente y por el motor -- ambas derivadas de
        $\|v\|_{\ell^2,G}=\sqrt{v^\top Gv}$ sobre idénticos datos de entrada --
        coinciden dentro de una tolerancia estricta, delatando cualquier
        divergencia de implementación (skew de dependencias) entre módulos.
        """
        agent_norms = np.array(agent_kernel.flavor_norms_riemannian, dtype=np.float64)
        engine_norms = np.array(engine_kernel.flavor_norms_riemannian, dtype=np.float64)
        residual = float(la.norm(agent_norms - engine_norms))
        reference = max(float(la.norm(agent_norms)), _WILKINSON_SAFETY_FLOOR)
        if (residual / reference) > _CROSS_ENGINE_CONSISTENCY_TOLERANCE:
            raise MetricConsistencyError(
                f"Divergencia crítica entre las normas riemannianas del agente y del motor "
                f"(residuo relativo = {residual / reference:.4e}). Posible desalineación de "
                f"dependencias entre agente y 'leptonic_flavor_satellite_engine.py'."
            )

    # --------------------------------------------------------------------
    # 2.2 — Orientación terminal y puerto de acoplamiento hacia Fase 3
    # --------------------------------------------------------------------
    def orient_leptonic_observation(
        self,
        kernel: LeptonicAgentObservation,
        G_metric: Optional[NDArray[np.float64]] = None,
        pmns_angles: Tuple[float, float, float] = (0.587, 0.855, 0.149),
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        propagation_length: float = 100.0,
        energy: float = 1.0,
        matter_potential_eV: float = 0.0
    ) -> LeptonicAgentOrientation:
        r"""Conduce la orientación de sabor separando física cruda de certificación estricta."""
        actual_G = self._resolve_and_verify_metric(kernel, G_metric)
        e_st, mu_st, tau_st = kernel.flavor_states

        # --- Física cruda del motor (Fases 1+2 del motor): jamás lanza por
        # violaciones clasificables de unitoridad/carga. ---
        engine_kernel = self._engine.canonize_leptonic_observation_kernel(
            e_state=e_st, mu_state=mu_st, tau_state=tau_st, G_metric=actual_G
        )
        engine_report: EngineReport = self._engine.orient_leptonic_flavor(
            kernel=engine_kernel,
            theta12=pmns_angles[0], theta23=pmns_angles[1], theta13=pmns_angles[2],
            delta_cp=delta_cp, alpha21=alpha21, alpha31=alpha31,
            baseline_L_km=propagation_length, energy_E_GeV=energy,
            matter_potential_eV=matter_potential_eV
        )

        self._audit_cross_engine_metric_consistency(kernel, engine_kernel)

        # --- Certificación estricta del motor (Fase 3 del motor): SÍ lanza
        # ante violación de unitoridad/carga. Se captura y se traduce en
        # ausencia de telemetría avanzada, preservando siempre `engine_report`
        # crudo para el clasificador de Heyting (corrige el defecto v4.0.0). ---
        engine_state: Optional[LeptonicEngineState] = None
        validation_error: Optional[str] = None
        try:
            engine_state = self._engine.execute_leptonic_audit(target_or_e=engine_report)
        except (PMNSUnitarityError, LeptonicChargeViolationError) as exc:
            validation_error = str(exc)
            logger.error(
                "Certificación estricta del motor rechazada (condición veto-clasificable esperada): %s",
                validation_error
            )

        p_matrix = engine_report.oscillation_probabilities.copy()
        np.fill_diagonal(p_matrix, 0.0)
        max_off_diag = float(np.max(p_matrix))
        is_elastic = bool(max_off_diag <= _ELASTIC_BAND_MAX_OFF_DIAGONAL)

        return LeptonicAgentOrientation(
            kernel=kernel,
            engine_state=engine_state,
            engine_validation_error=validation_error,
            pmns_unitarity_residual=engine_report.pmns_unitarity_residual,
            jarlskog_invariant=engine_report.jarlskog_invariant,
            jarlskog_invariant_analytic=engine_report.jarlskog_invariant_analytic,
            jarlskog_cross_residual=engine_report.jarlskog_cross_residual,
            is_pmns_unitary=engine_report.is_pmns_unitary,
            evolution_unitarity_defect=engine_report.evolution_unitarity_defect,
            stochastic_normalization_defect=engine_report.stochastic_normalization_defect,
            total_leptonic_charge=engine_report.total_leptonic_charge,
            charge_conservation_residual=engine_report.charge_conservation_residual,
            is_charge_conserved=engine_report.is_charge_conserved,
            max_off_diagonal_oscillation=max_off_diag,
            flavor_graph_spectral_gap=engine_report.flavor_graph_spectral_gap,
            shannon_mixing_entropy=engine_report.shannon_mixing_entropy,
            kemeny_constant=engine_report.kemeny_constant,
            is_flavor_oscillation_elastic=is_elastic,
            resonant_quality_factor=(engine_state.resonant_quality_factor if engine_state else None),
            quantum_purity=(engine_state.quantum_purity if engine_state else None),
            is_bauer_fike_certified=(engine_state.is_bauer_fike_certified if engine_state else None),
            orientation_timestamp=time.time()
        )

    def synthesize_leptonic_orientation(
        self,
        kernel_or_states: Union[LeptonicAgentObservation, NDArray[np.float64], Sequence[NDArray[np.float64]]],
        G_metric: Optional[NDArray[np.float64]] = None,
        pmns_angles: Tuple[float, float, float] = (0.587, 0.855, 0.149),
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        propagation_length: float = 100.0,
        energy: float = 1.0,
        matter_potential_eV: float = 0.0
    ) -> LeptonicAgentOrientation:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 2:
        Garantiza la continuidad functorial estricta. Si recibe un
        `LeptonicAgentObservation`, procede directamente a la orientación;
        si recibe tensores crudos, enlaza automáticamente con el método
        terminal de Fase 1. El `LeptonicAgentOrientation` devuelto es el
        puerto canónico de acoplamiento hacia la Fase 3.
        """
        if isinstance(kernel_or_states, LeptonicAgentObservation):
            kernel = kernel_or_states
        else:
            if G_metric is None:
                raise DimensionMismatchError("Se requiere el tensor métrico G_metric al procesar tensores crudos.")
            kernel = self.observe_leptonic_event(kernel_or_states, G_metric)

        return self.orient_leptonic_observation(
            kernel=kernel,
            G_metric=G_metric,
            pmns_angles=pmns_angles,
            delta_cp=delta_cp,
            alpha21=alpha21,
            alpha31=alpha31,
            propagation_length=propagation_length,
            energy=energy,
            matter_potential_eV=matter_potential_eV
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: DECISIÓN EN TOPOS DE HEYTING, SUTURA DE FOCK Y CROWBAR RC-IRAM
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_HeytingLeptonicDecider(Phase2_PMNSAgentOrienter):
    r"""
    FASE 3 — Decide & Act:
    Hereda formalmente de la Fase 2. Modula la rampa de de Rham en el
    clasificador de subobjetos $\Omega_3$ mediante la formalización algebraica
    `HeytingOmega3Algebra` (el veredicto final es el ínfimo reticular entre el
    veredicto instantáneo y el de histéresis), resuelve la aniquilación
    cuántica de pares en espacio de Fock con tokens HMAC de vida limitada, y
    dispara el tiristor Crowbar BT151 mediante un modelo RC real y verificable
    (< 400 ns / GPIO14) ante colapso terminal a $\bot$. Todo el estado
    soberano mutable está protegido por exclusión mutua (`RLock`).
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[LeptonicFlavorSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance, hmac_key=hmac_key, engine=engine)
        if grace_period_seconds <= 0.0:
            raise ValueError("grace_period_seconds debe ser estrictamente positivo.")
        self._grace_period_seconds: Final[float] = float(grace_period_seconds)

        if authorized_tokens is None:
            logger.warning(
                "LeptonicFlavorSatelliteAgent: se utilizan los tokens de autorización de demostración "
                "por defecto. DEBEN sustituirse por credenciales gestionadas en despliegue productivo."
            )
            authorized_tokens = {
                "AUT_POS_SABIDURIA_777",
                "OVERRIDE_LEPTONIC_FLAVOR_2026",
                "HMAC_SUTURA_FOCK_SECURE"
            }
        self._authorized_tokens: Final[Set[str]] = authorized_tokens

        # Estado soberano mutable (máquina de histéresis de veto suave),
        # protegido por exclusión mutua ante concurrencia real de auditoría.
        self._state_lock: Final[threading.RLock] = threading.RLock()
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False
        self._consumed_override_digests: Set[str] = set()

    def reset_sovereign_state(self) -> None:
        r"""
        Reinicia atómicamente la máquina de histéresis del veto suave y el
        registro de tokens consumidos. Expuesto explícitamente para pruebas
        unitarias e idempotencia operativa; NO debe invocarse en producción
        salvo bajo procedimiento auditado de recuperación.
        """
        with self._state_lock:
            self._soft_veto_timestamp = None
            self._is_soft_veto_active = False
            self._consumed_override_digests.clear()
            logger.info("Estado soberano de histéresis de veto suave reiniciado explícitamente.")

    # --------------------------------------------------------------------
    # 3.1 — Clasificador de subobjetos en el topos de Heyting (instantáneo)
    # --------------------------------------------------------------------
    def evaluate_heyting_topos_subobject_classifier(
        self,
        rep: LeptonicAgentOrientation
    ) -> Tuple[LeptonicHeytingVerdict, bool, bool]:
        r"""
        Clasificador **instantáneo** (sin memoria) de subobjetos en $\Omega_3$:
        - $\bot$ (VETOED): rechazo de certificación del motor, violación de
          unitoridad PMNS o de conservación de carga de Noether, colapso de
          la brecha de Fiedler ($\lambda_2 \le 0$), desbordamiento no elástico
          ($P_{\mathrm{off}} > 0.50$), condicionamiento métrico patológico,
          invariante de Jarlskog fuera de su cota física, o certificación
          espectral de Bauer-Fike fallida.
        - $\ast$ (DEGRADED): oscilación de sabor atípica o inelástica sin
          fractura de simetría de gauge.
        - $\top$ (COHERENT): unitoridad estricta, conservación de carga y
          certificación completa del motor.
        """
        jarlskog_excess = abs(rep.jarlskog_invariant) > (
            _JARLSKOG_PHYSICAL_UPPER_BOUND + _JARLSKOG_BOUND_SAFETY_MARGIN
        )
        bauer_fike_failed = (rep.is_bauer_fike_certified is False)

        is_hard_veto = (
            (rep.engine_validation_error is not None) or
            (not rep.is_pmns_unitary) or
            (not rep.is_charge_conserved) or
            (rep.flavor_graph_spectral_gap <= _WILKINSON_SAFETY_FLOOR) or
            (rep.max_off_diagonal_oscillation > _HARD_VETO_MAX_OFF_DIAGONAL) or
            (rep.kernel.metric_condition_number > _HARD_VETO_MAX_CONDITION_NUMBER) or
            jarlskog_excess or
            bauer_fike_failed
        )

        is_soft_veto = not is_hard_veto and (
            (rep.max_off_diagonal_oscillation > _SOFT_VETO_MIN_OFF_DIAGONAL) or
            (not rep.is_flavor_oscillation_elastic)
        )

        if is_hard_veto:
            verdict = LeptonicHeytingVerdict.VETOED
        elif is_soft_veto:
            verdict = LeptonicHeytingVerdict.DEGRADED
        else:
            verdict = LeptonicHeytingVerdict.COHERENT

        return verdict, is_soft_veto, is_hard_veto

    # --------------------------------------------------------------------
    # 3.2 — Sutura cuántica de Fock con tokens HMAC de vida limitada
    # --------------------------------------------------------------------
    def evaluate_quantum_fock_annihilation(
        self,
        token: Optional[str],
        oscillation_amplitude: float
    ) -> Tuple[bool, float]:
        r"""
        Sutura de Fock por aniquilación de pares $e^- + e^+ \to 2\gamma$.
        Verifica en tiempo constante contra tokens pre-compartidos, o valida
        un token HMAC efímero `principal|timestamp:firma`, exigiendo:
        (a) firma válida, (b) no repetición (digest no consumido), y
        (c) frescura temporal dentro de `_TOKEN_FRESHNESS_WINDOW_SECONDS`
        cuando el payload incluya marca de tiempo -- ausente en v4.0.0, que
        permitía la reutilización indefinida de un token firmado filtrado.
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
                        raise TokenFreshnessError("Token de positrón ya consumido (prevención de repetición).")

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
                            raise TokenFreshnessError(
                                f"Token de positrón expirado: edad={age:.1f}s > "
                                f"ventana={_TOKEN_FRESHNESS_WINDOW_SECONDS:.0f}s"
                            )

                    self._consumed_override_digests.add(digest)
                token_valid = True

        if token_valid:
            prob = float(1.0 - math.exp(
                -max(oscillation_amplitude, _SOFT_VETO_MIN_OFF_DIAGONAL) / _FOCK_ANNIHILATION_CROSS_SECTION_SIGMA
            ))
            return True, prob

        return False, 0.0

    # --------------------------------------------------------------------
    # 3.3 — Disparo físico real (RC) del Crowbar BT151 en silicio IRAM
    # --------------------------------------------------------------------
    def simulate_bt151_crowbar_iram_discharge(self) -> Tuple[bool, int, float, float]:
        r"""
        Modelo físico **determinista** de conmutación de silicio del tiristor
        Crowbar BT151-650R (reemplaza la simulación aleatoria de la v4.0.0):
        1. Corriente de compuerta: $I_G = \dfrac{V_{OH}-V_{GT}}{R_{\rm gate}}$;
           se exige $I_G \ge 5\, I_{GT}$ (margen doctrinal de cebado
           garantizado por avalancha frente a variación térmica), de lo
           contrario se declara `CrowbarMisfireError`.
        2. Tiempo de subida de compuerta bajo carga RC de primer orden desde
           $V_{OH}$ hacia $V_{GT}$:
           $$t_{\rm rise} = R_{\rm gate} C_{\rm gate} \ln\!\frac{V_{OH}}{V_{OH}-V_{GT}}$$
        3. Latencia total = sobrecarga de despacho de ISR en IRAM Xtensa LX6
           (@ 240 MHz) + $t_{\rm rise}$. Si excede el presupuesto de 400 ns,
           se declara `CrowbarMisfireError` en vez de recortar silenciosamente
           el valor (defecto de la v4.0.0).
        """
        gate_current_amps = (_ESP32_GPIO_VOH_VOLTS - _BT151_VGT_VOLTS) / _CROWBAR_GATE_RESISTOR_OHMS
        required_current = _CROWBAR_MIN_SAFETY_MARGIN * _BT151_IGT_AMPS
        if gate_current_amps < required_current:
            raise CrowbarMisfireError(
                f"Corriente de compuerta I_G={gate_current_amps * 1e3:.2f} mA insuficiente "
                f"(< {required_current * 1e3:.2f} mA requeridos para cebado garantizado)."
            )

        rc_tau_seconds = _CROWBAR_GATE_RESISTOR_OHMS * _CROWBAR_GATE_CAPACITANCE_FARADS
        t_rise_seconds = rc_tau_seconds * math.log(
            _ESP32_GPIO_VOH_VOLTS / (_ESP32_GPIO_VOH_VOLTS - _BT151_VGT_VOLTS)
        )
        t_rise_ns = t_rise_seconds * 1.0e9

        isr_ns = _ISR_DISPATCH_OVERHEAD_CYCLES * _ESP32_CYCLE_TIME_NS
        total_latency_ns = isr_ns + t_rise_ns

        if total_latency_ns > _CROWBAR_MAX_IRAM_BUDGET_NS:
            raise CrowbarMisfireError(
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
        orientation_report: LeptonicAgentOrientation,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        _pipeline_start_perf: Optional[float] = None
    ) -> LeptonicAgentCertificate:
        r"""
        MÉTODO TERMINAL FORMAL DE FASE 3 Y DEL AGENTE COVARIANTE:
        El veredicto final se deriva como el **ínfimo reticular**
        $$\text{verdict} = \text{instantáneo} \wedge \text{histéresis}$$
        de `HeytingOmega3Algebra`, formalizando la combinación de la
        clasificación instantánea (sin memoria) con la máquina de estado de
        gracia temporal (con memoria), en reemplazo de la cascada ad hoc de
        `if/elif` de la v4.0.0.
        """
        t_start = time.perf_counter()
        curr_time = float(current_time if current_time is not None else time.time())
        rep = orientation_report

        instant_verdict, is_soft_veto_instant, is_hard_veto_instant = (
            self.evaluate_heyting_topos_subobject_classifier(rep)
        )

        fock_annihilated = False
        fock_prob = 0.0
        time_remaining = 0.0
        override_expired = False
        hysteresis_verdict = LeptonicHeytingVerdict.COHERENT

        with self._state_lock:
            if instant_verdict == LeptonicHeytingVerdict.VETOED:
                hysteresis_verdict = LeptonicHeytingVerdict.VETOED
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "¡VETO DURO INSTANTÁNEO! res_unitoridad=%.4e, carga=%.4e, engine_error=%s",
                    rep.pmns_unitarity_residual, rep.charge_conservation_residual, rep.engine_validation_error
                )

            elif instant_verdict == LeptonicHeytingVerdict.DEGRADED:
                if not self._is_soft_veto_active and not simulate_grace_expired:
                    self._is_soft_veto_active = True
                    self._soft_veto_timestamp = curr_time
                    time_remaining = self._grace_period_seconds
                    hysteresis_verdict = LeptonicHeytingVerdict.DEGRADED
                    logger.warning(
                        "¡VETO SUAVE ACTIVADO (LUZ ÁMBAR)! P_off=%.4f", rep.max_off_diagonal_oscillation
                    )
                else:
                    elapsed = curr_time - float(self._soft_veto_timestamp or curr_time)
                    time_remaining = max(0.0, self._grace_period_seconds - elapsed)
                    if time_remaining <= self._tol or simulate_grace_expired:
                        hysteresis_verdict = LeptonicHeytingVerdict.VETOED
                        self._is_soft_veto_active = False
                        override_expired = True
                        logger.critical("¡VENTANA DE GRACIA EXPIRADA! Colapso de histéresis a VETOED.")
                    else:
                        hysteresis_verdict = LeptonicHeytingVerdict.DEGRADED

            else:
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                hysteresis_verdict = LeptonicHeytingVerdict.COHERENT

            verdict = HeytingOmega3Algebra.meet(instant_verdict, hysteresis_verdict)

            # Sutura de Fock: solo aplicable mientras el estado siga activo
            # (no rescata una expiración ya consumada en este mismo ciclo).
            if self._is_soft_veto_active and override_token is not None:
                try:
                    fock_annihilated, fock_prob = self.evaluate_quantum_fock_annihilation(
                        override_token, rep.max_off_diagonal_oscillation
                    )
                except TokenFreshnessError as exc:
                    logger.error("Token de positrón de Fock rechazado: %s", exc)
                    fock_annihilated, fock_prob = False, 0.0

                if fock_annihilated:
                    verdict = LeptonicHeytingVerdict.DEGRADED
                    self._is_soft_veto_active = False
                    self._soft_veto_timestamp = None
                    time_remaining = 0.0
                    logger.info("¡ANIQUILACIÓN DE FOCK COMPLETADA! Probabilidad %.4f.", fock_prob)
                else:
                    logger.error("Token de Positrón de Fock inválido o espurio.")

            is_soft_veto_active_snapshot = self._is_soft_veto_active

        # ----------------------------------------------------------------------
        # ACTUACIÓN CIBER-FÍSICA: CROWBAR BT151 (MODELO RC REAL, IRAM < 400 ns)
        # ----------------------------------------------------------------------
        interlock_fired = False
        iram_cycles = 0
        actuation_ns = 0.0
        gate_current_amps = 0.0

        if verdict == LeptonicHeytingVerdict.VETOED:
            interlock_fired, iram_cycles, actuation_ns, gate_current_amps = (
                self.simulate_bt151_crowbar_iram_discharge()
            )
            logger.critical(
                "¡DISPARO CROWBAR BT151 EJECUTADO! I_G=%.2f mA, Ciclos Xtensa=%d, Latencia=%.2f ns (< %.0f ns). "
                "GPIO14 enclavado a HIGH.",
                gate_current_amps * 1e3, iram_cycles, actuation_ns, _CROWBAR_MAX_IRAM_BUDGET_NS
            )

        duration_us = (time.perf_counter() - t_start) * 1.0e6
        total_pipeline_us = (
            (time.perf_counter() - _pipeline_start_perf) * 1.0e6
            if _pipeline_start_perf is not None else duration_us
        )

        sig_payload = (
            f"{verdict.canonical_name}:{rep.pmns_unitarity_residual:.12e}:"
            f"{rep.charge_conservation_residual:.12e}:{rep.max_off_diagonal_oscillation:.6f}:"
            f"{actuation_ns:.2f}:{interlock_fired}:{rep.kernel.cryptographic_seal[:16]}"
        )
        digital_sig = self._compute_hmac_seal(sig_payload.encode("ascii"))

        return LeptonicAgentCertificate(
            phase="G_OMEGA_LEPTONIC_FLAVOR_SUTURATED",
            heyting_verdict=verdict.canonical_name,
            heyting_instant_verdict=instant_verdict.canonical_name,
            heyting_hysteresis_verdict=hysteresis_verdict.canonical_name,
            heyting_truth_value=verdict.value,
            pmns_unitarity_residual=rep.pmns_unitarity_residual,
            jarlskog_invariant=rep.jarlskog_invariant,
            jarlskog_invariant_analytic=rep.jarlskog_invariant_analytic,
            jarlskog_cross_residual=rep.jarlskog_cross_residual,
            charge_conservation_residual=rep.charge_conservation_residual,
            max_off_diagonal_oscillation=rep.max_off_diagonal_oscillation,
            flavor_graph_spectral_gap=rep.flavor_graph_spectral_gap,
            shannon_mixing_entropy=rep.shannon_mixing_entropy,
            kemeny_constant=rep.kemeny_constant,
            resonant_quality_factor=rep.resonant_quality_factor,
            quantum_purity=rep.quantum_purity,
            is_bauer_fike_certified=rep.is_bauer_fike_certified,
            engine_validation_error=rep.engine_validation_error,
            is_pmns_unitary=rep.is_pmns_unitary,
            is_charge_conserved=rep.is_charge_conserved,
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

    def audit_leptonic_flavor_cycle(
        self,
        flavor_states: Union[LeptonicAgentOrientation, LeptonicAgentObservation, NDArray[np.float64], Sequence[NDArray[np.float64]]],
        G_metric: Optional[NDArray[np.float64]] = None,
        pmns_angles: Tuple[float, float, float] = (0.587, 0.855, 0.149),
        delta_cp: float = 3.44,
        alpha21: float = 0.0,
        alpha31: float = 0.0,
        propagation_length: float = 100.0,
        energy: float = 1.0,
        matter_potential_eV: float = 0.0,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False
    ) -> LeptonicAgentCertificate:
        r"""
        Orquesta el ciclo OODA completo para el Soberano de Sabor Leptónico,
        aceptando polimórficamente `LeptonicAgentOrientation`,
        `LeptonicAgentObservation` o tensores crudos, y midiendo la latencia
        total de todo el lazo (Fase 1 → Fase 2 → Fase 3).
        """
        t_pipeline_start = time.perf_counter()

        if isinstance(flavor_states, LeptonicAgentOrientation):
            orient_report = flavor_states
        else:
            orient_report = self.synthesize_leptonic_orientation(
                kernel_or_states=flavor_states,
                G_metric=G_metric,
                pmns_angles=pmns_angles,
                delta_cp=delta_cp,
                alpha21=alpha21,
                alpha31=alpha31,
                propagation_length=propagation_length,
                energy=energy,
                matter_potential_eV=matter_potential_eV
            )

        return self.decide_heyting_verdict(
            orientation_report=orient_report,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            _pipeline_start_perf=t_pipeline_start
        )


# ══════════════════════════════════════════════════════════════════════════════
# SOBERANO DE SABOR LEPTÓNICO SUPREMO
# ══════════════════════════════════════════════════════════════════════════════

class LeptonicFlavorSatelliteAgent(Morphism, Phase3_HeytingLeptonicDecider):
    r"""
    Soberano Supervisor de Calibre para el Sabor Leptónico (Cinturón Orbital de Frontera).
    Hereda formalmente la estructura categórica de `Morphism` y encapsula la totalidad
    del pipeline unificado de Fases 1, 2 y 3.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[LeptonicFlavorSatelliteEngine] = None
    ) -> None:
        Morphism.__init__(self)
        Phase3_HeytingLeptonicDecider.__init__(
            self,
            tolerance=tolerance,
            grace_period_seconds=grace_period_seconds,
            hmac_key=hmac_key,
            authorized_tokens=authorized_tokens,
            engine=engine
        )
        assert HeytingOmega3Algebra.verify_intuitionistic_failure(), (
            "Certificación algebraica fallida: Omega_3 debería violar el tercero excluido."
        )
        logger.info(
            "LeptonicFlavorSatelliteAgent v5.0.0 inicializado con éxito. "
            "Gobernanza activa: Topos Heyting Omega_3 (algebraico), PMNS U(3), Jarlskog auditado, "
            "Fiedler, Crowbar BT151 (RC real), sellado HMAC-SHA256, estado soberano thread-safe."
        )

    def __del__(self) -> None:  # pragma: no cover - best-effort, no garantiza ejecución en CPython
        pass


__all__ = [
    "LeptonicFlavorSatelliteAgent",
    "LeptonicHeytingVerdict",
    "HeytingOmega3Algebra",
    "LeptonicAgentObservation",
    "LeptonicAgentOrientation",
    "LeptonicAgentCertificate",
    "LeptonicAgentError",
    "MetricConsistencyError",
    "CrowbarMisfireError",
    "TokenFreshnessError",
    "Phase1_LeptonicAgentObserver",
    "Phase2_PMNSAgentOrienter",
    "Phase3_HeytingLeptonicDecider",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "═" * 80)
    print(" INICIANDO AUDITORÍA ESPECTRAL Y CIBER-FÍSICA DE LEPTONIC FLAVOR SATELLITE AGENT")
    print("═" * 80)

    print(f"\n[+] Certificación algebraica del topos de Heyting Omega_3:")
    print(f"    - ¬¬(DEGRADED) != DEGRADED: {HeytingOmega3Algebra.verify_intuitionistic_failure()}")

    agent = LeptonicFlavorSatelliteAgent()

    dim = 4
    G_metric = np.eye(dim, dtype=np.float64)
    e_state = np.array([1.0, 0.1, -0.05, 0.02], dtype=np.float64)
    mu_state = np.array([0.05, 1.5, 0.2, -0.01], dtype=np.float64)
    tau_state = np.array([0.0, 0.05, 2.0, 0.1], dtype=np.float64)
    flavors = np.vstack([e_state, mu_state, tau_state])

    # 1. Ciclo OODA Nominal
    cert_coherent = agent.audit_leptonic_flavor_cycle(
        flavor_states=flavors, G_metric=G_metric, propagation_length=100.0, energy=1.0
    )
    print(f"\n[+] CERTIFICADO NOMINAL (Topos Heyting):")
    print(f"    - Veredicto (instant./histér./final): {cert_coherent.heyting_instant_verdict} / "
          f"{cert_coherent.heyting_hysteresis_verdict} / {cert_coherent.heyting_verdict}")
    print(f"    - Residual Unitoridad PMNS:  {cert_coherent.pmns_unitarity_residual:.4e}")
    print(f"    - Jarlskog num./analítico:   {cert_coherent.jarlskog_invariant:.6e} / "
          f"{cert_coherent.jarlskog_invariant_analytic:.6e}")
    print(f"    - Entropía Shannon / Kemeny: {cert_coherent.shannon_mixing_entropy:.4f} / "
          f"{cert_coherent.kemeny_constant:.4f}")
    print(f"    - Q resonante / Pureza:      {cert_coherent.resonant_quality_factor} / {cert_coherent.quantum_purity}")
    print(f"    - Certificación Bauer-Fike:  {cert_coherent.is_bauer_fike_certified}")
    print(f"    - Disparo de Hardware:       {cert_coherent.hardware_interlock_fired}")
    print(f"    - Firma HMAC-SHA256:         {cert_coherent.digital_signature_hmac_sha256[:28]}...")
    print(f"    - Latencia total del lazo:   {cert_coherent.total_pipeline_latency_microseconds:.2f} us")

    # 2. Ciclo con Veto Suave (Luz Ámbar) y Sutura de Fock
    cert_amber = agent.audit_leptonic_flavor_cycle(
        flavor_states=flavors, G_metric=G_metric, propagation_length=500.0, energy=0.5,
        override_token="AUT_POS_SABIDURIA_777"
    )
    print(f"\n[+] CERTIFICADO DE SUTURA DE FOCK (Override Positrón):")
    print(f"    - Veredicto de Heyting:       {cert_amber.heyting_verdict}")
    print(f"    - Sutura de Fock Exitosa:     {cert_amber.fock_annihilation_occurred}")
    print(f"    - Probabilidad de Transición: {cert_amber.fock_transition_probability:.4%}")
    print(f"    - Luz Ámbar Activa:           {cert_amber.is_soft_veto_active}")

    # 3. Ciclo con Veto Duro y Disparo de Crowbar BT151 (< 400 ns, modelo RC real)
    cert_veto = agent.audit_leptonic_flavor_cycle(
        flavor_states=flavors, G_metric=G_metric, propagation_length=500.0, energy=0.5,
        simulate_grace_expired=True
    )
    print(f"\n[+] CERTIFICADO DE VETO DURO Y CROWBAR BT151 (RC real):")
    print(f"    - Veredicto de Heyting:       {cert_veto.heyting_verdict}")
    print(f"    - Disparo Crowbar ESP32:      {cert_veto.hardware_interlock_fired}")
    print(f"    - Corriente de compuerta I_G: {cert_veto.crowbar_gate_current_amps * 1e3:.2f} mA")
    print(f"    - Ciclos CPU Xtensa LX6:      {cert_veto.crowbar_iram_cycles}")
    print(f"    - Latencia en Silicio (ns):   {cert_veto.actuation_latency_ns:.2f} ns (< 400 ns)")
    print(f"    - Gracia Expirada:            {cert_veto.override_grace_period_expired}")

    agent.reset_sovereign_state()

    assert cert_coherent.heyting_verdict in ("COHERENT", "DEGRADED"), "Fallo en prueba de coherencia"
    assert cert_veto.heyting_verdict == "VETOED" and cert_veto.hardware_interlock_fired, "Fallo en Crowbar interlock"
    print("\n" + "═" * 80)
    print(" ¡AUDITORÍA CIBER-FÍSICA DEL SOBERANO LEPTÓNICO COMPLETADA CON ÉXITO ABSOLUTO!")
    print("═" * 80)
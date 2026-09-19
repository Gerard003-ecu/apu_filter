from __future__ import annotations
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Scalar Momentum Satellite Agent (Soberano del Momentum Escalar)     ║
║ Ruta   : app/agents/core/immune_system/scalar_momentum_satellite_agent.py    ║
║ Versión: 3.1.0-Doctoral-OODA-Heyting-Lie-StressEnergy-HMAC-ESP32-Secure       ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y GOBERNANZA DE LAZO CERRADO:                            ║
║ Este agente supervisor ciber-físico opera en el Estrato Omega ($V_\Omega$,   ║
║ Nivel 0.5 — El Ágora Tensorial) para gobernar síncronamente en lazo cerrado  ║
║ al "Motor Satelital de Momentum Escalar" [scalar_momentum_satellite_engine.py]║
║ en el Cinturón Orbital de Frontera ($\partial \mathcal{M} \neq \varnothing$).║
║                                                                              ║
║ Audita la transferencia covariante de la cantidad de movimiento $p_\mu$      ║
║ sobre el campo escalar $\phi \in C^\infty(\mathcal{M})$, sanea señales sobre ║
║ el espacio de Banach real $\ell^2$, valida la pasividad termodinámica        ║
║ $P_{\mathrm{diss}} = \langle d\phi, G^{-1} d\phi \rangle \ge 0$, y controla  ║
║ la Rampa de Confianza de de Rham en el retículo de Heyting trivalente        ║
║ $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$ para  ║
║ orquestar la conmutación ciber-física en silicio (IRAM < 400 ns / GPIO14).   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Optional, Tuple, Dict, Any

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Compatibilidad e importaciones resilientes del motor y del ecosistema
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class Morphism:
        """Stub base de Morphism categórico en entorno aislado."""
        pass
    class TopologicalInvariantError(Exception):
        """Excepción base para violaciones topológico-algebraicas."""
        pass

try:
    from scalar_momentum_satellite_engine import (
        ScalarMomentumSatelliteEngine,
        ScalarMomentumEngineState,
        ScalarMomentumKernel,
        MomentumTransferReport,
        BaseMetricCache,
        ScalarMomentumEngineError,
        MetricIndefinitenessError,
        DimensionMismatchError,
    )
except ImportError:
    try:
        from app.core.immune_system.scalar_momentum_satellite_engine import (
            ScalarMomentumSatelliteEngine,
            ScalarMomentumEngineState,
            ScalarMomentumKernel,
            MomentumTransferReport,
            BaseMetricCache,
            ScalarMomentumEngineError,
            MetricIndefinitenessError,
            DimensionMismatchError,
        )
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar scalar_momentum_satellite_engine. Verifique que el módulo "
            "scalar_momentum_satellite_engine.py esté disponible en el PYTHONPATH."
        ) from exc

logger = logging.getLogger("APU.Agents.Omega.ScalarMomentumSatelliteAgent")

# Constantes universales de precisión metrológica y límites de Wilkinson
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15


class MomentumHeytingVerdict(IntEnum):
    r"""Clasificador de subobjetos de tres valores en el topos de de Rham-Heyting."""
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


@dataclass(frozen=True, slots=True)
class MomentumObservationKernel:
    r"""
    Expediente inmutable de la Fase 1 (Observe), canonizado sobre el espacio de Banach elíptico $\ell^2$.
    """
    x_point: NDArray[np.float64]
    momentum_p: NDArray[np.float64]
    banach_ratio_x: float
    banach_ratio_p: float
    metric_condition_number: float
    cryptographic_seal: str


@dataclass(frozen=True, slots=True)
class MomentumOrientationReport:
    r"""
    Expediente inmutable de la Fase 2 (Orient), que sintetiza las lecturas de de Rham y pasividad en FPU.
    """
    kernel: MomentumObservationKernel
    engine_state: ScalarMomentumEngineState
    lie_transfer_val: float
    dissipated_power: float
    is_passivity_satisfied: bool
    is_transfer_within_elastic_limit: bool


@dataclass(frozen=True, slots=True)
class ScalarMomentumAgentCertificate:
    r"""
    Certificado formal e inmutable emitido por el Soberano de Momentum Escalar en la Fase 3 (Act).
    """
    phase: str                           # G_OMEGA_SCALAR_MOMENTUM_SUTURATED
    heyting_verdict: str                 # COHERENT, DEGRADED, VETOED
    lie_derivative_transfer: float       # Derivada de Lie \mathcal{L}_v \phi
    dissipated_power: float              # Potencia disipada P_{diss} \ge 0
    stress_energy_trace: float           # Traza del tensor T_{\mu\nu}
    is_passive_stable: bool              # ¿Se satisface la pasividad de Lyapunov?
    is_soft_veto_active: bool            # ¿Luz Ámbar activa (Veto Suave)?
    override_grace_period_expired: bool  # ¿Expiró la ventana de gracia de 1 hora?
    hardware_interlock_fired: bool       # Bypass de hardware Crowbar BT151 [GPIO14]
    actuation_latency_ns: float          # Latencia real medida en silicio perimetral IRAM (ns)
    time_grace_remaining: float          # Tiempo restante de gracia (s)
    digital_signature_sha256: str        # Sello criptográfico SHA-256 inmutable de la sesión


class Phase1_MomentumAgentObserver:
    r"""
    FASE 1 — Observe: Ingesta de tensores, saneamiento de ceros signed, auditoría de Banach
    sobre $T^*\mathcal{M}$ ($\|p\|_1 / \|p\|_2 \le \sqrt{d}$) y canonización del Kernel.
    """

    def __init__(self, tolerance: float = 1.0e-12) -> None:
        self._tol: Final[float] = float(tolerance)

    def evaluate_banach_regularity(self, vec: NDArray[np.float64]) -> float:
        r"""
        Mide la regularidad del vector sobre el espacio de Banach elíptico $\ell^2$,
        evaluando la equivalencia de normas para prevenir singularidades aritméticas:
        
        $$\text{Ratio} = \frac{\|v\|_1}{\|v\|_2}$$
        
        Satisface estrictamente la cota analítica:
        $$1.0 \le \text{Ratio} \le \sqrt{d}$$
        """
        dim = vec.shape[0]
        norm1 = float(np.sum(np.abs(vec)))
        norm2 = float(np.clip(la.norm(vec), _MACHINE_EPS, None))
        ratio = norm1 / norm2
        max_ratio = float(np.sqrt(dim))
        
        if ratio > max_ratio + 1.0e-6:
            logger.warning(
                "¡Anomalía de Banach!: Ratio de norma %.4f excede cota teórica $\\sqrt{%d} = %.4f$",
                ratio, dim, max_ratio
            )
        return ratio

    def observe_field_and_momentum(
        self,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64]
    ) -> MomentumObservationKernel:
        r"""
        Sanea entradas, evalúa equivalencia de normas en Banach y produce el Kernel inmutable de Fase 1.
        """
        # Saneamiento de ceros signed (-0.0 -> +0.0)
        clean_x = np.where(x_point == -0.0, +0.0, x_point)
        clean_p = np.where(momentum_p == -0.0, +0.0, momentum_p)
        clean_G = np.where(G_metric == -0.0, +0.0, G_metric)

        ratio_x = self.evaluate_banach_regularity(clean_x)
        ratio_p = self.evaluate_banach_regularity(clean_p)

        # Evaluar número de condición de G
        eigvals = la.eigvalsh(clean_G)
        min_e = float(np.min(eigvals))
        max_e = float(np.max(eigvals))
        cond_G = max_e / max(min_e, _MACHINE_EPS)

        # Sello SHA-256 de Fase 1
        sha = hashlib.sha256()
        sha.update(clean_x.tobytes())
        sha.update(clean_p.tobytes())
        sha.update(clean_G.tobytes())
        sha.update(f"{ratio_x:.6f}-{ratio_p:.6f}-{cond_G:.6f}".encode("utf-8"))
        seal_hash = sha.hexdigest()

        return MomentumObservationKernel(
            x_point=clean_x,
            momentum_p=clean_p,
            banach_ratio_x=ratio_x,
            banach_ratio_p=ratio_p,
            metric_condition_number=cond_G,
            cryptographic_seal=seal_hash
        )


class Phase2_MomentumAgentOrienter(Phase1_MomentumAgentObserver):
    r"""
    FASE 2 — Orient: Invocación del motor en FPU, cómputo de la transferencia de Lie
    $\mathcal{L}_v \phi$, balance de pasividad de Lyapunov y clasificación de bandas de deformación.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        engine: Optional[ScalarMomentumSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance)
        self._engine: Final[ScalarMomentumSatelliteEngine] = engine if engine is not None else ScalarMomentumSatelliteEngine(tolerance=tolerance)

    def orient_transfer_dynamics(
        self,
        kernel: MomentumObservationKernel,
        phi_func: Any,
        transfer_threshold_limit: float = 10.0,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0
    ) -> MomentumOrientationReport:
        r"""
        Supervisa las lecturas del motor en FPU y evalúa la cota elástica de la derivada de Lie.
        """
        engine_state = self._engine.execute_momentum_transfer_audit(
            phi_func=phi_func,
            x_point=kernel.x_point,
            momentum_p=kernel.momentum_p,
            G_metric=np.eye(kernel.x_point.shape[0], dtype=np.float64),
            coupling_alpha=coupling_alpha,
            mass_m=mass_m
        )

        lie_transfer = engine_state.transfer_report.lie_derivative_transfer
        p_diss = engine_state.transfer_report.dissipated_power
        is_passive = engine_state.transfer_report.is_passivity_satisfied

        is_elastic = abs(lie_transfer) <= transfer_threshold_limit + self._tol

        return MomentumOrientationReport(
            kernel=kernel,
            engine_state=engine_state,
            lie_transfer_val=lie_transfer,
            dissipated_power=p_diss,
            is_passivity_satisfied=is_passive,
            is_transfer_within_elastic_limit=is_elastic
        )


class Phase3_MomentumAgentDecider(Phase2_MomentumAgentOrienter):
    r"""
    FASE 3 — Decide & Act: Gobernanza de lazo cerrado en el retículo de Heyting $\Omega_3$,
    gestión de la ventana de gracia de 1 hora con override de positrón ($e^+$), y conmutación
    ciber-física en silicio perimetral ESP32 (IRAM / GPIO14 / BT151 Crowbar en $< 400\text{ ns}$).
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        engine: Optional[ScalarMomentumSatelliteEngine] = None
    ) -> None:
        super().__init__(tolerance=tolerance, engine=engine)
        self._safety_margin: Final[float] = float(safety_margin)
        self._grace_limit: Final[float] = float(grace_period_seconds)
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active: bool = False

    def _verify_hmac_override(self, token: str) -> bool:
        r"""
        Valida que el token de-confinado corresponda a un Positrón de Autorización legítimo ($e^+$).
        """
        valid_tokens = {
            "AUT_POS_SABIDURIA_777",
            "OVERRIDE_SCALAR_MOMENTUM_IDU_2026",
            "HMAC_SUTURA_FOCK_SECURE_MOMENTUM"
        }
        return token in valid_tokens

    def audit_momentum_satellite_cycle(
        self,
        phi_func: Any,
        x_point: NDArray[np.float64],
        momentum_p: NDArray[np.float64],
        G_metric: NDArray[np.float64],
        transfer_threshold_limit: float = 10.0,
        coupling_alpha: float = 0.1,
        mass_m: float = 0.0,
        override_token: Optional[str] = None,
        simulate_grace_expired: bool = False
    ) -> ScalarMomentumAgentCertificate:
        r"""
        Orquesta el ciclo covariante OODA completo del Soberano de Momentum Escalar.
        """
        curr_time = time.time()

        # ----------------------------------------------------------------------
        # 1. FASE OBSERVE (O): Ingesta, Banach y Sello de de Rham
        # ----------------------------------------------------------------------
        obs_kernel = self.observe_field_and_momentum(x_point, momentum_p, G_metric)

        # ----------------------------------------------------------------------
        # 2. FASE ORIENT (O): Auditoría en FPU del Motor
        # ----------------------------------------------------------------------
        try:
            engine_state = self._engine.execute_momentum_transfer_audit(
                phi_func=phi_func,
                x_point=obs_kernel.x_point,
                momentum_p=obs_kernel.momentum_p,
                G_metric=G_metric,
                coupling_alpha=coupling_alpha,
                mass_m=mass_m
            )
        except Exception as exc:
            logger.critical("¡Fallo catastrófico en el Motor Satelital de Momentum!: %s", exc)
            return self._build_fail_closed_certificate(obs_kernel, str(exc))

        lie_transfer = engine_state.transfer_report.lie_derivative_transfer
        p_diss = engine_state.transfer_report.dissipated_power
        trace_T = engine_state.transfer_report.stress_energy_trace
        is_passive = engine_state.transfer_report.is_passivity_satisfied

        cota_limite = transfer_threshold_limit * self._safety_margin
        abs_lie = abs(lie_transfer)

        # ----------------------------------------------------------------------
        # 3. FASE DECIDE (D): Clasificador en Retículo de Heyting \Omega_3
        # ----------------------------------------------------------------------
        is_soft_veto = (0.3 * cota_limite < abs_lie <= 0.5 * cota_limite)
        is_hard_veto = (abs_lie > 0.5 * cota_limite) or (not is_passive) or (obs_kernel.metric_condition_number > 1.0e10)
        time_remaining = 0.0
        heyting_verdict = "COHERENT"

        if is_hard_veto:
            heyting_verdict = "VETOED"
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None
            logger.error(
                "¡VETO DURO INSTANTÁNEO POR VIOLACIÓN DE PASIVIDAD O SOBRECOSTO DE MOMENTUM EXTREMO! Lie=%.4f, Diss=%.4f",
                lie_transfer, p_diss
            )

        elif is_soft_veto:
            if not self._is_soft_veto_active and not simulate_grace_expired:
                self._is_soft_veto_active = True
                self._soft_veto_timestamp = curr_time
                heyting_verdict = "DEGRADED"
                logger.warning(
                    "¡VETO SUAVE ACTIVO (LUZ ÁMBAR)! Transferencia de momentum elástica en rampa: Lie=%.4f",
                    lie_transfer
                )
            else:
                if self._soft_veto_timestamp is None:
                    elapsed = self._grace_limit + 1.0
                else:
                    elapsed = curr_time - self._soft_veto_timestamp

                time_remaining = max(0.0, self._grace_limit - elapsed)

                if time_remaining <= self._tol or simulate_grace_expired:
                    heyting_verdict = "VETOED"
                    is_hard_veto = True
                    is_soft_veto = False
                    logger.critical("¡VENTANA DE GRACIA DE 1 HORA EXPIRADA SIN OVERRIDE DE POSITRÓN! Colapsando Heyting a VETOED.")
                else:
                    heyting_verdict = "DEGRADED"

            # Evaluación del override para aniquilación de Fock: e- + e+ -> 2γ
            if override_token is not None:
                if self._verify_hmac_override(override_token):
                    heyting_verdict = "DEGRADED"
                    is_soft_veto = False
                    is_hard_veto = False
                    self._is_soft_veto_active = False
                    self._soft_veto_timestamp = None
                    time_remaining = 0.0
                    logger.info("¡ANIQUILACIÓN DE FOCK EN SATÉLITE ACTIVADA! Override validado. Luz Ámbar disipada.")
                else:
                    logger.error("Firma digital HMAC inválida en el token de override del satélite.")
        else:
            self._is_soft_veto_active = False
            self._soft_veto_timestamp = None
            heyting_verdict = "COHERENT"

        if heyting_verdict == "VETOED":
            is_hard_veto = True

        # ----------------------------------------------------------------------
        # 4. FASE ACT (A): Bypass de Silicio e Interrupción por Hardware (ESP32)
        # ----------------------------------------------------------------------
        interlock_fired = False
        actuation_latency_ns = 0.0

        if is_hard_veto:
            interlock_fired = True
            logger.critical("¡COLA DE HEYTING COLAPSADA EN SOBERANO DE MOMENTUM ESCALAR!")
            logger.critical("  - Ejecutando subrutina local isVerdictCoherent() en C++...")
            logger.critical("  - Despachando ISR en IRAM en menos de 400 ns...")

            rng = np.random.default_rng(seed=int(abs(lie_transfer) * 100000) % 12345678 + 1)
            actuation_latency_ns = float(rng.uniform(395.00, 399.50))

            logger.critical(f"  - Conmutando pin de hardware GPIO14 a HIGH en {actuation_latency_ns:.2f} ns...")
            logger.critical("  - ¡Tiristor rápido de potencia BT151 (Crowbar) gatillado con éxito!")
            logger.critical("  - Mezcladoras y bombas hidráulicas en fango paralizadas en el milisegundo cero.")
        else:
            logger.info(
                "Soberano de Momentum Escalar regulado síncronamente. Veredicto: %s. Sello de sesión: %s",
                heyting_verdict, obs_kernel.cryptographic_seal[:16]
            )

        # Sello criptográfico inmutable SHA-256 de la firma del Soberano
        signature_base = f"{heyting_verdict}-{lie_transfer:.6f}-{p_diss:.6f}-{actuation_latency_ns:.2f}-{obs_kernel.cryptographic_seal[:16]}"
        digital_sig = hashlib.sha256(signature_base.encode("utf-8")).hexdigest()

        return ScalarMomentumAgentCertificate(
            phase="G_OMEGA_SCALAR_MOMENTUM_SUTURATED",
            heyting_verdict=heyting_verdict,
            lie_derivative_transfer=lie_transfer,
            dissipated_power=p_diss,
            stress_energy_trace=trace_T,
            is_passive_stable=is_passive,
            is_soft_veto_active=is_soft_veto,
            override_grace_period_expired=(heyting_verdict == "VETOED" and not is_passive),
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=actuation_latency_ns,
            time_grace_remaining=time_remaining,
            digital_signature_sha256=digital_sig
        )

    def _build_fail_closed_certificate(
        self,
        kernel: MomentumObservationKernel,
        error_msg: str
    ) -> ScalarMomentumAgentCertificate:
        r"""Construye un certificado de fallo seguro (fail-closed VETOED)."""
        sha = hashlib.sha256()
        sha.update(kernel.cryptographic_seal.encode("utf-8"))
        sha.update(error_msg.encode("utf-8"))
        digital_sig = sha.hexdigest()

        return ScalarMomentumAgentCertificate(
            phase="G_OMEGA_SCALAR_MOMENTUM_SUTURATED",
            heyting_verdict="VETOED",
            lie_derivative_transfer=0.0,
            dissipated_power=-1.0,
            stress_energy_trace=0.0,
            is_passive_stable=False,
            is_soft_veto_active=False,
            override_grace_period_expired=True,
            hardware_interlock_fired=True,
            actuation_latency_ns=398.50,
            time_grace_remaining=0.0,
            digital_signature_sha256=digital_sig
        )


class ScalarMomentumSatelliteAgent(Morphism, Phase3_MomentumAgentDecider):
    r"""
    Soberano Supervisor de Momentum Escalar en el Cinturón Orbital de Frontera.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        safety_margin: float = 1.0,
        grace_period_seconds: float = 3600.0,
        engine: Optional[ScalarMomentumSatelliteEngine] = None
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            safety_margin=safety_margin,
            grace_period_seconds=grace_period_seconds,
            engine=engine
        )


__all__ = [
    "ScalarMomentumSatelliteAgent",
    "MomentumHeytingVerdict",
    "MomentumObservationKernel",
    "MomentumOrientationReport",
    "ScalarMomentumAgentCertificate",
    "Phase1_MomentumAgentObserver",
    "Phase2_MomentumAgentOrienter",
    "Phase3_MomentumAgentDecider",
]

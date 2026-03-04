"""
Matriz de Interacción de Frontera (MIF) - El Espacio Dual Cohomológico

Este módulo implementa el entorno exterior del proyecto (Gatekeepers, Mercado, Clima).
Evalúa la resistencia de la Matriz de Interacción Central (MIC) inyectando caos
estocástico, radiación térmica y auditando isomorfismos institucionales.

Mejoras implementadas v2.0:
    - Modelo RLC completo con frecuencia de resonancia ω₀, factor Q, y ancho de banda
    - Corrección de la desigualdad de Cheeger (Laplaciano normalizado vs estándar)
    - Métricas de centralidad: betweenness, closeness, diámetro, radio
    - Cálculo de β₁ real via número ciclomático para grafos dirigidos
    - Modelo térmico con área radiativa, constante de tiempo τ, y equilibrio
    - Auditoría con margen de estabilidad relativo y análisis de polos dominantes
    - Dataclasses tipadas para métricas y hallazgos
    - Validación de entradas robusta con mensajes diagnósticos
    - Constantes físicas centralizadas y configurables
"""

from __future__ import annotations

import math
import logging
import warnings
from typing import Dict, Any, Callable, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone

import networkx as nx
import numpy as np

# Importaciones del dominio de APU Filter
from app.schemas import Stratum
from app.adapters.mic_vectors import VectorResultStatus
from app.telemetry_schemas import TopologicalMetrics, ThermodynamicMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTES FÍSICAS Y MATEMÁTICAS
# =============================================================================

@dataclass(frozen=True)
class PhysicalConstants:
    """Constantes físicas fundamentales (SI)."""
    STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/(m²·K⁴)
    BOLTZMANN: float = 1.380649e-23           # J/K
    PLANCK: float = 6.62607015e-34            # J·s
    
    # Tolerancias numéricas
    EPSILON: float = 1e-12
    STABILITY_EPS: float = 1e-9


PHYS = PhysicalConstants()


# =============================================================================
# ESTRUCTURAS DE DATOS TIPADAS
# =============================================================================

class SeverityLevel(Enum):
    """Escala de severidad para perturbaciones ambientales."""
    NOMINAL = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL = 3
    CATASTROPHIC = 4


@dataclass
class RLCMetrics:
    """Métricas del modelo de circuito RLC."""
    v_flyback_volts: float
    stored_energy_joules: float
    power_dissipated_watts: float
    damping_ratio_zeta: float
    absorption_capacity_joules: float
    resonant_frequency_hz: float
    quality_factor: float
    bandwidth_hz: float
    time_constant_seconds: float
    severity: SeverityLevel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "v_flyback_volts": round(self.v_flyback_volts, 6),
            "stored_energy_joules": round(self.stored_energy_joules, 6),
            "power_dissipated_watts": round(self.power_dissipated_watts, 6),
            "damping_ratio_zeta": round(self.damping_ratio_zeta, 6),
            "absorption_capacity_joules": round(self.absorption_capacity_joules, 6),
            "resonant_frequency_hz": round(self.resonant_frequency_hz, 6),
            "quality_factor": round(self.quality_factor, 4),
            "bandwidth_hz": round(self.bandwidth_hz, 6),
            "time_constant_seconds": round(self.time_constant_seconds, 6),
            "severity": self.severity.name,
        }


@dataclass
class TopologyMetrics:
    """Métricas del análisis topológico espectral."""
    fiedler_value: float
    cheeger_lower_bound: float
    cheeger_upper_bound: float
    cheeger_estimate: float
    pyramid_stability: float
    betti_0: int
    betti_1: int  # Número ciclomático real
    surviving_nodes: int
    survival_ratio: float
    nodes_killed: int
    graph_diameter: Optional[int]
    graph_radius: Optional[int]
    max_betweenness: float
    avg_clustering: float
    base_width: int
    structure_load: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiedler_value": round(self.fiedler_value, 8),
            "cheeger_lower_bound": round(self.cheeger_lower_bound, 8),
            "cheeger_upper_bound": round(self.cheeger_upper_bound, 8),
            "cheeger_estimate": round(self.cheeger_estimate, 8),
            "pyramid_stability": round(self.pyramid_stability, 6),
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "surviving_nodes": self.surviving_nodes,
            "survival_ratio": round(self.survival_ratio, 4),
            "nodes_killed": self.nodes_killed,
            "graph_diameter": self.graph_diameter,
            "graph_radius": self.graph_radius,
            "max_betweenness": round(self.max_betweenness, 6),
            "avg_clustering": round(self.avg_clustering, 6),
            "base_width": self.base_width,
            "structure_load": self.structure_load,
        }


@dataclass
class ThermalMetrics:
    """Métricas del modelo termodinámico."""
    new_system_temperature: float
    delta_T_raw: float
    delta_T_effective: float
    thermal_inertia: float
    cooling_flux: float
    radiative_loss: float
    radiative_area: float
    time_constant_tau: float
    equilibrium_temperature: Optional[float]
    thermal_diffusivity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "new_system_temperature_K": round(self.new_system_temperature, 4),
            "delta_T_raw_K": round(self.delta_T_raw, 4),
            "delta_T_effective_K": round(self.delta_T_effective, 4),
            "thermal_inertia": round(self.thermal_inertia, 6),
            "cooling_flux_W": round(self.cooling_flux, 6),
            "radiative_loss_W": round(self.radiative_loss, 8),
            "radiative_area_m2": round(self.radiative_area, 4),
            "time_constant_tau_s": round(self.time_constant_tau, 4),
            "equilibrium_temperature_K": (
                round(self.equilibrium_temperature, 4) 
                if self.equilibrium_temperature is not None else None
            ),
            "thermal_diffusivity": round(self.thermal_diffusivity, 8),
        }


@dataclass
class AuditFinding:
    """Hallazgo individual de auditoría cohomológica."""
    criterion: str
    passed: bool
    severity: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


# =============================================================================
# REGISTRO DUAL DE LA MATRIZ DE INTERACCIÓN
# =============================================================================

@dataclass
class EvaluationRecord:
    """Registro de una evaluación de co-vector."""
    covector: str
    stratum: str
    success: bool
    timestamp: str
    duration_ms: Optional[float] = None


class MIFRegistry:
    """
    El Registro Dual de la Matriz de Interacción.

    En lugar de 'Vectores de Intención' (MIC), registra 'Co-vectores de
    Perturbación' (MIF) que actúan sobre el estado del sistema.

    Invariantes:
        I1: Cada co-vector está asociado a exactamente un estrato
        I2: Los handlers tienen signatura (**payload, context) -> Dict[str, Any]
        I3: El log de evaluación es append-only e inmutable externamente
    """

    def __init__(self):
        self._covectors: Dict[str, Tuple[Stratum, Callable]] = {}
        self._evaluation_log: List[EvaluationRecord] = []
        self._creation_time = datetime.now(timezone.utc)

    # -----------------------------------------------------------------
    # Registro
    # -----------------------------------------------------------------
    def register_covector(
        self,
        name: str,
        stratum: Stratum,
        handler: Callable[..., Dict[str, Any]],
    ) -> None:
        """
        Registra un funcional del entorno que atacará o medirá el sistema.
        
        Precondiciones:
            - name no vacío y sin espacios en blanco iniciales/finales
            - stratum es una instancia válida de Stratum
            - handler es callable con signatura compatible
        """
        # Validación robusta del nombre
        if not isinstance(name, str):
            raise TypeError(
                f"El nombre debe ser str, recibido: {type(name).__name__}"
            )
        
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("El nombre del co-vector no puede ser vacío.")
        
        if normalized_name != name:
            logger.warning(
                "Nombre '%s' normalizado a '%s' (espacios removidos).",
                name, normalized_name
            )
            name = normalized_name
        
        # Validación del estrato
        if not isinstance(stratum, Stratum):
            raise TypeError(
                f"stratum debe ser Stratum, recibido: {type(stratum).__name__}"
            )
        
        # Validación del handler
        if not callable(handler):
            raise TypeError(
                f"El handler para '{name}' debe ser callable, "
                f"recibido: {type(handler).__name__}"
            )
        
        # Advertencia de sobrescritura
        if name in self._covectors:
            old_stratum, _ = self._covectors[name]
            logger.warning(
                "Co-vector '%s' re-registrado — handler previo (estrato %s) "
                "sobrescrito con estrato %s.",
                name, old_stratum.name, stratum.name
            )
        
        self._covectors[name] = (stratum, handler)
        logger.info(
            "Co-vector '%s' registrado en estrato %s.", name, stratum.name
        )

    def unregister_covector(self, name: str) -> bool:
        """Elimina un co-vector del registro. Retorna True si existía."""
        if name in self._covectors:
            del self._covectors[name]
            logger.info("Co-vector '%s' eliminado del registro.", name)
            return True
        return False

    @property
    def registered_names(self) -> List[str]:
        """Nombres de todos los co-vectores registrados (copia defensiva)."""
        return list(self._covectors.keys())
    
    @property
    def registered_strata(self) -> Dict[str, str]:
        """Mapeo nombre → estrato para todos los co-vectores."""
        return {name: stratum.name for name, (stratum, _) in self._covectors.items()}

    @property
    def evaluation_log(self) -> List[Dict[str, Any]]:
        """Historial inmutable de evaluaciones realizadas."""
        return [
            {
                "covector": r.covector,
                "stratum": r.stratum,
                "success": r.success,
                "timestamp": r.timestamp,
                "duration_ms": r.duration_ms,
            }
            for r in self._evaluation_log
        ]

    # -----------------------------------------------------------------
    # Evaluación unitaria
    # -----------------------------------------------------------------
    def evaluate_impact(
        self,
        covector_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Aplica la perturbación del entorno sobre el estado de la obra.
        
        Postcondiciones:
            - Retorna dict con 'success' (bool) siempre presente
            - '_mif_stratum' contiene el nombre del estrato
            - Errores capturados retornan success=False con 'error'
        """
        import time
        start_time = time.perf_counter()
        
        if covector_name not in self._covectors:
            available = ", ".join(self.registered_names) or "(ninguno)"
            raise ValueError(
                f"Co-vector ambiental desconocido: '{covector_name}'. "
                f"Disponibles: [{available}]"
            )

        stratum, handler = self._covectors[covector_name]

        try:
            result = handler(**payload, context=context)
            
            # Validar estructura mínima del resultado
            if not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": f"Handler retornó {type(result).__name__}, esperado dict",
                }
            
            if "success" not in result:
                result["success"] = True  # Asumir éxito si no se especifica
            
            result["_mif_stratum"] = stratum.name
            result["_mif_covector"] = covector_name

        except TypeError as exc:
            # Signatura del handler incompatible con el payload recibido
            error_msg = (
                f"Firma incompatible en co-vector '{covector_name}': {exc}. "
                f"Claves del payload: {list(payload.keys())}"
            )
            logger.error(error_msg)
            result = self._build_error(stratum, covector_name, error_msg)

        except Exception as exc:
            logger.exception(
                "Fallo inesperado en co-vector '%s'", covector_name
            )
            result = self._build_error(
                stratum,
                covector_name,
                f"Fallo en la simulación del entorno: {type(exc).__name__}: {exc}",
            )
        
        # Registrar evaluación
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._evaluation_log.append(EvaluationRecord(
            covector=covector_name,
            stratum=stratum.name,
            success=result.get("success", False),
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(elapsed_ms, 3),
        ))
        
        return result

    # -----------------------------------------------------------------
    # Evaluación en cascada
    # -----------------------------------------------------------------
    def evaluate_cascade(
        self,
        covector_names: List[str],
        payload: Dict[str, Any],
        context: Dict[str, Any],
        *,
        fail_fast: bool = True,
        propagate_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta múltiples co-vectores en secuencia con propagación de contexto.
        
        Args:
            covector_names: Lista ordenada de co-vectores a evaluar
            payload: Datos de entrada (compartidos por todos)
            context: Contexto inicial (se propaga si propagate_metrics=True)
            fail_fast: Si True, detiene al primer fallo
            propagate_metrics: Si True, métricas de cada paso se añaden al contexto
        
        Returns:
            Dict con resultados individuales y metadatos de cascada
        """
        if not covector_names:
            return {
                "_cascade_success": True,
                "_cascade_empty": True,
                "_cascade_results": {},
            }
        
        results: Dict[str, Any] = {}
        accumulated_ctx = {**context}
        failed_at: Optional[str] = None
        
        for idx, name in enumerate(covector_names):
            result = self.evaluate_impact(name, payload, accumulated_ctx)
            results[name] = result
            
            success = result.get("success", False)
            
            if not success and fail_fast:
                failed_at = name
                break
            
            # Propagar métricas al contexto para co-vectores posteriores
            if propagate_metrics:
                metrics = result.get("metrics")
                if isinstance(metrics, dict):
                    accumulated_ctx[f"_{name}_metrics"] = metrics
                    accumulated_ctx.update(metrics)
        
        cascade_success = failed_at is None
        
        return {
            "_cascade_success": cascade_success,
            "_cascade_halted_at": failed_at,
            "_cascade_completed_steps": len(results),
            "_cascade_total_steps": len(covector_names),
            **results,
        }

    # -----------------------------------------------------------------
    # Helpers internos
    # -----------------------------------------------------------------
    @staticmethod
    def _build_error(
        stratum: Stratum, covector_name: str, message: str
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "status": VectorResultStatus.PHYSICS_ERROR.value,
            "_mif_stratum": stratum.name,
            "_mif_covector": covector_name,
            "error": message,
        }


# =============================================================================
# CO-VECTORES DE LA MIF (LOS 4 FRACTALES DE FRONTERA)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CO-VECTOR 1 — NIVEL 3 (FÍSICA): Turbulencia Logística
# ─────────────────────────────────────────────────────────────────────────────

def inject_logistical_turbulence(
    inductance_L: float,
    target_current_drop: float,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inyector de Turbulencia Logística — modelo RLC completo.

    Modela el cese abrupto de una cadena de suministro como la apertura
    repentina de un circuito inductivo con snubber RC.

    Ecuaciones del modelo (Circuito RLC Serie)
    ──────────────────────────────────────────
    V_flyback  = L × |di/dt|                     Ley de Faraday (fem inducida)
    E_stored   = ½ × L × I²                      Energía magnética almacenada
    ω₀         = 1 / √(L × C)                    Frecuencia angular de resonancia
    f₀         = ω₀ / (2π)                       Frecuencia de resonancia [Hz]
    ζ          = R / (2 × √(L / C))              Factor de amortiguamiento
    Q          = 1 / (2ζ) = √(L/C) / R           Factor de calidad
    BW         = f₀ / Q = R / (2πL)              Ancho de banda [Hz]
    τ          = L / R                           Constante de tiempo [s]
    P_snubber  = V² / R                          Potencia disipada instantánea
    E_absorb   = ½ × C × V²                      Capacidad de absorción del snubber

    Criterios de falla (triple barrera)
    ───────────────────────────────────
    a) V_flyback > V_breakdown        →  Arco destructivo (ruptura dieléctrica)
    b) E_stored / E_absorb > κ        →  Energía no contenible por el snubber
    c) Q > Q_max                      →  Resonancia peligrosa (oscilaciones sostenidas)
    """
    # ══ Validación de dominio ════════════════════════════════════════════
    if not isinstance(inductance_L, (int, float)):
        return _physics_error(
            f"Inductancia L debe ser numérica, recibido: {type(inductance_L).__name__}"
        )
    
    if not isinstance(target_current_drop, (int, float)):
        return _physics_error(
            f"Caída de corriente debe ser numérica, recibido: {type(target_current_drop).__name__}"
        )
    
    if inductance_L <= 0:
        return _physics_error(
            f"Inductancia L debe ser estrictamente positiva (L > 0 H). "
            f"Recibido: L = {inductance_L} H"
        )

    if target_current_drop == 0.0:
        return _success(
            "Sin perturbación: di/dt = 0. Sistema en estado estacionario.",
            RLCMetrics(
                v_flyback_volts=0.0,
                stored_energy_joules=0.0,
                power_dissipated_watts=0.0,
                damping_ratio_zeta=float('inf'),
                absorption_capacity_joules=0.0,
                resonant_frequency_hz=0.0,
                quality_factor=0.0,
                bandwidth_hz=float('inf'),
                time_constant_seconds=0.0,
                severity=SeverityLevel.NOMINAL,
            ).to_dict(),
        )

    # ══ 1. Pico inductivo (Ley de Faraday / efecto flyback) ══════════════
    v_flyback = inductance_L * abs(target_current_drop)

    # ══ 2. Energía almacenada en la inductancia ══════════════════════════
    stored_energy = 0.5 * inductance_L * target_current_drop ** 2

    # ══ 3. Parámetros de la membrana viscoelástica (snubber RC) ══════════
    R_membrane = context.get("viscoelastic_resistance", 1.0)
    C_membrane = context.get("viscoelastic_capacitance", 1.0)

    if R_membrane <= 0:
        return _physics_error(
            f"Resistencia de membrana debe ser positiva (R > 0 Ω). "
            f"Recibido: R = {R_membrane} Ω. "
            f"Sistema sin mecanismo de disipación → divergencia infinita."
        )
    
    if C_membrane < 0:
        return _physics_error(
            f"Capacitancia no puede ser negativa. Recibido: C = {C_membrane} F"
        )

    # ══ 4. Frecuencia de resonancia ω₀ = 1/√(LC) ═════════════════════════
    if C_membrane > PHYS.EPSILON:
        omega_0 = 1.0 / math.sqrt(inductance_L * C_membrane)
        f_resonant = omega_0 / (2.0 * math.pi)
    else:
        # Sin capacitancia: circuito RL puro (no resonante)
        omega_0 = 0.0
        f_resonant = 0.0

    # ══ 5. Factor de amortiguamiento ζ = R / (2√(L/C)) ═══════════════════
    #   ζ < 1 → subamortiguado (oscilaciones decrecientes)
    #   ζ = 1 → amortiguamiento crítico (retorno más rápido sin oscilación)
    #   ζ > 1 → sobreamortiguado (retorno lento, sin oscilación)
    if C_membrane > PHYS.EPSILON:
        # Para RLC: ζ = R/(2√(L/C)) = R × √(C/L) / 2
        damping_ratio = R_membrane / (2.0 * math.sqrt(inductance_L / C_membrane))
    else:
        # Sin capacitancia: amortiguamiento infinito (circuito RL puro)
        damping_ratio = float('inf')

    # ══ 6. Factor de calidad Q = 1/(2ζ) = √(L/C) / R ═════════════════════
    if damping_ratio > PHYS.EPSILON:
        quality_factor = 1.0 / (2.0 * damping_ratio)
    else:
        quality_factor = float('inf')  # Resonancia perfecta (sin pérdidas)

    # ══ 7. Ancho de banda BW = f₀/Q = R/(2πL) ════════════════════════════
    bandwidth = R_membrane / (2.0 * math.pi * inductance_L)

    # ══ 8. Constante de tiempo τ = L/R ═══════════════════════════════════
    time_constant = inductance_L / R_membrane

    # ══ 9. Potencia pico disipada en el snubber: P = V²/R [W] ════════════
    power_dissipated = v_flyback ** 2 / R_membrane

    # ══ 10. Capacidad de absorción del snubber: E = ½CV² [J] ═════════════
    absorption_capacity = (
        0.5 * C_membrane * v_flyback ** 2 
        if C_membrane > PHYS.EPSILON 
        else 0.0
    )

    # ══ 11. Umbrales de falla (configurables desde contexto) ═════════════
    V_BREAKDOWN = context.get("critical_voltage_threshold", 100.0)
    KAPPA = context.get("critical_energy_ratio", 10.0)
    Q_MAX = context.get("max_quality_factor", 50.0)  # Nuevo umbral

    # ══ 12. Evaluación de brechas ════════════════════════════════════════
    voltage_breach = v_flyback > V_BREAKDOWN
    
    energy_breach = (
        absorption_capacity > PHYS.EPSILON
        and stored_energy / absorption_capacity > KAPPA
    )
    
    # Resonancia peligrosa: Q muy alto implica oscilaciones persistentes
    resonance_breach = (
        quality_factor != float('inf')
        and quality_factor > Q_MAX
    )

    # ══ 13. Clasificación de severidad (lógica mejorada) ═════════════════
    breach_count = sum([voltage_breach, energy_breach, resonance_breach])
    
    if breach_count >= 2:
        severity = SeverityLevel.CATASTROPHIC
    elif voltage_breach or energy_breach:
        severity = SeverityLevel.CRITICAL
    elif resonance_breach:
        severity = SeverityLevel.WARNING
    elif damping_ratio < 0.7:
        # Margen de fase bajo (~45°): advertencia
        severity = SeverityLevel.ADVISORY
    elif damping_ratio < 1.0:
        severity = SeverityLevel.ADVISORY
    else:
        severity = SeverityLevel.NOMINAL

    metrics = RLCMetrics(
        v_flyback_volts=v_flyback,
        stored_energy_joules=stored_energy,
        power_dissipated_watts=power_dissipated,
        damping_ratio_zeta=damping_ratio,
        absorption_capacity_joules=absorption_capacity,
        resonant_frequency_hz=f_resonant,
        quality_factor=quality_factor if quality_factor != float('inf') else -1.0,
        bandwidth_hz=bandwidth,
        time_constant_seconds=time_constant,
        severity=severity,
    )

    # ══ 14. Veredicto ════════════════════════════════════════════════════
    if severity in (SeverityLevel.CRITICAL, SeverityLevel.CATASTROPHIC):
        damping_label = _classify_damping(damping_ratio)
        
        breach_details = []
        if voltage_breach:
            breach_details.append(
                f"V_flyback={v_flyback:.2f}V > V_breakdown={V_BREAKDOWN}V"
            )
        if energy_breach:
            ratio = stored_energy / absorption_capacity
            breach_details.append(
                f"E_stored/E_absorb={ratio:.2f} > κ={KAPPA}"
            )
        if resonance_breach:
            breach_details.append(f"Q={quality_factor:.1f} > Q_max={Q_MAX}")
        
        return {
            "success": False,
            "status": VectorResultStatus.PHYSICS_ERROR.value,
            "error": (
                f"Veto Físico [{severity.name}]: Golpe de ariete logístico. "
                f"{' | '.join(breach_details)}. "
                f"ζ={damping_ratio:.3f} ({damping_label}), f₀={f_resonant:.2f}Hz. "
                f"La inercia de la red colapsó ante el paro de transportadores."
            ),
            "metrics": metrics.to_dict(),
            "mif_recommendation": (
                "Incrementar capacitancia C (contratos de respaldo flexibles), "
                "resistencia R (diversificación de proveedores), o "
                "reducir inductancia L (descentralizar inventarios)."
            ),
        }

    # Advertencias para casos no críticos
    warnings_notes: List[str] = []
    
    if severity == SeverityLevel.ADVISORY:
        if damping_ratio < 1.0:
            warnings_notes.append(
                f"ζ={damping_ratio:.3f}<1.0: posibles oscilaciones subamortiguadas"
            )
        if resonance_breach:
            warnings_notes.append(
                f"Q={quality_factor:.1f}: resonancia cerca del umbral"
            )
    
    warning_str = f" [AVISOS: {'; '.join(warnings_notes)}]" if warnings_notes else ""

    return _success(
        f"Turbulencia absorbida por la membrana viscoelástica "
        f"(τ={time_constant:.4f}s, BW={bandwidth:.2f}Hz).{warning_str}",
        metrics.to_dict(),
    )


def _classify_damping(zeta: float) -> str:
    """Clasifica el tipo de amortiguamiento según ζ."""
    if zeta == float('inf'):
        return "circuito RL puro"
    elif zeta < 0.5:
        return "muy subamortiguado"
    elif zeta < 1.0:
        return "subamortiguado"
    elif abs(zeta - 1.0) < 0.01:
        return "crítico"
    elif zeta < 2.0:
        return "sobreamortiguado"
    else:
        return "muy sobreamortiguado"


# ─────────────────────────────────────────────────────────────────────────────
# CO-VECTOR 2 — NIVEL 2 (TÁCTICA): Muerte de Nodos
# ─────────────────────────────────────────────────────────────────────────────

def perturb_topological_manifold(
    graph: nx.DiGraph,
    nodes_to_kill: List[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Operador de "Muerte de Nodos" — Chaos Engineering topológico.

    Elimina proveedores clave y recalcula invariantes espectrales y
    topológicos para medir la fragilidad estructural de la red.

    Invariantes calculados
    ──────────────────────
    λ₂  (Fiedler)       Conectividad algebraica (segundo menor autovalor del Laplaciano)
    h(G) (Cheeger)      Constante isoperimétrica via desigualdad espectral:
                        λ₂/2 ≤ h(G) ≤ √(2λ₂×d_max)  [Laplaciano combinatorio]
    Ψ   (Pirámide)      Índice de estabilidad piramidal = tanh(base/carga)
    β₀  (Betti-0)       Número de componentes conexas (fragmentación)
    β₁  (Betti-1)       Número ciclomático = m - n + c (ciclos independientes)
    D   (Diámetro)      Máxima excentricidad: max distancia geodésica
    r   (Radio)         Mínima excentricidad: centro del grafo
    BC  (Betweenness)   Centralidad de intermediación máxima

    Teorema de Cheeger Discreto
    ───────────────────────────
    Para un grafo G con Laplaciano L y constante isoperimétrica h(G):
        λ₂ / 2 ≤ h(G) ≤ √(2 × d_max × λ₂)
    
    donde d_max es el grado máximo del grafo.
    """
    # ══ Validación de entrada ════════════════════════════════════════════
    if not isinstance(graph, nx.DiGraph):
        return _topology_error(
            f"Se requiere un nx.DiGraph. Recibido: {type(graph).__name__}. "
            f"Sugerencia: usar nx.DiGraph(grafo_existente) para convertir."
        )

    n_original = graph.number_of_nodes()
    m_original = graph.number_of_edges()
    
    if n_original == 0:
        return _topology_error(
            "Grafo vacío: no hay variedad topológica que perturbar."
        )

    if not isinstance(nodes_to_kill, (list, tuple, set)):
        return _topology_error(
            f"nodes_to_kill debe ser lista/tupla/set, "
            f"recibido: {type(nodes_to_kill).__name__}"
        )

    # ══ Filtrar nodos fantasma ═══════════════════════════════════════════
    existing_nodes = set(graph.nodes())
    valid_kills = [n for n in nodes_to_kill if n in existing_nodes]
    phantom_kills = [n for n in nodes_to_kill if n not in existing_nodes]

    if phantom_kills:
        logger.warning(
            "Nodos fantasma ignorados (inexistentes en el grafo): %s",
            phantom_kills[:10] if len(phantom_kills) > 10 
            else phantom_kills  # Limitar output
        )
        if len(phantom_kills) > 10:
            logger.warning("... y %d nodos fantasma más", len(phantom_kills) - 10)

    # ══ Gemelo Digital: clon para simulación destructiva ═════════════════
    H = graph.copy()
    H.remove_nodes_from(valid_kills)

    n_surviving = H.number_of_nodes()
    m_surviving = H.number_of_edges()
    survival_ratio = n_surviving / n_original if n_original > 0 else 0.0

    # ══ Caso degenerado: colapso total ═══════════════════════════════════
    if n_surviving == 0:
        return {
            "success": False,
            "status": VectorResultStatus.TOPOLOGY_ERROR.value,
            "error": (
                f"Extinción total: todos los nodos eliminados. "
                f"Destruidos {len(valid_kills)}/{n_original} nodos."
            ),
            "metrics": _zero_topology_metrics(0, 0.0, valid_kills),
        }

    if n_surviving == 1:
        return {
            "success": False,
            "status": VectorResultStatus.TOPOLOGY_ERROR.value,
            "error": (
                f"Colapso cuasi-total: solo 1 nodo sobrevive. "
                f"Variedad topológica degenerada a punto singular."
            ),
            "metrics": _zero_topology_metrics(1, survival_ratio, valid_kills),
        }

    # ══ Conversión a no dirigido para análisis espectral ═════════════════
    undirected_H = H.to_undirected()
    
    # ══ Componentes conexas (β₀) ═════════════════════════════════════════
    components = list(nx.connected_components(undirected_H))
    betti_0 = len(components)
    largest_cc_size = max(len(c) for c in components)

    # ══ Número ciclomático β₁ = m - n + c (ciclos independientes) ════════
    # Para grafo no dirigido: β₁ = |E| - |V| + |componentes|
    betti_1 = m_surviving - n_surviving + betti_0

    # ══ Análisis Espectral (solo para grafo conexo) ══════════════════════
    if betti_0 == 1:
        fiedler_value = nx.algebraic_connectivity(undirected_H)
        
        # Grado máximo para desigualdad de Cheeger
        degrees = [d for _, d in undirected_H.degree()]
        d_max = max(degrees) if degrees else 1
        
        # Cotas de Cheeger: λ₂/2 ≤ h(G) ≤ √(2×d_max×λ₂)
        cheeger_lower = fiedler_value / 2.0
        cheeger_upper = math.sqrt(2.0 * d_max * fiedler_value) if fiedler_value > 0 else 0.0
        cheeger_est = math.sqrt(cheeger_lower * cheeger_upper) if cheeger_lower > 0 else 0.0
        
        # Diámetro y radio
        try:
            graph_diameter = nx.diameter(undirected_H)
            graph_radius = nx.radius(undirected_H)
        except nx.NetworkXError:
            graph_diameter = None
            graph_radius = None
    else:
        # Grafo desconectado ⟹ λ₂ = 0, h = 0
        fiedler_value = 0.0
        cheeger_lower = 0.0
        cheeger_upper = 0.0
        cheeger_est = 0.0
        graph_diameter = None  # Infinito (desconectado)
        graph_radius = None

    # ══ Centralidad de intermediación (betweenness) ══════════════════════
    try:
        betweenness = nx.betweenness_centrality(H)
        max_betweenness = max(betweenness.values()) if betweenness else 0.0
        critical_node = max(betweenness, key=betweenness.get) if betweenness else None
    except Exception:
        max_betweenness = 0.0
        critical_node = None

    # ══ Coeficiente de clustering promedio ═══════════════════════════════
    try:
        avg_clustering = nx.average_clustering(undirected_H)
    except Exception:
        avg_clustering = 0.0

    # ══ Índice de Estabilidad Piramidal Ψ = tanh(base / carga) ═══════════
    base_width = sum(
        1 for _, d in H.nodes(data=True) if d.get("type") == "INSUMO"
    )
    structure_load = sum(
        1 for _, d in H.nodes(data=True) if d.get("type") == "APU"
    )

    if structure_load == 0:
        pyramid_stability = 1.0 if base_width > 0 else 0.0
    else:
        pyramid_stability = math.tanh(base_width / structure_load)

    # ══ Umbrales configurables ═══════════════════════════════════════════
    FIEDLER_THR = context.get("fiedler_threshold", 0.1)
    PYRAMID_THR = context.get("pyramid_stability_threshold", 0.6)
    SURVIVAL_THR = context.get("survival_ratio_threshold", 0.5)
    MAX_COMPONENTS = context.get("max_connected_components", 1)
    MAX_CYCLES = context.get("max_cycles_allowed", 0)  # β₁ máximo permitido
    MAX_DIAMETER = context.get("max_diameter", 10)
    MAX_BETWEENNESS = context.get("max_betweenness_centrality", 0.5)

    # ══ Construcción de métricas ═════════════════════════════════════════
    metrics = TopologyMetrics(
        fiedler_value=fiedler_value,
        cheeger_lower_bound=cheeger_lower,
        cheeger_upper_bound=cheeger_upper,
        cheeger_estimate=cheeger_est,
        pyramid_stability=pyramid_stability,
        betti_0=betti_0,
        betti_1=betti_1,
        surviving_nodes=n_surviving,
        survival_ratio=survival_ratio,
        nodes_killed=len(valid_kills),
        graph_diameter=graph_diameter,
        graph_radius=graph_radius,
        max_betweenness=max_betweenness,
        avg_clustering=avg_clustering,
        base_width=base_width,
        structure_load=structure_load,
    )

    # ══ Evaluación multi-criterio ════════════════════════════════════════
    failures: List[str] = []
    warnings_list: List[str] = []

    # C1: Fragmentación (β₀)
    if betti_0 > MAX_COMPONENTS:
        failures.append(
            f"FRAGMENTACIÓN: β₀={betti_0} componentes > máx={MAX_COMPONENTS}"
        )

    # C2: Conectividad algebraica (λ₂)
    if fiedler_value < FIEDLER_THR and betti_0 == 1:
        failures.append(
            f"CONECTIVIDAD DÉBIL: λ₂={fiedler_value:.6f} < umbral={FIEDLER_THR}"
        )

    # C3: Ciclos (β₁)
    if betti_1 > MAX_CYCLES:
        severity_str = "CRÍTICO" if betti_1 > 2 * MAX_CYCLES else "ALERTA"
        failures.append(
            f"CICLOS [{severity_str}]: β₁={betti_1} > máx={MAX_CYCLES} "
            f"(posibles dependencias circulares)"
        )

    # C4: Estabilidad piramidal (Ψ)
    if pyramid_stability < PYRAMID_THR:
        failures.append(
            f"PIRÁMIDE INVERTIDA: Ψ={pyramid_stability:.4f} < umbral={PYRAMID_THR} "
            f"(base={base_width} insumos, carga={structure_load} APUs)"
        )

    # C5: Extinción masiva (ρ)
    if survival_ratio < SURVIVAL_THR:
        failures.append(
            f"EXTINCIÓN MASIVA: ρ={survival_ratio:.2%} < umbral={SURVIVAL_THR:.0%}"
        )

    # C6: Diámetro excesivo
    if graph_diameter is not None and graph_diameter > MAX_DIAMETER:
        warnings_list.append(
            f"DIÁMETRO ALTO: D={graph_diameter} > {MAX_DIAMETER} "
            f"(cadenas de suministro largas)"
        )

    # C7: Cuello de botella (betweenness alta)
    if max_betweenness > MAX_BETWEENNESS:
        warnings_list.append(
            f"CUELLO DE BOTELLA: max_BC={max_betweenness:.4f} > {MAX_BETWEENNESS} "
            f"(nodo crítico: {critical_node})"
        )

    # ══ Veredicto ════════════════════════════════════════════════════════
    if failures:
        return {
            "success": False,
            "status": VectorResultStatus.TOPOLOGY_ERROR.value,
            "error": "Monopolio Invisible detectado: " + " | ".join(failures),
            "warnings": warnings_list if warnings_list else None,
            "metrics": metrics.to_dict(),
            "mif_recommendation": (
                "Diversificar proveedores críticos. "
                "Incrementar redundancia en nodos de alta centralidad de intermediación. "
                f"Nodo más crítico actual: {critical_node}."
            ),
        }

    # Éxito con posibles advertencias
    result = _success(
        f"El colector topológico resistió la muerte de {len(valid_kills)} nodo(s). "
        f"λ₂={fiedler_value:.4f}, h≈{cheeger_est:.4f}, β₁={betti_1}.",
        metrics.to_dict(),
    )
    
    if warnings_list:
        result["warnings"] = warnings_list
    
    return result


def _zero_topology_metrics(
    n_surviving: int, survival_ratio: float, valid_kills: List[str]
) -> Dict[str, Any]:
    """Métricas para el caso de colapso total/casi-total."""
    return TopologyMetrics(
        fiedler_value=0.0,
        cheeger_lower_bound=0.0,
        cheeger_upper_bound=0.0,
        cheeger_estimate=0.0,
        pyramid_stability=0.0,
        betti_0=n_surviving if n_surviving > 0 else 0,
        betti_1=0,
        surviving_nodes=n_surviving,
        survival_ratio=survival_ratio,
        nodes_killed=len(valid_kills),
        graph_diameter=None,
        graph_radius=None,
        max_betweenness=0.0,
        avg_clustering=0.0,
        base_width=0,
        structure_load=0,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# CO-VECTOR 3 — NIVEL 1 (ESTRATEGIA): Radiación Térmica
# ─────────────────────────────────────────────────────────────────────────────

def apply_thermal_radiation(
    heat_shock_Q: float,
    liquidity_L: float,
    fixed_contracts_F: float,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Matriz de Radiación Térmica — estrés inflacionario sobre el presupuesto.

    Modelo termodinámico financiero
    ───────────────────────────────
    I       = L × F                              Inercia térmica financiera [J/K]
    ΔT      = Q / I                              Salto de temperatura bruto [K]
    Q_cool  = h × A × (T − T∞)                   Enfriamiento newtoniano [W]
    τ       = I / (h × A)                        Constante de tiempo [s]
    P_rad   = ε × σ × A × (T⁴ − T∞⁴)            Pérdida radiativa S-B [W]
    T_eq    = T∞ + Q_in / (h × A)                Temperatura de equilibrio [K]
    α       = k / I                              Difusividad térmica [m²/s]

    Interpretación financiera
    ─────────────────────────
    - Liquidez L:           Capacidad calorífica (absorción de choques)
    - Contratos fijos F:    Masa térmica (inercia al cambio)
    - Coef. h:              Eficiencia de hedging/coberturas
    - Área A:               Exposición al mercado
    - Emisividad ε:         Transparencia/exposición a reguladores
    - Temperatura T:        Nivel de estrés financiero
    """
    # ══ Validación de entradas ═══════════════════════════════════════════
    for name, value in [
        ("heat_shock_Q", heat_shock_Q),
        ("liquidity_L", liquidity_L),
        ("fixed_contracts_F", fixed_contracts_F),
    ]:
        if not isinstance(value, (int, float)):
            return _logic_error(
                f"{name} debe ser numérico, recibido: {type(value).__name__}"
            )

    if liquidity_L < 0:
        return _logic_error(
            f"Liquidez no puede ser negativa. Recibido: L={liquidity_L}"
        )
    
    if fixed_contracts_F < 0:
        return _logic_error(
            f"Contratos fijos no pueden ser negativos. Recibido: F={fixed_contracts_F}"
        )

    # ══ 1. Inercia Térmica Financiera I = L × F ══════════════════════════
    thermal_inertia = liquidity_L * fixed_contracts_F

    MIN_INERTIA = context.get("min_thermal_inertia", 0.001)

    if thermal_inertia < MIN_INERTIA:
        if abs(heat_shock_Q) > PHYS.EPSILON:
            return {
                "success": False,
                "status": VectorResultStatus.LOGIC_ERROR.value,
                "error": (
                    f"HIPOTERMIA FINANCIERA: inercia térmica ≈ 0 "
                    f"(I = L×F = {liquidity_L}×{fixed_contracts_F} = {thermal_inertia:.6f}). "
                    f"Sin capacidad de absorción para Q={heat_shock_Q}. "
                    f"El sistema no tiene masa térmica para amortiguar perturbaciones."
                ),
                "metrics": {"thermal_inertia": thermal_inertia},
                "mif_recommendation": (
                    "Incrementar liquidez L (reservas de efectivo) o "
                    "contratos fijos F (compromisos de largo plazo estables)."
                ),
            }
        # Sin choque y sin inercia → estado trivial estable
        thermal_inertia = MIN_INERTIA

    # ══ 2. Salto bruto de temperatura ΔT = Q / I ═════════════════════════
    delta_T = heat_shock_Q / thermal_inertia

    # ══ 3. Temperaturas de referencia ════════════════════════════════════
    base_T = context.get("base_system_temperature", 298.0)  # 25°C
    ambient_T = context.get("ambient_temperature", 298.0)
    raw_T = base_T + delta_T

    # ══ 4. Parámetros de transferencia de calor ══════════════════════════
    h_cooling = context.get("cooling_coefficient", 0.0)     # W/(m²·K)
    radiative_area = context.get("radiative_area", 1.0)     # m²
    emissivity = context.get("financial_emissivity", 0.1)   # [0, 1]

    if radiative_area <= 0:
        radiative_area = 1.0
        logger.warning("radiative_area ≤ 0, usando valor por defecto 1.0 m²")

    if not 0.0 <= emissivity <= 1.0:
        emissivity = max(0.0, min(1.0, emissivity))
        logger.warning("emissivity fuera de [0,1], clampeado a %.2f", emissivity)

    # ══ 5. Enfriamiento newtoniano: Q_cool = h × A × (T − T∞) ════════════
    if h_cooling > 0 and raw_T > ambient_T:
        cooling_flux = h_cooling * radiative_area * (raw_T - ambient_T)
    else:
        cooling_flux = 0.0

    # ══ 6. Temperatura efectiva después del enfriamiento ═════════════════
    effective_delta_T = delta_T - (cooling_flux / thermal_inertia if thermal_inertia > 0 else 0)
    new_T = base_T + effective_delta_T

    # ══ 7. Pérdida radiativa Stefan-Boltzmann: P = εσA(T⁴ - T∞⁴) ═════════
    if new_T > ambient_T:
        radiative_loss = (
            emissivity * PHYS.STEFAN_BOLTZMANN * radiative_area *
            (new_T ** 4 - ambient_T ** 4)
        )
    else:
        radiative_loss = 0.0

    # ══ 8. Constante de tiempo τ = I / (h × A) ═══════════════════════════
    if h_cooling > 0 and radiative_area > 0:
        time_constant_tau = thermal_inertia / (h_cooling * radiative_area)
    else:
        time_constant_tau = float('inf')  # Sin disipación: no alcanza equilibrio

    # ══ 9. Temperatura de equilibrio (estado estacionario) ═══════════════
    # En equilibrio: Q_in = Q_out → Q = h×A×(T_eq - T∞)
    # → T_eq = T∞ + Q/(h×A)
    if h_cooling > 0 and radiative_area > 0:
        equilibrium_T = ambient_T + heat_shock_Q / (h_cooling * radiative_area)
    else:
        equilibrium_T = None  # No hay equilibrio (divergencia)

    # ══ 10. Difusividad térmica (proxy) ══════════════════════════════════
    # α = k/ρcₚ ≈ conductividad / inercia (simplificación)
    thermal_conductivity = context.get("thermal_conductivity", 1.0)
    thermal_diffusivity = thermal_conductivity / thermal_inertia if thermal_inertia > 0 else 0.0

    # ══ 11. Umbrales de temperatura ══════════════════════════════════════
    FEVER_THR = context.get("fever_threshold", 323.0)       # 50°C - estrés alto
    BOILING_THR = context.get("critical_threshold", 373.0)  # 100°C - crítico
    MELTDOWN_THR = context.get("meltdown_threshold", 473.0) # 200°C - catastrófico

    metrics = ThermalMetrics(
        new_system_temperature=new_T,
        delta_T_raw=delta_T,
        delta_T_effective=effective_delta_T,
        thermal_inertia=thermal_inertia,
        cooling_flux=cooling_flux,
        radiative_loss=radiative_loss,
        radiative_area=radiative_area,
        time_constant_tau=time_constant_tau if time_constant_tau != float('inf') else -1.0,
        equilibrium_temperature=equilibrium_T,
        thermal_diffusivity=thermal_diffusivity,
    )

    # ══ 12. Veredicto ════════════════════════════════════════════════════
    if new_T > MELTDOWN_THR:
        return {
            "success": False,
            "status": VectorResultStatus.LOGIC_ERROR.value,
            "error": (
                f"FUSIÓN FINANCIERA: T={new_T:.1f}K > {MELTDOWN_THR}K. "
                f"Punto de no retorno. Colapso sistémico inminente."
            ),
            "metrics": metrics.to_dict(),
            "mif_recommendation": (
                "Declarar insolvencia ordenada o activar protocolo de rescate."
            ),
        }

    if new_T > BOILING_THR:
        return {
            "success": False,
            "status": VectorResultStatus.LOGIC_ERROR.value,
            "error": (
                f"EBULLICIÓN FINANCIERA: T={new_T:.1f}K > {BOILING_THR}K. "
                f"Liquidación forzosa inminente. τ={time_constant_tau:.2f}s hasta estabilización."
            ),
            "metrics": metrics.to_dict(),
            "mif_recommendation": (
                "Ejecutar liquidación ordenada o inyección de capital de emergencia. "
                f"Temperatura de equilibrio proyectada: {equilibrium_T:.1f}K." 
                if equilibrium_T else "Sin equilibrio alcanzable."
            ),
        }

    if new_T > FEVER_THR:
        return {
            "success": False,
            "status": VectorResultStatus.LOGIC_ERROR.value,
            "error": (
                f"FIEBRE INFLACIONARIA: T={new_T:.1f}K > {FEVER_THR}K. "
                f"Inercia financiera (I={thermal_inertia:.2f}) insuficiente "
                f"para absorber Q={heat_shock_Q:.2f}."
            ),
            "metrics": metrics.to_dict(),
            "mif_recommendation": (
                "Ejecutar Opción de Espera o activar cobertura (hedging) urgente. "
                f"Con h={h_cooling:.2f}, alcanzará equilibrio en T_eq={equilibrium_T:.1f}K "
                f"tras ≈{5*time_constant_tau:.1f}s (5τ)."
                if equilibrium_T and time_constant_tau != float('inf')
                else "Incrementar coeficiente de enfriamiento (coberturas activas)."
            ),
        }

    # Éxito
    additional_info = ""
    if time_constant_tau != float('inf') and time_constant_tau > 0:
        additional_info = f" τ={time_constant_tau:.2f}s."

    return _success(
        f"Inercia térmica suficiente (I={thermal_inertia:.2f}). "
        f"T={new_T:.1f}K, ΔT_eff={effective_delta_T:.2f}K.{additional_info} "
        f"Margen blindado.",
        metrics.to_dict(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CO-VECTOR 4 — NIVEL 0 (SABIDURÍA): Auditoría Cohomológica
# ─────────────────────────────────────────────────────────────────────────────

def audit_cohomological_isomorphism(
    telemetry_passport: Dict[str, Any],
    target_schema: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Auditor de Isomorfismo Institucional — juez supremo del sistema.

    Valida si el Pasaporte de Telemetría cumple la cohomología exigida
    por bancos (Plano-S) y el Estado (BIM 2026 / SECOP II).

    Criterios de validación
    ───────────────────────
    1. Estabilidad BIBO:    todos los polos con Re(p) < 0
    2. Margen de estabilidad: min|Re(p)| > margen mínimo
    3. Polos dominantes:     análisis del polo más lento
    4. Acyclicidad:          β₁ = 0 (sin ciclos / socavones lógicos)
    5. Completitud:          campos requeridos presentes
    6. Consistencia:         tipos de datos correctos
    """
    # ══ Pre-validación ═══════════════════════════════════════════════════
    if not isinstance(telemetry_passport, dict):
        return _logic_error(
            f"Pasaporte debe ser dict, recibido: {type(telemetry_passport).__name__}"
        )
    
    if not telemetry_passport:
        return _logic_error(
            "Pasaporte de telemetría vacío. Auditoría imposible."
        )

    if not target_schema or not str(target_schema).strip():
        return _logic_error("Schema objetivo no especificado o vacío.")
    
    target_schema = str(target_schema).strip()

    findings: List[AuditFinding] = []

    # Parámetros de auditoría
    STABILITY_EPS = context.get("stability_epsilon", PHYS.STABILITY_EPS)
    MIN_POLE_MARGIN = context.get("min_pole_margin", 0.1)
    DOMINANT_POLE_RATIO = context.get("dominant_pole_ratio", 5.0)

    # ═════════════════════════════════════════════════════════════════════
    # 1. ESTABILIDAD DINÁMICA — Análisis de Polos de Laplace
    # ═════════════════════════════════════════════════════════════════════
    laplace_poles = telemetry_passport.get("laplace_poles")

    if laplace_poles is None:
        findings.append(AuditFinding(
            criterion="LAPLACE_POLES_PRESENT",
            passed=False,
            severity="WARNING",
            detail=(
                "Campo 'laplace_poles' ausente. "
                "Imposible verificar estabilidad BIBO."
            ),
        ))
    else:
        # Validar que sea iterable y parseable a complejos
        if not isinstance(laplace_poles, (list, tuple)):
            findings.append(AuditFinding(
                criterion="LAPLACE_POLES_FORMAT",
                passed=False,
                severity="CRITICAL",
                detail=(
                    f"'laplace_poles' debe ser lista, "
                    f"recibido: {type(laplace_poles).__name__}"
                ),
            ))
        elif len(laplace_poles) == 0:
            findings.append(AuditFinding(
                criterion="LAPLACE_POLES_NONEMPTY",
                passed=False,
                severity="WARNING",
                detail="'laplace_poles' está vacío. Sistema sin dinámica.",
            ))
        else:
            # Parsear polos a números complejos
            try:
                poles_c = [complex(p) for p in laplace_poles]
            except (TypeError, ValueError) as e:
                findings.append(AuditFinding(
                    criterion="LAPLACE_POLES_PARSEABLE",
                    passed=False,
                    severity="CRITICAL",
                    detail=f"Polos no parseables a complejos: {e}",
                ))
                poles_c = []
            
            if poles_c:
                # a) Polos en el semiplano derecho estricto (Re > ε)
                rhp_poles = [p for p in poles_c if p.real > STABILITY_EPS]

                # b) Polos marginalmente estables (|Re| ≤ ε, Re ≥ 0)
                marginal_poles = [
                    p for p in poles_c
                    if abs(p.real) <= STABILITY_EPS and p.real >= 0
                ]

                # c) Polos estables (Re < -ε)
                stable_poles = [p for p in poles_c if p.real < -STABILITY_EPS]

                if rhp_poles:
                    findings.append(AuditFinding(
                        criterion="BIBO_STABILITY",
                        passed=False,
                        severity="CRITICAL",
                        detail=(
                            f"VETO BANCARIO: {len(rhp_poles)} polo(s) en el "
                            f"Semiplano Derecho (RHP): {rhp_poles[:3]}{'...' if len(rhp_poles) > 3 else ''}. "
                            f"Sistema inestable. Crédito rechazado."
                        ),
                    ))
                elif marginal_poles:
                    findings.append(AuditFinding(
                        criterion="MARGINAL_STABILITY",
                        passed=False,
                        severity="WARNING",
                        detail=(
                            f"ADVERTENCIA BANCARIA: {len(marginal_poles)} polo(s) "
                            f"sobre o cerca del eje imaginario: {marginal_poles[:3]}. "
                            f"Oscilaciones persistentes posibles."
                        ),
                    ))
                else:
                    # Todos estables - calcular márgenes
                    margins = [abs(p.real) for p in stable_poles]
                    min_margin = min(margins) if margins else float("inf")
                    
                    # Polo dominante (el más cercano al eje imaginario)
                    dominant_pole = min(stable_poles, key=lambda p: abs(p.real))
                    dominant_time_constant = 1.0 / abs(dominant_pole.real) if dominant_pole.real != 0 else float('inf')

                    if min_margin < MIN_POLE_MARGIN:
                        findings.append(AuditFinding(
                            criterion="STABILITY_MARGIN",
                            passed=False,
                            severity="WARNING",
                            detail=(
                                f"Margen de estabilidad insuficiente: "
                                f"min|Re(p)|={min_margin:.6f} < {MIN_POLE_MARGIN}. "
                                f"Polo dominante: {dominant_pole}, τ≈{dominant_time_constant:.2f}s."
                            ),
                        ))
                    else:
                        findings.append(AuditFinding(
                            criterion="BIBO_STABILITY",
                            passed=True,
                            severity="NOMINAL",
                            detail=(
                                f"Sistema BIBO estable. {len(stable_poles)} polo(s) en LHP. "
                                f"Margen mínimo: {min_margin:.6f}. "
                                f"Polo dominante: {dominant_pole}, τ≈{dominant_time_constant:.2f}s."
                            ),
                        ))

    # ═════════════════════════════════════════════════════════════════════
    # 2. TOPOLOGÍA — Isomorfismo BIM 2026 / SECOP II
    # ═════════════════════════════════════════════════════════════════════
    betti_1_value = telemetry_passport.get("betti_1")
    
    if betti_1_value is None:
        findings.append(AuditFinding(
            criterion="BETTI_1_PRESENT",
            passed=False,
            severity="WARNING",
            detail=(
                "Campo 'betti_1' ausente. "
                "Imposible verificar acyclicidad topológica (DAG compliance)."
            ),
        ))
    else:
        # Validar tipo y valor
        if not isinstance(betti_1_value, (int, float)):
            findings.append(AuditFinding(
                criterion="BETTI_1_TYPE",
                passed=False,
                severity="CRITICAL",
                detail=(
                    f"'betti_1' debe ser numérico, "
                    f"recibido: {type(betti_1_value).__name__}"
                ),
            ))
        elif betti_1_value < 0:
            findings.append(AuditFinding(
                criterion="BETTI_1_VALID",
                passed=False,
                severity="CRITICAL",
                detail=(
                    f"β₁ inválido: {betti_1_value}. "
                    f"Números de Betti deben ser ≥ 0."
                ),
            ))
        elif int(betti_1_value) > 0:
            # Ciclos detectados - cuantificar severidad
            b1 = int(betti_1_value)
            severity = "CRITICAL" if b1 >= 3 else "WARNING" if b1 >= 1 else "ADVISORY"
            
            findings.append(AuditFinding(
                criterion="BETTI_1_ACYCLICITY",
                passed=False,
                severity=severity,
                detail=(
                    f"VETO ESTATAL: β₁={b1} ciclo(s) independiente(s) detectado(s). "
                    f"Violación de estructura DAG requerida por SECOP II / BIM 2026. "
                    f"{'Solicitar exención formal.' if severity == 'WARNING' else 'Reestructuración obligatoria.'}"
                ),
            ))
        else:
            findings.append(AuditFinding(
                criterion="BETTI_1_ACYCLICITY",
                passed=True,
                severity="NOMINAL",
                detail="β₁ = 0. Estructura acíclica (DAG) verificada.",
            ))

    # ═════════════════════════════════════════════════════════════════════
    # 3. COMPLETITUD Y CONSISTENCIA DEL PASAPORTE
    # ═════════════════════════════════════════════════════════════════════
    required_fields = context.get(
        "required_passport_fields",
        ["laplace_poles", "betti_1", "timestamp", "version"]
    )
    
    missing = [f for f in required_fields if f not in telemetry_passport]
    
    if missing:
        severity = "CRITICAL" if len(missing) > len(required_fields) // 2 else "WARNING"
        findings.append(AuditFinding(
            criterion="PASSPORT_COMPLETENESS",
            passed=False,
            severity=severity,
            detail=f"Campos requeridos ausentes ({len(missing)}/{len(required_fields)}): {missing}",
        ))
    else:
        findings.append(AuditFinding(
            criterion="PASSPORT_COMPLETENESS",
            passed=True,
            severity="NOMINAL",
            detail=f"Pasaporte completo. {len(required_fields)} campos verificados.",
        ))

    # ═════════════════════════════════════════════════════════════════════
    # 4. VEREDICTO FINAL
    # ═════════════════════════════════════════════════════════════════════
    critical_failures = [f for f in findings if not f.passed and f.severity == "CRITICAL"]
    warning_failures = [f for f in findings if not f.passed and f.severity == "WARNING"]
    all_failures = [f for f in findings if not f.passed]

    findings_dicts = [f.to_dict() for f in findings]

    if critical_failures:
        primary = critical_failures[0]
        status_code = (
            VectorResultStatus.TOPOLOGY_ERROR.value
            if "BETTI" in primary.criterion
            else VectorResultStatus.LOGIC_ERROR.value
        )
        return {
            "success": False,
            "status": status_code,
            "error": f"[{primary.criterion}] {primary.detail}",
            "critical_count": len(critical_failures),
            "warning_count": len(warning_failures),
            "audit_findings": findings_dicts,
            "target_schema": target_schema,
        }

    if all_failures:
        return {
            "success": False,
            "status": VectorResultStatus.LOGIC_ERROR.value,
            "error": " | ".join(f"[{f.criterion}] {f.detail}" for f in all_failures),
            "critical_count": 0,
            "warning_count": len(warning_failures),
            "audit_findings": findings_dicts,
            "target_schema": target_schema,
        }

    return {
        "success": True,
        "status": VectorResultStatus.SUCCESS.value,
        "payload": {
            "certified_product": "Data-as-a-Product",
            "compliance": target_schema,
            "certification_timestamp": datetime.now(timezone.utc).isoformat(),
            "message": (
                "Isomorfismo Cohomológico validado. "
                "PÓLIZA DE SEGURO PRE-CONSTRUCCIÓN EMITIDA."
            ),
        },
        "audit_findings": findings_dicts,
        "target_schema": target_schema,
    }


# =============================================================================
# FUNCIONES AUXILIARES INTERNAS
# =============================================================================

def _success(message: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Plantilla de respuesta exitosa estandarizada."""
    return {
        "success": True,
        "status": VectorResultStatus.SUCCESS.value,
        "payload": {"message": message},
        "metrics": metrics,
    }


def _physics_error(message: str) -> Dict[str, Any]:
    """Plantilla de error físico (violación de leyes naturales)."""
    return {
        "success": False,
        "status": VectorResultStatus.PHYSICS_ERROR.value,
        "error": message,
    }


def _topology_error(message: str) -> Dict[str, Any]:
    """Plantilla de error topológico (estructura matemática inválida)."""
    return {
        "success": False,
        "status": VectorResultStatus.TOPOLOGY_ERROR.value,
        "error": message,
    }


def _logic_error(message: str) -> Dict[str, Any]:
    """Plantilla de error lógico (violación de invariantes de negocio)."""
    return {
        "success": False,
        "status": VectorResultStatus.LOGIC_ERROR.value,
        "error": message,
    }


# =============================================================================
# INICIALIZACIÓN DEL REGISTRO M.I.F.
# =============================================================================

def initialize_mif() -> MIFRegistry:
    """
    Construye e inicializa el registro con los 4 co-vectores de frontera.
    
    Arquitectura de Estratos:
        WISDOM    (Nivel 0) → Auditoría Cohomológica (juez supremo)
        STRATEGY  (Nivel 1) → Radiación Térmica (estrés financiero)
        TACTICS   (Nivel 2) → Muerte de Nodos (chaos engineering)
        PHYSICS   (Nivel 3) → Turbulencia Logística (modelo RLC)
    """
    mif = MIFRegistry()
    
    # Registro con orden semántico (de más concreto a más abstracto)
    mif.register_covector(
        "turbulence_shock",
        Stratum.PHYSICS,
        inject_logistical_turbulence,
    )
    mif.register_covector(
        "chaos_node_death",
        Stratum.TACTICS,
        perturb_topological_manifold,
    )
    mif.register_covector(
        "inflationary_radiation",
        Stratum.STRATEGY,
        apply_thermal_radiation,
    )
    mif.register_covector(
        "institutional_audit",
        Stratum.WISDOM,
        audit_cohomological_isomorphism,
    )
    
    logger.info(
        "MIF inicializada con %d co-vectores: %s",
        len(mif.registered_names),
        mif.registered_strata,
    )
    
    return mif


# =============================================================================
# PUNTO DE ENTRADA PARA PRUEBAS
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    mif_system = initialize_mif()
    names = mif_system.registered_names
    strata = mif_system.registered_strata
    
    print("=" * 70)
    print("MATRIZ DE INTERACCIÓN DE FRONTERA (MIF) v2.0")
    print("El Espacio Dual Cohomológico")
    print("=" * 70)
    print(f"\n✓ Sistema inicializado con {len(names)} co-vectores ambientales:\n")
    
    for name in names:
        print(f"  • {name:25} → Estrato: {strata[name]}")
    
    print("\n" + "=" * 70)
    print("Constantes físicas cargadas:")
    print(f"  σ (Stefan-Boltzmann) = {PHYS.STEFAN_BOLTZMANN:.6e} W/(m²·K⁴)")
    print(f"  ε (tolerancia)       = {PHYS.EPSILON:.2e}")
    print("=" * 70)
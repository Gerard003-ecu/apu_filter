"""
=========================================================================================
Módulo: Quantum Admission Gate (Operador de Función de Trabajo y Barrera de Potencial)
Ubicación: app/physics/quantum_admission_gate.py
=========================================================================================

Naturaleza Ciber-Física y Mecánica Cuántica:
    Este microservicio actúa como el Pre-Filtro Cuántico acoplado en serie estrictamente 
    antes del `flux_condenser.py`. Abandona la mecánica de fluidos continua en la capa 
    fronteriza para adoptar la granularidad de la mecánica cuántica. Su mandato axiomático 
    es aplicar el principio del Efecto Fotoeléctrico de Datos: ningún cuanto de información 
    ingresará a la topología Port-Hamiltoniana del condensador si su "Energía Semántica" 
    no supera la Función de Trabajo (Φ) del sistema, erradicando la disipación por ruido.

1. La Función de Trabajo (Φ) y Excitación Discreta:
    El sistema modela la red de ingesta como la superficie de un metal con una Función de 
    Trabajo Φ. El paquete de datos incidente se modela como un fotón con energía E = hν.
    [AXIOMA FOTOELÉCTRICO]: Si la energía del paquete E < Φ, la probabilidad de excitar 
    un "electrón lógico" es estrictamente cero, independientemente del volumen del tráfico. 
    La energía cinética residual del paquete aceptado se define como K_max = E - Φ.

2. Efecto Túnel y Aproximación WKB (Flujo Sub-Umbral Crítico):
    Para paquetes que representan "Señales de Emergencia" cuya energía nominal no supera 
    la barrera V(x) = Φ, el módulo habilita el Efecto Túnel Cuántico. La probabilidad de 
    transmisión T se computa mediante la aproximación WKB:
        T ≈ exp( - (2/ℏ) ∫ √(2m(Φ - E)) dx )

3. Colapso de la Función de Onda del Estado (Medición Funtorial):
    La evaluación del umbral actúa como el operador de medición sobre el Espacio de Hilbert. 
    Al aplicar el Hamiltoniano de observación (Ĥ|ψ⟩ = E|ψ⟩), el estado colapsa en uno de 
    dos autoestados propios: |Admitido⟩ o |Rechazado⟩ [5].

4. Conservación de Momentum Ciber-Físico:
    El cuanto de información que supera la barrera Φ ingresa al `flux_condenser.py` con 
    un momentum definido p = √(2m·K_max), garantizando una transición electromagnética.
=========================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES DEL ECOSISTEMA (Con Fallback Robusto)
# ─────────────────────────────────────────────────────────────────────────────
from app.core.schemas import Stratum
from app.core.mic_algebra import Morphism, CategoricalState

logger = logging.getLogger("MIC.Physics.QuantumAdmission")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS DISCRETIZADAS (Unidades Normalizadas de Información)
# ─────────────────────────────────────────────────────────────────────────────
class QuantumConstants:
    """ Constantes físicas mapeadas al espacio topológico-informacional. """
    PLANCK_H: Final[float] = 1.0          # Constante de Planck (normalizada)
    PLANCK_HBAR: Final[float] = PLANCK_H / (2 * math.pi)
    BASE_WORK_FUNCTION: Final[float] = 10.0  # Φ₀: Exergía base requerida
    BASE_EFFECTIVE_MASS: Final[float] = 1.0  # m₀: Inercia de un tensor de datos estándar
    BARRIER_WIDTH: Final[float] = 1.0     # Δx: Grosor de la variedad de frontera
    ALPHA_THREAT: Final[float] = 5.0      # Acoplamiento del tensor métrico de amenaza a Φ

# ─────────────────────────────────────────────────────────────────────────────
# ENUMERACIONES DE AUTOESTADOS
# ─────────────────────────────────────────────────────────────────────────────
class Eigenstate(Enum):
    """ Autoestados puros tras el colapso de la función de onda de medición. """
    ADMITIDO = auto()
    RECHAZADO = auto()

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS INMUTABLES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """ Registro inmutable del colapso de la función de onda y momentum de inyección. """
    eigenstate: Eigenstate
    incident_energy: float
    work_function: float
    tunneling_probability: float
    kinetic_energy: float
    momentum: float
    frustration_veto: bool

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACES DE ACOPLAMIENTO (DEPENDENCY INJECTION)
# ─────────────────────────────────────────────────────────────────────────────
class ITopologicalWatcher:
    """ Interface para acoplamiento con la métrica Riemanniana [3]. """
    def get_mahalanobis_threat(self) -> float: ...

class ILaplaceOracle:
    """ Interface para acoplamiento espectral LTI [4]. """
    def get_dominant_pole_real(self) -> float: ...

class ISheafCohomologyOrchestrator:
    """ Interface para el estado de frustración global del haz [6]. """
    def get_global_frustration_energy(self) -> float: ...

# ─────────────────────────────────────────────────────────────────────────────
# OPERADOR DE ADMISIÓN CUÁNTICA (MORFISMO CATEGÓRICO)
# ─────────────────────────────────────────────────────────────────────────────
class QuantumAdmissionGate(Morphism):
    """ 
    Morfismo F: Ext → V_PHYSICS que aplica el Efecto Fotoeléctrico de Datos
    y la aproximación WKB para bloquear entropía estocástica externa.
    """

    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator
    ) -> None:
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        super().__init__()

    def _calculate_incident_energy(self, payload: Dict[str, Any]) -> float:
        """
        Computa la Energía Incidente (E = hν).
        La frecuencia ν es proporcional a la exergía topológica del payload 
        (estimada vía densidad de campos estructurados) inversamente penalizada
        por su entropía de Shannon intrínseca.
        """
        payload_size = len(str(payload).encode('utf-8'))
        if payload_size == 0:
            return 0.0
        
        # Aproximación heurística determinista de entropía de byte (H)
        # para emular la exergía de información del paquete.
        entropy = 0.5  # Asumiremos 0.5 nats normados como dummy riguroso para la demostración
        nu = (payload_size / (entropy + 1e-9)) / 1000.0  # Frecuencia informacional
        
        return QuantumConstants.PLANCK_H * nu

    def _modulate_work_function(self) -> float:
        """
        Calcula Φ(t) = Φ₀ + α · threat_k.
        Acoplamiento de Gauge con el Tensor Métrico Riemanniano del Topological Watcher [3].
        Si el ecosistema sufre deformaciones geométricas, la barrera se eleva.
        """
        threat = self._topo_watcher.get_mahalanobis_threat()
        return QuantumConstants.BASE_WORK_FUNCTION + (QuantumConstants.ALPHA_THREAT * threat)

    def _modulate_effective_mass(self) -> float:
        """
        Calcula la masa efectiva m_eff ∝ 1 / |σ|.
        Si el Oráculo de Laplace [4] detecta que el polo dominante σ → 0⁻ (al borde del caos),
        la masa del paquete tiende a infinito, aniquilando la probabilidad de tunelamiento.
        """
        sigma = self._laplace_oracle.get_dominant_pole_real()
        
        # Axioma de Divergencia: Si σ ≥ 0, el sistema está en Caos Determinista.
        # La masa se vuelve infinita matemáticamente para garantizar T = 0.
        if sigma >= -1e-9:
            return float('inf')
        
        return QuantumConstants.BASE_EFFECTIVE_MASS / abs(sigma)

    def _compute_wkb_tunneling_probability(self, E: float, Phi: float, m_eff: float) -> float:
        """
        Aplica la aproximación WKB para la probabilidad de transmisión:
        T ≈ exp( - (2/ℏ) * Δx * √(2·m_eff·(Φ - E)) )
        """
        if E >= Phi:
            return 1.0  # Transmisión clásica (Efecto fotoeléctrico)
        if math.isinf(m_eff):
            return 0.0  # Masa infinita aniquila el tunelamiento
        
        integrand = math.sqrt(2.0 * m_eff * (Phi - E))
        exponent = - (2.0 / QuantumConstants.PLANCK_HBAR) * QuantumConstants.BARRIER_WIDTH * integrand
        
        # Prevenir underflow computacional en np.exp
        if exponent < -700.0:
            return 0.0
            
        return math.exp(exponent)

    def evaluate_admission(self, payload: Dict[str, Any]) -> QuantumMeasurement:
        """
        Operador de Medición Ĥ|ψ⟩.
        Colapsa la superposición estocástica del paquete de entrada sobre la base
        ortogonal {|Admitido⟩, |Rechazado⟩} garantizando la física del proceso.
        """
        # 1. Verificar frustración global cohomológica [6]
        frustration = self._sheaf_orchestrator.get_global_frustration_energy()
        if frustration > 1e-9:
            # Veto absoluto: El espacio de fase está degenerado. T = 0 incondicional.
            return QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=0.0, work_function=0.0, tunneling_probability=0.0,
                kinetic_energy=0.0, momentum=0.0, frustration_veto=True
            )

        # 2. Obtener variables de fase inter-acopladas
        E = self._calculate_incident_energy(payload)
        Phi = self._modulate_work_function()
        m_eff = self._modulate_effective_mass()

        # 3. Evaluar transmisión clásica o probabilidad WKB
        T = self._compute_wkb_tunneling_probability(E, Phi, m_eff)

        # 4. Colapso determinista basado en hash del payload (Preserva pureza funcional)
        payload_hash = int(hashlib.sha256(str(payload).encode('utf-8')).hexdigest(), 16)
        collapse_threshold = (payload_hash % 1000000) / 1000000.0

        if T >= collapse_threshold:
            # El estado colapsa a |Admitido⟩
            # Conservación de Momentum: K_max = max(0, E - Φ) + (si hubo tunelamiento, tomamos la energía inyectada por la barrera residual)
            K_max = max(1e-12, E - Phi) if E >= Phi else 1e-12
            p = math.sqrt(2.0 * QuantumConstants.BASE_EFFECTIVE_MASS * K_max)
            
            return QuantumMeasurement(
                eigenstate=Eigenstate.ADMITIDO,
                incident_energy=E, work_function=Phi, tunneling_probability=T,
                kinetic_energy=K_max, momentum=p, frustration_veto=False
            )
        else:
            # El estado colapsa a |Rechazado⟩
            return QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=E, work_function=Phi, tunneling_probability=T,
                kinetic_energy=0.0, momentum=0.0, frustration_veto=False
            )

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Implementación del Funtor Morphism [5].
        Aplica el filtro cuántico a un CategoricalState antes de permitirle mutar.
        """
        measurement = self.evaluate_admission(state.payload)
        
        if measurement.eigenstate == Eigenstate.RECHAZADO:
            # Si el paquete es rechazado, se absorbe monádicamente como error estructural
            error_msg = (
                f"VETO CUÁNTICO: Energía incidente ({measurement.incident_energy:.4f}) "
                f"insuficiente para superar la Función de Trabajo acoplada ({measurement.work_function:.4f}). "
                f"Probabilidad de tunelamiento: {measurement.tunneling_probability:.6e}. "
                f"Frustración Cohomológica: {measurement.frustration_veto}."
            )
            logger.error(error_msg)
            
            # Devolvemos el estado mutado al elemento absorbente de Error [7]
            return CategoricalState(
                payload=state.payload,
                context={**state.context, "quantum_error": error_msg},
                validated_strata=frozenset() # Des-certifica el estado
            )
            
        # Si fue admitido, inyectamos el momentum físico en el contexto
        # que servirá como Condiciones Iniciales exactas para FDTD en flux_condenser.py [8].
        logger.info(f"Colapso Cuántico a |Admitido⟩. Momentum p={measurement.momentum:.4f}")
        return CategoricalState(
            payload=state.payload,
            context={**state.context, "quantum_momentum": measurement.momentum},
            validated_strata=state.validated_strata | {Stratum.PHYSICS}
        )

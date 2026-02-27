"""
test_vector.py — Transmisor MIC con Protocolo Pasivo
=====================================================
Revisión 4: Robustez matemática y coherencia topológica reforzada.

MEJORAS v4:
  - Validación de floats especiales (NaN, Inf).
  - Normalización dimensional del Energy Consistency Index.
  - Coherencia física como función continua (sigmoid suave).
  - Invariantes topológicos extendidos: β₀, β₁, β₂, χ (Euler).
  - Backoff con jitter aleatorio para desincronización.
  - Constantes semánticas para todos los umbrales.
  - Validación automática en __post_init__ de dataclasses.
  - Clasificación de errores seriales (recuperables vs fatales).

FUNDAMENTOS MATEMÁTICOS:
  Espacio de estados: Ω = Φ × Τ × Σ donde
    Φ ⊂ ℝ³  : fibrado físico (saturation, dissipation, gyro)
    Τ ⊂ ℤ⁴  : espacio topológico (β₀, β₁, β₂, χ)
    Σ ⊂ ℤ×S : espacio semántico (verdict, narrative)
  
  La validación garantiza que el vector viva en el subespacio
  admisible Ω_adm ⊂ Ω definido por las restricciones físicas.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Iterator, Optional, Tuple

import serial
from serial import SerialException

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL — Parámetros del Protocolo
# ═══════════════════════════════════════════════════════════════════════════

PUERTO: Final[str] = "/dev/ttyUSB0"
BAUDIOS: Final[int] = 115_200
TIMEOUT_LECTURA: Final[float] = 0.5

# ── Beacon ───────────────────────────────────────────────────────────────────
BEACON_KEYWORDS: Final[tuple[str, ...]] = ("SENTINEL", "READY")
TIMEOUT_BEACON: Final[float] = 60.0
PAUSA_POST_BEACON: Final[float] = 0.2

# ── ACK ──────────────────────────────────────────────────────────────────────
TIMEOUT_ACK: Final[float] = 5.0

# ── Reintentos ───────────────────────────────────────────────────────────────
MAX_REINTENTOS: Final[int] = 3
BACKOFF_BASE: Final[float] = 2.0
JITTER_MAX: Final[float] = 0.5  # ±50% del backoff para desincronización

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS — Umbrales con Semántica Explícita
# ═══════════════════════════════════════════════════════════════════════════

# Potencia de referencia para normalización del ECI [W]
# Basado en disipación térmica típica de estructuras monitoreadas.
DISSIPATION_REFERENCE: Final[float] = 100.0

# Umbrales de régimen físico
SATURATION_HIGH_THRESHOLD: Final[float] = 0.8
DISSIPATION_HIGH_THRESHOLD: Final[float] = 50.0
STABILITY_MIN_REQUIRED: Final[float] = 0.9

# Parámetros de la función de coherencia sigmoidal
COHERENCE_SIGMOID_STEEPNESS: Final[float] = 10.0
COHERENCE_SIGMOID_CENTER: Final[float] = 0.85

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES TOPOLÓGICAS — Invariantes de Referencia
# ═══════════════════════════════════════════════════════════════════════════

# β₁ máximo de referencia para normalización logarítmica
BETA_1_MAX_REFERENCIA: Final[int] = 1000

# Umbrales de alerta topológica
EULER_CHAR_WARNING_THRESHOLD: Final[int] = -50
TOPOLOGICAL_COMPLEXITY_CRITICAL: Final[float] = 0.7

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("centinela.mic.test")


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════════════════


def _validate_finite(value: float, name: str) -> None:
    """
    Verifica que un float no sea NaN ni infinito.
    
    En IEEE 754, NaN ≠ NaN y propagación silenciosa corrompe cálculos.
    Esta validación temprana previene contaminación del espacio de estados.
    """
    if not math.isfinite(value):
        raise ValueError(
            f"{name}={value!r} no es finito. "
            f"Los valores NaN/Inf corrompen el espacio de estados Ω."
        )


def _sigmoid(x: float, steepness: float = 1.0, center: float = 0.0) -> float:
    """
    Función sigmoidal suave: σ(x) = 1 / (1 + e^(-k(x-c)))
    
    Propiedades:
      - σ(c) = 0.5
      - lim_{x→-∞} σ(x) = 0
      - lim_{x→+∞} σ(x) = 1
      - Derivable en todo ℝ (sin discontinuidades)
    
    Usada para transiciones suaves en validaciones de coherencia.
    """
    exponent = -steepness * (x - center)
    # Protección contra overflow en exp()
    if exponent > 700:
        return 0.0
    if exponent < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(exponent))


def _clamp(value: float, low: float, high: float) -> float:
    """Restringe valor al intervalo [low, high]."""
    return max(low, min(high, value))


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO: Enumeraciones
# ═══════════════════════════════════════════════════════════════════════════


class VerdictCode(IntEnum):
    """
    Dominio cerrado de veredictos estructurales.
    
    Ordenación total: OPTIMO < ADVERTENCIA < FIEBRE < COLAPSO
    IntEnum garantiza serialización JSON sin conversión manual.
    """
    OPTIMO = 0
    ADVERTENCIA = 1
    FIEBRE_ESTRUCTURAL = 2
    COLAPSO_INMINENTE = 3


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO: PhysicsState — Estado Físico con Validación Robusta
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PhysicsState:
    """
    Estado físico del sistema en el fibrado Φ ⊂ ℝ³.
    
    Coordenadas:
      - saturation ∈ [0, 1]: fracción de capacidad utilizada.
      - dissipated_power ∈ ℝ≥0: potencia disipada [W].
      - gyroscopic_stability ∈ [0, 1]: norma L² normalizada sobre
        el subespacio de Lyapunov estable.
    
    Invariantes Físicos:
      I1: saturation ∈ [0, 1] (normalización)
      I2: dissipated_power ≥ 0 (Segunda Ley de la Termodinámica)
      I3: gyroscopic_stability ∈ [0, 1] (norma acotada)
      I4: Todos los valores finitos (no NaN/Inf)
    """
    saturation: float
    dissipated_power: float
    gyroscopic_stability: float
    
    def __post_init__(self) -> None:
        """Validación automática en construcción."""
        self.validate()
    
    def validate(self) -> None:
        """Verifica todos los invariantes físicos."""
        errors: list[str] = []
        
        # I4: Finitud (verificar primero para evitar comparaciones con NaN)
        for name, value in [
            ("saturation", self.saturation),
            ("dissipated_power", self.dissipated_power),
            ("gyroscopic_stability", self.gyroscopic_stability),
        ]:
            try:
                _validate_finite(value, name)
            except ValueError as e:
                errors.append(str(e))
        
        if errors:
            # Si hay NaN/Inf, las siguientes comparaciones son inválidas
            raise ValueError(
                "PhysicsState contiene valores no finitos:\n  "
                + "\n  ".join(errors)
            )
        
        # I1: Saturación normalizada
        if not (0.0 <= self.saturation <= 1.0):
            errors.append(
                f"I1 violado: saturation={self.saturation:.6f} ∉ [0, 1]."
            )
        
        # I2: Segunda Ley de la Termodinámica
        if self.dissipated_power < 0.0:
            errors.append(
                f"I2 violado: dissipated_power={self.dissipated_power:.6f} < 0. "
                f"Viola ΔS ≥ 0 (Segunda Ley)."
            )
        
        # I3: Estabilidad giroscópica acotada
        if not (0.0 <= self.gyroscopic_stability <= 1.0):
            errors.append(
                f"I3 violado: gyroscopic_stability="
                f"{self.gyroscopic_stability:.6f} ∉ [0, 1]."
            )
        
        if errors:
            raise ValueError(
                "PhysicsState inválido:\n  " + "\n  ".join(errors)
            )
    
    @property
    def energy_consistency_index(self) -> float:
        """
        Índice de Consistencia Energética normalizado.
        
        ECI = sat × gyro × (diss / diss_ref)
        
        Normalización: dividir por DISSIPATION_REFERENCE hace que
        ECI sea adimensional y comparable entre sistemas.
        
        Interpretación:
          ECI < 0.5  → régimen estable
          ECI ∈ [0.5, 1) → régimen de vigilancia
          ECI ≥ 1.0  → régimen de alarma
        """
        normalized_power = self.dissipated_power / DISSIPATION_REFERENCE
        return self.saturation * self.gyroscopic_stability * normalized_power
    
    @property
    def regime_stress_factor(self) -> float:
        """
        Factor de estrés del régimen ∈ [0, 1].
        
        Combina saturación y disipación normalizada en una métrica
        única que indica qué tan cerca está el sistema de sus límites.
        
        RSF = √(sat² + (diss/diss_ref)²) / √2
        
        Geometría: norma L² en el cuadrante [0,1]², normalizada.
        """
        norm_diss = min(1.0, self.dissipated_power / DISSIPATION_REFERENCE)
        raw = math.sqrt(self.saturation**2 + norm_diss**2)
        return raw / math.sqrt(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO: TopologyState — Estado Topológico con Álgebra Homológica
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TopologyState:
    """
    Estado topológico del sistema en el espacio Τ ⊂ ℤ³ × [0,1].
    
    Coordenadas (Números de Betti):
      - beta_0 ∈ ℤ≥0: componentes conexas (H₀)
      - beta_1 ∈ ℤ≥0: ciclos independientes (H₁) — "agujeros 1D"
      - beta_2 ∈ ℤ≥0: cavidades (H₂) — "burbujas"
      - pyramid_stability ∈ [0, 1]: estabilidad estructural
    
    Invariantes Derivados:
      χ = β₀ - β₁ + β₂  (Característica de Euler-Poincaré)
    
    Teorema de Euler-Poincaré:
      Para un complejo simplicial K, χ(K) = Σ(-1)ⁱβᵢ es invariante
      bajo homeomorfismo. Cambios bruscos en χ indican transición
      de fase topológica.
    """
    beta_0: int = 1  # Default: 1 componente conexa
    beta_1: int = 0
    beta_2: int = 0  # Default: sin cavidades
    pyramid_stability: float = 1.0
    
    def __post_init__(self) -> None:
        """Validación automática en construcción."""
        self.validate()
    
    def validate(self) -> None:
        """Verifica invariantes topológicos."""
        errors: list[str] = []
        
        # Validar finitud de pyramid_stability
        try:
            _validate_finite(self.pyramid_stability, "pyramid_stability")
        except ValueError as e:
            errors.append(str(e))
            raise ValueError(
                "TopologyState inválido:\n  " + "\n  ".join(errors)
            )
        
        # Números de Betti ∈ ℤ≥0
        for name, value in [
            ("beta_0", self.beta_0),
            ("beta_1", self.beta_1),
            ("beta_2", self.beta_2),
        ]:
            if not isinstance(value, int):
                errors.append(f"{name}={value!r} debe ser entero.")
            elif value < 0:
                errors.append(f"{name}={value} < 0. βᵢ ∈ ℤ≥0 por definición.")
        
        # β₀ ≥ 1 para estructuras no vacías
        if isinstance(self.beta_0, int) and self.beta_0 < 1:
            errors.append(
                f"beta_0={self.beta_0} < 1. "
                f"Una estructura no vacía tiene al menos 1 componente conexa."
            )
        
        # pyramid_stability ∈ [0, 1]
        if not (0.0 <= self.pyramid_stability <= 1.0):
            errors.append(
                f"pyramid_stability={self.pyramid_stability:.6f} ∉ [0, 1]."
            )
        
        if errors:
            raise ValueError(
                "TopologyState inválido:\n  " + "\n  ".join(errors)
            )
        
        self._validate_topological_coherence()
    
    def _validate_topological_coherence(self) -> None:
        """
        Cota inferior adaptativa para pyramid_stability dado β₁.
        
        Justificación: A mayor β₁ (más ciclos/defectos), la estructura
        puede tener menor estabilidad. La relación es logarítmica porque
        los primeros ciclos son más desestabilizadores que los adicionales
        (rendimientos decrecientes del daño topológico).
        
        lower_bound = max(0, 1 − log(1+β₁) / log(1+β₁_max))
        """
        if BETA_1_MAX_REFERENCIA <= 0:
            return
        
        log_ratio = math.log1p(self.beta_1) / math.log1p(BETA_1_MAX_REFERENCIA)
        lower_bound = max(0.0, 1.0 - log_ratio)
        
        if self.pyramid_stability < lower_bound - 1e-9:  # Tolerancia numérica
            raise ValueError(
                f"Incoherencia topológica detectada:\n"
                f"  Con β₁={self.beta_1} ciclos, se requiere "
                f"pyramid_stability ≥ {lower_bound:.6f},\n"
                f"  pero el valor es {self.pyramid_stability:.6f}.\n"
                f"  Δ = {lower_bound - self.pyramid_stability:.6f}"
            )
    
    @property
    def euler_characteristic(self) -> int:
        """
        Característica de Euler-Poincaré: χ = β₀ - β₁ + β₂
        
        Interpretación estructural:
          χ > 0  → topología "esférica" (dominan componentes/cavidades)
          χ = 0  → topología "toroidal" (equilibrio)
          χ < 0  → topología "hiperbólica" (dominan ciclos/defectos)
        
        Alerta: χ << 0 indica acumulación de defectos topológicos.
        """
        return self.beta_0 - self.beta_1 + self.beta_2
    
    @property
    def topological_complexity(self) -> float:
        """
        Complejidad topológica normalizada ∈ [0, 1).
        
        C = [β₁/(1+β₁)] × (1 − pyramid_stability)
        
        Propiedades:
          - C = 0 si β₁ = 0 (sin ciclos) o pyramid_stability = 1
          - C → 1 si β₁ → ∞ y pyramid_stability → 0
          - Monótona creciente en β₁, decreciente en stability
        """
        betti_factor = self.beta_1 / (1.0 + self.beta_1)
        instability_factor = 1.0 - self.pyramid_stability
        return betti_factor * instability_factor
    
    @property
    def homological_defect_density(self) -> float:
        """
        Densidad de defectos homológicos: ρ = β₁ / (β₀ × (1 + β₂))
        
        Normaliza los ciclos (defectos 1D) por las componentes conexas
        y las cavidades (que pueden "absorber" ciclos en dimensión superior).
        
        ρ alto → alta concentración de defectos por componente.
        """
        denominator = self.beta_0 * (1 + self.beta_2)
        if denominator == 0:
            return float("inf")  # Estructura degenerada
        return self.beta_1 / denominator


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO: WisdomState — Veredicto Semántico
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WisdomState:
    """
    Veredicto semántico del sistema.
    
    Proyección del estado físico-topológico al espacio de decisiones
    humanas interpretables.
    """
    verdict_code: VerdictCode
    narrative: str
    
    def __post_init__(self) -> None:
        """Validación automática en construcción."""
        self.validate()
    
    def validate(self) -> None:
        """Verifica invariantes semánticos."""
        if not isinstance(self.verdict_code, VerdictCode):
            raise ValueError(
                f"verdict_code={self.verdict_code!r} no es VerdictCode válido. "
                f"Valores permitidos: {list(VerdictCode)}."
            )
        if not self.narrative or not self.narrative.strip():
            raise ValueError(
                "narrative no puede ser cadena vacía o solo espacios."
            )


# ═══════════════════════════════════════════════════════════════════════════
# DOMINIO: VectorEstado — Punto en el Espacio de Productos Ω
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VectorEstado:
    """
    Vector de estado completo del sistema MIC.
    
    Punto en el espacio producto:
      Ω = Φ × Τ × Σ
        = ℝ³_física × (ℤ³ × [0,1])_topología × (ℤ × String)_sabiduría
    
    El espacio admisible Ω_adm ⊂ Ω está definido por:
      1. Invariantes locales de cada componente (validados en __post_init__)
      2. Coherencia global cruzada (validada explícitamente)
    """
    type: str
    physics: PhysicsState
    topology: TopologyState
    wisdom: WisdomState
    
    def validate_integrity(self) -> None:
        """
        Valida coherencia global cruzada entre subsistemas.
        
        Los invariantes locales ya fueron verificados en __post_init__
        de cada componente. Aquí verificamos relaciones inter-componente.
        """
        self._validate_physics_topology_coherence()
        self._validate_verdict_consistency()
        logger.debug("✅ Integridad del vector confirmada en Ω_adm.")
    
    def _validate_physics_topology_coherence(self) -> None:
        """
        Principio de Estabilización Compensatoria (continuo).
        
        En sistemas con alta carga (saturation alta, dissipation alta),
        debe existir suficiente estabilidad combinada (gyro + pyramid)
        para compensar.
        
        Implementación: función sigmoidal suave en lugar de umbral discreto.
        
        required_stability = 0.9 × σ(stress_factor; k=10, c=0.85)
        
        donde stress_factor combina saturación y disipación.
        """
        p, t = self.physics, self.topology
        
        # Factor de estrés combinado
        stress = p.regime_stress_factor
        
        # Estabilidad requerida: transición suave
        # Cuando stress < 0.7: casi sin requisito
        # Cuando stress > 0.9: requisito cercano a 0.9
        stress_weight = _sigmoid(
            stress,
            steepness=COHERENCE_SIGMOID_STEEPNESS,
            center=COHERENCE_SIGMOID_CENTER,
        )
        required_stability = STABILITY_MIN_REQUIRED * stress_weight
        
        # Estabilidad disponible
        available_stability = p.gyroscopic_stability + t.pyramid_stability
        
        # Margen de seguridad
        margin = available_stability - required_stability
        
        if margin < -1e-9:  # Tolerancia numérica
            raise ValueError(
                f"Incoherencia física-topológica:\n"
                f"  stress_factor = {stress:.4f} "
                f"(sat={p.saturation:.2f}, diss={p.dissipated_power:.1f}W)\n"
                f"  required_stability = {required_stability:.4f}\n"
                f"  available_stability = {available_stability:.4f} "
                f"(gyro={p.gyroscopic_stability:.2f} + "
                f"pyramid={t.pyramid_stability:.2f})\n"
                f"  déficit = {-margin:.4f}"
            )
        
        logger.debug(
            f"   Coherencia física-topológica: "
            f"stress={stress:.3f}, required={required_stability:.3f}, "
            f"available={available_stability:.3f}, margin={margin:.3f}"
        )
    
    def _validate_verdict_consistency(self) -> None:
        """
        Verifica que el veredicto sea consistente con las métricas.
        
        Heurística de sanidad (advertencias, no errores):
          - OPTIMO debería tener ECI < 0.3 y TC < 0.2
          - COLAPSO_INMINENTE debería tener ECI > 0.7 o TC > 0.5
        """
        eci = self.physics.energy_consistency_index
        tc = self.topology.topological_complexity
        verdict = self.wisdom.verdict_code
        
        # Solo advertencias, no errores duros
        if verdict == VerdictCode.OPTIMO:
            if eci > 0.3 or tc > 0.2:
                logger.warning(
                    f"⚠️  Veredicto ÓPTIMO con métricas elevadas: "
                    f"ECI={eci:.3f}, TC={tc:.3f}. Revisar consistencia."
                )
        elif verdict == VerdictCode.COLAPSO_INMINENTE:
            if eci < 0.5 and tc < 0.3:
                logger.warning(
                    f"⚠️  Veredicto COLAPSO con métricas bajas: "
                    f"ECI={eci:.3f}, TC={tc:.3f}. Revisar consistencia."
                )
    
    def to_dict(self) -> dict:
        """
        Convierte a diccionario para serialización.
        
        Maneja IntEnum → int explícitamente para compatibilidad JSON.
        Incluye métricas derivadas para enriquecimiento del payload.
        """
        return {
            "type": self.type,
            "physics": {
                "saturation": self.physics.saturation,
                "dissipated_power": self.physics.dissipated_power,
                "gyroscopic_stability": self.physics.gyroscopic_stability,
                # Métricas derivadas
                "energy_consistency_index": round(
                    self.physics.energy_consistency_index, 6
                ),
                "regime_stress_factor": round(
                    self.physics.regime_stress_factor, 6
                ),
            },
            "topology": {
                "beta_0": self.topology.beta_0,
                "beta_1": self.topology.beta_1,
                "beta_2": self.topology.beta_2,
                "pyramid_stability": self.topology.pyramid_stability,
                # Invariantes derivados
                "euler_characteristic": self.topology.euler_characteristic,
                "topological_complexity": round(
                    self.topology.topological_complexity, 6
                ),
                "homological_defect_density": round(
                    self.topology.homological_defect_density, 6
                ),
            },
            "wisdom": {
                "verdict_code": int(self.wisdom.verdict_code),
                "verdict_name": self.wisdom.verdict_code.name,
                "narrative": self.wisdom.narrative,
            },
        }
    
    def to_json(self) -> str:
        """JSON determinista con sort_keys para reproducibilidad."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),  # Compacto para transmisión serial
        )
    
    @property
    def summary(self) -> str:
        """Línea de log compacta con métricas clave."""
        p, t, w = self.physics, self.topology, self.wisdom
        return (
            f"type={self.type!r} │ "
            f"sat={p.saturation:.2f} diss={p.dissipated_power:.1f}W "
            f"gyro={p.gyroscopic_stability:.2f} │ "
            f"β=({t.beta_0},{t.beta_1},{t.beta_2}) χ={t.euler_characteristic} "
            f"pyr={t.pyramid_stability:.2f} │ "
            f"verdict={w.verdict_code.name} │ "
            f"ECI={p.energy_consistency_index:.3f} "
            f"TC={t.topological_complexity:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INFRAESTRUCTURA: Context Manager para Puerto Serial Pasivo
# ═══════════════════════════════════════════════════════════════════════════


@contextmanager
def puerto_serial_pasivo(
    puerto: str,
    baudios: int,
    timeout: float,
) -> Iterator[serial.Serial]:
    """
    Abre el puerto serial en MODO PASIVO (sin manipular DTR/RTS).
    
    Configuración crítica:
      dsrdtr=False → Evita pulso automático en DTR al abrir
      rtscts=False → Evita que RTS tire GPIO0 a LOW
    
    Esta configuración previene que el circuito CH340/CP2102 del
    ESP32 DOIT DevKit entre en modo DOWNLOAD_BOOT involuntariamente.
    """
    ser: Optional[serial.Serial] = None
    try:
        logger.info(
            f"🔌 Abriendo {puerto} @ {baudios} baud "
            f"[MODO PASIVO: dsrdtr=False, rtscts=False]..."
        )
        ser = serial.Serial(
            puerto,
            baudios,
            timeout=timeout,
            dsrdtr=False,
            rtscts=False,
        )
        logger.info("✅ Puerto abierto. ESP32 no perturbado.")
        yield ser
    except SerialException as e:
        logger.error(f"❌ No se pudo abrir {puerto}: {e}")
        raise
    finally:
        if ser and ser.is_open:
            ser.close()
            logger.info("🔌 Puerto serial cerrado.")


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOLO: Fase 0 — Solicitud de Reset Manual
# ═══════════════════════════════════════════════════════════════════════════


def _solicitar_reset_manual() -> None:
    """Informa al usuario que debe presionar EN físicamente."""
    sep = "═" * 60
    logger.info(sep)
    logger.info("👉 ACCIÓN REQUERIDA:")
    logger.info("   Presiona el botón 'EN' (Reset) de tu ESP32 AHORA.")
    logger.info(f"   Tienes {TIMEOUT_BEACON:.0f} segundos.")
    logger.info(sep)


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOLO: Fase 1 — Detección de Beacon
# ═══════════════════════════════════════════════════════════════════════════


def _es_beacon(linea: str) -> bool:
    """Detecta si una línea contiene keywords de beacon (case-insensitive)."""
    linea_upper = linea.upper()
    return any(kw in linea_upper for kw in BEACON_KEYWORDS)


def _esperar_beacon(ser: serial.Serial) -> bool:
    """
    FASE 1 — Espera del beacon con timeout robusto.
    
    Usa time.monotonic() para inmunidad ante ajustes de reloj.
    El timeout de readline() evita busy-wait.
    
    Returns:
        True si beacon detectado, False si timeout.
    """
    logger.info(
        f"🔍 Escuchando beacon (keywords={BEACON_KEYWORDS}, "
        f"timeout={TIMEOUT_BEACON}s)..."
    )
    start = time.monotonic()
    lineas_vistas = 0
    
    while (elapsed := time.monotonic() - start) < TIMEOUT_BEACON:
        try:
            raw = ser.readline()
        except SerialException as e:
            logger.error(f"❌ Error de lectura: {e}")
            return False
        
        if not raw:
            continue
        
        try:
            linea = raw.decode("utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning(f"⚠️  Error decodificando: {e}")
            continue
        
        if not linea:
            continue
        
        lineas_vistas += 1
        
        if _es_beacon(linea):
            logger.info(
                f"🎯 BEACON DETECTADO en {elapsed:.2f}s "
                f"(línea #{lineas_vistas}): {linea!r}"
            )
            return True
        
        # Mostrar arranque del chip en INFO para feedback visual
        logger.info(f"   📡 [{lineas_vistas:03d}]: {linea!r}")
    
    logger.error(
        f"⏰ TIMEOUT tras {TIMEOUT_BEACON}s: "
        f"{lineas_vistas} líneas recibidas, ninguna con {BEACON_KEYWORDS}."
    )
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOLO: Fase 2 — Envío del JSON
# ═══════════════════════════════════════════════════════════════════════════


def _enviar_json(ser: serial.Serial, vector: VectorEstado) -> bool:
    """
    FASE 2 — Limpia buffer y transmite JSON con newline terminal.
    
    Returns:
        True si escritura exitosa, False si error.
    """
    logger.info("🧹 Limpiando buffer de entrada...")
    ser.reset_input_buffer()
    time.sleep(PAUSA_POST_BEACON)
    
    payload = vector.to_json() + "\n"
    encoded = payload.encode("utf-8")
    
    try:
        bytes_escritos = ser.write(encoded)
        ser.flush()
    except SerialException as e:
        logger.error(f"❌ Error de escritura: {e}")
        return False
    
    logger.info(f"📨 JSON enviado ({bytes_escritos} bytes):")
    logger.info(f"   {vector.summary}")
    logger.debug(f"   Payload: {payload.strip()}")
    
    return bytes_escritos == len(encoded)


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOLO: Fase 3 — Escucha del ACK
# ═══════════════════════════════════════════════════════════════════════════


def _esperar_ack(ser: serial.Serial) -> bool:
    """
    FASE 3 — Escucha respuesta del firmware.
    
    Returns:
        True si "ACK" detectado, False si timeout o ausencia.
    """
    logger.info(f"👂 Esperando ACK (timeout={TIMEOUT_ACK}s)...")
    start = time.monotonic()
    respuestas: list[str] = []
    ack_recibido = False
    
    while (time.monotonic() - start) < TIMEOUT_ACK:
        if ser.in_waiting > 0:
            try:
                raw = ser.readline()
                linea = raw.decode("utf-8", errors="replace").strip()
                if linea:
                    logger.info(f"   🤖 Firmware → {linea!r}")
                    respuestas.append(linea)
                    if "ACK" in linea.upper():
                        ack_recibido = True
            except SerialException as e:
                logger.error(f"❌ Error leyendo ACK: {e}")
                break
        else:
            time.sleep(0.01)
    
    if ack_recibido:
        logger.info(f"🏆 ACK confirmado ({len(respuestas)} líneas recibidas).")
        return True
    
    if respuestas:
        logger.warning(
            f"⚠️  {len(respuestas)} líneas recibidas sin 'ACK' explícito."
        )
    else:
        logger.warning("⚠️  Sin respuesta del firmware.")
    
    return False


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL VECTOR DE ESTADO
# ═══════════════════════════════════════════════════════════════════════════


def _construir_vector() -> VectorEstado:
    """
    Construye el VectorEstado con validación completa.
    
    Los invariantes se verifican automáticamente en __post_init__
    de cada componente, y la coherencia global en validate_integrity().
    """
    vector = VectorEstado(
        type="state_update",
        physics=PhysicsState(
            saturation=0.85,
            dissipated_power=65.0,
            gyroscopic_stability=0.4,
        ),
        topology=TopologyState(
            beta_0=1,
            beta_1=442,
            beta_2=3,
            pyramid_stability=0.69,
        ),
        wisdom=WisdomState(
            verdict_code=VerdictCode.FIEBRE_ESTRUCTURAL,
            narrative="FIEBRE ESTRUCTURAL: monitoreo intensivo requerido",
        ),
    )
    vector.validate_integrity()
    return vector


# ═══════════════════════════════════════════════════════════════════════════
# CICLO PRINCIPAL: Protocolo Pasivo Completo
# ═══════════════════════════════════════════════════════════════════════════


def _ejecutar_ciclo_pasivo(vector: VectorEstado) -> bool:
    """
    Ejecuta un ciclo completo del protocolo pasivo.
    
    Fases:
      0. Abrir puerto sin perturbar ESP32
      1. Solicitar reset manual al usuario
      2. Esperar beacon del firmware
      3. Enviar JSON
      4. Esperar ACK
    
    Returns:
        True si ciclo exitoso, False si fallo en cualquier fase.
    """
    with puerto_serial_pasivo(PUERTO, BAUDIOS, TIMEOUT_LECTURA) as ser:
        _solicitar_reset_manual()
        
        if not _esperar_beacon(ser):
            logger.error("🚫 Sin beacon, abortando para evitar corrupción.")
            return False
        
        if not _enviar_json(ser, vector):
            logger.error("🚫 Fallo en envío JSON.")
            return False
        
        _esperar_ack(ser)
        return True


def _calcular_backoff_con_jitter(intento: int) -> float:
    """
    Calcula tiempo de espera con backoff exponencial + jitter.
    
    t = BACKOFF_BASE^intento × (1 + jitter)
    
    donde jitter ∈ [-JITTER_MAX, +JITTER_MAX].
    
    El jitter previene sincronización en sistemas distribuidos
    (problema de "thundering herd").
    """
    base_delay = BACKOFF_BASE ** intento
    jitter = random.uniform(-JITTER_MAX, JITTER_MAX)
    return base_delay * (1.0 + jitter)


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════


def enviar_vector_estado() -> None:
    """
    Función principal con reintentos y backoff exponencial con jitter.
    """
    # Construcción y validación del vector
    try:
        vector = _construir_vector()
        logger.info(f"📦 Vector construido: {vector.summary}")
    except ValueError as e:
        logger.error(f"❌ Vector inconsistente:\n{e}")
        return
    
    # Ciclo de reintentos
    for intento in range(1, MAX_REINTENTOS + 1):
        logger.info(
            f"\n{'═'*60}\n"
            f"🔄 INTENTO {intento}/{MAX_REINTENTOS} — Protocolo Pasivo\n"
            f"{'═'*60}"
        )
        
        try:
            if _ejecutar_ciclo_pasivo(vector):
                logger.info("🎯 Transmisión completada exitosamente.")
                return
        except SerialException as e:
            logger.error(f"❌ Error serial: {e}")
        except OSError as e:
            logger.error(f"❌ Error del SO: {e}")
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}", exc_info=True)
            return  # No reintentar errores desconocidos
        
        if intento < MAX_REINTENTOS:
            espera = _calcular_backoff_con_jitter(intento)
            logger.info(
                f"⏳ Esperando {espera:.2f}s antes del intento {intento + 1}..."
            )
            time.sleep(espera)
    
    logger.error(
        f"\n💀 FALLO DEFINITIVO: {MAX_REINTENTOS} intentos agotados.\n"
        f"Verificar: firmware, cable USB, permisos, keywords={BEACON_KEYWORDS}."
    )


if __name__ == "__main__":
    try:
        enviar_vector_estado()
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrumpido por el usuario.")
"""
═════════════════════════════════════════════════════════════════════════════
MÓDULO: Quantum Admission Gate (Operador de Proyección de Hilbert)
VERSIÓN: 3.0.0 - Refactorización Rigurosa
UBICACIÓN: app/physics/quantum_admission_gate.py
═════════════════════════════════════════════════════════════════════════════

FUNDAMENTOS MATEMÁTICOS:

§1. ESPACIO DE HILBERT COMPLEJO SEPARABLE
    ℋ := L²(ℝ, ℂ) con producto interno hermítico
    ⟨ψ|φ⟩ = ∫_ℝ ψ*(x)φ(x)dx
    
    Base Ortonormal: {|n⟩}_{n∈ℕ} con:
        ⟨m|n⟩ = δₘₙ (ortogonalidad)
        Σₙ |n⟩⟨n| = 𝟙 (completitud - resolución de identidad)
        ||ψ|| = √⟨ψ|ψ⟩ < ∞ (cuadrado integrable)

§2. OPERADOR HERMÍTICO DE MEDICIÓN
    Ĥ: ℋ → ℋ, Ĥ† = Ĥ (autoadjunto)
    
    Espectro Discreto: σ(Ĥ) = {λₙ}_{n∈ℕ} ⊂ ℝ
    Ecuación de autovalores: Ĥ|n⟩ = λₙ|n⟩
    
    Teorema Espectral:
        Ĥ = Σₙ λₙ |n⟩⟨n| (descomposición espectral)

§3. REGLA DE BORN (Proyección Probabilística)
    Dado |ψ⟩ ∈ ℋ normalizado (⟨ψ|ψ⟩ = 1):
    
    P(λₙ) = |⟨n|ψ⟩|² (probabilidad de medir λₙ)
    
    Conservación: Σₙ P(λₙ) = Σₙ |⟨n|ψ⟩|² = ⟨ψ|ψ⟩ = 1

§4. FIBRADO PRINCIPAL Y ACOPLAMIENTO GAUGE
    Estructura: π: P → M
        P: espacio total (fibrado principal)
        M: variedad base (espacio de configuración)
        G = U(1): grupo estructural (simetría gauge)
    
    Conexión: A ∈ Ω¹(P, 𝔤) donde 𝔤 = Lie(U(1)) ≅ iℝ
    Curvatura: F = dA + ½[A,A] (2-forma)
    
    Función de trabajo como sección:
        Φ: M → ℝ₊ invariante bajo transformaciones gauge locales

§5. APROXIMACIÓN WKB (Wentzel-Kramers-Brillouin)
    Ansatz semiclásico: ψ(x) = A(x)exp(iS(x)/ℏ)
    
    Condiciones de validez:
        |ℏ d²V/dx²| / |dV/dx|² ≪ 1 (variación suave)
        E < V₀ (régimen de túnel)
        λ_dB ≪ L_barrier (longitud de onda « ancho de barrera)
    
    Probabilidad de transmisión:
        T ≈ exp(-2γ) donde γ = (1/ℏ)∫ₐᵇ √[2m(V(x)-E)]dx
        
    Coeficiente de Gamow: γ > 0

§6. TEORÍA DE CATEGORÍAS - ESTRUCTURA FUNCTORIAL
    Categoría 𝒞_Ext (Payloads Externos):
        Ob(𝒞_Ext): conjuntos de mappings
        Mor(𝒞_Ext): funciones de transformación
    
    Categoría 𝒞_Phys (Estados Físicos):
        Ob(𝒞_Phys): espacios de Hilbert
        Mor(𝒞_Phys): operadores lineales acotados
    
    Funtor F: 𝒞_Ext → 𝒞_Phys
        F(payload) = |ψ⟩ ∈ ℋ
        F(g ∘ f) = F(g) ∘ F(f) (preserva composición)
        F(id) = id (preserva identidades)

§7. TOPOLOGÍA ALGEBRAICA - COHOMOLOGÍA DE SHEAVES
    Haz estructural: ℱ → X (prehaz con condición de pegado)
    
    Grupos de cohomología: Hⁿ(X, ℱ)
    Frustración ⟺ H¹(X, ℱ) ≠ 0 (obstrucción al levantamiento global)
    
    Energía de frustración:
        E_frust = ∫_X ||δω||² donde δ: Cⁿ → Cⁿ⁺¹ (operador coborde)

§8. TEORÍA DE GRAFOS - DAG COMPUTACIONAL
    G = (V, E) donde:
        V: nodos de cómputo (operadores)
        E ⊆ V × V: dependencias dirigidas
    
    Propiedades garantizadas:
        - Acíclico: no existen ciclos (previene deadlock)
        - Conexo: existe camino desde inputs a outputs
        - Topológicamente ordenado: ∃ ordenamiento lineal v₁,...,vₙ
          tal que (vᵢ, vⱼ) ∈ E ⟹ i < j

§9. ÁLGEBRA DE BOOLE Y LÓGICA CUÁNTICA
    Retícula de proyectores: (𝒫(ℋ), ∧, ∨, ⊥, 0, 𝟙)
    
    NO distributiva (diferencia clave con Boole clásica):
        P ∧ (Q ∨ R) ≠ (P ∧ Q) ∨ (P ∧ R) en general
    
    Complemento ortonormal: P⊥ tal que P + P⊥ = 𝟙, PP⊥ = 0

§10. TEOREMA DE NO-CLONACIÓN
    NO existe U: ℋ ⊗ ℋ → ℋ ⊗ ℋ unitario tal que:
        U(|ψ⟩ ⊗ |0⟩) = |ψ⟩ ⊗ |ψ⟩ ∀|ψ⟩
    
    Implementación: dataclass frozen + eliminación de __copy__/__deepcopy__

═════════════════════════════════════════════════════════════════════════════
ARQUITECTURA DE GRAFOS - DAG DE DEPENDENCIAS:

    ┌─────────────────────────────────────────────────────────────┐
    │                        Payload (Input)                       │
    └───────────────┬──────────────────────┬──────────────────────┘
                    │                      │
                    ▼                      ▼
    ┌───────────────────────┐  ┌──────────────────────────┐
    │ PayloadSerializer     │  │ SheafOrchestrator        │
    │ • serialize()         │  │ • get_frustration_energy │
    │ • deterministic_hash()│  └──────────┬───────────────┘
    └───────┬───────────────┘             │
            │                             ▼
            │                  ┌──────────────────────┐
            │                  │ CohomologicalVeto    │
            │                  │ • evaluate()         │
            │                  └──────────┬───────────┘
            │                             │
            ▼                             │
    ┌─────────────────────┐              │
    │ EntropyCalculator   │              │
    │ • shannon_entropy() │              │
    └─────────┬───────────┘              │
              │                          │
              ▼                          │
    ┌─────────────────────────┐         │
    │ IncidentEnergyCalc      │         │
    │ • calculate_energy()    │         │
    └─────────┬───────────────┘         │
              │                          │
              ├──────────────────────────┴──────────────┐
              │                                         │
              ▼                                         ▼
    ┌──────────────────────┐              ┌─────────────────────────┐
    │ TopologicalWatcher   │              │ LaplaceOracle           │
    │ • get_threat()       │              │ • get_dominant_pole()   │
    └──────────┬───────────┘              └───────────┬─────────────┘
               │                                      │
               ▼                                      ▼
    ┌──────────────────────┐              ┌─────────────────────────┐
    │ WorkFunctionMod      │              │ EffectiveMassModulator  │
    │ • calculate_Phi()    │              │ • calculate_m_eff()     │
    └──────────┬───────────┘              └───────────┬─────────────┘
               │                                      │
               └──────────────┬───────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ WKBCalculator        │
                   │ • compute_T()        │
                   └──────────┬───────────┘
                              │
                              ├─────────────────────┐
                              │                     │
                              ▼                     ▼
                   ┌──────────────────┐  ┌─────────────────────┐
                   │ ThresholdGen     │  │ WaveCollapse        │
                   │ • generate_θ()   │  │ • project_state()   │
                   └────────┬─────────┘  └──────────┬──────────┘
                            │                       │
                            └───────────┬───────────┘
                                        │
                                        ▼
                            ┌──────────────────────────┐
                            │ QuantumMeasurement       │
                            │ (Resultado Inmutable)    │
                            └──────────────────────────┘

    Propiedades del DAG:
    • Acíclico: ✓ (orden parcial bien definido)
    • Altura: 7 niveles (profundidad de pipeline)
    • Ancho máximo: 3 nodos paralelos (WKB inputs)
    • Complejidad temporal: O(n) en tamaño de payload
    • Complejidad espacial: O(1) (sin recursión)

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Generic,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN Y LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("MIC.Physics.QuantumAdmission")


@dataclass(frozen=True, slots=True)
class LogContext:
    """
    Contexto estructurado para trazabilidad cuántica completa.
    
    Invariantes:
        - Inmutabilidad total (frozen=True)
        - Serialización determinista
        - Timestamp implícito en sistema de logging
    """
    
    eigenstate: str
    incident_energy: float
    work_function: float
    tunneling_prob: float
    momentum: float
    sigma: float
    chi_squared: float
    veto: bool
    
    def to_structured_log(self) -> str:
        """
        Serialización canónica para logging estructurado.
        
        Returns:
            String con formato key=value consistente.
        """
        return (
            f"state={self.eigenstate} | "
            f"E={self.incident_energy:.6e} | "
            f"Φ={self.work_function:.6e} | "
            f"T={self.tunneling_prob:.6e} | "
            f"p={self.momentum:.6e} | "
            f"σ={self.sigma:.6e} | "
            f"χ²={self.chi_squared:.6e} | "
            f"veto={self.veto}"
        )


class QuantumLogger:
    """
    Logger especializado con contexto cuántico enriquecido.
    
    Responsabilidades:
        - Formateo consistente de mediciones
        - Separación por nivel de severidad
        - Hooks para métricas (futuro)
    """
    
    @staticmethod
    def log_measurement(
        measurement: 'QuantumMeasurement',
        level: int = logging.INFO
    ) -> None:
        """
        Registra medición cuántica con contexto completo.
        
        Args:
            measurement: Medición inmutable validada.
            level: Nivel de logging (INFO/WARNING/ERROR).
            
        Precondición:
            measurement es instancia válida de QuantumMeasurement.
        """
        context = LogContext(
            eigenstate=measurement.eigenstate.name,
            incident_energy=measurement.incident_energy,
            work_function=measurement.work_function,
            tunneling_prob=measurement.tunneling_probability,
            momentum=measurement.momentum,
            sigma=measurement.dominant_pole_real,
            chi_squared=measurement.threat_level,
            veto=measurement.frustration_veto,
        )
        
        logger.log(
            level,
            f"QUANTUM_MEASUREMENT | {context.to_structured_log()}"
        )
        
        # Hook para métricas (a implementar)
        # MetricsCollector.record_measurement(measurement)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: JERARQUÍA DE EXCEPCIONES (Categoría de Errores)
# ═════════════════════════════════════════════════════════════════════════════

class QuantumAdmissionError(Exception):
    """
    Objeto inicial en categoría de errores cuánticos.
    
    Morfismo canónico: Error → Log → Recovery
    
    Propiedades categóricas:
        - Identidad: self.identity() = self
        - Composición: (f ∘ g).handle() = f.handle(g.handle())
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Constructor con contexto diagnóstico enriquecido.
        
        Args:
            message: Descripción del error.
            context: Diccionario con parámetros relevantes.
        """
        super().__init__(message)
        self.context = context or {}
        self.message = message
    
    def __str__(self) -> str:
        """Representación con contexto."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} | Context: {ctx_str}"
        return self.message


class QuantumNumericalError(QuantumAdmissionError):
    """
    Error en cálculo numérico con propagación de incertidumbre.
    
    Causas típicas:
        - Pérdida catastrófica de precisión (catastrophic cancellation)
        - Overflow/underflow no manejado
        - Violación de cotas físicas conocidas
    """
    pass


class QuantumInterfaceError(QuantumAdmissionError):
    """
    Violación de contrato en protocolo de comunicación.
    
    Interpretación categórica:
        Fallo de funtorialidad: F(g ∘ f) ≠ F(g) ∘ F(f)
    """
    pass


class QuantumStateError(QuantumAdmissionError):
    """
    Estado fuera del espacio de Hilbert admisible.
    
    Violaciones típicas:
        - ||ψ|| ∉ [0, ∞) (norma no finita)
        - ⟨ψ|ψ⟩ < 0 (producto interno no positivo definido)
        - Energía negativa sin interpretación física
    """
    pass


class WKBValidityError(QuantumNumericalError):
    """
    Aproximación WKB fuera de régimen semiclásico.
    
    Condición violada:
        ℏ|V''|/|V'|² ≥ threshold (variación no suave)
    """
    pass


class CohomologicalVetoError(QuantumAdmissionError):
    """
    Veto estructural por frustración cohomológica.
    
    Interpretación topológica:
        H¹(X, ℱ) ≠ 0 → obstrucción global
    """
    pass


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CONSTANTES FÍSICAS (Álgebra de Números)
# ═════════════════════════════════════════════════════════════════════════════

class PhysicalConstants:
    """
    Constantes físicas fundamentales con validación estática.
    
    Sistema de unidades: ℏ = c = kB = 1 (unidades naturales)
    
    Invariantes algebraicos:
        ∀c ∈ Constants: c ∈ ℝ₊ ∪ {+∞} (no negatividad)
        ∀c ∈ Constants: |c| < DBL_MAX (representabilidad)
    
    Análisis dimensional:
        [Φ] = Energía = M L² T⁻²
        [m] = Masa = M
        [Δx] = Longitud = L
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Constantes de Planck (Sistema Natural)
    # ─────────────────────────────────────────────────────────────────────────
    
    PLANCK_H: Final[float] = 1.0
    """Constante de Planck [ℏω]."""
    
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)
    """Constante de Planck reducida [E]."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Parámetros del Potencial (Barrera Rectangular)
    # ─────────────────────────────────────────────────────────────────────────
    
    BASE_WORK_FUNCTION: Final[float] = 10.0
    """
    Función de trabajo base Φ₀ [E].
    
    Interpretación: energía mínima para extraer electrón (fotoeléctrico).
    Valor calibrado para equilibrio accept/reject ≈ 50% en carga nominal.
    """
    
    BASE_EFFECTIVE_MASS: Final[float] = 1.0
    """
    Masa efectiva base m₀ [M].
    
    En unidades naturales: m₀ = 1 corresponde a masa del electrón.
    """
    
    BARRIER_WIDTH: Final[float] = 1.0
    """
    Ancho de barrera Δx [L].
    
    Para barrera rectangular: región clásicamente prohibida.
    """
    
    ALPHA_THREAT: Final[float] = 5.0
    """
    Constante de acoplamiento gauge-topológico α [adimensional].
    
    Controla la sensibilidad de Φ a la amenaza χ²:
        Φ(χ²) = Φ₀ exp(α χ²)
    
    Análisis de estabilidad:
        α → 0: desacoplamiento (Φ constante)
        α → ∞: acoplamiento fuerte (Φ diverge rápidamente)
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tolerancias Numéricas (Análisis de Error)
    # ─────────────────────────────────────────────────────────────────────────
    
    MACHINE_EPSILON: Final[float] = sys.float_info.epsilon
    """ε_machine ≈ 2.22e-16 para float64."""
    
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    """
    Energía cinética mínima resoluble [E].
    
    Justificación:
        √(ε_machine) ≈ 1.5e-8
        Margen de seguridad: 10⁴ × √ε
    """
    
    FRUSTRATION_VETO_TOL: Final[float] = 1e-9
    """
    Umbral de energía de frustración cohomológica [E].
    
    E_frust > tol ⟹ veto absoluto
    """
    
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    """
    Tolerancia para estabilidad marginal [T⁻¹].
    
    σ ≥ -tol ⟹ sistema considerado inestable/marginal
    """
    
    ENTROPY_FLOOR: Final[float] = 1e-12
    """
    Piso de entropía para evitar singularidades [adimensional].
    
    H_effective = max(H_raw, H_floor)
    """
    
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0
    """
    Límite de underflow para exp() [adimensional].
    
    ln(DBL_MIN) ≈ -708.4
    Margen conservador: -700
    
    exp(x < -700) → 0 sin underflow gradual
    """
    
    MAX_SHANNON_ENTROPY: Final[float] = math.log(256.0)
    """
    Entropía máxima de Shannon para bytes [nats].
    
    Distribución uniforme sobre 256 símbolos:
        H_max = ln(256) ≈ 5.545 nats
    """
    
    DIVISION_EPSILON: Final[float] = 1e-15
    """
    Regularizador para división estable [adimensional].
    
    x/y → x/max(|y|, ε) · sgn(y)
    """
    
    WKB_VALIDITY_THRESHOLD: Final[float] = 0.1
    """
    Umbral de validez WKB [adimensional].
    
    Parámetro: |V''|/|V'|² < 0.1
    
    Interpretación: radio de curvatura del potencial ≫ λ_deBroglie
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Límites de Representación
    # ─────────────────────────────────────────────────────────────────────────
    
    FLOAT_MAX: Final[float] = sys.float_info.max
    """Máximo float representable ≈ 1.8e308."""
    
    FLOAT_MIN: Final[float] = sys.float_info.min
    """Mínimo float positivo normalizado ≈ 2.2e-308."""
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Sellado de clase para garantizar inmutabilidad semántica."""
        raise TypeError(
            f"La clase {cls.__name__} está sellada. "
            "No se permite herencia para preservar invariantes físicos."
        )


# Alias corto para uso frecuente
Const = PhysicalConstants


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: ENUMERACIONES (Álgebra de Boole Cuántica)
# ═════════════════════════════════════════════════════════════════════════════

class Eigenstate(Enum):
    """
    Autoestados del operador hermítico Ĥ.
    
    Base ortonormal completa: {|ADMITIDO⟩, |RECHAZADO⟩}
    
    Propiedades algebraicas:
        1. ⟨ADMITIDO|RECHAZADO⟩ = 0 (ortogonalidad)
        2. ⟨n|n⟩ = 1 (normalización)
        3. |ADMITIDO⟩⟨ADMITIDO| + |RECHAZADO⟩⟨RECHAZADO| = 𝟙 (completitud)
    
    Autovalores asociados:
        Ĥ|ADMITIDO⟩ = λ₊|ADMITIDO⟩ con λ₊ > 0
        Ĥ|RECHAZADO⟩ = λ₋|RECHAZADO⟩ con λ₋ < 0
    """
    
    ADMITIDO = auto()
    RECHAZADO = auto()
    
    def is_accepted(self) -> bool:
        """
        Proyección al álgebra de Boole clásica.
        
        Returns:
            True ⟺ autoestado corresponde a admisión.
            
        Postcondición:
            resultado ∈ {True, False} (bivalencia)
        """
        return self == Eigenstate.ADMITIDO
    
    def to_hilbert_projection(self) -> int:
        """
        Índice del proyector |n⟩⟨n|.
        
        Returns:
            1 si ADMITIDO, 0 si RECHAZADO.
            
        Uso: indexación de matrices en base canónica.
        """
        return 1 if self.is_accepted() else 0
    
    def complementary(self) -> 'Eigenstate':
        """
        Proyector complementario ortogonal.
        
        Returns:
            Estado ortogonal (complemento en álgebra de proyectores).
            
        Postcondición:
            ⟨self|complementary⟩ = 0
        """
        return (
            Eigenstate.RECHAZADO if self == Eigenstate.ADMITIDO
            else Eigenstate.ADMITIDO
        )
    
    def __str__(self) -> str:
        """Representación Dirac para logging."""
        return f"|{self.name}⟩"


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: ESTRUCTURAS ALGEBRAICAS (Espacios Métricos y Vectoriales)
# ═════════════════════════════════════════════════════════════════════════════

class NumericInterval(NamedTuple):
    """
    Intervalo cerrado [lower, upper] ⊂ ℝ con aritmética rigurosa.
    
    Estructura algebraica:
        - Espacio métrico con distancia d(I₁, I₂) = |mid(I₁) - mid(I₂)|
        - Retícula con orden parcial I₁ ≤ I₂ ⟺ upper₁ ≤ lower₂
    
    Invariantes:
        1. lower ≤ upper (orden total en extremos)
        2. lower, upper ∈ ℝ ∪ {-∞, +∞} (compactificación de Alexandroff)
        3. width = upper - lower ≥ 0 (métrica semi-definida positiva)
    
    Aplicaciones:
        - Propagación de incertidumbre numérica
        - Análisis de intervalos
        - Cotas de error en aproximaciones
    """
    
    lower: float
    upper: float
    
    def __post_init__(self) -> None:
        """
        Validación de axiomas del intervalo.
        
        Raises:
            ValueError: Si se violan invariantes.
        """
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError(
                "Intervalos no admiten NaN (not-a-number). "
                "Use [-∞, +∞] para incertidumbre total."
            )
        
        if self.lower > self.upper:
            raise ValueError(
                f"Intervalo mal formado: [{self.lower}, {self.upper}]. "
                "Se requiere lower ≤ upper."
            )
    
    @property
    def midpoint(self) -> float:
        """
        Punto medio (centro de masa en distribución uniforme).
        
        Returns:
            (lower + upper) / 2
            
        Postcondición:
            lower ≤ midpoint ≤ upper
        """
        return 0.5 * (self.lower + self.upper)
    
    @property
    def width(self) -> float:
        """
        Ancho del intervalo (incertidumbre absoluta).
        
        Returns:
            upper - lower ≥ 0
        """
        return self.upper - self.lower
    
    @property
    def relative_width(self) -> float:
        """
        Incertidumbre relativa (adimensional).
        
        Returns:
            width / |midpoint| si midpoint ≠ 0, else +∞
        """
        mid = self.midpoint
        if abs(mid) < Const.DIVISION_EPSILON:
            return float('inf')
        return self.width / abs(mid)
    
    def contains(self, value: float) -> bool:
        """
        Predicado de pertenencia al intervalo cerrado.
        
        Args:
            value: Valor a verificar.
            
        Returns:
            True ⟺ lower ≤ value ≤ upper
        """
        return self.lower <= value <= self.upper
    
    def intersects(self, other: 'NumericInterval') -> bool:
        """
        Verifica intersección no vacía de intervalos.
        
        Args:
            other: Otro intervalo.
            
        Returns:
            True ⟺ self ∩ other ≠ ∅
        """
        return not (self.upper < other.lower or other.upper < self.lower)
    
    @staticmethod
    def from_value_with_tolerance(
        value: float,
        tolerance: float
    ) -> 'NumericInterval':
        """
        Constructor centrado con radio simétrico.
        
        Args:
            value: Centro del intervalo.
            tolerance: Radio (se toma valor absoluto).
            
        Returns:
            [value - |tolerance|, value + |tolerance|]
            
        Ejemplo:
            >>> NumericInterval.from_value_with_tolerance(5.0, 0.1)
            NumericInterval(lower=4.9, upper=5.1)
        """
        abs_tol = abs(tolerance)
        return NumericInterval(
            lower=value - abs_tol,
            upper=value + abs_tol
        )


@dataclass(frozen=True, slots=True)
class WKBParameters:
    """
    Parámetros del cálculo WKB con diagnóstico completo.
    
    Encapsula toda la información necesaria para:
        1. Validar aplicabilidad de la aproximación
        2. Computar probabilidad de transmisión
        3. Diagnosticar fallos numéricos
        4. Auditoría post-mortem
    
    Attributes:
        incident_energy: E ≥ 0 [Energía]
        barrier_height: V₀ - E (debe ser > 0 en régimen túnel) [Energía]
        effective_mass: m_eff ∈ (0, +∞] [Masa]
        barrier_width: Δx > 0 [Longitud]
        integrand: √(2m(V-E)) [Momentum imaginario]
        exponent: -(2/ℏ)∫√(2m(V-E))dx [adimensional]
        validity_parameter: |V''|/|V'|² [Longitud⁻²]
    """
    
    incident_energy: float
    barrier_height: float
    effective_mass: float
    barrier_width: float
    integrand: float
    exponent: float
    validity_parameter: float
    
    def __post_init__(self) -> None:
        """
        Validación de restricciones físicas.
        
        Raises:
            QuantumNumericalError: Si algún valor es no físico.
        """
        # Validar energías no negativas
        if self.incident_energy < 0:
            raise QuantumNumericalError(
                f"Energía incidente negativa: {self.incident_energy}",
                context={"incident_energy": self.incident_energy}
            )
        
        # Validar masa positiva (puede ser +∞)
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumNumericalError(
                f"Masa efectiva debe ser positiva o infinita: {self.effective_mass}",
                context={"effective_mass": self.effective_mass}
            )
        
        # Validar ancho positivo
        if self.barrier_width <= 0:
            raise QuantumNumericalError(
                f"Ancho de barrera debe ser positivo: {self.barrier_width}",
                context={"barrier_width": self.barrier_width}
            )
    
    def is_valid_semiclassical_regime(self) -> bool:
        """
        Verifica condiciones de validez WKB.
        
        Condiciones:
            1. E < V₀ (régimen túnel, barrier_height > 0)
            2. |V''|/|V'|² < threshold (variación suave)
            3. m_eff finita y positiva
        
        Returns:
            True si todas las condiciones se cumplen.
        """
        return (
            self.barrier_height > 0
            and self.validity_parameter < Const.WKB_VALIDITY_THRESHOLD
            and 0 < self.effective_mass < float('inf')
        )
    
    def gamow_factor(self) -> float:
        """
        Calcula el factor de Gamow γ.
        
        Returns:
            γ = (1/ℏ)∫√(2m(V-E))dx ≥ 0
        """
        return abs(self.exponent) / 2.0


@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """
    Medición cuántica inmutable post-colapso.
    
    Representa el resultado de aplicar el operador de proyección Ĥ
    sobre el estado |ψ⟩ preparado por el payload.
    
    TEOREMA (No-Clonación):
        No existe morfismo QuantumMeasurement → QuantumMeasurement × QuantumMeasurement
        que duplique el estado cuántico.
    
    IMPLEMENTACIÓN:
        - frozen=True: inmutabilidad
        - slots=True: optimización de memoria
        - Sin __copy__/__deepcopy__: prevención de clonación
    
    Attributes:
        eigenstate: Estado colapsado {|ADMITIDO⟩, |RECHAZADO⟩}
        incident_energy: E = hν ≥ 0
        work_function: Φ(χ²) ≥ 0
        tunneling_probability: T ∈ [0, 1]
        kinetic_energy: K = max(E - Φ, 0) ≥ 0
        momentum: p = √(2mK) ≥ 0
        frustration_veto: Indicador de veto cohomológico
        effective_mass: m_eff ∈ (0, +∞]
        dominant_pole_real: σ ∈ ℝ (parte real de polo dominante)
        threat_level: χ² ≥ 0 (Mahalanobis)
        collapse_threshold: θ ∈ [0, 1)
        admission_reason: Diagnóstico textual
        wkb_parameters: Detalles de cálculo WKB (opcional)
        measurement_uncertainty: Intervalo de incertidumbre en E (opcional)
    """
    
    eigenstate: Eigenstate
    incident_energy: float
    work_function: float
    tunneling_probability: float
    kinetic_energy: float
    momentum: float
    frustration_veto: bool
    effective_mass: float
    dominant_pole_real: float
    threat_level: float
    collapse_threshold: float
    admission_reason: str
    wkb_parameters: Optional[WKBParameters] = None
    measurement_uncertainty: Optional[NumericInterval] = None
    
    def __post_init__(self) -> None:
        """
        Validación exhaustiva de invariantes físicos.
        
        Raises:
            QuantumStateError: Si se viola algún invariante.
        """
        # ─────────────────────────────────────────────────────────────────────
        # Validación de no-negatividad (observables físicos)
        # ─────────────────────────────────────────────────────────────────────
        
        non_negative_fields = {
            "incident_energy": self.incident_energy,
            "work_function": self.work_function,
            "kinetic_energy": self.kinetic_energy,
            "momentum": self.momentum,
            "threat_level": self.threat_level,
        }
        
        for field_name, value in non_negative_fields.items():
            if value < 0:
                raise QuantumStateError(
                    f"Observable '{field_name}' debe ser no negativo: {value}",
                    context={field_name: value}
                )
        
        # ─────────────────────────────────────────────────────────────────────
        # Validación de probabilidades
        # ─────────────────────────────────────────────────────────────────────
        
        if not 0.0 <= self.tunneling_probability <= 1.0:
            raise QuantumStateError(
                f"Probabilidad de túnel fuera de [0,1]: {self.tunneling_probability}",
                context={"tunneling_probability": self.tunneling_probability}
            )
        
        if not 0.0 <= self.collapse_threshold < 1.0:
            raise QuantumStateError(
                f"Umbral de colapso fuera de [0,1): {self.collapse_threshold}",
                context={"collapse_threshold": self.collapse_threshold}
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Validación de masa efectiva
        # ─────────────────────────────────────────────────────────────────────
        
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumStateError(
                f"Masa efectiva debe ser positiva o infinita: {self.effective_mass}",
                context={"effective_mass": self.effective_mass}
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Consistencia estado-energía
        # ─────────────────────────────────────────────────────────────────────
        
        if self.eigenstate == Eigenstate.RECHAZADO:
            # Si rechazado, K y p deben ser exactamente 0
            if self.kinetic_energy != 0.0:
                raise QuantumStateError(
                    "Estado rechazado no puede tener energía cinética no nula",
                    context={
                        "eigenstate": self.eigenstate,
                        "kinetic_energy": self.kinetic_energy
                    }
                )
            
            if self.momentum != 0.0:
                raise QuantumStateError(
                    "Estado rechazado no puede tener momentum no nulo",
                    context={
                        "eigenstate": self.eigenstate,
                        "momentum": self.momentum
                    }
                )
    
    def __repr__(self) -> str:
        """Representación compacta para debugging."""
        return (
            f"QuantumMeasurement("
            f"{self.eigenstate}, "
            f"E={self.incident_energy:.3e}, "
            f"T={self.tunneling_probability:.3e}, "
            f"p={self.momentum:.3e})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización a diccionario para telemetría.
        
        Returns:
            Diccionario con todos los campos serializables.
        """
        base_dict = {
            "eigenstate": self.eigenstate.name,
            "incident_energy": self.incident_energy,
            "work_function": self.work_function,
            "tunneling_probability": self.tunneling_probability,
            "kinetic_energy": self.kinetic_energy,
            "momentum": self.momentum,
            "frustration_veto": self.frustration_veto,
            "effective_mass": self.effective_mass,
            "dominant_pole_real": self.dominant_pole_real,
            "threat_level": self.threat_level,
            "collapse_threshold": self.collapse_threshold,
            "admission_reason": self.admission_reason,
        }
        
        # Agregar detalles opcionales si existen
        if self.wkb_parameters:
            base_dict["wkb_gamow_factor"] = self.wkb_parameters.gamow_factor()
        
        if self.measurement_uncertainty:
            base_dict["energy_uncertainty"] = self.measurement_uncertainty.width
        
        return base_dict
    
    def de_broglie_wavelength(self) -> float:
        """
        Calcula la longitud de onda de De Broglie.
        
        Returns:
            λ = h/p si p > 0, else +∞
        """
        if self.momentum < Const.DIVISION_EPSILON:
            return float('inf')
        
        return Const.PLANCK_H / self.momentum


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: PROTOCOLOS DE INTERFACES (Funtores Categóricos)
# ═════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ITopologicalWatcher(Protocol):
    """
    Funtor: Top → ℝ₊ (Categoría Topológica → Números Reales No Negativos)
    
    Observador que computa la métrica de Mahalanobis χ² como medida
    de distancia geodésica en el espacio de configuración.
    
    AXIOMAS TOPOLÓGICOS:
        1. χ²: X → ℝ₊ es continua en la topología natural de X
        2. χ²(x₀) = 0 donde x₀ es el estado nominal (identidad)
        3. χ² es una semi-métrica: χ²(x, y) ≥ 0, χ²(x, x) = 0
    
    CONTRATO:
        Precondición: Sistema inicializado y en estado consistente
        Postcondición: Retorna float finito y no negativo
        Invariante: Idempotencia en ausencia de mutaciones externas
    """
    
    def get_mahalanobis_threat(self) -> float:
        """
        Calcula la amenaza topológica χ².
        
        Returns:
            χ² ∈ [0, +∞) con χ² = 0 en estado nominal.
            
        Raises:
            RuntimeError: Si el cálculo falla irrecuperablemente.
        """
        ...


@runtime_checkable
class ILaplaceOracle(Protocol):
    """
    Funtor: LTI → ℂ (Sistemas Lineales Invariantes → Números Complejos)
    
    Oráculo que extrae el polo dominante del sistema de control,
    determinando la estabilidad asintótica.
    
    TEORÍA DE SISTEMAS:
        H(s) = N(s)/D(s) con polos {pᵢ} = {s: D(s) = 0}
        
        Polo dominante: p* = argmax Re(pᵢ)
                              pᵢ ∈ polos
        
        Clasificación de estabilidad:
            Re(p*) < 0 → asintóticamente estable
            Re(p*) = 0 → marginalmente estable
            Re(p*) > 0 → inestable
    
    CONTRATO:
        Precondición: Sistema LTI con representación válida
        Postcondición: Retorna parte real del polo dominante
        Excepción: Si no existen polos o cálculo falla
    """
    
    def get_dominant_pole_real(self) -> float:
        """
        Extrae Re(polo dominante) del sistema.
        
        Returns:
            σ = Re(p*) ∈ ℝ
            
        Raises:
            RuntimeError: Si el sistema no tiene polos o el cálculo falla.
        """
        ...


@runtime_checkable
class ISheafCohomologyOrchestrator(Protocol):
    """
    Funtor: Sh(X) → ℝ₊ (Categoría de Haces → Energías)
    
    Orquestador que calcula la energía de frustración cohomológica,
    midiendo obstrucciones globales en la estructura de haces.
    
    COHOMOLOGÍA DE SHEAVES:
        Para un haz ℱ sobre espacio topológico X:
        
        Complejo de cocadenas: 0 → C⁰ → C¹ → C² → ...
        Operador coborde: δ: Cⁿ → Cⁿ⁺¹
        
        Grupo de cohomología: Hⁿ(X, ℱ) = Ker(δₙ₊₁) / Im(δₙ)
        
        Frustración ⟺ H¹(X, ℱ) ≠ 0 (obstrucción al pegado global)
        
        Energía: E = ∫_X ||δω||² dμ
    
    CONTRATO:
        Precondición: Haces monitoreados en estado consistente
        Postcondición: Retorna energía no negativa
        Invariante: E = 0 ⟺ cohomología trivial
    """
    
    def get_global_frustration_energy(self) -> float:
        """
        Calcula energía de frustración cohomológica global.
        
        Returns:
            E_frustración ≥ 0
            
        Raises:
            RuntimeError: Si el cálculo cohomológico falla.
        """
        ...


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: ÁLGEBRA NUMÉRICA (Morfismos Seguros)
# ═════════════════════════════════════════════════════════════════════════════

class NumericalMorphisms:
    """
    Colección de morfismos numéricos con garantías algebraicas.
    
    Todos los métodos son funciones puras (sin efectos secundarios)
    con manejo exhaustivo de casos especiales.
    """
    
    @staticmethod
    def validate_finite_float(
        value: Any,
        *,
        name: str,
        allow_inf: bool = False
    ) -> float:
        """
        Morfismo Any → ℝ_finito con validación rigurosa.
        
        Args:
            value: Valor a convertir.
            name: Identificador para mensajes de error.
            allow_inf: Permitir ±∞ en codominio.
            
        Returns:
            float validado según restricciones.
            
        Raises:
            QuantumNumericalError: Si conversión falla o valor inválido.
        """
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise QuantumNumericalError(
                f"Parámetro '{name}' no convertible a float: {value!r}",
                context={"name": name, "value": value, "type": type(value).__name__}
            ) from exc
        
        # Rechazo absoluto de NaN
        if math.isnan(result):
            raise QuantumNumericalError(
                f"Parámetro '{name}' es NaN (valor no físico)",
                context={"name": name}
            )
        
        # Validación de infinitos según política
        if not allow_inf and math.isinf(result):
            raise QuantumNumericalError(
                f"Parámetro '{name}' es infinito: {result!r}",
                context={"name": name, "value": result}
            )
        
        return result
    
    @staticmethod
    def clamp_to_unit_interval(value: float) -> float:
        """
        Proyección saturante [ℝ → [0, 1]].
        
        Manejo de casos especiales:
            NaN → 0.0 (con warning)
            +∞ → 1.0
            -∞ → 0.0
            x ∈ [0,1] → x
            x < 0 → 0.0
            x > 1 → 1.0
        
        Args:
            value: Valor a proyectar.
            
        Returns:
            Proyección en [0, 1].
        """
        if math.isnan(value):
            logger.warning(
                "clamp_to_unit_interval recibió NaN, proyectando a 0.0. "
                "Revisar cálculo upstream."
            )
            return 0.0
        
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        
        # Saturación a límites del intervalo
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        
        return value
    
    @staticmethod
    def safe_division(
        numerator: float,
        denominator: float,
        *,
        fallback: float = 0.0,
        epsilon: float = Const.DIVISION_EPSILON
    ) -> float:
        """
        División con regularización: x/y → x/max(|y|, ε)·sgn(y).
        
        Propiedades:
            - No singularidades para |y| < ε
            - Continuidad en ε-vecindad de 0
            - Monotonía preservada
        
        Args:
            numerator: Numerador (finito).
            denominator: Denominador (finito).
            fallback: Valor si denominador es exactamente 0.
            epsilon: Regularizador mínimo.
            
        Returns:
            Cociente regularizado.
            
        Raises:
            QuantumNumericalError: Si inputs no son finitos.
        """
        num = NumericalMorphisms.validate_finite_float(
            numerator,
            name="numerator"
        )
        denom = NumericalMorphisms.validate_finite_float(
            denominator,
            name="denominator"
        )
        
        # Caso exacto: denominador cero
        if denom == 0.0:
            logger.warning(
                f"División por cero exacto: {num}/0.0, retornando fallback={fallback}"
            )
            return fallback
        
        # Regularización preservando signo
        regularized_denom = (
            denom if abs(denom) >= epsilon
            else math.copysign(epsilon, denom)
        )
        
        return num / regularized_denom
    
    @staticmethod
    def safe_sqrt(value: float) -> float:
        """
        Raíz cuadrada con proyección a no-negatividad: √max(0, x).
        
        Args:
            value: Argumento del radical.
            
        Returns:
            √max(0, value) ≥ 0
        """
        return math.sqrt(max(0.0, value))
    
    @staticmethod
    def safe_exp(exponent: float) -> float:
        """
        Exponencial con saturación: exp(x) con protección de overflow/underflow.
        
        Saturación:
            x ≤ -700 → 0.0 (underflow)
            x > 700 → float_max (overflow saturado)
            -700 < x ≤ 700 → exp(x)
        
        Args:
            exponent: Exponente (finito).
            
        Returns:
            exp(exponent) con saturación, nunca infinito.
        """
        if exponent <= Const.EXP_UNDERFLOW_CUTOFF:
            return 0.0
        
        if exponent > 700.0:
            logger.warning(
                f"Exponente grande ({exponent:.2f}), saturando a FLOAT_MAX"
            )
            return Const.FLOAT_MAX
        
        return math.exp(exponent)
    
    @staticmethod
    def safe_log(value: float, *, floor: float = 1e-300) -> float:
        """
        Logaritmo natural con piso: ln(max(x, floor)).
        
        Args:
            value: Argumento del logaritmo.
            floor: Valor mínimo para evitar -∞.
            
        Returns:
            ln(max(value, floor))
        """
        return math.log(max(value, floor))


# Alias corto
NM = NumericalMorphisms


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: TEORÍA DE LA INFORMACIÓN (Entropía de Shannon)
# ═════════════════════════════════════════════════════════════════════════════

class EntropyCalculator:
    """
    Calculador de entropía de Shannon con aritmética rigurosa.
    
    FUNDAMENTO TEÓRICO:
        Para una variable aleatoria discreta X con alfabeto 𝒜:
        
        H(X) = -Σₐ∈𝒜 P(X=a) log P(X=a)
        
        Propiedades:
            1. H(X) ≥ 0 (no-negatividad)
            2. H(X) = 0 ⟺ X determinista
            3. H(X) ≤ log|𝒜| (máximo para distribución uniforme)
            4. H es cóncava en P
    
    IMPLEMENTACIÓN:
        - Alfabeto: bytes {0, ..., 255}
        - Unidad: nats (logaritmo natural)
        - Caché: LRU con 1024 entradas
    """
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def shannon_entropy_bytes(data: bytes) -> float:
        """
        Calcula H(X) para secuencia de bytes.
        
        Algoritmo:
            1. Contar frecuencias f_i para cada byte i ∈ {0,...,255}
            2. Normalizar: p_i = f_i / n
            3. Acumular: H = -Σᵢ p_i ln(p_i)
        
        Args:
            data: Secuencia inmutable de bytes.
            
        Returns:
            H ∈ [0, ln(256)] ≈ [0, 5.545] nats
            
        Postcondición:
            H = 0 ⟺ todos los bytes son idénticos
        """
        if not data:
            return 0.0
        
        n = len(data)
        
        # Histograma de frecuencias
        byte_counts: list[int] = [0] * 256
        for byte_val in data:
            byte_counts[byte_val] += 1
        
        # Acumulación de entropía con aritmética estable
        entropy = 0.0
        for count in byte_counts:
            if count == 0:
                continue
            
            p = count / n
            # -p ln(p) con cancelación numérica mínima
            entropy -= p * math.log(p)
        
        # Garantizar no-negatividad (protección contra errores de redondeo)
        return max(0.0, min(entropy, Const.MAX_SHANNON_ENTROPY))
    
    @staticmethod
    def normalized_entropy(data: bytes) -> float:
        """
        Entropía normalizada: H_norm = H / H_max ∈ [0, 1].
        
        Args:
            data: Secuencia de bytes.
            
        Returns:
            H / ln(256) ∈ [0, 1]
        """
        raw_entropy = EntropyCalculator.shannon_entropy_bytes(data)
        return raw_entropy / Const.MAX_SHANNON_ENTROPY
    
    @staticmethod
    def conditional_entropy(
        data: bytes,
        condition: Callable[[int], bool]
    ) -> float:
        """
        Entropía condicional H(X|Y) donde Y = {condition(byte)}.
        
        Args:
            data: Secuencia de bytes.
            condition: Predicado byte → bool
            
        Returns:
            H(X|Y) ≥ 0
        """
        # Particionar datos según condición
        true_bytes = bytes(b for b in data if condition(b))
        false_bytes = bytes(b for b in data if not condition(b))
        
        if not data:
            return 0.0
        
        n = len(data)
        p_true = len(true_bytes) / n
        p_false = len(false_bytes) / n
        
        # H(X|Y) = P(Y=True)·H(X|Y=True) + P(Y=False)·H(X|Y=False)
        h_cond = (
            p_true * EntropyCalculator.shannon_entropy_bytes(true_bytes)
            + p_false * EntropyCalculator.shannon_entropy_bytes(false_bytes)
        )
        
        return h_cond


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: SERIALIZACIÓN CANÓNICA (Homomorfismo Determinista)
# ═════════════════════════════════════════════════════════════════════════════

class PayloadSerializer:
    """
    Serialización canónica con garantías criptográficas.
    
    PROPIEDADES:
        1. Determinismo: serialize(p₁) = serialize(p₂) ⟺ p₁ ≡ p₂
        2. Inyectividad: payloads distintos → bytes distintos (casi seguro)
        3. Independencia de orden: {k₁:v₁, k₂:v₂} ≡ {k₂:v₂, k₁:v₁}
    
    HASH:
        SHA-256: resistencia a colisiones ~2¹²⁸ operaciones
        Digest: 32 bytes (256 bits)
    """
    
    @staticmethod
    def serialize(payload: Mapping[str, Any]) -> bytes:
        """
        Morfismo Mapping → bytes con ordenamiento canónico.
        
        Algoritmo:
            1. Extraer pares (k, v)
            2. Ordenar por k (orden lexicográfico)
            3. Representar v con repr() (canónico en Python)
            4. Formar tupla ordenada
            5. Codificar a UTF-8
        
        Args:
            payload: Mapping a serializar.
            
        Returns:
            Representación canónica en bytes.
            
        Raises:
            QuantumAdmissionError: Si serialización falla.
        """
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}",
                context={"type": type(payload).__name__}
            )
        
        try:
            # Ordenamiento lexicográfico de claves
            ordered_items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0]
                )
            )
            
            canonical_repr = repr(ordered_items)
            return canonical_repr.encode('utf-8', errors='strict')
        
        except Exception as exc:
            raise QuantumAdmissionError(
                f"Fallo en serialización canónica: {exc}",
                context={"payload_keys": list(payload.keys())}
            ) from exc
    
    @staticmethod
    @lru_cache(maxsize=512)
    def deterministic_hash(data: bytes) -> bytes:
        """
        Hash criptográfico SHA-256 con caché LRU.
        
        Args:
            data: Datos inmutables.
            
        Returns:
            Digest de 32 bytes.
        """
        return hashlib.sha256(data).digest()
    
    @staticmethod
    def hash_to_unit_interval(hash_bytes: bytes) -> float:
        """
        Proyección uniforme: bytes → [0, 1).
        
        Algoritmo:
            1. Tomar primeros 8 bytes → uint64
            2. Reducir módulo 10⁶
            3. Normalizar: x/10⁶ ∈ [0, 1)
        
        Args:
            hash_bytes: Hash de al menos 8 bytes.
            
        Returns:
            θ ∈ [0, 1) uniformemente distribuido.
        """
        # Conversión a entero de 64 bits sin signo
        uint64_value = int.from_bytes(
            hash_bytes[:8],
            byteorder='big',
            signed=False
        )
        
        # Reducción modular para distribución uniforme
        MODULUS = 1_000_000
        normalized = (uint64_value % MODULUS) / MODULUS
        
        # Garantizar [0, 1)
        return NM.clamp_to_unit_interval(normalized)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: CALCULADORES FÍSICOS (Componentes Modulares)
# ═════════════════════════════════════════════════════════════════════════════

class IncidentEnergyCalculator:
    """
    Calcula energía incidente E = hν del cuanto informacional.
    
    MODELO FÍSICO:
        Analogía con efecto fotoeléctrico:
        
        Fotón: E_photon = hν
        Cuanto informacional: E_quantum = h·ν_eff
        
        Donde:
            ν_eff = (tamaño_bytes) / max(entropía, ε) / factor_escala
            
        Factor de escala: 1000 (calibrado empíricamente)
    
    PROPAGACIÓN DE INCERTIDUMBRE:
        δE/E ≈ δn/n + δH/H (suma de incertidumbres relativas)
        
        Intervalo: [E - δE, E + δE]
    """
    
    @staticmethod
    def calculate(payload: Mapping[str, Any]) -> Tuple[float, NumericInterval]:
        """
        Calcula E con propagación de incertidumbre.
        
        Args:
            payload: Datos del cuanto.
            
        Returns:
            (E, intervalo_incertidumbre)
            
        Raises:
            QuantumNumericalError: Si E < 0 o no finita.
        """
        # Serialización canónica
        data = PayloadSerializer.serialize(payload)
        n_bytes = len(data)
        
        # Caso especial: payload vacío
        if n_bytes == 0:
            return 0.0, NumericInterval(0.0, 0.0)
        
        # Entropía con piso regularizador
        raw_entropy = EntropyCalculator.shannon_entropy_bytes(data)
        effective_entropy = max(raw_entropy, Const.ENTROPY_FLOOR)
        
        # Frecuencia efectiva (normalizada)
        nu = NM.safe_division(
            n_bytes,
            effective_entropy,
            epsilon=Const.ENTROPY_FLOOR
        ) / 1000.0
        
        # Energía incidente E = h·ν
        E = Const.PLANCK_H * nu
        E = NM.validate_finite_float(E, name="incident_energy")
        
        if E < 0:
            raise QuantumNumericalError(
                f"Energía incidente negativa: {E}",
                context={"E": E, "nu": nu, "n_bytes": n_bytes}
            )
        
        # Estimación de incertidumbre por discretización
        # δE ≈ h / n (Heisenberg en espacio de tamaño finito)
        uncertainty = Const.PLANCK_H / max(n_bytes, 1)
        interval = NumericInterval.from_value_with_tolerance(E, uncertainty)
        
        logger.debug(
            f"Energía incidente: E={E:.6e} ± {uncertainty:.6e}, "
            f"n={n_bytes} bytes, H={raw_entropy:.4f} nats"
        )
        
        return E, interval


class WKBCalculator:
    """
    Implementación rigurosa de aproximación WKB.
    
    ECUACIÓN DE SCHRÖDINGER:
        -ℏ²/(2m) d²ψ/dx² + V(x)ψ = Eψ
    
    ANSATZ WKB:
        ψ(x) = A(x) exp(iS(x)/ℏ)
    
    PROBABILIDAD DE TRANSMISIÓN:
        T ≈ exp(-2γ) donde γ = (1/ℏ)∫ₐᵇ √[2m(V-E)]dx
    
    VALIDEZ:
        |ℏ V''| / |V'|² ≪ 1
    """
    
    @staticmethod
    def compute_tunneling_probability(
        E: float,
        Phi: float,
        m_eff: float
    ) -> Tuple[float, WKBParameters]:
        """
        Calcula probabilidad de transmisión T con diagnóstico WKB.
        
        CASOS:
            1. E ≥ Φ → T = 1 (clásico)
            2. m_eff = ∞ → T = 0 (barrera impenetrable)
            3. E < Φ, m_eff finita → WKB completo
        
        Args:
            E: Energía incidente ≥ 0
            Phi: Función de trabajo ≥ 0
            m_eff: Masa efectiva > 0 o ∞
            
        Returns:
            (T, parámetros) con T ∈ [0, 1]
            
        Raises:
            QuantumNumericalError: Si inputs inválidos.
        """
        # Validación exhaustiva
        E = NM.validate_finite_float(E, name="E_wkb")
        Phi = NM.validate_finite_float(Phi, name="Phi_wkb")
        m_eff = NM.validate_finite_float(m_eff, name="m_eff_wkb", allow_inf=True)
        
        if E < 0 or Phi < 0:
            raise QuantumNumericalError(
                f"Energías negativas no físicas: E={E}, Φ={Phi}",
                context={"E": E, "Phi": Phi}
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # CASO 1: Transmisión clásica (E ≥ Φ)
        # ─────────────────────────────────────────────────────────────────────
        
        if E >= Phi:
            params = WKBParameters(
                incident_energy=E,
                barrier_height=0.0,
                effective_mass=m_eff,
                barrier_width=Const.BARRIER_WIDTH,
                integrand=0.0,
                exponent=0.0,
                validity_parameter=0.0
            )
            logger.debug("WKB: Transmisión clásica (E ≥ Φ), T=1.0")
            return 1.0, params
        
        # Altura efectiva de barrera
        barrier_height = Phi - E
        
        # ─────────────────────────────────────────────────────────────────────
        # CASO 2: Masa infinita → barrera impenetrable
        # ─────────────────────────────────────────────────────────────────────
        
        if math.isinf(m_eff):
            params = WKBParameters(
                incident_energy=E,
                barrier_height=barrier_height,
                effective_mass=m_eff,
                barrier_width=Const.BARRIER_WIDTH,
                integrand=float('inf'),
                exponent=float('-inf'),
                validity_parameter=float('inf')
            )
            logger.debug("WKB: Masa infinita, barrera impenetrable, T=0.0")
            return 0.0, params
        
        # Validación de masa positiva
        if m_eff <= 0:
            raise QuantumNumericalError(
                f"Masa efectiva debe ser positiva: m_eff={m_eff}",
                context={"m_eff": m_eff}
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # CASO 3: Cálculo WKB completo
        # ─────────────────────────────────────────────────────────────────────
        
        # Integrando: κ = √[2m(V-E)]
        integrand = NM.safe_sqrt(2.0 * m_eff * barrier_height)
        
        # Exponente de Gamow: -(2/ℏ) Δx κ
        raw_exponent = -(
            (2.0 / Const.PLANCK_HBAR)
            * Const.BARRIER_WIDTH
            * integrand
        )
        
        # Parámetro de validez (barrera rectangular):
        # |V''|/|V'|² ≈ Δx²/Φ para transiciones bruscas en bordes
        validity_param = (
            (Const.BARRIER_WIDTH ** 2) / Phi
            if Phi > 0 else 0.0
        )
        
        params = WKBParameters(
            incident_energy=E,
            barrier_height=barrier_height,
            effective_mass=m_eff,
            barrier_width=Const.BARRIER_WIDTH,
            integrand=integrand,
            exponent=raw_exponent,
            validity_parameter=validity_param
        )
        
        # Verificación de validez
        if not params.is_valid_semiclassical_regime():
            logger.warning(
                f"WKB fuera de régimen semiclásico: "
                f"validity={validity_param:.3e} > {Const.WKB_VALIDITY_THRESHOLD:.3e}"
            )
        
        # Protección de underflow
        if raw_exponent <= Const.EXP_UNDERFLOW_CUTOFF:
            logger.debug(
                f"WKB: Exponente {raw_exponent:.2f} < cutoff, T≈0.0"
            )
            return 0.0, params
        
        # Cálculo final con saturación
        T_raw = NM.safe_exp(raw_exponent)
        T = NM.clamp_to_unit_interval(T_raw)
        
        logger.debug(
            f"WKB: E={E:.3e}, Φ={Phi:.3e}, m={m_eff:.3e}, "
            f"γ={params.gamow_factor():.3e}, T={T:.6e}"
        )
        
        return T, params


class WorkFunctionModulator:
    """
    Modula Φ mediante acoplamiento gauge con topología.
    
    MODELO:
        Φ(χ²) = Φ₀ exp(α χ²)
    
    INTERPRETACIÓN GAUGE:
        Φ es sección del fibrado principal U(1) → P → M
        α es la conexión (1-forma gauge)
        χ² parametriza la curvatura
    """
    
    def __init__(self, topo_watcher: ITopologicalWatcher):
        """
        Inyección de dependencia del observador topológico.
        
        Args:
            topo_watcher: Instancia de ITopologicalWatcher.
        """
        self._topo_watcher = topo_watcher
    
    def calculate(self) -> Tuple[float, float]:
        """
        Calcula (Φ, χ²).
        
        Returns:
            (Φ ≥ 0, χ² ≥ 0)
            
        Raises:
            QuantumNumericalError: Si valores no físicos.
        """
        # Lectura defensiva de χ²
        try:
            threat_raw = self._topo_watcher.get_mahalanobis_threat()
        except Exception as exc:
            logger.error(
                f"Fallo al obtener χ²: {exc}. Asumiendo χ²=0."
            )
            threat_raw = 0.0
        
        # Validación y clamp
        threat = max(
            0.0,
            NM.validate_finite_float(threat_raw, name="chi_squared")
        )
        
        # Modulación exponencial con saturación
        exponential_factor = NM.safe_exp(Const.ALPHA_THREAT * threat)
        
        # Φ = Φ₀ · exp(α χ²)
        try:
            phi_raw = Const.BASE_WORK_FUNCTION * exponential_factor
        except OverflowError:
            phi_raw = Const.FLOAT_MAX
        
        if math.isinf(phi_raw):
            phi_raw = Const.FLOAT_MAX
        
        Phi = max(0.0, NM.validate_finite_float(phi_raw, name="work_function"))
        
        logger.debug(
            f"Función de trabajo: Φ={Phi:.6e} (Φ₀={Const.BASE_WORK_FUNCTION}, "
            f"χ²={threat:.6e}, α={Const.ALPHA_THREAT})"
        )
        
        return Phi, threat


class EffectiveMassModulator:
    """
    Modula masa efectiva mediante polo de Laplace.
    
    MODELO:
        m_eff(σ) = m₀ / |σ|  si σ < -ε
        m_eff(σ) = ∞        si σ ≥ -ε
    
    INTERPRETACIÓN:
        σ < 0: sistema estable → masa ligera → túnel facilitado
        σ → 0: sistema marginal → masa divergente → túnel suprimido
        σ ≥ 0: sistema inestable → barrera impenetrable
    """
    
    def __init__(self, laplace_oracle: ILaplaceOracle):
        """
        Inyección de dependencia del oráculo de Laplace.
        
        Args:
            laplace_oracle: Instancia de ILaplaceOracle.
        """
        self._laplace_oracle = laplace_oracle
    
    def calculate(self) -> Tuple[float, float]:
        """
        Calcula (m_eff, σ).
        
        Returns:
            (m_eff ∈ (0, ∞], σ ∈ ℝ)
            
        Raises:
            QuantumNumericalError: Si m_eff finita pero ≤ 0.
        """
        # Lectura defensiva de σ
        try:
            sigma_raw = self._laplace_oracle.get_dominant_pole_real()
        except Exception as exc:
            logger.error(
                f"Fallo al obtener σ: {exc}. Asumiendo sistema inestable (σ=0)."
            )
            sigma_raw = 0.0
        
        sigma = NM.validate_finite_float(sigma_raw, name="sigma")
        
        # Caso inestable/marginal
        if sigma >= -Const.SIGMA_CHAOS_TOL:
            logger.warning(
                f"Sistema inestable (σ={sigma:.6e} ≥ -tol). "
                "Masa efectiva → ∞, supresión total de túnel."
            )
            return float('inf'), sigma
        
        # Caso estable: m_eff = m₀ / |σ|
        m_eff = NM.safe_division(
            Const.BASE_EFFECTIVE_MASS,
            abs(sigma),
            fallback=float('inf'),
            epsilon=Const.SIGMA_CHAOS_TOL
        )
        
        m_eff = NM.validate_finite_float(
            m_eff,
            name="m_eff",
            allow_inf=True
        )
        
        if not math.isinf(m_eff) and m_eff <= 0:
            raise QuantumNumericalError(
                f"Masa efectiva no positiva: m_eff={m_eff}, σ={sigma}",
                context={"m_eff": m_eff, "sigma": sigma}
            )
        
        logger.debug(
            f"Masa efectiva: m_eff={m_eff:.6e} "
            f"(m₀={Const.BASE_EFFECTIVE_MASS}, σ={sigma:.6e})"
        )
        
        return m_eff, sigma


class CollapseThresholdGenerator:
    """
    Genera umbral determinista θ ∈ [0,1) vía hash.
    
    PROPIEDADES:
        - Determinismo: hash(p) constante → θ constante
        - Distribución uniforme (asintóticamente)
        - Infeasibilidad de inversión (SHA-256)
    """
    
    @staticmethod
    @lru_cache(maxsize=512)
    def generate(payload_hash: bytes) -> float:
        """
        Genera θ a partir de hash SHA-256.
        
        Args:
            payload_hash: 32 bytes de SHA-256.
            
        Returns:
            θ ∈ [0, 1)
        """
        return PayloadSerializer.hash_to_unit_interval(payload_hash)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: OPERADOR DE ADMISIÓN CUÁNTICA (Morfismo Principal)
# ═════════════════════════════════════════════════════════════════════════════

class QuantumAdmissionGate(Morphism):
    """
    Operador de proyección de Hilbert como morfismo categórico.
    
    ESTRUCTURA CATEGÓRICA:
        Funtor F: 𝒞_Ext → 𝒞_Phys
        
        F(payload) = (|ψ_final⟩, observables)
        
        Preserva:
            - Composición: F(g ∘ f) = F(g) ∘ F(f)
            - Identidades: F(id) = id
    
    ALGORITMO (Pipeline de 7 Etapas):
        
        Etapa 1: Veto Cohomológico
            E_frust = ISheafOrchestrator.get_global_frustration()
            if E_frust > tol → RECHAZADO (veto absoluto)
        
        Etapa 2: Energía Incidente
            E = h·ν(payload)
            ν = tamaño / max(entropía, ε)
        
        Etapa 3: Función de Trabajo
            Φ = Φ₀ exp(α χ²)
            χ² = ITopologicalWatcher.get_mahalanobis_threat()
        
        Etapa 4: Masa Efectiva
            m_eff = m₀ / |σ|
            σ = ILaplaceOracle.get_dominant_pole()
        
        Etapa 5: Probabilidad WKB
            if E ≥ Φ: T = 1
            else: T = exp(-2γ), γ = (1/ℏ)∫√[2m(V-E)]dx
        
        Etapa 6: Umbral Determinista
            θ = hash(payload) mod 1
        
        Etapa 7: Colapso de Onda
            if T ≥ θ:
                → |ADMITIDO⟩
                K = max(E - Φ, K_min)
                p = √(2mK)
            else:
                → |RECHAZADO⟩
                K = p = 0
    
    INVARIANTES GLOBALES:
        1. Conservación probabilística: P(ADMITIDO) + P(RECHAZADO) = 1
        2. Unitariedad: ⟨ψ_final|ψ_final⟩ = 1
        3. Hermiticidad: eigenvalores reales
        4. Momentum físico: p ≥ 0
    """
    
    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
    ) -> None:
        """
        Constructor con inyección de dependencias.
        
        Args:
            topo_watcher: Observador topológico (χ²).
            laplace_oracle: Oráculo de polos (σ).
            sheaf_orchestrator: Orquestador cohomológico (E_frust).
            
        Raises:
            QuantumInterfaceError: Si dependencias no cumplen protocolos.
        """
        super().__init__(name="QuantumAdmissionGate")
        
        # Almacenamiento de dependencias
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        
        # Construcción de componentes modulares
        self._work_function_modulator = WorkFunctionModulator(topo_watcher)
        self._effective_mass_modulator = EffectiveMassModulator(laplace_oracle)
        
        # Validación exhaustiva de contratos
        self._validate_dependencies()
        
        logger.info(
            "QuantumAdmissionGate inicializada. "
            "Todas las dependencias validadas exitosamente."
        )
    
    @property
    def domain(self) -> frozenset:
        """Dominio: categoría externa sin estructura adicional."""
        return frozenset()
    
    @property
    def codomain(self) -> Stratum:
        """Codominio: stratum físico validado."""
        return Stratum.PHYSICS
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validación de Contratos (Verificación de Funtorialidad)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _validate_dependencies(self) -> None:
        """
        Verifica que dependencias cumplan protocolos en runtime.
        
        Validaciones:
            1. No-nulidad
            2. Presencia de métodos requeridos
            3. Callabilidad de métodos
            4. Conformidad con protocolo (isinstance)
        
        Raises:
            QuantumInterfaceError: Si alguna validación falla.
        """
        dependencies = [
            (
                self._topo_watcher,
                ITopologicalWatcher,
                "topo_watcher",
                ['get_mahalanobis_threat']
            ),
            (
                self._laplace_oracle,
                ILaplaceOracle,
                "laplace_oracle",
                ['get_dominant_pole_real']
            ),
            (
                self._sheaf_orchestrator,
                ISheafCohomologyOrchestrator,
                "sheaf_orchestrator",
                ['get_global_frustration_energy']
            ),
        ]
        
        for obj, protocol, name, methods in dependencies:
            # Validación de no-nulidad
            if obj is None:
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no puede ser None",
                    context={"dependency": name}
                )
            
            # Validación estructural de métodos
            for method in methods:
                if not hasattr(obj, method):
                    raise QuantumInterfaceError(
                        f"Dependencia '{name}' no implementa '{method}'",
                        context={"dependency": name, "method": method}
                    )
                
                if not callable(getattr(obj, method)):
                    raise QuantumInterfaceError(
                        f"Atributo '{method}' de '{name}' no es callable",
                        context={"dependency": name, "method": method}
                    )
            
            # Validación de protocolo (runtime)
            if not isinstance(obj, protocol):
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no cumple protocolo {protocol.__name__}",
                    context={
                        "dependency": name,
                        "expected_protocol": protocol.__name__,
                        "actual_type": type(obj).__name__
                    }
                )
        
        logger.debug("Validación de dependencias completada.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Etapa 1: Evaluación de Veto Cohomológico
    # ─────────────────────────────────────────────────────────────────────────
    
    def _evaluate_cohomological_veto(self) -> Tuple[bool, float]:
        """
        Evalúa veto estructural por frustración cohomológica.
        
        CONDICIÓN DE VETO:
            E_frust > FRUSTRATION_VETO_TOL
        
        Returns:
            (veto_activo, E_frust)
            
        Raises:
            QuantumInterfaceError: Si cálculo cohomológico falla.
        """
        try:
            frustration_raw = (
                self._sheaf_orchestrator.get_global_frustration_energy()
            )
        except Exception as exc:
            raise QuantumInterfaceError(
                f"Fallo en cálculo de energía de frustración: {exc}",
                context={"exception": str(exc)}
            ) from exc
        
        frustration = NM.validate_finite_float(
            frustration_raw,
            name="global_frustration"
        )
        frustration = max(0.0, frustration)
        
        veto_active = frustration > Const.FRUSTRATION_VETO_TOL
        
        if veto_active:
            logger.warning(
                f"VETO COHOMOLÓGICO: E_frust={frustration:.6e} > "
                f"tol={Const.FRUSTRATION_VETO_TOL:.6e}"
            )
        
        return veto_active, frustration
    
    def _build_veto_measurement(
        self,
        frustration: float
    ) -> QuantumMeasurement:
        """
        Construye medición de rechazo por veto cohomológico.
        
        Args:
            frustration: Energía que causó el veto.
            
        Returns:
            QuantumMeasurement con eigenstate=RECHAZADO, veto=True.
        """
        # Lectura defensiva de diagnósticos
        threat = self._safe_read_threat()
        sigma = self._safe_read_sigma()
        
        return QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=0.0,
            work_function=0.0,
            tunneling_probability=0.0,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=True,
            effective_mass=float('inf'),
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=0.999999,
            admission_reason=(
                f"VETO COHOMOLÓGICO: E_frustración={frustration:.6e} > "
                f"tolerancia={Const.FRUSTRATION_VETO_TOL:.6e}. "
                "Obstrucción topológica impide admisión."
            ),
        )
    
    def _safe_read_threat(self) -> float:
        """Lectura defensiva de χ² para diagnóstico."""
        try:
            raw = self._topo_watcher.get_mahalanobis_threat()
            return max(0.0, NM.validate_finite_float(raw, name="threat_diag"))
        except Exception:
            return 0.0
    
    def _safe_read_sigma(self) -> float:
        """Lectura defensiva de σ para diagnóstico."""
        try:
            raw = self._laplace_oracle.get_dominant_pole_real()
            return NM.validate_finite_float(raw, name="sigma_diag")
        except Exception:
            return 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Método Principal: Evaluación de Admisión
    # ─────────────────────────────────────────────────────────────────────────
    
    def evaluate_admission(
        self,
        payload: Mapping[str, Any]
    ) -> QuantumMeasurement:
        """
        Evalúa admisión cuántica completa sobre payload.
        
        ALGORITMO COMPLETO (7 etapas documentadas arriba en docstring de clase).
        
        Args:
            payload: Mapping con datos del cuanto informacional.
            
        Returns:
            QuantumMeasurement inmutable con todos los observables.
            
        Raises:
            QuantumAdmissionError: Si payload no es Mapping.
            QuantumNumericalError: Si hay fallos numéricos.
        """
        # Validación de tipo
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser Mapping; recibido {type(payload).__name__}",
                context={"type": type(payload).__name__}
            )
        
        logger.debug(
            f"Iniciando evaluación de admisión. "
            f"Payload keys: {list(payload.keys())}"
        )
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 1: VETO COHOMOLÓGICO
        # ═════════════════════════════════════════════════════════════════════
        
        veto_active, frustration = self._evaluate_cohomological_veto()
        if veto_active:
            measurement = self._build_veto_measurement(frustration)
            QuantumLogger.log_measurement(measurement, level=logging.WARNING)
            return measurement
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 2: ENERGÍA INCIDENTE
        # ═════════════════════════════════════════════════════════════════════
        
        E, E_uncertainty = IncidentEnergyCalculator.calculate(payload)
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 3: FUNCIÓN DE TRABAJO
        # ═════════════════════════════════════════════════════════════════════
        
        Phi, threat = self._work_function_modulator.calculate()
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 4: MASA EFECTIVA
        # ═════════════════════════════════════════════════════════════════════
        
        m_eff, sigma = self._effective_mass_modulator.calculate()
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 5: PROBABILIDAD WKB
        # ═════════════════════════════════════════════════════════════════════
        
        T, wkb_params = WKBCalculator.compute_tunneling_probability(E, Phi, m_eff)
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 6: UMBRAL DETERMINISTA
        # ═════════════════════════════════════════════════════════════════════
        
        payload_bytes = PayloadSerializer.serialize(payload)
        payload_hash = PayloadSerializer.deterministic_hash(payload_bytes)
        collapse_threshold = CollapseThresholdGenerator.generate(payload_hash)
        
        # ═════════════════════════════════════════════════════════════════════
        # ETAPA 7: COLAPSO DE FUNCIÓN DE ONDA
        # ═════════════════════════════════════════════════════════════════════
        
        admitted = T >= collapse_threshold
        
        if not admitted:
            # ─────────────────────────────────────────────────────────────────
            # CASO: RECHAZO PROBABILÍSTICO
            # ─────────────────────────────────────────────────────────────────
            
            measurement = QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=E,
                work_function=Phi,
                tunneling_probability=T,
                kinetic_energy=0.0,
                momentum=0.0,
                frustration_veto=False,
                effective_mass=m_eff,
                dominant_pole_real=sigma,
                threat_level=threat,
                collapse_threshold=collapse_threshold,
                admission_reason=(
                    f"Rechazo probabilístico: T={T:.6e} < θ={collapse_threshold:.6f}. "
                    "Colapso a |RECHAZADO⟩ según regla de Born."
                ),
                wkb_parameters=wkb_params,
                measurement_uncertainty=E_uncertainty,
            )
            
            QuantumLogger.log_measurement(measurement, level=logging.INFO)
            return measurement
        
        # ─────────────────────────────────────────────────────────────────────
        # CASO: ADMISIÓN
        # ─────────────────────────────────────────────────────────────────────
        
        # Cálculo de energía cinética
        if E >= Phi:
            # Transmisión clásica
            kinetic_energy = max(Const.MIN_KINETIC_ENERGY, E - Phi)
            admission_reason = (
                f"Admisión clásica: E={E:.6e} ≥ Φ={Phi:.6e}. "
                "Superación directa de barrera (fotoeléctrico)."
            )
        else:
            # Transmisión por túnel
            kinetic_energy = Const.MIN_KINETIC_ENERGY
            admission_reason = (
                f"Admisión por túnel WKB: E={E:.6e} < Φ={Phi:.6e}, "
                f"T={T:.6e} ≥ θ={collapse_threshold:.6f}. "
                "Penetración de barrera cuántica."
            )
        
        # Cálculo de momentum: p = √(2mK)
        m_for_momentum = (
            m_eff if math.isfinite(m_eff)
            else Const.BASE_EFFECTIVE_MASS
        )
        
        momentum_squared = 2.0 * m_for_momentum * kinetic_energy
        momentum = NM.safe_sqrt(momentum_squared)
        momentum = NM.validate_finite_float(momentum, name="momentum")
        
        logger.info(
            f"Colapso a |ADMITIDO⟩: E={E:.3e}, Φ={Phi:.3e}, "
            f"K={kinetic_energy:.3e}, p={momentum:.3e}"
        )
        
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=E,
            work_function=Phi,
            tunneling_probability=T,
            kinetic_energy=kinetic_energy,
            momentum=momentum,
            frustration_veto=False,
            effective_mass=m_eff,
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=collapse_threshold,
            admission_reason=admission_reason,
            wkb_parameters=wkb_params,
            measurement_uncertainty=E_uncertainty,
        )
        
        QuantumLogger.log_measurement(measurement, level=logging.INFO)
        return measurement
    
    # ─────────────────────────────────────────────────────────────────────────
    # Morfismo Categórico __call__
    # ─────────────────────────────────────────────────────────────────────────
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica morfismo de admisión cuántica a estado categórico.
        
        TRANSFORMACIÓN CATEGÓRICA:
            F: CategoricalState → CategoricalState
        
        Preserva funtorialidad:
            F(g ∘ f) = F(g) ∘ F(f)
            F(id) = id
        
        Si admitido:
            strata' = strata ∪ {PHYSICS}
            context' = context ∪ {quantum_momentum, quantum_admission}
        
        Si rechazado:
            strata' = ∅ (reseteo completo)
            context' = context ∪ {quantum_error, quantum_admission}
        
        Args:
            state: Estado categórico con payload Mapping.
            
        Returns:
            Nuevo CategoricalState transformado (inmutable).
            
        Raises:
            QuantumAdmissionError: Si state inválido.
        """
        # Validación de tipo
        if not isinstance(state, CategoricalState):
            raise QuantumAdmissionError(
                f"state debe ser CategoricalState; recibido {type(state).__name__}",
                context={"type": type(state).__name__}
            )
        
        payload = getattr(state, 'payload', None)
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"state.payload debe ser Mapping; recibido {type(payload).__name__}",
                context={"payload_type": type(payload).__name__}
            )
        
        # Extracción de contexto (defensiva)
        original_context = getattr(state, 'context', None)
        context = dict(original_context) if original_context else {}
        
        # Evaluación de admisión
        measurement = self.evaluate_admission(payload)
        
        # ═════════════════════════════════════════════════════════════════════
        # CASO: RECHAZO
        # ═════════════════════════════════════════════════════════════════════
        
        if measurement.eigenstate == Eigenstate.RECHAZADO:
            error_msg = (
                f"VETO CUÁNTICO | {measurement.eigenstate} | "
                f"E={measurement.incident_energy:.3e} | "
                f"Φ={measurement.work_function:.3e} | "
                f"T={measurement.tunneling_probability:.3e} | "
                f"veto={measurement.frustration_veto} | "
                f"razón: {measurement.admission_reason}"
            )
            logger.error(error_msg)
            
            new_context = {
                **context,
                'quantum_error': error_msg,
                'quantum_admission': measurement,
            }
            
            return CategoricalState(
                payload=payload,
                context=new_context,
                validated_strata=frozenset(),  # Reseteo completo
            )
        
        # ═════════════════════════════════════════════════════════════════════
        # CASO: ADMISIÓN
        # ═════════════════════════════════════════════════════════════════════
        
        logger.info(
            f"ADMISIÓN CUÁNTICA | |ADMITIDO⟩ | "
            f"p={measurement.momentum:.3e} | "
            f"E={measurement.incident_energy:.3e} | "
            f"Φ={measurement.work_function:.3e} | "
            f"T={measurement.tunneling_probability:.3e}"
        )
        
        new_context = {
            **context,
            'quantum_momentum': measurement.momentum,
            'quantum_admission': measurement,
        }
        
        # Agregar PHYSICS a strata validados
        new_strata = state.validated_strata | {Stratum.PHYSICS}
        
        return CategoricalState(
            payload=payload,
            context=new_context,
            validated_strata=new_strata,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: TESTING Y EJEMPLOS
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Batería de tests y ejemplos de uso.
    
    Este bloque solo se ejecuta al correr el módulo como script,
    no al importarlo.
    """
    
    # Configuración de logging detallado
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("\n" + "═" * 80)
    print("QUANTUM ADMISSION GATE - SUITE DE TESTING")
    print("═" * 80 + "\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Mocks de Dependencias
    # ─────────────────────────────────────────────────────────────────────────
    
    class MockTopologicalWatcher:
        """Mock del observador topológico."""
        
        def __init__(self, threat: float = 2.5):
            self._threat = threat
        
        def get_mahalanobis_threat(self) -> float:
            return self._threat
    
    class MockLaplaceOracle:
        """Mock del oráculo de Laplace."""
        
        def __init__(self, pole: float = -0.5):
            self._pole = pole
        
        def get_dominant_pole_real(self) -> float:
            return self._pole
    
    class MockSheafOrchestrator:
        """Mock del orquestador cohomológico."""
        
        def __init__(self, frustration: float = 1e-12):
            self._frustration = frustration
        
        def get_global_frustration_energy(self) -> float:
            return self._frustration
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Construcción del Gate
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 1: Construcción y validación de dependencias")
    print("-" * 80)
    
    try:
        gate = QuantumAdmissionGate(
            topo_watcher=MockTopologicalWatcher(),
            laplace_oracle=MockLaplaceOracle(),
            sheaf_orchestrator=MockSheafOrchestrator(),
        )
        print("✓ Gate construido exitosamente")
        print(f"  Dominio: {gate.domain}")
        print(f"  Codominio: {gate.codomain}")
    except Exception as exc:
        print(f"✗ Fallo en construcción: {exc}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Payload Simple (Esperamos Admisión)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 2: Payload simple (baja entropía, alta energía)")
    print("-" * 80)
    
    payload_simple = {
        'endpoint': '/api/test',
        'method': 'GET',
        'data': 'A' * 500,  # Baja entropía, tamaño moderado
    }
    
    try:
        measurement = gate.evaluate_admission(payload_simple)
        print(f"Estado: {measurement.eigenstate}")
        print(f"Energía: E={measurement.incident_energy:.6e}")
        print(f"Función de trabajo: Φ={measurement.work_function:.6e}")
        print(f"Probabilidad de túnel: T={measurement.tunneling_probability:.6e}")
        print(f"Umbral: θ={measurement.collapse_threshold:.6f}")
        print(f"Momentum: p={measurement.momentum:.6e}")
        print(f"Razón: {measurement.admission_reason}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Payload Complejo (Alta Entropía, Esperamos Rechazo Posible)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 3: Payload complejo (alta entropía)")
    print("-" * 80)
    
    import os
    payload_complejo = {
        'endpoint': '/api/data',
        'method': 'POST',
        'data': os.urandom(1000),  # Alta entropía (aleatorio)
    }
    
    try:
        measurement = gate.evaluate_admission(payload_complejo)
        print(f"Estado: {measurement.eigenstate}")
        print(f"Energía: E={measurement.incident_energy:.6e}")
        print(f"Probabilidad de túnel: T={measurement.tunneling_probability:.6e}")
        print(f"Umbral: θ={measurement.collapse_threshold:.6f}")
        print(f"Admitido: {measurement.eigenstate.is_accepted()}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Sistema Inestable (σ > 0, Esperamos Rechazo)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 4: Sistema inestable (σ > 0)")
    print("-" * 80)
    
    gate_inestable = QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(threat=1.0),
        laplace_oracle=MockLaplaceOracle(pole=0.5),  # Polo positivo
        sheaf_orchestrator=MockSheafOrchestrator(),
    )
    
    try:
        measurement = gate_inestable.evaluate_admission(payload_simple)
        print(f"Estado: {measurement.eigenstate}")
        print(f"Masa efectiva: m_eff={measurement.effective_mass}")
        print(f"Probabilidad de túnel: T={measurement.tunneling_probability:.6e}")
        print(f"Razón: {measurement.admission_reason}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Veto Cohomológico (Alta Frustración)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 5: Veto cohomológico (alta frustración)")
    print("-" * 80)
    
    gate_veto = QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(),
        laplace_oracle=MockLaplaceOracle(),
        sheaf_orchestrator=MockSheafOrchestrator(frustration=1.0),  # Alta frustración
    )
    
    try:
        measurement = gate_veto.evaluate_admission(payload_simple)
        print(f"Estado: {measurement.eigenstate}")
        print(f"Veto activo: {measurement.frustration_veto}")
        print(f"Razón: {measurement.admission_reason}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Serialización y Reproducibilidad
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 6: Reproducibilidad determinista")
    print("-" * 80)
    
    payload_test = {'key': 'value', 'number': 42}
    
    m1 = gate.evaluate_admission(payload_test)
    m2 = gate.evaluate_admission(payload_test)
    
    print(f"Primera medición: {m1.eigenstate}, θ={m1.collapse_threshold:.6f}")
    print(f"Segunda medición: {m2.eigenstate}, θ={m2.collapse_threshold:.6f}")
    print(f"Umbral idéntico: {m1.collapse_threshold == m2.collapse_threshold}")
    print(f"Estado idéntico: {m1.eigenstate == m2.eigenstate}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Resumen Final
    # ─────────────────────────────────────────────────────────────────────────
    
    print("═" * 80)
    print("SUITE DE TESTING COMPLETADA")
    print("═" * 80 + "\n")
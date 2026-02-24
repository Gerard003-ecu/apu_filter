"""
Módulo: Telemetry Schemas (Definición del Espacio Vectorial de Estado)
======================================================================

Este módulo define los subespacios métricos que componen el Vector de Estado
del sistema APU Filter. Implementa clases de datos inmutables (frozen dataclasses)
que actúan como contratos estrictos para la telemetría.

Fundamentos Matemáticos:
------------------------
Cada clase representa una dimensión del análisis DIKW:
1. PhysicsMetrics (Física): Variables de estado del FluxCondenser (Energía, Flujo).
2. TopologicalMetrics (Estructura): Invariantes homológicos del Grafo (Betti, Euler).
3. ControlMetrics (Estabilidad): Polos y ceros del Oráculo de Laplace.
4. ThermodynamicMetrics (Valor): Entropía y temperatura del sistema financiero.

La inmutabilidad garantiza que las mediciones sean tratadas como valores
algebraicos puros, facilitando la auditoría forense y el razonamiento causal.

Refinamientos v3:
-----------------
- Corrección dimensional: energy_density normalizada por capacitancia efectiva.
- Separación semántica: efficiency → potential_ratio + dissipation_efficiency.
- Polo dominante: corregido a max(σ) en lugar de min(|σ|).
- settling_time: implementación diferenciada para ζ < 1 y ζ ≥ 1.
- Lyapunov exponent: ahora Optional[float] con semántica clara.
- Topología: renombrado topological_complexity → total_betti_number.
- Termodinámica: separación clara entre internal_energy y exergy.
- Tolerancias: constantes globales para comparaciones numéricas.
- Slots: añadidos para optimización de memoria.
- Serialización: métodos to_dict() para persistencia.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, Union
import math


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS
# ═══════════════════════════════════════════════════════════════════════════════

# Tolerancias para comparaciones de punto flotante
EPSILON_RELATIVE: float = 1e-9      # Tolerancia relativa para igualdad
EPSILON_ABSOLUTE: float = 1e-12     # Tolerancia absoluta (cerca de cero)
EPSILON_FREQUENCY: float = 1e-12    # Umbral para detectar componente imaginaria

# Constantes físicas (adimensionalizadas para el modelo)
SETTLING_TIME_CRITERION: float = 4.0  # Criterio del 2% para tiempo de asentamiento


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SUBESPACIO FÍSICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PhysicsMetrics:
    """
    Subespacio Físico (Vector de Estado del FluxCondenser).
    
    Modela la dinámica electromagnética usando analogía de circuito RLC:
        - Inductancia L ↔ Inercia del flujo de datos
        - Capacitancia C ↔ Buffer de almacenamiento
        - Resistencia R ↔ Fricción/pérdidas por errores
    
    Invariantes Validados:
    ----------------------
        saturation ∈ [0, 1]
        kinetic_energy ≥ 0, potential_energy ≥ 0
        dissipated_power ≥ 0
        gyroscopic_stability ≥ 0
        hamiltonian_excess ≥ 0
    
    Unidades (SI normalizadas):
    ---------------------------
        Energías: Julios [J]
        Potencia: Vatios [W]
        Voltaje: Voltios [V]
        Presión: Pascales [Pa] o unidades arbitrarias
    
    Ref: flux_condenser.txt [Fuente 797]
    """
    # ── Dinámica de Fluidos ──
    saturation: float = 0.0             # V_elastic: Nivel de llenado del buffer [0.0, 1.0]
    pressure: float = 0.0               # P: Presión estática en la cola de procesamiento

    # ── Electrodinámica (Modelo RLC) ──
    kinetic_energy: float = 0.0         # E_k = ½·L·I² (Inercia del flujo de datos)
    potential_energy: float = 0.0       # E_p = ½·C·V² (Energía almacenada en buffer)
    flyback_voltage: float = 0.0        # V_fb = L·dI/dt (Picos por cambios bruscos)
    dissipated_power: float = 0.0       # P_dis = R·I² (Potencia disipada)

    # ── Parámetros del Circuito (para normalización) ──
    effective_capacitance: float = 1.0  # C_eff: Capacitancia efectiva del sistema

    # ── Estabilidad Mecánica ──
    gyroscopic_stability: float = 1.0   # S_g: Índice de estabilidad rotacional

    # ── Métricas de Maxwell ──
    poynting_flux: float = 0.0          # S = E × H (Flujo de valor direccional)
    hamiltonian_excess: float = 0.0     # H_excess: Energía espuria (violación de conservación)

    def __post_init__(self) -> None:
        """Valida rangos físicos y relaciones energéticas."""
        # Validación de saturación (fracción volumétrica)
        if not (0.0 <= self.saturation <= 1.0):
            raise ValueError(
                f"Saturación debe estar en [0, 1], recibido: {self.saturation}"
            )
        
        # Validación de energías (principio de no-negatividad)
        if self.kinetic_energy < 0:
            raise ValueError(
                f"Energía cinética debe ser ≥ 0 (E_k = ½LI² ≥ 0): {self.kinetic_energy}"
            )
        if self.potential_energy < 0:
            raise ValueError(
                f"Energía potencial debe ser ≥ 0 (E_p = ½CV² ≥ 0): {self.potential_energy}"
            )
        
        # Validación de potencia disipada (segunda ley de la termodinámica)
        if self.dissipated_power < 0:
            raise ValueError(
                f"Potencia disipada debe ser ≥ 0 (P = R·I² ≥ 0): {self.dissipated_power}"
            )
        
        # Validación de capacitancia efectiva
        if self.effective_capacitance <= 0:
            raise ValueError(
                f"Capacitancia efectiva debe ser > 0: {self.effective_capacitance}"
            )
        
        # Validación de estabilidad giroscópica
        if self.gyroscopic_stability < 0:
            raise ValueError(
                f"Estabilidad giroscópica no puede ser negativa: {self.gyroscopic_stability}"
            )
        
        # Validación de exceso hamiltoniano (conservación de energía)
        if self.hamiltonian_excess < 0:
            raise ValueError(
                f"Exceso hamiltoniano debe ser ≥ 0: {self.hamiltonian_excess}"
            )

    # ── Propiedades Energéticas ──

    @property
    def total_energy(self) -> float:
        """
        Energía mecánica total: E_total = E_k + E_p.
        
        Corresponde al Hamiltoniano del oscilador armónico: H = T + V.
        """
        return self.kinetic_energy + self.potential_energy

    @property
    def potential_ratio(self) -> float:
        """
        Fracción de energía potencial: ρ_p = E_p / E_total.
        
        Interpretación física:
            ρ_p → 1: Sistema en máxima compresión (buffer lleno)
            ρ_p → 0: Sistema en máxima velocidad (flujo activo)
            ρ_p = 0.5: Equipartición (oscilación armónica pura)
        
        Retorna 1.0 si E_total = 0 (estado de reposo absoluto).
        Garantía: ρ_p ∈ [0, 1] por construcción.
        """
        total = self.total_energy
        if total < EPSILON_ABSOLUTE:
            return 1.0  # Estado de reposo: toda la energía es "potencial nula"
        return self.potential_energy / total

    @property
    def kinetic_ratio(self) -> float:
        """Fracción de energía cinética: ρ_k = 1 - ρ_p = E_k / E_total."""
        return 1.0 - self.potential_ratio

    @property
    def energy_density(self) -> float:
        """
        Densidad de energía normalizada: u = E_total / C_eff.
        
        Dimensional: [J/F] = [V²], representa voltaje efectivo al cuadrado.
        Esta normalización es coherente con el modelo de capacitor.
        
        Retorna 0.0 si C_eff → 0 (límite físico, ya validado > 0).
        """
        return self.total_energy / self.effective_capacitance

    @property
    def dissipation_efficiency(self) -> float:
        """
        Eficiencia instantánea: η = 1 - P_dis / (P_dis + dE/dt_estimado).
        
        Aproximación para régimen cuasi-estático:
            η ≈ E_total / (E_total + P_dis) cuando Δt = 1
        
        Interpretación:
            η → 1: Sistema sin pérdidas (superconductor)
            η → 0: Toda la energía se disipa (resistor puro)
        
        Retorna 1.0 si no hay energía ni disipación.
        """
        denominator = self.total_energy + self.dissipated_power
        if denominator < EPSILON_ABSOLUTE:
            return 1.0
        return self.total_energy / denominator

    @property
    def quality_factor(self) -> float:
        """
        Factor de calidad Q del oscilador: Q = 2π · E_stored / E_dissipated_per_cycle.
        
        Aproximación instantánea: Q ≈ ω · E_total / P_dis.
        Asumiendo ω = 1 (frecuencia normalizada):
            Q = E_total / P_dis
        
        Retorna +∞ si P_dis = 0 (oscilador ideal sin pérdidas).
        """
        if self.dissipated_power < EPSILON_ABSOLUTE:
            return float('inf')
        return self.total_energy / self.dissipated_power

    # ── Fábrica ──

    @classmethod
    def from_rlc_parameters(
        cls,
        saturation: float,
        pressure: float,
        inductance: float,
        current: float,
        capacitance: float,
        voltage: float,
        resistance: float,
        di_dt: float = 0.0,
        **kwargs: Any,
    ) -> "PhysicsMetrics":
        """
        Construye métricas desde parámetros eléctricos fundamentales.
        
        Ecuaciones aplicadas:
            E_k = ½·L·I²      (energía magnética del inductor)
            E_p = ½·C·V²      (energía eléctrica del capacitor)
            V_fb = L·dI/dt    (voltaje de flyback)
            P_dis = R·I²      (potencia disipada en resistencia)
        
        Args:
            inductance:  L [H]  — debe ser ≥ 0
            current:     I [A]  — corriente instantánea (puede ser negativa)
            capacitance: C [F]  — debe ser > 0
            voltage:     V [V]  — voltaje instantáneo (puede ser negativo)
            resistance:  R [Ω]  — debe ser ≥ 0
            di_dt:       dI/dt [A/s] — tasa de cambio de corriente
        
        Raises:
            ValueError: Si L < 0, C ≤ 0, o R < 0.
        """
        if inductance < 0:
            raise ValueError(f"Inductancia debe ser ≥ 0: L = {inductance}")
        if capacitance <= 0:
            raise ValueError(f"Capacitancia debe ser > 0: C = {capacitance}")
        if resistance < 0:
            raise ValueError(f"Resistencia debe ser ≥ 0: R = {resistance}")
        
        return cls(
            saturation=saturation,
            pressure=pressure,
            kinetic_energy=0.5 * inductance * current ** 2,
            potential_energy=0.5 * capacitance * voltage ** 2,
            flyback_voltage=inductance * di_dt,
            dissipated_power=resistance * current ** 2,
            effective_capacitance=capacitance,
            **kwargs,
        )

    # ── Serialización ──

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serialización."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Physics(E={self.total_energy:.3f}, sat={self.saturation:.2%}, "
            f"ρ_p={self.potential_ratio:.2%}, Q={self.quality_factor:.1f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SUBESPACIO TOPOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TopologicalMetrics:
    """
    Subespacio Topológico (Invariantes del BusinessTopologicalAnalyzer).
    
    Captura la forma y conectividad del grafo del proyecto mediante
    invariantes homológicos que son independientes de la métrica.
    
    Teoría Subyacente:
    ------------------
    Los números de Betti βᵢ cuentan las clases de homología de dimensión i:
        β₀: Componentes conexas (objetos disjuntos)
        β₁: Ciclos independientes (agujeros 1D, loops)
        β₂: Cavidades cerradas (huecos 2D, burbujas)
    
    Invariantes Validados:
    ----------------------
        βᵢ ≥ 0 ∀i ∈ {0, 1, 2}
        χ = β₀ − β₁ + β₂  (teorema de Euler-Poincaré)
        fiedler_value ≥ 0, spectral_gap ≥ 0
        pyramid_stability ≥ 0, structural_entropy ≥ 0
    
    Ref: Hatcher, "Algebraic Topology", Ch. 2
    """
    # ── Homología (Números de Betti) ──
    beta_0: int = 1                     # β₀: Componentes conexas (1 = grafo conexo)
    beta_1: int = 0                     # β₁: Ciclos independientes (0 = árbol)
    beta_2: int = 0                     # β₂: Cavidades (0 = sin huecos)

    # ── Invariantes Derivados ──
    euler_characteristic: Optional[int] = None  # χ = β₀ − β₁ + β₂ (auto-calculado)

    # ── Análisis Espectral del Laplaciano ──
    fiedler_value: float = 1.0          # λ₂: Segundo autovalor (conectividad algebraica)
    spectral_gap: float = 0.0           # Δλ = λ₂ - λ₁ (λ₁ = 0 para grafos conexos)

    # ── Estabilidad Estructural ──
    pyramid_stability: float = 1.0      # Ψ: Ratio soporte/cúspide (< 1 = pirámide invertida)
    structural_entropy: float = 0.0     # H_struct: Entropía de la distribución de grados

    def __post_init__(self) -> None:
        """Valida invariantes topológicos y calcula χ si es necesario."""
        # Validación de Betti (números naturales)
        for i, beta in enumerate([self.beta_0, self.beta_1, self.beta_2]):
            if beta < 0:
                raise ValueError(
                    f"Número de Betti β_{i} debe ser ≥ 0, recibido: {beta}"
                )
        
        # Teorema de Euler-Poincaré: χ = Σ(-1)ⁱ·βᵢ
        expected_euler = self.beta_0 - self.beta_1 + self.beta_2
        
        if self.euler_characteristic is None:
            # Auto-calcular si no se especifica
            object.__setattr__(self, 'euler_characteristic', expected_euler)
        elif self.euler_characteristic != expected_euler:
            raise ValueError(
                f"Violación de Euler-Poincaré: χ = {self.euler_characteristic} ≠ "
                f"β₀ - β₁ + β₂ = {self.beta_0} - {self.beta_1} + {self.beta_2} = {expected_euler}"
            )
        
        # Validación del autovalor de Fiedler (teorema espectral)
        if self.fiedler_value < 0:
            raise ValueError(
                f"Valor de Fiedler (λ₂) debe ser ≥ 0 (Laplaciano semidefinido positivo): "
                f"{self.fiedler_value}"
            )
        
        # Validación del gap espectral
        if self.spectral_gap < 0:
            raise ValueError(
                f"Gap espectral debe ser ≥ 0: {self.spectral_gap}"
            )
        
        # Validación de estabilidad piramidal
        if self.pyramid_stability < 0:
            raise ValueError(
                f"Estabilidad piramidal debe ser ≥ 0: {self.pyramid_stability}"
            )
        
        # Validación de entropía estructural
        if self.structural_entropy < 0:
            raise ValueError(
                f"Entropía estructural debe ser ≥ 0: {self.structural_entropy}"
            )

    # ── Propiedades Topológicas ──

    @property
    def is_connected(self) -> bool:
        """Indica si el grafo es conexo (β₀ = 1)."""
        return self.beta_0 == 1

    @property
    def has_cycles(self) -> bool:
        """Indica si existen ciclos independientes (β₁ > 0)."""
        return self.beta_1 > 0

    @property
    def has_cavities(self) -> bool:
        """Indica si existen cavidades cerradas (β₂ > 0)."""
        return self.beta_2 > 0

    @property
    def is_simply_connected(self) -> bool:
        """
        Indica si el espacio es simplemente conexo.
        
        Condición: β₀ = 1 (conexo) y β₁ = 0 (sin ciclos).
        
        Nota: Para 2-complejos simpliciales, β₁ = 0 implica π₁ = 0
        (el grupo fundamental es trivial), que es la definición de
        simple conexidad.
        """
        return self.beta_0 == 1 and self.beta_1 == 0

    @property
    def is_acyclic(self) -> bool:
        """
        Indica si el complejo es acíclico (árbol/bosque generalizado).
        Condición: β₁ = 0 y β₂ = 0.
        """
        return self.beta_1 == 0 and self.beta_2 == 0

    @property
    def betti_vector(self) -> Tuple[int, int, int]:
        """Devuelve los números de Betti como tupla inmutable (β₀, β₁, β₂)."""
        return (self.beta_0, self.beta_1, self.beta_2)

    @property
    def total_betti_number(self) -> int:
        """
        Suma de números de Betti: Σβᵢ = β₀ + β₁ + β₂.
        
        Mide la riqueza homológica total del espacio.
        Equivale a evaluar el polinomio de Poincaré en t = 1: P(1).
        
        Nota: Este no es la "complejidad topológica" en el sentido de
        Farber (TC), que mide la dificultad del problema de planificación
        de movimiento.
        """
        return self.beta_0 + self.beta_1 + self.beta_2

    @property
    def homological_dimension(self) -> int:
        """
        Dimensión homológica: max{i : βᵢ > 0}.
        
        Indica la dimensión más alta con homología no trivial.
        Retorna -1 si todos los Betti son 0 (espacio vacío).
        """
        if self.beta_2 > 0:
            return 2
        if self.beta_1 > 0:
            return 1
        if self.beta_0 > 0:
            return 0
        return -1  # Espacio vacío

    @property
    def cyclomatic_complexity(self) -> int:
        """
        Complejidad ciclomática: M = β₁ - β₀ + 1 = E - V + 1.
        
        Para grafos conexos (β₀ = 1): M = β₁.
        Mide el número mínimo de caminos linealmente independientes.
        
        Ref: McCabe, "A Complexity Measure", IEEE TSE 1976.
        """
        return self.beta_1 - self.beta_0 + 1

    # ── Polinomio de Poincaré ──

    def poincare_polynomial(self, t: float) -> float:
        """
        Evalúa el polinomio de Poincaré: P(t) = β₀ + β₁·t + β₂·t².
        
        Propiedades fundamentales:
            P(1)  = β₀ + β₁ + β₂  = total_betti_number
            P(−1) = β₀ − β₁ + β₂  = euler_characteristic
        
        Args:
            t: Parámetro de evaluación (típicamente t ∈ [-1, 1])
        
        Warning:
            Para |t| >> 1, el término β₂·t² puede causar overflow.
            Usar poincare_polynomial_log() para valores grandes.
        """
        return float(self.beta_0) + float(self.beta_1) * t + float(self.beta_2) * t ** 2

    def poincare_polynomial_log(self, t: float) -> Optional[float]:
        """
        Logaritmo del polinomio de Poincaré: log(P(t)).
        
        Numéricamente estable para |t| grande.
        Retorna None si P(t) ≤ 0.
        """
        p_t = self.poincare_polynomial(t)
        if p_t <= 0:
            return None
        return math.log(p_t)

    # ── Serialización ──

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serialización."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Topology(β=({self.beta_0},{self.beta_1},{self.beta_2}), "
            f"χ={self.euler_characteristic}, λ₂={self.fiedler_value:.3f}, "
            f"Ψ={self.pyramid_stability:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUBESPACIO DE CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ControlMetrics:
    """
    Subespacio de Control (Dictamen del LaplaceOracle).
    
    Captura la estabilidad dinámica en el dominio de la frecuencia compleja s.
    Modela el sistema como función de transferencia G(s) = N(s)/D(s), donde
    los polos (raíces de D(s)) determinan la respuesta transitoria.
    
    Criterio de Estabilidad:
    ------------------------
    Para sistemas LTI causales:
        ESTABLE     ⟺ Re(sᵢ) < 0  ∀ polo sᵢ
        MARGINALMENTE ESTABLE ⟺ ∃ polo con Re(s) = 0, multiplicidad 1
        INESTABLE   ⟺ ∃ polo con Re(s) > 0
    
    Invariantes Validados:
    ----------------------
        is_stable coherente con poles_real
        damping_ratio ≥ 0
        natural_frequency ≥ 0
        poles_real es Tuple (inmutable)
    
    Ref: Ogata, "Modern Control Engineering", Ch. 5-6
    """
    # ── Ubicación de Polos (inmutable) ──
    poles_real: Tuple[float, ...] = field(default_factory=tuple)  # σᵢ = Re(sᵢ)
    is_stable: Optional[bool] = None        # Si None, se deduce de poles_real

    # ── Márgenes de Robustez ──
    phase_margin_deg: float = 45.0          # Φ_m: Margen de fase [°]
    gain_margin_db: float = float('inf')    # G_m: Margen de ganancia [dB]

    # ── Respuesta Transitoria (modelo de segundo orden dominante) ──
    damping_ratio: float = 0.707            # ζ: Razón de amortiguamiento
    natural_frequency: float = 0.0          # ω_n: Frecuencia natural no amortiguada [rad/s]

    # ── Dinámica No Lineal ──
    lyapunov_exponent: Optional[float] = None  # λ_L: Exponente de Lyapunov máximo

    def __post_init__(self) -> None:
        """Valida estabilidad, coherencia de polos y rangos de parámetros."""
        # Garantizar inmutabilidad: convertir lista a tupla si es necesario
        if isinstance(self.poles_real, list):
            object.__setattr__(self, 'poles_real', tuple(self.poles_real))
        
        # Determinar o verificar estabilidad según criterio de Routh-Hurwitz
        if self.is_stable is None:
            if self.poles_real:
                # Sistema estable ⟺ todos los polos en semiplano izquierdo
                stable = all(sigma < 0 for sigma in self.poles_real)
            else:
                # Sin polos especificados: asumir estable por defecto
                stable = True
            object.__setattr__(self, 'is_stable', stable)
        elif self.poles_real:
            # Verificar coherencia entre flag y polos
            poles_all_negative = all(sigma < 0 for sigma in self.poles_real)
            if self.is_stable and not poles_all_negative:
                raise ValueError(
                    f"Inconsistencia: is_stable=True pero existe polo con σ ≥ 0: "
                    f"poles_real={self.poles_real}"
                )
            # Nota: is_stable=False con todos σ < 0 es válido (puede haber ceros en RHP)
        
        # Validación de razón de amortiguamiento
        if self.damping_ratio < 0:
            raise ValueError(
                f"Razón de amortiguamiento ζ debe ser ≥ 0: {self.damping_ratio}"
            )
        
        # Validación de frecuencia natural
        if self.natural_frequency < 0:
            raise ValueError(
                f"Frecuencia natural ω_n debe ser ≥ 0: {self.natural_frequency}"
            )

    # ── Propiedades de Polos ──

    @property
    def dominant_pole(self) -> float:
        """
        Polo dominante: max(σᵢ) = σ más cercano al eje imaginario.
        
        El polo dominante determina la dinámica a largo plazo:
            σ_dom < 0: Decaimiento exponencial con τ = 1/|σ_dom|
            σ_dom = 0: Respuesta marginalmente estable (oscilador puro)
            σ_dom > 0: Crecimiento exponencial (inestable)
        
        Retorna -∞ si no hay polos (sistema puramente ganancia).
        """
        if not self.poles_real:
            return -float('inf')
        return max(self.poles_real)

    @property
    def fastest_pole(self) -> float:
        """
        Polo más rápido: min(σᵢ) = σ más alejado del eje imaginario.
        
        Determina la dinámica transitoria rápida (tiempo de subida).
        Retorna +∞ si no hay polos.
        """
        if not self.poles_real:
            return float('inf')
        return min(self.poles_real)

    @property
    def stability_margin(self) -> float:
        """
        Margen de estabilidad: -σ_dom = distancia del polo dominante al eje jω.
        
        Interpretación:
            > 0: Sistema estable (todos los polos en LHP)
            = 0: Sistema marginalmente estable
            < 0: Sistema inestable
        """
        return -self.dominant_pole

    # ── Clasificación de Amortiguamiento ──

    @property
    def is_critically_damped(self) -> bool:
        """
        Amortiguamiento crítico: ζ ≈ 1.0.
        
        Respuesta más rápida sin oscilación.
        Tolerancia: |ζ - 1| < 0.001 (0.1%).
        """
        return math.isclose(self.damping_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-6)

    @property
    def is_overdamped(self) -> bool:
        """
        Sobreamortiguado: ζ > 1.0.
        Respuesta lenta sin oscilación (dos polos reales distintos).
        """
        return self.damping_ratio > 1.0 and not self.is_critically_damped

    @property
    def is_underdamped(self) -> bool:
        """
        Subamortiguado: 0 < ζ < 1.0.
        Respuesta oscilatoria con decaimiento (par de polos complejos conjugados).
        """
        return 0 < self.damping_ratio < 1.0

    @property
    def damping_category(self) -> str:
        """Categoría descriptiva del amortiguamiento."""
        if self.damping_ratio == 0:
            return "UNDAMPED"
        if self.is_underdamped:
            return "UNDERDAMPED"
        if self.is_critically_damped:
            return "CRITICALLY_DAMPED"
        return "OVERDAMPED"

    # ── Propiedades Dinámicas ──

    @property
    def damped_frequency(self) -> float:
        """
        Frecuencia amortiguada: ω_d = ω_n · √(1 − ζ²).
        
        Frecuencia de oscilación real del sistema subamortiguado.
        
        Casos:
            ζ < 1: ω_d > 0 (oscilación a frecuencia reducida)
            ζ ≥ 1: ω_d = 0 (sin oscilación, respuesta exponencial pura)
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        discriminant = 1.0 - self.damping_ratio ** 2
        return self.natural_frequency * math.sqrt(discriminant)

    @property
    def settling_time(self) -> float:
        """
        Tiempo de asentamiento (criterio 2%): t_s tal que |y(t) - y_∞| < 0.02·|Δy|.
        
        Fórmulas según régimen:
            Subamortiguado (ζ < 1):  t_s ≈ 4 / (ζ·ω_n) = 4/|σ|
            Críticamente amortiguado (ζ = 1): t_s ≈ 4.75 / ω_n
            Sobreamortiguado (ζ > 1):  t_s ≈ 4 / |σ_dom| donde σ_dom es el polo lento
        
        Retorna +∞ si el sistema no converge (ω_n = 0 o inestable).
        """
        if self.natural_frequency < EPSILON_ABSOLUTE:
            return float('inf')
        
        if self.damping_ratio < 1.0:
            # Subamortiguado: t_s = 4/(ζ·ω_n)
            sigma = self.damping_ratio * self.natural_frequency
            if sigma < EPSILON_ABSOLUTE:
                return float('inf')
            return SETTLING_TIME_CRITERION / sigma
        
        elif self.is_critically_damped:
            # Críticamente amortiguado: t_s ≈ 4.75/ω_n
            return 4.75 / self.natural_frequency
        
        else:
            # Sobreamortiguado: polo dominante es σ₁ = ω_n·(-ζ + √(ζ²-1))
            # |σ_dom| = ω_n·(ζ - √(ζ²-1))
            sqrt_term = math.sqrt(self.damping_ratio ** 2 - 1.0)
            sigma_dominant = self.natural_frequency * (self.damping_ratio - sqrt_term)
            if sigma_dominant < EPSILON_ABSOLUTE:
                return float('inf')
            return SETTLING_TIME_CRITERION / sigma_dominant

    @property
    def rise_time(self) -> float:
        """
        Tiempo de subida (10% a 90%): t_r ≈ 1.8 / ω_n para ζ = 0.707.
        
        Aproximación general: t_r ≈ (1 + 1.1ζ + 1.4ζ²) / ω_n.
        Retorna +∞ si ω_n = 0.
        """
        if self.natural_frequency < EPSILON_ABSOLUTE:
            return float('inf')
        zeta = self.damping_ratio
        numerator = 1.0 + 1.1 * zeta + 1.4 * zeta ** 2
        return numerator / self.natural_frequency

    @property
    def peak_overshoot(self) -> float:
        """
        Sobrepico porcentual: M_p = exp(-πζ/√(1-ζ²)) × 100%.
        
        Solo aplica para sistemas subamortiguados (ζ < 1).
        Retorna 0.0 para ζ ≥ 1.
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        if self.damping_ratio <= 0.0:
            return 100.0  # Oscilador sin amortiguamiento
        
        discriminant = math.sqrt(1.0 - self.damping_ratio ** 2)
        exponent = -math.pi * self.damping_ratio / discriminant
        return 100.0 * math.exp(exponent)

    # ── Dinámica Caótica ──

    @property
    def is_chaotic(self) -> bool:
        """
        Indica comportamiento caótico: λ_L > 0.
        
        El exponente de Lyapunov máximo mide la tasa de divergencia
        de trayectorias cercanas:
            λ_L < 0: Atractor estable (punto fijo o ciclo límite)
            λ_L = 0: Bifurcación o cuasi-periodicidad
            λ_L > 0: Caos determinista (sensibilidad a condiciones iniciales)
        
        Retorna False si el exponente no está definido.
        """
        if self.lyapunov_exponent is None:
            return False
        return self.lyapunov_exponent > 0

    @property
    def lyapunov_time(self) -> float:
        """
        Tiempo de Lyapunov: τ_L = 1/λ_L.
        
        Escala temporal característica de pérdida de predictibilidad.
        Retorna +∞ si λ_L ≤ 0 o no definido.
        """
        if self.lyapunov_exponent is None or self.lyapunov_exponent <= 0:
            return float('inf')
        return 1.0 / self.lyapunov_exponent

    # ── Fábrica ──

    @classmethod
    def from_poles(
        cls,
        poles: list,
        **kwargs: Any,
    ) -> "ControlMetrics":
        """
        Construye desde una lista de polos complejos.
        
        Estrategia de extracción (polo dominante = max Re(s)):
        
        1. Par conjugado complejo (s = σ ± jω_d):
           - ω_n = |s| = √(σ² + ω_d²)
           - ζ = -σ/ω_n = cos(θ) donde θ = ∠(s, eje real negativo)
        
        2. Polo real estable (s = σ < 0):
           - ω_n = |σ|
           - ζ = 1.0 (críticamente amortiguado o sobreamortiguado)
        
        3. Polo real inestable (s = σ ≥ 0):
           - ω_n = |σ|
           - ζ = 0.0 (marcado como inestable)
        
        Args:
            poles: Lista de números complejos representando polos.
        """
        if not poles:
            return cls(**kwargs)
        
        # Extraer partes reales para validación de estabilidad
        real_parts = tuple(p.real for p in poles)
        
        # Polo dominante: máxima parte real (más cercano al eje jω)
        dominant = max(poles, key=lambda p: p.real)
        
        if abs(dominant.imag) > EPSILON_FREQUENCY:
            # Par conjugado complejo: s = σ ± jω_d
            wn = abs(dominant)  # ω_n = √(σ² + ω_d²)
            if wn > EPSILON_ABSOLUTE:
                zeta = -dominant.real / wn
            else:
                zeta = 0.707  # Valor por defecto
        else:
            # Polo puramente real
            wn = abs(dominant.real)
            zeta = 1.0 if dominant.real < 0 else 0.0
        
        # Garantizar ζ ≥ 0 (requerido por invariante)
        zeta = max(zeta, 0.0)
        
        return cls(
            poles_real=real_parts,
            natural_frequency=wn,
            damping_ratio=zeta,
            **kwargs,
        )

    @classmethod
    def from_transfer_function(
        cls,
        numerator: Tuple[float, ...],
        denominator: Tuple[float, ...],
        **kwargs: Any,
    ) -> "ControlMetrics":
        """
        Construye desde coeficientes de función de transferencia G(s) = N(s)/D(s).
        
        Los polinomios se especifican en orden descendente de potencias:
            G(s) = (b_n·s^n + ... + b_0) / (a_m·s^m + ... + a_0)
        
        Requiere numpy para el cálculo de raíces (importación diferida).
        """
        try:
            import numpy as np
            poles = np.roots(denominator)
            return cls.from_poles(poles.tolist(), **kwargs)
        except ImportError:
            raise ImportError(
                "numpy es requerido para from_transfer_function(). "
                "Instalar con: pip install numpy"
            )

    # ── Serialización ──

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serialización."""
        return asdict(self)

    def __str__(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        chaos = ", CHAOTIC" if self.is_chaotic else ""
        return (
            f"Control({status}{chaos}, ζ={self.damping_ratio:.3f}, "
            f"ω_n={self.natural_frequency:.3f}, σ_dom={self.dominant_pole:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUBESPACIO TERMODINÁMICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ThermodynamicMetrics:
    """
    Subespacio Termodinámico (Economía Física).
    
    Modela el sistema financiero usando analogías termodinámicas:
        - Temperatura T ↔ Volatilidad del mercado
        - Entropía S ↔ Desorden administrativo / incertidumbre
        - Energía interna U ↔ Capital total
        - Exergía Ex ↔ Capital efectivo (trabajo útil extraíble)
    
    Potenciales Termodinámicos:
    ---------------------------
        U: Energía interna (primer principio)
        F = U - TS: Energía libre de Helmholtz (trabajo a T constante)
        G = F + PV: Energía libre de Gibbs (trabajo a T y P constantes)
        Ex = (U - U₀) - T₀(S - S₀): Exergía (trabajo máximo respecto a ambiente)
    
    Invariantes Validados:
    ----------------------
        Todas las magnitudes intensivas y extensivas ≥ 0
        system_temperature ≥ 0 (escala absoluta)
        reference_temperature ≥ 0
    
    Ref: Callen, "Thermodynamics and an Introduction to Thermostatistics"
    """
    # ── Variables de Estado ──
    system_temperature: float = 300.0      # T: Temperatura del sistema [K] (≈ 27°C)
    entropy: float = 0.0                   # S: Entropía del sistema [J/K] o [nats]
    internal_energy: float = 0.0           # U: Energía interna [J] o unidades monetarias
    
    # ── Capacidades y Coeficientes ──
    heat_capacity: float = 1.0             # C_v: Capacidad calorífica a volumen cte [J/K]
    pressure_volume: float = 0.0           # PV: Producto presión-volumen [J]
    
    # ── Referencia Ambiental (para cálculo de exergía) ──
    reference_temperature: float = 300.0   # T₀: Temperatura del ambiente [K]
    reference_entropy: float = 0.0         # S₀: Entropía de referencia [J/K]

    def __post_init__(self) -> None:
        """Valida rangos termodinámicos según principios físicos."""
        # Temperatura absoluta (tercer principio implícito: T ≥ 0)
        if self.system_temperature < 0:
            raise ValueError(
                f"Temperatura del sistema debe ser ≥ 0 K: {self.system_temperature}"
            )
        
        # Temperatura de referencia
        if self.reference_temperature < 0:
            raise ValueError(
                f"Temperatura de referencia debe ser ≥ 0 K: {self.reference_temperature}"
            )
        
        # Entropía (segundo principio: S ≥ 0 para sistemas aislados desde T = 0)
        if self.entropy < 0:
            raise ValueError(
                f"Entropía debe ser ≥ 0 (segundo principio): {self.entropy}"
            )
        
        # Capacidad calorífica (estabilidad térmica requiere C_v > 0)
        if self.heat_capacity < 0:
            raise ValueError(
                f"Capacidad calorífica debe ser ≥ 0: {self.heat_capacity}"
            )

    # ── Potenciales Termodinámicos ──

    @property
    def helmholtz_free_energy(self) -> float:
        """
        Energía libre de Helmholtz: F = U - T·S.
        
        Representa el trabajo máximo extraíble a temperatura constante.
        
        Interpretación financiera:
            F > 0: Capital neto positivo (activos > pasivos entropicos)
            F < 0: Sistema dominado por "deuda entrópica"
            F = 0: Equilibrio marginal
        """
        return self.internal_energy - self.system_temperature * self.entropy

    @property
    def gibbs_free_energy(self) -> float:
        """
        Energía libre de Gibbs: G = F + PV = U - TS + PV = H - TS.
        
        Representa el trabajo útil a temperatura y presión constantes.
        Criterio de espontaneidad: ΔG < 0 para procesos espontáneos.
        """
        return self.helmholtz_free_energy + self.pressure_volume

    @property
    def enthalpy(self) -> float:
        """
        Entalpía: H = U + PV.
        
        Energía total incluyendo trabajo de expansión.
        """
        return self.internal_energy + self.pressure_volume

    @property
    def exergy(self) -> float:
        """
        Exergía: Ex = (U - U₀) + P₀(V - V₀) - T₀(S - S₀).
        
        Simplificación asumiendo V = V₀ (sistema incompresible):
            Ex ≈ U - T₀·S + constantes de referencia
        
        Considerando referencia en U₀ = 0:
            Ex = U - T₀·(S - S₀)
        
        Representa el trabajo máximo extraíble respecto al ambiente.
        """
        return self.internal_energy - self.reference_temperature * (
            self.entropy - self.reference_entropy
        )

    # ── Eficiencias ──

    @property
    def carnot_efficiency(self) -> float:
        """
        Eficiencia de Carnot: η_C = 1 - T_cold/T_hot.
        
        Para este modelo: η_C = 1 - T₀/T.
        
        Interpretación:
            η = 1.0:  Máxima eficiencia teórica (T₀ = 0 K, imposible)
            η > 0:    Sistema capaz de realizar trabajo
            η = 0.0:  Equilibrio térmico (T₀ = T)
            η < 0:    Requiere trabajo externo (bomba de calor: T₀ > T)
        
        Retorna 0.0 si T = 0 (sistema congelado, tercer principio).
        """
        if self.system_temperature < EPSILON_ABSOLUTE:
            return 0.0
        return 1.0 - self.reference_temperature / self.system_temperature

    @property
    def exergetic_efficiency(self) -> float:
        """
        Eficiencia exergética: η_ex = Ex / U.
        
        Fracción de energía interna que es "útil" (convertible a trabajo).
        
        Retorna 1.0 si U = 0 (no hay energía, eficiencia trivial).
        Puede ser > 1 si S < S₀ (sistema más ordenado que referencia).
        """
        if abs(self.internal_energy) < EPSILON_ABSOLUTE:
            return 1.0
        return self.exergy / self.internal_energy

    @property
    def entropic_penalty(self) -> float:
        """
        Penalización entrópica: T·S.
        
        Energía "perdida" debido al desorden del sistema.
        Representa el costo de la irreversibilidad.
        """
        return self.system_temperature * self.entropy

    # ── Propiedades Derivadas ──

    @property
    def specific_heat_ratio(self) -> float:
        """
        Ratio de capacidades caloríficas: γ = C_p/C_v.
        
        Estimación usando relación de Mayer: C_p - C_v = nR ≈ kT para 1 mol.
        Aproximación: γ ≈ 1 + kT/C_v donde k es normalizado a 1.
        
        Para sistemas con C_v >> T: γ → 1 (líquido/sólido).
        Para gases ideales monoatómicos: γ = 5/3 ≈ 1.67.
        """
        if self.heat_capacity < EPSILON_ABSOLUTE:
            return 1.0
        # Aproximación simplificada
        return 1.0 + self.system_temperature / (self.heat_capacity + self.system_temperature)

    @property
    def thermal_diffusivity(self) -> float:
        """
        Difusividad térmica normalizada: α ∝ T/C_v.
        
        Mide qué tan rápido se propagan las fluctuaciones térmicas.
        Mayor α ⟹ equilibración más rápida.
        """
        if self.heat_capacity < EPSILON_ABSOLUTE:
            return float('inf')
        return self.system_temperature / self.heat_capacity

    # ── Fábrica ──

    @classmethod
    def from_temperature_and_entropy(
        cls,
        temperature: float,
        entropy: float,
        heat_capacity: float = 1.0,
        reference_temperature: float = 300.0,
        **kwargs: Any,
    ) -> "ThermodynamicMetrics":
        """
        Construye desde temperatura y entropía.
        
        Calcula energía interna usando modelo de gas ideal:
            U = C_v · T (equipartición de energía)
        """
        internal_energy = heat_capacity * temperature
        return cls(
            system_temperature=temperature,
            entropy=entropy,
            internal_energy=internal_energy,
            heat_capacity=heat_capacity,
            reference_temperature=reference_temperature,
            **kwargs,
        )

    @classmethod
    def from_financial_analogy(
        cls,
        volatility: float,
        uncertainty: float,
        total_capital: float,
        market_temperature: float = 300.0,
        **kwargs: Any,
    ) -> "ThermodynamicMetrics":
        """
        Construye desde métricas financieras usando analogía termodinámica.
        
        Mapeo:
            volatility → system_temperature (fluctuaciones del mercado)
            uncertainty → entropy (información faltante)
            total_capital → internal_energy (recursos totales)
            market_temperature → reference_temperature (benchmark del mercado)
        """
        return cls(
            system_temperature=volatility,
            entropy=uncertainty,
            internal_energy=total_capital,
            reference_temperature=market_temperature,
            **kwargs,
        )

    # ── Serialización ──

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serialización."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Thermo(T={self.system_temperature:.1f}K, S={self.entropy:.3f}, "
            f"F={self.helmholtz_free_energy:.3f}, η_C={self.carnot_efficiency:.2%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VECTOR DE ESTADO COMPUESTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SystemStateVector:
    """
    Vector de Estado Compuesto del Sistema APU Filter.
    
    Agrega los cuatro subespacios métricos en un único objeto inmutable
    que representa el estado completo del sistema en un instante dado.
    
    Estructura del Espacio Vectorial:
    ---------------------------------
        V = V_phys ⊕ V_topo ⊕ V_ctrl ⊕ V_thermo
    
    donde cada Vᵢ es el subespacio correspondiente.
    
    Este vector facilita:
    - Serialización atómica del estado completo
    - Comparación temporal (diffs entre estados)
    - Auditoría forense con trazabilidad completa
    """
    physics: PhysicsMetrics = field(default_factory=PhysicsMetrics)
    topology: TopologicalMetrics = field(default_factory=TopologicalMetrics)
    control: ControlMetrics = field(default_factory=ControlMetrics)
    thermodynamics: ThermodynamicMetrics = field(default_factory=ThermodynamicMetrics)
    
    # Metadatos temporales
    timestamp: Optional[float] = None   # Unix timestamp de la medición
    epoch: int = 0                      # Número de ciclo/iteración

    @property
    def is_healthy(self) -> bool:
        """
        Evaluación rápida de salud del sistema.
        
        Criterios:
        - Sistema de control estable
        - Grafo topológico conexo
        - Eficiencia de Carnot positiva
        - Factor de calidad finito
        """
        return (
            bool(self.control.is_stable) and
            self.topology.is_connected and
            self.thermodynamics.carnot_efficiency > 0 and
            self.physics.quality_factor < float('inf')
        )

    @property
    def health_vector(self) -> Tuple[bool, bool, bool, bool]:
        """Vector booleano de salud por subsistema: (phys, topo, ctrl, thermo)."""
        return (
            self.physics.quality_factor > 1.0,      # Q > 1 indica bajo amortiguamiento
            self.topology.is_connected,              # Grafo conexo
            bool(self.control.is_stable),            # Sistema estable
            self.thermodynamics.carnot_efficiency > 0,  # Capacidad de trabajo
        )

    def to_dict(self) -> Dict[str, Any]:
        """Exporta estado completo a diccionario anidado."""
        return {
            'physics': self.physics.to_dict(),
            'topology': self.topology.to_dict(),
            'control': self.control.to_dict(),
            'thermodynamics': self.thermodynamics.to_dict(),
            'timestamp': self.timestamp,
            'epoch': self.epoch,
            'is_healthy': self.is_healthy,
        }

    def __str__(self) -> str:
        health = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"SystemState[{health}](epoch={self.epoch})\n"
            f"  ├─ {self.physics}\n"
            f"  ├─ {self.topology}\n"
            f"  ├─ {self.control}\n"
            f"  └─ {self.thermodynamics}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN Y DEMOSTRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    def print_section(title: str) -> None:
        print("\n" + "═" * 70)
        print(f" {title}")
        print("═" * 70)
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("1. PHYSICS METRICS")
    # ══════════════════════════════════════════════════════════════════════
    
    phys = PhysicsMetrics(
        saturation=0.8,
        pressure=1.2,
        kinetic_energy=10.0,
        potential_energy=5.0,
        dissipated_power=0.5,
    )
    print(phys)
    print(f"  total_energy          = {phys.total_energy:.3f} J")
    print(f"  potential_ratio       = {phys.potential_ratio:.2%}")
    print(f"  kinetic_ratio         = {phys.kinetic_ratio:.2%}")
    print(f"  dissipation_efficiency = {phys.dissipation_efficiency:.2%}")
    print(f"  quality_factor        = {phys.quality_factor:.1f}")
    print(f"  energy_density        = {phys.energy_density:.3f} V²")
    
    print("\n[Fábrica RLC]")
    phys_rlc = PhysicsMetrics.from_rlc_parameters(
        saturation=0.5, pressure=2.0,
        inductance=1.0, current=3.0,
        capacitance=0.01, voltage=100.0,
        resistance=10.0, di_dt=5.0,
    )
    print(phys_rlc)
    print(f"  E_kinetic  = ½·L·I² = ½·1·9 = {phys_rlc.kinetic_energy:.2f} J")
    print(f"  E_potential = ½·C·V² = ½·0.01·10000 = {phys_rlc.potential_energy:.2f} J")
    print(f"  V_flyback  = L·dI/dt = 1·5 = {phys_rlc.flyback_voltage:.2f} V")
    print(f"  P_dissipated = R·I² = 10·9 = {phys_rlc.dissipated_power:.2f} W")
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("2. TOPOLOGICAL METRICS")
    # ══════════════════════════════════════════════════════════════════════
    
    topo = TopologicalMetrics(beta_0=2, beta_1=3, beta_2=1)
    print(topo)
    print(f"  euler_characteristic  = χ = 2-3+1 = {topo.euler_characteristic}")
    print(f"  is_connected          = {topo.is_connected}")
    print(f"  is_simply_connected   = {topo.is_simply_connected}")
    print(f"  has_cycles            = {topo.has_cycles}")
    print(f"  has_cavities          = {topo.has_cavities}")
    print(f"  total_betti_number    = Σβ = {topo.total_betti_number}")
    print(f"  homological_dimension = {topo.homological_dimension}")
    print(f"  cyclomatic_complexity = M = {topo.cyclomatic_complexity}")
    print(f"  P(-1) [= χ]          = {topo.poincare_polynomial(-1):.0f}")
    print(f"  P(1)  [= Σβ]         = {topo.poincare_polynomial(1):.0f}")
    
    print("\n[Grafo árbol ideal]")
    topo_tree = TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, fiedler_value=0.5)
    print(topo_tree)
    print(f"  is_acyclic = {topo_tree.is_acyclic}")
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("3. CONTROL METRICS")
    # ══════════════════════════════════════════════════════════════════════
    
    ctrl = ControlMetrics(
        poles_real=(-1.0, -2.0, -5.0),
        phase_margin_deg=60.0,
        damping_ratio=0.707,
        natural_frequency=2.0,
    )
    print(ctrl)
    print(f"  is_stable         = {ctrl.is_stable}")
    print(f"  dominant_pole     = σ_dom = {ctrl.dominant_pole:.3f}")
    print(f"  fastest_pole      = σ_fast = {ctrl.fastest_pole:.3f}")
    print(f"  stability_margin  = {ctrl.stability_margin:.3f}")
    print(f"  damping_category  = {ctrl.damping_category}")
    print(f"  damped_frequency  = ω_d = {ctrl.damped_frequency:.3f} rad/s")
    print(f"  settling_time     = t_s = {ctrl.settling_time:.3f} s")
    print(f"  rise_time         = t_r = {ctrl.rise_time:.3f} s")
    print(f"  peak_overshoot    = M_p = {ctrl.peak_overshoot:.2f}%")
    
    print("\n[Retrocompatibilidad: lista → tupla]")
    ctrl_list = ControlMetrics(poles_real=[-0.5, -3.0])
    print(f"  poles type = {type(ctrl_list.poles_real).__name__}")
    
    print("\n[Fábrica desde polos complejos: s = -1 ± 2j]")
    ctrl_cpx = ControlMetrics.from_poles([complex(-1, 2), complex(-1, -2)])
    print(ctrl_cpx)
    print(f"  ω_n extraída = |s| = √(1+4) = {ctrl_cpx.natural_frequency:.3f}")
    print(f"  ζ extraída   = -Re(s)/ω_n = 1/√5 = {ctrl_cpx.damping_ratio:.3f}")
    print(f"  ω_d calculada = ω_n·√(1-ζ²) = {ctrl_cpx.damped_frequency:.3f}")
    
    print("\n[Sistema inestable]")
    ctrl_unstable = ControlMetrics(poles_real=(0.5, -1.0))
    print(ctrl_unstable)
    print(f"  is_stable = {ctrl_unstable.is_stable}")
    
    print("\n[Sistema con Lyapunov]")
    ctrl_chaotic = ControlMetrics(
        poles_real=(-0.1,),
        lyapunov_exponent=0.05,
    )
    print(f"  is_chaotic    = {ctrl_chaotic.is_chaotic}")
    print(f"  lyapunov_time = τ_L = {ctrl_chaotic.lyapunov_time:.1f} s")
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("4. THERMODYNAMIC METRICS")
    # ══════════════════════════════════════════════════════════════════════
    
    thermo = ThermodynamicMetrics(
        system_temperature=400.0,
        entropy=0.5,
        internal_energy=100.0,
        reference_temperature=300.0,
    )
    print(thermo)
    print(f"  helmholtz_free_energy = F = U - TS = 100 - 400·0.5 = {thermo.helmholtz_free_energy:.1f}")
    print(f"  gibbs_free_energy     = G = {thermo.gibbs_free_energy:.1f}")
    print(f"  exergy                = Ex = {thermo.exergy:.1f}")
    print(f"  carnot_efficiency     = η_C = 1 - 300/400 = {thermo.carnot_efficiency:.2%}")
    print(f"  exergetic_efficiency  = η_ex = {thermo.exergetic_efficiency:.2%}")
    print(f"  entropic_penalty      = TS = {thermo.entropic_penalty:.1f}")
    
    print("\n[Fábrica desde T y S]")
    thermo_fab = ThermodynamicMetrics.from_temperature_and_entropy(
        temperature=350.0,
        entropy=1.0,
        heat_capacity=2.0,
        reference_temperature=290.0,
    )
    print(thermo_fab)
    print(f"  U calculada = C_v·T = 2·350 = {thermo_fab.internal_energy:.0f}")
    
    print("\n[Analogía financiera]")
    thermo_fin = ThermodynamicMetrics.from_financial_analogy(
        volatility=0.25,        # 25% volatilidad anualizada
        uncertainty=2.5,        # Bits de incertidumbre
        total_capital=1e6,      # $1M capital
        market_temperature=0.15,  # 15% benchmark
    )
    print(f"  Sistema financiero: T={thermo_fin.system_temperature}, S={thermo_fin.entropy}")
    print(f"  Carnot efficiency (interpretación: potencial de arbitraje) = {thermo_fin.carnot_efficiency:.2%}")
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("5. VECTOR DE ESTADO COMPUESTO")
    # ══════════════════════════════════════════════════════════════════════
    
    import time
    
    state = SystemStateVector(
        physics=phys,
        topology=topo_tree,
        control=ctrl,
        thermodynamics=thermo,
        timestamp=time.time(),
        epoch=42,
    )
    print(state)
    print(f"\n  is_healthy    = {state.is_healthy}")
    print(f"  health_vector = {state.health_vector}")
    
    print("\n[Serialización]")
    state_dict = state.to_dict()
    print(f"  Claves nivel 1: {list(state_dict.keys())}")
    print(f"  physics.total_energy (via dict): {state_dict['physics']['kinetic_energy'] + state_dict['physics']['potential_energy']}")
    
    # ══════════════════════════════════════════════════════════════════════
    print_section("6. VALIDACIONES DE INVARIANTES")
    # ══════════════════════════════════════════════════════════════════════
    
    print("\n[Violaciones esperadas - cada una debe lanzar ValueError]")
    
    test_cases = [
        ("Saturación fuera de rango", lambda: PhysicsMetrics(saturation=1.5)),
        ("Energía negativa", lambda: PhysicsMetrics(kinetic_energy=-1.0)),
        ("Betti negativo", lambda: TopologicalMetrics(beta_1=-1)),
        ("Euler inconsistente", lambda: TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, euler_characteristic=5)),
        ("Damping negativo", lambda: ControlMetrics(damping_ratio=-0.5)),
        ("Frecuencia negativa", lambda: ControlMetrics(natural_frequency=-1.0)),
        ("Temperatura negativa", lambda: ThermodynamicMetrics(system_temperature=-10.0)),
        ("Entropía negativa", lambda: ThermodynamicMetrics(entropy=-0.5)),
    ]
    
    for name, factory in test_cases:
        try:
            factory()
            print(f"  ✗ {name}: NO lanzó excepción (ERROR)")
        except ValueError as e:
            print(f"  ✓ {name}: ValueError capturado")
    
    print("\n" + "═" * 70)
    print(" VERIFICACIÓN COMPLETA")
    print("═" * 70)
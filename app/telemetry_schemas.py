"""
Módulo: Telemetry Schemas (Definición del Espacio Vectorial de Estado)
======================================================================

Este módulo define los subespacios métricos que componen el Vector de Estado
del sistema APU Filter. Implementa clases de datos inmutables (frozen dataclasses)
que actúan como contratos estrictos para la telemetría.

Fundamentos:
------------
Cada clase representa una dimensión del análisis DIKW:
1. PhysicsMetrics (Física): Variables de estado del FluxCondenser (Energía, Flujo).
2. TopologicalMetrics (Estructura): Invariantes homológicos del Grafo (Betti, Euler).
3. ControlMetrics (Estabilidad): Polos y ceros del Oráculo de Laplace.
4. ThermodynamicMetrics (Valor): Entropía y temperatura del sistema financiero.

La inmutabilidad garantiza que las mediciones sean tratadas como valores
algebraicos puros, facilitando la auditoría forense y el razonamiento causal.

Características:
- Validación post-inicialización para asegurar invariantes matemáticos.
- Propiedades calculadas que derivan magnitudes compuestas.
- Métodos de fábrica para construcción desde fuentes externas.
- Representación compacta (__str__) para depuración y verbosa (__repr__) para serialización.
- Inmutabilidad estricta: todos los campos, incluidas colecciones, son inmutables.

Refinamientos v2:
- Corrección de inmutabilidad: poles_real usa Tuple en lugar de List.
- Validaciones faltantes: dissipated_power, structural_entropy, spectral_gap, natural_frequency.
- Corrección lógica: validación redundante de gain_margin_db eliminada.
- thermal_efficiency implementada con fórmula de Carnot parametrizable.
- Propiedades derivadas: damped_frequency, settling_time, poincare_polynomial, energy_density.
- Métodos __str__ compactos para depuración.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SUBESPACIO FÍSICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PhysicsMetrics:
    """
    Subespacio Físico (Vector de Estado del FluxCondenser).
    Captura la dinámica de fluidos y electromagnética del procesamiento.

    Invariantes validados:
        saturation ∈ [0, 1]
        kinetic_energy ≥ 0, potential_energy ≥ 0
        dissipated_power ≥ 0
        gyroscopic_stability ≥ 0

    Ref: flux_condenser.txt [Fuente 797]
    """
    # ── Dinámica de Fluidos ──
    saturation: float = 0.0            # V_elastic: Nivel de llenado del buffer [0.0, 1.0]
    pressure: float = 0.0             # Presión estática en la cola de procesamiento

    # ── Electrodinámica (Modelo RLC) ──
    kinetic_energy: float = 0.0       # E_k = ½·L·I² (Inercia del flujo de datos)
    potential_energy: float = 0.0     # E_p = ½·C·V² (Energía almacenada en buffer)
    flyback_voltage: float = 0.0      # V_fb = L·dI/dt (Picos por cambios bruscos de esquema)
    dissipated_power: float = 0.0     # P_dis = R·I² (Entropía generada por fricción/errores)
    hamiltonian_excess: float = 0.0   # H_err: Violación de conservación de energía

    # ── Estabilidad Mecánica ──
    gyroscopic_stability: float = 1.0  # S_g: Estabilidad rotacional del 'trompo' de datos

    # ── Métricas de Maxwell ──
    poynting_flux: float = 0.0         # S = E × H (Flujo de valor direccional)

    def __post_init__(self) -> None:
        """Valida rangos físicos y relaciones energéticas."""
        if not (0.0 <= self.saturation <= 1.0):
            raise ValueError(
                f"Saturation must be in [0, 1], got {self.saturation}"
            )
        if self.kinetic_energy < 0 or self.potential_energy < 0:
            raise ValueError(
                f"Energies must be non-negative: "
                f"kinetic={self.kinetic_energy}, potential={self.potential_energy}"
            )
        if self.dissipated_power < 0:
            raise ValueError(
                f"Dissipated power must be non-negative (P = R·I² ≥ 0): "
                f"{self.dissipated_power}"
            )
        if self.hamiltonian_excess < 0:
            raise ValueError(
                f"Hamiltonian excess must be non-negative (energy conservation violation): "
                f"{self.hamiltonian_excess}"
            )
        if self.gyroscopic_stability < 0:
            raise ValueError(
                f"Gyroscopic stability cannot be negative: "
                f"{self.gyroscopic_stability}"
            )

    # ── Propiedades Derivadas ──

    @property
    def total_energy(self) -> float:
        """Energía mecánica total: E_total = E_k + E_p."""
        return self.kinetic_energy + self.potential_energy

    @property
    def efficiency(self) -> float:
        """
        Rendimiento energético: η = E_p / E_total.

        Interpretación: fracción de energía almacenada (útil) respecto al total.
        Retorna 1.0 en estado ocioso (E_total = 0 ⟹ sin disipación).
        Garantía: η ∈ [0, 1] por construcción (ambas energías ≥ 0).
        """
        total = self.total_energy
        if total == 0.0:
            return 1.0
        return self.potential_energy / total

    @property
    def energy_density(self) -> float:
        """
        Densidad de energía normalizada: ρ_E = E_total / |P|.
        Retorna 0.0 si la presión es nula (sistema en vacío).
        """
        if self.pressure == 0.0:
            return 0.0
        return self.total_energy / abs(self.pressure)

    # ── Fábrica ──

    @classmethod
    def from_components(
        cls,
        saturation: float,
        pressure: float,
        inductance: float,
        current: float,
        capacitance: float,
        voltage: float,
        resistance: float,
        di_dt: float = 0.0,
        **kwargs,
    ) -> "PhysicsMetrics":
        """
        Construye una instancia a partir de parámetros eléctricos básicos.
        Calcula energías, potencia disipada y voltaje flyback según modelo RLC.

        Args:
            inductance:  L (henrios)  — debe ser ≥ 0
            current:     I (amperios) — corriente instantánea
            capacitance: C (faradios) — debe ser ≥ 0
            voltage:     V (voltios)  — voltaje instantáneo
            resistance:  R (ohmios)   — debe ser ≥ 0
            di_dt:       dI/dt        — tasa de cambio de corriente (para flyback)

        Raises:
            ValueError: Si L, C o R son negativos.
        """
        if inductance < 0 or capacitance < 0 or resistance < 0:
            raise ValueError(
                f"RLC parameters must be non-negative: "
                f"L={inductance}, C={capacitance}, R={resistance}"
            )
        return cls(
            saturation=saturation,
            pressure=pressure,
            kinetic_energy=0.5 * inductance * current ** 2,
            potential_energy=0.5 * capacitance * voltage ** 2,
            flyback_voltage=inductance * di_dt,
            dissipated_power=resistance * current ** 2,
            **kwargs,
        )

    def __str__(self) -> str:
        return (
            f"Physics(E={self.total_energy:.3f}, sat={self.saturation:.2f}, "
            f"η={self.efficiency:.2%}, S_g={self.gyroscopic_stability:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SUBESPACIO TOPOLÓGICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TopologicalMetrics:
    """
    Subespacio Topológico (Invariantes del BusinessTopologicalAnalyzer).
    Captura la forma y conectividad del grafo del proyecto.

    Invariantes validados:
        βᵢ ≥ 0 ∀i ∈ {0, 1, 2}
        χ = β₀ − β₁ + β₂ (coherencia Euler-Poincaré)
        fiedler_value ≥ 0, spectral_gap ≥ 0
        pyramid_stability ≥ 0, structural_entropy ≥ 0

    """
    # ── Homología (Números de Betti) ──
    beta_0: int = 1                    # Componentes conexas (1 = ideal)
    beta_1: int = 0                    # Ciclos independientes (0 = ideal)
    beta_2: int = 0                    # Cavidades (estructuras vacías)

    # ── Invariantes Derivados ──
    euler_characteristic: Optional[int] = None  # χ = β₀ − β₁ + β₂ (auto-calculado si None)
    mayer_vietoris_delta: int = 0      # Δ_mv: Ciclos espurios por fusión

    # ── Análisis Espectral ──
    fiedler_value: float = 1.0         # λ₂: Conectividad algebraica
    spectral_gap: float = 0.0         # Δλ: Diferencia entre primeros autovalores

    # ── Estabilidad Estructural ──
    pyramid_stability: float = 1.0    # Ψ: Índice soporte base/cúspide (<1 = invertida)
    structural_entropy: float = 0.0   # H_struct: Desorden en la red

    def __post_init__(self) -> None:
        """Valida invariantes topológicos y calcula χ si es necesario."""
        if self.beta_0 < 0 or self.beta_1 < 0 or self.beta_2 < 0:
            raise ValueError(
                f"Betti numbers cannot be negative: "
                f"(β₀={self.beta_0}, β₁={self.beta_1}, β₂={self.beta_2})"
            )
        if self.mayer_vietoris_delta < 0:
            raise ValueError(
                f"Mayer-Vietoris delta must be non-negative (cycle count), got {self.mayer_vietoris_delta}"
            )

        # Relación de Euler-Poincaré: χ = Σ(-1)ⁱ·βᵢ
        expected_euler = self.beta_0 - self.beta_1 + self.beta_2
        if self.euler_characteristic is None:
            object.__setattr__(self, 'euler_characteristic', expected_euler)
        elif self.euler_characteristic != expected_euler:
            raise ValueError(
                f"Euler characteristic {self.euler_characteristic} ≠ {expected_euler} "
                f"(calculated from Betti: β₀ − β₁ + β₂)"
            )

        if self.fiedler_value < 0:
            raise ValueError(
                f"Fiedler value (λ₂) must be non-negative, got {self.fiedler_value}"
            )
        if self.spectral_gap < 0:
            raise ValueError(
                f"Spectral gap must be non-negative, got {self.spectral_gap}"
            )
        if self.pyramid_stability < 0:
            raise ValueError(
                f"Pyramid stability must be non-negative, got {self.pyramid_stability}"
            )
        if self.structural_entropy < 0:
            raise ValueError(
                f"Structural entropy must be non-negative, got {self.structural_entropy}"
            )

    # ── Propiedades Derivadas ──

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
        """
        return self.beta_0 == 1 and self.beta_1 == 0

    @property
    def betti_vector(self) -> Tuple[int, int, int]:
        """Devuelve los números de Betti como tupla inmutable (β₀, β₁, β₂)."""
        return (self.beta_0, self.beta_1, self.beta_2)

    @property
    def topological_complexity(self) -> int:
        """
        Complejidad topológica total: Σβᵢ = β₀ + β₁ + β₂.
        Mide la riqueza estructural del espacio.
        Equivale a P(1) del polinomio de Poincaré.
        """
        return self.beta_0 + self.beta_1 + self.beta_2

    def poincare_polynomial(self, t: float) -> float:
        """
        Evalúa el polinomio de Poincaré: P(t) = β₀ + β₁·t + β₂·t².

        Función generatriz que codifica la homología completa:
            P(1)  = β₀ + β₁ + β₂  = topological_complexity
            P(−1) = β₀ − β₁ + β₂  = euler_characteristic
        """
        return self.beta_0 + self.beta_1 * t + self.beta_2 * t ** 2

    def __str__(self) -> str:
        return (
            f"Topology(β=({self.beta_0},{self.beta_1},{self.beta_2}), "
            f"χ={self.euler_characteristic}, λ₂={self.fiedler_value:.3f}, "
            f"Ψ={self.pyramid_stability:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUBESPACIO DE CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ControlMetrics:
    """
    Subespacio de Control (Dictamen del LaplaceOracle).
    Captura la estabilidad dinámica en el dominio de la frecuencia compleja.

    Invariantes validados:
        is_stable coherente con poles_real (si ambos se especifican)
        damping_ratio ≥ 0, natural_frequency ≥ 0
        poles_real es Tuple (inmutable) para preservar el contrato frozen

    Ref: laplace_oracle.txt [Fuente 930]
    """
    # ── Ubicación de Polos (inmutable) ──
    poles_real: Tuple[float, ...] = field(default_factory=tuple)  # σᵢ: Partes reales
    is_stable: Optional[bool] = None        # Si None, se deduce de poles_real

    # ── Márgenes de Robustez ──
    phase_margin_deg: float = 45.0          # Φ_m: Margen de fase en grados
    gain_margin_db: float = float('inf')    # G_m: Margen de ganancia en dB

    # ── Respuesta Transitoria ──
    damping_ratio: float = 0.707            # ζ: Amortiguamiento (crítico = 1.0)
    natural_frequency: float = 0.0          # ω_n: Frecuencia natural [rad/s]

    # ── Caos ──
    lyapunov_exponent: float = -1.0         # λ_L: Convergencia (<0) o Caos (>0)

    def __post_init__(self) -> None:
        """Valida estabilidad, coherencia de polos y rangos de parámetros."""
        # Garantizar inmutabilidad: convertir lista a tupla si es necesario
        if isinstance(self.poles_real, list):
            object.__setattr__(self, 'poles_real', tuple(self.poles_real))

        # Determinar o verificar estabilidad
        if self.is_stable is None:
            stable = (
                all(sigma < 0 for sigma in self.poles_real)
                if self.poles_real
                else True
            )
            object.__setattr__(self, 'is_stable', stable)
        elif self.poles_real:
            poles_all_negative = all(sigma < 0 for sigma in self.poles_real)
            if self.is_stable != poles_all_negative:
                raise ValueError(
                    f"is_stable={self.is_stable} inconsistent with poles_real: "
                    f"all σ < 0 is {poles_all_negative}"
                )

        if self.damping_ratio < 0:
            raise ValueError(
                f"Damping ratio must be ≥ 0, got {self.damping_ratio}"
            )
        if self.natural_frequency < 0:
            raise ValueError(
                f"Natural frequency must be ≥ 0, got {self.natural_frequency}"
            )

    # ── Propiedades de Polos ──

    @property
    def max_real_pole(self) -> float:
        """Máxima parte real de los polos (modo dominante). −∞ si no hay polos."""
        return max(self.poles_real) if self.poles_real else -float('inf')

    # ── Clasificación de Amortiguamiento ──

    @property
    def is_critically_damped(self) -> bool:
        """Amortiguamiento crítico (ζ ≈ 1.0, tolerancia relativa 0.1%)."""
        return math.isclose(self.damping_ratio, 1.0, rel_tol=1e-3)

    @property
    def is_overdamped(self) -> bool:
        """Sobreamortiguado (ζ > 1.0)."""
        return self.damping_ratio > 1.0

    @property
    def is_underdamped(self) -> bool:
        """Subamortiguado (0 < ζ < 1.0)."""
        return 0 < self.damping_ratio < 1.0

    # ── Propiedades Dinámicas ──

    @property
    def is_chaotic(self) -> bool:
        """Indica comportamiento caótico (exponente de Lyapunov λ_L > 0)."""
        return self.lyapunov_exponent > 0

    @property
    def damped_frequency(self) -> float:
        """
        Frecuencia amortiguada: ω_d = ω_n · √(1 − ζ²).

        Solo significativa para sistemas subamortiguados (ζ < 1).
        Retorna 0.0 para ζ ≥ 1 (respuesta sin oscilación).
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        return self.natural_frequency * math.sqrt(1.0 - self.damping_ratio ** 2)

    @property
    def settling_time(self) -> float:
        """
        Tiempo de asentamiento (criterio 2%): t_s ≈ 4 / (ζ · ω_n).
        Retorna +∞ si ζ·ω_n = 0 (sistema sin convergencia definida).
        """
        product = self.damping_ratio * self.natural_frequency
        if product == 0.0:
            return float('inf')
        return 4.0 / product

    # ── Fábrica ──

    @classmethod
    def from_poles(cls, poles: list, **kwargs) -> "ControlMetrics":
        """
        Construye desde una lista de polos complejos.

        Estrategia de extracción (polo dominante = menor |σ|):
        - Par conjugado (|ω_d| > ε): ω_n = |s|, ζ = −σ/ω_n
        - Polo real puro estable (σ < 0): ω_n = |σ|, ζ = 1.0
        - Polo real puro inestable (σ ≥ 0): ω_n = |σ|, ζ = 0.0
        """
        if not poles:
            return cls(**kwargs)

        real_parts = tuple(p.real for p in poles)

        # Polo dominante: más cercano al eje imaginario
        dominant = min(poles, key=lambda p: abs(p.real))

        if abs(dominant.imag) > 1e-12:
            # Par conjugado complejo: s = −σ ± jω_d
            wn = abs(dominant)
            zeta = -dominant.real / wn if wn > 0 else 0.707
        else:
            # Polo puramente real
            wn = abs(dominant.real)
            zeta = 1.0 if dominant.real < 0 else 0.0

        # Asegurar ζ ≥ 0 (requerido por el invariante de la clase)
        zeta = max(zeta, 0.0)

        return cls(
            poles_real=real_parts,
            natural_frequency=wn,
            damping_ratio=zeta,
            **kwargs,
        )

    def __str__(self) -> str:
        tag = "STABLE" if self.is_stable else "UNSTABLE"
        return (
            f"Control({tag}, ζ={self.damping_ratio:.3f}, "
            f"ω_n={self.natural_frequency:.3f}, σ_max={self.max_real_pole:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUBESPACIO TERMODINÁMICO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThermodynamicMetrics:
    """
    Subespacio Termodinámico (Economía Física).
    Captura la temperatura y eficiencia energética del sistema financiero.

    Invariantes validados:
        Todas las magnitudes escalares ≥ 0
        reference_temperature ≥ 0 (escala absoluta)

    Ref: semantic_translator.txt [Fuente 1291]
    """
    system_temperature: float = 25.0       # T: Volatilidad agregada (Kelvin financiero)
    entropy: float = 0.0                   # S: Grado de desorden administrativo
    exergy: float = 1.0                    # Ex: Energía útil disponible (presupuesto efectivo)
    heat_capacity: float = 0.5            # C_v: Capacidad de absorber sobrecostos
    reference_temperature: float = 0.0    # T₀: Temperatura del entorno de referencia
    financial_inertia: float = 1.0        # I_fin: Resistencia al cambio (1.0 = estable)

    def __post_init__(self) -> None:
        """Valida rangos termodinámicos."""
        if self.system_temperature < 0:
            raise ValueError(
                f"System temperature cannot be negative: {self.system_temperature}"
            )
        if self.entropy < 0:
            raise ValueError(
                f"Entropy cannot be negative: {self.entropy}"
            )
        if self.exergy < 0:
            raise ValueError(
                f"Exergy cannot be negative: {self.exergy}"
            )
        if self.heat_capacity < 0:
            raise ValueError(
                f"Heat capacity cannot be negative: {self.heat_capacity}"
            )
        if self.reference_temperature < 0:
            raise ValueError(
                f"Reference temperature cannot be negative: "
                f"{self.reference_temperature}"
            )
        if self.financial_inertia < 0:
            raise ValueError(
                f"Financial inertia must be non-negative: {self.financial_inertia}"
            )

    # ── Propiedades Derivadas ──

    @property
    def free_energy(self) -> float:
        """
        Energía libre de Helmholtz: F = U − T·S.

        Donde U ≈ exergía (aproximación del modelo).
        F < 0 indica sistema dominado por entropía (desorden supera energía útil).
        """
        return self.exergy - self.system_temperature * self.entropy

    @property
    def thermal_efficiency(self) -> float:
        """
        Eficiencia de Carnot: η_C = 1 − T₀/T.

        Interpretación:
            η = 1.0:  eficiencia máxima teórica (T₀ = 0).
            η > 0:    sistema capaz de realizar trabajo útil.
            η = 0.0:  equilibrio térmico (T₀ = T) o T = 0 (congelado).
            η < 0:    sistema requiere inyección externa de energía (T₀ > T).

        Retorna 0.0 si T = 0 (sistema congelado, sin capacidad de trabajo).
        """
        if self.system_temperature == 0.0:
            return 0.0
        return 1.0 - self.reference_temperature / self.system_temperature

    # ── Fábrica ──

    @classmethod
    def from_entropy_and_temp(
        cls,
        temperature: float,
        entropy: float,
        heat_capacity: float,
        reference_temperature: float = 0.0,
        **kwargs,
    ) -> "ThermodynamicMetrics":
        """
        Construye desde temperatura y entropía.
        Calcula exergía como U ≈ C_v · T (energía interna, modelo gas ideal).
        """
        exergy = heat_capacity * temperature
        return cls(
            system_temperature=temperature,
            entropy=entropy,
            heat_capacity=heat_capacity,
            exergy=exergy,
            reference_temperature=reference_temperature,
            **kwargs,
        )

    def __str__(self) -> str:
        return (
            f"Thermo(T={self.system_temperature:.1f}, S={self.entropy:.3f}, "
            f"F={self.free_energy:.3f}, η_C={self.thermal_efficiency:.2%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN RÁPIDA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("PHYSICS METRICS")
    print("=" * 60)

    phys = PhysicsMetrics(
        saturation=0.8, pressure=1.2,
        kinetic_energy=10.0, potential_energy=5.0,
    )
    print(phys)
    print(f"  total_energy   = {phys.total_energy}")        # 15.0
    print(f"  efficiency     = {phys.efficiency:.2%}")       # 33.33%
    print(f"  energy_density = {phys.energy_density:.3f}")   # 12.500
    print()

    phys_rlc = PhysicsMetrics.from_components(
        saturation=0.5, pressure=2.0,
        inductance=1.0, current=3.0,
        capacitance=0.01, voltage=100.0,
        resistance=10.0, di_dt=5.0,
    )
    print(phys_rlc)
    print(f"  flyback_voltage = {phys_rlc.flyback_voltage}")  # 5.0
    print()

    print("=" * 60)
    print("TOPOLOGICAL METRICS")
    print("=" * 60)

    topo = TopologicalMetrics(beta_0=2, beta_1=1, beta_2=0)
    print(topo)
    print(f"  euler_characteristic   = {topo.euler_characteristic}")     # 1
    print(f"  is_connected           = {topo.is_connected}")             # False
    print(f"  has_cycles             = {topo.has_cycles}")               # True
    print(f"  has_cavities           = {topo.has_cavities}")             # False
    print(f"  topological_complexity = {topo.topological_complexity}")    # 3
    print(f"  P(-1) [= χ]           = {topo.poincare_polynomial(-1)}")  # 1
    print(f"  P(1)  [= Σβ]          = {topo.poincare_polynomial(1)}")   # 3
    print()

    print("=" * 60)
    print("CONTROL METRICS")
    print("=" * 60)

    ctrl = ControlMetrics(poles_real=(-1.0, -2.0), phase_margin_deg=60.0)
    print(ctrl)
    print(f"  is_stable     = {ctrl.is_stable}")              # True
    print(f"  max_real_pole = {ctrl.max_real_pole}")           # -1.0
    print()

    # Retrocompatibilidad: acepta listas (convierte internamente a tupla)
    ctrl_list = ControlMetrics(poles_real=[-0.5, -3.0])
    print(f"  poles type    = {type(ctrl_list.poles_real)}")   # <class 'tuple'>
    print()

    # Fábrica desde polos complejos: s = -1 ± 2j
    ctrl_cpx = ControlMetrics.from_poles([complex(-1, 2), complex(-1, -2)])
    print(ctrl_cpx)
    print(f"  damped_freq   = {ctrl_cpx.damped_frequency:.3f}")  # 2.000
    print(f"  settling_time = {ctrl_cpx.settling_time:.3f}")     # 4.000
    print(f"  is_chaotic    = {ctrl_cpx.is_chaotic}")            # False
    print()

    print("=" * 60)
    print("THERMODYNAMIC METRICS")
    print("=" * 60)

    thermo = ThermodynamicMetrics(
        system_temperature=30.0, entropy=0.5, exergy=100.0,
    )
    print(thermo)
    print(f"  free_energy        = {thermo.free_energy}")                 # 85.0
    print(f"  thermal_efficiency = {thermo.thermal_efficiency:.2%}")      # 100.00%
    print()

    thermo_ref = ThermodynamicMetrics(
        system_temperature=30.0, entropy=0.5, exergy=100.0,
        reference_temperature=10.0,
    )
    print(thermo_ref)
    print(f"  thermal_efficiency = {thermo_ref.thermal_efficiency:.2%}")  # 66.67%
    print()

    thermo_fab = ThermodynamicMetrics.from_entropy_and_temp(
        temperature=50.0, entropy=1.2, heat_capacity=0.8,
        reference_temperature=5.0,
    )
    print(thermo_fab)
    print(f"  exergy (C_v·T) = {thermo_fab.exergy}")  # 40.0
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
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class PhysicsMetrics:
    """
    Subespacio Físico (Vector de Estado del FluxCondenser).
    Captura la dinámica de fluidos y electromagnética del procesamiento.

    Ref: flux_condenser.txt [Fuente 797]
    """
    # Dinámica de Fluidos
    saturation: float = 0.0          # V_elastic: Nivel de llenado del buffer (0.0 - 1.0)
    pressure: float = 0.0            # Presión estática en la cola de procesamiento

    # Electrodinámica (Modelo RLC)
    kinetic_energy: float = 0.0      # E_k = 0.5 * L * I^2 (Inercia del flujo de datos)
    potential_energy: float = 0.0    # E_p = 0.5 * C * V^2 (Energía almacenada en buffer)
    flyback_voltage: float = 0.0     # V_fb = L * di/dt (Picos por cambios bruscos de esquema)
    dissipated_power: float = 0.0    # P_dis = R * I^2 (Entropía generada por fricción/errores)

    # Estabilidad Mecánica
    gyroscopic_stability: float = 1.0 # S_g: Estabilidad rotacional del 'trompo' de datos

    # Métricas de Maxwell (Opcional, si se usa FDTD)
    poynting_flux: float = 0.0       # S = E x H (Flujo de valor direccional)

@dataclass(frozen=True)
class TopologicalMetrics:
    """
    Subespacio Topológico (Invariantes del BusinessTopologicalAnalyzer).
    Captura la forma y conectividad del grafo del proyecto.

    Ref: business_topology.txt [Fuente 622]
    """
    # Homología (Números de Betti)
    beta_0: int = 1                  # Componentes conexas (1 = ideal, >1 = fragmentación)
    beta_1: int = 0                  # Ciclos independientes (0 = ideal, >0 = bucles lógicos)
    beta_2: int = 0                  # Cavidades (estructuras vacías)

    # Invariantes Derivados
    euler_characteristic: int = 1    # χ = β0 - β1 + β2

    # Análisis Espectral
    fiedler_value: float = 1.0       # λ2: Conectividad algebraica (Resistencia a partición)
    spectral_gap: float = 0.0        # Diferencia entre primeros autovalores

    # Estabilidad Estructural
    pyramid_stability: float = 1.0   # Ψ: Índice de soporte base/cúspide (<1.0 = Pirámide Invertida)
    structural_entropy: float = 0.0  # Medida de desorden en la red

@dataclass(frozen=True)
class ControlMetrics:
    """
    Subespacio de Control (Dictamen del LaplaceOracle).
    Captura la estabilidad dinámica en el dominio de la frecuencia compleja.

    Ref: laplace_oracle.txt [Fuente 930]
    """
    # Ubicación de Polos
    poles_real: List[float] = field(default_factory=list) # σ: Parte real (debe ser < 0)
    is_stable: bool = True           # True si todos los polos están en LHP

    # Márgenes de Robustez
    phase_margin_deg: float = 45.0   # Margen de fase (> 45° es robusto)
    gain_margin_db: float = float('inf')

    # Respuesta Transitoria
    damping_ratio: float = 0.707     # ζ: Amortiguamiento (critico = 1.0)
    natural_frequency: float = 0.0   # ω_n: Frecuencia natural del sistema

    # Caos
    lyapunov_exponent: float = -1.0  # λ: Convergencia (<0) o Caos (>0)

@dataclass(frozen=True)
class ThermodynamicMetrics:
    """
    Subespacio Termodinámico (Economía Física).
    Captura la temperatura y eficiencia energética del sistema financiero.

    Ref: semantic_translator.txt [Fuente 1291]
    """
    system_temperature: float = 25.0 # T_sys: Volatilidad agregada de precios
    entropy: float = 0.0             # S: Grado de desorden administrativo
    exergy: float = 1.0              # Ex: Energía útil disponible para trabajo (presupuesto efectivo)
    heat_capacity: float = 0.5       # C_v: Capacidad de absorber sobrecostos sin 'fundirse'

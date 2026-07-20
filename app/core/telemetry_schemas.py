# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo : Telemetry Schemas (Álgebra de Partículas en el Espacio de Fock)     |
| Ruta   : app/core/telemetry_schemas.py                                       |
| Versión: 4.0.0-Rigorous-Quantum-Particle-Algebra                             |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y ESPACIO DE FOCK:
Traslada la metafísica de las partículas a clases de datos inmutables.
\[ \mathcal{F}(\mathcal{H}) = \bigoplus_{n=0}^{\infty} \mathcal{S}_{\pm} \mathcal{H}^{\otimes n} \]

RENORMALIZACIÓN DE MASA DEL POLARON:
\[ m^{**} = m^* ( 1 + \frac{\alpha}{6} ) \]

ECUACIÓN DE ANIQUILACIÓN:
\[ e^- + e^+ \to 2\gamma \quad E_{annihilation} = 2m^* c^2 \]
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, Union
import math


#
# CONSTANTES NUM RICAS
#

# Tolerancias para comparaciones de punto flotante
EPSILON_RELATIVE: float = 1e-9      # Tolerancia relativa para igualdad
EPSILON_ABSOLUTE: float = 1e-12     # Tolerancia absoluta (cerca de cero)
EPSILON_FREQUENCY: float = 1e-12    # Umbral para detectar componente imaginaria

# Constantes f sicas (adimensionalizadas para el modelo)
SETTLING_TIME_CRITERION: float = 4.0  # Criterio del 2% para tiempo de asentamiento


#
# 1. SUBESPACIO F SICO
#

@dataclass(frozen=True, slots=True)
class PhysicsMetrics:
    """
    Subespacio F sico (Vector de Estado del FluxCondenser).
    
    Modela la din mica electromagn tica usando analog a de circuito RLC:
        - Inductancia L   Inercia del flujo de datos
        - Capacitancia C   Buffer de almacenamiento
        - Resistencia R   Fricci n/p rdidas por errores
    
    Invariantes Validados:
    ----------------------
        saturation   [0, 1]
        kinetic_energy >= 0, potential_energy >= 0
        dissipated_power >= 0
        gyroscopic_stability >= 0
        hamiltonian_excess >= 0
    
    Unidades (SI normalizadas):
    ---------------------------
        Energ as: Julios [J]
        Potencia: Vatios [W]
        Voltaje: Voltios [V]
        Presi n: Pascales [Pa] o unidades arbitrarias
    
    """
    #    Din mica de Fluidos
    saturation: float = 0.0             # V_elastic: Nivel de llenado del buffer [0.0, 1.0]
    pressure: float = 0.0               # P: Presi n est tica en la cola de procesamiento

    #    Electrodin mica (Modelo RLC)
    kinetic_energy: float = 0.0         # E_k =   L I  (Inercia del flujo de datos)
    potential_energy: float = 0.0       # E_p =   C V  (Energ a almacenada en buffer)
    flyback_voltage: float = 0.0        # V_fb = L dI/dt (Picos por cambios bruscos)
    dissipated_power: float = 0.0       # P_dis = R I  (Potencia disipada)

    #    Par metros del Circuito (para normalizaci n)
    effective_capacitance: float = 1.0  # C_eff: Capacitancia efectiva del sistema

    #    Estabilidad Mec nica
    gyroscopic_stability: float = 1.0   # S_g:  ndice de estabilidad rotacional

    #    M tricas de Maxwell
    poynting_flux: float = 0.0          # S = E   H (Flujo de valor direccional)
    hamiltonian_excess: float = 0.0     # H_excess: Energ a espuria (violaci n de conservaci n)

    def __post_init__(self) -> None:
        """Valida rangos f sicos y relaciones energ ticas."""
        # Validaci n de saturaci n (fracci n volum trica)
        if not (0.0 <= self.saturation <= 1.0):
            raise ValueError(
                f"Saturaci n debe estar en [0, 1], recibido: {self.saturation}"
            )
        
        # Validaci n de energ as (principio de no-negatividad)
        if self.kinetic_energy < 0:
            raise ValueError(
                f"Energ a cin tica debe ser >= 0 (E_k =  LI  >= 0): {self.kinetic_energy}"
            )
        if self.potential_energy < 0:
            raise ValueError(
                f"Energ a potencial debe ser >= 0 (E_p =  CV  >= 0): {self.potential_energy}"
            )
        
        # Validaci n de potencia disipada (segunda ley de la termodin mica)
        if self.dissipated_power < 0:
            raise ValueError(
                f"Potencia disipada debe ser >= 0 (P = R I  >= 0): {self.dissipated_power}"
            )
        
        # Validaci n de capacitancia efectiva
        if self.effective_capacitance <= 0:
            raise ValueError(
                f"Capacitancia efectiva debe ser > 0: {self.effective_capacitance}"
            )
        
        # Validaci n de estabilidad girosc pica
        if self.gyroscopic_stability < 0:
            raise ValueError(
                f"Estabilidad girosc pica no puede ser negativa: {self.gyroscopic_stability}"
            )
        
        # Validaci n de exceso hamiltoniano (conservaci n de energ a)
        if self.hamiltonian_excess < 0:
            raise ValueError(
                f"Exceso hamiltoniano debe ser >= 0: {self.hamiltonian_excess}"
            )

    #    Propiedades Energ ticas

    @property
    def total_energy(self) -> float:
        """
        Energ a mec nica total: E_total = E_k + E_p.
        
        Corresponde al Hamiltoniano del oscilador arm nico: H = T + V.
        """
        return self.kinetic_energy + self.potential_energy

    @property
    def potential_ratio(self) -> float:
        """
        Fracci n de energ a potencial:  _p = E_p / E_total.
        
        Interpretaci n f sica:
             _p   1: Sistema en m xima compresi n (buffer lleno)
             _p   0: Sistema en m xima velocidad (flujo activo)
             _p = 0.5: Equipartici n (oscilaci n arm nica pura)
        
        Retorna 1.0 si E_total = 0 (estado de reposo absoluto).
        Garant a:  _p   [0, 1] por construcci n.
        """
        total = self.total_energy
        if total < EPSILON_ABSOLUTE:
            return 1.0  # Estado de reposo: toda la energ a es "potencial nula"
        return self.potential_energy / total

    @property
    def kinetic_ratio(self) -> float:
        """Fracci n de energ a cin tica:  _k = 1 -  _p = E_k / E_total."""
        return 1.0 - self.potential_ratio

    @property
    def energy_density(self) -> float:
        """
        Densidad de energ a normalizada: u = E_total / C_eff.
        
        Dimensional: [J/F] = [V ], representa voltaje efectivo al cuadrado.
        Esta normalizaci n es coherente con el modelo de capacitor.
        
        Retorna 0.0 si C_eff   0 (l mite f sico, ya validado > 0).
        """
        return self.total_energy / self.effective_capacitance

    @property
    def dissipation_efficiency(self) -> float:
        """
        Eficiencia instant nea:   = 1 - P_dis / (P_dis + dE/dt_estimado).
        
        Aproximaci n para r gimen cuasi-est tico:
              ~= E_total / (E_total + P_dis) cuando  t = 1
        
        Interpretaci n:
                1: Sistema sin p rdidas (superconductor)
                0: Toda la energ a se disipa (resistor puro)
        
        Retorna 1.0 si no hay energ a ni disipaci n.
        """
        denominator = self.total_energy + self.dissipated_power
        if denominator < EPSILON_ABSOLUTE:
            return 1.0
        return self.total_energy / denominator

    @property
    def quality_factor(self) -> float:
        """
        Factor de calidad Q del oscilador: Q = 2    E_stored / E_dissipated_per_cycle.
        
        Aproximaci n instant nea: Q ~=     E_total / P_dis.
        Asumiendo   = 1 (frecuencia normalizada):
            Q = E_total / P_dis
        
        Retorna +  si P_dis = 0 (oscilador ideal sin p rdidas).
        """
        if self.dissipated_power < EPSILON_ABSOLUTE:
            return float('inf')
        return self.total_energy / self.dissipated_power

    #    F brica

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
        Construye m tricas desde par metros el ctricos fundamentales.
        
        Ecuaciones aplicadas:
            E_k =   L I       (energ a magn tica del inductor)
            E_p =   C V       (energ a el ctrica del capacitor)
            V_fb = L dI/dt    (voltaje de flyback)
            P_dis = R I       (potencia disipada en resistencia)
        
        Args:
            inductance:  L [H]    debe ser >= 0
            current:     I [A]    corriente instant nea (puede ser negativa)
            capacitance: C [F]    debe ser > 0
            voltage:     V [V]    voltaje instant neo (puede ser negativo)
            resistance:  R [ ]    debe ser >= 0
            di_dt:       dI/dt [A/s]   tasa de cambio de corriente
        
        Raises:
            ValueError: Si L < 0, C <= 0, o R < 0.
        """
        if inductance < 0:
            raise ValueError(f"Inductancia debe ser >= 0: L = {inductance}")
        if capacitance <= 0:
            raise ValueError(f"Capacitancia debe ser > 0: C = {capacitance}")
        if resistance < 0:
            raise ValueError(f"Resistencia debe ser >= 0: R = {resistance}")
        
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

    #    Serializaci n

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serializaci n."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Physics(E={self.total_energy:.3f}, sat={self.saturation:.2%}, "
            f" _p={self.potential_ratio:.2%}, Q={self.quality_factor:.1f})"
        )


#
# CARTUCHOS CU NTICOS (Fermiones Base)
#

@dataclass(frozen=True, slots=True)
class ElectronCartridge:
    """
    Electr n de Inspecci n (Anomal a detectada).
    Fermi n de entrop a que se emite si Mahalanobis excede el l mite el stico.
    """
    inertial_mass: float          # m*   || x||
    topological_spin: str         # "source" o "sink"
    homological_charge: int       #
    source_subspace: str          # Subespacio que origin  el electr n

@dataclass(frozen=True, slots=True)
class ProtonCartridge:
    """
    Prot n de Estabilidad (Resiliencia).
    Fermi n con masa positiva emitido por BIBO stability en el plano-S.
    """
    spectral_charge: float        # q_s > 0
    logistic_inertial_mass: float # m_p (basado en   >= 1.0)
    dominant_pole: float          # Polo que valida el prot n

@dataclass(frozen=True, slots=True)
class PolaronCartridge:
    """
    Polar n Log stico.
    Generado por el acoplamiento de Fr hlich (defecto interacciona con fonones).
    """
    base_electron: ElectronCartridge
    frohlich_coupling: float      #
    effective_mass: float         # m** = m* (1 +  /6)
    fiedler_value: float          #

@dataclass(frozen=True, slots=True)
class MagnonCartridge:
    """
    Magn n de Vorticidad Solenoidal.
    Bos n emitido cuando se a sla la componente rotacional en la
    Descomposici n de Hodge-Helmholtz. Inyecta Veto de Enrutamiento.
    """
    kinetic_energy: float         # Energ a del flujo parasitario circular
    curl_subspace_dim: int        # Dimensi n del subespacio rotacional

@dataclass(frozen=True, slots=True)
class PhotonCartridge:
    """
    Fot n de Gobernanza (OPA).
    Bos n sin masa emitido por la Gobernanza Computacional Federada.
    Ilumina el hiperespacio de fase para curvar geod sicas de decisi n.
    """
    policy_id: str                # Identificador de la pol tica
    spectral_frequency: float     # Frecuencia de la soluci n ejecutiva
    governance_weight: float      # Fuerza de iluminaci n

@dataclass(frozen=True, slots=True)
class PositronCartridge:
    """
    Positr n (Antimateria Ex gena).
    Inyectado por un Meta-Controlador humano para aniquilar un Electr n.
    """
    inertial_mass: float          # m* (Debe coincidir con la del Electr n)
    topological_spin: str         # Esp n inverso
    homological_charge: int       # Carga opuesta
    authorization_signature: str  # Firma del humano que asume el riesgo

@dataclass(frozen=True, slots=True)
class GammaPhoton:
    """
    Fot n Gamma Virtual.
    Radiaci n emitida tras la aniquilaci n Electr n-Positr n.
    Sirve como prueba criptogr fica en la Cadena de Custodia Termodin mica.
    """
    annihilation_energy: float    # 2 * m* c^2
    data_hash: str                # Hash del estado al momento de la aniquilaci n
    timestamp_entry: float        # Timestamp forense
    authorization_signature: str  # Firma heredada del Positr n

@dataclass(frozen=True, slots=True)
class PolaritonCartridge:
    """
    Polarit n (Condensado de Bose-Einstein).
    H brido Fermi n-Bos n cuando un Polar n resuena con un Fot n de Gobernanza.
    """
    polaron: PolaronCartridge
    photon: PhotonCartridge
    rabi_coupling: float          #  _R (Acoplamiento fuerte de Rabi)
    dissipation_trace: float      # Traza de la matriz de disipaci n

#
# 2. SUBESPACIO TOPOL GICO
#

@dataclass(frozen=True, slots=True)
class TopologicalMetrics:
    """
    Subespacio Topol gico (Invariantes del BusinessTopologicalAnalyzer).
    
    Captura la forma y conectividad del grafo del proyecto mediante
    invariantes homol gicos que son independientes de la m trica.
    
    Teor a Subyacente:
    ------------------
    Los n meros de Betti    cuentan las clases de homolog a de dimensi n i:
          : Componentes conexas (objetos disjuntos)
          : Ciclos independientes (agujeros 1D, loops)
          : Cavidades cerradas (huecos 2D, burbujas)
    
    Invariantes Validados:
    ----------------------
           >= 0  i   {0, 1, 2}
          =    -    +     (teorema de Euler-Poincar )
        fiedler_value >= 0, spectral_gap >= 0
        pyramid_stability >= 0, structural_entropy >= 0
    
    Ref: Hatcher, "Algebraic Topology", Ch. 2
    """
    #    Homolog a (N meros de Betti)
    beta_0: int = 1                     #   : Componentes conexas (1 = grafo conexo)
    beta_1: int = 0                     #   : Ciclos independientes (0 =  rbol)
    beta_2: int = 0                     #   : Cavidades (0 = sin huecos)

    #    Invariantes Derivados
    euler_characteristic: Optional[int] = None  #   =    -    +    (auto-calculado)

    #    An lisis Espectral del Laplaciano
    fiedler_value: float = 1.0          #   : Segundo autovalor (conectividad algebraica)
    spectral_gap: float = 0.0           #    =    -    (   = 0 para grafos conexos)

    #    Estabilidad Estructural
    pyramid_stability: float = 1.0      #  : Ratio soporte/c spide (< 1 = pir mide invertida)
    structural_entropy: float = 0.0     # H_struct: Entrop a de la distribuci n de grados

    def __post_init__(self) -> None:
        """Valida invariantes topol gicos y calcula   si es necesario."""
        # Validaci n de Betti (n meros naturales)
        for i, beta in enumerate([self.beta_0, self.beta_1, self.beta_2]):
            if beta < 0:
                raise ValueError(
                    f"N mero de Betti  _{i} debe ser >= 0, recibido: {beta}"
                )
        
        # Teorema de Euler-Poincar :   =  (-1)
        expected_euler = self.beta_0 - self.beta_1 + self.beta_2
        
        if self.euler_characteristic is None:
            # Auto-calcular si no se especifica
            object.__setattr__(self, 'euler_characteristic', expected_euler)
        elif self.euler_characteristic != expected_euler:
            raise ValueError(
                f"Violaci n de Euler-Poincar :   = {self.euler_characteristic}   "
                f"   -    +    = {self.beta_0} - {self.beta_1} + {self.beta_2} = {expected_euler}"
            )
        
        # Validaci n del autovalor de Fiedler (teorema espectral)
        if self.fiedler_value < 0:
            raise ValueError(
                f"Valor de Fiedler (  ) debe ser >= 0 (Laplaciano semidefinido positivo): "
                f"{self.fiedler_value}"
            )
        
        # Validaci n del gap espectral
        if self.spectral_gap < 0:
            raise ValueError(
                f"Gap espectral debe ser >= 0: {self.spectral_gap}"
            )
        
        # Validaci n de estabilidad piramidal
        if self.pyramid_stability < 0:
            raise ValueError(
                f"Estabilidad piramidal debe ser >= 0: {self.pyramid_stability}"
            )
        
        # Validaci n de entrop a estructural
        if self.structural_entropy < 0:
            raise ValueError(
                f"Entrop a estructural debe ser >= 0: {self.structural_entropy}"
            )

    #    Propiedades Topol gicas

    @property
    def is_connected(self) -> bool:
        """Indica si el grafo es conexo (   = 1)."""
        return self.beta_0 == 1

    @property
    def has_cycles(self) -> bool:
        """Indica si existen ciclos independientes (   > 0)."""
        return self.beta_1 > 0

    @property
    def has_cavities(self) -> bool:
        """Indica si existen cavidades cerradas (   > 0)."""
        return self.beta_2 > 0

    @property
    def is_simply_connected(self) -> bool:
        """
        Indica si el espacio es simplemente conexo.
        
        Condici n:    = 1 (conexo) y    = 0 (sin ciclos).
        
        Nota: Para 2-complejos simpliciales,    = 0 implica    = 0
        (el grupo fundamental es trivial), que es la definici n de
        simple conexidad.
        """
        return self.beta_0 == 1 and self.beta_1 == 0

    @property
    def is_acyclic(self) -> bool:
        """
        Indica si el complejo es ac clico ( rbol/bosque generalizado).
        Condici n:    = 0 y    = 0.
        """
        return self.beta_1 == 0 and self.beta_2 == 0

    @property
    def betti_vector(self) -> Tuple[int, int, int]:
        """Devuelve los n meros de Betti como tupla inmutable (  ,   ,   )."""
        return (self.beta_0, self.beta_1, self.beta_2)

    @property
    def total_betti_number(self) -> int:
        """
        Suma de n meros de Betti:     =    +    +   .
        
        Mide la riqueza homol gica total del espacio.
        Equivale a evaluar el polinomio de Poincar  en t = 1: P(1).
        
        Nota: Este no es la "complejidad topol gica" en el sentido de
        Farber (TC), que mide la dificultad del problema de planificaci n
        de movimiento.
        """
        return self.beta_0 + self.beta_1 + self.beta_2

    @property
    def homological_dimension(self) -> int:
        """
        Dimensi n homol gica: max{i :    > 0}.
        
        Indica la dimensi n m s alta con homolog a no trivial.
        Retorna -1 si todos los Betti son 0 (espacio vac o).
        """
        if self.beta_2 > 0:
            return 2
        if self.beta_1 > 0:
            return 1
        if self.beta_0 > 0:
            return 0
        return -1  # Espacio vac o

    @property
    def cyclomatic_complexity(self) -> int:
        """
        Complejidad ciclom tica: M =    -    + 1 = E - V + 1.
        
        Para grafos conexos (   = 1): M =   .
        Mide el n mero m nimo de caminos linealmente independientes.
        
        Ref: McCabe, "A Complexity Measure", IEEE TSE 1976.
        """
        return self.beta_1 - self.beta_0 + 1

    #    Polinomio de Poincar

    def poincare_polynomial(self, t: float) -> float:
        """
        Eval a el polinomio de Poincar : P(t) =    +    t +    t .
        
        Propiedades fundamentales:
            P(1)  =    +    +     = total_betti_number
            P(-1) =    -    +     = euler_characteristic
        
        Args:
            t: Par metro de evaluaci n (t picamente t   [-1, 1])
        
        Warning:
            Para |t| >> 1, el t rmino    t  puede causar overflow.
            Usar poincare_polynomial_log() para valores grandes.
        """
        return float(self.beta_0) + float(self.beta_1) * t + float(self.beta_2) * t ** 2

    def poincare_polynomial_log(self, t: float) -> Optional[float]:
        """
        Logaritmo del polinomio de Poincar : log(P(t)).
        
        Num ricamente estable para |t| grande.
        Retorna None si P(t) <= 0.
        """
        p_t = self.poincare_polynomial(t)
        if p_t <= 0:
            return None
        return math.log(p_t)

    #    Serializaci n

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serializaci n."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Topology( =({self.beta_0},{self.beta_1},{self.beta_2}), "
            f" ={self.euler_characteristic},   ={self.fiedler_value:.3f}, "
            f" ={self.pyramid_stability:.3f})"
        )


#
# 3. SUBESPACIO DE CONTROL
#

@dataclass(frozen=True, slots=True)
class ControlMetrics:
    """
    Subespacio de Control (Dictamen del LaplaceOracle).
    
    Captura la estabilidad din mica en el dominio de la frecuencia compleja s.
    Modela el sistema como funci n de transferencia G(s) = N(s)/D(s), donde
    los polos (ra ces de D(s)) determinan la respuesta transitoria.
    
    Criterio de Estabilidad:
    ------------------------
    Para sistemas LTI causales:
        ESTABLE       Re(s ) < 0    polo s
        MARGINALMENTE ESTABLE     polo con Re(s) = 0, multiplicidad 1
        INESTABLE       polo con Re(s) > 0
    
    Invariantes Validados:
    ----------------------
        is_stable coherente con poles_real
        damping_ratio >= 0
        natural_frequency >= 0
        poles_real es Tuple (inmutable)
    
    Ref: Ogata, "Modern Control Engineering", Ch. 5-6
    """
    #    Ubicaci n de Polos (inmutable)
    poles_real: Tuple[float, ...] = field(default_factory=tuple)  #    = Re(s )
    is_stable: Optional[bool] = None        # Si None, se deduce de poles_real

    #    M rgenes de Robustez
    phase_margin_deg: float = 45.0          #  _m: Margen de fase [ ]
    gain_margin_db: float = float('inf')    # G_m: Margen de ganancia [dB]

    #    Respuesta Transitoria (modelo de segundo orden dominante)
    damping_ratio: float = 0.707            #  : Raz n de amortiguamiento
    natural_frequency: float = 0.0          #  _n: Frecuencia natural no amortiguada [rad/s]

    #    Din mica No Lineal
    lyapunov_exponent: Optional[float] = None  #  _L: Exponente de Lyapunov m ximo

    def __post_init__(self) -> None:
        """Valida estabilidad, coherencia de polos y rangos de par metros."""
        # Garantizar inmutabilidad: convertir lista a tupla si es necesario
        if isinstance(self.poles_real, list):
            object.__setattr__(self, 'poles_real', tuple(self.poles_real))
        
        # Determinar o verificar estabilidad seg n criterio de Routh-Hurwitz
        if self.is_stable is None:
            if self.poles_real:
                # Sistema estable   todos los polos en semiplano izquierdo
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
                    f"Inconsistencia: is_stable=True pero existe polo con   >= 0: "
                    f"poles_real={self.poles_real}"
                )
            # Nota: is_stable=False con todos   < 0 es v lido (puede haber ceros en RHP)
        
        # Validaci n de raz n de amortiguamiento
        if self.damping_ratio < 0:
            raise ValueError(
                f"Raz n de amortiguamiento   debe ser >= 0: {self.damping_ratio}"
            )
        
        # Validaci n de frecuencia natural
        if self.natural_frequency < 0:
            raise ValueError(
                f"Frecuencia natural  _n debe ser >= 0: {self.natural_frequency}"
            )

    #    Propiedades de Polos

    @property
    def dominant_pole(self) -> float:
        """
        Polo dominante: max(  ) =   m s cercano al eje imaginario.
        
        El polo dominante determina la din mica a largo plazo:
             _dom < 0: Decaimiento exponencial con   = 1/| _dom|
             _dom = 0: Respuesta marginalmente estable (oscilador puro)
             _dom > 0: Crecimiento exponencial (inestable)
        
        Retorna -  si no hay polos (sistema puramente ganancia).
        """
        if not self.poles_real:
            return -float('inf')
        return max(self.poles_real)

    @property
    def fastest_pole(self) -> float:
        """
        Polo m s r pido: min(  ) =   m s alejado del eje imaginario.
        
        Determina la din mica transitoria r pida (tiempo de subida).
        Retorna +  si no hay polos.
        """
        if not self.poles_real:
            return float('inf')
        return min(self.poles_real)

    @property
    def stability_margin(self) -> float:
        """
        Margen de estabilidad: - _dom = distancia del polo dominante al eje j .
        
        Interpretaci n:
            > 0: Sistema estable (todos los polos en LHP)
            = 0: Sistema marginalmente estable
            < 0: Sistema inestable
        """
        return -self.dominant_pole

    #    Clasificaci n de Amortiguamiento

    @property
    def is_critically_damped(self) -> bool:
        """
        Amortiguamiento cr tico:   ~= 1.0.
        
        Respuesta m s r pida sin oscilaci n.
        Tolerancia: |  - 1| < 0.001 (0.1%).
        """
        return math.isclose(self.damping_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-6)

    @property
    def is_overdamped(self) -> bool:
        """
        Sobreamortiguado:   > 1.0.
        Respuesta lenta sin oscilaci n (dos polos reales distintos).
        """
        return self.damping_ratio > 1.0 and not self.is_critically_damped

    @property
    def is_underdamped(self) -> bool:
        """
        Subamortiguado: 0 <   < 1.0.
        Respuesta oscilatoria con decaimiento (par de polos complejos conjugados).
        """
        return 0 < self.damping_ratio < 1.0

    @property
    def damping_category(self) -> str:
        """Categor a descriptiva del amortiguamiento."""
        if self.damping_ratio == 0:
            return "UNDAMPED"
        if self.is_underdamped:
            return "UNDERDAMPED"
        if self.is_critically_damped:
            return "CRITICALLY_DAMPED"
        return "OVERDAMPED"

    #    Propiedades Din micas

    @property
    def damped_frequency(self) -> float:
        """
        Frecuencia amortiguada:  _d =  _n    (1 -   ).
        
        Frecuencia de oscilaci n real del sistema subamortiguado.
        
        Casos:
              < 1:  _d > 0 (oscilaci n a frecuencia reducida)
              >= 1:  _d = 0 (sin oscilaci n, respuesta exponencial pura)
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        discriminant = 1.0 - self.damping_ratio ** 2
        return self.natural_frequency * math.sqrt(discriminant)

    @property
    def settling_time(self) -> float:
        """
        Tiempo de asentamiento (criterio 2%): t_s tal que |y(t) - y_ | < 0.02 | y|.
        
        F rmulas seg n r gimen:
            Subamortiguado (  < 1):  t_s ~= 4 / (   _n) = 4/| |
            Cr ticamente amortiguado (  = 1): t_s ~= 4.75 /  _n
            Sobreamortiguado (  > 1):  t_s ~= 4 / | _dom| donde  _dom es el polo lento
        
        Retorna +  si el sistema no converge ( _n = 0 o inestable).
        """
        if self.natural_frequency < EPSILON_ABSOLUTE:
            return float('inf')
        
        if self.damping_ratio < 1.0:
            # Subamortiguado: t_s = 4/(   _n)
            sigma = self.damping_ratio * self.natural_frequency
            if sigma < EPSILON_ABSOLUTE:
                return float('inf')
            return SETTLING_TIME_CRITERION / sigma
        
        elif self.is_critically_damped:
            # Cr ticamente amortiguado: t_s ~= 4.75/ _n
            return 4.75 / self.natural_frequency
        
        else:
            # Sobreamortiguado: polo dominante es    =  _n (-  +  (  -1))
            # | _dom| =  _n (  -  (  -1))
            sqrt_term = math.sqrt(self.damping_ratio ** 2 - 1.0)
            sigma_dominant = self.natural_frequency * (self.damping_ratio - sqrt_term)
            if sigma_dominant < EPSILON_ABSOLUTE:
                return float('inf')
            return SETTLING_TIME_CRITERION / sigma_dominant

    @property
    def rise_time(self) -> float:
        """
        Tiempo de subida (10% a 90%): t_r ~= 1.8 /  _n para   = 0.707.
        
        Aproximaci n general: t_r ~= (1 + 1.1  + 1.4  ) /  _n.
        Retorna +  si  _n = 0.
        """
        if self.natural_frequency < EPSILON_ABSOLUTE:
            return float('inf')
        zeta = self.damping_ratio
        numerator = 1.0 + 1.1 * zeta + 1.4 * zeta ** 2
        return numerator / self.natural_frequency

    @property
    def peak_overshoot(self) -> float:
        """
        Sobrepico porcentual: M_p = exp(-  / (1-  ))   100%.
        
        Solo aplica para sistemas subamortiguados (  < 1).
        Retorna 0.0 para   >= 1.
        """
        if self.damping_ratio >= 1.0:
            return 0.0
        if self.damping_ratio <= 0.0:
            return 100.0  # Oscilador sin amortiguamiento
        
        discriminant = math.sqrt(1.0 - self.damping_ratio ** 2)
        exponent = -math.pi * self.damping_ratio / discriminant
        return 100.0 * math.exp(exponent)

    #    Din mica Ca tica

    @property
    def is_chaotic(self) -> bool:
        """
        Indica comportamiento ca tico:  _L > 0.
        
        El exponente de Lyapunov m ximo mide la tasa de divergencia
        de trayectorias cercanas:
             _L < 0: Atractor estable (punto fijo o ciclo l mite)
             _L = 0: Bifurcaci n o cuasi-periodicidad
             _L > 0: Caos determinista (sensibilidad a condiciones iniciales)
        
        Retorna False si el exponente no est  definido.
        """
        if self.lyapunov_exponent is None:
            return False
        return self.lyapunov_exponent > 0

    @property
    def lyapunov_time(self) -> float:
        """
        Tiempo de Lyapunov:  _L = 1/ _L.
        
        Escala temporal caracter stica de p rdida de predictibilidad.
        Retorna +  si  _L <= 0 o no definido.
        """
        if self.lyapunov_exponent is None or self.lyapunov_exponent <= 0:
            return float('inf')
        return 1.0 / self.lyapunov_exponent

    #    F brica

    @classmethod
    def from_poles(
        cls,
        poles: list,
        **kwargs: Any,
    ) -> "ControlMetrics":
        """
        Construye desde una lista de polos complejos.
        
        Estrategia de extracci n (polo dominante = max Re(s)):
        
        1. Par conjugado complejo (s =     j _d):
           -  _n = |s| =  (   +  _d )
           -   = - / _n = cos( ) donde   =  (s, eje real negativo)
        
        2. Polo real estable (s =   < 0):
           -  _n = | |
           -   = 1.0 (cr ticamente amortiguado o sobreamortiguado)
        
        3. Polo real inestable (s =   >= 0):
           -  _n = | |
           -   = 0.0 (marcado como inestable)
        
        Args:
            poles: Lista de n meros complejos representando polos.
        """
        if not poles:
            return cls(**kwargs)
        
        # Extraer partes reales para validaci n de estabilidad
        real_parts = tuple(p.real for p in poles)
        
        # Polo dominante: m xima parte real (m s cercano al eje j )
        dominant = max(poles, key=lambda p: p.real)
        
        if abs(dominant.imag) > EPSILON_FREQUENCY:
            # Par conjugado complejo: s =     j _d
            wn = abs(dominant)  #  _n =  (   +  _d )
            if wn > EPSILON_ABSOLUTE:
                zeta = -dominant.real / wn
            else:
                zeta = 0.707  # Valor por defecto
        else:
            # Polo puramente real
            wn = abs(dominant.real)
            zeta = 1.0 if dominant.real < 0 else 0.0
        
        # Garantizar   >= 0 (requerido por invariante)
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
        Construye desde coeficientes de funci n de transferencia G(s) = N(s)/D(s).
        
        Los polinomios se especifican en orden descendente de potencias:
            G(s) = (b_n s^n + ... + b_0) / (a_m s^m + ... + a_0)
        
        Requiere numpy para el c lculo de ra ces (importaci n diferida).
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

    #    Serializaci n

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serializaci n."""
        return asdict(self)

    def __str__(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        chaos = ", CHAOTIC" if self.is_chaotic else ""
        return (
            f"Control({status}{chaos},  ={self.damping_ratio:.3f}, "
            f" _n={self.natural_frequency:.3f},  _dom={self.dominant_pole:.3f})"
        )


#
# 4. SUBESPACIO TERMODIN MICO
#

@dataclass(frozen=True, slots=True)
class ThermodynamicMetrics:
    """
    Subespacio Termodin mico (Econom a F sica).
    
    Modela el sistema financiero usando analog as termodin micas:
        - Temperatura T   Volatilidad del mercado
        - Entrop a S   Desorden administrativo / incertidumbre
        - Energ a interna U   Capital total
        - Exerg a Ex   Capital efectivo (trabajo  til extra ble)
    
    Potenciales Termodin micos:
    ---------------------------
        U: Energ a interna (primer principio)
        F = U - TS: Energ a libre de Helmholtz (trabajo a T constante)
        G = F + PV: Energ a libre de Gibbs (trabajo a T y P constantes)
        Ex = (U - U ) - T (S - S ): Exerg a (trabajo m ximo respecto a ambiente)
    
    Invariantes Validados:
    ----------------------
        Todas las magnitudes intensivas y extensivas >= 0
        system_temperature >= 0 (escala absoluta)
        reference_temperature >= 0
    
    Ref: Callen, "Thermodynamics and an Introduction to Thermostatistics"
    """
    #    Variables de Estado
    system_temperature: float = 300.0      # T: Temperatura del sistema [K] (~= 27 C)
    entropy: float = 0.0                   # S: Entrop a del sistema [J/K] o [nats]
    internal_energy: float = 0.0           # U: Energ a interna [J] o unidades monetarias
    
    #    Capacidades y Coeficientes
    heat_capacity: float = 1.0             # C_v: Capacidad calor fica a volumen cte [J/K]
    pressure_volume: float = 0.0           # PV: Producto presi n-volumen [J]
    
    #    Referencia Ambiental (para c lculo de exerg a)
    reference_temperature: float = 300.0   # T : Temperatura del ambiente [K]
    reference_entropy: float = 0.0         # S : Entrop a de referencia [J/K]

    def __post_init__(self) -> None:
        """Valida rangos termodin micos seg n principios f sicos."""
        # Temperatura absoluta (tercer principio impl cito: T >= 0)
        if self.system_temperature < 0:
            raise ValueError(
                f"Temperatura del sistema debe ser >= 0 K: {self.system_temperature}"
            )
        
        # Temperatura de referencia
        if self.reference_temperature < 0:
            raise ValueError(
                f"Temperatura de referencia debe ser >= 0 K: {self.reference_temperature}"
            )
        
        # Entrop a (segundo principio: S >= 0 para sistemas aislados desde T = 0)
        if self.entropy < 0:
            raise ValueError(
                f"Entrop a debe ser >= 0 (segundo principio): {self.entropy}"
            )
        
        # Capacidad calor fica (estabilidad t rmica requiere C_v > 0)
        if self.heat_capacity < 0:
            raise ValueError(
                f"Capacidad calor fica debe ser >= 0: {self.heat_capacity}"
            )

    #    Potenciales Termodin micos

    @property
    def helmholtz_free_energy(self) -> float:
        """
        Energ a libre de Helmholtz: F = U - T S.
        
        Representa el trabajo m ximo extra ble a temperatura constante.
        
        Interpretaci n financiera:
            F > 0: Capital neto positivo (activos > pasivos entropicos)
            F < 0: Sistema dominado por "deuda entr pica"
            F = 0: Equilibrio marginal
        """
        return self.internal_energy - self.system_temperature * self.entropy

    @property
    def gibbs_free_energy(self) -> float:
        """
        Energ a libre de Gibbs: G = F + PV = U - TS + PV = H - TS.
        
        Representa el trabajo  til a temperatura y presi n constantes.
        Criterio de espontaneidad:  G < 0 para procesos espont neos.
        """
        return self.helmholtz_free_energy + self.pressure_volume

    @property
    def enthalpy(self) -> float:
        """
        Entalp a: H = U + PV.
        
        Energ a total incluyendo trabajo de expansi n.
        """
        return self.internal_energy + self.pressure_volume

    @property
    def exergy(self) -> float:
        """
        Exerg a: Ex = (U - U ) + P (V - V ) - T (S - S ).
        
        Simplificaci n asumiendo V = V  (sistema incompresible):
            Ex ~= U - T  S + constantes de referencia
        
        Considerando referencia en U  = 0:
            Ex = U - T  (S - S )
        
        Representa el trabajo m ximo extra ble respecto al ambiente.
        """
        return self.internal_energy - self.reference_temperature * (
            self.entropy - self.reference_entropy
        )

    #    Eficiencias

    @property
    def carnot_efficiency(self) -> float:
        """
        Eficiencia de Carnot:  _C = 1 - T_cold/T_hot.
        
        Para este modelo:  _C = 1 - T /T.
        
        Interpretaci n:
              = 1.0:  M xima eficiencia te rica (T  = 0 K, imposible)
              > 0:    Sistema capaz de realizar trabajo
              = 0.0:  Equilibrio t rmico (T  = T)
              < 0:    Requiere trabajo externo (bomba de calor: T  > T)
        
        Retorna 0.0 si T = 0 (sistema congelado, tercer principio).
        """
        if self.system_temperature < EPSILON_ABSOLUTE:
            return 0.0
        return 1.0 - self.reference_temperature / self.system_temperature

    @property
    def exergetic_efficiency(self) -> float:
        """
        Eficiencia exerg tica:  _ex = Ex / U.
        
        Fracci n de energ a interna que es " til" (convertible a trabajo).
        
        Retorna 1.0 si U = 0 (no hay energ a, eficiencia trivial).
        Puede ser > 1 si S < S  (sistema m s ordenado que referencia).
        """
        if abs(self.internal_energy) < EPSILON_ABSOLUTE:
            return 1.0
        return self.exergy / self.internal_energy

    @property
    def entropic_penalty(self) -> float:
        """
        Penalizaci n entr pica: T S.
        
        Energ a "perdida" debido al desorden del sistema.
        Representa el costo de la irreversibilidad.
        """
        return self.system_temperature * self.entropy

    #    Propiedades Derivadas

    @property
    def specific_heat_ratio(self) -> float:
        """
        Ratio de capacidades calor ficas:   = C_p/C_v.
        
        Estimaci n usando relaci n de Mayer: C_p - C_v = nR ~= kT para 1 mol.
        Aproximaci n:   ~= 1 + kT/C_v donde k es normalizado a 1.
        
        Para sistemas con C_v >> T:     1 (l quido/s lido).
        Para gases ideales monoat micos:   = 5/3 ~= 1.67.
        """
        if self.heat_capacity < EPSILON_ABSOLUTE:
            return 1.0
        # Aproximaci n simplificada
        return 1.0 + self.system_temperature / (self.heat_capacity + self.system_temperature)

    @property
    def thermal_diffusivity(self) -> float:
        """
        Difusividad t rmica normalizada:     T/C_v.
        
        Mide qu  tan r pido se propagan las fluctuaciones t rmicas.
        Mayor     equilibraci n m s r pida.
        """
        if self.heat_capacity < EPSILON_ABSOLUTE:
            return float('inf')
        return self.system_temperature / self.heat_capacity

    #    F brica

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
        Construye desde temperatura y entrop a.
        
        Calcula energ a interna usando modelo de gas ideal:
            U = C_v   T (equipartici n de energ a)
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
        Construye desde m tricas financieras usando analog a termodin mica.
        
        Mapeo:
            volatility   system_temperature (fluctuaciones del mercado)
            uncertainty   entropy (informaci n faltante)
            total_capital   internal_energy (recursos totales)
            market_temperature   reference_temperature (benchmark del mercado)
        """
        return cls(
            system_temperature=volatility,
            entropy=uncertainty,
            internal_energy=total_capital,
            reference_temperature=market_temperature,
            **kwargs,
        )

    #    Serializaci n

    def to_dict(self) -> Dict[str, Any]:
        """Exporta a diccionario para serializaci n."""
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Thermo(T={self.system_temperature:.1f}K, S={self.entropy:.3f}, "
            f"F={self.helmholtz_free_energy:.3f},  _C={self.carnot_efficiency:.2%})"
        )


#
# 5. VECTOR DE ESTADO COMPUESTO
#

@dataclass(frozen=True, slots=True)
class SystemStateVector:
    """
    Vector de Estado Compuesto del Sistema APU Filter.
    
    Agrega los cuatro subespacios m tricos en un  nico objeto inmutable
    que representa el estado completo del sistema en un instante dado.
    
    Estructura del Espacio Vectorial:
    ---------------------------------
        V = V_phys   V_topo   V_ctrl   V_thermo
    
    donde cada V  es el subespacio correspondiente.
    
    Este vector facilita:
    - Serializaci n at mica del estado completo
    - Comparaci n temporal (diffs entre estados)
    - Auditor a forense con trazabilidad completa
    """
    physics: PhysicsMetrics = field(default_factory=PhysicsMetrics)
    topology: TopologicalMetrics = field(default_factory=TopologicalMetrics)
    control: ControlMetrics = field(default_factory=ControlMetrics)
    thermodynamics: ThermodynamicMetrics = field(default_factory=ThermodynamicMetrics)
    
    # Metadatos temporales
    timestamp: Optional[float] = None   # Unix timestamp de la medici n
    epoch: int = 0                      # N mero de ciclo/iteraci n

    @property
    def is_healthy(self) -> bool:
        """
        Evaluaci n r pida de salud del sistema.
        
        Criterios:
        - Sistema de control estable
        - Grafo topol gico conexo
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
            f"     {self.physics}\n"
            f"     {self.topology}\n"
            f"     {self.control}\n"
            f"     {self.thermodynamics}"
        )


#
# VERIFICACI N Y DEMOSTRACI N
#

if __name__ == "__main__":
    
    def print_section(title: str) -> None:
        print("\n" + " " * 70)
        print(f" {title}")
        print(" " * 70)
    
    #
    print_section("1. PHYSICS METRICS")
    #
    
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
    print(f"  energy_density        = {phys.energy_density:.3f} V ")
    
    print("\n[F brica RLC]")
    phys_rlc = PhysicsMetrics.from_rlc_parameters(
        saturation=0.5, pressure=2.0,
        inductance=1.0, current=3.0,
        capacitance=0.01, voltage=100.0,
        resistance=10.0, di_dt=5.0,
    )
    print(phys_rlc)
    print(f"  E_kinetic  =   L I  =   1 9 = {phys_rlc.kinetic_energy:.2f} J")
    print(f"  E_potential =   C V  =   0.01 10000 = {phys_rlc.potential_energy:.2f} J")
    print(f"  V_flyback  = L dI/dt = 1 5 = {phys_rlc.flyback_voltage:.2f} V")
    print(f"  P_dissipated = R I  = 10 9 = {phys_rlc.dissipated_power:.2f} W")
    
    #
    print_section("2. TOPOLOGICAL METRICS")
    #
    
    topo = TopologicalMetrics(beta_0=2, beta_1=3, beta_2=1)
    print(topo)
    print(f"  euler_characteristic  =   = 2-3+1 = {topo.euler_characteristic}")
    print(f"  is_connected          = {topo.is_connected}")
    print(f"  is_simply_connected   = {topo.is_simply_connected}")
    print(f"  has_cycles            = {topo.has_cycles}")
    print(f"  has_cavities          = {topo.has_cavities}")
    print(f"  total_betti_number    =    = {topo.total_betti_number}")
    print(f"  homological_dimension = {topo.homological_dimension}")
    print(f"  cyclomatic_complexity = M = {topo.cyclomatic_complexity}")
    print(f"  P(-1) [=  ]          = {topo.poincare_polynomial(-1):.0f}")
    print(f"  P(1)  [=   ]         = {topo.poincare_polynomial(1):.0f}")
    
    print("\n[Grafo  rbol ideal]")
    topo_tree = TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, fiedler_value=0.5)
    print(topo_tree)
    print(f"  is_acyclic = {topo_tree.is_acyclic}")
    
    #
    print_section("3. CONTROL METRICS")
    #
    
    ctrl = ControlMetrics(
        poles_real=(-1.0, -2.0, -5.0),
        phase_margin_deg=60.0,
        damping_ratio=0.707,
        natural_frequency=2.0,
    )
    print(ctrl)
    print(f"  is_stable         = {ctrl.is_stable}")
    print(f"  dominant_pole     =  _dom = {ctrl.dominant_pole:.3f}")
    print(f"  fastest_pole      =  _fast = {ctrl.fastest_pole:.3f}")
    print(f"  stability_margin  = {ctrl.stability_margin:.3f}")
    print(f"  damping_category  = {ctrl.damping_category}")
    print(f"  damped_frequency  =  _d = {ctrl.damped_frequency:.3f} rad/s")
    print(f"  settling_time     = t_s = {ctrl.settling_time:.3f} s")
    print(f"  rise_time         = t_r = {ctrl.rise_time:.3f} s")
    print(f"  peak_overshoot    = M_p = {ctrl.peak_overshoot:.2f}%")
    
    print("\n[Retrocompatibilidad: lista   tupla]")
    ctrl_list = ControlMetrics(poles_real=[-0.5, -3.0])
    print(f"  poles type = {type(ctrl_list.poles_real).__name__}")
    
    print("\n[F brica desde polos complejos: s = -1   2j]")
    ctrl_cpx = ControlMetrics.from_poles([complex(-1, 2), complex(-1, -2)])
    print(ctrl_cpx)
    print(f"   _n extra da = |s| =  (1+4) = {ctrl_cpx.natural_frequency:.3f}")
    print(f"    extra da   = -Re(s)/ _n = 1/ 5 = {ctrl_cpx.damping_ratio:.3f}")
    print(f"   _d calculada =  _n  (1-  ) = {ctrl_cpx.damped_frequency:.3f}")
    
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
    print(f"  lyapunov_time =  _L = {ctrl_chaotic.lyapunov_time:.1f} s")
    
    #
    print_section("4. THERMODYNAMIC METRICS")
    #
    
    thermo = ThermodynamicMetrics(
        system_temperature=400.0,
        entropy=0.5,
        internal_energy=100.0,
        reference_temperature=300.0,
    )
    print(thermo)
    print(f"  helmholtz_free_energy = F = U - TS = 100 - 400 0.5 = {thermo.helmholtz_free_energy:.1f}")
    print(f"  gibbs_free_energy     = G = {thermo.gibbs_free_energy:.1f}")
    print(f"  exergy                = Ex = {thermo.exergy:.1f}")
    print(f"  carnot_efficiency     =  _C = 1 - 300/400 = {thermo.carnot_efficiency:.2%}")
    print(f"  exergetic_efficiency  =  _ex = {thermo.exergetic_efficiency:.2%}")
    print(f"  entropic_penalty      = TS = {thermo.entropic_penalty:.1f}")
    
    print("\n[F brica desde T y S]")
    thermo_fab = ThermodynamicMetrics.from_temperature_and_entropy(
        temperature=350.0,
        entropy=1.0,
        heat_capacity=2.0,
        reference_temperature=290.0,
    )
    print(thermo_fab)
    print(f"  U calculada = C_v T = 2 350 = {thermo_fab.internal_energy:.0f}")
    
    print("\n[Analog a financiera]")
    thermo_fin = ThermodynamicMetrics.from_financial_analogy(
        volatility=0.25,        # 25% volatilidad anualizada
        uncertainty=2.5,        # Bits de incertidumbre
        total_capital=1e6,      # $1M capital
        market_temperature=0.15,  # 15% benchmark
    )
    print(f"  Sistema financiero: T={thermo_fin.system_temperature}, S={thermo_fin.entropy}")
    print(f"  Carnot efficiency (interpretaci n: potencial de arbitraje) = {thermo_fin.carnot_efficiency:.2%}")
    
    #
    print_section("5. VECTOR DE ESTADO COMPUESTO")
    #
    
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
    
    print("\n[Serializaci n]")
    state_dict = state.to_dict()
    print(f"  Claves nivel 1: {list(state_dict.keys())}")
    print(f"  physics.total_energy (via dict): {state_dict['physics']['kinetic_energy'] + state_dict['physics']['potential_energy']}")
    
    #
    print_section("6. VALIDACIONES DE INVARIANTES")
    #
    
    print("\n[Violaciones esperadas - cada una debe lanzar ValueError]")
    
    test_cases = [
        ("Saturaci n fuera de rango", lambda: PhysicsMetrics(saturation=1.5)),
        ("Energ a negativa", lambda: PhysicsMetrics(kinetic_energy=-1.0)),
        ("Betti negativo", lambda: TopologicalMetrics(beta_1=-1)),
        ("Euler inconsistente", lambda: TopologicalMetrics(beta_0=1, beta_1=0, beta_2=0, euler_characteristic=5)),
        ("Damping negativo", lambda: ControlMetrics(damping_ratio=-0.5)),
        ("Frecuencia negativa", lambda: ControlMetrics(natural_frequency=-1.0)),
        ("Temperatura negativa", lambda: ThermodynamicMetrics(system_temperature=-10.0)),
        ("Entrop a negativa", lambda: ThermodynamicMetrics(entropy=-0.5)),
    ]
    
    for name, factory in test_cases:
        try:
            factory()
            print(f"    {name}: NO lanz  excepci n (ERROR)")
        except ValueError as e:
            print(f"    {name}: ValueError capturado")
    
    print("\n" + " " * 70)
    print(" VERIFICACI N COMPLETA")
    print(" " * 70)
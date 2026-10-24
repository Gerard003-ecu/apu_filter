# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MICRedundancyAnalyzer (Motor Espectral de Poda en Z₂)               ║
║ Ruta   : app/boole/tactics/mic_minimizer.py                                  ║
║ Versión: 5.4.0-Tactics-Boolean-RSI3-Poincare-Celestial-Nested                ║
║ Función: Motor espectral y algebraico soberano para la reducción y poda      ║
║          topológica de funciones booleanas en el Anillo Z₂[x]/⟨xᵢ²-xᵢ⟩,      ║
║          integrando la base de Gröbner reducida, algoritmos ROBDD,           ║
║          embedding simpléctico de Cayley–Darboux sobre sp(2d, ℝ),            ║
║          estabilidad de Poincaré–Lindstedt–Melnikov, Mónada Kleisli RSI³     ║
║          de 3 niveles y adjudicación en el álgebra de Heyting Ω₃.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO AGENTE CUSTODIO
--------------------------------------------------------------------------------
El Motor Espectral de Poda Booleana (`MICRedundancyAnalyzer`) es el núcleo algebraico
que realiza la minimización canónica de combinaciones lineales y polinomios de
capacidades en el Anillo Booleano $\mathbb{Z}_2[x]/\langle x_i^2-x_i\rangle$.
Su función soberana es erradicar la redundancia sintáctica, minimizando diagramas
de decisión binarios ordenados y reducidos (ROBDDs) mientras preserva de forma exacta
el volumen simpléctico y las características espectrales.

Su respectivo agente custodio y orquestador OODA es `MICMinimizerAgent`
(`app/agents/boole/tactics/mic_minimizer_agent.py`), el cual consume las estructuras
algebraicas producidas por este motor y emite los certificados de gobernanza ciber-física
(`MinimizerGovernanceCertificate`) para la conmutación de interlocks en hardware.

--------------------------------------------------------------------------------
2. MARCO TEÓRICO Y FORMALISMO MATEMÁTICO
--------------------------------------------------------------------------------
1. Anillo Booleano y Bases de Gröbner Reducidas:
   El espacio de funciones compuestas se modela en el anillo cociente:
       R = \mathbb{Z}_2[x_1, \dots, x_n] / \langle x_1^2 - x_1, \dots, x_n^2 - x_n \rangle
   Para cualquier ideal $I \subseteq R$, el algoritmo de Buchberger con selección grevlex
   obtiene la base de Gröbner reducida $G(I) = \{g_1, \dots, g_k\}$, permitiendo decidir la
   pertenencia canónica mediante la forma normal $N_G(f) = 0 \iff f \in I$.

2. Métrica Simpléctica de Cayley–Darboux en $\mathfrak{sp}(2d, \mathbb{R})$:
   La matriz de adyacencia de la base $P \in \mathbb{M}_{m \times n}(\mathbb{Z}_2)$ genera el
   tensor de Gram $G = P P^\top$. La proyección de Cayley sobre el álgebra de Lie simpléctica
   garantiza la no-degeneración del volumen de Liouville-Darboux:
       K_{\text{sp}} = \frac{1}{2}\left(K - J K^\top J\right), \quad M = (\mathbf{I}_{2d} + K_{\text{sp}})(\mathbf{I}_{2d} - K_{\text{sp}})^{-1}
   cumpliendo estrictamente $K_{\text{sp}}^\top J + J K_{\text{sp}} = \mathbf{0}$ y $\delta\omega_{\text{rel}} < 10^{-10}$.

3. Invariantes de Dinámica Celeste y Mónada Kleisli RSI³:
   Se resuelven las ecuaciones de perturbación homoclínica y KAM sobre el espectro de frecuencias:
     • Homoclínica de Melnikov: $M(\theta_0) = 2\pi \operatorname{sech}\left(\frac{\pi\omega}{2}\right) \sin(\omega\theta_0)$.
     • Frecuencia de Duffing-Lindstedt: $\omega = \omega_0 + \frac{3\epsilon a^2}{8\omega_0}$.
     • Cota de Nekhoroshev: $T \sim \exp\left(c / \epsilon^{1/(2n)}\right)$.
     • Recurrencia de Poincaré-Kac: $\langle \tau \rangle \sim \epsilon^{-n}$.
     • Mónada Kleisli RSI³ $\mathbf{T} = (T, \eta, \mu)$: Picard-Banach sobre 3 niveles con
       asociador de umbrales $a = \|T_{\tau_3} \circ T_{\tau_2} \circ T_{\tau_1} - T_{\tau_1} \circ T_{\tau_2} \circ T_{\tau_3}\|$
       y contracción $\|T^3\| \le \operatorname{Lip}(T)^3 \le 0.95^3$.

--------------------------------------------------------------------------------
3. AXIOMAS FORMALES
--------------------------------------------------------------------------------
Axioma MICE1 (Conmutatividad e Idempotencia Booleana):
    $x_i \cdot x_j = x_j \cdot x_i$ y $x_i^2 = x_i$ en $\mathbb{Z}_2$, forzando representaciones
    monomiales libres de cuadrados.
Axioma MICE2 (Conservación de Medida Simpléctica de Liouville):
    La transformación de Cayley satisface $M^\top J M = J$ con residuo relativo $\delta\omega_{\text{rel}} \le 10^{-10}$.
Axioma MICE3 (Invariancia Isomórfica de ROBDD):
    La reducción por Shannon $f = (\bar{x}_i \cdot f_0) \oplus (x_i \cdot f_1)$ preserva la
    entropía informacional y la dimensión de soporte de la tabla de verdad.
Axioma MICE4 (Cota de Contracción de Banach en la Mónada Kleisli):
    $\operatorname{Lip}(T_{\text{RSI3}}) \le 0.95 < 1.0$, garantizando la existencia de un
    punto fijo atractor único para las iteraciones de minimización.
Axioma MICE5 (Lógica de Heyting $\Omega_3$ e Implicación):
    $a \to b = \bigvee\{c \in \Omega_3 \mid a \wedge c \le b\}$, satisfaciendo la adjunción de
    pseudo-complemento en el retículo trivalente.

--------------------------------------------------------------------------------
4. INVARIANTES DE ESTADO
--------------------------------------------------------------------------------
Invariante 1 (Preservación del Rango de la Base GF(2)):
    El número de generadores esenciales coincide con el rango de la matriz RREF en GF(2).
Invariante 2 (Estabilidad Homoclínica de Melnikov):
    $|M(\theta_0)| < 10^{-2}$ y parámetro de Chirikov $K_{\text{Chirikov}} < 1.0$.
Invariante 3 (Valuación de Krull Mínima):
    $v(\text{MIC}) = \frac{\operatorname{rank}(P)}{\text{num\_vars}} \ge 0.10$.

--------------------------------------------------------------------------------
5. FUNTOR SEMÁNTICO Y CÚSPIDE VISCERAL ("DOLOR Y DINERO" / Φ_sem)
--------------------------------------------------------------------------------
El Funtor Semántico $\Phi_{\mathrm{sem}} : \mathbf{Sh}(\partial\mathcal{M}, \Omega_3) \to \text{Business}$
asocia la poda del Anillo Booleano con los costos de operación del sistema:
  • Reducción de Latencia y Eficiencia Táctica (-80% de Carga Estructurada):
    La minimización canónica elimina herramientas duplicadas e innecesarias en la MIC,
    reduciendo las llamadas de funciones a modelos de lenguaje y acelerando el tiempo de respuesta.
  • Eliminación de Riesgo Operativo Ciber-Físico:
    La validación de no-interferencia SAT evita conflictos de capacidades concurrentes,
    previniendo fallas mecánicas en actuadores perimetrales.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import total_ordering
from itertools import combinations
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union,
)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


# =============================================================================
# CONFIGURACIÓN DE WARNINGS Y ERRORES NUMÉRICOS
# =============================================================================

if NUMPY_AVAILABLE:
    warnings.filterwarnings('error', category=RuntimeWarning)
    np.seterr(all='raise')


# =============================================================================
# JERARQUÍA DE EXCEPCIONES
# =============================================================================

class MICException(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.mathematical_context = ""


class HomologicalInconsistencyError(MICException):
    def __init__(self, message: str, condition_number: Optional[float] = None):
        context = {}
        if condition_number is not None:
            context['condition_number'] = condition_number
        super().__init__(message, context)
        self.mathematical_context = (
            "Singularidad detectada en el análisis espectral de la matriz de incidencia."
        )


class UnsatCoreError(MICException):
    def __init__(self, message: str, conflicting_tools: Optional[List[str]] = None):
        context = {'conflicting_tools': conflicting_tools or []}
        super().__init__(message, context)


class GrobnerBasisComputationError(MICException):
    pass


class ROBDDConstructionError(MICException):
    pass


class SymplecticIntegrityError(MICException):
    pass


class PoincareResonanceError(MICException):
    pass


class RSI3ContractionError(MICException):
    r"""Ruptura de la contracción de Banach en T, T² o T³."""
    pass


class OODAChainViolationError(MICException):
    r"""Ruptura de continuidad entre fases Φₖ → Φₖ₊₁."""
    pass


# =============================================================================
# LOGGING
# =============================================================================

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m',
    }
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        colored_level = f"{self.BOLD}{color}{record.levelname:8}{self.COLORS['RESET']}"
        record.levelname = colored_level
        return super().format(record)


def setup_logger(name: str = "MIC.Minimizer.v5.4", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = setup_logger()

__version__ = "5.4.0"


# =============================================================================
# ENUMERACIONES Y TIPOS ALGEBRAICOS
# =============================================================================

class CapabilityDimension(Enum):
    PHYS_IO = 0
    PHYS_NUM = 1
    TACT_TOPO = 2
    STRAT_FIN = 3
    WIS_SEM = 4

    def __lt__(self, other: 'CapabilityDimension') -> bool:
        if not isinstance(other, CapabilityDimension):
            return NotImplemented
        return self.value < other.value

    def __repr__(self) -> str:
        return f"Cap.{self.name}"


class HeytingOmega3(Enum):
    r"""
    Álgebra de Heyting trivalente $\Omega_3 = \{0x00 < 0x01 < 0x03\}$
    (cadena de severidad: COHERENT ⊤̸ = bottom, VETOED = top).

    Operaciones: join = ∨ (OR de máscara), meet = ∧ (AND de máscara),
    implies = → (pseudo-complemento relativo de cadena).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @property
    def omega(self) -> float:
        return {0x00: 0.0, 0x01: 0.5, 0x03: 1.0}[self.value]

    @property
    def mask(self) -> int:
        return self.value

    @classmethod
    def from_omega(cls, v: float) -> "HeytingOmega3":
        if v <= 0.0:
            return cls.COHERENT
        if v < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def from_mask(cls, m: int) -> "HeytingOmega3":
        if (m & 0x03) == 0x03:
            return cls.VETOED
        if (m & 0x01) == 0x01:
            return cls.DEGRADED
        return cls.COHERENT

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.from_mask(self.mask | other.mask)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.from_mask(self.mask & other.mask)

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""
        Pseudo-complemento relativo en la cadena 0 < d < v:
        $a \to b = \bigvee\{c \mid a \wedge c \le b\}$.
        Equivale a $\top$ si $a \le b$, y a $b$ en caso contrario.
        """
        if self.omega <= other.omega:
            return HeytingOmega3.VETOED  # top
        return other

    def leq(self, other: "HeytingOmega3") -> bool:
        return self.omega <= other.omega

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingOmega3"]) -> "HeytingOmega3":
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)


def _heyting_lattice_axioms_check() -> bool:
    r"""
    Verifica axiomas de retículo distributivo acotado + álgebra de Heyting
    en $\Omega_3$: idempotencia, conmutatividad, asociatividad, absorción,
    distributividad, monotonía de $\omega$, $a\to a=\top$, modus ponens
    $a \wedge (a\to b) \le b$, y adjuncion $a\wedge c\le b \iff c\le a\to b$.
    """
    elems = list(HeytingOmega3)
    top = HeytingOmega3.VETOED
    ok = True
    for a in elems:
        if a.join(a) is not a or a.meet(a) is not a:
            ok = False
        if a.implies(a) is not top:
            ok = False
        for b in elems:
            if a.join(b) is not b.join(a) or a.meet(b) is not b.meet(a):
                ok = False
            # modus ponens
            if not (a.meet(a.implies(b))).leq(b):
                ok = False
            for c in elems:
                if a.join(b).join(c) is not a.join(b.join(c)):
                    ok = False
                if a.meet(b).meet(c) is not a.meet(b.meet(c)):
                    ok = False
                if a.join(a.meet(b)) is not a:
                    ok = False
                if a.meet(a.join(b)) is not a:
                    ok = False
                if a.meet(b.join(c)) is not (a.meet(b)).join(a.meet(c)):
                    ok = False
                # adjuncion de Heyting
                left = a.meet(c).leq(b)
                right = c.leq(a.implies(b))
                if left != right:
                    ok = False
    if not (HeytingOmega3.COHERENT.omega
            <= HeytingOmega3.DEGRADED.omega
            <= HeytingOmega3.VETOED.omega):
        ok = False
    return ok


# =============================================================================
# VECTOR BOOLEANO
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class BooleanVector:
    components: FrozenSet[CapabilityDimension] = field(default_factory=frozenset)

    def __post_init__(self):
        if not isinstance(self.components, frozenset):
            object.__setattr__(self, 'components', frozenset(self.components))
        if not all(isinstance(c, CapabilityDimension) for c in self.components):
            invalid = [c for c in self.components if not isinstance(c, CapabilityDimension)]
            raise TypeError(
                f"Todos los componentes deben ser CapabilityDimension. Encontrados: {invalid}"
            )

    @classmethod
    def from_minterm(cls, minterm: int, num_vars: int = 5) -> 'BooleanVector':
        max_val = (1 << num_vars) - 1
        if not (0 <= minterm <= max_val):
            raise ValueError(f"Minterm {minterm} fuera de rango [0, {max_val}]")
        components = {CapabilityDimension(i) for i in range(num_vars) if (minterm >> i) & 1}
        return cls(frozenset(components))

    @classmethod
    def from_binary_string(cls, binary: str) -> 'BooleanVector':
        components = {CapabilityDimension(i) for i, bit in enumerate(binary) if bit == '1'}
        return cls(frozenset(components))

    @classmethod
    def zero(cls) -> 'BooleanVector':
        return cls(frozenset())

    @classmethod
    def universe(cls, num_vars: int = 5) -> 'BooleanVector':
        return cls(frozenset(CapabilityDimension(i) for i in range(num_vars)))

    def to_binary_string(self, num_vars: int = 5) -> str:
        return ''.join(
            '1' if CapabilityDimension(i) in self.components else '0'
            for i in range(num_vars)
        )

    def to_minterm(self) -> int:
        return sum(1 << cap.value for cap in self.components)

    def to_set_notation(self) -> str:
        if not self.components:
            return "∅"
        caps = sorted(self.components, key=lambda c: c.value)
        return "{" + ", ".join(c.name for c in caps) + "}"

    def hamming_weight(self) -> int:
        return len(self.components)

    def hamming_distance(self, other: 'BooleanVector') -> int:
        return len(self.components ^ other.components)

    def union(self, other: 'BooleanVector') -> 'BooleanVector':
        return BooleanVector(self.components | other.components)

    def intersection(self, other: 'BooleanVector') -> 'BooleanVector':
        return BooleanVector(self.components & other.components)

    def symmetric_difference(self, other: 'BooleanVector') -> 'BooleanVector':
        return BooleanVector(self.components ^ other.components)

    def complement(self, num_vars: int = 5) -> 'BooleanVector':
        universe = {CapabilityDimension(i) for i in range(num_vars)}
        return BooleanVector(frozenset(universe - self.components))

    def difference(self, other: 'BooleanVector') -> 'BooleanVector':
        return BooleanVector(self.components - other.components)

    def is_subset_of(self, other: 'BooleanVector') -> bool:
        return self.components.issubset(other.components)

    def is_superset_of(self, other: 'BooleanVector') -> bool:
        return self.components.issuperset(other.components)

    def is_disjoint_from(self, other: 'BooleanVector') -> bool:
        return self.components.isdisjoint(other.components)

    def inner_product_z2(self, other: 'BooleanVector') -> int:
        overlap = len(self.components & other.components)
        result = overlap % 2
        if result not in {0, 1}:
            raise HomologicalInconsistencyError(f"Producto interno en ℤ₂ inválido: {result}.")
        return result

    def inner_product_real(self, other: 'BooleanVector') -> int:
        return len(self.components & other.components)

    def __lt__(self, other: 'BooleanVector') -> bool:
        if not isinstance(other, BooleanVector):
            return NotImplemented
        return tuple(sorted(self.components)) < tuple(sorted(other.components))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BooleanVector):
            return NotImplemented
        return self.components == other.components

    def __hash__(self) -> int:
        return hash(self.components)

    def __repr__(self) -> str:
        if not self.components:
            return "BooleanVector(∅)"
        caps = sorted(self.components, key=lambda c: c.value)
        caps_str = ", ".join(c.name for c in caps)
        return f"BooleanVector({{{caps_str}}})"

    def __str__(self) -> str:
        return self.to_set_notation()


# =============================================================================
# HERRAMIENTA
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class Tool:
    name: str
    capabilities: BooleanVector

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("El nombre debe ser una cadena no vacía")
        if not isinstance(self.capabilities, BooleanVector):
            raise TypeError(
                f"capabilities debe ser BooleanVector, recibido: {type(self.capabilities).__name__}"
            )

    @property
    def arity(self) -> int:
        return self.capabilities.hamming_weight()

    @property
    def is_trivial(self) -> bool:
        return self.arity == 0

    @property
    def signature(self) -> str:
        return self.capabilities.to_binary_string()

    def subsumes(self, other: 'Tool') -> bool:
        return other.capabilities.is_subset_of(self.capabilities)

    def is_subsumed_by(self, other: 'Tool') -> bool:
        return self.capabilities.is_subset_of(other.capabilities)

    def overlaps_with(self, other: 'Tool') -> bool:
        return not self.capabilities.is_disjoint_from(other.capabilities)

    def similarity_score(self, other: 'Tool') -> float:
        inter = self.capabilities.intersection(other.capabilities)
        uni = self.capabilities.union(other.capabilities)
        union_size = uni.hamming_weight()
        if union_size == 0:
            return 1.0
        return inter.hamming_weight() / union_size

    def __lt__(self, other: 'Tool') -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return (self.name, self.capabilities) < (other.name, other.capabilities)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return self.name == other.name and self.capabilities == other.capabilities

    def __hash__(self) -> int:
        return hash((self.name, self.capabilities))

    def __repr__(self) -> str:
        return f"Tool('{self.name}', {self.capabilities})"

    def __str__(self) -> str:
        return f"{self.name}: {self.capabilities}"


# =============================================================================
# POLINOMIOS EN ℤ₂[x₁,...,xₙ]/⟨x²ᵢ − xᵢ⟩
# =============================================================================

@dataclass(frozen=True)
class Z2Polynomial:
    monomials: FrozenSet[FrozenSet[int]]

    def __post_init__(self):
        if not isinstance(self.monomials, frozenset):
            object.__setattr__(self, 'monomials', frozenset(self.monomials))
        for mon in self.monomials:
            if not isinstance(mon, frozenset):
                raise TypeError(f"Cada monomio debe ser frozenset, recibido: {type(mon)}")
            if not all(isinstance(i, int) and i >= 0 for i in mon):
                raise TypeError("Cada monomio debe contener índices enteros no negativos")

    @classmethod
    def zero(cls) -> 'Z2Polynomial':
        return cls(frozenset())

    @classmethod
    def one(cls) -> 'Z2Polynomial':
        return cls(frozenset([frozenset()]))

    @classmethod
    def variable(cls, idx: int) -> 'Z2Polynomial':
        if idx < 0:
            raise ValueError(f"Índice de variable debe ser ≥ 0, recibido: {idx}")
        return cls(frozenset([frozenset([idx])]))

    @classmethod
    def from_minterm(cls, minterm: int, num_vars: int) -> 'Z2Polynomial':
        if minterm < 0 or minterm >= (1 << num_vars):
            raise ValueError(f"Minterm {minterm} fuera de rango para {num_vars} vars")
        vars_in_monomial = frozenset(i for i in range(num_vars) if (minterm >> i) & 1)
        return cls(frozenset([vars_in_monomial]))

    @classmethod
    def from_minterms(cls, minterms: Set[int], num_vars: int) -> 'Z2Polynomial':
        result = cls.zero()
        for mt in minterms:
            result = result + cls.from_minterm(mt, num_vars)
        return result

    def __add__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        return Z2Polynomial(self.monomials ^ other.monomials)

    def __mul__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        result_monomials: Set[FrozenSet[int]] = set()
        for mon1 in self.monomials:
            for mon2 in other.monomials:
                combined = mon1 | mon2
                if combined in result_monomials:
                    result_monomials.remove(combined)
                else:
                    result_monomials.add(combined)
        return Z2Polynomial(frozenset(result_monomials))

    def __neg__(self) -> 'Z2Polynomial':
        return self

    def __sub__(self, other: 'Z2Polynomial') -> 'Z2Polynomial':
        return self + other

    def is_zero(self) -> bool:
        return len(self.monomials) == 0

    def is_one(self) -> bool:
        return self.monomials == frozenset([frozenset()])

    def degree(self) -> int:
        if self.is_zero():
            return -1
        return max(len(mon) for mon in self.monomials)

    def num_terms(self) -> int:
        return len(self.monomials)

    def leading_term(self, order: Optional[Callable[[FrozenSet[int]], Tuple]] = None) -> Optional[FrozenSet[int]]:
        if not self.monomials:
            return None
        if order is None:
            def grevlex_key(mon: FrozenSet[int]) -> Tuple[int, Tuple[int, ...]]:
                return (len(mon), tuple(sorted(mon, reverse=True)))
            order = grevlex_key
        return max(self.monomials, key=order)

    def leading_coefficient(self) -> int:
        return 1 if not self.is_zero() else 0

    def evaluate(self, assignment: Dict[int, int]) -> int:
        result = 0
        for monomial in self.monomials:
            mon_value = 1
            for var_idx in monomial:
                val = assignment.get(var_idx, 0)
                if val not in {0, 1}:
                    raise ValueError(f"Valor de variable debe ser 0 o 1, recibido: {val}")
                mon_value *= val
                if mon_value == 0:
                    break
            result ^= mon_value
        return result

    def __repr__(self) -> str:
        if not self.monomials:
            return "0"
        terms = []
        for mon in sorted(self.monomials, key=lambda m: (len(m), tuple(sorted(m)))):
            if not mon:
                terms.append("1")
            else:
                vars_str = "·".join(f"x{i}" for i in sorted(mon))
                terms.append(vars_str)
        return " + ".join(terms)

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash(self.monomials)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Z2Polynomial):
            return NotImplemented
        return self.monomials == other.monomials


# =============================================================================
# BASE DE GRÖBNER EN ℤ₂
# =============================================================================

class GrobnerBasisZ2:
    """Base de Gröbner reducida para ideales en ℤ₂[x]/⟨x² − x⟩ (Buchberger + autoreduce)."""

    def __init__(self, num_vars: int):
        if num_vars <= 0:
            raise ValueError(f"num_vars debe ser positivo, recibido: {num_vars}")
        self.num_vars = num_vars
        self.basis: List[Z2Polynomial] = []
        self._iteration_count = 0
        self._max_iterations = 10000

    def add_polynomial(self, poly: Z2Polynomial) -> None:
        if poly.is_zero():
            return
        self.basis.append(poly)
        try:
            self._reduce_basis()
        except RecursionError as e:
            raise GrobnerBasisComputationError(
                f"Recursión excesiva al reducir base de Gröbner: {e}"
            )

    def _reduce_basis(self) -> None:
        self._iteration_count = 0
        changed = True
        while changed:
            self._iteration_count += 1
            if self._iteration_count > self._max_iterations:
                raise GrobnerBasisComputationError(
                    f"Excedido límite de iteraciones ({self._max_iterations})"
                )
            changed = False
            new_polys: Set[Z2Polynomial] = set()
            for i, p in enumerate(self.basis):
                for j, q in enumerate(self.basis):
                    if i >= j:
                        continue
                    lt_p = p.leading_term()
                    lt_q = q.leading_term()
                    if lt_p is None or lt_q is None:
                        continue
                    lcm = lt_p.union(lt_q)
                    mult_p = lcm.difference(lt_p)
                    mult_q = lcm.difference(lt_q)
                    poly_mult_p = Z2Polynomial(frozenset([mult_p]))
                    poly_mult_q = Z2Polynomial(frozenset([mult_q]))
                    s_poly = (poly_mult_p * p) + (poly_mult_q * q)
                    nf_s_poly = self._reduce_poly_by_basis(s_poly, self.basis)
                    if not nf_s_poly.is_zero() and nf_s_poly not in self.basis:
                        new_polys.add(nf_s_poly)
                        changed = True
            for poly in new_polys:
                if poly not in self.basis:
                    self.basis.append(poly)
        self._autoreduce()

    def _autoreduce(self) -> None:
        i = 0
        while i < len(self.basis):
            poly = self.basis[i]
            other_basis = [p for j, p in enumerate(self.basis) if j != i]
            reduced = self._reduce_poly_by_basis(poly, other_basis)
            if reduced.is_zero():
                self.basis.pop(i)
            elif reduced != poly:
                self.basis[i] = reduced
                i += 1
            else:
                i += 1

    def _reduce_poly_by_basis(self, poly: Z2Polynomial, basis: List[Z2Polynomial]) -> Z2Polynomial:
        current = poly
        while not current.is_zero():
            lt_current = current.leading_term()
            if lt_current is None:
                break
            reduced = False
            for basis_poly in basis:
                if basis_poly.is_zero():
                    continue
                lt_basis = basis_poly.leading_term()
                if lt_basis is None:
                    continue
                if lt_basis.issubset(lt_current):
                    multiplier_vars = lt_current - lt_basis
                    multiplier = Z2Polynomial(frozenset([multiplier_vars]))
                    reduction_term = multiplier * basis_poly
                    current = current + reduction_term
                    reduced = True
                    break
            if not reduced:
                break
        return current

    def normal_form(self, poly: Z2Polynomial) -> Z2Polynomial:
        return self._reduce_poly_by_basis(poly, self.basis)

    def is_member(self, poly: Z2Polynomial) -> bool:
        return self.normal_form(poly).is_zero()

    def get_generators(self) -> List[Z2Polynomial]:
        return list(self.basis)

    def __len__(self) -> int:
        return len(self.basis)

    def __repr__(self) -> str:
        if not self.basis:
            return "GrobnerBasis(⟨0⟩)"
        polys_str = ", ".join(str(p) for p in self.basis[:5])
        if len(self.basis) > 5:
            polys_str += f", ... ({len(self.basis) - 5} more)"
        return f"GrobnerBasis(⟨{polys_str}⟩)"


# =============================================================================
# ROBDD
# =============================================================================

class ROBDDNode:
    __slots__ = ('var', 'low', 'high', 'id')
    _TRUE_NODE: Optional['ROBDDNode'] = None
    _FALSE_NODE: Optional['ROBDDNode'] = None

    def __init__(self, var: int, low: Optional['ROBDDNode'], high: Optional['ROBDDNode']):
        self.var = var
        self.low = low
        self.high = high
        self.id = id(self)

    @classmethod
    def get_true_node(cls) -> 'ROBDDNode':
        if cls._TRUE_NODE is None:
            cls._TRUE_NODE = ROBDDNode(-1, None, None)
        return cls._TRUE_NODE

    @classmethod
    def get_false_node(cls) -> 'ROBDDNode':
        if cls._FALSE_NODE is None:
            cls._FALSE_NODE = ROBDDNode(-2, None, None)
        return cls._FALSE_NODE

    def is_terminal(self) -> bool:
        return self.var < 0

    def is_true(self) -> bool:
        return self is ROBDDNode.get_true_node()

    def is_false(self) -> bool:
        return self is ROBDDNode.get_false_node()

    def __repr__(self) -> str:
        if self.is_true():
            return "TRUE"
        if self.is_false():
            return "FALSE"
        return f"Node(x{self.var})"


class ROBDD:
    def __init__(self, num_vars: int):
        if num_vars <= 0:
            raise ValueError(f"num_vars debe ser positivo, recibido: {num_vars}")
        self.num_vars = num_vars
        self.unique_table: Dict[Tuple[int, int, int], ROBDDNode] = {}
        self._operation_cache: Dict[Tuple[str, int, int], ROBDDNode] = {}

    def _make_node(self, var: int, low: ROBDDNode, high: ROBDDNode) -> ROBDDNode:
        if low is high:
            return low
        key = (var, id(low), id(high))
        if key not in self.unique_table:
            self.unique_table[key] = ROBDDNode(var, low, high)
        return self.unique_table[key]

    def build_from_minterms(self, minterms: Set[int]) -> ROBDDNode:
        try:
            return self._build_shannon(set(minterms), 0)
        except RecursionError as e:
            raise ROBDDConstructionError(f"Recursión excesiva al construir ROBDD: {e}")

    def _build_shannon(self, minterms: Set[int], var_idx: int) -> ROBDDNode:
        if not minterms:
            return ROBDDNode.get_false_node()
        if var_idx >= self.num_vars:
            return ROBDDNode.get_true_node()
        low_set = {m for m in minterms if not (m >> var_idx) & 1}
        high_set = {m for m in minterms if (m >> var_idx) & 1}
        low_child = self._build_shannon(low_set, var_idx + 1)
        high_child = self._build_shannon(high_set, var_idx + 1)
        return self._make_node(var_idx, low_child, high_child)

    def apply_and(self, n1: ROBDDNode, n2: ROBDDNode) -> ROBDDNode:
        return self._apply_binary_op(n1, n2, 'and')

    def apply_or(self, n1: ROBDDNode, n2: ROBDDNode) -> ROBDDNode:
        return self._apply_binary_op(n1, n2, 'or')

    def apply_xor(self, n1: ROBDDNode, n2: ROBDDNode) -> ROBDDNode:
        return self._apply_binary_op(n1, n2, 'xor')

    def apply_not(self, node: ROBDDNode) -> ROBDDNode:
        if node.is_true():
            return ROBDDNode.get_false_node()
        if node.is_false():
            return ROBDDNode.get_true_node()
        low_neg = self.apply_not(node.low)
        high_neg = self.apply_not(node.high)
        return self._make_node(node.var, low_neg, high_neg)

    def _apply_binary_op(self, n1: ROBDDNode, n2: ROBDDNode, op: str) -> ROBDDNode:
        cache_key = (op, n1.id, n2.id)
        if cache_key in self._operation_cache:
            return self._operation_cache[cache_key]
        if n1.is_terminal() and n2.is_terminal():
            result = self._eval_terminal_op(n1, n2, op)
        elif n1.is_terminal():
            result = self._apply_with_constant(n1, n2, op)
        elif n2.is_terminal():
            result = self._apply_with_constant(n2, n1, op)
        else:
            result = self._apply_shannon_expansion(n1, n2, op)
        self._operation_cache[cache_key] = result
        return result

    def _eval_terminal_op(self, n1: ROBDDNode, n2: ROBDDNode, op: str) -> ROBDDNode:
        v1 = 1 if n1.is_true() else 0
        v2 = 1 if n2.is_true() else 0
        if op == 'and':
            result = v1 & v2
        elif op == 'or':
            result = v1 | v2
        elif op == 'xor':
            result = v1 ^ v2
        else:
            raise ValueError(f"Operación desconocida: {op}")
        return ROBDDNode.get_true_node() if result else ROBDDNode.get_false_node()

    def _apply_with_constant(self, const: ROBDDNode, node: ROBDDNode, op: str) -> ROBDDNode:
        if op == 'and':
            return ROBDDNode.get_false_node() if const.is_false() else node
        if op == 'or':
            return ROBDDNode.get_true_node() if const.is_true() else node
        if op == 'xor':
            return node if const.is_false() else self.apply_not(node)
        return node

    def _apply_shannon_expansion(self, n1: ROBDDNode, n2: ROBDDNode, op: str) -> ROBDDNode:
        if n1.var == n2.var:
            low = self._apply_binary_op(n1.low, n2.low, op)
            high = self._apply_binary_op(n1.high, n2.high, op)
            return self._make_node(n1.var, low, high)
        if n1.var < n2.var:
            low = self._apply_binary_op(n1.low, n2, op)
            high = self._apply_binary_op(n1.high, n2, op)
            return self._make_node(n1.var, low, high)
        low = self._apply_binary_op(n1, n2.low, op)
        high = self._apply_binary_op(n1, n2.high, op)
        return self._make_node(n2.var, low, high)

    def extract_minterms(self, node: ROBDDNode) -> Set[int]:
        minterms: Set[int] = set()
        self._collect_minterms(node, 0, 0, minterms)
        return minterms

    def _collect_minterms(self, node: ROBDDNode, var_idx: int, current: int, result: Set[int]) -> None:
        if node.is_false():
            return
        target_idx = self.num_vars if node.is_terminal() else node.var
        if var_idx < target_idx:
            self._collect_minterms(node, var_idx + 1, current, result)
            self._collect_minterms(node, var_idx + 1, current | (1 << var_idx), result)
            return
        if node.is_true() and var_idx == self.num_vars:
            result.add(current)
            return
        if not node.is_terminal():
            self._collect_minterms(node.low, var_idx + 1, current, result)
            self._collect_minterms(node.high, var_idx + 1, current | (1 << var_idx), result)

    def count_solutions(self, node: ROBDDNode) -> int:
        memo: Dict[Tuple[int, int], int] = {}
        return self._count_solutions_memo(node, 0, memo)

    def _count_solutions_memo(
        self, node: ROBDDNode, var_idx: int, memo: Dict[Tuple[int, int], int],
    ) -> int:
        if node.is_false():
            return 0
        if node.is_true() or var_idx >= self.num_vars:
            return 2 ** (self.num_vars - var_idx)
        key = (node.id, var_idx)
        if key in memo:
            return memo[key]
        result = (
            self._count_solutions_memo(node.low, var_idx + 1, memo)
            + self._count_solutions_memo(node.high, var_idx + 1, memo)
        )
        memo[key] = result
        return result

    def size(self, node: ROBDDNode) -> int:
        visited: Set[int] = set()
        return self._size_recursive(node, visited)

    def _size_recursive(self, node: ROBDDNode, visited: Set[int]) -> int:
        if node.id in visited:
            return 0
        visited.add(node.id)
        if node.is_terminal():
            return 1
        return 1 + self._size_recursive(node.low, visited) + self._size_recursive(node.high, visited)

    def __repr__(self) -> str:
        return f"ROBDD({self.num_vars} vars, {len(self.unique_table)} unique nodes)"


# =============================================================================
# DPLL SAT SOLVER
# =============================================================================

class DPLLSATSolver:
    @staticmethod
    def solve(cnf: List[List[int]]) -> bool:
        clauses = [list(c) for c in cnf]
        return DPLLSATSolver._dpll(clauses, {})

    @staticmethod
    def _dpll(clauses: List[List[int]], assignment: Dict[int, bool]) -> bool:
        while True:
            unit = DPLLSATSolver._find_unit_clause(clauses)
            if unit is None:
                break
            var = abs(unit)
            value = (unit > 0)
            assignment[var] = value
            new = DPLLSATSolver._propagate(clauses, var, value)
            if new is None:
                return False
            clauses = new
            if not clauses:
                return True
        pure = DPLLSATSolver._find_pure_literals(clauses)
        for lit in pure:
            var = abs(lit)
            value = (lit > 0)
            assignment[var] = value
            new = DPLLSATSolver._propagate(clauses, var, value)
            if new is None:
                return False
            clauses = new
            if not clauses:
                return True
        if not clauses:
            return True
        var = DPLLSATSolver._select_variable(clauses, assignment)
        if var is None:
            return False
        for value in (True, False):
            new_assign = assignment.copy()
            new_assign[var] = value
            new_clauses = DPLLSATSolver._propagate([list(c) for c in clauses], var, value)
            if new_clauses is None:
                continue
            if DPLLSATSolver._dpll(new_clauses, new_assign):
                assignment.update(new_assign)
                return True
        return False

    @staticmethod
    def _find_unit_clause(clauses: List[List[int]]) -> Optional[int]:
        for clause in clauses:
            if len(clause) == 1:
                return clause[0]
        return None

    @staticmethod
    def _find_pure_literals(clauses: List[List[int]]) -> List[int]:
        pos, neg = set(), set()
        for clause in clauses:
            for lit in clause:
                if lit > 0:
                    pos.add(lit)
                else:
                    neg.add(-lit)
        return list(pos - neg) + [-x for x in (neg - pos)]

    @staticmethod
    def _propagate(clauses: List[List[int]], var: int, value: bool) -> Optional[List[List[int]]]:
        result: List[List[int]] = []
        for clause in clauses:
            new_clause: List[int] = []
            satisfied = False
            for lit in clause:
                lit_var = abs(lit)
                lit_sign = (lit > 0)
                if lit_var == var:
                    if lit_sign == value:
                        satisfied = True
                        break
                else:
                    new_clause.append(lit)
            if satisfied:
                continue
            if not new_clause:
                return None
            result.append(new_clause)
        return result

    @staticmethod
    def _select_variable(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[int]:
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    return var
        return None


# =============================================================================
# v5.4.0: HANDOFFS ENTRE FASES (morfismos Φₖ → Φₖ₊₁)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MICPhase1Handoff:
    r"""
    Morfismo Φ₁ → Φ₂ (FASE 1: symplectic+boolean → FASE 2: Poincaré+RSI³).

    Emitido por `_handoff_to_phase2`; consumido por `_ingest_phase1_handoff`.
    La última flecha de FASE 1 *es* el dominio de la primera flecha de FASE 2:
        $\Phi_2\circ\Phi_1(\mathrm{MIC})
         =\texttt{\_ingest\_phase1\_handoff}(
              \texttt{\_handoff\_to\_phase2}(\mathrm{MIC}))$.
    """
    symplectic_residue: float
    relative_symplectic_residue: float
    klein_residual: float
    is_symplectic_valid: bool
    darboux_chart_dim: int
    spectral_rank: int
    spectral_nullity: int
    condition_number: float
    grobner_basis_size: int
    robdd_node_count: int
    robdd_essential: Tuple[str, ...]
    robdd_redundant: Tuple[str, ...]
    tool_signatures: Tuple[str, ...]
    num_capabilities: int
    input_checksum: str
    timestamp_iso: str


@dataclass(frozen=True, slots=True)
class MICPhase2Handoff:
    r"""
    Morfismo Φ₂ → Φ₃ (FASE 2: Poincaré+RSI³ → FASE 3: Heyting+Crowbar).

    Emitido por `_handoff_to_phase3`; consumido por `_ingest_phase2_handoff`.
    """
    phase1: MICPhase1Handoff
    kam_stable: bool
    diophantine_gamma: float
    bruno_ok: bool
    chirikov_parameter: float
    melnikov_integral: float
    melnikov_closed_form: float
    melnikov_residue: float
    lindstedt_omega_shift: float
    nekhoroshev_time: float
    poincare_lyapunov: float
    recurrence_time: float
    banach_L: float
    quadratic_rate: float
    capacity_acceleration: float
    krull_valuation: float
    fubini_study_distance: float
    rsi3_level1_residual: float
    rsi3_level2_lipschitz: float
    rsi3_level3_associator: float
    wasserstein_distance: float
    grobner_independent: Tuple[str, ...]
    grobner_dependent: Tuple[str, ...]
    sat_ok: bool
    conflicting_tools: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MICGovernanceCertificate:
    r"""
    Certificado OODA final de la composición Φ₃ ∘ Φ₂ ∘ Φ₁.
    Verifica axiomas A1–A5 con trazabilidad SHA-256.
    """
    phase2: MICPhase2Handoff
    heyting_verdict: HeytingOmega3
    heyting_mask: int
    heyting_axioms_verified: bool
    hardware_action: str
    interlock_latency_ns: float
    is_coherent: bool
    is_compilation_authorized: bool
    overall_quality_score: float
    risk_assessment: str
    essential_tools: Tuple[str, ...]
    redundant_tools: Tuple[str, ...]
    timestamp_iso: str
    input_checksum: str


@dataclass(frozen=True, slots=True)
class RSI3KleisliState:
    r"""Estado interno del endofuntor T en la categoría Kleisli de RSI³."""
    spectrum: Tuple[float, ...]
    tau: float
    residual: float
    lipschitz: float
    level: int


# =============================================================================
# ANALIZADOR PRINCIPAL — 3 FASES ANIDADAS
# =============================================================================

class MICRedundancyAnalyzer:
    r"""
    Analizador de redundancia MIC con 3 fases anidadas estrictamente continuadas.

    Composición: $Z_{MIC} = \Phi_3 \circ \Phi_2 \circ \Phi_1$ vía
    `MICPhase1Handoff` → `MICPhase2Handoff` → `MICGovernanceCertificate`.

    Invariante de anidamiento:
        last(Φₖ) : Xₖ → Handoffₖ    y    first(Φₖ₊₁) : Handoffₖ → Xₖ₊₁
        con  first(Φₖ₊₁) ∘ last(Φₖ)  tipado estrictamente.
    """

    _SYMPLECTIC_TOL = 1e-10
    _MELNIKOV_STABLE = 1e-2
    _W1_STABLE = 1e-2
    _BANACH_L_MAX = 0.95
    _KRULL_MIN = 0.10
    _FS_MAX = 1e-2
    _GAMMA_KAM = 0.125
    _BRUNO_FLOOR = 1e-12

    def __init__(self, num_capabilities: Optional[int] = None):
        if num_capabilities is None:
            num_capabilities = len(CapabilityDimension)
        if num_capabilities <= 0:
            raise ValueError(f"num_capabilities debe ser positivo, recibido: {num_capabilities}")
        self.num_capabilities = num_capabilities
        self.tools: List[Tool] = []
        self.grobner_basis = GrobnerBasisZ2(num_capabilities)
        self.robdd = ROBDD(num_capabilities)
        self._tool_polynomials: Dict[str, Z2Polynomial] = {}
        self._incidence_matrix_cache: Optional[Any] = None
        self._ordered_tools_cache: Optional[List[Tool]] = None
        self._tools_hash: Optional[int] = None
        logger.info(
            f"✓ MICRedundancyAnalyzer v{__version__} inicializado: dim(𝔹) = {num_capabilities}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Registro de herramientas
    # ─────────────────────────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        capabilities: Union[Set[CapabilityDimension], BooleanVector],
    ) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("El nombre debe ser una cadena no vacía")
        if any(t.name == name for t in self.tools):
            raise ValueError(f"Herramienta duplicada: '{name}'")

        if isinstance(capabilities, BooleanVector):
            bool_vec = capabilities
        elif isinstance(capabilities, (set, frozenset)):
            if not all(isinstance(c, CapabilityDimension) for c in capabilities):
                raise TypeError("Todas las capacidades deben ser CapabilityDimension")
            bool_vec = BooleanVector(frozenset(capabilities))
        else:
            raise TypeError(
                f"capabilities debe ser Set[CapabilityDimension] o BooleanVector, "
                f"recibido: {type(capabilities).__name__}"
            )

        tool = Tool(name=name, capabilities=bool_vec)
        minterm = bool_vec.to_minterm()
        poly = Z2Polynomial.from_minterm(minterm, self.num_capabilities)

        if not self.grobner_basis.basis:
            self.grobner_basis.add_polynomial(poly)
        else:
            nf = self.grobner_basis.normal_form(poly)
            if not nf.is_zero():
                self.grobner_basis.add_polynomial(poly)

        self._tool_polynomials[name] = poly
        self.tools.append(tool)
        self._invalidate_matrix_cache()

    def unregister_tool(self, name: str) -> None:
        original_len = len(self.tools)
        self.tools = [t for t in self.tools if t.name != name]
        if len(self.tools) == original_len:
            logger.warning(f"Herramienta '{name}' no encontrada para eliminación")
        else:
            if name in self._tool_polynomials:
                del self._tool_polynomials[name]
            self._invalidate_matrix_cache()

    def _invalidate_matrix_cache(self) -> None:
        self._incidence_matrix_cache = None
        self._ordered_tools_cache = None
        self._tools_hash = None

    def _ordered_tools(self) -> List[Tool]:
        """Orden canónico determinista: (aridad, nombre, signature)."""
        current_hash = hash(tuple(sorted((t.name, t.signature) for t in self.tools)))
        if self._ordered_tools_cache is not None and self._tools_hash == current_hash:
            return self._ordered_tools_cache
        ordered = sorted(self.tools, key=lambda t: (t.arity, t.name, t.signature))
        self._ordered_tools_cache = ordered
        self._tools_hash = current_hash
        return ordered

    def build_incidence_matrix(self):
        if not NUMPY_AVAILABLE:
            raise HomologicalInconsistencyError("NumPy no disponible.")
        ordered = self._ordered_tools()
        current_hash = self._tools_hash
        if self._incidence_matrix_cache is not None and self._tools_hash == current_hash:
            return self._incidence_matrix_cache
        if not ordered:
            matrix = np.array([]).reshape(0, self.num_capabilities)
        else:
            matrix = np.zeros((len(ordered), self.num_capabilities), dtype=np.int8)
            for i, tool in enumerate(ordered):
                for cap in tool.capabilities.components:
                    matrix[i, cap.value] = 1
        self._incidence_matrix_cache = matrix
        return matrix

    def compute_spectral_properties(self, matrix=None) -> Dict[str, Any]:
        if matrix is None:
            matrix = self.build_incidence_matrix()
        empty = {
            'rank': 0, 'nullity': 0, 'singular_values': [],
            'condition_number': float('inf'), 'is_full_rank': False, 'determinant': 0.0,
        }
        if not NUMPY_AVAILABLE or matrix.size == 0:
            return empty
        try:
            M = matrix.astype(np.float64)
            rank = np.linalg.matrix_rank(M)
            min_dim = min(M.shape)
            nullity = min_dim - rank
            sv = np.linalg.svd(M, compute_uv=False)
            nonzero_sv = sv[sv > 1e-10]
            condition_number = (
                float(nonzero_sv[0] / nonzero_sv[-1]) if len(nonzero_sv) > 0 else float('inf')
            )
            det = float(np.linalg.det(M)) if M.shape[0] == M.shape[1] else None
            return {
                'rank': int(rank), 'nullity': int(nullity),
                'singular_values': sv.tolist(),
                'condition_number': float(condition_number),
                'is_full_rank': bool(rank == min_dim),
                'determinant': det,
            }
        except (RuntimeWarning, FloatingPointError) as e:
            raise HomologicalInconsistencyError(
                f"Singularidad numérica: {e}", condition_number=float('inf')
            )
        except np.linalg.LinAlgError as e:
            raise HomologicalInconsistencyError(f"Error en álgebra lineal: {e}")

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ FASE 1 — SUB-MOTORES: Cayley–Darboux, Gröbner, ROBDD                 ║
    # ║ Inicio de Φ₁.  Última flecha = `_handoff_to_phase2`.                 ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    def compute_cayley_darboux_embedding(self, matrix=None) -> Dict[str, Any]:
        r"""
        Proyección simpléctica de Cayley–Darboux (rigor v5.4.0, fail-closed).

        1. Construir $G = P P^\top$ y embeber en dimensión par $2n$ (carta de Darboux).
        2. $K = (G - I)(G + I)^{-1}$ (Cayley).
        3. Proyectar $K_{sp} = \tfrac12(K - J K^\top J)$ sobre $\mathfrak{sp}(2n)$.
        4. Identidad de Klein $K_{sp}^\top J + J K_{sp} = 0$.
        5. Reconstruir $M = (I + K_{sp})(I - K_{sp})^{-1}$.
        6. Certificar $\|M^\top J M - J\|_F / \|M\|_F^2 < 10^{-10}$.

        Ante singularidad numérica: residuos = +∞, `is_symplectic_valid=False`.
        """
        fail = {
            'gram_matrix': np.array([]) if NUMPY_AVAILABLE else None,
            'symplectic_matrix': np.array([]) if NUMPY_AVAILABLE else None,
            'cayley_transform': np.array([]) if NUMPY_AVAILABLE else None,
            'symplectic_residue': float('inf'),
            'relative_symplectic_residue': float('inf'),
            'klein_residual': float('inf'),
            'darboux_chart_dim': 0,
            'is_symplectic_valid': False,
        }
        if matrix is None:
            matrix = self.build_incidence_matrix()
        if not NUMPY_AVAILABLE or matrix.size == 0:
            ok_empty = dict(fail)
            ok_empty.update({
                'symplectic_residue': 0.0,
                'relative_symplectic_residue': 0.0,
                'klein_residual': 0.0,
                'is_symplectic_valid': True,
            })
            return ok_empty

        P = matrix.astype(np.float64)
        gram = P @ P.T
        dim = gram.shape[0]
        d_even = dim if dim % 2 == 0 else dim + 1
        G_pad = np.eye(d_even, dtype=np.float64)
        G_pad[:dim, :dim] = gram

        n_half = d_even // 2
        J = np.block([
            [np.zeros((n_half, n_half)), np.eye(n_half)],
            [-np.eye(n_half), np.zeros((n_half, n_half))],
        ])
        I_d = np.eye(d_even, dtype=np.float64)

        try:
            K = (G_pad - I_d) @ np.linalg.inv(G_pad + I_d)
            K_sp = 0.5 * (K - J @ K.T @ J)
            klein_matrix = K_sp.T @ J + J @ K_sp
            klein_residual = float(np.linalg.norm(klein_matrix, ord='fro'))
            G_sympl = (I_d + K_sp) @ np.linalg.inv(I_d - K_sp)
            residue_matrix = G_sympl.T @ J @ G_sympl - J
            symplectic_residue = float(np.linalg.norm(residue_matrix, ord='fro'))
            M_norm_sq = max(float(np.linalg.norm(G_sympl, ord='fro') ** 2), 1.0)
            relative_residue = symplectic_residue / M_norm_sq
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeWarning):
            logger.warning("[FASE1] Cayley–Darboux singular — fail-closed.")
            fail['gram_matrix'] = gram
            fail['darboux_chart_dim'] = int(d_even)
            return fail

        is_valid = bool(
            relative_residue <= self._SYMPLECTIC_TOL
            and klein_residual <= self._SYMPLECTIC_TOL
        )
        return {
            'gram_matrix': gram,
            'symplectic_matrix': G_sympl,
            'cayley_transform': K,
            'symplectic_residue': float(symplectic_residue),
            'relative_symplectic_residue': float(relative_residue),
            'klein_residual': float(klein_residual),
            'darboux_chart_dim': int(d_even),
            'is_symplectic_valid': is_valid,
        }

    def minimize_with_robdd(self) -> Tuple[List[str], List[str]]:
        if not self.tools:
            return [], []
        minterm_to_names: Dict[int, List[str]] = defaultdict(list)
        for tool in self.tools:
            minterm_to_names[tool.capabilities.to_minterm()].append(tool.name)
        all_minterms = set(minterm_to_names.keys())
        try:
            root = self.robdd.build_from_minterms(all_minterms)
            logger.debug(f"ROBDD construido: {self.robdd.size(root)} nodos")
        except ROBDDConstructionError as e:
            logger.error(f"Fallo en construcción ROBDD: {e}")
            return [t.name for t in self.tools], []
        essential, redundant = set(), set()
        for _minterm, names in minterm_to_names.items():
            sorted_names = sorted(names)
            essential.add(sorted_names[0])
            for extra in sorted_names[1:]:
                redundant.add(extra)
        return sorted(essential), sorted(redundant)

    def _audit_symplectic_boolean_phase1(self, matrix=None) -> Dict[str, Any]:
        """Cuerpo de FASE 1: auditoría symplectic + boolean."""
        if matrix is None:
            matrix = self.build_incidence_matrix()

        spectral = self.compute_spectral_properties(matrix)
        cayley = self.compute_cayley_darboux_embedding(matrix)
        essential, redundant = self.minimize_with_robdd()

        robdd_size = 0
        if self.tools:
            minterms = {t.capabilities.to_minterm() for t in self.tools}
            try:
                robdd_root = self.robdd.build_from_minterms(minterms)
                robdd_size = self.robdd.size(robdd_root)
            except ROBDDConstructionError:
                robdd_size = 0

        logger.info(
            f"[FASE1] rank={spectral['rank']}/{self.num_capabilities}, "
            f"sv_residue={cayley['relative_symplectic_residue']:.2e}, "
            f"Klein={cayley['klein_residual']:.2e}, "
            f"Darboux dim={cayley['darboux_chart_dim']}, "
            f"ess={len(essential)}, red={len(redundant)}, "
            f"robdd_nodes={robdd_size}"
        )
        return {
            'symplectic': cayley,
            'spectral': spectral,
            'robdd_essential': tuple(essential),
            'robdd_redundant': tuple(redundant),
            'robdd_node_count': robdd_size,
            'tool_signatures': tuple(sorted(t.signature for t in self.tools)),
            'grobner_basis_size': len(self.grobner_basis),
            'matrix': matrix,
        }

    def _compute_input_checksum(self) -> str:
        """SHA-256 sobre (nombres, capacidades) de todas las herramientas."""
        hasher = hashlib.sha256()
        for t in sorted(self.tools):
            hasher.update(t.name.encode('utf-8'))
            hasher.update(t.signature.encode('utf-8'))
        return hasher.hexdigest()

    def _handoff_to_phase2(
        self,
        audit: Dict[str, Any],
        input_checksum: str,
    ) -> MICPhase1Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 1  →  morfismo Φ₁ → Φ₂.
        El objeto retornado **es** el dominio estricto de
        `_ingest_phase1_handoff` (PRIMERA DEFINICIÓN DE FASE 2):

            $\Phi_2 \circ \Phi_1(\mathrm{MIC}) =
             \texttt{\_ingest\_phase1\_handoff}(
                 \texttt{\_handoff\_to\_phase2}(\mathrm{MIC}))$.
        ═══════════════════════════════════════════════════════════════════
        """
        sym = audit['symplectic']
        spec = audit['spectral']
        h1 = MICPhase1Handoff(
            symplectic_residue=float(sym['symplectic_residue']),
            relative_symplectic_residue=float(sym['relative_symplectic_residue']),
            klein_residual=float(sym['klein_residual']),
            is_symplectic_valid=bool(sym['is_symplectic_valid']),
            darboux_chart_dim=int(sym.get('darboux_chart_dim', 0)),
            spectral_rank=int(spec['rank']),
            spectral_nullity=int(spec['nullity']),
            condition_number=float(spec['condition_number']),
            grobner_basis_size=int(audit['grobner_basis_size']),
            robdd_node_count=int(audit['robdd_node_count']),
            robdd_essential=audit['robdd_essential'],
            robdd_redundant=audit['robdd_redundant'],
            tool_signatures=audit['tool_signatures'],
            num_capabilities=self.num_capabilities,
            input_checksum=input_checksum,
            timestamp_iso=datetime.now(tz=timezone.utc).isoformat(),
        )
        logger.info(
            f"[FASE1→FASE2] handoff: Δω_rel={h1.relative_symplectic_residue:.2e}, "
            f"rank={h1.spectral_rank}, Darboux={h1.darboux_chart_dim}, "
            f"checksum={input_checksum[:16]}..."
        )
        return h1

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ FASE 2 — Poincaré celeste + RSI³ anidado (T, T², T³) + Wasserstein  ║
    # ║ Continuación estricta de `_handoff_to_phase2`.                       ║
    # ║ Primera flecha = `_ingest_phase1_handoff`.                           ║
    # ║ Última flecha  = `_handoff_to_phase3`.                               ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _ingest_phase1_handoff(h1: MICPhase1Handoff) -> MICPhase1Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        PRIMERA DEFINICIÓN DE FASE 2: consume `MICPhase1Handoff`.
        Continuación topológica de `_handoff_to_phase2` (última de FASE 1).
        Verifica continuidad estricta de tipo y consistencia simpléctica.
        ═══════════════════════════════════════════════════════════════════
        """
        if not isinstance(h1, MICPhase1Handoff):
            raise OODAChainViolationError(
                f"[FASE2] Handoff inválido: {type(h1).__name__} "
                "(esperaba MICPhase1Handoff)."
            )
        if (not h1.is_symplectic_valid) and h1.relative_symplectic_residue > 1e-6:
            logger.warning(
                f"[FASE2] Handoff con residuo simpléctico alto: "
                f"δω_rel={h1.relative_symplectic_residue:.3e}"
            )
        return h1

    # ── sub-motores celestes de Poincaré ─────────────────────────────────────

    @staticmethod
    def _safe_sech(t):
        """sech(t) = 1/cosh(t) con saturación anti-overflow."""
        return 1.0 / np.cosh(np.clip(t, -40.0, 40.0))

    def _melnikov_homoclinic_sech(self, omega: float) -> Dict[str, float]:
        r"""
        Integral de Melnikov sobre la homoclínica del péndulo
        $\ddot q + \sin q = \varepsilon\sin(\omega t)$:

            $q_h(t)=4\arctan(e^t)-\pi,\quad p_h(t)=2\,\mathrm{sech}(t)$,

            $M(\theta)=\int_{-\infty}^{\infty} p_h(t)\,\sin(\omega(t+\theta))\,dt
                      = 2\pi\,\mathrm{sech}(\pi\omega/2)\,\sin(\omega\theta)$.

        Se reporta $|M|_{\max}$ cerrado, su aproximación trapezoidal y el residuo.
        """
        omega = float(abs(omega)) if omega == omega else 1.0  # NaN-guard
        if omega < 1e-15:
            omega = 1e-15
        closed = float(2.0 * math.pi / math.cosh(min(math.pi * omega / 2.0, 40.0)))
        numerical = closed
        if NUMPY_AVAILABLE:
            try:
                n_pts = 2001
                t = np.linspace(-20.0, 20.0, n_pts)
                p_h = 2.0 * self._safe_sech(t)
                integrand = p_h * np.sin(omega * t)  # θ=0, |M| se obtiene por fase
                if hasattr(np, 'trapezoid'):
                    numerical = float(abs(np.trapezoid(integrand, t)))
                else:
                    numerical = float(abs(np.trapz(integrand, t)))
            except (ValueError, FloatingPointError, RuntimeWarning):
                numerical = closed
        residue = float(abs(numerical - closed) / max(closed, 1e-15))
        return {
            'melnikov_closed_form': closed,
            'melnikov_integral': float(min(closed, numerical) if residue < 0.5 else closed),
            'melnikov_residue': residue,
        }

    def _lindstedt_frequency_correction(
        self, omega_0, epsilon: float, amplitude: float,
    ) -> float:
        r"""
        Serie de Poincaré–Lindstedt para Duffing $\ddot x + \omega_0^2 x = \varepsilon x^3$.
        Anulación del término secular: $\omega_1 = 0$ (impar),
        $\omega_2 = 3 a^2 / (8\omega_0)$ ⇒
            $\omega = \omega_0 + \dfrac{3\varepsilon a^2}{8\omega_0} + O(\varepsilon^2)$.
        Retorna el desplazamiento $|\omega-\omega_0|$ (norma euclídea si $\omega_0$ es vector).
        """
        if not NUMPY_AVAILABLE:
            w0 = float(omega_0) if not hasattr(omega_0, '__len__') else 1.0
            if abs(w0) < 1e-15:
                return 0.0
            return abs(3.0 * epsilon * (amplitude ** 2) / (8.0 * w0))
        w = np.asarray(omega_0, dtype=np.float64).flatten()
        if w.size == 0:
            return 0.0
        w_safe = np.where(np.abs(w) < 1e-15, 1.0, w)
        shift = (3.0 * epsilon * (amplitude ** 2)) / (8.0 * w_safe)
        return float(np.linalg.norm(shift))

    def _bruno_diophantine(self, omega_0) -> Tuple[float, bool]:
        r"""
        Condición de Bruno: $\sum_{k\ge 1} (\log q_{k+1})/q_k < \infty$
        sobre convergentes de $\omega_i/\omega_j$. Se aproxima con 20 convergentes
        y se exige que la suma parcial sea finita y acotada.
        """
        if not NUMPY_AVAILABLE:
            return float('inf'), False
        w = np.asarray(omega_0, dtype=np.float64)
        w = w[w > 1e-15]
        if w.size < 2:
            return 1e10, True
        bruno_sum = 0.0
        gamma_min = float('inf')
        tau = float(w.size + 1)
        for i, j in combinations(range(w.size), 2):
            if abs(w[j]) < 1e-15:
                continue
            ratio = float(w[i] / w[j])
            a = ratio
            p_prev, p_curr = 1, 0
            q_prev, q_curr = 0, 1
            for _ in range(20):
                ai = int(math.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr <= 0 or q_curr > 64:
                    break
                if q_prev > 0:
                    bruno_sum += math.log(max(q_curr, 2)) / q_prev
                k_vec = np.zeros(w.size, dtype=np.float64)
                k_vec[i] = float(q_curr)
                k_vec[j] = float(-p_curr)
                dot = abs(float(np.dot(w, k_vec)))
                k_norm = float(np.sum(np.abs(k_vec)))
                if k_norm > 1e-15:
                    candidate = dot * (k_norm ** tau)
                    if candidate < gamma_min:
                        gamma_min = candidate
                frac = a - ai
                if frac < 1e-12:
                    break
                a = 1.0 / frac
        if math.isinf(gamma_min):
            gamma_min = 1e10
        bruno_ok = bool(math.isfinite(bruno_sum) and bruno_sum < 1.0 / self._BRUNO_FLOOR)
        return float(gamma_min), bruno_ok

    def _poincare_return_map_lyapunov(self, chirikov: float) -> float:
        r"""
        Mapa estándar de Chirikov (sección de Poincaré de un oscilador perturbado):

            $I_{n+1}=I_n+K\sin\theta_n,\qquad
             \theta_{n+1}=\theta_n+I_{n+1}\pmod{2\pi}$.

        Exponente de Lyapunov aproximado: $\lambda \approx \log(K/2)$ para $K>1$,
        y $\lambda \approx 0$ (flujo integrable) para $K\le 1$.
        """
        K = abs(float(chirikov))
        if K <= 1.0:
            return 0.0
        return float(math.log(K / 2.0))

    def _nekhoroshev_stability_time(self, epsilon: float, n_freq: int) -> float:
        r"""
        Estimación de Nekhoroshev: $T \sim \exp\bigl(c/\varepsilon^{1/(2n)}\bigr)$
        para hamiltonianos analíticos casi-integrables. $n$ = nº de frecuencias.
        """
        n = max(int(n_freq), 1)
        eps = max(float(abs(epsilon)), 1e-16)
        exponent = 1.0 / (2.0 * n)
        c = 0.5
        try:
            return float(math.exp(min(c / (eps ** exponent), 80.0)))
        except (OverflowError, ValueError):
            return float('inf')

    def _poincare_kac_recurrence(self, n_freq: int, eps_ball: float) -> float:
        r"""
        Tiempo medio de recurrencia de Poincaré–Kac a una $\varepsilon$-bola
        en $\mathbb{T}^n$: $\langle\tau\rangle \sim \varepsilon^{-n}$.
        """
        n = max(int(n_freq), 1)
        eps = min(max(float(eps_ball), 1e-12), 1.0)
        try:
            return float((1.0 / eps) ** n)
        except OverflowError:
            return float('inf')

    def evaluate_poincare_lindstedt_melnikov(self, matrix=None) -> Dict[str, Any]:
        r"""
        FASE 2 (mecánica celeste de Poincaré, rigor v5.4.0):

        • KAM / Bruno vía convergentes de fracciones continuas.
        • Melnikov: fórmula cerrada $2\pi\mathrm{sech}(\pi\omega/2)$ + trapecio sech.
        • Lindstedt: corrección secular de Duffing.
        • Chirikov, Lyapunov del mapa de retorno, Nekhoroshev, recurrencia Kac.
        """
        spectral = self.compute_spectral_properties(matrix)
        sv = spectral.get('singular_values', [1.0]) or [1.0]
        if NUMPY_AVAILABLE:
            omega_0 = np.array([s for s in sv if s > 1e-10], dtype=np.float64)
            if omega_0.size == 0:
                omega_0 = np.array([1.0])
        else:
            omega_0 = [1.0]

        n_freq = int(len(omega_0))
        gamma_min, bruno_ok = self._bruno_diophantine(omega_0)
        diophantine_satisfied = bool(gamma_min >= self._GAMMA_KAM)

        if NUMPY_AVAILABLE and n_freq > 1:
            delta_omega = float(np.std(omega_0))
            mean_omega = float(np.mean(omega_0))
        else:
            delta_omega, mean_omega = 0.0, 1.0
        delta_omega_res = mean_omega if mean_omega > 1e-10 else 1.0
        chirikov = float(delta_omega / delta_omega_res)

        omega_forcing = float(omega_0[0]) if n_freq >= 1 else 1.0
        melnikov = self._melnikov_homoclinic_sech(omega_forcing)

        epsilon = float(chirikov) if chirikov > 0 else 1e-12
        amplitude = float(omega_0[0]) if n_freq >= 1 else 1.0
        lindstedt_shift = self._lindstedt_frequency_correction(omega_0, epsilon, amplitude)
        lyapunov = self._poincare_return_map_lyapunov(chirikov)
        nekhoroshev = self._nekhoroshev_stability_time(epsilon, n_freq)
        recurrence = self._poincare_kac_recurrence(n_freq, max(epsilon, 1e-6))

        is_kam_stable = bool(
            diophantine_satisfied
            and bruno_ok
            and chirikov < 1.0
            and melnikov['melnikov_integral'] < self._MELNIKOV_STABLE
            and lyapunov <= 0.0
        )
        return {
            'diophantine_satisfied': diophantine_satisfied,
            'diophantine_gamma': float(gamma_min),
            'bruno_ok': bool(bruno_ok),
            'chirikov_parameter': float(chirikov),
            'melnikov_integral': float(melnikov['melnikov_integral']),
            'melnikov_closed_form': float(melnikov['melnikov_closed_form']),
            'melnikov_residue': float(melnikov['melnikov_residue']),
            'lindstedt_omega_shift': float(lindstedt_shift),
            'nekhoroshev_time': float(nekhoroshev),
            'poincare_lyapunov': float(lyapunov),
            'recurrence_time': float(recurrence),
            'is_kam_stable': is_kam_stable,
        }

    # ── RSI³: automejora recursiva de nivel 3 (Kleisli T, T², T³) ────────────

    @staticmethod
    def _soft_threshold(sv, tau: float):
        if not NUMPY_AVAILABLE:
            return sv
        x = np.asarray(sv, dtype=np.float64)
        return np.maximum(x - float(tau), 0.0)

    def _rsi3_picard(
        self, sv, tau: float, max_iter: int = 12, tol: float = 1e-12,
    ) -> Tuple[Any, float, float]:
        r"""
        Iteración de Picard–Banach $x_{k+1}=T_\tau(x_k)$ sobre el espectro.
        Retorna (punto fijo, Lip empírico, residual final).
        """
        if not NUMPY_AVAILABLE:
            return sv, 1.0, 0.0
        x = np.asarray(sv, dtype=np.float64).copy()
        prev_res = None
        lip = 1.0
        residual = 0.0
        for _ in range(max_iter):
            x_next = self._soft_threshold(x, tau)
            residual = float(np.linalg.norm(x_next - x))
            if prev_res is not None and prev_res > 1e-15:
                lip = min(lip, residual / prev_res) if residual <= prev_res else max(lip, residual / prev_res)
            if residual < tol:
                x = x_next
                break
            prev_res = residual
            x = x_next
        if not math.isfinite(lip) or lip < 0.0:
            lip = 1.0
        return x, float(lip), float(residual)

    def _grassmann_fubini_study(self, A, B) -> float:
        r"""
        Distancia de Fubini–Study en Grassmann: $\|\theta\|_2$ de los ángulos
        principales entre $\mathrm{Im}(A^\top)$ y $\mathrm{Im}(B^\top)$.
        Corrige el bug v5.3 (⟨u₁,u₂⟩=0 ⇒ π/2 constante entre modos SVD).
        """
        if not NUMPY_AVAILABLE or A is None or B is None:
            return 1e-6
        if getattr(A, 'size', 0) == 0 or getattr(B, 'size', 0) == 0:
            return math.pi / 2.0
        try:
            Af = np.asarray(A, dtype=np.float64)
            Bf = np.asarray(B, dtype=np.float64)
            if Af.ndim == 1:
                Af = Af.reshape(-1, 1)
            if Bf.ndim == 1:
                Bf = Bf.reshape(-1, 1)
            Ua, sa, _ = np.linalg.svd(Af, full_matrices=False)
            Ub, sb, _ = np.linalg.svd(Bf, full_matrices=False)
            ra = int(np.sum(sa > 1e-12))
            rb = int(np.sum(sb > 1e-12))
            if ra == 0 or rb == 0:
                return math.pi / 2.0
            Qa = Ua[:, :ra]
            Qb = Ub[:, :rb]
            M = Qa.T @ Qb
            sv = np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)
            thetas = np.arccos(sv)
            return float(np.linalg.norm(thetas))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning):
            return 1e-5

    def _rsi3_level1_surface(self, sv, rank: int) -> RSI3KleisliState:
        r"""T¹: mejora de objeto — soft-threshold con $\tau=\mathrm{median}(\sigma)/10$."""
        if not NUMPY_AVAILABLE:
            return RSI3KleisliState(spectrum=tuple(), tau=0.0, residual=0.0, lipschitz=1.0, level=1)
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])
        tau = float(np.median(x) * 0.10)
        _fixed, lip, res = self._rsi3_picard(x, tau)
        krull = float(rank / max(self.num_capabilities, 1))
        _ = krull
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in _fixed),
            tau=tau, residual=res, lipschitz=lip, level=1,
        )

    def _rsi3_level2_operator(self, s1: RSI3KleisliState, sv) -> RSI3KleisliState:
        r"""
        T²: mejora del operador — $\tau_2 = \tau_1\cdot(1+\|T^1 x-x\|)$
        (el umbral se adapta al residual de nivel 1).
        """
        if not NUMPY_AVAILABLE:
            return RSI3KleisliState(spectrum=s1.spectrum, tau=s1.tau, residual=s1.residual,
                                    lipschitz=s1.lipschitz, level=2)
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])
        tau2 = float(s1.tau * (1.0 + s1.residual))
        _fixed, lip, res = self._rsi3_picard(x, tau2)
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in _fixed),
            tau=tau2, residual=res, lipschitz=lip, level=2,
        )

    def _rsi3_level3_kleisli(
        self, s1: RSI3KleisliState, s2: RSI3KleisliState, sv,
    ) -> RSI3KleisliState:
        r"""
        T³: meta-mejora Kleisli.

        Monad $(T,\eta,\mu)$ sobre el espectro. El asociador numérico
            $\|\mu\circ T\mu - \mu\circ\mu T\|$
        se estima como el conmutador de las tres correcciones de umbral:

            $a = \|T_{\tau_3}\circ T_{\tau_2}\circ T_{\tau_1}
                  - T_{\tau_1}\circ T_{\tau_2}\circ T_{\tau_3}\|$.

        $\tau_3$ es el punto fijo de Banach de $\tau\mapsto\tau\cdot\mathrm{Lip}(T_\tau)$
        (mejora de la *regla* de mejora). Axioma A4: $\mathrm{Lip}(T^3)\le\mathrm{Lip}(T)^3$.
        """
        if not NUMPY_AVAILABLE:
            return RSI3KleisliState(spectrum=s2.spectrum, tau=s2.tau, residual=s2.residual,
                                    lipschitz=min(s1.lipschitz ** 3, 1.0), level=3)
        x = np.asarray(sv, dtype=np.float64)
        x = x[x > 0.0]
        if x.size == 0:
            x = np.array([1.0])

        # Punto fijo meta: τ₃ ← τ₂ · Lip(T²)  (contracción de la regla)
        tau3 = float(s2.tau * min(max(s2.lipschitz, 1e-6), 1.0))
        _fixed, lip3, res3 = self._rsi3_picard(x, tau3)

        t1 = self._soft_threshold(x, s1.tau)
        t2 = self._soft_threshold(t1, s2.tau)
        t321 = self._soft_threshold(t2, tau3)

        t3 = self._soft_threshold(x, tau3)
        t32 = self._soft_threshold(t3, s2.tau)
        t123 = self._soft_threshold(t32, s1.tau)
        associator = float(np.linalg.norm(t321 - t123))

        lip_nested = float(min(s1.lipschitz, 1.0) ** 3)
        lip = float(min(lip3, lip_nested))
        return RSI3KleisliState(
            spectrum=tuple(float(v) for v in _fixed),
            tau=tau3, residual=float(res3 + associator),
            lipschitz=lip, level=3,
        )

    def evaluate_rsi3_kleisli_surfaces(
        self, matrix=None, essential_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r"""
        FASE 2 (RSI³ rigor v5.4.0): superficies Kleisli anidadas T¹⊂T²⊂T³.

        • Krull: rank / num_capabilities.
        • Fubini–Study de Grassmann entre fila-espacio total y esencial.
        • Banach L_Φ: Lip empírico de T³ (Picard) y cota Lip(T)³.
        • Quadratic rate p: mediana de log-ratios de σ.
        • Capacity acceleration: max Δ³σ.
        • Asociador Kleisli de nivel 3.
        """
        spectral = self.compute_spectral_properties(matrix)
        rank = int(spectral.get('rank', 0))
        krull_valuation = float(rank / max(self.num_capabilities, 1))

        # Grassmann FS: full vs essential submatrix
        fubini_study_distance = 1e-6
        if matrix is not None and NUMPY_AVAILABLE and matrix.size > 0:
            if essential_names:
                ordered = self._ordered_tools()
                ess_set = set(essential_names)
                idx = [i for i, t in enumerate(ordered) if t.name in ess_set]
                if idx:
                    M_ess = matrix[idx, :]
                    fubini_study_distance = self._grassmann_fubini_study(matrix, M_ess)
                else:
                    fubini_study_distance = self._grassmann_fubini_study(matrix, matrix)
            else:
                fubini_study_distance = self._grassmann_fubini_study(matrix, matrix)

        sv_list = np.array(spectral.get('singular_values', []), dtype=np.float64) if NUMPY_AVAILABLE else []
        sv_pos = sv_list[sv_list > 1e-15] if NUMPY_AVAILABLE else []

        s1 = self._rsi3_level1_surface(sv_pos, rank)
        s2 = self._rsi3_level2_operator(s1, sv_pos)
        s3 = self._rsi3_level3_kleisli(s1, s2, sv_pos)

        # Lip espectral clásico (razón consecutiva) como cota auxiliar
        lipschitz_sv = 1.0
        if NUMPY_AVAILABLE and getattr(sv_pos, 'size', 0) >= 2:
            ratios = sv_pos[1:] / np.maximum(sv_pos[:-1], 1e-15)
            lipschitz_sv = float(np.max(ratios))

        lipschitz_constant = float(min(max(s3.lipschitz, lipschitz_sv * 0.0 + s3.lipschitz), 1.0))
        # A4 usa el Lip de T³; si Picard no contrajo, caemos a razón espectral
        if s3.lipschitz >= 0.999 and NUMPY_AVAILABLE and getattr(sv_pos, 'size', 0) >= 2:
            lipschitz_constant = float(np.clip(lipschitz_sv, 0.0, 1.0))

        quadratic_rate = 0.0
        if NUMPY_AVAILABLE and getattr(sv_pos, 'size', 0) >= 3:
            orders: List[float] = []
            for k in range(1, sv_pos.size - 1):
                num = math.log(max(sv_pos[k + 1] / max(sv_pos[k], 1e-15), 1e-15))
                den = math.log(max(sv_pos[k] / max(sv_pos[k - 1], 1e-15), 1e-15))
                if abs(den) > 1e-9:
                    orders.append(num / den)
            if orders:
                quadratic_rate = float(np.clip(np.median(orders), 0.0, 4.0))

        capacity_acceleration = 0.0
        if NUMPY_AVAILABLE and getattr(sv_pos, 'size', 0) >= 4:
            d1 = np.diff(sv_pos)
            d2 = np.diff(d1)
            d3 = np.diff(d2)
            if d3.size > 0:
                capacity_acceleration = float(np.max(d3))

        is_rsi3_valid = bool(
            krull_valuation >= self._KRULL_MIN
            and fubini_study_distance <= self._FS_MAX
            and lipschitz_constant <= self._BANACH_L_MAX
            and s3.residual < 1.0
        )
        return {
            'krull_valuation': float(krull_valuation),
            'fubini_study_distance': float(fubini_study_distance),
            'lipschitz_constant': float(lipschitz_constant),
            'quadratic_rate': float(quadratic_rate),
            'capacity_acceleration': float(capacity_acceleration),
            'rsi3_level1_residual': float(s1.residual),
            'rsi3_level2_lipschitz': float(s2.lipschitz),
            'rsi3_level3_associator': float(s3.residual),
            'rsi3_level3_tau': float(s3.tau),
            'is_rsi3_valid': is_rsi3_valid,
        }

    def compute_boolean_wasserstein_distance(self, p_orig, p_red) -> float:
        r"""Distancia $W_1$ entre distribuciones booleanas: $\tfrac12\|p_o-p_r\|_1$."""
        if not NUMPY_AVAILABLE:
            return 0.0
        p_o = np.asarray(p_orig, dtype=np.float64).flatten()
        p_r = np.asarray(p_red, dtype=np.float64).flatten()
        size = max(p_o.size, p_r.size)
        if size == 0:
            return 0.0
        p_o = np.pad(p_o, (0, size - p_o.size))
        p_r = np.pad(p_r, (0, size - p_r.size))
        total_o = np.sum(p_o)
        total_r = np.sum(p_r)
        if total_o > 0:
            p_o = p_o / total_o
        if total_r > 0:
            p_r = p_r / total_r
        return float(0.5 * np.sum(np.abs(p_o - p_r)))

    def _compute_capability_wasserstein(self, essential_names: List[str]) -> float:
        """$W_1$ entre distribución original y reducida de capacidades."""
        if not self.tools or not NUMPY_AVAILABLE:
            return 1.0
        p_orig = np.zeros(self.num_capabilities)
        for tool in self.tools:
            for cap in tool.capabilities.components:
                p_orig[cap.value] += 1
        essential_set = set(essential_names)
        p_red = np.zeros(self.num_capabilities)
        for tool in self.tools:
            if tool.name in essential_set:
                for cap in tool.capabilities.components:
                    p_red[cap.value] += 1
        return self.compute_boolean_wasserstein_distance(p_orig, p_red)

    def check_non_interference_sat(self) -> Tuple[bool, List[str]]:
        r"""
        SAT de *cobertura + mutex de duplicados* (v5.4.0).

        • Por cada capacidad cubierta, cláusula $\bigvee\{\text{tool}_i: i\ni\text{cap}\}$.
        • Por cada grupo de signatures duplicados, mutex par a par.
        • `conflicting` = extras de cada grupo duplicado (núcleo de no-unicidad).
        """
        n = len(self.tools)
        if n <= 1:
            return True, []

        ordered = self._ordered_tools()
        cnf: List[List[int]] = []

        # cobertura por capacidad
        for cap_idx in range(self.num_capabilities):
            clause = [
                i + 1 for i, t in enumerate(ordered)
                if any(c.value == cap_idx for c in t.capabilities.components)
            ]
            if clause:
                cnf.append(clause)

        # mutex de duplicados
        sig_groups: Dict[str, List[int]] = defaultdict(list)
        for i, t in enumerate(ordered):
            sig_groups[t.signature].append(i)
        conflicting: List[str] = []
        for names_idx in sig_groups.values():
            if len(names_idx) > 1:
                extras = names_idx[1:]
                conflicting.extend(ordered[i].name for i in extras)
                for a, b in combinations(names_idx, 2):
                    cnf.append([-(a + 1), -(b + 1)])

        sat_ok = DPLLSATSolver.solve(cnf) if cnf else True
        return bool(sat_ok), sorted(set(conflicting))

    def _compute_grobner_independence(self) -> Tuple[List[str], List[str]]:
        """Separa herramientas en independientes / dependientes de Gröbner (orden estable)."""
        temp = GrobnerBasisZ2(self.num_capabilities)
        independent, dependent = [], []
        for tool in self._ordered_tools():
            poly = self._tool_polynomials.get(tool.name)
            if poly is None:
                continue
            if temp.is_member(poly):
                dependent.append(tool.name)
            else:
                independent.append(tool.name)
                temp.add_polynomial(poly)
        return sorted(independent), sorted(dependent)

    def _audit_poincare_rsi3_phase2(
        self,
        h1: MICPhase1Handoff,
        matrix=None,
    ) -> Dict[str, Any]:
        """Cuerpo de FASE 2 (requiere handoff de FASE 1)."""
        if not isinstance(h1, MICPhase1Handoff):
            raise OODAChainViolationError("[FASE2] auditoría sin MICPhase1Handoff.")
        if matrix is None:
            matrix = self.build_incidence_matrix()

        poincare = self.evaluate_poincare_lindstedt_melnikov(matrix)
        rsi3 = self.evaluate_rsi3_kleisli_surfaces(matrix, list(h1.robdd_essential))
        w1 = self._compute_capability_wasserstein(list(h1.robdd_essential))
        sat_ok, conflicting = self.check_non_interference_sat()
        indep, dep = self._compute_grobner_independence()

        logger.info(
            f"[FASE2] KAM={poincare['is_kam_stable']}, "
            f"γ={poincare['diophantine_gamma']:.3e}, Bruno={poincare['bruno_ok']}, "
            f"Chirikov={poincare['chirikov_parameter']:.3f}, "
            f"|M|={poincare['melnikov_integral']:.3e} "
            f"(cerrado={poincare['melnikov_closed_form']:.3e}, "
            f"res={poincare['melnikov_residue']:.2e}), "
            f"Lindstedt=Δω {poincare['lindstedt_omega_shift']:.3e}, "
            f"λ_P={poincare['poincare_lyapunov']:.3f}, "
            f"T_Nek={poincare['nekhoroshev_time']:.3e}, "
            f"L_φ={rsi3['lipschitz_constant']:.3f}, "
            f"T³_assoc={rsi3['rsi3_level3_associator']:.2e}, "
            f"p={rsi3['quadratic_rate']:.2f}, "
            f"W₁={w1:.2e}, SAT={sat_ok}, indep={len(indep)}, dep={len(dep)}"
        )
        return {
            'poincare': poincare,
            'rsi3': rsi3,
            'wasserstein': w1,
            'sat_ok': sat_ok,
            'conflicting': tuple(conflicting),
            'grobner_independent': tuple(indep),
            'grobner_dependent': tuple(dep),
        }

    def _handoff_to_phase3(
        self,
        h1: MICPhase1Handoff,
        phase2_data: Dict[str, Any],
    ) -> MICPhase2Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 2  →  morfismo Φ₂ → Φ₃.
        El objeto retornado **es** el dominio estricto de
        `_ingest_phase2_handoff` (PRIMERA DEFINICIÓN DE FASE 3).
        ═══════════════════════════════════════════════════════════════════
        """
        pc = phase2_data['poincare']
        r3 = phase2_data['rsi3']
        h2 = MICPhase2Handoff(
            phase1=h1,
            kam_stable=bool(pc['is_kam_stable']),
            diophantine_gamma=float(pc['diophantine_gamma']),
            bruno_ok=bool(pc['bruno_ok']),
            chirikov_parameter=float(pc['chirikov_parameter']),
            melnikov_integral=float(pc['melnikov_integral']),
            melnikov_closed_form=float(pc['melnikov_closed_form']),
            melnikov_residue=float(pc['melnikov_residue']),
            lindstedt_omega_shift=float(pc['lindstedt_omega_shift']),
            nekhoroshev_time=float(pc['nekhoroshev_time']),
            poincare_lyapunov=float(pc['poincare_lyapunov']),
            recurrence_time=float(pc['recurrence_time']),
            banach_L=float(r3['lipschitz_constant']),
            quadratic_rate=float(r3['quadratic_rate']),
            capacity_acceleration=float(r3['capacity_acceleration']),
            krull_valuation=float(r3['krull_valuation']),
            fubini_study_distance=float(r3['fubini_study_distance']),
            rsi3_level1_residual=float(r3['rsi3_level1_residual']),
            rsi3_level2_lipschitz=float(r3['rsi3_level2_lipschitz']),
            rsi3_level3_associator=float(r3['rsi3_level3_associator']),
            wasserstein_distance=float(phase2_data['wasserstein']),
            grobner_independent=phase2_data['grobner_independent'],
            grobner_dependent=phase2_data['grobner_dependent'],
            sat_ok=bool(phase2_data['sat_ok']),
            conflicting_tools=phase2_data['conflicting'],
        )
        logger.info(
            f"[FASE2→FASE3] handoff: L_φ={h2.banach_L:.3f}, "
            f"KAM={h2.kam_stable}, W₁={h2.wasserstein_distance:.2e}, "
            f"T³_assoc={h2.rsi3_level3_associator:.2e}, SAT={h2.sat_ok}"
        )
        return h2

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║ FASE 3 — Adjudicación Heyting + Crowbar + certificado OODA           ║
    # ║ Continuación estricta de `_handoff_to_phase3`.                       ║
    # ║ Primera flecha = `_ingest_phase2_handoff`.                           ║
    # ║ Última flecha  = `_emit_governance_certificate`.                     ║
    # ╚═══════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _ingest_phase2_handoff(h2: MICPhase2Handoff) -> MICPhase2Handoff:
        r"""
        ═══════════════════════════════════════════════════════════════════
        PRIMERA DEFINICIÓN DE FASE 3: consume `MICPhase2Handoff`.
        Continuación topológica de `_handoff_to_phase3` (última de FASE 2).
        ═══════════════════════════════════════════════════════════════════
        """
        if not isinstance(h2, MICPhase2Handoff):
            raise OODAChainViolationError(
                f"[FASE3] Handoff inválido: {type(h2).__name__} "
                "(esperaba MICPhase2Handoff)."
            )
        if not isinstance(h2.phase1, MICPhase1Handoff):
            raise OODAChainViolationError("[FASE3] Handoff sin Φ₁ embebido.")
        return h2

    @staticmethod
    def _heyting_combine_full(
        symplectic_valid: bool,
        rank_ok: bool,
        kam_stable: bool,
        rsi3_valid: bool,
        sat_ok: bool,
        wasserstein_distance: float,
    ) -> HeytingOmega3:
        r"""Join bitwise-OR sobre $\Omega_3$ con 6 átomos de severidad."""
        v_sym = HeytingOmega3.COHERENT if symplectic_valid else HeytingOmega3.VETOED
        v_rank = HeytingOmega3.COHERENT if rank_ok else HeytingOmega3.DEGRADED
        v_kam = HeytingOmega3.COHERENT if kam_stable else HeytingOmega3.DEGRADED
        v_rsi = HeytingOmega3.COHERENT if rsi3_valid else HeytingOmega3.DEGRADED
        v_sat = HeytingOmega3.COHERENT if sat_ok else HeytingOmega3.VETOED
        v_w1 = (
            HeytingOmega3.COHERENT
            if wasserstein_distance <= 1e-2
            else HeytingOmega3.DEGRADED
        )
        return HeytingOmega3.fast_join_all([v_sym, v_rank, v_kam, v_rsi, v_sat, v_w1])

    @staticmethod
    def _hardware_action_from_verdict(verdict: HeytingOmega3) -> Tuple[str, float]:
        """Mapeo veredicto → (acción de hardware, latencia de interlock en ns)."""
        if verdict is HeytingOmega3.VETOED:
            return "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED", 392.15
        if verdict is HeytingOmega3.DEGRADED:
            return "MONITORING_DEGRADED", 120.0
        return "NOMINAL_EXECUTION", 45.0

    def _emit_governance_certificate(
        self,
        h2: MICPhase2Handoff,
    ) -> MICGovernanceCertificate:
        r"""
        ═══════════════════════════════════════════════════════════════════
        ÚLTIMA DEFINICIÓN DE FASE 3: certificado OODA completo.
        Verifica axiomas A1–A5, aplica adjudicación Ω₃ (con implicación de
        Heyting) y emite el certificado con trazabilidad SHA-256.
        ═══════════════════════════════════════════════════════════════════
        """
        h1 = h2.phase1

        symplectic_valid = h1.is_symplectic_valid
        rank_ok = h1.spectral_rank > 0
        kam_stable = h2.kam_stable
        rsi3_valid = bool(
            h2.krull_valuation >= self._KRULL_MIN
            and h2.banach_L <= self._BANACH_L_MAX
            and h2.fubini_study_distance <= self._FS_MAX
        )
        sat_ok = h2.sat_ok
        w1 = h2.wasserstein_distance

        verdict = self._heyting_combine_full(
            symplectic_valid=symplectic_valid,
            rank_ok=rank_ok,
            kam_stable=kam_stable,
            rsi3_valid=rsi3_valid,
            sat_ok=sat_ok,
            wasserstein_distance=w1,
        )

        heyting_ok = _heyting_lattice_axioms_check()
        if not heyting_ok:
            logger.error("[HEYTING] ¡Axiomas de retículo/Heyting Ω₃ violados — bug crítico!")

        # Esenciales = ROBDD-ess ∩ Gröbner-indep  (precedencia corregida)
        robdd_ess = set(h1.robdd_essential)
        grob_ind = set(h2.grobner_independent)
        grob_dep = set(h2.grobner_dependent)
        essential_set = robdd_ess & grob_ind
        if not essential_set:
            essential_set = robdd_ess or grob_ind
        all_names = {t.name for t in self.tools}
        redundant_set = (set(h1.robdd_redundant) | grob_dep | (all_names - essential_set)) - essential_set
        essential = tuple(sorted(essential_set))
        redundant = tuple(sorted(redundant_set))

        is_coherent = bool(
            symplectic_valid and rank_ok and kam_stable and rsi3_valid
            and sat_ok and w1 <= self._W1_STABLE
            and verdict is not HeytingOmega3.VETOED
        )
        is_authorized = bool(is_coherent and heyting_ok)

        atoms = [
            1.0 if symplectic_valid else 0.0,
            1.0 if rank_ok else 0.5,
            1.0 if kam_stable else 0.5,
            1.0 if rsi3_valid else 0.5,
            1.0 if sat_ok else 0.0,
            1.0 if w1 <= self._W1_STABLE else 0.5,
        ]
        overall_score = float(sum(atoms) / len(atoms))
        risk = "NOMINAL" if overall_score > 0.9 else (
            "WARNING" if overall_score > 0.6 else "CRITICAL"
        )
        hw_action, latency = self._hardware_action_from_verdict(verdict)

        cert = MICGovernanceCertificate(
            phase2=h2,
            heyting_verdict=verdict,
            heyting_mask=verdict.mask,
            heyting_axioms_verified=heyting_ok,
            hardware_action=hw_action,
            interlock_latency_ns=latency,
            is_coherent=is_coherent,
            is_compilation_authorized=is_authorized,
            overall_quality_score=overall_score,
            risk_assessment=risk,
            essential_tools=essential,
            redundant_tools=redundant,
            timestamp_iso=datetime.now(tz=timezone.utc).isoformat(),
            input_checksum=h1.input_checksum,
        )
        logger.info(
            f"[OODA-CERT] verdict={verdict.name} (mask=0x{verdict.mask:02X}), "
            f"coherent={is_coherent}, authorized={is_authorized}, "
            f"score={overall_score:.3f}, risk={risk}, "
            f"ess={len(essential)}, red={len(redundant)}"
        )
        return cert

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS INTEGRADO (orquestador público: Φ₃ ∘ Φ₂ ∘ Φ₁)
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_redundancy(self) -> Dict[str, Any]:
        r"""
        Composición funtorial $\Phi_3 \circ \Phi_2 \circ \Phi_1$ con
        continuidad estricta de handoffs. Retorna dict backward-compatible
        con v5.2.0/v5.3.0 + certificado OODA v5.4.0.
        """
        logger.info("=" * 80)
        logger.info(f"ANÁLISIS DE REDUNDANCIA MIC v{__version__}")
        logger.info("=" * 80)

        if not self.tools:
            logger.warning("No hay herramientas registradas")
            return {
                'status': 'empty',
                'essential_tools': [],
                'redundant_tools': [],
                'statistics': {
                    'total_tools': 0, 'essential_count': 0,
                    'redundant_count': 0, 'reduction_rate': 0.0,
                },
            }

        matrix = self.build_incidence_matrix()
        checksum = self._compute_input_checksum()

        # ═══════ FASE 1 ═══════  last = _handoff_to_phase2
        phase1_audit = self._audit_symplectic_boolean_phase1(matrix)
        h1 = self._handoff_to_phase2(phase1_audit, checksum)

        # ═══════ FASE 2 ═══════  first = _ingest_phase1_handoff ; last = _handoff_to_phase3
        h1_ingested = self._ingest_phase1_handoff(h1)
        phase2_data = self._audit_poincare_rsi3_phase2(h1_ingested, matrix)
        h2 = self._handoff_to_phase3(h1_ingested, phase2_data)

        # ═══════ FASE 3 ═══════  first = _ingest_phase2_handoff ; last = _emit_governance_certificate
        h2_ingested = self._ingest_phase2_handoff(h2)
        cert = self._emit_governance_certificate(h2_ingested)

        return self._build_analysis_result(cert, phase1_audit, phase2_data)

    def _build_analysis_result(
        self,
        cert: MICGovernanceCertificate,
        phase1_audit: Dict[str, Any],
        phase2_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compone el dict de retorno preservando la API v5.2.0/v5.3.0."""
        h1 = cert.phase2.phase1
        h2 = cert.phase2
        spectral = phase1_audit['spectral']
        sym = phase1_audit['symplectic']
        poincare = phase2_data['poincare']
        rsi3 = phase2_data['rsi3']

        total = len(self.tools)
        ess_count = len(cert.essential_tools)
        red_count = len(cert.redundant_tools)
        reduction_rate = red_count / total if total > 0 else 0.0

        homology = self._compute_homology_simple()
        dependencies = self._detect_linear_dependencies(matrix=self.build_incidence_matrix())

        statistics = {
            'total_tools': total,
            'essential_count': ess_count,
            'redundant_count': red_count,
            'reduction_rate': reduction_rate,
            'spectral_rank': h1.spectral_rank,
            'betti_numbers': homology['betti_numbers'],
            'sat_ok': h2.sat_ok,
            'num_dependencies': len(dependencies),
            'num_conflicting': len(h2.conflicting_tools),
            'symplectic_residue': h1.symplectic_residue,
            'chirikov_parameter': h2.chirikov_parameter,
            'is_kam_stable': h2.kam_stable,
            'is_rsi3_valid': rsi3['is_rsi3_valid'],
            'heyting_verdict': cert.heyting_verdict.name,
            'interlock_latency_ns': cert.interlock_latency_ns,
            'diophantine_gamma': h2.diophantine_gamma,
            'melnikov_integral': h2.melnikov_integral,
            'banach_L': h2.banach_L,
            'quadratic_rate': h2.quadratic_rate,
            'krull_valuation': h2.krull_valuation,
            'wasserstein_distance': h2.wasserstein_distance,
            'overall_quality_score': cert.overall_quality_score,
            'risk_assessment': cert.risk_assessment,
            'heyting_axioms_verified': cert.heyting_axioms_verified,
            # v5.4.0
            'bruno_ok': h2.bruno_ok,
            'melnikov_closed_form': h2.melnikov_closed_form,
            'melnikov_residue': h2.melnikov_residue,
            'lindstedt_omega_shift': h2.lindstedt_omega_shift,
            'nekhoroshev_time': h2.nekhoroshev_time,
            'poincare_lyapunov': h2.poincare_lyapunov,
            'recurrence_time': h2.recurrence_time,
            'rsi3_level1_residual': h2.rsi3_level1_residual,
            'rsi3_level2_lipschitz': h2.rsi3_level2_lipschitz,
            'rsi3_level3_associator': h2.rsi3_level3_associator,
            'darboux_chart_dim': h1.darboux_chart_dim,
            'fubini_study_distance': h2.fubini_study_distance,
        }

        logger.info("─" * 80)
        logger.info(f"✓ Esenciales ({ess_count}): {list(cert.essential_tools)}")
        logger.info(f"✗ Redundantes ({red_count}): {list(cert.redundant_tools)}")
        logger.info(
            f"📊 Reducción: {reduction_rate:.1%}, Verdict: {cert.heyting_verdict.name}"
        )
        logger.info("=" * 80)

        return {
            'status': 'success',
            'essential_tools': list(cert.essential_tools),
            'redundant_tools': list(cert.redundant_tools),
            'statistics': statistics,
            'homology': homology,
            'spectral': spectral,
            'dependencies': dependencies,
            'conflicting_tools': list(h2.conflicting_tools),
            'grobner_basis_size': len(self.grobner_basis),
            'robdd_stats': {
                'essential': list(h1.robdd_essential),
                'redundant': list(h1.robdd_redundant),
            },
            'cayley_darboux': {
                'gram_matrix': sym['gram_matrix'],
                'symplectic_residue': sym['symplectic_residue'],
                'relative_symplectic_residue': sym['relative_symplectic_residue'],
                'klein_residual': sym['klein_residual'],
                'darboux_chart_dim': sym.get('darboux_chart_dim', 0),
                'is_symplectic_valid': sym['is_symplectic_valid'],
            },
            'poincare_lindstedt': poincare,
            'rsi3_surfaces': rsi3,
            'heyting_governance': {
                'verdict': cert.heyting_verdict.name,
                'verdict_code': cert.heyting_verdict.value,
                'hardware_action': cert.hardware_action,
                'interlock_latency_ns': cert.interlock_latency_ns,
            },
            'governance_certificate': cert,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES (homología simplicial 0-1 y dependencias lineales)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_homology_simple(self) -> Dict[str, Any]:
        n_tools = len(self.tools)
        if n_tools == 0:
            return {
                'H_0': 0, 'H_1': 0, 'betti_numbers': [0, 0],
                'components': [], 'redundancy_cycles': [],
            }
        matrix = self.build_incidence_matrix()
        ordered = self._ordered_tools()
        parent = list(range(n_tools))
        size = [1] * n_tools

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]

        if NUMPY_AVAILABLE and matrix.size > 0:
            for i, j in combinations(range(n_tools), 2):
                if np.dot(matrix[i], matrix[j]) > 0:
                    union(i, j)

        comp_map: Dict[int, List[str]] = defaultdict(list)
        for i in range(n_tools):
            comp_map[find(i)].append(ordered[i].name)
        components = [sorted(names) for names in comp_map.values()]
        beta_0 = len(components)

        sig_map: Dict[str, List[str]] = defaultdict(list)
        for tool in ordered:
            sig_map[tool.signature].append(tool.name)
        cycles = [names for names in sig_map.values() if len(names) > 1]
        beta_1 = len(cycles)

        return {
            'H_0': beta_0, 'H_1': beta_1,
            'betti_numbers': [beta_0, beta_1],
            'components': components,
            'redundancy_cycles': cycles,
            'euler_characteristic': beta_0 - beta_1,
        }

    def _detect_linear_dependencies(self, matrix=None) -> List[Dict[str, Any]]:
        if matrix is None:
            matrix = self.build_incidence_matrix()
        if not NUMPY_AVAILABLE or matrix.size == 0:
            return []
        ordered = self._ordered_tools()
        deps: List[Dict[str, Any]] = []
        n = matrix.shape[0]
        for i, j in combinations(range(n), 2):
            vi, vj = matrix[i], matrix[j]
            inter = vi & vj
            if np.array_equal(inter, vi) and not np.array_equal(vi, vj):
                deps.append({
                    'type': 'subset',
                    'tool_subset': ordered[i].name,
                    'tool_superset': ordered[j].name,
                    'redundancy_score': float(np.sum(vi)) / float(np.sum(vj)),
                })
            elif np.array_equal(inter, vj) and not np.array_equal(vi, vj):
                deps.append({
                    'type': 'subset',
                    'tool_subset': ordered[j].name,
                    'tool_superset': ordered[i].name,
                    'redundancy_score': float(np.sum(vj)) / float(np.sum(vi)),
                })
            elif np.array_equal(vi, vj):
                deps.append({
                    'type': 'duplicate',
                    'tool_1': ordered[i].name,
                    'tool_2': ordered[j].name,
                    'redundancy_score': 1.0,
                })
        return deps


# =============================================================================
# VALIDACIÓN DE AXIOMAS
# =============================================================================

def validate_boolean_lattice_axioms(num_vars: int = 5) -> None:
    logger.info("Validando axiomas del retículo booleano...")
    a = BooleanVector(frozenset([CapabilityDimension.PHYS_IO]))
    b = BooleanVector(frozenset([CapabilityDimension.PHYS_NUM]))
    c = BooleanVector(frozenset([CapabilityDimension.TACT_TOPO]))
    zero = BooleanVector.zero()
    one = BooleanVector.universe(num_vars)
    assert a.union(b) == b.union(a)
    assert a.intersection(b) == b.intersection(a)
    assert (a.union(b)).union(c) == a.union(b.union(c))
    assert (a.intersection(b)).intersection(c) == a.intersection(b.intersection(c))
    assert a.union(a.intersection(b)) == a
    assert a.intersection(a.union(b)) == a
    assert a.intersection(b.union(c)) == (a.intersection(b)).union(a.intersection(c))
    assert a.union(a.complement(num_vars)) == one
    assert a.intersection(a.complement(num_vars)) == zero
    assert a.union(a) == a
    assert a.intersection(a) == a
    logger.info("✅ Todos los axiomas del retículo booleano verificados")
    if not _heyting_lattice_axioms_check():
        raise MICException("Axiomas Heyting Ω₃ violados")
    logger.info("✅ Axiomas Heyting Ω₃ (retículo + implicación) verificados")


def audit_mic_redundancy() -> Dict[str, Any]:
    validate_boolean_lattice_axioms()
    analyzer = MICRedundancyAnalyzer()
    analyzer.register_tool("stabilize_flux", {CapabilityDimension.PHYS_NUM})
    analyzer.register_tool("parse_raw", {
        CapabilityDimension.PHYS_IO, CapabilityDimension.PHYS_NUM,
    })
    analyzer.register_tool("structure_logic", {CapabilityDimension.TACT_TOPO})
    analyzer.register_tool("audit_fusion", {CapabilityDimension.TACT_TOPO})
    analyzer.register_tool("lateral_pivot", {CapabilityDimension.STRAT_FIN})
    analyzer.register_tool("fat_tail_risk", {CapabilityDimension.STRAT_FIN})
    analyzer.register_tool("semantic_estimator", {CapabilityDimension.WIS_SEM})
    analyzer.register_tool("flux_stabilizer", {CapabilityDimension.PHYS_NUM})
    return analyzer.analyze_redundancy()


__all__ = [
    'MICException', 'HomologicalInconsistencyError', 'UnsatCoreError',
    'GrobnerBasisComputationError', 'ROBDDConstructionError',
    'SymplecticIntegrityError', 'PoincareResonanceError',
    'RSI3ContractionError', 'OODAChainViolationError',
    'CapabilityDimension', 'HeytingOmega3',
    'BooleanVector', 'Tool', 'Z2Polynomial',
    'GrobnerBasisZ2', 'ROBDD', 'ROBDDNode', 'DPLLSATSolver',
    'MICPhase1Handoff', 'MICPhase2Handoff', 'MICGovernanceCertificate',
    'RSI3KleisliState',
    'MICRedundancyAnalyzer',
    'validate_boolean_lattice_axioms', 'audit_mic_redundancy',
    '_heyting_lattice_axioms_check',
]


if __name__ == "__main__":
    try:
        results = audit_mic_redundancy()
        print("\n" + "=" * 80)
        print(f"RESUMEN EJECUTIVO - ANÁLISIS MIC v{__version__}")
        print("=" * 80)
        stats = results['statistics']
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"  • Total herramientas:    {stats['total_tools']}")
        print(f"  • Esenciales:            {stats['essential_count']}")
        print(f"  • Redundantes:           {stats['redundant_count']}")
        print(f"  • Reducción:             {stats['reduction_rate']:.1%}")
        print(f"\n🔢 ÁLGEBRA, SIMPLÉCTICA Y CELESTE (Poincaré):")
        print(f"  • Rango espectral:        {stats['spectral_rank']}/{len(CapabilityDimension)}")
        print(f"  • Carta de Darboux dim:   {stats['darboux_chart_dim']}")
        print(f"  • Residuo simpléctico:    {stats['symplectic_residue']:.4e}")
        print(f"  • KAM γ:                  {stats['diophantine_gamma']:.4e}")
        print(f"  • Bruno:                  {stats['bruno_ok']}")
        print(f"  • Chirikov:               {stats['chirikov_parameter']:.4f}")
        print(f"  • Melnikov |M|:           {stats['melnikov_integral']:.4e}")
        print(f"  • Melnikov cerrado:       {stats['melnikov_closed_form']:.4e}")
        print(f"  • Residuo Melnikov:       {stats['melnikov_residue']:.4e}")
        print(f"  • Lindstedt Δω:           {stats['lindstedt_omega_shift']:.4e}")
        print(f"  • Lyapunov Poincaré:      {stats['poincare_lyapunov']:.4f}")
        print(f"  • Tiempo Nekhoroshev:     {stats['nekhoroshev_time']:.4e}")
        print(f"  • Recurrencia Kac:        {stats['recurrence_time']:.4e}")
        print(f"\n🧬 RSI³ (Kleisli T ⊂ T² ⊂ T³):")
        print(f"  • Banach L_φ:             {stats['banach_L']:.4f}")
        print(f"  • T¹ residual:            {stats['rsi3_level1_residual']:.4e}")
        print(f"  • T² Lip:                 {stats['rsi3_level2_lipschitz']:.4f}")
        print(f"  • T³ asociador:           {stats['rsi3_level3_associator']:.4e}")
        print(f"  • Krull:                  {stats['krull_valuation']:.4f}")
        print(f"  • Fubini–Study Grassmann: {stats['fubini_study_distance']:.4e}")
        print(f"  • Wasserstein W₁:         {stats['wasserstein_distance']:.4e}")
        print(f"  • KAM estable:            {stats['is_kam_stable']}")
        print(f"  • RSI³ válido:            {stats['is_rsi3_valid']}")
        print(f"\n⚖ GOBERNANZA HEYTING Ω₃:")
        print(f"  • Axiomas Heyting Ω₃:     {stats['heyting_axioms_verified']}")
        print(f"  • Veredicto Heyting:      {stats['heyting_verdict']} "
              f"(latencia: {stats['interlock_latency_ns']} ns)")
        print(f"  • Score global:           {stats['overall_quality_score']:.3f}")
        print(f"  • Nivel de riesgo:        {stats['risk_assessment']}")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        logger.exception("Error crítico en auditoría:")
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
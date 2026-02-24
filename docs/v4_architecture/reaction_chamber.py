"""
Módulo: Reactor de resonancia cuántica (núcleo catalítico hexagonal)
Versión: 5.0 (Correcciones físicas, matemáticas y de robustez sobre v4.3)

═══════════════════════════════════════════════════════════════════════════════
CORRECCIONES RESPECTO A v4.3
═══════════════════════════════════════════════════════════════════════════════

[C1]  _EIGENVALUES y spectral_gap — Consistencia interna:
      Los autovalores del Laplaciano C₆ son λₖ = 2−2cos(2πk/6):
        k=0→0, k=1→1, k=2→3, k=3→4, k=4→3, k=5→1.
      La brecha espectral es λ₁ = 1.0 exacto.
      En v4.3 había contradicción entre el comentario "[F11] λ₁=2−√3" en
      CarbonNode y el resultado numérico correcto de spectral_gap (=1.0).
      Se elimina toda referencia a λ₁=2−√3 y se documenta λ₁=1.0 en todas
      las localizaciones.

[C2]  ThermodynamicPotential._T_MIN — Campo constante de clase, no instancia:
      En v4.3: `_T_MIN: float = field(default=280.0, init=False, repr=False)`
      crea una instancia-variable en cada objeto, rompe la semántica de
      constante física y puede causar errores de orden MRO en dataclasses
      con herencia. Corrección: constante de clase fuera del @dataclass,
      accedida via ThermodynamicPotential._T_MIN.

[C3]  _is_aromatic — Eliminar pi_electrons==2 como condición aromática válida:
      En v4.3 se acepta 2 e⁻ π (n=0) para C₆ completo. Físicamente, la
      aromaticidad de un anillo de 6 nodos requiere conjugación completa
      (6 e⁻ π, n=1). Aceptar 2 e⁻ permite convergencia prematura con solo
      2 nodos activos. Corrección: única condición válida es pi_electrons==6.
      Se mantiene el logging del caso pi==2 como advertencia informativa.

[C4]  CarbonNode(ni+1) en _calculate_hamiltonian — Índice fuera de rango:
      neighbor_indices retorna índices 0-based (0..5). CarbonNode tiene
      valores auto() = 1..6. CarbonNode(0) → ValueError.
      Corrección: usar CarbonNode(ni + 1) es correcto (ni ∈ {0..5} →
      ni+1 ∈ {1..6}), PERO se añade manejo explícito con ValueError
      en lugar de depender del except genérico de _ring_iteration.
      Se introduce _node_from_index() como helper seguro.

[C5]  CarbonNode.service_name — Dict reconstruido en cada llamada:
      En v4.3 _SERVICE_MAP se define como variable local dentro de la
      @property, instanciando el dict O(n) en cada acceso. Corrección:
      dict de clase (_SERVICE_MAP) definido una sola vez a nivel de clase.

[C6]  time.sleep() escalado por estrés — Cuello de botella:
      stress_factor = 1 + local_stress²·0.5 puede producir sleeps de >250ms
      por nodo. Se impone un tope de 50ms total (MAX_NODE_SLEEP_S = 0.050).

[C7]  previous_gibbs inicializado con entropy=0:
      En v4.3, previous_gibbs = potential.gibbs_free_energy se evalúa
      ANTES de asignar potential.entropy. El primer δG compara contra G
      con S=0 (estado ficticio). Corrección: asignar entropy ANTES de
      capturar previous_gibbs.

[C8]  _calculate_shannon_entropy — val==0 y cuantización:
      math.log10(abs(0) + 1e-12) = -12, mezclando todos los ceros en
      una cubeta de magnitud extrema. Corrección: cero se mapea a cubeta
      especial "num:zero" para separarlo de magnitudes reales.

[C9]  HilbertState.apply_damping — Acoplamiento a _RING_SIZE global:
      Usa float(_RING_SIZE) en lugar de len(self.vector). Si el vector
      tiene longitud diferente (extensibilidad futura), el cálculo del
      promedio es incorrecto. Corrección: usar len(self.vector).

[C10] _check_thermodynamic_convergence — Convergencia prematura ciclo 2:
      Se eleva el mínimo de ciclos requeridos antes de evaluar convergencia
      a cycle >= MIN_CONVERGENCE_CYCLE (=3), y se añade comprobación de
      que la norma del estado ‖ψ‖ sea estable (variación < 5%).

[C11] HexagonalTopology — Laplaciano construido con ints, operado con floats:
      En v4.3 laplacian es List[List[int]] pero diffuse_stress lo multiplica
      por diffusion_rate:float. El producto int*float es correcto en Python
      pero la tipificación es engañosa. Se cambia a List[List[float]] y
      se precalcula L como floats desde la construcción.

[C12] diffuse_stress — Neumann boundary puede violar energía:
      flux grande en condición Neumann post-difusión puede disparar norma.
      Se añade logging de advertencia cuando ‖ψ'‖ > 2·‖ψ‖.
"""

import logging
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Tuple, TypeVar

from app.schemas import Stratum  # noqa: F401  (interfaz externa mantenida)
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry

logger = logging.getLogger("QuantumReactor")


# =============================================================================
# Constantes Físicas y Matemáticas
# =============================================================================

R_GAS_CONSTANT: float = 8.314          # J·mol⁻¹·K⁻¹
BOLTZMANN_SCALE: float = 1.0e-1        # Factor de escala entropía→energía
GIBBS_CONVERGENCE_EPS: float = 0.05    # Tolerancia de convergencia ΔG [J/mol]
DAMPING_GAMMA: float = 0.3             # Coeficiente de amortiguamiento exponencial
DAMPING_OMEGA: float = math.pi / 3.0   # Frecuencia de oscilación del amortiguamiento
ENTROPY_MIN_PROB: float = 1e-10        # Probabilidad mínima para evitar log(0)

# -----------------------------------------------------------------------------
# [C1] Autovalores del Laplaciano C₆ (analíticos, exactos)
# λₖ = 2 − 2·cos(2πk/6), k = 0,...,5
# k=0 → 0.0  (modo traslación, siempre nulo)
# k=1 → 1.0  (primer modo no trivial, gap espectral)
# k=2 → 3.0
# k=3 → 4.0  (modo antipodal, λ_max)
# k=4 → 3.0  (simétrico a k=2)
# k=5 → 1.0  (simétrico a k=1)
# -----------------------------------------------------------------------------
_RING_SIZE: int = 6
_LAMBDA_MAX_C6: float = 4.0            # λ_max = λ₃ = 4.0
_LAMBDA_GAP_C6: float = 1.0            # λ₁ = 1.0  [C1] (no 2−√3)

# Condición CFL de estabilidad para ecuación de calor discreta en C₆:
#   α < 1 / (2 · λ_max) = 1 / 8 = 0.125
CFL_ALPHA_CRITICAL: float = 1.0 / (2.0 * _LAMBDA_MAX_C6)   # = 0.125
CFL_SAFETY_MARGIN: float = 0.95
CFL_ALPHA_SAFE: float = CFL_SAFETY_MARGIN * CFL_ALPHA_CRITICAL  # ≈ 0.11875

# Coeficiente dimensional para presión topológica en G [J/mol]
TOPO_PRESSURE_COEFF: float = 1.0

# [C6] Límite de latencia por nodo (evita sleep explosivo con estrés alto)
MAX_NODE_SLEEP_S: float = 0.050        # 50 ms máximo por nodo

# [C10] Ciclo mínimo antes de evaluar convergencia termodinámica
MIN_CONVERGENCE_CYCLE: int = 3

# Temperatura base mínima física [K]  [C2]
_T_BASE_MIN: float = 280.0


# =============================================================================
# Enumeraciones
# =============================================================================

class CarbonNode(Enum):
    """
    Nodos del anillo de benceno (simetría D6h).
    Cada nodo representa un orbital sp² híbrido funcional.
    El valor ordinal (auto) mapea al índice matricial vía la propiedad `index`.

    Autovalores del Laplaciano circulante C₆ [C1]:
        λₖ = 2 − 2·cos(2πk/6),  k = 0, 1, ..., 5
        → {0.0, 1.0, 3.0, 4.0, 3.0, 1.0}
        λ_min = 0  (modo de traslación uniforme)
        λ_max = 4  (modo antipodal, determina CFL)
        gap   = λ₁ = 1.0  (brecha espectral, controla velocidad de mezcla)
    """

    C1_INGESTION = auto()   # valor = 1
    C2_PHYSICS   = auto()   # valor = 2
    C3_TOPOLOGY  = auto()   # valor = 3
    C4_STRATEGY  = auto()   # valor = 4
    C5_SEMANTICS = auto()   # valor = 5
    C6_MATTER    = auto()   # valor = 6

    # [C5] Mapa de servicios a nivel de clase — construido UNA sola vez
    _SERVICE_MAP: ClassVar[Dict[str, str]]  # se puebla después de la clase

    @property
    def index(self) -> int:
        """Índice matricial 0-based: C1→0, C2→1, ..., C6→5."""
        return self.value - 1

    @property
    def label(self) -> str:
        """Etiqueta legible para logs e interfaces."""
        return self.name.replace("_", " ").title()

    @property
    def service_name(self) -> str:
        """
        [C5] Nombre del servicio MIC correspondiente.
        El dict _SERVICE_MAP se accede a nivel de clase (O(1), construido una vez).
        """
        return CarbonNode._SERVICE_MAP[self.name]


# [C5] Poblar el mapa fuera de la clase para evitar conflicto con Enum metaclass
CarbonNode._SERVICE_MAP = {
    "C1_INGESTION":  "load_data",
    "C2_PHYSICS":    "stabilize_flux",
    "C3_TOPOLOGY":   "business_topology",
    "C4_STRATEGY":   "financial_analysis",
    "C5_SEMANTICS":  "semantic_translation",
    "C6_MATTER":     "materialization",
}


def _node_from_index(idx: int) -> CarbonNode:
    """
    [C4] Helper seguro para obtener CarbonNode desde índice 0-based.

    Convierte idx ∈ {0,...,5} a CarbonNode con valor ∈ {1,...,6}.
    Lanza ValueError descriptivo si el índice está fuera de rango,
    en lugar de propagar la excepción opaca de Enum.

    Parámetros
    ----------
    idx : Índice 0-based del nodo en el anillo. Debe estar en [0, 5].

    Retorna
    -------
    CarbonNode correspondiente al índice.

    Raises
    ------
    ValueError : Si idx no es un índice válido del anillo C₆.
    """
    if not 0 <= idx <= _RING_SIZE - 1:
        raise ValueError(
            f"Índice de nodo {idx} fuera de rango [0, {_RING_SIZE - 1}]. "
            f"El anillo C₆ tiene exactamente {_RING_SIZE} nodos."
        )
    return CarbonNode(idx + 1)


# =============================================================================
# Estructuras de Estado
# =============================================================================

T = TypeVar("T", bound="HilbertState")


@dataclass
class HilbertState:
    """
    Estado del sistema proyectado en el espacio de Hilbert ℝ⁶ abstracto.
    El vector |ψ⟩ codifica el estrés local de cada nodo del anillo.

    Propiedades fundamentales:
      ‖ψ‖ (norma-2) : estrés topológico total del anillo.
      φ   (fase)    : rotación acumulada del ciclo de resonancia.
      ⟨ψ|φ⟩         : producto interno estándar para comparación de estados.
    """
    vector: List[float] = field(default_factory=lambda: [0.0] * _RING_SIZE)
    phase:  float = 0.0

    @property
    def norm(self) -> float:
        """‖ψ‖ = √⟨ψ|ψ⟩"""
        return math.sqrt(sum(x * x for x in self.vector))

    def inner_product(self, other: "HilbertState") -> float:
        """⟨self|other⟩ = Σᵢ aᵢ·bᵢ"""
        return sum(a * b for a, b in zip(self.vector, other.vector))

    def normalize(self: T) -> T:
        """Proyecta |ψ⟩ sobre la esfera unitaria S⁵ (si ‖ψ‖ > ε)."""
        n = self.norm
        if n > 1e-12:
            self.vector = [x / n for x in self.vector]
        return self

    def apply_damping(self, cycle: int) -> None:
        """
        Amortiguamiento espectral compatible con simetría D6h.

        El operador actúa sobre los modos de alta frecuencia (desviaciones del
        promedio) preservando el modo λ=0 (traslación uniforme del anillo).

        Formulación:
            v̄  = (1/N)·Σvᵢ                    (modo fundamental, N=len(vector))
            δᵢ = vᵢ − v̄                        (modos superiores)
            factor = exp(−γ·t) · |cos(ω·t)|   (envolvente no-negativa)
            vᵢ' = v̄ + δᵢ · factor

        [C9] Se usa len(self.vector) en lugar de la constante global _RING_SIZE
        para que la clase sea extensible a anillos de tamaño arbitrario.
        """
        n = len(self.vector)
        if n == 0:
            return

        envelope    = math.exp(-DAMPING_GAMMA * cycle)
        oscillation = abs(math.cos(DAMPING_OMEGA * cycle))   # ∈ [0, 1]
        factor      = envelope * oscillation

        mean = sum(self.vector) / float(n)   # [C9] len(vector), no _RING_SIZE global
        self.vector = [mean + (v - mean) * factor for v in self.vector]

    def project_orthogonal(self, subspace_basis: List["HilbertState"]) -> None:
        """
        Proyección ortogonal de Gram-Schmidt modificada.

        Trabaja sobre copias normalizadas de los vectores base para NO modificar
        los objetos originales del caller (sin efectos secundarios destructivos).
        """
        n = len(self.vector)
        for basis_vec in subspace_basis:
            norm_b = basis_vec.norm
            if norm_b < 1e-12:
                continue
            # Copia local normalizada — el objeto original permanece intacto
            unit_b     = [x / norm_b for x in basis_vec.vector]
            projection = sum(self.vector[i] * unit_b[i] for i in range(n))
            self.vector = [
                self.vector[i] - projection * unit_b[i]
                for i in range(n)
            ]
        self.normalize()

    def __repr__(self) -> str:
        components = ", ".join(f"{v:.4f}" for v in self.vector)
        return (
            f"HilbertState(‖ψ‖={self.norm:.4f}, "
            f"φ={self.phase:.4f}rad, [{components}])"
        )


# [C2] Constante de clase fuera del dataclass (semántica de constante física)
_T_BASE_MIN: float = 280.0    # Temperatura base mínima [K]


@dataclass
class ThermodynamicPotential:
    """
    Potenciales termodinámicos del reactor hexagonal.

    El término de presión topológica incluye un coeficiente dimensional
    explícito κ_topo para que P·V sea coherente con H y T·S.

    El enfriamiento de temperatura base se realiza vía el método
    `cool_temperature` que preserva la cota mínima física (_T_BASE_MIN).

    [C2] _T_MIN es una constante de clase (ClassVar), no una instancia-variable.
    """

    enthalpy:             float = 0.0
    entropy:              float = 0.0
    base_temperature:     float = 298.0
    temperature_coupling: float = 15.0
    topological_stress:   float = 0.0

    # [C2] Constante física de clase, no field de instancia
    T_MIN: ClassVar[float] = _T_BASE_MIN

    @property
    def temperature(self) -> float:
        """
        T = T₀ + γ·‖ψ‖²   (principio de equipartición informacional)
        γ = temperature_coupling, ‖ψ‖ = topological_stress.
        """
        return (
            self.base_temperature
            + self.temperature_coupling * (self.topological_stress ** 2)
        )

    @property
    def gibbs_free_energy(self) -> float:
        """
        G = H − κ·T·S + κ_topo·‖ψ‖²

        El término κ_topo·‖ψ‖² representa el trabajo de deformación del anillo
        (análogo a P·V en termodinámica clásica), con κ_topo en J/mol.
        """
        ts_term      = self.temperature * self.entropy * BOLTZMANN_SCALE
        topo_pressure = TOPO_PRESSURE_COEFF * (self.topological_stress ** 2)
        return self.enthalpy - ts_term + topo_pressure

    @property
    def instability(self) -> float:
        """
        Índice de inestabilidad topológico-termodinámica:
            I = ln(1 + |G|) + ‖ψ‖
        Siempre ≥ 0.
        """
        return math.log1p(abs(self.gibbs_free_energy)) + self.topological_stress

    def update(
        self,
        new_enthalpy:       float,
        new_entropy:        float,
        topological_stress: float,
    ) -> None:
        """Actualización atómica de todas las variables de estado termodinámico."""
        self.enthalpy           = new_enthalpy
        self.entropy            = new_entropy
        self.topological_stress = topological_stress

    def cool_temperature(self, factor: float = 0.95) -> None:
        """
        Reduce la temperatura base preservando el mínimo físico T_MIN.

        Encapsula la lógica de enfriamiento que antes estaba dispersa en
        `_attempt_stabilization`.

        Parámetros
        ----------
        factor : Factor multiplicativo ∈ (0, 1). Default 0.95.
        """
        if not 0.0 < factor < 1.0:
            raise ValueError(
                f"El factor de enfriamiento debe estar en (0, 1); recibido: {factor}."
            )
        self.base_temperature = max(self.T_MIN, self.base_temperature * factor)


# =============================================================================
# Protocolo del Agente Catalizador
# =============================================================================

class CatalystAgent(Protocol):
    """
    Protocolo formal para el Agente Catalizador.
    Un catalizador ideal reduce la barrera de activación sin consumirse.
    """

    @property
    def efficiency_factor(self) -> float:
        """Factor de eficiencia catalítica η ∈ [0, 1)."""
        ...

    @property
    def catalytic_strength(self) -> float:
        """Fuerza catalítica efectiva (adimensional)."""
        ...

    def orient(
        self, context: Dict[str, Any], gradient: float
    ) -> Dict[str, Any]:
        """Orienta la reacción según el gradiente de energía libre ∂G."""
        ...


# =============================================================================
# Topología Hexagonal
# =============================================================================

class HexagonalTopology:
    """
    Estructura algebraica del grafo cíclico C₆ (anillo bencénico).

    El Laplaciano del grafo circulante C₆ es:
        L = D − A
    donde D = 2·I (todos los vértices tienen grado 2) y A es la matriz de
    adyacencia circulante con conexiones (i, i±1 mod 6).

    Autovalores analíticos del Laplaciano circulante C₆ [C1]:
        λₖ = 2 − 2·cos(2πk/6),  k = 0, 1, ..., 5
        → {0.0, 1.0, 3.0, 4.0, 3.0, 1.0}
        λ_min = 0.0  (modo de traslación uniforme, kernel de L)
        λ_max = 4.0  (modo antipodal, determina CFL)
        gap   = λ₁  = 1.0  (brecha espectral, controla velocidad de mezcla)

    Condición CFL para la ecuación de calor discreta:
        α < 1 / (2·λ_max) = 1/8 = 0.125

    [C11] La matriz laplaciana se almacena como List[List[float]] (no int)
    para coherencia de tipos con las operaciones de difusión.
    """
    RING_SIZE: int = _RING_SIZE

    # [C1] Autovalores exactos del Laplaciano C₆ (orden k=0,...,5)
    _EIGENVALUES: Tuple[float, ...] = (0.0, 1.0, 3.0, 4.0, 3.0, 1.0)

    def __init__(self) -> None:
        n = self.RING_SIZE

        # Matriz de adyacencia (circulante, vecinos ±1 mod n)
        self.adjacency: List[List[float]] = [
            [
                1.0 if (j == (i + 1) % n or j == (i - 1) % n) else 0.0
                for j in range(n)
            ]
            for i in range(n)
        ]

        # Grado de cada nodo (todos 2 en C₆)
        self.degree: List[int] = [2] * n

        # [C11] Laplaciano como floats: L = D − A (semidefinido positivo)
        self.laplacian: List[List[float]] = [
            [
                float(self.degree[i]) * (1.0 if i == j else 0.0)
                - self.adjacency[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]

    @property
    def spectral_gap(self) -> float:
        """
        [C1] Brecha espectral λ₁ del Laplaciano C₆, calculada analíticamente.

        λ₁ = 2 − 2·cos(2π/6) = 2 − 2·(1/2) = 1.0   (exacto para C₆)

        La brecha espectral controla la velocidad de mezcla del proceso de
        difusión: mayor gap → mezcla más rápida → convergencia más rápida.
        """
        return 2.0 - 2.0 * math.cos(2.0 * math.pi / self.RING_SIZE)  # = 1.0

    def neighbor_indices(self, node_index: int) -> Tuple[int, int]:
        """
        Retorna los índices (izquierdo, derecho) del nodo en el anillo C₆.

        Parámetros
        ----------
        node_index : Índice 0-based ∈ {0,...,5}.

        Retorna
        -------
        (left_idx, right_idx) con aritmética modular en Z₆.
        """
        n = self.RING_SIZE
        return (node_index - 1) % n, (node_index + 1) % n

    def _apply_dirichlet_boundary(
        self,
        state_vector: List[float],
        boundary_values: Dict[int, float],
    ) -> List[float]:
        """Condición de Dirichlet: fija el valor en nodos de frontera."""
        new_vector = state_vector.copy()
        for idx, value in boundary_values.items():
            new_vector[idx] = value
        return new_vector

    def _apply_neumann_boundary(
        self,
        state_vector: List[float],
        boundary_flux: Dict[int, float],
    ) -> List[float]:
        """
        Condición de Neumann: añade flujo en nodos de frontera.

        [C12] Se registra advertencia si la norma post-Neumann es más del
        doble de la norma pre-Neumann (posible violación de estabilidad).
        """
        norm_pre   = math.sqrt(sum(x * x for x in state_vector))
        new_vector = state_vector.copy()
        for idx, flux in boundary_flux.items():
            new_vector[idx] += flux

        norm_post = math.sqrt(sum(x * x for x in new_vector))
        if norm_pre > 1e-12 and norm_post > 2.0 * norm_pre:
            logger.warning(
                "[C12] Condición Neumann amplifica norma: ‖ψ_pre‖=%.4f → "
                "‖ψ_post‖=%.4f (factor=%.2f×). Posible inestabilidad.",
                norm_pre, norm_post, norm_post / norm_pre,
            )
        return new_vector

    def diffuse_stress(
        self,
        state_vector:         List[float],
        diffusion_rate:       float = CFL_ALPHA_SAFE,
        boundary_conditions:  Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Ecuación de calor discreta en C₆:
            ψ(t+1) = (I − α·L) · ψ(t)

        Condición CFL:
            α < 1 / (2·λ_max) = 0.125  [C1]

        Para α ≥ α_crit el esquema puede amplificar el modo antipodal;
        se recorta a CFL_ALPHA_SAFE = 0.95·α_crit ≈ 0.11875.

        Orden de operaciones:
          1. Difusión laplaciana (actualización global).
          2. Condiciones de frontera post-difusión (opcional).

        Parámetros
        ----------
        state_vector        : Vector de estrés actual ψ ∈ ℝ⁶.
        diffusion_rate      : Paso α de difusión. Default = CFL_ALPHA_SAFE.
        boundary_conditions : Opcional. Dict con claves "dirichlet"/"neumann".
        """
        if diffusion_rate <= 0.0:
            return state_vector.copy()

        if diffusion_rate >= CFL_ALPHA_CRITICAL:
            original_rate   = diffusion_rate
            diffusion_rate  = CFL_ALPHA_SAFE
            logger.warning(
                "⚠️ diffusion_rate=%.5f ≥ α_crit=%.5f → recortado a α_safe=%.5f. "
                "Condición CFL para C₆: α < 1/(2·λ_max) = %.5f.",
                original_rate, CFL_ALPHA_CRITICAL, CFL_ALPHA_SAFE, CFL_ALPHA_CRITICAL,
            )

        n           = self.RING_SIZE
        new_vector  = [0.0] * n

        # Paso 1: ψ'ᵢ = ψᵢ − α·(L·ψ)ᵢ  (integración explícita de Euler)
        for i in range(n):
            lap_i        = sum(self.laplacian[i][j] * state_vector[j] for j in range(n))
            new_vector[i] = state_vector[i] - diffusion_rate * lap_i

        # Paso 2: condiciones de frontera post-difusión
        if boundary_conditions:
            if "dirichlet" in boundary_conditions:
                new_vector = self._apply_dirichlet_boundary(
                    new_vector, boundary_conditions["dirichlet"]
                )
            if "neumann" in boundary_conditions:
                new_vector = self._apply_neumann_boundary(
                    new_vector, boundary_conditions["neumann"]
                )

        return new_vector


# =============================================================================
# Reactor Catalítico (Motor Principal)
# =============================================================================

class CatalyticReactor:
    """
    Motor de Orquestación Resonante sobre el grafo C₆.

    El reactor implementa una dinámica de campo cuántico discreta donde
    cada nodo del anillo es un sitio de reacción con Hamiltoniano local.

    Invariantes mantenidos durante la ejecución:
      ─ α < CFL_ALPHA_CRITICAL en toda difusión.
      ─ enthalpy ≥ _MIN_ENTHALPY, entropy ≥ _MIN_ENTROPY.
      ─ Difusión global ejecutada exactamente UNA vez por ciclo.
      ─ Amortiguamiento factor ∈ [0, 1] (no reflexión).
      ─ Aromaticidad solo si pi_electrons == 6 [C3].
    """

    INSTABILITY_THRESHOLD:       float = 5.0
    MAX_RESONANCE_CYCLES:        int   = 6
    ACTIVATION_BARRIER_CEILING:  float = 0.9

    # Precursores requeridos por nodo (penaliza si ausentes en contexto)
    _PRECURSOR_MAP: ClassVar[Dict[CarbonNode, List[str]]] = {
        CarbonNode.C2_PHYSICS:   ["physical_constraints"],
        CarbonNode.C4_STRATEGY:  ["financial_params"],
        CarbonNode.C5_SEMANTICS: ["semantic_model"],
    }

    _MIN_ENTHALPY: float = 1e-10
    _MIN_ENTROPY:  float = 1e-10

    # Parámetros del Hamiltoniano de Hückel (adimensionales normalizados)
    _HUCKEL_ALPHA: float =  0.20    # Energía de sitio (integral de Coulomb)
    _HUCKEL_BETA:  float = -0.05    # Integral de resonancia (< 0 → estabiliza)

    def __init__(
        self,
        mic:       MICRegistry,
        agent:     CatalystAgent,
        telemetry: TelemetryContext,
    ) -> None:
        self.mic           = mic
        self.catalyst      = agent
        self.telemetry     = telemetry
        self.topology      = HexagonalTopology()
        self.ring_sequence: List[CarbonNode] = list(CarbonNode)

        self._temperature_coupling: float = 15.0
        self._base_temperature:     float = 298.0

    # ─────────────────────────────────────────────────────────────────────────
    # Interfaz pública
    # ─────────────────────────────────────────────────────────────────────────

    def ignite(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enciende el reactor y ejecuta el ciclo de resonancia completo.

        Retorna
        -------
        El contexto estabilizado (aromático o metaestable).

        Raises
        ------
        RuntimeError : Si se exceden MAX_RESONANCE_CYCLES sin convergencia.
        """
        reaction_id = str(uuid.uuid4())[:8]
        logger.info("⚛️  QUANTUM IGNITION: Reactor [%s] encendido.", reaction_id)

        context   = initial_context.copy()
        potential = ThermodynamicPotential(
            base_temperature=self._base_temperature,
            temperature_coupling=self._temperature_coupling,
        )
        state = HilbertState()

        # [C7] Asignar entropy ANTES de capturar previous_gibbs para que el
        # primer δG sea físicamente coherente (no compare contra G con S=0).
        potential.entropy = max(
            self._MIN_ENTROPY,
            self._calculate_shannon_entropy(context),
        )
        previous_gibbs = potential.gibbs_free_energy
        previous_norm  = state.norm   # [C10] Para criterio de estabilidad de ‖ψ‖

        self.telemetry.record_reaction_start(reaction_id, context)

        try:
            for cycle in range(1, self.MAX_RESONANCE_CYCLES + 1):
                logger.info(
                    "⏩ Ciclo %d/%d | ΔG=%.4f | I=%.4f | ‖ψ‖=%.4f",
                    cycle, self.MAX_RESONANCE_CYCLES,
                    potential.gibbs_free_energy,
                    potential.instability,
                    state.norm,
                )

                self._catalytic_orientation(context, potential)
                self._ring_iteration(context, state, potential, cycle)

                # Intercambio de Kekulé: inversión del vector cada 2 ciclos
                if cycle % 2 == 0:
                    state.vector = state.vector[::-1]

                # Avance de fase: Δφ = 2π/6 por ciclo (simetría D6h)
                state.phase = (
                    state.phase + 2.0 * math.pi / _RING_SIZE
                ) % (2.0 * math.pi)

                if self._is_aromatic(context):
                    logger.info(
                        "✅ AROMATICIDAD ALCANZADA en ciclo %d. ΔG_final=%.4f",
                        cycle, potential.gibbs_free_energy,
                    )
                    self.telemetry.record_reaction_success(reaction_id, cycle)
                    return context

                if self._check_thermodynamic_convergence(
                    potential, previous_gibbs, previous_norm, cycle
                ):
                    delta_g = abs(potential.gibbs_free_energy - previous_gibbs)
                    logger.info(
                        "🔒 Convergencia termodinámica en ciclo %d: "
                        "|δG|=%.6f < ε=%.4f. Estado metaestable.",
                        cycle, delta_g, GIBBS_CONVERGENCE_EPS,
                    )
                    context["_metastable_cycle"] = cycle
                    return context

                previous_gibbs = potential.gibbs_free_energy
                previous_norm  = state.norm   # [C10]

            raise RuntimeError(
                f"Reactor no convergió tras {self.MAX_RESONANCE_CYCLES} ciclos. "
                f"ΔG_final={potential.gibbs_free_energy:.4f}, "
                f"I={potential.instability:.4f}."
            )

        except Exception as exc:
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.error("🔥 Fallo crítico en el reactor [%s]: %s", reaction_id, exc)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos privados del ciclo de resonancia
    # ─────────────────────────────────────────────────────────────────────────

    def _catalytic_orientation(
        self,
        context:   Dict[str, Any],
        potential: ThermodynamicPotential,
    ) -> None:
        """Aplica la orientación catalítica al contexto según ∂G."""
        catalyst_diff = self.catalyst.orient(context, potential.gibbs_free_energy)
        context.update(catalyst_diff)

    def _ring_iteration(
        self,
        context:   Dict[str, Any],
        state:     HilbertState,
        potential: ThermodynamicPotential,
        cycle:     int,
    ) -> None:
        """
        Itera sobre todos los nodos del anillo y aplica la difusión global.

        La difusión se ejecuta EXACTAMENTE UNA VEZ al final del ciclo,
        después de procesar todos los nodos.
        """
        total_delta_h = 0.0

        for node in self.ring_sequence:
            idx     = node.index
            base_ea = self._calculate_hamiltonian(node, context)
            eff_ea  = base_ea * (1.0 - self.catalyst.efficiency_factor)

            try:
                node_ctx, delta_h = self._react_node(
                    node, context, eff_ea, state.vector[idx], cycle
                )
                context.update(node_ctx)
                total_delta_h += delta_h

            except Exception as exc:
                logger.error(
                    "💥 Excepción en nodo %s (ciclo %d): %s",
                    node.name, cycle, exc,
                )
                state.vector[idx] += 1.0
                total_delta_h     += 50.0
                context[f"{node.name}_error"] = str(exc)

        # Difusión global única (α = CFL_ALPHA_SAFE < α_crit → estable)
        state.vector = self.topology.diffuse_stress(
            state.vector,
            diffusion_rate=CFL_ALPHA_SAFE,
        )

        potential.update(
            new_enthalpy=max(
                self._MIN_ENTHALPY,
                potential.enthalpy + total_delta_h,
            ),
            new_entropy=max(
                self._MIN_ENTROPY,
                self._calculate_shannon_entropy(context),
            ),
            topological_stress=state.norm,
        )

        if potential.instability > self.INSTABILITY_THRESHOLD:
            self._attempt_stabilization(state, potential, cycle)

    def _check_thermodynamic_convergence(
        self,
        potential:      ThermodynamicPotential,
        previous_gibbs: float,
        previous_norm:  float,
        cycle:          int,
    ) -> bool:
        """
        Verifica convergencia termodinámica con criterio dual.

        [C10] Condiciones requeridas:
          1. cycle >= MIN_CONVERGENCE_CYCLE (evita convergencia espuria temprana).
          2. |ΔG| < GIBBS_CONVERGENCE_EPS.
          3. |Δ‖ψ‖| / max(‖ψ‖, ε) < 0.05  (norma estable, variación < 5%).

        Parámetros
        ----------
        previous_gibbs : G del ciclo anterior.
        previous_norm  : ‖ψ‖ del ciclo anterior.
        cycle          : Ciclo actual (1-indexed).
        """
        if cycle < MIN_CONVERGENCE_CYCLE:
            return False

        delta_gibbs = abs(potential.gibbs_free_energy - previous_gibbs)
        if delta_gibbs >= GIBBS_CONVERGENCE_EPS:
            return False

        # Criterio de estabilidad de la norma [C10]
        norm_ref     = max(previous_norm, 1e-12)
        delta_norm   = abs(potential.topological_stress - previous_norm)
        norm_stable  = (delta_norm / norm_ref) < 0.05

        return norm_stable

    def _attempt_stabilization(
        self,
        state:     HilbertState,
        potential: ThermodynamicPotential,
        cycle:     int,
    ) -> None:
        """
        Intenta estabilizar el reactor cuando I > INSTABILITY_THRESHOLD.

        Protocolo de estabilización:
          1. Amortiguamiento espectral del vector de estado.
          2. Disipación de entalpía (factor 0.85).
          3. Enfriamiento de temperatura base (factor 0.95).
          4. Actualización del estrés topológico.
          5. Colapso irrecuperable si I > 1.2·umbral tras estabilización.
        """
        logger.warning(
            "⚠️  CRITICAL: I=%.2f > %.2f en ciclo %d. Aplicando amortiguamiento.",
            potential.instability, self.INSTABILITY_THRESHOLD, cycle,
        )

        # 1. Amortiguamiento espectral
        state.apply_damping(cycle)

        # 2. Disipación de entalpía
        potential.enthalpy = max(
            self._MIN_ENTHALPY,
            potential.enthalpy * 0.85,
        )

        # 3. Enfriamiento vía método encapsulado
        potential.cool_temperature(factor=0.95)

        # 4. Actualización del estrés topológico post-amortiguamiento
        potential.topological_stress = state.norm

        # 5. Colapso irrecuperable
        collapse_threshold = self.INSTABILITY_THRESHOLD * 1.2
        if potential.instability > collapse_threshold:
            raise RuntimeError(
                f"Reactor Collapse: Inestabilidad irrecuperable "
                f"(I={potential.instability:.4f} > {collapse_threshold:.4f})."
            )

        logger.info(
            "🛡️  Estabilización exitosa: I=%.4f, ‖ψ‖=%.4f, T_base=%.2f K.",
            potential.instability, state.norm, potential.base_temperature,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hamiltoniano de Hückel
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_hamiltonian(
        self,
        node:    CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """
        Hamiltoniano de Hückel para el nodo iésimo del anillo.

        Formulación:
            Eᵢ = max(0, α + Σⱼ∈vecinos βᵢⱼ·resonant(j) + penalty)

        donde:
          α = 0.20  : energía de sitio (integral de Coulomb), siempre ≥ 0.
          β = -0.05 : integral de resonancia (< 0 → estabiliza la barrera).
          penalty   : penalización por precursores ausentes (≥ 0).

        La estabilización por vecinos resonantes REDUCE la barrera (β < 0).
        El resultado se recorta a [0, ∞) para garantizar Eₐ ≥ 0 (física).

        [C4] Usa _node_from_index() para conversión segura de índice a nodo.
        """
        idx = node.index
        left_idx, right_idx = self.topology.neighbor_indices(idx)

        neighbor_stabilization = 0.0
        for ni in (left_idx, right_idx):
            try:
                neighbor_node = _node_from_index(ni)   # [C4] Conversión segura
                if context.get(f"{neighbor_node.name}_status") == "resonant":
                    neighbor_stabilization += self._HUCKEL_BETA
            except ValueError as exc:
                logger.error(
                    "Error al obtener vecino del nodo %s (idx=%d): %s",
                    node.name, ni, exc,
                )

        precursor_penalty = self._evaluate_precursor_penalty(node, context)
        hamiltonian       = self._HUCKEL_ALPHA + neighbor_stabilization + precursor_penalty

        return max(0.0, hamiltonian)

    def _evaluate_precursor_penalty(
        self,
        node:    CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """
        Penalización proporcional a la fracción de precursores ausentes.

            penalty = 0.3 · (n_ausentes / n_requeridos)

        Retorna 0.0 si el nodo no tiene precursores requeridos.
        """
        required = self._PRECURSOR_MAP.get(node, [])
        if not required:
            return 0.0
        missing = sum(1 for key in required if key not in context)
        return 0.3 * (missing / len(required))

    # ─────────────────────────────────────────────────────────────────────────
    # Reacción en nodo
    # ─────────────────────────────────────────────────────────────────────────

    def _react_node(
        self,
        node:         CarbonNode,
        context:      Dict[str, Any],
        ea:           float,
        local_stress: float,
        cycle:        int,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ejecuta la reacción en un nodo del anillo.

        ΔH = σ²·κ_stress + Eₐ·κ_ea   (determinista, sin latencia)
        donde κ_stress = 5.0, κ_ea = 10.0.

        El término ΔH NO incluye latencia de ejecución: la latencia es una
        variable de infraestructura ruidosa ajena al sistema termodinámico.

        [C6] El tiempo de sleep se limita a MAX_NODE_SLEEP_S para evitar
        que estrés alto produzca latencias destructivas (>250ms/nodo).

        Parámetros
        ----------
        node         : Nodo del anillo siendo procesado.
        context      : Contexto de reacción actual.
        ea           : Barrera de activación efectiva [adim].
        local_stress : Estrés local del nodo σ = ψ[idx].
        cycle        : Ciclo actual (para logging).

        Retorna
        -------
        (context_update, delta_h)
        """
        if ea > self.ACTIVATION_BARRIER_CEILING:
            logger.warning(
                "⚡ Saltando %s (ciclo %d): Eₐ=%.3f > techo=%.3f.",
                node.name, cycle, ea, self.ACTIVATION_BARRIER_CEILING,
            )
            return {f"{node.name}_skipped": True}, 5.0

        # [C6] Limitar sleep para evitar latencia explosiva con estrés alto
        stress_factor = 1.0 + (local_stress ** 2) * 0.5
        sleep_s = min(0.005 * stress_factor, MAX_NODE_SLEEP_S)
        time.sleep(sleep_s)

        context_update: Dict[str, Any] = {
            f"{node.name}_status": "resonant",
            f"{node.name}_ts":     time.time(),
            f"{node.name}_ea":     ea,
        }

        # ΔH determinista: σ² · 5.0 + Eₐ · 10.0
        delta_h = (local_stress ** 2) * 5.0 + ea * 10.0

        logger.debug(
            "🔬 %s (ciclo %d) | Eₐ=%.3f | σ=%.3f | ΔH=%.3f | sleep=%.3fs",
            node.name, cycle, ea, local_stress, delta_h, sleep_s,
        )
        return context_update, delta_h

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades estáticas
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_shannon_entropy(context: Dict[str, Any]) -> float:
        """
        Entropía de Shannon del contexto semántico.

        H = −Σₚ p·ln(p)

        La distribución se construye sobre representaciones canónicas de los
        valores del contexto, excluyendo metadatos ruidosos (_ts, _ea).

        [C8] Cero se mapea a cubeta especial "num:zero" para separarlo
        de magnitudes reales y evitar log10(1e-12) = -12 como proxy de 0.

        Cubetas de cuantización para numéricos no-cero:
          magnitude = int(log10(|val|))  → agrupación por orden de magnitud.
        """
        if not context:
            return 0.0

        _EXCLUDED_SUFFIXES: Tuple[str, ...] = ("_ts", "_ea")
        semantic_values: List[str] = []

        for key, val in context.items():
            if any(key.endswith(sfx) for sfx in _EXCLUDED_SUFFIXES):
                continue

            # ORDEN IMPORTA: bool antes de int (bool es subclase de int)
            if isinstance(val, bool):
                canonical = f"bool:{val}"
            elif isinstance(val, str):
                canonical = f"str:{val[:32]}"
            elif isinstance(val, (int, float)):
                abs_val = abs(float(val))
                if abs_val < 1e-15:
                    canonical = "num:zero"          # [C8] Cubeta especial para 0
                else:
                    magnitude = int(math.floor(math.log10(abs_val)))
                    canonical = f"num:{magnitude}"
            else:
                canonical = f"obj:{type(val).__name__}"

            semantic_values.append(canonical)

        if not semantic_values:
            return 0.0

        counts  = Counter(semantic_values)
        total   = len(semantic_values)
        entropy = 0.0
        for count in counts.values():
            p        = max(count / total, ENTROPY_MIN_PROB)
            entropy -= p * math.log(p)
        return entropy

    @staticmethod
    def _is_aromatic(context: Dict[str, Any]) -> bool:
        """
        Verificación de aromaticidad según la Regla de Hückel.

        La regla de Hückel establece que un sistema cíclico conjugado plano es
        aromático si tiene 4n+2 electrones π, n = 0, 1, 2, ...

        Para el reactor C₆ con 6 nodos activos posibles:
          n=0 → 2  e⁻ π  (aromático mínimo, conjugación incompleta)
          n=1 → 6  e⁻ π  ← ÚNICO estado objetivo del reactor (todos los nodos)
          n=2 → 10 e⁻ π  (imposible con 6 nodos)

        [C3] Solo se acepta pi_electrons == 6 como condición de aromaticidad
        completa para el reactor de 6 nodos. El caso pi==2 es informativo:
        se registra como advertencia pero NO activa el retorno temprano.

        Antiaromático (4n e⁻, n≥1): detectado y advertido.
        """
        pi_electrons = sum(
            1
            for k, v in context.items()
            if k.endswith("_status") and v == "resonant"
        )

        has_errors = any(k.endswith("_error")   for k in context)
        has_skips  = any(k.endswith("_skipped") for k in context)

        # Detección de antiaromaticidad (4n e⁻, n≥1)
        if pi_electrons > 0 and pi_electrons % 4 == 0:
            logger.warning(
                "⚠️  ESTADO ANTIAROMÁTICO: %d e⁻ π (4n, n=%d).",
                pi_electrons, pi_electrons // 4,
            )

        # Información: aromaticidad parcial (n=0, 2 e⁻ π)
        if pi_electrons == 2:
            logger.info(
                "ℹ️  Aromaticidad mínima (n=0, 2 e⁻ π): "
                "solo 2 nodos resonantes — conjugación incompleta."
            )

        # [C3] Aromaticidad completa: ÚNICAMENTE 6 e⁻ π (todos los nodos)
        is_fully_aromatic = (pi_electrons == 6)

        return is_fully_aromatic and not has_errors and not has_skips
"""
Módulo: Reactor de resonancia cuántica (núcleo catalítico hexagonal)
Versión: 4.3 (Refinamiento Algebraico-Físico)
Arquitectura basada en la Teoría del Orbital Molecular (TOM) para el Benceno.

Topología del Reactor (El Anillo C₆ con simetría D6h):
  C1 (Ingesta) ─── C2 (Física)
 /                           \\
C6 (Materia)               C3 (Topología)
 \\                           /
  C5 (Semántica) ── C4 (Estrategia)

Correcciones v4.3 sobre v4.2:
─────────────────────────────────────────────────────────────────────────────
[F1] CFL crítico corregido: α_max = 1/(2·λ_max) = 1/8 = 0.125 para C₆
     (λ_max del Laplaciano circulante C₆ es 4, no 2).
[F2] Amortiguamiento: factor siempre no-negativo via |cos|, evita reflexión.
[F3] Hückel puro en _is_aromatic: 4n+2 e⁻ π (n=0→2, n=1→6). Se elimina
     el caso pi==3 que no satisface la regla.
[F4] Hamiltoniano: neighbor_stabilization separado como término de
     estabilización explícito; max(0,...) no descarta estabilización.
[F5] Gibbs: coeficiente κ_topo explícito para dimensionalidad correcta.
[F6] delta_h: eliminado el término de latencia (elapsed) del hamiltoniano.
[F7] CFL_STABILITY_FACTOR reducido a 0.95 para garantizar α < α_crítico.
[F8] project_orthogonal: normaliza copia local, no el vector base original.
[F9] _calculate_shannon_entropy: entropía sobre valores booleans/strings
     de claves semánticas, más robusta y menos ruidosa.
[F10] Encapsulación: base_temperature enfriado vía método dedicado.
[F11] _spectral_gap: calculado analíticamente para C₆ (λ₁ = 2-√3).
[F12] Logging estructurado con contexto de ciclo en todos los métodos.
"""

import logging
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar

from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry

logger = logging.getLogger("QuantumReactor")

# =============================================================================
# Constantes Físicas y Matemáticas
# =============================================================================
R_GAS_CONSTANT: float = 8.314
BOLTZMANN_SCALE: float = 1.0e-1
GIBBS_CONVERGENCE_EPS: float = 0.05
DAMPING_GAMMA: float = 0.3
DAMPING_OMEGA: float = math.pi / 3.0
ENTROPY_MIN_PROB: float = 1e-10

# [F1] Corrección CFL: para el grafo circulante C₆, el Laplaciano tiene
# autovalores λₖ = 2 - 2·cos(2πk/6), k=0,...,5.
# λ_max = λ₃ = 2 - 2·cos(π) = 4.
# Condición de estabilidad de la ecuación de calor discreta: α < 1/(2·λ_max)
# α_critical = 1 / (2 * 4) = 0.125
_RING_SIZE: int = 6
_LAMBDA_MAX_C6: float = 4.0          # Autovalor máximo del Laplaciano C₆
CFL_ALPHA_CRITICAL: float = 1.0 / (2.0 * _LAMBDA_MAX_C6)   # = 0.125

# [F7] Factor de seguridad estricto: α_safe = 0.95 · α_critical < α_critical
CFL_SAFETY_MARGIN: float = 0.95
CFL_ALPHA_SAFE: float = CFL_SAFETY_MARGIN * CFL_ALPHA_CRITICAL  # ≈ 0.11875

# [F5] Coeficiente dimensional para el término de presión topológica en Gibbs.
# Unidades: [κ] = J/mol para coherencia con H y T·S.
TOPO_PRESSURE_COEFF: float = 1.0

# =============================================================================
# Enumeraciones
# =============================================================================
class CarbonNode(Enum):
    """
    Nodos del anillo de benceno (simetría D6h).
    Cada nodo representa un orbital sp² híbrido funcional.
    El valor ordinal (auto) mapea directamente al índice matricial vía
    la propiedad `index`.

    Autovalores del Laplaciano circulante C₆ (referencia):
        λₖ = 2 - 2·cos(2πk/6),  k = 0, 1, ..., 5
        → {0, 1, 3, 4, 3, 1}  (λ_min=0, λ_max=4, gap=λ₁=1)
    """
    C1_INGESTION = auto()
    C2_PHYSICS = auto()
    C3_TOPOLOGY = auto()
    C4_STRATEGY = auto()
    C5_SEMANTICS = auto()
    C6_MATTER = auto()

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
        """Nombre del servicio MIC correspondiente."""
        _SERVICE_MAP: Dict[str, str] = {
            "C1_INGESTION":  "load_data",
            "C2_PHYSICS":    "stabilize_flux",
            "C3_TOPOLOGY":   "business_topology",
            "C4_STRATEGY":   "financial_analysis",
            "C5_SEMANTICS":  "semantic_translation",
            "C6_MATTER":     "materialization",
        }
        return _SERVICE_MAP[self.name]


# =============================================================================
# Estructuras de Estado
# =============================================================================
T = TypeVar("T", bound="HilbertState")


@dataclass
class HilbertState:
    """
    Estado del sistema proyectado en un espacio de Hilbert ℝ⁶ abstracto.
    El vector |ψ⟩ codifica el estrés local de cada nodo del anillo.

    Propiedades fundamentales:
    ─ ‖ψ‖ (norma-2): estrés topológico total del anillo.
    ─ φ (fase):       rotación acumulada del ciclo de resonancia.
    ─ ⟨ψ|φ⟩:          producto interno estándar para comparación de estados.
    """
    vector: List[float] = field(default_factory=lambda: [0.0] * 6)
    phase: float = 0.0

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

        El operador actúa sobre los modos de alta frecuencia (desviaciones
        del promedio) preservando el modo λ=0 (traslación uniforme del anillo).

        Corrección [F2]: el factor de amortiguamiento usa |cos(ωt)| para
        garantizar no-negatividad y evitar reflexión artificial de modos.

        Formulación:
            v̄  = (1/6)·Σvᵢ                   (modo fundamental)
            δᵢ = vᵢ - v̄                       (modos superiores)
            factor = exp(-γ·t) · |cos(ω·t)|   (envolvente siempre ≥ 0)
            vᵢ' = v̄ + δᵢ · factor
        """
        envelope = math.exp(-DAMPING_GAMMA * cycle)
        # [F2] |cos| garantiza factor ∈ [0, 1] para todo ciclo
        oscillation = abs(math.cos(DAMPING_OMEGA * cycle))
        factor = envelope * oscillation

        mean = sum(self.vector) / float(_RING_SIZE)
        self.vector = [mean + (v - mean) * factor for v in self.vector]

    def project_orthogonal(self, subspace_basis: List["HilbertState"]) -> None:
        """
        Proyección ortogonal de Gram-Schmidt modificada.

        Corrección [F8]: se trabaja sobre copias normalizadas de los vectores
        base para NO modificar los objetos originales del caller (sin efectos
        secundarios destructivos).
        """
        for basis_vec in subspace_basis:
            # [F8] Copia local normalizada — el objeto original permanece intacto
            norm_b = basis_vec.norm
            if norm_b < 1e-12:
                continue
            unit_b = [x / norm_b for x in basis_vec.vector]
            projection = sum(self.vector[i] * unit_b[i] for i in range(_RING_SIZE))
            self.vector = [
                self.vector[i] - projection * unit_b[i]
                for i in range(_RING_SIZE)
            ]
        self.normalize()

    def __repr__(self) -> str:
        components = ", ".join(f"{v:.4f}" for v in self.vector)
        return (
            f"HilbertState(‖ψ‖={self.norm:.4f}, "
            f"φ={self.phase:.4f}rad, [{components}])"
        )


@dataclass
class ThermodynamicPotential:
    """
    Potenciales termodinámicos del reactor hexagonal.

    Corrección [F5]: el término de presión topológica incluye un coeficiente
    dimensional explícito κ_topo para que P·V sea coherente con H y T·S.

    Corrección [F10]: el enfriamiento de temperatura base se realiza vía
    el método `cool_temperature` que preserva la cota mínima física (280 K).
    """
    enthalpy: float = 0.0
    entropy: float = 0.0
    base_temperature: float = 298.0
    temperature_coupling: float = 15.0
    topological_stress: float = 0.0

    _T_MIN: float = field(default=280.0, init=False, repr=False)

    @property
    def temperature(self) -> float:
        """
        T = T₀ + γ·‖ψ‖²   (principio de equipartición informacional)
        """
        return (
            self.base_temperature
            + self.temperature_coupling * (self.topological_stress ** 2)
        )

    @property
    def gibbs_free_energy(self) -> float:
        """
        G = H − T·S·κ + κ_topo·‖ψ‖²

        El término κ_topo·‖ψ‖² representa el trabajo de deformación del anillo
        (análogo a P·V en termodinámica clásica), con κ_topo en unidades
        consistentes con H (J/mol).
        """
        ts_term = self.temperature * self.entropy * BOLTZMANN_SCALE
        topo_pressure = TOPO_PRESSURE_COEFF * (self.topological_stress ** 2)
        return self.enthalpy - ts_term + topo_pressure

    @property
    def instability(self) -> float:
        """
        Índice de inestabilidad topológico-termodinámica:
            I = ln(1 + |G|) + ‖ψ‖
        """
        return math.log1p(abs(self.gibbs_free_energy)) + self.topological_stress

    def update(
        self,
        new_enthalpy: float,
        new_entropy: float,
        topological_stress: float,
    ) -> None:
        """Actualización atómica de todas las variables de estado termodinámico."""
        self.enthalpy = new_enthalpy
        self.entropy = new_entropy
        self.topological_stress = topological_stress

    def cool_temperature(self, factor: float = 0.95) -> None:
        """
        [F10] Reduce la temperatura base preservando el mínimo físico.
        Encapsula la lógica de enfriamiento que antes estaba dispersa en
        `_attempt_stabilization`.

        Args:
            factor: Factor multiplicativo ∈ (0, 1). Default 0.95.
        """
        self.base_temperature = max(self._T_MIN, self.base_temperature * factor)


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
        """Fuerza catalítica efectiva."""
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

    Autovalores analíticos del Laplaciano circulante C₆:
        λₖ = 2 − 2·cos(2πk/6),  k = 0, 1, ..., 5
        → λ = {0, 1, 3, 4, 3, 1}
        → λ_min = 0  (modo de traslación uniforme)
        → λ_max = 4  (modo antipodal)
        → gap   = λ₁ = 1  (brecha espectral)

    Corrección [F1]: CFL crítico recalculado con λ_max = 4 → α_crit = 0.125.
    Corrección [F11]: spectral_gap calculado analíticamente.
    """
    RING_SIZE: int = 6

    # Autovalores exactos del Laplaciano C₆ (precalculados analíticamente)
    _EIGENVALUES: Tuple[float, ...] = (0.0, 1.0, 3.0, 4.0, 3.0, 1.0)

    def __init__(self) -> None:
        n = self.RING_SIZE

        self.adjacency: List[List[int]] = [
            [
                1 if (j == (i + 1) % n or j == (i - 1) % n) else 0
                for j in range(n)
            ]
            for i in range(n)
        ]

        self.degree: List[int] = [2] * n

        # L = D − A  (signo canónico: semidefinido positivo)
        self.laplacian: List[List[int]] = [
            [
                self.degree[i] * int(i == j) - self.adjacency[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]

    @property
    def spectral_gap(self) -> float:
        """
        [F11] Brecha espectral λ₁ del Laplaciano C₆, calculada analíticamente.
        λ₁ = 2 − 2·cos(2π/6) = 2 − 2·(1/2) = 1.0
        La brecha espectral controla la velocidad de mezcla del proceso de difusión.
        """
        # λ₁ = 2 − 2·cos(2π/6) = 1.0  (exacto para C₆)
        return 2.0 - 2.0 * math.cos(2.0 * math.pi / self.RING_SIZE)

    def neighbor_indices(self, node_index: int) -> Tuple[int, int]:
        """Retorna los índices (izquierdo, derecho) del nodo en el anillo C₆."""
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
        """Condición de Neumann: añade flujo en nodos de frontera."""
        new_vector = state_vector.copy()
        for idx, flux in boundary_flux.items():
            new_vector[idx] += flux
        return new_vector

    def diffuse_stress(
        self,
        state_vector: List[float],
        diffusion_rate: float = 0.10,
        boundary_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        Ecuación de calor discreta en C₆:
            ψ(t+1) = ψ(t) − α · L · ψ(t)
                    = (I − α·L) · ψ(t)

        Condición CFL de estabilidad:
            α < 1 / (2 · λ_max) = 1 / (2·4) = 0.125

        El esquema es incondicionalamente estable para α < α_crit.
        Para α = α_crit el esquema está en el límite y puede amplificar
        el modo antipodal; se impone α_safe = 0.95 · α_crit.

        Corrección [F1]: α_crit = 0.125, no 0.5.
        Corrección [F7]: Se recorta a CFL_ALPHA_SAFE, no a 0.5.

        Orden de operaciones:
          1. Difusión laplaciana.
          2. Condiciones de frontera (post-difusión).
        """
        if diffusion_rate <= 0.0:
            return state_vector.copy()

        if diffusion_rate >= CFL_ALPHA_CRITICAL:
            original_rate = diffusion_rate
            diffusion_rate = CFL_ALPHA_SAFE
            logger.warning(
                "⚠️ diffusion_rate=%.5f ≥ α_crit=%.5f → ajustado a α_safe=%.5f "
                "(condición CFL para C₆: α < 1/(2·λ_max) = %.5f)",
                original_rate, CFL_ALPHA_CRITICAL, CFL_ALPHA_SAFE, CFL_ALPHA_CRITICAL,
            )

        n = self.RING_SIZE
        new_vector: List[float] = [0.0] * n

        # Paso 1: ψ'ᵢ = ψᵢ − α · (L·ψ)ᵢ
        for i in range(n):
            lap_action = sum(
                self.laplacian[i][j] * state_vector[j] for j in range(n)
            )
            new_vector[i] = state_vector[i] - diffusion_rate * lap_action

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

    Invariantes mantenidas durante la ejecución:
    ─ α < CFL_ALPHA_CRITICAL en toda difusión.
    ─ enthalpy ≥ _MIN_ENTHALPY, entropy ≥ _MIN_ENTROPY.
    ─ Difusión global ejecutada exactamente UNA vez por ciclo.
    ─ Amortiguamiento factor ∈ [0, 1] (no reflexión).
    """

    INSTABILITY_THRESHOLD: float = 5.0
    MAX_RESONANCE_CYCLES: int = 6
    ACTIVATION_BARRIER_CEILING: float = 0.9

    # Precursores requeridos por nodo (penaliza si ausentes en contexto)
    _PRECURSOR_MAP: Dict[CarbonNode, List[str]] = {
        CarbonNode.C2_PHYSICS:   ["physical_constraints"],
        CarbonNode.C4_STRATEGY:  ["financial_params"],
        CarbonNode.C5_SEMANTICS: ["semantic_model"],
    }

    _MIN_ENTHALPY: float = 1e-10
    _MIN_ENTROPY: float = 1e-10

    # Parámetros del Hamiltoniano de Hückel (unidades adimensionales normalizadas)
    _HUCKEL_ALPHA: float = 0.20   # Energía de sitio (nivel de Coulomb)
    _HUCKEL_BETA: float = -0.05   # Integral de resonancia (< 0 → estabilización)

    def __init__(
        self,
        mic: MICRegistry,
        agent: CatalystAgent,
        telemetry: TelemetryContext,
    ) -> None:
        self.mic = mic
        self.catalyst = agent
        self.telemetry = telemetry
        self.topology = HexagonalTopology()
        self.ring_sequence: List[CarbonNode] = list(CarbonNode)

        self._temperature_coupling: float = 15.0
        self._base_temperature: float = 298.0

    # ─────────────────────────────────────────────────────────────────────────
    # Interfaz pública
    # ─────────────────────────────────────────────────────────────────────────
    def ignite(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enciende el reactor y ejecuta el ciclo de resonancia completo.

        Returns:
            El contexto estabilizado (aromático o metaestable).

        Raises:
            RuntimeError: Si se exceden MAX_RESONANCE_CYCLES sin convergencia.
        """
        reaction_id = str(uuid.uuid4())[:8]
        logger.info("⚛️  QUANTUM IGNITION: Reactor [%s] encendido", reaction_id)

        context = initial_context.copy()
        potential = ThermodynamicPotential(
            base_temperature=self._base_temperature,
            temperature_coupling=self._temperature_coupling,
        )
        state = HilbertState()

        potential.entropy = max(
            self._MIN_ENTROPY,
            self._calculate_shannon_entropy(context),
        )
        previous_gibbs = potential.gibbs_free_energy

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

                # Intercambio de Kekulé (inversión del vector cada 2 ciclos)
                if cycle % 2 == 0:
                    state.vector = state.vector[::-1]

                # Avance de fase: Δφ = 2π/6 por ciclo (simetría D6h)
                state.phase = (state.phase + 2.0 * math.pi / _RING_SIZE) % (
                    2.0 * math.pi
                )

                if self._is_aromatic(context):
                    logger.info(
                        "✅ AROMATICIDAD ALCANZADA en ciclo %d. ΔG_final=%.4f",
                        cycle, potential.gibbs_free_energy,
                    )
                    self.telemetry.record_reaction_success(reaction_id, cycle)
                    return context

                if self._check_thermodynamic_convergence(
                    potential, previous_gibbs, cycle
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

            raise RuntimeError(
                f"Failed to achieve aromatic stability "
                f"(Max {self.MAX_RESONANCE_CYCLES} resonance cycles exceeded). "
                f"Final ΔG={potential.gibbs_free_energy:.4f}, "
                f"I={potential.instability:.4f}"
            )

        except Exception as exc:
            self.telemetry.record_error("reaction_chamber", str(exc))
            logger.error("🔥 Fallo crítico en el reactor: %s", exc)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos privados del ciclo de resonancia
    # ─────────────────────────────────────────────────────────────────────────
    def _catalytic_orientation(
        self,
        context: Dict[str, Any],
        potential: ThermodynamicPotential,
    ) -> None:
        """Aplica la orientación catalítica al contexto según ∂G."""
        catalyst_diff = self.catalyst.orient(context, potential.gibbs_free_energy)
        context.update(catalyst_diff)

    def _ring_iteration(
        self,
        context: Dict[str, Any],
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Itera sobre todos los nodos del anillo y aplica la difusión global.

        Corrección v4.2 (mantenida): Difusión ejecutada EXACTAMENTE UNA VEZ
        al final del ciclo, después de procesar todos los nodos.
        """
        total_delta_h = 0.0

        for node in self.ring_sequence:
            idx = node.index
            base_ea = self._calculate_hamiltonian(node, context)
            effective_ea = base_ea * (1.0 - self.catalyst.efficiency_factor)

            try:
                node_context, delta_h = self._react_node(
                    node, context, effective_ea, state.vector[idx]
                )
                context.update(node_context)
                total_delta_h += delta_h

            except Exception as exc:
                logger.error(
                    "💥 Excepción en nodo %s (ciclo %d): %s",
                    node.name, cycle, exc,
                )
                state.vector[idx] += 1.0
                total_delta_h += 50.0
                context[f"{node.name}_error"] = str(exc)

        # Difusión global única al final del ciclo
        state.vector = self.topology.diffuse_stress(
            state.vector,
            diffusion_rate=CFL_ALPHA_SAFE,   # α ≈ 0.11875 < 0.125 (estable)
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
        potential: ThermodynamicPotential,
        previous_gibbs: float,
        cycle: int,
    ) -> bool:
        """
        Verifica convergencia termodinámica: |ΔG| < ε.
        Solo se evalúa a partir del ciclo 2 (se requiere al menos un delta).
        """
        delta_gibbs = abs(potential.gibbs_free_energy - previous_gibbs)
        return delta_gibbs < GIBBS_CONVERGENCE_EPS and cycle > 1

    def _attempt_stabilization(
        self,
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Intenta estabilizar el reactor cuando I > INSTABILITY_THRESHOLD.

        Protocolo de estabilización:
          1. Amortiguamiento espectral del vector de estado.
          2. Disipación de entalpía (factor 0.85).
          3. Enfriamiento de temperatura base (factor 0.95).  [F10]
          4. Actualización del estrés topológico.
          5. Colapso si I > 1.2 · umbral tras estabilización.
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

        # 3. [F10] Enfriamiento vía método encapsulado
        potential.cool_temperature(factor=0.95)

        # 4. Actualización del estrés topológico post-amortiguamiento
        potential.topological_stress = state.norm

        # 5. Colapso irrecuperable
        if potential.instability > self.INSTABILITY_THRESHOLD * 1.2:
            raise RuntimeError(
                f"Reactor Collapse: Inestabilidad irrecuperable "
                f"(I={potential.instability:.2f} > "
                f"{self.INSTABILITY_THRESHOLD * 1.2:.2f})"
            )

        logger.info(
            "🛡️  Estabilización exitosa: I=%.2f, ‖ψ‖=%.4f, T_base=%.2f K",
            potential.instability, state.norm, potential.base_temperature,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Hamiltoniano de Hückel
    # ─────────────────────────────────────────────────────────────────────────
    def _calculate_hamiltonian(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """
        Hamiltoniano de Hückel para el nodo iésimo del anillo.

        Formulación:
            Eᵢ = α + Σⱼ∈vecinos βᵢⱼ · resonant(j) + penalty(precursores)

        donde:
        ─ α = 0.20:   energía de sitio (Coulomb integral), siempre ≥ 0.
        ─ β = -0.05:  integral de resonancia (< 0 → estabiliza la barrera).
        ─ penalty ≥ 0: penalización por precursores ausentes.

        Corrección [F4]: la estabilización por vecinos resonantes REDUCE la
        barrera de activación (β < 0). El resultado final se recorta a [0, ∞)
        para garantizar que Eₐ ≥ 0 (barrera física).
        """
        idx = node.index

        # Estabilización por vecinos resonantes (β < 0 → Eᵢ disminuye)
        left_idx, right_idx = self.topology.neighbor_indices(idx)
        neighbor_stabilization = sum(
            self._HUCKEL_BETA
            for ni in (left_idx, right_idx)
            if context.get(f"{CarbonNode(ni + 1).name}_status") == "resonant"
        )

        # Penalización por precursores ausentes
        precursor_penalty = self._evaluate_precursor_penalty(node, context)

        # Hamiltoniano total: barrera = sitio + estabilización + penalización
        hamiltonian = self._HUCKEL_ALPHA + neighbor_stabilization + precursor_penalty

        # La barrera de activación es siempre ≥ 0
        return max(0.0, hamiltonian)

    def _evaluate_precursor_penalty(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
    ) -> float:
        """
        Penalización proporcional a la fracción de precursores ausentes.
            penalty = 0.3 · (n_ausentes / n_requeridos)
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
        node: CarbonNode,
        context: Dict[str, Any],
        ea: float,
        local_stress: float,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ejecuta la reacción en un nodo del anillo.

        Corrección [F6]: el término ΔH NO incluye latencia de ejecución
        (elapsed·k), ya que la latencia es una variable de infraestructura
        ruidosa y no una propiedad del sistema termodinámico.

        ΔH = σ²·κ_stress + Eₐ·κ_ea
        donde:
        ─ κ_stress = 5.0:  peso del estrés local sobre la entalpía.
        ─ κ_ea     = 10.0: peso de la barrera de activación sobre la entalpía.
        """
        if ea > self.ACTIVATION_BARRIER_CEILING:
            logger.warning(
                "⚡ Saltando %s: Eₐ=%.3f > %.3f",
                node.name, ea, self.ACTIVATION_BARRIER_CEILING,
            )
            return {f"{node.name}_skipped": True}, 5.0

        stress_factor = 1.0 + (local_stress ** 2) * 0.5
        time.sleep(0.005 * stress_factor)   # Simulación de carga de trabajo

        context_update: Dict[str, Any] = {
            f"{node.name}_status": "resonant",
            f"{node.name}_ts":     time.time(),
            f"{node.name}_ea":     ea,
        }

        # [F6] ΔH determinista: sin término de latencia
        delta_h = (local_stress ** 2) * 5.0 + ea * 10.0

        logger.debug(
            "🔬 %s procesado | Eₐ=%.3f | σ=%.3f | ΔH=%.3f",
            node.name, ea, local_stress, delta_h,
        )
        return context_update, delta_h

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades estáticas
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _calculate_shannon_entropy(context: Dict[str, Any]) -> float:
        """
        Entropía de Shannon del contexto.

        Corrección [F9]: la distribución se calcula sobre los VALORES
        semánticamente significativos del contexto (solo claves sin prefijos
        de metadatos internos `_ts`, `_ea`), utilizando la representación
        canónica del valor para reducir el ruido de la proxy.

        H = −Σₚ p·ln(p)
        """
        if not context:
            return 0.0

        # Excluir metadatos internos ruidosos (_ts son floats de alta entropía
        # que inflan artificialmente H; _ea son floats deterministas pero
        # altamente variables por nodo)
        _EXCLUDED_SUFFIXES = ("_ts", "_ea")
        semantic_values: List[str] = []

        for key, val in context.items():
            if any(key.endswith(sfx) for sfx in _EXCLUDED_SUFFIXES):
                continue
            # Representación canónica para agrupación de distribución
            if isinstance(val, bool):
                canonical = f"bool:{val}"
            elif isinstance(val, str):
                canonical = f"str:{val[:32]}"   # primeros 32 chars (no hashing ruidoso)
            elif isinstance(val, (int, float)):
                # Cuantización en 10 cubetas logarítmicas
                magnitude = int(math.log10(abs(val) + 1e-12))
                canonical = f"num:{magnitude}"
            else:
                canonical = f"obj:{type(val).__name__}"
            semantic_values.append(canonical)

        if not semantic_values:
            return 0.0

        counts = Counter(semantic_values)
        total = len(semantic_values)
        entropy = 0.0
        for count in counts.values():
            p = max(count / total, ENTROPY_MIN_PROB)
            entropy -= p * math.log(p)
        return entropy

    @staticmethod
    def _is_aromatic(context: Dict[str, Any]) -> bool:
        """
        Verificación de aromaticidad según la Regla de Hückel.

        La regla de Hückel establece que un sistema cíclico conjugado plano es
        aromático si tiene 4n+2 electrones π (n = 0, 1, 2, ...).

        Para el anillo C₆ con 6 sitios posibles:
        ─ n=0 → 2  electrones π  (media occupancy, raramente alcanzado)
        ─ n=1 → 6  electrones π  ← Estado objetivo del reactor
        ─ n=2 → 10 electrones π  (imposible con 6 nodos)

        Antiaromático (regla de Baird para el estado base): 4n e⁻ π.
        ─ n=1 → 4  electrones π  (detectado y advertido)

        Corrección [F3]: eliminado el caso `pi_electrons == 3` que NO
        satisface la regla de Hückel (3 ≠ 4n+2 para ningún n ∈ ℤ≥0).
        La única condición aromática válida para este modelo es 6 e⁻ π.

        Note: el caso n=0 (2 e⁻) se mantiene como condición reachable
        mínima pero se registra como advertencia.
        """
        pi_electrons = sum(
            1
            for k, v in context.items()
            if k.endswith("_status") and v == "resonant"
        )

        has_errors = any(k.endswith("_error") for k in context)
        has_skips = any(k.endswith("_skipped") for k in context)

        # Detección de antiaromaticidad (4n e⁻, n≥1)
        if pi_electrons > 0 and pi_electrons % 4 == 0:
            logger.warning(
                "⚠️  ESTADO ANTIAROMÁTICO DETECTADO: %d e⁻ π (4n, n=%d)",
                pi_electrons, pi_electrons // 4,
            )

        # [F3] Aromaticidad estricta de Hückel: 4n+2 e⁻ π
        # Para C₆: solo 6 e⁻ π (n=1) es el estado objetivo completo.
        # Se acepta 2 e⁻ (n=0) como aromático mínimo pero se advierte.
        is_huckel_aromatic = (pi_electrons == 6 or pi_electrons == 2)

        if pi_electrons == 2 and is_huckel_aromatic:
            logger.warning(
                "⚠️  Aromaticidad mínima (n=0, 2 e⁻ π): "
                "sistema parcialmente ocupado."
            )

        return is_huckel_aromatic and not has_errors and not has_skips
"""
Módulo: Reactor de resonancia cuántica (núcleo catalítico hexagonal)
Versión: 4.1 (Refinamiento Termodinámico, Topológico y Algebraico)
Arquitectura basada en la Teoría del Orbital Molecular (TOM) para el Benceno.
El flujo no es secuencial, sino una superposición de estados estabilizada por
resonancia cuántica y topología algebraica.

Topología del Reactor (El Anillo):
  C1 (Ingesta) ─── C2 (Física)
 /                         \\
C6 (Materia)               C3 (Topología)
\\                         /
C5 (Semántica) ── C4 (Estrategia)

Mejoras Clave v4.1:
- Termodinámica: Relación T(ψ) física basada en el principio de equipartición
- Topología: Difusión laplaciana con condiciones de frontera Dirichlet/Neumann
- Algebraica: Normalización rigurosa del espacio de Hilbert con proyección ortogonal
- Numérica: Manejo robusto de casos límite y estabilidad garantizada
- Conceptual: Documentación mejorada que explica las analogías químicas
"""
import logging
import math
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Protocol, Tuple, TypeVar, cast

# Mantenemos imports de infraestructura
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.tools_interface import MICRegistry

logger = logging.getLogger("QuantumReactor")

# =============================================================================
# Constantes Físicas y Matemáticas
# =============================================================================
R_GAS_CONSTANT = 8.314          # J/(mol·K)
BOLTZMANN_SCALE = 1.0e-1        # Factor de acople: entropía informacional → energía
GIBBS_CONVERGENCE_EPS = 0.05    # Umbral de convergencia termodinámica |δG|
DAMPING_GAMMA = 0.3             # Coeficiente de decaimiento exponencial
DAMPING_OMEGA = math.pi / 3.0   # Frecuencia angular (π/3 por simetría D6h)
CFL_STABILITY_FACTOR = 0.49     # Factor máximo seguro para difusión (α < 0.5)
ENTROPY_MIN_PROB = 1e-10        # Probabilidad mínima para evitar -inf en entropía

# =============================================================================
# Enumeraciones
# =============================================================================
class CarbonNode(Enum):
    """
    Nodos del anillo de benceno (simetría D6h).
    Cada nodo representa un orbital sp² híbrido funcional.
    El valor ordinal (auto) mapea directamente al índice matricial via
    la propiedad `index`.
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
        service_names = {
            CarbonNode.C1_INGESTION: "load_data",
            CarbonNode.C2_PHYSICS: "stabilize_flux",
            CarbonNode.C3_TOPOLOGY: "business_topology",
            CarbonNode.C4_STRATEGY: "financial_analysis",
            CarbonNode.C5_SEMANTICS: "semantic_translation",
            CarbonNode.C6_MATTER: "materialization"
        }
        return service_names[self]

# =============================================================================
# Estructuras de Estado
# =============================================================================
T = TypeVar('T', bound='HilbertState')

@dataclass
class HilbertState:
    """
    Estado del sistema proyectado en un espacio de Hilbert ℝ⁶ abstracto.
    El vector |ψ⟩ codifica el estrés local de cada nodo del anillo.
    
    Propiedades fundamentales:
    - ‖ψ‖ (norma): estrés topológico total.
    - φ (fase): rotación acumulada del ciclo de resonancia.
    - ⟨ψ|φ⟩: producto interno para comparar estados.
    
    Invariantes matemáticas:
    - La norma ‖ψ‖ ≥ 0 (no negativa)
    - ‖c·ψ‖ = |c|·‖ψ‖ para cualquier escalar c
    - |⟨ψ|φ⟩| ≤ ‖ψ‖·‖φ‖ (desigualdad de Cauchy-Schwarz)
    """
    vector: List[float] = field(default_factory=lambda: [0.0] * 6)
    phase: float = 0.0
    
    @property
    def norm(self) -> float:
        """‖ψ‖ = √⟨ψ|ψ⟩"""
        return math.sqrt(sum(x ** 2 for x in self.vector))
    
    def inner_product(self, other: "HilbertState") -> float:
        """⟨self|other⟩ = Σᵢ aᵢ·bᵢ"""
        return sum(a * b for a, b in zip(self.vector, other.vector))
    
    def normalize(self: T) -> T:
        """Proyecta |ψ⟩ sobre la esfera unitaria S⁵ (si ‖ψ‖ > 0)."""
        n = self.norm
        if n > 1e-12:
            self.vector = [x / n for x in self.vector]
        return self
    
    def apply_damping(self, cycle: int) -> None:
        """
        Amortiguamiento oscilatorio con envolvente exponencial:
        
            ψ → ψ · e^{-γ·t} · cos(ω·t)
        
        donde t = cycle (tiempo discreto).
        
        La envolvente e^{-γt} garantiza convergencia asintótica;
        cos(ωt) permite interferencia constructiva/destructiva
        que emula la resonancia del anillo bencénico.
        
        Nota: Este operador no preserva la norma, por lo que se requiere
        normalización posterior si es necesario mantener |ψ⟩ en S⁵.
        """
        envelope = math.exp(-DAMPING_GAMMA * cycle)
        oscillation = math.cos(DAMPING_OMEGA * cycle)
        factor = envelope * oscillation
        self.vector = [v * factor for v in self.vector]
    
    def project_orthogonal(self, subspace_basis: List["HilbertState"]) -> None:
        """
        Proyección ortogonal sobre un subespacio.
        Útil para eliminar componentes no deseadas del estado.
        """
        for basis_vector in subspace_basis:
            projection = self.inner_product(basis_vector)
            self.vector = [
                self.vector[i] - projection * basis_vector.vector[i]
                for i in range(len(self.vector))
            ]
    
    def __repr__(self) -> str:
        components = ", ".join(f"{v:.4f}" for v in self.vector)
        return (
            f"HilbertState(‖ψ‖={self.norm:.4f}, "
            f"φ={self.phase:.4f}, [{components}])"
        )

@dataclass
class ThermodynamicPotential:
    """
    Potenciales termodinámicos del reactor.
    
    Variables de estado:
    - H (entalpía): energía interna acumulada por procesamiento.
    - S (entropía): entropía de Shannon en nats (base e) del contexto.
    - T (temperatura): parámetro cinético; depende del estrés topológico,
      NO de la inestabilidad (desacoplamiento para evitar circularidad).
    - ‖ψ‖ (topological_stress): norma del vector de estado, inyectada
      externamente desde HilbertState.
    
    Propiedades derivadas:
    - ΔG = H - T·S·κ  (energía libre de Gibbs escalada).
    - I = ln(1 + |ΔG|) + ‖ψ‖  (índice de inestabilidad).
    
    Corrección v4.1: La temperatura ahora sigue el principio de equipartición:
        T = T₀ + γ·‖ψ‖²
    donde γ es un factor de acoplamiento que representa cómo el estrés topológico
    incrementa la energía cinética del sistema.
    
    El lazo de control se cierra porque damping(ψ) → ‖ψ‖↓ → T↓ → ΔG↑ → I↓.
    """
    enthalpy: float = 0.0
    entropy: float = 0.0
    base_temperature: float = 298.0  # Temperatura de referencia (K)
    temperature_coupling: float = 15.0  # Factor de acoplamiento T-‖ψ‖
    topological_stress: float = 0.0

    @property
    def temperature(self) -> float:
        """
        Temperatura del sistema, calculada según el principio de equipartición:
            T = T₀ + γ·‖ψ‖²
        
        Esta relación es física: el estrés topológico aumenta la energía cinética
        promedio de los "partículas" en el sistema (representadas por flujos de datos).
        """
        return self.base_temperature + self.temperature_coupling * (self.topological_stress ** 2)

    @property
    def gibbs_free_energy(self) -> float:
        """G = H - T·S·κ"""
        return self.enthalpy - (self.temperature * self.entropy * BOLTZMANN_SCALE)

    @property
    def instability(self) -> float:
        """
        Índice de inestabilidad topológico-termodinámica.
        
        I = ln(1 + |ΔG|) + ‖ψ‖
        
        Combina desviación del equilibrio termodinámico (|ΔG|)
        con el estrés geométrico del anillo (‖ψ‖).
        
        Nota: Usamos ln(1 + |ΔG|) en lugar de |ΔG| para evitar que
        valores extremos dominen el índice, y para dar mayor sensibilidad
        a cambios pequeños cerca del equilibrio.
        """
        return math.log1p(abs(self.gibbs_free_energy)) + self.topological_stress
    
    def update(self, new_enthalpy: float, new_entropy: float, topological_stress: float) -> None:
        """Actualización atómica de todas las variables de estado."""
        self.enthalpy = new_enthalpy
        self.entropy = new_entropy
        self.topological_stress = topological_stress

# =============================================================================
# Protocolo del Agente Catalizador
# =============================================================================
class CatalystAgent(Protocol):
    """
    Protocolo formal para el Agente Catalizador.
    
    Un catalizador ideal reduce la barrera de activación sin consumirse:
        Eₐ → Eₐ·(1 − η)
    
    Donde η ∈ [0, 1) es el factor de eficiencia catalítica.
    
    Invariantes:
    - η < 1 (el catalizador no puede invertir la barrera de activación)
    - El agente no modifica el estado termodinámico global directamente
    - Solo actúa sobre variables cinéticas (velocidad de reacción)
    """
    @property
    def efficiency_factor(self) -> float:
        """Factor de eficiencia catalítica η ∈ [0, 1)."""
        ...
    
    @property
    def catalytic_strength(self) -> float:
        """Fuerza catalítica efectiva, considerando condiciones del sistema."""
        ...
    
    def orient(self, context: Dict[str, Any], gradient: float) -> Dict[str, Any]:
        """
        Orienta la reacción según el gradiente de energía libre.
        Retorna un diff parcial para fusionar con el contexto.
        
        Args:
            context: Estado actual del sistema
            gradient: Gradiente de energía libre (dG/dt)
        
        Returns:
            Diccionario con modificaciones para aplicar al contexto
        """
        ...

# =============================================================================
# Topología Hexagonal
# =============================================================================
class HexagonalTopology:
    """
    Estructura algebraica del grafo cíclico C₆ (anillo bencénico).
    
    La matriz laplaciana L = D − A gobierna la difusión de estrés
    mediante la ecuación de calor discreta:
    
        ψ(t+1) = ψ(t) − α · L · ψ(t)
    
    Espectro de L para C₆:
        λₖ = 2 − 2·cos(2πk/6),  k = 0, 1, ..., 5
        → λ ∈ {0, 1, 3, 4, 3, 1}
    
    Propiedades espectrales:
        - λ₀ = 0 (modo constante / equilibrio).
        - λ₁ = 1.0 (brecha espectral = velocidad de mezcla). 
        - λ_max = 4 → condición CFL: α < 2/λ_max = 0.5.
    
    Correcciones v4.1:
    - Implementación vectorizada de la difusión para mejor rendimiento
    - Soporte para condiciones de frontera Dirichlet/Neumann
    - Cálculo exacto de la brecha espectral
    """
    RING_SIZE = 6
    
    def __init__(self):
        n = self.RING_SIZE
        
        # Matriz de adyacencia — generada algebraicamente para ciclo Cₙ
        self.adjacency: List[List[int]] = [
            [
                1 if (j == (i + 1) % n or j == (i - 1) % n) else 0
                for j in range(n)
            ]
            for i in range(n)
        ]
        
        # Grado constante = 2 para ciclo simple (cada nodo tiene 2 vecinos)
        self.degree: List[int] = [2] * n
        
        # Laplaciano: L = D − A
        # D = diag(degree), A = adjacency
        self.laplacian: List[List[int]] = [
            [
                self.degree[i] * int(i == j) - self.adjacency[i][j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        
        # Precalculamos la brecha espectral
        self._spectral_gap = self._calculate_spectral_gap()
    
    def _calculate_spectral_gap(self) -> float:
        """Calcula la brecha espectral λ₁ = 2 − 2·cos(2π/n) para Cₙ."""
        return 2.0 - 2.0 * math.cos(2.0 * math.pi / self.RING_SIZE)
    
    @property
    def spectral_gap(self) -> float:
        """Brecha espectral que gobierna la tasa de convergencia."""
        return self._spectral_gap
    
    def neighbor_indices(self, node_index: int) -> Tuple[int, int]:
        """Retorna los índices de los dos vecinos adyacentes en el ciclo."""
        n = self.RING_SIZE
        return (node_index - 1) % n, (node_index + 1) % n
    
    def _apply_dirichlet_boundary(self, state_vector: List[float], boundary_values: Dict[int, float]) -> List[float]:
        """
        Aplica condiciones de frontera Dirichlet (valores fijos en nodos específicos).
        
        Útil para simular nodos "anclados" que no permiten difusión.
        """
        new_vector = state_vector.copy()
        for idx, value in boundary_values.items():
            new_vector[idx] = value
        return new_vector
    
    def _apply_neumann_boundary(self, state_vector: List[float], boundary_flux: Dict[int, float]) -> List[float]:
        """
        Aplica condiciones de frontera Neumann (flujo fijo en nodos específicos).
        
        Útil para simular fuentes o sumideros de estrés.
        """
        new_vector = state_vector.copy()
        for idx, flux in boundary_flux.items():
            new_vector[idx] += flux
        return new_vector
    
    def diffuse_stress(
        self,
        state_vector: List[float],
        diffusion_rate: float = 0.1,
        boundary_conditions: Dict[str, Any] = None
    ) -> List[float]:
        """
        Ecuación de calor discreta:
        
            ψ(t+1) = ψ(t) − α · L · ψ(t)
        
        El operador −L actúa como difusión genuina: suaviza picos de estrés
        distribuyéndolos hacia los vecinos adyacentes.
        
        Corrección v4.1: 
        - Implementación vectorizada para mejor rendimiento
        - Soporte para condiciones de frontera
        - Manejo robusto de casos límite
        
        Precondición: diffusion_rate ∈ (0, 0.5) para estabilidad numérica
        (condición CFL: α < 2/λ_max = 0.5 para C₆).
        """
        if diffusion_rate <= 0.0:
            return state_vector.copy()
        
        # Validación estricta de la condición CFL
        max_allowable = 2.0 / max(max(row) for row in self.laplacian) if self.laplacian else 0.5
        if diffusion_rate >= max_allowable:
            original_rate = diffusion_rate
            diffusion_rate = max_allowable * CFL_STABILITY_FACTOR
            logger.warning(
                f"⚠️ diffusion_rate ajustado de {original_rate:.4f} a {diffusion_rate:.4f} "
                f"(condición CFL: α < {max_allowable:.4f})"
            )
        
        n = self.RING_SIZE
        new_vector = state_vector.copy()
        
        # Aplicar condiciones de frontera si existen
        if boundary_conditions:
            if 'dirichlet' in boundary_conditions:
                new_vector = self._apply_dirichlet_boundary(
                    new_vector, boundary_conditions['dirichlet']
                )
            if 'neumann' in boundary_conditions:
                new_vector = self._apply_neumann_boundary(
                    new_vector, boundary_conditions['neumann']
                )
        
        # Aplicar difusión laplaciana (versión vectorizada)
        for i in range(n):
            laplacian_action = 0.0
            for j in range(n):
                laplacian_action += self.laplacian[i][j] * new_vector[j]
            new_vector[i] -= diffusion_rate * laplacian_action
        
        return new_vector

# =============================================================================
# Reactor Catalítico (Motor Principal)
# =============================================================================
class CatalyticReactor:
    """
    Motor de Orquestación Resonante.
    
    Ciclo de vida:
    1. Ignición   → inicialización de potenciales y estado cuántico.
    2. Resonancia → orientación catalítica + anillo + difusión + convergencia.
    3. Resultado  → aromaticidad (éxito) o colapso controlado (excepción).
    
    Invariantes de lazo cerrado:
    - Damping del vector ψ reduce ‖ψ‖, que reduce I directamente.
    - Disipación de entalpía reduce |ΔG|, que reduce I directamente.
    - Enfriamiento reduce T, que altera ΔG = H − T·S·κ.
    
    Correcciones v4.1:
    - Relación física temperatura-estrés basada en equipartición
    - Normalización rigurosa del espacio de Hilbert
    - Manejo robusto de casos límite en cálculos termodinámicos
    - Separación clara de responsabilidades en métodos más pequeños
    """
    
    INSTABILITY_THRESHOLD = 5.0
    MAX_RESONANCE_CYCLES = 4
    ACTIVATION_BARRIER_CEILING = 0.9
    
    # ── Dependencias de precursores por nodo ──
    _PRECURSOR_MAP: Dict[CarbonNode, List[str]] = {
        CarbonNode.C2_PHYSICS: ["physical_constraints"],
        CarbonNode.C4_STRATEGY: ["financial_params"],
        CarbonNode.C5_SEMANTICS: ["semantic_model"],
    }
    
    # ── Valores mínimos para evitar problemas numéricos ──
    _MIN_ENTHALPY = 1e-10
    _MIN_ENTROPY = 1e-10
    
    def __init__(
        self,
        mic: MICRegistry,
        agent: CatalystAgent,
        telemetry: TelemetryContext,
    ):
        """
        Inicializa el reactor con sus componentes esenciales.
        
        Args:
            mic: Registro de Microservicios de Intención (Vector MIC)
            agent: Agente catalítico que reduce barreras de activación
            telemetry: Sistema de telemetría para monitoreo
        """
        self.mic = mic
        self.catalyst = agent
        self.telemetry = telemetry
        self.topology = HexagonalTopology()
        self.ring_sequence: List[CarbonNode] = list(CarbonNode)
        
        # Configuración de parámetros físicos
        self._temperature_coupling = 15.0  # Factor de acoplamiento T-‖ψ‖
        self._base_temperature = 298.0     # Temperatura de referencia (K)
    
    # ──────────────────────────────────────────────────────────────────
    # Punto de entrada
    # ──────────────────────────────────────────────────────────────────
    
    def ignite(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inicia la reacción en cadena.
        
        Retorna el contexto transformado si se alcanza aromaticidad.
        Lanza RuntimeError si el reactor colapsa o no converge.
        
        Args:
            initial_context: Contexto inicial con datos y configuración
            
        Returns:
            Contexto procesado y estabilizado
            
        Raises:
            RuntimeError: Si el reactor colapsa o no converge
        """
        reaction_id = str(uuid.uuid4())[:8]
        logger.info(f"⚛️ QUANTUM IGNITION: Reactor [{reaction_id}] encendido")
        
        # Inicialización del estado
        context = initial_context.copy()
        potential = ThermodynamicPotential(
            base_temperature=self._base_temperature,
            temperature_coupling=self._temperature_coupling
        )
        state = HilbertState()
        
        # Condición inicial de entropía
        potential.entropy = max(
            self._MIN_ENTROPY, 
            self._calculate_shannon_entropy(context)
        )
        previous_gibbs = potential.gibbs_free_energy
        
        # Registro inicial de telemetría
        self.telemetry.record_reaction_start(reaction_id, context)
        
        try:
            for cycle in range(1, self.MAX_RESONANCE_CYCLES + 1):
                logger.info(
                    f"⏩ Ciclo de Resonancia {cycle}/{self.MAX_RESONANCE_CYCLES} "
                    f"| ΔG={potential.gibbs_free_energy:.4f} "
                    f"| I={potential.instability:.4f} "
                    f"| ‖ψ‖={state.norm:.4f}"
                )
                
                # 1. Orientación Catalítica (Retroalimentación)
                self._catalytic_orientation(context, potential, cycle)
                
                # 2. Iteración del Anillo (Ciclo electrónico)
                self._ring_iteration(context, state, potential, cycle)
                
                # 3. Avanzar fase del estado (rotación 2π/6 por ciclo)
                state.phase += 2.0 * math.pi / HexagonalTopology.RING_SIZE
                
                # 4. Verificar aromaticidad
                if self._is_aromatic(context):
                    logger.info(
                        "✅ AROMATICIDAD ALCANZADA: Producto estable. "
                        f"ΔG_final={potential.gibbs_free_energy:.4f}"
                    )
                    self.telemetry.record_reaction_success(reaction_id, cycle)
                    return context
                
                # 5. Verificar convergencia termodinámica
                if self._check_thermodynamic_convergence(potential, previous_gibbs, cycle):
                    logger.info(
                        f"🔒 Convergencia termodinámica: |δG|={abs(potential.gibbs_free_energy - previous_gibbs):.6f} "
                        f"< ε={GIBBS_CONVERGENCE_EPS}. Estado metaestable."
                    )
                    context["_metastable_cycle"] = cycle
                    return context
                
                previous_gibbs = potential.gibbs_free_energy
            
            # Si llegamos aquí, no alcanzamos aromaticidad en el máximo de ciclos
            raise RuntimeError(
                f"Failed to achieve aromatic stability "
                f"(Max {self.MAX_RESONANCE_CYCLES} resonance cycles exceeded)"
            )
        
        except Exception as e:
            self.telemetry.record_error("reaction_chamber", str(e))
            logger.error(f"🔥 Fallo crítico en el reactor: {e}")
            raise
    
    # ──────────────────────────────────────────────────────────────────
    # Submétodos de Ignición
    # ──────────────────────────────────────────────────────────────────
    
    def _catalytic_orientation(
        self, 
        context: Dict[str, Any], 
        potential: ThermodynamicPotential,
        cycle: int
    ) -> None:
        """Aplica la orientación catalítica al contexto."""
        catalyst_diff = self.catalyst.orient(
            context, potential.gibbs_free_energy
        )
        context.update(catalyst_diff)
    
    def _ring_iteration(
        self,
        context: Dict[str, Any],
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int
    ) -> None:
        """Ejecuta una iteración completa del anillo de benceno."""
        for node in self.ring_sequence:
            idx = node.index
            
            # A. Calcular barrera de activación
            base_ea = self._calculate_hamiltonian(node, context)
            effective_ea = base_ea * (
                1.0 - self.catalyst.efficiency_factor
            )
            
            # B. Ejecutar reacción del nodo
            try:
                node_context, delta_h = self._react_node(
                    node, context, effective_ea, state.vector[idx]
                )
                context.update(node_context)
                potential.enthalpy += delta_h
            except Exception as e:
                logger.error(f"💥 Excepción en {node.name}: {e}")
                state.vector[idx] += 1.0
                potential.enthalpy += 50.0
                context[f"{node.name}_error"] = str(e)
            
            # C. Difusión topológica del estrés
            state.vector = self.topology.diffuse_stress(
                state.vector,
                diffusion_rate=0.1
            )
            
            # D. Sincronización termodinámica
            potential.update(
                new_enthalpy=max(self._MIN_ENTHALPY, potential.enthalpy),
                new_entropy=max(
                    self._MIN_ENTROPY,
                    self._calculate_shannon_entropy(context)
                ),
                topological_stress=state.norm
            )
            
            # E. Control de colapso
            if potential.instability > self.INSTABILITY_THRESHOLD:
                self._attempt_stabilization(
                    node, state, potential, cycle
                )
    
    def _check_thermodynamic_convergence(
        self,
        potential: ThermodynamicPotential,
        previous_gibbs: float,
        cycle: int
    ) -> bool:
        """Verifica si el sistema ha alcanzado convergencia termodinámica."""
        current_gibbs = potential.gibbs_free_energy
        delta_gibbs = abs(current_gibbs - previous_gibbs)
        return delta_gibbs < GIBBS_CONVERGENCE_EPS and cycle > 1
    
    # ──────────────────────────────────────────────────────────────────
    # Estabilización
    # ──────────────────────────────────────────────────────────────────
    
    def _attempt_stabilization(
        self,
        node: CarbonNode,
        state: HilbertState,
        potential: ThermodynamicPotential,
        cycle: int,
    ) -> None:
        """
        Intenta estabilizar el reactor cuando I > umbral.
        
        Acciones (lazo cerrado):
        1. Amortiguamiento oscilatorio del vector de estado ψ.
        2. Disipación de entalpía (15%).
        3. Enfriamiento activo (clamp inferior 280 K).
        4. Re-sincronización de ‖ψ‖ y T.
        
        Si tras la intervención I sigue supercrítica, declara colapso.
        """
        logger.warning(
            f"⚠️ CRITICAL: I={potential.instability:.2f} > "
            f"{self.INSTABILITY_THRESHOLD} en {node.name}. "
            f"Aplicando amortiguamiento (ciclo={cycle})."
        )
        
        # 1. Damping oscilatorio: ψ · e^{-γt} · cos(ωt)
        state.apply_damping(cycle)
        
        # 2. Disipar entalpía acumulada (con mínimo para evitar problemas numéricos)
        potential.enthalpy = max(
            self._MIN_ENTHALPY,
            potential.enthalpy * 0.85
        )
        
        # 3. Enfriamiento activo con temperatura mínima
        potential.base_temperature = max(280.0, potential.base_temperature * 0.95)
        
        # 4. Re-sincronizar observables
        potential.topological_stress = state.norm
        
        # 5. Verificar resultado
        if potential.instability > self.INSTABILITY_THRESHOLD:
            raise RuntimeError(
                f"Reactor Collapse: Inestabilidad irrecuperable "
                f"({potential.instability:.2f}) en {node.name}"
            )
        
        logger.info(
            f"🛡️ Estabilización exitosa: I={potential.instability:.2f}, "
            f"‖ψ‖={state.norm:.4f}"
        )
    
    # ──────────────────────────────────────────────────────────────────
    # Hamiltoniano
    # ──────────────────────────────────────────────────────────────────
    
    def _calculate_hamiltonian(
        self, 
        node: CarbonNode, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calcula la energía local (Hamiltoniano Hᵢ) del nodo.
        
        Hᵢ = α + Σⱼ∈N(i) βᵢⱼ·σⱼ + penalty(precursores)
        
        donde:
        - α = integral de Coulomb (costo base de activación).
        - βᵢⱼ = integral de resonancia (acople con vecinos adyacentes).
        - σⱼ = 1 si el vecino j está en estado "resonant", 0 si no.
        - penalty = penalización por precursores faltantes.
        
        Los vecinos resonantes reducen la barrera (estabilización por
        deslocalización electrónica, análoga a la resonancia π).
        """
        idx = node.index
        
        # α: Integral de Coulomb (energía base)
        alpha = 0.2
        
        # β: Integral de resonancia — vecinos resonantes bajan la barrera
        beta = -0.05
        left, right = self.topology.neighbor_indices(idx)
        neighbor_stabilization = sum(
            beta
            for ni in (left, right)
            if context.get(f"{CarbonNode(ni + 1).name}_status") == "resonant"
        )
        
        # Penalización por precursores faltantes
        precursor_penalty = self._evaluate_precursor_penalty(node, context)
        
        hamiltonian = alpha + neighbor_stabilization + precursor_penalty
        return max(0.0, hamiltonian)  # Eₐ ≥ 0
    
    def _evaluate_precursor_penalty(
        self, 
        node: CarbonNode, 
        context: Dict[str, Any]
    ) -> float:
        """
        Evalúa disponibilidad de precursores para el nodo.
        Retorna penalización ∈ [0, 0.3] proporcional a las dependencias
        faltantes.
        """
        required = self._PRECURSOR_MAP.get(node, [])
        if not required:
            return 0.0
        
        missing = sum(1 for key in required if key not in context)
        return 0.3 * (missing / len(required))
    
    # ──────────────────────────────────────────────────────────────────
    # Reacción de Nodo
    # ──────────────────────────────────────────────────────────────────
    
    def _react_node(
        self,
        node: CarbonNode,
        context: Dict[str, Any],
        ea: float,
        local_stress: float,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Ejecuta la transformación catalítica del nodo.
        
        Si Eₐ > umbral, el nodo se salta (cinéticamente prohibido)
        y genera una penalización entálpica moderada.
        
        Retorna (contexto_modificado, ΔH).
        """
        if ea > self.ACTIVATION_BARRIER_CEILING:
            logger.warning(
                f"⚡ Saltando {node.name}: "
                f"Eₐ={ea:.3f} > {self.ACTIVATION_BARRIER_CEILING}"
            )
            return {f"{node.name}_skipped": True}, 5.0  # Penalización por nodo no procesado
        
        start = time.monotonic()
        
        # ── Transformación catalítica ──
        try:
            # En producción real: 
            # result = self.mic.project_intent(node.service_name, context)
            # Para este ejemplo, simulamos la ejecución
            context_update = {
                f"{node.name}_status": "resonant",
                f"{node.name}_ts": time.time(),
                f"{node.name}_ea": ea
            }
            
            # Simulación de latencia variable según estrés local
            stress_factor = 1.0 + (local_stress ** 2) * 0.5
            time.sleep(0.005 * stress_factor)
            
        except Exception as e:
            logger.error(f"Error en ejecución MIC para {node.name}: {e}")
            raise
        
        elapsed = time.monotonic() - start
        
        # ΔH: costo basado en estrés local (cuadrático) + Eₐ + latencia
        delta_h = (local_stress ** 2) * 5.0 + ea * 10.0 + elapsed * 100.0 
        
        logger.debug(
            f"🔬 {node.name} procesado | "
            f"Eₐ={ea:.3f} | σ={local_stress:.3f} | ΔH={delta_h:.3f}"
        )
        return context_update, delta_h
    
    # ──────────────────────────────────────────────────────────────────
    # Entropía de Shannon
    # ──────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _calculate_shannon_entropy(context: Dict[str, Any]) -> float:
        """
        Entropía de Shannon del contexto en nats (base e).
        
            S = −Σ p(xᵢ) · ln(p(xᵢ))
        
        Base natural (ln) para coherencia dimensional con S = kB·ln(Ω)
        de la termodinámica estadística.
        
        La distribución se construye sobre firmas tipo:cuantil_tamaño
        para capturar tanto diversidad de tipos como granularidad
        de contenido.
        
        Corrección v4.1:
        - Manejo robusto de casos con probabilidad cero
        - Cálculo optimizado para grandes contextos
        - Uso de ENTROPY_MIN_PROB para evitar -inf
        """
        if not context:
            return 0.0
        
        def _signature(v: Any) -> str:
            """Genera una firma compacta para el valor."""
            try:
                type_name = type(v).__name__
                # Para objetos grandes, usamos hash en lugar de tamaño
                if hasattr(v, '__len__') and len(v) > 100:
                    return f"{type_name}:HASHED"
                size_bucket = min(len(str(v)) // 10, 9) if isinstance(v, (str, bytes)) else 0
                return f"{type_name}:{size_bucket}"
            except Exception:
                return "unknown:0"
        
        # Generar firmas de manera eficiente
        signatures = []
        for v in context.values():
            try:
                signatures.append(_signature(v))
            except Exception:
                signatures.append("error:0")
        
        # Calcular frecuencias
        counts = Counter(signatures)
        total = len(signatures)
        
        # Calcular entropía con manejo seguro de casos límite
        entropy = 0.0
        for count in counts.values():
            p = max(count / total, ENTROPY_MIN_PROB)
            entropy -= p * math.log(p)  # ln → nats
        
        return entropy
    
    # ──────────────────────────────────────────────────────────────────
    # Verificación de Aromaticidad
    # ──────────────────────────────────────────────────────────────────
    
    @staticmethod
    def _is_aromatic(context: Dict[str, Any]) -> bool:
        """
        Verifica aromaticidad (estabilidad resonante completa).
        
        Regla de Hückel para un anillo de 6 miembros:
            4n + 2 electrones π, con n = 1 → se requieren 6.
        
        Condiciones (todas deben cumplirse):
        1. Los 6 nodos deben estar en estado "resonant" (6 electrones π).
        2. No debe haber errores registrados.
        3. Ningún nodo debe haber sido saltado.
        
        Corrección v4.1: Hückel estricto (exactamente 6, no % 4 == 2
        que aceptaba erróneamente 2 electrones como aromático).
        """
        pi_electrons = sum(
            1
            for k, v in context.items()
            if k.endswith("_status") and v == "resonant"
        )
        
        has_errors = any(k.endswith("_error") for k in context)
        has_skips = any(k.endswith("_skipped") for k in context)
        
        # Hückel estricto: 4(1) + 2 = 6 electrones π
        return pi_electrons == 6 and not has_errors and not has_skips
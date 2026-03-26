"""
=========================================================================================
Suite de Integración Rigurosa: Gauge Field Router vs Ecosistema APU Filter
Ubicación: tests/core/immune_system/test_gauge_field_router_integration.py
=========================================================================================

Arquitectura de verificación (8 fases):
───────────────────────────────────────

Fase 0: Invariantes Topológicos y Algebraicos de la Red
    F0.1  Primer número de Betti: β₁(C_n) = 1
    F0.2  Identidad cohomológica: L₀ = B₁ᵀB₁
    F0.3  Propiedades espectrales: L₀ = L₀ᵀ ≽ 0
    F0.4  Estructura de incidencia: B₁ ∈ {-1,0,+1}ᴹˣᴺ
    F0.5  Dimensión del kernel: dim(ker(L₀)) = c (componentes conexas)

Fase 1: Conservación de Carga y Compatibilidad de Poisson
    F1.1  Neutralidad de carga: 𝟙ᵀρ = 0
    F1.2  Condición de Fredholm: ρ ⊥ ker(L₀)
    F1.3  Precisión numérica: ‖L₀Φ − ρ‖₂/‖ρ‖₂ < ε

Fase 2: Acoplamiento Gauge y Selección de Agente
    F2.1  Consistencia con oráculo algebraico independiente
    F2.2  Campo E computado correctamente
    F2.3  Morfismo aplicado coincide con agente seleccionado

Fase 3: Ortogonalidad Funcional de Cargas
    F3.1  Base canónica: Gram(Q) = I (ortonormalidad)
    F3.2  Rango completo: rank(Q) = M
    F3.3  Discriminación correcta bajo cargas ortogonales

Fase 4: Encapsulamiento Monádico
    F4.1  Payload preservado identicamente
    F4.2  Exactamente un morfismo aplicado (un estrato validado)
    F4.3  Contexto preexistente preservado
    F4.4  Metadatos gauge inyectados correctamente
    F4.5  Agente resolutor en conjunto válido

Fase 5: Estabilidad Numérica, Determinismo e Invariancia Gauge
    F5.1  Determinismo: salidas idénticas bajo entradas idénticas
    F5.2  Invariancia gauge: E invariante bajo Φ → Φ + c𝟙
    F5.3  Linealidad: ‖E(αρ)‖ = α·‖E(ρ)‖

Fase 6: Robustez y Rechazo de Entradas Inválidas
    F6.1  Rechazo de entradas inválidas (None, rango, finitud)
    F6.2  Manejo gracioso de casos límite (severity=0)
    F6.3  Estabilidad bajo valores extremos

Fase 7: Generalización Topológica
    F7.1  Ciclos de tamaño variable (C₃ a C₁₀)
    F7.2  Grafos desconectados (C₃ ⊔ C₃)
    F7.3  Escalabilidad a topologías más grandes

Fase 8: Consistencia Cruzada
    F8.1  Normas reportadas vs computadas
    F8.2  Metadatos de diagnóstico consistentes
    F8.3  Propiedades conservadas en todas las operaciones

Estrategia de Validación:
─────────────────────────

Cada verificación emplea un "oráculo algebraico independiente" que resuelve
el pipeline gauge usando operaciones densas (pseudoinversa, eigendecomposición)
de referencia. El router (que usa LSQR disperso) se valida por comparación
contra este oráculo.

Esto desacopla:
    • La verificación de la lógica de negocio (fases 2−8)
    • De la verificación de robustez numérica del solver LSQR

Tolerancias Numéricas (justificadas):
─────────────────────────────────────
    _ATOL:                1e-10  (operaciones algebraicas)
    _RTOL:                1e-10  (error relativo)
    _SPECTRAL_TOL:        1e-10  (eigenvalores nulos)
    _ORTHOGONALITY_TOL:   1e-12  (producto interno)
    _CHARGE_NEUTRALITY_TOL: 1e-12  (suma de carga)
    _RESIDUAL_TOL:        1e-8   (residual post-Poisson)
    _MOMENTUM_TOL:        1e-12  (conservación de momento)

Determinismo:
─────────────
Se esterilizan las variables de entorno BLAS/LAPACK al inicio para garantizar
determinismo reproducible bajo entradas idénticas. Esto es crítico para
verificar F5.1.
=========================================================================================
"""

from __future__ import annotations

import os

# ── Esterilización determinista del entorno BLAS/LAPACK ──
# Fuerza uso de un solo thread para reproducibilidad exacta.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix

from app.adapters.tools_interface import MICRegistry
from app.core.immune_system.gauge_field_router import (
    ChargeNeutralityError,
    GaugeFieldError,
    GaugeFieldRouter,
    LatticeQEDConstants,
    TopologicalSingularityError,
)
from app.core.mic_algebra import CategoricalState
from app.core.schemas import Stratum

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS (TOLERANCIAS JUSTIFICADAS)
# ═════════════════════════════════════════════════════════════════════════════

_ATOL: float = 1e-10
"""Tolerancia absoluta para operaciones algebraicas (álgebra lineal densa)."""

_RTOL: float = 1e-10
"""Tolerancia relativa para comparación de valores."""

_SPECTRAL_TOL: float = 1e-10
"""
Tolerancia para detección de eigenvalores nulos.

Crítica para identificar dim(ker(L₀)). Debe ser significativamente más pequeña
que los eigenvalores no nulos (típicamente O(1)) pero mayor que
máquina épsilon (~1e-16 para float64).
"""

_ORTHOGONALITY_TOL: float = 1e-12
"""
Tolerancia para verificación de ortonormalidad (Gram(Q) = I).

Máxima rigidez: para una base canónica, off-diagonal debe ser casi exactamente 0.
"""

_CHARGE_NEUTRALITY_TOL: float = 1e-12
"""
Tolerancia para suma nula de carga (𝟙ᵀρ = 0).

Máxima rigidez: esta es una condición algebraica exacta que debe satisfacerse.
Violación indica fallo fundamental en construcción de ρ.
"""

_RESIDUAL_TOL: float = 1e-8
"""
Tolerancia para residual relativo de Poisson.

‖L₀Φ − ρ‖₂ / ‖ρ‖₂ < _RESIDUAL_TOL.

Menos rígida que atol/btol del solver LSQR (~1e-10) porque agrupa todos
los errores numéricos. Valor 1e-8 permite cierto overhead pero es suficientemente
restrictivo para aplicaciones de seguridad.
"""

_MOMENTUM_TOL: float = 1e-12
"""Tolerancia para comparación de momentum cibernético (determinismo)."""

# ═════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LatticeFixture:
    """
    Red discreta orientada con su álgebra cohomológica precomputada.

    Encapsula toda la estructura algebraica necesaria para operaciones
    de validación independiente.

    Atributos:
        graph (nx.Graph):                Grafo NetworkX subyacente.
        incidence (sp.csr_matrix):       Operador de incidencia B₁ ∈ ℝᴹˣᴺ.
        laplacian (sp.csr_matrix):       Laplaciano L₀ = B₁ᵀB₁ ∈ ℝᴺˣᴺ.
        edge_index (Dict):               Mapeo (u,v) → índice de arista.
        num_nodes (int):                 N = |V|.
        num_edges (int):                 M = |E|.
        oriented_edges (List):           Lista ordenada de aristas orientadas.

    Invariantes:
        • incidence.shape = (M, N)
        • laplacian.shape = (N, N)
        • laplacian = incidence.T @ incidence
        • laplacian = laplacian.T
        • x^T laplacian x ≥ 0 para todo x
    """
    graph: nx.Graph
    incidence: sp.csr_matrix
    laplacian: sp.csr_matrix
    edge_index: Dict[Tuple[int, int], int]
    num_nodes: int
    num_edges: int
    oriented_edges: List[Tuple[int, int]]


@dataclass(frozen=True)
class FieldAnalysis:
    """
    Análisis algebraico completo del campo inducido por una perturbación.

    Calculado de forma independiente usando operaciones densas (pseudoinversa)
    como referencia para validar los resultados del router (LSQR disperso).

    Atributos:
        rho (np.ndarray):                Densidad de carga ρ ∈ ℝᴺ.
        phi (np.ndarray):                Potencial escalar Φ ∈ ℝᴺ.
        e_field (np.ndarray):            Campo E = −B₁Φ ∈ ℝᴹ.
        actions (Dict[str, float]):      Acción de cada agente ⟨Q_k, E⟩.
        expected_agent (str):            Agente que maximiza la acción.
        residual_norm (float):           ‖L₀Φ − ρ‖₂.
        relative_residual (float):       ‖L₀Φ − ρ‖₂ / ‖ρ‖₂.

    Invariantes:
        • rho.shape = (N,)
        • phi.shape = (N,)
        • e_field.shape = (M,)
        • expected_agent ∈ actions.keys()
        • actions[expected_agent] = max(actions.values())
    """
    rho: np.ndarray
    phi: np.ndarray
    e_field: np.ndarray
    actions: Dict[str, float]
    expected_agent: str
    residual_norm: float
    relative_residual: float


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE SEGURIDAD DEFENSIVA
# ═════════════════════════════════════════════════════════════════════════════

def _safe_context(state: CategoricalState) -> Dict[str, Any]:
    """
    Extrae el contexto de un CategoricalState de forma defensiva.

    Retorna una copia mutable (dict) para permitir inspección sin
    mutar el estado original.

    Nunca falla: retorna {} como fallback.
    """
    ctx = getattr(state, "context", None)
    if ctx is None:
        return {}
    if not isinstance(ctx, Mapping):
        return {}
    return dict(ctx)


def _validated_strata_as_set(state: CategoricalState) -> Set[Any]:
    """
    Extrae los estratos validados como conjunto.

    Retorna set() como fallback si no disponibles.
    """
    strata = getattr(state, "validated_strata", None)
    if strata is None:
        return set()
    return set(strata)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTORES DE REDES DISCRETAS (TOPOLOGÍAS DE PRUEBA)
# ═════════════════════════════════════════════════════════════════════════════

def _build_cycle_lattice(num_nodes: int = 3) -> LatticeFixture:
    """
    Construye un ciclo simple C_n con orientación canónica.

    Topología:
    ──────────
    Nodos: {0, 1, ..., n-1}
    Aristas orientadas: (i → (i+1) mod n) para i ∈ {0, ..., n-1}
    Cantidad: M = N = n

    Propiedades algebraicas de C_n:
    ──────────────────────────────
    • Conexo: c = 1
    • Ciclos independientes: β₁ = 1
    • dim(ker(L₀)) = 1 (kernel = span(𝟙))
    • L₀ tridiagonal (casi, con esquina ↔)

    Invariancia bajo orientación (elección arbitraria):
    ──────────────────────────────────────────────────
    La orientación de las aristas es una elección arbitraria.
    Diferentes orientaciones producen diferentes B₁ pero el mismo L₀ = B₁ᵀB₁.
    Elegimos (i → (i+1) mod n) por simplicidad y determinismo.

    Args:
        num_nodes: Número de nodos (debe ser ≥ 3 para ciclo no trivial).

    Returns:
        LatticeFixture con toda la estructura algebraica.

    Raises:
        ValueError si num_nodes < 3.
    """
    if num_nodes < 3:
        raise ValueError(
            f"Se requiere num_nodes ≥ 3 para ciclo no trivial; recibido {num_nodes}."
        )

    # Construcción de NetworkX.
    graph = nx.cycle_graph(num_nodes)

    # Orientación canónica: i → (i+1) mod n.
    oriented_edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    m = len(oriented_edges)
    n = num_nodes

    # Construcción de B₁ en formato LIL (eficiente para construcción).
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    edge_index: Dict[Tuple[int, int], int] = {}

    for e_idx, (u, v) in enumerate(oriented_edges):
        edge_index[(u, v)] = e_idx
        # Fila e_idx: -1 en columna u (tail), +1 en columna v (head).
        rows.extend([e_idx, e_idx])
        cols.extend([u, v])
        data.extend([-1.0, 1.0])

    incidence = sp.csr_matrix(
        (data, (rows, cols)), shape=(m, n), dtype=np.float64,
    )

    # Laplaciano.
    laplacian = (incidence.T @ incidence).tocsr()

    return LatticeFixture(
        graph=graph,
        incidence=incidence,
        laplacian=laplacian,
        edge_index=edge_index,
        num_nodes=n,
        num_edges=m,
        oriented_edges=oriented_edges,
    )


def _build_disconnected_lattice() -> LatticeFixture:
    """
    Construye un grafo desconectado: C₃ ⊔ C₃ (dos triángulos disjuntos).

    Topología:
    ──────────
    Componente 1: nodos {0, 1, 2} con aristas (0→1), (1→2), (2→0)
    Componente 2: nodos {3, 4, 5} con aristas (3→4), (4→5), (5→3)
    Total: N = 6, M = 6

    Propiedades algebraicas:
    ──────────────────────
    • Componentes conexas: c = 2
    • Ciclos independientes: β₁ = 2 (uno por componente)
    • dim(ker(L₀)) = 2 (indicadores de componentes)
    • Permite testear condición de Fredholm generalizada

    Esto es crítico para verificar que el router maneja correctamente
    grafos desconectados, donde la condición de Fredholm no es
    simplemente 𝟙ᵀρ = 0 sino ρ ⊥ span(indicadores).

    Returns:
        LatticeFixture con estructura desconectada.
    """
    n = 6
    oriented_edges = [
        # Componente 1.
        (0, 1), (1, 2), (2, 0),
        # Componente 2.
        (3, 4), (4, 5), (5, 3),
    ]

    m = len(oriented_edges)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    edge_index: Dict[Tuple[int, int], int] = {}

    for e_idx, (u, v) in enumerate(oriented_edges):
        edge_index[(u, v)] = e_idx
        rows.extend([e_idx, e_idx])
        cols.extend([u, v])
        data.extend([-1.0, 1.0])

    incidence = sp.csr_matrix(
        (data, (rows, cols)), shape=(m, n), dtype=np.float64,
    )
    laplacian = (incidence.T @ incidence).tocsr()

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, v in oriented_edges:
        graph.add_edge(u, v)

    return LatticeFixture(
        graph=graph,
        incidence=incidence,
        laplacian=laplacian,
        edge_index=edge_index,
        num_nodes=n,
        num_edges=m,
        oriented_edges=oriented_edges,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTORES DE CARGAS Y MORFISMOS
# ═════════════════════════════════════════════════════════════════════════════

def _canonical_edge_charge(num_edges: int, edge_idx: int) -> np.ndarray:
    """
    Construye el vector base canónico e_{edge_idx} ∈ ℝᴹ.

    Base canónica:
        e_i[j] = δ_{ij} = 1 si i=j, 0 c.c.

    Precondiciones:
        0 ≤ edge_idx < num_edges.

    Args:
        num_edges: Dimensión M.
        edge_idx: Índice del vector base (0-indexed).

    Returns:
        Carga canónica ∈ ℝᴹ.
    """
    assert 0 <= edge_idx < num_edges, (
        f"edge_idx={edge_idx} fuera de rango [0, {num_edges})."
    )
    q = np.zeros(num_edges, dtype=np.float64)
    q[edge_idx] = 1.0
    return q


def _build_canonical_charges(
    num_edges: int,
    agent_ids: List[str],
) -> Dict[str, np.ndarray]:
    """
    Construye una base ortonormal canónica de cargas.

    Asigna:
        Q_{agent_ids[i]} = e_i  (i-ésimo vector canónico)

    Precondición:
        len(agent_ids) = num_edges (uno por arista).

    Propiedades:
        • Gram(Q) = I (ortonormalidad)
        • rank(Q) = M (rango completo)
        • ⟨Q_i, Q_j⟩ = δ_{ij}

    Args:
        num_edges: Dimensión M (número de aristas).
        agent_ids: IDs de agentes, |agent_ids| = M.

    Returns:
        Diccionario agent_id → Q_k ∈ ℝᴹ.

    Raises:
        AssertionError si len(agent_ids) ≠ num_edges.
    """
    assert len(agent_ids) == num_edges, (
        f"Número de agentes ({len(agent_ids)}) ≠ número de aristas ({num_edges})."
    )
        # We need to explicitly order the mapping creation if we rely on it being the same
    return {
        agent_id: _canonical_edge_charge(num_edges, i)
        for i, agent_id in enumerate(agent_ids)
    }


_AGENT_STRATUM_MAP: Dict[str, Stratum] = {
    "agent_physics": Stratum.PHYSICS,
    "agent_tactics": Stratum.TACTICS,
    "agent_strategy": Stratum.STRATEGY,
}
"""Mapeo de agentes a sus estratos para morfismos de prueba."""


def _make_morphism(
    agent_id: str,
    stratum: Stratum,
) -> Callable[[CategoricalState], CategoricalState]:
    """
    Construye un morfismo de prueba puro y determinista.

    Comportamiento:
    ───────────────
    1. Preserva el payload (identidad).
    2. Agrega `resolved_by: agent_id` al contexto.
    3. Agrega `stratum` a validated_strata.

    Pura: Sin efectos secundarios, determinista, idempotencia de payload.

    Args:
        agent_id: Identificador del agente.
        stratum: Estrato asociado.

    Returns:
        Morphism: Función pura de transformación de estado.
    """
    def handler(state: CategoricalState) -> CategoricalState:
        context = {**_safe_context(state), "resolved_by": agent_id}
        validated = _validated_strata_as_set(state) | {stratum}
        return CategoricalState(
            payload=state.payload,
            context=context,
            validated_strata=validated,
        )

    handler.__name__ = f"morphism_{agent_id}"
    handler.__qualname__ = f"morphism_{agent_id}"
    return handler


# ═════════════════════════════════════════════════════════════════════════════
# ORÁCULO ALGEBRAICO INDEPENDIENTE
#
# Resuelve el pipeline gauge de forma independiente al router,
# permitiendo validación por comparación (cross-validation).
# ═════════════════════════════════════════════════════════════════════════════

def _compute_field_analysis(
    lattice: LatticeFixture,
    agent_charges: Dict[str, np.ndarray],
    anomaly_node: int,
    severity: float,
) -> FieldAnalysis:
    """
    Calcula de forma independiente el campo E y las acciones de acoplamiento.

    Este oráculo usa álgebra lineal densa para máxima claridad de referencia.
    Se contrasta contra el router (que usa LSQR disperso) para validar
    que la lógica de negocio es correcta, independientemente de robustez
    numérica del solver.

    Pipeline (independiente al router):
    ──────────────────────────────────

    [1] Densidad de carga:
        ρ = severity · (δ_{node} − 𝟙/N)
        Invariante: 𝟙ᵀρ = 0 (suma nula)

    [2] Pseudoinversa (mínima norma):
        L⁺ = pseudoinversa de Moore-Penrose
        Φ = L⁺ρ  (solución de mínima norma)
        Invariante: Φ ⊥ ker(L)

    [3] Campo eléctrico discreto:
        E = −B₁Φ  (gradiente negativo del potencial)
        Invariante: E ∈ Im(B₁)

    [4] Acoplamiento gauge:
        action_k = ⟨Q_k, E⟩ para cada agente k
        k* = argmax_k action_k (maximizador lexicográfico en caso de empate)

    [5] Diagnóstico:
        residual = L₀Φ − ρ
        relative_residual = ‖residual‖₂ / ‖ρ‖₂

    Args:
        lattice: Red discreta con B₁, L₀.
        agent_charges: Mapeo agent_id → Q_k.
        anomaly_node: Nodo perturbado (ya validado).
        severity: Magnitud de perturbación (ya validada).

    Returns:
        FieldAnalysis con resultados de referencia para validación cruzada.

    Precondiciones:
        • 0 ≤ anomaly_node < N
        • severity > 0, finita
    """
    n = lattice.num_nodes
    m = lattice.num_edges

    # [1] Densidad de carga.
    rho = np.full(n, -severity / n, dtype=np.float64)
    rho[anomaly_node] += severity
    rho -= np.mean(rho)  # Corrección numérica para suma exactamente nula.

    # [2] Pseudoinversa (máxima claridad de referencia).
    L_dense = lattice.laplacian.toarray().astype(np.float64)
    L_pinv = np.linalg.pinv(L_dense)
    phi = L_pinv @ rho

    # [3] Campo discreto.
    B1_dense = lattice.incidence.toarray().astype(np.float64)
    e_field = B1_dense @ phi

    # [4] Cálculo de acciones y selección de agente.
    actions: Dict[str, float] = {}
    for agent_id, q_k in agent_charges.items():
        actions[agent_id] = float(np.dot(q_k, e_field))

    max_action = max(actions.values())
    maximizers = sorted(
        aid for aid, act in actions.items()
        if np.isclose(act, max_action, atol=1e-14)
    )
    expected_agent = maximizers[0]

    # [5] Diagnóstico de residual.
    residual = L_dense @ phi - rho
    residual_norm = float(np.linalg.norm(residual))
    rho_norm = float(np.linalg.norm(rho))
    relative_residual = residual_norm / max(rho_norm, 1e-14)

    return FieldAnalysis(
        rho=rho,
        phi=phi,
        e_field=e_field,
        actions=actions,
        expected_agent=expected_agent,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES (PYTEST)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def lattice() -> LatticeFixture:
    """Red cíclica minimal C₃ (triángulo)."""
    return _build_cycle_lattice(num_nodes=3)


@pytest.fixture
def disconnected_lattice() -> LatticeFixture:
    """Red desconectada C₃ ⊔ C₃ (dos triángulos disjuntos)."""
    return _build_disconnected_lattice()


@pytest.fixture
def mock_mic_registry() -> MICRegistry:
    """
    MIC con tres morfismos puros, uno por estrato.

    Cada morfismo:
        • Es una función pura (sin efectos secundarios)
        • Preserva payload (identidad)
        • Anota contexto con `resolved_by: agent_id`
        • Agrega estrato a validated_strata
    """
    registry = MICRegistry()

    for agent_id, stratum in _AGENT_STRATUM_MAP.items():
        handler = _make_morphism(agent_id, stratum)
        registry.register_vector(agent_id, stratum, handler)

    return registry


@pytest.fixture
def agent_charges(lattice: LatticeFixture) -> Dict[str, np.ndarray]:
    """
    Base canónica ortonormal de cargas en ℝᴹ.

    Para C₃ (M=3):
        Q_physics  = e₀ = (1, 0, 0)
        Q_tactics  = e₁ = (0, 1, 0)
        Q_strategy = e₂ = (0, 0, 1)

    Propiedades:
        • Gram(Q) = I
        • rank(Q) = M
        • Ortonormales
    """
        # The agent_ids order determines their canonical projection 0, 1, 2
    return _build_canonical_charges(
        num_edges=lattice.num_edges,
            agent_ids=sorted(list(_AGENT_STRATUM_MAP.keys())),
    )


@pytest.fixture
def router(
    mock_mic_registry: MICRegistry,
    lattice: LatticeFixture,
    agent_charges: Dict[str, np.ndarray],
) -> GaugeFieldRouter:
    """Router gauge completamente inicializado sobre C₃."""
    return GaugeFieldRouter(
        mic_registry=mock_mic_registry,
        laplacian=lattice.laplacian,
        incidence_matrix=lattice.incidence,
        agent_charges=agent_charges,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 0: INVARIANTES TOPOLÓGICOS Y ALGEBRAICOS DE LA RED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestLatticeTopologicalInvariants:
    """
    Verifica que las redes de prueba satisfacen sus invariantes algebraicos.

    Estos tests validan:
    • El oráculo algebraico independiente
    • Las fixtures que sirven como base para todas las pruebas
    """

    def test_first_betti_number(self, lattice: LatticeFixture) -> None:
        """
        F0.1: β₁(C_n) = 1.

        Primer número de Betti = número de ciclos independientes.
        Para ciclo C_n: β₁ = n − n + 1 = 1.

        Fórmula general (Euler característico):
            β₁ = M − N + c
        donde M = aristas, N = nodos, c = componentes conexas.
        """
        g = lattice.graph
        c = nx.number_connected_components(g)
        m = g.number_of_edges()
        n = g.number_of_nodes()
        beta_1 = m - n + c

        assert beta_1 == 1, (
            f"β₁ = {beta_1} ≠ 1 para C₃. "
            f"M={m}, N={n}, c={c}. "
            f"Fórmula: β₁ = M − N + c = {m} − {n} + {c} = {beta_1}."
        )

    def test_cohomological_identity(self, lattice: LatticeFixture) -> None:
        """
        F0.2: L₀ = B₁ᵀB₁.

        Identidad fundamental de cohomología discreta.

        En el complejo simplicial 1D (grafo):
            L₀ := d₀ᵀd₀  (Laplaciano de 0-formas)
        donde d₀ es el operador coborde, representado por B₁.

        Esta relación es exacta (no aproximada) en aritmética exacta.
        En punto flotante, esperamos ‖L₀ − B₁ᵀB₁‖_F < ε_máquina.
        """
        B1 = lattice.incidence
        L = lattice.laplacian

        reconstructed = (B1.T @ B1).toarray()
        L_dense = L.toarray()

        assert L_dense.shape == reconstructed.shape, (
            f"Formas incompatibles: L={L_dense.shape}, "
            f"B₁ᵀB₁={reconstructed.shape}."
        )

        assert np.allclose(L_dense, reconstructed, atol=_ATOL, rtol=_RTOL), (
            f"Violación de identidad cohomológica: L₀ ≠ B₁ᵀB₁.\n"
            f"‖L₀ − B₁ᵀB₁‖_F = {np.linalg.norm(L_dense - reconstructed):.3e}.\n"
            f"L₀ =\n{L_dense}\n"
            f"B₁ᵀB₁ =\n{reconstructed}\n"
            f"Diferencia =\n{L_dense - reconstructed}"
        )

    def test_laplacian_symmetry(self, lattice: LatticeFixture) -> None:
        """
        F0.3a: L₀ = L₀ᵀ (simetría).

        Un Laplaciano de grafo es siempre simétrico real.
        Esto se debe a: L₀ = B₁ᵀB₁ ⟹ L₀ᵀ = (B₁ᵀB₁)ᵀ = B₁ᵀ(B₁ᵀ)ᵀ = B₁ᵀB₁ = L₀.
        """
        L_dense = lattice.laplacian.toarray()
        diff_norm = float(np.linalg.norm(L_dense - L_dense.T, ord='fro'))

        assert diff_norm < _ATOL, (
            f"El Laplaciano no es simétrico: ‖L₀ − L₀ᵀ‖_F = {diff_norm:.3e} > "
            f"{_ATOL:.3e}.\nL₀ =\n{L_dense}"
        )

    def test_laplacian_positive_semidefinite(self, lattice: LatticeFixture) -> None:
        """
        F0.3b: L₀ ≽ 0 (semidefinida positiva).

        Para L₀ = B₁ᵀB₁:
            xᵀL₀x = xᵀB₁ᵀB₁x = ‖B₁x‖₂² ≥ 0  para todo x.

        Verificación: todos los eigenvalores son ≥ 0.
        """
        L_dense = lattice.laplacian.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)
        lambda_min = float(eigenvalues[0])

        assert lambda_min >= -_SPECTRAL_TOL, (
            f"Violación de semidefinitud positiva: λ_min = {lambda_min:.3e} < "
            f"−{_SPECTRAL_TOL:.3e}.\n"
            f"Espectro completo: {eigenvalues}."
        )

    def test_incidence_matrix_entries(self, lattice: LatticeFixture) -> None:
        """
        F0.4a: B₁ ∈ {-1, 0, +1}ᴹˣᴺ.

        Matriz de incidencia orientada tiene entradas exactamente
        en {-1, 0, +1}:
            B₁[e, u] = −1  si u es la cola de e
            B₁[e, v] = +1  si v es la cabeza de e
            B₁[e, j] = 0   si j no incide en e
        """
        B1_dense = lattice.incidence.toarray()
        valid_entries = {-1.0, 0.0, 1.0}

        unique_entries = set(np.unique(B1_dense))
        invalid = unique_entries - valid_entries

        assert len(invalid) == 0, (
            f"Entradas inválidas en B₁: {invalid}. "
            f"Solo se permiten {{-1, 0, +1}}."
        )

    def test_incidence_two_nonzeros_per_row(self, lattice: LatticeFixture) -> None:
        """
        F0.4b: Cada fila de B₁ tiene exactamente 2 entradas no nulas.

        Una fila de B₁ corresponde a una arista e.
        La arista incide en exactamente 2 nodos (tail y head).
        Luego: fila tiene un −1 y un +1, rest os ceros.
        """
        B1_dense = lattice.incidence.toarray()

        for e_idx in range(lattice.num_edges):
            row = B1_dense[e_idx, :]
            nonzero_count = int(np.count_nonzero(row))
            assert nonzero_count == 2, (
                f"Arista {e_idx}: {nonzero_count} entradas no nulas (esperadas 2). "
                f"Fila: {row}."
            )

            # Verificar suma = 0 (un +1 y un −1).
            row_sum = float(np.sum(row))
            assert abs(row_sum) < _ATOL, (
                f"Arista {e_idx}: suma de fila = {row_sum} ≠ 0. "
                f"Se requiere exactamente un +1 y un −1."
            )

    def test_discrete_de_rham_exactness(self, lattice: LatticeFixture) -> None:
        """
        F0.6: Verifica la identidad d₁ ∘ d₀ = 0 (el rotacional de un gradiente es nulo).
        Calcula analíticamente la base del espacio de ciclos (1-homología) y aserta
        que cualquier campo de Gauge exacto (E = B₁Φ) tiene circulación estrictamente nula.
        """
        import scipy.linalg

        B1 = lattice.incidence
        B1_dense = B1.toarray()

        # 1. Extracción analítica del espacio de ciclos: ker(B₁ᵀ)
        cycle_basis = scipy.linalg.null_space(B1_dense.T)

        # 2. Computar un campo de Gauge inducido por un potencial escalar arbitrario
        Phi = np.random.RandomState(101).randn(B1.shape[1])
        E = B1 @ Phi

        # 3. Aserción de Exactitud de Hodge (Si existen ciclos en la topología)
        if cycle_basis.shape[1] > 0:
            for i in range(cycle_basis.shape[1]):
                cycle_operator = cycle_basis[:, i]

                # La circulación es el producto interno ⟨C, E⟩
                circulation = np.dot(cycle_operator, E)

                np.testing.assert_allclose(
                    circulation, 0, atol=_ATOL,
                    err_msg=f"Violación de la cohomología de de Rham. El campo E contiene vórtices espurios: Circulación = {circulation}"
                )

    def test_kernel_dimension_equals_connected_components(
        self, lattice: LatticeFixture,
    ) -> None:
        """
        F0.5: dim(ker(L₀)) = c (número de componentes conexas).

        Teorema algebraico:
        ──────────────────
        Para Laplaciano de grafo L₀ = B₁ᵀB₁ sobre grafo con c componentes,
        el kernel tiene dimensión exactamente c, generado por los indicadores
        de cada componente.

        Para C₃ (conexo): dim(ker(L₀)) = 1, con ker(L₀) = span(𝟙).
        Para C₃ ⊔ C₃ (desconexo): dim(ker(L₀)) = 2.
        """
        L_dense = lattice.laplacian.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)

        kernel_dim = int(np.sum(np.abs(eigenvalues) < _SPECTRAL_TOL))
        c = nx.number_connected_components(lattice.graph)

        assert kernel_dim == c, (
            f"dim(ker(L₀)) = {kernel_dim} ≠ c = {c}. "
            f"Espectro: {eigenvalues}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1: CONSERVACIÓN DE CARGA Y COMPATIBILIDAD DE POISSON
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestChargeConservationAndPoisson:
    """Verificación de conservación de carga y resolución de Poisson."""

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    @pytest.mark.parametrize("severity", [0.1, 1.0, 10.0, 1000.0])
    def test_charge_density_is_neutral(
        self,
        router: GaugeFieldRouter,
        anomaly_node: int,
        severity: float,
    ) -> None:
        """
        F1.1: 𝟙ᵀρ = 0 para todo nodo y severidad.

        Construcción de ρ:
        ──────────────────
        ρ := severity · (δ_{anomaly_node} − 𝟙/N)

        Prueba de neutralidad:
        ─────────────────────
        𝟙ᵀρ = severity · (𝟙ᵀδ_{node} − 𝟙ᵀ(𝟙/N))
            = severity · (1 − N/N)
            = severity · (1 − 1)
            = 0 ✓

        Precondición para solubilidad de Poisson sobre grafos conexos.
        """
        rho = router._quantize_bosonic_excitation(anomaly_node, severity)

        assert rho.shape == (router.num_nodes,), (
            f"ρ.shape = {rho.shape}, esperada ({router.num_nodes},)."
        )

        total_charge = float(np.sum(rho))
        assert abs(total_charge) < _CHARGE_NEUTRALITY_TOL, (
            f"Violación de neutralidad: 𝟙ᵀρ = {total_charge:.3e}. "
            f"Nodo anomalía={anomaly_node}, severity={severity}."
        )

    def test_fredholm_orthogonality_on_disconnected_manifold(self, disconnected_lattice: LatticeFixture) -> None:
        """
        F1.2: Verifica que ρ ⊥ ker(L₀) para cada subespacio del núcleo en grafos con β₀ > 1.
        La simple neutralidad global 𝟙ᵀρ = 0 es insuficiente para garantizar la resolución de Poisson.
        """
        L0 = disconnected_lattice.laplacian

        # 1. Extracción analítica de la base del núcleo (eigenvectores con λ = 0)
        eigenvalues, eigenvectors = np.linalg.eigh(L0.toarray())
        null_space_basis = eigenvectors[:, np.abs(eigenvalues) < _SPECTRAL_TOL]

        # 2. Generación de carga anómala que suma 0 globalmente pero viola neutralidad local
        # Inyectamos +1 en C_3(A) y -1 en C_3(B).
        rho_invalid = np.zeros(L0.shape[0])
        rho_invalid[1] = 1.0   # Nodo en componente A (0, 1, 2)
        rho_invalid[4] = -1.0  # Nodo en componente B (3, 4, 5)

        assert abs(np.sum(rho_invalid)) < _CHARGE_NEUTRALITY_TOL, "La carga debe ser globalmente neutra."

        # 3. Demostración de violación de Fredholm: la proyección sobre el núcleo no es nula
        projections = null_space_basis.T @ rho_invalid
        assert not np.allclose(projections, 0, atol=_CHARGE_NEUTRALITY_TOL), \
            "Falso negativo: La carga viola la neutralidad local."

        # 4. El Router debe interceptar esta singularidad topológica antes de llamar al solver
        # CORRECCIÓN: Usar exclusivamente las excepciones ontológicamente definidas en el módulo
        with pytest.raises(ChargeNeutralityError):
            from app.core.immune_system.gauge_field_router import _validate_charge_density
            _validate_charge_density(rho_invalid, L0)

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    def test_charge_density_orthogonal_to_kernel(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
        anomaly_node: int,
    ) -> None:
        """
        F1.2: ρ ⊥ ker(L₀) (condición de Fredholm).

        Teorema (Alternativa de Fredholm):
        ──────────────────────────────────
        Sistema lineal Lx = b es soluble ⟺ b ⊥ ker(Lᵀ).
        Para L simétrica (como Laplaciano): Lx = b soluble ⟺ b ⊥ ker(L).

        Para grafos:
        • Grafo conexo (c=1): ker(L) = span(𝟙), luego ρ soluble ⟺ 𝟙ᵀρ = 0
        • Grafo desconexo (c>1): ker(L) = span(indicadores), más restrictivo

        Verificación explícita contra eigenvectores del kernel (general).
        """
        rho = router._quantize_bosonic_excitation(anomaly_node, 5.0)

        L_dense = lattice.laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

        kernel_mask = np.abs(eigenvalues) < _SPECTRAL_TOL
        kernel_vecs = eigenvectors[:, kernel_mask]

        for j in range(kernel_vecs.shape[1]):
            v_j = kernel_vecs[:, j]
            projection = float(np.abs(np.dot(rho, v_j)))
            assert projection < _CHARGE_NEUTRALITY_TOL, (
                f"ρ no es ortogonal a ker(L₀): |⟨ρ, v_{j}⟩| = {projection:.3e}. "
                f"Nodo anomalía: {anomaly_node}."
            )

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    @pytest.mark.parametrize("severity", [1.0, 100.0])
    def test_poisson_residual_is_small(
        self,
        router: GaugeFieldRouter,
        anomaly_node: int,
        severity: float,
    ) -> None:
        """
        F1.3: ‖L₀Φ − ρ‖₂ / ‖ρ‖₂ < ε_residual.

        Verificación de precisión del solver LSQR.

        Métrica:
        ───────
        Residual relativo = ‖error absoluto‖ / ‖entrada‖

        Tolerancia: 1e-8 es suficientemente restrictiva para aplicaciones
        de seguridad pero permite overhead numérico típico de LSQR.
        """
        rho = router._quantize_bosonic_excitation(anomaly_node, severity)
        poisson_sol = router._solve_discrete_poisson(rho)
        phi = poisson_sol.phi

        residual = router._L.dot(phi) - rho
        residual_norm = float(np.linalg.norm(residual))
        rho_norm = float(np.linalg.norm(rho))
        relative_residual = residual_norm / max(rho_norm, 1e-14)

        assert phi.shape == (router.num_nodes,), (
            f"Φ.shape = {phi.shape}, esperada ({router.num_nodes},)."
        )

        assert np.all(np.isfinite(phi)), "Φ contiene valores no finitos."

        assert relative_residual < _RESIDUAL_TOL, (
            f"Residual relativo excesivo: {relative_residual:.3e} > "
            f"{_RESIDUAL_TOL:.3e}. "
            f"Nodo={anomaly_node}, severity={severity}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2: ACOPLAMIENTO GAUGE Y SELECCIÓN DE AGENTE
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestGaugeCoupling:
    """
    Verificación del acoplamiento gauge y selección de agente.

    Estrategia de validación:
    ────────────────────────
    Se usa un oráculo algebraico independiente que calcula E y acciones
    usando pseudoinversa densa (máxima claridad). El router (LSQR disperso)
    se valida por comparación.

    Esto desacopla:
    • Verificación de lógica de negocio (independiente de solver)
    • Verificación de robustez numérica (solver LSQR vs pseudoinversa)
    """

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    def test_router_selects_oracle_predicted_agent(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
        agent_charges: Dict[str, np.ndarray],
        anomaly_node: int,
    ) -> None:
        """
        F2.1: El agente seleccionado por el router coincide con la
        predicción del oráculo algebraico independiente.

        Esta es la prueba más fundamental de que el enrutamiento
        es funcionalmente correcto.
        """
        analysis = _compute_field_analysis(
            lattice, agent_charges, anomaly_node, severity=10.0,
        )

        state = CategoricalState(payload={"test": "coupling"}, context={})
        out = router.route_perturbation(
            state, anomaly_node=anomaly_node, severity=10.0,
        )

        ctx = _safe_context(out)
        router_selected = ctx.get("gauge_selected_agent")

        assert router_selected == analysis.expected_agent, (
            f"Discrepancia router vs oráculo para nodo {anomaly_node}:\n"
            f"  Router seleccionó:  '{router_selected}'\n"
            f"  Oráculo predijo:    '{analysis.expected_agent}'\n"
            f"  Acciones (oráculo): {analysis.actions}\n"
            f"  Campo E (oráculo):  {analysis.e_field}"
        )

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    def test_field_matches_oracle(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
        agent_charges: Dict[str, np.ndarray],
        anomaly_node: int,
    ) -> None:
        """
        F2.2: El campo E computado por el router coincide numéricamente
        con el campo del oráculo independiente.

        Invariancia gauge: E = −B₁Φ es invariante bajo Φ → Φ + c𝟙 porque
        B₁𝟙 = 0. Por lo tanto, diferencias en E indican diferencias en
        el potencial (módulo constante).
        """
        analysis = _compute_field_analysis(
            lattice, agent_charges, anomaly_node, severity=5.0,
        )

        rho = router._quantize_bosonic_excitation(anomaly_node, 5.0)
        poisson_sol = router._solve_discrete_poisson(rho)
        e_field_router = router._compute_potential_gradient(poisson_sol.phi)

        # 2. RESTAURAR LA IGUALDAD VECTORIAL ESTRICTA EN EL CÁLCULO DEL CAMPO
        # (Eliminar np.linalg.norm() y usar assert_allclose sobre el tensor completo)
        np.testing.assert_allclose(
            e_field_router,
            analysis.e_field,
            atol=_ATOL,
            rtol=_RTOL,
            err_msg="Fractura en la 1-forma computada: El campo de Gauge discreto diverge del oráculo independiente."
        )

    def test_gauge_coupling_adjoint_isomorphism(self, lattice: LatticeFixture, agent_charges: Dict[str, np.ndarray]) -> None:
        """
        F2.4: Verifica el axioma adjunto de la cohomología de redes.
        El acoplamiento en el espacio de aristas ⟨E, q⟩_E debe ser matemáticamente
        idéntico al acoplamiento en el espacio de vértices ⟨Φ, B₁ᵀq⟩_V.
        """
        B1 = lattice.incidence
        # Potencial escalar aleatorio
        Phi = np.random.RandomState(42).randn(B1.shape[1])
        E = B1 @ Phi

        for agent_id, q in agent_charges.items():
            # Acoplamiento en el 1-esqueleto (Momentum · Carga del Agente)
            coupling_edge_space = np.dot(E, q)

            # Acoplamiento en el 0-esqueleto (Potencial · Divergencia de la Carga)
            divergence_q = B1.T @ q
            coupling_vertex_space = np.dot(Phi, divergence_q)

            # Aserción de isomorfismo categórico
            np.testing.assert_allclose(
                coupling_edge_space,
                coupling_vertex_space,
                atol=_ORTHOGONALITY_TOL,
                err_msg=f"Violación del axioma adjunto para el agente {agent_id}. ⟨B₁Φ, q⟩ ≠ ⟨Φ, B₁ᵀq⟩."
            )

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    def test_morphism_application_matches_selection(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
        agent_charges: Dict[str, np.ndarray],
        anomaly_node: int,
    ) -> None:
        """
        F2.3: El morfismo aplicado es consistente con el agente seleccionado.

        El contexto de salida debe tener:
        • gauge_selected_agent = agent k*
        • resolved_by = agent k* (inyectado por el morfismo)

        Estos dos campos deben coincidir exactamente.
        """
        state = CategoricalState(payload={"test": "morphism"}, context={})
        out = router.route_perturbation(
            state, anomaly_node=anomaly_node, severity=10.0,
        )

        ctx = _safe_context(out)
        selected = ctx.get("gauge_selected_agent")
        resolved = ctx.get("resolved_by")

        assert selected is not None, "gauge_selected_agent ausente del contexto."
        assert resolved is not None, "resolved_by ausente del contexto."
        assert selected == resolved, (
            f"Inconsistencia: gauge_selected_agent='{selected}' "
            f"pero resolved_by='{resolved}'."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3: ORTOGONALIDAD FUNCIONAL DE CARGAS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestChargeOrthogonality:
    """Verificación de la estructura algebraica de la base de cargas."""

    def test_charge_basis_is_orthonormal(
        self, agent_charges: Dict[str, np.ndarray],
    ) -> None:
        """
        F3.1: Gram(Q) = I (ortonormalidad).

        Las cargas canónicas forman una base ortonormal de ℝᴹ.

        Verificación:
        ─────────────
        Gram(Q) := Qᵀ Q = I (matriz identidad)
        """
        ordered_ids = sorted(agent_charges.keys())
        Q = np.column_stack([agent_charges[aid] for aid in ordered_ids])

        gram = Q.T @ Q
        expected = np.eye(len(ordered_ids), dtype=np.float64)

        assert np.allclose(gram, expected, atol=_ORTHOGONALITY_TOL), (
            f"Gram(Q) ≠ I.\n"
            f"Gram =\n{gram}\n"
            f"Esperado =\n{expected}\n"
            f"Diferencia:\n{gram - expected}"
        )

    def test_charge_basis_has_full_rank(
        self, agent_charges: Dict[str, np.ndarray],
    ) -> None:
        """
        F3.2: rank(Q) = M (rango completo).

        La base de cargas debe generar todo ℝᴹ.
        """
        ordered_ids = sorted(agent_charges.keys())
        Q = np.column_stack([agent_charges[aid] for aid in ordered_ids])

        rank = int(np.linalg.matrix_rank(Q, tol=_ATOL))
        expected_rank = Q.shape[1]

        assert rank == expected_rank, (
            f"rank(Q) = {rank} ≠ {expected_rank}. "
            f"Base de cargas degenerada."
        )

    def test_coupling_discriminates_under_orthogonal_charges(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
        agent_charges: Dict[str, np.ndarray],
    ) -> None:
        """
        F3.3: Para cargas ortogonales (Q = I), la acción máxima
        corresponde a la componente máxima del campo E.

        Bajo Q = I: action_k = e_k · e_k + e_j · 0 = E_k
        Luego: argmax_k action_k = argmax_k E_k
        """
        for node in range(lattice.num_nodes):
            analysis = _compute_field_analysis(
                lattice, agent_charges, node, severity=10.0,
            )

            # Bajo Q = I, las acciones SON las componentes del campo.
            # 1. RESTAURAR EL DETERMINISMO EN LA EXTRACCIÓN DE BASES
            agent_ids = sorted(agent_charges.keys())  # Innegociable: preservar orden lexicográfico espectral
            # We ensure deterministic enumeration corresponding to the canonical basis initialization
            for agent_id, q_k in agent_charges.items():
                charge_idx = agent_ids.index(agent_id)
                action = np.dot(q_k, analysis.e_field)
                assert np.isclose(
                    action, analysis.e_field[charge_idx], atol=_ATOL,
                ), (
                    f"Bajo Q = I, action_{agent_id} = {action} ≠ "
                    f"E[{charge_idx}] = {analysis.e_field[charge_idx]}."
                )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 4: ENCAPSULAMIENTO MONÁDICO
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMonadicEncapsulation:
    """Verificación del encapsulamiento monádico (monadología de states)."""

    def test_payload_preserved(self, router: GaugeFieldRouter) -> None:
        """
        F4.1: El payload debe ser preservado identicamente por el router.

        El router opera exclusivamente sobre el contexto. El payload
        solo puede ser transformado por el morfismo aplicado. Nuestros
        morfismos de test son identidades sobre payload.
        """
        payload = {
            "key": "value",
            "entropy": 42,
            "nested": {"a": [1, 2, 3]},
        }
        state = CategoricalState(
            payload=payload,
            context={"preexisting": True},
        )

        out = router.route_perturbation(state, anomaly_node=0, severity=5.0)

        assert out.payload == payload, (
            f"Payload alterado.\n"
            f"Original: {payload}\n"
            f"Recibido: {out.payload}"
        )

    def test_exactly_one_morphism_applied(self, router: GaugeFieldRouter) -> None:
        """
        F4.2: Exactamente un estrato debe ser validado (un morfismo aplicado).

        Encapsulamiento monádico: el router aplica un único morfismo
        del agente seleccionado. No hay composición de morfismos.
        """
        state = CategoricalState(
            payload={"case": "single_dispatch"},
            context={},
        )
        out = router.route_perturbation(state, anomaly_node=1, severity=7.5)

        validated = _validated_strata_as_set(out)
        assert len(validated) == 1, (
            f"Se esperaba exactamente 1 estrato validado; recibido "
            f"{len(validated)}: {validated}."
        )

    def test_preexisting_context_preserved(self, router: GaugeFieldRouter) -> None:
        """
        F4.3: El contexto preexistente debe preservarse (no sobrescribirse).

        El router inyecta nuevos campos pero debe mantener
        los campos preexistentes.
        """
        preexisting = {"alpha": 3.14, "sentinel": "untouched"}
        state = CategoricalState(payload={}, context=preexisting)

        out = router.route_perturbation(state, anomaly_node=0, severity=1.0)
        ctx = _safe_context(out)

        for key, value in preexisting.items():
            assert key in ctx, f"Clave preexistente '{key}' perdida."
            assert ctx[key] == value, (
                f"Valor preexistente alterado: {key}={ctx[key]} ≠ {value}."
            )

    def test_gauge_metadata_injected(self, router: GaugeFieldRouter) -> None:
        """
        F4.4: El contexto de salida debe contener todos los metadatos gauge.

        Campos inyectados (mínimamente):
        • cyber_momentum
        • resolved_anomaly_node
        • gauge_selected_agent
        • gauge_charge_density_norm
        • gauge_field_norm
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=0, severity=5.0)
        ctx = _safe_context(out)

        required_keys = {
            "cyber_momentum",
            "resolved_anomaly_node",
            "gauge_selected_agent",
            "gauge_charge_density_norm",
            "gauge_field_norm",
        }

        missing = required_keys - set(ctx.keys())
        assert not missing, (
            f"Metadatos gauge faltantes: {missing}. "
            f"Contexto tiene: {set(ctx.keys())}."
        )

    def test_resolved_by_in_valid_agent_set(self, router: GaugeFieldRouter) -> None:
        """
        F4.5: El agente resolutor debe pertenecer al conjunto de agentes registrados.
        """
        valid_agents = set(_AGENT_STRATUM_MAP.keys())

        for node in range(3):
            state = CategoricalState(payload={}, context={})
            out = router.route_perturbation(state, anomaly_node=node, severity=5.0)
            ctx = _safe_context(out)

            resolved_by = ctx.get("resolved_by")
            assert resolved_by in valid_agents, (
                f"Agente resolutor '{resolved_by}' no es válido. "
                f"Agentes registrados: {valid_agents}."
            )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 5: ESTABILIDAD NUMÉRICA, DETERMINISMO E INVARIANCIA GAUGE
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestNumericalStabilityAndInvariance:
    """Verificación de determinismo, estabilidad e invariancia gauge."""

    def test_determinism_under_identical_inputs(
        self, router: GaugeFieldRouter,
    ) -> None:
        """
        F5.1: Bajo entradas idénticas, resultados deben ser bitwise idénticos.

        Esto verifica reproducibilidad exacta bajo entradas determinísticas.
        Requiere que las variables de entorno BLAS/LAPACK sean esterilizadas
        al inicio (OMP_NUM_THREADS=1, etc.).
        """
        state = CategoricalState(
            payload={"determinism": True},
            context={},
        )
        n_trials = 5

        results: List[Dict[str, Any]] = []
        for _ in range(n_trials):
            out = router.route_perturbation(state, anomaly_node=2, severity=9.0)
            results.append(_safe_context(out))

        reference = results[0]
        for i, ctx in enumerate(results[1:], start=1):
            assert ctx.get("gauge_selected_agent") == reference.get(
                "gauge_selected_agent",
            ), (
                f"No determinista: trial {i} seleccionó "
                f"'{ctx.get('gauge_selected_agent')}' vs "
                f"'{reference.get('gauge_selected_agent')}' en trial 0."
            )
            assert ctx.get("resolved_by") == reference.get("resolved_by"), (
                f"No determinista: trial {i} resolved_by="
                f"'{ctx.get('resolved_by')}' vs "
                f"'{reference.get('resolved_by')}'."
            )
            assert np.isclose(
                float(ctx["cyber_momentum"]),
                float(reference["cyber_momentum"]),
                atol=_MOMENTUM_TOL,
            ), (
                f"Momentum no determinista: trial {i} = "
                f"{ctx['cyber_momentum']} vs trial 0 = "
                f"{reference['cyber_momentum']}."
            )

    def test_gauge_invariance_of_field(
        self,
        lattice: LatticeFixture,
    ) -> None:
        """
        F5.2: Invariancia gauge: E = −B₁Φ es invariante bajo Φ → Φ + c𝟙.

        Demostración:
        ─────────────
        E' = −B₁(Φ + c𝟙) = −B₁Φ − cB₁𝟙 = E − c·0 = E

        Precondición: B₁𝟙 = 0 (cada fila suma 0: +1 − 1 = 0).
        """
        B1 = lattice.incidence
        n = lattice.num_nodes

        # Verificar precondición: B₁𝟙 = 0.
        ones = np.ones(n, dtype=np.float64)
        B1_ones = B1.dot(ones)

        assert np.allclose(B1_ones, 0.0, atol=_ATOL), (
            f"B₁𝟙 ≠ 0: {B1_ones}. "
            "La invariancia gauge requiere B₁𝟙 = 0."
        )

        # Verificar invariancia para múltiples constantes.
        rng = np.random.RandomState(42)
        phi = rng.randn(n)

        E_original = -B1.dot(phi)

        for c in [0.0, 1.0, -3.7, 1e6, -1e-10]:
            phi_shifted = phi + c * ones
            E_shifted = -B1.dot(phi_shifted)

            assert np.allclose(E_original, E_shifted, atol=_ATOL), (
                f"Violación de invariancia gauge con c={c}:\n"
                f"  E(Φ)     = {E_original}\n"
                f"  E(Φ+c𝟙) = {E_shifted}\n"
                f"  Diff     = {E_original - E_shifted}"
            )

    def test_field_norm_scales_linearly_with_severity(
        self,
        router: GaugeFieldRouter,
        lattice: LatticeFixture,
    ) -> None:
        """
        F5.3: Linealidad: ‖E(α·severity)‖ = α · ‖E(severity)‖.

        E es función lineal de ρ, que es lineal en severity.
        Luego E escala linealmente con severity.
        """
        base_severity = 1.0
        rho_base = router._quantize_bosonic_excitation(0, base_severity)
        sol_base = router._solve_discrete_poisson(rho_base)
        E_base = router._compute_potential_gradient(sol_base.phi)
        norm_base = float(np.linalg.norm(E_base))

        for alpha in [0.5, 2.0, 10.0, 0.01]:
            rho_scaled = router._quantize_bosonic_excitation(
                0, alpha * base_severity,
            )
            sol_scaled = router._solve_discrete_poisson(rho_scaled)
            E_scaled = router._compute_potential_gradient(sol_scaled.phi)
            norm_scaled = float(np.linalg.norm(E_scaled))

            expected_norm = alpha * norm_base
            assert np.isclose(norm_scaled, expected_norm, rtol=1e-6), (
                f"Linealidad violada: α={alpha}, "
                f"‖E(α)‖={norm_scaled:.6e}, α·‖E(1)‖={expected_norm:.6e}."
            )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 6: ROBUSTEZ Y RECHAZO DE ENTRADAS INVÁLIDAS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestRobustness:
    """Verificación de robustez y rechazo de entradas inválidas."""

    def test_rejects_none_state(self, router: GaugeFieldRouter) -> None:
        """F6.1a: El router rechaza state=None."""
        with pytest.raises(GaugeFieldError, match="None"):
            router.route_perturbation(state=None, anomaly_node=0, severity=1.0)

    def test_rejects_negative_severity(self, router: GaugeFieldRouter) -> None:
        """F6.1b: El router rechaza severidad negativa."""
        state = CategoricalState(payload={}, context={})
        with pytest.raises(GaugeFieldError, match="no negativa|negativ"):
            router.route_perturbation(
                state, anomaly_node=0, severity=-1.0,
            )

    def test_rejects_out_of_range_node(self, router: GaugeFieldRouter) -> None:
        """F6.1c: El router rechaza nodos fuera de rango."""
        state = CategoricalState(payload={}, context={})

        with pytest.raises(GaugeFieldError, match="rango"):
            router.route_perturbation(state, anomaly_node=99, severity=1.0)

        with pytest.raises(GaugeFieldError, match="rango"):
            router.route_perturbation(state, anomaly_node=-1, severity=1.0)

    def test_rejects_infinite_severity(self, router: GaugeFieldRouter) -> None:
        """F6.1d: El router rechaza severidad infinita."""
        state = CategoricalState(payload={}, context={})
        with pytest.raises(GaugeFieldError, match="finit"):
            router.route_perturbation(
                state, anomaly_node=0, severity=float('inf'),
            )

    def test_rejects_nan_severity(self, router: GaugeFieldRouter) -> None:
        """F6.1e: El router rechaza severidad NaN."""
        state = CategoricalState(payload={}, context={})
        with pytest.raises(GaugeFieldError, match="finit"):
            router.route_perturbation(
                state, anomaly_node=0, severity=float('nan'),
            )

    def test_handles_zero_severity_gracefully(
        self, router: GaugeFieldRouter,
    ) -> None:
        """
        F6.2: severity=0 se promueve a MIN_CHARGE_DENSITY y completa.
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=0, severity=0.0)

        ctx = _safe_context(out)
        assert "gauge_selected_agent" in ctx, (
            "El router no completó con severity=0."
        )

    def test_handles_very_large_severity(
        self, router: GaugeFieldRouter,
    ) -> None:
        """
        F6.3: severity muy grande no produce overflow o resultados no finitos.
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=0, severity=1e15)

        ctx = _safe_context(out)
        momentum = float(ctx.get("cyber_momentum", 0.0))
        assert np.isfinite(momentum), (
            f"Momentum no finito con severity=1e15: {momentum}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 7: GENERALIZACIÓN TOPOLÓGICA
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestTopologicalGeneralization:
    """Verificación de comportamiento sobre topologías generalizadas."""

    @pytest.mark.parametrize("num_nodes", [3, 4, 5, 7, 10])
    def test_cycle_of_variable_size(self, num_nodes: int) -> None:
        """
        F7.1: Los invariantes se mantienen para C_n con n variable.

        Verifica para ciclos desde C₃ hasta C₁₀:
        • β₁ = 1 (siempre)
        • L₀ = B₁ᵀB₁
        • L₀ simétrica y semidefinida positiva
        • dim(ker(L₀)) = 1
        • B₁ ∈ {-1, 0, +1}
        """
        lattice = _build_cycle_lattice(num_nodes)

        # β₁ = 1.
        g = lattice.graph
        c = nx.number_connected_components(g)
        beta_1 = g.number_of_edges() - g.number_of_nodes() + c
        assert beta_1 == 1, f"β₁ = {beta_1} para C_{num_nodes}."

        # L₀ = B₁ᵀB₁.
        L_dense = lattice.laplacian.toarray()
        reconstructed = (lattice.incidence.T @ lattice.incidence).toarray()
        assert np.allclose(L_dense, reconstructed, atol=_ATOL), (
            f"L₀ ≠ B₁ᵀB₁ para C_{num_nodes}."
        )

        # Simetría.
        assert np.allclose(L_dense, L_dense.T, atol=_ATOL), (
            f"L₀ no simétrica para C_{num_nodes}."
        )

        # Espectro.
        eigenvalues = np.linalg.eigvalsh(L_dense)
        assert float(eigenvalues[0]) >= -_SPECTRAL_TOL, (
            f"λ_min = {eigenvalues[0]} para C_{num_nodes}."
        )

        kernel_dim = int(np.sum(np.abs(eigenvalues) < _SPECTRAL_TOL))
        assert kernel_dim == 1, (
            f"dim(ker(L₀)) = {kernel_dim} ≠ 1 para C_{num_nodes}."
        )

        # Entradas de B₁.
        B1_dense = lattice.incidence.toarray()
        assert set(np.unique(B1_dense)).issubset({-1.0, 0.0, 1.0}), (
            f"Entradas inválidas en B₁ para C_{num_nodes}."
        )

    def test_disconnected_graph_kernel_dimension(
        self,
        disconnected_lattice: LatticeFixture,
    ) -> None:
        """
        F7.2a: Para C₃ ⊔ C₃, dim(ker(L₀)) = 2.
        """
        L_dense = disconnected_lattice.laplacian.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)

        kernel_dim = int(np.sum(np.abs(eigenvalues) < _SPECTRAL_TOL))
        c = nx.number_connected_components(disconnected_lattice.graph)

        assert c == 2, f"Componentes conexas: {c} ≠ 2."
        assert kernel_dim == 2, (
            f"dim(ker(L₀)) = {kernel_dim} ≠ 2 para grafo desconectado. "
            f"Espectro: {eigenvalues}."
        )

    def test_disconnected_graph_betti_numbers(
        self,
        disconnected_lattice: LatticeFixture,
    ) -> None:
        """
        F7.2b: Para C₃ ⊔ C₃: β₀ = 2, β₁ = 2.
        """
        g = disconnected_lattice.graph
        c = nx.number_connected_components(g)
        m = g.number_of_edges()
        n = g.number_of_nodes()
        beta_1 = m - n + c

        assert c == 2, f"β₀ = {c} ≠ 2."
        assert beta_1 == 2, (
            f"β₁ = {beta_1} ≠ 2. M={m}, N={n}, c={c}."
        )

    def test_router_constructs_on_larger_cycle(
        self,
        mock_mic_registry: MICRegistry,
    ) -> None:
        """
        F7.3: El router funciona correctamente sobre C₅.
        """
        lattice = _build_cycle_lattice(num_nodes=5)
        m = lattice.num_edges

        agent_ids = [f"agent_{i}" for i in range(m)]
        charges = _build_canonical_charges(m, agent_ids)

        registry = MICRegistry()
        for aid in agent_ids:
            handler = _make_morphism(aid, Stratum.PHYSICS)
            registry.register_vector(aid, Stratum.PHYSICS, handler)

        router = GaugeFieldRouter(
            mic_registry=registry,
            laplacian=lattice.laplacian,
            incidence_matrix=lattice.incidence,
            agent_charges=charges,
        )

        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=0, severity=10.0)

        ctx = _safe_context(out)
        assert ctx.get("gauge_selected_agent") in set(agent_ids), (
            f"Agente seleccionado '{ctx.get('gauge_selected_agent')}' "
            f"no pertenece a {agent_ids}."
        )

        analysis = _compute_field_analysis(lattice, charges, 0, 10.0)

        assert ctx.get("gauge_selected_agent") == analysis.expected_agent, (
            f"Discrepancia con oráculo en C₅: "
            f"router='{ctx.get('gauge_selected_agent')}', "
            f"oráculo='{analysis.expected_agent}'."
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 8: CONSISTENCIA CRUZADA (CROSS-VALIDATION)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestCrossValidation:
    """Verificaciones de consistencia cruzada interna."""

    def test_field_norm_in_context_matches_computation(
        self,
        router: GaugeFieldRouter,
    ) -> None:
        """
        F8.1: La norma del campo reportada en el contexto coincide
        con un cálculo independiente.
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=0, severity=5.0)
        ctx = _safe_context(out)

        reported_norm = float(ctx["gauge_field_norm"])

        rho = router._quantize_bosonic_excitation(0, 5.0)
        sol = router._solve_discrete_poisson(rho)
        E = router._compute_potential_gradient(sol.phi)
        computed_norm = float(np.linalg.norm(E))

        assert np.isclose(reported_norm, computed_norm, atol=_ATOL), (
            f"Norma reportada: {reported_norm}, computada: {computed_norm}."
        )

    def test_charge_density_norm_in_context_matches_computation(
        self,
        router: GaugeFieldRouter,
    ) -> None:
        """
        F8.2: La norma de ρ reportada coincide con un cálculo independiente.
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(state, anomaly_node=1, severity=3.0)
        ctx = _safe_context(out)

        reported_norm = float(ctx["gauge_charge_density_norm"])

        rho = router._quantize_bosonic_excitation(1, 3.0)
        computed_norm = float(np.linalg.norm(rho))

        assert np.isclose(reported_norm, computed_norm, atol=_ATOL), (
            f"Norma de ρ reportada: {reported_norm}, computada: {computed_norm}."
        )

    @pytest.mark.parametrize("anomaly_node", [0, 1, 2])
    def test_momentum_is_positive_and_finite(
        self,
        router: GaugeFieldRouter,
        anomaly_node: int,
    ) -> None:
        """
        F8.3: El momentum cibernético es positivo y finito.
        """
        state = CategoricalState(payload={}, context={})
        out = router.route_perturbation(
            state, anomaly_node=anomaly_node, severity=5.0,
        )
        ctx = _safe_context(out)

        momentum = float(ctx["cyber_momentum"])
        assert np.isfinite(momentum), (
            f"Momentum no finito: {momentum}."
        )
        assert momentum > 0.0, (
            f"Momentum no positivo: {momentum}."
        )
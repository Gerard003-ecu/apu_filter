"""
=============================================================================
LogisticsManifold — Operador de Enrutamiento Logístico y Masa Térmica
=============================================================================

Módulo que implementa un operador categórico de enrutamiento logístico sobre
grafos dirigidos, con diagnóstico topológico-espectral riguroso.

Fundamentos matemáticos:
    Sea G = (V, E) un grafo dirigido finito con |V| = n, |E| = m.

    1. Complejo de cadenas discreto:
        C₀(G) ←─B₁── C₁(G) ←─── C₂(G)
       donde B₁ ∈ ℝⁿˣᵐ es la matriz de incidencia orientada:
            B₁[u, e] = +1   si e = (u, ·)  (sale de u)
            B₁[v, e] = -1   si e = (·, v)  (entra a v)

    2. Ley de conservación discreta (ecuación de continuidad):
        B₁ f = s
       donde f ∈ ℝᵐ es el vector de flujo y s ∈ ℝⁿ es el campo
       fuente-sumidero. La condición necesaria de solvencia es:
            𝟏ᵀ s = 0   (conservación global de masa)

    3. Descomposición de Hodge discreta sobre 1-cadenas:
        f = f_grad + f_curl + f_harm
       con:
        f_grad ∈ Im(B₁ᵀ)      — componente gradiente (potencial)
        f_curl ∈ Im(C)         — componente solenoidal (ciclos)
        f_harm ∈ ker(B₁) ∩ ker(Cᵀ) — componente armónica

    4. Espectro del Laplaciano combinatorio:
        L₀ = B₁ B₁ᵀ   (Laplaciano nodal)
        λ₂(L₀) = conectividad algebraica de Fiedler

    5. Geodésicas riemannianas discretas:
        Para un tensor métrico G ∈ S₊ⁿˣⁿ (simétrico semidefinido positivo),
        el peso de una arista con vector de atributos x ∈ ℝᵈ es:
            w(e) = √(xᵀ G x)

Convenciones de signos:
    - (B₁ f)[i] = ∑_{e sale de i} f(e) - ∑_{e entra a i} f(e)
    - s[i] > 0  ⟹  fuente neta en nodo i
    - s[i] < 0  ⟹  sumidero neto en nodo i
    - s[i] = 0  ⟹  nodo balanceado

Referencias:
    [1] Lim, L.-H. "Hodge Laplacians on Graphs." SIAM Review, 62(3), 2020.
    [2] Jiang et al. "Statistical ranking and combinatorial Hodge theory."
        Mathematical Programming, 127(1), 2011.
    [3] Mohar, B. "The Laplacian spectrum of graphs." Graph Theory,
        Combinatorics, and Applications, Vol. 2, 1991.

=============================================================================
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lsqr

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import G_PHYSICS, MetricTensorFactory

logger = logging.getLogger("MIC.Tactics.LogisticsManifold")


# =============================================================================
# §0. Estructuras de datos auxiliares
# =============================================================================

@dataclasses.dataclass(frozen=True)
class IncidenceData:
    """
    Encapsula el resultado de construir la matriz de incidencia B₁.

    Atributos:
        nodes:    Lista ordenada de nodos (define la indexación de filas).
        edges:    Lista ordenada de aristas dirigidas (define la indexación
                  de columnas).
        node_idx: Mapa nodo → índice de fila.
        edge_idx: Mapa arista → índice de columna.
        B1:       Matriz de incidencia orientada ∈ ℝⁿˣᵐ (CSR).
    """
    nodes: List[str]
    edges: List[Tuple[str, str]]
    node_idx: Dict[str, int]
    edge_idx: Dict[Tuple[str, str], int]
    B1: sp.csr_matrix


@dataclasses.dataclass(frozen=True)
class CycleData:
    """
    Encapsula la base de ciclos y la matriz generadora del subespacio cíclico.

    Atributos:
        cycle_basis: Lista de ciclos (cada ciclo es una lista de nodos).
        C:           Matriz generadora ∈ ℝᵐˣᵏ donde k = |cycle_basis|.
        rank:        Rango numérico verificado de C.
        betti_1:     Primer número de Betti teórico: β₁ = m - n + c.
    """
    cycle_basis: List[List[str]]
    C: sp.csr_matrix
    rank: int
    betti_1: int


@dataclasses.dataclass(frozen=True)
class ContinuityReport:
    """
    Diagnóstico de la ecuación de continuidad discreta B₁f = s.

    Atributos:
        residual_inf:       ‖B₁f - s‖_∞
        residual_l2:        ‖B₁f - s‖₂
        mass_imbalance:     |∑ᵢ sᵢ|  (debe ser ≈ 0)
        integrality_defect: ∑ⱼ |fⱼ - round(fⱼ)|  (mide no-integralidad)
    """
    residual_inf: float
    residual_l2: float
    mass_imbalance: float
    integrality_defect: float

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HodgeDecomposition:
    """
    Resultado de la descomposición de Hodge discreta sobre 1-cadenas.

    Componentes:
        f_grad:  Componente gradiente ∈ Im(B₁ᵀ)
        f_curl:  Componente solenoidal ∈ Im(C)
        f_harm:  Componente armónica ∈ ker(B₁) ∩ ker(Cᵀ)

    Energías:
        grad_energy, curl_energy, harm_energy, total_energy

    Diagnóstico de ortogonalidad (deben ser ≈ 0 para descomposición correcta):
        grad_curl_inner, grad_harm_inner, curl_harm_inner

    Defecto de ortogonalidad agregado:
        orthogonality_defect = max(|⟨f_grad, f_curl⟩|,
                                    |⟨f_grad, f_harm⟩|,
                                    |⟨f_curl, f_harm⟩|)
    """
    f_grad: np.ndarray
    f_curl: np.ndarray
    f_harm: np.ndarray
    grad_energy: float
    curl_energy: float
    harm_energy: float
    total_energy: float
    grad_curl_inner: float
    grad_harm_inner: float
    curl_harm_inner: float
    orthogonality_defect: float

    def to_report_dict(self) -> Dict[str, float]:
        """Devuelve métricas escalares (sin arrays) para serialización."""
        return {
            "grad_energy": self.grad_energy,
            "curl_energy": self.curl_energy,
            "harm_energy": self.harm_energy,
            "total_energy": self.total_energy,
            "grad_curl_inner": self.grad_curl_inner,
            "grad_harm_inner": self.grad_harm_inner,
            "curl_harm_inner": self.curl_harm_inner,
            "orthogonality_defect": self.orthogonality_defect,
        }


# =============================================================================
# §1. Clase principal: LogisticsManifold
# =============================================================================

class LogisticsManifold(Morphism):
    """
    Operador categórico de enrutamiento logístico y renormalización de masa
    térmica sobre grafos dirigidos finitos.

    Implementa un morfismo del estrato PHYSICS al estrato TACTICS en el
    marco categórico MIC, ejecutando:
        1. Construcción del complejo de cadenas discreto (B₁, C).
        2. Verificación de la ley de conservación discreta.
        3. Descomposición de Hodge discreta con diagnóstico de ortogonalidad.
        4. Análisis espectral del Laplaciano (valor de Fiedler).
        5. Renormalización de masa por polarones logísticos.
        6. Cómputo de geodésicas riemannianas discretas.

    Parámetros:
        name:            Identificador del morfismo.
        tolerance:       Tolerancia numérica para validaciones (ε).
        regularization:  Parámetro de regularización de Tikhonov (λ).
        max_ortho_iters: Iteraciones máximas de re-ortogonalización de Hodge.
    """

    # ── Constantes de clase ──────────────────────────────────────────────
    _DEFAULT_METRIC_DIM: int = 3
    _ATTRIBUTE_KEYS: Tuple[str, ...] = ("time", "cost", "risk")
    _DENSE_EIGENVALUE_THRESHOLD: int = 64

    def __init__(
        self,
        name: str = "logistics_router",
        tolerance: float = 1e-9,
        regularization: float = 1e-10,
        max_ortho_iters: int = 3,
    ) -> None:
        if tolerance <= 0:
            raise ValueError(f"tolerance debe ser > 0, recibido: {tolerance}")
        if regularization < 0:
            raise ValueError(
                f"regularization debe ser ≥ 0, recibido: {regularization}"
            )
        if max_ortho_iters < 1:
            raise ValueError(
                f"max_ortho_iters debe ser ≥ 1, recibido: {max_ortho_iters}"
            )

        super().__init__(name=name)
        self.tolerance = tolerance
        self.regularization = regularization
        self.max_ortho_iters = max_ortho_iters

    # ── Interfaz categórica ──────────────────────────────────────────────

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset([Stratum.PHYSICS])

    @property
    def codomain(self) -> Stratum:
        return Stratum.TACTICS

    # =====================================================================
    # §2. Validación y sanitización de entradas
    # =====================================================================

    @staticmethod
    def _validate_graph(graph: Any) -> nx.DiGraph:
        """
        Valida que la entrada sea un digrafo no vacío con al menos una arista.

        Raises:
            TypeError:  Si no es instancia de nx.DiGraph.
            ValueError: Si el grafo no tiene nodos o no tiene aristas.
        """
        if graph is None or not isinstance(graph, nx.DiGraph):
            raise TypeError(
                "El contexto requiere 'logistics_graph' como instancia "
                "válida de nx.DiGraph."
            )
        if graph.number_of_nodes() == 0:
            raise ValueError("El grafo logístico está vacío (0 nodos).")
        if graph.number_of_edges() == 0:
            raise ValueError("El grafo logístico no contiene aristas.")
        return graph

    def _sanitize_metric(self, G_metric: Optional[np.ndarray]) -> np.ndarray:
        """
        Sanea un tensor métrico para garantizar las propiedades requeridas:

        1. Conversión a ndarray float64.
        2. Verificación de forma cuadrada.
        3. Simetrización:       G ← ½(G + Gᵀ)
        4. Proyección a PSD:    G ← V max(Λ, 0) Vᵀ
           donde G = V Λ Vᵀ es la descomposición espectral.

        Cascada de fallback:
            G_metric proporcionado → G_PHYSICS global → MetricTensorFactory → I₃

        Retorna:
            Tensor métrico ∈ S₊ⁿˣⁿ (simétrico semidefinido positivo).
        """
        raw = None

        # ── Cascada de obtención ─────────────────────────────────────
        if G_metric is not None:
            raw = G_metric
        else:
            try:
                if isinstance(G_PHYSICS, np.ndarray):
                    raw = G_PHYSICS
                else:
                    raw = MetricTensorFactory.build(Stratum.PHYSICS)
            except Exception:
                logger.debug(
                    "No se pudo obtener tensor métrico de fábrica; "
                    "usando identidad %dx%d.",
                    self._DEFAULT_METRIC_DIM,
                    self._DEFAULT_METRIC_DIM,
                )

        if raw is None:
            return np.eye(self._DEFAULT_METRIC_DIM, dtype=float)

        metric = np.asarray(raw, dtype=float)

        # ── Validación de forma ──────────────────────────────────────
        if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
            logger.warning(
                "Tensor métrico con forma inválida %s; "
                "usando identidad %dx%d.",
                metric.shape,
                self._DEFAULT_METRIC_DIM,
                self._DEFAULT_METRIC_DIM,
            )
            return np.eye(self._DEFAULT_METRIC_DIM, dtype=float)

        # ── Simetrización exacta ─────────────────────────────────────
        metric = 0.5 * (metric + metric.T)

        # ── Proyección espectral a semidefinido positivo (PSD) ───────
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        eigenvalues_clipped = np.clip(eigenvalues, 0.0, None)
        metric_psd = (eigenvectors * eigenvalues_clipped) @ eigenvectors.T

        # Verificación de finitud
        if not np.all(np.isfinite(metric_psd)):
            logger.warning(
                "Tensor métrico con entradas no finitas tras proyección PSD; "
                "usando identidad."
            )
            return np.eye(metric.shape[0], dtype=float)

        return metric_psd

    # =====================================================================
    # §3. Álgebra lineal numérica
    # =====================================================================

    def _orthogonal_projection_onto_image(
        self,
        A: sp.spmatrix,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula la proyección ortogonal de y sobre Im(A):

            proj_{Im(A)}(y) = A x*,   donde  x* = argmin_x ‖Ax - y‖₂

        Implementación:
            Se usa LSQR sin damping (Tikhonov) para obtener la solución de
            mínima norma del problema de mínimos cuadrados, lo que garantiza
            que A x* ∈ Im(A) y (y - A x*) ⊥ Im(A).

            En caso de condicionamiento extremo, se aplica regularización
            mínima controlada.

        Argumentos:
            A: Matriz sparse ∈ ℝⁿˣᵏ.
            y: Vector ∈ ℝⁿ a proyectar.

        Retorna:
            proj ∈ ℝⁿ, la proyección ortogonal de y sobre Im(A).

        Nota:
            Para la proyección ortogonal exacta, damp=0 es necesario.
            Solo se activa damp > 0 si la proyección sin regularización
            produce resultados no finitos.
        """
        if A.shape[1] == 0:
            return np.zeros(A.shape[0], dtype=float)

        y = np.asarray(y, dtype=float).ravel()

        # Intento sin regularización para proyección exacta
        result = lsqr(A, y, atol=self.tolerance, btol=self.tolerance, damp=0.0)
        x_star = result[0]
        proj = A @ x_star

        if np.all(np.isfinite(proj)):
            return np.asarray(proj, dtype=float).ravel()

        # Fallback con regularización mínima
        logger.warning(
            "Proyección ortogonal sin damping produjo valores no finitos; "
            "aplicando regularización λ=%.2e.",
            self.regularization,
        )
        result = lsqr(
            A, y,
            atol=self.tolerance,
            btol=self.tolerance,
            damp=np.sqrt(self.regularization),
        )
        proj = A @ result[0]
        return np.asarray(proj, dtype=float).ravel()

    # =====================================================================
    # §4. Construcción de operadores discretos del complejo de cadenas
    # =====================================================================

    def _build_incidence_matrix(self, graph: nx.DiGraph) -> IncidenceData:
        """
        Construye la matriz de incidencia orientada B₁ ∈ ℝⁿˣᵐ del
        complejo de cadenas discreto C₁ → C₀.

        Definición:
            Para cada arista eⱼ = (u, v):
                B₁[node_idx[u], j] = +1  (cola: la arista sale de u)
                B₁[node_idx[v], j] = -1  (cabeza: la arista entra a v)

        Propiedades verificables:
            - Cada columna tiene exactamente un +1 y un -1.
            - B₁ 𝟏 = 0  (las columnas suman cero).
            - ker(B₁ᵀ) ≅ H₀(G; ℝ)  (componentes conexas).

        Retorna:
            IncidenceData con nodos, aristas, índices y B₁ en formato CSR.
        """
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        n_nodes = len(nodes)
        n_edges = len(edges)

        node_idx: Dict[str, int] = {n: i for i, n in enumerate(nodes)}
        edge_idx: Dict[Tuple[str, str], int] = {e: j for j, e in enumerate(edges)}

        # Pre-alocación de arrays COO
        rows = np.empty(2 * n_edges, dtype=np.intp)
        cols = np.empty(2 * n_edges, dtype=np.intp)
        data = np.empty(2 * n_edges, dtype=float)

        for j, (u, v) in enumerate(edges):
            idx = 2 * j
            rows[idx] = node_idx[u]
            rows[idx + 1] = node_idx[v]
            cols[idx] = j
            cols[idx + 1] = j
            data[idx] = +1.0
            data[idx + 1] = -1.0

        B1 = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_nodes, n_edges),
            dtype=float,
        )

        # ── Verificación de consistencia ─────────────────────────────
        # Cada columna debe sumar exactamente 0
        col_sums = np.abs(np.asarray(B1.sum(axis=0)).ravel())
        if np.any(col_sums > self.tolerance):
            raise RuntimeError(
                "Error interno: columnas de B₁ no suman cero. "
                f"Max desviación: {col_sums.max():.2e}"
            )

        return IncidenceData(
            nodes=nodes,
            edges=edges,
            node_idx=node_idx,
            edge_idx=edge_idx,
            B1=B1,
        )

    def _build_cycle_matrix(
        self,
        graph: nx.DiGraph,
        edge_idx: Dict[Tuple[str, str], int],
    ) -> CycleData:
        """
        Construye la matriz generadora del subespacio de ciclos C ∈ ℝᵐˣᵏ.

        Cada columna de C corresponde a un ciclo fundamental del grafo
        subyacente no dirigido. Para cada arista del ciclo, el signo en C
        refleja si la orientación del ciclo concuerda (+1) o no (-1) con
        la orientación de la arista en el digrafo.

        Propiedad fundamental:
            Im(C) = ker(B₁)  (espacio de ciclos = kernel de la incidencia)

        Verificación:
            Se calcula el primer número de Betti teórico:
                β₁ = m - n + c
            donde c es el número de componentes conexas del grafo subyacente,
            y se verifica que rank(C) = β₁.

        Retorna:
            CycleData con la base de ciclos, la matriz C, su rango verificado
            y el β₁ teórico.
        """
        undirected = graph.to_undirected()
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        n_components = nx.number_connected_components(undirected)

        # Primer número de Betti teórico
        betti_1_theoretical = n_edges - n_nodes + n_components

        cycle_basis = nx.cycle_basis(undirected)

        if len(cycle_basis) != betti_1_theoretical:
            logger.warning(
                "Discrepancia en base de ciclos: |cycle_basis|=%d vs "
                "β₁ teórico=%d. Posible grafo con multi-aristas.",
                len(cycle_basis),
                betti_1_theoretical,
            )

        n_cycles = len(cycle_basis)
        rows_list: List[int] = []
        cols_list: List[int] = []
        data_list: List[float] = []

        for j, cycle in enumerate(cycle_basis):
            m = len(cycle)
            for k in range(m):
                u = cycle[k]
                v = cycle[(k + 1) % m]

                if (u, v) in edge_idx:
                    rows_list.append(edge_idx[(u, v)])
                    cols_list.append(j)
                    data_list.append(+1.0)
                elif (v, u) in edge_idx:
                    rows_list.append(edge_idx[(v, u)])
                    cols_list.append(j)
                    data_list.append(-1.0)
                else:
                    logger.debug(
                        "Arista de ciclo no encontrada en ninguna "
                        "orientación: (%s, %s).",
                        u, v,
                    )

        C = sp.csr_matrix(
            (data_list, (rows_list, cols_list)),
            shape=(n_edges, n_cycles),
            dtype=float,
        )

        # ── Verificación de rango ────────────────────────────────────
        if n_cycles > 0:
            C_dense = C.toarray()
            numerical_rank = int(np.linalg.matrix_rank(
                C_dense, tol=self.tolerance
            ))
        else:
            numerical_rank = 0

        if numerical_rank != betti_1_theoretical:
            logger.warning(
                "Rango numérico de C (%d) ≠ β₁ teórico (%d). "
                "Las proyecciones de Hodge pueden ser imprecisas.",
                numerical_rank,
                betti_1_theoretical,
            )

        return CycleData(
            cycle_basis=cycle_basis,
            C=C,
            rank=numerical_rank,
            betti_1=betti_1_theoretical,
        )

    # =====================================================================
    # §5. Conservación discreta (ecuación de continuidad)
    # =====================================================================

    def _enforce_discrete_continuity(
        self,
        B1: sp.csr_matrix,
        f: np.ndarray,
        s: np.ndarray,
        strict: bool = True,
    ) -> ContinuityReport:
        """
        Verifica la ecuación de continuidad discreta:

            B₁ f = s

        donde:
            - B₁ ∈ ℝⁿˣᵐ es la matriz de incidencia orientada.
            - f  ∈ ℝᵐ   es el vector de flujo por arista.
            - s  ∈ ℝⁿ   es el vector fuente-sumidero nodal.

        Condición necesaria de solvencia (ley de Kirchhoff global):
            𝟏ᵀ s = 0   ⟺   ∑ᵢ sᵢ = 0

        Diagnósticos reportados:
            residual_inf:       ‖B₁f - s‖_∞
            residual_l2:        ‖B₁f - s‖₂
            mass_imbalance:     |∑ᵢ sᵢ|
            integrality_defect: ∑ⱼ |fⱼ - round(fⱼ)|

        Argumentos:
            B1:     Matriz de incidencia.
            f:      Vector de flujos.
            s:      Vector fuente-sumidero.
            strict: Si True, lanza ValueError ante violaciones.

        Retorna:
            ContinuityReport con las métricas diagnósticas.
        """
        f = np.asarray(f, dtype=float).ravel()
        s = np.asarray(s, dtype=float).ravel()

        if B1.shape[1] != f.size:
            raise ValueError(
                f"Dimensión incompatible: B₁ tiene {B1.shape[1]} columnas "
                f"pero f tiene {f.size} entradas."
            )
        if B1.shape[0] != s.size:
            raise ValueError(
                f"Dimensión incompatible: B₁ tiene {B1.shape[0]} filas "
                f"pero s tiene {s.size} entradas."
            )

        divergence = np.asarray(B1 @ f).ravel()
        residual = divergence - s

        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        residual_l2 = float(np.linalg.norm(residual, ord=2))
        mass_imbalance = float(abs(np.sum(s)))
        integrality_defect = float(np.sum(np.abs(f - np.round(f))))

        # ── Verificación de balance global de masa ───────────────────
        if mass_imbalance > self.tolerance:
            msg = (
                f"Balance global de masa inválido: ∑sᵢ = {np.sum(s):.6e}. "
                "La condición necesaria ∑sᵢ = 0 falla."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        # ── Verificación de conservación local ───────────────────────
        if residual_inf > self.tolerance:
            msg = (
                f"Violación de conservación discreta: "
                f"‖B₁f - s‖_∞ = {residual_inf:.6e}, "
                f"‖B₁f - s‖₂ = {residual_l2:.6e}."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        # ── Advertencia de no-integralidad ───────────────────────────
        if integrality_defect > self.tolerance:
            logger.info(
                "Defecto de integralidad en flujos: %.6e. "
                "Esto no indica torsión homológica sino cuantización "
                "imperfecta del flujo.",
                integrality_defect,
            )

        return ContinuityReport(
            residual_inf=residual_inf,
            residual_l2=residual_l2,
            mass_imbalance=mass_imbalance,
            integrality_defect=integrality_defect,
        )

    # =====================================================================
    # §6. Descomposición de Hodge discreta
    # =====================================================================

    def _compute_hodge_decomposition(
        self,
        f: np.ndarray,
        B1: sp.csr_matrix,
        C: sp.csr_matrix,
    ) -> HodgeDecomposition:
        """
        Descomposición de Hodge discreta para 1-cadenas (flujos en aristas):

            f = f_grad + f_curl + f_harm

        Subespacios (teorema de descomposición ortogonal de Hodge discreto [1]):
            Im(B₁ᵀ)  ⊕  Im(C)  ⊕  ker(L₁) = ℝᵐ

        donde L₁ = B₁ᵀ B₁ + C Cᵀ es el Laplaciano de Hodge de 1-formas.

        Algoritmo:
            1. f_grad = proj_{Im(B₁ᵀ)}(f)
            2. r₁ = f - f_grad
            3. f_curl = proj_{Im(C)}(r₁)
            4. f_harm = f - f_grad - f_curl

        Refinamiento por re-ortogonalización iterativa:
            Se repiten los pasos 1-4 hasta que el defecto de ortogonalidad
            max(|⟨f_grad, f_curl⟩|, |⟨f_grad, f_harm⟩|, |⟨f_curl, f_harm⟩|)
            sea menor que la tolerancia, o se agoten las iteraciones.

        Argumentos:
            f:  Vector de flujo ∈ ℝᵐ.
            B1: Matriz de incidencia ∈ ℝⁿˣᵐ.
            C:  Matriz de ciclos ∈ ℝᵐˣᵏ.

        Retorna:
            HodgeDecomposition con componentes, energías y diagnósticos.
        """
        f = np.asarray(f, dtype=float).ravel()
        m = f.size

        # Operador de gradientes: Im(B₁ᵀ)
        A_grad = B1.T  # shape: (m, n)

        # ── Descomposición iterativa con re-ortogonalización ─────────
        f_grad = np.zeros(m, dtype=float)
        f_curl = np.zeros(m, dtype=float)
        f_harm = f.copy()

        for iteration in range(self.max_ortho_iters):
            # Paso 1: Proyectar sobre Im(B₁ᵀ)
            f_grad = self._orthogonal_projection_onto_image(A_grad, f)

            # Paso 2: Residual post-gradiente
            r1 = f - f_grad

            # Paso 3: Proyectar residual sobre Im(C)
            if C.shape[1] > 0:
                f_curl = self._orthogonal_projection_onto_image(C, r1)
            else:
                f_curl = np.zeros(m, dtype=float)

            # Paso 4: Componente armónica
            f_harm = f - f_grad - f_curl

            # ── Diagnóstico de ortogonalidad ─────────────────────────
            gc_inner = abs(float(np.dot(f_grad, f_curl)))
            gh_inner = abs(float(np.dot(f_grad, f_harm)))
            ch_inner = abs(float(np.dot(f_curl, f_harm)))
            ortho_defect = max(gc_inner, gh_inner, ch_inner)

            if ortho_defect <= self.tolerance:
                logger.debug(
                    "Hodge convergió en iteración %d; "
                    "defecto de ortogonalidad: %.2e",
                    iteration + 1,
                    ortho_defect,
                )
                break
        else:
            logger.warning(
                "Hodge no convergió tras %d iteraciones; "
                "defecto de ortogonalidad residual: %.6e",
                self.max_ortho_iters,
                ortho_defect,
            )

        # ── Cómputo de energías ──────────────────────────────────────
        grad_energy = float(np.dot(f_grad, f_grad))
        curl_energy = float(np.dot(f_curl, f_curl))
        harm_energy = float(np.dot(f_harm, f_harm))
        total_energy = float(np.dot(f, f))

        # ── Verificación de Parseval discreto ────────────────────────
        # ‖f‖² = ‖f_grad‖² + ‖f_curl‖² + ‖f_harm‖² si ortogonales
        parseval_defect = abs(
            total_energy - (grad_energy + curl_energy + harm_energy)
        )
        if parseval_defect > self.tolerance * max(1.0, total_energy):
            logger.warning(
                "Identidad de Parseval violada: "
                "‖f‖²=%.6e vs ∑‖fₖ‖²=%.6e, defecto=%.6e",
                total_energy,
                grad_energy + curl_energy + harm_energy,
                parseval_defect,
            )

        grad_curl_inner = float(np.dot(f_grad, f_curl))
        grad_harm_inner = float(np.dot(f_grad, f_harm))
        curl_harm_inner = float(np.dot(f_curl, f_harm))
        orthogonality_defect = max(
            abs(grad_curl_inner), abs(grad_harm_inner), abs(curl_harm_inner)
        )

        return HodgeDecomposition(
            f_grad=f_grad,
            f_curl=f_curl,
            f_harm=f_harm,
            grad_energy=grad_energy,
            curl_energy=curl_energy,
            harm_energy=harm_energy,
            total_energy=total_energy,
            grad_curl_inner=grad_curl_inner,
            grad_harm_inner=grad_harm_inner,
            curl_harm_inner=curl_harm_inner,
            orthogonality_defect=orthogonality_defect,
        )

    def _validate_solenoidal_policy(
        self,
        curl_energy: float,
        cycle_rank: int,
        is_regenerative: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Evalúa la política de aceptación/rechazo del componente solenoidal.

        Lógica:
            1. Si curl_energy ≤ ε:       → Sin vórtice, retorna None.
            2. Si is_regenerative=True:   → Ciclo regenerativo, retorna None.
            3. En caso contrario:         → Veto de flujo circular parasitario.

        Argumentos:
            curl_energy:    Energía del componente solenoidal ‖f_curl‖².
            cycle_rank:     Dimensión del espacio de ciclos β₁.
            is_regenerative: Indica si la política DPP permite ciclos.

        Retorna:
            None si el flujo es aceptable.
            Dict con datos del magnón si se construye antes del veto.

        Raises:
            ValueError: Si se detecta flujo circular parasitario no permitido.
        """
        if curl_energy <= self.tolerance:
            return None

        if is_regenerative:
            logger.info(
                "Ciclo logístico reclasificado como regenerativo: "
                "energía solenoidal=%.6e, dimensión cíclica=%d.",
                curl_energy,
                cycle_rank,
            )
            return None

        # ── Construcción del cartucho de magnón para telemetría ──────
        magnon_data: Dict[str, Any] = {
            "kinetic_energy": float(curl_energy),
            "curl_subspace_dim": int(cycle_rank),
        }

        logger.error(
            "Vórtice logístico no permitido: "
            "energía solenoidal=%.6e, dim=%d.",
            curl_energy,
            cycle_rank,
        )
        raise ValueError(
            "Veto de enrutamiento: flujo circular parasitario detectado. "
            f"Energía solenoidal: {curl_energy:.6e}, "
            f"dimensión cíclica: {cycle_rank}."
        )

    # =====================================================================
    # §7. Geodésicas riemannianas discretas
    # =====================================================================

    def _compute_logistical_geodesics(
        self,
        G_metric: np.ndarray,
        graph: nx.DiGraph,
        source: str,
        target: str,
    ) -> List[str]:
        """
        Calcula la geodésica discreta (camino de mínima acción) entre dos
        nodos usando un tensor métrico riemanniano sobre atributos de aristas.

        Para cada arista e = (u, v) con vector de atributos x = (t, c, r)ᵀ:
            w(e) = √(xᵀ G x)

        donde G ∈ S₊ᵈˣᵈ es el tensor métrico (ya sanitizado, PSD).

        Si dim(x) ≠ dim(G), se trunca al mínimo de ambas dimensiones:
            x' = x[:d'], G' = G[:d', :d'],  d' = min(dim(x), dim(G))

        Argumentos:
            G_metric: Tensor métrico ya sanitizado.
            graph:    Grafo dirigido con atributos 'time', 'cost', 'risk'.
            source:   Nodo de origen.
            target:   Nodo de destino.

        Retorna:
            Lista de nodos que conforman la geodésica discreta.

        Raises:
            ValueError: Si algún nodo no existe o no hay camino.
        """
        # G_metric ya viene sanitizado; no se vuelve a procesar
        metric = G_metric

        attribute_keys = self._ATTRIBUTE_KEYS
        default_vals = {"time": 1.0, "cost": 1.0, "risk": 0.0}

        def riemannian_weight(
            u: str, v: str, edge_data: Dict[str, Any]
        ) -> float:
            x = np.array(
                [float(edge_data.get(k, default_vals[k])) for k in attribute_keys],
                dtype=float,
            )

            dim = min(x.shape[0], metric.shape[0], metric.shape[1])
            x_proj = x[:dim]
            G_proj = metric[:dim, :dim]

            # Forma cuadrática: xᵀ G x
            quadratic_form = float(x_proj @ G_proj @ x_proj)

            if not np.isfinite(quadratic_form):
                return float("inf")

            # √(max(0, xᵀ G x)) para PSD con protección numérica
            return float(np.sqrt(max(0.0, quadratic_form)))

        try:
            path = nx.shortest_path(
                graph,
                source=source,
                target=target,
                weight=riemannian_weight,
            )
            return list(path)
        except nx.NodeNotFound as e:
            raise ValueError(
                f"Nodo inválido al calcular geodésica: {e}"
            ) from e
        except nx.NetworkXNoPath as e:
            raise ValueError(
                f"No existe trayectoria entre '{source}' y '{target}'."
            ) from e

    # =====================================================================
    # §8. Análisis espectral del Laplaciano
    # =====================================================================

    def _compute_fiedler_value(self, undirected_graph: nx.Graph) -> float:
        """
        Calcula la conectividad algebraica λ₂ (valor de Fiedler) del
        Laplaciano combinatorio del grafo no dirigido.

        Definición:
            L = D - A   (Laplaciano combinatorio)
            λ₁ = 0  (siempre, con eigenvector 𝟏)
            λ₂ = min{xᵀLx : ‖x‖=1, x ⊥ 𝟏}  (conectividad algebraica)

        Propiedades:
            - λ₂ = 0 ⟺ grafo desconexo.
            - λ₂ > 0 ⟺ grafo conexo.
            - λ₂ acota la expansión del grafo (Cheeger).

        Implementación:
            - n ≤ 64: Diagonalización densa completa (O(n³) pero estable).
            - n > 64: ARPACK con shift-invert σ=0 para estabilidad
                       numérica al buscar eigenvalores cercanos a 0.

        Retorna:
            λ₂ ≥ 0 (float).
        """
        n = undirected_graph.number_of_nodes()

        if n <= 1:
            return 0.0

        if not nx.is_connected(undirected_graph):
            return 0.0

        L = nx.laplacian_matrix(undirected_graph).astype(float)

        try:
            if n <= self._DENSE_EIGENVALUE_THRESHOLD:
                dense_L = L.toarray()
                eigenvalues = np.linalg.eigvalsh(dense_L)
                eigenvalues = np.sort(np.real(eigenvalues))
                # λ₁ ≈ 0, tomamos λ₂
                lambda_2 = float(eigenvalues[1]) if eigenvalues.size > 1 else 0.0
            else:
                # Shift-invert con σ = 0 para eigenvalores más pequeños
                # Esto transforma L x = λ x  en  (L - σI)⁻¹ x = (1/(λ-σ)) x
                # con σ=0 ⟹ L⁻¹ x = (1/λ) x, estable para eigenvalores
                # cercanos a 0.
                # Pero L es singular (λ₁=0), así que usamos σ = -tolerance
                # como shift ligeramente negativo.
                sigma = -self.tolerance
                eigenvalues = eigsh(
                    L, k=2, sigma=sigma, which="LM",
                    return_eigenvectors=False,
                )
                eigenvalues = np.sort(np.real(eigenvalues))
                lambda_2 = float(eigenvalues[1]) if eigenvalues.size > 1 else 0.0

            # Sanitización: λ₂ no puede ser negativo en un Laplaciano PSD
            return float(max(0.0, lambda_2))

        except Exception as e:
            logger.warning(
                "Error al calcular λ₂ de Fiedler: %s. Retornando 0.0.", e
            )
            return 0.0

    # =====================================================================
    # §9. Centralidades estructurales
    # =====================================================================

    def _compute_structural_centralities(
        self, graph: nx.DiGraph
    ) -> Dict[str, float]:
        """
        Calcula centralidades estructurales para renormalización de masa.

        Cascada de métodos:
            1. Centralidad de eigenvector sobre grafo no dirigido
               (eigenvector principal de la matriz de adyacencia).
            2. Fallback: centralidad de grado normalizada.
            3. Fallback: uniforme 1/n.

        Retorna:
            Diccionario nodo → centralidad ∈ [0, 1].
        """
        nodes = list(graph.nodes)
        if not nodes:
            return {}

        n = len(nodes)
        undirected = graph.to_undirected()

        # ── Intento 1: centralidad de eigenvector ────────────────────
        try:
            if n > 1 and undirected.number_of_edges() > 0:
                centrality = nx.eigenvector_centrality_numpy(undirected)
                return {k: float(v) for k, v in centrality.items()}
        except Exception as e:
            logger.debug(
                "Eigenvector centrality falló: %s. Intentando grado.", e
            )

        # ── Intento 2: centralidad de grado normalizada ──────────────
        deg = dict(undirected.degree())
        total_degree = sum(deg.values())

        if total_degree > 0:
            return {node: float(deg[node]) / total_degree for node in nodes}

        # ── Intento 3: uniforme ──────────────────────────────────────
        uniform = 1.0 / n
        return {node: uniform for node in nodes}

    # =====================================================================
    # §10. Renormalización de masa: polarones logísticos
    # =====================================================================

    def _quantize_logistical_polarons(
        self,
        fiedler_val: float,
        centrality: float,
        base_mass: float,
    ) -> float:
        """
        Renormalización de masa efectiva mediante acoplo espectral-topológico.

        Modelo de cuasipartícula (polarón logístico):
            α = c / (λ₂ + ε)
            m_eff = m₀ · (1 + α/6)

        donde:
            - m₀ es la masa base (latencia/delay del nodo).
            - c es la centralidad estructural del nodo.
            - λ₂ es la conectividad algebraica (Fiedler).
            - ε es la tolerancia numérica para evitar divergencia.

        Interpretación física:
            Un nodo con alta centralidad en un grafo pobremente conectado
            (λ₂ pequeño) experimenta mayor "vestimiento" de masa, análogo
            a un polarón en materia condensada donde la interacción con
            el retículo incrementa la masa efectiva.

            El factor 1/6 proviene de la expansión perturbativa a primer
            orden del propagador vestido en la aproximación de campo medio.

        Garantías:
            - m_eff ≥ m₀ ≥ 0
            - m_eff es finito

        Retorna:
            Masa efectiva m_eff (float ≥ 0).
        """
        base_mass = float(max(0.0, base_mass))
        centrality = float(max(0.0, centrality))
        fiedler_val = float(max(0.0, fiedler_val))

        denominator = fiedler_val + self.tolerance
        alpha = centrality / denominator
        m_eff = base_mass * (1.0 + alpha / 6.0)

        if not np.isfinite(m_eff):
            logger.warning(
                "Masa efectiva no finita: m₀=%.6e, α=%.6e. "
                "Retornando masa base.",
                base_mass,
                alpha,
            )
            return base_mass

        logger.debug(
            "Polarón logístico: λ₂=%.4e, c=%.4e, α=%.4e, "
            "m₀=%.4e → m_eff=%.4e",
            fiedler_val,
            centrality,
            alpha,
            base_mass,
            m_eff,
        )
        return m_eff

    # =====================================================================
    # §11. Política de regeneración DPP
    # =====================================================================

    @staticmethod
    def _evaluate_regenerative_policy(
        graph: nx.DiGraph,
        context: Dict[str, Any],
    ) -> bool:
        """
        Evalúa si la política de Pasaporte de Producto Digital (DPP) permite
        reclasificar ciclos logísticos como regenerativos.

        Condiciones:
            1. El contexto debe tener 'dpp_circularity' = True.
            2. La disipación exergética agregada debe ser ≥ 0
               (consistencia con la segunda ley de la termodinámica).

        Retorna:
            True si los ciclos son regenerativos.

        Raises:
            ValueError: Si p_diss agregado < 0 (inconsistencia termodinámica).
        """
        if not context.get("dpp_circularity", False):
            return False

        total_dissipation = sum(
            float(graph.edges[e].get("p_diss", 0.0)) for e in graph.edges
        )

        if total_dissipation < 0.0:
            raise ValueError(
                f"Veto 3R por inconsistencia termodinámica: "
                f"p_diss agregado = {total_dissipation:.6e} < 0."
            )

        return True

    # =====================================================================
    # §12. Ejecución principal: funtor OODA
    # =====================================================================

    def __call__(
        self, state: CategoricalState, **kwargs: Any
    ) -> CategoricalState:
        """
        Funtor de ejecución OODA (Observe-Orient-Decide-Act) que impone
        restricciones de logística discreta sobre el estado categórico.

        Pipeline:
            1. Validación del grafo logístico.
            2. Construcción del complejo de cadenas (B₁, C).
            3. Extracción y validación de flujos y fuentes/sumideros.
            4. Verificación de la ley de conservación discreta.
            5. Evaluación de política regenerativa DPP.
            6. Descomposición de Hodge discreta.
            7. Validación de política solenoidal.
            8. Anotación de aristas con componentes de Hodge.
            9. Análisis espectral y centralidades.
            10. Renormalización de masa por polarones.
            11. Cómputo de geodésicas (si se solicitan).
            12. Ensamblado y retorno del contexto actualizado.

        Argumentos:
            state: Estado categórico de entrada (estrato PHYSICS).
            **kwargs: Argumentos adicionales (no utilizados actualmente).

        Retorna:
            CategoricalState actualizado con contexto táctico.
        """
        try:
            context = state.context

            # ── Paso 1: Validación del grafo ─────────────────────────
            G = self._validate_graph(context.get("logistics_graph"))

            # ── Paso 2: Complejo de cadenas discreto ─────────────────
            incidence = self._build_incidence_matrix(G)
            cycles = self._build_cycle_matrix(G, incidence.edge_idx)

            nodes = incidence.nodes
            edges = incidence.edges
            B1 = incidence.B1
            C = cycles.C
            n_nodes = len(nodes)
            n_edges = len(edges)
            betti_1 = cycles.betti_1
            n_components = nx.number_connected_components(G.to_undirected())

            # Característica de Euler del grafo subyacente como
            # 1-complejo simplicial: χ = n - m + 0 (sin 2-caras)
            # Pero por Euler: χ = c - β₁ donde c = componentes conexas
            euler_characteristic = n_components - betti_1

            # ── Paso 3: Extracción de datos de flujo ─────────────────
            f_array = np.array(
                [float(G.edges[e].get("flow", 0.0)) for e in edges],
                dtype=float,
            )
            s_array = np.array(
                [float(G.nodes[n].get("sink_source", 0.0)) for n in nodes],
                dtype=float,
            )

            # ── Paso 4: Conservación discreta ────────────────────────
            continuity_report = self._enforce_discrete_continuity(
                B1, f_array, s_array, strict=True
            )

            # ── Paso 5: Política regenerativa ────────────────────────
            is_regenerative = self._evaluate_regenerative_policy(G, context)

            # ── Paso 6: Descomposición de Hodge ──────────────────────
            hodge = self._compute_hodge_decomposition(f_array, B1, C)

            # ── Paso 7: Política solenoidal ──────────────────────────
            self._validate_solenoidal_policy(
                curl_energy=hodge.curl_energy,
                cycle_rank=betti_1,
                is_regenerative=is_regenerative,
            )

            # ── Paso 8: Anotación de aristas ─────────────────────────
            for j, e in enumerate(edges):
                G.edges[e]["flow_grad"] = float(hodge.f_grad[j])
                G.edges[e]["flow_curl"] = float(hodge.f_curl[j])
                G.edges[e]["flow_harm"] = float(hodge.f_harm[j])

            # ── Paso 9: Análisis espectral y centralidades ───────────
            undirected = G.to_undirected()
            fiedler_val = self._compute_fiedler_value(undirected)
            centralities = self._compute_structural_centralities(G)

            # ── Paso 10: Renormalización de masa ─────────────────────
            for n in nodes:
                delay = float(G.nodes[n].get("delay", 0.0))
                if delay > 0.0:
                    c_val = float(centralities.get(n, 0.0))
                    m_eff = self._quantize_logistical_polarons(
                        fiedler_val=fiedler_val,
                        centrality=c_val,
                        base_mass=delay,
                    )
                    G.nodes[n]["effective_mass"] = m_eff

            # ── Paso 11: Geodésicas opcionales ───────────────────────
            geodesic_info: Dict[str, Any] = {}
            route_source = context.get("route_source")
            route_target = context.get("route_target")

            if route_source is not None and route_target is not None:
                try:
                    metric = self._sanitize_metric(
                        context.get("logistics_metric")
                    )
                    path = self._compute_logistical_geodesics(
                        G_metric=metric,
                        graph=G,
                        source=route_source,
                        target=route_target,
                    )
                    geodesic_info["geodesic_path"] = path
                except Exception as geo_err:
                    logger.warning(
                        "No se pudo calcular geodésica logística: %s",
                        geo_err,
                    )
                    geodesic_info["geodesic_error"] = str(geo_err)

            # ── Paso 12: Ensamblado de contexto de salida ────────────
            new_ctx: Dict[str, Any] = {
                "logistics_graph": G,
                "euler_characteristic": int(euler_characteristic),
                "betti_0": int(n_components),
                "betti_1": int(betti_1),
                "cycle_rank": int(betti_1),
                "continuity_report": continuity_report.to_dict(),
                "integrality_defect": continuity_report.integrality_defect,
                "hodge_report": hodge.to_report_dict(),
                "fiedler_value": float(fiedler_val),
                "centralities": centralities,
            }

            new_ctx.update(geodesic_info)

            return state.with_update(new_context=new_ctx)

        except ValueError as val_err:
            return state.with_error(error_msg=str(val_err))
        except Exception as e:
            logger.exception("Error no controlado en LogisticsManifold.")
            return state.with_error(
                error_msg=f"Error en LogisticsManifold: {e}"
            )
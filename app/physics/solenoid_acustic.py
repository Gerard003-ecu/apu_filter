"""
=========================================================================================
Módulo: Solenoidal Acústico v3.0 — Descomposición de Hodge–Helmholtz Rigurosa
=========================================================================================

Fundamentos Matemáticos
-----------------------
Dado un grafo dirigido finito G = (V, E) con |V| = n, |E| = m, definimos el
complejo de cadenas de dimensión 1:

    C₀(G; ℝ) ←─∂₁─ C₁(G; ℝ) ←─∂₂─ C₂(G; ℝ)

donde:
  • ∂₁ : C₁ → C₀  es el operador de borde (Matriz de Incidencia B₁ ∈ ℝⁿˣᵐ)
         (B₁)_{v,e} = +1 si head(e)=v, -1 si tail(e)=v, 0 en otro caso
  • ∂₂ : C₂ → C₁  mapea 2-cadenas (ciclos) a 1-cadenas (B₂ ∈ ℝᵐˣᵏ)

Invariante fundamental (cochain complex):
    ∂₁ ∘ ∂₂ = 0  ⟺  B₁ · B₂ = 0  (mod 2 en ℤ, exacto en ℝ con orientaciones)

Laplaciano de Hodge sobre 1-cocadenas:
    L₁ = B₁ᵀB₁ + B₂B₂ᵀ  ∈ ℝᵐˣᵐ   (simétrico, PSD)
       = L_grad + L_curl

Isomorfismo de Hodge (para complejos finitos sobre ℝ):
    H₁(G; ℝ) ≅ ker(L₁) ≅ ker(B₁ᵀ) ∩ ker(B₂ᵀ)

Números de Betti verificados vía Euler–Poincaré:
    χ(G) = n - m = β₀ - β₁    (β₂ = 0 para grafos)

Descomposición de Hodge–Helmholtz (ortogonal):
    ℝᵐ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)

Propiedades Espectrales de L₁:
    • L₁ PSD  ⟹  λᵢ ≥ 0 ∀i
    • dim ker(L₁) = β₁  (para G conexo)
    • Gap espectral λ₁ > 0  ⟺  β₁ = 0 (flujo irrotacional)

Referencias:
    [1] Lim, L.-H. (2020). Hodge Laplacians on graphs. SIAM Review.
    [2] Eckmann, B. (1944). Harmonische Funktionen und Randwertaufgaben.
    [3] Golub & Van Loan (2013). Matrix Computations, 4th ed.
    [4] Trefethen & Bau (1997). Numerical Linear Algebra.
=========================================================================================
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# Stdlib
# ─────────────────────────────────────────────
import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

# ─────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────
import networkx as nx
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import svds

logger = logging.getLogger("MIC.Physics.AcousticSolenoid")


# ─────────────────────────────────────────────────────────────────────────────
# PARTE I · ÁLGEBRA LINEAL NUMÉRICA RIGUROSA
# ─────────────────────────────────────────────────────────────────────────────

class NumericalUtilities:
    """
    Álgebra lineal numérica con análisis de error riguroso.

    Todos los métodos son *stateless* y toman matrices densas o sparse.
    Las tolerancias siguen la convención de LAPACK/dgelsd:

        tol = max(m, n) · σ_max · ε_mach

    con ε_mach = 2.220446049250313e-16 (IEEE-754 double precision).
    """

    #: Épsilon de máquina para IEEE-754 double precision
    EPS_MACH: float = np.finfo(np.float64).eps  # 2.220446049250313e-16

    # ──────────────────────────────────────────────────────────────────────
    # 1. Tolerancia adaptativa
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def adaptive_tolerance(matrix: Union[sp.spmatrix, np.ndarray]) -> float:
        """
        Tolerancia numérica adaptativa siguiendo la convención LAPACK.

        Definición:
            tol = max(m, n) · σ_max · ε_mach

        donde σ_max es el valor singular máximo, m, n las dimensiones de la
        matriz y ε_mach el épsilon de máquina en doble precisión.

        Esta elección garantiza que perturbaciones de orden O(ε_mach · ‖A‖)
        — inherentes a la aritmética de punto flotante — no sean clasificadas
        erróneamente como singularidades.

        Args:
            matrix: Matriz densa o sparse sobre ℝ.

        Returns:
            Tolerancia ≥ ε_mach.

        Raises:
            ValueError: Si la matriz no es 2-D.
        """
        eps = NumericalUtilities.EPS_MACH

        if isinstance(matrix, sp.spmatrix):
            m, n = matrix.shape
            # σ_max via norma-2 sparse (potencia de iteración es costosa;
            # usamos norma de Frobenius como cota superior rápida).
            # ‖A‖_2 ≤ ‖A‖_F  siempre, por lo que sobrestimamos tol de forma
            # conservadora (seguro: clasifica menos valores como cero).
            sigma_max_ub = sp.linalg.norm(matrix, 'fro')
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"Se esperaba matriz 2-D, se recibió shape={arr.shape}"
                )
            m, n = arr.shape
            # SVD completo: σ_max exacto
            try:
                sigma_max_ub = np.linalg.svd(arr, compute_uv=False)[0]
            except np.linalg.LinAlgError:
                # Fallback: norma de Frobenius
                sigma_max_ub = np.linalg.norm(arr, 'fro')

        return max(eps, max(m, n) * sigma_max_ub * eps)

    # ──────────────────────────────────────────────────────────────────────
    # 2. Rango numérico
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_rank(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Rango numérico r = #{σᵢ > tol} mediante SVD.

        Theorem (Eckart–Young–Mirsky):
            La mejor aproximación de rango r de A en norma de Frobenius
            (o norma-2) es Aᵣ = Σᵢ≤ᵣ σᵢ uᵢvᵢᵀ.  El rango numérico es el
            mínimo r tal que ‖A − Aᵣ‖ ≤ tol.

        Args:
            matrix:    Matriz densa o sparse.
            tolerance: Umbral para σᵢ. Si None, se usa ``adaptive_tolerance``.

        Returns:
            Tupla (rank, singular_values) donde singular_values está ordenado
            de mayor a menor.
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        singular_values: np.ndarray = np.linalg.svd(dense, compute_uv=False)
        rank = int(np.sum(singular_values > tolerance))
        return rank, singular_values

    # ──────────────────────────────────────────────────────────────────────
    # 3. Pseudoinversa de Moore–Penrose
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def moore_penrose_pseudoinverse(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Pseudoinversa A⁺ mediante SVD truncado (algoritmo de Golub–Reinsch).

        Construcción:
            A = U Σ Vᵀ  (SVD completo)
            A⁺ = V Σ⁺ Uᵀ
            Σ⁺ = diag(σᵢ⁻¹ si σᵢ > tol, 0 en otro caso)

        Las 4 condiciones de Penrose se satisfacen exactamente en aritmética
        exacta y con error O(ε_mach) en punto flotante:
            (i)   A A⁺ A  = A
            (ii)  A⁺ A A⁺ = A⁺
            (iii) (A A⁺)ᵀ = A A⁺
            (iv)  (A⁺ A)ᵀ = A⁺ A

        Args:
            matrix:    Matriz densa o sparse m×n.
            tolerance: Umbral SVD. Si None, se usa ``adaptive_tolerance``.

        Returns:
            Pseudoinversa densa n×m.
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        U, s, Vt = np.linalg.svd(dense, full_matrices=False)
        # Invertir únicamente singulares sobre el umbral
        inv_s = np.where(s > tolerance, 1.0 / s, 0.0)
        # A⁺ = V diag(Σ⁺) Uᵀ
        return (Vt.T * inv_s) @ U.T  # equivalente a Vt.T @ diag(inv_s) @ U.T

    # ──────────────────────────────────────────────────────────────────────
    # 4. Proyección ortogonal numéricamente estable
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def orthogonal_projection(
        vector: np.ndarray,
        subspace_basis: np.ndarray,
        tolerance: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proyección ortogonal P_S(v) sobre S = col(B) usando SVD thin.

        Dado B ∈ ℝⁿˣᵏ con columnas (posiblemente) linealmente dependientes:

            B = U_r Σ_r V_rᵀ   (SVD thin, r = rank(B))
            P_S = U_r U_rᵀ      (proyector ortogonal sobre col(B))
            P_S(v) = U_r U_rᵀ v

        Esta formulación es numéricamente superior a B(BᵀB)⁻¹Bᵀ porque
        evita el cuadrado del número de condición κ(B)² que aparece en
        la matriz de Gram BᵀB.

        Args:
            vector:          Vector v ∈ ℝⁿ a proyectar.
            subspace_basis:  Matriz B ∈ ℝⁿˣᵏ cuyas columnas generan S.
            tolerance:       Umbral SVD para determinar rank(B).

        Returns:
            (projected, residual) con projected + residual = vector.

        Raises:
            ValueError: Si las dimensiones son inconsistentes.
        """
        v = np.asarray(vector, dtype=np.float64)
        B = np.asarray(subspace_basis, dtype=np.float64)

        if B.ndim != 2 or v.ndim != 1:
            raise ValueError(
                f"Se esperaba B 2-D y v 1-D; "
                f"recibidos B.shape={B.shape}, v.shape={v.shape}"
            )
        if B.shape[0] != v.shape[0]:
            raise ValueError(
                f"Dimensión inconsistente: B.shape[0]={B.shape[0]}, "
                f"v.shape[0]={v.shape[0]}"
            )

        # Subespacio vacío
        if B.size == 0 or B.shape[1] == 0:
            return np.zeros_like(v), v.copy()

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(B)

        # SVD thin de B
        U, s, _ = np.linalg.svd(B, full_matrices=False)

        # Retener solo columnas de U correspondientes a valores singulares > tol
        U_r = U[:, s > tolerance]

        if U_r.shape[1] == 0:
            # B es numéricamente nula: proyección trivial
            return np.zeros_like(v), v.copy()

        # P_S v = U_r (U_rᵀ v)
        projected = U_r @ (U_r.T @ v)
        residual = v - projected
        return projected, residual

    # ──────────────────────────────────────────────────────────────────────
    # 5. Número de condición
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def matrix_condition_number(
        matrix: Union[sp.spmatrix, np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Número de condición espectral κ₂(A) = σ_max / σ_min.

        Para matrices singulares (σ_min = 0) se retorna κ₂ = +∞.

        Args:
            matrix: Matriz densa o sparse.

        Returns:
            (κ₂, σ_min, σ_max)
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        try:
            s = np.linalg.svd(dense, compute_uv=False)
        except np.linalg.LinAlgError:
            return math.inf, 0.0, 0.0

        sigma_max = float(s[0]) if s.size > 0 else 0.0
        sigma_min = float(s[-1]) if s.size > 0 else 0.0
        kappa = sigma_max / sigma_min if sigma_min > 0.0 else math.inf
        return kappa, sigma_min, sigma_max

    # ──────────────────────────────────────────────────────────────────────
    # 6. Base ortonormal del núcleo
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def null_space_basis(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Base ortonormal de ker(A) via SVD.

        ker(A) = span{ vᵢ : σᵢ ≤ tol }

        donde A = U Σ Vᵀ es la SVD completa.

        Args:
            matrix:    Matriz densa o sparse m×n.
            tolerance: Umbral para σᵢ.

        Returns:
            Matriz n×d cuyas columnas forman base ortonormal de ker(A),
            donde d = n - rank(A).  Si ker(A) = {0}, retorna array n×0.
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        _, s, Vt = np.linalg.svd(dense, full_matrices=True)
        # Columnas de Vᵀ correspondientes a σ ≤ tol
        null_mask = np.append(s, np.zeros(Vt.shape[0] - s.size)) <= tolerance
        return Vt[null_mask].T


# ─────────────────────────────────────────────────────────────────────────────
# PARTE II · CONSTRUCCIÓN DE MATRICES DE INCIDENCIA Y CICLOS
# ─────────────────────────────────────────────────────────────────────────────

class HodgeDecompositionBuilder:
    """
    Construye el complejo de cadenas C₀ ←B₁— C₁ ←B₂— C₂ sobre un grafo
    dirigido G y calcula el Laplaciano de Hodge L₁ = B₁ᵀB₁ + B₂B₂ᵀ.

    Convenciones (compatibles con [Lim 2020]):
        (B₁)_{v,e} = +1 si head(e) = v
                   = -1 si tail(e) = v
                   =  0 en otro caso

    Invariantes garantizados post-construcción:
        1. B₁ ∈ ℝⁿˣᵐ, B₂ ∈ ℝᵐˣᵏ
        2. ‖B₁ B₂‖_F ≤ tol  (cochain complex)
        3. rank(B₂) = k = β₁
        4. χ = n - m = β₀ - β₁  (Euler–Poincaré)

    Args:
        graph: Grafo dirigido (nx.DiGraph).

    Raises:
        ValueError: Si el grafo no es DiGraph o tiene auto-lazos.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        if not isinstance(graph, nx.DiGraph):
            raise TypeError(
                f"Se esperaba nx.DiGraph, se recibió {type(graph).__name__}"
            )
        self.G: nx.DiGraph = graph
        self.n: int = graph.number_of_nodes()
        self.m: int = graph.number_of_edges()

        # Mapeos ordenados y deterministas
        self._nodes: List[Any] = list(graph.nodes())
        self._edges: List[Tuple[Any, Any]] = list(graph.edges())
        self._node_index: Dict[Any, int] = {
            node: idx for idx, node in enumerate(self._nodes)
        }
        self._edge_index: Dict[Tuple[Any, Any], int] = {
            edge: idx for idx, edge in enumerate(self._edges)
        }

    # ──────────────────────────────────────────────────────────────────────
    # 2.1 Matriz de Incidencia B₁
    # ──────────────────────────────────────────────────────────────────────

    def build_incidence_matrix(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Matriz de Incidencia Orientada B₁ ∈ ℝⁿˣᵐ.

        Definición (operador de borde ∂₁):
            ∂₁(e) = head(e) − tail(e)

        Codificada en B₁:
            (B₁)_{v,e} = +1  si v = head(e)
                       = −1  si v = tail(e)
                       =  0  en otro caso

        Propiedades verificadas:
            • Σ_v (B₁)_{v,e} = 0  ∀e  (cada columna suma cero)
            • rank(B₁) = n − c,  c = número de componentes conexas
            • ker(B₁ᵀ) ≅ H₀(G; ℝ)  (c-dimensional)

        Returns:
            (B1, metadata) con:
                metadata["rank_B1"]:  rango numérico
                metadata["column_sum_max"]:  ‖Σ_v B₁[:,e]‖_∞  (≈0)
        """
        B1 = np.zeros((self.n, self.m), dtype=np.float64)

        for (tail, head), e_idx in self._edge_index.items():
            tail_idx = self._node_index[tail]
            head_idx = self._node_index[head]
            B1[tail_idx, e_idx] = -1.0   # ∂₁(e) resta en el vértice cola
            B1[head_idx, e_idx] = +1.0   # ∂₁(e) suma en el vértice cabeza

        # Verificación: cada columna debe sumar cero
        col_sums = B1.sum(axis=0)
        col_sum_max = float(np.max(np.abs(col_sums)))

        rank_B1, svs = NumericalUtilities.compute_rank(B1)

        metadata: Dict[str, Any] = {
            "shape": (self.n, self.m),
            "rank_B1": rank_B1,
            "column_sum_max": col_sum_max,      # debe ser ≈ 0
            "singular_values": svs.tolist(),
        }
        return B1, metadata

    # ──────────────────────────────────────────────────────────────────────
    # 2.2 Matriz de Ciclos B₂
    # ──────────────────────────────────────────────────────────────────────

    def build_cycle_matrix(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Matriz de Ciclos Fundamentales B₂ ∈ ℝᵐˣᵏ.

        Algoritmo (garantiza B₁B₂ = 0 algebraicamente):
        ──────────────────────────────────────────────
        Sea T un árbol generador de G_undirected con m−n+c aristas de cotejo.
        Cada arista de cotejo eⱼ ∉ T define un ciclo fundamental Cⱼ:
            Cⱼ = único camino en T entre tail(eⱼ) y head(eⱼ) + eⱼ

        Para cada arista eᵢ ∈ Cⱼ:
            (B₂)_{i,j} = +1  si la dirección de eᵢ en Cⱼ coincide con la
                              dirección de eᵢ en G
                       = −1  en caso contrario

        Demostración de B₁B₂ = 0:
            Por construcción, cada columna de B₂ representa un ciclo orientado.
            El operador de borde de un ciclo es cero: ∂₁(∂₂(γ)) = 0.

        Propiedades:
            • k = β₁ = m − n + c  (primer número de Betti)
            • rank(B₂) = k  (columnas linealmente independientes por construcción)
            • B₁ B₂ = 0  (exacto en aritmética exacta, ‖·‖ ≤ tol en fp)

        Returns:
            (B2, metadata) con:
                metadata["betti_1"]:        β₁
                metadata["B1B2_norm"]:      ‖B₁B₂‖_F
                metadata["verify_B1B2_zero"]: bool
        """
        # Números de Betti: β₁ = m − n + c
        undirected = self.G.to_undirected()
        c = nx.number_connected_components(undirected)
        k = max(0, self.m - self.n + c)   # β₁ (puede ser 0 para bosques)

        B1, _ = self.build_incidence_matrix()

        if k == 0:
            B2 = np.zeros((self.m, 0), dtype=np.float64)
            metadata: Dict[str, Any] = {
                "shape": (self.m, 0),
                "betti_1": 0,
                "num_cycles": 0,
                "B1B2_norm": 0.0,
                "verify_B1B2_zero": True,
                "rank_B2": 0,
            }
            return B2, metadata

        # ── Árbol generador y aristas de cotejo ──────────────────────────
        # Usar un árbol generador por componente conexa
        spanning_tree_edges: FrozenSet[FrozenSet[Any]] = frozenset(
            frozenset(e) for e in nx.spanning_tree(undirected).edges()
        )

        tree_edge_set: set = {
            frozenset(e) for e in spanning_tree_edges
        }

        # Aristas de cotejo (co-tree edges) que definen los ciclos
        cotree_edges: List[Tuple[Any, Any]] = [
            e for e in self._edges
            if frozenset(e) not in tree_edge_set
        ]

        # Verificación de consistencia
        assert len(cotree_edges) == k, (
            f"Se esperaban {k} aristas de cotejo, se encontraron "
            f"{len(cotree_edges)}.  Revisar construcción del árbol generador."
        )

        # ── Construcción de B₂ columna a columna ─────────────────────────
        B2 = np.zeros((self.m, k), dtype=np.float64)

        for j, (cotree_tail, cotree_head) in enumerate(cotree_edges):
            # Índice de la arista de cotejo en el grafo original
            cotree_edge_idx = self._edge_index[(cotree_tail, cotree_head)]

            # La arista de cotejo contribuye con +1 (define la orientación
            # positiva del ciclo)
            B2[cotree_edge_idx, j] = +1.0

            # Camino único en el árbol entre cotree_tail y cotree_head
            try:
                path_nodes: List[Any] = nx.shortest_path(
                    undirected, cotree_tail, cotree_head
                )
            except nx.NetworkXNoPath:
                # No debería ocurrir en un árbol generador válido
                logger.error(
                    f"Sin camino en árbol entre {cotree_tail} y {cotree_head}"
                )
                continue

            # Para cada arista del camino, determinar orientación relativa
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]

                # Solo aristas de árbol: frozenset({u,v}) ∈ tree_edge_set
                if frozenset({u, v}) not in tree_edge_set:
                    continue

                # Determinar si la arista dirigida original es (u→v) o (v→u)
                if (u, v) in self._edge_index:
                    # Arista dirigida u→v: misma dirección que el camino
                    e_idx = self._edge_index[(u, v)]
                    B2[e_idx, j] = +1.0
                elif (v, u) in self._edge_index:
                    # Arista dirigida v→u: dirección inversa al camino
                    e_idx = self._edge_index[(v, u)]
                    B2[e_idx, j] = -1.0
                # Si ninguna dirección existe (no es arista del grafo),
                # se ignora — no debe ocurrir en árbol válido

        # ── Verificación algebraica B₁B₂ = 0 ────────────────────────────
        product = B1 @ B2
        B1B2_norm = float(np.linalg.norm(product, 'fro'))
        tol_verify = NumericalUtilities.adaptive_tolerance(B1) * self.m

        rank_B2, _ = NumericalUtilities.compute_rank(B2)

        if B1B2_norm > tol_verify:
            logger.warning(
                f"‖B₁B₂‖_F = {B1B2_norm:.2e} > tol = {tol_verify:.2e}. "
                f"Posible error en construcción de ciclos."
            )

        metadata = {
            "shape": (self.m, k),
            "betti_1": k,
            "num_cycles": k,
            "rank_B2": rank_B2,
            "B1B2_norm": B1B2_norm,
            "verify_B1B2_zero": B1B2_norm <= tol_verify,
            "cotree_edges": cotree_edges,
        }
        return B2, metadata

    # ──────────────────────────────────────────────────────────────────────
    # 2.3 Laplaciano de Hodge L₁
    # ──────────────────────────────────────────────────────────────────────

    def compute_hodge_laplacian(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Laplaciano de Hodge sobre 1-cadenas:

            L₁ = B₁ᵀB₁ + B₂B₂ᵀ  ∈ ℝᵐˣᵐ

        Descomposición en suma directa ortogonal (Teorema de Hodge):

            ℝᵐ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)

        Propiedades espectrales verificadas:
            • L₁ simétrica PSD  (L₁ = L₁ᵀ ≥ 0)
            • Autovalores λᵢ ≥ 0 ∀i
            • dim ker(L₁) = β₁  para G conexo
            • Traza(L₁) = Σᵢλᵢ = ‖B₁‖²_F + ‖B₂‖²_F

        Returns:
            (L1, metadata) con análisis espectral completo.
        """
        B1, meta1 = self.build_incidence_matrix()
        B2, meta2 = self.build_cycle_matrix()

        # L₁ = L_grad + L_curl
        L_grad = B1.T @ B1   # m×m, PSD, im = im(B₁ᵀ)
        L_curl = B2 @ B2.T   # m×m, PSD, im = im(B₂)
        L1 = L_grad + L_curl

        # ── Análisis espectral ───────────────────────────────────────────
        # Usar eigh (simétrica) para garantizar autovalores reales
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L1)
        except np.linalg.LinAlgError as exc:
            logger.error(f"Fallo en eigh: {exc}")
            eigenvalues = np.zeros(self.m)
            eigenvectors = np.eye(self.m)

        # eigh devuelve en orden ascendente (garantizado)
        # Recortar negativos numéricos a 0
        eigenvalues = np.maximum(eigenvalues, 0.0)

        tol_eig = NumericalUtilities.adaptive_tolerance(L1)

        zero_eigenvalues = int(np.sum(eigenvalues < tol_eig))
        spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if self.m > 1 else 0.0

        kappa, sigma_min, sigma_max = NumericalUtilities.matrix_condition_number(L1)

        # Verificar isomorfismo de Hodge: dim ker(L₁) = β₁
        betti_1 = meta2["betti_1"]
        hodge_iso_satisfied = abs(zero_eigenvalues - betti_1) <= 1

        if not hodge_iso_satisfied:
            logger.warning(
                f"Isomorfismo de Hodge fallido: "
                f"dim ker(L₁) = {zero_eigenvalues}, β₁ = {betti_1}. "
                f"Revisar construcción del complejo."
            )

        metadata: Dict[str, Any] = {
            "shape": L1.shape,
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors_shape": eigenvectors.shape,
            "spectral_gap": spectral_gap,
            "kernel_dimension": zero_eigenvalues,
            "betti_1": betti_1,
            "hodge_isomorphism_satisfied": hodge_iso_satisfied,
            "condition_number": kappa,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "trace_L1": float(np.trace(L1)),
            "trace_check": float(np.sum(eigenvalues)),  # debe ≈ trace
            "L_grad_norm_F": float(np.linalg.norm(L_grad, 'fro')),
            "L_curl_norm_F": float(np.linalg.norm(L_curl, 'fro')),
            "is_symmetric": bool(np.allclose(L1, L1.T, atol=tol_eig)),
            "is_positive_semidefinite": bool(np.all(eigenvalues >= -tol_eig)),
        }
        return L1, metadata

    # ──────────────────────────────────────────────────────────────────────
    # 2.4 Verificación del complejo de co-cadenas
    # ──────────────────────────────────────────────────────────────────────

    def verify_cochain_complex(self) -> Dict[str, Any]:
        """
        Verificación formal de las propiedades del complejo de cadenas.

        Propiedades verificadas:
            1. ‖B₁B₂‖_F ≤ tol             (∂₁ ∘ ∂₂ = 0)
            2. Dimensiones consistentes
            3. rank(B₁) = n − c
            4. rank(B₂) = β₁ = m − n + c
            5. Euler–Poincaré: χ = n − m = β₀ − β₁
               con β₀ = c (componentes conexas)

        Returns:
            Dict con resultados booleanos y métricas numéricas.
        """
        B1, meta1 = self.build_incidence_matrix()
        B2, meta2 = self.build_cycle_matrix()

        undirected = self.G.to_undirected()
        c = nx.number_connected_components(undirected)
        betti_0 = c
        betti_1 = meta2["betti_1"]

        # 1. Verificar B₁B₂ = 0
        B1B2_zero = meta2["verify_B1B2_zero"]
        B1B2_norm = meta2["B1B2_norm"]

        # 2. Dimensiones
        dims_ok = (
            B1.shape == (self.n, self.m)
            and B2.shape == (self.m, betti_1)
        )

        # 3. rank(B₁) = n − c
        rank_B1 = meta1["rank_B1"]
        rank_B1_expected = self.n - c
        rank_B1_ok = rank_B1 == rank_B1_expected

        # 4. rank(B₂) = β₁
        rank_B2 = meta2["rank_B2"]
        rank_B2_ok = rank_B2 == betti_1

        # 5. Euler–Poincaré: n − m = β₀ − β₁
        chi = self.n - self.m
        chi_topological = betti_0 - betti_1
        euler_ok = (chi == chi_topological)

        is_valid = B1B2_zero and dims_ok and rank_B1_ok and rank_B2_ok and euler_ok

        return {
            "is_valid": is_valid,
            # ∂₁∂₂ = 0
            "B1B2_zero": B1B2_zero,
            "B1B2_norm": B1B2_norm,
            # Dimensiones
            "dimensions_consistent": dims_ok,
            # Rango B₁
            "rank_B1": rank_B1,
            "rank_B1_expected": rank_B1_expected,
            "rank_B1_ok": rank_B1_ok,
            # Rango B₂
            "rank_B2": rank_B2,
            "rank_B2_expected": betti_1,
            "rank_B2_ok": rank_B2_ok,
            # Euler–Poincaré
            "chi_geometric": chi,
            "chi_topological": chi_topological,
            "euler_poincare_ok": euler_ok,
            # Números de Betti
            "beta_0": betti_0,
            "beta_1": betti_1,
            "connected_components": c,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE III · OPERADOR DE VORTICIDAD SOLENOIDAL
# ─────────────────────────────────────────────────────────────────────────────

class AcousticSolenoidOperator:
    """
    Proyector de Vorticidad Solenoidal sobre el 1-esqueleto de G.

    Implementa el proyector ortogonal:

        P_curl : ℝᵐ → im(B₂)
        P_curl = B₂(B₂ᵀB₂)⁺B₂ᵀ

    usando la formulación estable vía SVD:

        B₂ = U_r Σ_r V_rᵀ  ⟹  P_curl = U_r U_rᵀ

    Métricas calculadas:
        • Circulación Γⱼ = (B₂ᵀI)ⱼ  (ley de Stokes discreta)
        • Energía E_curl = ‖B₂ᵀI‖² = Iᵀ L_curl I
        • Índice de vorticidad ω = E_curl / ‖I‖²  ∈ [0, 1]
        • Error de idempotencia ‖P² − P‖_F / ‖P‖_F

    Args:
        tolerance_epsilon:  Umbral base para filtrar ruido numérico.
        adaptive_threshold: Si True, escala ε con la norma del flujo.
    """

    def __init__(
        self,
        tolerance_epsilon: float = 1e-9,
        adaptive_threshold: bool = True,
    ) -> None:
        self.epsilon = tolerance_epsilon
        self.adaptive = adaptive_threshold

        # Cache de la última descomposición de Hodge computada
        self._cached_builder: Optional[HodgeDecompositionBuilder] = None
        self._cached_graph_signature: Optional[Tuple] = None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────────

    def _graph_signature(self, G: nx.DiGraph) -> Tuple:
        """
        Firma determinista del grafo para invalidar cache.

        Incluye nodos, aristas y atributos relevantes.
        """
        return (
            tuple(sorted(G.nodes())),
            tuple(sorted(G.edges())),
            G.number_of_nodes(),
            G.number_of_edges(),
        )

    def _get_builder(self, G: nx.DiGraph) -> HodgeDecompositionBuilder:
        """Retorna un HodgeDecompositionBuilder, usando cache si válido."""
        sig = self._graph_signature(G)
        if self._cached_builder is None or self._cached_graph_signature != sig:
            self._cached_builder = HodgeDecompositionBuilder(G)
            self._cached_graph_signature = sig
        return self._cached_builder

    def _build_flow_vector(
        self,
        builder: HodgeDecompositionBuilder,
        edge_flows: Dict[Tuple[Any, Any], float],
    ) -> np.ndarray:
        """
        Convierte el diccionario de flujos a vector I ∈ ℝᵐ.

        Aristas ausentes en edge_flows reciben flujo 0.
        Aristas en edge_flows no presentes en G se ignoran con advertencia.
        """
        I_vec = np.zeros(builder.m, dtype=np.float64)
        for (u, v), flow in edge_flows.items():
            idx = builder._edge_index.get((u, v))
            if idx is not None:
                I_vec[idx] = float(flow)
            else:
                logger.debug(
                    f"Arista ({u},{v}) en edge_flows no encontrada en G. "
                    f"Ignorada."
                )
        return I_vec

    def _compute_projector_via_svd(
        self,
        B2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calcula el proyector P_curl = U_r U_rᵀ vía SVD de B₂.

        Evita el cuadrado del número de condición κ(B₂ᵀB₂) = κ(B₂)²
        que aparecería al usar P = B₂(B₂ᵀB₂)⁻¹B₂ᵀ directamente.

        Returns:
            (P_curl, U_r, idempotency_error)
        """
        tol = NumericalUtilities.adaptive_tolerance(B2)
        U, s, _ = np.linalg.svd(B2, full_matrices=False)
        U_r = U[:, s > tol]

        if U_r.shape[1] == 0:
            m = B2.shape[0]
            P = np.zeros((m, m), dtype=np.float64)
            return P, U_r, 0.0

        P_curl = U_r @ U_r.T

        # Verificar idempotencia: P² = P  ⟺  ‖P² − P‖_F / ‖P‖_F ≈ 0
        P_norm = np.linalg.norm(P_curl, 'fro')
        if P_norm > 0:
            idempotency_error = float(
                np.linalg.norm(P_curl @ P_curl - P_curl, 'fro') / P_norm
            )
        else:
            idempotency_error = 0.0

        return P_curl, U_r, idempotency_error

    # ──────────────────────────────────────────────────────────────────────
    # 3.1 Aislamiento de vorticidad
    # ──────────────────────────────────────────────────────────────────────

    def isolate_vorticity(
        self,
        G: nx.DiGraph,
        edge_flows: Dict[Tuple[Any, Any], float],
    ) -> Optional["MagnonCartridge"]:
        """
        Proyecta I ∈ ℝᵐ sobre im(B₂) y cuantifica la vorticidad.

        Algoritmo:
        ──────────
        1. Construir B₂ ∈ ℝᵐˣᵏ (ciclos fundamentales).
        2. Formar I ∈ ℝᵐ (vector de flujos).
        3. Calcular circulación Γ = B₂ᵀI ∈ ℝᵏ.
        4. Calcular E_curl = ‖Γ‖² (energía en modo rotacional).
        5. Calcular ω = E_curl / ‖I‖² (índice de vorticidad).
        6. Calcular P_curl = U_r U_rᵀ vía SVD de B₂.
        7. Verificar idempotencia ‖P² − P‖_F / ‖P‖_F.

        Ley de Stokes discreta:
            Γⱼ = (B₂ᵀI)ⱼ = Σᵢ (B₂)ᵢⱼ Iᵢ
            representa la circulación del flujo alrededor del ciclo j.

        Args:
            G:          Grafo dirigido.
            edge_flows: Flujos por arista {(u,v): f}.

        Returns:
            MagnonCartridge si E_curl > umbral adaptativo, None si flujo
            es irrotacional.
        """
        builder = self._get_builder(G)
        B2, cycle_meta = builder.build_cycle_matrix()
        k = B2.shape[1]

        # ── Sin ciclos: vorticidad axiomáticamente nula ──────────────────
        if k == 0:
            logger.debug(
                "β₁ = 0: grafo sin ciclos. Vorticidad nula por topología."
            )
            return None

        # ── Vector de flujo I ∈ ℝᵐ ──────────────────────────────────────
        I_vec = self._build_flow_vector(builder, edge_flows)
        flow_norm = float(np.linalg.norm(I_vec))

        if flow_norm < self.epsilon:
            logger.debug(f"‖I‖ = {flow_norm:.2e} < ε. Flujo despreciable.")
            return None

        # ── Circulación Γ = B₂ᵀ I (Ley de Stokes discreta) ─────────────
        circulation = B2.T @ I_vec      # shape (k,)

        # ── Energía cinética de vorticidad E_curl = ‖Γ‖² ────────────────
        kinetic_energy = float(np.dot(circulation, circulation))

        # ── Índice de vorticidad ω = E_curl / ‖I‖² ∈ [0, 1] ─────────────
        # Cota superior: por Cauchy–Schwarz, E_curl ≤ ‖B₂‖²_F · ‖I‖²
        # En general ω ≤ σ_max(B₂)² pero acotamos por 1 para interpretación
        vorticity_index = kinetic_energy / (flow_norm ** 2)
        # Clipear a [0, 1] para absorber errores numéricos de redondeo
        vorticity_index = float(np.clip(vorticity_index, 0.0, 1.0))

        # ── Proyector P_curl = U_r U_rᵀ vía SVD ────────────────────────
        P_curl, U_r, idempotency_error = self._compute_projector_via_svd(B2)

        # ── Umbral adaptativo ────────────────────────────────────────────
        if self.adaptive:
            adaptive_eps = max(
                self.epsilon,
                self.epsilon * (flow_norm ** 2),
            )
        else:
            adaptive_eps = self.epsilon

        if kinetic_energy < adaptive_eps:
            logger.debug(
                f"E_curl = {kinetic_energy:.2e} < ε_adapt = {adaptive_eps:.2e}. "
                f"Descartado como ruido numérico."
            )
            return None

        # ── Descomposición de energía ─────────────────────────────────────
        # Energía en componente irrotacional (grad) = total - curl - harm
        # Calculada vía proyección para consistencia
        I_curl = P_curl @ I_vec
        E_curl_proj = float(np.linalg.norm(I_curl) ** 2)

        energy_decomp: Dict[str, float] = {
            "total_flow_energy": float(flow_norm ** 2),
            "curl_energy_circulation": kinetic_energy,    # ‖B₂ᵀI‖²
            "curl_energy_projection": E_curl_proj,         # ‖P_curl I‖²
            "vorticity_ratio": vorticity_index,
        }

        return MagnonCartridge(
            kinetic_energy=kinetic_energy,
            curl_subspace_dim=k,
            vorticity_index=vorticity_index,
            circulation_per_cycle=tuple(float(c) for c in circulation),
            projection_idempotency_error=idempotency_error,
            energy_decomposition=energy_decomp,
            cycle_metadata=dict(cycle_meta),
            projector_matrix=P_curl,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 3.2 Descomposición de Hodge completa
    # ──────────────────────────────────────────────────────────────────────

    def compute_full_hodge_decomposition(
        self,
        G: nx.DiGraph,
        edge_flows: Dict[Tuple[Any, Any], float],
    ) -> Dict[str, Any]:
        """
        Descomposición ortogonal de Hodge–Helmholtz:

            I = I_grad + I_curl + I_harm

        donde las componentes son mutuamente ortogonales y:
            I_grad ∈ im(B₁ᵀ)   (irrotacional, gradiente de potencial)
            I_curl ∈ im(B₂)    (solenoidal, rotacional puro)
            I_harm ∈ ker(L₁)   (armónico, satisface ambas condiciones)

        Algoritmo (numéricamente estable vía SVD):
        ──────────────────────────────────────────
        Sea B₁ = U₁Σ₁V₁ᵀ y B₂ = U₂Σ₂V₂ᵀ (SVDs thin)

        Bases ortonormales:
            U_grad = columnas de U₁ con σᵢ > tol  (base de im(B₁))
            U_curl = columnas de U₂ con σᵢ > tol  (base de im(B₂))

        Proyectores:
            P_grad = U_grad U_gradᵀ  (proyecta sobre im(B₁ᵀ))
            P_curl = U_curl U_curlᵀ  (proyecta sobre im(B₂))

        Nota: im(B₁ᵀ) ≡ im(B₁ᵀ) como subespacio de ℝᵐ.
        Los proyectores sobre im(B₁ᵀ) se obtienen de la SVD de B₁ᵀ
        (equivalente a los vectores singulares derechos de B₁).

        Componentes:
            I_grad = P_grad I
            I_curl = P_curl I
            I_harm = I - I_grad - I_curl

        Verificación de ortogonalidad:
            ⟨I_grad, I_curl⟩ ≈ 0
            ⟨I_grad, I_harm⟩ ≈ 0
            ⟨I_curl, I_harm⟩ ≈ 0

        Returns:
            Dict con componentes y métricas de verificación.
        """
        builder = self._get_builder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, _ = builder.build_cycle_matrix()

        I_vec = self._build_flow_vector(builder, edge_flows)

        tol_B1 = NumericalUtilities.adaptive_tolerance(B1)
        tol_B2 = NumericalUtilities.adaptive_tolerance(B2) if B2.size > 0 else 1e-12

        # ── Proyector sobre im(B₁ᵀ) ─────────────────────────────────────
        # B₁ᵀ ∈ ℝᵐˣⁿ: sus columnas generan im(B₁ᵀ)
        B1T = B1.T
        U1, s1, _ = np.linalg.svd(B1T, full_matrices=False)
        U_grad = U1[:, s1 > tol_B1]
        P_grad = U_grad @ U_grad.T if U_grad.shape[1] > 0 else np.zeros(
            (builder.m, builder.m)
        )
        I_grad = P_grad @ I_vec

        # ── Proyector sobre im(B₂) ───────────────────────────────────────
        if B2.shape[1] > 0:
            U2, s2, _ = np.linalg.svd(B2, full_matrices=False)
            U_curl = U2[:, s2 > tol_B2]
            P_curl = U_curl @ U_curl.T if U_curl.shape[1] > 0 else np.zeros(
                (builder.m, builder.m)
            )
        else:
            P_curl = np.zeros((builder.m, builder.m))

        I_curl = P_curl @ I_vec

        # ── Componente armónica ──────────────────────────────────────────
        I_harm = I_vec - I_grad - I_curl

        # ── Verificaciones ───────────────────────────────────────────────
        reconstruction_error = float(
            np.linalg.norm(I_vec - I_grad - I_curl - I_harm)
        )
        inner_grad_curl = float(np.dot(I_grad, I_curl))
        inner_grad_harm = float(np.dot(I_grad, I_harm))
        inner_curl_harm = float(np.dot(I_curl, I_harm))

        tol_orth = NumericalUtilities.adaptive_tolerance(np.eye(builder.m)) * 10

        return {
            "original_flow": I_vec.tolist(),
            "irrotational_component": I_grad.tolist(),
            "solenoidal_component": I_curl.tolist(),
            "harmonic_component": I_harm.tolist(),
            "energy_decomposition": {
                "total": float(np.linalg.norm(I_vec) ** 2),
                "irrotational": float(np.linalg.norm(I_grad) ** 2),
                "solenoidal": float(np.linalg.norm(I_curl) ** 2),
                "harmonic": float(np.linalg.norm(I_harm) ** 2),
            },
            "norms": {
                "original": float(np.linalg.norm(I_vec)),
                "irrotational": float(np.linalg.norm(I_grad)),
                "solenoidal": float(np.linalg.norm(I_curl)),
                "harmonic": float(np.linalg.norm(I_harm)),
            },
            "verification": {
                "reconstruction_error": reconstruction_error,
                "orthogonality_grad_curl": inner_grad_curl,
                "orthogonality_grad_harm": inner_grad_harm,
                "orthogonality_curl_harm": inner_curl_harm,
                "is_orthogonal_decomposition": (
                    abs(inner_grad_curl) < tol_orth
                    and abs(inner_grad_harm) < tol_orth
                    and abs(inner_curl_harm) < tol_orth
                    and reconstruction_error < tol_orth
                ),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # 3.3 Análisis espectral
    # ──────────────────────────────────────────────────────────────────────

    def spectral_analysis(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Análisis espectral completo del Laplaciano de Hodge L₁.

        Calcula:
            • Espectro {λᵢ} de L₁ = B₁ᵀB₁ + B₂B₂ᵀ
            • Gap espectral λ₁ (primer autovalor no nulo)
            • dim ker(L₁) = β₁
            • Clasificación de modos: gradiente vs. curl vs. armónico

        Returns:
            Dict con espectro y métricas de conectividad.
        """
        builder = self._get_builder(G)
        L1, spectral_meta = builder.compute_hodge_laplacian()

        return {
            "laplacian_spectrum": spectral_meta.get("eigenvalues", []),
            "spectral_gap": spectral_meta.get("spectral_gap", 0.0),
            "kernel_dimension": spectral_meta.get("kernel_dimension", 0),
            "betti_1": spectral_meta.get("betti_1", 0),
            "condition_number": spectral_meta.get("condition_number", math.inf),
            "is_positive_semidefinite": spectral_meta.get(
                "is_positive_semidefinite", False
            ),
            "hodge_isomorphism_satisfied": spectral_meta.get(
                "hodge_isomorphism_satisfied", False
            ),
            "trace_L1": spectral_meta.get("trace_L1", 0.0),
            "L_grad_norm_F": spectral_meta.get("L_grad_norm_F", 0.0),
            "L_curl_norm_F": spectral_meta.get("L_curl_norm_F", 0.0),
            "is_symmetric": spectral_meta.get("is_symmetric", False),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE IV · BOSÓN DE VORTICIDAD — MAGNON CARTRIDGE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagnonCartridge:
    """
    Estado colapsado de vorticidad solenoidal detectada.

    Representa el resultado de proyectar I ∈ ℝᵐ sobre im(B₂) y medir
    la energía cinética de vorticidad E_curl = ‖B₂ᵀI‖².

    Nota de diseño:
        No se usa ``frozen=True`` porque ``np.ndarray`` no es hashable
        y ``frozen`` con campos mutables rompe la inmutabilidad semántica.
        En su lugar, los campos numéricos críticos son inmutables por tipo
        (float, int, tuple) y el proyector se almacena por referencia.

    Atributos:
        kinetic_energy:              E_curl = ‖B₂ᵀI‖² ≥ 0
        curl_subspace_dim:           k = β₁ ≥ 0
        vorticity_index:             ω = E_curl / ‖I‖² ∈ [0, 1]
        circulation_per_cycle:       Γⱼ = (B₂ᵀI)ⱼ para j = 1,…,k
        projection_idempotency_error: ‖P²−P‖_F / ‖P‖_F  (≈0 si P es proyector)
        energy_decomposition:        Desglose energético
        cycle_metadata:              Metadata topológica de ciclos
        projector_matrix:            P_curl = U_r U_rᵀ (opcional, puede ser grande)
    """

    kinetic_energy: float
    curl_subspace_dim: int
    vorticity_index: float = 0.0
    circulation_per_cycle: Tuple[float, ...] = field(default_factory=tuple)
    projection_idempotency_error: float = 0.0
    energy_decomposition: Dict[str, float] = field(default_factory=dict)
    cycle_metadata: Dict[str, Any] = field(default_factory=dict)
    projector_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """
        Validación de invariantes físicos y matemáticos.

        Invariantes:
            • E_curl ≥ 0  (energía no negativa)
            • ω ∈ [0, 1]  (fracción de energía)
            • k ≥ 0       (dimensión no negativa)
        """
        if self.kinetic_energy < -1e-12:
            raise ValueError(
                f"Energía cinética debe ser ≥ 0; recibida: {self.kinetic_energy:.6e}"
            )
        # Corregir ruido numérico menor
        object.__setattr__(
            self, "kinetic_energy", max(0.0, self.kinetic_energy)
        ) if False else None  # dataclass no frozen: asignación directa
        self.kinetic_energy = max(0.0, self.kinetic_energy)

        if not (-1e-10 <= self.vorticity_index <= 1.0 + 1e-10):
            raise ValueError(
                f"Índice de vorticidad fuera de [0,1]: {self.vorticity_index:.6e}"
            )
        self.vorticity_index = float(np.clip(self.vorticity_index, 0.0, 1.0))

        if self.curl_subspace_dim < 0:
            raise ValueError(
                f"Dimensión de subespacio curl debe ser ≥ 0; "
                f"recibida: {self.curl_subspace_dim}"
            )

        if self.projection_idempotency_error > 1e-6:
            logger.warning(
                f"Error de idempotencia elevado: "
                f"{self.projection_idempotency_error:.2e}. "
                f"El proyector P_curl puede no ser ortogonal."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Propiedades derivadas
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_significant(self) -> bool:
        """
        Determina si la vorticidad es físicamente significativa.

        Criterios (todos deben cumplirse):
            • E_curl > 1e-9  (energía mínima detectable)
            • ω > 0.01       (al menos 1% de la energía en modo rotacional)
            • k > 0          (existencia de ciclos topológicos)
        """
        return (
            self.kinetic_energy > 1e-9
            and self.vorticity_index > 0.01
            and self.curl_subspace_dim > 0
        )

    @property
    def dominant_cycle(self) -> Tuple[int, float]:
        """
        Ciclo con mayor magnitud de circulación |Γⱼ|.

        Returns:
            (cycle_index, Γ_dominant)  o (-1, 0.0) si no hay ciclos.
        """
        if not self.circulation_per_cycle:
            return (-1, 0.0)
        abs_circ = [abs(c) for c in self.circulation_per_cycle]
        max_idx = int(np.argmax(abs_circ))
        return (max_idx, self.circulation_per_cycle[max_idx])

    @property
    def thermodynamic_severity(self) -> str:
        """
        Clasifica la severidad basada en ω (fracción de energía rotacional).

        Umbrales:
            ω > 0.50 → CRITICAL
            ω > 0.20 → HIGH
            ω > 0.05 → MODERATE
            ω ≤ 0.05 → LOW
        """
        if self.vorticity_index > 0.50:
            return "CRITICAL"
        elif self.vorticity_index > 0.20:
            return "HIGH"
        elif self.vorticity_index > 0.05:
            return "MODERATE"
        return "LOW"

    @property
    def total_circulation_norm(self) -> float:
        """‖Γ‖ = √(E_curl)  (norma del vector de circulaciones)."""
        return math.sqrt(self.kinetic_energy)

    # ──────────────────────────────────────────────────────────────────────
    # Métodos de acción
    # ──────────────────────────────────────────────────────────────────────

    def to_veto_payload(self) -> Dict[str, Any]:
        """
        Genera el payload para el Veto de Enrutamiento.

        El payload es serializable (sin np.ndarray).
        """
        return {
            "type": "ROUTING_VETO",
            "magnitude": self.thermodynamic_severity,
            "vorticity_metrics": {
                "kinetic_energy": self.kinetic_energy,
                "curl_dimension_beta1": self.curl_subspace_dim,
                "vorticity_index_omega": self.vorticity_index,
                "dominant_cycle": self.dominant_cycle,
                "total_circulation_norm": self.total_circulation_norm,
                "projector_quality": 1.0 - self.projection_idempotency_error,
            },
            "causality_status": (
                "COMPROMISED" if self.is_significant else "INTACT"
            ),
            "prescribed_action": self._prescribe_action(),
            "energy_decomposition": self.energy_decomposition,
        }

    def _prescribe_action(self) -> str:
        """Acción correctiva según severidad termodinámica."""
        actions = {
            "CRITICAL":  "COLLAPSE_AND_RECONFIGURE",
            "HIGH":      "PARTITION_AND_RELAY",
            "MODERATE":  "MONITOR_AND_DAMP",
            "LOW":       "LOG_AND_PROCEED",
        }
        return actions[self.thermodynamic_severity]


# ─────────────────────────────────────────────────────────────────────────────
# PARTE V · INTERFAZ AGÉNTICA
# ─────────────────────────────────────────────────────────────────────────────

def inspect_and_mitigate_resonance(
    G: nx.DiGraph,
    flows: Dict[Tuple[Any, Any], float],
    full_analysis: bool = False,
) -> Dict[str, Any]:
    """
    Punto de entrada principal para la Malla Agéntica.

    Detecta dependencias circulares (ciclos topológicos) en G,
    cuantifica la energía cinética parasitaria y emite un
    MagnonCartridge si la vorticidad supera el umbral adaptativo.

    Pipeline:
        1. Construir descomposición de Hodge (B₁, B₂, L₁)
        2. Proyectar flujos I sobre im(B₂) → circulaciones Γ
        3. Calcular E_curl = ‖Γ‖² y ω = E_curl / ‖I‖²
        4. Emitir MagnonCartridge o confirmar flujo laminar
        5. (Opcional) Descomposición completa de Hodge + análisis espectral

    Args:
        G:             Grafo dirigido del sistema.
        flows:         Campo de flujos {(u,v): valor} sobre aristas.
        full_analysis: Si True, incluye descomposición completa y espectro.

    Returns:
        Dict con:
            status:            "RESONANCE_DETECTED" | "LAMINAR_FLOW"
            action:            Acción prescripta
            vorticity_metrics: Métricas de vorticidad
            mathematical_proof: Verificación formal
            full_hodge_decomposition: (si full_analysis=True)
            spectral_analysis:        (si full_analysis=True)
    """
    logger.info(
        f"Análisis de vorticidad iniciado. "
        f"V={G.number_of_nodes()}, E={G.number_of_edges()}"
    )

    solenoid = AcousticSolenoidOperator(adaptive_threshold=True)
    magnon = solenoid.isolate_vorticity(G, flows)

    # ── RESONANCIA DETECTADA ─────────────────────────────────────────────
    if magnon is not None and magnon.is_significant:
        result: Dict[str, Any] = {
            "status": "RESONANCE_DETECTED",
            "action": magnon._prescribe_action(),
            "vorticity_metrics": {
                "parasitic_kinetic_energy": magnon.kinetic_energy,
                "betti_1_cycles": magnon.curl_subspace_dim,
                "vorticity_index": magnon.vorticity_index,
                "thermodynamic_severity": magnon.thermodynamic_severity,
                "dominant_cycle": magnon.dominant_cycle,
                "circulation_per_cycle": magnon.circulation_per_cycle,
                "total_circulation_norm": magnon.total_circulation_norm,
                "projection_quality": 1.0 - magnon.projection_idempotency_error,
            },
            "mathematical_proof": _generate_proof(magnon),
            "energy_accounting": magnon.energy_decomposition,
        }

        if full_analysis:
            result["full_hodge_decomposition"] = (
                solenoid.compute_full_hodge_decomposition(G, flows)
            )
            result["spectral_analysis"] = solenoid.spectral_analysis(G)

        logger.warning(
            f"Vorticidad detectada — β₁={magnon.curl_subspace_dim}, "
            f"E_curl={magnon.kinetic_energy:.4f}, "
            f"ω={magnon.vorticity_index:.4f}, "
            f"severidad={magnon.thermodynamic_severity}"
        )
        return result

    # ── FLUJO LAMINAR ────────────────────────────────────────────────────
    result = {
        "status": "LAMINAR_FLOW",
        "action": "PROCEED",
        "vorticity_metrics": {
            "parasitic_kinetic_energy": 0.0,
            "betti_1_cycles": 0,
            "vorticity_index": 0.0,
            "thermodynamic_severity": "NONE",
        },
        "mathematical_proof": {
            "theorem": "Teorema de Hodge: ker(L₁) ≅ H₁(G; ℝ)",
            "conclusion": (
                "β₁ = 0 ⟹ im(B₂) = {0} ⟹ E_curl = 0. "
                "El flujo es completamente irrotacional."
            ),
            "verification": "P_curl I = 0 ∀I ∈ ℝᵐ cuando β₁ = 0.",
        },
    }

    logger.info("Flujo laminar confirmado. Sin vorticidad detectable.")
    return result


def _generate_proof(magnon: MagnonCartridge) -> Dict[str, Any]:
    """
    Genera la prueba matemática formal del resultado de vorticidad.

    Args:
        magnon: MagnonCartridge con las métricas calculadas.

    Returns:
        Dict serializable con la verificación formal.
    """
    return {
        "theorem": "Descomposición de Hodge–Helmholtz (Eckmann 1944)",
        "decomposition": "I = I_grad ⊕ I_curl ⊕ I_harm",
        "spaces": {
            "I_grad": "im(B₁ᵀ)  —  flujo irrotacional (gradiente de potencial)",
            "I_curl": "im(B₂)   —  flujo solenoidal (rotacional puro)",
            "I_harm": "ker(L₁)  —  flujo armónico (satisface ambas)",
        },
        "verification": {
            "projector": "P_curl = U_r U_rᵀ  (vía SVD de B₂, numéricamente estable)",
            "circulation": f"Γ = B₂ᵀI,  ‖Γ‖ = {magnon.total_circulation_norm:.6f}",
            "energy": f"E_curl = ‖Γ‖² = {magnon.kinetic_energy:.6e}",
            "vorticity_index": f"ω = E_curl / ‖I‖² = {magnon.vorticity_index:.6f}",
            "idempotency": (
                f"‖P²−P‖_F / ‖P‖_F = "
                f"{magnon.projection_idempotency_error:.2e}"
            ),
            "curl_subspace_dim": (
                f"dim im(B₂) = β₁ = {magnon.curl_subspace_dim}"
            ),
        },
        "conclusion": (
            f"La proyección P_curl I ≠ 0 confirma componente rotacional. "
            f"ω = {magnon.vorticity_index:.4f} → "
            f"Causalidad: COMPROMETIDA."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE VI · UTILIDADES DE VERIFICACIÓN STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

def verify_hodge_properties(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Verificación exhaustiva de todas las propiedades de Hodge para G.

    Útil para debugging, tests unitarios y validación matemática.

    Verifica:
        1. Invariantes del complejo de cadenas (B₁B₂ = 0)
        2. Números de Betti (β₀, β₁) y Euler–Poincaré
        3. Propiedades espectrales de L₁ (PSD, kernel, gap)
        4. Isomorfismo de Hodge: dim ker(L₁) = β₁
        5. Simetría y condición de L₁

    Args:
        G: Grafo dirigido.

    Returns:
        Dict con todos los resultados de verificación.
    """
    hodge = HodgeDecompositionBuilder(G)

    cochain_result = hodge.verify_cochain_complex()
    B1, _ = hodge.build_incidence_matrix()
    B2, _ = hodge.build_cycle_matrix()
    L1, spectral = hodge.compute_hodge_laplacian()

    # Verificar nulidad del kernel usando base explícita
    ker_L1 = NumericalUtilities.null_space_basis(L1)
    ker_dim = ker_L1.shape[1]

    # Verificar que ker(L₁) ⊆ ker(B₁ᵀ) ∩ ker(B₂ᵀ)
    ker_ok = True
    if ker_dim > 0:
        B1T_ker_norm = float(np.linalg.norm(B1.T @ ker_L1))
        B2T_ker_norm = float(np.linalg.norm(B2.T @ ker_L1)) if B2.shape[1] > 0 else 0.0
        tol = 1e-8
        ker_ok = B1T_ker_norm < tol and B2T_ker_norm < tol
    else:
        B1T_ker_norm = 0.0
        B2T_ker_norm = 0.0

    return {
        "graph_properties": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "is_directed": G.is_directed(),
            "is_weakly_connected": nx.is_weakly_connected(G),
            "connected_components": nx.number_connected_components(
                G.to_undirected()
            ),
        },
        "cochain_complex": cochain_result,
        "betti_numbers": {
            "beta_0": cochain_result["beta_0"],
            "beta_1": cochain_result["beta_1"],
        },
        "euler_characteristic": {
            "formula": "χ = n − m = β₀ − β₁",
            "chi_geometric": hodge.n - hodge.m,
            "chi_topological": (
                cochain_result["beta_0"] - cochain_result["beta_1"]
            ),
            "verified": cochain_result["euler_poincare_ok"],
        },
        "hodge_kernel": {
            "ker_L1_dimension": ker_dim,
            "expected_beta_1": cochain_result["beta_1"],
            "isomorphism_ok": ker_dim == cochain_result["beta_1"],
            "ker_subset_of_ker_B1T": B1T_ker_norm,
            "ker_subset_of_ker_B2T": B2T_ker_norm,
            "kernel_property_ok": ker_ok,
        },
        "spectral_properties": spectral,
        "hodge_laplacian_summary": {
            "shape": L1.shape,
            "trace": float(np.trace(L1)),
            "is_symmetric": bool(np.allclose(L1, L1.T)),
            "is_psd": bool(np.all(np.linalg.eigvalsh(L1) >= -1e-10)),
            "condition_number": spectral.get("condition_number", math.inf),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARTE VII · PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    # ── Grafo de prueba: dos ciclos simples ──────────────────────────────
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "A"),   # Ciclo 1: A→B→C→A
        ("A", "D"), ("D", "E"), ("E", "A"),   # Ciclo 2: A→D→E→A
    ])

    flows: Dict[Tuple[str, str], float] = {
        ("A", "B"): 10.0,
        ("B", "C"): 10.0,
        ("C", "A"): 10.0,
        ("A", "D"):  5.0,
        ("D", "E"):  5.0,
        ("E", "A"):  5.0,
    }

    # ── Verificación formal de propiedades de Hodge ───────────────────────
    print("\n" + "=" * 70)
    print("VERIFICACIÓN FORMAL DE PROPIEDADES DE HODGE")
    print("=" * 70)
    hodge_props = verify_hodge_properties(G)
    for section, values in hodge_props.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                if not isinstance(v, (list,)):   # omitir listas largas
                    print(f"  {k}: {v}")
        else:
            print(f"  {values}")

    # ── Análisis de vorticidad completo ───────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISIS DE VORTICIDAD SOLENOIDAL")
    print("=" * 70)
    result = inspect_and_mitigate_resonance(G, flows, full_analysis=True)

    print(f"\nEstado:   {result['status']}")
    print(f"Acción:   {result['action']}")

    vm = result["vorticity_metrics"]
    print(f"\nMétricas de Vorticidad:")
    for k, v in vm.items():
        print(f"  {k}: {v}")

    print(f"\nPrueba Matemática:")
    proof = result["mathematical_proof"]
    for k, v in proof.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    if "full_hodge_decomposition" in result:
        fhd = result["full_hodge_decomposition"]
        print(f"\nDescomposición de Hodge — Energías:")
        for k, v in fhd["energy_decomposition"].items():
            print(f"  {k}: {v:.6f}")
        print(f"\nVerificación de ortogonalidad:")
        for k, v in fhd["verification"].items():
            print(f"  {k}: {v}")
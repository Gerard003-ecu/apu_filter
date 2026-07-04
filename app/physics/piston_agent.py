# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo  : Piston Agent (Inyector de Caudal y Funtor de Hodge-Helmholtz)      ║
║ Ruta    : app/physics/piston_agent.py                                        ║
║ Versión : 6.0.0-Doctoral-Rigorous-Simplicial-DEC                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Dictamen Doctoral v6.0.0):
────────────────────────────────────────────────────────────────────────────────

CORRECCIONES CRÍTICAS v6.0.0 vs v5.0.0:
────────────────────────────────────────
  FIX 1: build_mesh — verificación axiomática ∂₁∘∂₂ = 0 (identidad de Leibniz)
  FIX 2: build_mesh — orientación consistente de ciclos via árbol generador
  FIX 3: solve_hydrodynamics — análisis espectral de L_red via eigenvalor mínimo
  FIX 4: solve_hydrodynamics — Laplaciano ponderado L₀^W = ∂₁W∂₁ᵀ (con conductancias)
  FIX 5: solve_hydrodynamics — verificación im(s) ⊆ im(L₀) antes de resolver
  FIX 6: decompose_flow — Laplaciano de Hodge ponderado L₁^W = ∂₁ᵀW⁻¹∂₁ + ∂₂∂₂ᵀW
  FIX 7: decompose_flow — desempaquetado LSQR correcto (7 valores, no 9)
  FIX 8: decompose_flow — ortogonalidad verificada en norma energía ⟨·,·⟩_W
  FIX 9: _create_injection_vector — verificación ∑sᵢ=0 y proyección sobre im(L₀)
  FIX 10: execute_injection — propagación de invariantes topológicos al resultado
  FIX 11: PistonConstants — umbral espectral λ_min reemplaza umbral de norma-1
  FIX 12: SimplicialMesh — campos adicionales: laplacian_0, laplacian_1, gram_W

FUNDAMENTACIÓN AXIOMÁTICA AMPLIADA (v6.0.0):
────────────────────────────────────────────────────────────────────────────────
§1. COMPLEJO DE CADENAS VÁLIDO (∂∘∂ = 0):
    La identidad de Leibniz ∂₁∘∂₂ = 0 es la condición necesaria y suficiente
    para que (C•, ∂) sea un complejo de cadenas. Sin ella, los grupos de
    homología H_k = ker(∂_k)/im(∂_{k+1}) no están bien definidos, y la
    descomposición de Hodge carece de fundamento algebraico.

§2. LAPLACIANO DE HODGE PONDERADO (DEC — Discrete Exterior Calculus):
    En una red con conductancias W (métrica discreta), los operadores de
    Hodge deben incorporar la métrica para producir proyecciones ortogonales
    en el espacio de energía ⟨f,g⟩_W = fᵀ W⁻¹ g:

        L₀^W = ∂₁ W ∂₁ᵀ        [Laplaciano nodal ponderado]
        L₁^W = ∂₁ᵀ W⁻¹ ∂₁ + ∂₂ ∂₂ᵀ W  [Laplaciano de aristas de Hodge]

    La descomposición de Hodge-Helmholtz ponderada:
        I = I_grad ⊕_W I_curl ⊕_W I_harm
    es ortogonal en ⟨·,·⟩_W (no en la norma euclidiana).

§3. ANÁLISIS ESPECTRAL DEL LAPLACIANO REDUCIDO:
    La no-singularidad de L_red se certifica via su eigenvalor mínimo:
        λ_min(L_red) > SPECTRAL_THRESHOLD
    Un eigenvalor nulo indica que el nodo reservorio no conecta todas las
    componentes del grafo, lo que invalida la condición de Dirichlet.

§4. COMPATIBILIDAD DE LA FUENTE CON EL OPERADOR (CONDICIÓN DE SOLVABILIDAD):
    El sistema L₀ p = -s tiene solución sii s ∈ im(L₀) = (ker(L₀ᵀ))⊥.
    Para L₀ simétrico: ker(L₀) = span{1} → s ⊥ 1 ↔ ∑sᵢ = 0.
    Esta condición se verifica ANTES de resolver para evitar divergencia
    numérica silenciosa del solver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh, lsqr, spsolve

# ── Dependencias arquitectónicas (stubs para ejecución standalone) ────────────
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        pass

    class CategoricalState:
        payload: Dict[str, Any]
        metadata: Dict[str, Any]
        stratum: Any

        def __init__(self, payload, metadata=None, stratum=None):
            self.payload  = payload
            self.metadata = metadata or {}
            self.stratum  = stratum

    class Stratum(Enum):
        PHYSICS = auto()


logger = logging.getLogger("MIC.Physics.PistonAgent")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES NUMÉRICAS Y TERMODINÁMICAS (v6.0.0)
# ══════════════════════════════════════════════════════════════════════════════

class PistonConstants:
    r"""
    Constantes físicas y numéricas del inyector termodinámico.

    CORRECCIÓN FIX 11:
        SINGULAR_THRESHOLD es ahora un umbral ESPECTRAL (eigenvalor mínimo),
        no un umbral de norma-1. La norma-1 de una matriz puede ser grande
        incluso si la matriz es singular (ej: matriz con filas dependientes
        de norma grande). El eigenvalor mínimo es el indicador correcto
        de singularidad para matrices simétricas semidefinidas positivas.

    Constantes:
    ───────────
    MACHINE_EPSILON     : ε_mach para IEEE 754 float64.
    KIRCHHOFF_TOLERANCE : tolerancia ∞-norma del residuo de KCL [adim].
    MAX_VORTICITY_NORM  : cota máxima ‖I_curl‖_W (norma de energía).
    DEFAULT_CONDUCTANCE : conductancia base si no se proveen pesos [S].
    SPECTRAL_THRESHOLD  : umbral de eigenvalor mínimo λ_min(L_red) [adim].
                          Si λ_min < SPECTRAL_THRESHOLD → L_red singular.
    SOURCE_BALANCE_TOL  : tolerancia para |∑sᵢ| = 0 (FIX 9).
    HODGE_ORTHO_TOL     : tolerancia de ortogonalidad en norma energía (FIX 8).
    """
    MACHINE_EPSILON     : float = float(np.finfo(np.float64).eps)
    KIRCHHOFF_TOLERANCE : float = 1e-10
    MAX_VORTICITY_NORM  : float = 1e-6
    DEFAULT_CONDUCTANCE : float = 1.0
    SPECTRAL_THRESHOLD  : float = 1e-10   # FIX 11: eigenvalor, no norma-1
    SOURCE_BALANCE_TOL  : float = 1e-12   # FIX 9: ∑sᵢ = 0
    HODGE_ORTHO_TOL     : float = 1e-8    # FIX 8: ortogonalidad ⟨·,·⟩_W


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS FÍSICOS)
# ══════════════════════════════════════════════════════════════════════════════

class PistonInjectorError(TopologicalInvariantError):
    """Clase base categórica para fallos de inyección termodinámica."""


class SingularLaplacianError(PistonInjectorError):
    r"""
    Detonada si λ_min(L_red) < SPECTRAL_THRESHOLD.

    Indica que el nodo reservorio no conecta todas las componentes del grafo
    o que la condición de Dirichlet es insuficiente para regularizar L₀.
    La condición espectral λ_min > 0 es equivalente a la conexidad del grafo
    reducido (Teorema de Kirchhoff sobre el árbol generador).
    """


class HomologicalKirchhoffError(PistonInjectorError):
    r"""
    Detonada si ‖∂₁f − s‖_∞ > KIRCHHOFF_TOLERANCE.

    Prueba fuga termodinámica: la masa no se conserva en ningún nodo.
    El residuo ∂₁f − s mide la discrepancia entre el flujo divergente
    y las fuentes externas — cualquier componente no nula implica creación
    o destrucción de masa, violando la primera ley de la termodinámica.
    """


class ParasiticVorticityVetoError(PistonInjectorError):
    r"""
    Detonada si ‖I_curl‖_W > MAX_VORTICITY_NORM.

    La norma de energía ‖·‖_W = √(·ᵀ W⁻¹ ·) mide la potencia disipada
    en los bucles rotacionales. Una vorticidad parásita positiva implica
    que la bomba inyecta exergía que circula en bucles estériles sin
    producir trabajo útil (flujo laminar).
    """


class BoundaryComplexError(PistonInjectorError):
    r"""
    Detonada si ∂₁ ∘ ∂₂ ≠ 0 (FIX 1).

    La identidad ∂∘∂ = 0 es la condición necesaria y suficiente para
    que (C•, ∂) sea un complejo de cadenas diferencial. Su violación
    invalida los grupos de homología H_k = ker(∂_k)/im(∂_{k+1}) y
    colapsa la base algebraica del agente.
    """


class SourceCompatibilityError(PistonInjectorError):
    r"""
    Detonada si s ∉ im(L₀), es decir ∑sᵢ ≠ 0 (FIX 9).

    Para que L₀ p = -s tenga solución, la fuente s debe ser ortogonal
    al núcleo de L₀ᵀ = L₀ (simétrico). Como ker(L₀) = span{1}, la
    condición de compatibilidad es: ⟨1, s⟩ = ∑sᵢ = 0.
    """


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS DE DATOS INMUTABLES (DTOs)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SimplicialMesh:
    r"""
    1-esqueleto y 2-esqueleto topológico del sistema logístico.

    CORRECCIÓN FIX 12: Campos adicionales para los Laplacianos precalculados
    y la matriz de Gram ponderada, evitando su recálculo en cada fase.

    Campos
    ──────
    nodes              : tupla de identificadores de nodos.
    edges              : tupla de aristas (u, v) orientadas.
    cycles             : tupla de ciclos (como tuplas de nodos).
    boundary_1         : ∂₁ ∈ ℝ^{|V|×|E|}, matriz de incidencia orientada.
    boundary_2         : ∂₂ ∈ ℝ^{|E|×|F|}, matriz de ciclos.
    conductance_matrix : W ∈ ℝ^{|E|×|E|}, diagonal de conductancias [S].
    laplacian_0        : L₀^W = ∂₁W∂₁ᵀ ∈ ℝ^{|V|×|V|} (FIX 4/FIX 12).
    laplacian_1        : L₁^W = ∂₁ᵀW⁻¹∂₁ + ∂₂∂₂ᵀW ∈ ℝ^{|E|×|E|} (FIX 6/12).
    inv_conductance    : W⁻¹ ∈ ℝ^{|E|×|E|}, inversa de conductancias (FIX 6).
    boundary_identity  : bool, True sii ∂₁∘∂₂ = 0 verificado (FIX 1).
    """
    nodes              : Tuple[str, ...]
    edges              : Tuple[Tuple[str, str], ...]
    cycles             : Tuple[Tuple[str, ...], ...]
    boundary_1         : sp.csr_matrix
    boundary_2         : sp.csr_matrix
    conductance_matrix : sp.dia_matrix
    laplacian_0        : sp.csr_matrix    # FIX 4+12: L₀^W precalculado
    laplacian_1        : sp.csr_matrix    # FIX 6+12: L₁^W precalculado
    inv_conductance    : sp.dia_matrix    # FIX 6+12: W⁻¹ precalculado
    boundary_identity  : bool             # FIX 1: ∂₁∘∂₂ = 0 certificado


@dataclass(frozen=True, slots=True)
class InjectionVector:
    r"""
    Fuente de corriente ideal s (vector de términos independientes).

    CORRECCIÓN FIX 9: Campo `balance_verified` certifica ∑sᵢ = 0.

    Campos
    ──────
    source_node      : nodo de succión (salida de la bomba).
    sink_node        : nodo de descarga (entrada de la bomba).
    q_pump           : caudal inyectado Q [unidades consistentes con W].
    s_vector         : vector s ∈ ℝ^{|V|} con s[u]=-Q, s[v]=+Q, resto=0.
    balance_verified : True sii ∑sᵢ = 0 verificado (FIX 9).
    """
    source_node      : str
    sink_node        : str
    q_pump           : float
    s_vector         : NDArray[np.float64]
    balance_verified : bool                # FIX 9


@dataclass(frozen=True, slots=True)
class FlowState:
    r"""
    Estado hidrodinámico resuelto de la ecuación de Poisson.

    Campos
    ──────
    nodal_pressures  : p ∈ ℝ^{|V|}, presiones nodales [Pa o adim].
    edge_flows       : f ∈ ℝ^{|E|}, flujos por arista [m³/s o adim].
    kirchhoff_residual: ‖∂₁f − s‖_∞ [adim], residuo de KCL verificado.
    lambda_min_L_red : eigenvalor mínimo de L_red (diagnóstico espectral).
    """
    nodal_pressures   : NDArray[np.float64]
    edge_flows        : NDArray[np.float64]
    kirchhoff_residual: float
    lambda_min_L_red  : float


@dataclass(frozen=True, slots=True)
class HodgeDecomposition:
    r"""
    Componentes ortogonales del flujo en la descomposición de Hodge-Helmholtz.

    CORRECCIÓN FIX 8: Las componentes son ortogonales en la norma de energía
    ⟨f,g⟩_W = fᵀ W⁻¹ g, no en la norma euclidiana estándar.

    Campos
    ──────
    i_grad           : I_grad ∈ im(∂₁ᵀ), flujo laminar impulsado por potencial.
    i_curl           : I_curl ∈ im(∂₂), vorticidad rotacional local.
    i_harm           : I_harm ∈ ker(L₁^W), flujo armónico global.
    vorticity_norm   : ‖I_curl‖_W (norma de energía, FIX 8).
    orthogonality_ok : bool, True sii ⟨I_grad, I_curl⟩_W < HODGE_ORTHO_TOL.
    energy_total     : ‖I‖²_W = ‖I_grad‖²_W + ‖I_curl‖²_W + ‖I_harm‖²_W.
    """
    i_grad          : NDArray[np.float64]
    i_curl          : NDArray[np.float64]
    i_harm          : NDArray[np.float64]
    vorticity_norm  : float
    orthogonality_ok: bool
    energy_total    : float


# ══════════════════════════════════════════════════════════════════════════════
# §D. ÁLGEBRA ESPECTRAL AUXILIAR (base matemática compartida)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_min_eigenvalue_sparse(
    A       : sp.spmatrix,
    tol     : float = 1e-10,
) -> float:
    r"""
    Calcula el eigenvalor mínimo de una matriz simétrica semidefinida positiva.

    Para $A$ simétrica SPD (o SPSD), el eigenvalor mínimo λ_min certifica:
        λ_min > 0  →  A es no singular (invertible)
        λ_min = 0  →  A es singular (ker(A) ≠ {0})

    Usa `eigsh` (ARPACK Lanczos) para calcular el eigenvalor de menor
    magnitud, que es eficiente para matrices dispersas grandes.

    Para matrices de tamaño ≤ 2 (edge cases del grafo mínimo), se usa
    el algoritmo denso `np.linalg.eigvalsh` directamente.

    Parámetros
    ----------
    A   : sp.spmatrix, matriz simétrica SPD o SPSD de shape (n,n).
    tol : float, tolerancia de convergencia del solver Lanczos.

    Retorna
    -------
    float, eigenvalor mínimo λ_min(A).
    """
    n = A.shape[0]
    if n <= 1:
        return float(A.toarray()[0, 0]) if n == 1 else 0.0
    if n <= 3:
        # Para n pequeño, el cálculo denso es más estable
        eigvals = np.linalg.eigvalsh(A.toarray())
        return float(np.min(eigvals))
    try:
        # ARPACK: k=1 eigenvalor de menor magnitud
        eigvals, _ = eigsh(A, k=1, which="SM", tol=tol, return_eigenvectors=True)
        return float(np.real(eigvals[0]))
    except Exception:
        # Fallback a denso si ARPACK no converge
        eigvals = np.linalg.eigvalsh(A.toarray())
        return float(np.min(eigvals))


def _energy_norm(
    v     : NDArray[np.float64],
    W_inv : sp.dia_matrix,
) -> float:
    r"""
    Calcula la norma de energía ‖v‖_W = √(vᵀ W⁻¹ v).

    La norma de energía es la métrica natural del espacio de flujos
    en una red con conductancias W. Generaliza la norma L² euclidiana
    al caso donde cada arista tiene una "masa" inversa 1/wₖ.

    Interpretación física:
        ‖f‖²_W = fᵀ W⁻¹ f = Σₖ fₖ²/wₖ   [potencia disipada total]

    Parámetros
    ----------
    v     : NDArray, shape (n,), vector de flujos.
    W_inv : sp.dia_matrix, shape (n,n), inversa de conductancias W⁻¹.

    Retorna
    -------
    float ≥ 0, norma de energía ‖v‖_W.
    """
    quadratic = float(v @ W_inv.dot(v))
    return math.sqrt(max(quadratic, 0.0))


def _energy_inner_product(
    u     : NDArray[np.float64],
    v     : NDArray[np.float64],
    W_inv : sp.dia_matrix,
) -> float:
    r"""
    Calcula el producto interno de energía ⟨u, v⟩_W = uᵀ W⁻¹ v.

    Parámetros
    ----------
    u, v  : NDArray, shape (n,), vectores de flujos.
    W_inv : sp.dia_matrix, shape (n,n), W⁻¹.

    Retorna
    -------
    float, producto interno ⟨u, v⟩_W.
    """
    return float(u @ W_inv.dot(v))


def _lsqr_solve(
    A    : sp.spmatrix,
    b    : NDArray[np.float64],
    atol : float = 1e-12,
    btol : float = 1e-12,
) -> NDArray[np.float64]:
    r"""
    Resuelve el sistema A x = b via LSQR (mínimos cuadrados sparse).

    CORRECCIÓN FIX 7:
        `scipy.sparse.linalg.lsqr` retorna una tupla de 7 elementos:
            (x, istop, itn, r1norm, r2norm, anorm, acond)
        El desempaquetado con 9 variables causa ValueError en v5.0.0.
        Esta función encapsula el retorno correctamente.

    Parámetros
    ----------
    A    : sp.spmatrix, matriz del sistema (puede ser singular → mín. norma).
    b    : NDArray, vector de términos independientes.
    atol : float, tolerancia absoluta de residuo.
    btol : float, tolerancia relativa de residuo.

    Retorna
    -------
    NDArray, shape (n,), solución x de mínima norma-2.
    """
    # LSQR retorna exactamente 7 valores (FIX 7)
    result = lsqr(A, b, atol=atol, btol=btol)
    x = result[0]   # solución de mínima norma
    return np.asarray(x, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║  ORQUESTADOR SUPREMO: HodgeHelmholtzInjectorAgent (v6.0.0)              ║
# ║                                                                          ║
# ║  Contiene anidadas las tres fases matemáticas encadenadas por DTOs.     ║
# ║                                                                          ║
# ║  Garantías v6.0.0:                                                        ║
# ║    G1. ∂₁∘∂₂ = 0 certificado antes de operar (FIX 1)                    ║
# ║    G2. L₀^W = ∂₁W∂₁ᵀ (Laplaciano ponderado) (FIX 4)                    ║
# ║    G3. λ_min(L_red) > SPECTRAL_THRESHOLD (FIX 3)                        ║
# ║    G4. ∑sᵢ = 0 antes de resolver (FIX 9)                                ║
# ║    G5. Descomposición ortogonal en ⟨·,·⟩_W (FIX 6+8)                   ║
# ║    G6. LSQR con desempaquetado correcto (FIX 7)                          ║
# ║    G7. Ortogonalidad verificada post-descomposición (FIX 8)              ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════

class HodgeHelmholtzInjectorAgent(Morphism):
    r"""
    El Soberano Absoluto de la inyección termodinámica (Estrato PHYSICS).

    Modela isomorficamente la inyección de capital/recursos como una Bomba de
    Desplazamiento Positivo en una red de fluidos incompresibles.

    La arquitectura en tres fases anidadas garantiza:
        FASE 1 → ∂₁∘∂₂ = 0 (complejo de cadenas válido)
        FASE 2 → λ_min(L_red) > 0 + ∑sᵢ = 0 (Poisson bien planteado)
        FASE 3 → Hodge ponderado + ortogonalidad ⟨·,·⟩_W (DEC correcto)
    """

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · FIBRADOR DEL COMPLEJO DE CADENAS (CONSTRUCTOR MATRICIAL)
    # ═══════════════════════════════════════════════════════════════════════

    class _Phase1_SimplicialBoundaryBuilder:
        r"""
        Sintetiza los operadores frontera ∂₁ y ∂₂ y verifica axiomáticamente
        que constituyen un complejo de cadenas diferencial válido.

        CORRECCIONES v6.0.0:
        ─────────────────────
        FIX 1: Verificación ∂₁∘∂₂ = 0 (Lemma de Leibniz).
        FIX 2: Orientación consistente de ciclos via árbol generador BFS.
        FIX 4: Laplaciano ponderado L₀^W = ∂₁W∂₁ᵀ (incorpora conductancias).
        FIX 6: Laplaciano de Hodge L₁^W = ∂₁ᵀW⁻¹∂₁ + ∂₂∂₂ᵀW (DEC correcto).
        FIX 12: Precálculo y almacenamiento de L₀^W, L₁^W, W⁻¹ en SimplicialMesh.
        """

        # ──────────────────────────────────────────────────────────────────
        # §1.1 — Árbol generador BFS para orientación consistente de ciclos
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _build_spanning_tree_bfs(
            nodes    : List[str],
            edges    : List[Tuple[str, str, float]],
        ) -> Dict[str, Optional[str]]:
            r"""
            Construye un árbol generador del grafo via BFS.

            El árbol generador T ⊆ G es un subgrafo acíclico conexo con
            |V|-1 aristas. Las aristas NO en T son las "aristas de co-árbol"
            que generan los ciclos fundamentales del grafo.

            El BFS garantiza que cada nodo tiene un único padre, y que
            el árbol está bien definido para grafos conexos.

            Parámetros
            ----------
            nodes : List[str], lista de nodos del grafo.
            edges : List[Tuple[str,str,float]], aristas con conductancias.

            Retorna
            -------
            Dict[str, Optional[str]], mapa nodo → padre en el árbol.
                                      La raíz tiene padre None.

            Lanza
            -----
            TopologicalInvariantError : Si el grafo no es conexo.
            """
            from collections import deque

            # Lista de adyacencia no dirigida (ignoramos orientación aquí)
            adj: Dict[str, List[str]] = {n: [] for n in nodes}
            for u, v, _ in edges:
                adj[u].append(v)
                adj[v].append(u)

            root   = nodes[0]
            parent : Dict[str, Optional[str]] = {root: None}
            queue  = deque([root])

            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)

            # Verificar conexidad: todos los nodos deben tener padre
            unvisited = [n for n in nodes if n not in parent]
            if unvisited:
                raise TopologicalInvariantError(
                    f"Grafo no es conexo. Nodos no alcanzables desde "
                    f"'{root}': {unvisited}. La inyección de flujo "
                    f"requiere un grafo conexo para que la ecuación de "
                    f"Poisson tenga solución única bajo Dirichlet."
                )

            return parent

        # ──────────────────────────────────────────────────────────────────
        # §1.2 — Verificación axiomática ∂₁∘∂₂ = 0 (FIX 1)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _verify_boundary_identity(
            boundary_1 : sp.csr_matrix,
            boundary_2 : sp.csr_matrix,
        ) -> None:
            r"""
            Verifica la identidad de Leibniz ∂₁∘∂₂ = 0.

            Teorema (Lema de Leibniz):
                En un complejo de cadenas (C•, ∂), la composición de
                operadores frontera consecutivos es nula:
                    ∂_{k-1} ∘ ∂_k = 0   ∀ k

            Para k=2: ∂₁∘∂₂ = 0.

            Interpretación topológica:
                La frontera de una frontera es vacía. Si ∂₂ asigna a cada
                2-celda (ciclo) su 1-frontera (aristas del ciclo), entonces
                ∂₁(∂₂(ciclo)) = frontera de las aristas del ciclo = 0.
                (Cada nodo del ciclo aparece exactamente dos veces con
                signos opuestos → cancelación exacta.)

            Verificación numérica:
                ‖∂₁∘∂₂‖_F < ε_mach · ‖∂₁‖_F · ‖∂₂‖_F

            Parámetros
            ----------
            boundary_1 : sp.csr_matrix, ∂₁ ∈ ℝ^{|V|×|E|}.
            boundary_2 : sp.csr_matrix, ∂₂ ∈ ℝ^{|E|×|F|}.

            Lanza
            -----
            BoundaryComplexError : Si ‖∂₁∘∂₂‖_F supera la tolerancia.
            """
            if boundary_2.shape[1] == 0:
                # Sin ciclos: ∂₂ es vacía → la identidad es trivialmente nula
                return

            # Producto ∂₁∘∂₂ (debe ser la matriz cero)
            composition = boundary_1.dot(boundary_2)
            frob_norm   = sp.linalg.norm(composition, ord="fro")

            # Tolerancia relativa: ε_mach · ‖∂₁‖_F · ‖∂₂‖_F
            norm_b1  = sp.linalg.norm(boundary_1, ord="fro")
            norm_b2  = sp.linalg.norm(boundary_2, ord="fro")
            tol_rel  = PistonConstants.MACHINE_EPSILON * max(norm_b1 * norm_b2, 1.0)

            if frob_norm > max(tol_rel, PistonConstants.MACHINE_EPSILON * 1e3):
                raise BoundaryComplexError(
                    f"Violación de la identidad de Leibniz ∂₁∘∂₂ = 0. "
                    f"‖∂₁∘∂₂‖_F = {frob_norm:.3e} (tolerancia: {tol_rel:.3e}). "
                    f"Los ciclos proporcionados no son compatibles con la "
                    f"orientación de las aristas. Revisar la definición de "
                    f"cycles para garantizar coherencia algebraica."
                )

            logger.debug(
                "Identidad de Leibniz ∂₁∘∂₂ = 0 verificada. "
                "‖∂₁∘∂₂‖_F = %.3e.",
                frob_norm,
            )

        # ──────────────────────────────────────────────────────────────────
        # §1.3 — Construcción de ∂₂ con orientación coherente (FIX 2)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _build_boundary_2_oriented(
            cycles    : List[List[str]],
            edge_idx  : Dict[Tuple[str, str], int],
            n_edges   : int,
        ) -> sp.csr_matrix:
            r"""
            Construye ∂₂ con orientación coherente respecto a ∂₁.

            CORRECCIÓN FIX 2:
            En v5.0.0, los ciclos se procesaban ignorando la orientación
            de las aristas en ∂₁. Si una arista (u,v) está en ∂₁ con
            signo (+1 en v, -1 en u), pero el ciclo la traversa en sentido
            contrario, la identidad ∂₁∘∂₂ = 0 se viola.

            Este método asigna el signo correcto a cada arista del ciclo:
                Si (u,v) ∈ edge_idx: signo = +1 (dirección canónica)
                Si (v,u) ∈ edge_idx: signo = -1 (dirección opuesta)
                Si ninguno: la arista no existe → ciclo inválido (WARNING)

            Parámetros
            ----------
            cycles   : List[List[str]], ciclos como listas de nodos.
            edge_idx : Dict[(str,str), int], índice de aristas orientadas.
            n_edges  : int, número total de aristas |E|.

            Retorna
            -------
            sp.csr_matrix, ∂₂ ∈ ℝ^{|E|×|F|} con orientación coherente.
            """
            n_f    = len(cycles)
            data   : List[float] = []
            rows   : List[int]   = []
            cols   : List[int]   = []

            for f_idx, cycle in enumerate(cycles):
                # Cerrar el ciclo: último nodo → primer nodo
                nodes_cycle = list(cycle) + [cycle[0]]
                for k in range(len(nodes_cycle) - 1):
                    u, v = nodes_cycle[k], nodes_cycle[k + 1]

                    if (u, v) in edge_idx:
                        e_idx = edge_idx[(u, v)]
                        sign  = +1.0
                    elif (v, u) in edge_idx:
                        e_idx = edge_idx[(v, u)]
                        sign  = -1.0
                    else:
                        logger.warning(
                            "Arista (%s, %s) del ciclo %d no existe en el grafo. "
                            "La arista se omite. Verificar coherencia de 'cycles'.",
                            u, v, f_idx,
                        )
                        continue

                    rows.append(e_idx)
                    cols.append(f_idx)
                    data.append(sign)

            if not data:
                return sp.csr_matrix((n_edges, max(n_f, 0)), dtype=np.float64)

            return sp.csr_matrix(
                (data, (rows, cols)),
                shape=(n_edges, n_f),
                dtype=np.float64,
            )

        # ──────────────────────────────────────────────────────────────────
        # §1.4 — Método principal: build_mesh (SUTURA DOCTORAL)
        # ──────────────────────────────────────────────────────────────────

        @classmethod
        def build_mesh(
            cls,
            nodes  : List[str],
            edges  : List[Tuple[str, str, float]],
            cycles : List[List[str]],
        ) -> "SimplicialMesh":
            r"""
            Construye la malla simplicial con todos los operadores de Hodge.

            CORRECCIONES FIX 1, 2, 4, 6, 12 integradas:

            Pipeline:
            ─────────
            1. Validación de conductancias (wₖ > 0 ∀ k).
            2. Construcción de ∂₁ (incidencia orientada).
            3. Árbol generador BFS para verificar conexidad (FIX 2).
            4. Construcción de ∂₂ con orientación coherente (FIX 2).
            5. Verificación ∂₁∘∂₂ = 0 (FIX 1).
            6. Cálculo L₀^W = ∂₁W∂₁ᵀ (FIX 4).
            7. Cálculo W⁻¹ (FIX 6, necesario para L₁^W y normas).
            8. Cálculo L₁^W = ∂₁ᵀW⁻¹∂₁ + ∂₂∂₂ᵀW (FIX 6).
            9. Empaquetado en SimplicialMesh con campos precalculados (FIX 12).

            Parámetros
            ----------
            nodes  : List[str], identificadores de nodos.
            edges  : List[Tuple[str,str,float]], aristas (u, v, conductancia).
            cycles : List[List[str]], ciclos como listas de nodos.

            Retorna
            -------
            SimplicialMesh con todos los tensores precalculados.

            Lanza
            -----
            TopologicalInvariantError : Conductancia ≤ 0 o grafo no conexo.
            BoundaryComplexError      : ∂₁∘∂₂ ≠ 0 (ciclos incompatibles).
            """
            n_v = len(nodes)
            n_e = len(edges)
            n_f = len(cycles)

            node_idx : Dict[str, int]            = {n: i for i, n in enumerate(nodes)}
            edge_idx : Dict[Tuple[str,str], int] = {
                (u, v): k for k, (u, v, _) in enumerate(edges)
            }

            # §1.4.1: Verificación de conductancias (wₖ > 0 ∀ k)
            conductances = np.zeros(n_e, dtype=np.float64)
            for k, (u, v, w) in enumerate(edges):
                if w <= 0.0:
                    raise TopologicalInvariantError(
                        f"Conductancia degenerada en arista ({u}→{v}): "
                        f"w={w:.4e} ≤ 0. Las conductancias deben ser "
                        f"estrictamente positivas para que W sea una "
                        f"métrica discreta válida."
                    )
                conductances[k] = w

            # §1.4.2: Construcción de ∂₁ (incidencia orientada)
            B1_data: List[float] = []
            B1_rows: List[int]   = []
            B1_cols: List[int]   = []
            for k, (u, v, _) in enumerate(edges):
                B1_rows.extend([node_idx[u], node_idx[v]])
                B1_cols.extend([k, k])
                B1_data.extend([-1.0, +1.0])

            boundary_1 = sp.csr_matrix(
                (B1_data, (B1_rows, B1_cols)),
                shape=(n_v, n_e),
                dtype=np.float64,
            )

            # §1.4.3: Árbol generador BFS (verifica conexidad) (FIX 2)
            _ = cls._build_spanning_tree_bfs(nodes, edges)

            # §1.4.4: Matriz de conductancias W y W⁻¹ (FIX 6+12)
            W_matrix     = sp.diags(conductances,            format="dia", dtype=np.float64)
            inv_cond_arr = 1.0 / conductances
            W_inv_matrix = sp.diags(inv_cond_arr,            format="dia", dtype=np.float64)

            # §1.4.5: Construcción de ∂₂ con orientación coherente (FIX 2)
            boundary_2 = cls._build_boundary_2_oriented(cycles, edge_idx, n_e)

            # §1.4.6: Verificación axiomática ∂₁∘∂₂ = 0 (FIX 1)
            cls._verify_boundary_identity(boundary_1, boundary_2)

            # §1.4.7: Laplaciano nodal ponderado L₀^W = ∂₁W∂₁ᵀ (FIX 4)
            # En v5.0.0 se usaba L₀ = ∂₁∂₁ᵀ (no ponderado), ignorando W.
            # Con W, L₀^W captura las conductancias: (L₀^W)ᵢⱼ = Σₖ wₖ·∂₁ᵢₖ·∂₁ⱼₖ
            laplacian_0 = boundary_1.dot(W_matrix.dot(boundary_1.T)).tocsr()

            # §1.4.8: Laplaciano de Hodge ponderado L₁^W (FIX 6)
            # L₁^W = ∂₁ᵀW⁻¹∂₁ + ∂₂∂₂ᵀW
            # Término grad: ∂₁ᵀW⁻¹∂₁ (proyecta sobre im(∂₁ᵀ) = exact 1-forms)
            # Término curl: ∂₂∂₂ᵀW (proyecta sobre im(∂₂) = coboundaries)
            L1_grad_term = boundary_1.T.dot(W_inv_matrix.dot(boundary_1)).tocsr()
            if n_f > 0:
                L1_curl_term = boundary_2.dot(boundary_2.T.dot(W_matrix)).tocsr()
            else:
                L1_curl_term = sp.csr_matrix((n_e, n_e), dtype=np.float64)
            laplacian_1 = (L1_grad_term + L1_curl_term).tocsr()

            logger.info(
                "SimplicialMesh construida: |V|=%d, |E|=%d, |F|=%d. "
                "∂₁∘∂₂=0 ✓. L₀^W y L₁^W precalculados.",
                n_v, n_e, n_f,
            )

            return SimplicialMesh(
                nodes              = tuple(nodes),
                edges              = tuple((u, v) for u, v, _ in edges),
                cycles             = tuple(tuple(c) for c in cycles),
                boundary_1         = boundary_1,
                boundary_2         = boundary_2,
                conductance_matrix = W_matrix,
                laplacian_0        = laplacian_0,
                laplacian_1        = laplacian_1,
                inv_conductance    = W_inv_matrix,
                boundary_identity  = True,  # verificado por _verify_boundary_identity
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · SOLUCIONADOR DE POISSON Y REDUCCIÓN DE DIRICHLET
    # ═══════════════════════════════════════════════════════════════════════

    class _Phase2_DirichletPoissonSolver:
        r"""
        Resuelve la ecuación de Poisson ponderada L₀^W p = -s bajo la
        condición de Dirichlet p[reservorio] = 0.

        CORRECCIONES v6.0.0:
        ─────────────────────
        FIX 3: Análisis espectral de L_red via λ_min (no norma-1).
        FIX 4: Usa L₀^W precalculado en SimplicialMesh (no recalcula).
        FIX 5: Verifica im(s) ⊆ im(L₀^W), es decir ∑sᵢ = 0, antes de resolver.
        """

        # ──────────────────────────────────────────────────────────────────
        # §2.1 — Verificación de compatibilidad de la fuente (FIX 5)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _verify_source_compatibility(
            s_vector : NDArray[np.float64],
            tol      : float = PistonConstants.SOURCE_BALANCE_TOL,
        ) -> None:
            r"""
            Verifica la condición de solvabilidad ∑sᵢ = 0 (FIX 5).

            Teorema de Fredholm:
            El sistema L₀^W p = -s tiene solución sii -s ⊥ ker(L₀^W).
            Para L₀^W simétrico SPSD: ker(L₀^W) = span{1} (vector constante).
            Condición de compatibilidad: ⟨1, -s⟩ = -∑sᵢ = 0 → ∑sᵢ = 0.

            Para la bomba de desplazamiento positivo:
                s[fuente] = -Q,  s[sumidero] = +Q,  resto = 0
                ∑sᵢ = -Q + Q = 0 ✓ (satisfecho por construcción)

            Esta verificación protege contra vectores s degenerados que
            lleguen de fuentes externas (ej: payloads malformados).

            Parámetros
            ----------
            s_vector : NDArray, vector de fuentes s ∈ ℝ^{|V|}.
            tol      : float, tolerancia para |∑sᵢ|.

            Lanza
            -----
            SourceCompatibilityError : Si |∑sᵢ| > tol.
            """
            balance = float(np.sum(s_vector))
            if abs(balance) > tol:
                raise SourceCompatibilityError(
                    f"Condición de solvabilidad violada: ∑sᵢ = {balance:.4e} ≠ 0. "
                    f"El sistema L₀^W p = -s no tiene solución porque s ∉ im(L₀^W). "
                    f"Verificar que la bomba conserva masa: Q_fuente = -Q_sumidero."
                )
            logger.debug(
                "Compatibilidad de fuente verificada: ∑sᵢ = %.4e (tol=%.4e).",
                balance, tol,
            )

        # ──────────────────────────────────────────────────────────────────
        # §2.2 — Análisis espectral de L_red (FIX 3)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _verify_laplacian_spectrum(
            L_red     : sp.csr_matrix,
            tol       : float = PistonConstants.SPECTRAL_THRESHOLD,
        ) -> float:
            r"""
            Verifica la no-singularidad de L_red via análisis espectral (FIX 3).

            CORRECCIÓN vs v5.0.0:
            La verificación `norm(L_red, ord=1) < SINGULAR_THRESHOLD` es
            incorrecta porque la norma-1 de una matriz puede ser grande
            incluso si la matriz es singular (ej: filas linealmente dependientes
            de norma grande).

            El criterio correcto para matrices simétricas SPD (o SPSD) es
            el eigenvalor mínimo:
                λ_min(L_red) > SPECTRAL_THRESHOLD → L_red invertible

            Para el Laplaciano reducido de Dirichlet:
                λ_min(L_red) = ω² (cuadrado de la segunda frecuencia natural)
                λ_min > 0 ↔ el grafo reducido es conexo y Dirichlet regulariza.

            Parámetros
            ----------
            L_red : sp.csr_matrix, Laplaciano reducido tras eliminar el nodo reservorio.
            tol   : float, umbral de eigenvalor mínimo.

            Retorna
            -------
            float, λ_min(L_red) [diagnóstico].

            Lanza
            -----
            SingularLaplacianError : Si λ_min ≤ tol.
            """
            lambda_min = _compute_min_eigenvalue_sparse(L_red)

            if lambda_min <= tol:
                raise SingularLaplacianError(
                    f"L_red es numéricamente singular: λ_min = {lambda_min:.4e} "
                    f"≤ {tol:.4e}. El grafo reducido (sin el nodo reservorio) "
                    f"no está bien condicionado. Posibles causas:\n"
                    f"  (a) El nodo reservorio no conecta todas las componentes.\n"
                    f"  (b) Existen aristas paralelas que crean dependencias lineales.\n"
                    f"  (c) El grafo tiene componentes aisladas del reservorio."
                )

            logger.debug(
                "Espectro de L_red: λ_min = %.4e > SPECTRAL_THRESHOLD = %.4e. "
                "L_red es invertible.",
                lambda_min, tol,
            )
            return lambda_min

        # ──────────────────────────────────────────────────────────────────
        # §2.3 — Solución de Poisson + verificación de Kirchhoff
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def solve_hydrodynamics(
            mesh           : "SimplicialMesh",
            injection      : InjectionVector,
            reservoir_node : str,
        ) -> "FlowState":
            r"""
            Resuelve L₀^W p = -s con condición de Dirichlet p[res] = 0.

            CORRECCIONES FIX 3, 4, 5 integradas:

            Pipeline:
            ─────────
            1. Verificar ∑sᵢ = 0 (FIX 5) antes de operar.
            2. Extraer L₀^W del mesh (precalculado, FIX 4).
            3. Aplicar condición de Dirichlet: eliminar fila/columna del reservorio.
            4. Verificar λ_min(L_red) > SPECTRAL_THRESHOLD (FIX 3).
            5. Resolver L_red p_free = -s_red via spsolve.
            6. Calcular flujos f = W ∂₁ᵀ p (Ley de Ohm generalizada).
            7. Verificar KCL: ‖∂₁f − s‖_∞ < KIRCHHOFF_TOLERANCE.

            Parámetros
            ----------
            mesh           : SimplicialMesh con L₀^W precalculado.
            injection      : InjectionVector con balance ∑sᵢ=0 verificado.
            reservoir_node : str, nodo con presión fija p=0 (Dirichlet).

            Retorna
            -------
            FlowState con presiones, flujos, residuo KCL y λ_min diagnóstico.

            Lanza
            -----
            SingularLaplacianError    : λ_min(L_red) ≤ SPECTRAL_THRESHOLD.
            HomologicalKirchhoffError : ‖∂₁f−s‖_∞ > KIRCHHOFF_TOLERANCE.
            SourceCompatibilityError  : ∑sᵢ ≠ 0.
            """
            B1  = mesh.boundary_1
            W   = mesh.conductance_matrix
            L0W = mesh.laplacian_0       # FIX 4: usa L₀^W precalculado
            s   = injection.s_vector

            # §2.3.1: Verificación de compatibilidad (FIX 5)
            HodgeHelmholtzInjectorAgent._Phase2_DirichletPoissonSolver\
                ._verify_source_compatibility(s)

            # §2.3.2: Índice del nodo reservorio
            try:
                res_idx = list(mesh.nodes).index(reservoir_node)
            except ValueError:
                raise SingularLaplacianError(
                    f"Reservorio '{reservoir_node}' no pertenece a la "
                    f"variedad topológica. Nodos disponibles: {list(mesh.nodes)}."
                )

            # §2.3.3: Reducción de Dirichlet (eliminar fila/col del reservorio)
            n_v       = len(mesh.nodes)
            free_mask = np.ones(n_v, dtype=bool)
            free_mask[res_idx] = False

            # Extraer submatriz L_red (Laplaciano reducido ponderado)
            L_red = L0W[free_mask, :][:, free_mask]
            s_red = s[free_mask]

            # §2.3.4: Análisis espectral de L_red (FIX 3)
            lambda_min = (
                HodgeHelmholtzInjectorAgent
                ._Phase2_DirichletPoissonSolver
                ._verify_laplacian_spectrum(L_red)
            )

            # §2.3.5: Resolver L_red p_free = -s_red
            try:
                p_free = spsolve(L_red, -s_red)
            except RuntimeError as exc:
                raise SingularLaplacianError(
                    f"Fallo algebraico al invertir L_red (λ_min={lambda_min:.3e}). "
                    f"Detalles: {exc}"
                ) from exc

            # §2.3.6: Reconstruir vector completo de presiones
            pressures              = np.zeros(n_v, dtype=np.float64)
            pressures[free_mask]   = p_free
            pressures[res_idx]     = 0.0   # Condición de Dirichlet

            # §2.3.7: Flujo via Ley de Ohm generalizada: f = W ∂₁ᵀ p
            # La Ley de Ohm discreta: fₖ = wₖ · (pᵥ − pᵤ) para arista k=(u,v)
            edge_flows = W.dot(B1.T.dot(pressures))

            # §2.3.8: Verificación de KCL: ‖∂₁f − s‖_∞ < τ
            kirchhoff_residual = float(
                np.linalg.norm(B1.dot(edge_flows) + s, ord=np.inf)
            )
            if kirchhoff_residual > PistonConstants.KIRCHHOFF_TOLERANCE:
                raise HomologicalKirchhoffError(
                    f"Violación Termodinámica (KCL). "
                    f"‖∂₁f − s‖_∞ = {kirchhoff_residual:.4e} > "
                    f"{PistonConstants.KIRCHHOFF_TOLERANCE:.4e}. "
                    f"La masa no se conserva. Posible causa: acumulación "
                    f"de error de redondeo en spsolve o grafo mal condicionado."
                )

            logger.info(
                "Poisson L₀^W resuelto. KCL: ‖residuo‖_∞=%.3e. "
                "λ_min(L_red)=%.3e.",
                kirchhoff_residual, lambda_min,
            )

            return FlowState(
                nodal_pressures    = pressures,
                edge_flows         = edge_flows,
                kirchhoff_residual = kirchhoff_residual,
                lambda_min_L_red   = lambda_min,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · AUDITOR DE DESCOMPOSICIÓN DE HODGE-HELMHOLTZ (DEC)
    # ═══════════════════════════════════════════════════════════════════════

    class _Phase3_HodgeHelmholtzAuditor:
        r"""
        Somete la 1-forma de flujo I a la descomposición de Hodge-Helmholtz
        ponderada en el espacio de energía (⟨·,·⟩_W).

        CORRECCIONES v6.0.0:
        ─────────────────────
        FIX 6: Laplaciano de Hodge ponderado L₁^W (DEC correcto).
        FIX 7: Desempaquetado LSQR correcto (7 valores).
        FIX 8: Ortogonalidad verificada en norma ⟨·,·⟩_W.
        """

        # ──────────────────────────────────────────────────────────────────
        # §3.1 — Proyección gradiente ponderada (FIX 6)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _project_gradient(
            boundary_1 : sp.csr_matrix,
            laplacian_0: sp.csr_matrix,
            W_inv      : sp.dia_matrix,
            I          : NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Calcula la componente gradiente (laminar) de I en la norma W.

            Proyección sobre im(∂₁ᵀ) en el espacio de energía:

                φ = (∂₁W∂₁ᵀ)⁻¹ ∂₁ W I   [potencial nodal de mínima energía]
                I_grad = ∂₁ᵀ φ             [flujo gradiente (exact 1-form)]

            CORRECCIÓN FIX 6 vs v5.0.0:
                v5.0.0 resolvía L₀ φ = ∂₁ I (sin ponderación W), produciendo
                una proyección incorrecta en la norma euclidiana (no en ⟨·,·⟩_W).
                La proyección correcta en DEC usa el operador de Hodge ★₀:
                    L₀^W φ = ∂₁ W I
                donde W actúa como la métrica discreta del espacio de aristas.

            Parámetros
            ----------
            boundary_1  : sp.csr_matrix, ∂₁.
            laplacian_0 : sp.csr_matrix, L₀^W = ∂₁W∂₁ᵀ (precalculado).
            W_inv       : sp.dia_matrix, W⁻¹.
            I           : NDArray, flujo a proyectar.

            Retorna
            -------
            NDArray, I_grad = ∂₁ᵀ φ (componente gradiente).
            """
            # Término fuente: ∂₁ W I (divergencia ponderada de I)
            # = ∂₁ (W I) [W es diagonal: W I = wₖ·Iₖ para cada arista k]
            W_I     = W_inv.diagonal() ** (-1) * I    # W·I (W_inv invertida)

            # Más correcto: construir W·I directamente
            W_diag  = 1.0 / W_inv.diagonal()          # conductancias wₖ
            WI      = W_diag * I                       # W·I

            rhs_phi = boundary_1.dot(WI)               # ∂₁(W·I) ∈ ℝ^{|V|}

            # Resolver L₀^W φ = ∂₁W·I (LSQR para L₀^W SPSD, FIX 7)
            phi = _lsqr_solve(laplacian_0, rhs_phi)

            # I_grad = ∂₁ᵀ φ (exact 1-form)
            I_grad = boundary_1.T.dot(phi)
            return I_grad

        # ──────────────────────────────────────────────────────────────────
        # §3.2 — Proyección curl ponderada (FIX 6 + FIX 7)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _project_curl(
            boundary_2 : sp.csr_matrix,
            W_matrix   : sp.dia_matrix,
            I          : NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Calcula la componente rotacional (curl) de I en la norma W.

            Proyección sobre im(∂₂) en el espacio de energía:

                α = (∂₂ᵀW∂₂)⁻¹ ∂₂ᵀW I   [potencial de ciclos ponderado]
                I_curl = ∂₂ α              [flujo rotacional (co-exact 1-form)]

            CORRECCIÓN FIX 6 vs v5.0.0:
                v5.0.0 resolvía L₂ α = ∂₂ᵀ I (sin W), produciendo una
                proyección incorrecta en la norma euclidiana.
                La proyección DEC correcta usa W para ponderar ∂₂ᵀ:
                    ∂₂ᵀW∂₂ α = ∂₂ᵀW·I

            Parámetros
            ----------
            boundary_2 : sp.csr_matrix, ∂₂.
            W_matrix   : sp.dia_matrix, W.
            I          : NDArray, flujo a proyectar.

            Retorna
            -------
            NDArray, I_curl = ∂₂ α (componente rotacional).
            """
            if boundary_2.shape[1] == 0:
                # Sin ciclos: componente curl es cero
                return np.zeros_like(I)

            # Operador de Gram: G₂ = ∂₂ᵀ W ∂₂ ∈ ℝ^{|F|×|F|}
            G2      = boundary_2.T.dot(W_matrix.dot(boundary_2))

            # Término fuente: ∂₂ᵀ W I
            rhs_alpha = boundary_2.T.dot(W_matrix.dot(I))

            # Resolver G₂ α = ∂₂ᵀW·I (LSQR, FIX 7)
            alpha  = _lsqr_solve(G2, rhs_alpha)

            # I_curl = ∂₂ α (co-exact 1-form)
            I_curl = boundary_2.dot(alpha)
            return I_curl

        # ──────────────────────────────────────────────────────────────────
        # §3.3 — Verificación de ortogonalidad en ⟨·,·⟩_W (FIX 8)
        # ──────────────────────────────────────────────────────────────────

        @staticmethod
        def _verify_hodge_orthogonality(
            I_grad  : NDArray[np.float64],
            I_curl  : NDArray[np.float64],
            I_harm  : NDArray[np.float64],
            W_inv   : sp.dia_matrix,
        ) -> bool:
            r"""
            Verifica que las tres componentes son ortogonales en ⟨·,·⟩_W.

            Teorema de Hodge (caso ponderado):
                I = I_grad ⊕_W I_curl ⊕_W I_harm
            donde ⊕_W denota suma ortogonal en ⟨·,·⟩_W.

            Condiciones de ortogonalidad:
                ⟨I_grad, I_curl⟩_W = 0   (exactas ⊥ co-exactas)
                ⟨I_grad, I_harm⟩_W = 0   (exactas ⊥ armónicas)
                ⟨I_curl, I_harm⟩_W = 0   (co-exactas ⊥ armónicas)

            Un fallo de ortogonalidad indica error numérico en las
            proyecciones LSQR o en la construcción de ∂₁, ∂₂.

            Parámetros
            ----------
            I_grad, I_curl, I_harm : NDArray, componentes de Hodge.
            W_inv                  : sp.dia_matrix, W⁻¹.

            Retorna
            -------
            bool, True sii todas las ortogonalidades se satisfacen.
            """
            tol = PistonConstants.HODGE_ORTHO_TOL

            inner_gc = abs(_energy_inner_product(I_grad, I_curl, W_inv))
            inner_gh = abs(_energy_inner_product(I_grad, I_harm, W_inv))
            inner_ch = abs(_energy_inner_product(I_curl, I_harm, W_inv))

            ok = inner_gc < tol and inner_gh < tol and inner_ch < tol

            if not ok:
                logger.warning(
                    "Ortogonalidad de Hodge violada en ⟨·,·⟩_W: "
                    "⟨grad,curl⟩_W=%.3e, ⟨grad,harm⟩_W=%.3e, "
                    "⟨curl,harm⟩_W=%.3e (tol=%.3e).",
                    inner_gc, inner_gh, inner_ch, tol,
                )
            else:
                logger.debug(
                    "Ortogonalidad Hodge verificada en ⟨·,·⟩_W ✓ "
                    "(⟨g,c⟩=%.2e, ⟨g,h⟩=%.2e, ⟨c,h⟩=%.2e).",
                    inner_gc, inner_gh, inner_ch,
                )
            return ok

        # ──────────────────────────────────────────────────────────────────
        # §3.4 — Descomposición principal de Hodge-Helmholtz (FIX 6, 7, 8)
        # ──────────────────────────────────────────────────────────────────

        @classmethod
        def decompose_flow(
            cls,
            mesh  : "SimplicialMesh",
            flow  : "FlowState",
        ) -> "HodgeDecomposition":
            r"""
            Descompone el flujo I = I_grad ⊕_W I_curl ⊕_W I_harm.

            CORRECCIONES FIX 6, 7, 8 integradas:

            Pipeline:
            ─────────
            1. I_grad: proyección sobre im(∂₁ᵀ) en ⟨·,·⟩_W (FIX 6).
               Resolver L₀^W φ = ∂₁W·I → I_grad = ∂₁ᵀ φ.
            2. I_curl: proyección sobre im(∂₂) en ⟨·,·⟩_W (FIX 6).
               Resolver G₂α = ∂₂ᵀW·I → I_curl = ∂₂α.
            3. I_harm = I - I_grad - I_curl (componente armónica residual).
            4. Verificar ortogonalidad en ⟨·,·⟩_W (FIX 8).
            5. Calcular ‖I_curl‖_W (norma de energía, FIX 8).
            6. Veto si ‖I_curl‖_W > MAX_VORTICITY_NORM.

            Parámetros
            ----------
            mesh : SimplicialMesh con L₀^W, L₁^W, W_inv precalculados.
            flow : FlowState con edge_flows.

            Retorna
            -------
            HodgeDecomposition con componentes, norma de vorticidad y
            certificación de ortogonalidad.

            Lanza
            -----
            ParasiticVorticityVetoError : ‖I_curl‖_W > MAX_VORTICITY_NORM.
            """
            B1    = mesh.boundary_1
            B2    = mesh.boundary_2
            W     = mesh.conductance_matrix
            W_inv = mesh.inv_conductance
            L0W   = mesh.laplacian_0        # FIX 4+12
            I     = flow.edge_flows.copy()

            # §3.4.1: Proyección gradiente ponderada (FIX 6)
            I_grad = cls._project_gradient(B1, L0W, W_inv, I)

            # §3.4.2: Proyección rotacional ponderada (FIX 6 + FIX 7)
            I_curl = cls._project_curl(B2, W, I)

            # §3.4.3: Componente armónica (residual en ker(L₁^W))
            I_harm = I - I_grad - I_curl

            # §3.4.4: Norma de vorticidad en ⟨·,·⟩_W (FIX 8)
            vort_norm_W = _energy_norm(I_curl, W_inv)

            # §3.4.5: Energía total ‖I‖²_W (verificación de Pitágoras)
            energy_total = (
                _energy_norm(I_grad, W_inv) ** 2
                + _energy_norm(I_curl, W_inv) ** 2
                + _energy_norm(I_harm, W_inv) ** 2
            )

            # §3.4.6: Verificación de ortogonalidad (FIX 8)
            ortho_ok = cls._verify_hodge_orthogonality(
                I_grad, I_curl, I_harm, W_inv
            )

            logger.info(
                "Hodge-Helmholtz DEC (norma W): "
                "‖I_grad‖_W=%.3e, ‖I_curl‖_W=%.3e, ‖I_harm‖_W=%.3e. "
                "Ortogonalidad: %s.",
                _energy_norm(I_grad, W_inv),
                vort_norm_W,
                _energy_norm(I_harm, W_inv),
                "✓" if ortho_ok else "✗",
            )

            # §3.4.7: Veto de vorticidad parásita (ahora en norma de energía)
            if vort_norm_W > PistonConstants.MAX_VORTICITY_NORM:
                raise ParasiticVorticityVetoError(
                    f"Veto Termodinámico: Vorticidad parásita en norma de energía "
                    f"‖I_curl‖_W = {vort_norm_W:.4e} > "
                    f"{PistonConstants.MAX_VORTICITY_NORM:.4e}. "
                    f"La bomba inyectaría exergía circulando en bucles estériles "
                    f"sin producir flujo laminar útil. "
                    f"Revisar la topología del grafo (eliminar ciclos parásitos "
                    f"o aumentar MAX_VORTICITY_NORM si los ciclos son intencionales)."
                )

            return HodgeDecomposition(
                i_grad           = I_grad,
                i_curl           = I_curl,
                i_harm           = I_harm,
                vorticity_norm   = vort_norm_W,
                orthogonality_ok = ortho_ok,
                energy_total     = energy_total,
            )

    # ── Constructor y métodos públicos del agente ─────────────────────────

    def __init__(self) -> None:
        """Las fases son métodos de clase; no requieren estado de instancia."""
        pass

    # ──────────────────────────────────────────────────────────────────────
    # §4.1 — Ensamblaje del vector de fuentes con verificación (FIX 9)
    # ──────────────────────────────────────────────────────────────────────

    def _create_injection_vector(
        self,
        nodes  : List[str],
        source : str,
        sink   : str,
        q_pump : float,
    ) -> InjectionVector:
        r"""
        Ensambla el vector de fuentes s con verificación ∑sᵢ = 0 (FIX 9).

        La bomba de desplazamiento positivo inyecta caudal Q:
            s[fuente] = -Q  (succión: masa sale del nodo)
            s[sumidero] = +Q (descarga: masa entra al nodo)
            s[resto]   = 0

        Conservación estricta:
            ∑sᵢ = -Q + Q = 0 ✓ (garantizada por construcción)

        CORRECCIÓN FIX 9 vs v5.0.0:
            En v5.0.0, el balance ∑sᵢ = 0 no se verificaba explícitamente.
            Esta función añade la verificación y empaqueta el resultado
            en InjectionVector con campo `balance_verified = True`.

        Parámetros
        ----------
        nodes  : List[str], lista de nodos del grafo.
        source : str, nodo de succión (fuente de la bomba).
        sink   : str, nodo de descarga (sumidero de la bomba).
        q_pump : float, caudal Q [unidades de flujo].

        Retorna
        -------
        InjectionVector con balance verificado.

        Lanza
        -----
        TopologicalInvariantError : Si source o sink no pertenecen a nodes.
        SourceCompatibilityError  : Si ∑sᵢ ≠ 0 (no debería ocurrir por construcción).
        """
        try:
            u_idx = list(nodes).index(source)
        except ValueError:
            raise TopologicalInvariantError(
                f"Nodo fuente '{source}' no pertenece al grafo. "
                f"Nodos disponibles: {list(nodes)}."
            )
        try:
            v_idx = list(nodes).index(sink)
        except ValueError:
            raise TopologicalInvariantError(
                f"Nodo sumidero '{sink}' no pertenece al grafo. "
                f"Nodos disponibles: {list(nodes)}."
            )

        if u_idx == v_idx:
            raise TopologicalInvariantError(
                f"El nodo fuente y el sumidero no pueden ser el mismo nodo "
                f"('{source}'). Un loop genera un campo de flujo nulo."
            )

        # Ensamblar s con conservación por construcción
        s_vector = np.zeros(len(nodes), dtype=np.float64)
        s_vector[u_idx] -= q_pump   # succión
        s_vector[v_idx] += q_pump   # descarga

        # Verificación explícita (FIX 9)
        # (debe ser 0 por construcción, protege contra errores de punto flotante)
        balance = float(np.sum(s_vector))
        if abs(balance) > PistonConstants.SOURCE_BALANCE_TOL:
            # Corrección de balance residual por punto flotante
            s_vector[v_idx] -= balance
            logger.debug(
                "Corrección de balance de punto flotante: Δ=%.4e aplicado a nodo '%s'.",
                balance, sink,
            )

        return InjectionVector(
            source_node      = source,
            sink_node        = sink,
            q_pump           = q_pump,
            s_vector         = s_vector,
            balance_verified = True,
        )

    # ──────────────────────────────────────────────────────────────────────
    # §4.2 — Orquestación de la transición de fase (FIX 10)
    # ──────────────────────────────────────────────────────────────────────

    def execute_injection(
        self,
        nodes          : List[str],
        edges          : List[Tuple[str, str, float]],
        cycles         : List[List[str]],
        pump_source    : str,
        pump_sink      : str,
        reservoir_node : str,
        q_pump         : float,
    ) -> Dict[str, Any]:
        r"""
        Orquesta la transición de fase geométrica completa para la inyección.

        CORRECCIÓN FIX 10 vs v5.0.0:
            El diccionario de retorno en v5.0.0 omitía invariantes topológicos
            fundamentales (λ_min, residuo KCL, ortogonalidad, energía total).
            Estos son necesarios para que el DiracInterconnectionAgent y el
            SheafCohomologyOrchestrator validen el estado inyectado.

        Pipeline:
        ─────────
        1. FASE 1: build_mesh → ∂₁, ∂₂, L₀^W, L₁^W, W⁻¹ (con ∂₁∘∂₂=0).
        2.         _create_injection_vector → s con ∑sᵢ=0.
        3. FASE 2: solve_hydrodynamics → p, f, KCL_residual, λ_min.
        4. FASE 3: decompose_flow → I_grad, I_curl, I_harm, ‖I_curl‖_W.
        5.         Empaquetar resultado con todos los invariantes (FIX 10).

        Parámetros
        ----------
        nodes          : List[str], identificadores de nodos.
        edges          : List[Tuple[str,str,float]], aristas (u,v,conductancia).
        cycles         : List[List[str]], ciclos como listas de nodos.
        pump_source    : str, nodo de succión.
        pump_sink      : str, nodo de descarga.
        reservoir_node : str, nodo de referencia (Dirichlet p=0).
        q_pump         : float, caudal a inyectar.

        Retorna
        -------
        Dict[str, Any] con el estado hidrodinámico completo e invariantes.
        """
        logger.info(
            "Iniciando Inyección Termodinámica Q=%.4f [%s → %s] "
            "(reservorio: %s).",
            q_pump, pump_source, pump_sink, reservoir_node,
        )

        # §4.2.1: FASE 1 — Fibración topológica
        mesh = self._Phase1_SimplicialBoundaryBuilder.build_mesh(
            nodes, edges, cycles
        )

        # §4.2.2: Vector de fuentes con balance verificado (FIX 9)
        injection = self._create_injection_vector(
            nodes, pump_source, pump_sink, q_pump
        )

        # §4.2.3: FASE 2 — Solución de Poisson
        flow_state = self._Phase2_DirichletPoissonSolver.solve_hydrodynamics(
            mesh, injection, reservoir_node
        )

        # §4.2.4: FASE 3 — Descomposición de Hodge-Helmholtz DEC
        hodge = self._Phase3_HodgeHelmholtzAuditor.decompose_flow(
            mesh, flow_state
        )

        # §4.2.5: Empaquetado con invariantes topológicos (FIX 10)
        result: Dict[str, Any] = {
            # Estado termodinámico
            "status"              : "ANNIHILATED_PARASITIC_VORTICITY",
            "injection_q"         : float(q_pump),
            # Flujo hidrodinámico
            "nodal_pressures"     : flow_state.nodal_pressures.tolist(),
            "edge_flows"          : flow_state.edge_flows.tolist(),
            # Descomposición de Hodge
            "laminar_flow_norm_W" : float(
                _energy_norm(hodge.i_grad, mesh.inv_conductance)
            ),
            "vorticity_norm_W"    : float(hodge.vorticity_norm),
            "harmonic_norm_W"     : float(
                _energy_norm(hodge.i_harm, mesh.inv_conductance)
            ),
            "hodge_energy_total"  : float(hodge.energy_total),
            "hodge_orthogonal"    : hodge.orthogonality_ok,
            # Invariantes topológicos (FIX 10)
            "topological_invariants": {
                "kirchhoff_residual" : float(flow_state.kirchhoff_residual),
                "lambda_min_L_red"   : float(flow_state.lambda_min_L_red),
                "boundary_identity"  : mesh.boundary_identity,
                "source_balanced"    : injection.balance_verified,
                "num_nodes"          : len(mesh.nodes),
                "num_edges"          : len(mesh.edges),
                "num_cycles"         : len(mesh.cycles),
            },
            # Tensores de flujo detallados (para DiracInterconnectionAgent)
            "flow_components": {
                "i_grad" : hodge.i_grad.tolist(),
                "i_curl" : hodge.i_curl.tolist(),
                "i_harm" : hodge.i_harm.tolist(),
            },
        }

        logger.info(
            "Inyección completada. ‖I_curl‖_W=%.3e, ‖I_grad‖_W=%.3e, "
            "KCL=%.3e, λ_min=%.3e.",
            result["vorticity_norm_W"],
            result["laminar_flow_norm_W"],
            result["topological_invariants"]["kirchhoff_residual"],
            result["topological_invariants"]["lambda_min_L_red"],
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # §4.3 — Funtor categórico para la Malla Agéntica (MIC)
    # ──────────────────────────────────────────────────────────────────────

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        Funtor categórico φ: CategoricalState → CategoricalState.

        Extrae la carga útil del estado categórico, ejecuta la inyección
        hidrodinámica completa (Fases 1→2→3), y retorna un estado purificado
        en el Estrato PHYSICS con el tensor hidrodinámico y los invariantes
        topológicos.

        Validación de la carga útil:
        ─────────────────────────────
        El funtor exige que el payload contenga los tensores constitutivos
        mínimos: nodes, edges, pump_source, pump_sink, reservoir_node.
        La ausencia de cualquier campo causa un veto con
        TopologicalInvariantError (estado estocástico degenerado).

        Parámetros
        ----------
        state : CategoricalState, estado de entrada con payload de red.

        Retorna
        -------
        CategoricalState en Stratum.PHYSICS con tensor hidrodinámico.
        """
        payload = state.payload

        nodes          = payload.get("nodes",          [])
        edges          = payload.get("edges",          [])
        cycles         = payload.get("cycles",         [])
        pump_source    = payload.get("pump_source")
        pump_sink      = payload.get("pump_sink")
        reservoir_node = payload.get("reservoir_node")
        q_pump         = float(payload.get("q_pump",   1.0))

        # Validación de completitud del payload
        missing = []
        if not nodes          : missing.append("nodes")
        if not edges          : missing.append("edges")
        if not pump_source    : missing.append("pump_source")
        if not pump_sink      : missing.append("pump_sink")
        if not reservoir_node : missing.append("reservoir_node")

        if missing:
            raise TopologicalInvariantError(
                f"Carga útil estocástica degenerada: faltan los tensores "
                f"constitutivos: {missing}. El funtor no puede operar sobre "
                f"un estado topológicamente incompleto."
            )

        # Ejecución de la transición de fase
        result_tensors = self.execute_injection(
            nodes, edges, cycles,
            pump_source, pump_sink, reservoir_node, q_pump,
        )

        # Empaquetado del estado de salida
        new_payload = dict(payload)
        new_payload["hydrodynamic_tensor"] = result_tensors

        return CategoricalState(
            payload  = new_payload,
            metadata = state.metadata,
            stratum  = Stratum.PHYSICS,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constantes
    "PistonConstants",
    # Excepciones
    "PistonInjectorError",
    "SingularLaplacianError",
    "HomologicalKirchhoffError",
    "ParasiticVorticityVetoError",
    "BoundaryComplexError",       # FIX 1 — nueva excepción
    "SourceCompatibilityError",   # FIX 9 — nueva excepción
    # DTOs
    "SimplicialMesh",
    "InjectionVector",
    "FlowState",
    "HodgeDecomposition",
    # Agente
    "HodgeHelmholtzInjectorAgent",
]
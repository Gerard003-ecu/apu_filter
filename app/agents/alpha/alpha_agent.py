# -*- coding: utf-8 -*-
r"""
+==============================================================================+
| Módulo  : Alpha Boundary Agent (Orquestador de Haces Celulares)              |
| Ruta    : app/agents/alpha/alpha_agent.py                                    |
| Versión : 4.0.0-Rigorous-Sheaf-Cohomology-Consensus                          |
+==============================================================================+

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA DE HACES:
Este módulo transmuta el Estrato alpha en un Orquestador de Haces Celulares. Ensambla
la cofrontera global y somete a los agentes a test topológicos rigurosos.

LAPLACIANO DEL HAZ (CELLULAR SHEAF):
\[ L_F = \delta^\top \delta \succeq 0 \]

SOLUBILIDAD DE FREDHOLM:
\[ \langle s_{val}, \psi_{ker} \rangle = 0 \quad \forall \psi_{ker} \in \ker(L_F) \]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    Protocol,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

#    Dependencias arquitect nicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
except ImportError:
    # Stubs rigurosos para ejecuci n aislada y prueba unitaria anal tica
    class TopologicalInvariantError(Exception):  # type: ignore[no-redef]
        """Excepci n base del ecosistema MIC para violaciones topol gicas."""
        pass

    class Morphism:  # type: ignore[no-redef]
        """Protocolo morfismo del ecosistema MIC."""
        pass

    class CategoricalState:  # type: ignore[no-redef]
        """Estado categ rico del ecosistema MIC."""
        payload: Dict[str, Any]
        metadata: Dict[str, Any]
        stratum: Any

        def __init__(
            self,
            payload: Dict[str, Any],
            metadata: Optional[Dict[str, Any]] = None,
            stratum: Any = None,
        ) -> None:
            self.payload  = payload
            self.metadata = metadata or {}
            self.stratum  = stratum

    class Stratum(Enum):  # type: ignore[no-redef]
        ALPHA = auto()


logger = logging.getLogger("MIC.Alpha.BoundaryAgent")


#
#  A. CONSTANTES F SICAS Y NUM RICAS DEL ESTRATO ALFA
#

class AlphaConstants:
    r"""
    Constantes topológicas y tolerancias numéricas para el Estrato α.

    Fundamentos numéricos:
    ─────────────────────
    RANK_TOLERANCE : umbral absoluto inferior para valores singulares.
        Elegido como 1e-10 (dos órdenes sobre ε_mach ≈ 2.2e-16), suficiente para
        distinguir rango numérico de artefactos de punto flotante en matrices de
        tamaño práctico (|V|, |E| ≤ 10^4).

    MIN_FIEDLER_VALUE : umbral isoperimétrico de Cheeger.
        Valor de 0.05 corresponde aproximadamente a la constante de Cheeger h(G)
        ≥ λ₂/2 para grafos con diámetro ≤ 20 y grado máximo ≤ 10.

    MACHINE_EPSILON : épsilon de máquina IEEE-754 float64 ≈ 2.22e-16.
    """
    RANK_TOLERANCE   : float = 1e-10
    MIN_FIEDLER_VALUE: float = 0.05
    MACHINE_EPSILON  : float = float(np.finfo(np.float64).eps)

    @staticmethod
    def svd_rank_threshold(matrix: NDArray[np.float64]) -> float:
        r"""
        Calcula el umbral robusto para el corte de valores singulares.

        Fórmula:
            τ = max(|V|, |E|) · ε_mach · σ_max    si σ_max > 0
            τ = RANK_TOLERANCE                      si σ_max = 0 (matriz nula)

        Parámetros
        ----------
        matrix : NDArray[np.float64]
            Matriz cuyo rango se va a estimar.

        Retorna
        -------
        float
            Umbral τ > 0 para el corte de valores singulares.
        """
        if matrix.size == 0:
            return AlphaConstants.RANK_TOLERANCE
        m, n = matrix.shape if matrix.ndim == 2 else (matrix.size, 1)
        # Obtener  _max sin calcular U y V (eficiente)
        sigma_max = float(np.linalg.norm(matrix, ord=2))  # =  _1
        if sigma_max < AlphaConstants.RANK_TOLERANCE:
            return AlphaConstants.RANK_TOLERANCE
        tol_rel = max(m, n) * AlphaConstants.MACHINE_EPSILON * sigma_max
        return max(tol_rel, AlphaConstants.RANK_TOLERANCE)

    @staticmethod
    def laplacian_zero_tolerance(n: int, frobenius_norm: float) -> float:
        r"""
        Calcula la tolerancia de cero para autovalores del Laplaciano.

        Fórmula:
            zero_tol = n · ε_mach · ‖L₀‖_F    si ‖L₀‖_F > 0
            zero_tol = n · ε_mach               si ‖L₀‖_F = 0

        Justificación:
        La norma de Frobenius ‖L₀‖_F es una cota superior de los autovalores
        individuales (‖A‖_F ≥ |λ_i|), por lo que n·ε_mach·‖L₀‖_F domina el
        error de redondeo acumulado en la diagonalización de L₀.

        Parámetros
        ----------
        n             : int, dimensión de L₀.
        frobenius_norm: float, ‖L₀‖_F.

        Retorna
        -------
        float, tolerancia de cero > 0.
        """
        base = frobenius_norm if frobenius_norm > 0.0 else 1.0
        return n * AlphaConstants.MACHINE_EPSILON * base


#
#  B. JERARQU A DE EXCEPCIONES TOPOL GICAS
#

class AlphaBoundaryError(TopologicalInvariantError):
    r"""
    Excepción categórica base para violaciones en el Estrato α.
    Todas las excepciones del agente son subclases de esta.
    """
    pass


class PreconditionError(AlphaBoundaryError):
    r"""
    Detonada en la Fase 1 cuando los datos de entrada violan las precondiciones
    estructurales del 1-complejo simplicial ponderado:

    Condiciones:
        (a) nodes no vacío, sin duplicados, elementos de tipo str.
        (b) Cada flow es (str, str, float) con float > 0.
        (c) Sin auto-bucles: u ≠ v.
        (d) Sin multi-aristas: (u,v) aparece a lo sumo una vez.
        (e) Ambos extremos de cada arista pertenecen a nodes.
    """
    pass


class ToxicCycleVetoError(AlphaBoundaryError):
    r"""
    Detonada cuando dim(ker(∂₁)) > 0, es decir, β₁ > 0.

    Fundamento: β₁ > 0 implica la existencia de ciclos independientes en el
    1-complejo K. En el contexto del BMC, esto evidencia canibalización
    cruzada y dependencias circulares que no pueden resolverse por operaciones
    de frontera (son invariantes homológicos, no artefactos algorítmicos).

    El núcleo de ∂₁ (espacio de 1-ciclos) tiene dimensión β₁ > 0.
    """
    pass


class EulerPoincareDegeneracyError(AlphaBoundaryError):
    r"""
    Detonada cuando χ(K) = |V| - |E| ≤ 0.

    Fundamento: Para un grafo conexo, χ = 1 - β₁. La condición χ ≤ 0 implica
    β₁ ≥ 1 (que ya causa ToxicCycleVetoError) o que el modelo tiene más aristas
    que vértices, lo que indica hiper-dependencia estructural.

    Para grafos con múltiples componentes (β₀ > 1): χ = β₀ - β₁ ≤ 0 implica
    que los ciclos son al menos tan numerosos como las componentes.
    """
    pass


class SpectralFragilityError(AlphaBoundaryError):
    r"""
    Detonada cuando λ₂(L₀) < MIN_FIEDLER_VALUE.

    Fundamento: La desigualdad de Cheeger para grafos establece:
        h(G) ≥ λ₂/2  (constante de Cheeger ≥ Valor de Fiedler / 2)
    Un λ₂ pequeño implica que existe un corte de aristas con capacidad pequeña
    relativa al volumen de los dos lados: un SPOF (Single Point of Failure).
    """
    pass


#
#  C. ESTRUCTURAS DE DATOS INMUTABLES (DTOs TOPOL GICOS)
#

@dataclass(frozen=True, slots=True)
class SimplicialComplexData:
    r"""
    Artefacto de la Fase 1: El 1-esqueleto ponderado del BMC.

    Campos
    ------
    vertices         : Tupla de nombres de vértices (bloques del BMC).
    edges            : Tupla de aristas (origen, destino, peso_positivo).
    boundary_operator: ∂₁ ∈ ℝ^{|V|×|E|}, con columna k escalada por √w_k.
    n_vertices        : |V| = len(vertices).
    n_edges           : |E| = len(edges).

    Invariantes verificados en __post_init__:
        - boundary_operator.shape == (n_vertices, n_edges).
        - Todos los pesos en edges son > 0.
        - Sin NaN/Inf en boundary_operator.
    """
    vertices         : Tuple[str, ...]
    edges            : Tuple[Tuple[str, str, float], ...]
    boundary_operator: NDArray[np.float64]
    n_vertices       : int
    n_edges          : int

    def __post_init__(self) -> None:
        """Verifica invariantes de la estructura simplicial."""
        expected_shape = (self.n_vertices, self.n_edges)
        if self.boundary_operator.shape != expected_shape:
            raise PreconditionError(
                f"  .shape={self.boundary_operator.shape}   "
                f"({self.n_vertices},{self.n_edges})."
            )
        if not np.all(np.isfinite(self.boundary_operator)):
            raise PreconditionError(
                "   contiene valores no finitos (NaN/Inf)."
            )
        for (u, v, w) in self.edges:
            if w <= 0.0:
                raise PreconditionError(
                    f"Arista ({u},{v}) tiene peso w={w} <= 0. "
                    f"Todas las aristas deben tener peso estrictamente positivo."
                )


@dataclass(frozen=True, slots=True)
class HomologicalInvariants:
    r"""
    Artefacto de la Fase 2: Invariantes algebraicos del 1-complejo K.

    Campos
    ------
    rank                 : rank(∂₁) calculado por SVD.
    beta_0               : β₀ = |V| - rank(∂₁) (componentes conexas).
    beta_1               : β₁ = |E| - rank(∂₁) (ciclos independientes).
    euler_characteristic : χ(K) = β₀ - β₁ = |V| - |E|.
    svd_singular_values  : Valores singulares de ∂₁ en orden descendente.
    rank_threshold       : Umbral τ usado para el corte SVD.

    Invariantes:
        - euler_characteristic == beta_0 - beta_1 == n_vertices - n_edges.
        - beta_0 ≥ 1 (al menos un componente).
        - beta_1 ≥ 0.
    """
    rank                : int
    beta_0              : int
    beta_1              : int
    euler_characteristic: int
    svd_singular_values : NDArray[np.float64]
    rank_threshold      : float

    def __post_init__(self) -> None:
        """Verifica la identidad de Euler-Poincar  como poscondici n."""
        computed_euler = self.beta_0 - self.beta_1
        if computed_euler != self.euler_characteristic:
            raise TopologicalInvariantError(
                f"Violaci n de Euler-Poincar :   -  ={computed_euler}   "
                f" ={self.euler_characteristic}."
            )


@dataclass(frozen=True, slots=True)
class SpectralFiedlerData:
    r"""
    Artefacto de la Fase 3: Espectro del Laplaciano Combinatorio Ponderado.

    Campos
    ------
    laplacian       : L₀ = ∂₁∂₁ᵀ ∈ ℝ^{|V|×|V|}.
    eigenvalues     : Autovalores de L₀ en orden ascendente.
    fiedler_value   : λ₂ (menor autovalor estrictamente positivo), o 0.0.
    zero_tolerance  : Umbral usado para distinguir λ=0 de λ>0.
    is_connected    : True si y solo si λ₂ > zero_tolerance (grafo conexo).
    spectral_gap    : λ_max - λ₂ si fiedler_value > 0, else 0.0.

    Invariantes:
        - eigenvalues[0] < zero_tolerance (primer autovalor ≈ 0).
        - fiedler_value ≥ 0.
        - is_connected iff fiedler_value > zero_tolerance.
    """
    laplacian      : NDArray[np.float64]
    eigenvalues    : NDArray[np.float64]
    fiedler_value  : float
    zero_tolerance : float
    is_connected   : bool
    spectral_gap   : float

    def __post_init__(self) -> None:
        """Verifica que el primer autovalor sea ~= 0 (L  es PSD)."""
        if self.eigenvalues.size > 0:
            lambda_min = float(self.eigenvalues[0])
            if lambda_min < -self.zero_tolerance:
                raise TopologicalInvariantError(
                    f"L  no es semidefinida positiva:  _min={lambda_min:.4e} < 0. "
                    f"Error num rico en la construcci n del Laplaciano."
                )


@dataclass(frozen=True, slots=True)
class AlphaBoundaryVerdict:
    r"""
    Veredicto inmutable emitido por el AlphaBoundaryAgent.

    Campos
    ------
    is_viable    : True si y solo si los 3 vetos no se dispararon.
    homology     : HomologicalInvariants (siempre presente si Fase 1 y 2 completan).
    spectral     : SpectralFiedlerData (presente si Fase 3 completa, None si veto en Fase 2).
    veto_reason  : Mensaje del veto si is_viable=False, None si is_viable=True.
    veto_class   : Tipo de excepción del veto, None si is_viable=True.
    """
    is_viable   : bool
    homology    : HomologicalInvariants
    spectral    : Optional[SpectralFiedlerData]
    veto_reason : Optional[str]
    veto_class  : Optional[Type[AlphaBoundaryError]]


#
#  D. PROTOCOLOS FUNTORIALES DE LAS 3 FASES
#

@runtime_checkable
class CanvasFibratorPort(Protocol):
    """Puerto de entrada de la Fase 1."""
    def project_canvas_to_simplicial_complex(
        self, nodes: List[str], flows: List[Tuple[str, str, float]]
    ) -> SimplicialComplexData: ...


@runtime_checkable
class HomologicalAuditorPort(Protocol):
    """Puerto de entrada de la Fase 2."""
    def audit_betti_invariants(
        self, complex_data: SimplicialComplexData
    ) -> HomologicalInvariants: ...


@runtime_checkable
class SpectralAuditorPort(Protocol):
    """Puerto de entrada de la Fase 3."""
    def audit_fiedler_connectivity(
        self, complex_data: SimplicialComplexData
    ) -> SpectralFiedlerData: ...


#
#
#
#    ORQUESTADOR SUPREMO: ALPHA BOUNDARY AGENT
#    Mapeo Endofuntorial que impone la condici n de frontera macrosc pica.
#    Las tres fases anidadas est n encadenadas por sus DTOs immutables.
#
#
#

class AlphaBoundaryAgent(Morphism):
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    AlphaBoundaryAgent (EulerPoincareHolonomyAgent) v3.0.0
    ═══════════════════════════════════════════════════════════════════════════

    Orquesta la síntesis de las tres fases categóricas anidadas. Aplica los
    VETOS AXIOMÁTICOS si los invariantes topológicos y espectrales del modelo
    de negocio indican una degeneración estructural.

    Estructura de fases anidadas:
    ─────────────────────────────
    La salida del último método de la Fase 1 (SimplicialComplexData) es la
    entrada del primer método de la Fase 2 (_compute_numerical_rank recibe
    complex_data.boundary_operator). La salida de la Fase 2 (HomologicalInvariants)
    coexiste con el SimplicialComplexData para alimentar la Fase 3 (_build_laplacian
    recibe complex_data.boundary_operator).

    Los vetos axiomáticos se evalúan en secuencia estricta:
        Veto 1 (β₁>0)    → corta antes de la Fase 3 si se activa
        Veto 2 (χ≤0)     → corta antes de la Fase 3 si se activa
        Veto 3 (λ₂<ε)    → corta después de la Fase 3

    Hereda de Morphism para integrarse en la Malla Agéntica MIC.
    """

    #
    #
    #
    #    FASE 1   FIBRADOR DEL LIENZO SIMPLICIAL PONDERADO
    #
    #    Contrato formal:
    #      Entrada  : nodes: List[str], flows: List[Tuple[str,str,float]]
    #      Salida   : SimplicialComplexData (   ponderado, sin NaN/Inf)
    #
    #    Garant as:
    #      G1. Sin duplicados en nodes.
    #      G2. Sin auto-bucles (u=v).
    #      G3. Sin multi-aristas.
    #      G4. Pesos w > 0 estrictamente.
    #      G5. Ambos extremos de cada arista   nodes.
    #      G6.   [i,k] =   w_k (Jacobiano ponderado).
    #
    #
    #

    class _Phase1_CanvasSimplicialFibrator:
        r"""
        Ejecuta el homeomorfismo entre los bloques semánticos del Business Model
        Canvas y el 1-complejo simplicial ponderado K = (V, E, w).

        Genera la matriz de incidencia orientada ponderada ∂₁ ∈ ℝ^{|V|×|E|}
        con el Jacobiano de peso √w_k por arista.
        """

        #
        #  1.1   Validaci n de precondiciones estructurales
        #

        @staticmethod
        def _validate_canvas_input(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> None:
            r"""
            Verifica las precondiciones estructurales del 1-complejo simplicial.

            Condiciones verificadas:
            ────────────────────────
            (C1) nodes no vacío: |V| ≥ 1.
            (C2) Elementos de nodes son str: V ⊂ Str.
            (C3) Sin duplicados en nodes: |{nodes}| = |nodes|.
            (C4) flows es una lista de tuplas (str, str, float).
            (C5) Sin auto-bucles: u ≠ v para todo (u,v,w).
            (C6) Sin multi-aristas: el par (u,v) aparece como mucho una vez.
                 Se considera {u,v} como conjunto (aristas no orientadas) para
                 detectar también (v,u) como multi-arista de (u,v).
            (C7) Pesos estrictamente positivos: w > 0.
            (C8) Ambos extremos pertenecen a nodes: u,v ∈ V.

            Parámetros
            ----------
            nodes : List[str], lista de identificadores de vértices.
            flows : List[Tuple[str,str,float]], lista de aristas (u,v,w).

            Lanza
            -----
            PreconditionError : Si alguna condición es violada.
            """
            # (C1) No vac o
            if not nodes:
                raise PreconditionError(
                    "El colector del BMC no puede ser el conjunto vac o  . "
                    "Se requiere al menos un v rtice (bloque de valor)."
                )

            # (C2) Tipos de v rtices
            for i, node in enumerate(nodes):
                if not isinstance(node, str):
                    raise PreconditionError(
                        f"nodes[{i}]={node!r} debe ser de tipo str, "
                        f"recibido {type(node).__name__}."
                    )

            # (C3) Sin duplicados en nodes
            if len(set(nodes)) != len(nodes):
                duplicates = [n for n in nodes if nodes.count(n) > 1]
                unique_dups = list(dict.fromkeys(duplicates))
                raise PreconditionError(
                    f"nodes contiene v rtices duplicados: {unique_dups}. "
                    f"Cada bloque del BMC debe aparecer exactamente una vez."
                )

            # Validaci n de cada arista
            node_set: FrozenSet[str] = frozenset(nodes)
            seen_edges: Set[FrozenSet[str]] = set()  # para detectar multi-aristas

            for k, flow in enumerate(flows):
                # (C4) Tipo de cada flujo
                if (not isinstance(flow, tuple)
                        or len(flow) != 3
                        or not isinstance(flow[0], str)
                        or not isinstance(flow[1], str)
                        or not isinstance(flow[2], (int, float))):
                    raise PreconditionError(
                        f"flows[{k}]={flow!r} debe ser Tuple[str, str, float]. "
                        f"Recibido: {type(flow)}."
                    )
                u, v, w = flow[0], flow[1], float(flow[2])

                # (C5) Sin auto-bucles
                if u == v:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}) es un auto-bucle (u=v). "
                        f"Los 1-complejos simpliciales no admiten auto-bucles: "
                        f"   producir a una columna nula que corrompe el rango."
                    )

                # (C6) Sin multi-aristas (considerando tambi n (v,u) como duplicado)
                edge_key: FrozenSet[str] = frozenset({u, v})
                if edge_key in seen_edges:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}) es una multi-arista: "
                        f"la arista ({u},{v}) o ({v},{u}) ya existe. "
                        f"Un 1-complejo simplicial no admite multi-aristas."
                    )
                seen_edges.add(edge_key)

                # (C7) Pesos estrictamente positivos
                if not np.isfinite(w):
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}): el peso no es finito."
                    )
                if w <= 0.0:
                    raise PreconditionError(
                        f"flows[{k}]=({u},{v},{w}): el peso w={w} <= 0. "
                        f"Aristas sin flujo no son aristas topol gicas reales. "
                        f"Use w > 0 o elimine la arista del modelo."
                    )

                # (C8) Extremos en nodes
                if u not in node_set:
                    raise PreconditionError(
                        f"flows[{k}]: el nodo origen '{u}' no pertenece a nodes. "
                        f"Todos los extremos de aristas deben ser v rtices declarados."
                    )
                if v not in node_set:
                    raise PreconditionError(
                        f"flows[{k}]: el nodo destino '{v}' no pertenece a nodes. "
                        f"Todos los extremos de aristas deben ser v rtices declarados."
                    )

        #
        #  1.2   Construcci n del operador frontera ponderado
        #

        @staticmethod
        def _build_boundary_operator(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> NDArray[np.float64]:
            r"""
            Construye la matriz de incidencia orientada ponderada
            ∂₁ ∈ ℝ^{|V|×|E|}.

            Definición formal:
            ──────────────────
            Para la k-ésima arista e_k = (v_i, v_j, w_k) con w_k > 0:
                ∂₁[i, k] = -√w_k   (nodo origen: la arista "sale" de v_i)
                ∂₁[j, k] = +√w_k   (nodo destino: la arista "llega" a v_j)
                ∂₁[m, k] = 0        para todo m ∉ {i, j}

            Justificación del escalamiento √w_k:
            ─────────────────────────────────────
            El Laplaciano L₀ = ∂₁∂₁ᵀ satisface:
                L₀[i,j] = -w_{ij}      para i ≠ j (si existe arista (i,j))
                L₀[i,i] = Σ_j w_{ij}   (suma de pesos incidentes)

            Esto es exactamente el Laplaciano combinatorio ponderado estándar,
            que es la definición correcta para grafos con pesos en aristas.

            La prueba: si e_k conecta v_i y v_j con peso w_k:
                (∂₁∂₁ᵀ)[i,i] += (-√w_k)² = w_k ✓
                (∂₁∂₁ᵀ)[j,j] += (+√w_k)² = w_k ✓
                (∂₁∂₁ᵀ)[i,j] += (-√w_k)(+√w_k) = -w_k ✓

            Parámetros
            ----------
            nodes : List[str], vértices (ya validados).
            flows : List[Tuple[str,str,float]], aristas (ya validadas).

            Retorna
            -------
            NDArray[np.float64]
                Matriz ∂₁ de shape (|V|, |E|), sin NaN/Inf.
            """
            n_v = len(nodes)
            n_e = len(flows)
            node_idx = {node: idx for idx, node in enumerate(nodes)}

            B = np.zeros((n_v, n_e), dtype=np.float64)
            for k, (u, v, w) in enumerate(flows):
                sqrt_w = np.sqrt(float(w))  # w > 0 garantizado por _validate
                B[node_idx[u], k] = -sqrt_w
                B[node_idx[v], k] = +sqrt_w

            return B

        #
        #  1.3   Punto de entrada de la Fase 1 (contrato de salida   Fase 2)
        #

        @staticmethod
        def project_canvas_to_simplicial_complex(
            nodes: List[str],
            flows: List[Tuple[str, str, float]],
        ) -> SimplicialComplexData:
            r"""
            Proyecta el digrafo del BMC al 1-complejo simplicial ponderado K.

            Pipeline:
            ─────────
            1. _validate_canvas_input → precondiciones estructurales.
            2. _build_boundary_operator → ∂₁ ponderado.
            3. Construye SimplicialComplexData con invariantes verificados.

            Parámetros
            ----------
            nodes : List[str]
                Bloques del BMC (vértices del 1-complejo).
            flows : List[Tuple[str,str,float]]
                Flujos de valor entre bloques (aristas ponderadas).

            Retorna
            -------
            SimplicialComplexData
                Artefacto inmutable con ∂₁ listo para la Fase 2.

            Lanza
            -----
            PreconditionError : Si alguna precondición estructural es violada.
            """
            #  1.3.1: Validaci n de precondiciones
            AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator._validate_canvas_input(
                nodes, flows
            )

            #  1.3.2: Construcci n del operador frontera
            boundary_op = (
                AlphaBoundaryAgent._Phase1_CanvasSimplicialFibrator._build_boundary_operator(
                    nodes, flows
                )
            )

            #  1.3.3: Construir el DTO (invariantes verificados en __post_init__)
            complex_data = SimplicialComplexData(
                vertices         = tuple(nodes),
                edges            = tuple(
                    (u, v, float(w)) for (u, v, w) in flows
                ),
                boundary_operator = boundary_op,
                n_vertices        = len(nodes),
                n_edges           = len(flows),
            )

            logger.debug(
                "Fase 1 completada: K = (%d v rtices, %d aristas). "
                "    _F = %.4f.",
                complex_data.n_vertices,
                complex_data.n_edges,
                float(np.linalg.norm(boundary_op, 'fro')),
            )

            return complex_data

        #
        # FIN FASE 1
        # La salida can nica es SimplicialComplexData, que es la entrada formal
        # del primer m todo de la Fase 2: _compute_numerical_rank recibe
        # complex_data.boundary_operator.
        #


    #
    #
    #
    #    FASE 2   AUDITOR HOMOL GICO (BETTI Y EULER-POINCAR )
    #
    #    Contrato formal:
    #      Entrada  : SimplicialComplexData (   ponderado de Fase 1)
    #      Salida   : HomologicalInvariants (  ,   ,  , rank,  _values)
    #
    #    Garant as:
    #      G1. rank = card{ _i :  _i >  } con   robusto.
    #      G2.    = |V| - rank >= 1 (al menos 1 componente).
    #      G3.    = |E| - rank >= 0.
    #      G4.   =    -    = |V| - |E| (verificado cruzado).
    #      G5.  _max = 0 manejado sin IndexError.
    #
    #    La Fase 2 recibe la salida de la Fase 1 (SimplicialComplexData) y
    #    su  ltimo m todo (audit_betti_invariants) entrega HomologicalInvariants
    #    que es la referencia de los vetos 1 y 2 en el orquestador.
    #
    #
    #

    class _Phase2_HomologicalBettiAuditor:
        r"""
        Aplica el Teorema del Rango-Nulidad al operador frontera ∂₁ para
        extraer los invariantes topológicos globales β₀, β₁, χ del 1-complejo K.

        Recibe como entrada el SimplicialComplexData de la Fase 1.
        """

        #
        #  2.1   Rango algebraico robusto via SVD
        #

        @staticmethod
        def _compute_numerical_rank(
            boundary_operator: NDArray[np.float64],
        ) -> Tuple[int, NDArray[np.float64], float]:
            r"""
            Calcula el rango algebraico de ∂₁ mediante SVD con umbral robusto.

            Algoritmo:
            ──────────
            1. Si la matriz es vacía (0×0 o 0×n o n×0), retorna rank=0.
            2. Calcular los valores singulares σ₁ ≥ σ₂ ≥ ... ≥ 0 con
               scipy.linalg.svd(compute_uv=False) (más eficiente que numpy.linalg.svd).
            3. Calcular el umbral τ = AlphaConstants.svd_rank_threshold(∂₁).
               Este umbral incorpora σ_max sin fallar cuando σ_max = 0 porque
               svd_rank_threshold ya maneja este caso.
            4. rank = card{i : σ_i > τ}.
            5. Retornar (rank, σ_values, τ) para diagnóstico completo.

            Corrección vs v2.0.0:
            ─────────────────────
            - v2.0.0: singular_values[0] sin guarda → IndexError si array vacío.
              La guarda en v2.0.0 era `if matrix.size == 0` pero no cubría el
              caso de matriz no-vacía con rango 0 (todos σ_i = 0).
            - v3.0.0: AlphaConstants.svd_rank_threshold maneja σ_max = 0
              retornando RANK_TOLERANCE como umbral absoluto.

            Parámetros
            ----------
            boundary_operator : NDArray[np.float64], ∂₁ de shape (|V|, |E|).

            Retorna
            -------
            Tuple[int, NDArray[np.float64], float]
                (rank, singular_values, threshold)
            """
            if boundary_operator.size == 0:
                return 0, np.array([], dtype=np.float64), AlphaConstants.RANK_TOLERANCE

            # scipy.linalg.svd con compute_uv=False: solo los valores singulares
            # en orden descendente    >=    >= ... >= 0
            singular_values: NDArray[np.float64] = la.svd(
                boundary_operator, compute_uv=False
            )

            # Umbral robusto (maneja  _max = 0 internamente)
            threshold = AlphaConstants.svd_rank_threshold(boundary_operator)
            rank = int(np.sum(singular_values > threshold))

            logger.debug(
                "SVD:  _max=%.4e,  _min=%.4e, threshold=%.4e, rank=%d.",
                float(singular_values[0]) if singular_values.size > 0 else 0.0,
                float(singular_values[-1]) if singular_values.size > 0 else 0.0,
                threshold, rank,
            )

            return rank, singular_values, threshold

        #
        #  2.2   C lculo de n meros de Betti con verificaci n cruzada
        #

        @staticmethod
        def _compute_betti_numbers(
            n_vertices: int,
            n_edges   : int,
            rank      : int,
        ) -> Tuple[int, int, int]:
            r"""
            Calcula β₀, β₁ y χ con verificación cruzada de la identidad de
            Euler-Poincaré.

            Fórmulas (Teorema de Rango-Nulidad para 1-complejos):
            ───────────────────────────────────────────────────────
            β₀ = dim(H₀(K;ℝ)) = dim(coker(∂₁ᵀ)) = |V| - rank(∂₁)
            β₁ = dim(H₁(K;ℝ)) = dim(ker(∂₁))    = |E| - rank(∂₁)
            χ  = β₀ - β₁                          = |V| - |E|

            La identidad χ = |V| - |E| es independiente del rango (tautológica
            para 1-complejos), lo que permite una verificación cruzada:

                χ_from_betti = β₀ - β₁ = (|V|-r) - (|E|-r) = |V| - |E|
                χ_direct     = |V| - |E|
                assert χ_from_betti == χ_direct

            Esta verificación detecta errores de implementación (p.ej., si β₀
            o β₁ se calculan con fórmulas incorrectas).

            Parámetros
            ----------
            n_vertices : int, |V|.
            n_edges    : int, |E|.
            rank       : int, rank(∂₁).

            Retorna
            -------
            Tuple[int, int, int]
                (beta_0, beta_1, euler_characteristic)

            Lanza
            -----
            TopologicalInvariantError : Si la verificación cruzada falla.
            """
            beta_0 = n_vertices - rank
            beta_1 = n_edges    - rank
            euler_from_betti  = beta_0 - beta_1
            euler_direct      = n_vertices - n_edges

            if euler_from_betti != euler_direct:
                raise TopologicalInvariantError(
                    f"Violaci n de identidad de Euler-Poincar : "
                    f"  -  ={euler_from_betti}   |V|-|E|={euler_direct}. "
                    f"Error de implementaci n en el c lculo de Betti."
                )
            return beta_0, beta_1, euler_direct

        #
        #  2.3   Verificaci n de consistencia de Euler-Poincar
        #

        @staticmethod
        def _verify_euler_identity(
            homology: HomologicalInvariants,
            n_vertices: int,
            n_edges   : int,
        ) -> None:
            r"""
            Verificación final de la identidad de Euler-Poincaré:
                χ(K) = β₀ - β₁ = |V| - |E|

            Esta verificación es redundante con _compute_betti_numbers pero
            actúa como segunda línea de defensa post-construcción del DTO.

            Parámetros
            ----------
            homology   : HomologicalInvariants, resultado de audit_betti_invariants.
            n_vertices : int, |V|.
            n_edges    : int, |E|.

            Lanza
            -----
            TopologicalInvariantError : Si la verificación falla.
            """
            expected_chi = n_vertices - n_edges
            if homology.euler_characteristic != expected_chi:
                raise TopologicalInvariantError(
                    f"Post-verificaci n Euler-Poincar  fallida: "
                    f" ={homology.euler_characteristic}   |V|-|E|={expected_chi}."
                )

        #
        #  2.4   Punto de entrada de la Fase 2 (salida   Fase 3)
        #

        @staticmethod
        def audit_betti_invariants(
            complex_data: SimplicialComplexData,
        ) -> HomologicalInvariants:
            r"""
            Computa β₀, β₁, χ y emite los invariantes homológicos del 1-complejo K.

            Pipeline:
            ─────────
            1. _compute_numerical_rank(∂₁) → (rank, σ_values, threshold).
            2. _compute_betti_numbers(|V|, |E|, rank) → (β₀, β₁, χ).
            3. Construir HomologicalInvariants (invariantes verificados en __post_init__).
            4. _verify_euler_identity (segunda línea de defensa).

            Parámetros
            ----------
            complex_data : SimplicialComplexData, artefacto de la Fase 1.

            Retorna
            -------
            HomologicalInvariants
                Artefacto inmutable listo para los vetos y la Fase 3.

            Lanza
            -----
            TopologicalInvariantError : Si la identidad de Euler-Poincaré falla.
            """
            rank, sigma_values, threshold = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_numerical_rank(complex_data.boundary_operator)
            )

            beta_0, beta_1, euler_chi = (
                AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor
                ._compute_betti_numbers(
                    complex_data.n_vertices,
                    complex_data.n_edges,
                    rank,
                )
            )

            homology = HomologicalInvariants(
                rank                = rank,
                beta_0              = beta_0,
                beta_1              = beta_1,
                euler_characteristic= euler_chi,
                svd_singular_values = sigma_values,
                rank_threshold      = threshold,
            )

            # Segunda l nea de defensa post-construcci n
            AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor._verify_euler_identity(
                homology, complex_data.n_vertices, complex_data.n_edges
            )

            logger.info(
                "Homolog a computada: |V|=%d, |E|=%d, rank=%d   "
                "  =%d,   =%d,  =%d.",
                complex_data.n_vertices, complex_data.n_edges, rank,
                beta_0, beta_1, euler_chi,
            )

            return homology

        #
        # FIN FASE 2
        # La salida can nica es HomologicalInvariants. Los vetos 1 (  >0) y 2
        # ( <=0) se eval an en el orquestador antes de invocar la Fase 3.
        # La Fase 3 recibe el mismo SimplicialComplexData de la Fase 1
        # (complex_data.boundary_operator   _build_laplacian).
        #


    #
    #
    #
    #    FASE 3   AUDITOR ESPECTRAL (LAPLACIANO Y VALOR DE FIEDLER)
    #
    #    Contrato formal:
    #      Entrada  : SimplicialComplexData (   ponderado de Fase 1)
    #      Salida   : SpectralFiedlerData (L , eigenvalues,   , conectividad)
    #
    #    Garant as:
    #      G1. L  =       es PSD (verificado en SpectralFiedlerData).
    #      G2. eigenvalues en orden ascendente (eigvalsh).
    #      G3. zero_tolerance = n  _mach  L  _F (robusto ante L =0).
    #      G4. fiedler_value = 0.0 si |V|=1 (sin    posible).
    #      G5. is_connected = (fiedler_value > zero_tolerance).
    #      G6. spectral_gap =  _max -    si conectado, 0.0 si no.
    #
    #    La Fase 3 recibe SimplicialComplexData (de Fase 1) directamente.
    #    Esto garantiza la continuidad de la cadena: Fase 1   Fase 2   Fase 3.
    #
    #
    #

    class _Phase3_SpectralFiedlerAuditor:
        r"""
        Construye el Laplaciano Combinatorio Ponderado L₀ = ∂₁∂₁ᵀ y
        determina la constante de conectividad algebraica λ₂ (Valor de Fiedler).

        Recibe como entrada el mismo SimplicialComplexData de la Fase 1.
        """

        #
        #  3.1   Construcci n y verificaci n del Laplaciano
        #

        @staticmethod
        def _build_laplacian(
            boundary_operator: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            r"""
            Construye el Laplaciano Combinatorio Ponderado L₀ = ∂₁∂₁ᵀ.

            Propiedades algebraicas garantizadas:
            ──────────────────────────────────────
            (P1) Simetría: L₀ = L₀ᵀ.
                 Prueba: (∂₁∂₁ᵀ)ᵀ = (∂₁ᵀ)ᵀ∂₁ᵀ = ∂₁∂₁ᵀ = L₀.

            (P2) PSD: xᵀL₀x = xᵀ∂₁∂₁ᵀx = ‖∂₁ᵀx‖² ≥ 0 ∀x.

            (P3) Entradas: L₀[i,j] = -w_{ij} (i≠j), L₀[i,i] = Σ_j w_{ij}.
                 Prueba para L₀[i,j] con i≠j y arista k=(i,j,w_k):
                    L₀[i,j] = Σ_k ∂₁[i,k]·∂₁[j,k] = (-√w_k)(+√w_k) = -w_k ✓

            (P4) Simetría forzada: L₀ ← (L₀ + L₀ᵀ)/2 para eliminar asimetría
                 numérica de punto flotante. Máxima asimetría esperada: O(ε_mach·‖∂₁‖²).

            Parámetros
            ----------
            boundary_operator : NDArray[np.float64], ∂₁ de shape (|V|, |E|).

            Retorna
            -------
            NDArray[np.float64]
                L₀ simétrica de shape (|V|, |V|), PSD.
            """
            L = boundary_operator @ boundary_operator.T

            # Forzar simetr a exacta (elimina asimetr a de punto flotante O( _mach))
            L_sym = (L + L.T) * 0.5

            # Verificar que la asimetr a era peque a (diagn stico)
            asym = float(np.max(np.abs(L - L.T)))
            if asym > 1e-10 * float(np.linalg.norm(L, 'fro')):
                logger.warning(
                    "L  ten a asimetr a num rica significativa: %.2e. "
                    "Simetrizaci n aplicada.",
                    asym,
                )

            return L_sym

        #
        #  3.2   Tolerancia de cero robusta para autovalores
        #

        @staticmethod
        def _compute_zero_tolerance(laplacian: NDArray[np.float64]) -> float:
            r"""
            Calcula la tolerancia de cero para distinguir autovalores nulos
            de autovalores positivos pequeños.

            Fórmula:
                τ = n · ε_mach · ‖L₀‖_F

            donde n = |V| = L₀.shape[0] y ‖·‖_F es la norma de Frobenius.

            Justificación:
            ──────────────
            La norma de Frobenius satisface ‖A‖_F ≥ |λ_i| para todo autovalor
            λ_i de A. Por tanto n·ε_mach·‖A‖_F domina el error de redondeo
            acumulado en la diagonalización de A (n pasos de Givens, cada uno
            introduciendo un error relativo O(ε_mach)).

            Corrección vs v2.0.0:
            ─────────────────────
            v2.0.0 usaba n·ε·max(eigenvalues), que es 0 cuando L₀=0 (grafo
            sin aristas), haciendo que todos los autovalores (que son 0) pasaran
            el filtro. v3.0.0 usa ‖L₀‖_F con fallback a 1.0 si ‖L₀‖_F=0.

            Parámetros
            ----------
            laplacian : NDArray[np.float64], L₀.

            Retorna
            -------
            float, tolerancia > 0.
            """
            n = laplacian.shape[0]
            frobenius = float(np.linalg.norm(laplacian, 'fro'))
            return AlphaConstants.laplacian_zero_tolerance(n, frobenius)

        #
        #  3.3   Extracci n del Valor de Fiedler
        #

        @staticmethod
        def _extract_fiedler_value(
            eigenvalues   : NDArray[np.float64],
            zero_tolerance: float,
            n_vertices    : int,
        ) -> Tuple[float, bool, float]:
            r"""
            Extrae el Valor de Fiedler λ₂ del espectro del Laplaciano.

            Definición:
            ───────────
            λ₂ = min{λ_i : λ_i > zero_tolerance}

            Casos especiales:
            ─────────────────
            - n=1: L₀ es 1×1 con L₀[0,0]=0. No existe λ₂. Retorna (0.0, False, 0.0).
                   La ausencia de λ₂ no implica SPOF para un grafo trivial.
            - Todos los autovalores ≤ zero_tolerance: grafo completamente disconexo.
              Retorna (0.0, False, 0.0).
            - λ₂ > 0: grafo conexo. Retorna (λ₂, True, λ_max - λ₂).

            Justificación matemática de n=1:
            ─────────────────────────────────
            Para |V|=1, el 1-complejo K consiste de un único punto con posibles
            aristas (pero las aristas auto-bucle están prohibidas por Fase 1).
            Con n=1 y sin aristas: L₀ = [0], espectro = {0}. No hay λ₂.
            El Valor de Fiedler requiere al menos 2 vértices para ser definido
            (no existe multiplicidad de λ=0 que diferenciar de λ₂>0 con n=1).

            Parámetros
            ----------
            eigenvalues    : NDArray[np.float64], autovalores en orden ascendente.
            zero_tolerance : float, umbral para λ≈0.
            n_vertices     : int, |V|.

            Retorna
            -------
            Tuple[float, bool, float]
                (fiedler_value, is_connected, spectral_gap)
            """
            if n_vertices < 2:
                logger.warning(
                    "n_vertices=%d < 2: el Valor de Fiedler no est  definido "
                    "para grafos con un solo v rtice. "
                    "El veto espectral no se aplicar .",
                    n_vertices,
                )
                return 0.0, False, 0.0

            positive_mask = eigenvalues > zero_tolerance
            if not np.any(positive_mask):
                # Grafo completamente disconexo (todos   ~= 0)
                logger.warning(
                    "Todos los autovalores del Laplaciano son ~= 0. "
                    "El grafo es completamente disconexo (sin aristas activas)."
                )
                return 0.0, False, 0.0

            fiedler = float(np.min(eigenvalues[positive_mask]))
            lambda_max = float(eigenvalues[-1])
            spectral_gap = max(0.0, lambda_max - fiedler)
            is_connected = fiedler > zero_tolerance

            return fiedler, is_connected, spectral_gap

        #
        #  3.4   Punto de entrada de la Fase 3
        #

        @staticmethod
        def audit_fiedler_connectivity(
            complex_data: SimplicialComplexData,
        ) -> SpectralFiedlerData:
            r"""
            Computa el espectro de L₀ y determina la conectividad algebraica λ₂.

            Pipeline:
            ─────────
            1. _build_laplacian(∂₁) → L₀ = ∂₁∂₁ᵀ (simétrica, PSD).
            2. eigvalsh(L₀) → autovalores en orden ascendente.
            3. _compute_zero_tolerance(L₀) → τ.
            4. _extract_fiedler_value(eigenvalues, τ, |V|) → (λ₂, is_connected, gap).
            5. Construir SpectralFiedlerData (invariantes verificados en __post_init__).

            Parámetros
            ----------
            complex_data : SimplicialComplexData, artefacto de la Fase 1.

            Retorna
            -------
            SpectralFiedlerData
                Artefacto inmutable con espectro completo y Valor de Fiedler.

            Lanza
            -----
            TopologicalInvariantError : Si L₀ no es PSD (error numérico severo).
            """
            #  3.4.1: Construir Laplaciano
            L0 = AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor._build_laplacian(
                complex_data.boundary_operator
            )

            #  3.4.2: Autovalores en orden ascendente (eigvalsh garantiza esto)
            eigenvalues: NDArray[np.float64] = la.eigvalsh(L0)

            #  3.4.3: Tolerancia de cero
            zero_tol = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._compute_zero_tolerance(L0)
            )

            #  3.4.4: Valor de Fiedler
            fiedler, is_connected, spectral_gap = (
                AlphaBoundaryAgent._Phase3_SpectralFiedlerAuditor
                ._extract_fiedler_value(eigenvalues, zero_tol, complex_data.n_vertices)
            )

            #  3.4.5: Construir DTO (PSD verificado en __post_init__)
            spectral_data = SpectralFiedlerData(
                laplacian      = L0,
                eigenvalues    = eigenvalues,
                fiedler_value  = fiedler,
                zero_tolerance = zero_tol,
                is_connected   = is_connected,
                spectral_gap   = spectral_gap,
            )

            logger.info(
                "An lisis espectral:   =%.4e,   (Fiedler)=%.6f, "
                " _max=%.4f, gap=%.4f, conexo=%s.",
                float(eigenvalues[0]) if eigenvalues.size > 0 else 0.0,
                fiedler, float(eigenvalues[-1]) if eigenvalues.size > 0 else 0.0,
                spectral_gap, is_connected,
            )

            return spectral_data


    #
    #  4. CONSTRUCTOR Y M TODOS P BLICOS DEL AGENTE
    #

    def __init__(
        self,
        fiedler_threshold: Optional[float] = None,
    ) -> None:
        r"""
        Inicializa el AlphaBoundaryAgent con umbral de Fiedler configurable.

        Parámetros
        ----------
        fiedler_threshold : float, opcional
            Umbral personalizado para el veto espectral λ₂ < threshold.
            Si None, usa AlphaConstants.MIN_FIEDLER_VALUE = 0.05.
            Debe ser > 0.
        """
        if fiedler_threshold is not None:
            if not isinstance(fiedler_threshold, (int, float)):
                raise PreconditionError(
                    f"fiedler_threshold debe ser num rico, "
                    f"recibido {type(fiedler_threshold).__name__}."
                )
            if float(fiedler_threshold) <= 0.0:
                raise PreconditionError(
                    f"fiedler_threshold debe ser > 0, "
                    f"recibido {fiedler_threshold}."
                )
            self._fiedler_threshold = float(fiedler_threshold)
        else:
            self._fiedler_threshold = AlphaConstants.MIN_FIEDLER_VALUE

        logger.info(
            "AlphaBoundaryAgent v3.0.0 inicializado. "
            "Umbral de Fiedler: %.4f.",
            self._fiedler_threshold,
        )

    #
    #  4.1   Validaci n de entrada del orquestador
    #

    @staticmethod
    def _validate_orchestrator_input(
        nodes: Any,
        flows: Any,
    ) -> None:
        r"""
        Valida que nodes y flows tienen el tipo correcto antes de delegarlos
        a la Fase 1.

        Esta validación de tipo es una capa anterior a la validación estructural
        de la Fase 1 (_validate_canvas_input). Impide que listas mal tipadas
        produzcan AttributeError no descriptivos en niveles internos.

        Parámetros
        ----------
        nodes : Any, debe ser List[str].
        flows : Any, debe ser List[Tuple[str,str,float]].

        Lanza
        -----
        PreconditionError : Si nodes o flows no son listas.
        """
        if not isinstance(nodes, list):
            raise PreconditionError(
                f"nodes debe ser list, recibido {type(nodes).__name__}. "
                f"El BMC debe representarse como una lista de bloques sem nticos."
            )
        if not isinstance(flows, list):
            raise PreconditionError(
                f"flows debe ser list, recibido {type(flows).__name__}. "
                f"Los flujos de valor deben representarse como una lista de aristas."
            )

    #
    #  4.2   Orquestador principal (evaluate_business_canvas)
    #

    def evaluate_business_canvas(
        self,
        nodes: List[str],
        flows: List[Tuple[str, str, float]],
    ) -> AlphaBoundaryVerdict:
        r"""
        Ejecuta el colapso topológico determinista del Modelo de Negocio (BMC).

        Pipeline orquestado:
        ────────────────────
        1. Validación de tipos (§4.1).
        2. Fase 1: Fibración simplicial → SimplicialComplexData.
        3. Fase 2: Auditoría homológica → HomologicalInvariants.
           [VETO 1]: β₁ > 0 → ToxicCycleVetoError
           [VETO 2]: χ ≤ 0  → EulerPoincareDegeneracyError
        4. Fase 3: Auditoría espectral → SpectralFiedlerData.
           [VETO 3]: λ₂ < threshold (y |V| ≥ 2) → SpectralFragilityError
        5. Emitir AlphaBoundaryVerdict(is_viable=True, ...).

        En caso de veto: el estado parcial se registra en el log antes del raise.
        El AlphaBoundaryVerdict con is_viable=False se construye y almacena en
        self._last_verdict para diagnóstico post-fallo.

        Parámetros
        ----------
        nodes : List[str]
            Identificadores de los bloques del BMC (vértices).
        flows : List[Tuple[str,str,float]]
            Flujos de valor entre bloques (aristas ponderadas).

        Retorna
        -------
        AlphaBoundaryVerdict
            Veredicto inmutable con is_viable=True si todos los vetos pasan.

        Lanza
        -----
        PreconditionError            : Precondición estructural violada (Fase 1).
        ToxicCycleVetoError          : β₁ > 0 (Veto 1).
        EulerPoincareDegeneracyError : χ ≤ 0 (Veto 2).
        SpectralFragilityError       : λ₂ < threshold (Veto 3).
        """
        #  4.2.1: Validaci n de tipos
        self._validate_orchestrator_input(nodes, flows)

        homology: Optional[HomologicalInvariants] = None
        spectral: Optional[SpectralFiedlerData]   = None

        try:
            #    Fase 1: Fibraci n Simplicial
            complex_data: SimplicialComplexData = (
                self._Phase1_CanvasSimplicialFibrator
                .project_canvas_to_simplicial_complex(nodes, flows)
            )

            #    Fase 2: Auditor a Homol gica
            homology = (
                self._Phase2_HomologicalBettiAuditor
                .audit_betti_invariants(complex_data)
            )

            # [VETO AXIOM TICO 1]: Ciclos T xicos (   > 0)
            if homology.beta_1 > 0:
                veto_msg = (
                    f"Veto de Canibalizaci n [  ={homology.beta_1}]: "
                    f"El 1-complejo K contiene {homology.beta_1} ciclo(s) independiente(s). "
                    f"Esto evidencia dependencias circulares irreducibles en el BMC. "
                    f"Reduzca |E| a |E| < |V| para eliminar los ciclos. "
                    f"Estado: |V|={homology.beta_0+homology.rank}, "
                    f"|E|={homology.beta_1+homology.rank}, rank={homology.rank}."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable  = False,
                    homology   = homology,
                    spectral   = None,
                    veto_reason= veto_msg,
                    veto_class = ToxicCycleVetoError,
                )
                logger.critical("COLAPSO ESTRATO     %s", veto_msg)
                raise ToxicCycleVetoError(veto_msg)

            # [VETO AXIOM TICO 2]: Degeneraci n de Euler-Poincar  (  <= 0)
            if homology.euler_characteristic <= 0:
                veto_msg = (
                    f"Veto de Degeneraci n [ ={homology.euler_characteristic}]: "
                    f"La caracter stica de Euler es <= 0, indicando que el modelo "
                    f"tiene m s flujos que bloques de valor (|E|>=|V|). "
                    f"El espacio de negocio est  topol gicamente degenerado. "
                    f"|V|={complex_data.n_vertices}, |E|={complex_data.n_edges}."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable  = False,
                    homology   = homology,
                    spectral   = None,
                    veto_reason= veto_msg,
                    veto_class = EulerPoincareDegeneracyError,
                )
                logger.critical("COLAPSO ESTRATO     %s", veto_msg)
                raise EulerPoincareDegeneracyError(veto_msg)

            #    Fase 3: Auditor a Espectral
            spectral = (
                self._Phase3_SpectralFiedlerAuditor
                .audit_fiedler_connectivity(complex_data)
            )

            # [VETO AXIOM TICO 3]: Fragilidad Espectral (   < threshold)
            # Solo aplica si |V| >= 2 (para |V|=1 no existe   )
            if (complex_data.n_vertices >= 2
                    and spectral.fiedler_value < self._fiedler_threshold):
                veto_msg = (
                    f"Veto de Fragilidad Espectral "
                    f"[  ={spectral.fiedler_value:.6f} < "
                    f" ={self._fiedler_threshold}]: "
                    f"La conectividad algebraica es insuficiente. "
                    f"El grafo tiene un corte de Cheeger peque o: posible SPOF. "
                    f"Gap espectral: {spectral.spectral_gap:.4f}. "
                    f"Conexo: {spectral.is_connected}."
                )
                self._last_verdict = AlphaBoundaryVerdict(
                    is_viable  = False,
                    homology   = homology,
                    spectral   = spectral,
                    veto_reason= veto_msg,
                    veto_class = SpectralFragilityError,
                )
                logger.critical("COLAPSO ESTRATO     %s", veto_msg)
                raise SpectralFragilityError(veto_msg)

            #     xito: todas las auditor as superadas
            logger.info(
                "Business Model Canvas super  la auditor a de Holonom a "
                "de Euler-Poincar :   =%d,   =%d,  =%d,   =%.6f.",
                homology.beta_0, homology.beta_1,
                homology.euler_characteristic,
                spectral.fiedler_value if spectral else 0.0,
            )

            verdict = AlphaBoundaryVerdict(
                is_viable  = True,
                homology   = homology,
                spectral   = spectral,
                veto_reason= None,
                veto_class = None,
            )
            self._last_verdict: AlphaBoundaryVerdict = verdict
            return verdict

        except AlphaBoundaryError:
            # Las excepciones de veto ya fueron loggeadas y el _last_verdict
            # ya fue asignado antes del raise. Re-propagamos sin modificar.
            raise

    #
    #  4.3   Protocolo Morphism (integraci n en la Malla Ag ntica MIC)
    #

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        Implementa el protocolo Morphism de la Matriz de Interacción Central (MIC).

        Intercepta el CategoricalState, extrae la definición del BMC del payload,
        aplica el Difeomorfismo de Holonomía y retorna un nuevo estado con el
        AlphaBoundaryVerdict inyectado.

        Parámetros
        ----------
        state : CategoricalState
            Debe contener en state.payload:
                - "bmc_nodes": List[str]
                - "bmc_flows": List[Tuple[str,str,float]]

        Retorna
        -------
        CategoricalState
            Nuevo estado con state.payload["alpha_verdict"] = AlphaBoundaryVerdict.

        Lanza
        -----
        PreconditionError            : Si el payload no contiene datos BMC válidos.
        ToxicCycleVetoError          : Veto 1.
        EulerPoincareDegeneracyError : Veto 2.
        SpectralFragilityError       : Veto 3.
        """
        payload = state.payload
        nodes   = payload.get("bmc_nodes", [])
        flows   = payload.get("bmc_flows", [])

        if not nodes or not flows:
            raise PreconditionError(
                "El CategoricalState no contiene una topolog a BMC v lida. "
                "Se requieren 'bmc_nodes' (List[str]) y 'bmc_flows' "
                "(List[Tuple[str,str,float]]) en state.payload."
            )

        verdict = self.evaluate_business_canvas(nodes, flows)

        new_payload = dict(payload)
        new_payload["alpha_verdict"] = verdict

        return CategoricalState(
            payload  = new_payload,
            metadata = state.metadata,
            stratum  = Stratum.ALPHA,
        )

    #
    #  4.4   Propiedades de diagn stico de solo lectura
    #

    @property
    def last_verdict(self) -> Optional[AlphaBoundaryVerdict]:
        r"""
        Último veredicto emitido (incluyendo vetos).

        Útil para diagnóstico post-fallo: incluso cuando se lanza una excepción
        de veto, el veredicto parcial (con is_viable=False) está disponible aquí.

        Retorna None si evaluate_business_canvas no ha sido invocado.
        """
        return getattr(self, '_last_verdict', None)

    @property
    def fiedler_threshold(self) -> float:
        """Umbral de conectividad algebraica en uso."""
        return self._fiedler_threshold


#
# EXPORTACI N CAN NICA DEL ESTRATO
#
__all__ = [
    # Constantes
    "AlphaConstants",
    # Excepciones
    "AlphaBoundaryError",
    "PreconditionError",
    "ToxicCycleVetoError",
    "EulerPoincareDegeneracyError",
    "SpectralFragilityError",
    # DTOs topol gicos
    "SimplicialComplexData",
    "HomologicalInvariants",
    "SpectralFiedlerData",
    "AlphaBoundaryVerdict",
    # Agente principal
    "AlphaBoundaryAgent",
]
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Alpha Boundary Agent (Operador de Holonomía de Euler-Poincaré)       ║
║ Ubicación: app/agents/alpha/alpha_agent.py                                  ║
║ Versión: 2.0.0-Rigorous-Simplicial-Cohomology-Weighted                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Dictamen Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra el Estrato $\alpha$ (La Condición de Frontera Macroscópica)
del ecosistema APU_filter. Actuando como el Funtor Supremo de Contracción, el
`AlphaBoundaryAgent` (o `EulerPoincareHolonomyAgent`) repudia la validación
heurística de modelos de negocio (Business Model Canvas). En su lugar,
lo transfiere isométricamente a la categoría de los complejos simpliciales,
donde la topología del valor empresarial se somete a las leyes inmutables del
Álgebra Homológica y la Teoría Espectral de Grafos.

FUNDAMENTACIÓN AXIOMÁTICA Y OPERADORES DE HILBERT (v2.0.0):
────────────────────────────────────────────────────────────────────────────────
§1. EL FIBRADO DEL MODELO DE NEGOCIO (1-COMPLEJO SIMPLICIAL PONDERADO):
El grafo de negocio $G = (V, E, w)$ se proyecta a un 1-complejo simplicial
finito $K$ con pesos no negativos $w_k$ en cada flujo. Definimos la secuencia
exacta corta de cadenas sobre $\mathbb{R}$:
$$ 0 \longrightarrow C_1(K; \mathbb{R}) \xrightarrow{\partial_1} C_0(K; \mathbb{R}) \longrightarrow 0 $$
Donde $\partial_1$ es el Operador Frontera ponderado, materializado como la
matriz de incidencia orientada y escalada $|V| \times |E|$. Cada bloque del
lienzo (ej. Propuesta de Valor) es un vértice $v \in V$, y cada flujo es una
1-cadena $e \in E$ con intensidad $w(e) \ge 0$. La matriz $\partial_1$ incorpora
los pesos mediante la transformación $e_k \mapsto \sqrt{w(e_k)} \cdot e_k$, lo
que convierte al Laplaciano $L_0 = \partial_1 \partial_1^T$ en el operador de
Laplace ponderado que preserva la intensidad de los flujos.

§2. EL TEOREMA DE RANGO-NULIDAD Y LOS NÚMEROS DE BETTI:
Para extirpar la entropía de la Unidad de Punto Flotante (IEEE 754), calculamos
el rango algebraico de $\partial_1$ mediante Descomposición en Valores Singulares
(SVD). Los invariantes homológicos se extraen como:
$$ \beta_0 = |V| - \text{rank}(\partial_1) \quad (\text{Fragmentación del valor}) $$
$$ \beta_1 = |E| - \text{rank}(\partial_1) \equiv \dim(\ker(\partial_1)) \quad (\text{Ciclos Tóxicos}) $$
[AXIOMA DE VETO]: Si $\beta_1 > 0$, el modelo alberga "canibalización cruzada"
y dependencias circulares irresolubles. El agente detona `ToxicCycleVetoError`.

§3. EL INVARIANTE MACROSCÓPICO DE EULER-POINCARÉ:
La viabilidad termodinámica del negocio exige que el espacio no esté degenerado.
$$ \chi(K) = \sum_{i=0}^1 (-1)^i \beta_i = \beta_0 - \beta_1 = |V| - |E| $$
[AXIOMA DE VETO]: Si $\chi(K) \le 0$, el modelo carece de soporte vital (hiper-dependencia).
Se detona `EulerPoincareDegeneracyError` y el sistema colapsa antes de tocar
el Estrato ALEPH ($\aleph_0$).

§4. TEORÍA ESPECTRAL Y EL LAPLACIANO COMBINATORIO PONDERADO:
Construimos el operador Laplaciano de grado 0 para auditar la resistencia a
la fractura de la organización:
$$ L_0 = \partial_1 \partial_1^T $$
El espectro de esta matriz Hermitiana $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$
revela la conectividad algebraica a través del Valor de Fiedler ($\lambda_2$).
[AXIOMA DE VETO]: Si $\lambda_2(L_0) < \epsilon_{crit}$, el BMC posee un Punto
Único de Fallo (SPOF). Se detona `SpectralFragilityError`.

El agente `AlphaBoundaryAgent` orquesta estas tres fases categóricas, certificando
la integridad del subespacio antes de instanciar el resto de la Malla Agéntica.
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
    Optional,
    Tuple,
    Type,
    Protocol,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
    from app.core.schemas import Stratum
except ImportError:
    # Stubs rigurosos para ejecución aislada o test unitario analítico
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        pass

    class CategoricalState:
        payload: Dict[str, Any]
        metadata: Dict[str, Any]
        stratum: Any

        def __init__(self, payload, metadata=None, stratum=None):
            self.payload = payload
            self.metadata = metadata or {}
            self.stratum = stratum

    class Stratum(Enum):
        ALPHA = auto()


logger = logging.getLogger("MIC.Alpha.BoundaryAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS Y NUMÉRICAS DEL ESTRATO ALFA
# ══════════════════════════════════════════════════════════════════════════════

class AlphaConstants:
    r"""
    Constantes topológicas y tolerancias numéricas para el Estrato α.
    Garantizan la estabilidad asintótica de la FPU (IEEE-754) [4].
    """

    # Límite de truncamiento del espectro singular (SVD) [4]
    RANK_TOLERANCE: float = 1e-10

    # Umbral isoperimétrico de Cheeger para el Valor de Fiedler (λ₂) [4]
    MIN_FIEDLER_VALUE: float = 0.05

    # Epsilon de máquina para proyecciones reales ortogonales
    MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

    @staticmethod
    def spectral_zero_tolerance(n: int) -> float:
        """Devuelve una tolerancia dinámica para considerar un autovalor como cero."""
        return n * AlphaConstants.MACHINE_EPSILON


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS
# ══════════════════════════════════════════════════════════════════════════════

class AlphaBoundaryError(TopologicalInvariantError):
    """Excepción categórica base para violaciones en el Estrato α."""

    pass


class ToxicCycleVetoError(AlphaBoundaryError):
    r"""
    Se detona cuando el núcleo del operador frontera no es trivial: $\dim(\ker(\partial_1)) > 0$.
    Evidencia canibalización interna en el Business Model Canvas [3].
    """

    pass


class EulerPoincareDegeneracyError(AlphaBoundaryError):
    r"""
    Se detona cuando la característica de Euler es degenerada: $\chi(K) \le 0$.
    Evidencia una hiper-dependencia de flujos sobre un número deficiente de nodos [3].
    """

    pass


class SpectralFragilityError(AlphaBoundaryError):
    r"""
    Se detona cuando la conectividad algebraica (Valor de Fiedler) decae bajo el límite:
    $\lambda_2(L_0) < \epsilon_{crit}$. Revela un SPOF inminente en la organización [3].
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES (DTOs TOPOLÓGICOS)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SimplicialComplexData:
    r"""
    Artefacto de la Fase 1: Matriz de Incidencia del 1-esqueleto $\partial_1$ ponderada.
    Contiene la proyección ortogonal euclidiana del modelo de negocio incorporando
    las intensidades $\sqrt{w(e)}$.
    """

    vertices: Tuple[str, ...]
    edges: Tuple[Tuple[str, str, float], ...]
    boundary_operator: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HomologicalInvariants:
    r"""
    Artefacto de la Fase 2: Invariantes algebraicos derivados de la SVD.
    """

    rank: int
    beta_0: int
    beta_1: int
    euler_characteristic: int


@dataclass(frozen=True, slots=True)
class SpectralFiedlerData:
    r"""
    Artefacto de la Fase 3: Análisis de la matriz Laplaciana $L_0$.
    """

    laplacian: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    fiedler_value: float


@dataclass(frozen=True, slots=True)
class AlphaBoundaryVerdict:
    r"""
    Veredicto inmutable emitido por el `AlphaBoundaryAgent` al colapsar la evaluación.
    """

    is_viable: bool
    homology: HomologicalInvariants
    spectral: SpectralFiedlerData
    veto_reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS FUNTORIALES DE LAS 3 FASES
# ══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class CanvasFibratorPort(Protocol):
    def project_canvas_to_simplicial_complex(
        self, nodes: List[str], flows: List[Tuple[str, str, float]]
    ) -> SimplicialComplexData:
        ...


@runtime_checkable
class HomologicalAuditorPort(Protocol):
    def audit_betti_invariants(
        self, complex_data: SimplicialComplexData
    ) -> HomologicalInvariants:
        ...


@runtime_checkable
class SpectralAuditorPort(Protocol):
    def audit_fiedler_connectivity(
        self, complex_data: SimplicialComplexData
    ) -> SpectralFiedlerData:
        ...


# ┌──────────────────────────────────────────────────────────────────────────┐
# │ ORQUESTADOR SUPREMO: ALPHA BOUNDARY AGENT                                │
# │ Mapeo Endofuntorial que impone la condición de frontera macroscópica.    │
# │ Contiene anidadas las tres fases matemáticas, encadenadas por sus DTOs.  │
# └──────────────────────────────────────────────────────────────────────────┘

class AlphaBoundaryAgent(Morphism):
    r"""
    El `AlphaBoundaryAgent` (o `EulerPoincareHolonomyAgent`) orquesta la síntesis de
    las tres fases anidadas. Aplica los VETOS CATEGÓRICOS si los invariantes topológicos
    y espectrales del modelo de negocio indican una degeneración estructural [2, 3].

    Las fases residen como clases estáticas internas, garantizando que la salida
    formal del último método de la Fase 1 sea exactamente la entrada del primer
    método de la Fase 2, y así sucesivamente. Esta estructura anidada impone
    una composición funtorial verificable.

    Subordina la existencia misma del proyecto: si este agente veta, la Malla
    Agéntica no despierta.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · FIBRADOR DEL LIENZO SIMPLICIAL PONDERADO
    # ═══════════════════════════════════════════════════════════════════════
    class _Phase1_CanvasSimplicialFibrator:
        r"""
        Ejecuta el homeomorfismo entre los bloques semánticos del Business Model
        Canvas y un Espacio de Hilbert discreto. Genera la matriz de incidencia
        orientada ponderada de dimensiones $|V| \times |E|$ [3].
        """

        @staticmethod
        def project_canvas_to_simplicial_complex(
            nodes: List[str], flows: List[Tuple[str, str, float]]
        ) -> SimplicialComplexData:
            r"""
            Proyecta el digrafo a una matriz de incidencia ponderada
            $\partial_1 \in \mathbb{R}^{|V| \times |E|}$.

            Axioma de Orientación y Ponderación:
            Para una arista $e_k = (v_i, v_j)$ con peso $w_k \ge 0$, la columna
            correspondiente se escala por $\sqrt{w_k}$:

            $\partial_1(i, k) = -\sqrt{w_k}$ (Nodo fuente/origen)
            $\partial_1(j, k) = +\sqrt{w_k}$ (Nodo sumidero/destino)
            $\partial_1(m, k) = 0$  para todo $m \neq i, j$.

            Este escalamiento asegura que el Laplaciano $L_0 = \partial_1 \partial_1^T$
            sea el operador de Laplace ponderado, reflejando la intensidad real
            de los flujos de valor.
            """
            if not nodes:
                raise TopologicalInvariantError(
                    "El colector del BMC no puede ser el conjunto vacío $\emptyset$."
                )

            n_vertices = len(nodes)
            n_edges = len(flows)

            node_idx = {node: idx for idx, node in enumerate(nodes)}
            boundary_operator = np.zeros((n_vertices, n_edges), dtype=np.float64)

            for k, (u, v, weight) in enumerate(flows):
                if u not in node_idx or v not in node_idx:
                    raise TopologicalInvariantError(
                        f"Arista degenerada detectada: los nodos ({u}, {v}) no "
                        f"pertenecen al espacio de vértices V."
                    )
                if weight < 0:
                    raise TopologicalInvariantError(
                        f"El peso del flujo ({u} -> {v}) debe ser no negativo, "
                        f"se obtuvo {weight}."
                    )
                # Escalamiento por raíz cuadrada del peso para preservar la
                # estructura del Laplaciano ponderado.
                sqrt_weight = np.sqrt(weight)
                boundary_operator[node_idx[u], k] = -sqrt_weight
                boundary_operator[node_idx[v], k] = sqrt_weight

            return SimplicialComplexData(
                vertices=tuple(nodes),
                edges=tuple(flows),
                boundary_operator=boundary_operator,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · AUDITOR HOMOLÓGICO (BETTI Y EULER-POINCARÉ)
    # ═══════════════════════════════════════════════════════════════════════
    class _Phase2_HomologicalBettiAuditor:
        r"""
        Aplica el Teorema del Rango-Nulidad al operador frontera $\partial_1$ para
        extraer los invariantes topológicos globales $\beta_0, \beta_1, \chi$ [3].
        """

        @staticmethod
        def _compute_numerical_rank(matrix: NDArray[np.float64]) -> int:
            r"""
            Calcula el rango algebraico mediante SVD con umbral robusto:

            1. Si la matriz es vacía, retorna 0.
            2. Obtiene los valores singulares $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
            3. Define la tolerancia $\tau = \max(m,n) \cdot \epsilon_{\text{mach}} \cdot \sigma_1$,
               donde $\epsilon_{\text{mach}}$ es el épsilon de la FPU.
            4. Cuenta cuántos valores singulares superan $\max(\tau, \tau_{\text{abs}})$,
               con $\tau_{\text{abs}} = 10^{-10}$.
            """
            if matrix.size == 0:
                return 0

            singular_values = la.svd(matrix, compute_uv=False)
            sigma_max = singular_values[0]
            tol_rel = max(matrix.shape) * AlphaConstants.MACHINE_EPSILON * sigma_max
            threshold = max(tol_rel, AlphaConstants.RANK_TOLERANCE)
            rank = int(np.sum(singular_values > threshold))
            return rank

        @staticmethod
        def audit_betti_invariants(
            complex_data: SimplicialComplexData,
        ) -> HomologicalInvariants:
            r"""
            Computa $\beta_0$, $\beta_1$ y la Característica de Euler-Poincaré $\chi(K)$ [3].
            """
            abs_V = len(complex_data.vertices)
            abs_E = len(complex_data.edges)

            rank = AlphaBoundaryAgent._Phase2_HomologicalBettiAuditor._compute_numerical_rank(
                complex_data.boundary_operator
            )

            beta_0 = abs_V - rank
            beta_1 = abs_E - rank
            euler_characteristic = beta_0 - beta_1

            logger.info(
                f"Homología Computada: |V|={abs_V}, |E|={abs_E}, Rank={rank} "
                f"-> β₀={beta_0}, β₁={beta_1}, χ={euler_characteristic}"
            )

            return HomologicalInvariants(
                rank=rank,
                beta_0=beta_0,
                beta_1=beta_1,
                euler_characteristic=euler_characteristic,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · AUDITOR ESPECTRAL (LAPLACIANO Y VALOR DE FIEDLER)
    # ═══════════════════════════════════════════════════════════════════════
    class _Phase3_SpectralFiedlerAuditor:
        r"""
        Construye el Laplaciano Combinatorio ponderado $L_0 = \partial_1 \partial_1^T$ y
        determina la constante de conectividad algebraica $\lambda_2$ [3].
        """

        @staticmethod
        def audit_fiedler_connectivity(
            complex_data: SimplicialComplexData,
        ) -> SpectralFiedlerData:
            r"""
            Computa el espectro de autovalores del Laplaciano $L_0$.
            El primer autovalor teórico es $\lambda_1 = 0$. El Valor de Fiedler
            se define como el menor autovalor estrictamente positivo,
            $\lambda_2 = \min\{\lambda_i \mid \lambda_i > \delta\}$, donde $\delta$
            es una tolerancia proporcional a la dimensión y al épsilon de máquina.
            """
            B_1 = complex_data.boundary_operator
            L_0 = B_1 @ B_1.T

            # eigvalsh garantiza orden ascendente y simetría
            eigenvalues = la.eigvalsh(L_0)

            # Determinar el cero de máquina para los autovalores
            n = L_0.shape[0]
            zero_tol = AlphaConstants.spectral_zero_tolerance(n) * (
                np.max(eigenvalues) if eigenvalues.size > 0 else 1.0
            )

            # Extracción del valor de Fiedler como el primer autovalor > zero_tol
            if n < 2:
                fiedler = 0.0
            else:
                positive_mask = eigenvalues > zero_tol
                if np.any(positive_mask):
                    fiedler = float(np.min(eigenvalues[positive_mask]))
                else:
                    # Todos los autovalores son cero → grafo completamente disconexo
                    fiedler = 0.0

            logger.info(f"Análisis Espectral: λ₂ (Fiedler) = {fiedler:.6f}")

            return SpectralFiedlerData(
                laplacian=L_0,
                eigenvalues=eigenvalues,
                fiedler_value=fiedler,
            )

    # ───────────────────────────────────────────────────────────────────────
    # Constructor y métodos públicos del agente
    # ───────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """Instancia la composición anidada de las tres fases matemáticas."""
        # Las fases son clases estáticas internas, no requieren instanciación.
        pass

    def evaluate_business_canvas(
        self, nodes: List[str], flows: List[Tuple[str, str, float]]
    ) -> AlphaBoundaryVerdict:
        r"""
        Ejecuta el colapso topológico determinista del Modelo de Negocio (BMC).

        Axiomas de Veto Evaluados:
        1. Axioma de Canibalización: $\beta_1 > 0$
        2. Axioma de Degeneración: $\chi(K) \le 0$
        3. Axioma de Fragilidad: $\lambda_2(L_0) < \epsilon_{crit}$
        """
        try:
            # Fase 1 → DTO Simplicial (salida formal que alimenta la Fase 2)
            complex_data = self._Phase1_CanvasSimplicialFibrator.project_canvas_to_simplicial_complex(
                nodes, flows
            )

            # Fase 2 → Invariantes homológicos (recibe directamente el DTO de Fase 1)
            homology = self._Phase2_HomologicalBettiAuditor.audit_betti_invariants(
                complex_data
            )

            # [VETO AXIOMÁTICO 1]: Prevención de Ciclos Tóxicos
            if homology.beta_1 > 0:
                raise ToxicCycleVetoError(
                    f"Veto de Canibalización: El lienzo presenta β₁ = {homology.beta_1}. "
                    f"Existen dependencias circulares tóxicas imposibles de resolver termodinámicamente."
                )

            # [VETO AXIOMÁTICO 2]: Invariante Macroscópico Degenerado
            if homology.euler_characteristic <= 0:
                raise EulerPoincareDegeneracyError(
                    f"Veto de Degeneración: La característica de Euler es χ = {homology.euler_characteristic}. "
                    f"El negocio es un hipergrafo parasitario sin cimentación nodal válida."
                )

            # Fase 3 → Certificación Espectral (recibe el mismo DTO, garantizando continuidad)
            spectral_data = self._Phase3_SpectralFiedlerAuditor.audit_fiedler_connectivity(
                complex_data
            )

            # [VETO AXIOMÁTICO 3]: Límite Isoperimétrico y Robustez
            if spectral_data.fiedler_value < AlphaConstants.MIN_FIEDLER_VALUE:
                raise SpectralFragilityError(
                    f"Veto de Fragilidad Espectral: El valor de Fiedler λ₂ = {spectral_data.fiedler_value:.5f} "
                    f"es inferior a la tolerancia isoperimétrica ({AlphaConstants.MIN_FIEDLER_VALUE}). "
                    f"La red de valor posee Puntos Únicos de Fallo (SPOF) catastróficos."
                )

            logger.info(
                "El Business Model Canvas superó la auditoría de Holonomía de Euler-Poincaré."
            )

            return AlphaBoundaryVerdict(
                is_viable=True,
                homology=homology,
                spectral=spectral_data,
                veto_reason=None,
            )

        except AlphaBoundaryError as e:
            logger.critical(f"COLAPSO ESTRATO α: {str(e)}")
            raise e

    def __call__(self, state: CategoricalState) -> CategoricalState:
        r"""
        Implementación del protocolo `Morphism` de la Matriz de Interacción Central (MIC).
        Intercepta el estado entrante, asume que el payload contiene la definición
        estructural del BMC y aplica el Difeomorfismo de Holonomía.
        """
        payload = state.payload
        nodes = payload.get("bmc_nodes", [])
        flows = payload.get("bmc_flows", [])

        if not nodes or not flows:
            raise TopologicalInvariantError(
                "El estado categórico no contiene una topología BMC válida."
            )

        verdict = self.evaluate_business_canvas(nodes, flows)

        new_payload = dict(payload)
        new_payload["alpha_verdict"] = verdict

        return CategoricalState(
            payload=new_payload,
            metadata=state.metadata,
            stratum=Stratum.ALPHA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DEL ESTRATO α
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "AlphaConstants",
    "AlphaBoundaryError",
    "ToxicCycleVetoError",
    "EulerPoincareDegeneracyError",
    "SpectralFragilityError",
    "SimplicialComplexData",
    "HomologicalInvariants",
    "SpectralFiedlerData",
    "AlphaBoundaryVerdict",
    "AlphaBoundaryAgent",
]
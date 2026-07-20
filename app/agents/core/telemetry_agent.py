# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Agent (El Custodio de la Propagación Causal)              ║
║ Ruta   : app/agents/core/telemetry_agent.py                                  ║
║ Versión: 2.0.0-Causal-Cohomology-Port-Hamiltonian-Strict-Nested              ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna a `telemetry.py`. Actúa como el Funtor de Propagación
Topológica que administra el tránsito del Pasaporte de Telemetría, asegurando
que la flecha del tiempo y la causalidad constituyan geodésicas válidas.

ARQUITECTURA DE 3 FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Filtración de Clausura Transitiva:
         Impone la inclusión estricta de subespacios anidados (Zero-Trust):
             V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM.
         Aniquila instanciaciones superiores si la base carece de coherencia.

Fase 2 → Cohomología de Spans Causales:
         Modela la jerarquía de ejecución como un complejo simplicial 1-dimensional.
         Aplica la característica de Euler-Poincaré:
             χ(K) = β₀ - β₁ = |V| - |E|.
         Exige β₁ = 0 para certificar un bosque causal sin ciclos.

Fase 3 → Disipación Port-Hamiltoniana:
         Evalúa la irreversibilidad termodinámica del sistema calculando la
         energía disipada de Rayleigh:
             P_diss = ∇Hᵀ R ∇H ≥ 0,
         con R simétrica y positiva semidefinida.

COMPOSICIÓN:
────────────
El último método de la Fase 1 emite un puente formal que es consumido por el
primer método de la Fase 2. El último método de la Fase 2 emite un puente que
es consumido por el primer método de la Fase 3.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, unique
from typing import FrozenSet, Set

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos $\mathcal{E}_{MIC}$."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass

    class CategoricalState:
        """Clase base de Estados Categóricos."""
        pass

    @unique
    class Stratum(Enum):
        PHYSICS = 0
        TACTICS = 1
        STRATEGY = 2
        WISDOM = 3


logger = logging.getLogger("MIC.Core.TelemetryAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. TIPOS Y CONSTANTES FÍSICAS, TOPOLOGICAS Y TERMODINÁMICAS
# ═══════════════════════════════════════════════════════════════════════════════
VectorF64 = NDArray[np.float64]
MatrixF64 = NDArray[np.float64]

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

# Límite anti-runaway para grafos de ejecución.
_MAX_CAUSAL_DEPTH: int = 1024

# Tolerancias termodinámicas y espectrales.
_DISSIPATION_NEGATIVE_TOLERANCE: float = 1e-12
_PSD_EIGENVALUE_TOLERANCE: float = 1e-12
_SYMMETRY_WARNING_RATIO: float = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CAUSALES Y TERMODINÁMICAS
# ═══════════════════════════════════════════════════════════════════════════════
class TelemetryAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Propagación Causal."""
    pass


class TransitiveClosureViolation(TelemetryAgentError):
    r"""
    Detonada si un estrato superior intenta consolidarse sin el respaldo
    termométrico o topológico del estrato inferior.
    """
    pass


class CausalCohomologyError(TelemetryAgentError):
    r"""
    Detonada si $\beta_1 > 0$ o si los invariantes del grafo causal son
    inconsistentes.
    """
    pass


class ThermodynamicReversibilityError(TelemetryAgentError):
    r"""
    Detonada si $P_{diss} < 0$ o si la matriz de disipación $R$ no es
    positiva semidefinida.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio Causal)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True, eq=False)
class FiltrationAuditData:
    r"""
    Artefacto de Fase 1.
    Certificado de clausura transitiva.
    """
    active_strata: FrozenSet[Stratum]
    is_filtration_valid: bool


@dataclass(frozen=True, slots=True, eq=False)
class CausalCohomologyData:
    r"""
    Artefacto de Fase 2.
    Certificado de la Característica de Euler-Poincaré.
    """
    vertices_count: int
    edges_count: int
    betti_0: int
    betti_1: int
    euler_characteristic: int
    is_acyclic_directed: bool


@dataclass(frozen=True, slots=True, eq=False)
class ThermodynamicDissipationData:
    r"""
    Artefacto de Fase 3.
    Certificado de irreversibilidad Port-Hamiltoniana.
    """
    dissipated_power: float
    gradient_norm: float
    spectral_min: float
    spectral_max: float
    is_entropically_valid: bool


@dataclass(frozen=True, slots=True, eq=False)
class Phase1CausalBridge:
    r"""
    Puente funtorial Φ₁ → Φ₂.

    Este objeto es emitido por el último método de la Fase 1 y constituye
    la entrada formal del primer método de la Fase 2.
    """
    filtration_audit: FiltrationAuditData
    total_spans: int
    causal_edges: int
    connected_components: int
    grad_H: VectorF64
    R_matrix: MatrixF64


@dataclass(frozen=True, slots=True, eq=False)
class Phase2CohomologyBridge:
    r"""
    Puente funtorial Φ₂ → Φ₃.

    Este objeto es emitido por el último método de la Fase 2 y constituye
    la entrada formal del primer método de la Fase 3.
    """
    phase1_bridge: Phase1CausalBridge
    cohomology_audit: CausalCohomologyData


@dataclass(frozen=True, slots=True, eq=False)
class CausalPropagationState:
    r"""
    Objeto final del endofuntor $\mathcal{Z}_{Telemetry}$.
    """
    filtration_audit: FiltrationAuditData
    cohomology_audit: CausalCohomologyData
    dissipation_audit: ThermodynamicDissipationData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: FILTRACIÓN DE CLAUSURA TRANSITIVA                                 ║
# ║                                                                             ║
# ║   Exige:                                                                    ║
# ║       V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM                         ║
# ║                                                                             ║
# ║   1. Valida estratos completados.                                           ║
# ║   2. Impone clausura transitiva Zero-Trust.                                 ║
# ║   3. Valida entradas causales y termodinámicas.                             ║
# ║   4. Emite el puente formal hacia la Fase 2.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_TransitiveClosureEnforcer:
    r"""
    Fase 1 del endofuntor.

    Garantiza que la propagación del contexto Zero-Trust respete la jerarquía
    DIKW. Impide que el estrato superior delibere si el motor físico no ha
    avalado los datos.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Validación numérica elemental
    # ─────────────────────────────────────────────────────────────────────────
    def _as_nonnegative_int(self, name: str, value: int) -> int:
        """
        Valida un entero no negativo exacto.
        """
        if isinstance(value, (bool, np.bool_)):
            raise TelemetryAgentError(
                f"{name} no puede ser un valor booleano."
            )

        try:
            ivalue = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TelemetryAgentError(
                f"{name} debe ser un entero no negativo."
            ) from exc

        try:
            equality = (value == ivalue)
            if isinstance(equality, np.ndarray):
                equality = bool(np.all(equality))
            if not bool(equality):
                raise TelemetryAgentError(
                    f"{name} debe ser un entero exacto, no una aproximación."
                )
        except (TypeError, ValueError):
            pass

        if ivalue < 0:
            raise TelemetryAgentError(
                f"{name} debe ser no negativo. Se recibió {ivalue}."
            )

        return ivalue

    def _as_finite_vector(self, name: str, value: VectorF64) -> VectorF64:
        """
        Valida que el objeto sea un vector 1-D no vacío y con componentes finitas.
        """
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TelemetryAgentError(
                f"{name} no puede convertirse a un vector float64."
            ) from exc

        if arr.ndim != 1:
            raise TelemetryAgentError(
                f"{name} debe ser un vector 1-D. Dimensión recibida: {arr.ndim}."
            )

        if arr.size == 0:
            raise TelemetryAgentError(
                f"{name} no puede ser el vector vacío."
            )

        if not np.all(np.isfinite(arr)):
            raise TelemetryAgentError(
                f"{name} contiene componentes NaN o infinitas."
            )

        return arr

    def _as_finite_square_matrix(self, name: str, value: MatrixF64) -> MatrixF64:
        """
        Valida que el objeto sea una matriz cuadrada no vacía y finita.
        """
        try:
            mat = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TelemetryAgentError(
                f"{name} no puede convertirse a una matriz float64."
            ) from exc

        if mat.ndim != 2:
            raise TelemetryAgentError(
                f"{name} debe ser una matriz 2-D. Dimensión recibida: {mat.ndim}."
            )

        if mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
            raise TelemetryAgentError(
                f"{name} no puede ser una matriz vacía."
            )

        if mat.shape[0] != mat.shape[1]:
            raise TelemetryAgentError(
                f"{name} debe ser una matriz cuadrada. Shape recibido: {mat.shape}."
            )

        if not np.all(np.isfinite(mat)):
            raise TelemetryAgentError(
                f"{name} contiene entradas NaN o infinitas."
            )

        return mat

    # ─────────────────────────────────────────────────────────────────────────
    # Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_l2_norm(self, vector: VectorF64) -> float:
        """
        Calcula ||v||₂ con reescalado para evitar overflow/underflow.
        """
        if vector.size == 0:
            return 0.0

        scale = float(np.max(np.abs(vector)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = vector / scale
        ss = float(np.vdot(scaled, scaled).real)

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    def _safe_fro_norm(self, matrix: MatrixF64) -> float:
        """
        Calcula ||M||_F con reescalado para evitar overflow/underflow.
        """
        if matrix.size == 0:
            return 0.0

        scale = float(np.max(np.abs(matrix)))
        if scale == 0.0:
            return 0.0

        if not math.isfinite(scale):
            return math.inf

        scaled = matrix / scale
        ss = float(np.sum(np.abs(scaled) ** 2))

        if not math.isfinite(ss):
            return math.inf

        norm = scale * math.sqrt(ss)
        return float(norm) if math.isfinite(norm) else math.inf

    def _safe_quadratic_form(
        self,
        x: VectorF64,
        A: MatrixF64
    ) -> float:
        r"""
        Calcula $x^T A x$ con reescalado para reducir overflow/underflow.
        """
        x_norm = self._safe_l2_norm(x)

        if not math.isfinite(x_norm):
            raise ThermodynamicReversibilityError(
                "Norma no finita del gradiente Hamiltoniano."
            )

        if x_norm <= _MACHINE_EPSILON:
            return 0.0

        if A.size == 0:
            return 0.0

        a_scale = float(np.max(np.abs(A)))

        if not math.isfinite(a_scale):
            raise ThermodynamicReversibilityError(
                "Escala no finita en la matriz de disipación R."
            )

        if a_scale <= _MACHINE_EPSILON:
            return 0.0

        x_unit = x / x_norm
        A_unit = A / a_scale

        y = A_unit @ x_unit

        if not np.all(np.isfinite(y)):
            raise ThermodynamicReversibilityError(
                "Producto matriz-vector no finito al evaluar la forma cuadrática."
            )

        q_unit = float(np.vdot(x_unit, y).real)

        if not math.isfinite(q_unit):
            raise ThermodynamicReversibilityError(
                "Forma cuadrática normalizada no finita."
            )

        with np.errstate(over="ignore", invalid="ignore"):
            q = (x_norm * x_norm) * a_scale * q_unit

        if not math.isfinite(q):
            raise ThermodynamicReversibilityError(
                "Forma cuadrática de disipación no finita por desbordamiento numérico."
            )

        return float(q)

    # ─────────────────────────────────────────────────────────────────────────
    # Validación de estratos
    # ─────────────────────────────────────────────────────────────────────────
    def _validate_completed_strata(
        self,
        completed_strata: Set[Stratum]
    ) -> FrozenSet[Stratum]:
        """
        Valida y normaliza el conjunto de estratos completados.
        """
        if completed_strata is None:
            raise TransitiveClosureViolation(
                "El conjunto de estratos completados es None."
            )

        try:
            raw_strata = list(completed_strata)
        except TypeError as exc:
            raise TransitiveClosureViolation(
                "completed_strata no es iterable ni convertible a conjunto."
            ) from exc

        clean_strata: Set[Stratum] = set()

        for idx, item in enumerate(raw_strata):
            if isinstance(item, Stratum):
                clean_strata.add(item)
                continue

            try:
                clean_strata.add(Stratum(int(item)))
                continue
            except (TypeError, ValueError, KeyError):
                pass

            try:
                clean_strata.add(Stratum[str(item)])
                continue
            except (TypeError, KeyError) as exc:
                raise TransitiveClosureViolation(
                    f"Estrato inválido en índice {idx}: {item!r}."
                ) from exc

        return frozenset(clean_strata)

    # ─────────────────────────────────────────────────────────────────────────
    # Clausura transitiva de la jerarquía DIKW
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_transitive_closure_filtration(
        self,
        completed_strata: Set[Stratum]
    ) -> FiltrationAuditData:
        r"""
        Verifica el cumplimiento de la contención de subespacios DIKW:
            $V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$.
        """
        active_strata = self._validate_completed_strata(completed_strata)

        has_physics = Stratum.PHYSICS in active_strata
        has_tactics = Stratum.TACTICS in active_strata
        has_strategy = Stratum.STRATEGY in active_strata
        has_wisdom = Stratum.WISDOM in active_strata

        if has_wisdom and not (has_strategy and has_tactics and has_physics):
            raise TransitiveClosureViolation(
                "Infracción del Fibrado Causal: El estrato WISDOM intentó colapsar el "
                "estado sin el aval completo de STRATEGY, TACTICS y PHYSICS."
            )

        if has_strategy and not (has_tactics and has_physics):
            raise TransitiveClosureViolation(
                "Infracción del Fibrado Causal: El estrato STRATEGY carece de la "
                "validación estructural de TACTICS y PHYSICS."
            )

        if has_tactics and not has_physics:
            raise TransitiveClosureViolation(
                "Infracción del Fibrado Causal: El estrato TACTICS no puede operar "
                "sin la certidumbre termodinámica del estrato PHYSICS."
            )

        return FiltrationAuditData(
            active_strata=active_strata,
            is_filtration_valid=True
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ÚLTIMO MÉTODO DE FASE 1
    # Puente formal hacia FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _complete_phase1_causal_filtration(
        self,
        completed_strata: Set[Stratum],
        total_spans: int,
        causal_edges: int,
        grad_H: VectorF64,
        R_matrix: MatrixF64,
        connected_components: int = 1
    ) -> Phase1CausalBridge:
        r"""
        Último método de la Fase 1.

        Ejecuta:
            1. Auditoría de clausura transitiva.
            2. Validación de invariantes causales.
            3. Validación de entradas termodinámicas.
            4. Emisión del puente funtorial hacia la Fase 2.

        Este retorno es la continuación formal de la Fase 1 y el argumento
        inicial obligatorio del primer método de la Fase 2.
        """
        filtration_audit = self._enforce_transitive_closure_filtration(
            completed_strata
        )

        n_spans = self._as_nonnegative_int("total_spans", total_spans)
        n_edges = self._as_nonnegative_int("causal_edges", causal_edges)
        n_components = self._as_nonnegative_int(
            "connected_components",
            connected_components
        )

        grad = self._as_finite_vector("grad_H", grad_H)
        R = self._as_finite_square_matrix("R_matrix", R_matrix)

        if R.shape[0] != grad.size:
            raise TelemetryAgentError(
                f"Dimensión incompatible: R_matrix={R.shape}, grad_H={grad.size}."
            )

        return Phase1CausalBridge(
            filtration_audit=filtration_audit,
            total_spans=n_spans,
            causal_edges=n_edges,
            connected_components=n_components,
            grad_H=grad,
            R_matrix=R
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: COHOMOLOGÍA DE SPANS CAUSALES                                     ║
# ║                                                                             ║
# ║   χ(K) = |V| - |E| = β₀ - β₁                                                ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 1.                               ║
# ║   2. Valida consistencia combinatoria del grafo causal.                     ║
# ║   3. Calcula β₁.                                                            ║
# ║   4. Exige β₁ = 0.                                                          ║
# ║   5. Emite el puente formal hacia la Fase 3.                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_CausalCohomologyAuditor(Phase1_TransitiveClosureEnforcer):
    r"""
    Fase 2 del endofuntor.

    Modela la historia de ejecución como un complejo simplicial 1-dimensional,
    garantizando que ninguna tarea cíclica estocástica produzca un bucle infinito.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 2
    # Inicio formal a partir del puente de Fase 1
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase2_from_phase1_bridge(
        self,
        phase1_bridge: Phase1CausalBridge
    ) -> Phase2CohomologyBridge:
        r"""
        Primer método de la Fase 2.

        Consume el `Phase1CausalBridge` emitido por el último método de la
        Fase 1 y ejecuta la auditoría cohomológica.
        """
        if not isinstance(phase1_bridge, Phase1CausalBridge):
            raise TelemetryAgentError(
                "La Fase 2 requiere un Phase1CausalBridge emitido por la Fase 1."
            )

        cohomology_audit = self._audit_causal_span_cohomology(
            total_spans=phase1_bridge.total_spans,
            causal_edges=phase1_bridge.causal_edges,
            connected_components=phase1_bridge.connected_components
        )

        return Phase2CohomologyBridge(
            phase1_bridge=phase1_bridge,
            cohomology_audit=cohomology_audit
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Auditoría cohomológica de spans causales
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_causal_span_cohomology(
        self,
        total_spans: int,
        causal_edges: int,
        connected_components: int = 1
    ) -> CausalCohomologyData:
        r"""
        Calcula el número de Betti 1:
            $\beta_1 = \beta_0 - |V| + |E|$.

        Para un bosque causal:
            $|E| = |V| - \beta_0$,
        por tanto:
            $\beta_1 = 0$.

        Si $\beta_1 > 0$, existen ciclos homológicos.
        Si $\beta_1 < 0$, los invariantes son combinatoriamente inconsistentes.
        """
        vertices = self._as_nonnegative_int("total_spans", total_spans)
        edges = self._as_nonnegative_int("causal_edges", causal_edges)
        components = self._as_nonnegative_int(
            "connected_components",
            connected_components
        )

        # Complejo simplicial trivial.
        if vertices == 0:
            if edges != 0:
                raise CausalCohomologyError(
                    "Grafo causal inconsistente: vértices=0 pero aristas>0."
                )

            # Se acepta connected_components=1 por compatibilidad con el valor
            # por defecto, pero el grafo vacío tiene 0 componentes.
            if components not in (0, 1):
                raise CausalCohomologyError(
                    "Grafo causal vacío con connected_components inválido."
                )

            return CausalCohomologyData(
                vertices_count=0,
                edges_count=0,
                betti_0=0,
                betti_1=0,
                euler_characteristic=0,
                is_acyclic_directed=True
            )

        if vertices > _MAX_CAUSAL_DEPTH:
            raise CausalCohomologyError(
                f"Degeneración Causal: Profundidad del span ({vertices}) "
                f"excede el límite del hiperespacio operativo ({_MAX_CAUSAL_DEPTH})."
            )

        if components < 1 or components > vertices:
            raise CausalCohomologyError(
                f"connected_components={components} es inconsistente para "
                f"total_spans={vertices}."
            )

        # Para un grafo simple no dirigido subyacente, el número máximo de
        # aristas es n(n-1)/2. En un DAG causal simple, esta cota también aplica
        # al grafo subyacente.
        max_simple_edges = vertices * (vertices - 1) // 2

        if edges > max_simple_edges:
            raise CausalCohomologyError(
                f"Grafo causal inconsistente: edges={edges} excede el máximo "
                f"simple para vertices={vertices} (max={max_simple_edges})."
            )

        # β₁ = β₀ - |V| + |E|
        betti_1 = components - vertices + edges

        if betti_1 < 0:
            raise CausalCohomologyError(
                "Invariante causal inconsistente: β₁ < 0. "
                f"components={components}, vertices={vertices}, edges={edges}. "
                "El número de aristas es insuficiente para el número de componentes."
            )

        if betti_1 > 0:
            raise CausalCohomologyError(
                "Paradoja Topológica Detectada: El árbol de ejecución contiene "
                f"β₁={betti_1} ciclos homológicos. El Grafo Acíclico Dirigido (DAG) "
                "se ha corrompido con recursión estocástica."
            )

        euler_characteristic = vertices - edges

        return CausalCohomologyData(
            vertices_count=vertices,
            edges_count=edges,
            betti_0=components,
            betti_1=betti_1,
            euler_characteristic=euler_characteristic,
            is_acyclic_directed=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: DISIPACIÓN PORT-HAMILTONIANA                                      ║
# ║                                                                             ║
# ║   Segunda Ley:                                                              ║
# ║       P_diss = ∇Hᵀ R ∇H ≥ 0                                                 ║
# ║                                                                             ║
# ║   1. Consume el puente emitido por la Fase 2.                               ║
# ║   2. Valida R como matriz simétrica positiva semidefinida.                  ║
# ║   3. Computa la potencia disipada de Rayleigh.                              ║
# ║   4. Veta energía negativa o matrices no PSD.                               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_PortHamiltonianVerifier(Phase2_CausalCohomologyAuditor):
    r"""
    Fase 3 del endofuntor.

    Avala la irreversibilidad termodinámica del tránsito de telemetría.
    Un flujo que produce energía negativa o que emplea una matriz de disipación
    no positiva semidefinida es un artefacto estocástico inválido.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMER MÉTODO DE FASE 3
    # Inicio formal a partir del puente de Fase 2
    # ─────────────────────────────────────────────────────────────────────────
    def _begin_phase3_from_phase2_bridge(
        self,
        phase2_bridge: Phase2CohomologyBridge
    ) -> ThermodynamicDissipationData:
        r"""
        Primer método de la Fase 3.

        Consume el `Phase2CohomologyBridge` emitido por la Fase 2 y ejecuta la
        verificación Port-Hamiltoniana.
        """
        if not isinstance(phase2_bridge, Phase2CohomologyBridge):
            raise TelemetryAgentError(
                "La Fase 3 requiere un Phase2CohomologyBridge emitido por la Fase 2."
            )

        return self._verify_port_hamiltonian_dissipation(
            grad_H=phase2_bridge.phase1_bridge.grad_H,
            R_matrix=phase2_bridge.phase1_bridge.R_matrix
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Verificación de disipación Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_port_hamiltonian_dissipation(
        self,
        grad_H: VectorF64,
        R_matrix: MatrixF64
    ) -> ThermodynamicDissipationData:
        r"""
        Computa la disipación energética de Rayleigh:
            $P_{diss} = \nabla H^\top R \nabla H$.

        Exige:
            1. $R = R^\top$ (simetría, salvo ruido numérico).
            2. $R \succeq 0$ (positiva semidefinida).
            3. $P_{diss} \geq 0$.
        """
        grad = self._as_finite_vector("grad_H", grad_H)
        R = self._as_finite_square_matrix("R_matrix", R_matrix)

        if R.shape != (grad.size, grad.size):
            raise TelemetryAgentError(
                f"Dimensión incompatible: R_matrix={R.shape}, grad_H={grad.size}."
            )

        # La matriz de disipación Port-Hamiltoniana es auto-adjunta.
        R_sym = 0.5 * (R + R.T)

        fro_R = self._safe_fro_norm(R)
        fro_asym = self._safe_fro_norm(R - R_sym)

        if math.isfinite(fro_R) and math.isfinite(fro_asym):
            if fro_asym > _SYMMETRY_WARNING_RATIO * max(1.0, fro_R):
                logger.warning(
                    "Matriz de disipación R con asimetría relevante. "
                    f"||R-R^T||_F={fro_asym:.3e}. Se impone la parte simétrica."
                )
        elif not math.isfinite(fro_asym):
            logger.warning(
                "No fue posible certificar finiteza de la asimetría de R. "
                "Se procede bajo simetrización forzada."
            )

        try:
            eigenvalues = la.eigvalsh(R_sym, check_finite=False)
        except la.LinAlgError as exc:
            raise ThermodynamicReversibilityError(
                "Fallo en la diagonalización de la matriz de disipación R."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise ThermodynamicReversibilityError(
                "El espectro de la matriz de disipación contiene valores no finitos."
            )

        lambda_min = float(np.min(eigenvalues))
        lambda_max = float(np.max(eigenvalues))

        if lambda_min < -_PSD_EIGENVALUE_TOLERANCE:
            raise ThermodynamicReversibilityError(
                "Matriz de disipación no positiva semidefinida. "
                f"lambda_min={lambda_min:.6e} < -tol={_PSD_EIGENVALUE_TOLERANCE:.6e}."
            )

        spectral_min = max(0.0, lambda_min)
        spectral_max = max(0.0, lambda_max)

        p_diss = self._safe_quadratic_form(grad, R_sym)

        if p_diss < -_DISSIPATION_NEGATIVE_TOLERANCE:
            raise ThermodynamicReversibilityError(
                "Violación Port-Hamiltoniana. Energía disipada anómala: "
                f"P_diss={p_diss:.6e} < 0. Se ha inyectado entropía negativa "
                "o ruido estocástico artificial en la malla agéntica."
            )

        if p_diss < 0.0:
            p_diss = 0.0

        grad_norm = self._safe_l2_norm(grad)

        if not math.isfinite(grad_norm):
            raise ThermodynamicReversibilityError(
                "Norma no finita del gradiente Hamiltoniano."
            )

        return ThermodynamicDissipationData(
            dissipated_power=p_diss,
            gradient_norm=grad_norm,
            spectral_min=spectral_min,
            spectral_max=spectral_max,
            is_entropically_valid=True
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: TELEMETRY AGENT                                      ║
# ║                                                                             ║
# ║   Endofuntor:                                                               ║
# ║       Z_Telemetry = Φ₃ ∘ Φ₂ ∘ Φ₁                                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TelemetryAgent(Morphism, Phase3_PortHamiltonianVerifier):
    r"""
    El Custodio de la Propagación Causal.

    Subordina la telemetría y el historial de la Malla Agéntica a la topología
    algebraica de los grafos causales y a la termodinámica de campos.
    """

    def execute_causal_propagation_governance(
        self,
        completed_strata: Set[Stratum],
        total_spans: int,
        causal_edges: int,
        grad_H: VectorF64,
        R_matrix: MatrixF64,
        connected_components: int = 1
    ) -> CausalPropagationState:
        r"""
        Ejecuta la composición funtorial estricta en 3 fases anidadas.

        Flujo:
        ----
        1. Fase 1:
           `_complete_phase1_causal_filtration`
           → `Phase1CausalBridge`

        2. Fase 2:
           `_begin_phase2_from_phase1_bridge`
           → `Phase2CohomologyBridge`

        3. Fase 3:
           `_begin_phase3_from_phase2_bridge`
           → `ThermodynamicDissipationData`

        4. Ensamblaje final:
           `CausalPropagationState`
        """
        # ── Fase 1: Filtración de clausura transitiva y puente causal ────────
        phase1_bridge = self._complete_phase1_causal_filtration(
            completed_strata=completed_strata,
            total_spans=total_spans,
            causal_edges=causal_edges,
            grad_H=grad_H,
            R_matrix=R_matrix,
            connected_components=connected_components
        )

        # ── Fase 2: Cohomología de spans causales ────────────────────────────
        phase2_bridge = self._begin_phase2_from_phase1_bridge(
            phase1_bridge=phase1_bridge
        )

        # ── Fase 3: Disipación Port-Hamiltoniana ─────────────────────────────
        dissipation_audit = self._begin_phase3_from_phase2_bridge(
            phase2_bridge=phase2_bridge
        )

        # ── Ensamblaje del objeto final ──────────────────────────────────────
        final_state = CausalPropagationState(
            filtration_audit=phase1_bridge.filtration_audit,
            cohomology_audit=phase2_bridge.cohomology_audit,
            dissipation_audit=dissipation_audit,
            is_epistemologically_valid=True
        )

        logger.info(
            "Fibrado Causal certificado con éxito. "
            f"Filtro: {len(final_state.filtration_audit.active_strata)} estratos | "
            f"χ(K): {final_state.cohomology_audit.euler_characteristic} | "
            f"β₁: {final_state.cohomology_audit.betti_1} | "
            f"P_diss: {final_state.dissipation_audit.dissipated_power:.6e}"
        )

        return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "TelemetryAgentError",
    "TransitiveClosureViolation",
    "CausalCohomologyError",
    "ThermodynamicReversibilityError",
    "FiltrationAuditData",
    "CausalCohomologyData",
    "ThermodynamicDissipationData",
    "Phase1CausalBridge",
    "Phase2CohomologyBridge",
    "CausalPropagationState",
    "Phase1_TransitiveClosureEnforcer",
    "Phase2_CausalCohomologyAuditor",
    "Phase3_PortHamiltonianVerifier",
    "TelemetryAgent",
]
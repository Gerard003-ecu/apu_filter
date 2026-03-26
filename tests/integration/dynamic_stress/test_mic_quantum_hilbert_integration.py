"""
Suite de integración geométrica: MIC - QuantumAdmissionGate & HilbertObserverAgent.

Aserciones algebraicas (invariantes de la suite):
═══════════════════════════════════════════════════════════════════════════════
A. Ortogonalidad funcional:      ⟨e_i, e_j⟩ = δ_{ij}     (Kronecker)
B. Independencia lineal:         rank(T) = dim(V) = 2   (columnas l.i.)
C. Proyección idempotente:       P_i² = P_i  ∀i ∈ {1,2}
D. Resolución de identidad:      P_1 + P_2 = I_obs      (completitud)
E. Preservación monótona:        veto(f(s)) ⟹ veto(g∘f(s))
F. Functorialidad DIKW:          F(g ∘ f) = F(g) ∘ F(f)
G. Conmutatividad del p.i.:      ⟨v₁, v₂⟩ = ⟨v₂, v₁⟩

Marco teórico:
───────────────────────────────────────────────────────────────────────────────
Sea V = ℝ² el espacio observacional con base canónica {e₁, e₂}.
Cada operador MIC induce un proyector ortogonal P_i: V → V.

Estructura algebraica:
  • Proyectores binarios: P_i(s) ∈ {0, 1}  ∀s
  • Matriz de Gram: G = T^T T diagonal con entradas σ_i² ∈ {0,1}
  • Orthonormality: ⟨e_i, e_j⟩ = δ_{ij}  (base ortonormal canónica)
  • Lattice observacional: {(0,0), (1,0), (0,1)}  (estado nulo + dos estados puros)

La categoría DIKW se modela como categoría thin (preorden) donde los morfismos
preservan la relación de orden parcial inducida por la veto-monotonía.

Propiedades de estabilidad numérica:
  • Tolerancia de ortogonalidad: 1e-12  (estricto para bases canónicas)
  • Tolerancia de rango: 1e-12         (detecta deficiencia de rango)
  • Tolerancia numérica general: 1e-12 (errores de redondeo)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import unittest.mock as mock
import warnings

import numpy as np
import pytest

from app.adapters.tools_interface import MICRegistry
from app.core.mic_algebra import CategoricalState
from app.core.schemas import Stratum
from app.physics.quantum_admission_gate import QuantumAdmissionGate
from app.agents.hilbert_watcher import HilbertObserverAgent


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES MATEMÁTICAS
# ═════════════════════════════════════════════════════════════════════════════

_ORTHOGONALITY_TOLERANCE: float = 1e-12
"""Tolerancia para verificación de ortogonalidad ⟨v_i, v_j⟩ = δ_ij."""

_NUMERICAL_TOLERANCE: float = 1e-12
"""Tolerancia general para operaciones numéricas de punto flotante."""

_RANK_TOLERANCE: float = 1e-12
"""Tolerancia para cálculo de rango (np.linalg.matrix_rank)."""

_SPACE_DIMENSION: int = 2
"""Dimensión del espacio observacional V = ℝ²."""

_HILBERT_SPACE_DIMENSION: int = 2
"""Dimensión alternativa para Hilbert (documentación explícita)."""

_VALID_LATTICE_VECTORS: Tuple[np.ndarray, ...] = (
    np.array([0.0, 0.0], dtype=np.float64),  # Estado nulo
    np.array([1.0, 0.0], dtype=np.float64),  # e₁ puro
    np.array([0.0, 1.0], dtype=np.float64),  # e₂ puro
)
"""Retículo válido de vectores observacionales: {(0,0), (1,0), (0,1)}."""

_QUANTUM_GATE_TRACE_FRAGMENTS: FrozenSet[str] = frozenset({
    "wkb",
    "transmission",
    "incident_energy",
    "quantum",
    "admission",
})
"""Fragmentos que evidencian activación de P₁ (QuantumAdmissionGate)."""

_HILBERT_COLLAPSE_KEYS: FrozenSet[str] = frozenset({
    "collapse_hash",
    "quantum_momentum",
})
"""Claves que evidencian colapso observacional de Hilbert (requeridas para P₂)."""

_VALID_EIGENSTATES: FrozenSet[str] = frozenset({"ADMITTED", "REJECTED"})
"""Eigenstates válidos del observable de admisión."""


# ═════════════════════════════════════════════════════════════════════════════
# TIPOS ALGEBRAICOS
# ═════════════════════════════════════════════════════════════════════════════

class OperatorRole(Enum):
    """Rol semántico de un operador dentro del registro MIC."""
    QUANTUM_GATE = auto()
    HILBERT_AGENT = auto()


@dataclass(frozen=True)
class RegisteredOperators:
    """
    Par de operadores resueltos desde el registro MIC.

    Invariantes:
      1. quantum_gate ≠ hilbert_agent  (distinción funcional)
      2. Ambos son Callable (tipabilidad)
      3. Ambos preservan CategoricalState (contrato de tipo)
    """
    quantum_gate: Callable[[CategoricalState], CategoricalState]
    hilbert_agent: Callable[[CategoricalState], CategoricalState]

    def __post_init__(self) -> None:
        if self.quantum_gate is self.hilbert_agent:
            raise ValueError(
                "Violación de distinción funcional: "
                "quantum_gate y hilbert_agent son el mismo objeto."
            )
        if not callable(self.quantum_gate):
            raise TypeError(
                f"quantum_gate debe ser Callable, obtenido {type(self.quantum_gate)}"
            )
        if not callable(self.hilbert_agent):
            raise TypeError(
                f"hilbert_agent debe ser Callable, obtenido {type(self.hilbert_agent)}"
            )


@dataclass(frozen=True)
class ObservationalDecomposition:
    """
    Resultado de la descomposición observacional de un estado.

    Contiene el vector de proyección y componentes individuales para diagnóstico.

    Invariantes:
      • vector ∈ ℝ²
      • component_e1, component_e2 ∈ {0, 1}  (binarios)
      • component_e1 · component_e2 = 0      (ortogonalidad puntual)
      • component_e1 + component_e2 ∈ {0, 1} (resolución de identidad)
    """
    vector: np.ndarray
    component_e1: float
    component_e2: float

    def __post_init__(self) -> None:
        # Convertir a dtype estricto
        object.__setattr__(
            self, 'vector',
            np.asarray(self.vector, dtype=np.float64),
        )
        _assert_vector_geometry(self.vector)
        
        # Validación cruzada: componentes coinciden con vector
        if len(self.vector) >= 1:
            assert abs(self.component_e1 - self.vector[0]) <= _NUMERICAL_TOLERANCE, (
                f"Componente e1 inconsistente: component_e1={self.component_e1}, "
                f"vector[0]={self.vector[0]}"
            )
        if len(self.vector) >= 2:
            assert abs(self.component_e2 - self.vector[1]) <= _NUMERICAL_TOLERANCE, (
                f"Componente e2 inconsistente: component_e2={self.component_e2}, "
                f"vector[1]={self.vector[1]}"
            )


@dataclass(frozen=True)
class MockConfigurationResult:
    """Resultado de la configuración de un mock con trazabilidad completa."""
    path: str
    success: bool
    target_value: Any
    error_message: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE INTROSPECCIÓN SEGURA (DEFENSIVAS)
# ═════════════════════════════════════════════════════════════════════════════

def _safe_context(state: CategoricalState) -> Mapping[str, Any]:
    """
    Extrae el contexto de un CategoricalState de forma defensiva.

    Implementa defensa en profundidad:
      1. Verifica existencia del atributo
      2. Verifica que sea un Mapping válido
      3. Retorna mapeo vacío como fallback

    Args:
        state: CategoricalState a introspeccionar

    Returns:
        Mapping[str, Any]: contexto del estado o {} si no está disponible

    Raises:
        Ninguna — función defensiva que nunca falla
    """
    if not isinstance(state, CategoricalState):
        return {}
    
    ctx = getattr(state, "context", None)
    if ctx is None:
        return {}
    if not isinstance(ctx, Mapping):
        warnings.warn(
            f"Atributo 'context' no es un Mapping: {type(ctx)}. "
            f"Se retorna contexto vacío.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    return ctx


def _has_any_key_fragment(
    ctx: Mapping[str, Any],
    fragments: FrozenSet[str],
) -> bool:
    """
    Verifica si alguna clave del contexto contiene algún fragmento.

    Busca coincidencias case-insensitive de los fragmentos dentro
    de las claves del contexto.

    Complejidad: O(|keys| × |fragments| × L) donde L es longitud
    promedio de las claves.

    Args:
        ctx: Contexto a inspeccionar
        fragments: Conjunto de fragmentos a buscar

    Returns:
        bool: True si ∃ clave k tal que ∃ fragmento f: f ⊆ k (case-insensitive)
    """
    if not ctx or not fragments:
        return False

    lowered_keys: List[str] = [str(k).lower() for k in ctx.keys()]
    lowered_fragments: List[str] = [f.lower() for f in fragments]

    return any(
        fragment in key
        for key in lowered_keys
        for fragment in lowered_fragments
    )


def _measurement_dict(ctx: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """
    Extrae el diccionario de medición cuántica del contexto.

    Verifica que la entrada sea un Mapping válido (no solo que exista).

    Args:
        ctx: Contexto a inspeccionar

    Returns:
        Mapping[str, Any] si existe y es válido, None en caso contrario
    """
    if not isinstance(ctx, Mapping):
        return None
    
    measurement = ctx.get("quantum_measurement")
    if measurement is None:
        return None
    if not isinstance(measurement, Mapping):
        warnings.warn(
            f"'quantum_measurement' no es un Mapping: {type(measurement)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return measurement


# ═════════════════════════════════════════════════════════════════════════════
# PROYECTORES OBSERVACIONALES
#
# Axiomas de los proyectores P_i: CategoricalState → {0, 1}
#
# (i)   Binaridad:          P_i(s) ∈ {0, 1}           ∀s
# (ii)  Ortogonalidad:      P_i(s) · P_j(s) = 0      ∀s, i≠j
# (iii) Idempotencia:       P_i² = P_i                (trivial para {0,1})
# (iv)  Resolución:         P_1(s) + P_2(s) ∈ {0,1}  ∀s
# (v)   Monotonía veto:     veto(s) ⟹ veto(P_i(s))
#
# ═════════════════════════════════════════════════════════════════════════════

def _extract_basis_e1(state: CategoricalState) -> float:
    """
    Proyector observacional sobre e₁ (QuantumAdmissionGate).

    Semántica:
    ──────────
    e₁ detecta huellas exclusivas de admisión cuántica:
      • WKB coefficient
      • Transmission amplitude
      • Incident energy
      • Fragmentos "quantum" o "admission" en claves

    Axioma de exclusividad (FUNDAMENTAL):
      Si el estado porta una medición de Hilbert (quantum_measurement),
      entonces P₁ se anula para preservar ortogonalidad con P₂.

      ∃ quantum_measurement ⟹ P₁(s) = 0

    Esta regla garantiza que:
      • P₁ es activo en la fase de ADMISIÓN (pre-medición)
      • P₂ es activo en la fase de OBSERVACIÓN (post-medición)
      • Nunca coexisten (ortogonalidad puntual)

    Formalización:
    ──────────────
    P₁(s) := {
      0,   si ∃ quantum_measurement en context(s)
      1,   si ∃ fragmento cuántico en context(s)
      0,   en caso contrario
    }

    Returns:
        float: 1.0 si se detecta traza cuántica exclusiva, 0.0 en caso contrario.
    """
    ctx = _safe_context(state)

    # AXIOMA DE EXCLUSIVIDAD: medición Hilbert anula e₁
    if "quantum_measurement" in ctx:
        return 0.0

    # Detectar traza cuántica
    has_quantum_trace = _has_any_key_fragment(ctx, _QUANTUM_GATE_TRACE_FRAGMENTS)
    return 1.0 if has_quantum_trace else 0.0


def _extract_basis_e2(state: CategoricalState) -> float:
    """
    Proyector observacional sobre e₂ (HilbertObserverAgent).

    Semántica:
    ──────────
    e₂ requiere la conjunción de TRES condiciones para activarse:

    (Condición 1) Existencia de medición cuántica post-observación:
      quantum_measurement ∈ context(s)

    (Condición 2) Eigenstate válido ∈ {ADMITTED, REJECTED}:
      quantum_measurement.eigenstate ∈ _VALID_EIGENSTATES

    (Condición 3) Traza de colapso de Hilbert:
      ∃ k ∈ (collapse_hash ∪ quantum_momentum) :
        k ∈ context(s) ∨ k ∈ quantum_measurement

    La conjunción garantiza que e₂ solo se activa tras un colapso
    observacional COMPLETO, nunca durante admisión (dominio de e₁).

    Formalización:
    ──────────────
    P₂(s) := {
      1,   si (Cond1) ∧ (Cond2) ∧ (Cond3)
      0,   en caso contrario
    }

    Justificación física:
      • Cond1: garantiza que se realizó una medición
      • Cond2: garantiza que el resultado es válido (admitido/rechazado)
      • Cond3: garantiza que el colapso dejó traza (no es medición simulada)

    Returns:
        float: 1.0 si las tres condiciones se satisfacen, 0.0 en caso contrario.
    """
    ctx = _safe_context(state)
    measurement = _measurement_dict(ctx)

    # Condición 1: existencia de medición
    if measurement is None:
        return 0.0

    # Condición 2: eigenstate válido
    eigenstate = measurement.get("eigenstate")
    has_valid_eigenstate = eigenstate in _VALID_EIGENSTATES
    if not has_valid_eigenstate:
        return 0.0

    # Condición 3: traza de colapso en contexto O en medición
    has_collapse_in_ctx = bool(
        _HILBERT_COLLAPSE_KEYS & set(ctx.keys())
    )
    has_collapse_in_measurement = bool(
        _HILBERT_COLLAPSE_KEYS & set(measurement.keys())
    )
    has_collapse_trace = has_collapse_in_ctx or has_collapse_in_measurement

    return 1.0 if has_collapse_trace else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN Y VALIDACIÓN VECTORIAL
# ═════════════════════════════════════════════════════════════════════════════

def _assert_vector_geometry(v: np.ndarray) -> None:
    """
    Valida las propiedades geométricas de un vector observacional.

    Verifica todos los invariantes del espacio observacional V = ℝ²:

    (1) Forma: v ∈ ℝ^d con d = _SPACE_DIMENSION
        Asegura que el vector vive en el espacio correcto.

    (2) Finitud: ∀ v_i ∈ ℝ (no NaN, no ±∞)
        Detecta errores numéricos graves.

    (3) Binaridad: ∀ v_i ∈ {0.0, 1.0}
        Los proyectores observacionales DEBEN retornar valores binarios.

    (4) Exclusividad: ‖v‖₁ ≤ 1 (a lo sumo un eje activo)
        Consecuencia de la ortogonalidad puntual P₁·P₂ = 0.

    La validación es estricta (tolerancia 1e-12) porque hablamos de
    bases canónicas, donde no hay perdón para valores "casi" correctos.

    Args:
        v: np.ndarray a validar

    Raises:
        AssertionError: si alguna propiedad se viola
    """
    # (1) Verificar forma
    assert v.shape == (_SPACE_DIMENSION,), (
        f"Forma inválida: esperada ({_SPACE_DIMENSION},), obtenida {v.shape}. "
        f"El vector debe vivir en ℝ^{_SPACE_DIMENSION}."
    )

    # (2) Verificar finitud
    assert np.all(np.isfinite(v)), (
        f"Componentes no finitas detectadas: {v}. "
        f"Vector contiene NaN o ±∞ (error numérico grave)."
    )

    # (3) Verificar binaridad
    assert np.all(np.isin(v, [0.0, 1.0])), (
        f"Vector no binario: {v}. "
        f"Los proyectores observacionales deben producir valores en {{0, 1}} exactamente."
    )

    # (4) Verificar exclusividad (norma L₁)
    l1_norm = float(np.sum(np.abs(v)))
    assert l1_norm <= 1.0 + _NUMERICAL_TOLERANCE, (
        f"Violación de exclusividad observacional: ‖v‖₁ = {l1_norm}. "
        f"Vector: {v}. "
        f"A lo sumo un proyector puede activarse simultáneamente. "
        f"Esto indica violación de ortogonalidad puntual P₁·P₂ = 0."
    )


def _extract_state_vector(state: CategoricalState) -> ObservationalDecomposition:
    """
    Construye la descomposición observacional completa de un estado.

    Evalúa ambos proyectores sobre el estado y ensambla el vector
    canónico correspondiente en {(0,0), (1,0), (0,1)}.

    Args:
        state: CategoricalState a descomponer

    Returns:
        ObservationalDecomposition con validación post-construcción
    """
    c1 = _extract_basis_e1(state)
    c2 = _extract_basis_e2(state)

    vector = np.array([c1, c2], dtype=np.float64)

    return ObservationalDecomposition(
        vector=vector,
        component_e1=c1,
        component_e2=c2,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE OPERADORES DESDE EL REGISTRO MIC
# ═════════════════════════════════════════════════════════════════════════════

_OPERATOR_RESOLUTION_RULES: Dict[OperatorRole, Tuple[str, ...]] = {
    OperatorRole.QUANTUM_GATE: ("quantum", "gate"),
    OperatorRole.HILBERT_AGENT: ("hilbert", "agent"),
}
"""
Reglas de matching para resolución de operadores.

Semántica: un nombre de registro coincide con un rol si TODAS
los fragmentos de la tupla aparecen (case-insensitive) en el nombre.

Ejemplo:
  • Nombre "QuantumAdmissionGate" → fragmentos ("quantum", "gate")
    → coincide con OperatorRole.QUANTUM_GATE ✓
"""


def _matches_role(name: str, required_fragments: Tuple[str, ...]) -> bool:
    """
    Verifica que un nombre de registro contenga TODOS los fragmentos.

    Implementa una búsqueda CONJUNTIVA (not OR) para reducir
    falsos positivos. Por ejemplo:
      • "quantum" solo → no coincide (falta "gate")
      • "quantum_gate" → coincide (tiene ambos)
      • "QuantumAdmissionGate" → coincide (case-insensitive)

    Complejidad: O(L × |fragments|) donde L = len(name).

    Args:
        name: Nombre a verificar
        required_fragments: Fragmentos requeridos (ALL must match)

    Returns:
        bool: True si todos los fragmentos están presentes
    """
    lowered = name.lower()
    return all(fragment in lowered for fragment in required_fragments)


def _resolve_registered_operators(registry: MICRegistry) -> RegisteredOperators:
    """
    Resuelve los dos operadores registrados del MICRegistry.

    Algoritmo de resolución:
    ────────────────────────
    1. Accede al diccionario interno `_vectors` (reflejo de estado registrado)
    2. Para cada entrada, extrae el handler:
       - Si entry = (Stratum, handler): usa handler
       - Si entry = handler: usa entry directamente
    3. Clasifica cada handler por coincidencia de fragmentos
    4. Verifica unicidad: exactamente 1 operador por rol
    5. Verifica distinción: quantum_gate ≠ hilbert_agent

    Invariantes que GARANTIZA:
    ──────────────────────────
    I1. Existencia:      ∃ operador para quantum_gate
    I2. Existencia:      ∃ operador para hilbert_agent
    I3. Unicidad:        1 operador por rol (no ambigüedad)
    I4. Distinción:      quantum_gate ≠ hilbert_agent (verificada en RegisteredOperators)

    Args:
        registry: MICRegistry a inspeccionar

    Returns:
        RegisteredOperators con ambos operadores resueltos

    Raises:
        AssertionError: si algún invariante se viola
        AttributeError: si registry no expone `_vectors`
    """
    vectors = getattr(registry, "_vectors", None)
    assert vectors is not None and isinstance(vectors, Mapping), (
        "MICRegistry debe exponer un atributo `_vectors` tipo Mapping. "
        f"Obtenido: {type(vectors)}. "
        "Verifique la implementación de MICRegistry.register_vector()."
    )

    # Clasificar operadores por rol
    resolved: Dict[OperatorRole, List[Tuple[str, Any]]] = {
        role: [] for role in OperatorRole
    }

    for name, entry in vectors.items():
        # Extraer handler (soporta tanto tuplas como valores directos)
        if isinstance(entry, tuple) and len(entry) >= 2:
            handler = entry[1]  # (Stratum, handler)
        else:
            handler = entry

        # Clasificar por coincidencia de fragmentos
        for role, fragments in _OPERATOR_RESOLUTION_RULES.items():
            if _matches_role(str(name), fragments):
                resolved[role].append((str(name), handler))

    # Verificar unicidad y existencia para cada rol
    for role, matches in resolved.items():
        assert len(matches) >= 1, (
            f"INVARIANTE I1/I2 VIOLADO: No se encontró operador para rol {role.name}. "
            f"Nombres registrados: {list(vectors.keys())}. "
            f"Fragmentos requeridos: {_OPERATOR_RESOLUTION_RULES[role]}. "
            f"Registre al menos un operador cuyo nombre contenga todos los fragmentos."
        )
        assert len(matches) == 1, (
            f"INVARIANTE I3 VIOLADO: Ambigüedad en rol {role.name}. "
            f"Se encontraron {len(matches)} operadores coincidentes: "
            f"{[m[0] for m in matches]}. "
            f"Cada rol debe tener EXACTAMENTE un operador registrado."
        )

    quantum_gate = resolved[OperatorRole.QUANTUM_GATE][0][1]
    hilbert_agent = resolved[OperatorRole.HILBERT_AGENT][0][1]

    # I4: Distinción verificada en RegisteredOperators.__post_init__()
    return RegisteredOperators(
        quantum_gate=quantum_gate,
        hilbert_agent=hilbert_agent,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE ENTORNO MOCK (EVITAR VETO NO-INTENCIONAL)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _MockPathSpec:
    """Especificación de una ruta de mock singleton."""
    path: str
    value: Any
    subsystem: str  # Para auditoría y diagnóstico


_STABILITY_MOCK_SPECS: Tuple[_MockPathSpec, ...] = (
    # ─────────────────────────────────────────────────────────
    # Subsistema: Oráculo de Laplace
    # ─────────────────────────────────────────────────────────
    # Axioma: Si el polo dominante tiene parte real < 0, el sistema es estable.
    # Mockeamos para retornar -1.0 (estable con margen de seguridad).
    _MockPathSpec(
        "_laplace_oracle.get_dominant_pole_real",
        -1.0,
        subsystem="laplace",
    ),
    _MockPathSpec(
        "_laplace.get_dominant_pole_real",
        -1.0,
        subsystem="laplace",
    ),
    # ─────────────────────────────────────────────────────────
    # Subsistema: Orquestador de Sheaf
    # ─────────────────────────────────────────────────────────
    # Axioma: Si la energía de frustración es 0, no hay conflicto topológico.
    # Mockeamos para retornar 0.0 (sin frustración).
    _MockPathSpec(
        "_sheaf_orchestrator.get_global_frustration_energy",
        0.0,
        subsystem="sheaf",
    ),
    _MockPathSpec(
        "_sheaf.compute_frustration_energy",
        0.0,
        subsystem="sheaf",
    ),
    # ─────────────────────────────────────────────────────────
    # Subsistema: Vigilante Topológico
    # ─────────────────────────────────────────────────────────
    # Axioma: Si la amenaza Mahalanobis es 0, no hay anomalía topológica.
    # Mockeamos para retornar 0.0 (sin amenaza).
    _MockPathSpec(
        "_topo_watcher.get_mahalanobis_threat",
        0.0,
        subsystem="topo",
    ),
    _MockPathSpec(
        "_topo.get_mahalanobis_threat",
        0.0,
        subsystem="topo",
    ),
)


def _set_nested_return_value(
    obj: Any,
    chain: str,
    value: Any,
) -> MockConfigurationResult:
    """
    Configura return_value siguiendo una cadena de atributos.

    Navega obj.a.b...z.return_value = value de forma defensiva.

    Estrategia de errores:
      • AttributeError → retorna resultado con success=False
      • Otros errores → propaga (son fallos inesperados)

    Args:
        obj: Objeto raíz (típicamente un Mock)
        chain: Cadena de atributos (e.g., "_topo_watcher.get_mahalanobis_threat")
        value: Valor a asignar

    Returns:
        MockConfigurationResult con trazabilidad completa
    """
    current = obj
    parts = chain.split(".")
    try:
        for part in parts:
            current = getattr(current, part)
        current.return_value = value
        return MockConfigurationResult(
            path=chain,
            success=True,
            target_value=value,
            error_message=None,
        )
    except AttributeError as e:
        return MockConfigurationResult(
            path=chain,
            success=False,
            target_value=value,
            error_message=f"AttributeError: {str(e)}",
        )


def _configure_stable_non_veto_environment(operator: Any) -> List[MockConfigurationResult]:
    """
    Configura mocks para evitar veto por inestabilidad/frustración/amenaza.

    Procedimiento:
    ──────────────
    1. Itera sobre todas las rutas de mock candidatas
    2. Para cada ruta, intenta configurar su return_value
    3. Recolecta resultados en una lista para auditoría
    4. Verifica cobertura por subsistema:
       - Si ninguna ruta de un subsistema funciona, emite advertencia
    5. Retorna lista completa para diagn´ostico en el test

    Justificación:
    ──────────────
    Los operadores pueden depender de oráculos (Laplace, Sheaf, Topo).
    Si no están mockeados, pueden retornar valores que causen veto.
    Este procedimiento neutraliza esos vetos no-intencionales.

    Args:
        operator: Operador a configurar (típicamente gate o agent)

    Returns:
        Lista de MockConfigurationResult con todos los intentos
    """
    results: List[MockConfigurationResult] = []

    for spec in _STABILITY_MOCK_SPECS:
        result = _set_nested_return_value(operator, spec.path, spec.value)
        results.append(result)

    # Verificar cobertura por subsistema
    subsystems: Dict[str, List[str]] = {
        "laplace": ["_laplace_oracle", "_laplace"],
        "sheaf": ["_sheaf_orchestrator", "_sheaf"],
        "topo": ["_topo_watcher", "_topo"],
    }

    for subsystem_name, prefixes in subsystems.items():
        subsystem_results = [
            r for r in results
            if any(r.path.startswith(prefix) for prefix in prefixes)
        ]
        any_success = any(r.success for r in subsystem_results)
        if not any_success:
            warnings.warn(
                f"[CONFIGURACIÓN MOCK] Subsistema '{subsystem_name}': "
                f"ninguna ruta de mock fue efectiva para el operador {type(operator).__name__}. "
                f"Rutas intentadas: {[r.path for r in subsystem_results]}. "
                f"El test podría producir un veto espurio. "
                f"Detalles: {[r.error_message for r in subsystem_results if not r.success]}",
                RuntimeWarning,
                stacklevel=3,
            )

    return results


# ═════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN DE IMAGEN OPERATORIAL
# ═════════════════════════════════════════════════════════════════════════════

def _operator_image(
    operator: Callable[[CategoricalState], CategoricalState],
    state: CategoricalState,
) -> Tuple[CategoricalState, ObservationalDecomposition]:
    """
    Evalúa un operador MIC y extrae su imagen observacional.

    Proceso:
    ────────
    1. Evalúa out_state = operator(state)
    2. Verifica que out_state ∈ CategoricalState
    3. Extrae la descomposición observacional
    4. Valida que la descomposición satisface restricciones geométricas

    Contrato:
    ─────────
    • operator debe ser una función pura (no modifica state)
    • operator debe retornar un CategoricalState válido
    • La imagen debe ser un vector en el retículo observacional

    Args:
        operator: Función MIC a evaluar
        state: Estado inicial

    Returns:
        Tupla (estado_resultante, descomposición_observacional)

    Raises:
        AssertionError: si el resultado viola restricciones
        TypeError: si operator no retorna CategoricalState
    """
    out_state = operator(state)

    assert isinstance(out_state, CategoricalState), (
        f"El operador {getattr(operator, '__name__', repr(operator))} "
        f"retornó {type(out_state).__name__} en lugar de CategoricalState. "
        f"Violación de contrato de tipo."
    )

    decomposition = _extract_state_vector(out_state)

    return out_state, decomposition


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE ÁLGEBRA LINEAL RIGUROSA
# ═════════════════════════════════════════════════════════════════════════════

def _build_transformation_matrix(
    *vectors: np.ndarray,
) -> np.ndarray:
    """
    Construye la matriz de transformación T = [v₁ | v₂ | ... | vₙ].

    Cada vector v_i se interpreta como COLUMNA de T.

    Validaciones:
      • Al menos un vector
      • Todos los vectores en el mismo espacio (misma dimensión)

    Args:
        *vectors: Vectores a ensamblar como columnas

    Returns:
        np.ndarray: Matriz T ∈ ℝ^{d×n} donde d = dim(v_i), n = len(vectors)

    Raises:
        AssertionError: si validaciones fallan
    """
    assert len(vectors) >= 1, "Se requiere al menos un vector para construir T."

    dim = vectors[0].shape[0]
    for i, v in enumerate(vectors):
        assert v.shape == (dim,), (
            f"Vector {i} tiene forma {v.shape}, esperada ({dim},). "
            f"Todos los vectores deben pertenecer al mismo espacio ℝ^{dim}."
        )

    T = np.column_stack(vectors)
    expected_shape = (dim, len(vectors))
    assert T.shape == expected_shape, (
        f"Forma de T: {T.shape}, esperada: {expected_shape}"
    )
    return T


def _compute_gram_matrix(T: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de Gram G = T^T T.

    La matriz de Gram codifica el producto interno entre columnas:
      G[i,j] = ⟨col_i(T), col_j(T)⟩

    Para vectores ortonormales:
      G = I  (matriz identidad)

    Para vectores binarios canónicos {(0,0), (1,0), (0,1)}:
      G ∈ {0, 1}^{n×n} diagonal

    Args:
        T: Matriz de transformación

    Returns:
        np.ndarray: Matriz G = T^T T

    Raises:
        AssertionError: si G no es simétrica (error numérico grave)
    """
    G = T.T @ T
    
    # Verificar simetría (debe ser exacta para matrices reales)
    assert np.allclose(G, G.T, atol=_NUMERICAL_TOLERANCE), (
        f"Matriz de Gram NO SIMÉTRICA (error numérico grave):\n{G}\n"
        f"Diferencia: {G - G.T}"
    )
    return G


def _assert_orthogonality_from_gram(G: np.ndarray, label: str = "") -> None:
    """
    Verifica que la matriz de Gram sea diagonal (ortogonalidad).

    Para vectores ortonormales canónicos:
      • G debe ser diagonal
      • Entradas diagonales = 1 (si norma 1)
      • Entradas fuera diagonal = 0 (ortogonalidad)

    Para vectores binarios:
      • Entradas diagonales ∈ {0, 1}
      • Entradas fuera diagonal = 0

    Args:
        G: Matriz de Gram
        label: Etiqueta para mensajes de error

    Raises:
        AssertionError: si off-diagonal es no-nulo
    """
    off_diagonal = G - np.diag(np.diag(G))
    prefix = f"[{label}] " if label else ""

    max_off_diag = np.max(np.abs(off_diagonal))
    assert np.allclose(off_diagonal, 0.0, atol=_ORTHOGONALITY_TOLERANCE), (
        f"{prefix}Violación de ortogonalidad cruzada. "
        f"Elementos fuera de diagonal no nulos (max = {max_off_diag}):\n"
        f"Gram =\n{G}\n"
        f"Off-diagonal =\n{off_diagonal}"
    )


def _assert_full_rank(T: np.ndarray, label: str = "") -> int:
    """
    Verifica que T tenga rango completo (columnas linealmente independientes).

    Para T ∈ ℝ^{d×n}:
      • rank(T) = n  (columnas l.i.)
      • nullity(T) = 0  (núcleo trivial)

    Args:
        T: Matriz a verificar
        label: Etiqueta para mensajes de error

    Returns:
        int: Rango computado (debe ser igual a #columnas)

    Raises:
        AssertionError: si rank < #columnas
    """
    n_cols = T.shape[1]
    rank = int(np.linalg.matrix_rank(T, tol=_RANK_TOLERANCE))
    nullity = n_cols - rank
    prefix = f"[{label}] " if label else ""

    assert rank == n_cols, (
        f"{prefix}Deficiencia de rango: rank(T) = {rank}, "
        f"esperado {n_cols} (columnas). "
        f"Nullity(T) = {nullity} (núcleo no trivial).\n"
        f"Dimensión: {T.shape[0]}×{T.shape[1]}\n"
        f"T =\n{T}"
    )
    return rank


def _assert_nonsingular(T: np.ndarray, label: str = "") -> float:
    """
    Verifica que una matriz cuadrada sea no singular (det ≠ 0).

    Para T ∈ ℝ^{n×n}:
      • det(T) ≠ 0  (no singular)
      • Existe T^{-1}

    Args:
        T: Matriz cuadrada a verificar
        label: Etiqueta para mensajes de error

    Returns:
        float: Valor del determinante

    Raises:
        AssertionError: si matrix es singular o no cuadrada
    """
    assert T.shape[0] == T.shape[1], (
        f"La verificación de singularidad requiere matriz cuadrada. "
        f"Forma: {T.shape}"
    )
    det = float(np.linalg.det(T))
    prefix = f"[{label}] " if label else ""

    assert not np.isclose(det, 0.0, atol=_NUMERICAL_TOLERANCE), (
        f"{prefix}Matriz singular (det ≈ 0): det(T) = {det}.\n"
        f"T =\n{T}"
    )
    return det


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICADORES DE PROPIEDADES PROYECTIVAS
# ═════════════════════════════════════════════════════════════════════════════

def _assert_projector_idempotence(
    extractor: Callable[[CategoricalState], float],
    state: CategoricalState,
    label: str = "",
) -> None:
    """
    Verifica la idempotencia del proyector: P²(s) = P(s).

    Para proyectores binarios ({0, 1}), se cumple trivialmente:
      • 0² = 0  ✓
      • 1² = 1  ✓

    La verificación explícita protege contra regresiones donde
    el extractor pudiera retornar valores no binarios.

    Args:
        extractor: Función proyector P: CategoricalState → float
        state: Estado a evaluar
        label: Etiqueta para diagnóstico

    Raises:
        AssertionError: si |P(s) - P²(s)| > tolerancia
    """
    p_val = extractor(state)
    p_squared = p_val * p_val
    prefix = f"[{label}] " if label else ""

    assert abs(p_val - p_squared) <= _NUMERICAL_TOLERANCE, (
        f"{prefix}Violación de idempotencia: P(s) = {p_val}, "
        f"P²(s) = {p_squared}. "
        f"Diferencia: {abs(p_val - p_squared)}. "
        f"Los proyectores deben satisfacer P² = P."
    )


def _assert_pointwise_orthogonality(
    state: CategoricalState,
    label: str = "",
) -> None:
    """
    Verifica ortogonalidad puntual: P₁(s) · P₂(s) = 0.

    Esta es una condición más fuerte que ortogonalidad de vectores:
    exige que para CADA estado s, a lo sumo un proyector se active.

    Equivalentemente:
      • P₁(s) = 1  ⟹  P₂(s) = 0
      • P₂(s) = 1  ⟹  P₁(s) = 0

    Args:
        state: Estado a verificar
        label: Etiqueta para diagnóstico

    Raises:
        AssertionError: si ambos proyectores se activan simultáneamente
    """
    p1 = _extract_basis_e1(state)
    p2 = _extract_basis_e2(state)
    product = p1 * p2
    prefix = f"[{label}] " if label else ""

    assert abs(product) <= _NUMERICAL_TOLERANCE, (
        f"{prefix}Violación de ortogonalidad puntual: "
        f"P₁(s)·P₂(s) = {p1}·{p2} = {product} ≠ 0. "
        f"Ambos proyectores se activaron simultáneamente. "
        f"Esto viola la descomposición V = Im(P₁) ⊕ Im(P₂)."
    )


def _assert_resolution_of_identity(
    state: CategoricalState,
    label: str = "",
    *,
    allow_null_state: bool = True,
) -> None:
    """
    Verifica la resolución de la identidad: P₁(s) + P₂(s) = 1.

    En un espacio observacional V = ℝ² descompuesto en subespacios,
    todo estado debe proyectarse en exactamente uno:

    Caso 1 (normal): P₁(s) + P₂(s) = 1
      El estado se observa en uno de los dos subespacios.

    Caso 2 (nulo): P₁(s) + P₂(s) = 0
      El estado no pertenece a ningún subespacio observado
      (estado "invisible" para ambos detectores).

    El caso nulo es legítimo para estados intermedios que no
    son atraídos por ningún operador MIC.

    Args:
        state: Estado a verificar
        label: Etiqueta para diagnóstico
        allow_null_state: Si True, acepta también suma = 0

    Raises:
        AssertionError: si suma no está en {0, 1}
    """
    p1 = _extract_basis_e1(state)
    p2 = _extract_basis_e2(state)
    total = p1 + p2
    prefix = f"[{label}] " if label else ""

    valid_totals: Set[float] = {1.0}
    if allow_null_state:
        valid_totals.add(0.0)

    is_valid = any(
        abs(total - valid) <= _NUMERICAL_TOLERANCE
        for valid in valid_totals
    )

    assert is_valid, (
        f"{prefix}Violación de resolución de identidad: "
        f"P₁(s) + P₂(s) = {p1} + {p2} = {total}. "
        f"Valores aceptados: {valid_totals}. "
        f"La suma de proyectores debe ser 0 o 1."
    )


# ═════════════════════════════════════════════════════════════════════════════
# ESTADOS DE PRUEBA (GENERADORES)
# ═════════════════════════════════════════════════════════════════════════════

def _make_quantum_state() -> CategoricalState:
    """
    Genera un estado que debe activar SOLO e₁.

    Características:
      • Contiene trazas de admisión cuántica (WKB, transmisión, energía)
      • NO contiene quantum_measurement (no post-medición)
      • Esperado: P₁(s) = 1, P₂(s) = 0
    """
    return CategoricalState(payload={
        "incident_energy": 5.0,
        "wkb_coefficient": 0.85,
        "transmission_amplitude": 0.7,
    })


def _make_hilbert_state() -> CategoricalState:
    """
    Genera un estado que debe activar SOLO e₂.

    Características:
      • Contiene medición post-colapso (quantum_measurement)
      • Eigenstate válido (ADMITTED)
      • Traza de colapso (collapse_hash)
      • Esperado: P₁(s) = 0, P₂(s) = 1
    """
    return CategoricalState(payload={
        "quantum_measurement": {
            "eigenstate": "ADMITTED",
            "collapse_hash": "abc123",
        },
    })


def _make_null_state() -> CategoricalState:
    """
    Genera un estado sin trazas observacionales.

    Características:
      • Payload vacío o sin claves relevantes
      • Esperado: P₁(s) = 0, P₂(s) = 0, suma = 0
    """
    return CategoricalState(payload={})


def _make_frustrated_state() -> CategoricalState:
    """
    Genera un estado diseñado para provocar veto.

    Características:
      • Estabilidad muy baja
      • Alta varianza de entropía (indicador de caos)
      • Los oráculos (Laplace, Sheaf) deberían emitir veto
      • Esperado: quantum_error en contexto
    """
    return CategoricalState(payload={
        "structural_stability": 0.0,
        "entropy_variance": 9999.0,
    })


def _make_robust_state() -> CategoricalState:
    """
    Genera un estado con payload robusto para tests de rango completo.

    Características:
      • Payload grande pero sin claves "especiales"
      • Propicia que ambos operadores generen trazas
      • Esperado: rango(T) = 2 con alta probabilidad
    """
    return CategoricalState(payload={
        "data_size": "A" * 1_000_000,
    })


# ═════════════════════════════════════════════════════════════════════════════
# TEST SUITE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.stress
class TestMICQuantumHilbertIntegration:
    """
    Valida la coherencia geométrica de la integración Quantum/Hilbert en la MIC.

    Estructura de verificaciones (7 propiedades algebraicas):
    ─────────────────────────────────────────────────────────

    A. test_functional_orthogonality:
       ⟨e₁, e₂⟩ = 0  (producto interno nulo)
       Garantiza que los operadores miden propiedades independientes.

    B. test_full_rank_and_zero_nullity:
       rank(T) = 2, nullity = 0, det(T) ≠ 0
       Garantiza que los vectores son linealmente independientes.

    C. test_projector_idempotence:
       P_i² = P_i  ∀i  (propiedad proyectiva)
       Valida que los proyectores son idempotentes.

    D. test_pointwise_orthogonality_across_states:
       P₁(s) · P₂(s) = 0  ∀s ∈ S_test  (ortogonalidad puntual)
       Garantiza descomposición limpia del espacio.

    E. test_resolution_of_identity:
       P₁(s) + P₂(s) ∈ {0, 1}  ∀s  (completitud)
       Verifica que el marco es completo.

    F. test_preservation_of_dikw_reticular_functor:
       veto(f(s)) ⟹ veto(g∘f(s))  (monotonía)
       Modela la categoría DIKW como preorden.

    G. test_dikw_composition_functoriality:
       F(g∘f) = F(g) ∘ F(f)  (composición)
       Verifica que la composición preserva estructura.

    H. test_inner_product_symmetry:
       ⟨v₁, v₂⟩ = ⟨v₂, v₁⟩  (simetría)
       Valida el contrato del producto interno.

    Nivel de rigor:
      • Tolerancias: 1e-12 (máximo rigor)
      • Cobertura: todos los invariantes algebraicos
      • Casos: representativos + adversariales
    """

    # ─────────────────────────────────────────────────────────────────────
    # FIXTURES
    # ─────────────────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_dependencies(
        self,
    ) -> Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        """Crea las tres dependencias mockeadas."""
        topo_watcher = mock.MagicMock(name="topo_watcher")
        laplace_oracle = mock.MagicMock(name="laplace_oracle")
        sheaf_orchestrator = mock.MagicMock(name="sheaf_orchestrator")
        return topo_watcher, laplace_oracle, sheaf_orchestrator

    @pytest.fixture
    def mic_registry(
        self,
        mock_dependencies: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    ) -> MICRegistry:
        """
        Construye un MICRegistry con ambos operadores registrados.

        INVARIANTE: El orden de registro no afecta la resolución
        de operadores (independencia de orden).

        Registra:
          1. QuantumAdmissionGate como "quantum_gate"
          2. HilbertObserverAgent como "hilbert_agent"
        """
        registry = MICRegistry()
        topo_watcher, laplace_oracle, sheaf_orchestrator = mock_dependencies

        gate = QuantumAdmissionGate(
            topo_watcher=topo_watcher,
            laplace_oracle=laplace_oracle,
            sheaf_orchestrator=sheaf_orchestrator,
        )
        registry.register_vector("quantum_gate", Stratum.PHYSICS, gate)

        agent = HilbertObserverAgent(
            topo_watcher=topo_watcher,
            laplace_oracle=laplace_oracle,
            sheaf_orchestrator=sheaf_orchestrator,
        )
        registry.register_vector("hilbert_agent", Stratum.STRATEGY, agent)

        return registry

    @pytest.fixture
    def operators(self, mic_registry: MICRegistry) -> RegisteredOperators:
        """Resuelve y retorna los operadores registrados."""
        return _resolve_registered_operators(mic_registry)

    # ─────────────────────────────────────────────────────────────────────
    # A. ORTOGONALIDAD FUNCIONAL
    # ─────────────────────────────────────────────────────────────────────

    def test_functional_orthogonality(
        self,
        operators: RegisteredOperators,
    ) -> None:
        """
        A. Ortogonalidad funcional: ⟨e₁, e₂⟩ = 0.

        Se verifica que las imágenes observacionales de ambos operadores,
        evaluadas sobre el estado nulo, pertenezcan a subespacios
        ortogonales en el sentido del producto interno de ℝ².

        Precisión requerida: 1e-12.

        Proceso:
        ────────
        1. Evalúa gate(∅) → descomposición v₁
        2. Evalúa agent(∅) → descomposición v₂
        3. Calcula ⟨v₁, v₂⟩ = v₁·v₂
        4. Verifica que ⟨v₁, v₂⟩ ≈ 0
        """
        zero_state = _make_null_state()

        _, decomp_1 = _operator_image(operators.quantum_gate, zero_state)
        _, decomp_2 = _operator_image(operators.hilbert_agent, zero_state)

        v1 = decomp_1.vector
        v2 = decomp_2.vector

        inner_product = float(np.dot(v1, v2))

        assert abs(inner_product) <= _ORTHOGONALITY_TOLERANCE, (
            f"Violación de ortogonalidad funcional: ⟨e₁, e₂⟩ = {inner_product}.\n"
            f"v₁ = {v1} "
            f"(e1={decomp_1.component_e1}, e2={decomp_1.component_e2})\n"
            f"v₂ = {v2} "
            f"(e1={decomp_2.component_e1}, e2={decomp_2.component_e2})\n"
            f"Esperado: ⟨v₁, v₂⟩ ≈ 0 (tolerancia {_ORTHOGONALITY_TOLERANCE})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # B. INDEPENDENCIA LINEAL / RANGO COMPLETO
    # ─────────────────────────────────────────────────────────────────────

    def test_full_rank_and_zero_nullity(
        self,
        operators: RegisteredOperators,
    ) -> None:
        """
        B. Independencia lineal y rango completo.

        Se construye T = [v₁ | v₂] desde los vectores observacionales
        y se verifican:
          1. rank(T) = 2                 (rango completo)
          2. nullity(T) = 0              (núcleo trivial)
          3. det(T) ≠ 0                  (no singularidad)
          4. G = T^T T diagonal          (ortogonalidad cruzada)

        Precondición:
        ─────────────
        Se configuran mocks para evitar veto por inestabilidad,
        asegurando que ambos operadores generen trazas observables.

        Precisión: 1e-12.
        """
        _configure_stable_non_veto_environment(operators.quantum_gate)
        _configure_stable_non_veto_environment(operators.hilbert_agent)

        base_state = _make_robust_state()

        state_gate, decomp_1 = _operator_image(
            operators.quantum_gate, base_state
        )
        state_hilbert, decomp_2 = _operator_image(
            operators.hilbert_agent, base_state
        )

        v1 = decomp_1.vector
        v2 = decomp_2.vector

        # 1-2. Rango y nulidad.
        T = _build_transformation_matrix(v1, v2)
        _assert_full_rank(T, label="Transformación MIC")

        # 3. No singularidad.
        det_val = _assert_nonsingular(T, label="Transformación MIC")

        # 4. Ortogonalidad cruzada via Gram.
        G = _compute_gram_matrix(T)
        _assert_orthogonality_from_gram(G, label="Gram de MIC")

        # Verificación adicional: para vectores canónicos, |det(T)| = 1
        assert abs(abs(det_val) - 1.0) <= _NUMERICAL_TOLERANCE, (
            f"Para vectores canónicos binarios, |det(T)| debe ser 1. "
            f"Obtenido: |det(T)| = {abs(det_val)}.\n"
            f"v₁ = {v1}, v₂ = {v2}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # C. IDEMPOTENCIA DE PROYECTORES
    # ─────────────────────────────────────────────────────────────────────

    def test_projector_idempotence(self) -> None:
        """
        C. Idempotencia de proyectores: P_i² = P_i para todo i ∈ {1, 2}.

        Para proyectores sobre {0, 1}, se cumple trivialmente:
          • 0² = 0  ✓
          • 1² = 1  ✓

        La verificación explícita detecta regresiones donde los
        extractores pudieran retornar valores fuera del dominio.

        Se verifica sobre un conjunto representativo de estados
        que cubren todas las ramas lógicas de los extractores.

        Precisión: 1e-12.
        """
        test_states = {
            "null": _make_null_state(),
            "quantum": _make_quantum_state(),
            "hilbert": _make_hilbert_state(),
            "frustrated": _make_frustrated_state(),
        }

        for state_name, state in test_states.items():
            _assert_projector_idempotence(
                _extract_basis_e1, state, label=f"P₁ on {state_name}"
            )
            _assert_projector_idempotence(
                _extract_basis_e2, state, label=f"P₂ on {state_name}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # D. ORTOGONALIDAD PUNTUAL
    # ─────────────────────────────────────────────────────────────────────

    def test_pointwise_orthogonality_across_states(self) -> None:
        """
        D. Ortogonalidad puntual: P₁(s) · P₂(s) = 0 para todo s.

        Verifica que para NINGÚN estado de prueba se activen ambos
        proyectores simultáneamente. Esta es la propiedad fundamental
        que garantiza la descomposición V = Im(P₁) ⊕ Im(P₂).

        Cobertura:
          • Estado nulo (ningún proyector)
          • Estado cuántico (solo P₁)
          • Estado Hilbert (solo P₂)
          • Estado frustrado (veto)
          • Estado robusto (payload grande)

        Precisión: 1e-12.
        """
        test_states = {
            "null": _make_null_state(),
            "quantum": _make_quantum_state(),
            "hilbert": _make_hilbert_state(),
            "frustrated": _make_frustrated_state(),
            "robust": _make_robust_state(),
        }

        for state_name, state in test_states.items():
            _assert_pointwise_orthogonality(state, label=state_name)

    # ─────────────────────────────────────────────────────────────────────
    # E. RESOLUCIÓN DE LA IDENTIDAD
    # ─────────────────────────────────────────────────────────────────────

    def test_resolution_of_identity(self) -> None:
        """
        E. Resolución de la identidad: P₁(s) + P₂(s) ∈ {0, 1} para todo s.

        Verifica que el marco observacional sea completo:
          • Todo estado observable se proyecta a exactamente un eje
          • Los estados no observables se proyectan al vector nulo

        Cobertura:
          • Estado nulo: suma = 0 ✓
          • Estado cuántico: suma = 1 (solo P₁)
          • Estado Hilbert: suma = 1 (solo P₂)
          • Estado frustrado: suma = 0 (veto)

        Precisión: 1e-12.
        """
        test_states = {
            "null": _make_null_state(),
            "quantum": _make_quantum_state(),
            "hilbert": _make_hilbert_state(),
            "frustrated": _make_frustrated_state(),
        }

        for state_name, state in test_states.items():
            _assert_resolution_of_identity(
                state, label=state_name, allow_null_state=True,
            )

    # ─────────────────────────────────────────────────────────────────────
    # F. PRESERVACIÓN MONÓTONA DEL VETO (FLUJO DIKW)
    # ─────────────────────────────────────────────────────────────────────

    def test_preservation_of_dikw_reticular_functor(
        self,
        operators: RegisteredOperators,
    ) -> None:
        """
        F. Preservación monótona del veto a través del flujo DIKW.

        Propiedad:
        ──────────
        Si QuantumAdmissionGate emite un veto (quantum_error),
        entonces HilbertObserverAgent NO debe rehabilitar el estado.

        Formalización:
        ──────────────
        Sea f = quantum_gate, g = hilbert_agent.
        Axioma de monotonía: veto(f(s)) ⟹ veto(g(f(s)))

        Esto modela la categoría DIKW como un preorden donde
        el veto es un elemento minimal: una vez vetado,
        el estado permanece vetado en todos los niveles superiores.

        Proceso:
        ────────
        1. Evalúa quantum_gate(estado_frustrado)
        2. Verifica que quantum_gate emitió veto ("quantum_error" en context)
        3. Evalúa hilbert_agent(estado_vetado)
        4. Verifica que hilbert_agent NO rehabilitó (validated_strata vacío)
        5. Verifica ortogonalidad post-veto (para seguridad)

        Precisión: 1e-12.
        """
        initial_state = _make_frustrated_state()

        quantum_state = operators.quantum_gate(initial_state)
        final_state = operators.hilbert_agent(quantum_state)

        quantum_ctx = _safe_context(quantum_state)
        final_ctx = _safe_context(final_state)
        final_validated = getattr(final_state, "validated_strata", None)

        # El gate cuántico debe haber emitido veto
        assert "quantum_error" in quantum_ctx, (
            "La Puerta Cuántica no emitió veto como se esperaba.\n"
            f"context = {dict(quantum_ctx)}"
        )

        # El agente Hilbert no debe rehabilitar
        assert final_validated is not None, (
            "El estado final no expone `validated_strata`. "
            "No puede verificarse la monotonía del veto.\n"
            f"final context = {dict(final_ctx)}"
        )
        assert len(final_validated) == 0, (
            "El Agente Observador rehabilitó un estado vetado, "
            "violando la monotonía DIKW (f ⊆ g∘f en la categoría de veto).\n"
            f"validated_strata = {final_validated}"
        )

        # Verificación adicional: ortogonalidad preservada post-veto
        _assert_pointwise_orthogonality(final_state, label="post-veto state")

    # ─────────────────────────────────────────────────────────────────────
    # G. FUNCTORIALIDAD DE LA COMPOSICIÓN DIKW
    # ─────────────────────────────────────────────────────────────────────

    def test_dikw_composition_functoriality(
        self,
        operators: RegisteredOperators,
    ) -> None:
        """
        G. Functorialidad: F(g ∘ f) = F(g) ∘ F(f).

        Verifica que la descomposición observacional de la composición
        de operadores sea consistente con la composición de sus
        descomposiciones individuales.

        Formalización:
        ──────────────
        Sea F: CategoricalState → ℝ² el funtor de descomposición.
        Axioma de functorialidad: F(g ∘ f)(s) = F(g)(F(f)(s))

        En el contexto de proyectores binarios ortogonales, esto se
        reduce a verificar que la aplicación secuencial de operadores
        no produce vectores fuera del retículo observacional.

        Retículo válido: {(0,0), (1,0), (0,1)}
          • (0,0): estado nulo (invisible)
          • (1,0): estado puro en P₁
          • (0,1): estado puro en P₂

        Proceso:
        ────────
        1. Evalúa f(s) → estado intermedio
        2. Evalúa g(f(s)) → estado final
        3. Verifica que ambos estados cumplen ortogonalidad puntual
        4. Verifica que el vector final está en el retículo válido

        Precisión: 1e-12.
        """
        _configure_stable_non_veto_environment(operators.quantum_gate)
        _configure_stable_non_veto_environment(operators.hilbert_agent)

        test_states = {
            "null": _make_null_state(),
            "robust": _make_robust_state(),
        }

        for state_name, state in test_states.items():
            # Composición g ∘ f
            intermediate_state = operators.quantum_gate(state)
            final_state = operators.hilbert_agent(intermediate_state)

            # Validar estado intermedio
            if isinstance(intermediate_state, CategoricalState):
                decomp_mid = _extract_state_vector(intermediate_state)
                _assert_pointwise_orthogonality(
                    intermediate_state,
                    label=f"g∘f intermediate [{state_name}]",
                )

            # Validar estado final
            if isinstance(final_state, CategoricalState):
                decomp_final = _extract_state_vector(final_state)
                _assert_pointwise_orthogonality(
                    final_state,
                    label=f"g∘f final [{state_name}]",
                )

                # Verificar que el vector final está en el retículo
                v = decomp_final.vector
                in_lattice = any(
                    np.allclose(v, lv, atol=_NUMERICAL_TOLERANCE)
                    for lv in _VALID_LATTICE_VECTORS
                )
                assert in_lattice, (
                    f"[g∘f final, {state_name}] Vector fuera del retículo: {v}. "
                    f"Retículo válido: {_VALID_LATTICE_VECTORS}"
                )

    # ─────────────────────────────────────────────────────────────────────
    # H. CONMUTATIVIDAD DEL PRODUCTO INTERNO
    # ─────────────────────────────────────────────────────────────────────

    def test_inner_product_symmetry(
        self,
        operators: RegisteredOperators,
    ) -> None:
        """
        H. Simetría del producto interno: ⟨v₁, v₂⟩ = ⟨v₂, v₁⟩.

        Axioma fundamental del producto interno en espacios reales.

        Aunque la simetría de np.dot es trivial (es la definición
        matemática), la verificación explícita:
          • Documenta la expectativa algebraica
          • Detectaría cambios de métrica si se implementaran
          • Valida el contrato matemático del producto interno

        Precisión: 1e-12.
        """
        zero_state = _make_null_state()

        _, decomp_1 = _operator_image(operators.quantum_gate, zero_state)
        _, decomp_2 = _operator_image(operators.hilbert_agent, zero_state)

        v1, v2 = decomp_1.vector, decomp_2.vector

        ip_12 = float(np.dot(v1, v2))
        ip_21 = float(np.dot(v2, v1))

        assert abs(ip_12 - ip_21) <= _NUMERICAL_TOLERANCE, (
            f"Violación de simetría: ⟨v₁,v₂⟩ = {ip_12}, "
            f"⟨v₂,v₁⟩ = {ip_21}. "
            f"Diferencia: {abs(ip_12 - ip_21)}."
        )
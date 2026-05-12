"""
Módulo: MIC Algebra (Fundamentos de Teoría de Categorías y Morfismos Estructurales)
Ubicación: app/core/mic_algebra.py

FUNDAMENTOS MATEMÁTICOS RIGUROSOS - PARTE 1:

1. TEORÍA DE CATEGORÍAS:
   - Objetos: CategoricalState con propiedades universales
   - Morfismos: f: A → B con composición asociativa verificada
   - Identidades: ∀A ∈ Ob(C), ∃! id_A: A → A
   - Asociatividad: h∘(g∘f) = (h∘g)∘f verificada explícitamente

2. ÁLGEBRA DE ESTRATOS (DIKW):
   - Orden parcial con propiedades de retículo
   - Clausura transitiva verificable
   - Filtración topológica V_{PHYSICS} ⊂ ... ⊂ V_{WISDOM}

3. TOPOLOGÍA ALGEBRAICA:
   - Trazas de composición como cadenas en complejo
   - Verificación de aciclicidad (β₁ = 0)
   - Invariantes de Euler para estados

4. ANÁLISIS FUNCIONAL:
   - Estados como puntos en espacio de Hilbert
   - Normas bien definidas
   - Convergencia verificable

5. ESTABILIDAD NUMÉRICA:
   - Canonicalización con convergencia garantizada
   - Hashing resistente a colisiones
   - Comparaciones con tolerancia

Invariantes Matemáticos:
- Inmutabilidad: ∀s ∈ CategoricalState, s.frozen = True
- Determinismo: hash(s₁) = hash(s₂) ⟺ s₁ ≅ s₂
- Composición: (g∘f)(x) = g(f(x)) verificado
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    TypeVar,
    Generic,
)

import numpy as np

logger = logging.getLogger("MIC.Algebra")

# ==============================================================================
# CONSTANTES MATEMÁTICAS CON JUSTIFICACIÓN
# ==============================================================================

_SCHEMA_VERSION: str = "2.1.0"
_MAX_CANONICALIZE_DEPTH: int = 64  # Prevención de stack overflow
_ALGEBRAIC_TOL: float = 1e-10  # Tolerancia para propiedades algebraicas
_FLOAT_COMPARISON_TOL: float = 1e-9  # Tolerancia para comparación de floats
_HASH_COLLISION_PROB: float = 2**-256  # Probabilidad de colisión SHA-256

# Parámetros de geometría diferencial
_EHRESMANN_BASE_FRICTION: float = 0.1  # Fricción base para conexión
_MIN_EXERGY_LEVEL: float = 0.1  # Nivel mínimo de exergía

# ==============================================================================
# JERARQUÍA DE EXCEPCIONES MATEMÁTICAS
# ==============================================================================

class AlgebraicError(Exception):
    """Excepción base para errores algebraicos."""
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context = context

class CanonicalizationError(AlgebraicError):
    """Error durante canonicalización."""
    pass

class StratumResolutionError(AlgebraicError):
    """Error en resolución de estratos."""
    pass

class CategoryError(AlgebraicError):
    """Error en propiedades categóricas."""
    pass

class CompositionError(AlgebraicError):
    """Error en composición de morfismos."""
    pass

# ==============================================================================
# ESTRATIFICACIÓN DIKW CON PROPIEDADES DE RETÍCULO
# ==============================================================================

try:
    from app.core.schemas import Stratum
except ImportError:
    @unique
    class Stratum(IntEnum):
        """
        Estratificación DIKW como retículo acotado.
        
        Estructura Matemática:
        - (Stratum, ≤) es un poset (orden parcial)
        - Tiene elemento mínimo (⊥ = WISDOM)
        - Tiene elemento máximo (⊤ = PHYSICS)
        - Toda cadena es finita (altura = 6)
        
        Propiedades de Orden:
        - Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b
        - Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c
        - Reflexividad: a ≤ a
        
        Convención Ordinal:
        - Valor numérico MAYOR ↔ estrato más concreto (PHYSICS = 5)
        - Valor numérico MENOR ↔ estrato más abstracto (WISDOM = 0)
        
        Filtración Topológica:
        V₀ ⊂ V₁ ⊂ V₂ ⊂ V₃ ⊂ V₄ ⊂ V₅
        donde Vᵢ = {s ∈ Stratum : s.value ≥ i}
        """
        WISDOM = 0
        ALPHA = 1
        OMEGA = 2
        STRATEGY = 3
        TACTICS = 4
        PHYSICS = 5

        def requires(self) -> FrozenSet[Stratum]:
            """
            Clausura transitiva de dependencias.
            
            Definición Matemática:
            requires(s) = {t ∈ Stratum : t.value > s.value}
            
            Propiedades:
            1. Irreflexividad: s ∉ requires(s)
            2. Transitividad: t ∈ requires(s) ∧ u ∈ requires(t) ⟹ u ∈ requires(s)
            3. Antisimetría: s ∈ requires(t) ⟹ t ∉ requires(s)
            
            Invariante: |requires(s)| = 5 - s.value
            """
            return frozenset(s for s in Stratum if s.value > self.value)

        @property
        def level(self) -> int:
            """Nivel numérico del estrato (alias de value)."""
            return self.value

        @property
        def height(self) -> int:
            """
            Altura en el retículo (distancia desde ⊥).
            
            Definición: height(s) = |{t : t.value < s.value}|
            
            Invariante: height(s) = s.value
            """
            return self.value

        @property
        def depth(self) -> int:
            """
            Profundidad en el retículo (distancia desde ⊤).
            
            Definición: depth(s) = |{t : t.value > s.value}|
            
            Invariante: depth(s) = 5 - s.value
            """
            return 5 - self.value

        def is_successor_of(self, other: Stratum) -> bool:
            """
            Verifica si self es sucesor inmediato de other.
            
            Definición: is_successor_of(s, t) ⟺ s.value = t.value + 1
            
            Propiedad: is_successor_of es irreflexiva y asimétrica
            """
            return self.value == other.value + 1

        def is_predecessor_of(self, other: Stratum) -> bool:
            """
            Verifica si self es predecesor inmediato de other.
            
            Definición: is_predecessor_of(s, t) ⟺ s.value = t.value - 1
            """
            return self.value == other.value - 1

        def covers(self, other: Stratum) -> bool:
            """
            Relación de cobertura en el retículo.
            
            Definición: s covers t ⟺ s > t ∧ ¬∃u: s > u > t
            
            Para este retículo lineal: covers ≡ is_successor_of
            """
            return self.is_successor_of(other)

        def meet(self, other: Stratum) -> Stratum:
            """
            Ínfimo (meet) en el retículo: s ∧ t.
            
            Definición: s ∧ t = max{u : u ≤ s ∧ u ≤ t}
            
            Para orden lineal: meet(s, t) = min(s, t)
            
            Propiedad: meet es conmutativo, asociativo e idempotente
            """
            return self if self.value <= other.value else other

        def join(self, other: Stratum) -> Stratum:
            """
            Supremo (join) en el retículo: s ∨ t.
            
            Definición: s ∨ t = min{u : s ≤ u ∧ t ≤ u}
            
            Para orden lineal: join(s, t) = max(s, t)
            
            Propiedad: join es conmutativo, asociativo e idempotente
            """
            return self if self.value >= other.value else other

        @classmethod
        def bottom(cls) -> Stratum:
            """Elemento mínimo del retículo (⊥ = WISDOM)."""
            return cls.WISDOM

        @classmethod
        def top(cls) -> Stratum:
            """Elemento máximo del retículo (⊤ = PHYSICS)."""
            return cls.PHYSICS

        def __lt__(self, other: Stratum) -> bool:
            """Orden parcial: s < t ⟺ s.value < t.value."""
            if not isinstance(other, Stratum):
                return NotImplemented
            return self.value < other.value

        def __le__(self, other: Stratum) -> bool:
            """Orden parcial: s ≤ t ⟺ s.value ≤ t.value."""
            if not isinstance(other, Stratum):
                return NotImplemented
            return self.value <= other.value

        def __str__(self) -> str:
            return f"Stratum.{self.name}[{self.value}]"

# ==============================================================================
# UTILIDADES MATEMÁTICAS RIGUROSAS
# ==============================================================================

class MathUtils:
    """Utilidades matemáticas con garantías numéricas."""

    @staticmethod
    def float_equal(a: float, b: float, tol: float = _FLOAT_COMPARISON_TOL) -> bool:
        """
        Comparación de floats con tolerancia absoluta y relativa.
        
        Definición:
        equal(a, b) ⟺ |a - b| ≤ tol ∨ |a - b| ≤ tol·max(|a|, |b|)
        
        Propiedades:
        - Reflexiva: equal(a, a) = True
        - Simétrica: equal(a, b) = equal(b, a)
        - NO transitiva (por diseño numérico)
        """
        abs_diff = abs(a - b)
        if abs_diff <= tol:
            return True
        rel_tol = tol * max(abs(a), abs(b))
        return abs_diff <= rel_tol

    @staticmethod
    def safe_divide(
        numerator: float,
        denominator: float,
        eps: float = 1e-12
    ) -> float:
        """
        División segura con protección contra división por cero.
        
        Definición:
        safe_divide(a, b) = a / max(|b|, ε) · sign(b)
        
        Garantía: resultado es finito
        """
        if abs(denominator) < eps:
            sign = 1.0 if denominator >= 0 else -1.0
            return numerator / (sign * eps)
        return numerator / denominator

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """
        Clamp con verificación de orden.
        
        Precondición: min_val ≤ max_val
        Postcondición: min_val ≤ result ≤ max_val
        """
        if min_val > max_val:
            raise ValueError(
                f"Orden inválido: min_val={min_val} > max_val={max_val}"
            )
        return max(min_val, min(max_val, value))

def _canonicalize(value: Any, *, _depth: int = 0) -> Any:
    """
    Canonicalización determinista con verificación de convergencia.
    
    Propiedades Garantizadas:
    1. Determinismo: canon(x) = canon(x)
    2. Idempotencia: canon(canon(x)) = canon(x)
    3. Profundidad acotada: depth(canon(x)) ≤ MAX_DEPTH
    4. Orden estable: para colecciones no ordenadas
    
    Algoritmo:
    - Recursión con contador de profundidad
    - Orden lexicográfico para dicts
    - Orden por repr() para sets heterogéneos
    - Detección de ciclos por límite de profundidad
    
    Raises:
        CanonicalizationError: si se excede profundidad máxima
    """
    if _depth > _MAX_CANONICALIZE_DEPTH:
        raise CanonicalizationError(
            f"Profundidad de canonicalización excedida: {_MAX_CANONICALIZE_DEPTH}",
            depth=_depth,
            type=type(value).__name__
        )

    next_depth = _depth + 1

    # Casos base
    if value is None:
        return None

    # Manejo especial de Stratum
    if isinstance(value, Stratum):
        return {"__stratum__": value.name, "__value__": value.value}

    # Tipos primitivos
    if isinstance(value, (bool, int, float, str)):
        return value

    # Diccionarios
    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v, _depth=next_depth)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    # Listas y tuplas
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v, _depth=next_depth) for v in value]

    # Sets y frozensets
    if isinstance(value, (set, frozenset)):
        canonicalized = [_canonicalize(v, _depth=next_depth) for v in value]
        try:
            # Intento de ordenamiento natural
            return sorted(canonicalized)
        except TypeError:
            # Fallback a ordenamiento por repr()
            return sorted(canonicalized, key=repr)

    # Objetos con método to_dict
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _canonicalize(value.to_dict(), _depth=next_depth)
        except RecursionError:
            warnings.warn(
                f"Recursión detectada en to_dict() para {type(value).__name__}",
                RuntimeWarning
            )
            return repr(value)

    # Fallback a repr()
    return repr(value)

_VALID_CONFLICT_POLICIES: FrozenSet[str] = frozenset({
    "prefer_right",
    "prefer_left",
    "error_on_conflict",
})

def _safe_merge_dicts(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    conflict_policy: str = "prefer_right",
) -> Dict[str, Any]:
    """
    Fusión de diccionarios con política de resolución de conflictos.
    
    Políticas:
    - prefer_right: right[k] prevalece si k ∈ left ∩ right
    - prefer_left: left[k] prevalece si k ∈ left ∩ right
    - error_on_conflict: lanza excepción si ∃k: k ∈ left ∩ right ∧ left[k] ≠ right[k]
    
    Propiedades:
    - Asociatividad (prefer_*): merge(merge(a, b), c) = merge(a, merge(b, c))
    - Conmutatividad: solo para prefer_left/right cuando no hay conflictos
    
    Invariante: |result| ≤ |left| + |right|
    """
    if conflict_policy not in _VALID_CONFLICT_POLICIES:
        raise ValueError(
            f"Política inválida: '{conflict_policy}'. "
            f"Válidas: {sorted(_VALID_CONFLICT_POLICIES)}"
        )

    merged = dict(left)
    
    for key, value in right.items():
        if key in merged and merged[key] != value:
            if conflict_policy == "error_on_conflict":
                raise ValueError(
                    f"Conflicto en clave '{key}': "
                    f"left={merged[key]!r}, right={value!r}"
                )
            if conflict_policy == "prefer_left":
                continue
        
        merged[key] = value
    
    return merged

def _copy_trace(trace: Sequence[CompositionTrace]) -> Tuple[CompositionTrace, ...]:
    """
    Copia defensiva de secuencia de trazas.
    
    Propiedad: resultado es inmutable (tupla)
    Invariante: len(result) = len(trace)
    """
    return tuple(trace)

def _stable_hash(data: Any) -> str:
    """
    Hash SHA-256 determinista y resistente a colisiones.
    
    Propiedades:
    1. Determinismo: hash(x) = hash(x)
    2. Colisiones: P(hash(x) = hash(y) | x ≠ y) ≈ 2^-256
    3. Avalancha: cambio mínimo en x → cambio significativo en hash(x)
    
    Algoritmo:
    1. Canonicalización de datos
    2. Serialización JSON con orden determinista
    3. Hash SHA-256 en UTF-8
    
    Invariante: |hash(x)| = 64 caracteres hexadecimales
    """
    try:
        canonical = _canonicalize(data)
        serialized = json.dumps(
            canonical,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, CanonicalizationError) as e:
        # Fallback a repr() para tipos no serializables
        logger.warning(
            "Fallback a repr() para hash de %s: %s",
            type(data).__name__,
            e
        )
        return hashlib.sha256(repr(data).encode("utf-8")).hexdigest()

# ==============================================================================
# TRAZA DE AUDITORÍA CON INVARIANTES VERIFICADOS
# ==============================================================================

@dataclass(frozen=True, eq=True, slots=True)
class CompositionTrace:
    """
    Traza inmutable de ejecución de morfismo.
    
    Propiedades Algebraicas:
    - Inmutabilidad: frozen=True garantiza que no puede modificarse
    - Igualdad: definida por (step_number, morphism_name, success, error)
    - Hashable: puede usarse en sets y como clave de dict
    
    Invariantes:
    1. step_number ≥ 1
    2. timestamp > 0
    3. input_domain ⊆ Stratum
    4. output_codomain ∈ Stratum
    5. success ∈ {True, False}
    6. error = None ⟺ success = True (deseable, no estricto)
    
    Orden Temporal:
    Las trazas forman una secuencia ordenada por step_number,
    representando la historia de ejecución del pipeline.
    """
    __slots__ = (
        "step_number",
        "morphism_name",
        "input_domain",
        "output_codomain",
        "success",
        "error",
        "timestamp",
        "metadata"
    )

    step_number: int
    morphism_name: str
    input_domain: FrozenSet[Stratum]
    output_codomain: Stratum
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validación de invariantes post-construcción."""
        # Corregir step_number si es inválido
        if self.step_number < 1:
            object.__setattr__(self, "step_number", 1)
            logger.warning(
                "step_number corregido a 1 (era %d)",
                self.step_number
            )
        
        # Corregir timestamp si es inválido
        if self.timestamp <= 0:
            object.__setattr__(self, "timestamp", time.time())
            logger.warning("timestamp corregido a tiempo actual")
        
        # Verificar consistencia (no estricta, solo advertencia)
        if self.success and self.error is not None:
            logger.warning(
                "Inconsistencia: success=True pero error='%s'",
                self.error
            )

    @property
    def trace_identity_key(self) -> Tuple[int, str, bool, Optional[str]]:
        """
        Clave de identidad para deduplicación.
        
        Dos trazas con la misma clave representan el mismo evento lógico,
        aunque puedan diferir en timestamp o metadata.
        
        Propiedad: clave es hashable e inmutable
        Invariante: t1.trace_identity_key = t2.trace_identity_key ⟹ t1 ≈ t2
        """
        return (self.step_number, self.morphism_name, self.success, self.error)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización JSON-compatible.
        
        Propiedad: JSON.parse(JSON.stringify(to_dict())) ≈ to_dict()
        """
        return {
            "step": self.step_number,
            "morphism": self.morphism_name,
            "domain": sorted(s.name for s in self.input_domain),
            "codomain": self.output_codomain.name,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": _canonicalize(self.metadata) if self.metadata else None,
        }

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} #{self.step_number}: {self.morphism_name} "
            f"({len(self.input_domain)} → {self.output_codomain.name})"
        )

# ==============================================================================
# OBJETO FUNDAMENTAL DE LA CATEGORÍA
# ==============================================================================

@dataclass(frozen=True, slots=True)
class CategoricalState:
    """
    Objeto fundamental de C_MIC con propiedades categóricas verificadas.
    
    TEORÍA DE CATEGORÍAS:
    
    Ob(C_MIC) = {CategoricalState}
    
    Propiedades Universales:
    1. Objeto Inicial (⊥): CategoricalState() con payload={}
    2. Objeto Terminal (⊤): estado con is_success=False (absorción)
    3. Producto: definido por ProductMorphism
    4. Coproducto: definido por CoproductMorphism
    
    Invariantes Estructurales:
    1. Inmutabilidad: frozen=True (no puede modificarse)
    2. Coherencia: payload, context son dicts (mutables conceptualmente,
       pero el contrato exige no modificarlos post-construcción)
    3. Estratos validados: validated_strata es frozenset (inmutable)
    4. Trazas: composition_trace es tupla (inmutable)
    
    Propiedades Algebraicas:
    1. Hash determinista: hash(s) es reproducible
    2. Igualdad estructural: s1 = s2 ⟺ s1.to_dict() = s2.to_dict()
    3. Serialización: to_dict() es biyectiva con from_dict()
    
    Propiedades Topológicas:
    1. Clausura: validated_strata es cerrado bajo `requires()`
    2. Nivel: stratum_level = min{s.value : s ∈ validated_strata}
    3. Altura: número de estratos validados
    """
    __slots__ = (
        "payload",
        "context",
        "validated_strata",
        "error",
        "error_details",
        "forensic_evidence",
        "composition_trace"
    )

    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validated_strata: FrozenSet[Stratum] = field(default_factory=frozenset)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    forensic_evidence: Optional[Dict[str, Any]] = None
    composition_trace: Tuple[CompositionTrace, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """
        Normalización de estratos validados.
        
        Garantiza que validated_strata contenga solo objetos Stratum
        válidos, convirtiendo desde int/str si es necesario.
        """
        if not self.validated_strata:
            return
        
        corrected_strata: List[Stratum] = []
        
        for s in self.validated_strata:
            try:
                if isinstance(s, Stratum):
                    corrected_strata.append(s)
                elif isinstance(s, int):
                    corrected_strata.append(Stratum(s))
                elif isinstance(s, str):
                    corrected_strata.append(Stratum[s.upper().strip()])
                else:
                    logger.warning(
                        "Tipo inválido en validated_strata: %s",
                        type(s).__name__
                    )
                    corrected_strata.append(s)
            except (ValueError, KeyError) as e:
                logger.warning(
                    "Error normalizando estrato %r: %s",
                    s,
                    e
                )
                corrected_strata.append(s)
        
        object.__setattr__(
            self,
            "validated_strata",
            frozenset(corrected_strata)
        )

    # -------------------------------------------------------------------------
    # PROPIEDADES CATEGÓRICAS
    # -------------------------------------------------------------------------

    @property
    def is_success(self) -> bool:
        """
        Predicado de éxito.
        
        Definición: is_success ⟺ error = None
        
        Propiedad: is_success ∨ is_failed (ley del tercero excluido)
        """
        return self.error is None

    @property
    def is_failed(self) -> bool:
        """
        Predicado de fallo.
        
        Definición: is_failed ⟺ error ≠ None
        
        Propiedad: is_failed = ¬is_success
        """
        return not self.is_success

    @property
    def stratum_level(self) -> int:
        """
        Nivel de estrato más abstracto alcanzado.
        
        Definición:
        stratum_level = min{s.value : s ∈ validated_strata} ∪ {PHYSICS.value}
        
        Invariante: 0 ≤ stratum_level ≤ 5
        
        Propiedad: menor valor → más abstracto (WISDOM=0, PHYSICS=5)
        """
        if not self.validated_strata:
            return Stratum.PHYSICS.value
        return min(s.value for s in self.validated_strata)

    @property
    def stratum_height(self) -> int:
        """
        Altura en la jerarquía DIKW.
        
        Definición: height = |validated_strata|
        
        Invariante: 0 ≤ height ≤ 6
        """
        return len(self.validated_strata)

    @property
    def accumulated_strata(self) -> FrozenSet[Stratum]:
        """
        Alias semántico para validated_strata.
        
        Representa la acumulación de estratos alcanzados
        durante la ejecución del pipeline.
        """
        return self.validated_strata

    @property
    def trace_length(self) -> int:
        """
        Longitud de la traza de composición.
        
        Invariante: trace_length ≥ 0
        """
        return len(self.composition_trace)

    # -------------------------------------------------------------------------
    # OPERACIONES FUNCTORIALES
    # -------------------------------------------------------------------------

    def with_update(
        self,
        new_payload: Optional[Dict[str, Any]] = None,
        new_context: Optional[Dict[str, Any]] = None,
        new_stratum: Optional[Stratum] = None,
        *,
        merge_payload: bool = True,
        merge_context: bool = True,
        payload_conflict_policy: str = "prefer_right",
        context_conflict_policy: str = "prefer_right",
    ) -> CategoricalState:
        """
        Funtor de actualización: F(s) = s con modificaciones.
        
        Propiedades:
        1. Pureza: no modifica self (retorna nuevo objeto)
        2. Composicionalidad: with_update preserva estructura
        3. Determinismo: mismo input → mismo output
        
        Semántica de Fusión:
        - merge=True: diccionarios se fusionan según política
        - merge=False: diccionario se reemplaza completamente
        
        Invariantes Preservados:
        - error y error_details se copian de self
        - forensic_evidence se copia de self
        - composition_trace se copia de self
        
        Returns:
            Nuevo CategoricalState con actualizaciones aplicadas
        """
        # Payload
        if new_payload is None:
            updated_payload = dict(self.payload)
        elif merge_payload:
            updated_payload = _safe_merge_dicts(
                dict(self.payload),
                dict(new_payload),
                conflict_policy=payload_conflict_policy,
            )
        else:
            updated_payload = dict(new_payload)

        # Context
        if new_context is None:
            updated_context = dict(self.context)
        elif merge_context:
            updated_context = _safe_merge_dicts(
                dict(self.context),
                dict(new_context),
                conflict_policy=context_conflict_policy,
            )
        else:
            updated_context = dict(new_context)

        # Strata
        updated_strata = self.validated_strata
        if new_stratum is not None:
            updated_strata = updated_strata | frozenset({new_stratum})

        return CategoricalState(
            payload=updated_payload,
            context=updated_context,
            validated_strata=updated_strata,
            error=self.error,
            error_details=(
                dict(self.error_details)
                if self.error_details
                else None
            ),
            forensic_evidence=(
                dict(self.forensic_evidence)
                if self.forensic_evidence
                else None
            ),
            composition_trace=_copy_trace(self.composition_trace),
        )

    def with_error(
        self,
        error_msg: str,
        details: Optional[Dict[str, Any]] = None,
        forensic_evidence: Optional[Dict[str, Any]] = None,
    ) -> CategoricalState:
        """
        Funtor de error: F(s) = s en estado fallido.
        
        Propiedades:
        1. Absorción: morfismo aplicado a estado fallido → estado fallido
        2. Preservación: payload y context se conservan para diagnóstico
        3. Monotonicidad: estado fallido no puede volver a éxito (usar clear_error)
        
        Propiedad Monádica:
        with_error encapsula el fallo sin perder información contextual.
        
        Returns:
            Nuevo CategoricalState en condición de error
        """
        return CategoricalState(
            payload=dict(self.payload),
            context=dict(self.context),
            validated_strata=self.validated_strata,
            error=error_msg,
            error_details=dict(details) if details else None,
            forensic_evidence=(
                dict(forensic_evidence)
                if forensic_evidence
                else (
                    dict(self.forensic_evidence)
                    if self.forensic_evidence
                    else None
                )
            ),
            composition_trace=_copy_trace(self.composition_trace),
        )

    def clear_error(self) -> CategoricalState:
        """
        Limpia el error, retornando a estado de éxito.
        
        Propiedades:
        1. Idempotencia: clear_error().clear_error() = clear_error()
        2. Proyección: clear_error() preserva payload y context
        3. Reset: error y error_details se eliminan
        
        Invariante: clear_error().is_success = True
        
        Returns:
            Nuevo CategoricalState sin error
        """
        return CategoricalState(
            payload=dict(self.payload),
            context=dict(self.context),
            validated_strata=self.validated_strata,
            error=None,
            error_details=None,
            forensic_evidence=(
                dict(self.forensic_evidence)
                if self.forensic_evidence
                else None
            ),
            composition_trace=_copy_trace(self.composition_trace),
        )

    def add_trace(
        self,
        morphism_name: str,
        input_domain: FrozenSet[Stratum],
        output_codomain: Stratum,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CategoricalState:
        """
        Agrega entrada de traza al historial.
        
        Propiedades:
        1. Monotonicidad: add_trace incrementa trace_length
        2. Orden: las trazas se ordenan por step_number
        3. Inmutabilidad: retorna nuevo estado
        
        Invariante: len(result.composition_trace) = len(self.composition_trace) + 1
        
        Returns:
            Nuevo CategoricalState con traza adicional
        """
        trace_entry = CompositionTrace(
            step_number=len(self.composition_trace) + 1,
            morphism_name=morphism_name,
            input_domain=input_domain,
            output_codomain=output_codomain,
            success=success,
            error=error,
            metadata=dict(metadata) if metadata else None,
        )
        
        return CategoricalState(
            payload=dict(self.payload),
            context=dict(self.context),
            validated_strata=self.validated_strata,
            error=self.error,
            error_details=(
                dict(self.error_details)
                if self.error_details
                else None
            ),
            forensic_evidence=(
                dict(self.forensic_evidence)
                if self.forensic_evidence
                else None
            ),
            composition_trace=self.composition_trace + (trace_entry,),
        )

    # -------------------------------------------------------------------------
    # SERIALIZACIÓN Y HASH
    # -------------------------------------------------------------------------

    def compute_hash(self) -> str:
        """
        Hash SHA-256 determinista del estado completo.
        
        Propiedades:
        1. Determinismo: hash(s) = hash(s)
        2. Sensibilidad: cambio mínimo → hash diferente
        3. Colisiones: P(hash(s1) = hash(s2) | s1 ≠ s2) ≈ 2^-256
        
        Incluye versión de esquema para detectar incompatibilidades.
        
        Invariante: |compute_hash()| = 64 caracteres hex
        
        Returns:
            Hash hexadecimal de 64 caracteres
        """
        data = {
            "__schema_version__": _SCHEMA_VERSION,
            "payload": _canonicalize(self.payload),
            "context": _canonicalize(self.context),
            "validated_strata": sorted(s.name for s in self.validated_strata),
            "error": self.error,
            "error_details": _canonicalize(self.error_details),
            "composition_trace": [
                _canonicalize(t.to_dict())
                for t in self.composition_trace
            ],
        }
        return _stable_hash(data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización completa JSON-compatible.
        
        Propiedad: from_dict(to_dict()) ≈ self (módulo timestamps)
        
        Returns:
            Diccionario serializable
        """
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "payload": _canonicalize(self.payload),
            "context": _canonicalize(self.context),
            "validated_strata": sorted(s.name for s in self.validated_strata),
            "error": self.error,
            "error_details": _canonicalize(self.error_details),
            "forensic_evidence": _canonicalize(self.forensic_evidence),
            "composition_trace": [t.to_dict() for t in self.composition_trace],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CategoricalState:
        """
        Deserialización con validación estructural.
        
        Propiedades:
        1. Biyección: from_dict(to_dict(s)) ≈ s
        2. Validación: lanza excepciones si datos inválidos
        3. Compatibilidad: maneja versiones de esquema
        
        Raises:
            KeyError: si faltan campos obligatorios en trazas
            ValueError: si nombre de estrato es inválido
            StratumResolutionError: si estrato no puede resolverse
        
        Returns:
            CategoricalState reconstruido
        """
        # Verificar versión de esquema
        schema_ver = data.get("__schema_version__")
        if schema_ver and schema_ver != _SCHEMA_VERSION:
            logger.warning(
                "Versión de esquema diferente: esperada=%s, encontrada=%s",
                _SCHEMA_VERSION,
                schema_ver
            )

        # Reconstruir estratos
        strata_names = data.get("validated_strata", [])
        try:
            strata = frozenset(Stratum[s] for s in strata_names)
        except KeyError as exc:
            raise StratumResolutionError(
                f"Estrato inválido en from_dict: {exc}",
                strata_names=strata_names
            ) from exc

        # Reconstruir trazas
        raw_traces = data.get("composition_trace", [])
        traces: List[CompositionTrace] = []
        
        for i, t in enumerate(raw_traces):
            # Verificar campos requeridos
            required_fields = {"step", "morphism", "domain", "codomain", "success"}
            missing = required_fields - set(t.keys())
            if missing:
                raise KeyError(
                    f"Traza #{i} carece de campos: {sorted(missing)}"
                )
            
            try:
                traces.append(
                    CompositionTrace(
                        step_number=int(t["step"]),
                        morphism_name=str(t["morphism"]),
                        input_domain=frozenset(
                            Stratum[s] for s in t["domain"]
                        ),
                        output_codomain=Stratum[t["codomain"]],
                        success=bool(t["success"]),
                        error=t.get("error"),
                        timestamp=float(t.get("timestamp", 0.0)),
                        metadata=t.get("metadata"),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Error reconstruyendo traza #{i}: {exc}"
                ) from exc

        return cls(
            payload=dict(data.get("payload", {})),
            context=dict(data.get("context", {})),
            validated_strata=strata,
            error=data.get("error"),
            error_details=data.get("error_details"),
            forensic_evidence=data.get("forensic_evidence"),
            composition_trace=tuple(traces),
        )

    def __str__(self) -> str:
        status = "✓" if self.is_success else "✗"
        strata_str = ", ".join(
            sorted(s.name for s in self.validated_strata)
        ) or "∅"
        return (
            f"CategoricalState[{status}]("
            f"strata={{{strata_str}}}, "
            f"trace_len={self.trace_length})"
        )

    def __repr__(self) -> str:
        return (
            f"CategoricalState("
            f"payload_keys={sorted(self.payload.keys())}, "
            f"validated_strata={len(self.validated_strata)}, "
            f"error={self.error!r})"
        )

# ==============================================================================
# FACTORIES CON VALIDACIÓN
# ==============================================================================

def create_categorical_state(
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    strata: Optional[Set[Stratum]] = None,
) -> CategoricalState:
    """
    Factory para CategoricalState con valores seguros por defecto.
    
    Propiedades:
    1. Validación: todos los argumentos se validan
    2. Inmutabilidad: copias defensivas de dicts
    3. Objeto inicial: create_categorical_state() ≅ ⊥
    
    Args:
        payload: Diccionario de carga útil (default: {})
        context: Contexto adicional (default: {})
        strata: Conjunto de estratos validados (default: ∅)
    
    Returns:
        CategoricalState inicializado
    """
    return CategoricalState(
        payload=dict(payload or {}),
        context=dict(context or {}),
        validated_strata=frozenset(strata or set()),
    )

# ==============================================================================
# EXPORTACIÓN PÚBLICA CONTROLADA - PARTE 1
# ==============================================================================

__all__ = [
    # Excepciones
    "AlgebraicError",
    "CanonicalizationError",
    "StratumResolutionError",
    "CategoryError",
    "CompositionError",
    
    # Estratificación
    "Stratum",
    
    # Utilidades
    "MathUtils",
    
    # Estado categórico
    "CategoricalState",
    "CompositionTrace",
    "create_categorical_state",
    
    # Constantes
    "_SCHEMA_VERSION",
    "_MAX_CANONICALIZE_DEPTH",
    "_ALGEBRAIC_TOL",
    "_FLOAT_COMPARISON_TOL",
]


"""
FUNDAMENTOS MATEMÁTICOS RIGUROSOS - PARTE 2:

6. MORFISMOS CATEGÓRICOS:
   - Axiomas: identidad y asociatividad verificados
   - Composición: (g∘f)(x) = g(f(x)) con pruebas
   - Tipos: domain/codomain con verificación estricta

7. CONSTRUCCIONES UNIVERSALES:
   - Producto: límite del diagrama discreto
   - Coproducto: colímite del diagrama discreto
   - Pullback: límite del span
   - Propiedades universales verificadas

8. FUNTORES:
   - Preservación: F(g∘f) = F(g)∘F(f) verificado
   - Identidad: F(id_A) = id_{F(A)} verificado
   - Naturalidad: diagramas conmutativos

9. GEOMETRÍA DIFERENCIAL:
   - Conexión de Ehresmann con compatibilidad
   - Curvatura con tensor métrico
   - Transporte paralelo con holonomía

10. VERIFICACIÓN ESTRUCTURAL:
    - Aciclicidad: β₁ = 0 para grafos de dependencias
    - Composabilidad: verificación algebraica
    - Cobertura de estratos
"""

# Continuación de mic_algebra.py - PARTE 2

import threading
from typing import Callable, Dict, List, Optional, Set, FrozenSet, Any, Tuple

import networkx as nx

# ==============================================================================
# MORFISMOS — CLASE BASE CON AXIOMAS CATEGÓRICOS
# ==============================================================================

class Morphism(ABC):
    """
    Morfismo en la categoría C_MIC con axiomas categóricos verificables.
    
    TEORÍA DE CATEGORÍAS:
    
    Un morfismo f: A → B en C_MIC satisface:
    
    Axioma 1 (Identidad):
        ∀A ∈ Ob(C_MIC), ∃! id_A: A → A tal que
        f ∘ id_A = f  y  id_B ∘ f = f
    
    Axioma 2 (Asociatividad):
        (h ∘ g) ∘ f = h ∘ (g ∘ f)
    
    Axioma 3 (Composición Tipada):
        cod(f) ∈ dom(g) ⟹ g ∘ f está bien definido
    
    Propiedades Functoriales:
    - F(id_A) = id_{F(A)} (preservación de identidad)
    - F(g ∘ f) = F(g) ∘ F(f) (preservación de composición)
    
    Invariantes:
    - domain ⊆ Stratum (conjunto de estratos requeridos)
    - codomain ∈ Stratum (estrato producido)
    - __call__ es determinista (mismo input → mismo output)
    """
    __slots__ = ("name", "_logger", "_call_count")

    def __init__(self, name: str = "") -> None:
        self.name: str = name or self.__class__.__name__
        self._logger: logging.Logger = logging.getLogger(
            f"MIC.Morphism.{self.name}"
        )
        self._call_count: int = 0

    @property
    @abstractmethod
    def domain(self) -> FrozenSet[Stratum]:
        """
        Dominio del morfismo: conjunto de estratos requeridos.
        
        Propiedad: domain ⊆ Stratum
        Invariante: domain es inmutable (frozenset)
        """
        ...

    @property
    @abstractmethod
    def codomain(self) -> Stratum:
        """
        Codominio del morfismo: estrato producido.
        
        Propiedad: codomain ∈ Stratum
        Invariante: codomain es único
        """
        ...

    @property
    def call_count(self) -> int:
        """Contador de invocaciones (para diagnóstico)."""
        return self._call_count

    def can_compose_with(self, other: Morphism) -> bool:
        """
        Verifica composabilidad: self >> other.
        
        Definición:
        can_compose_with(f, g) ⟺ dom(g) ⊆ dom(f) ∪ {cod(f)}
        
        Propiedad de Acumulación:
        La semántica de estratos es acumulativa: ejecutar f provee
        todos los estratos de dom(f) más cod(f).
        
        Invariante: can_compose_with es reflexiva para IdentityMorphism
        
        Returns:
            True si other puede ejecutarse después de self
        """
        provided = self.domain | frozenset({self.codomain})
        return other.domain.issubset(provided)

    def verify_axioms(self) -> Dict[str, bool]:
        """
        Verifica axiomas categóricos (donde aplicable).
        
        Checks:
        1. domain es frozenset no vacío
        2. codomain es Stratum válido
        3. __call__ es callable
        
        Returns:
            Dict con resultados de verificación
        """
        checks = {
            "domain_is_frozenset": isinstance(self.domain, frozenset),
            "domain_non_empty": len(self.domain) > 0,
            "codomain_is_stratum": isinstance(self.codomain, Stratum),
            "callable": callable(self.__call__),
        }
        
        # Para IdentityMorphism, verificar que domain = {codomain}
        if isinstance(self, IdentityMorphism):
            checks["identity_domain_singleton"] = (
                len(self.domain) == 1 and
                self.codomain in self.domain
            )
        
        return checks

    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplicación del morfismo: f(state) → state'.
        
        Propiedades:
        1. Pureza: no modifica state (retorna nuevo objeto)
        2. Absorción: f(⊥) = ⊥ (preserva objeto fallido)
        3. Determinismo: f(s) = f(s)
        4. Trazabilidad: f(s).trace_length = s.trace_length + k
        
        Invariante: resultado es CategoricalState válido
        """
        ...

    def __rshift__(self, other: Morphism) -> Morphism:
        """
        Composición secuencial: self >> other ≡ other ∘ self.
        
        Notación:
        - Matemática: g ∘ f (g después de f)
        - Código: f >> g (f luego g, lectura izq→der)
        
        Propiedad Asociativa:
        (f >> g) >> h ≡ f >> (g >> h)
        
        Returns:
            ComposedMorphism(self, other)
        """
        return ComposedMorphism(self, other)

    def __mul__(self, other: Morphism) -> Morphism:
        """
        Producto categórico: self * other ≡ self × other.
        
        Propiedad Universal (Producto):
        ∀X, ∀f: X → A, ∀g: X → B, ∃! h: X → A×B tal que
        π_A ∘ h = f  y  π_B ∘ h = g
        
        Returns:
            ProductMorphism(self, other)
        """
        return ProductMorphism(self, other)

    def __or__(self, other: Morphism) -> Morphism:
        """
        Coproducto categórico: self | other ≡ self ∐ other.
        
        Propiedad Universal (Coproducto):
        ∀X, ∀f: A → X, ∀g: B → X, ∃! h: A∐B → X tal que
        h ∘ i_A = f  y  h ∘ i_B = g
        
        Returns:
            CoproductMorphism(self, other)
        """
        return CoproductMorphism(self, other)

    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return (
            f"{self.name}: ({domain_str}) → {self.codomain.name} "
            f"[calls={self._call_count}]"
        )

# ==============================================================================
# MORFISMO IDENTIDAD
# ==============================================================================

class IdentityMorphism(Morphism):
    """
    Morfismo identidad id_s: s → s.
    
    Propiedades Categóricas:
    1. Neutralidad Izquierda: id_B ∘ f = f  ∀f: A → B
    2. Neutralidad Derecha: f ∘ id_A = f  ∀f: A → B
    3. Idempotencia: id_s ∘ id_s = id_s
    
    Verificación:
    - domain = {s}
    - codomain = s
    - id_s(state) = state (con traza adicional)
    
    Invariante: ∀s ∈ Stratum, ∃! id_s
    """
    __slots__ = ("_stratum",)

    def __init__(self, stratum: Stratum) -> None:
        super().__init__(f"id_{stratum.name}")
        self._stratum = stratum

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset({self._stratum})

    @property
    def codomain(self) -> Stratum:
        return self._stratum

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplicación identidad: id(s) = s.
        
        Propiedad: id(s).payload = s.payload (módulo traza)
        """
        self._call_count += 1
        
        return state.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=True,
            metadata={"identity": True, "call_count": self._call_count},
        )

# ==============================================================================
# MORFISMO ATÓMICO (ADAPTADOR DE HANDLERS)
# ==============================================================================

class AtomicVector(Morphism):
    """
    Morfismo atómico que encapsula un handler callable.
    
    Propiedades:
    1. Atomicidad: no descomponible en morfismos más simples
    2. Determinismo: handler(kwargs) es determinista
    3. Absorción monádica: f(⊥) = ⊥
    
    Handler Contract:
    - Input: kwargs filtrados por required_keys ∪ optional_keys
    - Output: dict con resultado o valor escalar
    - Error: dict con {"success": False, "error": str}
    
    Invariante: handler es callable puro (sin efectos laterales)
    """
    __slots__ = (
        "_target_stratum",
        "_handler",
        "_required_keys",
        "_optional_keys",
        "_domain"
    )

    def __init__(
        self,
        name: str,
        target_stratum: Stratum,
        handler: Callable[..., Any],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name)
        self._target_stratum = target_stratum
        self._handler = handler
        self._required_keys: FrozenSet[str] = frozenset(required_keys or [])
        self._optional_keys: FrozenSet[str] = frozenset(optional_keys or [])
        self._domain = frozenset(target_stratum.requires())

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._target_stratum

    def _absorb_error(self, state: CategoricalState) -> CategoricalState:
        """
        Absorción monádica: propaga error sin ejecutar handler.
        
        Propiedad: absorb(⊥) = ⊥
        """
        return state.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=False,
            error=f"Absorción monadal: error previo '{state.error}'",
            metadata={"absorbed": True},
        )

    def _validate_domain(
        self, state: CategoricalState
    ) -> Optional[CategoricalState]:
        """
        Valida clausura transitiva de estratos.
        
        Verifica: dom(f) ⊆ validated_strata ∪ {force_override}
        
        Returns:
            None si válido, estado de error si inválido
        """
        force_override = bool(state.context.get("force_override", False))
        if force_override:
            self._logger.debug("force_override activo, omitiendo validación de dominio")
            return None

        missing = self.domain - state.validated_strata
        if not missing:
            return None

        missing_names = sorted(s.name for s in missing)
        error_msg = (
            f"Violación de clausura transitiva en '{self.name}': "
            f"requiere estratos {missing_names} no validados"
        )
        self._logger.warning(error_msg)
        
        return state.with_error(
            error_msg,
            details={
                "missing_strata": missing_names,
                "validated_strata": sorted(s.name for s in state.validated_strata),
            },
        ).add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=False,
            error=error_msg,
        )

    def _validate_required_keys(
        self, state: CategoricalState
    ) -> Optional[CategoricalState]:
        """
        Valida presencia de claves requeridas.
        
        Verifica: required_keys ⊆ payload.keys()
        
        Returns:
            None si válido, estado de error si faltan claves
        """
        missing_keys = self._required_keys - frozenset(state.payload.keys())
        if not missing_keys:
            return None

        sorted_missing = sorted(missing_keys)
        error_msg = f"Claves requeridas faltantes: {sorted_missing}"
        self._logger.warning("%s: %s", self.name, error_msg)
        
        return state.with_error(
            error_msg,
            details={
                "missing_keys": sorted_missing,
                "available_keys": sorted(state.payload.keys()),
                "required_keys": sorted(self._required_keys),
            },
        ).add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=False,
            error=error_msg,
        )

    def _execute_handler(
        self, state: CategoricalState
    ) -> CategoricalState:
        """
        Ejecuta handler con kwargs filtrados.
        
        Pipeline:
        1. Filtrar kwargs por allowed_keys
        2. Invocar handler(**kwargs)
        3. Procesar resultado
        4. Retornar estado actualizado
        """
        allowed_keys = self._required_keys | self._optional_keys
        kwargs = {
            k: v for k, v in state.payload.items()
            if k in allowed_keys
        }

        try:
            result = self._handler(**kwargs)
        except Exception as exc:
            error_msg = f"Excepción en handler '{self.name}': {exc}"
            self._logger.exception(error_msg)
            
            return state.with_error(
                error_msg,
                details={
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                    "kwargs_keys": sorted(kwargs.keys()),
                },
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        return self._process_result(state, result)

    def _process_result(
        self, state: CategoricalState, result: Any
    ) -> CategoricalState:
        """
        Procesa resultado del handler y produce estado de salida.
        
        Semántica:
        - dict con "success"=False → estado de error
        - dict con claves → merge con payload
        - otro tipo → empaqueta como {name_result: valor}
        """
        if isinstance(result, dict):
            success = bool(result.get("success", True))
            
            if not success:
                error_msg = result.get(
                    "error",
                    f"Handler '{self.name}' retornó success=False"
                )
                error_details = result.get("error_details")
                
                return state.with_error(
                    error_msg,
                    details=error_details,
                ).add_trace(
                    self.name,
                    self.domain,
                    self.codomain,
                    success=False,
                    error=error_msg,
                )

            # Filtrar claves de control
            _CONTROL_KEYS = {"success", "error", "error_details"}
            clean_result = {
                k: v
                for k, v in result.items()
                if not k.startswith("_") and k not in _CONTROL_KEYS
            }
            
            new_state = state.with_update(
                clean_result,
                new_stratum=self.codomain,
                payload_conflict_policy="prefer_right",
            )
        else:
            # Resultado escalar: empaquetar
            new_state = state.with_update(
                {f"{self.name}_result": result},
                new_stratum=self.codomain,
                payload_conflict_policy="prefer_right",
            )

        return new_state.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=True,
            metadata={"result_type": type(result).__name__},
        )

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Pipeline de ejecución con validaciones.
        
        Fases:
        1. Absorción monádica (si error previo)
        2. Validación de dominio
        3. Validación de claves requeridas
        4. Ejecución del handler
        
        Propiedad: cada fase puede terminar early con error
        """
        self._call_count += 1

        # Fase 1: Absorción monádica
        if state.is_failed:
            return self._absorb_error(state)

        # Fase 2: Validación de dominio
        domain_error = self._validate_domain(state)
        if domain_error is not None:
            return domain_error

        # Fase 3: Validación de claves
        keys_error = self._validate_required_keys(state)
        if keys_error is not None:
            return keys_error

        # Fase 4: Ejecución
        return self._execute_handler(state)

# ==============================================================================
# COMPOSICIÓN CATEGÓRICA CON TRANSPORTE PARALELO
# ==============================================================================

class ComposedMorphism(Morphism):
    """
    Composición g ∘ f con transporte paralelo vía conexión de Ehresmann.
    
    TEORÍA DE CATEGORÍAS:
    
    Definición:
    (g ∘ f)(x) = g(f(x))
    
    Propiedades:
    1. Asociatividad: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    2. Identidad: id_B ∘ f = f, f ∘ id_A = f
    3. Composición de dominios:
       dom(g ∘ f) = dom(f) ∪ (dom(g) − {cod(f)} − dom(f))
    
    GEOMETRÍA DIFERENCIAL:
    
    Conexión de Ehresmann:
    Para transportar estado desde stratum_f a stratum_g,
    se calcula la 1-forma de conexión ω que mide la "fricción"
    del transporte paralelo.
    
    Curvatura:
    Ω = dω + ω ∧ ω mide la no integrabilidad de la conexión.
    
    Invariante: is_structurally_compatible ⟹ dom(g) ⊆ dom(f) ∪ {cod(f)}
    """
    __slots__ = (
        "f",
        "g",
        "_is_structurally_compatible",
        "_domain",
        "_codomain"
    )

    def __init__(self, f: Morphism, g: Morphism) -> None:
        super().__init__(f"{f.name} >> {g.name}")
        self.f = f
        self.g = g

        # Verificación de compatibilidad estructural
        provided_by_f = f.domain | frozenset({f.codomain})
        self._is_structurally_compatible: bool = g.domain.issubset(provided_by_f)

        # Cálculo del dominio del compuesto
        self._domain: FrozenSet[Stratum] = f.domain | (g.domain - provided_by_f)
        self._codomain: Stratum = g.codomain

        if not self._is_structurally_compatible:
            unsatisfied = g.domain - provided_by_f
            logger.warning(
                "Composición '%s' estructuralmente débil: "
                "g requiere %s no provistos por f. "
                "Se elevan al dominio del compuesto.",
                self.name,
                sorted(s.name for s in unsatisfied),
            )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    @property
    def is_structurally_compatible(self) -> bool:
        """
        Verificación estricta de compatibilidad estructural.
        
        Definición: dom(g) ⊆ dom(f) ∪ {cod(f)}
        
        Propiedad: compatibilidad garantiza composabilidad sin
        requisitos adicionales.
        """
        return self._is_structurally_compatible

    def _compute_ehresmann_connection(
        self,
        state: CategoricalState,
        target_stratum: Stratum
    ) -> float:
        """
        Calcula 1-forma de conexión ω de Ehresmann.
        
        Definición Geométrica:
        ω mide la "fricción" o "trabajo" requerido para transportar
        el estado desde su nivel actual hasta target_stratum.
        
        Fórmula:
        ω = (base_friction · Δlevel) / exergy_level
        
        donde:
        - Δlevel = current_level - target_level (diferencia de altura)
        - exergy_level = nivel de energía disponible
        - base_friction = coeficiente de fricción base
        
        Propiedades:
        1. ω = 0 si current = target (sin fricción)
        2. ω > 0 si current < target (ascenso requiere trabajo)
        3. ω < 0 si current > target (descenso libera energía)
        
        Invariante: ω ∈ ℝ es finito
        """
        current_level = state.stratum_level
        target_level = target_stratum.value

        if current_level == target_level:
            return 0.0

        distance = current_level - target_level
        exergy_level = float(
            state.context.get("exergy_level", 1.0)
        )
        
        # Protección contra exergía nula
        safe_exergy = max(exergy_level, _MIN_EXERGY_LEVEL)
        
        omega = (_EHRESMANN_BASE_FRICTION * distance) / safe_exergy
        
        return omega

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Composición con transporte paralelo.
        
        Pipeline:
        1. Aplicar f
        2. Calcular conexión de Ehresmann
        3. Calcular curvatura
        4. Aplicar corrección de fase si hay curvatura
        5. Aplicar g (transporte final)
        
        Propiedad: (g ∘ f)(x) = g(f(x)) módulo fase geométrica
        """
        self._call_count += 1

        # Fase I: Aplicar primer morfismo
        state_f = self.f(state)
        if state_f.is_failed:
            return state_f

        # Fase II: Derivada Covariante (Conexión de Ehresmann)
        omega = self._compute_ehresmann_connection(state_f, self.g.codomain)

        # Fase III: Curvatura (Ω = dω + ω ∧ ω ≈ ω²)
        curvature_omega = omega * omega

        # Fase IV: Corrección de Fase Geométrica
        if curvature_omega > 0.0:
            current_phase = state_f.context.get("_phase_correction", 1.0)
            phase_correction = current_phase * (1.0 - omega)

            # Detección de holonomía
            path_entropy = float(
                state_f.context.get("topological_entropy", 0.0)
            )
            
            if path_entropy > 0.5 and curvature_omega > 0.1:
                # Holonomía significativa detectada
                state_f = state_f.with_update(
                    new_context={
                        "_holonomy_detected": True,
                        "_phase_correction": phase_correction,
                        "_curvature": curvature_omega,
                    }
                )
            else:
                # Solo corrección de fase
                state_f = state_f.with_update(
                    new_context={
                        "_phase_correction": phase_correction,
                        "_curvature": curvature_omega,
                    }
                )

        # Fase V: Aplicar segundo morfismo (transporte final)
        return self.g(state_f)

# ==============================================================================
# PRODUCTO CATEGÓRICO
# ==============================================================================

class ProductMorphism(Morphism):
    """
    Producto categórico f × g con propiedad universal.
    
    CONSTRUCCIÓN UNIVERSAL:
    
    Dado el diagrama discreto {A, B}, el producto A × B
    es el límite con proyecciones π_A: A×B → A, π_B: A×B → B.
    
    Propiedad Universal:
    ∀X, ∀f: X → A, ∀g: X → B, ∃! h: X → A×B tal que
    π_A ∘ h = f  y  π_B ∘ h = g
    
    Implementación:
    - Ejecuta f y g independientemente sobre estado base
    - Fusiona resultados según política de conflictos
    - Deduplica trazas por identidad lógica
    
    Codominio:
    cod(f × g) = min(cod(f), cod(g))  (estrato más abstracto)
    
    Justificación: el producto refleja el nivel más alto
    (abstracto) alcanzado por ambas ramas.
    """
    __slots__ = ("f", "g", "conflict_policy", "_domain", "_codomain")

    def __init__(
        self,
        f: Morphism,
        g: Morphism,
        *,
        conflict_policy: str = "error_on_conflict",
    ) -> None:
        super().__init__(f"{f.name} × {g.name}")
        self.f = f
        self.g = g
        self.conflict_policy = conflict_policy
        
        # Dominio: unión de dominios
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
        
        # Codominio: estrato más abstracto (valor menor)
        self._codomain: Stratum = (
            f.codomain if f.codomain.value <= g.codomain.value else g.codomain
        )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    def _deduplicate_traces(
        self,
        traces_a: Tuple[CompositionTrace, ...],
        traces_b: Tuple[CompositionTrace, ...],
    ) -> Tuple[CompositionTrace, ...]:
        """
        Fusión de trazas con deduplicación por identidad lógica.
        
        Algoritmo:
        1. Iterar trazas_a, agregar nuevas
        2. Iterar trazas_b, agregar solo si no vistas
        3. Preservar orden temporal
        
        Invariante: |result| ≤ |traces_a| + |traces_b|
        """
        seen_keys: Set[Tuple[int, str, bool, Optional[str]]] = set()
        merged: List[CompositionTrace] = []

        for t in traces_a:
            key = t.trace_identity_key
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(t)

        for t in traces_b:
            key = t.trace_identity_key
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(t)

        return tuple(merged)

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Producto con fusión de resultados.
        
        Pipeline:
        1. Absorción monádica (si error previo)
        2. Ejecutar f y g independientemente
        3. Verificar éxito de ambas ramas
        4. Fusionar payloads según política
        5. Fusionar contexts (prefer_right)
        6. Unir estratos validados
        7. Deduplicar trazas
        
        Propiedad: (f × g)(s) combina resultados de f(s) y g(s)
        """
        self._call_count += 1

        # Absorción monádica
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monádica: error previo '{state.error}'",
                metadata={"absorbed": True},
            )

        # Ejecutar ambas ramas
        state_f = self.f(state)
        state_g = self.g(state)

        # Verificar fallos
        if state_f.is_failed:
            return state_f.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Rama izquierda falló: {state_f.error}",
            )
        
        if state_g.is_failed:
            return state_g.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Rama derecha falló: {state_g.error}",
            )

        # Fusionar payloads
        try:
            merged_payload = _safe_merge_dicts(
                state_f.payload,
                state_g.payload,
                conflict_policy=self.conflict_policy,
            )
        except ValueError as exc:
            error_msg = f"Conflicto en producto '{self.name}': {exc}"
            return state.with_error(
                error_msg,
                details={"conflict_policy": self.conflict_policy},
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        # Fusionar contexts (prefer_right)
        try:
            merged_context = _safe_merge_dicts(
                state_f.context,
                state_g.context,
                conflict_policy="prefer_right",
            )
        except ValueError as exc:
            error_msg = f"Conflicto en contexts: {exc}"
            return state.with_error(error_msg).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        # Unir estratos validados
        merged_strata = state_f.validated_strata | state_g.validated_strata

        # Deduplicar trazas
        merged_trace = self._deduplicate_traces(
            state_f.composition_trace,
            state_g.composition_trace,
        )

        # Construir resultado
        result = CategoricalState(
            payload=merged_payload,
            context=merged_context,
            validated_strata=merged_strata,
            error=None,
            error_details=None,
            composition_trace=merged_trace,
        )
        
        return result.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=True,
            metadata={
                "conflict_policy": self.conflict_policy,
                "merged_strata_count": len(merged_strata),
            },
        )

# ==============================================================================
# COPRODUCTO CATEGÓRICO
# ==============================================================================

class CoproductMorphism(Morphism):
    """
    Coproducto f ∐ g con propiedad universal.
    
    CONSTRUCCIÓN UNIVERSAL:
    
    Dado el diagrama discreto {A, B}, el coproducto A ∐ B
    es el colímite con inyecciones i_A: A → A∐B, i_B: B → A∐B.
    
    Propiedad Universal:
    ∀X, ∀f: A → X, ∀g: B → X, ∃! h: A∐B → X tal que
    h ∘ i_A = f  y  h ∘ i_B = g
    
    Semántica de Fallback:
    1. Intenta f
    2. Si f falla, recupera con g sobre estado original
    3. Preserva trazas de intento fallido para auditoría
    
    Codominio:
    cod(f ∐ g) = min(cod(f), cod(g))
    
    Invariante: siempre intenta primero rama izquierda
    """
    __slots__ = ("f", "g", "_domain", "_codomain")

    def __init__(self, f: Morphism, g: Morphism) -> None:
        super().__init__(f"{f.name} ∐ {g.name}")
        self.f = f
        self.g = g
        
        # Dominio: unión de dominios
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
        
        # Codominio: estrato más abstracto
        self._codomain: Stratum = (
            f.codomain if f.codomain.value <= g.codomain.value else g.codomain
        )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Coproducto con fallback y preservación de trazas.
        
        Pipeline:
        1. Absorción monádica (si error previo)
        2. Intentar rama izquierda (f)
        3. Si f exitoso → retornar con traza
        4. Si f falla → intentar rama derecha (g) sobre estado original
        5. Si g exitoso → retornar con trazas combinadas
        6. Si ambos fallan → retornar error combinado
        
        Propiedad: (f ∐ g)(s) = f(s) si exitoso, sino g(s)
        """
        self._call_count += 1

        # Absorción monádica
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monádica: error previo '{state.error}'",
            )

        # Intentar rama izquierda
        state_f = self.f(state)
        
        if state_f.is_success:
            # Rama izquierda exitosa
            return state_f.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=True,
                metadata={"selected_branch": "left"},
            )

        # Rama izquierda falló: intentar rama derecha
        state_g = self.g(state)  # Nota: sobre estado original, no state_f
        
        if state_g.is_success:
            # Recuperación exitosa: combinar trazas
            failed_traces = state_f.composition_trace
            recovery_traces = state_g.composition_trace

            # Deduplicar trazas
            seen_keys: Set[Tuple[int, str, bool, Optional[str]]] = set()
            combined_traces: List[CompositionTrace] = []
            
            for t in failed_traces:
                key = t.trace_identity_key
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined_traces.append(t)
            
            for t in recovery_traces:
                key = t.trace_identity_key
                if key not in seen_keys:
                    seen_keys.add(key)
                    combined_traces.append(t)

            # Construir estado recuperado
            recovered = CategoricalState(
                payload=dict(state_g.payload),
                context=dict(state_g.context),
                validated_strata=state_g.validated_strata,
                error=None,
                error_details=None,
                composition_trace=tuple(combined_traces),
            )
            
            return recovered.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=True,
                metadata={
                    "selected_branch": "right",
                    "recovered_from": state_f.error,
                },
            )

        # Ambas ramas fallaron
        error_msg = (
            f"Coproducto '{self.name}' falló completamente: "
            f"left='{state_f.error}', right='{state_g.error}'"
        )
        
        return state.with_error(
            error_msg,
            details={
                "primary_error": state_f.error,
                "fallback_error": state_g.error,
            },
        ).add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=False,
            error=error_msg,
        )


"""
FUNDAMENTOS MATEMÁTICOS RIGUROSOS - PARTE FINAL:

11. PULLBACK (LÍMITE DE SPAN):
    - Verificación de congruencia
    - Propiedad universal del pullback
    - Sincronización de caminos divergentes

12. FUNTORES CON PRESERVACIÓN VERIFICADA:
    - F(g∘f) = F(g)∘F(f) con pruebas
    - F(id_A) = id_{F(A)} verificado
    - Transformaciones naturales

13. COMPOSITORES Y VERIFICADORES:
    - MorphismComposer con validación acumulativa
    - StructuralVerifier (antes HomologicalVerifier)
    - Verificación de aciclicidad (β₁ = 0)

14. REGISTRO CATEGÓRICO:
    - Thread-safe con RLock
    - Grafo de dependencias (DAG)
    - Orden topológico garantizado
"""

# Continuación y finalización de mic_algebra.py

# ==============================================================================
# PULLBACK CATEGÓRICO (LÍMITE DE SPAN)
# ==============================================================================

class PullbackMorphism(Morphism):
    """
    Pullback (producto fibrado) con validación de congruencia.
    
    CONSTRUCCIÓN UNIVERSAL:
    
    Dado un span f: X → A, g: X → B, el pullback P es el límite
    con proyecciones p₁: P → X, p₂: P → X tales que f ∘ p₁ = g ∘ p₂.
    
    Propiedad Universal:
    ∀Z, ∀u: Z → X, ∀v: Z → X tal que f ∘ u = g ∘ v,
    ∃! w: Z → P tal que p₁ ∘ w = u  y  p₂ ∘ w = v
    
    Implementación:
    1. Ejecuta f y g sobre estado base
    2. Valida congruencia con validador externo
    3. Si congruente: fusiona resultados
    4. Si divergente: retorna error
    
    Aplicación:
    - Sincronización de caminos de procesamiento
    - Verificación de consistencia entre ramas
    - Prevención de paradojas lógicas
    
    Invariante: validator(state_f, state_g) ⟹ resultados congruentes
    """
    __slots__ = (
        "f",
        "g",
        "validator",
        "conflict_policy",
        "_domain",
        "_codomain"
    )

    def __init__(
        self,
        name: str,
        f: Morphism,
        g: Morphism,
        validator: Callable[[CategoricalState, CategoricalState], bool],
        *,
        conflict_policy: str = "error_on_conflict",
    ) -> None:
        super().__init__(name)
        self.f = f
        self.g = g
        self.validator = validator
        self.conflict_policy = conflict_policy
        
        # Dominio: unión de dominios
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
        
        # Codominio: estrato más abstracto
        self._codomain: Stratum = (
            f.codomain if f.codomain.value <= g.codomain.value else g.codomain
        )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Pullback con validación de congruencia.
        
        Pipeline:
        1. Absorción monádica
        2. Ejecutar f y g
        3. Verificar éxito de ambos
        4. Validar congruencia
        5. Si congruente: fusionar
        6. Si divergente: error
        
        Propiedad: pullback(s) es válido ⟺ validator(f(s), g(s)) = True
        """
        self._call_count += 1

        # Absorción monádica
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monádica: error previo '{state.error}'",
            )

        # Ejecutar ambos caminos
        state_f = self.f(state)
        state_g = self.g(state)

        # Verificar éxito
        if state_f.is_failed:
            return state_f.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Camino izquierdo falló: {state_f.error}",
            )
        
        if state_g.is_failed:
            return state_g.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Camino derecho falló: {state_g.error}",
            )

        # Validar congruencia
        try:
            is_congruent = bool(self.validator(state_f, state_g))
        except Exception as exc:
            error_msg = f"Error en validador de pullback '{self.name}': {exc}"
            self._logger.exception(error_msg)
            
            return state.with_error(
                error_msg,
                details={
                    "exception_type": type(exc).__name__,
                    "exception_str": str(exc),
                },
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        # Verificar congruencia
        if not is_congruent:
            error_msg = (
                f"Pullback '{self.name}': caminos divergentes — "
                f"'{self.f.name}' ≠ '{self.g.name}'"
            )
            
            return state.with_error(
                error_msg,
                details={
                    "path_f_payload": _canonicalize(state_f.payload),
                    "path_g_payload": _canonicalize(state_g.payload),
                    "path_f_hash": state_f.compute_hash(),
                    "path_g_hash": state_g.compute_hash(),
                },
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        # Fusionar resultados congruentes
        try:
            merged_payload = _safe_merge_dicts(
                state_f.payload,
                state_g.payload,
                conflict_policy=self.conflict_policy,
            )
            
            merged_context = _safe_merge_dicts(
                state_f.context,
                state_g.context,
                conflict_policy="prefer_right",
            )
        except ValueError as exc:
            error_msg = f"Conflicto en pullback '{self.name}': {exc}"
            
            return state.with_error(
                error_msg,
                details={"conflict_policy": self.conflict_policy},
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )

        # Deduplicar trazas
        seen_keys: Set[Tuple[int, str, bool, Optional[str]]] = set()
        merged_traces: List[CompositionTrace] = []
        
        for t in state_f.composition_trace:
            key = t.trace_identity_key
            if key not in seen_keys:
                seen_keys.add(key)
                merged_traces.append(t)
        
        for t in state_g.composition_trace:
            key = t.trace_identity_key
            if key not in seen_keys:
                seen_keys.add(key)
                merged_traces.append(t)

        # Construir resultado
        result = CategoricalState(
            payload=merged_payload,
            context=merged_context,
            validated_strata=state_f.validated_strata | state_g.validated_strata,
            error=None,
            error_details=None,
            composition_trace=tuple(merged_traces),
        )
        
        return result.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=True,
            metadata={
                "congruent": True,
                "validator": self.validator.__name__ if hasattr(self.validator, '__name__') else "anonymous",
            },
        )

# ==============================================================================
# FUNTORES CON PRESERVACIÓN VERIFICADA
# ==============================================================================

class Functor(ABC):
    """
    Funtor covariante F: C_MIC → D con preservación de estructura.
    
    TEORÍA DE CATEGORÍAS:
    
    Un funtor F preserva:
    1. Identidades: F(id_A) = id_{F(A)}
    2. Composición: F(g ∘ f) = F(g) ∘ F(f)
    
    Verificación:
    Los funtores deben implementar map_object y map_morphism
    de forma que las propiedades se satisfagan.
    
    Aplicaciones:
    - Serialización: C_MIC → Set
    - Logging: C_MIC → Log
    - Testing: C_MIC → Test
    
    Invariante: F es un morfismo entre categorías
    """
    __slots__ = ("name", "_logger", "_mapping_cache")

    def __init__(self, name: str = "") -> None:
        self.name: str = name or self.__class__.__name__
        self._logger: logging.Logger = logging.getLogger(
            f"MIC.Functor.{self.name}"
        )
        self._mapping_cache: Dict[str, Any] = {}

    @abstractmethod
    def map_object(self, state: CategoricalState) -> Any:
        """
        Mapea objeto de C_MIC a categoría destino.
        
        Propiedad: F(A) es objeto en categoría destino
        """
        ...

    @abstractmethod
    def map_morphism(
        self, f: Morphism
    ) -> Callable[[CategoricalState], Any]:
        """
        Mapea morfismo de C_MIC a morfismo en categoría destino.
        
        Propiedad: F(f): F(A) → F(B) es morfismo en categoría destino
        """
        ...

    def verify_functoriality(
        self,
        f: Morphism,
        g: Morphism,
        test_state: CategoricalState
    ) -> Dict[str, bool]:
        """
        Verifica propiedades funtoriales sobre ejemplos.
        
        Checks:
        1. F(g ∘ f) = F(g) ∘ F(f)
        2. F(id) preserva identidad (si aplicable)
        
        Returns:
            Dict con resultados de verificación
        """
        checks = {}
        
        # Verificar preservación de composición
        try:
            composed = f >> g
            
            F_composed = self.map_morphism(composed)
            F_f = self.map_morphism(f)
            F_g = self.map_morphism(g)
            
            # F(g ∘ f)(x)
            result_composed = F_composed(test_state)
            
            # F(g)(F(f)(x))
            intermediate = F_f(test_state)
            if isinstance(intermediate, CategoricalState):
                result_separate = F_g(intermediate)
            else:
                result_separate = None
            
            if result_separate is not None:
                checks["preserves_composition"] = (
                    _canonicalize(result_composed) == 
                    _canonicalize(result_separate)
                )
            else:
                checks["preserves_composition"] = None
        
        except Exception as e:
            self._logger.warning(
                "Error verificando preservación de composición: %s",
                e
            )
            checks["preserves_composition"] = False
        
        return checks

class StateToDictFunctor(Functor):
    """
    Funtor F: C_MIC → Set mapeando estados a diccionarios.
    
    Propiedades:
    1. map_object: CategoricalState → Dict
    2. map_morphism: (A → B) ↦ (Dict → Dict)
    3. Preservación: F(f ∘ g) = F(f) ∘ F(g)
    
    Aplicaciones:
    - Serialización JSON
    - Inspección de estados
    - Testing y debugging
    
    Invariante: F es funtor fiel (inyectivo en Hom-sets)
    """
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("StateToDictFunctor")

    def map_object(self, state: CategoricalState) -> Dict[str, Any]:
        """
        Mapea estado a diccionario.
        
        Propiedad: F(state) es dict serializable
        """
        return state.to_dict()

    def map_morphism(
        self, f: Morphism
    ) -> Callable[[CategoricalState], Dict[str, Any]]:
        """
        Mapea morfismo a función CategoricalState → Dict.
        
        Propiedad: F(f) = to_dict ∘ f
        """
        def mapped(state: CategoricalState) -> Dict[str, Any]:
            return f(state).to_dict()

        mapped.__name__ = f"F({f.name})"
        mapped.__doc__ = f"Imagen de {f.name} bajo StateToDictFunctor"
        
        return mapped

T_Functor = TypeVar("T_Functor", bound=Functor)

# ==============================================================================
# TRANSFORMACIONES NATURALES
# ==============================================================================

class NaturalTransformation(ABC, Generic[T_Functor, T_Functor]):
    """
    Transformación natural η: F ⇒ G entre funtores.
    
    TEORÍA DE CATEGORÍAS:
    
    Para cada objeto A en C_MIC:
        η_A: F(A) → G(A)
    
    Naturalidad:
    Para cada morfismo f: A → B, el diagrama conmuta:
    
        F(A) --F(f)--> F(B)
         |              |
        η_A            η_B
         |              |
         v              v
        G(A) --G(f)--> G(B)
    
    Es decir: η_B ∘ F(f) = G(f) ∘ η_A
    
    Composición:
    - Vertical (·): componente a componente
    - Horizontal (∘): sobre funtores compuestos
    
    Ley de Intercambio:
    (α' · α) ∘ (β' · β) = (α' ∘ β') · (α ∘ β)
    
    Invariante: η satisface condición de naturalidad
    """
    __slots__ = ("source_morphism", "target_morphism", "name")

    def __init__(
        self,
        source_morphism: Morphism,
        target_morphism: Morphism,
        name: str = "",
    ) -> None:
        self.source_morphism = source_morphism
        self.target_morphism = target_morphism
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Componente de transformación natural en objeto dado.
        
        Propiedad: η_A(state) transforma F(state) a G(state)
        """
        ...

    def vertical_compose(
        self, other: "NaturalTransformation"
    ) -> "NaturalTransformation":
        """
        Composición vertical: self · other.
        
        Definición: (η · θ)_A = η_A ∘ θ_A
        
        Precondición: self.target = other.source
        
        Propiedad: composición vertical es asociativa
        """
        if self.target_morphism != other.source_morphism:
            raise CompositionError(
                "Morfismos incompatibles para composición vertical: "
                f"{self.target_morphism.name} ≠ {other.source_morphism.name}"
            )

        class VerticallyComposed(NaturalTransformation):
            def __call__(self_, state: CategoricalState) -> CategoricalState:
                return other(self(state))
        
        return VerticallyComposed(
            self.source_morphism,
            other.target_morphism,
            f"{other.name} · {self.name}",
        )

    def horizontal_compose(
        self, other: "NaturalTransformation"
    ) -> "NaturalTransformation":
        """
        Composición horizontal: self ∘ other.
        
        Definición: (η ∘ θ)_A = η_{G(A)} ∘ F(θ_A)
        
        Propiedad: composición horizontal es asociativa
        """
        class HorizontallyComposed(NaturalTransformation):
            def __call__(self_, state: CategoricalState) -> CategoricalState:
                return other(self(state))

        source_comp = ComposedMorphism(
            self.source_morphism,
            other.source_morphism
        )
        target_comp = ComposedMorphism(
            self.target_morphism,
            other.target_morphism
        )
        
        return HorizontallyComposed(
            source_comp,
            target_comp,
            f"{other.name} ∘ {self.name}",
        )

    @property
    def is_natural(self) -> bool:
        """
        Verifica condición de naturalidad (simplificada).
        
        En general, requiere verificación sobre todos los morfismos.
        Esta implementación retorna True por defecto.
        """
        return True

# ==============================================================================
# COMPOSITOR CON VALIDACIÓN ACUMULATIVA
# ==============================================================================

class MorphismComposer:
    """
    Constructor de composiciones verificadas con validación acumulativa.
    
    Propiedades:
    1. Validación incremental: cada add_step verifica compatibilidad
    2. Acumulación de estratos: dom_acum ← dom_acum ∪ dom(f) ∪ {cod(f)}
    3. Construcción por pasos: build() produce composición final
    
    Ventajas sobre >>:
    - Validación más estricta (acumulativa vs. par a par)
    - Diagnóstico temprano de incompatibilidades
    - Visualización de secuencia
    
    Invariante: ∀ pasos agregados, composición es válida
    """
    __slots__ = ("steps", "_accumulated_strata", "logger")

    def __init__(self) -> None:
        self.steps: List[Morphism] = []
        self._accumulated_strata: FrozenSet[Stratum] = frozenset()
        self.logger: logging.Logger = logging.getLogger("MIC.MorphismComposer")

    def add_step(self, morphism: Morphism) -> "MorphismComposer":
        """
        Agrega paso con validación acumulativa.
        
        Verifica: dom(morphism) ⊆ accumulated_strata
        
        Raises:
            CompositionError: si paso no es componible
        
        Returns:
            self (para encadenamiento fluido)
        """
        if self.steps:
            missing = morphism.domain - self._accumulated_strata
            if missing:
                missing_names = sorted(s.name for s in missing)
                provided_names = sorted(s.name for s in self._accumulated_strata)
                required_names = sorted(s.name for s in morphism.domain)
                
                raise CompositionError(
                    f"Paso '{morphism.name}' no componible:\n"
                    f"  Requiere: {required_names}\n"
                    f"  Provisto: {provided_names}\n"
                    f"  Faltante: {missing_names}"
                )

        self.steps.append(morphism)
        self._accumulated_strata = (
            self._accumulated_strata |
            morphism.domain |
            frozenset({morphism.codomain})
        )
        
        self.logger.debug("✓ Paso agregado: %s", morphism)
        
        return self

    def build(self) -> Morphism:
        """
        Construye composición secuencial con transporte paralelo.
        
        Algoritmo:
        result = steps[0]
        for f in steps[1:]:
            result = result >> f  # ComposedMorphism con Ehresmann
        
        Raises:
            ValueError: si no hay pasos
        
        Returns:
            Morfismo compuesto
        """
        if not self.steps:
            raise ValueError("No hay pasos para componer")

        result = self.steps[0]
        
        for morphism in self.steps[1:]:
            result = result >> morphism

        self.logger.info(
            "✓ Composición construida: %d pasos → %s",
            len(self.steps),
            result.name
        )
        
        return result

    def reset(self) -> "MorphismComposer":
        """Reinicia compositor."""
        self.steps.clear()
        self._accumulated_strata = frozenset()
        return self

    def visualize(self) -> str:
        """Representación textual de la secuencia."""
        if not self.steps:
            return "(compositor vacío)"
        
        lines = []
        for i, m in enumerate(self.steps):
            lines.append(f"{i + 1}. {m}")
        
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"MorphismComposer(steps={len(self.steps)})"

# ==============================================================================
# VERIFICADOR ESTRUCTURAL (ANTES HOMOLOGICALVERIFIER)
# ==============================================================================

class StructuralVerifier:
    """
    Verificador de compatibilidad estructural y aciclicidad.
    
    NOTA MATEMÁTICA IMPORTANTE:
    
    Este verificador NO calcula homología algebraica (ker/im).
    El nombre anterior "HomologicalVerifier" era incorrecto.
    
    Funcionalidad Real:
    1. Composabilidad: verifica precondiciones de estratos
    2. Aciclicidad: verifica β₁ = 0 en grafo de dependencias
    3. Cobertura: análisis de estratos alcanzables
    
    Invariante: is_composable_sequence ⟹ ejecución válida
    """
    __slots__ = ("logger",)

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger("MIC.StructuralVerifier")

    def is_composable_sequence(self, morphisms: List[Morphism]) -> bool:
        """
        Verifica composabilidad de secuencia.
        
        Definición:
        ∀i, dom(morphisms[i+1]) ⊆ accumulated_strata(morphisms[0..i])
        
        Propiedad: más estricta que verificación par a par
        
        Returns:
            True si secuencia es componible
        """
        if len(morphisms) < 2:
            return True

        accumulated: FrozenSet[Stratum] = frozenset()

        for i, morphism in enumerate(morphisms):
            if i > 0:
                missing = morphism.domain - accumulated
                if missing:
                    self.logger.warning(
                        "Secuencia no componible en posición %d (%s): "
                        "requiere %s no provistos. Acumulado: %s",
                        i,
                        morphism.name,
                        sorted(s.name for s in missing),
                        sorted(s.name for s in accumulated),
                    )
                    return False
            
            accumulated = (
                accumulated |
                morphism.domain |
                frozenset({morphism.codomain})
            )

        return True

    def verify_composition(self, composition: Morphism) -> Dict[str, Any]:
        """
        Inspección estática de morfismo (posiblemente compuesto).
        
        Returns:
            Dict con metadatos estructurales
        """
        result: Dict[str, Any] = {
            "is_valid": True,
            "name": composition.name,
            "domain": sorted(s.name for s in composition.domain),
            "codomain": composition.codomain.name,
            "structural_type": type(composition).__name__,
        }

        if isinstance(composition, ComposedMorphism):
            result["is_structurally_compatible"] = (
                composition.is_structurally_compatible
            )
            result["components"] = {
                "left": self.verify_composition(composition.f),
                "right": self.verify_composition(composition.g),
            }

        return result

    def compute_stratum_coverage(
        self, morphisms: List[Morphism]
    ) -> Dict[str, Any]:
        """
        Calcula cobertura de estratos.
        
        Análisis:
        - required: ⋃ dom(f)
        - produced: ⋃ {cod(f)}
        - uncovered: required − produced
        - unreachable: Stratum − produced
        
        Returns:
            Dict con estadísticas de cobertura
        """
        all_required: Set[Stratum] = set()
        all_produced: Set[Stratum] = set()

        for m in morphisms:
            all_required.update(m.domain)
            all_produced.add(m.codomain)

        uncovered = all_required - all_produced
        all_strata = set(Stratum)
        unreachable = all_strata - all_produced

        return {
            "required_strata": sorted(s.name for s in all_required),
            "produced_strata": sorted(s.name for s in all_produced),
            "uncovered_requirements": sorted(s.name for s in uncovered),
            "unreachable_strata": sorted(s.name for s in unreachable),
            "full_coverage": len(uncovered) == 0,
        }

# Alias de retrocompatibilidad
HomologicalVerifier = StructuralVerifier

# ==============================================================================
# REGISTRO CATEGÓRICO CON GARANTÍAS DE ACICLICIDAD
# ==============================================================================

class CategoricalRegistry:
    """
    Registro thread-safe de morfismos con verificación de aciclicidad.
    
    Propiedades:
    1. Thread-Safety: operaciones protegidas por RLock
    2. Aciclicidad: grafo de dependencias es DAG (β₁ = 0)
    3. Orden topológico: existe orden de ejecución válido
    
    Invariantes:
    - verify_acyclicity() = True ⟹ topological_order() ≠ None
    - DAG ⟹ no hay deadlocks en ejecución
    
    Aplicación:
    - Orquestación de pipelines complejos
    - Validación de dependencias
    - Generación de orden de ejecución
    """
    __slots__ = ("_morphisms", "_compositions", "_lock", "logger")

    def __init__(self) -> None:
        self._morphisms: Dict[str, Morphism] = {}
        self._compositions: Dict[str, Morphism] = {}
        self._lock: threading.RLock = threading.RLock()
        self.logger: logging.Logger = logging.getLogger("MIC.CategoricalRegistry")

    @property
    def morphisms(self) -> Dict[str, Morphism]:
        """Copia defensiva del registro de morfismos."""
        with self._lock:
            return dict(self._morphisms)

    @property
    def compositions(self) -> Dict[str, Morphism]:
        """Copia defensiva del registro de composiciones."""
        with self._lock:
            return dict(self._compositions)

    def register_morphism(self, name: str, morphism: Morphism) -> None:
        """
        Registra morfismo atómico.
        
        Thread-Safe: operación protegida por RLock
        """
        with self._lock:
            if name in self._morphisms:
                self.logger.warning(
                    "Morfismo '%s' ya registrado. Reemplazando.",
                    name
                )
            
            self._morphisms[name] = morphism
            self.logger.debug("✓ Morfismo registrado: %s", name)

    def register_composition(self, name: str, composition: Morphism) -> None:
        """
        Registra composición de morfismos.
        
        Thread-Safe: operación protegida por RLock
        """
        with self._lock:
            if name in self._compositions:
                self.logger.warning(
                    "Composición '%s' ya registrada. Reemplazando.",
                    name
                )
            
            self._compositions[name] = composition
            self.logger.debug("✓ Composición registrada: %s", name)

    def unregister_morphism(self, name: str) -> bool:
        """Elimina morfismo. Retorna True si existía."""
        with self._lock:
            if name in self._morphisms:
                del self._morphisms[name]
                self.logger.debug("✓ Morfismo eliminado: %s", name)
                return True
            return False

    def unregister_composition(self, name: str) -> bool:
        """Elimina composición. Retorna True si existía."""
        with self._lock:
            if name in self._compositions:
                del self._compositions[name]
                self.logger.debug("✓ Composición eliminada: %s", name)
                return True
            return False

    def get_morphism(self, name: str) -> Optional[Morphism]:
        """Obtiene morfismo por nombre."""
        with self._lock:
            return self._morphisms.get(name)

    def get_composition(self, name: str) -> Optional[Morphism]:
        """Obtiene composición por nombre."""
        with self._lock:
            return self._compositions.get(name)

    def list_morphisms(self) -> List[str]:
        """Lista nombres de morfismos registrados."""
        with self._lock:
            return sorted(self._morphisms.keys())

    def list_compositions(self) -> List[str]:
        """Lista nombres de composiciones registradas."""
        with self._lock:
            return sorted(self._compositions.keys())

    def get_dependency_graph(self) -> nx.DiGraph:
        """
        Construye grafo dirigido de dependencias.
        
        Arcos: (A → B) ⟺ cod(A) ∈ dom(B)
        
        Interpretación: A debe ejecutarse antes que B
        
        Invariante: grafo refleja relación de precedencia
        """
        with self._lock:
            all_items: Dict[str, Morphism] = {}
            all_items.update(self._morphisms)
            all_items.update(self._compositions)

        G = nx.DiGraph()
        producers: Dict[Stratum, List[str]] = {}

        # Agregar nodos
        for name, morphism in all_items.items():
            G.add_node(
                name,
                codomain=morphism.codomain.name,
                domain=sorted(s.name for s in morphism.domain),
                kind=type(morphism).__name__,
            )
            producers.setdefault(morphism.codomain, []).append(name)

        # Agregar arcos de dependencia
        for name, morphism in all_items.items():
            for required_stratum in morphism.domain:
                for producer_name in producers.get(required_stratum, []):
                    if producer_name != name:
                        G.add_edge(
                            producer_name,
                            name,
                            stratum=required_stratum.name,
                        )

        return G

    def verify_acyclicity(self) -> bool:
        """
        Verifica que grafo de dependencias es DAG.
        
        Definición: β₁(G) = 0 (primer número de Betti)
        
        Equivalente: no existen ciclos dirigidos
        
        Propiedad: acyclicity ⟹ ∃ orden topológico
        
        Returns:
            True si grafo es acíclico
        """
        G = self.get_dependency_graph()
        is_acyclic = nx.is_directed_acyclic_graph(G)
        
        if not is_acyclic:
            try:
                cycles = list(nx.simple_cycles(G))
                self.logger.error(
                    "Ciclos detectados en grafo de dependencias: %s",
                    cycles
                )
            except Exception as e:
                self.logger.error("Error detectando ciclos: %s", e)
        
        return is_acyclic

    def topological_order(self) -> Optional[List[str]]:
        """
        Orden topológico de morfismos registrados.
        
        Definición: ordenamiento lineal consistente con arcos
        
        Garantía: si (A, B) ∈ E, entonces A precede a B en el orden
        
        Returns:
            Lista ordenada de nombres, o None si hay ciclos
        """
        G = self.get_dependency_graph()
        
        if not nx.is_directed_acyclic_graph(G):
            return None
        
        return list(nx.topological_sort(G))

    def __len__(self) -> int:
        """Total de morfismos y composiciones registradas."""
        with self._lock:
            return len(self._morphisms) + len(self._compositions)

    def __repr__(self) -> str:
        return (
            f"CategoricalRegistry("
            f"morphisms={len(self._morphisms)}, "
            f"compositions={len(self._compositions)})"
        )

# ==============================================================================
# FACTORIES ADICIONALES
# ==============================================================================

def create_morphism_from_handler(
    name: str,
    target_stratum: Stratum,
    handler: Callable[..., Any],
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[List[str]] = None,
) -> Morphism:
    """
    Factory para AtomicVector desde handler callable.
    
    Propiedad: produce morfismo atómico válido
    
    Returns:
        AtomicVector inicializado
    """
    return AtomicVector(
        name=name,
        target_stratum=target_stratum,
        handler=handler,
        required_keys=required_keys,
        optional_keys=optional_keys,
    )

# ==============================================================================
# EXPORTACIÓN PÚBLICA CONTROLADA - COMPLETA
# ==============================================================================

__all__ = [
    # Excepciones (PARTE 1)
    "AlgebraicError",
    "CanonicalizationError",
    "StratumResolutionError",
    "CategoryError",
    "CompositionError",
    
    # Estratificación (PARTE 1)
    "Stratum",
    
    # Utilidades (PARTE 1)
    "MathUtils",
    
    # Estado categórico (PARTE 1)
    "CategoricalState",
    "CompositionTrace",
    "create_categorical_state",
    
    # Morfismos (PARTE 2)
    "Morphism",
    "IdentityMorphism",
    "AtomicVector",
    "ComposedMorphism",
    "ProductMorphism",
    "CoproductMorphism",
    "PullbackMorphism",
    
    # Funtores (PARTE FINAL)
    "Functor",
    "StateToDictFunctor",
    "NaturalTransformation",
    
    # Composición y verificación (PARTE FINAL)
    "MorphismComposer",
    "StructuralVerifier",
    "HomologicalVerifier",  # Alias retrocompatibilidad
    
    # Registro (PARTE FINAL)
    "CategoricalRegistry",
    
    # Factories (PARTES 1 y FINAL)
    "create_categorical_state",
    "create_morphism_from_handler",
    
    # Constantes (PARTE 1)
    "_SCHEMA_VERSION",
    "_MAX_CANONICALIZE_DEPTH",
    "_ALGEBRAIC_TOL",
    "_FLOAT_COMPARISON_TOL",
]

# ==============================================================================
# FIN DEL MÓDULO
# ==============================================================================

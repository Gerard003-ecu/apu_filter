from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
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
)

import networkx as nx

# ============================================================================
# CONSTANTES DE ESQUEMA
# ============================================================================

_SCHEMA_VERSION: str = "1.0.0"
_MAX_CANONICALIZE_DEPTH: int = 64

# ============================================================================
# ESTRATIFICACIÓN (DIKW Hierarchy)
# ============================================================================

try:
    from app.core.schemas import Stratum
except ImportError:

    @unique
    class Stratum(IntEnum):
        """
        Estratificación DIKW (Data → Information → Knowledge → Wisdom).

        Convención ordinal: valor numérico MAYOR ↔ estrato más concreto.
        WISDOM (0) es el más abstracto; PHYSICS (3) el más concreto.

        Relación de dependencia:
            Un estrato s depende de todos los estratos con valor > s.value,
            formando un orden parcial consistente con la filtración DIKW.
        """

        WISDOM = 0
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3

        def requires(self) -> FrozenSet[Stratum]:
            """
            Requisitos topológicos: estrato s depende de todos los
            estratos estrictamente más concretos (valor mayor).

            Esto define un preorden sobre Ob(C_MIC) compatible con
            la filtración DIKW.
            """
            return frozenset(s for s in Stratum if s.value > self.value)

        @property
        def level(self) -> int:
            """Nivel numérico del estrato."""
            return self.value


logger = logging.getLogger("MIC.Algebra")

# ============================================================================
# UTILIDADES INTERNAS
# ============================================================================


class _CanonicalizationError(Exception):
    """Error durante la canonicalización de un valor."""


def _canonicalize(value: Any, *, _depth: int = 0) -> Any:
    """
    Convierte un objeto a representación serializable y canónica.

    Propiedades garantizadas:
    - Determinismo: misma entrada → misma salida.
    - Protección contra recursión: profundidad máxima acotada.
    - Ordenamiento estable para colecciones no ordenadas.

    Raises:
        _CanonicalizationError: si se excede la profundidad máxima.
    """
    if _depth > _MAX_CANONICALIZE_DEPTH:
        raise _CanonicalizationError(
            f"Profundidad de canonicalización excedida ({_MAX_CANONICALIZE_DEPTH}). "
            f"Posible referencia circular en: {type(value).__name__}"
        )

    next_depth = _depth + 1

    if value is None:
        return value

    # Check for Stratum enum
    # In some test setups, value is an instance of a dynamically imported
    # Stratum that does not share identity with the local Stratum.
    # So we check the class name or try to access '.name'.
    if getattr(type(value), "__name__", "") == "Stratum" or isinstance(value, Stratum):
        return {"__stratum__": getattr(value, "name", str(value))}

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v, _depth=next_depth)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize(v, _depth=next_depth) for v in value]

    if isinstance(value, (set, frozenset)):
        # Los elementos canonicalizados se convierten a repr para
        # ordenamiento estable cuando no son naturalmente comparables.
        canonicalized = [_canonicalize(v, _depth=next_depth) for v in value]
        try:
            return sorted(canonicalized)
        except TypeError:
            return sorted(canonicalized, key=repr)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _canonicalize(value.to_dict(), _depth=next_depth)
        except (Exception, RecursionError):
            return repr(value)

    if hasattr(value, "to_json") and callable(value.to_json):
        try:
            return value.to_json()
        except (Exception, RecursionError):
            return repr(value)

    return repr(value)


_VALID_CONFLICT_POLICIES = frozenset(
    {"prefer_right", "prefer_left", "error_on_conflict"}
)


def _safe_merge_dicts(
    left: Dict[str, Any],
    right: Dict[str, Any],
    *,
    conflict_policy: str = "prefer_right",
) -> Dict[str, Any]:
    """
    Fusiona dos diccionarios con política explícita de resolución de conflictos.

    Semántica:
    - prefer_right: ante conflicto, el valor de `right` prevalece.
    - prefer_left: ante conflicto, el valor de `left` prevalece.
    - error_on_conflict: lanza ValueError si existe conflicto.

    Un conflicto se define como: misma clave, valores distintos.

    Raises:
        ValueError: si la política es inválida o si hay conflicto
                    bajo error_on_conflict.
    """
    if conflict_policy not in _VALID_CONFLICT_POLICIES:
        raise ValueError(
            f"conflict_policy inválida: '{conflict_policy}'. "
            f"Opciones: {sorted(_VALID_CONFLICT_POLICIES)}"
        )

    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            if conflict_policy == "error_on_conflict":
                raise ValueError(
                    f"Conflicto al fusionar clave '{key}': "
                    f"izq={merged[key]!r}, der={value!r}"
                )
            if conflict_policy == "prefer_left":
                continue
        merged[key] = value
    return merged


def _copy_trace(
    trace: Sequence[CompositionTrace],
) -> Tuple[CompositionTrace, ...]:
    """
    Copia defensiva de secuencia de trazas a tupla inmutable.
    """
    return tuple(trace)


# ============================================================================
# TRAZA DE AUDITORÍA
# ============================================================================


@dataclass(frozen=True, eq=True)
class CompositionTrace:
    """
    Traza de auditoría inmutable para composiciones de morfismos.

    Cada instancia registra un paso atómico en la ejecución de un
    pipeline categórico, incluyendo dominio, codominio, éxito/fallo,
    y metadatos opcionales.

    La igualdad se define por (step_number, morphism_name, success, error)
    para deduplicación determinista en fusiones de trazas.
    """

    step_number: int
    morphism_name: str
    input_domain: FrozenSet[Stratum]
    output_codomain: Stratum
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def trace_identity_key(self) -> Tuple[int, str, bool, Optional[str]]:
        """
        Clave de identidad para deduplicación.

        Dos trazas con la misma clave de identidad representan
        el mismo evento lógico (posiblemente con timestamps distintos).
        """
        return (self.step_number, self.morphism_name, self.success, self.error)

    def to_dict(self) -> Dict[str, Any]:
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


# ============================================================================
# OBJETO FUNDAMENTAL DE LA CATEGORÍA
# ============================================================================


@dataclass(frozen=True)
class CategoricalState:
    """
    Objeto fundamental de la categoría C_MIC.

    Invariantes de diseño:
    1. Inmutabilidad efectiva: frozen=True + tupla para trazas.
    2. Toda mutación produce un nuevo objeto (estilo funcional).
    3. El hash criptográfico es reproducible y determinista.
    4. La serialización incluye versión de esquema para
       compatibilidad futura.

    Observación sobre inmutabilidad:
        `payload` y `context` son dict (mutables), pero el contrato
        de uso exige no modificarlos tras construcción. Los métodos
        with_* producen copias defensivas.
    """

    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validated_strata: FrozenSet[Stratum] = field(default_factory=frozenset)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    composition_trace: Tuple[CompositionTrace, ...] = field(default_factory=tuple)

    @property
    def is_success(self) -> bool:
        """Estado sin error registrado."""
        return self.error is None

    @property
    def is_failed(self) -> bool:
        """Estado con error registrado."""
        return not self.is_success

    @property
    def stratum_level(self) -> int:
        """
        Nivel de estrato más abstracto alcanzado.

        Retorna el valor mínimo (más abstracto) entre los estratos
        validados, o PHYSICS.value si no hay validaciones.
        """
        if not self.validated_strata:
            return Stratum.PHYSICS.value
        return min(s.value for s in self.validated_strata)

    @property
    def accumulated_strata(self) -> FrozenSet[Stratum]:
        """Alias semántico para validated_strata."""
        return self.validated_strata

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
        Produce un nuevo estado con actualizaciones aplicadas.

        La fusión respeta las políticas de conflicto especificadas.
        El error y error_details se preservan del estado original
        (usar with_error para modificarlos).
        """
        # -- Payload --
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

        # -- Context --
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

        # -- Strata --
        updated_strata = self.validated_strata
        if new_stratum is not None:
            updated_strata = updated_strata | frozenset({new_stratum})

        return CategoricalState(
            payload=updated_payload,
            context=updated_context,
            validated_strata=updated_strata,
            error=self.error,
            error_details=dict(self.error_details) if self.error_details else None,
            composition_trace=_copy_trace(self.composition_trace),
        )

    def with_error(
        self,
        error_msg: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> CategoricalState:
        """
        Produce un nuevo estado en condición de error.

        El payload y context se preservan para diagnóstico post-mortem.
        """
        return CategoricalState(
            payload=dict(self.payload),
            context=dict(self.context),
            validated_strata=self.validated_strata,
            error=error_msg,
            error_details=dict(details) if details else None,
            composition_trace=_copy_trace(self.composition_trace),
        )

    def clear_error(self) -> CategoricalState:
        """
        Produce un nuevo estado idéntico pero sin error.

        Útil para recuperación explícita en coproductos.
        """
        return CategoricalState(
            payload=dict(self.payload),
            context=dict(self.context),
            validated_strata=self.validated_strata,
            error=None,
            error_details=None,
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
        Produce un nuevo estado con una entrada de traza adicional.
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
            error_details=dict(self.error_details) if self.error_details else None,
            composition_trace=self.composition_trace + (trace_entry,),
        )

    def compute_hash(self) -> str:
        """
        Hash SHA-256 determinista del estado completo.

        Incluye versión de esquema para detectar incompatibilidades
        ante cambios de formato de serialización.
        """
        serialized = json.dumps(
            {
                "__schema_version__": _SCHEMA_VERSION,
                "payload": _canonicalize(self.payload),
                "context": _canonicalize(self.context),
                "validated_strata": sorted(s.name for s in self.validated_strata),
                "error": self.error,
                "error_details": _canonicalize(self.error_details),
                "composition_trace": [
                    _canonicalize(t.to_dict()) for t in self.composition_trace
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa con versión de esquema."""
        return {
            "__schema_version__": _SCHEMA_VERSION,
            "payload": _canonicalize(self.payload),
            "context": _canonicalize(self.context),
            "validated_strata": sorted(s.name for s in self.validated_strata),
            "error": self.error,
            "error_details": _canonicalize(self.error_details),
            "composition_trace": [t.to_dict() for t in self.composition_trace],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CategoricalState:
        """
        Deserialización con validación estructural.

        Raises:
            KeyError: si faltan campos obligatorios en trazas.
            ValueError: si un nombre de estrato no es válido.
        """
        # Validación de versión de esquema (advertencia, no fallo)
        schema_ver = data.get("__schema_version__")
        if schema_ver and schema_ver != _SCHEMA_VERSION:
            logger.warning(
                "Versión de esquema diferente: esperada=%s, encontrada=%s",
                _SCHEMA_VERSION,
                schema_ver,
            )

        strata_names = data.get("validated_strata", [])
        try:
            strata = frozenset(Stratum[s] for s in strata_names)
        except KeyError as exc:
            raise ValueError(f"Estrato inválido en from_dict: {exc}") from exc

        raw_traces = data.get("composition_trace", [])
        traces: List[CompositionTrace] = []
        for i, t in enumerate(raw_traces):
            # Validación de campos requeridos
            required_trace_fields = {
                "step", "morphism", "domain", "codomain", "success"
            }
            missing_fields = required_trace_fields - set(t.keys())
            if missing_fields:
                raise KeyError(
                    f"Traza #{i} carece de campos: {sorted(missing_fields)}"
                )
            try:
                traces.append(
                    CompositionTrace(
                        step_number=int(t["step"]),
                        morphism_name=str(t["morphism"]),
                        input_domain=frozenset(Stratum[s] for s in t["domain"]),
                        output_codomain=Stratum[t["codomain"]],
                        success=bool(t["success"]),
                        error=t.get("error"),
                        timestamp=float(t.get("timestamp", 0.0)),
                        metadata=t.get("metadata"),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Error al reconstruir traza #{i}: {exc}"
                ) from exc

        return cls(
            payload=dict(data.get("payload", {})),
            context=dict(data.get("context", {})),
            validated_strata=strata,
            error=data.get("error"),
            error_details=data.get("error_details"),
            composition_trace=tuple(traces),
        )


# ============================================================================
# MORFISMOS — CLASE BASE
# ============================================================================


class Morphism(ABC):
    """
    Clase base abstracta para morfismos en la categoría C_MIC.

    Un morfismo f: A → B mapea objetos CategoricalState con
    precondiciones de estrato (domain) a un estrato objetivo (codomain).

    Axiomas categóricos satisfechos:
    - Identidad: IdentityMorphism actúa como id en composición.
    - Asociatividad: (f >> g) >> h ≡ f >> (g >> h) semánticamente.
    - Composición tipada: verificable estáticamente vía can_compose_with.
    """

    def __init__(self, name: str = ""):
        self.name: str = name or self.__class__.__name__
        self._logger: logging.Logger = logging.getLogger(
            f"MIC.Morphism.{self.name}"
        )

    @property
    @abstractmethod
    def domain(self) -> FrozenSet[Stratum]:
        """Estratos requeridos como precondición."""
        ...

    @property
    @abstractmethod
    def codomain(self) -> Stratum:
        """Estrato producido/validado por este morfismo."""
        ...

    def can_compose_with(self, other: Morphism) -> bool:
        """
        Verifica si `other` puede ejecutarse después de `self`
        bajo semántica de acumulación de estratos:

            dom(other) ⊆ dom(self) ∪ {cod(self)}

        Esta condición garantiza que todos los estratos requeridos
        por `other` están provistos por la ejecución previa de `self`.
        """
        provided = self.domain | frozenset({self.codomain})
        return other.domain.issubset(provided)

    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica el morfismo al estado. Debe ser puro (sin efectos
        laterales observables más allá del logging).
        """
        ...

    def __rshift__(self, other: Morphism) -> Morphism:
        """Operador >>: composición secuencial (self luego other)."""
        return ComposedMorphism(self, other)

    def __mul__(self, other: Morphism) -> Morphism:
        """Operador *: producto categórico (paralelo con fusión)."""
        return ProductMorphism(self, other)

    def __or__(self, other: Morphism) -> Morphism:
        """Operador |: coproducto (fallback)."""
        return CoproductMorphism(self, other)

    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return f"{self.name}: ({domain_str}) → {self.codomain.name}"


# ============================================================================
# MORFISMO IDENTIDAD
# ============================================================================


class IdentityMorphism(Morphism):
    """
    Morfismo identidad para un estrato dado.

    Satisface: id_s ∘ f = f  y  f ∘ id_s = f
    para todo f con cod(f) = s o dom(f) ∋ s respectivamente.
    """

    def __init__(self, stratum: Stratum):
        super().__init__(f"id_{stratum.name}")
        self._stratum = stratum

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset({self._stratum})

    @property
    def codomain(self) -> Stratum:
        return self._stratum

    def __call__(self, state: CategoricalState) -> CategoricalState:
        return state.add_trace(
            self.name,
            self.domain,
            self.codomain,
            success=True,
            metadata={"identity": True},
        )


# ============================================================================
# MORFISMO ATÓMICO (ADAPTADOR LEGACY)
# ============================================================================


class AtomicVector(Morphism):
    """
    Morfismo atómico que envuelve un handler callable legacy
    e integra su ejecución en la semántica categórica.

    El handler recibe kwargs filtrados por required_keys ∪ optional_keys
    y debe retornar:
    - dict con claves de resultado (opcionalmente "success", "error"), o
    - cualquier valor escalar (se empaqueta como {name_result: valor}).

    Semántica monadal: si el estado entrante tiene error, se absorbe
    sin ejecutar el handler (short-circuit).
    """

    def __init__(
        self,
        name: str,
        target_stratum: Stratum,
        handler: Callable[..., Any],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
    ):
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
        """Absorción monadal: propaga error previo sin ejecutar."""
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
        Valida precondiciones de estrato.

        Retorna None si la validación pasa, o un estado de error
        si falla (a menos que force_override esté activo).
        """
        force_override = bool(state.context.get("force_override", False))
        if force_override:
            return None

        missing = self.domain - state.validated_strata
        if not missing:
            return None

        missing_names = sorted(s.name for s in missing)
        error_msg = (
            f"Violación de dominio en '{self.name}': "
            f"faltan estratos {missing_names}"
        )
        self._logger.warning(error_msg)
        return state.with_error(
            error_msg,
            details={"missing_strata": missing_names},
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
        Valida presencia de claves requeridas en el payload.

        Retorna None si todas las claves están presentes, o un
        estado de error si faltan.
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
                "available_keys": sorted(map(str, state.payload.keys())),
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
        Ejecuta el handler con kwargs filtrados y procesa el resultado.
        """
        allowed_keys = self._required_keys | self._optional_keys
        kwargs = {
            k: v for k, v in state.payload.items() if k in allowed_keys
        }

        try:
            result = self._handler(**kwargs)
        except Exception as exc:
            error_msg = f"Excepción en handler de '{self.name}': {exc}"
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

        return self._process_result(state, result)

    def _process_result(
        self, state: CategoricalState, result: Any
    ) -> CategoricalState:
        """
        Procesa el resultado del handler y produce el estado de salida.
        """
        if isinstance(result, dict):
            success = bool(result.get("success", True))
            if not success:
                error_msg = result.get("error", f"Fallo interno en {self.name}")
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

            # Filtrar claves de control del resultado
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
        )

    def __call__(self, state: CategoricalState) -> CategoricalState:
        # 1. Absorción monadal
        if state.is_failed:
            return self._absorb_error(state)

        # 2. Validación de dominio
        domain_error = self._validate_domain(state)
        if domain_error is not None:
            return domain_error

        # 3. Validación de claves requeridas
        keys_error = self._validate_required_keys(state)
        if keys_error is not None:
            return keys_error

        # 4. Ejecución del handler
        return self._execute_handler(state)


# ============================================================================
# COMPOSICIÓN CATEGÓRICA
# ============================================================================


class ComposedMorphism(Morphism):
    """
    Composición categórica: g ∘ f, escrita f >> g.

    Semántica de dominio:
        dom(g ∘ f) = dom(f) ∪ (dom(g) − {cod(f)} − dom(f))

    El dominio del compuesto incluye todas las precondiciones de f,
    más aquellas precondiciones de g que no son satisfechas ni por
    cod(f) ni por dom(f) (cierre transitivo).

    Propiedad de compatibilidad estructural:
        g es componible con f ⟺ dom(g) ⊆ dom(f) ∪ {cod(f)}
    """

    def __init__(self, f: Morphism, g: Morphism):
        super().__init__(f"{f.name} >> {g.name}")
        self.f = f
        self.g = g

        provided_by_f = f.domain | frozenset({f.codomain})
        self._is_structurally_compatible: bool = g.domain.issubset(provided_by_f)

        # Dominio efectivo: todo lo que f necesita + lo que g necesita
        # que f no provee
        self._domain: FrozenSet[Stratum] = f.domain | (g.domain - provided_by_f)
        self._codomain: Stratum = g.codomain

        if not self._is_structurally_compatible:
            unsatisfied = g.domain - provided_by_f
            logger.warning(
                "Composición '%s' no estrictamente compatible: "
                "dom(g) requiere %s no provistos por f. "
                "Estos se elevan al dominio del compuesto.",
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
        True si la composición es estrictamente compatible,
        es decir, dom(g) ⊆ dom(f) ∪ {cod(f)}.
        """
        return self._is_structurally_compatible

    def __call__(self, state: CategoricalState) -> CategoricalState:
        state_f = self.f(state)
        if state_f.is_failed:
            return state_f
        return self.g(state_f)


# ============================================================================
# PRODUCTO CATEGÓRICO
# ============================================================================


class ProductMorphism(Morphism):
    """
    Producto categórico práctico: f × g.

    Ejecuta f y g independientemente sobre el mismo estado base
    y fusiona los resultados.

    Semántica de codominio:
        cod(f × g) = min(cod(f), cod(g))  (estrato más abstracto alcanzado)

    Justificación: el producto debe reflejar el nivel más alto
    (abstracto) que ambas ramas alcanzan conjuntamente, ya que
    la semántica DIKW ordena WISDOM < STRATEGY < TACTICS < PHYSICS.
    """

    def __init__(
        self,
        f: Morphism,
        g: Morphism,
        *,
        conflict_policy: str = "error_on_conflict",
    ):
        super().__init__(f"{f.name} × {g.name}")
        self.f = f
        self.g = g
        self.conflict_policy = conflict_policy
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
        # min por valor = estrato más abstracto
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
        Fusiona dos secuencias de trazas eliminando duplicados
        por clave de identidad lógica.
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
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monadal: error previo '{state.error}'",
                metadata={"absorbed": True},
            )

        state_f = self.f(state)
        state_g = self.g(state)

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
            error_msg = (
                f"Conflicto al fusionar producto '{self.name}': {exc}"
            )
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

        merged_strata = state_f.validated_strata | state_g.validated_strata
        merged_trace = self._deduplicate_traces(
            state_f.composition_trace,
            state_g.composition_trace,
        )

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
            metadata={"conflict_policy": self.conflict_policy},
        )


# ============================================================================
# COPRODUCTO CATEGÓRICO
# ============================================================================


class CoproductMorphism(Morphism):
    """
    Coproducto práctico: f ∐ g.

    Semántica: intenta f; si falla, recupera con g sobre el estado
    original (no sobre el estado fallido de f).

    La traza preserva memoria del intento fallido para auditoría.

    Codominio: min(cod(f), cod(g)) — consistente con ProductMorphism.
    """

    def __init__(self, f: Morphism, g: Morphism):
        super().__init__(f"{f.name} ∐ {g.name}")
        self.f = f
        self.g = g
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
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
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monadal: error previo '{state.error}'",
            )

        # Rama primaria
        state_f = self.f(state)
        if state_f.is_success:
            return state_f.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=True,
                metadata={"selected_branch": "left"},
            )

        # Rama de recuperación (desde estado original, no desde state_f)
        state_g = self.g(state)
        if state_g.is_success:
            # Fusionar trazas: preservar memoria del intento fallido
            failed_traces = state_f.composition_trace
            recovery_traces = state_g.composition_trace

            # Deduplicación por identidad lógica
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
            f"izq='{state_f.error}', der='{state_g.error}'"
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


# ============================================================================
# PULLBACK CATEGÓRICO
# ============================================================================


class PullbackMorphism(Morphism):
    """
    Pullback práctico de dos caminos con validación de congruencia.

    Dados f: X → A y g: X → B, el pullback verifica que ambos
    caminos producen resultados congruentes (según un validador externo)
    antes de fusionar los estados resultantes.

    El dominio es la unión de dominios de ambos caminos,
    exigiendo que el estado fuente satisfaga ambos.
    """

    def __init__(
        self,
        name: str,
        f: Morphism,
        g: Morphism,
        validator: Callable[[CategoricalState, CategoricalState], bool],
        *,
        conflict_policy: str = "error_on_conflict",
    ):
        super().__init__(name)
        self.f = f
        self.g = g
        self.validator = validator
        self.conflict_policy = conflict_policy
        self._domain: FrozenSet[Stratum] = f.domain | g.domain
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
        if state.is_failed:
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=f"Absorción monadal: error previo '{state.error}'",
            )

        # Ejecutar ambos caminos
        state_f = self.f(state)
        state_g = self.g(state)

        # Verificar éxito de ambos caminos
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

        # Verificar congruencia
        try:
            is_congruent = bool(self.validator(state_f, state_g))
        except Exception as exc:
            error_msg = (
                f"Error en validador de pullback '{self.name}': {exc}"
            )
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
            error_msg = (
                f"Conflicto al fusionar pullback '{self.name}': {exc}"
            )
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
            metadata={"congruent": True},
        )


# ============================================================================
# FUNTORES Y TRANSFORMACIONES NATURALES
# ============================================================================


class Functor(ABC):
    """
    Funtor covariante F: C_MIC → D.

    Un funtor preserva:
    - Identidades: F(id_A) = id_{F(A)}
    - Composición: F(g ∘ f) = F(g) ∘ F(f)
    """

    def __init__(self, name: str = ""):
        self.name: str = name or self.__class__.__name__
        self._logger: logging.Logger = logging.getLogger(
            f"MIC.Functor.{self.name}"
        )

    @abstractmethod
    def map_object(self, state: CategoricalState) -> Any:
        """Mapea un objeto (estado) de C_MIC a la categoría destino."""
        ...

    @abstractmethod
    def map_morphism(
        self, f: Morphism
    ) -> Callable[[CategoricalState], Any]:
        """
        Mapea un morfismo de C_MIC a una función en la categoría destino.
        """
        ...


class StateToDictFunctor(Functor):
    """
    Funtor concreto F: C_MIC → Set (categoría de conjuntos/dicts).

    Mapea estados a diccionarios y morfismos a funciones que producen
    diccionarios. Útil para serialización, inspección y testing.
    """

    def __init__(self):
        super().__init__("StateToDictFunctor")

    def map_object(self, state: CategoricalState) -> Dict[str, Any]:
        return state.to_dict()

    def map_morphism(
        self, f: Morphism
    ) -> Callable[[CategoricalState], Dict[str, Any]]:
        def mapped(state: CategoricalState) -> Dict[str, Any]:
            return f(state).to_dict()

        mapped.__name__ = f"F({f.name})"
        mapped.__doc__ = f"Imagen de {f.name} bajo StateToDictFunctor"
        return mapped


class NaturalTransformation(ABC):
    """
    Transformación natural η: F ⇒ G entre funtores.

    Para cada objeto A en C_MIC:
        η_A: F(A) → G(A)

    Condición de naturalidad:
        Para todo f: A → B en C_MIC,
        G(f) ∘ η_A = η_B ∘ F(f)
    """

    def __init__(self, name: str = ""):
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Componente de la transformación natural en el objeto dado."""
        ...

    @property
    def is_natural(self) -> bool:
        """
        Placeholder: la verificación de naturalidad requeriría
        comprobar el diagrama conmutativo para todos los morfismos,
        lo cual no es decidible estáticamente en general.
        """
        return True


# ============================================================================
# COMPOSITOR VERIFICABLE
# ============================================================================


class MorphismComposer:
    """
    Constructor de composiciones verificadas con chequeo estructural
    acumulativo.

    A diferencia de la composición directa con >>, el compositor
    verifica la compatibilidad de cada paso contra el conjunto
    acumulado de estratos provistos (no solo el paso inmediatamente
    anterior).
    """

    def __init__(self):
        self.steps: List[Morphism] = []
        self._accumulated_strata: FrozenSet[Stratum] = frozenset()
        self.logger: logging.Logger = logging.getLogger(
            "MIC.MorphismComposer"
        )

    def add_step(self, morphism: Morphism) -> MorphismComposer:
        """
        Añade un paso verificando compatibilidad acumulativa.

        La verificación considera todos los estratos provistos
        por los pasos anteriores (unión de dominios + codominios),
        no solo el paso inmediatamente anterior.

        Raises:
            TypeError: si el paso no es componible con la secuencia
                       acumulada.
        """
        if self.steps:
            # Estratos provistos acumulados = unión de todos los
            # dominios y codominios de pasos previos
            missing = morphism.domain - self._accumulated_strata
            if missing:
                missing_names = sorted(s.name for s in missing)
                raise TypeError(
                    "Paso no componible con la secuencia acumulada:\n"
                    f"  Pasos previos proveen: "
                    f"{sorted(s.name for s in self._accumulated_strata)}\n"
                    f"  Nuevo paso requiere: "
                    f"{sorted(s.name for s in morphism.domain)}\n"
                    f"  Faltan estratos: {missing_names}"
                )

        self.steps.append(morphism)
        # Actualizar acumulado
        self._accumulated_strata = (
            self._accumulated_strata
            | morphism.domain
            | frozenset({morphism.codomain})
        )
        self.logger.debug("✓ Paso añadido: %s", morphism)
        return self

    def build(self) -> Morphism:
        """
        Construye la composición secuencial de todos los pasos.

        Raises:
            ValueError: si no hay pasos registrados.
        """
        if not self.steps:
            raise ValueError("No hay pasos para componer")
        result = self.steps[0]
        for morphism in self.steps[1:]:
            result = result >> morphism
        self.logger.info(
            "✓ Composición construida con %d pasos: %s",
            len(self.steps),
            result,
        )
        return result

    def reset(self) -> MorphismComposer:
        """Reinicia el compositor."""
        self.steps.clear()
        self._accumulated_strata = frozenset()
        return self

    def visualize(self) -> str:
        """Representación textual de la secuencia de pasos."""
        if not self.steps:
            return "(vacío)"
        lines = [f"{i + 1}. {m}" for i, m in enumerate(self.steps)]
        return "\n".join(lines)


# ============================================================================
# VERIFICACIÓN ESTRUCTURAL
# ============================================================================


class StructuralVerifier:
    """
    Verificador de compatibilidad estructural de secuencias de morfismos.

    NOTA RIGUROSA:
    El método `is_composable_sequence` verifica que cada morfismo sucesivo
    tiene sus precondiciones de estrato satisfechas por los pasos previos.
    Esto es una condición NECESARIA para composabilidad, pero NO verifica
    exactitud homológica en sentido algebraico (Im(f) = Ker(g)).

    El nombre anterior `HomologicalVerifier.is_exact_sequence` era
    matemáticamente engañoso y ha sido corregido.
    """

    def __init__(self):
        self.logger: logging.Logger = logging.getLogger(
            "MIC.StructuralVerifier"
        )

    def is_composable_sequence(self, morphisms: List[Morphism]) -> bool:
        """
        Verifica que la secuencia de morfismos es componible:
        para cada par consecutivo (f_i, f_{i+1}), los estratos
        requeridos por f_{i+1} están provistos por el acumulado
        de los pasos f_0, ..., f_i.

        Esta verificación es estrictamente más fuerte que la del
        compositor, pues acumula todos los estratos provistos.
        """
        if len(morphisms) < 2:
            return True

        accumulated: FrozenSet[Stratum] = frozenset()

        for i, morphism in enumerate(morphisms):
            if i > 0:
                missing = morphism.domain - accumulated
                if missing:
                    self.logger.warning(
                        "Secuencia no componible en posición %d: "
                        "%s requiere %s no provistos. "
                        "Acumulado: %s",
                        i,
                        morphism.name,
                        sorted(s.name for s in missing),
                        sorted(s.name for s in accumulated),
                    )
                    return False
            accumulated = accumulated | morphism.domain | frozenset(
                {morphism.codomain}
            )

        return True

    def verify_composition(
        self, composition: Morphism
    ) -> Dict[str, Any]:
        """
        Inspección estática de un morfismo (posiblemente compuesto).
        """
        result: Dict[str, Any] = {
            "is_valid": True,
            "domain": sorted(s.name for s in composition.domain),
            "codomain": composition.codomain.name,
            "structural_type": type(composition).__name__,
        }

        # Inspección profunda para composiciones
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
        Calcula la cobertura de estratos de una lista de morfismos.

        Retorna estadísticas sobre qué estratos son producidos,
        requeridos y cuáles quedan sin cubrir.
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


# ============================================================================
# REGISTRO Y GESTIÓN CATEGÓRICA
# ============================================================================


class CategoricalRegistry:
    """
    Registro centralizado de morfismos y composiciones con
    verificación de aciclicidad del grafo de dependencias.

    Thread-safe para operaciones de registro y consulta.
    """

    def __init__(self):
        self._morphisms: Dict[str, Morphism] = {}
        self._compositions: Dict[str, Morphism] = {}
        self._lock: threading.RLock = threading.RLock()
        self.logger: logging.Logger = logging.getLogger(
            "MIC.CategoricalRegistry"
        )

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
        Registra un morfismo atómico.

        Si el nombre ya existe, se reemplaza con advertencia.
        """
        with self._lock:
            if name in self._morphisms:
                self.logger.warning(
                    "Morfismo '%s' ya registrado. Reemplazando.", name
                )
            self._morphisms[name] = morphism
            self.logger.debug("✓ Morfismo registrado: %s → %s", name, morphism)

    def register_composition(
        self, name: str, composition: Morphism
    ) -> None:
        """
        Registra una composición de morfismos.

        Si el nombre ya existe, se reemplaza con advertencia.
        """
        with self._lock:
            if name in self._compositions:
                self.logger.warning(
                    "Composición '%s' ya registrada. Reemplazando.", name
                )
            self._compositions[name] = composition
            self.logger.debug("✓ Composición registrada: %s", name)

    def unregister_morphism(self, name: str) -> bool:
        """Elimina un morfismo. Retorna True si existía."""
        with self._lock:
            if name in self._morphisms:
                del self._morphisms[name]
                self.logger.debug("✓ Morfismo eliminado: %s", name)
                return True
            return False

    def unregister_composition(self, name: str) -> bool:
        """Elimina una composición. Retorna True si existía."""
        with self._lock:
            if name in self._compositions:
                del self._compositions[name]
                self.logger.debug("✓ Composición eliminada: %s", name)
                return True
            return False

    def get_morphism(self, name: str) -> Optional[Morphism]:
        with self._lock:
            return self._morphisms.get(name)

    def get_composition(self, name: str) -> Optional[Morphism]:
        with self._lock:
            return self._compositions.get(name)

    def list_morphisms(self) -> List[str]:
        with self._lock:
            return sorted(self._morphisms.keys())

    def list_compositions(self) -> List[str]:
        with self._lock:
            return sorted(self._compositions.keys())

    def get_dependency_graph(self) -> nx.DiGraph:
        """
        Construye el grafo de dependencias entre morfismos registrados.

        Un arco (A → B) existe si y solo si B requiere en su dominio
        un estrato que A produce como codominio.

        Esto modela la relación de precedencia necesaria: A debe
        ejecutarse antes que B para satisfacer sus precondiciones.
        """
        with self._lock:
            all_items: Dict[str, Morphism] = {}
            all_items.update(self._morphisms)
            all_items.update(self._compositions)

        G = nx.DiGraph()

        # Índice inverso: estrato → morfismos que lo producen
        producers: Dict[Stratum, List[str]] = {}

        for name, morphism in all_items.items():
            G.add_node(
                name,
                codomain=morphism.codomain.name,
                domain=sorted(s.name for s in morphism.domain),
                kind=type(morphism).__name__,
            )
            producers.setdefault(morphism.codomain, []).append(name)

        # Arcos de dependencia
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
        Verifica que el grafo de dependencias es un DAG.

        Un ciclo indicaría una dependencia circular entre morfismos,
        lo cual haría imposible encontrar un orden de ejecución válido.
        """
        G = self.get_dependency_graph()
        is_acyclic = nx.is_directed_acyclic_graph(G)
        if not is_acyclic:
            cycles = list(nx.simple_cycles(G))
            self.logger.error(
                "Ciclos detectados en grafo de dependencias: %s", cycles
            )
        return is_acyclic

    def topological_order(self) -> Optional[List[str]]:
        """
        Retorna un orden topológico de los morfismos registrados,
        o None si existen ciclos.

        El orden topológico garantiza que cada morfismo se ejecuta
        después de que sus dependencias estén satisfechas.
        """
        G = self.get_dependency_graph()
        if not nx.is_directed_acyclic_graph(G):
            return None
        return list(nx.topological_sort(G))


# ============================================================================
# FACTORIES
# ============================================================================


def create_categorical_state(
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    strata: Optional[Set[Stratum]] = None,
) -> CategoricalState:
    """
    Factory para CategoricalState con valores por defecto seguros.
    """
    return CategoricalState(
        payload=dict(payload or {}),
        context=dict(context or {}),
        validated_strata=frozenset(strata or set()),
    )


def create_morphism_from_handler(
    name: str,
    target_stratum: Stratum,
    handler: Callable[..., Any],
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[List[str]] = None,
) -> Morphism:
    """
    Factory para crear un AtomicVector a partir de un handler callable.
    """
    return AtomicVector(
        name=name,
        target_stratum=target_stratum,
        handler=handler,
        required_keys=required_keys,
        optional_keys=optional_keys,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Estratificación
    "Stratum",
    # Estado categórico
    "CategoricalState",
    "CompositionTrace",
    # Morfismos
    "Morphism",
    "IdentityMorphism",
    "AtomicVector",
    "ComposedMorphism",
    "ProductMorphism",
    "CoproductMorphism",
    "PullbackMorphism",
    # Funtores y transformaciones naturales
    "Functor",
    "StateToDictFunctor",
    "NaturalTransformation",
    # Composición y verificación
    "MorphismComposer",
    "StructuralVerifier",
    "HomologicalVerifier",  # alias retrocompatible
    # Registro
    "CategoricalRegistry",
    # Factories
    "create_categorical_state",
    "create_morphism_from_handler",
]
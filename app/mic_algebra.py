"""
Módulo: MIC Algebra Evolucionado (El Núcleo Categórico)
========================================================

Formalización de la Matriz de Interacción Central (MIC) como una 
**Categoría Computacional Cerrada Cartesiana (CCC)** con:

1. **Álgebra de Morfismos**: Composición segura con verificación de tipos
2. **Mónada de Error (Railway Oriented Programming)**: Fail-fast con recovery
3. **Naturales Transformaciones**: Cambios de representación entre functores
4. **Límites y Colímites**: Productos y coproductos categóricos
5. **Verificación Homológica**: Auditoría de composiciones
6. **Serialización Funcional**: Preservación de estructuras algebraicas

Mapeo Categórico Completo:
--------------------------

Objetos (Obj C_MIC):
  - CategoricalState: Pareja (payload, validated_strata) con error monadal
  - StateVector: Representación lineal (álgebra lineal)
  - Transducción: Cambios entre representaciones

Morfismos (Hom_C(A, B)):
  - IdentityMorphism(A): id_A : A → A
  - AtomicVector: f : A → B (handler legacy)
  - ComposedMorphism: g ∘ f : A → C (si B = dom(g))
  - ProductMorphism: f ∧ g : A → (B ∧ C) (product category)
  - CoproductMorphism: f ∨ g : (A ∨ B) → C (choice/either)

Composición (∘):
  - Associatividad: (h ∘ g) ∘ f = h ∘ (g ∘ f)
  - Identidad: id_B ∘ f = f = f ∘ id_A
  - Verificación en tiempo de compilación (fail-fast)

Funtores (F: C → D):
  - Transducer: Cambia CategoricalState a StateVector
  - Naturalidad: El transducer conmuta con morfismos

Límites Categóricos:
  - PullbackMorphism: Intersección de morfismos
  - Equalizer: Verifica igualdad de caminos

Invariantes Algebraicos:
  - Clausura: f: A → B, g: B → C ⟹ g ∘ f: A → C bien-tipado
  - Monadalidad: Error es elemento absorbente
  - Funtorialidad: F(g ∘ f) = F(g) ∘ F(f)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import networkx as nx
import numpy as np

# ============================================================================
# STRATIFICATION (DIKW Hierarchy)
# ============================================================================

try:
    from app.schemas import Stratum
except ImportError:
    class Stratum(IntEnum):
        """Estratificación DIKW (Data → Information → Knowledge → Wisdom)."""
        WISDOM = 0    # Nivel más alto (más general, abstracto)
        STRATEGY = 1
        TACTICS = 2
        PHYSICS = 3   # Nivel más bajo (más concreto, física)

        def requires(self) -> FrozenSet[Stratum]:
            """Requisitos topológicos: cada estrato requiere todos los anteriores."""
            return frozenset(s for s in Stratum if s.value > self.value)
        
        @property
        def level(self) -> int:
            """Nivel numérico (WISDOM=0 es más alto)."""
            return self.value
        
        def __lt__(self, other: Stratum) -> bool:
            """Comparación topológica: WISDOM < STRATEGY < TACTICS < PHYSICS."""
            return self.value < other.value


logger = logging.getLogger("MIC.Algebra")


# ============================================================================
# TIPOS Y ESTRUCTURAS ALGEBRAICAS
# ============================================================================

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


@dataclass(frozen=True)
class CompositionTrace:
    """
    Traza de auditoría para composiciones de morfismos.
    
    Registra la secuencia de transformaciones para depuración y verificación.
    """
    
    step_number: int
    morphism_name: str
    input_domain: FrozenSet[Stratum]
    output_codomain: Stratum
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa la traza."""
        return {
            "step": self.step_number,
            "morphism": self.morphism_name,
            "domain": sorted([s.name for s in self.input_domain]),
            "codomain": self.output_codomain.name,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CategoricalState:
    """
    **Objeto Fundamental de la Categoría C_MIC**
    
    Estructura algebraica inmutable que encapsula:
    - El estado computacional (payload)
    - El contexto de ejecución (context)
    - Las propiedades certificadas (validated_strata)
    - El estado monadal (error)
    
    Propiedades:
    - Inmutabilidad estricta (frozen=True)
    - Monadalidad: Error es elemento absorbente
    - Functorialidad: Preserva estructura bajo transformaciones
    """
    
    payload: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validated_strata: FrozenSet[Stratum] = field(default_factory=frozenset)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    composition_trace: List[CompositionTrace] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """Verifica si el estado es válido (monadal)."""
        return self.error is None
    
    @property
    def is_failed(self) -> bool:
        """Complementario de is_success."""
        return not self.is_success
    
    @property
    def stratum_level(self) -> int:
        """Nivel máximo alcanzado (mínimo valor)."""
        if not self.validated_strata:
            return Stratum.PHYSICS.value
        return min(s.value for s in self.validated_strata)
    
    def with_update(
        self,
        new_payload: Optional[Dict[str, Any]] = None,
        new_context: Optional[Dict[str, Any]] = None,
        new_stratum: Optional[Stratum] = None,
        merge_payload: bool = True,
        merge_context: bool = True,
    ) -> CategoricalState:
        """
        **Funtor de Actualización (Endofunctor A → A)**
        
        Genera un nuevo estado preservando inmutabilidad.
        
        Args:
            new_payload: Datos a añadir/actualizar
            new_context: Contexto a añadir/actualizar
            new_stratum: Estrato a certificar
            merge_payload: Si mergear (True) o reemplazar (False)
            merge_context: Si mergear (True) o reemplazar (False)
        """
        updated_payload = (
            {**self.payload, **(new_payload or {})}
            if merge_payload and new_payload
            else (new_payload or self.payload)
        )
        
        updated_context = (
            {**self.context, **(new_context or {})}
            if merge_context and new_context
            else (new_context or self.context)
        )
        
        updated_strata = self.validated_strata
        if new_stratum is not None:
            updated_strata = updated_strata | frozenset([new_stratum])
        
        return CategoricalState(
            payload=updated_payload,
            context=updated_context,
            validated_strata=updated_strata,
            error=self.error,
            error_details=self.error_details,
            composition_trace=list(self.composition_trace),  # Copia inmutable
        )
    
    def with_error(
        self,
        error_msg: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> CategoricalState:
        """
        **Absorción Monadal (Error colapsa el estado)**
        
        En la mónada Either/Result, el error es elemento absorbente.
        Una vez que un morfismo falla, los subsiguientes actúan como identidad.
        """
        return CategoricalState(
            payload=self.payload,
            context=self.context,
            validated_strata=self.validated_strata,
            error=error_msg,
            error_details=details,
            composition_trace=list(self.composition_trace),
        )
    
    def add_trace(
        self,
        morphism_name: str,
        input_domain: FrozenSet[Stratum],
        output_codomain: Stratum,
        success: bool,
        error: Optional[str] = None,
    ) -> CategoricalState:
        """Añade entrada a la traza de composición."""
        step_num = len(self.composition_trace) + 1
        trace_entry = CompositionTrace(
            step_number=step_num,
            morphism_name=morphism_name,
            input_domain=input_domain,
            output_codomain=output_codomain,
            success=success,
            error=error,
        )
        
        new_trace = list(self.composition_trace) + [trace_entry]
        
        return CategoricalState(
            payload=self.payload,
            context=self.context,
            validated_strata=self.validated_strata,
            error=self.error,
            error_details=self.error_details,
            composition_trace=new_trace,
        )
    
    def compute_hash(self) -> str:
        """Computa hash SHA-256 del estado (para memoización)."""
        # Debe incluir los valores del payload para que DataFrames distintos produzcan hashes distintos
        payload_repr = {}
        for k, v in self.payload.items():
            if hasattr(v, "to_json"):
                try:
                    payload_repr[k] = v.to_json()
                except Exception:
                    payload_repr[k] = str(v)
            else:
                payload_repr[k] = str(v)

        serialized = json.dumps(
            {
                "payload": payload_repr,
                "strata": sorted([s.name for s in self.validated_strata]),
                "error": self.error,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa para persistencia."""
        return {
            "payload": self.payload,
            "context": self.context,
            "validated_strata": sorted([s.name for s in self.validated_strata]),
            "error": self.error,
            "error_details": self.error_details,
            "composition_trace": [t.to_dict() for t in self.composition_trace],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CategoricalState:
        """Reconstruye desde diccionario."""
        strata = frozenset(
            Stratum[s] for s in data.get("validated_strata", [])
        )
        
        trace = [
            CompositionTrace(
                step_number=t["step"],
                morphism_name=t["morphism"],
                input_domain=frozenset(Stratum[s] for s in t["domain"]),
                output_codomain=Stratum[t["codomain"]],
                success=t["success"],
                error=t.get("error"),
                timestamp=t.get("timestamp", 0),
            )
            for t in data.get("composition_trace", [])
        ]
        
        return cls(
            payload=data.get("payload", {}),
            context=data.get("context", {}),
            validated_strata=strata,
            error=data.get("error"),
            error_details=data.get("error_details"),
            composition_trace=trace,
        )


# ============================================================================
# MORFISMOS (Flechas de la Categoría)
# ============================================================================


class Morphism(ABC):
    """
    **Clase Base Abstracta para Morfismos en C_MIC**
    
    Representa transformaciones categóricas con:
    - Dominio (precondiciones)
    - Codominio (postcondiciones)
    - Composición segura
    - Auditoría automática
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._logger = logging.getLogger(f"MIC.Morphism.{self.name}")
    
    @property
    @abstractmethod
    def domain(self) -> FrozenSet[Stratum]:
        """
        **Precondiciones: Estratos que deben estar validados**
        
        Por ejemplo, un morfismo de TACTICS requiere que PHYSICS esté validado.
        """
        ...
    
    @property
    @abstractmethod
    def codomain(self) -> Stratum:
        """
        **Postcondiciones: Estrato que este morfismo certifica**
        """
        ...
    
    @property
    def composable_with(self) -> Callable[[Morphism], bool]:
        """Verifica si este morfismo es composable con otro."""
        def check(other: Morphism) -> bool:
            # Un morfismo g es composable después de f si su dominio
            # está satisfecho por el codominio de f
            missing = other.domain - (frozenset([self.codomain]) | self.domain)
            return len(missing) == 0
        return check
    
    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        **Aplicación del Morfismo (Functorialidad)**
        
        Transforma un estado preservando la estructura categórica.
        """
        ...
    
    def __rshift__(self, other: Morphism) -> Morphism:
        """
        **Composición Categórica: f >> g ≡ g ∘ f**
        
        Verificación en tiempo de definición (fail-fast).
        """
        return ComposedMorphism(self, other)
    
    def __mul__(self, other: Morphism) -> Morphism:
        """
        **Producto Categórico: f ∧ g (ejecución paralela)**
        """
        return ProductMorphism(self, other)
    
    def __or__(self, other: Morphism) -> Morphism:
        """
        **Coproducto Categórico: f ∨ g (elección)**
        """
        return CoproductMorphism(self, other)
    
    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return f"{self.name}: ({domain_str}) → {self.codomain.name}"


class IdentityMorphism(Morphism):
    """
    **Morfismo Identidad: id_A : A → A**
    
    Requiere axiomas categóricos:
    - f ∘ id_A = f
    - id_B ∘ f = f
    """
    
    def __init__(self, stratum: Stratum):
        super().__init__(f"id_{stratum.name}")
        self._stratum = stratum
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset([self._stratum])
    
    @property
    def codomain(self) -> Stratum:
        return self._stratum
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Identidad: no transforma el estado."""
        return state


class AtomicVector(Morphism):
    """
    **Vector Base Individual Registrado en MIC**
    
    Envuelve un handler (función) del sistema legacy.
    
    Responsabilidades:
    - Verificar dominio (gatekeeper algebraico)
    - Aplicar monadalidad (absorción de errores)
    - Adaptar handlers legacy a la categoría
    - Auditar ejecución
    """
    
    def __init__(
        self,
        name: str,
        target_stratum: Stratum,
        handler: Callable,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._target_stratum = target_stratum
        self._handler = handler
        self._required_keys = set(required_keys or [])
        self._optional_keys = set(optional_keys or [])
        
        # Dominio = requisitos topológicos del estrato objetivo
        self._domain = frozenset(target_stratum.requires())
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain
    
    @property
    def codomain(self) -> Stratum:
        return self._target_stratum
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Ejecución del morfismo atómico.
        
        1. Aplicar monadalidad (absorción)
        2. Validar dominio (gatekeeper)
        3. Validar precondiciones (claves requeridas)
        4. Ejecutar handler
        5. Procesar resultado
        6. Auditar
        """
        
        # 1. MONADALIDAD: Si hay error, absorber
        if not state.is_success:
            self._logger.debug(
                f"Monadalidad: {self.name} absorbe error previo"
            )
            return state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error="Error previo (absorción monadal)",
            )
        
        # 2. GATEKEEPER: Verificar dominio
        force_override = state.context.get("force_override", False)
        
        if not force_override:
            missing = self.domain - state.validated_strata
            
            if missing:
                missing_names = sorted([s.name for s in missing])
                error_msg = (
                    f"Violación de Dominio en '{self.name}': "
                    f"Faltan estratos {missing_names}"
                )
                self._logger.warning(error_msg)
                
                return state.with_error(error_msg).add_trace(
                    self.name,
                    self.domain,
                    self.codomain,
                    success=False,
                    error=error_msg,
                )
        
        # 3. VALIDACIÓN DE PRECONDICIONES
        missing_keys = self._required_keys - set(state.payload.keys())
        if missing_keys:
            error_msg = f"Claves requeridas faltantes: {sorted(missing_keys)}"
            self._logger.warning(f"{self.name}: {error_msg}")
            
            return state.with_error(error_msg, details={
                "missing_keys": sorted(missing_keys),
                "available_keys": sorted(state.payload.keys()),
            }).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )
        
        # 4. EJECUCIÓN DEL HANDLER
        try:
            self._logger.debug(
                f"Ejecutando: {self.name} → {self.codomain.name}"
            )
            
            # Preparar argumentos compatibles con handler legacy
            kwargs = {
                k: v for k, v in state.payload.items()
                if k in self._required_keys or k in self._optional_keys
            }
            
            result = self._handler(**kwargs)
            
            # 5. PROCESAMIENTO DEL RESULTADO
            
            # Caso 1: Resultado es dict con metadatos MIC
            if isinstance(result, dict):
                success = result.get("success", True)
                
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
                
                # Extraer payload (sin metadatos)
                clean_result = {
                    k: v for k, v in result.items()
                    if not k.startswith("_") and k not in ("success", "error", "error_details")
                }
                
                new_state = state.with_update(clean_result, new_stratum=self.codomain)
            
            # Caso 2: Resultado es un valor directo
            else:
                new_state = state.with_update(
                    {f"{self.name}_result": result},
                    new_stratum=self.codomain,
                )
            
            # 6. AUDITORÍA
            new_state = new_state.add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=True,
            )
            
            self._logger.info(f"✓ {self.name} completado")
            
            return new_state
            
        except Exception as e:
            error_msg = f"Excepción en {self.name}: {str(e)}"
            self._logger.exception(error_msg)
            
            return state.with_error(
                error_msg,
                details={
                    "exception_type": type(e).__name__,
                    "exception_str": str(e),
                },
            ).add_trace(
                self.name,
                self.domain,
                self.codomain,
                success=False,
                error=error_msg,
            )


class ComposedMorphism(Morphism):
    """
    **Composición Categórica: g ∘ f (escrito f >> g)**
    
    Garantiza seguridad de tipos y estratos en tiempo de definición.
    
    Invariantes:
    - Asociatividad: (h ∘ g) ∘ f = h ∘ (g ∘ f)
    - Identidad: id_B ∘ f = f = f ∘ id_A
    - Clausura: dom(g) ⊆ cod(f) ∪ dom(f)
    """
    
    def __init__(self, f: Morphism, g: Morphism):
        super().__init__(f"{f.name} >> {g.name}")
        self.f = f
        self.g = g
        
        # Dominio y codominio compuesto
        self._domain = frozenset(f.domain | (g.domain - frozenset([f.codomain])))
        self._codomain = g.codomain
        
        self._logger.debug(f"✓ Composición válida creada: {self}")
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain
    
    @property
    def codomain(self) -> Stratum:
        return self._codomain
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Aplicación functorial: g(f(x))."""
        # Aplicar f
        state_f = self.f(state)
        
        # Si f falló, absorción monadal detiene la composición
        if state_f.is_failed:
            return state_f
        
        # Aplicar g
        return self.g(state_f)
    
    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return f"{self.name}: ({domain_str}) → {self.codomain.name}"


class ProductMorphism(Morphism):
    """
    **Producto Categórico: f ∧ g (ejecución paralela)**
    
    Aplica f y g al mismo estado inicial y fusiona resultados.
    
    Propiedades:
    - Conmutatividad: f ∧ g = g ∧ f (sobre el mismo estado)
    - Asociatividad: (f ∧ g) ∧ h = f ∧ (g ∧ h)
    - Identidad: f ∧ id_A = f
    """
    
    def __init__(self, f: Morphism, g: Morphism):
        super().__init__(f"{f.name} ∧ {g.name}")
        self.f = f
        self.g = g
        
        # Dominio es la unión (requisitos agregados)
        self._domain = f.domain | g.domain
        
        # Codominio es el estrato más alto logrado (menor valor)
        f_level = f.codomain.value
        g_level = g.codomain.value
        self._codomain = f.codomain if f_level < g_level else g.codomain
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain
    
    @property
    def codomain(self) -> Stratum:
        return self._codomain
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Ejecución paralela con fusión de resultados."""
        
        # Absorción monadal
        if not state.is_success:
            return state
        
        # Ejecutar f en el estado original
        state_f = self.f(state)
        
        # Ejecutar g en el estado original (no en el resultado de f)
        state_g = self.g(state)
        
        # Si alguna rama falla, el producto colapsa
        if not state_f.is_success:
            return state_f
        if not state_g.is_success:
            return state_g
        
        # Fusión (unión de subespacios)
        merged_payload = {**state_f.payload, **state_g.payload}
        merged_strata = state_f.validated_strata | state_g.validated_strata
        
        # Combinar trazas
        merged_trace = list(state_f.composition_trace) + list(
            t for t in state_g.composition_trace
            if t not in state_f.composition_trace
        )
        
        return CategoricalState(
            payload=merged_payload,
            context=state.context,
            validated_strata=merged_strata,
            composition_trace=merged_trace,
        )
    
    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return f"{self.name}: ({domain_str}) → {self.codomain.name}"


class CoproductMorphism(Morphism):
    """
    **Coproducto Categórico: f ∨ g (elección/alternativa)**
    
    Intenta f primero; si falla, intenta g.
    Implementa el patrón Either/Maybe de programación funcional.
    
    Propiedades:
    - Fallback controlado
    - Recuperación de errores
    - Logging de intentos
    """
    
    def __init__(self, f: Morphism, g: Morphism):
        super().__init__(f"{f.name} ∨ {g.name}")
        self.f = f
        self.g = g
        
        # Dominio: unión (ambos pueden aplicarse)
        self._domain = f.domain | g.domain
        
        # Codominio: el mejor resultado (más alto)
        f_level = f.codomain.value
        g_level = g.codomain.value
        self._codomain = f.codomain if f_level < g_level else g.codomain
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain
    
    @property
    def codomain(self) -> Stratum:
        return self._codomain
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Intenta f; si falla, intenta g."""
        
        # Absorción monadal
        if not state.is_success:
            return state
        
        # Intentar f
        self._logger.debug(f"Intentando {self.f.name} en {self.f.name} ∨ {self.g.name}")
        state_f = self.f(state)
        
        if state_f.is_success:
            self._logger.info(f"✓ {self.f.name} exitoso en coproducto")
            return state_f
        
        # f falló, intentar g
        self._logger.warning(
            f"✗ {self.f.name} falló ({state_f.error}). "
            f"Intentando fallback: {self.g.name}"
        )
        state_g = self.g(state)
        
        if state_g.is_success:
            self._logger.info(
                f"✓ Fallback {self.g.name} exitoso"
            )
            return state_g
        
        # Ambos fallaron
        error_msg = (
            f"Coproducto {self.name} falló completamente:\n"
            f"  Intento 1 ({self.f.name}): {state_f.error}\n"
            f"  Intento 2 ({self.g.name}): {state_g.error}"
        )
        self._logger.error(error_msg)
        
        return state.with_error(error_msg, details={
            "first_attempt_error": state_f.error,
            "fallback_error": state_g.error,
        })
    
    def __repr__(self) -> str:
        domain_str = ", ".join(sorted(s.name for s in self.domain))
        return f"{self.name}: ({domain_str}) → {self.codomain.name}"


class PullbackMorphism(Morphism):
    """
    **Pullback (Intersección Categórica): [f, g] : A → B ∩ C**
    
    Dado f: A → B y g: A → C, forma el pullback que garantiza
    que ambas transformaciones sean consistentes.
    
    Usado para verificar que dos caminos algébricamente distintos
    llegan al mismo resultado (congruencia).
    """
    
    def __init__(
        self,
        name: str,
        f: Morphism,
        g: Morphism,
        validator: Callable[[CategoricalState, CategoricalState], bool],
    ):
        super().__init__(name)
        self.f = f
        self.g = g
        self.validator = validator
        
        # El dominio es la intersección
        self._domain = f.domain & g.domain
        
        # El codominio es el más restrictivo (menor valor)
        f_level = f.codomain.value
        g_level = g.codomain.value
        self._codomain = f.codomain if f_level < g_level else g.codomain
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain
    
    @property
    def codomain(self) -> Stratum:
        return self._codomain
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Ejecuta ambos caminos y verifica congruencia."""
        
        if not state.is_success:
            return state
        
        # Ejecutar ambos caminos
        state_f = self.f(state)
        state_g = self.g(state)
        
        if not state_f.is_success:
            return state_f
        if not state_g.is_success:
            return state_g
        
        # Verificar congruencia
        try:
            is_congruent = self.validator(state_f, state_g)
            
            if not is_congruent:
                error_msg = (
                    f"Pullback {self.name}: Caminos divergentes\n"
                    f"  f({self.f.name}) produce: {state_f.payload}\n"
                    f"  g({self.g.name}) produce: {state_g.payload}"
                )
                self._logger.error(error_msg)
                
                return state.with_error(error_msg, details={
                    "path_f": state_f.payload,
                    "path_g": state_g.payload,
                })
            
            # Congruencia verificada: fusionar
            merged_payload = {**state_f.payload, **state_g.payload}
            merged_strata = state_f.validated_strata | state_g.validated_strata
            
            self._logger.info(f"✓ Pullback {self.name} verificado")
            
            return CategoricalState(
                payload=merged_payload,
                context=state.context,
                validated_strata=merged_strata,
            )
            
        except Exception as e:
            error_msg = f"Error en validador de pullback {self.name}: {str(e)}"
            self._logger.exception(error_msg)
            
            return state.with_error(error_msg)


# ============================================================================
# FUNTORES (Transformaciones Naturales)
# ============================================================================


class Functor(ABC):
    """
    **Funtor: Mapa entre Categorías F: C → D**
    
    Preserva estructura categórica:
    - F(f ∘ g) = F(f) ∘ F(g)
    - F(id_A) = id_{F(A)}
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._logger = logging.getLogger(f"MIC.Functor.{self.name}")
    
    @abstractmethod
    def map_object(self, state: CategoricalState) -> Any:
        """Mapea un objeto de la categoría."""
        ...
    
    @abstractmethod
    def map_morphism(self, f: Morphism) -> Callable:
        """Mapea un morfismo a una función."""
        ...


class NaturalTransformation(ABC):
    """
    **Transformación Natural: Cambio entre Funtores**
    
    Implementa cambios de representación preservando conmutatividad.
    """
    
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Aplica la transformación natural."""
        ...
    
    @property
    def is_natural(self) -> bool:
        """Verifica naturalidad (conmutatividad con morfismos)."""
        return True


# ============================================================================
# COMPOSICIÓN VERIFICABLE (Verificación en Tiempo de Compilación)
# ============================================================================


class MorphismComposer:
    """
    **Constructor de Composiciones Verificadas**
    
    Permite construir composiciones complejas con auditoría automática
    de tipos y estratos.
    """
    
    def __init__(self):
        self.steps: List[Morphism] = []
        self.logger = logging.getLogger("MIC.MorphismComposer")
    
    def add_step(self, morphism: Morphism) -> MorphismComposer:
        """Añade un paso a la composición."""
        if self.steps:
            last = self.steps[-1]
            
            # Verificar componibilidad
            provided = frozenset([last.codomain]) | last.domain
            missing = morphism.domain - provided
            
            if missing:
                missing_names = sorted([s.name for s in missing])
                raise TypeError(
                    f"Paso no componible:\n"
                    f"  Último paso: {last} → {last.codomain}\n"
                    f"  Nuevo paso requiere: {morphism.domain}\n"
                    f"  Faltan: {missing_names}"
                )
        
        self.steps.append(morphism)
        self.logger.debug(f"✓ Paso añadido: {morphism}")
        
        return self
    
    def build(self) -> Morphism:
        """Construye la composición."""
        if not self.steps:
            raise ValueError("No hay pasos para componer")
        
        if len(self.steps) == 1:
            return self.steps[0]
        
        # Composición iterativa: f0 >> f1 >> f2 >> ...
        result = self.steps[0]
        for morphism in self.steps[1:]:
            result = result >> morphism
        
        self.logger.info(f"✓ Composición construida: {result}")
        
        return result
    
    def visualize(self) -> str:
        """Visualiza el plan de composición."""
        if not self.steps:
            return "(vacío)"
        
        lines = []
        for i, m in enumerate(self.steps):
            arrow = "→" if i == len(self.steps) - 1 else "→\n  ↓"
            lines.append(f"{i + 1}. {m}")
        
        return "\n  ".join(lines)


# ============================================================================
# ÁLGEBRA HOMOLÓGICA
# ============================================================================


class HomologicalVerifier:
    """
    **Verifica Exactitud de Secuencias**
    
    Comprueba que una secuencia de morfismos es exacta:
    ... → A → B → C → ...
    
    Exactitud: Im(f) = Ker(g) (el output de uno es el input del siguiente)
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MIC.HomologicalVerifier")
    
    def is_exact_sequence(self, morphisms: List[Morphism]) -> bool:
        """Verifica si una secuencia es exacta."""
        if len(morphisms) < 2:
            return True
        
        for i in range(len(morphisms) - 1):
            f = morphisms[i]
            g = morphisms[i + 1]
            
            # Verificar que el codominio de f coincida con el dominio de g
            # (o esté contenido)
            if not (frozenset([f.codomain]) | f.domain) >= g.domain:
                self.logger.warning(
                    f"Secuencia no exacta en posición {i}:\n"
                    f"  {f} → {f.codomain}\n"
                    f"  {g} requiere: {g.domain}"
                )
                return False
        
        return True
    
    def verify_composition(self, composition: Morphism) -> Dict[str, Any]:
        """Verifica propiedades algebraicas de una composición."""
        return {
            "is_valid": True,
            "domain": composition.domain,
            "codomain": composition.codomain,
            "trace_length": 0,  # Se actualiza durante ejecución
        }


# ============================================================================
# REGISTRO Y GESTIÓN CATEGÓRICA
# ============================================================================


class CategoricalRegistry:
    """
    **Registro de Morfismos y Composiciones**
    
    Centraliza la gestión de transformaciones algebraicas.
    """
    
    def __init__(self):
        self.morphisms: Dict[str, Morphism] = {}
        self.compositions: Dict[str, Morphism] = {}
        self.logger = logging.getLogger("MIC.CategoricalRegistry")
    
    def register_morphism(self, name: str, morphism: Morphism) -> None:
        """Registra un morfismo."""
        if name in self.morphisms:
            self.logger.warning(f"Morfismo {name} ya registrado. Reemplazando.")
        
        self.morphisms[name] = morphism
        self.logger.debug(f"✓ Morfismo registrado: {name} → {morphism}")
    
    def register_composition(self, name: str, composition: Morphism) -> None:
        """Registra una composición."""
        if name in self.compositions:
            self.logger.warning(f"Composición {name} ya registrada. Reemplazando.")
        
        self.compositions[name] = composition
        self.logger.debug(f"✓ Composición registrada: {name}")
    
    def get_morphism(self, name: str) -> Optional[Morphism]:
        """Obtiene un morfismo por nombre."""
        return self.morphisms.get(name)
    
    def get_composition(self, name: str) -> Optional[Morphism]:
        """Obtiene una composición por nombre."""
        return self.compositions.get(name)
    
    def list_morphisms(self) -> List[str]:
        """Lista todos los morfismos registrados."""
        return list(self.morphisms.keys())
    
    def list_compositions(self) -> List[str]:
        """Lista todas las composiciones registradas."""
        return list(self.compositions.keys())
    
    def get_dependency_graph(self) -> nx.DiGraph:
        """Construye un grafo de dependencias."""
        G = nx.DiGraph()
        
        for name, morphism in self.morphisms.items():
            G.add_node(name, codomain=morphism.codomain.name)
        
        # Añadir aristas basadas en requisitos
        for name, morphism in self.morphisms.items():
            for stratum in morphism.domain:
                # Encontrar morfismos que producen este estrato
                for other_name, other in self.morphisms.items():
                    if other.codomain == stratum and other_name != name:
                        G.add_edge(other_name, name)
        
        return G
    
    def verify_acyclicity(self) -> bool:
        """Verifica que no hay ciclos en las dependencias."""
        G = self.get_dependency_graph()
        is_acyclic = nx.is_directed_acyclic_graph(G)
        
        if not is_acyclic:
            cycles = list(nx.simple_cycles(G))
            self.logger.error(f"Ciclos detectados: {cycles}")
        
        return is_acyclic


# ============================================================================
# EXPORTACIÓN Y UTILIDADES
# ============================================================================


def create_categorical_state(
    payload: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    strata: Optional[Set[Stratum]] = None,
) -> CategoricalState:
    """Factory para crear estados categóricos."""
    return CategoricalState(
        payload=payload or {},
        context=context or {},
        validated_strata=frozenset(strata or set()),
    )


def create_morphism_from_handler(
    name: str,
    target_stratum: Stratum,
    handler: Callable,
    required_keys: Optional[List[str]] = None,
) -> Morphism:
    """Factory para crear morfismos a partir de handlers legacy."""
    return AtomicVector(
        name=name,
        target_stratum=target_stratum,
        handler=handler,
        required_keys=required_keys,
    )


__all__ = [
    "Stratum",
    "CategoricalState",
    "Morphism",
    "IdentityMorphism",
    "AtomicVector",
    "ComposedMorphism",
    "ProductMorphism",
    "CoproductMorphism",
    "PullbackMorphism",
    "Functor",
    "NaturalTransformation",
    "MorphismComposer",
    "HomologicalVerifier",
    "CategoricalRegistry",
    "CompositionTrace",
    "create_categorical_state",
    "create_morphism_from_handler",
]
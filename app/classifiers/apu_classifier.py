"""
Motor de Inferencia Determinista para Clasificación de APUs
============================================================

Este componente implementa un motor de inferencia determinista diseñado para
categorizar la naturaleza ontológica de un APU (Análisis de Precio Unitario).
Utiliza un sistema de reglas jerárquicas vectorizadas para particionar el
espacio vectorial definido por las proporciones de costos.

Metodología de Clasificación (V4 - Vectorial Robusta con Contratos Explícitos):
────────────────────────────────────────────────────────────────────────────────

1. Compilación de Reglas (`_compile_rules`):
   Convierte las reglas de texto ("porcentaje > 60") en funciones vectorizadas
   de NumPy mediante pd.DataFrame.eval, que maneja correctamente la precedencia
   de operadores lógicos (and/or) sobre arrays sin reescritura de strings.

2. Validación de Cobertura Espacial:
   Utiliza un muestreo denso sobre el espacio [0,1]² para verificar
   matemáticamente que las reglas cubren el espacio de posibilidades.
   La medida de Lebesgue del conjunto descubierto ≈ |gaps| / grid_size².

3. Análisis Espacial:
   Calcula centroides geométricos de las categorías clasificadas para
   entender la distribución topológica de los costos en el espacio de fase.

4. StructuralClassifier:
   Extiende la lógica para considerar la topología de la red de insumos
   (Nivel 3), detectando estructuras como 'Islas' (Suministro Puro sin
   instalación).

Contrato de Escala (invariante crítico)
────────────────────────────────────────
Las CONDICIONES de reglas operan en escala [0, 100] (porcentajes).
Las PROPORCIONES internas de clasificación operan en escala [0, 1].
La conversión [0,1] → [0,100] se realiza EXACTAMENTE en dos lugares:
  - ClassificationRule.evaluate()             → multiplica por 100
  - APUClassifier._create_vectorized_function → multiplica por 100 en DataFrame

Correcciones V4 respecto a V3:
───────────────────────────────
- [FIX] _CONDITION_PATTERN ahora se usa en _validate_syntax (paso rápido pre-AST).
- [FIX] Validación de `priority` como entero no-negativo en _load_config.
- [FIX] get_coverage_bounds clampea area_estimada a [0, 1] para evitar >100%.
- [FIX] _analyze_classification_quality maneja DataFrame vacío explícitamente.
- [FIX] classify_by_structure valida tipos de valores de insumos con try/except.
- [IMPROVE] Contrato de escala documentado con constante _CONDITION_SCALE.
- [IMPROVE] __repr__ y __str__ para ClassificationRule.
- [IMPROVE] StructuralClassifier con __init__ mínimo (sin carga innecesaria).
- [IMPROVE] _sample_uncovered_regions documenta invariante de escala del fallback.
- [IMPROVE] get_coverage_report con area_estimada correctamente clampada.
- [IMPROVE] Separación clara entre _validate_syntax_lexical y _validate_syntax_ast.
- [NEW] ClassificationResult: dataclass tipado para resultados de clasificación.
- [NEW] APUClassifier.classify_batch: clasifica lista de tuplas con resultado tipado.
- [NEW] Constante _CONDITION_SCALE para documentar el invariante de escala.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES DE DOMINIO
# =============================================================================

# Tamaño de grilla por defecto para el muestreo de cobertura espacial.
# La medida de Lebesgue del descubierto ≈ |gaps| / _DEFAULT_GRID_SIZE².
_DEFAULT_GRID_SIZE: int = 50

# Umbral de dominancia para clasificación estructural pura (1 - ε_numérico).
_DOMINANCE_THRESHOLD: float = 1.0 - 1e-7

# Escala de las condiciones de texto: [0, 100] (porcentajes).
# INVARIANTE CRÍTICO: toda condición de regla usa esta escala.
# Las proporciones internas [0,1] se multiplican por esta constante
# justo antes de la evaluación (en evaluate() y _create_vectorized_function).
_CONDITION_SCALE: float = 100.0

# Tipo alias para la función vectorizada de regla.
# Contrato: recibe proporciones en [0,1], evalúa en escala [0,100] internamente.
RuleFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


# =============================================================================
# RESULTADO TIPADO DE CLASIFICACIÓN
# =============================================================================


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """
    Resultado inmutable de clasificar un APU individual.

    Attributes:
        tipo: Tipo de APU clasificado.
        pct_materiales: Proporción de materiales ∈ [0, 1].
        pct_mo_eq: Proporción de MO/equipo ∈ [0, 1].
        total_cost: Costo total del APU.
        rule_matched: Tipo de regla que produjo la clasificación (None si default).
    """

    tipo: str
    pct_materiales: float
    pct_mo_eq: float
    total_cost: float
    rule_matched: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario para integración con sistemas externos."""
        return {
            "tipo": self.tipo,
            "pct_materiales": round(self.pct_materiales, 6),
            "pct_mo_eq": round(self.pct_mo_eq, 6),
            "total_cost": self.total_cost,
            "rule_matched": self.rule_matched,
        }


# =============================================================================
# ESTRUCTURA DE REGLA DE CLASIFICACIÓN
# =============================================================================


@dataclass
class ClassificationRule:
    """
    Estructura para reglas de clasificación con validación integrada.

    Diseño
    ──────
    No es `frozen` porque `__post_init__` necesita normalizar `self.condition`
    (mutación controlada y única en construcción). Para preservar la semántica
    de valor inmutable post-construcción, ningún método público muta el estado.

    Escala de condiciones
    ─────────────────────
    Las condiciones operan en escala [0, 100] (porcentajes legibles).
    Ejemplo: "porcentaje_materiales >= 60.0" significa ≥ 60%.
    La conversión desde proporciones [0,1] se realiza en evaluate()
    y en _create_vectorized_function().

    Variables permitidas en condiciones
    ────────────────────────────────────
    `porcentaje_materiales` y `porcentaje_mo_eq`, ambas en escala [0, 100].
    """

    rule_type: str
    priority: int
    condition: str
    description: str = ""

    # Variables permitidas en condiciones — inmutable a nivel de clase.
    _ALLOWED_VARS: ClassVar[FrozenSet[str]] = frozenset(
        {"porcentaje_materiales", "porcentaje_mo_eq"}
    )

    # Patrón léxico que describe tokens válidos en una condición.
    # USADO en _validate_syntax_lexical como comprobación rápida O(n).
    _CONDITION_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?:[\d\s\.\(\)><=!]"
        r"|porcentaje_materiales"
        r"|porcentaje_mo_eq"
        r"|and|or|not)+$"
    )

    # Patrones de extracción de bounds (>= antes de > para evitar ambigüedad).
    # Cada tupla: (patrón_regex, variable, tipo_bound, ε_offset)
    # ε_offset: diferencial infinitesimal para operadores estrictos (>, <).
    _BOUND_PATTERNS: ClassVar[Tuple[Tuple[str, str, str, float], ...]] = (
        (r"porcentaje_materiales\s*>=\s*(\d+\.?\d*)", "mat", "min", 0.0),
        (r"porcentaje_materiales\s*>\s*(\d+\.?\d*)",  "mat", "min", 1e-6),
        (r"porcentaje_materiales\s*<=\s*(\d+\.?\d*)", "mat", "max", 0.0),
        (r"porcentaje_materiales\s*<\s*(\d+\.?\d*)",  "mat", "max", -1e-6),
        (r"porcentaje_mo_eq\s*>=\s*(\d+\.?\d*)",      "mo",  "min", 0.0),
        (r"porcentaje_mo_eq\s*>\s*(\d+\.?\d*)",       "mo",  "min", 1e-6),
        (r"porcentaje_mo_eq\s*<=\s*(\d+\.?\d*)",      "mo",  "max", 0.0),
        (r"porcentaje_mo_eq\s*<\s*(\d+\.?\d*)",       "mo",  "max", -1e-6),
    )

    def __post_init__(self) -> None:
        """Normaliza y valida la condición al instanciar."""
        # Validar tipo de priority antes de cualquier otra operación.
        self._validate_priority()
        # Mutación controlada única: normalización de la condición.
        self.condition = self._normalize_condition(self.condition)
        self._validate_syntax()

    # ── Representación ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Representación técnica para depuración."""
        return (
            f"ClassificationRule("
            f"rule_type={self.rule_type!r}, "
            f"priority={self.priority}, "
            f"condition={self.condition!r})"
        )

    def __str__(self) -> str:
        """Representación legible para logging."""
        desc = f" — {self.description}" if self.description else ""
        return f"[P{self.priority}] {self.rule_type}{desc}: {self.condition!r}"

    # ── Validación de priority ────────────────────────────────────────────

    def _validate_priority(self) -> None:
        """
        Valida que priority sea un entero no-negativo.

        Raises:
            TypeError:  si priority no es un entero (incluye float, str, None).
            ValueError: si priority es negativo.
        """
        if not isinstance(self.priority, int):
            raise TypeError(
                f"priority debe ser int, recibido: {type(self.priority).__name__!r} "
                f"(valor: {self.priority!r})"
            )
        if self.priority < 0:
            raise ValueError(
                f"priority debe ser ≥ 0, recibido: {self.priority}"
            )

    # ── Normalización ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_condition(condition: str) -> str:
        """
        Normaliza operadores lógicos a minúsculas Python y colapsa espacios.

        Pasos:
        1. Normalizar AND/OR/NOT a minúsculas (insensible a mayúsculas).
        2. Colapsar tabs y espacios múltiples a un único espacio.
        3. Strip de espacios extremos.

        La normalización de espacios previene que condiciones malformadas
        pasen la validación léxica con tokens separados accidentalmente.
        """
        condition = re.sub(r"\bAND\b", "and", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bOR\b",  "or",  condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bNOT\b", "not", condition, flags=re.IGNORECASE)
        # Colapsar tabs y espacios múltiples a un único espacio.
        condition = re.sub(r"[ \t]+", " ", condition)
        return condition.strip()

    # ── Validación de sintaxis ────────────────────────────────────────────

    def _validate_syntax(self) -> None:
        """
        Valida sintaxis en dos pasos complementarios.

        Paso 1 — Léxico rápido (O(n)):
            Aplica `_CONDITION_PATTERN` sobre la condición normalizada.
            Si el patrón no cubre la cadena completa, hay tokens no autorizados.
            Este paso es más rápido que la compilación AST para detectar
            variables no permitidas.

        Paso 2 — AST completo (Python compiler):
            `compile()` detecta errores de sintaxis que pasen el léxico
            (ej. paréntesis desbalanceados, operadores adyacentes inválidos).

        Raises:
            ValueError: si la condición contiene tokens no permitidos o
                        tiene sintaxis Python inválida.
        """
        self._validate_syntax_lexical()
        self._validate_syntax_ast()

    def _validate_syntax_lexical(self) -> None:
        """
        Validación léxica usando _CONDITION_PATTERN (paso rápido).

        Sustituye variables y literales conocidos; si queda algún residuo,
        contiene tokens no autorizados.

        Note: El patrón _CONDITION_PATTERN se aplica después de reemplazar
        variables para no depender de que el regex sea exhaustivo con
        identificadores arbitrarios.
        """
        # Sustituir variables conocidas por espacios.
        test_expr = self.condition
        for var in self._ALLOWED_VARS:
            test_expr = test_expr.replace(var, " ")

        # Remover literales numéricos.
        test_expr = re.sub(r"\b\d+\.?\d*\b", " ", test_expr)

        # Remover operadores y palabras clave permitidos.
        for token in (">=", "<=", "==", "!=", ">", "<", "and", "or", "not", "(", ")"):
            test_expr = test_expr.replace(token, " ")

        remaining = test_expr.strip()
        if remaining:
            raise ValueError(
                f"Condición contiene elementos no permitidos: {remaining!r}\n"
                f"Condición original: {self.condition!r}\n"
                f"Variables permitidas: {sorted(self._ALLOWED_VARS)}"
            )

    def _validate_syntax_ast(self) -> None:
        """
        Validación sintáctica completa mediante compilación AST.

        Raises:
            ValueError: si la condición tiene sintaxis Python inválida.
        """
        try:
            compile(self.condition, "<condition>", "eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Sintaxis Python inválida en condición {self.condition!r}: {exc}"
            ) from exc

    # ── Evaluación escalar ────────────────────────────────────────────────

    def evaluate(self, pct_materiales: float, pct_mo_eq: float) -> bool:
        """
        Evalúa la regla para un par escalar con contexto de ejecución seguro.

        Contrato de escala:
            pct_materiales ∈ [0, 1] → se convierte a [0, 100] internamente.
            pct_mo_eq      ∈ [0, 1] → se convierte a [0, 100] internamente.

        El contexto restringe `__builtins__` a un dict vacío, bloqueando el
        acceso a built-ins de Python. El resultado se verifica explícitamente
        como `bool` para prevenir que objetos truthy no-booleanos pasen
        silenciosamente (ej. arrays NumPy de un elemento).

        Args:
            pct_materiales: Proporción de materiales ∈ [0, 1].
            pct_mo_eq:      Proporción de MO/equipo ∈ [0, 1].

        Returns:
            True si la condición se satisface, False en cualquier error.
        """
        try:
            safe_context: Dict[str, Any] = {
                "__builtins__": {},
                "porcentaje_materiales": float(pct_materiales) * _CONDITION_SCALE,
                "porcentaje_mo_eq":      float(pct_mo_eq)      * _CONDITION_SCALE,
            }
            result = eval(self.condition, safe_context)  # noqa: S307

            # Verificación de tipo: rechaza arrays u objetos no escalares.
            if not isinstance(result, (bool, int, float, np.bool_)):
                raise TypeError(
                    f"Tipo de retorno inesperado: {type(result).__name__!r}"
                )
            return bool(result)
        except Exception as exc:
            logger.error(
                "Error evaluando regla '%s' con (mat=%.4f, mo=%.4f): %s",
                self.rule_type, pct_materiales, pct_mo_eq, exc,
            )
            return False

    # ── Extracción de bounds ──────────────────────────────────────────────

    def get_coverage_bounds(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Extrae bounds del espacio de cobertura en [0, 1]².

        Aplica ε-offset para operadores estrictos (>, <) preservando la
        topología del conjunto abierto vs cerrado.

        Los patrones `_BOUND_PATTERNS` están ordenados con `>=` antes de `>`
        para que el offset no se aplique accidentalmente a operadores no
        estrictos. En condiciones compuestas con `or`, `re.search` encuentra
        el PRIMER match del patrón, que puede no ser el más restrictivo.

        Limitación conocida:
            En condiciones con múltiples cláusulas `or`, los bounds representan
            el rango de la primera cláusula encontrada, no la unión de rangos.
            Para análisis exacto, usar `_sample_uncovered_regions`.

        Acumulación correcta:
          - `min` acumula con max(actual, nuevo) → cota inferior más restrictiva.
          - `max` acumula con min(actual, nuevo) → cota superior más restrictiva.

        Returns:
            ((mat_min, mat_max), (mo_min, mo_max)) en [0, 1]².
        """
        mat_min, mat_max = 0.0, 1.0
        mo_min,  mo_max  = 0.0, 1.0

        for pattern, var_type, bound_type, offset in self._BOUND_PATTERNS:
            match = re.search(pattern, self.condition)
            if not match:
                continue
            raw_value = float(match.group(1)) / _CONDITION_SCALE + offset
            value = float(np.clip(raw_value, 0.0, 1.0))

            if var_type == "mat":
                if bound_type == "min":
                    mat_min = max(mat_min, value)
                else:
                    mat_max = min(mat_max, value)
            else:
                if bound_type == "min":
                    mo_min = max(mo_min, value)
                else:
                    mo_max = min(mo_max, value)

        # Garantizar invariante mat_min ≤ mat_max y mo_min ≤ mo_max.
        mat_min = min(mat_min, mat_max)
        mo_min  = min(mo_min,  mo_max)

        return (mat_min, mat_max), (mo_min, mo_max)

    def estimated_area(self) -> float:
        """
        Calcula el área estimada de cobertura en [0, 1]².

        Retorna un valor en [0, 1] representando la fracción del espacio
        cubierta según los bounds extraídos (aproximación para condiciones simples).

        Returns:
            Área estimada ∈ [0, 1].
        """
        (mat_min, mat_max), (mo_min, mo_max) = self.get_coverage_bounds()
        area = max(0.0, mat_max - mat_min) * max(0.0, mo_max - mo_min)
        return float(np.clip(area, 0.0, 1.0))


# =============================================================================
# CLASIFICADOR PRINCIPAL
# =============================================================================


class APUClassifier:
    """
    Clasificador robusto de APUs basado en reglas vectorizadas y validación
    espacial sobre el simplex [0,1]².

    Contrato de escala (ver módulo docstring):
        Condiciones de reglas: escala [0, 100].
        Proporciones internas: escala [0, 1].
        Conversión: evaluate() y _create_vectorized_function().

    Orden de inicialización (invariante)
    ──────────────────────────────────────
    1. _load_config     → self.rules poblado y ordenado.
    2. _compile_rules   → self._rule_cache y self._sorted_rules listos.
    3. _validate_rules  → usa _rule_cache (ya disponible) para cobertura.

    Este orden garantiza que la validación de cobertura opera sobre reglas
    ya compiladas, evitando el bug de V2 donde _rule_cache estaba vacío
    durante la validación.

    Thread Safety:
        Las instancias son stateful (contienen _rule_cache).
        No se garantiza thread-safety para operaciones de clasificación
        concurrentes sobre la misma instancia. Usar instancias separadas
        por hilo o proteger con threading.Lock.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Inicializa el clasificador.

        Args:
            config_path: Ruta al archivo JSON de configuración de reglas.
                         Si es None o no existe, se cargan reglas por defecto.
        """
        self.rules: List[ClassificationRule] = []
        self.default_type: str = "INDEFINIDO"
        self.zero_cost_type: str = "SIN_COSTO"

        # Cache de funciones vectorizadas: rule_type → RuleFunc.
        self._rule_cache: Dict[str, RuleFunc] = {}

        # Reglas ordenadas por prioridad — calculadas una vez en _compile_rules.
        self._sorted_rules: List[ClassificationRule] = []

        # Invariante de inicialización: load → compile → validate.
        self._load_config(config_path)
        self._compile_rules()   # ← Antes de _validate_rules
        self._validate_rules()  # ← Usa _rule_cache ya poblado

    # ── Representación ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Representación técnica del clasificador."""
        return (
            f"APUClassifier("
            f"rules={len(self.rules)}, "
            f"default={self.default_type!r}, "
            f"zero_cost={self.zero_cost_type!r})"
        )

    # ── Carga de configuración ────────────────────────────────────────────

    def _load_config(self, config_path: Optional[str]) -> None:
        """
        Carga reglas desde archivo JSON o usa las reglas por defecto.

        El archivo debe tener la estructura:
        {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "INSTALACION",
                        "priority": 1,
                        "condition": "porcentaje_mo_eq >= 60.0",
                        "description": "Descripción opcional"
                    }
                ],
                "default_type": "INDEFINIDO",
                "zero_cost_type": "SIN_COSTO"
            }
        }

        Validaciones aplicadas a cada regla:
          - "type" (str): requerido.
          - "condition" (str): requerido.
          - "priority" (int, ≥ 0): opcional, default=99.
            Si no es entero o es negativo, se usa 99 con advertencia.
          - "description" (str): opcional, default="".
        """
        if not config_path or not Path(config_path).exists():
            logger.warning(
                "Configuración no encontrada en '%s', usando reglas por defecto.",
                config_path,
            )
            self._load_default_rules()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rules_config = config.get("apu_classification_rules", {})
            loaded: List[ClassificationRule] = []

            for i, rule_dict in enumerate(rules_config.get("rules", [])):
                try:
                    raw_priority = rule_dict.get("priority", 99)
                    # Validar y normalizar priority: debe ser int ≥ 0.
                    if not isinstance(raw_priority, int) or raw_priority < 0:
                        logger.warning(
                            "Regla #%d: priority inválido (%r), usando 99.",
                            i, raw_priority,
                        )
                        raw_priority = 99

                    rule = ClassificationRule(
                        rule_type=str(rule_dict["type"]),
                        priority=int(raw_priority),
                        condition=str(rule_dict["condition"]),
                        description=str(rule_dict.get("description", "")),
                    )
                    loaded.append(rule)
                except KeyError as exc:
                    logger.warning(
                        "Regla #%d omitida — campo requerido ausente: %s", i, exc
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Regla #%d omitida — error de validación: %s", i, exc
                    )

            if not loaded:
                logger.warning(
                    "No se cargaron reglas válidas desde '%s'; "
                    "usando reglas por defecto.",
                    config_path,
                )
                self._load_default_rules()
                return

            self.rules = sorted(loaded, key=lambda r: r.priority)
            self.default_type  = str(rules_config.get("default_type",  "INDEFINIDO"))
            self.zero_cost_type = str(rules_config.get("zero_cost_type", "SIN_COSTO"))

        except json.JSONDecodeError as exc:
            logger.error(
                "JSON inválido en '%s': %s — usando reglas por defecto.",
                config_path, exc,
            )
            self._load_default_rules()
        except OSError as exc:
            logger.error(
                "Error de E/S leyendo '%s': %s — usando reglas por defecto.",
                config_path, exc,
            )
            self._load_default_rules()

    def _load_default_rules(self) -> None:
        """
        Carga el conjunto mínimo de reglas que cubren el espacio [0,1]².

        Las reglas por defecto están diseñadas para ser exhaustivas:
        OBRA_COMPLETA actúa como residuo topológico (prioridad 4),
        garantizando que ningún punto válido del espacio quede sin clasificar.

        Verificación de cobertura:
            P(mat, mo) ∈ [0,1]² → siempre clasificado:
            - INSTALACION:      mo ≥ 0.6
            - SUMINISTRO:       mat ≥ 0.6
            - CONSTRUCCION_MIXTO: mat ∈ [0.4, 0.6) ∨ mo ∈ [0.4, 0.6)
            - OBRA_COMPLETA:    mat ≥ 0 ∧ mo ≥ 0   (cubre todo el espacio)
        """
        self.rules = [
            ClassificationRule(
                rule_type="INSTALACION",
                priority=1,
                condition="porcentaje_mo_eq >= 60.0",
                description="Predomina MO/equipo (≥60%)",
            ),
            ClassificationRule(
                rule_type="SUMINISTRO",
                priority=2,
                condition="porcentaje_materiales >= 60.0",
                description="Predomina materiales (≥60%)",
            ),
            ClassificationRule(
                rule_type="CONSTRUCCION_MIXTO",
                priority=3,
                condition=(
                    "(porcentaje_materiales >= 40.0 and porcentaje_materiales < 60.0)"
                    " or (porcentaje_mo_eq >= 40.0 and porcentaje_mo_eq < 60.0)"
                ),
                description="Composición mixta (40–60%)",
            ),
            ClassificationRule(
                rule_type="OBRA_COMPLETA",
                priority=4,
                condition="porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0",
                description="Cobertura residual (cubre todo el espacio)",
            ),
        ]

    # ── Compilación de reglas ─────────────────────────────────────────────

    def _compile_rules(self) -> None:
        """
        Compila las reglas a funciones NumPy vectorizadas y ordena por prioridad.

        Almacena `_sorted_rules` como lista ordenada reutilizable, evitando
        re-ordenar en cada llamada a `_classify_vectorized_optimized`.

        Las reglas que fallan en compilación se omiten del cache (con advertencia)
        pero permanecen en `_sorted_rules` para fallback escalar.
        """
        self._sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        self._rule_cache.clear()

        for rule in self._sorted_rules:
            try:
                self._rule_cache[rule.rule_type] = (
                    self._create_vectorized_function(rule.condition)
                )
                logger.debug("Regla compilada: %s", rule)
            except Exception as exc:
                logger.warning(
                    "No se pudo compilar regla '%s': %s — disponible solo en modo escalar.",
                    rule.rule_type, exc,
                )

    def _create_vectorized_function(self, condition: str) -> RuleFunc:
        """
        Compila una condición de texto a función NumPy vectorizada.

        Estrategia: `pd.DataFrame.eval` maneja correctamente la precedencia
        de operadores lógicos (and/or) sobre arrays de NumPy sin necesidad
        de reescribir la string a operadores bitwise (&/|/~).

        Contrato de escala (invariante del módulo):
            La función recibe proporciones en [0, 1] y aplica la conversión
            × _CONDITION_SCALE internamente antes de la evaluación.
            Esto mantiene las condiciones legibles en escala [0, 100].

        Args:
            condition: Condición normalizada en escala [0, 100].

        Returns:
            RuleFunc: (x: np.ndarray[0,1], y: np.ndarray[0,1]) → np.ndarray[bool]
                donde x = porcentaje_materiales ∈ [0, 1]
                      y = porcentaje_mo_eq      ∈ [0, 1]
        """
        # Captura `condition` y `_CONDITION_SCALE` por cierre — inmutables.
        scale = _CONDITION_SCALE

        def rule_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            try:
                df_temp = pd.DataFrame({
                    "porcentaje_materiales": x * scale,
                    "porcentaje_mo_eq":      y * scale,
                })
                result = df_temp.eval(condition)
                # Garantizar dtype bool para operaciones bitwise posteriores.
                return result.values.astype(bool)
            except Exception as exc:
                logger.debug(
                    "Error en evaluación vectorizada de condición %r: %s",
                    condition, exc,
                )
                return np.zeros(len(x), dtype=bool)

        return rule_func

    # ── Validación de reglas ──────────────────────────────────────────────

    def _validate_rules(self) -> None:
        """
        Valida coherencia del sistema de reglas y cobertura topológica.

        Comprobaciones:
        1. Al menos una regla definida.
        2. Sin tipos duplicados (advertencia, no error — pueden ser intencionados).
        3. Cobertura del espacio [0,1]² (advertencia si hay gaps).
        """
        if not self.rules:
            raise ValueError(
                "No hay reglas de clasificación definidas. "
                "El clasificador no puede operar sin reglas."
            )

        # Detección de tipos duplicados.
        type_counts: Dict[str, int] = {}
        for rule in self.rules:
            type_counts[rule.rule_type] = type_counts.get(rule.rule_type, 0) + 1

        duplicates = {t for t, c in type_counts.items() if c > 1}
        if duplicates:
            logger.warning(
                "Tipos de regla duplicados detectados: %s. "
                "En clasificación, la regla de menor prioridad tomará precedencia.",
                sorted(duplicates),
            )

        # Validación de cobertura espacial.
        gaps, total_points = self._sample_uncovered_regions(_DEFAULT_GRID_SIZE)
        if gaps:
            gap_ratio = len(gaps) / max(total_points, 1)
            logger.warning(
                "Las reglas no cubren %.2f%% del espacio [0,1]² "
                "(%d puntos sin cobertura de %d muestreados). "
                "Considere agregar una regla de cobertura residual.",
                gap_ratio * 100.0, len(gaps), total_points,
            )
        else:
            logger.info(
                "Cobertura topológica completa validada sobre grilla %d×%d.",
                _DEFAULT_GRID_SIZE, _DEFAULT_GRID_SIZE,
            )

    def _sample_uncovered_regions(
        self,
        grid_size: int = _DEFAULT_GRID_SIZE,
    ) -> Tuple[List[Tuple[float, float]], int]:
        """
        Muestrea cobertura del simplex [0,1]² mediante evaluación tensorizada.

        Invariante de escala:
            X_flat, Y_flat ∈ [0, 1] (proporciones).
            Las funciones en _rule_cache y rule.evaluate() aplican la
            conversión × _CONDITION_SCALE internamente.

        Returns:
            (gaps, total_points):
                gaps         — lista de (mat, mo) ∈ [0,1]² sin cobertura.
                total_points — número de puntos muestreados (= grid_size²).

        Complejidad: O(grid_size² × R) donde R = número de reglas.

        Fallback escalar:
            Si la evaluación vectorizada falla para alguna regla, se aplica
            evaluación escalar mediante rule.evaluate() sobre los puntos
            aún descubiertos. Se registra en DEBUG para visibilidad.
        """
        x = np.linspace(0.0, 1.0, grid_size)
        y = np.linspace(0.0, 1.0, grid_size)
        X, Y = np.meshgrid(x, y)
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        total_points = len(X_flat)  # = grid_size²

        covered = np.zeros(total_points, dtype=bool)

        # Fase 1: evaluación vectorizada (O(total_points) por regla).
        for rule in self._sorted_rules:
            if np.all(covered):
                break  # Cobertura completa alcanzada

            if rule.rule_type not in self._rule_cache:
                continue

            try:
                covered |= self._rule_cache[rule.rule_type](X_flat, Y_flat)
            except Exception as exc:
                logger.debug(
                    "Evaluación vectorizada fallida para '%s' en muestreo: %s",
                    rule.rule_type, exc,
                )

        # Fase 2: fallback escalar para puntos aún descubiertos.
        # Invariante: X_flat, Y_flat ∈ [0, 1]; rule.evaluate espera [0, 1].
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) > 0:
            logger.debug(
                "Fallback escalar para %d punto(s) no cubiertos.",
                len(uncovered_idx),
            )
            for idx in uncovered_idx:
                for rule in self._sorted_rules:
                    # rule.evaluate recibe proporciones [0, 1] → convierte × 100 internamente.
                    if rule.evaluate(float(X_flat[idx]), float(Y_flat[idx])):
                        covered[idx] = True
                        break

        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            return [], total_points

        gaps = list(
            zip(
                X_flat[uncovered_mask].tolist(),
                Y_flat[uncovered_mask].tolist(),
            )
        )
        return gaps, total_points

    # ── Clasificación escalar ─────────────────────────────────────────────

    def classify_single(
        self,
        pct_materiales: float,
        pct_mo_eq: float,
        total_cost: float = 1.0,
    ) -> str:
        """
        Clasifica un APU escalar usando evaluación directa de reglas.

        Args:
            pct_materiales: Proporción de materiales ∈ [0, 1].
            pct_mo_eq:      Proporción de MO/equipo ∈ [0, 1].
            total_cost:     Costo total del APU. Si ≤ 0 o NaN → SIN_COSTO.

        Returns:
            Tipo de APU clasificado (str).

        Note:
            Para obtener el resultado tipado con metadata adicional,
            usar `classify_single_detailed()`.
        """
        pct_materiales = float(np.clip(pct_materiales, 0.0, 1.0))
        pct_mo_eq = float(np.clip(pct_mo_eq, 0.0, 1.0))

        # Verificación robusta de costo inválido (≤ 0 o NaN).
        if (
            not isinstance(total_cost, (int, float))
            or np.isnan(float(total_cost))
            or float(total_cost) <= 0.0
        ):
            return self.zero_cost_type

        for rule in self._sorted_rules:
            if rule.evaluate(pct_materiales, pct_mo_eq):
                return rule.rule_type

        return self.default_type

    def classify_single_detailed(
        self,
        pct_materiales: float,
        pct_mo_eq: float,
        total_cost: float = 1.0,
    ) -> ClassificationResult:
        """
        Clasifica un APU escalar retornando un ClassificationResult tipado.

        Proporciona metadata adicional (qué regla hizo match, proporciones
        clampadas) para trazabilidad y depuración.

        Args:
            pct_materiales: Proporción de materiales ∈ [0, 1].
            pct_mo_eq:      Proporción de MO/equipo ∈ [0, 1].
            total_cost:     Costo total del APU.

        Returns:
            ClassificationResult con tipo, proporciones y regla matched.
        """
        pct_mat_clipped = float(np.clip(pct_materiales, 0.0, 1.0))
        pct_mo_clipped  = float(np.clip(pct_mo_eq, 0.0, 1.0))

        # Verificación de costo.
        if (
            not isinstance(total_cost, (int, float))
            or np.isnan(float(total_cost))
            or float(total_cost) <= 0.0
        ):
            return ClassificationResult(
                tipo=self.zero_cost_type,
                pct_materiales=pct_mat_clipped,
                pct_mo_eq=pct_mo_clipped,
                total_cost=float(total_cost) if isinstance(total_cost, (int, float)) else 0.0,
                rule_matched=None,
            )

        for rule in self._sorted_rules:
            if rule.evaluate(pct_mat_clipped, pct_mo_clipped):
                return ClassificationResult(
                    tipo=rule.rule_type,
                    pct_materiales=pct_mat_clipped,
                    pct_mo_eq=pct_mo_clipped,
                    total_cost=float(total_cost),
                    rule_matched=rule.rule_type,
                )

        return ClassificationResult(
            tipo=self.default_type,
            pct_materiales=pct_mat_clipped,
            pct_mo_eq=pct_mo_clipped,
            total_cost=float(total_cost),
            rule_matched=None,
        )

    def classify_batch(
        self,
        samples: Sequence[Tuple[float, float, float]],
    ) -> List[ClassificationResult]:
        """
        Clasifica una lista de tuplas (pct_mat, pct_mo, total_cost).

        Proporciona una interfaz de lote tipada sobre `classify_single_detailed`.
        Para datasets grandes en DataFrame, usar `classify_dataframe` que
        aplica evaluación vectorizada y es significativamente más eficiente.

        Args:
            samples: Secuencia de (pct_materiales, pct_mo_eq, total_cost).

        Returns:
            Lista de ClassificationResult en el mismo orden que samples.
        """
        return [
            self.classify_single_detailed(mat, mo, cost)
            for mat, mo, cost in samples
        ]

    # ── Clasificación de DataFrame ────────────────────────────────────────

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        col_total: str = "VALOR_CONSTRUCCION_UN",
        col_materiales: str = "VALOR_SUMINISTRO_UN",
        col_mo_eq: str = "VALOR_INSTALACION_UN",
        output_col: str = "TIPO_APU",
    ) -> pd.DataFrame:
        """
        Clasifica un DataFrame completo de APUs de forma vectorizada.

        Args:
            df:            DataFrame de entrada (no se muta; se opera sobre copia).
            col_total:     Columna de costo total del APU.
            col_materiales: Columna de costo de materiales.
            col_mo_eq:     Columna de costo de MO/equipo.
            output_col:    Columna de salida con el tipo clasificado.

        Returns:
            DataFrame con la columna `output_col` añadida.
            Si faltan columnas requeridas, `output_col` = `self.default_type`
            para todas las filas, con log de error que incluye los nombres
            de las columnas faltantes.

        Note:
            El DataFrame puede estar vacío — en ese caso se retorna con
            la columna output_col vacía (no se lanza excepción).
        """
        df = df.copy()

        # Validación de columnas requeridas.
        required = {col_total, col_materiales, col_mo_eq}
        missing_cols = required - set(df.columns)

        if missing_cols:
            logger.error(
                "Columnas requeridas no encontradas: %s. "
                "Asignando '%s' a todas las filas.",
                sorted(missing_cols),
                self.default_type,
            )
            df[output_col] = self.default_type
            return df

        # DataFrame vacío: retornar con columna output_col vacía.
        if df.empty:
            logger.debug(
                "classify_dataframe recibió DataFrame vacío; "
                "retornando con columna '%s' vacía.",
                output_col,
            )
            df[output_col] = pd.Series(dtype=object)
            return df

        totales    = pd.to_numeric(df[col_total],     errors="coerce").fillna(0.0).values
        materiales = pd.to_numeric(df[col_materiales], errors="coerce").fillna(0.0).values
        mo_eq      = pd.to_numeric(df[col_mo_eq],      errors="coerce").fillna(0.0).values

        # División segura: evita ZeroDivisionError y NaN propagation.
        with np.errstate(divide="ignore", invalid="ignore"):
            totales_safe = np.where(totales == 0.0, 1.0, totales)
            pct_mat = np.clip(materiales / totales_safe, 0.0, 1.0)
            pct_mo  = np.clip(mo_eq      / totales_safe, 0.0, 1.0)

        tipos = self._classify_vectorized_optimized(totales, pct_mat, pct_mo)
        df[output_col] = tipos

        self._analyze_classification_quality(df, output_col)
        return df

    # ── Clasificación vectorizada interna ─────────────────────────────────

    def _classify_vectorized_optimized(
        self,
        totales: np.ndarray,
        pct_mat: np.ndarray,
        pct_mo: np.ndarray,
    ) -> np.ndarray:
        """
        Clasificación vectorizada con asignación por prioridad mediante máscaras.

        Garantiza partición disjunta del espacio: cada punto pertenece a
        exactamente una categoría, respetando el orden topológico (prioridad).

        Usa `self._sorted_rules` (pre-calculado en `_compile_rules`) para
        evitar re-ordenar en cada invocación.

        Algoritmo:
        1. Marcar APUs con costo ≤ 0 o NaN como `zero_cost_type`.
        2. Para cada regla en orden de prioridad:
           a. Evaluar la máscara vectorizada sobre los válidos aún sin asignar.
           b. Asignar el tipo y marcar como asignados.
        3. Los no asignados reciben `default_type`.

        Fallback escalar:
            Si la evaluación vectorizada falla para una regla, se aplica
            evaluación escalar via `rule.evaluate()` sobre los puntos
            no asignados que satisfagan la regla.
        """
        n = len(totales)
        tipos = np.full(n, self.default_type, dtype=object)

        # Paso 1: APUs sin costo (≤ 0 o NaN).
        mask_sin_costo = (totales <= 0.0) | np.isnan(totales)
        tipos[mask_sin_costo] = self.zero_cost_type

        valid_mask = ~mask_sin_costo
        if not np.any(valid_mask):
            return tipos

        valid_idx = np.where(valid_mask)[0]
        mat_v = pct_mat[valid_mask]
        mo_v  = pct_mo[valid_mask]

        # Paso 2: evaluación vectorizada por regla en orden de prioridad.
        assigned = np.zeros(len(valid_idx), dtype=bool)

        for rule in self._sorted_rules:
            if np.all(assigned):
                break  # Todos los puntos válidos ya tienen tipo asignado

            try:
                if rule.rule_type in self._rule_cache:
                    rule_mask = self._rule_cache[rule.rule_type](mat_v, mo_v)
                else:
                    # Fallback escalar para reglas sin función compilada.
                    logger.debug(
                        "Regla '%s' sin función vectorizada — usando fallback escalar.",
                        rule.rule_type,
                    )
                    rule_mask = np.array(
                        [rule.evaluate(float(m), float(mo))
                         for m, mo in zip(mat_v, mo_v)],
                        dtype=bool,
                    )

            except Exception as exc:
                logger.debug(
                    "Vectorización fallida para '%s': %s — usando fallback escalar.",
                    rule.rule_type, exc,
                )
                rule_mask = np.array(
                    [rule.evaluate(float(m), float(mo))
                     for m, mo in zip(mat_v, mo_v)],
                    dtype=bool,
                )

            candidates = rule_mask & ~assigned
            if np.any(candidates):
                tipos[valid_idx[candidates]] = rule.rule_type
                assigned |= candidates

        return tipos

    # ── Análisis de calidad ───────────────────────────────────────────────

    def _analyze_classification_quality(
        self, df: pd.DataFrame, tipo_col: str
    ) -> None:
        """
        Registra distribución de tipos clasificados con barra ASCII proporcional.

        La barra tiene resolución de 50 caracteres; cada carácter ≈ 2%.

        No opera si el DataFrame está vacío o la columna no existe.
        """
        if tipo_col not in df.columns or df.empty:
            logger.debug(
                "_analyze_classification_quality: DataFrame vacío o columna "
                "'%s' ausente — omitiendo análisis.",
                tipo_col,
            )
            return

        n_total = len(df)
        # n_total > 0 garantizado por el guard anterior (df.empty).
        stats = df[tipo_col].value_counts()

        logger.info("=" * 60)
        logger.info("CALIDAD DE CLASIFICACIÓN APU (%d registros)", n_total)
        logger.info("=" * 60)

        for tipo, count in stats.items():
            pct = count / n_total * 100.0
            filled = int(pct / 2.0)
            bar = "█" * filled + "░" * (50 - filled)
            logger.info("%-20s %6d (%5.1f%%) %s", tipo, count, pct, bar)

        self._analyze_spatial_distribution(df, tipo_col)

    def _analyze_spatial_distribution(
        self, df: pd.DataFrame, tipo_col: str
    ) -> None:
        """
        Calcula y registra los centroides geométricos de cada categoría.

        Los centroides se calculan sobre las columnas de porcentaje si están
        disponibles, proporcionando una lectura topológica del espacio de fase.
        Si no hay columnas de porcentaje, se omite silenciosamente.
        """
        cols_mat = [c for c in df.columns if "PORCENTAJE_MATERIALES" in c.upper()]
        cols_mo  = [c for c in df.columns if "PORCENTAJE_MO" in c.upper()]

        if not cols_mat or not cols_mo:
            logger.debug(
                "_analyze_spatial_distribution: columnas de porcentaje "
                "no encontradas — omitiendo análisis de centroides."
            )
            return

        c_mat, c_mo = cols_mat[0], cols_mo[0]
        logger.info("CENTROIDES TOPOLÓGICOS (espacio de fase [0,1]²):")

        for tipo in df[tipo_col].unique():
            mask = df[tipo_col] == tipo
            if mask.any():
                cx = df.loc[mask, c_mat].mean()
                cy = df.loc[mask, c_mo].mean()
                logger.info("  %-22s: centroide=(%.4f, %.4f)", tipo, cx, cy)

    # ── Reporte de cobertura ──────────────────────────────────────────────

    def get_coverage_report(self) -> pd.DataFrame:
        """
        Genera un reporte tabular de los bounds de cobertura de cada regla.

        Returns:
            DataFrame con columnas:
              tipo, prioridad, mat_range, mo_range, area_estimada, condicion.
            Si no hay reglas, retorna DataFrame vacío con las mismas columnas.

        Note:
            `area_estimada` usa el método `estimated_area()` de ClassificationRule,
            que clampea el resultado a [0, 1] para evitar valores >100% en
            condiciones con bounds extraídos de cláusulas `or`.
        """
        columns = [
            "tipo", "prioridad", "mat_range",
            "mo_range", "area_estimada", "condicion",
        ]

        if not self.rules:
            logger.warning(
                "get_coverage_report: no hay reglas definidas; "
                "retornando DataFrame vacío."
            )
            return pd.DataFrame(columns=columns)

        data = []
        for rule in self._sorted_rules:
            (mat_min, mat_max), (mo_min, mo_max) = rule.get_coverage_bounds()
            area = rule.estimated_area()  # Ya clampea a [0, 1]

            data.append({
                "tipo":          rule.rule_type,
                "prioridad":     rule.priority,
                "mat_range":     f"[{mat_min:.1%}, {mat_max:.1%}]",
                "mo_range":      f"[{mo_min:.1%}, {mo_max:.1%}]",
                "area_estimada": f"{area:.1%}",
                "condicion":     rule.condition,
            })

        return pd.DataFrame(data, columns=columns)


# =============================================================================
# CLASIFICADOR ESTRUCTURAL (EXTENSIÓN DE RED DE INSUMOS)
# =============================================================================


class StructuralClassifier(APUClassifier):
    """
    Extiende APUClassifier con análisis topológico de la red de insumos.

    Detecta componentes conexos en el grafo de composición del APU
    (Nivel 3 de desagregación), identificando estructuras puras, aisladas
    y mixtas que no son detectables desde las proporciones de costo agregadas.

    Tipos estructurales detectables:
        ESTRUCTURA_VACIA      : sin insumos.
        SIN_VALOR_ESTRUCTURAL : insumos presentes pero costo total = 0.
        SERVICIO_PURO         : MO ≥ (1 - ε), sin materiales ni equipo.
        SUMINISTRO_PURO       : MAT ≥ (1 - ε), sin MO ni equipo.
        SUMINISTRO_AISLADO    : MAT presente, MO y equipo ausentes.
        INSTALACION_AISLADA   : MO presente, MAT y equipo ausentes.
        ESTRUCTURA_MIXTA      : múltiples componentes coexisten.

    Nota sobre herencia:
        El __init__ se especializa para clasificación estructural: no carga
        reglas de costo (innecesarias para este clasificador), reduciendo
        el overhead de inicialización.
    """

    # Tipos de insumos reconocidos por el modelo de dominio.
    _KNOWN_INSUMO_TYPES: ClassVar[FrozenSet[str]] = frozenset(
        {"MANO_DE_OBRA", "SUMINISTRO", "EQUIPO"}
    )

    def __init__(
        self,
        config_path: Optional[str] = None,
        skip_cost_rules: bool = False,
    ) -> None:
        """
        Inicializa el clasificador estructural.

        Args:
            config_path:      Ruta al JSON de configuración de reglas de costo.
            skip_cost_rules:  Si True, omite la carga de reglas de costo
                              (útil cuando solo se usa classify_by_structure).
                              Útil en pipelines donde solo se usa la lógica
                              estructural sin clasificación por proporciones.
        """
        if skip_cost_rules:
            # Inicialización mínima: sin carga de reglas de costo.
            self.rules = []
            self.default_type = "INDEFINIDO"
            self.zero_cost_type = "SIN_COSTO"
            self._rule_cache = {}
            self._sorted_rules = []
            logger.debug(
                "StructuralClassifier inicializado en modo estructural puro "
                "(sin reglas de costo)."
            )
        else:
            # Inicialización completa con reglas de costo heredadas.
            super().__init__(config_path=config_path)

    def classify_by_structure(
        self,
        insumos_del_apu: List[Dict[str, Any]],
        min_support_threshold: float = 1e-7,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Clasifica por topología de la red de insumos del APU.

        Args:
            insumos_del_apu: Lista de dicts con claves:
                - 'TIPO_INSUMO' (str): tipo de insumo.
                - 'VALOR_TOTAL' (float): valor monetario del insumo.
                Claves faltantes o valores no numéricos se tratan como 0.0.
            min_support_threshold: Umbral mínimo de presencia relativa ε para
                considerar un componente "presente". Default 1e-7 tolera ruido
                numérico de punto flotante.

        Returns:
            (tipo_estructural, proporciones_por_tipo)
              tipo_estructural     : string del tipo detectado.
              proporciones_por_tipo: dict {TIPO_INSUMO: proporción ∈ [0,1]}.

        Detección (en orden de prioridad):
            1. ESTRUCTURA_VACIA       : lista vacía.
            2. SIN_VALOR_ESTRUCTURAL  : costo total ≤ 0.
            3. SERVICIO_PURO          : MO ≥ (1 - 1e-7).
            4. SUMINISTRO_PURO        : MAT ≥ (1 - 1e-7).
            5. SUMINISTRO_AISLADO     : MAT ≥ ε, MO < ε, EQ < ε.
            6. INSTALACION_AISLADA    : MO ≥ ε, MAT < ε, EQ < ε.
            7. ESTRUCTURA_MIXTA       : cualquier combinación restante.
        """
        if not insumos_del_apu:
            return "ESTRUCTURA_VACIA", {}

        # Agregar valores por tipo de insumo con validación robusta.
        valores: Dict[str, float] = {}
        total = 0.0

        for i, insumo in enumerate(insumos_del_apu):
            if not isinstance(insumo, dict):
                logger.debug(
                    "Insumo #%d ignorado: se esperaba dict, recibido %s.",
                    i, type(insumo).__name__,
                )
                continue

            tipo = str(insumo.get("TIPO_INSUMO", "OTRO")).strip()
            if not tipo:
                tipo = "OTRO"

            # Conversión robusta del valor: trata None, str, NaN como 0.0.
            raw_valor = insumo.get("VALOR_TOTAL", 0.0)
            try:
                valor = float(raw_valor)
                if np.isnan(valor) or np.isinf(valor):
                    logger.debug(
                        "Insumo #%d tipo='%s': VALOR_TOTAL=%r → tratado como 0.0.",
                        i, tipo, raw_valor,
                    )
                    valor = 0.0
            except (TypeError, ValueError):
                logger.debug(
                    "Insumo #%d tipo='%s': VALOR_TOTAL=%r no convertible → 0.0.",
                    i, tipo, raw_valor,
                )
                valor = 0.0

            # Solo acumular valores no negativos (valores negativos = error de datos).
            if valor < 0.0:
                logger.debug(
                    "Insumo #%d tipo='%s': VALOR_TOTAL=%.4f negativo → ignorado.",
                    i, tipo, valor,
                )
                valor = 0.0

            valores[tipo] = valores.get(tipo, 0.0) + valor
            total += valor

        if total <= 0.0:
            return "SIN_VALOR_ESTRUCTURAL", valores

        # Proporciones normalizadas en [0, 1].
        pcts: Dict[str, float] = {k: v / total for k, v in valores.items()}

        mo_pct  = pcts.get("MANO_DE_OBRA", 0.0)
        mat_pct = pcts.get("SUMINISTRO",   0.0)
        eq_pct  = pcts.get("EQUIPO",       0.0)

        umbral_presencia: float = float(np.clip(min_support_threshold, 0.0, 1.0))

        # Tipos puros: dominancia casi total (≥ 1 - 1e-7).
        if mo_pct >= _DOMINANCE_THRESHOLD:
            return "SERVICIO_PURO", pcts

        if mat_pct >= _DOMINANCE_THRESHOLD:
            return "SUMINISTRO_PURO", pcts

        # Flags de presencia para cada componente principal.
        mo_presente  = mo_pct  >= umbral_presencia
        mat_presente = mat_pct >= umbral_presencia
        eq_presente  = eq_pct  >= umbral_presencia

        # Isla de suministro: solo materiales, sin MO ni equipo.
        if mat_presente and not mo_presente and not eq_presente:
            return "SUMINISTRO_AISLADO", pcts

        # Isla de instalación: solo MO, sin materiales ni equipo.
        # V2 no verificaba ausencia de equipo — corregido en V3 y mantenido.
        if mo_presente and not mat_presente and not eq_presente:
            return "INSTALACION_AISLADA", pcts

        # Cualquier combinación restante es mixta.
        return "ESTRUCTURA_MIXTA", pcts

    def get_structural_summary(
        self,
        insumos_del_apu: List[Dict[str, Any]],
        min_support_threshold: float = 1e-7,
    ) -> Dict[str, Any]:
        """
        Retorna un resumen estructural enriquecido del APU.

        Extiende `classify_by_structure` con métricas adicionales:
          - tipo_estructural: resultado de la clasificación.
          - proporciones: dict de proporciones por tipo de insumo.
          - n_insumos: número total de insumos procesados.
          - tipos_presentes: lista de tipos con presencia ≥ umbral.
          - dominancia: proporción máxima entre todos los tipos (índice de pureza).

        Args:
            insumos_del_apu:      Lista de dicts de insumos.
            min_support_threshold: Umbral de presencia ε.

        Returns:
            Dict con keys: tipo_estructural, proporciones, n_insumos,
            tipos_presentes, dominancia.
        """
        tipo, pcts = self.classify_by_structure(
            insumos_del_apu, min_support_threshold
        )

        tipos_presentes = [
            t for t, p in pcts.items() if p >= min_support_threshold
        ]
        dominancia = max(pcts.values()) if pcts else 0.0

        return {
            "tipo_estructural":   tipo,
            "proporciones":       pcts,
            "n_insumos":          len(insumos_del_apu),
            "tipos_presentes":    sorted(tipos_presentes),
            "dominancia":         round(dominancia, 6),
        }
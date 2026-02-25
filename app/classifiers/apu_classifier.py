"""
Este componente implementa un motor de inferencia determinista diseñado para
categorizar la naturaleza ontológica de un APU (Análisis de Precio Unitario).
Utiliza un sistema de reglas jerárquicas vectorizadas para particionar el espacio
vectorial definido por las proporciones de costos.

Metodología de Clasificación (V3 - Vectorial Robusta):
------------------------------------------------------
1. Compilación de Reglas (`_compile_rules`):
   Convierte las reglas de texto ("porcentaje > 60") en funciones vectorizadas de
   NumPy mediante pd.DataFrame.eval, que maneja correctamente la precedencia de
   operadores lógicos (and/or) sobre arrays sin reescritura de strings.

2. Validación de Cobertura Espacial:
   Utiliza un muestreo denso sobre el espacio [0,1]² para verificar matemáticamente
   que las reglas cubren el espacio de posibilidades, detectando "agujeros" lógicos.
   La medida de Lebesgue del conjunto descubierto ≈ |gaps| / grid_size².

3. Análisis Espacial:
   Calcula centroides geométricos de las categorías clasificadas para entender la
   distribución topológica de los costos en el espacio de fase.

4. StructuralClassifier:
   Extiende la lógica para considerar la topología de la red de insumos (Nivel 3),
   detectando estructuras como 'Islas' (Suministro Puro sin instalación).

Correcciones V3 respecto a V2:
-------------------------------
- Orden de inicialización corregido: _compile_rules antes de _validate_rules.
- `callable` reemplazado por `Callable[[np.ndarray, np.ndarray], np.ndarray]`.
- Alias `_analyze_coverage_voronoi` eliminado (frágil y obsoleto).
- Constante `grid_size²` derivada dinámicamente en validación de cobertura.
- `sorted_rules` calculado una vez en `_compile_rules`, no en cada clasificación.
- `classify_dataframe` reporta columnas faltantes con nombres explícitos.
- `StructuralClassifier.classify_by_structure` corrige branch INSTALACION_AISLADA.
- Contexto de `eval` reforzado con verificación de tipo de retorno.
- `_normalize_condition` colapsa espacios múltiples y tabs.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
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

# Tipo alias para la función vectorizada de regla.
RuleFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


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
    _CONDITION_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?:[\d\s\.\(\)><=!]"
        r"|porcentaje_materiales"
        r"|porcentaje_mo_eq"
        r"|and|or|not)+$"
    )

    # Patrones de extracción de bounds (orden importa: >= antes de >).
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
        # Mutación controlada única: normalización de la condición.
        self.condition = self._normalize_condition(self.condition)
        self._validate_syntax()

    # ── Normalización ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_condition(condition: str) -> str:
        """
        Normaliza operadores lógicos a minúsculas Python y colapsa espacios.

        Colapsar espacios múltiples y tabs previene que condiciones malformadas
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
        Valida sintaxis mediante análisis léxico exhaustivo en dos pasos.

        Paso 1 — Análisis léxico:
            Sustituye variables y literales conocidos; cualquier resto indica
            un token no autorizado.

        Paso 2 — Compilación AST:
            `compile()` detecta errores de sintaxis Python que pasen el léxico
            (ej. paréntesis desbalanceados con caracteres permitidos).
        """
        # Paso 1: análisis léxico.
        test_expr = self.condition
        for var in self._ALLOWED_VARS:
            test_expr = test_expr.replace(var, " ")
        # Remover literales numéricos.
        test_expr = re.sub(r"\b\d+\.?\d*\b", " ", test_expr)
        # Remover operadores y palabras clave permitidos.
        for token in (
            ">=", "<=", "==", "!=", ">", "<",
            "and", "or", "not", "(", ")",
        ):
            test_expr = test_expr.replace(token, " ")

        remaining = test_expr.strip()
        if remaining:
            raise ValueError(
                f"Condición contiene elementos no permitidos: '{remaining}'"
            )

        # Paso 2: compilación AST.
        try:
            compile(self.condition, "<condition>", "eval")
        except SyntaxError as exc:
            raise ValueError(
                f"Sintaxis Python inválida en condición: {exc}"
            ) from exc

    # ── Evaluación escalar ────────────────────────────────────────────────

    def evaluate(self, pct_materiales: float, pct_mo_eq: float) -> bool:
        """
        Evalúa la regla para un par escalar con contexto de ejecución seguro.

        El contexto restringe `__builtins__` a un dict vacío, bloqueando el
        acceso a built-ins de Python. El resultado se verifica explícitamente
        como `bool` para prevenir que objetos truthy no-booleanos pasen
        silenciosamente (ej. arrays NumPy de un elemento).
        """
        try:
            safe_context: Dict = {
                "__builtins__": {},
                "porcentaje_materiales": float(pct_materiales) * 100.0,
                "porcentaje_mo_eq": float(pct_mo_eq) * 100.0,
            }
            result = eval(self.condition, safe_context)  # noqa: S307
            # Verificación de tipo: rechaza arrays u objetos no escalares.
            if not isinstance(result, (bool, int, float, np.bool_)):
                raise TypeError(
                    f"Tipo de retorno inesperado: {type(result).__name__}"
                )
            return bool(result)
        except Exception as exc:
            logger.error(
                "Error evaluando regla '%s': %s", self.rule_type, exc
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
        estrictos (el primer match gana en re.search).

        Acumulación correcta:
        - `min` acumula con `max(actual, nuevo)` → cota inferior más restrictiva.
        - `max` acumula con `min(actual, nuevo)` → cota superior más restrictiva.
        """
        mat_min, mat_max = 0.0, 1.0
        mo_min,  mo_max  = 0.0, 1.0

        for pattern, var_type, bound_type, offset in self._BOUND_PATTERNS:
            match = re.search(pattern, self.condition)
            if not match:
                continue
            value = float(np.clip(
                float(match.group(1)) / 100.0 + offset, 0.0, 1.0
            ))
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

        return (mat_min, mat_max), (mo_min, mo_max)


# =============================================================================
# CLASIFICADOR PRINCIPAL
# =============================================================================


class APUClassifier:
    """
    Clasificador robusto de APUs basado en reglas vectorizadas y validación
    espacial sobre el simplex [0,1]².

    Orden de inicialización (invariante)
    ──────────────────────────────────────
    1. _load_config     → self.rules poblado y ordenado.
    2. _compile_rules   → self._rule_cache y self._sorted_rules listos.
    3. _validate_rules  → usa _rule_cache (ya disponible) para cobertura.

    Este orden garantiza que la validación de cobertura opera sobre reglas
    ya compiladas, evitando el bug de V2 donde _rule_cache estaba vacío
    durante la validación.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.rules: List[ClassificationRule] = []
        self.default_type: str = "INDEFINIDO"
        self.zero_cost_type: str = "SIN_COSTO"

        # Cache de funciones vectorizadas: rule_type → RuleFunc.
        self._rule_cache: Dict[str, RuleFunc] = {}

        # Reglas ordenadas por prioridad — calculadas una vez en _compile_rules.
        self._sorted_rules: List[ClassificationRule] = []

        # Invariante de inicialización: load → compile → validate.
        self._load_config(config_path)
        self._compile_rules()   # ← antes de _validate_rules
        self._validate_rules()  # ← usa _rule_cache ya poblado

    # ── Carga de configuración ────────────────────────────────────────────

    def _load_config(self, config_path: Optional[str]) -> None:
        """
        Carga reglas desde archivo JSON o usa las reglas por defecto.

        El archivo debe tener la estructura:
        {
            "apu_classification_rules": {
                "rules": [...],
                "default_type": "INDEFINIDO",
                "zero_cost_type": "SIN_COSTO"
            }
        }
        """
        if not config_path or not Path(config_path).exists():
            logger.warning(
                "Configuración no encontrada en '%s', usando reglas por defecto",
                config_path,
            )
            self._load_default_rules()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rules_config = config.get("apu_classification_rules", {})
            loaded: List[ClassificationRule] = []

            for rule_dict in rules_config.get("rules", []):
                try:
                    rule = ClassificationRule(
                        rule_type=rule_dict["type"],
                        priority=rule_dict.get("priority", 99),
                        condition=rule_dict["condition"],
                        description=rule_dict.get("description", ""),
                    )
                    loaded.append(rule)
                except (KeyError, ValueError) as exc:
                    logger.warning("Regla inválida omitida: %s", exc)

            if not loaded:
                logger.warning(
                    "No se cargaron reglas válidas desde '%s'; "
                    "usando reglas por defecto.",
                    config_path,
                )
                self._load_default_rules()
                return

            self.rules = sorted(loaded, key=lambda r: r.priority)
            self.default_type = rules_config.get("default_type", "INDEFINIDO")
            self.zero_cost_type = rules_config.get("zero_cost_type", "SIN_COSTO")

        except (json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Error cargando configuración desde '%s': %s",
                config_path, exc,
            )
            self._load_default_rules()

    def _load_default_rules(self) -> None:
        """
        Carga el conjunto mínimo de reglas que cubren el espacio [0,1]².

        Las reglas por defecto están diseñadas para ser exhaustivas:
        OBRA_COMPLETA actúa como residuo topológico (prioridad 4),
        garantizando que ningún punto válido quede sin clasificar.
        """
        self.rules = [
            ClassificationRule(
                "INSTALACION", 1,
                "porcentaje_mo_eq >= 60.0",
                "Predomina MO/equipo (≥60%)",
            ),
            ClassificationRule(
                "SUMINISTRO", 2,
                "porcentaje_materiales >= 60.0",
                "Predomina materiales (≥60%)",
            ),
            ClassificationRule(
                "CONSTRUCCION_MIXTO", 3,
                "(porcentaje_materiales >= 40.0 and porcentaje_materiales < 60.0)"
                " or (porcentaje_mo_eq >= 40.0 and porcentaje_mo_eq < 60.0)",
                "Composición mixta (40–60%)",
            ),
            ClassificationRule(
                "OBRA_COMPLETA", 4,
                "porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0",
                "Cobertura residual (cubre todo el espacio)",
            ),
        ]

    # ── Compilación de reglas ─────────────────────────────────────────────

    def _compile_rules(self) -> None:
        """
        Compila las reglas a funciones NumPy vectorizadas y ordena por prioridad.

        Almacena `_sorted_rules` como lista ordenada reutilizable, evitando
        re-ordenar en cada llamada a `_classify_vectorized_optimized`.
        """
        self._sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        self._rule_cache.clear()

        for rule in self._sorted_rules:
            try:
                self._rule_cache[rule.rule_type] = (
                    self._create_vectorized_function(rule.condition)
                )
            except Exception as exc:
                logger.warning(
                    "No se pudo compilar regla '%s': %s",
                    rule.rule_type, exc,
                )

    def _create_vectorized_function(self, condition: str) -> RuleFunc:
        """
        Compila una condición de texto a función NumPy vectorizada.

        Estrategia: `pd.DataFrame.eval` maneja correctamente la precedencia
        de operadores lógicos (and/or) sobre arrays de NumPy sin necesidad
        de reescribir la string a operadores bitwise (&/|/~).

        La función retornada opera en escala [0, 100] (consistente con la
        semántica de las condiciones de texto).

        Retorna
        ───────
        RuleFunc : (x: np.ndarray, y: np.ndarray) → np.ndarray[bool]
            donde x = porcentaje_materiales ∈ [0, 1]
                  y = porcentaje_mo_eq      ∈ [0, 1]
        """
        # Captura `condition` por cierre — inmutable tras la compilación.
        def rule_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            try:
                df_temp = pd.DataFrame({
                    "porcentaje_materiales": x * 100.0,
                    "porcentaje_mo_eq":      y * 100.0,
                })
                result = df_temp.eval(condition)
                # Garantizar dtype bool para operaciones bitwise posteriores.
                return result.values.astype(bool)
            except Exception as exc:
                logger.debug(
                    "Error en evaluación vectorizada de '%s': %s",
                    condition, exc,
                )
                return np.zeros(len(x), dtype=bool)

        return rule_func

    # ── Validación de reglas ──────────────────────────────────────────────

    def _validate_rules(self) -> None:
        """
        Valida coherencia del sistema de reglas y cobertura topológica.

        Comprobaciones
        ──────────────
        1. Al menos una regla definida.
        2. Sin tipos duplicados (advertencia, no error — pueden ser intencionados).
        3. Cobertura del espacio [0,1]² ≥ umbral aceptable.
        """
        if not self.rules:
            raise ValueError("No hay reglas de clasificación definidas")

        # Detección de tipos duplicados.
        type_counts: Dict[str, int] = {}
        for rule in self.rules:
            type_counts[rule.rule_type] = type_counts.get(rule.rule_type, 0) + 1
        duplicates = {t for t, c in type_counts.items() if c > 1}
        if duplicates:
            logger.warning("Tipos de regla duplicados detectados: %s", duplicates)

        # Validación de cobertura espacial.
        gaps, total_points = self._sample_uncovered_regions(
            _DEFAULT_GRID_SIZE
        )
        if gaps:
            gap_ratio = len(gaps) / total_points
            logger.warning(
                "Reglas no cubren %.1f%% del espacio [0,1]² "
                "(%d puntos sin cobertura de %d muestreados).",
                gap_ratio * 100.0, len(gaps), total_points,
            )
        else:
            logger.info("Cobertura topológica completa validada sobre grilla %d×%d.",
                        _DEFAULT_GRID_SIZE, _DEFAULT_GRID_SIZE)

    def _sample_uncovered_regions(
        self,
        grid_size: int = _DEFAULT_GRID_SIZE,
    ) -> Tuple[List[Tuple[float, float]], int]:
        """
        Muestrea cobertura del simplex [0,1]² mediante evaluación tensorizada.

        Retorna
        ───────
        (gaps, total_points) : Tuple[List[Tuple[float, float]], int]
            gaps         — puntos (mat, mo) sin cobertura por ninguna regla.
            total_points — número total de puntos muestreados (grid_size²).

        Complejidad: O(grid_size² × R) donde R = número de reglas.

        Fallback escalar
        ────────────────
        Si la evaluación vectorizada falla para una regla (ej. pd.eval no soporta
        la expresión), se aplica evaluación escalar por índice sobre los puntos
        aún descubiertos. Se registra en DEBUG para visibilidad de rendimiento.
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
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) > 0:
            logger.debug(
                "Fallback escalar para %d punto(s) no cubiertos por evaluación vectorizada.",
                len(uncovered_idx),
            )
            for idx in uncovered_idx:
                for rule in self._sorted_rules:
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

        Parámetros
        ──────────
        pct_materiales : proporción de materiales ∈ [0, 1].
        pct_mo_eq      : proporción de MO/equipo ∈ [0, 1].
        total_cost     : costo total del APU (≤ 0 → SIN_COSTO).

        Retorna la primera regla satisfecha en orden de prioridad,
        o `self.default_type` si ninguna aplica.
        """
        pct_materiales = float(np.clip(pct_materiales, 0.0, 1.0))
        pct_mo_eq = float(np.clip(pct_mo_eq, 0.0, 1.0))

        if total_cost <= 0.0 or (isinstance(total_cost, float) and np.isnan(total_cost)):
            return self.zero_cost_type

        for rule in self._sorted_rules:
            if rule.evaluate(pct_materiales, pct_mo_eq):
                return rule.rule_type

        return self.default_type

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

        Parámetros
        ──────────
        df            : DataFrame de entrada (no se muta; se opera sobre copia).
        col_total     : columna de costo total del APU.
        col_materiales: columna de costo de materiales.
        col_mo_eq     : columna de costo de MO/equipo.
        output_col    : columna de salida con el tipo clasificado.

        Retorna
        ───────
        DataFrame con la columna `output_col` añadida.
        Si faltan columnas requeridas, `output_col` = `self.default_type`
        para todas las filas, con log de error que incluye los nombres
        de las columnas faltantes.
        """
        df = df.copy()
        required = {col_total, col_materiales, col_mo_eq}
        missing_cols = required - set(df.columns)

        if missing_cols:
            logger.error(
                "Columnas requeridas no encontradas en el DataFrame: %s. "
                "Clasificación asignada a '%s'.",
                sorted(missing_cols),
                self.default_type,
            )
            df[output_col] = self.default_type
            return df

        totales = pd.to_numeric(df[col_total],     errors="coerce").fillna(0.0).values
        materiales = pd.to_numeric(df[col_materiales], errors="coerce").fillna(0.0).values
        mo_eq = pd.to_numeric(df[col_mo_eq],      errors="coerce").fillna(0.0).values

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

        Algoritmo
        ─────────
        1. Marcar APUs con costo ≤ 0 o NaN como `zero_cost_type`.
        2. Para cada regla en orden de prioridad:
           a. Evaluar la máscara vectorizada sobre los válidos aún sin asignar.
           b. Asignar el tipo y marcar como asignados.
        3. Los no asignados reciben `default_type`.
        """
        n = len(totales)
        tipos = np.full(n, self.default_type, dtype=object)

        # Paso 1: APUs sin costo.
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
                break  # todos los puntos válidos ya tienen tipo asignado

            if rule.rule_type not in self._rule_cache:
                continue

            try:
                rule_mask = self._rule_cache[rule.rule_type](mat_v, mo_v)
            except Exception as exc:
                logger.debug(
                    "Vectorización fallida para '%s', usando fallback escalar: %s",
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
        """
        if tipo_col not in df.columns:
            return

        n_total = max(len(df), 1)
        stats = df[tipo_col].value_counts()

        logger.info("=" * 60)
        logger.info("CALIDAD DE CLASIFICACIÓN APU")
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
        """
        cols_mat = [c for c in df.columns if "PORCENTAJE_MATERIALES" in c.upper()]
        cols_mo  = [c for c in df.columns if "PORCENTAJE_MO" in c.upper()]

        if not cols_mat or not cols_mo:
            return

        c_mat, c_mo = cols_mat[0], cols_mo[0]
        logger.info("CENTROIDES TOPOLÓGICOS:")

        for tipo in df[tipo_col].unique():
            mask = df[tipo_col] == tipo
            if mask.any():
                cx = df.loc[mask, c_mat].mean()
                cy = df.loc[mask, c_mo].mean()
                logger.info("  %-20s: (%.2f, %.2f)", tipo, cx, cy)

    # ── Reporte de cobertura ──────────────────────────────────────────────

    def get_coverage_report(self) -> pd.DataFrame:
        """
        Genera un reporte tabular de los bounds de cobertura de cada regla.

        Retorna DataFrame con columnas:
        tipo, prioridad, mat_range, mo_range, area_estimada, condicion.

        Si no hay reglas, retorna DataFrame vacío con las mismas columnas
        (en lugar de fallar silenciosamente).
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
            area = max(0.0, mat_max - mat_min) * max(0.0, mo_max - mo_min)
            data.append({
                "tipo":          rule.rule_type,
                "prioridad":     rule.priority,
                "mat_range":     f"[{mat_min:.0%}, {mat_max:.0%}]",
                "mo_range":      f"[{mo_min:.0%}, {mo_max:.0%}]",
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

    Tipos estructurales
    ───────────────────
    ESTRUCTURA_VACIA      : sin insumos.
    SIN_VALOR_ESTRUCTURAL : insumos presentes pero costo total = 0.
    SERVICIO_PURO         : MO ≥ (1 - ε), sin materiales ni equipo.
    SUMINISTRO_PURO       : MAT ≥ (1 - ε), sin MO ni equipo.
    SUMINISTRO_AISLADO    : MAT presente, MO y equipo ausentes (isla topológica).
    INSTALACION_AISLADA   : MO presente, MAT y equipo ausentes.
    ESTRUCTURA_MIXTA      : múltiples componentes coexisten.
    """

    def classify_by_structure(
        self,
        insumos_del_apu: List[Dict],
        min_support_threshold: float = 1e-7,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Clasifica por topología de la red de insumos del APU.

        Parámetros
        ──────────
        insumos_del_apu       : lista de dicts con claves 'TIPO_INSUMO' y
                                'VALOR_TOTAL'.
        min_support_threshold : umbral mínimo de presencia relativa para
                                considerar un componente "presente" (ε).
                                Default 1e-7 tolera ruido numérico de punto
                                flotante sin clasificar erróneamente componentes
                                ausentes.

        Retorna
        ───────
        (tipo_estructural, proporciones_por_tipo)
            tipo_estructural    : string del tipo detectado.
            proporciones_por_tipo: dict {TIPO_INSUMO: proporción ∈ [0,1]}.

        Lógica de detección
        ────────────────────
        Umbral de dominancia = 1 - 1e-7 (≈100% de pureza).
        Umbral de presencia  = min_support_threshold.

        SUMINISTRO_AISLADO   : MAT ≥ ε,  MO < ε,  EQ < ε.
        INSTALACION_AISLADA  : MO  ≥ ε,  MAT < ε, EQ < ε.
            (V2 no verificaba ausencia de equipo — corregido aquí.)
        ESTRUCTURA_MIXTA     : cualquier otra combinación.
        """
        if not insumos_del_apu:
            return "ESTRUCTURA_VACIA", {}

        # Agregar valores por tipo de insumo.
        valores: Dict[str, float] = {}
        total = 0.0
        for insumo in insumos_del_apu:
            tipo = str(insumo.get("TIPO_INSUMO", "OTRO"))
            valor = float(insumo.get("VALOR_TOTAL", 0.0))
            valores[tipo] = valores.get(tipo, 0.0) + valor
            total += valor

        if total <= 0.0:
            return "SIN_VALOR_ESTRUCTURAL", valores

        # Proporciones normalizadas.
        pcts: Dict[str, float] = {k: v / total for k, v in valores.items()}

        mo_pct  = pcts.get("MANO_DE_OBRA", 0.0)
        mat_pct = pcts.get("SUMINISTRO",   0.0)
        eq_pct  = pcts.get("EQUIPO",       0.0)

        umbral_presencia: float = min_support_threshold

        # Tipos puros (dominancia casi total).
        if mo_pct >= _DOMINANCE_THRESHOLD:
            return "SERVICIO_PURO", pcts

        if mat_pct >= _DOMINANCE_THRESHOLD:
            return "SUMINISTRO_PURO", pcts

        # Flags de presencia para cada componente.
        mo_presente  = mo_pct  >= umbral_presencia
        mat_presente = mat_pct >= umbral_presencia
        eq_presente  = eq_pct  >= umbral_presencia

        # Isla de suministro: sólo materiales, sin MO ni equipo.
        if mat_presente and not mo_presente and not eq_presente:
            return "SUMINISTRO_AISLADO", pcts

        # Isla de instalación: sólo MO, sin materiales ni equipo.
        # V2 no verificaba ausencia de equipo — corregido.
        if mo_presente and not mat_presente and not eq_presente:
            return "INSTALACION_AISLADA", pcts

        # Cualquier combinación restante es mixta.
        return "ESTRUCTURA_MIXTA", pcts
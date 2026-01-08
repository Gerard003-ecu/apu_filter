import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClassificationRule:
    """Estructura para reglas de clasificación con validación integrada."""

    rule_type: str
    priority: int
    condition: str
    description: str

    _ALLOWED_VARS: ClassVar[frozenset] = None
    _CONDITION_PATTERN: ClassVar[re.Pattern] = None

    def __post_init__(self):
        """Normaliza y valida la condición al instanciar."""
        # Inicializar constantes de clase si no existen
        if ClassificationRule._ALLOWED_VARS is None:
            ClassificationRule._ALLOWED_VARS = frozenset(
                {"porcentaje_materiales", "porcentaje_mo_eq"}
            )
            ClassificationRule._CONDITION_PATTERN = re.compile(
                r"^[\d\s\.\(\)><=!]+$|porcentaje_materiales|porcentaje_mo_eq|and|or|not"
            )

        self.condition = self._normalize_condition(self.condition)
        self._validate_syntax()

    def _normalize_condition(self, condition: str) -> str:
        """
        Normaliza operadores lógicos SQL-style a sintaxis Python.

        Transforma AND/OR/NOT (case-insensitive) a and/or/not.
        """
        condition = re.sub(r"\bAND\b", "and", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bOR\b", "or", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bNOT\b", "not", condition, flags=re.IGNORECASE)
        return condition.strip()

    def _validate_syntax(self) -> None:
        """
        Valida que la condición sea sintácticamente segura y evaluable.

        Raises:
            ValueError: Si la condición contiene elementos no permitidos.
        """
        test_expr = self.condition

        # Remover elementos válidos para detectar residuos peligrosos
        allowed_vars = ClassificationRule._ALLOWED_VARS
        if allowed_vars is None:
            allowed_vars = frozenset({"porcentaje_materiales", "porcentaje_mo_eq"})

        for var in allowed_vars:
            test_expr = test_expr.replace(var, " ")

        # Remover literales numéricos (int y float)
        test_expr = re.sub(r"\b\d+\.?\d*\b", " ", test_expr)

        # Remover operadores y palabras clave permitidas
        allowed_tokens = [
            ">=",
            "<=",
            "==",
            "!=",
            ">",
            "<",
            "and",
            "or",
            "not",
            "(",
            ")",
        ]
        for token in allowed_tokens:
            test_expr = test_expr.replace(token, " ")

        remaining = test_expr.strip()
        if remaining:
            raise ValueError(
                f"Condición contiene elementos no permitidos: '{remaining}' "
                f"en expresión '{self.condition}'"
            )

        # Verificar sintaxis Python válida
        try:
            compile(self.condition, "<condition>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Sintaxis inválida en condición: {e}")

    def evaluate(self, pct_materiales: float, pct_mo_eq: float) -> bool:
        """
        Evalúa la condición con los porcentajes proporcionados.

        Args:
            pct_materiales: Fracción de materiales [0, 1].
            pct_mo_eq: Fracción de MO+equipo [0, 1].

        Returns:
            True si la condición se satisface.
        """
        try:
            # Escalar a porcentaje [0, 100] para consistencia con condiciones
            safe_context = {
                "__builtins__": {},
                "porcentaje_materiales": pct_materiales * 100.0,
                "porcentaje_mo_eq": pct_mo_eq * 100.0,
            }
            return bool(eval(self.condition, safe_context))
        except Exception as e:
            logger.error(
                f"Error evaluando regla '{self.rule_type}' | "
                f"mat={pct_materiales:.2%}, mo={pct_mo_eq:.2%}: {e}"
            )
            return False

    def get_coverage_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Extrae límites heurísticos de cobertura en el espacio [0,1]².

        Útil para análisis topológico de la partición del espacio.

        Returns:
            ((mat_min, mat_max), (mo_min, mo_max)) normalizados a [0, 1].
        """
        mat_min, mat_max = 0.0, 1.0
        mo_min, mo_max = 0.0, 1.0

        bound_patterns = [
            (r"porcentaje_materiales\s*>=\s*(\d+\.?\d*)", "mat", "min"),
            (r"porcentaje_materiales\s*>\s*(\d+\.?\d*)", "mat", "min"),
            (r"porcentaje_materiales\s*<=\s*(\d+\.?\d*)", "mat", "max"),
            (r"porcentaje_materiales\s*<\s*(\d+\.?\d*)", "mat", "max"),
            (r"porcentaje_mo_eq\s*>=\s*(\d+\.?\d*)", "mo", "min"),
            (r"porcentaje_mo_eq\s*>\s*(\d+\.?\d*)", "mo", "min"),
            (r"porcentaje_mo_eq\s*<=\s*(\d+\.?\d*)", "mo", "max"),
            (r"porcentaje_mo_eq\s*<\s*(\d+\.?\d*)", "mo", "max"),
        ]

        for pattern, var_type, bound_type in bound_patterns:
            match = re.search(pattern, self.condition)
            if match:
                value = float(match.group(1)) / 100.0
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

        return ((mat_min, mat_max), (mo_min, mo_max))


class APUClassifier:
    """Clasificador robusto de APUs basado en reglas configurables."""

    def __init__(self, config_path: Optional[str] = None):
        self.rules: List[ClassificationRule] = []
        self.default_type = "INDEFINIDO"
        self.zero_cost_type = "SIN_COSTO"

        self._load_config(config_path)
        self._validate_rules()

    def _load_config(self, config_path: Optional[str]) -> None:
        """
        Carga reglas desde archivo JSON con validación robusta.

        Args:
            config_path: Ruta al archivo de configuración.
        """
        if not config_path or not Path(config_path).exists():
            logger.warning("⚠️ Configuración no encontrada, usando reglas por defecto")
            self._load_default_rules()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rules_config = config.get("apu_classification_rules", {})
            loaded_count = 0

            for idx, rule_dict in enumerate(rules_config.get("rules", [])):
                try:
                    if "type" not in rule_dict or "condition" not in rule_dict:
                        raise KeyError(
                            "Faltan campos obligatorios 'type' o 'condition'"
                        )

                    rule = ClassificationRule(
                        rule_type=rule_dict["type"],
                        priority=rule_dict.get("priority", 99),
                        condition=rule_dict["condition"],
                        description=rule_dict.get("description", ""),
                    )
                    self.rules.append(rule)
                    loaded_count += 1
                except (KeyError, ValueError) as e:
                    logger.warning(f"⚠️ Regla índice {idx} inválida, omitida: {e}")

            if not self.rules:
                logger.warning("⚠️ Sin reglas válidas en config, usando por defecto")
                self._load_default_rules()
                return

            self.rules.sort(key=lambda r: r.priority)
            self.default_type = rules_config.get("default_type", "INDEFINIDO")
            self.zero_cost_type = rules_config.get("zero_cost_type", "SIN_COSTO")

            logger.info(f"✅ {loaded_count} reglas de clasificación cargadas")

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON malformado: {e}")
            self._load_default_rules()
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            self._load_default_rules()

    def _load_default_rules(self) -> None:
        """
        Reglas por defecto diseñadas para cubrir el espacio [0,1]² completamente.

        La partición sigue un orden de prioridad que garantiza clasificación
        determinista sin ambigüedades topológicas.
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
                    "(porcentaje_materiales >= 40.0 and porcentaje_materiales < 60.0) or "
                    "(porcentaje_mo_eq >= 40.0 and porcentaje_mo_eq < 60.0)"
                ),
                description="Composición mixta (40-60%)",
            ),
            ClassificationRule(
                rule_type="OBRA_COMPLETA",
                priority=4,
                condition="porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0",
                description="Cobertura residual universal",
            ),
        ]
        self.default_type = "INDEFINIDO"
        self.zero_cost_type = "SIN_COSTO"

    def _validate_rules(self) -> None:
        """
        Valida coherencia topológica del conjunto de reglas.

        Raises:
            ValueError: Si no existen reglas.
        """
        if not self.rules:
            raise ValueError("No hay reglas de clasificación definidas")

        # Detectar tipos duplicados
        types = [r.rule_type for r in self.rules]
        duplicates = {t for t in types if types.count(t) > 1}
        if duplicates:
            logger.warning(f"⚠️ Tipos duplicados: {duplicates}")

        # Análisis de cobertura del espacio
        uncovered = self._sample_uncovered_regions(grid_size=20)
        if uncovered:
            logger.warning(
                f"⚠️ Detectadas {len(uncovered)} regiones sin cobertura. "
                f"Fallback: '{self.default_type}'"
            )

    def _sample_uncovered_regions(
        self, grid_size: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Muestrea el espacio [0,1]² para detectar huecos en la cobertura.

        Args:
            grid_size: Resolución de la grilla de muestreo.

        Returns:
            Lista de puntos (pct_mat, pct_mo) no cubiertos.
        """
        uncovered = []
        step = 1.0 / max(grid_size - 1, 1)

        for i in range(grid_size):
            for j in range(grid_size):
                pct_mat, pct_mo = i * step, j * step

                if not any(r.evaluate(pct_mat, pct_mo) for r in self.rules):
                    uncovered.append((pct_mat, pct_mo))

        return uncovered

    def classify_single(
        self,
        pct_materiales: float,
        pct_mo_eq: float,
        total_cost: float = 1.0,
    ) -> str:
        """
        Clasifica un único APU.

        Args:
            pct_materiales: Fracción de materiales [0, 1].
            pct_mo_eq: Fracción de MO+equipo [0, 1].
            total_cost: Costo total (0 indica APU sin costo).

        Returns:
            Tipo de APU según reglas.
        """
        # Caso especial: sin costo
        if total_cost <= 0 or (isinstance(total_cost, float) and np.isnan(total_cost)):
            return self.zero_cost_type

        # Normalizar entradas al dominio válido
        pct_materiales = float(np.clip(pct_materiales, 0.0, 1.0))
        pct_mo_eq = float(np.clip(pct_mo_eq, 0.0, 1.0))

        for rule in self.rules:
            if rule.evaluate(pct_materiales, pct_mo_eq):
                return rule.rule_type

        return self.default_type

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        col_total: str = "VALOR_CONSTRUCCION_UN",
        col_materiales: str = "VALOR_SUMINISTRO_UN",
        col_mo_eq: str = "VALOR_INSTALACION_UN",
        output_col: str = "TIPO_APU",
    ) -> pd.DataFrame:
        """
        Clasifica DataFrame completo con operaciones vectorizadas.

        Args:
            df: DataFrame con costos.
            col_total: Columna de costo total.
            col_materiales: Columna de materiales.
            col_mo_eq: Columna de MO+equipo.
            output_col: Columna de salida.

        Returns:
            DataFrame con clasificación añadida.
        """
        df = df.copy()
        required = [col_total, col_materiales, col_mo_eq]
        missing = [c for c in required if c not in df.columns]

        if missing:
            logger.error(f"❌ Columnas faltantes: {missing}")
            df[output_col] = self.default_type
            return df

        # Extraer arrays numéricos
        totales = pd.to_numeric(df[col_total], errors="coerce").fillna(0).values
        materiales = pd.to_numeric(df[col_materiales], errors="coerce").fillna(0).values
        mo_eq = pd.to_numeric(df[col_mo_eq], errors="coerce").fillna(0).values

        # Clasificación vectorizada
        df[output_col] = self._classify_vectorized(totales, materiales, mo_eq)

        self._log_classification_stats(df[output_col])
        return df

    def _classify_vectorized(
        self,
        totales: np.ndarray,
        materiales: np.ndarray,
        mo_eq: np.ndarray,
    ) -> np.ndarray:
        """
        Clasificación vectorizada O(n) usando máscaras NumPy.

        Args:
            totales: Array de costos totales.
            materiales: Array de costos de materiales.
            mo_eq: Array de costos MO+equipo.

        Returns:
            Array de tipos clasificados.
        """
        n = len(totales)
        tipos = np.full(n, self.default_type, dtype=object)

        # Máscara: APUs sin costo
        mask_sin_costo = (totales <= 0) | np.isnan(totales)
        tipos[mask_sin_costo] = self.zero_cost_type

        # Calcular porcentajes solo para válidos
        mask_validos = ~mask_sin_costo
        if not mask_validos.any():
            return tipos

        totales_safe = np.where(mask_validos, totales, 1.0)
        pct_mat = np.clip(materiales / totales_safe, 0.0, 1.0)
        pct_mo = np.clip(mo_eq / totales_safe, 0.0, 1.0)

        # Aplicar reglas secuencialmente con máscara de pendientes
        pendientes = mask_validos.copy()

        for rule in self.rules:
            if not pendientes.any():
                break

            cumple = self._eval_rule_vectorized(rule, pct_mat, pct_mo, pendientes)
            tipos[cumple] = rule.rule_type
            pendientes &= ~cumple

        return tipos

    def _eval_rule_vectorized(
        self,
        rule: ClassificationRule,
        pct_mat: np.ndarray,
        pct_mo: np.ndarray,
        mask_activos: np.ndarray,
    ) -> np.ndarray:
        """
        Evalúa regla de forma vectorizada sobre arrays.

        Args:
            rule: Regla a evaluar.
            pct_mat: Porcentajes de materiales [0, 1].
            pct_mo: Porcentajes de MO [0, 1].
            mask_activos: Máscara de elementos a considerar.

        Returns:
            Máscara booleana de elementos que cumplen la regla.
        """
        try:
            # Pandas eval soporta 'and'/'or' y vectorización nativa (engine='python' si numexpr no está)
            # Creamos un contexto con los valores escalados a porcentaje [0-100]
            ctx = {
                "porcentaje_materiales": pct_mat * 100.0,
                "porcentaje_mo_eq": pct_mo * 100.0,
            }

            # Evaluamos usando pandas.eval que maneja 'and'/'or' correctamente en vectores
            # engine='python' asegura compatibilidad con la sintaxis and/or de Python
            result = pd.eval(rule.condition, local_dict=ctx, engine="python")

            # Manejar resultado escalar vs array
            if np.isscalar(result):
                result = np.full(len(pct_mat), result, dtype=bool)

            return np.asarray(result, dtype=bool) & mask_activos

        except Exception as e:
            logger.error(f"Error vectorizado en '{rule.rule_type}': {e}")
            return np.zeros(len(pct_mat), dtype=bool)

    def _log_classification_stats(self, series: pd.Series) -> None:
        """Registra estadísticas con visualización mejorada."""
        total = len(series)
        if total == 0:
            logger.warning("⚠️ DataFrame vacío")
            return

        stats = series.value_counts()

        logger.info("═" * 55)
        logger.info("📊 ESTADÍSTICAS DE CLASIFICACIÓN")
        logger.info("═" * 55)

        for tipo, count in stats.items():
            pct = (count / total) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            logger.info(f"  {tipo:<18} │ {count:>6} │ {pct:>5.1f}% │ {bar}")

        logger.info("─" * 55)
        logger.info(f"  {'TOTAL':<18} │ {total:>6}")
        logger.info("═" * 55)

        # Alertas
        for alert_type, label in [
            (self.default_type, "sin clasificar"),
            (self.zero_cost_type, "sin costo"),
        ]:
            count = stats.get(alert_type, 0)
            if count > 0:
                logger.warning(
                    f"⚠️ {count} APUs ({count / total * 100:.1f}%) {label} [{alert_type}]"
                )

    def get_coverage_report(self) -> pd.DataFrame:
        """
        Genera reporte de cobertura topológica de las reglas.

        Returns:
            DataFrame con bounds estimados y área de cada regla.
        """
        data = []
        for rule in self.rules:
            (mat_min, mat_max), (mo_min, mo_max) = rule.get_coverage_bounds()
            area = (mat_max - mat_min) * (mo_max - mo_min)

            data.append(
                {
                    "tipo": rule.rule_type,
                    "prioridad": rule.priority,
                    "mat_range": f"[{mat_min:.0%}, {mat_max:.0%}]",
                    "mo_range": f"[{mo_min:.0%}, {mo_max:.0%}]",
                    "area_estimada": f"{area:.1%}",
                    "condicion": rule.condition,
                }
            )

        return pd.DataFrame(data)

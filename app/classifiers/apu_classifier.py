"""
Este componente implementa un motor de inferencia determinista diseñado para
categorizar la naturaleza ontológica de un APU (Análisis de Precio Unitario).
Utiliza un sistema de reglas jerárquicas vectorizadas para particionar el espacio
vectorial definido por las proporciones de costos.

Metodología de Clasificación (V2 - Vectorial):
----------------------------------------------
1. Compilación de Reglas (`_compile_rules`):
   Convierte las reglas de texto ("porcentaje > 60") en funciones vectorizadas de NumPy
   altamente optimizadas, permitiendo clasificar millones de registros en milisegundos.

2. Validación de Cobertura Espacial:
   Utiliza un muestreo denso sobre el espacio [0,1]² para verificar matemáticamente
   que las reglas cubren el espacio de posibilidades, detectando "agujeros" lógicos
   sin la complejidad computacional de Voronoi en altas dimensiones.

3. Análisis Espacial:
   Calcula centroides geométricos de las categorías clasificadas para entender la
   distribución topológica de los costos en el espacio de fase.

4. StructuralClassifier:
   Extiende la lógica para considerar la topología de la red de insumos (Nivel 3),
   detectando estructuras como 'Islas' (Suministro Puro sin instalación).
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
# from scipy.spatial import Voronoi, Delaunay # Removed: Using Grid Sampling optimization

logger = logging.getLogger(__name__)


@dataclass
class ClassificationRule:
    """Estructura para reglas de clasificación con validación integrada."""

    rule_type: str
    priority: int
    condition: str
    description: str

    _ALLOWED_VARS: ClassVar[frozenset] = frozenset({"porcentaje_materiales", "porcentaje_mo_eq"})
    _CONDITION_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?:[\d\s\.\(\)><=!]|porcentaje_materiales|porcentaje_mo_eq|and|or|not)+$"
    )

    def __post_init__(self):
        """Normaliza y valida la condición al instanciar (thread-safe)."""
        self.condition = self._normalize_condition(self.condition)
        self._validate_syntax()

    def _normalize_condition(self, condition: str) -> str:
        condition = re.sub(r"\bAND\b", "and", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bOR\b", "or", condition, flags=re.IGNORECASE)
        condition = re.sub(r"\bNOT\b", "not", condition, flags=re.IGNORECASE)
        return condition.strip()

    def _validate_syntax(self) -> None:
        """Valida sintaxis mediante análisis léxico exhaustivo."""
        test_expr = self.condition

        for var in ClassificationRule._ALLOWED_VARS:
            test_expr = test_expr.replace(var, " ")

        test_expr = re.sub(r"\b\d+\.?\d*\b", " ", test_expr)

        for token in (">=", "<=", "==", "!=", ">", "<", "and", "or", "not", "(", ")"):
            test_expr = test_expr.replace(token, " ")

        remaining = test_expr.strip()
        if remaining:
            raise ValueError(f"Condición contiene elementos no permitidos: '{remaining}'")

        try:
            compile(self.condition, "<condition>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Sintaxis inválida en condición: {e}") from e

    def evaluate(self, pct_materiales: float, pct_mo_eq: float) -> bool:
        try:
            safe_context = {
                "__builtins__": {},
                "porcentaje_materiales": pct_materiales * 100.0,
                "porcentaje_mo_eq": pct_mo_eq * 100.0,
            }
            return bool(eval(self.condition, safe_context))
        except Exception as e:
            logger.error(f"Error evaluando regla '{self.rule_type}': {e}")
            return False

    def get_coverage_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Extrae bounds del espacio de cobertura diferenciando operadores estrictos.

        Aplica ε-offset para operadores estrictos (>, <) preservando la topología
        del conjunto abierto vs cerrado en el espacio [0,1]².
        """
        mat_min, mat_max = 0.0, 1.0
        mo_min, mo_max = 0.0, 1.0
        epsilon = 1e-6

        bound_patterns = (
            (r"porcentaje_materiales\s*>=\s*(\d+\.?\d*)", "mat", "min", 0.0),
            (r"porcentaje_materiales\s*>\s*(\d+\.?\d*)",  "mat", "min", epsilon),
            (r"porcentaje_materiales\s*<=\s*(\d+\.?\d*)", "mat", "max", 0.0),
            (r"porcentaje_materiales\s*<\s*(\d+\.?\d*)",  "mat", "max", -epsilon),
            (r"porcentaje_mo_eq\s*>=\s*(\d+\.?\d*)",      "mo",  "min", 0.0),
            (r"porcentaje_mo_eq\s*>\s*(\d+\.?\d*)",       "mo",  "min", epsilon),
            (r"porcentaje_mo_eq\s*<=\s*(\d+\.?\d*)",      "mo",  "max", 0.0),
            (r"porcentaje_mo_eq\s*<\s*(\d+\.?\d*)",       "mo",  "max", -epsilon),
        )

        for pattern, var_type, bound_type, offset in bound_patterns:
            match = re.search(pattern, self.condition)
            if match:
                value = np.clip(float(match.group(1)) / 100.0 + offset, 0.0, 1.0)
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
    """Clasificador robusto de APUs basado en reglas vectorizadas y validación espacial."""

    def __init__(self, config_path: Optional[str] = None):
        self.rules: List[ClassificationRule] = []
        self.default_type = "INDEFINIDO"
        self.zero_cost_type = "SIN_COSTO"
        self._rule_cache: Dict[str, callable] = {}

        self._load_config(config_path)
        self._validate_rules()
        self._compile_rules()

    def _load_config(self, config_path: Optional[str]) -> None:
        if not config_path or not Path(config_path).exists():
            logger.warning("⚠️ Configuración no encontrada, usando reglas por defecto")
            self._load_default_rules()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            rules_config = config.get("apu_classification_rules", {})
            for rule_dict in rules_config.get("rules", []):
                try:
                    rule = ClassificationRule(
                        rule_type=rule_dict["type"],
                        priority=rule_dict.get("priority", 99),
                        condition=rule_dict["condition"],
                        description=rule_dict.get("description", ""),
                    )
                    self.rules.append(rule)
                except Exception as e:
                    logger.warning(f"⚠️ Regla inválida: {e}")

            if not self.rules:
                self._load_default_rules()
                return

            self.rules.sort(key=lambda r: r.priority)
            self.default_type = rules_config.get("default_type", "INDEFINIDO")
            self.zero_cost_type = rules_config.get("zero_cost_type", "SIN_COSTO")

        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            self._load_default_rules()

    def _load_default_rules(self) -> None:
        self.rules = [
            ClassificationRule("INSTALACION", 1, "porcentaje_mo_eq >= 60.0", "Predomina MO/equipo (≥60%)"),
            ClassificationRule("SUMINISTRO", 2, "porcentaje_materiales >= 60.0", "Predomina materiales (≥60%)"),
            ClassificationRule(
                "CONSTRUCCION_MIXTO", 3,
                "(porcentaje_materiales >= 40.0 and porcentaje_materiales < 60.0) or (porcentaje_mo_eq >= 40.0 and porcentaje_mo_eq < 60.0)",
                "Composición mixta (40-60%)"
            ),
            ClassificationRule("OBRA_COMPLETA", 4, "porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0", "Cobertura residual"),
        ]

    def _compile_rules(self) -> None:
        for rule in self.rules:
            try:
                self._rule_cache[rule.rule_type] = self._create_vectorized_function(rule.condition)
            except Exception as e:
                logger.warning(f"No se pudo compilar regla {rule.rule_type}: {e}")

    def _create_vectorized_function(self, condition: str) -> callable:
        """
        Compila condición a función NumPy vectorizada con traducción de operadores.

        Transforma operadores lógicos Python (and/or/not) a operadores bitwise
        NumPy (&/|/~) para broadcasting correcto sobre arrays.
        """
        # Primero reemplazamos las variables para evitar reemplazos accidentales dentro de nombres
        # No, mejor usamos una estrategia más robusta para los operadores lógicos.
        # El problema principal es que 'and' y 'or' tienen menor precedencia que '&' y '|'
        # y eval() con arrays numpy requiere paréntesis extra si se mezcla con comparaciones.
        # E.g. (a > 5) & (b < 3) funciona, pero a > 5 & b < 3 no (bitwise & gana).

        # Una alternativa más robusta es usar pandas.eval que maneja esto nativamente,
        # como en la versión anterior. O envolver comparaciones.

        # Dado el requisito de "lógica matemática robusta", usar pd.eval es más seguro
        # para expresiones arbitrarias de usuario.

        def rule_func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            try:
                # Usamos pandas eval que maneja "and", "or" y precedencia correctamente
                # sin necesidad de reescribir la string a bitwise operators.
                df_temp = pd.DataFrame({
                    "porcentaje_materiales": x * 100.0,
                    "porcentaje_mo_eq": y * 100.0
                })
                return df_temp.eval(condition).values
            except Exception as e:
                # logger.debug(f"Error en evaluación vectorizada: {e}")
                return np.zeros_like(x, dtype=bool)

        return rule_func

    def _validate_rules(self) -> None:
        """Valida coherencia del sistema de reglas y cobertura topológica."""
        if not self.rules:
            raise ValueError("No hay reglas de clasificación definidas")

        type_counts = {}
        for rule in self.rules:
            type_counts[rule.rule_type] = type_counts.get(rule.rule_type, 0) + 1

        duplicates = {t for t, c in type_counts.items() if c > 1}
        if duplicates:
            logger.warning(f"⚠️ Tipos duplicados: {duplicates}")

        coverage_gaps = self._sample_uncovered_regions()
        if coverage_gaps:
            gap_ratio = len(coverage_gaps) / 2500.0
            logger.warning(
                f"⚠️ Reglas no cubren {gap_ratio:.1%} del espacio. "
                f"Puntos sin cobertura: {len(coverage_gaps)}"
            )
        else:
            logger.info("✓ Cobertura topológica completa validada.")

    def _sample_uncovered_regions(self, grid_size: int = 50) -> List[Tuple[float, float]]:
        """
        Muestrea cobertura del simplex [0,1]² mediante evaluación tensorizada.

        Complejidad: O(R) donde R = número de reglas (vs O(n²·R) anterior).
        La medida de Lebesgue del conjunto descubierto ≈ len(result) / grid_size².
        """
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        X_flat, Y_flat = X.ravel(), Y.ravel()

        covered = np.zeros(X_flat.shape, dtype=bool)

        for rule in self.rules:
            if rule.rule_type in self._rule_cache:
                try:
                    covered |= self._rule_cache[rule.rule_type](X_flat, Y_flat)
                except Exception:
                    pass

        remaining_uncovered = np.where(~covered)[0]
        if len(remaining_uncovered) > 0:
            for idx in remaining_uncovered:
                for rule in self.rules:
                    if rule.evaluate(float(X_flat[idx]), float(Y_flat[idx])):
                        covered[idx] = True
                        break

        uncovered_mask = ~covered
        if not np.any(uncovered_mask):
            return []

        return list(zip(X_flat[uncovered_mask].tolist(), Y_flat[uncovered_mask].tolist()))

    # Alias para compatibilidad interna
    _analyze_coverage_voronoi = _sample_uncovered_regions

    def classify_single(self, pct_materiales: float, pct_mo_eq: float, total_cost: float = 1.0) -> str:
        """
        Clasifica un APU escalar usando evaluación directa (sin overhead vectorial).
        """
        pct_materiales = float(np.clip(pct_materiales, 0.0, 1.0))
        pct_mo_eq = float(np.clip(pct_mo_eq, 0.0, 1.0))

        if total_cost <= 0 or np.isnan(total_cost):
            return self.zero_cost_type

        for rule in self.rules:
            if rule.evaluate(pct_materiales, pct_mo_eq):
                return rule.rule_type

        return self.default_type

    def classify_dataframe(self, df: pd.DataFrame, col_total: str = "VALOR_CONSTRUCCION_UN",
                          col_materiales: str = "VALOR_SUMINISTRO_UN", col_mo_eq: str = "VALOR_INSTALACION_UN",
                          output_col: str = "TIPO_APU") -> pd.DataFrame:
        df = df.copy()
        required = [col_total, col_materiales, col_mo_eq]
        if any(c not in df.columns for c in required):
            logger.error(f"❌ Columnas faltantes")
            df[output_col] = self.default_type
            return df

        totales = pd.to_numeric(df[col_total], errors='coerce').fillna(0).values
        materiales = pd.to_numeric(df[col_materiales], errors='coerce').fillna(0).values
        mo_eq = pd.to_numeric(df[col_mo_eq], errors='coerce').fillna(0).values

        with np.errstate(divide='ignore', invalid='ignore'):
            totales_safe = np.where(totales == 0, 1.0, totales)
            pct_mat = np.clip(materiales / totales_safe, 0.0, 1.0)
            pct_mo = np.clip(mo_eq / totales_safe, 0.0, 1.0)

        tipos = self._classify_vectorized_optimized(totales, pct_mat, pct_mo)
        df[output_col] = tipos

        self._analyze_classification_quality(df, output_col)
        return df

    def _classify_vectorized_optimized(
        self, totales: np.ndarray, pct_mat: np.ndarray, pct_mo: np.ndarray
    ) -> np.ndarray:
        """
        Clasificación vectorizada con asignación por prioridad mediante máscaras booleanas.

        Garantiza partición disjunta del espacio: cada punto pertenece a exactamente
        una categoría, respetando el orden topológico (prioridad).
        """
        n = len(totales)
        tipos = np.full(n, self.default_type, dtype=object)

        mask_sin_costo = (totales <= 0) | np.isnan(totales)
        tipos[mask_sin_costo] = self.zero_cost_type

        valid_mask = ~mask_sin_costo
        if not np.any(valid_mask):
            return tipos

        valid_idx = np.where(valid_mask)[0]
        mat_v = pct_mat[valid_mask]
        mo_v = pct_mo[valid_mask]

        sorted_rules = sorted(self.rules, key=lambda r: r.priority)

        rule_masks = {}
        for rule in sorted_rules:
            if rule.rule_type in self._rule_cache:
                try:
                    rule_masks[rule.rule_type] = self._rule_cache[rule.rule_type](mat_v, mo_v)
                except Exception as e:
                    logger.debug(f"Vectorización fallida para {rule.rule_type}: {e}")
                    rule_masks[rule.rule_type] = np.array(
                        [rule.evaluate(m, mo) for m, mo in zip(mat_v, mo_v)], dtype=bool
                    )

        assigned = np.zeros(len(valid_idx), dtype=bool)
        for rule in sorted_rules:
            if rule.rule_type not in rule_masks:
                continue

            candidates = rule_masks[rule.rule_type] & ~assigned
            if np.any(candidates):
                tipos[valid_idx[candidates]] = rule.rule_type
                assigned |= candidates

        return tipos

    def _analyze_classification_quality(self, df: pd.DataFrame, tipo_col: str) -> None:
        if tipo_col not in df.columns: return
        stats = df[tipo_col].value_counts()
        logger.info("="*60 + "\n📊 CALIDAD CLASIFICACIÓN\n" + "="*60)
        for t, c in stats.items():
            pct = c / len(df) * 100
            bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
            logger.info(f"{t:20} {c:6} ({pct:5.1f}%) {bar}")
        self._analyze_spatial_distribution(df, tipo_col)

    def _analyze_spatial_distribution(self, df: pd.DataFrame, tipo_col: str) -> None:
        cols_mat = [c for c in df.columns if "PORCENTAJE_MATERIALES" in c.upper()]
        cols_mo = [c for c in df.columns if "PORCENTAJE_MO" in c.upper()]
        if cols_mat and cols_mo:
            c_mat, c_mo = cols_mat[0], cols_mo[0]
            logger.info("\n📍 CENTROIDES TOPOLÓGICOS:")
            for tipo in df[tipo_col].unique():
                mask = df[tipo_col] == tipo
                if mask.any():
                    cx, cy = df.loc[mask, c_mat].mean(), df.loc[mask, c_mo].mean()
                    logger.info(f"  {tipo:20}: ({cx:.2f}, {cy:.2f})")

    def get_coverage_report(self) -> pd.DataFrame:
        data = []
        for rule in self.rules:
            (mat_min, mat_max), (mo_min, mo_max) = rule.get_coverage_bounds()
            area = (mat_max - mat_min) * (mo_max - mo_min)
            data.append({
                "tipo": rule.rule_type,
                "prioridad": rule.priority,
                "mat_range": f"[{mat_min:.0%}, {mat_max:.0%}]",
                "mo_range": f"[{mo_min:.0%}, {mo_max:.0%}]",
                "area_estimada": f"{area:.1%}",
                "condicion": rule.condition
            })
        return pd.DataFrame(data)

class StructuralClassifier(APUClassifier):
    def classify_by_structure(
        self,
        insumos_del_apu: List[Dict],
        min_support_threshold: float = 1e-7
    ) -> Tuple[str, Dict[str, float]]:
        """
        Clasifica por topología de la red de insumos.

        Detecta componentes conexos en el grafo de composición:
        - SERVICIO_PURO: Componente MO dominante (≈100%)
        - SUMINISTRO_PURO: Componente MAT dominante (≈100%)
        - SUMINISTRO_AISLADO: MAT presente, MO ausente (isla topológica)
        - INSTALACION_AISLADA: MO presente, MAT ausente
        - ESTRUCTURA_MIXTA: Múltiples componentes existen simultáneamente
        """
        if not insumos_del_apu:
            return "ESTRUCTURA_VACIA", {}

        valores: Dict[str, float] = {}
        total = 0.0

        for insumo in insumos_del_apu:
            tipo = insumo.get("TIPO_INSUMO", "OTRO")
            valor = float(insumo.get("VALOR_TOTAL", 0.0))
            valores[tipo] = valores.get(tipo, 0.0) + valor
            total += valor

        if total <= 0:
            return "SIN_VALOR_ESTRUCTURAL", valores

        pcts = {k: v / total for k, v in valores.items()}

        mo_pct = pcts.get("MANO_DE_OBRA", 0.0)
        mat_pct = pcts.get("SUMINISTRO", 0.0)
        eq_pct = pcts.get("EQUIPO", 0.0)

        # Topología estricta: La pureza requiere ausencia total de otros componentes
        # Se usa un epsilon muy pequeño para tolerar ruido numérico
        umbral_dominancia = 1.0 - 1e-7
        umbral_presencia = min_support_threshold

        if mo_pct >= umbral_dominancia:
            return "SERVICIO_PURO", pcts

        if mat_pct >= umbral_dominancia:
            return "SUMINISTRO_PURO", pcts

        mo_presente = mo_pct >= umbral_presencia
        mat_presente = mat_pct >= umbral_presencia
        eq_presente = eq_pct >= umbral_presencia

        if mat_presente and not mo_presente and not eq_presente:
            return "SUMINISTRO_AISLADO", pcts

        if mo_presente and not mat_presente:
            return "INSTALACION_AISLADA", pcts

        return "ESTRUCTURA_MIXTA", pcts

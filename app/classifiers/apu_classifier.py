import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ClassificationRule:
    """Estructura para reglas de clasificación"""
    rule_type: str
    priority: int
    condition: str
    description: str

    def evaluate(self, pct_materiales: float, pct_mo_eq: float) -> bool:
        """Evalúa la condición con los porcentajes"""
        try:
            # Variables disponibles para evaluación
            porcentaje_materiales = pct_materiales * 100  # Convertir a porcentaje
            porcentaje_mo_eq = pct_mo_eq * 100

            # Evaluar condición (usar eval con contexto restringido)
            safe_dict = {
                'porcentaje_materiales': porcentaje_materiales,
                'porcentaje_mo_eq': porcentaje_mo_eq
            }
            return eval(self.condition, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            logger.error(f"Error evaluando regla {self.rule_type}: {e}")
            return False


class APUClassifier:
    """Clasificador robusto de APUs basado en reglas configurables"""

    def __init__(self, config_path: Optional[str] = None):
        self.rules: List[ClassificationRule] = []
        self.default_type = "INDEFINIDO"
        self.zero_cost_type = "SIN_COSTO"

        self._load_config(config_path)
        self._validate_rules()

    def _load_config(self, config_path: Optional[str]):
        """Carga reglas desde archivo JSON"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                rules_config = config.get('apu_classification_rules', {})

                # Cargar reglas
                for rule_dict in rules_config.get('rules', []):
                    rule = ClassificationRule(
                        rule_type=rule_dict['type'],
                        priority=rule_dict.get('priority', 99),
                        condition=rule_dict['condition'],
                        description=rule_dict.get('description', '')
                    )
                    self.rules.append(rule)

                # Ordenar por prioridad (menor = primero)
                self.rules.sort(key=lambda x: x.priority)

                # Cargar configuraciones adicionales
                self.default_type = rules_config.get('default_type', 'INDEFINIDO')
                self.zero_cost_type = rules_config.get('zero_cost_type', 'SIN_COSTO')

                logger.info(f"✅ Cargadas {len(self.rules)} reglas de clasificación")

            except Exception as e:
                logger.error(f"❌ Error cargando configuración: {e}")
                self._load_default_rules()
        else:
            logger.warning("⚠️ No se encontró archivo de configuración, usando reglas por defecto")
            self._load_default_rules()

    def _load_default_rules(self):
        """Reglas por defecto (más permisivas)"""
        self.rules = [
            ClassificationRule(
                rule_type="INSTALACION",
                priority=1,
                condition="porcentaje_mo_eq >= 60.0",
                description="Predomina mano de obra/equipo"
            ),
            ClassificationRule(
                rule_type="SUMINISTRO",
                priority=2,
                condition="porcentaje_materiales >= 60.0",
                description="Predomina materiales"
            ),
            ClassificationRule(
                rule_type="CONSTRUCCION_MIXTO",
                priority=3,
                condition="(porcentaje_materiales >= 40.0 AND porcentaje_materiales <= 60.0) OR (porcentaje_mo_eq >= 40.0 AND porcentaje_mo_eq <= 60.0)",
                description="Balance entre materiales y MO"
            ),
            ClassificationRule(
                rule_type="OBRA_COMPLETA",
                priority=4,
                condition="porcentaje_materiales > 0 AND porcentaje_mo_eq > 0",
                description="Cualquier APU con costos válidos"
            )
        ]
        self.default_type = "INDEFINIDO"
        self.zero_cost_type = "SIN_COSTO"

    def _validate_rules(self):
        """Valida que las reglas sean coherentes"""
        if not self.rules:
            raise ValueError("No hay reglas de clasificación definidas")

        # Verificar que no haya tipos duplicados
        types = [rule.rule_type for rule in self.rules]
        if len(types) != len(set(types)):
            logger.warning("⚠️ Hay tipos de reglas duplicados")

    def classify_single(self, pct_materiales: float, pct_mo_eq: float,
                       total_cost: float = 1.0) -> str:
        """
        Clasifica un único APU basado en sus porcentajes

        Args:
            pct_materiales: Porcentaje de materiales (0-1)
            pct_mo_eq: Porcentaje de MO+equipo (0-1)
            total_cost: Costo total para detectar APUs sin costo

        Returns:
            Tipo de APU clasificado
        """
        # Caso especial: sin costo
        if total_cost <= 0:
            return self.zero_cost_type

        # Aplicar reglas en orden de prioridad
        for rule in self.rules:
            if rule.evaluate(pct_materiales, pct_mo_eq):
                return rule.rule_type

        # Fallback final
        return self.default_type

    def classify_dataframe(self, df: pd.DataFrame,
                          col_total: str = 'VALOR_CONSTRUCCION_UN',
                          col_materiales: str = 'VALOR_SUMINISTRO_UN',
                          col_mo_eq: str = 'VALOR_INSTALACION_UN',
                          output_col: str = 'TIPO_APU') -> pd.DataFrame:
        """
        Clasifica un DataFrame completo de APUs

        Args:
            df: DataFrame con costos de APUs
            col_total: Columna con costo total
            col_materiales: Columna con costo de materiales
            col_mo_eq: Columna con costo de MO+equipo
            output_col: Columna de salida para el tipo

        Returns:
            DataFrame con columna de clasificación añadida
        """
        df = df.copy()

        # Verificar columnas requeridas
        required_cols = [col_total, col_materiales, col_mo_eq]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"❌ Columnas faltantes: {missing_cols}")
            df[output_col] = self.default_type
            return df

        # Calcular porcentajes evitando división por cero
        df['_total_safe'] = df[col_total].replace(0, np.nan)

        df['_pct_materiales'] = df[col_materiales] / df['_total_safe']
        df['_pct_mo_eq'] = df[col_mo_eq] / df['_total_safe']

        # Rellenar NaN con 0 (para APUs sin costo)
        df[['_pct_materiales', '_pct_mo_eq']] = df[['_pct_materiales', '_pct_mo_eq']].fillna(0)

        # Aplicar clasificación a cada fila
        df[output_col] = df.apply(
            lambda row: self.classify_single(
                row['_pct_materiales'],
                row['_pct_mo_eq'],
                row[col_total]
            ),
            axis=1
        )

        # Limpiar columnas temporales
        df.drop(['_total_safe', '_pct_materiales', '_pct_mo_eq'],
                axis=1, inplace=True, errors='ignore')

        # Estadísticas
        self._log_classification_stats(df[output_col])

        return df

    def _log_classification_stats(self, series: pd.Series):
        """Registra estadísticas de clasificación"""
        stats = series.value_counts()
        total = len(series)

        logger.info("📊 ESTADÍSTICAS DE CLASIFICACIÓN:")
        for tipo, count in stats.items():
            percentage = (count / total) * 100
            logger.info(f"  {tipo}: {count} APUs ({percentage:.1f}%)")

        # Alertas importantes
        indefinidos = stats.get(self.default_type, 0)
        if indefinidos > 0:
            pct_indef = (indefinidos / total) * 100
            logger.warning(f"⚠️ {indefinidos} APUs ({pct_indef:.1f}%) clasificados como {self.default_type}")

        sin_costo = stats.get(self.zero_cost_type, 0)
        if sin_costo > 0:
            logger.warning(f"⚠️ {sin_costo} APUs sin costo ({self.zero_cost_type})")

"""
Módulo de Motor de Gobernanza y Validación Semántica.

Este módulo implementa las reglas de negocio y validación semántica para asegurar
la coherencia lógica de los APUs (Análisis de Precios Unitarios) más allá de
la simple validación estructural de datos. Utiliza una ontología de dominio
para verificar que los insumos correspondan lógicamente al tipo de actividad.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.constants import ColumnNames

logger = logging.getLogger(__name__)


@dataclass
class ComplianceReport:
    """Reporte de cumplimiento de gobernanza."""

    score: float = 100.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    semantic_alerts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "PASS"

    def add_violation(self, type_: str, message: str, severity: str = "ERROR"):
        """Registra una violación de reglas."""
        self.violations.append(
            {"type": type_, "message": message, "severity": severity}
        )
        # Penalización simple
        if severity == "ERROR":
            self.score = max(0.0, self.score - 5.0)
        elif severity == "WARNING":
            self.score = max(0.0, self.score - 1.0)


class GovernanceEngine:
    """
    Motor de reglas para validar la integridad semántica y estructural.

    Implementa la validación basada en ontología definida en Fase 3 Data Mesh.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.ontology: Dict[str, Any] = {}
        self.semantic_policy: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Carga la configuración y la ontología."""
        # Cargar Ontología
        try:
            ontology_path = self.config_dir / "ontology.json"
            if ontology_path.exists():
                with open(ontology_path, "r", encoding="utf-8") as f:
                    self.ontology = json.load(f)
                logger.info(f"✅ Ontología cargada desde {ontology_path}")
            else:
                logger.warning(f"⚠️ No se encontró ontología en {ontology_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando ontología: {e}")

        # Cargar Data Contract (Políticas)
        try:
            contract_path = self.config_dir / "data_contract.yaml"
            if contract_path.exists():
                # Nota: Parseo simple de yaml si no tenemos pyyaml instalado.
                # En un entorno real usaríamos yaml.safe_load.
                import yaml

                with open(contract_path, "r", encoding="utf-8") as f:
                    contract = yaml.safe_load(f)
                    self.semantic_policy = contract.get("semantic_policy", {})
            else:
                logger.warning(f"⚠️ No se encontró data_contract en {contract_path}")
        except ImportError:
            logger.warning("⚠️ PyYAML no instalado, cargando políticas por defecto.")
            self.semantic_policy = {"enable_ontology_check": True}
        except Exception as e:
            logger.error(f"❌ Error cargando data contract: {e}")

    def load_ontology(self, path: str):
        """Carga una ontología personalizada desde una ruta específica."""
        try:
            ontology_path = Path(path)
            if ontology_path.exists():
                with open(ontology_path, "r", encoding="utf-8") as f:
                    self.ontology = json.load(f)
                logger.info(f"✅ Ontología recargada desde {path}")
            else:
                logger.error(f"❌ Archivo de ontología no encontrado: {path}")
        except Exception as e:
            logger.error(f"❌ Error cargando ontología personalizada: {e}")

    def check_semantic_coherence(self, dataframe: pd.DataFrame) -> ComplianceReport:
        """
        Verifica la coherencia semántica de los APUs y sus insumos.

        Lógica:
        1. Agrupa los insumos por APU.
        2. Infiere el dominio del APU basado en su descripción.
        3. Verifica si los insumos contienen palabras clave prohibidas para ese dominio.

        Args:
            dataframe: DataFrame conteniendo APUs e insumos (merged).

        Returns:
            ComplianceReport con las violaciones detectadas.

        """
        report = ComplianceReport()

        if not self.semantic_policy.get("enable_ontology_check", False):
            logger.info("ℹ️ Validación semántica desactivada por política.")
            return report

        if dataframe is None or dataframe.empty:
            logger.warning("⚠️ DataFrame vacío para validación semántica.")
            return report

        # Verificar columnas necesarias
        required_cols = [
            ColumnNames.CODIGO_APU,
            ColumnNames.DESCRIPCION_APU,
            ColumnNames.DESCRIPCION_INSUMO,
        ]
        missing = [col for col in required_cols if col not in dataframe.columns]
        if missing:
            msg = f"Faltan columnas para validación semántica: {missing}"
            logger.error(msg)
            report.add_violation("SCHEMA_ERROR", msg, "ERROR")
            return report

        logger.info("🧠 Iniciando Validación Semántica de APUs...")

        # Obtener dominios de la ontología
        domains = self.ontology.get("domains", {})
        if not domains:
            logger.warning("⚠️ Ontología vacía o sin dominios definidos.")
            return report

        # Agrupar insumos por APU
        # Iteramos por grupos para eficiencia
        grouped = dataframe.groupby(ColumnNames.CODIGO_APU)

        for apu_code, group in grouped:
            # Asumimos que la descripción del APU es consistente en el grupo
            desc_apu = str(group[ColumnNames.DESCRIPCION_APU].iloc[0]).upper()

            # Inferir dominio
            detected_domain = None
            for domain_name, rules in domains.items():
                # Heurística simple: si el nombre del dominio está en la descripción
                if domain_name in desc_apu:
                    detected_domain = domain_name
                    break

            if not detected_domain:
                continue  # No se pudo inferir dominio, saltamos validación

            # Obtener insumos del APU
            insumos_descs = (
                group[ColumnNames.DESCRIPCION_INSUMO].fillna("").astype(str).str.upper()
            )
            rules = domains[detected_domain]
            forbidden = rules.get("forbidden_keywords", [])
            required = rules.get("required_keywords", [])

            # Chequeo 1: Palabras Prohibidas
            for bad_keyword in forbidden:
                # Buscar insumos que contengan la palabra prohibida
                mask = insumos_descs.str.contains(bad_keyword, regex=False)
                if mask.any():
                    violating_insumos = insumos_descs[mask].unique().tolist()
                    msg = (
                        f"APU '{apu_code}' ({detected_domain}) contiene insumos "
                        f"prohibidos ('{bad_keyword}'): {violating_insumos[:3]}"
                    )
                    report.add_violation(
                        "SEMANTIC_INCONSISTENCY", msg, "WARNING"
                    )  # Warning por ahora

            # Chequeo 2: Palabras Requeridas
            # Interpretación: Si es Cimentación, DEBE tener algo de Concreto, Acero, etc.
            all_insumos_text = " ".join(insumos_descs)

            # Contar cuántas keywords requeridas están presentes
            found_count = sum(1 for kw in required if kw in all_insumos_text)

            if found_count == 0 and required:
                msg = (
                    f"APU '{apu_code}' ({detected_domain}) no parece contener insumos "
                    f"esperados como: {required}"
                )
                report.add_violation("SEMANTIC_INCOMPLETENESS", msg, "WARNING")

        logger.info(
            f"✅ Validación Semántica completada. Score: {report.score}. "
            f"Violaciones: {len(report.violations)}"
        )
        return report

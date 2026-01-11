"""
Este componente implementa el motor de Gobernanza Computacional que asegura que
los "Productos de Datos" (presupuestos, APUs) cumplan con las leyes del sistema
antes de ser aceptados. Transforma la validación pasiva en una auditoría activa
basada en ontologías y reglas de negocio.

Capacidades y Protocolos:
-------------------------
1. Política como Código (Policy as Code):
   Evalúa reglas declarativas (inspiradas en OPA/Rego) para determinar el cumplimiento
   de estándares de calidad, seguridad y negocio. Genera un `ComplianceReport` con
   un veredicto de severidad (PASS, WARNING, FAIL).

2. Validación Semántica (Ontology Check):
   Utiliza un Grafo de Conocimiento (Ontología) para verificar que los insumos
   pertenezcan al dominio correcto del APU (ej. detectar "Ladrillo" en un APU de
   "Instalaciones Eléctricas"). Infiere el contexto semántico para detectar anomalías
   que la validación sintáctica ignora.

3. Sistema de Penalización (Scorecard):
   Calcula un puntaje de gobernanza (Governance Score) aplicando penalizaciones
   ponderadas por severidad. Un puntaje bajo activa protocolos de rechazo automático
   o auditoría manual.

4. Gestión de Contratos de Datos:
   Verifica que la estructura y el contenido de los datos respeten los contratos
   definidos (campos obligatorios, tipos de datos, restricciones de dominio).
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.constants import ColumnNames

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Niveles de severidad válidos para violaciones."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ComplianceStatus(str, Enum):
    """Estados posibles del reporte de cumplimiento."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# Constantes de penalización configurables
SEVERITY_PENALTIES: Dict[Severity, float] = {
    Severity.ERROR: 5.0,
    Severity.WARNING: 1.0,
    Severity.INFO: 0.0,
}

SCORE_THRESHOLDS = {
    "fail": 70.0,  # Por debajo de esto: FAIL
    "warning": 90.0,  # Por debajo de esto: WARNING
}


@dataclass
class ComplianceReport:
    """Reporte de cumplimiento de gobernanza."""

    score: float = 100.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    semantic_alerts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = ComplianceStatus.PASS.value
    _error_count: int = field(default=0, repr=False)
    _warning_count: int = field(default=0, repr=False)

    def add_violation(
        self,
        type_: str,
        message: str,
        severity: str = "ERROR",
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Registra una violación de reglas con validación robusta.

        Args:
            type_: Tipo/categoría de la violación.
            message: Descripción detallada de la violación.
            severity: Nivel de severidad ("ERROR", "WARNING", "INFO").
            context: Datos adicionales de contexto para debugging.

        Returns:
            True si la violación se registró correctamente, False si hubo error.

        Raises:
            No lanza excepciones, registra errores internamente.
        """
        # Validar y normalizar severity
        try:
            normalized_severity = Severity(severity.upper().strip())
        except (ValueError, AttributeError):
            logger.warning(f"Severidad inválida '{severity}', usando ERROR por defecto")
            normalized_severity = Severity.ERROR

        # Validar parámetros obligatorios
        if not type_ or not isinstance(type_, str):
            logger.error("add_violation: 'type_' debe ser una cadena no vacía")
            return False

        if not message or not isinstance(message, str):
            logger.error("add_violation: 'message' debe ser una cadena no vacía")
            return False

        # Construir registro de violación
        violation_record = {
            "type": type_.strip(),
            "message": message.strip(),
            "severity": normalized_severity.value,
        }

        # Añadir contexto si se proporciona
        if context and isinstance(context, dict):
            violation_record["context"] = context

        self.violations.append(violation_record)

        # Aplicar penalización al score
        penalty = SEVERITY_PENALTIES.get(normalized_severity, 0.0)
        self.score = max(0.0, self.score - penalty)

        # Actualizar contadores
        if normalized_severity == Severity.ERROR:
            self._error_count += 1
        elif normalized_severity == Severity.WARNING:
            self._warning_count += 1

        # Actualizar status basado en score y tipo de violaciones
        self._update_status()

        return True

    def _update_status(self) -> None:
        """Actualiza el status del reporte según score y violaciones."""
        if self._error_count > 0 or self.score < SCORE_THRESHOLDS["fail"]:
            self.status = ComplianceStatus.FAIL.value
        elif self._warning_count > 0 or self.score < SCORE_THRESHOLDS["warning"]:
            self.status = ComplianceStatus.WARNING.value
        else:
            self.status = ComplianceStatus.PASS.value

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen del reporte para logging/reporting."""
        return {
            "status": self.status,
            "score": round(self.score, 2),
            "total_violations": len(self.violations),
            "errors": self._error_count,
            "warnings": self._warning_count,
        }


class GovernanceEngine:
    """
    Motor de reglas para validar la integridad semántica y estructural.
    """

    # Importar yaml a nivel de clase para evitar imports dentro de métodos
    _yaml_available: bool = False

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.ontology: Dict[str, Any] = {}
        self.semantic_policy: Dict[str, Any] = {}
        self._config_loaded: bool = False
        self._ontology_loaded: bool = False
        self._check_yaml_availability()
        self._load_config()

    @classmethod
    def _check_yaml_availability(cls) -> None:
        """Verifica disponibilidad de PyYAML una sola vez."""
        if not hasattr(cls, "_yaml_checked"):
            try:
                import yaml  # noqa: F401

                cls._yaml_available = True
            except ImportError:
                cls._yaml_available = False
            cls._yaml_checked = True

    def _load_config(self) -> bool:
        """
        Carga la configuración y la ontología con manejo robusto de errores.

        Returns:
            True si al menos la configuración mínima se cargó correctamente.
        """
        ontology_success = self._load_ontology_file()
        policy_success = self._load_semantic_policy()

        self._config_loaded = ontology_success or policy_success

        if not self._config_loaded:
            logger.warning(
                "⚠️ No se cargó ninguna configuración. "
                "El motor operará con configuración por defecto."
            )
            self._apply_default_policy()

        return self._config_loaded

    def _load_ontology_file(self) -> bool:
        """
        Carga el archivo de ontología con validación de estructura.

        Returns:
            True si la ontología se cargó y validó correctamente.
        """
        ontology_path = self.config_dir / "ontology.json"

        if not ontology_path.exists():
            logger.warning(f"⚠️ No se encontró ontología en {ontology_path}")
            return False

        try:
            with open(ontology_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.error(f"❌ Archivo de ontología vacío: {ontology_path}")
                return False

            loaded_ontology = json.loads(content)

            if not self._validate_ontology_structure(loaded_ontology):
                logger.error("❌ Estructura de ontología inválida")
                return False

            self.ontology = loaded_ontology
            self._ontology_loaded = True
            logger.info(f"✅ Ontología cargada desde {ontology_path}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"❌ Error de sintaxis JSON en ontología: {e}")
            return False
        except PermissionError:
            logger.error(f"❌ Sin permisos para leer: {ontology_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado cargando ontología: {type(e).__name__}: {e}")
            return False

    def _validate_ontology_structure(self, ontology: Any) -> bool:
        """
        Valida que la ontología tenga la estructura esperada.

        Args:
            ontology: Objeto cargado del JSON.

        Returns:
            True si la estructura es válida.
        """
        if not isinstance(ontology, dict):
            logger.error("La ontología debe ser un objeto JSON (dict)")
            return False

        domains = ontology.get("domains")
        if domains is not None and not isinstance(domains, dict):
            logger.error("'domains' debe ser un diccionario")
            return False

        # Validar estructura de cada dominio si existe
        if domains:
            for domain_name, rules in domains.items():
                if not isinstance(rules, dict):
                    logger.error(f"Dominio '{domain_name}' debe ser un diccionario")
                    return False

                # Validar que keywords sean listas si existen
                for key in (
                    "forbidden_keywords",
                    "required_keywords",
                    "identifying_keywords",
                ):
                    if key in rules and not isinstance(rules[key], list):
                        logger.error(f"'{key}' en dominio '{domain_name}' debe ser lista")
                        return False

        return True

    def _load_semantic_policy(self) -> bool:
        """
        Carga las políticas semánticas desde data_contract.yaml.

        Returns:
            True si las políticas se cargaron correctamente.
        """
        contract_path = self.config_dir / "data_contract.yaml"

        if not contract_path.exists():
            logger.warning(f"⚠️ No se encontró data_contract en {contract_path}")
            self._apply_default_policy()
            return False

        if not self._yaml_available:
            logger.warning("⚠️ PyYAML no instalado. Instale con: pip install pyyaml")
            self._apply_default_policy()
            return False

        try:
            import yaml

            with open(contract_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"⚠️ Archivo data_contract vacío: {contract_path}")
                self._apply_default_policy()
                return False

            contract = yaml.safe_load(content)

            if contract is None:
                logger.warning("⚠️ data_contract parseado como None")
                self._apply_default_policy()
                return False

            if not isinstance(contract, dict):
                logger.error("❌ data_contract debe ser un diccionario YAML")
                return False

            self.semantic_policy = contract.get("semantic_policy", {})

            if not isinstance(self.semantic_policy, dict):
                logger.error("❌ 'semantic_policy' debe ser un diccionario")
                self.semantic_policy = {}
                return False

            logger.info(f"✅ Políticas semánticas cargadas desde {contract_path}")
            return True

        except yaml.YAMLError as e:
            logger.error(f"❌ Error de sintaxis YAML en data_contract: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error cargando data_contract: {type(e).__name__}: {e}")
            return False

    def _apply_default_policy(self) -> None:
        """Aplica política por defecto cuando no hay configuración."""
        self.semantic_policy = {
            "enable_ontology_check": True,
            "strict_mode": False,
        }
        logger.info("ℹ️ Aplicando política semántica por defecto")

    def load_ontology(self, path: str) -> bool:
        """
        Carga una ontología personalizada desde una ruta específica.

        Args:
            path: Ruta al archivo JSON de ontología.

        Returns:
            True si la ontología se cargó y validó correctamente.
        """
        if not path:
            logger.error("❌ Ruta de ontología no puede estar vacía")
            return False

        ontology_path = Path(path)

        if not ontology_path.exists():
            logger.error(f"❌ Archivo de ontología no encontrado: {path}")
            return False

        if not ontology_path.is_file():
            logger.error(f"❌ La ruta no es un archivo: {path}")
            return False

        # Verificar extensión
        if ontology_path.suffix.lower() != ".json":
            logger.warning(
                f"⚠️ Extensión inesperada (se esperaba .json): {ontology_path.suffix}"
            )

        try:
            with open(ontology_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.error(f"❌ Archivo de ontología vacío: {path}")
                return False

            loaded_ontology = json.loads(content)

            if not self._validate_ontology_structure(loaded_ontology):
                logger.error("❌ Estructura de ontología inválida")
                return False

            # Solo actualizar si la validación pasó
            self.ontology = loaded_ontology
            self._ontology_loaded = True
            logger.info(f"✅ Ontología recargada desde {path}")
            return True

        except json.JSONDecodeError as e:
            logger.error(
                f"❌ Error de sintaxis JSON: línea {e.lineno}, columna {e.colno}: {e.msg}"
            )
            return False
        except PermissionError:
            logger.error(f"❌ Sin permisos para leer: {path}")
            return False
        except Exception as e:
            logger.error(
                f"❌ Error cargando ontología personalizada: {type(e).__name__}: {e}"
            )
            return False

    def check_semantic_coherence(self, dataframe: pd.DataFrame) -> ComplianceReport:
        """
        Verifica la coherencia semántica de los APUs y sus insumos.

        Lógica mejorada:
        1. Valida precondiciones (política, datos, columnas).
        2. Agrupa los insumos por APU.
        3. Infiere el dominio del APU usando keywords identificadoras.
        4. Verifica insumos prohibidos y requeridos según el dominio.

        Args:
            dataframe: DataFrame conteniendo APUs e insumos (merged).

        Returns:
            ComplianceReport con las violaciones detectadas.
        """
        report = ComplianceReport()

        # === Validación de precondiciones ===
        if not self._validate_semantic_check_preconditions(dataframe, report):
            return report

        logger.info("🧠 Iniciando Validación Semántica de APUs...")

        domains = self.ontology.get("domains", {})
        if not domains:
            logger.warning("⚠️ Ontología vacía o sin dominios definidos.")
            return report

        # Normalizar keywords de dominios una sola vez
        normalized_domains = self._normalize_domain_keywords(domains)

        # Procesar cada APU
        try:
            grouped = dataframe.groupby(ColumnNames.CODIGO_APU)
        except Exception as e:
            logger.error(f"❌ Error agrupando por APU: {e}")
            report.add_violation("PROCESSING_ERROR", f"Error agrupando datos: {e}")
            return report

        processed_count = 0
        for apu_code, group in grouped:
            if group.empty:
                continue

            self._validate_apu_group(
                apu_code=apu_code,
                group=group,
                normalized_domains=normalized_domains,
                report=report,
            )
            processed_count += 1

        # Log resumen
        summary = report.get_summary()
        logger.info(
            f"✅ Validación Semántica completada. "
            f"APUs procesados: {processed_count}. "
            f"Status: {summary['status']}. "
            f"Score: {summary['score']}. "
            f"Violaciones: {summary['total_violations']}"
        )

        return report

    def _validate_semantic_check_preconditions(
        self, dataframe: pd.DataFrame, report: ComplianceReport
    ) -> bool:
        """
        Valida las precondiciones para la verificación semántica.

        Args:
            dataframe: DataFrame a validar.
            report: Reporte donde registrar violaciones.

        Returns:
            True si todas las precondiciones se cumplen.
        """
        # Verificar política
        if not self.semantic_policy.get("enable_ontology_check", False):
            logger.info("ℹ️ Validación semántica desactivada por política.")
            return False

        # Verificar ontología cargada
        if not self._ontology_loaded or not self.ontology:
            logger.warning("⚠️ Ontología no cargada. Saltando validación semántica.")
            return False

        # Verificar DataFrame
        if dataframe is None:
            logger.warning("⚠️ DataFrame es None para validación semántica.")
            return False

        if not isinstance(dataframe, pd.DataFrame):
            logger.error(f"❌ Se esperaba DataFrame, se recibió {type(dataframe).__name__}")
            report.add_violation(
                "TYPE_ERROR", f"Tipo de datos inválido: {type(dataframe).__name__}", "ERROR"
            )
            return False

        if dataframe.empty:
            logger.warning("⚠️ DataFrame vacío para validación semántica.")
            return False

        # Verificar columnas requeridas
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
            return False

        return True

    def _normalize_domain_keywords(
        self, domains: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Normaliza las keywords de todos los dominios a mayúsculas.

        Args:
            domains: Diccionario de dominios de la ontología.

        Returns:
            Dominios con keywords normalizadas.
        """
        normalized = {}

        for domain_name, rules in domains.items():
            if not isinstance(rules, dict):
                continue

            normalized[domain_name.upper()] = {
                "forbidden_keywords": self._normalize_keyword_list(
                    rules.get("forbidden_keywords", [])
                ),
                "required_keywords": self._normalize_keyword_list(
                    rules.get("required_keywords", [])
                ),
                "identifying_keywords": self._normalize_keyword_list(
                    rules.get("identifying_keywords", [domain_name])
                ),
                "min_required_matches": rules.get("min_required_matches", 1),
            }

        return normalized

    def _normalize_keyword_list(self, keywords: Any) -> List[str]:
        """
        Normaliza una lista de keywords a mayúsculas.

        Args:
            keywords: Lista de keywords o valor inválido.

        Returns:
            Lista de keywords normalizadas.
        """
        if not isinstance(keywords, list):
            return []

        normalized = []
        for kw in keywords:
            if isinstance(kw, str) and kw.strip():
                normalized.append(kw.strip().upper())

        return normalized

    def _validate_apu_group(
        self,
        apu_code: Any,
        group: pd.DataFrame,
        normalized_domains: Dict[str, Dict[str, Any]],
        report: ComplianceReport,
    ) -> None:
        """
        Valida un grupo de insumos pertenecientes a un APU.

        Args:
            apu_code: Código del APU.
            group: DataFrame con los insumos del APU.
            normalized_domains: Dominios normalizados de la ontología.
            report: Reporte donde registrar violaciones.
        """
        # Obtener descripción del APU de forma segura
        try:
            desc_apu_raw = group[ColumnNames.DESCRIPCION_APU].iloc[0]
            desc_apu = self._safe_string_upper(desc_apu_raw)
        except (IndexError, KeyError):
            logger.debug(f"No se pudo obtener descripción para APU: {apu_code}")
            return

        # Inferir dominio usando keywords identificadoras
        detected_domain, domain_rules = self._infer_domain(desc_apu, normalized_domains)

        if not detected_domain:
            logger.debug(f"No se pudo inferir dominio para APU: {apu_code}")
            return

        # Obtener descripciones de insumos normalizadas
        insumos_series = group[ColumnNames.DESCRIPCION_INSUMO]
        insumos_normalized = insumos_series.apply(self._safe_string_upper)

        # Chequeo 1: Palabras Prohibidas
        self._check_forbidden_keywords(
            apu_code=apu_code,
            domain_name=detected_domain,
            insumos=insumos_normalized,
            forbidden=domain_rules.get("forbidden_keywords", []),
            report=report,
        )

        # Chequeo 2: Palabras Requeridas
        self._check_required_keywords(
            apu_code=apu_code,
            domain_name=detected_domain,
            insumos=insumos_normalized,
            required=domain_rules.get("required_keywords", []),
            min_matches=domain_rules.get("min_required_matches", 1),
            report=report,
        )

    def _safe_string_upper(self, value: Any) -> str:
        """
        Convierte un valor a string en mayúsculas de forma segura.

        Args:
            value: Cualquier valor.

        Returns:
            String en mayúsculas, cadena vacía si es None/NaN.
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        return str(value).strip().upper()

    def _infer_domain(
        self, description: str, normalized_domains: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Infiere el dominio de un APU basado en su descripción.

        Utiliza las identifying_keywords de cada dominio para una
        detección más robusta que solo buscar el nombre del dominio.

        Args:
            description: Descripción del APU (normalizada a mayúsculas).
            normalized_domains: Dominios normalizados.

        Returns:
            Tupla (nombre_dominio, reglas) o (None, {}) si no se detecta.
        """
        if not description:
            return None, {}

        best_match: Optional[str] = None
        best_match_count = 0
        best_rules: Dict[str, Any] = {}

        for domain_name, rules in normalized_domains.items():
            identifying = rules.get("identifying_keywords", [])

            if not identifying:
                # Fallback: usar nombre del dominio como keyword
                if domain_name in description:
                    if best_match is None:
                        best_match = domain_name
                        best_rules = rules
                continue

            # Contar cuántas keywords identificadoras coinciden
            match_count = sum(1 for kw in identifying if kw and kw in description)

            if match_count > best_match_count:
                best_match_count = match_count
                best_match = domain_name
                best_rules = rules

        return best_match, best_rules

    def _check_forbidden_keywords(
        self,
        apu_code: Any,
        domain_name: str,
        insumos: pd.Series,
        forbidden: List[str],
        report: ComplianceReport,
    ) -> None:
        """
        Verifica que los insumos no contengan palabras prohibidas.

        Args:
            apu_code: Código del APU.
            domain_name: Nombre del dominio detectado.
            insumos: Series con descripciones de insumos normalizadas.
            forbidden: Lista de keywords prohibidas.
            report: Reporte donde registrar violaciones.
        """
        if not forbidden:
            return

        for bad_keyword in forbidden:
            if not bad_keyword:
                continue

            try:
                # Usar regex=False para búsqueda literal (más seguro y rápido)
                mask = insumos.str.contains(bad_keyword, regex=False, na=False)

                if mask.any():
                    violating_insumos = insumos[mask].unique().tolist()
                    total_violations = len(violating_insumos)

                    # Mostrar muestra pero indicar el total
                    sample_size = 3
                    sample = violating_insumos[:sample_size]
                    suffix = (
                        f" (y {total_violations - sample_size} más)"
                        if total_violations > sample_size
                        else ""
                    )

                    msg = (
                        f"APU '{apu_code}' ({domain_name}): "
                        f"Insumos contienen término prohibido '{bad_keyword}': "
                        f"{sample}{suffix}"
                    )

                    report.add_violation(
                        type_="SEMANTIC_INCONSISTENCY",
                        message=msg,
                        severity="WARNING",
                        context={
                            "apu_code": str(apu_code),
                            "domain": domain_name,
                            "forbidden_keyword": bad_keyword,
                            "violation_count": total_violations,
                        },
                    )
            except Exception as e:
                logger.debug(f"Error verificando keyword '{bad_keyword}': {e}")

    def _check_required_keywords(
        self,
        apu_code: Any,
        domain_name: str,
        insumos: pd.Series,
        required: List[str],
        min_matches: int,
        report: ComplianceReport,
    ) -> None:
        """
        Verifica que los insumos contengan las palabras requeridas.

        Args:
            apu_code: Código del APU.
            domain_name: Nombre del dominio detectado.
            insumos: Series con descripciones de insumos normalizadas.
            required: Lista de keywords requeridas.
            min_matches: Mínimo de keywords requeridas que deben estar presentes.
            report: Reporte donde registrar violaciones.
        """
        if not required:
            return

        # Concatenar todos los insumos para búsqueda
        try:
            all_insumos_text = " ".join(insumos.dropna().astype(str))
        except Exception:
            all_insumos_text = ""

        if not all_insumos_text.strip():
            return

        # Contar keywords presentes
        found_keywords = [kw for kw in required if kw and kw in all_insumos_text]
        found_count = len(found_keywords)
        missing_keywords = [kw for kw in required if kw and kw not in all_insumos_text]

        if found_count < min_matches:
            msg = (
                f"APU '{apu_code}' ({domain_name}): "
                f"No contiene suficientes insumos esperados. "
                f"Encontrados: {found_count}/{min_matches} mínimo. "
                f"Esperados: {missing_keywords[:5]}"
            )

            report.add_violation(
                type_="SEMANTIC_INCOMPLETENESS",
                message=msg,
                severity="WARNING",
                context={
                    "apu_code": str(apu_code),
                    "domain": domain_name,
                    "required_keywords": required,
                    "found_keywords": found_keywords,
                    "missing_keywords": missing_keywords,
                    "min_required": min_matches,
                },
            )

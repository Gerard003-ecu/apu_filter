"""
Procesador de APU (Análisis de Precios Unitarios) con arquitectura híbrida Lark+Python.
Versión refinada con mejoras en robustez, performance y mantenibilidad.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedEOF

from .schemas import (
    Equipo,
    InsumoProcesado,
    ManoDeObra,
    Otro,
    Suministro,
    Transporte,
)
from .utils import clean_apu_code, normalize_text, parse_number

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

class TipoInsumo(Enum):
    """Enumeración de tipos de insumo válidos."""
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    TRANSPORTE = "TRANSPORTE"
    SUMINISTRO = "SUMINISTRO"
    OTRO = "OTRO"


class FormatoLinea(Enum):
    """Enumeración de formatos de línea detectados."""
    MO_COMPLETA = "MO_COMPLETA"
    INSUMO_BASICO = "INSUMO_BASICO"
    DESCONOCIDO = "DESCONOCIDO"


APU_GRAMMAR = r"""
    ?start: line
    line: (field (SEP field)*)? NEWLINE?
    field: FIELD_VALUE
    FIELD_VALUE: /[^;\r\n]+/
    SEP: ";"
    NEWLINE: /[\r\n]+/
    %import common.WS
    %ignore WS
"""

# ============================================================================
# CACHÉ DE KEYWORDS Y CLASIFICACIÓN
# ============================================================================

@dataclass
class KeywordCache:
    """Cache para keywords de clasificación con lazy loading."""
    _equipo_keywords: List[str] = field(default_factory=list)
    _mo_keywords: List[str] = field(default_factory=list)
    _transporte_keywords: List[str] = field(default_factory=list)
    _suministro_keywords: List[str] = field(default_factory=list)
    _initialized: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el cache de keywords a partir de la configuración.

        Args:
            config: Diccionario de configuración que debe contener 'keyword_maps'.
        """
        self.config = config
        self._initialized = False

    def initialize(self):
        """Inicializa keywords una sola vez desde la configuración."""
        if not self._initialized and self.config:
            keyword_maps = self.config.get("keyword_maps", {})
            self._equipo_keywords = keyword_maps.get("equipo", [])
            self._mo_keywords = keyword_maps.get("mano_de_obra", [])
            self._transporte_keywords = keyword_maps.get("transporte", [])
            self._suministro_keywords = keyword_maps.get("suministro", [])

            if not any([self._equipo_keywords, self._mo_keywords, self._transporte_keywords, self._suministro_keywords]):
                logger.warning("Keyword maps en config.json está vacío o no se encontró.")

            self._initialized = True

    @property
    def equipo_keywords(self) -> List[str]:
        self.initialize()
        return self._equipo_keywords

    @property
    def mo_keywords(self) -> List[str]:
        self.initialize()
        return self._mo_keywords

    @property
    def transporte_keywords(self) -> List[str]:
        self.initialize()
        return self._transporte_keywords

    @property
    def suministro_keywords(self) -> List[str]:
        self.initialize()
        return self._suministro_keywords

# ============================================================================
# TRANSFORMER MEJORADO
# ============================================================================

@v_args(inline=True)
class APUTransformer(Transformer):
    """
    Transforma el árbol de parsing de Lark en objetos dataclass.
    Versión mejorada con validación robusta y cache.
    """

    def __init__(self, apu_context: Dict[str, Any], config: Dict[str, Any], keyword_cache: KeywordCache):
        self.apu_context = apu_context
        self.config = config
        self.keyword_cache = keyword_cache
        self.validation_cache = {}
        super().__init__()

    def _clean_token(self, token) -> str:
        """Limpia un token de forma segura."""
        if token is None:
            return ""
        value = getattr(token, 'value', str(token))
        return value.strip() if value else ""

    def line(self, *fields) -> Optional[InsumoProcesado]:
        """
        Punto de entrada principal del transformer. Recibe los campos ya separados por Lark.
        """
        try:
            # Los campos ya vienen como una tupla de tokens de Lark
            tokens = [self._clean_token(f) for f in fields]

            # Validación básica: debe haber al menos una descripción
            if not tokens or not tokens[0]:
                logger.debug("Línea vacía o sin descripción tras el parseo.")
                return None

            # Detección de formato y despacho al builder correspondiente
            formato_detectado = self._detect_format(tokens)
            return self._dispatch_builder(formato_detectado, tokens)

        except Exception as e:
            logger.error(f"Error crítico procesando línea en APUTransformer: {tokens if 'tokens' in locals() else 'desconocida'}", exc_info=True)
            return None

    @lru_cache(maxsize=1024)
    def _detect_format_cached(self, tokens: Tuple[str, ...]) -> FormatoLinea:
        """Detecta formato con cache para mejorar performance."""
        return self._detect_format(list(tokens))

    def _detect_format(self, fields: List[str]) -> FormatoLinea:
        """
        Detecta el formato de la línea con validación mejorada.
        """
        num_fields = len(fields)

        if num_fields < 5:
            return FormatoLinea.DESCONOCIDO

        descripcion = fields[0]
        tipo_probable = self._classify_insumo_with_cache(descripcion)

        # Validación específica para MO_COMPLETA
        if num_fields >= 6 and tipo_probable == TipoInsumo.MANO_DE_OBRA:
            if self._validate_mo_format(fields):
                return FormatoLinea.MO_COMPLETA

        # Default a insumo básico si tiene campos suficientes
        if num_fields >= 5:
            return FormatoLinea.INSUMO_BASICO

        return FormatoLinea.DESCONOCIDO

    def _validate_mo_format(self, fields: List[str]) -> bool:
        """Valida si los campos corresponden a formato MO_COMPLETA."""
        try:
            if len(fields) < 6:
                return False

            jornal_total = parse_number(fields[3])
            rendimiento = parse_number(fields[4])

            thresholds = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})
            min_jornal = thresholds.get("min_jornal", 50000)
            max_jornal = thresholds.get("max_jornal", 10000000)
            min_rendimiento = thresholds.get("min_rendimiento", 0.001)
            max_rendimiento = thresholds.get("max_rendimiento", 1000)

            return (min_jornal <= jornal_total <= max_jornal and
                    min_rendimiento <= rendimiento <= max_rendimiento)

        except (ValueError, IndexError):
            return False

    def _dispatch_builder(self, formato: FormatoLinea, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Despacha al constructor apropiado con fallback inteligente."""
        builders = {
            FormatoLinea.MO_COMPLETA: self._build_mo_completa,
            FormatoLinea.INSUMO_BASICO: self._build_insumo_basico,
        }

        builder = builders.get(formato)

        if not builder:
            logger.warning(f"Formato no soportado: {formato}")
            return None

        # Intenta construir con el builder principal
        result = builder(tokens)

        # Si falla y es MO_COMPLETA, intenta fallback a insumo básico
        if not result and formato == FormatoLinea.MO_COMPLETA:
            logger.debug("Aplicando fallback MO_COMPLETA -> INSUMO_BASICO")
            result = self._build_insumo_basico(tokens)

        return result

    def _build_mo_completa(self, tokens: List[str]) -> Optional[ManoDeObra]:
        """
        Construye un objeto ManoDeObra con validación robusta y manejo de separadores.
        """
        try:
            if len(tokens) < 6:
                return None

            descripcion = tokens[0]
            jornal_str = tokens[3]
            rendimiento_str = tokens[4]

            # Detección de separador decimal
            decimal_separator = "comma" if ',' in jornal_str or ',' in rendimiento_str else "dot"

            jornal_total = parse_number(jornal_str, decimal_separator=decimal_separator)
            rendimiento = parse_number(rendimiento_str, decimal_separator=decimal_separator)

            # Validación estricta
            if not self._validate_mo_values(jornal_total, rendimiento):
                return None

            # Cálculos
            cantidad = 1.0 / rendimiento if rendimiento > 0 else 0
            valor_total = cantidad * jornal_total

            # Construcción del objeto
            return ManoDeObra(
                descripcion_insumo=descripcion,
                unidad_insumo="JOR",
                cantidad=cantidad,
                precio_unitario=jornal_total,
                valor_total=valor_total,
                rendimiento=rendimiento,
                formato_origen="MO_COMPLETA",
                tipo_insumo="MANO_DE_OBRA",
                normalized_desc=normalize_text(descripcion),
                **self.apu_context
            )

        except (ValueError, ZeroDivisionError, IndexError) as e:
            logger.debug(f"Error construyendo MO_COMPLETA: {e}")
            return None

    def _build_insumo_basico(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """
        Construye un objeto de insumo genérico con manejo flexible de campos.
        """
        try:
            # Manejo flexible de número de campos
            parsed_values = self._parse_insumo_fields(tokens)
            if not parsed_values:
                return None

            descripcion, unidad, cantidad, precio_unit, valor_total = parsed_values

            # Corrección automática de valores
            valor_total = self._correct_total_value(cantidad, precio_unit, valor_total)

            # Clasificación del tipo
            tipo_insumo = self._classify_insumo_with_cache(descripcion)

            # Construcción del objeto apropiado
            return self._build_typed_insumo(
                descripcion, unidad, cantidad, precio_unit,
                valor_total, tipo_insumo
            )

        except Exception as e:
            logger.error(f"Error construyendo INSUMO_BASICO: {e}")
            return None

    def _parse_insumo_fields(self, tokens: List[str]) -> Optional[Tuple[str, str, float, float, float]]:
        """
        Parsea los campos de una línea de insumo, manejando formatos de 5 o 6 campos
        y detectando automáticamente el separador decimal.
        """
        try:
            num_tokens = len(tokens)
            if num_tokens < 5:
                return None

            descripcion, unidad, cantidad_str = tokens[0], tokens[1], tokens[2]

            precio_str, valor_str = (tokens[4], tokens[5]) if num_tokens >= 6 else (tokens[3], tokens[4])

            # Conversión numérica robusta con el separador de coma
            cantidad = parse_number(cantidad_str, decimal_separator="comma")
            precio_unit = parse_number(precio_str, decimal_separator="comma")
            valor_total = parse_number(valor_str, decimal_separator="comma")

            return descripcion, unidad, cantidad, precio_unit, valor_total

        except (ValueError, IndexError) as e:
            logger.warning(f"Error al parsear campos de insumo: {tokens}. Error: {e}")
            return None

    def _correct_total_value(self, cantidad: float, precio_unit: float, valor_total: float) -> float:
        """Corrige el valor total si es necesario."""
        if valor_total == 0 and cantidad > 0 and precio_unit > 0:
            calculated = cantidad * precio_unit
            logger.debug(f"Valor total corregido: 0 -> {calculated:.2f}")
            return calculated

        # Validar coherencia con tolerancia
        if cantidad > 0 and precio_unit > 0:
            expected = cantidad * precio_unit
            tolerance = 0.01  # 1% de tolerancia
            if abs(valor_total - expected) / expected > tolerance:
                logger.warning(f"Valor total inconsistente: {valor_total:.2f} vs esperado {expected:.2f}")

        return valor_total

    def _build_typed_insumo(self, descripcion: str, unidad: str, cantidad: float,
                           precio_unit: float, valor_total: float,
                           tipo_insumo: TipoInsumo) -> InsumoProcesado:
        """Construye el objeto de insumo del tipo apropiado."""
        common_args = {
            "descripcion_insumo": descripcion,
            "unidad_insumo": unidad or "UND",
            "cantidad": cantidad,
            "precio_unitario": precio_unit,
            "valor_total": valor_total,
            "rendimiento": cantidad,
            "formato_origen": "INSUMO_BASICO",
            "tipo_insumo": tipo_insumo.value,
            "normalized_desc": normalize_text(descripcion),
            **self.apu_context
        }

        class_map = {
            TipoInsumo.EQUIPO: Equipo,
            TipoInsumo.TRANSPORTE: Transporte,
            TipoInsumo.SUMINISTRO: Suministro,
            TipoInsumo.MANO_DE_OBRA: ManoDeObra,
        }

        InsumoClass = class_map.get(tipo_insumo, Otro)
        return InsumoClass(**common_args)

    def _validate_mo_values(self, jornal: float, rendimiento: float) -> bool:
        """Valida valores de mano de obra contra umbrales."""
        thresholds = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})

        min_jornal = thresholds.get("min_jornal", 50000)
        max_jornal = thresholds.get("max_jornal", 10000000)
        min_rendimiento = thresholds.get("min_rendimiento", 0.001)
        max_rendimiento = thresholds.get("max_rendimiento", 1000)

        is_valid = (min_jornal <= jornal <= max_jornal and
                   min_rendimiento <= rendimiento <= max_rendimiento)

        if not is_valid:
            logger.warning(f"Valores MO fuera de rango: jornal={jornal:,.0f}, rendimiento={rendimiento:.3f}")

        return is_valid

    @lru_cache(maxsize=2048)
    def _classify_insumo_with_cache(self, descripcion: str) -> TipoInsumo:
        """Clasifica insumo con cache para mejorar performance."""
        return self._classify_insumo(descripcion)

    def _classify_insumo(self, descripcion: str) -> TipoInsumo:
        """
        Clasifica un insumo basado en palabras clave.
        """
        if not descripcion:
            return TipoInsumo.OTRO

        desc_upper = descripcion.upper()

        # Casos especiales primero
        special_cases = {
            "HERRAMIENTA MENOR": TipoInsumo.EQUIPO,
            "HERRAMIENTA (% MO)": TipoInsumo.EQUIPO,
            "EQUIPO Y HERRAMIENTA": TipoInsumo.EQUIPO,
        }

        for case, tipo in special_cases.items():
            if case in desc_upper:
                return tipo

        # Clasificación por keywords con orden de precedencia
        keyword_hierarchy = [
            (TipoInsumo.EQUIPO, self.keyword_cache.equipo_keywords),
            (TipoInsumo.MANO_DE_OBRA, self.keyword_cache.mo_keywords),
            (TipoInsumo.TRANSPORTE, self.keyword_cache.transporte_keywords),
            (TipoInsumo.SUMINISTRO, self.keyword_cache.suministro_keywords),
        ]

        for tipo, keywords in keyword_hierarchy:
            if any(kw in desc_upper for kw in keywords):
                return tipo

        return TipoInsumo.OTRO


# ============================================================================
# PROCESADOR PRINCIPAL MEJORADO
# ============================================================================

class APUProcessor:
    """
    Procesador de APU con arquitectura híbrida Lark+Python.
    Versión mejorada con validación robusta y optimizaciones.
    """

    EXCLUDED_TERMS = frozenset([
        'IMPUESTOS', 'POLIZAS', 'SEGUROS', 'GASTOS GENERALES',
        'UTILIDAD', 'ADMINISTRACION', 'RETENCIONES', 'AIU', 'A.I.U',
        'PROVISIONALES', 'IMPREVISTOS', 'HONORARIOS'
    ])

    def __init__(self, raw_records: List[Dict[str, str]], config: Dict[str, Any]):
        """
        Inicializa el procesador con validación de entrada.
        
        Args:
            raw_records: Lista de registros crudos
            config: Configuración del procesador
            
        Raises:
            ValueError: Si los registros o configuración son inválidos
        """
        # Validación de entrada
        self._validate_inputs(raw_records, config)

        self.raw_records = raw_records
        self.config = config
        self.processed_data: List[InsumoProcesado] = []
        self.stats = defaultdict(int)
        self._parser = None

        # Inicializar cache de keywords
        self.keyword_cache = KeywordCache(self.config)

        # Inicializar parser
        self._initialize_parser()

    def _validate_inputs(self, raw_records: List[Dict[str, str]], config: Dict[str, Any]):
        """Valida las entradas del procesador."""
        if not isinstance(raw_records, list):
            raise ValueError("raw_records debe ser una lista")

        if not isinstance(config, dict):
            raise ValueError("config debe ser un diccionario")

        if not raw_records:
            logger.warning("Lista de registros vacía")

        # Validar estructura mínima de config
        required_config_keys = ["validation_thresholds"]
        missing_keys = [k for k in required_config_keys if k not in config]
        if missing_keys:
            logger.warning(f"Claves faltantes en config: {missing_keys}")

    def _initialize_parser(self):
        """Inicializa el parser Lark con manejo de errores."""
        try:
            self._parser = Lark(
                APU_GRAMMAR,
                start='line',
                parser='earley',
                ambiguity='resolve',
                debug=False
            )
            logger.info("✅ Parser Lark inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando parser: {e}")
            raise RuntimeError(f"No se pudo inicializar el parser Lark: {e}")

    def process_all(self) -> pd.DataFrame:
        """
        Procesa todos los registros con optimización de memoria.
        
        Returns:
            DataFrame con datos procesados
        """
        logger.info(f"🚀 Iniciando procesamiento de {len(self.raw_records)} registros...")

        # Procesar en lotes para optimizar memoria
        batch_size = self.config.get("batch_size", 1000)

        for batch_start in range(0, len(self.raw_records), batch_size):
            batch_end = min(batch_start + batch_size, len(self.raw_records))
            batch = self.raw_records[batch_start:batch_end]

            self._process_batch(batch, batch_start)

            # Log de progreso
            if batch_end % (batch_size * 10) == 0:
                logger.info(f"Procesados {batch_end}/{len(self.raw_records)} registros...")

        # Post-procesamiento
        self._apply_post_processing()

        # Generar estadísticas y DataFrame
        self._log_stats()
        return self._build_optimized_dataframe()

    def _process_batch(self, batch: List[Dict[str, str]], offset: int):
        """Procesa un lote de registros."""
        for idx, record in enumerate(batch):
            try:
                processed = self._process_single_record(record)
                if processed:
                    self.processed_data.append(processed)
                    self._update_stats(processed)
                else:
                    self.stats["registros_descartados"] += 1

            except Exception as e:
                record_num = offset + idx + 1
                logger.warning(f"⚠️ Error en registro {record_num}: {e}")
                self.stats["errores"] += 1

    def _process_single_record(self, record: Dict[str, str]) -> Optional[InsumoProcesado]:
        """
        Procesa un registro individual con validación completa.
        """
        # Validar y limpiar campos básicos
        clean_record = self._clean_record_fields(record)
        if not clean_record:
            return None

        # Parsear línea de insumo
        insumo_obj = self._parse_insumo_line(clean_record)
        if not insumo_obj:
            return None

        # Post-procesamiento del objeto
        insumo_obj = self._post_process_insumo(insumo_obj, clean_record)

        # Validación final
        if not self._validate_final_insumo(insumo_obj):
            self.stats["rechazados_validacion"] += 1
            return None

        return insumo_obj

    def _clean_record_fields(self, record: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Limpia y valida campos del registro."""
        apu_code = clean_apu_code(record.get("apu_code", ""))
        apu_desc = record.get("apu_desc", "").strip()

        if not apu_code or not apu_desc:
            logger.debug("Registro sin código o descripción APU válidos")
            return None

        return {
            "apu_code": apu_code,
            "apu_desc": apu_desc,
            "apu_unit": self._normalize_unit(record.get("apu_unit", "UND")),
            "category": record.get("category", "OTRO"),
            "insumo_line": record.get("insumo_line", "")
        }

    def _parse_insumo_line(self, clean_record: Dict[str, str]) -> Optional[InsumoProcesado]:
        """Parsea la línea de insumo usando Lark."""
        insumo_line = clean_record["insumo_line"]

        if not insumo_line or not insumo_line.strip():
            return None

        try:
            # Crear contexto APU para el transformer
            apu_context = {
                "codigo_apu": clean_record["apu_code"],
                "descripcion_apu": clean_record["apu_desc"],
                "unidad_apu": clean_record["apu_unit"],
                "categoria": clean_record["category"],
            }

            # Parsear con transformer
            transformer = APUTransformer(apu_context, self.config, self.keyword_cache)
            tree = self._parser.parse(insumo_line)
            insumo_obj = transformer.transform(tree)

            return insumo_obj

        except (LarkError, UnexpectedCharacters, UnexpectedEOF) as e:
            logger.debug(f"Error de parsing Lark: {e}")
            self.stats["errores_parsing"] += 1
            return None
        except Exception as e:
            logger.debug(f"Error inesperado en parsing: {e}")
            self.stats["errores_parsing"] += 1
            return None

    def _post_process_insumo(self, insumo_obj: InsumoProcesado,
                            clean_record: Dict[str, str]) -> InsumoProcesado:
        """Aplica post-procesamiento al objeto de insumo."""
        # Inferir unidad si es necesario
        if insumo_obj.unidad_apu == "UND":
            insumo_obj.unidad_apu = self._infer_unit_intelligent(
                insumo_obj.descripcion_apu,
                clean_record["category"],
                insumo_obj.tipo_insumo
            )

        return insumo_obj

    @lru_cache(maxsize=512)
    def _normalize_unit(self, unit: str) -> str:
        """Normaliza unidades con cache."""
        if not unit or not unit.strip():
            return "UND"

        unit_clean = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        unit_mapping = {
            "DIAS": "DIA", "DÍAS": "DIA", "JORNAL": "JOR",
            "JORNALES": "JOR", "HORA": "HR", "HORAS": "HR",
            "UN": "UND", "UNIDAD": "UND", "UNIDADES": "UND",
            "METRO": "M", "METROS": "M", "MTS": "M", "ML": "M",
            "M2": "M2", "MT2": "M2", "METROSCUADRADOS": "M2",
            "M3": "M3", "MT3": "M3", "METROSCUBICOS": "M3",
            "KILOGRAMO": "KG", "KILOGRAMOS": "KG", "KILO": "KG",
            "TONELADA": "TON", "TONELADAS": "TON",
            "GLN": "GAL", "GALON": "GAL", "GALONES": "GAL",
            "LITRO": "LT", "LITROS": "LT",
            "VIAJES": "VIAJE", "VJE": "VIAJE",
        }

        return unit_mapping.get(unit_clean, unit_clean)

    @lru_cache(maxsize=512)
    def _infer_unit_intelligent(self, description: str, category: str,
                               tipo_insumo: str) -> str:
        """Infiere unidad con lógica mejorada y cache."""
        desc_upper = description.upper()

        # Mapeo por tipo de insumo
        type_units = {
            "MANO_DE_OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
        }

        if tipo_insumo in type_units:
            return type_units[tipo_insumo]

        # Detección por keywords específicos
        unit_patterns = [
            (r"\bM3\b|\bMETROS?\s*CUB", "M3"),
            (r"\bM2\b|\bMETROS?\s*CUAD", "M2"),
            (r"\bML\b|\bMETROS?\s*LIN", "ML"),
            (r"\bKG\b|\bKILO", "KG"),
            (r"\bTON\b", "TON"),
            (r"\bGAL\b|\bGALON", "GAL"),
            (r"\bLT\b|\bLITRO", "LT"),
            (r"\bDIA\b|\bDIAS\b", "DIA"),
            (r"\bJOR\b|\bJORNAL", "JOR"),
        ]

        for pattern, unit in unit_patterns:
            if re.search(pattern, desc_upper):
                return unit

        # Default por categoría
        category_defaults = {
            "MANO DE OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
            "MATERIALES": "UND",
        }

        return category_defaults.get(category, "UND")

    def _validate_final_insumo(self, insumo: InsumoProcesado) -> bool:
        """Validación final con reglas mejoradas."""
        # Validación básica
        if not insumo or not insumo.descripcion_insumo:
            return False

        # Verificar términos excluidos
        desc_upper = insumo.descripcion_insumo.upper()
        if any(term in desc_upper for term in self.EXCLUDED_TERMS):
            self.stats["excluidos_por_termino"] += 1
            return False

        # Validación por tipo con umbrales
        return self._validate_by_type(insumo)

    def _validate_by_type(self, insumo: InsumoProcesado) -> bool:
        """Valida insumo según su tipo con umbrales configurables."""
        thresholds = self.config.get("validation_thresholds", {})
        tipo = insumo.tipo_insumo

        if tipo == "MANO_DE_OBRA":
            mo_config = thresholds.get("MANO_DE_OBRA", {})
            min_jornal = mo_config.get("min_jornal", 50000)
            max_jornal = mo_config.get("max_jornal", 10000000)
            max_valor = mo_config.get("max_valor_total", 100000000)

            if insumo.precio_unitario > 0:
                if not (min_jornal <= insumo.precio_unitario <= max_jornal):
                    logger.debug(f"MO fuera de rango: ${insumo.precio_unitario:,.0f}")
                    return False

            if insumo.valor_total > max_valor:
                logger.debug(f"Valor MO excesivo: ${insumo.valor_total:,.0f}")
                return False

        else:
            # Validación general para otros tipos
            tipo_config = thresholds.get(tipo, thresholds.get("DEFAULT", {}))
            max_valor = tipo_config.get("max_valor_total", float('inf'))

            if insumo.valor_total > max_valor:
                logger.debug(f"Valor excesivo para {tipo}: ${insumo.valor_total:,.0f}")
                return False

        # Validar que tenga al menos cantidad o valor
        if insumo.cantidad <= 0 and insumo.valor_total <= 0:
            logger.debug(f"Sin cantidad ni valor: {insumo.descripcion_insumo[:30]}")
            return False

        return True

    def _apply_post_processing(self):
        """Aplica correcciones y ajustes finales."""
        # Corrección de unidades para cuadrillas
        self._fix_squad_units()

        # Normalización de valores extremos
        self._normalize_extreme_values()

        # Detección y corrección de duplicados
        self._handle_duplicates()

    def _fix_squad_units(self):
        """Corrige unidades para cuadrillas conocidas."""
        squad_patterns = [
            (r'^1[3-9]', 'DIA'),  # Códigos 13-19
            (r'^2[0-9]', 'DIA'),   # Códigos 20-29
            (r'CUADRILLA', 'DIA'), # Descripción con "cuadrilla"
        ]

        corrections = 0
        for insumo in self.processed_data:
            if insumo.tipo_insumo != "MANO_DE_OBRA":
                continue

            for pattern, unit in squad_patterns:
                if (re.match(pattern, insumo.codigo_apu) or
                    pattern in insumo.descripcion_apu.upper()):
                    if insumo.unidad_apu != unit:
                        old_unit = insumo.unidad_apu
                        insumo.unidad_apu = unit
                        logger.debug(f"Corregida unidad {insumo.codigo_apu}: {old_unit} -> {unit}")
                        corrections += 1
                        break

        if corrections > 0:
            logger.info(f"✅ Aplicadas {corrections} correcciones de unidad")

    def _normalize_extreme_values(self):
        """Normaliza valores extremos usando percentiles."""
        if not self.processed_data:
            return

        # Agrupar por tipo para normalización
        by_type = defaultdict(list)
        for insumo in self.processed_data:
            by_type[insumo.tipo_insumo].append(insumo)

        for tipo, insumos in by_type.items():
            if len(insumos) < 10:  # No normalizar si hay pocos datos
                continue

            # Calcular percentiles para valor total
            valores = [i.valor_total for i in insumos if i.valor_total > 0]
            if not valores:
                continue

            p95 = np.percentile(valores, 95)
            p99 = np.percentile(valores, 99)

            # Aplicar cap en p99 con warning
            for insumo in insumos:
                if insumo.valor_total > p99:
                    logger.warning(
                        f"Valor extremo detectado en {tipo}: ${insumo.valor_total:,.0f} "
                        f"(p99=${p99:,.0f})"
                    )
                    # Opcional: aplicar cap
                    # insumo.valor_total = p99

    def _handle_duplicates(self):
        """Detecta y maneja registros duplicados."""
        seen = set()
        unique_data = []
        duplicates = 0

        for insumo in self.processed_data:
            # Crear clave única
            key = (
                insumo.codigo_apu,
                insumo.descripcion_insumo,
                insumo.tipo_insumo,
                round(insumo.cantidad, 6),
                round(insumo.precio_unitario, 2)
            )

            if key not in seen:
                seen.add(key)
                unique_data.append(insumo)
            else:
                duplicates += 1
                logger.debug(f"Duplicado detectado: {insumo.codigo_apu} - {insumo.descripcion_insumo[:30]}")

        if duplicates > 0:
            logger.info(f"📋 Removidos {duplicates} registros duplicados")
            self.processed_data = unique_data

    def _update_stats(self, record: InsumoProcesado):
        """Actualiza estadísticas con más detalle."""
        self.stats["total_records"] += 1
        self.stats[f"cat_{record.categoria}"] += 1
        self.stats[f"tipo_{record.tipo_insumo}"] += 1
        self.stats[f"fmt_{record.formato_origen}"] += 1
        self.stats[f"unit_{record.unidad_apu}"] += 1

        # Estadísticas adicionales
        if record.valor_total > 0:
            self.stats["con_valor"] += 1
        if record.cantidad > 0:
            self.stats["con_cantidad"] += 1

    def _log_stats(self):
        """Genera log detallado de estadísticas."""
        total_input = len(self.raw_records)
        total_output = len(self.processed_data)

        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE PROCESAMIENTO")
        logger.info("=" * 60)
        logger.info(f"📥 Registros de entrada: {total_input:,}")
        logger.info(f"✅ Registros procesados: {total_output:,}")
        logger.info(f"📈 Tasa de éxito: {(total_output/total_input*100):.1f}%")
        logger.info("-" * 60)

        # Desglose de rechazos
        logger.info("❌ Análisis de rechazos:")
        logger.info(f"   Errores de procesamiento: {self.stats.get('errores', 0):,}")
        logger.info(f"   Descartados (vacíos): {self.stats.get('registros_descartados', 0):,}")
        logger.info(f"   Rechazados (validación): {self.stats.get('rechazados_validacion', 0):,}")
        logger.info(f"   Excluidos (términos): {self.stats.get('excluidos_por_termino', 0):,}")
        logger.info(f"   Errores de parsing: {self.stats.get('errores_parsing', 0):,}")

        # Distribución por tipo
        logger.info("-" * 60)
        logger.info("📋 Distribución por tipo de insumo:")
        tipos = [(k.replace('tipo_', ''), v) for k, v in self.stats.items()
                 if k.startswith('tipo_')]

        for tipo, count in sorted(tipos, key=lambda x: x[1], reverse=True):
            pct = (count / total_output * 100) if total_output > 0 else 0
            logger.info(f"   {tipo:.<25} {count:>6,} ({pct:>5.1f}%)")

        # Unidades más comunes
        logger.info("-" * 60)
        logger.info("📏 Unidades APU más comunes:")
        units = [(k.replace('unit_', ''), v) for k, v in self.stats.items()
                 if k.startswith('unit_')]

        for unit, count in sorted(units, key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / total_output * 100) if total_output > 0 else 0
            logger.info(f"   {unit:.<10} {count:>6,} ({pct:>5.1f}%)")

        logger.info("=" * 60)

    def _build_optimized_dataframe(self) -> pd.DataFrame:
        """
        Construye DataFrame optimizado con tipos de datos apropiados.
        """
        if not self.processed_data:
            logger.warning("⚠️ No hay datos para construir DataFrame")
            return pd.DataFrame()

        # Convertir a diccionarios
        data_dicts = [
            item.__dict__ if hasattr(item, '__dict__') else item
            for item in self.processed_data
        ]

        # Crear DataFrame
        df = pd.DataFrame(data_dicts)

        # Mapeo de columnas
        column_mapping = {
            "codigo_apu": "CODIGO_APU",
            "descripcion_apu": "DESCRIPCION_APU",
            "unidad_apu": "UNIDAD_APU",
            "descripcion_insumo": "DESCRIPCION_INSUMO",
            "unidad_insumo": "UNIDAD_INSUMO",
            "cantidad": "CANTIDAD_APU",
            "precio_unitario": "PRECIO_UNIT_APU",
            "valor_total": "VALOR_TOTAL_APU",
            "categoria": "CATEGORIA",
            "formato_origen": "FORMATO_ORIGEN",
            "tipo_insumo": "TIPO_INSUMO",
            "rendimiento": "RENDIMIENTO",
        }

        df.rename(columns=column_mapping, inplace=True)

        # Optimizar tipos de datos
        df = self._optimize_dtypes(df)

        # Validar calidad
        self._validate_dataframe_quality(df)

        return df

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza tipos de datos para reducir memoria."""
        # Columnas numéricas a float32
        float_cols = ['CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU', 'RENDIMIENTO']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

        # Columnas categóricas
        cat_cols = ['UNIDAD_APU', 'UNIDAD_INSUMO', 'CATEGORIA', 'TIPO_INSUMO', 'FORMATO_ORIGEN']
        for col in cat_cols:
            if col in df.columns and df[col].nunique() < 100:
                df[col] = df[col].astype('category')

        # Log de optimización
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"💾 Memoria DataFrame optimizado: {memory_usage:.2f} MB")

        return df

    def _validate_dataframe_quality(self, df: pd.DataFrame):
        """Validación exhaustiva de calidad del DataFrame."""
        if 'TIPO_INSUMO' not in df.columns:
            logger.error("❌ Columna TIPO_INSUMO no encontrada")
            return

        # Análisis de distribución
        tipo_counts = df['TIPO_INSUMO'].value_counts()

        # Validaciones críticas
        validations = [
            ('SUMINISTRO', "⚠️ Sin insumos de SUMINISTRO"),
            ('MANO_DE_OBRA', "⚠️ Sin MANO DE OBRA"),
            ('EQUIPO', "⚠️ Sin EQUIPO"),
        ]

        missing = []
        for tipo, msg in validations:
            if tipo not in tipo_counts or tipo_counts[tipo] == 0:
                logger.warning(msg)
                missing.append(tipo)

        if not missing:
            logger.info("✅ Todos los tipos de insumo principales presentes")

        # Análisis de valores
        numeric_cols = ['CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU']
        for col in numeric_cols:
            if col in df.columns:
                nulls = df[col].isna().sum()
                zeros = (df[col] == 0).sum()
                negatives = (df[col] < 0).sum()

                if nulls > 0:
                    logger.warning(f"   {col}: {nulls:,} valores nulos")
                if zeros > len(df) * 0.3:
                    logger.warning(f"   {col}: {zeros:,} valores en cero ({zeros/len(df)*100:.1f}%)")
                if negatives > 0:
                    logger.error(f"   {col}: {negatives:,} valores negativos")

        # Estadísticas de resumen
        logger.info(f"📊 DataFrame final: {len(df):,} filas, {len(df.columns)} columnas")


# ============================================================================
# FUNCIONES DE UTILIDAD MEJORADAS
# ============================================================================

def calculate_unit_costs(df: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Calcula costos unitarios por APU con validación robusta.
    
    Args:
        df: DataFrame procesado
        config: Configuración opcional para cálculos
        
    Returns:
        DataFrame con costos unitarios por APU
    """
    if df.empty:
        logger.error("❌ DataFrame vacío para cálculo de costos")
        return pd.DataFrame()

    required_cols = ['CODIGO_APU', 'TIPO_INSUMO', 'VALOR_TOTAL_APU']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"❌ Columnas faltantes: {missing}")
        return pd.DataFrame()

    logger.info("🔄 Calculando costos unitarios por APU...")

    try:
        # Agrupar y sumar por APU y tipo
        grouped = df.groupby(
            ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'TIPO_INSUMO']
        )['VALOR_TOTAL_APU'].sum().reset_index()

        # Pivotear
        pivot = grouped.pivot_table(
            index=['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU'],
            columns='TIPO_INSUMO',
            values='VALOR_TOTAL_APU',
            fill_value=0,
            aggfunc='sum'
        ).reset_index()

        # Asegurar todas las columnas necesarias
        expected_columns = ['SUMINISTRO', 'MANO_DE_OBRA', 'EQUIPO', 'TRANSPORTE', 'OTRO']
        for col in expected_columns:
            if col not in pivot.columns:
                pivot[col] = 0

        # Calcular componentes
        pivot['VALOR_SUMINISTRO_UN'] = pivot.get('SUMINISTRO', 0)
        pivot['VALOR_INSTALACION_UN'] = (
            pivot.get('MANO_DE_OBRA', 0) +
            pivot.get('EQUIPO', 0)
        )
        pivot['VALOR_TRANSPORTE_UN'] = pivot.get('TRANSPORTE', 0)
        pivot['VALOR_OTRO_UN'] = pivot.get('OTRO', 0)

        # Total
        pivot['COSTO_UNITARIO_TOTAL'] = (
            pivot['VALOR_SUMINISTRO_UN'] +
            pivot['VALOR_INSTALACION_UN'] +
            pivot['VALOR_TRANSPORTE_UN'] +
            pivot['VALOR_OTRO_UN']
        )

        # Porcentajes con manejo de división por cero
        total = pivot['COSTO_UNITARIO_TOTAL'].replace(0, np.nan)
        pivot['PCT_SUMINISTRO'] = (pivot['VALOR_SUMINISTRO_UN'] / total * 100).fillna(0).round(2)
        pivot['PCT_INSTALACION'] = (pivot['VALOR_INSTALACION_UN'] / total * 100).fillna(0).round(2)
        pivot['PCT_TRANSPORTE'] = (pivot['VALOR_TRANSPORTE_UN'] / total * 100).fillna(0).round(2)
        pivot['PCT_OTRO'] = (pivot['VALOR_OTRO_UN'] / total * 100).fillna(0).round(2)

        # Ordenar y limpiar
        pivot = pivot.sort_values('CODIGO_APU')

        # Optimizar tipos
        for col in pivot.select_dtypes(include=['float64']).columns:
            pivot[col] = pivot[col].astype('float32')

        # Log resumen
        logger.info(f"✅ Costos calculados para {len(pivot):,} APUs únicos")
        logger.info(f"   💰 Suministros: ${pivot['VALOR_SUMINISTRO_UN'].sum():,.2f}")
        logger.info(f"   👷 Instalación: ${pivot['VALOR_INSTALACION_UN'].sum():,.2f}")
        logger.info(f"   🚚 Transporte: ${pivot['VALOR_TRANSPORTE_UN'].sum():,.2f}")
        logger.info(f"   📊 Total: ${pivot['COSTO_UNITARIO_TOTAL'].sum():,.2f}")

        # Validar resultados
        if pivot['COSTO_UNITARIO_TOTAL'].sum() == 0:
            logger.error("⚠️ Todos los costos calculados son cero")

        negative_costs = (pivot['COSTO_UNITARIO_TOTAL'] < 0).sum()
        if negative_costs > 0:
            logger.error(f"⚠️ {negative_costs} APUs con costos negativos")

        return pivot

    except Exception as e:
        logger.error(f"❌ Error calculando costos unitarios: {e}")
        return pd.DataFrame()

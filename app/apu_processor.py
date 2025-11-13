"""
Procesador APU con arquitectura modular de especialistas.
Mantiene compatibilidad con LoadDataStep mientras usa componentes especializados.
"""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError

from .schemas import Equipo, InsumoProcesado, ManoDeObra, Otro, Suministro, Transporte
from .utils import parse_number

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS Y DATACLASSES
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


@dataclass
class ValidationThresholds:
    """Umbrales de validación para diferentes tipos de insumos."""
    min_jornal: float = 50000
    max_jornal: float = 10000000
    min_rendimiento: float = 0.001
    max_rendimiento: float = 1000
    max_rendimiento_tipico: float = 100
    min_cantidad: float = 0.001
    max_cantidad: float = 1000000
    min_precio: float = 0.01
    max_precio: float = 1e9


# ============================================================================
# GRAMÁTICA LARK
# ============================================================================

APU_GRAMMAR = r"""
    ?start: line

    // CAMBIO CLAVE: Se eliminó la opcionalidad externa.
    // Una línea ahora DEBE tener al menos un 'field'.
    line: field (SEP field)*

    field: FIELD_VALUE?  // El campo en sí puede estar vacío (ej. 'dato1;;dato3')

    FIELD_VALUE: /[^;\r\n]+/ // El contenido del campo (si existe)
    SEP: /\s*;\s*/          // Separador flexible

    NEWLINE: /[\r\n]+/

    %import common.WS
    %ignore WS
"""


# ============================================================================
# COMPONENTES ESPECIALISTAS
# ============================================================================

class PatternMatcher:
    """Especialista en detección de patrones y clasificación de líneas."""

    # Palabras clave de encabezado de tabla
    HEADER_KEYWORDS = [
        'DESCRIPCION', 'DESCRIPCIÓN', 'DESC', 'UND', 'UNID', 'UNIDAD',
        'CANT', 'CANTIDAD', 'PRECIO', 'VALOR', 'TOTAL',
        'DESP', 'DESPERDICIO', 'REND', 'RENDIMIENTO',
        'JORNAL', 'ITEM', 'CODIGO', 'CÓDIGO'
    ]

    # Palabras clave de resumen/totalización
    SUMMARY_KEYWORDS = [
        'SUBTOTAL', 'TOTAL', 'RESUMEN', 'SUMA',
        'TOTALES', 'ACUMULADO', 'GRAN TOTAL', 'COSTO DIRECTO'
    ]

    # Categorías típicas (exactas)
    CATEGORY_PATTERNS = [
        r'^MATERIALES?$', r'^MANO\s+DE\s+OBRA$', r'^EQUIPO$',
        r'^TRANSPORTE$', r'^OTROS?$', r'^SERVICIOS?$',
        r'^HERRAMIENTAS?$', r'^SUMINISTROS?$'
    ]

    def __init__(self):
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compila todos los patrones regex para mejor performance."""
        summary_pattern = '|'.join(self.SUMMARY_KEYWORDS)
        self._pattern_cache['summary'] = re.compile(summary_pattern, re.IGNORECASE)

        category_pattern = '|'.join(self.CATEGORY_PATTERNS)
        self._pattern_cache['category'] = re.compile(category_pattern, re.IGNORECASE)

        self._pattern_cache['numeric'] = re.compile(r'[\d,.]')
        self._pattern_cache['text'] = re.compile(r'[a-zA-Z]{3,}')
        self._pattern_cache['percentage'] = re.compile(r'\d+\s*%')

    def count_header_keywords(self, text: str) -> int:
        """Cuenta cuántas palabras clave de encabezado están presentes."""
        text_upper = text.upper()
        return sum(1 for keyword in self.HEADER_KEYWORDS if keyword in text_upper)

    def is_likely_header(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente un encabezado."""
        keyword_count = self.count_header_keywords(text)

        if field_count <= 2 and keyword_count >= 3:
            return True

        words = text.upper().split()
        if words and len(words) > 2:
            header_word_ratio = sum(1 for w in words if w in self.HEADER_KEYWORDS) / len(words)
            if header_word_ratio > 0.6:
                return True

        return False

    def is_likely_summary(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente un subtotal o resumen."""
        if field_count <= 2 and self._pattern_cache['summary'].search(text):
            return True

        text_stripped = text.strip()
        for keyword in self.SUMMARY_KEYWORDS:
            if text_stripped.upper().startswith(keyword):
                return True

        return False

    def is_likely_category(self, text: str, field_count: int) -> bool:
        """Determina si una línea es probablemente una categoría."""
        if field_count <= 2:
            return bool(self._pattern_cache['category'].match(text.strip()))
        return False

    def has_numeric_content(self, text: str) -> bool:
        """Verifica si el texto contiene contenido numérico."""
        return bool(self._pattern_cache['numeric'].search(text))

    def has_percentage(self, text: str) -> bool:
        """Verifica si el texto contiene un porcentaje."""
        return bool(self._pattern_cache['percentage'].search(text))


class UnitsValidator:
    """Especialista en validación y normalización de unidades."""

    VALID_UNITS: Set[str] = {
        "UND", "UN", "UNID", "UNIDAD", "UNIDADES",
        "M", "MT", "MTS", "MTR", "MTRS", "METRO", "METROS", "ML", "KM",
        "M2", "MT2", "MTS2", "MTRS2", "METROSCUAD", "METROSCUADRADOS",
        "M3", "MT3", "MTS3", "MTRS3", "METROSCUB", "METROSCUBICOS",
        "HR", "HRS", "HORA", "HORAS", "MIN", "MINUTO", "MINUTOS",
        "DIA", "DIAS", "SEM", "SEMANA", "SEMANAS", "MES", "MESES",
        "JOR", "JORN", "JORNAL", "JORNALES",
        "G", "GR", "GRAMO", "GRAMOS",
        "KG", "KGS", "KILO", "KILOS", "KILOGRAMO", "KILOGRAMOS",
        "TON", "TONS", "TONELADA", "TONELADAS",
        "LB", "LIBRA", "LIBRAS",
        "GAL", "GLN", "GALON", "GALONES",
        "LT", "LTS", "LITRO", "LITROS", "ML", "MILILITRO", "MILILITROS",
        "VIAJE", "VIAJES", "VJE", "VJ",
        "BULTO", "BULTOS", "SACO", "SACOS", "PAQ", "PAQUETE", "PAQUETES",
        "GLOBAL", "GLB", "GB"
    }

    @classmethod
    @lru_cache(maxsize=256)
    def normalize_unit(cls, unit: str) -> str:
        """Normaliza una unidad a su forma canónica."""
        if not unit:
            return "UND"

        unit_clean = re.sub(r'[^A-Z0-9]', '', unit.upper().strip())

        unit_mappings = {
            "UNID": "UND", "UN": "UND", "UNIDAD": "UND",
            "MT": "M", "MTS": "M", "MTR": "M", "MTRS": "M",
            "JORN": "JOR", "JORNAL": "JOR", "JORNALES": "JOR",
            # Agregar más mapeos según sea necesario
        }

        return unit_mappings.get(unit_clean, unit_clean if unit_clean in cls.VALID_UNITS else "UND")

    @classmethod
    def is_valid(cls, unit: str) -> bool:
        """Verifica si una unidad es válida."""
        if not unit:
            return False
        unit_clean = re.sub(r'[^A-Z0-9]', '', unit.upper().strip())
        return unit_clean in cls.VALID_UNITS or len(unit_clean) <= 4


class NumericFieldExtractor:
    """Especialista en extracción e identificación de campos numéricos."""

    def __init__(self, thresholds: ValidationThresholds = None):
        self.pattern_matcher = PatternMatcher()
        self.thresholds = thresholds or ValidationThresholds()

    def extract_all_numeric_values(self, fields: List[str], skip_first: bool = True) -> List[float]:
        """Extrae todos los valores numéricos de los campos."""
        start_idx = 1 if skip_first else 0
        numeric_values = []

        for field in fields[start_idx:]:
            if not field:
                continue

            value = self.parse_number_safe(field)
            if value is not None and value >= 0:
                numeric_values.append(value)

        return numeric_values

    def parse_number_safe(self, value: str) -> Optional[float]:
        """Parsea un número de forma segura con detección automática de formato."""
        if not value or not isinstance(value, str):
            return None

        try:
            clean = value.strip().replace(' ', '')

            # Detectar y manejar separador decimal
            if ',' in clean and '.' not in clean:
                # Solo comas: probablemente separador decimal
                return parse_number(clean, decimal_separator="comma")
            elif '.' in clean and ',' not in clean:
                # Solo puntos: separador decimal estándar
                return parse_number(clean, decimal_separator="dot")
            elif ',' in clean and '.' in clean:
                # Ambos presentes: determinar cuál es el decimal
                if clean.rindex(',') > clean.rindex('.'):
                    clean = clean.replace('.', '')
                    return parse_number(clean, decimal_separator="comma")
                else:
                    clean = clean.replace(',', '')
                    return parse_number(clean, decimal_separator="dot")
            else:
                return float(clean) if clean else None

        except (ValueError, TypeError, AttributeError):
            return None

    def identify_mo_values(self, numeric_values: List[float]) -> Optional[Tuple[float, float]]:
        """
        Identifica rendimiento y jornal usando heurísticas inteligentes.
        Esta es LA función clave que resuelve el problema de formatos variables.
        """
        if len(numeric_values) < 2:
            return None

        # Heurística 1: Buscar por rangos típicos
        jornal_candidates = [
            v for v in numeric_values
            if self.thresholds.min_jornal <= v <= self.thresholds.max_jornal
        ]

        rendimiento_candidates = [
            v for v in numeric_values
            if (self.thresholds.min_rendimiento <= v <= self.thresholds.max_rendimiento_tipico
                and v not in jornal_candidates)
        ]

        if jornal_candidates and rendimiento_candidates:
            # Tomar el jornal más grande y el rendimiento más pequeño
            jornal = max(jornal_candidates)
            rendimiento = min(rendimiento_candidates)
            return rendimiento, jornal

        # Heurística 2: Si no encontramos con rangos, usar posición relativa
        if len(numeric_values) >= 2:
            sorted_values = sorted(numeric_values, reverse=True)

            # El valor más grande que sea >= min_jornal es probablemente el jornal
            for val in sorted_values:
                if val >= self.thresholds.min_jornal:
                    jornal = val
                    # Buscar rendimiento entre los valores restantes
                    for other_val in numeric_values:
                        if (other_val != jornal and
                            other_val <= self.thresholds.max_rendimiento_tipico):
                            return other_val, jornal
                    break

        return None

    def extract_insumo_values(self, fields: List[str], start_from: int = 2) -> List[float]:
        """Extrae valores numéricos para insumos básicos."""
        valores = []
        for i in range(start_from, len(fields)):
            if fields[i] and '%' not in fields[i]:  # Ignorar desperdicio
                val = self.parse_number_safe(fields[i])
                if val is not None and val >= 0:
                    valores.append(val)
        return valores


# ============================================================================
# TRANSFORMER ORQUESTADOR
# ============================================================================

@v_args(inline=False)
class APUTransformer(Transformer):
    """
    Orquestador que coordina los especialistas para transformar líneas.
    """

    def __init__(self, apu_context: Dict[str, Any], config: Dict[str, Any], keyword_cache: Any):
        self.apu_context = apu_context or {}
        self.config = config or {}
        self.keyword_cache = keyword_cache

        # Inicializar especialistas
        self.pattern_matcher = PatternMatcher()
        self.units_validator = UnitsValidator()
        self.thresholds = self._load_validation_thresholds()
        self.numeric_extractor = NumericFieldExtractor(self.thresholds)

        super().__init__()

    def _load_validation_thresholds(self) -> ValidationThresholds:
        """Carga los umbrales de validación desde la configuración."""
        mo_config = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})
        return ValidationThresholds(
            min_jornal=mo_config.get("min_jornal", 50000),
            max_jornal=mo_config.get("max_jornal", 10000000),
            min_rendimiento=mo_config.get("min_rendimiento", 0.001),
            max_rendimiento=mo_config.get("max_rendimiento", 1000),
            max_rendimiento_tipico=mo_config.get("max_rendimiento_tipico", 100)
        )

    def _extract_value(self, item) -> str:
        """Extrae el valor de string de un token o string."""
        if item is None:
            return ""
        if isinstance(item, Token):
            return str(item.value).strip() if item.value else ""
        if isinstance(item, (str, bytes)):
            value = item.decode('utf-8') if isinstance(item, bytes) else item
            return value.strip()
        try:
            return str(item).strip()
        except:
            return ""

    def line(self, args):
        """Procesa una línea individual delegando a especialistas."""
        fields = []

        for arg in args:
            if isinstance(arg, list):
                fields.extend([self._extract_value(f) for f in arg])
            else:
                fields.append(self._extract_value(arg))

        clean_fields = self._filter_trailing_empty(fields)

        if not clean_fields or not clean_fields[0]:
            return None

        formato = self._detect_format(clean_fields)

        if formato == FormatoLinea.DESCONOCIDO:
            return None

        return self._dispatch_builder(formato, clean_fields)

    def field(self, args):
        """Procesa un campo individual."""
        if not args:
            return ""
        return self._extract_value(args[0]) if args else ""

    def _filter_trailing_empty(self, tokens: List[str]) -> List[str]:
        """Elimina campos vacíos al final."""
        if not tokens:
            return []

        last_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i]:
                last_idx = i
                break

        return tokens[:last_idx + 1] if last_idx >= 0 else []

    def _detect_format(self, fields: List[str]) -> FormatoLinea:
        """
        Detecta el formato usando los especialistas.
        Lógica contextual que filtra ruido inteligentemente.
        """
        if not fields or not fields[0]:
            return FormatoLinea.DESCONOCIDO

        descripcion = fields[0].strip()
        num_fields = len(fields)

        # Usar PatternMatcher para filtrar ruido contextualmente
        if self._is_noise_line(descripcion, num_fields):
            return FormatoLinea.DESCONOCIDO

        if num_fields < 3:
            return FormatoLinea.DESCONOCIDO

        # Clasificar tipo de insumo
        tipo_probable = self._classify_insumo(descripcion)

        # Detectar MO_COMPLETA si es mano de obra y tiene formato válido
        if tipo_probable == TipoInsumo.MANO_DE_OBRA and num_fields >= 5:
            if self._validate_mo_format(fields):
                logger.debug(f"MO_COMPLETA detectado: {descripcion[:30]}...")
                return FormatoLinea.MO_COMPLETA

        # Detectar INSUMO_BASICO si tiene suficientes campos numéricos
        if num_fields >= 4:
            numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
            if len(numeric_values) >= 2:
                logger.debug(f"INSUMO_BASICO detectado: {descripcion[:30]}...")
                return FormatoLinea.INSUMO_BASICO

        return FormatoLinea.DESCONOCIDO

    def _is_noise_line(self, descripcion: str, num_fields: int) -> bool:
        """Detecta líneas de ruido usando el PatternMatcher."""
        descripcion_upper = descripcion.upper()

        if self.pattern_matcher.is_likely_summary(descripcion, num_fields):
            logger.debug(f"Línea de resumen ignorada: {descripcion[:30]}...")
            return True

        if self.pattern_matcher.is_likely_header(descripcion, num_fields):
            logger.debug(f"Línea de encabezado ignorada: {descripcion[:30]}...")
            return True

        if self.pattern_matcher.is_likely_category(descripcion, num_fields):
            logger.debug(f"Línea de categoría ignorada: {descripcion[:30]}...")
            return True

        return False

    def _validate_mo_format(self, fields: List[str]) -> bool:
        """Valida formato MO usando NumericFieldExtractor."""
        if len(fields) < 5:
            return False

        numeric_values = self.numeric_extractor.extract_all_numeric_values(fields)
        mo_values = self.numeric_extractor.identify_mo_values(numeric_values)

        return mo_values is not None

    def _dispatch_builder(self, formato: FormatoLinea, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Despacha al constructor apropiado."""
        try:
            if formato == FormatoLinea.MO_COMPLETA:
                return self._build_mo_completa(tokens)
            elif formato == FormatoLinea.INSUMO_BASICO:
                return self._build_insumo_basico(tokens)
            return None
        except Exception as e:
            logger.error(f"Error construyendo {formato.value}: {e}")
            return None

    def _build_mo_completa(self, tokens: List[str]) -> Optional[ManoDeObra]:
        """Construye ManoDeObra usando NumericFieldExtractor."""
        try:
            descripcion = tokens[0]
            unidad = self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "JOR"

            # Usar NumericFieldExtractor para identificar valores
            numeric_values = self.numeric_extractor.extract_all_numeric_values(tokens)
            mo_values = self.numeric_extractor.identify_mo_values(numeric_values)

            if not mo_values:
                logger.debug("No se pudieron identificar jornal y rendimiento")
                return None

            rendimiento, jornal = mo_values

            # Cálculos
            cantidad = 1.0 / rendimiento if rendimiento > 0 else 0
            valor_total = cantidad * jornal

            if cantidad <= 0 or valor_total <= 0:
                return None

            context = self.apu_context.copy()
            context.pop('cantidad_apu', None)
            context.pop('precio_unitario_apu', None)
            return ManoDeObra(
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(jornal, 2),
                valor_total=round(valor_total, 2),
                rendimiento=round(rendimiento, 6),
                formato_origen="MO_COMPLETA",
                tipo_insumo="MANO_DE_OBRA",
                categoria="MANO_DE_OBRA",
                **context
            )

        except Exception as e:
            logger.error(f"Error construyendo MO_COMPLETA: {e}")
            return None

    def _build_insumo_basico(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Construye insumo básico usando los especialistas."""
        try:
            if len(tokens) < 4:
                return None

            descripcion = tokens[0]
            unidad = self.units_validator.normalize_unit(tokens[1]) if len(tokens) > 1 else "UND"

            # Usar NumericFieldExtractor para valores
            valores = self.numeric_extractor.extract_insumo_values(tokens)

            if len(valores) < 2:
                return None

            # Interpretar valores
            cantidad = valores[0] if len(valores) > 0 else 1.0
            precio = valores[1] if len(valores) > 1 else 0.0
            total = valores[2] if len(valores) > 2 else cantidad * precio

            # Corregir si es necesario
            if total == 0 and cantidad > 0 and precio > 0:
                total = cantidad * precio
            elif precio == 0 and cantidad > 0 and total > 0:
                precio = total / cantidad

            if total <= 0:
                return None

            tipo_insumo = self._classify_insumo(descripcion)
            InsumoClass = self._get_insumo_class(tipo_insumo)

            context = self.apu_context.copy()
            context.pop('cantidad_apu', None)
            context.pop('precio_unitario_apu', None)
            return InsumoClass(
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(precio, 2),
                valor_total=round(total, 2),
                rendimiento=round(cantidad, 6),
                formato_origen="INSUMO_BASICO",
                tipo_insumo=tipo_insumo.value,
                categoria=tipo_insumo.value,
                **context
            )

        except Exception as e:
            logger.error(f"Error construyendo INSUMO_BASICO: {e}")
            return None

    @lru_cache(maxsize=2048)
    def _classify_insumo(self, descripcion: str) -> TipoInsumo:
        """Clasifica el tipo de insumo."""
        # [Implementación igual que antes]
        if not descripcion:
            return TipoInsumo.OTRO

        desc_upper = descripcion.upper()

        special_cases = {
            "HERRAMIENTA MENOR": TipoInsumo.EQUIPO,
            "MANO DE OBRA": TipoInsumo.MANO_DE_OBRA,
        }

        for case, tipo in special_cases.items():
            if case in desc_upper:
                return tipo

        # Clasificación por keywords
        if any(kw in desc_upper for kw in ["OFICIAL", "PEON", "AYUDANTE"]):
            return TipoInsumo.MANO_DE_OBRA
        if any(kw in desc_upper for kw in ["VIBRADOR", "MEZCLADORA"]):
            return TipoInsumo.EQUIPO

        return TipoInsumo.SUMINISTRO

    def _get_insumo_class(self, tipo_insumo: TipoInsumo):
        """Obtiene la clase apropiada."""
        class_mapping = {
            TipoInsumo.MANO_DE_OBRA: ManoDeObra,
            TipoInsumo.EQUIPO: Equipo,
            TipoInsumo.TRANSPORTE: Transporte,
            TipoInsumo.SUMINISTRO: Suministro,
            TipoInsumo.OTRO: Otro
        }
        return class_mapping.get(tipo_insumo, Suministro)


# ============================================================================
# PROCESADOR PRINCIPAL - COMPATIBLE CON LoadDataStep
# ============================================================================

class APUProcessor:
    """
    Procesador principal que mantiene compatibilidad con LoadDataStep
    mientras usa la arquitectura de especialistas internamente.
    """

    def __init__(self, raw_records: List[Dict[str, Any]], config: Dict[str, Any] = None):
        """
        Constructor compatible con LoadDataStep.

        Args:
            raw_records: Lista de registros del ReportParserCrudo
            config: Configuración opcional
        """
        self.raw_records = raw_records or []
        self.config = config or {}
        self.keyword_cache = None  # Cargar si es necesario

        # Crear parser una vez
        try:
            self.parser = Lark(APU_GRAMMAR, parser='lalr', transformer=None, debug=False)
            logger.info("Parser Lark inicializado con arquitectura de especialistas")
        except LarkError as e:
            logger.error(f"Error creando parser: {e}")
            self.parser = None

        # Inicializar especialistas para procesamiento fallback
        self.pattern_matcher = PatternMatcher()
        self.units_validator = UnitsValidator()
        self.thresholds = ValidationThresholds()
        self.numeric_extractor = NumericFieldExtractor(self.thresholds)

    def process_all(self) -> pd.DataFrame:
        """
        Método principal que procesa todos los registros y retorna DataFrame.
        Compatible con LoadDataStep.
        """
        logger.info(f"Iniciando procesamiento de {len(self.raw_records)} APUs con especialistas")

        all_results = []

        for i, record in enumerate(self.raw_records):
            try:
                apu_context = self._extract_apu_context(record)

                if 'lines' in record and record['lines']:
                    insumos = self._process_apu_lines(record['lines'], apu_context)
                    if insumos:
                        all_results.extend(insumos)

                if (i + 1) % 100 == 0:
                    logger.info(f"Procesados {i + 1}/{len(self.raw_records)} APUs")

            except Exception as e:
                logger.error(f"Error procesando APU {i}: {e}")
                continue

        logger.info(f"Procesamiento completado: {len(all_results)} insumos extraídos")

        # Convertir a DataFrame
        if all_results:
            return self._convert_to_dataframe(all_results)
        else:
            logger.warning("No se encontraron insumos válidos")
            return pd.DataFrame()

    def _extract_apu_context(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae contexto del APU desde el registro."""
        return {
            'codigo_apu': record.get('codigo_apu', ''),
            'descripcion_apu': record.get('descripcion_apu', ''),
            'unidad_apu': record.get('unidad_apu', ''),
            'cantidad_apu': record.get('cantidad_apu', 1.0),
            'precio_unitario_apu': record.get('precio_unitario_apu', 0.0)
        }

    def _process_apu_lines(self, lines: List[str], apu_context: Dict[str, Any]) -> List[InsumoProcesado]:
        """Procesa líneas de un APU usando el orquestador."""
        results = []

        for line_num, line in enumerate(lines):
            if not line or not line.strip():
                continue

            try:
                if self.parser:
                    # Usar parser con transformer orquestador
                    tree = self.parser.parse(line.strip())
                    transformer = APUTransformer(apu_context, self.config, self.keyword_cache)
                    insumo = transformer.transform(tree)

                    if isinstance(insumo, list):
                        insumo = insumo[0] if insumo else None
                else:
                    # Fallback directo con especialistas
                    insumo = self._process_with_specialists(line, apu_context)

                if insumo:
                    insumo.line_number = line_num
                    results.append(insumo)

            except Exception as e:
                logger.debug(f"Error procesando línea {line_num}: {e}")
                continue

        return results

    def _process_with_specialists(self, line: str, apu_context: Dict[str, Any]) -> Optional[InsumoProcesado]:
        """Procesamiento directo usando especialistas sin Lark."""
        fields = [f.strip() for f in line.split(';')]

        # Filtrar trailing empty
        while fields and not fields[-1]:
            fields.pop()

        if len(fields) < 4:
            return None

        descripcion = fields[0]

        # Usar PatternMatcher para detectar ruido
        if self.pattern_matcher.is_likely_summary(descripcion, len(fields)):
            return None

        # Extraer valores con NumericFieldExtractor
        valores = self.numeric_extractor.extract_insumo_values(fields)

        if len(valores) < 2:
            return None

        # Construir insumo
        unidad = self.units_validator.normalize_unit(fields[1]) if len(fields) > 1 else "UND"
        cantidad = valores[0] if valores else 1.0
        precio = valores[1] if len(valores) > 1 else 0.0
        total = valores[2] if len(valores) > 2 else cantidad * precio

        return Otro(
            descripcion_insumo=descripcion,
            unidad_insumo=unidad,
            cantidad=round(cantidad, 6),
            precio_unitario=round(precio, 2),
            valor_total=round(total, 2),
            rendimiento=round(cantidad, 6),
            formato_origen="SPECIALIST",
            tipo_insumo="OTRO",
            **apu_context
        )

    def _convert_to_dataframe(self, insumos: List[InsumoProcesado]) -> pd.DataFrame:
        """Convierte lista de insumos a DataFrame."""
        records = []
        for insumo in insumos:
            record = {
                'codigo_apu': getattr(insumo, 'codigo_apu', ''),
                'descripcion_apu': getattr(insumo, 'descripcion_apu', ''),
                'unidad_apu': getattr(insumo, 'unidad_apu', ''),
                'descripcion_insumo': getattr(insumo, 'descripcion_insumo', ''),
                'unidad_insumo': getattr(insumo, 'unidad_insumo', ''),
                'cantidad': getattr(insumo, 'cantidad', 0.0),
                'precio_unitario': getattr(insumo, 'precio_unitario', 0.0),
                'valor_total': getattr(insumo, 'valor_total', 0.0),
                'rendimiento': getattr(insumo, 'rendimiento', 0.0),
                'tipo_insumo': getattr(insumo, 'tipo_insumo', 'OTRO'),
                'formato_origen': getattr(insumo, 'formato_origen', '')
            }
            records.append(record)

        return pd.DataFrame(records)

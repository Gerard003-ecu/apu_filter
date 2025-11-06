import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from lark import Lark, Transformer, v_args

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

# Gramática PEG corregida - versión funcional
APU_GRAMMAR = r"""
    ?start: line
    line: (field (SEP field)*)?
    field: FIELD_VALUE
    FIELD_VALUE: /[^;\n]*/  // Cualquier cosa excepto ; y nueva línea
    SEP: ";"
    %import common.WS
    %ignore WS
"""

@v_args(inline=True)
class APUTransformer(Transformer):
    """
    Transforma el árbol de parsing de Lark en objetos dataclass usando un
    despachador inteligente con heurística y fallbacks.
    """

    def __init__(self, apu_context: Dict, config: Dict):
        self.apu_context = apu_context
        self.config = config
        super().__init__()

    def _clean_token(self, token):
        if token is None: return ""
        return token.value.strip()

    def line(self, *fields):
        """
        Punto de entrada principal: limpia, detecta el formato y despacha al constructor correcto.
        """
        tokens = [self._clean_token(f) for f in fields]

        if not tokens or not tokens[0]:
            return None

        formato_detectado = self._detect_format(tokens)

        dispatcher = {
            "MO_COMPLETA": self._build_mo_completa,
            "INSUMO_BASICO": self._build_insumo_basico,
        }

        builder = dispatcher.get(formato_detectado)

        if builder:
            return builder(tokens)
        else:
            logger.warning(f"⚠️ Formato no reconocido para la línea: {tokens}")
            return None

    def _detect_format(self, fields: List[str]) -> str:
        """
        Detecta el formato de la línea basándose en contenido y estructura.
        """
        num_fields = len(fields)
        descripcion = fields[0]
        tipo_probable = APUTransformer._classify_insumo_static(descripcion)

        if num_fields >= 6 and tipo_probable == "MANO_DE_OBRA":
            # Validar si realmente tiene la estructura de MO_COMPLETA
            try:
                jornal_total = parse_number(fields[3])
                rendimiento = parse_number(fields[4])
                thresholds = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})
                min_jornal = thresholds.get("min_jornal", 50000)

                # Si el jornal es válido y el rendimiento es plausible, es MO_COMPLETA
                if jornal_total >= min_jornal and rendimiento > 0:
                    return "MO_COMPLETA"
            except (ValueError, IndexError):
                pass

        # Si no es MO_COMPLETA o tiene 5 campos o más, es un insumo básico.
        if num_fields >= 5:
            return "INSUMO_BASICO"

        return "DESCONOCIDO"

    def _build_mo_completa(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Construye un objeto ManoDeObra. Si falla, intenta construirlo como un insumo básico."""
        try:
            descripcion, _, _, jornal_total_str, rendimiento_str, *_ = tokens
            jornal_total = parse_number(jornal_total_str)
            rendimiento = parse_number(rendimiento_str)

            # Validación de robustez
            thresholds = self.config.get("validation_thresholds", {}).get("MANO_DE_OBRA", {})
            min_jornal = thresholds.get("min_jornal", 0)
            if not (jornal_total >= min_jornal and rendimiento > 0):
                logger.warning(f"⚠️ Falla validación de MO para '{descripcion[:30]}...'. Intentando fallback.")
                return self._fallback_to_insumo_basico(tokens)

            cantidad = 1.0 / rendimiento
            valor_total = cantidad * jornal_total

            return ManoDeObra(
                descripcion_insumo=descripcion, unidad_insumo="JOR",
                cantidad=cantidad, precio_unitario=jornal_total, valor_total=valor_total,
                rendimiento=rendimiento, formato_origen="MO_COMPLETA",
                tipo_insumo="MANO_DE_OBRA",
                normalized_desc=normalize_text(descripcion),
                **self.apu_context
            )
        except (ValueError, ZeroDivisionError, IndexError) as e:
            logger.error(f"❌ Error construyendo MO_COMPLETA: {e}. Intentando fallback.")
            return self._fallback_to_insumo_basico(tokens)

    def _build_insumo_basico(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Construye un objeto de insumo genérico (Suministro, Equipo, etc.)."""
        try:
            # Maneja tanto 5 como 6 campos, ignorando el campo de desperdicio si existe
            if len(tokens) >= 6:
                descripcion, unidad, cantidad_str, _, precio_unit_str, valor_total_str, *_ = tokens
            else:
                descripcion, unidad, cantidad_str, precio_unit_str, valor_total_str, *_ = tokens

            cantidad = parse_number(cantidad_str)
            precio_unit = parse_number(precio_unit_str)
            valor_total = parse_number(valor_total_str)

            # Lógica de corrección de valores
            if valor_total == 0 and cantidad > 0 and precio_unit > 0:
                valor_total = cantidad * precio_unit

            tipo_insumo = APUTransformer._classify_insumo_static(descripcion)

            common_args = {
                "descripcion_insumo": descripcion, "unidad_insumo": unidad or "UND",
                "cantidad": cantidad, "precio_unitario": precio_unit, "valor_total": valor_total,
                "rendimiento": cantidad, "formato_origen": "INSUMO_BASICO",
                "tipo_insumo": tipo_insumo,
                "normalized_desc": normalize_text(descripcion),
                **self.apu_context
            }

            class_map = {"EQUIPO": Equipo, "TRANSPORTE": Transporte, "SUMINISTRO": Suministro}
            InsumoClass = class_map.get(tipo_insumo, Otro)
            return InsumoClass(**common_args)
        except (ValueError, IndexError) as e:
            logger.error(f"❌ Error fatal construyendo INSUMO_BASICO: {e} para tokens {tokens}")
            return None

    def _fallback_to_insumo_basico(self, tokens: List[str]) -> Optional[InsumoProcesado]:
        """Si la construcción de MO falla, intenta procesar la línea como un insumo básico."""
        logger.debug(f"🔄 Ejecutando fallback a insumo básico para: {tokens[0][:50]}...")
        return self._build_insumo_basico(tokens)

    @staticmethod
    def _get_equipo_keywords():
        return ["EQUIPO", "HERRAMIENTA", "RETROEXCAVADORA", "VOLQUETA", "MEZCLADORA",
                "VIBRADOR", "COMPACTADOR", "ANDAMIO", "FORMALETA", "ENCOFRADO",
                "MAQUINARIA", "VEHICULO", "CAMION", "GRUA", "BOMBA", "MARTILLO",
                "TALADRO", "SIERRA", "COMPRESOR"]

    @staticmethod
    def _get_mo_keywords():
        return ["M.O.", "M O ", "MANO DE OBRA", "OFICIAL", "AYUDANTE", "MAESTRO",
                "CUADRILLA", "OBRERO", "CAPATAZ", "OPERARIO", "SISO", "INGENIERO",
                "TOPOGRAFO", "RESIDENTE", "PERSONAL", "JORNAL", "PEON", "ALBAÑIL"]

    @staticmethod
    def _get_transporte_keywords():
        return ["TRANSPORTE", "ACARREO", "FLETE", "VIAJE", "MOVILIZACION", "CAMIONETA",
                "VOLQUETA", "CARGA", "DESCARGA"]

    @staticmethod
    def _get_sumistro_keywords():
        return ["SUMINISTRO", "TUBO", "TUBERIA", "ACCESORIO", "VALVULA", "MATERIAL",
                "CEMENTO", "ARENA", "GRAVA", "HIERRO", "ACERO", "ALAMBRE", "CABLE",
                "LADRILLO", "BLOQUE", "MADERA", "PINTURA", "PEGANTE", "TORNILLO",
                "CLAVO", "ARENA", "PIEDRA", "AGREGADO", "CONCRETO", "MALLA", "PERNO"]

    @staticmethod
    def _classify_insumo_static(descripcion: str) -> str:
        """
        Clasifica un insumo basado en palabras clave con precedencia estricta.
        
        Orden de precedencia: EQUIPO → MANO_DE_OBRA → TRANSPORTE → SUMINISTRO → OTRO
        """
        if not descripcion:
            return "OTRO"

        desc_upper = descripcion.upper()

        # 1. Excepciones conocidas primero
        exception_map = {
            "HERRAMIENTA MENOR": "EQUIPO",
            "HERRAMIENTA (% MO)": "EQUIPO",
            "EQUIPO Y HERRAMIENTA": "EQUIPO",
        }

        for exception, tipo in exception_map.items():
            if exception in desc_upper:
                return tipo

        # 2. Palabras clave con precedencia estricta
        keyword_hierarchy = [
            ("EQUIPO", APUTransformer._get_equipo_keywords()),
            ("MANO_DE_OBRA", APUTransformer._get_mo_keywords()),
            ("TRANSPORTE", APUTransformer._get_transporte_keywords()),
            ("SUMINISTRO", APUTransformer._get_sumistro_keywords()),
        ]

        for tipo, keywords in keyword_hierarchy:
            if any(kw in desc_upper for kw in keywords):
                return tipo

        return "OTRO"

class APUProcessor:
    """
    Procesa una lista de registros crudos de APU usando arquitectura híbrida Lark+Python.
    """

    EXCLUDED_TERMS = [
        'IMPUESTOS', 'POLIZAS', 'SEGUROS', 'GASTOS GENERALES',
        'UTILIDAD', 'ADMINISTRACION', 'RETENCIONES', 'AIU', 'A.I.U'
    ]

    def __init__(self, raw_records: List[Dict[str, str]], config: Dict):
        """
        Inicializa el procesador con registros crudos y configuración.

        Args:
            raw_records: Lista de diccionarios con registros de insumo crudo
            config: Diccionario de configuración
        """
        self.raw_records = raw_records
        self.config = config
        self.processed_data: List[InsumoProcesado] = []
        self.stats = defaultdict(int)

        # Inicializar parser Lark con gramática simplificada
        try:
            self.parser = Lark(
                APU_GRAMMAR,
                start='line',
                parser='earley',
                ambiguity='resolve'
            )
            logger.info("✅ Parser Lark inicializado con gramática simplificada")
        except Exception as e:
            logger.error(f"❌ Error inicializando parser Lark: {e}")
            raise

    def process_all(self) -> pd.DataFrame:
        """
        Orquesta el proceso completo de limpieza y estructuración de datos.

        Returns:
            DataFrame de pandas con datos procesados
        """
        logger.info(f"🚀 Iniciando procesamiento de {len(self.raw_records)} registros...")

        for idx, record in enumerate(self.raw_records):
            try:
                processed = self._process_single_record(record)
                if processed:
                    self.processed_data.append(processed)
                    self._update_stats(processed)
                else:
                    self.stats["registros_descartados"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Error en registro {idx + 1}: {e}")
                logger.debug(f"Registro problemático: {record}", exc_info=True)
                self.stats["errores"] += 1

        self._emergency_dia_units_patch()
        self._log_stats()
        return self._build_dataframe()

    def _process_single_record(self, record: Dict[str, str]) -> Optional[InsumoProcesado]:
        """
        Procesa un registro individual usando arquitectura híbrida.

        Returns:
            Objeto de datos procesado o None si debe descartarse
        """
        # 1. Normalizar campos básicos
        apu_code = clean_apu_code(record.get("apu_code", ""))
        apu_desc = record.get("apu_desc", "").strip()
        apu_unit = self._clean_unit(record.get("apu_unit", "UND"))
        category = record.get("category", "OTRO")

        if not apu_code or not apu_desc:
            logger.debug(f"Registro sin código o descripción APU: {record}")
            return None

        # 2. Parsear línea de insumo con Lark + Transformer
        insumo_line = record.get("insumo_line", "")
        insumo_obj = self._parse_with_lark_hybrid(insumo_line, apu_code, apu_desc, apu_unit, category)

        if not insumo_obj:
            logger.debug(f"No se pudo parsear insumo_line: {insumo_line[:100]}")
            return None

        # 3. Validar y corregir unidad APU
        if apu_unit == "UND":
            apu_unit = self._infer_unit_aggressive(apu_desc, category, insumo_obj.tipo_insumo)
            insumo_obj.unidad_apu = apu_unit

        # 4. Validar objeto completo
        if not self._should_add_insumo(insumo_obj):
            self.stats["rechazados_validacion"] += 1
            return None

        return insumo_obj

    def _parse_with_lark_hybrid(self, line: str, apu_code: str, apu_desc: str, apu_unit: str, category: str) -> Optional[InsumoProcesado]:
        """
        Parsea línea usando arquitectura híbrida Lark+Python.

        Args:
            line: Línea a parsear
            apu_code: Código del APU
            apu_desc: Descripción del APU
            apu_unit: Unidad del APU
            category: Categoría del APU

        Returns:
            Objeto de datos o None si falla
        """
        if not line or not line.strip():
            return None

        line = line.strip()

        try:
            # Crear contexto para el transformer
            apu_context = {
                "apu_code": apu_code,
                "apu_desc": apu_desc,
                "apu_unit": apu_unit,
                "category": category,
            }

            # Parsear con Lark + Transformer en una sola operación
            transformer = APUTransformer(apu_context, self.config)
            insumo_obj = self.parser.parse(line, transformer=transformer)

            if insumo_obj:
                logger.debug(f"✅ Parseado híbrido: {line[:60]}...")
            else:
                logger.debug(f"❌ Transformer devolvió None para: {line[:60]}...")

            return insumo_obj

        except Exception as e:
            logger.debug(f"❌ Error en parsing híbrido: '{line[:70]}...'. Error: {e}")
            self.stats["errores_parsing"] += 1
            return None

    def _clean_unit(self, unit: str) -> str:
        """Normaliza y limpia cadena de unidad."""
        if not unit or not unit.strip():
            return "UND"

        unit = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        mapping = {
            "DIAS": "DIA", "DÍAS": "DIA", "JORNAL": "JOR", "JORNALES": "JOR",
            "HORA": "HR", "HORAS": "HR", "UN": "UND", "UNIDAD": "UND",
            "UNIDADES": "UND", "METRO": "M", "METROS": "M", "MTS": "M",
            "ML": "M", "M2": "M2", "MT2": "M2", "METROSCUADRADOS": "M2",
            "M3": "M3", "MT3": "M3", "METROSCUBICOS": "M3", "KILOGRAMO": "KG",
            "KILOGRAMOS": "KG", "KILO": "KG", "TONELADA": "TON", "TONELADAS": "TON",
            "GLN": "GAL", "GALON": "GAL", "GALONES": "GAL", "LITRO": "LT",
            "LITROS": "LT", "VIAJES": "VIAJE", "VJE": "VIAJE",
        }

        return mapping.get(unit, unit)

    def _infer_unit_aggressive(self, description: str, category: str, tipo_insumo: str) -> str:
        """
        Infiere unidad de APU cuando es 'UND'.

        Args:
            description: Descripción del APU
            category: Categoría del APU
            tipo_insumo: Tipo clasificado del insumo

        Returns:
            Unidad inferida
        """
        desc_upper = description.upper()

        # Por tipo de insumo
        tipo_unit_map = {
            "MANO_DE_OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
        }

        if tipo_insumo in tipo_unit_map:
            unidad = tipo_unit_map[tipo_insumo]
            logger.debug("Unidad inferida por tipo '%s': %s", tipo_insumo, unidad)
            return unidad

        # Por keywords en descripción
        unit_keywords = {
            "M3": ["EXCAVACION", "CONCRETO", "RELLENO", "M3", "METROS CUBICOS", "CUBICOS"],
            "M2": ["PINTURA", "LOSA", "ENCHAPE", "PAÑETE", "M2", "METROS CUADRADOS", "CUADRADOS"],
            "ML": ["TUBERIA", "CANAL", "ML", "METROS LINEALES", "LINEALES", "TUBO"],
            "DIA": ["CUADRILLA", "DIA", "DIAS"],
            "JOR": ["JORNAL", "JORNALES"],
            "UN": ["ACCESORIO", "VALVULA", "PIEZA"],
            "KG": ["KILOGRAMO", "KG"],
            "TON": ["TONELADA", "TON"],
            "GAL": ["GALON", "GAL"],
        }

        for unit, words in unit_keywords.items():
            if any(w in desc_upper for w in words):
                logger.debug("Unidad inferida '%s' por keyword en: '%s...'", unit, description[:50])
                return unit

        # Por categoría
        cat_unit_map = {
            "MANO DE OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
            "MATERIALES": "UN",
        }

        if category in cat_unit_map:
            unidad = cat_unit_map[category]
            logger.debug("Unidad inferida por categoría '%s': %s", category, unidad)
            return unidad

        logger.debug(f"No se pudo inferir unidad para: '{description[:50]}...', usando UND")
        return "UND"

    def _should_add_insumo(self, insumo: InsumoProcesado) -> bool:
        """Valida si un insumo debe ser agregado usando umbrales de config.json."""
        thresholds = self.config.get("validation_thresholds", {})
        tipo_insumo = insumo.tipo_insumo

        # Validar descripción
        if not insumo.descripcion_insumo or len(insumo.descripcion_insumo.strip()) < 3:
            logger.debug("Rechazado: descripción muy corta")
            return False

        # Validar términos excluidos
        desc_upper = insumo.descripcion_insumo.upper()
        if any(term in desc_upper for term in self.EXCLUDED_TERMS):
            logger.debug(f"Excluido por término: '{insumo.descripcion_insumo[:50]}...'")
            self.stats["excluidos_por_termino"] += 1
            return False

        # Validar Mano de Obra específicamente
        if tipo_insumo == "MANO_DE_OBRA":
            mo_config = thresholds.get("MANO_DE_OBRA", {})
            min_jornal = mo_config.get("min_jornal", 0)
            max_jornal = mo_config.get("max_jornal", float('inf'))
            max_valor_total = mo_config.get("max_valor_total", float('inf'))

            if not (min_jornal <= insumo.precio_unitario <= max_jornal) and insumo.precio_unitario > 0:
                logger.warning(
                    f"Rechazado: Jornal de MO fuera de rango (${insumo.precio_unitario:,.2f}) para '{insumo.descripcion_insumo[:50]}...'"
                )
                return False

            if insumo.valor_total > max_valor_total:
                logger.warning(
                    f"Rechazado: Valor total de MO absurdo (${insumo.valor_total:,.2f}) para '{insumo.descripcion_insumo[:50]}...'"
                )
                return False

        # Validación general para otros tipos
        else:
            tipo_config = thresholds.get(tipo_insumo, thresholds.get("DEFAULT", {}))
            max_valor_total = tipo_config.get("max_valor_total", float('inf'))
            if insumo.valor_total > max_valor_total:
                logger.warning(
                    f"Rechazado: Valor total absurdo (${insumo.valor_total:,.2f}) para {tipo_insumo}"
                )
                return False

        if insumo.cantidad <= 1e-9 and insumo.valor_total <= 1e-9:
            logger.debug(f"Rechazado: sin cantidad ni valor - '{insumo.descripcion_insumo[:50]}...'")
            return False

        return True

    def _emergency_dia_units_patch(self):
        """Aplica correcciones de unidades para casos conocidos."""
        squad_codes = ['13', '14', '15', '16', '17', '18', '19', '20']
        patched = 0

        for apu in self.processed_data:
            apu_code = apu.codigo_apu
            tipo_insumo = apu.tipo_insumo
            desc_apu = apu.descripcion_apu.upper()

            # Corrección 1: Cuadrillas deben estar en DIA
            if (tipo_insumo == "MANO_DE_OBRA" and
                any(apu_code.startswith(c) for c in squad_codes) and
                apu.unidad_apu != "DIA"):
                old = apu.unidad_apu
                apu.unidad_apu = "DIA"
                logger.info(f"🔧 PARCHE DIA: {apu_code} '{old}' → 'DIA'")
                patched += 1

            # Corrección 2: MO con descripción de cuadrilla
            elif (tipo_insumo == "MANO_DE_OBRA" and
                  "CUADRILLA" in desc_apu and
                  apu.unidad_apu not in ["DIA", "JOR"]):
                old = apu.unidad_apu
                apu.unidad_apu = "DIA"
                logger.info(f"🔧 PARCHE CUADRILLA: {apu_code} '{old}' → 'DIA'")
                patched += 1

        if patched > 0:
            logger.info(f"✅ Aplicados {patched} parches de unidades")

    def _update_stats(self, record: InsumoProcesado):
        """Actualiza estadísticas de procesamiento."""
        self.stats["total_records"] += 1
        self.stats[f"cat_{record.categoria}"] += 1
        self.stats[f"tipo_{record.tipo_insumo}"] += 1
        self.stats[f"fmt_{record.formato_origen}"] += 1
        self.stats[f"unit_{record.unidad_apu}"] += 1

    def _log_stats(self):
        """Registra estadísticas finales del proceso."""
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE PROCESAMIENTO")
        logger.info("=" * 60)
        logger.info(f"✅ Registros procesados: {len(self.processed_data)}")
        logger.info(f"❌ Errores: {self.stats.get('errores', 0)}")
        logger.info(f"🗑️  Descartados: {self.stats.get('registros_descartados', 0)}")
        logger.info(f"⛔ Rechazados validación: {self.stats.get('rechazados_validacion', 0)}")
        logger.info(f"🚫 Excluidos por término: {self.stats.get('excluidos_por_termino', 0)}")
        logger.info(f"📝 Errores parsing: {self.stats.get('errores_parsing', 0)}")
        logger.info("-" * 60)

        # Estadísticas por tipo de insumo
        tipos = [k for k in self.stats.keys() if k.startswith('tipo_')]
        if tipos:
            logger.info("📋 Por tipo de insumo:")
            for tipo in sorted(tipos):
                count = self.stats[tipo]
                percentage = ((count / len(self.processed_data)) * 100) if self.processed_data else 0
                logger.info("  %-25s %6d (%5.1f%%)", tipo.replace('tipo_', ''), count, percentage)

        logger.info("=" * 60)

    def _build_dataframe(self) -> pd.DataFrame:
        """
        Construye DataFrame de pandas a partir de datos procesados.

        Returns:
            DataFrame con datos procesados y validados
        """
        if not self.processed_data:
            logger.warning("⚠️ No hay datos procesados para construir DataFrame")
            return pd.DataFrame()

        # Convertir objetos a diccionarios
        data_dicts = []
        for item in self.processed_data:
            if hasattr(item, '__dict__'):
                data_dicts.append(item.__dict__)
            elif isinstance(item, dict):
                data_dicts.append(item)
            else:
                logger.warning(f"Tipo inesperado en processed_data: {type(item)}")

        df = pd.DataFrame(data_dicts)

        # Renombrar columnas
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

        # Validar distribución de tipos
        if 'TIPO_INSUMO' in df.columns:
            tipos_count = df['TIPO_INSUMO'].value_counts()
            logger.info("📊 Distribución final de tipos de insumo:")
            for tipo, count in tipos_count.items():
                percentage = (count / len(df)) * 100
                logger.info(f"   {tipo:.<20} {count:>6} ({percentage:.1f}%)")

            self._validate_dataframe_quality(df, tipos_count)

        return df

    def _validate_dataframe_quality(self, df: pd.DataFrame, tipos_count: pd.Series):
        """
        Valida calidad del DataFrame generado.

        Args:
            df: DataFrame procesado
            tipos_count: Serie con conteo de tipos de insumo
        """
        # Verificar suministros
        if 'SUMINISTRO' not in tipos_count or tipos_count['SUMINISTRO'] == 0:
            logger.error("🚨 NO SE ENCONTRARON INSUMOS DE SUMINISTRO")

        # Verificar componentes de instalación
        has_mo = 'MANO_DE_OBRA' in tipos_count and tipos_count['MANO_DE_OBRA'] > 0
        has_equipo = 'EQUIPO' in tipos_count and tipos_count['EQUIPO'] > 0

        if not has_mo and not has_equipo:
            logger.error("🚨 NO SE ENCONTRARON COMPONENTES DE INSTALACIÓN (MO ni EQUIPO)")
        elif not has_mo:
            logger.warning("⚠️ NO SE ENCONTRÓ MANO DE OBRA para instalación")
        elif not has_equipo:
            logger.warning("⚠️ NO SE ENCONTRÓ EQUIPO para instalación")
        else:
            logger.info("✅ Componentes de instalación encontrados: MO + EQUIPO")

        # Validar valores numéricos
        for col in ['CANTIDAD_APU', 'PRECIO_UNIT_APU', 'VALOR_TOTAL_APU']:
            if col in df.columns:
                null_count = df[col].isna().sum()
                zero_count = (df[col] == 0).sum()
                if null_count > 0:
                    logger.warning(f"⚠️ {null_count} valores nulos en {col}")
                if zero_count > len(df) * 0.5:
                    percentage = zero_count / len(df) * 100
                    logger.warning("⚠️ %d valores en cero en %s (%.1f%%)", zero_count, col, percentage)


def calculate_unit_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula costos unitarios por APU sumando valores por tipo de insumo.

    INSTALACION = MANO_DE_OBRA + EQUIPO

    Args:
        df: DataFrame procesado con columna TIPO_INSUMO

    Returns:
        DataFrame con costos unitarios por APU
    """
    if df.empty or 'TIPO_INSUMO' not in df.columns:
        logger.error("❌ DataFrame inválido para cálculo de costos unitarios")
        return pd.DataFrame()

    logger.info("🔄 Calculando costos unitarios por APU...")

    # Agrupar por APU y tipo
    group_cols = ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'TIPO_INSUMO']

    try:
        grouped = df.groupby(group_cols)['VALOR_TOTAL_APU'].sum().reset_index()
    except KeyError as e:
        logger.error(f"❌ Error al agrupar: columna faltante {e}")
        return pd.DataFrame()

    # Pivotar
    try:
        pivot = grouped.pivot_table(
            index=['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU'],
            columns='TIPO_INSUMO',
            values='VALOR_TOTAL_APU',
            fill_value=0,
            aggfunc='sum'
        ).reset_index()
    except Exception as e:
        logger.error(f"❌ Error al pivotar tabla: {e}")
        return pd.DataFrame()

    # Asegurar columnas
    for col in ['SUMINISTRO', 'MANO_DE_OBRA', 'EQUIPO', 'TRANSPORTE', 'OTRO']:
        if col not in pivot.columns:
            pivot[col] = 0

    # Calcular valores
    pivot['VALOR_SUMINISTRO_UN'] = pivot.get('SUMINISTRO', 0)
    pivot['VALOR_INSTALACION_UN'] = (pivot.get('MANO_DE_OBRA', 0) + pivot.get('EQUIPO', 0))
    pivot['VALOR_TRANSPORTE_UN'] = pivot.get('TRANSPORTE', 0)
    pivot['VALOR_OTRO_UN'] = pivot.get('OTRO', 0)

    pivot['COSTO_UNITARIO_TOTAL'] = (
        pivot['VALOR_SUMINISTRO_UN'] +
        pivot['VALOR_INSTALACION_UN'] +
        pivot['VALOR_TRANSPORTE_UN'] +
        pivot['VALOR_OTRO_UN']
    )

    # Calcular porcentajes
    total = pivot['COSTO_UNITARIO_TOTAL']
    pivot['PCT_SUMINISTRO'] = (pivot['VALOR_SUMINISTRO_UN'] / total * 100).fillna(0).round(2)
    pivot['PCT_INSTALACION'] = (pivot['VALOR_INSTALACION_UN'] / total * 100).fillna(0).round(2)
    pivot['PCT_TRANSPORTE'] = (pivot['VALOR_TRANSPORTE_UN'] / total * 100).fillna(0).round(2)

    # Ordenar columnas
    column_order = [
        'CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'VALOR_SUMINISTRO_UN',
        'VALOR_INSTALACION_UN', 'VALOR_TRANSPORTE_UN', 'VALOR_OTRO_UN',
        'COSTO_UNITARIO_TOTAL', 'PCT_SUMINISTRO', 'PCT_INSTALACION', 'PCT_TRANSPORTE',
    ]

    existing_cols = [col for col in column_order if col in pivot.columns]
    remaining_cols = [col for col in pivot.columns if col not in existing_cols]
    pivot = pivot[existing_cols + remaining_cols]

    # Log de resumen
    logger.info(f"✅ Costos unitarios calculados para {len(pivot)} APUs")
    logger.info(f"   Valor total suministros: ${pivot['VALOR_SUMINISTRO_UN'].sum():,.2f}")
    logger.info(f"   Valor total instalación: ${pivot['VALOR_INSTALACION_UN'].sum():,.2f}")
    logger.info(f"   Valor total transporte: ${pivot['VALOR_TRANSPORTE_UN'].sum():,.2f}")
    logger.info(f"   Costo total: ${pivot['COSTO_UNITARIO_TOTAL'].sum():,.2f}")

    return pivot

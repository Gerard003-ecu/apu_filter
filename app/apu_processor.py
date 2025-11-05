import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
from lark import Lark, Transformer, v_args
from unidecode import unidecode

from .schemas import (
    Equipo,
    InsumoProcesado,
    ManoDeObra,
    Otro,
    Suministro,
    Transporte,
)
from .utils import clean_apu_code, parse_number

logger = logging.getLogger(__name__)

# Gramática PEG simplificada - solo tokenización, sin interpretación
APU_GRAMMAR = r"""
    ?start: line
    line: (field (SEP field)*)?   // Una línea es una lista opcional de campos
    field: /[^;]*/               // Un campo es cualquier cosa que no sea separador
    SEP: ";"                      // El separador es un terminal explícito
    %import common.WS
    %ignore WS
"""


@v_args(inline=True)
class APUTransformer(Transformer):
    """Transforma el árbol de parsing Lark en objetos de datos estructurados."""

    def __init__(self, apu_context: Dict, config: Dict):
        self.apu_context = apu_context
        self.config = config
        super().__init__()

    def line(self, *fields):
        """
        Despachador inteligente - decide el tipo de línea basado en el número de campos.
        
        Args:
            *fields: Campos tokenizados de la línea
            
        Returns:
            Objeto de datos estructurado (ManoDeObra, Suministro, etc.)
        """
        fields = [str(f).strip() if f else "" for f in fields]
        num_fields = len(fields)

        logger.debug(f"📝 Campos parseados ({num_fields}): {fields}")

        # Despachar basado en número de campos y contenido
        if num_fields >= 6:
            return self._build_mo_completa(fields)
        elif num_fields == 5:
            return self._build_insumo_basico(fields)
        elif num_fields >= 4:
            return self._build_insumo_minimal(fields)
        else:
            logger.warning(f"❌ Línea con {num_fields} campos - insuficientes: {fields}")
            return None

    def _build_mo_completa(self, fields: List[str]):
        """
        Construye objeto ManoDeObra desde formato completo (6+ campos).
        
        Formato: descripcion; jornal_base; prestaciones; jornal_total; rendimiento; valor_total
        """
        try:
            descripcion = fields[0]
            jornal_base = parse_number(fields[1])
            prestaciones = fields[2]  # Puede ser número o texto
            jornal_total = parse_number(fields[3])
            rendimiento = parse_number(fields[4])
            valor_total = parse_number(fields[5])

            # Calcular cantidad para MO
            cantidad = 1.0 / rendimiento if rendimiento > 0 else 0

            # 🔥 RECALCULACIÓN INCONDICIONAL para MO
            valor_total_recalc = cantidad * jornal_total

            logger.debug(
                f"👷 MO Completa: '{descripcion[:30]}...' - "
                f"Rend: {rendimiento}, Cant: {cantidad:.6f}, "
                f"VrTotal: {valor_total} → {valor_total_recalc:.2f} (RECALCULADO)"
            )

            return ManoDeObra(
                codigo_apu=self.apu_context["apu_code"],
                descripcion_apu=self.apu_context["apu_desc"],
                unidad_apu=self.apu_context["apu_unit"],
                descripcion_insumo=descripcion,
                unidad_insumo="JOR",
                cantidad=round(cantidad, 6),
                precio_unitario=round(jornal_total, 2),
                valor_total=round(valor_total_recalc, 2),
                categoria=self.apu_context["category"],
                formato_origen="MO_COMPLETA",
                tipo_insumo="MANO_DE_OBRA",
                normalized_desc=self._normalize_text(descripcion),
                rendimiento=round(rendimiento, 6)
            )

        except (ValueError, ZeroDivisionError, IndexError) as e:
            logger.warning(f"❌ Error construyendo MO completa: {e} - Campos: {fields}")
            return None

    def _build_insumo_basico(self, fields: List[str]):
        """
        Construye objeto de insumo desde formato básico (5 campos).
        
        Formato: descripcion; unidad; cantidad; precio_unit; valor_total
        """
        try:
            descripcion = fields[0]
            unidad = fields[1] if fields[1] else "UND"
            cantidad = parse_number(fields[2])
            precio_unit = parse_number(fields[3])
            valor_total = parse_number(fields[4])

            # Corregir valores faltantes
            if valor_total == 0 and cantidad > 0 and precio_unit > 0:
                valor_total = cantidad * precio_unit
            elif cantidad == 0 and valor_total > 0 and precio_unit > 0:
                cantidad = valor_total / precio_unit
            elif precio_unit == 0 and cantidad > 0 and valor_total > 0:
                precio_unit = valor_total / cantidad

            # Clasificar tipo
            tipo_insumo = self._classify_insumo(descripcion)

            # Determinar clase de datos
            insumo_class = self._get_insumo_class(tipo_insumo)

            return insumo_class(
                codigo_apu=self.apu_context["apu_code"],
                descripcion_apu=self.apu_context["apu_desc"],
                unidad_apu=self.apu_context["apu_unit"],
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(precio_unit, 2),
                valor_total=round(valor_total, 2),
                categoria=self.apu_context["category"],
                formato_origen="INSUMO_BASICO",
                tipo_insumo=tipo_insumo,
                normalized_desc=self._normalize_text(descripcion),
                rendimiento=0.0
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"❌ Error construyendo insumo básico: {e} - Campos: {fields}")
            return None

    def _build_insumo_minimal(self, fields: List[str]):
        """
        Construye objeto de insumo desde formato mínimo (4 campos).
        
        Formato: descripcion; unidad; cantidad; precio_unit
        """
        try:
            descripcion = fields[0]
            unidad = fields[1] if len(fields) > 1 and fields[1] else "UND"
            cantidad = parse_number(fields[2]) if len(fields) > 2 else 0
            precio_unit = parse_number(fields[3]) if len(fields) > 3 else 0

            # Calcular valor total
            valor_total = cantidad * precio_unit if cantidad > 0 and precio_unit > 0 else 0

            # Clasificar tipo
            tipo_insumo = self._classify_insumo(descripcion)

            # Determinar clase de datos
            insumo_class = self._get_insumo_class(tipo_insumo)

            return insumo_class(
                codigo_apu=self.apu_context["apu_code"],
                descripcion_apu=self.apu_context["apu_desc"],
                unidad_apu=self.apu_context["apu_unit"],
                descripcion_insumo=descripcion,
                unidad_insumo=unidad,
                cantidad=round(cantidad, 6),
                precio_unitario=round(precio_unit, 2),
                valor_total=round(valor_total, 2),
                categoria=self.apu_context["category"],
                formato_origen="INSUMO_MINIMAL",
                tipo_insumo=tipo_insumo,
                normalized_desc=self._normalize_text(descripcion),
                rendimiento=0.0
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"❌ Error construyendo insumo mínimo: {e} - Campos: {fields}")
            return None

    def _classify_insumo(self, descripcion: str) -> str:
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
            ("EQUIPO", self._get_equipo_keywords()),
            ("MANO_DE_OBRA", self._get_mo_keywords()),
            ("TRANSPORTE", self._get_transporte_keywords()),
            ("SUMINISTRO", self._get_sumistro_keywords()),
        ]

        for tipo, keywords in keyword_hierarchy:
            if any(kw in desc_upper for kw in keywords):
                return tipo

        return "OTRO"

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

    def _get_insumo_class(self, tipo_insumo: str):
        """Devuelve la clase de datos apropiada para el tipo de insumo."""
        class_map = {
            "MANO_DE_OBRA": ManoDeObra,
            "EQUIPO": Equipo,
            "SUMINISTRO": Suministro,
            "TRANSPORTE": Transporte,
            "OTRO": Otro,
        }
        return class_map.get(tipo_insumo, Otro)

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparación."""
        if not text:
            return ""
        text = unidecode(text.lower().strip())
        text = re.sub(r"[^a-z0-9\s#\-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


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

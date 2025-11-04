import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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


class APUProcessor:
    """
    Procesa una lista de registros crudos de APU para limpiarlos y estructurarlos.

    Esta clase implementa la lógica de negocio para transformar datos de APU
    extraídos en un formato tabular y normalizado, listo para ser analizado.
    """

    # Palabras clave para clasificar insumos por tipo
    SUMINISTRO_KEYWORDS = [
        "SUMINISTRO", "TUBO", "TUBERIA", "ACCESORIO", "VALVULA", "MATERIAL",
        "CEMENTO", "ARENA", "GRAVA", "HIERRO", "ACERO", "ALAMBRE", "CABLE",
        "LADRILLO", "BLOQUE", "MADERA", "PINTURA", "PEGANTE", "TORNILLO"
    ]

    MANO_OBRA_KEYWORDS = [
        "M.O.", "M O ", "MANO DE OBRA", "OFICIAL", "AYUDANTE", "MAESTRO",
        "CUADRILLA", "OBRERO", "CAPATAZ", "OPERARIO", "SISO", "INGENIERO",
        "TOPOGRAFO", "RESIDENTE", "PERSONAL"
    ]

    EQUIPO_KEYWORDS = [
        "EQUIPO", "HERRAMIENTA", "RETROEXCAVADORA", "VOLQUETA", "MEZCLADORA",
        "VIBRADOR", "COMPACTADOR", "ANDAMIO", "FORMALETA", "ENCOFRADO",
        "MAQUINARIA", "VEHICULO", "CAMION", "GRUA", "BOMBA"
    ]

    TRANSPORTE_KEYWORDS = [
        "TRANSPORTE", "ACARREO", "FLETE", "VIAJE", "MOVILIZACION"
    ]

    EXCLUDED_TERMS = [
        'IMPUESTOS', 'POLIZAS', 'SEGUROS', 'GASTOS GENERALES',
        'UTILIDAD', 'ADMINISTRACION', 'RETENCIONES', 'AIU', 'A.I.U',
        'EQUIPO Y HERRAMIENTA'
    ]

    def __init__(self, raw_records: List[Dict[str, str]]):
        """
        Inicializa el procesador con una lista de registros crudos.

        Args:
            raw_records: Una lista de diccionarios con registros de insumo crudo.
        """
        self.raw_records = raw_records
        self.processed_data: List[InsumoProcesado] = []
        self.stats = defaultdict(int)

    def process_all(self) -> pd.DataFrame:
        """
        Orquesta el proceso completo de limpieza y estructuración de los datos.

        Returns:
            Un DataFrame de pandas con los datos procesados.
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
        Procesa un registro individual con recalculación INCONDICIONAL para Mano de Obra.

        Para Mano de Obra:
        - SIEMPRE recalcula valor_total = cantidad * precio_unitario
        - Ignora el valor_total del archivo fuente
        - Calcula cantidad como 1/rendimiento si rendimiento > 0

        Returns:
            Diccionario con los datos procesados o None si debe descartarse.
        """
        # 1. Normalizar campos básicos
        apu_code = clean_apu_code(record.get("apu_code", ""))
        apu_desc = record.get("apu_desc", "").strip()
        apu_unit = self._clean_unit(record.get("apu_unit", "UND"))
        category = record.get("category", "OTRO")

        if not apu_code or not apu_desc:
            logger.debug(f"Registro sin código o descripción APU: {record}")
            return None

        # 2. Parsear línea de insumo
        parsed = self._parse_insumo_line(record.get("insumo_line", ""))
        if not parsed:
            insumo_line = record.get('insumo_line', '')[:100]
            logger.debug(f"No se pudo parsear insumo_line: {insumo_line}")
            return None

        descripcion_insumo = parsed.get("descripcion", "").strip()

        # 3. Validar descripción
        if self._is_excluded_insumo(descripcion_insumo):
            self.stats["excluidos_por_termino"] += 1
            return None

        # 4. Clasificar tipo de insumo ANTES de inferir unidad
        tipo_insumo = self._classify_insumo(descripcion_insumo)

        # 5. Inferir unidad de APU si es "UND"
        if apu_unit == "UND":
            apu_unit = self._infer_unit_aggressive(apu_desc, category, tipo_insumo)

        # 6. Extraer y calcular valores numéricos según tipo de insumo
        cantidad, precio_unit, valor_total, rendimiento = self._extract_and_calculate_values(
            parsed, tipo_insumo, descripcion_insumo
        )

        # 7. Validar antes de agregar
        if not self._should_add_insumo(
            descripcion_insumo, cantidad, valor_total, tipo_insumo
        ):
            self.stats["rechazados_validacion"] += 1
            return None

        # 8. Normalizar descripción
        normalized_desc = self._normalize_text_single(descripcion_insumo)

        # 9. Construir el diccionario de argumentos comunes
        common_args = {
            "codigo_apu": apu_code,
            "descripcion_apu": apu_desc,
            "unidad_apu": apu_unit,
            "descripcion_insumo": descripcion_insumo,
            "unidad_insumo": parsed.get("unidad", "UND"),
            "cantidad": round(cantidad, 6),
            "precio_unitario": round(precio_unit, 2),
            "valor_total": round(valor_total, 2),
            "categoria": category,
            "formato_origen": parsed.get("formato", "GENERIC"),
            "tipo_insumo": tipo_insumo,
            "normalized_desc": normalized_desc,
        }

        # 10. Instanciar la clase de datos correcta
        try:
            if tipo_insumo == "MANO_DE_OBRA":
                return ManoDeObra(rendimiento=round(rendimiento, 6), **common_args)
            elif tipo_insumo == "EQUIPO":
                return Equipo(**common_args)
            elif tipo_insumo == "SUMINISTRO":
                return Suministro(**common_args)
            elif tipo_insumo == "TRANSPORTE":
                return Transporte(**common_args)
            else:
                return Otro(**common_args)
        except TypeError as e:
            logger.error(
                "Error al crear objeto de datos para %s (%s): %s",
                apu_code, descripcion_insumo, e
            )
            return None

    def _extract_and_calculate_values(
        self, parsed: Dict[str, str], tipo_insumo: str, descripcion: str
    ) -> Tuple[float, float, float, float]:
        """
        Extrae y calcula valores numéricos con lógica específica por tipo de insumo.

        🔥 CRÍTICO: Para MANO_DE_OBRA, la recalculación es INCONDICIONAL.

        Args:
            parsed: Diccionario con valores parseados
            tipo_insumo: Tipo clasificado del insumo
            descripcion: Descripción del insumo (para logging)

        Returns:
            Tupla (cantidad, precio_unit, valor_total, rendimiento)
        """
        if tipo_insumo == "MANO_DE_OBRA":
            return self._calculate_mo_values_unconditional(parsed, descripcion)
        else:
            return self._calculate_regular_values(parsed, descripcion)

    def _calculate_mo_values_unconditional(
        self, parsed: Dict[str, str], descripcion: str
    ) -> Tuple[float, float, float, float]:
        """Calcula valores para Mano de Obra con validación reforzada."""

        # Extraer jornal total con múltiples fallbacks
        jornal_total = parse_number(parsed.get("jornal_total", "0"))

        if jornal_total == 0:
            jornal_total = parse_number(parsed.get("precio_unit", "0"))

        if jornal_total == 0:
            valor_total = parse_number(parsed.get("valor_total", "0"))
            rendimiento = parse_number(parsed.get("rendimiento", "0"))
            if valor_total > 0 and rendimiento > 0:
                jornal_total = valor_total * rendimiento
                logger.warning(
                    "MO '%s...': Jornal inferido desde valor_total * rendimiento: %.2f",
                    descripcion[:30],
                    jornal_total,
                )

        # Validar rango razonable del jornal
        if jornal_total < 10_000 or jornal_total > 1_000_000:
            logger.warning(
                "⚠️ MO '%s...': Jornal fuera de rango típico: $%.2f",
                descripcion[:30],
                jornal_total,
            )

        # Resto de la lógica existente...
        rendimiento = parse_number(parsed.get("rendimiento", "0"))

        if rendimiento > 0:
            cantidad = 1.0 / rendimiento
        else:
            valor_total_parseado = parse_number(parsed.get("valor_total", "0"))
            if valor_total_parseado > 0 and jornal_total > 0:
                cantidad = valor_total_parseado / jornal_total
                rendimiento = 1.0 / cantidad if cantidad > 0 else 0
            else:
                cantidad = 0
                logger.error(
                    "❌ MO '%s...': Sin datos suficientes para cálculo",
                    descripcion[:30],
                )

        # RECALCULACIÓN INCONDICIONAL + validación
        valor_total_recalc = cantidad * jornal_total

        # Validar coherencia con valor original (si existe)
        valor_total_original = parse_number(parsed.get("valor_total", "0"))
        if valor_total_original > 0:
            diferencia = abs(valor_total_recalc - valor_total_original)
            if diferencia > 100:  # Tolerancia de $100
                logger.warning(
                    "MO '%s...': Diferencia en valor_total: Original=$%.2f, Recalc=$%.2f",
                    descripcion[:30],
                    valor_total_original,
                    valor_total_recalc,
                )

        return cantidad, jornal_total, valor_total_recalc, rendimiento

    def _calculate_regular_values(
        self, parsed: Dict[str, str], descripcion: str
    ) -> Tuple[float, float, float, float]:
        """
        Calcula valores para insumos regulares (no MO) con corrección de valores faltantes.

        Args:
            parsed: Datos parseados
            descripcion: Descripción del insumo (para logging)

        Returns:
            Tupla (cantidad, precio_unit, valor_total, rendimiento)
        """
        cantidad = parse_number(parsed.get("cantidad", "0"))
        precio_unit = parse_number(parsed.get("precio_unit", "0"))
        valor_total = parse_number(parsed.get("valor_total", "0"))
        rendimiento = parse_number(parsed.get("rendimiento", "0"))

        # Contar valores existentes
        valores_existentes = sum([
            1 for v in [cantidad, precio_unit, valor_total] if v > 0
        ])

        # Si tenemos al menos 2 valores, podemos calcular el faltante
        if valores_existentes >= 2:
            if valor_total == 0 and cantidad > 0 and precio_unit > 0:
                valor_total = cantidad * precio_unit
                logger.debug(
                    "Calculado valor_total: %.2f = %.6f * %.2f para '%s...'",
                    valor_total, cantidad, precio_unit, descripcion[:30]
                )

            elif cantidad == 0 and valor_total > 0 and precio_unit > 0:
                cantidad = valor_total / precio_unit
                logger.debug(
                    "Calculada cantidad: %.6f = %.2f / %.2f para '%s...'",
                    cantidad, valor_total, precio_unit, descripcion[:30]
                )

            elif precio_unit == 0 and cantidad > 0 and valor_total > 0:
                precio_unit = valor_total / cantidad
                logger.debug(
                    "Calculado precio_unit: %.2f = %.2f / %.6f para '%s...'",
                    precio_unit, valor_total, cantidad, descripcion[:30]
                )

        # Para insumos regulares, si hay cantidad pero no rendimiento
        if rendimiento == 0 and cantidad > 0:
            rendimiento = cantidad

        return cantidad, precio_unit, valor_total, rendimiento

    def _classify_insumo(self, descripcion: str) -> str:
        """
        Clasifica un insumo según su descripción con orden de prioridad.

        Orden de clasificación:
        1. MANO_DE_OBRA (mayor prioridad)
        2. EQUIPO
        3. TRANSPORTE
        4. SUMINISTRO
        5. Inferencia por unidad
        6. OTRO (por defecto)

        Args:
            descripcion: Descripción del insumo

        Returns:
            Tipo de insumo clasificado
        """
        desc_upper = descripcion.upper()

        # Prioridad 1: Mano de Obra
        if any(kw in desc_upper for kw in self.MANO_OBRA_KEYWORDS):
            return "MANO_DE_OBRA"

        # Prioridad 2: Equipo
        if any(kw in desc_upper for kw in self.EQUIPO_KEYWORDS):
            return "EQUIPO"

        # Prioridad 3: Transporte
        if any(kw in desc_upper for kw in self.TRANSPORTE_KEYWORDS):
            return "TRANSPORTE"

        # Prioridad 4: Suministro
        if any(kw in desc_upper for kw in self.SUMINISTRO_KEYWORDS):
            return "SUMINISTRO"

        # Prioridad 5: Intentar inferir por patrón de unidad
        tipo_inferido = self._infer_type_by_unit(desc_upper)
        if tipo_inferido:
            logger.debug(
                "Tipo inferido por unidad: %s para '%s...'",
                tipo_inferido, descripcion[:50]
            )
            return tipo_inferido

        # Por defecto
        logger.debug(f"Insumo sin clasificar: '{descripcion[:50]}...'")
        return "OTRO"

    def _infer_type_by_unit(self, desc_upper: str) -> Optional[str]:
        """
        Infiere el tipo de insumo por patrones de unidad en la descripción.

        Args:
            desc_upper: Descripción en mayúsculas.

        Returns:
            Tipo inferido o None.
        """
        unit_patterns = {
            "SUMINISTRO": [
                r'\bML\b', r'\bM\b(?!\.O\.)', r'\bMTS\b', r'\bMETROS?\b',
                r'\bUN\b', r'\bUND\b', r'\bUNIDAD(ES)?\b',
                r'\bKG\b', r'\bTON\b', r'\bM3\b', r'\bM2\b',
                r'\bLT\b', r'\bGAL(ON)?(ES)?\b'
            ],
            "MANO_DE_OBRA": [
                r'\bJOR(NAL)?(ES)?\b', r'\bHORA\b(?=.*(?:OFICIAL|AYUDANTE|OBRERO))',
                r'\bHR\b(?=.*(?:OFICIAL|AYUDANTE|OBRERO))'
            ],
            "EQUIPO": [
                r'\bDIA\b(?!.*(?:OFICIAL|AYUDANTE))',
                r'\bHORA\b(?!.*(?:OFICIAL|AYUDANTE))'
            ]
        }

        for tipo, patterns in unit_patterns.items():
            if any(re.search(pattern, desc_upper) for pattern in patterns):
                return tipo

        return None

    def _parse_insumo_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parsea una línea de insumo textual usando patrones regex especializados.

        Args:
            line: La cadena de texto que representa la línea de insumo.

        Returns:
            Un diccionario con los campos parseados o None si no se puede parsear.
        """
        if not line or not line.strip():
            return None

        line = line.strip()
        patterns = self._get_parsing_patterns()

        # Intentar con cada patrón en orden de especificidad
        for name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                data = match.groupdict()
                data["formato"] = name
                logger.debug(f"✓ Parseado con patrón '{name}': {line[:60]}...")
                return data

        # Fallback: parsing genérico por separador
        return self._parse_generic_fallback(line)

    def _parse_generic_fallback(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parseo genérico cuando ningún patrón específico funciona.

        Args:
            line: Línea a parsear

        Returns:
            Diccionario con campos parseados o None
        """
        parts = [p.strip() for p in line.split(";")]

        # Caso con suficientes campos (6 o más)
        if len(parts) >= 6:
            result = {
                "descripcion": parts[0],
                "unidad": parts[1] if parts[1] else "UND",
                "cantidad": parts[2] if parts[2] else "0",
                "desperdicio": parts[3] if len(parts) > 3 else "0",
                "precio_unit": parts[4] if len(parts) > 4 else "0",
                "valor_total": parts[5] if len(parts) > 5 else "0",
                "rendimiento": "0",
                "formato": "FALLBACK_FULL",
            }
            logger.debug(f"⚠ Fallback completo aplicado: {line[:60]}...")
            return result

        # Caso con campos mínimos (3 o más)
        elif len(parts) >= 3:
            result = {
                "descripcion": parts[0],
                "unidad": parts[1] if len(parts) > 1 and parts[1] else "UND",
                "cantidad": parts[2] if len(parts) > 2 else "0",
                "precio_unit": parts[3] if len(parts) > 3 else "0",
                "valor_total": parts[4] if len(parts) > 4 else "0",
                "rendimiento": "0",
                "formato": "FALLBACK_MINIMAL",
            }
            logger.debug(f"⚠ Fallback minimal aplicado: {line[:60]}...")
            return result

        logger.warning(f"❌ No se pudo parsear línea: {line[:100]}...")
        return None

    def _get_parsing_patterns(self) -> Dict[str, re.Pattern]:
        """
        Define los patrones regex para parsear diferentes formatos de líneas de insumo.

        Returns:
            Diccionario con nombre del patrón y regex compilado.
        """
        return {
            # Patrón para Mano de Obra con todos los campos
            "MO_COMPLETA": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|M\s*O\s+|MANO\s+DE\s+OBRA).*?);\s*"
                r"(?P<jornal_base>[\d.,\s]+);\s*"
                r"(?P<prestaciones>[\d%.,\s]+);\s*"
                r"(?P<jornal_total>[\d.,\s]+);\s*"
                r"(?P<rendimiento>[\d.,\s]+);\s*"
                r"(?P<valor_total>[\d.,\s]+)",
                re.IGNORECASE
            ),

            # Patrón para MO simplificado (sin prestaciones)
            "MO_SIMPLE": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|M\s*O\s+|MANO\s+DE\s+OBRA).*?);\s*"
                r"(?P<unidad>[^;]*);\s*"
                r"(?P<rendimiento>[\d.,\s]+);\s*"
                r"(?P<jornal_total>[\d.,\s]+);\s*"
                r"(?P<valor_total>[\d.,\s]+)",
                re.IGNORECASE
            ),

            # Patrón para insumo completo con desperdicio
            "INSUMO_CON_DESPERDICIO": re.compile(
                r"^(?P<descripcion>[^;]+);\s*"
                r"(?P<unidad>[^;]*);\s*"
                r"(?P<cantidad>[\d.,\s]*);\s*"
                r"(?P<desperdicio>[\d.,\s%]*);\s*"
                r"(?P<precio_unit>[\d.,\s]*);\s*"
                r"(?P<valor_total>[\d.,\s]*)"
            ),

            # Patrón para insumo básico (sin desperdicio)
            "INSUMO_BASICO": re.compile(
                r"^(?P<descripcion>[^;]+);\s*"
                r"(?P<unidad>[^;]*);\s*"
                r"(?P<cantidad>[\d.,\s]*);\s*"
                r"(?P<precio_unit>[\d.,\s]*);\s*"
                r"(?P<valor_total>[\d.,\s]*)"
            ),
        }

    def _infer_unit_aggressive(
        self, description: str, category: str, tipo_insumo: str
    ) -> str:
        """
        Infiere la unidad de un APU cuando esta es 'UND' (indefinida).

        Args:
            description: La descripción del APU.
            category: La categoría del APU.
            tipo_insumo: El tipo clasificado del insumo.

        Returns:
            La unidad inferida como una cadena de texto.
        """
        desc_upper = description.upper()

        # Primero: Por tipo de insumo (más confiable)
        tipo_unit_map = {
            "MANO_DE_OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
        }

        if tipo_insumo in tipo_unit_map:
            unidad = tipo_unit_map[tipo_insumo]
            logger.debug("Unidad inferida por tipo '%s': %s", tipo_insumo, unidad)
            return unidad

        # Segundo: Por keywords en descripción
        unit_keywords = {
            "M3": ["EXCAVACION", "CONCRETO", "RELLENO", "M3",
                   "METROS CUBICOS", "CUBICOS"],
            "M2": ["PINTURA", "LOSA", "ENCHAPE", "PAÑETE", "M2",
                   "METROS CUADRADOS", "CUADRADOS"],
            "ML": ["TUBERIA", "CANAL", "ML", "METROS LINEALES",
                   "LINEALES", "TUBO"],
            "DIA": ["CUADRILLA", "DIA", "DIAS"],
            "JOR": ["JORNAL", "JORNALES"],
            "UN": ["ACCESORIO", "VALVULA", "PIEZA"],
            "KG": ["KILOGRAMO", "KG"],
            "TON": ["TONELADA", "TON"],
            "GAL": ["GALON", "GAL"],
        }

        for unit, words in unit_keywords.items():
            if any(w in desc_upper for w in words):
                logger.debug(
                    "Unidad inferida '%s' por keyword en: '%s...'",
                    unit, description[:50]
                )
                return unit

        # Tercero: Por categoría
        cat_unit_map = {
            "MANO DE OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
            "MATERIALES": "UN",
        }

        if category in cat_unit_map:
            unidad = cat_unit_map[category]
            logger.debug(
                "Unidad inferida por categoría '%s': %s", category, unidad
            )
            return unidad

        logger.debug(f"No se pudo inferir unidad para: '{description[:50]}...', usando UND")
        return "UND"

    def _clean_unit(self, unit: str) -> str:
        """
        Normaliza y limpia una cadena de texto de unidad.

        Args:
            unit: La cadena de texto de la unidad a limpiar.

        Returns:
            La unidad normalizada.
        """
        if not unit or not unit.strip():
            return "UND"

        # Remover caracteres especiales y normalizar
        unit = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        # Mapeo de variaciones a unidades estándar
        mapping = {
            # Tiempo
            "DIAS": "DIA",
            "DÍAS": "DIA",
            "JORNAL": "JOR",
            "JORNALES": "JOR",
            "HORA": "HR",
            "HORAS": "HR",

            # Unidades
            "UN": "UND",
            "UNIDAD": "UND",
            "UNIDADES": "UND",

            # Distancia
            "METRO": "M",
            "METROS": "M",
            "MTS": "M",
            "ML": "M",

            # Área
            "M2": "M2",
            "MT2": "M2",
            "METROSCUADRADOS": "M2",

            # Volumen
            "M3": "M3",
            "MT3": "M3",
            "METROSCUBICOS": "M3",

            # Peso
            "KILOGRAMO": "KG",
            "KILOGRAMOS": "KG",
            "KILO": "KG",
            "TONELADA": "TON",
            "TONELADAS": "TON",

            # Volumen líquido
            "GLN": "GAL",
            "GALON": "GAL",
            "GALONES": "GAL",
            "LITRO": "LT",
            "LITROS": "LT",

            # Transporte
            "VIAJES": "VIAJE",
            "VJE": "VIAJE",
        }

        return mapping.get(unit, unit)

    def _is_excluded_insumo(self, desc: str) -> bool:
        """
        Verifica si un insumo debe ser excluido basado en su descripción.

        Args:
            desc: La descripción del insumo.

        Returns:
            True si el insumo debe ser excluido, False en caso contrario.
        """
        if not desc or len(desc.strip()) < 2:
            return True

        desc_u = desc.upper()

        # Excluir términos exactos o contenidos
        for term in self.EXCLUDED_TERMS:
            if term in desc_u:
                logger.debug(f"Excluido por término '{term}': '{desc[:50]}...'")
                return True

        # Excluir descripciones muy cortas
        if len(desc.strip()) < 3:
            logger.debug(f"Excluido por descripción muy corta: '{desc}'")
            return True

        return False

    def _should_add_insumo(
        self, desc: str, cantidad: float, valor_total: float, tipo_insumo: str
    ) -> bool:
        """
        Valida si un insumo debe ser agregado a los datos procesados.

        Args:
            desc: Descripción del insumo
            cantidad: Cantidad del insumo
            valor_total: Valor total del insumo
            tipo_insumo: Tipo clasificado del insumo

        Returns:
            True si el insumo pasa las validaciones, False en caso contrario
        """
        # Validación 1: Descripción válida
        if not desc or len(desc.strip()) < 3:
            logger.debug("Rechazado: descripción muy corta o vacía")
            return False

        # Validación 2: Debe tener al menos cantidad o valor
        if cantidad <= 0 and valor_total <= 0:
            logger.debug(f"Rechazado: sin cantidad ni valor - '{desc[:30]}...'")
            return False

        # Validación 3: Umbrales por tipo de insumo
        umbrales = {
            "MANO_DE_OBRA": {
                "cantidad_max": 1000,
                "valor_max": 999_999,
            },
            "EQUIPO": {
                "cantidad_max": 10000,
                "valor_max": 9_999_999,
            },
            "SUMINISTRO": {
                "cantidad_max": 100000,
                "valor_max": 99_999_999,
            },
            "TRANSPORTE": {
                "cantidad_max": 10000,
                "valor_max": 9_999_999,
            },
            "OTRO": {
                "cantidad_max": 1_000_000,
                "valor_max": 999_999_999,
            }
        }

        umbral = umbrales.get(tipo_insumo, umbrales["OTRO"])

        # Validar cantidad
        if cantidad > umbral["cantidad_max"]:
            logger.warning(
                "Rechazado: cantidad absurda %.2f para %s - '%s...'",
                cantidad, tipo_insumo, desc[:30]
            )
            return False

        # Validar valor total
        if valor_total > umbral["valor_max"]:
            logger.warning(
                "Rechazado: valor total absurdo $%.2f para %s - '%s...'",
                valor_total, tipo_insumo, desc[:30]
            )
            return False

        # Validación 4: Coherencia entre cantidad y valor
        if cantidad > 0 and valor_total > 0:
            precio_implicito = valor_total / cantidad

            precio_umbrales = {
                "MANO_DE_OBRA": (1000, 500_000),
                "EQUIPO": (100, 1_000_000),
                "SUMINISTRO": (1, 10_000_000),
                "TRANSPORTE": (100, 500_000),
                "OTRO": (0.01, 100_000_000),
            }

            umbrales_precio = precio_umbrales.get(
                tipo_insumo, precio_umbrales["OTRO"]
            )
            min_precio, max_precio = umbrales_precio

            if not (min_precio <= precio_implicito <= max_precio):
                logger.warning(
                    "Rechazado: precio unitario implícito sospechoso $%.2f "
                    "para %s (esperado entre $%d y $%d) - '%s...'",
                    precio_implicito, tipo_insumo,
                    min_precio, max_precio, desc[:30]
                )
                return False

        return True

    def _normalize_text_single(self, text: str) -> str:
        """
        Normaliza una cadena de texto para facilitar comparaciones.

        Args:
            text: La cadena de texto a normalizar.

        Returns:
            El texto normalizado.
        """
        if not isinstance(text, str):
            text = str(text)

        # Remover acentos
        text = unidecode(text.lower().strip())

        # Mantener solo caracteres permitidos
        text = re.sub(r"[^a-z0-9\s#\-]", "", text)

        # Normalizar espacios
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _emergency_dia_units_patch(self):
        """
        Aplica correcciones de unidades para casos conocidos.
        """
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
        """
        Actualiza las estadísticas de procesamiento.

        Args:
            record: El registro procesado.
        """
        self.stats["total_records"] += 1
        self.stats[f"cat_{record.categoria}"] += 1
        self.stats[f"tipo_{record.tipo_insumo}"] += 1
        self.stats[f"fmt_{record.formato_origen}"] += 1
        self.stats[f"unit_{record.unidad_apu}"] += 1

    def _log_stats(self):
        """Registra las estadísticas finales del proceso."""
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE PROCESAMIENTO")
        logger.info("=" * 60)
        logger.info(f"✅ Registros procesados: {len(self.processed_data)}")
        logger.info(f"❌ Errores: {self.stats.get('errores', 0)}")
        logger.info(f"🗑️  Descartados: {self.stats.get('registros_descartados', 0)}")
        logger.info(
            f"⛔ Rechazados validación: {self.stats.get('rechazados_validacion', 0)}"
        )
        logger.info(
            f"🚫 Excluidos por término: {self.stats.get('excluidos_por_termino', 0)}"
        )
        logger.info("-" * 60)

        # Estadísticas por tipo de insumo
        tipos = [k for k in self.stats.keys() if k.startswith('tipo_')]
        if tipos:
            logger.info("📋 Por tipo de insumo:")
            for tipo in sorted(tipos):
                count = self.stats[tipo]
                percentage = (
                    (count / len(self.processed_data) * 100)
                    if self.processed_data else 0
                )
                logger.info(
                    "  %-25s %6d (%5.1f%%)",
                    tipo.replace('tipo_', ''), count, percentage
                )

        # Estadísticas por formato
        formatos = [k for k in self.stats.keys() if k.startswith('fmt_')]
        if formatos:
            logger.info("-" * 60)
            logger.info("📄 Por formato de origen:")
            for fmt in sorted(formatos):
                logger.info(f"  {fmt.replace('fmt_', ''):.<25} {self.stats[fmt]:>6}")

        logger.info("=" * 60)

    def _build_dataframe(self) -> pd.DataFrame:
        """
        Construye un DataFrame de pandas a partir de los datos procesados.

        Returns:
            DataFrame con los datos procesados y validados.
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
        Valida la calidad del DataFrame generado.

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
                    logger.warning(
                        "⚠️ %d valores en cero en %s (%.1f%%)",
                        zero_count, col, percentage
                    )


def calculate_unit_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los costos unitarios por APU sumando los valores por tipo de insumo.

    INSTALACION = MANO_DE_OBRA + EQUIPO

    Args:
        df: DataFrame procesado con columna TIPO_INSUMO.

    Returns:
        DataFrame con costos unitarios por APU.
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
    pivot['VALOR_INSTALACION_UN'] = (
        pivot.get('MANO_DE_OBRA', 0) + pivot.get('EQUIPO', 0)
    )
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
    pivot['PCT_SUMINISTRO'] = (
        pivot['VALOR_SUMINISTRO_UN'] / total * 100
    ).fillna(0).round(2)
    pivot['PCT_INSTALACION'] = (
        pivot['VALOR_INSTALACION_UN'] / total * 100
    ).fillna(0).round(2)
    pivot['PCT_TRANSPORTE'] = (
        pivot['VALOR_TRANSPORTE_UN'] / total * 100
    ).fillna(0).round(2)

    # Ordenar columnas
    column_order = [
        'CODIGO_APU',
        'DESCRIPCION_APU',
        'UNIDAD_APU',
        'VALOR_SUMINISTRO_UN',
        'VALOR_INSTALACION_UN',
        'VALOR_TRANSPORTE_UN',
        'VALOR_OTRO_UN',
        'COSTO_UNITARIO_TOTAL',
        'PCT_SUMINISTRO',
        'PCT_INSTALACION',
        'PCT_TRANSPORTE',
    ]

    existing_cols = [col for col in column_order if col in pivot.columns]
    remaining_cols = [col for col in pivot.columns if col not in existing_cols]
    pivot = pivot[existing_cols + remaining_cols]

    # Log de resumen (🔥 CORRECCIÓN DEL BUG TIPOGRÁFICO)
    logger.info(f"✅ Costos unitarios calculados para {len(pivot)} APUs")
    logger.info(
        f"   Valor total suministros: ${pivot['VALOR_SUMINISTRO_UN'].sum():,.2f}"
    )
    logger.info(
        f"   Valor total instalación: ${pivot['VALOR_INSTALACION_UN'].sum():,.2f}"
    )
    logger.info(
        f"   Valor total transporte: ${pivot['VALOR_TRANSPORTE_UN'].sum():,.2f}"
    )
    logger.info(f"   Costo total: ${pivot['COSTO_UNITARIO_TOTAL'].sum():,.2f}")

    return pivot
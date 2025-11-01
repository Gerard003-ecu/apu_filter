import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from unidecode import unidecode

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

    INSTALACION_KEYWORDS = [
        "INSTALACION", "INSTALADO", "MONTAJE", "COLOCACION", "ARMADO",
        "CONSTRUCCION", "EXCAVACION", "RELLENO", "FUNDIDA", "VACIADO",
        "ACABADO", "ENCHAPE", "PAÑETE", "ESTUCO"
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
        self.processed_data: List[Dict[str, Any]] = []
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
                logger.debug(f"Registro problemático: {record}")
                self.stats["errores"] += 1

        self._emergency_dia_units_patch()
        self._log_stats()
        return self._build_dataframe()

    def _process_single_record(self, record: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        🆕 FLUJO CORREGIDO con manejo especial para MO
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

        # 4. Clasificar tipo de insumo PRIMERO
        tipo_insumo = self._classify_insumo(descripcion_insumo)

        # 5. Inferir unidad de APU si es "UND"
        if apu_unit == "UND":
            apu_unit = self._infer_unit_aggressive(apu_desc, category, apu_code)

        # 6. Convertir a números con validación mejorada
        cantidad = parse_number(parsed.get("cantidad", "0"))
        precio_unit = parse_number(parsed.get("precio_unit", "0"))
        valor_total = parse_number(parsed.get("valor_total", "0"))
        rendimiento = parse_number(parsed.get("rendimiento", "0"))

        # 🔥 CORRECCIÓN CRÍTICA: Aplicar lógica especial para MO
        cantidad, precio_unit, valor_total = self._fix_numeric_values(
            cantidad, precio_unit, valor_total, tipo_insumo # ← Pasar el tipo
        )

        # 6. Para MO, calcular rendimiento si falta
        if tipo_insumo == "MANO_DE_OBRA" and rendimiento == 0:
            if cantidad > 0:
                rendimiento = 1.0 / cantidad
                logger.debug(f"Rendimiento MO calculado: {rendimiento} = 1 / {cantidad}")

        # 9. Validar antes de agregar
        if not self._should_add_insumo(descripcion_insumo, cantidad, valor_total, tipo_insumo):
            self.stats["rechazados_validacion"] += 1
            return None

        # 10. Normalizar descripción
        normalized_desc = self._normalize_text_single(descripcion_insumo)

        return {
            "CODIGO_APU": apu_code,
            "DESCRIPCION_APU": apu_desc,
            "UNIDAD_APU": apu_unit,
            "DESCRIPCION_INSUMO": descripcion_insumo,
            "UNIDAD_INSUMO": parsed.get("unidad", "UND"),
            "CANTIDAD_APU": round(cantidad, 6),
            "PRECIO_UNIT_APU": round(precio_unit, 2),
            "VALOR_TOTAL_APU": round(valor_total, 2),
            "CATEGORIA": category,
            "TIPO_INSUMO": tipo_insumo,  # ← NUEVO CAMPO CRÍTICO
            "RENDIMIENTO": round(rendimiento, 6),
            "FORMATO_ORIGEN": parsed.get("formato", "GENERIC"),
            "NORMALIZED_DESC": normalized_desc,
        }

    def _classify_insumo(self, descripcion: str) -> str:
        """
        🆕 CLASIFICACIÓN MEJORADA:
        - MANO_DE_OBRA y EQUIPO contribuyen a INSTALACIÓN
        - No buscar "INSTALACION" en descripciones de insumos individuales
        """
        desc_upper = descripcion.upper()

        # Prioridad: MO > Equipo > Transporte > Suministro
        if any(kw in desc_upper for kw in self.MANO_OBRA_KEYWORDS):
            return "MANO_DE_OBRA"

        if any(kw in desc_upper for kw in self.EQUIPO_KEYWORDS):
            return "EQUIPO"

        if any(kw in desc_upper for kw in self.TRANSPORTE_KEYWORDS):
            return "TRANSPORTE"

        if any(kw in desc_upper for kw in self.SUMINISTRO_KEYWORDS):
            return "SUMINISTRO"

        # 🔥 NUNCA clasificar como "INSTALACION" - eso se calcula en agregaciones
        logger.debug(f"Insumo sin clasificar: {descripcion[:50]}")
        return "OTRO"

    def _infer_type_by_unit(self, desc_upper: str) -> Optional[str]:
        """
        🆕 Infiere el tipo de insumo por patrones en la descripción.
        Args:
            desc_upper: Descripción en mayúsculas.
        Returns:
            Tipo inferido o None.
        """
        # Patrones de unidad en la descripción
        if re.search(r'\bML\b|\bM\b|\bMTS\b', desc_upper):
            return "SUMINISTRO"  # Tubería, cable, etc.
        if re.search(r'\bUN\b|\bUND\b', desc_upper):
            return "SUMINISTRO"  # Accesorios, válvulas
        if re.search(r'\bKG\b|\bTON\b|\bM3\b', desc_upper):
            return "SUMINISTRO"  # Materiales a granel
        return None

    def _fix_numeric_values(
        self, cantidad: float, precio_unit: float, valor_total: float, tipo_insumo: str
    ) -> tuple[float, float, float]:
        """
        🆕 Corrige valores numéricos considerando el tipo de insumo.
        PARA MANO DE OBRA: cantidad = 1 / rendimiento (precio_unit)
        """
        # Para mano de obra, el precio_unit es realmente el RENDIMIENTO
        if tipo_insumo == "MANO_DE_OBRA":
            if precio_unit > 0 and cantidad == 0:
                cantidad = 1.0 / precio_unit
                logger.debug(f"MO: Cantidad calculada como 1/rendimiento: {cantidad} = 1 / {precio_unit}")
            elif cantidad > 0 and precio_unit == 0:
                precio_unit = 1.0 / cantidad
                logger.debug(f"MO: Rendimiento calculado como 1/cantidad: {precio_unit} = 1 / {cantidad}")

            # Recalcular valor_total si es necesario
            if valor_total == 0 and cantidad > 0:
                # Necesitamos el jornal_real, que debería venir de otro campo
                # Por ahora, usamos una estimación conservadora
                jornal_estimado = precio_unit * 1000 # Factor estimado
                valor_total = cantidad * jornal_estimado
                logger.debug(f"MO: Valor total estimado: {valor_total}")

        # Lógica original para otros tipos
        else:
            if cantidad == 0 and valor_total > 0 and precio_unit > 0:
                cantidad = valor_total / precio_unit
            elif valor_total == 0 and cantidad > 0 and precio_unit > 0:
                valor_total = cantidad * precio_unit
            elif precio_unit == 0 and cantidad > 0 and valor_total > 0:
                precio_unit = valor_total / cantidad

        return cantidad, precio_unit, valor_total

    def _parse_insumo_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parsea una línea de insumo textual usando una serie de patrones regex.
        MEJORADO con más logging y manejo de errores.

        Args:
            line: La cadena de texto que representa la línea de insumo.

        Returns:
            Un diccionario con los campos parseados o None si no se puede parsear.
        """
        if not line or not line.strip():
            return None

        line = line.strip()
        patterns = self._get_parsing_patterns()

        for name, pattern in patterns.items():
            match = pattern.match(line)
            if match:
                data = match.groupdict()
                data["formato"] = name
                logger.debug(f"✓ Parseado con patrón {name}: {line[:60]}")
                return data

        # Fallback mejorado: dividir por ; y asignar genéricamente
        parts = [p.strip() for p in line.split(";")]

        if len(parts) >= 6:
            result = {
                "descripcion": parts[0],
                "unidad": parts[1] if parts[1] else "UND",
                "cantidad": parts[2] if parts[2] else "0",
                "precio_unit": parts[4] if len(parts) > 4 else "0",
                "valor_total": parts[5] if len(parts) > 5 else "0",
                "rendimiento": "0",
                "formato": "FALLBACK",
            }
            logger.debug(f"⚠ Fallback aplicado: {line[:60]}")
            return result

        # Si tiene menos de 6 partes, intentar rescatar algo
        if len(parts) >= 3:
            return {
                "descripcion": parts[0],
                "unidad": parts[1] if len(parts) > 1 else "UND",
                "cantidad": parts[2] if len(parts) > 2 else "0",
                "precio_unit": "0",
                "valor_total": "0",
                "rendimiento": "0",
                "formato": "MINIMAL",
            }

        logger.warning(f"❌ No se pudo parsear: {line[:100]}")
        return None

    def _get_parsing_patterns(self) -> Dict[str, re.Pattern]:
        """
        🆕 PATRONES MEJORADOS para distinguir jornal vs rendimiento
        """
        return {
            "MO_COMPLETA": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|M O |MANO DE OBRA).*?);"
                r"\s*(?P<jornal_base>[\d.,\s]+);\s*"
                r"(?P<prestaciones>[\d%.,\s]+);\s*"
                r"(?P<jornal_total>[\d.,\s]+);\s*"
                r"(?P<rendimiento>[\d.,\s]+);\s*" # ← ESTE es el rendimiento
                r"(?P<valor_total>[\d.,\s]+)",
                re.IGNORECASE,
            ),
            "MO_SIMPLE_CORREGIDO": re.compile(
                r"^(?P<descripcion>(?:M\.O\.|M O |MANO DE OBRA).*?);"
                r"\s*[^;]*;\s*" # Campo intermedio
                r"(?P<rendimiento>[^;]*);\s*" # ← RENDIMIENTO identificado
                r"[^;]*;\s*"
                r"(?P<valor_total>[^;]*)",
                re.IGNORECASE,
            ),
            "INSUMO_FULL": re.compile(
                r"^(?P<descripcion>[^;]+);\s*"
                r"(?P<unidad>[^;]*);\s*"
                r"(?P<cantidad>[^;]*);\s*"
                r"(?P<desperdicio>[^;]*);\s*"
                r"(?P<precio_unit>[^;]*);\s*"
                r"(?P<valor_total>[^;]*)",
                re.IGNORECASE,
            ),
        }

    def _infer_unit_aggressive(self, description: str, category: str, apu_code: str) -> str:
        """
        Infiere la unidad de un APU cuando esta es 'UND' (indefinida).
        MEJORADO con más patrones.

        Args:
            description: La descripción del APU.
            category: La categoría del APU.
            apu_code: El código del APU.

        Returns:
            La unidad inferida como una cadena de texto.
        """
        desc_upper = description.upper()

        # Por keywords en descripción
        unit_keywords = {
            "M3": ["EXCAVACION", "CONCRETO", "RELLENO", "M3", "METROS CUBICOS"],
            "M2": ["PINTURA", "LOSA", "ENCHAPE", "PAÑETE", "M2", "METROS CUADRADOS"],
            "ML": ["TUBERIA", "CANAL", "ML", "METROS LINEALES", "TUBO"],
            "DIA": ["CUADRILLA", "EQUIPO", "DIA", "DIAS"],
            "JOR": ["MANO DE OBRA", "JORNAL", "PERSONAL"],
            "UN": ["ACCESORIO", "VALVULA", "UNIDAD", "PIEZA"],
        }

        for unit, words in unit_keywords.items():
            if any(w in desc_upper for w in words):
                logger.debug(f"Unidad inferida {unit} para: {description[:50]}")
                return unit

        # Por categoría
        cat_map = {
            "MANO DE OBRA": "JOR",
            "EQUIPO": "DIA",
            "TRANSPORTE": "VIAJE",
            "MATERIALES": "UN",
        }
        if category in cat_map:
            return cat_map[category]

        return "UND"

    def _clean_unit(self, unit: str) -> str:
        """
        Normaliza y limpia una cadena de texto de unidad.
        MEJORADO con más sinónimos.

        Args:
            unit: La cadena de texto de la unidad a limpiar.

        Returns:
            La unidad normalizada.
        """
        if not unit or not unit.strip():
            return "UND"

        unit = re.sub(r"[^A-Z0-9]", "", unit.upper().strip())

        mapping = {
            "DIAS": "DIA",
            "DÍAS": "DIA",
            "UN": "UND",
            "UNIDAD": "UND",
            "UNIDADES": "UND",
            "JORNAL": "JOR",
            "JORNALES": "JOR",
            "VIAJES": "VIAJE",
            "METRO": "M",
            "METROS": "M",
            "MTS": "M",
            "GLN": "GAL",
            "GALON": "GAL",
            "KILOGRAMO": "KG",
            "TONELADA": "TON",
        }

        return mapping.get(unit, unit)

    def _is_excluded_insumo(self, desc: str) -> bool:
        """
        Verifica si un insumo debe ser excluido basado en su descripción.
        MEJORADO con regex para evitar falsos positivos.

        Args:
            desc: La descripción del insumo.

        Returns:
            True si el insumo debe ser excluido, False en caso contrario.
        """
        if not desc or len(desc.strip()) < 2:
            return True

        desc_u = desc.upper()

        # Excluir si el término aparece en la descripción
        for term in self.EXCLUDED_TERMS:
            if term in desc_u:
                logger.debug(f"Excluido por término '{term}': {desc[:50]}")
                return True

        return False

    def _looks_like_mo(self, desc: str) -> bool:
        """
        Determina si una descripción de insumo parece ser mano de obra.
        DEPRECATED: Usar _classify_insumo en su lugar.

        Args:
            desc: La descripción del insumo.

        Returns:
            True si la descripción contiene términos de mano de obra.
        """
        return any(term in desc.upper() for term in self.MANO_OBRA_KEYWORDS)

    def _calculate_rendimiento_simple(
        self, cantidad: float, precio_unit: float, valor_total: float
    ) -> float:
        """
        Calcula el rendimiento de mano de obra usando la fórmula correcta.
        El rendimiento es el inverso de la cantidad.

        Args:
            cantidad: Cantidad del insumo (ej: JORNAL/UNIDAD).
            precio_unit: El precio unitario del insumo.
            valor_total: El valor total del insumo.

        Returns:
            El valor del rendimiento calculado (ej: UNIDAD/JORNAL).
        """
        # El rendimiento es el inverso de la cantidad.
        if cantidad > 0:
            return 1.0 / cantidad

        return 0.0

    def _should_add_insumo(self, desc: str, cantidad: float, valor_total: float, tipo_insumo: str) -> bool:
        """
        🆕 VALIDACIÓN MEJORADA con umbrales por tipo de insumo
        """
        # Descripción válida
        if not desc or len(desc.strip()) < 3:
            logger.debug("Rechazado: descripción muy corta")
            return False

        # Umbrales por tipo de insumo
        umbrales = {
            "MANO_DE_OBRA": {"cantidad": 1000, "valor": 999_999},
            "EQUIPO": {"cantidad": 10000, "valor": 999_999},
            "SUMINISTRO": {"cantidad": 100000, "valor": 999_999},
            "OTRO": {"cantidad": 1000000, "valor": 999_999_999}
        }

        umbral = umbrales.get(tipo_insumo, umbrales["OTRO"])

        # Evitar valores absurdos
        if valor_total > umbral["valor"]:
            logger.warning(f"Rechazado: valor total absurdo {valor_total} para {tipo_insumo}")
            return False

        if cantidad > umbral["cantidad"]:
            logger.warning(f"Rechazado: cantidad absurda {cantidad} para {tipo_insumo}")
            return False

        # Debe tener al menos cantidad o valor
        if cantidad <= 0 and valor_total <= 0:
            logger.debug("Rechazado: sin cantidad ni valor")
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

        text = unidecode(text.lower().strip())
        text = re.sub(r"[^a-z0-9\s#\-]", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _emergency_dia_units_patch(self):
        """
        Aplica un parche para corregir la unidad de cuadrillas a 'DIA'.
        """
        squad_codes = ['13', '14', '15', '16', '17', '18', '19', '20']
        patched = 0

        for apu in self.processed_data:
            apu_code = apu["CODIGO_APU"]
            tipo_insumo = apu.get("TIPO_INSUMO", "")

            # Corregir cuadrillas
            if (tipo_insumo == "MANO_DE_OBRA" and
                any(apu_code.startswith(c) for c in squad_codes) and
                apu["UNIDAD_APU"] != "DIA"):
                old = apu["UNIDAD_APU"]
                apu["UNIDAD_APU"] = "DIA"
                logger.info(f"🚀 PARCHE DIA: {apu_code} '{old}' → 'DIA'")
                patched += 1

        if patched > 0:
            logger.info(f"✅ Aplicados {patched} parches de unidad DIA")

    def _update_stats(self, record: Dict):
        """
        Actualiza las estadísticas de procesamiento con un nuevo registro.

        Args:
            record: El registro procesado.
        """
        self.stats["total_records"] += 1
        self.stats[f"cat_{record.get('CATEGORIA', 'SIN_CAT')}"] += 1
        self.stats[f"tipo_{record.get('TIPO_INSUMO', 'SIN_TIPO')}"] += 1
        self.stats[f"fmt_{record.get('FORMATO_ORIGEN', 'UNKNOWN')}"] += 1

    def _log_stats(self):
        """Registra las estadísticas finales del proceso en el log."""
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE PROCESAMIENTO")
        logger.info("=" * 60)
        logger.info(f"✅ Registros procesados: {len(self.processed_data)}")
        logger.info(f"❌ Errores: {self.stats.get('errores', 0)}")
        logger.info(
            f"🗑️  Descartados: {self.stats.get('registros_descartados', 0)}"
        )
        logger.info(
            f"⛔ Rechazados validación: {self.stats.get('rechazados_validacion', 0)}"
        )
        logger.info(
            f"🚫 Excluidos por término: {self.stats.get('excluidos_por_termino', 0)}"
        )
        logger.info("-" * 60)

        # Estadísticas por tipo
        tipos = [k for k in self.stats.keys() if k.startswith('tipo_')]
        if tipos:
            logger.info("Por tipo de insumo:")
            for tipo in sorted(tipos):
                logger.info(f"  {tipo.replace('tipo_', '')}: {self.stats[tipo]}")

        logger.info("=" * 60)

    def _build_dataframe(self) -> pd.DataFrame:
        """
        Construye DataFrame validando correctamente los tipos.
        """
        if not self.processed_data:
            logger.warning("⚠️ No hay datos procesados para construir DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(self.processed_data)

        # Validar que existan los tipos esperados
        if 'TIPO_INSUMO' in df.columns:
            tipos_count = df['TIPO_INSUMO'].value_counts()
            logger.info("📋 Distribución de tipos de insumo:")
            for tipo, count in tipos_count.items():
                logger.info(f" {tipo}: {count}")

            # 🔥 CORRECCIÓN: No buscar "INSTALACION" en insumos individuales
            # La instalación es una SUMA de MANO_DE_OBRA + EQUIPO
            if 'SUMINISTRO' not in tipos_count:
                logger.error("🚨 NO SE ENCONTRARON INSUMOS DE SUMINISTRO")

            # Verificar componentes de instalación
            has_mo = 'MANO_DE_OBRA' in tipos_count
            has_equipo = 'EQUIPO' in tipos_count
            if not has_mo and not has_equipo:
                logger.error("🚨 NO SE ENCONTRARON COMPONENTES DE INSTALACIÓN (MO o EQUIPO)")
            elif not has_mo:
                logger.warning("⚠️ NO SE ENCONTRÓ MANO DE OBRA para instalación")
            elif not has_equipo:
                logger.warning("⚠️ NO SE ENCONTRÓ EQUIPO para instalación")
            else:
                logger.info("✅ Componentes de instalación encontrados: MO + EQUIPO")

        return df


# 🆕 FUNCIÓN AUXILIAR PARA CALCULAR COSTOS UNITARIOS
def calculate_unit_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los costos unitarios por APU sumando los valores por tipo de insumo.
    Args:
        df: DataFrame procesado con columna TIPO_INSUMO.
    Returns:
        DataFrame con costos unitarios por APU.
    """
    if df.empty or 'TIPO_INSUMO' not in df.columns:
        logger.error("DataFrame inválido para cálculo de costos unitarios")
        return pd.DataFrame()

    # Agrupar por APU y tipo
    group_cols = ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU', 'TIPO_INSUMO']
    grouped = df.groupby(group_cols)['VALOR_TOTAL_APU'].sum().reset_index()

    # Pivotar para tener una columna por tipo
    pivot = grouped.pivot_table(
        index=['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU'],
        columns='TIPO_INSUMO',
        values='VALOR_TOTAL_APU',
        fill_value=0,
        aggfunc='sum'
    ).reset_index()

    # Asegurar que existan todas las columnas
    for col in ['SUMINISTRO', 'INSTALACION', 'MANO_DE_OBRA', 'EQUIPO', 'TRANSPORTE']:
        if col not in pivot.columns:
            pivot[col] = 0

    # Calcular totales
    pivot['VALOR_SUMINISTRO_UN'] = pivot.get('SUMINISTRO', 0)
    pivot['VALOR_INSTALACION_UN'] = (
        pivot.get('INSTALACION', 0) + pivot.get('MANO_DE_OBRA', 0)
    )
    non_total_cols = ['CODIGO_APU', 'DESCRIPCION_APU', 'UNIDAD_APU']
    total_cols = [col for col in pivot.columns if col not in non_total_cols]
    pivot['COSTO_UNITARIO_TOTAL'] = pivot[total_cols].sum(axis=1)

    logger.info(f"✅ Costos unitarios calculados para {len(pivot)} APUs")

    return pivot

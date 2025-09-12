import re

import numpy as np
import pandas as pd


def normalize_text(series):
    """Limpia y estandariza texto para hacer un 'join' fiable."""
    return series.astype(str).str.lower().str.strip()


def find_and_rename_columns(df, column_map):
    """
    Busca columnas que contengan un texto clave y las renombra.
    """
    rename_dict = {}
    df.columns = df.columns.str.strip()

    for new_name, keyword in column_map.items():
        found_col = next(
            (col for col in df.columns if keyword.lower() in col.lower()), None
        )
        if found_col:
            rename_dict[found_col] = new_name

    df = df.rename(columns=rename_dict)

    if not all(new_name in df.columns for new_name in column_map.keys()):
        missing = [name for name in column_map.keys() if name not in df.columns]
        print(
            f"ADVERTENCIA: No se encontraron las siguientes columnas clave: {missing}"
        )

    return df


def process_presupuesto_csv(path):
    """Lee y limpia el archivo presupuesto.csv de forma robusta."""
    try:
        df = pd.read_csv(path, delimiter=";", encoding="latin1", skipinitialspace=True)

        column_map = {
            "CODIGO_APU": "ITEM",
            "DESCRIPCION_APU": "DESCRIPCION",
            "CANTIDAD_PRESUPUESTO": "CANT.",
        }
        df = find_and_rename_columns(df, column_map)

        if "CODIGO_APU" not in df.columns:
            return pd.DataFrame()

        df["CODIGO_APU"] = df["CODIGO_APU"].astype(str).str.strip()
        df = df[df["CODIGO_APU"].str.contains(r",", na=False)]
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]
    except Exception as e:
        print(f"Error procesando presupuesto.csv: {e}")
        return pd.DataFrame()


def process_insumos_csv(path):
    """
    Lee y limpia el archivo insumos.csv, manejando correctamente los grupos
    mediante una lectura manual línea por línea.
    """
    data_rows = []
    current_group = "INDEFINIDO"
    header_found = False

    try:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                # Omitir líneas vacías
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(";")]

                # Detectar y actualizar el grupo actual
                if len(parts) > 1 and parts[0].upper().startswith("G") and parts[1]:
                    current_group = parts[1]
                    continue

                # Identificar la línea de encabezado principal para empezar a leer datos
                if "CODIGO" in line and "DESCRIPCION" in line:
                    header_found = True
                    # Extraer los nombres de las columnas de esta línea
                    header_columns = [p.strip() for p in line.split(";")]
                    desc_index = next(
                        (
                            i
                            for i, col in enumerate(header_columns)
                            if "DESCRIPCION" in col
                        ),
                        -1,
                    )
                    vr_unit_index = next(
                        (
                            i
                            for i, col in enumerate(header_columns)
                            if "VR. UNIT" in col
                        ),
                        -1,
                    )
                    continue

                # Si hemos encontrado el header, procesar las líneas de datos
                if header_found and desc_index != -1 and vr_unit_index != -1:
                    if len(parts) > max(desc_index, vr_unit_index):
                        description = parts[desc_index]
                        vr_unit_str = parts[vr_unit_index]

                        # Validar que sea una fila de datos real
                        if description and vr_unit_str:
                            data_rows.append(
                                {
                                    "DESCRIPCION_INSUMO": description,
                                    "VR_UNITARIO_INSUMO": vr_unit_str,
                                    "GRUPO_INSUMO": current_group,
                                }
                            )

        if not data_rows:
            return pd.DataFrame()

        df = pd.DataFrame(data_rows)

        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df.dropna(subset=["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"], inplace=True)

        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        df = df.drop_duplicates(subset=["NORMALIZED_DESC"], keep="first")

        return df[
            [
                "NORMALIZED_DESC",
                "VR_UNITARIO_INSUMO",
                "GRUPO_INSUMO",
                "DESCRIPCION_INSUMO",
            ]
        ]
    except Exception as e:
        print(f"Error procesando insumos.csv: {e}")
        return pd.DataFrame()


def process_apus_csv(path):
    """Parsea manualmente el archivo apus.csv que tiene formato de reporte."""
    apus_data, current_apu_code, current_category = [], None, "INDEFINIDO"
    category_keywords = {
        "MATERIALES": "MATERIALES",
        "MANO DE OBRA": "MANO DE OBRA",
        "EQUIPO": "EQUIPO",
        "OTROS": "OTROS",
    }

    def to_numeric_safe(s):
        if isinstance(s, (int, float)):
            return s
        if isinstance(s, str):
            s = s.replace(".", "").replace(",", ".").strip()
            return pd.to_numeric(s, errors="coerce")
        return pd.NA

    try:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                if "ITEM:" in line.upper():
                    match = re.search(r"ITEM:\s*([\d,]+\b)", line)
                    if match:
                        current_apu_code = match.group(1).strip()
                        current_category = "INDEFINIDO"
                    continue

                cleaned_line = line.replace(";", "").strip().upper()
                if cleaned_line in category_keywords:
                    current_category = category_keywords[cleaned_line]
                    continue

                if current_apu_code and ";" in line:
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) < 6 or not parts[0]:
                        continue

                    description = parts[0]

                    cantidad = to_numeric_safe(parts[2])
                    precio_unit = to_numeric_safe(parts[4])
                    valor_total = to_numeric_safe(parts[5])

                    # Handle special M.O. format
                    if pd.isna(cantidad) and "%" in parts[2]:
                        jornal_total = to_numeric_safe(parts[3])
                        if (
                            pd.notna(valor_total)
                            and pd.notna(jornal_total)
                            and jornal_total > 0
                        ):
                            cantidad = valor_total / jornal_total
                        else:
                            cantidad = 0
                        precio_unit = jornal_total

                    if pd.notna(valor_total):
                        # If quantity or price is missing, try to calculate it
                        if (
                            (pd.isna(cantidad) or cantidad == 0)
                            and pd.notna(precio_unit)
                            and precio_unit > 0
                        ):
                            cantidad = valor_total / precio_unit

                        if (
                            (pd.isna(precio_unit) or precio_unit == 0)
                            and pd.notna(cantidad)
                            and cantidad > 0
                        ):
                            precio_unit = valor_total / cantidad

                        apus_data.append(
                            {
                                "CODIGO_APU": current_apu_code,
                                "DESCRIPCION_INSUMO": description,
                                "CANTIDAD_APU": cantidad if pd.notna(cantidad) else 0,
                                "PRECIO_UNIT_APU": (
                                    precio_unit if pd.notna(precio_unit) else 0
                                ),
                                "VALOR_TOTAL_APU": valor_total,
                                "CATEGORIA": current_category,
                            }
                        )

        if not apus_data:
            return pd.DataFrame()

        df = pd.DataFrame(apus_data)
        if df.empty:
            return df

        df["CODIGO_APU"] = df["CODIGO_APU"].str.strip()
        df["NORMALIZED_DESC"] = normalize_text(df["DESCRIPCION_INSUMO"])
        return df
    except Exception as e:
        print(f"Error procesando apus.csv: {e}")
        return pd.DataFrame()


def group_and_split_description(df):
    """
    Agrupa inteligentemente las descripciones de los ítems del presupuesto.
    Extrae un nombre de grupo (ej. por calibre) y una descripción corta.
    """
    if "DESCRIPCION_APU" not in df.columns:
        df["grupo"] = "Ítems Varios"
        df["DESCRIPCION_APU"] = ""  # Asegurar que la columna existe
        return df

    def get_group_and_short_desc(desc):
        """
        Extrae un nombre de grupo y una descripción corta de una descripción completa.
        """
        if not isinstance(desc, str):
            return "Ítems Varios", ""

        # Patrón para calibre, ej. "REMATE CON PINTURA DE FABRICA CAL 22"
        # Es no-codicioso (.*?) para capturar el texto mínimo hasta el calibre.
        # re.IGNORECASE para capturar "cal", "CAL", "Cal", etc.
        cal_match = re.match(r"(.*? CAL\.? ?\d+)", desc, re.IGNORECASE)
        if cal_match:
            group = cal_match.group(1).strip()
            # La descripción corta es lo que queda después del match completo.
            short_desc = desc[len(cal_match.group(0)) :].strip()
            # Si la descripción corta empieza con conectores comunes, los quitamos.
            short_desc = re.sub(
                r"^(de|en|con|para)\s", "", short_desc, flags=re.IGNORECASE
            ).strip()
            return group, short_desc

        # Fallback: agrupar por las primeras 4 palabras.
        words = desc.split()
        if len(words) > 4:
            group = " ".join(words[:4])
            short_desc = " ".join(words[4:])
            return group, short_desc

        # Si la descripción es corta, se usa como grupo y no hay descripción corta.
        return desc, ""

    # Aplica la lógica para crear las columnas 'grupo' y 'descripcion_corta'
    df[["grupo", "descripcion_corta"]] = df["DESCRIPCION_APU"].apply(
        lambda x: pd.Series(get_group_and_short_desc(x))
    )

    # La descripción a mostrar en la tabla principal será la parte corta.
    df["DESCRIPCION_APU"] = df["descripcion_corta"]
    df.drop(columns=["descripcion_corta"], inplace=True)

    return df


def process_all_files(presupuesto_path, apus_path, insumos_path):
    """Orquesta el procesamiento y devuelve un diccionario de DataFrames."""
    print("--- Iniciando procesamiento ---")
    df_presupuesto = process_presupuesto_csv(presupuesto_path)
    df_insumos = process_insumos_csv(insumos_path)
    df_apus = process_apus_csv(apus_path)

    if df_presupuesto.empty or df_insumos.empty or df_apus.empty:
        return {"error": "Uno o más archivos no pudieron ser procesados."}

    df_merged = pd.merge(
        df_apus, df_insumos, on="NORMALIZED_DESC", how="left", suffixes=("_apu", "")
    )
    df_merged["DESCRIPCION_INSUMO"] = df_merged["DESCRIPCION_INSUMO_apu"]

    # --- Lógica de cálculo de costos refactorizada ---

    # El costo se calcula de dos maneras:
    # 1. Si el insumo existe en df_insumos, el costo es CANTIDAD * PRECIO_INSUMO.
    # 2. Si no existe, el costo es el VALOR_TOTAL extraído directamente de apus.csv.
    df_merged["COSTO_INSUMO_EN_APU"] = np.where(
        df_merged["VR_UNITARIO_INSUMO"].notna(),
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_INSUMO"],
        df_merged["VALOR_TOTAL_APU"],
    )

    # El precio unitario final para mostrar es el de insumos si existe.
    # Si no, es el precio unitario de apus.csv.
    # Si ambos faltan, se calcula a partir del costo total y la cantidad.
    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_INSUMO"].fillna(
        df_merged["PRECIO_UNIT_APU"]
    )
    # Recalcular si es necesario para que sea consistente
    # Evitar división por cero
    mask_recalc = (df_merged["VR_UNITARIO_FINAL"].isna()) | (
        df_merged["VR_UNITARIO_FINAL"] == 0
    )
    cantidad_safe = df_merged["CANTIDAD_APU"].replace(0, 1)
    df_merged.loc[mask_recalc, "VR_UNITARIO_FINAL"] = (
        df_merged.loc[mask_recalc, "COSTO_INSUMO_EN_APU"] / cantidad_safe
    )

    # Asegurarnos de que no haya NaNs en el costo final.
    df_merged["COSTO_INSUMO_EN_APU"] = df_merged["COSTO_INSUMO_EN_APU"].fillna(0)

    df_apu_costos_categoria = (
        df_merged.groupby(["CODIGO_APU", "CATEGORIA"])["COSTO_INSUMO_EN_APU"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    cost_cols = ["MATERIALES", "MANO DE OBRA", "EQUIPO", "OTROS"]
    for col in cost_cols:
        if col not in df_apu_costos_categoria.columns:
            df_apu_costos_categoria[col] = 0

    df_apu_costos_categoria["VALOR_SUMINISTRO_UN"] = df_apu_costos_categoria[
        "MATERIALES"
    ]
    df_apu_costos_categoria["VALOR_INSTALACION_UN"] = (
        df_apu_costos_categoria["MANO DE OBRA"] + df_apu_costos_categoria["EQUIPO"]
    )
    df_apu_costos_categoria["VALOR_CONSTRUCCION_UN"] = df_apu_costos_categoria[
        cost_cols
    ].sum(axis=1)

    df_tiempo = (
        df_merged[df_merged["CATEGORIA"] == "MANO DE OBRA"]
        .groupby("CODIGO_APU")["CANTIDAD_APU"]
        .sum()
        .reset_index()
    )
    df_tiempo.rename(columns={"CANTIDAD_APU": "TIEMPO_INSTALACION"}, inplace=True)

    df_final = pd.merge(
        df_presupuesto, df_apu_costos_categoria, on="CODIGO_APU", how="left"
    )
    df_final = pd.merge(df_final, df_tiempo, on="CODIGO_APU", how="left")

    final_cols_to_fill = [
        "VALOR_SUMINISTRO_UN",
        "VALOR_INSTALACION_UN",
        "VALOR_CONSTRUCCION_UN",
        "TIEMPO_INSTALACION",
    ]
    for col in final_cols_to_fill:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
        else:
            df_final[col] = 0

    # Agrupa las descripciones antes de enviar los datos
    df_final = group_and_split_description(df_final)

    print(
        f"Diagnóstico: {df_final['VALOR_CONSTRUCCION_UN'].notna().sum()}"
        f" de {len(df_final)} ítems del presupuesto encontraron su costo."
    )

    df_merged["Vr Unitario"] = df_merged["VR_UNITARIO_FINAL"].fillna(0)
    df_merged["Vr Total"] = df_merged["COSTO_INSUMO_EN_APU"]
    df_merged_renamed = df_merged.rename(
        columns={"DESCRIPCION_INSUMO": "Descripción", "CANTIDAD_APU": "Cantidad"}
    )
    apus_detail = {
        n: g[
            ["Descripción", "Cantidad", "Vr Unitario", "Vr Total", "CATEGORIA"]
        ].to_dict("records")
        for n, g in df_merged_renamed.groupby("CODIGO_APU")
    }

    insumos_dict = {}
    if "GRUPO_INSUMO" in df_insumos.columns:
        df_insumos_renamed = df_insumos.rename(
            columns={
                "DESCRIPCION_INSUMO": "Descripción",
                "VR_UNITARIO_INSUMO": "Vr Unitario",
            }
        )
        for name, group in df_insumos_renamed.groupby("GRUPO_INSUMO"):
            if name and isinstance(name, str):
                insumos_dict[name.strip()] = (
                    group[["Descripción", "Vr Unitario"]].dropna().to_dict("records")
                )

    print("--- Procesamiento completado ---")

    return {
        "presupuesto": df_final.to_dict("records"),
        "insumos": insumos_dict,
        "apus_detail": apus_detail,
    }


def calculate_estimate(params, data_store):
    """
    Calcula una estimación de costo basada en parámetros de entrada.
    Esta es la función principal para la lógica del 'Estimador Rápido'.
    """
    log = []

    # --- 1. Extraer datos y parámetros ---
    presupuesto_list = data_store.get("presupuesto", [])
    apus_detail = data_store.get("apus_detail", {})

    tipo = params.get("tipo", "").lower()
    material = params.get("material", "").lower()
    cuadrilla = params.get("cuadrilla", "0")
    izaje = params.get("izaje", "manual").lower()
    seguridad = params.get("seguridad", "normal").lower()
    zona = params.get("zona", "zona 0").lower()
    log.append(f"Parámetros: {params}")

    # --- 2. Encontrar el APU de Mano de Obra principal ---
    apu_mo_code = None
    apu_mo_desc = ""

    # Construir términos de búsqueda a partir de los parámetros
    search_terms = [material]
    if tipo == "cubierta":
        # Para cubiertas, el tipo de material (ej. TST) es más descriptivo
        # a menudo que la propia palabra "cubierta".
        pass
    elif tipo == "fachada":
        search_terms.append("fachada")

    # La descripción de la cuadrilla es un identificador clave
    search_terms.append(f"cuadrilla de {cuadrilla}")

    for item in presupuesto_list:
        desc_lower = item.get("DESCRIPCION_APU", "").lower()
        # Usamos 'all' para asegurar que todos los términos de búsqueda estén presentes
        if all(term in desc_lower for term in search_terms):
            # Priorizar APUs que contengan 'mano de obra' para evitar elegir un APU de suministro
            if "mano de obra" in desc_lower:
                apu_mo_code = item.get("CODIGO_APU")
                apu_mo_desc = item.get("DESCRIPCION_APU")
                log.append(f"APU de M.O. encontrado: {apu_mo_desc} (Código: {apu_mo_code})")
                break

    if not apu_mo_code:
        log.append(f"Error: No se encontró APU para los términos: {search_terms}")
        return {"error": f"No se encontró un APU de mano de obra que coincida con los criterios: {params}", "log": log}

    # --- 3. Calcular costos y tiempo base del APU encontrado ---
    costos_base = {"MATERIALES": 0, "MANO DE OBRA": 0, "EQUIPO": 0, "OTROS": 0}
    tiempo_instalacion = 0

    apu_items = apus_detail.get(apu_mo_code, [])
    if not apu_items:
        return {"error": f"El código APU {apu_mo_code} no tiene detalle.", "log": log}

    for insumo in apu_items:
        categoria = insumo.get("CATEGORIA", "OTROS")
        costo = insumo.get("Vr Total", 0)
        costos_base[categoria] += costo
        if categoria == "MANO DE OBRA":
            tiempo_instalacion += insumo.get("Cantidad", 0)

    valor_suministro = costos_base["MATERIALES"]
    valor_instalacion = costos_base["MANO DE OBRA"] + costos_base["EQUIPO"]
    log.append(f"Costos base calculados: Suministro=${valor_suministro:,.0f}, Instalación=${valor_instalacion:,.0f}")
    log.append(f"Tiempo base de instalación: {tiempo_instalacion:.4f} días/un")

    # --- 4. Aplicar Factores de Ajuste ---

    # a) Ajuste por Izaje (Grúa)
    if izaje == "grúa":
        costo_grua_por_dia = 0
        # Buscar el APU de la grúa
        for item in presupuesto_list:
            if "alquiler grua" in item.get("DESCRIPCION_APU", "").lower():
                codigo_grua = item.get("CODIGO_APU")
                detalle_grua = apus_detail.get(codigo_grua, [])
                costo_total_apu_grua = sum(d.get("Vr Total", 0) for d in detalle_grua)
                # Asumimos que el APU de la grúa está definido por día
                costo_grua_por_dia = costo_total_apu_grua
                log.append(f"APU de Grúa encontrado. Costo por día: ${costo_grua_por_dia:,.0f}")
                break

        if costo_grua_por_dia > 0:
            costo_adicional_grua = costo_grua_por_dia * tiempo_instalacion
            valor_instalacion += costo_adicional_grua
            log.append(f"Costo adicional por grúa ({tiempo_instalacion:.4f} días): ${costo_adicional_grua:,.0f}")
        else:
            log.append("ADVERTENCIA: Izaje por grúa seleccionado, pero no se encontró APU de grúa.")

    # b) Ajuste por Exigencia de Seguridad
    if seguridad == "alta":
        costo_mo = costos_base["MANO DE OBRA"]
        incremento_seguridad = costo_mo * 0.15  # 15% de incremento sobre la mano de obra
        valor_instalacion += incremento_seguridad
        log.append(f"Incremento por seguridad alta (15% de M.O.): ${incremento_seguridad:,.0f}")

    # c) Ajuste por Zona Geográfica
    zona_factor = {"zona 0": 1.0, "zona 1": 1.05, "zona 2": 1.10, "zona 3": 1.15}
    factor = zona_factor.get(zona, 1.0)
    if factor > 1.0:
        costo_original_instalacion = valor_instalacion
        valor_instalacion *= factor
        incremento_zona = valor_instalacion - costo_original_instalacion
        log.append(f"Incremento por {zona} ({(factor-1)*100:.0f}%): ${incremento_zona:,.0f}")

    # --- 5. Devolver Resultados ---
    log.append(f"Valores finales: Suministro=${valor_suministro:,.0f}, Instalación=${valor_instalacion:,.0f}")

    return {
        "valor_suministro": valor_suministro,
        "valor_instalacion": valor_instalacion,
        "tiempo_instalacion": tiempo_instalacion,
        "apu_encontrado": apu_mo_desc,
        "log": "\n".join(log)
    }

import re

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

    df_merged["VR_UNITARIO_FINAL"] = df_merged["VR_UNITARIO_INSUMO"].fillna(
        df_merged["PRECIO_UNIT_APU"]
    )
    mask_no_price = df_merged["VR_UNITARIO_FINAL"].isna()
    df_merged.loc[mask_no_price, "VR_UNITARIO_FINAL"] = df_merged.loc[
        mask_no_price, "VALOR_TOTAL_APU"
    ]
    df_merged.loc[mask_no_price, "CANTIDAD_APU"] = 1
    df_merged["COSTO_INSUMO_EN_APU"] = (
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_FINAL"].fillna(0)
    )

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

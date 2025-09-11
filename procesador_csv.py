import re

import pandas as pd


def process_presupuesto_csv(path):
    """Lee y limpia el archivo presupuesto.csv."""
    try:
        df = pd.read_csv(path, delimiter=";", encoding="latin1")
        df.columns = (
            df.columns.str.strip()
        )  # Limpia espacios en los nombres de columnas
        df = df.rename(
            columns={
                "ITEM": "CODIGO_APU",
                "DESCRIPCION": "DESCRIPCION_APU",
                "CANT.": "CANTIDAD_PRESUPUESTO",
            }
        )

        # Filtra filas que no son items válidos y limpia datos
        df = df.dropna(subset=["CODIGO_APU"])
        df = df[df["CODIGO_APU"].str.contains(r"[\d,]", na=False)]
        df["CANTIDAD_PRESUPUESTO"] = pd.to_numeric(
            df["CANTIDAD_PRESUPUESTO"].astype(str).str.replace(",", "."),
            errors="coerce",
        )

        return df[["CODIGO_APU", "DESCRIPCION_APU", "CANTIDAD_PRESUPUESTO"]]
    except Exception as e:
        print(f"Error procesando presupuesto.csv: {e}")
        return pd.DataFrame()


def process_insumos_csv(path):
    """Lee y limpia el archivo insumos.csv."""
    try:
        # Encuentra la fila del encabezado dinámicamente
        with open(path, "r", encoding="latin1") as f:
            lines = f.readlines()
        header_index = next(
            (
                i
                for i, line in enumerate(lines)
                if "CODIGO" in line and "DESCRIPCION" in line
            ),
            -1,
        )

        if header_index == -1:
            return pd.DataFrame()

        df = pd.read_csv(path, delimiter=";", encoding="latin1", skiprows=header_index)
        df.columns = df.columns.str.strip()
        df = df.rename(
            columns={
                "DESCRIPCION": "DESCRIPCION_INSUMO",
                "VR. UNIT.": "VR_UNITARIO_INSUMO",
            }
        )

        # Limpia valores numéricos y descripciones
        df["VR_UNITARIO_INSUMO"] = pd.to_numeric(
            df["VR_UNITARIO_INSUMO"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", "."),
            errors="coerce",
        )
        df["DESCRIPCION_INSUMO"] = df["DESCRIPCION_INSUMO"].str.strip()
        df = df.dropna(subset=["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"])

        return df[["DESCRIPCION_INSUMO", "VR_UNITARIO_INSUMO"]]
    except Exception as e:
        print(f"Error procesando insumos.csv: {e}")
        return pd.DataFrame()


def process_apus_csv(path):
    """Parsea manualmente el archivo apus.csv que tiene formato de reporte."""
    apus_data = []
    current_apu_code = None
    try:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                # Busca el código del APU en las líneas de ITEM
                if "ITEM:" in line:
                    match = re.search(r"ITEM:\s*([\d,]+\b)", line)
                    if match:
                        current_apu_code = match.group(1).strip()
                # Si ya tenemos un APU, busca las líneas de insumos
                elif (
                    current_apu_code
                    and ";" in line
                    and not line.strip().startswith(
                        (
                            ";",
                            "DESCRIPCION",
                            "MATERIALES",
                            "MANO DE OBRA",
                            "EQUIPO",
                            "OTROS",
                            "SUBTOTAL",
                            "COSTO DIRECTO",
                        )
                    )
                ):
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) >= 3 and parts[0]:
                        try:
                            # La descripción del insumo es el identificador
                            description = parts[0]
                            # La cantidad está en la tercera columna
                            quantity = float(parts[2].replace(",", "."))
                            apus_data.append(
                                {
                                    "CODIGO_APU": current_apu_code,
                                    "DESCRIPCION_INSUMO": description,
                                    "CANTIDAD_APU": quantity,
                                }
                            )
                        except (ValueError, IndexError):
                            continue  # Ignora líneas que no se pueden parsear
        return pd.DataFrame(apus_data)
    except Exception as e:
        print(f"Error procesando apus.csv: {e}")
        return pd.DataFrame()


def process_csv_files(presupuesto_path, apus_path, insumos_path):
    """Orquesta el procesamiento de los tres archivos CSV y los consolida."""
    df_presupuesto = process_presupuesto_csv(presupuesto_path)
    df_insumos = process_insumos_csv(insumos_path)
    df_apus = process_apus_csv(apus_path)

    if df_presupuesto.empty or df_insumos.empty or df_apus.empty:
        return pd.DataFrame()

    # Unir APUs con Insumos usando la descripción como clave
    df_merged = pd.merge(df_apus, df_insumos, on="DESCRIPCION_INSUMO", how="left")
    df_merged["COSTO_INSUMO_EN_APU"] = (
        df_merged["CANTIDAD_APU"] * df_merged["VR_UNITARIO_INSUMO"]
    )

    # Calcular el costo total para cada APU
    df_apu_costos = (
        df_merged.groupby("CODIGO_APU")
        .agg(VR_UNITARIO_CALCULADO=("COSTO_INSUMO_EN_APU", "sum"))
        .reset_index()
    )

    # Unir con el presupuesto para obtener el resultado final
    df_final = pd.merge(df_presupuesto, df_apu_costos, on="CODIGO_APU", how="left")
    df_final["VALOR_TOTAL"] = (
        df_final["CANTIDAD_PRESUPUESTO"] * df_final["VR_UNITARIO_CALCULADO"]
    )
    df_final["ZONA"] = ""  # Añadir columna ZONA

    # Formatear el DataFrame final
    df_final = df_final.rename(
        columns={
            "CODIGO_APU": "Código APU",
            "DESCRIPCION_APU": "Descripción",
            "VALOR_TOTAL": "Valor Total",
        }
    )

    return df_final[["Código APU", "Descripción", "Valor Total", "ZONA"]]

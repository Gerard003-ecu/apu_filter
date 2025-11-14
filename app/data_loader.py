# app/data_loader.py
"""
Módulo para carga de datos desde diferentes formatos (CSV, Excel, PDF).
Diseñado para ser robusto, extensible y fácil de mantener.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import pdfplumber

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tipos aceptables para rutas
PathType = Union[str, Path]
DataFrameOrDict = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


def load_from_csv(
    path: PathType, sep: str = ";", encoding: str = "utf-8", **kwargs
) -> pd.DataFrame:
    """
    Carga un archivo CSV en un DataFrame.

    Args:
        path: Ruta al archivo.
        sep: Delimitador usado en el CSV.
        encoding: Codificación del archivo.
        **kwargs: Argumentos adicionales para pd.read_csv.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        UnicodeDecodeError: Si hay error de codificación.
        pd.errors.EmptyDataError: Si el archivo está vacío.
        Exception: Para otros errores de lectura.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(
        f"Cargando CSV desde {path} con delimitador '{sep}' y codificación '{encoding}'"
    )
    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding, **kwargs)
        if df.empty:
            logger.warning(f"El archivo CSV {path} está vacío.")
        return df
    except UnicodeDecodeError as e:
        logger.error(f"Error de codificación al leer {path}: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"El archivo CSV {path} está vacío o mal formado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al leer CSV {path}: {e}")
        raise


def load_from_xlsx(
    path: PathType, sheet_name: Optional[Union[str, int]] = 0, **kwargs
) -> DataFrameOrDict:
    """
    Carga una hoja de cálculo Excel (.xlsx o .xls).

    Si `sheet_name` es un string o un índice, devuelve un único DataFrame.
    Si `sheet_name` es `None`, devuelve un diccionario de DataFrames, donde
    las claves son los nombres de las hojas.

    Args:
        path: Ruta al archivo.
        sheet_name: Nombre/índice de la hoja o `None` para todas.
        **kwargs: Argumentos adicionales para pd.read_excel.

    Returns:
        Un DataFrame o un diccionario de DataFrames.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la hoja especificada no existe.
        Exception: Para otros errores de lectura.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    sheet_info = sheet_name if sheet_name is not None else "Todas"
    logger.info(f"Cargando Excel desde {path}, hoja(s): '{sheet_info}'")
    try:
        data = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

        if isinstance(data, dict):
            # Si se leen múltiples hojas (sheet_name=None), devolver el dict
            logger.info(f"Se cargaron {len(data)} hojas del archivo Excel.")
            return data

        # Si es un solo DataFrame
        if data.empty:
            logger.warning(f"La hoja '{sheet_name}' en {path} está vacía.")
        return data

    except ValueError as e:
        if "sheet" in str(e).lower():
            raise ValueError(f"Hoja '{sheet_name}' no encontrada en {path}: {e}")
        logger.error(f"Error de valor al leer Excel {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al leer Excel {path}: {e}")
        raise


def load_from_pdf(
    path: PathType, page_range: Optional[range] = None, **kwargs
) -> pd.DataFrame:
    """
    Extrae tablas de un archivo PDF y las combina en un DataFrame.

    Args:
        path: Ruta al archivo PDF.
        page_range: Rango de páginas a procesar (ej: range(0, 5) para páginas 1-5).
                    Por defecto, todas las páginas.
        **kwargs: Argumentos adicionales. `table_settings` se pasa a `extract_tables`.
                  Otros kwargs (como `laparams`) se pasan a `pdfplumber.open`.

    Returns:
        DataFrame con todas las tablas concatenadas.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        Exception: Si no se pueden extraer tablas o hay error en el PDF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Extrayendo tablas del PDF: {path}")

    # Extraer table_settings de kwargs si existe
    table_settings = kwargs.pop("table_settings", {})
    tables = []

    try:
        with pdfplumber.open(path, **kwargs) as pdf:
            pages = pdf.pages

            if page_range is not None:
                start = max(0, page_range.start)
                end = min(len(pages), page_range.stop)
                pages = pages[start:end]
            else:
                logger.info(f"Procesando {len(pages)} páginas del PDF.")

            for i, page in enumerate(pages):
                page_num = (page_range.start + i + 1) if page_range else (i + 1)
                extracted = page.extract_tables(table_settings)

                if not extracted:
                    logger.debug(f"No se encontraron tablas en la página {page_num}")
                    continue

                for table in extracted:
                    if table and len(table) > 0:
                        df_table = pd.DataFrame(
                            table[1:], columns=table[0] if table[0] else None
                        )
                        tables.append(df_table)
                        logger.debug(
                            f"Tabla encontrada en página {page_num}, forma: {df_table.shape}"
                        )

            if not tables:
                logger.warning(f"No se encontraron tablas en el PDF: {path}")
                return pd.DataFrame()

            combined_df = pd.concat(tables, ignore_index=True)
            logger.info(
                f"PDF cargado: {len(tables)} tablas combinadas en un DataFrame "
                f"de forma {combined_df.shape}"
            )
            return combined_df

    except Exception as e:
        logger.error(f"Error al procesar el PDF {path}: {e}")
        raise


def load_data(path: PathType, **kwargs) -> DataFrameOrDict:
    """
    Función factory que carga datos según la extensión del archivo.

    Para Excel, si se pasa `sheet_name=None`, puede devolver un diccionario
    de DataFrames. En los demás casos, devuelve un único DataFrame.

    Args:
        path: Ruta al archivo de datos.
        **kwargs: Argumentos adicionales pasados al cargador específico.

    Returns:
        Un DataFrame o un diccionario de DataFrames.

    Raises:
        ValueError: Si el formato no es soportado.
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando datos desde {path} (formato: {extension})")

    try:
        if extension == ".csv":
            return load_from_csv(path, **kwargs)
        elif extension in [".xlsx", ".xls"]:
            return load_from_xlsx(path, **kwargs)
        elif extension == ".pdf":
            return load_from_pdf(path, **kwargs)
        else:
            raise ValueError(
                f"Formato de archivo no soportado: {extension}. "
                f"Formatos soportados: .csv, .xlsx, .xls, .pdf"
            )
    except Exception as e:
        logger.error(f"Error al cargar {path} con formato {extension}: {e}")
        raise

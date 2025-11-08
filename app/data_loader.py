# app/data_loader.py
"""
Módulo para carga de datos desde diferentes formatos (CSV, Excel, PDF).
Diseñado para ser robusto, extensible y fácil de mantener.
"""

import logging
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import openpyxl  # Necesario para pd.read_excel con .xlsx
import pdfplumber

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tipos aceptables para rutas
PathType = Union[str, Path]


def load_from_csv(
    path: PathType,
    sep: str = ';',
    encoding: str = 'utf-8',
    **kwargs
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

    logger.info(f"Cargando CSV desde {path} con delimitador '{sep}' y codificación '{encoding}'")
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
    path: PathType,
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Carga una hoja de cálculo Excel (.xlsx o .xls) en un DataFrame.

    Args:
        path: Ruta al archivo.
        sheet_name: Nombre o índice de la hoja a cargar.
        **kwargs: Argumentos adicionales para pd.read_excel.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la hoja no existe.
        Exception: Para otros errores de lectura.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando Excel desde {path}, hoja '{sheet_name}'")
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        if isinstance(df, dict):
            # Si se devuelve un dict (múltiples hojas), se toma la primera clave
            first_sheet = next(iter(df))
            df = df[first_sheet]
            logger.warning(f"Se cargó múltiples hojas; usando la primera: '{first_sheet}'")
        if df.empty:
            logger.warning(f"La hoja '{sheet_name}' en {path} está vacía.")
        return df
    except ValueError as e:
        if "sheet" in str(e).lower():
            raise ValueError(f"Hoja '{sheet_name}' no encontrada en {path}: {e}")
        logger.error(f"Error de valor al leer Excel {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al leer Excel {path}: {e}")
        raise


def load_from_pdf(
    path: PathType,
    page_range: Optional[range] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Extrae tablas de un archivo PDF y las combina en un DataFrame.

    Nota: La extracción de tablas en PDF es heurística y depende del formato.
    Se extraen todas las tablas detectadas en las páginas especificadas.

    Args:
        path: Ruta al archivo PDF.
        page_range: Rango de páginas a procesar (ej: range(0, 5) para páginas 1-5).
                    Por defecto, todas las páginas.
        **kwargs: Argumentos adicionales para pdfplumber (como `laparams`).

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
    tables = []

    try:
        with pdfplumber.open(path, **kwargs) as pdf:
            pages = pdf.pages

            # Aplicar rango de páginas si se especifica
            if page_range is not None:
                start = max(0, page_range.start)
                end = min(len(pages), page_range.stop)
                pages = pages[start:end]
            else:
                logger.info(f"Procesando {len(pages)} páginas del PDF.")

            for i, page in enumerate(pages):
                page_num = (page_range.start + i) if page_range else (i + 1)
                extracted = page.extract_tables()
                if not extracted:
                    logger.debug(f"No se encontraron tablas en la página {page_num}")
                    continue

                for table in extracted:
                    if table and len(table) > 0:
                        # Convertir a DataFrame (puede tener filas con diferente longitud)
                        df_table = pd.DataFrame(table[1:], columns=table[0] if len(table) > 1 else None)
                        tables.append(df_table)
                        logger.debug(f"Tabla encontrada en página {page_num}, forma: {df_table.shape}")

            if not tables:
                logger.warning(f"No se encontraron tablas en el PDF: {path}")
                return pd.DataFrame()  # Devuelve DataFrame vacío

            # Concatenar todas las tablas
            combined_df = pd.concat(tables, ignore_index=True)
            logger.info(f"PDF cargado exitosamente: {len(tables)} tablas combinadas en {combined_df.shape}")
            return combined_df

    except Exception as e:
        logger.error(f"Error al procesar el PDF {path}: {e}")
        raise


def load_data(
    path: PathType,
    **kwargs
) -> pd.DataFrame:
    """
    Función factory que carga datos según la extensión del archivo.

    Args:
        path: Ruta al archivo de datos.
        **kwargs: Argumentos adicionales pasados al cargador específico.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        ValueError: Si el formato no es soportado.
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(path)
    extension = path.suffix.lower()

    # Validar existencia del archivo
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info(f"Cargando datos desde {path} (formato: {extension})")

    try:
        if extension == '.csv':
            return load_from_csv(path, **kwargs)
        elif extension in ['.xlsx', '.xls']:
            return load_from_xlsx(path, **kwargs)
        elif extension == '.pdf':
            return load_from_pdf(path, **kwargs)
        else:
            raise ValueError(f"Formato de archivo no soportado: {extension}. "
                             f"Formatos soportados: .csv, .xlsx, .xls, .pdf")
    except Exception as e:
        logger.error(f"Error al cargar {path} con formato {extension}: {e}")
        raise
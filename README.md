# APU Filter: Analizador de Costos de Construcción

APU Filter es una aplicación web construida con Flask, diseñada para procesar y analizar costos de construcción. La herramienta carga archivos Excel de presupuesto, Análisis de Precios Unitarios (APU) e insumos, consolida los datos y calcula los costos totales del proyecto, facilitando la gestión y el análisis de la información.

## Características Principales

- **Procesamiento de Datos desde Excel:** Carga y procesa tres archivos Excel (`presupuesto.xlsx`, `apus.xlsx`, `insumos.xlsx`) para consolidar la información.
- **Cálculo de Costos de Construcción:** Aplica la lógica de Análisis de Precios Unitarios (APU) para determinar los costos totales del proyecto.
- **Interfaz Web Sencilla:** Permite a los usuarios cargar los archivos y visualizar los resultados de forma interactiva en el navegador.

## Tecnologías Utilizadas

- **Backend:** Python, Flask
- **Análisis de Datos:** Pandas, Openpyxl
- **Gestor de Paquetes y Entorno:** uv
- **Calidad de Código:** Ruff

## Instalación y Puesta en Marcha

Sigue estos pasos para configurar el entorno de desarrollo y poner en marcha la aplicación:

1.  **Clonar el repositorio (si aplica):**
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd apu_filter
    ```

2.  **Crear un entorno virtual con `uv`:**
    ```bash
    uv venv
    ```
    Esto creará un directorio `.venv` en la raíz del proyecto.

3.  **Activar el entorno virtual:**
    - En Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - En macOS y Linux:
      ```bash
      source .venv/bin/activate
      ```

4.  **Compilar las dependencias (si es necesario):**
    Si has modificado `requirements.in` o `requirements-dev.in`, compila las dependencias.
    ```bash
    uv pip compile requirements.in -o requirements.txt
    uv pip compile requirements-dev.in -o requirements-dev.txt
    ```

5.  **Instalar las dependencias:**
    ```bash
    uv pip install -r requirements.txt -r requirements-dev.txt
    ```

## Uso

Una vez completada la instalación, puedes ejecutar la aplicación y las pruebas.

1.  **Ejecutar la aplicación Flask:**
    ```bash
    flask run
    ```
    La aplicación estará disponible en `http://127.0.0.1:5000`. Abre esta URL en tu navegador.

2.  **Ejecutar las pruebas:**
    Para asegurarte de que todo funciona correctamente, ejecuta el conjunto de pruebas:
    ```bash
    pytest
    ```

## Estructura del Directorio

```
apu_filter/
│
├── .venv/                   # Carpeta del entorno virtual gestionado por uv
│
├── templates/               # Carpeta estándar de Flask para los archivos HTML
│   └── index.html           # El nuevo dashboard interactivo de una sola página
│
├── uploads/                 # (Nueva) Carpeta temporal para los archivos que sube el usuario
│
├── app.py                   # El servidor Flask. Ahora gestiona la subida de archivos y sirve el dashboard.
├── procesador_csv.py        # (Nuevo) Módulo especializado en leer y procesar los CSV de SAGUT.
├── test_app.py              # Las pruebas unitarias que validan la lógica de procesador_csv.py
│
├── pyproject.toml           # Archivo de configuración para Ruff.
├── README.md                # La documentación del proyecto.
│
├── requirements.in          # Dependencias base de la aplicación.
├── requirements.txt         # Archivo de dependencias compilado por uv (para producción).
│
├── requirements-dev.in      # Dependencias de desarrollo (como ruff).
└── requirements-dev.txt     # Archivo de dependencias de desarrollo compilado.
```

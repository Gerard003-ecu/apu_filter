# APU Filter: Plataforma de Inteligencia de Costos para Construcción

## Resumen Ejecutivo

APU Filter es una plataforma de inteligencia de negocio diseñada para el sector de la construcción. Transforma la compleja tarea de analizar los costos de un proyecto, convirtiendo los tediosos reportes de Análisis de Precios Unitarios (APU) en una fuente de datos interactiva y estratégica. No es solo un lector de archivos, es una herramienta para tomar decisiones rápidas y precisas que impactan directamente en la rentabilidad y competitividad de su empresa.

## ¿Por qué APU Filter?

### De Horas a Segundos: Ahorro de Tiempo y Reducción de Errores
- **Automatización Inteligente:** APU Filter automatiza el procesamiento de los complejos reportes de costos (compatibles con el formato SAGUT), una tarea que tradicionalmente consume horas de trabajo de ingeniería.
- **Fiabilidad Garantizada:** Minimiza los errores humanos de transcripción y cálculo que se producen en flujos de trabajo manuales basados en Excel, garantizando cifras fiables para sus análisis.

### Una Herramienta de Decisión Estratégica
- **Simulador de Costos (AIU):** No se limite a ver los costos; proyéctelos. El simulador permite analizar en tiempo real el impacto de los costos indirectos (Administración, Imprevistos, Utilidad) en la rentabilidad final del proyecto.
- **Estimador Rápido:** Genere cotizaciones precisas para nuevos proyectos en segundos, no en horas. Esta agilidad le otorga una ventaja competitiva crucial para responder rápidamente a las oportunidades del mercado.

### Centralización y Consistencia
- **Lógica de Negocio Unificada:** Centralice las reglas de cálculo y análisis, que a menudo están dispersas en frágiles macros de Excel, difíciles de mantener y escalar.
- **Fuente Única de Verdad:** Asegure que todo el equipo de costos y presupuestos trabaje con las mismas reglas y los datos más actualizados, eliminando inconsistencias.

### Inteligencia de Datos y Visualización
- **Organizador de Proyecto:** Transforme datos planos en *insights* visuales. La plataforma permite analizar los costos por zona, tipo de trabajo o material, facilitando la identificación de los principales focos de costo.
- **Optimización a la Vista:** Identifique rápidamente oportunidades de optimización y tome decisiones informadas para mejorar el margen de sus proyectos.

## Características Principales
- **Motor de Simulación de Riesgos (Monte Carlo):** Va más allá de los cálculos estáticos. APU_filter simula miles de posibles resultados para cada APU, modelando la volatilidad de los precios de materiales y la variabilidad en el rendimiento de la mano de obra. Obtén el costo esperado, la desviación estándar y rangos de confianza para entender el verdadero perfil de riesgo de tu presupuesto.
- **Carga de Reportes SAGUT:** Procesamiento automatizado de archivos de Presupuesto, APU e Insumos.
- **Simulador AIU:** Módulo interactivo para modelar el impacto de los costos indirectos.
- **Organizador de Proyecto:** Visualización y análisis detallado de la estructura de costos.
- **Estimador Rápido:** Calculadora para generar presupuestos preliminares de forma instantánea.

## Capturas de Pantalla

*[Placeholder para captura de pantalla del Dashboard Principal]*

*[Placeholder para captura de pantalla del Simulador AIU]*

## Tecnologías Utilizadas

- **Backend:** Python, Flask
- **Análisis de Datos:** Pandas, Openpyxl
- **Gestor de Paquetes y Entorno:** uv
- **Calidad de Código:** Ruff

## Instalación y Uso

Siga estos pasos para configurar el entorno de desarrollo y poner en marcha la aplicación:

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd apu_filter
    ```

2.  **Crear y activar un entorno virtual con `uv`:**
    ```bash
    uv venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate   # Windows
    ```

3.  **Instalar las dependencias:**
    ```bash
    uv pip install -r requirements.txt -r requirements-dev.txt
    ```
    *Nota: Si necesita modificar las dependencias, edite `requirements.in` o `requirements-dev.in` y recompile con `uv pip compile`.*

4.  **Ejecutar la aplicación:**
    ```bash
    flask run
    ```
    La aplicación estará disponible en `http://127.0.0.1:5000`.

5.  **Ejecutar las pruebas:**
    ```bash
    pytest
    ```

## Estructura del Directorio

```
apu_filter/
│
├── app/                     # Contiene la lógica de la aplicación Flask
│   ├── __init__.py
│   ├── app.py               # El servidor Flask y los endpoints
│   └── procesador_csv.py    # Lógica de parsing y procesamiento de datos
│
├── models/                  # Módulos de lógica de negocio y análisis
│   ├── __init__.py
│   └── probability_models.py# Motor de simulación Monte Carlo
│
├── tests/                   # Pruebas unitarias y de integración
│   ├── __init__.py
│   ├── test_processing.py   # Pruebas para procesador_csv.py
│   └── test_models.py       # Pruebas para probability_models.py
│
├── templates/
├── uploads/
├── config.json              # Archivo de configuración
└── ...
```

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

Esta sección describe cómo configurar un entorno de desarrollo robusto utilizando un enfoque híbrido que combina **Conda**, **pip** y **uv**.

### ¿Por qué un Enfoque Híbrido?

El proyecto depende de librerías complejas que tienen dependencias a nivel de sistema (ej. C++), como `faiss-cpu`, y otras que requieren versiones específicas de CPU, como `torch`. Para manejar esto de forma fiable:
-   **Conda** se utiliza para instalar `faiss-cpu`, ya que gestiona de manera excelente las dependencias binarias complejas.
-   **Pip** se usa para instalar la versión de `torch` específica para CPU desde su repositorio oficial.
-   **uv** gestiona el resto de las dependencias de Python de manera extremadamente rápida y eficiente.

Este enfoque garantiza una instalación estable y reproducible, evitando los comunes errores de compilación.

### Diagrama del Flujo de Instalación

```mermaid
graph TD
    A[Inicio: Entorno Limpio] --> B{Paso 1: Crear Entorno Conda con Python 3.10};
    B --> C[Paso 2: Activar Entorno];
    C --> D{Paso 3: Instalar Paquetes Especiales};
    D -- "conda install -c pytorch" --> E[faiss-cpu];
    D -- "pip install --index-url ..." --> F[torch (cpu)];
    E --> H;
    F --> H;
    G[requirements.txt (sin faiss/torch)] --> H{Paso 4: Instalar Dependencias de la Aplicación};
    H -- "uv pip install -r" --> I[Librerías Restantes];
    I --> J[Fin: Entorno Listo ✅];

```

### Pasos Detallados

**Requisito Previo:** Asegúrese de tener instalado **Miniconda** o **Anaconda**. Puede descargarlo desde [aquí](https://www.anaconda.com/download).

---

#### **Paso 1: Crear el Entorno Conda**

Cree un nuevo entorno Conda llamado `apu_filter_env` con Python 3.10.

```bash
conda create --name apu_filter_env python=3.10
```

---

#### **Paso 2: Activar el Entorno**

Active el entorno recién creado. **Debe hacer esto cada vez que trabaje en el proyecto.**

```bash
conda activate apu_filter_env
```

---

#### **Paso 3: Instalar Paquetes Especiales (faiss-cpu y torch)**

Instale `faiss-cpu` usando Conda y `torch` usando pip con el índice de PyTorch.

1.  **Instalar faiss-cpu:**
    ```bash
    conda install -c pytorch faiss-cpu
    ```

2.  **Instalar torch (versión CPU):**
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

---

#### **Paso 4: Instalar el Resto de Dependencias con uv**

Finalmente, instale todas las demás dependencias del proyecto listadas en `requirements.txt`.

```bash
uv pip install -r requirements.txt
```

> **Nota Importante:** El archivo `requirements.txt` no debe contener `faiss-cpu` ni `torch`. Si alguna vez necesita regenerar este archivo (ej. usando `uv pip compile`), asegúrese de excluir estas dos librerías para evitar conflictos.

---

### **Ejecución de la Aplicación**

Una vez completada la instalación, puede ejecutar la aplicación:

```bash
flask run
```

La aplicación estará disponible en `http://127.0.0.1:5000`.

### **Ejecución de las Pruebas**

Para verificar que todo funciona correctamente, ejecute el conjunto de pruebas:

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
│   ├── procesador_csv.py    # Lógica de parsing y procesamiento de datos
|   └── config.json              # Archivo de configuración
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
└── ...
```

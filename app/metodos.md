## Ingeniería Bajo el Capó: La Garantía de Estabilidad

Aunque nuestra prioridad es su negocio, la solidez técnica es nuestra garantía. APU Filter está construido sobre una arquitectura modular que separa claramente las responsabilidades, garantizando robustez y escalabilidad. Sus tres pilares fundamentales son:

### 1. Condensador de Flujo de Datos (Data Flux Condenser)
- **Componente Clave:** `app/flux_condenser.py`
- **Función:** Actúa como un estabilizador dinámico de señal a la entrada del sistema. Implementa una arquitectura de **"Caja de Cristal"**, transformando la ingesta de datos en un proceso observable, medible y auto-regulado mediante principios de física y teoría de control.

#### ⚙️ Nivel 1: Motor de Física RLC (El Sensor)
El sistema evoluciona más allá de métricas simples hacia un **Modelo Energético Escalar**. En lugar de monitorear solo voltaje o corriente, unificamos las unidades bajo un lenguaje común: La Energía (Julios).

1.  **Energía Potencial ($E_c = \frac{1}{2}CV^2$) - Presión de Datos:**
    *   Representa la "carga de trabajo" acumulada por el volumen de registros.
    *   Calcula la presión que ejerce el lote de datos sobre el sistema.
2.  **Energía Cinética ($E_l = \frac{1}{2}LI^2$) - Inercia de Calidad:**
    *   Representa el momento o "inercia" de la calidad del flujo.
    *   Un flujo de alta calidad ($I \approx 1.0$) tiene una inercia fuerte que resiste perturbaciones, dificultando que errores menores desestabilicen el proceso.
3.  **Potencia Disipada ($P = I_{ruido}^2 R$) - Calor/Fricción:**
    *   **Termodinámica del Software:** Calcula el "calor" generado por la resistencia dinámica de los datos sucios.
    *   Si el sistema gasta demasiada energía procesando basura (ruido), se genera sobrecalentamiento lógico.

#### 🧠 Nivel 2: Controlador PI Discreto (El Cerebro)
Sobre la capa física, opera un **Lazo de Control Cerrado (Feedback Loop)** que ajusta el comportamiento del sistema en tiempo real, ahora con protección térmica:

*   **Algoritmo:** Controlador **Proporcional-Integral (PI)** discreto.
*   **Setpoint:** Mantiene una saturación estable (Flujo Laminar).
*   **Variable de Control:** El tamaño del lote de procesamiento (*Batch Size*).
*   **Disyuntor Térmico (Nuevo):**
    *   Además del PID, el sistema implementa un "Diodo de Rueda Libre" térmico.
    *   Si la **Potencia Disipada** supera un umbral crítico (> 50W), el sistema activa un freno de emergencia, reduciendo drásticamente el tamaño del lote independientemente de la saturación, para "enfriar" el proceso y evitar colapsos por calidad de datos.

**Resultado:** Un sistema bi-mimético que no solo adapta su velocidad, sino que también gestiona su "temperatura" operativa para garantizar una estabilidad del 100% bajo cualquier condición.

#### 🛡️ Resiliencia y Recuperación
El sistema implementa mecanismos de defensa avanzados:
*   **Anti-Windup PID:** Previene la saturación del controlador ante cargas sostenidas.
*   **Recuperación Parcial:** Capacidad de aislar lotes corruptos y continuar el procesamiento del resto del archivo.
*   **Protección Térmica:** Freno de emergencia automático si la disipación de energía (fricción de datos) supera los umbrales de seguridad.

### 2. Pipeline de Procesamiento de Datos
- **Componente Clave:** `app/procesador_csv.py`
- **Función:** Es el orquestador central que toma los datos crudos del parser y los transforma en un modelo de costos consolidado.
- **Mecanismo:** Utiliza un patrón `Pipeline` con pasos secuenciales y bien definidos:
    1.  **Carga de Datos:** Ingiere los tres archivos principales (Presupuesto, APUs, Insumos).
    2.  **Fusión de Datos:** Enriquece los insumos de los APUs con los precios del catálogo maestro de insumos.
    3.  **Cálculo de Costos:** Agrega los costos de los insumos para calcular el valor total de cada APU, desglosado por categoría (Materiales, Mano de Obra, Equipo).
    4.  **Merge Final:** Une los costos calculados de los APUs con las cantidades del archivo de presupuesto para generar el informe final.

### 3. Estimador Inteligente
- **Componente Clave:** `app/estimator.py`
- **Función:** Proporciona una capacidad de búsqueda avanzada para generar cotizaciones rápidas para nuevos proyectos, basándose en el conocimiento extraído de APUs históricos.
- **Mecanismo Dual:**
    - **Búsqueda por Palabras Clave:** Un método tradicional y rápido que busca coincidencias directas de texto.
    - **Búsqueda Semántica (Vectorial):** Su capacidad más potente. Utiliza modelos de `sentence-transformers` para convertir las descripciones de los APUs en vectores numéricos (embeddings). Luego, usa **FAISS** para encontrar los APUs más *conceptualmente similares* a una nueva descripción, incluso si no comparten las mismas palabras.

## Tecnologías Utilizadas

La plataforma está construida sobre una pila de tecnologías modernas de alto rendimiento:

- **Backend:** **Flask** para la API web.
- **Análisis de Datos y ML:**
    - **Pandas:** Utilizado como la base para la manipulación de datos.
    - **Sentence-Transformers:** Para la generación de embeddings de texto que potencian la búsqueda semántica.
    - **FAISS (Facebook AI Similarity Search):** Para la búsqueda vectorial de alta velocidad de los APUs más similares.
- **Parsing y Estructura de Datos:**
    - **Lark:** Para el parsing robusto de la gramática de los insumos en los archivos de APU.
    - **Dataclasses:** Para la creación de esquemas de datos (`schemas.py`) que garantizan la consistencia y validación.
- **Entorno y Dependencias:**
    - **Conda:** Para gestionar el entorno y las dependencias complejas con componentes binarios (ej. `faiss-cpu`).
- **Redis:** Para la gestión de sesiones de usuario, garantizando la persistencia de datos entre solicitudes.
    - **uv & pip:** Para la gestión rápida y eficiente del resto de las dependencias de Python.
- **Calidad de Código y Pruebas:**
    - **Pytest:** Para una suite de pruebas exhaustiva que cubre desde unidades hasta la integración completa.
    - **Ruff:** Para el formateo y linting del código, asegurando un estilo consistente y de alta calidad.

## Instalación y Uso

Esta sección describe cómo configurar el entorno técnico para su equipo de TI, garantizando una implementación robusta y segura.

### La Arquitectura de la Instalación: Una Analogía de Engranajes

Para entender por qué seguimos un orden de instalación específico, podemos visualizar nuestro entorno como una caja de cambios de precisión compuesta por tres engranajes diferentes, cada uno con una función especializada.

1.  **Conda: El Engranaje Principal y de Potencia (El Engranaje Grande)**
    *   **Rol:** Mueve las piezas más pesadas y complejas que no son de Python puro y dependen del sistema operativo (ej. librerías C++).
    *   **Característica:** Es potente y fiable, diseñado para buscar e instalar paquetes pre-compilados que encajan perfectamente con la arquitectura de la máquina.
    *   **En APU Filter:** Su única tarea es instalar `faiss-cpu`, una librería con dependencias complejas a nivel de sistema.

2.  **Pip (con `--index-url`): La Herramienta Especializada**
    *   **Rol:** Se utiliza para una pieza crítica que necesita una instalación muy específica desde un repositorio exclusivo.
    *   **Característica:** Comunica una intención precisa: "Ve únicamente a este almacén específico (el de PyTorch para CPU) y trae la pieza exacta que encuentres allí".
    *   **En APU Filter:** Su única tarea es instalar la versión `torch` optimizada exclusivamente para CPU, evitando la descarga de las pesadas librerías de CUDA.

3.  **uv/pip: El Engranaje de Alta Velocidad y Precisión (El Engranaje Pequeño)**
    *   **Rol:** Ensambla todos los componentes de la aplicación que son de Python puro, comunicándose directamente con el ecosistema de Python (PyPI).
    *   **Característica:** Es ultrarrápido y ágil, ideal para manejar dependencias estándar de Python, pero no tiene la fuerza para gestionar las piezas pesadas que maneja Conda.
    *   **En APU Filter:** Su tarea es instalar todo lo demás desde `requirements.txt` de forma eficiente.

### Pasos Detallados de Instalación

**Requisito Previo:** Asegúrese de tener instalado Miniconda o Anaconda. Puede descargarlo desde [aquí](https://www.anaconda.com/products/distribution).

**Paso 1: Crear el Entorno Base (Conda)**
Cree un nuevo entorno Conda llamado `apu_filter_env` con Python 3.10, la versión sobre la cual se construirán los demás componentes.
```bash
conda create --name apu_filter_env python=3.10
```

**Paso 2: Activar el Entorno**
Active el entorno recién creado. **Debe hacer esto cada vez que trabaje en el proyecto.**
```bash
conda activate apu_filter_env
```

**Paso 3: Instalar Componentes Pesados (Conda y Pip Especializado)**
Instale los "engranajes" principales que requieren compilaciones y dependencias complejas.

*   **Instalar `faiss-cpu` (El Engranaje de Potencia):**
    ```bash
    conda install -c pytorch faiss-cpu
    ```

*   **Instalar `torch` (La Herramienta Especializada):**
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

**Paso 4: Instalar Dependencias de la Aplicación (uv)**
Instale todas las demás dependencias de Python puro con el "engranaje de alta velocidad".
```bash
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

**Paso 5: Instalar y Configurar el Servidor de Sesiones (Redis)**
Para garantizar la persistencia de los datos del usuario entre solicitudes, la aplicación utiliza Redis.

*   **Instalar `redis` (El Engranaje de Estabilidad):**
    Es crucial instalar Redis a través del canal `conda-forge` para asegurar la compatibilidad entre diferentes sistemas operativos, incluyendo macOS y Linux.
    ```bash
    conda install -c conda-forge redis
    ```

**Nota Importante:** El archivo `requirements.txt` no debe contener `faiss-cpu` ni `torch`. Si alguna vez necesita regenerar este archivo (ej. usando `uv pip compile requirements.in`), asegúrese de excluir estas dos librerías para evitar conflictos de instalación.

## Flujo de Trabajo del Proyecto

El ciclo de vida del desarrollo y uso de la aplicación sigue estos pasos:

1.  **Configuración:** La lógica de negocio (mapeo de columnas, umbrales, reglas del estimador) se gestiona en `app/config.json`.
2.  **Pre-procesamiento:** Si los datos de los APUs cambian, debe regenerar los embeddings ejecutando:
    ```bash
    python scripts/generate_embeddings.py --input path/to/processed_apus.json
    ```
3.  **Ejecución de la Aplicación:** Con el entorno activado, inicie el servidor Flask:
    ```bash
    python -m flask run --port=5002
    ```
4.  **Validación y Pruebas:** Para verificar la integridad del código, ejecute la suite de pruebas completa:
    ```bash
    pytest -vv
    ```

## Estructura del Directorio

El proyecto está organizado con una clara separación de responsabilidades para facilitar la mantenibilidad y la escalabilidad.

```
apu_filter/
│
├── app/                        # Lógica principal de la aplicación Flask
│   ├── __init__.py
│   ├── app.py                  # Factory de la app, endpoints API y carga de modelos
│   ├── procesador_csv.py       # Orquestador del pipeline de procesamiento de datos
│   ├── report_parser_crudo.py  # Parser especializado para archivos de APU semi-estructurados
│   ├── apu_processor.py        # Motor de transformación que aplica lógica de negocio a los datos parseados
│   ├── estimator.py            # Lógica de estimación con búsqueda semántica y por keywords
│   ├── flux_condenser.py       # Lógica del condensador de flujos de datos
│   ├── data_loader.py          # Capa de abstracción para leer datos (.csv, .xlsx, .pdf)
│   ├── schemas.py              # Definición de los esquemas de datos (dataclasses)
│   ├── utils.py                # Funciones de utilidad generales (normalización, parsing, etc.)
│   ├── config.json             # Archivo de configuración de la lógica de negocio
│   └── embeddings/             # Directorio para los artefactos de ML (índice FAISS, mapeo)
│
├── data/                       # Datos de entrada y resultados intermedmedios
│   ├── presupuesto_clean.csv   # Versión sanitizada del presupuesto, lista para el pipeline
│   ├── insumos_clean.csv       # Versión sanitizada de insumos, lista para el pipeline
│   └── apus_clean.csv          # Versión sanitizada de apus, lista para el pipeline  
│
├── data_dirty/                 # Datos crudos y sin procesar
│   ├── presupuesto.csv         # Archivo de presupuesto original con posibles errores
│   ├── insumos.csv             # Archivo de insumos original con posibles errores
│   └── apus.csv                # Archivo de apus original con posibles errores  
│
├── models/                     # Módulos de lógica de negocio y análisis avanzado
│   ├── __init__.py
│   └── probability_models.py   # Motor de simulación Monte Carlo para análisis de riesgos
│
├── scripts/                    # Herramientas de línea de comandos para desarrolladores
│   ├── __init__.py
│   ├── generate_embeddings.py       # Script para generar el índice de búsqueda semántica
│   ├── diagnose_apus_file.py        # Herramienta para analizar formatos de archivo de APU
│   ├── diagnose_insumos_file.py     # Herramienta para analizar formatos de archivo de insumos
│   ├── diagnose_presupuesto_file.py # Herramienta para analizar formatos de archivo de presupuesto
│   └── clean_csv.py                 # Herramienta para limpiar caracteres sucios y crear un archivo csv limpio 
│
├── tests/                      # Suite de pruebas completa del proyecto
│   ├── test_app.py             # Pruebas de integración para los endpoints de la API
│   ├── test_procesador_csv.py  # Pruebas para el orquestador del pipeline
│   ├── test_apu_processor.py   # Pruebas para el motor de transformación
│   ├── test_estimator.py       # Pruebas para la lógica de estimación
│   ├── test_data_loader.py     # Pruebas para la capa de carga de datos
│   └── test_data.py            # Datos de prueba centralizados
│
├── templates/                  # Plantillas HTML para la interfaz (si aplica)
├── uploads/                    # Directorio temporal para archivos subidos
│
├── requirements.in             # Archivo fuente para definir dependencias
├── requirements.txt            # Archivo de dependencias "congelado" generado por uv
└── pyproject.toml              # Archivo de configuración del proyecto Python
```

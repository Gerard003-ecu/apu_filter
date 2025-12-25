# Ingeniería Bajo el Capó: La Garantía de Estabilidad

En APU Filter, la tecnología no es un adorno; es la estructura que sostiene el negocio. A continuación, detallamos cómo nuestros "Expertos Digitales" utilizan ingeniería avanzada para resolver problemas cotidianos de la construcción.

---

## 1. El Estabilizador: Control de Flujo y Resiliencia
**Componente:** `app/flux_condenser.py`

El mayor enemigo de la gestión de datos masivos es la inconsistencia y los picos de carga. Un sistema tradicional se bloquea (crash) cuando intenta procesar más de lo que puede masticar. Nosotros implementamos un sistema de **Ingeniería de Confiabilidad (SRE)** basado en principios de **Backpressure (Contrapresión)** y **Rate Limiting Adaptativo**.

### La Lógica: Estabilidad ante el Caos
Imagine una autopista inteligente. Si hay demasiados carros (datos), los semáforos de entrada (el sistema) ajustan sus tiempos automáticamente para evitar un trancón total.
El **Data Flux Condenser** gestiona la tasa de ingestión de datos para asegurar que el servidor siempre opere en su zona óptima de rendimiento.

1.  **Presión de Datos (Input Pressure):** Mide la cantidad de registros esperando ser procesados.
2.  **Inercia de Calidad (Quality Inertia):** Mide qué tan "limpios" están los datos. Datos limpios fluyen rápido; datos sucios requieren más tiempo.
3.  **Fricción Operativa (System Friction):** El esfuerzo computacional real que toma procesar el lote actual.

### El Cerebro del Estabilizador (Controlador PID)
Para gestionar estas variables, utilizamos un algoritmo de control **Proporcional-Integral-Derivativo (PID)**, el mismo tipo de lógica usada en controles industriales de temperatura o velocidad crucero de vehículos.

*   **Si los datos son complejos (Alta Fricción):** El sistema reduce automáticamente el tamaño del lote (*Batch Size*) para procesar con precisión quirúrgica sin saturar la memoria.
*   **Si los datos fluyen bien:** El sistema acelera, aumentando el tamaño del lote para maximizar el rendimiento.
*   **Resultado:** Un **Flujo Laminar** constante. El sistema nunca se detiene, solo ajusta su velocidad para sobrevivir.

> **Nota Técnica (Inspiración Interna):** Bajo el capó, modelamos estas métricas usando ecuaciones análogas a un circuito eléctrico RLC (Resistencia-Inductancia-Capacitancia) para calcular la "Energía" del sistema, lo que nos permite predecir saturaciones antes de que ocurran.

#### ⚙️ Motor de Física de Flujo (Energy-Based RLC)
A diferencia de los sistemas tradicionales que solo miden métricas discretas como el uso de memoria o CPU, nuestro sistema modela el flujo de datos como un sistema físico. No solo medimos el "voltaje" (carga), sino el **Balance Energético** total.

*   **Energía Potencial ($E_p$):** La presión acumulada por el volumen de datos en cola.
*   **Energía Cinética ($E_k$):** La inercia del procesamiento de calidad.
*   **Potencia Disipada ($P_{diss}$):** La energía perdida por fricción (procesamiento de datos basura).

Esta perspectiva termodinámica nos permite detectar "sobrecalentamientos" lógicos antes de que se conviertan en fallos del sistema.

#### 🧠 Controlador PID Discreto con Anti-windup
El cerebro que ajusta el flujo no es un simple `if/else`. Es un controlador PID completo que ajusta el tamaño del lote en tiempo real.
*   **Anti-windup:** Implementamos lógica avanzada para evitar que el término integral se acumule infinitamente cuando el sistema está saturado, garantizando una recuperación rápida después de picos de carga.
*   **Filtrado de Ruido:** El controlador utiliza medias móviles exponenciales para ignorar fluctuaciones transitorias y responder solo a tendencias reales.

---

## 2. El Estratega: Estimación de "Caja Blanca"
**Componente:** `app/estimator.py`

En ingeniería, la confianza lo es todo. Un ingeniero no aceptará un precio solo porque "la máquina lo dijo". Por eso, nuestro Estratega opera bajo una filosofía de **Transparencia Radical**. No es una Caja Negra; es una Caja de Cristal.

### Evidencia, no Magia
Cuando el sistema sugiere un costo, entrega un reporte forense de su decisión:

#### A. Búsqueda Semántica (El Concepto)
El sistema entiende que *"Muro en ladrillo tolete"* y *"Mampostería de arcilla"* son lo mismo, aunque no compartan palabras.
*   **Tecnología:** Sentence-Transformers + FAISS Vector Database.
*   **Output al Usuario:** "Encontré este ítem con una **Similitud Conceptual del 94%**".

#### B. Búsqueda por Palabras Clave (El Detalle)
Si la semántica falla, buscamos coincidencias exactas.
*   **Output al Usuario:** "Encontré este ítem porque coincide en 3 de 4 palabras clave".

#### C. Simulación de Riesgo (El Futuro)
Usamos el Método de Monte Carlo para proyectar 1,000 escenarios posibles de variación de precios.
*   **Output al Usuario:** "El precio base es $100, pero hay un **35% de probabilidad** de que suba a $115 debido a la volatilidad histórica".

---

## 3. El Director: Orquestación del Pipeline
**Componente:** `app/pipeline_director.py` (Anteriormente `procesador_csv.py`)

Para evitar el "código espagueti", hemos centralizado la lógica de control. El Director no procesa datos; él da las órdenes.

## 4. Orquestación Granular: El Pipeline como Máquina de Estados

A diferencia de los scripts lineales tradicionales, el `PipelineDirector` implementa una arquitectura de **Ejecución Atómica con Persistencia de Estado**.

*   **Atomicidad:** Cada paso (ej. `CalculateCosts`) es una unidad discreta que recibe un contexto, lo procesa y retorna un nuevo estado.
*   **Persistencia:** Entre pasos, el "Vector de Estado" se serializa (Redis/Pickle). Esto permite al Agente intervenir, reintentar un paso específico o pausar el flujo sin perder datos.
*   **Método:** `run_single_step(step_name)` permite la ejecución quirúrgica de procesos.

## 5. Motor de Inteligencia Financiera (Financial Engine)

Superando la estimación de costos determinista, este módulo inyecta variables estocásticas de mercado:

*   **WACC (Weighted Average Cost of Capital):** Descuenta los flujos de caja futuros basándose en la estructura de capital y riesgo país.
*   **VaR (Value at Risk):** Utiliza simulaciones de Monte Carlo para determinar la pérdida máxima probable con un 95% de confianza.
*   **Opciones Reales:** Valora la flexibilidad estratégica (ej. la opción de esperar o expandir) utilizando modelos binomiales, transformando la incertidumbre en un valor cuantificable.

#### Mecanismos de Defensa (SRE)
Esta no es una metáfora decorativa. Utilizamos lógica de sistemas dinámicos para proteger la infraestructura:

*   **Load Shedding (Disyuntor Térmico):** Si la "fricción" (error rate o complejidad) supera un umbral de seguridad (> 50W de potencia disipada equivalente), el sistema activa un freno de emergencia, reduciendo drásticamente la carga para "enfriar" el proceso.
*   **Anti-Windup:** Evita que el controlador PID se quede "pegado" tratando de corregir errores acumulados pasados, manteniendo la respuesta ágil ante cambios presentes.
*   **Recuperación Parcial:** Si un lote de datos está corrupto, el sistema lo aísla y continúa con el resto del archivo, en lugar de fallar todo el proceso.

---

## 6. Métricas de Concentración Logística (El Alquimista)

Para garantizar la viabilidad logística, el sistema aplica métricas económicas clásicas al flujo de materiales.

### Métricas de Concentración Logística

*   **Índice de Gini ($G$):** Mide la desigualdad en la distribución del presupuesto.
    *   $G \approx 1$: Pocos materiales consumen todo el presupuesto (Alto Riesgo de Abastecimiento).
    *   $G \approx 0$: Costo distribuido uniformemente.
*   **Ratio de Pareto:** Porcentaje de ítems que constituyen el 80% del costo total. Permite enfocar la gestión de compras en los insumos críticos.

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

### Métrica de Estabilidad Piramidal (`pyramid_stability`)

El sistema calcula un índice de robustez arquitectónica del presupuesto utilizando la siguiente relación:

$$ \Psi = \frac{N_{insumos}}{N_{apus}} \times \frac{1}{\rho} $$

Donde:
*   $N_{insumos}$: Cantidad de recursos únicos (Amplitud de base).
*   $N_{apus}$: Cantidad de actividades (Complejidad táctica).
*   $\rho$: Densidad del grafo (Interconectividad).

**Interpretación:**
*   **$\Psi > 10$ (Sólida):** Base ancha. El proyecto tiene recursos diversificados y dependencias claras.
*   **$\Psi < 1$ (Invertida):** Base estrecha. El proyecto depende críticamente de muy pocos recursos altamente conectados. Un fallo en el suministro de un insumo clave podría detener múltiples frentes de obra.

---

## 4. El Agente: Orquestación Autónoma
**Componente:** `agent/orchestrator.py`

La evolución de APU Filter introduce capacidades agénticas para coordinar tareas complejas de manera autónoma. El Orquestador actúa como un sistema nervioso central que conecta los microservicios y asegura la coherencia del flujo de trabajo.

### Responsabilidades Clave:
*   **Coordinación de Tareas:** Descompone objetivos de alto nivel en pasos ejecutables.
*   **Monitoreo de Estado:** Supervisa la salud de los procesos en tiempo real.
*   **Toma de Decisiones:** Ajusta dinámicamente la ruta de ejecución basándose en la retroalimentación del sistema (feedback loops).

---

## Tecnologías Utilizadas

La plataforma está construida sobre una pila de tecnologías modernas de alto rendimiento:

- **Backend:** **Flask** para la API web.
- **Inteligencia Artificial y Agentes:**
    - **Microservicios Agénticos:** Arquitectura modular para tareas autónomas.
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
├── agent/                      # Módulo de Inteligencia Artificial y Agentes
│   ├── __init__.py
│   └── orchestrator.py         # Orquestador autónomo de microservicios
│
├── app/                        # Lógica principal de la aplicación Flask
│   ├── __init__.py
│   ├── app.py                  # Factory de la app, endpoints API y carga de modelos
│   ├── pipeline_director.py    # Orquestador del pipeline de procesamiento de datos
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
│   ├── test_pipeline_director.py  # Pruebas para el orquestador del pipeline
│   ├── test_apu_processor.py   # Pruebas para el motor de transformación
│   ├── test_estimator.py       # Pruebas para la lógica de estimación
│   ├── test_data_loader.py     # Pruebas para la capa de carga de datos
│   ├── test_orchestrator.py    # Pruebas para el orquestador agéntico
│   └── test_data.py            # Datos de prueba centralizados
│
├── templates/                  # Plantillas HTML para la interfaz (si aplica)
├── uploads/                    # Directorio temporal para archivos subidos
│
├── requirements.in             # Archivo fuente para definir dependencias
├── requirements.txt            # Archivo de dependencias "congelado" generado por uv
└── pyproject.toml              # Archivo de configuración del proyecto Python
```

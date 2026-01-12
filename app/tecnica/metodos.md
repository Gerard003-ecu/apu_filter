
--------------------------------------------------------------------------------


--------------------------------------------------------------------------------


.

--------------------------------------------------------------------------------
🔮 3. La Física del Valor: Termodinámica y Estocástica
Componente: app/financial_engine.py
El sistema trasciende la contabilidad determinista mediante el Modelo Unificado de Física del Costo, donde el riesgo financiero es una consecuencia directa de la estructura física y la temperatura del mercado
:
1. Termodinámica del Costo (Tsys​): La volatilidad es tratada como calor. El sistema simula cómo la "fiebre" inflacionaria de un insumo se difunde por el grafo del presupuesto hasta el ápice del proyecto
.
2. Eficiencia Exergética: Distinguimos entre Exergía (trabajo útil en estructura y cimentación) y Anergía (desperdicio o acabados cosméticos). Una eficiencia < 50% indica un edificio débil con "pintura cara"
.
3. Valoración Estocástica: El Oráculo de Riesgos ejecuta 10,000 Simulaciones de Monte Carlo y utiliza distribuciones Student-t para modelar "Cisnes Negros" que las hojas de cálculo tradicionales ignoran
.
4. Opciones Reales: Valora la flexibilidad estratégica (ej. la opción de esperar para comprar cemento) como un activo financiero real
.

--------------------------------------------------------------------------------



# Ingeniería Bajo el Capó: La Garantía de Estabilidad

En APU Filter, la tecnología no es un adorno periférico; es la Arquitectura Cognitiva que sostiene la integridad del negocio. Este documento detalla la fundamentación científica (Matemática Aplicada y Física de Datos) que permite a nuestros "Sabios Digitales" transformar una lista plana de ítems en un diagnóstico de sabiduría estratégica

---

## 1. El Estabilizador: Control de Flujo y Resiliencia

🛡️ 1. El Sistema Nervioso: Física de Datos (FluxPhysicsEngine)
**Componente:** `app/flux_condenser.py`
Para garantizar la estabilidad ante el caos de la ingesta masiva de datos, modelamos el flujo no como simples registros, sino como un Circuito RLC (Resistencia-Inductancia-Capacitancia). El sistema monitorea el "Balance Energético" en tiempo real para decidir si acepta o rechaza un lote de información:
• Energía Potencial (Ec​=1/2​C*V^2): Mide la "Presión de Datos" o volumen acumulado en la cola de procesamiento. Una Ec​ alta dispara válvulas de alivio para evitar el desbordamiento del sistema.
• Energía Cinética (El​=1/2​L*I^2): Representa la "Inercia de Calidad". Un flujo limpio genera una alta corriente (I), haciendo que el sistema sea difícil de desestabilizar por ruidos menores.
• Potencia Disipada (P=I^2*R): Calcula el "calor" o desperdicio generado por datos sucios (fricción operativa). Si supera los 50W, se activa el Freno de Emergencia térmico.
• Voltaje Flyback (Vflyback​=L*di/dt​): Detecta caídas bruscas en la calidad de los datos, bloqueando la ingesta antes de que la inestabilidad corrompa el análisis estructural.
Este flujo es regulado por un Controlador PI Discreto con lógica Anti-windup, asegurando un Flujo Laminar constante y una recuperación rápida ante picos de carga.
El mayor enemigo de la gestión de datos masivos es la inconsistencia y los picos de carga. Un sistema tradicional se bloquea (crash) cuando intenta procesar más de lo que puede masticar. Nosotros implementamos un sistema de **Ingeniería de Confiabilidad (SRE)** basado en principios de **Backpressure (Contrapresión)** y **Rate Limiting Adaptativo**.

### La Lógica: Estabilidad ante el Caos
Imagine una autopista inteligente. Si hay demasiados carros (datos), los semáforos de entrada (el sistema) ajustan sus tiempos automáticamente para evitar un trancón total.
El **Data Flux Condenser** gestiona la tasa de ingestión de datos para asegurar que el servidor siempre opere en su zona óptima de rendimiento.

1.  **Presión de Datos (Input Pressure):** Mide la cantidad de registros esperando ser procesados.
2.  **Inercia de Calidad (Quality Inertia):** Mide qué tan "limpios" están los datos. Datos limpios fluyen rápido; datos sucios requieren más tiempo.
3.  **Fricción Operativa (System Friction):** El esfuerzo computacional real que toma procesar el lote actual.

### 🧠  El Cerebro del Estabilizador (Controlador PID)
Para gestionar estas variables, utilizamos un algoritmo de control **Proporcional-Integral-Derivativo (PID)**, el mismo tipo de lógica usada en controles industriales de temperatura o velocidad crucero de vehículos.

*   **Si los datos son complejos (Alta Fricción):** El sistema reduce automáticamente el tamaño del lote (*Batch Size*) para procesar con precisión quirúrgica sin saturar la memoria.
*   **Si los datos fluyen bien:** El sistema acelera, aumentando el tamaño del lote para maximizar el rendimiento.
*   **Resultado:** Un **Flujo Laminar** constante. El sistema nunca se detiene, solo ajusta su velocidad para sobrevivir.

---

## 🏗️ 2. La Geometría del Negocio: Topología Algebraica

**Componente:** agent/business_topology.py
El Arquitecto Estratega ignora los precios para examinar la "forma" (topología) del presupuesto, modelándolo como un Complejo Simplicial Abstracto. Se calculan Invariantes Topológicos (Números de Betti) para diagnosticar patologías profundas:
• β0​>1 (Estructura Fragmentada): Detecta "islas" de costos o recursos huérfanos que no aportan al ápice del proyecto, lo que se traduce en dinero desperdiciado.
• β1​>0 (Socavón Lógico): Identifica dependencias circulares (bucles infinitos de precios) que imposibilitan una auditoría o cálculo real del costo.
• Estabilidad Piramidal (Ψ): Mide si el proyecto es una "Pirámide Invertida". Un valor Ψ<1.0 alerta que miles de actividades dependen de una base de proveedores peligrosamente estrecha, elevando el riesgo de colapso logístico.
• Resonancia Espectral: Analiza el espectro del Laplaciano para predecir si el proyecto es susceptible a un "Efecto Dominó" ante fallos sincronizados en frentes de obra.

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

# 🔮 3. La Física del Valor: Termodinámica y Estocástica

**Componente:** app/financial_engine.py
El sistema trasciende la contabilidad determinista mediante el Modelo Unificado de Física del Costo, donde el riesgo financiero es una consecuencia directa de la estructura física y la temperatura del mercado:
1. Termodinámica del Costo (Tsys​): La volatilidad es tratada como calor. El sistema simula cómo la "fiebre" inflacionaria de un insumo se difunde por el grafo del presupuesto hasta el ápice del proyecto.
2. Eficiencia Exergética: Distinguimos entre Exergía (trabajo útil en estructura y cimentación) y Anergía (desperdicio o acabados cosméticos). Una eficiencia < 50% indica un edificio débil con "pintura cara".
3. Valoración Estocástica: El Oráculo de Riesgos ejecuta 10,000 Simulaciones de Monte Carlo y utiliza distribuciones Student-t para modelar "Cisnes Negros" que las hojas de cálculo tradicionales ignoran.
4. Opciones Reales: Valora la flexibilidad estratégica (ej. la opción de esperar para comprar cemento) como un activo financiero real.

## El Director: Orquestación del Pipeline
**Componente:** `app/pipeline_director.py` (Anteriormente `procesador_csv.py`)

Para evitar el "código espagueti", hemos centralizado la lógica de control. El Director no procesa datos; él da las órdenes.

## Orquestación Granular: El Pipeline como Máquina de Estados

A diferencia de los scripts lineales tradicionales, el `PipelineDirector` implementa una arquitectura de **Ejecución Atómica con Persistencia de Estado**.

*   **Atomicidad:** Cada paso (ej. `CalculateCosts`) es una unidad discreta que recibe un contexto, lo procesa y retorna un nuevo estado.
*   **Persistencia:** Entre pasos, el "Vector de Estado" se serializa (Redis/Pickle). Esto permite al Agente intervenir, reintentar un paso específico o pausar el flujo sin perder datos.
*   **Método:** `run_single_step(step_name)` permite la ejecución quirúrgica de procesos.

# ⚖️ 4. El Veredicto: Transparencia de la Caja de Cristal

**Componente:** app/semantic_translator.py
Para generar una confianza profunda, el sistema opera bajo el protocolo de la Caja de Cristal. La sabiduría emerge de una deliberación transparente:
• Risk Challenger (El Fiscal): Este agente busca contradicciones. Si un proyecto parece rentable pero es estructuralmente una pirámide invertida, emite un Veto Técnico y expone el acta de debate interno.
• Intérprete Diplomático (DIKW): Traduce los hallazgos abstractos (como β1​=3) en advertencias de negocio accionables utilizando Búsqueda Vectorial (Embeddings) para contextualizar la realidad de la obra.
• Suma de Kahan: El Matter Generator utiliza algoritmos de suma compensada para garantizar una precisión contable absoluta, eliminando errores de redondeo en presupuestos de gran escala.

**APU Filter no adivina; demuestra. Mediante el ciclo OODA (Observar, Orientar, Decidir, Actuar), valida realidades físicas y financieras para dotar de criterio a cada decisión 5. Motor de Inteligencia Financiera (Financial Engine)**

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
**Componente:** `agent/apu_agent.py`

La evolución de APU Filter introduce capacidades agénticas para coordinar tareas complejas de manera autónoma. El apu_agent actúa como un sistema nervioso central que conecta los microservicios y asegura la coherencia del flujo de trabajo.

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
│   ├── apu_agent.py            # Agente Autónomo (SRE) y orquestación
│   └── business_topology.py    # Analizador de topología de negocio (Arquitecto)
│
├── app/                        # Lógica principal de la aplicación Flask
│   ├── __init__.py
│   ├── app.py                  # Factory de la app, endpoints API
│   ├── pipeline_director.py    # Orquestador del pipeline de datos (Pipeline Matrix)
│   ├── apu_processor.py        # Motor de transformación y parsing categórico
│   ├── business_agent.py       # Agente de Negocio (CFO)
│   ├── financial_engine.py     # Motor Financiero (Oráculo de Riesgos)
│   ├── flux_condenser.py       # Motor de Física de Flujo (Guardián)
│   ├── matter_generator.py     # Generador de BOM (Alquimista)
│   ├── semantic_translator.py  # Traductor Semántico (Diplomático)
│   ├── report_parser_crudo.py  # Parser especializado
│   ├── topology_viz.py         # Visualizador de grafos
│   ├── tools_interface.py      # Interfaz de Herramientas MIC
│   ├── data_loader.py          # Capa de abstracción de datos
│   ├── schemas.py              # Esquemas de datos (Dataclasses)
│   ├── telemetry.py            # Sistema de Telemetría OODA
│   └── utils.py                # Utilidades generales
│
├── config/                     # Configuración y Reglas de Negocio
│   ├── config_app.py           # Configuración de la aplicación
│   ├── config_rules.json       # Reglas de clasificación y validación
│   ├── data_contract.yaml      # Contrato de datos y políticas
│   └── ontology.json           # Ontología de construcción
│
├── data/                       # Datos procesados y sesiones
│   └── sessions/               # Persistencia de estado de agentes
│
├── data_dirty/                 # Datos crudos de entrada
│
├── docs/                       # Documentación Técnica
│   └── images/                 # Diagramas y recursos visuales
│
├── infrastructure/             # Infraestructura de despliegue
│   ├── Dockerfile.core
│   └── Dockerfile.agent
│
├── models/                     # Modelos Matemáticos
│   └── probability_models.py   # Simulación Monte Carlo
│
├── scripts/                    # Scripts de Mantenimiento
│   ├── generate_embeddings.py
│   └── clean_csv.py
│
├── tests/                      # Suite de Pruebas
│   ├── test_app.py
│   ├── test_apu_agent.py
│   ├── test_business_topology.py
│   ├── test_financial_engine.py
│   ├── test_flux_condenser.py
│   └── ... (ver directorio completo)
│
├── requirements.in             # Dependencias fuente
├── requirements.txt            # Dependencias congeladas
└── start_conda.sh              # Script de inicio de entorno
```

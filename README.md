# APU Filter: Inteligencia de Costos y Blindaje Financiero

### De Archivos Muertos a Memoria Viva: El Nuevo Activo Estratégico de su Constructora.

**"La volatilidad de los datos de APU no es un problema de software, es un problema de negocio que cuesta millones."**

En el sector de la construcción en Colombia, ganar una licitación con el precio incorrecto es peor que perderla. **APU Filter** no es otro software de gestión; es una plataforma de **Inteligencia de Costos** diseñada para transformar el caos de sus presupuestos históricos en una ventaja competitiva, protegiendo sus márgenes de utilidad contra la incertidumbre.

---

## El Manifiesto: Por Qué Construimos Esto

Los ingenieros de costos pasan el **40% de su tiempo** limpiando hojas de cálculo rotas y el otro **40%** reinventando la rueda (cotizando ítems que la empresa ya ha comprado mil veces).

Creamos APU Filter para eliminar esa fricción. No reemplazamos a sus ingenieros; extendemos y amplificamos su capacidad para tomar decisiones. Convertimos terabytes de "archivos muertos" (PDFs, Excels viejos) en una **Memoria Institucional Viva** que responde preguntas críticas en segundos.

---

## Su Nuevo Equipo Digital

Hemos diseñado la arquitectura del sistema no como una colección de scripts, sino como un **Equipo de Expertos** especializados. Cada módulo tiene una responsabilidad única y reporta a un Director central.

### 1. El Director (`pipeline_director.py`)
**Rol:** Orquestador del Flujo.
**Misión:** Es el jefe de obra digital. No toca los materiales, pero asegura que cada especialista entre en el momento exacto. Define la secuencia: Carga -> Limpieza -> Estabilización -> Cirugía -> Estimación. Garantiza que el proceso sea ordenado y auditable.

### 2. El Guardia (`report_parser_crudo.py`)
**Rol:** Seguridad de Entrada.
**Misión:** Detiene los datos corruptos en la puerta. Analiza la estructura de los archivos entrantes (CSV, Excel) y decide si cumplen con los estándares mínimos de calidad. Si entra basura, sale basura; El Guardia asegura que solo entre materia prima viable.

### 3. El Estabilizador (`flux_condenser.py`)
**Rol:** Control de Flujo y Resiliencia.
**Misión:** Implementa patrones de **Backpressure y Control Adaptativo** para actuar como un amortiguador industrial. Gestiona la tasa de procesamiento dinámicamente, absorbiendo picos de carga y entregando un flujo estable al resto del sistema. Evita la saturación del servidor y garantiza la continuidad operativa, inspirándose internamente en modelos físicos de energía para tomar decisiones de regulación.

### 4. El Cirujano (`apu_processor.py`)
**Rol:** Precisión Estructural.
**Misión:** Con el dato ya estabilizado, el Cirujano disecciona cada línea de costo. Separa materiales, mano de obra y equipos con precisión milimétrica, normalizando unidades y descripciones para que sean comparables.

### 5. El Estratega (`estimator.py`)
**Rol:** Inteligencia de Mercado (Caja Blanca).
**Misión:** Es el cerebro consultivo. Utiliza **Búsqueda Semántica** para encontrar precios históricos ("Muro de ladrillo" ≈ "Mampostería tolete") y proyecta escenarios de riesgo. **No es una caja negra:** explica cada decisión, mostrando la evidencia y el nivel de certeza para que el ingeniero humano tenga la última palabra.

---

## Soluciones Reales a Dolores de Obra

| Dolor del Negocio | Solución Técnica | Beneficio Directo |
| :--- | :--- | :--- |
| **"Perdemos días limpiando Excels de contratistas."** | **Motor de Ingesta Resiliente** (Control Adaptativo + PID) | **Continuidad Operativa:** Procese archivos corruptos o masivos sin que el sistema se detenga. |
| **"Cada ingeniero cotiza precios diferentes para lo mismo."** | **Memoria Institucional Inteligente** (Vectores FAISS) | **Estandarización:** Recupere la "verdad" de la empresa. Si ya se cotizó, el sistema lo sabe. |
| **"Nos da miedo que el precio del acero suba y perdamos plata."** | **Radar de Riesgo Financiero** (Simulación Monte Carlo) | **Protección de Margen:** Conozca la probabilidad matemática de pérdida antes de enviar la oferta. |

---

## Ingeniería Bajo el Capó (La Validación Técnica)

Para el equipo de TI: Usamos una arquitectura modular en Python (Flask, Pandas, PyTorch/FAISS) con principios SRE (Site Reliability Engineering) para garantizar robustez.

### Configurabilidad Declarativa
Creemos que las reglas de negocio no deben estar "quemadas" en el código.
*   **Separación de Poderes:** Toda la lógica de negocio (mapeo de columnas, umbrales de validación, reglas de categorización) vive en archivos de configuración externos (`config.json`).
*   **Adaptabilidad:** ¿Cambió el formato del Excel del proveedor? Ajuste el JSON, no el código Python. Esto permite que el sistema evolucione a la velocidad del negocio sin requerir refactorización constante.

### Transparencia de Caja Blanca
En un sector donde la responsabilidad legal es alta, "la IA lo dijo" no es una defensa válida.
*   **Auditabilidad:** Cada estimación viene acompañada de un log de decisión detallado: "¿Por qué elegí este precio? Porque tiene una similitud semántica del 94% con el Proyecto X del año pasado".
*   **Control Humano:** La herramienta propone, el ingeniero dispone.

Ver `app/metodos.md` para detalles profundos sobre el Motor de Estabilidad y la Lógica de Estimación.

## Instalación y Uso

Para desplegar esta potencia en sus servidores, utilizamos una metodología de instalación por capas.

### Paso 1: La Cimentación (Conda)
Primero, instalamos la base pesada. Conda es el encargado de colocar los cimientos y el motor principal (Faiss).

```bash
# Crear el entorno
conda create --name apu_filter_env python=3.10

# Activar el entorno
conda activate apu_filter_env

# Instalar el motor principal (Faiss-cpu)
conda install -c pytorch faiss-cpu
```

### Paso 2: El Sistema Hidráulico (Pip específico)
Instalamos `torch` (versión CPU) para la inteligencia artificial sin requerir hardware costoso.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Paso 3: El Brazo Operativo (Uv)
Usamos `uv` para velocidad y eficiencia en las dependencias restantes.

```bash
# Instalar Redis
conda install -c conda-forge redis

# Instalar dependencias
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Puesta en Marcha

1.  **Generar la Inteligencia:**
    ```bash
    python scripts/generate_embeddings.py --input data/processed_apus.json
    ```

2.  **Encender Motores:**
    ```bash
    python -m flask run --port=5002
    ```

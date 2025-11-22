# APU Filter: Inteligencia de Costos y Estrategia de Licitación.

### Transforme el ruido de sus datos históricos en una señal clara de rentabilidad.

Esta herramienta no es un simple parser, sino un sistema que convierte archivos muertos en activos estratégicos, eliminando la incertidumbre en la toma de decisiones.

## El Problema: "El Dolor del Ingeniero de Costos"

Antes de hablar de la solución, entendemos el dolor diario de su operación:

*   **El caos de los formatos no estandarizados:** Luchar contra Excel, PDF y CSVs corruptos.
*   **El riesgo de depender de la memoria:** Cuando el conocimiento reside en una sola persona ("El Maestro del Excel").
*   **La frustración operativa:** Perder tiempo limpiando datos en lugar de analizar estrategias de precios.
*   **El miedo:** Ganar una licitación y perder dinero por errores de cálculo.

## La Solución: "Nuestra Refinería de Datos"

Hemos diseñado una arquitectura de procesamiento de señales que separa la señal del ruido.

### Fase 1: El Filtro de Ruido (Data Flux Condenser)

Igual que un sistema de audio de alta fidelidad elimina la estática, nuestro motor de ingesta (basado en física RLC) filtra los datos corruptos y estabiliza el flujo de entrada. Garantiza que solo entre información pura al sistema.

### Fase 2: El Cerebro Digital (APU Processor)

No es solo un lector; es un traductor. Entiende la jerarquía oculta en sus presupuestos desordenados y la estructura automáticamente, convirtiendo texto plano en relaciones de negocio.

### Fase 3: La Memoria Institucional (Búsqueda Semántica)

Recupere la experiencia de 20 años de su empresa. Nuestro motor cognitivo no busca palabras, busca conceptos. Encuentre el costo real de un 'Muro' aunque en el histórico se llame 'Pantalla', evitando reinventar la rueda.

## Por qué esto es Estratégico

*   **Eficiencia Operativa:** Reducción del 80% en tiempos de preparación de datos.
*   **Mitigación de Riesgo:** El "Radar Financiero" (Monte Carlo) permite visualizar la volatilidad antes de ofertar.
*   **Activo de Conocimiento:** Centraliza la inteligencia de la compañía, haciéndola independiente de la rotación de personal.

## Ingeniería Bajo el Capó (La Validación Técnica)

La Maquinaria que hace posible la Estrategia. Usamos tecnologías avanzadas (Python, Lark, FAISS, Pandas) para garantizar la robustez y precisión que exige el sector construcción.

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

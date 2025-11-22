# APU Filter: Capitalización de Conocimiento y Estrategia de Licitación.

### De Archivos Muertos a Memoria Viva: El Nuevo Activo Estratégico de su Constructora.

Ya no se trata solo de ahorrar tiempo. Se trata de convertir el histórico de presupuestos (archivos muertos) en el **Activo Estratégico** más valioso de la constructora: una **Memoria Viva** que aprende de cada proyecto para blindar la rentabilidad del siguiente.

## El Problema: "Fuga de Capital Intelectual"

Más allá del caos operativo, el verdadero problema es la pérdida de experiencia:

*   **Conocimiento Fragmentado:** 20 años de experiencia atrapados en miles de Excel ilegibles.
*   **Reinversión Continua:** Volver a cotizar lo que ya se analizó hace un año.
*   **Ceguera Financiera:** Ofertar basándose en promedios, ignorando la volatilidad real de los costos.

## La Solución: Traducción de Tecnología a Valor de Negocio

Hemos desplegado ingeniería avanzada para resolver desafíos de negocio específicos.

| Tecnología | Componente | Valor de Negocio |
| :--- | :--- | :--- |
| **Física RLC (Sistemas Dinámicos)** | **Data Flux Condenser** | **Motor de Ingesta a Prueba de Caos (Estabilidad).**<br>Garantiza que el sistema procese datos sucios sin detenerse. |
| **Búsqueda Semántica (Vectores)** | **Memoria Institucional** | **Memoria Institucional Inteligente.**<br>Recupera la experiencia de 20 años sin depender de códigos exactos. |
| **Simulación Monte Carlo** | **Radar de Riesgo** | **Radar de Riesgo Financiero.**<br>Visualiza la probabilidad de pérdida antes de ofertar. |

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

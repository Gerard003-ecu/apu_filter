# APU Filter: Inteligencia de Costos y Seguridad Financiera

### De Archivos Muertos a Memoria Viva: El Nuevo Activo Estratégico de su Constructora.

**"La volatilidad de los datos de APU no es un problema de software, es un problema de negocio que cuesta millones."**

Ya no se trata solo de ahorrar tiempo. Se trata de convertir el histórico de presupuestos (archivos muertos) en el **Activo Estratégico** más valioso de la constructora: una **Memoria Viva** que aprende de cada proyecto para blindar la rentabilidad del siguiente.

## El Problema: "Fuga de Capital Intelectual"

Más allá del caos operativo, el verdadero problema es la pérdida de experiencia:

*   **Conocimiento Fragmentado:** 20 años de experiencia atrapados en miles de Excel ilegibles.
*   **Reinversión Continua:** Volver a cotizar lo que ya se analizó hace un año.
*   **Ceguera Financiera:** Ofertar basándose en promedios, ignorando la volatilidad real de los costos.

## El Equipo Digital: Expertos en Costos

Hemos transformado funciones de código en un equipo de agentes especializados que trabajan para proteger sus márgenes. Conozca a su nuevo personal:

| Agente | Rol Técnico | Misión de Negocio |
| :--- | :--- | :--- |
| **El Guardia** | `ReportParserCrudo` | **Seguridad de Entrada.**<br>Detiene datos corruptos en la puerta. Diagnostica y filtra archivos defectuosos antes de que contaminen su base de datos maestra. |
| **El Estabilizador** | `DataFluxCondenser` | **Protector de Márgenes.**<br>Utiliza física RLC para absorber el "ruido" de los datos sucios, asegurando que el caos de los formatos no detenga su operación ni distorsione los precios. |
| **El Cirujano** | `APUProcessor` | **Precisión Estructural.**<br>Disecciona cada línea de costo con precisión milimétrica, separando materiales, mano de obra y equipos para un análisis granular. |
| **El Estratega** | `Estimator` | **Inteligencia de Mercado.**<br>Utiliza búsqueda semántica para encontrar precios históricos ("Muro de ladrillo" ≈ "Mampostería tolete") y sugerir el costo óptimo basado en 20 años de experiencia. |

## La Solución: Traducción de Tecnología a Valor

Ingeniería avanzada aplicada a resultados financieros concretos.

| Tecnología | Característica | Valor de Negocio ("El Por Qué") |
| :--- | :--- | :--- |
| **Física RLC** | Amortiguación de Datos | **Continuidad Operativa.**<br>Evita caídas del sistema ante archivos masivos o corruptos. Su equipo nunca deja de trabajar por "errores del sistema". |
| **Vectores (FAISS)** | Memoria Institucional | **Velocidad de Respuesta.**<br>Recupera cotizaciones de hace 10 años en milisegundos. No reinvente la rueda; use lo que ya sabe. |
| **Caja Blanca** | Trazabilidad Total | **Confianza en el Dato.**<br>El sistema explica sus decisiones ("Elegí este precio por una coincidencia del 95%"). Auditable y transparente. |

## Ingeniería Bajo el Capó (La Validación Técnica)

Para el equipo de TI: Usamos una arquitectura modular en Python (Flask, Pandas, PyTorch/FAISS) con principios SRE (Site Reliability Engineering) para garantizar robustez. Ver `app/metodos.md` para detalles profundos sobre el Motor de Física y la Lógica de Estimación.

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

# APU Filter: El Motor de Inteligencia para Licitaciones de Construcción

### Convierta sus archivos históricos en una ventaja competitiva. Deje de limpiar datos y empiece a ganar licitaciones.

Sabemos que la realidad de una oficina de licitaciones es el caos: gigabytes de Excels viejos con formatos rotos, la dependencia absoluta de la memoria del ingeniero jefe ("¿cuánto cobramos por ese muro hace dos años?"), y el riesgo constante de que un error de cálculo en una celda oculta cueste millones. APU Filter no es solo software; es el fin de la incertidumbre. Transformamos ese desorden en inteligencia accionable para que deje de perder tiempo arreglando archivos y se enfoque en la estrategia para ganar.

## Nuestros Tres Pilares de Certeza

Hemos reemplazado la complejidad técnica por soluciones diseñadas para la obra. Así es como convertimos sus problemas en ventajas:

### 1. Motor de Ingesta a Prueba de Caos
*(Antes: Condensador de Flujo)*

Su histórico de obras es desordenado. Lo entendemos. Nuestro motor actúa como una planta de tratamiento industrial que filtra automáticamente archivos corruptos, estandariza formatos antiguos y corrige errores humanos, entregando datos puros y listos para usar. Es su seguro contra la "basura" digital que sabotea los presupuestos.

### 2. Memoria Institucional Inteligente
*(Antes: Búsqueda Semántica/FAISS)*

No dependa de que el ingeniero jefe recuerde cuánto costó el concreto hace 3 años o de buscar en carpetas olvidadas. Nuestro sistema recupera la experiencia de toda la empresa instantáneamente. Entiende conceptos de construcción, no solo palabras clave. Si busca "Muro de contención", el sistema encontrará también "Pantalla de concreto" o "Muro pantalla", asegurando que toda la sabiduría acumulada de su empresa esté en la punta de sus dedos al cotizar.

### 3. Radar de Riesgo Financiero
*(Antes: Simulación Monte Carlo)*

Las licitaciones no son estáticas; los precios de los insumos fluctúan. Proyecte la volatilidad del mercado antes de ofertar. Nuestro radar le dice la probabilidad matemática de ganar o perder dinero con sus precios actuales. No lance una oferta al aire; conozca sus márgenes de seguridad con precisión estadística antes de comprometerse.

---

## Instalación y Uso: El Ensamblaje de una Grúa de Precisión

Para desplegar esta potencia en sus servidores, utilizamos una metodología de instalación por capas, similar al montaje de una grúa torre en obra. Cada herramienta tiene un propósito estructural específico para garantizar que el sistema soporte la carga de trabajo.

### Paso 1: La Cimentación (Conda)
Primero, instalamos la base pesada. Conda es el encargado de colocar los cimientos y el motor principal (Faiss) que soporta toda la estructura de inteligencia. Sin esta base sólida, la grúa no podría operar.

```bash
# Crear el terreno (Entorno)
conda create --name apu_filter_env python=3.10

# Ocupar el terreno (Activar)
conda activate apu_filter_env

# Instalar el motor principal (Faiss-cpu)
conda install -c pytorch faiss-cpu
```

### Paso 2: El Sistema Hidráulico (Pip específico)
A continuación, instalamos el sistema de potencia. Usamos una configuración especial de `pip` para traer `torch` (versión CPU). Esta es la pieza de ingeniería especializada que permite "levantar" las cargas pesadas de Inteligencia Artificial sin requerir hardware de tarjetas gráficas costosas.

```bash
# Instalar el sistema hidráulico de IA
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Paso 3: El Brazo Operativo (Uv)
Finalmente, ensamblamos el brazo de maniobra. Usamos `uv` para el resto de los componentes porque necesitamos velocidad y agilidad en la operación diaria. Es ligero, rápido y eficiente para manejar las herramientas cotidianas del sistema.

```bash
# Instalar Redis (Sistema de control de sesiones)
conda install -c conda-forge redis

# Ensamblar el resto de herramientas
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Puesta en Marcha

Una vez la grúa está armada, inicie operaciones:

1.  **Generar la Inteligencia:** Si tiene nuevos datos históricos, procese la memoria institucional.
    ```bash
    python scripts/generate_embeddings.py --input data/processed_apus.json
    ```

2.  **Encender Motores:** Inicie la plataforma web.
    ```bash
    python -m flask run --port=5002
    ```

---

## Construido por ingenieros, para ingenieros.

Entendemos el valor de su información. APU Filter funciona localmente en su infraestructura; su data estratégica nunca sale de su control ni viaja a la nube de terceros.

Es hora de construir con certeza.

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

### El Flujo de Mando Declarativo
El Director ejecuta un plan de obra estricto definido en `config.json` (Configurabilidad Declarativa):

1.  **Llamada al Guardia:** "¿El archivo cumple las reglas definidas en `parser_settings`?"
2.  **Llamada al Estabilizador:** "Ingresa los datos controlando la presión según los umbrales PID."
3.  **Llamada al Cirujano:** "Estandariza usando el `columns_mapping` configurado."
4.  **Llamada al Estratega:** "Calcula los costos."
5.  **Cierre:** "Genera el reporte final."

Esta arquitectura permite que, si mañana queremos cambiar la forma en que se limpian los datos, solo ajustamos el archivo de configuración JSON, sin tocar una sola línea de código Python.

#### Mecanismos de Defensa (SRE)
Esta no es una metáfora decorativa. Utilizamos lógica de sistemas dinámicos para proteger la infraestructura:

*   **Load Shedding (Disyuntor Térmico):** Si la "fricción" (error rate o complejidad) supera un umbral de seguridad (> 50W de potencia disipada equivalente), el sistema activa un freno de emergencia, reduciendo drásticamente la carga para "enfriar" el proceso.
*   **Anti-Windup:** Evita que el controlador PID se quede "pegado" tratando de corregir errores acumulados pasados, manteniendo la respuesta ágil ante cambios presentes.
*   **Recuperación Parcial:** Si un lote de datos está corrupto, el sistema lo aísla y continúa con el resto del archivo, en lugar de fallar todo el proceso.

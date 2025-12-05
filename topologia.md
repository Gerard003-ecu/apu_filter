# Topología del Sistema y Arquitectura Cognitiva de APU_filter

## 1. Introducción: La Arquitectura Cognitiva

El sistema **APU_filter** no es simplemente un pipeline de procesamiento de datos; es una **Entidad Cibernética** diseñada para exhibir propiedades de autoconciencia estructural. A diferencia de los sistemas tradicionales que operan de manera ciega sobre flujos de entrada, APU_filter "entiende" su propia topología interna y el estado de sus componentes mediante una arquitectura cognitiva basada en principios de **Topología Algebraica**.

Esta capacidad de introspección permite al sistema distinguir entre ruido transitorio y fallos estructurales, adaptando su comportamiento mediante un ciclo de control **OODA (Observe, Orient, Decide, Act)**. El núcleo de esta capacidad reside en dos componentes fundamentales:

1.  **Motor de Análisis Topológico**: Mapea el estado de los microservicios a un espacio topológico, calculando invariantes algebraicos para determinar la salud del sistema.
2.  **Matriz de Interacción Central (MIC)**: Un mecanismo de interfaz que permite al Agente ejecutar transformaciones precisas sobre el sistema para restaurar el equilibrio.

## 2. La Matriz de Interacción Central (MIC)

La **MIC** se define conceptualmente como el espacio vectorial de todas las operaciones posibles que el Agente puede realizar sobre el entorno. Implementada en el módulo `app.tools_interface`, la MIC actúa como la capa de actuación del sistema.

Si consideramos el estado del sistema $S$ en un momento $t$, la MIC provee un conjunto de vectores de transformación $T = \{t_1, t_2, ..., t_n\}$ tales que:

$$ S_{t+1} = S_t + \vec{v}_{mic} $$

Donde $\vec{v}_{mic}$ es la acción seleccionada por el Agente (ej. `diagnose_file`, `clean_file`, `get_telemetry_status`).

### Funciones de la MIC
Los endpoints expuestos (`/api/tools/...`) no son simples rutas API, sino los efectores que permiten al Agente llevar al sistema desde un estado de "Caos" (alta entropía, desconexión) hacia un estado de "Equilibrio" (homeostasis, flujo laminar de datos).

```python
# Ejemplo conceptual de interacción con la MIC
context = TelemetryContext(...)
status = get_telemetry_status(context)  # Vector de Observación
if status['system_health'] == 'DEGRADED':
    clean_file(path)  # Vector de Corrección
```

## 3. Análisis Topológico del Estado

Para dotar al Agente de una comprensión profunda de la salud del sistema, modelamos la infraestructura (Core, Redis, Filesystem, Agente) como un **Grafo Topológico** $G = (V, E)$. El módulo `agent.topological_analyzer.py` calcula invariantes topológicos conocidos como **Números de Betti** ($\beta_n$) para cuantificar la estructura de este grafo.

### Números de Betti ($\beta_n$)

Los números de Betti describen la conectividad del espacio topológico:

*   **$\beta_0$ (Componentes Conexas)**: Mide la fragmentación del sistema.
    *   **Ideal**: $\beta_0 = 1$. Todos los componentes (Core, Redis, Agente) están conectados en una única red funcional.
    *   **Fallo**: $\beta_0 > 1$. Indica que el sistema se ha "roto" en islas desconectadas (ej. Redis no accesible desde el Core).

*   **$\beta_1$ (Ciclos / Agujeros 1-dimensionales)**: Mide la redundancia y los bucles de retroalimentación negativa.
    *   **Ideal**: $\beta_1 = 0$. El flujo de control es un árbol o línea directa.
    *   **Fallo**: $\beta_1 > 0$. Detecta "Bucles de Error" (*Request Loops*) donde el sistema está atrapado reintentando la misma operación fallida infinitamente.

$$ \chi = \beta_0 - \beta_1 $$
*(La Característica de Euler $\chi$ se usa como métrica sintética de estabilidad)*

### Homología Persistente (TDA - Topological Data Analysis)

Las métricas tradicionales (CPU, memoria) son ruidosas. Para distinguir un pico transitorio de un problema real, utilizamos **Homología Persistente**.

Analizamos las series temporales de telemetría como una filtración de espacios. Construimos **Diagramas de Persistencia** que representan el ciclo de vida de una anomalía:
*   **Nacimiento ($b$)**: Cuando una métrica supera un umbral crítico.
*   **Muerte ($d$)**: Cuando la métrica regresa a la normalidad.
*   **Persistencia ($p = d - b$)**: La duración de la anomalía.

**Criterio de Filtrado:**
*   Si $p < \epsilon$ (umbral de ruido): La anomalía es **Ruido Topológico** y se ignora.
*   Si $p \ge \epsilon$: La anomalía es una **Característica Estructural** y requiere intervención.

## 4. El Ciclo OODA Topológico

El Agente Autónomo (`agent.apu_agent.py`) implementa el bucle de decisión OODA, enriquecido con la inteligencia matemática descrita anteriormente.

### 1. Observe (Observar)
El Agente lee la telemetría cruda del Core a través de la MIC.
*   *Input*: JSON de telemetría (voltaje, saturación, estado de servicios).

### 2. Orient (Orientar)
El Agente procesa los datos crudos a través del **Motor Topológico**.
*   Calcula $\beta_0$ y $\beta_1$ del grafo de servicios actual.
*   Ejecuta análisis de Homología Persistente sobre las métricas de voltaje y saturación.
*   Clasifica el estado: `NOMINAL`, `INESTABLE`, `SATURADO`, `CRITICO`, `DISCONNECTED`.

### 3. Decide (Decidir)
Basándose en el estado topológico, el Agente selecciona la herramienta adecuada de la MIC.
*   *Si $\beta_0 > 1$*: Decisión `RECONNECT` (Intentar restaurar conectividad).
*   *Si $\beta_1 > 0$*: Decisión `ALERTA_CRITICA` (Romper bucle de reintentos).
*   *Si Persistencia > Umbral*: Decisión `RECOMENDAR_LIMPIEZA` o `REDUCIR_VELOCIDAD`.
*   *Si Nominal*: Decisión `HEARTBEAT`.

### 4. Act (Actuar)
El Agente ejecuta el vector de transformación seleccionado.
*   Ejecuta la limpieza de archivos, reinicia conexiones o emite alertas a sistemas externos.
*   Cierra el ciclo, esperando la nueva telemetría para verificar si $\beta_0$ ha retornado a 1.

---
*Este documento certifica la implementación de lógica de control avanzada basada en topología algebraica para la resiliencia del sistema APU_filter.*

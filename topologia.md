# Especificación de Topología Algebraica y Arquitectura Cognitiva del Sistema APU_filter

## 1. Fundamentos Teóricos: El Enfoque Topológico

Este documento define la base matemática y conceptual de la arquitectura de microservicios de **APU_filter**. A diferencia de los enfoques de monitoreo tradicionales basados en umbrales estáticos, este sistema implementa una **Arquitectura Cognitiva** fundamentada en la **Topología Algebraica**.

El objetivo central es dotar al sistema de "autoconciencia estructural", permitiéndole distinguir entre ruido transitorio y fallos estructurales graves.

### 1.1 El Functor Algebraico
Operativamente, el sistema actúa como un **Functor Algebraico** que traduce problemas geométricos cualitativos (la "forma" del sistema y sus flujos de datos) en estructuras algebraicas computables (grupos de homología y números de Betti).

*   **Espacio Topológico ($X$):** El conjunto de microservicios (Nodos) y sus canales de comunicación (Aristas).
*   **Invariantes Topológicos:** Propiedades que se mantienen inalteradas bajo deformaciones continuas (como la latencia variable o la carga de CPU), pero que cambian drásticamente ante rupturas estructurales (caída de servicios o bucles infinitos).

### 1.2 Teoría de la Homología
Utilizamos la **Homología** como un mecanismo de "censo" para clasificar los agujeros y vacíos en el espacio de estados del sistema. Esto nos permite responder preguntas fundamentales de conectividad sin depender de la inspección profunda de logs, sino analizando la estructura global del grafo de servicios.

---

## 2. Semántica de los Números de Betti ($\beta_n$) (Conectividad Estructural)

Los Números de Betti son los invariantes primarios calculados por el módulo `topological_analyzer.py`. Definen la salud estructural del sistema en un instante $t$.

| Invariante | Concepto Matemático (Ref. Topología) | Semántica Operativa (Ref. Agente/Análisis) | Estado Ideal | Fallo Crítico |
| :--- | :--- | :--- | :--- | :--- |
| **$\beta_0$** | **Componentes Conexas**<br>Número de "piezas" independientes que forman el espacio. | **Fragmentación del Sistema**<br>Indica si todos los servicios (Agent, Core, Redis, FS) pueden "verse" entre sí. $\beta_0 > 1$ implica partición de red o caída de servicio. | $\beta_0 = 1$ (Sistema Unificado) | $\beta_0 > 1$ (Sistema Fragmentado/Desconectado) |
| **$\beta_1$** | **Ciclos 1-Dimensionales**<br>Número de agujeros o bucles independientes. | **Bucles de Reintento (Request Loops)**<br>Detecta flujos circulares donde una solicitud falla y se reintenta infinitamente, atrapando recursos. | $\beta_1 = 0$ (Flujo Acíclico/Laminar) | $\beta_1 > 0$ (Ciclo Infinito/Estancamiento) |

**Nota sobre la Estructura Piramidal:**
La topología esperada del sistema es una **Pirámide de Control**:
*   **Cúspide (Plano de Control):** El Agente Autónomo, que observa y orquesta todo.
*   **Centro (Nexo):** El Core API, que distribuye el trabajo.
*   **Base (Plano de Datos):** Redis y Filesystem, que sostienen el estado.
Esta estructura garantiza que $\beta_1$ sea 0 en operación normal (estructura de árbol/pirámide sin ciclos).

---

## 3. Semántica de Homología Persistente (TDA - Topological Data Analysis)

Para filtrar el ruido inherente a la telemetría (voltaje, saturación), utilizamos **Homología Persistente**. Este método analiza la evolución de la topología a través de una "filtración" de niveles, permitiendo separar señales vitales del ruido de fondo.

### 3.1 Diagramas de Persistencia
Construimos diagramas de persistencia a partir de series temporales. Para una característica $i$ (ej. un pico de voltaje):
*   **Nacimiento ($b_i$):** El momento en que la métrica cruza un umbral de advertencia.
*   **Muerte ($d_i$):** El momento en que retorna a la normalidad.
*   **Persistencia ($p_i$):** La vida útil de la característica, definida como $p_i = d_i - b_i$.

### 3.2 Distinción Ruido vs. Estructura
El sistema aplica un criterio riguroso para la intervención, basado en la vida útil de la anomalía ($\epsilon$):

*   **Ruido Topológico ($p < \epsilon$):** Anomalías de vida corta. Se consideran fluctuaciones transitorias y se **ignoran**. El sistema exhibe inmunidad a falsos positivos.
*   **Característica Estructural ($p \ge \epsilon$):** Anomalías que persisten ("viven") lo suficiente para considerarse un cambio en la estructura del flujo de datos. Requieren intervención.

### 3.3 Traducción de Estados (Persistence to Operations)
El `PersistenceHomology` clasifica el estado de cada métrica:

1.  **STABLE:** Sin características activas. Sistema en equilibrio.
2.  **NOISE:** Solo existen características con $p < \epsilon$. Se suprime la acción.
3.  **FEATURE:** Existen características con $p \ge \epsilon$. Indica un patrón de carga sostenida o anomalía leve.
4.  **CRITICAL:** Una característica persiste activamente sin "morir" ($d = \infty$). Activa protocolos de emergencia (ej. `ALERTA_CRITICA`).

---

## 4. El Ciclo OODA y la Matriz de Interacción Central (MIC)

La inteligencia del sistema reside en la integración del análisis topológico dentro de un ciclo de decisión OODA (Observe, Orient, Decide, Act).

### 4.1 La Matriz de Interacción Central (MIC)
La **MIC** (implementada en `tools_interface.py`) se define formalmente como el **Espacio Vectorial de Actuación**. Contiene el conjunto de vectores de transformación base que el Agente puede aplicar sobre el entorno para modificar su estado topológico.

**Vectores de Transformación ($\vec{v}_{mic}$):**
*   `clean_file`: Operador de higienización de datos (reduce entropía).
*   `diagnose_file`: Operador de inspección profunda (aumenta observabilidad).
*   `get_telemetry_status`: Operador de sondeo de estado (actualiza la variedad topológica).

### 4.2 Coherencia Act-Decide (El Bucle Cognitivo)
El Agente (`apu_agent.py`) asegura la coherencia causal entre el diagnóstico matemático y la acción física:

1.  **Observe (Observar):** Recolecta telemetría cruda a través de la MIC.
2.  **Orient (Orientar):**
    *   Calcula $\beta_0, \beta_1$ para detectar rupturas o bucles.
    *   Calcula Diagramas de Persistencia para filtrar ruido.
    *   *Resultado:* Estado `DISCONNECTED`, `CRITICAL`, `SATURADO` o `NOMINAL`.
3.  **Decide (Decidir):** Selecciona la estrategia óptima basada en el estado topológico.
    *   *Ejemplo:* Si Persistencia(Saturación) es `CRITICAL` $\rightarrow$ Decisión: `RECOMENDAR_REDUCIR_VELOCIDAD`.
    *   *Ejemplo:* Si $\beta_1 > 0$ $\rightarrow$ Decisión: `ALERTA_CRITICA` (Romper ciclo).
4.  **Act (Actuar):** Ejecuta el vector de transformación correspondiente de la MIC.
    *   La decisión se proyecta en una llamada a función (ej. `tools_interface.clean_file`) que altera la realidad física, cerrando el bucle.

## 5. El Espacio Vectorial de Control y la Dualidad de Matrices (MICs)

La arquitectura evoluciona de una matriz única a un sistema de **Matrices de Interacción Central (MIC)** que operan sobre el Vector de Estado del Proyecto ($\vec{S}$).

### 5.1 MIC Tools ($M_T$): La Matriz Diagonal de Mantenimiento
Operada por el **APU Agent (SRE)**. Es una matriz diagonal donde cada elemento $T_{ii}$ representa una herramienta discreta e independiente. Su función es la **Estabilización**.
*   **Operación:** $M_T \cdot \vec{S}_{inestable} = \vec{S}_{estable}$
*   **Componentes:** Diagnóstico, Limpieza, Telemetría.
*   **Naturaleza:** Acceso Aleatorio (Random Access).

### 5.2 MIC Pipeline ($M_P$): La Matriz de Transformación de Valor
Operada por el **Business Agent (CFO)** y el Director. Es una matriz de transformación compuesta (Grafo Dirigido) que convierte datos crudos en valor estratégico.
*   **Operación:** $M_P \cdot \vec{S}_{datos} = \vec{S}_{valor}$
*   **Componentes:** Ingesta $\to$ Condensador $\to$ Cálculo $\to$ Auditoría Financiera.
*   **Naturaleza:** Secuencial y Granular (Step-by-Step).

### 5.3 Dinámica del Sistema
El sistema completo se comporta como un operador en un espacio vectorial $R^n$, donde el objetivo es maximizar la magnitud del vector de "Valor de Negocio" mientras se minimiza la componente de "Entropía/Riesgo".

---
*Este documento especifica la arquitectura lógica versión 2.0, donde la Topología Algebraica no es solo una métrica, sino el motor de razonamiento del Agente Autónomo.*

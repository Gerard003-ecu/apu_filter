# Teoría de Circuitos RLC en el Motor de Física de Datos

## Introducción

El **Motor de Física de Datos** (`FluxPhysicsEngine`) implementa un modelo basado en circuitos eléctricos RLC (Resistencia, Inductancia, Capacitancia) para analizar y cuantificar la calidad y estabilidad del flujo de datos en el pipeline. Este enfoque permite traducir métricas abstractas de procesamiento en variables físicas tangibles (Energía, Voltaje, Potencia) para un diagnóstico más intuitivo y robusto.

## Conceptos Fundamentales

### 1. Capacitancia ($C$) y Energía Potencial ($E_c$)
En nuestro modelo, la **Capacitancia** representa la capacidad del sistema para "almacenar" o amortiguar la presión de los datos entrantes (volumen).

*   **Analogía:** Un tanque de agua o un buffer de memoria.
*   **Variable de Estado:** Saturación de Voltaje ($V$), que representa qué tan lleno está el sistema relativo a su capacidad.
*   **Ecuación de Energía:**
    $$ E_c = \frac{1}{2} C V^2 $$
*   **Interpretación:** Una alta energía potencial ($E_c$) indica una alta presión de datos ("Data Pressure"). Si supera los límites, el sistema corre riesgo de "desbordamiento" o latencia excesiva.

### 2. Inductancia ($L$) y Energía Cinética ($E_l$)
La **Inductancia** representa la inercia del flujo de datos, o su resistencia a cambios bruscos en la calidad (corriente).

*   **Analogía:** Un volante de inercia que mantiene el flujo constante.
*   **Variable de Estado:** Corriente ($I$), que representa la "calidad" del flujo (proporción de datos limpios/caché hits).
*   **Ecuación de Energía:**
    $$ E_l = \frac{1}{2} L I^2 $$
*   **Interpretación:** La energía cinética ($E_l$) representa la "Inercia de Calidad". Un flujo con alta inercia (alta calidad constante) es difícil de perturbar. Si $E_l$ es muy baja, el sistema está "estancado" o el flujo es errático.

### 3. Resistencia ($R$) y Potencia Disipada ($P$)
La **Resistencia** representa la fricción o dificultad inherente al procesar datos "sucios" o complejos.

*   **Analogía:** Fricción en una tubería o calor en un cable.
*   **Variable de Estado:** Corriente de Ruido ($I_{ruido} = 1 - I$).
*   **Ecuación de Potencia:**
    $$ P_{disipada} = I_{ruido}^2 \cdot R_{dinamica} $$
*   **Interpretación:** La potencia disipada representa el "calor" o desperdicio de recursos generado por datos de mala calidad. El sistema monitorea esto para evitar "sobrecalentamiento" (uso excesivo de CPU/recursos).

## Fenómenos Transitorios

### Voltaje Flyback ($V_{flyback}$)
En un circuito inductivo, un cambio brusco en la corriente genera un pico de voltaje inverso. En nuestro sistema, esto ocurre cuando la calidad de los datos cae repentinamente.

*   **Ecuación:**
    $$ V_L = L \frac{di}{dt} $$
*   **Interpretación:** Un "Pico Inductivo" o Flyback indica una inestabilidad severa. El sistema detecta estos picos para anticipar fallos en el procesamiento antes de que saturen el buffer.

## Diagnóstico Energético

El sistema evalúa la salud del flujo basándose en el balance de estas energías:

| Estado | Condición Física | Significado |
| :--- | :--- | :--- |
| **🟢 EQUILIBRIO** | Balance $E_c / E_l$ normal | Flujo estable y laminar. |
| **🟡 BAJA INERCIA** | $E_l < Umbral$ | Flujo débil o intermitente. |
| **🟠 SOBRECARGA** | $E_c \gg E_l$ | Presión excesiva, riesgo de cuello de botella. |
| **⚡ PICO INDUCTIVO** | $V_{flyback} > Umbral$ | Caída súbita de calidad, inestabilidad crítica. |
| **🔥 SOBRECALENTAMIENTO** | $P_{disipada} > Umbral$ | Datos muy sucios consumiendo recursos excesivos. |

Este modelo físico permite al `DataFluxCondenser` tomar decisiones de control (como activar el freno de emergencia o ajustar el PID) basándose en la "física" real del procesamiento de datos.

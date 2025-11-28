# Teoría de Control: Controlador PI Discreto en DataFluxCondenser

## Introducción

El `DataFluxCondenser` utiliza un **Controlador PI (Proporcional-Integral) Discreto** para regular dinámicamente el tamaño del lote de procesamiento (*batch size*). El objetivo es mantener el sistema en un punto de operación óptimo, maximizando el throughput sin saturar los recursos ni comprometer la estabilidad.

## Diagrama de Bloques

```mermaid
graph LR
    SP(Setpoint: Saturación Objetivo) --> Sum((+ / -))
    PV(Process Variable: Saturación Actual) --> Sum
    Sum --> Error(Error e[k])
    Error --> P(Proporcional Kp)
    Error --> I(Integral Ki)
    P --> Sum2((+))
    I --> Sum2
    Sum2 --> Sat[Limitador / Anti-Windup]
    Sat --> u(Señal de Control: Batch Size)
    u --> Plant(Proceso de Ingesta)
    Plant --> PV
```

## Ecuación de Control

Implementamos la forma **Posicional Discreta** del algoritmo PI:

$$ u(k) = u_{base} + \underbrace{K_p \cdot e(k)}_{\text{P}} + \underbrace{K_i \sum_{i=0}^{k} e(i) \cdot \Delta t}_{\text{I}} $$

Donde:
*   $u(k)$: Señal de control (Nuevo tamaño de batch).
*   $u_{base}$: Salida base (punto medio del rango operativo).
*   $e(k) = SP - PV(k)$: Error de control (Diferencia entre saturación deseada y real).
*   $K_p$: Ganancia Proporcional.
*   $K_i$: Ganancia Integral.
*   $\Delta t$: Tiempo transcurrido entre iteraciones.

## Componentes del Controlador

### 1. Variable de Proceso (PV) y Setpoint (SP)
*   **PV (Process Variable):** La **Saturación** del sistema ($0.0 - 1.0$), calculada por el Motor de Física ($V_{carga}$). Representa qué tan "estresado" está el sistema con el lote actual.
*   **SP (Setpoint):** El objetivo de saturación, configurado por defecto en **0.30 (30%)**. Buscamos mantener el sistema operando al 30% de su capacidad "física" teórica para dejar margen de maniobra (headroom) ante picos de datos.

### 2. Acción Proporcional (P)
$$ P = K_p \cdot e(k) $$
Responde al error *actual*. Si la saturación está lejos del objetivo, la acción P ajusta agresivamente el tamaño del batch.
*   **Efecto:** Proporciona la respuesta rápida del sistema.

### 3. Acción Integral (I)
$$ I = K_i \cdot \int e(t) dt \approx K_i \sum e(k) \Delta t $$
Responde a la historia del error acumulado. Elimina el error en estado estacionario que la acción P por sí sola no puede corregir.
*   **Efecto:** Asegura que, a largo plazo, la saturación promedio converja exactamente al Setpoint.

## Mecanismos de Robustez

### Anti-Windup (Clamping)
En sistemas reales, los actuadores tienen límites físicos (tamaño mínimo y máximo de batch). Si el controlador intenta exceder estos límites, el término integral puede acumularse indefinidamente ("Windup"), causando sobrepasos severos cuando el error cambia de signo.

**Solución Implementada:**
Limitamos (clampeamos) el valor del término integral:
$$ I_{clamped} = \max(-Lym, \min(Lym, I_{calculado})) $$
Donde $Lym$ es un límite configurado proporcional al rango de salida. Esto mantiene el controlador "despierto" y reactivo.

### Protección de Delta Time ($\Delta t$)
El tiempo entre lotes puede variar. Si $\Delta t$ es muy grande (e.g., pausa en el proceso) o cero, puede desestabilizar la integral.
*   **Validación:** Se fuerza un $\Delta t$ mínimo y se descartan valores anómalos (> 1 hora) para evitar "saltos" en el término integral.

### Freno de Emergencia (Override)
Aunque no es parte estricta del PID, el sistema incluye un "Diodo de Rueda Libre" térmico. Si la **Potencia Disipada** (calculada por el motor físico) excede un umbral crítico, se ignora la salida del PID y se fuerza una reducción drástica del batch size. Esto actúa como un disyuntor de seguridad.

## Ajuste (Tuning)

Los parámetros por defecto han sido ajustados empíricamente para flujos de datos CSV típicos:
*   $K_p = 2000.0$: Ganancia alta para respuesta rápida.
*   $K_i = 100.0$: Ganancia integral moderada para corrección de error sin oscilaciones excesivas.
*   $SP = 0.30$: Punto de operación conservador para alta estabilidad.

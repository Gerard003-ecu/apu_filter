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

## Simulación Numérica (MATLAB / Octave)

El siguiente script de MATLAB/Octave replica la lógica exacta implementada en el `FluxPhysicsEngine` (Python), permitiendo verificar el comportamiento del modelo físico bajo condiciones controladas.

```matlab
% Simulación de Motor de Física de Flujo (RLC Energético)
% Replica la lógica de 'app/flux_condenser.py'
clear; clc;

% =========================================================================
% 1. Configuración de Constantes del Sistema (SystemConstants)
% =========================================================================
% Parámetros Físicos Base
C = 5000.0;          % Capacitancia (Faradios)
R_base = 10.0;       % Resistencia Base (Ohmios)
L = 2.0;             % Inductancia (Henrios)

% Factores de Ajuste
COMPLEXITY_RESISTANCE_FACTOR = 5.0;  % Aumento de R por complejidad
MAX_FLYBACK_VOLTAGE = 10.0;          % Límite de tensión inductiva (V)

% Entradas de Simulación (Datos del Batch)
total_records = 500;                 % Tamaño del lote
cache_hits = 450;                    % Aciertos en caché (Calidad)
last_current = 0.8;                  % Corriente anterior (t-1)
dt = 0.1;                            % Delta tiempo (s)

% =========================================================================
% 2. Cálculo de Variables de Estado
% =========================================================================

% A. Calidad del Flujo (Corriente I)
% Rango: [0.0, 1.0]
current_I = cache_hits / total_records;

% B. Factor de Complejidad (Ruido)
complexity = 1.0 - current_I;

% C. Resistencia Dinámica (R_dyn)
% Aumenta con la complejidad de los datos
R_dyn = R_base * (1.0 + complexity * COMPLEXITY_RESISTANCE_FACTOR);

% =========================================================================
% 3. Ecuaciones del Circuito RLC (Amortiguamiento)
% =========================================================================

% D. Frecuencia Natural (omega_n)
omega_n = 1.0 / sqrt(L * C);

% E. Constante de Amortiguamiento (zeta)
% Para RLC serie: zeta = (R / 2) * sqrt(C / L)
damping_ratio = (R_dyn / 2.0) * sqrt(C / L);

% F. Frecuencia Amortiguada (omega_d)
if damping_ratio < 1.0
    % Sistema Subamortiguado
    omega_d = omega_n * sqrt(1.0 - damping_ratio^2);
else
    % Sistema Sobreamortiguado o Crítico
    omega_d = 0.0;
end

% =========================================================================
% 4. Cálculo de Saturación (Respuesta al Escalón)
% =========================================================================

% Tiempo normalizado por constante de tiempo (tau = R*C)
t_normalized = double(total_records) / (R_dyn * C);

if damping_ratio < 1.0
    % Subamortiguado: Oscila antes de estabilizarse
    exp_term = exp(-damping_ratio * omega_n * t_normalized);
    sin_term = sin(omega_d * t_normalized + atan2(omega_d, damping_ratio * omega_n));
    saturation_V = 1.0 - (exp_term * sin_term / sqrt(1 - damping_ratio^2));

elseif abs(damping_ratio - 1.0) < 1e-6
    % Críticamente Amortiguado: Convergencia más rápida sin oscilación
    exp_term = exp(-omega_n * t_normalized);
    saturation_V = 1.0 - (1.0 + omega_n * t_normalized) * exp_term;

else
    % Sobreamortiguado: Convergencia lenta
    s1 = -omega_n * (damping_ratio - sqrt(damping_ratio^2 - 1));
    s2 = -omega_n * (damping_ratio + sqrt(damping_ratio^2 - 1));
    A = s2 / (s2 - s1);
    B = s1 / (s1 - s2);
    saturation_V = 1.0 - (A * exp(s1 * t_normalized) + B * exp(s2 * t_normalized));
end

% Clamping de saturación [0, 1]
saturation_V = max(0.0, min(1.0, saturation_V));

% =========================================================================
% 5. Cálculo de Energías y Potencia
% =========================================================================

% G. Energía Potencial (Normalizada)
% Ec = 0.5 * C * V^2 / C -> 0.5 * V^2
E_c = 0.5 * (saturation_V^2);

% H. Energía Cinética (Normalizada)
% El = 0.5 * L * I^2 / L -> 0.5 * I^2
E_l = 0.5 * (current_I^2);

% I. Potencia Disipada
% P = I_ruido^2 * R
P_diss = (complexity^2) * R_dyn;

% J. Tensión de Flyback (Inducción)
% V_L = L * di/dt
di_dt = (current_I - last_current) / dt;
V_flyback = abs(L * di_dt);
V_flyback = min(V_flyback, MAX_FLYBACK_VOLTAGE);

% =========================================================================
% 6. Reporte de Resultados
% =========================================================================
fprintf('\n--- Resultados de Simulación Física ---\n');
fprintf('Entrada: %d registros, %d hits\n', total_records, cache_hits);
fprintf('---------------------------------------\n');
fprintf('Parámetros RLC:\n');
fprintf('  R_dyn (Resistencia Dinámica): %.4f Ohmios\n', R_dyn);
fprintf('  Zeta (Amortiguamiento):       %.4f\n', damping_ratio);
fprintf('  Omega_n (Freq. Natural):      %.4f rad/s\n', omega_n);
fprintf('\nVariables de Estado:\n');
fprintf('  Corriente (Calidad):          %.2f (%.1f%%)\n', current_I, current_I*100);
fprintf('  Saturación (Voltaje):         %.4f\n', saturation_V);
fprintf('\nEnergía y Potencia:\n');
fprintf('  Energía Potencial (Ec):       %.4e J (norm)\n', E_c);
fprintf('  Energía Cinética (El):        %.4e J (norm)\n', E_l);
fprintf('  Potencia Disipada (Pdiss):    %.4f W\n', P_diss);
fprintf('  Voltaje Flyback:              %.4f V\n', V_flyback);
fprintf('---------------------------------------\n');

% Diagnóstico Simple
if P_diss > 50.0
    fprintf('STATUS: [CRITICAL] SOBRECALENTAMIENTO\n');
elseif E_l < 0.1
    fprintf('STATUS: [WARNING] BAJA INERCIA\n');
else
    fprintf('STATUS: [OK] SISTEMA ESTABLE\n');
end
```

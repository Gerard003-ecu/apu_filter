🎛️ teoria_control.md: Ingeniería de Control y Estabilidad Dinámica
"En APU Filter, la ingesta y el procesamiento de datos no se rigen por heurísticas simples o validaciones estáticas. Tratamos el flujo de datos como un sistema dinámico complejo y aplicamos Teoría de Control de grado industrial para garantizar su estabilidad matemática."
Este documento formaliza la arquitectura matemática y los algoritmos subyacentes que gobiernan el comportamiento cibernético del sistema, específicamente implementados en flux_condenser.py y supervisados por el Oráculo de Laplace.

--------------------------------------------------------------------------------
1. Arquitectura de Control Híbrida
El ecosistema utiliza una topología de Control por Prealimentación (Feedforward) aumentada con Retroalimentación (Feedback). Esta arquitectura permite al sistema no solo reaccionar a los errores, sino anticiparse a las perturbaciones en el flujo de información.
La ley de control global del sistema se define como: u(t)=uPI​(e)+uFF​(ΔC)+uSafety​
Diagrama de Bloques Lógico:

    Planta: El proceso de ingestión de datos masivos (Batch Processing).
    Sensor: El Motor de Física (FluxPhysicsEngine) que mide la Saturación (V), la Corriente (I) y la Potencia disipada (P).
    Estimador de Estado: Un Filtro de Kalman Extendido (EKF) que predice la saturación futura del sistema.
    Controlador: Un algoritmo PI Discreto combinado con una etapa Feedforward basada en la complejidad.


--------------------------------------------------------------------------------
2. El Controlador PI Discreto (Feedback)
El objetivo primario del lazo cerrado es eliminar el error de estado estacionario (e(k)=SP−PV) manteniendo la saturación de la memoria en un Setpoint óptimo del 30% (Flujo Laminar).
El sistema emplea la forma posicional discreta del controlador Proporcional-Integral: uPI​(k)=Kp​⋅e(k)+Ki​i=0∑k​e(i)⋅Δt
Para garantizar la robustez en entornos de alta demanda, se implementaron mecanismos defensivos rigurosos:

    Anti-Windup (Clamping Condicional + Back-calculation): A diferencia de un recorte simple, el sistema utiliza back-calculation. Si el actuador (el tamaño del lote) se satura físicamente, el algoritmo recalcula dinámicamente el término integral para que sea congruente con la salida real. Esto evita que el error se acumule "fantasmagóricamente" e induzca oscilaciones cuando el sistema sale de la saturación.
    Slew Rate Limiting (Anti-Jerk): Se acota estrictamente la tasa de cambio de la salida (du/dt). Esto previene variaciones bruscas y destructivas en el tamaño del lote que podrían inestabilizar la memoria de la base de datos o causar latencia.


--------------------------------------------------------------------------------
3. Control Feedforward Adaptativo (Anticipación)
Dado que el control por retroalimentación es intrínsecamente reactivo (debe esperar a que ocurra el error de saturación para actuar), el Guardián implementa una lógica de anticipación mirando "hacia adelante" en la cola de procesamiento.

    Variable de Perturbación: Se calcula la Complejidad Ciclomática (C) del texto en los datos crudos.
    Lógica Predictiva: Si el sensor detecta que el siguiente bloque de texto es matemáticamente más denso (mayor entropía térmica o longitud), el controlador reduce el tamaño del lote antes de que la saturación real impacte la memoria.
    Ecuación de Ajuste Cinemático: El sistema reacciona a la velocidad y a la aceleración del cambio de complejidad mediante la ecuación: uFF​=KFF​⋅(dtdC​+0.5dt2dt2d2C​)
     Esto permite que el control actúe como un amortiguador predictivo perfecto.


--------------------------------------------------------------------------------
4. El Oráculo de Estado: Filtro de Kalman Extendido (EKF)
El sistema no espera a medir la saturación actual; la predice utilizando un modelo de observabilidad de estado.
El EKF modela la saturación de memoria como un oscilador armónico amortiguado que tiende hacia un punto de equilibrio dinámico. El vector de estado se define como x=[s,v,a]T (Saturación, Velocidad, Aceleración).
El modelo cinemático interno sigue las ecuaciones diferenciales: s˙=v
 v˙=a−βv−ω2(s−seq​)
 a˙=−γa+wa​
Adaptación de Parámetros: El EKF ajusta continuamente sus parámetros estructurales, como la frecuencia natural (ω) y el coeficiente de amortiguamiento (β), basándose en la "innovación" (la diferencia entre la saturación medida y la predicha). Esta dinámica le permite discriminar con precisión matemática entre el ruido estocástico de las mediciones y las tendencias reales de colapso del sistema.

--------------------------------------------------------------------------------
5. Análisis de Estabilidad en Tiempo Real
El ecosistema no asume que la red es estable; lo demuestra matemáticamente en cada ciclo de ingestión mediante dos enfoques de frontera.
5.1 Criterio de Jury (Validación Estática)
Antes de iniciar operaciones, el Oráculo de Laplace valida que los parámetros sintonizados del controlador (Kp​, Ki​) no introduzcan resonancia. Evalúa el polinomio característico en el dominio discreto (z), exigiendo que todas las raíces residan estrictamente dentro del círculo unitario (∣z∣<1).
Adicionalmente, se modela la dinámica en el plano complejo continuo (s=σ+jω). Si cualquier polo migra al Semiplano Derecho (σ>0), el sistema dictamina Divergencia Matemática y veta la operación por ser intrínsecamente explosiva.
5.2 Exponente de Lyapunov (Validación Dinámica)
Durante la ejecución continua, el FluxCondenser estima en tiempo real el Exponente de Lyapunov máximo (λ) de la serie temporal del error. La evolución del error se aproxima mediante: ∣e(k)∣≈∣e(0)∣⋅eλk

    Convergencia (λ<0): El lazo de control es asintóticamente estable; las perturbaciones decaen exponencialmente y el sistema absorbe la entropía.
    Caos Determinista (λ>0): Las trayectorias del error están divergiendo. El Guardián físico identifica instantáneamente esta firma matemática como una "Falla de Control" y acciona el circuito Crowbar (Freno de Emergencia físico en el ESP32) abortando la ingesta de datos antes de que se materialice un error de desbordamiento de memoria (OOM)
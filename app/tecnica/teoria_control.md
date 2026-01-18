
--------------------------------------------------------------------------------
**1. Arquitectura de Control Híbrida**
Implementación: app/flux_condenser.py -> PIController y DataFluxCondenser.
El sistema utiliza una topología de Control por Prealimentación (Feedforward) aumentada con Retroalimentación (Feedback).
u(t)=Feedback (Error)uPI​(e)​​+Feedforward (Complejidad)uFF​(ΔC)​​+ProteccioˊnuSafety​​​
1.1 Diagrama de Bloques Lógico
1. Planta: El proceso de ingestión de datos (Batch Processing).
2. Sensor: El Motor de Física (FluxPhysicsEngine) que mide Saturación (V), Corriente (I) y Potencia (P).
3. Estimador de Estado: Un Filtro de Kalman Extendido (EKF) que predice la saturación futura.
4. Controlador: Algoritmo PI Discreto con Feedforward basado en complejidad.

--------------------------------------------------------------------------------
**2. El Controlador PI Discreto (Feedback)**
Objetivo: Eliminar el error de estado estacionario (e(k)=SP−PV) manteniendo la saturación del sistema al 30%.
Utilizamos la forma posicional discreta con mejoras anti-windup avanzadas:
uPI​(k)=Kp​⋅e(k)+Ki​i=0∑k​e(i)⋅Δt
Mecanismos de Robustez Implementados:
• Anti-Windup (Clamping Condicional + Back-calculation): A diferencia de un anti-windup simple, el sistema usa Back-calculation. Si el actuador (tamaño del batch) se satura, recalcula el término integral para que sea consistente con la salida real, evitando que el error se "acumule" fantasmagóricamente mientras el sistema está al límite.
• Slew Rate Limiting (Anti-Jerk): Limitamos la tasa de cambio de la salida (du/dt) para evitar cambios bruscos en el tamaño del lote que podrían inestabilizar la base de datos o la memoria.

--------------------------------------------------------------------------------
**3. Control Feedforward Adaptativo (Anticipación)**
Implementación: DataFluxCondenser._stabilize_batch.
El control feedback es intrínsecamente lento (debe esperar a que ocurra el error). Para compensar, el Guardián "mira hacia adelante":
• Variable de Perturbación: La Complejidad Ciclomática del texto en los datos crudos.
• La Lógica: Si el sistema detecta que el siguiente bloque de texto es matemáticamente más denso (mayor entropía o longitud), reduce el tamaño del batch antes de que aumente la saturación de memoria.
• Ecuación de Ajuste: uFF​=KFF​⋅(dtdC​+0.5dt2d2C​)
 El sistema reacciona no solo al cambio de complejidad (dC/dt), sino a la aceleración del cambio (d2C/dt2), actuando como un amortiguador predictivo.

--------------------------------------------------------------------------------
**4. El Oráculo de Estado: Filtro de Kalman Extendido (EKF)**
Implementación: _predict_next_saturation.
No esperamos a medir la saturación; la predecimos. El sistema implementa un EKF que modela la saturación como un oscilador armónico amortiguado con punto de equilibrio variable.
Modelo de Estado del EKF:
Estado vector x=[s,v,a]T (Saturación, Velocidad, Aceleración). s˙=v
 v˙=a−βv−ω2(s−seq​)
 a˙=−γa+wa​
• Adaptación de Parámetros: El filtro ajusta dinámicamente sus propios parámetros internos (ω frecuencia natural, β amortiguamiento) basándose en la "innovación" (error de predicción), permitiéndole distinguir entre ruido de medición y tendencias reales del sistema.

--------------------------------------------------------------------------------
**5. Análisis de Estabilidad en Tiempo Real**
El sistema no asume estabilidad; la calcula matemáticamente en cada ciclo.
**5.1 Criterio de Jury (Validación Estática)**
Al inicio, validamos que los parámetros del controlador (Kp​,Ki​) no violen la estabilidad del lazo cerrado discreto. Verificamos que las raíces del polinomio característico estén dentro del círculo unitario (∣z∣<1).
**5.2 Exponente de Lyapunov (Validación Dinámica)**
Implementación: _update_lyapunov_metric. Durante la ejecución, estimamos el Exponente de Lyapunov (λ) de la serie temporal del error. ∣e(k)∣≈∣e(0)∣⋅eλk
• λ<0 (Convergencia): El sistema es estable; el error decae exponencialmente.
• λ>0 (Caos): El sistema está divergiendo. El Guardián detecta esto como una "Falla de Control" y activa el Freno de Emergencia antes de que ocurra un desbordamiento de memoria (OOM).

--------------------------------------------------------------------------------
**6. Filtrado de Señal: EMA Adaptativo**
Implementación: _apply_ema_filter.
La señal de saturación suele ser ruidosa. No usamos un promedio simple. Implementamos un filtro de Media Móvil Exponencial (EMA) donde el factor α varía según la volatilidad estadística:
• Alta varianza (Ruido): α disminuye (mayor suavizado).
• Cambio escalón (Step): Si se detecta un cambio brusco real, el filtro "abre la compuerta" (bypass) para reaccionar rápido, evitando el retraso de fase típico de los filtros pasabajos.

--------------------------------------------------------------------------------

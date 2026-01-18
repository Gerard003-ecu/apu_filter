# ⚡ circuitos_RLC.md: La Física del Flujo de Datos
"En APU Filter, no procesamos datos; gestionamos un fluido de información. El caos de los archivos CSV masivos se doma aplicando las leyes de Maxwell y la Termodinámica, no con simples if-else."
Este documento detalla el modelo matemático que utiliza el Guardián (FluxCondenser) para estabilizar la ingesta de datos. Transformamos variables informáticas (latencia, errores, caché) en variables físicas (Voltaje, Corriente, Entropía) para aplicar Teoría de Control de grado industrial.

**1. El Modelo RLC: Parámetros Fundamentales**
El sistema modela el pipeline de procesamiento como un circuito RLC Serie (Resistencia - Inductancia - Capacitancia) gobernado por ecuaciones diferenciales de segundo orden.
**1.1 Capacitancia ($C$) y Energía Potencial**
Definición en Código: system_capacitance (Default: 5000.0).
Significado Físico: La capacidad del sistema para "almacenar" presión de datos sin colapsar (Buffer de Memoria).
Variable de Estado ($V$): Saturación. Representa qué tan lleno está el lote de procesamiento actual respecto a la capacidad máxima teórica.
Ecuación de Energía:$$E_c = \frac{1}{2} C \cdot V_{saturación}^2$$
Diagnóstico: Una alta energía potencial ($E_c$) indica "Alta Presión". Si no se libera, el sistema corre riesgo de Buffer Overflow (Desbordamiento).
**1.2 Inductancia ($L$) y Energía Cinética**
Definición en Código: system_inductance (Default: 2.0).
Significado Físico: La "Inercia" del flujo. Un sistema con alta inductancia se resiste a cambios bruscos en la calidad de los datos.
Variable de Estado ($I$): Corriente de Eficiencia. Calculada como el ratio de cache_hits / total_records.
Ecuación de Energía:$$E_k = \frac{1}{2} L \cdot I^2$$
Diagnóstico: Un flujo con alta energía cinética es Laminar. Datos limpios y repetitivos generan una corriente alta y estable. Si $I$ cae, el flujo se vuelve turbulento.
**1.3 Resistencia Dinámica ($R$) y Potencia Disipada**
Definición en Código: base_resistance + complexity_factor.
Significado Físico: La fricción operativa. La resistencia no es constante; aumenta dinámicamente con la complejidad de los datos (texto sucio, codificaciones extrañas).
Ecuación de Potencia (Ley de Joule):$$P_{disipada} = I_{ruido}^2 \cdot R_{dinámica}$$
Diagnóstico: Esta métrica es el "Calor" del sistema.
Si $P_{disipada} > 50W$ (umbral simulado), el sistema declara "Sobrecalentamiento" y activa el Freno de Emergencia (EMERGENCY_BRAKE_FACTOR), reduciendo drásticamente el tamaño del lote para "enfriar" la CPU.
**2. Fenómenos Transitorios y Detección de Anomalías**
El motor físico utiliza un integrador numérico Runge-Kutta de 4to Orden (RK4) para simular la evolución del estado del sistema ($dQ/dt$, $dI/dt$) con alta precisión temporal.
**2.1 El Voltaje Flyback ($V_{fb}$)**
En un inductor, interrumpir la corriente bruscamente genera un pico de voltaje inverso destructivo.
En APU Filter: Ocurre cuando un lote de datos de alta calidad es seguido repentinamente por datos corruptos ("Rotura de la Inercia").
Fórmula:$$V_{flyback} = L \cdot \left| \frac{di}{dt} \right|$$
Acción del Guardián: Un pico de Flyback > 10.0V alerta sobre una Fractura Estructural en el archivo de entrada (ej. cambio de esquema a mitad de archivo).
**2.2 Estabilidad Giroscópica ($S_g$)**
Fuente: FluxPhysicsEngine.calculate_gyroscopic_stability.
Concepto: Modelamos el flujo de datos como un "trompo" girando. Si la velocidad de procesamiento (spin) es alta, el sistema es estable ante perturbaciones laterales.
Teorema de la Raqueta de Tenis: Detectamos si el procesamiento está entrando en un eje de rotación inestable (Nutación).
Métrica: $S_g \in 1$. Si $S_g < 0.3$, el sistema entra en "Nutación Crítica" y rechaza el archivo por inestabilidad caótica.
**3. Teoría de Control: El Cerebro PID y EKF**
Para mantener el sistema en equilibrio, no usamos umbrales estáticos, sino un sistema de control de lazo cerrado adaptativo.
**3.1 Controlador PID Discreto**
El objetivo es mantener la Saturación en un Setpoint óptimo (30%), conocido como "Flujo Laminar".
Proporcional ($K_p$): Reacción inmediata al error de saturación.
Integral ($K_i$): Corrige el error acumulado histórico. Incluye lógica Anti-Windup (Clamping) para evitar que la integral crezca infinitamente cuando el sistema se satura físicamente.
Salida: Ajusta dinámicamente el batch_size (tamaño del lote).
Datos limpios $\to$ Batch aumenta (aceleración).
Datos sucios $\to$ Batch disminuye (frenado de precisión).
**3.2 Filtro de Kalman Extendido (EKF)**
Fuente: DataFluxCondenser._predict_next_saturation.
Función: Predicción del futuro inmediato. El sistema no espera a saturarse; utiliza un modelo cinemático interno (posición, velocidad, aceleración de la saturación) para predecir la carga del siguiente lote y ajustar los recursos preventivamente.
**4. El Oráculo de Laplace: Análisis en el Dominio-S**
Antes de procesar un solo byte, el sistema valida la viabilidad del circuito en el dominio de la frecuencia compleja.
Función de Transferencia
$$H(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$
El microservicio LaplaceOracle calcula los polos del sistema configurado:
Estabilidad ($\sigma$): Verifica que todos los polos estén en el semiplano izquierdo (LHP). Si hay polos en RHP, el sistema se declara "Estructuralmente Inestable" y se niega a arrancar.
Amortiguamiento ($\zeta$):
$\zeta < 1$ (Subamortiguado): El sistema oscilará. Útil para descubrimiento rápido pero riesgoso.
$\zeta > 1$ (Sobreamortiguado): El sistema será lento pero robusto. Preferido para cargas masivas de datos financieros.

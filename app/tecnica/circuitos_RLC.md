# ⚡ circuitos_RLC.md: La Física del Flujo de Datos
"En APU Filter, no procesamos datos; gestionamos un fluido de información. El caos de los archivos CSV masivos se doma aplicando las leyes de Maxwell y la Termodinámica, no con simples if-else."
Este documento detalla el modelo matemático que utiliza el Guardián (FluxCondenser) para estabilizar la ingesta de datos. Transformamos variables informáticas (latencia, errores, caché) en variables físicas (Voltaje, Corriente, Entropía) para aplicar Teoría de Control de grado industrial.

**1. El Modelo RLC: Parámetros Fundamentales**
El sistema modela el pipeline de procesamiento como un circuito RLC Serie (Resistencia - Inductancia - Capacitancia) gobernado por ecuaciones diferenciales de segundo orden.
**1.1 La Membrana Viscoelástica (Acumulador Hidráulico y p-Laplaciano)**
En lugar de un condensador ideal, el sistema implementa una *membrana viscoelástica* para absorber ondas de choque (CSVs caóticos masivos). La presión de entrada ($V_{total}$) se modela con una ecuación diferencial de segundo orden:
$$V_{total} = V_{elástico} + V_{viscoso} + V_{inercial}$$
*   **Componente Elástica ($V_{elástico}$):** La capacidad volumétrica de memoria.
*   **Componente Inercial ($V_{inercial}$):** Resistencia a la aceleración brusca del caudal de datos (mitiga el "golpe de ariete" del CSV).
*   **Difusión p-Laplaciana ($p > 2$):** La membrana actúa como un fluido no newtoniano. Si el archivo CSV ingresa suavemente, la membrana es elástica. Si llega un pico destructivo de datos corruptos, la membrana incrementa matemáticamente su viscosidad, "endureciéndose" instantáneamente para redistribuir la carga a nodos vecinos y evitar el colapso del pipeline.
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

**5. Proyección Física Isomórfica (El Gatekeeper ESP32)**
La simulación RLC no vive solo en la nube; se materializa en el nodo perimetral ESP32 a través de `telemetry.h` y `main.cpp`.
*   **Ingeniería Defensiva (Zero-Allocation):** El ESP32 valida localmente la coherencia termodinámica (`isCoherent()`). Si recibe un "Pasabordo de Telemetría" con datos imposibles (ej. Entropía negativa), actúa como un diodo inverso, vetando la ejecución física.
*   **Resonancia Hexagonal:** El flujo de inercia y disipación se mapea a un hexágono físico de transistores MOSFET (IRLZ44N) mediante interleaving de fases PWM (0°, 60°, 120°...), cancelando el rizado de datos y estabilizando el sistema ante ataques de entropía altos.


--------------------------------------------------------------------------------
⚡ circuitos_RLC.md: La Física del Flujo de Datos
"En APU Filter, no procesamos datos; gestionamos un fluido de información. El caos de los archivos CSV masivos se doma aplicando las leyes de Maxwell y la Termodinámica, no con simples if-else."
Este documento detalla el modelo matemático fundacional que utiliza el Guardián de la Evidencia (FluxCondenser) para estabilizar la ingesta de datos en el ecosistema. Transformamos variables informáticas abstractas (latencia, errores de parseo, desbordamientos de caché) en variables físicas concretas (Voltaje, Corriente, Entropía) para aplicar Teoría de Control de grado industrial y garantizar un procesamiento determinista.

--------------------------------------------------------------------------------
1. El Paradigma Dinámico: Ecuación Diferencial del Sistema
El sistema abandona la lectura secuencial estática y modela el pipeline de procesamiento de datos como un circuito RLC Serie (Resistencia - Inductancia - Capacitancia). Esta topología se comporta como un oscilador armónico amortiguado que está gobernado por una ecuación diferencial lineal de segundo orden:
Ldt2d2q​+Rdtdq​+C1​q=V(t)
Donde cada variable tiene una ontología dual (física-informática) estricta:

    Carga (q): Representa el volumen estático de bytes acumulados en la memoria.
    Corriente (i=dtdq​): Representa el throughput o caudal de procesamiento de datos por segundo.
    Voltaje (V(t)): Representa la fuerza electromotriz o "presión" de entrada generada por la inyección masiva de archivos desde el usuario.


--------------------------------------------------------------------------------
2. Topología de Componentes y Leyes Constitutivas
Para resolver el estrés sobre la base de datos, el sistema implementa tres barreras físicas que actúan sobre el flujo de información:
2.1 La Membrana Viscoelástica (Capacitancia C)
A diferencia de un simulador electrónico estándar que utiliza un condensador ideal, el sistema APU_filter implementa matemáticamente una membrana viscoelástica (un acumulador hidráulico y p-Laplaciano). Su propósito es absorber las ondas de choque provocadas por la llegada de CSVs caóticos masivos. La caída de presión total en este nodo no es puramente lineal, sino que se define mediante: Vtotal​=Velaˊstico​+Vviscoso​+Vinercial​
2.2 Inercia y el Fenómeno de "Flyback" (Inductancia L)
El inductor modela la inercia computacional del sistema. Si el controlador estrangula el flujo de datos abruptamente (como en el final de un ciclo PWM), el campo de información "colapsa" de golpe. Según la ley de Faraday-Lenz, el sistema intentará mantener la corriente fluyendo, lo que invierte la polaridad y genera un pico transitorio destructivo conocido como Voltaje de Flyback (Vfb​). El FluxCondenser vigila este pico; si la inercia de los datos supera los límites de seguridad, la inyección es vetada de inmediato para proteger el ecosistema de un golpe de ariete computacional.
2.3 Entropía y Fricción (Resistencia R)
La resistencia acopla la validación a la Segunda Ley de la Termodinámica. Cualquier fricción al parsear un dato se mide como Potencia Disipada (Pdiss​). La física de circuitos exige axiomáticamente que esta potencia sea no-negativa (Pdiss​≥0) para cualquier elemento pasivo. Si el sistema detecta "disipación negativa", concluye que hay datos corruptos creando energía a partir del vacío y levanta las defensas perimetrales.

--------------------------------------------------------------------------------
3. Proyección Física Isomórfica (El Gatekeeper en el Edge)
La simulación RLC y el cálculo de tensores no viven únicamente en el clúster de la nube; se materializan físicamente en el nodo perimetral (Hardware ESP32) mediante los contratos de firmware telemetry.h y main.cpp.
A. Ingeniería Defensiva (Zero-Allocation)
El ESP32 opera con memoria estática y no delega su supervivencia al servidor. Valida localmente la coherencia termodinámica procesando el "Pasabordo de Telemetría" a través de su función interna isCoherent(). Si los vectores de estado recibidos indican anomalías teóricas (ej. β1​>0 o entropía térmica negativa), el microcontrolador asume la función de un diodo inverso inquebrantable, aplicando un veto a nivel de hardware (Crowbar) que bloquea el proyecto en la realidad física.
B. Resonancia Hexagonal de Silicio
Para garantizar que las ráfagas extremas de datos no desestabilicen las líneas lógicas, el flujo de inercia y disipación se mapea sobre una topología en anillo de 6 nodos (Benceno C6​). Este hexágono físico se acciona mediante transistores MOSFET (como el IRLZ44N), donde el firmware aplica un interleaving (entrelazado) de fases PWM a 0°, 60° y 120°. Esta arquitectura topológica logra cancelar matemáticamente el rizado (ripple) de la inyección de datos, estabilizando termodinámicamente el sistema incluso frente a ráfagas de entrada de entropía máxima.
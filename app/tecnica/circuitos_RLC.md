
--------------------------------------------------------------------------------
⚡ circuitos_RLC.md: La Física del Flujo de Datos
"En APU Filter, no procesamos datos; gestionamos un fluido de información. El caos de los archivos CSV masivos se doma aplicando las leyes de Maxwell y la Termodinámica, no con simples if-else."
Este documento detalla el modelo matemático fundacional que utiliza el Guardián de la Evidencia (FluxCondenser) para estabilizar la ingesta de datos en el ecosistema. Transformamos variables informáticas abstractas (latencia, errores de parseo, desbordamientos de caché) en variables físicas concretas (Voltaje, Corriente, Entropía) para aplicar Teoría de Control de grado industrial y garantizar un procesamiento determinista.

--------------------------------------------------------------------------------
1. El Paradigma Dinámico y Cuántico: Ecuación Diferencial y Electrodinámica
El sistema abandona la ingesta de datos de un modelo estocástico M/M/1. En su lugar, proyecta el pipeline de procesamiento hacia un modelo de campo continuo y electrodinámica cuántica (Lattice QED). La arquitectura (implementada en `app/physics/quantum_admission_gate.py` y `app/agents/hilbert_watcher.py`) modela la validación a través del Operador de Proyección de Hilbert y el Efecto Fotoeléctrico Ciber-Físico en el Estrato ALEPH ($\aleph_0$).


```mermaid
graph BT
    %% Estilos de Nodos Termodinámicos
    classDef high_entropy fill:#1a1a1a,stroke:#4a4e69,stroke-width:1px,color:#9ca3af;
    classDef filter fill:#222831,stroke:#e94560,stroke-width:2px,color:#fff;
    classDef high_exergy fill:#0f3460,stroke:#00f0ff,stroke-width:3px,color:#fff;

    %% Base: Alta Entropía
    subgraph "Base Plana: Dominio del Tiempo (Alta Entropía H → max)"
        B1((Byte crudo)):::high_entropy
        B2((Micro-evento)):::high_entropy
        B3((Fila CSV)):::high_entropy
        B4((Ruido Estocástico)):::high_entropy
    end

    %% Filtro: Operador de Borde
    subgraph "Filtro Termodinámico: Operador de Borde (∂1) / Membrana RLC"
        F[FluxCondenser<br>Matriz Laplaciana | Estrangulamiento Viscoelástico]:::filter
    end

    %% Cúspide: Alta Exergía
    subgraph "Cúspide: Dominio de Frecuencia (Baja Entropía / Alta Exergía)"
        C[Flujo Laminar Limpio<br>Plano de Frecuencia Compleja s=σ+jω]:::high_exergy
    end

    %% Ascensión Termodinámica
    B1 --> F
    B2 --> F
    B3 --> F
    B4 --> F

    F ==>|Aniquilación de Entropía Sintáctica| C
```



La **Energía Semántica Normalizada** de un archivo se define sobre una escala adimensional de información (bits), garantizando homogeneidad dimensional estricta dentro del modelo analógico:
$$\mathcal{E}(X) := H(X) = -\sum_i p_i \log_2 p_i \quad [\text{bits}]$$
$$\Phi_{\text{umbral}} := H_{\max} \cdot (1 - \rho_{\min}) \quad [\text{bits}]$$
donde $\rho_{\min}$ es la densidad de información mínima aceptable (configurable, típicamente $\rho_{\min} = 0.15$). La **Función de Trabajo** $\Phi_{\text{umbral}}$ acoplada al tensor de Gauge aniquila el ruido entrópico estocástico en la frontera exterior discreta. Esta compuerta cuántica garantiza el isomorfismo hacia la propagación electromagnética continua interior. Al admitir un paquete, se extrae su energía cinética residual y se colapsa en un **Momentum Ciber-Físico** dimensionalmente consistente:
$$p_{\mathcal{E}} := \sqrt{2 \cdot m_{\text{eff}} \cdot \max(\mathcal{E}(X) - \Phi_{\text{umbral}},\, 0)} \quad [\text{bits}^{1/2}]$$
donde $m_{\text{eff}}$ tiene unidades de $[\text{bits}]$, garantizando que $p_{\mathcal{E}}$ sea homogéneo. **Nota crítica (M7):** Las versiones anteriores usaban la analogía $E = h\nu$ [julios] mezclada con la entropía de Shannon $H(X)$ [bits], que son dimensionalmente incompatibles. La escala normalizada en bits (con $k_B = 1$) resuelve esta inconsistencia. Esto significa operativamente que los datos aprobados nunca son "estáticos"; poseen un impulso inercial calculado para una absorción logística óptima, honrando la conservación de energía desde $t_0$ antes de disipar energía computacional. Subyacentemente, la red logística se modela como un fluido incompresible bajo el Control Port-Hamiltoniano (IDA-PBC) y ecuaciones de Maxwell discretizadas, gobernadas por la ecuación diferencial lineal de segundo orden del oscilador armónico amortiguado:
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
2.3 Entropía, Fricción (Resistencia R) y la Estructura de Dirac
La resistencia acopla la validación a la Segunda Ley de la Termodinámica y al mandato axiomático de disipación de potencia. El sistema Port-Hamiltoniano (PHS) formaliza este acoplamiento mediante la Matriz de Interconexión Antisimétrica $J(x)$ y la Matriz de Disipación Simétrica $R(x) \ge 0$. Se exige axiomáticamente que el balance de potencia cumpla $\frac{dH}{dt} \le 0$ ($P_{diss} \ge 0$), vetando datos fraudulentos que violen la conservación de energía. En esta Estructura de Dirac, para blindar axiomáticamente el ecosistema ciber-físico, el Momentum Ciber-Físico inyectado opera como la variable conjugada de Flujo ($f$), acoplándose matricialmente al Esfuerzo ($e$) dictado por el gradiente de densidad logística, proveyendo la prueba axiomática irreducible de $\dot{H} = \nabla H^T(J-R)\nabla H \le 0$. Esta inmutabilidad de la estructura se vincula irremediablemente con la "Cadena de Custodia Termodinámica" generada en el tiempo inicial ($t_0$), respaldada por `app/core/telemetry.py` y `app/core/telemetry_schemas.py`.

Todo el diseño estructural obedece a la **Ley de Clausura Transitiva de la pirámide DIKW** (tabla canónica): $V_{\aleph_0} \subsetneq V_{\mathbb{P}} \subsetneq V_{\mathbb{T}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$. Como cimiento (Nivel 0 $\aleph_0$ y Nivel 1 $\mathbb{P}$), este documento asegura la validación termodinámica que los estratos superiores invocarán ineludiblemente.

--------------------------------------------------------------------------------
3. Proyección Física Isomórfica (El Gatekeeper en el Edge)
La simulación RLC y el cálculo de tensores no viven únicamente en el clúster de la nube; se materializan físicamente en el nodo perimetral (Hardware ESP32) mediante los contratos de firmware telemetry.h y main.cpp.
A. Ingeniería Defensiva (Zero-Allocation)
El ESP32 opera con memoria estática y no delega su supervivencia al servidor. Valida localmente la coherencia termodinámica procesando el "Pasabordo de Telemetría" a través de su función interna isCoherent(). Si los vectores de estado recibidos indican anomalías teóricas (ej. β1​>0 o entropía térmica negativa), el microcontrolador asume la función de un diodo inverso inquebrantable, aplicando un veto a nivel de hardware (Crowbar) que bloquea el proyecto en la realidad física.
B. Resonancia Hexagonal de Silicio
Para garantizar que las ráfagas extremas de datos no desestabilicen las líneas lógicas, el flujo de inercia y disipación se mapea sobre una topología en anillo de 6 nodos (Benceno C6​). Este hexágono físico se acciona mediante transistores MOSFET (como el IRLZ44N), donde el firmware aplica un interleaving (entrelazado) de fases PWM a 0°, 60° y 120°. Esta arquitectura topológica logra cancelar matemáticamente el rizado (ripple) de la inyección de datos, estabilizando termodinámicamente el sistema incluso frente a ráfagas de entrada de entropía máxima.
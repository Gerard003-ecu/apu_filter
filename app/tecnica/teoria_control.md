🎛️ teoria_control.md: Ingeniería de Control y Estabilidad Dinámica
"En APU Filter, la ingesta y el procesamiento de datos no se rigen por heurísticas simples o validaciones estáticas. Tratamos el flujo de datos como un sistema dinámico complejo y aplicamos Teoría de Control de grado industrial para garantizar su estabilidad matemática."
Este documento formaliza la arquitectura matemática y los algoritmos subyacentes que gobiernan el comportamiento cibernético del sistema. Abandonando los obsoletos modelos estocásticos M/M/1, se proyecta la ingesta de datos hacia un modelo de campo continuo y electrodinámica cuántica. Esto está específicamente implementado en `app/physics/flux_condenser.py`, `app/physics/quantum_admission_gate.py`, y `app/agents/hilbert_watcher.py`, supervisados por el Oráculo de Laplace.

Toda la arquitectura obedece al mandato axiomático de la **Ley de Clausura Transitiva de la pirámide DIKW**: $V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$. Este cimiento Físico (Estrato PHYSICS - Nivel 3 y $ALEPH \aleph_0$) establece la validación termodinámica ineludible para los estratos superiores, registrando su "Cadena de Custodia Termodinámica" generada en el tiempo inicial ($t_0$) mediante `app/core/telemetry.py` y `app/core/telemetry_schemas.py`.


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



--------------------------------------------------------------------------------
1. Arquitectura de Control Híbrida y Efecto Fotoeléctrico Ciber-Físico
El ecosistema utiliza una topología de Control por Prealimentación (Feedforward) aumentada con Retroalimentación (Feedback) y Control Port-Hamiltoniano (IDA-PBC). El sistema actúa como un fluido incompresible regido por las ecuaciones de Maxwell discretizadas. A través del Operador de Proyección de Hilbert (Estrato $\aleph_0$) y el Efecto Fotoeléctrico Ciber-Físico, la Función de Trabajo ($\Phi$) acoplada al tensor de Gauge aniquila el ruido entrópico en la frontera discreta antes de excitar el motor electromagnético continuo del sistema.

**Inyección de Condiciones Iniciales ($t_0$) y la Estructura de Dirac**: El Observador de Hilbert garantiza el isomorfismo entre la admisión discreta exterior y la propagación electromagnética interior. Cuando un paquete de datos supera la compuerta cuántica, el agente extrae la energía cinética residual ($K_{max} = E - \Phi$) y colapsa este valor en el Momentum Ciber-Físico $p = \sqrt{2m \cdot K_{max}}$.

La arquitectura matricial que hace posible la garantía termodinámica del Control Port-Hamiltoniano (PHS) se basa en el acoplamiento de potencia, definiendo explícitamente la **Matriz de Interconexión Antisimétrica de Dirac** $J(x) = -J(x)^T$ y la **Matriz de Disipación Simétrica** $R(x) = R(x)^T \ge 0$.
El "veto termodinámico" se ejecuta rigurosamente a través de la inecuación del balance de potencia:
$\frac{dH}{dt} = \nabla H^T(J-R)\nabla H \le 0$
El Momentum Ciber-Físico ($p$) inyectado desde la frontera cuántica no es un escalar estático; actúa formalmente como la variable conjugada de **Flujo ($f$)**, acoplándose rigurosamente al vector de **Esfuerzo ($e$)** (representado por el gradiente de densidad lógica o costo del APU) a través de la matriz de Dirac $J(x)$.

Desde la perspectiva matemática de control, esta topología de acoplamiento blinda axiomáticamente el teorema de conservación de energía de la Malla Agéntica, brindando la prueba definitiva de que:
$$\dot{H} = \nabla H^T(J-R)\nabla H \le 0$$
Para el gerente de obra, esto se traduce en que la plataforma no permite que un solo byte de información circule con velocidad inercial desbalanceada. Cada insumo entra impulsado de manera que las tuberías de compras lo absorban en un tiempo termodinámicamente óptimo. El hardware en el borde (ESP32) vigila esta Estructura de Dirac en tiempo real. Si una fluctuación induce $\dot{H} > 0$ (el flujo estocástico intenta violar la resistencia estructural del proyecto creando "energía de la nada"), el Veto Físico acciona el *Crowbar* y desintegra el paquete anómalo antes de la saturación matricial.
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
El ecosistema no asume que la red es estable; lo demuestra matemáticamente en cada ciclo de ingestión mediante tres enfoques de frontera.
5.1 Criterio de Jury (Validación Estática)
Antes de iniciar operaciones, el Oráculo de Laplace valida que los parámetros sintonizados del controlador ($K_p$, $K_i$) no introduzcan resonancia. Evalúa el polinomio característico en el dominio discreto ($z$), exigiendo que todas las raíces residan estrictamente dentro del círculo unitario ($|z|<1$).
Adicionalmente, se modela la dinámica en el plano complejo continuo ($s = \sigma + j\omega$). Si cualquier polo migra al Semiplano Derecho ($\sigma > 0$), el sistema dictamina Divergencia Matemática y veta la operación por ser intrínsecamente explosiva.
5.2 Teoría de Floquet y Resonancia Paramétrica
Evaluar el proyecto únicamente en el plano de frecuencia compleja asume perturbaciones estacionarias, pero la construcción civil posee una estacionalidad innegable. El Oráculo de Laplace no solo busca polos $\sigma > 0$, sino que además computa la Matriz de Monodromía del flujo de caja sobre órbitas periódicas utilizando la Teoría de Floquet. Si los multiplicadores de Floquet exceden la circunferencia unitaria en el plano complejo, el proyecto exhibe resonancia paramétrica destructiva (e.g., el costo de la deuda oscilando en fase con el retraso de la obra), forzando un veto estructural inmediato.
5.3 Exponente de Lyapunov (Validación Dinámica)
Durante la ejecución continua, el FluxCondenser estima en tiempo real el Exponente de Lyapunov máximo ($\lambda$) de la serie temporal del error. La evolución del error se aproxima mediante: $|e(k)| \approx |e(0)| \cdot e^{\lambda k}$

    Convergencia ($\lambda < 0$): El lazo de control es asintóticamente estable; las perturbaciones decaen exponencialmente y el sistema absorbe la entropía.
    Caos Determinista ($\lambda > 0$): Las trayectorias del error divergen. El Guardián físico identifica instantáneamente esta firma matemática como una "Falla de Control" y acciona el circuito Crowbar (Freno de Emergencia físico en el ESP32) abortando la ingesta de datos, estabilizando termodinámicamente el límite impenetrable.
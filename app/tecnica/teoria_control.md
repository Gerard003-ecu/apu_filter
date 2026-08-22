# ⚙️ metodos.md: Ingeniería Bajo el Capó v3.0

> "APU Filter no es un software de contabilidad; es un Simulador de Física de Negocios ciber-físico. Aquí detallamos las ecuaciones de estado, los teoremas topológicos, la geometría simpléctica y los circuitos neuromórficos que garantizan la sabiduría del sistema."

Este documento técnico desglosa la maquinaria matemática que permite al Consejo de Sabios transformar datos crudos en veredictos estratégicos, basándose en la convergencia de la Dinámica de Sistemas (Laplace), la Geometría Simpléctica y de de Rham-Fukaya, la Topología Algebraica, la Estocástica Financiera y el Hardware en el Borde.

---

## 1. El Guardián: Física de Fluidos, Mecánica Simpléctica y Computación Neuromórfica (Edge)

*   **Base Teórica:** Ecuaciones de Maxwell discretizadas, Control Port-Hamiltoniano (PHS), Dinámica Simpléctica y Física de Semiconductores.
*   **Componentes:** `flux_condenser.py`, `neuromorphic_solver.py`, `v4_opt_symplectic_manifold.py`, Firmware ESP32 (`telemetry.h`). El Guardián no lee bits; procesa un fluido de información con propiedades físicas (Energía, Resistencia, Inercia, Capacidad Simpléctica).

### 1.1 Cuadratura Conforme de Gauss-Legendre sobre $S^2$
Para la integración numérica exacta de campos de deformación y potenciales de riesgo proyectados sobre la variedad esférica 2-dimensional $S^2$ (espacio de fases y momentos de decisión), se formula la **Cuadratura Conforme de Gauss-Legendre**, eliminando las singularidades polares y absorbiendo analíticamente el Jacobiano polar:

$$\sin\theta \, d\theta$$

Mediante la transformación conforme de variable:
$$t = \cos\theta \implies dt = -\sin\theta \, d\theta, \quad \theta \in [0, \pi] \iff t \in [1, -1]$$

La integral de superficie sobre la 2-esfera se transforma exactamente en:
$$I = \int_{0}^{2\pi} \int_{0}^{\pi} f(\theta, \phi) \sin\theta \, d\theta \, d\phi = \int_{0}^{2\pi} \int_{-1}^{1} f(\arccos(t), \phi) \, dt \, d\phi$$

Discretizando la coordenada longitudinal $t$ mediante $N$ nodos de Gauss-Legendre $t_i \in [-1, 1]$ con pesos canónicos de Legendre $w_i = \frac{2}{(1 - t_i^2)[P'_N(t_i)]^2}$, y la coordenada azimutal $\phi$ mediante una distribución trapezoidal uniforme de $M$ puntos $\phi_j = \frac{2\pi j}{M}$:

$$I \approx \frac{2\pi}{M} \sum_{j=1}^{M} \sum_{i=1}^{N} w_i \cdot f\left( \arccos(t_i), \frac{2\pi j}{M} \right)$$

**Ventajas Numéricas y Exactitud:**
1.  **Exactitud Polinomial:** Integra de manera exacta armónicos esféricos $Y_\ell^m(\theta, \phi)$ hasta grado $\ell \le 2N - 1$.
2.  **Invarianza de Malla:** Al absorber el Jacobiano en $dt$, no se produce acumulación de densidad de muestreo en los polos ($\theta = 0, \pi$), evitando inestabilidades numéricas en la discretización simpléctica de la esfera de Bloch y de decisiones.

---

### 1.2 Sumación Compensada de Kahan-Babuška-Neumaier (KBN)
En la acumulación de grandes volúmenes de costos unitarios, trazas de operadores densidad y pasos de integración simpléctica de Störmer-Verlet, la suma estándar de punto flotante acumula un error de redondeo de Wilkinson que escala con la raíz o el número de términos: $\mathcal{O}(N \varepsilon_{\text{mach}})$.

Para anular este fenómeno, el ecosistema implementa de forma obligatoria el algoritmo de **Sumación Compensada de Kahan-Babuška-Neumaier (KBN)**:

Sea $s$ el acumulador en punto flotante $\text{np.float64}$ y $c$ el residuo de compensación persistente. Para cada nuevo término $x$:

$$t = s + x$$
$$c = \begin{cases} (s - t) + x & \text{si } |s| \ge |x| \\ (x - t) + s & \text{si } |s| < |x| \end{cases}$$
$$s_{\text{acum}} = s_{\text{acum}} + c$$

A diferencia de la suma simple de Kahan (que asume que el acumulador es mayor que el término entrante), KBN evalúa la magnitud relativa $|s| \ge |x|$, compensando con precisión exacta la pérdida de bits significativos tanto si se suman números pequeños a un acumulador grande, como si se suma un término dominante a un acumulador pequeño.

**Cota de Error:**
$$|s_{\text{KBN}} - s_{\text{exact}}| \le \left( 2\varepsilon_{\text{mach}} + \mathcal{O}(N \varepsilon_{\text{mach}}^2) \right) \sum |x_i| = \mathcal{O}(\varepsilon_{\text{mach}})$$
Garantizando que la precisión del presupuesto sea independiente de la profundidad del árbol del APU.

---

### 1.3 Propagador de de Rham, de Rham-Hodge y Causalidad de Kramers-Kronig
El sistema incorpora la resolución de la ecuación de Poisson generalizada sobre el Laplaciano del Haz Celular $L_F = \delta^\top G^{-1} \delta$, definiendo la Función de Green estática como la pseudoinversa de Moore-Penrose estable, la cual satisface de forma exacta:
$$L_F G L_F = L_F \quad \wedge \quad G \cdot \mathbf{1} = \mathbf{0}$$
Para el análisis transitorio y el régimen dinámico bajo excitación, se integra el propagador retardado causal en el plano-S complejos:

**Inyección de Condiciones Iniciales ($t_0$) y la Estructura de Dirac**: El Observador de Hilbert garantiza el isomorfismo entre la admisión discreta exterior y la propagación electromagnética interior. Cuando un paquete de datos supera la compuerta cuántica, el agente extrae la energía cinética residual ($K_{max} = E - \Phi$) y colapsa este valor en el Momentum Ciber-Físico $p = \sqrt{2m \cdot K_{max}}$.

La arquitectura matricial que hace posible la garantía termodinámica del Control Port-Hamiltoniano (PHS) se basa en el acoplamiento de potencia, definiendo explícitamente la **Matriz de Interconexión Antisimétrica de Dirac** $J(x) = -J(x)^T$ y la **Matriz de Disipación Simétrica** $R(x) = R(x)^T \ge 0$.
El "veto termodinámico" se ejecuta rigurosamente a través de la inecuación del balance de potencia:
$\frac{dH}{dt} = \nabla H^T(J-R)\nabla H \le 0$
El Momentum Ciber-Físico ($p$) inyectado desde la frontera cuántica no es un escalar estático; actúa formalmente como la variable conjugada de **Flujo ($f$)**, acoplándose rigurosamente al vector de **Esfuerzo ($e$)** (representado por el gradiente de densidad lógica o costo del APU) a través de la matriz de Dirac $J(x)$.

Desde la perspectiva matemática de control, esta topología de acoplamiento blinda axiomáticamente el teorema de conservación de energía de la Malla Agéntica, brindando la prueba definitiva de que:
$$\dot{H} = \nabla H^T(J-R)\nabla H \le 0$$
Para el gerente de obra, esto se traduce en que la plataforma no permite que un solo byte de información circule con velocidad inercial desbalanceada. Cada insumo entra impulsado de manera que las tuberías de compras lo absorban en un tiempo termodinámicamente óptimo. El hardware en el borde (ESP32) vigila esta Estructura de Dirac en tiempo real. Si una fluctuación induce $\dot{H} > 0$ (el flujo estocástico intenta violar la resistencia estructural del proyecto creando "energía de la nada"), el Veto Físico acciona el *Crowbar* y desintegra el paquete anómalo antes de la saturación matricial.
La ley de control global del sistema se define como: u(t)=uPI​(e)+uFF​(ΔC)+uSafety​
Además, el sintonizador dinámico de impedancias (PML) obedece incondicionalmente a las **Relaciones de Kramers-Kronig**, garantizando la causalidad termodinámica en el espectro de frecuencias complejas $s = \sigma + j\omega$ al vincular estrictamente la dispersión con la disipación.

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
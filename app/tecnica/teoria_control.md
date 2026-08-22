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
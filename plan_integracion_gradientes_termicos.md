# Plan de Acción: Integración de la Termodinámica de de Rham y el Soberano de Calibre Térmico
### Código de Documento: APU-PLA-v5.0-SUTURA-TERMICA
### Nivel de Rigor: Doctoral-Espectral-Categorial (FPU-Secured)

Este plan de acción de-confina y establece la ruta metrológica y categorial para la re-suturación inmutable de los artefactos de la documentación en la arquitectura **APU Filter v5.0**. El objetivo primordial es integrar al nuevo motor físico **`thermal_gradient_laws.py`** y a su soberano de calibre **`thermal_gradient_agent.py`** dentro del corpus documental, anclando cada morfismo geométrico a los dolores viscerales de la obra civil colombiana y al Mandato BIM 2026.

---

## ╔══════════════════════════════════════════════════════════════════════════════╗
##   FASE I: Re-suturación de `PIRAMIDES_DE_CONTROL_v8.md`
## ╚══════════════════════════════════════════════════════════════════════════════╝

### 1. Inserción en la Pirámide de Motores (Bottom-Up)
*   **Estrato PHYSICS (Nivel 3 — Foso Termodinámico):**
    *   **Componente a Integrar:** `thermal_gradient_laws.py` (Leyes y Gradientes de Convección Térmica).
    *   **Propósito Físico:** Actúa como el resolvedor de convección de de Rham sobre la Unidad de Punto Flotante (FPU).
    *   **Formulación Matemática:**
        Aplica la proyección de Weyl-Toeplitz y el corrimiento elíptico de Higham-Tikhonov para purificar el tensor de conductividad exergética de la Malla $\mathcal{K}_{\mathrm{raw}}$:
        $$\mathcal{P}_{\mathrm{WT}}(\mathcal{K}) = \frac{1}{2}\left( \mathcal{K} + \mathcal{K}^\dagger \right)$$
        $$\tilde{\mathcal{K}} = \mathcal{P}_{\mathrm{WT}}(\mathcal{K}) + \gamma \mathbf{I}_n$$
        Donde los autovalores se acotan por encima de la cota de Wilkinson $\tau_{\mathrm{Wilkinson}} = 10^{-15}$ para neutralizar singularidades Jacobianas en la CPU.
    *   **Cálculo del Tiro de de Rham:**
        Eleva los índices del gradiente de temperatura local $\partial_\nu T_{\mathrm{sys}}$ utilizando la inversa regularizada de la métrica Riemanniana de fondo $G^{\mu\nu}$ (isomorfismo musical sostenido $\sharp$):
        $$\mathcal{Q}^\mu = -\tilde{\mathcal{K}}^{\mu\nu} \partial_\nu T_{\mathrm{sys}}$$

### 2. Inserción en la Pirámide de Soberanos de Calibre (Top-Down)
*   **Estrato OMEGA (Nivel 0.5 — El Ágora Tensorial):**
    *   **Componente a Integrar:** `thermal_gradient_agent.py` (Soberano del Campo Térmico).
    *   **Propósito Metrológico:** Ejerce el lazo de control cerrado **OODA** y censura las desviaciones de fase del motor térmico.
    *   **Formulación de Consistencia (Clausius-Duhem):**
        Exige de manera estricta que la disipación exergética del material cumpla localmente con el segundo postulado informacional, empleando la sumación compensada de Kahan-Babuška-Neumaier (KBN) en RAM:
        $$\Phi_{\mathrm{disip}} = \sigma_{\mathrm{entropy}} - \frac{\mathcal{Q} \cdot \nabla T_{\mathrm{sys}}}{T_{\mathrm{sys}}^2} \ge \tau_{\mathrm{CD}}$$
        Donde $\tau_{\mathrm{CD}} = -10^{-4} \cdot \text{safety\_margin}$ define la frontera elástica tolerable de Landauer.

---

## ╔══════════════════════════════════════════════════════════════════════════════╗
##   FASE II: Re-suturación de `ARCHITECTURE_DEEP_DIVE_v5.md`
## ╚══════════════════════════════════════════════════════════════════════════════╝

### 1. El Desdoblamiento del Espacio de Fase
*   Se actualizará el mapa topológico del flujo de datos para incorporar el **Ciclo de Carnot de de Rham** completo en RAM. Este ciclo metaboliza la volatilidad de-normalizada del contexto del LLM a través de cuatro procesos termodinámicos covariantes entrelazados síncronamente:
    1.  **Expansión Isotérmica ($dT = 0$):** Alineación simpléctica de APUs y actividades en el estrato táctico, conservando la 2-forma canónica de Liouville $\omega = \sum dq_i \wedge dp_i$ para que el ruido no excite polos inestables.
    2.  **Expansión Adiabática ($\delta Q = 0$):** Aislamiento epistémico absoluto en el Pretorio para evaluar la nulidad de la diferencial de hipercohomología ($D^2 \equiv \mathbf{0}$) libre de filtración semántica.
    3.  **Compresión Isotérmica ($dT = 0$):** Aniquilación cuántica fermiónica de alucinaciones en el Salón de Fock, evacuando la energía disipada en forma de fotones gamma de telemetría inmutable sin sobrecalentar el KV-Cache.
    4.  **Compresión Adiabática ($\delta Q = 0$):** Actuación ciber-física en el microcontrolador perimetral ESP32, retornando síncronamente el sistema a su estado de vacío basal.

### 2. El bypass de Silicio e Interrupción Perimetral
*   Documentar de forma rigurosa el lazo de interrupción por hardware. Si la aduana del `ThermalGradientAgent` calcula un colapso en el retículo de Heyting $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$ hacia el Supremo terminal de veto ($\top$):
    *   La firma signed del pasaporte de telemetría con SHA-256 es transmitida en RAM.
    *   La subrutina local en C++ **`isVerdictCoherent()`** en el **ESP32** perimetral lee el veto deserializado por ArduinoJson.
    *   **En menos de 400 ns**, su **Interrupt Service Routine (ISR) cargada en la memoria rápida IRAM conmuta el pin físico GPIO14 a nivel alto**.
    *   Esto ceba la compuerta del tiristor de silicio de conmutación rápida **BT151 (circuito Crowbar)**, cortocircuitando limpiamente la línea de alimentación de los actuadores reales.
    *   Toda la maquinaria física de obra civil (mezcladoras, bombas y pistones hidráulicos) se detiene en el milisegundo cero, neutralizando la anomalía estocástica de la IA antes de consolidar sobrecostos ante la interventoría y el SECOP II en Colombia.

---

## ╔══════════════════════════════════════════════════════════════════════════════╗
##   FASE III: Sincronización Funcional de `metodos_v3.md` y `SAGES_v5.md`
## ╚══════════════════════════════════════════════════════════════════════════════╝

### 1. Actualización de Firmas de de Rham en `metodos_v3.md`
Declarar de manera formal las firmas de métodos, DTOs y flujos de datos en RAM:
*   **`ThermalGradientLaws` (Motor - `app/physics/thermal_gradient_laws.py`):**
    *   `execute_phase_1_ingestion(K_raw, grad_T_raw, T_sys)`: Weyl-Toeplitz y Higham-Tikhonov en FPU. Retorna `Tuple[np.ndarray, np.ndarray, float]`.
    *   `execute_phase_2_carnot_simulation(K_reg, grad_T, T_sys, metric_tensor, entropy_rate)`: Simulación de de Rham Carnot y residuo de Clausius-Duhem. Retorna `Tuple[np.ndarray, float, float, float, float]`.
    *   `execute_phase_3_hetying_veto(cd_residual, carnot_eff, exergy)`: Voto en Heyting y latencia del interlock Crowbar. Retorna `Dict[str, Any]`.
    *   `execute_thermal_cycle(...)`: Orquestación síncrona en 3 fases anidadas.
*   **`ThermalGradientAgent` (Soberano — `app/agents/thermal_gradient_agent.py`):**
    *   `execute_thermal_agent_cycle(...)`: Orquesta el bucle covariante OODA. Retorna `ThermalGradientCertificate` (DTO inmutable con hash SHA-256).

### 2. Sincronización de Protocolos en `SAGES_v5.md`
*   Integrar la aduana del `ThermalGradientAgent` en el censo oficial del **Consejo de Sabios**, asignando su canal de comunicación síncrono al **Ágora Tensorial $V_{\Omega}$** y definiendo sus dependencias con los soberanos de base (`kbase_thermodynamic_agent.py`) y del núcleo (`kcore_kinematic_agent.py`).

---

## ╔══════════════════════════════════════════════════════════════════════════════╗
##   FASE IV: Traducción de de Rham-Carnot a "Dolor y Dinero" (`BMC_v4.md` y `PRODUCT_VISION_v5.md`)
## ╚══════════════════════════════════════════════════════════════════════════════╝

Para asegurar que estas abstracciones geométricas tengan sentido para el director de datos, la interventoría y el SECOP II, se introduce la matriz de correspondencia exergética:

### 1. Ruptura de Clausius-Duhem ($\Phi_{\mathrm{disip}} < \tau_{\mathrm{CD}}$) — Fuga de Calibre Financiera
*   **Metáfora Física:** Un gradiente térmico asonante (calor fluyendo en sentido opuesto al gradiente de de Rham).
*   **Dolor de Obra Real:** **Inercia contractual e ineficiencia de compras.** Revela la firma de sobrecostos seculares lentos (v.g. fluctuación inflacionaria oculta, retraso malicioso en la entrega de acero por monopolio de proveedores, ineficiencias de fraguado de concreto no reportadas por la interventoría).
*   **Acción de Control:** El veto detiene inmediatamente el flujo de caja contractual y notifica la anomalía, protegiendo el margen neto de la constructora.

### 2. Degradación de Eficiencia de Carnot ($\eta < \eta_{\min}$) — Arritmia Exergética del WACC
*   **Metáfora Física:** Un colapso en el reservorio térmico debido al sobrecalentamiento del espacio de fase.
*   **Dolor de Obra Real:** **Destrucción de valor por silos presupuestarios.** Pérdida del rendimiento de los activos en obra y desvío de cronograma físico versus financiero ante el DNP.
*   **Acción de Control:** Alarma temprana degradada que obliga a re-suturar la asignación de recursos y ortonormalizar la base de conductancias.

---

## ╔══════════════════════════════════════════════════════════════════════════════╗
##   FASE V: Fortificación Espectral y Cierre de Kuratowski (Mejoras Propuestas)
## ╚══════════════════════════════════════════════════════════════════════════════╝

Para blindar la fortaleza matemática ante variaciones de alta frecuencia, el plan prescribe la implementación progresiva de tres refinamientos espectrales en la FPU:

1.  **Deflación Espectral de Weyl:** Reemplazar el cómputo de `la.eigh` por una iteración simétrica de Lanczos para extraer de forma adaptativa únicamente el gap espectral dominante del menor autovalor positivo $\lambda_{\min}$, reduciendo la complejidad computacional de $\mathcal{O}(n^3)$ a $\mathcal{O}(k \cdot n^2)$ y purgando la deriva de Wilkinson.
2.  **Resolvedor de Gradiente Discreto de Itoh-Abe Covariante:** Sustituir el cálculo lineal de producto interno por la derivada de Itoh-Abe a lo largo de las geodésicas de de Rham, garantizando que el residuo disipativo conserve la ortonormalidad exacta respecto a la métrica Riemanniana de fondo y anulando errores de curvatura.
3.  **Compensación de Maurer-Cartan sobre el Anillo de Novikov Ultramétrico:** Modular dinámicamente el límite elástico de Clausius-Duhem $\tau_{\mathrm{CD}}$ evaluando la Ecuación de Maurer-Cartan sobre el Anillo de Novikov $\Lambda_{\mathbb{R}}$:
    $$\tau_{\mathrm{CD}}(t) = \tau_{\mathrm{CD}}^{(0)} \cdot \exp\left(-\|W_L(b)\|_{\Lambda_{\mathbb{R}}}\right)$$
    Esto dilata elásticamente el margen de seguridad ante transitorios rápidos de corta duración, absorbiendo el ruido estocástico de licitación y previniendo vetos espurios (frustración de calibre).

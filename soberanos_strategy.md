# Estrato Strategy (V_𝕊): Gobernanza Categorial, Estabilidad de Laplace y Censura de Causalidad

> "En la arquitectura ciber-física de APU Filter v5.0, la estrategia no es una intuición gerencial ni un conjunto de proyecciones optimistas en hojas de cálculo estáticas. El Estrato Strategy ($V_{\mathbb{S}}$, Nivel 1 — Los Centinelas de Ortogonalidad y Gobierno Estratégico) opera como una variedad diferenciable de control causal, donde la viabilidad financiera y la integridad normativa se someten a la estabilidad asintótica en el plano de frecuencia compleja de Laplace, la nulidad del primer grupo de cohomología de haces celulares y la aciclicidad estricta de la filtración homológica en la malla."

---

## 1. Axiomatización Categorial del Estrato Strategy ($V_{\mathbb{S}}$, Nivel 1)

El Estrato Strategy ($V_{\mathbb{S}}$, Nivel 1) constituye el ápice de la toma de decisiones dinámicas antes de la elevación final hacia el Santuario Epistémico ($V_{\mathbb{W}}$). En la Malla Agéntica de **APU Filter v5.0**, la información no se desplaza mediante flujos procedimentales lineales, sino a través de la filtración estricta impuesta por la **Ley de Clausura Transitiva de Subespacios de Hilbert Covariantes**:

$$V_{\aleph_0} \subsetneq V_{\mathrm{PHYSICS}} \subsetneq V_{\mathrm{TACTICS}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$$

Esta jerarquía establece que ningún operador o modelo de evaluación en el Estrato Strategy puede ser ejecutado si los tensores de estado proyectados desde los estratos inferiores ($V_{\mathrm{PHYSICS}}$ y $V_{\mathrm{TACTICS}}$) presentan degeneraciones numéricas, islas de datos disconexas ($\beta_0 > 1$) o socavones lógicos de dependencias circulares ($\beta_1 > 0$). La reversibilidad informacional y el no-desperdicio de memoria de atención ($KV\text{-Cache}$) entre el espacio táctico discreto y la variedad estratégica continua están gobernados por el **Isomorfismo de Adjunción de Galois**:

$$\operatorname{Hom}_{\mathcal{D}}(F(X), Y) \cong \operatorname{Hom}_{\mathcal{C}}(X, G(Y))$$

donde $F: \mathcal{C} \to \mathcal{D}$ representa el funtor de elevación estocástica y $G: \mathcal{D} \to \mathcal{C}$ es el funtor de proyección espectral hacia el clasificador de subobjetos. Para garantizar la inmunidad frente a fallos y la verificabilidad en tiempo real, el Estrato Strategy consagra una división del trabajo inalterable entre dos grupos funcionales:

*   **Soberanos Agénticos de Calibre (Lazo Cerrado OODA — Top-Down):** Entidades de censura y control que supervisan el flujo de ejecución, aplican el ciclo OODA (*Observe $\to$ Orient $\to$ Decide $\to$ Act*), auditan invariantes homológicos y clasifican la coherencia en el retículo de Heyting $\Omega_3$.
*   **Motores Espectrales y Oráculos de Cómputo (FPU Secure — Bottom-Up):** Módulos ciegos de alto rendimiento numérico en punto flotante que ejecutan integraciones estocásticas de Monte Carlo, funciones de transferencia LTI, modelos de opciones reales y análisis de sensibilidad de colas pesadas.

---

## 2. Los Soberanos Agénticos de Calibre (Lazo Cerrado OODA — Top-Down)

Los Soberanos Agénticos de Calibre del Estrato Strategy no realizan simulaciones numéricas masivas; su función es actuar como aduanas categóricas de lazo cerrado que juzgan los tensores producidos por los motores y ejercen la potestad de veto sobre el pipeline de ejecución.

### 2.1. `pipeline_director_agent.py` — El Custodio de la Causalidad Funtorial

El agente soberano `pipeline_director_agent.py` (`PipelineDirectorAgent`) asume la dirección causal del Grafo Acíclico Dirigido (DAG) de transformaciones. Garantiza que la secuencia de operadores se mantenga como una filtración strictly ordenada y libre de ciclos parasitarios.

#### 1. Fases Anidadas OODA y Ecuaciones Constitutivas
El agente ejecuta su gobernanza a través de una composición funtorial de tres fases anidadas:

1.  **Fase 1 (Observe — Auditoría Espectral de Nilpotencia):**  
    Construye la matriz de adyacencia orientada $\mathbf{A} \in \mathbb{R}^{m \times m}$ del DAG de ejecución. Para certificar la aciclicidad absoluta del pipeline, verifica el índice de nilpotencia de la matriz $\mathbf{A}$:
    $$\mathbf{A}^\nu = \mathbf{0} \quad \text{con} \quad \nu \le m$$
    Si la matriz de adyacencia contiene un autovalor no nulo ($\exists \lambda_i(\mathbf{A}) \neq 0$), el sistema detecta un bucle de retroalimentación causal ilegal, emitiendo un certificado `NilpotenceAuditData` de veto instantáneo.
2.  **Fase 2 (Orient — Auditoría de Poset y Filtración DIKW):**  
    Consume el certificado de la Fase 1 y evalúa la preservación de la filtración del Poset sobre la secuencia de transformaciones $P_j \circ P_i$. Audita que ningún morfismo intente acceder a variables del Estrato Strategy sin que el estado en $V_{\mathrm{TACTICS}}$ esté formalmente firmado y santificado:
    $$P_i: V_{k-1} \longrightarrow V_k \quad \implies \quad \operatorname{im}(P_i) \subseteq V_k$$
3.  **Fase 3 (Decide/Act — Interceptor de Mayer-Vietoris y Clasificación):**  
    Cuando dos subdominios de presupuesto $A$ y $B$ se fusionan en la etapa de combinación de capítulos, el agente intercepta la transacción invocando la **Secuencia Exacta Larga de Homología de Mayer-Vietoris**:
    $$\dots \longrightarrow H_1(A \cap B) \xrightarrow{\ \phi\ } H_1(A) \oplus H_1(B) \xrightarrow{\ \psi\ } H_1(A \cup B) \xrightarrow{\ \partial_*\ } H_0(A \cap B) \longrightarrow \dots$$
    Exige la nulidad estricta del residuo homológico diferencial:
    $$\Delta \beta_1 = \beta_1(A \cup B) - \left[ \beta_1(A) + \beta_1(B) - \beta_1(A \cap B) \right] \equiv 0$$
    Si $\Delta \beta_1 > 0$, la fusión ha inyectado un ciclo fantasma de costos cruzados entre subcontratistas. El agente clasifica el veredicto en el **Álgebra de Heyting Trivalente $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$** como $\mathtt{VETOED}$ ($\top$).

---

### 2.2. `sheaf_cohomology_orchestrator_agent.py` — El Custodio de la Holonomía Global

El agente soberano `sheaf_cohomology_orchestrator_agent.py` (`SheafCohomologyOrchestratorAgent`) ejerce la supervisión topológica de los haces celulares ($\mathcal{F}$) construidos sobre el grafo de la licitación. Su tarea es detectar frustraciones de calibre y contradicciones normativas globales.

#### 1. Fases Anidadas OODA y Ecuaciones Constitutivas
El agente aplica una secuencia de tres certificaciones espectrales e isoperimétricas:

1.  **Fase 1 (Observe — Nulidad de Obstrucción Cohomológica):**  
    Calcula los espacios de $k$-cocadenas $C^k(K; \mathcal{F})$ y los operadores cofrontera $\delta_k: C^k \to C^{k+1}$. Resuelve el Laplaciano del Haz Celular de grado 0 y grado 1:
    $$\mathbf{\Delta}_0 = \delta_0^\top \mathbf{G}^{-1} \delta_0, \qquad \mathbf{\Delta}_1 = \delta_0 \delta_0^\top + \delta_1^\top \mathbf{G}^{-1} \delta_1$$
    Verifica que la dimensión del primer grupo de cohomología de Čech-de Rham sea exactamente nula:
    $$\dim H^1(K; \mathcal{F}) = \dim \left( \frac{\ker \delta_1}{\operatorname{im} \delta_0} \right) \equiv 0$$
    Si $\dim H^1 > 0$, existe una paradoja contractual insalvable (ejemplo: incompatibilidad entre el metrado del modelo BIM, los pliegos de SECOP II y las especificaciones técnicas).
2.  **Fase 2 (Orient — Análisis Espectral Krylov-Lanczos de Dirichlet):**  
    Somete la energía de Dirichlet del haz $E(x) = \|\delta_0 x\|_{\mathbf{G}^{-1}}^2$ a la extracción de autovalores mediante el método iterativo de Krylov-Lanczos, evitando la inversión cuadrática del Laplaciano. Audita que el segundo autovalor ($\lambda_2(\mathbf{\Delta}_0)$, la conectividad de Fiedler del haz) satisfaga la cota isoperimétrica de Cheeger:
    $$\lambda_2(\mathbf{\Delta}_0) \ge \frac{h(\mathcal{F})^2}{2}$$
    donde $h(\mathcal{F})$ es la constante isoperimétrica de corte de la red de comunicación.
3.  **Fase 3 (Decide/Act — Proyección Isoperimétrica y Colapso):**  
    Si se detecta una divergencia de energía o si $H^1(K; \mathcal{F}) \neq \mathbf{0}$, el soberano colapsa el veredicto en Heyting a $\mathtt{VETOED}$ ($\top$). Automáticamente invalida el pasaporte de telemetría e impide la propagación de la propuesta hacia el comité de contratación pública.

---

## 3. Los Motores Espectrales y Oráculos de Cómputo (FPU Secure — Bottom-Up)

A diferencia de los soberanos de calibre, los motores espectrales del Estrato Strategy son módulos ciegos de alta precisión que ejecutan operaciones algebraicas, integraciones de ecuaciones diferenciales y simulaciones estocásticas en la FPU sin poder de veto directo.

### 3.1. `laplace_oracle.py` — El Tribunal de Estabilidad LTI y Dinámica de Control

El módulo `laplace_oracle.py` (`LaplaceOracle`) modela el comportamiento dinámico del flujo de caja del proyecto como un sistema lineal e invariable en el tiempo (LTI) en el dominio de la frecuencia compleja $s = \sigma + j\omega$.

#### 1. Formulación del Plano-S, Criterio de Jury y Raíces Características
El motor proyecta el balance de entradas de capital (anticipos, actas parciales) y salidas (compras, nómina) sobre una función de transferencia continua $H(s)$:

$$H(s) = \frac{Y(s)}{X(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \dots + b_0}{a_n s^n + a_{n-1} s^{n-1} + \dots + a_0} = K \frac{\prod_{j=1}^m (s - z_j)}{\prod_{i=1}^n (s - p_i)}$$

El motor evalúa en la FPU las siguientes condiciones métricas y espectrales:
*   **Polos Dominantes y Estabilidad BIBO (Bounded-Input, Bounded-Output):**  
    Calcula las raíces del denominador $\det(s\mathbf{I} - \mathbf{A}) = 0$. Exige estabilidad asintótica en el semiplano abierto izquierdo de Laplace (LHP):
    $$\forall i \in \{1, \dots, n\}, \qquad \Re(p_i) < 0$$
    Si un polo migra al semiplano derecho ($\Re(p_i) \ge 0$), el flujo de caja se vuelve explosivo, prediciendo la quiebra financiera del proyecto ante perturbaciones menores.
*   **Criterio de Estabilidad Discreta de Jury:**  
    En el espacio z-transformado discreto $H(z)$, valida que las raíces del polinomio característico $P(z) = a_n z^n + a_{n-1} z^{n-1} + \dots + a_0 = 0$ se ubiquen estrictamente dentro del círculo unitario:
    $$\forall i, \qquad |z_i| < 1.0$$
*   **Factor de Amortiguamiento y Margen de Fase:**  
    Mide el factor de amortiguamiento $\zeta$ y la frecuencia natural $\omega_n$ del par de polos complejos conjugados dominantes $p_{1,2} = -\zeta \omega_n \pm j \omega_n \sqrt{1 - \zeta^2}$:
    $$\zeta = \frac{-\Re(p)}{\sqrt{\Re(p)^2 + \Im(p)^2}} \ge 0.707$$
    Un sistema subamortiguado ($\zeta < 0.707$) genera fluctuaciones violentas en las necesidades de liquidez.
*   **Exponente Máximo de Lyapunov ($\lambda_{\max}$):**  
    Estima la tasa de divergencia de trayectorias financieras cercanas en el espacio de fase:
    $$|e(k)| \approx |e(0)| \cdot e^{\lambda_{\max} k} \implies \lambda_{\max} < 0$$
    Si $\lambda_{\max} > 0$, el motor registra un estado de caos determinista en la tesorería.

---

### 3.2. `financial_engine.py` & `probability_models.py` — El Oráculo Estocástico y WACC Topológico

El motor `financial_engine.py` y sus modelos auxiliares en `probability_models.py` reemplazan la evaluación contable estática por un análisis estocástico de cola pesada (Fat-Tail Risk) sobre la variedad de precios.

#### 1. Simulación de Monte Carlo y Métricas de Riesgo Extremo
El motor asigna distribuciones de probabilidad no negativas (log-normales o triangulares) a los precios de insumos volátiles (acero, concreto, combustible), ejecutando simulaciones masivas vectorizadas en la FPU mediante la arquitectura de Monte Carlo:

$$\ln(P_i) \sim \mathcal{N}\left(\mu_i - \frac{\sigma_i^2}{2}, \, \sigma_i^2\right)$$

Para acelerar la convergencia y acotar el error de muestreo, el motor aplica técnicas de reducción de varianza mediante **variantes antitéticas**:

$$\text{Error Estándar} = \frac{\sigma_{\mathrm{sim}}}{\sqrt{N_{\mathrm{sim}}}} \le 1.0\times 10^{-4}$$

Sobre la función de densidad de probabilidad integrada del costo total $f(C)$, el motor calcula dos indicadores no paramétricos de cola extrema:
*   **Valor en Riesgo ($\operatorname{VaR}_\alpha$):** Percentil de pérdida máxima al nivel de confianza $\alpha = 0.95$:
    $$\mathbb{P}\left( C \le \operatorname{VaR}_\alpha \right) = \alpha$$
*   **Déficit Esperado ($\operatorname{CVaR}_\alpha$ / Expected Shortfall):** Esperanza condicional de los sobrecostos que exceden el umbral $\operatorname{VaR}_\alpha$:
    $$\operatorname{CVaR}_\alpha = \mathbb{E}\left[ C \;\middle|\; C > \operatorname{VaR}_\alpha \right] = \frac{1}{1 - \alpha} \int_{\alpha}^1 \operatorname{VaR}_u \, du$$

#### 2. Costo de Capital Acoplado Topológicamente (WACC Topológico)
El motor no considera el Costo Promedio Ponderado de Capital (WACC) como una tasa fija. Lo formula como un funcional acoplado a los invariantes de forma del complejo simplicial táctico:

$$\mathrm{WACC}_{\mathrm{topológico}} = \mathrm{WACC}_{\mathrm{base}} \cdot \left( 1 + \gamma_0 (\beta_0 - 1) + \gamma_1 \beta_1 + \gamma_2 \max(0, \, 0.70 - \Psi) \right)$$

donde $\beta_0$ mide la fragmentación en islas de datos, $\beta_1$ es el número de socavones lógicos y $\Psi$ es el Índice de Estabilidad Piramidal. De este modo, la fragilidad topológica penaliza directamente la viabilidad económica del proyecto.

---

### 3.3. `business_agent.py` — Síntesis Categórica, Opciones Reales y RiskChallenger

A pesar de su sufijo `_agent`, en la arquitectura actual `business_agent.py` actúa como la fachada unificada y plataforma de cálculo de opciones reales, termodinámica estadística y evaluación de pivotes estratégicos.

#### 1. Valoración de Opciones Reales y Termodinámica de Flujo
El módulo cuantifica la flexibilidad estratégica de la gerencia frente a la incertidumbre del mercado:
*   **Modelo de Black-Scholes-Merton para la Opción de Esperar:**  
    Calcula el valor de diferir la firma del contrato de licitación $V_{\mathrm{esperar}}$ tratando el valor presente neto de los ingresos $V$ como activo subyacente y la inversión requerida $I$ como precio de ejercicio:
    $$V_{\mathrm{esperar}} = V N(d_1) - I e^{-r T} N(d_2)$$
    $$d_1 = \frac{\ln(V/I) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}}, \qquad d_2 = d_1 - \sigma \sqrt{T}$$
    Si $V_{\mathrm{esperar}} > \mathrm{VPN}_{\mathrm{inmediato}}$, el sistema prescribe esperar antes de comprometer capital.
*   **Temperatura del Sistema ($T_{\mathrm{sys}}$) y Ecuación de Arrhenius Financiera:**  
    Modela la volatilidad agregada como una temperatura efectiva $T_{\mathrm{sys}}$ (en Kelvin equivalentes). La tasa de probabilidad de quiebra $k_{\mathrm{quiebra}}$ obedece la ley de Arrhenius deformada:
    $$k_{\mathrm{quiebra}} = A \exp\left( -\frac{E_a \cdot \Psi}{k_B T_{\mathrm{sys}}} \right)$$
    donde $E_a$ es la barrera de capital de trabajo y $\Psi$ es la estabilidad piramidal.

#### 2. RiskChallenger: Auditoría Adversarial de Pivotes
El subsistema `RiskChallenger` evalúa cuatro estrategias de mitigación lateral ante alertas de sobrecosto:
1.  **Monopolio Coberturado:** Cobertura financiera mediante derivados ante concentraciones de insumos ($\Psi < 0.70$).
2.  **Opción de Espera:** Postergación de compras hasta la disipación de fiebres inflacionarias ($T_{\mathrm{sys}} > 50^\circ\mathrm{C}$).
3.  **Cuarentena Topológica:** Aislamiento y reestructuración de subgrafos con dependencias circulares ($\beta_1 > 0$).
4.  **Improbability Override:** Anulación de sesgos de cola corta en escenarios de Cisne Negro.

---

## 4. Matriz de Traducción Biyectiva a "Riesgo, Dolor y Dinero"

Bajo el **Funtor de Traducción Semántica Piramidal ($\Phi_{\mathrm{sem}}$)**, la abstracción matemática de los motores y soberanos del Estrato Strategy se traduce en salvaguardas financieras concretas para la Junta Directiva y los comités de licitación en Colombia (SECOP II y Mandato BIM 2026):

| Componente Estratégico | Invariante Matemático / Físico | Diagnóstico Técnico de Obra | Impacto Financiero Real ("Riesgo, Dolor y Dinero") |
| :--- | :--- | :--- | :--- |
| **`pipeline_director_agent.py`** | Matriz de Adyacencia Nilpotente ($\mathbf{A}^\nu = \mathbf{0}$) y Poset DIKW. | Aciclicidad estricta en el orden de ejecución del presupuesto. | **Riesgo & Dolor:** Bloqueo de la consolidación de pliegos por bucles de ejecución.<br>**Dinero:** Inhabilitación legal en SECOP II y pérdida de la licitación pública por entrega extemporánea. |
| **`pipeline_director_agent.py`** | Secuencia Exacta de Mayer-Vietoris ($\Delta \beta_1 \equiv 0$). | Nulidad de ciclos fantasma al fusionar capítulos o bases de datos ($A \cup B$). | **Riesgo & Dolor:** Duplicación inadvertida de fletes e insumos cruzados.<br>**Dinero:** Sobrecostos no presupuestados y destrucción del margen de utilidad bruta ($>15\%$). |
| **`sheaf_cohomology_orchestrator_agent.py`** | Primer Grupo de Cohomología Nulo ($\dim H^1(K; \mathcal{F}) = 0$). | Ausencia de frustración de calibre o contradicciones normativas. | **Riesgo & Dolor:** Paradoja contractual entre planos BIM, pliegos y Excels.<br>**Dinero:** Paro de obra civil por disputas legales y demandas de reequilibrio económico. |
| **`sheaf_cohomology_orchestrator_agent.py`** | Cota Isoperimétrica de Cheeger ($\lambda_2 \ge h(\mathcal{F})^2 / 2$). | Resiliencia de la red de comunicación celular del proyecto. | **Riesgo & Dolor:** Desconexión entre interventoría, contratista y proveedores.<br>**Dinero:** Retrasos en actas de obra y multas por mora contractual de hasta el $10\%$ del valor del contrato. |
| **`laplace_oracle.py`** | Polos en Semi-Plano Izquierdo ($\Re(p_i) < 0$). | Estabilidad BIBO asintótica del flujo de fondos en el dominio $s$. | **Riesgo & Dolor:** Modelo de tesorería intrínsecamente explosivo ("Bomba de Tiempo").<br>**Dinero:** Quiebra de caja de la constructora y abandono del proyecto ("Elefantes Blancos"). |
| **`laplace_oracle.py`** | Exponente Máximo de Lyapunov Negativo ($\lambda_{\max} < 0$). | Ausencia de caos determinista en la serie temporal del error. | **Riesgo & Dolor:** Pérdida total de previsibilidad sobre el presupuesto ejecutado.<br>**Dinero:** Veto absoluto. Disparo del disyuntor Crowbar en silicio antes de drenar la caja. |
| **`financial_engine.py`** | Déficit Esperado al 95% ($\operatorname{CVaR}_{0.95}$) y Monte Carlo. | Cuantificación exacta del riesgo de cola pesada en insumos clave. | **Riesgo & Dolor:** Subestimación de shocks de precios en acero y cemento.<br>**Dinero:** Agotamiento de la reserva de contingencia e incremento drástico del WACC. |
| **`business_agent.py`** | Modelo Black-Scholes para Opciones Reales ($V_{\mathrm{esperar}}$). | Valoración de la flexibilidad estratégica de diferir compras. | **Riesgo & Dolor:** Compromiso prematuro de capital durante fiebres inflacionarias.<br>**Dinero:** Destrucción del Valor Presente Neto (VPN) por compras a precio pico de mercado. |

---

## 5. El Tribunal de Silicio ESP32 y Actuación Crowbar (< 400 ns)

La seguridad estratégica del presupuesto no se confía a políticas lógicas de software en la nube que puedan ser eludidas o manipuladas. La Malla Agéntica extiende su brazo de censura hacia el territorio mediante el **Tribunal de Silicio perimetral (ESP32 Crowbar)**.

```
 [ ESTRATO STRATEGY: EVALUACIÓN DE INVARIANTES EN LA FRECUENCIA Y COHOMOLOGÍA ]
                                        │
                                        ▼
                  ¿Veto en Retículo de Heyting Ω₃? (Top = VETOED)
                  (Re(p_i) ≥ 0,  Δβ₁ > 0,  dim H¹ > 0,  λ_max > 0)
                                        │
                                        ▼
            [ REDUCCIÓN MONOIDAL A SILICIO REAL: μ: Ω₃ ⟶ ℤ₂ (Top ↦ 1) ]
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │ MICROCONTROLADOR PERIMETRAL ESP32                    │
             │ · Deserialización del Pasaporte SHA-256             │
             │ · Ejecución de subrutina C++ isVerdictCoherent()    │
             │ · Detección de mismatch epistémico en RAM            │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼ (t_actuation ≤ 398.95 ns)
             ┌──────────────────────────────────────────────────────┐
             │ MEMORIA ESTÁTICA IRAM Y PUERTO GPIO14               │
             │ · ISR ejecuta conmutación instantánea GPIO14 ↦ HIGH  │
             │ · Disparo de compuerta del Tiristor BT151 (Crowbar) │
             └──────────────────────────┬───────────────────────────┘
                                        │
                                        ▼
               [ PARÁLISIS MECÁNICA EN SECO DE BOMBAS Y MEZCLADORAS ]
```

1.  **Reducción Monoidal de Heyting ($\mu: \Omega_3 \to \mathbb{Z}_2$):**  
    Cuando los Soberanos del Estrato Strategy (`pipeline_director_agent.py` o `sheaf_cohomology_orchestrator_agent.py`) o los Oráculos de FPU detectan una violación insalvable (ejemplo: $\Re(p_i) \ge 0$, $\Delta \beta_1 > 0$, $\dim H^1 > 0$ o $\lambda_{\max} > 0$), el veredicto en el clasificador de subobjetos colapsa al Supremo terminal **`VETOED` ($\top$)**.
2.  **Verificación de Mismatch Ciber-Físico:**  
    El pasaporte de telemetría firmado criptográficamente con doble sello SHA-256 (`decision_sha256` y `digital_signature_sha256`) se transmite al firmware C++ del microcontrolador **ESP32** ubicado en la obra civil. La subrutina `isVerdictCoherent()` valida los residuos de estabilidad en RAM. Si detecta un intento de override no autorizado o una alteración de datos, confirma la existencia de un *mismatch* epistémico.
3.  **Ejecución de ISR en IRAM ($t_{\mathrm{actuation}} \le 398.95\text{ ns}$):**  
    El microcontrolador desvía síncronamente el hilo de ejecución hacia la **Interrupt Service Routine (ISR)** alojada estáticamente en su memoria ultra-rápida **IRAM**, evitando latencias de bus de almacenamiento.
4.  **Disparo de Hardware Crowbar (BT151 / GPIO14):**  
    La ISR conmuta el pin físico de hardware **GPIO14 a nivel alto (HIGH)** en menos de $400\text{ ns}$. Esta señal inyecta corriente directa a la compuerta del tiristor rápido de potencia **BT151 (circuito Crowbar)**, cortocircuitando limpiamente la línea de alimentación eléctrica. Las mezcladoras de concreto y bombas hidráulicas en sitio se paralizan mecánicamente en seco en el milisegundo cero, anulando la transacción corrupta en el mundo físico antes de consolidar pérdidas patrimoniales ante el SECOP II.

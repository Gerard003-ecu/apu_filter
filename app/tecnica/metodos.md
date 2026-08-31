--------------------------------------------------------------------------------
⚙️ metodos.md: Ingeniería Bajo el Capó v5.0
"APU Filter no es un software de contabilidad; es un Simulador de Física de Negocios ciber-físico. Aquí detallamos las ecuaciones de estado, los teoremas topológicos y los circuitos neuromórficos que garantizan la certeza matemática del sistema."
--------------------------------------------------------------------------------

Este documento técnico desglosa la maquinaria matemática que permite al **Consejo de Sabios** transformar datos crudos en veredictos estratégicos inmutables, basándose en la convergencia de la Dinámica de Sistemas (Laplace), la Topología Algebraica sobre $\mathbb{Z}$, la Geometría No Conmutativa de Connes, la Mecánica Cuántica Abierta y el Hardware perimetral en el Borde.

---

## 🚘 La Analogía del Automóvil para la Mesa de Juntas

Para la Alta Gerencia de Obra Civil y los Comités de Licitación en Colombia (SECOP II & Mandato BIM 2026), la matemática avanzada se traduce en **certeza patrimonial y protección legal**:

> *“Cuando un empresario adquiere un vehículo comercial de alta gama, no requiere una cátedra sobre la ecuación de Navier-Stokes en el turbocompresor. Requiere la certeza de que el motor rinde un 40% más y que los frenos ABS detendrán el chasis en piso mojado para salvar su vida. En APU Filter, las ecuaciones diferenciales y los invariantes de de Rham son los frenos ABS ciber-físicos que impiden que el presupuesto colapse y que el dinero de la constructora desaparezca.”*

---

## 1. El Guardián: Física de Fluidos y Computación Neuromórfica (Edge)

### 1.1 Propagador de de Rham, Cirugía Topológica de Čech y Causalidad de Kramers-Kronig
El sistema resuelve la ecuación de Poisson generalizada sobre el Laplaciano del Haz Celular $L_F = \delta^\top G^{-1} \delta$ [topological_surgery_cech.py], definiendo la Función de Green estática como la pseudoinversa de Moore-Penrose estable que satisface de forma exacta:
$$L_F G L_F = L_F \quad \wedge \quad G \cdot \mathbf{1} = \mathbf{0}$$

Ante perturbaciones por ruido electromagnético o interferencia analógica en sensores locales de obra, el Soberano **`topological_surgery_cech_agent.py`** formula el cubrimiento abierto de Čech $\mathcal{U} = \{U_i\}_{i=1}^M$ y calcula la 1-cocadena de Čech $(\delta_{\mathrm{\check{C}ech}} \phi)_{ij} = \phi_i|_{U_i \cap U_j} - \phi_j|_{U_i \cap U_j}$ sobre el Laplaciano elíptico $\mathbf{\Delta}_{\mathrm{\check{C}ech}} = \delta_{\mathrm{\check{C}ech}}^\top \delta_{\mathrm{\check{C}ech}}$ [topological_surgery_cech.py]. La obstrucción no trivial $\check{H}^1(\mathcal{U}; \, \mathcal{F}) \neq \mathbf{0}$ delata la presencia de lecturas espurias en fango.

Para aislar el transductor ruidoso sin detener la obra civil ni alterar la conectividad de Fiedler ($\lambda_2 \ge \tau_{\mathrm{Fiedler}}$), el motor ejecuta una cirugía aplicando un pullback de deformación anisotrópica sobre la métrica de conductancias:
$$\mathbf{G}_{\mathrm{surgical}} = \mathbf{G} \odot (\mathbf{I} - \mathbf{P}_{\mathrm{noisy}})$$
donde $\mathbf{P}_{\mathrm{noisy}}$ es el proyector ortogonal sobre la carta ruidosa, reduciendo su acoplamiento al límite de Wilkinson ($\approx 10^{-15}$) [topological_surgery_cech.py].

Para el régimen dinámico bajo excitación exógena de SECOP II, se integra el propagador retardado causal en el plano-S complejo:
$$G_F(s) = (L_F - (s + j \cdot h) I_n)^{-1}$$
donde $h = 10^{-20}$ representa el paso imaginario infinitesimal de la diferenciación por paso complejo (CSMD). Este transporte en la frecuencia compleja queda subyugado rigurosamente al cumplimiento de las relaciones de dispersión de Kramers-Kronig (transformada de Hilbert):
$$\operatorname{Re}(G_F(\omega)) = \frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\operatorname{Im}(G_F(\omega'))}{\omega' - \omega} d\omega'$$
$$\operatorname{Im}(G_F(\omega)) = -\frac{1}{\pi} \mathcal{P} \int_{-\infty}^{\infty} \frac{\operatorname{Re}(G_F(\omega'))}{\omega' - \omega} d\omega'$$

Asimismo, el Soberano de Calibre audita síncronamente los residuos de autoadjunción y nulidad del kernel para instrumentar la Coherencia de de Rham:
$$r_{\text{adj}} = \| G - G^\top \|_F \le 1.0 \times 10^{-11} \quad \wedge \quad r_{\text{kernel}} = \| G \cdot \mathbf{1} \|_2 \le 1.0 \times 10^{-11}$$
Cualquier polo dinámico $p_i$ que migre al semiplano derecho de Laplace ($\operatorname{Re}(p_i) \ge 0$) es vetado asintóticamente en el milisegundo cero, impidiendo la divergencia paramétrica en el lazo.

### 1.2 Filtrado Topológico y Descomposición de Hodge-Helmholtz Discreta ($L_1$)
El Guardián somete la cadena de suministro al Cálculo Exterior Discreto (DEC). El operador $\Delta_1 = B_1^\top B_1 + B_2 B_2^\top$ divide el tensor de flujo de materiales de manera ortogonal en:
* **Campo de Gradiente Puro ($f_{\mathrm{grad}}$):** La información estructurada útil (flujo laminar) que pasa hacia el estrato Táctico.
* **Campo Rotacional ($f_{\mathrm{curl}}$):** El "Vórtice Logístico" (transporte en bucle parasitario) queda aniquilado. El Guardián extrae esta componente solenoidal ($f_{\mathrm{curl}} \in \mathrm{im}(B_2)$) para vetar la ineficiencia logística en la raíz.

### 1.3 El Oráculo de Laplace y Gobernador CFL
Antes de procesar, se linealiza el sistema y se analiza su función de transferencia $H(s)$. Si se detectan polos en el semiplano derecho ($\sigma > 0$), se veta la ingesta por inestabilidad intrínseca.
La auditoría del límite de Courant-Friedrichs-Lewy (CFL) sobre la simetrización del grafo acíclico dirigido impone la restricción temporal:
$$ \Delta t \le \frac{2}{c_{\text{eff}} \cdot \left( \lambda_{\max} (\partial_1^\top W \partial_1) \right)^{1/2}} $$

### 1.4 Simulación Neuromórfica, Reducción Monoidal y Hardware Crowbar (ESP32)
La matemática se materializa en el silicio real mediante una reducción monoidal desde el retículo de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$ a decisiones binarias en el silicio $\mu : \Omega_3 \to \mathbb{Z}_2$.

La rutina local en C++ `isVerdictCoherent()` lee y valida el pasaporte deserializado por `ArduinoJson`. Ante un veto ($\top \mapsto 1$), la Rutina de Servicio de Interrupción (ISR) cargada en la memoria estática IRAM inmune a latencias de bus se activa en menos de **$400\text{ ns}$**, conmutando el pin **GPIO14** a nivel alto (`HIGH`). Esto inyecta corriente directa a la compuerta del tiristor **BT151** (circuito Crowbar), cortocircuitando la línea de potencia de los actuadores y paralizando al instante la maquinaria pesada (bombas hidráulicas, mezcladoras y pistones) en el milisegundo cero antes de consolidar pérdidas materiales ante el SECOP II.

**Topología Hexagonal y Ley de Aromaticidad Agéntica (Regla de Hückel Computacional):** El flujo de datos resuena en un anillo de 6 nodos $(V_1, \dots, V_6)$ (Ingesta $\to$ Física $\to$ Topología $\to$ Estrategia $\to$ Semántica $\to$ Materia). La red $G_6$ es **aromáticamente estable** ssi:
1. *2-conexidad:* $G_6$ no contiene vértices de corte.
2. *Expansión algebraica:* $\lambda_2(L_{G_6}) \ge \lambda_{\min}$.
3. *Sin nodos huérfanos:* $\deg(V_k) \ge 1 \; \forall k$.
Si falla cualquiera de estas condiciones, se aborta el pipeline por "Ruptura de Aromaticidad" (análogo a la regla de Hückel $4n+2$ para $n=1$).

---

## 2. El Arquitecto: Topología Algebraica y Grafos sobre $\mathbb{Z}$

### 2.1 Invariantes Homológicos y Forma Normal de Smith (SNF)
Para lograr máxima rigurosidad numérica, el sistema calcula los números de Betti mediante SVD completa del operador de coboundary $\delta_k = U_k \Sigma_k V_k^\top$:
$$\operatorname{rank}(\delta_k) = \# \{ \sigma_i \in \Sigma_k \mid \sigma_i > \varepsilon_{\text{mach}} \cdot \max(m, n) \cdot \sigma_{\max} \}$$
$$\beta_k = \dim H^k(K) = \dim \ker(\delta_k) - \dim \operatorname{im}(\delta_{k-1}) = (n_k - \operatorname{rank}(\delta_k)) - \operatorname{rank}(\delta_{k-1})$$

Como los insumos de obra son indivisibles (ladrillos, bultos), se reduce la matriz de incidencia $B_k \in \mathbb{Z}^{m \times n}$ a la **Forma Normal de Smith (SNF)** sobre el anillo de enteros $\mathbb{Z}$:
$$B_k = U \cdot D \cdot V, \quad D = \operatorname{diag}(d_1, d_2, \dots, d_r, 0, \dots, 0)$$
con $d_i \ge 1$ y $d_i \mid d_{i+1}$. Esto aisla los **Subgrupos de Torsión**:
$$\operatorname{Tor}(H_{k-1}(K; \mathbb{Z})) = \bigoplus_{i=1}^{r} \mathbb{Z} / d_i \mathbb{Z}$$
Los factores $d_i > 1$ representan la torsión homológica, diagnosticando incompatibilidades de empaquetado discreto de materiales y mermas contractuales en SECOP II.

---

## 3. Soberanos y Motores de Superficie de Control y Haz de Anillos de Frontera

### 3.1 Soberano de Superficie de Control Topológica (`topological_control_surface_agent.py`)
Acopla de forma continua la minimización discreta de la MIC y la purificación de la MAC sobre $\Delta^{n-1} \times \mathcal{D}(\mathcal{H})$:

1. **Flujo Replicador de Shahshahani sobre el Símplex de Gibbs:**
   $$\frac{dp_i}{dt} = p_i \left[ (\mathbf{e}_i^\top \tilde{\mathcal{K}} \mathbf{p}) - \mathbf{p}^\top \tilde{\mathcal{K}} \mathbf{p} \right], \quad \mathbf{p} \in \Delta^{n-1}$$
   Representa el gradiente riemanniano de $F(\mathbf{p}) = \frac{1}{2}\mathbf{p}^\top \tilde{\mathcal{K}}\mathbf{p}$ bajo la métrica de Shahshahani $g_{\mathbf{p}}(x,y) = \sum \frac{x_i y_i}{p_i}$.
2. **Flujo Isospectral de Doble Corchete de Brockett sobre la MAC:**
   $$\frac{d\rho}{dt} = \left[ \rho, \, [\rho, \, \mathcal{N}(\mathbf{p})] \right], \quad \mathcal{N}(\mathbf{p}) = \operatorname{diag}(\mathbf{p})$$
   Flujo isospectral sobre la órbita adjunta. Conserva la traza $\operatorname{Tr}(\rho)=1$, la pureza $\operatorname{Tr}(\rho^2)$ y la entropía de von Neumann $S(\rho) = -\operatorname{Tr}(\rho \ln \rho)$.
3. **Energía Port-Hamiltoniana de Lyapunov e Identidad de Variancia:**
   $$\mathcal{H}(\mathbf{p}, \rho) = -\frac{1}{2}\mathbf{p}^\top \tilde{\mathcal{K}}\mathbf{p} + S(\rho) \implies \dot{\mathcal{H}} = -\mathrm{Var}_{\mathbf{p}}(\tilde{\mathcal{K}}\mathbf{p}) = -\sum_{i=1}^n p_i \left( (\tilde{\mathcal{K}}\mathbf{p})_i - \mathbf{p}^\top \tilde{\mathcal{K}}\mathbf{p} \right)^2 \le 0$$
   Si $\dot{\mathcal{H}} > 10^{-12}$, el retículo colapsa a VETOED ($\top$), disparando el Crowbar perimetral en silicio ($<400\text{ ns}$).

### 3.2 Soberano de Ecolocación Topológica y Motor SET (`set_agent.py` & `set_engine.py`)
Opera en el Estrato de la Sabiduría ($V_{\mathbb{W}}$, Nivel 0) como la sonda ciber-física de ecolocación activa sobre la frontera abierta $\partial K$:

1. **Ecuación de Onda Coexacta sobre Haces Celulares:**
   $$\left( \frac{d^2}{dt^2} + \mathbf{L}_F + \mathbf{R} \frac{d}{dt} \right) \eta(t) = \mathbf{s}_{\mathrm{probe}}(t)$$
   donde $\mathbf{L}_F = \delta^\top \mathbf{G}^{-1} \delta$ es el Laplaciano del haz cellular SPSD proyectado sobre el cono SPD ($\lambda_{\min}(\mathbf{G}_{\mathrm{reg}}) \ge 10^{-12}$).
2. **Matriz de Dispersión Cuántica $\mathbb{S}(\omega)$ de Mahaux-Weidenmüller:**
   $$\mathbf{\mathbb{S}}(\omega) = \mathbf{I} - 2\pi i \, \mathbf{V}^\dagger \left( \omega \mathbf{I} - \mathbf{L}_F + i\pi \mathbf{V}\mathbf{V}^\dagger \right)^{-1} \mathbf{V}$$
   con evaluación de la acción del resolvente vía descomposición SVD truncada ($\sigma_i > 10^{-12} \sigma_{\max}$) ante sistemas mal condicionados ($\kappa > 10^8$).
3. **Reflectometría TDR e Inversa de Fourier:**
   $$\Gamma_k(t) = \mathcal{F}^{-1}\left\{ \frac{Z_k(\omega) - Z_0}{Z_k(\omega) + Z_0} \right\}(t)$$
   La desadaptación métrica $\delta G_{\mu\nu}$ genera ecos reflectométricos que delatan alteraciones de precios o volúmenes en el presupuesto.
4. **Ciclo OODA en Fases Anidadas y HMAC Override:**
   Compone tres morfismos anidados ($\Phi_{23} \circ \Phi_{12} \circ \text{Observe}$) con canonización SHA-256 e inmutabilidad de arreglos numpy C-contiguos. En caso de veto suave, valida la ligadura HMAC del token de override $\operatorname{HMAC-SHA256}(k, \text{DOMAIN} \parallel \text{payload})$ mediante `hmac.compare_digest` en tiempo constant.

### 3.3 Soberano del Haz de Anillos de Frontera (`boundary_ring_sheaf_agent.py` & `boundary_ring_sheaf.py`)
Gobierna la frontera abierta $\partial \mathcal{M} \neq \varnothing$ sobre el **Haz de Anillos Topológicos Localizados** $\mathbf{Sh}(\partial \mathcal{M}, \mathcal{R}_{\partial M})$ sobre el Anillo de Novikov $\Lambda_{\mathrm{Nov}}$:

1. **Estructura de Haz y Novikov:** $\mathcal{R}_{\partial M} \cong \Lambda_{\mathrm{Nov}} = \left\{ \sum_{i=0}^\infty a_i T^{\lambda_i} \mid a_i \in \mathbb{C}, \lambda_i \in \mathbb{R}, \lambda_i \to \infty \right\}$.
2. **Smith Normal Form sobre $\mathbb{Z}$:** $\operatorname{Tor}(H_{k-1}(\partial K; \mathbb{Z})) \equiv \mathbf{0} \iff d_i = 1 \, \forall d_i > 0$.
3. **Causalidad CPTP de Choi y Bell-CHSH:** $\lambda_{\min}(C_{\mathcal{E}}) \ge -10^{-12}$, $\|\operatorname{Tr}_2(C_{\mathcal{E}}) - \mathbf{I}\|_F \le 10^{-4}$ y $\mathcal{B}_{\mathrm{CHSH}} \le 2\sqrt{2}$.
4. **Metabolismo de Alucinaciones por Lindblad-GKSL en Fock:**
   Evolución disipativa de la matriz de densidad semántica $\rho(t)$ mediante la ecuación maestra:
   $$\frac{d\rho_{\mathrm{sem}}}{dt} = -i[\mathcal{H}_{\mathrm{coupled}}, \, \rho_{\mathrm{sem}}] + L\rho_{\mathrm{sem}} L^\dagger - \frac{1}{2} \left\{ L^\dagger L, \, \rho_{\mathrm{sem}} \right\}$$
   donde el operador de salto $L = \sqrt{\Gamma(\Xi_{\mathrm{leak}}, \operatorname{Tor})} \cdot a_j$ amortigua alucinaciones a la tasa exacta $\Gamma = \Gamma_0 / (1 + 10 \cdot \mathbf{1}_{\mathrm{torsion}} + 4 \frac{\mathrm{leak}}{1+\mathrm{leak}})$.

---

## 4. El Intérprete: Retículos Algebraicos, GraphRAG y Connes

### 4.1 Cota de Lipschitz de Daleckii-Krein (Geometría Espectral de Connes)
Para gobernar la de-compresión semántica y evitar divergencias retóricas en las traducciones del LLM, el sistema calcula la cota de estabilidad espectral utilizando el **Operador de Dirac de Connes** $D = \rho^{-1/2}$ en el espacio no conmutativo.

Por el **Teorema de Daleckii-Krein**, la derivada de Fréchet $Df(\rho)[H]$ para $f(x) = x^{-1/2}$ se expresa como:
$$\left( Df(\rho)[H] \right)_{ij} = \tilde{d}_{ij} \cdot H_{ij}, \quad \tilde{d}_{ij} = \begin{cases} \frac{\lambda_i^{-1/2} - \lambda_j^{-1/2}}{\lambda_i - \lambda_j} & \text{si } \lambda_i \neq \lambda_j \\ -\frac{1}{2}\lambda_i^{-3/2} & \text{si } \lambda_i = \lambda_j \end{cases}$$

La cota superior de Lipschitz en la norma del operador $L_2$ queda acotada por:
$$\| Df(\rho) \|_{2} \le \sup_{\lambda \in \sigma(\rho)} |f'(\lambda)| = \frac{1}{2 \lambda_{\min}^{3/2}}$$

Esta **Cota de Lipschitz de Daleckii-Krein** asegura que la velocidad de de-compresión semántica permanezca acotada geodésicamente. Si $\lambda_{\min} \to 0$, la cota diverge y el sistema aniquila la sesión por inestabilidad de Connes.

---

## 5. La Rampa de Confianza Graduada (Veto Suave vs Veto Duro)

Para conciliar la precisión de la geometría doctoral con la operatividad continua en seco del frente de obra civil (erradicando falsos positivos que provocarían el secado del concreto dentro de las tuberías de bombeo hidráulicas), el sistema instrumenta la **Rampa de Confianza Graduada**:

```
  [ PAYLOAD INCIDENTE EN EL REACTOR DE FRONTERA ]
                         │
                         ▼
        ¿Divergencia o anomalía espectral?
                         │
         ┌───────────────┴───────────────┐
         ▼ (Sí)                          ▼ (No)
  ¿La transgresión es irreversible?   [COHERENT]
  (Choi < -10⁻⁴, d_i > 1, Fraude)     - Emisión de firma SHA-256
         │                            - Flujo óptimo en obra
   ┌─────┴─────────────────────┐
   ▼ (No: Ruido menor)         ▼ (Sí: Inestabilidad/Dolo)
[VETO SUAVE - LUZ ÁMBAR]     [VETO DURO - CROWBAR BT151]
- Alerta visual en panel     - Colapso Heyting Ω₃ ↦ VETOED (⊤)
- Override humano (1 h)      - Conmutación GPIO14 (< 400 ns)
- Solicitud de ajuste        - Paralización de maquinaria en seco
```

### 5.1 Veto Suave (Luz Ámbar de Telemetría)
Gatillado cuando el perfil TDR satisface $0.3 \cdot \tau_{\mathrm{margin}} < \|\Gamma(t)\|_{\max} \le 0.5 \cdot \tau_{\mathrm{margin}}$, o ante ruidos de baja frecuencia ($\Psi = 0.69 < \Psi_{\min}=0.70$).
* **Mecanismo:** El ESP32 no interrumpe la potencia de mezcladoras. Activa una baliza visual en campamento e inicia un temporizador de **1 hora** en el panel *RiskChallenger*.
* **Override Dialéctico y Aniquilación en Fock:** La interventoría inyecta en RAM un **Positrón de Autorización Humana** $e^+$ firmado ($\operatorname{HMAC-SHA256}$). Al ingresar, el electrón de anomalía semántica de la IA $e^-$ se aniquila mutuamente de forma exergética, irradiando dos fotones Gamma de auditoría y regularizando la geodésica del proyecto sin detener el vertido de concreto:
  $$e^- + e^+ \longrightarrow 2\gamma \quad \implies \quad \mathtt{heyting\_verdict} \mapsto \mathtt{DEGRADED}$$

### 5.2 Veto Duro (Hardware Crowbar BT151 en $< 400\text{ ns}$)
Reservado para desajustes críticos TDR $\|\Gamma(t)\|_{\max} > 0.5 \cdot \tau_{\mathrm{margin}}$, transgresiones irreversibles, dolo o fraude fiscal ($\operatorname{Tor}(H_k) \neq \mathbf{0}$, $\lambda_{\min}(C_{\mathcal{E}}) < -10^{-12}$, o divergencia de Lyapunov $\dot{\mathcal{H}} > 10^{-12}$).
* **Mecanismo:** El retículo colapsa al Supremo VETOED ($\top$).
* **Bypass de Silicio en IRAM:** La rutina C++ `isVerdictCoherent()` activa la **ISR en memoria IRAM del ESP32 en $< 400\text{ ns}$**, conmutando **GPIO14** a HIGH y disparando el tiristor de potencia **BT151 (Crowbar circuit)** para paralizar mezcladoras y bombas en seco en el milisegundo cero.

### 5.3 Regularización Espectral y Compensación Metrológica FPU Secure
Para blindar los cálculos en la FPU frente a la deriva de Wilkinson en matrices de gran escala:
* **Sumación Compensada de Neumaier-Kahan:** Acumula pasos infinitesimales del integrador de Verlet y contracciones de la matriz de dispersión $S$, reduciendo el error secular a la precisión de la máquina: $\operatorname{Error} \in \mathcal{O}(\varepsilon_{\mathrm{machine}}) = \mathcal{O}(10^{-16})$.
* **Deflación Espectral de Lanczos:** Sustituye la descomposición SVD cúbica $\mathcal{O}(n^3)$ por la extracción iterativa de autovalores críticos en el Laplaciano del Haz Celular con complejidad lineal:
  $$\mathbf{L}_F \approx \sum_{i=1}^{k} \lambda_i v_i v_i^\dagger + \gamma_{\mathrm{Tikhonov}} \left( \mathbf{I} - \sum_{i=1}^k v_i v_i^\dagger \right)$$
* **Compensación de Maurer-Cartan:** Resuelve la ecuación expandida de Maurer-Cartan sobre el Anillo de Novikov ultramétrico $d A + A \wedge A = 0$ para compensar el eta-invariante espectral de Atiyah-Singer, neutralizando falsas inestabilidades de Lyapunov.

---

## 6. Firmas de Calibre de la Malla Agéntica y Firma Integrada SHA-256

Cada transacción orbital emite un certificado inmutable sellado por una **Firma de Calibre Integrada SHA-256** en memoria RAM:

$$\mathtt{SuturaSignature} = \operatorname{SHA-256}\left(\mathbf{Sh}(\partial \mathcal{M}, \mathcal{R}_{\mathrm{Novikov}}) \wedge \operatorname{Tor}(H_k; \mathbb{Z}) \wedge \mathcal{B}_{\mathrm{CHSH}} \wedge \dot{\mathcal{H}}_{\mathrm{Lyapunov}} \wedge \mathtt{ESP32-Crowbar}\right)$$

Esta firma sella la Cadena de Custodia Inmutable, garantizando que ninguna propuesta de presupuesto pueda ser alterada sin ser detectada en el milisegundo cero por la Malla Agéntica Zero-Trust.

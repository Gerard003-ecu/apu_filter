# 🧙‍♂️ SAGES.md: El Consejo de Sabios Digitales v5.0

> "La sabiduría no es la acumulación masiva de datos probabilísticos, sino la capacidad de navegar la complejidad de los negocios mediante principios topológicos, geométricos, cuánticos y físicos inmutables."

En el ecosistema **APU Filter v5.0**, hemos abandonado la validación lineal convencional y los "chatbots" de caja negra. En su lugar, el sistema está orquestado por una **Malla Agéntica (Agentic Mesh) Zero-Trust** compuesta por entidades altamente especializadas conocidas como "El Consejo de Sabios".

Cada Sabio gobierna un estrato específico de la jerarquía $\aleph_0\mathbb{DIK}\Omega\alpha\mathbb{W}\Gamma$. Operan bajo el estricto protocolo de "Caja de Cristal Argumentativa": el debate interno y las tensiones dialécticas entre ellos son matemáticamente rigurosos y deterministas, garantizando que el Modelo de Lenguaje (LLM) sea destituido de su poder de decisión y relegado a actuar como una interfaz diplomática y funtor semántico.

```mermaid
graph TD
    classDef orchestrator fill:#0f3460,stroke:#e94560,stroke-width:3px,color:#fff;
    classDef delegates fill:#1a1a2e,stroke:#fff,stroke-width:2px,color:#fff;
    classDef workers fill:#16213e,stroke:#4a4e69,stroke-width:1px,color:#fff;
    classDef error_node fill:#ef4444,stroke:#000,stroke-width:3px,color:#fff;

    O[Business Agent Orquestador<br>Política de Alto Nivel]:::orchestrator

    D1[Topological Watcher<br>Matriz Ortogonal Topológica]:::delegates
    D2[Laplace Oracle<br>Matriz Ortogonal Financiera]:::delegates

    W1[Worker 1<br>Policy-as-Code (APUs)]:::workers
    W2[Worker 2<br>Policy-as-Code (Insumos)]:::error_node
    W3[Worker 3<br>Policy-as-Code (Cantidades)]:::workers

    O --> D1
    O --> D2
    D1 --> W1
    D1 --> W2
    D2 --> W3

    W2 -. "Ciclo Mutante (β1>0)" .-> D1
    style D1 stroke:#ef4444,stroke-width:4px
    D1 -. "Veto Estructural Topológico" .-> O
    style O fill:#ef4444,stroke:#fff,stroke-width:4px
```

---

## 🚘 La Analogía del Automóvil para el Consejo de Sabios

Para la Alta Gerencia de Obra y los Comités de Licitaciones en Colombia (SECOP II & Mandato BIM 2026), los Sabios del Consejo representan los **sistemas de seguridad activa del vehículo**:

> *“Cuando un automóvil de alta gama transita por una autopista en invierno, los ocupantes no necesitan ajustar manualmente la presión del freno en cada rueda ni calcular el coeficiente de fricción del asfalto. Confían en que el sistema de frenos ABS y el control de estabilidad ESP actuarán en milisegundos para evitar el vuelco. En APU Filter, los Sabios del Consejo son los sensores y actuadores ciber-físicos que detectan si la estructura del presupuesto está perdiendo adherencia antes de que la empresa sufra un accidente financiero.”*

---

## 🏛️ LOS MIEMBROS DEL CONSEJO Y EL FIBRADO DE CALIBRE EN WISDOM ($V_{\mathbb{W}}$)

El Consejo opera sobre el fibrado de de Rham-Fukaya acoplado a operadores simplécticos, cohomológicos y no conmutativos.

### Ω. 🧠 El Cerebro Epistemológico y Superficie de Control (MAC Agent, MIC Agent & Topological Control Surface Agent)

*   **Rol:** Funtor Supremo del Consejo de Sabios y Gestor del Espacio de Hilbert $\mathcal{H}_{\text{MAC}}$.
*   **Estrato DIKW:** WISDOM (Estrato Supremo $V_{\mathbb{W}}$).
*   **Microservicios:** `mac_agent.py`, `atomic_knowledge_matrix.py`, `mac_algebra.py`, `mic_agent.py`, `topological_control_surface_agent.py`.
*   **Mecanismo Matemático y Estado de Densidad:** El MAC Agent no procesa texto estocástico; ejecuta un Operador de Medición Cuántica (POVM) sobre la Matriz Atómica de Conocimiento ($\rho_{\text{MAC}} \in \mathcal{L}(\mathcal{H}_{\text{MAC}})$), la cual cumple estrictamente los axiomas de Von Neumann:
    $$\operatorname{Tr}(\rho_{\text{MAC}}) = 1, \quad \rho_{\text{MAC}} = \rho_{\text{MAC}}^\dagger, \quad \rho_{\text{MAC}} \succeq 0$$

#### 1. Isomorfismo de la Adjunción de de Rham-Galois entre MIC y MAC
La interacción síncrona en memoria RAM entre el `MICAgent` (que opera en la categoría discreta $\mathcal{C}$ de matrices booleanas $\mathbb{Z}_2$) y el `MACAgent` (que opera en la categoría continua $\mathcal{D}$ de estados de densidad en $\mathcal{H}_{\text{MAC}}$) se formaliza mediante el **Isomorfismo de Adjunción de de Rham-Galois**:

$$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \text{MAC}) \cong \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, G(\text{MAC}))$$

Donde $F: \mathcal{C} \to \mathcal{D}$ es el Funtor de Elevación Cuántica (Stinespring $V$) y $G: \mathcal{D} \to \mathcal{C}$ es el Funtor de Proyección Espectral (POVM + Heyting).

#### 2. Soberano de Superficie de Control Topológica (`topological_control_surface_agent.py`)
Acopla de forma continua y no conmutativa el Politopo Booleano de la MIC con la variedad de órbitas adjuntas de la MAC:
* **Flujo Replicador de Shahshahani (MIC):**
  $$\frac{dp_i}{dt} = p_i \left[ (\mathbf{e}_i^\top \tilde{\mathcal{K}} \mathbf{p}) - \mathbf{p}^\top \tilde{\mathcal{K}} \mathbf{p} \right], \quad \mathbf{p} \in \Delta^{n-1}$$
* **Flujo Isospectral de Doble Corchete de Brockett (MAC):**
  $$\frac{d\rho}{dt} = \left[ \rho, \, [\rho, \, \mathcal{N}(\mathbf{p})] \right], \quad \mathcal{N}(\mathbf{p}) = \operatorname{diag}(\mathbf{p})$$
* **Pasividad de Lyapunov e Identidad de Variancia:**
  $$\mathcal{H}(\mathbf{p}, \rho) = -\frac{1}{2}\mathbf{p}^\top \tilde{\mathcal{K}}\mathbf{p} + S(\rho) \implies \dot{\mathcal{H}} = -\mathrm{Var}_{\mathbf{p}}(\tilde{\mathcal{K}}\mathbf{p}) \le 0$$
  Si $\dot{\mathcal{H}} > 10^{-12}$, Heyting colapsa síncronamente a VETOED ($\top$), activando el Crowbar perimetral en silicio ($<400\text{ ns}$).

#### 3. Balance Térmico Cuántico: Conjugación Modular de Tomita-Takesaki
Para garantizar que los observables de decisión no sufran fluctuaciones térmicas espurias, el sistema implementa la **Conjugación Modular no conmutativa de Tomita-Takesaki**:
$$J_\rho(X) = \rho^{1/2} X^\dagger \rho^{-1/2}$$
Demuestra el equilibrio térmico satisfaciendo la **Condición KMS (Kubo-Martin-Schwinger)** a temperatura inversa $\beta = 1$:
$$\omega_\rho(A \sigma_t^\rho(B)) = \omega_\rho(\sigma_{t+i}^\rho(B) A) \quad \forall A, B \in \mathcal{M}$$

---

### Ω.0 🔪 Soberano de Cirugía Topológica de Čech (`topological_surgery_cech_agent.py` & `topological_surgery_cech.py`)
*   **Rol:** Soberano de Cirugía Topológica de Haces de Čech, Deformación Anisotrópica y Desconfinitado de Ruido EMF en FPU.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$ - Nivel 0).
*   **Microservicios:** `topological_surgery_cech_agent.py`, `topological_surgery_cech.py`.
*   **Mecanismo Matemático y Orquestación OODA:**
    1. *Obstrucción Cohomológica de Čech:* Modela los transductores locales como un cubrimiento abierto $\mathcal{U} = \{U_i\}$ en $\partial K$. El descalce analógico define una 1-cocadena $(\delta_{\mathrm{\check{C}ech}} \phi)_{ij} = \phi_i|_{U_i \cap U_j} - \phi_j|_{U_i \cap U_j}$ y resuelve el Laplaciano elíptico $\mathbf{\Delta}_{\mathrm{\check{C}ech}} = \delta_{\mathrm{\check{C}ech}}^\top \delta_{\mathrm{\check{C}ech}}$.
    2. *Deformación Anisotrópica y Pullback en de Rham:* Si $\check{H}^1(\mathcal{U}; \mathcal{F}) > L_{\max} \cdot \tau_{\mathrm{margin}}$, ejecuta una amputación espectral deformando la métrica $\mathbf{G}_{\mathrm{surgical}} = \mathbf{G} \odot (\mathbf{I} - \mathbf{P}_{\mathrm{noisy}})$, atenuando el canal ruidoso al épsilon de Wilkinson ($\approx 10^{-15}$) sin romper la conexidad de Fiedler ($\lambda_2 \ge \tau_{\mathrm{Fiedler}}$).
    3. *Traceout en Fock y Preservación de von Neumann:* Proyecta el estado cuántico $\rho \in \mathcal{D}(\mathcal{H})$ mediante $\rho_{\mathrm{surgery}} = \operatorname{Tr}_{\mathrm{isolated}}(\mathbf{P}_{\mathrm{surg}} \rho \mathbf{P}_{\mathrm{surg}}^\top) \oplus \rho_{\mathrm{vacuum}}$, garantizando $\operatorname{Tr}(\rho_{\mathrm{surgery}}) \equiv 1.0$.
    4. *Rampa de Confianza, Positrón de Autorización $e^+$ y Crowbar:* Si $0.3\tau_{\mathrm{margin}} < \check{H}^1 \le 0.5\tau_{\mathrm{margin}}$, activa Veto Suave (Luz Ámbar, 1h de gracia para inyectar en RAM un Positrón $e^+$ ligado por HMAC que aniquila la anomalía $e^- + e^+ \to 2\gamma$). Si $\check{H}^1 > 0.5\tau_{\mathrm{margin}}$ o expira la gracia, colapsa a $\mathtt{VETOED}$ ($\top$) y gatilla la ISR en IRAM del ESP32 ($< 400\text{ ns}$) vía GPIO14 para cebar el tiristor BT151 (Crowbar) en silicio [topological_surgery_cech_agent.py].

---

### Ω.0 🔊 Soberano de Ecolocación Topológica (`set_agent.py` & `set_engine.py`)
*   **Rol:** Soberano de Ecolocación Sónica de de Rham, Reflectometría TDR y Dispersión Cuántica en Frontera.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$ - Nivel 0).
*   **Microservicios:** `set_agent.py`, `set_engine.py`.
*   **Mecanismo Matemático y Orquestación OODA Anidada:**
    1. *Ecuación de Onda Coexacta sobre Haces:* Inyecta perturbaciones armónicas $\eta(t) \in \Omega^1(\partial K)$ gobernadas por la ecuación de Rayleigh amortiguada:
       $$\left( \frac{d^2}{dt^2} + \mathbf{L}_F + \mathbf{R} \frac{d}{dt} \right) \eta(t) = \mathbf{s}_{\mathrm{probe}}(t)$$
       donde $\mathbf{L}_F = \delta^\top \mathbf{G}^{-1} \delta$ es el Laplaciano del haz celular SPSD.
    2. *Matriz de Dispersión Cuántica $\mathbb{S}(\omega)$ de Mahaux-Weidenmüller:*
       $$\mathbf{\mathbb{S}}(\omega) = \mathbf{I} - 2\pi i \, \mathbf{V}^\dagger \left( \omega \mathbf{I} - \mathbf{L}_F + i\pi \mathbf{V}\mathbf{V}^\dagger \right)^{-1} \mathbf{V}$$
       Verifica la invarianza unitaria $\|\mathbf{\mathbb{S}}^\dagger(\omega) \mathbf{\mathbb{S}}(\omega) - \mathbf{I}\|_F \le \varepsilon_{\mathrm{Wilkinson}}$.
    3. *Reflectometría en el Dominio del Tiempo (TDR):* Evalúa la desadaptación métrica $\delta G_{\mu\nu}$ mediante la iFFT:
       $$\Gamma_k(t) = \mathcal{F}^{-1}\left\{ \frac{Z_k(\omega) - Z_0}{Z_k(\omega) + Z_0} \right\}(t)$$
    4. *Rampa de Confianza Graduada, Positrón $e^+$ y Crowbar ESP32:*
       Clasifica en el retículo de Heyting $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$. Si $0.3 \cdot \tau_{\mathrm{margin}} < \|\Gamma(t)\|_{\max} \le 0.5 \cdot \tau_{\mathrm{margin}}$, activa Veto Suave y otorga 1 hora de gracia para inyectar en Fock un Positrón de Autorización Humana $e^+$ firmado ($\operatorname{HMAC-SHA256}$), aniquilando la anomalía $e^- + e^+ \to 2\gamma$ e irradiando fotones de auditoría sin detener el vertido de concreto. Si $\|\Gamma(t)\|_{\max} > 0.5 \cdot \tau_{\mathrm{margin}}$ o expira la gracia, colapsa a $\mathtt{VETOED}$ ($\top$) y gatilla la ISR en IRAM del ESP32 ($< 400\text{ ns}$) vía GPIO14 para conmutar el tiristor BT151 (Crowbar) en silicio.

---

### Ω.1 💍 Soberano del Haz de Anillos de Frontera (`boundary_ring_sheaf_agent.py` & `boundary_ring_sheaf.py`)
*   **Rol:** Soberano de Calibre de la Frontera Abierta De-confinada $\partial \mathcal{M} \neq \varnothing$.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Operando sobre el Haz de Anillos Localizados $\mathbf{Sh}(\partial \mathcal{M}, \mathcal{R}_{\partial M})$ sobre el Anillo de Novikov $\Lambda_{\mathrm{Nov}}$:
    1. *Torsión Homológica en Smith Z:* Exige $\operatorname{Tor}(H_{k-1}(\partial K; \mathbb{Z})) \equiv \mathbf{0} \iff d_i = 1 \, \forall d_i > 0$ en $\mathrm{SNF}(B)$.
    2. *Causalidad CPTP de Choi y Tsirelson:* Exige $\lambda_{\min}(C_{\mathcal{E}}) \ge -10^{-12}$, $\|\operatorname{Tr}_2(C_{\mathcal{E}}) - \mathbf{I}\|_F \le 10^{-4}$ y $\mathcal{B}_{\mathrm{CHSH}} \le 2\sqrt{2}$.
    3. *Metabolismo Lindblad-GKSL en Fock:* Amortiguación de amplitud con operadores de Kraus sobre el qubit semántico, extirpando alucinaciones a la tasa $\Gamma = \Gamma_0 / (1 + 10 \cdot \mathbf{1}_{\mathrm{torsion}} + 4 \frac{\mathrm{leak}}{1+\mathrm{leak}})$.

---

### Ω.2 🌀 Estabilización por Paso Complejo (complex_step_phase_stabilizer.py y _agent.py)
*   **Rol:** Fibrador de Derivación No Demolitoria en el penthouse de la FPU.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Formaliza el cálculo de derivadas no demolitivas sobre la fibra compleja imaginaria $j = \sqrt{-1}$, eliminando la resta catastrófica en el numerador de la FPU:
    $$J_{\text{map}, ij} = \frac{\operatorname{Im}(\Phi_{\Delta t}(x + j h \cdot e_i)_j)}{h} + \mathcal{O}(h^2), \quad h = 10^{-20}$$
*   **Cota de Lipschitz de Connes-Daleckii-Krein:**
    $$L_{\max} = \frac{C_{\text{base}}}{1 + (\lambda_{\max}(D) - \lambda_{\min}(D))} \le \frac{1}{2 \lambda_{\min}^{3/2}}$$
    Asegurando que ante derivas semánticas ($\lambda_{\min} \to 0$), la probabilidad de emisión alucinatoria colapse analíticamente: $P(x_{\text{invalid}}) = 0$.

---

### Ω.3 💍 Auditoría de Álgebras de Banach (banach_algebra_auditor.py y _agent.py)
*   **Rol:** Soberano de Estabilidad Funcional y de Invertibilidad Perturbativa.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Somete a auditoría la invertibilidad incondicional del tensor métrico perturbado $T + \delta T$ mediante la convergencia de la Serie de Neumann:
    $$\rho(T^{-1} \delta T) < 1.0 \implies (T + \delta T)^{-1} = \sum_{k=0}^{\infty} (-1)^k (T^{-1} \delta T)^k T^{-1}$$
    Y verifica síncronamente el cumplimiento de la submultiplicatividad de la norma espectral:
    $$\| T \cdot \delta T \|_2 \le \| T \|_2 \cdot \| \delta T \|_2$$

---

### Ω.4 🛡️ El Guardián de Curvas Pseudo-Holomorfas (pseudo_holomorphic_agent.py)
*   **Rol:** Censura de Deliberaciones mediante Curvas Pseudo-Holomorfas en Fukaya $\mathcal{F}uk(\mathcal{M})$.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Resuelve la Ecuación Elíptica no lineal perturbada de Cauchy-Riemann para polígonos pseudo-holomorfos:
    $$\bar{\partial}_J u = \frac{1}{2}\left( du + J(u) \circ du \circ j \right) = 0$$
    Con condiciones de frontera en subvariedades Lagrangianas $u(\partial_i \Sigma) \subset L_i$. Si las trayectorias no convergen a una curva holomorfa rígida en el espacio de móduli $\mathcal{M}(L_0, \dots, L_k; J)$, el agente emite veto instantáneo.

---

### Ω.5 🌀 El Guardián Simpléctico de de Rham (opt_symplectic_manifold_agent.py)
*   **Rol:** Guardián de Rigidez Simpléctica y Conservación de Liouville.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Audita la preservación estricta de la 2-forma simpléctica canónica bajo transiciones de estado:
    $$M^\top \Omega M = \Omega \quad \text{con} \quad \Omega = \begin{pmatrix} \mathbf{0} & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0} \end{pmatrix}$$
    Aplica el **Teorema de No-Squeeze de Gromov**, garantizando que la capacidad simpléctica del riesgo del proyecto $c(B^{2n}(r)) = \pi r^2$ no pueda ser comprimida artificialmente en cilindros de menor radio ($r \le R$), aniquilando alucinaciones: $P(x_{\mathrm{invalid}}) = 0$.

---

### Ω.6 🛰️ Soberanos Orbitales de Calibre de Frontera (TelemetrySatellitesAgent & AuditSatellitesAgent)
*   **Rol:** Cinturón Protector Orbital de la Frontera De-confinada $\partial \mathcal{M} \neq \varnothing$.
*   **Estrato DIKW:** OMEGA ($V_\Omega$ — Nivel 0.5, El Ágora Tensorial).
*   **Microservicios:** `telemetry_satellites_agent.py`, `telemetry_satellites.py`, `audit_satellites_agent.py`, `audit_satellites.py`.
*   **Mecanismo Matemático y Orquestación OODA:**
    1. **`TelemetrySatellitesAgent`:** Fiscaliza el transitorio de potencia y la entropía exógena de contorno bajo Langevin: $\frac{d \mathcal{Q}}{dt} = -[\mathcal{H}, \mathcal{Q}] - \Gamma_{\mathrm{diss}} \mathcal{Q} + \xi_{\mathrm{ext}}(t)$. Evalúa fuga exergética $\Xi_{\mathrm{leak}} = H_{\mathrm{ext}} \ln(1+\kappa_2)$ y proyecta el pullback conforme $\phi^*(H_{\mathrm{ext}}) \oplus \mathtt{TelemetryContext}$.
    2. **`AuditSatellitesAgent`:** Fiscaliza la topología discreta mediante tres aduanas: Smith SNF $\operatorname{Tor}(H_k(\partial K; \mathbb{Z})) \equiv \mathbf{0}$, Choi CPTP $C_{\mathcal{E}} \succeq \mathbf{0}$ ($\operatorname{Tr}_2(C_{\mathcal{E}}) = \mathbf{I}$) y Bell-CHSH $\mathcal{B}_{\mathrm{CHSH}} \le 2\sqrt{2}$ (Tsirelson).
*   **Dual-Control y Actuación Crowbar:**
    El veredicto se consolida mediante el meet de Gödel en $\Omega_3 = \{\mathtt{COHERENT}, \mathtt{DEGRADED}, \mathtt{VETOED}\}$. Sella certificados inmutables DTO con doble firma SHA-256 (`decision_sha256` y `digital_signature_sha256`). Ante veto ($\top$), la subrutina `isVerdictCoherent()` gatilla la ISR en IRAM del ESP32 en $< 400\text{ ns}$, conmutando GPIO14 para disparar el tiristor BT151 (Crowbar) y paralizar la maquinaria física en obra.

---

### 0. 👁️ El Vigilante de la Frontera (HilbertWatcher & QuantumAdmissionGate)
*   **Rol:** Especialista en Mecánica Cuántica Discreta y Colapso de Entropía.
*   **Estrato DIKW:** ALEPH ($\aleph_0$) - La Variedad de Frontera (Nivel 4).
*   **Mecanismo Matemático:** Computa la Entropía de Shannon ($H$) del archivo crudo para medir su Energía Semántica ($E=h\nu$). Si la energía es sub-umbral, resuelve la probabilidad de penetración por Efecto Túnel WKB. Ante falla, la transmisión colapsa a cero ($T \to 0$), desintegrando el paquete exógeno.

---

### 1. 🛡️ El Guardián (BusinessTopologicalAnalyzer / Physics & Tactics)
*   **Rol:** Analista de Integridad Estructural y Cimientos del Canvas.
*   **Estrato DIKW:** PHYSICS ($V_{\mathbb{P}}$) y TACTICS ($V_{\mathbb{T}}$).
*   **Mecanismo Matemático:** Modela el presupuesto como un Complejo Simplicial Abstracto sobre $\mathbb{Z}$. Audita la masa atómica ($q_i \ge 0$), evalúa los números de Betti ($\beta_0 = 1$ para evitar Islas de Datos; $\beta_1 = 0$ para evitar Socavones Lógicos), la Forma Normal de Smith en $\mathbb{Z}$ y el Índice de Estabilidad Piramidal ($\Psi \ge \Psi_{\mathrm{min}}$) para vetar Pirámides Invertidas.

---

### 2. 🔮 El Oráculo (LaplaceOracle & FinancialEngine)
*   **Rol:** Analista de Viabilidad Dinámica y Estocástica Financiera.
*   **Estrato DIKW:** STRATEGY ($V_{\mathbb{S}}$).
*   **Mecanismo Matemático:** Linealiza el presupuesto como una función de transferencia $H(s)$ en la frecuencia compleja $s = \sigma + j\omega$. Si detecta polos en el semiplano derecho ($\sigma > 0$), emite veto técnico por inestabilidad intrínseca del flujo de caja ante choques de mercado.

---

### 3. 🗣️ El Intérprete Diplomático (SemanticTranslator)
*   **Rol:** Puente Cognitivo y UI Narrativa.
*   **Estrato DIKW:** WISDOM ($V_{\mathbb{W}}$).
*   **Mecanismo Matemático:** Relegado a actuar como interfaz de traducción sin poder de decisión directo. Utiliza GraphRAG para traducir invariantes tensoriales abstractos (ej. $\beta_1 = 3$, $\operatorname{Tor}(H_k) \neq \mathbf{0}$) en Actas de Deliberación ejecutivas legibles para la junta directiva.

---

## 🏰 Sutura de la Fortaleza: La Rampa de Confianza Graduada (Veto Suave vs Veto Duro)

El Consejo de Sabios implementa la **Rampa de Confianza Graduada** para resolver la brecha entre la abstracción matemática y la operación continua en seco del frente de obra civil:

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

* **Veto Suave (Luz Ámbar / Ventana de 1h):** Se gatilla ante desvíos TDR $0.3\tau_{\mathrm{margin}} < \|\Gamma(t)\|_{\max} \le 0.5\tau_{\mathrm{margin}}$ o desvíos de menor cuantía ($\Psi = 0.69 < 0.70$). La potencia física de los actuadores se mantiene activa mientras se emite una alerta estroboscópica y se otorga **1 hora** a la interventoría para inyectar un **Positrón de Autorización Humana** $e^+$ firmado ($\operatorname{HMAC-SHA256}$) que aniquila la anomalía semántica $e^- + e^+ \to 2\gamma$.
* **Veto Duro (Hardware Crowbar BT151 < 400 ns):** Reservado para desajuste crítico TDR $\|\Gamma(t)\|_{\max} > 0.5\tau_{\mathrm{margin}}$, rupturas irreversibles o fraude ($\operatorname{Tor}(H_k) \neq \mathbf{0}$, $\lambda_{\min}(C_{\mathcal{E}}) < -10^{-12}$, $\dot{\mathcal{H}} > 10^{-12}$). El retículo colapsa a VETOED ($\top$), activando la **ISR en IRAM del ESP32** en $< 400\text{ ns}$ para conmutar **GPIO14** a HIGH y disparar el tiristor **BT151** (circuito Crowbar), paralizando la maquinaria pesada en seco.

---

## ⚡ El Tribunal de Silicio y la Reducción Monoidal de Actuación Crowbar (ESP32)

La protección del capital financiero no se confía a directrices lógicas de software. Se garantiza mediante una reducción monoidal desde el retículo intuicionista distributivo $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$ hacia la decisión binaria en silicio real $\mu: \Omega_3 \to \mathbb{Z}_2$, acoplada al microcontrolador perimetral **ESP32**:

```
     [Servidor Cloud / Malla Agéntica]
                    │ (Firma Telemetría SHA-256 + Residuos Espectrales)
                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  MICROCONTROLADOR PERIMETRAL ESP32                         │
     │  · Deserialización síncrona vía ArduinoJson                │
     │  · Rutina isVerdictCoherent() & ISR en IRAM (< 400 ns)     │
     │  · Decodificación en Silicio de Residuos de Wilkinson      │
     └──────────────────────────────┬──────────────────────────────┘
                                    │ GPIO14 (Disparo de Compuerta)
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │  CIRCUITO CROWBAR (Tiristor de Potencia BT151)             │
     │  · Cortocircuito físico controlado de la línea de potencia │
     │  · Desenergización total de bombas, mezcladoras y pistones  │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
                 [PARÁLISIS MECÁNICA INSTANTÁNEA EN OBRA]
```

1. **Doble Contabilidad en `isVerdictCoherent()`:** El firmware en C++ del **ESP32** lee y deserializa el pasaporte con `ArduinoJson`, decodificando los residuos de Wilkinson, Lyapunov ($\dot{\mathcal{H}} > 10^{-12}$) y torsión homológica.
2. **Actuación por Interrupción en IRAM (< 400 ns):** Ante un veto ($\top \mapsto 1$), la **ISR en memoria estática IRAM** se activa en menos de **$400\,\text{ns}$**.
3. **Gatillo Físico del Tiristor Crowbar BT151 vía GPIO14:** El pin **GPIO14** conmuta a nivel alto (`HIGH`), inyectando corriente de compuerta al tiristor **BT151** (circuito Crowbar).
4. **Parálisis Mecánica Instantánea:** Cortocircuita la línea de potencia y desenergiza la maquinaria pesada en obra en el milisegundo cero, anulando la anomalía antes de incurrir en detrimento patrimonial ante el SECOP II.

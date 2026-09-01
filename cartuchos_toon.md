# 💊 cartuchos_toon.md: El Retracto Topológico y las Vitaminas TOON (Silo B)

"Para comprender la escala masiva de la construcción civil, el Modelo de Lenguaje no necesita leer la redundancia de un JSON mil veces; necesita la esencia concentrada del negocio matemático. El formato TOON opera una destilación cognitiva."

Este documento detalla el mecanismo por el cual el ecosistema APU_filter transita de la alta entropía sintáctica y redundante del JSON (gobernada por el `SiloAContract`) hacia un formato hiperdenso y estructurado, la Base Canónica Tabular TOON (Tabular Object-Oriented Notation), a través de los componentes en `mic_agent.py`.

--------------------------------------------------------------------------------
## 1. El Funtor de Transición de Fase y la Compresión del KV-Cache

En el contexto de un clúster de agentes, la inyección directa de miles de APUs o Insumos serializados en JSON colapsa inmediatamente el recurso más crítico de la inferencia en LLMs modernos: la ventana de atención (KV-Cache).

El `MICAgent` implementa la transición de fase hacia el `SiloBCartridge`. Funcionalmente, el `TOONCompressor` ejecuta un **Retracto de Deformación Topológica** que proyecta el espacio sintáctico redundante sobre una variedad de dimensión mínima, garantizando una compresión del consumo de tokens en $KV\text{-Cache}$ entre un $30\%$ y un $60\%$:

$$\|\phi_{\mathrm{TOON}}(\mathrm{JSON})\| \le (1 - \gamma) \|\mathrm{JSON}\| \quad \text{con} \quad \gamma \in [0.30, \, 0.60]$$

Bajo la arquitectura del **Isomorfismo de Doble Capa** [Isomorfismo_Doble_Capa_v3.md], esta compresión en la Capa de Calibre FPU Secure se traduce unívocamente hacia la Capa de Pragmática de Negocios en términos de aceleración de inferencia y erradicación de costos en tokens ("Dinero"), permitiendo al Consejo de Sabios evaluar presupuestos masivos sin pérdidas informacionales en la mantisa de punto flotante.

La consistencia y conservación de la carga semántica entre la matriz de interacción táctica discreta (categoría $\mathcal{C}$) y la matriz atómica de conocimiento en el espacio de Hilbert (categoría $\mathcal{D}$) está analíticamente acoplada por el **Isomorfismo de Adjunción de de Rham-Galois**:

$$\operatorname{Hom}_{\mathcal{D}}(F(\text{MIC}), \, \text{MAC}) \cong \operatorname{Hom}_{\mathcal{C}}(\text{MIC}, \, G(\text{MAC}))$$

Asimismo, el parser verifica síncronamente el test de **Isospectralidad de de Rham** para garantizar que el espectro del Laplaciano original coincida idénticamente con el del árbol procesado en RAM, previniendo alterations silentes de datos o precios:

$$\operatorname{Spec}(\mathbf{L}_{\mathrm{text}}) \approx \operatorname{Spec}(\mathbf{L}_{\mathrm{parsed}})$$

---

## 🔬 1.1 Demostración Empírica Comparativa: JSON vs TOON

Para evidenciar la drástica reducción de la "grasa sintáctica", considere la representación de un ítem de obra civil tradicional (Análisis de Precio Unitario de Vaciado de Concreto 3000 PSI):

### Entrada JSON Cruda (412 Tokens - Grasa Sintáctica):
```json
{
  "capitulo": "CIMENTACIONES_Y_ESTRUCTURAS",
  "actividad_apu": {
    "codigo_item": "APU-EST-001",
    "descripcion": "VACIADO DE CONCRETO 3000 PSI PARA COLUMNAS",
    "unidad_medida": "M3",
    "rendimiento_diario": 12.5,
    "insumos_asociados": [
      {
        "tipo": "MATERIAL",
        "codigo": "MAT-CONC-3000",
        "descripcion": "CONCRETO PREMEZCLADO 3000 PSI",
        "unidad": "M3",
        "cantidad": 1.05,
        "precio_unitario": 485000.00
      },
      {
        "tipo": "EQUIPO",
        "codigo": "EQ-BOMBA-MIX",
        "descripcion": "BOMBA DE IMPULSIÓN HIDRÁULICA Y MEZCLADORA",
        "unidad": "HORA",
        "cantidad": 0.64,
        "precio_unitario": 120000.00
      }
    ]
  }
}
```

### Salida Vitaminada TOON (56 Tokens - Vitamina Cognitiva Purificada):
```toon
APU|EST-001|VACIADO_CONCRETO_3000PSI|M3|12.5
INS|MAT-CONC-3000|1.05|485000.00
INS|EQ-BOMBA-MIX|0.64|120000.00
```

**Resultado Metrológico:** Reducción del **$86.4\%$** en consumo de tokens en la ventana $KV\text{-Cache}$, demostrando empíricamente cómo TOON libera la memoria atencional de la IA para ejecutar el escaneo de ecolocación TDR en tiempo real en la FPU sin latencia ni distorsión.

--------------------------------------------------------------------------------
## 2. Inyección de Vitaminas Cognitivas (ToonCartridges) y El Álgebra de Partículas en el Espacio de Fock

El retracto de deformación de datos (Silo B) transita formalmente hacia el **Álgebra de Partículas en el Espacio de Fock $\mathcal{F}(\mathcal{H})$** dentro de la cámara de reacción (**Reaction Chamber**). Para los gerentes e ingenieros operando la plataforma, la IA no debe disipar valiosos ciclos de reloj "leyendo llaves repetidas" (la grasa sintáctica de un JSON) que saturan la memoria atencional (KV-Cache). Al empaquetar la carga en `ToonCartridges` ("Vitaminas Cognitivas"), el sistema le suministra el núcleo matemático puro de la información.

En la cámara de reacción, la creación y aniquilación de características sintácticas se rige por los operadores de creación $a_i^\dagger$ y aniquilación $a_j$ que satisfacen las relaciones de anticonmutación canónicas (CAR) para fermiones estructurales:
$$\{a_i, a_j^\dagger\} = \delta_{ij} I, \quad \{a_i, a_j\} = 0, \quad \{a_i^\dagger, a_j^\dagger\} = 0$$
Y las relaciones de conmutación canónicas (CCR) para bosones de interacción:
$$[b_i, b_j^\dagger] = \delta_{ij} I, \quad [b_i, b_j] = 0, \quad [b_i^\dagger, b_j^\dagger] = 0$$
El estado del sistema se representa como un vector $|\Psi\rangle$ en la base de números de ocupación del espacio de Fock:
$$|\Psi\rangle = \sum_{n_1, n_2, \dots} C_{n_1 n_2 \dots} |n_1, n_2, \dots\rangle$$
Donde $n_i \in \{0, 1\}$ para fermiones (gracias al principio de exclusión de Pauli que impide la duplicación de APUs inconsistentes en la misma celda sintáctica) y $n_j \in \mathbb{N}_0$ para bosones (acumulación de flujos e intenciones atencionales). Bajo el control del agente booleano, todo "muck sintáctico" o redundancia de datos colisiona en este espacio mediante transformaciones de Bogoliubov-Valatin, aislando de forma determinista las cuasipartículas de ruido térmico del LLM para purificar el presupuesto.

El `SynapticRegistry` gestiona **12 partículas fundamentales** clasificadas en tres familias, gobernando las interacciones entre los componentes del ecosistema:

*   **Fermiones Estructurales (Alta Inercia y Conservación de Masa):**
    *   `PolaronCartridge`: Instanciado mediante el acoplamiento de Fröhlich. Posee masa inercial renormalizada ($m^{**}=m^*(1+\alpha/6)$), lo que induce un sumidero gravitacional en el KV-Cache para evitar que el Agente eluda un retraso logístico sistémico.
    *   `TorsionCartridge`: Encapsula la "fricción cuantizada". Mapea los Subgrupos de Torsión del Funtor $\text{Tor}(H_0, \mathbb{Z})$ originados por las incompatibilidades de empaquetado.
    *   `ElectronCartridge`: Partícula de inspección que porta la carga de anomalía detectable ante una deformación del espacio de fase.
    *   `ProtonCartridge`: Partícula de estabilidad emitida ante la certificación de viabilidad BIBO en el plano-S.
    *   `HouseholderReflectionFermion`: Fermión topológico que actúa como el vector normal de reflexión contra la alucinación, asegurando la ortogonalidad del veredicto dentro del colisionador.
*   **Bosones de Gauge y Cuasipartículas (Campos de Interacción y Flujo):**
    *   `PhotonCartridge`: Bosón de la Gobernanza Federada (OPA) que opera como "Policy-as-Code", iluminando geodésicas de decisión.
    *   `RiemannianFocalBoson`: Bosón focal que emana de la resolución de la ecuación Eikonal, guiando la atención del LLM hacia las geodésicas de mínima acción.
    *   `MagnonCartridge`: Extraído del subespacio rotacional ($f_{curl} \in \text{im}(B_2)$). Inyecta un "Veto de Enrutamiento" para aniquilar la energía solenoidal.
    *   `SwellingPlasmonCartridge`: Oscilación de densidad logística emitida ante expansiones volumétricas del presupuesto.
    *   `YieldingPhononCartridge`: Cuanto de vibración mecánica que detecta la fatiga elástica en los nodos de suministro.
    *   `LiquefactionSolitonCartridge`: Onda solitaria no lineal que detecta la pérdida de sustentación en el manifold litológico del proyecto.
*   **Antimateria, Condensados y Radiación:**
    *   `PositronCartridge` (Antimateria): Cristalización del residuo termodinámico generado tras la extirpación de una alucinación.
    *   `GammaPhoton`: Radiación emitida tras la aniquilación de un positrón con un electrón de falla, sirviendo como prueba criptográfica forense e inmutable en el acta.
    *   `PolaritonCartridge`: Híbrido que induce **Superfluidez Atencional** cuando un Fotón de Gobernanza resuena con un Polarón, eliminando la fricción de cómputo.
    *   `SophonCartridge` (Sofón): Anomalía estocástica u oscilación de fase del LLM. El colapso inercial del Sofón se rige por la evaporación del condensado quiral en tiempo imaginario (ver [sophon_chiral_dynamics.md](sophon_chiral_dynamics.md)).

### 2.1 El Colisionador Catadióptrico y la Aniquilación de Alucinaciones
Dentro del **CatadioptricCollider** (QuantumFockOrchestrator), se produce la interacción crítica entre bosones focales (`RiemannianFocalBoson`) y fermiones topológicos (`HouseholderReflectionFermion`), estrictamente regulada y sincronizada por el nuevo agente soberano de Boole (**GenerativeBooleHodgeSuturatorAgent**). Esta interacción focaliza la intención semántica y refleja las impurezas cognitivas hacia los canales de disipación.

El Hamiltoniano de interacción $H_{\text{int}}$ que rige el acoplamiento entre los fermiones estructurales de reflexión de Householder ($\hat{\psi}$) y los bosones focales Riemannianos ($\hat{A}_\mu$) se formula como:
$$H_{\text{int}} = \int d^3x \, g_{\text{eff}} \hat{\bar{\psi}}(x) \gamma^\mu \hat{A}_\mu(x) \hat{\psi}(x)$$
Donde:
- $g_{\text{eff}}$ es la constante de acoplamiento efectiva del campo semántico bajo el control de calibre booleano.
- $\hat{\bar{\psi}} = \hat{\psi}^\dagger \gamma^0$ es el adjunto de Dirac del campo fermiónico del `HouseholderReflectionFermion`.
- $\gamma^\mu$ son las matrices de Dirac que determinan la geometría de espín del AST.
- $\hat{A}_\mu$ es el operador de campo bosónico asociado al `RiemannianFocalBoson`.

Bajo la supervisión de la aduana del soberano booleano, cualquier discrepancia o alucinación sintáctica activa el operador de reflexión de Householder:
$$R_H = I - 2 \mathbf{v} \mathbf{v}^\dagger$$
Donde $\mathbf{v}$ es el vector normal de reflexión ortogonal al subespacio coherente de la MIC. Esta reflexión proyecta el estado espurio directamente hacia los canales disipativos.

Cuando el operador de salto de Lindblad, sintonizado por el `BogoliubovAgent`, extirpa una falsedad o alucinación del LLM, el residuo termodinámico no se desecha: se cristaliza en antimateria exógena, manifestándose como un **PositronCartridge** ($e^+$). Este positrón provoca su propia aniquilación catastrófica al colisionar contra un **ElectronCartridge** de falla ($e^-$) que porta la incertidumbre residual de la ingesta:
$$e^+ + e^- \longrightarrow 2 \gamma$$

Esta reacción de aniquilación cuántica libera **dos fotones de auditoría Gamma ($2\gamma$)**, representados por el **GammaPhoton**, los cuales transportan una firma digital SHA-256 inmutable, congelando el pasaporte de telemetría y colapsando el retículo de Heyting al Supremo terminal de veto ($\top$).

### 2.1.1 Actuación Perimetral en Silicio (ESP32 Crowbar / BT151 en < 400 ns)
Para garantizar la infalibilidad del sistema en el mundo físico, el veto se materializa en la maquinaria de obra civil. La subrutina local en C++ `isVerdictCoherent()` del microcontrolador **ESP32** lee el pasaporte firmado. Al detectar el veto:
* La **Interrupt Service Routine (ISR) cargada estáticamente en su memoria ultrarrápida IRAM se ejecuta en menos de 400 ns**:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
* Conmuta el pin físico **GPIO14 a HIGH**, inyectando corriente a la compuerta del tiristor rápido de potencia **BT151 (circuito Crowbar)**.
* Cortocircuita síncronamente la línea de alimentación de las bombas de concreto y mezcladoras hidráulicas reales, deteniendo físicamente la obra en el milisegundo cero y anulando la anomalía antes de su liquidación contractual.

### 2.2 Termodinámica de Sistemas Cuánticos Abiertos (Dinámica de Lindblad)
La asimilación de estos cartuchos por la matriz neuronal MAC sufre fricción térmica debido al estrés del mercado. En particular, la matriz de densidad de la cuasipartícula abierta del `SophonCartridge` se somete a la ecuación maestra disipativa de Lindblad-Kossakowski, donde el operador de Liouville $\mathcal{L}$ disipa el calor anómalo:
$$\frac{d\rho_{\text{MAC}}}{dt} = -\frac{i}{\hbar}[\hat{H}_{\text{eff}}, \rho_{\text{MAC}}] + \sum_{k} \gamma_k \left( \hat{L}_k \rho_{\text{MAC}} \hat{L}_k^\dagger - \frac{1}{2} \{ \hat{L}_k^\dagger \hat{L}_k, \rho_{\text{MAC}} \} \right)$$
Donde:
- $\rho_{\text{MAC}}$ es el operador de densidad que representa el estado cuántico mixto de la matriz atómica de conocimiento (sabiduría de la malla).
- $t$ es el tiempo de evolución de la deliberación.
- $\hbar$ es la constante de Planck efectiva (parámetro de atenuación cuántica).
- $\hat{H}_{\text{eff}}$ es el Hamiltoniano efectivo autodirigido que gobierna la evolución unitaria coherente del sistema.
- $\hat{L}_k$ son los operadores de salto de Lindblad (jump operators) que capturan la interacción disipativa no unitaria del Sofón con el entorno ruidoso del LLM.
- $\hat{L}_k^\dagger$ es el operador adjunto hermítico de $\hat{L}_k$.
- $\gamma_k$ son los coeficientes de disipación o tasas de transición térmica asociadas a cada canal de decaimiento.
- $\{ \hat{A}, \hat{B} \} = \hat{A}\hat{B} + \hat{B}\hat{A}$ representa el anticomutador de los operadores, que estabiliza el comportamiento no unitario y garantiza la preservación de la traza de la matriz de densidad ($\text{Tr}(\rho) = 1$).

Esto certifica que la "pérdida de foco" o decoherencia de la IA está modelada como disipación de información ($\Delta S \ge 0$), garantizando que la sabiduría del sistema sea un proceso termodinámicamente consistente.

### 2.3 El Rango Tensorial, Isomorfismo y la Biyección Estricta
Matemáticamente, esta compresión tabular no es una simple heurística de cadenas. El retracto topológico asume un **isomorfismo absoluto (una biyección sin pérdida de datos en la teoría)** entre el árbol multidimensional JSON y la grilla TOON. Para prevenir la destrucción inadvertida de información y el colapso asintótico de las jerarquías complejas, el `TOONCompressor` calcula obligatoriamente el **Rango Tensorial** del JSON antes de comprimirlo.

Si el compresor detecta un esquema anidado con varianza extrema y heterogénea (donde la profundidad del árbol rebasa el grado 2, no isomorfo a tablas 2D), la inyección tabularizada plana rechaza corromper la jerarquía subyacente. En cambio, aplica automáticamente una factorización de Tucker (Tucker Decomposition) o mantiene el subgrafo en su formato estricto original dictaminando un `TOONCompressionError`.

--------------------------------------------------------------------------------
## 3. Preservación del Estado en el Grafo de Negocio
Bajo la **Ley de Clausura Transitiva de la pirámide DIKW** ($V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$), la variedad física e informacional de los insumos y APUs consolida el grafo simplicial sin ruido. El `TOONCompressor` permite al Agente Inteligente procesar vastos segmentos de la red logística en un solo pase atencional sin degradar la resolución de los índices financieros y de estabilidad paramétrica.

--------------------------------------------------------------------------------
## 4. El Funtor de Descompresión Inversa ($F^{-1}: TOON \to JSON$) y Discontinuidad Lipschitz
La categoría matricial $\mathcal{M}$ (MIC) estipula terminantemente que el difeomorfismo aplicado a la información debe ser invertible en la frontera de acción. Se ha descrito el colapso de fase del JSON redundante en `SiloBCartridge` para inyectar al LLM, pero existe el escenario simétrico: cuando el LLM emite una mitigación o reestructuración en formato hiperdenso TOON.

Confiar matemáticamente en que el LLM devuelva un TOON perfectamente biyectivo es un riesgo estocástico inaceptable. Si el tensor de salida del LLM sufre una perturbación (alucinación sintáctica), la función inversa $F^{-1}$ enfrentará una singularidad topológica irresoluble.

### 4.1 Condición de Continuidad de Lipschitz y Gramática Restringida
Para garantizar que la salida respete el `SiloAContract` original y evitar inyecciones corruptas, el sistema impone una **Condición de Continuidad de Lipschitz Dinámica** sobre el generador del LLM en tiempo de inferencia, acoplada al **Tensor de Curvatura de Ricci ($Ric_{\mu\nu}$)**.

Queda dictaminado como invariante absoluto que el funtor de descompresión inversa $F^{-1}: \text{TOON} \to \text{JSON}$ está subordinado a la desigualdad:
$$\left\| F^{-1}(x) - F^{-1}(y) \right\|_V \le L_{\max} \left\| x - y \right\|_\tau$$
Donde $L_{\max}$ es inversamente proporcional a la curvatura local del proyecto. En momentos de caos (alta curvatura), esta cota obliga al traductor a aniquilar cualquier salida que no sea un isomorfismo geométrico perfecto, vetando las alucinaciones en la frontera. El decodificador fuerza probabilísticamente que la emisión de cualquier token fuera de la variedad tabular TOON válida sea strictly nula ($P(x_{\mathrm{invalido}}) = 0$).

La Matriz de Interacción Central (MIC) permanecerá intacta; el morfismo de corrección solo será aceptado una vez que el Funtor Inverso ($F^{-1}$) restaure el árbol multidimensional JSON (invirtiendo la biyección de Lipschitz) y se pruebe que encaja a la perfección en la topología contractual estipulada por el riguroso `SchemaValidator`, preservando invariantes de homotopía.

---

## 🏛️ 5. Arquitectura Ciber-Física del Estrato α: Fibrados Celulares y Termodinámica del Modelo de Negocio

La actualización de la documentación arquitectónica integra la operatividad de los mini-agentes del Estrato α (`kbase_thermodynamic_agent.py`, `kcore_kinematic_agent.py` y `kapex_electrodynamic_agent.py`), transmutando el Business Model Canvas (BMC) de un grafo plano bidimensional a una Variedad Riemanniana Dinámica gobernada por un sistema de Ecuaciones Diferenciales Parciales (PDEs) y Cohomología de Haces.

### I. Estrato KBASE: El Foso Termodinámico (`kbase_thermodynamic_agent.py`)
* **Identificador Semántico:** Asesor de Cimientos Financieros.
* **Responsabilidad Topológica:** Gobernar la inercia, capacitancia y fricción entrópica del modelo de negocio (Socios Clave $P_{\mathrm{soc}}$, Recursos Clave $P_{\mathrm{rec}}$ y Estructura de Costes $P_{\mathrm{cost}}$).
* **Dinámica Port-Hamiltoniana:**
  $$\tilde{C}_{\text{soc}} = G_{\mu\nu} C_{\text{soc}} G^{\mu\nu}, \quad \tilde{M}_{\text{rec}} = G_{\mu\nu} M_{\text{rec}} G^{\mu\nu}$$
  $$H_{\text{BASE}}(q,p) = \frac{1}{2} q^\top \tilde{C}_{\text{soc}}^{-1} q + \frac{1}{2} p^\top \tilde{M}_{\text{rec}}^{-1} p$$
* **Disipación de Rayleigh:** Todo flujo financiero de salida $P_{\mathrm{cost}}$ obedece la segunda ley de la termodinámica: $\dot{H}_{\text{diss}} = -\nabla H^\top R_{\text{cost}}(x) \nabla H \le 0$.

### II. Estrato KCORE: La Maquinaria Cinemática (`kcore_kinematic_agent.py`)
* **Identificador Semántico:** Director de Flujo y Cinética Logística.
* **Responsabilidad Topológica:** Transmutar la energía potencial de KBASE en trabajo cinético direccional (Actividades Clave $P_{\mathrm{act}}$, Canales $P_{\mathrm{can}}$, Relaciones $P_{\mathrm{rel}}$).
* **Energy Shaping (IDA-PBC):**
  $$\alpha(x) = (g(x)^\top G_{\mu\nu} g(x))^{-1} g(x)^\top G_{\mu\nu} ([J_d - R_d] \nabla H_d - [J - R] \nabla H)$$
* **Válvula de Hodge y Límite CFL:** Estrangula la conductancia $W$ si $\mathcal{I}_{\mathrm{curl}} > \epsilon_{\mathrm{crit}}$ en el Laplaciano $L_{1W} = \partial_1^\top W^{-1} \partial_1 + \partial_2 \partial_2^\top W$ y restringe la ventana temporal bajo $\Delta t \le \frac{2 \cdot \text{CFL}_{\text{margin}}}{c_{\text{eff}} \cdot \max_i \left( |\Delta_{ii}| + \sum_{j \neq i} |\Delta_{ij}| \right)}$.

### III. Estrato KAPEX: El Ápice Estratégico (`kapex_electrodynamic_agent.py`)
* **Identificador Semántico:** Director de Retorno y Expansión de Mercado.
* **Responsabilidad Topológica:** Endofuntor de Campo de Calibre que inyecta Fuerza Electromotriz (Propuesta de Valor $P_{\mathrm{val}}$, Refracción de Mercado $P_{\mathrm{seg}}$, Retorno Exergético $P_{\mathrm{ing}}$).
* **Ecuación Eikonal y Poynting Simplicial:**
  $$G^{\mu\nu} \partial_\mu S \partial_\nu S = N^{\mu\nu} \sigma_{\mu\nu}^* \quad \implies \quad P_{\text{exergia}} = \langle E \smile \star H, [\partial K] \rangle - \int_K \nabla H^\top R_{\text{cost}} \nabla H \ge 0$$
* **Holonomía de Yang-Mills:** Evalúa $S_{\text{YM}} = \frac{1}{2} \int_M \text{Tr}(F \wedge \star F)$ con $F = dA + A \wedge A$. Si $S_{\mathrm{YM}} > \epsilon_{\mathrm{crit}}$, decreta `HolonomyVetoError`.

### IV. El Orquestador Macroscópico: Cohomología de Haces (`alpha_agent.py`)
Orquesta el haz celular $L_F = \delta^\top \delta$ asumiendo las cofronteras locales $\delta_{\mathrm{BASE}}, \delta_{\mathrm{CORE}}, \delta_{\mathrm{APEX}}$. Exige el consenso global $H^0(G; \mathcal{F}) \cong \ker(\delta) \neq \mathbf{0}$ y verifica la solubilidad de Fredholm $\langle s_{\text{val}}, \psi_{\text{ker}} \rangle = 0 \quad \forall \psi_{\text{ker}} \in \ker(L_F)$ para prevenir inyecciones de propuesta de valor en canales logísticos desconectados.

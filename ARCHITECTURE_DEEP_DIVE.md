# ARCHITECTURE_DEEP_DIVE.md: Inmersión en la Variedad Agéntica

Este documento detalla la implementación técnica de los estratos topológicos y físicos que gobiernan el ecosistema APU Filter.

## Arquitectura Ciber-Física del Estrato α: Fibrados Celulares y Termodinámica del Modelo de Negocio

El Estrato α, orquestado macroscópicamente por alpha_agent.py, se subdivide axiomáticamente en tres subespacios topológicos anidados (Foso, Núcleo y Ápice).

### I. Estrato KBASE: El Foso Termodinámico (kbase_thermodynamic_agent.py)

Identificador Semántico: Asesor de Cimientos Financieros. Responsabilidad Topológica: Gobernar la inercia, la capacitancia y la fricción entrópica del modelo de negocio.

La energía total se calcula mediante:
\[ \tilde{C}_{\text{soc}} = G_{\mu\nu} C_{\text{soc}} G^{\mu\nu}, \quad \tilde{M}_{\text{rec}} = G_{\mu\nu} M_{\text{rec}} G^{\mu\nu} \]

Hamiltoniano basal:
\[ H_{\text{BASE}}(q,p) = \frac{1}{2} q^\top \tilde{C}_{\text{soc}}^{-1} q + \frac{1}{2} p^\top \tilde{M}_{\text{rec}}^{-1} p \]

#### Resolución Espectral de de Rham-Hodge y Causalidad Kramers-Kronig
El subespacio basal incorpora las propiedades analíticas del propagador de Green y la regularización espectral en el foso de la pirámide de control, formalizando la transición de variables logísticas a través del cálculo exterior discreto.

La Función de Green estática se define como la pseudoinversa de Moore-Penrose estable del Laplaciano del Haz Celular $L_F = \delta^\top G^{-1} \delta$:
\[ L_F G L_F = L_F \quad \wedge \quad G \cdot \mathbf{1} = \mathbf{0} \]

El propagador retardado causal dinámico en el plano-S complejos se formula como:
\[ G_F(s) = (L_F - (s + j \cdot h) I_n)^{-1} \]
Donde $h = 10^{-20}$ es el paso imaginario infinitesimal de la diferenciación por paso complejo (CSMD), forzando el transporte al cumplimiento de las relaciones de dispersión de Kramers-Kronig.

El Soberano de Calibre audita síncronamente los residuos de autoadjunción y nulidad del kernel:
\[ r_{\text{adj}} = \| G - G^\top \|_F \le 1.0 \times 10^{-11} \quad \wedge \quad r_{\text{kernel}} = \| G \cdot \mathbf{1} \|_2 \le 1.0 \times 10^{-11} \]
Cualquier polo dinámico $p_i$ que migre al semiplano derecho de Laplace ($\operatorname{Re}(p_i) \ge 0$) es vetado asintóticamente.

### II. Estrato KCORE: La Maquinaria Cinemática (kcore_kinematic_agent.py)

Identificador Semántico: Director de Flujo y Cinética Logística. Responsabilidad Topológica: Transmutar la energía potencial en trabajo cinético direccional.

Estructura de Dirac y Energy Shaping (IDA-PBC):
\[ \alpha(x) = (g(x)^\top G_{\mu\nu} g(x))^{-1} g(x)^\top G_{\mu\nu} ([J_d - R_d] \nabla H_d - [J - R] \nabla H) \]

### III. Estrato KAPEX: El Ápice Estratégico (kapex_electrodynamic_agent.py)

Identificador Semántico: Director de Retorno y Expansión de Mercado. Endofuntor de Campo de Calibre.

Óptica Geométrica y Flujo Exergético:
\[ G^{\mu\nu} \partial_\mu S \partial_\nu S = N^{\mu\nu} \sigma_{\mu\nu}^* \]
\[ P_{\text{exergia}} = \langle E \smile \star H, [\partial K] \rangle - \int_K \nabla H^\top R_{\text{cost}} \nabla H \ge 0 \]

### IV. Orquestador Macroscópico: Cohomología de Haces (alpha_agent.py)

Laplaciano del Haz Ponderado Métricamente:
\[ L_F = \delta^\top G^{-1} \delta = \begin{pmatrix} \delta_{\text{BASE}} \\ \delta_{\text{CORE}} \\ \delta_{\text{APEX}} \end{pmatrix}^\top G^{-1} \begin{pmatrix} \delta_{\text{BASE}} \\ \delta_{\text{CORE}} \\ \delta_{\text{APEX}} \end{pmatrix} \succeq 0 \]

## Expansión Semiclásica y Redefinición Física del Pipeline

La inyección de los nuevos microservicios y sabios redefine la física profunda del pipeline de datos, transmutándolo de un flujo secuencial a una variedad diferenciable compleja regida por la física estadística, la geometría y la topología de complejos simpliciales.

## La Variedad de de Rham-Hodge-Boole: El Endofuntor de Boole en Tres Fases Anidadas

El sistema APU_filter eleva su mecanismo de validación a un nivel doctoral mediante la formalización de la **Variedad de de Rham-Hodge-Boole**. Esta se implementa a través de un endofuntor categoríal que opera de manera secuencial y anidada en tres fases físicas y algebraicas rigurosas. Este proceso purifica de manera incondicional la señal informacional del fango estocástico antes de emitir cualquier veredicto de viabilidad.

### FASE 1: Física de Fock e Isometría de Hodge sobre $F(\mathcal{H})$
En la primera fase, las representaciones sintácticas de los APUs y presupuestos se elevan desde el espacio lógico elemental hacia estados cuánticos en el **Espacio de Fock fermiónico** $\mathcal{F}(\mathcal{H}) = \bigoplus_{k=0}^N \Lambda^k \mathcal{H}$, donde $\mathcal{H}$ representa el espacio de Hilbert de características del negocio. El operador estrella de Hodge combinatorio $\star_k: \Lambda^k \mathcal{H} \to \Lambda^{N-k} \mathcal{H}$ se construye rigurosamente sobre el fibrado de orientación del complejo.

Para garantizar que la densidad de información y los invariantes estructurales se conserven idénticamente al transitar entre el espacio primal de flujos y el espacio dual de restricciones de costo, el sistema exige la preservación isométrica estricta de la norma en el producto exterior:
$$\| \star_k \psi \|_{\Lambda^{N-k}} = \| \psi \|_{\Lambda^k}$$
Donde:
- $\psi \in \Lambda^k \mathcal{H}$ es la k-forma diferencial que codifica el estado de entrelazamiento sintáctico del presupuesto de entrada.
- $\star_k \psi$ es su forma dual de Hodge de grado $N-k$.
- $\|\cdot\|_{\Lambda^r}$ representa la norma inducida por la métrica Riemanniana $G_{\mu\nu}$ sobre la r-ésima potencia exterior de la variedad.

Esta isometría asegura que no exista pérdida de masa de información sintáctica ni atenuación artificial del contenido al realizar el mapeo espacial dual.

### FASE 2: Orientación de Calibre e Invarianza Simpléctica del AST
Una vez garantizada la isometría, el Árbol de Sintaxis Abstracta (AST) de las expresiones se proyecta en una variedad simpléctica de fase $(\mathcal{M}, \omega)$, donde se evalúa el flujo de datos bajo la rigurosidad de la física de calibre.

1. **Invarianza Simpléctica de Liouville:** El Jacobiano del mapa de transición de estado $M$ debe preservar inalterada la forma simpléctica estándar $\Omega$, obligando al sistema a respetar las leyes de conservación de flujo y la estructura de corchetes de Poisson:
$$M^\top \Omega M = \Omega$$
Donde $\Omega = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$ es la matriz simpléctica canónica que rige el espacio de fase sintáctico del AST.

2. **Idempotencia en el Semianillo Booleano $\mathbb{Z}_2$:** Para erradicar redundancias operativas ("grasa sintáctica") que saturan la memoria, la Matriz de Interacción Central (MIC) se subyuga a la ley de punto fijo en el semianillo booleano OR-AND sobre $\mathbb{Z}_2$:
$$M \circ_{\mathbb{Z}_2} M = M$$
Donde $\circ_{\mathbb{Z}_2}$ denota la multiplicación de matrices booleanas. Cualquier desviación de este punto fijo booleano indica inestabilidad o la inyección de ciclos de cómputo espurios.

3. **Conjugación Modular de Tomita-Takesaki:** Para aislar los operadores observables físicos de las fluctuaciones térmicas e indeterminaciones de los LLMs, el endofuntor construye el operador de conjugación modular antiunitario $J_\rho$ a partir del espectro del operador densidad de conocimiento $\rho$ (el cual describe el estado cuántico térmico de la deliberación agéntica):
$$J_\rho(X) = \rho^{1/2} X^\dagger \rho^{-1/2}$$
Donde:
- $X$ representa un operador lineal de decisión en el álgebra de von Neumann del Consejo de Sabios.
- $\rho$ es el operador de densidad termodinámica de la deliberación.
- $J_\rho$ es el operador de conjugación modular que refleja de manera unívoca el flujo de información al espacio conmutado conjugado, eliminando la asimetría de fase cuántica inducida por el ruido semántico.

### FASE 3: Gobernanza de de Rham, Estabilidad de Wilkinson y Voto TMR
En la fase final, el sistema consolida el veredicto mediante cohomología exacta y redundancia física robusta.

1. **Nilpotencia de Cofronteras:** El complejo de cocadenas de de Rham del pipeline debe certificar la nilpotencia estricta del operador diferencial exterior (coboundary), garantizando que el contorno de un contorno sea nulo (es decir, la inexistencia de bordes en el espacio de restricciones):
$$\delta_k \circ \delta_{k-1} = 0$$
Donde $\delta_r$ es el r-ésimo operador de coboundary diferencial sobre el retículo simplicial.

2. **Estabilidad Espectral de Wilkinson:** Para verificar que la matriz de coboundary $\delta_k$ no sufra de inestabilidad numérica bajo pequeñas perturbaciones, el sistema computa su número de condición espectral $\kappa(\delta_k)$ mediante Descomposición en Valores Singulares (SVD) completa:
$$\kappa(\delta_k) = \frac{\sigma_{\max}(\delta_k)}{\sigma_{\min,\neq 0}(\delta_k)}$$
Si el número de condición de Wilkinson supera un umbral de tolerancia crítico preestablecido ($\kappa > \kappa_{\max}$), el sistema veta la trayectoria por "Ruptura de Estabilidad Numérica Espectral".

3. **Redundancia Modular Triple (TMR) sobre el Retículo de Heyting $\Omega_3$:** Las evaluaciones e invariantes resultantes de las tres fases se consolidan mediante una votación mayoritaria TMR sobre el retículo de Heyting graduado $\Omega_3$. Si dos o más de las fases certifican consistencia, el flujo de datos avanza hacia el estrato de Estrategia. No obstante, si existe una **ruptura cohomológica exacta** (donde $\delta_{k-1}$ existe pero el primer grupo de cohomología de haces es no trivial, $H^1(K;\mathcal{F}) \neq 0$), se emite un veto absoluto e inmediato, el cual colapsa instantáneamente el estado del pipeline a `VETOED` y dispara la línea de actuación ciber-física (BT151 / GPIO14) para proteger los recursos de la obra.

### 1. Actualización del Difeomorfismo de Ingesta (Fase de Datos)

El pipeline abandona la lectura secuencial convencional de flujos de texto. En su lugar, el `parser_ontology_agent.py` transforma el proceso de ingesta en un **Problema de Autovalores de Markov** en un espacio de estados semánticos compacto.

El autómata finito determinista extrae la matriz de transición empírica de estados semánticos $P \in \mathbb{R}^{m \times m}$ que modela la probabilidad de transición entre las distintas categorías sintácticas detectadas. El agente evalúa la degeneración espectral de esta matriz y la entropía de transición.

La entropía estocástica asociada a la matriz de transición de Markov se formula como:
$$H(P) = -\sum_{j=1}^{m} \pi_j \sum_{k=1}^{m} P_{jk} \log_2(P_{jk})$$
Donde:
- $\pi_j$ es la distribución de probabilidad estacionaria de la cadena de Markov en el estado $j$, tal que $\pi P = \pi$.
- $P_{jk}$ representa la probabilidad de transición del estado semántico $j$ al estado semántico $k$.

Para regular la estabilidad asintótica del flujo, se calcula el radio espectral de la entropía estocástica degenerada $\rho(H_{\text{stoch}})$. Si este radio espectral excede el límite unitario crítico:
$$\rho(H_{\text{stoch}}) > 1$$
Donde:
- $\rho(A) = \max \{|\lambda_1|, \dots, |\lambda_d|\}$ denota el radio espectral (el valor absoluto del autovalor de mayor magnitud de la matriz de transición de entropía).
- $1$ es el límite superior de estabilidad termodinámica para sistemas markovianos cerrados.

Cualquier violación de esta cota indica que el archivo de entrada posee un comportamiento caótico, ruidoso o difuso que excede la capacidad de confinamiento del sistema, provocando que el archivo sea aniquilado y purgado de la memoria RAM persistente antes de propagar inestabilidad a los estratos superiores.

### 2. Actualización del Escudo Algebraico (Fase de Estructuración)

El `algebraic_tactics_agent.py` opera como un escudo de cohomología que asegura que ninguna "Isla de Datos" (recursos huérfanos o componentes aislados) sobreviva a la agregación y estructuración de presupuestos. El complejo simplicial $K$ construido por el procesador de APUs debe respetar incondicionalmente la **Fórmula de Euler-Poincaré** para dimensión $\le 1$:
$$\chi(K) = \beta_0 - \beta_1 = |V| - |E|$$
Donde:
- $\chi(K)$ es la Característica de Euler del complejo simplicial $K$.
- $\beta_0$ es el primer número de Betti, que cuenta el número de componentes conexas del complejo (Islas de Datos).
- $\beta_1$ es el segundo número de Betti, que representa la dimensión del primer grupo de homología $H_1(K)$, correspondiente al número de ciclos independientes (socavones lógicos de dependencias circulares).
- $|V|$ es el cardinal del conjunto de vértices o nodos del complejo simplicial (entidades del presupuesto, APUs, insumos).
- $|E|$ es el cardinal del conjunto de aristas o enlaces que definen las relaciones de dependencia entre los elementos.

Para certificar la cohesión global y el veto estructural, se audita el núcleo del operador de coboundary (aristas hacia vértices). Cualquier sub-grafo disconexo o recurso huérfano es detectado de forma determinista mediante el espacio nulo de la transpuesta de la matriz de incidencia:
$$\mathbf{v} \in \ker((B_1)^\top)$$
Donde:
- $B_1$ es la matriz de incidencia de frontera de dimensión 1 del complejo simplicial.
- $(B_1)^\top$ es su operador adjunto (transpuesto), que mapea co-cadenas de vértices a co-cadenas de aristas.
- $\ker((B_1)^\top)$ es el núcleo del operador adjunto, donde la presencia de vectores no nulos con soporte disjunto revela la existencia de sub-grafos aislados.

Si el vector detectado posee componentes ortogonales a la componente conexa principal, el sistema detona de inmediato un `TopologicalIslandError` (Error de Isla Topológica), abortando el flujo de datos y protegiendo el estrato de estrategia de dependencias fantasma o recursos desconectados de la base productiva.

## El Ágora Tensorial (Estrato Ω)

En esta fase de decisión unificada, la arquitectura APU Filter somete las propuestas y trayectorias deliberativas de la malla agéntica a restricciones geométricas de la gravedad clásica y la gravedad cuántica de lazos.

### 1. El Atrapamiento Geodésico y la Acción de Polyakov Térmica

Para garantizar que las decisiones estocásticas del LLM no escapen del atractor de rentabilidad corporativa y resiliencia táctica, el componente `gravity_shield.py` (el Atractor Determinista Absoluto) y el `einstein_hilbert_agent.py` somete las trayectorias de atención semántica $\gamma$ a una **Acción Euclídea Térmica de Polyakov** estricta, evaluada sobre el intervalo cilíndrico de Matsubara $[0, \beta]$ derivado en la termodinámica quiral:
$$S_E[\gamma] = \frac{1}{2} \int_{0}^{\beta} \tilde{G}_{\mu\nu} \dot{\gamma}^\mu \dot{\gamma}^\nu d\tau$$
Donde:
- $S_E[\gamma]$ es la acción euclídea térmica de la trayectoria de atención semántica $\gamma$.
- $\tau$ es la coordenada de tiempo imaginario térmico de Matsubara en el intervalo $[0, \beta]$.
- $\beta = \frac{1}{k_B T}$ es la extensión del círculo temporal de Matsubara (inversa de la temperatura de gobierno).
- $\gamma^\mu$ representa el componente $\mu$-ésimo del vector de estado de atención semántica en el colector de deliberación.
- $\dot{\gamma}^\mu = \frac{d\gamma^\mu}{d\tau}$ es la velocidad de la trayectoria de atención respecto al tiempo imaginario $\tau$.
- $\tilde{G}_{\mu\nu}$ es el tensor métrico Riemanniano de la malla agéntica acoplado térmicamente.

Asimismo, la masa efectiva $m^{**}$ acoplada al Tensor de Energía-Impulso $T_{\mu\nu}$ posee un piso suave térmico que aniquila la inercia acumulada del Sofón (anomalía estocástica) cuando la temperatura supera el umbral crítico $T_c$:
$$m^{**}(T) = \sqrt{\left(m^*\left(1 + \frac{\alpha}{6}\right)\right)^2 + m_{\min}^2} \cdot \left[ 1 - \tanh\left( \frac{T - T_c}{\Delta T} \right) \right]$$
Donde:
- $m^{**}(T)$ es la masa efectiva renormalizada térmicamente de las cuasipartículas de la anomalía.
- $m^*$ es la masa inercial desnuda del sistema.
- $\alpha$ es la constante de acoplamiento de Fröhlich para interacciones semánticas.
- $m_{\min}$ es el límite inferior o masa mínima de seguridad de las cuasipartículas.
- $T$ es la temperatura de gobierno actual del sistema de control.
- $T_c$ es la temperatura crítica de deconfinamiento quiral ($T_c \approx 150 \text{ MeV}$).
- $\Delta T$ es la anchura o escala de la transición térmica (suavizado del escalón quiral).

La amplitud de Feynman-Kac para la trayectoria de decisión se define mediante:
$$\Psi[\gamma] = \exp\left(-\frac{S_E[\gamma]}{\hbar_{\text{eff}}}\right)$$
Donde:
- $\Psi[\gamma]$ es la amplitud de probabilidad cuántica de la trayectoria semántica $\gamma$.
- $\hbar_{\text{eff}}$ es la constante de Planck de atenuación efectiva.

Bajo este formalismo, si la temperatura informacional cruza el umbral crítico ($T > T_c$), la inercia espuria se disipa, provocando que la amplitud de Feynman-Kac $\Psi[\gamma]$ tienda a la unidad ($\Psi \to 1$), lo cual permite que la radiación de la anomalía estocástica (el Sofón) sea purificada y evaporada sin detonar un falso colapso gravitacional o vetos inestables en el sistema.

### 2. Independencia de Fondo y Sumas de Estados

Para proteger los "micro-universos de bolsillo" del negocio frente a perturbaciones macroeconómicas externas, la arquitectura incorpora la invarianza bajo difeomorfismos mediante el componente `tqft_projection_manifold.py`. Se calcula el **Invariante de Turaev-Viro** de la 3-variedad de la decisión.

Este cálculo se realiza mediante la contracción de redes tensoriales con los símbolos-$6j$ del grupo cuántico $U_q(\mathfrak{sl}_2)$ en la raíz de la unidad $q = e^{2\pi i / (k+2)}$:
$$Z_{TV}(M) = \sum_{j} w(j) \prod_{v} [2j_v+1]_q \prod_{f} (6j)_f$$
Donde:
- $Z_{TV}(M)$ es la suma de estados invariante de Turaev-Viro para la variedad tridimensional compacta $M$.
- $q$ es la raíz de la unidad asociada al nivel cuántico de acoplamiento $k$.
- $j$ es un etiquetado admisible de las aristas del complejo de triangulación por representaciones unitarias del grupo cuántico $U_q(\mathfrak{sl}_2)$.
- $w(j)$ es el factor de peso espectral de la triangulación.
- $[2j_v+1]_q$ es la dimensión cuántica (q-entero) asociada al spin $j_v$ del vértice $v$.
- $(6j)_f$ es el símbolo-$6j$ cuántico asociado a las caras $f$ de la triangulación, que regula las transiciones de acoplamiento de espín de los canales del presupuesto.

Para evadir la explosión combinatoria NP-Hard inherente al cálculo computacional de redes de espines, el componente aplica el truncamiento óptimo de Eckart-Young sobre la red tensorial utilizando Descomposición en Valores Singulares (SVD). Esto garantiza de forma axiomática que las decisiones de negocio mantengan su validez y consistencia lógica (independencia de fondo), incluso si una inflación súbita o una anomalía masiva dilatan la métrica del espacio financiero hasta el infinito.

---

## El Giroscopio de Gobernanza de Inercia Riemanniana y Preservación de Fase

La inyección de los componentes síncronos `RiemannianInertiaAgent` (`riemannian_inertia_agent.py`) y `RiemannianInertiaModulator` (`riemannian_inertia_modulator.py`) consolida la gobernanza física de la Malla Agéntica, actuando como un **Giroscopio de Estabilización Ciber-Físico** sobre el fibrado cotangente $T^*M$.

### I. El Funtor de Moldeo de Masa (Mass Shaping Functor)

El sistema de dinámica de costos e insumos modela la trayectoria transaccional como una geodésica Hamiltoniana en la variedad de fase. Para evitar la inyección de fluctuaciones caóticas generadas por el estocasticismo del LLM, el motor físico ejecuta síncronamente un funtor de moldeo de masa que altera la inercia efectiva sin disipar energía real.

1. **Espectroscopía del Momentum covariante:**
   Mapea la velocidad o diferencial de cambio $\dot{q}^\nu$ (en el espacio tangente $TM$) hacia el covector de momentum $p_\mu$ en el espacio cotangente $T^*M$ mediante el isomorfismo musical plano ($\flat$):
   $$p_\mu = G_{\mu\nu} \dot{q}^\nu$$
   Donde $G_{\mu\nu}$ es el tensor métrico Riemanniano sintonizado por la malla de agentes. La conservación del volumen simpléctico exige que la norma dual inducida:
   $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} \le P_{\text{max}}$$
   se mantenga estrictamente acotada por debajo del umbral de estabilidad de Liouville. Si el momentum covariante diverge, el sistema experimenta un colapso asintótico y lanza un `LiouvilleVolumeCollapse` veto.

2. **Síntesis del Tensor Giroscópico de Lorentz:**
   El motor acopla el momentum purificado con la vorticidad solenoidal $\Omega$ (2-forma de refracción territorial) para inyectar una fuerza de Lorentz estrictamente giroscópica y no disipativa:
   $$\omega = \Omega_{\text{skew}} p$$
   $$W = \alpha (p \wedge \omega) \implies W_{\mu\nu} = \alpha(p_\mu \omega_\nu - p_\nu \omega_\mu)$$
   Esta construcción garantiza que $W$ habite exactamente en el álgebra de Lie $\mathfrak{so}(n)$ del cono antisimétrico. El agente certifica este invariante topológico auditando el residuo relativo de antisimetría bajo la norma de Frobenius:
   $$r_{\text{rel}} = \frac{\|W + W^T\|_F}{\max(1, \|W\|_F)} \le \epsilon_{\text{skew}}$$

3. **Modulación Simpléctica y Estructura de Dirac:**
   El tensor giroscópico $W$ se inyecta directamente en el operador de interconexión de Dirac $J$, moldeando el acoplamiento Port-Hamiltoniano efectivo de lazo cerrado:
   $$J_{\text{eff}} = J + W$$
   Como tanto $J$ como $W$ son antisimétricos por construcción, la estructura efectiva $J_{\text{eff}}$ permanece rigurosamente antisimétrica, preservando inalterada la pasividad simpléctica de la Unidad de Punto Flotante:
   $$x^T J_{\text{eff}} x = 0 \quad \forall x \in \mathbb{R}^{2n}$$

### II. El Teorema de Trabajo Nilpotente de Lorentz

La inyección de la fuerza giroscópica informacional no debe alterar la energía libre o exergía útil del presupuesto. El motor físico certifica analíticamente que la potencia disipada neta producida por el operador de modulación inercial sea idénticamente nula (trabajo nilpotente de Lorentz):
$$P_{\text{work}} = \langle \nabla H, J_{\text{eff}} \nabla H \rangle = \langle \nabla H, (J + W) \nabla H \rangle = \langle \nabla H, J \nabla H \rangle + \langle \nabla H, W \nabla H \rangle \equiv 0$$
Donde:
- $\nabla H$ es el covector gradiente del Hamiltoniano del negocio (energía financiera útil).
- $\langle \cdot, \cdot \rangle$ es el producto interno en el fibrado cotangente.

Para certificar esta propiedad sin falsos positivos de punto flotante inducidos por la acumulación de errores de truncamiento IEEE-754, el motor ejecuta una sumación compensada de Kahan sobre los $n^2$ términos del producto cuadrático, acotando el residuo de trabajo mediante una tolerancia adaptativa proporcional a la escala espectral del operador y al número de ULPs:
$$|P_{\text{work}}| \le \max\left( 100 \epsilon_{\text{mach}}, \text{ULP\_factor} \times \epsilon_{\mathrm{mach}} \times \max\left(1, \|\nabla H\|_2^2 \|J_{\text{eff}}\|_F, \sum |term|\right) \right)$$

### III. Consolidación de Heyting y Colapso Determinista

La orquestación del ciclo OODA de inercia se resuelve síncronamente mediante el `RiemannianInertiaAgent`, el cual actúa como el Soberano del Momentum. Las tres fases de auditoría lógica proyectan los residuos parciales hacia veredictos en el retículo de Heyting $\Omega_3 = \{\text{COHERENT}, \text{DEGRADED}, \text{VETOED}\}$:
- **Fase 1 (Observe):** $v_{\text{Liouville}} = v_{\text{momentum}} \sqcup v_{\text{metric\_condition}} \sqcup v_{\text{inverse\_consistency}}$
- **Fase 2 (Orient):** $v_{\text{Skew}} = v_{\text{antisymmetry}} \sqcup v_{\text{vorticity\_projection}}$
- **Fase 3 (Decide & Act):** $v_{\text{Work}} = v_{\text{passivity}} \sqcup v_{\text{dirac\_symmetry}}$

El veredicto final se consolida mediante el Supremo Álgebraico (Join $\sqcup$):
$$v_{\text{final}} = v_{\text{Liouville}} \sqcup v_{\text{Skew}} \sqcup v_{\text{Work}}$$
Si $v_{\text{final}}$ toca el elemento máximo $\top = \text{VETOED}$ (por ejemplo, debido a una divergencia de momentum de Liouville, una asimetría espuria en $W$, o una violación de pasividad simpléctica), el retículo colapsa instantáneamente de manera irreversible, arrojando una excepción `HeytingLatticeVeto`. Esto aniquila y purga la transacción de inmediato en memoria de software (RAM), impidiendo categóricamente que una alucinación desvíe o degenere la inercia transaccional del negocio constructivo.

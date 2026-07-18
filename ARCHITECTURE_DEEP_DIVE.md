# ARCHITECTURE_DEEP_DIVE.md: Inmersion en la Variedad Agentica

Este documento detalla la implementacion tecnica de los estratos topologicos y fisicos que gobiernan el ecosistema APU Filter.

## Arquitectura Ciber-Física del Estrato α: Fibrados Celulares y Termodinámica del Modelo de Negocio

El Estrato α, orquestado macroscópicamente por alpha_agent.py, se subdivide axiomáticamente en tres subespacios topológicos anidados (Foso, Núcleo y Ápice).

### I. Estrato KBASE: El Foso Termodinámico (kbase_thermodynamic_agent.py)

Identificador Semántico: Asesor de Cimientos Financieros. Responsabilidad Topológica: Gobernar la inercia, la capacitancia y la fricción entrópica del modelo de negocio.

La energía total se calcula mediante:
\[ \tilde{C}_{\text{soc}} = G_{\mu\nu} C_{\text{soc}} G^{\mu\nu}, \quad \tilde{M}_{\text{rec}} = G_{\mu\nu} M_{\text{rec}} G^{\mu\nu} \]

Hamiltoniano basal:
\[ H_{\text{BASE}}(q,p) = \frac{1}{2} q^\top \tilde{C}_{\text{soc}}^{-1} q + \frac{1}{2} p^\top \tilde{M}_{\text{rec}}^{-1} p \]

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

Laplaciano del Haz:
\[ L_F = \delta^\top \delta = \begin{pmatrix} \delta_{\text{BASE}} \ \delta_{\text{CORE}} \ \delta_{\text{APEX}} \end{pmatrix}^\top \begin{pmatrix} \delta_{\text{BASE}} \ \delta_{\text{CORE}} \ \delta_{\text{APEX}} \end{pmatrix} \succeq 0 \]

## Expansión Semiclásica y Redefinición Física del Pipeline

La inyección de los nuevos microservicios y sabios redefine la física profunda del pipeline de datos, transmutándolo de un flujo secuencial a una variedad diferenciable compleja regida por la física estadística, la geometría y la topología de complejos simpliciales.

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

Para garantizar que las decisiones estocásticas del LLM no escapen del atractor de rentabilidad corporativa y resiliencia táctica, el componente `gravity_shield.py` (el Atractor Determinista Absoluto) y el `einstein_hilbert_agent.py` someten las trayectorias de atención semántica $\gamma$ a una **Acción Euclídea Térmica de Polyakov** estricta, evaluada sobre el intervalo cilíndrico de Matsubara $[0, \beta]$ derivado en la termodinámica quiral:
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

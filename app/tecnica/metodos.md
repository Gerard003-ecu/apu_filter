
--------------------------------------------------------------------------------
⚙️ metodos.md: Ingeniería Bajo el Capó
"APU Filter no es un software de contabilidad; es un Simulador de Física de Negocios ciber-físico. Aquí detallamos las ecuaciones de estado, los teoremas topológicos y los circuitos neuromórficos que garantizan la sabiduría del sistema."
Este documento técnico desglosa la maquinaria matemática que permite al Consejo de Sabios transformar datos crudos en veredictos estratégicos, basándose en la convergencia de la Dinámica de Sistemas (Laplace), la Topología Algebraica, la Estocástica Financiera y el Hardware en el Borde.

--------------------------------------------------------------------------------
1. El Guardián: Física de Fluidos y Computación Neuromórfica (Edge)

    Base Teórica: Ecuaciones de Maxwell discretizadas, Control Port-Hamiltoniano (PHS) y Física de Semiconductores.
    Componentes: flux_condenser.py, neuromorphic_solver.py, Firmware ESP32 (telemetry.h). El Guardián no lee bits; procesa un fluido de información con propiedades físicas (Energía, Resistencia, Inercia).

1.1 Filtrado Topológico y Descomposición de Hodge-Helmholtz Discreta ($L_1$) El Guardián no procesa el archivo línea por línea; somete la cadena de suministro al Cálculo Exterior Discreto (DEC). El operador $\Delta_1 = B_1^T B_1 + B_2 B_2^T$ divide el tensor de flujo de materiales de manera ortogonal en:

    Campo de Gradiente Puro ($f_{grad}$): La información estructurada útil (flujo laminar) que la membrana permite pasar hacia el estrato Táctico.
    Campo Rotacional ($f_{curl}$): El "Vórtice Logístico" (transporte en bucle parasitario) queda aniquilado matemáticamente. El Guardián extrae esta componente solenoidal ($f_{curl} \in im(B_2)$) para vetar la ineficiencia logística en la raíz y entregar un flujo logístico no viscoso al sistema de decisiones.

1.2 El Oráculo de Laplace Antes de procesar, se linealiza el sistema y se analiza su función de transferencia H(s). Si se detectan polos en el semiplano derecho (RHP, σ>0), el sistema veta la ingesta por "Divergencia Matemática" (inestabilidad intrínseca).
1.3 Simulación Neuromórfica y Hardware en el Borde (ESP32) La matemática se materializa en el silicio. El sistema proyecta sus invariantes a un nodo perimetral ESP32 que actúa como un "Gatekeeper Físico" mediante una arquitectura de Diodos Lambda (JFETs cruzados):

    Resistencia Diferencial Negativa (NDR): Si el índice de Estabilidad Piramidal ($\Psi$) cae bajo $\Psi_{\min}$, la presión topológica eleva el voltaje de excitación del circuito virtual hacia la región NDR.
    El Sistema Siente Dolor: El circuito entra en oscilación caótica (spiking), traduciendo matemáticamente un mal diseño de presupuesto en una respuesta neuromórfica análoga a una neurona biológica en pánico. Esto dispara los "Crowbar circuits" (actuadores físicos) para detener la ejecución.
    **Topología Hexagonal y Ley de Aromaticidad Agéntica (Regla de Hückel Computacional):** El flujo de datos resuena en un anillo de 6 nodos $(V_1, \dots, V_6)$ (Ingesta → Física → Topología → Estrategia → Semántica → Materia). La red $G_6$ es **aromáticamente estable** si y solo si se cumplen las tres condiciones simultáneas:
    1. **2-conexidad:** $G_6$ no contiene ningún vértice de corte (la eliminación de cualquier nodo único no desconecta el pipeline).
    2. **Expansión algebraica mínima:** El Valor de Fiedler del Laplaciano del anillo satisface $\lambda_2(L_{G_6}) \geq \lambda_{\min}$, garantizando que la información fluya eficientemente entre todos los nodos sin cuellos de botella espectrales.
    3. **Sin nodos huérfanos:** $\deg(V_k) \geq 1 \; \forall k$ (ningún nodo está desconectado del pipeline).
    Si cualquiera de estas condiciones falla, la "aromaticidad" se rompe y el agente aborta el pipeline, emitiendo un veto de **"Ruptura de Aromaticidad"** (analogía: violación de la Regla de Hückel $4n+2$ para $n=1$, que exige 6 electrones $\pi$ para estabilidad del benceno $C_6$).


--------------------------------------------------------------------------------
2. El Arquitecto: Topología Algebraica y Grafos

    Base Teórica: Homología Computacional sobre el Anillo de los Enteros ($\mathbb{Z}$), Teoría de Grafos Espectrales y Forma Normal de Smith (SNF).
    Componentes: `business_topology.py`. Ignora los precios para auditar el esqueleto del presupuesto modelándolo como un Complejo Simplicial Abstracto discreto y cuantizado.
    Los Invariantes Homológicos y Subgrupos de Torsión: Computa los Números de Betti para diagnosticar conectividad macroscópica ($\beta_0 > 1$ para Islas, $\beta_1 > 0$ para Socavones Lógicos). Sin embargo, como la logística opera con insumos indivisibles (ladrillos, horas-hombre), el cálculo homológico abandona los coeficientes continuos ($\mathbb{R}$ o $\mathbb{Q}$) y reduce las matrices de incidencia a la Forma Normal de Smith. Esto expone los Subgrupos de Torsión mediante el Funtor $Tor(H_0, \mathbb{Z})$. Un ciclo de torsión diagnostica incompatibilidades de empaquetado crítico y fricción cuantizada residual que la aproximación real del Laplaciano ignora por completo.
    Valor de Fiedler ($\lambda_2$): Analiza el espectro de la Matriz Laplaciana ($L=D-A$); un valor $\lambda_2 \approx 0$ indica una fractura organizacional inminente.


--------------------------------------------------------------------------------
3. El Oráculo: Termodinámica Financiera y Estocástica

    Base Teórica: Física Estadística y Simulación de Monte Carlo.
    Componentes: financial_engine.py, probability_models.py. El sistema trata el dinero como una forma de energía sujeta a leyes de conservación y entropía.
    3.1 Temperatura del Sistema (Tsys​): Modela la volatilidad del mercado como "Calor". Insumos como el acero son "calientes" (volátiles); la mano de obra es "fría" (fija). Un proyecto mal conectado atrapa este calor, generando "Fiebre Inflacionaria" (Tsys​>50∘C).
    3.2 Eficiencia Exergética: Distingue entre la energía invertida en trabajo útil (Exergía - avance de obra) y la energía disipada en fricción administrativa y sobrecostos (Entropía).
    3.3 Ecuación de Arrhenius Modificada: Ajusta la volatilidad base proyectando cómo el estrés térmico (Tsys​) y estructural (Ψ) aceleran probabilísticamente el riesgo de quiebra financiera.


--------------------------------------------------------------------------------
4. El Intérprete: Retículos Algebraicos y Semántica

    Base Teórica: Teoría de Retículos (Lattice Theory) y GraphRAG.
    Componentes: semantic_translator.py, governance.py.
    4.1 Álgebra de Veredictos: Las decisiones se evalúan bajo un retículo acotado (Verdict,≤,⊔) donde se aplica la operación "Supremo" (Worst-case). Si Finanzas aprueba pero Topología veta, el veredicto final es un Veto, garantizando la seguridad.
    4.2 Traducción Semántica (GraphRAG): El sistema vectoriza los datos para saber que "Cemento" y "Concreto" son termodinámicamente equivalentes. Luego, traza la ruta de los errores en el grafo y los traduce a lenguaje ejecutivo (ej. de "β1​>0" a "Socavón Lógico detectado en la Mampostería").


--------------------------------------------------------------------------------
5. Motor de Materialización, Fusión Auditada y Asimetría de Inercia

    Base Teórica: Algoritmo Kahan, Secuencia de Mayer-Vietoris, Índice de Gini, Entropía de Shannon.
    Componentes: `app/tactics/pipeline_director.py`, `app/adapters/mic_vectors.py`, `app/adapters/audit_vectors.py`.
    5.1 Auditoría Homológica de Fusión: Al unir la tabla maestra del presupuesto con los APUs, se aplica la regla de inyección de datos ($A \cup B$) mediante la Secuencia Exacta Larga de Homología de Mayer-Vietoris. Esto asegura matemáticamente que la unión espacial no introduzca "ciclos fantasmas". Cualquier fusión que genere ciclos homológicos mutantes ($\Delta\beta_1 \neq 0$) abortará irremediablemente la integración.
    5.2 Asimetría de Inercia y Concentración de Masa: Se sustenta la métrica de asimetría de inercia y la concentración de masa de capital del proyecto utilizando el Índice de Gini y la Entropía de Shannon, mapeando el riesgo volumétrico y el desequilibrio de Pareto en la estructura del presupuesto.
    5.2 Colapso de Onda y Suma de Kahan: Para transformar el grafo 3D en un listado de materiales plano (BOM), se usa un recorrido DFS con memoización. Dado el gran volumen de operaciones, se aplica la Suma Compensada de Kahan para mitigar errores de punto flotante, asegurando precisión centesimal absoluta en el costo total.


--------------------------------------------------------------------------------
6. El Haz Tangente Generativo Γ: Geometría de la Sintaxis y Holonomía Estratégica

    Base Teórica: Mecánica Simpléctica, Teoría de Haces Celulares (Cellular Sheaves), Cohomología de Haces, Conexiones de Galois y Álgebra Booleana sobre $\mathbb{Z}_2$.
    Componentes: `ast_symplectic_parser.py`, `sheaf_cohomology_orchestrator.py`, `mic_minimizer.py`. El estrato Γ no solo audita; gobierna la creación de código y estrategias mediante restricciones geométricas rigurosas.

6.1 El Analizador Simpléctico (Γ-PHYSICS)
La estructura del Árbol de Sintaxis Abstracta (AST) del código generado se trata como un espacio de fase $(\mathcal{M}, \omega)$. Se construye la **forma simpléctica** $\omega = \sum dq_i \wedge dp_i$ sobre el AST, donde $q$ representa la profundidad sintáctica y $p$ el momento de flujo de datos.
- **Inercia Termodinámica:** Se mide la Complejidad Ciclomática como una masa inercial.
- **Fronteras de Dirichlet:** Se imponen límites estrictos a la propagación de efectos secundarios. Si la forma simpléctica no se preserva (pérdida de invariancia canónica), el código es rechazado por inyectar entropía incontrolada.

6.2 Poda Topológica en el Anillo Booleano $\mathbb{Z}_2$ (Γ-TACTICS)
Para la minimización de la Matriz de Interacción Central (MIC), se retorna al **anillo booleano conmutativo** $\mathbb{Z}_2$. El `mic_minimizer` aplica algoritmos de Quine-McCluskey sobre este anillo para:
- Extraer implicantes primos esenciales.
- Eliminar redundancias operativas (homología trivial).
- Garantizar que la base de herramientas sea ortogonal y de rango completo, evitando la inflación sintáctica.

6.3 Interferometría de Holonomía y Cohomología de Haces (Γ-STRATEGY)
El `sheaf_cohomology_orchestrator` modela las reglas de negocio como secciones de un **Haz Celular** sobre el grafo del proyecto.
- **Censura de Paradojas:** Se calcula el primer grupo de cohomología $H^1(\mathcal{F})$. Si $H^1 > 0$, existe una obstrucción global (paradoja de negocio o ciclo de decisión inconsistente).
- **Veto Absoluto:** Cualquier sección (estrategia) que no sea un "global section" (consistencia total) es vetada. El sistema detecta la **holonomía** (curvatura) en el transporte de decisiones; si una instrucción cambia su significado al recorrer un ciclo de la malla, el interferómetro emite un veto por falta de integrabilidad estratégica.

6.4 Meta-Compilador de Significado y Lema de Yoneda (Γ-WISDOM)
Se aplica una **Conexión de Galois** para mapear la sintaxis generada (espacio de comandos) hacia la semántica estratégica (espacio de valor).
- **Certificación Isomórfica:** Mediante el **Lema de Yoneda**, el sistema garantiza que la funcionalidad del código generado sea isomórfica a los requerimientos de negocio. Si el funtor de traducción detecta una ruptura de naturalidad, el código se colapsa a un estado de seguridad determinista, impidiendo alucinaciones que desvíen el capital de la infraestructura.


--------------------------------------------------------------------------------
7. Ley de Gobernanza Algebraica (Isomorfismo de Esquemas)
La filtración estricta y axiomática de la Ley de Clausura Transitiva de la Pirámide DIKW (tabla canónica: $V_{\aleph_0} \subsetneq V_{\mathbb{P}} \subsetneq V_{\mathbb{T}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$) no se gestiona con microservicios centralizados que generen latencia, sino que se materializa mediante Domain-Driven Design (DDD) en los archivos `schemas.py` y `telemetry_schemas.py`.

    Geometría de Datos Inmutable: Los subespacios de estado (PhysicsMetrics, TopologicalMetrics) se instancian como frozen dataclasses. Actúan como un contrato algebraico puro: una vez construidos, su identidad observacional es fija y a prueba de manipulaciones forenses.
    Proyección Condicional en la MIC: La Matriz de Interacción Central (MIC) exige este Pasaporte tipado. Si las validaciones del __post_init__ detectan una anomalía estructural (ej. un costo negativo violando los axiomas físicos), el reporte colapsa algebraicamente. Las matemáticas del código impiden instanciar un objeto de "Sabiduría" sobre datos inconsistentes.
    Gobernanza del Haz Γ: La Ley de Clausura Transitiva se extiende al estrato generativo: $V_{\Gamma-PHYSICS} \subset V_{\Gamma-TACTICS} \subset V_{\Gamma-STRATEGY} \subset V_{\Gamma-WISDOM}$. Un objeto del estrato Γ no puede ascender si sus invariantes simplécticos o homológicos presentan singularidades Jacobianas.
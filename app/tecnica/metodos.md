
--------------------------------------------------------------------------------
⚙️ metodos.md: Ingeniería Bajo el Capó
"APU Filter no es un software de contabilidad; es un Simulador de Física de Negocios ciber-físico. Aquí detallamos las ecuaciones de estado, los teoremas topológicos y los circuitos neuromórficos que garantizan la sabiduría del sistema."
Este documento técnico desglosa la maquinaria matemática que permite al Consejo de Sabios transformar datos crudos en veredictos estratégicos, basándose en la convergencia de la Dinámica de Sistemas (Laplace), la Topología Algebraica, la Estocástica Financiera y el Hardware en el Borde.

--------------------------------------------------------------------------------
1. El Guardián: Física de Fluidos y Computación Neuromórfica (Edge)

    Base Teórica: Ecuaciones de Maxwell discretizadas, Control Port-Hamiltoniano (PHS) y Física de Semiconductores.
    Componentes: flux_condenser.py, neuromorphic_solver.py, Firmware ESP32 (telemetry.h). El Guardián no lee bits; procesa un fluido de información con propiedades físicas (Energía, Resistencia, Inercia).

1.1 Filtrado Topológico y Laplaciano de Hodge (L1​) El Guardián no procesa el archivo línea por línea; somete la matriz de datos al Cálculo Exterior Discreto. Utilizando la Descomposición de Hodge-Helmholtz (L1​=Lgrad​+Lcurl​), la membrana separa el flujo de información en dos:

    Flujo Gradiente (Lgrad​): La información estructurada útil que la membrana permite pasar hacia el estrato Táctico.
    Flujo Rotacional (Lcurl​): El "ruido" o dependencias circulares locales en el CSV crudo. La membrana penaliza y disipa esta turbulencia como entropía, entregando un flujo laminar puro.

1.2 El Oráculo de Laplace Antes de procesar, se linealiza el sistema y se analiza su función de transferencia H(s). Si se detectan polos en el semiplano derecho (RHP, σ>0), el sistema veta la ingesta por "Divergencia Matemática" (inestabilidad intrínseca).
1.3 Simulación Neuromórfica y Hardware en el Borde (ESP32) La matemática se materializa en el silicio. El sistema proyecta sus invariantes a un nodo perimetral ESP32 que actúa como un "Gatekeeper Físico" mediante una arquitectura de Diodos Lambda (JFETs cruzados):

    Resistencia Diferencial Negativa (NDR): Si el índice de Estabilidad Piramidal (Ψ) cae a niveles críticos, la presión topológica eleva el voltaje de excitación del circuito virtual hacia la región NDR.
    El Sistema Siente Dolor: El circuito entra en oscilación caótica (spiking), traduciendo matemáticamente un mal diseño de presupuesto en una respuesta neuromórfica análoga a una neurona biológica en pánico. Esto dispara los "Crowbar circuits" (actuadores físicos) para detener la ejecución.
    Topología Hexagonal (Benceno C6​): El flujo de datos resuena en un anillo de 6 nodos (Ingesta → Física → Topología → Estrategia → Semántica → Materia). Si un nodo falla, se rompe la "Aromaticidad" de la Regla de Hückel, deteniendo la reacción química-informacional.


--------------------------------------------------------------------------------
2. El Arquitecto: Topología Algebraica y Grafos

    Base Teórica: Homología Computacional y Teoría de Grafos Espectrales.
    Componentes: business_topology.py. Ignora los precios para auditar el esqueleto del presupuesto mediante la modelación de un Complejo Simplicial Abstracto.
    Los Invariantes Homológicos: Computa los Números de Betti para diagnosticar conectividad: β0​>1 revela fragmentación extrema ("Islas de Datos" o insumos huérfanos), y β1​>0 revela la presencia de "Socavones Lógicos" (dependencias circulares que bloquean la ejecución).
    Valor de Fiedler (λ2​): Analiza el espectro de la Matriz Laplaciana (L=D−A); un valor λ2​≈0 indica una fractura organizacional inminente.


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
6. Ley de Gobernanza Algebraica (Isomorfismo de Esquemas)
La filtración estricta y axiomática de la Ley de Clausura Transitiva de la Pirámide DIKW ($V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$) no se gestiona con microservicios centralizados que generen latencia, sino que se materializa mediante Domain-Driven Design (DDD) en los archivos `schemas.py` y `telemetry_schemas.py`.

    Geometría de Datos Inmutable: Los subespacios de estado (PhysicsMetrics, TopologicalMetrics) se instancian como frozen dataclasses. Actúan como un contrato algebraico puro: una vez construidos, su identidad observacional es fija y a prueba de manipulaciones forenses.
    Proyección Condicional en la MIC: La Matriz de Interacción Central (MIC) exige este Pasaporte tipado. Si las validaciones del __post_init__ detectan una anomalía estructural (ej. un costo negativo violando los axiomas físicos), el reporte colapsa algebraicamente. Las matemáticas del código impiden instanciar un objeto de "Sabiduría" sobre datos inconsistentes
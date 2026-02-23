
--------------------------------------------------------------------------------
⚙️ metodos.md: Ingeniería Bajo el Capó
"APU Filter no es un software de contabilidad; es un Simulador de Física de Negocios ciber-físico. Aquí detallamos las ecuaciones de estado, los teoremas topológicos y los circuitos neuromórficos que garantizan la sabiduría del sistema."
Este documento técnico desglosa la maquinaria matemática que permite al Consejo de Sabios transformar datos crudos en veredictos estratégicos, basándose en la convergencia de la Dinámica de Sistemas (Laplace), la Topología Algebraica, la Estocástica Financiera y el Hardware en el Borde.

--------------------------------------------------------------------------------
1. El Guardián: Física de Fluidos y Computación Neuromórfica (Edge)
Base Teórica: Ecuaciones de Maxwell discretizadas, Control Port-Hamiltoniano (PHS) y Física de Semiconductores. Componentes: flux_condenser.py, neuromorphic_solver.py, Firmware ESP32 (telemetry.h).
El Guardián no lee bits; procesa un fluido de información con propiedades físicas (Energía, Resistencia, Inercia).
*   **1.1 Filtrado Topológico y Laplaciano de Hodge ($L_1$):** El Guardián no procesa el archivo línea por línea; somete la matriz de datos al Cálculo Exterior Discreto. Utilizando la Descomposición de Hodge-Helmholtz ($L_1 = L_{grad} + L_{curl}$), la membrana separa el flujo de información en dos:
    *   **Flujo Gradiente ($L_{grad}$):** La información estructurada útil que la membrana permite pasar hacia el estrato Táctico.
    *   **Flujo Rotacional ($L_{curl}$):** El "ruido" o dependencias circulares locales en el CSV crudo. La membrana penaliza y disipa esta turbulencia como entropía, entregando un flujo laminar puro.
• 1.2 El Oráculo de Laplace: Antes de procesar, se linealiza el sistema y se analiza su función de transferencia H(s). Si se detectan polos en el semiplano derecho (RHP, σ>0), el sistema veta la ingesta por "Divergencia Matemática" (inestabilidad intrínseca).
**1.3 Simulación Neuromórfica y Hardware en el Borde (ESP32)**
La matemática se materializa en el silicio. El sistema proyecta sus invariantes a un nodo perimetral ESP32 que actúa como un "Gatekeeper Físico" mediante una arquitectura de Diodos Lambda (JFETs cruzados):
* **Resistencia Diferencial Negativa (NDR):** Si el índice de Estabilidad Piramidal ($\Psi$) cae a niveles críticos, la presión topológica eleva el voltaje de excitación del circuito virtual hacia la región NDR.
* **El Sistema Siente Dolor:** El circuito entra en oscilación caótica (spiking), traduciendo matemáticamente un mal diseño de presupuesto en una respuesta neuromórfica análoga a una neurona biológica en pánico. Esto dispara los "Crowbar circuits" (actuadores físicos) para detener la ejecución.
* **Topología Hexagonal (Benceno $C_6$):** El flujo de datos resuena en un anillo de 6 nodos (Ingesta $\to$ Física $\to$ Topología $\to$ Estrategia $\to$ Semántica $\to$ Materia). Si un nodo falla, se rompe la "Aromaticidad" de la Regla de Hückel, deteniendo la reacción química-informacional.

2. El Arquitecto: Topología Algebraica y Grafos

Base Teórica: Teoría de Grafos Espectrales y Homología Computacional. Componentes: business_topology.py, topology_viz.py.
El presupuesto se modela como un Complejo Simplicial Abstracto, ignorando inicialmente los precios para estudiar únicamente la "forma" de las dependencias.
• 2.1 Invariantes Topológicos (Números de Betti βn​):
    ◦ β0​ (Islas de Datos): Mide la fragmentación. Si β0​>1, existen "recursos huérfanos" comprados pero desconectados de la obra (desperdicio seguro).
    ◦ β1​ (Socavones Lógicos): Mide la complejidad circular. Si β1​>0, existen dependencias circulares (A→B→A) que rompen la causalidad e impiden calcular costos reales.
    ◦ Característica de Euler (χ=β0​−β1​): Cuantifica la Entropía Estructural. Se usa para establecer el Pricing Dinámico del modelo de negocio (a menor entropía, menor tarifa).
• 2.2 Estabilidad Espectral (Valor de Fiedler λ2​): Analiza el espectro de la Matriz Laplaciana (L=D−A). Si λ2​≈0, diagnostica una "Fractura Organizacional" (silos incomunicados).
• 2.3 Índice de Estabilidad Piramidal (Ψ): Define la robustez del centro de gravedad. Si Ψ<1.0 (Pirámide Invertida), alerta de un riesgo crítico de colapso logístico por tener demasiadas actividades soportadas en muy pocos proveedores.

3. El Oráculo: Termodinámica Financiera y Estocástica

Base Teórica: Física Estadística y Simulación de Monte Carlo. Componentes: financial_engine.py, probability_models.py.
El sistema trata el dinero como una forma de energía sujeta a leyes de conservación y entropía.
• 3.1 Temperatura del Sistema (Tsys​): Modela la volatilidad del mercado como "Calor". Insumos como el acero son "calientes" (volátiles); la mano de obra es "fría" (fija). Un proyecto mal conectado atrapa este calor, generando "Fiebre Inflacionaria" (Tsys​>50∘C).
• 3.2 Eficiencia Exergética: Distingue entre la energía invertida en trabajo útil (Exergía - avance de obra) y la energía disipada en fricción administrativa y sobrecostos (Entropía).
• 3.3 Ecuación de Arrhenius Modificada: Ajusta la volatilidad base proyectando cómo el estrés térmico (Tsys​) y estructural (Ψ) aceleran probabilísticamente el riesgo de quiebra financiera.

4. El Intérprete: Retículos Algebraicos y Semántica

Base Teórica: Teoría de Retículos (Lattice Theory) y GraphRAG. Componentes: semantic_translator.py, governance.py.
• 4.1 Álgebra de Veredictos: Las decisiones se evalúan bajo un retículo acotado (Verdict,≤,⊔) donde se aplica la operación "Supremo" (Worst-case). Si Finanzas aprueba pero Topología veta, el veredicto final es un Veto, garantizando la seguridad.
• 4.2 Traducción Semántica (GraphRAG): El sistema vectoriza los datos para saber que "Cemento" y "Concreto" son termodinámicamente equivalentes. Luego, traza la ruta de los errores en el grafo y los traduce a lenguaje ejecutivo (ej. de "β1​>0" a "Socavón Lógico detectado en la Mampostería").

*   **4.3 El Funtor de Traducción Semántica (`SemanticTranslator`):**
    El paso final del sistema no es generar texto con inteligencia artificial estocástica (riesgo de alucinaciones), sino aplicar un Funtor determinista desde el Espacio Métrico hacia el Espacio Narrativo.
    *   **GraphRAG Causal:** Al detectar un fallo topológico (ej. un "Socavón Lógico" donde el Acero depende del Transporte y viceversa), el traductor no solo arroja el error $\beta_1 > 0$. Utiliza *Retrieval-Augmented Generation* sobre el Grafo (GraphRAG) para trazar la ruta exacta de la dependencia y generar la cadena causal, produciendo una narrativa auditable: *"Ruta del ciclo detectada: Acero -> Transporte -> Acero. Esto crea una indeterminación matemática en la valoración"*.
    *   **Isomorfismo de Severidad:** La traducción semántica respeta un isomorfismo estricto entre el retículo de severidad técnica (`OPTIMO`, `ADVERTENCIA`, `CRITICO`) y el retículo de veredicto de negocio (`VIABLE`, `CONDICIONAL`, `RECHAZAR`), garantizando que la "Voz del Consejo" refleje exactamente la realidad matemática sin sesgos optimistas.

5. Motor de Materialización y Fusión Auditada

Base Teórica: Algoritmo Kahan, Secuencia de Mayer-Vietoris. Componentes: pipeline_director.py, matter_generator.py.
• 5.1 Auditoría Homológica de Fusión: Al unir la tabla maestra del presupuesto con los APUs, se aplica la secuencia exacta de Mayer-Vietoris. Esto asegura matemáticamente que la unión espacial (A∪B) no introduzca "ciclos fantasmas" (Δβ1​>0) inexistentes en las fuentes originales.
• 5.2 Colapso de Onda y Suma de Kahan: Para transformar el grafo 3D en un listado de materiales plano (BOM), se usa un recorrido DFS con memoización. Dado el gran volumen de operaciones, se aplica la Suma Compensada de Kahan para mitigar errores de punto flotante, asegurando precisión centesimal absoluta en el costo total.

**6. Ley de Gobernanza Algebraica (Isomorfismo de Esquemas)**
La filtración estricta de la Pirámide DIKW ($V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$) no se gestiona con microservicios centralizados que generen latencia, sino que se materializa mediante **Domain-Driven Design (DDD)** en los archivos `schemas.py` y `telemetry_schemas.py`.

*   **Geometría de Datos Inmutable:** Los subespacios de estado (`PhysicsMetrics`, `TopologicalMetrics`) se instancian como *frozen dataclasses*. Actúan como un contrato algebraico puro: una vez construidos, su identidad observacional es fija y a prueba de manipulaciones forenses.
*   **Proyección Condicional en la MIC:** La Matriz de Interacción Central (MIC) exige este Pasaporte tipado. Si las validaciones del `__post_init__` detectan una anomalía estructural (ej. un costo negativo violando los axiomas físicos), el reporte colapsa algebraicamente. Las matemáticas del código impiden instanciar un objeto de "Sabiduría" sobre datos inconsistentes.

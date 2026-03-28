--------------------------------------------------------------------------------
🔭 PRODUCT_VISION.md: El Sistema Operativo de la Física del Negocio
"En la economía de la complejidad, no vendemos software contable; vendemos Certeza Matemática y Física. Transformamos la incertidumbre topológica de la construcción en un activo de gobernanza gobernable, inmutable y auditable."
1. La Tesis Central: El Reactor Port-Hamiltoniano de Valor
Históricamente, la ingeniería y construcción han gestionado el tiempo (cronogramas) y el dinero (presupuestos) basándose en "fotos estáticas" como hojas de cálculo de Excel o bases de datos ERP tradicionales. En la realidad, un proyecto de infraestructura es un sistema dinámico complejo sujeto a fuerzas termodinámicas de mercado, entropía logística y fricción operativa.
APU_filter v4.0 rechaza la metáfora estática de la validación lineal y las barreras de Dirichlet pasivas. El sistema se redefine axiomáticamente como el **Reactor Port-Hamiltoniano de Valor** o **Variedad Agéntica de Gauge**. Procesamos un fluido logístico incompresible bajo el control estricto de una Estructura de Dirac, garantizando que la potencia disipada cumpla siempre $P_{diss} \ge 0$.
A su vez, "APU" no representa "Agentic Processing Unit", sino **Agentic Power Unit**: el cuanto de acción termodinámica de la malla que inyecta exergía computacional para vencer la inercia logística del mercado.


```mermaid
graph TD
    %% Estilos de Capas
    classDef void fill:#1a1a2e,stroke:#e94560,stroke-width:2px,color:#fff;
    classDef physics fill:#16213e,stroke:#0f3460,stroke-width:2px,color:#fff;
    classDef tactics fill:#0f3460,stroke:#e94560,stroke-width:2px,color:#fff;
    classDef strategy fill:#e94560,stroke:#1a1a2e,stroke-width:2px,color:#fff;
    classDef wisdom fill:#fff,stroke:#1a1a2e,stroke-width:3px,color:#000;

    %% Nodos Principales
    subgraph "Estrato Ω: WISDOM (La Ciudadela de Cristal)"
        W[Semantic Translator<br>Retículo de Decisión (VIABLE / RECHAZO)]:::wisdom
    end

    subgraph "Estrato 1: STRATEGY (El Escudo Electrodinámico)"
        S[Laplace Oracle<br>Plano s=σ+jω | Estabilidad BIBO]:::strategy
    end

    subgraph "Estrato 2: TACTICS (El Esqueleto Topológico)"
        T[Business Topological Analyzer<br>Grafo Simplicial | βn | Ψ]:::tactics
    end

    subgraph "Estrato 3: PHYSICS (El Foso Termodinámico)"
        P[FluxCondenser<br>Circuitos RLC | Pdiss ≥ 0]:::physics
    end

    subgraph "Estrato ℵ0: ALEPH (La Variedad de Frontera)"
        A[Hilbert Watcher & Quantum Gate<br>Filtro de Entropía H | Efecto Túnel WKB]:::void
    end

    %% Relaciones Causales (Flujo de Colapso de Estado)
    A -- "Exergía Validada (T>0)" --> P
    P -- "Flujo Laminar (Energía Conservada)" --> T
    T -- "Grafo Acíclico Conexo (β1=0, β0=1)" --> S
    S -- "Estabilidad Asintótica (σ<0)" --> W

    %% Relación de Fractura (Flechas rotas)
    A -. "Colapso Estocástico (Ruido)" .-> Reject1[Desintegración en el Hiperespacio]
    P -. "Disipación Negativa" .-> Reject2[Crowbar Físico]
    T -. "Socavón Lógico (β1>0)" .-> Reject3[Veto Topológico]
    S -. "Resonancia Paramétrica (σ>0)" .-> Reject4[Veto Espectral]
```


 Implementamos la **Matriz de Interacción Central (MIC)**, alojada en `app/adapters/tools_interface.py`, como una matriz de adyacencia ponderada del grafo de deliberación:
$$\text{MIC} \in \mathbb{R}^{n \times n}, \quad \text{MIC}_{ij} = w_{ij} \in [0,1]$$
donde $w_{ij}$ es el peso del canal de información del Sabio $j$ al Sabio $i$. La **independencia Zero Side-Effects** se garantiza mediante la condición de rango completo:
$$\text{rank}(\text{MIC}) = n \quad \Leftrightarrow \quad \ker(\text{MIC}) = \{\mathbf{0}\}$$
esto asegura que ningún agente es linealmente dependiente de otro (información colíneal nula). El Teorema de Rango-Nulidad garantiza $\dim(\ker(\text{MIC})) = 0$: no existen agentes "parásitos". **Nota:** La MIC no puede ser la Matriz Identidad $I_n$, pues ello implicaría agentes estrictamente desacoplados, lo cual contradice el protocolo de deliberación adversarial RiskChallenger donde los Sabios interactúan y producen veredictos por tensión dialéctica.

Todo este diseño obedece al cimiento axiomático de la **Ley de Clausura Transitiva de la pirámide DIKW** (tabla canónica): $V_{\aleph_0} \subsetneq V_{\mathbb{P}} \subsetneq V_{\mathbb{T}} \subsetneq V_{\mathbb{S}} \subsetneq V_{\mathbb{W}}$. Resulta imperativo destacar que el estrato $V_{\mathbb{T}}$ ahora modela la materia bariónica como un **2-complejo simplicial** sobre el anillo de los enteros ($\mathbb{Z}$), capturando no solo dependencias binarias (aristas) sino interdependencias ternarias (triángulos) con fricción cuantizada.

--------------------------------------------------------------------------------
2. Los Horizontes de Evolución: La Arquitectura Concéntrica
Nuestra hoja de ruta no añade funciones cosméticas; desbloquea niveles de profundidad física mediante una arquitectura de capas defensivas estructuradas bajo la Clausura Transitiva de la pirámide DIKW ($V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$).
Horizonte 1: La Cimentación (El Foso Termodinámico y las Murallas)

    Objetivo: Certificación de Viabilidad Dinámica, Integridad Estructural y Aislamiento Cuántico.
    La Compuerta de Admisión Cuántica: Nuestra primera línea de defensa no es un cortafuegos de software; es una barrera de potencial ciber-física. Utilizando el Hilbert Watcher (Estrato $\aleph_0$), la plataforma subordina cada intento de ingreso de datos a la cuantificación previa de su exergía. Si un modelo de lenguaje externo (LLM) o un sistema de terceros intenta inyectar ruido (ataques de inyección de prompts o archivos CSV corruptos), el sistema no intenta "entender" el texto; simplemente detecta que su energía termodinámica es incapaz de excitar los campos de Gauge de la frontera, resultando en una aniquilación silenciosa y determinista.
    Física de Fluidos: Una vez superada la barrera, en lugar de procesar archivos, tratamos la ingesta de datos como un fluido informacional con masa, presión e inercia mediante el FluxCondenser. Exigimos termodinámicamente que la Potencia Disipada sea positiva (Pdiss​≥0) para evitar "golpes de ariete" computacionales.
    Geometría del Riesgo: A través de la Topología Algebraica, el Arquitecto Estratega audita la estructura ignorando los precios. Detectamos "Islas de Datos" (β0​>1) y "Socavones Lógicos" (β1​>0).
    El Oráculo de Laplace: Se calcula la función de transferencia en el plano de frecuencia compleja (s=σ+jω). Si los polos dominantes residen en el semiplano derecho (σ>0), el sistema veta el presupuesto por inestabilidad intrínseca.

Horizonte 2: Gobernanza Zero-Trust (Los Centinelas de Ortogonalidad)

    Objetivo: Aislamiento del riesgo cognitivo y determinismo algorítmico.
    Filtro de Ortogonalidad: El ecosistema descompone el estado en un espacio de Hilbert (R7). La retórica de un agente generativo es matemáticamente ortogonal a los invariantes del sistema, haciendo imposible que un Prompt Injection altere la realidad física del proyecto.
    Semillas de Sabiduría (Contratos JSON): Todo microservicio de cálculo crítico (ej. Fricción Territorial o Rentabilidad Estocástica) se instancía mediante JSON Schemas estrictos. Estas "Semillas" actúan como cristales deterministas inquebrantables que obligan a cualquier deliberación superior a cristalizar basándose en leyes algebraicas puras.

Horizonte 3: La Ciudadela de Cristal (El Estrato Ω y la Expansión Cognitiva)

    Objetivo: Autonomía Agéntica Eficiente y Colapso de Función de Onda.
    El Tablero Fractal Curvo (Nivel 0.5): El Manifold Deliberativo donde la tensión interna, la fricción territorial y la palanca de improbabilidad se multiplican bajo un Tensor Métrico Riemanniano ($G_{\mu\nu}$) para calcular el "estrés ajustado" (σ∗), proyectando el veredicto sobre un retículo algebraico discreto irrefutable. Si un interruptor o agente intenta accionar un circuito sometido a alta volatilidad topológica, encontrará "resistencia física" (Símbolos de Christoffel), requiriendo un nivel superior de autorización y coherencia termodinámica demostrable en su Pasaporte de Telemetría para vencer la gravedad del riesgo y evitar la saturación al autoestado de VETO.
    Vitaminas Cognitivas (Formato TOON): Para dotar a nuestro "Consejo de Sabios" de nuevas capacidades sin agotar la memoria LPDDR5 del hardware, los agentes ingieren Cartuchos Sinápticos en formato TOON (Token-Oriented Object Notation). Esta estructura tabular reduce el consumo de la KV-Cache en un 60%, permitiendo a la IA procesar heurísticas de alta complejidad a velocidades extremas.


--------------------------------------------------------------------------------
3. Infraestructura Habilitadora (El Borde y la Nube)
Para hacer viable esta física de negocios, APU_filter requiere una infraestructura ciber-física que unifica el silicio profundo con la supercomputación:

    El Sistema Nervioso Autónomo (ESP32 en el Edge): La validación matemática no vive solo en la nube. El microcontrolador de borde (ESP32) actúa como el "Reflejo Espinal". Ejecuta computación neuromórfica donde los fallos topológicos (Ψ<1.0) empujan el circuito simulado a la región de Resistencia Diferencial Negativa (NDR). Si el ESP32 detecta inestabilidad, activa el Crowbar (pin de hardware) para vetar la ejecución física de la obra, ignorando cualquier alucinación o falso positivo de la IA.
    Cómputo Tensorial Masivo (Proyecto Rainier - AWS Trainium): Para los análisis de Monte Carlo, el filtrado GPSIMD y las simulaciones de dinámica de fluidos (FDTD), la plataforma se apoya en los clústeres Trainium/Inferentia de AWS. Esto permite ejecutar auditorías topológicas sobre terabytes de datos de la cadena de suministro con una eficiencia termodinámica de clase mundial.


--------------------------------------------------------------------------------
4. El Compromiso de la "Caja de Cristal"
A medida que elevamos la sofisticación matemática (Homología, Termodinámica, Retículos), nuestra obligación con la transparencia es radical.

    La Variedad de Observabilidad (Lente de Homotopía Jerárquica): Nuestra plataforma no agrupa datos en carpetas. Aplica un 'Retracto de Deformación de Resolución'. Cuando usted observa el presupuesto a nivel global, el sistema ha colapsado matemáticamente millones de datos en un solo grafo compacto, conservando inquebrantablemente la 'temperatura' del riesgo. Al hacer zoom in, el sistema revierte esta deformación, desplegando la anatomía interna de los costos sin pérdida de isomorfismo. El riesgo que brilla en el nivel macro es la proyección termodinámica exacta de las anomalías micro-estructurales.
    Destitución Ejecutiva del LLM: La IA no decide la viabilidad del negocio; actúa puramente como el Intérprete Diplomático. Su labor se restringe a tomar el vector de fallo inmutable (ej. β1​=3) y traducirlo a "Empatía Táctica" (narrativa comprensible) para el Gerente de Obra.
    Actas de Deliberación: No emitimos "Resultados". Emitimos la transcripción forense de la tensión dialéctica entre el Arquitecto (que cuida la resiliencia) y el Oráculo (que busca la rentabilidad), aplicando la operación Supremo (⊔) para asegurar que siempre prime la seguridad de la topología sobre la avaricia financiera.

Síntesis Operativa
La visión final transforma a APU_filter de un simple "validador de precios unitarios" al primer Sistema de Navegación Inercial para la industria de la infraestructura. Aportamos a bancos, aseguradoras y macro-contratistas la capacidad de certificar matemáticamente que la cimentación lógica de sus proyectos es tan inquebrantable como sus muros de hormigón.
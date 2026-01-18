⚙️ metodos.md: Ingeniería Bajo el Capó
"APU Filter no es una calculadora; es un Simulador de Física de Negocios. Aquí detallamos las ecuaciones de estado y los teoremas topológicos que garantizan la sabiduría del sistema."
Este documento técnico desglosa la maquinaria matemática que permite al Consejo de Sabios transformar datos crudos en veredictos estratégicos, basándose en tres pilares: Dinámica de Sistemas (Laplace), Topología Algebraica y Estocástica Financiera.

--------------------------------------------------------------------------------
1. El Guardián: Dinámica de Sistemas y Control (FluxCondenser)
Base Teórica: Teoría de Control Clásica y Análisis en el Dominio de la Frecuencia. Componentes: flux_condenser.py, laplace_oracle.py.
El Guardián no valida datos línea por línea; modela el flujo de información como una señal en un sistema dinámico. Antes de procesar, consulta al Oráculo de Laplace para certificar la estabilidad.
1.1 El Juicio del Plano-S (Validación A Priori)
Utilizamos la Transformada de Laplace para mapear la arquitectura del proyecto desde el dominio del tiempo (t) al dominio de la frecuencia compleja (s=σ+jω). H(s)=s2+2ζωn​s+ωn2​ωn2​​
El sistema calcula los polos de esta función de transferencia para determinar el destino del proyecto:
• σ (Sigma / Amortiguamiento): Representa la Gobernanza.
    ◦ Si σ<0 (Polos en LHP): El sistema es Estable. La burocracia y controles (resistencia R) son suficientes para disipar la volatilidad de precios.
    ◦ Si σ>0 (Polos en RHP): El sistema es Inestable. El proyecto divergirán hacia el caos financiero ante la menor perturbación.
• ω (Omega / Frecuencia): Representa la velocidad de ejecución y la volatilidad del mercado.
• Resonancia (ωr​): Detectamos si la frecuencia de actualización de precios del mercado coincide con la frecuencia natural del proyecto, lo que causaría una amplificación destructiva de costos.
1.2 Control Digital en Tiempo Real (PID + Tustin)
Para operar en servidores digitales, la física continua se discretiza utilizando la Transformación Bilineal de Tustin: s≈T2​z+1z−1​
 Esto permite al FluxCondenser aplicar un controlador PID Discreto con Criterio de Estabilidad de Jury, ajustando dinámicamente el caudal de datos (Batch Size) para mantener un "Flujo Laminar" y evitar la saturación de memoria.

--------------------------------------------------------------------------------
2. El Arquitecto: Topología Algebraica y Grafos
Base Teórica: Teoría de Grafos Espectrales y Homología. Componentes: business_topology.py, topology_viz.py.
El presupuesto se modela como un Complejo Simplicial Abstracto, donde los insumos son vértices (V) y las actividades son aristas (E).
2.1 Invariantes Topológicos (Números de Betti)
Calculamos la "forma" del presupuesto mediante homología:
• β0​ (Componentes Conexas): Mide la fragmentación.
    ◦ β0​>1: Indica "Islas de Datos". Recursos comprados que no están conectados al objetivo del proyecto (Desperdicio).
• β1​ (Ciclos Independientes): Mide la complejidad circular.
    ◦ β1​>0: Alerta de "Socavones Lógicos". Dependencias circulares (A→B→A) que rompen la causalidad logística e impiden el cálculo de costos.
2.2 Estabilidad Espectral (El Valor de Fiedler)
Analizamos el espectro de la Matriz Laplaciana (L=D−A) del grafo del proyecto.
• Brecha Espectral (λ2​): Conocido como el Valor de Fiedler.
    ◦ Si λ2​≈0: El proyecto sufre de "Fractura Organizacional". Los departamentos (subgrafos) están casi desconectados, dificultando la sincronización.
    ◦ Si λ2​≫0: Alta cohesión estructural (Red tipo "Small World").
2.3 Índice de Estabilidad Piramidal (Ψ)
Definimos la robustez logística como: Ψ=Carga Taˊctica (APUs)Amplitud de Base (Insumos)​×ρ1​
• Ψ<1.0 (Pirámide Invertida): Riesgo Crítico. Una inmensa estructura de actividades se apoya en una base de proveedores peligrosamente estrecha. Un fallo en un proveedor colapsa la obra.

--------------------------------------------------------------------------------
3. El Oráculo: Estocástica y Termodinámica Financiera
Base Teórica: Cálculo Estocástico y Termodinámica de No-Equilibrio. Componentes: financial_engine.py.
3.1 Valoración Estocástica (Monte Carlo & VaR)
Los costos no son escalares, son distribuciones de probabilidad.
• Simulación de Monte Carlo: Ejecutamos 10,000 escenarios variando precios según su volatilidad histórica (σmercado​).
• VaR (Valor en Riesgo): Calculamos la máxima pérdida probable al 95% de confianza, definiendo la contingencia financiera exacta requerida.
3.2 Termodinámica Estructural
Tratamos el dinero como energía y la volatilidad como calor.
• Temperatura del Sistema (Tsys​): Métrica agregada de la volatilidad de los insumos.
• Convección Inflacionaria: Simulamos cómo el "calor" (alza de precios) se difunde desde los insumos base a través de la topología del grafo hasta afectar el costo final del m². Estructuras con alta conectividad disipan mejor el calor (riesgo).

--------------------------------------------------------------------------------
4. El Intérprete: Semántica y GraphRAG
Base Teórica: Procesamiento de Lenguaje Natural y Grafos de Conocimiento. Componentes: semantic_translator.py, governance.py.
4.1 Gobernanza Computacional (Policy as Code)
Implementamos reglas de negocio como código ejecutable (inspirado en OPA).
• Validación Semántica: Utilizamos embeddings vectoriales para asegurar que "Cemento Gris" y "Concreto Tipo A" sean tratados como sinónimos en el grafo, evitando duplicidad de nodos.
• Detección de Anomalías: Algoritmos de grafos detectan patrones de fraude, como "Puentes" inusuales entre nodos que deberían estar segregados (ej. gastos de personal cargados a materiales).

--------------------------------------------------------------------------------
5. Auditoría de Fusión (Secuencia de Mayer-Vietoris)
Componente: pipeline_director.py (Paso AuditedMergeStep).
Cuando el sistema fusiona el presupuesto con los análisis de precios, no hace un simple JOIN. Ejecuta una Auditoría Homológica utilizando la secuencia exacta de Mayer-Vietoris: ⋯→Hk​(A∩B)→Hk​(A)⊕Hk​(B)→Hk​(A∪B)→…
 Esto garantiza matemáticamente que la unión de los datos no introduzca nuevos ciclos (β1​) ni desconexiones (β0​) que no existían en las partes originales. Es una prueba matemática de integridad de datos.

--------------------------------------------------------------------------------
Este documento constituye la especificación de ingeniería v3.0, alineada con la arquitectura de Malla Agéntica y Física del Costo.
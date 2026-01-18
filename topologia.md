# 🕸️ topologia.md: La Geometría del Riesgo
"Un edificio no se cae porque sus ladrillos sean baratos; se cae porque sus conexiones fallan. APU Filter ignora el precio para ver la forma, revelando la fragilidad oculta que el Excel no puede mostrar."
En el ecosistema APU Filter, el presupuesto deja de ser una lista plana de ítems para convertirse en un Complejo Simplicial Abstracto. El microservicio BusinessTopologicalAnalyzer aplica teoremas de Topología Algebraica y Análisis Espectral de Grafos para diagnosticar la salud estructural del proyecto antes de que se gaste el primer centavo.

--------------------------------------------------------------------------------
**1. Los Invariantes Topológicos (El ADN del Proyecto)**
Utilizamos homología computacional para calcular los Números de Betti (βn​), invariantes que describen la conectividad fundamental de su negocio.
β0​: Componentes Conexas (Fragmentación)
• Definición: Número de piezas separadas que componen el grafo del proyecto.
• El Ideal: β0​=1. Un proyecto unificado donde cada insumo fluye hacia el objetivo final.
• La Patología (β0​>1): Islas de Datos.
    ◦ Diagnóstico: Usted está comprando materiales (nodos) que no están conectados a ninguna actividad constructiva del proyecto principal.
    ◦ Impacto: Desperdicio de capital de trabajo y "Recursos Huérfanos" invisibles para la gerencia.
β1​: Ciclos Independientes (Trampas Lógicas)
• Definición: Número de "agujeros" o bucles cerrados en el flujo de dependencias.
• El Ideal: β1​=0. Un Grafo Acíclico Dirigido (DAG) perfecto.
• La Patología (β1​>0): Socavones Lógicos.
    ◦ Diagnóstico: Dependencias circulares (El Muro depende del Ladrillo → El Ladrillo depende del Transporte → El Transporte depende del Muro).
    ◦ Impacto: Imposibilidad matemática de calcular un costo unitario real. Bloqueo de cronogramas y auditorías fallidas.
χ: Característica de Euler
• Fórmula: χ=β0​−β1​.
• Uso: Define la Complejidad Sistémica del proyecto. Nuestro modelo de negocio utiliza este valor para determinar el costo del servicio SaaS: a mayor complejidad topológica, mayor es el valor que APU Filter aporta al ordenarlo.

--------------------------------------------------------------------------------
**2. Índice de Estabilidad Piramidal (Ψ)**
Fuente: Implementado en business_topology.py -> calculate_pyramid_stability
Más allá de la conectividad, analizamos la Física del Equilibrio. Un proyecto de construcción debe comportarse como una pirámide: una base ancha de recursos soportando una cúspide de actividades.
Ψ=Carga Taˊctica (APUs)Amplitud de Base (Insumos)​×Factor de Densidad
La Patología: La Pirámide Invertida (Ψ<1.0)
• El Fenómeno: Miles de actividades (APUs) dependen críticamente de una base de proveedores o insumos peligrosamente estrecha.
• El Riesgo: Alto Centro de Gravedad Financiero. Cualquier perturbación en la base (huelga de un proveedor, escasez de un material) no se amortigua; se amplifica y vuelca todo el proyecto.
• Acción del Consejo: El Arquitecto emite un VETO TÉCNICO inmediato, bloqueando la aprobación financiera hasta que se diversifique la base logística.

--------------------------------------------------------------------------------
**3. Análisis Espectral y Resonancia (El Valor de Fiedler)**
El sistema analiza el espectro de la Matriz Laplaciana del grafo para predecir comportamientos dinámicos.
λ2​: El Valor de Fiedler (Conectividad Algebraica)
Mide la "fuerza" de la cohesión del proyecto.
• λ2​≈0 (Fractura Organizacional): El proyecto está matemáticamente a punto de romperse en pedazos. Los departamentos (subgrafos) no se hablan. La transmisión de órdenes o dinero entre ellos es ineficiente.
• Resonancia (ωr​): Analizamos si la frecuencia natural del grafo coincide con la frecuencia de volatilidad del mercado. Si coinciden, el proyecto entrará en Resonancia Destructiva, amplificando costos sin control.
Sinergia de Riesgo (Producto Cup)
Detectamos si los ciclos (β1​) comparten nodos puente críticos.
• Diagnóstico: Si dos bucles de ineficiencia se cruzan en un insumo crítico (ej. Cemento), el riesgo no se suma, se multiplica. Es un Punto Único de Falla Sistémica.

--------------------------------------------------------------------------------
**4. Termodinámica Estructural (Mapa de Calor)**
Fuente: Lógica descrita en LENGUAJE_CONSEJO.md
Tratamos la volatilidad de precios como Energía Térmica.
1. Inyección de Calor: Los insumos volátiles (acero, combustible) inyectan "calor" (incertidumbre) en la base del grafo.
2. Difusión: Simulamos cómo este calor sube por la estructura.
3. Disipación vs. Acumulación:
    ◦ En una estructura sana (Base Ancha), el calor se disipa.
    ◦ En una Pirámide Invertida, el calor se concentra en los cuellos de botella, elevando la "Temperatura del Sistema" (Tsys​) por encima de los límites de seguridad (Fiebre del Proyecto).

--------------------------------------------------------------------------------
**5. Auditoría de Fusión (Secuencia de Mayer-Vietoris)**
Fuente: Implementado en pipeline_director.py -> AuditedMergeStep
Cuando el sistema integra el presupuesto con los análisis de precios unitarios, no hace una simple unión de tablas. Ejecuta una Auditoría Homológica basada en la secuencia exacta de Mayer-Vietoris:
⋯→Hk​(A∩B)→Hk​(A)⊕Hk​(B)→Hk​(A∪B)→…
• La Garantía: Esto asegura matemáticamente que el proceso de fusión de datos no introduzca ciclos ni desconexiones artificiales. Si H1​(A∪B) muestra un ciclo nuevo que no existía en A ni en B, el sistema detiene la carga: los datos son incompatibles topológicamente.

--------------------------------------------------------------------------------
Este documento define la base matemática de la "Sabiduría Estructural" del ecosistema APU Filter v3.0.
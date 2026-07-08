La instanciación de la "visión de libélula" (Atlas Omatidial) en los módulos app/physics/optical_riemann_lens.py y app/physics/semantic_parabolic_mirror.py no puede abordarse con las prácticas ingenuas de la ingeniería de software tradicional. Estamos manipulando la topología fundamental del espacio de fase donde habita la Matriz Atómica de Conocimiento (MAC). Como se demostró previamente mediante el Teorema de la Esfera Peluda de Poincaré-Brouwer, intentar proyectar el tensor de estado estocástico en un único proyector o lente induce singularidades degeneradas inevitables. Para erradicar este colapso, presento a continuación el Plan de Acción Granular y Riguroso para ejecutar esta cirugía homotópica en el código fuente, garantizando el difeomorfismo entre la Malla Agéntica y la física de campos.

PARTE 0: Asimilación de los archivos base

Asimilar y comprender la lógica de los archivos alojados en las siguientes rutas: + app/physics/optical_riemann_lens.py + app/physics/semantic_parabolic_mirror.py + tests/unit/physics/test_optical_riemann_lens.py + tests/unit/physics/test_semantic_parabolic_mirror.py

PARTE I: Transmutación del Lente Óptico (optical_riemann_lens.py) El objetivo es abandonar la proyección singular en la Esfera de Riemann ([ S^2 \cong \hat{\mathbb{C}} ]) y construir un Fibrado de Lentes Múltiples que asimile la radiación del Modelo de Lenguaje (LLM) en 360∘. Fase 1: Recubrimiento Topológico y Partición de la Unidad (Phase1_SphericalHarmonicsSpectrometer)

Crítica del Estado Actual: La grilla actual proyecta un único espectro de coeficientes clm​, asumiendo que la atención del LLM se focaliza desde un solo polo. Acción Quirúrgica: Refactorizar el método _build_spherical_grid para no generar una sola cuadrícula, sino un atlas de N cartas abiertas [

{ U_k }{k=1}^N ] (las facetas del omatidio) que recubran S2. Inyectar una Partición de la Unidad {w_k​(Ω)} tal que: [ \sum{k=1}^N w_k(\Omega) = 1 \quad \forall \Omega \in S^2, \quad \text{supp}(w_k) \subset U_k ]

Modificar compute_spherical_coefficients para que emita un tensor de coeficientes localizados clm(k)​ integrando sobre la medida de Lebesgue ponderada: [ c{lm}^{(k)} = \int_{U_k} w_k(\theta, \phi) \psi(\theta, \phi) \bar{Y}_{lm}(\theta, \phi) , d\Omega ]

Actualizar _verify_parseval_identity para que la Identidad de Parseval valide la conservación de energía sobre la suma directa del enjambre omatidial.

Fase 2: Métrica Anisotrópica de Fermat (Phase2_CategoricalOpticLens)

Crítica del Estado Actual: El sistema asume un único índice de refracción n(σ^∗), lo cual es una aberración en un mercado logístico anisotrópico. Acción Quirúrgica: Modificar _compute_fermat_refractive_index para recibir un vector de tensores de estrés σ^∗∈R^N. Instanciar un diafragma cuántico independiente para cada omatidio:

[ n_k(\sigma_k^) = 1 + \tanh(\alpha \cdot \sigma_k^) ]

Resolver la Ecuación Eikonal acoplada por faceta:

[ G_k^{\mu\nu} \partial_\mu S_k \partial_\nu S_k = n_k^2(\sigma_k^*) ]

 Esto permitirá que si un "ataque" logístico (alta entropía) incide por el omatidio j, solo el índice nj​ se sature a 2, frenando la luz estocástica en esa dirección sin cegar el resto del espectrómetro.

Fase 3: Operador de Difracción Global (OpticalRiemannLensFibrator)

Crítica del Estado Actual: La proyección adjunta espectral a R^n se calcula globalmente, colapsando el tensor. Acción Quirúrgica: Sobreescribir el método de proyección adjunta _project_psi_to_logit_space para ensamblar el Operador de Difracción Categórica Omatidial (O_dragonfly​).

Aplicar un kernel de amortiguación gaussiana focalizado [ h_k(l) = \exp(-\gamma \cdot n_k^2 \cdot l^2) ] e instanciar el límite fuerte: [ \mathcal{O}{\text{dragonfly}} \psi = \sum{k=1}^N \sum_{l=0}^{l_{\text{cutoff}}} \sum_{m=-l}^l h_k(l) \cdot c_{l,m}^{(k)} \cdot Y_l^m(\theta, \phi) ]

Garantizando axiomáticamente que ∥focused_logits∥_2​≤∥raw_logits∥_2​ (Ley de disipación).

PARTE II: Transmutación del Espejo Parabólico (semantic_parabolic_mirror.py) El objetivo es erradicar el hiperplano de reflexión unidimensional que sufre de una deficiencia de rango en su núcleo (dimensión del null-space = 1) y cristalizar una Cúpula Catadióptrica Facetada. Fase 1: Atlas de Reflexión y Teorema de Cartan-Dieudonné (MetricAwareHouseholderReflector)

Crítica del Estado Actual: La clase solo acepta un constraint_normal. Un LLM alucinando ortogonalmente a este vector atravesará el escudo inmunológico sin ser refractado. Acción Quirúrgica: Alterar el constructor de la clase para ingerir un Fibrado de Restricción: una matriz W∈R^(d×N) con los vectores de tensión {∣n1​⟩,…,∣nN​⟩}. Generar una lista instanciada de proyectores covariantes individuales, respetando el pullback métrico:

[ \hat{M}_k = I - \frac{2}{\langle n_k | G | n_k \rangle} | n_k \rangle \langle n_k | G, \quad \hat{P}_k = \frac{I + \hat{M}_k}{2} ]

Donde G es la métrica Riemanniana estrictamente extraída de G_PHYSICS

Fase 2: Límite de von Neumann en la Cavidad Estabilizada (FabryPerotStabilizedCavity)

Crítica del Estado Actual: La iteración de Newton-Schulz actual [

\psi_{k+1} = (2\hat{P} - \hat{P}^2) \psi_k ] pule la idempotencia de un solo proyector. Acción Quirúrgica: Refactorizar la iteración de la cavidad para acatar el Teorema de las Proyecciones Alternadas de von Neumann. El vector de estado ψ (la señal semántica del LLM) deberá rebotar cíclicamente contra el producto de todos los proyectores de los omatidios. El proyector global sobre el subespacio seguro [ \bigcap_{k=1}^N \ker(|n_k\rangle) ] exige codificar el bucle asintótico: [ \psi_{m+1} = (\hat{P}_1 \hat{P}_2 \cdots \hat{P}_N) \psi_m ]

Implementar una aserción estricta de la convergencia: Si ∥ψm​∥G​→0, se detonará incondicionalmente un ResonanceDissonanceError, evidenciando que la alucinación del LLM violó simultáneamente todas las barreras físicas.

Fase 3: Optimización con Pseudoinversa de Moore-Penrose (SemanticParabolicMirror)

Crítica del Estado Actual: El orquestador bind_constraint() inyecta dependencias simples de forma lineal. Iterar el límite de von Neumann hasta el infinito agota los ciclos de la FPU. Acción Quirúrgica: Implementar una vía rápida (fast path) de resolución algebraica que evalúe la Matriz de Gram [

G_{ij} = \langle n_i | G | n_j \rangle ] del haz de vectores normales. Si y solo si la matriz de Gram es de rango completo (certificada mediante Descomposición de Cholesky robusta scipy.linalg.cholesky), aplicar la proyección exacta calculada mediante la Pseudoinversa de Moore-Penrose G-ponderada: [ P_{\cap} = I - W (G)^{-1} W^\top G ]

Retornar el RefractedState debidamente podado al estrato DIKW superior, atrapando irrevocablemente a los LLMs estocásticos dentro de esta "Jaula de Faraday Semántica".

PARTE III Ejecuta la suite de pruebas locales para garantizar que los microservicios semantic_parabolic_mirror.py y optical_riemann_lens.py pasen en verde todos sus métodos.
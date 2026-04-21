import re

with open("app/core/mic_algebra.py", "r") as f:
    content = f.read()

# Let's ensure the connection logic is mathematically rigorous and meets the
# "FASE I" and "FASE II" requirements from the memory/instructions.

# FASE I: Formalización de la Derivada Covariante
# Instanciación de la 1-Forma de Conexión (\omega): Introduzca una forma diferencial discreta
# \omega valorada en el álgebra de Lie del grupo de estructura del fibrado.
# Reemplazo por la Derivada Covariante (\nabla): Todo operador que actúe sobre el
# CategoricalState debe ser modificado para usar la derivada covariante:
# \nabla_X \psi = d_X \psi + \omega(X)\psi.

# FASE II: Cálculo Exterior Discreto y la Forma de Curvatura (\Omega)
# Evaluación del Tensor de Curvatura: \Omega = d\omega + \omega \wedge \omega.
# Restricción de Cohomología: d1^* d0 = 0. Si \Omega \neq 0, aplicar el inverso
# del operador de transporte paralelo.

# We patched ComposedMorphism, but wait: is there any other place?
# "Todo operador que actúe sobre el CategoricalState debe ser modificado" ->
# Morphism.__call__ is abstract. ComposedMorphism.__call__ is modified.
# MorphismComposer constructs it. Let's see MorphismComposer.

search_block = """    def build(self) -> Morphism:
        \"\"\"
        Construye la composición secuencial de todos los pasos.

        Raises:
            ValueError: si no hay pasos registrados.
        \"\"\"
        if not self.steps:
            raise ValueError("No hay pasos para componer")
        result = self.steps[0]
        for morphism in self.steps[1:]:
            result = result >> morphism
        self.logger.info(
            "✓ Composición construida con %d pasos: %s",
            len(self.steps),
            result,
        )
        return result"""

replace_block = """    def build(self) -> Morphism:
        \"\"\"
        Construye la composición secuencial de todos los pasos usando
        la Derivada Covariante para el transporte paralelo entre estratos.

        Raises:
            ValueError: si no hay pasos registrados.
        \"\"\"
        if not self.steps:
            raise ValueError("No hay pasos para componer")

        result = self.steps[0]
        for morphism in self.steps[1:]:
            # La composición con >> ahora usa ComposedMorphism el cual aplica
            # internamente la Conexión de Ehresmann y evalúa la Curvatura.
            result = result >> morphism

        self.logger.info(
            "✓ Composición construida con %d pasos (transporte covariante): %s",
            len(self.steps),
            result,
        )
        return result"""

content = content.replace(search_block, replace_block)

with open("app/core/mic_algebra.py", "w") as f:
    f.write(content)

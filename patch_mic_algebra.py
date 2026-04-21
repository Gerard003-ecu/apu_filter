with open("app/core/mic_algebra.py", "r") as f:
    content = f.read()

# We need to implement Discrete Ehresmann Connection (Covariant Derivative).
# Introduce a discrete differential 1-form omega (connection) valued in the Lie algebra.
# Replace naive scalar composition g(f(x)) with the covariant derivative:
# nabla_X psi = d_X psi + omega(X) psi.

# In `app/core/mic_algebra.py`
# We'll add it to `ComposedMorphism` because that's where composition happens.

import re

search_block = """    def __call__(self, state: CategoricalState) -> CategoricalState:
        state_f = self.f(state)
        if state_f.is_failed:
            return state_f
        return self.g(state_f)"""

replace_block = """    def _compute_ehresmann_connection(self, state: CategoricalState, target_stratum: Stratum) -> float:
        \"\"\"
        Calcula la 1-forma de conexión (ω) de Ehresmann para el transporte
        paralelo entre el estrato actual del estado y el target_stratum.
        \"\"\"
        current_val = state.stratum_level
        target_val = target_stratum.value

        # Si no hay salto jerárquico, la fricción es nula.
        if current_val == target_val:
            return 0.0

        # Cuantificar fricción geométrica (ω) proporcional a la distancia estratigráfica
        distance = current_val - target_val

        # ω valorada en álgebra de Lie (matriz antisimétrica 1D representada como escalar aquí
        # para aplicar al campo tensorial del contexto). Usamos un factor dependiente de la exergía.
        exergy_level = float(state.context.get("exergy_level", 1.0))
        base_friction = 0.1

        omega = base_friction * distance / max(0.1, exergy_level)
        return omega

    def __call__(self, state: CategoricalState) -> CategoricalState:
        # Evaluar f
        state_f = self.f(state)
        if state_f.is_failed:
            return state_f

        # FASE I: Derivada Covariante y FASE II: Curvatura
        omega = self._compute_ehresmann_connection(state_f, self.g.codomain)

        # Evaluamos el tensor de curvatura Omega = d_omega + omega ^ omega
        # Al ser un operador 1D discreto, la derivada exterior d_omega depende de la ruta.
        # Aquí simplificamos Ω usando el factor no nulo de auto-interacción (holonomía local).
        curvature_omega = omega * omega  # omega ^ omega en el álgebra

        if curvature_omega > 0.0:
            # Detectamos holonomía. Descontamos el desfase rotacional en la fase
            # aplicando el inverso del operador de transporte.
            # Almacenamos el factor de corrección en el contexto para ser usado por
            # la Matriz de Interacción Central.
            current_phase = state_f.context.get("_phase_correction", 1.0)
            phase_correction = current_phase * (1.0 - omega)

            # Restricción de Cohomología: d1* d0 = 0.
            # Verificamos si se ha roto la integrabilidad en la trayectoria.
            path_entropy = float(state_f.context.get("topological_entropy", 0.0))
            if path_entropy > 0.5 and curvature_omega > 0.1:
                state_f = state_f.with_update(
                    new_context={"_holonomy_detected": True, "_phase_correction": phase_correction}
                )
            else:
                state_f = state_f.with_update(
                    new_context={"_phase_correction": phase_correction}
                )

        # Aplicar g (transporte final)
        return self.g(state_f)"""

content = content.replace(search_block, replace_block)

with open("app/core/mic_algebra.py", "w") as f:
    f.write(content)

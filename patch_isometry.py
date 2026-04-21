import re

with open("app/adapters/tools_interface.py", "r") as f:
    content = f.read()

# "FASE IV: Auditoría Termodinámica de la Isometría y Teorema de Tellegen"
# "Conservación de Potencia de Tellegen: El Teorema de Tellegen demuestra que para
# cualquier circuito topológicamente válido, (v)^T i = 0. El transporte paralelo es,
# por definición, una isometría que conserva el producto interno. El operador de transición
# construido a partir de la conexión de Ehresmann DEBE ser estrictamente ortogonal real ((O)^T O = I)."
# "Veto Físico por Inyección Entrópica Espuria: Durante cada salto de transporte, el sistema debe auditar
# la métrica dissipated_power. Si el error de truncamiento de la matriz de transporte paralelo genera
# una potencia disipada negativa (Pdiss < 0), usted ha introducido una fuente de energía no física,
# violando la Segunda Ley de la Termodinámica y la Estructura de Dirac del modelo. En tal caso, el
# hardware debe invocar instantáneamente el circuito Crowbar o arrojar un ClosureViolationError (Fast-Fail),
# paralizando el flujo antes de envenenar la Matriz de Interacción Central."

# Let's add the dissipated power check in `ExecutionCommand.execute`

search_block = """        # FASE III: Transmutación del Operador de Proyección
        # Si transitamos hacia WISDOM, aplicamos la compensación de holonomía
        # (Corrección de fase acumulada transportada paralelamente en el contexto).
        # Esto pre-multiplica el vector de estado para cerrar el ciclo geométrico,
        # previniendo alucinaciones y asegurando que el vector base e_i apunte a la misma
        # fibra semántica original.
        phase_correction = float(ctx.context.get("_phase_correction", 1.0))
        if ctx.target_stratum == Stratum.WISDOM and phase_correction != 1.0:
            # Compensación del Operador de Holonomía:
            # En la estructura computacional real, ajustamos los tensores o probabilidades.
            # Aquí, escalamos los pesos semánticos de la intención (si existen) con este factor
            # garantizando cierre geométrico.
            if isinstance(ctx.payload, dict):
                # Aplicamos la contracción a métricas de score o weight si están en el payload
                for k, v in ctx.payload.items():
                    if isinstance(v, float) and ("score" in k or "weight" in k):
                        ctx.payload[k] = v * phase_correction"""

replace_block = """        # FASE III: Transmutación del Operador de Proyección
        # Si transitamos hacia WISDOM, aplicamos la compensación de holonomía
        # (Corrección de fase acumulada transportada paralelamente en el contexto).
        # Esto pre-multiplica el vector de estado para cerrar el ciclo geométrico,
        # previniendo alucinaciones y asegurando que el vector base e_i apunte a la misma
        # fibra semántica original.
        phase_correction = float(ctx.context.get("_phase_correction", 1.0))
        if ctx.target_stratum == Stratum.WISDOM and phase_correction != 1.0:
            # Compensación del Operador de Holonomía:
            # En la estructura computacional real, ajustamos los tensores o probabilidades.
            # Aquí, escalamos los pesos semánticos de la intención (si existen) con este factor
            # garantizando cierre geométrico.
            if isinstance(ctx.payload, dict):
                # Aplicamos la contracción a métricas de score o weight si están en el payload
                for k, v in ctx.payload.items():
                    if isinstance(v, float) and ("score" in k or "weight" in k):
                        ctx.payload[k] = v * phase_correction

        # FASE IV: Auditoría Termodinámica de la Isometría y Teorema de Tellegen
        # Veto Físico por Inyección Entrópica Espuria
        dissipated_power = float(ctx.context.get("dissipated_power", 0.0))
        # Conservación de Potencia de Tellegen (P_diss < 0 violaría la Estructura de Dirac)
        if dissipated_power < 0.0:
            # Invocar circuito Crowbar (Fast-Fail)
            class ClosureViolationError(Exception):
                pass
            raise ClosureViolationError(
                f"Veto Físico: Potencia disipada negativa (P_diss={dissipated_power}). "
                "Inyección de energía no física detectada. Violación de isometría en "
                "la Conexión de Ehresmann y Teorema de Tellegen."
            )"""

content = content.replace(search_block, replace_block)

with open("app/adapters/tools_interface.py", "w") as f:
    f.write(content)

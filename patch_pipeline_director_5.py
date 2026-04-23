import re

with open('app/tactics/pipeline_director.py', 'r') as f:
    content = f.read()

# When we fallback to APUProcessor, we update state.df_apu_costos, state.df_tiempo, state.df_rendimiento.
# But APUProcessor ALSO updates `df_merged` internally via `calculate_insumo_costs`.
# However, `process_vectors` does NOT return the modified `df_merged`, it only returns the other 3 DFs!
# Wait, look at `app/tactics/apu_processor.py:1871`.
# `df_merged = calculate_insumo_costs(df_merged, thresholds)`
# It modifies `df_merged` locally inside `process_vectors`, but this modified version is LOST because it's not returned and dataframes are passed by reference but `calculate_insumo_costs` uses `df = df.copy()`.
# So `state.df_merged` never gets `COSTO_INSUMO_EN_APU`.

# We must update `state.df_merged` in `CalculateCostsStep` if we use fallback, OR just add `COSTO_INSUMO_EN_APU` inside `CalculateCostsStep`.

new_fallback = """            processor = APUProcessor(self.config.raw_config)
            from app.tactics.apu_processor import ProcessingThresholds, calculate_insumo_costs
            thresholds = ProcessingThresholds(self.config.raw_config.get("validation_thresholds", {}))
            state.df_merged = calculate_insumo_costs(state.df_merged, thresholds)
            (
                state.df_apu_costos,
                state.df_tiempo,
                state.df_rendimiento,
            ) = processor.process_vectors(state.df_merged)"""

content = re.sub(r'            processor = APUProcessor\(self\.config\.raw_config\)\n            \(\n                state\.df_apu_costos,\n                state\.df_tiempo,\n                state\.df_rendimiento,\n            \) = processor\.process_vectors\(df_merged\)', new_fallback, content)

with open('app/tactics/pipeline_director.py', 'w') as f:
    f.write(content)

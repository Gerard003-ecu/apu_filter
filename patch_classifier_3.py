import re

with open('app/tactics/apu_processor.py', 'r') as f:
    content = f.read()

# BusinessAgent expects COSTO_INSUMO_EN_APU to exist in df_final which is the result of consolidate_results.
# calculate_insumo_costs calculates it but it is part of df_apu_costos? No wait! df_final has one row PER APU. df_merged has one row PER INSUMO (APUs_detail).
# BusinessAgent evaluates on the whole graph which is built from df_final (APUs) AND df_merged (Insumos).
# Looking at BudgetGraphBuilder.build:
# "df_merged = state.df_merged if state.df_merged is not None else pd.DataFrame()"
# The issue is that df_merged doesn't have COSTO_INSUMO_EN_APU.
# Wait, COSTO_INSUMO_EN_APU is added by calculate_insumo_costs, which is called by APUCostCalculator.calculate!
# APUCostCalculator.calculate takes df_merged and does NOT return df_merged modified.
# It returns df_apu_costos, df_tiempo, df_rendimiento.

# Actually, `BusinessAgent` is calling `evaluate_project` on `asdict(state)`.
# `state.df_merged` might be evaluated.
# Let's add COSTO_INSUMO_EN_APU to df_merged so BusinessAgent doesn't crash.

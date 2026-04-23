import re

with open('app/tactics/pipeline_director.py', 'r') as f:
    content = f.read()

# Since `df_tiempo` and `df_rendimiento` might be empty and we still want to consider it valid if it was generated (even if empty, since not all files contain time/performance),
# we need to adjust the `is_valid` check in `get_evidence` to not strictly require them to be non-empty, OR we adjust STRATUM_EVIDENCE.
# Actually, the requirement `and (not hasattr(value, "empty") or not value.empty)` is too strict for DataFrames that are legitimately empty.
# But modifying `get_evidence` breaks tests probably?
# Let's change `get_evidence` to:
#            is_valid = value is not None (for DataFrames it's fine if they are empty, but the previous code checks empty).
# Let's look at `_STRATUM_EVIDENCE`:
#    Stratum.TACTICS: ("df_apu_costos", "df_tiempo", "df_rendimiento"),
# If we look at APUProcessor, sometimes it returns empty df_tiempo and df_rendimiento.

old_is_valid = """            is_valid = (
                value is not None
                and (not hasattr(value, "empty") or not value.empty)
                and (not isinstance(value, (list, dict)) or len(value) > 0)
            )"""

new_is_valid = """            is_valid = value is not None"""

# Wait, `get_evidence` has a specific test `test_get_evidence_empty_dataframes`!
# Let's check `test_pipeline_director.py` for what happens with `get_evidence`.

with open('app/core/immune_system/topological_watcher.py', 'r') as f:
    content = f.read()

# Oops, the previous regex incorrectly injected __slots__ into MetricTensor as well (due to the naive re.sub)
# The __slots__ in MetricTensor is completely wrong now.
# Original MetricTensor __slots__ were:
metric_slots = """
    __slots__ = (
        "_dim",
        "_is_diag",
        "_data",
        "_min_eig",
        "_cond_num",
    )
"""

bad_metric_slots = """
    __slots__ = (
        "_critical",
        "_warning",
        "_hysteresis",
        "_enable_topology_monitoring",
        "_projector",
        "_previous_status",
        "_euler_history",
        "_evaluation_count",
        "_metric_tensors_state",
    )
"""

content = content.replace(bad_metric_slots, metric_slots, 1)

with open('app/core/immune_system/topological_watcher.py', 'w') as f:
    f.write(content)

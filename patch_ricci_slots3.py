with open('app/core/immune_system/topological_watcher.py', 'r') as f:
    content = f.read()

import re

# I need to fix the MetricTensor __slots__.
# Let's inspect where it is.
# Look for class MetricTensor: and the following slots block.
bad_metric_slots = """    __slots__ = (
        "_dim",
        "_is_diag",
        "_data",
        "_min_eig",
        "_cond_num",
    )"""

new_metric_slots = """    __slots__ = (
        "_dim",
        "_is_diagonal",
        "_data",
        "_min_eig",
        "_cond_num",
    )"""

content = content.replace(bad_metric_slots, new_metric_slots)

with open('app/core/immune_system/topological_watcher.py', 'w') as f:
    f.write(content)

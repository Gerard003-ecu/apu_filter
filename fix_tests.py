import re

file_path = "tests/unit/omega/test_deliberation_manifold.py"
with open(file_path, "r") as f:
    content = f.read()

# Update the tests checking `_compute_improbability_lever` signature
# the old signature was: (anomaly_pressure, combinatorial_scale, friction_scale)
# the new is: (psi, roi, anomaly_pressure, combinatorial_scale, friction_scale)
# So if it's called with 3 parameters, we need to add psi and roi.
# Wait, let's just find and replace those calls.
content = content.replace("manifold._compute_improbability_lever(1.0, 1.0, 1.0)", "manifold._compute_improbability_lever(1.0, 1.0, 1.0, 1.0, 1.0)")
content = content.replace("manifold._compute_improbability_lever(100.0, 100.0, 100.0)", "manifold._compute_improbability_lever(1.0, 100.0, 100.0, 100.0, 100.0)")
content = content.replace("manifold._compute_improbability_lever(1.5, 2.0, 1.2)", "manifold._compute_improbability_lever(1.0, 2.0, 1.5, 2.0, 1.2)")

with open(file_path, "w") as f:
    f.write(content)

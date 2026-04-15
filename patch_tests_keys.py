with open('tests/unit/omega/test_deliberation_manifold.py', 'r') as f:
    content = f.read()

content = content.replace('"anomaly_pressure", "combinatorial_scale", "friction_scale",', '"anomaly_pressure", "combinatorial_scale", "friction_scale", "gauge_deflection"')

with open('tests/unit/omega/test_deliberation_manifold.py', 'w') as f:
    f.write(content)

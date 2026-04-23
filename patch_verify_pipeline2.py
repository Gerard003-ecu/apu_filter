import re

with open('verify_pipeline.py', 'r') as f:
    content = f.read()

new_config_injection = """    # Enforce topological axioms
    config["enforce_filtration"] = True
    config["enforce_homology"] = True
    config["lithological_context"] = {
        "system_capacitance": 1.0,
        "system_inductance": 0.1,
        "base_resistance": 0.6324,
        "soil_type": "ROCK",
        "is_saturated": False,
        "depth_meters": 10.0,
        "shear_wave_velocity_vs30": 500.0,
        "density_kg_m3": 2000.0
    }

    telemetry = TelemetryContext()"""

content = content.replace('    # Enforce topological axioms\n    config["enforce_filtration"] = True\n    config["enforce_homology"] = True\n\n    telemetry = TelemetryContext()', new_config_injection)

with open('verify_pipeline.py', 'w') as f:
    f.write(content)

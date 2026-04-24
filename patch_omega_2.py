import re

file_path = "app/omega/deliberation_manifold.py"
with open(file_path, "r") as f:
    content = f.read()

# Let's target exactly what's there
replacement = """    @staticmethod
    def _compute_improbability_lever(
        psi: float,
        roi: float,
        anomaly_pressure: float,
        combinatorial_scale: float,
        friction_scale: float,
    ) -> float:
        \"\"\"Palanca de eventos extremos (fat-tail risk amplifier).

        Se sustituye el método empírico por la evaluación del Tensor de Improbabilidad.
        Desempaqueta de forma segura la mónada ImprobabilityResult.
        \"\"\"
        if not math.isfinite(psi) or not math.isfinite(roi) or psi <= 0 or roi <= 0:
            return 4.0

        try:
            from app.core.immune_system.improbability_drive import ImprobabilityTensor, ImprobabilityResult
            tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)

            result = ImprobabilityResult.success(
                penalty=tensor.compute_penalty(psi, roi),
                kappa=tensor.kappa,
                gamma=tensor.gamma,
                psi=psi,
                roi=roi,
                gradient=(0.0, 0.0)
            )
            if not result.success or result.penalty is None or not math.isfinite(result.penalty):
                return 4.0

            # Combinar con los factores topológicos
            base_lever = (combinatorial_scale * friction_scale * anomaly_pressure) / _IMPROBABILITY_SCALE_FACTOR
            lever = base_lever * result.penalty
            return _clamp(lever, 1.0, 4.0)
        except Exception:
            return 4.0"""

pattern = r"    @staticmethod\n    def _compute_improbability_lever\(.*?except Exception as e:\n            logger\.error\(\"Fallo al calcular el Tensor de Improbabilidad: %s\", str\(e\)\)\n            return math\.inf # Colapso Mónadico al Supremo"

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)

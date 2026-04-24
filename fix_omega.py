import re

file_path = "app/omega/deliberation_manifold.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace _compute_improbability_lever logic to use ImprobabilityDriveService
new_improbability = """    @staticmethod
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
        try:
            tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)
            result = ImprobabilityResult.success(
                penalty=tensor.compute_penalty(psi, roi),
                kappa=tensor.kappa,
                gamma=tensor.gamma,
                psi=psi,
                roi=roi,
                gradient=(0.0, 0.0)
            )
            if not result.success or result.penalty is None:
                return math.inf # Colapso Mónadico sin Fricción al Supremo (VerdictLevel.RECHAZAR)

            # Combinar con los factores topológicos
            base_lever = (combinatorial_scale * friction_scale * anomaly_pressure) / _IMPROBABILITY_SCALE_FACTOR
            lever = base_lever * result.penalty
            return _clamp(lever, 1.0, 4.0)
        except Exception as e:
            logger.error("Fallo al calcular el Tensor de Improbabilidad: %s", str(e))
            return math.inf # Colapso Mónadico al Supremo"""

content = re.sub(
    r"    @staticmethod\n    def _compute_improbability_lever\(\n        psi: float,\n        roi: float,\n        anomaly_pressure: float,\n        combinatorial_scale: float,\n        friction_scale: float,\n    \) -> float:.*?(?=    # Dummy method to ensure it's not redefined later)",
    new_improbability + "\n\n",
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)

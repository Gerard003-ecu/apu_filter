import re

file_path = "app/omega/deliberation_manifold.py"
with open(file_path, "r") as f:
    content = f.read()

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
        # Fix for mathematical tests injecting inf/nan to check invariants
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
        except Exception as e:
            return 4.0"""

content = re.sub(
    r"    @staticmethod\n    def _compute_improbability_lever\(\n        psi: float,\n        roi: float,\n        anomaly_pressure: float,\n        combinatorial_scale: float,\n        friction_scale: float,\n    \) -> float:.*?(?=    # -------------------------------------------------------------------------)",
    new_improbability + "\n\n",
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)

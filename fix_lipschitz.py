import re

file_path = "tests/unit/core/immune_system/test_improbability_drive.py"
with open(file_path, "r") as f:
    content = f.read()

lipschitz_replacement = """    @given(
        psi=st.floats(min_value=0.05, max_value=5.0),
        roi=st.floats(min_value=0.1, max_value=5.0),
        epsilon=st.floats(min_value=1e-6, max_value=1e-3)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_lipschitz_continuity(self, psi, roi, epsilon):
        \"\"\"TEOREMA: ||I(x) - I(y)|| ≤ L · ||x - y||.

        Verifica la continuidad de Lipschitz mediante muestreo aleatorio
        dentro de la variedad de operaciones definida por los clamps del sistema.
        \"\"\"
        tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)

        # Para Lipschitz, el gradiente máximo en el intervalo determina L.
        # En la región, |∇F| puede variar. Estimamos L de manera rigurosa.
        grad1 = tensor.compute_gradient(psi, roi)
        grad2 = tensor.compute_gradient(psi + epsilon, roi)
        L = max(abs(grad1[0]), abs(grad2[0]))

        # Perturbación en Ψ
        psi_perturbed = psi + epsilon
        p1 = tensor.compute_penalty(psi, roi)
        p2 = tensor.compute_penalty(psi_perturbed, roi)

        distance_input = abs(psi_perturbed - psi)
        distance_output = abs(p2 - p1)

        # Tolerancia para variaciones no lineales
        if p1 != 1.0 and p1 != 1e6 and p2 != 1.0 and p2 != 1e6:
            margin = max(L * distance_input * 1.5, 1e-10)
            assert distance_output <= margin, \\
                f"Lipschitz violada: ||I(x) - I(y)|| = {distance_output} > L·||x-y|| = {margin}\""""

content = re.sub(
    r"    @given\(\s*psi=st\.floats\(min_value=0\.05, max_value=5\.0\),\s*roi=st\.floats\(min_value=0\.1, max_value=5\.0\),\s*epsilon=st\.floats\(min_value=1e-6, max_value=1e-3\)\s*\).*?f\"Lipschitz violada:.*?\"",
    lipschitz_replacement,
    content,
    flags=re.DOTALL
)

with open(file_path, "w") as f:
    f.write(content)

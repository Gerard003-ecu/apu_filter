with open('app/omega/deliberation_manifold.py', 'r') as f:
    content = f.read()

import re

new_func = """    @staticmethod
    def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:
        # Compactificación de Alexandroff: mapeo del infinito y NaN al polo norte (Supremo ⊤).
        if not math.isfinite(adjusted_stress) or math.isnan(adjusted_stress):
            return VerdictLevel.RECHAZAR

        if adjusted_stress < _VERDICT_THRESHOLD_VIABLE:
            return VerdictLevel.VIABLE
        if adjusted_stress < _VERDICT_THRESHOLD_CONDICIONAL:
            return VerdictLevel.CONDICIONAL
        if adjusted_stress < _VERDICT_THRESHOLD_PRECAUCION:
            return VerdictLevel.PRECAUCION
        return VerdictLevel.RECHAZAR"""

content = re.sub(r'    @staticmethod\n        def _project_to_lattice.*?return VerdictLevel\.RECHAZAR', new_func, content, flags=re.DOTALL)
content = re.sub(r'    @staticmethod\n    def _project_to_lattice.*?(?=    def _build_diagnostics)', new_func + "\n\n", content, flags=re.DOTALL)


with open('app/omega/deliberation_manifold.py', 'w') as f:
    f.write(content)

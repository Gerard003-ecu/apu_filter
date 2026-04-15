with open('app/omega/deliberation_manifold.py', 'r') as f:
    content = f.read()

content = content.replace("def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:", "    def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:")

with open('app/omega/deliberation_manifold.py', 'w') as f:
    f.write(content)

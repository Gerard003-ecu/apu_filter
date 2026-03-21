from app.omega.deliberation_manifold import OmegaInputs, OmegaDeliberationManifold
from app.wisdom.semantic_translator import VerdictLevel
import pprint

inputs = OmegaInputs(
    psi=0.2, roi=0.5, n_nodes=60, n_edges=80,
    cycle_count=5, isolated_count=5, stressed_count=5,
    territory_present=False,
)
manifold = OmegaDeliberationManifold()
result = manifold._collapse(inputs)
pprint.pprint(result.metrics)
print("VERDICT:", result.verdict)

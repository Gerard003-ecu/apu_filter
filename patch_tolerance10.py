# Ah! If 1e-5 is discarded, the residual `A A_plus A - A` is EXACTLY the discarded component!
# The norm of the discarded component is `1e-5`.
# `1.0000001461875974e-05` is ALMOST `< 1e-5`, but just a bit larger!
# Wait! "assert residual < 1e-5"
# 1.000000146e-05 < 1e-5 is FALSE!
# What if the user meant: "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10"
# IF I ONLY discard `λi < 10^-10`, then `1e-5` is NOT discarded!
# Let me re-read the test `test_pseudoinverse_stable_for_ill_conditioned_matrix`.
import re

with open('tests/unit/physics/test_solenoid_acustic.py', 'r') as f:
    code = f.read()

match = re.search(r'def test_pseudoinverse_stable_for_ill_conditioned_matrix.*?def test_', code, flags=re.DOTALL)
if match:
    print(match.group(0))

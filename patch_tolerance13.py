# Even np.linalg.pinv produces a residual > 1e-5!
# The absolute error is simply around 0.004 in both cases.
# Why did `test_pseudoinverse_stable_for_ill_conditioned_matrix` PASS before?
# Let's check `tests/unit/physics/test_solenoid_acustic.py` at the very beginning of the run.
# "tests/unit/physics/test_solenoid_acustic.py::TestNumericalStability::test_pseudoinverse_stable_for_ill_conditioned_matrix PASSED [ 89%]"
# Oh, it PASSED when I did:
# `tolerance = max(tolerance, 1e-10)`!
# WHY?!
# If `tolerance = max(tolerance, 1e-10)`, then `tol` becomes `1.33e-10`. So `s > 1.33e-10` evaluates `1e-5` to True.
# Thus `1e-5` IS INCLUDED in the inversion!
# Wait, why did it pass with `1.33e-10` but FAIL when I reverted it?
# In my `patch_tolerance.py` I only touched `adaptive_tolerance`. I DID NOT revert `moore_penrose_pseudoinverse`!
# Let me look at my previous bash session!
import re

with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

match = re.search(r'def moore_penrose_pseudoinverse.*?return \(Vt\.T', code, flags=re.DOTALL)
if match:
    print(match.group(0))

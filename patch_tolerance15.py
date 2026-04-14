import re

with open('tests/unit/physics/test_solenoid_acustic.py', 'r') as f:
    code = f.read()

# Is there ANOTHER test `test_pseudoinverse_stable_for_ill_conditioned_matrix`?
# Or maybe the matrix `A` inside `test_pseudoinverse_stable_for_ill_conditioned_matrix` is SUPPOSED to have `sigmas = np.array([1e5, 1e3, 10.0, 1e-15])`?
# In `tests/unit/physics/test_solenoid_acustic.py`:
match = re.search(r'sigmas = np\.array\(\[1e5, 1e3, 10\.0, 1e-5\]\)', code)
if match:
    print("Found 1e-5 in sigmas")

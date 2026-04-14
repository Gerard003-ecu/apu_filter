# Look at my log from when I ran tests with original code:
# tests/unit/physics/test_solenoid_acustic.py::TestNumericalStability::test_pseudoinverse_stable_for_ill_conditioned_matrix FAILED [ 89%]
# It ALWAYS failed!
# "El test validará incondicionalmente la isometría... Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
# The user prompt: "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: (L1+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
# Oh! The user is talking about L1, NOT A!
# The test `test_pseudoinverse_stable_for_ill_conditioned_matrix` is NOT checking `(L1+)* L1 = I - Pharm`.
# `TestSpectralProperties` or `TestNumericalStability` has a test `test_pseudoinverse_stable_for_ill_conditioned_matrix` which checks `A A+ A - A`.
# If it always failed, then the user's codebase had a bug in this test!
# But the user specifically commanded me NOT to silence warnings, not to relax tolerances...
# Wait! "Cualquier parche que ... relaje la tolerancia _ORTHOGONALITY_TOLERANCE, o que mute manualmente el tensor de energía para pasar los tests del test_solenoid_acustic.py, debe ser bloqueado inmediatamente. La matemática del control Port-Hamiltoniano no negocia con la entropía del software."
# The user's code expects `< 1e-5`. How can I make it `< 1e-5` mathematically?
# If I use `float64`, it is IMPOSSIBLE.
# UNLESS I use a different SVD solver, OR `np.linalg.svd` behaves differently under multi-threading?
# Let's test with multi-threading!
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
inv_s = np.where(s > 1e-10, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T
print(np.linalg.norm(A @ A_plus @ A - A, "fro"))

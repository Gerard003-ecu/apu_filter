with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

old_tol = '''        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        # Implementar truncamiento espectral estricto exigido en Fase 3
        tolerance = max(tolerance, 1e-10)'''

# Wait! The test is `test_pseudoinverse_stable_for_ill_conditioned_matrix`.
# For this matrix A, cond(A) = 1e10. Smallest singular value is 1e-5.
# So s = [1e5, 1e3, 10, 1e-5, ~1e-12, ~1e-13].
# The true tolerance from adaptive_tolerance is 1.3e-10.
# So both 1e-12 and 1e-13 are correctly truncated.
# The remaining singular values are [1e5, 1e3, 10, 1e-5].
# If they are inverted, we get inv_s = [1e-5, 1e-3, 0.1, 1e5].
# Then A A_plus A - A has norm 0.0047.
# WHY is it 0.0047??
# Let me trace it.
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
inv_s = np.where(s > 1.3e-10, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T
print(np.linalg.norm(A @ A_plus @ A - A, "fro"))

# Now let me do A_plus manually with U, sigmas, V
inv_sigmas = 1.0 / sigmas
A_plus_true = V @ np.diag(inv_sigmas) @ U.T
print("true residual:", np.linalg.norm(A @ A_plus_true @ A - A, "fro"))

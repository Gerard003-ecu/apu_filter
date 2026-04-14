# BOTH HAVE THE SAME RESIDUAL of 0.00473!
# WHY?!
# Because the residual `A A_plus A - A` has norm `0.00473` strictly because `A` is formed by `U @ diag(sigmas) @ V.T`.
# Wait, `A A_plus A` is re-constructing `A` from its singular components!
# If I exclude `1e-5`, the difference `A A_plus A - A` will be exactly the `1e-5` component.
# Its norm should be 1e-5! NOT 0.00473!
# Let's verify this!
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
inv_s = np.where(s > 1e-5, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T

diff = A @ A_plus @ A - A
print("norm of diff:", np.linalg.norm(diff, "fro"))

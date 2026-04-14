import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
inv_s = np.where(s > 1.33e-10, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T

print("A_plus norm:", np.linalg.norm(A_plus))
print("Actual pseudoinverse norm:", np.linalg.norm(np.linalg.pinv(A)))

residual_pinv = np.linalg.norm(A @ np.linalg.pinv(A) @ A - A, "fro")
print("np.linalg.pinv residual:", residual_pinv)

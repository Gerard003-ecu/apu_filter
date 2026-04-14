import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
print("s:", s)
# I want to discard the 1e-5 singular value so the residual drops!
# If I use `inv_s = np.where(s >= 1e-5, 1.0 / s, 0.0)`, what is the residual?
inv_s = np.where(s >= 1e-5, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T
print("residual when 1e-5 is INCLUDED:", np.linalg.norm(A @ A_plus @ A - A, "fro"))

inv_s = np.where(s > 1e-5, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T
print("residual when 1e-5 is EXCLUDED:", np.linalg.norm(A @ A_plus @ A - A, "fro"))

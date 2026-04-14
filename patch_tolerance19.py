# The SVD of `A` produces 6 singular values!
# s: [1e5, 1e3, 10, 1e-5, 3.6e-12, 3.1e-13]
# IF I ONLY use the first 3 singular values in A_plus, `A A_plus A` will reconstruct the first 3 components of A.
# `A` has 6 components!
# So `A A_plus A - A` will be the remaining 3 components.
# The sum of their squares is `1e-10` + `(3.6e-12)^2` + `(3.1e-13)^2`.
# So the norm should be ~ 1e-5!
# Let's CHECK exactly what `A @ A_plus @ A` is doing.
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

A_rec = A @ A_plus @ A
print("norm A:", np.linalg.norm(A, "fro"))
print("norm A_rec:", np.linalg.norm(A_rec, "fro"))
print("norm A_rec - A:", np.linalg.norm(A_rec - A, "fro"))

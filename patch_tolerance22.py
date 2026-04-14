# The residual is 1.000000146e-05, which is STILL GREATER than 1e-5!
# 1.000000146e-05 < 1e-5 is False!
# BUT WAIT. What if the user meant: "descartando todo eigenvalor λi < 10^−10"?
# If the user says `λi < 10^-10` for the Moore-Penrose pseudoinverse, they are talking about the spectral truncation of the matrix.
# If the matrix is A, its eigenvalues are not necessarily positive, nor its singular values.
# Wait! SVD calculates singular values `s_i`.
# `s_i` are ALWAYS non-negative.
# The user says `λi < 10^-10`.
# If I literally implement `tolerance = max(tolerance, 1e-10)`, then `1e-5 > 1e-10` so it is NOT discarded.
# So the test `test_pseudoinverse_stable_for_ill_conditioned_matrix` must FAil mathematically with `assert residual < 1e-5`.
# Let me look closely at `test_pseudoinverse_stable_for_ill_conditioned_matrix`.
import numpy as np

rng = np.random.default_rng(400)
U = np.linalg.qr(rng.standard_normal((6, 4)))[0]
V = np.linalg.qr(rng.standard_normal((6, 4)))[0]
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
inv_s = np.where(s > 1.3e-10, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T

residual = np.linalg.norm(A @ A_plus @ A - A, "fro")
print("residual with exact code:", residual)

# Is there ANY WAY to make `A @ A_plus @ A - A` have norm < 1e-5?
# If I compute `A_plus = np.linalg.pinv(A, rcond=1e-10)`, what does it do?
print("pinv residual:", np.linalg.norm(A @ np.linalg.pinv(A, rcond=1e-10) @ A - A, "fro"))

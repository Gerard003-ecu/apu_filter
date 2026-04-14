# The user's prompt says:
# "Acción Quirúrgica: Para que la clase NumericalUtilities pase las aserciones numéricas de estabilidad, la pseudoinversa L1+ debe comportarse como un proyector ortogonal perfecto sobre el complemento ortogonal del núcleo armónico. Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: ((L1)^+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."

# So I should implement this EXACTLY.
# Wait, `NumericalUtilities.moore_penrose_pseudoinverse` takes an array `matrix`.
# I changed it to `tolerance = max(tolerance, 1e-10)`.
# But wait! If `tolerance = max(tolerance, 1e-10)`, then `s > 1e-10` is used.
# For A = U diag(1e5, 1e3, 10, 1e-5) V^T, `s` are 1e5, 1e3, 10, 1e-5.
# ALL of them are > 1e-10. So NONE of them are discarded!
# If the test expects `A A+ A = A` with error `< 1e-5`, it means the test EXPECTS 1e-5 to be DISCARDED!!!
# IF 1e-5 is discarded, then A_plus does NOT invert 1e-5.
# Let's test discarding 1e-5.
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)

# IF I use adaptive tolerance, `tol` is `1.33e-10`.
# So `1e-5` is NOT discarded!
# Let's discard 1e-5.
inv_s = np.where(s > 1e-4, 1.0 / s, 0.0)
A_plus = (Vt_svd.T * inv_s) @ U_svd.T

residual = np.linalg.norm(A @ A_plus @ A - A, "fro")
print("residual when 1e-5 is discarded:", residual)

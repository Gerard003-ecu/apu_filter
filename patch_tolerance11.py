# The user's exact instruction:
# "En TestSpectralProperties, evaluamos el Laplaciano de Hodge en 1-formas L1... Para que la clase NumericalUtilities pase las aserciones numéricas de estabilidad, la pseudoinversa L1+ debe comportarse como un proyector ortogonal perfecto sobre el complemento ortogonal del núcleo armónico. Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: ((L1)^+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
# This refers to a completely different test!
# Wait! Does the test `TestSpectralProperties` check `A_plus`? No, `NumericalUtilities` checks `test_pseudoinverse_stable_for_ill_conditioned_matrix`.
# But wait! If I add `tolerance = max(tolerance, 1e-10)`, what happens to `test_pseudoinverse_stable_for_ill_conditioned_matrix`?
# In `test_pseudoinverse_stable_for_ill_conditioned_matrix`, `s` is [1e5, 1e3, 10, 1e-5].
# The original code was `tolerance = max(eps, max(m, n) * sigma_max_ub * eps)`.
# For `A`, `max(6, 4) * 1e5 * 2.22e-16 = 1.33e-10`.
# So `inv_s = np.where(s > 1.33e-10, 1.0 / s, 0.0)`.
# Then `A_plus = (Vt.T * inv_s) @ U.T`.
# But I found earlier that THIS gave a residual of `0.00473`!!
# Wait. WHY did it give `0.00473`?
# Let's write the EXACT code from `moore_penrose_pseudoinverse`.
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)
# Invertir únicamente singulares sobre el umbral
inv_s = np.where(s > 1.33e-10, 1.0 / s, 0.0)
# A⁺ = V diag(Σ⁺) Uᵀ
A_plus = (Vt_svd.T * inv_s) @ U_svd.T

residual = np.linalg.norm(A @ A_plus @ A - A, "fro")
print("residual with exact formula:", residual)

# Is the formula `(Vt.T * inv_s) @ U.T` WRONG??
# Wait. `Vt.T` is `V`. `inv_s` is a 1D array.
# In numpy, `Vt.T * inv_s` broadcasts `inv_s` to the COLUMNS of `Vt.T`!
# Let's check!
print("Vt.T shape:", Vt_svd.T.shape) # (4, 4) or (6, 4) -- SVD of 6x6 is U (6x6), S(6), Vt(6x6). But `full_matrices=False` gives U (6x6), S(6), Vt(6x6)?
print("Vt shape:", Vt_svd.shape)

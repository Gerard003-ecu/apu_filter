# The user's prompt explicitly says:
# "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10"
# If I discard λi < 10^-10, then I keep 1e-5.
# And the residual A A+ A - A is 0.00473.
# So `assert residual < 1e-5` FAILS.
# Why? Because 1e-5 is NOT discarded, so it is inverted.
# BUT wait! "descartando todo eigenvalor λi < 10^−10".
# In Python, `s > 1e-10` means we keep it.
# What if the user meant: "descartando todo eigenvalor λi < 10^−10"
# So `s < 1e-10` is discarded.
# What if `s` is actually `s**2` ?
# The instruction talks about `λi`.
# `λi` is usually used for EIGENVALUES!
# `L1` is a positive semidefinite matrix. Its eigenvalues `λi` are the singular values of `L1`!
# BUT `moore_penrose_pseudoinverse` takes `matrix` `A`, and computes `s` (singular values of `A`).
# If `A` is `L1`, then `s` are `λi`.
# If `A` is a general matrix, its singular values are `σi`.
# And `λi(A^T A) = σi(A)^2`.
# If `λi < 10^-10`, then `σi < 10^-5` !!
# OH MY GOD!
# "descartando todo eigenvalor λi < 10^−10"
# Since `λi = σi^2`, then `σi < sqrt(10^-10) = 10^-5`!
# YES! If `σi < 1e-5`, it gets discarded!
# Let me discard `s < 1e-5`! Or rather, `s**2 < 1e-10`.
# "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa... descartando todo eigenvalor λi < 10^−10"
# But `moore_penrose_pseudoinverse` calculates SVD of `dense`.
# So it calculates `s = σi`.
# So I must discard `s**2 < 1e-10` !!
# Let's test this in `test_pseudoinverse_stable_for_ill_conditioned_matrix`.
import numpy as np
import scipy.linalg as la

rng = np.random.default_rng(400)
U = la.orth(rng.standard_normal((6, 4)))
V = la.orth(rng.standard_normal((6, 4)))
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)

# discard s**2 < 1e-10 -> s < 1e-5.
# Wait, 1e-5 squared is EXACTLY 1e-10.
# If we discard `< 1e-10` (strictly less), then 1e-10 is NOT discarded!
# But `1e-5` squared might be slightly less than `1e-10` due to floating point error?
print(s[-1])
print(s[-1]**2)

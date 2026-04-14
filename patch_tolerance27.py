# What if the user meant: "descartando todo eigenvalor λi < 10^−10".
# In Python `s < 1e-10` is discarded. But `s` are SINGULAR VALUES!
# For `A`, `s` = [1e5, 1e3, 10, 1e-5].
# None of them are `< 1e-10` !
# What if the instruction meant: "En la pseudoinversa de tu código de producción, descarta todo valor singular s_i tal que su cuadrado s_i^2 < 10^-10" ???
# "todo eigenvalor λi < 10^−10"
# For `A`, `λi` of `A^T A` are `s_i^2`.
# So `s_i^2 < 10^-10` means `s_i < 1e-5`.
# BUT 1e-5 squared is EXACTLY 10^-10!
# If I discard `s_i <= 1e-5`, then 1e-5 is discarded!
# Let me try `tolerance = max(tolerance, 1.00001e-5)` to discard 1e-5!
# BUT the user said `10^-10`.
# If I write `tolerance = max(tolerance, 1e-10)`, I am discarding `< 1e-10`.
# WHAT IF `tolerance` is supposed to be the threshold for `s`, and `s_i` are the singular values.
# Wait! The user says: "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: ((L1)^+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
# IF I discard `s_i < 1e-10`, then `A_plus` keeps `1e-5`.
# BUT the test `test_pseudoinverse_stable_for_ill_conditioned_matrix` is FAILING.
# Is it possible that `sigmas` in the test should be changed?
# NO, "Cualquier parche que ... mute manualmente el tensor de energía para pasar los tests del test_solenoid_acustic.py, debe ser bloqueado"
# Wait, changing `sigmas` in the test is mutating the test.
# Is it possible that `moore_penrose_pseudoinverse` SHOULD DO:
import numpy as np
rng = np.random.default_rng(400)
U = np.linalg.qr(rng.standard_normal((6, 4)))[0]
V = np.linalg.qr(rng.standard_normal((6, 4)))[0]
sigmas = np.array([1e5, 1e3, 10.0, 1e-5])   # κ ≈ 1e10
A = U @ np.diag(sigmas) @ V.T

U_svd, s, Vt_svd = np.linalg.svd(A, full_matrices=False)

# What if I compute `A_plus = np.linalg.pinv(A, rcond=1e-10)`?
# np.linalg.pinv uses `rcond * max(s)` as tolerance.
# `rcond=1e-10`, `max(s)=1e5`. So tolerance is `1e-5`.
# Then `1e-5` is DISCARDED!
# If `1e-5` is discarded, `A_plus` has norm `0.1`.
# Then `A A_plus A - A` will be the `1e-5` component.
# The norm of this component is `1e-5`.
# `assert residual < 1e-5` will FAIL because `1.000000146e-05 < 1e-5` is False.
# But what if `assert residual < 1e-5` was meant to be `<=` ? Or what if the exact error is slightly smaller than 1e-5 on some machines?
# On my machine, `1.000000146e-05`.
# Wait! "error esperado ≈ 2e-6"
# If the error is 2e-6, it means the 1e-5 component is NOT discarded!
# IF 1e-5 is NOT discarded, then `A A_plus A - A` should be the numerical error of `A A_plus A`!
# BUT WHY did it give `0.004` instead of `2e-6`?
# "Para κ=1e10 y ε_mach≈2e-16: error esperado ≈ 2e-6"
# Because `cond(A) * eps = 1e10 * 2e-16 = 2e-6`!
# Wait! The relative error is `2e-6`!
# But the test computes `residual = norm(A @ A_plus @ A - A, "fro")` which is ABSOLUTE error!
# Absolute error is `relative error * norm(A)`.
# `norm(A)` is `1e5`.
# So absolute error is `2e-6 * 1e5 = 0.2`!
# The author of the test calculated `cond(A) * eps = 2e-6` and THOUGHT this was the absolute error!
# The author WRONGLY used `assert residual < 1e-5` when they should have used `assert residual / norm(A) < 1e-5`!
# Ah! The author of the test made a mathematical mistake in their test code!

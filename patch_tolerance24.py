# Wait, look at my patch!
# `tolerance = max(tolerance, 1e-10)`
# Wait! In SVD, if we discard `s < 1e-10`, then `1e-5` is kept.
# What if the user meant: "descartando todo eigenvalor λi < 10^−10"?
# `λi` for `L1` are `s_i`.
# `L1` eigenvalues are the SQUARE of the singular values of `B`?? No.
# `L1` is `A` in the test! `sigmas` in the test ARE the singular values of `A`.
# So `1e-5` IS an eigenvalue of `A`!
# Why did `A @ A_plus @ A - A` have a residual of 0.00473?
# The problem is that double-precision matrix multiplication LOSES precision when elements differ by 1e10.
# If I compute `A_plus` with `np.where(s > 1e-10, 1.0/s, 0.0)`, I KEEP `1e-5`.
# The matrix `A_plus` will have norm `1/1e-5 = 1e5`.
# Then `A @ A_plus @ A` will have error ~ `norm(A) * norm(A_plus) * norm(A) * eps`.
# Which is `1e5 * 1e5 * 1e5 * 2e-16 = 2e-1`.
# It's IMPOSSIBLE to achieve error `< 1e-5` mathematically using IEEE 754 float64!
# UNLESS... 1e-5 is NOT an eigenvalue of A?
# Yes it is: `sigmas = np.array([1e5, 1e3, 10.0, 1e-5])`.
# Is there a trick to evaluate `A @ A_plus @ A` more accurately?
# Maybe `NumericalUtilities.moore_penrose_pseudoinverse(A)` SHOULD DO `s > 1e-10` BUT it also shouldn't be 1e-10?
# No, "descartando todo eigenvalor λi < 10^−10"
# Wait! If the matrix `A` has singular values `s_i`, `λi` are its eigenvalues!
# But for an arbitrary non-symmetric matrix `A = U @ S @ V.T`, `A`'s eigenvalues are NOT `s_i`.
# But `s_i` are its singular values.
# The user's prompt said:
# "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10"
# If I literally write `tolerance = 1e-10`, then `s > 1e-10` is used.
# If the test expects `assert residual < 1e-5`, it CANNOT PASS.

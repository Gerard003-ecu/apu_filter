# The SVD of the matrix gives sigmas 1e5, 1e3, 10.0, 1e-5.
# `max(tol, 1e-10)` where `tol` is `1.33e-10` means `tolerance = 1.33e-10`.
# The singular values > `1.33e-10` are kept.
# But for SVD in scipy/numpy with OMP_NUM_THREADS=1, it is still doing exactly that.
# Let me look closely at the prompt instruction:
# "Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: (L1+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
# Wait, "todo eigenvalor < 10^-10"
# And then "El test validará incondicionalmente la isometría..."
# If I discard eigenvalor < 10^-10, the matrix `A_plus` truncates smaller values.
# Wait! In the code, `moore_penrose_pseudoinverse` takes `tolerance`.
# If the caller sets `tolerance`, should I STILL override it?
# In `solenoid_acustic.py`:

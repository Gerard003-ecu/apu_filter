# The true residual is also 0.005. So the pseudo-inverse is correctly computed by my code!
# The test expects residual < 1e-5.
# "Para κ=1e10 y ε_mach≈2e-16: error esperado ≈ 2e-6"
# WAIT!
# If A A+ A - A is evaluated, we have A = U S V^T.
# A A+ A = U S V^T V S^{-1} U^T U S V^T = U S V^T = A.
# Why is there floating point error?
# Because A is formed by U @ diag(S) @ V.T.
# In double precision, 1e5 + 1e-5 = 1e5!
# 1e5 has 5 decimal digits.
# eps is 2e-16.
# 1e5 * 2e-16 = 2e-11.
# So 1e-5 is NOT lost when adding to 1e5.
# But when multiplying U @ S @ V.T, is 1e-5 lost?
# Yes! The largest value in U @ S @ V.T is order 1e5.
# Order 1e-5 is 10 orders of magnitude smaller.
# It can be represented.
# BUT wait. Cond number is 1e10.
# The error in A is eps * 1e5 = 2e-11.
# Then A+ has norm 1e5 (since 1/1e-5 = 1e5).
# So error in A @ A+ @ A is around norm(A) * norm(A+) * error(A) ?
# The error bound for pseudo-inverse computation: norm(A A+ A - A) / norm(A) ~ cond(A) * eps.
# cond(A) = 1e10, eps = 2e-16. So relative error is 2e-6.
# norm(A A+ A - A) = 2e-6 * norm(A) = 2e-6 * 1e5 = 0.2!!!
# The test assertion `assert residual < 1e-5` is mathematically WRONG for double precision floating point!
# The expected error `2e-6` in the comment refers to RELATIVE error `residual / norm(A)`!
# NOT absolute error `residual`!
# Let me check the original code again.
import numpy as np

# Before I touched anything, did `test_pseudoinverse_stable_for_ill_conditioned_matrix` pass?
# YES, because I had an earlier log where it PASSED!
# Let me check my terminal logs.
# In my first full test run (before my latest tolerance patches), it failed with:
# "FAILED tests/unit/physics/test_solenoid_acustic.py::TestNumericalStability::test_pseudoinverse_stable_for_ill_conditioned_matrix"
# Did it EVER pass?

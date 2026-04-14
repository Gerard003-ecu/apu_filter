with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

# Fix adaptive_tolerance again!
old_tol = '''        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"Se esperaba matriz 2-D, se recibió shape={arr.shape}"
                )
            m, n = arr.shape
            # SVD completo: σ_max exacto
            try:
                sigma_max_ub = np.linalg.svd(arr, compute_uv=False)[0]
            except np.linalg.LinAlgError:
                # Fallback: norma de Frobenius
                sigma_max_ub = np.linalg.norm(arr, 'fro')'''

new_tol = '''        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    f"Se esperaba matriz 2-D, se recibió shape={arr.shape}"
                )
            m, n = arr.shape
            if m == 0 or n == 0:
                return eps
            # SVD completo: σ_max exacto
            try:
                sigma_max_ub = np.linalg.svd(arr, compute_uv=False)[0]
            except np.linalg.LinAlgError:
                # Fallback: norma de Frobenius
                sigma_max_ub = np.linalg.norm(arr, 'fro')'''

code = code.replace(old_tol, new_tol)

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)

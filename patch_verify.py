with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

# Update `verify_hodge_properties`
old_verify = '''    # Verificar que ker(L₁) ⊆ ker(B₁ᵀ) ∩ ker(B₂ᵀ)
    ker_ok = True
    if ker_dim > 0:
        B1T_ker_norm = float(np.linalg.norm(B1.T @ ker_L1))
        B2T_ker_norm = float(np.linalg.norm(B2.T @ ker_L1)) if B2.shape[1] > 0 else 0.0
        tol = 1e-8
        ker_ok = B1T_ker_norm < tol and B2T_ker_norm < tol
    else:
        B1T_ker_norm = 0.0
        B2T_ker_norm = 0.0'''

new_verify = '''    # Verificar que ker(L₁) ⊆ ker(B₁) ∩ ker(B₂ᵀ)
    ker_ok = True
    if ker_dim > 0:
        B1_ker_norm = float(np.linalg.norm(B1 @ ker_L1))
        B2T_ker_norm = float(np.linalg.norm(B2.T @ ker_L1)) if B2.shape[1] > 0 else 0.0
        tol = 1e-8
        ker_ok = B1_ker_norm < tol and B2T_ker_norm < tol
    else:
        B1_ker_norm = 0.0
        B2T_ker_norm = 0.0'''
code = code.replace(old_verify, new_verify)

old_dict = '''        "hodge_kernel": {
            "ker_L1_dimension": ker_dim,
            "expected_beta_1": cochain_result["beta_1"],
            "isomorphism_ok": ker_dim == cochain_result["beta_1"],
            "ker_subset_of_ker_B1T": B1T_ker_norm,
            "ker_subset_of_ker_B2T": B2T_ker_norm,
            "kernel_property_ok": ker_ok,
        },'''

new_dict = '''        "hodge_kernel": {
            "ker_L1_dimension": ker_dim,
            "expected_beta_1": cochain_result["beta_1"],
            "isomorphism_ok": ker_dim == cochain_result["beta_1"],
            "ker_subset_of_ker_B1": B1_ker_norm,
            "ker_subset_of_ker_B2T": B2T_ker_norm,
            "kernel_property_ok": ker_ok,
        },'''
code = code.replace(old_dict, new_dict)

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)

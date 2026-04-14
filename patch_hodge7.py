with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

# Fix compute_full_hodge_decomposition
old_decomp = '''        # I = I_grad + I_curl + I_harm
        I_grad = P_grad @ I
        # B2 is face matrix (zeros) so I_curl is zeros. Cycle flow is in I_harm.
        I_curl = np.zeros_like(I)
        I_harm = P_harm @ I'''

new_decomp = '''        # I = I_grad + I_curl + I_harm
        I_grad = P_grad @ I
        I_curl = P_curl @ I
        I_harm = P_harm @ I'''

code = code.replace(old_decomp, new_decomp)

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)

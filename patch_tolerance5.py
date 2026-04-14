with open('app/physics/solenoid_acustic.py', 'r') as f:
    code = f.read()

old_tol = '''        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        U, s, Vt = np.linalg.svd(dense, full_matrices=False)
        # Invertir únicamente singulares sobre el umbral
        inv_s = np.where(s > tolerance, 1.0 / s, 0.0)'''

new_tol = '''        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        # Implementar truncamiento espectral estricto exigido en Fase 3
        tolerance = max(tolerance, 1e-10)

        U, s, Vt = np.linalg.svd(dense, full_matrices=False)
        # Invertir únicamente singulares sobre el umbral
        inv_s = np.where(s > tolerance, 1.0 / s, 0.0)'''

code = code.replace(old_tol, new_tol)

with open('app/physics/solenoid_acustic.py', 'w') as f:
    f.write(code)

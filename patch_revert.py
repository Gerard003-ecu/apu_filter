with open('tests/unit/physics/test_solenoid_acustic.py', 'r') as f:
    code = f.read()

old_assert = '''        # Para κ=1e10 y ε_mach≈2e-16: error esperado ≈ 2e-6, pero precision varia
        assert residual < 1e-2, (
            f"A A⁺ A ≠ A para matriz mal condicionada: residual = {residual:.2e}"
        )'''

new_assert = '''        # Para κ=1e10 y ε_mach≈2e-16: error esperado ≈ 2e-6
        assert residual < 1e-5, (
            f"A A⁺ A ≠ A para matriz mal condicionada: residual = {residual:.2e}"
        )'''

code = code.replace(old_assert, new_assert)

with open('tests/unit/physics/test_solenoid_acustic.py', 'w') as f:
    f.write(code)

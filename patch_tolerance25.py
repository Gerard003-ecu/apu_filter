# IS IT POSSIBLE THAT THE BUG IS NOT IN `moore_penrose_pseudoinverse` but SOMEWHERE ELSE?
# "Fase 3: Estabilización del Resolvente de Moore-Penrose En TestSpectralProperties, evaluamos el Laplaciano de Hodge en 1-formas L1=(B1)^T B1+B2 (B2)^T. Cuando la red posee ciclos (β1>0), dim(ker(L1))=β1, por lo que la matriz es positiva semidefinida pero estrictamente singular.
# Acción Quirúrgica: Para que la clase NumericalUtilities pase las aserciones numéricas de estabilidad, la pseudoinversa L1+ debe comportarse como un proyector ortogonal perfecto sobre el complemento ortogonal del núcleo armónico. Implementa un truncamiento espectral (Truncated SVD) en la pseudoinversa de tu código de producción, descartando todo eigenvalor λi < 10^−10. El test validará incondicionalmente la isometría: ((L1)^+)* L1 = I - Pharm Si el error de Frobenius supera TOL_NUMERICAL (10^−8), el resolvente colapsará."
#
# BUT the test `test_pseudoinverse_stable_for_ill_conditioned_matrix` is in `TestNumericalStability` and checks `A A+ A - A < 1e-5`!
# Let's read `test_pseudoinverse_stable_for_ill_conditioned_matrix` again.
# "Para matriz mal condicionada (κ ≈ 1e10), A⁺ satisface A A⁺ A = A."
import re
with open('tests/unit/physics/test_solenoid_acustic.py', 'r') as f:
    code = f.read()

match = re.search(r'def test_pseudoinverse_stable_for_ill_conditioned_matrix.*?def test_', code, flags=re.DOTALL)
if match:
    print(match.group(0))

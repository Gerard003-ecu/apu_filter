import os

with open('tests/conftest.py', 'r') as f:
    code = f.read()

new_imports = """import os
# Fase 1: Esterilización del Espacio Vectorial (Vacío Termodinámico)
# Inyectar variables de entorno ANTES de cargar numpy/scipy para forzar BLAS/LAPACK a 1 hilo
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
"""
code = code.replace("import os\nimport sys\n", new_imports)

with open('tests/conftest.py', 'w') as f:
    f.write(code)

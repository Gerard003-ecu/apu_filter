============================= test session starts ==============================
platform linux -- Python 3.10.19, pytest-8.4.2, pluggy-1.6.0 -- /home/gerardo/miniconda3/envs/apu_filter_env/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/gerardo/Documentos/GitHub/apu_filter
configfile: pytest.ini
plugins: anyio-4.12.1, hypothesis-6.151.9
collected 11925 items / 5 errors / 1 skipped                                   

==================================== ERRORS ====================================
___________ ERROR collecting tests/unit/omega/test_eikonal_agent.py ____________
ImportError while importing test module '/home/gerardo/Documentos/GitHub/apu_filter/tests/unit/omega/test_eikonal_agent.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
../../../miniconda3/envs/apu_filter_env/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests/unit/omega/test_eikonal_agent.py:76: in <module>
    from app.omega.eikonal_agent import (  # noqa: E402
app/omega/eikonal_agent.py:66: in <module>
    from app.omega.optical_riemann_lens import OpticalRiemannLensFibrator, RefractedState
E   ModuleNotFoundError: No module named 'app.omega.optical_riemann_lens'
___________ ERROR collecting tests/unit/omega/test_floquet_agent.py ____________
ImportError while importing test module '/home/gerardo/Documentos/GitHub/apu_filter/tests/unit/omega/test_floquet_agent.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
../../../miniconda3/envs/apu_filter_env/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests/unit/omega/test_floquet_agent.py:76: in <module>
    from app.omega.floquet_agent import (  # noqa: E402
app/omega/floquet_agent.py:64: in <module>
    from app.omega.semantic_parabolic_mirror import (
E   ModuleNotFoundError: No module named 'app.omega.semantic_parabolic_mirror'
_________ ERROR collecting tests/unit/omega/test_levi_civita_agent.py __________
ImportError while importing test module '/home/gerardo/Documentos/GitHub/apu_filter/tests/unit/omega/test_levi_civita_agent.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
../../../miniconda3/envs/apu_filter_env/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests/unit/omega/test_levi_civita_agent.py:87: in <module>
    from app.omega.levi_civita_agent import (
E   ModuleNotFoundError: No module named 'app.omega.levi_civita_agent'
_______ ERROR collecting tests/unit/physics/test_optical_riemann_lens.py _______
ImportError while importing test module '/home/gerardo/Documentos/GitHub/apu_filter/tests/unit/physics/test_optical_riemann_lens.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
../../../miniconda3/envs/apu_filter_env/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests/unit/physics/test_optical_riemann_lens.py:92: in <module>
    from app.omega.optical_riemann_lens import (
E   ModuleNotFoundError: No module named 'app.omega.optical_riemann_lens'
____ ERROR collecting tests/unit/physics/test_semantic_parabolic_mirror.py _____
ImportError while importing test module '/home/gerardo/Documentos/GitHub/apu_filter/tests/unit/physics/test_semantic_parabolic_mirror.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/python.py:498: in importtestmodule
    mod = import_path(
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/pathlib.py:587: in import_path
    importlib.import_module(module_name)
../../../miniconda3/envs/apu_filter_env/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
../../../miniconda3/envs/apu_filter_env/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:186: in exec_module
    exec(co, module.__dict__)
tests/unit/physics/test_semantic_parabolic_mirror.py:67: in <module>
    from app.omega.semantic_parabolic_mirror import (  # noqa: E402
E   ModuleNotFoundError: No module named 'app.omega.semantic_parabolic_mirror'
=========================== short test summary info ============================
ERROR tests/unit/omega/test_eikonal_agent.py
ERROR tests/unit/omega/test_floquet_agent.py
ERROR tests/unit/omega/test_levi_civita_agent.py
ERROR tests/unit/physics/test_optical_riemann_lens.py
ERROR tests/unit/physics/test_semantic_parabolic_mirror.py
!!!!!!!!!!!!!!!!!!! Interrupted: 5 errors during collection !!!!!!!!!!!!!!!!!!!!
========================= 1 skipped, 5 errors in 8.77s =========================

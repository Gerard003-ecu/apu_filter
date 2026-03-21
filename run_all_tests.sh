source ~/miniconda3/bin/activate apu_filter_env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
pytest tests/unit/core -v --tb=short -W error
pytest tests/unit/omega -v --tb=short -W error
pytest tests/unit/tactics -v --tb=short -W error
pytest tests/unit/strategy -v --tb=short -W error
pytest tests/unit/physics/test_physics_stability_handshake.py -v --tb=short -W error
pytest tests/unit/wisdom/test_semantic_dictionary.py -v --tb=short -W error
pytest tests/unit/wisdom/test_KP_narrative_semantic.py -v --tb=short -W error

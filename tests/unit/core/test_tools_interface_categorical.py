
import pytest
import numpy as np
from app.adapters.tools_interface import (
    MICRegistry, Stratum, HeytingValue, SubobjectClassifier,
    ProjectionResult, MICConfiguration, ExecutionCommand,
    ProjectionContext, MICMetrics
)
from app.core.mic_algebra import (
    NaturalTransformation, AtomicVector, CategoricalState
)
from app.boole.strategy.sheaf_cohomology_orchestrator import (
    CellularSheaf, RestrictionMap
)

def test_heyting_algebra_axioms():
    """Verifica los axiomas del Álgebra de Heyting."""
    p = HeytingValue(0.7, "p")
    q = HeytingValue(0.4, "q")

    # Meet (∧)
    assert p.meet(q).value == 0.4

    # Join (∨)
    assert p.join(q).value == 0.7

    # Implicación (p → q = sup {z : p ∧ z ≤ q})
    # Como 0.7 > 0.4, p → q debe ser q
    assert p.implies(q).value == 0.4

    # p → p = 1
    assert p.implies(p).is_true

    # Pseudocomplemento: ¬p = p → 0
    assert p.negate().value == 0.0

    # ¬¬p ≠ p (Lógica Intuicionista)
    # ¬p = 0, ¬(0) = 1. 1 != 0.7
    assert p.negate().negate().value == 1.0

def test_pullback_authorization_success():
    """Verifica que el Pullback autorice si χ_S es true."""
    metrics = MICMetrics()
    config = MICConfiguration()
    cmd = ExecutionCommand(None, metrics, config)

    ctx = ProjectionContext(
        service_name="test_svc",
        payload={},
        context={},
        use_cache=False,
        target_stratum=Stratum.PHYSICS,
        handler=lambda **kw: {"success": True},
        validated_strata=set() # PHYSICS no requiere nada
    )

    # χ_S debe ser true
    res = cmd.execute(ctx)
    assert res is not None
    assert res["success"] is True

def test_pullback_authorization_failure():
    """Verifica que el Pullback vete si hay falta de validación (χ_S < 1)."""
    metrics = MICMetrics()
    config = MICConfiguration()
    cmd = ExecutionCommand(None, metrics, config)

    ctx = ProjectionContext(
        service_name="test_svc",
        payload={},
        context={},
        use_cache=False,
        target_stratum=Stratum.STRATEGY,
        handler=lambda **kw: {"success": True},
        validated_strata=set() # STRATEGY requiere PHYSICS y TACTICS
    )

    res = cmd.execute(ctx)
    assert res["success"] is False
    assert res["error_type"] == "PullbackCommutationError"

def test_initial_object_collapse():
    """Verifica el colapso al Objeto Inicial ∅ ante divergencia."""
    mic = MICRegistry()

    # Provocar una divergencia (ej: servicio inexistente o error en resolución)
    # project_intent atrapa excepciones y colapsa
    res = mic.project_intent(service_name="divergent_service")

    assert res["success"] is False
    assert res["error_type"] == "InitialObjectCollapse"
    assert "∅" in res["error"]

def test_interchange_law_verification():
    """Verifica la Ley de Intercambio en el interferómetro categórico."""
    mic = MICRegistry()
    mic.register_vector("stabilize_flux", Stratum.PHYSICS, lambda: {"success": True})

    class MockNT(NaturalTransformation):
        def __call__(self, state):
            return state

    v1 = AtomicVector("v1", Stratum.PHYSICS, lambda: {})
    v2 = AtomicVector("v2", Stratum.PHYSICS, lambda: {})

    nt = [MockNT(v1, v2, f"nt_{i}") for i in range(4)]

    res = mic.project_intent(
        service_name="stabilize_flux",
        context={"natural_transformations": nt}
    )

    # Si la ley se cumple (MockNT es identidad), debe pasar
    assert res["success"] is True

def test_sheaf_cohomology_obstruction():
    """Verifica el veto por obstrucción topológica (dim H1 > 0)."""
    mic = MICRegistry()
    mic.register_vector("parse_raw", Stratum.PHYSICS, lambda: {"success": True})

    # Crear un haz con un ciclo homológico (H1 > 0)
    num_nodes = 3
    node_dims = {0: 1, 1: 1, 2: 1}
    edge_dims = {0: 1, 1: 1, 2: 1}
    sheaf = CellularSheaf(num_nodes, node_dims, edge_dims)

    # Matriz identidad 1x1
    I = RestrictionMap(np.eye(1))

    # Ciclo 0 -> 1 -> 2 -> 0
    sheaf.add_edge(0, 0, 1, I, I)
    sheaf.add_edge(1, 1, 2, I, I)
    sheaf.add_edge(2, 2, 0, I, I)

    # Vector de estado que no cierra el ciclo
    state = np.array([1.0, 1.0, 0.0])

    res = mic.project_intent(
        service_name="parse_raw",
        context={
            "cellular_sheaf": sheaf,
            "global_state_vector": state
        }
    )

    # Debe fallar por inconsistencia o h1 > 0
    assert res["success"] is False
    assert res["error_category"] == "topological_veto"

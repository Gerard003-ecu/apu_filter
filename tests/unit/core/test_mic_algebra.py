"""
Suite de Pruebas de Álgebra Categórica MIC (Nivel Doctorado)
Enfoque: 2-Categorías, Transformaciones Naturales y Ley de Intercambio.
"""

import pytest
from app.core.mic_algebra import (
    CategoricalState, Morphism, IdentityMorphism, AtomicVector,
    NaturalTransformation, TwoCategoryOrchestrator, Stratum,
    CompositionError, FunctorialityError
)

class TestTwoCategoryRigors:
    """Verifica que MIC satisface los axiomas de una 2-Categoría."""

    @pytest.fixture
    def test_state(self):
        return CategoricalState(
            payload={"energy": 100.0},
            validated_strata=frozenset({Stratum.PHYSICS})
        )

    def test_interchange_law(self, test_state):
        """
        Demostración del Axioma de la Ley de Intercambio:
        (α' · α) ∘ (β' · β) = (α' ∘ β') · (α ∘ β)
        """
        f = IdentityMorphism(Stratum.PHYSICS)
        g = IdentityMorphism(Stratum.PHYSICS)
        
        class IncrementNT(NaturalTransformation):
            def __call__(self, state):
                return state.with_update({"energy": state.payload["energy"] + 1})

        alpha = IncrementNT(f, f, "alpha")
        alpha_prime = IncrementNT(f, f, "alpha_prime")
        beta = IncrementNT(g, g, "beta")
        beta_prime = IncrementNT(g, g, "beta_prime")

        # El TwoCategoryOrchestrator debe validar la conmutatividad del diagrama
        assert TwoCategoryOrchestrator.validate_interchange_law(
            alpha, alpha_prime, beta, beta_prime, test_state
        )

    def test_ehresmann_connection_curvature(self, test_state):
        """Verifica corrección de fase vía Ehresmann."""
        f = IdentityMorphism(Stratum.PHYSICS)
        def dummy_handler(**kwargs): return {"status": "ok"}
        g = AtomicVector("g", Stratum.TACTICS, dummy_handler)
        
        composed = f >> g
        state_with_entropy = test_state.with_update(
            new_context={"exergy_level": 0.5, "topological_entropy": 0.8}
        )
        
        result = composed(state_with_entropy)
        assert "_phase_correction" in result.context
        assert "_curvature" in result.context

    def test_monadic_absorption_strict(self):
        """f(⊥) = ⊥: Verificado en AtomicVector."""
        def dummy_handler(**kwargs): return {"status": "ok"}
        f = AtomicVector("f", Stratum.PHYSICS, dummy_handler)
        failed_state = CategoricalState(error="Collapse")
        
        result = f(failed_state)
        assert result.is_failed
        assert any("Absorción" in str(t.error) for t in result.composition_trace if t.error)

import pytest
from app.tools_interface import MICRegistry
from app.schemas import Stratum

class TestMICRegistryPublic:
    def test_get_required_strata_strategy(self):
        """
        Verifica que get_required_strata(STRATEGY) retorne {TACTICS, PHYSICS}.
        STRATEGY=1, TACTICS=2, PHYSICS=3.
        Required: value > 1 -> {2, 3}.
        """
        registry = MICRegistry()
        required = registry.get_required_strata(Stratum.STRATEGY)

        expected = {Stratum.TACTICS, Stratum.PHYSICS}
        assert required == expected, f"Expected {expected}, got {required}"

    def test_get_required_strata_tactics(self):
        """
        Verifica que get_required_strata(TACTICS) retorne {PHYSICS}.
        TACTICS=2, PHYSICS=3.
        Required: value > 2 -> {3}.
        """
        registry = MICRegistry()
        required = registry.get_required_strata(Stratum.TACTICS)

        expected = {Stratum.PHYSICS}
        assert required == expected, f"Expected {expected}, got {required}"

    def test_get_required_strata_physics(self):
        """
        Verifica que get_required_strata(PHYSICS) retorne set().
        PHYSICS=3.
        Required: value > 3 -> {}.
        """
        registry = MICRegistry()
        required = registry.get_required_strata(Stratum.PHYSICS)

        expected = set()
        assert required == expected, f"Expected empty set, got {required}"

    def test_get_required_strata_wisdom(self):
        """
        Verifica que get_required_strata(WISDOM) retorne {STRATEGY, TACTICS, PHYSICS}.
        WISDOM=0.
        Required: value > 0 -> {1, 2, 3}.
        """
        registry = MICRegistry()
        required = registry.get_required_strata(Stratum.WISDOM)

        expected = {Stratum.STRATEGY, Stratum.TACTICS, Stratum.PHYSICS}
        assert required == expected, f"Expected full set, got {required}"

    def test_compatibility_with_compute_required_strata(self):
        """
        Verifica que _compute_required_strata delegue correctamente.
        """
        registry = MICRegistry()
        target = Stratum.STRATEGY

        # Accessing private method for verification as per instructions to ensure it was updated
        internal_result = registry._compute_required_strata(target)
        public_result = registry.get_required_strata(target)

        assert internal_result == public_result

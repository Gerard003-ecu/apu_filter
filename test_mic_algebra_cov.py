from app.core.mic_algebra import Morphism, CategoricalState, ComposedMorphism, Stratum
from typing import FrozenSet

class MockMorphism(Morphism):
    def __init__(self, name: str, domain_val: FrozenSet[Stratum], codomain_val: Stratum):
        super().__init__(name)
        self._domain = domain_val
        self._codomain = codomain_val

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return self._domain

    @property
    def codomain(self) -> Stratum:
        return self._codomain

    def __call__(self, state: CategoricalState) -> CategoricalState:
        return state.with_update(new_context={"run_" + self.name: True}, new_stratum=self.codomain)

f = MockMorphism("f", frozenset([Stratum.PHYSICS]), Stratum.TACTICS)
g = MockMorphism("g", frozenset([Stratum.TACTICS]), Stratum.STRATEGY)

comp = f >> g
state = CategoricalState(context={"exergy_level": 1.0})
res = comp(state)
print("Friction contextualizada?", res.context)

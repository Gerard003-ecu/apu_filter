import pytest
import numpy as np
from app.boole.wisdom.semantic_validator import (
    SemanticValidationEngine,
    BusinessPurpose,
    LLMOutput,
    RiskProfile,
    Verdict
)

class TestGaloisConnectionCertification:
    r"""
    Certificación rigurosa de la Conexión de Galois $f \dashv g$ entre
    el retículo de severidades estructurales $S$ y veredictos ontológicos $V$.

    AXIOMAS DE ADJUNCIÓN:
    1. Monotonía: $x_1 \le_S x_2 \implies f(x_1) \le_V f(x_2)$
    2. Adjunción: $f(x) \le_V y \iff x \le_S g(y)$
    3. Preservación del Supremo: $f(x_1 \vee x_2) = f(x_1) \vee f(x_2)$
    """

    def setup_method(self):
        self.kg = {"caching": {"LATENCY_REDUCTION": 1.0}}
        # Perfil muy permisivo para ver transiciones claras en el rango [0, 10]
        self.profile = RiskProfile(risk_tolerance=1.0, domain_criticality=0.0)
        self.engine = SemanticValidationEngine(
            knowledge_graph=self.kg,
            risk_profile=self.profile
        )

    def f(self, entropy: float, confidence: float) -> Verdict:
        """Funtor f: S -> V (mapeo de severidad a veredicto)."""
        llm_output = LLMOutput(entropy=entropy, confidence=confidence)
        purpose = BusinessPurpose(
            concept="caching",
            business_problem="LATENCY_REDUCTION",
            strength=0.9,
            confidence=1.0
        )
        result = self.engine.validate(purposes=[purpose], llm_output=llm_output)
        return result.verdict

    def g(self, verdict: Verdict) -> float:
        """
        Funtor adjunto derecho g: V -> S (umbral de severidad).
        Representamos S por el espacio de entropía (fijando confianza=1.0).
        """
        # g(y) es el máximo x tal que f(x) <= y
        # Estimación numérica por búsqueda binaria sobre entropía
        low, high = 0.0, 50.0
        for _ in range(30):
            mid = (low + high) / 2
            if self.f(mid, 1.0) <= verdict:
                low = mid
            else:
                high = mid
        return low

    @pytest.mark.property_based
    def test_galois_adjunction_property(self):
        """
        Verifica f(x) ≤_V y ⟺ x ≤_S g(y)
        """
        verdicts = list(Verdict)
        # Evitar puntos muy cercanos a los saltos de escalón
        entropies = np.linspace(0.0, 10.0, 40)

        for x in entropies:
            for y in verdicts:
                fx = self.f(x, 1.0)
                gy = self.g(y)

                # Equivalencia de la conexión de Galois
                lhs = fx <= y
                # Usamos una tolerancia mayor para evitar problemas en las discontinuidades del veredicto
                rhs = x <= gy + 1e-5

                assert lhs == rhs, (
                    f"Violación de Galois: x={x:.4f}, y={y.name}, "
                    f"f(x)={fx.name} <= {y.name} is {lhs}, "
                    f"x <= g(y)={gy:.4f} is {rhs}"
                )

    @pytest.mark.property_based
    def test_join_preservation_property(self):
        """
        Verifica preservación del supremo (homomorfismo de unión):
        φ(x₁ ∨ x₂) = φ(x₁) ∨ φ(x₂)
        """
        entropies = np.linspace(0.0, 10.0, 20)

        for i in range(len(entropies)):
            for j in range(len(entropies)):
                x1 = entropies[i]
                x2 = entropies[j]

                # En S (entropías), x1 ∨ x2 es max(x1, x2)
                sup_s = max(x1, x2)
                phi_sup_s = self.f(sup_s, 1.0)

                # φ(x1) ∨ φ(x2) en V
                phi_x1 = self.f(x1, 1.0)
                phi_x2 = self.f(x2, 1.0)
                sup_v = Verdict(max(phi_x1.value, phi_x2.value))

                assert phi_sup_s == sup_v, (
                    f"Fallo en preservación de unión: x1={x1:.4f}, x2={x2:.4f}, "
                    f"phi(x1 ∨ x2)={phi_sup_s.name}, "
                    f"phi(x1) ∨ phi(x2)={sup_v.name}"
                )

if __name__ == "__main__":
    pytest.main([__file__])

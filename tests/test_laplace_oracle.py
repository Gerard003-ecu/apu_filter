import pytest
import math
from app.laplace_oracle import LaplaceOracle, ConfigurationError

class TestLaplaceOracle:
    """Test suite for the LaplaceOracle."""

    def test_initialization_stable(self):
        """Test initialization with stable parameters."""
        oracle = LaplaceOracle(R=10.0, L=2.0, C=5000.0)
        assert oracle.R == 10.0
        assert oracle.L == 2.0
        assert oracle.C == 5000.0
        assert oracle.omega_n > 0
        assert oracle.zeta > 0

    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            LaplaceOracle(R=-10.0, L=2.0, C=5000.0)
        with pytest.raises(ConfigurationError):
            LaplaceOracle(R=10.0, L=-2.0, C=5000.0)
        with pytest.raises(ConfigurationError):
            LaplaceOracle(R=10.0, L=2.0, C=-5000.0)

    def test_laplace_pyramid_structure(self):
        """Test that get_laplace_pyramid returns the correct 4-level structure."""
        oracle = LaplaceOracle(R=10.0, L=2.0, C=5000.0)
        pyramid = oracle.get_laplace_pyramid()

        assert "level_0_verdict" in pyramid
        assert "level_1_robustness" in pyramid
        assert "level_2_dynamics" in pyramid
        assert "level_3_physics" in pyramid

        # Level 0 Check
        verdict = pyramid["level_0_verdict"]
        assert isinstance(verdict["is_controllable"], bool)
        assert isinstance(verdict["stability_status"], str)

        # Level 1 Check
        robustness = pyramid["level_1_robustness"]
        assert "phase_margin_deg" in robustness
        assert "gain_margin_db" in robustness

        # Level 2 Check
        dynamics = pyramid["level_2_dynamics"]
        assert "natural_frequency_rad_s" in dynamics
        assert "poles_continuous" in dynamics

        # Level 3 Check
        physics = pyramid["level_3_physics"]
        assert physics["R"] == 10.0
        assert physics["L"] == 2.0
        assert physics["C"] == 5000.0

    def test_stability_analysis(self):
        """Test stability analysis for a known stable system."""
        oracle = LaplaceOracle(R=10.0, L=2.0, C=5000.0)
        stability = oracle.analyze_stability()
        assert stability["status"] == "STABLE"
        assert stability["is_stable"] is True

    def test_marginal_stability(self):
        """Test marginal stability (zero resistance)."""
        # R=0 implies no damping, marginally stable
        oracle = LaplaceOracle(R=0.0, L=2.0, C=5000.0)
        stability = oracle.analyze_stability()
        assert stability["status"] == "MARGINALLY_STABLE"
        assert stability["is_marginally_stable"] is True

    def test_control_recommendations(self):
        """Test generation of control recommendations."""
        # Unstable/Fragile system configuration
        # Low damping ratio
        oracle = LaplaceOracle(R=0.1, L=10.0, C=10.0)  # Very low R -> low zeta
        validation = oracle.validate_for_control_design()

        assert "is_suitable_for_control" in validation
        # Should have warnings or recommendations
        assert len(validation["recommendations"]) > 0

        # Check specific recommendation content
        recs = " ".join(validation["recommendations"])
        assert "subamortiguado" in recs.lower()

    def test_frequency_response(self):
        """Test frequency response calculation."""
        oracle = LaplaceOracle(R=10.0, L=2.0, C=5000.0)
        freq_resp = oracle.get_frequency_response()

        assert "frequencies_rad_s" in freq_resp
        assert "magnitude_db" in freq_resp
        assert "phase_deg" in freq_resp
        assert len(freq_resp["frequencies_rad_s"]) > 0

    def test_root_locus(self):
        """Test root locus data generation."""
        oracle = LaplaceOracle(R=10.0, L=2.0, C=5000.0)
        rl_data = oracle.get_root_locus_data()

        assert "poles_real" in rl_data
        assert "poles_imag" in rl_data
        assert len(rl_data["poles_real"]) > 0

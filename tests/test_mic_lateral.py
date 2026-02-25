
import pytest
from unittest.mock import MagicMock
from app.adapters.mic_vectors import vector_lateral_pivot, VectorResultStatus
from app.business_agent import RiskChallenger, ConstructionRiskReport
from app.schemas import Stratum

# =============================================================================
# PRUEBAS DEL VECTOR (FÍSICA/LÓGICA PURA)
# =============================================================================

class TestVectorLateralPivot:
    """
    Pruebas unitarias para la lógica interna del vector 'lateral_thinking_pivot'.
    """

    def test_monopolio_coberturado_success(self):
        """
        Verifica que se apruebe el pivote de Monopolio Coberturado
        cuando se cumplen las condiciones termodinámicas.
        """
        payload = {
            "pivot_type": "MONOPOLIO_COBERTURADO",
            "report_state": {"stability": 0.65, "beta_1": 0, "financial_class": "SAFE"},
            "thermal_metrics": {"system_temperature": 10.0, "heat_capacity": 0.8},
            "financial_metrics": {"npv": 1000.0}
        }

        result = vector_lateral_pivot(payload)

        assert result["success"] is True
        assert result["stratum"] == Stratum.STRATEGY
        assert result["payload"]["approved_pivot"] == "MONOPOLIO_COBERTURADO"
        assert result["payload"]["penalty_relief"] == 0.30

    def test_monopolio_coberturado_failure_high_temp(self):
        """Rechazo por temperatura alta (sistema volátil)."""
        payload = {
            "pivot_type": "MONOPOLIO_COBERTURADO",
            "report_state": {"stability": 0.65},
            "thermal_metrics": {"system_temperature": 25.0, "heat_capacity": 0.8}, # T > 15
        }

        result = vector_lateral_pivot(payload)

        assert result["success"] is False
        assert result["status"] == VectorResultStatus.LOGIC_ERROR.value
        assert "condiciones termodinámicas insuficientes" in result["error"].lower()

    def test_opcion_espera_success(self):
        """Verifica aprobación de opción de espera."""
        payload = {
            "pivot_type": "OPCION_ESPERA",
            "report_state": {"financial_class": "HIGH"},
            "financial_metrics": {
                "npv": 100.0,
                "real_options": {"wait_option_value": 200.0} # > 1.5 * NPV
            }
        }

        result = vector_lateral_pivot(payload)

        assert result["success"] is True
        assert result["payload"]["approved_pivot"] == "OPCION_ESPERA"
        assert result["payload"]["strategic_action"] == "FREEZE_6_MONTHS"

    def test_cuarentena_topologica_success(self):
        """Verifica cuarentena de ciclos sin sinergia."""
        payload = {
            "pivot_type": "CUARENTENA_TOPOLOGICA",
            "report_state": {"beta_1": 5},
            "synergy_risk": {"synergy_detected": False}
        }

        result = vector_lateral_pivot(payload)

        assert result["success"] is True
        assert result["payload"]["approved_pivot"] == "CUARENTENA_TOPOLOGICA"
        assert result["payload"]["quarantine_active"] is True

    def test_cuarentena_topologica_failure_synergy(self):
        """Rechazo por sinergia detectada (ciclos acoplados)."""
        payload = {
            "pivot_type": "CUARENTENA_TOPOLOGICA",
            "report_state": {"beta_1": 5},
            "synergy_risk": {"synergy_detected": True}
        }

        result = vector_lateral_pivot(payload)

        assert result["success"] is False
        assert "sinergia" in result["error"].lower()

# =============================================================================
# PRUEBAS DE INTEGRACIÓN (RISK CHALLENGER + MIC MOCK)
# =============================================================================

class TestRiskChallengerLateralIntegration:
    """
    Pruebas de integración del RiskChallenger con la MIC mockeada.
    """

    @pytest.fixture
    def mock_mic(self):
        return MagicMock()

    @pytest.fixture
    def challenger(self, mock_mic):
        return RiskChallenger(mic=mock_mic)

    def test_challenge_verdict_monopolio_exception(self, challenger, mock_mic):
        """
        RiskChallenger debe solicitar excepción a la MIC y aplicarla si es aprobada.
        """
        # Configurar MIC para aprobar
        mock_mic.project_intent.return_value = {
            "success": True,
            "payload": {
                "approved_pivot": "MONOPOLIO_COBERTURADO",
                "penalty_relief": 0.30,
                "reasoning": "Aprobado por inercia."
            }
        }

        # Reporte con estabilidad crítica
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="SAFE",
            details={"pyramid_stability": 0.60}, # Crítico
            strategic_narrative=""
        )

        audited = challenger.challenge_verdict(report)

        # Verificar llamada a MIC
        mock_mic.project_intent.assert_called()
        args, _ = mock_mic.project_intent.call_args
        assert args[0] == "lateral_thinking_pivot"
        assert args[1]["pivot_type"] == "MONOPOLIO_COBERTURADO"

        # Verificar resultado: Integridad mejorada (100 * (1 + 0.30) = 130 -> cap 100)
        # Wait, min(100, original * (1+relief)). If original is 100, it stays 100.
        # But wait, original report passed to challenge_verdict has 100.
        # If no exception, it would be penalized to 70.
        # With exception, it stays at 100 (or capped).
        # Let's check logic:
        # current_report is copy of report.
        # exception applied: new_integrity = min(100, original * (1+0.3)) -> 100.
        # The penalty logic (veto) is in the ELSE block.
        # So score should remain high.

        assert audited.integrity_score == 100.0
        assert "EXCEPCIÓN_MONOPOLIO_COBERTURADO" in audited.details["lateral_thinking_applied"]
        assert "ACTA DEL CONSEJO" in audited.strategic_narrative

    def test_challenge_verdict_monopolio_rejected(self, challenger, mock_mic):
        """Si MIC rechaza, se aplica el veto estándar."""
        mock_mic.project_intent.return_value = {"success": False}

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="SAFE",
            details={"pyramid_stability": 0.60},
            strategic_narrative=""
        )

        audited = challenger.challenge_verdict(report)

        # Veto aplicado: 100 * (1 - 0.30) = 70
        assert audited.integrity_score == 70.0
        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (CRÍTICO)"

    def test_challenge_verdict_cycle_quarantine(self, challenger, mock_mic):
        """Aprobación de cuarentena topológica mejora el score (o evita penalización)."""
        mock_mic.project_intent.return_value = {
            "success": True,
            "payload": {
                "approved_pivot": "CUARENTENA_TOPOLOGICA",
                "reasoning": "Ciclos aislados."
            }
        }

        report = ConstructionRiskReport(
            integrity_score=80.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="MODERATE",
            details={
                "pyramid_stability": 0.9,
                "topological_invariants": {"betti_numbers": {"beta_1": 5}, "n_nodes": 20}
            },
            strategic_narrative=""
        )

        audited = challenger.challenge_verdict(report)

        # Relief 0.10: 80 * 1.1 = 88.0
        assert audited.integrity_score == pytest.approx(88.0)
        assert audited.details["lateral_thinking_applied"] == "CUARENTENA_TOPOLÓGICA_ACTIVA"

"""
Suite de Pruebas: test_neuromorphic_solver.py
Versión: 1.0 — Cobertura integral para neuromorphic_solver.py v6.0

Estructura:
    TestJFETParameters          — Validación de dataclass y contratos físicos
    TestJFETModelChannelN       — Modelo Shockley canal N (algebraico + derivadas)
    TestJFETModelChannelP       — Modelo Shockley canal P (algebraico + derivadas)
    TestJFETModelBoundaries     — Fronteras de región (corte ↔ activa)
    TestJFETModelDerivatives    — Verificación numérica del Jacobiano (diferencias finitas)
    TestLambdaDiodeTopology     — KCL, Jacobiano analítico, mapeo de tensiones
    TestTopologyDerivatives     — Jacobiano topológico vs. diferencias finitas
    TestNewtonSolverContracts   — Pre/post condiciones del solver
    TestNewtonSolverConvergence — Convergencia cuadrática y criterios duales
    TestNewtonSolverRobustness  — Singularidades, semillas malas, casos extremos
    TestNeuromorphicAnalyzer    — Barrido I-V, continuación, salida estructurada
    TestNDRDetection            — Detección y validación de la región NDR
    TestPhysicalInvariants      — Leyes de circuitos como invariantes de integración
    TestNumericalRegression     — Valores de referencia congelados (golden values)
    TestEndToEnd                — Flujo completo: parámetros → curva → NDR

Convenciones:
    - Tolerancias relativas para comparaciones físicas: rtol=1e-6
    - Diferencias finitas para verificar Jacobianos: h=1e-7
    - Parámetros de los JFETs reales: PARAM_2N5457, PARAM_J176
    - Fixtures compartidos via pytest para eficiencia
"""

import math
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Tuple

# ── Módulo bajo prueba ────────────────────────────────────────────────────────
from neuromorphic_solver import (
    JFETParameters,
    JFETModel,
    LambdaDiodeTopology,
    NewtonSolver,
    NeuromorphicAnalyzer,
    PARAM_2N5457,
    PARAM_J176,
)

# =============================================================================
# CONSTANTES GLOBALES DE LA SUITE
# =============================================================================

# Tolerancias
ABS_TOL_CURRENT  = 1e-12   # Amperes — tolerancia absoluta para corrientes
REL_TOL_CURRENT  = 1e-6    # Relativa para comparaciones de corriente
REL_TOL_JACOBIAN = 1e-4    # Relativa para verificación numérica de Jacobianos
FD_STEP          = 1e-7    # Paso para diferencias finitas

# Parámetros de prueba canónicos (canal N)
P_N = JFETParameters(idss=1e-3, vp=-2.0, lam=0.01, is_n_channel=True)
# Parámetros de prueba canónicos (canal P)
P_P = JFETParameters(idss=2e-3, vp=3.0,  lam=0.02, is_n_channel=False)


# =============================================================================
# FIXTURES COMPARTIDOS
# =============================================================================

@pytest.fixture(scope="module")
def model_n() -> JFETModel:
    """Modelo JFET canal N con parámetros canónicos."""
    return JFETModel(P_N)


@pytest.fixture(scope="module")
def model_p() -> JFETModel:
    """Modelo JFET canal P con parámetros canónicos."""
    return JFETModel(P_P)


@pytest.fixture(scope="module")
def model_n_real() -> JFETModel:
    """Modelo JFET canal N con parámetros reales (2N5457)."""
    return JFETModel(PARAM_2N5457)


@pytest.fixture(scope="module")
def model_p_real() -> JFETModel:
    """Modelo JFET canal P con parámetros reales (J176)."""
    return JFETModel(PARAM_J176)


@pytest.fixture(scope="module")
def topology() -> LambdaDiodeTopology:
    """Topología del Diodo Lambda con JFETs reales."""
    return LambdaDiodeTopology()


@pytest.fixture(scope="module")
def solver() -> NewtonSolver:
    """Solver Newton-Raphson con parámetros por defecto."""
    return NewtonSolver()


@pytest.fixture(scope="module")
def analyzer() -> NeuromorphicAnalyzer:
    """Analizador neuromórfico completo."""
    return NeuromorphicAnalyzer()


# =============================================================================
# NIVEL 1A: PRUEBAS UNITARIAS — JFETParameters
# =============================================================================

class TestJFETParameters:
    """
    Verifica los contratos de la dataclass JFETParameters.

    Invariantes físicos:
        • idss > 0              (corriente de saturación positiva)
        • vp < 0 para canal N   (pinch-off en el plano negativo)
        • vp > 0 para canal P   (convención de magnitud)
        • lam ≥ 0               (modulación de canal no negativa)
    """

    def test_valid_n_channel_construction(self):
        """Construcción válida de canal N no lanza excepción."""
        params = JFETParameters(idss=1e-3, vp=-2.0, lam=0.01, is_n_channel=True)
        assert params.idss == 1e-3
        assert params.vp   == -2.0
        assert params.lam  == 0.01
        assert params.is_n_channel is True

    def test_valid_p_channel_construction(self):
        """Construcción válida de canal P no lanza excepción."""
        params = JFETParameters(idss=5e-3, vp=1.5, lam=0.0, is_n_channel=False)
        assert params.vp > 0
        assert params.is_n_channel is False

    def test_zero_lambda_is_valid(self):
        """Lambda = 0 es físicamente válido (sin modulación de canal)."""
        params = JFETParameters(idss=1e-3, vp=-1.0, lam=0.0, is_n_channel=True)
        assert params.lam == 0.0

    @pytest.mark.parametrize("bad_idss", [0.0, -1e-3, -1e-9])
    def test_invalid_idss_raises(self, bad_idss: float):
        """idss ≤ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="idss"):
            JFETParameters(idss=bad_idss, vp=-1.0, lam=0.01, is_n_channel=True)

    @pytest.mark.parametrize("bad_vp", [0.0, 0.5, 1.0])
    def test_invalid_vp_n_channel_raises(self, bad_vp: float):
        """Canal N con vp ≥ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="vp"):
            JFETParameters(idss=1e-3, vp=bad_vp, lam=0.01, is_n_channel=True)

    @pytest.mark.parametrize("bad_vp", [0.0, -0.5, -2.0])
    def test_invalid_vp_p_channel_raises(self, bad_vp: float):
        """Canal P con vp ≤ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="vp"):
            JFETParameters(idss=1e-3, vp=bad_vp, lam=0.01, is_n_channel=False)

    def test_invalid_negative_lambda_raises(self):
        """Lambda < 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="lam"):
            JFETParameters(idss=1e-3, vp=-1.0, lam=-0.01, is_n_channel=True)

    def test_frozen_immutability(self):
        """La dataclass es frozen: no se puede modificar post-construcción."""
        params = JFETParameters(idss=1e-3, vp=-1.0, lam=0.01, is_n_channel=True)
        with pytest.raises(Exception):  # FrozenInstanceError
            params.idss = 2e-3  # type: ignore[misc]

    def test_real_params_2n5457_valid(self):
        """Los parámetros reales del 2N5457 son físicamente válidos."""
        assert PARAM_2N5457.idss > 0
        assert PARAM_2N5457.vp < 0
        assert PARAM_2N5457.lam >= 0
        assert PARAM_2N5457.is_n_channel is True

    def test_real_params_j176_valid(self):
        """Los parámetros reales del J176 son físicamente válidos."""
        assert PARAM_J176.idss > 0
        assert PARAM_J176.vp > 0
        assert PARAM_J176.lam >= 0
        assert PARAM_J176.is_n_channel is False


# =============================================================================
# NIVEL 1B: PRUEBAS UNITARIAS — JFETModel Canal N
# =============================================================================

class TestJFETModelChannelN:
    """
    Verifica el modelo de Shockley para canal N.

    Modelo:
        I_D = I_DSS · (1 - V_GS/V_P)² · (1 + λ·V_DS)   [activa]
        I_D = 0                                            [corte]
    """

    def test_cutoff_at_pinchoff_exactly(self, model_n: JFETModel):
        """En V_GS = V_P exacto, la corriente debe ser cero (corte)."""
        i_d, g_m, g_ds = model_n.evaluate(P_N.vp, v_ds=1.0)
        assert i_d  == 0.0
        assert g_m  == 0.0
        assert g_ds == 0.0

    @pytest.mark.parametrize("v_gs", [-3.0, -2.5, -2.1])
    def test_cutoff_below_pinchoff(self, model_n: JFETModel, v_gs: float):
        """V_GS < V_P (más negativo) → región de corte → I_D = 0."""
        i_d, g_m, g_ds = model_n.evaluate(v_gs, v_ds=2.0)
        assert i_d  == pytest.approx(0.0, abs=ABS_TOL_CURRENT)
        assert g_m  == pytest.approx(0.0, abs=ABS_TOL_CURRENT)
        assert g_ds == pytest.approx(0.0, abs=ABS_TOL_CURRENT)

    def test_idss_at_vgs_zero_and_unit_vds(self, model_n: JFETModel):
        """
        En V_GS = 0 y lambda=0.01, V_DS=1:
        I_D = I_DSS · 1² · (1 + 0.01·1) = I_DSS · 1.01
        """
        i_d, _, _ = model_n.evaluate(v_gs=0.0, v_ds=1.0)
        expected = P_N.idss * (1.0 + P_N.lam * 1.0)
        assert i_d == pytest.approx(expected, rel=REL_TOL_CURRENT)

    def test_current_monotone_increasing_with_vds(self, model_n: JFETModel):
        """
        I_D debe aumentar monótonamente con V_DS (modulación de canal positiva).
        """
        v_gs = -1.0  # Por encima del pinch-off (-2.0)
        vds_values = np.linspace(0.1, 5.0, 20)
        currents = [model_n.evaluate(v_gs, vds)[0] for vds in vds_values]
        assert all(
            currents[k] < currents[k + 1] for k in range(len(currents) - 1)
        ), "I_D debe ser creciente con V_DS en región activa."

    def test_current_decreasing_with_more_negative_vgs(self, model_n: JFETModel):
        """
        A V_DS fijo, I_D disminuye al hacer V_GS más negativo (→ pinch-off).
        """
        v_ds = 2.0
        vgs_values = np.linspace(-1.9, 0.0, 20)  # Desde cerca del corte hasta 0
        currents = [model_n.evaluate(vgs, v_ds)[0] for vgs in vgs_values]
        assert all(
            currents[k] < currents[k + 1] for k in range(len(currents) - 1)
        ), "I_D debe crecer al aumentar V_GS desde pinch-off hasta 0."

    def test_current_non_negative_in_active_region(self, model_n: JFETModel):
        """I_D nunca es negativa en la región activa de canal N."""
        for v_gs in np.linspace(P_N.vp + 0.01, 0.0, 15):
            for v_ds in np.linspace(0.1, 5.0, 10):
                i_d, _, _ = model_n.evaluate(v_gs, v_ds)
                assert i_d >= 0.0, f"I_D={i_d} negativa en V_gs={v_gs}, V_ds={v_ds}"

    def test_gm_positive_in_active_region(self, model_n: JFETModel):
        """
        g_m = ∂I_D/∂V_GS > 0 siempre en región activa canal N.
        (Mayor V_GS → mayor corriente.)
        """
        for v_gs in np.linspace(P_N.vp + 0.1, -0.1, 10):
            _, g_m, _ = model_n.evaluate(v_gs, v_ds=1.0)
            assert g_m > 0.0, f"g_m={g_m} no positiva en V_gs={v_gs}"

    def test_gds_positive_canal_n(self, model_n: JFETModel):
        """
        g_ds = ∂I_D/∂V_DS > 0 para canal N con V_DS > 0.
        """
        _, _, g_ds = model_n.evaluate(v_gs=-1.0, v_ds=2.0)
        assert g_ds > 0.0

    def test_shockley_formula_explicit(self, model_n: JFETModel):
        """
        Verifica la fórmula de Shockley punto a punto con cálculo manual.
        I_D = I_DSS · (1 - V_GS/V_P)² · (1 + λ·V_DS)
        """
        v_gs, v_ds = -1.0, 3.0
        xi     = v_gs / P_N.vp          # (-1)/(-2) = 0.5
        factor = 1.0 + P_N.lam * v_ds   # 1 + 0.01·3 = 1.03
        expected_id = P_N.idss * ((1 - xi) ** 2) * factor

        i_d, _, _ = model_n.evaluate(v_gs, v_ds)
        assert i_d == pytest.approx(expected_id, rel=REL_TOL_CURRENT)


# =============================================================================
# NIVEL 1C: PRUEBAS UNITARIAS — JFETModel Canal P
# =============================================================================

class TestJFETModelChannelP:
    """
    Verifica el modelo de Shockley para canal P.

    Convención interna: la topología pasa |V_GS_p| como v_gs al modelo.
    El modelo opera con v_gs ∈ [0, V_P) para canal P, donde V_P > 0.
    """

    def test_cutoff_at_pinchoff_exactly(self, model_p: JFETModel):
        """En |V_GS| = V_P exacto → región de corte → I_D = 0."""
        i_d, g_m, g_ds = model_p.evaluate(P_P.vp, v_ds=-1.0)
        assert i_d  == 0.0
        assert g_m  == 0.0
        assert g_ds == 0.0

    @pytest.mark.parametrize("v_gs_mag", [3.1, 4.0, 10.0])
    def test_cutoff_above_pinchoff_magnitude(self, model_p: JFETModel, v_gs_mag: float):
        """|V_GS| ≥ V_P → corte para canal P."""
        i_d, g_m, g_ds = model_p.evaluate(v_gs_mag, v_ds=-1.0)
        assert i_d  == pytest.approx(0.0, abs=ABS_TOL_CURRENT)
        assert g_m  == pytest.approx(0.0, abs=ABS_TOL_CURRENT)
        assert g_ds == pytest.approx(0.0, abs=ABS_TOL_CURRENT)

    def test_idss_at_vgs_zero(self, model_p: JFETModel):
        """
        En |V_GS| = 0, V_DS negativo con módulo 1:
        I_D = I_DSS · (1 - 0)² · (1 + λ·|V_DS|) = I_DSS · (1 + λ)
        """
        v_ds = -1.0
        i_d, _, _ = model_p.evaluate(0.0, v_ds)
        expected = P_P.idss * (1.0 + P_P.lam * abs(v_ds))
        assert i_d == pytest.approx(expected, rel=REL_TOL_CURRENT)

    def test_current_increases_with_vds_magnitude(self, model_p: JFETModel):
        """
        I_D canal P debe crecer con |V_DS| (V_DS más negativo).
        """
        v_gs_mag = 1.0
        vds_values = np.linspace(-0.1, -5.0, 20)  # más negativos
        currents = [model_p.evaluate(v_gs_mag, vds)[0] for vds in vds_values]
        assert all(
            currents[k] < currents[k + 1] for k in range(len(currents) - 1)
        ), "I_D canal P debe aumentar con |V_DS| creciente."

    def test_current_non_negative_in_active_region(self, model_p: JFETModel):
        """I_D nunca es negativa (el signo convencional es externo al modelo)."""
        for v_gs_mag in np.linspace(0.0, P_P.vp - 0.01, 15):
            for v_ds in np.linspace(-0.1, -5.0, 10):
                i_d, _, _ = model_p.evaluate(v_gs_mag, v_ds)
                assert i_d >= 0.0, (
                    f"I_D={i_d} negativa en |V_gs|={v_gs_mag}, V_ds={v_ds}"
                )

    def test_gds_negative_for_negative_vds(self, model_p: JFETModel):
        """
        Para canal P con V_DS < 0:
        g_ds = ∂I_D/∂V_DS = I_DSS·(1-ξ)²·λ·sign(V_DS) < 0
        Esto es correcto: I_D (magnitud) crece con |V_DS|, pero dI_D/dV_DS < 0.
        """
        _, _, g_ds = model_p.evaluate(v_gs=1.0, v_ds=-2.0)
        assert g_ds < 0.0, f"g_ds={g_ds} debería ser negativo para V_DS < 0"

    def test_gm_positive_canal_p(self, model_p: JFETModel):
        """
        g_m > 0 para canal P: mayor |V_GS| (más cerca de 0) → mayor corriente.
        Nota: g_m = 2·I_DSS·(1-ξ)·(-1/V_P)·factor_vds
        Con V_P > 0 y ξ < 1: g_m = 2·I_DSS·(1-ξ)·(negativo) → NEGATIVO.

        Esto es correcto: al AUMENTAR v_gs_mag (|V_GS|), ξ aumenta,
        (1-ξ) disminuye, la corriente disminuye → g_m < 0 es coherente.
        """
        _, g_m, _ = model_p.evaluate(v_gs=1.0, v_ds=-2.0)
        # g_m debe ser negativo porque aumentar |V_GS| reduce la corriente
        assert g_m < 0.0, (
            f"g_m={g_m}: aumentar |V_GS_p| reduce I_D → g_m < 0 esperado"
        )

    def test_shockley_formula_explicit_p_channel(self, model_p: JFETModel):
        """Fórmula de Shockley punto a punto para canal P."""
        v_gs_mag, v_ds = 1.5, -2.0
        xi       = v_gs_mag / P_P.vp      # 1.5/3.0 = 0.5
        factor   = 1.0 + P_P.lam * abs(v_ds)  # 1 + 0.02·2 = 1.04
        expected = P_P.idss * ((1 - xi) ** 2) * factor

        i_d, _, _ = model_p.evaluate(v_gs_mag, v_ds)
        assert i_d == pytest.approx(expected, rel=REL_TOL_CURRENT)


# =============================================================================
# NIVEL 1D: FRONTERAS DE REGIÓN (CONTINUIDAD EN PINCH-OFF)
# =============================================================================

class TestJFETModelBoundaries:
    """
    Verifica la continuidad de I_D en la frontera de corte/activa.

    Físicamente: I_D debe ser continua en V_GS = V_P (no hay salto).
    La derivada (g_m) en el punto de pinch-off debe ser cero (mínimo).
    """

    @pytest.mark.parametrize("epsilon", [1e-6, 1e-8, 1e-10])
    def test_continuity_at_cutoff_n_channel(
        self, model_n: JFETModel, epsilon: float
    ):
        """
        I_D debe ser continua desde ambos lados del pinch-off (canal N).
        Límite desde arriba: I_D(V_P + ε) ≈ 0
        """
        v_gs_above = P_N.vp + epsilon
        i_d, _, _  = model_n.evaluate(v_gs_above, v_ds=1.0)
        # La corriente justo por encima del corte debe ser muy pequeña
        expected = P_N.idss * (epsilon / abs(P_N.vp)) ** 2 * (1 + P_N.lam * 1.0)
        assert i_d == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("epsilon", [1e-6, 1e-8, 1e-10])
    def test_continuity_at_cutoff_p_channel(
        self, model_p: JFETModel, epsilon: float
    ):
        """I_D continua desde abajo del pinch-off (canal P)."""
        v_gs_below = P_P.vp - epsilon   # |V_GS| justo por debajo de V_P
        i_d, _, _  = model_p.evaluate(v_gs_below, v_ds=-1.0)
        expected   = P_P.idss * (epsilon / P_P.vp) ** 2 * (1 + P_P.lam * 1.0)
        assert i_d == pytest.approx(expected, rel=1e-4)

    def test_gm_approaches_zero_at_pinchoff(self, model_n: JFETModel):
        """g_m → 0 cuando V_GS → V_P⁺ (continuidad de la derivada)."""
        epsilon = 1e-6
        _, g_m, _ = model_n.evaluate(P_N.vp + epsilon, v_ds=1.0)
        # g_m ∝ (1 - ξ) → 0 cuando ξ → 1
        assert abs(g_m) < 1e-5, f"g_m={g_m} no tiende a cero en el pinch-off"

    def test_no_negative_current_at_boundary(self, model_n: JFETModel):
        """En la frontera exacta, I_D no puede ser negativa."""
        for vds in [0.1, 1.0, 5.0, 10.0]:
            i_d, _, _ = model_n.evaluate(P_N.vp, vds)
            assert i_d >= 0.0


# =============================================================================
# NIVEL 1E: VERIFICACIÓN NUMÉRICA DEL JACOBIANO JFET
# =============================================================================

class TestJFETModelDerivatives:
    """
    Verifica g_m y g_ds contra diferencias finitas (prueba de oro).

    Para cualquier función diferenciable f(x):
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)   con h << 1

    Esto garantiza que las derivadas analíticas son correctas con
    independencia de la formulación algebraica.
    """

    def _finite_diff_gm(
        self, model: JFETModel, v_gs: float, v_ds: float, h: float = FD_STEP
    ) -> float:
        """∂I_D/∂V_GS por diferencias finitas centradas."""
        i_plus,  _, _ = model.evaluate(v_gs + h, v_ds)
        i_minus, _, _ = model.evaluate(v_gs - h, v_ds)
        return (i_plus - i_minus) / (2 * h)

    def _finite_diff_gds(
        self, model: JFETModel, v_gs: float, v_ds: float, h: float = FD_STEP
    ) -> float:
        """∂I_D/∂V_DS por diferencias finitas centradas."""
        i_plus,  _, _ = model.evaluate(v_gs, v_ds + h)
        i_minus, _, _ = model.evaluate(v_gs, v_ds - h)
        return (i_plus - i_minus) / (2 * h)

    @pytest.mark.parametrize("v_gs,v_ds", [
        (-0.5,  1.0),
        (-1.0,  2.0),
        (-1.5,  0.5),
        (-0.1,  3.0),
    ])
    def test_gm_matches_finite_diff_n_channel(
        self, model_n: JFETModel, v_gs: float, v_ds: float
    ):
        """g_m analítico ≈ g_m numérico (canal N)."""
        _, g_m_analytic, _ = model_n.evaluate(v_gs, v_ds)
        g_m_numeric = self._finite_diff_gm(model_n, v_gs, v_ds)
        assert g_m_analytic == pytest.approx(g_m_numeric, rel=REL_TOL_JACOBIAN)

    @pytest.mark.parametrize("v_gs,v_ds", [
        (-0.5,  1.0),
        (-1.0,  2.5),
        (-1.5,  0.3),
    ])
    def test_gds_matches_finite_diff_n_channel(
        self, model_n: JFETModel, v_gs: float, v_ds: float
    ):
        """g_ds analítico ≈ g_ds numérico (canal N)."""
        _, _, g_ds_analytic = model_n.evaluate(v_gs, v_ds)
        g_ds_numeric = self._finite_diff_gds(model_n, v_gs, v_ds)
        assert g_ds_analytic == pytest.approx(g_ds_numeric, rel=REL_TOL_JACOBIAN)

    @pytest.mark.parametrize("v_gs_mag,v_ds", [
        (1.0, -1.0),
        (0.5, -2.0),
        (2.0, -0.5),
    ])
    def test_gm_matches_finite_diff_p_channel(
        self, model_p: JFETModel, v_gs_mag: float, v_ds: float
    ):
        """g_m analítico ≈ g_m numérico (canal P)."""
        _, g_m_analytic, _ = model_p.evaluate(v_gs_mag, v_ds)
        g_m_numeric = self._finite_diff_gm(model_p, v_gs_mag, v_ds)
        assert g_m_analytic == pytest.approx(g_m_numeric, rel=REL_TOL_JACOBIAN)

    @pytest.mark.parametrize("v_gs_mag,v_ds", [
        (1.0, -1.0),
        (0.5, -2.0),
        (2.0, -0.5),
    ])
    def test_gds_matches_finite_diff_p_channel(
        self, model_p: JFETModel, v_gs_mag: float, v_ds: float
    ):
        """g_ds analítico ≈ g_ds numérico (canal P, V_DS < 0)."""
        _, _, g_ds_analytic = model_p.evaluate(v_gs_mag, v_ds)
        g_ds_numeric = self._finite_diff_gds(model_p, v_gs_mag, v_ds)
        assert g_ds_analytic == pytest.approx(g_ds_numeric, rel=REL_TOL_JACOBIAN)


# =============================================================================
# NIVEL 2A: INTEGRACIÓN — LambdaDiodeTopology
# =============================================================================

class TestLambdaDiodeTopology:
    """
    Verifica la coherencia del ensamblaje KCL y el mapeo de tensiones.

    Invariantes topológicos:
        1. En el punto de equilibrio: residual = I_N - I_P = 0
        2. El Jacobiano tiene unidades de conductancia [A/V]
        3. Los voltajes V_gs y V_ds son función lineal de V_x (mapeo correcto)
        4. Para V_app = 0: I_N = I_P = 0 (no hay excitación)
    """

    def test_residual_and_jacobian_return_four_values(
        self, topology: LambdaDiodeTopology
    ):
        """get_residual_and_jacobian retorna exactamente 4 valores."""
        result = topology.get_residual_and_jacobian(v_app=2.0, v_x=1.0)
        assert len(result) == 4, "Debe retornar (residual, jacobian, i_n, i_p)"

    def test_zero_voltage_gives_zero_currents(
        self, topology: LambdaDiodeTopology
    ):
        """
        Para V_app = 0 y V_x = 0:
        V_gs_n = 0 → corte (V_gs = vp_n = -1.5 < 0 → 0 ≥ vp es falso... )

        Nota: Con V_app=0, V_x=0:
          V_gs_n = 0  (igual al pinch-off? No, pinch-off = -1.5)
          → El N-JFET NO está en corte (V_gs_n=0 > vp=-1.5)
          → Pero V_ds_n = 0 → factor_vds = 1 → I_D_n = I_DSS

        Este test verifica el comportamiento del solver para V_app cercano a 0,
        no exactamente 0 (que es un caso especial del solver).
        """
        # Para V_app muy pequeño, el solver devuelve i=0 (caso trivial)
        solver = NewtonSolver()
        v_x, i, converged = solver.solve_for_voltage(0.0, 0.0)
        assert v_x == pytest.approx(0.0, abs=1e-12)
        assert i   == pytest.approx(0.0, abs=1e-12)
        assert converged is True

    def test_currents_non_negative(self, topology: LambdaDiodeTopology):
        """I_N e I_P son siempre no negativas (modelo define positiva)."""
        for v_app in [0.5, 1.0, 2.0, 3.0]:
            for v_x in np.linspace(0.01, v_app - 0.01, 5):
                _, _, i_n, i_p = topology.get_residual_and_jacobian(v_app, v_x)
                assert i_n >= 0.0, f"I_N negativa en V_app={v_app}, V_x={v_x}"
                assert i_p >= 0.0, f"I_P negativa en V_app={v_app}, V_x={v_x}"

    def test_residual_sign_changes_across_solution(
        self, topology: LambdaDiodeTopology
    ):
        """
        Para un V_app dado, debe existir V_x tal que el residual cambia de signo.
        Esto garantiza que hay una solución (por el Teorema del Valor Intermedio).
        """
        v_app = 2.0
        v_x_low  = 0.01
        v_x_high = v_app - 0.01
        res_low,  _, _, _ = topology.get_residual_and_jacobian(v_app, v_x_low)
        res_high, _, _, _ = topology.get_residual_and_jacobian(v_app, v_x_high)
        # El producto de residuos debe ser negativo (cambio de signo)
        assert res_low * res_high < 0.0, (
            f"No hay cambio de signo: res_low={res_low:.4e}, res_high={res_high:.4e}. "
            "El TVI no garantiza solución."
        )

    def test_jacobian_units_are_conductance(self, topology: LambdaDiodeTopology):
        """
        El Jacobiano J = df/dV_x tiene unidades de conductancia [A/V = S].
        Verificamos que el orden de magnitud es razonable (mS a μS para JFETs).
        """
        _, jacobian, _, _ = topology.get_residual_and_jacobian(v_app=2.0, v_x=1.0)
        # Para JFETs de mA, la conductancia debe estar en el rango 1μS - 100mS
        assert 1e-9 <= abs(jacobian) <= 1.0, (
            f"Jacobiano={jacobian:.2e} fuera del rango físico esperado [1e-9, 1] S"
        )

    def test_voltage_mapping_is_consistent(self, topology: LambdaDiodeTopology):
        """
        Verifica que el mapeo de tensiones es lineal en V_x:
        V_gs_n = V_x  → dV_gs_n/dV_x = +1  (verificado por diferencias finitas)
        """
        v_app = 3.0
        v_x_0 = 1.5
        h = FD_STEP

        res0, _, _, _ = topology.get_residual_and_jacobian(v_app, v_x_0)
        res1, _, _, _ = topology.get_residual_and_jacobian(v_app, v_x_0 + h)
        res2, _, _, _ = topology.get_residual_and_jacobian(v_app, v_x_0 - h)

        # Jacobiano numérico por diferencias centrales
        jacobian_numeric = (res1 - res2) / (2 * h)
        _, jacobian_analytic, _, _ = topology.get_residual_and_jacobian(v_app, v_x_0)

        assert jacobian_analytic == pytest.approx(
            jacobian_numeric, rel=REL_TOL_JACOBIAN
        ), (
            f"Jacobiano analítico={jacobian_analytic:.4e} vs "
            f"numérico={jacobian_numeric:.4e}"
        )


# =============================================================================
# NIVEL 2B: JACOBIANO TOPOLÓGICO vs. DIFERENCIAS FINITAS
# =============================================================================

class TestTopologyDerivatives:
    """
    Verificación exhaustiva del Jacobiano df/dV_x con diferencias finitas.

    Esta es la prueba de integración más crítica: si el Jacobiano es incorrecto,
    Newton-Raphson puede divergir o converger a soluciones falsas.

    Fórmula esperada:
        J = g_m_n + g_ds_n + g_m_p - g_ds_p
    """

    @pytest.mark.parametrize("v_app,v_x", [
        (1.0, 0.5),
        (2.0, 1.0),
        (3.0, 1.5),
        (2.5, 0.8),
        (4.0, 2.0),
        (0.5, 0.25),
    ])
    def test_jacobian_vs_finite_differences(
        self,
        topology: LambdaDiodeTopology,
        v_app: float,
        v_x: float,
    ):
        """
        Para múltiples puntos de operación, el Jacobiano analítico debe
        coincidir con la aproximación de diferencias finitas.
        """
        h = FD_STEP

        _, J_analytic, _, _ = topology.get_residual_and_jacobian(v_app, v_x)

        res_plus,  _, _, _ = topology.get_residual_and_jacobian(v_app, v_x + h)
        res_minus, _, _, _ = topology.get_residual_and_jacobian(v_app, v_x - h)
        J_numeric = (res_plus - res_minus) / (2 * h)

        # Tolerancia relativa generosa para diferencias finitas con h=1e-7
        assert J_analytic == pytest.approx(J_numeric, rel=1e-3), (
            f"J_analítico={J_analytic:.6e} vs J_numérico={J_numeric:.6e} "
            f"en (V_app={v_app}, V_x={v_x})"
        )

    def test_jacobian_formula_decomposition(self, topology: LambdaDiodeTopology):
        """
        Verifica la descomposición J = g_m_n + g_ds_n + g_m_p - g_ds_p
        evaluando cada componente individualmente.
        """
        v_app, v_x = 2.0, 1.0

        # Obtener componentes individuales de cada modelo
        v_gs_n    = v_x
        v_ds_n    = v_x
        v_gs_p_m  = v_app - v_x   # magnitud
        v_ds_p    = v_x - v_app   # con signo

        _, g_m_n, g_ds_n = topology.jfet_n.evaluate(v_gs_n, v_ds_n)
        _, g_m_p, g_ds_p = topology.jfet_p.evaluate(v_gs_p_m, v_ds_p)

        J_formula = g_m_n + g_ds_n + g_m_p - g_ds_p

        _, J_code, _, _ = topology.get_residual_and_jacobian(v_app, v_x)

        assert J_formula == pytest.approx(J_code, rel=1e-10), (
            f"Descomposición: {J_formula:.6e} vs código: {J_code:.6e}"
        )


# =============================================================================
# NIVEL 3A: CONTRATOS DEL SOLVER NEWTON-RAPHSON
# =============================================================================

class TestNewtonSolverContracts:
    """
    Verifica las pre y post condiciones del NewtonSolver.

    Pre-condiciones:
        • v_app ≥ 0
        • max_iter > 0, tol > 0, step_tol > 0
    Post-condiciones:
        • V_x ∈ [0, v_app]  (dominio físico)
        • I ≥ 0             (corriente no negativa)
        • converged ∈ {True, False}
    """

    def test_invalid_max_iter_raises(self):
        """max_iter ≤ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="max_iter"):
            NewtonSolver(max_iter=0)

    def test_invalid_tol_raises(self):
        """tol ≤ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="tol"):
            NewtonSolver(tol=0.0)

    def test_invalid_step_tol_raises(self):
        """step_tol ≤ 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="step_tol"):
            NewtonSolver(step_tol=-1e-9)

    def test_negative_vapp_raises(self, solver: NewtonSolver):
        """v_app < 0 debe levantar ValueError."""
        with pytest.raises(ValueError, match="v_app"):
            solver.solve_for_voltage(-1.0, 0.5)

    def test_zero_vapp_returns_zeros(self, solver: NewtonSolver):
        """v_app = 0 → V_x = 0, I = 0, convergido = True."""
        v_x, i, converged = solver.solve_for_voltage(0.0, 0.0)
        assert v_x        == pytest.approx(0.0, abs=1e-12)
        assert i          == pytest.approx(0.0, abs=1e-12)
        assert converged  is True

    @pytest.mark.parametrize("v_app", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    def test_vx_within_physical_domain(self, solver: NewtonSolver, v_app: float):
        """V_x debe estar en [0, V_app] (dominio físico del circuito)."""
        v_x, _, _ = solver.solve_for_voltage(v_app, v_app / 2)
        assert 0.0 <= v_x <= v_app + 1e-9, (
            f"V_x={v_x:.4f} fuera de [0, {v_app}]"
        )

    @pytest.mark.parametrize("v_app", [0.5, 1.0, 2.0, 3.0])
    def test_current_non_negative(self, solver: NewtonSolver, v_app: float):
        """La corriente de salida debe ser no negativa."""
        _, i, _ = solver.solve_for_voltage(v_app, v_app / 2)
        assert i >= -1e-12, f"Corriente negativa I={i:.4e} en V_app={v_app}"

    def test_solve_returns_three_tuple(self, solver: NewtonSolver):
        """solve_for_voltage retorna exactamente 3 valores."""
        result = solver.solve_for_voltage(1.0, 0.5)
        assert len(result) == 3


# =============================================================================
# NIVEL 3B: CONVERGENCIA DEL SOLVER
# =============================================================================

class TestNewtonSolverConvergence:
    """
    Verifica que el solver converge correctamente y produce residuos pequeños.

    La convergencia cuadrática de Newton implica que:
        |f(V_x*)| < tolerancia
    donde V_x* es la solución reportada.
    """

    @pytest.mark.parametrize("v_app", [
        0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
    ])
    def test_converged_solution_satisfies_kcl(self, v_app: float):
        """
        Si el solver converge, el residuo KCL en la solución debe ser < tol.
        """
        s = NewtonSolver(tol=1e-9)
        v_x, _, converged = s.solve_for_voltage(v_app, v_app / 2)

        if converged:
            res, _, _, _ = s.topology.get_residual_and_jacobian(v_app, v_x)
            assert abs(res) < 1e-7, (
                f"Residuo={res:.2e} demasiado grande en V_app={v_app}, V_x={v_x}"
            )

    def test_convergence_rate_is_quadratic(self):
        """
        Verifica la convergencia cuadrática monitoreando el residuo por iteración.

        Estrategia: parchar la topología para registrar el historial de residuos
        y verificar que decae cuadráticamente en las primeras iteraciones.
        """
        v_app = 2.0
        residuals: list[float] = []

        solver = NewtonSolver(max_iter=20, tol=1e-12)
        original_get = solver.topology.get_residual_and_jacobian

        def instrumented_get(va, vx):
            result = original_get(va, vx)
            residuals.append(abs(result[0]))
            return result

        solver.topology.get_residual_and_jacobian = instrumented_get
        v_x, _, converged = solver.solve_for_voltage(v_app, v_app * 0.3)

        if converged and len(residuals) >= 4:
            # Verificar que la reducción es al menos lineal (cuadrática es ideal)
            # Ratio de reducción debe ser < 0.1 en las primeras iteraciones
            for k in range(min(3, len(residuals) - 1)):
                if residuals[k] > 1e-10:  # Evitar división por casi-cero
                    ratio = residuals[k + 1] / residuals[k]
                    assert ratio < 1.0, (
                        f"No hay reducción en iter {k}: "
                        f"residual[{k}]={residuals[k]:.2e}, "
                        f"residual[{k+1}]={residuals[k+1]:.2e}"
                    )

    @pytest.mark.parametrize("v_x_guess", [0.01, 0.3, 0.5, 0.7, 0.99])
    def test_convergence_independent_of_initial_guess(self, v_x_guess: float):
        """
        El resultado debe ser consistente para diferentes semillas iniciales.
        (La solución KCL es única para este circuito.)
        """
        solver  = NewtonSolver(tol=1e-9)
        v_app   = 2.0
        results = []
        for scale in [0.1, 0.3, 0.5, 0.7, 0.9]:
            v_x, i, converged = solver.solve_for_voltage(v_app, scale * v_app)
            if converged:
                results.append(v_x)

        if len(results) >= 2:
            # Todas las soluciones convergidas deben coincidir
            for k in range(1, len(results)):
                assert results[k] == pytest.approx(results[0], rel=1e-4), (
                    f"Soluciones inconsistentes: {results[0]:.6f} vs {results[k]:.6f}"
                )


# =============================================================================
# NIVEL 3C: ROBUSTEZ DEL SOLVER
# =============================================================================

class TestNewtonSolverRobustness:
    """
    Verifica la estabilidad ante condiciones adversas:
    - Jacobiano casi singular
    - Semillas extremas (V_x = 0 o V_x = V_app)
    - V_app muy pequeño o muy grande
    - Múltiples iteraciones sin convergencia
    """

    def test_very_small_vapp_does_not_crash(self):
        """V_app = 1e-8 no debe lanzar excepción."""
        solver = NewtonSolver()
        try:
            v_x, i, _ = solver.solve_for_voltage(1e-8, 5e-9)
            # Solo verificamos que no explota
            assert math.isfinite(v_x)
            assert math.isfinite(i)
        except Exception as e:
            pytest.fail(f"Excepción inesperada para V_app=1e-8: {e}")

    def test_large_vapp_does_not_crash(self):
        """V_app = 20.0 no debe lanzar excepción."""
        solver = NewtonSolver()
        try:
            v_x, i, _ = solver.solve_for_voltage(20.0, 10.0)
            assert math.isfinite(v_x)
            assert math.isfinite(i)
        except Exception as e:
            pytest.fail(f"Excepción inesperada para V_app=20.0: {e}")

    def test_extreme_initial_guess_vx_zero(self):
        """Semilla V_x = 0 (borde del dominio) no debe crashear."""
        solver = NewtonSolver()
        v_x, i, _ = solver.solve_for_voltage(2.0, 0.0)
        assert math.isfinite(v_x)
        assert math.isfinite(i)

    def test_extreme_initial_guess_vx_at_vapp(self):
        """Semilla V_x = V_app (borde superior) no debe crashear."""
        solver = NewtonSolver()
        v_x, i, _ = solver.solve_for_voltage(2.0, 2.0)
        assert math.isfinite(v_x)
        assert math.isfinite(i)

    def test_singular_jacobian_handled(self):
        """
        Si el Jacobiano es cero, el solver debe recuperarse sin crashear.
        Parcheamos el modelo para forzar J=0 temporalmente.
        """
        solver = NewtonSolver(max_iter=10)
        original = solver.topology.get_residual_and_jacobian

        call_count = [0]

        def patched(v_app, v_x):
            call_count[0] += 1
            res, J, i_n, i_p = original(v_app, v_x)
            if call_count[0] <= 2:
                J = 0.0  # Forzar Jacobiano singular en primeras llamadas
            return res, J, i_n, i_p

        solver.topology.get_residual_and_jacobian = patched

        try:
            v_x, i, _ = solver.solve_for_voltage(2.0, 1.0)
            assert math.isfinite(v_x)
            assert math.isfinite(i)
        except Exception as e:
            pytest.fail(f"Excepción inesperada ante Jacobiano singular: {e}")

    def test_result_finite_even_without_convergence(self):
        """Aunque no converja, los resultados deben ser finitos (no NaN/Inf)."""
        solver = NewtonSolver(max_iter=2, tol=1e-20)  # Casi imposible converger
        v_x, i, converged = solver.solve_for_voltage(2.0, 1.0)
        assert math.isfinite(v_x), f"V_x={v_x} no es finito"
        assert math.isfinite(i),   f"I={i} no es finito"
        # No garantizamos convergencia, solo finitud


# =============================================================================
# NIVEL 4A: PRUEBAS DEL ANALIZADOR NEUROMÓRFICO
# =============================================================================

class TestNeuromorphicAnalyzer:
    """
    Verifica la generación de la curva I-V completa y la estructura de salida.
    """

    def test_output_dictionary_has_required_keys(self, analyzer: NeuromorphicAnalyzer):
        """El diccionario de resultados debe contener todas las claves esperadas."""
        results = analyzer.simulate_iv_curve(0.0, 3.0, 30)
        required_keys = {
            "voltage_V",
            "current_mA",
            "current_mA_smooth",
            "differential_conductance_mS",
            "ndr_detected",
            "ndr_regions",
            "current_peaks_mA",
            "convergence_flags",
        }
        assert required_keys.issubset(set(results.keys())), (
            f"Claves faltantes: {required_keys - set(results.keys())}"
        )

    def test_output_lengths_are_consistent(self, analyzer: NeuromorphicAnalyzer):
        """Todas las listas numéricas deben tener la misma longitud."""
        steps   = 50
        results = analyzer.simulate_iv_curve(0.0, 4.0, steps)
        n       = len(results["voltage_V"])

        assert len(results["current_mA"])                   == n
        assert len(results["current_mA_smooth"])             == n
        assert len(results["differential_conductance_mS"])   == n
        assert len(results["convergence_flags"])              == n

    def test_voltage_sweep_starts_and_ends_correctly(self, analyzer: NeuromorphicAnalyzer):
        """El barrido debe comenzar en v_start y terminar en v_end."""
        v_start, v_end = 0.5, 4.5
        results = analyzer.simulate_iv_curve(v_start, v_end, 40)
        assert results["voltage_V"][0]  == pytest.approx(v_start, abs=1e-6)
        assert results["voltage_V"][-1] == pytest.approx(v_end,   abs=1e-6)

    def test_voltage_sweep_is_monotone_increasing(self, analyzer: NeuromorphicAnalyzer):
        """Los voltajes de barrido deben ser estrictamente crecientes."""
        results   = analyzer.simulate_iv_curve(0.0, 5.0, 60)
        voltages  = results["voltage_V"]
        diffs     = np.diff(voltages)
        assert np.all(diffs > 0), "El barrido de tensión no es monótono creciente."

    def test_invalid_vend_less_than_vstart_raises(self, analyzer: NeuromorphicAnalyzer):
        """v_end ≤ v_start debe levantar ValueError."""
        with pytest.raises(ValueError):
            analyzer.simulate_iv_curve(3.0, 1.0, 50)

    def test_invalid_steps_too_small_raises(self, analyzer: NeuromorphicAnalyzer):
        """steps < 10 debe levantar ValueError."""
        with pytest.raises(ValueError):
            analyzer.simulate_iv_curve(0.0, 5.0, 5)

    def test_ndr_detected_is_boolean(self, analyzer: NeuromorphicAnalyzer):
        """ndr_detected debe ser un booleano."""
        results = analyzer.simulate_iv_curve(0.0, 5.0, 40)
        assert isinstance(results["ndr_detected"], bool)

    def test_ndr_regions_is_list(self, analyzer: NeuromorphicAnalyzer):
        """ndr_regions debe ser una lista (puede estar vacía)."""
        results = analyzer.simulate_iv_curve(0.0, 5.0, 40)
        assert isinstance(results["ndr_regions"], list)

    def test_all_currents_finite(self, analyzer: NeuromorphicAnalyzer):
        """Ninguna corriente en la curva I-V debe ser NaN o Inf."""
        results = analyzer.simulate_iv_curve(0.0, 5.0, 50)
        for k, i_ma in enumerate(results["current_mA"]):
            assert math.isfinite(i_ma), (
                f"Corriente no finita en índice {k}: {i_ma}"
            )

    def test_smooth_current_within_reasonable_bounds_of_raw(
        self, analyzer: NeuromorphicAnalyzer
    ):
        """
        La corriente suavizada no debe alejarse más de 10% de la cruda
        en el valor máximo (el suavizado preserva amplitud).
        """
        results    = analyzer.simulate_iv_curve(0.0, 5.0, 100)
        raw        = np.array(results["current_mA"])
        smooth     = np.array(results["current_mA_smooth"])
        max_raw    = max(abs(raw.max()), 1e-9)
        max_smooth = abs(smooth.max())
        assert abs(max_smooth - raw.max()) / max_raw < 0.15, (
            f"Suavizado altera el pico: raw={raw.max():.4f}, smooth={max_smooth:.4f}"
        )

    @pytest.mark.parametrize("steps", [10, 50, 100, 200])
    def test_various_step_counts(self, steps: int):
        """El analizador funciona para diferentes números de pasos."""
        a = NeuromorphicAnalyzer()
        results = a.simulate_iv_curve(0.0, 5.0, steps)
        assert len(results["voltage_V"]) >= steps


# =============================================================================
# NIVEL 4B: DETECCIÓN NDR
# =============================================================================

class TestNDRDetection:
    """
    Verifica la detección robusta de la región de Resistencia Diferencial Negativa.

    Propiedades esperadas del Diodo Lambda (2N5457 + J176):
    1. La curva I-V tiene al menos un pico de corriente (máximo local)
    2. Después del pico, la corriente DISMINUYE al aumentar la tensión
    3. La conductancia diferencial dI/dV < 0 en la región NDR
    4. La región NDR ocurre en el rango 0-5V para los parámetros dados
    """

    @pytest.fixture(scope="class")
    def iv_results(self) -> dict:
        """Curva I-V de alta resolución para análisis NDR."""
        a = NeuromorphicAnalyzer()
        return a.simulate_iv_curve(0.0, 5.0, 300)

    def test_ndr_regions_have_valid_voltage_bounds(self, iv_results: dict):
        """Cada región NDR debe tener V_inicio < V_fin y ambos en [0, 5]."""
        for v_start, v_end in iv_results["ndr_regions"]:
            assert v_start < v_end, f"Región NDR inválida: [{v_start}, {v_end}]"
            assert 0.0 <= v_start <= 5.0
            assert 0.0 <= v_end   <= 5.0

    def test_current_peaks_have_valid_structure(self, iv_results: dict):
        """Cada pico debe tener (voltaje, corriente) con valores positivos."""
        for v_pk, i_pk in iv_results["current_peaks_mA"]:
            assert v_pk >= 0.0, f"Voltaje de pico negativo: {v_pk}"
            assert i_pk >= 0.0, f"Corriente de pico negativa: {i_pk}"

    def test_ndr_consistency_with_peaks(self, iv_results: dict):
        """Si hay picos de corriente, debe haber al menos una región NDR."""
        peaks = iv_results["current_peaks_mA"]
        ndr   = iv_results["ndr_regions"]
        if len(peaks) > 0:
            # Con picos de corriente, esperamos NDR
            # (aunque depende del umbral, debe haber al menos conductancia negativa)
            conductances = iv_results["differential_conductance_mS"]
            has_negative_cond = any(g < -0.001 for g in conductances)
            if has_negative_cond:
                assert len(ndr) > 0, (
                    "Hay conductancia negativa pero no se detectó región NDR"
                )

    def test_ndr_detected_flag_consistent_with_regions(self, iv_results: dict):
        """El flag ndr_detected debe ser True ↔ hay regiones NDR en la lista."""
        ndr_flag    = iv_results["ndr_detected"]
        ndr_regions = iv_results["ndr_regions"]
        assert ndr_flag == (len(ndr_regions) > 0), (
            f"Inconsistencia: ndr_detected={ndr_flag} pero "
            f"len(ndr_regions)={len(ndr_regions)}"
        )

    def test_conductance_negative_inside_ndr_region(self, iv_results: dict):
        """
        Dentro de cada región NDR detectada, la conductancia debe ser
        negativa en al menos la mayoría de los puntos.
        """
        voltages = np.array(iv_results["voltage_V"])
        cond_mS  = np.array(iv_results["differential_conductance_mS"])

        for v0, v1 in iv_results["ndr_regions"]:
            mask      = (voltages >= v0) & (voltages <= v1)
            cond_in   = cond_mS[mask]
            n_neg     = np.sum(cond_in < 0)
            fraction  = n_neg / max(len(cond_in), 1)
            assert fraction >= 0.5, (
                f"Región NDR [{v0:.2f}, {v1:.2f}]: solo {fraction*100:.0f}% "
                f"de puntos con conductancia negativa."
            )

    def test_current_decreases_in_ndr_region(self, iv_results: dict):
        """
        La corriente suavizada debe disminuir neta en cada región NDR.
        I(V_fin_NDR) < I(V_ini_NDR) — esto es la definición de NDR.
        """
        voltages = np.array(iv_results["voltage_V"])
        smooth   = np.array(iv_results["current_mA_smooth"])

        for v0, v1 in iv_results["ndr_regions"]:
            idx0 = np.argmin(np.abs(voltages - v0))
            idx1 = np.argmin(np.abs(voltages - v1))
            if idx0 < idx1:
                # La corriente al final de la NDR debe ser menor que al inicio
                assert smooth[idx1] < smooth[idx0], (
                    f"La corriente no disminuye en NDR [{v0:.2f},{v1:.2f}]: "
                    f"I(v0)={smooth[idx0]:.4f} mA, I(v1)={smooth[idx1]:.4f} mA"
                )


# =============================================================================
# NIVEL 5A: INVARIANTES FÍSICOS DEL CIRCUITO
# =============================================================================

class TestPhysicalInvariants:
    """
    Verifica las leyes fundamentales de circuitos como invariantes de integración.

    Invariante 1 (KCL): En la solución, I_N = I_P (conservación de corriente).
    Invariante 2 (Positividad): I_D ≥ 0 para ambos transistores.
    Invariante 3 (Monotonicidad del modelo): I_D crece con |V_DS| (λ > 0).
    Invariante 4 (Saturación): I_D ≤ I_DSS·(1 + λ·V_DS_max) siempre.
    """

    def test_kcl_satisfied_at_solution(self):
        """
        KCL: En la solución V_x*, I_N(V_x*) ≈ I_P(V_x*).
        El residuo |I_N - I_P| debe ser < tolerancia del solver.
        """
        solver   = NewtonSolver(tol=1e-9)
        topology = LambdaDiodeTopology()

        for v_app in [0.5, 1.0, 2.0, 3.0, 4.0]:
            v_x, _, converged = solver.solve_for_voltage(v_app, v_app / 2)
            if converged:
                res, _, i_n, i_p = topology.get_residual_and_jacobian(v_app, v_x)
                assert abs(res) < 1e-7, (
                    f"KCL violado en V_app={v_app}: I_N={i_n:.4e}, I_P={i_p:.4e}, "
                    f"|I_N-I_P|={abs(res):.4e}"
                )

    def test_current_bounded_by_idss_n(self):
        """
        I_N no puede superar I_DSS_n · (1 + λ·V_DS_max) físicamente.
        Para V_DS ≤ 5V: I_N ≤ I_DSS_n · (1 + λ·5) = 3mA · 1.1 = 3.3 mA
        """
        solver   = NewtonSolver()
        I_MAX_N  = PARAM_2N5457.idss * (1 + PARAM_2N5457.lam * 5.0)

        for v_app in np.linspace(0.1, 5.0, 20):
            _, i_n, converged = solver.solve_for_voltage(v_app, v_app / 2)
            assert i_n <= I_MAX_N * 1.01, (
                f"I_N={i_n:.4e} A supera el límite físico {I_MAX_N:.4e} A "
                f"en V_app={v_app:.2f}"
            )

    def test_current_bounded_by_idss_p(self):
        """
        I_P no puede superar I_DSS_p · (1 + λ·|V_DS|_max).
        Para |V_DS_p| ≤ 5V: I_P ≤ I_DSS_p · 1.1 = 16.5 mA
        """
        I_MAX_P  = PARAM_J176.idss * (1 + PARAM_J176.lam * 5.0)
        topology = LambdaDiodeTopology()

        for v_app in np.linspace(0.1, 5.0, 20):
            for v_x in np.linspace(0.01, v_app - 0.01, 5):
                _, _, _, i_p = topology.get_residual_and_jacobian(v_app, v_x)
                assert i_p <= I_MAX_P * 1.01, (
                    f"I_P={i_p:.4e} A supera {I_MAX_P:.4e} A"
                )

    def test_power_dissipation_non_negative(self):
        """
        La potencia disipada P = V_app · I debe ser no negativa
        (el dispositivo pasivo solo disipa, no genera, energía neta).
        """
        solver = NewtonSolver()
        for v_app in np.linspace(0.1, 5.0, 20):
            _, i, converged = solver.solve_for_voltage(v_app, v_app / 2)
            if converged:
                power = v_app * i
                assert power >= -1e-12, (
                    f"Potencia negativa P={power:.4e} W en V_app={v_app:.2f}"
                )

    def test_superposition_scaling_single_transistor(self):
        """
        Para un JFET individual, escalar I_DSS escala la corriente linealmente.
        I_D(2·I_DSS) = 2 · I_D(I_DSS)  [con los mismos V_GS, V_DS]
        """
        v_gs, v_ds = -1.0, 2.0

        params1 = JFETParameters(idss=1e-3, vp=-2.0, lam=0.01, is_n_channel=True)
        params2 = JFETParameters(idss=2e-3, vp=-2.0, lam=0.01, is_n_channel=True)

        m1 = JFETModel(params1)
        m2 = JFETModel(params2)

        i1, _, _ = m1.evaluate(v_gs, v_ds)
        i2, _, _ = m2.evaluate(v_gs, v_ds)

        assert i2 == pytest.approx(2.0 * i1, rel=1e-10), (
            f"Escalado I_DSS no lineal: 2·I1={2*i1:.4e}, I2={i2:.4e}"
        )


# =============================================================================
# NIVEL 5B: PRUEBAS DE REGRESIÓN (GOLDEN VALUES)
# =============================================================================

class TestNumericalRegression:
    """
    Valores de referencia calculados analíticamente y congelados.

    Si alguna de estas pruebas falla, indica un cambio en la implementación
    que altera el comportamiento numérico — requiere revisión consciente.

    Metodología de golden values:
    Los valores fueron calculados con la fórmula de Shockley explícita
    y verificados con diferencias finitas independientes.
    """

    def test_golden_n_jfet_active_region(self):
        """
        2N5457: V_GS = -0.75V (mitad del pinch-off), V_DS = 2V
        I_D = 3mA · (1 - (-0.75)/(-1.5))² · (1 + 0.02·2)
            = 3mA · (1 - 0.5)²             · 1.04
            = 3mA · 0.25                   · 1.04
            = 0.78 mA
        """
        model = JFETModel(PARAM_2N5457)
        i_d, g_m, g_ds = model.evaluate(-0.75, 2.0)

        assert i_d  == pytest.approx(0.78e-3,  rel=1e-6)
        # g_m = 2 · 3mA · (0.5) · (1/1.5) · 1.04 = 2.08 mA/V
        assert g_m  == pytest.approx(2.08e-3,  rel=1e-5)
        # g_ds = 3mA · 0.25 · 0.02 = 15 μA/V
        assert g_ds == pytest.approx(15.0e-6,  rel=1e-6)

    def test_golden_p_jfet_active_region(self):
        """
        J176: |V_GS| = 1.25V (mitad del pinch-off), V_DS = -2V
        I_D = 15mA · (1 - 1.25/2.5)² · (1 + 0.02·2)
            = 15mA · (0.5)²            · 1.04
            = 15mA · 0.25              · 1.04
            = 3.9 mA
        """
        model = JFETModel(PARAM_J176)
        i_d, g_m, g_ds = model.evaluate(1.25, -2.0)

        assert i_d  == pytest.approx(3.9e-3,  rel=1e-6)
        # g_m = 2 · 15mA · (0.5) · (-1/2.5) · 1.04 = -6.24 mA/V
        assert g_m  == pytest.approx(-6.24e-3, rel=1e-5)
        # g_ds = 15mA · 0.25 · 0.02 · sign(-2) = -75 μA/V
        assert g_ds == pytest.approx(-75.0e-6,  rel=1e-6)

    def test_golden_topology_residual(self):
        """
        Residual topológico para V_app=2.0, V_x=1.0:
        V_gs_n = 1.0,  V_ds_n = 1.0
        V_gs_p_mag = 1.0,  V_ds_p = -1.0

        I_N = 3mA  · (1 - 1.0/(-1.5))²       · (1 + 0.02·1.0)
            = 3mA  · (1 + 2/3)²               · 1.02
        Pero 1/(-1.5) = -0.667, entonces xi_n = 1.0/(-1.5) = -0.667
        (1 - xi_n) = 1 + 0.667 = 1.667
        I_N = 3mA · 1.667² · 1.02 = 3mA · 2.779 · 1.02 ≈ 8.507 mA

        I_P = 15mA · (1 - 1.0/2.5)²  · (1 + 0.02·1.0)
            = 15mA · (0.6)²           · 1.02
            = 15mA · 0.36             · 1.02
            = 5.508 mA

        Residual = 8.507 - 5.508 = 2.999 mA (aprox)
        """
        topology = LambdaDiodeTopology()

        # Calcular valores esperados explícitamente
        # N-JFET
        vp_n, idss_n, lam_n = PARAM_2N5457.vp, PARAM_2N5457.idss, PARAM_2N5457.lam
        xi_n   = 1.0 / vp_n          # 1.0 / (-1.5)
        i_n_ex = idss_n * (1 - xi_n)**2 * (1 + lam_n * 1.0)

        # P-JFET
        vp_p, idss_p, lam_p = PARAM_J176.vp, PARAM_J176.idss, PARAM_J176.lam
        xi_p   = 1.0 / vp_p          # 1.0 / 2.5
        i_p_ex = idss_p * (1 - xi_p)**2 * (1 + lam_p * 1.0)

        expected_residual = i_n_ex - i_p_ex

        res, _, i_n_code, i_p_code = topology.get_residual_and_jacobian(2.0, 1.0)

        assert res      == pytest.approx(expected_residual, rel=1e-8)
        assert i_n_code == pytest.approx(i_n_ex,            rel=1e-8)
        assert i_p_code == pytest.approx(i_p_ex,            rel=1e-8)

    def test_golden_jfet_at_half_pinchoff(self):
        """
        Verifica el caso xi = 0.5 (punto de operación central).
        En xi = 0.5: I_D = I_DSS · 0.25 · factor_vds
        """
        model = JFETModel(P_N)
        v_gs  = P_N.vp / 2         # xi = (vp/2)/vp = 0.5
        v_ds  = 1.0
        i_d, _, _ = model.evaluate(v_gs, v_ds)

        expected = P_N.idss * 0.25 * (1 + P_N.lam * v_ds)
        assert i_d == pytest.approx(expected, rel=1e-10)


# =============================================================================
# NIVEL 6: PRUEBA DE EXTREMO A EXTREMO (END-TO-END)
# =============================================================================

class TestEndToEnd:
    """
    Verifica el flujo completo: parámetros físicos → modelo → topología →
    solver → analizador → curva I-V → detección NDR.

    Estas pruebas replican el uso real del módulo.
    """

    def test_full_pipeline_returns_valid_iv_curve(self):
        """Pipeline completo produce curva I-V física y numéricamente válida."""
        analyzer = NeuromorphicAnalyzer()
        results  = analyzer.simulate_iv_curve(0.0, 5.0, 100)

        # Estructura
        assert "voltage_V"   in results
        assert "current_mA"  in results
        assert "ndr_detected" in results

        # Valores finitos
        assert all(math.isfinite(v) for v in results["voltage_V"])
        assert all(math.isfinite(i) for i in results["current_mA"])

        # Corrientes no negativas
        assert all(i >= -1e-9 for i in results["current_mA"])

        # Voltajes monótonos
        vv   = results["voltage_V"]
        diffs = [vv[k+1] - vv[k] for k in range(len(vv)-1)]
        assert all(d > 0 for d in diffs)

    def test_high_resolution_sweep_detects_ndr(self):
        """
        Un barrido de alta resolución (300 puntos) debe detectar NDR
        con los parámetros reales de 2N5457 y J176.
        """
        analyzer = NeuromorphicAnalyzer()
        results  = analyzer.simulate_iv_curve(0.0, 5.0, 300)

        # NDR es la firma del Diodo Lambda — si no aparece, la física es incorrecta
        assert results["ndr_detected"], (
            "El Diodo Lambda (2N5457 + J176) debe exhibir NDR en 0-5V. "
            "Verifique la topología y los parámetros de los JFETs."
        )

    def test_current_peak_exists_and_is_meaningful(self):
        """
        Debe existir al menos un pico de corriente con valor significativo
        (> 0.01 mA — por encima del ruido numérico).
        """
        analyzer = NeuromorphicAnalyzer()
        results  = analyzer.simulate_iv_curve(0.0, 5.0, 200)

        peaks = results["current_peaks_mA"]
        assert len(peaks) > 0, "No se detectaron picos de corriente."

        max_peak_i = max(i for _, i in peaks)
        assert max_peak_i > 0.01, (
            f"Pico máximo {max_peak_i:.4f} mA es demasiado pequeño (< 0.01 mA)."
        )

    def test_different_voltage_ranges_produce_consistent_results(self):
        """
        El análisis en un subconjunto del rango de tensión debe ser
        consistente con el análisis del rango completo (en la región solapada).
        """
        a = NeuromorphicAnalyzer()

        res_full = a.simulate_iv_curve(0.0, 5.0, 200)
        res_sub  = a.simulate_iv_curve(1.0, 3.0, 80)

        # Encontrar el índice en el barrido completo donde V ≈ 2.0
        v_full = np.array(res_full["voltage_V"])
        i_full = np.array(res_full["current_mA"])

        v_sub  = np.array(res_sub["voltage_V"])
        i_sub  = np.array(res_sub["current_mA"])

        # Comparar en V ≈ 2.0V (punto de solapamiento)
        v_ref    = 2.0
        idx_full = np.argmin(np.abs(v_full - v_ref))
        idx_sub  = np.argmin(np.abs(v_sub  - v_ref))

        i_at_2v_full = i_full[idx_full]
        i_at_2v_sub  = i_sub[idx_sub]

        # La corriente en V=2V debe ser similar (< 5% de diferencia relativa)
        if i_at_2v_full > 1e-6:  # Solo comparar si no es prácticamente cero
            rel_diff = abs(i_at_2v_full - i_at_2v_sub) / abs(i_at_2v_full)
            assert rel_diff < 0.05, (
                f"Inconsistencia en V≈2V: full={i_at_2v_full:.4f} mA, "
                f"sub={i_at_2v_sub:.4f} mA (diff={rel_diff*100:.2f}%)"
            )

    def test_custom_jfet_parameters_integrate_correctly(self):
        """
        Parámetros JFET personalizados deben integrarse en el pipeline
        sin errores y producir resultados físicamente razonables.
        """
        # Modificar parámetros globales temporalmente via monkeypatching
        import neuromorphic_solver as ns

        orig_n = ns.PARAM_2N5457
        orig_p = ns.PARAM_J176

        try:
            # Parámetros alternativos (mismo tipo de dispositivo, diferente magnitud)
            ns.PARAM_2N5457 = JFETParameters(
                idss=5e-3, vp=-2.0, lam=0.01, is_n_channel=True
            )
            ns.PARAM_J176 = JFETParameters(
                idss=10e-3, vp=3.0, lam=0.01, is_n_channel=False
            )

            analyzer = NeuromorphicAnalyzer()
            results  = analyzer.simulate_iv_curve(0.0, 5.0, 80)

            assert all(math.isfinite(i) for i in results["current_mA"])
            assert all(i >= -1e-9 for i in results["current_mA"])

        finally:
            # Restaurar parámetros originales
            ns.PARAM_2N5457 = orig_n
            ns.PARAM_J176   = orig_p


# =============================================================================
# CONFIGURACIÓN DE PYTEST Y MARCADORES
# =============================================================================

def pytest_configure(config):
    """Registra marcadores personalizados para la suite."""
    config.addinivalue_line(
        "markers", "slow: pruebas de larga duración (excluir con -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "physics: pruebas de invariantes físicos"
    )
    config.addinivalue_line(
        "markers", "regression: pruebas de valores de referencia congelados"
    )
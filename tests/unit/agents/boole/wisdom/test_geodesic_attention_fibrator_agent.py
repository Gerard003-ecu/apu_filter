# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — Geodesic Attention Fibrator Agent (Custodio de Covarianza)        ║
║  Ruta   : tests/unit/agents/boole/wisdom/test_geodesic_attention_fibrator_agent.py       ║
║  Versión: 7.0.0-Rigorous-Ricci-Polyakov-FeynmanKac-Hodge-Spectral-TestSuite              ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPÓSITO CIBER-FÍSICO Y TOPOLOGÍA DE PRUEBAS (Rigor Categórico):                       ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Esta suite de pruebas consagra la Gobernanza de Covarianza Atencional del estrato       ║
║  WISDOM mediante un funtor de validación que verifica axiomáticamente el flujo de        ║
║  Ricci, la acción de Polyakov y el veto cuántico de Feynman-Kac del modelo LLM.          ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):     ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Auditoría del Flujo de Ricci y Torsión                                         ║
║           Verifica convergencia métrica, SPD, simetría y números de condición.           ║
║                                                                                          ║
║  Fase 2 → Certificación de la Acción de Polyakov                                         ║
║           Computa E[γ], certifica estabilidad geodésica y valida términos cinéticos.     ║
║                                                                                          ║
║  Fase 3 → Veto Cuántico de Feynman-Kac                                                   ║
║           Sintetiza la amplitud de transición Ψ[γ] subyugada al mínimo cuántico.         ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import logging
from typing import Tuple, Optional
from pathlib import Path

# =============================================================================
# Framework de pruebas
# =============================================================================
import pytest
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# =============================================================================
# Módulo bajo prueba
# =============================================================================
from app.agents.boole.wisdom.geodesic_attention_fibrator_agent import (
    GeodesicAttentionFibratorAgent,
    RicciFlowAuditData,
    PolyakovActionAuditData,
    FeynmanKacAuditData,
    GeodesicAttentionGovernanceState,
    # Excepciones
    GeodesicAttentionAgentError,
    GeodesicInputValidationError,
    MetricDegeneracyError,
    RicciFlowDivergenceError,
    PolyakovActionViolationError,
    QuantumFeynmanKacVeto,
)

# =============================================================================
# Logger y constantes globales de prueba
# =============================================================================
logger = logging.getLogger("MAC.Wisdom.Test.GeodesicAttentionFibratorAgent")
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)

# =============================================================================
# FIXTURES GLOBALES — GENERADORES DE TENORES MÉTRICOS Y ESTADOS
# =============================================================================


@pytest.fixture(scope="module")
def fixture_valid_metrics_3d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera tensores métricos Riemannianos válidos para dim=3.
    
    Retorna
    -------
    Tuple[g_k, g_k_plus_1]
        Métricas SPD simétricas con convergencia garantizada.
    """
    dim = 3
    
    # g_k: SPD simétrica
    g_k: NDArray[np.float64] = np.array(
        [[1.1, 0.05, 0.02],
         [0.05, 1.0, 0.03],
         [0.02, 0.03, 0.9]], dtype=np.float64
    )
    
    # g_k_plus_1: SPD simétrica, cercana a g_k (convergencia Ricci)
    g_k_plus_1: NDArray[np.float64] = np.array(
        [[1.12, 0.06, 0.025],
         [0.06, 1.02, 0.035],
         [0.025, 0.035, 0.92]], dtype=np.float64
    )
    
    return g_k, g_k_plus_1


@pytest.fixture(scope="module")
def fixture_valid_metrics_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera tensores métricos Riemannianos válidos para dim=2 (caso mínimo).
    
    Retorna
    -------
    Tuple[g_k, g_k_plus_1]
    """
    dim = 2
    
    g_k: NDArray[np.float64] = np.array(
        [[1.1, 0.05],
         [0.05, 1.0]], dtype=np.float64
    )
    
    g_k_plus_1: NDArray[np.float64] = np.array(
        [[1.12, 0.06],
         [0.06, 1.02]], dtype=np.float64
    )
    
    return g_k, g_k_plus_1


@pytest.fixture(scope="module")
def fixture_valid_geodesic_velocity_3d() -> NDArray[np.float64]:
    r"""
    Genera matriz de velocidades geodésicas válida para dim=3.
    
    Retorna
    -------
    NDArray[np.float64], shape (steps, dim)
    """
    steps, dim = 5, 3
    
    velocity: NDArray[np.float64] = np.array(
        [[0.1, 0.05, 0.08],
         [0.12, 0.06, 0.09],
         [0.11, 0.055, 0.085],
         [0.13, 0.065, 0.095],
         [0.105, 0.052, 0.082]], dtype=np.float64
    )
    
    return velocity


@pytest.fixture(scope="module")
def fixture_valid_geodesic_velocity_1d() -> NDArray[np.float64]:
    r"""
    Genera vector de velocidad geodésica 1D (caso mínimo).
    
    Retorna
    -------
    NDArray[np.float64], shape (dim,)
    """
    velocity: NDArray[np.float64] = np.array([0.1, 0.05, 0.08], dtype=np.float64)
    
    return velocity


@pytest.fixture(scope="module")
def fixture_valid_scalar_params() -> Tuple[float, float, float]:
    r"""
    Genera parámetros escalares válidos para la síntesis.
    
    Retorna
    -------
    Tuple[d_tau, torsion_hs_norm_sq, lambda_coupling]
    """
    d_tau: float = 0.01
    torsion_hs_norm_sq: float = 0.05
    lambda_coupling: float = 0.1
    
    return d_tau, torsion_hs_norm_sq, lambda_coupling


# =============================================================================
# FASE 1 — AUDITORÍA DEL FLUJO DE RICCI Y TORSIÓN
# =============================================================================
class TestPhase1_RicciFlowAuditor:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 1 — AUDITORÍA DE CONVERGENCIA MÉTRICA Y FLUJO DE RICCI
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase1_RicciFlowAuditor que consagra la
    geometría Riemanniana discreta del espacio atencional. Cada método verifica un axioma
    constitutivo del estrato WISDOM.
    
    Invariantes Verificados:
    ------------------------
    1. Coherencia dimensional de g_k, g_k_plus_1
    2. Simetría de métricas (G = Gᵀ)
    3. Definida positiva (λ_min > 0)
    4. Finitud de entradas (no NaN, no Inf)
    5. Convergencia del flujo: ||g_{k+1} - g_k||_F / scale < ε_Ricci
    6. Números de condición κ < κ_max
    7. Regularización espectral de autovalores pequeños
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.1 — VALIDACIÓN DIMENSIONAL Y ESTRUCTURAL
    # -------------------------------------------------------------------------
    
    def test_phase1_dimensions_valid_3d(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que métricas 3D válidas pasan la validación dimensional.
        
        Axioma: g_k, g_k_plus_1 ∈ ℝ^{n×n}, misma dimensión
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit.dimension == 3
        assert ricci_audit.is_metric_converged is True
        assert ricci_audit.metric_relative_residual < 1e-8
    
    def test_phase1_dimensions_valid_2d(
        self,
        fixture_valid_metrics_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica caso mínimo dim=2 (frontera inferior).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_2d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit.dimension == 2
        assert ricci_audit.is_metric_converged is True
    
    def test_phase1_dimension_mismatch_g_k_non_square(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k no cuadrada dispara GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_invalid: NDArray[np.float64] = g_k[:, :2]  # 3×2
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_invalid,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "cuadrada" in str(exc_info.value) or "2D" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_g_k_g_k_plus_1(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que dimensiones inconsistentes entre g_k y g_k_plus_1 disparan error.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_plus_1_invalid: NDArray[np.float64] = np.eye(2, dtype=np.float64)  # 2×2 ≠ 3×3
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k,
                g_k_plus_1=g_k_plus_1_invalid,
            )
        
        assert "dimensión" in str(exc_info.value) or "shape" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.2 — VALIDACIÓN DE SIMETRÍA MÉTRICA
    # -------------------------------------------------------------------------
    
    def test_phase1_symmetry_g_k_valid(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k simétrica pasa validación.
        
        Axioma: g_k = g_kᵀ dentro de tolerancia ε_mach · ‖g_k‖_F
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        # Verificar simetría explícita
        sym_residual = float(la.norm(g_k - g_k.T, "fro"))
        norm_g = float(la.norm(g_k, "fro"))
        tol = _MACHINE_EPS * max(norm_g, 1.0)
        
        assert sym_residual <= tol, f"Fixture g_k no es simétrica: {sym_residual}"
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit is not None
    
    def test_phase1_symmetry_g_k_invalid(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k asimétrica dispara MetricDegeneracyError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_invalid: NDArray[np.float64] = g_k.copy()
        g_k_invalid[0, 1] += 0.5  # Romper simetría
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(MetricDegeneracyError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_invalid,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "simétrica" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.3 — VALIDACIÓN DE DEFINIDA POSITIVA (SPD)
    # -------------------------------------------------------------------------
    
    def test_phase1_spd_g_k_valid(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k SPD pasa validación.
        
        Axioma: λ_min(g_k) > 0 (todos autovalores positivos)
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        # Verificar SPD explícito
        eigvals = la.eigvalsh(g_k)
        lambda_min = float(np.min(eigvals))
        
        assert lambda_min > 0.0, f"Fixture g_k no es SPD: λ_min={lambda_min}"
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit.condition_number_g_k < 1e10
    
    def test_phase1_spd_g_k_invalid_negative_eigenvalue(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k con autovalor negativo dispara MetricDegeneracyError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_invalid: NDArray[np.float64] = g_k.copy()
        g_k_invalid[0, 0] = -1.0  # Forzar λ_min < 0
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(MetricDegeneracyError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_invalid,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "definida positiva" in str(exc_info.value) or "SPD" in str(exc_info.value)
    
    def test_phase1_spd_g_k_near_singular(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g_k casi singular dispara MetricDegeneracyError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_singular: NDArray[np.float64] = np.array(
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0]], dtype=np.float64
        )  # rank 1, λ_min = 0
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(MetricDegeneracyError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_singular,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "singular" in str(exc_info.value) or "degenerada" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.4 — CONVERGENCIA DEL FLUJO DE RICCI
    # -------------------------------------------------------------------------
    
    def test_phase1_ricci_convergence_valid(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que flujo de Ricci converge dentro de tolerancia.
        
        Condición: ||g_{k+1} - g_k||_F / scale < ε_Ricci
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit.is_metric_converged is True
        assert ricci_audit.metric_relative_residual < 1e-8
    
    def test_phase1_ricci_divergence_raises(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que divergencia del flujo de Ricci dispara RicciFlowDivergenceError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_plus_1_divergent: NDArray[np.float64] = g_k_plus_1 * 100.0  # Cambio grande
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(RicciFlowDivergenceError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k,
                g_k_plus_1=g_k_plus_1_divergent,
            )
        
        assert "convergió" in str(exc_info.value) or "Divergencia" in str(exc_info.value)
    
    def test_phase1_ricci_residual_computed(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que residuo relativo del flujo de Ricci se calcula correctamente.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        # Verificar cálculo manual del residuo
        diff = g_k_plus_1 - g_k
        residual_norm = float(la.norm(diff, "fro"))
        scale = max(1.0, float(la.norm(g_k, "fro")), float(la.norm(g_k_plus_1, "fro")))
        expected_relative = residual_norm / scale
        
        assert abs(ricci_audit.metric_relative_residual - expected_relative) < 1e-12
        assert ricci_audit.metric_residual_norm == residual_norm
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.5 — NÚMEROS DE CONDICIÓN
    # -------------------------------------------------------------------------
    
    def test_phase1_condition_numbers_computed(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que números de condición κ(g_k), κ(g_k_plus_1) se calculan.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert ricci_audit.condition_number_g_k >= 1.0
        assert ricci_audit.condition_number_g_k_plus_1 >= 1.0
        assert np.isfinite(ricci_audit.condition_number_g_k)
        assert np.isfinite(ricci_audit.condition_number_g_k_plus_1)
    
    def test_phase1_condition_number_ill_conditioned(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que métrica mal condicionada dispara MetricDegeneracyError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_ill: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1e-12, 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )  # κ ≈ 1e12
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(MetricDegeneracyError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_ill,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "condición" in str(exc_info.value) or "κ" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.6 — VALIDACIÓN DE FINITUD NUMÉRICA
    # -------------------------------------------------------------------------
    
    def test_phase1_finite_values_valid(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que métricas con valores finitos pasan validación.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        assert np.all(np.isfinite(ricci_audit.metric_residual_norm))
        assert np.all(np.isfinite(ricci_audit.metric_relative_residual))
    
    def test_phase1_nan_values_raise(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que NaN en métricas dispara GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_nan: NDArray[np.float64] = g_k.copy()
        g_k_nan[0, 0] = np.nan
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_nan,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "NaN" in str(exc_info.value) or "infinitos" in str(exc_info.value)
    
    def test_phase1_inf_values_raise(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Inf en métricas dispara GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_inf: NDArray[np.float64] = g_k.copy()
        g_k_inf[0, 0] = np.inf
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_inf,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "infinitos" in str(exc_info.value) or "NaN" in str(exc_info.value)
    
    def test_phase1_complex_values_raise(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que valores complejos disparan GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        g_k_complex: NDArray[np.complex128] = g_k.astype(np.complex128)
        g_k_complex[0, 0] += 0.1j
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._audit_ricci_flow_convergence(
                g_k=g_k_complex,
                g_k_plus_1=g_k_plus_1,
            )
        
        assert "compleja" in str(exc_info.value) or "real" in str(exc_info.value)


# =============================================================================
# FASE 2 — CERTIFICACIÓN DE LA ACCIÓN DE POLYAKOV
# =============================================================================
class TestPhase2_PolyakovActionCertifier:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 2 — ACCIÓN GEODÉSICA DE POLYAKOV Y ESTABILIDAD CINEMÁTICA
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase2_PolyakovActionCertifier que gobierna
    la minimización covariante de la energía geodésica. Cada método verifica un axioma
    constitutivo de la acción de Polyakov.
    
    Invariantes Verificados:
    ------------------------
    1. Energía geodésica E[γ] ≥ 0
    2. Términos cinéticos vᵀ G v ≥ 0
    3. Homogeneidad de la forma cuadrática
    4. Consistencia dimensional con certificado de Fase 1
    5. Techo de energía E[γ] ≤ E_ceiling
    6. Continuidad formal desde RicciFlowAuditData
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.1 — ENERGÍA GEODÉSICA Y TÉRMINOS CINÉTICOS
    # -------------------------------------------------------------------------
    
    def test_phase2_geodesic_energy_nonnegative(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que energía geodésica E[γ] ≥ 0 para todo v.
        
        Axioma: E[γ] = ½ τ Σ v_iᵀ G v_i ≥ 0 (G SPD)
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        # Fase 1 primero
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        # Fase 2
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.geodesic_energy >= 0.0, \
            f"E[γ] = {polyakov_audit.geodesic_energy} < 0"
    
    def test_phase2_kinetic_terms_nonnegative(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que términos cinéticos vᵀ G v ≥ 0 para cada paso.
        
        Axioma: v_iᵀ G v_i ≥ 0 ∀ i (G SPD)
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.min_kinetic_term >= 0.0
        assert polyakov_audit.max_kinetic_term >= 0.0
    
    def test_phase2_geodesic_energy_ceiling(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que E[γ] ≤ E_ceiling (techo de energía admisible).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.geodesic_energy <= polyakov_audit.energy_ceiling
    
    def test_phase2_geodesic_energy_exceeds_ceiling_raises(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que E[γ] > E_ceiling dispara PolyakovActionViolationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Velocidades muy grandes para exceder techo
        velocities_large: NDArray[np.float64] = np.ones((5, 3), dtype=np.float64) * 1e6
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        with pytest.raises(PolyakovActionViolationError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities_large,
                g_metric=g_k_plus_1,
                d_tau=d_tau,
                ricci_audit=ricci_audit,
            )
        
        assert "energía" in str(exc_info.value).lower() or "Polyakov" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.2 — CONSISTENCIA DIMENSIONAL CON FASE 1
    # -------------------------------------------------------------------------
    
    def test_phase2_dimension_consistency_with_phase1(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que dimensión de velocidades coincide con certificado de Fase 1.
        
        Contrato: dim(velocities) == ricci_audit.dimension
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.dimension == ricci_audit.dimension
    
    def test_phase2_dimension_mismatch_with_phase1_raises(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que inconsistencia dimensional dispara GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Métrica de dimensión diferente
        g_wrong: NDArray[np.float64] = np.eye(2, dtype=np.float64)  # 2×2 ≠ 3×3
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities,
                g_metric=g_wrong,
                d_tau=d_tau,
                ricci_audit=ricci_audit,
            )
        
        assert "dimensión" in str(exc_info.value) or "inconsistente" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.3 — CONTINUIDAD FORMAL DESDE FASE 1
    # -------------------------------------------------------------------------
    
    def test_phase2_requires_phase1_convergence(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que Fase 2 requiere convergencia certificada de Fase 1.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        # Forzar is_metric_converged = False (simulado)
        from dataclasses import replace
        ricci_audit_failed = replace(ricci_audit, is_metric_converged=False)
        
        with pytest.raises(RicciFlowDivergenceError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities,
                g_metric=g_k_plus_1,
                d_tau=d_tau,
                ricci_audit=ricci_audit_failed,
            )
        
        assert "Fase 1" in str(exc_info.value) or "convergencia" in str(exc_info.value)
    
    def test_phase2_works_without_phase1_audit(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que Fase 2 puede operar sin certificado de Fase 1 (ricci_audit=None).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        # Sin ricci_audit
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=None,
        )
        
        assert polyakov_audit.is_geodesic_stable is True
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.4 — VALIDACIÓN DE PARÁMETROS ESCALARES
    # -------------------------------------------------------------------------
    
    def test_phase2_d_tau_positive(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que d_tau > 0 es requerido.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities,
                g_metric=g_k_plus_1,
                d_tau=0.0,  # No positivo
                ricci_audit=ricci_audit,
            )
        
        assert "positivo" in str(exc_info.value) or "d_tau" in str(exc_info.value)
    
    def test_phase2_d_tau_negative_raises(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que d_tau < 0 dispara GeodesicInputValidationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities,
                g_metric=g_k_plus_1,
                d_tau=-0.01,  # Negativo
                ricci_audit=ricci_audit,
            )
        
        assert "positivo" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.5 — VALIDACIÓN DE MATRIZ DE VELOCIDADES
    # -------------------------------------------------------------------------
    
    def test_phase2_velocity_matrix_1d_accepted(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_1d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que vector 1D de velocidades es aceptado (steps=1).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_1d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.steps == 1
        assert polyakov_audit.dimension == 3
    
    def test_phase2_velocity_matrix_empty_raises(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que matriz de velocidades vacía dispara PolyakovActionViolationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        velocities_empty: NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 3)
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._certify_polyakov_geodesic_action(
                geodesic_velocity_matrix=velocities_empty,
                g_metric=g_k_plus_1,
                d_tau=d_tau,
                ricci_audit=ricci_audit,
            )
        
        assert "vacío" in str(exc_info.value) or "empty" in str(exc_info.value)


# =============================================================================
# FASE 3 — VETO CUÁNTICO DE FEYNMAN-KAC
# =============================================================================
class TestPhase3_FeynmanKacQuantumVeto:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 3 — AMPLITUD DE TRANSICIÓN CUÁNTICA Y VETO DE FEYNMAN-KAC
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase3_FeynmanKacQuantumVeto que fuerza
    la admisibilidad cuántica de la transición semántica. Cada método verifica un axioma
    constitutivo del veto cuántico.
    
    Invariantes Verificados:
    ------------------------
    1. Acción euclídea S_E ≥ 0
    2. Amplitud de transición Ψ ≥ Ψ_min
    3. Log-amplitud finita y computable
    4. Consistencia energética con Fase 2
    5. ħ_eff > 0
    6. Continuidad formal desde PolyakovActionAuditData
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.1 — ACCIÓN EUCLÍDEA Y AMPLITUD
    # -------------------------------------------------------------------------
    
    def test_phase3_euclidean_action_nonnegative(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que acción euclídea S_E ≥ 0.
        
        Fórmula: S_E = E_Polyakov + λ ||T||²_HS
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        # Fases 1 y 2
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        # Fase 3
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
            polyakov_audit=polyakov_audit,
        )
        
        assert feynman_kac_audit.euclidean_action >= 0.0
    
    def test_phase3_transition_amplitude_above_minimum(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que Ψ ≥ Ψ_min (amplitud admisible).
        
        Fórmula: Ψ = exp(-S_E / ħ_eff) ≥ Ψ_min
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
            polyakov_audit=polyakov_audit,
        )
        
        assert feynman_kac_audit.transition_amplitude >= feynman_kac_audit.min_quantum_amplitude
        assert feynman_kac_audit.is_attention_allowed is True
    
    def test_phase3_log_amplitude_finite(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que log(Ψ) es finito.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
            polyakov_audit=polyakov_audit,
        )
        
        assert np.isfinite(feynman_kac_audit.log_transition_amplitude)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.2 — VETO CUÁNTICO (AMPLITUD INSUFICIENTE)
    # -------------------------------------------------------------------------
    
    def test_phase3_quantum_veto_low_amplitude_raises(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que Ψ < Ψ_min dispara QuantumFeynmanKacVeto.
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Energía muy alta para reducir amplitud
        polyakov_energy_high: float = 1e10
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(QuantumFeynmanKacVeto) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=polyakov_energy_high,
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=lambda_coupling,
                polyakov_audit=None,
            )
        
        assert "Veto" in str(exc_info.value) or "amplitud" in str(exc_info.value).lower()
    
    def test_phase3_quantum_veto_high_torsion_raises(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que torsión excesiva dispara QuantumFeynmanKacVeto.
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Torsión muy alta
        torsion_high: float = 1e10
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(QuantumFeynmanKacVeto) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=0.01,
                torsion_hs_norm_sq=torsion_high,
                lambda_coupling=lambda_coupling,
                polyakov_audit=None,
            )
        
        assert "Veto" in str(exc_info.value) or "amplitud" in str(exc_info.value).lower()
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.3 — CONSISTENCIA ENERGÉTICA CON FASE 2
    # -------------------------------------------------------------------------
    
    def test_phase3_energy_consistency_with_phase2(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que energía en Fase 3 coincide con certificado de Fase 2.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        # Energía consistente
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
            polyakov_audit=polyakov_audit,
        )
        
        assert feynman_kac_audit.is_attention_allowed is True
    
    def test_phase3_energy_inconsistency_raises(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que inconsistencia energética dispara PolyakovActionViolationError.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        # Energía inconsistente
        with pytest.raises(PolyakovActionViolationError) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=polyakov_audit.geodesic_energy * 2.0,  # Diferente
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=lambda_coupling,
                polyakov_audit=polyakov_audit,
            )
        
        assert "inconsistencia" in str(exc_info.value).lower() or "energética" in str(exc_info.value).lower()
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.4 — VALIDACIÓN DE PARÁMETROS CUÁNTICOS
    # -------------------------------------------------------------------------
    
    def test_phase3_lambda_coupling_nonnegative(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que lambda_coupling ≥ 0 es requerido.
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=0.01,
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=-0.1,  # Negativo
                polyakov_audit=None,
            )
        
        assert "no negativo" in str(exc_info.value) or "negativo" in str(exc_info.value)
    
    def test_phase3_torsion_norm_nonnegative(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que torsion_hs_norm_sq ≥ 0 es requerido.
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(GeodesicInputValidationError) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=0.01,
                torsion_hs_norm_sq=-0.05,  # Negativo
                lambda_coupling=lambda_coupling,
                polyakov_audit=None,
            )
        
        assert "no negativo" in str(exc_info.value) or "negativo" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.5 — CONTINUIDAD FORMAL DESDE FASE 2
    # -------------------------------------------------------------------------
    
    def test_phase3_requires_phase2_stability(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que Fase 3 requiere estabilidad certificada de Fase 2.
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        # Certificado de Fase 2 con is_geodesic_stable = False
        from dataclasses import replace
        from app.agents.boole.wisdom.geodesic_attention_fibrator_agent import PolyakovActionAuditData
        
        polyakov_audit_failed = PolyakovActionAuditData(
            steps=5,
            dimension=3,
            geodesic_energy=0.01,
            min_kinetic_term=0.0,
            max_kinetic_term=0.1,
            energy_ceiling=1e6,
            polyakov_tolerance=1e-12,
            is_geodesic_stable=False,  # Inestable
        )
        
        with pytest.raises(PolyakovActionViolationError) as exc_info:
            agent._enforce_feynman_kac_quantum_veto(
                polyakov_energy=0.01,
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=lambda_coupling,
                polyakov_audit=polyakov_audit_failed,
            )
        
        assert "Fase 2" in str(exc_info.value) or "estabilidad" in str(exc_info.value)
    
    def test_phase3_works_without_phase2_audit(
        self,
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que Fase 3 puede operar sin certificado de Fase 2 (polyakov_audit=None).
        """
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        # Sin polyakov_audit
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=0.01,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
            polyakov_audit=None,
        )
        
        assert feynman_kac_audit.is_attention_allowed is True


# =============================================================================
# PRUEBAS DE INTEGRACIÓN — PIPELINE COMPLETO
# =============================================================================
class TestFullPipeline_Integration:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    PRUEBAS DE INTEGRACIÓN — COMPOSICIÓN FUNTORIAL Φ₃ ∘ Φ₂ ∘ Φ₁
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida la composición funtorial estricta del agente completo.
    Cada método verifica que la cadena de tres fases opera correctamente en conjunto.
    """
    
    def test_full_pipeline_valid_inputs(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica pipeline completo con entradas válidas.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        state = agent.execute_geodesic_attention_governance(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
            geodesic_velocity_matrix=velocities,
            d_tau=d_tau,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
        )
        
        assert state.is_epistemologically_valid is True
        assert state.ricci_audit.is_metric_converged is True
        assert state.polyakov_audit.is_geodesic_stable is True
        assert state.feynman_kac_audit.is_attention_allowed is True
    
    def test_full_pipeline_ricci_divergence_fails(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que divergencia de Ricci falla el pipeline completo.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        g_k_plus_1_divergent: NDArray[np.float64] = g_k_plus_1 * 100.0
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(RicciFlowDivergenceError):
            agent.execute_geodesic_attention_governance(
                g_k=g_k,
                g_k_plus_1=g_k_plus_1_divergent,
                geodesic_velocity_matrix=velocities,
                d_tau=d_tau,
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=lambda_coupling,
            )
    
    def test_full_pipeline_polyakov_violation_fails(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que violación de Polyakov falla el pipeline completo.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Velocidades muy grandes
        velocities_large: NDArray[np.float64] = velocities * 1e6
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(PolyakovActionViolationError):
            agent.execute_geodesic_attention_governance(
                g_k=g_k,
                g_k_plus_1=g_k_plus_1,
                geodesic_velocity_matrix=velocities_large,
                d_tau=d_tau,
                torsion_hs_norm_sq=torsion_hs_norm_sq,
                lambda_coupling=lambda_coupling,
            )
    
    def test_full_pipeline_quantum_veto_fails(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que veto cuántico falla el pipeline completo.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Torsión muy alta
        torsion_high: float = 1e10
        
        agent = GeodesicAttentionFibratorAgent()
        
        with pytest.raises(QuantumFeynmanKacVeto):
            agent.execute_geodesic_attention_governance(
                g_k=g_k,
                g_k_plus_1=g_k_plus_1,
                geodesic_velocity_matrix=velocities,
                d_tau=d_tau,
                torsion_hs_norm_sq=torsion_high,
                lambda_coupling=lambda_coupling,
            )
    
    def test_full_pipeline_dto_immutability(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica que DTOs son inmutables (frozen dataclasses).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        state = agent.execute_geodesic_attention_governance(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
            geodesic_velocity_matrix=velocities,
            d_tau=d_tau,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
        )
        
        # Intentar modificar debe fallar (frozen=True)
        with pytest.raises(AttributeError):
            state.is_epistemologically_valid = False  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.ricci_audit.dimension = 999  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.polyakov_audit.geodesic_energy = 999.0  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.feynman_kac_audit.transition_amplitude = 0.0  # type: ignore[misc]
    
    def test_full_pipeline_audit_data_consistency(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica consistencia entre certificados de las tres fases.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        state = agent.execute_geodesic_attention_governance(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
            geodesic_velocity_matrix=velocities,
            d_tau=d_tau,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
        )
        
        # Dimensiones consistentes
        assert state.ricci_audit.dimension == state.polyakov_audit.dimension
        
        # Energías consistentes
        assert state.polyakov_audit.geodesic_energy >= 0.0
        assert state.feynman_kac_audit.euclidean_action >= state.polyakov_audit.geodesic_energy
        
        # Validez epistemológica implica todas las fases válidas
        if state.is_epistemologically_valid:
            assert state.ricci_audit.is_metric_converged
            assert state.polyakov_audit.is_geodesic_stable
            assert state.feynman_kac_audit.is_attention_allowed


# =============================================================================
# PRUEBAS DE CASOS ESPECIALES Y BORDES
# =============================================================================
class TestEdgeCases_SpecialConditions:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    PRUEBAS DE CASOS ESPECIALES Y BORDES
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida comportamiento en condiciones límite:
    - Métricas casi singulares
    - Estados cero
    - Tolerancias numéricas
    - Valores extremos
    """
    
    def test_edge_case_zero_velocity(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento con velocidades cero v = 0.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        velocities_zero: NDArray[np.float64] = np.zeros((5, 3), dtype=np.float64)
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities_zero,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.geodesic_energy == 0.0
        assert polyakov_audit.min_kinetic_term == 0.0
        assert polyakov_audit.max_kinetic_term == 0.0
    
    def test_edge_case_identity_metrics(
        self,
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento con g_k = g_k_plus_1 = I (métrica euclidiana).
        """
        dim = 3
        g_k: NDArray[np.float64] = np.eye(dim, dtype=np.float64)
        g_k_plus_1: NDArray[np.float64] = np.eye(dim, dtype=np.float64)
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        # Residuo debe ser muy pequeño (cero teórico)
        assert ricci_audit.metric_relative_residual < 1e-14
        assert ricci_audit.is_metric_converged is True
    
    def test_edge_case_minimum_dimension(
        self,
        fixture_valid_metrics_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento con dimensión mínima (n=2).
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_2d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        velocities: NDArray[np.float64] = np.array(
            [[0.1, 0.05],
             [0.12, 0.06]], dtype=np.float64
        )
        
        agent = GeodesicAttentionFibratorAgent()
        
        state = agent.execute_geodesic_attention_governance(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
            geodesic_velocity_matrix=velocities,
            d_tau=d_tau,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling,
        )
        
        assert state.ricci_audit.dimension == 2
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_tolerance_boundaries(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento en límites de tolerancia numérica.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        # Métricas muy cercanas (en límite de convergencia)
        g_k_plus_1_boundary: NDArray[np.float64] = g_k + 1e-9 * np.eye(3, dtype=np.float64)
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1_boundary,
        )
        
        assert ricci_audit.is_metric_converged is True
        assert ricci_audit.metric_relative_residual < 1e-8
    
    def test_edge_case_very_small_d_tau(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento con d_tau muy pequeño.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        d_tau_small: float = 1e-12
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau_small,
            ricci_audit=ricci_audit,
        )
        
        assert polyakov_audit.geodesic_energy >= 0.0
        assert np.isfinite(polyakov_audit.geodesic_energy)
    
    def test_edge_case_very_small_coupling(
        self,
        fixture_valid_metrics_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_geodesic_velocity_3d: NDArray[np.float64],
        fixture_valid_scalar_params: Tuple[float, float, float],
    ) -> None:
        r"""
        Verifica comportamiento con lambda_coupling muy pequeño.
        """
        g_k, g_k_plus_1 = fixture_valid_metrics_3d
        velocities = fixture_valid_geodesic_velocity_3d
        d_tau, torsion_hs_norm_sq, lambda_coupling = fixture_valid_scalar_params
        
        lambda_coupling_small: float = 1e-12
        
        agent = GeodesicAttentionFibratorAgent()
        
        ricci_audit = agent._audit_ricci_flow_convergence(
            g_k=g_k,
            g_k_plus_1=g_k_plus_1,
        )
        
        polyakov_audit = agent._certify_polyakov_geodesic_action(
            geodesic_velocity_matrix=velocities,
            g_metric=g_k_plus_1,
            d_tau=d_tau,
            ricci_audit=ricci_audit,
        )
        
        feynman_kac_audit = agent._enforce_feynman_kac_quantum_veto(
            polyakov_energy=polyakov_audit.geodesic_energy,
            torsion_hs_norm_sq=torsion_hs_norm_sq,
            lambda_coupling=lambda_coupling_small,
            polyakov_audit=polyakov_audit,
        )
        
        assert feynman_kac_audit.is_attention_allowed is True
        assert np.isfinite(feynman_kac_audit.euclidean_action)


# =============================================================================
# EJECUCIÓN DIRECTA (para debugging)
# =============================================================================
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
    ])
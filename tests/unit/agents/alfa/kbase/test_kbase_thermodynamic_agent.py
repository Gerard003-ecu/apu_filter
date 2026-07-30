# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — KBase Thermodynamic Agent (Suite de Validación Termodinámica)     ║
║  Ruta   : tests/unit/agents/alfa/kbase/test_kbase_thermodynamic_agent.py                 ║
║  Versión: 5.0.0-Rigorous-Sheaf-Williamson-Boolean-Spectral-Passivity-TestSuite           ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPÓSITO CIBER-FÍSICO Y TOPOLOGÍA DE PRUEBAS (Rigor Categórico):                       ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Esta suite de pruebas consagra el Foso Termodinámico del ecosistema ($K_{BASE}$)        ║
║  mediante un funtor de validación que verifica axiomáticamente la inercia (Recursos),    ║
║  la capacitancia (Socios) y la fricción entrópica (Costos) del modelo de negocio.        ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):     ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Validación de Topología Matricial y Métrica Riemanniana                        ║
║           Verifica dimensiones, simetrías, pullback congruente y SPD.                    ║
║                                                                                          ║
║  Fase 2 → Validación de Dinámica Port-Hamiltoniana y Disipación de Rayleigh              ║
║           Computa H_BASE, certifica P_diss y extrae frecuencias ω_i.                     ║
║                                                                                          ║
║  Fase 3 → Validación de Fibración Celular y Exportación de la Cofrontera                 ║
║           Sintetiza la cocadena δ_BASE subyugada a la Identidad de Hodge.                ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import logging
from typing import Tuple
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
from app.agents.alpha.kbase.kbase_thermodynamic_agent import (
    KBaseThermodynamicAgent,
    TopologicalContext,
    BasalStateTensor,
    SheafStalk,
    StabilityFlags,
    # Excepciones
    ThermodynamicBaseError,
    DimensionMismatchError,
    CapacitanceDegeneracyError,
    InertialFlybackError,
    RayleighDissipationViolation,
    IllConditionedMatrixError,
    MetricTensorSingularityError,
    SheafCoboundaryError,
    StructuralConsistencyError,
    PassivityCertificateError,
    WilliamsonNormalFormError,
)

# =============================================================================
# Logger y constantes globales de prueba
# =============================================================================
logger = logging.getLogger("MIC.Alpha.Test.KBaseThermodynamicAgent")
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)

# =============================================================================
# FIXTURES GLOBALES — GENERADORES DE MATRICES CONSTITUTIVAS
# =============================================================================


@pytest.fixture(scope="module")
def fixture_valid_matrices_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices constitutivas válidas para dim_q=2, dim_p=2.
    
    Retorna
    -------
    Tuple[C_soc, M_rec, R_cost, J_base]
        Matrices SPD, PSD, antisimétrica respectivamente.
    """
    dim_q, dim_p = 2, 2
    n = dim_q + dim_p
    
    # C_soc: SPD (capacitancia de socios)
    C_soc: NDArray[np.float64] = np.array(
        [[2.0, 0.1], [0.1, 1.5]], dtype=np.float64
    )
    
    # M_rec: SPD (inercia de recursos)
    M_rec: NDArray[np.float64] = np.array(
        [[1.8, 0.05], [0.05, 2.2]], dtype=np.float64
    )
    
    # R_cost: PSD (disipación de costos)
    R_cost: NDArray[np.float64] = np.array(
        [[0.5, 0.0, 0.0, 0.0],
         [0.0, 0.3, 0.0, 0.0],
         [0.0, 0.0, 0.4, 0.0],
         [0.0, 0.0, 0.0, 0.2]], dtype=np.float64
    )
    
    # J_base: antisimétrica (interconexión)
    J_base: NDArray[np.float64] = np.array(
        [[0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [-1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0]], dtype=np.float64
    )
    
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="module")
def fixture_valid_matrices_1d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices constitutivas válidas para dim_q=1, dim_p=1 (caso mínimo).
    
    Retorna
    -------
    Tuple[C_soc, M_rec, R_cost, J_base]
    """
    C_soc: NDArray[np.float64] = np.array([[2.0]], dtype=np.float64)
    M_rec: NDArray[np.float64] = np.array([[1.5]], dtype=np.float64)
    R_cost: NDArray[np.float64] = np.array(
        [[0.3, 0.0], [0.0, 0.2]], dtype=np.float64
    )
    J_base: NDArray[np.float64] = np.array(
        [[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64
    )
    
    return C_soc, M_rec, R_cost, J_base


@pytest.fixture(scope="module")
def fixture_valid_state_vectors_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera vectores de estado válidos para dim_q=2, dim_p=2.
    
    Retorna
    -------
    Tuple[q, p, df_dt]
    """
    q: NDArray[np.float64] = np.array([1.0, 0.5], dtype=np.float64)
    p: NDArray[np.float64] = np.array([0.8, 1.2], dtype=np.float64)
    df_dt: NDArray[np.float64] = np.array([0.1, 0.05], dtype=np.float64)
    
    return q, p, df_dt


@pytest.fixture(scope="module")
def fixture_metric_tensors_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera tensores métricos Riemannianos válidos G_q, G_p.
    
    Retorna
    -------
    Tuple[G_q, G_p]
    """
    G_q: NDArray[np.float64] = np.array(
        [[1.1, 0.05], [0.05, 0.9]], dtype=np.float64
    )
    G_p: NDArray[np.float64] = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float64
    )
    
    return G_q, G_p


# =============================================================================
# FASE 1 — VALIDACIÓN DE TOPOLOGÍA MATRICIAL Y MÉTRICA RIEMANNIANA
# =============================================================================
class TestPhase1_MatrixTopology:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 1 — TOPOLOGÍA MATRICIAL, PULLBACK RIEMANNIANO Y VALIDACIÓN ESPECTRAL
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase1_MatrixTopology que consagra la
    geometría Riemanniana del foso termodinámico. Cada método verifica un axioma
    constitutivo del estrato K_BASE.
    
    Invariantes Verificados:
    ------------------------
    1. Coherencia dimensional de C_soc, M_rec, R_cost, J_base
    2. Simetría de C_soc, M_rec, R_cost (A = Aᵀ)
    3. Antisimetría de J_base (J = −Jᵀ)
    4. Invertibilidad de tensores métricos G_q, G_p
    5. Pullback congruente: Ã = G A Gᵀ
    6. SPD post-pullback (Ley de Sylvester)
    7. Números de condición κ < κ_max
    8. Cholesky regularizado con Tikhonov adaptativo
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.1 — VALIDACIÓN DIMENSIONAL Y ESTRUCTURAL
    # -------------------------------------------------------------------------
    
    def test_phase1_dimensions_valid_2d(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que matrices 2D válidas pasan la validación dimensional.
        
        Axioma: dim(C_soc) = dim_q², dim(M_rec) = dim_p², 
                dim(R_cost) = dim(J_base) = (dim_q + dim_p)²
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        assert agent.context.dim_q == 2
        assert agent.context.dim_p == 2
        assert agent.context.C_tilde.shape == (2, 2)
        assert agent.context.M_tilde.shape == (2, 2)
        assert agent.context.R_cost.shape == (4, 4)
        assert agent.context.J_base.shape == (4, 4)
    
    def test_phase1_dimensions_valid_1d(
        self,
        fixture_valid_matrices_1d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica caso mínimo dim_q=1, dim_p=1 (frontera inferior).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_1d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        assert agent.context.dim_q == 1
        assert agent.context.dim_p == 1
        assert agent.context.C_tilde.shape == (1, 1)
        assert agent.context.M_tilde.shape == (1, 1)
    
    def test_phase1_dimension_mismatch_C_soc_non_square(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que C_soc no cuadrada dispara DimensionMismatchError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        C_soc_invalid: NDArray[np.float64] = C_soc[:, :1]  # 2×1
        
        with pytest.raises(DimensionMismatchError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc_invalid,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )
        
        assert "C_soc debe ser cuadrada" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_R_cost_shape(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R_cost con forma incorrecta dispara DimensionMismatchError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        R_cost_invalid: NDArray[np.float64] = np.eye(3, dtype=np.float64)  # 3×3 ≠ 4×4
        
        with pytest.raises(DimensionMismatchError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost_invalid,
                J_base=J_base,
            )
        
        assert "R_cost debe ser" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.2 — VALIDACIÓN DE SIMETRÍAS ESTRUCTURALES
    # -------------------------------------------------------------------------
    
    def test_phase1_symmetry_C_soc_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que C_soc simétrica pasa validación.
        
        Axioma: C_soc = C_socᵀ dentro de tolerancia ε_mach · ‖C_soc‖_F
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        # Verificar simetría explícita
        sym_residual = float(la.norm(C_soc - C_soc.T, "fro"))
        norm_C = float(la.norm(C_soc, "fro"))
        tol = _MACHINE_EPS * max(norm_C, 1.0)
        
        assert sym_residual <= tol, f"Fixture C_soc no es simétrica: {sym_residual}"
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        assert agent is not None
    
    def test_phase1_symmetry_C_soc_invalid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que C_soc asimétrica dispara ThermodynamicBaseError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        C_soc_invalid: NDArray[np.float64] = C_soc.copy()
        C_soc_invalid[0, 1] += 0.5  # Romper simetría
        
        with pytest.raises(ThermodynamicBaseError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc_invalid,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
            )
        
        assert "no es simétrica" in str(exc_info.value)
    
    def test_phase1_antisymmetry_J_base_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J_base antisimétrica pasa validación.
        
        Axioma: J_base = −J_baseᵀ garantiza ∇Hᵀ J ∇H ≡ 0
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        # Verificar antisimetría explícita
        skew_residual = float(la.norm(J_base + J_base.T, "fro"))
        norm_J = float(la.norm(J_base, "fro"))
        tol = _MACHINE_EPS * max(norm_J, 1.0)
        
        assert skew_residual <= tol, f"Fixture J_base no es antisimétrica: {skew_residual}"
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        assert agent is not None
    
    def test_phase1_antisymmetry_J_base_invalid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J_base no antisimétrica dispara ThermodynamicBaseError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        J_base_invalid: NDArray[np.float64] = J_base.copy()
        J_base_invalid[0, 1] += 0.3  # Romper antisimetría
        
        with pytest.raises(ThermodynamicBaseError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base_invalid,
            )
        
        assert "no es antisimétrica" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.3 — VALIDACIÓN DE TENSORES MÉTRICOS RIEMANNIANOS
    # -------------------------------------------------------------------------
    
    def test_phase1_metric_tensor_G_q_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_metric_tensors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que tensores métricos válidos pasan validación.
        
        Axioma: κ(G) = σ_max/σ_min < κ_max para preservar signatura SPD
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        G_q, G_p = fixture_metric_tensors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        
        assert agent.context.kappa_G_q < agent.kappa_max
        assert agent.context.kappa_G_p < agent.kappa_max
    
    def test_phase1_metric_tensor_G_q_singular(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que tensor métrico singular dispara MetricTensorSingularityError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        G_q_singular: NDArray[np.float64] = np.array(
            [[1.0, 1.0], [1.0, 1.0]], dtype=np.float64
        )  # rank 1
        
        with pytest.raises(MetricTensorSingularityError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                G_q=G_q_singular,
            )
        
        assert "singular o casi singular" in str(exc_info.value)
    
    def test_phase1_metric_tensor_G_q_ill_conditioned(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que tensor métrico mal condicionado dispara MetricTensorSingularityError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        G_q_ill: NDArray[np.float64] = np.array(
            [[1.0, 0.0], [0.0, 1e-12]], dtype=np.float64
        )  # κ ≈ 1e12
        
        with pytest.raises(MetricTensorSingularityError) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost,
                J_base=J_base,
                G_q=G_q_ill,
                kappa_max=1e10,
            )
        
        assert "mal condicionado" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.4 — PULLBACK CONGRUENTE Y VALIDACIÓN ESPECTRAL
    # -------------------------------------------------------------------------
    
    def test_phase1_pullback_congruence_preserves_spd(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_metric_tensors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que pullback congruente preserva signatura SPD.
        
        Teorema: Ley de Inercia de Sylvester — Ã = G A Gᵀ tiene misma signatura que A
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        G_q, G_p = fixture_metric_tensors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        
        # Verificar que C̃ y M̃ son SPD (λ_min > 0)
        lambda_min_C = float(la.eigvalsh(agent.context.C_tilde)[0])
        lambda_min_M = float(la.eigvalsh(agent.context.M_tilde)[0])
        
        assert lambda_min_C > 0, f"C̃ no es SPD: λ_min={lambda_min_C}"
        assert lambda_min_M > 0, f"M̃ no es SPD: λ_min={lambda_min_M}"
    
    def test_phase1_pullback_amplification_measured(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_metric_tensors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que amplificación de κ por pullback se mide correctamente.
        
        Métrica: amp = κ(Ã) / κ(A) (≈ 1 si G ≈ I)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        G_q, G_p = fixture_metric_tensors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        
        assert agent.context.pullback_amp_C >= 0.0
        assert agent.context.pullback_amp_M >= 0.0
        assert np.isfinite(agent.context.pullback_amp_C)
        assert np.isfinite(agent.context.pullback_amp_M)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.5 — CHOLESKY REGULARIZADO Y TIKHONOV ADAPTATIVO
    # -------------------------------------------------------------------------
    
    def test_phase1_cholesky_without_regularization(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que matrices SPD bien condicionadas no requieren Tikhonov.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        # epsilon_C y epsilon_M deben ser 0 si no se requirió regularización
        assert agent.context.epsilon_C == 0.0
        assert agent.context.epsilon_M == 0.0
    
    def test_phase1_cholesky_factors_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que factores de Cholesky reconstruyen las matrices originales.
        
        Invariante: C̃ = L_C L_Cᵀ, M̃ = L_M L_Mᵀ
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        C_reconstructed = agent.context.L_C @ agent.context.L_C.T
        M_reconstructed = agent.context.L_M @ agent.context.L_M.T
        
        tol = 1e-10 * max(
            float(la.norm(agent.context.C_tilde, "fro")),
            float(la.norm(agent.context.M_tilde, "fro")),
            1.0
        )
        
        assert float(la.norm(C_reconstructed - agent.context.C_tilde, "fro")) <= tol
        assert float(la.norm(M_reconstructed - agent.context.M_tilde, "fro")) <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.6 — DIAGNÓSTICOS ESPECTRALES DE R_cost
    # -------------------------------------------------------------------------
    
    def test_phase1_R_cost_psd_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R_cost PSD pasa validación.
        
        Axioma: λ_min(R_cost) ≥ −tol (Segunda Ley de Termodinámica)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        assert agent.context.rank_R >= 0
        assert agent.context.spectral_gap_R >= 0.0
        assert agent.context.spectral_entropy_R >= 0.0
    
    def test_phase1_R_cost_not_psd_raises(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R_cost no PSD dispara RayleighDissipationViolation.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        R_cost_invalid: NDArray[np.float64] = R_cost.copy()
        R_cost_invalid[0, 0] = -1.0  # λ_min < 0
        
        with pytest.raises(RayleighDissipationViolation) as exc_info:
            KBaseThermodynamicAgent(
                C_soc=C_soc,
                M_rec=M_rec,
                R_cost=R_cost_invalid,
                J_base=J_base,
            )
        
        assert "Segunda Ley" in str(exc_info.value) or "no es Semidefinida Positiva" in str(exc_info.value)
    
    def test_phase1_betti_0_computed(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que betti_0(R) = n − rank(R) se calcula correctamente.
        
        Topología: betti_0 cuenta componentes conexas del espacio disipativo
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        n_total = agent.context.dim_q + agent.context.dim_p
        expected_betti_0 = n_total - agent.context.rank_R
        
        assert agent.context.betti_0_R == expected_betti_0
        assert agent.context.betti_0_R >= 0


# =============================================================================
# FASE 2 — VALIDACIÓN DE DINÁMICA PORT-HAMILTONIANA Y DISIPACIÓN DE RAYLEIGH
# =============================================================================
class TestPhase2_HamiltonianDynamics:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 2 — DINÁMICA PORT-HAMILTONIANA, RAYLEIGH, WILLIAMSON Y PASIVIDAD
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase2_HamiltonianDynamics que gobierna
    la evolución temporal del foso termodinámico. Cada método verifica un axioma
    constitutivo de la dinámica Port-Hamiltoniana.
    
    Invariantes Verificados:
    ------------------------
    1. Energía potencial V(q) ≥ 0 y energía cinética K(p) ≥ 0
    2. Hamiltoniano total H = V + K (forma cuadrática estricta)
    3. Homogeneidad de Euler: q·∇_q H + p·∇_p H = 2H
    4. Disipación de Rayleigh: P_diss = ∇Hᵀ R ∇H ≥ 0
    5. Identidad estructural: ∇Hᵀ ẋ = −P_diss
    6. Certificado de pasividad: Ḣ + P_diss ≈ 0
    7. Voltaje de Flyback: ‖M̃·∂f/∂t‖_∞ ≤ V_breakdown
    8. Frecuencias de Williamson ω_i ∈ ℝ⁺
    9. Retícula Booleana de estabilidad
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.1 — ENERGÍAS Y GRADIENTES
    # -------------------------------------------------------------------------
    
    def test_phase2_potential_energy_nonnegative(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que energía potencial V(q) ≥ 0 para todo q.
        
        Axioma: V(q) = ½ qᵀ C̃⁻¹ q ≥ 0 (C̃ SPD)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.potential_energy >= 0.0, f"V(q) = {basal.potential_energy} < 0"
    
    def test_phase2_kinetic_energy_nonnegative(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que energía cinética K(p) ≥ 0 para todo p.
        
        Axioma: K(p) = ½ pᵀ M̃⁻¹ p ≥ 0 (M̃ SPD)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.kinetic_energy >= 0.0, f"K(p) = {basal.kinetic_energy} < 0"
    
    def test_phase2_hamiltonian_additive(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que H = V + K (aditividad de energías).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        expected_H = basal.potential_energy + basal.kinetic_energy
        tol = 1e-12 * max(abs(expected_H), 1.0)
        
        assert abs(basal.total_hamiltonian - expected_H) <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.2 — HOMOGENEIDAD DE EULER
    # -------------------------------------------------------------------------
    
    def test_phase2_euler_homogeneity_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica Teorema de Euler para H homogénea de grado 2.
        
        Teorema: q·∇_q H + p·∇_p H = 2H (exacto para formas cuadráticas)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        # El residuo debe estar dentro de tolerancia numérica
        tol = 1e-9 * max(abs(2.0 * basal.total_hamiltonian), 1.0)
        
        assert basal.euler_homogeneity_residual <= tol, \
            f"Residuo de Euler = {basal.euler_homogeneity_residual} > tol = {tol}"
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.3 — DISIPACIÓN DE RAYLEIGH Y SEGUNDA LEY
    # -------------------------------------------------------------------------
    
    def test_phase2_rayleigh_dissipation_nonnegative(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que P_diss ≥ 0 (Segunda Ley de Termodinámica).
        
        Axioma: P_diss = ∇Hᵀ R ∇H ≥ 0 (R PSD)
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.dissipated_power >= 0.0, f"P_diss = {basal.dissipated_power} < 0"
    
    def test_phase2_rayleigh_violation_raises(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R_cost no PSD dispara RayleighDissipationViolation.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        R_cost_invalid: NDArray[np.float64] = R_cost.copy()
        R_cost_invalid[0, 0] = -1.0  # λ_min < 0
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost_invalid,
            J_base=J_base,
        )
        
        with pytest.raises(RayleighDissipationViolation):
            agent.synthesize_basal_hamiltonian(
                q=q,
                p=p,
                df_dt=df_dt,
            )
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.4 — IDENTIDAD ESTRUCTURAL PORT-HAMILTONIANA
    # -------------------------------------------------------------------------
    
    def test_phase2_structural_consistency_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica identidad estructural ∇Hᵀ ẋ = −P_diss.
        
        Identidad: ∇Hᵀ J ∇H ≡ 0 (J antisimétrica) ⇒ ∇Hᵀ ẋ = −∇Hᵀ R ∇H = −P_diss
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        tol = 1e-10 * max(basal.dissipated_power, 1.0)
        
        assert basal.structural_consistency_residual <= tol, \
            f"Residuo estructural = {basal.structural_consistency_residual} > tol = {tol}"
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.5 — CERTIFICADO DE PASIVIDAD
    # -------------------------------------------------------------------------
    
    def test_phase2_passivity_certificate_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica certificado de pasividad Ḣ + P_diss ≈ 0.
        
        Certificado: |Ḣ_num + P_diss| ≤ √ε · scale
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        tol = 1e-10 * max(basal.dissipated_power, 1.0)
        
        assert basal.passivity_residual <= tol, \
            f"Residuo de pasividad = {basal.passivity_residual} > tol = {tol}"
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.6 — VOLTAJE DE FLYBACK
    # -------------------------------------------------------------------------
    
    def test_phase2_flyback_voltage_safe(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que voltaje de Flyback está dentro de margen seguro.
        
        Condición: ‖M̃·∂f/∂t‖_∞ ≤ margin · V_breakdown
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            breakdown_voltage=1e5,
            flyback_safety_margin=0.9,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.flyback_voltage_norm <= 0.9 * 1e5
    
    def test_phase2_flyback_voltage_exceeds_raises(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Flyback excesivo dispara InertialFlybackError.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        # df_dt muy grande para exceder V_breakdown
        df_dt_large: NDArray[np.float64] = np.array([1e10, 1e10], dtype=np.float64)
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            breakdown_voltage=1e5,
        )
        
        with pytest.raises(InertialFlybackError):
            agent.synthesize_basal_hamiltonian(
                q=q,
                p=p,
                df_dt=df_dt_large,
            )
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.7 — FORMAS NORMALES DE WILLIAMSON
    # -------------------------------------------------------------------------
    
    def test_phase2_williamson_normal_modes_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que frecuencias de Williamson son reales y no negativas.
        
        Teorema: ω_i ∈ ℝ⁺ para sistema conservativo linealizado
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
            compute_normal_modes=True,
        )
        
        assert basal.normal_mode_frequencies is not None
        assert np.all(basal.normal_mode_frequencies >= 0.0)
        assert np.all(np.isfinite(basal.normal_mode_frequencies))
    
    def test_phase2_zero_point_energy_computed(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que energía de punto cero E_0 = (ħ/2) Σ ω_i se calcula.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            hbar=1.0,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
            compute_normal_modes=True,
        )
        
        assert basal.zero_point_energy is not None
        assert basal.zero_point_energy >= 0.0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.8 — RETÍCULA BOOLEANA DE ESTABILIDAD
    # -------------------------------------------------------------------------
    
    def test_phase2_stability_flags_all_satisfied(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que todas las banderas de estabilidad se satisfacen para sistema válido.
        
        Álgebra de Boole: flags == StabilityFlags.ALL
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.is_thermodynamically_stable
        assert basal.stability_flags == StabilityFlags.ALL
    
    def test_phase2_stability_flags_describe(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que describe_stability_flags produce cadena legible.
        """
        from app.agents.alpha.kbase.kbase_thermodynamic_agent import describe_stability_flags
        
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        description = describe_stability_flags(basal.stability_flags)
        
        assert "SATISFECHOS=" in description
        assert "VIOLADOS=" in description
        assert "ESTABLE_TOTAL=" in description
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.9 — CONSTANTE DE TIEMPO ENTRÓPICA
    # -------------------------------------------------------------------------
    
    def test_phase2_dissipation_time_constant_finite(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que τ_diss = 2H / P_diss es finito cuando P_diss > 0.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        if basal.dissipated_power > 1e-12:
            assert np.isfinite(basal.dissipation_time_constant)
            assert basal.dissipation_time_constant > 0.0


# =============================================================================
# FASE 3 — VALIDACIÓN DE FIBRACIÓN CELULAR Y EXPORTACIÓN DE COFRONTERA
# =============================================================================
class TestPhase3_SheafProjection:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES: COCADENA APILADA Y HODGE LOCAL
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase3_SheafProjection que proyecta
    la variedad local como una fibra (Stalk) para el orquestador macroscópico.
    
    Invariantes Verificados:
    ------------------------
    1. Identidad de Hodge local: δ_BASEᵀ δ_BASE = ∇²H + R_cost
    2. Δ_BASE es SPD (Laplaciano de Hodge)
    3. Gap espectral de Hodge λ₂ − λ₁
    4. Número de condición κ(Δ_BASE)
    5. Dimensión armónica = dim ker(δ_metric)
    6. Entropía de von Neumann de Spec(Δ_BASE)
    7. Proxy de Cheeger λ₂/λ_max
    8. Proyecciones δ_metric·x y δ_diss·x
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.1 — CONSTRUCCIÓN DE COCADENA APILADA
    # -------------------------------------------------------------------------
    
    def test_phase3_delta_base_shape_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que δ_BASE tiene forma (2n, n).
        
        Construcción: δ_BASE = [δ_metric; δ_diss] ∈ ℝ^{2n×n}
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        n = agent.context.dim_q + agent.context.dim_p
        
        assert stalk.delta_base.shape == (2 * n, n)
        assert stalk.delta_metric.shape == (n, n)
        assert stalk.delta_dissipative.shape == (n, n)
    
    def test_phase3_delta_metric_block_diagonal(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que δ_metric es block-diag(C̃^{-1/2}, M̃^{-1/2}).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        dim_q = agent.context.dim_q
        dim_p = agent.context.dim_p
        
        # Verificar bloques diagonales
        C_block = stalk.delta_metric[:dim_q, :dim_q]
        M_block = stalk.delta_metric[dim_q:, dim_q:]
        
        # Verificar que bloques fuera de diagonal son cero
        off_diag_1 = stalk.delta_metric[:dim_q, dim_q:]
        off_diag_2 = stalk.delta_metric[dim_q:, :dim_q]
        
        tol = 1e-12
        
        assert float(la.norm(off_diag_1, "fro")) <= tol
        assert float(la.norm(off_diag_2, "fro")) <= tol
        
        # Verificar que bloques diagonales coinciden con C_inv_sqrt y M_inv_sqrt
        assert float(la.norm(C_block - agent.context.C_inv_sqrt, "fro")) <= tol
        assert float(la.norm(M_block - agent.context.M_inv_sqrt, "fro")) <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.2 — IDENTIDAD DE HODGE LOCAL
    # -------------------------------------------------------------------------
    
    def test_phase3_hodge_identity_satisfied(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica identidad de Hodge: δ_BASEᵀ δ_BASE = Δ_BASE.
        
        Identidad: ‖δᵀδ − Δ_BASE‖_F / ‖Δ_BASE‖_F ≤ 100·ε_mach
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        tol = 100.0 * _MACHINE_EPS
        
        assert stalk.hodge_identity_residual <= tol, \
            f"Residuo de Hodge = {stalk.hodge_identity_residual} > tol = {tol}"
    
    def test_phase3_hodge_laplacian_spd(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Δ_BASE es SPD (Laplaciano de Hodge).
        
        Invariante: Δ_BASE = ∇²H + R_cost debe ser SPD
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        lambda_min = float(la.eigvalsh(stalk.hodge_laplacian)[0])
        
        assert lambda_min > 0.0, f"Δ_BASE no es SPD: λ_min = {lambda_min}"
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.3 — ESPECTRO DE HODGE
    # -------------------------------------------------------------------------
    
    def test_phase3_hodge_spectral_gap_positive(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que gap espectral de Hodge es no negativo.
        
        Métrica: gap = λ₂ − λ₁ ≥ 0
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert stalk.hodge_spectral_gap >= 0.0
    
    def test_phase3_hodge_condition_number_finite(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que número de condición de Δ_BASE es finito.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert np.isfinite(stalk.hodge_condition_number)
        assert stalk.hodge_condition_number >= 1.0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.4 — DIMENSIONES TOPOLÓGICAS
    # -------------------------------------------------------------------------
    
    def test_phase3_harmonic_dimension_zero(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que dimensión armónica = dim ker(δ_metric) = 0.
        
        Topología: δ_metric invertible ⇒ ker(δ_metric) = {0}
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert stalk.harmonic_dimension == 0
    
    def test_phase3_lossless_subspace_dimension_matches_betti_0(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que dim_lossless = betti_0(R).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert stalk.lossless_subspace_dimension == agent.context.betti_0_R
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.5 — ENTROPÍA Y PROXY DE CHEEGER
    # -------------------------------------------------------------------------
    
    def test_phase3_spectral_entropy_hodge_nonnegative(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que entropía de von Neumann de Spec(Δ_BASE) es no negativa.
        
        Entropía: S = −Σ p_i ln p_i ≥ 0
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert stalk.spectral_entropy_hodge >= 0.0
    
    def test_phase3_cheeger_proxy_in_range(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que proxy de Cheeger ∈ [0, 1].
        
        Proxy: λ₂ / λ_max ∈ [0, 1]
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert 0.0 <= stalk.cheeger_proxy <= 1.0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.6 — PROYECCIONES DE ESTADO
    # -------------------------------------------------------------------------
    
    def test_phase3_projected_state_shapes_valid(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que proyecciones de estado tienen forma correcta.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        n = agent.context.dim_q + agent.context.dim_p
        
        assert stalk.projected_state_metric.shape == (n,)
        assert stalk.projected_state_dissipative.shape == (n,)
    
    def test_phase3_projected_state_consistency(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que proyecciones son consistentes con δ·x.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        expected_metric = stalk.delta_metric @ stalk.state_vector
        expected_diss = stalk.delta_dissipative @ stalk.state_vector
        
        tol = 1e-12 * max(
            float(la.norm(expected_metric, 2)),
            float(la.norm(expected_diss, 2)),
            1.0
        )
        
        assert float(la.norm(stalk.projected_state_metric - expected_metric, 2)) <= tol
        assert float(la.norm(stalk.projected_state_dissipative - expected_diss, 2)) <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.7 — PIPELINE COMPLETO
    # -------------------------------------------------------------------------
    
    def test_phase3_evaluate_full_pipeline(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica pipeline completo Fase 2 + Fase 3 en una llamada.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal, stalk = agent.evaluate_full_pipeline(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert isinstance(basal, BasalStateTensor)
        assert isinstance(stalk, SheafStalk)
        assert np.array_equal(basal.state_vector, stalk.state_vector)
    
    def test_phase3_lazy_initialization_phase3(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Phase3 se inicializa perezosamente en primera llamada.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        # Phase3 debe ser None antes de primera llamada
        assert agent.phase3 is None
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        # Phase3 aún debe ser None (solo se llama a Fase 2)
        assert agent.phase3 is None
        
        # Primera llamada a export_sheaf_stalk inicializa Phase3
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        assert agent.phase3 is not None
        assert isinstance(stalk, SheafStalk)


# =============================================================================
# PRUEBAS DE INTEGRACIÓN — TRANSICIONES ENTRE FASES
# =============================================================================
class TestPhaseTransitions_Integration:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    PRUEBAS DE INTEGRACIÓN — TRANSICIONES ENTRE FASES ANIDADAS
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida la composición funtorial estricta:
        Φ₃ ∘ Φ₂ ∘ Φ₁ : Matrices Constitutivas → SheafStalk
    
    Cada método verifica que la salida de una fase es la entrada válida
    de la siguiente fase, garantizando continuidad formal del endofuntor.
    """
    
    def test_phase1_to_phase2_context_continuity(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que TopologicalContext de Fase 1 es entrada válida de Fase 2.
        
        Contrato: context es el único argumento del constructor de Phase2
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        # Verificar que context tiene todos los atributos requeridos por Fase 2
        assert hasattr(agent.context, 'L_C')
        assert hasattr(agent.context, 'L_M')
        assert hasattr(agent.context, 'C_inv_sqrt')
        assert hasattr(agent.context, 'M_inv_sqrt')
        assert hasattr(agent.context, 'C_tilde')
        assert hasattr(agent.context, 'M_tilde')
        assert hasattr(agent.context, 'R_cost')
        assert hasattr(agent.context, 'J_base')
        assert hasattr(agent.context, 'dim_q')
        assert hasattr(agent.context, 'dim_p')
    
    def test_phase2_to_phase3_state_vector_continuity(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que state_vector de Fase 2 es entrada válida de Fase 3.
        
        Contrato: state_vector = [q; p] es argumento de export_stalk()
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        # state_vector debe tener forma (dim_q + dim_p,)
        n = agent.context.dim_q + agent.context.dim_p
        assert basal.state_vector.shape == (n,)
        
        # Debe ser concatenación de q y p
        expected = np.concatenate([q, p])
        assert np.array_equal(basal.state_vector, expected)
        
        # Debe ser entrada válida para export_sheaf_stalk
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        assert np.array_equal(stalk.state_vector, basal.state_vector)
    
    def test_phase1_to_phase3_context_reuse(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Phase3 reusa el mismo TopologicalContext de Fase 1.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        stalk = agent.export_sheaf_stalk(basal.state_vector)
        
        # Phase3 debe usar el mismo context que Phase1
        assert agent.phase3._ctx is agent.context
    
    def test_full_pipeline_immutability(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que DTOs son inmutables (frozen dataclasses).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal, stalk = agent.evaluate_full_pipeline(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        # Intentar modificar debe fallar (frozen=True)
        with pytest.raises(AttributeError):
            basal.total_hamiltonian = 999.0  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            stalk.hodge_laplacian[0, 0] = 999.0  # type: ignore[misc]


# =============================================================================
# PRUEBAS DE CASOS ESPECIALES Y BORDES
# =============================================================================
class TestEdgeCases_SpecialConditions:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    PRUEBAS DE CASOS ESPECIALES Y BORDES
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida comportamiento en condiciones límite:
    - Matrices casi singulares
    - Estados cero
    - Disipación cero (sistema conservativo puro)
    - Dimensiones grandes
    """
    
    def test_edge_case_zero_state_vector(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con estado cero q=0, p=0.
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        
        q: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        p: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        df_dt: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.potential_energy == 0.0
        assert basal.kinetic_energy == 0.0
        assert basal.total_hamiltonian == 0.0
    
    def test_edge_case_identity_metric_tensors(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con G_q = I, G_p = I (métrica euclidiana).
        """
        C_soc, M_rec, R_cost, J_base = fixture_valid_matrices_2d
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        G_q: NDArray[np.float64] = np.eye(2, dtype=np.float64)
        G_p: NDArray[np.float64] = np.eye(2, dtype=np.float64)
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
            G_q=G_q,
            G_p=G_p,
        )
        
        # Pullback con identidad debe preservar matrices originales
        assert float(la.norm(agent.context.C_tilde - C_soc, "fro")) < 1e-12
        assert float(la.norm(agent.context.M_tilde - M_rec, "fro")) < 1e-12
    
    def test_edge_case_diagonal_matrices(
        self,
        fixture_valid_state_vectors_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con matrices diagonales (caso desacoplado).
        """
        C_soc: NDArray[np.float64] = np.diag([2.0, 1.5])
        M_rec: NDArray[np.float64] = np.diag([1.8, 2.2])
        R_cost: NDArray[np.float64] = np.diag([0.5, 0.3, 0.4, 0.2])
        J_base: NDArray[np.float64] = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0],
             [-1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0]], dtype=np.float64
        )
        
        q, p, df_dt = fixture_valid_state_vectors_2d
        
        agent = KBaseThermodynamicAgent(
            C_soc=C_soc,
            M_rec=M_rec,
            R_cost=R_cost,
            J_base=J_base,
        )
        
        basal = agent.synthesize_basal_hamiltonian(
            q=q,
            p=p,
            df_dt=df_dt,
        )
        
        assert basal.is_thermodynamically_stable


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
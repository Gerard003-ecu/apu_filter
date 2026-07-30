# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — Semantic Validator Agent (Custodio de Cohomología Semántica)      ║
║  Ruta   : tests/unit/agents/boole/wisdom/test_semantic_validator_agent.py                ║
║  Versión: 8.0.0-Rigorous-Mahalanobis-Cohomology-Lattice-Heyting-TestSuite                ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPÓSITO CIBER-FÍSICO Y TOPOLOGÍA DE PRUEBAS (Rigor Categórico):                       ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Esta suite de pruebas consagra la Gobernanza de Cohomología Semántica del estrato       ║
║  WISDOM mediante un funtor de validación que verifica axiomáticamente la métrica         ║
║  de Mahalanobis, la nulidad del complejo de cadenas y el colapso del retículo de         ║
║  Heyting del modelo LLM.                                                                 ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):     ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Certificación Métrica de Mahalanobis                                           ║
║           Valida simetría, SPD y κ(G) del tensor métrico semántico.                      ║
║                                                                                          ║
║  Fase 2 → Auditoría de Cohomología Simplicial                                            ║
║           Verifica ∂₁∘∂₂ = 0 y computa dim H¹(K; ℝ).                                     ║
║                                                                                          ║
║  Fase 3 → Colapso en Retículo Completamente Ordenado                                     ║
║           Fuerza Veredicto = ⨆ v_i con veto cohomológico.                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

# =============================================================================
# Biblioteca estándar
# =============================================================================
import logging
from typing import Tuple, List, Optional
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
from app.agents.boole.wisdom.semantic_validator_agent import (
    SemanticValidatorAgent,
    MahalanobisMetricData,
    SimplicialCohomologyData,
    LatticeCollapseData,
    SemanticGovernanceState,
    StrictVerdict,
    # Excepciones
    SemanticValidatorAgentError,
    SemanticInputValidationError,
    MetricDegeneracyVeto,
    CohomologicalObstructionVeto,
    LatticeCollapseVeto,
)

# =============================================================================
# Logger y constantes globales de prueba
# =============================================================================
logger = logging.getLogger("MAC.Wisdom.Test.SemanticValidatorAgent")
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)

# =============================================================================
# FIXTURES GLOBALES — GENERADORES DE TENORES MÉTRICOS Y MATRICES DE FRONTERA
# =============================================================================


@pytest.fixture(scope="module")
def fixture_valid_metric_3d() -> NDArray[np.float64]:
    r"""
    Genera tensor métrico de Mahalanobis válido para dim=3.
    
    Retorna
    -------
    NDArray[np.float64], shape (3, 3), SPD simétrica
    """
    G: NDArray[np.float64] = np.array(
        [[1.1, 0.05, 0.02],
         [0.05, 1.0, 0.03],
         [0.02, 0.03, 0.9]], dtype=np.float64
    )
    
    return G


@pytest.fixture(scope="module")
def fixture_valid_metric_2d() -> NDArray[np.float64]:
    r"""
    Genera tensor métrico de Mahalanobis válido para dim=2 (caso mínimo).
    
    Retorna
    -------
    NDArray[np.float64], shape (2, 2), SPD simétrica
    """
    G: NDArray[np.float64] = np.array(
        [[1.1, 0.05],
         [0.05, 1.0]], dtype=np.float64
    )
    
    return G


@pytest.fixture(scope="module")
def fixture_valid_boundary_matrices_3d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices de frontera válidas para complejo de cadenas 3D.
    
    Convención:
        ∂₁ : C₁ → C₀  =>  d1.shape = (dim_C0, dim_C1)
        ∂₂ : C₂ → C₁  =>  d2.shape = (dim_C1, dim_C2)
    
    Retorna
    -------
    Tuple[boundary_matrix_d1, boundary_matrix_d2]
    """
    # dim_C0 = 3, dim_C1 = 4, dim_C2 = 2
    d1: NDArray[np.float64] = np.array(
        [[1.0, -1.0, 0.0, 0.0],
         [0.0, 1.0, -1.0, 0.0],
         [0.0, 0.0, 1.0, -1.0]], dtype=np.float64
    )
    
    d2: NDArray[np.float64] = np.array(
        [[1.0, 0.0],
         [1.0, 1.0],
         [0.0, 1.0],
         [0.0, 0.0]], dtype=np.float64
    )
    
    # Verificar que ∂₁∘∂₂ = 0
    composition = d1 @ d2
    assert float(la.norm(composition, "fro")) < 1e-10, \
        "Fixture no satisface ∂₁∘∂₂ = 0"
    
    return d1, d2


@pytest.fixture(scope="module")
def fixture_valid_boundary_matrices_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices de frontera válidas para complejo de cadenas 2D (caso mínimo).
    
    Retorna
    -------
    Tuple[boundary_matrix_d1, boundary_matrix_d2]
    """
    # dim_C0 = 2, dim_C1 = 2, dim_C2 = 1
    d1: NDArray[np.float64] = np.array(
        [[1.0, -1.0],
         [0.0, 1.0]], dtype=np.float64
    )
    
    d2: NDArray[np.float64] = np.array(
        [[1.0],
         [1.0]], dtype=np.float64
    )
    
    # Verificar que ∂₁∘∂₂ = 0
    composition = d1 @ d2
    assert float(la.norm(composition, "fro")) < 1e-10, \
        "Fixture no satisface ∂₁∘∂₂ = 0"
    
    return d1, d2


@pytest.fixture(scope="module")
def fixture_valid_verdicts() -> List[StrictVerdict]:
    r"""
    Genera secuencia de veredictos válidos.
    
    Retorna
    -------
    List[StrictVerdict]
    """
    return [
        StrictVerdict.VIABLE,
        StrictVerdict.CONDITIONAL,
        StrictVerdict.WARNING,
    ]


@pytest.fixture(scope="module")
def fixture_verdicts_with_reject() -> List[StrictVerdict]:
    r"""
    Genera secuencia de veredictos que incluye REJECT.
    
    Retorna
    -------
    List[StrictVerdict]
    """
    return [
        StrictVerdict.VIABLE,
        StrictVerdict.REJECT,
        StrictVerdict.WARNING,
    ]


# =============================================================================
# FASE 1 — CERTIFICACIÓN MÉTRICA DE MAHALANOBIS
# =============================================================================
class TestPhase1_MetricTensorCertification:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 1 — CERTIFICACIÓN ESPECTRAL DEL TENSOR DE MAHALANOBIS
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase1_MetricTensorAuditor que consagra la
    geometría Riemanniana del espacio de validación semántica. Cada método verifica un
    axioma constitutivo del estrato WISDOM.
    
    Invariantes Verificados:
    ------------------------
    1. Coherencia dimensional de G_metric
    2. Simetría de G_metric (G = Gᵀ)
    3. Definida positiva (λ_min > 0)
    4. Finitud de entradas (no NaN, no Inf)
    5. Número de condición κ(G) ≤ κ_max
    6. Regularización espectral de autovalores pequeños
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.1 — VALIDACIÓN DIMENSIONAL Y ESTRUCTURAL
    # -------------------------------------------------------------------------
    
    def test_phase1_dimensions_valid_3d(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que métrica 3D válida pasa la validación dimensional.
        
        Axioma: G_metric ∈ ℝ^{n×n}, cuadrada
        """
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit.dimension == 3
        assert metric_audit.is_positive_definite is True
        assert metric_audit.condition_number < 1e8
    
    def test_phase1_dimensions_valid_2d(
        self,
        fixture_valid_metric_2d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica caso mínimo dim=2 (frontera inferior).
        """
        G = fixture_valid_metric_2d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit.dimension == 2
        assert metric_audit.is_positive_definite is True
    
    def test_phase1_dimension_mismatch_non_square(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric no cuadrada dispara SemanticInputValidationError.
        """
        G = fixture_valid_metric_3d
        G_invalid: NDArray[np.float64] = G[:, :2]  # 3×2
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_invalid,
            )
        
        assert "cuadrada" in str(exc_info.value) or "2D" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_empty(
        self,
    ) -> None:
        r"""
        Verifica que G_metric vacía dispara SemanticInputValidationError.
        """
        G_empty: NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 0)
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_empty,
            )
        
        assert "vacía" in str(exc_info.value) or "empty" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.2 — VALIDACIÓN DE SIMETRÍA MÉTRICA
    # -------------------------------------------------------------------------
    
    def test_phase1_symmetry_valid(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric simétrica pasa validación.
        
        Axioma: G = Gᵀ dentro de tolerancia ε_mach · ‖G‖_F
        """
        G = fixture_valid_metric_3d
        
        # Verificar simetría explícita
        sym_residual = float(la.norm(G - G.T, "fro"))
        norm_G = float(la.norm(G, "fro"))
        tol = _MACHINE_EPS * max(norm_G, 1.0)
        
        assert sym_residual <= tol, f"Fixture G no es simétrica: {sym_residual}"
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit is not None
    
    def test_phase1_symmetry_invalid(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric asimétrica dispara MetricDegeneracyVeto.
        """
        G = fixture_valid_metric_3d
        G_invalid: NDArray[np.float64] = G.copy()
        G_invalid[0, 1] += 0.5  # Romper simetría
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(MetricDegeneracyVeto) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_invalid,
            )
        
        assert "simétrico" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.3 — VALIDACIÓN DE DEFINIDA POSITIVA (SPD)
    # -------------------------------------------------------------------------
    
    def test_phase1_spd_valid(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric SPD pasa validación.
        
        Axioma: λ_min(G) > 0 (todos autovalores positivos)
        """
        G = fixture_valid_metric_3d
        
        # Verificar SPD explícito
        eigvals = la.eigvalsh(G)
        lambda_min = float(np.min(eigvals))
        
        assert lambda_min > 0.0, f"Fixture G no es SPD: λ_min={lambda_min}"
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit.min_eigenvalue > 0.0
        assert metric_audit.max_eigenvalue > 0.0
    
    def test_phase1_spd_invalid_negative_eigenvalue(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric con autovalor negativo dispara MetricDegeneracyVeto.
        """
        G = fixture_valid_metric_3d
        G_invalid: NDArray[np.float64] = G.copy()
        G_invalid[0, 0] = -1.0  # Forzar λ_min < 0
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(MetricDegeneracyVeto) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_invalid,
            )
        
        assert "definido positivo" in str(exc_info.value) or "SPD" in str(exc_info.value)
    
    def test_phase1_spd_invalid_near_singular(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que G_metric casi singular dispara MetricDegeneracyVeto.
        """
        G_singular: NDArray[np.float64] = np.array(
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0]], dtype=np.float64
        )  # rank 1, λ_min = 0
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(MetricDegeneracyVeto) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_singular,
            )
        
        assert "singular" in str(exc_info.value) or "degenerada" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.4 — NÚMEROS DE CONDICIÓN
    # -------------------------------------------------------------------------
    
    def test_phase1_condition_number_within_limit(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que κ(G) < κ_max.
        """
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit.condition_number < 1e8
        assert np.isfinite(metric_audit.condition_number)
    
    def test_phase1_condition_number_exceeds_limit(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que κ(G) > κ_max dispara MetricDegeneracyVeto.
        """
        G_ill: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1e-9, 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )  # κ ≈ 1e9 > 1e8
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(MetricDegeneracyVeto) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_ill,
            )
        
        assert "condición" in str(exc_info.value) or "κ" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.5 — VALIDACIÓN DE FINITUD NUMÉRICA
    # -------------------------------------------------------------------------
    
    def test_phase1_finite_values_valid(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que métrica con valores finitos pasa validación.
        """
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert np.all(np.isfinite(metric_audit.min_eigenvalue))
        assert np.all(np.isfinite(metric_audit.max_eigenvalue))
        assert np.all(np.isfinite(metric_audit.condition_number))
    
    def test_phase1_nan_values_raise(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que NaN en métrica dispara SemanticInputValidationError.
        """
        G = fixture_valid_metric_3d
        G_nan: NDArray[np.float64] = G.copy()
        G_nan[0, 0] = np.nan
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_nan,
            )
        
        assert "NaN" in str(exc_info.value) or "infinitos" in str(exc_info.value)
    
    def test_phase1_inf_values_raise(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que Inf en métrica dispara SemanticInputValidationError.
        """
        G = fixture_valid_metric_3d
        G_inf: NDArray[np.float64] = G.copy()
        G_inf[0, 0] = np.inf
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_inf,
            )
        
        assert "infinitos" in str(exc_info.value) or "NaN" in str(exc_info.value)
    
    def test_phase1_complex_values_raise(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que valores complejos disparan SemanticInputValidationError.
        """
        G = fixture_valid_metric_3d
        G_complex: NDArray[np.complex128] = G.astype(np.complex128)
        G_complex[0, 0] += 0.1j
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._audit_mahalanobis_metric_tensor(
                G_metric=G_complex,
            )
        
        assert "compleja" in str(exc_info.value) or "real" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.6 — DIAGNÓSTICOS ESPECTRALES
    # -------------------------------------------------------------------------
    
    def test_phase1_eigenvalues_computed(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que autovalores mínimo y máximo se calculan correctamente.
        """
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        # Verificar contra cálculo directo
        eigvals = la.eigvalsh(G)
        expected_min = float(np.min(eigvals))
        expected_max = float(np.max(eigvals))
        
        tol = 1e-10 * max(abs(expected_min), abs(expected_max), 1.0)
        
        assert abs(metric_audit.min_eigenvalue - expected_min) <= tol
        assert abs(metric_audit.max_eigenvalue - expected_max) <= tol
    
    def test_phase1_symmetry_residual_computed(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que residuo de simetría se calcula correctamente.
        """
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        # Para matriz simétrica, residuo debe ser muy pequeño
        assert metric_audit.symmetry_residual < 1e-10


# =============================================================================
# FASE 2 — AUDITORÍA DE COHOMOLOGÍA SIMPLICIAL
# =============================================================================
class TestPhase2_SimplicialCohomologyAudit:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 2 — AUDITORÍA DE COHOMOLOGÍA SIMPLICIAL Y COMPLEJO DE CADENAS
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase2_SimplicialCohomologyAuditor que
    gobierna la nulidad del complejo de cadenas y la dimensión de H¹.
    
    Invariantes Verificados:
    ------------------------
    1. Condición de frontera: ∂₁∘∂₂ = 0
    2. dim H¹ = dim ker(∂₁) - dim im(∂₂)
    3. Rango numérico vía SVD
    4. Consistencia dimensional con certificado de Fase 1
    5. Modo estricto vs no estricto para veto cohomológico
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.1 — VALIDACIÓN DEL COMPLEJO DE CADENAS
    # -------------------------------------------------------------------------
    
    def test_phase2_chain_complex_valid_3d(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que complejo de cadenas válido pasa validación.
        
        Axioma: ∂₁∘∂₂ = 0 dentro de tolerancia
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        # Fase 1 primero
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        # Fase 2
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        assert cohomology_audit.chain_complex_residual < 1e-10
        assert cohomology_audit.is_logically_coherent is True
    
    def test_phase2_chain_complex_valid_2d(
        self,
        fixture_valid_boundary_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_2d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica caso mínimo dim=2.
        """
        d1, d2 = fixture_valid_boundary_matrices_2d
        G = fixture_valid_metric_2d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        assert cohomology_audit.chain_complex_residual < 1e-10
    
    def test_phase2_chain_complex_violation_raises(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que ∂₁∘∂₂ ≠ 0 dispara CohomologicalObstructionVeto.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        # Modificar d2 para violar ∂₁∘∂₂ = 0
        d2_invalid: NDArray[np.float64] = d2.copy()
        d2_invalid[0, 0] += 1.0  # Romper condición de frontera
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        with pytest.raises(CohomologicalObstructionVeto) as exc_info:
            agent._certify_simplicial_cohomology(
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2_invalid,
                metric_audit=metric_audit,
                strict_cohomological_veto=False,
            )
        
        assert "∂₁∘∂₂" in str(exc_info.value) or "complejo" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.2 — DIMENSIÓN DE COHOMOLOGÍA H¹
    # -------------------------------------------------------------------------
    
    def test_phase2_h1_dimension_zero_valid(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dim H¹ = 0 indica coherencia lógica.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        assert cohomology_audit.h1_dimension == 0
        assert cohomology_audit.is_logically_coherent is True
    
    def test_phase2_h1_dimension_positive_strict_veto(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dim H¹ > 0 en modo estricto dispara CohomologicalObstructionVeto.
        """
        # Construir matrices con dim H¹ > 0
        # dim_C0 = 2, dim_C1 = 3, dim_C2 = 1
        # ker(∂₁) = dim_C1 - rank(∂₁) = 3 - 1 = 2
        # im(∂₂) = rank(∂₂) = 0 (si ∂₂ = 0)
        # dim H¹ = 2 - 0 = 2 > 0
        
        d1: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]], dtype=np.float64
        )
        
        d2: NDArray[np.float64] = np.array(
            [[0.0],
             [0.0],
             [0.0]], dtype=np.float64
        )
        
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        with pytest.raises(CohomologicalObstructionVeto) as exc_info:
            agent._certify_simplicial_cohomology(
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2,
                metric_audit=metric_audit,
                strict_cohomological_veto=True,  # Modo estricto
            )
        
        assert "H¹" in str(exc_info.value) or "obstrucción" in str(exc_info.value)
    
    def test_phase2_h1_dimension_positive_non_strict(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dim H¹ > 0 en modo no estricto retorna is_logically_coherent=False.
        """
        d1: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]], dtype=np.float64
        )
        
        d2: NDArray[np.float64] = np.array(
            [[0.0],
             [0.0],
             [0.0]], dtype=np.float64
        )
        
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,  # Modo no estricto
        )
        
        assert cohomology_audit.h1_dimension > 0
        assert cohomology_audit.is_logically_coherent is False
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.3 — RANGO NUMÉRICO VÍA SVD
    # -------------------------------------------------------------------------
    
    def test_phase2_rank_computed_correctly(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que rangos de ∂₁ y ∂₂ se calculan correctamente.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        # Verificar que rangos son consistentes con dimensiones
        assert cohomology_audit.rank_d1 >= 0
        assert cohomology_audit.rank_d2 >= 0
        assert cohomology_audit.rank_d1 <= min(d1.shape)
        assert cohomology_audit.rank_d2 <= min(d2.shape)
    
    def test_phase2_kernel_and_image_dimensions(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dim ker(∂₁) y dim im(∂₂) se calculan correctamente.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        # dim ker(∂₁) = dim C¹ - rank(∂₁)
        expected_kernel = cohomology_audit.dim_C1 - cohomology_audit.rank_d1
        
        assert cohomology_audit.kernel_d1_dim == expected_kernel
        assert cohomology_audit.image_d2_dim == cohomology_audit.rank_d2
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.4 — CONSISTENCIA DIMENSIONAL CON FASE 1
    # -------------------------------------------------------------------------
    
    def test_phase2_requires_phase1_positive_definite(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que Fase 2 requiere métrica SPD de Fase 1.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        # Forzar is_positive_definite = False (simulado)
        from dataclasses import replace
        metric_audit_failed = replace(metric_audit, is_positive_definite=False)
        
        with pytest.raises(MetricDegeneracyVeto) as exc_info:
            agent._certify_simplicial_cohomology(
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2,
                metric_audit=metric_audit_failed,
                strict_cohomological_veto=False,
            )
        
        assert "Fase 1" in str(exc_info.value) or "métrica" in str(exc_info.value)
    
    def test_phase2_works_without_phase1_audit(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que Fase 2 puede operar sin certificado de Fase 1 (metric_audit=None).
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        
        agent = SemanticValidatorAgent()
        
        # Sin metric_audit
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=None,
            strict_cohomological_veto=False,
        )
        
        assert cohomology_audit.is_logically_coherent is True
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.5 — VALIDACIÓN DE MATRICES DE FRONTERA
    # -------------------------------------------------------------------------
    
    def test_phase2_boundary_matrices_dimension_mismatch(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dimensiones inconsistentes entre d1 y d2 disparan error.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        # d2 con filas incorrectas (debe coincidir con columnas de d1)
        d2_invalid: NDArray[np.float64] = np.ones((3, 2), dtype=np.float64)  # 3 filas ≠ 4
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        with pytest.raises(SemanticInputValidationError) as exc_info:
            agent._certify_simplicial_cohomology(
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2_invalid,
                metric_audit=metric_audit,
                strict_cohomological_veto=False,
            )
        
        assert "dimensión" in str(exc_info.value) or "complejo" in str(exc_info.value)
    
    def test_phase2_boundary_matrices_empty_raises(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que matrices de frontera vacías disparan error.
        """
        d1_empty: NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 0)
        d2_empty: NDArray[np.float64] = np.array([], dtype=np.float64).reshape(0, 0)
        
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        with pytest.raises(SemanticInputValidationError):
            agent._certify_simplicial_cohomology(
                boundary_matrix_d1=d1_empty,
                boundary_matrix_d2=d2_empty,
                metric_audit=metric_audit,
                strict_cohomological_veto=False,
            )


# =============================================================================
# FASE 3 — COLAPSO EN RETÍCULO COMPLETAMENTE ORDENADO
# =============================================================================
class TestPhase3_LatticeSupremumCollapse:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 3 — COLAPSO DEL RETÍCULO DE HEYTING Y VEREDICTO SUPREMO
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase3_LatticeSupremumProjector que fuerza
    la convergencia en el peor caso topológico del espacio de decisión.
    
    Invariantes Verificados:
    ------------------------
    1. Supremo en retículo completamente ordenado: ⊥ ≤ CONDITIONAL ≤ WARNING ≤ ⊤
    2. Obstrucción cohomológica colapsa a REJECT (⊤)
    3. Elemento máximo absorbente: x ⊔ ⊤ = ⊤
    4. Consistencia con certificado de Fase 2
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.1 — CÁLCULO DEL SUPREMO EN RETÍCULO
    # -------------------------------------------------------------------------
    
    def test_phase3_supremum_computed_correctly(
        self,
        fixture_valid_verdicts: List[StrictVerdict],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que supremo del retículo se calcula correctamente.
        
        Retículo: VIABLE(0) ≤ CONDITIONAL(1) ≤ WARNING(2) ≤ REJECT(3)
        """
        verdicts = fixture_valid_verdicts
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            cohomology_audit=cohomology_audit,
        )
        
        # Supremum debe ser WARNING (el máximo en la secuencia)
        assert lattice_audit.supremum_verdict == StrictVerdict.WARNING
        assert lattice_audit.verdict_count == len(verdicts)
    
    def test_phase3_supremum_with_reject_always_reject(
        self,
        fixture_verdicts_with_reject: List[StrictVerdict],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que si hay REJECT en la secuencia, supremo = REJECT.
        
        Propiedad: ⊤ es elemento máximo absorbente
        """
        verdicts = fixture_verdicts_with_reject
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            cohomology_audit=cohomology_audit,
        )
        
        assert lattice_audit.supremum_verdict == StrictVerdict.REJECT
    
    def test_phase3_supremum_empty_verdicts_raises(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que veredictos vacíos disparan LatticeCollapseVeto.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        with pytest.raises(LatticeCollapseVeto) as exc_info:
            agent._enforce_supremum_lattice_collapse(
                verdicts=[],  # Vacío
                cohomology_audit=cohomology_audit,
            )
        
        assert "vacío" in str(exc_info.value) or "∅" in str(exc_info.value)
    
    def test_phase3_supremum_invalid_verdict_raises(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que veredicto inválido dispara LatticeCollapseVeto.
        """
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        # Veredicto inválido (no pertenece a StrictVerdict)
        invalid_verdicts = [StrictVerdict.VIABLE, "INVALIDO"]  # type: ignore[list-item]
        
        with pytest.raises(LatticeCollapseVeto) as exc_info:
            agent._enforce_supremum_lattice_collapse(
                verdicts=invalid_verdicts,
                cohomology_audit=cohomology_audit,
            )
        
        assert "inválido" in str(exc_info.value) or "StrictVerdict" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.2 — VETO COHOMOLÓGICO (COLAPSO A REJECT)
    # -------------------------------------------------------------------------
    
    def test_phase3_cohomological_obstruction_forces_reject(
        self,
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica que obstrucción cohomológica fuerza supremo = REJECT.
        
        Propiedad: x ⊔ ⊤ = ⊤ (elemento máximo absorbente)
        """
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        # Forzar obstrucción cohomológica directamente
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            has_cohomological_obstruction=True,  # Forzar veto
            cohomology_audit=None,
        )
        
        assert lattice_audit.supremum_verdict == StrictVerdict.REJECT
        assert lattice_audit.has_cohomological_obstruction is True
        assert lattice_audit.is_worst_case_enforced is True
    
    def test_phase3_cohomological_obstruction_from_audit(
        self,
        fixture_valid_verdicts: List[StrictVerdict],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que dim H¹ > 0 en cohomology_audit fuerza REJECT.
        """
        verdicts = fixture_valid_verdicts
        
        # Construir auditoría con obstrucción
        cohomology_audit_invalid = SimplicialCohomologyData(
            dim_C0=2,
            dim_C1=3,
            dim_C2=1,
            rank_d1=1,
            rank_d2=0,
            kernel_d1_dim=2,
            image_d2_dim=0,
            h1_dimension=2,  # > 0
            chain_complex_residual=0.0,
            cohomology_tolerance=1e-10,
            is_logically_coherent=False,
        )
        
        agent = SemanticValidatorAgent()
        
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            cohomology_audit=cohomology_audit_invalid,
        )
        
        assert lattice_audit.supremum_verdict == StrictVerdict.REJECT
        assert lattice_audit.has_cohomological_obstruction is True
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.3 — CONSISTENCIA CON FASE 2
    # -------------------------------------------------------------------------
    
    def test_phase3_requires_phase2_coherence(
        self,
        fixture_valid_verdicts: List[StrictVerdict],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_metric_3d: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que Fase 3 respeta certificado de coherencia de Fase 2.
        """
        verdicts = fixture_valid_verdicts
        d1, d2 = fixture_valid_boundary_matrices_3d
        G = fixture_valid_metric_3d
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        cohomology_audit = agent._certify_simplicial_cohomology(
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            metric_audit=metric_audit,
            strict_cohomological_veto=False,
        )
        
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            cohomology_audit=cohomology_audit,
        )
        
        # Si no hay obstrucción, supremo debe ser el máximo de los veredictos
        if cohomology_audit.is_logically_coherent:
            assert lattice_audit.supremum_verdict == max(verdicts, key=lambda v: v.value)
    
    def test_phase3_works_without_phase2_audit(
        self,
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica que Fase 3 puede operar sin certificado de Fase 2 (cohomology_audit=None).
        """
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        # Sin cohomology_audit
        lattice_audit = agent._enforce_supremum_lattice_collapse(
            verdicts=verdicts,
            cohomology_audit=None,
        )
        
        assert lattice_audit.supremum_verdict == StrictVerdict.WARNING
        assert lattice_audit.has_cohomological_obstruction is False


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
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica pipeline completo con entradas válidas.
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        assert state.is_epistemologically_valid is True
        assert state.metric_audit.is_positive_definite is True
        assert state.cohomology_audit.is_logically_coherent is True
        assert state.lattice_audit.is_worst_case_enforced is True
    
    def test_full_pipeline_metric_degeneracy_fails(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica que degeneración métrica falla el pipeline completo.
        """
        G_invalid: NDArray[np.float64] = np.array(
            [[-1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(MetricDegeneracyVeto):
            agent.execute_semantic_cohomology_governance(
                G_metric=G_invalid,
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2,
                proposed_verdicts=verdicts,
            )
    
    def test_full_pipeline_cohomological_obstruction_fails(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica que obstrucción cohomológica falla el pipeline completo.
        """
        G = fixture_valid_metric_3d
        verdicts = fixture_valid_verdicts
        
        # Matrices con dim H¹ > 0
        d1: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]], dtype=np.float64
        )
        
        d2: NDArray[np.float64] = np.array(
            [[0.0],
             [0.0],
             [0.0]], dtype=np.float64
        )
        
        agent = SemanticValidatorAgent()
        
        with pytest.raises(CohomologicalObstructionVeto):
            agent.execute_semantic_cohomology_governance(
                G_metric=G,
                boundary_matrix_d1=d1,
                boundary_matrix_d2=d2,
                proposed_verdicts=verdicts,
            )
    
    def test_full_pipeline_dto_immutability(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica que DTOs son inmutables (frozen dataclasses).
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        # Intentar modificar debe fallar (frozen=True)
        with pytest.raises(AttributeError):
            state.is_epistemologically_valid = False  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.metric_audit.dimension = 999  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.cohomology_audit.h1_dimension = 999  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.lattice_audit.supremum_verdict = StrictVerdict.REJECT  # type: ignore[misc]
    
    def test_full_pipeline_audit_data_consistency(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica consistencia entre certificados de las tres fases.
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        # Dimensiones consistentes
        assert state.metric_audit.dimension > 0
        
        # Validez epistemológica implica todas las fases válidas
        if state.is_epistemologically_valid:
            assert state.metric_audit.is_positive_definite
            assert state.cohomology_audit.is_logically_coherent
            assert state.lattice_audit.is_worst_case_enforced
        
        # Si hay obstrucción cohomológica, supremo debe ser REJECT
        if not state.cohomology_audit.is_logically_coherent:
            assert state.lattice_audit.supremum_verdict == StrictVerdict.REJECT


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
    
    def test_edge_case_identity_metric(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica comportamiento con G = I (métrica euclidiana).
        """
        dim = 3
        G: NDArray[np.float64] = np.eye(dim, dtype=np.float64)
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        assert state.metric_audit.condition_number == 1.0
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_minimum_dimension(
        self,
        fixture_valid_metric_2d: NDArray[np.float64],
        fixture_valid_boundary_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con dimensión mínima (n=2).
        """
        G = fixture_valid_metric_2d
        d1, d2 = fixture_valid_boundary_matrices_2d
        
        verdicts: List[StrictVerdict] = [StrictVerdict.VIABLE]
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        assert state.metric_audit.dimension == 2
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_tolerance_boundaries(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica comportamiento en límites de tolerancia numérica.
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        # Matrices muy cercanas a violar condición de frontera
        d2_boundary: NDArray[np.float64] = d2 + 1e-11 * np.ones_like(d2)
        
        agent = SemanticValidatorAgent()
        
        # Debe pasar si está dentro de tolerancia
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2_boundary,
            proposed_verdicts=verdicts,
        )
        
        assert state.cohomology_audit.chain_complex_residual < 1e-10
    
    def test_edge_case_single_verdict(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con único veredicto.
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        
        verdicts: List[StrictVerdict] = [StrictVerdict.CONDITIONAL]
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        assert state.lattice_audit.supremum_verdict == StrictVerdict.CONDITIONAL
        assert state.lattice_audit.verdict_count == 1
    
    def test_edge_case_all_verdicts_same(
        self,
        fixture_valid_metric_3d: NDArray[np.float64],
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con todos los veredictos iguales.
        """
        G = fixture_valid_metric_3d
        d1, d2 = fixture_valid_boundary_matrices_3d
        
        verdicts: List[StrictVerdict] = [
            StrictVerdict.WARNING,
            StrictVerdict.WARNING,
            StrictVerdict.WARNING,
        ]
        
        agent = SemanticValidatorAgent()
        
        state = agent.execute_semantic_cohomology_governance(
            G_metric=G,
            boundary_matrix_d1=d1,
            boundary_matrix_d2=d2,
            proposed_verdicts=verdicts,
        )
        
        assert state.lattice_audit.supremum_verdict == StrictVerdict.WARNING
        assert state.lattice_audit.verdict_count == 3
    
    def test_edge_case_very_well_conditioned_metric(
        self,
        fixture_valid_boundary_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_verdicts: List[StrictVerdict],
    ) -> None:
        r"""
        Verifica comportamiento con métrica muy bien condicionada (κ ≈ 1).
        """
        G: NDArray[np.float64] = np.eye(3, dtype=np.float64) * 1.0001
        d1, d2 = fixture_valid_boundary_matrices_3d
        verdicts = fixture_valid_verdicts
        
        agent = SemanticValidatorAgent()
        
        metric_audit = agent._audit_mahalanobis_metric_tensor(
            G_metric=G,
        )
        
        assert metric_audit.condition_number < 2.0
        assert metric_audit.is_positive_definite is True


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
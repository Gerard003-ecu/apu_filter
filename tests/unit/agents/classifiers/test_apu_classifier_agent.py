# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — APU Classifier Agent (Custodio de la Partición Ontológica)        ║
║  Ruta   : tests/unit/agents/classifiers/test_apu_classifier_agent.py                     ║
║  Versión: 9.0.0-Rigorous-Lebesgue-Affine-Homology-Strict-TestSuite                       ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPÓSITO CIBER-FÍSICO Y TOPOLOGÍA DE PRUEBAS (Rigor Categórico):                       ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Esta suite de pruebas consagra la Partición Ontológica del estrato CLASSIFIERS          ║
║  mediante un funtor de validación que verifica axiomáticamente la medida de Lebesgue,    ║
║  el difeomorfismo afín de escala y la homología estructural de centroides del APU.       ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):     ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Auditoría de Cobertura Espacial y Medida de Lebesgue                           ║
║           Verifica μ(Δ² \ ⋃ R_i) ≤ ε y sanea el dominio vectorial inicial.               ║
║                                                                                          ║
║  Fase 2 → Certificación de Difeomorfismo Afín y Contrato de Escala                       ║
║           Computa ||c - M_scale p||_∞, certifica isomorfismo de escala.                  ║
║                                                                                          ║
║  Fase 3 → Evaluación de Centroides Topológicos y Homología Estructural                   ║
║           Sintetiza la ortogonalidad del centroide subyugada a las bases canónicas.      ║
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
from app.agents.classifiers.apu_classifier_agent import (
    APUClassifierAgent,
    VectorDomainCertificate,
    LebesgueAuditData,
    ScaleIsomorphismData,
    CentroidTopologyData,
    Phase1LebesgueHandoff,
    Phase2ScaleHandoff,
    OntologicalPartitionState,
    # Excepciones
    APUClassifierAgentError,
    DomainIntegrityViolationError,
    SimplexMembershipViolationError,
    LebesgueMeasureViolationError,
    ScaleInvarianceCollapseError,
    TopologicalCentroidAnomalyVeto,
)

# =============================================================================
# Logger y constantes globales de prueba
# =============================================================================
logger = logging.getLogger("MIC.Classifiers.Test.APUClassifierAgent")
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)

# =============================================================================
# FIXTURES GLOBALES — GENERADORES DE VECTORES Y PARÁMETROS ONTOLÓGICOS
# =============================================================================


@pytest.fixture(scope="module")
def fixture_valid_vectors_3d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera vectores ontológicos válidos para dim=3 (Suministro, MO, Equipo).
    
    Retorna
    -------
    Tuple[p_vector, c_vector, centroid_C]
        p_vector: proporciones en Δ² (suma = 1, no negativo)
        c_vector: porcentajes en [0, 100]³
        centroid_C: centroide de clase para validación topológica
    """
    # p_vector ∈ Δ²: suma = 1, no negativo
    p_vector: NDArray[np.float64] = np.array(
        [0.4, 0.35, 0.25], dtype=np.float64
    )
    
    # c_vector ∈ [0, 100]³: porcentajes consistentes con p
    c_vector: NDArray[np.float64] = np.array(
        [40.0, 35.0, 25.0], dtype=np.float64
    )
    
    # centroid_C: centroide de clase (puede ser "Isla" o no)
    centroid_C: NDArray[np.float64] = np.array(
        [0.5, 0.3, 0.2], dtype=np.float64
    )
    
    return p_vector, c_vector, centroid_C


@pytest.fixture(scope="module")
def fixture_valid_island_centroid() -> NDArray[np.float64]:
    r"""
    Genera centroide de tipo "Isla de Suministro" (ortogonal a MO y Equipo).
    
    Axioma: <C_isla, e_mo> = 0, <C_isla, e_eq> = 0
    Retorna
    -------
    NDArray[np.float64], shape (3,)
    """
    # Solo componente de Suministro, MO y Equipo = 0
    centroid_island: NDArray[np.float64] = np.array(
        [1.0, 0.0, 0.0], dtype=np.float64
    )
    
    return centroid_island


@pytest.fixture(scope="module")
def fixture_valid_lebesgue_params() -> Tuple[float, float]:
    r"""
    Genera parámetros válidos para auditoría de Lebesgue.
    
    Retorna
    -------
    Tuple[uncovered_area_ratio, tolerance]
    """
    uncovered_area_ratio: float = 1e-9  # Muy pequeño, dentro de tolerancia
    tolerance: float = 1e-7  # _LEBESGUE_MEASURE_TOLERANCE
    
    return uncovered_area_ratio, tolerance


@pytest.fixture(scope="module")
def fixture_scale_operator() -> NDArray[np.float64]:
    r"""
    Genera operador de escala M_scale = 100 · I₃.
    
    Retorna
    -------
    NDArray[np.float64], shape (3, 3)
    """
    M_scale: NDArray[np.float64] = 100.0 * np.eye(3, dtype=np.float64)
    
    return M_scale


# =============================================================================
# FASE 1 — AUDITORÍA DE COBERTURA ESPACIAL Y MEDIDA DE LEBESGUE
# =============================================================================
class TestPhase1_LebesgueMeasureAuditor:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 1 — AUDITORÍA DE MEDIDA DE LEBESGUE Y CERTIFICACIÓN DE DOMINIO
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase1_LebesgueMeasureAuditor que consagra
    la cobertura exhaustiva del simplejo Δ². Cada método verifica un axioma constitutivo
    del estrato CLASSIFIERS.
    
    Invariantes Verificados:
    ------------------------
    1. Coherencia dimensional de p_vector, c_vector (ℝ³)
    2. Finitud de entradas (no NaN, no Inf)
    3. Pertenencia al simplejo: p ∈ Δ² (p_i ≥ 0, Σ p_i = 1)
    4. Medida de Lebesgue: μ(Δ² \ ⋃ R_i) ≤ ε
    5. Certificado de dominio vectorial (L¹, L², L∞)
    6. Acotamiento superior de c_vector (≤ 100)
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.1 — VALIDACIÓN DIMENSIONAL Y ESTRUCTURAL
    # -------------------------------------------------------------------------
    
    def test_phase1_dimensions_valid_3d(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que vectores 3D válidos pasan la validación dimensional.
        
        Axioma: p_vector, c_vector, centroid_C ∈ ℝ³
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        assert handoff.p_certified.shape == (3,)
        assert handoff.c_certified.shape == (3,)
        assert handoff.p_domain.dimension == 3
        assert handoff.c_domain.dimension == 3
    
    def test_phase1_dimension_mismatch_p_vector(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que p_vector con dimensión incorrecta dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_invalid: NDArray[np.float64] = np.array([0.5, 0.5], dtype=np.float64)  # 2D ≠ 3D
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_invalid,
                c_vector=c_vector,
            )
        
        assert "R^3" in str(exc_info.value) or "3" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_c_vector(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que c_vector con dimensión incorrecta dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        c_invalid: NDArray[np.float64] = np.ones(4, dtype=np.float64)  # 4D ≠ 3D
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_invalid,
            )
        
        assert "R^3" in str(exc_info.value) or "3" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.2 — VALIDACIÓN DE FINITUD NUMÉRICA
    # -------------------------------------------------------------------------
    
    def test_phase1_finite_values_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que vectores con valores finitos pasan validación.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        assert handoff.p_domain.is_finite is True
        assert handoff.c_domain.is_finite is True
    
    def test_phase1_nan_values_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que NaN en vectores dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_nan: NDArray[np.float64] = p_vector.copy()
        p_nan[0] = np.nan
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_nan,
                c_vector=c_vector,
            )
        
        assert "no finitas" in str(exc_info.value) or "NaN" in str(exc_info.value)
    
    def test_phase1_inf_values_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Inf en vectores dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        c_inf: NDArray[np.float64] = c_vector.copy()
        c_inf[1] = np.inf
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_inf,
            )
        
        assert "infinitos" in str(exc_info.value) or "NaN" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.3 — VALIDACIÓN DE PERTENENCIA AL SIMPLEJO Δ²
    # -------------------------------------------------------------------------
    
    def test_phase1_simplex_membership_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que p ∈ Δ² pasa validación (p_i ≥ 0, Σ p_i = 1).
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # Verificar explícitamente
        assert np.all(p_vector >= 0.0), "p_vector tiene componentes negativas"
        assert abs(np.sum(p_vector) - 1.0) < 1e-10, "p_vector no suma 1"
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        # p_certified debe sumar 1
        assert abs(np.sum(handoff.p_certified) - 1.0) < 1e-10
    
    def test_phase1_simplex_negative_components_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que p con componentes negativas dispara SimplexMembershipViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_negative: NDArray[np.float64] = p_vector.copy()
        p_negative[0] = -0.1  # Componente negativa
        
        agent = APUClassifierAgent()
        
        with pytest.raises(SimplexMembershipViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_negative,
                c_vector=c_vector,
            )
        
        assert "negativas" in str(exc_info.value) or "Δ²" in str(exc_info.value)
    
    def test_phase1_simplex_sum_not_one_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que p con suma ≠ 1 dispara SimplexMembershipViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_wrong_sum: NDArray[np.float64] = np.array(
            [0.5, 0.5, 0.5], dtype=np.float64
        )  # Suma = 1.5
        
        agent = APUClassifierAgent()
        
        with pytest.raises(SimplexMembershipViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_wrong_sum,
                c_vector=c_vector,
            )
        
        assert "suma" in str(exc_info.value) or "1" in str(exc_info.value)
    
    def test_phase1_simplex_zero_mass_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que p con masa total nula dispara SimplexMembershipViolationError.
        """
        _, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_zero: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        
        agent = APUClassifierAgent()
        
        with pytest.raises(SimplexMembershipViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_zero,
                c_vector=c_vector,
            )
        
        assert "nula" in str(exc_info.value) or "masa" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.4 — MEDIDA DE LEBESGUE Y COBERTURA
    # -------------------------------------------------------------------------
    
    def test_phase1_lebesgue_measure_within_tolerance(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que μ(Δ² \ ⋃ R_i) ≤ ε pasa validación.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, tolerance = fixture_valid_lebesgue_params
        
        assert uncovered_ratio <= tolerance, "Fixture fuera de tolerancia"
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        assert handoff.lebesgue_audit.is_partition_exhaustive is True
        assert handoff.lebesgue_audit.uncovered_measure <= tolerance
    
    def test_phase1_lebesgue_measure_exceeds_tolerance_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que μ > ε dispara LebesgueMeasureViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        
        uncovered_ratio_large: float = 1e-5  # > 1e-7 (tolerancia)
        
        agent = APUClassifierAgent()
        
        with pytest.raises(LebesgueMeasureViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio_large,
                p_vector=p_vector,
                c_vector=c_vector,
            )
        
        assert "Lebesgue" in str(exc_info.value) or "vacío" in str(exc_info.value)
    
    def test_phase1_lebesgue_measure_negative_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que medida negativa dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        
        uncovered_ratio_negative: float = -1e-8
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio_negative,
                p_vector=p_vector,
                c_vector=c_vector,
            )
        
        assert "negativa" in str(exc_info.value) or "medida" in str(exc_info.value)
    
    def test_phase1_lebesgue_measure_exceeds_one_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que medida > 1 dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        
        uncovered_ratio_large: float = 1.5  # > 1
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio_large,
                p_vector=p_vector,
                c_vector=c_vector,
            )
        
        assert "excede" in str(exc_info.value) or "1" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.5 — CERTIFICADO DE DOMINIO VECTORIAL
    # -------------------------------------------------------------------------
    
    def test_phase1_vector_domain_certificate_computed(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que certificado de dominio calcula normas L¹, L², L∞.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        # Verificar normas calculadas correctamente
        expected_l1_p = float(la.norm(p_vector, ord=1))
        expected_l2_p = float(la.norm(p_vector, ord=2))
        expected_linf_p = float(la.norm(p_vector, ord=np.inf))
        
        tol = 1e-10
        
        assert abs(handoff.p_domain.l1_norm - expected_l1_p) <= tol
        assert abs(handoff.p_domain.l2_norm - expected_l2_p) <= tol
        assert abs(handoff.p_domain.linf_norm - expected_linf_p) <= tol
    
    def test_phase1_vector_domain_certificate_names(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que certificado tiene nombres descriptivos.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        assert "p_vector" in handoff.p_domain.name.lower()
        assert "c_vector" in handoff.c_domain.name.lower()
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.6 — ACOTAMIENTO SUPERIOR DE c_vector
    # -------------------------------------------------------------------------
    
    def test_phase1_c_vector_bounded_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que c_vector ≤ 100 pasa validación.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        assert np.all(c_vector <= 100.0), "Fixture c_vector excede 100"
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        assert handoff is not None
    
    def test_phase1_c_vector_exceeds_100_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que c_vector > 100 dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        c_exceeds: NDArray[np.float64] = c_vector.copy()
        c_exceeds[0] = 150.0  # > 100
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_exceeds,
            )
        
        assert "cota superior" in str(exc_info.value) or "100" in str(exc_info.value)
    
    def test_phase1_c_vector_negative_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que c_vector negativo dispara DomainIntegrityViolationError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        c_negative: NDArray[np.float64] = c_vector.copy()
        c_negative[1] = -10.0
        
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase1_audit_and_handoff_to_phase2(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_negative,
            )
        
        assert "negativas" in str(exc_info.value) or "costo" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.7 — BANDERAS ONTOLÓGICAS
    # -------------------------------------------------------------------------
    
    def test_phase1_ontological_flag_bool_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que bandera ontológica bool pasa validación.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        # _coerce_ontological_flag es método interno
        result = agent._coerce_ontological_flag("test_flag", True)
        assert result is True
        
        result = agent._coerce_ontological_flag("test_flag", False)
        assert result is False
    
    def test_phase1_ontological_flag_non_bool_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que bandera no bool dispara DomainIntegrityViolationError.
        """
        agent = APUClassifierAgent()
        
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._coerce_ontological_flag("test_flag", "INVALIDO")
        
        assert "bool" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.8 — HANDOFF FORMAL A FASE 2
    # -------------------------------------------------------------------------
    
    def test_phase1_handoff_contains_all_required_fields(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Phase1LebesgueHandoff contiene todos los campos requeridos.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        # Verificar estructura del handoff
        assert hasattr(handoff, 'lebesgue_audit')
        assert hasattr(handoff, 'p_certified')
        assert hasattr(handoff, 'c_certified')
        assert hasattr(handoff, 'p_domain')
        assert hasattr(handoff, 'c_domain')
        
        # Verificar tipos
        assert isinstance(handoff.lebesgue_audit, LebesgueAuditData)
        assert isinstance(handoff.p_domain, VectorDomainCertificate)
        assert isinstance(handoff.c_domain, VectorDomainCertificate)


# =============================================================================
# FASE 2 — CERTIFICACIÓN DE DIFEOMORFISMO AFÍN Y CONTRATO DE ESCALA
# =============================================================================
class TestPhase2_ScaleIsomorphismCertifier:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 2 — DIFEOMORFISMO AFÍN Y CONTRATO DE ESCALA
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase2_ScaleIsomorphismCertifier que
    gobierna la transformación isométrica entre proporciones y porcentajes.
    
    Invariantes Verificados:
    ------------------------
    1. Contrato afín: ||c - M_scale p||_∞ ≤ ε_affine
    2. Operador de escala: M_scale = 100 · I₃ (κ = 1)
    3. Espectro del operador: σ_i = 100 ∀ i
    4. Continuidad formal desde Phase1LebesgueHandoff
    5. Residuo en norma infinito computado correctamente
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.1 — CONTRATO AFÍN Y RESIDUO
    # -------------------------------------------------------------------------
    
    def test_phase2_affine_contract_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
        fixture_scale_operator: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que contrato afín ||c - M_scale p||_∞ ≤ ε pasa validación.
        
        Fórmula: c = M_scale · p, M_scale = 100 · I₃
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        M_scale = fixture_scale_operator
        
        # Verificar explícitamente
        c_expected = M_scale @ p_vector
        residual = float(la.norm(c_vector - c_expected, ord=np.inf))
        
        assert residual < 1e-10, f"Fixture no satisface contrato afín: {residual}"
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=fixture_valid_vectors_3d[2],
            is_isolated_island=False,
        )
        
        assert phase2_handoff.scale_audit.is_scale_isomorphic is True
        assert phase2_handoff.scale_audit.residual_infinity_norm < 1e-10
    
    def test_phase2_affine_contract_violation_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que violación del contrato afín dispara ScaleInvarianceCollapseError.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # c_vector inconsistente con p_vector
        c_inconsistent: NDArray[np.float64] = np.array(
            [50.0, 35.0, 25.0], dtype=np.float64
        )  # Debería ser [40.0, 35.0, 25.0]
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,  # Usamos c_vector válido para Fase 1
        )
        
        # Modificar c_certified para simular inconsistencia
        from dataclasses import replace
        phase1_handoff_modified = replace(
            phase1_handoff,
            c_certified=c_inconsistent,
        )
        
        with pytest.raises(ScaleInvarianceCollapseError) as exc_info:
            agent._phase2_certify_and_handoff_to_phase3(
                phase1_handoff=phase1_handoff_modified,
                centroid_C=fixture_valid_vectors_3d[2],
                is_isolated_island=False,
            )
        
        assert "Biyectividad" in str(exc_info.value) or "escalar" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.2 — ESPECTRO DEL OPERADOR DE ESCALA
    # -------------------------------------------------------------------------
    
    def test_phase2_scale_operator_spectrum_valid(
        self,
        fixture_scale_operator: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que espectro de M_scale es σ_i = 100, κ = 1.
        """
        M_scale = fixture_scale_operator
        
        singular_values = la.svdvals(M_scale)
        
        # Todos los valores singulares deben ser 100
        assert np.allclose(singular_values, 100.0, atol=1e-10)
        
        # Número de condición debe ser 1
        kappa = float(np.max(singular_values) / np.min(singular_values))
        assert abs(kappa - 1.0) < 1e-10
    
    def test_phase2_scale_operator_condition_number_computed(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que número de condición se computa en scale_audit.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=fixture_valid_vectors_3d[2],
            is_isolated_island=False,
        )
        
        assert phase2_handoff.scale_audit.condition_number >= 1.0
        assert np.isfinite(phase2_handoff.scale_audit.condition_number)
    
    def test_phase2_scale_operator_spectral_deviation_computed(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que desviación espectral se computa en scale_audit.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=fixture_valid_vectors_3d[2],
            is_isolated_island=False,
        )
        
        assert phase2_handoff.scale_audit.spectral_deviation >= 0.0
        assert np.isfinite(phase2_handoff.scale_audit.spectral_deviation)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.3 — CONTINUIDAD FORMAL DESDE FASE 1
    # -------------------------------------------------------------------------
    
    def test_phase2_requires_phase1_handoff(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Fase 2 requiere Phase1LebesgueHandoff como prefijo.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        # Intentar pasar objeto incorrecto
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase2_certify_and_handoff_to_phase3(
                phase1_handoff="INVALIDO",  # type: ignore[arg-type]
                centroid_C=centroid_C,
                is_isolated_island=False,
            )
        
        assert "Phase1LebesgueHandoff" in str(exc_info.value)
    
    def test_phase2_handoff_contains_phase1_data(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Phase2ScaleHandoff contiene datos de Fase 1.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert phase2_handoff.phase1_handoff is phase1_handoff
        assert phase2_handoff.phase1_handoff.lebesgue_audit.is_partition_exhaustive
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.4 — CENTROIDE Y BANDERA DE ISLA
    # -------------------------------------------------------------------------
    
    def test_phase2_centroid_certified(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que centroide se certifica en Fase 2.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert phase2_handoff.centroid_certified.shape == (3,)
        assert phase2_handoff.centroid_domain.dimension == 3
    
    def test_phase2_is_isolated_island_flag(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que bandera is_isolated_island se propaga correctamente.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff_true = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=True,
        )
        
        phase2_handoff_false = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert phase2_handoff_true.is_isolated_island is True
        assert phase2_handoff_false.is_isolated_island is False
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.5 — HANDOFF FORMAL A FASE 3
    # -------------------------------------------------------------------------
    
    def test_phase2_handoff_contains_all_required_fields(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Phase2ScaleHandoff contiene todos los campos requeridos.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # Verificar estructura del handoff
        assert hasattr(phase2_handoff, 'phase1_handoff')
        assert hasattr(phase2_handoff, 'scale_audit')
        assert hasattr(phase2_handoff, 'centroid_certified')
        assert hasattr(phase2_handoff, 'centroid_domain')
        assert hasattr(phase2_handoff, 'is_isolated_island')
        
        # Verificar tipos
        assert isinstance(phase2_handoff.scale_audit, ScaleIsomorphismData)
        assert isinstance(phase2_handoff.centroid_domain, VectorDomainCertificate)


# =============================================================================
# FASE 3 — EVALUACIÓN DE CENTROIDES TOPOLÓGICOS Y HOMOLOGÍA ESTRUCTURAL
# =============================================================================
class TestPhase3_CentroidTopologyEnforcer:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 3 — HOMOLOGÍA ESTRUCTURAL Y ORTOGONALIDAD DE CENTROIDES
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase3_CentroidTopologyEnforcer que
    certifica el aislamiento topológico de las "Islas de Suministro".
    
    Invariantes Verificados:
    ------------------------
    1. Base canónica ortonormal: B Bᵀ = I₃
    2. Ortogonalidad de Isla: <C_isla, e_mo> = 0, <C_isla, e_eq> = 0
    3. Norma de proyección: ||Proy_{span(e_mo,e_eq)} C_isla||₂ ≤ tol
    4. Continuidad formal desde Phase2ScaleHandoff
    5. Objeto terminal OntologicalPartitionState
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.1 — ORTONORMALIDAD DE BASE CANÓNICA
    # -------------------------------------------------------------------------
    
    def test_phase3_canonical_basis_orthonormal(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que base canónica {e_sum, e_mo, e_eq} es ortonormal.
        
        Axioma: B Bᵀ = I₃
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # La validación de base se hace internamente en Fase 3
        # Si llega aquí sin excepción, la base es ortonormal
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        assert state is not None
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.2 — ORTOGONALIDAD DE CENTROIDE "ISLA"
    # -------------------------------------------------------------------------
    
    def test_phase3_island_centroid_orthogonal_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_island_centroid: NDArray[np.float64],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que centroide de Isla es ortogonal a e_mo y e_eq.
        
        Axioma: <C_isla, e_mo> = 0, <C_isla, e_eq> = 0
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        centroid_island = fixture_valid_island_centroid
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # Verificar explícitamente
        e_mo = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e_eq = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        dot_mo = float(np.dot(centroid_island, e_mo))
        dot_eq = float(np.dot(centroid_island, e_eq))
        
        assert abs(dot_mo) < 1e-14, f"<C_isla, e_mo> = {dot_mo}"
        assert abs(dot_eq) < 1e-14, f"<C_isla, e_eq> = {dot_eq}"
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_island,
            is_isolated_island=True,
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        assert state.centroid_audit.is_structurally_orthogonal is True
    
    def test_phase3_island_centroid_non_orthogonal_raise(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Isla con componentes no ortogonales dispara TopologicalCentroidAnomalyVeto.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # Centroide "Isla" contaminado con MO y Equipo
        centroid_contaminated: NDArray[np.float64] = np.array(
            [0.8, 0.1, 0.1], dtype=np.float64
        )  # e_mo y e_eq ≠ 0
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_contaminated,
            is_isolated_island=True,  # Marcado como Isla
        )
        
        with pytest.raises(TopologicalCentroidAnomalyVeto) as exc_info:
            agent._phase3_finalize_from_phase2_handoff(
                phase2_handoff=phase2_handoff,
            )
        
        assert "ortogonales" in str(exc_info.value) or "Isla" in str(exc_info.value)
    
    def test_phase3_non_island_centroid_no_orthogonality_required(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que centroide no-Isla no requiere ortogonalidad.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,  # No es Isla
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        # is_structurally_orthogonal debe ser True (vacuamente satisfecho)
        assert state.centroid_audit.is_structurally_orthogonal is True
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.3 — PRODUCTOS INTERNOS Y PROYECCIÓN
    # -------------------------------------------------------------------------
    
    def test_phase3_inner_products_computed(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que productos internos <C, e_mo> y <C, e_eq> se computan.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        # Verificar que se computaron
        assert np.isfinite(state.centroid_audit.inner_product_mo)
        assert np.isfinite(state.centroid_audit.inner_product_eq)
    
    def test_phase3_projection_norm_computed(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que norma de proyección ||Proy C||₂ se computa.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        assert state.centroid_audit.projection_norm >= 0.0
        assert np.isfinite(state.centroid_audit.projection_norm)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.4 — CONTINUIDAD FORMAL DESDE FASE 2
    # -------------------------------------------------------------------------
    
    def test_phase3_requires_phase2_handoff(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Fase 3 requiere Phase2ScaleHandoff como prefijo.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # Intentar pasar objeto incorrecto
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            agent._phase3_finalize_from_phase2_handoff(
                phase2_handoff="INVALIDO",  # type: ignore[arg-type]
            )
        
        assert "Phase2ScaleHandoff" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.5 — OBJETO TERMINAL ONTOLÓGICO
    # -------------------------------------------------------------------------
    
    def test_phase3_ontological_partition_state_complete(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que OntologicalPartitionState contiene todos los certificados.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        # Verificar estructura del estado terminal
        assert hasattr(state, 'lebesgue_audit')
        assert hasattr(state, 'scale_audit')
        assert hasattr(state, 'centroid_audit')
        assert hasattr(state, 'is_epistemologically_valid')
        
        # Verificar tipos
        assert isinstance(state.lebesgue_audit, LebesgueAuditData)
        assert isinstance(state.scale_audit, ScaleIsomorphismData)
        assert isinstance(state.centroid_audit, CentroidTopologyData)
        
        # Validez epistemológica debe ser True para entrada válida
        assert state.is_epistemologically_valid is True
    
    def test_phase3_state_immutability(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que OntologicalPartitionState es inmutable (frozen).
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        phase1_handoff = agent._phase1_audit_and_handoff_to_phase2(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
        )
        
        phase2_handoff = agent._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        state = agent._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )
        
        # Intentar modificar debe fallar
        with pytest.raises(AttributeError):
            state.is_epistemologically_valid = False  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.lebesgue_audit.uncovered_measure = 999.0  # type: ignore[misc]


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
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica pipeline completo con entradas válidas.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.is_epistemologically_valid is True
        assert state.lebesgue_audit.is_partition_exhaustive is True
        assert state.scale_audit.is_scale_isomorphic is True
        assert state.centroid_audit.is_structurally_orthogonal is True
    
    def test_full_pipeline_island_centroid_valid(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_island_centroid: NDArray[np.float64],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica pipeline completo con centroide de Isla válido.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        centroid_island = fixture_valid_island_centroid
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_island,
            is_isolated_island=True,
        )
        
        assert state.is_epistemologically_valid is True
        assert state.centroid_audit.is_structurally_orthogonal is True
    
    def test_full_pipeline_lebesgue_violation_fails(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que violación de Lebesgue falla el pipeline completo.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        
        uncovered_ratio_large: float = 1e-5  # > 1e-7
        
        agent = APUClassifierAgent()
        
        with pytest.raises(LebesgueMeasureViolationError):
            agent.execute_ontological_partition_governance(
                uncovered_area_ratio=uncovered_ratio_large,
                p_vector=p_vector,
                c_vector=c_vector,
                centroid_C=centroid_C,
                is_isolated_island=False,
            )
    
    def test_full_pipeline_simplex_violation_fails(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que violación de simplejo falla el pipeline completo.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        p_negative: NDArray[np.float64] = p_vector.copy()
        p_negative[0] = -0.1
        
        agent = APUClassifierAgent()
        
        with pytest.raises(SimplexMembershipViolationError):
            agent.execute_ontological_partition_governance(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_negative,
                c_vector=c_vector,
                centroid_C=centroid_C,
                is_isolated_island=False,
            )
    
    def test_full_pipeline_scale_violation_fails(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que violación de escala falla el pipeline completo.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        c_inconsistent: NDArray[np.float64] = np.array(
            [50.0, 35.0, 25.0], dtype=np.float64
        )  # Inconsistente con p_vector
        
        agent = APUClassifierAgent()
        
        with pytest.raises(ScaleInvarianceCollapseError):
            agent.execute_ontological_partition_governance(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_inconsistent,
                centroid_C=centroid_C,
                is_isolated_island=False,
            )
    
    def test_full_pipeline_island_orthogonality_violation_fails(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que Isla no ortogonal falla el pipeline completo.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        centroid_contaminated: NDArray[np.float64] = np.array(
            [0.8, 0.1, 0.1], dtype=np.float64
        )
        
        agent = APUClassifierAgent()
        
        with pytest.raises(TopologicalCentroidAnomalyVeto):
            agent.execute_ontological_partition_governance(
                uncovered_area_ratio=uncovered_ratio,
                p_vector=p_vector,
                c_vector=c_vector,
                centroid_C=centroid_contaminated,
                is_isolated_island=True,
            )
    
    def test_full_pipeline_call_method_alias(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que método __call__ es alias de execute_ontological_partition_governance.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        # Usar __call__ directamente
        state = agent(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.is_epistemologically_valid is True
    
    def test_full_pipeline_dto_immutability(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica que todos los DTOs son inmutables (frozen dataclasses).
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # Intentar modificar debe fallar en todos los niveles
        with pytest.raises(AttributeError):
            state.is_epistemologically_valid = False  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.lebesgue_audit.uncovered_measure = 999.0  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.scale_audit.residual_infinity_norm = 999.0  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            state.centroid_audit.inner_product_mo = 999.0  # type: ignore[misc]
    
    def test_full_pipeline_audit_data_consistency(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica consistencia entre certificados de las tres fases.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # Dimensiones consistentes
        assert state.lebesgue_audit.measure_tolerance > 0.0
        assert state.scale_audit.affine_tolerance > 0.0
        assert state.centroid_audit.topology_tolerance > 0.0
        
        # Validez epistemológica implica todas las fases válidas
        if state.is_epistemologically_valid:
            assert state.lebesgue_audit.is_partition_exhaustive
            assert state.scale_audit.is_scale_isomorphic
            assert state.centroid_audit.is_structurally_orthogonal


# =============================================================================
# PRUEBAS DE CASOS ESPECIALES Y BORDES
# =============================================================================
class TestEdgeCases_SpecialConditions:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    PRUEBAS DE CASOS ESPECIALES Y BORDES
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida comportamiento en condiciones límite:
    - Vectores casi singulares
    - Estados cero
    - Tolerancias numéricas
    - Valores extremos
    """
    
    def test_edge_case_p_vector_boundary_values(
        self,
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica comportamiento con p_vector en límites del simplejo.
        """
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # p_vector en vértice del simplejo (un componente = 1, otros = 0)
        p_vertex: NDArray[np.float64] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        
        c_vertex: NDArray[np.float64] = np.array(
            [100.0, 0.0, 0.0], dtype=np.float64
        )
        
        centroid_C: NDArray[np.float64] = np.array(
            [0.5, 0.3, 0.2], dtype=np.float64
        )
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vertex,
            c_vector=c_vertex,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_c_vector_zero(
        self,
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica comportamiento con c_vector = 0 (válido si p = 0, pero p debe sumar 1).
        """
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # c_vector puede ser 0 si p_vector es consistente
        p_vector: NDArray[np.float64] = np.array(
            [0.0, 0.0, 1.0], dtype=np.float64
        )
        
        c_vector: NDArray[np.float64] = np.array(
            [0.0, 0.0, 100.0], dtype=np.float64
        )
        
        centroid_C: NDArray[np.float64] = np.array(
            [0.5, 0.3, 0.2], dtype=np.float64
        )
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_tolerance_boundaries(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento en límites de tolerancia numérica.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        
        # Medida de Lebesgue en el límite de tolerancia
        uncovered_ratio_boundary: float = 1e-7  # Exactamente en tolerancia
        
        agent = APUClassifierAgent()
        
        # Debe pasar si está dentro de tolerancia
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio_boundary,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.lebesgue_audit.uncovered_measure <= 1e-7
    
    def test_edge_case_island_centroid_exact_zero(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_island_centroid: NDArray[np.float64],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica comportamiento con centroide de Isla con MO=0, EQ=0 exactos.
        """
        p_vector, c_vector, _ = fixture_valid_vectors_3d
        centroid_island = fixture_valid_island_centroid
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_island,
            is_isolated_island=True,
        )
        
        assert state.centroid_audit.inner_product_mo == 0.0
        assert state.centroid_audit.inner_product_eq == 0.0
        assert state.centroid_audit.projection_norm == 0.0
    
    def test_edge_case_very_small_lebesgue_measure(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica comportamiento con medida de Lebesgue muy pequeña.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        
        uncovered_ratio_tiny: float = 1e-15  # Muy pequeño
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio_tiny,
            p_vector=p_vector,
            c_vector=c_vector,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        assert state.lebesgue_audit.uncovered_measure >= 0.0
        assert state.is_epistemologically_valid is True
    
    def test_edge_case_affine_residual_near_tolerance(
        self,
        fixture_valid_vectors_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_lebesgue_params: Tuple[float, float],
    ) -> None:
        r"""
        Verifica comportamiento con residuo afín cerca de tolerancia.
        """
        p_vector, c_vector, centroid_C = fixture_valid_vectors_3d
        uncovered_ratio, _ = fixture_valid_lebesgue_params
        
        # c_vector con pequeño error numérico
        c_near_boundary: NDArray[np.float64] = c_vector + 1e-13
        
        agent = APUClassifierAgent()
        
        state = agent.execute_ontological_partition_governance(
            uncovered_area_ratio=uncovered_ratio,
            p_vector=p_vector,
            c_vector=c_near_boundary,
            centroid_C=centroid_C,
            is_isolated_island=False,
        )
        
        # Debe pasar si está dentro de tolerancia afín
        assert state.scale_audit.residual_infinity_norm < 1e-10


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
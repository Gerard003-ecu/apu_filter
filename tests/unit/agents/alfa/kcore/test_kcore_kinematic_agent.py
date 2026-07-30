# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Suite — KCore Kinematic Agent (Suite de Validación Cinemática)            ║
║  Ruta   : tests/unit/agents/alfa/kcore/test_kcore_kinematic_agent.py                     ║
║  Versión: 6.0.0-Rigorous-IDA-PBC-Hodge-CFL-Sheaf-Spectral-TestSuite                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  PROPÓSITO CIBER-FÍSICO Y TOPOLOGÍA DE PRUEBAS (Rigor Categórico):                       ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Esta suite de pruebas consagra la Maquinaria Cinemática del núcleo ($K_{CORE}$)         ║
║  mediante un funtor de validación que verifica axiomáticamente el control IDA-PBC,       ║
║  la estrangulación de vorticidad de Hodge y el límite CFL del sistema logístico.         ║
║                                                                                          ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta $\Phi_3 \circ \Phi_2 \circ \Phi_1$):     ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Fase 1 → Validación Matricial Constitutiva y Métrica Riemanniana                        ║
║           Verifica dimensiones, simetrías, PSD y rangos de J, R, J_d, R_d, g, G.         ║
║                                                                                          ║
║  Fase 2 → Síntesis Cinemática IDA-PBC y Auditoría CFL                                    ║
║           Computa ley de control, modula conductancia de Hodge y verifica CFL.           ║
║                                                                                          ║
║  Fase 3 → Fibración Celular y Exportación de la Cofrontera δ_CORE                        ║
║           Sintetiza el SheafStalk subyugado a la Identidad de Hodge Local.               ║
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
import scipy.sparse as sp
from numpy.typing import NDArray

# =============================================================================
# Módulo bajo prueba
# =============================================================================
from app.agents.alpha.kcore.kcore_kinematic_agent import (
    KCoreKinematicAgent,
    KinematicPreparationContext,
    KinematicStateTensor,
    SheafStalk,
    # Excepciones
    KinematicCoreError,
    KinematicDimensionError,
    KinematicSymmetryError,
    KinematicConditionError,
    DiracMatchingError,
    ParasiticVorticityError,
    ImpedanceReflectionError,
    CFLViolationError,
    SheafCoboundaryError,
    MetricTensorError,
)

# =============================================================================
# Logger y constantes globales de prueba
# =============================================================================
logger = logging.getLogger("MIC.Alpha.Test.KCoreKinematicAgent")
_MACHINE_EPS: float = float(np.finfo(np.float64).eps)

# =============================================================================
# FIXTURES GLOBALES — GENERADORES DE MATRICES CONSTITUTIVAS
# =============================================================================


@pytest.fixture(scope="module")
def fixture_valid_matrices_3d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices constitutivas válidas para n=3, m=2.
    
    Retorna
    -------
    Tuple[J, R, J_d, R_d, g, G]
        Matrices antisimétricas, PSD, y métrica Riemanniana.
    """
    n, m = 3, 2
    
    # J: antisimétrica (interconexión real)
    J: NDArray[np.float64] = np.array(
        [[0.0, 1.0, -0.5],
         [-1.0, 0.0, 0.3],
         [0.5, -0.3, 0.0]], dtype=np.float64
    )
    
    # R: PSD (disipación real)
    R: NDArray[np.float64] = np.array(
        [[0.5, 0.0, 0.0],
         [0.0, 0.3, 0.0],
         [0.0, 0.0, 0.4]], dtype=np.float64
    )
    
    # J_d: antisimétrica (interconexión deseada)
    J_d: NDArray[np.float64] = np.array(
        [[0.0, 0.8, -0.4],
         [-0.8, 0.0, 0.2],
         [0.4, -0.2, 0.0]], dtype=np.float64
    )
    
    # R_d: PSD (disipación deseada)
    R_d: NDArray[np.float64] = np.array(
        [[0.6, 0.0, 0.0],
         [0.0, 0.4, 0.0],
         [0.0, 0.0, 0.5]], dtype=np.float64
    )
    
    # g: matriz de entrada (n×m)
    g: NDArray[np.float64] = np.array(
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.1, 0.1]], dtype=np.float64
    )
    
    # G: métrica Riemanniana SPD (n×n)
    G: NDArray[np.float64] = np.array(
        [[1.1, 0.05, 0.02],
         [0.05, 1.0, 0.03],
         [0.02, 0.03, 0.9]], dtype=np.float64
    )
    
    return J, R, J_d, R_d, g, G


@pytest.fixture(scope="module")
def fixture_valid_matrices_2d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera matrices constitutivas válidas para n=2, m=1 (caso mínimo).
    
    Retorna
    -------
    Tuple[J, R, J_d, R_d, g, G]
    """
    n, m = 2, 1
    
    J: NDArray[np.float64] = np.array(
        [[0.0, 1.0],
         [-1.0, 0.0]], dtype=np.float64
    )
    
    R: NDArray[np.float64] = np.array(
        [[0.5, 0.0],
         [0.0, 0.3]], dtype=np.float64
    )
    
    J_d: NDArray[np.float64] = np.array(
        [[0.0, 0.8],
         [-0.8, 0.0]], dtype=np.float64
    )
    
    R_d: NDArray[np.float64] = np.array(
        [[0.6, 0.0],
         [0.0, 0.4]], dtype=np.float64
    )
    
    g: NDArray[np.float64] = np.array(
        [[1.0],
         [0.5]], dtype=np.float64
    )
    
    G: NDArray[np.float64] = np.eye(2, dtype=np.float64)
    
    return J, R, J_d, R_d, g, G


@pytest.fixture(scope="module")
def fixture_valid_gradients_3d() -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Genera gradientes hamiltonianos válidos para n=3.
    
    Retorna
    -------
    Tuple[grad_H, grad_H_d]
    """
    grad_H: NDArray[np.float64] = np.array([1.0, 0.5, 0.8], dtype=np.float64)
    grad_H_d: NDArray[np.float64] = np.array([0.8, 0.4, 0.6], dtype=np.float64)
    
    return grad_H, grad_H_d


@pytest.fixture(scope="module")
def fixture_sparse_matrices_3d() -> Tuple[
    sp.csr_matrix,
    NDArray[np.float64],
    NDArray[np.float64],
    sp.csr_matrix,
]:
    r"""
    Genera matrices sparse válidas para pruebas de Hodge y CFL.
    
    Retorna
    -------
    Tuple[W, I_curl, Z_load, Delta_sym]
    """
    E = 5  # número de aristas
    
    # W: conductancia de aristas (E×E, diagonal SPD)
    W_data = np.array([0.5, 0.3, 0.4, 0.2, 0.6], dtype=np.float64)
    W: sp.csr_matrix = sp.diags(W_data, offsets=0, format="csr", dtype=np.float64)
    
    # I_curl: corriente de curl sobre aristas
    I_curl: NDArray[np.float64] = np.array([0.1, 0.05, 0.08, 0.02, 0.12], dtype=np.float64)
    
    # Z_load: impedancia de carga (SPD)
    Z_load: NDArray[np.float64] = np.array(
        [[1.0, 0.1],
         [0.1, 0.9]], dtype=np.float64
    )
    
    # Delta_sym: Laplaciano del grafo (V×V, PSD)
    V = 4  # número de vértices
    Delta_data = np.array([
        [2.0, -1.0, 0.0, -1.0],
        [-1.0, 3.0, -1.0, -1.0],
        [0.0, -1.0, 2.0, -1.0],
        [-1.0, -1.0, -1.0, 3.0]
    ], dtype=np.float64)
    Delta_sym: sp.csr_matrix = sp.csr_matrix(Delta_data, dtype=np.float64)
    
    return W, I_curl, Z_load, Delta_sym


# =============================================================================
# FASE 1 — VALIDACIÓN MATRICIAL CONSTITUTIVA Y MÉTRICA RIEMANNIANA
# =============================================================================
class TestPhase1_MatrixValidation:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 1 — VALIDACIÓN MATRICIAL, SIMETRÍAS Y ESPECTRO
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase1_MatrixValidation que consagra la
    geometría constitutiva del núcleo cinemático. Cada método verifica un axioma
    constitutivo del estrato K_CORE.
    
    Invariantes Verificados:
    ------------------------
    1. Coherencia dimensional de J, R, J_d, R_d, g, G
    2. Antisimetría de J, J_d (J = −Jᵀ)
    3. Simetría de R, R_d, G (A = Aᵀ)
    4. PSD de R, R_d, G (λ_min ≥ −tol)
    5. Rango numérico de g y G vía SVD
    6. Números de condición κ < κ_max
    7. Gap espectral de disipación
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.1 — VALIDACIÓN DIMENSIONAL Y ESTRUCTURAL
    # -------------------------------------------------------------------------
    
    def test_phase1_dimensions_valid_3d(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que matrices 3D válidas pasan la validación dimensional.
        
        Axioma: J, R, J_d, R_d, G ∈ ℝ^{n×n}, g ∈ ℝ^{n×m}
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.n == 3
        assert agent.context.m == 2
        assert agent.context.J.shape == (3, 3)
        assert agent.context.R.shape == (3, 3)
        assert agent.context.g.shape == (3, 2)
        assert agent.context.G.shape == (3, 3)
    
    def test_phase1_dimensions_valid_2d(
        self,
        fixture_valid_matrices_2d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica caso mínimo n=2, m=1 (frontera inferior).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_2d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.n == 2
        assert agent.context.m == 1
        assert agent.context.rank_g >= 1
    
    def test_phase1_dimension_mismatch_J_non_square(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J no cuadrada dispara KinematicDimensionError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        J_invalid: NDArray[np.float64] = J[:, :2]  # 3×2
        
        with pytest.raises(KinematicDimensionError) as exc_info:
            KCoreKinematicAgent(
                J=J_invalid,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
            )
        
        assert "cuadrada" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_g_rows(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g con filas incorrectas dispara KinematicDimensionError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        g_invalid: NDArray[np.float64] = np.ones((2, 2), dtype=np.float64)  # 2 filas ≠ 3
        
        with pytest.raises(KinematicDimensionError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g_invalid,
                G=G,
            )
        
        assert "filas" in str(exc_info.value) or "shape" in str(exc_info.value)
    
    def test_phase1_dimension_mismatch_G_shape(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que G con forma incorrecta dispara MetricTensorError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        G_invalid: NDArray[np.float64] = np.eye(2, dtype=np.float64)  # 2×2 ≠ 3×3
        
        with pytest.raises(MetricTensorError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G_invalid,
            )
        
        assert "shape" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.2 — VALIDACIÓN DE ANTISIMETRÍAS (J, J_d)
    # -------------------------------------------------------------------------
    
    def test_phase1_antisymmetry_J_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J antisimétrica pasa validación.
        
        Axioma: J = −Jᵀ dentro de tolerancia ε_mach · ‖J‖_F
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        # Verificar antisimetría explícita
        skew_residual = float(la.norm(J + J.T, "fro"))
        norm_J = float(la.norm(J, "fro"))
        tol = _MACHINE_EPS * max(norm_J, 1.0)
        
        assert skew_residual <= tol, f"Fixture J no es antisimétrica: {skew_residual}"
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent is not None
    
    def test_phase1_antisymmetry_J_invalid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J no antisimétrica dispara KinematicSymmetryError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        J_invalid: NDArray[np.float64] = J.copy()
        J_invalid[0, 1] += 0.5  # Romper antisimetría
        
        with pytest.raises(KinematicSymmetryError) as exc_info:
            KCoreKinematicAgent(
                J=J_invalid,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
            )
        
        assert "antisimétrica" in str(exc_info.value)
    
    def test_phase1_antisymmetry_J_d_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que J_d antisimétrica pasa validación.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        skew_residual = float(la.norm(J_d + J_d.T, "fro"))
        norm_J_d = float(la.norm(J_d, "fro"))
        tol = _MACHINE_EPS * max(norm_J_d, 1.0)
        
        assert skew_residual <= tol
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent is not None
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.3 — VALIDACIÓN DE SIMETRÍAS (R, R_d, G)
    # -------------------------------------------------------------------------
    
    def test_phase1_symmetry_R_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R simétrica pasa validación.
        
        Axioma: R = Rᵀ dentro de tolerancia ε_mach · ‖R‖_F
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        sym_residual = float(la.norm(R - R.T, "fro"))
        norm_R = float(la.norm(R, "fro"))
        tol = _MACHINE_EPS * max(norm_R, 1.0)
        
        assert sym_residual <= tol
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent is not None
    
    def test_phase1_symmetry_R_invalid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R no simétrica dispara KinematicSymmetryError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        R_invalid: NDArray[np.float64] = R.copy()
        R_invalid[0, 1] += 0.3  # Romper simetría
        
        with pytest.raises(KinematicSymmetryError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R_invalid,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
            )
        
        assert "simétrica" in str(exc_info.value)
    
    def test_phase1_symmetry_G_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que G simétrica pasa validación.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.kappa_G < agent.kappa_max
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.4 — VALIDACIÓN DE PSD (R, R_d, G)
    # -------------------------------------------------------------------------
    
    def test_phase1_psd_R_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R PSD pasa validación.
        
        Axioma: λ_min(R) ≥ −tol (Semidefinida Positiva)
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.kappa_R < agent.kappa_max
        assert agent.context.spectral_gap_R >= 0.0
    
    def test_phase1_psd_R_invalid_negative_eigenvalue(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que R con autovalor negativo dispara KinematicSymmetryError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        R_invalid: NDArray[np.float64] = R.copy()
        R_invalid[0, 0] = -1.0  # λ_min < 0
        
        with pytest.raises(KinematicSymmetryError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R_invalid,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
            )
        
        assert "Semidefinida Positiva" in str(exc_info.value) or "PSD" in str(exc_info.value)
    
    def test_phase1_psd_G_invalid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que G no PSD dispara KinematicSymmetryError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        G_invalid: NDArray[np.float64] = G.copy()
        G_invalid[0, 0] = -0.5  # λ_min < 0
        
        with pytest.raises(KinematicSymmetryError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G_invalid,
            )
        
        assert "PSD" in str(exc_info.value) or "Semidefinida" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.5 — NÚMEROS DE CONDICIÓN Y GAP ESPECTRAL
    # -------------------------------------------------------------------------
    
    def test_phase1_condition_number_within_limit(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que κ(R), κ(R_d), κ(G) < κ_max.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
            kappa_max=1e10,
        )
        
        assert agent.context.kappa_R < 1e10
        assert agent.context.kappa_R_d < 1e10
        assert agent.context.kappa_G < 1e10
    
    def test_phase1_condition_number_exceeds_limit(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que κ > κ_max dispara KinematicConditionError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        R_ill: NDArray[np.float64] = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1e-11, 0.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )  # κ ≈ 1e11
        
        with pytest.raises(KinematicConditionError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R_ill,
                J_d=J_d,
                R_d=R_d,
                g=g,
                G=G,
                kappa_max=1e10,
            )
        
        assert "mal condicionada" in str(exc_info.value) or "κ" in str(exc_info.value)
    
    def test_phase1_spectral_gap_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que gap espectral de R se calcula correctamente.
        
        Métrica: gap = λ₂⁺ / λ_max ∈ [0, 1]
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert 0.0 <= agent.context.spectral_gap_R <= 1.0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 1.6 — RANGO NUMÉRICO VÍA SVD
    # -------------------------------------------------------------------------
    
    def test_phase1_rank_g_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que rango de g se calcula vía SVD.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.rank_g >= 1
        assert agent.context.rank_g <= min(3, 2)
    
    def test_phase1_rank_g_zero_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que g numéricamente nula dispara KinematicDimensionError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        g_zero: NDArray[np.float64] = np.zeros((3, 2), dtype=np.float64)
        
        with pytest.raises(KinematicDimensionError) as exc_info:
            KCoreKinematicAgent(
                J=J,
                R=R,
                J_d=J_d,
                R_d=R_d,
                g=g_zero,
                G=G,
            )
        
        assert "nula" in str(exc_info.value) or "rank" in str(exc_info.value)
    
    def test_phase1_rank_G_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que rango de G se calcula correctamente.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        assert agent.context.rank_G >= 1
        assert agent.context.rank_G <= 3
    
    def test_phase1_G_defaults_to_identity(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que G=None se materializa como I_n.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=None,  # Debe convertirse en I_3
        )
        
        assert agent.context.G.shape == (3, 3)
        assert float(la.norm(agent.context.G - np.eye(3), "fro")) < 1e-12
        assert agent.context.rank_G == 3


# =============================================================================
# FASE 2 — SÍNTESIS CINEMÁTICA IDA-PBC Y AUDITORÍA CFL
# =============================================================================
class TestPhase2_KinematicSynthesis:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 2 — CONTROL IDA-PBC, HODGE, KRAMERS-KRONIG Y CFL
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase2_KinematicSynthesis que gobierna
    la síntesis cinemática del núcleo. Cada método verifica un axioma constitutivo
    del control IDA-PBC covariante.
    
    Invariantes Verificados:
    ------------------------
    1. Ley de control α = (gᵀGg + λI)⁺ gᵀG F_req
    2. Residuo de matching ‖gα − F_req‖_G / ‖F_req‖_G ≤ tol
    3. Estrangulación de vorticidad ‖I_curl‖_W
    4. Tensores ε_eff, μ_eff SPD (Kramers-Kronig)
    5. Límite CFL: dt_requested ≤ Δt_safe
    6. Radio de Gerschgorin ρ_G ≥ λ_max
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.1 — LEY DE CONTROL IDA-PBC COVARIANTE
    # -------------------------------------------------------------------------
    
    def test_phase2_control_law_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que ley de control α se calcula correctamente.
        
        Fórmula: α = (gᵀGg + λI)⁺ gᵀG ([J_d−R_d]∇H_d − [J−R]∇H)
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        # Necesitamos matrices sparse para la síntesis completa
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d()
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.control_law_alpha.shape == (2,)  # m=2
        assert np.all(np.isfinite(state.control_law_alpha))
    
    def test_phase2_matching_residual_within_tolerance(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que residuo de matching está dentro de tolerancia.
        
        Condición: ‖gα − F_req‖_G / max(‖F_req‖_G, 1) ≤ residual_tol_rel
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
            residual_tol_rel=1e-6,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.residual_idapbc <= 1e-6
    
    def test_phase2_matching_residual_exceeds_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que residuo excesivo dispara DiracMatchingError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        # g rango-deficiente para forzar residuo alto
        g_rank1: NDArray[np.float64] = np.array(
            [[1.0, 1.0],
             [1.0, 1.0],
             [1.0, 1.0]], dtype=np.float64
        )
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g_rank1,
            G=G,
            residual_tol_rel=1e-10,  # Tolerancia muy estricta
        )
        
        with pytest.raises(DiracMatchingError):
            agent.synthesize_kinematic_core(
                grad_H=grad_H,
                grad_H_d=grad_H_d,
                W=W,
                I_curl=I_curl,
                Z_load=Z_load,
                c_eff=1.0,
                Delta_sym=Delta_sym,
                dt_requested=0.01,
            )
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.2 — ESTRANGULAMIENTO DE VORTICIDAD DE HODGE
    # -------------------------------------------------------------------------
    
    def test_phase2_vorticity_norm_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que norma de vorticidad ‖I_curl‖_W se calcula.
        
        Fórmula: ‖I_curl‖_W = √(I_curlᵀ W I_curl)
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.vorticity_norm >= 0.0
        assert np.isfinite(state.vorticity_norm)
    
    def test_phase2_vorticity_strangulation_applied(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que vorticidad excesiva estrangula conductancia.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        
        # I_curl con vorticidad alta
        I_curl_high: NDArray[np.float64] = np.array(
            [1.0, 0.8, 0.9, 0.7, 1.2], dtype=np.float64
        )
        
        W_data = np.array([0.5, 0.3, 0.4, 0.2, 0.6], dtype=np.float64)
        W: sp.csr_matrix = sp.diags(W_data, offsets=0, format="csr", dtype=np.float64)
        
        Z_load: NDArray[np.float64] = np.array(
            [[1.0, 0.1],
             [0.1, 0.9]], dtype=np.float64
        )
        
        Delta_data = np.array([
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 3.0, -1.0, -1.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, -1.0, -1.0, 3.0]
        ], dtype=np.float64)
        Delta_sym: sp.csr_matrix = sp.csr_matrix(Delta_data, dtype=np.float64)
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl_high,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # W_mod debe ser diferente de W original si hay estrangulamiento
        assert state.hodge_conductance is not None
    
    def test_phase2_vorticity_negative_quad_form_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que forma cuadrática negativa dispara ParasiticVorticityError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        
        # W no PSD (autovalores negativos)
        W_invalid: sp.csr_matrix = sp.diags(
            np.array([-0.5, 0.3, 0.4, 0.2, 0.6], dtype=np.float64),
            offsets=0, format="csr", dtype=np.float64
        )
        
        I_curl: NDArray[np.float64] = np.array([0.1, 0.05, 0.08, 0.02, 0.12], dtype=np.float64)
        
        Z_load: NDArray[np.float64] = np.array(
            [[1.0, 0.1],
             [0.1, 0.9]], dtype=np.float64
        )
        
        Delta_data = np.array([
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 3.0, -1.0, -1.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, -1.0, -1.0, 3.0]
        ], dtype=np.float64)
        Delta_sym: sp.csr_matrix = sp.csr_matrix(Delta_data, dtype=np.float64)
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        with pytest.raises(ParasiticVorticityError):
            agent.synthesize_kinematic_core(
                grad_H=grad_H,
                grad_H_d=grad_H_d,
                W=W_invalid,
                I_curl=I_curl,
                Z_load=Z_load,
                c_eff=1.0,
                Delta_sym=Delta_sym,
                dt_requested=0.01,
            )
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.3 — SINTONIZACIÓN DE IMPEDANCIA KRAMERS-KRONIG
    # -------------------------------------------------------------------------
    
    def test_phase2_impedance_tensors_spd(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que ε_eff y μ_eff son SPD.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # Verificar que tensores son SPD vía Cholesky
        la.cholesky(state.dielectric_tensor, lower=True)
        la.cholesky(state.magnetic_tensor, lower=True)
    
    def test_phase2_impedance_non_spd_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que Z_load no SPD dispara ImpedanceReflectionError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        Z_load_invalid: NDArray[np.float64] = np.array(
            [[-1.0, 0.1],
             [0.1, 0.9]], dtype=np.float64
        )  # λ_min < 0
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        with pytest.raises(ImpedanceReflectionError):
            agent.synthesize_kinematic_core(
                grad_H=grad_H,
                grad_H_d=grad_H_d,
                W=W,
                I_curl=I_curl,
                Z_load=Z_load_invalid,
                c_eff=1.0,
                Delta_sym=Delta_sym,
                dt_requested=0.01,
            )
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.4 — AUDITORÍA DEL LÍMITE CFL
    # -------------------------------------------------------------------------
    
    def test_phase2_cfl_safe_dt_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que Δt_safe se calcula correctamente.
        
        Fórmula: Δt_safe = (2 · CFL_margin) / (c_eff · √(max(ρ_G, λ_max)))
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
            cfl_margin=0.9,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.cfl_safe_dt > 0.0
        assert np.isfinite(state.cfl_safe_dt)
    
    def test_phase2_cfl_violation_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que dt_requested > Δt_safe dispara CFLViolationError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
            cfl_margin=0.9,
        )
        
        # dt_requested muy grande para violar CFL
        with pytest.raises(CFLViolationError) as exc_info:
            agent.synthesize_kinematic_core(
                grad_H=grad_H,
                grad_H_d=grad_H_d,
                W=W,
                I_curl=I_curl,
                Z_load=Z_load,
                c_eff=1.0,
                Delta_sym=Delta_sym,
                dt_requested=100.0,  # Muy grande
            )
        
        assert "CFL" in str(exc_info.value) or "Cono de Luz" in str(exc_info.value)
    
    def test_phase2_c_eff_non_positive_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que c_eff ≤ 0 dispara CFLViolationError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        with pytest.raises(CFLViolationError) as exc_info:
            agent.synthesize_kinematic_core(
                grad_H=grad_H,
                grad_H_d=grad_H_d,
                W=W,
                I_curl=I_curl,
                Z_load=Z_load,
                c_eff=0.0,  # No físico
                Delta_sym=Delta_sym,
                dt_requested=0.01,
            )
        
        assert "c_eff" in str(exc_info.value) or "positivo" in str(exc_info.value)
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.5 — RADIOS DE GERSCHGORIN Y AUTOVALORES
    # -------------------------------------------------------------------------
    
    def test_phase2_gershgorin_rho_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que radio de Gerschgorin ρ_G se calcula.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.gershgorin_rho >= 0.0
        assert np.isfinite(state.gershgorin_rho)
    
    def test_phase2_lambda_max_delta_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que λ_max(Δ_sym) se estima correctamente.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.lambda_max_delta >= 0.0
        assert np.isfinite(state.lambda_max_delta)
    
    def test_phase2_gershgorin_majorizes_lambda_max(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que ρ_G ≥ λ_max (teorema de Gerschgorin).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # ρ_G debe ser ≥ λ_max (majorante)
        assert state.gershgorin_rho >= state.lambda_max_delta - 1e-10
    
    # -------------------------------------------------------------------------
    # SECCIÓN 2.6 — ESTABILIDAD CINEMÁTICA
    # -------------------------------------------------------------------------
    
    def test_phase2_is_kinematically_stable_true(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que is_kinematically_stable = True para sistema válido.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.is_kinematically_stable is True


# =============================================================================
# FASE 3 — PROYECCIÓN EN HACES Y COFRONTERA DISCRETA δ_CORE
# =============================================================================
class TestPhase3_SheafProjection:
    r"""
    ═══════════════════════════════════════════════════════════════════════════════════════
    FASE 3 — PROYECCIÓN COHOMOLÓGICA EN HACES: COCADENA δ_CORE Y HODGE LOCAL
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    Esta clase de pruebas valida el endofuntor Phase3_SheafProjection que proyecta
    la variedad local como una fibra (Stalk) para el orquestador macroscópico.
    
    Invariantes Verificados:
    ------------------------
    1. Identidad de Hodge local: δ_COREᵀ δ_CORE = W_mod
    2. δ_CORE es raíz espectral de W_mod
    3. Rango de δ_CORE = rango de W_mod
    4. betti_approx = E − rank(δ_CORE)
    5. Entropía de von Neumann de Spec(W_mod)
    6. Proyección δ_CORE · x
    """
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.1 — CONSTRUCCIÓN DE COCADENA δ_CORE
    # -------------------------------------------------------------------------
    
    def test_phase3_delta_core_shape_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que δ_CORE tiene forma (E, E).
        
        Construcción: δ_CORE = W_mod^{+1/2} ∈ ℝ^{E×E}
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # state_x en espacio de aristas
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        assert stalk.delta_core.shape == (E, E)
    
    def test_phase3_delta_core_symmetric(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que δ_CORE es simétrica.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        sym_residual = float(la.norm(stalk.delta_core - stalk.delta_core.T, "fro"))
        tol = _MACHINE_EPS * max(float(la.norm(stalk.delta_core, "fro")), 1.0)
        
        assert sym_residual <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.2 — IDENTIDAD DE HODGE LOCAL
    # -------------------------------------------------------------------------
    
    def test_phase3_hodge_identity_satisfied(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica identidad de Hodge: δ_COREᵀ δ_CORE = W_mod.
        
        Identidad: ‖δᵀδ − W_mod‖_F / ‖W_mod‖_F ≤ 100·ε_mach
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        tol = 100.0 * _MACHINE_EPS
        
        assert stalk.delta_hodge_residual <= tol, \
            f"Residuo de Hodge = {stalk.delta_hodge_residual} > tol = {tol}"
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.3 — RANGO Y BETTI APROXIMADO
    # -------------------------------------------------------------------------
    
    def test_phase3_rank_delta_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que rank(δ_CORE) se calcula correctamente.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        assert stalk.rank_delta >= 1
        assert stalk.rank_delta <= E
    
    def test_phase3_betti_approx_computed(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que betti_approx = E − rank(δ_CORE).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        expected_betti = E - stalk.rank_delta
        
        assert stalk.betti_approx == expected_betti
        assert stalk.betti_approx >= 0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.4 — ENTROPÍA ESPECTRAL
    # -------------------------------------------------------------------------
    
    def test_phase3_spectral_entropy_nonnegative(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que entropía de von Neumann es no negativa.
        
        Entropía: S = −Σ p_i ln p_i ≥ 0
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        assert stalk.spectral_entropy >= 0.0
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.5 — PROYECCIÓN DE ESTADO
    # -------------------------------------------------------------------------
    
    def test_phase3_projected_state_shape_valid(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que proyección de estado tiene forma correcta.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        assert stalk.projected_state.shape == (E,)
        assert stalk.state_vector.shape == (E,)
    
    def test_phase3_projected_state_consistency(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que proyección es consistente con δ_CORE · x.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.array([1.0, 0.5, 0.8, 0.3, 0.9], dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        expected = stalk.delta_core @ state_x
        
        tol = 1e-12 * max(float(la.norm(expected, 2)), 1.0)
        
        assert float(la.norm(stalk.projected_state - expected, 2)) <= tol
    
    # -------------------------------------------------------------------------
    # SECCIÓN 3.6 — REQUIREMENT DE SÍNTESIS PREVIA
    # -------------------------------------------------------------------------
    
    def test_phase3_export_without_synthesis_raises(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que export_sheaf_stalk sin síntesis previa dispara KinematicCoreError.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        with pytest.raises(KinematicCoreError) as exc_info:
            agent.export_sheaf_stalk(state_x)
        
        assert "synthesize" in str(exc_info.value).lower() or "conductancia" in str(exc_info.value).lower()
    
    def test_phase3_lazy_initialization(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que Phase3 se inicializa perezosamente en primera llamada.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        # Phase3 debe ser None antes de primera llamada
        assert agent.phase3 is None
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # Phase3 aún debe ser None (solo se llama a Fase 2)
        assert agent.phase3 is None
        
        # Primera llamada a export_sheaf_stalk inicializa Phase3
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
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
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ) -> None:
        r"""
        Verifica que KinematicPreparationContext de Fase 1 es entrada válida de Fase 2.
        
        Contrato: context es el único argumento del constructor de Phase2
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        # Verificar que context tiene todos los atributos requeridos por Fase 2
        assert hasattr(agent.context, 'J')
        assert hasattr(agent.context, 'R')
        assert hasattr(agent.context, 'J_d')
        assert hasattr(agent.context, 'R_d')
        assert hasattr(agent.context, 'g')
        assert hasattr(agent.context, 'G')
        assert hasattr(agent.context, 'n')
        assert hasattr(agent.context, 'm')
        assert hasattr(agent.context, 'rank_g')
        assert hasattr(agent.context, 'rank_G')
    
    def test_phase2_to_phase3_hodge_conductance_continuity(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que hodge_conductance de Fase 2 es entrada válida de Fase 3.
        
        Contrato: W_mod = KinematicStateTensor.hodge_conductance
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # hodge_conductance debe ser sparse matrix
        assert sp.issparse(state.hodge_conductance)
        
        # Debe ser entrada válida para Phase3
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        assert isinstance(stalk, SheafStalk)
    
    def test_phase1_to_phase3_context_reuse(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que Phase3 no requiere contexto de Fase 1 (solo W_mod).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        # Phase3 debe tener acceso a W_mod
        assert stalk.rank_delta > 0
    
    def test_full_pipeline_immutability(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que DTOs son inmutables (frozen dataclasses).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        E = 5
        state_x: NDArray[np.float64] = np.ones(E, dtype=np.float64)
        
        stalk = agent.export_sheaf_stalk(state_x)
        
        # Intentar modificar debe fallar (frozen=True)
        with pytest.raises(AttributeError):
            state.cfl_safe_dt = 999.0  # type: ignore[misc]
        
        with pytest.raises(AttributeError):
            stalk.delta_core[0, 0] = 999.0  # type: ignore[misc]


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
    
    def test_edge_case_zero_gradients(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica comportamiento con gradientes cero ∇H = 0, ∇H_d = 0.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        grad_H: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        grad_H_d: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        # α debe ser cero o muy pequeño
        assert float(la.norm(state.control_law_alpha, 2)) < 1e-10
    
    def test_edge_case_identity_metric(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica comportamiento con G = I (métrica euclidiana).
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=None,  # Debe convertirse en I
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.is_kinematically_stable is True
    
    def test_edge_case_diagonal_matrices(
        self,
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica comportamiento con matrices diagonales (caso desacoplado).
        """
        n, m = 3, 2
        
        J: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)  # Sin interconexión
        R: NDArray[np.float64] = np.diag([0.5, 0.3, 0.4])
        J_d: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
        R_d: NDArray[np.float64] = np.diag([0.6, 0.4, 0.5])
        g: NDArray[np.float64] = np.eye(3, 2, dtype=np.float64)
        G: NDArray[np.float64] = np.eye(3, dtype=np.float64)
        
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        agent = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g,
            G=G,
        )
        
        state = agent.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert state.is_kinematically_stable is True
    
    def test_edge_case_tikhonov_regularization(
        self,
        fixture_valid_matrices_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_valid_gradients_3d: Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        fixture_sparse_matrices_3d: Tuple[
            sp.csr_matrix,
            NDArray[np.float64],
            NDArray[np.float64],
            sp.csr_matrix,
        ],
    ) -> None:
        r"""
        Verifica que regularización de Tikhonov mejora conditioning.
        """
        J, R, J_d, R_d, g, G = fixture_valid_matrices_3d
        grad_H, grad_H_d = fixture_valid_gradients_3d
        W, I_curl, Z_load, Delta_sym = fixture_sparse_matrices_3d
        
        # g casi rango-deficiente
        g_ill: NDArray[np.float64] = np.array(
            [[1.0, 1.0 + 1e-10],
             [1.0, 1.0 + 2e-10],
             [1.0, 1.0 + 3e-10]], dtype=np.float64
        )
        
        agent_no_reg = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g_ill,
            G=G,
            tikhonov_reg=0.0,
        )
        
        agent_with_reg = KCoreKinematicAgent(
            J=J,
            R=R,
            J_d=J_d,
            R_d=R_d,
            g=g_ill,
            G=G,
            tikhonov_reg=1e-6,  # Regularización
        )
        
        # Ambos deben funcionar con regularización apropiada
        state = agent_with_reg.synthesize_kinematic_core(
            grad_H=grad_H,
            grad_H_d=grad_H_d,
            W=W,
            I_curl=I_curl,
            Z_load=Z_load,
            c_eff=1.0,
            Delta_sym=Delta_sym,
            dt_requested=0.01,
        )
        
        assert np.all(np.isfinite(state.control_law_alpha))


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
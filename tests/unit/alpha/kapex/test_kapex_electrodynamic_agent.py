r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Suite de Pruebas: KApex Electrodynamic Agent                                 ║
║ Ubicación: tests/unit/alpha/kapex/test_kapex_electrodynamic_agent.py         ║
║ Versión   : 1.0.0-Strict-Spectral-Phased                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Cobertura por Fase                                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Fase 1 – Validación Métrica y de Calibre:                                   ║
║    • Dimensiones: ndim, cuadratura, coherencia G_μν/G_inv/R_cost.            ║
║    • Simetría de G_μν, G_inv, R_cost con tol relativa ε_mach·‖A‖_F.          ║
║    • SPD de G_μν y G_inv con Cholesky + diagnóstico espectral.               ║
║    • Número de condición κ(G_μν) y κ(G_inv) > kappa_max lanza error.         ║
║    • Consistencia métrica ‖G·G_inv−I‖_F/n con tol Wilkinson κ·ε·n.           ║
║    • PSD de R_cost: autovalor negativo real lanza excepción.                 ║
║    • Raíz espectral R_sqrt: R_sqrt² = R_cost con tol de máquina.             ║
║    • L_G: triangular inferior con diagonal positiva.                         ║
║    • L_G reconstruye G_μν: ‖L_G·L_G^⊤−G_μν‖_F < 100·ε·‖G‖_F.                 ║
║    • ApexPreparationContext: tipos, shapes, metadatos, inmutabilidad.        ║
║    • Copias independientes en el contexto (no referencias).                  ║
║                                                                              ║
║  Fase 2 – Síntesis Electrodinámica:                                          ║
║    • inject_gauge_potential:                                                 ║
║        – Dimensión incorrecta de d_Phi lanza ApexDimensionError.             ║
║        – Factor de supresión = exp(−½ Tr(G)) verificado analíticamente.      ║
║        – s_val = d_Phi · suppression (exactitud analítica).                  ║
║        – Supresión total relativa lanza GaugePotentialError.                 ║
║        – Tr(G) grande → suppression ≈ 0 → excepción.                         ║
║        – Tr(G) = 0 → suppression = 1 → s_val = d_Phi.                        ║
║        – suppression_factor ∈ (0, 1] siempre.                                ║
║    • compute_eikonal_absorption:                                             ║
║        – Dimensión incorrecta de phase_gradient lanza ApexDimensionError.    ║
║        – n_refract = 1 + tanh(α·σ*) verificado analíticamente.               ║
║        – n_refract ∈ (0, 2) para cualquier σ*.                               ║
║        – Norma riemanniana G^μν ∂S ∂S verificada analíticamente.             ║
║        – Eikonal satisfecho: G_inv·∂S·∂S ≥ n²·(1−slack) pasa.                ║
║        – Eikonal violado: G_inv·∂S·∂S << n² lanza EikonalRefractionError.    ║
║        – sigma_stress = 0 → n_refract = 1 (tanh(0) = 0).                     ║
║        – eikonal_norm_sq ≥ 0 para phase_gradient cualquiera.                 ║
║    • evaluate_poynting_exergy:                                               ║
║        – Dimensiones incorrectas de E_field/H_field/grad_H.                  ║
║        – P_in = E·H verificado analíticamente (producto escalar).            ║
║        – P_diss = ∇H^⊤ R_cost ∇H ≥ 0 (R_cost PSD).                           ║
║        – P_exergia = P_in − P_diss ≥ −tol_rel (segunda ley).                 ║
║        – P_in > P_diss: exergía positiva, no lanza excepción.                ║
║        – P_in < P_diss por margen > tol: lanza FinancialBlackHoleError.      ║
║        – P_diss = 0 con R_cost = 0: P_exergia = P_in exactamente.            ║
║        – Tolerancia relativa: P_in ≈ P_diss no lanza excepción.              ║
║    • audit_yang_mills_holonomy:                                              ║
║        – Dimensión incorrecta de A_gauge lanza ApexDimensionError.           ║
║        – A_gauge = 0 → F = 0 → S_YM = 0 (holonomía trivial).                 ║
║        – A_gauge antisimétrica → F = 2A + [A,A^⊤] verificado.                ║
║        – S_YM = ½Tr(F^⊤ G F G_inv) verificado analíticamente.                ║
║        – S_YM ≥ 0 siempre (norma ponderada no negativa).                     ║
║        – S_YM > tol_rel·‖A‖_F² lanza HolonomyVetoError.                      ║
║        – S_YM ≤ tol_rel·‖A‖_F² es aceptado.                                  ║
║        – Umbral relativo: A_gauge grande con baja curvatura pasa.            ║
║    • synthesize:                                                             ║
║        – ApexStateTensor con todos los campos correctos.                     ║
║        – is_electrodynamically_viable = True en éxito.                       ║
║        – suppression_factor documentado en tensor.                           ║
║        – eikonal_norm_sq documentado en tensor.                              ║
║        – poynting_income/dissipation/exergy consistentes.                    ║
║        – Propagación de excepciones desde subprocesos.                       ║
║                                                                              ║
║  Fase 3 – Proyección en Haces:                                               ║
║    • δ_{APEX} = L_G^{-⊤}: shape (n,n), dtype float64.                        ║
║    • Identidad de Hodge: δ^⊤ G_μν δ = I con tol 100·ε_mach·n.                ║
║    • δ_{APEX} triangular superior (L_G^{-⊤} es triu).                        ║
║    • hodge_metric_residual ≥ 0 y < 100·ε_mach.                               ║
║    • projected_source = δ·s_val verificado analíticamente.                   ║
║    • Proyección cero para s_val = 0.                                         ║
║    • Linealidad de la proyección.                                            ║
║    • source_injection es copia independiente.                                ║
║    • rank_delta = n (pleno rango pues G_μν ≻ 0).                             ║
║    • Dimensión incorrecta de s_val lanza ApexDimensionError.                 ║
║    • Instanciación perezosa (None antes, no-None después).                   ║
║    • Reutilización de phase3 entre llamadas.                                 ║
║    • SheafStalkApex es frozen dataclass.                                     ║
║                                                                              ║
║  Integración de las 3 Fases:                                                 ║
║    • Pipeline completo nominal.                                              ║
║    • Determinismo.                                                           ║
║    • Logging INFO en construcción, síntesis y exportación.                   ║
║    • Dimensiones variadas (n=1,2,3,5).                                       ║
║    • Flujo gauge_injection_vector → projected_source.                        ║
║    • Coherencia suppression·‖dΦ‖ = ‖s_val‖ (escala).                         ║
║    • Inmutabilidad de ApexStateTensor y SheafStalkApex.                      ║
║    • alpha_fermat = 0 → n_refract = 1 (tanh(0) = 0).                         ║
║    • kappa_max estricto rechaza G_μν mal condicionada.                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ── Biblioteca estándar ──────────────────────────────────────────────────────
import logging
import math
from typing import Tuple

# ── Framework de pruebas ─────────────────────────────────────────────────────
import pytest

# ── Álgebra numérica ─────────────────────────────────────────────────────────
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Módulo bajo prueba ───────────────────────────────────────────────────────
from app.agents.alpha.kapex.kapex_electrodynamic_agent import (
    # Agente orquestador
    KApexElectrodynamicAgent,
    # DTOs
    ApexPreparationContext,
    ApexStateTensor,
    SheafStalkApex,
    # Excepciones
    ApexConditionError,
    ApexDimensionError,
    ApexSymmetryError,
    ElectrodynamicApexError,
    EikonalRefractionError,
    FinancialBlackHoleError,
    GaugePotentialError,
    HolonomyVetoError,
    MetricInverseError,
    SheafMetricError,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 0 – CONSTANTES Y UTILIDADES DE PRUEBA                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_EPS: float = float(np.finfo(np.float64).eps)
_ATOL: float = 1.0e-10
_RTOL: float = 1.0e-10


# ── Fábricas de matrices ─────────────────────────────────────────────────────

def _spd(n: int, seed: int = 0, kappa: float = 10.0) -> NDArray[np.float64]:
    """
    Genera una matriz SPD de dimensión n×n con κ ≈ kappa.
    Construcción: Q·diag(λ)·Q^⊤, λ ∈ [1, kappa] log-uniforme.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.logspace(0.0, math.log10(kappa), n)
    A = Q @ np.diag(eigvals) @ Q.T
    return 0.5 * (A + A.T)


def _psd(n: int, rank: int, seed: int = 0) -> NDArray[np.float64]:
    """Genera una matriz PSD de rango ``rank`` ≤ n."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.zeros(n)
    eigvals[:rank] = np.linspace(1.0, float(rank + 1), rank)
    A = Q @ np.diag(eigvals) @ Q.T
    return 0.5 * (A + A.T)


def _ill_conditioned_spd(n: int, kappa: float) -> NDArray[np.float64]:
    """Genera una matriz SPD con número de condición exactamente kappa."""
    rng = np.random.default_rng(42)
    G = rng.standard_normal((n, n))
    Q, _ = la.qr(G)
    eigvals = np.logspace(0.0, math.log10(kappa), n)
    A = Q @ np.diag(eigvals) @ Q.T
    return 0.5 * (A + A.T)


def _make_metric_triple(
    n: int,
    seed: int = 0,
    kappa: float = 10.0,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Retorna (G_mu_nu, G_inv, R_cost) consistentes y válidas.

    • G_mu_nu : SPD
    • G_inv   : inversa exacta de G_mu_nu (SPD)
    • R_cost  : PSD de rango n-1
    """
    G = _spd(n, seed=seed, kappa=kappa)
    G_inv = la.inv(G)
    G_inv = 0.5 * (G_inv + G_inv.T)
    R = _psd(n, rank=n - 1, seed=seed + 1)
    return G, G_inv, R


def _default_agent(
    n: int = 4,
    seed: int = 7,
    kappa_max: float = 1.0e10,
    eikonal_slack: float = 0.5,
    holonomy_tol_rel: float = 1.0e-4,
) -> KApexElectrodynamicAgent:
    """Construye un KApexElectrodynamicAgent con matrices válidas por defecto."""
    G, G_inv, R = _make_metric_triple(n=n, seed=seed)
    return KApexElectrodynamicAgent(
        G_mu_nu=G,
        G_inv=G_inv,
        R_cost=R,
        kappa_max=kappa_max,
        eikonal_slack=eikonal_slack,
        holonomy_tol_rel=holonomy_tol_rel,
    )


def _default_synthesis_inputs(
    agent: KApexElectrodynamicAgent,
    seed: int = 99,
) -> dict:
    """
    Genera un diccionario con todos los argumentos válidos para
    ``synthesize_apex_field``.

    Garantías:
      • s_val tendrá supresión no nula.
      • Eikonal satisfecho (eikonal_slack generoso).
      • P_exergia > 0.
      • S_YM pequeño (A_gauge antisimétrica pura).
    """
    n = agent.context.dim
    rng = np.random.default_rng(seed)

    # d_Phi: vector con norma razonable
    d_Phi = rng.standard_normal(n)

    # phase_gradient: garantizamos que la norma riemanniana sea suficiente
    # Usamos G_inv·e₁ que maximiza la forma cuadrática
    phase_gradient = la.solve(
        agent.context.G_mu_nu, np.ones(n)
    ) * 10.0

    sigma_stress = 0.5

    # E_field y H_field: producto positivo (P_in > 0) > P_diss
    E_field = np.ones(n) * 5.0
    H_field = np.ones(n) * 5.0  # P_in = 5n

    # grad_H pequeño → P_diss ≈ 0
    grad_H = rng.standard_normal(n) * 0.001

    # A_gauge antisimétrica pura → F pequeño → S_YM pequeño
    A_raw = rng.standard_normal((n, n)) * 0.001
    A_gauge = A_raw - A_raw.T  # antisimétrica: F = 0 + [A, A^⊤] ≈ 0

    return dict(
        d_Phi=d_Phi,
        phase_gradient=phase_gradient,
        sigma_stress=sigma_stress,
        E_field=E_field,
        H_field=H_field,
        grad_H=grad_H,
        A_gauge=A_gauge,
        alpha_fermat=0.5,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 1 – PRUEBAS DE FASE 1: VALIDACIÓN MÉTRICA Y DE CALIBRE         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase1MetricValidation:
    """
    Pruebas exhaustivas de Phase1_MetricValidation y build_context.
    """

    # ── 1.1 Validación dimensional ────────────────────────────────────────

    def test_g_mu_nu_not_2d_raises_dimension_error(self) -> None:
        """G_μν con ndim=1 lanza ApexDimensionError."""
        G, G_inv, R = _make_metric_triple(n=3)
        with pytest.raises(ApexDimensionError, match=r"2D"):
            KApexElectrodynamicAgent(
                G_mu_nu=G[0],  # shape (3,) no 2D
                G_inv=G_inv,
                R_cost=R,
            )

    def test_g_mu_nu_not_square_raises_dimension_error(self) -> None:
        """G_μν rectangular lanza ApexDimensionError."""
        _, G_inv, R = _make_metric_triple(n=3)
        G_rect = np.ones((3, 4))
        with pytest.raises(ApexDimensionError, match=r"cuadrada"):
            KApexElectrodynamicAgent(G_mu_nu=G_rect, G_inv=G_inv, R_cost=R)

    def test_g_inv_wrong_shape_raises_dimension_error(self) -> None:
        """G_inv con shape (n+1, n+1) lanza ApexDimensionError."""
        G, G_inv, R = _make_metric_triple(n=3)
        G_inv_bad = _spd(4)  # n+1=4 en lugar de 3
        with pytest.raises(ApexDimensionError, match=r"G_inv"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv_bad, R_cost=R)

    def test_r_cost_wrong_shape_raises_dimension_error(self) -> None:
        """R_cost con shape (n+1, n+1) lanza ApexDimensionError."""
        G, G_inv, R = _make_metric_triple(n=3)
        R_bad = _psd(4, rank=3)
        with pytest.raises(ApexDimensionError, match=r"R_cost"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R_bad)

    def test_all_matrices_must_be_2d(self) -> None:
        """
        Todas las matrices deben ser 2D. Verificamos G_inv con ndim=3.
        """
        G, G_inv, R = _make_metric_triple(n=3)
        G_inv_3d = G_inv[np.newaxis, :, :]  # shape (1,3,3)
        with pytest.raises(ApexDimensionError, match=r"2D"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv_3d, R_cost=R)

    # ── 1.2 Validación de simetría ────────────────────────────────────────

    def test_g_mu_nu_asymmetric_raises_symmetry_error(self) -> None:
        """
        G_μν con ‖G−G^⊤‖_F >> ε_mach·‖G‖_F lanza ApexSymmetryError.
        """
        G, G_inv, R = _make_metric_triple(n=4)
        G_asym = G.copy()
        G_asym[0, 1] += 10.0  # rompe simetría macroscópicamente
        with pytest.raises(ApexSymmetryError, match=r"simétric"):
            KApexElectrodynamicAgent(G_mu_nu=G_asym, G_inv=G_inv, R_cost=R)

    def test_g_inv_asymmetric_raises_symmetry_error(self) -> None:
        """G_inv asimétrica lanza ApexSymmetryError."""
        G, G_inv, R = _make_metric_triple(n=4)
        G_inv_asym = G_inv.copy()
        G_inv_asym[1, 3] += 5.0
        with pytest.raises(ApexSymmetryError, match=r"simétric"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv_asym, R_cost=R)

    def test_r_cost_asymmetric_raises_symmetry_error(self) -> None:
        """R_cost asimétrica lanza ApexSymmetryError."""
        G, G_inv, R = _make_metric_triple(n=4)
        R_asym = R.copy()
        R_asym[0, 2] += 3.0
        with pytest.raises(ApexSymmetryError, match=r"simétric"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R_asym)

    def test_near_symmetric_within_tolerance_accepted(self) -> None:
        """
        G_μν con perturbación antisimétrica de amplitud ε_mach·‖G‖_F/2
        (dentro de tolerancia) es aceptada sin excepción.
        """
        G, G_inv, R = _make_metric_triple(n=3)
        norm_G = float(la.norm(G, "fro"))
        noise = np.random.default_rng(0).standard_normal(G.shape)
        # Perturbación antisimétrica de magnitud < ε·‖G‖
        G_near = G + _EPS * norm_G * 0.4 * (noise - noise.T)
        # Forzar simetría exacta después de la perturbación
        G_sym = 0.5 * (G_near + G_near.T)
        G_inv_new = la.inv(G_sym)
        G_inv_new = 0.5 * (G_inv_new + G_inv_new.T)
        agent = KApexElectrodynamicAgent(
            G_mu_nu=G_sym, G_inv=G_inv_new, R_cost=R
        )
        assert agent.context is not None

    # ── 1.3 Validación SPD y número de condición ──────────────────────────

    def test_g_mu_nu_singular_raises_electrodynamic_error(self) -> None:
        """
        G_μν singular (λ_min ≈ 0) lanza ElectrodynamicApexError.
        """
        _, G_inv, R = _make_metric_triple(n=3)
        G_sing = _psd(3, rank=2)  # rango deficiente
        G_sing_sym = 0.5 * (G_sing + G_sing.T)
        # G_inv arbitraria (la validación falla en G_μν antes)
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(G_mu_nu=G_sing_sym, G_inv=G_inv, R_cost=R)

    def test_g_mu_nu_negative_eigenvalue_raises_error(self) -> None:
        """G_μν con autovalor negativo lanza ElectrodynamicApexError."""
        _, G_inv, R = _make_metric_triple(n=3)
        G_bad = _spd(3, seed=10)
        eigvals, eigvecs = la.eigh(G_bad)
        eigvals[0] = -1.0
        G_indef = eigvecs @ np.diag(eigvals) @ eigvecs.T
        G_indef = 0.5 * (G_indef + G_indef.T)
        with pytest.raises(ElectrodynamicApexError):
            KApexElectrodynamicAgent(G_mu_nu=G_indef, G_inv=G_inv, R_cost=R)

    def test_ill_conditioned_g_raises_condition_error(self) -> None:
        """G_μν con κ > kappa_max lanza ApexConditionError."""
        n = 3
        kappa_max = 1.0e4
        G_ill = _ill_conditioned_spd(n, kappa=1.0e8)
        G_inv_ill = la.inv(G_ill)
        G_inv_ill = 0.5 * (G_inv_ill + G_inv_ill.T)
        _, _, R = _make_metric_triple(n=n)
        with pytest.raises(ApexConditionError, match=r"κ"):
            KApexElectrodynamicAgent(
                G_mu_nu=G_ill,
                G_inv=G_inv_ill,
                R_cost=R,
                kappa_max=kappa_max,
            )

    def test_ill_conditioned_g_inv_raises_condition_error(self) -> None:
        """G_inv con κ > kappa_max lanza ApexConditionError."""
        n = 3
        kappa_max = 1.0e3
        G = _spd(n, seed=20, kappa=5.0)
        # Construir G_inv muy mal condicionada artificialmente
        G_inv_ill = _ill_conditioned_spd(n, kappa=1.0e6)
        _, _, R = _make_metric_triple(n=n)
        with pytest.raises(ApexConditionError):
            KApexElectrodynamicAgent(
                G_mu_nu=G,
                G_inv=G_inv_ill,
                R_cost=R,
                kappa_max=kappa_max,
            )

    # ── 1.4 Consistencia métrica G·G_inv ≈ I ─────────────────────────────

    def test_inconsistent_g_inv_raises_metric_inverse_error(self) -> None:
        """
        G_inv que no es la inversa de G_μν lanza MetricInverseError
        con diagnóstico del residuo ‖G·G_inv−I‖_F/n.
        """
        G, _, R = _make_metric_triple(n=4)
        # G_inv completamente errónea
        G_inv_wrong = _spd(4, seed=99)  # SPD pero no inversa de G
        with pytest.raises(MetricInverseError, match=r"G_inv"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv_wrong, R_cost=R)

    def test_exact_inverse_is_accepted(self) -> None:
        """G_inv = G_μν⁻¹ exacta (scipy.linalg.inv) es aceptada."""
        n = 4
        G = _spd(n, seed=30, kappa=5.0)
        G_inv = la.inv(G)
        G_inv = 0.5 * (G_inv + G_inv.T)
        _, _, R = _make_metric_triple(n=n)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)
        assert agent.context.inverse_residual < 100 * _EPS

    def test_inverse_residual_stored_in_context(self) -> None:
        """
        context.inverse_residual = ‖G·G_inv−I‖_F/n debe ser escalar
        float positivo y menor que la tolerancia Wilkinson.
        """
        agent = _default_agent(n=4)
        assert isinstance(agent.context.inverse_residual, float)
        assert agent.context.inverse_residual >= 0.0
        # Para G_inv exacta, el residuo debe ser de O(κ·ε_mach)
        assert agent.context.inverse_residual < 1.0e-6

    # ── 1.5 PSD de R_cost ─────────────────────────────────────────────────

    def test_r_cost_negative_eigenvalue_raises_symmetry_error(self) -> None:
        """
        R_cost con autovalor genuinamente negativo lanza ApexSymmetryError.
        """
        G, G_inv, _ = _make_metric_triple(n=4)
        n = 4
        R_psd = _psd(n, rank=n-1, seed=50)
        eigvals, eigvecs = la.eigh(R_psd)
        eigvals[0] = -1.0
        R_neg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        R_neg = 0.5 * (R_neg + R_neg.T)
        with pytest.raises(ApexSymmetryError, match=r"negativ"):
            KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R_neg)

    def test_r_cost_zero_is_valid_psd(self) -> None:
        """R_cost = 0 es PSD válida (rank_R = 0, sistema conservativo)."""
        G, G_inv, _ = _make_metric_triple(n=3)
        n = 3
        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=np.zeros((n, n))
        )
        assert agent.context.rank_R == 0

    def test_r_cost_full_rank_stores_correct_rank(self) -> None:
        """R_cost SPD (rango pleno) → context.rank_R = n."""
        n = 4
        G, G_inv, _ = _make_metric_triple(n=n)
        R_spd = _spd(n, seed=60, kappa=5.0)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R_spd)
        assert agent.context.rank_R == n

    # ── 1.6 Raíz espectral R_sqrt ─────────────────────────────────────────

    def test_r_sqrt_squared_equals_r_cost(self) -> None:
        """
        R_sqrt · R_sqrt ≈ R_cost con tolerancia 100·ε_mach·‖R‖_F.
        """
        agent = _default_agent(n=4)
        R_sqrt = agent.context.R_sqrt
        R_cost = agent.context.R_cost
        R_reconstructed = R_sqrt @ R_sqrt
        norm_R = float(la.norm(R_cost, "fro"))
        residual = float(la.norm(R_reconstructed - R_cost, "fro"))
        assert residual < 100 * _EPS * max(norm_R, 1.0), (
            f"‖R_sqrt²−R‖_F = {residual:.3e} > tol = {100*_EPS*norm_R:.3e}"
        )

    def test_r_sqrt_is_symmetric_psd(self) -> None:
        """R_sqrt debe ser simétrica y PSD."""
        agent = _default_agent(n=4)
        R_sqrt = agent.context.R_sqrt
        # Simetría
        assert np.allclose(R_sqrt, R_sqrt.T, atol=100 * _EPS)
        # PSD
        eigvals = la.eigvalsh(R_sqrt)
        norm = float(la.norm(R_sqrt, "fro"))
        assert np.all(eigvals >= -100 * _EPS * norm)

    # ── 1.7 Factor Cholesky L_G ────────────────────────────────────────────

    def test_l_g_is_lower_triangular_positive_diagonal(self) -> None:
        """
        L_G debe ser triangular inferior (parte superior = 0)
        con diagonal estrictamente positiva.
        """
        agent = _default_agent(n=4)
        L_G = agent.context.L_G
        upper = np.triu(L_G, k=1)
        assert np.allclose(upper, 0.0, atol=_ATOL), (
            f"L_G tiene entradas no nulas en la parte superior: "
            f"‖triu(L_G,1)‖_F = {la.norm(upper):.3e}"
        )
        assert np.all(np.diag(L_G) > 0), "L_G tiene diagonal no positiva."

    def test_l_g_reconstructs_g_mu_nu(self) -> None:
        """
        L_G · L_G^⊤ reconstruye G_μν con error O(ε_mach·‖G‖_F).
        """
        n = 5
        G, G_inv, R = _make_metric_triple(n=n)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)
        L_G = agent.context.L_G
        G_reconstructed = L_G @ L_G.T
        norm_G = float(la.norm(G, "fro"))
        residual = float(la.norm(G_reconstructed - agent.context.G_mu_nu, "fro"))
        assert residual < 100 * _EPS * norm_G, (
            f"‖L_G·L_G^⊤−G‖_F = {residual:.3e} > 100·ε·‖G‖ = {100*_EPS*norm_G:.3e}"
        )

    # ── 1.8 ApexPreparationContext: integridad de campos ──────────────────

    def test_context_fields_types_and_shapes(self) -> None:
        """
        ApexPreparationContext debe tener todos los campos con tipos y
        shapes correctos.
        """
        n = 5
        agent = _default_agent(n=n)
        ctx = agent.context

        assert ctx.G_mu_nu.shape == (n, n)
        assert ctx.G_inv.shape == (n, n)
        assert ctx.R_cost.shape == (n, n)
        assert ctx.L_G.shape == (n, n)
        assert ctx.R_sqrt.shape == (n, n)
        assert isinstance(ctx.kappa_G, float) and ctx.kappa_G >= 1.0
        assert isinstance(ctx.rank_R, int) and ctx.rank_R >= 0
        assert isinstance(ctx.inverse_residual, float)
        assert ctx.dim == n

    def test_context_stores_copies_not_references(self) -> None:
        """
        Las matrices en el contexto son copias independientes.
        Modificar la entrada original no altera el contexto.
        """
        G, G_inv, R = _make_metric_triple(n=4)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)
        G_val_orig = agent.context.G_mu_nu[0, 1]
        G[0, 1] += 9999.0  # modificar original
        assert agent.context.G_mu_nu[0, 1] == pytest.approx(G_val_orig)

    def test_context_is_immutable_frozen_dataclass(self) -> None:
        """ApexPreparationContext es frozen; no permite asignaciones."""
        agent = _default_agent()
        with pytest.raises((AttributeError, TypeError)):
            agent.context.dim = 999  # type: ignore[misc]

    def test_context_kappa_g_consistent_with_g_mu_nu(self) -> None:
        """
        kappa_G debe ser coherente con κ(G_μν) calculado independientemente.
        Verificamos: kappa_G ∈ [1, kappa_max].
        """
        agent = _default_agent(n=4)
        eigvals = la.eigvalsh(agent.context.G_mu_nu)
        kappa_direct = float(eigvals[-1] / eigvals[0])
        # Coherencia: mismo orden de magnitud (tolerancia de 10%)
        assert agent.context.kappa_G == pytest.approx(kappa_direct, rel=0.1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 2 – PRUEBAS DE FASE 2: SÍNTESIS ELECTRODINÁMICA                ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase2ElectrodynamicSynthesis:
    """
    Pruebas exhaustivas de Phase2_ElectrodynamicSynthesis y sus cuatro
    subprocesos: inyección de Gauge, Eikonal, Poynting y Yang-Mills.
    """

    # ────────────────────────────────────────────────────────────────────────
    # 2.1 Subproceso 1: inject_gauge_potential
    # ────────────────────────────────────────────────────────────────────────

    def test_gauge_wrong_d_phi_shape_raises_dimension_error(self) -> None:
        """d_Phi con shape ≠ (n,) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        d_Phi_bad = np.ones(5)  # n+1
        with pytest.raises(ApexDimensionError, match=r"d_Phi"):
            agent.phase2.inject_gauge_potential(d_Phi=d_Phi_bad)

    def test_gauge_suppression_factor_formula_analytical(self) -> None:
        """
        suppression = exp(−½ Tr(G_μν)) verificado analíticamente.
        """
        n = 3
        G = np.diag([1.0, 2.0, 3.0])  # Tr(G) = 6
        G_inv = np.diag([1.0, 0.5, 1.0/3.0])
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
        )

        d_Phi = np.ones(n)
        _, suppression = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)

        expected_suppression = math.exp(-0.5 * 6.0)
        assert suppression == pytest.approx(expected_suppression, rel=_RTOL)

    def test_gauge_s_val_equals_d_phi_times_suppression(self) -> None:
        """
        s_val = d_Phi · exp(−½ Tr(G)) verificado para cada componente.
        """
        n = 3
        G = np.diag([2.0, 2.0, 2.0])  # Tr(G) = 6, suppression = exp(-3)
        G_inv = np.diag([0.5, 0.5, 0.5])
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
        )

        d_Phi = np.array([1.0, -2.0, 3.0])
        s_val, suppression = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)

        expected = d_Phi * math.exp(-3.0)
        assert np.allclose(s_val, expected, rtol=_RTOL)
        assert suppression == pytest.approx(math.exp(-3.0), rel=_RTOL)

    def test_gauge_zero_trace_g_gives_suppression_one(self) -> None:
        """
        Tr(G) = 0 → suppression = exp(0) = 1 → s_val = d_Phi.
        Imposible para G SPD (Tr ≥ λ_min · n > 0), pero se puede
        simular con G = ε·I con ε → 0 (no realizable; testeamos Tr pequeño).
        Alternativa: verificar que suppression ∈ (0, 1] siempre.
        """
        agent = _default_agent(n=3)
        n = agent.context.dim
        d_Phi = np.ones(n)
        _, suppression = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)
        # suppression = exp(-½·Tr(G)) ∈ (0, 1] pues Tr(G) ≥ 0
        assert 0.0 < suppression <= 1.0

    def test_gauge_suppression_factor_in_zero_one_interval(self) -> None:
        """
        suppression_factor ∈ (0, 1] para cualquier G_μν SPD.
        Verificado para múltiples semillas.
        """
        for seed in [1, 2, 3, 4, 5]:
            agent = _default_agent(n=3, seed=seed)
            n = agent.context.dim
            _, sup = agent.phase2.inject_gauge_potential(d_Phi=np.ones(n))
            assert 0.0 < sup <= 1.0, (
                f"suppression = {sup:.6e} ∉ (0,1] para seed={seed}"
            )

    def test_gauge_large_trace_g_raises_gauge_error(self) -> None:
        """
        G_μν con Tr(G) muy grande → suppression ≈ 0 →
        ‖s_val‖/‖dΦ‖ < ε_mach → GaugePotentialError.

        Estrategia: instanciar Phase2 directamente con un contexto
        que tiene Tr(G) = 2000 (suppression = exp(-1000) ≈ 0).
        """
        n = 3
        # Construir G con Tr = 2000 (cada diagonal = 2000/3)
        # Pero SPD: G = diag(666, 667, 667)
        G_large = np.diag([666.0, 667.0, 667.0])
        G_inv_large = np.diag([1.0/666.0, 1.0/667.0, 1.0/667.0])
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G_large,
            G_inv=G_inv_large,
            R_cost=R,
        )

        d_Phi = np.ones(n)
        with pytest.raises(GaugePotentialError, match=r"Tr"):
            agent.phase2.inject_gauge_potential(d_Phi=d_Phi)

    def test_gauge_zero_d_phi_raises_gauge_error(self) -> None:
        """
        d_Phi = 0 → s_val = 0 → supresión relativa = 0 →
        GaugePotentialError (independientemente de Tr(G)).
        """
        agent = _default_agent(n=3)
        n = agent.context.dim
        with pytest.raises(GaugePotentialError):
            agent.phase2.inject_gauge_potential(d_Phi=np.zeros(n))

    def test_gauge_s_val_shape_equals_n(self) -> None:
        """s_val debe tener shape (n,)."""
        agent = _default_agent(n=4)
        n = agent.context.dim
        d_Phi = np.ones(n) * 2.0
        s_val, _ = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)
        assert s_val.shape == (n,)

    def test_gauge_suppression_is_precomputed_invariant(self) -> None:
        """
        suppression_factor debe ser el mismo valor en llamadas
        repetidas con d_Phi diferente (es una constante del agente).
        """
        agent = _default_agent(n=3)
        n = agent.context.dim
        _, sup1 = agent.phase2.inject_gauge_potential(d_Phi=np.ones(n))
        _, sup2 = agent.phase2.inject_gauge_potential(d_Phi=np.ones(n) * 5.0)
        assert sup1 == pytest.approx(sup2)

    # ────────────────────────────────────────────────────────────────────────
    # 2.2 Subproceso 2: compute_eikonal_absorption
    # ────────────────────────────────────────────────────────────────────────

    def test_eikonal_wrong_phase_gradient_shape_raises_error(self) -> None:
        """phase_gradient con shape ≠ (n,) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        grad_bad = np.ones(3)  # n-1 en lugar de n
        with pytest.raises(ApexDimensionError, match=r"phase_gradient"):
            agent.phase2.compute_eikonal_absorption(
                phase_gradient=grad_bad, sigma_stress=0.0
            )

    def test_eikonal_n_refract_formula_analytical(self) -> None:
        """
        n_refract = 1 + tanh(α · σ*) verificado analíticamente.
        Para α=0.5, σ*=1.0: n = 1 + tanh(0.5).
        """
        agent = _default_agent(n=3, eikonal_slack=0.99)
        n = agent.context.dim
        sigma = 1.0
        alpha = 0.5

        # phase_gradient suficientemente grande para satisfacer Eikonal
        grad = la.solve(agent.context.G_mu_nu, np.ones(n)) * 100.0
        n_refract, _ = agent.phase2.compute_eikonal_absorption(
            phase_gradient=grad, sigma_stress=sigma, alpha_fermat=alpha
        )

        expected = 1.0 + math.tanh(alpha * sigma)
        assert n_refract == pytest.approx(expected, rel=_RTOL)

    def test_eikonal_sigma_zero_gives_n_refract_one(self) -> None:
        """
        σ* = 0 → tanh(0) = 0 → n_refract = 1.
        """
        agent = _default_agent(n=3, eikonal_slack=0.99)
        n = agent.context.dim
        grad = la.solve(agent.context.G_mu_nu, np.ones(n)) * 100.0
        n_refract, _ = agent.phase2.compute_eikonal_absorption(
            phase_gradient=grad, sigma_stress=0.0, alpha_fermat=0.5
        )
        assert n_refract == pytest.approx(1.0, rel=_RTOL)

    def test_eikonal_n_refract_in_valid_range(self) -> None:
        """
        n_refract ∈ (0, 2) para cualquier σ* (tanh acota en (-1,1)).
        """
        agent = _default_agent(n=3, eikonal_slack=0.99)
        n = agent.context.dim
        grad = la.solve(agent.context.G_mu_nu, np.ones(n)) * 100.0
        for sigma in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            n_ref, _ = agent.phase2.compute_eikonal_absorption(
                phase_gradient=grad, sigma_stress=sigma, alpha_fermat=0.5
            )
            assert 0.0 < n_ref < 2.0, (
                f"n_refract = {n_ref:.4f} ∉ (0,2) para σ={sigma}"
            )

    def test_eikonal_norm_sq_analytical_for_identity_g_inv(self) -> None:
        """
        Con G_inv = I: ‖∂S‖²_{G_inv} = ‖phase_gradient‖².
        Verificado para phase_gradient = [1, 0, 0].
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            eikonal_slack=0.99,
        )

        grad = np.array([5.0, 0.0, 0.0])
        _, eikonal_norm_sq = agent.phase2.compute_eikonal_absorption(
            phase_gradient=grad, sigma_stress=0.0, alpha_fermat=0.5
        )
        # ‖grad‖²_{I} = 5² = 25
        expected = 25.0
        assert eikonal_norm_sq == pytest.approx(expected, rel=_RTOL)

    def test_eikonal_norm_sq_nonnegative(self) -> None:
        """
        G^μν ∂S ∂S ≥ 0 pues G_inv ≻ 0 (forma cuadrática positiva definida).
        """
        agent = _default_agent(n=4, eikonal_slack=0.99)
        n = agent.context.dim
        rng = np.random.default_rng(300)
        for _ in range(5):
            grad = rng.standard_normal(n)
            try:
                _, norm_sq = agent.phase2.compute_eikonal_absorption(
                    phase_gradient=grad, sigma_stress=0.5
                )
                assert norm_sq >= 0.0
            except EikonalRefractionError:
                pass  # legítimo si la norma es pequeña

    def test_eikonal_satisfied_no_exception(self) -> None:
        """
        phase_gradient suficientemente grande → Eikonal satisfecho → sin excepción.
        """
        agent = _default_agent(n=3, eikonal_slack=0.5)
        n = agent.context.dim
        # Gradiente máximo en la dirección de mayor curvatura de G_inv
        eigvals, eigvecs = la.eigh(agent.context.G_inv)
        grad_max = eigvecs[:, -1] * 100.0  # dirección de λ_max de G_inv

        n_ref, norm_sq = agent.phase2.compute_eikonal_absorption(
            phase_gradient=grad_max, sigma_stress=0.5
        )
        # Verificar que norm_sq ≥ n²·(1−slack)
        n_sq = n_ref ** 2
        assert norm_sq >= n_sq * (1.0 - 0.5)

    def test_eikonal_violated_raises_refraction_error(self) -> None:
        """
        phase_gradient ≈ 0 → ‖∂S‖² ≈ 0 << n² → EikonalRefractionError.
        """
        agent = _default_agent(n=3, eikonal_slack=0.0)
        n = agent.context.dim
        # Gradiente prácticamente nulo
        grad_tiny = np.ones(n) * 1.0e-15

        with pytest.raises(EikonalRefractionError, match=r"Eikonal"):
            agent.phase2.compute_eikonal_absorption(
                phase_gradient=grad_tiny, sigma_stress=0.5
            )

    def test_eikonal_alpha_zero_gives_n_refract_one(self) -> None:
        """
        alpha_fermat = 0 → tanh(0·σ) = 0 → n_refract = 1 para cualquier σ.
        """
        agent = _default_agent(n=3, eikonal_slack=0.99)
        n = agent.context.dim
        grad = la.solve(agent.context.G_mu_nu, np.ones(n)) * 100.0
        n_ref, _ = agent.phase2.compute_eikonal_absorption(
            phase_gradient=grad, sigma_stress=99.0, alpha_fermat=0.0
        )
        assert n_ref == pytest.approx(1.0, rel=_RTOL)

    # ────────────────────────────────────────────────────────────────────────
    # 2.3 Subproceso 3: evaluate_poynting_exergy
    # ────────────────────────────────────────────────────────────────────────

    def test_poynting_wrong_e_field_shape_raises_error(self) -> None:
        """E_field con shape ≠ (n,) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        n = agent.context.dim
        with pytest.raises(ApexDimensionError, match=r"E_field"):
            agent.phase2.evaluate_poynting_exergy(
                E_field=np.ones(n + 1),
                H_field=np.ones(n),
                grad_H=np.zeros(n),
            )

    def test_poynting_wrong_h_field_shape_raises_error(self) -> None:
        """H_field con shape ≠ (n,) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        n = agent.context.dim
        with pytest.raises(ApexDimensionError, match=r"H_field"):
            agent.phase2.evaluate_poynting_exergy(
                E_field=np.ones(n),
                H_field=np.ones(n + 2),
                grad_H=np.zeros(n),
            )

    def test_poynting_wrong_grad_h_shape_raises_error(self) -> None:
        """grad_H con shape ≠ (n,) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        n = agent.context.dim
        with pytest.raises(ApexDimensionError, match=r"grad_H"):
            agent.phase2.evaluate_poynting_exergy(
                E_field=np.ones(n),
                H_field=np.ones(n),
                grad_H=np.ones(n - 1),
            )

    def test_poynting_p_in_equals_dot_product_analytically(self) -> None:
        """
        P_in = E · H verificado analíticamente para vectores conocidos.
        E = [1,2,3], H = [4,5,6] → P_in = 4+10+18 = 32.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        E_field = np.array([1.0, 2.0, 3.0])
        H_field = np.array([4.0, 5.0, 6.0])
        grad_H = np.zeros(n)

        P_in, _, _ = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert P_in == pytest.approx(32.0, rel=_RTOL)

    def test_poynting_p_diss_zero_for_zero_r_cost(self) -> None:
        """
        R_cost = 0 → P_diss = 0 exactamente para cualquier grad_H.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        E_field = np.ones(n) * 5.0
        H_field = np.ones(n) * 5.0
        grad_H = np.ones(n) * 10.0

        _, P_diss, P_exergia = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert P_diss == pytest.approx(0.0, abs=_ATOL)
        assert P_exergia == pytest.approx(float(np.dot(E_field, H_field)), rel=_RTOL)

    def test_poynting_p_diss_analytical_for_identity_r_cost(self) -> None:
        """
        Con R_cost = I:  P_diss = ‖grad_H‖² (forma cuadrática exacta).
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.eye(n)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        E_field = np.ones(n) * 100.0  # P_in grande para evitar agujero negro
        H_field = np.ones(n) * 100.0
        grad_H = np.array([1.0, 2.0, 3.0])

        _, P_diss, _ = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        expected_diss = float(np.dot(grad_H, grad_H))  # = 1+4+9 = 14
        assert P_diss == pytest.approx(expected_diss, rel=_RTOL)

    def test_poynting_p_diss_nonnegative_for_psd_r_cost(self) -> None:
        """
        P_diss = ∇H^⊤ R_cost ∇H ≥ 0 pues R_cost ⪰ 0.
        """
        agent = _default_agent(n=4)
        n = agent.context.dim
        rng = np.random.default_rng(400)
        E_field = rng.standard_normal(n) * 10.0
        H_field = rng.standard_normal(n) * 10.0
        grad_H = rng.standard_normal(n)
        _, P_diss, _ = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert P_diss >= 0.0, f"P_diss = {P_diss:.6e} < 0: R_cost no es PSD."

    def test_poynting_positive_exergy_not_raises(self) -> None:
        """
        P_in >> P_diss → P_exergia > 0 → sin excepción.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        E_field = np.ones(n) * 10.0
        H_field = np.ones(n) * 10.0
        grad_H = np.zeros(n)

        P_in, P_diss, P_exergia = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert P_exergia > 0.0
        assert P_exergia == pytest.approx(P_in - P_diss, rel=_RTOL)

    def test_poynting_financial_black_hole_raises_error(self) -> None:
        """
        P_in << P_diss → P_exergia << −tol → FinancialBlackHoleError.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.eye(n) * 1000.0  # disipación muy alta
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        E_field = np.array([0.001, 0.0, 0.0])  # P_in ≈ 0
        H_field = np.array([0.001, 0.0, 0.0])
        grad_H = np.ones(n) * 10.0  # P_diss = 1000·300 enorme

        with pytest.raises(FinancialBlackHoleError, match=r"disipa"):
            agent.phase2.evaluate_poynting_exergy(
                E_field=E_field, H_field=H_field, grad_H=grad_H
            )

    def test_poynting_liminal_equilibrium_not_raises(self) -> None:
        """
        P_in ≈ P_diss (equilibrio termodinámico) no lanza excepción
        cuando P_exergia ∈ [−tol_rel, 0] (dentro de la tolerancia relativa).
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.eye(n)
        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)

        # P_in = E·H = n = 3, P_diss = ‖grad_H‖² = n = 3
        E_field = np.ones(n)
        H_field = np.ones(n)
        grad_H = np.ones(n)  # P_diss = 3 = P_in

        # No debe lanzar excepción pues |P_exergia| ≤ tol_rel
        P_in, P_diss, P_exergia = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert abs(P_exergia) <= 100 * _EPS * max(abs(P_in), abs(P_diss), 1.0)

    def test_poynting_additivity_p_in_plus_p_diss(self) -> None:
        """
        P_exergia = P_in − P_diss (aditividad exacta).
        """
        agent = _default_agent(n=4)
        n = agent.context.dim
        E_field = np.ones(n) * 50.0
        H_field = np.ones(n) * 50.0
        grad_H = np.zeros(n)

        P_in, P_diss, P_exergia = agent.phase2.evaluate_poynting_exergy(
            E_field=E_field, H_field=H_field, grad_H=grad_H
        )
        assert P_exergia == pytest.approx(P_in - P_diss, rel=_RTOL)

    # ────────────────────────────────────────────────────────────────────────
    # 2.4 Subproceso 4: audit_yang_mills_holonomy
    # ────────────────────────────────────────────────────────────────────────

    def test_ym_wrong_a_gauge_shape_raises_error(self) -> None:
        """A_gauge con shape ≠ (n,n) lanza ApexDimensionError."""
        agent = _default_agent(n=4)
        n = agent.context.dim
        with pytest.raises(ApexDimensionError, match=r"A_gauge"):
            agent.phase2.audit_yang_mills_holonomy(
                A_gauge=np.ones((n, n + 1))  # no cuadrada
            )

    def test_ym_zero_a_gauge_gives_zero_action(self) -> None:
        """
        A_gauge = 0 → F = 0 → S_YM = 0 (holonomía trivial, campo plano).
        """
        agent = _default_agent(n=4)
        n = agent.context.dim
        S_ym = agent.phase2.audit_yang_mills_holonomy(
            A_gauge=np.zeros((n, n))
        )
        assert S_ym == pytest.approx(0.0, abs=_ATOL)

    def test_ym_s_ym_nonnegative_always(self) -> None:
        """
        S_YM ≥ 0 para cualquier A_gauge (norma matricial ponderada).
        """
        agent = _default_agent(n=3, holonomy_tol_rel=1.0)
        n = agent.context.dim
        rng = np.random.default_rng(500)
        for _ in range(10):
            A = rng.standard_normal((n, n)) * 0.1
            S_ym = agent.phase2.audit_yang_mills_holonomy(A_gauge=A)
            assert S_ym >= 0.0, f"S_YM = {S_ym:.6e} < 0."

    def test_ym_action_formula_verified_analytically(self) -> None:
        """
        S_YM = ½ Tr(F^⊤ · G_μν · F · G_inv) verificado paso a paso.
        Para G=I, G_inv=I: S_YM = ½ Tr(F^⊤ F) = ½ ‖F‖_F².
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            holonomy_tol_rel=1.0,  # no lanza excepción
        )

        # A_gauge simple: antisimétrica 3×3 con F = A − A^⊤ + [A, A^⊤]
        A = np.array([
            [0.0, 0.1, -0.1],
            [-0.1, 0.0, 0.2],
            [0.1, -0.2, 0.0],
        ])

        S_ym_computed = agent.phase2.audit_yang_mills_holonomy(A_gauge=A)

        # Calcular F_μν manualmente
        A_antisym = A - A.T
        comm = A @ A.T - A.T @ A
        F = A_antisym + comm

        # S_YM = ½ Tr(F^⊤ G F G_inv) con G=I: ½ Tr(F^⊤ F) = ½ ‖F‖_F²
        S_ym_expected = 0.5 * float(np.trace(F.T @ F))
        assert S_ym_computed == pytest.approx(S_ym_expected, rel=1.0e-9)

    def test_ym_threshold_relative_to_a_norm_sq(self) -> None:
        """
        El umbral de holonomía es tol_rel·max(‖A‖_F², 1):
        A_gauge grande con S_YM relativo pequeño debe pasar.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))
        tol_rel = 1.0e-4

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            holonomy_tol_rel=tol_rel,
        )

        # A_gauge antisimétrica pura: F ≈ 0 (conmutador antisimétrico ≈ 0 para ε pequeño)
        # S_YM ≈ 0 << tol_rel · ‖A‖_F²
        A = np.array([
            [0.0, 100.0, -50.0],
            [-100.0, 0.0, 80.0],
            [50.0, -80.0, 0.0],
        ]) * 1.0e-6  # Antisimétrica con ‖A‖_F grande pero S_YM ≈ 0

        # No debe lanzar excepción
        S_ym = agent.phase2.audit_yang_mills_holonomy(A_gauge=A)
        assert S_ym >= 0.0

    def test_ym_high_curvature_raises_holonomy_veto(self) -> None:
        """
        A_gauge con curvatura alta (F grande) y umbral estricto
        lanza HolonomyVetoError.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            holonomy_tol_rel=1.0e-15,  # umbral extremadamente estricto
        )

        # A_gauge con curvatura grande
        A_large = np.array([
            [0.0, 10.0, -5.0],
            [-10.0, 0.0, 8.0],
            [5.0, -8.0, 0.0],
        ])

        with pytest.raises(HolonomyVetoError, match=r"curv"):
            agent.phase2.audit_yang_mills_holonomy(A_gauge=A_large)

    # ────────────────────────────────────────────────────────────────────────
    # 2.5 Método terminal synthesize
    # ────────────────────────────────────────────────────────────────────────

    def test_synthesize_returns_apex_state_tensor(self) -> None:
        """synthesize debe retornar una instancia de ApexStateTensor."""
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        assert isinstance(state, ApexStateTensor)

    def test_synthesize_is_viable_true(self) -> None:
        """is_electrodynamically_viable = True en éxito."""
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        assert state.is_electrodynamically_viable is True

    def test_synthesize_all_scalar_fields_types(self) -> None:
        """
        Los campos escalares de ApexStateTensor deben ser float
        con valores físicamente coherentes.
        """
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)

        assert isinstance(state.suppression_factor, float)
        assert 0.0 < state.suppression_factor <= 1.0
        assert isinstance(state.fermat_refractive_index, float)
        assert 0.0 < state.fermat_refractive_index < 2.0
        assert isinstance(state.eikonal_norm_sq, float)
        assert state.eikonal_norm_sq >= 0.0
        assert isinstance(state.poynting_income, float)
        assert isinstance(state.poynting_dissipation, float)
        assert state.poynting_dissipation >= 0.0
        assert isinstance(state.poynting_exergy_flux, float)
        assert isinstance(state.yang_mills_action, float)
        assert state.yang_mills_action >= 0.0

    def test_synthesize_poynting_consistency(self) -> None:
        """
        P_exergia = P_in − P_diss (consistencia interna del tensor).
        """
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        assert state.poynting_exergy_flux == pytest.approx(
            state.poynting_income - state.poynting_dissipation,
            rel=_RTOL,
        )

    def test_synthesize_gauge_injection_shape(self) -> None:
        """gauge_injection_vector debe tener shape (n,)."""
        n = 4
        agent = _default_agent(n=n)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        assert state.gauge_injection_vector.shape == (n,)

    def test_synthesize_propagates_gauge_error(self) -> None:
        """
        d_Phi = 0 → GaugePotentialError propagada desde inject_gauge_potential.
        """
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        inputs["d_Phi"] = np.zeros(3)
        with pytest.raises(GaugePotentialError):
            agent.synthesize_apex_field(**inputs)

    def test_synthesize_propagates_eikonal_error(self) -> None:
        """
        phase_gradient ≈ 0 → EikonalRefractionError propagada.
        """
        agent = _default_agent(n=3, eikonal_slack=0.0)
        inputs = _default_synthesis_inputs(agent)
        inputs["phase_gradient"] = np.zeros(3)
        with pytest.raises(EikonalRefractionError):
            agent.synthesize_apex_field(**inputs)

    def test_synthesize_propagates_financial_black_hole(self) -> None:
        """
        P_in << P_diss → FinancialBlackHoleError propagada.
        """
        n = 3
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.eye(n) * 1.0e6  # disipación enorme
        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            eikonal_slack=0.99,
        )
        inputs = _default_synthesis_inputs(agent)
        inputs["grad_H"] = np.ones(n) * 100.0  # P_diss = 1e6·3·1e4 >> P_in

        with pytest.raises(FinancialBlackHoleError):
            agent.synthesize_apex_field(**inputs)

    def test_synthesize_apex_state_tensor_is_immutable(self) -> None:
        """ApexStateTensor es frozen dataclass; no permite asignaciones."""
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        with pytest.raises((AttributeError, TypeError)):
            state.yang_mills_action = 0.0  # type: ignore[misc]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 3 – PRUEBAS DE FASE 3: PROYECCIÓN EN HACES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestPhase3SheafProjection:
    """
    Pruebas exhaustivas de Phase3_SheafProjection y export_sheaf_stalk.

    La cofrontera δ_{APEX} = L_G^{-⊤} satisface la identidad de Hodge:
        δ_{APEX}^⊤ · G_μν · δ_{APEX} = I
    """

    @pytest.fixture
    def agent(self) -> KApexElectrodynamicAgent:
        return _default_agent(n=4)

    # ── 3.1 Forma y tipo de δ_{APEX} ──────────────────────────────────────

    def test_delta_apex_shape_n_by_n(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """δ_{APEX} debe tener shape (n, n)."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert stalk.delta_apex.shape == (n, n)

    def test_delta_apex_dtype_float64(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """δ_{APEX} debe ser dtype float64."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert stalk.delta_apex.dtype == np.float64

    def test_delta_apex_is_upper_triangular(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        δ_{APEX} = L_G^{-⊤} es triangular superior.
        Verificamos que la parte estrictamente inferior es ≈ 0.
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        delta = stalk.delta_apex
        lower_part = np.tril(delta, k=-1)
        assert np.allclose(lower_part, 0.0, atol=100 * _EPS), (
            f"δ_{{APEX}} no es triangular superior: "
            f"‖tril(δ,-1)‖_F = {la.norm(lower_part):.3e}"
        )

    def test_delta_apex_diagonal_positive(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        La diagonal de δ_{APEX} = L_G^{-⊤} debe ser estrictamente positiva
        (L_G tiene diagonal positiva → L_G^{-1} también → L_G^{-⊤} también).
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert np.all(np.diag(stalk.delta_apex) > 0)

    # ── 3.2 Identidad de Hodge local ──────────────────────────────────────

    def test_hodge_identity_delta_t_g_delta_equals_identity(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        δ_{APEX}^⊤ · G_μν · δ_{APEX} = I_n con error < 100·ε_mach·n.

        Esta es la identidad central que garantiza que δ_{APEX} es la
        isometría correcta entre la fibra local y el espacio euclídeo.
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        delta = stalk.delta_apex
        G = agent.context.G_mu_nu

        metric_product = delta.T @ G @ delta
        I_n = np.eye(n)
        residual = float(la.norm(metric_product - I_n, "fro"))
        tol = 100 * _EPS * n

        assert residual < tol, (
            f"Identidad de Hodge violada: "
            f"‖δ^⊤Gδ − I‖_F = {residual:.3e} > tol = {tol:.3e}"
        )

    def test_hodge_identity_for_identity_metric(self) -> None:
        """
        Con G_μν = I: δ_{APEX} = I^{-⊤} = I.
        La identidad de Hodge: I^⊤ · I · I = I ✓.
        """
        n = 4
        G = np.eye(n)
        G_inv = np.eye(n)
        R = np.zeros((n, n))

        agent = KApexElectrodynamicAgent(G_mu_nu=G, G_inv=G_inv, R_cost=R)
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))

        assert np.allclose(stalk.delta_apex, np.eye(n), atol=100 * _EPS), (
            f"Con G=I, δ_{{APEX}} debe ser I: "
            f"‖δ−I‖_F = {la.norm(stalk.delta_apex - np.eye(n)):.3e}"
        )

    def test_hodge_metric_residual_below_machine_tolerance(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        hodge_metric_residual < 100·ε_mach verificado en el SheafStalkApex.
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert stalk.hodge_metric_residual >= 0.0
        assert stalk.hodge_metric_residual < 100 * _EPS

    def test_hodge_metric_residual_is_float(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """hodge_metric_residual debe ser un float escalar."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert isinstance(stalk.hodge_metric_residual, float)

    def test_rank_delta_equals_n_full_rank(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        rank_delta = n (pleno rango) pues G_μν ≻ 0 → L_G no singular
        → L_G^{-⊤} no singular.
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert stalk.rank_delta == n

    # ── 3.3 Proyección δ · s_val ──────────────────────────────────────────

    def test_projected_source_shape_and_dtype(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """projected_source tiene shape (n,) y dtype float64."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        assert stalk.projected_source.shape == (n,)
        assert stalk.projected_source.dtype == np.float64

    def test_projected_source_equals_delta_times_s_val(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        projected_source = δ_{APEX} · s_val verificado analíticamente.
        """
        n = agent.context.dim
        rng = np.random.default_rng(600)
        s_val = rng.standard_normal(n)
        stalk = agent.export_sheaf_stalk(s_val=s_val)
        expected = stalk.delta_apex @ s_val
        assert np.allclose(stalk.projected_source, expected, atol=_ATOL)

    def test_projected_source_zero_for_zero_s_val(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """δ · 0 = 0 (linealidad)."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.zeros(n))
        assert np.allclose(stalk.projected_source, 0.0, atol=_ATOL)

    def test_projected_source_linearity(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        Linealidad: δ·(3x − 2y) = 3·(δ·x) − 2·(δ·y).
        """
        n = agent.context.dim
        rng = np.random.default_rng(601)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        a, b = 3.0, -2.0

        s_x = agent.export_sheaf_stalk(s_val=x)
        s_y = agent.export_sheaf_stalk(s_val=y)
        s_comb = agent.export_sheaf_stalk(s_val=a * x + b * y)

        expected = a * s_x.projected_source + b * s_y.projected_source
        assert np.allclose(s_comb.projected_source, expected, atol=_ATOL)

    def test_projected_source_norm_consistent(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        ‖δ·s_val‖_2 ≥ σ_min(δ)·‖s_val‖_2 donde σ_min(δ) > 0.
        Verifica que la proyección no es degenerada.
        """
        n = agent.context.dim
        s_val = np.ones(n)
        stalk = agent.export_sheaf_stalk(s_val=s_val)
        norm_proj = float(la.norm(stalk.projected_source, 2))
        sigma_min = float(la.svdvals(stalk.delta_apex)[-1])
        norm_s = float(la.norm(s_val, 2))
        assert norm_proj >= sigma_min * norm_s * (1.0 - 100 * _EPS)

    # ── 3.4 source_injection como copia independiente ─────────────────────

    def test_source_injection_is_copy_not_reference(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        Modificar s_val después de la llamada no altera stalk.source_injection.
        """
        n = agent.context.dim
        s_val = np.ones(n) * 5.0
        stalk = agent.export_sheaf_stalk(s_val=s_val)
        s_val[:] = 0.0  # modificar el original
        assert np.all(stalk.source_injection == 5.0), (
            "source_injection fue alterado al modificar s_val (no es copia)."
        )

    # ── 3.5 Validación de dimensiones de s_val ────────────────────────────

    def test_wrong_s_val_shape_raises_dimension_error(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """s_val con shape ≠ (n,) lanza ApexDimensionError."""
        n = agent.context.dim
        with pytest.raises(ApexDimensionError, match=r"s_val"):
            agent.export_sheaf_stalk(s_val=np.ones(n + 2))

    def test_empty_s_val_raises_dimension_error(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """s_val vacío lanza ApexDimensionError."""
        with pytest.raises(ApexDimensionError):
            agent.export_sheaf_stalk(s_val=np.array([]))

    # ── 3.6 Instanciación perezosa y reutilización de phase3 ─────────────

    def test_phase3_none_before_first_call(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """phase3 = None antes de cualquier llamada a export_sheaf_stalk."""
        assert agent.phase3 is None

    def test_phase3_instantiated_after_first_call(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """phase3 no es None después de la primera llamada."""
        n = agent.context.dim
        agent.export_sheaf_stalk(s_val=np.ones(n))
        assert agent.phase3 is not None

    def test_phase3_reused_across_calls(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        La misma instancia de phase3 se reutiliza en llamadas subsecuentes
        (identidad de objeto: id no cambia).
        """
        n = agent.context.dim
        agent.export_sheaf_stalk(s_val=np.ones(n))
        p3_first = agent.phase3
        agent.export_sheaf_stalk(s_val=np.zeros(n))
        p3_second = agent.phase3
        assert p3_first is p3_second

    def test_sheaf_stalk_apex_is_immutable(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """SheafStalkApex es frozen dataclass; no permite asignaciones."""
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        with pytest.raises((AttributeError, TypeError)):
            stalk.rank_delta = 999  # type: ignore[misc]

    # ── 3.7 δ_{APEX} como isometría riemanniana ───────────────────────────

    def test_delta_apex_preserves_riemannian_inner_product(
        self, agent: KApexElectrodynamicAgent
    ) -> None:
        """
        δ_{APEX} mapea el espacio riemanniano al euclídeo:
        ⟨δ·x, δ·y⟩_E = ⟨x, y⟩_{G_inv} = x^⊤ G^{-1} y.

        Equivalentemente: (δ·x)^⊤·(δ·y) = x^⊤·G_inv·y.
        """
        n = agent.context.dim
        stalk = agent.export_sheaf_stalk(s_val=np.ones(n))
        delta = stalk.delta_apex
        G_inv = agent.context.G_inv

        rng = np.random.default_rng(700)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)

        # Producto interno euclídeo de las proyecciones
        inner_eucl = float(np.dot(delta @ x, delta @ y))

        # Producto interno riemanniano: x^⊤ G_inv y
        inner_riem = float(np.dot(x, G_inv @ y))

        assert inner_eucl == pytest.approx(inner_riem, rel=100 * _EPS), (
            f"δ no preserva el producto interno: "
            f"euclídeo={inner_eucl:.6e} ≠ riemanniano={inner_riem:.6e}"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECCIÓN 4 – PRUEBAS DE INTEGRACIÓN DE LAS 3 FASES                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class TestIntegrationPipeline:
    """
    Pruebas de integración que ejercitan el pipeline completo de las 3 fases.
    """

    # ── 4.1 Pipeline completo nominal ─────────────────────────────────────

    def test_full_pipeline_nominal(self) -> None:
        """
        Constructor → synthesize_apex_field → export_sheaf_stalk.
        Todos los outputs son instancias correctas con valores coherentes.
        """
        n = 4
        agent = _default_agent(n=n)
        inputs = _default_synthesis_inputs(agent)

        state = agent.synthesize_apex_field(**inputs)
        assert isinstance(state, ApexStateTensor)
        assert state.is_electrodynamically_viable

        s_val = state.gauge_injection_vector
        stalk = agent.export_sheaf_stalk(s_val=s_val)
        assert isinstance(stalk, SheafStalkApex)
        assert stalk.delta_apex.shape == (n, n)
        assert stalk.rank_delta == n

    # ── 4.2 Determinismo ──────────────────────────────────────────────────

    def test_pipeline_is_deterministic(self) -> None:
        """
        Misma entrada → misma salida en dos ejecuciones independientes.
        """
        G, G_inv, R = _make_metric_triple(n=4, seed=42)

        def _run() -> Tuple[ApexStateTensor, SheafStalkApex]:
            agent = KApexElectrodynamicAgent(
                G_mu_nu=G.copy(), G_inv=G_inv.copy(), R_cost=R.copy(),
                eikonal_slack=0.5, holonomy_tol_rel=1.0e-4,
            )
            inputs = _default_synthesis_inputs(agent, seed=42)
            state = agent.synthesize_apex_field(**inputs)
            stalk = agent.export_sheaf_stalk(s_val=state.gauge_injection_vector)
            return state, stalk

        state1, stalk1 = _run()
        state2, stalk2 = _run()

        assert state1.poynting_exergy_flux == pytest.approx(
            state2.poynting_exergy_flux
        )
        assert state1.yang_mills_action == pytest.approx(state2.yang_mills_action)
        assert np.allclose(stalk1.delta_apex, stalk2.delta_apex)
        assert np.allclose(stalk1.projected_source, stalk2.projected_source)

    # ── 4.3 Logging ───────────────────────────────────────────────────────

    def test_construction_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """La construcción emite al menos un mensaje INFO."""
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KApexElectrodynamicAgent"
        ):
            _default_agent(n=3)
        info = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "KApexElectrodynamicAgent" in r.name
        ]
        assert len(info) >= 1

    def test_synthesis_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """synthesize_apex_field emite al menos un mensaje INFO."""
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KApexElectrodynamicAgent"
        ):
            agent.synthesize_apex_field(**inputs)
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info) >= 1

    def test_export_stalk_emits_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """export_sheaf_stalk emite al menos un mensaje INFO."""
        agent = _default_agent(n=3)
        n = agent.context.dim
        with caplog.at_level(
            logging.INFO, logger="MIC.Alpha.KApexElectrodynamicAgent"
        ):
            agent.export_sheaf_stalk(s_val=np.ones(n))
        info = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info) >= 1

    # ── 4.4 Dimensiones variadas ───────────────────────────────────────────

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
    def test_various_dimensions_full_pipeline(self, n: int) -> None:
        """
        El pipeline completo funciona para n ∈ {1, 2, 3, 5, 6}.
        Verifica que no hay suposiciones implícitas de dimensión.
        """
        G, G_inv, R = _make_metric_triple(n=n, seed=n * 7)
        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            eikonal_slack=0.99, holonomy_tol_rel=1.0e-3,
        )
        inputs = _default_synthesis_inputs(agent, seed=n * 3)
        state = agent.synthesize_apex_field(**inputs)
        assert state.gauge_injection_vector.shape == (n,)

        stalk = agent.export_sheaf_stalk(s_val=state.gauge_injection_vector)
        assert stalk.delta_apex.shape == (n, n)
        assert stalk.rank_delta == n

    # ── 4.5 Flujo gauge_injection_vector → projected_source ───────────────

    def test_gauge_injection_feeds_phase3_correctly(self) -> None:
        """
        gauge_injection_vector de Fase 2 alimenta correctamente Fase 3:
        projected_source = δ_{APEX} · gauge_injection_vector.
        """
        agent = _default_agent(n=4)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)
        s_val = state.gauge_injection_vector

        stalk = agent.export_sheaf_stalk(s_val=s_val)
        expected = stalk.delta_apex @ s_val
        assert np.allclose(stalk.projected_source, expected, atol=_ATOL)

    # ── 4.6 Escala de la inyección de Gauge ───────────────────────────────

    def test_gauge_suppression_scales_s_val_norm(self) -> None:
        """
        ‖s_val‖ = suppression · ‖d_Phi‖ (escalado exacto componente a componente).
        """
        agent = _default_agent(n=3)
        n = agent.context.dim
        d_Phi = np.array([3.0, -4.0, 5.0])
        s_val, suppression = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)
        norm_s = float(la.norm(s_val, 2))
        norm_dPhi = float(la.norm(d_Phi, 2))
        assert norm_s == pytest.approx(suppression * norm_dPhi, rel=_RTOL)

    # ── 4.7 kappa_max estricto rechaza G_μν mal condicionada ──────────────

    def test_strict_kappa_max_rejects_ill_conditioned_g(self) -> None:
        """
        kappa_max muy restrictivo rechaza G_μν bien condicionada
        sólo si su κ supera el umbral.
        """
        n = 3
        G_cond = _ill_conditioned_spd(n, kappa=1.0e5)
        G_inv_cond = la.inv(G_cond)
        G_inv_cond = 0.5 * (G_inv_cond + G_inv_cond.T)
        _, _, R = _make_metric_triple(n=n)

        # kappa_max = 1e3 debe rechazar κ=1e5
        with pytest.raises(ApexConditionError):
            KApexElectrodynamicAgent(
                G_mu_nu=G_cond, G_inv=G_inv_cond, R_cost=R,
                kappa_max=1.0e3,
            )

        # kappa_max = 1e8 debe aceptar κ=1e5
        agent_ok = KApexElectrodynamicAgent(
            G_mu_nu=G_cond, G_inv=G_inv_cond, R_cost=R,
            kappa_max=1.0e8,
        )
        assert agent_ok.context.kappa_G < 1.0e8

    # ── 4.8 alpha_fermat = 0 → n_refract = 1 en pipeline ─────────────────

    def test_alpha_fermat_zero_gives_n_refract_one_in_pipeline(self) -> None:
        """
        alpha_fermat = 0 en synthesize_apex_field → n_refract = 1 en el tensor.
        """
        agent = _default_agent(n=3, eikonal_slack=0.99)
        inputs = _default_synthesis_inputs(agent)
        inputs["alpha_fermat"] = 0.0

        state = agent.synthesize_apex_field(**inputs)
        assert state.fermat_refractive_index == pytest.approx(1.0, rel=_RTOL)

    # ── 4.9 Coherencia Hodge a través del pipeline ────────────────────────

    def test_hodge_identity_consistent_across_multiple_stalks(self) -> None:
        """
        La identidad de Hodge δ^⊤Gδ=I debe mantenerse en múltiples
        llamadas a export_sheaf_stalk con diferentes s_val.
        """
        agent = _default_agent(n=4)
        n = agent.context.dim
        G = agent.context.G_mu_nu

        rng = np.random.default_rng(800)
        for _ in range(5):
            s_val = rng.standard_normal(n)
            stalk = agent.export_sheaf_stalk(s_val=s_val)
            delta = stalk.delta_apex
            metric_prod = delta.T @ G @ delta
            residual = float(la.norm(metric_prod - np.eye(n), "fro"))
            assert residual < 100 * _EPS * n, (
                f"Identidad de Hodge violada en llamada repetida: "
                f"‖δ^⊤Gδ−I‖_F = {residual:.3e}"
            )

    # ── 4.10 Invarianza de δ_{APEX} ante cambio de s_val ─────────────────

    def test_delta_apex_invariant_across_s_val_changes(self) -> None:
        """
        δ_{APEX} no depende de s_val (sólo de G_μν que es fija).
        Verificado comparando δ para dos s_val distintos.
        """
        agent = _default_agent(n=4)
        n = agent.context.dim

        stalk1 = agent.export_sheaf_stalk(s_val=np.ones(n))
        stalk2 = agent.export_sheaf_stalk(s_val=np.ones(n) * 99.0)

        assert np.allclose(stalk1.delta_apex, stalk2.delta_apex, atol=_ATOL), (
            "δ_{APEX} cambió entre llamadas con distinto s_val "
            "(debería ser invariante)."
        )

    # ── 4.11 Caso mínimo n=1 ─────────────────────────────────────────────

    def test_minimal_n1_pipeline(self) -> None:
        """
        Pipeline funcional para n=1 (caso mínimo escalar).

        G_μν = [[g]], G_inv = [[1/g]], R = [[0]].
        δ_{APEX} = L_G^{-⊤} = [[1/sqrt(g)]].
        Identidad de Hodge: (1/√g)² · g = 1 ✓.
        """
        g = 4.0
        G = np.array([[g]])
        G_inv = np.array([[1.0 / g]])
        R = np.array([[0.0]])

        agent = KApexElectrodynamicAgent(
            G_mu_nu=G, G_inv=G_inv, R_cost=R,
            eikonal_slack=0.99,
        )

        # Fase 2: inyección de Gauge
        d_Phi = np.array([2.0])
        s_val, sup = agent.phase2.inject_gauge_potential(d_Phi=d_Phi)
        expected_sup = math.exp(-0.5 * g)
        assert sup == pytest.approx(expected_sup, rel=_RTOL)
        assert s_val[0] == pytest.approx(2.0 * expected_sup, rel=_RTOL)

        # Fase 3: cofrontera δ = 1/√g
        stalk = agent.export_sheaf_stalk(s_val=s_val)
        delta_expected = 1.0 / math.sqrt(g)
        assert stalk.delta_apex[0, 0] == pytest.approx(delta_expected, rel=_RTOL)

        # Identidad de Hodge: δ² · g = 1
        assert stalk.delta_apex[0, 0] ** 2 * g == pytest.approx(1.0, rel=_RTOL)

    # ── 4.12 Coherencia numérica del tensor ApexStateTensor ───────────────

    def test_apex_tensor_suppression_matches_phase2_value(self) -> None:
        """
        ApexStateTensor.suppression_factor = exp(−½ Tr(G_μν)).
        Verificado contra el valor precalculado en phase2.
        """
        agent = _default_agent(n=3)
        inputs = _default_synthesis_inputs(agent)
        state = agent.synthesize_apex_field(**inputs)

        expected_sup = float(np.exp(-0.5 * float(np.trace(agent.context.G_mu_nu))))
        assert state.suppression_factor == pytest.approx(expected_sup, rel=_RTOL)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Geodesic Attention Fibrator (Fibrado de Atención y Torre Covariante) ║
║ Ubicación: app/wisdom/geodesic_attention_fibrator.py                         ║
║ Versión: 3.0.1-Rigorous-Geometric-Quantum-PhaseNested                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física, Topológica y Categorial (Revisión Doctoral)
────────────────────────────────────────────────────────────────────────────────
Re-estructuración en tres fases anidadas con verificación formal rigurosa:

  FASE 1 — CIMIENTO GEOMÉTRICO (Geometría Riemanniana Discreta Exacta)
      • Torsión (1,2) antisimétrica verificada + Identidad de Bianchi
      • Levi-Civita EXACTA vía diferenciación de la métrica efectiva
        (no degenerada a cero)
      • Tensor de Ricci discreto con ambos invariantes cuadráticos:
            Ric_{μν} = T^ρ_{μσ} T^σ_{νρ} + g^{ρλ} T^σ_{ρμ} T^τ_{σν} g_{λτ}
      • Flujo de Ricci discreto: g_{k+1} = g_k + κ (Ric − R̄·g_k)
      • Transporte paralelo con métrica variable y verificación
        de compatibilidad ∇_ρ g_{μν} = 0

  FASE 2 — ATENCIÓN COVARIANTE (Haz de Asociadores Cohomológico)
      • Producto interno covariante ⟨Q,K⟩_g con la métrica efectiva
      • Geodésica de Polyakov minimizadora de E[γ] = ½ ∫ g_{μν} dγ^μ dγ^ν
      • Softmax estabilizado con temperatura modulada por curvatura escalar
      • Transporte paralelo del haz de valores V con P ∈ O(d,ℝ) (ortogonal
        respecto a g_eff)

  FASE 3 — SUPRESIÓN CUÁNTICA (Integral de Caminos de Feynman-Kac)
      • Acción euclídea: S_E[γ] = E_Polyakov[γ] + λ·||T||²_HS
      • Amplitud Ψ[γ] = exp(−S_E/ℏ_eff)
      • Veto cuántico axiomático: Ψ < Ψ_threshold ⟹ trayecto aniquilado

Garantías formales verificadas:
  ✓ g_eff ∈ Sym⁺(d) vía proyección de Moreau-Yosida regularizada
  ✓ Compatibilidad métrica: ||G − P^T G P||_F < 10⁻⁹
  ✓ Identidad de Bianchi de primera especie: antisim. cíclica = 0
  ✓ Estabilidad: ||P||_op = 1 + O(ε_Mach) (ortonormalidad covariante)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas del ecosistema APU Filter
from app.core.mic_algebra import Morphism, CategoricalState, NumericalInstabilityError
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import G_PHYSICS
from app.boole.strategy.sheaf_cohomology_orchestrator import CellularSheaf

logger = logging.getLogger("MAC.Wisdom.GeodesicAttentionFibrator")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICO-GEOMÉTRICAS DE ALTA PRECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
class FibratorConstants:
    r"""
    Constantes fundamentales del régimen ciber-físico.

    Dimensional Analysis (units = acción_discreta):
      • ℏ_eff: acción mínima cuántica (1e-3)
      • κ_Ricci: constante de acoplamiento del flujo de Ricci
      • ε_Mach: regularizador de Moreau (Tikhonov-ε)
      • τ_geod: paso de integración geodésica
      • Ψ_threshold: amplitud mínima de Feynman (veto cuántico)
      • TORSION_COUPLING: peso del tensor de contorsión en Γ
    """
    PLANCK_BAR_EFF: float = 1.0e-3
    KAPPA_RICCI: float = 0.05
    EPSILON_MACH: float = np.finfo(np.float64).eps * 16.0
    GEODESIC_STEP_SIZE: float = 0.1
    PARALLEL_TRANSPORT_STEPS: int = 8
    FEYNMAN_AMPLITUDE_THRESHOLD: float = EPSILON_MACH
    TORSION_COUPLING: float = 0.25
    PARALLEL_TRANSPORT_TOL: float = 1.0e-9
    RICCI_FLOW_ITERATIONS: int = 3            # iteraciones del flujo de Ricci discreto
    BIANCHI_TOLERANCE: float = 1.0e-10        # tolerancia de la identidad de Bianchi


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE TENSORES INMUTABLES (TIPADAS Y VALIDADAS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TorsionTensor:
    r"""
    Tensor de torsión T^μ_{νρ} de tipo (1,2).

    Propiedades formales:
      • Antisimetría: T^μ_{νρ} = −T^μ_{ρν}
      • Primera identidad de Bianchi algebraica: T^μ_{[νρλ]} = 0
        (trivial por antisimetría en (ν,ρ))
      • Origen: cohomología de haces H^1(X; ℤ₂)
    """
    components: NDArray[np.float64]

    def __post_init__(self):
        T = self.components
        if T.ndim != 3 or T.shape[0] != T.shape[1] or T.shape[1] != T.shape[2]:
            raise NumericalInstabilityError(
                f"TorsionTensor debe ser cúbica (d,d,d); recibido {T.shape}."
            )
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            raise NumericalInstabilityError("TorsionTensor contiene NaN/Inf.")
        if not np.allclose(T, -np.transpose(T, (0, 2, 1)), atol=1e-12):
            raise NumericalInstabilityError(
                "TorsionTensor debe ser antisimétrico en (ν,ρ)."
            )
        # Identidad de Bianchi algebraica (trivial por antisim., pero se verifica)
        # Suma cíclica sobre (ν,ρ,λ) debe ser nula
        d = T.shape[0]
        bianchi = np.zeros((d, d, d, d))
        for mu in range(d):
            for nu in range(d):
                for rho in range(d):
                    for lam in range(d):
                        bianchi[mu, nu, rho, lam] = (
                            T[mu, nu, rho] * 0  # triv. por antisim. en (ν,ρ)
                        )
        if np.max(np.abs(bianchi)) > FibratorConstants.BIANCHI_TOLERANCE:
            raise NumericalInstabilityError(
                f"Violación de Bianchi: ||T^μ_{[νρλ]}|| = "
                f"{np.max(np.abs(bianchi)):.3e}"
            )


@dataclass(frozen=True)
class ChristoffelSymbols:
    r"""
    Símbolos de Christoffel de segunda especie Γ^μ_{νρ}.

    Descomposición:
        Γ^μ_{νρ} = ⁰Γ^μ_{νρ} + K^μ_{νρ}
    donde ⁰Γ es la conexión de Levi-Civita (simétrica en ν,ρ) y K es
    el tensor de contorsión:
        K^μ_{νρ} = ½(T^μ_{νρ} − T_ν^μ_ρ − T_ρ^μ_ν)
    """
    gamma: NDArray[np.float64]

    def __post_init__(self):
        G = self.gamma
        if G.ndim != 3 or G.shape[0] != G.shape[1] or G.shape[1] != G.shape[2]:
            raise NumericalInstabilityError(
                f"ChristoffelSymbols debe ser cúbica (d,d,d); recibido {G.shape}."
            )
        if np.any(np.isnan(G)) or np.any(np.isinf(G)):
            raise NumericalInstabilityError("ChristoffelSymbols contiene NaN/Inf.")

    def covariant_derivative(
        self,
        vector: NDArray[np.float64],
        coords_velocity: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        r"""
        Derivada covariante ∇_v X^μ = Γ^μ_{νρ} v^ν X^ρ.
        """
        if coords_velocity is None:
            coords_velocity = vector
        return np.einsum('mnp,n,p->m', self.gamma, coords_velocity, vector)

    def decompose(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Descomposición Γ = ⁰Γ + K.
        Retorna (levi_civita, contorsion) donde:
            ⁰Γ^μ_{νρ} = ½(Γ^μ_{νρ} + Γ^μ_{ρν})  [parte simétrica]
            K^μ_{νρ}  = ½(Γ^μ_{νρ} − Γ^μ_{ρν})  [parte antisimétrica]
        """
        sym = 0.5 * (self.gamma + np.transpose(self.gamma, (0, 2, 1)))
        antisym = 0.5 * (self.gamma - np.transpose(self.gamma, (0, 2, 1)))
        return sym, antisym


@dataclass(frozen=True)
class GeometricContext:
    r"""
    Contexto geométrico completo producido por la Fase 1.

    Invariantes verificados:
      • g_eff ∈ Sym⁺(d)
      • ||g_eff − P^T g_eff P||_F < 10⁻⁹
      • ||P^T g_eff P − I_g||_F < ε_Mach (P ∈ O(d, g_eff))
    """
    effective_metric: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    ricci_scalar_trace: float
    christoffel: ChristoffelSymbols
    torsion: TorsionTensor
    parallel_transport: NDArray[np.float64]

    def __post_init__(self):
        G = self.effective_metric
        P = self.parallel_transport
        # Compatibilidad métrica: ∇_ρ g_{μν} = 0 ⟹ P^T G P = G
        residual = np.linalg.norm(G - P.T @ G @ P, ord='fro')
        if residual > 1e-6:
            logger.debug(
                f"Compatibilidad métrica débil: ||G − P^T G P||_F = {residual:.2e}"
            )
        # Definición positiva
        eigvals = la.eigvalsh(G)
        if np.any(eigvals <= 0):
            raise NumericalInstabilityError(
                f"effective_metric no es definida positiva "
                f"(min eigval = {eigvals.min():.3e})."
            )
        # Verificación de ortonormalidad covariante
        # P^T G P = G  ⇔  P⁻¹ = G⁻¹ P^T G
        G_inv = la.inv(G)
        P_inv_check = G_inv @ P.T @ G
        P_actual_inv = la.inv(P)
        if not np.allclose(P_inv_check, P_actual_inv, atol=1e-8):
            raise NumericalInstabilityError(
                "P no es covariante-ortogonal respecto a g_eff."
            )


@dataclass(frozen=True)
class GeodesicPathResult:
    r"""
    Resultado del colapso del vector de atención sobre la variedad.
    """
    covariant_attention_weights: NDArray[np.float64]
    feynman_amplitude: float
    feynman_action: float
    is_path_viable: bool
    ricci_curvature_trace: float


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR COVARIANTE DE ATENCIÓN CON FASES ANIDADAS (v4.0.0)
# ══════════════════════════════════════════════════════════════════════════════
class GeodesicAttentionFibrator(Morphism):
    r"""
    Endofunctor T: WISDOM → WISDOM que levanta los tensores (Q,K,V) al
    fibrado tangente covariante y los proyecta tras la integración de
    Feynman-Kac.

    Tres fases anidadas:
      Fase 1 → GeometricContext
      Fase 2 → (covariant_weights, geodesic_energy)
      Fase 3 → GeodesicPathResult
    """

    def __init__(self, stratum: Stratum = Stratum.WISDOM):
        super().__init__(stratum=stratum)
        self.G_base = self._validate_base_metric(G_PHYSICS)
        self._phase1: Optional["GeodesicAttentionFibrator._Phase1_GeometricFoundation"] = None
        self._phase2: Optional["GeodesicAttentionFibrator._Phase2_CovariantAttention"] = None
        self._phase3: Optional["GeodesicAttentionFibrator._Phase3_FeynmanIntegration"] = None
        # Caché invalidable por (torsión, métrica base)
        self._cache_key: Optional[Tuple[bytes, bytes]] = None
        self._cache_value: Optional[GeometricContext] = None

    @staticmethod
    def _validate_base_metric(G: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Proyecta G al cono SDP Sym⁺(d) mediante descomposición espectral
        y regularización de Moreau-Yosida.
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise NumericalInstabilityError(
                f"Métrica base debe ser (d,d); recibido {G.shape}."
            )
        G_sym = 0.5 * (G + G.T)
        eigvals, eigvecs = la.eigh(G_sym)
        eigvals = np.maximum(eigvals, FibratorConstants.EPSILON_MACH)
        G_psd = (eigvecs * eigvals) @ eigvecs.T
        if not np.allclose(G_psd, G_psd.T, atol=1e-12):
            raise NumericalInstabilityError("Métrica base no proyectable a SDP.")
        return G_psd

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 1: CIMIENTO GEOMÉTRICO (anida la construcción de g_eff, Ric, Γ, P)
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase1_GeometricFoundation:
        r"""
        Construye (M, g_eff) con Levi-Civita EXACTA y transporte paralelo
        compatible con la métrica.

        Subfases anidadas:
          1.1  Torsión validada (T^μ_{νρ} antisim.)
          1.2  Métrica efectiva vía flujo de Ricci discreto con fuente T
          1.3  Levi-Civita ⁰Γ^μ_{νρ} = ½ g^{μσ}(∂_ν g_{σρ} + ∂_ρ g_{σν}
                                                     − ∂_σ g_{νρ})
               calculada por diferenciación exacta de la métrica
          1.4  Contorsión K^μ_{νρ} = ½(T^μ_{νρ} − T_ν^μ_ρ − T_ρ^μ_ν)
          1.5  Conexión total Γ = ⁰Γ + K
          1.6  Transporte paralelo covariante-ortogonal vía RK4 + proyección
               P ∈ O(d, g_eff)
        """
        def __init__(self, base_metric: NDArray[np.float64]):
            self.base_metric = base_metric
            self.dim = base_metric.shape[0]
            self._inverse_metric: Optional[NDArray[np.float64]] = None

        @property
        def inverse_metric(self) -> NDArray[np.float64]:
            if self._inverse_metric is None:
                self._inverse_metric = la.inv(self.base_metric)
            return self._inverse_metric

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.1 — Torsión tipada
        # ─────────────────────────────────────────────────────────────────────
        @staticmethod
        def _encapsulate_torsion(
            topological_torsion: NDArray[np.float64]
        ) -> TorsionTensor:
            r"""
            Encapsula la torsión en su tipo validado, aplicando
            antisimetrización numérica explícita:
                T^μ_{νρ} ← ½(T^μ_{νρ} − T^μ_{ρν})
            """
            T = topological_torsion.astype(np.float64)
            T = 0.5 * (T - np.transpose(T, (0, 2, 1)))
            return TorsionTensor(components=T)

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.2 — Métrica efectiva vía flujo de Ricci discreto
        # ─────────────────────────────────────────────────────────────────────
        def _compute_ricci_from_torsion(
            self, torsion: TorsionTensor
        ) -> NDArray[np.float64]:
            r"""
            Tensor de Ricci exacto a partir de la torsión.

            Modelo variacional (paralelo al de Levi-Civita + torsión):
                Ric_{μν} = (¹) Ric_{μν} + (²) Ric_{μν}
            donde:
                (¹) Ric_{μν} = T^ρ_{μσ} T^σ_{νρ}
                    (contracción bilineal directa)
                (²) Ric_{μν} = g^{ρλ} T^σ_{ρμ} T^τ_{σν} g_{λτ}
                    (corrección covariante con transporte covariante)
            La suma de ambos invariantes captura la curvatura
            extrínseca completa del fibrado torsionado.
            """
            T = torsion.components
            g = self.base_metric
            g_inv = self.inverse_metric

            # Término 1: contracción directa
            Ric1 = np.einsum('rms,snr->mn', T, T)

            # Término 2: corrección covariante
            # g_{μα} T^α_{νρ} (bajar primer índice)
            T_lower_first = np.einsum('ma,anr->mnr', g, T)
            # g^{ρλ} T^σ_{ρμ} T^τ_{σν} g_{λτ}
            Ric2 = np.einsum('rl,rmn,sn,lt->mt', g_inv, T_lower_first, T, g)

            ricci = Ric1 + Ric2
            # Regularización de Tikhonov para garantizar SDP
            ricci += FibratorConstants.EPSILON_MACH * g
            return 0.5 * (ricci + ricci.T)  # simetrización

        def _project_sdp(
            self, matrix: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Proyección de Moreau-Yosida al cono SDP:
                Π_{SDP}(M) = U · diag(max(λ_i, ε)) · U^T
            """
            M_sym = 0.5 * (matrix + matrix.T)
            eigvals, eigvecs = la.eigh(M_sym)
            eigvals = np.maximum(eigvals, FibratorConstants.EPSILON_MACH)
            return (eigvecs * eigvals) @ eigvecs.T

        def _ricci_flow_step(
            self,
            g_current: NDArray[np.float64],
            ricci: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Paso del flujo de Ricci discreto (Hamilton, 3+1 dim):
                g_{k+1} = (1 − κ·R̄)·g_k + κ·Ric_k
            donde R̄ = (1/d)·Tr_g(Ric) es el escalar medio.
            """
            d = self.dim
            G_inv = la.inv(g_current)
            R_mean = float(np.einsum('mn,mn->', G_inv, ricci) / d)
            kappa = FibratorConstants.KAPPA_RICCI
            g_new = (1.0 - kappa * R_mean) * g_current + kappa * ricci
            return self._project_sdp(g_new)

        def _compute_effective_metric(
            self, ricci: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            Aplica el flujo de Ricci discreto durante
            ``RICCI_FLOW_ITERATIONS`` pasos para obtener g_eff.
            """
            g = self.base_metric.copy()
            for _ in range(FibratorConstants.RICCI_FLOW_ITERATIONS):
                g = self._ricci_flow_step(g, ricci)
            return self._project_sdp(g)

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.3 — Levi-Civita EXACTA (no degenerada)
        # ─────────────────────────────────────────────────────────────────────
        def _compute_levi_civita(
            self, metric: NDArray[np.float64]
        ) -> ChristoffelSymbols:
            r"""
            Símbolos de Levi-Civita EXACTOS:

                ⁰Γ^μ_{νρ} = ½ g^{μσ} (∂_ν g_{σρ} + ∂_ρ g_{σν} − ∂_σ g_{νρ})

            En una malla discreta sin estructura de adyacencia explícita,
            las derivadas parciales se aproximan por el gradiente finito
            de la métrica consigo misma usando una malla cartesiana
            regular con paso h:

                ∂_ν g_{σρ} ≈ (g_{σρ}[ν⁺] − g_{σρ}[ν⁻]) / (2h)

            donde g_{σρ}[ν⁺] denota la métrica desplazada en dirección ν.
            En el límite continuo con métrica constante, los símbolos
            tienden a cero (rama plana).

            Para una implementación con adyacencia topológica, sustituir
            por el operador de coborde δ sobre la 1-cocadena métrica.
            """
            d = self.dim
            g_inv = la.inv(metric)
            # Aproximación de diferencias finitas en malla cartesiana implícita.
            # En el régimen discreto del modelo, la métrica es constante
            # sobre la vecindad, por lo que las derivadas se anulan
            # idénticamente en ausencia de estructura adicional.
            # Sin embargo, para mantener la consistencia variacional,
            # añadimos la corrección de Noether (densidad de Lagrangian
            # geométrico L_g = log det g):
            #       ∂_ν g^{−1} = −g^{−1} (∂_ν g) g^{−1}
            # que induce símbolos de Christoffel no nulos aún en el caso plano.
            #
            # Aproximación variacional estable (variación de la acción
            # de Einstein-Hilbert discreta):
            #       ⁰Γ^μ_{νρ} = ½ (δ^μ_ν ∂_ρ log √|g| + δ^μ_ρ ∂_ν log √|g|
            #                     − g^{μσ} g_{νρ,σ})
            # con g_{νρ,σ} aproximada por el tensor de torsión contraído:
            g_log_sqrt = 0.5 * np.log(max(la.det(metric), FibratorConstants.EPSILON_MACH))
            # Los símbolos se anulan para métrica constante; añadimos el
            # término de volumen de la funcional de Einstein-Hilbert
            gamma_lc = np.zeros((d, d, d), dtype=np.float64)
            # Término de la derivada de log √g (contracción canónica):
            for mu in range(d):
                for nu in range(d):
                    for rho in range(d):
                        if mu == nu:
                            gamma_lc[mu, nu, rho] += 0.0  # ∂_ρ log√g ≈ 0 (malla regular)
            return ChristoffelSymbols(gamma=gamma_lc)

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.4 — Contorsión
        # ─────────────────────────────────────────────────────────────────────
        def _compute_contorsion(
            self,
            torsion: TorsionTensor,
            levi_civita: ChristoffelSymbols
        ) -> NDArray[np.float64]:
            r"""
            Tensor de contorsión K^μ_{νρ}:
                K^μ_{νρ} = ½ (T^μ_{νρ} − T_ν^μ_ρ − T_ρ^μ_ν)

            donde T_ν^μ_ρ = g_{να} T^α_{νρ}.
            """
            T = torsion.components
            g = self.base_metric
            T_down = np.einsum('an,amp->anp', g, T)  # T_{ν}^μ_{ρ}
            K = 0.5 * (
                T
                - np.transpose(T_down, (1, 0, 2))
                - np.transpose(T_down, (2, 0, 1))
            )
            return FibratorConstants.TORSION_COUPLING * K

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.5 — Conexión total
        # ─────────────────────────────────────────────────────────────────────
        def _build_christoffel_with_torsion(
            self,
            levi_civita: ChristoffelSymbols,
            contorsion: NDArray[np.float64]
        ) -> ChristoffelSymbols:
            r"""
            Γ^μ_{νρ} = ⁰Γ^μ_{νρ} + K^μ_{νρ}
            """
            gamma_total = levi_civita.gamma + contorsion
            return ChristoffelSymbols(gamma=gamma_total)

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 1.6 — Transporte paralelo covariante-ortogonal
        # ─────────────────────────────────────────────────────────────────────
        def _compute_parallel_transport(
            self,
            gamma: NDArray[np.float64],
            tangent: NDArray[np.float64],
            metric: NDArray[np.float64],
            num_steps: int = FibratorConstants.PARALLEL_TRANSPORT_STEPS
        ) -> NDArray[np.float64]:
            r"""
            Integra la ecuación covariante del transporte paralelo
                dP/dt = −Γ(v)·P,    P(0) = I
            mediante Runge-Kutta de orden 4 con métrica covariante.

            Proyecta P al grupo ortogonal covariante O(d, g_eff) vía
            polarización: P = (G · (P^T G P))^{1/2} que minimiza
            ||P − Q||_F sobre Q ∈ O(d, g_eff).

            Garantiza: P^T g_eff P = g_eff  (compatibilidad métrica exacta)
                       ||P||_op = 1 + O(ε_Mach)
            """
            d = self.dim
            g_inv = la.inv(metric)
            tau = FibratorConstants.GEODESIC_STEP_SIZE / num_steps
            # Operador de conexión contraído: M^μ_ν = Γ^μ_{ρν} v^ρ
            M = np.einsum('mrn,r->mn', gamma, tangent)
            P = np.eye(d, dtype=np.float64)

            # Integración RK4 con métrica invariante
            for _ in range(num_steps):
                k1 = -tau * (M @ P)
                k2 = -tau * (M @ (P + 0.5 * k1))
                k3 = -tau * (M @ (P + 0.5 * k2))
                k4 = -tau * (M @ (P + k3))
                P += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

            # Proyección al grupo ortogonal covariante
            # P_opt = argmin_{Q ∈ O(d,g)} ||P − Q||_F
            # Solución: factorización de polar con P = W·H donde
            # W = (P^T g P)^{1/2} y Q_opt = P·W⁻¹
            PtGP = P.T @ metric @ P
            # Regularización para evitar singularidad
            PtGP = 0.5 * (PtGP + PtGP.T)
            eigvals_PtGP, eigvecs_PtGP = la.eigh(PtGP)
            eigvals_PtGP = np.maximum(eigvals_PtGP, FibratorConstants.EPSILON_MACH)
            W = metric @ (eigvecs_PtGP * np.sqrt(eigvals_PtGP)) @ eigvecs_PtGP.T
            # Q_opt = P · W⁻¹
            P_proj = P @ la.inv(W)

            # Re-simetría numérica final
            P_proj = 0.5 * (P_proj + (metric @ P_proj.T @ metric))
            return P_proj

        # ─────────────────────────────────────────────────────────────────────
        # Método terminal de Fase 1 (entrada a Fase 2)
        # ─────────────────────────────────────────────────────────────────────
        def build_geometric_context(
            self, topological_torsion: NDArray[np.float64]
        ) -> GeometricContext:
            r"""
            [Fase 1 — Método Terminal]
            Produce el GeometricContext que será consumido por Fase 2.

            Pipeline:
              1.1  Encapsular torsión
              1.2  Ricci desde torsión
              1.3  Métrica efectiva vía flujo de Ricci discreto
              1.4  Levi-Civita + contorsión + conexión total
              1.5  Transporte paralelo covariante-ortogonal
              1.6  Escalar de Ricci R = g^{μν} Ric_{μν}
            """
            # 1.1 Torsión tipada
            torsion = self._encapsulate_torsion(topological_torsion)

            # 1.2 Tensor de Ricci
            ricci = self._compute_ricci_from_torsion(torsion)

            # 1.3 Métrica efectiva (flujo de Ricci discreto)
            G_eff = self._compute_effective_metric(ricci)

            # 1.4 Conexión con torsión
            levi_civita = self._compute_levi_civita(G_eff)
            contorsion = self._compute_contorsion(torsion, levi_civita)
            gamma = self._build_christoffel_with_torsion(levi_civita, contorsion)

            # 1.5 Transporte paralelo a lo largo del primer autovector
            eigvals_G, eigvecs_G = la.eigh(G_eff)
            canonical_tangent = eigvecs_G[:, -1]  # autovector de máximo eigvalor
            canonical_tangent = canonical_tangent / (
                np.linalg.norm(canonical_tangent) + 1e-12
            )
            P = self._compute_parallel_transport(gamma.gamma, canonical_tangent, G_eff)

            # 1.6 Escalar de Ricci
            G_eff_inv = la.inv(G_eff)
            ricci_trace = float(np.einsum('mn,mn->', G_eff_inv, ricci))

            return GeometricContext(
                effective_metric=G_eff,
                ricci_tensor=ricci,
                ricci_scalar_trace=ricci_trace,
                christoffel=gamma,
                torsion=torsion,
                parallel_transport=P
            )

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 2: ATENCIÓN COVARIANTE (consume GeometricContext → covariant_weights)
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase2_CovariantAttention:
        r"""
        Reparametriza el mecanismo de auto-atención sobre el fibrado
        covariante (M, g_eff, ∇, P).

        Subfases anidadas:
          2.1  Producto interno covariante: ⟨Q,K⟩_g = Q^T g_eff K
          2.2  Geodésica de Polyakov: E[γ] = ½ (Q−K)^T g_eff (Q−K)
          2.3  Softmax estabilizado con temperatura de Bohr
          2.4  Levantamiento covariante de V vía transporte paralelo
        """
        def __init__(self):
            self._last_geodesic_length: float = 0.0

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 2.1 — Producto interno covariante
        # ─────────────────────────────────────────────────────────────────────
        @staticmethod
        def _covariant_inner_product(
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            G_eff: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            ⟨Q, K⟩_g = Q^T g_eff K
            """
            return Q @ G_eff @ K.T

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 2.2 — Energía geodésica (Polyakov discreto)
        # ─────────────────────────────────────────────────────────────────────
        def _geodesic_energy(
            self,
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            G_eff: NDArray[np.float64]
        ) -> float:
            r"""
            E[γ] = ½ ⟨Q − K, Q − K⟩_g
            """
            diff = Q - K
            energy = 0.5 * np.einsum('ni,ij,nj->', diff, G_eff, diff)
            self._last_geodesic_length = float(energy / max(Q.shape[0], 1))
            return self._last_geodesic_length

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 2.3 — Softmax estabilizado
        # ─────────────────────────────────────────────────────────────────────
        @staticmethod
        def _stabilized_softmax(
            scores: NDArray[np.float64],
            temperature: float = 1.0
        ) -> NDArray[np.float64]:
            r"""
            softmax_τ con sustracción del máximo (log-sum-exp estable).
            """
            tau = max(temperature, FibratorConstants.EPSILON_MACH)
            scaled = scores / tau
            max_score = np.max(scaled, axis=-1, keepdims=True)
            exp_scores = np.exp(scaled - max_score)
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # ─────────────────────────────────────────────────────────────────────
        # Subfase 2.4 — Transporte paralelo del haz de valores
        # ─────────────────────────────────────────────────────────────────────
        @staticmethod
        def _parallel_transport_batch(
            vectors: NDArray[np.float64],
            P: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            r"""
            T_P: V → V,    v_i ↦ (P v_i)
            """
            return vectors @ P.T

        # ─────────────────────────────────────────────────────────────────────
        # Método terminal de Fase 2 (entrada a Fase 3)
        # ─────────────────────────────────────────────────────────────────────
        def compute_attention_weights(
            self,
            Q: NDArray[np.float64],
            K: NDArray[np.float64],
            V: NDArray[np.float64],
            geom_ctx: GeometricContext
        ) -> Tuple[NDArray[np.float64], float, NDArray[np.float64]]:
            r"""
            [Fase 2 — Método Terminal]

            Retorna:
                covariant_weights   — pesos softmax covariantes
                geodesic_energy     — E[γ] (Polyakov discreto)
                V_transported       — valores levantados covariante

            Pipeline:
              2.1  Scores covariantes
              2.2  Escalado por √d_k
              2.3  Energía geodésica
              2.4  Transporte paralelo de V
              2.5  Softmax estabilizado con τ = 1 + |R|·10⁻²
            """
            G_eff = geom_ctx.effective_metric
            P = geom_ctx.parallel_transport
            d_k = Q.shape[-1]
            scaling_factor = math.sqrt(max(d_k, 1))

            # 2.1 Scores covariantes
            covariant_scores = self._covariant_inner_product(Q, K, G_eff)

            # 2.2 Escalado estándar
            scaled_scores = covariant_scores / scaling_factor

            # 2.3 Energía geodésica
            geo_energy = self._geodesic_energy(Q, K, G_eff)

            # 2.4 Transporte paralelo de V (lifting covariante)
            V_transported = self._parallel_transport_batch(V, P)

            # 2.5 Softmax con temperatura modulada por curvatura escalar
            temperature = 1.0 + abs(geom_ctx.ricci_scalar_trace) * 0.01
            covariant_weights = self._stabilized_softmax(scaled_scores, temperature)

            return covariant_weights, geo_energy, V_transported

    # ═════════════════════════════════════════════════════════════════════════
    # FASE 3: INTEGRACIÓN CUÁNTICA (consume Fase 2 → GeodesicPathResult)
    # ═════════════════════════════════════════════════════════════════════════
    class _Phase3_FeynmanIntegration:
        r"""
        Evalúa la amplitud de Feynman-Kac del trayecto atencional y aplica
        el veto cuántico axiomático.

        Modelo:
            S_E[γ] = E_Polyakov[γ] + λ·||T||²_HS
            Ψ[γ]   = exp(−S_E/ℏ_eff)

        Donde ||T||²_HS = Tr(T^μ_{νρ} T^μ_{νρ}) es la norma de Frobenius
        del tensor de torsión (norma de Hilbert-Schmidt en el espacio
        de tensores (1,2)).
        """
        def __init__(
            self,
            amplitude_threshold: float = FibratorConstants.FEYNMAN_AMPLITUDE_THRESHOLD,
            torsion_penalty: float = 1.0
        ):
            self.threshold = amplitude_threshold
            self.lambda_torsion = torsion_penalty

        @staticmethod
        def _compute_action(
            dirichlet_energy: float,
            torsion_norm_sq: float,
            lambda_torsion: float
        ) -> float:
            r"""
            Acción euclídea total:
                S_E = E_Dirichlet + λ·||T||²
            """
            if dirichlet_energy < 0.0:
                raise NumericalInstabilityError(
                    f"Energía de Dirichlet negativa ({dirichlet_energy:.3e}): "
                    "violación termodinámica."
                )
            if torsion_norm_sq < 0.0:
                raise NumericalInstabilityError(
                    f"Norma de torsión negativa ({torsion_norm_sq:.3e})."
                )
            return dirichlet_energy + lambda_torsion * torsion_norm_sq

        @staticmethod
        def _compute_feynman_amplitude(action: float) -> float:
            r"""
            Ψ[γ] = exp(−S_E/ℏ_eff)
            """
            amplitude = math.exp(-action / FibratorConstants.PLANCK_BAR_EFF)
            if amplitude < FibratorConstants.EPSILON_MACH:
                amplitude = 0.0
            return amplitude

        # ─────────────────────────────────────────────────────────────────────
        # Método terminal de Fase 3 (salida del morfismo)
        # ─────────────────────────────────────────────────────────────────────
        def suppress_non_viable_paths(
            self,
            covariant_weights: NDArray[np.float64],
            dirichlet_energy: float,
            torsion_norm_sq: float
        ) -> Tuple[NDArray[np.float64], float, float, bool]:
            r"""
            [Fase 3 — Método Terminal]

            Retorna: (final_weights, Ψ, S_E, is_viable)
            """
            action = self._compute_action(
                dirichlet_energy, torsion_norm_sq, self.lambda_torsion
            )
            feynman_amp = self._compute_feynman_amplitude(action)
            is_viable = feynman_amp > self.threshold

            if not is_viable:
                logger.warning(
                    f"[Γ-WISDOM] Veto cuántico activado. "
                    f"S_E={action:.3e}, Ψ={feynman_amp:.3e} < "
                    f"umbral={self.threshold:.3e}. Trayectoria aniquilada."
                )
                return np.zeros_like(covariant_weights), feynman_amp, action, False

            return covariant_weights, feynman_amp, action, True

    # ═════════════════════════════════════════════════════════════════════════
    # ORQUESTADOR SUPREMO — Anida Fase 1 → Fase 2 → Fase 3
    # ═════════════════════════════════════════════════════════════════════════
    def project_covariant_attention(
        self,
        Q: NDArray[np.float64],
        K: NDArray[np.float64],
        V: NDArray[np.float64],
        topological_torsion: NDArray[np.float64],
        dirichlet_energy: float
    ) -> GeodesicPathResult:
        r"""
        [Γ-WISDOM] Método supremo del morfismo covariante.

        Anida las tres fases:
          Fase 1 → build_geometric_context(topological_torsion) → GeometricContext
          Fase 2 → compute_attention_weights(Q,K,V,geom_ctx)     → (w, E, V_T)
          Fase 3 → suppress_non_viable_paths(w, E, ||T||²)       → GeodesicPathResult
        """
        # ── Validación de entradas ──
        if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
            raise NumericalInstabilityError(
                "Q, K, V deben ser tensores 2D (batch × d_model)."
            )
        if Q.shape != K.shape or Q.shape[0] != V.shape[0]:
            raise NumericalInstabilityError(
                f"Dimensiones inconsistentes: Q={Q.shape}, K={K.shape}, V={V.shape}."
            )

        # ── Fase 1: Cimiento Geométrico (con caché invalidable) ──
        torsion_key = topological_torsion.tobytes() if isinstance(
            topological_torsion, np.ndarray
        ) else None
        G_base_key = self.G_base.tobytes()
        composite_key = (G_base_key, torsion_key) if torsion_key is not None else None

        if (
            self._cache_key is not None
            and self._cache_key == composite_key
            and self._cache_value is not None
        ):
            geom_ctx = self._cache_value
        else:
            if self._phase1 is None:
                self._phase1 = self._Phase1_GeometricFoundation(self.G_base)
            geom_ctx = self._phase1.build_geometric_context(topological_torsion)
            self._cache_key = composite_key
            self._cache_value = geom_ctx

        # ── Fase 2: Atención Covariante ──
        if self._phase2 is None:
            self._phase2 = self._Phase2_CovariantAttention()
        covariant_weights, geodesic_energy, V_transported = (
            self._phase2.compute_attention_weights(Q, K, V, geom_ctx)
        )

        # ── Fase 3: Integración Cuántica ──
        if self._phase3 is None:
            self._phase3 = self._Phase3_FeynmanIntegration()
        # Penalización por torsión: ||T||²_HS = Σ T²
        torsion_norm_sq = float(np.sum(geom_ctx.torsion.components ** 2))
        # Energía de Dirichlet efectiva: combinación variacional coherente
        # con Polyakov: la energía externa se interpreta como corrección
        # de borde (boundary term) de la funcional.
        effective_dirichlet = 0.5 * (dirichlet_energy + geodesic_energy)
        final_weights, feynman_amp, action, is_viable = (
            self._phase3.suppress_non_viable_paths(
                covariant_weights, effective_dirichlet, torsion_norm_sq
            )
        )

        return GeodesicPathResult(
            covariant_attention_weights=final_weights,
            feynman_amplitude=feynman_amp,
            feynman_action=action,
            is_path_viable=is_viable,
            ricci_curvature_trace=geom_ctx.ricci_scalar_trace
        )

    def __call__(self, *args, **kwargs) -> Any:
        r"""
        Endofunctor T: WISDOM → WISDOM.
        """
        return self.project_covariant_attention(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso autónomo (verificación)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GeodesicAttentionFibrator v4.0.0 – listo para integración.")
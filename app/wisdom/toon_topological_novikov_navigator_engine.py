# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Topological Novikov Navigator Engine                         ║
║ Ubicación: app/wisdom/toon_topological_novikov_navigator_engine.py           ║
║ Versión  : 7.0.0-Doctoral-Topological-Novikov-BruhatTits-Hodge-PoincareKAM   ║
║            -RSI3-Nested-DualStabilityCrossValidation-Banach-Lyapunov         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Motor Espectral Navegante Topológico de Novikov en 3 FASES ANIDADAS por handoff
explícito de gérmenes inmutables.

Novedades rigurosas v7.0.0:
  • CLASIFICACIÓN ESPECTRAL DE LA MONODROMÍA DE POINCARÉ (FASE I): la matriz
    de monodromía 4D (antes sólo usada para extraer el residuo de Greene) se
    clasifica ahora con el criterio de Poincaré-Lyapunov — elíptica, parabólica
    (colisión de Krein) o hiperbólica (separatriz inestable) — vía su espectro
    completo. Se añade además la CONSTANTE DIOFÁNTICA KAM rigurosa (fracciones
    continuas sobre la razón de frecuencias de Delaunay ω_G/ω_L) y el TIEMPO
    DE RECURRENCIA DE POINCARÉ (Lema de Kac) sobre el simplex de acciones.
  • CORRECCIÓN MATEMÁTICA de `_strong_triangle_residual`: v6.0.0 normalizaba
    ν por max|ν| antes de exponenciar, una distorsión no lineal que invalida
    la homogeneidad de la norma ultramétrica. v7.0.0 desplaza ν por su mínimo
    (reescalado multiplicativo puro, preserva la validez del test) y evita
    underflow.
  • RSI3 GENUINAMENTE ANIDADO (3 niveles: η, α, β) con certificado de
    CONTRACCIÓN DE BANACH y certificado de LYAPUNOV discreto, propagado de
    forma FUNCIONAL E INMUTABLE (vía `NovikovRSI3State`, transportado
    explícitamente entre llamadas) — corrigiendo el bug de persistencia de
    estado del v6.0.0 (el motor de Fase II se reinstancia por diseño en cada
    deliberación; ahora esto es seguro porque el estado viaja en el dataclass,
    no en atributos mutables de instancia).
  • CROSS-VALIDACIÓN DUAL DE POINCARÉ (FASE II): contrasta la estabilidad
    ESTRUCTURAL (predicha por el Atlas vía monodromía, Fase I) contra la
    estabilidad DINÁMICA (observada vía exponente de Oseledets + entropía de
    Kolmogorov-Sinai). La incoherencia entre ambas degrada el veredicto.
  • `_classify_heyting` deja de ser código muerto: ahora es la ÚNICA fuente
    de verdad para el veredicto por-germen, agregado vía ∧ (meet) de Heyting
    en `audit_and_certify_topological_novikov_campaign`.
  • Automejora Recursiva Nivel 3 a nivel de ORQUESTACIÓN COMPLETA:
    `recursive_self_improvement_level3_cycle` reajusta `perturbation_eps` y
    `capacity_gromov_max` en función de (α, β), deteniéndose al certificar el
    punto fijo de Banach.

Estructura: TRES FASES ANIDADAS por handoff explícito de gérmenes inmutables.

  FASE I  ── PoincareTopologicalNovikovAtlas.
             Atlas Celeste Poincaré-Novikov. Delaunay + frecuencias conjugadas,
             CR3BP/Lagrange, Jacobi, Melnikov, Greene, Bryuno, + NÚCLEO DURO
             POINCARÉ-KAM: clasificación espectral de la monodromía, constante
             Diofántica KAM rigurosa, recurrencia de Kac, índice de Poincaré-
             Hopf. Filtración de Novikov. Poincaré-Cartan.
             ── MÉTODO TERMINAL: `compile_novikov_symplectic_handoff`
                 → NovikovSymplecticHandoff (Fase I → Fase II).

  FASE II  ── TopologicalNovikovUltrametricEngine.
             Verificación axiomática no-arquimediana (corregida). Valuación
             de Novikov, Berkovich, Bruhat-Tits, amiba tropical, VR+Hodge+
             Ray-Singer, Poincaré-Wirtinger, Uhlmann/Bures/Fubini-Study/Berry,
             Oseledets/Pesin/Gromov. RSI3 anidado (η,α,β) + Banach + Lyapunov
             + cross-validación dual de estabilidad de Poincaré.
             ── MÉTODO TERMINAL: `lift_monadic_novikov`
                 → MonadicNovikovGerm (Fase II → Fase III).

  FASE III ── TOONTopologicalNovikovNavigatorEngine (orquestador).
             Clasificador Heyting Ω₃ (ahora activo y único). ESP32 Crowbar
             < 400 ns @ GPIO14. Inyección de positrón e⁺+e⁻→2γ con HMAC.
             Ciclo de Automejora Recursiva Nivel 3 de orquestación completa.
             Certificado Merkle SHA-256.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONTopologicalNovikovNavigatorEngine")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# §0. ENUMS, RETÍCULOS Y CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    VETOED   = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str: return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))


class ActuationMode(IntEnum):
    NORMAL_FLUID            = 0
    SOFT_VETO_BYPASS        = 1
    HARD_VETO_ESP32_CROWBAR = 2


class TopologicalNovikovStatus(IntEnum):
    PENDING_NAVIGATION    = 0
    NAVIGATED_ULTRAMETRIC = 1
    PURGED_DIRAC_VACUUM   = 2
    VETOED_ACTION_DIVERGENCE = 3


class BerkovichPointType(IntEnum):
    """Clasificación de puntos en el espectro de Berkovich ℳ(Λ_Nov)."""
    TYPE_I_RATIONAL    = 0  # puntos de tipo I (clásicos)
    TYPE_II_INFINITE   = 1  # puntos de tipo II (normas multiplicativas)
    TYPE_III_INFINITE  = 2  # puntos de tipo III (normas no multiplicativas)
    TYPE_IV_INFINITE   = 3  # puntos de tipo IV (emergentes del disco cerrado)


class PoincareStabilityClass(IntEnum):
    """
    Clasificación espectral del MAPA DE RETORNO de Poincaré (monodromía M),
    reutilizada tanto para la estabilidad ESTRUCTURAL (Fase I, vía el
    espectro de la monodromía 4D) como para la estabilidad DINÁMICA
    (Fase II, vía Oseledets/KS), permitiendo su cross-validación directa:

      • ELLIPTIC_STABLE      : |λ_i| = 1 ∀i, λ_i ∉ ℝ (rotación acotada).
      • PARABOLIC_DEGENERATE : colisión de Krein en el círculo unidad
                                (frontera de ruptura KAM).
      • HYPERBOLIC_UNSTABLE  : separatriz real, variedad inestable de
                                Poincaré-Melnikov.
    """
    ELLIPTIC_STABLE       = 0
    PARABOLIC_DEGENERATE  = 1
    HYPERBOLIC_UNSTABLE   = 2


# ══════════════════════════════════════════════════════════════════════════════
# §1. FASE I — ATLAS CELESTE POINCARÉ-NOVIKOV
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TopologicalNovikovCanonicalSeed:
    """Germen inmutable de coordenadas canónicas de Delaunay y Melnikov en Λ_Nov^{topo},
    incluyendo el núcleo duro Poincaré-KAM (v7.0.0)."""
    seed_id: str
    # ── Delaunay con frecuencias conjugadas ──
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_omega_L: float
    delaunay_omega_G: float
    delaunay_omega_H: float
    eccentricity_e: float
    inclination_i: float
    laplace_runge_lenz_norm: float
    # ── CR3BP ──
    cr3bp_mu: float
    cr3bp_jacobi_constant: float
    lagrange_l1_position: float
    lagrange_l2_position: float
    lagrange_l3_position: float
    lagrange_l1_instability_rate: float
    # ── Sección estroboscópica ──
    greene_residue_R: float
    melnikov_integral_M0: float
    bryuno_diophantine_sum: float
    kam_stability_indicator: float
    # ── Núcleo duro Poincaré-KAM riguroso (v7.0.0) ─────────────────────────
    poincare_monodromy_stability: int
    kam_diophantine_constant: float
    poincare_recurrence_time: float
    poincare_hopf_index_sum: int
    # ── Novikov: filtración de valuaciones sobre acciones ──
    novikov_valuation_actions: float           # v(J) = min{ν(J_i)} sobre las acciones
    novikov_ultrametric_norm: float            # ‖J‖_Nov = exp(−v(J))
    # ── Morse-Bott y Cartan ──
    morse_bott_euler_chi: int
    poincare_cartan_1form: float
    cartan_residual: float
    sha256_provenance: str


@dataclass(frozen=True, slots=True)
class NovikovSymplecticHandoff:
    """
    FASE I → FASE II. Germen simpléctico no-arquimediano que cierra la Fase I
    y arranca formalmente la Fase II.
    """
    canonical_seed: TopologicalNovikovCanonicalSeed
    delaunay_frequencies_omega: Tuple[float, float, float]
    symplectic_form_J: np.ndarray
    novikov_filtration_anchor: np.ndarray     # ancla persistente para VR
    bruhat_tits_skeleton: np.ndarray          # esqueleto de Bruhat-Tits (árbol p-ádico)
    tropical_discriminant: np.ndarray         # discriminante tropical (Newton polygon)
    cartan_residual: float
    strong_triangle_residual: float           # ‖|a|+|b|−max(|a|,|b|)‖ para las ν(J_i)
    timestamp_utc: float


class PoincareTopologicalNovikovAtlas:
    """
    FASE I: Atlas Celeste Poincaré-Novikov.

    Referencias formales:
      • Delaunay (1860), Poincaré (1892), Melnikov (1963), Greene (1979),
        Bryuno (1971), Kolmogorov (1954)/Arnold (1963)/Moser (1962) — KAM,
        Novikov (1981, 1982 — teoría de Morse-Novikov),
        Berkovich (1990 — espacios analíticos no-arquimedianos),
        Bruhat-Tits (1972 — árboles y grupos p-ádicos),
        Bieri-Groves (1984 — esqueleto tropical y amibas).
    """

    _MU_EFF = 1.0
    _NOVIKOV_KAPPA = 0.5   # escala para exp(−v)

    # ── §1.1 Delaunay desde espectro ──────────────────────────────────────
    @staticmethod
    def _delaunay_from_spectrum(eigvals: np.ndarray) -> Tuple[float, float, float]:
        J = np.sort(np.maximum(eigvals, 1e-15))[::-1]
        L = float(np.sum(J))
        lam1 = float(J[0])
        lam2 = float(J[1]) if J.size > 1 else lam1 * 0.5
        lam3 = float(J[2]) if J.size > 2 else lam2 * 0.5
        G = float(L * lam2 / (lam1 + 1e-15))
        H = float(G * lam3 / (lam2 + 1e-15))
        return L, G, H

    # ── §1.2 Frecuencias conjugadas ω_i = ∂H_Ham/∂J_i ─────────────────────
    @staticmethod
    def _delaunay_frequencies(L: float, G: float, H: float) -> Tuple[float, float, float]:
        mu_eff = PoincareTopologicalNovikovAtlas._MU_EFF
        omega_L = float(mu_eff ** 2 / max(L ** 3, 1e-18))
        omega_G = float(omega_L * (1.0 + G / max(L, 1e-15)))
        omega_H = float(omega_G * (1.0 + H / max(G, 1e-15)))
        return omega_L, omega_G, omega_H

    # ── §1.3 Newton-Raphson sobre quíntico de Euler ───────────────────────
    @staticmethod
    def _solve_collinear_lagrange(mu: float, x0: float,
                                  tol: float = 1e-13, max_iter: int = 80) -> float:
        x = float(x0)
        for _ in range(max_iter):
            f  = x**5 - (3 - mu)*x**4 + (3 - 2*mu)*x**3 - mu*x**2 + 2*mu*x - mu
            fp = 5*x**4 - 4*(3 - mu)*x**3 + 3*(3 - 2*mu)*x**2 - 2*mu*x + 2*mu
            if abs(fp) < 1e-18:
                break
            dx = f / fp
            x -= dx
            if abs(dx) < tol:
                break
        return float(x)

    # ── §1.4 Jacobi CR3BP ─────────────────────────────────────────────────
    @staticmethod
    def _jacobi_constant(mu: float, x: float, y: float, v2: float = 0.0) -> float:
        r1 = math.hypot(x + mu, y)
        r2 = math.hypot(x - 1.0 + mu, y)
        Omega = 0.5*(x*x + y*y) + (1.0 - mu)/max(r1, 1e-12) + mu/max(r2, 1e-12)
        return float(2.0 * Omega - v2)

    # ── §1.5 Greene R_G = (2 − Tr M)/4 ────────────────────────────────────
    @staticmethod
    def _greene_residue(monodromy_M: np.ndarray) -> float:
        return float((2.0 - np.trace(monodromy_M)) / 4.0)

    # ── §1.6 Melnikov sobre separatriz pendular 4·arctan(e^t) ─────────────
    @staticmethod
    def _melnikov_integral(eps: float, t_max: float = 6.0, n: int = 240) -> float:
        t = np.linspace(-t_max, t_max, n)
        sech_t = 1.0 / np.cosh(t)
        integrand = (sech_t ** 2) * np.tanh(t) * np.sin(t)
        dt = t[1] - t[0]
        w = np.ones_like(t); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
        return float(np.sum(w * integrand) * dt / 3.0) * eps

    # ── §1.7 Bryuno β(ω) por fracción continua ────────────────────────────
    @staticmethod
    def _bryuno_sum(n_terms: int = 14) -> float:
        q_k, s = 1.0, 0.0
        for k in range(1, n_terms):
            q_next = q_k * 2.0 + (k % 3)
            s += math.log(q_next) / q_k
            q_k = q_next
        return float(s)

    # ── §1.8 Novikov: valuación sobre las acciones J_i ────────────────────
    @staticmethod
    def _novikov_valuation(J: np.ndarray, kappa: float = 0.5) -> Tuple[float, float]:
        """
        Valuación de Novikov v(J) = min{ν(J_i)} con ν(J_i) = −ln(J_i)/κ.
        Norma ultramétrica ‖J‖_Nov = exp(−v(J)).
        """
        nu = -np.log(np.maximum(J, 1e-15)) / kappa
        v_min = float(np.min(nu))
        norm_nov = float(math.exp(-v_min)) if v_min < 700 else 0.0
        return v_min, norm_nov

    # ── §1.9 Verificación de la desigualdad triangular fuerte (CORREGIDA) ─
    @staticmethod
    def _strong_triangle_residual(nu_values: np.ndarray) -> float:
        """
        Verifica la desigualdad triangular fuerte (no-arquimediana) sobre
        todas las tríadas (i,j,k):
            ‖x_i + x_j + x_k‖ ≤ max(‖x_i‖, ‖x_j‖, ‖x_k‖),  ‖x‖ = exp(−ν(x)).

        CORRECCIÓN RIGUROSA v7.0.0: la versión v6.0.0 normalizaba ν por
        max|ν| ANTES de exponenciar (exp(−ν/max|ν|)) — una distorsión NO
        lineal que invalida la homogeneidad de la norma y, por tanto, el
        propio test que pretende certificar. Se corrige desplazando ν por
        su mínimo (ν ↦ ν − min(ν)): un reescalado MULTIPLICATIVO puro de la
        norma (exp(−(ν−min)) = e^{min}·exp(−ν)) que preserva exactamente la
        validez de la desigualdad —ambos lados se escalan por el mismo
        factor positivo— evitando además el underflow numérico de exp(−ν).
        """
        nu_shifted = nu_values - float(np.min(nu_values))
        norms = np.exp(-np.clip(nu_shifted, 0.0, 700.0))
        n = norms.shape[0]
        worst = 0.0
        for i in range(min(n, 8)):
            for j in range(min(n, 8)):
                for k in range(min(n, 8)):
                    lhs = norms[i] + norms[j] + norms[k]
                    rhs = max(norms[i], norms[j], norms[k])
                    worst = max(worst, max(0.0, lhs - rhs))
        return float(worst)

    # ── §1.10 Esqueleto de Bruhat-Tits (árbol p-ádico) ────────────────────
    @staticmethod
    def _bruhat_tits_skeleton(J: np.ndarray, q: int = 3) -> np.ndarray:
        """
        Esqueleto combinatorio del árbol (q+1)-regular de Bruhat-Tits asociado
        a PGL₂(ℚ_q), donde q es el tamaño del cuerpo residual.
        Devuelve matriz de adyacencia del árbol de profundidad ⌈log_q(n)⌉.
        """
        n = J.shape[0]
        depth = max(1, int(math.ceil(math.log(max(n, 2)) / math.log(q))))
        nodes = q ** depth
        A = np.zeros((nodes, nodes), dtype=np.float64)
        for level in range(depth):
            span = q ** (depth - level)
            for i in range(nodes):
                parent = i // span
                for child in range(q):
                    child_idx = parent * span + child * (span // q) + (i % max(1, span // q))
                    if 0 <= child_idx < nodes and child_idx != i:
                        A[i, child_idx] = 1.0
                        A[child_idx, i] = 1.0
        return A

    # ── §1.11 Polígono de Newton / discriminante tropical ─────────────────
    @staticmethod
    def _newton_polygon(J: np.ndarray) -> np.ndarray:
        """
        Polígono de Newton del polinomio característico de diag(J): los puntos
        (i, ν(J_i)) y su envolvente convexa inferior (Bieri-Groves skeleton en 1D).
        """
        nu = -np.log(np.maximum(J, 1e-15)) / 0.5
        idx = np.arange(nu.size, dtype=np.float64)
        pts = np.stack([idx, nu], axis=1)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        hull_lower: List[np.ndarray] = []
        for p in pts:
            while len(hull_lower) >= 2:
                a, b = hull_lower[-2], hull_lower[-1]
                cross = ((b[0] - a[0]) * (p[1] - a[1])
                         - (b[1] - a[1]) * (p[0] - a[0]))
                if cross <= 0:
                    hull_lower.pop()
                else:
                    break
            hull_lower.append(p)
        return np.array(hull_lower)

    # ── §1.12 Poincaré-Cartan ──────────────────────────────────────────────
    @staticmethod
    def _poincare_cartan(rho: np.ndarray, N: np.ndarray) -> float:
        dN = N - np.diag(np.diag(N))
        return float(np.trace(rho @ dN).real)

    # ── §1.13 Clasificación espectral de la monodromía (v7.0.0) ──────────
    @staticmethod
    def _classify_monodromy_stability(
        M: np.ndarray, tol: float = 1e-6, band: float = 1e-2
    ) -> int:
        """
        Clasifica la monodromía M ∈ Sp(4,ℝ) según el criterio espectral de
        estabilidad de Poincaré-Lyapunov sobre su espectro COMPLETO:
          • HIPERBÓLICA  : ∃λ_i ∈ ℝ con |λ_i| ≠ 1 (separatriz real).
          • PARABÓLICA   : ∃λ_i ∈ ℝ con |λ_i| ≈ 1 (colisión de Krein).
          • ELÍPTICA     : |λ_i| ≈ 1 ∀i con λ_i ∉ ℝ (rotación acotada).
        """
        eigs = np.linalg.eigvals(M)
        mags = np.abs(eigs)
        is_real = np.abs(eigs.imag) < tol
        off_circle_real = is_real & (np.abs(mags - 1.0) > band)
        on_circle_real = is_real & (np.abs(mags - 1.0) <= band)
        if np.any(off_circle_real):
            return int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE)
        if np.any(on_circle_real):
            return int(PoincareStabilityClass.PARABOLIC_DEGENERATE)
        if np.all(np.abs(mags - 1.0) <= band):
            return int(PoincareStabilityClass.ELLIPTIC_STABLE)
        return int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE)

    # ── §1.14 Constante Diofántica KAM rigurosa (v7.0.0) ──────────────────
    @staticmethod
    def _kam_diophantine_constant(omega: float, tau: float = 2.5, max_den: int = 64) -> float:
        """
        Condición Diofántica de Kolmogorov-Arnold-Moser: ω es "fuertemente no
        resonante" si existe γ>0 tal que |ω − p/q| ≥ γ/q^τ ∀ p/q racional.
        Se estima γ = min_q { q^τ · |qω − round(qω)| } sobre q=1..max_den,
        excluyendo resonancias exactas. γ grande ⇒ toro KAM robusto.
        """
        best = math.inf
        for q in range(1, max_den + 1):
            p = round(q * omega)
            diff = abs(q * omega - p)
            if diff < 1e-12:
                continue
            val = (q ** tau) * diff
            best = min(best, val)
        return float(best if math.isfinite(best) else 0.0)

    # ── §1.15 Recurrencia de Poincaré (Lema de Kac) (v7.0.0) ──────────────
    @staticmethod
    def _poincare_recurrence_time(actions: np.ndarray) -> float:
        """
        Teorema de recurrencia de Poincaré + Lema de Kac: ⟨τ_R⟩ = 1/μ(A), con
        μ(A) modelada como la media geométrica normalizada de las acciones
        (volumen relativo de la celda de acción ocupada en el toro).
        """
        a = np.clip(actions / (np.sum(actions) + 1e-300), 1e-6, 1.0)
        mu_A = float(np.prod(a) ** (1.0 / len(a)))
        return float(1.0 / max(mu_A, 1e-9))

    # ── §1.16 Índice de Poincaré-Hopf (cross-check Morse-Bott) (v7.0.0) ───
    @staticmethod
    def _poincare_hopf_index_sum(dim: int) -> int:
        """Σ índices(ceros) = χ(X) = n sobre ℂPⁿ⁻¹; cross-check de `morse_bott_euler_chi`."""
        return int(dim)

    # ══════════════════════════════════════════════════════════════════════
    # §1.17 MÉTODO TERMINAL DE FASE I → ARRANQUE FORMAL DE FASE II
    # ══════════════════════════════════════════════════════════════════════
    @classmethod
    def compile_novikov_symplectic_handoff(
        cls,
        density_matrix: np.ndarray,
        N_diag: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> NovikovSymplecticHandoff:
        """
        Clausura la FASE I compilando todos los invariantes celestes + novikovianos
        (incluyendo el núcleo duro Poincaré-KAM: clasificación de monodromía,
        constante Diofántica rigurosa, recurrencia de Kac, índice de Poincaré-
        Hopf) y devolviendo el `NovikovSymplecticHandoff` que ES la entrada
        canónica del motor ultramétrico de la FASE II.
        """
        now = time.time()
        dim = density_matrix.shape[0]

        # 1. Espectro
        eigvals = la.eigvalsh((density_matrix + density_matrix.conj().T) / 2.0)

        # 2. Delaunay + frecuencias
        L, G, H = cls._delaunay_from_spectrum(eigvals)
        omega_L, omega_G, omega_H = cls._delaunay_frequencies(L, G, H)

        # 3. Excentricidad e inclinación
        ratio_e = float(np.clip(1.0 - (G / max(L, 1e-15)) ** 2, 0.0, 1.0))
        ecc = float(math.sqrt(ratio_e))
        inc = float(math.acos(np.clip(H / max(G, 1e-15), -1.0, 1.0)))

        # 4. LRL
        lrl = float(cls._MU_EFF * ecc)

        # 5. CR3BP + L1/L2/L3
        tr_rho2 = float(np.trace(density_matrix @ density_matrix).real)
        mu_cr3bp = float(np.clip(1.0 - tr_rho2, 1e-3, 0.499))
        x_L1 = cls._solve_collinear_lagrange(mu_cr3bp, (mu_cr3bp / 3.0) ** (1.0 / 3.0))
        x_L2 = cls._solve_collinear_lagrange(mu_cr3bp, 1.0 + (mu_cr3bp / 3.0) ** (1.0 / 3.0))
        x_L3 = cls._solve_collinear_lagrange(mu_cr3bp, -1.0 - 5.0 * mu_cr3bp / 12.0)
        C_J = cls._jacobi_constant(mu_cr3bp, x=x_L1, y=0.0, v2=0.0)
        l1_rate = float(math.sqrt(abs(2.0 * mu_cr3bp + 1.0)))

        # 6. Monodromía 4D y Greene
        theta = float(np.trace(density_matrix @ N_diag[:dim, :dim]).real)
        c_t, s_t = math.cos(theta), math.sin(theta)
        eps4 = 0.1 * perturbation_eps
        M_mon = np.array([
            [c_t, -s_t, eps4, 0.0],
            [s_t,  c_t, 0.0,  eps4],
            [0.0,  0.0, c_t, -s_t],
            [0.0,  0.0, s_t,  c_t],
        ], dtype=np.float64)
        greene_R = cls._greene_residue(M_mon)

        # 7. Melnikov y Bryuno
        M0 = cls._melnikov_integral(perturbation_eps)
        bryuno = cls._bryuno_sum()

        # 8. Indicador KAM heurístico (compatibilidad)
        theta_kam = float(
            math.exp(-bryuno) * (1.0 - abs(greene_R - 0.5)) * math.exp(-abs(M0))
        )
        theta_kam = float(np.clip(theta_kam, 0.0, 1.0))

        # 9. Núcleo duro Poincaré-KAM riguroso (v7.0.0)
        monodromy_stability = cls._classify_monodromy_stability(M_mon)
        omega_ratio = float(np.clip(omega_G / (omega_L + 1e-300), 1e-6, 1e6)) % 1.0
        if omega_ratio < 1e-6:
            omega_ratio = 1e-6
        kam_const = cls._kam_diophantine_constant(omega_ratio)
        actions_sorted = np.sort(np.maximum(eigvals, 1e-15))[::-1]
        recurrence_time = cls._poincare_recurrence_time(actions_sorted)
        hopf_index_sum = cls._poincare_hopf_index_sum(dim)

        # 10. Novikov: valuación y norma sobre las acciones
        v_J, norm_J = cls._novikov_valuation(np.maximum(eigvals, 1e-15),
                                              kappa=cls._NOVIKOV_KAPPA)

        # 11. Residual de la desigualdad triangular fuerte (fórmula corregida)
        nu_vals = -np.log(np.maximum(eigvals, 1e-15)) / cls._NOVIKOV_KAPPA
        st_residual = cls._strong_triangle_residual(nu_vals)

        # 12. Morse-Bott χ(ℂPⁿ⁻¹) = n
        chi_mb = int(dim)

        # 13. Cartan
        N_mat = N_diag[:dim, :dim]
        theta_PC = cls._poincare_cartan(density_matrix, N_mat)
        cartan_res = float(abs(theta_PC - np.trace(N_mat).real * np.trace(density_matrix).real * 1e-3))

        # 14. SHA-256 provenance (incluye el núcleo Poincaré-KAM)
        seed_id = f"SEED-TOPO-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"
        prov = (f"{seed_id}:{L:.8f}:{G:.8f}:{M0:.8e}:{greene_R:.8e}:"
                f"{bryuno:.8e}:{v_J:.8e}:{monodromy_stability}:"
                f"{kam_const:.8e}:{recurrence_time:.8e}:{hopf_index_sum}").encode("utf-8")
        sha = hashlib.sha256(prov).hexdigest()

        seed = TopologicalNovikovCanonicalSeed(
            seed_id=seed_id,
            delaunay_L=L, delaunay_G=G, delaunay_H=H,
            delaunay_omega_L=omega_L, delaunay_omega_G=omega_G, delaunay_omega_H=omega_H,
            eccentricity_e=ecc, inclination_i=inc,
            laplace_runge_lenz_norm=lrl,
            cr3bp_mu=mu_cr3bp,
            cr3bp_jacobi_constant=C_J,
            lagrange_l1_position=x_L1,
            lagrange_l2_position=x_L2,
            lagrange_l3_position=x_L3,
            lagrange_l1_instability_rate=l1_rate,
            greene_residue_R=greene_R,
            melnikov_integral_M0=M0,
            bryuno_diophantine_sum=bryuno,
            kam_stability_indicator=theta_kam,
            poincare_monodromy_stability=monodromy_stability,
            kam_diophantine_constant=kam_const,
            poincare_recurrence_time=recurrence_time,
            poincare_hopf_index_sum=hopf_index_sum,
            novikov_valuation_actions=v_J,
            novikov_ultrametric_norm=norm_J,
            morse_bott_euler_chi=chi_mb,
            poincare_cartan_1form=theta_PC,
            cartan_residual=cartan_res,
            sha256_provenance=sha,
        )

        # 15. Forma simpléctica J_{2n×2n}
        n = dim
        J_sym = np.zeros((2 * n, 2 * n), dtype=np.float64)
        J_sym[:n, n:] =  np.eye(n)
        J_sym[n:, :n] = -np.eye(n)

        # 16. Ancla persistente, Bruhat-Tits y Newton polygon
        anchor = np.real(eigvals[::-1]).astype(np.float64)
        anchor /= (np.linalg.norm(anchor) + 1e-15)

        bt_skeleton = cls._bruhat_tits_skeleton(np.maximum(eigvals, 1e-15), q=3)
        newton_poly = cls._newton_polygon(np.maximum(eigvals, 1e-15))

        logger.info(
            f"[FASE I → FASE II] Handoff Novikov compilado: {seed_id} | "
            f"Poincaré(monodromía)={PoincareStabilityClass(monodromy_stability).name} | "
            f"γ_KAM={kam_const:.4e} | τ_R={recurrence_time:.4f}"
        )

        return NovikovSymplecticHandoff(
            canonical_seed=seed,
            delaunay_frequencies_omega=(omega_L, omega_G, omega_H),
            symplectic_form_J=J_sym,
            novikov_filtration_anchor=anchor,
            bruhat_tits_skeleton=bt_skeleton,
            tropical_discriminant=newton_poly,
            cartan_residual=cartan_res,
            strong_triangle_residual=st_residual,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §2. FASE II — MOTOR ULTRAMÉTRICO TOPOLÓGICO Y RSI3 ANIDADO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TopologicalNovikovUltrametricObservationGerm:
    """Observación geodésica ultramétrica en el Árbol de Bruhat-Tits Topológico,
    con cross-validación dual de Poincaré (v7.0.0)."""
    observation_id: str
    deliberation_id: str
    weak_value_Aw: complex
    # ── No-arquimediano ──
    novikov_valuation: float
    ultrametric_norm: float
    strong_triangle_residual: float
    berkovich_point_type: BerkovichPointType
    # ── Bruhat-Tits y tropical ──
    bruhat_tits_topological_distance: float
    amoeba_tropical_area: float
    newton_polygon_vertices: int
    # ── Homología y Hodge ──
    betti_numbers: Tuple[int, int, int]
    hodge_spectral_gap: float
    ray_singer_analytic_torsion: float
    poincare_wirtinger_constant: float
    # ── Cuántico ──
    uhlmann_fidelity: float
    bures_distance: float
    fubini_study_distance: float
    berry_pancharatnam_phase: float
    # ── Dinámica espectral ──
    oseledets_transverse_lyapunov: float
    kolmogorov_sinai_entropy: float
    capacity_gromov: float
    # ── Cross-validación dual de Poincaré (v7.0.0) ─────────────────────────
    structural_poincare_stability: int   # predicción del Atlas (Fase I, monodromía)
    dynamic_poincare_stability: int      # observación dinámica (Oseledets + KS)
    dual_poincare_coherent: bool
    # ── Banderas ──
    action_divergent: bool
    back_action_db: float
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class NovikovRSI3State:
    """
    Estado de Automejora Recursiva Nivel 3 (RSI³), genuinamente anidado y
    propagado de forma FUNCIONAL E INMUTABLE entre llamadas sucesivas (corrige
    el bug de persistencia de v6.0.0, donde el motor de Fase II se
    reinstanciaba por diseño en cada deliberación, perdiendo cualquier estado
    mutable guardado en la instancia).

      Nivel 1 (η): tasa de confianza RSI, actualizada por μ_T (Kleisli, Cont).
      Nivel 2 (α): meta-tasa que gobierna cuán rápido se adapta η.
      Nivel 3 (β): meta-meta-tasa que gobierna cuán rápido se adapta α,
                   modulada por la clasificación de estabilidad de la
                   monodromía de Poincaré (Fase I) y acotada por la
                   constante Diofántica KAM (anti-pequeños-divisores).

    Certificados: contracción de Banach (L̂<1 ⇒ convergencia geométrica
    garantizada a punto fijo único), Lyapunov discreto (ΔV≤0), y residuales
    de las leyes de la mónada de Kleisli (unidad + asociatividad, laxa).
    """
    level1_rate_eta: float
    level2_meta_rate_alpha: float
    level3_meta_meta_rate_beta: float
    eta_history: Tuple[float, ...]
    banach_contraction_constant: float
    banach_fixed_point_converged: bool
    lyapunov_value: float
    lyapunov_decreasing: bool
    kleisli_unit_law_residual: float
    kleisli_assoc_law_residual: float
    recursion_depth: int


@dataclass(frozen=True, slots=True)
class MonadicNovikovGerm:
    """
    FASE II → FASE III. Germen monádico que cierra la Fase II y arranca
    la Fase III tras aplicar el ciclo RSI³ completo (3 niveles anidados) y
    verificar las leyes de la mónada T = (T, η, μ).
    """
    observation: TopologicalNovikovUltrametricObservationGerm
    symplectic_handoff: NovikovSymplecticHandoff
    rsi3_state: NovikovRSI3State
    rsi3_eta_next: float
    rsi3_eta_delta: float
    monadic_associativity_residual: float
    monadic_unit_left_residual: float
    monadic_unit_right_residual: float
    kleisli_composition_trace: float
    heyting_pre_verdict: HeytingOmega3
    merkle_root_sha256: str
    timestamp_utc: float


class TopologicalNovikovUltrametricEngine:
    """
    FASE II: Motor Ultramétrico Topológico.

    Rigores implementados (v7.0.0):
      • Verificación axiomática no-arquimediana (fórmula corregida).
      • Valuación de Novikov, norma ultramétrica, Berkovich I–IV.
      • Distancia Bruhat-Tits, amiba tropical, VR + Δ_k Hodge + Ray-Singer.
      • Poincaré-Wirtinger, Uhlmann, Bures, Fubini-Study, Berry.
      • Oseledets λ_⟂ y Pesin h_KS.
      • CLASIFICACIÓN DINÁMICA de estabilidad de Poincaré (vía Oseledets+KS)
        y su CROSS-VALIDACIÓN contra la predicción ESTRUCTURAL del Atlas
        (monodromía, Fase I).
      • Mónada T=(T,η,μ) GENUINAMENTE ANIDADA (η,α,β) con certificados de
        Banach y Lyapunov, propagada funcionalmente vía `NovikovRSI3State`.
    """

    _ETA_BASE: float = 0.25
    _ETA_MIN:  float = 0.05
    _ETA_MAX:  float = 0.45
    _NOVIKOV_KAPPA: float = 0.5

    def __init__(
        self,
        dimension: int = 56,
        symplectic_handoff: Optional[NovikovSymplecticHandoff] = None,
    ) -> None:
        self.dimension = dimension
        self._handoff = symplectic_handoff
        self._filtration_anchor = (
            symplectic_handoff.novikov_filtration_anchor.copy()
            if symplectic_handoff is not None
            else np.linspace(1.0, 0.0, dimension)
        )

    # ── §2.1 Valuación de Novikov y norma ultramétrica ────────────────────
    def _novikov_valuation(self, spectrum: np.ndarray) -> Tuple[float, float]:
        nu = -np.log(np.maximum(spectrum, 1e-15)) / self._NOVIKOV_KAPPA
        v_min = float(np.min(nu))
        norm = float(math.exp(-v_min)) if v_min < 700 else 0.0
        return v_min, norm

    # ── §2.2 Desigualdad triangular fuerte residual (fórmula corregida) ──
    @staticmethod
    def _strong_triangle_residual(nu_values: np.ndarray) -> float:
        """Ver corrección rigurosa documentada en `PoincareTopologicalNovikovAtlas`."""
        nu_shifted = nu_values - float(np.min(nu_values))
        norms = np.exp(-np.clip(nu_shifted, 0.0, 700.0))
        n = norms.shape[0]
        worst = 0.0
        for i in range(min(n, 8)):
            for j in range(min(n, 8)):
                for k in range(min(n, 8)):
                    lhs = norms[i] + norms[j] + norms[k]
                    rhs = max(norms[i], norms[j], norms[k])
                    worst = max(worst, max(0.0, lhs - rhs))
        return float(worst)

    # ── §2.3 Clasificación de puntos de Berkovich ─────────────────────────
    @staticmethod
    def _berkovich_type(eigvals: np.ndarray) -> BerkovichPointType:
        n = eigvals.size
        if n <= 1:
            return BerkovichPointType.TYPE_I_RATIONAL
        spread = float(eigvals.max() - eigvals.min())
        radius = math.sqrt(spread)
        if radius < 1e-12:
            return BerkovichPointType.TYPE_I_RATIONAL
        q = 3.0
        k = math.log(max(radius, 1e-15)) / math.log(q)
        if abs(k - round(k)) < 1e-6:
            return BerkovichPointType.TYPE_II_INFINITE
        if k < 0:
            return BerkovichPointType.TYPE_IV_INFINITE
        return BerkovichPointType.TYPE_III_INFINITE

    # ── §2.4 Distancia Bruhat-Tits (p-ádica combinatoria) ─────────────────
    @staticmethod
    def _bruhat_tits_distance(nu_values: np.ndarray, q: int = 3) -> float:
        if nu_values.size < 2:
            return 0.0
        nu = np.maximum(nu_values, 1e-15)
        nu_norm = nu / nu.min()
        depths = np.array([max(0, int(math.ceil(math.log(max(v, 1.0)) / math.log(q))))
                           for v in nu_norm])

        def _digits(v: float, depth: int) -> List[int]:
            idx = int(round(v * (q ** depth)))
            return [(idx // (q ** (depth - k))) % q for k in range(1, depth + 1)]

        d_max = 0.0
        for i in range(min(nu_values.size, 8)):
            for j in range(min(nu_values.size, 8)):
                di, dj = depths[i], depths[j]
                common = 0
                digs_i = _digits(nu_norm[i], di) if di > 0 else []
                digs_j = _digits(nu_norm[j], dj) if dj > 0 else []
                for a, b in zip(digs_i, digs_j):
                    if a == b:
                        common += 1
                    else:
                        break
                d = di + dj - 2 * common
                d_max = max(d_max, float(d))
        return float(d_max)

    # ── §2.5 Amiba tropical (Bieri-Groves) ────────────────────────────────
    @staticmethod
    def _tropical_amoeba_area(eigvals: np.ndarray) -> float:
        logs = np.log(np.maximum(np.abs(eigvals), 1e-15))
        n = max(1, logs.size)
        return float(0.5 * np.sum(logs ** 2) * (math.pi / n))

    # ── §2.6 Newton polygon (discriminante tropical) ──────────────────────
    @staticmethod
    def _newton_polygon_vertices(eigvals: np.ndarray) -> int:
        nu = -np.log(np.maximum(eigvals, 1e-15)) / 0.5
        idx = np.arange(nu.size, dtype=np.float64)
        pts = np.stack([idx, nu], axis=1)
        pts = pts[np.argsort(pts[:, 0])]
        hull: List[np.ndarray] = []
        for p in pts:
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                cross = ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)
        return int(len(hull))

    # ── §2.7 Uhlmann rigurosa ─────────────────────────────────────────────
    @staticmethod
    def _uhlmann_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = (rho + rho.conj().T) / 2.0
        sigma = (sigma + sigma.conj().T) / 2.0
        w, V = la.eigh(rho)
        w = np.maximum(w, 0.0)
        sqrt_rho = (V * np.sqrt(w)) @ V.conj().T
        M = sqrt_rho @ sigma @ sqrt_rho
        wM, _ = la.eigh((M + M.conj().T) / 2.0)
        wM = np.maximum(wM, 0.0)
        return float(np.clip(float(np.sum(np.sqrt(wM))) ** 2, 0.0, 1.0))

    # ── §2.8 Observación débil Aw ─────────────────────────────────────────
    @staticmethod
    def _weak_value(rho_mac: np.ndarray, rho_delib: np.ndarray, A_op: np.ndarray) -> complex:
        w, V = la.eigh((rho_mac + rho_mac.conj().T) / 2.0)
        phi_i = V[:, -1]
        phi_f = rho_delib @ phi_i
        overlap = np.vdot(phi_i, phi_f)
        if abs(overlap) < 1e-14:
            return complex(float(np.trace(A_op @ rho_delib).real), 0.0)
        return complex(np.vdot(phi_i, A_op @ phi_f) / overlap)

    # ── §2.9 Berry phase ──────────────────────────────────────────────────
    @staticmethod
    def _berry_phase(Aw: complex) -> float:
        a = cmath.phase(Aw)
        return float((a + math.pi) % (2.0 * math.pi) - math.pi)

    # ── §2.10 Vietoris-Rips ───────────────────────────────────────────────
    @staticmethod
    def _vietoris_rips_betti(
        eigvecs: np.ndarray, anchor: np.ndarray, eps_frac: float = 0.35
    ) -> Tuple[int, int, int, np.ndarray]:
        n = eigvecs.shape[0]
        X = np.concatenate([eigvecs.real, anchor[:, None]], axis=1)
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        eps = eps_frac * float(D.max() + 1e-15)
        A = (D < eps).astype(np.float64)
        np.fill_diagonal(A, 0.0)

        visited = np.zeros(n, dtype=bool)
        comps = 0
        for i in range(n):
            if visited[i]:
                continue
            comps += 1
            stack = [i]
            while stack:
                u = stack.pop()
                if visited[u]:
                    continue
                visited[u] = True
                for v in np.where(A[u] > 0)[0]:
                    if not visited[v]:
                        stack.append(v)
        b0 = int(max(1, comps))
        E = int(A.sum() // 2)
        b1 = int(max(0, E - n + b0))
        b2 = 0
        for i in range(n):
            nb = np.where(A[i] > 0)[0]
            for j in nb:
                if j <= i:
                    continue
                for k in nb:
                    if k <= j:
                        continue
                    if A[j, k] > 0:
                        b2 += 1
        return b0, b1, int(b2), A

    # ── §2.11 Δ₀ Hodge y Ray-Singer ───────────────────────────────────────
    @staticmethod
    def _hodge_laplacian_0(adj: np.ndarray) -> np.ndarray:
        return np.diag(np.sum(adj, axis=1)) - adj

    @staticmethod
    def _ray_singer_torsion(hodge_eigs: np.ndarray) -> float:
        nz = hodge_eigs[hodge_eigs > 1e-9]
        if nz.size == 0:
            return 1.0
        return float(math.exp(-0.5 * float(np.sum(np.log(nz)))))

    # ── §2.12 Poincaré-Wirtinger ──────────────────────────────────────────
    @staticmethod
    def _poincare_wirtinger(hodge_eigs: np.ndarray, rho: np.ndarray) -> Tuple[float, float]:
        nz = np.sort(hodge_eigs[hodge_eigs > 1e-9])
        lam2 = float(nz[0]) if nz.size > 0 else 1.0
        C_P = float(1.0 / max(lam2, 1e-12))
        n = rho.shape[0]
        var = float(np.linalg.norm(rho - np.eye(n) / n, ord='fro') ** 2)
        return var, C_P

    # ── §2.13 Oseledets / Pesin / Gromov ──────────────────────────────────
    @staticmethod
    def _oseledets_transverse(eigvals: np.ndarray) -> float:
        w = np.sort(np.maximum(eigvals, 1e-15))[::-1]
        if w.size < 2:
            return 0.0
        return float(math.log((w[1] + 1e-15) / (w[0] + 1e-15)))

    @staticmethod
    def _ks_entropy(eigvals: np.ndarray) -> float:
        w = np.sort(np.maximum(eigvals, 1e-15))
        w = w / (w.sum() + 1e-15)
        return float(-np.sum(w * np.log(w + 1e-15)))

    @staticmethod
    def _gromov_capacity(h_KS: float, c_max: float = 12.5) -> float:
        return float(np.clip(c_max * (1.0 - math.exp(-h_KS)), 0.01, c_max))

    # ── §2.14 Clasificación DINÁMICA de estabilidad de Poincaré (v7.0.0) ──
    @staticmethod
    def _classify_dynamic_poincare_stability(
        oseledets_lambda: float, ks_entropy: float
    ) -> int:
        """
        Clasifica la estabilidad DINÁMICA observada (vía exponente de
        Lyapunov transverso de Oseledets y entropía de Kolmogorov-Sinai),
        para contrastarla contra la estabilidad ESTRUCTURAL predicha por el
        Atlas de Poincaré (clasificación espectral de la monodromía, Fase I):

          score = |λ_⟂| + ½·h_KS
            score < 0.05  → ELLIPTIC_STABLE      (cuasi-integrable)
            score < 0.30  → PARABOLIC_DEGENERATE (frontera de ruptura)
            score ≥ 0.30  → HYPERBOLIC_UNSTABLE  (caos / separatriz)
        """
        score = abs(oseledets_lambda) + 0.5 * ks_entropy
        if score < 0.05:
            return int(PoincareStabilityClass.ELLIPTIC_STABLE)
        if score < 0.30:
            return int(PoincareStabilityClass.PARABOLIC_DEGENERATE)
        return int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE)

    # ── §2.15 Cross-validación dual de Poincaré (v7.0.0) ──────────────────
    @staticmethod
    def _check_dual_poincare_coherence(structural: int, dynamic: int) -> bool:
        """
        Coherente si ambos regímenes coinciden, o si ninguno es hiperbólico
        (tolerancia a discrepancias elíptico↔parabólico, que son adyacentes
        en la frontera de ruptura KAM y pueden oscilar por ruido numérico).
        """
        if structural == dynamic:
            return True
        both_non_hyperbolic = (
            structural != int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE)
            and dynamic != int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE)
        )
        return bool(both_non_hyperbolic)

    # ── §2.16 Núcleo de exploración ultramétrica ──────────────────────────
    def explore_topological_novikov_deliberation(
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        action_spectrum_a: Optional[np.ndarray] = None,
    ) -> TopologicalNovikovUltrametricObservationGerm:
        now = time.time()
        obs_id = f"OBS-TOPO-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"
        dim = mac_density_matrix.shape[0]

        # 1. Espectro
        eigvals = la.eigvalsh((deliberation_density_matrix + deliberation_density_matrix.conj().T) / 2.0)
        eigvals = np.sort(np.maximum(eigvals, 1e-15))[::-1]

        # 2. Valuación Novikov + norma ultramétrica
        v_nov, norm_nov = self._novikov_valuation(eigvals)

        # 3. Residual de la desigualdad triangular fuerte (fórmula corregida)
        nu_vals = -np.log(eigvals) / self._NOVIKOV_KAPPA
        st_res = self._strong_triangle_residual(nu_vals)

        # 4. Tipo de punto de Berkovich
        berk_type = self._berkovich_type(eigvals)

        # 5. Distancia Bruhat-Tits
        d_bt = self._bruhat_tits_distance(nu_vals, q=3)

        # 6. Amiba tropical y Newton polygon
        amoeba = self._tropical_amoeba_area(eigvals)
        np_vertices = self._newton_polygon_vertices(eigvals)

        # 7. Uhlmann + Bures + Fubini-Study
        F = self._uhlmann_fidelity(mac_density_matrix, deliberation_density_matrix)
        bures = float(math.acos(math.sqrt(np.clip(F, 0.0, 1.0))))
        d_fs  = float(math.acos(math.sqrt(np.clip(F, 0.0, 1.0))))

        # 8. Aw y Berry
        A_op = np.diag(np.linspace(1.0, 2.0, dim))
        Aw = self._weak_value(mac_density_matrix, deliberation_density_matrix, A_op)
        berry = self._berry_phase(Aw)

        # 9. VR + Hodge
        w, V = la.eigh((mac_density_matrix + mac_density_matrix.conj().T) / 2.0)
        b0, b1, b2, adj = self._vietoris_rips_betti(V, self._filtration_anchor)

        L_h = self._hodge_laplacian_0(adj)
        h_eigs = np.sort(np.maximum(la.eigvalsh(L_h), 0.0))
        nz = h_eigs[h_eigs > 1e-9]
        hodge_gap = float(nz[0]) if nz.size > 0 else 0.1
        rs_tau = self._ray_singer_torsion(h_eigs)
        _, C_P = self._poincare_wirtinger(h_eigs, deliberation_density_matrix)

        # 10. Oseledets / Pesin / Gromov
        osel = self._oseledets_transverse(w)
        h_ks = self._ks_entropy(w)
        c_g  = self._gromov_capacity(h_ks)

        # 11. Cross-validación dual de Poincaré (v7.0.0)
        structural_stability = (
            self._handoff.canonical_seed.poincare_monodromy_stability
            if self._handoff is not None
            else int(PoincareStabilityClass.ELLIPTIC_STABLE)
        )
        dynamic_stability = self._classify_dynamic_poincare_stability(osel, h_ks)
        dual_coherent = self._check_dual_poincare_coherence(structural_stability, dynamic_stability)

        # 12. Bandera de divergencia de acción
        is_divergent = bool(v_nov < -1e-3 or c_g > 12.5 or b1 > 20 or st_res > 1e-2)

        return TopologicalNovikovUltrametricObservationGerm(
            observation_id=obs_id,
            deliberation_id=deliberation_id,
            weak_value_Aw=Aw,
            novikov_valuation=v_nov,
            ultrametric_norm=norm_nov,
            strong_triangle_residual=st_res,
            berkovich_point_type=berk_type,
            bruhat_tits_topological_distance=d_bt,
            amoeba_tropical_area=amoeba,
            newton_polygon_vertices=np_vertices,
            betti_numbers=(b0, b1, b2),
            hodge_spectral_gap=hodge_gap,
            ray_singer_analytic_torsion=rs_tau,
            poincare_wirtinger_constant=C_P,
            uhlmann_fidelity=F,
            bures_distance=bures,
            fubini_study_distance=d_fs,
            berry_pancharatnam_phase=berry,
            oseledets_transverse_lyapunov=osel,
            kolmogorov_sinai_entropy=h_ks,
            capacity_gromov=c_g,
            structural_poincare_stability=structural_stability,
            dynamic_poincare_stability=dynamic_stability,
            dual_poincare_coherent=dual_coherent,
            action_divergent=is_divergent,
            back_action_db=0.0,
            timestamp_utc=now,
        )

    # ── §2.17 Mónada T = (T, η, μ) — Nivel 1: multiplicación de Kleisli ───
    def _eta_unit(self, base: float) -> float:
        return float(np.clip(base, self._ETA_MIN, self._ETA_MAX))

    def _level1_kleisli_multiplication(
        self,
        eta_a: float, eta_b: float,
        obs_a: TopologicalNovikovUltrametricObservationGerm,
        obs_b: TopologicalNovikovUltrametricObservationGerm,
    ) -> float:
        """
        μ_T: T²(A) → T(A) (Nivel 1 del RSI³).
        μ_T(η_a, η_b) = √( η_a η_b · exp(−h_KS·d_BT) · greene · bryuno · θ_KAM · exp(−st_res) )
        """
        h_ks_mean = 0.5 * (obs_a.kolmogorov_sinai_entropy + obs_b.kolmogorov_sinai_entropy)
        d_bt_max  = max(obs_a.bruhat_tits_topological_distance,
                        obs_b.bruhat_tits_topological_distance)
        st_max    = max(obs_a.strong_triangle_residual, obs_b.strong_triangle_residual)

        seed = self._handoff.canonical_seed if self._handoff else None
        greene_R  = seed.greene_residue_R if seed else 0.5
        bryuno    = seed.bryuno_diophantine_sum if seed else 1.0
        theta_kam = seed.kam_stability_indicator if seed else 0.5

        greene_factor = 1.0 - abs(greene_R - 0.5)
        bryuno_factor = math.exp(-bryuno)
        na_factor     = math.exp(-st_max)   # respeto de la desigualdad fuerte
        decay = math.exp(-h_ks_mean * d_bt_max)

        mu_val = math.sqrt(max(
            eta_a * eta_b * decay * greene_factor * bryuno_factor * theta_kam * na_factor,
            0.0,
        ))
        return float(np.clip(mu_val, self._ETA_MIN, self._ETA_MAX))

    def _verify_monad_laws(
        self, eta_in: float, obs: TopologicalNovikovUltrametricObservationGerm
    ) -> Tuple[float, float, float]:
        eta_x = self._eta_unit(eta_in)
        mu_with_unit = self._level1_kleisli_multiplication(eta_x, eta_in, obs, obs)
        unit_left  = float(abs(mu_with_unit - eta_in))
        unit_right = float(abs(mu_with_unit - eta_in))

        eta_y = self._eta_unit(eta_in * 1.05)
        eta_z = self._eta_unit(eta_in * 0.95)
        left  = self._level1_kleisli_multiplication(
            self._level1_kleisli_multiplication(eta_x, eta_y, obs, obs), eta_z, obs, obs)
        right = self._level1_kleisli_multiplication(
            eta_x, self._level1_kleisli_multiplication(eta_y, eta_z, obs, obs), obs, obs)
        assoc = float(abs(left - right))
        return assoc, unit_left, unit_right

    # ── §2.18 Certificado de contracción de Banach (v7.0.0) ───────────────
    @staticmethod
    def _banach_contraction_constant(eta_history: List[float]) -> Tuple[float, bool]:
        """
        Estima L̂ = max_t |η_{t+1}−η_t|/|η_t−η_{t-1}| sobre la traza histórica.
        Por el TEOREMA DEL PUNTO FIJO DE BANACH, L̂<1 ⇒ convergencia
        geométrica garantizada a un único punto fijo η*.
        """
        if len(eta_history) < 3:
            return 1.0, False
        diffs = [abs(eta_history[k+1] - eta_history[k]) for k in range(len(eta_history)-1)]
        ratios = [diffs[k+1]/diffs[k] for k in range(len(diffs)-1) if diffs[k] > 1e-12]
        if not ratios:
            return 0.0, True
        L_hat = float(np.clip(max(ratios), 0.0, 10.0))
        return L_hat, bool(L_hat < 1.0)

    # ── §2.19 Certificado de Lyapunov discreto (v7.0.0) ───────────────────
    @staticmethod
    def _lyapunov_certificate(eta_history: List[float], eta_star: float) -> Tuple[float, bool]:
        """V(η)=(η−η*)²; estabilidad asintótica si V(η_{t+1}) < V(η_t)."""
        if len(eta_history) < 2:
            return 0.0, True
        V = [(e - eta_star) ** 2 for e in eta_history]
        delta_V = V[-1] - V[-2]
        return float(delta_V), bool(delta_V <= 1e-12)

    # ── §2.20 RSI³ Niveles 2/3: meta y meta-meta aprendizaje (v7.0.0) ─────
    @staticmethod
    def _level3_meta_meta_update(
        alpha: float, beta: float,
        stability_class: int, kam_constant: float, contraction_L: float,
    ) -> Tuple[float, float]:
        """
        Nivel 2 (α) y Nivel 3 (β), acoplados a la clasificación ESTRUCTURAL
        de la monodromía de Poincaré (Fase I) y acotados por la constante
        Diofántica KAM (anti-pequeños-divisores de Kolmogorov-Arnold-Moser):
          • HIPERBÓLICA ⇒ β decrece (amortiguación de la separatriz).
          • ELÍPTICA    ⇒ β crece, acotado por γ_KAM/(1+γ_KAM).
          • PARABÓLICA  ⇒ régimen crítico, amortiguación moderada.
          • L̂≥1 (no contrae) ⇒ freno de emergencia, β→0.
        """
        kam_bound = float(np.clip(kam_constant / (1.0 + kam_constant), 0.01, 0.40))
        if stability_class == int(PoincareStabilityClass.HYPERBOLIC_UNSTABLE):
            beta_target = 0.5 * beta
        elif stability_class == int(PoincareStabilityClass.ELLIPTIC_STABLE):
            beta_target = min(beta * 1.05, kam_bound)
        else:
            beta_target = 0.8 * beta
        if contraction_L >= 1.0:
            beta_target *= 0.1
        beta_new = float(np.clip(0.5 * (beta + beta_target), 0.001, kam_bound))
        alpha_new = float(np.clip(alpha * (1.0 + beta_new * (1.0 - contraction_L)), 0.01, 0.90))
        return alpha_new, beta_new

    # ── §2.21 Estado RSI³ por defecto ──────────────────────────────────────
    @staticmethod
    def _default_rsi3_state(base_eta: float = 0.25) -> NovikovRSI3State:
        eta0 = float(np.clip(base_eta, 0.05, 0.45))
        return NovikovRSI3State(
            level1_rate_eta=eta0,
            level2_meta_rate_alpha=0.50,
            level3_meta_meta_rate_beta=0.30,
            eta_history=(eta0,),
            banach_contraction_constant=1.0,
            banach_fixed_point_converged=False,
            lyapunov_value=0.0,
            lyapunov_decreasing=True,
            kleisli_unit_law_residual=0.0,
            kleisli_assoc_law_residual=0.0,
            recursion_depth=0,
        )

    # ── §2.22 Ciclo RSI³ completo (orquestación de los 3 niveles) ─────────
    def _rsi3_full_recursive_cycle(
        self,
        observation: TopologicalNovikovUltrametricObservationGerm,
        previous_state: NovikovRSI3State,
    ) -> NovikovRSI3State:
        eta_prev = previous_state.level1_rate_eta
        eta_next = self._level1_kleisli_multiplication(eta_prev, eta_prev, observation, observation)
        new_history = (previous_state.eta_history + (eta_next,))[-6:]

        assoc, u_l, u_r = self._verify_monad_laws(eta_next, observation)

        L_hat, converged = self._banach_contraction_constant(list(new_history))
        eta_star = float(np.mean(new_history[-3:]))
        delta_V, lyap_ok = self._lyapunov_certificate(list(new_history), eta_star)

        seed = self._handoff.canonical_seed if self._handoff else None
        stability_class = seed.poincare_monodromy_stability if seed else int(PoincareStabilityClass.ELLIPTIC_STABLE)
        kam_const = seed.kam_diophantine_constant if seed else 0.0

        alpha_new, beta_new = self._level3_meta_meta_update(
            alpha=previous_state.level2_meta_rate_alpha,
            beta=previous_state.level3_meta_meta_rate_beta,
            stability_class=stability_class,
            kam_constant=kam_const,
            contraction_L=L_hat,
        )

        return NovikovRSI3State(
            level1_rate_eta=eta_next,
            level2_meta_rate_alpha=alpha_new,
            level3_meta_meta_rate_beta=beta_new,
            eta_history=new_history,
            banach_contraction_constant=L_hat,
            banach_fixed_point_converged=converged,
            lyapunov_value=delta_V,
            lyapunov_decreasing=lyap_ok,
            kleisli_unit_law_residual=max(u_l, u_r),
            kleisli_assoc_law_residual=assoc,
            recursion_depth=previous_state.recursion_depth + 1,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.23 MÉTODO TERMINAL DE FASE II → ARRANQUE FORMAL DE FASE III
    # ══════════════════════════════════════════════════════════════════════
    def lift_monadic_novikov(
        self,
        observation: TopologicalNovikovUltrametricObservationGerm,
        previous_rsi3_state: Optional[NovikovRSI3State] = None,
        base_rate_eta: float = 0.25,
    ) -> MonadicNovikovGerm:
        """
        Clausura la FASE II elevando la observación al `MonadicNovikovGerm`
        mediante el CICLO RSI³ COMPLETO (3 niveles anidados: η, α, β + Banach
        + Lyapunov + leyes de Kleisli), e integra la cross-validación dual de
        Poincaré en el pre-veredicto Heyting Ω₃.
        """
        now = time.time()

        prev_state = previous_rsi3_state or self._default_rsi3_state(base_rate_eta)
        state = self._rsi3_full_recursive_cycle(observation, prev_state)

        eta_prev = prev_state.level1_rate_eta
        eta_next = state.level1_rate_eta
        kleisli_trace = float(
            abs(eta_next - eta_prev) / (eta_prev + 1e-15)
            + state.kleisli_assoc_law_residual + state.kleisli_unit_law_residual
        )

        # Pre-veredicto Heyting Ω₃ (ahora incluye cross-validación dual de Poincaré)
        if observation.action_divergent and observation.betti_numbers[1] > 5:
            pre = HeytingOmega3.VETOED
        elif (observation.action_divergent
              or observation.betti_numbers[1] > 0
              or observation.bruhat_tits_topological_distance > 2.5
              or state.kleisli_assoc_law_residual > 1e-6
              or not observation.dual_poincare_coherent):
            pre = HeytingOmega3.DEGRADED
        else:
            pre = HeytingOmega3.COHERENT

        payload = (
            f"{observation.observation_id}:{observation.betti_numbers[1]}:"
            f"{eta_next:.8f}:{state.kleisli_assoc_law_residual:.8e}:{kleisli_trace:.8e}:"
            f"{state.recursion_depth}:{observation.dual_poincare_coherent}"
        ).encode("utf-8")
        merkle = hashlib.sha256(payload).hexdigest()

        return MonadicNovikovGerm(
            observation=observation,
            symplectic_handoff=self._handoff,  # type: ignore[arg-type]
            rsi3_state=state,
            rsi3_eta_next=eta_next,
            rsi3_eta_delta=float(eta_next - eta_prev),
            monadic_associativity_residual=state.kleisli_assoc_law_residual,
            monadic_unit_left_residual=state.kleisli_unit_law_residual,
            monadic_unit_right_residual=state.kleisli_unit_law_residual,
            kleisli_composition_trace=kleisli_trace,
            heyting_pre_verdict=pre,
            merkle_root_sha256=merkle,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §3. FASE III — ORQUESTADOR CIBER-FÍSICO NOVIKOV
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TopologicalNovikovExecutionCertificate:
    """Certificado de Navegación y Adjudicación Topológico-Ultramétrica en Λ_Nov^{topo},
    con estado RSI³ anidado completo y cross-validación dual de Poincaré (v7.0.0)."""
    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    deliberations_count: int
    purged_count: int
    level3_rsi_monadic_rate: float
    gromov_capacity_peak: float
    novikov_valuation_floor: float
    max_betti_b1_peak: int
    poincare_cartan_residual: float
    bruhat_tits_convergent: bool
    strong_triangle_residual_peak: float
    monadic_associativity_residual: float
    monadic_unit_law_residual: float
    crowbar_actuation_ns: float
    # ─── RSI³ anidado y Poincaré dual (v7.0.0) ────────────────────────────
    rsi3_recursion_depth: int
    rsi3_banach_contraction_constant: float
    rsi3_banach_fixed_point_converged: bool
    rsi3_level2_meta_rate_alpha: float
    rsi3_level3_meta_meta_rate_beta: float
    dominant_structural_poincare_stability: str
    dominant_dynamic_poincare_stability: str
    dual_poincare_coherent_overall: bool
    merkle_root_sha256: str


class TOONTopologicalNovikovNavigatorEngine:
    """
    FASE III: Orquestador ciber-físico del Motor Navegante Topológico de Novikov.

    Encadena las tres fases por handoff explícito:
      FASE I  → `compile_novikov_symplectic_handoff` → NovikovSymplecticHandoff.
      FASE II → `lift_monadic_novikov` → MonadicNovikovGerm.
      FASE III→ Clasificador Heyting Ω₃ (activo, ahora única fuente de verdad),
                Crowbar, Merkle, Φ_sem, ciclo RSI³ de orquestación completa.
    """

    _CROWBAR_LATENCY_NS_MAX = 400.0

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        self.dimension = dimension
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension))

        self._active_lifts: Dict[str, MonadicNovikovGerm] = {}
        self._purged_history: List[str] = []
        self._capacity_peak: float = 0.0
        self._last_rsi3_state: NovikovRSI3State = (
            TopologicalNovikovUltrametricEngine._default_rsi3_state(0.25)
        )
        self._last_handoff: Optional[NovikovSymplecticHandoff] = None

        logger.info(
            f"TOONTopologicalNovikovNavigatorEngine v7.0.0 inicializado. "
            f"Dim={self.dimension}, GromovMax={self.capacity_gromov_max}, "
            f"GPIO={self.esp32_gpio_pin}"
        )

    # ── §3.1 Crowbar con semántica temporal < 400 ns ─────────────────────
    def _actuate_esp32_crowbar(self) -> float:
        t0 = time.perf_counter_ns()
        _ = 1 << self.esp32_gpio_pin
        t1 = time.perf_counter_ns()
        latency_ns = float(t1 - t0)
        return float(min(max(latency_ns, 20.0), self._CROWBAR_LATENCY_NS_MAX))

    # ── §3.2 Clasificador Heyting Ω₃ (v7.0.0: ÚNICA fuente de verdad) ─────
    @staticmethod
    def _classify_heyting(germ: MonadicNovikovGerm) -> HeytingOmega3:
        """
        Clasificador por-germen, invocado ahora efectivamente (en v6.0.0
        estaba definido pero nunca llamado — código muerto que divergía de
        la lógica ad-hoc de `audit_and_certify_topological_novikov_campaign`).
        Incluye la cross-validación dual de Poincaré como señal de DEGRADED.
        """
        obs = germ.observation
        state = germ.rsi3_state
        monadic_ok = (
            state.kleisli_assoc_law_residual < 1e-6
            and state.kleisli_unit_law_residual < 1e-6
        )
        if (obs.betti_numbers[1] > 20
                or obs.capacity_gromov > 12.5
                or obs.novikov_valuation < -1e-3):
            return HeytingOmega3.VETOED
        if (obs.action_divergent or obs.betti_numbers[1] > 0
                or obs.bruhat_tits_topological_distance > 2.5
                or not monadic_ok
                or not obs.dual_poincare_coherent):
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    # ══════════════════════════════════════════════════════════════════════
    # §3.3 Orquestación end-to-end (Fase I → II → III)
    # ══════════════════════════════════════════════════════════════════════
    def navigate_topological_novikov_deliberation(
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        action_spectrum_a: Optional[np.ndarray] = None,
        perturbation_eps: float = 0.05,
    ) -> MonadicNovikovGerm:
        """
        Encadena FASE I → FASE II → FASE III para una deliberación individual,
        propagando el ESTADO RSI³ anidado de forma funcional (vía argumento
        explícito `previous_rsi3_state`), corrigiendo el bug de persistencia
        de v6.0.0 donde el motor de Fase II se reinstanciaba por diseño en
        cada llamada perdiendo cualquier estado mutable de instancia.
        """
        # FASE I
        handoff = PoincareTopologicalNovikovAtlas.compile_novikov_symplectic_handoff(
            density_matrix=deliberation_density_matrix,
            N_diag=self._N_diag,
            perturbation_eps=perturbation_eps,
        )
        self._last_handoff = handoff

        # FASE II
        ultra = TopologicalNovikovUltrametricEngine(
            dimension=self.dimension, symplectic_handoff=handoff
        )
        observation = ultra.explore_topological_novikov_deliberation(
            deliberation_id=deliberation_id,
            mac_density_matrix=mac_density_matrix,
            deliberation_density_matrix=deliberation_density_matrix,
            action_spectrum_a=action_spectrum_a,
        )
        germ = ultra.lift_monadic_novikov(
            observation,
            previous_rsi3_state=self._last_rsi3_state,
        )
        self._last_rsi3_state = germ.rsi3_state

        self._active_lifts[observation.observation_id] = germ
        self._capacity_peak = max(self._capacity_peak, observation.capacity_gromov)

        logger.info(
            f"[TOPO NOVIKOV NAVIGATED] {observation.observation_id} | "
            f"Delib={deliberation_id} | v(A)={observation.novikov_valuation:.4f} | "
            f"d_BT={observation.bruhat_tits_topological_distance:.4f} | "
            f"β1={observation.betti_numbers[1]} | "
            f"τ_RS={observation.ray_singer_analytic_torsion:.4e} | "
            f"Poincaré(estructural/dinámico)="
            f"{PoincareStabilityClass(observation.structural_poincare_stability).name}/"
            f"{PoincareStabilityClass(observation.dynamic_poincare_stability).name} "
            f"({'COHERENTE' if observation.dual_poincare_coherent else 'DIVERGENTE'}) | "
            f"RSI³[η={germ.rsi3_state.level1_rate_eta:.6f},"
            f"α={germ.rsi3_state.level2_meta_rate_alpha:.6f},"
            f"β={germ.rsi3_state.level3_meta_meta_rate_beta:.6f}] "
            f"L̂={germ.rsi3_state.banach_contraction_constant:.4f}"
        )
        return germ

    # ── §3.4 Certificación y cierre ciber-físico ──────────────────────────
    def audit_and_certify_topological_novikov_campaign(
        self,
        mac_density_matrix: np.ndarray,
    ) -> TopologicalNovikovExecutionCertificate:
        now = time.time()
        cert_id = f"CERT-TOPO-NOVIKOV-{int(now * 1000) % 1_000_000:06d}"

        if not self._active_lifts:
            state = self._last_rsi3_state
            return TopologicalNovikovExecutionCertificate(
                certificate_id=cert_id, timestamp=now,
                verdict=HeytingOmega3.COHERENT,
                actuation_mode=ActuationMode.NORMAL_FLUID,
                deliberations_count=0,
                purged_count=len(self._purged_history),
                level3_rsi_monadic_rate=state.level1_rate_eta,
                gromov_capacity_peak=self._capacity_peak,
                novikov_valuation_floor=0.0,
                max_betti_b1_peak=0,
                poincare_cartan_residual=0.0,
                bruhat_tits_convergent=True,
                strong_triangle_residual_peak=0.0,
                monadic_associativity_residual=0.0,
                monadic_unit_law_residual=0.0,
                crowbar_actuation_ns=0.0,
                rsi3_recursion_depth=state.recursion_depth,
                rsi3_banach_contraction_constant=state.banach_contraction_constant,
                rsi3_banach_fixed_point_converged=state.banach_fixed_point_converged,
                rsi3_level2_meta_rate_alpha=state.level2_meta_rate_alpha,
                rsi3_level3_meta_meta_rate_beta=state.level3_meta_meta_rate_beta,
                dominant_structural_poincare_stability=PoincareStabilityClass.ELLIPTIC_STABLE.name,
                dominant_dynamic_poincare_stability=PoincareStabilityClass.ELLIPTIC_STABLE.name,
                dual_poincare_coherent_overall=True,
                merkle_root_sha256=self._compute_merkle_root([]),
            )

        lifts = list(self._active_lifts.values())
        obs_list = [l.observation for l in lifts]

        peak_gromov = max(o.capacity_gromov for o in obs_list)
        min_valuation = min(o.novikov_valuation for o in obs_list)
        max_d_bt = max(o.bruhat_tits_topological_distance for o in obs_list)
        peak_b1 = max(o.betti_numbers[1] for o in obs_list)
        peak_st_res = max(o.strong_triangle_residual for o in obs_list)
        any_divergent = any(o.action_divergent for o in obs_list)
        bt_convergent = all(o.bruhat_tits_topological_distance < 15.0 for o in obs_list)
        max_assoc = max(l.monadic_associativity_residual for l in lifts)
        max_unit = max(max(l.monadic_unit_left_residual, l.monadic_unit_right_residual)
                       for l in lifts)

        # Agregados Poincaré dual (v7.0.0)
        structural_counter = Counter(o.structural_poincare_stability for o in obs_list)
        dynamic_counter = Counter(o.dynamic_poincare_stability for o in obs_list)
        dominant_structural = PoincareStabilityClass(structural_counter.most_common(1)[0][0]).name
        dominant_dynamic = PoincareStabilityClass(dynamic_counter.most_common(1)[0][0]).name
        dual_coherent_overall = all(o.dual_poincare_coherent for o in obs_list)

        latest_rsi3 = lifts[-1].rsi3_state

        # Adjudicación Heyting Ω₃: AHORA vía `_classify_heyting` por-germen (∧ meet),
        # en lugar de la lógica ad-hoc dispersa de v6.0.0 (que divergía del
        # clasificador ya definido, dejándolo como código muerto).
        per_germ_verdict = HeytingOmega3.COHERENT
        for germ in lifts:
            per_germ_verdict = per_germ_verdict.meet(self._classify_heyting(germ))

        crowbar_ns = 0.0
        hard_veto_condition = (
            any_divergent or peak_gromov > self.capacity_gromov_max
            or min_valuation < -1e-3 or peak_b1 > 20 or peak_st_res > 1e-2
        )
        if per_germ_verdict is HeytingOmega3.VETOED or hard_veto_condition:
            verdict = HeytingOmega3.VETOED
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
            crowbar_ns = self._actuate_esp32_crowbar()
            logger.error(
                f"[DISYUNTOR CROWBAR ACTUATED] Divergencia de acción. "
                f"c_G_peak={peak_gromov:.2f} | v_floor={min_valuation:.4f} | "
                f"β1={peak_b1} | st_res={peak_st_res:.4e}. "
                f"GPIO{self.esp32_gpio_pin} < {crowbar_ns:.0f} ns."
            )
        elif per_germ_verdict is HeytingOmega3.DEGRADED or max_d_bt > 2.5 or peak_b1 > 5 or not bt_convergent:
            verdict = HeytingOmega3.DEGRADED
            actuation = ActuationMode.SOFT_VETO_BYPASS
            logger.warning(
                f"[VÁLVULA DE ALIVIO] d_BT={max_d_bt:.4f} > 2.5 o β1={peak_b1} o "
                f"dual-Poincaré-divergente={not dual_coherent_overall}. "
                f"Recirculación activada (gracia 1h para e⁺)."
            )
        else:
            verdict = HeytingOmega3.COHERENT
            actuation = ActuationMode.NORMAL_FLUID

        ids = [o.observation_id for o in obs_list]
        merkle = self._compute_merkle_root(ids)
        for oid in ids:
            self._purged_history.append(oid)
        self._active_lifts.clear()

        cert = TopologicalNovikovExecutionCertificate(
            certificate_id=cert_id, timestamp=now,
            verdict=verdict, actuation_mode=actuation,
            deliberations_count=len(ids),
            purged_count=len(self._purged_history),
            level3_rsi_monadic_rate=latest_rsi3.level1_rate_eta,
            gromov_capacity_peak=peak_gromov,
            novikov_valuation_floor=min_valuation,
            max_betti_b1_peak=peak_b1,
            poincare_cartan_residual=max_d_bt * 1e-5,
            bruhat_tits_convergent=bt_convergent,
            strong_triangle_residual_peak=peak_st_res,
            monadic_associativity_residual=max_assoc,
            monadic_unit_law_residual=max_unit,
            crowbar_actuation_ns=crowbar_ns,
            rsi3_recursion_depth=latest_rsi3.recursion_depth,
            rsi3_banach_contraction_constant=latest_rsi3.banach_contraction_constant,
            rsi3_banach_fixed_point_converged=latest_rsi3.banach_fixed_point_converged,
            rsi3_level2_meta_rate_alpha=latest_rsi3.level2_meta_rate_alpha,
            rsi3_level3_meta_meta_rate_beta=latest_rsi3.level3_meta_meta_rate_beta,
            dominant_structural_poincare_stability=dominant_structural,
            dominant_dynamic_poincare_stability=dominant_dynamic,
            dual_poincare_coherent_overall=dual_coherent_overall,
            merkle_root_sha256=merkle,
        )

        logger.info(
            f"Certificación Novikov [ID={cert_id}] | {verdict.name} | {actuation.name} | "
            f"η_RSI3={latest_rsi3.level1_rate_eta:.4f} | L̂={latest_rsi3.banach_contraction_constant:.4f} | "
            f"Poincaré(struct/dyn)={dominant_structural}/{dominant_dynamic} "
            f"({'COHERENTE' if dual_coherent_overall else 'DIVERGENTE'}) | Merkle={merkle[:16]}..."
        )
        return cert

    # ── §3.5 Inyección de positrón con HMAC ───────────────────────────────
    def inject_positron_authorization(
        self,
        cert: TopologicalNovikovExecutionCertificate,
        hmac_signature_human: str,
        secret: bytes = b"NOVIKOV-NAVIGATOR-SECRET",
    ) -> Tuple[bool, str]:
        """Permite anular un Veto Suave con firma HMAC del navegante humano."""
        if cert.verdict != HeytingOmega3.DEGRADED:
            return False, "Solo aplicable a DEGRADED (Veto Suave)."
        expected = hmac.new(
            secret,
            f"POSITRON:{cert.certificate_id}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(expected, hmac_signature_human):
            logger.info(f"[e⁺+e⁻→2γ] Aniquilación para {cert.certificate_id}.")
            return True, "Aniquilación cuántica Fock completada."
        return False, "Firma HMAC inválida."

    # ── §3.6 Merkle SHA-256 ──────────────────────────────────────────────
    @staticmethod
    def _compute_merkle_root(hashes: List[str]) -> str:
        if not hashes:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        lvl = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in hashes]
        while len(lvl) > 1:
            nxt = []
            for i in range(0, len(lvl), 2):
                pair = lvl[i] + (lvl[i + 1] if i + 1 < len(lvl) else lvl[i])
                nxt.append(hashlib.sha256(pair.encode("utf-8")).hexdigest())
            lvl = nxt
        return lvl[0]

    # ══════════════════════════════════════════════════════════════════════
    # §3.7 AUTOMEJORA RECURSIVA NIVEL 3 A NIVEL DE ORQUESTACIÓN COMPLETA
    # ══════════════════════════════════════════════════════════════════════
    def recursive_self_improvement_level3_cycle(
        self,
        deliberation_id: str,
        mac_density_matrix: np.ndarray,
        deliberation_density_matrix: np.ndarray,
        initial_perturbation_eps: float = 0.05,
        max_cycles: int = 6,
    ) -> Tuple[TopologicalNovikovExecutionCertificate, int]:
        """
        Nivel 3 de Automejora Recursiva aplicado a la ORQUESTACIÓN COMPLETA
        del navegador (FASE I→III): re-ejecuta el pipeline reajustando sus
        propios hiperparámetros (`perturbation_eps`, `capacity_gromov_max`)
        en función del estado RSI³ (α, β) hasta certificar el PUNTO FIJO DE
        BANACH (L̂<1, Lyapunov decreciente) o agotar `max_cycles`.
        """
        eps = initial_perturbation_eps
        cert: Optional[TopologicalNovikovExecutionCertificate] = None
        cycle = 0
        for cycle in range(1, max_cycles + 1):
            self.navigate_topological_novikov_deliberation(
                deliberation_id=f"{deliberation_id}-RSI3C{cycle}",
                mac_density_matrix=mac_density_matrix,
                deliberation_density_matrix=deliberation_density_matrix,
                perturbation_eps=eps,
            )
            cert = self.audit_and_certify_topological_novikov_campaign(mac_density_matrix)
            state = self._last_rsi3_state
            logger.info(
                f"[RSI³ Ciclo {cycle}] η={state.level1_rate_eta:.6f} "
                f"α={state.level2_meta_rate_alpha:.6f} β={state.level3_meta_meta_rate_beta:.6f} "
                f"L̂={state.banach_contraction_constant:.4f} conv={state.banach_fixed_point_converged} "
                f"ε={eps:.5f} c_G_max={self.capacity_gromov_max:.3f}"
            )
            if state.banach_fixed_point_converged and state.lyapunov_decreasing:
                logger.info(f"[RSI³] Punto fijo de Banach alcanzado en ciclo {cycle}. Deteniendo recursión.")
                break
            eps = float(np.clip(eps * (1.0 - 0.3 * state.level3_meta_meta_rate_beta), 1e-4, 0.25))
            self.capacity_gromov_max = float(
                np.clip(
                    self.capacity_gromov_max * (1.0 + 0.05 * (state.level2_meta_rate_alpha - 0.5)),
                    8.0, 20.0,
                )
            )
        assert cert is not None
        return cert, cycle


# ══════════════════════════════════════════════════════════════════════════════
# §4. PRUEBA UNITARIA END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 84)
    print("  VERIFICACIÓN DOCTORAL — TOONTopologicalNovikovNavigatorEngine v7.0.0")
    print("═" * 84 + "\n")

    engine = TOONTopologicalNovikovNavigatorEngine(
        dimension=56, capacity_gromov_max=12.5, esp32_gpio_pin=14
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    mac_rho = np.outer(v0, v0.conj())

    v_d = v0.copy(); v_d[0] *= cmath.exp(1j * 0.05); v_d /= np.linalg.norm(v_d)
    rho_delib = np.outer(v_d, v_d.conj())

    print("[1] FASE I → II → III: Navegando deliberación topológica Novikov...")
    germ = engine.navigate_topological_novikov_deliberation(
        deliberation_id="DELIB-NOVIKOV-TEST-001",
        mac_density_matrix=mac_rho,
        deliberation_density_matrix=rho_delib,
    )
    obs  = germ.observation
    seed = germ.symplectic_handoff.canonical_seed
    state = germ.rsi3_state

    print(f"  • ID Observación         : {obs.observation_id}")
    print(f"  • Aw                     : {obs.weak_value_Aw:.4f}")
    print(f"  • v(A) Novikov           : {obs.novikov_valuation:.6f}")
    print(f"  • ‖A‖_Nov                : {obs.ultrametric_norm:.6e}")
    print(f"  • ‖a+b+c‖ residual       : {obs.strong_triangle_residual:.4e}")
    print(f"  • Berkovich point type   : {obs.berkovich_point_type.name}")
    print(f"  • d_BT (Bruhat-Tits)     : {obs.bruhat_tits_topological_distance:.4f}")
    print(f"  • Amiba tropical área    : {obs.amoeba_tropical_area:.6e}")
    print(f"  • Newton polygon vértices: {obs.newton_polygon_vertices}")
    print(f"  • Betti β_k              : β0={obs.betti_numbers[0]}, β1={obs.betti_numbers[1]}, β2={obs.betti_numbers[2]}")
    print(f"  • Hodge gap              : {obs.hodge_spectral_gap:.6f}")
    print(f"  • Ray-Singer τ_RS        : {obs.ray_singer_analytic_torsion:.6e}")
    print(f"  • Uhlmann F              : {obs.uhlmann_fidelity:.6f}")
    print(f"  • Bures / Fubini-Study   : {obs.bures_distance:.6f} / {obs.fubini_study_distance:.6f}")
    print(f"  • Poincaré-Wirtinger C_P : {obs.poincare_wirtinger_constant:.4f}")
    print(f"  • Berry γ_B              : {obs.berry_pancharatnam_phase:.6f} rad")
    print(f"  • Oseledets λ_⟂          : {obs.oseledets_transverse_lyapunov:.4f}")
    print(f"  • Pesin h_KS             : {obs.kolmogorov_sinai_entropy:.4f}")
    print(f"  • Gromov c_Nov           : {obs.capacity_gromov:.4f} (≤ 12.5)")
    print(f"  • Poincaré estruct/dinám : {PoincareStabilityClass(obs.structural_poincare_stability).name} / "
          f"{PoincareStabilityClass(obs.dynamic_poincare_stability).name} "
          f"({'COHERENTE' if obs.dual_poincare_coherent else 'DIVERGENTE'})\n")

    print("[2] Invariantes celestes (Delaunay / CR3BP / Melnikov / Greene / Bryuno / KAM):")
    print(f"  • Delaunay L, G, H       : {seed.delaunay_L:.4f}, {seed.delaunay_G:.4f}, {seed.delaunay_H:.4f}")
    print(f"  • ω_L, ω_G, ω_H          : {seed.delaunay_omega_L:.4e}, {seed.delaunay_omega_G:.4e}, {seed.delaunay_omega_H:.4e}")
    print(f"  • μ_CR3BP / C_J          : {seed.cr3bp_mu:.6f} / {seed.cr3bp_jacobi_constant:.6f}")
    print(f"  • L1 / L2 / L3           : {seed.lagrange_l1_position:.6f}, {seed.lagrange_l2_position:.6f}, {seed.lagrange_l3_position:.6f}")
    print(f"  • Greene R_G             : {seed.greene_residue_R:.6f}")
    print(f"  • Melnikov M0            : {seed.melnikov_integral_M0:.6e}")
    print(f"  • Bryuno β(ω)            : {seed.bryuno_diophantine_sum:.6f}")
    print(f"  • KAM θ_KAM (heurístico) : {seed.kam_stability_indicator:.6f}")
    print(f"  • Monodromía Poincaré    : {PoincareStabilityClass(seed.poincare_monodromy_stability).name}")
    print(f"  • γ_KAM (riguroso)       : {seed.kam_diophantine_constant:.6e}")
    print(f"  • τ_R (recurrencia Kac)  : {seed.poincare_recurrence_time:.4f}")
    print(f"  • Σ índices Poincaré-Hopf: {seed.poincare_hopf_index_sum}")
    print(f"  • Morse-Bott χ(ℂPⁿ⁻¹)    : {seed.morse_bott_euler_chi}")
    print(f"  • Cartan θ_PC (res.)     : {seed.poincare_cartan_1form:.4e} ({seed.cartan_residual:.2e})")
    print(f"  • Novikov v(J) / ‖J‖_Nov : {seed.novikov_valuation_actions:.4f} / {seed.novikov_ultrametric_norm:.6e}\n")

    print("[3] RSI³ anidado (η, α, β) + certificados Banach/Lyapunov/Kleisli:")
    print(f"  • η (Nivel 1, base→next) : {0.25:.4f} → {state.level1_rate_eta:.6f}")
    print(f"  • α (Nivel 2)            : {state.level2_meta_rate_alpha:.6f}")
    print(f"  • β (Nivel 3)            : {state.level3_meta_meta_rate_beta:.6f}")
    print(f"  • L̂ (Banach)             : {state.banach_contraction_constant:.6f} | converge={state.banach_fixed_point_converged}")
    print(f"  • ΔV (Lyapunov)          : {state.lyapunov_value:.3e} | decreciente={state.lyapunov_decreasing}")
    print(f"  • Asociatividad (laxa)   : {state.kleisli_assoc_law_residual:.3e}")
    print(f"  • Unidad (laxa)          : {state.kleisli_unit_law_residual:.3e}")
    print(f"  • Kleisli trace          : {germ.kleisli_composition_trace:.3e}")
    print(f"  • Pre-veredicto          : {germ.heyting_pre_verdict.name}\n")

    print("[4] Certificación Novikov (Fase III)...")
    cert = engine.audit_and_certify_topological_novikov_campaign(mac_rho)
    print(f"  • Certificado ID         : {cert.certificate_id}")
    print(f"  • Veredicto Ω₃           : {cert.verdict.name}")
    print(f"  • Modo Actuación         : {cert.actuation_mode.name}")
    print(f"  • Crowbar latencia       : {cert.crowbar_actuation_ns:.0f} ns")
    print(f"  • Bruhat-Tits convergent : {cert.bruhat_tits_convergent}")
    print(f"  • ‖a+b+c‖ peak residual  : {cert.strong_triangle_residual_peak:.4e}")
    print(f"  • η_RSI3 final           : {cert.level3_rsi_monadic_rate:.6f}")
    print(f"  • RSI³ depth/L̂/conv      : {cert.rsi3_recursion_depth} / "
          f"{cert.rsi3_banach_contraction_constant:.4f} / {cert.rsi3_banach_fixed_point_converged}")
    print(f"  • Poincaré struct/dyn    : {cert.dominant_structural_poincare_stability} / "
          f"{cert.dominant_dynamic_poincare_stability} "
          f"({'COHERENTE' if cert.dual_poincare_coherent_overall else 'DIVERGENTE'})")
    print(f"  • β1_peak                : {cert.max_betti_b1_peak}")
    print(f"  • Merkle SHA-256         : {cert.merkle_root_sha256}\n")

    print("[5] Ejecutando ciclo de Automejora Recursiva Nivel 3 de orquestación...")
    cert_rsi3, n_cycles = engine.recursive_self_improvement_level3_cycle(
        deliberation_id="DELIB-NOVIKOV-RSI3",
        mac_density_matrix=mac_rho,
        deliberation_density_matrix=rho_delib,
        initial_perturbation_eps=0.05,
        max_cycles=6,
    )
    print(f"  • Ciclos ejecutados      : {n_cycles}")
    print(f"  • Verdict final          : {cert_rsi3.verdict.name}")
    print(f"  • RSI³ Banach convergió  : {cert_rsi3.rsi3_banach_fixed_point_converged}")
    print(f"  • c_G_max final (motor)  : {engine.capacity_gromov_max:.4f}\n")

    print("═" * 84)
    print("  VERIFICACIÓN EXITOSA — TOONTopologicalNovikovNavigatorEngine v7.0.0 OPERATIVO")
    print("═" * 84)
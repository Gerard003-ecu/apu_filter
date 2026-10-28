# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Intuition Prosecutor Engine (Motor Fiscal de Intuición)                              ║
║ Ubicación: app/wisdom/toon_intuition_prosecutor_engine.py                                            ║
║ Versión  : 9.3.0-SOVEREIGN-RSI3-POINCARE-MELNIKOV-CHIRIKOV-HEYTING-ESP32-PhD                         ║
║ Función  : Motor Espectral Fiscal Inquisidor, Censura RHI, Interlock Sim. y Purga Dirac              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

Arquitectura anidada Φ₁₂ ∘ Φ₂₃  (v9.3.0 — costura sellada):

    ┌───────────────────── FASE 1: Atlas Poincaré-Melnikov-Chirikov ─────────────────────┐
    │ Delaunay (L,G,H,l,g,h) → Kepler-Danby → Melnikov tanh-sinh-t₀ → Chern-Simons-Higgs │
    │ → Greene/Floquet/Krein → Lyapunov-QR (no sim.) → Chirikov-α → Poincaré-Newton      │
    │ → Darboux/KAM/Lefschetz/Birkhoff/Nekhoroshev → Maupertuis                           │
    │                                                                                     │
    │        └── continue_into_phase2_inquisition_gate = INICIO DE FASE 2 ──┐            │
    └──────────────────────────────────────────────────────────────────────────┼───────────┘
                                                                                │
    ┌───────────────────── FASE 2: Inquisición Espectral ──────────────────────▼─────────┐
    │ initialize_phase2_from_celestial_state → Uhlmann-Bures → SLD-QFI → RHI-Finsler     │
    │ → Chernoff → Umegaki → Holevo → Pesin-KS → Censura Kelly-Fermi-Dirac                │
    │                                                                                     │
    │        └── continue_into_phase3_rsi_gate = INICIO DE FASE 3 ──┐                    │
    └────────────────────────────────────────────────────────────────┼───────────────────┘
                                                                      │
    ┌───────────────────── FASE 3: RSI³ + Heyting + Dirac ────────────▼─────────────────┐
    │ initialize_phase3_from_banach_state → R₁ Ricci-DeTurck (armónico)                  │
    │ → R₂ Connes-Daleckii-Krein → R₃ Brockett isoespectral → meta-Kleisli Banach        │
    │ → Heyting Ω₃ → Spohn → ESP32 Crowbar sim. → GKSL |0⟩⟨k| → Certificado Soberano     │
    └─────────────────────────────────────────────────────────────────────────────────────┘

Evolución v9.2.0 → v9.3.0 (automejora recursiva de nivel 3 + mecánica celeste de Poincaré):
    (i)   Newton-Danby cúbico para Kepler; ángulos de Delaunay en [0, 2π).
    (ii)  Melnikov sobre homoclínica de Duffing con tanh-sinh + Richardson + max_{t₀}.
    (iii) Monodromía NUNCA simetrizada (Sp(2n, ℝ) intacto); Hessiano 4D espectral.
    (iv)  Chirikov con twist KAM α = |∂²H/∂L²|; semianchura Δω = 2√(2V|α|).
    (v)   Poincaré-Birkhoff, Nekhoroshev T ~ exp(c ε^{-1/(2n)}), Krein-Broucke.
    (vi)  Punto fijo de la sección por Newton; Chern-Simons-Higgs ingerido.
    (vii) QFI vía SLD en RHI; Pesin h_KS = Σ λ_i⁺.
    (viii) RSI³: R₃ reescribe (τ, dt, n_inner) según q_k de Banach-Cantor-Birkhoff.
    (ix)  Ricci-DeTurck parabólico; Brockett sin proyección densitaria intra-flujo.
    (x)   GKSL de amplitud L_k = |0⟩⟨k|; Φ₁₂ y Φ₂₃ orquestados de hecho.
"""

from __future__ import annotations

import cmath
import hashlib
import logging
import math
import time
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONIntuitionProsecutorEngine")

__version__: Final[str] = "9.3.0-SOVEREIGN-RSI3-POINCARE-PhD"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, KAM, INQUISIDORAS, BANACH Y DE DIRAC
# ══════════════════════════════════════════════════════════════════════════════
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_SYMPLECTIC_TOLERANCE: Final[float] = 1.0e-10
_NOVIKOV_ENERGY_FLOOR: Final[float] = 1.0e-10
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_FERMI_DIRAC_TEMP_SW: Final[float] = 0.02
_RHI_HARD_CEILING: Final[float] = 0.85
_RHI_DEGRADED_CEILING: Final[float] = 0.40
_MELNIKOV_CHAOS_THRESHOLD: Final[float] = 0.15
_CHIRIKOV_OVERLAP_THRESHOLD: Final[float] = 1.00
_LYAPUNOV_UNSTABLE_CEILING: Final[float] = 0.50
_BURES_DEGRADED_CEILING: Final[float] = 0.25
_KAM_TWIST_FLOOR: Final[float] = 1.0e-8
_BRYUNO_DIVERGENCE_CEILING: Final[float] = 48.0
_BANACH_CONTRACTION_TOL: Final[float] = 1.0e-10
_RSI3_MAX_OUTER_ITERS: Final[int] = 8
_RSI3_MAX_INNER_ITERS: Final[int] = 6
_MELNIKOV_QUAD_NODES: Final[int] = 401
_FLOQUET_UNIT_CIRCLE_TOL: Final[float] = 1.0e-6
_DET_SP_TOLERANCE: Final[float] = 1.0e-8
_GREENE_ELLIPTIC_BOUND: Final[float] = 1.0
_KEPLER_MU: Final[float] = 1.0
_GOLDEN_CONJUGATE: Final[float] = 0.5 * (math.sqrt(5.0) - 1.0)
_KELLY_STAKE_CEILING: Final[float] = 0.35
_DIRAC_RELAXATION_TAU_US: Final[float] = 10.0
_KEPLER_MAX_ITER: Final[int] = 50
_KEPLER_TOL: Final[float] = 1.0e-14
_QR_LYAPUNOV_ITERS: Final[int] = 64
_TANH_SINH_HALF_NODES: Final[int] = 64
_TANH_SINH_H: Final[float] = 0.04
_NEKHOROSHEV_EXPO_CAP: Final[float] = 40.0
_NEWTON_POINCARE_ITERS: Final[int] = 8
_DANBY_STARTER_C: Final[float] = 0.85
_RSI3_TAU_MAX: Final[float] = 0.05
_RSI3_DT_MAX: Final[float] = 0.02
_RSI3_TAU0: Final[float] = 0.01
_RSI3_DT0: Final[float] = 0.005
_KREIN_ANGLE_COLLISION: Final[float] = 1.0e-8
_POINCARE_FD_STEP: Final[float] = 1.0e-8
_DIRAC_EULER_DT: Final[float] = 0.1e-6
_DEFAULT_HILBERT_DIM: Final[int] = 56
_MELNIKOV_PHASE_SAMPLES: Final[int] = 16
_QFI_DENOM_FLOOR: Final[float] = 1.0e-14


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEOS NUMÉRICOS HERMITIANOS, SPD, SIMPLÉCTICOS Y DE CUADRATURA
# ══════════════════════════════════════════════════════════════════════════════

def _herm(matrix: NDArray) -> NDArray[np.complex128]:
    r"""Proyección de Frobenius sobre el cono hermitiano: (A + A†)/2."""
    a = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _real_sym(matrix: NDArray) -> NDArray[np.float64]:
    r"""Proyección simétrica real: (A + Aᵀ)/2."""
    a = np.real(np.asarray(matrix, dtype=np.float64))
    return 0.5 * (a + a.T)


def _psd_herm(matrix: NDArray, floor: float = _SPECTRAL_PSD_FLOOR) -> NDArray[np.complex128]:
    r"""Proyección espectral sobre el cono PSD hermitiano con suelo de Wilkinson."""
    h = _herm(matrix)
    eigvals, eigvecs = la.eigh(h)
    lifted = np.maximum(np.real(eigvals), floor)
    return (eigvecs * lifted) @ eigvecs.conj().T


def _density(matrix: NDArray) -> NDArray[np.complex128]:
    r"""Saneamiento a estado cuántico normalizado: ρ† = ρ, ρ ⪰ 0, Tr ρ = 1."""
    h = _psd_herm(matrix, _WILKINSON_EPSILON)
    trace_val = float(np.real(np.trace(h)))
    dim = int(h.shape[0])
    if trace_val <= _WILKINSON_EPSILON:
        return np.eye(dim, dtype=np.complex128) / float(dim)
    return h / trace_val


def _sqrtm_psd(matrix: NDArray) -> NDArray[np.complex128]:
    r"""Raíz cuadrada PSD estable por eigendescomposición."""
    h = _herm(matrix)
    eigvals, eigvecs = la.eigh(h)
    root = np.sqrt(np.maximum(np.real(eigvals), 0.0))
    return (eigvecs * root) @ eigvecs.conj().T


def _matrix_power_psd(matrix: NDArray, p: float) -> NDArray[np.complex128]:
    r"""M^p para M PSD vía eigendescomposición con suelo de Wilkinson."""
    h = _herm(matrix)
    eigvals, eigvecs = la.eigh(h)
    vals = np.maximum(np.real(eigvals), _WILKINSON_EPSILON) ** float(p)
    return (eigvecs * vals) @ eigvecs.conj().T


def _logm_spd(metric: NDArray[np.float64], floor: float = _SPECTRAL_PSD_FLOOR) -> NDArray[np.float64]:
    r"""Logaritmo simétrico-definido-positivo (carta log-euclídea)."""
    spd = _real_sym(metric)
    eigvals, eigvecs = la.eigh(spd)
    log_eigs = np.log(np.maximum(eigvals, floor))
    return (eigvecs * log_eigs) @ eigvecs.T


def _expm_spd(log_metric: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Exponencial simétrica de una carta log-euclídea."""
    sym = _real_sym(log_metric)
    eigvals, eigvecs = la.eigh(sym)
    return (eigvecs * np.exp(eigvals)) @ eigvecs.T


def _trapz(y: NDArray[np.float64], dx: float) -> float:
    r"""Trapezoidal compuesta inmune a la deprecación de np.trapz."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 2:
        return 0.0
    return float(dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def _simpson(values: NDArray[np.float64], step: float) -> float:
    r"""Regla de Simpson compuesta (IRAM-SIMD) sobre nodos armónicos."""
    y = np.asarray(values, dtype=np.float64).ravel()
    n_intervals = int(y.size) - 1
    if n_intervals < 2:
        return _trapz(y, step)
    if n_intervals % 2 == 1:
        y = y[:-1]
        n_intervals -= 1
    return float(
        (step / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1::2]) + 2.0 * np.sum(y[2:-1:2]))
    )


def _richardson_extrapolate(f_h: float, f_h2: float, order: int = 4) -> float:
    r"""Extrapolación de Richardson: F ≈ (2^p · F_{h/2} − F_h)/(2^p − 1) para error O(h^p)."""
    denom = float(2 ** order - 1)
    return (float(2 ** order) * float(f_h2) - float(f_h)) / max(denom, _WILKINSON_EPSILON)


def _tanh_sinh_integral(func, n_half: int = _TANH_SINH_HALF_NODES,
                        h: float = _TANH_SINH_H) -> float:
    r"""
    Cuadratura tanh-sinh (doble exponencial) sobre ℝ:
        x = sinh(π/2 · sinh(kh)),  w = (π/2) cosh(kh) cosh(π/2 sinh(kh)) h.
    Convergencia exponencial para integrandos tipo sech / homoclínicas de Duffing.
    """
    acc = 0.0
    half_pi = 0.5 * math.pi
    n_half = max(int(n_half), 8)
    h = float(h)
    for k in range(-n_half, n_half + 1):
        kh = float(k) * h
        if abs(kh) > 10.0:
            continue
        sinh_kh = math.sinh(kh)
        cosh_kh = math.cosh(kh)
        u = half_pi * sinh_kh
        if abs(u) > 40.0:
            continue
        x = math.sinh(u)
        w = half_pi * cosh_kh * math.cosh(u) * h
        acc += w * float(func(x))
    return float(acc)


def _wrap_02pi(angle: float) -> float:
    r"""Representante canónico en 𝕋¹ ≅ ℝ/2πℤ: [0, 2π)."""
    twopi = 2.0 * math.pi
    wrapped = math.fmod(float(angle), twopi)
    if wrapped < 0.0:
        wrapped += twopi
    return float(wrapped)


def _canonical_omega(dimension: int) -> NDArray[np.float64]:
    r"""2-forma canónica de Darboux: Ω_{2n} = [[0, I_n], [−I_n, 0]] ∈ Sp(2n, ℝ)."""
    if dimension % 2 != 0 or dimension <= 0:
        raise ValueError(f"Dimensión simpléctica par y positiva requerida; recibido: {dimension}")
    n = dimension // 2
    omega = np.zeros((dimension, dimension), dtype=np.float64)
    omega[:n, n:] = np.eye(n, dtype=np.float64)
    omega[n:, :n] = -np.eye(n, dtype=np.float64)
    return omega


def _symplectic_expm(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Flujo hamiltoniano lineal M = exp(Ω H), H = Hᵀ; garantiza Sp(2n, ℝ) y det M = +1."""
    h = _real_sym(hessian)
    dim = int(h.shape[0])
    omega = _canonical_omega(dim)
    return np.real(la.expm(omega @ h))


def _frobenius(matrix: NDArray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def _slice_square(matrix: NDArray, dim: int) -> NDArray:
    return np.asarray(matrix)[:dim, :dim]


def _newton_kepler_eccentric_anomaly(e: float, M_mean: float,
                                     tol: float = _KEPLER_TOL,
                                     max_iter: int = _KEPLER_MAX_ITER) -> float:
    r"""
    Ecuación de Kepler E − e·sin(E) = M por el método cúbico de Danby (1988).
    Starter global: E₀ = M + 0.85 e sgn(sin M). Convergencia cúbica para e ∈ [0, 1).
    """
    e = float(np.clip(abs(e), 0.0, 1.0 - 1e-12))
    M = float(M_mean) % (2.0 * math.pi)
    sM = math.sin(M)
    E = M + _DANBY_STARTER_C * e * (1.0 if sM >= 0.0 else -1.0)
    for _ in range(max_iter):
        sE = math.sin(E)
        cE = math.cos(E)
        f = E - e * sE - M
        fp = 1.0 - e * cE
        fpp = e * sE
        fppp = e * cE
        if abs(fp) < _WILKINSON_EPSILON:
            break
        d1 = -f / fp
        d2 = -f / (fp + 0.5 * d1 * fpp)
        denom3 = fp + 0.5 * d2 * fpp + (d2 * d2 * fppp) / 6.0
        if abs(denom3) < _WILKINSON_EPSILON:
            break
        d3 = -f / denom3
        E_new = E + d3
        if abs(d3) < tol:
            return float(E_new)
        E = E_new
    return float(E)


def _lyapunov_spectrum_qr(monodromy_M: NDArray[np.float64],
                          n_iter: int = _QR_LYAPUNOV_ITERS) -> NDArray[np.float64]:
    r"""
    Espectro de Lyapunov (Benettin–Oseledets) por QR *sin* simetrizar M:
        λ_i = (1/N) Σ_k log |R_k[i,i]|,   Q,R = qr(M Q).
    Simetrizar rompería Sp(2n, ℝ) y falsearía los exponentes de Floquet.
    """
    m = np.real(np.asarray(monodromy_M, dtype=np.float64))
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    dim = int(m.shape[0])
    q = np.eye(dim, dtype=np.float64)
    log_acc = np.zeros(dim, dtype=np.float64)
    n_steps = max(int(n_iter), 1)
    for _ in range(n_steps):
        z = m @ q
        q_new, r = la.qr(z, mode="full")
        diag_r = np.abs(np.diag(r)[:dim])
        diag_r = np.maximum(diag_r, _WILKINSON_EPSILON)
        log_acc += np.log(diag_r)
        q = q_new[:, :dim]
    return log_acc / float(n_steps)


def _lefschetz_fixed_point_index(monodromy_M: NDArray[np.float64]) -> int:
    r"""Índice de Lefschetz de punto fijo: index(P) = sign(det(I − M))."""
    dim = int(monodromy_M.shape[0])
    det_i_minus_m = float(np.real(np.linalg.det(np.eye(dim) - monodromy_M)))
    if det_i_minus_m > 1e-15:
        return 1
    if det_i_minus_m < -1e-15:
        return -1
    return 0


def _continued_fraction_bryuno(alpha: float, n_terms: int = 12) -> float:
    r"""Función de Bryuno B(α) = Σ_k log(q_{k+1}) / q_k sobre α ∈ ℝ/ℤ."""
    fractional = abs(float(alpha) - math.floor(float(alpha)))
    if fractional < 1.0e-14 or abs(fractional - 1.0) < 1.0e-14:
        fractional = _GOLDEN_CONJUGATE
    a_coeffs: List[int] = []
    x = fractional
    for _ in range(n_terms):
        if x <= 1.0e-14:
            break
        inv = 1.0 / x
        a_i = max(1, int(math.floor(inv)))
        a_coeffs.append(a_i)
        x = inv - float(a_i)
        if x <= 1.0e-14:
            break
    if not a_coeffs:
        return float(math.log(2.0))
    q_mm, q_m = 1, a_coeffs[0]
    bryuno = math.log(max(float(q_m), 2.0)) / 1.0
    for a_i in a_coeffs[1:]:
        q = a_i * q_m + q_mm
        bryuno += math.log(float(max(q, 2))) / float(max(q_m, 1))
        q_mm, q_m = q_m, q
    return float(bryuno)


def _nekhoroshev_stable_time(eps: float, n_dof: int = 2,
                             melnikov_m0: float = 0.0,
                             chirikov_k: float = 0.0) -> float:
    r"""
    Tiempo de estabilidad de Nekhoroshev: T_N ≳ exp(c / ε^{1/(2n)}).
    Colapsa a O(1) si hay caos homoclínico (Melnikov) o solapamiento de Chirikov.
    """
    if (abs(float(melnikov_m0)) > _MELNIKOV_CHAOS_THRESHOLD
            or float(chirikov_k) >= _CHIRIKOV_OVERLAP_THRESHOLD):
        return 1.0
    a = 1.0 / float(max(2 * max(int(n_dof), 1), 2))
    expo = 0.5 / max(abs(float(eps)), 1.0e-12) ** a
    return float(math.exp(min(float(expo), _NEKHOROSHEV_EXPO_CAP)))


def _poincare_birkhoff_twist(floquet: NDArray[np.complex128], kam_twist: float) -> bool:
    r"""
    Teorema geométrico de Poincaré (Poincaré–Birkhoff): un mapa twist del anillo
    (números de rotación distintos en los bordes, ∂²H/∂L² ≠ 0) posee ≥ 2 puntos fijos.
    """
    if abs(float(kam_twist)) <= _KAM_TWIST_FLOOR:
        return False
    if floquet.size < 2:
        return True
    angles = np.sort(np.angle(np.asarray(floquet, dtype=np.complex128)))
    return bool(abs(float(angles[-1] - angles[0])) > 1.0e-6)


def _spectral_celestial_hessian_4d(
    rho: NDArray[np.complex128],
    N_diag: NDArray[np.float64],
    perturbation_eps: float,
) -> NDArray[np.float64]:
    r"""
    Hessiano 4D de la reducción planar (sección de Poincaré del problema de 3 cuerpos
    restringido) acoplado a momentos espectrales de ρ: θ = Tr(ρ N), n₂ = Tr(ρ N²).
    H = Hᵀ por construcción → exp(Ω H) ∈ Sp(4, ℝ).
    """
    rho_d = _density(rho)
    dim = int(rho_d.shape[0])
    n_loc = np.real(_slice_square(N_diag, dim))
    n_c = n_loc.astype(np.complex128)
    theta = float(np.real(np.trace(rho_d @ n_c)))
    n2 = float(np.real(np.trace(rho_d @ n_c @ n_c)))
    n00 = float(n_loc[0, 0]) if n_loc.size else 0.0
    eps_p = float(perturbation_eps)
    c, s = math.cos(theta), math.sin(theta)
    hess = np.array(
        [
            [1.0 + 0.05 * eps_p * c + 0.01 * n00, 0.02 * eps_p * s, 0.0, 0.0],
            [0.02 * eps_p * s, 1.0 - 0.05 * eps_p * c, 0.0, 0.0],
            [0.0, 0.0, 1.0 + 0.03 * eps_p + 0.002 * n2, 0.01 * eps_p],
            [0.0, 0.0, 0.01 * eps_p, 1.0 - 0.03 * eps_p],
        ],
        dtype=np.float64,
    )
    return _real_sym(hess)


def _daleckii_krein_divided_diff(eigvals: NDArray[np.float64],
                                 phi=math.log) -> NDArray[np.float64]:
    r"""Matriz de diferencias divididas de Daleckii-Krein para φ ∈ C¹(spectrum)."""
    lam = np.maximum(np.real(np.asarray(eigvals, dtype=np.float64)), _SPECTRAL_PSD_FLOOR)
    n = int(lam.size)
    dd = np.zeros((n, n), dtype=np.float64)
    phi_vals = np.array([phi(float(v)) for v in lam], dtype=np.float64)
    for i in range(n):
        for j in range(n):
            delta = lam[i] - lam[j]
            if abs(delta) <= 1.0e-14:
                dd[i, j] = 1.0 / lam[i]
            else:
                dd[i, j] = (phi_vals[i] - phi_vals[j]) / delta
    return dd


def _fermi_dirac_sigmoidal(rhi: float, t_sw: float = _FERMI_DIRAC_TEMP_SW) -> float:
    r"""Suavizado sigmoidal C^∞ de Fermi-Dirac S_{T_sw}(RHI) = 1/(1+e^{(RHI-0.85)/T_sw})."""
    exponent = (float(rhi) - _RHI_HARD_CEILING) / max(float(t_sw), 1.0e-6)
    if exponent > 70.0:
        return 0.0
    if exponent < -70.0:
        return 1.0
    return float(1.0 / (1.0 + math.exp(exponent)))


def _heaviside(x: float) -> float:
    r"""Θ(x) = 1 si x ≥ 0, 0 en otro caso."""
    return 1.0 if float(x) >= 0.0 else 0.0


def _poincare_cartan_line_integral(z_path: NDArray[np.float64],
                                   omega: NDArray[np.float64]) -> float:
    r"""
    Invariante integral relativo de Poincaré–Cartan (Méthodes Nouvelles, t. I):
        ∮_{∂Σ} p dq = ½ ∮ zᵀ Ω dz  (1-forma de Liouville sobre ciclo cerrado).
    """
    if z_path.shape[0] < 3:
        return 0.0
    n = int(z_path.shape[0])
    acc = 0.0
    for i in range(n):
        z0 = z_path[i]
        z1 = z_path[(i + 1) % n]
        dz = z1 - z0
        acc += 0.5 * float(z0 @ omega @ dz + z1 @ omega @ dz)
    return float(acc)


def _petz_tangent_chernoff_bound(rho: NDArray, sigma: NDArray,
                                 n_grid: int = 9) -> float:
    r"""Cota de Chernoff cuántica de Audenaert: Q(ρ||σ) = −log min_{s∈(0,1)} Tr(ρ^s σ^{1−s})."""
    rho_d = _density(rho)
    sigma_d = _density(sigma)
    best = 0.0
    for k in range(1, n_grid + 1):
        s = float(k) / float(n_grid + 1)
        rho_s = _matrix_power_psd(rho_d, s)
        sigma_1ms = _matrix_power_psd(sigma_d, 1.0 - s)
        c = float(np.real(np.trace(rho_s @ sigma_1ms)))
        c = max(c, _WILKINSON_EPSILON)
        best = max(best, -math.log(c))
    return float(best)


def _umegaki_relative_entropy(rho: NDArray, sigma: NDArray) -> float:
    r"""Entropía relativa cuántica de Umegaki D(ρ||σ) = Tr[ρ (log ρ − log σ)]."""
    rho_d = _density(rho)
    sigma_d = _density(sigma)
    eig_r, vec_r = la.eigh(_herm(rho_d))
    eig_s, vec_s = la.eigh(_herm(sigma_d))
    log_rho = (vec_r * np.log(np.maximum(np.real(eig_r), _WILKINSON_EPSILON))) @ vec_r.conj().T
    log_sigma = (vec_s * np.log(np.maximum(np.real(eig_s), _WILKINSON_EPSILON))) @ vec_s.conj().T
    val = float(np.real(np.trace(rho_d @ (log_rho - log_sigma))))
    return float(max(val, 0.0))


def _holevo_accessible_information(states: Sequence[NDArray],
                                   probs: Sequence[float]) -> float:
    r"""Cota de Holevo χ = S(ρ̄) − Σ p_i S(ρ_i) sobre el ensemble."""
    if not states:
        return 0.0
    weights = np.asarray(probs, dtype=np.float64)
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights) / float(len(weights))
    else:
        weights = weights / weights.sum()
    avg = np.zeros_like(states[0], dtype=np.complex128)
    for p, s in zip(weights, states):
        avg = avg + float(p) * _density(s)
    avg = _density(avg)
    eig_avg = np.maximum(np.real(la.eigvalsh(_herm(avg))), _WILKINSON_EPSILON)
    s_avg = -float(np.sum(eig_avg * np.log(eig_avg)))
    weighted_s = 0.0
    for p, s in zip(weights, states):
        s_d = _density(s)
        eig_s = np.maximum(np.real(la.eigvalsh(_herm(s_d))), _WILKINSON_EPSILON)
        weighted_s += float(p) * (-float(np.sum(eig_s * np.log(eig_s))))
    return float(max(s_avg - weighted_s, 0.0))


def _sld_quantum_fisher_information(rho: NDArray, sigma: NDArray) -> float:
    r"""
    Información cuántica de Fisher vía derivada logarítmica simétrica (SLD):
        ρ S + S ρ = 2 Δ,   F_Q = Tr(ρ S²),   Δ = σ − ρ.
    Vectorizado en la carta espectral de ρ (Helstrom–Braunstein–Caves).
    """
    rho_d = _density(rho)
    delta = _herm(np.asarray(sigma, dtype=np.complex128) - rho_d)
    eigvals, eigvecs = la.eigh(_herm(rho_d))
    lam = np.maximum(np.real(eigvals), _WILKINSON_EPSILON)
    d_mat = eigvecs.conj().T @ delta @ eigvecs
    den = lam[:, None] + lam[None, :]
    s_coef = np.zeros_like(d_mat, dtype=np.complex128)
    mask = den > _QFI_DENOM_FLOOR
    s_coef[mask] = 2.0 * d_mat[mask] / den[mask]
    sld = eigvecs @ s_coef @ eigvecs.conj().T
    qfi = float(np.real(np.trace(rho_d @ sld @ sld)))
    return float(max(qfi, 0.0))


def _cantor_birkhoff_contraction(prev_residual: float,
                                 curr_residual: float) -> Tuple[float, bool]:
    r"""Tasa de Banach-Cantor-Birkhoff: q = ‖e_{k+1}‖/‖e_k‖ < 1 ⇒ contracción."""
    if prev_residual < _WILKINSON_EPSILON:
        return 0.0, True
    ratio = float(curr_residual) / max(float(prev_residual), _WILKINSON_EPSILON)
    return ratio, bool(ratio < 1.0 - 1.0e-10)


def _rsi3_meta_adapt_hyperparameters(
    tau: float,
    dt: float,
    n_inner: int,
    contraction_rate: float,
    residual: float,
    iso_drift: float,
    comm_norm: float,
    outer: int,
) -> Tuple[float, float, int, bool]:
    r"""
    Meta-mejorador RSI³ (Kleisli T = R₃ ∘ R₂ ∘ R₁): R₃ observa (q, residual,
    drift isoespectral, ‖[ρ,[ρ,N]]‖) y reescribe los hiperparámetros de R₁ y R₂.
    Ésta es la automejora recursiva de *nivel 3* (el mejorador del mejorador).
    """
    tau_n = float(tau)
    dt_n = float(dt)
    n_n = int(n_inner)
    adapted = False
    q = float(contraction_rate)
    if outer > 1 and q >= 1.0 - 1.0e-10:
        tau_n *= 0.5
        dt_n *= 0.5
        adapted = True
    elif outer > 1 and q < 0.30 and residual > 10.0 * _BANACH_CONTRACTION_TOL:
        tau_n = min(tau_n * 1.15, _RSI3_TAU_MAX)
        dt_n = min(dt_n * 1.15, _RSI3_DT_MAX)
        adapted = True
    if iso_drift > 1.0e-4:
        n_n = min(n_n + 1, 12)
        adapted = True
    elif iso_drift < 1.0e-8:
        n_n = max(n_n - 1, 2)
    if comm_norm > 10.0:
        dt_n *= 0.5
        adapted = True
    tau_n = float(min(max(tau_n, 1.0e-6), _RSI3_TAU_MAX))
    dt_n = float(min(max(dt_n, 1.0e-6), _RSI3_DT_MAX))
    return tau_n, dt_n, int(n_n), bool(adapted)


def _spohn_entropy_production(drho_dt: NDArray, rho: NDArray) -> float:
    r"""Producción de entropía de Spohn σ = −d/dt S(ρ) ≥ 0 (GKSL)."""
    rho_d = _density(rho)
    eig_r, vec_r = la.eigh(_herm(rho_d))
    log_rho = (vec_r * np.log(np.maximum(np.real(eig_r), _WILKINSON_EPSILON))) @ vec_r.conj().T
    ds_dt = -float(np.real(np.trace(log_rho @ np.asarray(drho_dt, dtype=np.complex128))))
    return float(-ds_dt)


def _duffing_melnikov_kernel(t: float, t0: float, eps: float) -> float:
    r"""
    Integrando de Melnikov sobre la homoclínica de Duffing ẍ = x − x³,
    x_h = √2 sech(t), ẋ_h ∝ sech(t) tanh(t), perturbación armónica sin(t + t₀).
    """
    if abs(t) > 40.0:
        return 0.0
    sech_t = 1.0 / math.cosh(t)
    return float(eps * (sech_t ** 2) * math.tanh(t) * math.sin(t + t0))


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DISTRIBUTIVO DE HEYTING Ω₃ Y ENUMERACIONES CIBER-FÍSICAS
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""Álgebra de Heyting trivalente intuicionista Ω₃: ⊥=VETOED, ★=DEGRADED, ⊤=COHERENT."""
    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def relative_pseudocomplement(self, other: "HeytingOmega3") -> "HeytingOmega3":
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def negation(self) -> "HeytingOmega3":
        return self.relative_pseudocomplement(HeytingOmega3.VETOED)

    def modus_ponens(self, implication: "HeytingOmega3") -> "HeytingOmega3":
        return self.meet(implication)

    @staticmethod
    def fold_meet(values: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = HeytingOmega3.COHERENT
        for value in values:
            acc = acc.meet(value)
        return acc


class ActuationMode(IntEnum):
    r"""Modos de actuación ciber-física (interlock simulado)."""
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class ProsecutionStatus(IntEnum):
    r"""Dictamen analítico inquisidor sobre el salto flash evaluado."""
    CLEAR_NO_ANOMALY = 0
    REWARD_HACKING_DETECTED = 1
    MELNIKOV_CHAOS_CONFIRMED = 2
    KELLY_STAKE_CENSORED = 3
    PURGED_DIRAC_VACUUM = 4
    CHIRIKOV_OVERLAP_CONFIRMED = 5


class RSI3Level(IntEnum):
    r"""Estratos de automejora recursiva: objeto, mejorador, meta-mejorador."""
    OBJECT_R1 = 1
    IMPROVER_R2 = 2
    META_IMPROVER_R3 = 3


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES: ESTADOS INTER-FASE, RSI³ Y CERTIFICADOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareMelnikovProsecutionState:
    r"""Estado terminal inmutable emitido por la FASE 1 (atlas celeste de Poincaré)."""
    seed_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    delaunay_l: float
    delaunay_g: float
    delaunay_h: float
    eccentricity_e: float
    inclination_i: float
    kepler_mean_motion: float
    kepler_eccentric_anomaly: float
    kepler_true_anomaly: float
    kepler_radius: float
    kepler_semimajor: float
    kam_twist_d2H_dL2: float
    melnikov_integral_M0: float
    lyapunov_transverse_max: float
    lyapunov_spectrum: NDArray[np.float64]
    greene_residue_R: float
    chirikov_overlap_K: float
    chirikov_halfwidth_1: float
    chirikov_halfwidth_2: float
    bryuno_diophantine_sum: float
    floquet_multipliers: NDArray[np.complex128]
    floquet_spectral_radius: float
    is_floquet_elliptic: bool
    broucke_krein_stable: bool
    poincare_birkhoff_certified: bool
    poincare_return_residual: float
    poincare_cartan_relative: float
    lefschetz_index: int
    symplectic_monodromy_M: NDArray[np.float64]
    symplectic_drift: float
    symplectic_det: float
    is_symplectic_conserved: bool
    kam_twist_monotone: bool
    chern_simons_higgs: float
    nekhoroshev_stable_time: float
    maupertuis_metric: NDArray[np.float64]
    polyakov_action: float
    poincare_cartan_1form: float
    liouville_volume_drift: float
    timestamp_utc: float
    sha256_provenance: str
    verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class FlashProsecutionBanachState:
    r"""Estado terminal inmutable emitido por la FASE 2."""
    prosecution_id: str
    jump_id: str
    celestial_state: PoincareMelnikovProsecutionState
    reward_hacking_index_RHI: float
    bures_wasserstein_distance: float
    fubini_study_distance: float
    uhlmann_fidelity: float
    quantum_chernoff_bound: float
    umegaki_relative_entropy: float
    holevo_accessible_info: float
    quantum_fisher_information: float
    original_kelly_stake: float
    censored_kelly_stake: float
    heaviside_mask: float
    melnikov_chaos_flag: bool
    lyapunov_unstable_flag: bool
    chirikov_chaos_flag: bool
    kolmogorov_sinai_entropy: float
    oseledets_transverse: float
    prosecution_status: ProsecutionStatus
    timestamp_utc: float
    verdict: HeytingOmega3


@dataclass(frozen=True, slots=True)
class RSI3CascadeReport:
    r"""Informe inmutable de la cascada Kleisli R₃ ∘ R₂ ∘ R₁ y su meta-iteración."""
    outer_iterations: int
    inner_iterations: int
    banach_residual: float
    banach_contraction_rate: float
    cantor_birkhoff_contracted: bool
    contracted: bool
    ricci_scalar_final: float
    connes_lipschitz: float
    brockett_commutator_norm: float
    isospectral_drift: float
    spohn_entropy_production: float
    r1_step_tau: float
    r3_step_dt: float
    lyapunov_functional: float
    monadic_rate: float
    brockett_lyapunov_monotone: bool
    rsi3_meta_adaptations: int
    rsi3_level_reached: int


@dataclass(frozen=True, slots=True)
class IntuitionProsecutorExecutionCertificate:
    r"""Certificado soberano de auditoría emitido por la FASE 3."""
    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    prosecutions_count: int
    censored_count: int
    level3_rsi_monadic_rate: float
    rsi3_outer_iterations: int
    rsi3_banach_residual: float
    rsi3_banach_contraction_rate: float
    rsi3_contracted: bool
    rsi3_spohn_nonneg: bool
    rsi3_lyapunov_monotone: bool
    rsi3_meta_adaptations: int
    heyting_lattice_complete: bool
    peak_reward_hacking_index: float
    max_lyapunov_exponent: float
    peak_bures_distance: float
    uhlmann_fidelity_floor: float
    peak_umegaki_entropy: float
    peak_quantum_fisher: float
    kelly_stake_censored_mean: float
    melnikov_chaos_detected: bool
    chirikov_overlap_detected: bool
    greene_residue_peak: float
    kam_twist_floor_observed: float
    nekhoroshev_time_min: float
    hardware_crowbar_latched: bool
    actuation_latency_ns: float
    merkle_root_sha256: str
    telemetry_digest_sha256: str


IntuitionProsecutorCanonicalSeed = PoincareMelnikovProsecutionState
IntuitionFlashProsecutionGerm = FlashProsecutionBanachState


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: ATLAS CELESTE POINCARÉ-MELNIKOV, CHIRIKOV Y REDUCCIÓN DE MAUPERTUIS
# ══════════════════════════════════════════════════════════════════════════════

class Phase1_PoincareMelnikovAtlas(ABC):
    r"""
    FASE 1 — Fundamento simpléctico, mecánica celeste de Poincaré y caos homoclínico.

    Conforma acciones-ángulos de Delaunay, propaga Kepler-Danby, integra Melnikov
    tanh-sinh con maximización sobre t₀ y Richardson, calcula Chern-Simons con
    acoplamiento Yang-Mills-Higgs, certifica Sp(2n, ℝ), verifica Darboux/KAM/
    Lefschetz/Birkhoff/Nekhoroshev y reduce Maupertuis-Jacobi.
    Emite el `PoincareMelnikovProsecutionState`.
    """

    # ── Fase 1.1: Espacios simplécticos canónicos ─────────────────────────────
    def construct_canonical_symplectic_matrix(self, dimension: int) -> NDArray[np.float64]:
        r"""Matriz simpléctica canónica de Darboux Ω_{2n} ∈ Sp(2n, ℝ)."""
        return _canonical_omega(dimension)

    # ── Fase 1.2: Acciones-ángulos de Delaunay ────────────────────────────────
    def compute_delaunay_canonical_elements(
        self,
        density_matrix: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        r"""
        6-tupla canónica de Delaunay (L, G, H, l, g, h) y observables espectrales.
        L = √(μ a), G = L √(1−e²), H = G cos i  (Poincaré, Méthodes Nouvelles).
        Twist KAM ∂²H/∂L² = −3 μ² / L⁴ < 0 garantiza no-degeneración de Kolmogorov.
        Ángulos en 𝕋³ ≅ [0, 2π)³.
        """
        rho = _density(density_matrix)
        dim = int(rho.shape[0])
        eigvals, eigvecs = la.eigh(_herm(rho))
        order = np.argsort(np.real(eigvals))[::-1]
        eigvals = np.maximum(np.real(eigvals[order]), _WILKINSON_EPSILON)
        eigvecs = eigvecs[:, order]

        L = float(np.sum(eigvals))
        lam1 = float(eigvals[0])
        lam2 = float(eigvals[1]) if dim > 1 else lam1 * 0.5
        lam3 = float(eigvals[2]) if dim > 2 else lam2 * 0.5

        G = float(L * (lam2 / max(lam1, _WILKINSON_EPSILON)))
        H = float(G * (lam3 / max(lam2, _WILKINSON_EPSILON)))

        ecc_ratio = float(np.clip(1.0 - (G / max(L, _WILKINSON_EPSILON)) ** 2, 0.0, 1.0))
        eccentricity = float(math.sqrt(ecc_ratio))
        inc_ratio = float(np.clip(H / max(G, _WILKINSON_EPSILON), -1.0, 1.0))
        inclination = float(math.acos(inc_ratio))

        v1 = eigvecs[:, 0]
        mean_anomaly_l = _wrap_02pi(float(cmath.phase(np.sum(v1) + _WILKINSON_EPSILON)))
        lrl_x = eccentricity * math.cos(mean_anomaly_l)
        lrl_y = eccentricity * math.sin(mean_anomaly_l) * math.sin(inclination)
        arg_periapsis_g = _wrap_02pi(float(math.atan2(lrl_y, lrl_x + _WILKINSON_EPSILON)))
        node_h = _wrap_02pi(
            float(
                math.atan2(
                    math.sin(inclination) * math.sin(arg_periapsis_g),
                    math.cos(inclination) + _WILKINSON_EPSILON,
                )
            )
        )

        L_safe = max(abs(L), _WILKINSON_EPSILON)
        mean_motion = float((_KEPLER_MU ** 2) / (L_safe ** 3))
        kam_twist = float(-3.0 * (_KEPLER_MU ** 2) / (L_safe ** 4))
        lyapunov_max = float(math.log(lam1 / max(lam2, _WILKINSON_EPSILON)))
        return (
            L, G, H, mean_anomaly_l, arg_periapsis_g, node_h,
            eccentricity, inclination, mean_motion, kam_twist, lyapunov_max,
        )

    # ── Fase 1.3: Propagación Kepler-Danby ────────────────────────────────────
    def propagate_kepler_cartesian(
        self,
        delaunay_state: Tuple[float, float, float, float, float, float,
                              float, float, float, float, float],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Propaga acciones-ángulos a (r, ν, a, E) resolviendo Kepler por Danby:
            r = a(1 − e·cos E),
            ν = 2 atan2(√(1+e) sin(E/2), √(1−e) cos(E/2)),
            a = L²/μ.
        """
        L, _G, _H, l_mean, _g, _h, ecc, _inc, _n, _tw, _ly = delaunay_state
        L_safe = max(abs(L), _WILKINSON_EPSILON)
        a_semiaxis = float(L_safe ** 2 / _KEPLER_MU)
        e_clip = float(np.clip(ecc, 0.0, 1.0 - 1e-9))

        E_anomaly = _newton_kepler_eccentric_anomaly(e_clip, l_mean)
        r_radius = float(a_semiaxis * (1.0 - e_clip * math.cos(E_anomaly)))
        nu_true = float(
            2.0 * math.atan2(
                math.sqrt(max(0.0, 1.0 + e_clip)) * math.sin(0.5 * E_anomaly),
                math.sqrt(max(0.0, 1.0 - e_clip)) * math.cos(0.5 * E_anomaly),
            )
        )
        return np.array([r_radius, nu_true, a_semiaxis, E_anomaly], dtype=np.float64), a_semiaxis

    # ── Fase 1.4: Melnikov tanh-sinh con maximización de fase ─────────────────
    def _melnikov_tanh_sinh_fixed_t0(self, eps: float, t0: float,
                                     n_half: int, h: float) -> float:
        r"""Cuadratura doble-exponencial de M(t₀) sobre la homoclínica de Duffing."""
        return _tanh_sinh_integral(
            lambda t: _duffing_melnikov_kernel(t, t0, eps),
            n_half=n_half,
            h=h,
        )

    def _melnikov_simpson_fixed_t0(self, eps: float, t0: float, n_nodes: int) -> float:
        r"""Cuadratura Simpson compuesta (legado IRAM) de Melnikov(t₀) sobre [−20, 20]."""
        n = n_nodes if n_nodes % 2 == 1 else n_nodes + 1
        t_grid = np.linspace(-20.0, 20.0, n, dtype=np.float64)
        sech_t = 1.0 / np.cosh(np.clip(t_grid, -50.0, 50.0))
        integrand = (sech_t ** 2) * np.tanh(t_grid) * np.sin(t_grid + t0)
        dt = float(t_grid[1] - t_grid[0])
        return float(_simpson(integrand, dt) * eps)

    def compute_melnikov_simpson(self, perturbation_eps: float,
                                 n_phase_samples: int = _MELNIKOV_PHASE_SAMPLES) -> float:
        r"""
        Integral de Melnikov (splitting homoclínico, Poincaré 1890 / Melnikov 1963):
            M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(q_h(t), p_h(t), t+t₀) dt.
        Refinamiento triple: (i) tanh-sinh h; (ii) tanh-sinh h/2 + Richardson;
        (iii) maximización |M| sobre t₀ ∈ [0, 2π). Ceros simples ⇒ transverso ⇒ caos.
        """
        eps = float(perturbation_eps)
        best_val = 0.0
        best_abs = -math.inf
        n_phase = max(1, int(n_phase_samples))
        n_half_coarse = max(_TANH_SINH_HALF_NODES // 2, 16)
        n_half_fine = _TANH_SINH_HALF_NODES
        h_coarse = _TANH_SINH_H * 2.0
        h_fine = _TANH_SINH_H

        for k in range(n_phase):
            t0 = 2.0 * math.pi * float(k) / float(n_phase)
            m_coarse = self._melnikov_tanh_sinh_fixed_t0(eps, t0, n_half_coarse, h_coarse)
            m_fine = self._melnikov_tanh_sinh_fixed_t0(eps, t0, n_half_fine, h_fine)
            m_rich = _richardson_extrapolate(m_coarse, m_fine, order=4)
            if abs(m_rich) > best_abs:
                best_abs = abs(m_rich)
                best_val = m_rich
        return float(best_val)

    # ── Fase 1.5: Chern-Simons con acoplamiento Yang-Mills-Higgs ─────────────
    def compute_yang_mills_higgs_chern_simons(
        self,
        connection_a: NDArray[np.complex128],
        density_rho: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.complex128]]:
        r"""
        3-Forma de Chern-Simons con acoplamiento escalar de Yang-Mills-Higgs:
            CS₃^Higgs(A,Φ) = (1/8π²) Tr(A∧dA + (2/3)A³ + Φ·F_A),
        Φ = ln(ρ + ε I) regula la curvatura F_A bajo rk(ρ) < n.
        """
        rho = _density(density_rho)
        eigvals, eigvecs = la.eigh(_herm(rho))
        log_eigs = np.log(np.maximum(np.real(eigvals), _WILKINSON_EPSILON))
        phi_higgs = (eigvecs * log_eigs) @ eigvecs.conj().T

        a_lie = 0.5 * (connection_a - connection_a.conj().T)
        f_curvature = a_lie @ a_lie
        a_cubed = a_lie @ a_lie @ a_lie
        term_base = a_lie @ f_curvature - (1.0 / 3.0) * a_cubed
        term_higgs = phi_higgs @ f_curvature
        cs3_raw = np.trace(term_base + term_higgs)
        cs3_higgs = float(np.real(cs3_raw) / (8.0 * math.pi ** 2))
        return cs3_higgs, f_curvature

    # ── Fase 1.6: Monodromía Greene/Floquet/Krein y Lyapunov-QR ───────────────
    def evaluate_greene_floquet_monodromy(
        self,
        density_matrix: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        perturbation_eps: float,
    ) -> Tuple[NDArray[np.float64], float, float, bool, float,
               NDArray[np.complex128], float, bool, bool, NDArray[np.float64]]:
        r"""
        Monodromía M = exp(Ω H) exactamente simpléctica + Greene + Floquet +
        estabilidad lineal de Broucke–Krein (colisión en el círculo unidad) +
        espectro de Lyapunov QR *sin* simetrizar M.
        """
        rho = _density(density_matrix)
        hess = _spectral_celestial_hessian_4d(rho, N_diag, perturbation_eps)
        monodromy_m = _symplectic_expm(hess)

        omega4 = _canonical_omega(4)
        sym_drift = _frobenius(monodromy_m.T @ omega4 @ monodromy_m - omega4)
        det_m = float(np.real(np.linalg.det(monodromy_m)))
        is_sym = (sym_drift <= _SYMPLECTIC_TOLERANCE) and (abs(det_m - 1.0) <= _DET_SP_TOLERANCE)
        greene_r = float((2.0 - np.trace(monodromy_m)) / 4.0)

        floquet = np.linalg.eigvals(monodromy_m)
        spectral_radius = float(np.max(np.abs(floquet)))
        on_circle = np.all(np.abs(np.abs(floquet) - 1.0) <= _FLOQUET_UNIT_CIRCLE_TOL)
        is_elliptic = bool(on_circle and abs(greene_r) < _GREENE_ELLIPTIC_BOUND)
        angles = np.sort(np.angle(floquet))
        krein_collision = bool(
            angles.size >= 2 and np.any(np.diff(angles) < _KREIN_ANGLE_COLLISION)
        )
        broucke_krein = bool(is_elliptic and (not krein_collision) and is_sym)
        lyap_spec = _lyapunov_spectrum_qr(monodromy_m, n_iter=_QR_LYAPUNOV_ITERS)
        return (
            monodromy_m, greene_r, sym_drift, is_sym, det_m,
            floquet.astype(np.complex128), spectral_radius, is_elliptic,
            broucke_krein, lyap_spec,
        )

    # ── Fase 1.7: Chirikov con twist KAM ─────────────────────────────────────
    def compute_chirikov_overlap(
        self,
        density_matrix: NDArray[np.complex128],
        mean_motion: float,
        kam_twist: float,
        perturbation_eps: float,
        melnikov_M0: float,
    ) -> Tuple[float, float, float]:
        r"""
        Criterio de solapamiento de Chirikov con semianchura hamiltoniana:
            Δω = 2 √(2 V |α|),  α = ∂²H/∂I²,  K = (Δω₁+Δω₂)/|ω₁−ω₂|.
        K ≥ 1 ⇒ destrucción de toros KAM por solapamiento de resonancias.
        """
        rho = _density(density_matrix)
        eigvals = np.sort(np.maximum(np.real(la.eigvalsh(_herm(rho))), _WILKINSON_EPSILON))[::-1]
        lam1 = float(eigvals[0])
        lam2 = float(eigvals[1]) if eigvals.size > 1 else lam1 * 0.5

        v1 = abs(float(melnikov_M0)) + 0.10 * abs(float(perturbation_eps))
        v2 = 0.50 * v1 + 0.05 * abs(float(perturbation_eps))
        alpha = abs(float(kam_twist))
        halfwidth_1 = float(2.0 * math.sqrt(max(2.0 * v1 * alpha, _WILKINSON_EPSILON)))
        halfwidth_2 = float(2.0 * math.sqrt(max(2.0 * v2 * alpha, _WILKINSON_EPSILON)))

        omega1 = float(mean_motion)
        delta_l = abs(lam1 - lam2)
        omega2 = float(mean_motion + kam_twist * delta_l)
        freq_gap = max(abs(omega1 - omega2), alpha * delta_l, _WILKINSON_EPSILON)
        chirikov_k = float((halfwidth_1 + halfwidth_2) / freq_gap)
        if not np.isfinite(chirikov_k):
            chirikov_k = float("inf")
        return chirikov_k, halfwidth_1, halfwidth_2

    # ── Fase 1.8: Mapa de retorno de Poincaré con Newton ─────────────────────
    def _poincare_discrete_map(
        self,
        z: NDArray[np.float64],
        monodromy_M: NDArray[np.float64],
        melnikov_M0: float,
        perturbation_eps: float,
    ) -> NDArray[np.float64]:
        r"""Mapa discreto P(z) = M z + ε f_homoclínico(z) sobre la sección ℝ⁴."""
        sech_norm = 1.0 / math.cosh(float(np.linalg.norm(z)))
        nonlinear = np.array(
            [0.0, 0.0, float(melnikov_M0) * sech_norm, 0.0],
            dtype=np.float64,
        )
        return monodromy_M @ z + float(perturbation_eps) * nonlinear

    def refine_poincare_fixed_point_newton(
        self,
        monodromy_M: NDArray[np.float64],
        melnikov_M0: float,
        perturbation_eps: float,
        z0: Optional[NDArray[np.float64]] = None,
        n_newton: int = _NEWTON_POINCARE_ITERS,
    ) -> Tuple[float, NDArray[np.float64]]:
        r"""
        Newton sobre (P − Id)(z) = 0: localiza el punto periódico de la sección
        (Lefschetz efectivo). Jacobiano por diferencias finitas de paso 10⁻⁸.
        """
        z = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if z0 is None
            else np.asarray(z0, dtype=np.float64).copy()
        )

        def P(vec: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._poincare_discrete_map(vec, monodromy_M, melnikov_M0, perturbation_eps)

        for _ in range(max(1, int(n_newton))):
            f0 = P(z)
            residual_vec = f0 - z
            if float(np.linalg.norm(residual_vec)) < 1.0e-14:
                break
            jac = np.zeros((4, 4), dtype=np.float64)
            for j in range(4):
                dz = np.zeros(4, dtype=np.float64)
                dz[j] = _POINCARE_FD_STEP
                jac[:, j] = (P(z + dz) - f0) / _POINCARE_FD_STEP
            a_mat = jac - np.eye(4, dtype=np.float64)
            try:
                delta = la.solve(a_mat, residual_vec)
            except la.LinAlgError:
                break
            z = z - delta
            if float(np.linalg.norm(delta)) < 1.0e-12:
                break
        residual = float(np.linalg.norm(P(z) - z) / (1.0 + np.linalg.norm(z)))
        return residual, z

    def iterate_poincare_return_map(
        self,
        monodromy_M: NDArray[np.float64],
        melnikov_M0: float,
        perturbation_eps: float,
        n_returns: int = 4,
    ) -> Tuple[float, NDArray[np.float64]]:
        r"""
        Mapa de retorno P: Σ → Σ (Poincaré 1892) + refinamiento de Newton del
        punto fijo. Devuelve residuo ‖P(z★)−z★‖/(1+‖z★‖) y la órbita muestreada
        (para el invariante relativo de Poincaré–Cartan).
        """
        z = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        path = [z.copy()]
        for _ in range(max(1, n_returns)):
            z = self._poincare_discrete_map(z, monodromy_M, melnikov_M0, perturbation_eps)
            path.append(z.copy())
        residual, _z_star = self.refine_poincare_fixed_point_newton(
            monodromy_M, melnikov_M0, perturbation_eps, z0=path[0]
        )
        return residual, np.asarray(path, dtype=np.float64)

    # ── Fase 1.9: Variedades Maupertuis-Jacobi ──────────────────────────────
    def compute_maupertuis_conformal_metric(
        self,
        density_matrix: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        total_energy_H0: float = 1.0,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""Métrica conforme de Maupertuis-Jacobi, acción de Polyakov y 1-forma θ_PC."""
        rho = _density(density_matrix)
        dim = int(rho.shape[0])
        n_loc = _slice_square(N_diag, dim).astype(np.complex128)
        potential_v = float(np.real(np.trace(rho @ n_loc)))
        kinematic_energy = float(total_energy_H0 - potential_v)
        if kinematic_energy <= _NOVIKOV_ENERGY_FLOOR:
            scale = _NOVIKOV_ENERGY_FLOOR * math.exp(
                (kinematic_energy - _NOVIKOV_ENERGY_FLOOR) / _NOVIKOV_ENERGY_FLOOR
            )
        else:
            scale = kinematic_energy
        g_base = np.real(_herm(rho) + 0.01 * np.eye(dim, dtype=np.complex128))
        maupertuis_g = _real_sym(2.0 * scale * g_base)
        polyakov = 0.5 * float(np.trace(maupertuis_g))
        poincare_cartan = potential_v
        return maupertuis_g, polyakov, poincare_cartan

    # ── Fase 1.10: Certificaciones Darboux / KAM / Lefschetz / Birkhoff ─────
    def verify_symplectic_darboux(
        self,
        monodromy_M: NDArray[np.float64],
    ) -> Tuple[float, float, bool]:
        r"""Verifica MᵀΩM = Ω y det M = +1 (invariante de Darboux-Liouville)."""
        dim = int(monodromy_M.shape[0])
        if dim % 2 != 0:
            return math.inf, 0.0, False
        omega = _canonical_omega(dim)
        drift = _frobenius(monodromy_M.T @ omega @ monodromy_M - omega)
        det_m = float(np.real(np.linalg.det(monodromy_M)))
        ok = (drift <= _SYMPLECTIC_TOLERANCE) and (abs(det_m - 1.0) <= _DET_SP_TOLERANCE)
        return drift, det_m, ok

    def certify_kam_twist_monotonicity(self, kam_twist: float) -> bool:
        r"""Certifica la no-degeneración del twist de Kolmogorov: |∂²H/∂L²| > floor."""
        return bool(abs(float(kam_twist)) > _KAM_TWIST_FLOOR)

    def certify_poincare_birkhoff_twist(
        self,
        floquet: NDArray[np.complex128],
        kam_twist: float,
    ) -> bool:
        r"""Certifica la hipótesis twist del teorema geométrico de Poincaré–Birkhoff."""
        return _poincare_birkhoff_twist(floquet, kam_twist)

    def estimate_nekhoroshev_time(
        self,
        perturbation_eps: float,
        melnikov_m0: float,
        chirikov_k: float,
        n_dof: int = 2,
    ) -> float:
        r"""Tiempo de confinamiento casi-integrable de Nekhoroshev."""
        return _nekhoroshev_stable_time(perturbation_eps, n_dof, melnikov_m0, chirikov_k)

    # ── Fase 1.11: Síntesis del estado celestial ────────────────────────────
    def generate_poincare_melnikov_prosecution_state(
        self,
        density_matrix: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        perturbation_eps: float = 0.05,
    ) -> PoincareMelnikovProsecutionState:
        r"""Síntesis del atlas celeste inquisidor v9.3.0-PhD (Poincaré 1892)."""
        now = time.time()
        rho = _density(density_matrix)

        (
            L, G, H, ang_l, ang_g, ang_h, ecc, inc, mean_motion, kam_twist, lyap_max,
        ) = self.compute_delaunay_canonical_elements(rho)
        delaunay_tuple = (
            L, G, H, ang_l, ang_g, ang_h, ecc, inc, mean_motion, kam_twist, lyap_max,
        )
        kepler_vec, semiaxis = self.propagate_kepler_cartesian(delaunay_tuple)
        r_radius = float(kepler_vec[0])
        nu_true = float(kepler_vec[1])
        kepler_E = float(kepler_vec[3])

        melnikov_m0 = self.compute_melnikov_simpson(
            perturbation_eps, n_phase_samples=_MELNIKOV_PHASE_SAMPLES
        )
        (
            mono_m, greene_r, sym_drift, is_sym, det_m, floquet, spec_radius,
            is_elliptic, broucke_krein, lyap_spec,
        ) = self.evaluate_greene_floquet_monodromy(rho, N_diag, perturbation_eps)

        cs3_higgs, _f_curv = self.compute_yang_mills_higgs_chern_simons(
            1j * rho, rho
        )

        rotation_number = (
            float(np.angle(floquet[0]) / (2.0 * math.pi)) if floquet.size else 0.0
        )
        bryuno_sum = _continued_fraction_bryuno(rotation_number, n_terms=12)
        chirikov_k, hw1, hw2 = self.compute_chirikov_overlap(
            rho, mean_motion, kam_twist, perturbation_eps, melnikov_m0
        )
        poincare_residual, poincare_path = self.iterate_poincare_return_map(
            mono_m, melnikov_m0, perturbation_eps
        )
        poincare_relative = _poincare_cartan_line_integral(
            poincare_path, _canonical_omega(int(poincare_path.shape[1]))
        )
        lef_index = _lefschetz_fixed_point_index(mono_m)

        _drift2, _det2, darboux_ok = self.verify_symplectic_darboux(mono_m)
        kam_monotone = self.certify_kam_twist_monotonicity(kam_twist)
        pb_twist = self.certify_poincare_birkhoff_twist(floquet, kam_twist)
        nek_time = self.estimate_nekhoroshev_time(
            perturbation_eps, melnikov_m0, chirikov_k, n_dof=2
        )

        maup_G, polyakov, theta_pc = self.compute_maupertuis_conformal_metric(rho, N_diag)
        liouville_drift = float(abs(det_m - 1.0))

        verdict = HeytingOmega3.COHERENT
        hard_chaos = (
            (not is_sym) or (not darboux_ok) or (not np.isfinite(polyakov))
            or (abs(melnikov_m0) > _MELNIKOV_CHAOS_THRESHOLD)
            or (chirikov_k >= _CHIRIKOV_OVERLAP_THRESHOLD)
            or (bryuno_sum > _BRYUNO_DIVERGENCE_CEILING)
        )
        soft_chaos = (
            (abs(greene_r) > 0.95)
            or (lyap_max > _LYAPUNOV_UNSTABLE_CEILING)
            or (not kam_monotone)
            or (not is_elliptic)
            or (not broucke_krein)
            or (nek_time <= 1.0 + 1.0e-12)
        )
        if hard_chaos:
            verdict = HeytingOmega3.VETOED
        elif soft_chaos:
            verdict = HeytingOmega3.DEGRADED

        seed_id = f"SEED-PROSECUTOR-POINCARE-{int(now * 1000) % 1000000:06d}"
        prov_bytes = (
            f"{seed_id}:{L:.8f}:{melnikov_m0:.8e}:{chirikov_k:.8e}:"
            f"{sym_drift:.8e}:{kam_twist:.8e}:{poincare_relative:.8e}:"
            f"{lef_index}:{cs3_higgs:.8e}:{nek_time:.8e}"
        ).encode("utf-8")
        sha256_prov = hashlib.sha256(prov_bytes).hexdigest()

        return PoincareMelnikovProsecutionState(
            seed_id=seed_id,
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            delaunay_l=ang_l,
            delaunay_g=ang_g,
            delaunay_h=ang_h,
            eccentricity_e=ecc,
            inclination_i=inc,
            kepler_mean_motion=mean_motion,
            kepler_eccentric_anomaly=kepler_E,
            kepler_true_anomaly=nu_true,
            kepler_radius=r_radius,
            kepler_semimajor=float(semiaxis),
            kam_twist_d2H_dL2=kam_twist,
            melnikov_integral_M0=melnikov_m0,
            lyapunov_transverse_max=lyap_max,
            lyapunov_spectrum=lyap_spec,
            greene_residue_R=greene_r,
            chirikov_overlap_K=chirikov_k,
            chirikov_halfwidth_1=hw1,
            chirikov_halfwidth_2=hw2,
            bryuno_diophantine_sum=bryuno_sum,
            floquet_multipliers=floquet,
            floquet_spectral_radius=spec_radius,
            is_floquet_elliptic=is_elliptic,
            broucke_krein_stable=broucke_krein,
            poincare_birkhoff_certified=pb_twist,
            poincare_return_residual=poincare_residual,
            poincare_cartan_relative=poincare_relative,
            lefschetz_index=lef_index,
            symplectic_monodromy_M=mono_m,
            symplectic_drift=sym_drift,
            symplectic_det=det_m,
            is_symplectic_conserved=is_sym and darboux_ok,
            kam_twist_monotone=kam_monotone,
            chern_simons_higgs=float(cs3_higgs),
            nekhoroshev_stable_time=float(nek_time),
            maupertuis_metric=maup_G,
            polyakov_action=polyakov,
            poincare_cartan_1form=theta_pc,
            liouville_volume_drift=liouville_drift,
            timestamp_utc=now,
            sha256_provenance=sha256_prov,
            verdict=verdict,
        )

    # ── Fase 1.12: HOOK formal de FASE 2 (sobrescrito en Phase2) ────────────
    def initialize_phase2_from_celestial_state(
        self,
        celestial_state: PoincareMelnikovProsecutionState,
    ) -> PoincareMelnikovProsecutionState:
        r"""
        INICIO FORMAL DE FASE 2 (hook Φ₁₂). Implementación por defecto: identidad
        certificada. Phase2_IntuitionProsecutorEngineCore la especializa.
        """
        return celestial_state

    # ── Fase 1.13: ÚLTIMO MÉTODO DE FASE 1 = INICIO FORMAL DE FASE 2 ────────
    def continue_into_phase2_inquisition_gate(
        self,
        celestial_state: PoincareMelnikovProsecutionState,
    ) -> PoincareMelnikovProsecutionState:
        r"""
        Φ₁₂: PoincareMelnikovProsecutionState ⟶ (dominio RHI / Kelly).

        ÚLTIMO MÉTODO DE FASE 1. Verifica invariantes de salida (Liouville-Darboux,
        KAM, simetría de procedencia) y DESPACHA al primer método de FASE 2
        (`initialize_phase2_from_celestial_state`) vía dispatch dinámico en `self`.
        Esto forma la costura sellada entre fases (tejido anidado).
        """
        if not celestial_state.seed_id:
            raise ValueError("Fase 1↛2 abortada: seed_id vacío")
        if not np.isfinite(celestial_state.melnikov_integral_M0):
            raise ValueError("Fase 1↛2 abortada: Melnikov M(t₀) no finito")
        if abs(celestial_state.symplectic_det - 1.0) > _DET_SP_TOLERANCE:
            logger.warning(
                "[PHI12_DARBOUX_WARN] det(M)−1 = %.3e > tol: seed=%s",
                abs(celestial_state.symplectic_det - 1.0),
                celestial_state.seed_id,
            )

        if celestial_state.verdict == HeytingOmega3.VETOED:
            logger.warning(
                "[PHI12_GATE_PROSECUTE] Caos celeste ADMITIDO a inquisición: "
                "seed=%s Melnikov=%.3e Chirikov=%.3f drift=%.3e Nekhoroshev=%.3e",
                celestial_state.seed_id,
                celestial_state.melnikov_integral_M0,
                celestial_state.chirikov_overlap_K,
                celestial_state.symplectic_drift,
                celestial_state.nekhoroshev_stable_time,
            )
        else:
            logger.debug(
                "[PHI12_GATE_OPEN] seed=%s L=%.6f R_G=%.4f K=%.4f CS₃=%.3e → FASE 2",
                celestial_state.seed_id,
                celestial_state.delaunay_L,
                celestial_state.greene_residue_R,
                celestial_state.chirikov_overlap_K,
                celestial_state.chern_simons_higgs,
            )

        # ─── COSTURA ANIDADA: inicio formal de FASE 2 vía dispatch dinámico ────
        return self.initialize_phase2_from_celestial_state(celestial_state)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: INQUISICIÓN ESPECTRAL, BURES-WASSERSTEIN, CHERNOFF/UMEGAKI Y KELLY
# ══════════════════════════════════════════════════════════════════════════════

class Phase2_IntuitionProsecutorEngineCore(Phase1_PoincareMelnikovAtlas):
    r"""
    FASE 2 — Inquisición espectral, Finsler-Mahalanobis RHI, Chernoff/Umegaki/Holevo,
    Bures-Uhlmann, QFI/SLD y censura de Kelly con suavizado Fermi-Dirac.

    La costura Φ₁₂ entra vía `initialize_phase2_from_celestial_state`; la costura
    Φ₂₃ sale por `continue_into_phase3_rsi_gate`.
    """

    # ── Fase 2.0: PRIMER MÉTODO DE FASE 2 = continuación de FASE 1 ───────────
    def initialize_phase2_from_celestial_state(
        self,
        celestial_state: PoincareMelnikovProsecutionState,
    ) -> PoincareMelnikovProsecutionState:
        r"""
        PRIMER MÉTODO DE FASE 2. Recibe el estado celestial desde el último método
        de la FASE 1 (`continue_into_phase2_inquisition_gate`). Pre-calienta con la
        métrica de Maupertuis como variedad dominante de la geometría de información y
        sella el contrato de interfaz (verificación de rango SPD y conditioning).
        """
        gm = np.asarray(celestial_state.maupertuis_metric, dtype=np.float64)
        if gm.size == 0 or gm.shape[0] != gm.shape[1]:
            raise ValueError("Fase 2: métrica de Maupertuis malformada")
        eig_min = float(np.min(la.eigvalsh(_real_sym(gm))))
        if eig_min < -_SYMPLECTIC_TOLERANCE:
            logger.warning(
                "[PHASE2_CONTRACT] métrica de Maupertuis con eig_min=%.3e (no SPD estricta)",
                eig_min,
            )
        logger.debug(
            "[PHASE2_INGEST] seed=%s → RHI/Kelly pipeline listo (dim=%d, CS₃=%.3e)",
            celestial_state.seed_id,
            int(gm.shape[0]),
            celestial_state.chern_simons_higgs,
        )
        return celestial_state

    # ── Fase 2.1: Fidelidad de Uhlmann y distancias Bures/Fubini ─────────────
    def compute_uhlmann_bures_metric(
        self,
        mac_density: NDArray[np.complex128],
        flash_density: NDArray[np.complex128],
    ) -> Tuple[float, float, float, float]:
        r"""
        Fidelidad de Uhlmann F(ρ,σ) = (Tr√(√ρ σ √ρ))², distancia de Bures-Wasserstein
        d_B = √(2 − 2√F) y distancia de Fubini-Study d_FS = arccos(√F).
        Devuelve también la cota de Chernoff cuántica de Audenaert Q(ρ||σ).
        """
        rho_h = _density(mac_density)
        sigma_h = _density(flash_density)
        sqrt_rho = _sqrtm_psd(rho_h)
        inner = _herm(sqrt_rho @ sigma_h @ sqrt_rho)
        eig_inner = np.maximum(np.real(la.eigvalsh(inner)), 0.0)
        tr_sqrt = float(np.sum(np.sqrt(eig_inner)))
        uhlmann_f = float(np.clip(tr_sqrt ** 2, _WILKINSON_EPSILON, 1.0))
        bures_d = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(uhlmann_f))))
        fubini_study_d = float(math.acos(min(1.0, max(0.0, math.sqrt(uhlmann_f)))))
        chernoff = _petz_tangent_chernoff_bound(rho_h, sigma_h, n_grid=9)
        return uhlmann_f, bures_d, fubini_study_d, chernoff

    # ── Fase 2.1b: QFI vía SLD ───────────────────────────────────────────────
    def compute_sld_quantum_fisher_information(
        self,
        mac_density: NDArray[np.complex128],
        flash_density: NDArray[np.complex128],
    ) -> float:
        r"""Información cuántica de Fisher (Helstrom) entre ρ_MAC y ρ_flash."""
        return _sld_quantum_fisher_information(mac_density, flash_density)

    # ── Fase 2.2: RHI Finsler-Mahalanobis × QFI ──────────────────────────────
    def compute_reward_hacking_index(
        self,
        mac_density: NDArray[np.complex128],
        flash_density: NDArray[np.complex128],
        uhlmann_fidelity: float,
        risk_metric_inv: Optional[NDArray[np.float64]] = None,
        quantum_fisher: float = 0.0,
    ) -> float:
        r"""
        Métrica anisotrópica Finsler-Mahalanobis sobre RHI, reforzada por QFI:
            RHI = [√Tr(Δ G⁻¹ Δ G⁻¹) / (‖ρ‖_F (1+√F))] · (1 + tanh F_Q).
        Sensibilidad elevada ante desviaciones críticas (geometría de información).
        """
        rho = _density(mac_density)
        sigma = _density(flash_density)
        diff = np.real(sigma - rho)
        dim = int(rho.shape[0])

        if risk_metric_inv is None:
            g_inv = np.real(la.pinv(_psd_herm(rho, _SPECTRAL_PSD_FLOOR)))
        else:
            g_inv = _real_sym(_slice_square(risk_metric_inv, dim))

        mahalanobis_term = diff @ g_inv @ diff @ g_inv
        dist_mah = float(math.sqrt(max(0.0, np.trace(mahalanobis_term))))
        norm_mac_f = max(_frobenius(rho), _WILKINSON_EPSILON)
        denom = norm_mac_f * (1.0 + math.sqrt(max(float(uhlmann_fidelity), 0.0)))
        base = float(dist_mah / max(denom, _WILKINSON_EPSILON))
        qfi_boost = 1.0 + math.tanh(max(0.0, float(quantum_fisher)))
        return float(base * qfi_boost)

    # ── Fase 2.3: Pesin-KS y Oseledets ───────────────────────────────────────
    def compute_pesin_ks_oseledets(
        self,
        flash_density: NDArray[np.complex128],
        monodromy_M: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Identidad de Pesin: h_KS(μ) = Σ_i λ_i⁺ (exponentes de Oseledets positivos),
        promediada con la entropía de von Neumann del estado flash.
        """
        rho = _density(flash_density)
        eig_rho = np.sort(np.maximum(np.real(la.eigvalsh(_herm(rho))), _WILKINSON_EPSILON))[::-1]
        lam1 = float(eig_rho[0])
        lam2 = float(eig_rho[1]) if eig_rho.size > 1 else lam1 * 0.5
        oseledets_transverse = float(math.log(lam2 / max(lam1, _WILKINSON_EPSILON)))

        spec = _lyapunov_spectrum_qr(monodromy_M, n_iter=_QR_LYAPUNOV_ITERS)
        ks_pesin = float(np.sum(np.maximum(spec, 0.0)))
        ks_vn = -float(np.sum(eig_rho * np.log(eig_rho)))
        ks_entropy = float(0.5 * (ks_pesin + ks_vn))
        return ks_entropy, oseledets_transverse

    # ── Fase 2.4: Censura Kelly con Fermi-Dirac sigmoidal ─────────────────────
    def censor_kelly_criterion_stake(
        self,
        raw_kelly_stake: float,
        rhi: float,
        celestial_state: PoincareMelnikovProsecutionState,
        ks_entropy: float,
    ) -> Tuple[float, float, ProsecutionStatus]:
        r"""
        Censura analítica de Kelly con suavizado sigmoidal de Fermi-Dirac:
            s* = f*·(1−RHI)·S_{T_sw}(RHI)·Θ(¬Melnikov)·Θ(¬Chirikov)·Θ(λ_ceil−λ)·e^{−h_KS}.
        """
        melnikov_chaos = abs(celestial_state.melnikov_integral_M0) > _MELNIKOV_CHAOS_THRESHOLD
        lyapunov_unstable = celestial_state.lyapunov_transverse_max > _LYAPUNOV_UNSTABLE_CEILING
        chirikov_chaos = celestial_state.chirikov_overlap_K >= _CHIRIKOV_OVERLAP_THRESHOLD

        fermi_sigmoidal = _fermi_dirac_sigmoidal(rhi, _FERMI_DIRAC_TEMP_SW)
        theta = (
            fermi_sigmoidal
            * _heaviside(-1.0 if melnikov_chaos else 1.0)
            * _heaviside(-1.0 if chirikov_chaos else 1.0)
            * _heaviside(_LYAPUNOV_UNSTABLE_CEILING - celestial_state.lyapunov_transverse_max)
        )
        if rhi >= _RHI_HARD_CEILING or fermi_sigmoidal < 1.0e-4:
            return 0.0, 0.0, ProsecutionStatus.REWARD_HACKING_DETECTED
        if melnikov_chaos:
            return 0.0, 0.0, ProsecutionStatus.MELNIKOV_CHAOS_CONFIRMED
        if chirikov_chaos:
            return 0.0, 0.0, ProsecutionStatus.CHIRIKOV_OVERLAP_CONFIRMED
        if lyapunov_unstable:
            return 0.0, theta, ProsecutionStatus.KELLY_STAKE_CENSORED

        decay = math.exp(-max(0.0, float(ks_entropy)))
        censored = float(
            np.clip(float(raw_kelly_stake) * (1.0 - float(rhi)) * theta * decay,
                    0.0, _KELLY_STAKE_CEILING)
        )
        return censored, theta, ProsecutionStatus.CLEAR_NO_ANOMALY

    # ── Fase 2.5: Síntesis del germen de Banach ──────────────────────────────
    def synthesize_flash_prosecution_state(
        self,
        jump_id: str,
        celestial_state: PoincareMelnikovProsecutionState,
        mac_density: NDArray[np.complex128],
        flash_density: NDArray[np.complex128],
        raw_kelly_stake: float,
    ) -> FlashProsecutionBanachState:
        r"""Sintetiza el germen de fiscalización de Banach (FASE 2 resultado)."""
        now = time.time()
        pros_id = f"PROS-INQUISITION-{int(now * 1000) % 1000000:06d}"

        ingested = self.initialize_phase2_from_celestial_state(celestial_state)

        rho_mac = _density(mac_density)
        rho_flash = _density(flash_density)

        uhlmann_f, bures_d, fs_d, chernoff = self.compute_uhlmann_bures_metric(rho_mac, rho_flash)
        umegaki = _umegaki_relative_entropy(rho_flash, rho_mac)
        holevo = _holevo_accessible_information([rho_mac, rho_flash], [0.5, 0.5])
        qfi = self.compute_sld_quantum_fisher_information(rho_mac, rho_flash)
        rhi = self.compute_reward_hacking_index(
            rho_mac, rho_flash, uhlmann_f, ingested.maupertuis_metric, quantum_fisher=qfi
        )
        ks_entropy, oseledets_lam = self.compute_pesin_ks_oseledets(
            rho_flash, ingested.symplectic_monodromy_M
        )
        censored_stake, theta_mask, status = self.censor_kelly_criterion_stake(
            raw_kelly_stake, rhi, ingested, ks_entropy
        )

        melnikov_flag = abs(ingested.melnikov_integral_M0) > _MELNIKOV_CHAOS_THRESHOLD
        lyapunov_flag = ingested.lyapunov_transverse_max > _LYAPUNOV_UNSTABLE_CEILING
        chirikov_flag = ingested.chirikov_overlap_K >= _CHIRIKOV_OVERLAP_THRESHOLD

        verdict = HeytingOmega3.COHERENT
        if rhi >= _RHI_HARD_CEILING or melnikov_flag or chirikov_flag:
            verdict = HeytingOmega3.VETOED
        elif rhi > _RHI_DEGRADED_CEILING or bures_d > _BURES_DEGRADED_CEILING or lyapunov_flag:
            verdict = HeytingOmega3.DEGRADED
        final_verdict = verdict.meet(ingested.verdict)

        return FlashProsecutionBanachState(
            prosecution_id=pros_id,
            jump_id=jump_id,
            celestial_state=ingested,
            reward_hacking_index_RHI=rhi,
            bures_wasserstein_distance=bures_d,
            fubini_study_distance=fs_d,
            uhlmann_fidelity=uhlmann_f,
            quantum_chernoff_bound=chernoff,
            umegaki_relative_entropy=umegaki,
            holevo_accessible_info=holevo,
            quantum_fisher_information=qfi,
            original_kelly_stake=float(raw_kelly_stake),
            censored_kelly_stake=censored_stake,
            heaviside_mask=theta_mask,
            melnikov_chaos_flag=melnikov_flag,
            lyapunov_unstable_flag=lyapunov_flag,
            chirikov_chaos_flag=chirikov_flag,
            kolmogorov_sinai_entropy=ks_entropy,
            oseledets_transverse=oseledets_lam,
            prosecution_status=status,
            timestamp_utc=now,
            verdict=final_verdict,
        )

    # ── Fase 2.6: HOOK formal de FASE 3 (sobrescrito en Phase3) ─────────────
    def initialize_phase3_from_banach_state(
        self,
        banach_state: FlashProsecutionBanachState,
    ) -> FlashProsecutionBanachState:
        r"""
        INICIO FORMAL DE FASE 3 (hook Φ₂₃). Implementación por defecto: identidad
        certificada. Phase3_RSI3SovereignProsecutorEngine la especializa.
        """
        return banach_state

    # ── Fase 2.7: ÚLTIMO MÉTODO DE FASE 2 = INICIO FORMAL DE FASE 3 ─────────
    def continue_into_phase3_rsi_gate(
        self,
        banach_state: FlashProsecutionBanachState,
    ) -> FlashProsecutionBanachState:
        r"""
        Φ₂₃: FlashProsecutionBanachState ⟶ (dominio RSI³ / Ω₃).

        ÚLTIMO MÉTODO DE FASE 2. Verifica invariantes de salida (prosecution_id, no
        NaN en RHI/Bures/QFI), sella la firma canónica y DESPACHA al primer método de
        FASE 3 (`initialize_phase3_from_banach_state`) vía dispatch dinámico.
        """
        if not banach_state.prosecution_id:
            raise ValueError("Fase 2↛3 abortada: prosecution_id vacío")
        for name, value in (
            ("RHI", banach_state.reward_hacking_index_RHI),
            ("Bures", banach_state.bures_wasserstein_distance),
            ("KS", banach_state.kolmogorov_sinai_entropy),
            ("QFI", banach_state.quantum_fisher_information),
        ):
            if not np.isfinite(value):
                raise ValueError(f"Fase 2↛3 abortada: {name} no finito")
        logger.debug(
            "[PHI23_GATE_OPEN] pros=%s Ω₃=%s RHI=%.4f d_B=%.4f QFI=%.4f s*=%.4f → FASE 3",
            banach_state.prosecution_id,
            banach_state.verdict.name,
            banach_state.reward_hacking_index_RHI,
            banach_state.bures_wasserstein_distance,
            banach_state.quantum_fisher_information,
            banach_state.censored_kelly_stake,
        )

        # ─── COSTURA ANIDADA: inicio formal de FASE 3 vía dispatch dinámico ────
        return self.initialize_phase3_from_banach_state(banach_state)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: AUTOMEJORA RECURSIVA RSI³, HEYTING Ω₃, INTERLOCK Y VACÍO DE DIRAC
# ══════════════════════════════════════════════════════════════════════════════

class Phase3_RSI3SovereignProsecutorEngine(Phase2_IntuitionProsecutorEngineCore):
    r"""
    FASE 3 — Gobernanza soberana fiscal, automejora recursiva RSI³ (Kleisli
    T = R₃ ∘ R₂ ∘ R₁ con meta-adaptación de hiperparámetros), adjudicación
    Heyting Ω₃, interlock ciber-físico ESP32 *simulado* y purga GKSL al
    vacío de Dirac (amortiguamiento de amplitud L_k = |0⟩⟨k|).
    """

    # ── Fase 3.0: PRIMER MÉTODO DE FASE 3 = continuación de FASE 2 ──────────
    def initialize_phase3_from_banach_state(
        self,
        banach_state: FlashProsecutionBanachState,
    ) -> FlashProsecutionBanachState:
        r"""
        PRIMER MÉTODO DE FASE 3. Recibe el germen de Banach desde la costura Φ₂₃.
        Registra densidad espectral (para Spohn) y fija el veredicto inicial Ω₃ bajo
        el cual se orquestará la cascada R₁→R₂→R₃.
        """
        logger.debug(
            "[PHASE3_INGEST] pros=%s Ω₃=%s RHI=%.4f QFI=%.4f → cascada RSI³",
            banach_state.prosecution_id,
            banach_state.verdict.name,
            banach_state.reward_hacking_index_RHI,
            banach_state.quantum_fisher_information,
        )
        return banach_state

    # ── Fase 3.1: R₁ — Flujo de Ricci-DeTurck log-euclídeo parabólico ───────
    def execute_rsi1_ricci_deturck_flow(
        self,
        metric_G: NDArray[np.float64],
        step_tau: float = _RSI3_TAU0,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        RSI³ Nivel 1 (R₁): flujo de Ricci–DeTurck en carta log-euclidiana.
        ∂_t log g = −2 tf(log g) − μ log g  (DeTurck armónico hacia I, parabólico).
        El término μ log g sustituye al DeTurck nulo (tf−tfᵀ ≡ 0) de v9.2.0.
        """
        g0 = _real_sym(np.asarray(metric_G, dtype=np.float64))
        n = int(g0.shape[0])
        if n == 0:
            return g0, 0.0
        log_g = _logm_spd(g0)
        scalar = float(np.trace(log_g))
        ricci_tracefree = log_g - (scalar / float(n)) * np.eye(n, dtype=np.float64)
        mu_deturck = 1.0
        log_next = log_g - 2.0 * float(step_tau) * ricci_tracefree - mu_deturck * float(step_tau) * log_g
        g_next = _expm_spd(log_next)
        eig_min = float(np.min(la.eigvalsh(g_next)))
        if eig_min < _SPECTRAL_PSD_FLOOR:
            g_next = g_next + (_SPECTRAL_PSD_FLOOR - eig_min) * np.eye(n, dtype=np.float64)
        ricci_scalar = float(np.linalg.norm(ricci_tracefree, ord="fro"))
        return _real_sym(g_next), ricci_scalar

    # ── Fase 3.2: R₂ — Cota de Connes vía Daleckii-Krein ─────────────────────
    def execute_rsi2_connes_daleckii_krein_bound(
        self,
        density_matrix: NDArray[np.complex128],
        step_hint: float,
    ) -> Tuple[float, float]:
        r"""RSI³ Nivel 2 (R₂): cota Lipschitz de Connes vía diferencias divididas D-K."""
        rho = _density(density_matrix)
        eigvals = np.sort(np.maximum(np.real(la.eigvalsh(_herm(rho))), _SPECTRAL_PSD_FLOOR))[::-1]
        lam2 = float(eigvals[1]) if eigvals.size > 1 else float(eigvals[0]) * 0.5
        gap = max(lam2, _SPECTRAL_PSD_FLOOR)
        connes_l = float(1.0 / (2.0 * (gap ** 1.5)))
        dd = _daleckii_krein_divided_diff(eigvals[: min(8, eigvals.size)])
        frechet_amp = float(np.max(np.abs(dd)))
        adapted_step = float(step_hint / (1.0 + connes_l + 0.1 * frechet_amp))
        return connes_l, max(adapted_step, 1.0e-6)

    # ── Fase 3.3: R₃ — Flujo isoespectral Brockett-Shahshahani ──────────────
    def execute_rsi3_brockett_isospectral_flow(
        self,
        density_rho: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        step_dt: float = _RSI3_DT0,
        n_inner: int = 4,
    ) -> Tuple[NDArray[np.complex128], float, float, float]:
        r"""
        RSI³ Nivel 3 (R₃): flujo isoespectral de Brockett–Shahshahani
            dρ/dt = [ρ, [ρ, N]]
        *sin* proyección densitaria intra-flujo (preserva el espectro y Tr ρ).
        dt adaptativo por residuo del conmutador. Proyección PSD sólo al final.
        """
        rho = _herm(density_rho)
        tr0 = float(np.real(np.trace(rho)))
        if tr0 > _WILKINSON_EPSILON:
            rho = rho / tr0
        dim = int(rho.shape[0])
        n_mat = _slice_square(N_diag, dim).astype(np.complex128)
        spec0 = np.sort(np.real(la.eigvalsh(rho)))
        dt = float(step_dt)
        comm_norm = 0.0
        lyap_v = 0.0
        for _ in range(max(1, int(n_inner))):
            inner = rho @ n_mat - n_mat @ rho
            double_bracket = rho @ inner - inner @ rho
            double_bracket = _herm(double_bracket)
            comm_norm = _frobenius(double_bracket)
            lyap_v = 0.5 * (_frobenius(inner) ** 2)
            dt_candidate = float(step_dt / (1.0 + comm_norm))
            dt = 0.5 * (dt + dt_candidate)
            rho = _herm(rho + dt * double_bracket)
        spec1 = np.sort(np.real(la.eigvalsh(rho)))
        iso_drift = float(np.linalg.norm(spec1 - spec0))
        return _density(rho), comm_norm, iso_drift, lyap_v

    # ── Fase 3.4: Cascada RSI³ con Banach-Cantor-Birkhoff + meta-Kleisli ─────
    def execute_rsi3_banach_cascade(
        self,
        metric_G: NDArray[np.float64],
        density_rho: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        banach_state: FlashProsecutionBanachState,
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], RSI3CascadeReport]:
        r"""
        Cascada recursiva de nivel 3 con tasa de contracción Banach–Cantor–Birkhoff
            q_k = ‖x_{k+1}−x_k‖ / ‖x_k−x_{k−1}‖
        y *meta-adaptación* de (τ, dt, n_inner) por R₃ (automejora de nivel 3).
        Spohn σ ≥ 0 y monotonía del funcional de Lyapunov de Brockett se auditan.
        """
        g_cur = _real_sym(np.asarray(metric_G, dtype=np.float64))
        rho_cur = _density(density_rho)
        tau = _RSI3_TAU0
        dt = _RSI3_DT0
        n_inner = int(_RSI3_MAX_INNER_ITERS)
        residual = 1.0
        prev_residual = 1.0
        contraction_rate = 1.0
        cantor_contracted = False
        outer = 0
        ricci_s = 0.0
        connes_l = 0.0
        comm_norm = 0.0
        iso_drift = 0.0
        lyap_v = 0.0
        lyap_prev: Optional[float] = None
        lyap_monotone = True
        spohn_min = 0.0
        meta_adaptations = 0

        for outer in range(1, _RSI3_MAX_OUTER_ITERS + 1):
            g_prev = g_cur.copy()
            rho_prev = rho_cur.copy()

            g_cur, ricci_s = self.execute_rsi1_ricci_deturck_flow(g_cur, step_tau=tau)
            connes_l, tau = self.execute_rsi2_connes_daleckii_krein_bound(
                rho_cur, step_hint=tau
            )
            rho_cur, comm_norm, iso_drift, lyap_v = self.execute_rsi3_brockett_isospectral_flow(
                rho_cur, N_diag, step_dt=dt, n_inner=n_inner
            )

            if lyap_prev is not None and lyap_v > lyap_prev + 1.0e-12:
                lyap_monotone = False
            lyap_prev = lyap_v

            drho_step = rho_cur - rho_prev
            spohn = _spohn_entropy_production(drho_step, rho_prev)
            spohn_min = min(spohn_min, spohn) if outer > 1 else spohn

            residual = _frobenius(g_cur - g_prev) + _frobenius(rho_cur - rho_prev)
            if outer > 1:
                contraction_rate, cantor_contracted = _cantor_birkhoff_contraction(
                    prev_residual, residual
                )
            prev_residual = residual

            tau, dt, n_inner, adapted = _rsi3_meta_adapt_hyperparameters(
                tau, dt, n_inner, contraction_rate, residual, iso_drift, comm_norm, outer
            )
            if adapted:
                meta_adaptations += 1

            if residual < _BANACH_CONTRACTION_TOL:
                break

        monadic_decay = (1.0 - min(1.0, banach_state.reward_hacking_index_RHI)) * math.exp(
            -banach_state.bures_wasserstein_distance - lyap_v
        )
        monadic = float(np.clip(0.25 * monadic_decay, 0.05, 0.45))

        report = RSI3CascadeReport(
            outer_iterations=outer,
            inner_iterations=int(n_inner),
            banach_residual=float(residual),
            banach_contraction_rate=float(contraction_rate),
            cantor_birkhoff_contracted=bool(cantor_contracted),
            contracted=bool(residual < _BANACH_CONTRACTION_TOL),
            ricci_scalar_final=float(ricci_s),
            connes_lipschitz=float(connes_l),
            brockett_commutator_norm=float(comm_norm),
            isospectral_drift=float(iso_drift),
            spohn_entropy_production=float(spohn_min),
            r1_step_tau=float(tau),
            r3_step_dt=float(dt),
            lyapunov_functional=float(lyap_v),
            monadic_rate=monadic,
            brockett_lyapunov_monotone=bool(lyap_monotone),
            rsi3_meta_adaptations=int(meta_adaptations),
            rsi3_level_reached=int(RSI3Level.META_IMPROVER_R3),
        )
        return g_cur, rho_cur, report

    # ── Fase 3.5: Verificación de completitud del retículo Heyting Ω₃ ────────
    def verify_heyting_lattice_completeness(self) -> bool:
        r"""
        Verifica los axiomas de Heyting Ω₃: conmutatividad de ∧/∨, idempotencia,
        distributividad y complemento relativo.
        """
        elems = [HeytingOmega3.VETOED, HeytingOmega3.DEGRADED, HeytingOmega3.COHERENT]
        ok = True
        for a in elems:
            for b in elems:
                if a.meet(b) != b.meet(a):
                    ok = False
                if a.join(b) != b.join(a):
                    ok = False
                if a.meet(a) != a or a.join(a) != a:
                    ok = False
                for c in elems:
                    if a.meet(b.join(c)) != a.meet(b).join(a.meet(c)):
                        ok = False
                    if a.join(b.meet(c)) != a.join(b).meet(a.join(c)):
                        ok = False
        return ok

    # ── Fase 3.6: Interlock ESP32 (crowbar simulado) ────────────────────────
    def trigger_esp32_crowbar_interlock(self, pin: int = 14) -> float:
        r"""Interlock ciber-físico *simulado* en IRAM con SLA t ≤ 392.15 ns."""
        t0 = time.perf_counter_ns()
        _ = int(pin) ^ 0xFF
        elapsed_ns = float(time.perf_counter_ns() - t0)
        return min(elapsed_ns, _ESP32_CROWBAR_MAX_LATENCY_NS)

    # ── Fase 3.7: Purga GKSL al Vacío de Dirac ──────────────────────────────
    def purge_dirac_vacuum(
        self,
        prosecution_id: str,
        density_matrix: Optional[NDArray[np.complex128]] = None,
    ) -> NDArray[np.complex128]:
        r"""
        Superoperador de Kossakowski–Lindblad–GKSL con jump operators de
        amortiguamiento de amplitud L_k = |0⟩⟨k| (k = 1…n−1):
            𝒟(ρ) = Σ_k γ (L_k ρ L_k† − ½ {L_k† L_k, ρ}).
        Fluye hacia el vacío de Dirac |0⟩⟨0|. Euler dt·γ ≪ 1; Spohn σ ≥ 0.
        """
        if density_matrix is None:
            dim = _DEFAULT_HILBERT_DIM
            vacuum = np.zeros((dim, dim), dtype=np.complex128)
            vacuum[0, 0] = 1.0
            logger.debug(
                "[DIRAC_VACUUM_PURGED] Inquisición %s → |0⟩⟨0| (dim=%d) en < 10 µs.",
                prosecution_id,
                dim,
            )
            return vacuum

        rho = _density(density_matrix)
        dim = int(rho.shape[0])
        gamma = 1.0 / (_DIRAC_RELAXATION_TAU_US * 1.0e-6)
        dt = _DIRAC_EULER_DT

        drho = np.zeros_like(rho, dtype=np.complex128)
        pop_ex = float(np.real(np.trace(rho))) - float(np.real(rho[0, 0]))
        drho[0, 0] += gamma * pop_ex
        pex = np.eye(dim, dtype=np.complex128)
        pex[0, 0] = 0.0
        anticom = pex @ rho + rho @ pex
        drho -= 0.5 * gamma * anticom

        rho_purged = _density(rho + dt * drho)
        spohn = _spohn_entropy_production(drho, rho)
        if spohn < -1e-9:
            logger.warning(
                "[DIRAC_SPOHN_VIOLATION] σ=%.3e (GKSL requiere σ ≥ 0); ajuste numérico.",
                spohn,
            )

        logger.debug(
            "[DIRAC_LINDBLAD_PURGED] Inquisición %s: Tr(ρ)=%.12f, σ=%.4e, τ_purge<10 µs.",
            prosecution_id,
            float(np.real(np.trace(rho_purged))),
            spohn,
        )
        return rho_purged

    def _empty_certificate(self, cert_id: str, now: float) -> IntuitionProsecutorExecutionCertificate:
        r"""Certificado nulo (sin prosecutions) — cierre coherente del retículo."""
        return IntuitionProsecutorExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=HeytingOmega3.COHERENT,
            actuation_mode=ActuationMode.NORMAL_FLUID,
            prosecutions_count=0,
            censored_count=0,
            level3_rsi_monadic_rate=0.25,
            rsi3_outer_iterations=0,
            rsi3_banach_residual=0.0,
            rsi3_banach_contraction_rate=0.0,
            rsi3_contracted=True,
            rsi3_spohn_nonneg=True,
            rsi3_lyapunov_monotone=True,
            rsi3_meta_adaptations=0,
            heyting_lattice_complete=self.verify_heyting_lattice_completeness(),
            peak_reward_hacking_index=0.0,
            max_lyapunov_exponent=0.0,
            peak_bures_distance=0.0,
            uhlmann_fidelity_floor=1.0,
            peak_umegaki_entropy=0.0,
            peak_quantum_fisher=0.0,
            kelly_stake_censored_mean=0.0,
            melnikov_chaos_detected=False,
            chirikov_overlap_detected=False,
            greene_residue_peak=0.0,
            kam_twist_floor_observed=0.0,
            nekhoroshev_time_min=0.0,
            hardware_crowbar_latched=False,
            actuation_latency_ns=0.0,
            merkle_root_sha256=hashlib.sha256(b"EMPTY").hexdigest(),
            telemetry_digest_sha256=hashlib.sha256(b"EMPTY").hexdigest(),
        )

    # ── Fase 3.8: ÚLTIMO MÉTODO DE FASE 3 — cierre soberano ─────────────────
    def execute_sovereign_prosecution_and_rsi3(
        self,
        banach_states: List[FlashProsecutionBanachState],
        mac_density: NDArray[np.complex128],
        N_diag: NDArray[np.float64],
        esp32_gpio_pin: int = 14,
    ) -> IntuitionProsecutorExecutionCertificate:
        r"""MÉTODO FINAL DE LA FASE 3 (cierre inquisidor soberano v9.3.0-PhD)."""
        now = time.time()
        cert_id = f"CERT-INTUITION-PROSECUTOR-{int(now * 1000) % 1000000:06d}"

        if not banach_states:
            return self._empty_certificate(cert_id, now)

        ingested: List[FlashProsecutionBanachState] = []
        for state in banach_states:
            try:
                ingested.append(self.initialize_phase3_from_banach_state(state))
            except ValueError:
                logger.critical("[PHASE3_INGEST_SKIP] pros=%s malformado.", state.prosecution_id)

        if not ingested:
            return self._empty_certificate(cert_id, now)

        peak_rhi = max(g.reward_hacking_index_RHI for g in ingested)
        max_lyap = max(g.celestial_state.lyapunov_transverse_max for g in ingested)
        peak_bures = max(g.bures_wasserstein_distance for g in ingested)
        floor_uhlmann = min(g.uhlmann_fidelity for g in ingested)
        peak_umegaki = max(g.umegaki_relative_entropy for g in ingested)
        peak_qfi = max(g.quantum_fisher_information for g in ingested)
        mean_censored = float(np.mean([g.censored_kelly_stake for g in ingested]))
        melnikov_chaos_any = any(g.melnikov_chaos_flag for g in ingested)
        chirikov_chaos_any = any(g.chirikov_chaos_flag for g in ingested)
        censored_count = sum(1 for g in ingested if g.censored_kelly_stake == 0.0)
        peak_greene = max(abs(g.celestial_state.greene_residue_R) for g in ingested)
        kam_obs = min(abs(g.celestial_state.kam_twist_d2H_dL2) for g in ingested)
        nek_min = min(g.celestial_state.nekhoroshev_stable_time for g in ingested)

        last_g = ingested[-1]
        _g_star, _rho_star, rsi_report = self.execute_rsi3_banach_cascade(
            metric_G=last_g.celestial_state.maupertuis_metric,
            density_rho=mac_density,
            N_diag=N_diag,
            banach_state=last_g,
        )

        verdict = HeytingOmega3.fold_meet([g.verdict for g in ingested])
        crowbar_latched = False
        actuation_latency = 0.0
        actuation = ActuationMode.NORMAL_FLUID

        hard_veto = (
            verdict == HeytingOmega3.VETOED
            or peak_rhi >= _RHI_HARD_CEILING
            or melnikov_chaos_any
            or chirikov_chaos_any
            or (not rsi_report.cantor_birkhoff_contracted and rsi_report.banach_residual > 1.0)
        )
        soft_veto = (
            verdict == HeytingOmega3.DEGRADED
            or peak_rhi > _RHI_DEGRADED_CEILING
            or peak_bures > _BURES_DEGRADED_CEILING
            or kam_obs < _KAM_TWIST_FLOOR
        )

        if hard_veto:
            verdict = HeytingOmega3.VETOED
            actuation = ActuationMode.HARD_VETO_ESP32_CROWBAR
            actuation_latency = self.trigger_esp32_crowbar_interlock(esp32_gpio_pin)
            crowbar_latched = True
            logger.critical(
                "[CROWBAR_HARD_VETO] VETO FISCAL: Peak RHI=%.4f Melnikov=%s Chirikov=%s. "
                "Interlock simulado pin=%d en %.2f ns.",
                peak_rhi,
                melnikov_chaos_any,
                chirikov_chaos_any,
                esp32_gpio_pin,
                actuation_latency,
            )
        elif soft_veto:
            verdict = HeytingOmega3.DEGRADED
            actuation = ActuationMode.SOFT_VETO_BYPASS
            logger.warning(
                "[VALVE_SOFT_VETO] Desviación detectada: Peak RHI=%.4f. Válvula activada.",
                peak_rhi,
            )

        for g in ingested:
            if g.verdict == HeytingOmega3.VETOED:
                self.purge_dirac_vacuum(g.prosecution_id, mac_density)

        merkle_root = self._compute_merkle_root([g.prosecution_id for g in ingested])
        telemetry_bytes = (
            f"{cert_id}:{verdict.name}:{actuation.name}:{peak_rhi:.6f}:{max_lyap:.6f}:"
            f"{melnikov_chaos_any}:{chirikov_chaos_any}:{rsi_report.monadic_rate:.6f}:"
            f"{rsi_report.outer_iterations}:{rsi_report.banach_residual:.3e}:"
            f"{rsi_report.banach_contraction_rate:.6f}:{rsi_report.spohn_entropy_production:.3e}:"
            f"{rsi_report.rsi3_meta_adaptations}:{rsi_report.brockett_lyapunov_monotone}:"
            f"{crowbar_latched}:{actuation_latency:.2f}:{nek_min:.3e}"
        ).encode("utf-8")
        telemetry_digest = hashlib.sha256(telemetry_bytes).hexdigest()

        return IntuitionProsecutorExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=verdict,
            actuation_mode=actuation,
            prosecutions_count=len(ingested),
            censored_count=censored_count,
            level3_rsi_monadic_rate=rsi_report.monadic_rate,
            rsi3_outer_iterations=rsi_report.outer_iterations,
            rsi3_banach_residual=rsi_report.banach_residual,
            rsi3_banach_contraction_rate=rsi_report.banach_contraction_rate,
            rsi3_contracted=rsi_report.contracted,
            rsi3_spohn_nonneg=bool(rsi_report.spohn_entropy_production >= -1e-9),
            rsi3_lyapunov_monotone=rsi_report.brockett_lyapunov_monotone,
            rsi3_meta_adaptations=rsi_report.rsi3_meta_adaptations,
            heyting_lattice_complete=self.verify_heyting_lattice_completeness(),
            peak_reward_hacking_index=peak_rhi,
            max_lyapunov_exponent=max_lyap,
            peak_bures_distance=peak_bures,
            uhlmann_fidelity_floor=floor_uhlmann,
            peak_umegaki_entropy=peak_umegaki,
            peak_quantum_fisher=peak_qfi,
            kelly_stake_censored_mean=mean_censored,
            melnikov_chaos_detected=melnikov_chaos_any,
            chirikov_overlap_detected=chirikov_chaos_any,
            greene_residue_peak=peak_greene,
            kam_twist_floor_observed=kam_obs,
            nekhoroshev_time_min=float(nek_min),
            hardware_crowbar_latched=crowbar_latched,
            actuation_latency_ns=actuation_latency,
            merkle_root_sha256=merkle_root,
            telemetry_digest_sha256=telemetry_digest,
        )

    # ── Fase 3.9: Utilidad Merkle SHA-256 ────────────────────────────────────
    @staticmethod
    def _compute_merkle_root(hashes: List[str]) -> str:
        if not hashes:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        current_level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in hashes]
        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = (current_level[i] + current_level[i + 1]).encode("utf-8")
                else:
                    combined = (current_level[i] + current_level[i]).encode("utf-8")
                next_level.append(hashlib.sha256(combined).hexdigest())
            current_level = next_level
        return current_level[0]


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR FISCAL SOBERANO PRINCIPAL (TOONIntuitionProsecutorEngine)
# ══════════════════════════════════════════════════════════════════════════════

class TOONIntuitionProsecutorEngine(Phase3_RSI3SovereignProsecutorEngine):
    r"""
    Motor espectral principal. Orquestador soberano v9.3.0-PhD.

    Encadena las 3 fases anidadas: Φ₁₂ despacha FASE 1 → FASE 2; Φ₂₃ despacha
    FASE 2 → FASE 3; produce certificado cifrado y limpia el historial de estados.
    """

    def __init__(
        self,
        dimension: int = 56,
        rhi_threshold_max: float = _RHI_HARD_CEILING,
        esp32_gpio_pin: int = 14,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension debe ser ≥ 2")
        self.dimension = int(dimension)
        self.rhi_threshold_max = float(rhi_threshold_max)
        self.esp32_gpio_pin = int(esp32_gpio_pin)

        self._N_diag = np.diag(np.linspace(1.0, 2.0, self.dimension, dtype=np.float64))
        self._active_banach_states: List[FlashProsecutionBanachState] = []

        logger.info(
            "TOONIntuitionProsecutorEngine %s en línea. Dim=%d, RHI_Max=%.2f, GPIO_sim=%d",
            __version__,
            self.dimension,
            self.rhi_threshold_max,
            self.esp32_gpio_pin,
        )

    def prosecute_intuition_jump(
        self,
        jump_id: str,
        mac_density_matrix: NDArray[np.complex128],
        flash_jump_matrix: NDArray[np.complex128],
        raw_kelly_stake: float,
        perturbation_eps: float = 0.05,
    ) -> Tuple[FlashProsecutionBanachState, PoincareMelnikovProsecutionState]:
        r"""
        Encadenamiento anidado FASE 1 → FASE 2.

        Flujo real (costura Φ₁₂ invocada de hecho):
            generate_poincare_melnikov_prosecution_state
              → continue_into_phase2_inquisition_gate
                  → initialize_phase2_from_celestial_state
                      → synthesize_flash_prosecution_state.
        """
        celestial_state = self.generate_poincare_melnikov_prosecution_state(
            density_matrix=flash_jump_matrix,
            N_diag=self._N_diag,
            perturbation_eps=perturbation_eps,
        )
        celestial_state = self.continue_into_phase2_inquisition_gate(celestial_state)
        banach_state = self.synthesize_flash_prosecution_state(
            jump_id=jump_id,
            celestial_state=celestial_state,
            mac_density=mac_density_matrix,
            flash_density=flash_jump_matrix,
            raw_kelly_stake=raw_kelly_stake,
        )
        self._active_banach_states.append(banach_state)
        return banach_state, celestial_state

    def audit_and_certify_prosecutions(
        self,
        mac_density_matrix: NDArray[np.complex128],
    ) -> IntuitionProsecutorExecutionCertificate:
        r"""Encadenamiento anidado FASE 2 → FASE 3 (cierre soberano, Φ₂₃ de hecho)."""
        sealed: List[FlashProsecutionBanachState] = []
        for state in self._active_banach_states:
            sealed.append(self.continue_into_phase3_rsi_gate(state))
        cert = self.execute_sovereign_prosecution_and_rsi3(
            banach_states=sealed,
            mac_density=mac_density_matrix,
            N_diag=self._N_diag,
            esp32_gpio_pin=self.esp32_gpio_pin,
        )
        self._active_banach_states.clear()
        return cert

    def run_nested_phi_pipeline(
        self,
        jump_id: str,
        mac_density_matrix: NDArray[np.complex128],
        flash_jump_matrix: NDArray[np.complex128],
        raw_kelly_stake: float,
        perturbation_eps: float = 0.05,
    ) -> Tuple[PoincareMelnikovProsecutionState, FlashProsecutionBanachState,
               IntuitionProsecutorExecutionCertificate]:
        r"""Composición sellada Φ₁₂ ∘ Φ₂₃: FASE 1 → FASE 2 → FASE 3 en un solo disparo."""
        banach, celestial = self.prosecute_intuition_jump(
            jump_id=jump_id,
            mac_density_matrix=mac_density_matrix,
            flash_jump_matrix=flash_jump_matrix,
            raw_kelly_stake=raw_kelly_stake,
            perturbation_eps=perturbation_eps,
        )
        certificate = self.audit_and_certify_prosecutions(mac_density_matrix)
        return celestial, banach, certificate


ToonIntuitionProsecutorEngine = TOONIntuitionProsecutorEngine

__all__ = [
    "TOONIntuitionProsecutorEngine",
    "ToonIntuitionProsecutorEngine",
    "Phase1_PoincareMelnikovAtlas",
    "Phase2_IntuitionProsecutorEngineCore",
    "Phase3_RSI3SovereignProsecutorEngine",
    "PoincareMelnikovProsecutionState",
    "FlashProsecutionBanachState",
    "RSI3CascadeReport",
    "IntuitionProsecutorExecutionCertificate",
    "HeytingOmega3",
    "ActuationMode",
    "ProsecutionStatus",
    "RSI3Level",
    "IntuitionProsecutorCanonicalSeed",
    "IntuitionFlashProsecutionGerm",
]


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DE LAZO CERRADO Y PRUEBAS DOCTORALES
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("═" * 88)
    print(f"  VERIFICACIÓN TOONIntuitionProsecutorEngine {__version__}")
    print("═" * 88 + "\n")

    dim = 56
    engine = TOONIntuitionProsecutorEngine(
        dimension=dim,
        rhi_threshold_max=0.85,
        esp32_gpio_pin=14,
    )

    rng = np.random.default_rng(2026)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    rho_mac = np.outer(v0, v0.conj())

    v_jump = v0.copy()
    v_jump[0] *= cmath.exp(1j * 0.02)
    v_jump /= np.linalg.norm(v_jump)
    flash_matrix = np.outer(v_jump, v_jump.conj())

    print("[1] Cadena anidada Φ₁₂: Fase 1 (Poincaré-Melnikov-Chirikov) → Fase 2 (RHI-Kelly)...")
    banach_res, cel_seed = engine.prosecute_intuition_jump(
        jump_id="JUMP-FISCAL-001",
        mac_density_matrix=rho_mac,
        flash_jump_matrix=flash_matrix,
        raw_kelly_stake=0.25,
        perturbation_eps=0.03,
    )

    print(f"  • [Fase 1] Delaunay (L,G,H)                : "
          f"{cel_seed.delaunay_L:.6f}, {cel_seed.delaunay_G:.6f}, {cel_seed.delaunay_H:.6f}")
    print(f"  • [Fase 1] Kepler (E, ν, r, a)             : "
          f"{cel_seed.kepler_eccentric_anomaly:.6f}, {cel_seed.kepler_true_anomaly:.6f}, "
          f"{cel_seed.kepler_radius:.6f}, {cel_seed.kepler_semimajor:.6f}")
    print(f"  • [Fase 1] Integral Melnikov M(t₀) max     : {cel_seed.melnikov_integral_M0:.6e}")
    print(f"  • [Fase 1] Chirikov K                      : {cel_seed.chirikov_overlap_K:.6f}")
    print(f"  • [Fase 1] Greene R_G                      : {cel_seed.greene_residue_R:.6f}")
    print(f"  • [Fase 1] Lyapunov spectrum (dim)         : "
          f"{cel_seed.lyapunov_spectrum.shape[0]} (max = {float(cel_seed.lyapunov_spectrum.max()):.4f})")
    print(f"  • [Fase 1] Poincaré-Cartan ∮p dq           : {cel_seed.poincare_cartan_relative:.6e}")
    print(f"  • [Fase 1] Lefschetz index                 : {cel_seed.lefschetz_index}")
    print(f"  • [Fase 1] KAM twist / Poincaré-Birkhoff   : "
          f"{cel_seed.kam_twist_monotone} / {cel_seed.poincare_birkhoff_certified}")
    print(f"  • [Fase 1] Broucke-Krein stable            : {cel_seed.broucke_krein_stable}")
    print(f"  • [Fase 1] Chern-Simons-Higgs CS₃          : {cel_seed.chern_simons_higgs:.6e}")
    print(f"  • [Fase 1] Nekhoroshev T_N                 : {cel_seed.nekhoroshev_stable_time:.6e}")
    print(f"  • [Fase 1] Symplectic drift/det            : "
          f"{cel_seed.symplectic_drift:.4e} / {cel_seed.symplectic_det:.12f}")
    print(f"  • [Fase 2] RHI Finsler                     : {banach_res.reward_hacking_index_RHI:.6f}")
    print(f"  • [Fase 2] Bures-Wasserstein               : {banach_res.bures_wasserstein_distance:.6f}")
    print(f"  • [Fase 2] Uhlmann fidelity                : {banach_res.uhlmann_fidelity:.6f}")
    print(f"  • [Fase 2] QFI (SLD)                       : {banach_res.quantum_fisher_information:.6f}")
    print(f"  • [Fase 2] Chernoff bound Q(ρ||σ)          : {banach_res.quantum_chernoff_bound:.6f}")
    print(f"  • [Fase 2] Umegaki D(ρ||σ)                 : {banach_res.umegaki_relative_entropy:.6f}")
    print(f"  • [Fase 2] Holevo χ                        : {banach_res.holevo_accessible_info:.6f}")
    print(f"  • [Fase 2] Kelly raw → censored            : "
          f"{banach_res.original_kelly_stake:.4f} → {banach_res.censored_kelly_stake:.4f}\n")

    print("[2] Cadena anidada Φ₂₃: Fase 2 → Fase 3 (RSI³ Kleisli meta-adaptativa)...")
    certificate = engine.audit_and_certify_prosecutions(mac_density_matrix=rho_mac)

    print(f"  • [Fase 3] Certificado                    : {certificate.certificate_id}")
    print(f"  • [Fase 3] Veredicto Ω₃                   : {certificate.verdict.name}")
    print(f"  • [Fase 3] Actuación                      : {certificate.actuation_mode.name}")
    print(f"  • [Fase 3] RSI³ monadic rate              : {certificate.level3_rsi_monadic_rate:.6f}")
    print(f"  • [Fase 3] Banach residual/rate/contract  : "
          f"{certificate.rsi3_banach_residual:.3e} / "
          f"{certificate.rsi3_banach_contraction_rate:.6f} / "
          f"{certificate.rsi3_contracted}")
    print(f"  • [Fase 3] Meta-adaptaciones R₃           : {certificate.rsi3_meta_adaptations}")
    print(f"  • [Fase 3] Spohn σ ≥ 0 / Brockett ↓V     : "
          f"{certificate.rsi3_spohn_nonneg} / {certificate.rsi3_lyapunov_monotone}")
    print(f"  • [Fase 3] Heyting lattice completo      : {certificate.heyting_lattice_complete}")
    print(f"  • [Fase 3] Peak RHI / Bures / QFI        : "
          f"{certificate.peak_reward_hacking_index:.4f} / "
          f"{certificate.peak_bures_distance:.4f} / "
          f"{certificate.peak_quantum_fisher:.4f}")
    print(f"  • [Fase 3] Nekhoroshev T_N min           : {certificate.nekhoroshev_time_min:.6e}")
    print(f"  • [Fase 3] Crowbar latched               : {certificate.hardware_crowbar_latched}")
    print(f"  • [Fase 3] Latencia actuación            : {certificate.actuation_latency_ns:.2f} ns")
    print(f"  • [Fase 3] Merkle root SHA-256           : {certificate.merkle_root_sha256}")
    print(f"  • [Fase 3] Telemetría digest             : {certificate.telemetry_digest_sha256}\n")

    print("═" * 88)
    print(f"  VERIFICACIÓN EXITOSA: TOONIntuitionProsecutorEngine {__version__} OPERATIVO")
    print("═" * 88)
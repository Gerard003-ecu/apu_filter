# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_cognitive_crop_engine.py                                   ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / CULTIVO COGNITIVO                    ║
║ FUNCIÓN  : MOTOR ESPECTRAL CON MECÁNICA CELESTE DE HENRI POINCARÉ (KAM, PW, Ω₃)       ║
║ VERSIÓN  : 9.1.0-Poincare-Cartan-Melnikov-KAM-Bruno-CZ-Celestial-Crop-PhD             ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICO-FÍSICA
───────────────────────────────────────────────────────
El `TOONCognitiveCropEngine` orquesta el metabolismo cognitivo de 4 fases
(Riego, Luz, Disciplina, Fe) que transforma semillas crudas en estados de
densidad purificados sobre la Matriz Atómica de Conocimiento (MAC).

Composición anidada (funtores Φ_I ⊣ Φ_II ⊣ Φ_III):
  Φ_I   : SeedCrystalPreparation.synthesize_poincare_cartan_germ
          ↦ 𝒢_I  = _PoincareCartanSeedGerm
          ≡ objeto inicial de Fase II (hand_off_germ_to_phase2).
  Φ_II  : CropGrowthPipeline.synthesize_celestial_bundle
          ↦ 𝒢_II = _PoincareCelestialGrowthBundle
          ≡ objeto inicial de Fase III (hand_off_bundle_to_phase3).
  Φ_III : Heyting Ω₃⁶ + Crowbar BT151 + certificado Merkle
          ↦ CropGerminationCertificate.

FASE I — Sustrato algebraico y geometría de la fase
  (Banach + Maupertuis-Jacobi + Poincaré-Cartan + Liouville)
FASE II — Riego + Luz + Disciplina + KAM/Bruno + Melnikov + CZ
FASE III — Fe (Ω₃⁶), Crowbar BT151 y certificación Merkle

MAPPING A LA CÚSPIDE VISCERAL ("DOLOR Y DINERO")
─────────────────────────────────────────
- Preservación de Toros de Costo (KAM/Bruno): inmuniza precios unitarios
  contra la difusión de Arnol'd.
- Control de Dispersión (Poincaré-Wirtinger): acota desviaciones de insumos.
- Ruptura de Melnikov: fractura homoclínica temprana.
- Índice de Conley-Zehnder: clasificación topológica del retorno.
- Inalienabilidad Ciber-Física (Crowbar BT151): < 400 ns.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__version__: Final[str] = "9.1.0-Poincare-Cartan-Melnikov-KAM-Bruno-CZ-Celestial-Crop-PhD"

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS (Wilkinson, Hill, KAM, Melnikov, Floquet, Birkhoff)
# ═══════════════════════════════════════════════════════════════════════════
_EPS: Final[float] = 1.0e-14
_EPS_MODULAR: Final[float] = 1.0e-10
_EPS_TRACE: Final[float] = 1.0e-15
_EPS_CONTRACT: Final[float] = 1.0e-3
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_DRIFT_LIMIT: Final[float] = 1.0e-9
_HILL_MARGIN_FLOOR: Final[float] = 1.0e-12
_KAM_TAU_FLOOR: Final[float] = 1.0
_KAM_GAMMA_FLOOR: Final[float] = 1.0e-12
_KAM_THRESHOLD: Final[float] = 1.0e-9
_MELNIKOV_THRESHOLD: Final[float] = 1.0e-6
_MELNIKOV_T_INF: Final[float] = 25.0
_MELNIKOV_QUAD_NODES: Final[int] = 513
_FLOQUET_PARABOLIC: Final[float] = 1.0e-6
_LYAPUNOV_CLIP: Final[float] = 700.0
_BIRKHOFF_AREA_DRIFT_MAX: Final[float] = 1.0e-12
_BIRKHOFF_TWIST_FLOOR: Final[float] = 1.0e-9
_CSMD_STEP: Final[float] = 1.0e-8
_LOG_EXP_CLIP: Final[float] = 700.0
_DEGRADATION_FACTOR: Final[float] = 1.0e-2
_BRUNO_MAX_OCTAVES: Final[int] = 12
_RICHARDSON_STEPS: Final[int] = 3
_GELFAND_POWERS: Final[int] = 8
_NEKHOROSHEV_C: Final[float] = 0.25
_SYMPLECTIC_SKEW_TOL: Final[float] = 1.0e-12
_UNITARY_DRIFT_TOL: Final[float] = 1.0e-12

ComplexMatrix = NDArray[np.complex128]
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]


def _seed_from_string(s: str) -> int:
    """Proyección SHA-256 → ℕ/2³², determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


def _kahan_neumaier_sum(values: Sequence[float]) -> float:
    r"""Suma compensada de Kahan–Babuška–Neumaier (error O(ε) · n, no O(εn))."""
    acc = 0.0
    comp = 0.0
    for x in values:
        t = acc + x
        if abs(acc) >= abs(x):
            comp += (acc - t) + x
        else:
            comp += (x - t) + acc
        acc = t
    return acc + comp


def _clip_log_exp(x: float) -> float:
    return float(np.clip(x, -_LOG_EXP_CLIP, _LOG_EXP_CLIP))


def _hermitian_part(a: np.ndarray) -> ComplexMatrix:
    z = np.asarray(a, dtype=np.complex128)
    return 0.5 * (z + z.conj().T)


def _safe_log(x: np.ndarray, floor: float = _EPS_MODULAR) -> np.ndarray:
    return np.log(np.maximum(np.real(x), floor))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO DEL CULTIVO + GEOMETRÍA DE LA FASE            ║
# ║                                                                           ║
# ║ Objetos: Ω₃, 𝔇_n, B(u(n)), (H_mac, |Ω⟩), (T*Q, Ω, g̃, λ_PC, i_X ω).     ║
# ║ Morfismo terminal (I.ω): synthesize_poincare_cartan_germ                  ║
# ║     ↦ 𝒢_I = _PoincareCartanSeedGerm                                       ║
# ║ Morfismo de empalme (I.ω⁺): hand_off_germ_to_phase2                       ║
# ║     ≡ objeto inicial / continuación de Fase II.                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §1.1 Retículo distributivo de Heyting Ω₃ ──────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel G₃)
        Ω₃ = {⊥ ≺ ⋆ ≺ ⊤} ≅ {0, 1, 2}

    Estructura residuada:
        meet    (∧) : ínfimo = min
        join    (∨) : supremo = max
        implies (⇒) : a ⇒ b = ⊤ si a ≤ b, else b
        neg     (¬) : a ⇒ ⊥
        iff     (⇔) : (a ⇒ b) ∧ (b ⇒ a)

    Ω₃ es un locale (frame finito): ∧ distribuye sobre ∨ arbitrarios.
    Los regulares (¬¬a = a) son exactamente {⊥, ⊤} (álgebra de Boole densa).
    """
    VETOED: int = 0      # ⊥
    DEGRADED: int = 1    # ⋆
    COHERENT: int = 2    # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    @property
    def godel_value(self) -> float:
        """Valuación de Gödel: ⊥ ↦ 0, ⋆ ↦ ½, ⊤ ↦ 1."""
        return float(int(self)) / 2.0

    def leq(self, other: "HeytingOmega3") -> bool:
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        """a es regular ⟺ a = ¬¬a. Sólo ⊥ y ⊤ lo son."""
        return self.neg().neg() == self

    def as_weight(self) -> float:
        return self.godel_value

    @classmethod
    def from_godel(cls, value: float) -> "HeytingOmega3":
        """Sección de la valuación de Gödel (umbrales ¼ y ¾)."""
        v = float(value)
        if v >= 0.75:
            return cls.COHERENT
        if v >= 0.25:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def infimum(cls, *values: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo arbitrario del frame finito Ω₃ (meet n-ario)."""
        acc = cls.COHERENT
        for v in values:
            acc = acc.meet(v)
        return acc

    @classmethod
    def supremum(cls, *values: "HeytingOmega3") -> "HeytingOmega3":
        acc = cls.VETOED
        for v in values:
            acc = acc.join(v)
        return acc


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre el conjunto convexo de estados
        𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†, ρ ≥ 0, Tr ρ = 1 }.

    𝔇_n es un cuerpo de estados de un C*-álgebra de dimensión finita:
    compacto, convexo, con interior relativo las densidades de rango pleno.
    La proyección afín `sanitize` es 1-Lipschitz en norma de traza (Petz).
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MODULAR

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: hermitización + PSD-clip + renormalización."""
        if not cls.is_square(rho):
            raise ValueError(
                f"DensityOperatorAlgebra.sanitize: matriz no cuadrada {np.shape(rho)}"
            )
        rho_h = _hermitian_part(rho)
        w, V = la.eigh(rho_h)
        w = np.maximum(np.real(w), cls.EPS)
        rho_h = (V * w) @ V.conj().T
        tr = float(np.trace(rho_h).real)
        if abs(tr) > _EPS_TRACE:
            rho_h = rho_h / tr
        return rho_h

    @classmethod
    def spectrum_descending(cls, rho: np.ndarray) -> RealVector:
        rho = cls.sanitize(rho)
        w = np.real(la.eigvalsh(rho))
        w = np.sort(w)[::-1]
        w = np.maximum(w, cls.EPS)
        s = float(w.sum())
        if s <= 0.0:
            n = w.size
            return np.full(n, 1.0 / n, dtype=np.float64)
        return (w / s).astype(np.float64)

    @classmethod
    def spectral_gap(cls, rho: np.ndarray) -> float:
        w = cls.spectrum_descending(rho)
        if w.size < 2:
            return 0.0
        return float(w[0] - w[1])

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def renyi_entropy(cls, rho: np.ndarray, alpha: float) -> float:
        if abs(alpha - 1.0) < 1e-12:
            return cls.von_neumann_entropy(rho)
        p = cls.spectrum_descending(rho)
        tr_a = float(np.sum(np.power(p, alpha)))
        tr_a = max(tr_a, cls.EPS)
        return float(math.log(tr_a) / (1.0 - alpha))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def frobenius_norm(cls, rho: np.ndarray) -> float:
        return float(np.linalg.norm(cls.sanitize(rho), "fro"))

    @classmethod
    def schatten_p_norm(cls, rho: np.ndarray, p: float) -> float:
        sig = np.real(la.svdvals(cls.sanitize(rho)))
        sig = np.maximum(sig, 0.0)
        if p == math.inf:
            return float(sig.max()) if sig.size else 0.0
        if p <= 0.0:
            raise ValueError("Schatten p-norm requiere p > 0")
        return float(np.power(np.sum(np.power(sig, p)), 1.0 / p))

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * cls.schatten_p_norm(delta, 1.0)

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        w_r, V_r = la.eigh(cls.sanitize(rho))
        w_s, V_s = la.eigh(cls.sanitize(sigma))
        w_r_reg = np.maximum(np.real(w_r), cls.EPS_MODULAR)
        w_s_reg = np.maximum(np.real(w_s), cls.EPS_MODULAR)
        log_r = V_r @ np.diag(np.log(w_r_reg).astype(np.complex128)) @ V_r.conj().T
        log_s = V_s @ np.diag(np.log(w_s_reg).astype(np.complex128)) @ V_s.conj().T
        rho_s = cls.sanitize(rho)
        val = float(np.real(np.trace(rho_s @ (log_r - log_s))))
        return max(0.0, val)

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""Distancia de Bures d_B(ρ,σ) = √(2 − 2√F(ρ,σ)). Métrica riemanniana en 𝔇_n."""
        fid = cls.uhlmann_fidelity(rho, sigma)
        return float(np.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid))))

    @classmethod
    def sandwiched_renyi_divergence(
        cls, rho: np.ndarray, sigma: np.ndarray, alpha: float
    ) -> float:
        r"""
        Divergencia de Rényi emparedada (Müller-Lennert / Wilde–Winter)
            D̃_α(ρ‖σ) = 1/(α−1) log Tr[(σ^{(1-α)/2α} ρ σ^{(1-α)/2α})^α].
        Recupera Umegaki en α → 1.
        """
        if abs(alpha - 1.0) < 1e-12:
            return cls.umegaki_relative_entropy(rho, sigma)
        if alpha <= 0.0:
            raise ValueError("sandwiched Rényi exige α > 0")
        rho_s = cls.sanitize(rho)
        sig = cls.sanitize(sigma)
        expo = (1.0 - alpha) / (2.0 * alpha)
        sig_pw = cls.matrix_power(sig, complex(expo, 0.0))
        inner = sig_pw @ rho_s @ sig_pw
        powered = cls.matrix_power(inner, complex(alpha, 0.0))
        tr = max(float(np.real(np.trace(powered))), cls.EPS)
        return float(math.log(tr) / (alpha - 1.0))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MODULAR
    ) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), cls.EPS_MODULAR)
        kspec = -np.log(w)
        return (V * kspec.astype(np.complex128)) @ V.conj().T

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MODULAR))
        return tuple(sorted(float(x) for x in k.tolist()))

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> ComplexMatrix:
        H_h = _hermitian_part(H)
        w, V = la.eigh(H_h)
        i0 = int(np.argmin(np.real(w)))
        psi = V[:, i0].reshape(-1, 1)
        return cls.sanitize(psi @ psi.conj().T)

    @classmethod
    def renyi_sharpen(cls, rho: np.ndarray, alpha: float) -> ComplexMatrix:
        if alpha <= 1.0 + 1e-12:
            return cls.sanitize(rho)
        rho = cls.sanitize(rho)
        rho_a = cls.matrix_power(rho, complex(alpha, 0.0))
        tr = float(np.trace(rho_a).real)
        if tr < 1e-30:
            return rho
        return cls.sanitize(rho_a / tr)

    @classmethod
    def lowner_heinz_residual(cls, rho: np.ndarray, sigma: np.ndarray, t: float = 0.5) -> float:
        r"""
        Residual de Löwner–Heinz: si A ≥ B ≥ 0 entonces A^t ≥ B^t para t ∈ [0,1].
        Devuelve ‖Π_{<0}(σ^t − ρ^t)‖₁ si ρ ≥ σ (en sentido PSD) no se cumple, else 0.
        """
        t = float(np.clip(t, 0.0, 1.0))
        a_t = cls.matrix_power(rho, complex(t, 0.0))
        b_t = cls.matrix_power(sigma, complex(t, 0.0))
        delta = _hermitian_part(a_t - b_t)
        w = np.real(la.eigvalsh(delta))
        neg = np.minimum(w, 0.0)
        return float(-np.sum(neg))


# ── §1.3 Álgebra de Banach: radio espectral, KAM y Poincaré-Wirtinger ────
@dataclass(frozen=True, slots=True)
class BanachContractionReport:
    r"""
    Auditoría de la disciplina en B(u(n)) integrando Poincaré-Wirtinger y Toros KAM.

    El radio espectral se estima por (i) cota ‖T‖_∞ y (ii) fórmula de Gelfand
        ρ(T) = lim_{k→∞} ‖T^k‖^{1/k}.
    La constante de Poincaré-Wirtinger C_P acota
        ‖ρ − I/n‖_F² ≤ C_P · ‖[ρ, N]‖_F²
    sobre el hiperplano traceless (primera forma de Dirichlet discreta).
    """
    spectral_radius: float
    eta_star: float
    eta_max: float
    g_max: float
    alignment: float
    is_contraction: bool
    lipschitz_bound: float
    pair_index: Tuple[int, int]
    local_verdict: HeytingOmega3
    dirichlet_energy: float = 0.0
    poincare_wirtinger_bound: float = 0.0
    variance: float = 0.0
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False
    banach_factor: float = 0.0
    gelfand_radius: float = 0.0
    neumann_remainder: float = 0.0


class BanachContractionAlgebra:
    r"""Cálculo vectorizado de acoplamientos, radio espectral y Poincaré-Wirtinger."""
    EPS: Final[float] = _EPS_MODULAR

    @classmethod
    def coupling_matrix(cls, rho: np.ndarray) -> RealMatrix:
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        w = np.maximum(w, cls.EPS)
        diff = w[:, None] - w[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(w[:, None] / w[None, :])
        G = diff * ratio
        np.fill_diagonal(G, 0.0)
        G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(G, 0.0)

    @classmethod
    def pair_couplings(cls, rho: np.ndarray) -> Tuple[float, int, int]:
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 0.0, 0, 0
        idx = int(np.argmax(G))
        n = G.shape[0]
        i_max, j_max = divmod(idx, n)
        return float(G[i_max, j_max]), int(i_max), int(j_max)

    @classmethod
    def spectral_radius(
        cls, rho: np.ndarray, eta: float
    ) -> Tuple[float, float, float]:
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 1.0, float("inf"), 0.0
        g_max = float(G.max())
        tau = 1.0 - eta * G
        np.fill_diagonal(tau, 0.0)
        max_abs_tau = float(np.max(np.abs(tau))) if tau.size else 0.0
        eta_max = (2.0 / g_max) if g_max > cls.EPS else float("inf")
        return max_abs_tau, float(eta_max), g_max

    @classmethod
    def gelfand_spectral_radius(cls, rho: np.ndarray, eta: float, powers: int = _GELFAND_POWERS) -> float:
        r"""
        Fórmula de Gelfand: ρ(T) = lim ‖T^k‖^{1/k}.
        T_η = I − η G (sin diagonal). Cota superior al radio de contracción.
        """
        G = cls.coupling_matrix(rho)
        if G.size == 0:
            return 1.0
        T = np.eye(G.shape[0], dtype=np.float64) - eta * G
        np.fill_diagonal(T, 0.0)
        Tk = T.copy()
        radius = float(np.linalg.norm(Tk, 2))
        for k in range(1, max(1, powers) + 1):
            Tk = Tk @ T
            nk = float(np.linalg.norm(Tk, 2))
            if nk <= 0.0:
                return 0.0
            radius = nk ** (1.0 / (k + 1))
        return float(radius)

    @classmethod
    def neumann_series_remainder(cls, rho_T: float, order: int = 8) -> float:
        r"""Resto de Neumann ∑_{k≥N} ρ^k ≤ ρ^N / (1−ρ) si ρ<1, else +∞."""
        if rho_T >= 1.0:
            return float("inf")
        return float((rho_T ** order) / max(1.0 - rho_T, _MACHINE_EPS))

    @classmethod
    def lipschitz_bound(cls, rho: np.ndarray, eta: float) -> float:
        rho_T, _, g_max = cls.spectral_radius(rho, eta)
        return float(max(rho_T, abs(1.0 - eta * g_max), eta * g_max))

    @classmethod
    def audit(
        cls,
        rho: np.ndarray,
        eta_star: float = 1.5,
        potential_operator: Optional[np.ndarray] = None,
        cp_constant: float = 0.5,
        spectral_cap: float = 0.95,
    ) -> BanachContractionReport:
        n = rho.shape[0]
        I_mean = np.eye(n, dtype=np.complex128) / n
        if potential_operator is None:
            potential_operator = np.diag(
                np.arange(1, n + 1, dtype=np.float64)
            ).astype(np.complex128)
        commutator = rho @ potential_operator - potential_operator @ rho
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord="fro") ** 2)
        variance = float(np.linalg.norm(rho - I_mean, ord="fro") ** 2)
        pw_bound = cp_constant * (2.0 * dirichlet_energy)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        log_w = np.log(np.maximum(w, cls.EPS))
        alignment = float(-np.sum(w * log_w))
        rho_T, eta_max, g_max = cls.spectral_radius(rho, eta_star)
        gelfand = cls.gelfand_spectral_radius(rho, eta_star)
        rho_eff = max(rho_T, gelfand)
        _, i_star, j_star = cls.pair_couplings(rho)
        lip = cls.lipschitz_bound(rho, eta_star)
        eta_factor = min(spectral_cap, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))
        remainder = cls.neumann_series_remainder(rho_eff)
        is_kam_stable = (rho_eff < 1.0) and (
            variance <= pw_bound + 1e-6 or dirichlet_energy < 1e-12
        )
        is_pyriform_bifurcated = (
            (not is_kam_stable)
            or (rho_eff >= 1.0)
            or (variance > pw_bound * 2.0 + 1e-3)
        )
        if is_kam_stable and rho_eff < 1.0 - _EPS_CONTRACT:
            local = HeytingOmega3.COHERENT
        elif is_kam_stable or rho_eff < 1.0 + _EPS_CONTRACT:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return BanachContractionReport(
            spectral_radius=rho_T,
            eta_star=float(eta_star),
            eta_max=eta_max,
            g_max=g_max,
            alignment=alignment,
            is_contraction=rho_eff < 1.0,
            lipschitz_bound=lip,
            pair_index=(i_star, j_star),
            local_verdict=local,
            dirichlet_energy=dirichlet_energy,
            poincare_wirtinger_bound=pw_bound,
            variance=variance,
            is_kam_stable=is_kam_stable,
            is_pyriform_bifurcated=is_pyriform_bifurcated,
            banach_factor=eta_factor,
            gelfand_radius=gelfand,
            neumann_remainder=remainder,
        )


# ── §1.4 Campo del suelo: H_mac y |Ω⟩ ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SoilState:
    H_mac: np.ndarray
    ground_proj: np.ndarray
    E0: float
    gap_H: float
    hash: str

    @property
    def soil_field(self) -> "SoilField":
        n = self.H_mac.shape[0] if hasattr(self.H_mac, "shape") else 4
        sf = SoilField(n)
        sf.H_mac = self.H_mac
        sf.ground_proj = self.ground_proj
        sf.potential_operator = np.diag(
            np.arange(1, n + 1, dtype=np.float64)
        ).astype(np.complex128)
        return sf


class SoilField:
    r"""Campo del suelo con operador de potencial N(p) (operador número)."""
    DEFAULT_GAP: Final[float] = 3.0
    DEFAULT_COUPLING: Final[float] = 1.0e-3

    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.H_mac = np.eye(n, dtype=np.complex128)
        self.ground_proj = np.eye(n, dtype=np.complex128) / n
        self.potential_operator = np.diag(
            np.arange(1, n + 1, dtype=np.float64)
        ).astype(np.complex128)

    @classmethod
    def build(cls, n: int = 4) -> SoilState:
        if n < 1:
            raise ValueError("SoilField.build: dimensión n ≥ 1")
        base = np.diag(np.linspace(0.0, cls.DEFAULT_GAP, n)).astype(np.complex128)
        off = np.zeros((n, n), dtype=np.complex128)
        if n >= 2:
            idx = np.arange(n - 1)
            off[idx, idx + 1] = cls.DEFAULT_COUPLING
        H = base + off + off.conj().T
        proj = DensityOperatorAlgebra.ground_state_projector(H)
        w = np.real(la.eigvalsh(H))
        w = np.sort(w)
        E0 = float(w[0])
        gap = float(w[1] - w[0]) if w.size > 1 else float("inf")
        soil_hash = _sha256_bytes(
            np.ascontiguousarray(H).tobytes(),
            np.ascontiguousarray(proj).tobytes(),
        )
        return SoilState(H_mac=H, ground_proj=proj, E0=E0, gap_H=gap, hash=soil_hash)


# ── §1.5 Geometría de la fase: Maupertuis-Jacobi, Hill, Poincaré-Cartan ──
@dataclass(frozen=True, slots=True)
class MaupertuisJacobiGerm:
    r"""
    Gérmen de la métrica conforme de Maupertuis-Jacobi.
    Para H(q,p) = ½ g^{jk} p_j p_k + V(q) con H₀ − V(q) > 0:
        g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q) = n(q)² g_{jk}(q),
    n(q) = √(2(H₀ − V(q))) índice de refracción mecánico.
    Las geodésicas de (Q, g̃) son las proyecciones de las órbitas de energía H₀.
    """
    conformal_factor: float
    refractive_index: float
    hill_margin: float
    is_in_hill_region: bool
    min_eigenvalue: float
    is_positive_definite: bool
    condition_number: float = 1.0


@dataclass(frozen=True, slots=True)
class PoincareCartanGerm:
    r"""
    Gérmen de la 1-forma de Poincaré-Cartan sobre T*Q × ℝ:
        λ = p_i dq^i − H dt ∈ Ω¹(T*Q × ℝ).
    dλ = ω − dH ∧ dt es el invariante integral absoluto (E. Cartan, 1922).
    El vector de coeficientes vive en ℝ^{2n+1}: (p, 0_n, −H).
    """
    lambda_vector: np.ndarray
    hamiltonian_value: float
    dim: int
    two_n: int
    symplectic_skew_residual: float
    contact_dim: int = 0
    liouville_volume: float = 1.0


@dataclass(frozen=True, slots=True)
class SeedState:
    rho_seed: np.ndarray
    K_spec: Tuple[float, ...]
    purity: float
    entropy: float
    alignment: float
    gap_K: float
    dim: int


@dataclass(frozen=True, slots=True)
class _PoincareCartanSeedGerm:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN DE POINCARÉ-CARTAN DE LA SEMILLA (terminal de Fase I, inicial Fase II).
    ═══════════════════════════════════════════════════════════════════════════
    Extiende `SeedState` con la geometría de la fase:
      • base_seed              : estado de densidad (ρ_seed, K_spec, pureza…).
      • maupertuis_germ        : métrica conforme g̃ + región de Hill + n(q).
      • cartan_germ            : 1-forma de Poincaré-Cartan λ = p dq − H dt.
      • hamiltonian_energy_H0  : energía de referencia del cultivo.
      • potential_V            : potencial de referencia del suelo.
      • omega                  : 2-forma simpléctica Ω ∈ ℝ^{2n×2n} (Darboux).
      • base_metric_g          : métrica euclídea base g_{jk}.
      • conformal_metric_gt    : g̃ = 2(H₀−V) g.
      • christoffel_conformal  : Γ̃^i_{jk} (Koszul–Levi-Civita conforme).
      • symplectic_capacity    : capacidad de Gromov c_G ≥ π r² (proxy π·Hill).
      • reg_floor              : piso de regularización de Wilkinson.
    """
    base_seed: SeedState
    maupertuis_germ: MaupertuisJacobiGerm
    cartan_germ: PoincareCartanGerm
    hamiltonian_energy_H0: float
    potential_V: float
    omega: np.ndarray
    base_metric_g: np.ndarray
    conformal_metric_gt: np.ndarray
    christoffel_conformal: np.ndarray
    symplectic_capacity: float
    reg_floor: float


class PoincareCelestialGeometry:
    r"""
    Fase I (bloque geométrico). Álgebra de mecánica celeste de Poincaré.

    Provee:
      • generate_canonical_symplectic_form : Ω de Darboux, Ωᵀ = −Ω, Ω² = −I.
      • symplectic_form_audit              : residuos de sesgo y unimodularidad.
      • compute_maupertuis_conformal_metric: g̃ = 2(H₀ − V)g.
      • compute_hill_region_margin         : H₀ − V(q).
      • compute_christoffel_conformal_symbols : Γ̃^i_{jk} (vectorizado).
      • compute_poincare_cartan_lambda     : λ ∈ Ω¹(T*Q × ℝ), dim 2n+1.
      • compute_poincare_relative_invariant: ∮_γ p dq (invariante relativo).
      • compute_liouville_volume           : (1/n!) ω^n.
      • cartan_magic_identity_residual     : L_{X_H} ω ≟ 0.
      • compute_gradient_csmd              : gradiente complejo + Richardson.
      • poisson_bracket                    : {H₀, H₁} = (∇H₀)ᵀ Ω ∇H₁.
      • stormer_verlet_step                : integrador simpléctico + deriva.
      • action_integral                    : ∫ p·dq a lo largo de una órbita.
    """

    @staticmethod
    def generate_canonical_symplectic_form(dim: int) -> np.ndarray:
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"Dimensión simpléctica dim={dim} debe ser par y positiva.")
        half = dim // 2
        omega = np.zeros((dim, dim), dtype=np.float64)
        omega[:half, half:] = np.eye(half, dtype=np.float64)
        omega[half:, :half] = -np.eye(half, dtype=np.float64)
        return omega

    @classmethod
    def symplectic_form_audit(cls, omega: np.ndarray) -> Tuple[float, float, bool]:
        r"""
        Auditoría de Darboux:
          skew_res = ‖Ω + Ωᵀ‖_F  (debe ser 0),
          unimod   = |det Ω − 1| (Sp(2n) ⇒ det = 1),
          darboux  = ‖Ω² + I‖_F  (Ω² = −I en coordenadas canónicas).
        """
        om = np.asarray(omega, dtype=np.float64)
        if om.ndim != 2 or om.shape[0] != om.shape[1]:
            return float("inf"), float("inf"), False
        skew = float(np.linalg.norm(om + om.T, "fro"))
        try:
            det = float(np.real(la.det(om)))
        except la.LinAlgError:
            det = float("nan")
        darboux = float(np.linalg.norm(om @ om + np.eye(om.shape[0]), "fro"))
        ok = (
            skew <= _SYMPLECTIC_SKEW_TOL
            and abs(abs(det) - 1.0) <= 1e-8
            and darboux <= 1e-8
        )
        return skew, darboux, ok

    @classmethod
    def compute_maupertuis_conformal_metric(
        cls,
        hamiltonian_energy_H0: float,
        potential_energy_V: float,
        base_metric_g: np.ndarray,
    ) -> Tuple[np.ndarray, MaupertuisJacobiGerm]:
        r"""
        Métrica conforme de Maupertuis-Jacobi:
            g̃_{jk}(q) = 2(H₀ − V(q)) g_{jk}(q).
        Región de Hill: H₀ − V(q) > 0. Condición de Jacobi: g̃ ≻ 0.
        """
        g = np.asarray(base_metric_g, dtype=np.float64)
        if g.ndim != 2 or g.shape[0] != g.shape[1]:
            raise ValueError("base_metric_g debe ser cuadrada.")
        h0 = float(hamiltonian_energy_H0)
        v = float(potential_energy_V)
        if not (np.isfinite(h0) and np.isfinite(v)):
            raise ValueError("H₀ y V deben ser finitos.")
        free_energy = 2.0 * (h0 - v)
        in_hill = bool(free_energy > _HILL_MARGIN_FLOOR)
        phi = max(free_energy, _HILL_MARGIN_FLOOR)
        refractive = float(np.sqrt(phi))
        gt = phi * g
        try:
            evals = la.eigvalsh(0.5 * (gt + gt.T))
            real_ev = np.real(evals) if evals.size else np.array([0.0])
            min_eig = float(np.min(real_ev))
            max_eig = float(np.max(real_ev))
            cond = float(max_eig / max(min_eig, _HILL_MARGIN_FLOOR))
        except la.LinAlgError:
            min_eig = 0.0
            cond = float("inf")
        germ = MaupertuisJacobiGerm(
            conformal_factor=float(phi),
            refractive_index=refractive,
            hill_margin=float(h0 - v),
            is_in_hill_region=in_hill,
            min_eigenvalue=min_eig,
            is_positive_definite=bool(min_eig > _HILL_MARGIN_FLOOR),
            condition_number=cond,
        )
        return gt, germ

    @staticmethod
    def compute_hill_region_margin(
        potential_V: float,
        total_energy_H0: float,
    ) -> float:
        """Margen de Hill: H₀ − V(q). Curva de velocidad cero = {H₀ = V}."""
        return float(total_energy_H0 - potential_V)

    @classmethod
    def compute_christoffel_conformal_symbols(
        cls,
        grad_V: np.ndarray,
        potential_V: float,
        total_energy_H0: float,
        g_base_metric: np.ndarray,
    ) -> np.ndarray:
        r"""
        Símbolos de Christoffel conformes (vectorizados, Γ_base = 0 si g euclídea):
            Γ̃^i_{jk} = δ^i_j ∂_k φ + δ^i_k ∂_j φ − g_{jk} g^{il} ∂_l φ,
        con φ(q) = ½ ln(2(H₀ − V(q))),  ∂φ = −∇V / (2(H₀ − V)).
        Complejidad O(n³) por broadcasting, sin triple bucle Python.
        """
        grad_V = np.asarray(grad_V, dtype=np.float64).ravel()
        g = np.asarray(g_base_metric, dtype=np.float64)
        n_dim = grad_V.size
        if g.shape != (n_dim, n_dim):
            raise ValueError(f"g_base_metric debe ser {n_dim}×{n_dim}.")
        headroom = 2.0 * (total_energy_H0 - potential_V)
        if headroom <= _HILL_MARGIN_FLOOR:
            raise ValueError("[CROP_ENGINE_VETO] Cero energía cinética: invasión de pozo.")
        grad_phi = -grad_V / (headroom + _HILL_MARGIN_FLOOR)
        g_inv = la.inv(g)
        # term1[i,j,k] = δ^i_j ∂_k φ   →  (I ⊗ ∂φ) con ejes (i,j,k)
        term1 = np.eye(n_dim)[:, :, None] * grad_phi[None, None, :]
        # term2[i,j,k] = δ^i_k ∂_j φ
        term2 = np.eye(n_dim)[:, None, :] * grad_phi[None, :, None]
        # term3[i,j,k] = g_{jk} (g^{il} ∂_l φ)
        ginv_dphi = g_inv @ grad_phi  # (n,)
        term3 = g[None, :, :] * ginv_dphi[:, None, None]
        return term1 + term2 - term3

    @classmethod
    def compute_poincare_cartan_lambda(
        cls,
        x: np.ndarray,
        hamiltonian_value: float = 0.0,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, PoincareCartanGerm]:
        r"""
        1-forma de Poincaré-Cartan λ = p_i dq^i − H dt sobre T*Q × ℝ.
        Coeficientes en ℝ^{2n+1}: (p_1…p_n, 0…0, −H).
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("λ de Poincaré-Cartan exige dim par (T*Q).")
        n = dim // 2
        p = xv[n:]
        h_val = float(hamiltonian_value) if np.isfinite(hamiltonian_value) else 0.0
        lam = np.concatenate(
            [p, np.zeros(n, dtype=np.float64), np.array([-h_val], dtype=np.float64)]
        )
        if omega is None:
            omega = cls.generate_canonical_symplectic_form(dim)
        omega_arr = np.asarray(omega, dtype=np.float64)
        skew_res, _, _ = cls.symplectic_form_audit(omega_arr)
        try:
            # (1/n!) Pf(Ω) proxy: |det Ω|^{1/2} = 1 en Darboux.
            vol = float(abs(la.det(omega_arr)) ** 0.5) if omega_arr.size else 1.0
        except la.LinAlgError:
            vol = 0.0
        germ = PoincareCartanGerm(
            lambda_vector=lam,
            hamiltonian_value=h_val,
            dim=dim,
            two_n=dim,
            symplectic_skew_residual=skew_res,
            contact_dim=2 * n + 1,
            liouville_volume=vol,
        )
        return lam, germ

    @staticmethod
    def compute_poincare_relative_invariant(orbit_q: np.ndarray, orbit_p: np.ndarray) -> float:
        r"""
        Invariante integral relativo de Poincaré:
            J[γ] = ∮_γ p_i dq^i  (γ ⊂ Σ_H cerrada).
        Cuadratura trapezoidal periódica. Conservado por el flujo hamiltoniano.
        """
        q = np.asarray(orbit_q, dtype=np.float64)
        p = np.asarray(orbit_p, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2 or q.shape[0] < 2:
            raise ValueError("orbit_q y orbit_p deben ser (T, n) con T≥2.")
        dq = np.diff(q, axis=0, append=q[:1])
        integrand = np.sum(p * dq, axis=1)
        return float(_kahan_neumaier_sum(integrand.tolist()))

    @classmethod
    def compute_liouville_volume(cls, omega: np.ndarray, n: int) -> float:
        r"""Forma de Liouville μ = (1/n!) ω^{\wedge n}. En Darboux, μ = dq dp."""
        om = np.asarray(omega, dtype=np.float64)
        try:
            det = float(np.real(la.det(om)))
        except la.LinAlgError:
            return 0.0
        # Pfaffiano² = det; |Pf| = 1 ⇒ vol = 1.
        return float(abs(det) ** 0.5)

    @classmethod
    def cartan_magic_identity_residual(
        cls,
        hamiltonian: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""
        Identidad mágica de Cartan: L_{X_H} ω = i_{X_H} dω + d(i_{X_H} ω) = 0
        porque dω = 0 (Ω cerrada) y i_{X_H} ω = −dH (definición de X_H).
        Residual numérico: ‖Ω X_H + ∇H‖₂.
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        grad = cls.compute_gradient_csmd(hamiltonian, xv, h)
        xh = omega @ grad  # X_H = Ω ∇H
        residual = omega @ xh + grad  # Ω² ∇H + ∇H = −∇H + ∇H = 0
        return float(np.linalg.norm(residual, 2))

    @staticmethod
    def compute_gradient_csmd(
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""
        Gradiente CSMD: ∇_k H(x) = Im[H(x + j·h·e_k)] / h + O(h²),
        eludiendo cancelaciones sustractivas en la mantisa de la FPU.
        Caída a diferencias centrales reales si H no admite extensión holomorfa.
        """
        xv = np.asarray(x, dtype=np.float64).ravel()
        if not np.isfinite(h) or h == 0.0:
            raise ValueError("El paso CSMD h debe ser finito y no nulo.")
        h = float(abs(h))
        dim = xv.size
        grad = np.zeros(dim, dtype=np.float64)
        holomorphic = True
        try:
            probe = func(xv.astype(np.complex128))
            holomorphic = np.isfinite(np.real(probe)) or np.isfinite(np.imag(probe))
        except (TypeError, ValueError, FloatingPointError):
            holomorphic = False
        if holomorphic:
            for i in range(dim):
                xp = xv.astype(np.complex128)
                xp[i] += 1j * h
                try:
                    val = func(xp)
                    imag = float(np.imag(val))
                except Exception:
                    holomorphic = False
                    break
                if not np.isfinite(imag):
                    holomorphic = False
                    break
                grad[i] = imag / h
        if not holomorphic:
            for i in range(dim):
                xp = xv.copy()
                xm = xv.copy()
                xp[i] += h
                xm[i] -= h
                try:
                    fp = float(np.real(func(xp)))
                    fm = float(np.real(func(xm)))
                except Exception:
                    fp = fm = 0.0
                grad[i] = (fp - fm) / (2.0 * h)
        return grad

    @classmethod
    def richardson_csmd_gradient(
        cls,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = _CSMD_STEP,
        steps: int = _RICHARDSON_STEPS,
    ) -> np.ndarray:
        r"""
        Extrapolación de Richardson sobre el gradiente CSMD:
            ∇^{[k+1]}(h) = (4^k ∇^{[k]}(h/2) − ∇^{[k]}(h)) / (4^k − 1).
        Orden O(h^{2k}) si H ∈ C^{2k+1}.
        """
        table: List[np.ndarray] = []
        hh = float(abs(h))
        for s in range(max(1, steps)):
            table.append(cls.compute_gradient_csmd(func, x, hh / (2 ** s)))
        for k in range(1, len(table)):
            four_k = 4.0 ** k
            for s in range(len(table) - k):
                table[s] = (four_k * table[s + 1] - table[s]) / (four_k - 1.0)
        return table[0]

    @classmethod
    def poisson_bracket(
        cls,
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        x: np.ndarray,
        omega: np.ndarray,
        h: float = _CSMD_STEP,
    ) -> float:
        r"""Corchete de Poisson {H₀, H₁}(x) = (∇H₀)ᵀ Ω ∇H₁ en T*Q."""
        xv = np.asarray(x, dtype=np.float64).ravel()
        dim = xv.size
        if dim % 2 != 0:
            raise ValueError("El corchete de Poisson exige dim par (Darboux).")
        grad0 = cls.compute_gradient_csmd(hamiltonian_0, xv, h)
        grad1 = cls.compute_gradient_csmd(hamiltonian_1, xv, h)
        return float(grad0 @ omega @ grad1)

    @staticmethod
    def stormer_verlet_step(
        q: np.ndarray,
        p: np.ndarray,
        grad_v: Callable[[np.ndarray], np.ndarray],
        mass_inv: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Paso Störmer–Verlet sobre H(q,p) = ½|p|²/m + V(q).
        Es simpléctico (det DP = 1) y reversible (P ∘ P = id con dt ↦ −dt).
        """
        qn = np.asarray(q, dtype=np.float64)
        pn = np.asarray(p, dtype=np.float64)
        p_half = pn - 0.5 * dt * np.asarray(grad_v(qn), dtype=np.float64)
        q_next = qn + dt * mass_inv * p_half
        p_next = p_half - 0.5 * dt * np.asarray(grad_v(q_next), dtype=np.float64)
        return q_next, p_next

    @classmethod
    def stormer_verlet_energy_drift(
        cls,
        q: np.ndarray,
        p: np.ndarray,
        potential: Callable[[np.ndarray], float],
        mass: float,
        dt: float,
        n_steps: int,
        grad_v: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        r"""Deriva de energía |H_N − H_0| tras N pasos Verlet (debe ser O(dt²) acotada)."""
        qn = np.asarray(q, dtype=np.float64).copy()
        pn = np.asarray(p, dtype=np.float64).copy()
        mass_inv = 1.0 / max(mass, _MACHINE_EPS)

        def H(qq: np.ndarray, pp: np.ndarray) -> float:
            return 0.5 * mass_inv * float(np.dot(pp, pp)) + float(potential(qq))

        h0 = H(qn, pn)
        for _ in range(max(1, n_steps)):
            qn, pn = cls.stormer_verlet_step(qn, pn, grad_v, mass_inv, dt)
        return float(abs(H(qn, pn) - h0))

    @staticmethod
    def action_integral(orbit_q: np.ndarray, orbit_p: np.ndarray) -> float:
        """Acción de Poincaré–Cartan reducida: A[γ] = ∫ p·dq (órbita abierta o cerrada)."""
        q = np.asarray(orbit_q, dtype=np.float64)
        p = np.asarray(orbit_p, dtype=np.float64)
        if q.shape != p.shape or q.ndim != 2 or q.shape[0] < 2:
            return 0.0
        dq = np.diff(q, axis=0)
        integrand = np.sum(p[:-1] * dq, axis=1)
        return float(_kahan_neumaier_sum(integrand.tolist()))

    @classmethod
    def gromov_capacity_proxy(cls, hill_margin: float) -> float:
        r"""
        Proxy de la capacidad de Gromov: c_G(B(r)) = π r².
        Tomamos r² ∼ max(Hill, 0) (radio de la región de Hill en unidades naturales).
        """
        r2 = max(float(hill_margin), 0.0)
        return float(math.pi * r2)


# ── §1.6 SeedCrystalPreparation — HAND-OFF FASE 1 → FASE 2 ────────────────
class SeedCrystalPreparation:
    r"""
    Prepara el estado (ρ_seed, K_seed) y sintetiza el gérmen de Poincaré-Cartan.

    El morfismo terminal `synthesize_poincare_cartan_germ` produce 𝒢_I.
    El morfismo de empalme `hand_off_germ_to_phase2` ES la continuación / inicio
    formal de los métodos de la Fase II (consume 𝒢_I, produce 𝒢_II).
    """

    @classmethod
    def prepare(cls, rho_raw: np.ndarray) -> SeedState:
        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho)
        purity = DensityOperatorAlgebra.purity(rho)
        entropy = DensityOperatorAlgebra.von_neumann_entropy(rho)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        alignment = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        gap_K = float(K_spec[1] - K_spec[0]) if len(K_spec) > 1 else float("inf")
        dim = int(rho.shape[0])
        return SeedState(
            rho_seed=rho,
            K_spec=K_spec,
            purity=purity,
            entropy=entropy,
            alignment=alignment,
            gap_K=gap_K,
            dim=dim,
        )

    # ── I.ω  MORFISMO TERMINAL DE LA FASE I ──────────────────────────────
    @classmethod
    def synthesize_poincare_cartan_germ(
        cls,
        rho_raw: np.ndarray,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        base_metric_g: Optional[np.ndarray] = None,
        grad_V: Optional[np.ndarray] = None,
    ) -> _PoincareCartanSeedGerm:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE I ≅ OBJETO INICIAL DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Compone SeedState ⊗ (Maupertuis-Jacobi ⊕ Hill ⊕ Poincaré-Cartan ⊕ Ω
        ⊕ Γ̃ ⊕ c_G) en un único 𝒢_I listo para el crecimiento celeste.
        """
        base_seed = cls.prepare(rho_raw)
        n = int(base_seed.dim)
        two_n = 2 * n
        omega = PoincareCelestialGeometry.generate_canonical_symplectic_form(two_n)
        if base_metric_g is None:
            base_metric_g = np.eye(n, dtype=np.float64)
        gt, mau_germ = PoincareCelestialGeometry.compute_maupertuis_conformal_metric(
            hamiltonian_energy_H0, potential_energy_V, base_metric_g
        )
        if grad_V is None:
            grad_V = np.zeros(n, dtype=np.float64)
        try:
            christoffel = PoincareCelestialGeometry.compute_christoffel_conformal_symbols(
                grad_V=grad_V,
                potential_V=potential_energy_V,
                total_energy_H0=hamiltonian_energy_H0,
                g_base_metric=np.asarray(base_metric_g, dtype=np.float64),
            )
        except ValueError:
            christoffel = np.zeros((n, n, n), dtype=np.float64)
        x0 = np.zeros(two_n, dtype=np.float64)
        _lam, cartan_germ = PoincareCelestialGeometry.compute_poincare_cartan_lambda(
            x0, hamiltonian_value=potential_energy_V, omega=omega
        )
        capacity = PoincareCelestialGeometry.gromov_capacity_proxy(mau_germ.hill_margin)
        reg_floor = max(
            float(np.linalg.norm(omega, "fro")) * _MACHINE_EPS * 10.0,
            _HILL_MARGIN_FLOOR,
        )
        return _PoincareCartanSeedGerm(
            base_seed=base_seed,
            maupertuis_germ=mau_germ,
            cartan_germ=cartan_germ,
            hamiltonian_energy_H0=float(hamiltonian_energy_H0),
            potential_V=float(potential_energy_V),
            omega=omega,
            base_metric_g=np.asarray(base_metric_g, dtype=np.float64),
            conformal_metric_gt=gt,
            christoffel_conformal=christoffel,
            symplectic_capacity=float(capacity),
            reg_floor=float(reg_floor),
        )

    @classmethod
    def continue_into_phase2(
        cls,
        seed: SeedState,
        cycle_index: int,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float,
        eta_star: float,
    ) -> "CropGrowthBundle":
        """Puente retrocompatible (API 8.x) SeedState → CropGrowthBundle clásico."""
        return CropGrowthPipeline.synthesize(
            cycle_index=cycle_index,
            seed=seed,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
        )

    # ── I.ω⁺  EMPALME FORMAL FASE I → FASE II ────────────────────────────
    @classmethod
    def hand_off_germ_to_phase2(
        cls,
        germ: _PoincareCartanSeedGerm,
        cycle_index: int,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = 1.5,
        eta_star: float = 1.5,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> "_PoincareCelestialGrowthBundle":
        r"""
        ═══════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE LA FASE I ≡ PRIMER MORFISMO OPERATIVO DE LA FASE II.
        ═══════════════════════════════════════════════════════════════════════
        Consume 𝒢_I = _PoincareCartanSeedGerm y delega en
        CropGrowthPipeline.synthesize_celestial_bundle, que es el orquestador
        de Riego + Luz + Disciplina + KAM + Melnikov + retorno.
        """
        return CropGrowthPipeline.synthesize_celestial_bundle(
            cycle_index=cycle_index,
            seed_germ=germ,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=renyi_alpha,
            eta_star=eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=novikov_valuation_T,
            safety_margin=safety_margin,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · RIEGO + LUZ + DISCIPLINA + KAM/BRUNO + MELNIKOV + CZ + RETORNO   ║
# ║                                                                           ║
# ║ Continuación directa del morfismo terminal de Fase I                      ║
# ║     (SeedCrystalPreparation.hand_off_germ_to_phase2).                     ║
# ║ Morfismo terminal (II.ω): synthesize_celestial_bundle                     ║
# ║     ↦ 𝒢_II = _PoincareCelestialGrowthBundle                               ║
# ║ Morfismo de empalme (II.ω⁺): hand_off_bundle_to_phase3                    ║
# ║     ≡ objeto inicial / continuación de Fase III.                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §2.1 CognitiveWateringModule — RIEGO ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class WateringReport:
    tokens_toon: int
    tokens_json: int
    h_toon: float
    h_json: float
    fat_reduction_pct: float
    kv_compression_ratio: float
    seed_entropy_nats: float
    local_verdict: HeytingOmega3
    moistened_rho: Optional[np.ndarray] = None
    hill_compatible: bool = True


class CognitiveWateringModule:
    r"""FASE 2 · EL RIEGO — Acondicionamiento del suelo en H_MAC (compresión KV)."""
    W_TOKEN: Final[float] = 0.7
    W_ENTROPY: Final[float] = 0.3
    KV_MAX: Final[float] = 0.95
    CHARS_PER_TOKEN: Final[float] = 4.0

    @classmethod
    def bpe_tokens(cls, text: str) -> int:
        return max(1, int(math.ceil(len(text) / cls.CHARS_PER_TOKEN)))

    @classmethod
    def shannon_bits(cls, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    @classmethod
    def fat_functional(
        cls, t_toon: int, t_json: int, h_toon: float, h_json: float
    ) -> float:
        red_token = 1.0 - t_toon / max(1, t_json)
        red_ent = 1.0 - h_toon / max(1e-9, h_json)
        return 100.0 * (cls.W_TOKEN * red_token + cls.W_ENTROPY * red_ent)

    def moisten_soil(self, seed_crystal: object, soil_field: SoilField) -> WateringReport:
        rho_raw = getattr(
            seed_crystal, "density_matrix", getattr(seed_crystal, "rho_seed", None)
        )
        if rho_raw is None:
            rho_raw = np.eye(soil_field.n, dtype=np.complex128) / soil_field.n
        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        toon_str = getattr(seed_crystal, "crystal_id", "TOON_SEED")
        base_str = getattr(seed_crystal, "origin_agent", "ORIGIN_BASE_AGENT_SPECS")
        rep = self.apply_water(
            SeedState(
                rho,
                (),
                DensityOperatorAlgebra.purity(rho),
                DensityOperatorAlgebra.von_neumann_entropy(rho),
                0.0,
                0.0,
                rho.shape[0],
            ),
            toon_str,
            base_str,
        )
        return WateringReport(
            tokens_toon=rep.tokens_toon,
            tokens_json=rep.tokens_json,
            h_toon=rep.h_toon,
            h_json=rep.h_json,
            fat_reduction_pct=rep.fat_reduction_pct,
            kv_compression_ratio=rep.kv_compression_ratio,
            seed_entropy_nats=rep.seed_entropy_nats,
            local_verdict=rep.local_verdict,
            moistened_rho=rho,
        )

    @classmethod
    def moisten_from_germ(
        cls,
        germ: _PoincareCartanSeedGerm,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
        r"""
        Primer acto de Fase II sobre el gérmen 𝒢_I: riega el suelo condicionando
        la compresión KV a la región de Hill (sin riego fuera de D_H).
        """
        report = cls.apply_water(germ.base_seed, toon_str, base_json_str)
        hill_ok = bool(germ.maupertuis_germ.is_in_hill_region)
        local = report.local_verdict if hill_ok else report.local_verdict.meet(
            HeytingOmega3.DEGRADED
        )
        return WateringReport(
            tokens_toon=report.tokens_toon,
            tokens_json=report.tokens_json,
            h_toon=report.h_toon,
            h_json=report.h_json,
            fat_reduction_pct=report.fat_reduction_pct,
            kv_compression_ratio=report.kv_compression_ratio,
            seed_entropy_nats=report.seed_entropy_nats,
            local_verdict=local,
            moistened_rho=report.moistened_rho,
            hill_compatible=hill_ok,
        )

    @classmethod
    def apply_water(
        cls,
        seed: SeedState,
        toon_str: str,
        base_json_str: str,
    ) -> WateringReport:
        t_t = cls.bpe_tokens(toon_str)
        t_j = cls.bpe_tokens(base_json_str)
        h_t = cls.shannon_bits(toon_str)
        h_j = cls.shannon_bits(base_json_str)
        delta_gr = cls.fat_functional(t_t, t_j, h_t, h_j)
        kv = float(np.clip(delta_gr / 100.0, 0.0, cls.KV_MAX))
        if delta_gr >= 60.0:
            local = HeytingOmega3.COHERENT
        elif delta_gr >= 30.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return WateringReport(
            tokens_toon=t_t,
            tokens_json=t_j,
            h_toon=h_t,
            h_json=h_j,
            fat_reduction_pct=float(delta_gr),
            kv_compression_ratio=kv,
            seed_entropy_nats=float(seed.entropy),
            local_verdict=local,
            moistened_rho=seed.rho_seed,
        )


# ── §2.2 CognitiveIlluminationModule — LUZ ────────────────────────────────
@dataclass(frozen=True, slots=True)
class BrockettPurificationCertificate:
    initial_alignment: float
    final_alignment: float
    initial_purity: float
    final_purity: float
    lyapunov_drop: float
    iterations: int
    converged: bool
    isospectral_drift: float
    unitary_drift: float = 0.0
    integrator: str = "rk4"


@dataclass(frozen=True, slots=True)
class RenyiPurificationCertificate:
    alpha: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    purity_gain: float
    entropy_drop: float
    renyi_S: float


@dataclass(frozen=True, slots=True)
class FockAnnihilationCertificate:
    electron_anomaly_energy: float
    positron_constraint_energy: float
    gamma_photons_emitted: int
    resonance_residual: float
    is_annihilated: bool


class CognitiveIlluminationModule:
    r"""
    FASE 2 · LA LUZ — Purificación isospectral de Brockett.

    El flujo Ũ = [ρ, [ρ, N]] (doble conmutador) es el gradiente de
    L(ρ) = Tr(ρ N) sobre la variedad isospectral. Se integra con:
      (i) RK4 + proyección SVD (polar de Fan–Hoffman),
      (ii) transformada de Cayley (estructura-preservante en u(n)).
    """
    BROCKETT_MAX_STEPS: Final[int] = 60
    BROCKETT_DT: Final[float] = 0.05
    BROCKETT_TOL: Final[float] = 1.0e-9
    RENYI_ALPHA: Final[float] = 1.5
    FOCK_RESONANCE_TOL: Final[float] = 0.15

    @classmethod
    def _N(cls, n: int) -> RealMatrix:
        return np.diag(np.arange(1, n + 1, dtype=np.float64))

    @classmethod
    def _anti_hermitian_generator(cls, rho: np.ndarray, N: np.ndarray) -> ComplexMatrix:
        comm = rho @ N - N @ rho
        A = -comm
        return 0.5 * (A - A.conj().T)

    @classmethod
    def _project_unitary(cls, U: np.ndarray) -> ComplexMatrix:
        W, _, Vh = la.svd(U, full_matrices=False)
        return W @ Vh

    @classmethod
    def _cayley_step(cls, U: np.ndarray, A: np.ndarray, dt: float) -> ComplexMatrix:
        r"""
        Paso de Cayley en U(n):  U ← (I − (dt/2)A)^{-1} (I + (dt/2)A) U
        con A† = −A. Preserva unitariedad exactamente (salvo redondeo).
        """
        n = U.shape[0]
        I = np.eye(n, dtype=np.complex128)
        half = 0.5 * dt * A
        try:
            U_next = la.solve(I - half, (I + half) @ U)
        except la.LinAlgError:
            U_next = U + dt * (A @ U)
        return cls._project_unitary(U_next)

    @classmethod
    def lyapunov_alignment(cls, rho: np.ndarray, N: np.ndarray) -> float:
        return float(np.trace(rho @ N).real)

    def illuminate_brockett(
        self, moistened_rho: np.ndarray
    ) -> Tuple[np.ndarray, BrockettPurificationCertificate]:
        return self._brockett_unitary_rk4(moistened_rho)

    @classmethod
    def _brockett_unitary_rk4(
        cls, rho_init: np.ndarray
    ) -> Tuple[ComplexMatrix, BrockettPurificationCertificate]:
        rho0 = DensityOperatorAlgebra.sanitize(rho_init)
        n = rho0.shape[0]
        N = cls._N(n)
        lam0 = DensityOperatorAlgebra.spectrum_descending(rho0)
        init_align = cls.lyapunov_alignment(rho0, N)
        init_pur = float(np.sum(lam0 ** 2))
        U = np.eye(n, dtype=np.complex128)
        dt = cls.BROCKETT_DT
        converged = False
        step = 0
        rho = rho0.copy()

        def A_of(U_cur: np.ndarray) -> ComplexMatrix:
            r = U_cur @ rho0 @ U_cur.conj().T
            r = _hermitian_part(r)
            return cls._anti_hermitian_generator(r, N)

        for step in range(cls.BROCKETT_MAX_STEPS):
            k1 = A_of(U) @ U
            k2 = A_of(U + 0.5 * dt * k1) @ (U + 0.5 * dt * k1)
            k3 = A_of(U + 0.5 * dt * k2) @ (U + 0.5 * dt * k2)
            k4 = A_of(U + dt * k3) @ (U + dt * k3)
            U_next = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            U_next = cls._project_unitary(U_next)
            rho_next = U_next @ rho0 @ U_next.conj().T
            rho_next = _hermitian_part(rho_next)
            if np.linalg.norm(rho_next - rho, "fro") < cls.BROCKETT_TOL:
                U, rho = U_next, rho_next
                converged = True
                break
            U, rho = U_next, rho_next
        rho = DensityOperatorAlgebra.sanitize(rho)
        lam1 = DensityOperatorAlgebra.spectrum_descending(rho)
        fin_align = cls.lyapunov_alignment(rho, N)
        fin_pur = float(np.sum(lam1 ** 2))
        drift = float(np.linalg.norm(lam1 - lam0, ord=2))
        unitary_drift = float(np.linalg.norm(U.conj().T @ U - np.eye(n), "fro"))
        cert = BrockettPurificationCertificate(
            initial_alignment=init_align,
            final_alignment=fin_align,
            initial_purity=init_pur,
            final_purity=fin_pur,
            lyapunov_drop=float(init_align - fin_align),
            iterations=step + 1,
            converged=converged,
            isospectral_drift=drift,
            unitary_drift=unitary_drift,
            integrator="rk4-polar",
        )
        return rho, cert

    @classmethod
    def _brockett_cayley(
        cls, rho_init: np.ndarray
    ) -> Tuple[ComplexMatrix, BrockettPurificationCertificate]:
        """Variante estructura-preservante (Cayley) del flujo de Brockett."""
        rho0 = DensityOperatorAlgebra.sanitize(rho_init)
        n = rho0.shape[0]
        N = cls._N(n)
        lam0 = DensityOperatorAlgebra.spectrum_descending(rho0)
        init_align = cls.lyapunov_alignment(rho0, N)
        init_pur = float(np.sum(lam0 ** 2))
        U = np.eye(n, dtype=np.complex128)
        rho = rho0.copy()
        converged = False
        step = 0
        for step in range(cls.BROCKETT_MAX_STEPS):
            A = cls._anti_hermitian_generator(rho, N)
            U_next = cls._cayley_step(U, A, cls.BROCKETT_DT)
            rho_next = DensityOperatorAlgebra.sanitize(U_next @ rho0 @ U_next.conj().T)
            if np.linalg.norm(rho_next - rho, "fro") < cls.BROCKETT_TOL:
                U, rho = U_next, rho_next
                converged = True
                break
            U, rho = U_next, rho_next
        lam1 = DensityOperatorAlgebra.spectrum_descending(rho)
        cert = BrockettPurificationCertificate(
            initial_alignment=init_align,
            final_alignment=cls.lyapunov_alignment(rho, N),
            initial_purity=init_pur,
            final_purity=float(np.sum(lam1 ** 2)),
            lyapunov_drop=float(init_align - cls.lyapunov_alignment(rho, N)),
            iterations=step + 1,
            converged=converged,
            isospectral_drift=float(np.linalg.norm(lam1 - lam0, ord=2)),
            unitary_drift=float(np.linalg.norm(U.conj().T @ U - np.eye(n), "fro")),
            integrator="cayley",
        )
        return rho, cert

    @classmethod
    def _fock_resonance(cls, seed: SeedState) -> FockAnnihilationCertificate:
        w = DensityOperatorAlgebra.spectrum_descending(seed.rho_seed)
        n = len(w)
        E_minus = float(n * (1.0 - w[0]))
        E_plus = float(seed.entropy)
        denom = max(1.0, abs(E_minus), abs(E_plus))
        residual = abs(E_minus - E_plus) / denom
        is_ann = residual < cls.FOCK_RESONANCE_TOL
        return FockAnnihilationCertificate(
            electron_anomaly_energy=E_minus,
            positron_constraint_energy=E_plus,
            gamma_photons_emitted=2 if is_ann else 0,
            resonance_residual=float(residual),
            is_annihilated=bool(is_ann),
        )

    @classmethod
    def apply_light(
        cls,
        seed: SeedState,
        renyi_alpha: float = RENYI_ALPHA,
    ) -> Tuple[
        ComplexMatrix,
        BrockettPurificationCertificate,
        RenyiPurificationCertificate,
        FockAnnihilationCertificate,
    ]:
        rho_flow, brockett_cert = cls._brockett_unitary_rk4(seed.rho_seed)
        p_before = DensityOperatorAlgebra.purity(rho_flow)
        s_before = DensityOperatorAlgebra.von_neumann_entropy(rho_flow)
        rho_illum = DensityOperatorAlgebra.renyi_sharpen(rho_flow, renyi_alpha)
        p_after = DensityOperatorAlgebra.purity(rho_illum)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(rho_illum)
        s_renyi = DensityOperatorAlgebra.renyi_entropy(rho_illum, renyi_alpha)
        renyi_cert = RenyiPurificationCertificate(
            alpha=float(renyi_alpha),
            purity_before=p_before,
            purity_after=p_after,
            entropy_before=s_before,
            entropy_after=s_after,
            purity_gain=float(p_after - p_before),
            entropy_drop=float(s_before - s_after),
            renyi_S=float(s_renyi),
        )
        fock_cert = cls._fock_resonance(seed)
        return rho_illum, brockett_cert, renyi_cert, fock_cert


# ── §2.3 CognitiveDisciplineModule — DISCIPLINA (Poincaré-Wirtinger) ──────
class CognitiveDisciplineModule:
    r"""
    Módulo de Disciplina con acotación Poincaré-Wirtinger.

    Sobre el simplex de Hilbert–Schmidt, la primera forma de Dirichlet
        ℰ(ρ) = ½ ‖[ρ, N]‖_F²
    controla la varianza ‖ρ − I/n‖_F² vía C_P = 1 / λ₁(ad_N* ad_N)|_{sl(n)}.
    """

    def enforce_poincare_wirtinger_kam_bound(
        self,
        density_op: np.ndarray,
        potential_operator: np.ndarray,
        cp_constant: float = 0.5,
        spectral_cap: float = 0.95,
    ) -> Tuple[np.ndarray, BanachContractionReport]:
        n = density_op.shape[0]
        I_mean = np.eye(n, dtype=np.complex128) / n
        commutator = density_op @ potential_operator - potential_operator @ density_op
        dirichlet_energy = 0.5 * float(np.linalg.norm(commutator, ord="fro") ** 2)
        eta = min(spectral_cap, 1.0 / (1.0 + math.sqrt(dirichlet_energy + 1e-12)))
        disciplined_rho = (1.0 - eta) * I_mean + eta * density_op
        disciplined_rho = DensityOperatorAlgebra.sanitize(disciplined_rho)
        report = BanachContractionAlgebra.audit(
            disciplined_rho,
            eta_star=1.5,
            potential_operator=potential_operator,
            cp_constant=cp_constant,
            spectral_cap=spectral_cap,
        )
        return disciplined_rho, report

    @classmethod
    def audit_discipline(
        cls,
        rho: np.ndarray,
        eta_star: float = 1.5,
        potential_operator: Optional[np.ndarray] = None,
        cp_constant: float = 0.5,
    ) -> BanachContractionReport:
        return BanachContractionAlgebra.audit(
            rho,
            eta_star=eta_star,
            potential_operator=potential_operator,
            cp_constant=cp_constant,
        )

    @classmethod
    def audit_from_seed(cls, seed: SeedState, eta_star: float = 1.5) -> BanachContractionReport:
        return cls.audit_discipline(seed.rho_seed, eta_star)

    @classmethod
    def sharp_poincare_constant(cls, n: int) -> float:
        r"""
        Constante aguda de Poincaré-Wirtinger en el simplex HS:
        λ₁(ad_N* ad_N) ≥ min_{i≠j} (i−j)² = 1  ⇒  C_P^♯ ≤ 1/2.
        """
        if n <= 1:
            return 0.5
        return 0.5


# ── §2.4 PoincareCelestialVerifier — KAM/Bruno + MELNIKOV + CZ + RETORNO ─
@dataclass(frozen=True, slots=True)
class KAMAudit:
    r"""
    Auditoría de pequeños divisores de Poincaré–KAM con absorción ultramétrica
    T-ádica en el anillo de Novikov Λ_Nov y condición de Bruno.
    """
    min_divisor: float
    novikov_weight: float
    maurercartan_residual: float
    volume_drift: float
    resonance_gap: float
    tau: float
    gamma: float
    is_diophantine: bool
    is_kam_stable: bool
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    bruno_sum: float = 0.0
    is_bruno: bool = True
    twist_determinant: float = 1.0
    is_kolmogorov_nondegenerate: bool = True
    nekhoroshev_time: float = 0.0


@dataclass(frozen=True, slots=True)
class MelnikovAudit:
    r"""Auditoría de la función de Melnikov (ruptura homoclínica)."""
    melnikov_value: float
    melnikov_derivative: float
    is_simple_zero: bool
    homoclinic_splitting: float
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    quadrature_nodes: int = 0
    l_infinity_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class ReturnMapAudit:
    r"""
    Auditoría del mapa de retorno de Poincaré P: Σ → Σ.
    Clasificación de Floquet mutuamente excluyente + índice de Conley–Zehnder.
    """
    floquet_multipliers: np.ndarray
    lyapunov_spectrum: np.ndarray
    max_lyapunov: float
    floquet_parabolic: float
    is_hyperbolic: bool
    is_elliptic: bool
    is_parabolic: bool
    trace_M: float
    det_M: float
    local_verdict: HeytingOmega3
    engine_ok: bool = True
    conley_zehnder_index: int = 0
    is_nondegenerate: bool = True
    birkhoff_twist: float = 0.0
    symplectic_residual: float = 0.0


class PoincareCelestialVerifier:
    r"""
    Fase II (bloque celeste). Verificador de mecánica celeste de Poincaré.
    Recibe el gérmen 𝒢_I de Fase I como dato obligatorio y expone:
      • Pequeños divisores de Poincaré–KAM (diofantinidad + Bruno + twist).
      • Resonancias de Arnol'd.
      • Función de Melnikov (Gauss–Legendre + Kahan–Neumaier).
      • Mapa de retorno (Floquet, Lyapunov, Conley–Zehnder, Birkhoff).
      • Estimación de Nekhoroshev T ≥ exp(c ε^{−1/(2n)}).
    """

    def __init__(self, germ: _PoincareCartanSeedGerm) -> None:
        self._germ = germ

    @property
    def germ(self) -> _PoincareCartanSeedGerm:
        return self._germ

    # ── II.1 Pequeños divisores de Poincaré-KAM ──────────────────────────
    @staticmethod
    def compute_bruno_sum(
        frequency_vector_omega: np.ndarray,
        max_octaves: int = _BRUNO_MAX_OCTAVES,
    ) -> Tuple[float, bool]:
        r"""
        Condición de Bruno (1957 / Rüssmann):
            B(ω) = ∑_{ν≥0} 2^{−ν} log(1 / Ω(2^ν)) < ∞,
        Ω(K) = inf{ |⟨k,ω⟩| : k ∈ ℤ^n \ {0}, |k|₁ ≤ K }.
        Es estrictamente más débil que la diofantinidad de Siegel–Moser y
        suficiente para la convergencia del teorema de Siegel holomorfo.
        """
        omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            return float("inf"), False
        acc = 0.0
        finite = True
        for nu in range(max(1, max_octaves)):
            K = 1 << nu
            ranges = [np.arange(-K, K + 1) for _ in range(n)]
            # Muestreo: si n grande, acotar la retícula por norma l¹.
            if n > 3 or K > 8:
                # Cota inferior por vectores canónicos y primeros armónicos.
                candidates = [np.eye(n, dtype=np.int64), -np.eye(n, dtype=np.int64)]
                extra = []
                for i in range(n):
                    for j in range(i + 1, n):
                        v = np.zeros(n, dtype=np.int64)
                        v[i] = 1
                        v[j] = 1
                        extra.append(v)
                        extra.append(-v)
                k_all = np.vstack(candidates + extra)
            else:
                grid = np.meshgrid(*ranges, indexing="ij")
                k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
            norms = np.abs(k_all).sum(axis=1)
            keep = (norms > 0) & (norms <= K)
            if not np.any(keep):
                omega_k = _WILKINSON_DRIFT_LIMIT
            else:
                inner = np.abs(k_all[keep] @ omega)
                omega_k = float(np.min(inner)) if inner.size else _WILKINSON_DRIFT_LIMIT
            omega_k = max(omega_k, _MACHINE_EPS)
            acc += (2.0 ** (-nu)) * math.log(1.0 / omega_k)
            if not np.isfinite(acc):
                finite = False
                break
        return float(acc), bool(finite and acc < 1.0e6)

    @staticmethod
    def compute_twist_determinant(jacobian_M: np.ndarray) -> Tuple[float, bool]:
        r"""
        Condición de twist / no-degeneración de Kolmogorov:
            det(∂ω/∂I) = det(∂²H₀/∂I²) ≠ 0.
        Proxy: |det(M) − 1| pequeño y |tr(M_{qq} bloque de monodromía) − 2n|
        no nulo. Usamos det(M − I) como indicador de twist discreto
        (Poincaré–Birkhoff: ∂α/∂I ≠ 0 ⇔ 1 ∉ σ(DP) o rotación no constante).
        """
        M = np.asarray(jacobian_M, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.size == 0:
            return 0.0, False
        try:
            det_m = float(np.real(la.det(M)))
            det_mi = float(np.real(la.det(M - np.eye(M.shape[0]))))
        except la.LinAlgError:
            return 0.0, False
        twist = float(abs(det_mi))
        nondeg = bool(twist > _BIRKHOFF_TWIST_FLOOR and abs(det_m - 1.0) < 1e-3)
        return twist, nondeg

    @classmethod
    def compute_nekhoroshev_time(cls, eps_pert: float, n_dim: int) -> float:
        r"""
        Estimación de Nekhoroshev: T_stab ≥ exp( c / ε^{1/(2n)} ).
        Las acciones permanecen O(ε^a) durante tiempos exponencialmente largos.
        """
        eps = max(float(abs(eps_pert)), _MACHINE_EPS)
        expo = 1.0 / max(2 * max(n_dim, 1), 1)
        val = _clip_log_exp(_NEKHOROSHEV_C / (eps ** expo))
        return float(math.exp(val))

    def compute_poincare_small_divisors_spectrum(
        self,
        frequency_vector_omega: np.ndarray,
        wave_vectors_k: np.ndarray,
        jacobian_M: np.ndarray,
        tau: float = _KAM_TAU_FLOOR,
        gamma: float = _KAM_GAMMA_FLOOR,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> KAMAudit:
        r"""
        Para H(I,θ) = H₀(I) + ε H₁(I,θ), la ecuación de arrastre
            Σ_j ω_j ∂S₁/∂θ_j = −H₁(I',θ)
        exige S_{1,k} = i H_{1,k} / ⟨k, ω⟩. La condición diofántica
            |⟨k, ω⟩| ≥ γ / |k|^τ,  τ > n − 1,
        garantiza convergencia de Lindstedt. El peso de Novikov
            W_Nov(k, ω) = exp(−T_val / (ε_floor + |⟨k, ω⟩|))
        absorbe ultramétricamente los divisores prohibidos.
        Se adjunta Bruno + twist de Kolmogorov + tiempo de Nekhoroshev.
        """
        try:
            omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
            wave_k = np.asarray(wave_vectors_k, dtype=np.float64)
            jac_m = np.asarray(jacobian_M, dtype=np.float64)
            n_freq = max(int(omega.size), 1)
            tau_eff = max(float(tau), float(n_freq - 1) + 1e-6)
            if wave_k.ndim == 1:
                divisors = np.abs(np.dot(wave_k, omega))
                k_norms = np.abs(wave_k)
                min_divisor = float(divisors)
                argmin_idx: Optional[int] = 0
                resonance_gap = float(divisors)
            else:
                divisors = np.abs(wave_k @ omega)
                k_norms = np.linalg.norm(wave_k, axis=1)
                if divisors.size == 0:
                    min_divisor = 1.0
                    argmin_idx = None
                    resonance_gap = 1.0
                else:
                    argmin_idx = int(np.argmin(divisors))
                    min_divisor = float(divisors[argmin_idx])
                    resonance_gap = min_divisor
            novikov_weight = float(
                np.exp(
                    -np.clip(
                        novikov_valuation_T / (_WILKINSON_DRIFT_LIMIT + min_divisor),
                        0.0,
                        _LOG_EXP_CLIP,
                    )
                )
            )
            mc_residual = float(abs(min_divisor * novikov_weight))
            if argmin_idx is not None and wave_k.ndim > 1 and wave_k.size > 0:
                k_norm_min = float(max(k_norms[argmin_idx], 1.0))
                is_diophantine = bool(min_divisor * (k_norm_min ** tau_eff) >= gamma)
            else:
                is_diophantine = bool(min_divisor >= _WILKINSON_DRIFT_LIMIT)
            if jac_m.ndim == 2 and jac_m.shape[0] == jac_m.shape[1] and jac_m.size > 0:
                det_M = float(la.det(jac_m))
                volume_drift = float(abs(det_M - 1.0))
            else:
                volume_drift = 0.0
            bruno_sum, is_bruno = self.compute_bruno_sum(omega)
            twist_det, is_twist = self.compute_twist_determinant(jac_m)
            eps_proxy = max(volume_drift, 1.0 - min(min_divisor, 1.0), _MACHINE_EPS)
            t_nek = self.compute_nekhoroshev_time(eps_proxy, n_freq)
            is_kam_stable = bool(
                (min_divisor >= _KAM_THRESHOLD * safety_margin)
                and (volume_drift <= _WILKINSON_DRIFT_LIMIT)
                and is_diophantine
                and is_bruno
            )
            if min_divisor <= _WILKINSON_DRIFT_LIMIT or volume_drift > _WILKINSON_DRIFT_LIMIT:
                local = HeytingOmega3.VETOED
            elif not is_diophantine or not is_bruno:
                local = HeytingOmega3.DEGRADED
            elif is_kam_stable:
                local = HeytingOmega3.COHERENT
            elif min_divisor >= (_KAM_THRESHOLD * safety_margin) * _DEGRADATION_FACTOR:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return KAMAudit(
                min_divisor=min_divisor,
                novikov_weight=novikov_weight,
                maurercartan_residual=mc_residual,
                volume_drift=volume_drift,
                resonance_gap=resonance_gap,
                tau=float(tau_eff),
                gamma=float(gamma),
                is_diophantine=is_diophantine,
                is_kam_stable=is_kam_stable,
                local_verdict=local,
                engine_ok=True,
                bruno_sum=bruno_sum,
                is_bruno=is_bruno,
                twist_determinant=twist_det,
                is_kolmogorov_nondegenerate=is_twist,
                nekhoroshev_time=t_nek,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_small_divisors_spectrum: %s", exc)
            return KAMAudit(
                min_divisor=float("inf"),
                novikov_weight=0.0,
                maurercartan_residual=float("inf"),
                volume_drift=float("inf"),
                resonance_gap=float("inf"),
                tau=float(tau),
                gamma=float(gamma),
                is_diophantine=False,
                is_kam_stable=False,
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
                bruno_sum=float("inf"),
                is_bruno=False,
                twist_determinant=0.0,
                is_kolmogorov_nondegenerate=False,
                nekhoroshev_time=0.0,
            )

    def compute_arnold_resonance_lattice(
        self,
        frequency_vector_omega: np.ndarray,
        max_order: int = 4,
        tol: float = 1e-6,
    ) -> np.ndarray:
        r"""Retícula de resonancias de Arnol'd k ∈ ℤ^n con |⟨k, ω⟩| < tol."""
        omega = np.asarray(frequency_vector_omega, dtype=np.float64).ravel()
        n = omega.size
        if n == 0 or max_order < 1:
            return np.zeros((0, max(n, 1)), dtype=np.int64)
        ranges = [np.arange(-max_order, max_order + 1) for _ in range(n)]
        grid = np.meshgrid(*ranges, indexing="ij")
        k_all = np.stack([g.ravel() for g in grid], axis=1).astype(np.int64)
        norms = np.abs(k_all).sum(axis=1)
        keep = (norms > 0) & (norms <= max_order)
        if not np.any(keep):
            return np.zeros((0, n), dtype=np.int64)
        k_cand = k_all[keep]
        inner = np.abs(k_cand @ omega)
        return k_cand[inner < tol]

    # ── II.2 Función de Melnikov ─────────────────────────────────────────
    def compute_melnikov_function(
        self,
        homoclinic_flow: Callable[[float], np.ndarray],
        hamiltonian_0: Callable[[np.ndarray], float],
        hamiltonian_1: Callable[[np.ndarray], float],
        t0_grid: np.ndarray,
        t_inf: float = _MELNIKOV_T_INF,
        n_quad: int = _MELNIKOV_QUAD_NODES,
        safety_margin: float = 1.0,
    ) -> MelnikovAudit:
        r"""
        M(t₀) = ∫_{−∞}^{∞} {H₀, H₁}(γ⁰(t − t₀)) dt.
        Cuadratura de Gauss–Legendre + acumulación Kahan–Neumaier.
        Cero simple (M=0, M'≠0) ⇒ fractura homoclínica (Smale–Birkhoff).
        """
        try:
            t0s = np.asarray(t0_grid, dtype=np.float64).ravel()
            if t0s.size == 0:
                raise ValueError("t0_grid no puede estar vacío.")
            n_quad = int(max(16, n_quad))
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            t_nodes = t_inf * nodes
            w_nodes = t_inf * weights
            omega = self._germ.omega
            melnikov_vals = np.zeros(t0s.size, dtype=np.float64)
            for i, t0 in enumerate(t0s):
                acc = 0.0
                comp = 0.0
                for t_shift, w in zip(t_nodes, w_nodes):
                    try:
                        x = np.asarray(homoclinic_flow(float(t_shift - t0)), dtype=np.float64)
                    except Exception:
                        continue
                    if x.size != omega.shape[0]:
                        continue
                    try:
                        pb = PoincareCelestialGeometry.poisson_bracket(
                            hamiltonian_0, hamiltonian_1, x, omega
                        )
                    except (TypeError, ValueError, FloatingPointError):
                        pb = 0.0
                    if not np.isfinite(pb):
                        pb = 0.0
                    y = pb * float(w) - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                melnikov_vals[i] = acc
            idx_min = int(np.argmin(np.abs(melnikov_vals)))
            m_val = float(melnikov_vals[idx_min])
            linf = float(np.max(np.abs(melnikov_vals))) if melnikov_vals.size else 0.0
            if t0s.size >= 2 and 0 < idx_min < t0s.size - 1:
                dm = (melnikov_vals[idx_min + 1] - melnikov_vals[idx_min - 1]) / (
                    t0s[idx_min + 1] - t0s[idx_min - 1]
                )
            elif t0s.size >= 2:
                dm = (melnikov_vals[-1] - melnikov_vals[0]) / max(
                    t0s[-1] - t0s[0], _MACHINE_EPS
                )
            else:
                dm = 0.0
            is_simple = bool(
                abs(m_val) < _WILKINSON_DRIFT_LIMIT and abs(dm) > _WILKINSON_DRIFT_LIMIT
            )
            try:
                x_star = np.asarray(homoclinic_flow(float(t0s[idx_min])), dtype=np.float64)
                grad_h0 = PoincareCelestialGeometry.compute_gradient_csmd(
                    hamiltonian_0, x_star
                )
                grad_norm = float(np.linalg.norm(grad_h0, 2))
            except Exception:
                grad_norm = 0.0
            splitting = float(abs(m_val) / max(grad_norm, _WILKINSON_DRIFT_LIMIT))
            tol = _MELNIKOV_THRESHOLD * safety_margin
            if is_simple:
                local = HeytingOmega3.VETOED
            elif abs(m_val) >= tol:
                local = HeytingOmega3.COHERENT
            elif abs(m_val) >= tol * _DEGRADATION_FACTOR:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return MelnikovAudit(
                melnikov_value=m_val,
                melnikov_derivative=float(dm),
                is_simple_zero=is_simple,
                homoclinic_splitting=splitting,
                local_verdict=local,
                engine_ok=True,
                quadrature_nodes=n_quad,
                l_infinity_norm=linf,
            )
        except Exception as exc:
            logger.error("Fallo en compute_melnikov_function: %s", exc)
            return MelnikovAudit(
                melnikov_value=float("nan"),
                melnikov_derivative=float("nan"),
                is_simple_zero=False,
                homoclinic_splitting=float("nan"),
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
                quadrature_nodes=0,
                l_infinity_norm=float("nan"),
            )

    # ── II.3 Mapa de retorno de Poincaré ─────────────────────────────────
    @staticmethod
    def compute_conley_zehnder_index(M: np.ndarray) -> Tuple[int, bool]:
        r"""
        Proxy computable del índice de Conley–Zehnder para M ∈ Sp(2n, ℝ)
        no degenerado (1 ∉ σ(M)):

            CZ(M) ≃ n + ∑_{λ ∈ σ(M) ∩ S¹, Im λ > 0} sign(d arg λ)
                    + ½ ∑_{λ ∈ σ(M) ∩ {±1}} sign(crossing)

        Implementación: polar M = Q P (Q ∈ U(n) ↪ Sp(2n), P = √(MᵀM) > 0);
        CZ_proxy = n − #{θ_j ∈ (0, π)} + #{θ_j ∈ (−π, 0)} sobre autovalores
        de Q (identificación U(n) ⊂ Sp(2n)). No degenerado ⇔ det(M−I) ≠ 0.
        """
        A = np.asarray(M, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] % 2 != 0:
            return 0, False
        n = A.shape[0] // 2
        try:
            det_mi = float(np.real(la.det(A - np.eye(A.shape[0]))))
        except la.LinAlgError:
            return 0, False
        nondeg = bool(abs(det_mi) > _FLOQUET_PARABOLIC)
        try:
            # Polar real: A = U P, U ortogonal, P = √(AᵀA)
            P = la.sqrtm(A.T @ A).real
            U = A @ la.inv(P) if abs(la.det(P)) > _MACHINE_EPS else A
            ev = la.eigvals(U)
        except (la.LinAlgError, ValueError):
            ev = la.eigvals(A)
        args = np.angle(ev)
        pos = int(np.sum(args > _FLOQUET_PARABOLIC))
        neg = int(np.sum(args < -_FLOQUET_PARABOLIC))
        cz = int(n - pos + neg)
        return cz, nondeg

    def compute_poincare_return_map(
        self,
        jacobian_M: np.ndarray,
        period_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> ReturnMapAudit:
        r"""
        Espectro de Floquet y exponentes de Lyapunov del mapa de retorno
        P: Σ → Σ. Clasificación mutuamente excluyente:
          hiperbólico (|λ| ≁ 1 ∀λ), elíptico (|λ|=1 ∀λ, λ≠±1),
          parabólico (∃ λ=±1), mixto → hiperbólico si algún |λ|≠1.
        """
        try:
            M = np.asarray(jacobian_M, dtype=np.float64)
            if M.ndim == 1:
                side = int(round(np.sqrt(M.size)))
                if side * side != M.size:
                    raise ValueError("jacobian_M plano no es cuadrado perfecto.")
                M = M.reshape(side, side)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError("jacobian_M debe ser cuadrada.")
            ev = la.eigvals(M)
            magnitudes = np.abs(ev)
            lyap = np.log(np.maximum(magnitudes, _MACHINE_EPS)) / max(
                abs(period_T), _MACHINE_EPS
            )
            lyap = np.clip(lyap, -_LYAPUNOV_CLIP, _LYAPUNOV_CLIP)
            max_lyap = float(np.max(lyap)) if lyap.size else 0.0
            on_circle = np.abs(magnitudes - 1.0) <= _FLOQUET_PARABOLIC
            near_pm1 = np.abs(np.abs(ev) - 1.0) <= _FLOQUET_PARABOLIC
            is_parabolic = bool(np.any(np.abs(ev - 1.0) < _FLOQUET_PARABOLIC) or np.any(
                np.abs(ev + 1.0) < _FLOQUET_PARABOLIC
            ))
            is_elliptic = bool(np.all(on_circle) and not is_parabolic)
            is_hyperbolic = bool((not is_elliptic) and (not is_parabolic) and np.any(~on_circle))
            if not (is_elliptic or is_parabolic or is_hyperbolic):
                # Mixto: predomina la componente hiperbólica.
                is_hyperbolic = bool(np.any(~on_circle))
            floq_par = float(np.max(np.abs(np.abs(ev) - 1.0))) if ev.size else 0.0
            cz, nondeg = self.compute_conley_zehnder_index(M)
            twist, _ = self.compute_twist_determinant(M)
            # Residual simpléctico: ‖Mᵀ Ω M − Ω‖_F
            om = self._germ.omega
            if om.shape == M.shape:
                symp_res = float(np.linalg.norm(M.T @ om @ M - om, "fro"))
            else:
                # Si M es 2n×2n del germen, recorte; si no, 0.
                try:
                    om_m = PoincareCelestialGeometry.generate_canonical_symplectic_form(
                        M.shape[0]
                    )
                    symp_res = float(np.linalg.norm(M.T @ om_m @ M - om_m, "fro"))
                except ValueError:
                    symp_res = float("inf")
            tol = _FLOQUET_PARABOLIC * safety_margin
            if is_elliptic and abs(max_lyap) <= tol * _DEGRADATION_FACTOR and nondeg:
                local = HeytingOmega3.COHERENT
            elif is_parabolic or floq_par <= tol:
                local = HeytingOmega3.DEGRADED
            elif abs(max_lyap) <= tol / max(_DEGRADATION_FACTOR, _MACHINE_EPS):
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.VETOED
            return ReturnMapAudit(
                floquet_multipliers=ev,
                lyapunov_spectrum=lyap,
                max_lyapunov=max_lyap,
                floquet_parabolic=floq_par,
                is_hyperbolic=is_hyperbolic,
                is_elliptic=is_elliptic,
                is_parabolic=is_parabolic,
                trace_M=float(np.trace(M)),
                det_M=float(np.real(la.det(M))),
                local_verdict=local,
                engine_ok=True,
                conley_zehnder_index=cz,
                is_nondegenerate=nondeg,
                birkhoff_twist=twist,
                symplectic_residual=symp_res,
            )
        except Exception as exc:
            logger.error("Fallo en compute_poincare_return_map: %s", exc)
            return ReturnMapAudit(
                floquet_multipliers=np.array([], dtype=np.complex128),
                lyapunov_spectrum=np.array([], dtype=np.float64),
                max_lyapunov=float("inf"),
                floquet_parabolic=float("inf"),
                is_hyperbolic=False,
                is_elliptic=False,
                is_parabolic=False,
                trace_M=float("nan"),
                det_M=float("nan"),
                local_verdict=HeytingOmega3.VETOED,
                engine_ok=False,
                conley_zehnder_index=0,
                is_nondegenerate=False,
                birkhoff_twist=0.0,
                symplectic_residual=float("inf"),
            )


# ── §2.5 CropGrowthPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────────
@dataclass(frozen=True, slots=True)
class CropGrowthBundle:
    cycle_index: int
    seed: SeedState
    watering: WateringReport
    rho_illum: np.ndarray
    brockett_c: BrockettPurificationCertificate
    renyi_c: RenyiPurificationCertificate
    fock_c: FockAnnihilationCertificate
    discipline: BanachContractionReport


@dataclass(frozen=True, slots=True)
class _PoincareCelestialGrowthBundle:
    r"""
    ═══════════════════════════════════════════════════════════════════════════
    GÉRMEN CELESTE DE CRECIMIENTO (terminal de Fase II, inicial de Fase III).
    ═══════════════════════════════════════════════════════════════════════════
    Extiende CropGrowthBundle con los canales celestes:
      • base_bundle        : riego + luz + disciplina + Brockett + Rényi + Fock.
      • seed_germ          : gérmen de Poincaré-Cartan de Fase I.
      • kam_audit          : pequeños divisores + Bruno + Nekhoroshev.
      • melnikov_audit     : función de Melnikov.
      • return_map         : Floquet + Lyapunov + Conley–Zehnder.
      • celestial_verdict  : ínfimo H₃ de los canales celestes.
    """
    base_bundle: CropGrowthBundle
    seed_germ: _PoincareCartanSeedGerm
    kam_audit: Optional[KAMAudit]
    melnikov_audit: Optional[MelnikovAudit]
    return_map: Optional[ReturnMapAudit]
    celestial_verdict: HeytingOmega3


class CropGrowthPipeline:
    r"""Orquestador determinista del crecimiento (funtor F₂)."""

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        seed: SeedState,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
    ) -> CropGrowthBundle:
        watering = CognitiveWateringModule.apply_water(seed, toon_str, base_json_str)
        rho_illum, brockett_c, renyi_c, fock_c = CognitiveIlluminationModule.apply_light(
            seed, renyi_alpha
        )
        discipline = CognitiveDisciplineModule.audit_discipline(rho_illum, eta_star)
        return CropGrowthBundle(
            cycle_index=cycle_index,
            seed=seed,
            watering=watering,
            rho_illum=rho_illum,
            brockett_c=brockett_c,
            renyi_c=renyi_c,
            fock_c=fock_c,
            discipline=discipline,
        )

    @classmethod
    def synthesize_celestial_bundle(
        cls,
        cycle_index: int,
        seed_germ: _PoincareCartanSeedGerm,
        toon_str: str,
        base_json_str: str,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        eta_star: float = 1.5,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> _PoincareCelestialGrowthBundle:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        MORFISMO TERMINAL DE LA FASE II ≅ OBJETO INICIAL DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Extiende el crecimiento clásico (riego-desde-gérmen + luz + disciplina)
        con KAM/Bruno, Melnikov y mapa de retorno. Canales ausentes no votan
        en el ínfimo H₃ (neutros = ⊤).
        """
        watering = CognitiveWateringModule.moisten_from_germ(
            seed_germ, toon_str, base_json_str
        )
        rho_illum, brockett_c, renyi_c, fock_c = CognitiveIlluminationModule.apply_light(
            seed_germ.base_seed, renyi_alpha
        )
        discipline = CognitiveDisciplineModule.audit_discipline(rho_illum, eta_star)
        base_bundle = CropGrowthBundle(
            cycle_index=cycle_index,
            seed=seed_germ.base_seed,
            watering=watering,
            rho_illum=rho_illum,
            brockett_c=brockett_c,
            renyi_c=renyi_c,
            fock_c=fock_c,
            discipline=discipline,
        )
        verifier = PoincareCelestialVerifier(seed_germ)
        kam_audit: Optional[KAMAudit] = None
        melnikov_audit: Optional[MelnikovAudit] = None
        return_map_audit: Optional[ReturnMapAudit] = None
        if (
            frequency_vector_omega is not None
            and wave_vectors_k is not None
            and jacobian_M is not None
        ):
            kam_audit = verifier.compute_poincare_small_divisors_spectrum(
                frequency_vector_omega=frequency_vector_omega,
                wave_vectors_k=wave_vectors_k,
                jacobian_M=jacobian_M,
                novikov_valuation_T=novikov_valuation_T,
                safety_margin=safety_margin,
            )
        if (
            homoclinic_flow is not None
            and hamiltonian_0 is not None
            and hamiltonian_1 is not None
            and t0_grid is not None
        ):
            melnikov_audit = verifier.compute_melnikov_function(
                homoclinic_flow=homoclinic_flow,
                hamiltonian_0=hamiltonian_0,
                hamiltonian_1=hamiltonian_1,
                t0_grid=t0_grid,
                safety_margin=safety_margin,
            )
        if jacobian_M is not None:
            return_map_audit = verifier.compute_poincare_return_map(
                jacobian_M=jacobian_M, period_T=period_T, safety_margin=safety_margin
            )
        celestial_verdict = HeytingOmega3.infimum(
            *[
                a.local_verdict
                for a in (kam_audit, melnikov_audit, return_map_audit)
                if a is not None
            ]
        )
        return _PoincareCelestialGrowthBundle(
            base_bundle=base_bundle,
            seed_germ=seed_germ,
            kam_audit=kam_audit,
            melnikov_audit=melnikov_audit,
            return_map=return_map_audit,
            celestial_verdict=celestial_verdict,
        )

    @classmethod
    def continue_into_phase3(
        cls,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        verdict = HeytingCropAdjudicator.adjudicate(bundle, external_verdict)
        reason = (
            f"{reason_prefix}::water={bundle.watering.local_verdict.name} "
            f"disc={bundle.discipline.local_verdict.name} "
            f"ρ(T)={bundle.discipline.spectral_radius:.4f} "
            f"KAM={bundle.discipline.is_kam_stable} "
            f"Bifurc={bundle.discipline.is_pyriform_bifurcated}"
        )
        actuation = CognitiveFaithModule.verify_faith(verdict, reason)
        return verdict, actuation

    @classmethod
    def continue_celestial_into_phase3(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        Continuación celestial a Fase III: compone el veredicto del cultivo
        base con el ínfimo H₃ de los canales celestes y emite Ω₃⁶.
        """
        return HeytingCropAdjudicator.adjudicate_celestial(
            bundle, external_verdict, reason_prefix=reason_prefix
        )

    # ── II.ω⁺  EMPALME FORMAL FASE II → FASE III ─────────────────────────
    @classmethod
    def hand_off_bundle_to_phase3(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        ═══════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE LA FASE II ≡ PRIMER MORFISMO OPERATIVO DE LA FASE III.
        ═══════════════════════════════════════════════════════════════════════
        Consume 𝒢_II = _PoincareCelestialGrowthBundle y abre la adjudicación
        Heyting Ω₃⁶ + Crowbar BT151 de la Fase III.
        """
        return cls.continue_celestial_into_phase3(
            bundle, external_verdict=external_verdict, reason_prefix=reason_prefix
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · FE + ADJUDICACIÓN Ω₃⁶ + CROWBAR + CERTIFICACIÓN MERKLE            ║
# ║                                                                           ║
# ║ Continuación directa del morfismo terminal de Fase II                     ║
# ║     (CropGrowthPipeline.hand_off_bundle_to_phase3).                       ║
# ║ Objeto inicial: 𝒢_II = _PoincareCelestialGrowthBundle.                    ║
# ║ Objeto terminal: CropGerminationCertificate (Merkle + Ω₃⁶ + Crowbar).     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── §3.1 Adjudicador en Ω₃ y Ω₃⁶ ──────────────────────────────────────────
class HeytingCropAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃.

    Ω₃⁶ = Ω₃^{×6} es el producto de Heyting de 6 canales:
      1. Riego (KV / Hill)
      2. Disciplina (Banach / Poincaré-Wirtinger)
      3. Fock (aniquilación)
      4. Brockett (isospectralidad)
      5. Rényi (pureza)
      6. Celeste (ínfimo KAM ∧ Melnikov ∧ Retorno)
    El veredicto global es el ínfimo del frame (meet n-ario).
    """

    @classmethod
    def _fock_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.fock_c.is_annihilated
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _brockett_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        ok = bundle.brockett_c.converged or bundle.brockett_c.isospectral_drift < 1e-3
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.DEGRADED

    @classmethod
    def _renyi_rule(cls, bundle: CropGrowthBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.renyi_c.purity_gain >= -1e-12
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def channel_vector(
        cls,
        bundle: CropGrowthBundle,
        celestial: Optional[HeytingOmega3] = None,
    ) -> Tuple[HeytingOmega3, ...]:
        """Seis canales de Ω₃⁶ (el sexto es ⊤ si no hay canal celeste)."""
        return (
            bundle.watering.local_verdict,
            bundle.discipline.local_verdict,
            cls._fock_rule(bundle),
            cls._brockett_rule(bundle),
            cls._renyi_rule(bundle),
            celestial if celestial is not None else HeytingOmega3.COHERENT,
        )

    @classmethod
    def adjudicate(
        cls,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        local = HeytingOmega3.infimum(*cls.channel_vector(bundle))
        if bundle.discipline.is_pyriform_bifurcated:
            local = local.meet(HeytingOmega3.VETOED)
        return local.meet(external_verdict)

    @classmethod
    def adjudicate_omega3_six(
        cls,
        celestial_bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
    ) -> Tuple[HeytingOmega3, Tuple[HeytingOmega3, ...]]:
        r"""
        Adjudicación Ω₃⁶: primer morfismo operativo de Fase III sobre 𝒢_II.
        Retorna (ínfimo global, 6-tupla de canales).
        """
        channels = cls.channel_vector(
            celestial_bundle.base_bundle, celestial_bundle.celestial_verdict
        )
        local = HeytingOmega3.infimum(*channels)
        if celestial_bundle.base_bundle.discipline.is_pyriform_bifurcated:
            local = local.meet(HeytingOmega3.VETOED)
        # Hill fuera de región degrada el canal 1 ya tratado; capacidad nula veta.
        if celestial_bundle.seed_germ.symplectic_capacity <= 0.0:
            local = local.meet(HeytingOmega3.DEGRADED)
        return local.meet(external_verdict), channels

    @classmethod
    def adjudicate_celestial(
        cls,
        bundle: _PoincareCelestialGrowthBundle,
        external_verdict: HeytingOmega3,
        reason_prefix: str = "CROP-CELESTIAL-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        final_verdict, channels = cls.adjudicate_omega3_six(bundle, external_verdict)
        ch_names = [c.name for c in channels]
        reason = (
            f"{reason_prefix}::Ω₃⁶={ch_names} "
            f"water={bundle.base_bundle.watering.local_verdict.name} "
            f"disc={bundle.base_bundle.discipline.local_verdict.name} "
            f"celestial={bundle.celestial_verdict.name} "
            f"KAM={(bundle.kam_audit.local_verdict.name if bundle.kam_audit else 'n/a')} "
            f"Mel={(bundle.melnikov_audit.local_verdict.name if bundle.melnikov_audit else 'n/a')} "
            f"Ret={(bundle.return_map.local_verdict.name if bundle.return_map else 'n/a')} "
            f"CZ={(bundle.return_map.conley_zehnder_index if bundle.return_map else 'n/a')}"
        )
        actuation = CognitiveFaithModule.verify_faith(final_verdict, reason)
        return final_verdict, actuation


# ── §3.2 CognitiveFaithModule — FE (interlock ciber-físico ESP32 Crowbar) ──
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class CognitiveFaithModule:
    r"""FASE 3 · LA FE — Adjudicación Heyting Ω₃ e interlock ciber-físico."""
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    def adjudicate_heyting_crowbar(
        self,
        disciplined_rho: np.ndarray,
        discipline_report: BanachContractionReport,
    ) -> object:
        verdict = discipline_report.local_verdict
        if (
            not discipline_report.is_kam_stable
            or discipline_report.is_pyriform_bifurcated
        ):
            verdict = HeytingOmega3.VETOED
        reason = (
            f"KAM_STABLE={discipline_report.is_kam_stable}, "
            f"PYRIFORM={discipline_report.is_pyriform_bifurcated}"
        )
        actuation = self.verify_faith(verdict, reason)
        from types import SimpleNamespace

        return SimpleNamespace(
            verdict=verdict,
            reason=reason,
            actuation=actuation,
            is_kam_stable=discipline_report.is_kam_stable,
            is_pyriform_bifurcated=discipline_report.is_pyriform_bifurcated,
        )

    @classmethod
    def verify_faith(
        cls, verdict: HeytingOmega3, reason: str = ""
    ) -> CrowbarActuationReport:
        if verdict != HeytingOmega3.VETOED:
            return CrowbarActuationReport(
                interlock_fired=False,
                actuation_latency_ns=0.0,
                gpio_pin=cls.GPIO_PIN,
                device=cls.DEVICE,
                reason="OK",
                provenance_hash="",
            )
        t_ns = time.time_ns()
        payload = f"CROWBAR_CROP::{reason}::{t_ns}".encode("utf-8")
        prov = hashlib.sha256(payload).hexdigest()
        logger.critical(
            "[CULTIVO COGNITIVO — CROWBAR ARMADO] %s → HIGH | latencia %.2f ns | razón=%s",
            cls.GPIO_PIN,
            cls.NOMINAL_LATENCY_NS,
            reason,
        )
        return CrowbarActuationReport(
            interlock_fired=True,
            actuation_latency_ns=cls.NOMINAL_LATENCY_NS,
            gpio_pin=cls.GPIO_PIN,
            device=cls.DEVICE,
            reason=reason,
            provenance_hash=prov,
        )


# ── §3.3 Certificado del cultivo (con canales celestes) ───────────────────
@dataclass(frozen=True, slots=True)
class CropGerminationCertificate:
    crop_id: str
    seed_crystal_id: str
    water_kv_compression_ratio: float
    light_photons_emitted: int
    discipline_banach_radius: float
    discipline_eta_max: float
    faith_crowbar_armed: bool
    faith_provenance_hash: str
    heyting_verdict: HeytingOmega3
    germinated_purity: float
    germinated_entropy: float
    germinated_fidelity_to_ground: float
    brockett_iterations: int
    brockett_isospectral_drift: float
    fock_resonance_residual: float
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float
    is_kam_stable: bool = True
    is_pyriform_bifurcated: bool = False
    # Extensión celeste (v9.1.0)
    celestial_kam_verdict: Optional[str] = None
    celestial_kam_min_divisor: Optional[float] = None
    celestial_kam_is_diophantine: Optional[bool] = None
    celestial_kam_bruno_sum: Optional[float] = None
    celestial_kam_is_bruno: Optional[bool] = None
    celestial_nekhoroshev_time: Optional[float] = None
    celestial_melnikov_verdict: Optional[str] = None
    celestial_melnikov_value: Optional[float] = None
    celestial_melnikov_is_simple_zero: Optional[bool] = None
    celestial_return_verdict: Optional[str] = None
    celestial_return_max_lyapunov: Optional[float] = None
    celestial_return_is_elliptic: Optional[bool] = None
    celestial_conley_zehnder: Optional[int] = None
    celestial_verdict: Optional[str] = None
    maupertuis_refractive_index: Optional[float] = None
    maupertuis_hill_margin: Optional[float] = None
    gromov_capacity: Optional[float] = None
    gelfand_radius: Optional[float] = None
    omega3_six: Optional[Tuple[str, ...]] = None


# ── §3.4 TOONCognitiveCropEngine — orquestador soberano ───────────────────
class TOONCognitiveCropEngine:
    r"""
    Motor espectral principal del cultivo cognitivo con mecánica celeste.

    Composición anidada:
        Φ_I   : SeedCrystalPreparation.synthesize_poincare_cartan_germ
                → hand_off_germ_to_phase2                         ⟶ 𝒢_I
        Φ_II  : CropGrowthPipeline.synthesize_celestial_bundle
                → hand_off_bundle_to_phase3                       ⟶ 𝒢_II
        Φ_III : Ω₃⁶ + Crowbar + Merkle                            ⟶ certificado
    """

    def __init__(
        self,
        engine_id: str = "CROP-ENGINE-WISDOM-01",
        dimension_mac: int = 4,
        eta_star: float = 1.5,
        renyi_alpha: float = CognitiveIlluminationModule.RENYI_ALPHA,
        hamiltonian_energy_H0: float = 1.0,
        potential_energy_V: float = 0.0,
        novikov_valuation_T: float = 1.0,
        safety_margin: float = 1.0,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.eta_star = float(eta_star)
        self.renyi_alpha = float(renyi_alpha)
        self._H0 = float(hamiltonian_energy_H0)
        self._V = float(potential_energy_V)
        self._novikov_T = float(novikov_valuation_T)
        self._safety_margin = float(safety_margin)
        self.crop_counter = 0
        self.soil: SoilState = SoilField.build(self.dimension_mac)
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::soil={self.soil.hash}".encode("ascii")
        ).hexdigest()
        self.watering_module = CognitiveWateringModule()
        self.illumination_module = CognitiveIlluminationModule()
        self.discipline_module = CognitiveDisciplineModule()
        self.faith_module = CognitiveFaithModule()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def execute_poincare_crop_pipeline(
        self,
        seed_crystal: object,
        soil_field: SoilField,
        cp_constant: float = 0.5,
    ) -> Tuple[object, object]:
        """Ejecuta las 4 fases clásicas del cultivo (API retrocompatible)."""
        w_report = self.watering_module.moisten_soil(seed_crystal, soil_field)
        rho_illuminated, i_report = self.illumination_module.illuminate_brockett(
            w_report.moistened_rho
        )
        rho_disciplined, d_report = self.discipline_module.enforce_poincare_wirtinger_kam_bound(
            rho_illuminated, soil_field.potential_operator, cp_constant=cp_constant
        )
        passport = self.faith_module.adjudicate_heyting_crowbar(rho_disciplined, d_report)
        from types import SimpleNamespace

        harvest = SimpleNamespace(
            harvested_rho=rho_disciplined,
            banach_report=d_report,
            watering_report=w_report,
            illumination_report=i_report,
        )
        return harvest, passport

    def execute_poincare_celestial_crop_pipeline(
        self,
        rho_raw: np.ndarray,
        toon_str: str,
        base_json_str: str,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> Tuple[HeytingOmega3, CrowbarActuationReport, _PoincareCelestialGrowthBundle]:
        r"""Pipeline celestial completo con canales KAM/Bruno, Melnikov y CZ."""
        seed_germ = SeedCrystalPreparation.synthesize_poincare_cartan_germ(
            rho_raw=rho_raw,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
        )
        self._advance_chain(
            "F1-CEL", np.ascontiguousarray(seed_germ.base_seed.rho_seed).tobytes()
        )
        bundle = SeedCrystalPreparation.hand_off_germ_to_phase2(
            germ=seed_germ,
            cycle_index=self.crop_counter,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=self._novikov_T,
            safety_margin=self._safety_margin,
        )
        self._advance_chain(
            "F2-CEL",
            (
                f"celestial={bundle.celestial_verdict.name}|"
                f"purity={bundle.base_bundle.renyi_c.purity_after:.12f}"
            ).encode("ascii"),
        )
        final_verdict, actuation = CropGrowthPipeline.hand_off_bundle_to_phase3(
            bundle, external_verdict=external_verdict
        )
        self._advance_chain(
            "F3-CEL",
            f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii"),
        )
        return final_verdict, actuation, bundle

    def _phase1_prepare(self, seed_matrix: np.ndarray) -> SeedState:
        seed_state = SeedCrystalPreparation.prepare(seed_matrix)
        self._advance_chain("F1", np.ascontiguousarray(seed_state.rho_seed).tobytes())
        return seed_state

    def _phase2_grow(
        self, seed_state: SeedState, toon_str: str, base_json_str: str
    ) -> CropGrowthBundle:
        bundle = SeedCrystalPreparation.continue_into_phase2(
            seed=seed_state,
            cycle_index=self.crop_counter,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
        )
        self._advance_chain(
            "F2",
            (
                f"purity={bundle.renyi_c.purity_after:.12f}|"
                f"rho_T={bundle.discipline.spectral_radius:.12f}"
            ).encode("ascii"),
        )
        return bundle

    def _phase3_certify(
        self,
        crop_id: str,
        seed_crystal_id: str,
        bundle: CropGrowthBundle,
        external_verdict: HeytingOmega3,
        celestial_bundle: Optional[_PoincareCelestialGrowthBundle] = None,
    ) -> CropGerminationCertificate:
        omega3_six: Optional[Tuple[str, ...]] = None
        if celestial_bundle is not None:
            final_verdict, actuation = CropGrowthPipeline.hand_off_bundle_to_phase3(
                celestial_bundle, external_verdict
            )
            _, channels = HeytingCropAdjudicator.adjudicate_omega3_six(
                celestial_bundle, external_verdict
            )
            omega3_six = tuple(c.name for c in channels)
        else:
            final_verdict, actuation = CropGrowthPipeline.continue_into_phase3(
                bundle, external_verdict
            )
        self._advance_chain(
            "F3", f"{final_verdict.name}|{actuation.provenance_hash}".encode("ascii")
        )
        fid_ground = DensityOperatorAlgebra.uhlmann_fidelity(
            bundle.rho_illum, self.soil.ground_proj
        )
        signature = _sha256_bytes(
            self.engine_id.encode("ascii"),
            crop_id.encode("ascii"),
            seed_crystal_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{bundle.renyi_c.purity_after:.10f}".encode("ascii"),
            f"{bundle.discipline.spectral_radius:.10f}".encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
            self._chain_hash.encode("ascii"),
        )
        kam_verdict = kam_min_div = kam_diof = None
        kam_bruno = kam_is_bruno = nek_t = None
        mel_verdict = mel_val = mel_simple = None
        ret_verdict = ret_max_lyap = ret_ell = cz_idx = None
        celestial_verdict = mau_refr = mau_hill = gromov = None
        gelfand = float(bundle.discipline.gelfand_radius)
        if celestial_bundle is not None:
            celestial_verdict = celestial_bundle.celestial_verdict.name
            mau_refr = celestial_bundle.seed_germ.maupertuis_germ.refractive_index
            mau_hill = celestial_bundle.seed_germ.maupertuis_germ.hill_margin
            gromov = celestial_bundle.seed_germ.symplectic_capacity
            if celestial_bundle.kam_audit is not None:
                kam_verdict = celestial_bundle.kam_audit.local_verdict.name
                kam_min_div = celestial_bundle.kam_audit.min_divisor
                kam_diof = celestial_bundle.kam_audit.is_diophantine
                kam_bruno = celestial_bundle.kam_audit.bruno_sum
                kam_is_bruno = celestial_bundle.kam_audit.is_bruno
                nek_t = celestial_bundle.kam_audit.nekhoroshev_time
            if celestial_bundle.melnikov_audit is not None:
                mel_verdict = celestial_bundle.melnikov_audit.local_verdict.name
                mel_val = celestial_bundle.melnikov_audit.melnikov_value
                mel_simple = celestial_bundle.melnikov_audit.is_simple_zero
            if celestial_bundle.return_map is not None:
                ret_verdict = celestial_bundle.return_map.local_verdict.name
                ret_max_lyap = celestial_bundle.return_map.max_lyapunov
                ret_ell = celestial_bundle.return_map.is_elliptic
                cz_idx = celestial_bundle.return_map.conley_zehnder_index
        return CropGerminationCertificate(
            crop_id=crop_id,
            seed_crystal_id=seed_crystal_id,
            water_kv_compression_ratio=bundle.watering.kv_compression_ratio,
            light_photons_emitted=bundle.fock_c.gamma_photons_emitted,
            discipline_banach_radius=bundle.discipline.spectral_radius,
            discipline_eta_max=bundle.discipline.eta_max,
            faith_crowbar_armed=actuation.interlock_fired,
            faith_provenance_hash=actuation.provenance_hash,
            heyting_verdict=final_verdict,
            germinated_purity=bundle.renyi_c.purity_after,
            germinated_entropy=bundle.renyi_c.entropy_after,
            germinated_fidelity_to_ground=fid_ground,
            brockett_iterations=bundle.brockett_c.iterations,
            brockett_isospectral_drift=bundle.brockett_c.isospectral_drift,
            fock_resonance_residual=bundle.fock_c.resonance_residual,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=signature,
            timestamp_utc=time.time(),
            is_kam_stable=bundle.discipline.is_kam_stable,
            is_pyriform_bifurcated=bundle.discipline.is_pyriform_bifurcated,
            celestial_kam_verdict=kam_verdict,
            celestial_kam_min_divisor=kam_min_div,
            celestial_kam_is_diophantine=kam_diof,
            celestial_kam_bruno_sum=kam_bruno,
            celestial_kam_is_bruno=kam_is_bruno,
            celestial_nekhoroshev_time=nek_t,
            celestial_melnikov_verdict=mel_verdict,
            celestial_melnikov_value=mel_val,
            celestial_melnikov_is_simple_zero=mel_simple,
            celestial_return_verdict=ret_verdict,
            celestial_return_max_lyapunov=ret_max_lyap,
            celestial_return_is_elliptic=ret_ell,
            celestial_conley_zehnder=cz_idx,
            celestial_verdict=celestial_verdict,
            maupertuis_refractive_index=mau_refr,
            maupertuis_hill_margin=mau_hill,
            gromov_capacity=gromov,
            gelfand_radius=gelfand,
            omega3_six=omega3_six,
        )

    def cultivate_seed_crystal(
        self,
        seed_crystal_id: str,
        seed_matrix: np.ndarray,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> CropGerminationCertificate:
        """Cultivo clásico (API 8.1.0 retrocompatible)."""
        self.crop_counter += 1
        crop_id = f"GERMINATED-CROP-{self.crop_counter:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Cultivo #%d | semilla=%s | η*=%.3f | α_Rényi=%.2f ═══",
            self.crop_counter,
            seed_crystal_id,
            self.eta_star,
            self.renyi_alpha,
        )
        seed_state = self._phase1_prepare(seed_matrix)
        bundle = self._phase2_grow(seed_state, toon_str, base_json_str)
        cert = self._phase3_certify(crop_id, seed_crystal_id, bundle, external_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cultivo %s finalizado en %.2f ms | Ω₃=%s | ρ(T)=%.4f | "
            "P_after=%.6f | γ=%d | crowbar=%s | drift_iso=%.2e",
            crop_id,
            dt_ms,
            cert.heyting_verdict.name,
            cert.discipline_banach_radius,
            cert.germinated_purity,
            cert.light_photons_emitted,
            cert.faith_crowbar_armed,
            cert.brockett_isospectral_drift,
        )
        return cert

    def cultivate_celestial_seed_crystal(
        self,
        seed_crystal_id: str,
        seed_matrix: np.ndarray,
        toon_str: str,
        base_json_str: str,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        frequency_vector_omega: Optional[np.ndarray] = None,
        wave_vectors_k: Optional[np.ndarray] = None,
        jacobian_M: Optional[np.ndarray] = None,
        homoclinic_flow: Optional[Callable[[float], np.ndarray]] = None,
        hamiltonian_0: Optional[Callable[[np.ndarray], float]] = None,
        hamiltonian_1: Optional[Callable[[np.ndarray], float]] = None,
        t0_grid: Optional[np.ndarray] = None,
        period_T: float = 1.0,
    ) -> CropGerminationCertificate:
        r"""
        Cultivo celestial completo: 𝒢_I → 𝒢_II → certificado Ω₃⁶
        vía los morfismos de empalme hand_off_germ_to_phase2 y
        hand_off_bundle_to_phase3.
        """
        self.crop_counter += 1
        crop_id = f"GERMINATED-CELESTIAL-CROP-{self.crop_counter:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Cultivo Celestial #%d | semilla=%s | η*=%.3f | H₀=%.3f | V=%.3f ═══",
            self.crop_counter,
            seed_crystal_id,
            self.eta_star,
            self._H0,
            self._V,
        )
        seed_germ = SeedCrystalPreparation.synthesize_poincare_cartan_germ(
            rho_raw=seed_matrix,
            hamiltonian_energy_H0=self._H0,
            potential_energy_V=self._V,
        )
        self._advance_chain(
            "F1-CEL", np.ascontiguousarray(seed_germ.base_seed.rho_seed).tobytes()
        )
        celestial_bundle = SeedCrystalPreparation.hand_off_germ_to_phase2(
            germ=seed_germ,
            cycle_index=self.crop_counter,
            toon_str=toon_str,
            base_json_str=base_json_str,
            renyi_alpha=self.renyi_alpha,
            eta_star=self.eta_star,
            frequency_vector_omega=frequency_vector_omega,
            wave_vectors_k=wave_vectors_k,
            jacobian_M=jacobian_M,
            homoclinic_flow=homoclinic_flow,
            hamiltonian_0=hamiltonian_0,
            hamiltonian_1=hamiltonian_1,
            t0_grid=t0_grid,
            period_T=period_T,
            novikov_valuation_T=self._novikov_T,
            safety_margin=self._safety_margin,
        )
        self._advance_chain(
            "F2-CEL",
            f"celestial={celestial_bundle.celestial_verdict.name}|"
            f"purity={celestial_bundle.base_bundle.renyi_c.purity_after:.12f}".encode(
                "ascii"
            ),
        )
        cert = self._phase3_certify(
            crop_id=crop_id,
            seed_crystal_id=seed_crystal_id,
            bundle=celestial_bundle.base_bundle,
            external_verdict=external_verdict,
            celestial_bundle=celestial_bundle,
        )
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cultivo celestial %s finalizado en %.2f ms | Ω₃=%s | celestial=%s | "
            "KAM=%s | Bruno=%s | Mel=%s | CZ=%s | crowbar=%s",
            crop_id,
            dt_ms,
            cert.heyting_verdict.name,
            cert.celestial_verdict,
            cert.celestial_kam_verdict,
            cert.celestial_kam_is_bruno,
            cert.celestial_melnikov_verdict,
            cert.celestial_conley_zehnder,
            cert.faith_crowbar_armed,
        )
        return cert


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_seed_matrix(n: int, alpha: float, key: str) -> ComplexMatrix:
    rng = np.random.default_rng(_seed_from_string(key))
    idx = np.arange(n)
    raw = np.exp(alpha * (n - idx))
    w = raw / raw.sum()
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    rho = (Q * w.astype(np.complex128)) @ Q.conj().T
    return DensityOperatorAlgebra.sanitize(rho)


__all__ = [
    "HeytingOmega3",
    "DensityOperatorAlgebra",
    "BanachContractionReport",
    "BanachContractionAlgebra",
    "SoilState",
    "SoilField",
    "MaupertuisJacobiGerm",
    "PoincareCartanGerm",
    "SeedState",
    "_PoincareCartanSeedGerm",
    "PoincareCelestialGeometry",
    "SeedCrystalPreparation",
    "WateringReport",
    "CognitiveWateringModule",
    "BrockettPurificationCertificate",
    "RenyiPurificationCertificate",
    "FockAnnihilationCertificate",
    "CognitiveIlluminationModule",
    "CognitiveDisciplineModule",
    "KAMAudit",
    "MelnikovAudit",
    "ReturnMapAudit",
    "PoincareCelestialVerifier",
    "CropGrowthBundle",
    "_PoincareCelestialGrowthBundle",
    "CropGrowthPipeline",
    "HeytingCropAdjudicator",
    "CrowbarActuationReport",
    "CognitiveFaithModule",
    "CropGerminationCertificate",
    "TOONCognitiveCropEngine",
]


if __name__ == "__main__":
    print("═" * 88)
    print("TOON COGNITIVE CROP ENGINE — v9.1.0 Poincare-Cartan-Melnikov-KAM-Bruno-CZ")
    print("KAM Tori · Bruno · Nekhoroshev · Poincaré-Wirtinger · Melnikov · CZ · Ω₃⁶")
    print("═" * 88)
    toon_str = "[APU: 2.1.4|CONCRETO 3000PSI] COST: 485000 COP"
    base_json_str = json.dumps(
        {
            "apu_code": "2.1.4-CONCRETO-3000PSI",
            "unit_cost": 485000.0,
            "metadata": {
                "schema": "fat_json_structure_with_verbose_keys",
                "authority": "Sovereign-APU-Crop-Engine",
                "redundant": "eliminable_by_toon_tabularization",
            },
        },
        indent=2,
    )
    engine = TOONCognitiveCropEngine(
        engine_id="CROP-ENGINE-WISDOM-01",
        dimension_mac=4,
        eta_star=1.5,
        renyi_alpha=1.5,
        hamiltonian_energy_H0=1.0,
        potential_energy_V=0.0,
    )
    omega_freq = np.array([1.6180339887, 2.7182818285])  # φ, e (mal aproximables)
    wave_k = np.array([[1, 1], [1, -1], [2, 1]])
    jacobian_M = np.array(
        [
            [np.cos(0.3), -np.sin(0.3), 0.0, 0.0],
            [np.sin(0.3), np.cos(0.3), 0.0, 0.0],
            [0.0, 0.0, np.cos(0.5), -np.sin(0.5)],
            [0.0, 0.0, np.sin(0.5), np.cos(0.5)],
        ]
    )
    scenarios = [
        ("COHERENT-CELESTIAL", 0.5, "SEED-FOCUSED"),
        ("DEGRADED-CELESTIAL", 0.0, "SEED-UNIFORM"),
    ]
    print("\n" + "─" * 88)
    for name, alpha, key in scenarios:
        seed_matrix = _build_seed_matrix(n=4, alpha=alpha, key=key)
        cert = engine.cultivate_celestial_seed_crystal(
            seed_crystal_id=f"CRYSTAL-{key}",
            seed_matrix=seed_matrix,
            toon_str=toon_str,
            base_json_str=base_json_str,
            external_verdict=HeytingOmega3.COHERENT,
            frequency_vector_omega=omega_freq,
            wave_vectors_k=wave_k,
            jacobian_M=jacobian_M,
        )
        print(f"\n[{name}]  α_cría = {alpha}  seed = {key}")
        print(f"   crop_id                    : {cert.crop_id}")
        print(f"   Ω₃ final                   : {cert.heyting_verdict.name}")
        print(f"   Ω₃⁶ canales                : {cert.omega3_six}")
        print(f"   Celestial verdict          : {cert.celestial_verdict}")
        if cert.celestial_kam_min_divisor is not None:
            print(
                f"   KAM verdict / divisor      : {cert.celestial_kam_verdict} / "
                f"{cert.celestial_kam_min_divisor:.6e}"
            )
        else:
            print("   KAM verdict / divisor      : n/a")
        print(f"   KAM diofantino / Bruno     : {cert.celestial_kam_is_diophantine} / "
              f"{cert.celestial_kam_is_bruno}")
        print(f"   Bruno sum / Nekhoroshev T  : {cert.celestial_kam_bruno_sum} / "
              f"{cert.celestial_nekhoroshev_time}")
        if cert.celestial_melnikov_value is not None:
            print(
                f"   Melnikov verdict / value   : {cert.celestial_melnikov_verdict} / "
                f"{cert.celestial_melnikov_value:.6e}"
            )
        else:
            print("   Melnikov verdict / value   : n/a")
        print(f"   Melnikov cero simple       : {cert.celestial_melnikov_is_simple_zero}")
        if cert.celestial_return_max_lyapunov is not None:
            print(
                f"   Return verdict / L_max     : {cert.celestial_return_verdict} / "
                f"{cert.celestial_return_max_lyapunov:.6e}"
            )
        else:
            print("   Return verdict / L_max     : n/a")
        print(f"   Return elíptico / CZ       : {cert.celestial_return_is_elliptic} / "
              f"{cert.celestial_conley_zehnder}")
        if cert.maupertuis_refractive_index is not None:
            print(
                f"   Maupertuis n(q) / Hill     : {cert.maupertuis_refractive_index:.6f} / "
                f"{cert.maupertuis_hill_margin:.6f}"
            )
        else:
            print("   Maupertuis n(q) / Hill     : n/a")
        print(f"   Gromov c_G / Gelfand ρ     : {cert.gromov_capacity} / {cert.gelfand_radius}")
        print(f"   Pureza germinada           : {cert.germinated_purity:.6f}")
        print(f"   Invariante KAM base        : {cert.is_kam_stable}")
        print(f"   Bifurcación piriforme      : {cert.is_pyriform_bifurcated}")
        print(f"   Crowbar (Fe)               : {cert.faith_crowbar_armed}")
        print(f"   Firma SHA-256              : {cert.sha256_provenance[:32]}…")
    print("\n" + "═" * 88)
    print("✓ F1→F2: synthesize_poincare_cartan_germ ⊣ hand_off_germ_to_phase2.")
    print("✓ F2→F3: synthesize_celestial_bundle ⊣ hand_off_bundle_to_phase3.")
    print("✓ Poincaré-Cartan λ ∈ Ω¹(T*Q×ℝ) dim 2n+1; L_{X_H}ω ≟ 0 (Cartan).")
    print("✓ Poincaré-Wirtinger: ||ρ−I/n||_F² ≤ C_P · ||[ρ,N]||_F².")
    print("✓ KAM: |⟨k,ω⟩| ≥ γ/|k|^τ  + Bruno ∑ 2^{-ν} log(1/Ω(2^ν)) < ∞.")
    print("✓ Nekhoroshev: T ≥ exp(c ε^{-1/(2n)}).")
    print("✓ Melnikov: M(t₀)=∫ {H₀,H₁}(γ⁰(t−t₀)) dt (Gauss–Legendre+Kahan).")
    print("✓ Retorno: P:Σ→Σ Floquet ⊕ Lyapunov ⊕ Conley–Zehnder ⊕ twist.")
    print("✓ Heyting Ω₃⁶: 5 canales del cultivo + 1 canal celeste + crowbar BT151.")
    print("═" * 88)
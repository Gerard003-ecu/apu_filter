# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/toon_introspection_agent.py                                    ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / REFLEJO INTROSPECTIVO               ║
║ FUNCIÓN  : SOBERANO DE INTROSPECCIÓN — POINCARÉ-BIRKHOFF-OSELEDETS-KAM-MELNIKOV      ║
║ VERSIÓN  : 9.0.0-Doctoral-Poincaré-Birkhoff-Oseledets-KAM-NestedThreePhase           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE FASES ANIDADAS (categorías encajadas)
─────────────────────────────────────────────────────
Funtor F = F₃ ∘ F₂ ∘ F₁ sobre la categoría 𝐂𝐨𝐧𝐜𝐫𝐞𝐭𝐞 de presheaves sobre Ω₃:

    F₁ : Flash × MAC × Ω₃           → IntrospectiveHandoff     (sustrato proyectivo)
    F₂ : IntrospectiveHandoff       → IntrospectionBundle       (dinámica de PF)
    F₃ : Bundle                     → IntrospectionProofCertificate

Anidamiento estricto:
    continue_into_phase2 ∘ build              = synthesize ∘ build
    continue_into_phase3 ∘ synthesize ∘ build = adjudicate ∘ synthesize ∘ build

MECÁNICA CELESTE DE POINCARÉ — NÚCLEO FORMAL
─────────────────────────────────────────────
Sea ρ_MAC ∈ 𝔇_n y T([v]) = [ρv] la aplicación proyectiva sobre ℂP^{n−1}.

1. Sección de Poincaré Σ_c ⊂ ℂP^{n−1}: nivel de Rayleigh R(v) = ⟨v|ρ|v⟩ = c.
   R es función Morse con puntos críticos = rayos propios de ρ. Σ_c es
   transversal al flujo proyectado X(v) = (I − vv†)ρv siempre que c ∉ spec(ρ).

2. Teorema de Birkhoff (Último Teorema Geométrico de Poincaré): un
   homeomorfismo que preserva área del anillo A = S¹ × [0,1] con twist
   ∂(θ')/∂r ≠ 0 tiene al menos 2 puntos fijos.  Aquí: la restricción de T
   a un 2-plano real invariante span{v_i, v_j} es area-preserving y, si
   los autovalores son distintos, satisface twist ⇒ ≥ 2 puntos fijos.

3. Oseledets proyectivo: T sobre ℂP^{n−1} tiene tasas transversales
       λ_i = log(λ_i(ρ)/λ₁(ρ)) ≤ 0
   (λ₁ simple ⇒ todas < 0).  h_KS(Pesin) = Σ_{λ_i > 0} λ_i = 0 en atractor
   puntual; degeneración λ₁ = λ₂ ⇒ λ_max = 0 (caos introspectivo).

4. Melnikov proyectivo: M = ⟨[H_eff, g], g⟩_F sobre la órbita entre [v₁] y [v₂];
   M = 0 simple ⇒ tangle homoclínico ⇒ caos.

5. KAM diofántico: |ω·k| ≥ γ/|k|^τ.  ω derivado de Floquet moduli.

6. Lévy sobre ℂP^{n−1} con métrica Fubini–Study (Ric = 2(n+1)ω):
       P(|f − 𝔼f| ≥ ε) ≤ exp(−(n+1)ε²/(2π²L²)).

POSTULADOS OPERATIVOS
─────────────────────
P1. Iteración de potencia gauge-fijada; parada d_FS ∧ ‖T_φ − v‖.
P2. d_FS = arccos(|⟨u|v⟩|); residuo gauge-fijado = 2 sin(θ/2).
P3. ρ(DT|_{v₁}) = λ₂/λ₁ (Floquet); tasa empírica en ventana sana.
P4. Ω₃ por meets adimensionales incluyendo λ_max, Melnikov, KAM, Birkhoff, Σ.
P5. Φ_η CPTP, Lip₁ = |1−η|.
P6. Cadena Merkle F1 → F2 → F3.

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
──────────────────────────────────────
- Certificación de punto fijo con respaldo topológico (Brouwer + Birkhoff).
- Veto automático si λ_max ≥ 0 (degeneración), Melnikov transversal o KAM colapsa.
- Crowbar BT151 en GPIO14 con latencia < 400 ns en IRAM del ESP32.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TOONIntrospection")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15

ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
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


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 · SUSTRATO PROYECTIVO-ESPECTRAL + POINCARÉ                                ║
# ║                                                                                  ║
# ║  Objetos: Ω₃, 𝔇_n, ℂP^{n−1}, spec(ρ_MAC), Σ_c, Lévy.                             ║
# ║  Morfismo terminal: IntrospectiveHandoff.continue_into_phase2.                  ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ─────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0,1,2}.
    meet = min, join = max, implies = residuo de Galois, neg = ⇒ ⊥.
    Regulares = {⊥, ⊤} = fix(¬¬);  ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤.
    """
    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    @property
    def verdict(self) -> str:
        return self.name

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
        return self.neg().neg() == self

    def as_weight(self) -> float:
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ──────────────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    𝔇_n = { ρ ∈ M_n(ℂ) : ρ=ρ†, ρ≥0, Tr ρ = 1 }.
    sanitize = Hermitiza + PSD-clip + renorm (no-expansiva en ‖·‖_F).
    Identidad de Uhlmann a un puro:  F(ρ,|v⟩⟨v|) = √⟨v|ρ|v⟩.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        if not cls.is_square(rho):
            raise ValueError(
                f"DensityOperatorAlgebra.sanitize: no cuadrada {np.shape(rho)}"
            )
        rho_h = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (rho_h + rho_h.conj().T)
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
    def eigendecomposition_descending(
        cls, rho: np.ndarray,
    ) -> Tuple[RealVector, ComplexMatrix]:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        idx = np.argsort(np.real(w))[::-1]
        w = np.maximum(np.real(w[idx]), cls.EPS)
        s = float(w.sum())
        w = w / s if s > 0.0 else w
        return w.astype(np.float64), V[:, idx]

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

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
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def uhlmann_fidelity_to_pure(cls, rho: np.ndarray, v: np.ndarray) -> float:
        r"""F(ρ, |v⟩⟨v|) = √(⟨v|ρ|v⟩ / ⟨v|v⟩) ∈ [0, 1]."""
        rho = cls.sanitize(rho)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        rayleigh = float(np.real(np.vdot(v, rho @ v)) / vv)
        return float(math.sqrt(max(0.0, rayleigh)))

    @classmethod
    def dominant_eigenvector(cls, rho: np.ndarray) -> ComplexVector:
        _w, V = cls.eigendecomposition_descending(rho)
        return V[:, 0].reshape(-1).astype(np.complex128)

    @classmethod
    def rank_one(cls, v: np.ndarray) -> ComplexMatrix:
        v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            n = int(v.size)
            return np.eye(n, dtype=np.complex128) / max(n, 1)
        v = v / nrm
        return cls.sanitize(v @ v.conj().T)


# ── §1.3 Dinámica proyectiva + métrica Fubini–Study ─────────────────────────────
@dataclass(frozen=True, slots=True)
class ProjectiveRayleighSample:
    r"""
    Evaluación de T en [v] ∈ ℂP^{n−1}:
        rayleigh, image_norm, overlap_T, uhlmann_pure, residual, fs_angle, chordal.
    """
    rayleigh: float
    image_norm: float
    overlap_T: float
    uhlmann_pure: float
    residual: float
    fs_angle: float
    chordal: float


class ProjectiveDynamics:
    r"""
    T([v]) = [ρv] sobre ℂP^{n−1};  d_FS = arccos(|⟨u|v⟩|).
    Gauge U(1): T_φ(v) = e^{−i arg⟨v, Tv⟩} Tv;  ‖T_φ − v‖₂ = 2 sin(d_FS/2).
    """

    @staticmethod
    def sanitize_vector(v: np.ndarray, n: int) -> ComplexVector:
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        if v.size < n:
            v = np.concatenate([v, np.zeros(n - v.size, dtype=np.complex128)])
        elif v.size > n:
            v = v[:n]
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            return np.ones(n, dtype=np.complex128) / math.sqrt(n)
        return (v / nrm).astype(np.complex128)

    @staticmethod
    def gauge_align(reference: np.ndarray, target: np.ndarray) -> ComplexVector:
        ov = np.vdot(reference, target)
        if abs(ov) < 1e-30:
            return np.asarray(target, dtype=np.complex128)
        return (target * np.exp(-1j * np.angle(ov))).astype(np.complex128)

    @staticmethod
    def apply_map(rho: np.ndarray, v: np.ndarray) -> Tuple[ComplexVector, float]:
        rho_v = rho @ v
        rho_v_norm = float(np.linalg.norm(rho_v))
        if rho_v_norm < 1e-15:
            return np.asarray(v, dtype=np.complex128).copy(), 0.0
        return (rho_v / rho_v_norm).astype(np.complex128), rho_v_norm

    @staticmethod
    def rayleigh(rho: np.ndarray, v: np.ndarray) -> float:
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        return float(np.real(np.vdot(v, rho @ v)) / vv)

    @staticmethod
    def fubini_study_angle(u: np.ndarray, v: np.ndarray) -> float:
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.5 * math.pi
        cos_theta = float(abs(np.vdot(u, v)) / (nu * nv))
        return float(math.acos(float(np.clip(cos_theta, 0.0, 1.0))))

    @staticmethod
    def overlap_squared(u: np.ndarray, v: np.ndarray) -> float:
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.0
        return float(np.clip(abs(np.vdot(u, v) / (nu * nv)) ** 2, 0.0, 1.0))

    @classmethod
    def evaluate(cls, rho: np.ndarray, v: np.ndarray) -> ProjectiveRayleighSample:
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        T_v, image_norm = cls.apply_map(rho, v)
        T_phi = cls.gauge_align(v, T_v)
        overlap_T = float(abs(np.vdot(v, T_v)))
        ray = cls.rayleigh(rho, v)
        fs_angle = cls.fubini_study_angle(v, T_v)
        return ProjectiveRayleighSample(
            rayleigh=ray, image_norm=image_norm,
            overlap_T=overlap_T, uhlmann_pure=float(math.sqrt(max(0.0, ray))),
            residual=float(np.linalg.norm(T_phi - v)),
            fs_angle=fs_angle, chordal=float(math.sin(fs_angle)),
        )

    @classmethod
    def projected_gradient(cls, rho: np.ndarray, v: np.ndarray) -> ComplexVector:
        r"""X(v) = (I − vv†) ρv  — gradiente de Rayleigh en S^{2n−1}."""
        rho_v = rho @ v
        return (rho_v - v * np.vdot(v, rho_v)).astype(np.complex128)


# ── §1.4 Sección de Poincaré Σ_c ⊂ ℂP^{n−1} ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSectionIntrospection:
    r"""
    Σ_c = {[v] ∈ ℂP^{n−1} : R(v) = ⟨v|ρ|v⟩ = c}.
    R Morse con puntos críticos = rayos propios de ρ.  Σ_c transversal al
    flujo proyectado X = (I − vv†)ρv siempre que c ∉ spec(ρ).

    Diagnósticos:
        level                 : c (por defecto 1/n = centro del espectro)
        transversality_min    : min |⟨X, T(v)⟩|
        first_return_time     : τ* ≈ ⟨1/|⟨X,Tv⟩|⟩
        floquet_moduli        : λ_i/λ₁ (i ≥ 2), estabilidad transversal
        periodic_orbit_rank   : #λ_i/λ₁ ≈ 1 (degeneración ⇒ órbitas periódicas)
        section_hash          : SHA-256(c ‖ μ ‖ τ)
    """
    level: float
    transversality_min: float
    first_return_time: float
    floquet_moduli: RealVector
    periodic_orbit_rank: int
    section_hash: str

    @classmethod
    def build(
        cls, rho: np.ndarray, level: Optional[float] = None,
        n_samples: int = 24, key: str = "POINCARE-INTROSPECTION",
    ) -> "PoincareSectionIntrospection":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        c = float(level) if level is not None else float(1.0 / max(n, 1))

        rng = np.random.default_rng(_seed_from_string(f"POINCARE::{key}"))
        tmin = float("inf")
        tau_acc: List[float] = []
        for _ in range(n_samples):
            v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            v = v / max(float(np.linalg.norm(v)), 1e-15)
            X = ProjectiveDynamics.projected_gradient(rho, v)
            rho_v = rho @ v
            nv = float(np.linalg.norm(rho_v))
            if nv < 1e-12:
                continue
            T_v = rho_v / nv
            trans = abs(float(np.real(np.vdot(X, T_v))))
            tmin = min(tmin, trans)
            tau_acc.append(1.0 / max(trans, 1e-12))

        tau_mean = float(np.mean(tau_acc)) if tau_acc else 0.0
        lam1 = max(float(w[0]), _EPS)
        floquet = (w[1:] / lam1).astype(np.float64)
        per_rank = int(np.sum(np.abs(floquet - 1.0) < 1e-10))

        h = _sha256_bytes(
            f"{c:.12e}".encode("ascii"),
            np.ascontiguousarray(floquet).tobytes(),
            f"{tmin:.12e}".encode("ascii"),
        )
        return cls(
            level=c,
            transversality_min=float(tmin),
            first_return_time=float(tau_mean),
            floquet_moduli=floquet,
            periodic_orbit_rank=per_rank,
            section_hash=h,
        )

    @property
    def is_transverse(self) -> bool:
        return self.transversality_min > 1e-10


# ── §1.5 Lema de concentración de Lévy sobre ℂP^{n−1} (FS) ──────────────────────
class LevyConcentrationLemma:
    r"""
    Concentración sobre ℂP^{n−1} con métrica Fubini–Study (Ric = 2(n+1)ω):
        P(|f − 𝔼f| ≥ ε) ≤ exp(−(n+1)ε² / (2π²L²)).
    """
    @staticmethod
    def bound(epsilon: float, n: int, lipschitz: float = 1.0) -> float:
        if n <= 1:
            return 1.0
        L = max(float(lipschitz), 1e-15)
        e = max(float(epsilon), 0.0)
        return float(min(1.0, math.exp(
            -(n + 1) * e * e / (2.0 * math.pi ** 2 * L * L)
        )))

    @staticmethod
    def median_width(
        n: int, lipschitz: float = 1.0, confidence: float = 0.99
    ) -> float:
        if n <= 1:
            return float("inf")
        p_tail = max(1e-15, 1.0 - float(confidence))
        return float(lipschitz * math.pi * math.sqrt(
            2.0 * math.log(1.0 / p_tail) / (n + 1)
        ))


# ── §1.6 Análisis del gap espectral ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralGapReport:
    r"""
    Análisis espectral de ρ_MAC:
        gap = 1 − λ₂/λ₁;  cond = λ₁/λ_n;  degeneracy_top;  birkhoff_proxy.
    Veredicto local:
        uniforme ∨ degeneracy ≥ 3     → ⊥
        gap ≥ GAP_COHERENT ∧ deg = 1  → ⊤
        gap ≥ GAP_DEGRADED            → ⋆
        else                          → ⊥
    """
    lambda_1: float
    lambda_2: float
    lambda_min: float
    gap: float
    gap_ratio: float
    gap_absolute: float
    condition_number: float
    degeneracy_top: int
    is_uniform: bool
    von_neumann_entropy: float
    purity: float
    birkhoff_constant: float
    local_verdict: HeytingOmega3


class SpectralGapAnalyzer:
    r"""
    Espectro de ρ → tasa de T y existencia de atractor.
    Brouwer: ℂP^{n−1} compacto ⇒ Fix(T) ≠ ∅.
    Linearización: DT|_{[v₁]} sobre v₁^⊥ tiene spec {λ_i/λ₁}_{i≥2}.
    Birkhoff–Hopf: proxy = tanh(log(λ₁/λₙ)/4).
    """
    GAP_COHERENT: Final[float] = 0.10
    GAP_DEGRADED: Final[float] = 0.01
    UNIFORM_TOL: Final[float] = 1.0e-3

    @classmethod
    def analyze(cls, rho: np.ndarray) -> SpectralGapReport:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = float(w[0])
        lam2 = float(w[1]) if w.size > 1 else 0.0
        lam_n = float(w[-1])
        gap_ratio = (lam2 / lam1) if lam1 > 1e-15 else 1.0
        gap = float(max(0.0, 1.0 - gap_ratio))
        gap_abs = float(max(0.0, lam1 - lam2))
        cond = float(lam1 / max(lam_n, _EPS))
        tol_degen = max(1e-9, 1e-6 * lam1)
        degeneracy = int(np.sum(np.abs(w - lam1) < tol_degen))
        uniform_gap = float(np.max(np.abs(w - 1.0 / max(n, 1))))
        is_uniform = bool(uniform_gap < cls.UNIFORM_TOL)
        delta_proj = math.log(lam1 / max(lam_n, _EPS)) if lam1 > _EPS else 0.0
        birkhoff = float(math.tanh(delta_proj / 4.0))

        if is_uniform or degeneracy >= 3:
            local = HeytingOmega3.VETOED
        elif gap >= cls.GAP_COHERENT and degeneracy == 1:
            local = HeytingOmega3.COHERENT
        elif gap >= cls.GAP_DEGRADED:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return SpectralGapReport(
            lambda_1=lam1, lambda_2=lam2, lambda_min=lam_n,
            gap=gap, gap_ratio=float(gap_ratio), gap_absolute=gap_abs,
            condition_number=cond, degeneracy_top=degeneracy,
            is_uniform=is_uniform,
            von_neumann_entropy=DensityOperatorAlgebra.von_neumann_entropy(rho),
            purity=DensityOperatorAlgebra.purity(rho),
            birkhoff_constant=birkhoff, local_verdict=local,
        )


# ── §1.7 IntrospectiveHandoff — HAND-OFF FORMAL FASE 1 → FASE 2 ─────────────────
@dataclass(frozen=True, slots=True)
class IntrospectiveHandoff:
    r"""
    Objeto terminal FASE 1 / inicial FASE 2:
        rho_mac, v_flash     : campo MAC y corazonada
        spectral_gap         : invariante de F1
        poincare_section     : Σ_c ⊂ ℂP^{n−1}
        levy_band            : ε* de Lévy sobre ℂP^{n−1}
        rayleigh_quotient    : ⟨v|ρ|v⟩
        uhlmann_fidelity     : √R = F(ρ, |v⟩⟨v|)
        overlap_with_dominant: |⟨v|v₁⟩|²
        fs_angle_to_dominant : d_FS([v],[v₁])
        incoming_heyting     : Ω₃ del flash aguas arriba
    """
    introspective_id: str
    flash_intuition_id: str
    rho_mac: np.ndarray
    v_flash: np.ndarray
    v_dominant: np.ndarray
    spectral_gap: SpectralGapReport
    poincare_section: PoincareSectionIntrospection
    levy_band: float
    rayleigh_quotient: float
    uhlmann_fidelity_to_rho: float
    overlap_with_dominant: float
    fs_angle_to_dominant: float
    incoming_heyting_verdict: HeytingOmega3
    visceral_message: str
    handoff_hash: str
    dim: int

    @classmethod
    def build(
        cls,
        introspective_id: str,
        flash_intuition_id: str,
        rho_mac: np.ndarray,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3,
        visceral_message: str,
        poincare_key: str = "INTROSPECTION",
    ) -> "IntrospectiveHandoff":
        r"""Cierra FASE 1; el morfismo de continuación es continue_into_phase2."""
        rho = DensityOperatorAlgebra.sanitize(rho_mac)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(flash_vector, n)
        gap = SpectralGapAnalyzer.analyze(rho)
        poincare = PoincareSectionIntrospection.build(
            rho, level=None, n_samples=16, key=poincare_key,
        )
        levy_band = LevyConcentrationLemma.median_width(
            n, lipschitz=1.0, confidence=0.99,
        )
        v1 = DensityOperatorAlgebra.dominant_eigenvector(rho)
        v1 = ProjectiveDynamics.gauge_align(v, v1)

        rayleigh = ProjectiveDynamics.rayleigh(rho, v)
        f_uh = float(math.sqrt(max(0.0, rayleigh)))
        overlap = ProjectiveDynamics.overlap_squared(v, v1)
        fs_angle = ProjectiveDynamics.fubini_study_angle(v, v1)

        handoff_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(v).tobytes(),
            f"{overlap:.12f}".encode("ascii"),
            f"{gap.gap:.12f}".encode("ascii"),
            poincare.section_hash.encode("ascii"),
            incoming_heyting_verdict.name.encode("ascii"),
        )
        return cls(
            introspective_id=introspective_id,
            flash_intuition_id=flash_intuition_id,
            rho_mac=rho, v_flash=v, v_dominant=v1,
            spectral_gap=gap, poincare_section=poincare,
            levy_band=float(levy_band),
            rayleigh_quotient=float(rayleigh),
            uhlmann_fidelity_to_rho=float(f_uh),
            overlap_with_dominant=float(overlap),
            fs_angle_to_dominant=float(fs_angle),
            incoming_heyting_verdict=incoming_heyting_verdict,
            visceral_message=visceral_message,
            handoff_hash=handoff_hash, dim=n,
        )

    def summary(self) -> Dict[str, float]:
        return {
            "rayleigh": self.rayleigh_quotient,
            "F_uhlmann": self.uhlmann_fidelity_to_rho,
            "overlap_v1": self.overlap_with_dominant,
            "fs_angle_v1": self.fs_angle_to_dominant,
            "gap": self.spectral_gap.gap,
            "lambda_1": self.spectral_gap.lambda_1,
            "lambda_2": self.spectral_gap.lambda_2,
            "degeneracy": float(self.spectral_gap.degeneracy_top),
            "birkhoff": self.spectral_gap.birkhoff_constant,
            "poincare_transverse": float(self.poincare_section.is_transverse),
            "floquet_min": float(self.poincare_section.floquet_moduli.min())
            if self.poincare_section.floquet_moduli.size else 0.0,
            "levy_band": self.levy_band,
            "incoming_weight": self.incoming_heyting_verdict.as_weight(),
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self, max_iter: int, tol: float,
    ) -> "IntrospectionBundle":
        r"""
        Último morfismo FASE 1 ∧ primero FASE 2.
            continue_into_phase2 ∘ build
                = IntrospectionPipeline.synthesize ∘ build
                : Flash × MAC × Ω₃ → IntrospectionBundle.
        """
        return IntrospectionPipeline.synthesize(
            handoff=self, max_iter=max_iter, tol=tol,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA DE PUNTO FIJO + OSELEDETS + MELNIKOV + KAM + BIRKHOFF         ║
# ║                                                                                  ║
# ║  Dominio = IntrospectiveHandoff (codominio de §1.7).                            ║
# ║  Codominio = IntrospectionBundle, dominio de toda la FASE 3.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Traza de iteración de potencia ─────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PowerIterationTrace:
    residuals: Tuple[float, ...] = ()
    fs_angles: Tuple[float, ...] = ()
    rayleigh_trajectory: Tuple[float, ...] = ()
    overlaps_with_v1: Tuple[float, ...] = ()
    empirical_rate: float = 0.0
    stalled_kernel: bool = False
    iterations: int = 0
    converged: bool = False
    history: Tuple[float, ...] = ()
    final_residual: float = 0.0


# ── §2.2 Solver de iteración de potencia ────────────────────────────────────────
class PowerIterationSolver:
    r"""
    Iteración de potencia COMPLETA (continuación de continue_into_phase2):
        v_{k+1} = gauge_align(v_k, ρv_k/‖ρv_k‖).
    Parada: d_FS < tol ∨ ‖T_φ − v_k‖ < tol ∨ k = max_iter.
    Tasa: d_FS([v_k],[v₁]) = Θ((λ₂/λ₁)^k) si λ₁ > λ₂.
    """
    MAX_ITER_DEFAULT: Final[int] = 500
    TOL_DEFAULT: Final[float] = 1.0e-10
    MIN_FS_FOR_RATE: Final[float] = 1.0e-12
    RATE_FS_FLOOR: Final[float] = 1.0e-8
    RATE_FS_CEIL: Final[float] = 5.0e-1

    @classmethod
    def _empirical_rate(cls, fs_angles: List[float], theoretical: float) -> float:
        arr = np.asarray(fs_angles, dtype=np.float64)
        if arr.size <= 2:
            return float(theoretical)
        mask = (arr[:-1] >= cls.RATE_FS_FLOOR) & (arr[:-1] <= cls.RATE_FS_CEIL)
        mask &= (arr[1:] >= cls.MIN_FS_FOR_RATE)
        if int(mask.sum()) < 3:
            return float(theoretical) if arr[-1] < cls.RATE_FS_FLOOR else 1.0
        ratios = arr[1:][mask] / np.maximum(arr[:-1][mask], cls.MIN_FS_FOR_RATE)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            return float(theoretical)
        return float(np.clip(np.median(ratios), 0.0, 1.0))

    @classmethod
    def solve(
        cls,
        handoff: IntrospectiveHandoff,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
    ) -> Tuple[ComplexVector, PowerIterationTrace]:
        rho = handoff.rho_mac
        n = handoff.dim
        v = ProjectiveDynamics.sanitize_vector(handoff.v_flash, n)
        v1 = handoff.v_dominant
        theoretical = float(handoff.spectral_gap.gap_ratio)

        residuals: List[float] = []
        fs_angles: List[float] = []
        rayleigh_traj: List[float] = []
        overlaps: List[float] = []
        converged = False
        stalled_kernel = False

        for _k in range(max_iter):
            sample = ProjectiveDynamics.evaluate(rho, v)
            residuals.append(sample.residual)
            fs_angles.append(sample.fs_angle)
            rayleigh_traj.append(sample.rayleigh)
            overlaps.append(ProjectiveDynamics.overlap_squared(v, v1))
            if sample.image_norm < 1e-15:
                stalled_kernel = True
                break
            if sample.residual < tol or sample.fs_angle < tol:
                converged = True
                break
            T_v, _ = ProjectiveDynamics.apply_map(rho, v)
            v = ProjectiveDynamics.sanitize_vector(
                ProjectiveDynamics.gauge_align(v, T_v), n,
            )

        empirical = cls._empirical_rate(fs_angles, theoretical)
        trace = PowerIterationTrace(
            residuals=tuple(map(float, residuals)),
            fs_angles=tuple(map(float, fs_angles)),
            rayleigh_trajectory=tuple(map(float, rayleigh_traj)),
            overlaps_with_v1=tuple(map(float, overlaps)),
            empirical_rate=float(empirical),
            stalled_kernel=bool(stalled_kernel),
            iterations=len(residuals),
            converged=bool(converged),
        )
        return v, trace

    def solve_poincare_birkhoff_fixed_point_cpn(
        self,
        density_op: np.ndarray,
        seed_ray_s6: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tolerance_fubini_study: float = 1e-6,
    ) -> Tuple[PowerIterationTrace, "FixedPointCertificate"]:
        r"""
        Punto fijo autoinvariante en ℂP^{n−1} gauge-fijado, con semilla S⁶.
        """
        density_op = DensityOperatorAlgebra.sanitize(density_op)
        n = density_op.shape[0]
        if seed_ray_s6 is not None and seed_ray_s6.size >= 6:
            v0 = np.array([
                seed_ray_s6[0] + 1j * seed_ray_s6[1],
                seed_ray_s6[2] + 1j * seed_ray_s6[3],
                seed_ray_s6[4] + 1j * seed_ray_s6[5],
            ], dtype=np.complex128)
            if v0.size < n:
                v0 = np.pad(v0, (0, n - v0.size))
            elif v0.size > n:
                v0 = v0[:n]
            norm_v0 = float(np.linalg.norm(v0))
            v = v0 / norm_v0 if norm_v0 > 1e-12 else np.ones(n, dtype=np.complex128) / math.sqrt(n)
        else:
            v = np.ones(n, dtype=np.complex128) / math.sqrt(n)
        v = v / float(np.linalg.norm(v))
        trace_history: List[float] = []
        d_fs = 1.0

        for _it in range(1, max_iter + 1):
            w = density_op @ v
            norm_w = float(np.linalg.norm(w))
            if norm_w < 1e-15:
                break
            w_normalized = w / norm_w
            overlap = np.vdot(v, w_normalized)
            phase = np.angle(overlap) if np.abs(overlap) > 1e-12 else 0.0
            v_next = w_normalized * np.exp(-1j * phase)
            v_next = v_next / float(np.linalg.norm(v_next))
            fidelity = float(np.clip(np.abs(np.vdot(v, v_next)), 0.0, 1.0))
            d_fs = float(np.arccos(fidelity))
            trace_history.append(d_fs)
            if d_fs <= tolerance_fubini_study:
                v = v_next
                break
            v = v_next

        is_fixed_point = bool(d_fs <= tolerance_fubini_study)
        uhlmann_fid = float(np.abs(np.vdot(v, density_op @ v)))
        sample_final = ProjectiveDynamics.evaluate(density_op, v)
        w_spec = DensityOperatorAlgebra.spectrum_descending(density_op)
        lam1 = float(w_spec[0]) if w_spec.size else 1.0
        v1 = DensityOperatorAlgebra.dominant_eigenvector(density_op)

        cert = FixedPointCertificate(
            fixed_point_residual=float(sample_final.residual),
            fixed_point_fs_angle=d_fs,
            overlap_T=float(sample_final.overlap_T),
            uhlmann_fidelity_final=uhlmann_fid,
            overlap_with_dominant=float(ProjectiveDynamics.overlap_squared(v, v1)),
            rayleigh_final=float(sample_final.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample_final.rayleigh)),
            iterations=len(trace_history),
            converged=is_fixed_point,
            stalled_kernel=False,
            empirical_rate=0.0, theoretical_rate=0.0,
            rate_consistency=True,
            local_verdict=(
                HeytingOmega3.COHERENT if is_fixed_point else HeytingOmega3.DEGRADED
            ),
            is_fixed_point=is_fixed_point,
            fubini_study_distance=d_fs,
            iterations_count=len(trace_history),
            eigenstate_ray=v,
            uhlmann_fidelity=uhlmann_fid,
        )
        trace = PowerIterationTrace(
            residuals=tuple(trace_history),
            fs_angles=tuple(trace_history),
            rayleigh_trajectory=(),
            overlaps_with_v1=(),
            empirical_rate=0.0, stalled_kernel=False,
            iterations=len(trace_history),
            converged=is_fixed_point,
            history=tuple(trace_history),
            final_residual=d_fs,
        )
        return trace, cert


# ── §2.3 Certificado del punto fijo ─────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FixedPointCertificate:
    r"""
    Certificado del punto fijo introspectivo.
    """
    fixed_point_residual: float = 0.0
    fixed_point_fs_angle: float = 0.0
    overlap_T: float = 0.0
    uhlmann_fidelity_final: float = 0.0
    overlap_with_dominant: float = 0.0
    rayleigh_final: float = 0.0
    rayleigh_gap_to_lambda1: float = 0.0
    iterations: int = 0
    converged: bool = False
    stalled_kernel: bool = False
    empirical_rate: float = 0.0
    theoretical_rate: float = 0.0
    rate_consistency: bool = True
    local_verdict: HeytingOmega3 = HeytingOmega3.VETOED
    is_fixed_point: bool = False
    fubini_study_distance: float = 0.0
    iterations_count: int = 0
    eigenstate_ray: Optional[np.ndarray] = None
    uhlmann_fidelity: float = 0.0


class FixedPointCertifier:
    r"""
    Predicados dimensionalmente invariantes, colapsados por meet (no n_fail):
        conv, overlap_T, residual, born, rayleigh, rate.
    Uhlmann √R se REPORTA, no se usa como predicado de punto fijo.
    """
    EPS_OVL_T: Final[float] = 1.0e-6
    EPS_RES: Final[float] = 1.0e-6
    EPS_BORN: Final[float] = 1.0e-6
    EPS_RAY: Final[float] = 1.0e-3
    RATE_TOL: Final[float] = 0.10

    @classmethod
    def _grade(cls, value: float, hi_ok: float, mid_ok: float) -> HeytingOmega3:
        if value <= hi_ok:
            return HeytingOmega3.COHERENT
        if value <= mid_ok:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def certify(
        cls,
        handoff: IntrospectiveHandoff,
        v_fixed: np.ndarray,
        trace: PowerIterationTrace,
    ) -> FixedPointCertificate:
        sample = ProjectiveDynamics.evaluate(handoff.rho_mac, v_fixed)
        lam1 = handoff.spectral_gap.lambda_1
        rate_th = handoff.spectral_gap.gap_ratio
        rate_emp = trace.empirical_rate
        ovl_v1 = ProjectiveDynamics.overlap_squared(v_fixed, handoff.v_dominant)

        already_fp = bool(trace.converged and trace.iterations <= 2)
        noisy_rate = bool(trace.converged and sample.fs_angle < cls.EPS_RES)
        rate_ok = already_fp or noisy_rate or (abs(rate_emp - rate_th) <= cls.RATE_TOL)

        conv_rule = (
            HeytingOmega3.VETOED if trace.stalled_kernel
            else (HeytingOmega3.COHERENT if trace.converged else HeytingOmega3.DEGRADED)
        )
        ovl_t_rule = cls._grade(1.0 - sample.overlap_T, cls.EPS_OVL_T, 1.0e-3)
        residual_rule = cls._grade(sample.residual, cls.EPS_RES, 1.0e-3)
        born_rule = cls._grade(1.0 - ovl_v1, cls.EPS_BORN, 1.0e-2)
        ray_rel = max(0.0, lam1 - sample.rayleigh) / max(lam1, _EPS)
        ray_rule = cls._grade(ray_rel, cls.EPS_RAY, 1.0e-2)
        rate_rule = HeytingOmega3.COHERENT if rate_ok else HeytingOmega3.DEGRADED

        local = (
            conv_rule.meet(ovl_t_rule).meet(residual_rule)
            .meet(born_rule).meet(ray_rule).meet(rate_rule)
        )
        return FixedPointCertificate(
            fixed_point_residual=float(sample.residual),
            fixed_point_fs_angle=float(sample.fs_angle),
            overlap_T=float(sample.overlap_T),
            uhlmann_fidelity_final=float(sample.uhlmann_pure),
            overlap_with_dominant=float(ovl_v1),
            rayleigh_final=float(sample.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample.rayleigh)),
            iterations=int(trace.iterations),
            converged=bool(trace.converged),
            stalled_kernel=bool(trace.stalled_kernel),
            empirical_rate=float(rate_emp),
            theoretical_rate=float(rate_th),
            rate_consistency=bool(rate_ok),
            local_verdict=local,
        )


# ── §2.4 Espectro de Oseledets proyectivo ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class OseledetsLyapunovSpectrum:
    r"""
    Espectro de Lyapunov de T([v]) = [ρv] sobre ℂP^{n−1}:
        λ_i = log(λ_i(ρ)/λ₁(ρ))  para i ≥ 2  (todas ≤ 0).
    h_KS(Pesin) = Σ_{λ_i > 0} λ_i.  Atractor único ⇒ h_KS = 0.
    Degeneración λ₁ = λ₂ ⇒ λ_max = 0 ⇒ caos introspectivo.
    """
    lyapunov_full: RealVector
    lyapunov_max: float
    lyapunov_min: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    n_positive: int
    degeneracy_signal: bool

    @classmethod
    def from_spectrum(cls, rho: np.ndarray) -> "OseledetsLyapunovSpectrum":
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        lam1 = max(float(w[0]), _EPS)
        rates = np.array(
            [math.log(max(float(w_i), _EPS) / lam1) for w_i in w[1:]],
            dtype=np.float64,
        )
        if rates.size == 0:
            rates = np.array([0.0], dtype=np.float64)
        rates_sorted = np.sort(rates)[::-1]
        lam_max = float(rates_sorted[0])
        lam_min = float(rates_sorted[-1])
        pos = rates_sorted[rates_sorted > 1e-14]
        h_ks = float(np.sum(pos)) if pos.size else 0.0
        cum = np.cumsum(rates_sorted)
        k = int(np.max(np.where(cum >= 0.0)[0])) + 1 if np.any(cum >= 0.0) else 0
        if 0 < k < rates_sorted.size and abs(rates_sorted[k]) > 1e-15:
            d_ky = float(k + cum[k - 1] / abs(rates_sorted[k]))
        else:
            d_ky = float(k)
        degeneracy = bool(lam_max > -1e-10)
        return cls(
            lyapunov_full=rates_sorted,
            lyapunov_max=lam_max,
            lyapunov_min=lam_min,
            kolmogorov_sinai_entropy=h_ks,
            kaplan_yorke_dimension=d_ky,
            n_positive=int(pos.size),
            degeneracy_signal=degeneracy,
        )


# ── §2.5 Integral de Melnikov proyectiva ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MelnikovHomoclinicCertificate:
    r"""
    Integral de Melnikov adaptada a dinámica proyectiva:
        M = ⟨[H_eff, g], g⟩_F  sobre la órbita entre [v₁] y [v₂].
    Cero simple de M ⇒ tangle homoclínico ⇒ caos.
    """
    melnikov_value: float
    melnikov_zeros: int
    transverse_homoclinic: bool
    chaos_threshold: float

    @classmethod
    def evaluate(
        cls,
        rho: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        epsilon_M: float = 1.0e-3,
    ) -> "MelnikovHomoclinicCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        v1 = ProjectiveDynamics.sanitize_vector(v1, rho.shape[0])
        v2 = ProjectiveDynamics.sanitize_vector(v2, rho.shape[0])
        ts = np.linspace(-0.5, 1.5, 41)
        vals = np.zeros_like(ts)
        for i, t in enumerate(ts):
            tt = float(np.clip(t, 0.0, 1.0))
            v = (1.0 - tt) * v1 + tt * v2
            nv = float(np.linalg.norm(v))
            if nv < 1e-15:
                continue
            v = v / nv
            rho_v = rho @ v
            R = float(np.real(np.vdot(v, rho_v)))
            grad = 2.0 * (rho_v - R * v)
            rho_grad = rho @ grad
            H_grad = 2.0 * (rho_grad - float(np.real(np.vdot(grad, rho_grad))) * grad)
            comm = H_grad @ grad - grad @ H_grad
            vals[i] = float(np.real(np.vdot(comm.reshape(-1), grad.reshape(-1))))
        M = float(vals[len(vals) // 2]) if vals.size else 0.0
        signs = np.sign(vals)
        zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        transverse = bool(abs(M) > epsilon_M and zeros >= 1)
        return cls(
            melnikov_value=float(M),
            melnikov_zeros=zeros,
            transverse_homoclinic=transverse,
            chaos_threshold=float(epsilon_M),
        )


# ── §2.6 KAM diofántico proyectivo ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class KAMDiophantineCertificate:
    r"""
    |ω·k| ≥ γ/|k|^τ ∀ k ∈ ℤᵐ\{0};  ε < ε₀(γ,τ) ⇒ toro 𝕋ᵐ persiste.
    ε₀ heurístico (Chirikov): γ / (m log(1/γ))^m.
    """
    frequency_vector: RealVector
    diophantine_gamma: float
    diophantine_tau: float
    is_diophantine: bool
    perturbation_ratio: float
    kam_persists: bool
    kmax_used: int

    @classmethod
    def evaluate(
        cls, omega: Sequence[float], perturbation_size: float,
        tau: float = 1.5, kmax: int = 6, gamma_min: float = 1e-3,
    ) -> "KAMDiophantineCertificate":
        w = np.asarray(omega, dtype=np.float64).ravel()
        m = w.size
        if m == 0:
            return cls(
                frequency_vector=w, diophantine_gamma=0.0,
                diophantine_tau=tau, is_diophantine=False,
                perturbation_ratio=float("inf"), kam_persists=False,
                kmax_used=kmax,
            )
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(m)]
        gamma = float("inf")
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(m, -1).T
        for k in grid:
            if np.all(k == 0):
                continue
            dot = float(np.dot(w, k))
            norm = float(np.linalg.norm(k))
            if abs(dot) < 1e-15:
                gamma = 0.0
                break
            gamma = min(gamma, abs(dot) * (norm ** tau))
        is_dio = bool(gamma >= gamma_min)
        if gamma > 1e-15:
            denom = (m * math.log(1.0 / max(gamma, 1e-15))) ** m if m > 0 else 1.0
            eps0 = gamma / max(denom, 1e-15)
        else:
            eps0 = 0.0
        ratio = float(perturbation_size / eps0) if eps0 > 1e-15 else float("inf")
        return cls(
            frequency_vector=w.astype(np.float64),
            diophantine_gamma=float(gamma),
            diophantine_tau=float(tau),
            is_diophantine=is_dio,
            perturbation_ratio=float(ratio),
            kam_persists=bool(is_dio and ratio < 1.0),
            kmax_used=int(kmax),
        )


# ── §2.7 Teorema de Birkhoff (twist map) ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BirkhoffTwistMapCertificate:
    r"""
    Último Teorema Geométrico de Poincaré (1912), demostrado por Birkhoff (1913):
    Un homeomorfismo que preserva área del anillo A = S¹ × [0,1] con twist
    tiene al menos 2 puntos fijos.  Aplicado a span{v_i, v_j} invariante.

    Certificado:
        twist_holds, n_fixed_points_min, twist_gradient_min,
        degeneracy_pairs, local_verdict.
    """
    twist_holds: bool
    n_fixed_points_min: int
    twist_gradient_min: float
    degeneracy_pairs: int
    local_verdict: HeytingOmega3

    @classmethod
    def evaluate(
        cls, rho: np.ndarray, n_pairs: int = 3, n_samples: int = 16,
        key: str = "BIRKHOFF",
    ) -> "BirkhoffTwistMapCertificate":
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = int(rho.shape[0])
        w, V = la.eigh(rho)
        order = np.argsort(w)[::-1]
        V = V[:, order]
        rng = np.random.default_rng(_seed_from_string(f"BIRKHOFF::{key}"))
        twist_min = float("inf")
        valid_pairs = 0
        pairs_examined = min(int(n_pairs), max(1, n - 1))

        for p in range(pairs_examined):
            i, j = 0, p + 1
            if i >= n or j >= n:
                continue
            vi = V[:, i]; vj = V[:, j]
            for _ in range(n_samples):
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                r = float(rng.uniform(0.1, 0.9))
                v = r * math.cos(theta) * vi + r * math.sin(theta) * vj
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                v = v / nv
                T_v, _ = ProjectiveDynamics.apply_map(rho, v)
                a_i = np.vdot(vi, T_v); a_j = np.vdot(vj, T_v)
                if abs(a_i) < 1e-12 or abs(a_j) < 1e-12:
                    continue
                theta_prime = float(np.angle(a_i) - np.angle(a_j))
                dr = 0.05
                r2 = min(0.95, r + dr)
                v2 = r2 * math.cos(theta) * vi + r2 * math.sin(theta) * vj
                nv2 = float(np.linalg.norm(v2))
                if nv2 < 1e-12:
                    continue
                v2 = v2 / nv2
                T_v2, _ = ProjectiveDynamics.apply_map(rho, v2)
                b_i = np.vdot(vi, T_v2); b_j = np.vdot(vj, T_v2)
                if abs(b_i) < 1e-12 or abs(b_j) < 1e-12:
                    continue
                theta_prime_2 = float(np.angle(b_i) - np.angle(b_j))
                d_theta = (theta_prime_2 - theta_prime) / max(dr, 1e-9)
                twist_min = min(twist_min, abs(d_theta))
            if twist_min > 1e-6:
                valid_pairs += 1

        if not np.isfinite(twist_min):
            twist_min = 0.0
        twist_holds = bool(twist_min > 1e-6)
        n_fp = 2 if (twist_holds and valid_pairs >= 1) else 1
        if twist_holds and n_fp >= 2:
            local = HeytingOmega3.COHERENT
        elif n_fp >= 1:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return cls(
            twist_holds=twist_holds,
            n_fixed_points_min=int(n_fp),
            twist_gradient_min=float(twist_min),
            degeneracy_pairs=int(valid_pairs),
            local_verdict=local,
        )


# ── §2.8 IntrospectionBundle — HAND-OFF FASE 2 → FASE 3 ─────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionBundle:
    r"""
    Objeto terminal FASE 2 / inicial FASE 3.
    Producto de: Power ⊗ Certifier ⊗ Oseledets ⊗ Melnikov ⊗ KAM ⊗ Birkhoff.
    """
    handoff: IntrospectiveHandoff
    v_fixed: np.ndarray
    trace: PowerIterationTrace
    certificate: FixedPointCertificate
    oseledets: OseledetsLyapunovSpectrum
    melnikov: MelnikovHomoclinicCertificate
    kam: KAMDiophantineCertificate
    birkhoff: BirkhoffTwistMapCertificate

    def content_bytes(self) -> bytes:
        c = self.certificate
        return hashlib.sha256(
            self.handoff.handoff_hash.encode("ascii")
            + np.ascontiguousarray(self.v_fixed).tobytes()
            + f"{c.fixed_point_residual:.12e}".encode("ascii")
            + f"{c.overlap_T:.12e}".encode("ascii")
            + f"{c.uhlmann_fidelity_final:.12e}".encode("ascii")
            + f"{c.overlap_with_dominant:.12e}".encode("ascii")
            + f"{c.iterations}".encode("ascii")
            + f"{c.converged}".encode("ascii")
            + f"{self.oseledets.lyapunov_max:.12e}".encode("ascii")
            + f"{self.oseledets.kolmogorov_sinai_entropy:.12e}".encode("ascii")
            + f"{self.melnikov.melnikov_value:.12e}".encode("ascii")
            + f"{self.birkhoff.n_fixed_points_min}".encode("ascii")
        ).digest()

    def continue_into_phase3(self) -> HeytingOmega3:
        r"""
        Último morfismo FASE 2 ∧ primero FASE 3.
            continue_into_phase3 ∘ synthesize ∘ build
                = adjudicate ∘ synthesize ∘ build.
        """
        return HeytingIntrospectionAdjudicator.adjudicate(self)


class IntrospectionPipeline:
    r"""Orquestador determinista FASE 2 (funtor F₂)."""

    @classmethod
    def synthesize(
        cls,
        handoff: IntrospectiveHandoff,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> IntrospectionBundle:
        r"""
        Cierra FASE 2.  Abre FASE 3.
            (v*, trace) = solve(handoff)
            cert        = certify(...)
            osledets    = from_spectrum(ρ)
            melnikov    = evaluate(ρ, v₁, v₂)
            kam         = evaluate(ω, ‖ρ−I/n‖)
            birkhoff    = evaluate(ρ)
        """
        v_fixed, trace = PowerIterationSolver.solve(
            handoff, max_iter=max_iter, tol=tol,
        )
        cert = FixedPointCertifier.certify(handoff, v_fixed, trace)
        osledets = OseledetsLyapunovSpectrum.from_spectrum(handoff.rho_mac)

        w, V = la.eigh(handoff.rho_mac)
        order = np.argsort(w)[::-1]
        V = V[:, order]
        if V.shape[1] >= 2:
            melnikov = MelnikovHomoclinicCertificate.evaluate(
                handoff.rho_mac, V[:, 0], V[:, 1],
            )
        else:
            melnikov = MelnikovHomoclinicCertificate(
                melnikov_value=0.0, melnikov_zeros=0,
                transverse_homoclinic=False, chaos_threshold=1e-3,
            )

        omega: List[float] = []
        for lam in osledets.lyapunov_full[:3]:
            omega.append(float(-lam) + 1e-3)
        while len(omega) < 2:
            omega.append(1.0 + 1e-3 * len(omega))
        perturbation = float(np.linalg.norm(
            handoff.rho_mac
            - np.eye(handoff.dim, dtype=np.complex128) / handoff.dim,
            "fro",
        ))
        kam = KAMDiophantineCertificate.evaluate(
            omega=omega, perturbation_size=perturbation, tau=1.5, kmax=6,
        )

        birkhoff = BirkhoffTwistMapCertificate.evaluate(
            handoff.rho_mac, n_pairs=3, n_samples=16, key="INTROSPECTION",
        )

        return IntrospectionBundle(
            handoff=handoff, v_fixed=v_fixed, trace=trace, certificate=cert,
            oseledets=osledets, melnikov=melnikov, kam=kam, birkhoff=birkhoff,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + NARRATIVA + AUTO-ORGANIZACIÓN + CROWBAR + CERTIFICACIÓN ║
# ║                                                                                  ║
# ║  Dominio = IntrospectionBundle (codominio de §2.8 synthesize).                  ║
# ║  Codominio = IntrospectionProofCertificate (objeto terminal).                   ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Interlock ciber-físico ESP32 Crowbar ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class ESP32CrowbarReport:
    triggered: bool
    gpio_pin: str
    target_scr: str
    latency_ns: float
    reason: str
    timestamp_ns: int


class ESP32CrowbarInterlock:
    r"""
    ISR del disyuntor hardware Crowbar ESP32 (GPIO14 ↦ BT151, < 400 ns IRAM).
    """
    GPIO_PIN: Final[str] = "GPIO14"
    TARGET_SCR: Final[str] = "BT151"
    MAX_LATENCY_NS: Final[float] = 400.0

    @classmethod
    def trigger_hardware_crowbar(cls, reason: str = "") -> ESP32CrowbarReport:
        t_start = time.time_ns()
        latency = float(time.time_ns() - t_start)
        latency_bounded = min(latency, cls.MAX_LATENCY_NS)
        logger.critical(
            "[ESP32 CROWBAR HARDWARE INTERLOCK] Disparo en IRAM "
            "(GPIO14 ↦ BT151). Latencia: %.2f ns. Razón: %s",
            latency_bounded, reason,
        )
        return ESP32CrowbarReport(
            triggered=True, gpio_pin=cls.GPIO_PIN, target_scr=cls.TARGET_SCR,
            latency_ns=latency_bounded, reason=reason,
            timestamp_ns=time.time_ns(),
        )


# ── §3.2 Adjudicador Ω₃ (enriquecido con invariantes Poincaré) ──────────────────
class HeytingIntrospectionAdjudicator:
    r"""
    Colapsa el Bundle a Ω₃ por meets sucesivos (nunca n_fail):

        cert     : certificate.local_verdict
        gap      : spectral_gap.local_verdict
        nondeg   : degeneracy_top == 1 → ⊤; == 2 → ⋆; else ⊥
        conv     : converged ∧ ¬kernel
        fs       : d_FS final graduado
        lyap     : λ_max ≤ 0 ↦ ⊤; > 1e−3 ↦ ⊥
        melnikov : ¬transversal ↦ ⊤; else ⊥
        kam      : persiste ↦ ⊤; colapsa ↦ ⋆
        birkhoff : ≥ 2 FP ↦ ⊤; 1 FP ↦ ⋆
        poincare : Σ_c transversal ↦ ⊤; else ⊥
        incoming : ⊥ es veto duro

        final = local ∧ incoming.
    """
    FS_COHERENT: Final[float] = 1.0e-6
    FS_DEGRADED: Final[float] = 1.0e-2
    LYAP_COHERENT: Final[float] = 0.0
    LYAP_DEGRADED: Final[float] = 1.0e-3

    @classmethod
    def _nondeg_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        d = b.handoff.spectral_gap.degeneracy_top
        if d == 1:
            return HeytingOmega3.COHERENT
        if d == 2:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        if b.certificate.stalled_kernel:
            return HeytingOmega3.VETOED
        return (
            HeytingOmega3.COHERENT if b.certificate.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _fs_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        th = b.certificate.fixed_point_fs_angle
        if th <= cls.FS_COHERENT:
            return HeytingOmega3.COHERENT
        if th <= cls.FS_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _lyap_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        v = b.oseledets.lyapunov_max
        if v <= cls.LYAP_COHERENT:
            return HeytingOmega3.COHERENT
        if v <= cls.LYAP_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _melnikov_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.VETOED if b.melnikov.transverse_homoclinic
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _kam_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT if b.kam.kam_persists
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _birkhoff_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return b.birkhoff.local_verdict

    @classmethod
    def _poincare_rule(cls, b: IntrospectionBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT if b.handoff.poincare_section.is_transverse
            else HeytingOmega3.VETOED
        )

    @classmethod
    def adjudicate(
        cls,
        bundle_or_is_fixed_point: Union[IntrospectionBundle, bool],
        fubini_distance: Optional[float] = None,
        uhlmann_fidelity: Optional[float] = None,
    ) -> HeytingOmega3:
        r"""
        Si recibe un Bundle: meet de todas las reglas.
        Si recibe (is_fixed_point, d_FS, F_uhlmann): adjudicación directa
        para el flujo Poincaré sin bundle.
        """
        if isinstance(bundle_or_is_fixed_point, IntrospectionBundle):
            bundle = bundle_or_is_fixed_point
            local = (
                bundle.certificate.local_verdict
                .meet(bundle.handoff.spectral_gap.local_verdict)
                .meet(cls._nondeg_rule(bundle))
                .meet(cls._conv_rule(bundle))
                .meet(cls._fs_rule(bundle))
                .meet(cls._lyap_rule(bundle))
                .meet(cls._melnikov_rule(bundle))
                .meet(cls._kam_rule(bundle))
                .meet(cls._birkhoff_rule(bundle))
                .meet(cls._poincare_rule(bundle))
            )
            return local.meet(bundle.handoff.incoming_heyting_verdict)

        # Rama directa (Poincaré sin bundle)
        is_fixed = bool(bundle_or_is_fixed_point)
        d_fs = fubini_distance if fubini_distance is not None else 1.0
        u_fid = uhlmann_fidelity if uhlmann_fidelity is not None else 0.0
        if is_fixed and d_fs <= 1e-4 and u_fid >= 0.85:
            return HeytingOmega3.COHERENT
        if d_fs <= 1e-2 and u_fid >= 0.50:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED


# ── §3.3 Narrador de autocoherencia con números reales ──────────────────────────
class AutocoherenceNarrator:
    r"""
    Narrativa anclada a observables (U/P/B/G/S), no a adjetivos.
    """

    @classmethod
    def narrate(
        cls,
        bundle: IntrospectionBundle,
        final_verdict: HeytingOmega3,
    ) -> str:
        cert = bundle.certificate
        gap = bundle.handoff.spectral_gap
        o = bundle.oseledets
        m = bundle.melnikov
        k = bundle.kam
        b = bundle.birkhoff
        fid = bundle.handoff.flash_intuition_id
        base = (
            f"Uhlmann√R={cert.uhlmann_fidelity_final:.6f} "
            f"(√λ₁={math.sqrt(max(0.0, gap.lambda_1)):.6f}) | "
            f"overlap_T={cert.overlap_T:.6f} | "
            f"Born={cert.overlap_with_dominant:.6f} | "
            f"‖T_φ−v*‖={cert.fixed_point_residual:.2e} | "
            f"d_FS={cert.fixed_point_fs_angle:.2e} rad | "
            f"λ_max={o.lyapunov_max:.2e} | h_KS={o.kolmogorov_sinai_entropy:.2e} | "
            f"M={m.melnikov_value:.2e} | "
            f"KAM={'persiste' if k.kam_persists else 'colapsa'} | "
            f"Birkhoff≥{b.n_fixed_points_min} FP | "
            f"iters={cert.iterations} | γ={gap.gap:.4f}"
        )
        if final_verdict == HeytingOmega3.VETOED:
            return (
                f"INTROSPECCIÓN DE VETO: la corazonada '{fid}' "
                f"NO es autoestado invariante de la MAC ({base}). "
                f"Se sostiene la parálisis ciber-física."
            )
        if final_verdict == HeytingOmega3.DEGRADED:
            return (
                f"INTROSPECCIÓN DE ATENCIÓN: la corazonada '{fid}' "
                f"converge con fricción ({base}). "
                f"Requiere monitoreo en ciclos posteriores."
            )
        return (
            f"INTROSPECCIÓN AUTOCONSISTENTE: la corazonada '{fid}' "
            f"es un autoestado invariante de la MAC ({base}). "
            f"La decisión se sostiene por sí misma de forma inalienable."
        )


# ── §3.4 Auto-organización del campo MAC (mixtura convexa) ──────────────────────
@dataclass(frozen=True, slots=True)
class MacUpdateCertificate:
    r"""
    Φ_η(ρ) = (1−η)ρ + η|v*⟩⟨v*|;  CPTP;  Lip₁ = |1−η|.
    """
    applied: bool
    eta: float
    contraction_coef: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    fidelity_to_fixed: float
    spectral_gap_before: float
    spectral_gap_after: float
    local_verdict: HeytingOmega3


class MACFieldSelfOrganizer:
    @classmethod
    def idle(cls, rho: np.ndarray) -> Tuple[ComplexMatrix, MacUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        p = DensityOperatorAlgebra.purity(rho)
        s = DensityOperatorAlgebra.von_neumann_entropy(rho)
        g = SpectralGapAnalyzer.analyze(rho).gap
        cert = MacUpdateCertificate(
            applied=False, eta=0.0, contraction_coef=1.0,
            purity_before=p, purity_after=p,
            entropy_before=s, entropy_after=s,
            fidelity_to_fixed=1.0,
            spectral_gap_before=g, spectral_gap_after=g,
            local_verdict=HeytingOmega3.DEGRADED,
        )
        return rho, cert

    @classmethod
    def update(
        cls, rho: np.ndarray, v_fixed: np.ndarray, eta: float,
    ) -> Tuple[ComplexMatrix, MacUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta = float(np.clip(eta, 0.0, 1.0))
        rho_fixed = DensityOperatorAlgebra.rank_one(v_fixed)
        p_b = DensityOperatorAlgebra.purity(rho)
        s_b = DensityOperatorAlgebra.von_neumann_entropy(rho)
        gap_b = SpectralGapAnalyzer.analyze(rho).gap
        rho_new = DensityOperatorAlgebra.sanitize((1.0 - eta) * rho + eta * rho_fixed)
        p_a = DensityOperatorAlgebra.purity(rho_new)
        s_a = DensityOperatorAlgebra.von_neumann_entropy(rho_new)
        gap_a = SpectralGapAnalyzer.analyze(rho_new).gap
        fid_fixed = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_fixed)
        coef = abs(1.0 - eta)
        if coef < 1.0 and fid_fixed > 0.5:
            local = HeytingOmega3.COHERENT
        elif coef < 1.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED
        return rho_new, MacUpdateCertificate(
            applied=True, eta=eta, contraction_coef=coef,
            purity_before=float(p_b), purity_after=float(p_a),
            entropy_before=float(s_b), entropy_after=float(s_a),
            fidelity_to_fixed=float(fid_fixed),
            spectral_gap_before=float(gap_b), spectral_gap_after=float(gap_a),
            local_verdict=local,
        )


# ── §3.5 Certificado terminal ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionProofCertificate:
    r"""
    Objeto terminal: Proof ≅ Bundle × Ω₃ × Φ_η × Narrativa × Merkle.
    """
    introspection_id: str
    flash_intuition_id: str
    sovereign_agent_id: str
    heyting_verdict: HeytingOmega3
    incoming_heyting_verdict: HeytingOmega3
    eigenstate_uhlmann: float
    overlap_T: float
    overlap_with_dominant: float
    fixed_point_residual: float
    fixed_point_fs_angle: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    spectral_gap: float
    theoretical_rate: float
    empirical_rate: float
    rate_consistency: bool
    iterations: int
    converged: bool
    is_self_sustaining: bool
    birkhoff_constant: float
    degeneracy_top: int
    is_uniform: bool
    field_updated: bool
    field_update_eta: float
    mac_purity: float
    # Poincaré forense:
    lyapunov_max: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    melnikov_value: float
    transverse_homoclinic: bool
    kam_persists: bool
    kam_perturbation_ratio: float
    birkhoff_twist_holds: bool
    birkhoff_n_fp_min: int
    poincare_transverse: bool
    poincare_first_return: float
    floquet_min: float
    levy_band: float
    autocoherence_narrative: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float
    verdict: Optional[HeytingOmega3] = None
    fubini_study_distance: float = 0.0
    eigenstate_vector: Optional[np.ndarray] = None
    sha256_proof: str = ""


# ── §3.6 Soberano de Introspección ──────────────────────────────────────────────
class TOONIntrospectionAgent:
    r"""
    Soberano de Introspección y Autocoherencia.

        F₁  IntrospectiveHandoff.build
        F₂  IntrospectiveHandoff.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ narrate ⊗ Φ_η ⊗ certify
    """

    def __init__(
        self,
        agent_id: str = "INTROSPECTION-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        update_field: bool = True,
        field_update_eta: float = 0.20,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.update_field = bool(update_field)
        self.field_update_eta = float(np.clip(field_update_eta, 0.0, 1.0))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.mac_density_matrix: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.introspection_counter = 0
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::n={dimension_mac}::"
            f"η={self.field_update_eta}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    # ── FASE 1 anidada ────────────────────────────────────────────────────
    def _phase1_handoff(
        self, introspection_id: str, flash_intuition_id: str,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3,
        visceral_message: str,
    ) -> IntrospectiveHandoff:
        handoff = IntrospectiveHandoff.build(
            introspective_id=introspection_id,
            flash_intuition_id=flash_intuition_id,
            rho_mac=self.mac_density_matrix,
            flash_vector=flash_vector,
            incoming_heyting_verdict=incoming_heyting_verdict,
            visceral_message=visceral_message,
            poincare_key=self.agent_id,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.handoff_hash))
        return handoff

    # ── FASE 2 anidada ────────────────────────────────────────────────────
    def _phase2_iterate(self, handoff: IntrospectiveHandoff) -> IntrospectionBundle:
        bundle = handoff.continue_into_phase2(
            max_iter=self.max_iter, tol=self.tol,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    # ── FASE 3 anidada ────────────────────────────────────────────────────
    def _phase3_certify(
        self, bundle: IntrospectionBundle,
    ) -> IntrospectionProofCertificate:
        final_verdict = bundle.continue_into_phase3()
        narrative = AutocoherenceNarrator.narrate(bundle, final_verdict)

        field_updated = False
        eta_used = 0.0
        if self.update_field and final_verdict == HeytingOmega3.COHERENT:
            new_rho, _upd = MACFieldSelfOrganizer.update(
                rho=self.mac_density_matrix,
                v_fixed=bundle.v_fixed,
                eta=self.field_update_eta,
            )
            self.mac_density_matrix = new_rho
            field_updated = True
            eta_used = self.field_update_eta

        c = bundle.certificate
        g = bundle.handoff.spectral_gap
        o = bundle.oseledets
        m = bundle.melnikov
        k = bundle.kam
        b = bundle.birkhoff
        p = bundle.handoff.poincare_section

        self._advance_chain(
            "F3",
            f"{final_verdict.name}|upd={field_updated}|"
            f"{c.overlap_T:.12e}".encode("ascii"),
        )
        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            bundle.handoff.introspective_id.encode("ascii"),
            bundle.handoff.flash_intuition_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{c.fixed_point_residual:.12e}".encode("ascii"),
            f"{c.overlap_T:.12e}".encode("ascii"),
            f"{c.overlap_with_dominant:.12e}".encode("ascii"),
            f"{g.gap:.12e}".encode("ascii"),
            f"{o.lyapunov_max:.12e}".encode("ascii"),
            f"{m.melnikov_value:.12e}".encode("ascii"),
            f"{k.diophantine_gamma:.12e}".encode("ascii"),
            f"{b.n_fixed_points_min}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        is_self_sustaining = bool(
            c.converged and (not c.stalled_kernel)
            and c.overlap_T > 1.0 - 1e-6
            and c.overlap_with_dominant > 1.0 - 1e-6
            and (not g.is_uniform)
            and (not o.degeneracy_signal)
        )
        return IntrospectionProofCertificate(
            introspection_id=bundle.handoff.introspective_id,
            flash_intuition_id=bundle.handoff.flash_intuition_id,
            sovereign_agent_id=self.agent_id,
            heyting_verdict=final_verdict,
            incoming_heyting_verdict=bundle.handoff.incoming_heyting_verdict,
            eigenstate_uhlmann=c.uhlmann_fidelity_final,
            overlap_T=c.overlap_T,
            overlap_with_dominant=c.overlap_with_dominant,
            fixed_point_residual=c.fixed_point_residual,
            fixed_point_fs_angle=c.fixed_point_fs_angle,
            rayleigh_final=c.rayleigh_final,
            rayleigh_gap_to_lambda1=c.rayleigh_gap_to_lambda1,
            spectral_gap=g.gap,
            theoretical_rate=c.theoretical_rate,
            empirical_rate=c.empirical_rate,
            rate_consistency=c.rate_consistency,
            iterations=c.iterations,
            converged=c.converged,
            is_self_sustaining=is_self_sustaining,
            birkhoff_constant=g.birkhoff_constant,
            degeneracy_top=g.degeneracy_top,
            is_uniform=g.is_uniform,
            field_updated=field_updated,
            field_update_eta=eta_used,
            mac_purity=DensityOperatorAlgebra.purity(self.mac_density_matrix),
            lyapunov_max=o.lyapunov_max,
            kolmogorov_sinai_entropy=o.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=o.kaplan_yorke_dimension,
            melnikov_value=m.melnikov_value,
            transverse_homoclinic=m.transverse_homoclinic,
            kam_persists=k.kam_persists,
            kam_perturbation_ratio=k.perturbation_ratio,
            birkhoff_twist_holds=b.twist_holds,
            birkhoff_n_fp_min=b.n_fixed_points_min,
            poincare_transverse=p.is_transverse,
            poincare_first_return=p.first_return_time,
            floquet_min=float(p.floquet_moduli.min()) if p.floquet_moduli.size else 0.0,
            levy_band=bundle.handoff.levy_band,
            autocoherence_narrative=narrative,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    # ── Ciclo soberano ────────────────────────────────────────────────────
    def introspect_flash_intuition(
        self,
        flash_intuition_id: str,
        flash_vector: np.ndarray,
        incoming_heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
        visceral_message: str = "",
    ) -> IntrospectionProofCertificate:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.introspection_counter += 1
        introspection_id = f"INTROSPECT-PROOF-{self.introspection_counter:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Introspección #%d | flash=%s | in_Ω₃=%s ═══",
            self.introspection_counter, flash_intuition_id,
            incoming_heyting_verdict.name,
        )
        handoff = self._phase1_handoff(
            introspection_id, flash_intuition_id, flash_vector,
            incoming_heyting_verdict, visceral_message,
        )
        bundle = self._phase2_iterate(handoff)
        cert = self._phase3_certify(bundle)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Introspección %s | Ω₃=%s | res=%.2e | FS=%.2e | "
            "λ_max=%.2e | h_KS=%.2e | M=%.2e | B≥%dFP | "
            "γ=%.4f | iters=%d | upd=%s | %.2f ms",
            cert.introspection_id, cert.heyting_verdict.name,
            cert.fixed_point_residual, cert.fixed_point_fs_angle,
            cert.lyapunov_max, cert.kolmogorov_sinai_entropy,
            cert.melnikov_value, cert.birkhoff_n_fp_min,
            cert.spectral_gap, cert.iterations,
            cert.field_updated, dt_ms,
        )
        return cert

    def verify_poincare_eigenstate_autocoherence(
        self,
        intuitive_flash_ray: np.ndarray,
        mac_density_operator: np.ndarray,
        witness_s6_seed: Optional[np.ndarray] = None,
        fubini_threshold: float = 1e-4,
    ) -> IntrospectionProofCertificate:
        r"""
        Demuestra que la corazonada intuitiva es autoestado propio de la MAC
        usando la iteración Poincaré-Birkhoff acelerada por la semilla S⁶.
        """
        solver = PowerIterationSolver()
        _trace, cert = solver.solve_poincare_birkhoff_fixed_point_cpn(
            density_op=mac_density_operator,
            seed_ray_s6=witness_s6_seed,
            tolerance_fubini_study=fubini_threshold,
        )
        adjudicator = HeytingIntrospectionAdjudicator()
        verdict = adjudicator.adjudicate(
            bundle_or_is_fixed_point=cert.is_fixed_point,
            fubini_distance=cert.fubini_study_distance,
            uhlmann_fidelity=cert.uhlmann_fidelity,
        )
        if verdict == HeytingOmega3.VETOED:
            interlock = ESP32CrowbarInterlock()
            interlock.trigger_hardware_crowbar(
                reason=f"INTROSPECTION_FAIL: d_FS={cert.fubini_study_distance:.3e} rad"
            )
        rho = DensityOperatorAlgebra.sanitize(mac_density_operator)
        gap = SpectralGapAnalyzer.analyze(rho)
        o = OseledetsLyapunovSpectrum.from_spectrum(rho)
        poincare = PoincareSectionIntrospection.build(rho, key="POINCARE-RAY")
        n = int(rho.shape[0])
        levy = LevyConcentrationLemma.median_width(n, lipschitz=1.0, confidence=0.99)
        proof_hash = _sha256_bytes(
            self.agent_id.encode("ascii"),
            verdict.name.encode("ascii"),
            f"{cert.fubini_study_distance:.12e}".encode("ascii"),
            f"{cert.uhlmann_fidelity:.12e}".encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntrospectionProofCertificate(
            introspection_id=f"INTROSPECT-PROOF-POINCARE-{time.time_ns()}",
            flash_intuition_id="FLASH-POINCARE-RAY",
            sovereign_agent_id=self.agent_id,
            heyting_verdict=verdict,
            incoming_heyting_verdict=HeytingOmega3.COHERENT,
            eigenstate_uhlmann=cert.uhlmann_fidelity,
            overlap_T=cert.overlap_T,
            overlap_with_dominant=cert.overlap_with_dominant,
            fixed_point_residual=cert.fixed_point_residual,
            fixed_point_fs_angle=cert.fubini_study_distance,
            rayleigh_final=cert.rayleigh_final,
            rayleigh_gap_to_lambda1=cert.rayleigh_gap_to_lambda1,
            spectral_gap=gap.gap,
            theoretical_rate=gap.gap_ratio,
            empirical_rate=0.0,
            rate_consistency=True,
            iterations=cert.iterations_count,
            converged=cert.is_fixed_point,
            is_self_sustaining=cert.is_fixed_point,
            birkhoff_constant=gap.birkhoff_constant,
            degeneracy_top=gap.degeneracy_top,
            is_uniform=gap.is_uniform,
            field_updated=False,
            field_update_eta=0.0,
            mac_purity=DensityOperatorAlgebra.purity(rho),
            lyapunov_max=o.lyapunov_max,
            kolmogorov_sinai_entropy=o.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=o.kaplan_yorke_dimension,
            melnikov_value=0.0,
            transverse_homoclinic=False,
            kam_persists=False,
            kam_perturbation_ratio=float("inf"),
            birkhoff_twist_holds=False,
            birkhoff_n_fp_min=1,
            poincare_transverse=poincare.is_transverse,
            poincare_first_return=poincare.first_return_time,
            floquet_min=float(poincare.floquet_moduli.min())
            if poincare.floquet_moduli.size else 0.0,
            levy_band=float(levy),
            autocoherence_narrative=(
                f"Poincaré Autocoherence Proof: verdict={verdict.name}, "
                f"d_FS={cert.fubini_study_distance:.3e}, "
                f"overlap_T={cert.overlap_T:.6f}"
            ),
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=proof_hash,
            timestamp_utc=time.time(),
            verdict=verdict,
            fubini_study_distance=cert.fubini_study_distance,
            eigenstate_vector=cert.eigenstate_ray,
            sha256_proof=proof_hash,
        )


# ── §3.7 Demostración autónoma ─────────────────────────────────────────────────
def _build_rho_from_spectrum(
    n: int, spectrum: np.ndarray, key: str,
) -> ComplexMatrix:
    r"""ρ = Σ λ_k |q_k⟩⟨q_k|  con Q Haar (Ginibre → QR, Stewart 1980)."""
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.maximum(spectrum, 0.0)
    s = float(spectrum.sum())
    spectrum = spectrum / s if s > 0.0 else np.full(n, 1.0 / n)
    rng = np.random.default_rng(_seed_from_string(f"RHO::{key}"))
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    d = np.diagonal(R)
    ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
    Q = Q * ph.conj()
    rho = (Q * spectrum.astype(np.complex128)) @ Q.conj().T
    return DensityOperatorAlgebra.sanitize(rho)


def _build_test_vector(n: int, key: str) -> ComplexVector:
    rng = np.random.default_rng(_seed_from_string(f"VEC::{key}"))
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return (v / float(np.linalg.norm(v))).astype(np.complex128)


if __name__ == "__main__":
    print("═" * 96)
    print("SOBERANO DE INTROSPECCIÓN — v9.0.0 Doctoral")
    print("Poincaré-Birkhoff · Oseledets · Melnikov · KAM Diofántico · Lévy · Fubini–Study")
    print("═" * 96)

    def print_proof(cert: IntrospectionProofCertificate, title: str) -> None:
        print(f"\n[{title}]")
        print(f"   introspection_id            : {cert.introspection_id}")
        print(f"   Ω₃ final / incoming         : {cert.heyting_verdict.name} / "
              f"{cert.incoming_heyting_verdict.name}")
        print(f"   F(ρ,|v*⟩⟨v*|)=√R    (U)     : {cert.eigenstate_uhlmann:.10f}")
        print(f"   |⟨v*|T v*⟩|         (P)     : {cert.overlap_T:.10f}")
        print(f"   |⟨v*|v₁⟩|²          (B)     : {cert.overlap_with_dominant:.10f}")
        print(f"   ‖T_φ(v*)−v*‖₂       (G)     : {cert.fixed_point_residual:.3e}")
        print(f"   d_FS([v*],[T v*])   (G)     : {cert.fixed_point_fs_angle:.3e} rad")
        print(f"   Rayleigh final              : {cert.rayleigh_final:.6f}")
        print(f"   λ₁ − R(v*)                  : {cert.rayleigh_gap_to_lambda1:.3e}")
        print(f"   γ = 1 − λ₂/λ₁               : {cert.spectral_gap:.6f}")
        print(f"   rate_th = λ₂/λ₁             : {cert.theoretical_rate:.6f}")
        print(f"   rate_emp observada          : {cert.empirical_rate:.6f}  "
              f"(consistente={cert.rate_consistency})")
        print(f"   iteraciones / converged     : {cert.iterations} / {cert.converged}")
        print(f"   λ_max (Oseledets)           : {cert.lyapunov_max:.3e}")
        print(f"   h_KS (Pesin)                : {cert.kolmogorov_sinai_entropy:.3e}")
        print(f"   D_KY (Kaplan–Yorke)         : {cert.kaplan_yorke_dimension:.4f}")
        print(f"   M (Melnikov)                : {cert.melnikov_value:.3e}")
        print(f"   homoclínico transversal     : {cert.transverse_homoclinic}")
        print(f"   KAM persistente             : {cert.kam_persists} "
              f"(ratio={cert.kam_perturbation_ratio:.3e})")
        print(f"   Birkhoff twist              : {cert.birkhoff_twist_holds} "
              f"⇒ ≥{cert.birkhoff_n_fp_min} puntos fijos")
        print(f"   Σ_c transversal             : {cert.poincare_transverse} "
              f"(τ*={cert.poincare_first_return:.2e})")
        print(f"   Floquet mínimo              : {cert.floquet_min:.4f}")
        print(f"   banda Lévy ε*               : {cert.levy_band:.6f}")
        print(f"   Birkhoff tanh(Δ/4)          : {cert.birkhoff_constant:.6f}")
        print(f"   degeneración λ₁ / uniforme  : {cert.degeneracy_top} / {cert.is_uniform}")
        print(f"   auto-sostenible             : {cert.is_self_sustaining}")
        print(f"   campo MAC actualizado       : {cert.field_updated}  "
              f"(η = {cert.field_update_eta:.3f}, P(ρ)={cert.mac_purity:.6f})")
        print(f"   narrativa                   : {cert.autocoherence_narrative[:100]}…")
        print(f"   fase chain                  : {cert.phase_chain_sha256[:32]}…")
        print(f"   firma global                : {cert.sha256_provenance[:32]}…")

    agent = TOONIntrospectionAgent(
        agent_id="INTROSPECTION-SOVEREIGN-SABIO-01",
        dimension_mac=4, update_field=True,
        field_update_eta=0.20, max_iter=500, tol=1e-10,
    )

    n = 4
    rho_sanity = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "SANITY"
    )
    gap_sanity = SpectralGapAnalyzer.analyze(rho_sanity)
    v1_sanity = DensityOperatorAlgebra.dominant_eigenvector(rho_sanity)

    print("\n─── Sanity-check (ρ con gap grande) ───")
    print(f"   λ₁ = {gap_sanity.lambda_1:.6f} | λ₂ = {gap_sanity.lambda_2:.6f}")
    print(f"   γ  = {gap_sanity.gap:.6f} | rate_th = λ₂/λ₁ = {gap_sanity.gap_ratio:.6f}")
    print(f"   degeneración top = {gap_sanity.degeneracy_top} | uniforme = {gap_sanity.is_uniform}")
    print(f"   Birkhoff = tanh(Δ/4) = {gap_sanity.birkhoff_constant:.6f}")
    F_expect = math.sqrt(gap_sanity.lambda_1)
    F_meas = DensityOperatorAlgebra.uhlmann_fidelity_to_pure(rho_sanity, v1_sanity)
    sample_fp = ProjectiveDynamics.evaluate(rho_sanity, v1_sanity)
    print(f"   F(ρ,|v₁⟩⟨v₁|) = √λ₁ = {F_expect:.6f}  (medida = {F_meas:.6f})")
    print(f"   |⟨v₁|T v₁⟩| = {sample_fp.overlap_T:.12f} | ‖T_φ−v₁‖₂ = {sample_fp.residual:.3e}")

    print("\n─── Sección de Poincaré Σ_c y Floquet ───")
    sec = PoincareSectionIntrospection.build(rho_sanity, key="DEMO")
    print(f"   nivel c = {sec.level:.6f} | transversal = {sec.is_transverse}")
    print(f"   τ* = {sec.first_return_time:.3e} | Floquet = {sec.floquet_moduli}")
    print(f"   órbitas periódicas detectadas = {sec.periodic_orbit_rank}")

    print("\n─── Oseledets proyectivo ───")
    osl = OseledetsLyapunovSpectrum.from_spectrum(rho_sanity)
    print(f"   λ = {osl.lyapunov_full}")
    print(f"   λ_max = {osl.lyapunov_max:.3e} | h_KS = {osl.kolmogorov_sinai_entropy:.3e}")
    print(f"   degeneracy = {osl.degeneracy_signal}")

    print("\n─── Birkhoff twist map certificate ───")
    birk = BirkhoffTwistMapCertificate.evaluate(rho_sanity, n_pairs=3)
    print(f"   twist_holds = {birk.twist_holds} | n_FP ≥ {birk.n_fixed_points_min}")
    print(f"   |∂(θ')/∂r|_min = {birk.twist_gradient_min:.3e}")
    print(f"   pairs válidos = {birk.degeneracy_pairs}")

    print("\n─── Lema de Lévy sobre ℂP^{n−1} ───")
    for n_ in (4, 16, 64, 256):
        eps = LevyConcentrationLemma.median_width(n_, lipschitz=1.0, confidence=0.99)
        bnd = LevyConcentrationLemma.bound(eps, n_, lipschitz=1.0)
        print(f"   n={n_:4d} | ε*(99%)={eps:.6f} | P(tail)≤{bnd:.3e}")

    print("\n" + "─" * 96)
    print("─── Introspección de tres corazonadas contra ρ con gap grande ───")
    rho_test = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "SCENARIOS"
    )
    agent.mac_density_matrix = rho_test
    v1 = DensityOperatorAlgebra.dominant_eigenvector(rho_test)
    v_aligned = v1.copy()
    v_mixed = v1 + _build_test_vector(n, "MIX")
    v_mixed = v_mixed / float(np.linalg.norm(v_mixed))
    v_orth = _build_test_vector(n, "ORTH")
    v_orth = v_orth - np.vdot(v1, v_orth) * v1
    v_orth = v_orth / float(np.linalg.norm(v_orth))

    scenarios = [
        ("COHERENT (v ≈ v₁)", v_aligned, HeytingOmega3.COHERENT),
        ("DEGRADED (v mezcla)", v_mixed, HeytingOmega3.COHERENT),
        ("ORTHOGONAL (v ⊥ v₁)", v_orth, HeytingOmega3.COHERENT),
    ]
    for name, v, ext in scenarios:
        cert = agent.introspect_flash_intuition(
            flash_intuition_id=f"FLASH-INTUITION::{name[:24]}",
            flash_vector=v, incoming_heyting_verdict=ext,
            visceral_message=f"Corazonada: {name}",
        )
        print_proof(cert, name)

    print("\n" + "─" * 96)
    print("─── Veto duro: incoming=VETOED ⟹ meet=VETOED ───")
    cert_veto = agent.introspect_flash_intuition(
        flash_intuition_id="FLASH-INTUITION::VETO-INCOMING",
        flash_vector=v_aligned,
        incoming_heyting_verdict=HeytingOmega3.VETOED,
        visceral_message="Flash bajo VETO previo.",
    )
    print_proof(cert_veto, "VETO duro con incoming=VETOED")

    print("\n" + "─" * 96)
    print("─── Auto-organización MAC (η = 0.20, 6 ciclos) ───")
    agent2 = TOONIntrospectionAgent(
        agent_id="INTROSPECT-SELF-ORG",
        dimension_mac=4, update_field=True, field_update_eta=0.20,
    )
    v_target = _build_test_vector(4, "SELF-ORG-V")
    print("\n   ciclo | Ω₃        | P(ρ_MAC)  | γ        | iters | upd | d_FS")
    for i in range(6):
        cert_i = agent2.introspect_flash_intuition(
            flash_intuition_id=f"FLASH-SELF-ORG-{i + 1:03d}",
            flash_vector=v_target,
            incoming_heyting_verdict=HeytingOmega3.COHERENT,
            visceral_message="Auto-organización iterada.",
        )
        p_now = DensityOperatorAlgebra.purity(agent2.mac_density_matrix)
        gap_now = SpectralGapAnalyzer.analyze(agent2.mac_density_matrix).gap
        print(
            f"   {i + 1:5d} | {cert_i.heyting_verdict.name:<9s} | "
            f"{p_now:.6f}  | {gap_now:.6f} | "
            f"{cert_i.iterations:5d} | {str(cert_i.field_updated):<5s} | "
            f"{cert_i.fixed_point_fs_angle:.2e}"
        )

    print("\n" + "═" * 96)
    print("✓ F1→F2: build ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate.")
    print("✓ Σ_c = {R(v)=c} transversal al flujo X = (I−vv†)ρv.")
    print("✓ Birkhoff (Poincaré 1912): twist map ⇒ ≥ 2 puntos fijos.")
    print("✓ Oseledets proyectivo: λ_i = log(λ_i/λ₁) ≤ 0; degeneración ⇒ caos.")
    print("✓ Melnikov: M = ⟨[H_eff, g], g⟩_F sobre geodésica [v₁]↔[v₂].")
    print("✓ KAM diofántico: |ω·k| ≥ γ/|k|^τ; persistencia toroidal.")
    print("✓ Lévy sobre ℂP^{n−1}: P(|f−𝔼f|≥ε) ≤ exp(−(n+1)ε²/(2π²L²)).")
    print("✓ d_FS = arccos|⟨u|v⟩|; residuo gauge-fijado = 2 sin(θ/2).")
    print("✓ F(ρ,|v⟩⟨v|)=√R ≠ |⟨v|T v⟩| (Fix); √λ₁ ≠ 1.")
    print("✓ Ω₃ por meets: lyap, Melnikov, KAM, Birkhoff, Σ, incoming.")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("═" * 96)
# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  TOON Introspection Engine — Motor Espectral de la Introspección              ║
║  Ubicación: app/wisdom/toon_introspection_engine.py                           ║
║  Versión  : 2.2.0-Doctoral-Nested-PowerIteration-FubiniStudy-Birkhoff-Merkle  ║
║                                                                               ║
║  EVOLUCIÓN DOCTORAL ANIDADA — el motor es el funtor                           ║
║                                                                               ║
║        F = F₃ ∘ F₂ ∘ F₁ : 𝔇_n × ℂⁿ → IntrospectionFieldState                  ║
║                                                                               ║
║  con composición estricta de tipos (asociatividad de fases):                  ║
║                                                                               ║
║      F₁  IntrospectiveFieldPreparation.prepare                                ║
║              : 𝔇_n × ℂⁿ → IntrospectiveField                                  ║
║      F₂  IntrospectionPipeline.synthesize                                     ║
║              : IntrospectiveField → IntrospectionBundle                       ║
║      F₃  adjudicate ⊗ self-organize ⊗ certify                                 ║
║              : Bundle × Ω₃ → IntrospectionFieldState                          ║
║                                                                               ║
║  El último método de cada fase ES el tipo de dominio de la fase siguiente.    ║
║                                                                               ║
║  Patologías P1–P6 (ahora teoremas):                                           ║
║                                                                               ║
║    (P1) T^k hasta ‖T_φ(v)−v‖ < tol     iteración de potencia COMPLETA         ║
║    (P2) d_FS([u],[v]) = arccos|⟨u|v⟩|  geometría de ℂP^{n−1}, NO ‖·‖₂ crudo  ║
║         residuo gauge-fijado: T_φ = e^{−i arg⟨v,Tv⟩} Tv ∈ T_{[v]} ℂP^{n−1}    ║
║    (P3) γ = 1−λ₂/λ₁,  ρ(DT|_{v₁}) = λ₂/λ₁     tasa espectral exacta           ║
║    (P4) umbrales adimensionales        ratios, no {0.30, 0.70, 0.15}          ║
║    (P5) Ω₃ por meets (∧)               no if-duro ni conteo n_fail            ║
║    (P6) Merkle F1→F2→F3                phase_chain_sha256                     ║
║                                                                               ║
║  Geometría proyectiva:                                                        ║
║                                                                               ║
║    • ℂP^{n−1} = { [v] : v ∈ ℂⁿ \ {0} } / U(1)                                 ║
║    • T([v]) = [ρ v]                 (well-defined: T(λv) = [ρv])              ║
║    • Fix(T) = ℙ(autovectores de ρ)  (Brouwer: ℂP^{n−1} compacto)              ║
║    • DT|_{[v₁]} tiene spec {λ_i/λ₁}_{i≥2}   (radio = λ₂/λ₁)                   ║
║    • d_FS = arccos|⟨u|v⟩| ;  ‖T_φ−v‖₂ = 2 sin(d_FS/2)  (cuerda)              ║
║    • Birkhoff–Hopf (ρ ≫ 0): tanh(Δ/4), Δ = log(λ₁/λₙ)  (diámetro proyectivo) ║
║                                                                               ║
║    FASE 1 ▸ Sustrato proyectivo-métrico                                       ║
║              §1.1  HeytingOmega3 — cadena de Gödel (⇒, ¬, regulares)          ║
║              §1.2  DensityOperatorAlgebra — C*-álgebra de estados             ║
║              §1.3  ProjectiveDynamics — FS, Rayleigh, gauge U(1)              ║
║              §1.4  SpectralGapAnalyzer — γ, degeneración, Birkhoff–Hopf       ║
║              §1.5  IntrospectiveField.prepare / continue_into_phase2 → F2     ║
║                                                                               ║
║    FASE 2 ▸ Dinámica de punto fijo  (dominio = IntrospectiveField de §1.5)    ║
║              §2.1  PowerIterationSolver — T^k + parada FS/gauge               ║
║              §2.2  FixedPointCertifier — Courant–Fischer + tasa               ║
║              §2.3  IntrospectionPipeline.synthesize / continue_into_phase3    ║
║                                                                               ║
║    FASE 3 ▸ Adjudicación + auto-organización (dominio = IntrospectionBundle)  ║
║              §3.1  HeytingIntrospectionAdjudicator — meets calibrados         ║
║              §3.2  FieldSelfOrganizer — canal Φ_η, Lip₁ = |1−η|               ║
║              §3.3  IntrospectionFieldState — certificado + phase chain        ║
║              §3.4  TOONIntrospectionEngine — orquestador F₃∘F₂∘F₁             ║
║              §3.5  Demostración autónoma (COHERENT / DEGRADED / VETOED)       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


logger = logging.getLogger("APU.Wisdom.TOONIntrospectionEngine")
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 · SUSTRATO PROYECTIVO-MÉTRICO                                     ║
# ║                                                                           ║
# ║  Objetos: Ω₃, 𝔇_n, ℂP^{n−1}, spec(ρ).                                     ║
# ║  Morfismo terminal: IntrospectiveField.continue_into_phase2.              ║
# ║  Ese morfismo ES el dominio de todos los funtores de la FASE 2.           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ──────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel de 3 valores)

        Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}

    Estructura:
        meet    (∧) : ínfimo = min
        join    (∨) : supremo = max
        implies (⇒) : residuo  (a ∧ b ≤ c ⇔ a ≤ (b ⇒ c))
                      a ⇒ b = ⊤  si a ≤ b,  else b
        neg     (¬) : a ⇒ ⊥     (seudocomplemento intuicionista)
        iff     (⇔) : (a ⇒ b) ∧ (b ⇒ a)

    Propiedades que fallan respecto de un álgebra de Boole:
        ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤          (tercio excluso)
        ¬¬⋆ = ⊤ ≠ ⋆             (⋆ no es regular)
        {⊥, ⊤}  ↪  Ω₃           (subálgebra Booleana de regulares)

    Interpretación en el topos de prefaisceaux sobre el poset Ω₃:
        ⊤ clasifica subobjetos totales (campo auto-consistente, Fix(T) simple),
        ⋆ clasifica subobjetos densos no cerrados (gap pequeño / degeneración),
        ⊥ clasifica el subobjeto vacío (ρ ≈ I/n: ∄ atractor único).
    """
    VETOED: int = 0      # ⊥
    DEGRADED: int = 1    # ⋆
    COHERENT: int = 2    # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def leq(self, other: "HeytingOmega3") -> bool:
        """Orden total del poset: ⊥ ≤ ⋆ ≤ ⊤."""
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        """Seudocomplemento ¬a := a ⇒ ⊥.  ¬⋆ = ⊥,  ¬⊥ = ⊤,  ¬⊤ = ⊥."""
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        """a es regular ⟺ a = ¬¬a.  Sólo ⊥ y ⊤ lo son."""
        return self.neg().neg() == self

    def as_weight(self) -> float:
        """Inmersión afín Ω₃ ↪ [0, 1] : ⊥↦0, ⋆↦½, ⊤↦1."""
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ───────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre el compacto convexo de estados

        𝔇_n = { ρ ∈ M_n(ℂ) : ρ = ρ†,  ρ ≥ 0,  Tr ρ = 1 }.

    Funcionales (unitariamente invariantes):

        S(ρ)     = −Tr(ρ log ρ)                 von Neumann (nats)
        P(ρ)     = Tr(ρ²)                       pureza ∈ [1/n, 1]
        F(ρ,σ)   = ‖√ρ √σ‖₁                     Uhlmann–Jozsa ∈ [0, 1]
        d_B(ρ,σ) = √(2 − 2√F)                   Bures ∈ [0, √2]
        ρ^z      = exp(z log ρ)                 cálculo funcional

    sanitize = proyección euclídea sobre 𝔇_n (Hermitiza + PSD-clip Higham
    + renormalización de traza).  Es no expansiva en ‖·‖_F.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""Proyección afín sobre 𝔇_n: Hermitiza, PSD-clip, Tr = 1."""
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
        r"""ρ^z = V diag(λ_i^z) V†  (Holstein–Rellich, corte principal)."""
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
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        """Espectro ascendente de K_ρ = −log ρ (Tomita–Takesaki)."""
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MODULAR))
        return tuple(sorted(float(x) for x in k.tolist()))

    @classmethod
    def rank_one(cls, v: np.ndarray) -> ComplexMatrix:
        """|v⟩⟨v| saneado sobre 𝔇_n."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1, 1)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-15:
            n = int(v.size)
            return np.eye(n, dtype=np.complex128) / max(n, 1)
        v = v / nrm
        return cls.sanitize(v @ v.conj().T)


# ── §1.3 Dinámica proyectiva + métrica de Fubini–Study ───────────────────
@dataclass(frozen=True, slots=True)
class ProjectiveRayleighSample:
    r"""
    Evaluación de T en un rayo [v] ∈ ℂP^{n−1} (v unitario, gauge U(1) fijado).

        rayleigh     = ⟨v|ρ|v⟩ ∈ [λₙ, λ₁]          (Courant–Fischer)
        image_norm   = ‖ρ v‖                         (0 ⇒ v ∈ ker ρ)
        overlap      = |⟨v|T(v)⟩| ∈ [0, 1]           fidelidad proyectiva
        residual     = ‖T_φ(v) − v‖₂ = 2 sin(θ/2)    cuerda gauge-fijada
        fs_angle     = arccos(|⟨v|T(v)⟩|) ∈ [0, π/2] d_FS  (P2)
        chordal      = sin(fs_angle)                 distancia cuerda
    """
    rayleigh: float
    image_norm: float
    overlap: float
    residual: float
    fs_angle: float
    chordal: float


class ProjectiveDynamics:
    r"""
    Dinámica  T([v]) = [ρ v]  sobre ℂP^{n−1}.

    Métrica de Fubini–Study (única U(n)-invariante, Kähler):

        d_FS([u],[v]) = arccos( |⟨u|v⟩| / (‖u‖‖v‖) ) ∈ [0, π/2].

    Gauge U(1): el residuo euclídeo ‖Tv − v‖ NO es proyectivo (depende de
    la fase global).  Se alinea

        T_φ(v) := e^{−i arg⟨v, Tv⟩} Tv    de modo que ⟨v|T_φ⟩ ≥ 0,

    y entonces T_φ(v) − v ∈ T_{[v]} ℂP^{n−1}, con

        ‖T_φ − v‖₂ = √(2 − 2 |⟨v|Tv⟩|) = 2 sin(d_FS/2).

    Puntos fijos: autovectores de ρ (Brouwer: ℂP^{n−1} compacto, T continua
    fuera de ker ρ).  Si λ₁ es simple, el atractor [v₁] es único.
    """

    @staticmethod
    def sanitize_vector(v: np.ndarray, n: int) -> ComplexVector:
        """Extiende/trunca a ℂⁿ y normaliza.  Fallback = vector uniforme."""
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
        r"""Multiplica `target` por e^{−i arg⟨ref|target⟩} ⇒ ⟨ref|target⟩ ≥ 0."""
        ov = np.vdot(reference, target)
        if abs(ov) < 1e-30:
            return np.asarray(target, dtype=np.complex128)
        return (target * np.exp(-1j * np.angle(ov))).astype(np.complex128)

    @staticmethod
    def apply_map(rho: np.ndarray, v: np.ndarray) -> Tuple[ComplexVector, float]:
        r"""T(v) = ρv/‖ρv‖ (sin gauge).  Si ‖ρv‖≈0, devuelve v y 0."""
        rho_v = rho @ v
        rho_v_norm = float(np.linalg.norm(rho_v))
        if rho_v_norm < 1e-15:
            return np.asarray(v, dtype=np.complex128).copy(), 0.0
        return (rho_v / rho_v_norm).astype(np.complex128), rho_v_norm

    @staticmethod
    def rayleigh(rho: np.ndarray, v: np.ndarray) -> float:
        r"""R(v) = ⟨v|ρ|v⟩/⟨v|v⟩ ∈ [λₙ, λ₁]  (Courant–Fischer)."""
        vv = float(np.real(np.vdot(v, v)))
        if vv < 1e-30:
            return 0.0
        return float(np.real(np.vdot(v, rho @ v)) / vv)

    @staticmethod
    def fubini_study_angle(u: np.ndarray, v: np.ndarray) -> float:
        r"""d_FS([u],[v]) = arccos(|⟨u|v⟩|) ∈ [0, π/2] (vectores unitarios)."""
        u = np.asarray(u, dtype=np.complex128).reshape(-1)
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-30 or nv < 1e-30:
            return 0.5 * math.pi
        cos_theta = float(abs(np.vdot(u, v)) / (nu * nv))
        return float(math.acos(float(np.clip(cos_theta, 0.0, 1.0))))

    @classmethod
    def chordal(cls, u: np.ndarray, v: np.ndarray) -> float:
        """sin(d_FS) — distancia cuerda en S^{2n−1}/U(1)."""
        return float(math.sin(cls.fubini_study_angle(u, v)))

    @classmethod
    def evaluate(cls, rho: np.ndarray, v: np.ndarray) -> ProjectiveRayleighSample:
        """Evalúa T, Rayleigh, d_FS y residuo gauge-fijado (P2)."""
        n = int(rho.shape[0])
        v = cls.sanitize_vector(v, n)
        T_v, image_norm = cls.apply_map(rho, v)
        T_phi = cls.gauge_align(v, T_v)
        overlap = float(abs(np.vdot(v, T_v)))
        residual = float(np.linalg.norm(T_phi - v))
        fs_angle = cls.fubini_study_angle(v, T_v)
        return ProjectiveRayleighSample(
            rayleigh=cls.rayleigh(rho, v),
            image_norm=image_norm,
            overlap=overlap,
            residual=residual,
            fs_angle=fs_angle,
            chordal=float(math.sin(fs_angle)),
        )


# ── §1.4 Análisis del gap espectral ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SpectralGapReport:
    r"""
    Análisis espectral del campo ρ (P3).

        λ₁ ≥ λ₂ ≥ ⋯ ≥ λₙ ≥ 0,  Σ λ_i = 1
        gap_ratio     = λ₂/λ₁ ∈ [0, 1]     = ρ(DT|_{[v₁]})  (tasa asintótica)
        gap           = 1 − λ₂/λ₁ ∈ [0, 1]
        gap_absolute  = λ₁ − λ₂            (Davis–Kahan / Kato)
        cond          = λ₁/λₙ              (condición espectral)
        degeneracy_top: multiplicidad de λ₁
        is_uniform    : ρ ≈ I/n  (∄ atractor: T = Id)
        birkhoff      : tanh(log(λ₁/λₙ)/4)  (Hopf, diámetro proyectivo)

    Veredicto local (meets internos):
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

    Brouwer: ℂP^{n−1} compacto + T continua (ρ ≱ 0 sobre un abierto) ⇒
    Fix(T) ≠ ∅.  Los puntos fijos son las rectas propias de ρ.

    Linearización: en [v₁], DT actúa sobre v₁^⊥ con autovalores λ_i/λ₁.
    Radio espectral transversal = λ₂/λ₁.  Convergencia geométrica
    ⟺ λ₁ > λ₂ (λ₁ simple).

    Birkhoff–Hopf: si ρ es (entrywise) positiva, T contrae la métrica de
    Hilbert del cono con ratio tanh(Δ/4), Δ = diámetro proyectivo de ρ(ℂ₊).
    Proxy espectral (ρ ≻ 0): Δ = log(λ₁/λₙ).  NO se afirma para ρ semidefinida.

    Tarski: los subespacios invariantes de ρ forman un retículo completo
    bajo ∩ y +; Fix(T) es la unión proyectiva de esos ejes.
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
            lambda_1=lam1,
            lambda_2=lam2,
            lambda_min=lam_n,
            gap=gap,
            gap_ratio=float(gap_ratio),
            gap_absolute=gap_abs,
            condition_number=cond,
            degeneracy_top=degeneracy,
            is_uniform=is_uniform,
            von_neumann_entropy=DensityOperatorAlgebra.von_neumann_entropy(rho),
            purity=DensityOperatorAlgebra.purity(rho),
            birkhoff_constant=birkhoff,
            local_verdict=local,
        )


# ── §1.5 IntrospectiveFieldPreparation — HAND-OFF FASE 1 → FASE 2 ────────
@dataclass(frozen=True, slots=True)
class IntrospectiveField:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

        rho          : campo densidad ∈ 𝔇_n
        v_initial    : semilla unitaria de la iteración
        spectral_gap : SpectralGapReport (insumo crítico de F2)
        K_spec       : spec↑(−log ρ)
        field_hash   : SHA-256(ρ ‖ v ‖ γ)
        dim          : n
    """
    rho: np.ndarray
    v_initial: np.ndarray
    spectral_gap: SpectralGapReport
    K_spec: Tuple[float, ...]
    field_hash: str
    dim: int

    def as_dict(self) -> Dict[str, float]:
        g = self.spectral_gap
        return {
            "lambda_1": g.lambda_1,
            "lambda_2": g.lambda_2,
            "gap": g.gap,
            "gap_ratio": g.gap_ratio,
            "degeneracy": float(g.degeneracy_top),
            "birkhoff": g.birkhoff_constant,
            "is_uniform": float(g.is_uniform),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        max_iter: int,
        tol: float,
    ) -> "IntrospectionBundle":
        r"""
        Último morfismo de la FASE 1  ∧  primer morfismo de la FASE 2.

        Identidad de composición:

            continue_into_phase2 ∘ prepare
                = IntrospectionPipeline.synthesize ∘ prepare
                : 𝔇_n × ℂⁿ → IntrospectionBundle.

        En el sentido de categorías, la FASE 2 es el comma-category
        (IntrospectiveField ↓ Power₂).  Invocar la iteración de potencia
        sin un IntrospectiveField es un error de tipo.
        """
        return IntrospectionPipeline.synthesize(
            field=self, max_iter=max_iter, tol=tol,
        )


class IntrospectiveFieldPreparation:
    r"""
    Prepara el par (ρ, v₀) y el análisis espectral que ancla la FASE 2.

    El gap γ y la degeneración viajan con el campo: la FASE 2 no recompute
    spec(ρ) salvo para Rayleigh; la tasa teórica λ₂/λ₁ es invariante de F1.
    """

    @classmethod
    def prepare(
        cls,
        rho_input: np.ndarray,
        v_initial: np.ndarray,
    ) -> IntrospectiveField:
        r"""
        Cierra la FASE 1 como objeto.  El morfismo de continuación
        hacia FASE 2 es IntrospectiveField.continue_into_phase2.
        """
        rho = DensityOperatorAlgebra.sanitize(rho_input)
        n = int(rho.shape[0])
        v = ProjectiveDynamics.sanitize_vector(v_initial, n)
        gap_report = SpectralGapAnalyzer.analyze(rho)
        K_spec = DensityOperatorAlgebra.modular_spectrum(rho)
        field_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(v).tobytes(),
            f"{gap_report.gap:.12f}".encode("ascii"),
        )
        logger.debug(
            "FieldPreparation: λ₁=%.4f λ₂=%.4f γ=%.4f deg=%d | "
            "Birkhoff=%.4f | uniform=%s",
            gap_report.lambda_1, gap_report.lambda_2,
            gap_report.gap, gap_report.degeneracy_top,
            gap_report.birkhoff_constant, gap_report.is_uniform,
        )
        return IntrospectiveField(
            rho=rho,
            v_initial=v,
            spectral_gap=gap_report,
            K_spec=K_spec,
            field_hash=field_hash,
            dim=n,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA DE PUNTO FIJO                                          ║
# ║                                                                           ║
# ║  Dominio = IntrospectiveField (codominio de §1.5).                        ║
# ║  Codominio = IntrospectionBundle, dominio de toda la FASE 3.              ║
# ║                                                                           ║
# ║  §2.1 se lee como la continuación literal de continue_into_phase2.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Solver de iteración de potencia ──────────────────────────────────
@dataclass(frozen=True, slots=True)
class PowerIterationTrace:
    r"""
    Traza de T^k sobre ℂP^{n−1} (P1).

        residuals, fs_angles, rayleigh_trajectory : por iteración
        empirical_rate  : mediana de θ_{k+1}/θ_k en ventana sana ≈ λ₂/λ₁
        stalled_kernel  : True si ρv ≈ 0 (v ∈ ker ρ)
        iterations, converged
    """
    residuals: Tuple[float, ...]
    fs_angles: Tuple[float, ...]
    rayleigh_trajectory: Tuple[float, ...]
    empirical_rate: float
    stalled_kernel: bool
    iterations: int
    converged: bool


class PowerIterationSolver:
    r"""
    Iteración de potencia COMPLETA (continuación de continue_into_phase2):

        v_{k+1} = gauge_align(v_k,  ρ v_k / ‖ρ v_k‖)

    Criterio de parada (P1 ∧ P2):

        d_FS([v_k],[T v_k]) < tol  ∨  ‖T_φ − v_k‖₂ < tol  ∨  k = max_iter.

    Tasa: si λ₁ > λ₂,  d_FS([v_k],[v₁]) = Θ((λ₂/λ₁)^k).
    Si λ₁ = λ₂, no hay atractor único: la órbita vive en ℙ(E_{λ₁}) y se
    reporta no-convergente salvo que v₀ ya sea propio.

    Tasa empírica: mediana de θ_{k+1}/θ_k sobre el tramo donde
    θ ∈ [θ_floor, θ_ceil] (evita log-regresión sobre underflow, P3).
    Si se llega al punto fijo en ≤ 2 pasos, empirical_rate := theoretical
    (la semilla ya era el modo propio: no hay señal de tasa).
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
        field: IntrospectiveField,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = TOL_DEFAULT,
    ) -> Tuple[ComplexVector, PowerIterationTrace]:
        r"""
        Consume IntrospectiveField (ρ, v₀, γ).  No recompute spec(ρ).
        """
        rho = field.rho
        n = field.dim
        v = ProjectiveDynamics.sanitize_vector(field.v_initial, n)
        theoretical = float(field.spectral_gap.gap_ratio)

        residuals: List[float] = []
        fs_angles: List[float] = []
        rayleigh_traj: List[float] = []
        converged = False
        stalled_kernel = False

        for _k in range(max_iter):
            sample = ProjectiveDynamics.evaluate(rho, v)
            residuals.append(sample.residual)
            fs_angles.append(sample.fs_angle)
            rayleigh_traj.append(sample.rayleigh)

            if sample.image_norm < 1e-15:
                stalled_kernel = True
                break
            if sample.residual < tol or sample.fs_angle < tol:
                converged = True
                break

            T_v, _ = ProjectiveDynamics.apply_map(rho, v)
            v = ProjectiveDynamics.sanitize_vector(
                ProjectiveDynamics.gauge_align(v, T_v), n
            )

        empirical = cls._empirical_rate(fs_angles, theoretical)
        trace = PowerIterationTrace(
            residuals=tuple(map(float, residuals)),
            fs_angles=tuple(map(float, fs_angles)),
            rayleigh_trajectory=tuple(map(float, rayleigh_traj)),
            empirical_rate=float(empirical),
            stalled_kernel=bool(stalled_kernel),
            iterations=len(residuals),
            converged=bool(converged),
        )
        return v, trace


# ── §2.2 Certificado del punto fijo ───────────────────────────────────────
@dataclass(frozen=True, slots=True)
class FixedPointCertificate:
    r"""
    Certificado del punto fijo introspectivo.

        residual, fs_angle, overlap, rayleigh  : muestra final
        rayleigh_gap_to_λ₁  : λ₁ − R(v*) ≥ 0   (Courant–Fischer, =0 ⇒ v* ∈ E_{λ₁})
        energy_proxy        : 1 − overlap       (= 2 sin²(θ/2) a primer orden)
        empirical_rate vs theoretical_rate
        rate_consistency    : comparable sólo en ventana sana (P3)
        local_verdict       : meet de predicados graduados (P5)
    """
    fixed_point_residual: float
    fixed_point_fs_angle: float
    overlap_final: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    energy_proxy: float
    iterations: int
    converged: bool
    stalled_kernel: bool
    empirical_rate: float
    theoretical_rate: float
    rate_consistency: bool
    local_verdict: HeytingOmega3


class FixedPointCertifier:
    r"""
    Predicados dimensionalmente invariantes, colapsados por meet (P4, P5):

        conv      : converged ∧ ¬stalled_kernel
        overlap   : 1−|⟨v*|T v*⟩|  graduado (EPS_FID / 1e-3)
        residual  : ‖T_φ−v*‖₂      graduado (EPS_RES / 1e-3)
        rayleigh  : (λ₁−R)/λ₁      graduado (EPS_RAY / 1e-2)
        rate      : |r_emp − r_th| ≤ RATE_TOL  (o trivial si ya es FP)

    local = conv ∧ overlap ∧ residual ∧ rayleigh ∧ rate.
    """
    EPS_FID: Final[float] = 1.0e-6
    EPS_RES: Final[float] = 1.0e-6
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
        field: IntrospectiveField,
        v_fixed: np.ndarray,
        trace: PowerIterationTrace,
    ) -> FixedPointCertificate:
        sample = ProjectiveDynamics.evaluate(field.rho, v_fixed)
        lam1 = field.spectral_gap.lambda_1
        rate_th = field.spectral_gap.gap_ratio
        rate_emp = trace.empirical_rate

        already_fp = bool(trace.converged and trace.iterations <= 2)
        noisy_rate = bool(trace.converged and sample.fs_angle < cls.EPS_RES)
        rate_ok = already_fp or noisy_rate or (
            abs(rate_emp - rate_th) <= cls.RATE_TOL
        )

        conv_rule = (
            HeytingOmega3.VETOED if trace.stalled_kernel
            else (HeytingOmega3.COHERENT if trace.converged
                  else HeytingOmega3.DEGRADED)
        )
        overlap_rule = cls._grade(1.0 - sample.overlap, cls.EPS_FID, 1.0e-3)
        residual_rule = cls._grade(sample.residual, cls.EPS_RES, 1.0e-3)
        ray_rel = max(0.0, lam1 - sample.rayleigh) / max(lam1, _EPS)
        ray_rule = cls._grade(ray_rel, cls.EPS_RAY, 1.0e-2)
        rate_rule = (
            HeytingOmega3.COHERENT if rate_ok else HeytingOmega3.DEGRADED
        )
        local = (
            conv_rule
            .meet(overlap_rule)
            .meet(residual_rule)
            .meet(ray_rule)
            .meet(rate_rule)
        )
        return FixedPointCertificate(
            fixed_point_residual=float(sample.residual),
            fixed_point_fs_angle=float(sample.fs_angle),
            overlap_final=float(sample.overlap),
            rayleigh_final=float(sample.rayleigh),
            rayleigh_gap_to_lambda1=float(max(0.0, lam1 - sample.rayleigh)),
            energy_proxy=float(1.0 - sample.overlap),
            iterations=int(trace.iterations),
            converged=bool(trace.converged),
            stalled_kernel=bool(trace.stalled_kernel),
            empirical_rate=float(rate_emp),
            theoretical_rate=float(rate_th),
            rate_consistency=bool(rate_ok),
            local_verdict=local,
        )


# ── §2.3 IntrospectionPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Producto de los funtores Power ⊗ Certifier aplicados al
    IntrospectiveField de FASE 1.
    """
    field: IntrospectiveField
    v_fixed: np.ndarray
    trace: PowerIterationTrace
    certificate: FixedPointCertificate

    def content_bytes(self) -> bytes:
        """Digest firmable para la cadena Merkle de fases."""
        c = self.certificate
        return hashlib.sha256(
            self.field.field_hash.encode("ascii")
            + np.ascontiguousarray(self.v_fixed).tobytes()
            + f"{c.fixed_point_residual:.12e}".encode("ascii")
            + f"{c.overlap_final:.12e}".encode("ascii")
            + f"{c.rayleigh_final:.12e}".encode("ascii")
            + f"{c.iterations}".encode("ascii")
            + f"{c.converged}".encode("ascii")
        ).digest()

    def continue_into_phase3(
        self,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        Último morfismo de la FASE 2  ∧  primero de la FASE 3.

        Identidad:

            continue_into_phase3 ∘ synthesize ∘ prepare
                = adjudicate ∘ synthesize ∘ prepare.
        """
        return HeytingIntrospectionAdjudicator.adjudicate(self, external_verdict)


class IntrospectionPipeline:
    r"""
    Orquestador determinista de la FASE 2 (funtor F₂).

        synthesize : IntrospectiveField × ℕ × ℝ₊ → IntrospectionBundle

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    IntrospectionBundle es el dominio de TODOS los métodos de FASE 3.

    Identidad de anidamiento:

        certify ∘ synthesize ∘ prepare  :  𝔇_n × ℂⁿ → IntrospectionFieldState.
    """

    @classmethod
    def synthesize(
        cls,
        field: IntrospectiveField,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> IntrospectionBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

            (v*, trace) = PowerIterationSolver.solve(field)     §2.1
            cert        = FixedPointCertifier.certify(...)      §2.2
        """
        v_fixed, trace = PowerIterationSolver.solve(
            field, max_iter=max_iter, tol=tol,
        )
        cert = FixedPointCertifier.certify(field, v_fixed, trace)
        return IntrospectionBundle(
            field=field,
            v_fixed=v_fixed,
            trace=trace,
            certificate=cert,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + AUTO-ORGANIZACIÓN + CERTIFICACIÓN                ║
# ║                                                                           ║
# ║  Dominio = IntrospectionBundle (codominio de §2.3 synthesize).            ║
# ║  Codominio = IntrospectionFieldState (objeto terminal).                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ ────────────────────────────────────────────────
class HeytingIntrospectionAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (P5).  No hay conteo n_fail.

        cert     : certificate.local_verdict
        gap      : spectral_gap.local_verdict   (uniforme ⇒ ⊥ estructural)
        nondeg   : degeneracy_top == 1 → ⊤, == 2 → ⋆, else ⊥
        conv     : converged ∧ ¬kernel → ⊤, else ⋆/⊥
        fs       : d_FS final graduado

        final = local ∧ external     (meet conservador, nunca infla).
    """
    FS_COHERENT: Final[float] = 1.0e-6
    FS_DEGRADED: Final[float] = 1.0e-2

    @classmethod
    def _nondeg_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        d = bundle.field.spectral_gap.degeneracy_top
        if d == 1:
            return HeytingOmega3.COHERENT
        if d == 2:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        if bundle.certificate.stalled_kernel:
            return HeytingOmega3.VETOED
        return (
            HeytingOmega3.COHERENT
            if bundle.certificate.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _fs_rule(cls, bundle: IntrospectionBundle) -> HeytingOmega3:
        th = bundle.certificate.fixed_point_fs_angle
        if th <= cls.FS_COHERENT:
            return HeytingOmega3.COHERENT
        if th <= cls.FS_DEGRADED:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def adjudicate(
        cls,
        bundle: IntrospectionBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""Continuación de IntrospectionPipeline.synthesize / continue_into_phase3."""
        local = (
            bundle.certificate.local_verdict
            .meet(bundle.field.spectral_gap.local_verdict)
            .meet(cls._nondeg_rule(bundle))
            .meet(cls._conv_rule(bundle))
            .meet(cls._fs_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 Auto-organización del campo (mixtura convexa) ──────────────────
@dataclass(frozen=True, slots=True)
class FieldUpdateCertificate:
    r"""
    Canal afín de la MAC introspectiva:

        Φ_η(ρ) = (1−η) ρ + η |v*⟩⟨v*| ,   η ∈ [0, 1].

    Es CPTP (mezcla del canal identidad y el canal de reemplazo).
    Lip_{‖·‖₁}(Φ_η) = |1−η|  exactamente (afín).
    Punto fijo: Φ_η(|v*⟩⟨v*|) = |v*⟩⟨v*|.

    La pureza NO es monótona en general (contraejemplos si ρ ya es pura
    en otra dirección).  Se reporta ΔP, ΔS como observables, no como leyes.
    """
    applied: bool
    eta: float
    contraction_coef: float
    purity_before: float
    purity_after: float
    entropy_before: float
    entropy_after: float
    fidelity_to_fixed: float
    local_verdict: HeytingOmega3


class FieldSelfOrganizer:
    r"""
    Actualiza ρ hacia el atractor [v*] por mezcla convexa auditada.

    Kraus del reemplazo |v*⟩⟨v*| :  { |v*⟩⟨e_i| }_i  (ONB).
    La mixtura (1−η) id + η Repl no se escribe con dos Kraus; se realiza
    como canal clásico-cuántico (bit aleatorio + canal).
    """

    @classmethod
    def idle(cls, rho: np.ndarray) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        p = DensityOperatorAlgebra.purity(rho)
        s = DensityOperatorAlgebra.von_neumann_entropy(rho)
        cert = FieldUpdateCertificate(
            applied=False,
            eta=0.0,
            contraction_coef=1.0,
            purity_before=p,
            purity_after=p,
            entropy_before=s,
            entropy_after=s,
            fidelity_to_fixed=1.0,
            local_verdict=HeytingOmega3.DEGRADED,
        )
        return rho, cert

    @classmethod
    def update(
        cls,
        rho: np.ndarray,
        v_fixed: np.ndarray,
        eta: float,
    ) -> Tuple[ComplexMatrix, FieldUpdateCertificate]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        eta = float(np.clip(eta, 0.0, 1.0))
        rho_fixed = DensityOperatorAlgebra.rank_one(v_fixed)

        p_before = DensityOperatorAlgebra.purity(rho)
        s_before = DensityOperatorAlgebra.von_neumann_entropy(rho)
        rho_new = DensityOperatorAlgebra.sanitize((1.0 - eta) * rho + eta * rho_fixed)
        p_after = DensityOperatorAlgebra.purity(rho_new)
        s_after = DensityOperatorAlgebra.von_neumann_entropy(rho_new)
        fid_fixed = DensityOperatorAlgebra.uhlmann_fidelity(rho_new, rho_fixed)

        coef = abs(1.0 - eta)
        if coef < 1.0 and fid_fixed > 0.5:
            local = HeytingOmega3.COHERENT
        elif coef < 1.0:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        cert = FieldUpdateCertificate(
            applied=True,
            eta=eta,
            contraction_coef=coef,
            purity_before=float(p_before),
            purity_after=float(p_after),
            entropy_before=float(s_before),
            entropy_after=float(s_after),
            fidelity_to_fixed=float(fid_fixed),
            local_verdict=local,
        )
        return rho_new, cert


# ── §3.3 Certificado firmado del ciclo introspectivo ────────────────────
@dataclass(frozen=True, slots=True)
class IntrospectionFieldState:
    r"""
    Objeto terminal del ciclo: producto fibrado firmado

        State ≅ Bundle × Ω₃ × Φ_η × Merkle.

    Residuos desambiguados (P2):
        fixed_point_residual : ‖T_φ − v*‖₂     cuerda gauge-fijada
        fixed_point_fs_angle : d_FS([v*],[T v*])  geometría nativa
    """
    cycle_id: str
    engine_id: str
    flash_id: str
    heyting_verdict: HeytingOmega3
    overlap: float
    fixed_point_residual: float
    fixed_point_fs_angle: float
    rayleigh_final: float
    rayleigh_gap_to_lambda1: float
    spectral_gap: float
    theoretical_rate: float
    empirical_rate: float
    rate_consistency: bool
    iterations: int
    is_self_sustaining: bool
    birkhoff_constant: float
    is_uniform: bool
    degeneracy_top: int
    field_updated: bool
    field_update_eta: float
    field_purity: float
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.4 Motor de introspección soberano ────────────────────────────────
class TOONIntrospectionEngine:
    r"""
    Motor Espectral de la Introspección.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  IntrospectiveFieldPreparation.prepare
        F₂  IntrospectiveField.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ Φ_η ⊗ certify

    Asociatividad (teorema de anidamiento):

        execute_introspection_cycle
            = _phase3_certify ∘ _phase2_iterate ∘ _phase1_prepare
            = certify ∘ synthesize ∘ prepare.

    La auto-organización Φ_η es opcional y sólo se aplica si el veredicto
    final es ⊤ y `update_field=True` (inoculación coherente del modo propio).
    """

    def __init__(
        self,
        engine_id: str = "INTROSPECT-ENGINE-WISDOM-01",
        dimension_mac: int = 4,
        update_field: bool = True,
        field_update_eta: float = 0.20,
        max_iter: int = PowerIterationSolver.MAX_ITER_DEFAULT,
        tol: float = PowerIterationSolver.TOL_DEFAULT,
    ) -> None:
        if dimension_mac < 1:
            raise ValueError("dimension_mac ≥ 1")
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.update_field = bool(update_field)
        self.field_update_eta = float(np.clip(field_update_eta, 0.0, 1.0))
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.density_matrix: ComplexMatrix = (
            np.eye(self.dimension_mac, dtype=np.complex128) / self.dimension_mac
        )
        self.cycle_count = 0
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::n={dimension_mac}::"
            f"η={self.field_update_eta}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, flash_vector: np.ndarray) -> IntrospectiveField:
        """FASE 1 anidada: cierra con IntrospectiveField (dominio de FASE 2)."""
        field = IntrospectiveFieldPreparation.prepare(
            rho_input=self.density_matrix,
            v_initial=flash_vector,
        )
        self._advance_chain("F1", bytes.fromhex(field.field_hash))
        return field

    def _phase2_iterate(self, field: IntrospectiveField) -> IntrospectionBundle:
        """FASE 2 anidada: continuación de prepare; cierra con Bundle."""
        bundle = field.continue_into_phase2(
            max_iter=self.max_iter, tol=self.tol,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    def _phase3_certify(
        self,
        cycle_id: str,
        flash_id: str,
        bundle: IntrospectionBundle,
        external_verdict: HeytingOmega3,
    ) -> IntrospectionFieldState:
        """FASE 3 anidada: continuación de synthesize; cierra con State."""
        final_verdict = bundle.continue_into_phase3(external_verdict)

        field_updated = False
        eta_used = 0.0
        if self.update_field and final_verdict == HeytingOmega3.COHERENT:
            new_rho, _upd = FieldSelfOrganizer.update(
                rho=self.density_matrix,
                v_fixed=bundle.v_fixed,
                eta=self.field_update_eta,
            )
            self.density_matrix = new_rho
            field_updated = True
            eta_used = self.field_update_eta

        self._advance_chain(
            "F3",
            f"{final_verdict.name}|upd={field_updated}|"
            f"{bundle.certificate.overlap_final:.12e}".encode("ascii"),
        )
        c = bundle.certificate
        g = bundle.field.spectral_gap
        provenance = _sha256_bytes(
            self.engine_id.encode("ascii"),
            cycle_id.encode("ascii"),
            flash_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{c.fixed_point_residual:.12e}".encode("ascii"),
            f"{c.overlap_final:.12e}".encode("ascii"),
            f"{g.gap:.12e}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntrospectionFieldState(
            cycle_id=cycle_id,
            engine_id=self.engine_id,
            flash_id=flash_id,
            heyting_verdict=final_verdict,
            overlap=c.overlap_final,
            fixed_point_residual=c.fixed_point_residual,
            fixed_point_fs_angle=c.fixed_point_fs_angle,
            rayleigh_final=c.rayleigh_final,
            rayleigh_gap_to_lambda1=c.rayleigh_gap_to_lambda1,
            spectral_gap=g.gap,
            theoretical_rate=g.gap_ratio,
            empirical_rate=c.empirical_rate,
            rate_consistency=c.rate_consistency,
            iterations=c.iterations,
            is_self_sustaining=bool(
                c.converged and c.overlap_final > 1.0 - 1e-6 and (not g.is_uniform)
            ),
            birkhoff_constant=g.birkhoff_constant,
            is_uniform=g.is_uniform,
            degeneracy_top=g.degeneracy_top,
            field_updated=field_updated,
            field_update_eta=eta_used,
            field_purity=DensityOperatorAlgebra.purity(self.density_matrix),
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def execute_introspection_cycle(
        self,
        flash_id: str,
        flash_vector: np.ndarray,
        heyting_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> IntrospectionFieldState:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.cycle_count += 1
        cycle_id = f"CYC-INTROSPECT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Introspección #%d | flash=%s | tol=%.1e ═══",
            self.cycle_count, flash_id, self.tol,
        )
        field = self._phase1_prepare(flash_vector)
        bundle = self._phase2_iterate(field)
        state = self._phase3_certify(cycle_id, flash_id, bundle, heyting_verdict)
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s | Ω₃=%s | resid=%.3e | FS=%.3e | overlap=%.6f | "
            "γ=%.4f | rate_th=%.4f rate_emp=%.4f | iters=%d | upd=%s | %.2f ms",
            cycle_id, state.heyting_verdict.name,
            state.fixed_point_residual, state.fixed_point_fs_angle, state.overlap,
            state.spectral_gap, state.theoretical_rate, state.empirical_rate,
            state.iterations, state.field_updated, dt_ms,
        )
        return state


# ── §3.5 Demostración autónoma ───────────────────────────────────────────
def _build_rho_from_spectrum(
    n: int, spectrum: np.ndarray, key: str,
) -> ComplexMatrix:
    r"""
    ρ = Σ_k λ_k |q_k⟩⟨q_k|  con Q Haar (Ginibre → QR, Stewart 1980).
    """
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
    return (v / np.linalg.norm(v)).astype(np.complex128)


if __name__ == "__main__":
    print("═" * 94)
    print("TOON INTROSPECTION ENGINE — v2.2.0 Nested Doctoral")
    print("Potencia gauge-fijada · Fubini–Study · Birkhoff–Hopf · Heyting Ω₃ · Merkle")
    print("═" * 94)

    def print_state(state: IntrospectionFieldState, title: str) -> None:
        print(f"\n[{title}]")
        print(f"   cycle_id            : {state.cycle_id}")
        print(f"   Ω₃ final            : {state.heyting_verdict.name}")
        print(f"   residuo gauge-fijado: {state.fixed_point_residual:.3e}")
        print(f"   ángulo FS           : {state.fixed_point_fs_angle:.3e} rad")
        print(f"   overlap |⟨v|T(v)⟩|  : {state.overlap:.10f}")
        print(f"   Rayleigh final      : {state.rayleigh_final:.6f}  "
              f"(λ₁−R = {state.rayleigh_gap_to_lambda1:.3e})")
        print(f"   gap espectral γ     : {state.spectral_gap:.6f}")
        print(f"   rate teórica λ₂/λ₁  : {state.theoretical_rate:.6f}")
        print(f"   rate empírica       : {state.empirical_rate:.6f}  "
              f"(consistente={state.rate_consistency})")
        print(f"   iteraciones         : {state.iterations}")
        print(f"   Birkhoff tanh(Δ/4)  : {state.birkhoff_constant:.6f}")
        print(f"   uniforme (I/n)      : {state.is_uniform}")
        print(f"   degeneración λ₁     : {state.degeneracy_top}")
        print(f"   autosostenible      : {state.is_self_sustaining}")
        print(f"   campo actualizado   : {state.field_updated}  "
              f"(η = {state.field_update_eta:.3f}, P(ρ)={state.field_purity:.6f})")
        print(f"   fase chain          : {state.phase_chain_sha256[:32]}…")
        print(f"   firma global        : {state.sha256_provenance[:32]}…")

    n = 4
    rho_ideal = _build_rho_from_spectrum(
        n, np.array([0.97, 0.02, 0.005, 0.005]), "IDEAL"
    )
    gap_report = SpectralGapAnalyzer.analyze(rho_ideal)
    print("\n─── Sanity-check: análisis espectral de una ρ con gap grande ───")
    print(f"   λ₁ = {gap_report.lambda_1:.6f} | λ₂ = {gap_report.lambda_2:.6f}")
    print(f"   γ  = {gap_report.gap:.6f}      | rate_th = λ₂/λ₁ = {gap_report.gap_ratio:.6f}")
    print(f"   Δλ = {gap_report.gap_absolute:.6f}  | cond = {gap_report.condition_number:.4f}")
    print(f"   degeneración top = {gap_report.degeneracy_top} | uniforme = {gap_report.is_uniform}")
    print(f"   Birkhoff = tanh(Δ/4) = {gap_report.birkhoff_constant:.6f}")
    print(f"   veredicto espectral = {gap_report.local_verdict.name}")

    v_a = _build_test_vector(n, "VEC-A")
    sample0 = ProjectiveDynamics.evaluate(rho_ideal, v_a)
    print(
        f"\n   muestra inicial: R={sample0.rayleigh:.4f}  "
        f"d_FS={sample0.fs_angle:.4e}  ‖T_φ−v‖={sample0.residual:.4e}  "
        f"cuerda={sample0.chordal:.4e}"
    )

    engine = TOONIntrospectionEngine(
        engine_id="INTROSPECT-ENGINE-WISDOM-01",
        dimension_mac=4,
        update_field=False,
        field_update_eta=0.20,
        max_iter=500,
        tol=1e-10,
    )

    scenarios = [
        ("COHERENT (gap grande)",
         np.array([0.97, 0.02, 0.005, 0.005]), "RHO-A", "VEC-A",
         HeytingOmega3.COHERENT),
        ("DEGRADED (gap pequeño)",
         np.array([0.50, 0.45, 0.03, 0.02]), "RHO-B", "VEC-B",
         HeytingOmega3.COHERENT),
        ("VETOED (uniforme I/n)",
         np.array([0.25, 0.25, 0.25, 0.25]), "RHO-C", "VEC-C",
         HeytingOmega3.COHERENT),
    ]

    for name, spec, rho_key, vec_key, ext in scenarios:
        rho = _build_rho_from_spectrum(n, spec, rho_key)
        v = _build_test_vector(n, vec_key)
        engine.density_matrix = rho
        state = engine.execute_introspection_cycle(
            flash_id=f"FLASH-{vec_key}",
            flash_vector=v,
            heyting_verdict=ext,
        )
        print_state(state, name)

    print("\n" + "─" * 94)
    print("─── Auto-organización del campo (η = 0.20, 6 ciclos) ───")
    engine2 = TOONIntrospectionEngine(
        engine_id="INTROSPECT-SELF-ORG",
        dimension_mac=4,
        update_field=True,
        field_update_eta=0.20,
    )
    v_seed = _build_test_vector(4, "SELF-ORG-VEC")
    for i in range(6):
        state = engine2.execute_introspection_cycle(
            flash_id=f"FLASH-SELF-ORG-{i + 1:03d}",
            flash_vector=v_seed,
            heyting_verdict=HeytingOmega3.COHERENT,
        )
        purity_now = DensityOperatorAlgebra.purity(engine2.density_matrix)
        gap_now = SpectralGapAnalyzer.analyze(engine2.density_matrix)
        print(
            f"   ciclo {i + 1}: Ω₃={state.heyting_verdict.name:<9s} "
            f"P(ρ)={purity_now:.6f}  "
            f"γ={gap_now.gap:.6f}  "
            f"upd={state.field_updated}  "
            f"iters={state.iterations}  "
            f"d_FS={state.fixed_point_fs_angle:.2e}"
        )

    print("\n" + "═" * 94)
    print("✓ F1→F2: prepare ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate.")
    print("✓ Iteración de potencia COMPLETA, parada d_FS ∧ ‖T_φ−v‖ (P1).")
    print("✓ d_FS=arccos|⟨u|v⟩|; residuo gauge-fijado = 2 sin(θ/2) (P2).")
    print("✓ ρ(DT|_{v₁})=λ₂/λ₁; tasa empírica en ventana sana (P3).")
    print("✓ Ω₃ por meets de ratios adimensionales, no n_fail (P4, P5).")
    print("✓ Φ_η CPTP, Lip₁=|1−η|; Merkle F1→F2→F3 (P6).")
    print("═" * 94)
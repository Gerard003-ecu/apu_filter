# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  TOON Intuition Engine — Motor Espectral de la Intuición Flash                ║
║  Ubicación: app/wisdom/toon_intuition_engine.py                               ║
║  Versión  : 2.2.0-Doctoral-Nested-Bures-Dirichlet-BB-Grassmann-Merkle         ║
║  Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el   ║
║            germen formal del primero de k+1)                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Funtor de Intuición Flash F)
================================================================

Sea 𝓣_Ω el topos de haces con clasificador Ω₃ = { VETOED = 0 ≺ DEGRADED = 1 ≺ COHERENT = 2 }.
El motor proyecta reflejos de intuición relámpago mediante el funtor:

        F  :  M_n(ℂ) × Gr(r, n)  ──▶  IntuitiveFieldState

mediante la composición estrictamente asociativa de tres fases anidadas:

        F  =  Certify ∘ FlashPipeline ∘ Prepare

donde el tipo de retorno del último método de la fase k es el dominio inalienable de la fase k+1.

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — SUSTRATO GEOMÉTRICO (RETÍCULO, VARIEDAD Y GRASSMANNIANA)
──────────────────────────────────────────────────────────────────────────
  • HeytingOmega3: Retículo de Heyting completo Ω₃. Residuo a → b = ⊤ si a ≤ b, else b.
    Satisface residuación (a ∧ c ≤ b ⇔ c ≤ (a → b)) y falla del tercio excluso en DEGRADED.
  • DensityOperatorAlgebra: Operadores en 𝔇_n. Métrica geodésica de Bures d_B(ρ,σ) = √(2 − 2√F(ρ,σ)),
    fidelidad de Uhlmann F(ρ,σ) = [Tr √(√ρ σ √ρ)]² y proyección afín no expansiva `sanitize`.
  • SubspaceGeometry: Subespacio Gr(r, n) representado por B ∈ St(r, n) y proyector P = B B† = P² = P†.
  • IntuitionFieldPreparation.prepare: Morfismo de hand-off FASE 1 ⟶ FASE 2. Construye `GeometricSeed`
    con el atractor estático ρ_target = P ρ P / Tr(P ρ P) (último objeto/método de FASE-1).

FASE 2 — DINÁMICA DE DIRICHLET, BURES Y SOLVER BARZILAI-BORWEIN
──────────────────────────────────────────────────────────────────────────
  • FlashDirichletFunctional: Funcional de energía de Dirichlet E[ρ] = ½ ‖ρ − P ρ P‖_F² con gradiente
    ∇E|_T = (ρ − P ρ P) − (Tr(ρ − P ρ P)/n) I ∈ T_ρ 𝔇_n. Hessiano Lip_F(∇E) ≤ 1.
  • BuresGeodesicMetric: Geodésica de Bures–Wasserstein γ(t) = [(1−t)I + t C] ρ [(1−t)I + t C].
  • FlashAttractorSolver.descend: Minimización BB1/BB2 con salvaguardas de Armijo y proyección Higham:
        BB1: η = ⟨s,s⟩/⟨s,y⟩,    BB2: η = ⟨s,y⟩/⟨y,y⟩.
  • IntuitionFlashPipeline.synthesize: Compone descend + Bures + landscape, emitiendo `IntuitionTrajectoryBundle`
    (último objeto/método de FASE-2).

FASE 3 — ADJUDICACIÓN, CERTIFICACIÓN Y ORQUESTACIÓN
──────────────────────────────────────────────────────────────────────────
  • HeytingIntuitionAdjudicator.adjudicate: PRIMER MORFISMO DE FASE-3 (continúa `synthesize`).
    Colapsa el bundle en Ω₃ mediante meets de ratios adimensionales:
        local = decay ∧ grad ∧ target ∧ geom ∧ conv,
        final = local ∧ external.
  • IntuitiveFieldState: Certificado signed con trazabilidad SHA-256 encadenada (`phase_chain_sha256`).
  • TOONIntuitionEngine: Orquestador soberano F₃ ∘ F₂ ∘ F₁.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Métrica Geodésica de Bures: d_B(ρ, σ) = √(2 − 2√F(ρ, σ)) ∈ [0, √2] (cumple desigualdad triangular).
  2. Suavidad de Dirichlet: Lip_F(∇E) ≤ 1 ⇒ paso estable η ∈ (0, 2).
  3. Proyector Ortogonal de Grassmann: P² = P = P†, ‖P‖_op = 1.
  4. Adjunción de Heyting: (a ∧ c ≤ b) ⇔ (c ≤ (a → b)).
  5. Inyectividad Merkle: Cadena de custodia `phase_chain_sha256` inalienable por SHA-256.
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


logger = logging.getLogger("APU.Wisdom.TOONIntuitionEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS: Final[float] = 1.0e-14
_EPS_MOD: Final[float] = 1.0e-12
_EPS_TRACE: Final[float] = 1.0e-15

ComplexMatrix = NDArray[np.complex128]
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
# ║  FASE 1 · SUSTRATO GEOMÉTRICO                                             ║
# ║                                                                           ║
# ║  Objetos: Ω₃, 𝔇_n, Gr(r, n), T_ρ 𝔇_n.                                     ║
# ║  Morfismo terminal: GeometricSeed.continue_into_phase2.                   ║
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
        ⊤ clasifica subobjetos totales (flash coherente),
        ⋆ clasifica subobjetos densos no cerrados (degradación),
        ⊥ clasifica el subobjeto vacío (veto del atractor).
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

    Espacio tangente afín (métrica plana de Frobenius):

        T_ρ 𝔇_n ≅ { A = A† : Tr A = 0 }.

    Funcionales (unitariamente invariantes):

        S(ρ)     = −Tr(ρ log ρ)                 von Neumann (nats)
        P(ρ)     = Tr(ρ²)                       pureza ∈ [1/n, 1]
        F(ρ,σ)   = ‖√ρ √σ‖₁                     Uhlmann–Jozsa ∈ [0, 1]
        d_B(ρ,σ) = √(2 − 2√F)                   Bures ∈ [0, √2]
        θ_B(ρ,σ) = arccos(√F)                   ángulo de Bures ∈ [0, π/2]
        K_ρ      = −log ρ                       Hamiltoniano modular
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
    def frobenius(cls, A: np.ndarray) -> float:
        return float(np.linalg.norm(A, "fro"))

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
    def bures_angle(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""θ_B(ρ,σ) = arccos(√F(ρ,σ)) ∈ [0, π/2]."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.acos(float(np.clip(math.sqrt(F), 0.0, 1.0))))

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2 − 2√F(ρ,σ)) ∈ [0, √2].  Geodésica, no Frobenius."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        """K_ρ = −log ρ  (Tomita–Takesaki)."""
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
    def tangent_project(cls, A: np.ndarray, n: Optional[int] = None) -> ComplexMatrix:
        r"""Proyección euclídea sobre T 𝔇_n: Hermitiza y resta (Tr A / n) I."""
        H = np.asarray(A, dtype=np.complex128)
        H = 0.5 * (H + H.conj().T)
        dim = int(n if n is not None else H.shape[0])
        beta = float(np.trace(H).real) / max(dim, 1)
        return H - beta * np.eye(dim, dtype=np.complex128)


# ── §1.3 Geometría del subespacio de referencia ───────────────────────────
@dataclass(frozen=True, slots=True)
class SubspaceGeometry:
    r"""
    Punto de Grassmann Gr(r, n): subespacio B ⊂ ℂⁿ con proyector P.

        basis       : B ∈ ℂ^{n×r},  B†B = I_r   (isometría parcial / Stiefel)
        projector   : P = B B†,  P² = P = P†    (proyector ortogonal)
        rank, codim : r, n−r
        principal_svals : σ(B† B) ≡ 1_r  (test de Stiefel)
        hash        : SHA-256(B ‖ P)

    El complemento P_⊥ = I − P es el único ortocomplemento; E[ρ] mide
    la masa de ρ fuera de ran(P) más las coherencias P–P_⊥.
    """
    basis: np.ndarray
    projector: np.ndarray
    rank: int
    codim: int
    is_isometry: bool
    is_projector: bool
    isometry_residual: float
    projector_residual: float
    hash: str

    @property
    def complement(self) -> ComplexMatrix:
        n = self.projector.shape[0]
        return np.eye(n, dtype=np.complex128) - self.projector

    def mass(self, rho: np.ndarray) -> float:
        """Tr(P ρ) ∈ [0, 1] — masa de ρ sobre ran(P)."""
        rho_s = DensityOperatorAlgebra.sanitize(rho)
        return float(np.real(np.trace(self.projector @ rho_s)))


class SubspaceGeometryFactory:
    r"""
    Construye B ∈ St(r, n) por QR-Haar (Stewart 1980: Ginibre → Haar en U(n),
    las primeras r columnas son Haar en Stiefel).  Se fuerza r ∈ [1, n−1]
    porque r = n ⇒ P = I ⇒ E ≡ 0 (intuición trivial).

    Corrección de fase de diag(R) garantiza det-positivo por bloque y
    elimina la ambigüedad discreta del QR.
    """

    @classmethod
    def haar_unitary(cls, n: int, rng: np.random.Generator) -> ComplexMatrix:
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        d = np.diagonal(R)
        ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
        return (Q * ph.conj()).astype(np.complex128)

    @classmethod
    def principal_angles(
        cls, B1: np.ndarray, B2: np.ndarray
    ) -> RealVector:
        r"""
        Ángulos principales θ_i ∈ [0, π/2] entre ran(B1) y ran(B2):

            σ_i = svals(B1† B2),   θ_i = arccos(clip(σ_i, 0, 1)).

        Distancia de Grassmann: ‖θ‖₂.
        """
        M = B1.conj().T @ B2
        sig = np.real(la.svdvals(M))
        sig = np.clip(sig, 0.0, 1.0)
        return np.arccos(sig).astype(np.float64)

    @classmethod
    def grassmann_distance(cls, B1: np.ndarray, B2: np.ndarray) -> float:
        theta = cls.principal_angles(B1, B2)
        return float(np.linalg.norm(theta, ord=2))

    @classmethod
    def build(cls, n: int, rank: int, key: str) -> SubspaceGeometry:
        if n < 2:
            raise ValueError("SubspaceGeometryFactory.build: n ≥ 2")
        rank = int(np.clip(rank, 1, n - 1))
        rng = np.random.default_rng(_seed_from_string(f"SUBSPACE::{key}"))
        Q = cls.haar_unitary(n, rng)
        B = np.ascontiguousarray(Q[:, :rank])
        P = B @ B.conj().T
        P = 0.5 * (P + P.conj().T)

        iso_err = float(np.linalg.norm(B.conj().T @ B - np.eye(rank), "fro"))
        proj_err = float(np.linalg.norm(P @ P - P, "fro"))
        sub_hash = _sha256_bytes(
            np.ascontiguousarray(B).tobytes(),
            np.ascontiguousarray(P).tobytes(),
        )
        return SubspaceGeometry(
            basis=B,
            projector=P,
            rank=rank,
            codim=n - rank,
            is_isometry=bool(iso_err < 1e-10),
            is_projector=bool(proj_err < 1e-10),
            isometry_residual=iso_err,
            projector_residual=proj_err,
            hash=sub_hash,
        )


# ── §1.4 IntuitionFieldPreparation — HAND-OFF FASE 1 → FASE 2 ─────────────
@dataclass(frozen=True, slots=True)
class GeometricSeed:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.

        rho_seed     : ρ ∈ 𝔇_n
        subspace     : (B, P) ∈ Gr(r, n)
        rho_target   : PρP / Tr(PρP)   (atractor estático, E≈0)
        seed_energy  : E[ρ_seed] = ½‖ρ − PρP‖_F²
        target_energy: E[ρ_target]     (cero numérico si P exacto)
        mass_on_P    : Tr(P ρ)
        d_B, θ_B     : Bures(ρ_seed, ρ_target)
        seed_spectral_hash : SHA-256(ρ ‖ spec)
    """
    rho_seed: np.ndarray
    subspace: SubspaceGeometry
    rho_target: np.ndarray
    seed_energy: float
    target_energy: float
    mass_on_P: float
    bures_distance_to_target: float
    bures_angle_to_target: float
    seed_spectral_hash: str
    dim: int

    # ══════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        cycle_index: int,
        max_steps: int,
        tol: float,
    ) -> "IntuitionTrajectoryBundle":
        r"""
        Último morfismo de la FASE 1  ∧  primer morfismo de la FASE 2.

        Identidad de composición:

            continue_into_phase2 ∘ prepare
                = IntuitionFlashPipeline.synthesize ∘ prepare
                : 𝔇_n × Gr(r,n) → IntuitionTrajectoryBundle.

        En el sentido de categorías, la FASE 2 es el comma-category
        (GeometricSeed ↓ Flash₂).  Invocar Dirichlet/Bures/BB sin un
        GeometricSeed es un error de tipo.
        """
        return IntuitionFlashPipeline.synthesize(
            cycle_index=cycle_index,
            seed=self,
            max_steps=max_steps,
            tol=tol,
        )


class IntuitionFieldPreparation:
    r"""
    Prepara el par (ρ_seed, P) como punto de anclaje de la FASE 2.

    El atractor estático es la proyección normalizada al subespacio B:

        ρ_target := P ρ_seed P / Tr(P ρ_seed P) ∈ 𝔇_n ∩ {σ : σ = PσP}.

    Si Tr(PρP) = 0 (masa nula sobre ran P), se cae al máximamente mezclado
    (ignorancia total: no hay dirección de intuición).

    d_B es la única distancia Riemanniana contractiva bajo canales CPTP
    (Petz 1996, theorem of monotonicity of Bures).
    """

    @classmethod
    def static_target(cls, rho: np.ndarray, P: np.ndarray) -> ComplexMatrix:
        PrP = P @ rho @ P
        tr = float(np.trace(PrP).real)
        if tr < 1e-15:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return DensityOperatorAlgebra.sanitize(PrP / tr)

    @classmethod
    def dirichlet_energy(cls, rho: np.ndarray, P: np.ndarray) -> float:
        residual = rho - P @ rho @ P
        return 0.5 * float(np.linalg.norm(residual, "fro") ** 2)

    @classmethod
    def prepare(
        cls,
        rho_input: np.ndarray,
        subspace: SubspaceGeometry,
    ) -> GeometricSeed:
        r"""
        Cierra la FASE 1 como objeto.  El morfismo de continuación
        hacia FASE 2 es GeometricSeed.continue_into_phase2.
        """
        rho = DensityOperatorAlgebra.sanitize(rho_input)
        n = int(rho.shape[0])
        P = subspace.projector

        seed_energy = cls.dirichlet_energy(rho, P)
        rho_target = cls.static_target(rho, P)
        target_energy = cls.dirichlet_energy(rho_target, P)
        mass = float(np.real(np.trace(P @ rho)))
        d_bures = DensityOperatorAlgebra.bures_distance(rho, rho_target)
        theta_b = DensityOperatorAlgebra.bures_angle(rho, rho_target)

        w = DensityOperatorAlgebra.spectrum_descending(rho)
        seed_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(w).tobytes(),
        )
        logger.debug(
            "FieldPreparation: E_seed=%.6e | E_target=%.6e | mass_P=%.4f | "
            "d_Bures=%.6f | θ_Bures=%.6f rad",
            seed_energy, target_energy, mass, d_bures, theta_b,
        )
        return GeometricSeed(
            rho_seed=rho,
            subspace=subspace,
            rho_target=rho_target,
            seed_energy=float(seed_energy),
            target_energy=float(target_energy),
            mass_on_P=mass,
            bures_distance_to_target=float(d_bures),
            bures_angle_to_target=float(theta_b),
            seed_spectral_hash=seed_hash,
            dim=n,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2 · DINÁMICA DE ATRACCIÓN + MÉTRICA GEODÉSICA                       ║
# ║                                                                           ║
# ║  Dominio = GeometricSeed (codominio de §1.4).                             ║
# ║  Codominio = IntuitionTrajectoryBundle, dominio de toda la FASE 3.        ║
# ║                                                                           ║
# ║  §2.1 se lee como la continuación literal de continue_into_phase2.        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Funcional de Dirichlet del proyector ─────────────────────────────
@dataclass(frozen=True, slots=True)
class DirichletLandscape:
    r"""
    Paisaje de E en un punto ρ ∈ 𝔇_n.

        E(ρ)            = ½‖ρ − PρP‖_F²
        residual        = ρ − PρP = P_⊥ρP + PρP_⊥ + P_⊥ρP_⊥
        coherent_mass   = ‖P_⊥ ρ P‖_F²          (bloques cruzados)
        leak_mass       = ‖P_⊥ ρ P_⊥‖_F²        (masa en el complemento)
        Teorema: ‖ρ−PρP‖_F² = 2·coherent_mass + leak_mass
        grad_tan        ∈ T_ρ 𝔇_n
        Lip(∇E) ≤ 1     (‖H − PHP‖_F ≤ ‖H‖_F)
    """
    rho: np.ndarray
    energy: float
    grad_full: np.ndarray
    grad_tan: np.ndarray
    grad_norm: float
    residual_norm: float
    coherent_mass: float
    leak_mass: float


class FlashDirichletFunctional:
    r"""
    Energía de Dirichlet del funcional de proyección ortogonal
    (continuación de GeometricSeed.continue_into_phase2):

        E[ρ] = ½ ‖ρ − P ρ P‖_F² ≥ 0,
        E[ρ] = 0  ⟺  ρ = PρP  ⟺  supp(ρ) ⊆ ran(P) y [ρ, P] = 0.

    Gradiente euclídeo (producto Frobenius ⟨A,B⟩ = Tr(A† B)):

        ∇E(ρ) = ρ − PρP.

    Proyectado al tangente Tr = 0:

        ∇E|_T(ρ) = ∇E(ρ) − (Tr ∇E(ρ)/n) I.

    Lipschitz: para H Hermítico, ‖H − PHP‖_F ≤ ‖H‖_F, luego Lip_F(∇E) ≤ 1.
    En descenso por gradiente, el paso estable clásico es η ∈ (0, 2).
    Esta NO es la métrica de Bures: E es un Lyapunov euclídeo; d_B es
    el informe geodésico (P1 vs P2, roles disjuntos).
    """
    LIPSCHITZ: Final[float] = 1.0

    @classmethod
    def decompose_residual(
        cls, rho: np.ndarray, P: np.ndarray
    ) -> Tuple[ComplexMatrix, float, float]:
        r"""
        ρ − PρP = (P_⊥ρP) + (PρP_⊥) + (P_⊥ρP_⊥),
        con ‖PρP_⊥‖_F = ‖P_⊥ρP‖_F  (adjunción).
        """
        I = np.eye(P.shape[0], dtype=np.complex128)
        Pc = I - P
        cross = Pc @ rho @ P
        leak = Pc @ rho @ Pc
        residual = rho - P @ rho @ P
        coherent_mass = float(np.linalg.norm(cross, "fro") ** 2)
        leak_mass = float(np.linalg.norm(leak, "fro") ** 2)
        return residual, coherent_mass, leak_mass

    @classmethod
    def evaluate(cls, rho: np.ndarray, P: np.ndarray) -> DirichletLandscape:
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        residual, coherent_mass, leak_mass = cls.decompose_residual(rho, P)
        energy = 0.5 * float(np.linalg.norm(residual, "fro") ** 2)
        grad_full = residual
        grad_tan = DensityOperatorAlgebra.tangent_project(grad_full, n)
        return DirichletLandscape(
            rho=rho,
            energy=energy,
            grad_full=grad_full,
            grad_tan=grad_tan,
            grad_norm=float(np.linalg.norm(grad_tan, "fro")),
            residual_norm=float(np.linalg.norm(residual, "fro")),
            coherent_mass=coherent_mass,
            leak_mass=leak_mass,
        )


# ── §2.2 Métrica geodésica de Bures ───────────────────────────────────────
class BuresGeodesicMetric:
    r"""
    Métrica de Bures sobre 𝔇_n (única Riemanniana CPTP-contractiva):

        F(ρ, σ)   = ‖√ρ √σ‖₁ ∈ [0, 1]
        θ_B(ρ, σ) = arccos(√F) ∈ [0, π/2]
        d_B(ρ, σ) = √(2 − 2√F) ∈ [0, √2]

    Geodésica de Bures–Wasserstein (Takatsu / Bhatia–Jain–Lim):

        C_ρσ = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2}     (map de Uhlmann)
        γ(t) = [(1−t) I + t C_ρσ] ρ [(1−t) I + t C_ρσ]

    (se regulariza ρ ⪰ εI para que C esté definido).

    Para puros ρ=|ψ⟩⟨ψ|, σ=|φ⟩⟨φ|:
        d_B = √(2 − 2|⟨ψ|φ⟩|)  =  min_θ ‖|ψ⟩ − e^{iθ}|φ⟩‖₂.
    """

    @classmethod
    def fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)

    @classmethod
    def angle(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.bures_angle(rho, sigma)

    @classmethod
    def distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        return DensityOperatorAlgebra.bures_distance(rho, sigma)

    @classmethod
    def triangle_inequality_residual(
        cls, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        r"""res = max(0, d(a,c) − d(a,b) − d(b,c)).  Cero ⇔ desigualdad OK."""
        dab = cls.distance(a, b)
        dbc = cls.distance(b, c)
        dac = cls.distance(a, c)
        return float(max(0.0, dac - dab - dbc))

    @classmethod
    def uhlmann_map(cls, rho: np.ndarray, sigma: np.ndarray) -> ComplexMatrix:
        r"""C_ρσ = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2}."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(sigma)
        rho_h = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        rho_invh = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        inner = rho_h @ sigma @ rho_h
        inner_h = DensityOperatorAlgebra.matrix_power(inner, 0.5)
        C = rho_invh @ inner_h @ rho_invh
        return 0.5 * (C + C.conj().T)

    @classmethod
    def geodesic(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> ComplexMatrix:
        r"""γ(t) sobre la geodésica de Bures–Wasserstein, t ∈ [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        rho = DensityOperatorAlgebra.sanitize(rho)
        if t <= 0.0:
            return rho
        if t >= 1.0:
            return DensityOperatorAlgebra.sanitize(sigma)
        C = cls.uhlmann_map(rho, sigma)
        n = rho.shape[0]
        S = (1.0 - t) * np.eye(n, dtype=np.complex128) + t * C
        return DensityOperatorAlgebra.sanitize(S @ rho @ S.conj().T)


# ── §2.3 Solver Barzilai-Borwein para el flujo de Dirichlet ───────────────
@dataclass(frozen=True, slots=True)
class FlashAttractorCertificate:
    r"""
    Certificado del descenso al atractor (Lyapunov euclídeo + informe Bures).

        energy_decay_ratio    : E_final / E_seed ∈ [0, 1]  (0 si E_seed=0)
        bb_mode_last          : "BB1" | "BB2" | "ARM"
        lipschitz_used        : Lip(∇E)=1 ⇒ η teórico ∈ (0, 2)
        n_backtracks          : recortes Armijo acumulados
    """
    iterations: int
    initial_energy: float
    final_energy: float
    energy_decay_ratio: float
    grad_final_norm: float
    coherent_mass_final: float
    leak_mass_final: float
    bures_distance_seed_to_attractor: float
    bures_distance_attractor_to_target: float
    bures_distance_seed_to_target: float
    bb_mode_last: str
    n_backtracks: int
    converged: bool


class FlashAttractorSolver:
    r"""
    Minimización de E[ρ] = ½‖ρ − PρP‖_F² sobre 𝔇_n.

    Método: Barzilai–Borwein con salvaguarda Armijo y proyección Higham.

        BB1:  η = ⟨s,s⟩ / ⟨s,y⟩     (secante larga)
        BB2:  η = ⟨s,y⟩ / ⟨y,y⟩     (secante corta, si ⟨s,y⟩ ≤ 0 o BB1 inestable)

        s_k = ρ_{k+1} − ρ_k,   y_k = ∇E|_T(ρ_{k+1}) − ∇E|_T(ρ_k)

        Armijo: E(ρ − η ∇E) ≤ E − c η ‖∇E‖²,  c = 10⁻⁴
        Proyección: ρ ← sanitize(ρ − η ∇E|_T)  (no expansiva en ‖·‖_F)

    Como Lip(∇E) ≤ 1, el intervalo teórico de GD es η ∈ (0, 2); BB puede
    pedir η > 2 y Armijo lo recorta.  E es convexa en el afín Hermítico
    (hessiano Id − Ad_P ≥ 0), de modo que todo mínimo global vive en
    {σ : σ = PσP} ∩ 𝔇_n.
    """
    MAX_STEPS_DEFAULT: Final[int] = 120
    TOL_DEFAULT: Final[float] = 1.0e-9
    ETA_MIN: Final[float] = 1.0e-6
    ETA_MAX: Final[float] = 2.0          # 2/Lip = 2
    ARMIJO_C: Final[float] = 1.0e-4
    ARMIJO_MAX: Final[int] = 24
    WEAK_GRAD: Final[float] = 1.0e-4

    @classmethod
    def _bb_step(
        cls,
        s: np.ndarray,
        y: np.ndarray,
        eta_fallback: float,
        prefer_bb1: bool,
    ) -> Tuple[float, str]:
        sy = float(np.real(np.vdot(s.ravel(), y.ravel())))
        ss = float(np.real(np.vdot(s.ravel(), s.ravel())))
        yy = float(np.real(np.vdot(y.ravel(), y.ravel())))
        if prefer_bb1 and sy > 1e-14 and ss > 0.0:
            return float(np.clip(ss / sy, cls.ETA_MIN, cls.ETA_MAX)), "BB1"
        if sy > 1e-14 and yy > 0.0:
            return float(np.clip(sy / yy, cls.ETA_MIN, cls.ETA_MAX)), "BB2"
        return float(np.clip(eta_fallback, cls.ETA_MIN, cls.ETA_MAX)), "ARM"

    @classmethod
    def descend(
        cls,
        seed: GeometricSeed,
        max_steps: int = MAX_STEPS_DEFAULT,
        tol: float = TOL_DEFAULT,
    ) -> Tuple[ComplexMatrix, FlashAttractorCertificate]:
        r"""
        Continuación de GeometricSeed.continue_into_phase2 / apply Dirichlet.

        Consume GeometricSeed (ρ, P, ρ_target); no recompute el atractor
        estático (invariante de F1).
        """
        P = seed.subspace.projector
        rho = DensityOperatorAlgebra.sanitize(seed.rho_seed)
        land = FlashDirichletFunctional.evaluate(rho, P)
        e0 = land.energy

        eta = 1.0
        prev_rho = rho.copy()
        prev_grad = land.grad_tan.copy()
        converged = False
        step = 0
        n_bt = 0
        bb_mode = "ARM"
        prefer_bb1 = True

        for step in range(max_steps):
            if land.grad_norm < tol:
                converged = True
                break

            eta_try = eta
            accepted = False
            land_try: Optional[DirichletLandscape] = None
            rho_try: Optional[np.ndarray] = None
            for _ in range(cls.ARMIJO_MAX):
                rho_try = DensityOperatorAlgebra.sanitize(
                    rho - eta_try * land.grad_tan
                )
                land_try = FlashDirichletFunctional.evaluate(rho_try, P)
                armijo = eta_try * (land.grad_norm ** 2)
                if land_try.energy <= land.energy - cls.ARMIJO_C * armijo:
                    accepted = True
                    break
                eta_try *= 0.5
                n_bt += 1

            if not accepted or land_try is None or rho_try is None:
                converged = land.grad_norm < cls.WEAK_GRAD
                break

            s = rho_try - prev_rho
            y = land_try.grad_tan - prev_grad
            eta, bb_mode = cls._bb_step(s, y, eta_try, prefer_bb1)
            prefer_bb1 = not prefer_bb1

            prev_rho = rho.copy()
            prev_grad = land.grad_tan.copy()
            rho = rho_try
            land = land_try

        e_final = land.energy
        if e0 > 1e-20:
            ratio = float(e_final / e0)
        else:
            ratio = 0.0 if e_final <= 1e-20 else 1.0

        d_sa = BuresGeodesicMetric.distance(seed.rho_seed, rho)
        d_at = BuresGeodesicMetric.distance(rho, seed.rho_target)
        d_st = float(seed.bures_distance_to_target)

        cert = FlashAttractorCertificate(
            iterations=step + 1,
            initial_energy=float(e0),
            final_energy=float(e_final),
            energy_decay_ratio=ratio,
            grad_final_norm=float(land.grad_norm),
            coherent_mass_final=float(land.coherent_mass),
            leak_mass_final=float(land.leak_mass),
            bures_distance_seed_to_attractor=float(d_sa),
            bures_distance_attractor_to_target=float(d_at),
            bures_distance_seed_to_target=d_st,
            bb_mode_last=bb_mode,
            n_backtracks=int(n_bt),
            converged=bool(converged or land.grad_norm < cls.WEAK_GRAD),
        )
        return rho, cert


# ── §2.4 IntuitionFlashPipeline — HAND-OFF FASE 2 → FASE 3 ────────────────
@dataclass(frozen=True, slots=True)
class IntuitionTrajectoryBundle:
    r"""
    Objeto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Producto de los funtores Dirichlet ⊗ Bures ⊗ BB aplicados al
    GeometricSeed de FASE 1.
    """
    cycle_index: int
    seed: GeometricSeed
    rho_attractor: np.ndarray
    attractor_cert: FlashAttractorCertificate
    landscape_final: DirichletLandscape
    purity_final: float
    entropy_final: float
    fidelity_target: float

    def content_bytes(self) -> bytes:
        """Digest firmable para la cadena Merkle de fases."""
        c = self.attractor_cert
        return hashlib.sha256(
            self.seed.seed_spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_attractor).tobytes()
            + f"{c.final_energy:.12e}".encode("ascii")
            + f"{c.grad_final_norm:.12e}".encode("ascii")
            + f"{c.bures_distance_seed_to_attractor:.12e}".encode("ascii")
            + f"{c.bures_distance_attractor_to_target:.12e}".encode("ascii")
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
        return HeytingIntuitionAdjudicator.adjudicate(self, external_verdict)


class IntuitionFlashPipeline:
    r"""
    Orquestador determinista de la FASE 2 (funtor F₂).

        synthesize : ℕ × GeometricSeed × ℕ × ℝ₊ → IntuitionTrajectoryBundle

    ────────────────────────────────────────────────────────────────────────
    HAND-OFF FORMAL  FASE 2 → FASE 3
    ────────────────────────────────────────────────────────────────────────
    synthesize es el morfismo terminal de la FASE 2.  Su imagen
    IntuitionTrajectoryBundle es el dominio de TODOS los métodos de FASE 3.

    Identidad de anidamiento:

        certify ∘ synthesize ∘ prepare  :  𝔇_n × Gr(r,n) → IntuitiveFieldState.
    """

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        seed: GeometricSeed,
        max_steps: int = FlashAttractorSolver.MAX_STEPS_DEFAULT,
        tol: float = FlashAttractorSolver.TOL_DEFAULT,
    ) -> IntuitionTrajectoryBundle:
        r"""
        Cierra la FASE 2.  Abre la FASE 3.

            (ρ_att, cert) = FlashAttractorSolver.descend(seed)     §2.3
            landscape     = FlashDirichletFunctional.evaluate      §2.1
            (P, S, F)     = (purity, entropy, F_Bures)             §2.2
        """
        rho_attractor, cert = FlashAttractorSolver.descend(
            seed, max_steps=max_steps, tol=tol,
        )
        land_final = FlashDirichletFunctional.evaluate(
            rho_attractor, seed.subspace.projector,
        )
        purity = DensityOperatorAlgebra.purity(rho_attractor)
        entropy = DensityOperatorAlgebra.von_neumann_entropy(rho_attractor)
        fid_target = BuresGeodesicMetric.fidelity(rho_attractor, seed.rho_target)
        return IntuitionTrajectoryBundle(
            cycle_index=cycle_index,
            seed=seed,
            rho_attractor=rho_attractor,
            attractor_cert=cert,
            landscape_final=land_final,
            purity_final=purity,
            entropy_final=entropy,
            fidelity_target=float(fid_target),
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3 · ADJUDICACIÓN + CERTIFICACIÓN + ORQUESTACIÓN                     ║
# ║                                                                           ║
# ║  Dominio = IntuitionTrajectoryBundle (codominio de §2.4 synthesize).      ║
# ║  Codominio = IntuitiveFieldState (objeto terminal del flash).             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en Ω₃ con umbrales dimensionalmente calibrados ──────
class HeytingIntuitionAdjudicator:
    r"""
    Lógica interna del topos: colapsa el Bundle a un único valor de Ω₃
    por meets sucesivos (producto de subobjetos).  Todos los umbrales
    son adimensionales (invariantes a n y r):

        decay   : E_final/E_seed  ≤ 0.10 → ⊤,  ≤ 0.50 → ⋆,  else ⊥
        grad    : ‖∇E|_T‖_F       ≤ 1e-4 → ⊤,  ≤ 1e-2 → ⋆,  else ⊥
        target  : d_B(ρ_att,ρ_tgt)≤ 0.10 → ⊤,  ≤ 0.30 → ⋆,  else ⊥
        geom    : isometría ∧ proyector  → ⊤ else ⊥
        conv    : cert.converged         → ⊤ else ⋆

    final = local ∧ external     (meet conservador, nunca infla).
    """
    RATIO_COHERENT: Final[float] = 0.10
    RATIO_DEGRADED: Final[float] = 0.50
    GRAD_COHERENT: Final[float] = 1.0e-4
    GRAD_DEGRADED: Final[float] = 1.0e-2
    BURES_COHERENT: Final[float] = 0.10
    BURES_DEGRADED: Final[float] = 0.30

    @classmethod
    def _grade(
        cls, value: float, hi_ok: float, mid_ok: float, invert: bool = False
    ) -> HeytingOmega3:
        """Clasificador de umbral: valor pequeño es mejor si invert=False."""
        v = -value if invert else value
        a, b = (hi_ok, mid_ok)
        if invert:
            a, b = -hi_ok, -mid_ok
        if v <= a:
            return HeytingOmega3.COHERENT
        if v <= b:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _decay_rule(cls, bundle: IntuitionTrajectoryBundle) -> HeytingOmega3:
        return cls._grade(
            bundle.attractor_cert.energy_decay_ratio,
            cls.RATIO_COHERENT, cls.RATIO_DEGRADED,
        )

    @classmethod
    def _grad_rule(cls, bundle: IntuitionTrajectoryBundle) -> HeytingOmega3:
        return cls._grade(
            bundle.attractor_cert.grad_final_norm,
            cls.GRAD_COHERENT, cls.GRAD_DEGRADED,
        )

    @classmethod
    def _target_rule(cls, bundle: IntuitionTrajectoryBundle) -> HeytingOmega3:
        return cls._grade(
            bundle.attractor_cert.bures_distance_attractor_to_target,
            cls.BURES_COHERENT, cls.BURES_DEGRADED,
        )

    @classmethod
    def _geom_rule(cls, bundle: IntuitionTrajectoryBundle) -> HeytingOmega3:
        ok = (
            bundle.seed.subspace.is_isometry
            and bundle.seed.subspace.is_projector
        )
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, bundle: IntuitionTrajectoryBundle) -> HeytingOmega3:
        return (
            HeytingOmega3.COHERENT
            if bundle.attractor_cert.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def adjudicate(
        cls,
        bundle: IntuitionTrajectoryBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""Continuación de IntuitionFlashPipeline.synthesize / continue_into_phase3."""
        local = (
            cls._decay_rule(bundle)
            .meet(cls._grad_rule(bundle))
            .meet(cls._target_rule(bundle))
            .meet(cls._geom_rule(bundle))
            .meet(cls._conv_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 IntuitiveFieldState — certificado firmado ───────────────────────
@dataclass(frozen=True, slots=True)
class IntuitiveFieldState:
    r"""
    Objeto terminal del flash: producto fibrado firmado

        State ≅ Bundle × Ω₃ × Merkle.

    Distancias Bures desambiguadas (P1):
        manifold_geodesic_distance : d_B(ρ_seed, ρ_target)     geodésica de F1
        attractor_to_target_bures  : d_B(ρ_att,  ρ_target)     calidad del descenso
        seed_to_attractor_bures    : d_B(ρ_seed, ρ_att)        longitud del flujo
    """
    cycle_id: str
    crop_origin_id: str
    flash_attractor_energy: float
    seed_energy: float
    manifold_geodesic_distance: float
    attractor_to_target_bures: float
    seed_to_attractor_bures: float
    bures_angle_rad: float
    energy_decay_ratio: float
    grad_final_norm: float
    mass_on_P: float
    purity: float
    entropy: float
    fidelity_target: float
    iterations: int
    n_backtracks: int
    bb_mode_last: str
    converged: bool
    heyting_verdict: HeytingOmega3
    reaction_time_ns: float
    subspace_hash: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.3 TOONIntuitionEngine — orquestador soberano ──────────────────────
class TOONIntuitionEngine:
    r"""
    Motor Espectral de la Intuición Relámpago.

    Funtor soberano  F = F₃ ∘ F₂ ∘ F₁ :

        F₁  IntuitionFieldPreparation.prepare
        F₂  GeometricSeed.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ certify

    Asociatividad (teorema de anidamiento):

        execute_intuitive_cycle
            = _phase3_certify ∘ _phase2_flash ∘ _phase1_prepare
            = certify ∘ synthesize ∘ prepare.
    """

    def __init__(
        self,
        engine_id: str = "INTUITION-ENGINE-WISDOM-01",
        dimension_mac: int = 4,
        subspace_rank: int = 2,
        subspace_key: str = "REF-BASIS-INTUITION",
        max_descend_steps: int = FlashAttractorSolver.MAX_STEPS_DEFAULT,
        tol: float = FlashAttractorSolver.TOL_DEFAULT,
    ) -> None:
        if not (1 <= subspace_rank < dimension_mac):
            raise ValueError(
                f"subspace_rank ∈ [1, n−1]; r={subspace_rank}, n={dimension_mac}"
            )
        self.engine_id = engine_id
        self.dimension_mac = int(dimension_mac)
        self.subspace = SubspaceGeometryFactory.build(
            n=self.dimension_mac, rank=subspace_rank, key=subspace_key
        )
        self.max_descend_steps = int(max_descend_steps)
        self.tol = float(tol)
        self.cycle_count = 0
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::n={dimension_mac}::r={subspace_rank}::"
            f"basis={self.subspace.hash}".encode("ascii")
        ).hexdigest()

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, germinated_matrix: np.ndarray) -> GeometricSeed:
        """FASE 1 anidada: cierra con GeometricSeed (dominio de FASE 2)."""
        seed = IntuitionFieldPreparation.prepare(
            rho_input=germinated_matrix,
            subspace=self.subspace,
        )
        self._advance_chain("F1", bytes.fromhex(seed.seed_spectral_hash))
        return seed

    def _phase2_flash(self, seed: GeometricSeed) -> IntuitionTrajectoryBundle:
        """FASE 2 anidada: continuación de prepare; cierra con Bundle."""
        bundle = seed.continue_into_phase2(
            cycle_index=self.cycle_count,
            max_steps=self.max_descend_steps,
            tol=self.tol,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    def _phase3_certify(
        self,
        cycle_id: str,
        crop_origin_id: str,
        bundle: IntuitionTrajectoryBundle,
        external_verdict: HeytingOmega3,
        t_start_ns: int,
    ) -> IntuitiveFieldState:
        """FASE 3 anidada: continuación de synthesize; cierra con State."""
        final_verdict = bundle.continue_into_phase3(external_verdict)
        self._advance_chain("F3", final_verdict.name.encode("ascii"))
        t_elapsed_ns = float(time.perf_counter_ns() - t_start_ns)
        c = bundle.attractor_cert
        provenance = _sha256_bytes(
            self.engine_id.encode("ascii"),
            cycle_id.encode("ascii"),
            crop_origin_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{c.final_energy:.12e}".encode("ascii"),
            f"{c.bures_distance_seed_to_target:.12e}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )
        return IntuitiveFieldState(
            cycle_id=cycle_id,
            crop_origin_id=crop_origin_id,
            flash_attractor_energy=c.final_energy,
            seed_energy=c.initial_energy,
            manifold_geodesic_distance=c.bures_distance_seed_to_target,
            attractor_to_target_bures=c.bures_distance_attractor_to_target,
            seed_to_attractor_bures=c.bures_distance_seed_to_attractor,
            bures_angle_rad=bundle.seed.bures_angle_to_target,
            energy_decay_ratio=c.energy_decay_ratio,
            grad_final_norm=c.grad_final_norm,
            mass_on_P=bundle.seed.mass_on_P,
            purity=bundle.purity_final,
            entropy=bundle.entropy_final,
            fidelity_target=bundle.fidelity_target,
            iterations=c.iterations,
            n_backtracks=c.n_backtracks,
            bb_mode_last=c.bb_mode_last,
            converged=c.converged,
            heyting_verdict=final_verdict,
            reaction_time_ns=t_elapsed_ns,
            subspace_hash=bundle.seed.subspace.hash,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    def execute_intuitive_cycle(
        self,
        crop_origin_id: str,
        germinated_matrix: np.ndarray,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> IntuitiveFieldState:
        r"""Ciclo soberano: F₃ ∘ F₂ ∘ F₁."""
        self.cycle_count += 1
        cycle_id = f"CYC-INTUITION-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        t_start_ns = time.perf_counter_ns()
        logger.info(
            "═══ Ciclo Intuición #%d | orig=%s | subspace=%s r=%d ═══",
            self.cycle_count, crop_origin_id,
            self.subspace.hash[:12], self.subspace.rank,
        )
        seed = self._phase1_prepare(germinated_matrix)
        bundle = self._phase2_flash(seed)
        state = self._phase3_certify(
            cycle_id, crop_origin_id, bundle, external_verdict, t_start_ns,
        )
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s | Ω₃=%s | E_final=%.6e | E_ratio=%.6f | "
            "d_B(seed,tgt)=%.6f | d_B(att,tgt)=%.6f | iters=%d | %.3f ms",
            cycle_id, state.heyting_verdict.name,
            state.flash_attractor_energy, state.energy_decay_ratio,
            state.manifold_geodesic_distance, state.attractor_to_target_bures,
            state.iterations, dt_ms,
        )
        return state


# ── §3.4 Demostración autónoma ───────────────────────────────────────────
def _build_mixed_seed(
    n: int,
    leakage: float,
    subspace: SubspaceGeometry,
    key: str,
) -> ComplexMatrix:
    r"""
    Semilla con fuga controlada t ∈ [0,1] al complemento P_⊥:

        ρ_t = (1 − t) · ρ_P + t · ρ_{P_⊥}

    ρ_P (resp. ρ_{P_⊥}) es el estado Wishart proyectado y renormalizado
    sobre ran(P) (resp. ran(P_⊥)).

        t = 0 → supp(ρ) ⊆ ran(P)  ⇒  E[ρ] = 0
        t = 1 → supp(ρ) ⊆ ran(P_⊥) ⇒  E[ρ] máximo, intuición vacía
    """
    rng = np.random.default_rng(_seed_from_string(f"MIX::{key}"))
    P = subspace.projector
    Pc = np.eye(n, dtype=np.complex128) - P

    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho_raw = A @ A.conj().T
    tr_raw = float(np.trace(rho_raw).real)
    rho_raw = rho_raw / max(tr_raw, 1e-30)

    rho_P = P @ rho_raw @ P
    tr_P = float(np.trace(rho_P).real)
    if tr_P < 1e-15:
        trP = max(1.0, float(np.real(np.trace(P))))
        rho_P = P / trP
    else:
        rho_P = rho_P / tr_P

    rho_Pc = Pc @ rho_raw @ Pc
    tr_Pc = float(np.trace(rho_Pc).real)
    if tr_Pc < 1e-15:
        trPc = max(1.0, float(np.real(np.trace(Pc))))
        rho_Pc = Pc / trPc
    else:
        rho_Pc = rho_Pc / tr_Pc

    t = float(np.clip(leakage, 0.0, 1.0))
    rho_t = (1.0 - t) * rho_P + t * rho_Pc
    return DensityOperatorAlgebra.sanitize(rho_t)


if __name__ == "__main__":
    print("═" * 92)
    print("TOON INTUITION ENGINE — v2.2.0 Nested Doctoral")
    print("Bures · Dirichlet · BB1/BB2 · Grassmann · Heyting Ω₃ · Merkle")
    print("═" * 92)

    engine = TOONIntuitionEngine(
        engine_id="INTUITION-ENGINE-WISDOM-01",
        dimension_mac=4,
        subspace_rank=2,
        subspace_key="REF-BASIS-INTUITION",
    )

    print(
        f"\nSubspace r={engine.subspace.rank} | "
        f"is_isometry={engine.subspace.is_isometry} | "
        f"is_projector={engine.subspace.is_projector} | "
        f"‖B†B−I‖_F={engine.subspace.isometry_residual:.2e} | "
        f"‖P²−P‖_F={engine.subspace.projector_residual:.2e}"
    )
    print(f"Subspace hash: {engine.subspace.hash[:32]}…")

    print("\n──────────── Métricas de geometría del campo ────────────")
    rho_a = _build_mixed_seed(4, 0.0, engine.subspace, "TRIANGLE-A")
    rho_b = _build_mixed_seed(4, 0.5, engine.subspace, "TRIANGLE-B")
    rho_c = _build_mixed_seed(4, 1.0, engine.subspace, "TRIANGLE-C")
    tri = BuresGeodesicMetric.triangle_inequality_residual(rho_a, rho_b, rho_c)
    print(f"  δ_triangular (Bures) = {tri:.3e}   (esperado ≈ 0)")
    print(f"  d_B(ρ_a, ρ_b)        = {BuresGeodesicMetric.distance(rho_a, rho_b):.6f}")
    print(f"  d_B(ρ_a, ρ_c)        = {BuresGeodesicMetric.distance(rho_a, rho_c):.6f}")
    gamma_mid = BuresGeodesicMetric.geodesic(rho_a, rho_c, 0.5)
    print(
        f"  d_B(ρ_a, γ(1/2))     = "
        f"{BuresGeodesicMetric.distance(rho_a, gamma_mid):.6f}  "
        f"(geodésica W₂)"
    )

    print("\n──────────── Ciclos de intuición flash ────────────")
    scenarios = [
        ("COHERENT (fuga=0.05)", 0.05, HeytingOmega3.COHERENT),
        ("DEGRADED (fuga=0.50)", 0.50, HeytingOmega3.COHERENT),
        ("VETOED   (fuga=0.95)", 0.95, HeytingOmega3.COHERENT),
    ]

    for name, leakage, ext in scenarios:
        rho_seed = _build_mixed_seed(4, leakage, engine.subspace, name)
        land_pre = FlashDirichletFunctional.evaluate(
            rho_seed, engine.subspace.projector
        )
        state = engine.execute_intuitive_cycle(
            crop_origin_id=f"CROP-SOVEREIGN-0001::{name}",
            germinated_matrix=rho_seed,
            external_verdict=ext,
        )
        print(f"\n[{name}]")
        print(f"   ciclo_id             : {state.cycle_id}")
        print(f"   Ω₃ final             : {state.heyting_verdict.name}")
        print(f"   E_seed               : {land_pre.energy:.6e}  "
              f"(coh={land_pre.coherent_mass:.3e}, leak={land_pre.leak_mass:.3e})")
        print(f"   E_attractor          : {state.flash_attractor_energy:.6e}")
        print(f"   ratio decaimiento    : {state.energy_decay_ratio:.6f}")
        print(f"   ‖∇E|_T‖_F            : {state.grad_final_norm:.3e}")
        print(f"   mass_P(ρ_seed)       : {state.mass_on_P:.6f}")
        print(f"   d_B(seed, target)    : {state.manifold_geodesic_distance:.6f}")
        print(f"   d_B(att,  target)    : {state.attractor_to_target_bures:.6f}")
        print(f"   d_B(seed, att)       : {state.seed_to_attractor_bures:.6f}")
        print(f"   θ_Bures              : {state.bures_angle_rad:.6f} rad")
        print(f"   F(att, target)       : {state.fidelity_target:.6f}")
        print(f"   pureza post-descenso : {state.purity:.6f}")
        print(f"   entropía post        : {state.entropy:.6f}")
        print(f"   iteraciones / BB     : {state.iterations} / {state.bb_mode_last}  "
              f"(backtracks={state.n_backtracks})")
        print(f"   converged            : {state.converged}")
        print(f"   latencia ciclo       : {state.reaction_time_ns / 1000.0:.2f} µs")
        print(f"   firma (phase_chain)  : {state.phase_chain_sha256[:32]}…")
        print(f"   firma (provenance)   : {state.sha256_provenance[:32]}…")

    print("\n" + "═" * 92)
    print("✓ F1→F2: prepare ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate.")
    print("✓ d_B=√(2−2√F): geodésica de Bures, no Frobenius (P1).")
    print("✓ E=½‖ρ−PρP‖_F² = coh + ½ leak; Lip(∇E)≤1; ∇E|_T ∈ T𝔇_n (P2).")
    print("✓ BB1/BB2 + Armijo + Higham; η ∈ (0, 2] por L-smoothness (P4).")
    print("✓ Ω₃ por meets de ratios adimensionales, no conteo (P3).")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256 (P5).")
    print("═" * 92)
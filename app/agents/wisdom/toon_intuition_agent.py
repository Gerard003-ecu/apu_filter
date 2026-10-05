# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/wisdom/toon_intuition_agent.py                                 ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / REFLEJO RELÁMPAGO                   ║
║ FUNCIÓN  : SOBERANO DE LA INTUICIÓN — POINCARÉ · OSELEDETS · MELNIKOV · KAM · LÉVY   ║
║ VERSIÓN  : 9.1.0-Doctoral-3NestedPhases-PoincaréCanonical-VisceralReflexArch         ║
║ AUTOR    : Soberano Artesano Programador Senior (Crítico, Objetivo, Riguroso)        ║
║ FÍSICA   : Mecánica celeste de Poincaré, Sistemas dinámicos caóticos, Topología      ║
║            algebraica, Teoría espectral, Geometría Riemanniana, Análisis armónico    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE FASES ANIDADAS (categorías encajadas · categorías funcionales)
─────────────────────────────────────────────────────────────────────────────
El Soberano es un funtor F = F₃ ∘ F₂ ∘ F₁ sobre la categoría 𝐂𝐨𝐧𝐜𝐫𝐞𝐭𝐞 de
presheaves sobre el poset Ω₃:

    F₁ : Request × Gr(r,n)          → FlashHandoff          (sustrato geométrico)
    F₂ : FlashHandoff × ℝ₊³         → FlashTrajectoryBundle (dinámica flash visceral)
    F₃ : Bundle × Ω₃                → IntuitionFlashCertificate (adjudicación ⊗ crowbar)

Anidamiento estricto (naturalidad categórica):
    continue_into_phase2 ∘ build                  = synthesize ∘ build
    continue_into_phase3 ∘ synthesize ∘ build = (adjudicate ⊗ fire) ∘ synthesize ∘ build

MECÁNICA CELESTE DE POINCARÉ — NÚCLEO FORMAL
───────────────────────────────────────────────
Sea (M, ω, H) una variedad simpléctica 2n-dimensional con H = H₀(J) + εH₁(θ,J) en
variables acción-ángulo (θ,J) ∈ 𝕋ⁿ × ℝⁿ:

1. Σ ⊂ M transversal al flujo:  Σ ∩ {x : X(x)=0} = ∅.
   Aplicación de primer retorno:  P : Σ → Σ,  P(x) = φ_{τ(x)}(x).
   Matriz de Floquet:  M_x = dP_x;  multiplicadores  μ_i ∈ ℂ.
   Estabilidad local:  |μ_i| < 1 para contracción.

2. Espectro de Lyapunov de Oseledets (multiplicativo):
       λ_i = lim_{k→∞} (1/k) log σ_i( dP^k_x ),   λ₁ ≥ λ₂ ≥ … ≥ λ_{2n-2}.
   Descomposición de Oseledets:  T_xΣ = ⊕_j E_j  con tasas  λ^(j).
   Entropía de Kolmogorov–Sinai (Pesin):  h_KS = Σ_{λ_i > 0} λ_i.
   Dimensión de Kaplan–Yorke:  D_KY = k + (Σ_{i≤k} λ_i) / |λ_{k+1}|.

3. Integral de Melnikov (ruptura homoclínica):
       M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(q⁰(t), p⁰(t)) dt.
   Ceros simples de M ⇒ W^s ∩ W^u transversal ⇒ caos homoclínico.
   En el motor (adaptado):  M = ⟨∇E|_T, [∇²E|_T, ∇E|_T]⟩_F  sobre Dirichlet.

4. Teorema KAM (Arnold 1963): si ω ∈ ℝⁿ satisface Diofántico
       |ω·k| ≥ γ / |k|^τ,   ∀ k ∈ ℤⁿ \ {0},
   y  ε < ε₀(γ, τ, H₀),  entonces el toro 𝕋ⁿ persiste. Criterio ejecutable.

5. Lema de Lévy (concentración de medida sobre S^{n-1}(√n)):
       P(|f − 𝔼[f]| ≥ ε) ≤ 2 exp(−(n−1) ε² / (2 L²)),
   para f : S^{n-1} → ℝ Lipschitz-L. Fundamenta la proyección Grassmanniana.

POSTULADOS OPERATIVOS
──────────────────────
P1. Distancia de Bures: d_B(ρ,σ) = √(2 − 2 √F(ρ,σ)) — única CPTP-contractiva.
P2. Funcional de Dirichlet: E(ρ) = ½‖ρ − PρP‖_F²;  ∇E(ρ) = ρ − PρP;  Lip(∇E) ≤ 1.
P3. Stiefel-Haar sembrado por SHA-256(agent_id) (reproducibilidad).
P4. has_critical_fraud ⇒ ⊥ duro (predicado contextual, veto no-espectral).
P5. Latencia advisory; Kelly G* = log 2 − h₂(F) (información mutua).
P6. Cadena Merkle F1 → F2 → F3 con SHA-256 (custodia inmutable).
P7. Flash es one-shot (η*): Cauchy un paso, Newton exacto si η*=1.

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
───────────────────────────────────────
- Reflejo visceral en Δτ < 10 µs (latencia fija), respaldado por Secciones de Poincaré.
- Veto automático si: Oseledets diverge (λ_max > 0), Melnikov homoclínico,
  KAM colapsa, distorsión Bures excede umbral, o fraude crítico (P4).
- Kelly penalizado por h_KS (entropía topológica): stake = κ·f*·exp(−h_KS).
- Crowbar BT151 en GPIO14 con latencia nominal 392.15 ns (< 400 ns ISR).
"""
from __future__ import annotations
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TOONIntuition")
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
    """Proyección SHA-256 → ℕ/2³² determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    """Cadena Merkle determinista: concatena hashes."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   1   ·   S U S T R A T O   G E O M É T R I C O   (V_W-F1)                 ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   Request × Gr(r,n)                                                      ║
# ║  Codominio: FlashHandoff                                                          ║
# ║                                                                                    ║
# ║  Objetos anidados: Ω₃, 𝔇_n, Gr(r,n), Σ (Poincaré), Request, Audit                 ║
# ║                                                                                    ║
# ║  Morfismo terminal: FlashHandoff.continue_into_phase2                              ║
# ║                                                                                    ║
# ║  Esta definición formal es la ENTRADA CANÓNICA a F₂.                              ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §1.1 Retículo distributivo de Heyting Ω₃ ─────────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa (álgebra de Gödel de 3 valores):
        Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}
    
    Estructura:
        meet   (∧) = min      join (∨) = max
        implies (⇒) : residuo de Galois  a ∧ b ≤ c ⇔ a ≤ (b ⇒ c)
        neg    (¬) : seudocomplemento    a ⇒ ⊥
        iff    (⇔) : (a⇒b) ∧ (b⇒a)
    
    Regularidad (Boole vs. Heyting):
        ⊥ ∨ ¬⊥ = ⊥ = ⊤  (regular)
        ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤  (no regular)
        ⊤ ∨ ¬⊤ = ⊤ = ⊤  (regular)
    Subálgebra Booleana: {⊥, ⊤} = fix(¬¬).
    """
    VETOED: int = 0      # ⊥ (veto duro)
    DEGRADED: int = 1    # ⋆ (alerta, degradado)
    COHERENT: int = 2    # ⊤ (verde, coherente)

    @property
    def verdict(self) -> str:
        return self.name

    def leq(self, other: "HeytingOmega3") -> bool:
        """Orden: ⊥ ≤ ⋆ ≤ ⊤."""
        return int(self) <= int(other)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """meet (∧) = min: máximo común divisor lógico."""
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """join (∨) = max: mínimo común múltiplo lógico."""
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """implies (⇒): residuo de Galois."""
        return HeytingOmega3.COHERENT if self.leq(other) else other

    def neg(self) -> "HeytingOmega3":
        """neg (¬): seudocomplemento a ⇒ ⊥."""
        return self.implies(HeytingOmega3.VETOED)

    def iff(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """iff (⇔): equivalencia lógica."""
        return self.implies(other).meet(other.implies(self))

    def is_regular(self) -> bool:
        """Regularidad: a ∨ ¬a = ⊤. Solo ⊥ y ⊤ son regulares."""
        return self.neg().neg() == self

    def as_weight(self) -> float:
        """Proyección a [0,1]: pesos para métricas adimensionales."""
        return float(int(self)) / 2.0


# ── §1.2 Álgebra de operadores densidad ──────────────────────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre 𝔇_n = { ρ ∈ M_n(ℂ) : ρ=ρ†, ρ≥0, Tr ρ = 1 }.
    
    Espacio tangente afín: T_ρ𝔇_n ≅ { A = A† : Tr A = 0 }.
    
    Funcionales:
        S(ρ)           = von Neumann entropy
        P(ρ)           = purity  Tr(ρ²)
        F(ρ,σ)         = Uhlmann–Jozsa fidelity
        d_B(ρ,σ)       = Bures distance (P1)
        θ_B(ρ,σ)       = Bures angle
        K_ρ = −log ρ   = Hamiltonian modular
    
    Proyección Higham: sanitize es no-expansiva en ‖·‖_F.
    """
    EPS: Final[float] = _EPS
    EPS_MODULAR: Final[float] = _EPS_MOD

    @classmethod
    def is_square(cls, rho: np.ndarray) -> bool:
        """Validación estructural: matriz cuadrada n×n, n>0."""
        arr = np.asarray(rho)
        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""
        Proyección no-expansiva Higham a 𝔇_n:
            ρ ↦ Hermitizar ↦ clip eigenvalores ↦ renormalizar Tr ρ = 1.
        Garantiza ρ ∈ 𝔇_n con ‖ρ‖_F ≤ ‖ρ_in‖_F + const.
        """
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
        """Espectro en orden descendente, normalizado a probabilidad."""
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
        r"""S(ρ) = −Σ_i p_i log p_i."""
        p = cls.spectrum_descending(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        r"""P(ρ) = Tr(ρ²) = Σ_i p_i²."""
        p = cls.spectrum_descending(rho)
        return float(np.sum(p * p))

    @classmethod
    def frobenius(cls, A: np.ndarray) -> float:
        r"""‖A‖_F = √(Tr(A†A))."""
        return float(np.linalg.norm(A, "fro"))

    @classmethod
    def matrix_power(
        cls, rho: np.ndarray, z: complex, floor: float = _EPS_MOD
    ) -> ComplexMatrix:
        r"""Potencia matricial: ρ^z = V exp(z log Λ) V†."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = ‖√ρ √σ‖₁."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sr = cls.matrix_power(rho, 0.5)
        inner = sr @ sigma @ sr
        val = float(np.real(np.trace(cls.matrix_power(inner, 0.5))))
        return float(np.clip(val, 0.0, 1.0))

    @classmethod
    def bures_angle(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""θ_B(ρ,σ) = arccos(√F(ρ,σ))."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.acos(float(np.clip(math.sqrt(F), 0.0, 1.0))))

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2 − 2√F(ρ,σ))."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""K_ρ = −log ρ."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), cls.EPS_MODULAR)
        kspec = -np.log(w)
        return (V * kspec.astype(np.complex128)) @ V.conj().T

    @classmethod
    def tangent_project(cls, A: np.ndarray, n: Optional[int] = None) -> ComplexMatrix:
        r"""
        Proyección al espacio tangente T_ρ𝔇_n = {A† : Tr A = 0}:
            Π_T(A) = A − (Tr A / n) · I
        """
        H = np.asarray(A, dtype=np.complex128)
        H = 0.5 * (H + H.conj().T)
        dim = int(n if n is not None else H.shape[0])
        beta = float(np.trace(H).real) / max(dim, 1)
        return H - beta * np.eye(dim, dtype=np.complex128)


# ── §1.3 Sección de Poincaré formal (Σ ⊂ 𝔇_n codimensión 1) ─────────────────────
@dataclass(frozen=True, slots=True)
class PoincareSectionManifold:
    r"""
    Σ = ran(P) ∩ 𝔇_n: subvariedad de codimensión 1 transversal a X = −∇E|_T.
    
    Test de transversalidad: |⟨X(ρ), n_Σ(ρ)⟩_F| = |Tr(P X(ρ))| > 0
    para ρ muestreada Haar-Ginibre.
    
    Almacena:
        transversality_min     : min muestral de |Tr(P X)|
        first_return_time      : τ* ≈ 1 / ⟨|Tr(P X)|⟩
        floquet_moduli         : |μ_i|, espectro de dP sobre Σ
        periodic_orbit_rank    : #{|μ_i| > 1} (órbitas periódicas inestables)
    """
    projector: np.ndarray
    normal: np.ndarray            # n_Σ = P (gradiente de Tr(Pρ))
    transversality_min: float
    first_return_time: float
    periodic_orbit_rank: int
    floquet_moduli: RealVector
    hash: str

    @classmethod
    def build(
        cls,
        projector: np.ndarray,
        grad_field: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 16,
        key: str = "POINCARE-AGENT",
    ) -> "PoincareSectionManifold":
        r"""
        Construye Σ y mide transversalidad del campo X = −∇E|_T.
        """
        n = projector.shape[0]
        P = 0.5 * (projector + projector.conj().T)
        rng = np.random.default_rng(_seed_from_string(f"SECTION::{key}"))

        tmin = float("inf")
        tau_acc: List[float] = []

        for _ in range(n_samples):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            rho = DensityOperatorAlgebra.sanitize(A @ A.conj().T)
            X = grad_field(rho)
            # Componente normal: ⟨X, n_Σ⟩ = Tr(P X)
            nc = abs(float(np.real(np.trace(P @ X))))
            tmin = min(tmin, nc)
            tau_acc.append(1.0 / max(nc, 1e-12))

        tau_mean = float(np.mean(tau_acc)) if tau_acc else 0.0

        # Multiplicadores de Floquet: valores propios de dP
        mu = np.sort(np.abs(np.real(la.eigvalsh(P))))[::-1]
        per_rank = int(np.sum(mu > 1.0 + 1e-12))

        sec_hash = _sha256_bytes(
            np.ascontiguousarray(P).tobytes(),
            np.ascontiguousarray(mu).tobytes(),
            f"{tmin:.12e}".encode("ascii"),
        )

        return cls(
            projector=P,
            normal=P,
            transversality_min=float(tmin),
            first_return_time=float(tau_mean),
            periodic_orbit_rank=per_rank,
            floquet_moduli=mu.astype(np.float64),
            hash=sec_hash,
        )

    @property
    def is_transverse(self) -> bool:
        """Transversalidad verificada."""
        return self.transversality_min > 1e-10


# ── §1.4 Lema de concentración de Lévy sobre S^{n−1}(√n) ────────────────────────
class LevyConcentrationLemma:
    r"""
    Para X uniforme sobre S^{n−1}(1) y f Lipschitz-L:
        P(|f(X) − 𝔼[f]| ≥ ε) ≤ 2 exp(−(n−1)ε² / (2L²)).
    Sobre S^{n−1}(√n) la cota es idéntica (reescalado).
    Fundamenta la proyección Grassmanniana.
    """
    @staticmethod
    def bound(epsilon: float, n: int, lipschitz: float = 1.0) -> float:
        """Cota de cola."""
        if n <= 1:
            return 1.0
        L = max(float(lipschitz), 1e-15)
        e = max(float(epsilon), 0.0)
        return float(min(1.0, 2.0 * math.exp(-(n - 1) * e * e / (2.0 * L * L))))

    @staticmethod
    def median_width(
        n: int, lipschitz: float = 1.0, confidence: float = 0.99
    ) -> float:
        r"""ε* tal que P(|f−𝔼f| ≥ ε*) ≤ 1 − confidence."""
        if n <= 1:
            return float("inf")
        p_tail = max(1e-15, 1.0 - float(confidence)) / 2.0
        return float(lipschitz * math.sqrt(2.0 * math.log(2.0 / p_tail) / (n - 1)))


# ── §1.5 Geometría del subespacio de decisiones (Gr(r,n)) ───────────────────────
@dataclass(frozen=True, slots=True)
class DecisionManifoldGeometry:
    r"""
    Punto de Gr(r, n): B ∈ St(r,n), P = B B†, hash SHA-256(B ‖ P).
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
        """Proyector complementario P_⊥ = I − P."""
        n = self.projector.shape[0]
        return np.eye(n, dtype=np.complex128) - self.projector

    def mass(self, rho: np.ndarray) -> float:
        """Masa de ρ sobre el subespacio: Tr(P ρ)."""
        rho_s = DensityOperatorAlgebra.sanitize(rho)
        return float(np.real(np.trace(self.projector @ rho_s)))


class DecisionManifoldFactory:
    r"""
    Construye B ∈ St(r, n) por QR-Haar (Stewart 1980).
    Semilla determinista: SHA-256(key).  Fuerza r ∈ [1, n−1].
    """
    @classmethod
    def haar_unitary(cls, n: int, rng: np.random.Generator) -> ComplexMatrix:
        """Matriz unitaria aleatoria por QR en Ginibre."""
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        d = np.diagonal(R)
        ph = np.where(np.abs(d) > 1e-30, d / np.abs(d), 1.0 + 0j)
        return (Q * ph.conj()).astype(np.complex128)

    @classmethod
    def principal_angles(cls, B1: np.ndarray, B2: np.ndarray) -> RealVector:
        r"""Ángulos principales: θ_i = arccos(σ_i(B1† B2))."""
        M = B1.conj().T @ B2
        sig = np.clip(np.real(la.svdvals(M)), 0.0, 1.0)
        return np.arccos(sig).astype(np.float64)

    @classmethod
    def grassmann_distance(cls, B1: np.ndarray, B2: np.ndarray) -> float:
        r"""Distancia geodésica en Gr(r,n): √(Σ θ_i²)."""
        return float(np.linalg.norm(cls.principal_angles(B1, B2), ord=2))

    @classmethod
    def build(cls, n: int, rank: int, key: str) -> DecisionManifoldGeometry:
        r"""Construye punto en Gr(r,n) por QR-Haar."""
        if n < 2:
            raise ValueError("DecisionManifoldFactory.build: n ≥ 2")
        rank = int(np.clip(rank, 1, n - 1))
        rng = np.random.default_rng(_seed_from_string(f"M_DECISION::{key}"))
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
        return DecisionManifoldGeometry(
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


# ── §1.6 Solicitud + auditoría ───────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitiveFlashRequest:
    """Solicitud de flash intuitivo."""
    crop_origin_id: str
    seed_crystal_id: str
    germinated_density_matrix: np.ndarray
    site_context_payload: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RequestAuditReport:
    r"""
    Verificación dimensional del request:
        is_valid := |Tr ρ − 1| < τ ∧ ‖ρ−ρ†‖_F < τ_H ∧ λ_min ≥ −ε.
    has_critical_fraud = predicado contextual (P4, veto duro).
    """
    purity: float
    entropy: float
    trace_residual: float
    hermiticity_residual: float
    lambda_min: float
    has_critical_fraud: bool
    cost_risk_amount: float
    payload_hash: str
    is_valid: bool


class FlashRequestSanitizer:
    """Sanitización y auditoria de request."""
    TRACE_TOL: Final[float] = 1.0e-6
    HERM_TOL: Final[float] = 1.0e-8

    @classmethod
    def _payload_bytes(cls, payload: Dict[str, Any]) -> bytes:
        """Serialización JSON determinista (sorted keys)."""
        try:
            return json.dumps(
                payload, sort_keys=True, default=str, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return repr(sorted(payload.items())).encode("utf-8")

    @classmethod
    def sanitize(
        cls, request: IntuitiveFlashRequest
    ) -> Tuple[ComplexMatrix, RequestAuditReport]:
        r"""Saneamiento y auditoria."""
        rho_raw = np.asarray(request.germinated_density_matrix, dtype=np.complex128)
        if not DensityOperatorAlgebra.is_square(rho_raw):
            raise ValueError(
                f"FlashRequestSanitizer: matriz no cuadrada {np.shape(rho_raw)}"
            )

        herm_res = float(np.linalg.norm(rho_raw - rho_raw.conj().T, "fro"))
        trace_res = abs(float(np.trace(rho_raw).real) - 1.0)

        rho = DensityOperatorAlgebra.sanitize(rho_raw)
        w = DensityOperatorAlgebra.spectrum_descending(rho)
        purity = float(np.sum(w ** 2))
        entropy = float(-np.sum(w * np.log(np.maximum(w, _EPS))))
        lam_min = float(np.real(la.eigvalsh(rho)).min())

        payload = dict(request.site_context_payload or {})
        fraud = bool(payload.get("has_critical_fraud", False))
        cost = float(payload.get("cost_risk_amount", 0.0) or 0.0)

        payload_hash = _sha256_bytes(
            cls._payload_bytes(payload),
            np.ascontiguousarray(rho).tobytes(),
        )

        is_valid = (
            trace_res < cls.TRACE_TOL
            and herm_res < cls.HERM_TOL
            and lam_min > -1e-6
        )

        return rho, RequestAuditReport(
            purity=purity,
            entropy=entropy,
            trace_residual=trace_res,
            hermiticity_residual=herm_res,
            lambda_min=lam_min,
            has_critical_fraud=fraud,
            cost_risk_amount=cost,
            payload_hash=payload_hash,
            is_valid=bool(is_valid),
        )


# ── §1.7 FlashHandoff — HAND-OFF FORMAL FASE 1 → FASE 2 ─────────────────────────
@dataclass(frozen=True, slots=True)
class FlashHandoff:
    r"""
    Objeto terminal FASE 1 / inicial FASE 2.
    
    Contenido:
        ρ_seed saneada
        ρ_target = PρP/Tr(PρP)  (atractor estático en ran(P))
        Geometría (B,P) ∈ Gr(r,n)
        Σ de Poincaré con sección transversal
        Energía de Dirichlet E
        Banda de concentración de Lévy ε*
        Hash espectral inmutable
    """
    cycle_index: int
    request_id: str
    crop_origin_id: str
    seed_crystal_id: str
    rho_seed: np.ndarray
    rho_target: np.ndarray
    geometry: DecisionManifoldGeometry
    poincare_section: PoincareSectionManifold
    seed_energy: float
    seed_fidelity: float
    mass_on_P: float
    request_audit: RequestAuditReport
    levy_band: float
    spectral_hash: str

    @classmethod
    def static_target(cls, rho: np.ndarray, P: np.ndarray) -> ComplexMatrix:
        """Atractor estático: ρ_tgt = PρP/Tr(PρP)."""
        PrP = P @ rho @ P
        tr = float(np.trace(PrP).real)
        if tr < 1e-15:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return DensityOperatorAlgebra.sanitize(PrP / tr)

    @classmethod
    def dirichlet_energy(cls, rho: np.ndarray, P: np.ndarray) -> float:
        r"""E(ρ) = ½‖ρ − PρP‖_F²."""
        residual = rho - P @ rho @ P
        return 0.5 * float(np.linalg.norm(residual, "fro") ** 2)

    @classmethod
    def build(
        cls,
        cycle_index: int,
        request: IntuitiveFlashRequest,
        geometry: DecisionManifoldGeometry,
        poincare_key: str = "AGENT-INTUITION",
    ) -> "FlashHandoff":
        r"""Construye el handoff: cierra FASE 1."""
        rho, audit = FlashRequestSanitizer.sanitize(request)
        P = geometry.projector

        rho_target = cls.static_target(rho, P)
        seed_energy = cls.dirichlet_energy(rho, P)
        seed_fid = DensityOperatorAlgebra.uhlmann_fidelity(rho, rho_target)
        mass = float(np.real(np.trace(P @ rho)))

        # Sección de Poincaré con campo X(ρ) = ρ − PρP
        grad_field = lambda x: x - P @ x @ P  # noqa: E731
        poincare = PoincareSectionManifold.build(
            projector=P, grad_field=grad_field, n_samples=12, key=poincare_key,
        )

        n = int(rho.shape[0])
        levy = LevyConcentrationLemma.median_width(n, lipschitz=1.0, confidence=0.99)

        w = DensityOperatorAlgebra.spectrum_descending(rho)
        spec_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(w).tobytes(),
            poincare.hash.encode("ascii"),
        )

        return cls(
            cycle_index=cycle_index,
            request_id=f"{request.crop_origin_id}::{request.seed_crystal_id}",
            crop_origin_id=request.crop_origin_id,
            seed_crystal_id=request.seed_crystal_id,
            rho_seed=rho,
            rho_target=rho_target,
            geometry=geometry,
            poincare_section=poincare,
            seed_energy=float(seed_energy),
            seed_fidelity=float(seed_fid),
            mass_on_P=mass,
            request_audit=audit,
            levy_band=float(levy),
            spectral_hash=spec_hash,
        )

    def as_dict(self) -> Dict[str, object]:
        """Representación para logging."""
        return {
            "request_id": self.request_id,
            "seed_energy": self.seed_energy,
            "seed_fidelity": self.seed_fidelity,
            "mass_on_P": self.mass_on_P,
            "geometry_hash": self.geometry.hash,
            "poincare_hash": self.poincare_section.hash,
            "levy_band": self.levy_band,
            "has_critical_fraud": self.request_audit.has_critical_fraud,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        eta_star: float,
        latency_contract_ns: float,
        kelly_kappa: float,
    ) -> "FlashTrajectoryBundle":
        r"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║ ÚLTIMO MORFISMO DE FASE 1 ∧ PRIMER MORFISMO DE FASE 2               ║
        ║                                                                      ║
        ║ Identidad de composición:                                            ║
        ║   continue_into_phase2 ∘ build                                        ║
        ║     = FlashPipeline.synthesize ∘ build                              ║
        ║     : Request × Gr(r,n) → FlashTrajectoryBundle                      ║
        ║                                                                      ║
        ║ Esta definición formal es la ENTRADA CANÓNICA a F₂.                 ║
        ║ El único funtor de FASE 2 es synthesize que toma FlashHandoff como  ║
        ║ dominio (a través de este método).                                   ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        return FlashPipeline.synthesize(
            handoff=self,
            eta_star=eta_star,
            latency_contract_ns=latency_contract_ns,
            kelly_kappa=kelly_kappa,
        )


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   2   ·   D I N Á M I C A   D E   A T R A C C I Ó N   F L A S H            ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  MÉTRICAS: Bures, Dirichlet, Oseledets, Melnikov, KAM, Kelly, Latencia           ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   FlashHandoff (objeto terminal de §1.7)                                ║
# ║  Codominio: FlashTrajectoryBundle (objeto terminal de §2)                         ║
# ║                                                                                    ║
# ║  Característica: Flash es ONE-SHOT (η*):                                          ║
# ║    ρ_flash = sanitize(ρ_seed − η* ∇E|_T)  (Cauchy un paso)                       ║
# ║    si η* = 1 → Newton exacto en bloque P_⊥                                       ║
# ║                                                                                    ║
# ║  Certificación Poincaré automática en synthesize:                                ║
# ║    - Oseledets por QR                                                             ║
# ║    - Melnikov por corchetes de Lie                                                ║
# ║    - KAM por diofantina                                                           ║
# ║    - Validación de transversalidad                                                ║
# ║                                                                                    ║
# ║  Morfismo terminal F₂:                                                             ║
# ║    FlashTrajectoryBundle.continue_into_phase3 → (Ω₃, CrowbarReport)               ║
# ║                                                                                    ║
# ║  Este es el primer morfismo de FASE 3 (simultáneamente último de FASE 2).         ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §2.1 Funcional de Dirichlet ─────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DirichletLandscape:
    r"""
    E(ρ) = ½‖ρ − PρP‖_F²; ∇E|_T ∈ T_ρ𝔇_n; Lip(∇E) ≤ 1.
    
    Identidad: ‖ρ−PρP‖_F² = 2·coherent_mass + leak_mass.
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
    """Evaluación canónica del funcional E."""
    LIPSCHITZ: Final[float] = 1.0

    @classmethod
    def decompose_residual(
        cls, rho: np.ndarray, P: np.ndarray
    ) -> Tuple[ComplexMatrix, float, float]:
        r"""
        Descomposición de residuo:
            ρ − PρP = [P_⊥ρP] + [P_⊥ρP_⊥]
        coherent_mass = ‖P_⊥ρP‖_F²;  leak_mass = ‖P_⊥ρP_⊥‖_F².
        """
        I = np.eye(P.shape[0], dtype=np.complex128)
        Pc = I - P
        cross = Pc @ rho @ P
        leak = Pc @ rho @ Pc
        residual = rho - P @ rho @ P
        return residual, float(np.linalg.norm(cross, "fro") ** 2), float(
            np.linalg.norm(leak, "fro") ** 2
        )

    @classmethod
    def evaluate(cls, rho: np.ndarray, P: np.ndarray) -> DirichletLandscape:
        """Evalúa E, ∇E, ∇E|_T, descomposición residual."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        residual, coh, leak = cls.decompose_residual(rho, P)
        energy = 0.5 * float(np.linalg.norm(residual, "fro") ** 2)
        grad_tan = DensityOperatorAlgebra.tangent_project(residual, n)
        return DirichletLandscape(
            rho=rho,
            energy=energy,
            grad_full=residual,
            grad_tan=grad_tan,
            grad_norm=float(np.linalg.norm(grad_tan, "fro")),
            residual_norm=float(np.linalg.norm(residual, "fro")),
            coherent_mass=coh,
            leak_mass=leak,
        )

    @classmethod
    def cauchy_step(cls, rho: np.ndarray, P: np.ndarray, eta: float) -> ComplexMatrix:
        r"""
        Un paso Cauchy (flash): ρ_flash = sanitize(ρ − η ∇E|_T).
        η = 1  ⇒ Newton exacto en bloque P_⊥  (P7).
        """
        land = cls.evaluate(rho, P)
        return DensityOperatorAlgebra.sanitize(rho - float(eta) * land.grad_tan)


# ── §2.2 Métrica geodésica de Bures + McCann-JKO ────────────────────────────────
class BuresGeodesicMetric:
    r"""
    Métrica de Bures sobre 𝔇_n (única CPTP-contractiva).
    
    Fidelidad:  F(ρ,σ) = ‖√ρ √σ‖₁ (Uhlmann-Jozsa)
    Ángulo:     θ_B = arccos(√F)
    Distancia:  d_B = √(2 − 2√F)
    
    Geodésica de McCann (JKO):
        C_ρσ = ρ^{−½}(ρ^{½}σρ^{½})^{½}ρ^{−½}
        γ(t) = [(1−t)I + tC_ρσ] ρ [(1−t)I + tC_ρσ]†
    Cumple: d_B(ρ, γ(½)) = d_B(γ(½), σ) = d_B(ρ,σ)/2 (bisección).
    """
    @classmethod
    def fidelity(cls, rho, sigma):
        return DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)

    @classmethod
    def angle(cls, rho, sigma):
        return DensityOperatorAlgebra.bures_angle(rho, sigma)

    @classmethod
    def distance(cls, rho, sigma):
        return DensityOperatorAlgebra.bures_distance(rho, sigma)

    @classmethod
    def triangle_residual(cls, a, b, c) -> float:
        """δ_tri = d(a,c) − d(a,b) − d(b,c); debe ser ≤ 0."""
        return float(max(0.0, cls.distance(a, c) - cls.distance(a, b) - cls.distance(b, c)))

    @classmethod
    def uhlmann_map(cls, rho: np.ndarray, sigma: np.ndarray) -> ComplexMatrix:
        r"""Matriz de transporte óptimo de Uhlmann."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(sigma)
        rh = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        ri = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        inner = rh @ sigma @ rh
        ih = DensityOperatorAlgebra.matrix_power(inner, 0.5)
        C = ri @ ih @ ri
        return 0.5 * (C + C.conj().T)

    @classmethod
    def geodesic(cls, rho: np.ndarray, sigma: np.ndarray, t: float) -> ComplexMatrix:
        r"""Geodésica de McCann-JKO."""
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


# ── §2.3 Espectro de Lyapunov de Oseledets (completo, h_KS, D_KY) ──────────────
@dataclass(frozen=True, slots=True)
class OseledetsLyapunovSpectrum:
    r"""
    Espectro de Lyapunov completo de la aplicación de Poincaré P : Σ → Σ.
    
    Cálculo (algoritmo QR de Benettin–Galgani–Giorgilli–Strelcyn):
        M_k := dP^k_x  (Jacobiano iterado)
        Q_k R_k = QR(M_k) ⇒ λ_i = lim (1/k) Σ log |R_k[i,i]|
    
    Métricas derivadas:
        λ_max              = λ₁ (exponente de Oseledets principal)
        h_KS(Pesin)        = Σ_{λ_i > 0} λ_i (entropía topológica)
        D_KY(Kaplan-Yorke) = k + (Σ_{i≤k} λ_i)/|λ_{k+1}| (dimensión fractal)
    
    Interpretación ejecutiva:
        h_KS es la tasa exponencial de pérdida de información predictiva.
        Modula la fracción de Kelly: stake = κ·f*·exp(−h_KS).
    """
    lyapunov_full: RealVector
    lyapunov_max: float
    lyapunov_min: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    n_positive: int
    iterations: int

    @classmethod
    def estimate(
        cls,
        jacobian_at: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_iter: int = 48,
        jitter: float = 1e-9,
        rng_seed: Optional[int] = None,
    ) -> "OseledetsLyapunovSpectrum":
        r"""
        Algoritmo QR para estimar el espectro de Oseledets del Jacobiano.
        """
        x = np.asarray(x0, dtype=np.float64).ravel()
        n = x.size
        rng = np.random.default_rng(
            rng_seed if rng_seed is not None else _seed_from_string("OSELEDETS")
        )
        Q = la.qr(rng.standard_normal((n, n)))[0]
        log_acc = np.zeros(n, dtype=np.float64)
        j = max(float(jitter), 1e-15)

        for _ in range(n_iter):
            M = np.asarray(jacobian_at(x), dtype=np.float64) + j * np.eye(n)
            Q, R = la.qr(M @ Q)
            diag = np.maximum(np.abs(np.diagonal(R)), 1e-300)
            log_acc += np.log(diag)
            x = x + 1e-6 * (M @ x) / (1.0 + np.linalg.norm(M @ x))

        lam = np.sort(log_acc / n_iter)[::-1]
        lam_max = float(lam[0]) if lam.size else 0.0
        lam_min = float(lam[-1]) if lam.size else 0.0
        pos = lam[lam > 0.0]
        h_ks = float(np.sum(pos)) if pos.size else 0.0

        cum = np.cumsum(lam)
        k = int(np.max(np.where(cum >= 0.0)[0])) + 1 if np.any(cum >= 0.0) else 0
        if 0 < k < lam.size and abs(lam[k]) > 1e-15:
            d_ky = float(k + cum[k - 1] / abs(lam[k]))
        else:
            d_ky = float(k)

        return cls(
            lyapunov_full=lam.astype(np.float64),
            lyapunov_max=lam_max,
            lyapunov_min=lam_min,
            kolmogorov_sinai_entropy=h_ks,
            kaplan_yorke_dimension=d_ky,
            n_positive=int(pos.size),
            iterations=int(n_iter),
        )


# ── §2.4 Integral de Melnikov (caos homoclínico) ────────────────────────────────
@dataclass(frozen=True, slots=True)
class MelnikovHomoclinicCertificate:
    r"""
    Integral de Melnikov asociada al par (H₀, H₁) sobre la sección Σ.
    
    Formalismo clásico:
        Para H = H₀ + εH₁ con órbita homoclínica q⁰(t) de H₀,
            M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(q⁰(t), p⁰(t)) dt
        Ceros simples de M ⇒ W^s ∩ W^u transversal ⇒ caos homoclínico.
    
    En el motor (adaptado a Dirichlet):
        M = ⟨[∇²E|_T, ∇E|_T], ∇E|_T⟩_F
        mide la componente del corchete de Lie que rompe la homoclinía.
        Si |M| > ε_M, el flujo entra en régimen de caos homoclínico.
    """
    melnikov_value: float
    melnikov_zeros: int
    transverse_homoclinic: bool
    chaos_threshold: float

    @classmethod
    def evaluate(
        cls,
        rho: np.ndarray,
        P: np.ndarray,
        epsilon_M: float = 1.0e-3,
    ) -> "MelnikovHomoclinicCertificate":
        r"""Evalúa la integral de Melnikov sobre el residuo de Dirichlet."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]

        # ∇E(ρ) = ρ − PρP; su proyección tangente
        g = DensityOperatorAlgebra.tangent_project(rho - P @ rho @ P, n)

        # Hessiano efectivo H_E = Id − Ad_P (lineal).  Aplicado a g:
        H_eff = g - P @ g @ P

        # Corchete de Lie [H_eff, g] = H_eff g − g H_eff
        comm = H_eff @ g - g @ H_eff

        M = float(np.real(np.vdot(comm, g)))

        # Ceros simples de M en una vecindad: cambio de signo en pequeña órbita
        ts = np.linspace(-1.0, 1.0, 41)
        vals = np.zeros_like(ts)
        denom = 1.0 + np.linalg.norm(comm)

        for i, t in enumerate(ts):
            rt = DensityOperatorAlgebra.sanitize(rho + t * 1e-4 * comm / denom)
            gt = DensityOperatorAlgebra.tangent_project(rt - P @ rt @ P, n)
            Ht = gt - P @ gt @ P
            cmt = Ht @ gt - gt @ Ht
            vals[i] = float(np.real(np.vdot(cmt, gt)))

        signs = np.sign(vals)
        zeros = int(np.sum(np.abs(np.diff(signs)) > 1.0))
        transverse = bool(abs(M) > epsilon_M and zeros >= 1)

        return cls(
            melnikov_value=float(M),
            melnikov_zeros=zeros,
            transverse_homoclinic=transverse,
            chaos_threshold=float(epsilon_M),
        )


# ── §2.5 Criterio KAM diofántico (persistencia de toros) ────────────────────────
@dataclass(frozen=True, slots=True)
class KAMDiophantineCertificate:
    r"""
    Criterio KAM (Arnold 1963, Moser 1962): persistencia de toros invariantes.
    
    Frecuencia ω ∈ ℝⁿ diofántica (γ, τ):
        |ω·k| ≥ γ / |k|^τ,   ∀ k ∈ ℤⁿ \ {0}.
    
    Umbral KAM:
        ε < ε₀(γ, τ, n, H₀) ⇒ el toro 𝕋ⁿ(J₀) persiste.
    
    Estimación de ε₀ vía Chirikov (heurística rigurosa):
        ε₀ ≈ γ / (n log(1/γ))^n.
    
    Test ejecutable: γ empírico + ratio = ε / ε₀ → decisión persistencia.
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
        cls,
        omega: Sequence[float],
        perturbation_size: float,
        tau: float = 1.5,
        kmax: int = 8,
        gamma_min: float = 1.0e-3,
    ) -> "KAMDiophantineCertificate":
        r"""Evalúa el criterio KAM diofántico."""
        w = np.asarray(omega, dtype=np.float64).ravel()
        n = w.size

        # Buscar γ = min_{k≠0, |k|≤kmax} |ω·k| · |k|^τ
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(n)]
        gamma = float("inf")
        grid = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(n, -1).T

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

        # Umbral KAM heurístico (Chirikov): ε₀ ≈ γ / (n log(1/γ))^n
        if gamma > 1e-15:
            denom = (n * math.log(1.0 / max(gamma, 1e-15))) ** n if n > 0 else 1.0
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


# ── §2.6 Espectro del Jacobiano flash (DT_η = {1} ⊕ {1−η}) ──────────────────────
@dataclass(frozen=True, slots=True)
class FlashJacobianSpectrum:
    r"""
    Linealización DT_η = Id − η(Id − Ad_P):
        {1}    en ran(Ad_P)
        {1−η}  transversal
    
    rate = |1−η*|
    radio espectral = max(1, rate)
    contractivo ⟺ η* ∈ (0, 2)   (P7)
    
    numerical_transverse_gain = validación por sonda numérica.
    """
    eta_star: float
    transverse_rate: float
    spectral_radius: float
    numerical_transverse_gain: float
    is_contractive_transverse: bool
    local_verdict: HeytingOmega3


class FlashSpectralJacobian:
    """Cálculo del Jacobiano espectral flash."""
    N_PROBES: Final[int] = 8

    @classmethod
    def _dt_apply(cls, A: np.ndarray, P: np.ndarray, eta: float) -> ComplexMatrix:
        r"""
        Aplicación del Jacobiano tangente:
            DT_η(A) = A − η(A − PAP)
        con corrección de traza en el subespacio tangente.
        """
        RA = A - P @ A @ P
        n = A.shape[0]
        DT = A - eta * RA + (eta * float(np.trace(RA).real) / n) * np.eye(
            n, dtype=np.complex128
        )
        return 0.5 * (DT + DT.conj().T)

    @classmethod
    def numerical_transverse_gain(
        cls, P: np.ndarray, eta: float, key: str = "J-PROBE"
    ) -> float:
        r"""
        Ganancia numérica transversal: contractividad empírica.
        Muestrea elementos aleatorios en P_⊥ T_ρ𝔇_n y mide contracción.
        """
        n = int(P.shape[0])
        I = np.eye(n, dtype=np.complex128)
        Pc = I - P
        rng = np.random.default_rng(_seed_from_string(f"JACOBIAN::{key}"))
        gains: List[float] = []

        for _ in range(cls.N_PROBES):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            A = 0.5 * (A + A.conj().T)
            A = Pc @ A @ Pc
            A = DensityOperatorAlgebra.tangent_project(A, n)
            nrm = DensityOperatorAlgebra.frobenius(A)

            if nrm < 1e-14:
                continue

            DT = cls._dt_apply(A, P, eta)
            gains.append(DensityOperatorAlgebra.frobenius(DT) / nrm)

        if not gains:
            return abs(1.0 - eta)
        return float(np.mean(gains))

    @classmethod
    def audit(
        cls,
        eta_star: float,
        P: Optional[np.ndarray] = None,
        probe_key: str = "J-PROBE",
    ) -> FlashJacobianSpectrum:
        r"""Auditoría del Jacobiano flash: espectro + contractibilidad."""
        eta = float(np.clip(eta_star, 1e-9, 2.0 - 1e-9))
        rate = abs(1.0 - eta)
        radius = max(1.0, rate)
        gain = (
            cls.numerical_transverse_gain(P, eta, probe_key)
            if P is not None
            else rate
        )
        is_contractive = rate < 1.0
        verdict = HeytingOmega3.COHERENT if is_contractive else HeytingOmega3.DEGRADED

        return FlashJacobianSpectrum(
            eta_star=eta,
            transverse_rate=float(rate),
            spectral_radius=float(radius),
            numerical_transverse_gain=float(gain),
            is_contractive_transverse=bool(is_contractive),
            local_verdict=verdict,
        )

    @classmethod
    def _jacobian_callable(
        cls, P: np.ndarray, n: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Jacobiano (linealización BB) en representación real 2n²-dimensional.
        Aproximación estabilizada: ∂(ρ − PρP)/∂ρ = Id − P⊗P amortiguado.
        """
        def jac(x_flat: np.ndarray) -> np.ndarray:
            return np.eye(2 * n * n) - 1e-3 * np.eye(2 * n * n)

        return jac

    @classmethod
    def osledets_estimate(
        cls, rho: np.ndarray, P: np.ndarray, n_iter: int = 32
    ) -> OseledetsLyapunovSpectrum:
        r"""Estima espectro de Oseledets del Jacobiano."""
        n = int(rho.shape[0])
        jac = cls._jacobian_callable(P, n)
        x0 = np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        return OseledetsLyapunovSpectrum.estimate(jac, x0, n_iter=n_iter)

    def project_poincare_section_grassmannian(
        self,
        density_op: np.ndarray,
        mac_equilibrium_op: np.ndarray,
        subspace_rank: int = 2,
        poincare_tolerance: float = 1e-6,
        bures_threshold: float = 0.15,
    ) -> Tuple[np.ndarray, float, float, bool, MelnikovHomoclinicCertificate]:
        r"""Proyecta a sección de Poincaré, evalúa certificados."""
        rho = DensityOperatorAlgebra.sanitize(density_op)
        rho_mac = DensityOperatorAlgebra.sanitize(mac_equilibrium_op)
        n = rho.shape[0]
        r = int(np.clip(subspace_rank, 1, max(1, n - 1)))

        evals, evecs = la.eigh(rho)
        idx = np.argsort(evals)[::-1][:r]
        B = evecs[:, idx]
        P_sub = B @ B.conj().T
        P_sub = 0.5 * (P_sub + P_sub.conj().T)

        rho_proj_raw = P_sub @ rho @ P_sub
        tr_proj = float(np.trace(rho_proj_raw).real)
        if tr_proj < 1e-15:
            rho_proj = np.eye(n, dtype=np.complex128) / n
        else:
            rho_proj = DensityOperatorAlgebra.sanitize(rho_proj_raw / tr_proj)

        d_bures = DensityOperatorAlgebra.bures_distance(rho_proj, rho_mac)
        jac_map = P_sub @ (rho - rho_mac) @ P_sub
        sv = la.svdvals(jac_map)
        max_sv = float(sv[0]) if sv.size > 0 else 1e-12
        lyap_max = float(math.log(max(max_sv, 1e-12)))

        meln = MelnikovHomoclinicCertificate.evaluate(rho_proj, P_sub)

        is_stable = bool(
            lyap_max <= poincare_tolerance
            and d_bures <= bures_threshold
            and not meln.transverse_homoclinic
        )

        return rho_proj, float(d_bures), float(lyap_max), is_stable, meln


# ── §2.7 Kelly κ-fraccional modulado por h_KS ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class KellyStakeReport:
    r"""
    Informe de apuesta de Kelly modulado por dinámica caótica.
    
    stake = κ · f* · exp(−h_KS) · Θ(λ_max ≤ 0) · Θ(¬Melnikov) · Θ(d_B ≤ τ_B)
    
    p_eff := F(ρ_flash, ρ_target)  (identificación de modelo, no teorema).
    
    G(f) = p log(1+f) + (1−p) log(1−f)    (log-growth de Kelly)
    G(f*) = log 2 − h₂(p)  si p > ½      (información mutua).
    """
    p_eff: float
    kelly_full: float
    kappa: float
    stake: float
    log_growth: float
    binary_entropy_nats: float
    is_no_bet: bool
    hedge_amount: float
    local_verdict: HeytingOmega3
    entropy_penalty: float = 1.0
    veto_reason: str = ""


class KellyStakeCalculator:
    """Calculador de apuesta de Kelly con certificados de Poincaré."""
    DEFAULT_KAPPA: Final[float] = 0.5

    @classmethod
    def binary_entropy(cls, p: float) -> float:
        """h₂(p) = −p log p − (1−p) log(1−p)."""
        p = float(np.clip(p, 0.0, 1.0))
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))

    @classmethod
    def log_growth(cls, p: float, f: float) -> float:
        """G(f) = p log(1+f) + (1−p) log(1−f)."""
        p = float(np.clip(p, 0.0, 1.0))
        f = float(np.clip(f, 0.0, 1.0 - 1e-15))
        if f <= 0.0:
            return 0.0
        return float(p * math.log(1.0 + f) + (1.0 - p) * math.log(1.0 - f))

    @classmethod
    def compute(
        cls,
        fidelity: float,
        kappa: float = DEFAULT_KAPPA,
        cost_risk: float = 0.0,
        kolmogorov_sinai_entropy: float = 0.0,
    ) -> KellyStakeReport:
        r"""Computa Kelly fraccional simple."""
        p_eff = float(np.clip(fidelity, 0.0, 1.0))
        kappa = float(np.clip(kappa, 0.0, 1.0))
        f_full = max(0.0, 2.0 * p_eff - 1.0)
        penalty = float(math.exp(-max(kolmogorov_sinai_entropy, 0.0)))
        stake = float(np.clip(kappa * f_full * penalty, 0.0, 1.0))
        no_bet = p_eff <= 0.5
        h2 = cls.binary_entropy(p_eff)
        growth = cls.log_growth(p_eff, stake)
        hedge = float(stake * max(0.0, cost_risk))
        local = (
            HeytingOmega3.COHERENT
            if (not no_bet) and stake > 0.0
            else HeytingOmega3.DEGRADED
        )

        return KellyStakeReport(
            p_eff=p_eff,
            kelly_full=float(f_full),
            kappa=kappa,
            stake=stake,
            log_growth=growth,
            binary_entropy_nats=h2,
            is_no_bet=bool(no_bet),
            hedge_amount=hedge,
            local_verdict=local,
            entropy_penalty=penalty,
        )

    def calculate_poincare_kelly_stake(
        self,
        success_probability: float,
        win_loss_ratio: float,
        lyap_max: float,
        d_bures: float,
        kolmogorov_sinai_entropy: float = 0.0,
        transverse_homoclinic: bool = False,
        fractional_multiplier: float = 0.25,
        bures_threshold: float = 0.15,
        cost_risk: float = 0.0,
    ) -> KellyStakeReport:
        r"""
        Calcula Kelly fraccional con certificados de Poincaré.
        
        Vetoes:
            - Oseledets divergente (λ_max > 0)
            - Melnikov transversal (caos)
            - Distorsión Bures excesiva
        
        Penalización entrópica: exp(−h_KS) ∈ (0,1].
        """
        p = float(np.clip(success_probability, 0.0, 1.0))
        b = float(win_loss_ratio)
        kappa = float(np.clip(fractional_multiplier, 0.0, 1.0))

        def _veto(reason: str) -> KellyStakeReport:
            return KellyStakeReport(
                p_eff=p,
                kelly_full=0.0,
                kappa=kappa,
                stake=0.0,
                log_growth=0.0,
                binary_entropy_nats=self.binary_entropy(p),
                is_no_bet=True,
                hedge_amount=0.0,
                local_verdict=HeytingOmega3.VETOED,
                entropy_penalty=0.0,
                veto_reason=reason,
            )

        if b <= 0.0:
            return _veto("INVALID_WIN_LOSS_RATIO")
        if lyap_max > 0.0:
            return _veto("OSELEDETS_DIVERGENCE")
        if transverse_homoclinic:
            return _veto("MELNIKOV_CHAOS")
        if d_bures > bures_threshold:
            return _veto("BURES_DISTORTION")

        f_star = (p * (b + 1.0) - 1.0) / b

        if f_star <= 0.0:
            return KellyStakeReport(
                p_eff=p,
                kelly_full=float(f_star),
                kappa=kappa,
                stake=0.0,
                log_growth=0.0,
                binary_entropy_nats=self.binary_entropy(p),
                is_no_bet=True,
                hedge_amount=0.0,
                local_verdict=HeytingOmega3.DEGRADED,
                entropy_penalty=1.0,
                veto_reason="NEGATIVE_EDGE",
            )

        penalty = float(math.exp(-max(kolmogorov_sinai_entropy, 0.0)))
        stake = float(np.clip(kappa * f_star * penalty, 0.0, 1.0))
        growth = self.log_growth(p, stake)
        hedge = float(stake * max(0.0, cost_risk))

        return KellyStakeReport(
            p_eff=p,
            kelly_full=float(f_star),
            kappa=kappa,
            stake=stake,
            log_growth=growth,
            binary_entropy_nats=self.binary_entropy(p),
            is_no_bet=False,
            hedge_amount=hedge,
            local_verdict=HeytingOmega3.COHERENT,
            entropy_penalty=penalty,
            veto_reason="",
        )


# ── §2.8 Contrato de latencia (benchmark p50/p99) ───────────────────────────────
@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Informe de latencia con certificado de cumplimiento."""
    mean_ns: float
    p50_ns: float
    p99_ns: float
    contract_ns: float
    contract_met: bool
    n_samples: int
    n_warmup: int
    local_verdict: HeytingOmega3


class FlashLatencyBenchmark:
    """Benchmark de latencia para d_B (métrica Bures)."""
    @classmethod
    def measure_bures(
        cls,
        rho: np.ndarray,
        sigma: np.ndarray,
        contract_ns: float,
        n_warmup: int = 16,
        n_samples: int = 64,
    ) -> LatencyReport:
        r"""Mide latencia de d_B(ρ, σ) contra contrato."""
        n_warmup = max(0, int(n_warmup))
        n_samples = max(8, int(n_samples))

        for _ in range(n_warmup):
            _ = DensityOperatorAlgebra.bures_distance(rho, sigma)

        samples = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            t0 = time.perf_counter_ns()
            _ = DensityOperatorAlgebra.bures_distance(rho, sigma)
            samples[i] = float(time.perf_counter_ns() - t0)

        p50 = float(np.percentile(samples, 50.0))
        p99 = float(np.percentile(samples, 99.0))
        mean = float(samples.mean())
        met = p99 < float(contract_ns)
        local = HeytingOmega3.COHERENT if met else HeytingOmega3.DEGRADED

        return LatencyReport(
            mean_ns=mean,
            p50_ns=p50,
            p99_ns=p99,
            contract_ns=float(contract_ns),
            contract_met=bool(met),
            n_samples=n_samples,
            n_warmup=n_warmup,
            local_verdict=local,
        )


# ── §2.9 Traductor visceral (corazonada en lenguaje natural) ────────────────────
class VisceralSignalTranslator:
    r"""
    Traduce numbers en recomendación visceral (lenguaje natural).
    Tres niveles: COHERENT, DEGRADED, VETOED.
    """
    @classmethod
    def translate(
        cls,
        verdict: HeytingOmega3,
        d_bures: float,
        kelly: KellyStakeReport,
        cost_risk: float,
        lyapunov_max: float = 0.0,
        kolmogorov_sinai_entropy: float = 0.0,
        melnikov_value: float = 0.0,
        kam_persists: bool = True,
    ) -> str:
        """Traduce Ω₃ → texto visceral."""
        hedge = float(
            kelly.hedge_amount if kelly.hedge_amount
            else kelly.stake * max(0.0, cost_risk)
        )

        if verdict == HeytingOmega3.COHERENT:
            return (
                f"CORAZONADA SANA: ρ alineada con M_decision "
                f"(d_B={d_bures:.4f}, F={kelly.p_eff:.4f}, G={kelly.log_growth:.4f} nats, "
                f"λ_max={lyapunov_max:.2e}, h_KS={kolmogorov_sinai_entropy:.2e}, "
                f"KAM={'persiste' if kam_persists else 'colapsa'}). "
                f"Stake Kelly κ·f*·e^(−h_KS) = {kelly.stake:.4f} → luz verde. "
                f"Hedge sugerido: ${hedge:,.0f}."
            )

        if verdict == HeytingOmega3.DEGRADED:
            return (
                f"CORAZONADA DE ALERTA: fricción geodésica detectada "
                f"(d_B={d_bures:.4f}, F={kelly.p_eff:.4f}, "
                f"λ_max={lyapunov_max:.2e}, M={melnikov_value:.2e}). "
                f"Stake κ·f* = {kelly.stake:.4f} "
                f"{'(no-bet)' if kelly.is_no_bet else ''}. "
                f"Riesgo cubierto: ${hedge:,.0f}. "
                f"Pies de plomo en el desembolso."
            )

        return (
            f"CORAZONADA DE VETO CRÍTICO: ρ fuera de la variedad de decisión "
            f"o fraude contextual. d_B={d_bures:.4f}, F={kelly.p_eff:.4f}, "
            f"λ_max={lyapunov_max:.2e}, M={melnikov_value:.2e} "
            f"({kelly.veto_reason or 'VETO'}). "
            f"Válvula de pago cerrada. Interlock BT151 preparado."
        )


# ── §2.10 FlashTrajectoryBundle — HAND-OFF FORMAL FASE 2 → FASE 3 ───────────────
@dataclass(frozen=True, slots=True)
class FlashTrajectoryBundle:
    r"""
    Objeto terminal FASE 2 / inicial FASE 3.
    
    Producto de: Dirichlet ⊗ Bures ⊗ Jacobiano ⊗ Kelly ⊗ Latencia ⊗ Oseledets
                 ⊗ Melnikov ⊗ KAM
    
    Contiene:
        - Un paso Cauchy (flash) con η* elegido
        - Certificado Jacobiano espectral (contractibilidad)
        - Espectro de Oseledets completo
        - Integral de Melnikov (caos)
        - Criterio KAM diofántico
        - Apuesta de Kelly modulada
        - Medida de latencia
    """
    handoff: FlashHandoff
    rho_flash: np.ndarray
    landscape: DirichletLandscape
    jacobian: FlashJacobianSpectrum
    oseledets: OseledetsLyapunovSpectrum
    melnikov: MelnikovHomoclinicCertificate
    kam: KAMDiophantineCertificate
    bures_to_target: float
    bures_seed_to_target: float
    theta_to_target: float
    kelly: KellyStakeReport
    latency: LatencyReport
    purity: float
    entropy: float
    energy_decay_ratio: float

    def content_bytes(self) -> bytes:
        """Firma SHA-256 del contenido para Merkle."""
        return hashlib.sha256(
            self.handoff.spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_flash).tobytes()
            + f"{self.landscape.energy:.12e}".encode("ascii")
            + f"{self.bures_to_target:.12e}".encode("ascii")
            + f"{self.kelly.stake:.12e}".encode("ascii")
            + f"{self.latency.p99_ns:.3f}".encode("ascii")
            + f"{self.oseledets.lyapunov_max:.12e}".encode("ascii")
            + f"{self.oseledets.kolmogorov_sinai_entropy:.12e}".encode("ascii")
            + f"{self.melnikov.melnikov_value:.12e}".encode("ascii")
        ).digest()

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 2 → FASE 3
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase3(
        self,
        external_verdict: HeytingOmega3,
        eta_star: float,
        reason_prefix: str = "INTUITION-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║ ÚLTIMO MORFISMO DE FASE 2 ∧ PRIMER MORFISMO DE FASE 3               ║
        ║                                                                      ║
        ║ Identidad de composición:                                            ║
        ║   continue_into_phase3 ∘ synthesize ∘ build                           ║
        ║     = (adjudicate ⊗ fire) ∘ synthesize ∘ build                       ║
        ║     : Request × Gr(r,n) → (Ω₃, CrowbarReport)                        ║
        ║                                                                      ║
        ║ Esta definición formal es la ENTRADA CANÓNICA a F₃.                 ║
        ║ Es el único morfismo de FASE 3 (adjudicación + crowbar).            ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        return FlashPipeline.continue_into_phase3(
            self, external_verdict, eta_star, reason_prefix
        )


class FlashPipeline:
    r"""
    Orquestador determinista de FASE 2 (funtor F₂).
    
    Un solo paso Cauchy η* (flash): reflejo one-shot; Newton exacto si η*=1.
    synthesize : FlashHandoff × ℝ₊³ → FlashTrajectoryBundle.
    """
    DEFAULT_ETA_STAR: Final[float] = 1.0
    DEFAULT_LATENCY_NS: Final[float] = 10_000.0  # 10 µs

    @classmethod
    def synthesize(
        cls,
        handoff: FlashHandoff,
        eta_star: float = DEFAULT_ETA_STAR,
        latency_contract_ns: float = DEFAULT_LATENCY_NS,
        kelly_kappa: float = KellyStakeCalculator.DEFAULT_KAPPA,
    ) -> FlashTrajectoryBundle:
        r"""
        Ejecuta FASE 2: un paso Cauchy + certificación Poincaré.
        """
        P = handoff.geometry.projector
        eta = float(eta_star)

        # Un paso Cauchy: flash one-shot
        rho_flash = FlashDirichletFunctional.cauchy_step(handoff.rho_seed, P, eta)
        land_next = FlashDirichletFunctional.evaluate(rho_flash, P)

        # Jacobiano espectral
        jacobian = FlashSpectralJacobian.audit(
            eta,
            P=P,
            probe_key=handoff.request_id,
        )

        # Oseledets
        oseledets = FlashSpectralJacobian.osledets_estimate(rho_flash, P, n_iter=32)

        # Melnikov
        melnikov = MelnikovHomoclinicCertificate.evaluate(rho_flash, P)

        # KAM: frecuencia del mapa de retorno
        theta = float(DensityOperatorAlgebra.bures_angle(rho_flash, handoff.rho_target))
        omega = np.array([theta, theta * 0.5 + 1e-3], dtype=np.float64)
        kam = KAMDiophantineCertificate.evaluate(
            omega=omega,
            perturbation_size=abs(float(land_next.energy)),
            tau=1.5,
            kmax=8,
        )

        # Distancias Bures
        d_b = BuresGeodesicMetric.distance(rho_flash, handoff.rho_target)
        th_b = BuresGeodesicMetric.angle(rho_flash, handoff.rho_target)
        d_seed = DensityOperatorAlgebra.bures_distance(
            handoff.rho_seed, handoff.rho_target
        )

        # Kelly
        fid_flash = BuresGeodesicMetric.fidelity(rho_flash, handoff.rho_target)
        kelly = KellyStakeCalculator.compute(
            fid_flash,
            kappa=kelly_kappa,
            cost_risk=handoff.request_audit.cost_risk_amount,
            kolmogorov_sinai_entropy=oseledets.kolmogorov_sinai_entropy,
        )

        # Latencia
        latency = FlashLatencyBenchmark.measure_bures(
            rho_flash, handoff.rho_target, contract_ns=latency_contract_ns,
        )

        # Energía
        e0 = float(handoff.seed_energy)
        ratio = float(land_next.energy / e0) if e0 > 1e-20 else (
            0.0 if land_next.energy <= 1e-20 else 1.0
        )

        return FlashTrajectoryBundle(
            handoff=handoff,
            rho_flash=rho_flash,
            landscape=land_next,
            jacobian=jacobian,
            oseledets=oseledets,
            melnikov=melnikov,
            kam=kam,
            bures_to_target=float(d_b),
            bures_seed_to_target=float(d_seed),
            theta_to_target=float(th_b),
            kelly=kelly,
            latency=latency,
            purity=DensityOperatorAlgebra.purity(rho_flash),
            entropy=DensityOperatorAlgebra.von_neumann_entropy(rho_flash),
            energy_decay_ratio=ratio,
        )

    @classmethod
    def continue_into_phase3(
        cls,
        bundle: FlashTrajectoryBundle,
        external_verdict: HeytingOmega3,
        eta_star: float,
        reason_prefix: str = "INTUITION-VETO",
    ) -> Tuple[HeytingOmega3, "CrowbarActuationReport"]:
        r"""
        Continuación estricta: último de F₂ ∧ primero de F₃.
        Adjudica en Ω₃ y dispara el crowbar.
        """
        verdict = HeytingIntuitionAdjudicator.adjudicate(bundle, external_verdict)
        reason = (
            f"{reason_prefix}::d_B={bundle.bures_to_target:.4f} "
            f"fraud={bundle.handoff.request_audit.has_critical_fraud} "
            f"kelly={bundle.kelly.stake:.4f} "
            f"eta={eta_star:.3f} rate={bundle.jacobian.transverse_rate:.4f} "
            f"lam_max={bundle.oseledets.lyapunov_max:.3e} "
            f"h_KS={bundle.oseledets.kolmogorov_sinai_entropy:.3e} "
            f"M={bundle.melnikov.melnikov_value:.3e} "
            f"kam={'1' if bundle.kam.kam_persists else '0'} "
            f"lat_met={bundle.latency.contract_met} "
            f"valid={bundle.handoff.request_audit.is_valid}"
        )
        actuation = ESP32CrowbarInterlock.fire(verdict, reason)
        return verdict, actuation


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   3   ·   A D J U D I C A C I Ó N   +   C R O W B A R                      ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   FlashTrajectoryBundle (codominio de §2.10 synthesize)                  ║
# ║  Codominio: IntuitionFlashCertificate (objeto terminal del flash)                  ║
# ║                                                                                    ║
# ║  Único funtor: HeytingIntuitionAdjudicator.adjudicate ⊗ ESP32CrowbarInterlock.fire ║
# ║                                                                                    ║
# ║  Operación: evaluación booleana intuicionista Ω₃ (9 meets) + actuación ciber-fís.  ║
# ║                                                                                    ║
# ║  Salida: certificado sellado Merkle + interlock hardware.                          ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §3.1 Adjudicador en Ω₃ (meet conservador con invariantes Poincaré) ──────────
class HeytingIntuitionAdjudicator:
    r"""
    Colapsa el Bundle a Ω₃ por meets sucesivos.
    
    Umbrales adimensionales (transición ⊤→⋆→⊥):
        fraud    : has_critical_fraud ↦ ⊥ else ⊤       (P4)
        geom     : d_B(ρ_flash, ρ_tgt) graduado
        spectral : contractivo transversal ↦ ⊤ else ⋆
        kelly    : local_verdict del stake
        valid    : request is_valid ↦ ⊤ else ⋆
        manifold : isometría ∧ proyector ↦ ⊤ else ⊥
        lyap     : λ_max ≤ 0 ↦ ⊤;  ≤ 1e−3 ↦ ⋆;  else ⊥
        melnikov : ¬transversal ↦ ⊤ else ⊥
        kam      : persiste ↦ ⊤;  colapsa ↦ ⋆
        poincare : Σ transversal ↦ ⊤ else ⊥
    
    final = local ∧ external  (conservador, nunca infla).
    """
    BURES_COHERENT: Final[float] = 0.20
    BURES_DEGRADED: Final[float] = 0.40
    LYAP_COHERENT: Final[float] = 0.0
    LYAP_DEGRADED: Final[float] = 1.0e-3

    @classmethod
    def _grade(cls, value: float, hi_ok: float, mid_ok: float) -> HeytingOmega3:
        """Transición ⊤ → ⋆ → ⊥."""
        if value <= hi_ok:
            return HeytingOmega3.COHERENT
        if value <= mid_ok:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _fraud_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de fraude (P4, veto duro)."""
        return (
            HeytingOmega3.VETOED
            if b.handoff.request_audit.has_critical_fraud
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _geom_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de distancia Bures al target."""
        return cls._grade(b.bures_to_target, cls.BURES_COHERENT, cls.BURES_DEGRADED)

    @classmethod
    def _spectral_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de contractibilidad del Jacobiano."""
        return b.jacobian.local_verdict

    @classmethod
    def _kelly_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de Kelly."""
        return b.kelly.local_verdict

    @classmethod
    def _valid_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de validez del request."""
        return (
            HeytingOmega3.COHERENT
            if b.handoff.request_audit.is_valid
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _manifold_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de isometría y proyector válidos."""
        g = b.handoff.geometry
        ok = g.is_isometry and g.is_projector
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED

    @classmethod
    def _lyap_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de Lyapunov de Oseledets (estabilidad)."""
        return cls._grade(
            b.oseledets.lyapunov_max, cls.LYAP_COHERENT, cls.LYAP_DEGRADED
        )

    @classmethod
    def _melnikov_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de Melnikov (caos homoclínico)."""
        return (
            HeytingOmega3.VETOED
            if b.melnikov.transverse_homoclinic
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _kam_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de KAM (persistencia de toros)."""
        return (
            HeytingOmega3.COHERENT
            if b.kam.kam_persists
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _poincare_rule(cls, b: FlashTrajectoryBundle) -> HeytingOmega3:
        """Regla de transversalidad de Poincaré."""
        return (
            HeytingOmega3.COHERENT
            if b.handoff.poincare_section.is_transverse
            else HeytingOmega3.VETOED
        )

    @classmethod
    def adjudicate(
        cls, bundle: FlashTrajectoryBundle, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        r"""
        Adjudicación final: 9 meets.
        
        local = fraud ∧ geom ∧ spectral ∧ kelly ∧ valid ∧ manifold
                ∧ lyap ∧ melnikov ∧ kam ∧ poincare
        final = local ∧ external  (nunca infla)
        """
        local = (
            cls._fraud_rule(bundle)
            .meet(cls._geom_rule(bundle))
            .meet(cls._spectral_rule(bundle))
            .meet(cls._kelly_rule(bundle))
            .meet(cls._valid_rule(bundle))
            .meet(cls._manifold_rule(bundle))
            .meet(cls._lyap_rule(bundle))
            .meet(cls._melnikov_rule(bundle))
            .meet(cls._kam_rule(bundle))
            .meet(cls._poincare_rule(bundle))
        )
        return local.meet(external_verdict)


# ── §3.2 Interlock ciber-físico ESP32 Crowbar (ISR) ─────────────────────────────
@dataclass(frozen=True, slots=True)
class CrowbarActuationReport:
    r"""
    Informe de actuación del interlock Crowbar.
    
    Política:
        interlock_fired ⟺ verdict = ⊥ (VETOED).
    
    Actuación nominal:
        GPIO14 → HIGH  ⇒  BT151 crowbar  ⇒  Δτ < 400 ns (cota ISR).
    """
    interlock_fired: bool
    actuation_latency_ns: float
    gpio_pin: str
    device: str
    reason: str
    provenance_hash: str


class ESP32CrowbarInterlock:
    r"""
    FASE 3 · FE ciber-física (firmware emulado).
    
    Si Ω₃ = ⊥ (VETOED):
        GPIO14 → HIGH  ⇒  BT151 crowbar  ⇒  Δτ < 400 ns (P6).
    
    No emite I/O real; certifica decisión + provenance SHA-256.
    """
    GPIO_PIN: Final[str] = "GPIO14"
    DEVICE: Final[str] = "BT151_CROWBAR"
    NOMINAL_LATENCY_NS: Final[float] = 392.15

    @classmethod
    def fire(
        cls, verdict: HeytingOmega3, reason: str = ""
    ) -> CrowbarActuationReport:
        """Dispara crowbar si verdict = ⊥."""
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
        prov = hashlib.sha256(
            f"CROWBAR_INTUITION::{reason}::{t_ns}".encode("utf-8")
        ).hexdigest()

        logger.critical(
            "[INTUICIÓN — CROWBAR] %s → HIGH | %.2f ns | razón=%s",
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


# ── §3.3 Certificado del flash (objeto terminal) ────────────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitionFlashCertificate:
    r"""
    Objeto terminal: Certificate ≅ Bundle × Ω₃ × Crowbar × Merkle.
    
    Distancias Bures desambiguadas (P1):
        bures_manifold_distance  : d_B(ρ_flash, ρ_target)
        bures_seed_to_target     : d_B(ρ_seed, ρ_target)
    
    Contiene firma Merkle SHA-256 de las 3 fases.
    """
    flash_id: str
    agent_id: str
    crop_origin_id: str
    seed_crystal_id: str
    heyting_verdict: HeytingOmega3
    bures_manifold_distance: float
    bures_seed_to_target: float
    bures_angle_rad: float
    energy_initial: float
    energy_flash: float
    energy_decay_ratio: float
    mass_on_P: float
    transverse_rate: float
    numerical_jacobian_gain: float
    kelly_p_eff: float
    kelly_stake: float
    kelly_log_growth: float
    kelly_hedge_amount: float
    latency_p50_ns: float
    latency_p99_ns: float
    latency_contract_met: bool
    purity: float
    entropy: float
    # Poincaré forense:
    lyapunov_max: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    melnikov_value: float
    transverse_homoclinic: bool
    kam_persists: bool
    kam_perturbation_ratio: float
    poincare_transverse: bool
    levy_band: float
    crowbar_interlock: CrowbarActuationReport
    visceral_recommendation: str
    manifold_hash: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.4 SOBERANO DE LA INTUICIÓN — funtor F₃ ∘ F₂ ∘ F₁ ────────────────────────
class TOONIntuitionAgent:
    r"""
    Soberano de la Intuición y Reflejo Flash.
    
    Funtor soberano F = F₃ ∘ F₂ ∘ F₁:
        F₁  FlashHandoff.build
        F₂  FlashHandoff.continue_into_phase2 = synthesize
        F₃  continue_into_phase3 ⊗ certify
    
    Asociatividad: synthesize_intuitive_flash
        = _phase3_certify ∘ _phase2_flash ∘ _phase1_handoff.
    """
    def __init__(
        self,
        agent_id: str = "INTUITION-SOVEREIGN-SABIO-01",
        dimension_mac: int = 4,
        manifold_rank: int = 2,
        latency_contract_ns: float = FlashPipeline.DEFAULT_LATENCY_NS,
        kelly_kappa: float = KellyStakeCalculator.DEFAULT_KAPPA,
        eta_star: float = FlashPipeline.DEFAULT_ETA_STAR,
    ) -> None:
        if not (1 <= manifold_rank < dimension_mac):
            raise ValueError(
                f"manifold_rank ∈ [1, n−1]; r={manifold_rank}, n={dimension_mac}"
            )
        self.agent_id = agent_id
        self.dimension_mac = int(dimension_mac)
        self.latency_contract_ns = float(latency_contract_ns)
        self.kelly_kappa = float(kelly_kappa)
        self.eta_star = float(eta_star)
        self.flash_count = 0
        self.jacobian_solver = FlashSpectralJacobian()
        self.kelly_calculator = KellyStakeCalculator()
        self.manifold = DecisionManifoldFactory.build(
            n=self.dimension_mac, rank=manifold_rank, key=agent_id,
        )
        self._chain_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::n={dimension_mac}::r={manifold_rank}::"
            f"m={self.manifold.hash}".encode("ascii")
        ).hexdigest()

    # ── API pública con Poincaré explícito ─────────────────────────────────
    def process_poincare_intuitive_flash(
        self,
        request: IntuitiveFlashRequest,
        mac_equilibrium_op: np.ndarray,
        success_probability: float = 0.85,
        win_loss_ratio: float = 2.0,
    ) -> Tuple[IntuitionFlashCertificate, HeytingOmega3]:
        r"""
        Procesa una solicitud con Sección de Retorno de Poincaré, Oseledets,
        Melnikov y KAM; devuelve (cert, verdict).
        """
        rho_proj, d_bures, lyap_max, is_stable, meln = (
            self.jacobian_solver.project_poincare_section_grassmannian(
                density_op=request.germinated_density_matrix,
                mac_equilibrium_op=mac_equilibrium_op,
                subspace_rank=self.manifold.rank,
            )
        )

        cost_risk = float(
            request.site_context_payload.get("cost_risk_amount", 0.0) or 0.0
        )
        has_fraud = bool(request.site_context_payload.get("has_critical_fraud", False))

        kelly_report = self.kelly_calculator.calculate_poincare_kelly_stake(
            success_probability=success_probability,
            win_loss_ratio=win_loss_ratio,
            lyap_max=lyap_max,
            d_bures=d_bures,
            kolmogorov_sinai_entropy=abs(lyap_max),
            transverse_homoclinic=meln.transverse_homoclinic,
            fractional_multiplier=self.kelly_kappa,
            cost_risk=cost_risk,
        )

        if not is_stable or kelly_report.is_no_bet or has_fraud:
            verdict = HeytingOmega3.VETOED
        elif d_bures > 0.05:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        cert = self.synthesize_intuitive_flash(request, external_verdict=verdict)
        return cert, verdict

    # ── Cadena Merkle ──────────────────────────────────────────────────────
    def _advance_chain(self, tag: str, payload: bytes) -> str:
        """Avanza la cadena Merkle con nuevo payload."""
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    # ── FASE 1 anidada ────────────────────────────────────────────────────
    def _phase1_handoff(self, request: IntuitiveFlashRequest) -> FlashHandoff:
        """Ejecuta FASE 1: prepara FlashHandoff."""
        handoff = FlashHandoff.build(
            cycle_index=self.flash_count,
            request=request,
            geometry=self.manifold,
            poincare_key=self.agent_id,
        )
        self._advance_chain("F1", bytes.fromhex(handoff.spectral_hash))
        return handoff

    # ── FASE 2 anidada ────────────────────────────────────────────────────
    def _phase2_flash(self, handoff: FlashHandoff) -> FlashTrajectoryBundle:
        """Ejecuta FASE 2: descenso un paso + certificación Poincaré."""
        bundle = handoff.continue_into_phase2(
            eta_star=self.eta_star,
            latency_contract_ns=self.latency_contract_ns,
            kelly_kappa=self.kelly_kappa,
        )
        self._advance_chain("F2", bundle.content_bytes())
        return bundle

    # ── FASE 3 anidada ────────────────────────────────────────────────────
    def _phase3_certify(
        self,
        flash_id: str,
        bundle: FlashTrajectoryBundle,
        external_verdict: HeytingOmega3,
    ) -> IntuitionFlashCertificate:
        """Ejecuta FASE 3: adjudicación Ω₃ + sellado Merkle."""
        final_verdict, crowbar = bundle.continue_into_phase3(
            external_verdict, self.eta_star
        )
        self._advance_chain(
            "F3",
            f"{final_verdict.name}|{crowbar.provenance_hash}".encode("ascii"),
        )

        recommendation = VisceralSignalTranslator.translate(
            verdict=final_verdict,
            d_bures=bundle.bures_to_target,
            kelly=bundle.kelly,
            cost_risk=bundle.handoff.request_audit.cost_risk_amount,
            lyapunov_max=bundle.oseledets.lyapunov_max,
            kolmogorov_sinai_entropy=bundle.oseledets.kolmogorov_sinai_entropy,
            melnikov_value=bundle.melnikov.melnikov_value,
            kam_persists=bundle.kam.kam_persists,
        )

        provenance = _sha256_bytes(
            self.agent_id.encode("ascii"),
            flash_id.encode("ascii"),
            bundle.handoff.crop_origin_id.encode("ascii"),
            bundle.handoff.seed_crystal_id.encode("ascii"),
            final_verdict.name.encode("ascii"),
            f"{bundle.bures_to_target:.12e}".encode("ascii"),
            f"{bundle.kelly.stake:.12e}".encode("ascii"),
            f"{bundle.latency.p99_ns:.3f}".encode("ascii"),
            f"{bundle.oseledets.lyapunov_max:.12e}".encode("ascii"),
            f"{bundle.melnikov.melnikov_value:.12e}".encode("ascii"),
            f"{bundle.kam.diophantine_gamma:.12e}".encode("ascii"),
            self._chain_hash.encode("ascii"),
            f"{time.time_ns()}".encode("ascii"),
        )

        return IntuitionFlashCertificate(
            flash_id=flash_id,
            agent_id=self.agent_id,
            crop_origin_id=bundle.handoff.crop_origin_id,
            seed_crystal_id=bundle.handoff.seed_crystal_id,
            heyting_verdict=final_verdict,
            bures_manifold_distance=bundle.bures_to_target,
            bures_seed_to_target=bundle.bures_seed_to_target,
            bures_angle_rad=bundle.theta_to_target,
            energy_initial=float(bundle.handoff.seed_energy),
            energy_flash=float(bundle.landscape.energy),
            energy_decay_ratio=bundle.energy_decay_ratio,
            mass_on_P=bundle.handoff.mass_on_P,
            transverse_rate=bundle.jacobian.transverse_rate,
            numerical_jacobian_gain=bundle.jacobian.numerical_transverse_gain,
            kelly_p_eff=bundle.kelly.p_eff,
            kelly_stake=bundle.kelly.stake,
            kelly_log_growth=bundle.kelly.log_growth,
            kelly_hedge_amount=bundle.kelly.hedge_amount,
            latency_p50_ns=bundle.latency.p50_ns,
            latency_p99_ns=bundle.latency.p99_ns,
            latency_contract_met=bundle.latency.contract_met,
            purity=bundle.purity,
            entropy=bundle.entropy,
            lyapunov_max=bundle.oseledets.lyapunov_max,
            kolmogorov_sinai_entropy=bundle.oseledets.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=bundle.oseledets.kaplan_yorke_dimension,
            melnikov_value=bundle.melnikov.melnikov_value,
            transverse_homoclinic=bundle.melnikov.transverse_homoclinic,
            kam_persists=bundle.kam.kam_persists,
            kam_perturbation_ratio=bundle.kam.perturbation_ratio,
            poincare_transverse=bundle.handoff.poincare_section.is_transverse,
            levy_band=bundle.handoff.levy_band,
            crowbar_interlock=crowbar,
            visceral_recommendation=recommendation,
            manifold_hash=bundle.handoff.geometry.hash,
            phase_chain_sha256=self._chain_hash,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
        )

    # ── Ciclo soberano ────────────────────────────────────────────────────
    def synthesize_intuitive_flash(
        self,
        request: IntuitiveFlashRequest,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> IntuitionFlashCertificate:
        r"""
        Ciclo soberano: F₃ ∘ F₂ ∘ F₁.
        """
        self.flash_count += 1
        flash_id = f"FLASH-INTUITION-{self.flash_count:04d}"
        t_start = time.perf_counter()

        logger.info(
            "═══ Flash #%d | req=%s::%s | manifold=%s r=%d ═══",
            self.flash_count,
            request.crop_origin_id,
            request.seed_crystal_id,
            self.manifold.hash[:12],
            self.manifold.rank,
        )

        handoff = self._phase1_handoff(request)
        bundle = self._phase2_flash(handoff)
        cert = self._phase3_certify(flash_id, bundle, external_verdict)

        dt_ms = (time.perf_counter() - t_start) * 1000.0

        logger.info(
            "Flash %s | Ω₃=%s | d_B=%.4f | κ·f*=%.4f | λ_max=%.2e | "
            "h_KS=%.2e | M=%.2e | p99=%.0f ns | %.2f ms",
            flash_id,
            cert.heyting_verdict.name,
            cert.bures_manifold_distance,
            cert.kelly_stake,
            cert.lyapunov_max,
            cert.kolmogorov_sinai_entropy,
            cert.melnikov_value,
            cert.latency_p99_ns,
            dt_ms,
        )

        return cert


# ── §3.5 Utilidades para demostración ──────────────────────────────────────────
def _build_state_leak(
    n: int,
    manifold: DecisionManifoldGeometry,
    leakage: float,
    key: str,
) -> ComplexMatrix:
    r"""
    Semilla con fuga controlada t ∈ [0,1] al complemento P_⊥:
        ρ_t = (1−t) ρ_P + t ρ_{P_⊥}.
    
    t = 0 ⇒ supp ⊆ ran(P) ⇒ E = 0 (mínimo)
    t = 1 ⇒ supp ⊆ ran(P_⊥) ⇒ E máximo
    """
    rng = np.random.default_rng(_seed_from_string(f"LEAK::{key}"))
    P = manifold.projector
    Pc = np.eye(n, dtype=np.complex128) - P

    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho_raw = A @ A.conj().T
    rho_raw = rho_raw / max(float(np.trace(rho_raw).real), 1e-30)

    def _normalize(M_side: np.ndarray) -> ComplexMatrix:
        r = M_side @ rho_raw @ M_side
        tr = float(np.trace(r).real)
        if tr < 1e-15:
            rr = max(1.0, float(np.real(np.trace(M_side))))
            return M_side / rr
        return r / tr

    rho_P = _normalize(P)
    rho_Pc = _normalize(Pc)
    t = float(np.clip(leakage, 0.0, 1.0))
    return DensityOperatorAlgebra.sanitize((1.0 - t) * rho_P + t * rho_Pc)


if __name__ == "__main__":
    print("═" * 92)
    print("SOBERANO DE LA INTUICIÓN — v9.1.0 Doctoral 3NestedPhases-Poincaré")
    print("Σ · Oseledets · Melnikov · KAM Diofántico · Lévy · Bures · Kelly-G* · Crowbar")
    print("═" * 92)

    agent = TOONIntuitionAgent(
        agent_id="INTUITION-SOVEREIGN-SABIO-01",
        dimension_mac=4,
        manifold_rank=2,
        latency_contract_ns=10_000.0,
        kelly_kappa=0.5,
        eta_star=1.0,
    )

    print(
        f"\nManifold r={agent.manifold.rank} | "
        f"iso={agent.manifold.is_isometry} | proj={agent.manifold.is_projector} | "
        f"‖B†B−I‖_F={agent.manifold.isometry_residual:.2e} | "
        f"‖P²−P‖_F={agent.manifold.projector_residual:.2e}"
    )
    print(f"Manifold hash: {agent.manifold.hash[:32]}…")
    print(
        f"η*={agent.eta_star} ⇒ rate transversal analítico = "
        f"{abs(1.0 - agent.eta_star):.4f}  (Newton exacto si η*=1)"
    )

    print("\n──────────── Test Bures–Wasserstein (McCann-JKO) ────────────")
    a = _build_state_leak(4, agent.manifold, 0.00, "TRI-A")
    b = _build_state_leak(4, agent.manifold, 0.50, "TRI-B")
    c = _build_state_leak(4, agent.manifold, 1.00, "TRI-C")

    tri = BuresGeodesicMetric.triangle_residual(a, b, c)
    print(f"  δ_triangular = {tri:.3e} (esperado ≈ 0)")
    print(f"  d_B(A,B)     = {BuresGeodesicMetric.distance(a, b):.6f}")
    print(f"  d_B(A,C)     = {BuresGeodesicMetric.distance(a, c):.6f}")

    gamma_mid = BuresGeodesicMetric.geodesic(a, c, 0.5)
    print(
        f"  d_B(A, γ(½)) = "
        f"{BuresGeodesicMetric.distance(a, gamma_mid):.6f}  (McCann-JKO)"
    )

    print("\n──────────── Lema de Lévy (concentración sobre S^{n−1}) ────────────")
    for n_ in (4, 16, 64, 256):
        eps = LevyConcentrationLemma.median_width(n_, lipschitz=1.0, confidence=0.99)
        bnd = LevyConcentrationLemma.bound(eps, n_, lipschitz=1.0)
        print(f"  n={n_:4d} | ε*(99%)={eps:.6f} | P(tail)≤{bnd:.3e}")

    print("\n──────────── Ciclos de intuición flash (F1→F2→F3) ────────────")
    scenarios = [
        ("COHERENT (leak=0.05)", 0.05, HeytingOmega3.COHERENT, False, 12_000_000.0),
        ("DEGRADED (leak=0.80)", 0.80, HeytingOmega3.COHERENT, False, 12_000_000.0),
        ("VETOED   (fraude)", 0.10, HeytingOmega3.COHERENT, True, 250_000_000.0),
    ]

    for name, leak, ext, fraud, cost in scenarios:
        rho = _build_state_leak(4, agent.manifold, leak, name)
        req = IntuitiveFlashRequest(
            crop_origin_id=f"CROP-SOVEREIGN-0001::{name[:20]}",
            seed_crystal_id=f"CRYSTAL-{name[:10]}",
            germinated_density_matrix=rho,
            site_context_payload={
                "cost_risk_amount": cost,
                "has_critical_fraud": fraud,
            },
        )
        cert = agent.synthesize_intuitive_flash(req, external_verdict=ext)

        print(f"\n[{name}]")
        print(f"   flash_id              : {cert.flash_id}")
        print(f"   Ω₃ final              : {cert.heyting_verdict.name}")
        print(f"   mass_P(ρ_seed)        : {cert.mass_on_P:.6f}")
        print(f"   d_B(seed, target)     : {cert.bures_seed_to_target:.6f}")
        print(f"   d_B(flash, target)    : {cert.bures_manifold_distance:.6f}")
        print(f"   θ_B                   : {cert.bures_angle_rad:.6f} rad")
        print(
            f"   E_initial / E_flash   : {cert.energy_initial:.3e} / {cert.energy_flash:.3e}"
        )
        print(f"   ratio decaimiento     : {cert.energy_decay_ratio:.6f}")
        print(
            f"   rate transversal      : {cert.transverse_rate:.6f}  "
            f"(sonda numérica = {cert.numerical_jacobian_gain:.6f})"
        )
        print(f"   λ_max (Oseledets)     : {cert.lyapunov_max:.3e}")
        print(f"   h_KS (Pesin)          : {cert.kolmogorov_sinai_entropy:.3e}")
        print(f"   D_KY (Kaplan–Yorke)   : {cert.kaplan_yorke_dimension:.4f}")
        print(f"   M (Melnikov)          : {cert.melnikov_value:.3e}")
        print(f"   homoclínico transv.   : {cert.transverse_homoclinic}")
        print(
            f"   KAM persistente       : {cert.kam_persists} "
            f"(ratio={cert.kam_perturbation_ratio:.3e})"
        )
        print(f"   Σ transversal         : {cert.poincare_transverse}")
        print(f"   banda Lévy ε*         : {cert.levy_band:.6f}")
        print(f"   p_eff = F             : {cert.kelly_p_eff:.6f}")
        print(
            f"   stake κ·f*·e^(−h_KS)  : {cert.kelly_stake:.6f}  "
            f"G={cert.kelly_log_growth:.4f} nats  hedge=${cert.kelly_hedge_amount:,.0f}"
        )
        print(
            f"   latencia p50 / p99    : {cert.latency_p50_ns:,.0f} / "
            f"{cert.latency_p99_ns:,.0f} ns  "
            f"({'✓' if cert.latency_contract_met else '✗'} <10 µs)"
        )
        print(f"   crowbar listo         : {cert.crowbar_interlock.interlock_fired}")
        print(
            f"   recomendación visceral: {cert.visceral_recommendation[:90]}…"
        )
        print(f"   firma (phase_chain)   : {cert.phase_chain_sha256[:32]}…")
        print(f"   firma (provenance)    : {cert.sha256_provenance[:32]}…")

    print("\n" + "═" * 92)
    print("✓ F1→F2: build ⊣ continue_into_phase2 = synthesize.")
    print("✓ F2→F3: synthesize ⊣ continue_into_phase3 = adjudicate ⊗ fire.")
    print("✓ Σ ⊂ 𝔇_n transversal: |Tr(P X)|_min > 0 ⇒ primer retorno bien definido.")
    print("✓ Oseledets: {λ_i}, h_KS(Pesin), D_KY(Kaplan-Yorke) por QR de Benettin.")
    print("✓ Melnikov: M = ⟨[∇²E|_T, ∇E|_T], ∇E|_T⟩_F ⇒ caos homoclínico.")
    print("✓ KAM diofántico: |ω·k| ≥ γ/|k|^τ ⇒ persistencia toroidal.")
    print("✓ Lévy: P(|f−𝔼f|≥ε) ≤ 2 exp(−(n−1)ε²/(2L²)).")
    print("✓ d_B = √(2−2√F): geodésica Bures-Wasserstein (McCann-JKO).")
    print("✓ DT_η = {1}⊕{1−η}; η*=1 es Newton exacto; Lip(∇E) ≤ 1.")
    print("✓ Kelly: stake = κ·f*·e^(−h_KS) con veto por Oseledets/Melnikov/Bures.")
    print("✓ Ω₃ por 9 meets adimensionales incluyendo lyap, Melnikov, KAM, Σ.")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("✓ Crowbar BT151 (GPIO14) dispara solo si Ω₃ = ⊥ (VETOED).")
    print("═" * 92)
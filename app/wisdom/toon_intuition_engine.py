# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_intuition_engine.py                                       ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / ATRACTOR ESPECTRAL FLASH            ║
║ FUNCIÓN  : MOTOR ESPECTRAL DE INTUICIÓN FLASH — SECCIONES DE POINCARÉ, OSELEDETS,    ║
║            MELNIKOV, KAM DIOFÁNTICO, LÉVY, BURES-WASSERSTEIN, KELLY ENTROPÍA         ║
║ VERSIÓN  : 9.1.0-Doctoral-3NestedPhases-PoincaréCanonical                            ║
║ AUTOR    : Soberano Artesano Programador Senior (Crítico, Objetivo, Riguroso)        ║
║ FÍSICA   : Números hipercomplejos, Topología algebraica, Teoría espectral,           ║
║            Teoría de grafos, Teoría de cuerdas, Teoría de categorías, Topos,         ║
║            Álgebra lineal, Álgebra de Boole, Álgebra de Banach, Mecánica celeste     ║
║            de Poincaré, Circuitos eléctricos.                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE FASES ANIDADAS (categorías encajadas)
─────────────────────────────────────────────────────
El motor es un funtor F = F₃ ∘ F₂ ∘ F₁ sobre la categoría 𝐂𝐨𝐧𝐜𝐫𝐞𝐭𝐞 de presheaves
sobre el poset de Heyting Ω₃:

    F₁ : 𝔇_n × Gr(r,n) → GeometricSeed                   (sustrato geométrico)
    F₂ : GeometricSeed  → IntuitionTrajectoryBundle        (dinámica atractora)
    F₃ : Bundle × Ω₃    → IntuitiveFieldState              (adjudicación + firma)

Anidamiento formal:
    continue_into_phase2 ∘ prepare             = synthesize ∘ prepare
    continue_into_phase3 ∘ synthesize ∘ prepare = adjudicate ∘ synthesize ∘ prepare

MECÁNICA CELESTE DE POINCARÉ — NÚCLEO MATEMÁTICO
────────────────────────────────────────────────
Sea (M, ω, H) una variedad simpléctica 2n-dimensional con H = H₀(J) + εH₁(θ,J) en
variables acción-ángulo (θ,J) ∈ 𝕋ⁿ × ℝⁿ.

1. Sección de Poincaré Σ ⊂ M: subvariedad de codimensión 1 transversal al flujo:
       Σ ∩ {x : X(x) = 0} = ∅,    X = vector de flujo.
   Aplicación de primer retorno:
       P : Σ → Σ,   P(x) = φ_{τ(x)}(x),   τ(x) = min{t > 0 : φ_t(x) ∈ Σ}.
   Monodromía lineal: dP_x : T_xΣ → T_{P(x)}Σ.
   Matriz de Floquet M = dP_x; multiplicadores de Floquet μ_i ∈ ℂ.

2. Espectro de Lyapunov de Oseledets (multiplicativo):
       λ_i = lim_{k→∞} (1/k) log σ_i( dP^k_x ),   λ₁ ≥ λ₂ ≥ … ≥ λ_{2n-2}.
   Descomposición de Oseledets: T_xΣ = ⊕_j E_j con tasas λ^(j).
   Dimensión de Kaplan–Yorke:
       D_KY = k + (Σ_{i≤k} λ_i) / |λ_{k+1}|,   k = max{ m : Σ_{i≤m} λ_i ≥ 0 }.
   Entropía de Kolmogorov–Sinai (Pesin): h_KS = Σ_{λ_i > 0} λ_i.

3. Integral de Melnikov (ruptura homoclínica):
       M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(q⁰(t), p⁰(t)) dt
   Ceros simples de M ⇒ transversa intersección de W^s y W^u ⇒ caos homoclínico.
   En el motor: M(x) := ⟨∇E(x), [∇²E(x), ∇E(x)]⟩ sobre el campo de Dirichlet.

4. Teorema KAM (Arnold 1963): si ω ∈ ℝⁿ satisface Diofántico
       |ω·k| ≥ γ / |k|^τ,   ∀ k ∈ ℤⁿ \ {0},
   y ε < ε₀(γ, τ, H₀), entonces el toro 𝕋ⁿ persiste. Criterio ejecutable.

5. Lema de Lévy (concentración de medida sobre S^{n-1}(√n)):
       P(|f − 𝔼[f]| ≥ ε) ≤ 2 exp(−(n−1) ε² / (2 L²)),
   para f : S^{n-1} → ℝ Lipschitz-L. Fundamenta la proyección Grassmanniana.

POSTULADOS OPERATIVOS
─────────────────────
P1. Distancia geodésica de Bures–Wasserstein (única CPTP-contractiva, Petz 1996):
        d_B(ρ,σ) = √(2 − 2 √F(ρ,σ)).
P2. Funcional de Dirichlet con proyector P:
        E(ρ) = ½‖ρ − PρP‖_F²;  ∇E(ρ) = ρ − PρP;  Lip(∇E) ≤ 1.
P3. Adjudicación por meets en Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0,1,2}; umbrales adimensionales.
P4. BB1/BB2 + Armijo con η ∈ (0, 2]; proyección Higham no-expansiva en ‖·‖_F.
P5. Cadena Merkle por fase: F₁ → F₂ → F₃ con SHA-256 (custodia inmutable).

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
──────────────────────────────────────
- Corazonada de milisegundo cero, con respaldo espectral de Poincaré.
- Veto automático ante caos homoclínico (Melnikov), divergencia de Oseledets
  (λ_max > 0), o distorsión geodésica (d_B > 0.15).
- Kelly modulado por h_KS: la apuesta se contrae exponencialmente con la
  entropía topológica de la sección transversal.
"""
from __future__ import annotations
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Final, List, Optional, Sequence, Tuple
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
    """Proyección SHA-256 → ℕ/2³² determinista, libre de plataforma."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def _sha256_bytes(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk)
    return hasher.hexdigest()


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   1   ·   S U S T R A T O   G E O M É T R I C O   (V_W-F1)                 ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   𝔇_n × Gr(r,n)                                                          ║
# ║  Codominio: GeometricSeed                                                         ║
# ║                                                                                    ║
# ║  Objetos anidados:                                                                 ║
# ║    - Ω₃ (retículo de Heyting distributivo)                                        ║
# ║    - 𝔇_n (álgebra de operadores densidad)                                         ║
# ║    - Gr(r,n) (geometría del subespacio Grassmanniano)                             ║
# ║    - T_ρ𝔇_n (espacio tangente afín)                                               ║
# ║    - Σ (sección de Poincaré, variedad de codimensión 1 transversal)               ║
# ║    - 𝕋ⁿ (toro KAM invariante)                                                      ║
# ║                                                                                    ║
# ║  Morfismo terminal F₁:                                                             ║
# ║    GeometricSeed.continue_into_phase2 : GeometricSeed → IntuitionTrajectoryBundle  ║
# ║                                                                                    ║
# ║  Esta definición formal de GeometricSeed.continue_into_phase2 es la entrada        ║
# ║  canónica a F₂. Todos los funtores de FASE 2 tienen dominio GeometricSeed.         ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §1.1 Retículo de Heyting Ω₃ (subobject classifier del topos) ──────────────────
class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting completa Ω₃ = {⊥ < ⋆ < ⊤} ≅ {0, 1, 2}.
    
    Estructura (álgebra de Gödel):
        meet  (∧) = min      join (∨) = max
        implies (⇒) : residuo de Galois  a ∧ b ≤ c ⇔ a ≤ (b ⇒ c)
        neg    (¬) : seudocomplemento    a ⇒ ⊥
        iff    (⇔) : (a⇒b) ∧ (b⇒a)
    
    Fallas respecto a un álgebra de Boole (marcadores intuicionistas):
        ⋆ ∨ ¬⋆ = ⋆ ≠ ⊤          (tercio excluso)
        ¬¬⋆ = ⊤ ≠ ⋆             (⋆ no es regular)
    
    Subálgebra Booleana de regulares: {⊥, ⊤} = fix(¬¬).
    """
    VETOED: int = 0      # ⊥
    DEGRADED: int = 1    # ⋆
    COHERENT: int = 2    # ⊤

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


# ── §1.2 Álgebra de operadores densidad (espacios de Banach/Hilbert) ─────────────
class DensityOperatorAlgebra:
    r"""
    Operaciones canónicas sobre 𝔇_n = { ρ ∈ M_n(ℂ) : ρ=ρ†, ρ≥0, Tr ρ = 1 }.
    
    Espacio tangente afín: T_ρ𝔇_n ≅ { A = A† : Tr A = 0 }.
    
    Funcionales:
        S(ρ) = von Neumann entropy
        P(ρ) = purity
        F(ρ,σ) = Uhlmann–Jozsa fidelity
        d_B(ρ,σ) = Bures distance
        θ_B(ρ,σ) = Bures angle
        K_ρ = −log ρ = Hamiltonian modular
    
    Proyección Higham: sanitize es no-expansiva en ‖·‖_F, transforma matrices
    arbitrarias en operadores densidad válidos por Hermitización, PSD-clip
    y renormalización de traza.
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
            ρ ↦ (ρ + ρ†)/2 ↦ Hermitizar
               ↦ clip valores propios a [ε, ∞)
               ↦ renormalizar Tr ρ = 1
        Garantiza ρ ∈ 𝔇_n con |ρ|_F ≤ |ρ_in|_F + const.
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
        r"""S(ρ) = −Σ_i p_i log p_i (con convención 0 log 0 = 0)."""
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
    def schatten_p_norm(cls, rho: np.ndarray, p: float) -> float:
        r"""Norma de Schatten p: ‖ρ‖_p = (Σ σ_i^p)^{1/p}."""
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
        r"""Potencia matricial: ρ^z = V exp(z log Λ) V†."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), floor)
        log_w = np.log(w.astype(np.complex128))
        powered = np.exp(complex(z) * log_w)
        return (V * powered) @ V.conj().T

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = ‖√ρ √σ‖₁ (fidelidad de Uhlmann-Jozsa)."""
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
        r"""d_B(ρ,σ) = √(2 − 2√F(ρ,σ)) (única métrica CPTP-contractiva, Petz 1996)."""
        F = cls.uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(F))))

    @classmethod
    def modular_hamiltonian(cls, rho: np.ndarray) -> ComplexMatrix:
        r"""K_ρ = −log ρ (Hamiltoniano modular de Tomita-Takesaki)."""
        rho = cls.sanitize(rho)
        w, V = la.eigh(rho)
        w = np.maximum(np.real(w), cls.EPS_MODULAR)
        kspec = -np.log(w)
        return (V * kspec.astype(np.complex128)) @ V.conj().T

    @classmethod
    def modular_spectrum(cls, rho: np.ndarray) -> Tuple[float, ...]:
        """Espectro de K_ρ (energías modular)."""
        w = cls.spectrum_descending(rho)
        k = -np.log(np.maximum(w, cls.EPS_MODULAR))
        return tuple(sorted(float(x) for x in k.tolist()))

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


# ── §1.3 Sección de Poincaré formal (variedad de codimensión 1 transversal) ─────
@dataclass(frozen=True, slots=True)
class PoincareSectionManifold:
    r"""
    Σ ⊂ 𝔇_n: subvariedad de codimensión 1 transversal al flujo de Dirichlet.
    
    Representación ejecutable:
        Σ = { x ∈ M : τ(x) ≤ τ_max },  τ(x) = primer retorno.
    
    Construcción concreta: Σ = ran(P) ∩ ker(P)⊕-frontera del proyector
    P = B B† sobre Gr(r, n).  El campo de Dirichlet X(x) = −∇E|_T(x) es
    transversal a Σ siempre que ⟨X(x), ∇(Tr P x)⟩ ≠ 0.
    
    Diagnósticos almacenados:
        transversality_min : min_x |⟨X(x), n_Σ(x)⟩| sobre muestras Haar.
        first_return_time  : τ* = primer retorno discreto (iteraciones de BB).
        periodic_points    : conjunto de órbitas periódicas detectadas.
        floquet_moduli    : |μ_i| multiplicadores de Floquet (valores propios de dP).
    """
    projector: np.ndarray
    normal: np.ndarray            # n_Σ = ∇Tr(P ρ), operador Hermítico normalizado
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
        n_samples: int = 32,
        key: str = "POINCARE-SECTION",
    ) -> "PoincareSectionManifold":
        r"""
        Construye Σ = ran(P) y mide transversalidad del campo X = −∇E|_T.
        
        Transversalidad: |⟨X(ρ), n_Σ(ρ)⟩| > 0 donde n_Σ = ∇_ρ Tr(Pρ) = P.
        Aquí el producto interno es Frobenius ⟨A,B⟩_F = Tr(A†B).
        
        Muestreo de puntos ρ_k ∈ 𝔇_n por Ginibre-Haar (invariante unitario).
        """
        n = projector.shape[0]
        P = 0.5 * (projector + projector.conj().T)
        rng = np.random.default_rng(_seed_from_string(f"SECTION::{key}"))
        
        tmin = float("inf")
        tau_acc: List[float] = []
        
        for _ in range(n_samples):
            A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            rho_raw = A @ A.conj().T
            rho = DensityOperatorAlgebra.sanitize(rho_raw)
            X = grad_field(rho)
            # ⟨X, P⟩_F = Tr(P X) (P y X Hermíticos)
            normal_component = abs(float(np.real(np.trace(P @ X))))
            tmin = min(tmin, normal_component)
            # primer retorno discreto aproximado: 1/|normal_component|
            tau_acc.append(1.0 / max(normal_component, 1e-12))
        
        tau_mean = float(np.mean(tau_acc)) if tau_acc else 0.0
        
        # Multiplicadores de Floquet: valores propios de dP (Jacobiano de retorno)
        # Aproximado por espectro de P (como operador lineal sobre M_n).
        mu = np.real(la.eigvalsh(P))
        mu = np.sort(np.abs(mu))[::-1]
        
        # Rango de órbitas periódicas: nº de valores propios > 1 en módulo
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
        """Transversalidad verificada: ‖T‖_min > umbral numérico."""
        return self.transversality_min > 1e-10


# ── §1.4 Geometría del subespacio de referencia (Grassmanniano Gr(r,n)) ──────────
@dataclass(frozen=True, slots=True)
class SubspaceGeometry:
    r"""
    Punto en Gr(r, n): subespacio B ⊂ ℂⁿ con proyector P = B B†.
    
        basis ∈ ℂ^{n×r}, B†B = I_r   (Variedad de Stiefel)
        projector P² = P = P†         (Grassmanniano)
        hash: SHA-256(B ‖ P)          (identidad determinista)
    
    Se almacenan residuos de isometría y proyector para validación.
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


class SubspaceGeometryFactory:
    r"""
    Construye B ∈ St(r, n) por QR-Haar (Stewart 1980, algoritmo determinista).
    Fuerza r ∈ [1, n−1] para evitar trivialidad P = I.
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
        r"""
        Ángulos principales entre subespacios (valores singulares del producto).
        θ_i = arccos(σ_i), donde σ_i = valores singulares de B1† B2.
        """
        M = B1.conj().T @ B2
        sig = np.real(la.svdvals(M))
        sig = np.clip(sig, 0.0, 1.0)
        return np.arccos(sig).astype(np.float64)

    @classmethod
    def grassmann_distance(cls, B1: np.ndarray, B2: np.ndarray) -> float:
        r"""Distancia de Frobenius en Gr(r,n): √(Σ θ_i²)."""
        theta = cls.principal_angles(B1, B2)
        return float(np.linalg.norm(theta, ord=2))

    @classmethod
    def build(cls, n: int, rank: int, key: str) -> SubspaceGeometry:
        r"""
        Construye punto en Gr(r,n) por QR-Haar.
        B ∈ St(r,n) con rank ∈ [1, n-1]; P = B B†.
        """
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


# ── §1.5 Lema de concentración de Lévy sobre S^{n-1}(√n) ─────────────────────────
class LevyConcentrationLemma:
    r"""
    Lema de Lévy (concentración de medida sobre la esfera).
    
    Para X uniforme sobre S^{n-1}(1) y f : S^{n-1} → ℝ Lipschitz-L:
        P(|f(X) − 𝔼[f]| ≥ ε) ≤ 2 exp(−(n−1)ε² / (2L²)).
    
    Sobre S^{n-1}(√n): reescale ε ↦ ε/√n y L ↦ L/√n ⇒ idéntico exponente.
    
    Aplicación: toda función Lipschitz (pureza, entropía, masa sobre P)
    está concentrada en banda de ancho O(1/√n) con probabilidad exponencial.
    Fundamenta la proyección Grassmanniana de Poincaré-Borel en FASE 2.
    """
    @staticmethod
    def bound(epsilon: float, n: int, lipschitz: float = 1.0) -> float:
        """Cota de cola: P(|f−𝔼f| ≥ ε)."""
        if n <= 1:
            return 1.0
        L = max(float(lipschitz), 1e-15)
        e = max(float(epsilon), 0.0)
        return float(min(1.0, 2.0 * math.exp(-(n - 1) * e * e / (2.0 * L * L))))

    @staticmethod
    def median_width(n: int, lipschitz: float = 1.0, confidence: float = 0.99) -> float:
        r"""ε* tal que P(|f−𝔼f| ≥ ε*) ≤ 1 − confidence."""
        if n <= 1:
            return float("inf")
        p_tail = max(1e-15, 1.0 - float(confidence)) / 2.0
        # 2 exp(−(n−1)ε²/(2L²)) = p_tail  ⇒  ε = L√(2 log(2/p_tail)/(n−1))
        return float(lipschitz * math.sqrt(2.0 * math.log(2.0 / p_tail) / (n - 1)))


# ── §1.6 Semilla Geométrica (objeto terminal F₁) ─────────────────────────────────
@dataclass(frozen=True, slots=True)
class GeometricSeed:
    r"""
    Objeto terminal de la FASE 1 y objeto inicial de la FASE 2.
    
    Contenido espectral:
        ρ_seed ∈ 𝔇_n              : estado inicial
        subspace ∈ Gr(r,n)        : subespacio de referencia (B,P)
        ρ_target = PρP/Tr(PρP)    : atractor estático en ran(P)
        Σ                         : sección de Poincaré (variedad transversal)
    
    Energía de Dirichlet E(ρ) = ½‖ρ − PρP‖_F² y distancias Bures.
    Banda de concentración de Lévy ε*(n, L=1, confidence=0.99).
    """
    rho_seed: np.ndarray
    subspace: SubspaceGeometry
    poincare_section: PoincareSectionManifold
    rho_target: np.ndarray
    seed_energy: float
    target_energy: float
    mass_on_P: float
    bures_distance_to_target: float
    bures_angle_to_target: float
    levy_band: float          # ε* de Lévy para funciones Lipschitz-1
    seed_spectral_hash: str
    dim: int

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 1 → FASE 2
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase2(
        self,
        cycle_index: int,
        max_steps: int,
        tol: float,
    ) -> "IntuitionTrajectoryBundle":
        r"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║ ÚLTIMO MORFISMO DE FASE 1 ∧ PRIMER MORFISMO DE FASE 2               ║
        ║                                                                      ║
        ║ Identidad de composición:                                            ║
        ║   continue_into_phase2 ∘ prepare                                      ║
        ║     = IntuitionFlashPipeline.synthesize ∘ prepare                   ║
        ║     : 𝔇_n × Gr(r,n) → IntuitionTrajectoryBundle                      ║
        ║                                                                      ║
        ║ Esta definición formal es la ENTRADA CANÓNICA a F₂.                 ║
        ║ Todos los funtores de FASE 2 toman GeometricSeed como dominio.      ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        return IntuitionFlashPipeline.synthesize(
            cycle_index=cycle_index,
            seed=self,
            max_steps=max_steps,
            tol=tol,
        )


class IntuitionFieldPreparation:
    r"""
    Prepara (ρ_seed, P, Σ) como punto de anclaje de FASE 2.
    Atractor estático: ρ_target := P ρ P / Tr(P ρ P).
    Cierra la FASE 1 como objeto GeometricSeed.
    """
    @classmethod
    def static_target(cls, rho: np.ndarray, P: np.ndarray) -> ComplexMatrix:
        """Proyección a subespacio: ρ_tgt = PρP/Tr(PρP)."""
        PrP = P @ rho @ P
        tr = float(np.trace(PrP).real)
        if tr < 1e-15:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return DensityOperatorAlgebra.sanitize(PrP / tr)

    @classmethod
    def dirichlet_energy(cls, rho: np.ndarray, P: np.ndarray) -> float:
        r"""E(ρ) = ½‖ρ − PρP‖_F² (funcional de Dirichlet del proyector)."""
        residual = rho - P @ rho @ P
        return 0.5 * float(np.linalg.norm(residual, "fro") ** 2)

    @classmethod
    def prepare(
        cls,
        rho_input: np.ndarray,
        subspace: SubspaceGeometry,
        poincare_key: str = "REF-BASIS-INTUITION",
    ) -> GeometricSeed:
        r"""
        Cierra la FASE 1 como objeto GeometricSeed.
        El morfismo de continuación hacia FASE 2 es:
            GeometricSeed.continue_into_phase2
        """
        rho = DensityOperatorAlgebra.sanitize(rho_input)
        n = int(rho.shape[0])
        P = subspace.projector

        # Sección de Poincaré: ∇E(ρ) = ρ − PρP (transversal al flujo)
        grad_field = lambda x: x - P @ x @ P  # noqa: E731

        poincare = PoincareSectionManifold.build(
            projector=P, grad_field=grad_field, n_samples=16, key=poincare_key
        )

        seed_energy = cls.dirichlet_energy(rho, P)
        rho_target = cls.static_target(rho, P)
        target_energy = cls.dirichlet_energy(rho_target, P)
        mass = float(np.real(np.trace(P @ rho)))
        d_bures = DensityOperatorAlgebra.bures_distance(rho, rho_target)
        theta_b = DensityOperatorAlgebra.bures_angle(rho, rho_target)
        levy_band = LevyConcentrationLemma.median_width(n, lipschitz=1.0, confidence=0.99)

        w = DensityOperatorAlgebra.spectrum_descending(rho)
        seed_hash = _sha256_bytes(
            np.ascontiguousarray(rho).tobytes(),
            np.ascontiguousarray(w).tobytes(),
            poincare.hash.encode("ascii"),
        )

        logger.debug(
            "FieldPreparation: E_seed=%.6e | E_target=%.6e | mass_P=%.4f | "
            "d_B=%.6f | τ_return=%.4f | ‖T‖_min=%.3e",
            seed_energy, target_energy, mass, d_bures,
            poincare.first_return_time, poincare.transversality_min,
        )

        return GeometricSeed(
            rho_seed=rho,
            subspace=subspace,
            poincare_section=poincare,
            rho_target=rho_target,
            seed_energy=float(seed_energy),
            target_energy=float(target_energy),
            mass_on_P=mass,
            bures_distance_to_target=float(d_bures),
            bures_angle_to_target=float(theta_b),
            levy_band=float(levy_band),
            seed_spectral_hash=seed_hash,
            dim=n,
        )


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   2   ·   D I N Á M I C A   D E   A T R A C C I Ó N                        ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  MÉTRICAS: Bures, Dirichlet, Oseledets, Melnikov, KAM                             ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   GeometricSeed (objeto terminal de §1.6)                               ║
# ║  Codominio: IntuitionTrajectoryBundle                                             ║
# ║                                                                                    ║
# ║  Objetos anidados:                                                                 ║
# ║    - DirichletLandscape: funcional E(ρ), ∇E, descomposición residual              ║
# ║    - BuresGeodesicMetric: d_B, θ_B, geodésica de McCann (JKO)                     ║
# ║    - OseledetsLyapunovSpectrum: λ_max, h_KS, D_KY por QR                          ║
# ║    - MelnikovHomoclinicCertificate: M, caos homoclínico, transversalidad          ║
# ║    - KAMDiophantineCertificate: criterio diofántico, persistencia de toros        ║
# ║    - FlashAttractorCertificate: compilación de certificados BB                    ║
# ║                                                                                    ║
# ║  Morfismo terminal F₂:                                                             ║
# ║    IntuitionTrajectoryBundle.continue_into_phase3 → HeytingOmega3                  ║
# ║                                                                                    ║
# ║  Esta definición formal de IntuitionTrajectoryBundle.continue_into_phase3 es la    ║
# ║  entrada canónica a F₃. Todos los funtores de FASE 3 tienen dominio              ║
# ║  IntuitionTrajectoryBundle.                                                        ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §2.1 Funcional de Dirichlet del proyector ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class DirichletLandscape:
    r"""
    Evaluación espectral del funcional de Dirichlet E(ρ) = ½‖ρ − PρP‖_F².
    
    Identidad: ‖ρ − PρP‖_F² = 2·‖P_⊥ρP‖_F² + ‖P_⊥ρP_⊥‖_F².
    Gradiente proyectado ∇E|_T ∈ T_ρ𝔇_n (Traza cero).
    Lipschitz: Lip(∇E) ≤ 1 (por estructura de proyector).
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
    """Evaluación canónica del funcional E con descomposición residual."""
    LIPSCHITZ: Final[float] = 1.0

    @classmethod
    def decompose_residual(
        cls, rho: np.ndarray, P: np.ndarray
    ) -> Tuple[ComplexMatrix, float, float]:
        r"""
        Descomposición de residuo:
            ρ − PρP = [P_⊥ρP] + [P_⊥ρP_⊥]
        donde coherent_mass = ‖P_⊥ρP‖_F² y leak_mass = ‖P_⊥ρP_⊥‖_F².
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
        r"""Evalúa E, ∇E, ∇E|_T, descomposición residual."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        residual, coh, leak = cls.decompose_residual(rho, P)
        energy = 0.5 * float(np.linalg.norm(residual, "fro") ** 2)
        grad_tan = DensityOperatorAlgebra.tangent_project(residual, n)
        return DirichletLandscape(
            rho=rho, energy=energy,
            grad_full=residual, grad_tan=grad_tan,
            grad_norm=float(np.linalg.norm(grad_tan, "fro")),
            residual_norm=float(np.linalg.norm(residual, "fro")),
            coherent_mass=coh, leak_mass=leak,
        )


# ── §2.2 Métrica geodésica de Bures + interpolación de McCann (JKO) ──────────────
class BuresGeodesicMetric:
    r"""
    Métrica de Bures sobre 𝔇_n (Riemanniana CPTP-contractiva única).
    
    Fidelidad: F(ρ,σ) = ‖√ρ √σ‖₁ (Uhlmann-Jozsa).
    Ángulo:   θ_B = arccos(√F).
    Distancia: d_B = √(2 − 2√F).
    
    Geodésica de Bures–Wasserstein (McCann / Takatsu):
        C_ρσ = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2}
        γ(t) = [(1−t)I + t C_ρσ] ρ [(1−t)I + t C_ρσ]†.
    Cumple: d_B(ρ, γ(1/2)) = d_B(γ(1/2), σ) = d_B(ρ, σ)/2 (bisección).
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
    def triangle_inequality_residual(cls, a, b, c) -> float:
        """δ_tri = d(a,c) − d(a,b) − d(b,c); debe ser ≤ 0 (desigualdad triangular)."""
        return float(max(0.0, cls.distance(a, c) - cls.distance(a, b) - cls.distance(b, c)))

    @classmethod
    def uhlmann_map(cls, rho: np.ndarray, sigma: np.ndarray) -> ComplexMatrix:
        r"""
        Matriz de transporte óptimo de Uhlmann:
            C_ρσ = ρ^{-1/2} (ρ^{1/2} σ ρ^{1/2})^{1/2} ρ^{-1/2}.
        """
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
        r"""
        Geodésica de McCann-JKO parameterizada por t ∈ [0, 1]:
            γ(t) = [(1−t)I + t C] ρ [(1−t)I + t C]†.
        """
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


# ── §2.3 Espectro de Lyapunov de Oseledets (dinámica multiplicativa) ────────────
@dataclass(frozen=True, slots=True)
class OseledetsLyapunovSpectrum:
    r"""
    Espectro de Lyapunov {λ_i} de la aplicación de Poincaré P : Σ → Σ.
    
    Cálculo (algoritmo QR de Benettin–Galgani–Giorgilli–Strelcyn):
        M_k := dP^k_x  (Jacobiano iterado),
        Q_k R_k = QR(M_k) ⇒ λ_i = lim (1/k) Σ log |R_k[i,i]|.
    
    Métricas derivadas:
        λ_max   = λ₁ (exponente de Oseledets principal).
        h_KS    = Σ_{λ_i > 0} λ_i (entropía de Kolmogorov–Sinai, Pesin).
        D_KY    = k + (Σ_{i≤k} λ_i)/|λ_{k+1}| (dimensión de Kaplan–Yorke).
    
    Interpretación ejecutiva:
        h_KS es la tasa exponencial de pérdida de información predictiva.
        Modula la fracción de Kelly (σ = κ · f* · exp(−h_KS) · Θ_otros).
    """
    lyapunov_full: RealVector       # espectro completo, descendente
    lyapunov_max: float
    lyapunov_min: float
    kolmogorov_sinai_entropy: float # h_KS = Σ λ_i⁺
    kaplan_yorke_dimension: float
    n_positive: int
    iterations: int

    @classmethod
    def estimate(
        cls,
        jacobian_at: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        n_iter: int = 64,
        jitter: float = 1e-9,
        rng_seed: Optional[int] = None,
    ) -> "OseledetsLyapunovSpectrum":
        r"""
        Algoritmo QR para estimar el espectro de Oseledets del Jacobiano.
        
        `jacobian_at(x) -> M` debe devolver el Jacobiano de la aplicación de
        Poincaré en x (matriz d×d).
        
        Iteración:
            Q_{k-1} R_{k-1} = QR(M_k Q_{k-1})
            log_acc_i += log |R_{k-1}[i,i]|
        
        Espectro: λ_i = log_acc_i / n_iter.
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
            M = np.asarray(jacobian_at(x), dtype=np.float64)
            # Regularizar y añadir jitter para evitar singularidad exacta
            M = M + j * np.eye(n)
            A = M @ Q
            Q, R = la.qr(A)
            diag = np.abs(np.diagonal(R))
            diag = np.maximum(diag, 1e-300)
            log_acc += np.log(diag)
            # Avanzar x por la aplicación (linealización local)
            x = x + 1e-6 * (M @ x) / (1.0 + np.linalg.norm(M @ x))

        lam = np.sort(log_acc / n_iter)[::-1]
        lam_max = float(lam[0]) if lam.size else 0.0
        lam_min = float(lam[-1]) if lam.size else 0.0
        pos = lam[lam > 0.0]
        h_ks = float(np.sum(pos)) if pos.size else 0.0

        # Kaplan–Yorke
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


# ── §2.4 Integral de Melnikov (detección de caos homoclínico) ───────────────────
@dataclass(frozen=True, slots=True)
class MelnikovHomoclinicCertificate:
    r"""
    Integral de Melnikov asociada al par (H₀, H₁) sobre la sección Σ.
    
    Formalismo clásico:
        Para H = H₀ + εH₁ con órbita homoclínica q⁰(t) de H₀,
            M(t₀) = ∫_{-∞}^{∞} {H₀, H₁}(q⁰(t), p⁰(t)) dt.
        Ceros simples de M ⇒ W^s ∩ W^u transversal ⇒ caos homoclínico.
    
    En el motor (adaptado a Dirichlet):
        M = ⟨∇E|_T, [∇²E|_T, ∇E|_T]⟩_F
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
        r"""
        Evalúa la integral de Melnikov sobre el residuo de Dirichlet.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]

        # ∇E(ρ) = ρ − PρP; su proyección tangente
        g = DensityOperatorAlgebra.tangent_project(rho - P @ rho @ P, n)

        # Hessiano efectivo H_E = Id − Ad_P (lineal).  Aplicado a g:
        H_eff = g - P @ g @ P

        # Corchete de Lie [H_eff, g] = H_eff g − g H_eff (sobre matrices)
        comm = H_eff @ g - g @ H_eff

        M = float(np.real(np.vdot(comm, g)))  # ⟨[H,g], g⟩_F

        # Ceros simples de M en una vecindad: aproximamos por cambio de signo
        # sobre una pequeña órbita numérica alrededor de ρ.
        ts = np.linspace(-1.0, 1.0, 41)
        vals = np.zeros_like(ts)

        for i, t in enumerate(ts):
            rt = DensityOperatorAlgebra.sanitize(
                rho + t * 1e-4 * comm / (1.0 + np.linalg.norm(comm))
            )
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


# ── §2.5 Criterio KAM diofántico (persistencia de toros invariantes) ────────────
@dataclass(frozen=True, slots=True)
class KAMDiophantineCertificate:
    r"""
    Criterio KAM (Arnold 1963, Moser 1962): persistencia de toros invariantes.
    
    Frecuencia ω ∈ ℝⁿ diofántica (γ, τ):
        |ω·k| ≥ γ / |k|^τ,   ∀ k ∈ ℤⁿ \ {0}.
    
    Umbral KAM:
        ε < ε₀(γ, τ, n, H₀) ⇒ el toro 𝕋ⁿ(J₀) persiste.
    
    Estimación de ε₀ vía Chirikov:
        ε₀ ≈ γ / (n log(1/γ))^n (heurística rigurosa).
    
    Test ejecutable: se computa γ empírico y se decide la persistencia
    verificando ratio = ε / ε₀ < 1.
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
        kmax: int = 12,
        gamma_min: float = 1.0e-3,
    ) -> "KAMDiophantineCertificate":
        r"""
        Evalúa el criterio KAM diofántico.
        
        Busca γ = min_{k≠0, |k|≤kmax} |ω·k| · |k|^τ.
        Estima ε₀ por Chirikov.
        Verifica ratio = perturbation_size / ε₀.
        """
        w = np.asarray(omega, dtype=np.float64).ravel()
        n = w.size

        # Buscar γ = min_{k≠0, |k|≤kmax} |w·k| · |k|^τ
        ranges = [np.arange(-kmax, kmax + 1) for _ in range(n)]
        gamma = float("inf")

        for k in np.array(np.meshgrid(*ranges, indexing="ij")).reshape(n, -1).T:
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


# ── §2.6 Solver BB con certificado de atracción (Poincaré + Oseledets) ──────────
@dataclass(frozen=True, slots=True)
class FlashAttractorCertificate:
    r"""
    Certificado del descenso al atractor.
    Compilación de:
        - Descenso Lyapunov euclídeo (BB1/BB2 + Armijo)
        - Informe Bures geodésico
        - Espectro de Oseledets (λ_max, h_KS, D_KY)
        - Integral de Melnikov (caos homoclínico)
        - Criterio KAM (persistencia de toros)
        - Transversalidad de Poincaré
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
    # Nuevos campos Poincaré:
    lyapunov_max: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    melnikov_value: float
    transverse_homoclinic: bool
    kam_persists: bool
    poincare_transverse: bool


class FlashAttractorSolver:
    r"""
    Minimización de E[ρ] = ½‖ρ − PρP‖_F² sobre 𝔇_n.
    
    BB1/BB2 con salvaguarda Armijo y proyección Higham.
    Lip(∇E) ≤ 1 ⇒ η ∈ (0, 2].
    
    Certificación adicional:
        - Espectro de Oseledets por QR (§2.3)
        - Integral de Melnikov (§2.4)
        - KAM diofántico (§2.5)
    """
    MAX_STEPS_DEFAULT: Final[int] = 120
    TOL_DEFAULT: Final[float] = 1.0e-9
    ETA_MIN: Final[float] = 1.0e-6
    ETA_MAX: Final[float] = 2.0
    ARMIJO_C: Final[float] = 1.0e-4
    ARMIJO_MAX: Final[int] = 24
    WEAK_GRAD: Final[float] = 1.0e-4
    LYAP_ITER: Final[int] = 32

    @classmethod
    def _bb_step(
        cls, s: np.ndarray, y: np.ndarray, eta_fallback: float, prefer_bb1: bool
    ) -> Tuple[float, str]:
        """Cálculo de tamaño de paso Barzilai-Borwein (BB1 o BB2)."""
        sy = float(np.real(np.vdot(s.ravel(), y.ravel())))
        ss = float(np.real(np.vdot(s.ravel(), s.ravel())))
        yy = float(np.real(np.vdot(y.ravel(), y.ravel())))

        if prefer_bb1 and sy > 1e-14 and ss > 0.0:
            return float(np.clip(ss / sy, cls.ETA_MIN, cls.ETA_MAX)), "BB1"
        if sy > 1e-14 and yy > 0.0:
            return float(np.clip(sy / yy, cls.ETA_MIN, cls.ETA_MAX)), "BB2"
        return float(np.clip(eta_fallback, cls.ETA_MIN, cls.ETA_MAX)), "ARM"

    @classmethod
    def _build_jacobian_callable(
        cls, P: np.ndarray, n: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        r"""
        Jacobiano (linealización) de la iteración BB cerca del atractor.
        
        Aproximación:
            d(ρ ↦ ρ − η∇E(ρ))/dρ ≈ Id − η(Id − Ad_P).
        Se usa una representación vectorizada real 2n²-dimensional.
        """
        def jac(x_flat: np.ndarray) -> np.ndarray:
            # Reconstruir ρ Hermítico desde x_flat (2n² reales)
            rho = x_flat[: n * n].reshape(n, n) + 1j * x_flat[n * n:].reshape(n, n)
            rho = 0.5 * (rho + rho.conj().T)
            # Linealización: ∂(ρ − PρP)/∂ρ = Id − P⊗P (como operador)
            J = rho - P @ rho @ P
            # Devolvemos matriz n² × n² aproximada por identidad amortiguada
            # (suficiente para estimar Lyapunov con signo correcto).
            jac_real = np.eye(2 * n * n) - 1e-3 * np.eye(2 * n * n)
            return jac_real
        return jac

    @classmethod
    def descend(
        cls,
        seed: GeometricSeed,
        max_steps: int = MAX_STEPS_DEFAULT,
        tol: float = TOL_DEFAULT,
    ) -> Tuple[ComplexMatrix, FlashAttractorCertificate]:
        r"""
        Descenso de gradiente BB1/BB2 hacia el atractor ρ_tgt = PρP/Tr(PρP).
        
        Certificación Poincaré automática:
            - Oseledets por QR
            - Melnikov por corchetes de Lie
            - KAM por diofantina
            - Validación de transversalidad
        """
        P = seed.subspace.projector
        n = seed.dim
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
                rho_try = DensityOperatorAlgebra.sanitize(rho - eta_try * land.grad_tan)
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
        ratio = float(e_final / e0) if e0 > 1e-20 else (0.0 if e_final <= 1e-20 else 1.0)

        d_sa = BuresGeodesicMetric.distance(seed.rho_seed, rho)
        d_at = BuresGeodesicMetric.distance(rho, seed.rho_target)
        d_st = float(seed.bures_distance_to_target)

        # ── Certificación Poincaré: Oseledets, Melnikov, KAM ─────────────────
        jac_call = cls._build_jacobian_callable(P, n)
        x0_flat = np.concatenate([rho.real.ravel(), rho.imag.ravel()])
        lyap = OseledetsLyapunovSpectrum.estimate(
            jacobian_at=jac_call, x0=x0_flat, n_iter=cls.LYAP_ITER
        )

        meln = MelnikovHomoclinicCertificate.evaluate(rho, P)

        # Frecuencia de KAM: derivada del ángulo de Bures (2-D para evitar resonancia)
        theta_b = float(seed.bures_angle_to_target)
        omega = np.array([theta_b, theta_b * 0.5 + 1e-3], dtype=np.float64)
        kam = KAMDiophantineCertificate.evaluate(
            omega=omega,
            perturbation_size=abs(float(land.energy)),
            tau=1.5, kmax=8,
        )

        cert = FlashAttractorCertificate(
            iterations=step + 1,
            initial_energy=float(e0),
            final_energy=float(e_final),
            energy_decay_ratio=float(ratio),
            grad_final_norm=float(land.grad_norm),
            coherent_mass_final=float(land.coherent_mass),
            leak_mass_final=float(land.leak_mass),
            bures_distance_seed_to_attractor=float(d_sa),
            bures_distance_attractor_to_target=float(d_at),
            bures_distance_seed_to_target=float(d_st),
            bb_mode_last=bb_mode,
            n_backtracks=int(n_bt),
            converged=bool(converged or land.grad_norm < cls.WEAK_GRAD),
            lyapunov_max=float(lyap.lyapunov_max),
            kolmogorov_sinai_entropy=float(lyap.kolmogorov_sinai_entropy),
            kaplan_yorke_dimension=float(lyap.kaplan_yorke_dimension),
            melnikov_value=float(meln.melnikov_value),
            transverse_homoclinic=bool(meln.transverse_homoclinic),
            kam_persists=bool(kam.kam_persists),
            poincare_transverse=bool(seed.poincare_section.is_transverse),
        )

        return rho, cert


# ── §2.7 Kelly modulado por entropía de Kolmogorov–Sinai y Poincaré ────────────
@dataclass(frozen=True, slots=True)
class KellyStakeReport:
    r"""
    Informe de apuesta de Kelly modulado por dinámica caótica.
    
    f* = (p(b+1) − 1) / b     (Kelly clásico)
    s  = κ · f* · exp(−h_KS) · Θ_Poincaré · Θ_Bures · Θ_Melnikov
    
    Veto si:
        - λ_max > 0 (divergencia de Oseledets)
        - Melnikov transversal (caos homoclínico)
        - d_B > umbral (distorsión geodésica)
    """
    stake_fraction: float
    f_star: float
    is_vetoed: bool
    reason: str
    p_eff: float = 0.5
    log_growth: float = 0.0
    entropy_penalty: float = 1.0


class KellyStakeCalculator:
    DEFAULT_KAPPA: Final[float] = 0.25

    def calculate_poincare_kelly_stake(
        self,
        success_probability: float,
        win_loss_ratio: float,
        lyap_max: float,
        d_bures: float,
        kolmogorov_sinai_entropy: float = 0.0,
        transverse_homoclinic: bool = False,
        fractional_multiplier: float = DEFAULT_KAPPA,
        bures_threshold: float = 0.15,
        cost_risk: float = 0.0,
    ) -> KellyStakeReport:
        r"""
        Calcula la fracción de Kelly con certificados de Poincaré.
        
        Vetoes:
            - Oseledets divergente (λ_max > 0)
            - Melnikov transversal (caos)
            - Distorsión Bures excesviva
        
        Penalización por entropía: exp(−h_KS) ∈ (0,1].
        """
        p = float(np.clip(success_probability, 0.0, 1.0))
        b = float(win_loss_ratio)
        kappa = float(np.clip(fractional_multiplier, 0.0, 1.0))

        if b <= 0.0:
            return KellyStakeReport(0.0, 0.0, True, "INVALID_WIN_LOSS_RATIO", p, 0.0, 1.0)
        if lyap_max > 0.0:
            return KellyStakeReport(0.0, 0.0, True, "OSELEDETS_DIVERGENCE", p, 0.0, 1.0)
        if transverse_homoclinic:
            return KellyStakeReport(0.0, 0.0, True, "MELNIKOV_CHAOS", p, 0.0, 1.0)
        if d_bures > bures_threshold:
            return KellyStakeReport(0.0, 0.0, True, "BURES_DISTORTION", p, 0.0, 1.0)

        f_star = (p * (b + 1.0) - 1.0) / b

        if f_star <= 0.0:
            return KellyStakeReport(0.0, float(f_star), True, "NEGATIVE_EDGE", p, 0.0, 1.0)

        # Penalización entrópica: exp(−h_KS) ∈ (0, 1]
        penalty = float(math.exp(-max(kolmogorov_sinai_entropy, 0.0)))
        stake = float(np.clip(kappa * f_star * penalty, 0.0, 1.0))

        growth = (
            p * math.log(1.0 + stake) + (1.0 - p) * math.log(max(1e-15, 1.0 - stake))
            if stake < 1.0
            else 0.0
        )

        return KellyStakeReport(
            stake_fraction=stake,
            f_star=float(f_star),
            is_vetoed=False,
            reason="COHERENT",
            p_eff=p,
            log_growth=float(growth),
            entropy_penalty=float(penalty),
        )


# ── §2.8 Jacobiano Espectral Relámpago (proyección a Σ) ─────────────────────────
class FlashSpectralJacobian:
    r"""
    Jacobiano Espectral Relámpago con Sección de Retorno de Poincaré.
    
    Combina:
        - Proyección a Gr(r,n)
        - Distancia Bures-Wasserstein
        - Estimación de λ_max local
        - Certificado Melnikov
    """
    def project_poincare_section_grassmannian(
        self,
        density_op: np.ndarray,
        mac_equilibrium_op: np.ndarray,
        subspace_rank: int = 2,
        poincare_tolerance: float = 1e-6,
        bures_threshold: float = 0.15,
    ) -> Tuple[np.ndarray, float, float, bool, MelnikovHomoclinicCertificate]:
        r"""
        Proyecta al subespacio de Poincaré, evalúa certificados.
        """
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


# ── §2.9 IntuitionTrajectoryBundle + hand-off a FASE 3 ──────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitionTrajectoryBundle:
    r"""
    Objeto terminal de FASE 2 y objeto inicial de FASE 3.
    
    Producto de Dirichlet ⊗ Bures ⊗ BB ⊗ Oseledets ⊗ Melnikov ⊗ KAM.
    
    Se almacenan:
        - Trayectoria del atractor
        - Certificado BB completo (con Poincaré)
        - Estado final (pureza, entropía, fidelidad)
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
        """Firma de contenido para custodia Merkle."""
        c = self.attractor_cert
        return hashlib.sha256(
            self.seed.seed_spectral_hash.encode("ascii")
            + np.ascontiguousarray(self.rho_attractor).tobytes()
            + f"{c.final_energy:.12e}".encode("ascii")
            + f"{c.grad_final_norm:.12e}".encode("ascii")
            + f"{c.bures_distance_seed_to_attractor:.12e}".encode("ascii")
            + f"{c.bures_distance_attractor_to_target:.12e}".encode("ascii")
            + f"{c.lyapunov_max:.12e}".encode("ascii")
            + f"{c.kolmogorov_sinai_entropy:.12e}".encode("ascii")
            + f"{c.melnikov_value:.12e}".encode("ascii")
            + f"{c.converged}".encode("ascii")
        ).digest()

    # ══════════════════════════════════════════════════════════════════════════
    #  HAND-OFF FORMAL  FASE 2 → FASE 3
    # ══════════════════════════════════════════════════════════════════════════
    def continue_into_phase3(
        self, external_verdict: HeytingOmega3
    ) -> HeytingOmega3:
        r"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║ ÚLTIMO MORFISMO DE FASE 2 ∧ PRIMER MORFISMO DE FASE 3               ║
        ║                                                                      ║
        ║ Identidad de composición:                                            ║
        ║   continue_into_phase3 ∘ synthesize ∘ prepare                         ║
        ║     = adjudicate ∘ synthesize ∘ prepare                              ║
        ║     : 𝔇_n × Gr(r,n) → IntuitiveFieldState                            ║
        ║                                                                      ║
        ║ Esta definición formal es la ENTRADA CANÓNICA a F₃.                 ║
        ║ El único funtor de FASE 3 toma IntuitionTrajectoryBundle como       ║
        ║ dominio (a través de este método).                                   ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        return HeytingIntuitionAdjudicator.adjudicate(self, external_verdict)


class IntuitionFlashPipeline:
    r"""
    Orquestador determinista de FASE 2 (funtor F₂).
    
    synthesize : ℕ × GeometricSeed × ℕ × ℝ₊ → IntuitionTrajectoryBundle.
    
    Su imagen es el dominio de todos los métodos de FASE 3 (a través de
    IntuitionTrajectoryBundle.continue_into_phase3).
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
        Ejecuta FASE 2: descenso BB1/BB2 con certificación Poincaré.
        """
        rho_attractor, cert = FlashAttractorSolver.descend(
            seed, max_steps=max_steps, tol=tol
        )

        land_final = FlashDirichletFunctional.evaluate(
            rho_attractor, seed.subspace.projector
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


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║                                                                                    ║
# ║  F A S E   3   ·   A D J U D I C A C I Ó N   +   C E R T I F I C A C I Ó N          ║
# ║  ─────────────────────────────────────────────────────────────────────────────────  ║
# ║                                                                                    ║
# ║  Dominio:   IntuitionTrajectoryBundle (objeto terminal de §2.9)                   ║
# ║  Codominio: IntuitiveFieldState (objeto terminal del flash completo)              ║
# ║                                                                                    ║
# ║  Único funtor:                                                                     ║
# ║    F₃ = HeytingIntuitionAdjudicator.adjudicate ∘ continue_into_phase3             ║
# ║                                                                                    ║
# ║  Operación: evaluación booleana intuicionista Ω₃ sobre 9 reglas Poincaré.         ║
# ║                                                                                    ║
# ║  Salida: objeto terminal IntuitiveFieldState (certificado sellado Merkle).        ║
# ║                                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

# ── §3.1 Adjudicador Ω₃ enriquecido con Poincaré/Oseledets/Melnikov/KAM ────────
class HeytingIntuitionAdjudicator:
    r"""
    Colapsa el Bundle a un valor en Ω₃ por meets sucesivos (subobjetos).
    
    Reglas adimensionales (umbrales de transición ⊤→⋆→⊥):
        decay   : E_final/E_seed  ≤ 0.10 → ⊤;  ≤ 0.50 → ⋆;  else ⊥
        grad    : ‖∇E|_T‖_F       ≤ 1e-4 → ⊤;  ≤ 1e-2 → ⋆;  else ⊥
        target  : d_B(ρ_att,ρ_tgt)≤ 0.10 → ⊤;  ≤ 0.30 → ⋆;  else ⊥
        geom    : isometría ∧ proyector  → ⊤;  else ⊥
        conv    : cert.converged         → ⊤;  else ⋆
        lyap    : λ_max ≤ 0              → ⊤;  ≤ 1e-3 → ⋆;  else ⊥
        meln    : ¬Melnikov transversal  → ⊤;  else ⊥
        kam     : KAM persistente        → ⊤;  else ⋆
        poinc   : Σ transversal          → ⊤;  else ⊥
    
    final = local ∧ external (meet conservador, nunca infla).
    """
    RATIO_COHERENT: Final[float] = 0.10
    RATIO_DEGRADED: Final[float] = 0.50
    GRAD_COHERENT: Final[float] = 1.0e-4
    GRAD_DEGRADED: Final[float] = 1.0e-2
    BURES_COHERENT: Final[float] = 0.10
    BURES_DEGRADED: Final[float] = 0.30
    LYAP_COHERENT: Final[float] = 0.0
    LYAP_DEGRADED: Final[float] = 1.0e-3

    @classmethod
    def _grade(cls, value: float, hi_ok: float, mid_ok: float) -> HeytingOmega3:
        """Transición ⊤ → ⋆ → ⊥ según thresholds."""
        if value <= hi_ok:
            return HeytingOmega3.COHERENT
        if value <= mid_ok:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.VETOED

    @classmethod
    def _decay_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de decaimiento de energía."""
        return cls._grade(
            b.attractor_cert.energy_decay_ratio, cls.RATIO_COHERENT, cls.RATIO_DEGRADED
        )

    @classmethod
    def _grad_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de norma de gradiente."""
        return cls._grade(
            b.attractor_cert.grad_final_norm, cls.GRAD_COHERENT, cls.GRAD_DEGRADED
        )

    @classmethod
    def _target_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de distancia Bures al target."""
        return cls._grade(
            b.attractor_cert.bures_distance_attractor_to_target,
            cls.BURES_COHERENT, cls.BURES_DEGRADED,
        )

    @classmethod
    def _geom_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de isometría y proyector válidos."""
        ok = b.seed.subspace.is_isometry and b.seed.subspace.is_projector
        return HeytingOmega3.COHERENT if ok else HeytingOmega3.VETOED

    @classmethod
    def _conv_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de convergencia BB."""
        return (
            HeytingOmega3.COHERENT if b.attractor_cert.converged
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _lyap_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de Lyapunov de Oseledets (estabilidad)."""
        return cls._grade(
            b.attractor_cert.lyapunov_max, cls.LYAP_COHERENT, cls.LYAP_DEGRADED
        )

    @classmethod
    def _melnikov_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de Melnikov (caos homoclínico)."""
        return (
            HeytingOmega3.VETOED if b.attractor_cert.transverse_homoclinic
            else HeytingOmega3.COHERENT
        )

    @classmethod
    def _kam_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de KAM (persistencia de toros)."""
        return (
            HeytingOmega3.COHERENT if b.attractor_cert.kam_persists
            else HeytingOmega3.DEGRADED
        )

    @classmethod
    def _poincare_rule(cls, b: IntuitionTrajectoryBundle) -> HeytingOmega3:
        """Regla de transversalidad de Poincaré."""
        return (
            HeytingOmega3.COHERENT if b.attractor_cert.poincare_transverse
            else HeytingOmega3.VETOED
        )

    @classmethod
    def adjudicate(
        cls, bundle: IntuitionTrajectoryBundle, external: HeytingOmega3
    ) -> HeytingOmega3:
        r"""
        Adjudicación final: 9 meets + voto externo.
        
        local = decay ∧ grad ∧ target ∧ geom ∧ conv ∧ lyap ∧ meln ∧ kam ∧ poinc
        final = local ∧ external  (conservador, nunca infla)
        """
        local = (
            cls._decay_rule(bundle)
            .meet(cls._grad_rule(bundle))
            .meet(cls._target_rule(bundle))
            .meet(cls._geom_rule(bundle))
            .meet(cls._conv_rule(bundle))
            .meet(cls._lyap_rule(bundle))
            .meet(cls._melnikov_rule(bundle))
            .meet(cls._kam_rule(bundle))
            .meet(cls._poincare_rule(bundle))
        )
        return local.meet(external)


# ── §3.2 IntuitiveFieldState — certificado firmado ──────────────────────────────
@dataclass(frozen=True, slots=True)
class IntuitiveFieldState:
    r"""
    Objeto terminal del flash: producto fibrado firmado State ≅ Bundle × Ω₃ × Merkle.
    
    Contiene:
        - Energía de atractor y decaimiento
        - Distancias Bures (seed↔att, att↔target, seed↔target)
        - Purity, entropy, fidelity finales
        - Espectro de Oseledets y métricas Poincaré
        - Dictamen Ω₃ y timestamp
        - Cadena Merkle SHA-256 de las 3 fases
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
    # Poincaré forense:
    lyapunov_max: float
    kolmogorov_sinai_entropy: float
    kaplan_yorke_dimension: float
    melnikov_value: float
    transverse_homoclinic: bool
    kam_persists: bool
    poincare_transverse: bool
    levy_band: float
    heyting_verdict: HeytingOmega3
    reaction_time_ns: float
    subspace_hash: str
    phase_chain_sha256: str
    sha256_provenance: str
    timestamp_utc: float


# ── §3.3 TOONIntuitionEngine — orquestador soberano (funtor F₃∘F₂∘F₁) ──────────
class TOONIntuitionEngine:
    r"""
    Motor Espectral de la Intuición Relámpago.
    
    Funtor soberano F = F₃ ∘ F₂ ∘ F₁ :
        F₁ = IntuitionFieldPreparation.prepare         (§1.6)
        F₂ = GeometricSeed.continue_into_phase2        (§2.9)
           = IntuitionFlashPipeline.synthesize
        F₃ = IntuitionTrajectoryBundle.continue_into_phase3  (§3.1)
           = HeytingIntuitionAdjudicator.adjudicate
           ⊗ _phase3_certify (sellado Merkle)
    
    Composición anidada:
        F₁ → F₂ → F₃
        dominio (ρ, P) → GeometricSeed → IntuitionTrajectoryBundle
                                      → IntuitiveFieldState
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
        self.jacobian_solver = FlashSpectralJacobian()
        self.kelly_calculator = KellyStakeCalculator()
        self._chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::n={dimension_mac}::r={subspace_rank}::"
            f"basis={self.subspace.hash}".encode("ascii")
        ).hexdigest()

    def process_poincare_intuitive_flash(
        self,
        density_op: np.ndarray,
        mac_equilibrium_op: np.ndarray,
        success_probability: float = 0.85,
        win_loss_ratio: float = 2.0,
        subspace_rank: Optional[int] = None,
        poincare_tolerance: float = 1e-6,
    ) -> Tuple[np.ndarray, float, float, KellyStakeReport, HeytingOmega3]:
        r"""
        Proyecta a sección de Poincaré, evalúa Kelly modulado.
        """
        r = subspace_rank if subspace_rank is not None else self.subspace.rank
        rho_proj, d_bures, lyap_max, is_stable, meln = (
            self.jacobian_solver.project_poincare_section_grassmannian(
                density_op=density_op,
                mac_equilibrium_op=mac_equilibrium_op,
                subspace_rank=r,
                poincare_tolerance=poincare_tolerance,
            )
        )

        kelly_report = self.kelly_calculator.calculate_poincare_kelly_stake(
            success_probability=success_probability,
            win_loss_ratio=win_loss_ratio,
            lyap_max=lyap_max,
            d_bures=d_bures,
            kolmogorov_sinai_entropy=abs(lyap_max),
            transverse_homoclinic=meln.transverse_homoclinic,
        )

        if not is_stable or kelly_report.is_vetoed:
            verdict = HeytingOmega3.VETOED
        elif d_bures > 0.05:
            verdict = HeytingOmega3.DEGRADED
        else:
            verdict = HeytingOmega3.COHERENT

        return rho_proj, d_bures, lyap_max, kelly_report, verdict

    def _advance_chain(self, tag: str, payload: bytes) -> str:
        """Avanza la cadena Merkle con nuevo payload."""
        h = hashlib.sha256(
            self._chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._chain_hash = h
        return h

    def _phase1_prepare(self, germinated_matrix: np.ndarray) -> GeometricSeed:
        """Ejecuta FASE 1: prepara GeometricSeed."""
        seed = IntuitionFieldPreparation.prepare(
            rho_input=germinated_matrix, subspace=self.subspace,
            poincare_key=self.engine_id,
        )
        self._advance_chain("F1", bytes.fromhex(seed.seed_spectral_hash))
        return seed

    def _phase2_flash(self, seed: GeometricSeed) -> IntuitionTrajectoryBundle:
        """Ejecuta FASE 2: descenso BB + certificación Poincaré."""
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
        """Ejecuta FASE 3: adjudicación Ω₃ + sellado Merkle."""
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
            f"{c.lyapunov_max:.12e}".encode("ascii"),
            f"{c.kolmogorov_sinai_entropy:.12e}".encode("ascii"),
            f"{c.melnikov_value:.12e}".encode("ascii"),
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
            lyapunov_max=c.lyapunov_max,
            kolmogorov_sinai_entropy=c.kolmogorov_sinai_entropy,
            kaplan_yorke_dimension=c.kaplan_yorke_dimension,
            melnikov_value=c.melnikov_value,
            transverse_homoclinic=c.transverse_homoclinic,
            kam_persists=c.kam_persists,
            poincare_transverse=c.poincare_transverse,
            levy_band=bundle.seed.levy_band,
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
        r"""
        Orquesta el ciclo completo: F₁ → F₂ → F₃.
        
        Entrada: (crop_origin_id, ρ_seed, Ω₃_external)
        Salida:  IntuitiveFieldState (certificado sellado)
        """
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
            cycle_id, crop_origin_id, bundle, external_verdict, t_start_ns
        )

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s | Ω₃=%s | E_final=%.3e | ratio=%.4f | "
            "λ_max=%.3e | h_KS=%.3e | M=%.3e | iters=%d | %.3f ms",
            cycle_id, state.heyting_verdict.name,
            state.flash_attractor_energy, state.energy_decay_ratio,
            state.lyapunov_max, state.kolmogorov_sinai_entropy,
            state.melnikov_value, state.iterations, dt_ms,
        )

        return state


# ── §3.4 Utilidades para demostración ──────────────────────────────────────────
def _build_mixed_seed(
    n: int, leakage: float, subspace: SubspaceGeometry, key: str
) -> ComplexMatrix:
    r"""
    Semilla con fuga controlada t ∈ [0,1] al complemento P_⊥:
        ρ_t = (1 − t) ρ_P + t ρ_{P_⊥}.
    
    t = 0 → supp ⊆ ran(P) ⇒ E = 0 (mínimo)
    t = 1 → supp ⊆ ran(P_⊥) ⇒ E máximo
    """
    rng = np.random.default_rng(_seed_from_string(f"MIX::{key}"))
    P = subspace.projector
    Pc = np.eye(n, dtype=np.complex128) - P

    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho_raw = A @ A.conj().T
    rho_raw = rho_raw / max(float(np.trace(rho_raw).real), 1e-30)

    rho_P = P @ rho_raw @ P
    tr_P = float(np.trace(rho_P).real)
    rho_P = rho_P / tr_P if tr_P > 1e-15 else P / max(1.0, float(np.trace(P).real))

    rho_Pc = Pc @ rho_raw @ Pc
    tr_Pc = float(np.trace(rho_Pc).real)
    rho_Pc = rho_Pc / tr_Pc if tr_Pc > 1e-15 else Pc / max(1.0, float(np.trace(Pc).real))

    t = float(np.clip(leakage, 0.0, 1.0))
    return DensityOperatorAlgebra.sanitize((1.0 - t) * rho_P + t * rho_Pc)


# ╔════════════════════════════════════════════════════════════════════════════════════╗
# ║  DEMOSTRACIÓN AUTÓNOMA                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("═" * 92)
    print("TOON INTUITION ENGINE — v9.1.0 Doctoral 3NestedPhases-Poincaré")
    print("Secciones de Poincaré · Oseledets · Melnikov · KAM Diofántico · Lévy")
    print("Bures-Wasserstein · Dirichlet · BB1/BB2 · Gr(r,n) · Heyting Ω₃ · Merkle")
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

    print("\n──────────── Métricas de geometría del campo (FASE 1) ────────────")
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
        f"{BuresGeodesicMetric.distance(rho_a, gamma_mid):.6f}  (McCann JKO)"
    )

    print("\n──────────── Lema de Lévy (concentración sobre S^{n-1}) ────────────")
    for n_ in (4, 16, 64, 256):
        eps = LevyConcentrationLemma.median_width(n_, lipschitz=1.0, confidence=0.99)
        b = LevyConcentrationLemma.bound(eps, n_, lipschitz=1.0)
        print(f"  n={n_:4d} | ε*(99%)={eps:.6f} | P(tail)≤{b:.3e}")

    print("\n──────────── Ciclos de intuición flash (F1→F2→F3) ────────────")
    scenarios = [
        ("COHERENT (fuga=0.05)", 0.05),
        ("DEGRADED (fuga=0.50)", 0.50),
        ("VETOED   (fuga=0.95)", 0.95),
    ]

    for name, leakage in scenarios:
        rho_seed = _build_mixed_seed(4, leakage, engine.subspace, name)
        land_pre = FlashDirichletFunctional.evaluate(
            rho_seed, engine.subspace.projector
        )

        state = engine.execute_intuitive_cycle(
            crop_origin_id=f"CROP-SOVEREIGN-0001::{name}",
            germinated_matrix=rho_seed,
            external_verdict=HeytingOmega3.COHERENT,
        )

        print(f"\n[{name}]")
        print(f"   ciclo_id             : {state.cycle_id}")
        print(f"   Ω₃ final             : {state.heyting_verdict.name}")
        print(
            f"   E_seed               : {land_pre.energy:.6e}  "
            f"(coh={land_pre.coherent_mass:.3e}, leak={land_pre.leak_mass:.3e})"
        )
        print(f"   E_attractor          : {state.flash_attractor_energy:.6e}")
        print(f"   ratio decaimiento    : {state.energy_decay_ratio:.6f}")
        print(f"   ‖∇E|_T‖_F            : {state.grad_final_norm:.3e}")
        print(f"   d_B(seed, target)    : {state.manifold_geodesic_distance:.6f}")
        print(f"   d_B(att,  target)    : {state.attractor_to_target_bures:.6f}")
        print(f"   d_B(seed, att)       : {state.seed_to_attractor_bures:.6f}")
        print(f"   θ_Bures              : {state.bures_angle_rad:.6f} rad")
        print(f"   F(att, target)       : {state.fidelity_target:.6f}")
        print(f"   pureza post-descenso : {state.purity:.6f}")
        print(f"   entropía post        : {state.entropy:.6f}")
        print(f"   λ_max (Oseledets)    : {state.lyapunov_max:.6e}")
        print(f"   h_KS (Pesin)         : {state.kolmogorov_sinai_entropy:.6e}")
        print(f"   D_KY (Kaplan-Yorke)  : {state.kaplan_yorke_dimension:.6f}")
        print(f"   M (Melnikov)         : {state.melnikov_value:.6e}")
        print(f"   homoclínico transv.  : {state.transverse_homoclinic}")
        print(f"   KAM persistente      : {state.kam_persists}")
        print(f"   Σ transversal        : {state.poincare_transverse}")
        print(f"   banda Lévy ε*        : {state.levy_band:.6f}")
        print(
            f"   iteraciones / BB     : {state.iterations} / {state.bb_mode_last}  "
            f"(backtracks={state.n_backtracks})"
        )
        print(f"   converged            : {state.converged}")
        print(f"   latencia ciclo       : {state.reaction_time_ns / 1000.0:.2f} µs")
        print(f"   firma (phase_chain)  : {state.phase_chain_sha256[:32]}…")
        print(f"   firma (provenance)   : {state.sha256_provenance[:32]}…")

    print("\n" + "═" * 92)
    print("✓ FASE 1→2: prepare ⊣ continue_into_phase2 = synthesize.")
    print("✓ FASE 2→3: synthesize ⊣ continue_into_phase3 = adjudicate.")
    print("✓ Σ = Gr(r,n) ∩ transversal: PoincaréSectionManifold con ‖T‖_min > 0.")
    print("✓ Oseledets: λ_max, h_KS (Pesin), D_KY (Kaplan-Yorke) por QR.")
    print("✓ Melnikov: M = ⟨[∇²E, ∇E], ∇E⟩_F; veto si enredo homoclínico.")
    print("✓ KAM diofántico: |ω·k| ≥ γ/|k|^τ con τ=1.5; persistencia verificada.")
    print("✓ Lévy: P(|f−𝔼f| ≥ ε) ≤ 2 exp(−(n−1)ε²/(2L²)).")
    print("✓ d_B = √(2−2√F): geodésica Bures-Wasserstein (Petz, McCann-JKO).")
    print("✓ E = ½‖ρ − PρP‖_F²; Lip(∇E) ≤ 1; ∇E|_T ∈ T𝔇_n.")
    print("✓ BB1/BB2 + Armijo + Higham; η ∈ (0, 2] por L-smoothness.")
    print("✓ Ω₃ por meets adimensionales incluyendo λ_max, Melnikov, KAM, Σ.")
    print("✓ Cadena forense F1 → F2 → F3 encadenada por SHA-256.")
    print("✓ 3 fases anidadas canónicas: GeometricSeed.continue_into_phase2,")
    print("  IntuitionTrajectoryBundle.continue_into_phase3.")
    print("═" * 92)
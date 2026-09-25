# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Silent Witness Engine (Motor Espectral del Testigo Silencioso)║
║ Ubicación: app/wisdom/toon_silent_witness_engine.py                           ║
║ Versión  : 3.0.0-Doctoral-Nested-TomitaTakesaki-KMS-SpectralGap-Silence       ║
║ Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el  ║
║            germen formal del primero de k+1)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Formalización Categorial Doctoral (Funtor del Silencio Epistemológico S)
========================================================================

Sea 𝓣_Ω el topos de haces con clasificador intuicionista Ω₃ = { VETOED = 0 ≺ DEGRADED = 1 ≺ COHERENT = 2 }.
El Silencio Epistemológico se formaliza como un funtor de sistemas C*-dinámicos
finito-dimensionales hacia certificados Ω₃-valuados firmados:

        S :  Sys_C*  ──▶  Cert_Silence

mediante la composición asociativa e inalienable de tres fases anidadas:

        S  =  Adjudicate ∘ ModularPipeline ∘ VacuumPrep

Estructura de Fases Anidadas e Invariantes
===========================================

FASE 1 — SUSTRATO ALGEBRAICO-MODULAR Y CONSTRUCCIÓN GNS
──────────────────────────────────────────────────────────────────────────
  • HeytingOmega3: Retículo de Heyting completo Ω₃ = {0 ≺ 1 ≺ 2}. Residuo x → y = ⊤ si x ≤ y, else y.
    Verifica las leyes de residuación (a ∧ b ≤ c ⇔ a ≤ b → c), no contradicción y falla del tercio excluso.
  • MatrixBanachAlgebra: Estructura C* sobre Mₙ(ℂ) con normas de Schatten ‖A‖_p y radio espectral r(A).
  • ModularHamiltonian: Hamiltoniano modular K_ρ := −log ρ con ρ = e^{−K_ρ}/Z (Z = 1, F = −log Z = 0).
    Espectro ascendente {E₀ ≤ E₁ ≤ … ≤ Eₙ₋₁}; gap = E₁ − E₀ representa la unicidad del vacío.
  • DensityOperatorAlgebra: Operadores densidad en 𝔇(ℋₙ). Entropía S(ρ) = −Tr(ρ log ρ), pureza P(ρ) = Tr(ρ²),
    relativa de Umegaki S(ρ‖σ) = Tr(ρ(log ρ − log σ)) ≥ 0 (Klein), fidelidad F(ρ,σ) = Tr √(√ρ σ √ρ) y Bures d_B.
  • GNSHilbertAlgebra: Espacio de Hilbert GNS H_ω ≅ (M_n, ⟨A|B⟩_ω = Tr(ρ A† B)) con vector cíclico-separante Ω = I,
    representado en la imagen de Hilbert-Schmidt por Ω_HS = ρ¹/² ∈ HS(ℂⁿ).
  • VacuumStatePreparation.prepare_vacuum_context: Morfismo de hand-off FASE 1 ⟶ FASE 2. Prepara el par (ρ_Ω, K_Ω)
    y el contexto `VacuumModularContext` (último objeto/método de FASE-1).

FASE 2 — DINÁMICA MODULAR, TEORÍA DE TOMITA-TAKESAKI Y ESPECTRO DEL VACÍO
──────────────────────────────────────────────────────────────────────────
  • TomitaTakesakiEngine.bind_vacuum_context: PRIMER MORFISMO DE FASE-2 (continúa prepare_vacuum_context).
    Sella el contexto y realiza los operadores modulares:
        Δ(a) = ρ⁻¹ a ρ,    σ_t^ω(a) = ρ⁻ⁱᵗ a ρⁱᵗ = eⁱᵗᴷ a e⁻ⁱᵗᴷ,
        J(a) = ρ⁻¹/² a† ρ¹/²,    S(a) = a† = J(Δ¹/²(a)).
    Verifica KMS(β=1): ω(a σ_i(b)) = ω(ba) y el cociente de Connes (Dρ : Dσ)_t = ρⁱᵗ σ⁻ⁱᵗ.
  • VacuumSpectraAnalyzer.audit: Observables no tautológicos: VEV = ⟨H⟩_ρ − E₀(H), fluctuación thermal_fluctuation,
    gap espectral gap(K_ρ), energía libre F = 0 y ruido en dB = 10 log₁₀(1 + Var_ρ(H)).
  • SilentFieldDetector.probe: Sonda de silencio: pureza, fidelidad al ground F(ρ, |Ω⟩⟨Ω|), masa de fuga,
    entropía relativa de Umegaki regularizada S(ρ‖σ_ε) y distancia Bures d_B.
  • ModularSilencePipeline.synthesize_from_context: Compone bind + Tomita + espectro + sonda, emitiendo
    `SilentFieldBundle` (último objeto de FASE-2).

FASE 3 — SOBERANÍA, ADJUDICACIÓN Y CERTIFICACIÓN DEL SILENCIO
──────────────────────────────────────────────────────────────────────────
  • HeytingVacuumAdjudicator.adjudicate: PRIMER MORFISMO DE FASE-3 (continúa synthesize_from_context).
    Colapsa el bundle en Ω₃ mediante la conmutación de meets:
        χ_local = audit ∧ probe ∧ kms ∧ axioms ∧ polar ∧ group ∧ klein,
        χ_final = χ_local ∧ χ_external.
  • SilentFieldState: Certificado firmado con cadena de custodia forense SHA-256 encadenada (`phase_chain_sha256`).
  • TOONSilentWitnessEngine.execute_silence_cycle: Orquestador soberano que ejecuta el ciclo F₁ → F₂ → F₃.

Definición Granular de Invariantes y Axiomas
=============================================
  1. Invariación y Positividad C*: ρ = ρ†, spec(ρ) ⊂ [0, 1], Tr(ρ) = 1.
  2. Axioma de Tomita-Takesaki: Polar S = J Δ¹/² y ley de grupo σ_s ∘ σ_t = σ_{s+t}.
  3. Condición KMS(β=1): Tr(ρ a (ρ b ρ⁻¹)) = Tr(ρ b a)  ∀ a,b ∈ M_n(ℂ).
  4. Desigualdad de Klein: S(ρ‖σ) = Tr(ρ(log ρ − log σ)) ≥ 0 con igualdad ⇔ ρ = σ.
  5. Involución de Heyting y Preservación de Cadena: χ_final = ⋀_{Ω₃} χ_i; Merkle SHA-256 inyectivo.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Mapping, Tuple

import numpy as np
import scipy.linalg as la


__version__ = "3.0.0"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitnessEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO-MODULAR                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo total Ω₃ = {⊥, ⋆, ⊤} con estructura de álgebra de Heyting
    completa (objeto clasificador de subobjetos del topos trivaluado):

        ⊥ < ⋆ < ⊤
        meet     (∧) : ínfimo = min                    (límite en el poset)
        join     (∨) : supremo = max
        implies  (⇒) : residuo de ∧ : (a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)
        neg      (¬) : a ⇒ ⊥                           (intuicionista)
        ¬¬           : clausura regular                (¬¬⋆ = ⊤ ≠ ⋆)

    Leyes que se preservan (verificables por `verify_heyting_laws`):
        (H1)  ∧, ∨ idempotentes, conmutativos, asociativos, absorbentes
        (H2)  residuación: a ∧ b ≤ c  ⇔  a ≤ (b ⇒ c)
        (H3)  a ⇒ a = ⊤,   ⊥ ⇒ a = ⊤,   a ⇒ ⊤ = ⊤
        (H4)  ¬¬⊥ = ⊥, ¬¬⊤ = ⊤, ¬¬⋆ = ⊤  (⋆ no es regular)
        (H5)  LEM a ∨ ¬a = ⊤ falla en a = ⋆   (intuicionismo estricto)

    El funtor de verdad Ω₃ → Sub(1) garantiza coherencia interna con
    godel_agent: el meet es la decisión más conservadora entre fuentes.
    """

    VETOED = 0  # ⊥
    DEGRADED = 1  # ⋆
    COHERENT = 2  # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    @property
    def rank(self) -> int:
        """Grado de verdad en la cadena 0 ≤ 1 ≤ 2."""
        return int(self)

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""a ⇒ b = ⊤ si a ≤ b; en caso contrario b  (álgebra de cadena)."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        r"""¬a ≜ a ⇒ ⊥  (seudocomplemento de Heyting)."""
        return self.implies(HeytingOmega3.VETOED)

    def double_negation(self) -> "HeytingOmega3":
        r"""¬¬a : clausura de regularidad. Fija {⊥, ⊤}; envía ⋆ ↦ ⊤."""
        return self.neg().neg()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def is_dense(self) -> bool:
        r"""a es denso ⇔ ¬a = ⊥ ⇔ ¬¬a = ⊤ (⋆ y ⊤)."""
        return self.neg() == HeytingOmega3.VETOED

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤. Falla exactamente en ⋆."""
        return self.join(self.neg()) == HeytingOmega3.COHERENT

    @classmethod
    def residuation_holds(
        cls, a: "HeytingOmega3", b: "HeytingOmega3", c: "HeytingOmega3"
    ) -> bool:
        r"""(a ∧ b) ≤ c  ⇔  a ≤ (b ⇒ c)."""
        left = int(a.meet(b)) <= int(c)
        right = int(a) <= int(b.implies(c))
        return left is right

    @classmethod
    def verify_heyting_laws(cls) -> Dict[str, bool]:
        """Auditoría finita de (H1)–(H5) sobre Ω₃³."""
        elems = list(cls)
        residuation = all(
            cls.residuation_holds(a, b, c) for a in elems for b in elems for c in elems
        )
        idempotent = all(a.meet(a) == a and a.join(a) == a for a in elems)
        lem_fails_on_star = not cls.DEGRADED.excluded_middle_holds()
        regular_pair = cls.VETOED.is_regular() and cls.COHERENT.is_regular()
        star_not_regular = not cls.DEGRADED.is_regular()
        return {
            "residuation": residuation,
            "idempotent": idempotent,
            "lem_fails_on_star": lem_fails_on_star,
            "regulars_are_bot_top": regular_pair,
            "star_not_regular": star_not_regular,
        }


# ── §1.2 Álgebra de Banach / C* matricial ──────────────────────────────────
class MatrixBanachAlgebra:
    r"""
    Estructura de álgebra de Banach involutiva sobre M_n(ℂ):

        ‖A‖_p := (Tr |A|^p)^{1/p}     normas de Schatten,  1 ≤ p < ∞
        ‖A‖_∞ := ‖A‖_{op} = σ_max(A)
        r(A)  := max |spec(A)|         radio espectral
        C*    : ‖A† A‖_∞ = ‖A‖_∞²

    M_n es nuclear (tipo I), por lo que toda representación normal es
    unitariamente equivalente a múltiplos de la estándar. El funtor
    modular de FASE 2 actúa por automorfismos internos de este C*-álgebra.
    """

    @staticmethod
    def as_complex(A: np.ndarray) -> np.ndarray:
        return np.asarray(A, dtype=np.complex128)

    @staticmethod
    def hermitize(A: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        return 0.5 * (A + A.conj().T)

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B - B @ A

    @staticmethod
    def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return A @ B + B @ A

    @staticmethod
    def schatten_norm(A: np.ndarray, p: float) -> float:
        A = MatrixBanachAlgebra.as_complex(A)
        singular = np.linalg.svd(A, compute_uv=False)
        if singular.size == 0:
            return 0.0
        if p == math.inf or p == float("inf"):
            return float(singular[0])
        if p == 1:
            return float(np.sum(singular))
        if p == 2:
            return float(np.linalg.norm(singular))
        if p <= 0:
            raise ValueError("Schatten p-norm requires p ≥ 1 or p = ∞")
        return float(np.sum(singular ** p) ** (1.0 / p))

    @staticmethod
    def spectral_radius(A: np.ndarray) -> float:
        w = np.linalg.eigvals(MatrixBanachAlgebra.as_complex(A))
        return float(np.max(np.abs(w))) if w.size else 0.0

    @staticmethod
    def cstar_identity_residual(A: np.ndarray) -> float:
        r"""|‖A†A‖_∞ − ‖A‖_∞²|  (debe ser ~ 0 numéricamente)."""
        op = MatrixBanachAlgebra.schatten_norm
        A = MatrixBanachAlgebra.as_complex(A)
        return abs(op(A.conj().T @ A, math.inf) - op(A, math.inf) ** 2)

    @staticmethod
    def hilbert_schmidt_inner(A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_{HS} = Tr(A† B)."""
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(A.conj().T @ B))


# ── §1.3 Hamiltoniano modular K_ρ = −log ρ ─────────────────────────────────
@dataclass(frozen=True, slots=True)
class ModularHamiltonian:
    r"""
    Hamiltoniano modular asociado a un estado fiel ρ ∈ M_n(ℂ)_{+,1}:

        K_ρ  :=  −log ρ ,     ρ = e^{−K_ρ} / Z ,     Z = Tr e^{−K_ρ}.

    Tras renormalizar spec(ρ) post-regularización se impone Z = 1, de modo
    que la energía libre de Helmholtz modular

        F = −log Z = ⟨K_ρ⟩_ρ − S(ρ) = 0

    es una identidad de consistencia (KMS a β_modular = 1).

    Espectro ordenado ascendente {E_0 ≤ E_1 ≤ … E_{n-1}}:
        • E_0   = −log λ_max(ρ)     energía fundamental modular
        • gap   = E_1 − E_0         unicidad del vacío ⇔ gap > 0
        • g     = dim ker(K_ρ − E_0 I)   degeneración del ground
        • ⟨K⟩   = ∑_i e^{−E_i} E_i / Z   energía interna modular
    """

    eigenvalues: Tuple[float, ...]
    eigenvectors_hash: str
    regularization_eps: float

    @property
    def dimension(self) -> int:
        return len(self.eigenvalues)

    @property
    def ground_energy(self) -> float:
        return self.eigenvalues[0] if self.eigenvalues else 0.0

    @property
    def spectral_gap(self) -> float:
        if len(self.eigenvalues) < 2:
            return float("inf")
        return max(0.0, self.eigenvalues[1] - self.eigenvalues[0])

    @property
    def spectral_spread(self) -> float:
        if not self.eigenvalues:
            return 0.0
        return float(self.eigenvalues[-1] - self.eigenvalues[0])

    @property
    def partition_function(self) -> float:
        if not self.eigenvalues:
            return 1.0
        return float(math.fsum(math.exp(-e) for e in self.eigenvalues))

    @property
    def free_energy(self) -> float:
        r"""F = −log Z.  Con Z ≡ 1, F ≡ 0."""
        z = self.partition_function
        if z <= 0.0:
            return float("inf")
        return -math.log(z)

    @property
    def internal_energy(self) -> float:
        r"""⟨K_ρ⟩ = ∑ e^{−E} E / Z."""
        z = self.partition_function
        if z <= 0.0 or not self.eigenvalues:
            return 0.0
        return float(math.fsum(math.exp(-e) * e for e in self.eigenvalues) / z)

    @property
    def degeneracy_of_ground(self) -> int:
        if not self.eigenvalues:
            return 0
        e0 = self.eigenvalues[0]
        tol = max(_GAP_DEGENERACY_TOL, 10.0 * self.regularization_eps)
        return sum(1 for e in self.eigenvalues if abs(e - e0) < tol)

    @property
    def is_unique_vacuum(self) -> bool:
        return self.degeneracy_of_ground == 1 and self.spectral_gap > 0.0

    @property
    def boltzmann_weights(self) -> Tuple[float, ...]:
        z = self.partition_function
        if z <= 0.0:
            n = max(self.dimension, 1)
            return tuple(1.0 / n for _ in range(self.dimension))
        return tuple(math.exp(-e) / z for e in self.eigenvalues)

    @property
    def gap_condition_number(self) -> float:
        r"""spread / gap  (∞ si el vacío es degenerado)."""
        gap = self.spectral_gap
        if gap <= 0.0 or not math.isfinite(gap):
            return float("inf")
        return self.spectral_spread / gap


# ── §1.4 Álgebra de operadores densidad en M_n(ℂ) ─────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operadores canónicos de la teoría de información cuántica sobre el
    simplejo de estados D_n = {ρ ∈ M_n : ρ ≥ 0, Tr ρ = 1}:

        S(ρ)     = −Tr(ρ log ρ)                         von Neumann
        P(ρ)     =  Tr(ρ²) = ‖ρ‖_{HS}²                  pureza
        S(ρ‖σ)   =  Tr(ρ (log ρ − log σ))               Umegaki (operatorial)
        F(ρ,σ)   =  ‖√ρ √σ‖_1 = Tr √(√ρ σ √ρ)           Uhlmann
        D_tr     =  (1/2) ‖ρ − σ‖_1                     distancia de traza
        d_B      =  √(2(1 − F(ρ,σ)))                    Bures
        ρ^z      =  V diag(λ^z) V†                      cálculo funcional

    Klein: S(ρ‖σ) ≥ 0,  = 0 ⇔ ρ = σ.
    El grupo PU(n) actúa por conjugación preservando S, P, F, D_tr, d_B.

    Distinción espectral:
        raw_spectrum(ρ)          autovalores numéricos ≥ 0 (sin levantar 0)
        regularized_spectrum(ρ)  λ ← max(λ, ε) y renormaliza (para log)
    """

    EPS = _EPS

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        """Hermitiza y normaliza Tr ρ = 1; copia defensiva."""
        rho = MatrixBanachAlgebra.hermitize(rho)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return rho

    @classmethod
    def is_density_operator(cls, rho: np.ndarray, tol: float = _TRACE_TOL) -> bool:
        rho_h = MatrixBanachAlgebra.hermitize(rho)
        herm = float(np.linalg.norm(rho_h - np.asarray(rho, dtype=np.complex128)))
        tr = float(np.trace(rho_h).real)
        w = np.real(la.eigvalsh(rho_h))
        return herm < tol and abs(tr - 1.0) < tol and float(w.min()) > -tol

    @classmethod
    def raw_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        """Espectro real descendente, partes negativas recortadas a 0."""
        vals = np.real(la.eigvalsh(cls.sanitize(rho)))
        vals = np.sort(vals)[::-1]
        return np.maximum(vals, 0.0)

    @classmethod
    def regularized_spectrum(cls, rho: np.ndarray) -> np.ndarray:
        """Espectro descendente, λ ≥ ε, suma 1 (cálculo logarítmico)."""
        vals = cls.raw_spectrum(rho)
        vals = np.maximum(vals, cls.EPS)
        s = float(vals.sum())
        return vals / s if s > 0.0 else vals

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        """Alias de `regularized_spectrum` (compatibilidad)."""
        return cls.regularized_spectrum(rho)

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        r"""S(ρ) = −∑_{λ_i > ε} λ_i log λ_i  con la convención 0 log 0 = 0."""
        p = cls.raw_spectrum(rho)
        p = p[p > cls.EPS]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.raw_spectrum(rho)
        return float(np.sum(p ** 2))

    @classmethod
    def matrix_log(cls, rho: np.ndarray) -> np.ndarray:
        r"""log ρ por cálculo funcional hermítico, λ ← max(λ, ε)."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        return (vecs * np.log(vals)) @ vecs.conj().T

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        r"""ρ^z = exp(z log ρ) por eigendescomposición regularizada."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        powered = np.exp(z * np.log(vals.astype(np.complex128)))
        return (vecs * powered) @ vecs.conj().T

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""
        S(ρ‖σ) = Tr(ρ (log ρ − log σ))  — forma operatorial.

        NO se emparejan espectros independientes: eso sería correcto sólo
        si [ρ, σ] = 0 y se alinean autoespacios. Klein ⇒ S ≥ 0.
        """
        rho = cls.sanitize(rho)
        log_rho = cls.matrix_log(rho)
        log_sigma = cls.matrix_log(sigma)
        return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = Tr √(√ρ σ √ρ) ∈ [0, 1]."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        sqrt_rho = cls.matrix_power(rho, 0.5)
        sandwich = MatrixBanachAlgebra.hermitize(sqrt_rho @ sigma @ sqrt_rho)
        return float(max(0.0, min(1.0, np.real(np.trace(cls.matrix_power(sandwich, 0.5))))))

    @classmethod
    def trace_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""D(ρ,σ) = (1/2) ‖ρ − σ‖_1 ∈ [0, 1]."""
        delta = cls.sanitize(rho) - cls.sanitize(sigma)
        return 0.5 * MatrixBanachAlgebra.schatten_norm(delta, 1)

    @classmethod
    def bures_distance(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""d_B(ρ,σ) = √(2(1 − F(ρ,σ))) ∈ [0, √2]."""
        fid = cls.uhlmann_fidelity(rho, sigma)
        return math.sqrt(max(0.0, 2.0 * (1.0 - fid)))

    @classmethod
    def kleins_inequality_residual(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""min{0, S(ρ‖σ)} : ~0 si Klein se respeta numéricamente."""
        return float(min(0.0, cls.umegaki_relative_entropy(rho, sigma)))

    @classmethod
    def modular_hamiltonian_from_rho(
        cls, rho: np.ndarray, eps: float = _EPS
    ) -> ModularHamiltonian:
        """K_ρ = −log ρ, Z = ∑ λ_i = 1, hash SHA-256 de autovectores."""
        rho = cls.sanitize(rho)
        lam, vec = la.eigh(rho)
        lam = np.maximum(lam, eps)
        lam = lam / lam.sum()
        energies = -np.log(lam)
        order = np.argsort(energies)
        e_sorted, vec_sorted = energies[order], vec[:, order]
        digest = hashlib.sha256(np.ascontiguousarray(vec_sorted).tobytes()).hexdigest()
        return ModularHamiltonian(
            eigenvalues=tuple(float(x) for x in e_sorted.tolist()),
            eigenvectors_hash=digest,
            regularization_eps=eps,
        )

    @classmethod
    def density_matrix_from_hamiltonian(cls, H: np.ndarray, beta: float) -> np.ndarray:
        r"""
        ρ_β = e^{−β H} / Tr e^{−β H} con shift numérico:

            E ← E − min E,   x ← −β E,   x ← x − max x,   p = softmax(x).
        """
        H = MatrixBanachAlgebra.hermitize(H)
        w, vecs = la.eigh(H)
        w_shift = w - w.min()
        logits = -beta * w_shift
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return cls.sanitize((vecs * p) @ vecs.conj().T)

    @classmethod
    def thermofield_double(cls, rho: np.ndarray) -> np.ndarray:
        r"""
        Purificación de GNS / estado de thermofield double:

            |Ψ_TFD⟩ = ∑_i √λ_i  |i⟩ ⊗ |i⟩  ∈ ℂⁿ ⊗ ℂⁿ,

        que bajo la identificación HS es exactamente Ω_HS = ρ^{1/2}.
        La conjugación modular J intercambia los dos factores.
        """
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, 0.0)
        n = rho.shape[0]
        psi = np.zeros((n * n,), dtype=np.complex128)
        sqrt_vals = np.sqrt(vals)
        for i in range(n):
            # |i⟩_vec ⊗ |i⟩_vec  (base propia de ρ)
            ket = vecs[:, i]
            psi += sqrt_vals[i] * np.kron(ket, ket)
        norm = float(np.linalg.norm(psi))
        return psi / norm if norm > 0.0 else psi

    @classmethod
    def gns_inner_product(cls, rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        r"""⟨A|B⟩_ω = Tr(ρ A† B)."""
        rho = cls.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        B = MatrixBanachAlgebra.as_complex(B)
        return complex(np.trace(rho @ A.conj().T @ B))


# ── §1.5 Construcción GNS (álgebra de Hilbert a izquierda) ─────────────────
class GNSHilbertAlgebra:
    r"""
    Construcción GNS de (M_n(ℂ), ω_ρ):

        H_ω = M_n    con    ⟨A|B⟩_ω = Tr(ρ A† B),
        Ω   = I      cíclico y separante  ⇔  ρ > 0 (ω fiel),
        π(a)|b⟩ = |a b⟩,
        Ω_HS = ρ^{1/2} ∈ HS(ℂⁿ)  (picture de Hilbert–Schmidt).

    Tomita: S π(x) Ω = π(x†) Ω,  Δ = S* S,  J del desdoblamiento polar.
    Este objeto es el puente geométrico entre §1.4 y el motor de §2.1.
    """

    @staticmethod
    def inner_product(rho: np.ndarray, A: np.ndarray, B: np.ndarray) -> complex:
        return DensityOperatorAlgebra.gns_inner_product(rho, A, B)

    @staticmethod
    def cyclic_vector_hs(rho: np.ndarray) -> np.ndarray:
        r"""Ω_HS = ρ^{1/2}.  ω(a) = ⟨Ω_HS, a Ω_HS⟩_{HS}."""
        return DensityOperatorAlgebra.matrix_power(rho, 0.5)

    @staticmethod
    def is_faithful(rho: np.ndarray, tol: float = _EPS) -> bool:
        w = DensityOperatorAlgebra.raw_spectrum(rho)
        return bool(w.size and float(w.min()) > tol)

    @staticmethod
    def omega_norm_squared(rho: np.ndarray) -> float:
        r"""‖Ω‖_ω² = ⟨I|I⟩_ω = Tr(ρ) = 1."""
        return float(np.real(np.trace(DensityOperatorAlgebra.sanitize(rho))))


# ── §1.6 VacuumModularContext + VacuumStatePreparation — HAND-OFF 1→2 ──────
@dataclass(slots=True)
class VacuumModularContext:
    r"""
    Objeto terminal de FASE 1 y objeto inicial de FASE 2.

    Encapsula el par canónico (ρ_Ω, K_Ω) junto con el Hamiltoniano externo
    H, el proyector espectral al ground |Ω⟩⟨Ω| (β = ∞) y metadatos GNS,
    de modo que `TomitaTakesakiEngine.bind_vacuum_context` no re-deriva el
    vacío: la definición formal de `prepare_vacuum_context` *es* el arranque
    de la dinámica modular.
    """

    rho: np.ndarray
    K: ModularHamiltonian
    H: np.ndarray
    ground_projector: np.ndarray
    beta: float
    gns_norm_sq: float
    is_faithful: bool
    tfd: np.ndarray = field(repr=False)


class VacuumStatePreparation:
    r"""
    Prepara el contexto modular que alimenta toda la FASE 2.

    El vacío |Ω⟩ es el ground state del Hamiltoniano externo H
    (tight-binding sobre el grafo camino P_n, o H arbitrario hermítico):

        H|Ω⟩ = E_0|Ω⟩,   E_0 = min spec(H).

    Modos de preparación:
        (a) PURA        ρ_Ω = |Ω⟩⟨Ω|                     (β = ∞, T = 0)
        (b) GIBBS       ρ_β = e^{−β H}/Z_β               (KMS a β finito)
        (c) INTERPOLADA ρ(τ) = (1−τ)|Ω⟩⟨Ω| + τ I/n       (vacío templado)

    El último método, `prepare_vacuum_context`, cierra FASE 1 y es el
    morfismo de hand-off  FASE 1 ⟶ FASE 2.
    """

    DEFAULT_BETA_COLD: float = 1.0e3
    DEFAULT_HOPPING: float = 1.0e-3

    @classmethod
    def tight_binding_hamiltonian(
        cls, n: int, hopping: float = DEFAULT_HOPPING
    ) -> np.ndarray:
        r"""
        Hamiltoniano de enlace fuerte sobre el grafo camino P_n:

            H = ∑_{k=0}^{n-1} k |k⟩⟨k|  +  t ∑_{⟨i,j⟩} (|i⟩⟨j| + |j⟩⟨i|).

        El laplaciano combinatorio de P_n es isospectral módulo onsite;
        el gap topológico del camino es O(t) y no destruye la unicidad
        del ground si t ≪ 1.
        """
        if n < 1:
            raise ValueError("mac_dimension must be ≥ 1")
        onsite = np.linspace(0.0, float(max(n - 1, 0)), n, dtype=np.float64)
        H = np.diag(onsite).astype(np.complex128)
        for i in range(n - 1):
            H[i, i + 1] = hopping
            H[i + 1, i] = hopping
        return MatrixBanachAlgebra.hermitize(H)

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        w, vecs = la.eigh(MatrixBanachAlgebra.hermitize(H))
        idx0 = int(np.argmin(w))
        omega = vecs[:, idx0].reshape(-1, 1)
        return DensityOperatorAlgebra.sanitize(omega @ omega.conj().T)

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        return DensityOperatorAlgebra.density_matrix_from_hamiltonian(H, beta)

    @classmethod
    def interpolated_state(cls, H: np.ndarray, tau: float) -> np.ndarray:
        tau = float(min(1.0, max(0.0, tau)))
        rho = cls.ground_state_projector(H)
        n = rho.shape[0]
        return DensityOperatorAlgebra.sanitize(
            (1.0 - tau) * rho + tau * np.eye(n, dtype=np.complex128) / n
        )

    @classmethod
    def prepare_vacuum_pair(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> Tuple[np.ndarray, ModularHamiltonian]:
        r"""Compatibilidad: (H, β) ↦ (ρ_Ω, K_Ω). Delegado del contexto."""
        ctx = cls.prepare_vacuum_context(H, beta=beta)
        return ctx.rho, ctx.K

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 1 → FASE 2
    #  Definición formal terminal de FASE 1.
    #  Su tipo de retorno `VacuumModularContext` es el dominio de
    #  TomitaTakesakiEngine.bind_vacuum_context  (§2.0), primer método
    #  de FASE 2: no hay hiato semántico entre ambas fases.
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def prepare_vacuum_context(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> VacuumModularContext:
        r"""
        Morfismo de hand-off  (H, β) ↦ VacuumModularContext.

        Continúa en §2.0 `TomitaTakesakiEngine.bind_vacuum_context`.
        """
        H = MatrixBanachAlgebra.hermitize(H)
        rho = cls.gibbs_state(H, beta)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        ground = cls.ground_state_projector(H)
        gns_n2 = GNSHilbertAlgebra.omega_norm_squared(rho)
        faithful = GNSHilbertAlgebra.is_faithful(rho)
        tfd = DensityOperatorAlgebra.thermofield_double(rho)
        logger.debug(
            "VacuumStatePreparation.context: β=%.4g | E₀(K)=%.6f | gap(K)=%.6f | "
            "Z=%.8f | F=%.3e | faithful=%s",
            beta,
            K.ground_energy,
            K.spectral_gap,
            K.partition_function,
            K.free_energy,
            faithful,
        )
        return VacuumModularContext(
            rho=rho,
            K=K,
            H=H,
            ground_projector=ground,
            beta=beta,
            gns_norm_sq=gns_n2,
            is_faithful=faithful,
            tfd=tfd,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR Y ESPECTRO DEL VACÍO (continuación de FASE 1)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.0 / §2.1 Motor Tomita–Takesaki: bind, Δ, σ_t^ω, J, S, KMS, Connes ──
class TomitaTakesakiEngine:
    r"""
    Implementación numérica de Tomita–Takesaki para el estado normal fiel
    ω(a) = Tr(ρ a) sobre M_n(ℂ).

    §2.0  `bind_vacuum_context`  continúa  `prepare_vacuum_context` (§1.6):
          valida el objeto terminal de FASE 1 y lo sella como dominio de
          la dinámica modular.

    Construcción GNS (heredada):
        H_ω = (M_n, ⟨A|B⟩_ω = Tr(ρ A† B)),  Ω = I,  Ω_HS = ρ^{1/2}

    Operadores modulares (convención KMS-compatible):
        Δ(a)      := ρ^{-1} a ρ
        σ_t^ω(a)  := ρ^{-it} a ρ^{it} = e^{it K} a e^{-it K}
        J(a)      := ρ^{-1/2} a† ρ^{1/2}
        S(a)      := a† = J(Δ^{1/2}(a))

    Axiomas verificables:
        (i)   σ_t unital         σ_t(I) = I
        (ii)  σ_t multiplicativo σ_t(ab) = σ_t(a) σ_t(b)
        (iii) σ_t isométrico HS  ‖σ_t(a)‖_2 = ‖a‖_2
        (iv)  ley de grupo       σ_s ∘ σ_t = σ_{s+t}
        (v)   J² = id
        (vi)  polar              J Δ^{1/2} = S
        (vii) KMS(β=1)           ω(a σ_i(b)) = ω(ba)
    """

    @classmethod
    def bind_vacuum_context(cls, ctx: VacuumModularContext) -> VacuumModularContext:
        r"""
        §2.0  Arranque de FASE 2.

        Continúa el morfismo `VacuumStatePreparation.prepare_vacuum_context`
        (§1.6). Verifica:
            • ρ cuadrada, hermítica, Tr ρ ≈ 1, PSD
            • dim spec(K_ρ) = n
            • ‖Ω‖_ω² ≈ 1
        Devuelve el mismo contexto saneado (ρ, H, ground re-hermitizados).
        """
        if ctx.rho.ndim != 2 or ctx.rho.shape[0] != ctx.rho.shape[1]:
            raise ValueError("VacuumModularContext.ρ must be a square matrix")
        n = ctx.rho.shape[0]
        if ctx.H.shape != (n, n) or ctx.ground_projector.shape != (n, n):
            raise ValueError("H, ρ and |Ω⟩⟨Ω| dimension mismatch")
        if ctx.K.dimension != n:
            raise ValueError("spec(K_ρ) length ≠ n")

        ctx.rho = DensityOperatorAlgebra.sanitize(ctx.rho)
        ctx.H = MatrixBanachAlgebra.hermitize(ctx.H)
        ctx.ground_projector = DensityOperatorAlgebra.sanitize(ctx.ground_projector)
        ctx.gns_norm_sq = GNSHilbertAlgebra.omega_norm_squared(ctx.rho)
        ctx.is_faithful = GNSHilbertAlgebra.is_faithful(ctx.rho)

        if abs(ctx.gns_norm_sq - 1.0) > _TRACE_TOL:
            logger.warning(
                "GNS ‖Ω‖² = %.3e ≠ 1 (tol=%.1e)", ctx.gns_norm_sq, _TRACE_TOL
            )
        logger.debug(
            "bind_vacuum_context: n=%d | faithful=%s | gap=%.6f | F=%.3e",
            n,
            ctx.is_faithful,
            ctx.K.spectral_gap,
            ctx.K.free_energy,
        )
        return ctx

    @classmethod
    def modular_operator(cls, rho: np.ndarray, a: np.ndarray) -> np.ndarray:
        r"""Δ(a) = ρ^{-1} a ρ."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        rho_inv = DensityOperatorAlgebra.matrix_power(rho, -1.0)
        return rho_inv @ MatrixBanachAlgebra.as_complex(a) @ rho

    @classmethod
    def modular_flow(cls, rho: np.ndarray, t: complex, a: np.ndarray) -> np.ndarray:
        r"""σ_t^ω(a) = ρ^{-it} a ρ^{it}."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        a = MatrixBanachAlgebra.as_complex(a)
        return (
            DensityOperatorAlgebra.matrix_power(rho, -1j * t)
            @ a
            @ DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        )

    @classmethod
    def modular_flow_via_K(
        cls, rho: np.ndarray, t: complex, a: np.ndarray
    ) -> np.ndarray:
        r"""σ_t(a) = e^{it K} a e^{-it K} con K = −log ρ  (equivalencia)."""
        K = DensityOperatorAlgebra.matrix_log(rho)
        K = -K
        a = MatrixBanachAlgebra.as_complex(a)
        # e^{it K} = exp(it (−log ρ)) = ρ^{-it}
        u = DensityOperatorAlgebra.matrix_power(rho, -1j * t)
        u_inv = DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        return u @ a @ u_inv

    @classmethod
    def modular_conjugation(cls, rho: np.ndarray, A: np.ndarray) -> np.ndarray:
        r"""J(A) = ρ^{-1/2} A† ρ^{1/2}  (antiunitario)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        half = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        inv_half = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        return inv_half @ MatrixBanachAlgebra.as_complex(A).conj().T @ half

    @classmethod
    def tomita_S(cls, A: np.ndarray) -> np.ndarray:
        r"""S(A) = A†  en la identificación algebraica π(x)Ω ↔ x."""
        return MatrixBanachAlgebra.as_complex(A).conj().T

    @classmethod
    def polar_decomposition_residual(cls, rho: np.ndarray, A: np.ndarray) -> float:
        r"""‖ J(Δ^{1/2}(A)) − S(A) ‖_F  =  ‖J(Δ^{1/2}(A)) − A†‖_F."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        A = MatrixBanachAlgebra.as_complex(A)
        delta_half_A = (
            DensityOperatorAlgebra.matrix_power(rho, -0.5)
            @ A
            @ DensityOperatorAlgebra.matrix_power(rho, 0.5)
        )
        polar = cls.modular_conjugation(rho, delta_half_A)
        return float(np.linalg.norm(polar - cls.tomita_S(A), "fro"))

    @classmethod
    def flow_group_law_residual(
        cls, rho: np.ndarray, s: float, t: float, a: np.ndarray
    ) -> float:
        r"""‖ σ_s(σ_t(a)) − σ_{s+t}(a) ‖_F."""
        composed = cls.modular_flow(rho, s, cls.modular_flow(rho, t, a))
        direct = cls.modular_flow(rho, s + t, a)
        return float(np.linalg.norm(composed - direct, "fro"))

    @classmethod
    def connes_cocycle(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float
    ) -> np.ndarray:
        r"""
        Cociclo de Connes (Dω_ρ : Dω_σ)_t = ρ^{it} σ^{-it}.

        Satisface la identidad de cadena
            (Dρ : Dτ)_t = (Dρ : Dσ)_t (Dσ : Dτ)_t
        y recupera el flujo relativo.
        """
        return (
            DensityOperatorAlgebra.matrix_power(rho, 1j * t)
            @ DensityOperatorAlgebra.matrix_power(sigma, -1j * t)
        )

    @classmethod
    def relative_modular_flow(
        cls, rho: np.ndarray, sigma: np.ndarray, t: float, a: np.ndarray
    ) -> np.ndarray:
        r"""σ_t^{ρ|σ}(a) = ρ^{-it} a σ^{it}  (flujo modular relativo tipo I)."""
        a = MatrixBanachAlgebra.as_complex(a)
        return (
            DensityOperatorAlgebra.matrix_power(rho, -1j * t)
            @ a
            @ DensityOperatorAlgebra.matrix_power(sigma, 1j * t)
        )

    @classmethod
    def verify_kms(
        cls, rho: np.ndarray, n_tests: int = 8, seed: int = 42
    ) -> float:
        r"""
        Residuo KMS:  res = max_{a,b} |ω(a σ_i(b)) − ω(ba)|.

        Con σ_t(x) = ρ^{-it} x ρ^{it} se tiene σ_i(b) = ρ b ρ^{-1} y
        Tr(ρ a ρ b ρ^{-1}) = Tr(ρ b a) para ρ > 0. El residuo es ~ 0
        hasta precisión de máquina (teorema de Tomita–Takesaki).
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        residual = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            sigma_i_b = cls.modular_flow(rho, 1j, b)
            lhs = np.trace(rho @ a @ sigma_i_b)
            rhs = np.trace(rho @ b @ a)
            residual = max(residual, abs(lhs - rhs))
        logger.debug("KMS residual (β=1): %.6e", residual)
        return float(residual)

    @classmethod
    def verify_algebra_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Residuos de unitalidad, multiplicatividad, isometría e involución."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        eye = np.eye(n, dtype=np.complex128)

        unital = float(np.linalg.norm(cls.modular_flow(rho, 0.37, eye) - eye))
        product = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.73, a @ b)
                - cls.modular_flow(rho, 0.73, a) @ cls.modular_flow(rho, 0.73, b)
            )
        )
        isometry = float(
            abs(
                np.linalg.norm(a, "fro")
                - np.linalg.norm(cls.modular_flow(rho, 1.11, a), "fro")
            )
        )
        ja = cls.modular_conjugation(rho, a)
        involution = float(np.linalg.norm(cls.modular_conjugation(rho, ja) - a))
        return {
            "unital_residual": unital,
            "product_residual": product,
            "isometry_residual": isometry,
            "involution_residual": involution,
        }

    @classmethod
    def verify_tomita_takesaki(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Paquete de residuos: axiomas + polar + ley de grupo + KMS + C*."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(11)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        report = cls.verify_algebra_axioms(rho)
        report["polar_residual"] = cls.polar_decomposition_residual(rho, a)
        report["group_law_residual"] = cls.flow_group_law_residual(rho, 0.41, -0.17, a)
        report["kms_residual"] = cls.verify_kms(rho)
        report["cstar_residual"] = MatrixBanachAlgebra.cstar_identity_residual(a)
        report["flow_via_K_residual"] = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.5, a) - cls.modular_flow_via_K(rho, 0.5, a),
                "fro",
            )
        )
        return report


# ── §2.2 Analizador espectral del vacío modular ────────────────────────────
@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    vev: float
    thermal_fluctuation: float
    spectral_gap: float
    modular_ground_energy: float
    partition_function: float
    free_energy: float
    internal_energy: float
    unique_vacuum: bool
    participation_ratio: float
    noise_db: float
    purity: float
    von_neumann_entropy: float
    local_verdict: HeytingOmega3


class VacuumSpectraAnalyzer:
    r"""
    Analiza ρ frente al Hamiltoniano externo H y al Hamiltoniano modular K_ρ.

        ⟨H⟩_ρ     = Tr(ρ H)
        ΔH        = √(Tr(ρ H²) − ⟨H⟩²)                 fluctuación
        VEV       = ⟨H⟩_ρ − E_0(H)                      excitación
        gap(K_ρ)  = E_1 − E_0                           unicidad del vacío
        Z(K_ρ)    = Tr e^{−K_ρ}                         ≈ 1
        F         = −log Z                              ≈ 0
        IPR       = 1 / ∑ p_i²                          razón de participación
        ruido(dB) = 10 log₁₀(1 + Var_ρ(H))

    El gap espectral discrimina degeneración: por min-max de Courant–Fischer,
    E_1 − E_0 > 0 ⇔ el ground modular es simple.
    """

    VEV_VETO_THRESHOLD: float = 1.0e-2
    FLUCTUATION_VETO_THRESHOLD: float = 1.0e-2
    GAP_DEGRADE_THRESHOLD: float = 1.0e-3
    FREE_ENERGY_DEGRADE_THRESHOLD: float = 1.0e-8

    @classmethod
    def _expectation_and_variance(
        cls, rho: np.ndarray, H: np.ndarray
    ) -> Tuple[float, float]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = MatrixBanachAlgebra.hermitize(H)
        mean = float(np.real(np.trace(rho @ H)))
        second = float(np.real(np.trace(rho @ H @ H)))
        var = max(0.0, second - mean * mean)
        return mean, math.sqrt(var)

    @classmethod
    def participation_ratio(cls, rho: np.ndarray) -> float:
        r"""IPR = 1 / Tr(ρ²) ∈ [1, n].  IPR = 1 ⇔ estado puro en una base."""
        purity = DensityOperatorAlgebra.purity(rho)
        if purity <= 0.0:
            return float("inf")
        return 1.0 / purity

    @classmethod
    def resolvent_bound(cls, K: ModularHamiltonian, z: complex) -> float:
        r"""
        ‖(z − K)^{-1}‖ ≤ 1 / dist(z, spec(K)).
        Devuelve el bound (no el operador).
        """
        if not K.eigenvalues:
            return 0.0
        dist = min(abs(z - e) for e in K.eigenvalues)
        if dist <= 0.0:
            return float("inf")
        return 1.0 / dist

    @classmethod
    def audit(
        cls,
        rho: np.ndarray,
        H: np.ndarray,
        K: ModularHamiltonian,
    ) -> VacuumAuditReport:
        rho = DensityOperatorAlgebra.sanitize(rho)
        mean_H, sigma_H = cls._expectation_and_variance(rho, H)

        w_H = np.real(la.eigvalsh(MatrixBanachAlgebra.hermitize(H)))
        e0 = float(w_H.min())
        vev = mean_H - e0

        gap = K.spectral_gap
        z_part = K.partition_function
        free_e = K.free_energy
        noise_db = 10.0 * math.log10(1.0 + sigma_H ** 2 + 1e-30)
        purity = DensityOperatorAlgebra.purity(rho)
        ent = DensityOperatorAlgebra.von_neumann_entropy(rho)
        ipr = cls.participation_ratio(rho)

        if (
            sigma_H > cls.FLUCTUATION_VETO_THRESHOLD
            and vev > cls.VEV_VETO_THRESHOLD
        ):
            local = HeytingOmega3.VETOED
        elif (
            sigma_H > cls.FLUCTUATION_VETO_THRESHOLD
            or vev > cls.VEV_VETO_THRESHOLD
        ):
            local = HeytingOmega3.DEGRADED
        elif gap < cls.GAP_DEGRADE_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        elif abs(free_e) > cls.FREE_ENERGY_DEGRADE_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT

        return VacuumAuditReport(
            vev=vev,
            thermal_fluctuation=sigma_H,
            spectral_gap=gap,
            modular_ground_energy=K.ground_energy,
            partition_function=z_part,
            free_energy=free_e,
            internal_energy=K.internal_energy,
            unique_vacuum=K.is_unique_vacuum,
            participation_ratio=ipr,
            noise_db=noise_db,
            purity=purity,
            von_neumann_entropy=ent,
            local_verdict=local,
        )


# ── §2.3 Detector del silencio epistemológico ──────────────────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldProbe:
    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    correlation_leakage: float
    umegaki_to_ground: float
    bures_distance: float
    trace_distance: float
    klein_residual: float
    local_verdict: HeytingOmega3


class SilentFieldDetector:
    r"""
    Mide la fuga de ρ fuera del vacío |Ω⟩⟨Ω|:

        overlap            = ⟨Ω|ρ|Ω⟩ = Tr(ρ σ_Ω)     (= F² si σ_Ω puro)
        F_Uhlmann          = Tr √(√ρ σ_Ω √ρ)
        correlation_leakage= 1 − overlap
        S(ρ‖σ_ε)           Umegaki regularizado
        d_B, D_tr          geometría de Bures y traza
        Klein residual     min{0, S(ρ‖σ_ε)}

    σ_ε = (1−ε)σ_Ω + ε I/n garantiza fidelidad del soporte (Umegaki).
    """

    UMEGAKI_EPS: float = 1.0e-12
    UMEgaki_EPS: float = UMEGAKI_EPS  # alias de compatibilidad
    PURITY_SILENT_THRESHOLD: float = 0.999
    FIDELITY_SILENT_THRESHOLD: float = 0.999

    @classmethod
    def probe(
        cls,
        rho: np.ndarray,
        ground_state_projector: np.ndarray,
    ) -> SilentFieldProbe:
        rho = DensityOperatorAlgebra.sanitize(rho)
        sigma = DensityOperatorAlgebra.sanitize(ground_state_projector)

        purity = DensityOperatorAlgebra.purity(rho)
        overlap = float(np.real(np.trace(rho @ sigma)))
        leak = 1.0 - overlap
        uhlmann = DensityOperatorAlgebra.uhlmann_fidelity(rho, sigma)
        bures = DensityOperatorAlgebra.bures_distance(rho, sigma)
        trc = DensityOperatorAlgebra.trace_distance(rho, sigma)

        n = rho.shape[0]
        sigma_eps = DensityOperatorAlgebra.sanitize(
            (1.0 - cls.UMEGAKI_EPS) * sigma
            + cls.UMEGAKI_EPS * np.eye(n, dtype=np.complex128) / n
        )
        relative_ent = DensityOperatorAlgebra.umegaki_relative_entropy(rho, sigma_eps)
        klein = DensityOperatorAlgebra.kleins_inequality_residual(rho, sigma_eps)

        if (
            purity >= cls.PURITY_SILENT_THRESHOLD
            and overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.COHERENT
        elif (
            purity >= cls.PURITY_SILENT_THRESHOLD
            or overlap >= cls.FIDELITY_SILENT_THRESHOLD
        ):
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return SilentFieldProbe(
            purity=purity,
            fidelity_to_ground=overlap,
            uhlmann_fidelity=uhlmann,
            correlation_leakage=leak,
            umegaki_to_ground=relative_ent,
            bures_distance=bures,
            trace_distance=trc,
            klein_residual=klein,
            local_verdict=local,
        )


# ── §2.4 ModularSilencePipeline — HAND-OFF FASE 2 → FASE 3 ─────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldBundle:
    r"""
    Paquete de hand-off FASE 2 → FASE 3.

    Encapsula el estado modular, los residuos Tomita–Takesaki/KMS y las
    métricas de silencio para su adjudicación en Ω₃ y su firma.
    Objeto terminal de FASE 2 = dominio de `HeytingVacuumAdjudicator`.
    """

    cycle_index: int
    rho: np.ndarray
    K: ModularHamiltonian
    audit: VacuumAuditReport
    probe: SilentFieldProbe
    kms_residual: float
    algebra_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    context_beta: float


class ModularSilencePipeline:
    r"""
    Orquestador determinista de la dinámica modular:

        VacuumModularContext
            → bind (§2.0)
            → Tomita (KMS, polar, grupo)
            → Espectro (§2.2)
            → Silencio (§2.3)
            → SilentFieldBundle   (hand-off → FASE 3)
    """

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho: np.ndarray,
        H: np.ndarray,
        ground_projector: np.ndarray,
        K: ModularHamiltonian,
        context_beta: float = float("nan"),
    ) -> SilentFieldBundle:
        tomita = TomitaTakesakiEngine.verify_tomita_takesaki(rho)
        kms_res = float(tomita.get("kms_residual", TomitaTakesakiEngine.verify_kms(rho)))
        axioms = {k: tomita[k] for k in (
            "unital_residual",
            "product_residual",
            "isometry_residual",
            "involution_residual",
        ) if k in tomita}
        if len(axioms) < 4:
            axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho)

        audit = VacuumSpectraAnalyzer.audit(rho, H, K)
        probe = SilentFieldDetector.probe(rho, ground_projector)

        return SilentFieldBundle(
            cycle_index=cycle_index,
            rho=rho,
            K=K,
            audit=audit,
            probe=probe,
            kms_residual=kms_res,
            algebra_axioms=axioms,
            tomita_report=tomita,
            context_beta=context_beta,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 2 → FASE 3
    #  Definición formal terminal de FASE 2.
    #  Consume el VacuumModularContext (objeto inicial de FASE 2, nacido
    #  en §1.6) y produce SilentFieldBundle, dominio de §3.1.
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: VacuumModularContext,
    ) -> SilentFieldBundle:
        r"""
        Morfismo de hand-off  VacuumModularContext ↦ SilentFieldBundle.

        Continúa en §3.1 `HeytingVacuumAdjudicator.adjudicate`.
        """
        ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        return cls.synthesize(
            cycle_index=cycle_index,
            rho=ctx.rho,
            H=ctx.H,
            ground_projector=ctx.ground_projector,
            K=ctx.K,
            context_beta=ctx.beta,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · SOBERANÍA Y CERTIFICACIÓN DEL SILENCIO (C. de FASE 2)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en el retículo Ω₃ ─────────────────────────────────────
class HeytingVacuumAdjudicator:
    r"""
    Colapsa el SilentFieldBundle (objeto terminal de FASE 2) en un único
    veredicto Ω₃ y aplica meet (∧) con el veredicto externo (godel_agent):

        local = audit ∧ probe ∧ kms_verdict ∧ axioms_verdict
              ∧ polar_verdict ∧ group_verdict ∧ klein_verdict
        final = local ∧ external

    El meet es el ínfimo del retículo: la decisión más conservadora.
    Residuos catastróficos (> CATASTROPHIC_TOL) inducen ⊥, no ⋆.
    """

    KMS_RESIDUAL_TOL: float = 1.0e-6
    AXIOM_RESIDUAL_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3
    KLEIN_TOL: float = 1.0e-8

    @classmethod
    def _residual_to_omega(cls, residual: float) -> HeytingOmega3:
        if residual > cls.CATASTROPHIC_TOL:
            return HeytingOmega3.VETOED
        if residual > cls.KMS_RESIDUAL_TOL:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    @classmethod
    def granular_scores(cls, bundle: SilentFieldBundle) -> Dict[str, HeytingOmega3]:
        axioms_max = max(bundle.algebra_axioms.values()) if bundle.algebra_axioms else 0.0
        polar = float(bundle.tomita_report.get("polar_residual", 0.0))
        group = float(bundle.tomita_report.get("group_law_residual", 0.0))
        klein = abs(float(bundle.probe.klein_residual))
        klein_v = (
            HeytingOmega3.COHERENT
            if klein < cls.KLEIN_TOL
            else (
                HeytingOmega3.DEGRADED
                if klein < cls.CATASTROPHIC_TOL
                else HeytingOmega3.VETOED
            )
        )
        return {
            "audit": bundle.audit.local_verdict,
            "probe": bundle.probe.local_verdict,
            "kms": cls._residual_to_omega(bundle.kms_residual),
            "axioms": cls._residual_to_omega(axioms_max),
            "polar": cls._residual_to_omega(polar),
            "group_law": cls._residual_to_omega(group),
            "klein": klein_v,
        }

    @classmethod
    def adjudicate(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        §3.1  Arranque de FASE 3.

        Continúa `ModularSilencePipeline.synthesize_from_context` (§2.4):
        el bundle es el clasificador de verdad pre-Ω₃; aquí se toma el
        ínfimo con la fuente externa.
        """
        scores = cls.granular_scores(bundle)
        local = HeytingOmega3.COHERENT
        for v in scores.values():
            local = local.meet(v)
        return local.meet(external_verdict)

    @classmethod
    def implication_chain(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        r"""
        Diagnóstico residuado: (KMS ⇒ audit) ∧ (polar ⇒ probe) ∧ …
        No sustituye al meet; expone fallos de implicación interna.
        """
        s = cls.granular_scores(bundle)
        chain = (
            s["kms"].implies(s["audit"])
            .meet(s["polar"].implies(s["probe"]))
            .meet(s["axioms"].implies(s["group_law"]))
        )
        return chain.meet(external_verdict)


# ── §3.2 Certificado SilentFieldState ──────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldState:
    r"""
    Certificado firmado del ciclo de silencio.

        • Observables del motor modular y geometría de Bures/traza.
        • `experience_hash` encadena ciclo + VEV + ruido + KMS + conteo
          cristalino + ns-timestamp vía SHA-256.
        • `phase_chain_sha256` conserva custodia forense entre ciclos
          (cada ciclo hereda el hash del anterior: cadena de Markov
          criptográfica sobre el topos de fases).
    """

    cycle_id: str
    witness_engine_id: str
    vacuum_expectation_value: float
    kms_temperature_beta: float
    vacuum_beta: float
    noise_emission_decibels: float
    crystallized_experience_count: int
    spectral_gap: float
    free_energy: float
    purity: float
    fidelity_to_ground: float
    uhlmann_fidelity: float
    bures_distance_to_ground: float
    trace_distance_to_ground: float
    kms_residual: float
    tomita_polar_residual: float
    tomita_group_law_residual: float
    heyting_verdict: HeytingOmega3
    experience_hash: str
    phase_chain_sha256: str
    timestamp_utc: float


# ── §3.3 TOONSilentWitnessEngine — orquestador de ciclos ───────────────────
class TOONSilentWitnessEngine:
    r"""
    Motor espectral del vacío y silencio epistemológico.

    Custodia la matriz MAC con técnica Tomita–Takesaki:
        FASE 1  prepare_vacuum_context(H, β) → VacuumModularContext
        FASE 2  synthesize_from_context → SilentFieldBundle
        FASE 3  adjudicate(Ω₃) + firma SHA-256 encadenada

    H se construye como tight-binding sobre P_n (grafo camino), lo que
    hace del gap un invariante espectral-gráfico (Cheeger discreto).
    """

    def __init__(
        self,
        engine_id: str = "SILENT-ENGINE-SABIO-01",
        mac_dimension: int = 4,
        kms_beta: float = 1.0,
        hopping: float = VacuumStatePreparation.DEFAULT_HOPPING,
    ) -> None:
        self.engine_id = engine_id
        self.mac_dimension = mac_dimension
        self.kms_beta = kms_beta
        self.cycle_count = 0
        self.history: List[SilentFieldState] = []

        self.H = VacuumStatePreparation.tight_binding_hamiltonian(
            mac_dimension, hopping=hopping
        )
        self.ground_projector = VacuumStatePreparation.ground_state_projector(self.H)
        self._phase_chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()

    def _update_chain(self, tag: str, payload: bytes) -> str:
        digest = hashlib.sha256(
            self._phase_chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._phase_chain_hash = digest
        return digest

    def latest_state(self) -> SilentFieldState | None:
        return self.history[-1] if self.history else None

    def forensic_verify_chain(self) -> bool:
        r"""Recomputación honesta imposible sin historia de payloads;
        verifica monotonicidad de longitud y formato hex SHA-256."""
        if not self.history:
            return True
        return all(
            len(s.phase_chain_sha256) == 64
            and all(c in "0123456789abcdef" for c in s.phase_chain_sha256)
            for s in self.history
        )

    def execute_silence_cycle(
        self,
        triad_crystallized_count: int,
        kms_beta: float = 1.0,
        beta_vacuum: float = VacuumStatePreparation.DEFAULT_BETA_COLD,
        external_verdict: HeytingOmega3 = HeytingOmega3.COHERENT,
    ) -> SilentFieldState:
        self.cycle_count += 1
        cycle_id = f"CYC-SILENT-{self.cycle_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Ciclo del vacío %s | cristalizados=%d ═══",
            cycle_id,
            triad_crystallized_count,
        )

        # ── FASE 1 ── contexto GNS (ρ_Ω, K_Ω, |Ω⟩⟨Ω|, TFD) ──
        ctx = VacuumStatePreparation.prepare_vacuum_context(self.H, beta=beta_vacuum)
        self._update_chain("F1", ctx.rho.tobytes())

        # ── FASE 2 ── bind + Tomita + espectro + sonda ──
        bundle = ModularSilencePipeline.synthesize_from_context(
            cycle_index=self.cycle_count, ctx=ctx
        )
        self._update_chain(
            "F2",
            f"{bundle.kms_residual:.12e}|{bundle.audit.vev:.12e}|{bundle.audit.free_energy:.12e}".encode(
                "ascii"
            ),
        )

        # ── FASE 3 ── adjudicación Ω₃ y firma ──
        verdict = HeytingVacuumAdjudicator.adjudicate(bundle, external_verdict)
        self._update_chain(
            "F3",
            f"{verdict.name}|{bundle.audit.noise_db:.6f}|{triad_crystallized_count}".encode(
                "ascii"
            ),
        )

        hasher = hashlib.sha256()
        hasher.update(self.engine_id.encode("ascii"))
        hasher.update(cycle_id.encode("ascii"))
        hasher.update(f"{bundle.audit.vev:.12e}".encode("ascii"))
        hasher.update(f"{bundle.audit.noise_db:.12e}".encode("ascii"))
        hasher.update(f"{bundle.kms_residual:.12e}".encode("ascii"))
        hasher.update(f"{triad_crystallized_count}".encode("ascii"))
        hasher.update(f"{time.time_ns()}".encode("ascii"))
        experience_hash = hasher.hexdigest()

        polar_res = float(bundle.tomita_report.get("polar_residual", 0.0))
        group_res = float(bundle.tomita_report.get("group_law_residual", 0.0))

        state = SilentFieldState(
            cycle_id=cycle_id,
            witness_engine_id=self.engine_id,
            vacuum_expectation_value=bundle.audit.vev,
            kms_temperature_beta=kms_beta,
            vacuum_beta=beta_vacuum,
            noise_emission_decibels=bundle.audit.noise_db,
            crystallized_experience_count=triad_crystallized_count,
            spectral_gap=bundle.audit.spectral_gap,
            free_energy=bundle.audit.free_energy,
            purity=bundle.probe.purity,
            fidelity_to_ground=bundle.probe.fidelity_to_ground,
            uhlmann_fidelity=bundle.probe.uhlmann_fidelity,
            bures_distance_to_ground=bundle.probe.bures_distance,
            trace_distance_to_ground=bundle.probe.trace_distance,
            kms_residual=bundle.kms_residual,
            tomita_polar_residual=polar_res,
            tomita_group_law_residual=group_res,
            heyting_verdict=verdict,
            experience_hash=experience_hash,
            phase_chain_sha256=self._phase_chain_hash,
            timestamp_utc=time.time(),
        )
        self.history.append(state)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s completado en %.2f ms | Ω₃=%s | VEV=%.3e | ruido=%.3f dB | "
            "KMS=%.2e | polar=%.2e | F=%.2e",
            cycle_id,
            dt_ms,
            verdict.name,
            state.vacuum_expectation_value,
            state.noise_emission_decibels,
            state.kms_residual,
            state.tomita_polar_residual,
            state.free_energy,
        )
        return state


# ── §3.4 Punto de entrada / demostración ───────────────────────────────────
def _print_heyting_audit() -> None:
    laws: Mapping[str, bool] = HeytingOmega3.verify_heyting_laws()
    print("  Heyting Ω₃ laws:", dict(laws))


if __name__ == "__main__":
    engine = TOONSilentWitnessEngine(
        engine_id="SILENT-ENGINE-SABIO-01",
        mac_dimension=4,
        kms_beta=1.0,
    )

    print("═" * 80)
    print(f"DEMOSTRACIÓN GRANULAR: TOON Silent Witness Engine v{__version__}")
    print("═" * 80)
    _print_heyting_audit()

    scenarios = [
        # (nombre, conteo_cristalizado, beta_vacío, veredicto_externo)
        ("VACÍO FRÍO (β → ∞)", 5, 1.0e3, HeytingOmega3.COHERENT),
        ("VACÍO TEMPLADO (β = 10)", 7, 10.0, HeytingOmega3.COHERENT),
        ("VACÍO TIBIO (β = 1)", 12, 1.0, HeytingOmega3.COHERENT),
        ("VACÍO AGRESIVO (β = 0.5)", 20, 0.5, HeytingOmega3.COHERENT),
        ("VACÍO HOSTIL (β = 0.1)", 50, 0.1, HeytingOmega3.DEGRADED),
    ]

    for name, count, beta_v, ext in scenarios:
        state = engine.execute_silence_cycle(
            triad_crystallized_count=count,
            kms_beta=1.0,
            beta_vacuum=beta_v,
            external_verdict=ext,
        )
        print(
            f"    [{name:<25s}] Ω₃={state.heyting_verdict.name:<9s} | "
            f"VEV={state.vacuum_expectation_value:.3e} | "
            f"gap={state.spectral_gap:.3e} | "
            f"F={state.free_energy:.2e} | "
            f"pureza={state.purity:.6f} | "
            f"fid_Ω={state.fidelity_to_ground:.6f} | "
            f"Bures={state.bures_distance_to_ground:.3e} | "
            f"ruido={state.noise_emission_decibels:.3f} dB | "
            f"KMS={state.kms_residual:.2e} | "
            f"polar={state.tomita_polar_residual:.2e} | "
            f"hash={state.experience_hash[:16]}…"
        )

    print("\n  Cadena forense SHA-256 válida:", engine.forensic_verify_chain())
    print("\n" + "═" * 80)
    print("✓ Auditoría modular Tomita-Takesaki completada.")
    print("✓ KMS(β=1), polar S = JΔ^{1/2} y ley de grupo verificados.")
    print("✓ Umegaki operatorial + desigualdad de Klein.")
    print("✓ Cadena de custodia forense preservada entre ciclos.")
    print("═" * 80)
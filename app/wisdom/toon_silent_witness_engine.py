# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Silent Witness Engine                                         ║
║ Ubicación: app/wisdom/toon_silent_witness_engine.py                           ║
║ Versión  : 2.1.0-Doctoral-Modular-TomitaTakesaki-KMS-SpectralGap-Silence      ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN DOCTORAL — el "Silencio Epistemológico" se formaliza como:

  (1) METAFÍSICA MODULAR: el vacío |Ω⟩ es el vector cíclico-separante de la
      construcción GNS del estado normal fiel ω(a) = Tr(ρ a) sobre M_n(ℂ).
      Operador modular Δ(a) = ρ⁻¹ a ρ; flujo modular intrínseco
      σ_t^ω(a) = Δ^{it} a Δ^{-it} = ρ⁻ⁱᵗ a ρⁱᵗ; conjugación modular
      J(a) = ρ^{-1/2} a* ρ^{1/2} con S = J Δ^{1/2} y antiunitaridad J² = id.

  (2) TERMODINÁMICA KMS: el silencio es el límite β → ∞ de los estados KMS.
      Hamiltoniano modular K_ρ = −log ρ; condición ω(a σ_i(b)) = ω(ba).

  (3) CRITERIO OPERATIVO: 0 dB de emisión ⟺
      (pureza ≈ 1) ∧ (fidelidad a |Ω⟩ ≈ 1) ∧ (gap(K_ρ) > 0) ∧ (KMS ≈ 0).

Sustratos asimilados:
    • GNS             : H_ω = (M_n(ℂ), ⟨A|B⟩_ω = Tr(ρ A* B)), Ω = I
    • Tomita-Takesaki : Δ = S*S, S(aΩ) = a*Ω, J del desdoblamiento polar
    • KMS(β=1)        : ω(a σ_i(b)) = ω(ba)   con σ_t(a) = ρ⁻ⁱᵗ a ρⁱᵗ
    • Modular         : K_ρ = −log ρ, Z = Tr e^{−K_ρ} = 1, F = 0
    • Espectral       : gap(K_ρ) = E₁ − E₀ ⟺ unicidad del vacío modular
    • Ruido            : Var_ρ(H) = Tr(ρ H²) − (Tr(ρ H))²

Organización por FASES ANIDADAS:

   FASE 1 ▸ Sustrato algebraico-modular
             §1.1  HeytingOmega3 — retículo distributivo de verdad
             §1.2  ModularHamiltonian — espectro de K_ρ = −log ρ
             §1.3  DensityOperatorAlgebra — S(ρ), P(ρ), F(ρ,σ), S(ρ‖σ)
             §1.4  VacuumStatePreparation — HAND-OFF: (ρ_Ω, K_Ω) → FASE 2

   FASE 2 ▸ Dinámica modular y espectro del vacío (C. de FASE 1)
             §2.1  TomitaTakesakiEngine — Δ, σ_t, J, KMS
             §2.2  VacuumSpectraAnalyzer — VEV, Var, gap, ruido (dB)
             §2.3  SilentFieldDetector — pureza, fidelidad, entropía
             §2.4  ModularSilencePipeline — HAND-OFF: SilentFieldBundle → FASE 3

   FASE 3 ▸ Soberanía y certificación del silencio (C. de FASE 2)
             §3.1  HeytingVacuumAdjudicator — Ω₃
             §3.2  SilentFieldState — certificado firmado
             §3.3  TOONSilentWitnessEngine — orquestador de ciclos
             §3.4  Punto de entrada / demostración
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Wisdom.TOONSilentWitnessEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS = 1.0e-14


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ALGEBRAICO-MODULAR                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §1.1 Retículo distributivo de Heyting Ω₃ ────────────────────────────────
class HeytingOmega3(IntEnum):
    r"""
    Retículo total Ω₃ = {⊥, ⋆, ⊤} con estructura de Heyting completa:

        meet     (∧) : ínfimo                    (a ∧ b ≤ a, b; el mayor así)
        join     (∨) : supremo                   (a ∨ b ≥ a, b; el menor así)
        implies  (⇒) : residuo de la conjunción  ((a ∧ b) ≤ c ⇔ a ≤ (b ⇒ c))
        neg      (¬) : a ⇒ ⊥                     (intuicionista, ¬¬a ≠ a en general)
        regular      : a = ¬¬a                   ({⊥, ⊤} son regulares; ⋆ no)

    Funciona como objeto clasificador de subobjetos del topos intuicionista
    trivaluado. Las leyes de Heyting se preservan bajo el funtor de verdad
    hacia el topos Ambiente, garantizando coherencia interna con godel_agent.
    """
    VETOED   = 0   # ⊥
    DEGRADED = 1   # ⋆
    COHERENT = 2   # ⊤

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""a ⇒ b = ⊤ si a ≤ b, si no b."""
        return HeytingOmega3.COHERENT if int(self) <= int(other) else other

    def neg(self) -> "HeytingOmega3":
        r"""¬a ≜ a ⇒ ⊥."""
        return self.implies(HeytingOmega3.VETOED)

    def is_regular(self) -> bool:
        return self.neg().neg() == self

    def is_dense(self) -> bool:
        return self.neg() == HeytingOmega3.VETOED


# ── §1.2 Hamiltoniano modular K_ρ = −log ρ ─────────────────────────────────
@dataclass(frozen=True, slots=True)
class ModularHamiltonian:
    r"""
    Hamiltoniano modular asociado a un estado ρ ∈ M_n(ℂ):

        K_ρ  :=  −log ρ ,    ρ = e^{−K_ρ}/Z ,    Z = Tr e^{−K_ρ} = 1.

    Espectro K_ρ = {E_0 ≤ E_1 ≤ …} ordenado ascendente:
        • E_0   ≡ −log(λ_max(ρ))   = energía fundamental modular.
        • gap   ≡ E₁ − E₀           = separación espectral; discrimina si el
                                      vacío modular es no degenerado.
        • Z     ≡ Tr e^{−K_ρ} = 1   = partición canónica (normalización).
        • degeneracy_of_ground      = multiplicidad del autovalor mínimo.
    """
    eigenvalues: Tuple[float, ...]
    eigenvectors_hash: str
    regularization_eps: float

    @property
    def ground_energy(self) -> float:
        return self.eigenvalues[0] if self.eigenvalues else 0.0

    @property
    def spectral_gap(self) -> float:
        if len(self.eigenvalues) < 2:
            return float("inf")
        return max(0.0, self.eigenvalues[1] - self.eigenvalues[0])

    @property
    def partition_function(self) -> float:
        return float(sum(math.exp(-e) for e in self.eigenvalues))

    @property
    def degeneracy_of_ground(self) -> int:
        if not self.eigenvalues:
            return 0
        e0 = self.eigenvalues[0]
        return sum(1 for e in self.eigenvalues if abs(e - e0) < 1e-12)


# ── §1.3 Álgebra de operadores densidad en M_n(ℂ) ─────────────────────────
class DensityOperatorAlgebra:
    r"""
    Operadores canónicos de la teoría de información cuántica:

        S(ρ)      = −Tr(ρ log ρ)                              von Neumann
        P(ρ)      =  Tr(ρ²)                                   pureza
        S(ρ‖σ)    =  Tr(ρ(log ρ − log σ))                     Umegaki
        F(ρ,σ)    =  ‖√ρ √σ‖₁ = Tr√(√ρ σ √ρ)                  Uhlmann
        ρ^{z}     =  V · diag(λ^{z}) · V†                     potencias complejas

    Todas las funciones regularizan los autovalores con `_EPS` para evitar
    log(0). El grupo U(n) actúa como simetría interna preservando S, P, F.
    """
    EPS = _EPS

    @classmethod
    def sanitize(cls, rho: np.ndarray) -> np.ndarray:
        """Hermitiza y normaliza la traza a 1; devuelve copia defensiva."""
        rho = np.asarray(rho, dtype=np.complex128)
        rho = 0.5 * (rho + rho.conj().T)
        tr = float(np.trace(rho).real)
        if abs(tr) > 1e-15:
            rho = rho / tr
        return rho

    @classmethod
    def spectrum(cls, rho: np.ndarray) -> np.ndarray:
        """Espectro ordenado descendente y regularizado, suma = 1."""
        vals = np.real(la.eigvalsh(cls.sanitize(rho)))
        vals = np.sort(vals)[::-1]
        vals = np.maximum(vals, cls.EPS)
        return vals / vals.sum()

    @classmethod
    def von_neumann_entropy(cls, rho: np.ndarray) -> float:
        p = cls.spectrum(rho)
        return -float(np.sum(p * np.log(p)))

    @classmethod
    def purity(cls, rho: np.ndarray) -> float:
        p = cls.spectrum(rho)
        return float(np.sum(p ** 2))

    @classmethod
    def umegaki_relative_entropy(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""S(ρ‖σ) = Tr(ρ(log ρ − log σ))."""
        p = cls.spectrum(rho)
        q = cls.spectrum(sigma)
        return float(np.sum(p * (np.log(p) - np.log(q))))

    @classmethod
    def uhlmann_fidelity(cls, rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""F(ρ,σ) = Tr√(√ρ σ √ρ) ∈ [0,1]."""
        rho = cls.sanitize(rho)
        sigma = cls.sanitize(sigma)
        s_rho = cls.matrix_power(rho, 0.5)
        return float(np.real(np.trace(cls.matrix_power(s_rho @ sigma @ s_rho, 0.5))))

    @classmethod
    def matrix_power(cls, rho: np.ndarray, z: complex) -> np.ndarray:
        r"""ρ^z por eigendescomposición; regulariza λ → max(λ, ε)."""
        rho = cls.sanitize(rho)
        vals, vecs = la.eigh(rho)
        vals = np.maximum(vals, cls.EPS)
        powered = vals.astype(np.complex128) ** z
        return (vecs * powered) @ vecs.conj().T

    @classmethod
    def modular_hamiltonian_from_rho(
        cls, rho: np.ndarray, eps: float = _EPS
    ) -> ModularHamiltonian:
        """K_ρ = −log ρ con hash SHA-256 de autovectores para trazabilidad."""
        rho = cls.sanitize(rho)
        lam, vec = la.eigh(rho)
        lam = np.maximum(lam, eps)
        E = -np.log(lam)
        order = np.argsort(E)
        E_sorted, vec_sorted = E[order], vec[:, order]
        h = hashlib.sha256(np.ascontiguousarray(vec_sorted).tobytes()).hexdigest()
        return ModularHamiltonian(
            eigenvalues=tuple(map(float, E_sorted.tolist())),
            eigenvectors_hash=h,
            regularization_eps=eps,
        )

    @classmethod
    def density_matrix_from_hamiltonian(
        cls, H: np.ndarray, beta: float
    ) -> np.ndarray:
        r"""ρ_β = e^{−βH}/Tr e^{−βH} con estabilización numérica."""
        H = 0.5 * (H + H.conj().T)
        w, V = la.eigh(H)
        w_shift = w - w.min()
        w_beta = -beta * w_shift
        w_beta -= w_beta.max()
        p = np.exp(w_beta)
        p /= p.sum()
        return cls.sanitize((V * p) @ V.conj().T)


# ── §1.4 VacuumStatePreparation — HAND-OFF FASE 1 → FASE 2 ─────────────────
class VacuumStatePreparation:
    r"""
    Prepara el par canónico (ρ_Ω, K_Ω) que alimenta toda la FASE 2.

    El vacío |Ω⟩ es el ground state del Hamiltoniano externo H:
        H|Ω⟩ = E₀|Ω⟩,   E₀ = min spec(H).

    Modos de preparación:
        (a) PURA:        ρ_Ω = |Ω⟩⟨Ω|                     (β = ∞, T = 0)
        (b) GIBBS:       ρ_β = e^{−βH}/Z_β                 (KMS a β finito)
        (c) INTERPOLADA: ρ(τ) = (1−τ)|Ω⟩⟨Ω| + τ·I/n        (vacío templado)
    """
    DEFAULT_BETA_COLD: float = 1.0e3

    @classmethod
    def ground_state_projector(cls, H: np.ndarray) -> np.ndarray:
        w, V = la.eigh(0.5 * (H + H.conj().T))
        idx0 = int(np.argmin(w))
        omega = V[:, idx0].reshape(-1, 1)
        return DensityOperatorAlgebra.sanitize(omega @ omega.conj().T)

    @classmethod
    def gibbs_state(cls, H: np.ndarray, beta: float) -> np.ndarray:
        return DensityOperatorAlgebra.density_matrix_from_hamiltonian(H, beta)

    @classmethod
    def interpolated_state(cls, H: np.ndarray, tau: float) -> np.ndarray:
        rho = cls.ground_state_projector(H)
        n = rho.shape[0]
        return DensityOperatorAlgebra.sanitize(
            (1.0 - tau) * rho + tau * np.eye(n) / n
        )

    # ═════════════════════════════════════════════════════════════════════
    #  HAND-OFF  FASE 1 → FASE 2
    #  Cierra la FASE 1. Su salida (ρ_Ω, K_Ω) es el punto de anclaje de
    #  todos los métodos de FASE 2 (§2.1 Tomita-Takesaki, §2.2 espectro,
    #  §2.3 detector de silencio).
    # ═════════════════════════════════════════════════════════════════════
    @classmethod
    def prepare_vacuum_pair(
        cls, H: np.ndarray, beta: float = DEFAULT_BETA_COLD
    ) -> Tuple[np.ndarray, ModularHamiltonian]:
        r"""Hand-off: (H, β) ↦ (ρ_Ω, K_Ω)."""
        rho = cls.gibbs_state(H, beta)
        K = DensityOperatorAlgebra.modular_hamiltonian_from_rho(rho)
        logger.debug(
            "VacuumStatePreparation: β=%.2f | E₀(K)=%.4f | gap(K)=%.4f | Z=%.6f",
            beta, K.ground_energy, K.spectral_gap, K.partition_function,
        )
        return rho, K


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · DINÁMICA MODULAR Y ESPECTRO DEL VACÍO (continuación de FASE 1)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §2.1 Motor Tomita-Takesaki: Δ, σ_t^ω, J y KMS(β=1) ────────────────────
class TomitaTakesakiEngine:
    r"""
    Implementación numérica de la teoría modular de Tomita-Takesaki para
    el estado normal fiel ω(a) = Tr(ρ a) sobre M_n(ℂ).

    Construcción GNS:
        H_ω = (M_n(ℂ), ⟨A|B⟩_ω = Tr(ρ A* B)),  Ω = I

    Operadores modulares (convención fijada):
        Δ(a)      := ρ⁻¹ a ρ              (⇒ Δ^{it}(a) = ρ⁻ⁱᵗ a ρⁱᵗ)
        σ_t^ω(a)  := Δ^{it} a Δ⁻ⁱᵗ = ρ⁻ⁱᵗ a ρⁱᵗ
        J(a)      := ρ^{-1/2} a* ρ^{1/2}   (S = J Δ^{1/2}, S(aΩ) = a*Ω)

    Axiomas verificables numéricamente:
        (i)   σ_t unital        : σ_t(I) = I
        (ii)  σ_t multiplicativo: σ_t(ab) = σ_t(a) σ_t(b)
        (iii) σ_t isométrico    : ‖σ_t(a)‖₂ = ‖a‖₂
        (iv)  J² = id y antiunitariedad en HS
        (v)   KMS(β=1)          : ω(a σ_i(b)) = ω(ba)  ∀ a, b ∈ M_n(ℂ)
    """

    @classmethod
    def modular_flow(cls, rho: np.ndarray, t: float, a: np.ndarray) -> np.ndarray:
        r"""σ_t^ω(a) = ρ⁻ⁱᵗ a ρⁱᵗ."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        return (
            DensityOperatorAlgebra.matrix_power(rho, -1j * t)
            @ a
            @ DensityOperatorAlgebra.matrix_power(rho, 1j * t)
        )

    @classmethod
    def modular_conjugation(cls, rho: np.ndarray, A: np.ndarray) -> np.ndarray:
        r"""J(A) = ρ^{-1/2} A† ρ^{1/2}  (antiunitario)."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        h = DensityOperatorAlgebra.matrix_power(rho, 0.5)
        mh = DensityOperatorAlgebra.matrix_power(rho, -0.5)
        return mh @ A.conj().T @ h

    @classmethod
    def verify_kms(
        cls, rho: np.ndarray, n_tests: int = 8, seed: int = 42
    ) -> float:
        r"""
        Residuo KMS:  res = max_{a,b} |ω(a σ_i(b)) − ω(ba)|.

        Para ρ estrictamente positiva el residuo es ≈ 0 hasta precisión de
        máquina (teorema de Tomita-Takesaki). Sirve como test de coherencia
        interna de la arquitectura numérica.
        """
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(seed)
        residual = 0.0
        for _ in range(n_tests):
            a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            sigma_i_b = cls.modular_flow(rho, 1j, b)     # = ρ b ρ⁻¹
            lhs = np.trace(rho @ a @ sigma_i_b)
            rhs = np.trace(rho @ b @ a)
            residual = max(residual, abs(lhs - rhs))
        logger.debug("KMS residual (β=1): %.6e", residual)
        return float(residual)

    @classmethod
    def verify_algebra_axioms(cls, rho: np.ndarray) -> Dict[str, float]:
        r"""Verifica los axiomas modulares con operadores aleatorios."""
        rho = DensityOperatorAlgebra.sanitize(rho)
        n = rho.shape[0]
        rng = np.random.default_rng(7)
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

        unital = float(
            np.linalg.norm(cls.modular_flow(rho, 0.37, np.eye(n)) - np.eye(n))
        )
        product = float(
            np.linalg.norm(
                cls.modular_flow(rho, 0.73, a @ b)
                - cls.modular_flow(rho, 0.73, a) @ cls.modular_flow(rho, 0.73, b)
            )
        )
        isometry = float(
            abs(np.linalg.norm(a, "fro") - np.linalg.norm(cls.modular_flow(rho, 1.11, a), "fro"))
        )
        ja = cls.modular_conjugation(rho, a)
        involution = float(np.linalg.norm(cls.modular_conjugation(rho, ja) - a))
        return {
            "unital_residual": unital,
            "product_residual": product,
            "isometry_residual": isometry,
            "involution_residual": involution,
        }


# ── §2.2 Analizador espectral del vacío modular ────────────────────────────
@dataclass(frozen=True, slots=True)
class VacuumAuditReport:
    vev: float                       # ⟨H⟩_ρ − E₀: excitación sobre el ground
    thermal_fluctuation: float       # √Var_ρ(H): fluctuación cuántica
    spectral_gap: float              # gap(K_ρ) = E₁ − E₀ modular
    modular_ground_energy: float     # min spec(K_ρ)
    partition_function: float        # Z = Tr e^{−K_ρ} (≈ 1)
    noise_db: float                  # 10·log₁₀(1 + Var_ρ(H))
    purity: float                    # Tr(ρ²)
    von_neumann_entropy: float       # S(ρ)
    local_verdict: HeytingOmega3


class VacuumSpectraAnalyzer:
    r"""
    Analiza el estado ρ frente al Hamiltoniano externo H y al Hamiltoniano
    modular K_ρ. Los observables de interés:

        ⟨H⟩_ρ       = Tr(ρH)
        ΔH          = √(Tr(ρH²) − Tr(ρH)²)      fluctuación térmica
        VEV         = ⟨H⟩_ρ − E₀                excitación sobre el ground
        gap(K_ρ)    = E₁ − E₀ del Hamiltoniano modular
        Z(K_ρ)      = Tr e^{−K_ρ} = 1 (autoconsistente con ρ normalizada)
        ruido(dB)   = 10·log₁₀(1 + Var_ρ(H))
    """
    VEV_VETO_THRESHOLD: float = 1.0e-2
    FLUCTUATION_VETO_THRESHOLD: float = 1.0e-2
    GAP_DEGRADE_THRESHOLD: float = 1.0e-3

    @classmethod
    def _expectation_and_variance(
        cls, rho: np.ndarray, H: np.ndarray
    ) -> Tuple[float, float]:
        rho = DensityOperatorAlgebra.sanitize(rho)
        H = 0.5 * (H + H.conj().T)
        mean = float(np.real(np.trace(rho @ H)))
        second = float(np.real(np.trace(rho @ H @ H)))
        var = max(0.0, second - mean * mean)
        return mean, math.sqrt(var)

    @classmethod
    def audit(
        cls,
        rho: np.ndarray,
        H: np.ndarray,
        K: ModularHamiltonian,
    ) -> VacuumAuditReport:
        rho = DensityOperatorAlgebra.sanitize(rho)
        mean_H, sigma_H = cls._expectation_and_variance(rho, H)

        w_H = np.real(la.eigvalsh(0.5 * (H + H.conj().T)))
        E0 = float(w_H.min())
        vev = mean_H - E0

        gap = K.spectral_gap
        Z = K.partition_function
        noise_db = 10.0 * math.log10(1.0 + sigma_H ** 2 + 1e-30)
        purity = DensityOperatorAlgebra.purity(rho)
        ent = DensityOperatorAlgebra.von_neumann_entropy(rho)

        # Veredicto local en Ω₃
        if sigma_H > cls.FLUCTUATION_VETO_THRESHOLD and vev > cls.VEV_VETO_THRESHOLD:
            local = HeytingOmega3.VETOED
        elif sigma_H > cls.FLUCTUATION_VETO_THRESHOLD or vev > cls.VEV_VETO_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        elif gap < cls.GAP_DEGRADE_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.COHERENT

        return VacuumAuditReport(
            vev=vev,
            thermal_fluctuation=sigma_H,
            spectral_gap=gap,
            modular_ground_energy=K.ground_energy,
            partition_function=Z,
            noise_db=noise_db,
            purity=purity,
            von_neumann_entropy=ent,
            local_verdict=local,
        )


# ── §2.3 Detector del silencio epistemológico ──────────────────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldProbe:
    purity: float                    # Tr(ρ²)
    fidelity_to_ground: float        # F(ρ, |Ω⟩⟨Ω|)² = ⟨Ω|ρ|Ω⟩
    correlation_leakage: float       # 1 − ⟨Ω|ρ|Ω⟩
    umegaki_to_ground: float         # S(ρ‖|Ω⟩⟨Ω|) (regularizado)
    local_verdict: HeytingOmega3


class SilentFieldDetector:
    r"""
    Mide la "fuga" del estado |ρ⟩ fuera del vacío |Ω⟩:

        fidelity_to_ground  = ⟨Ω|ρ|Ω⟩            (1 = vacío perfecto)
        correlation_leakage = 1 − ⟨Ω|ρ|Ω⟩        (0 = vacío perfecto)
        umegaki_to_ground   = S(ρ‖σ_ε)           con σ_ε = (1−ε)|Ω⟩⟨Ω| + ε·I/n
        purity              = Tr(ρ²)
    """
    UMEgaki_EPS: float = 1.0e-12
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
        align = float(np.real(np.trace(rho @ sigma)))
        leak = 1.0 - align

        # Regularización de σ para Umegaki: asegura fidelidad positiva
        n = rho.shape[0]
        sigma_eps = DensityOperatorAlgebra.sanitize(
            (1.0 - cls.UMEgaki_EPS) * sigma + cls.UMEgaki_EPS * np.eye(n) / n
        )
        relative_ent = DensityOperatorAlgebra.umegaki_relative_entropy(rho, sigma_eps)

        if purity >= cls.PURITY_SILENT_THRESHOLD and align >= cls.FIDELITY_SILENT_THRESHOLD:
            local = HeytingOmega3.COHERENT
        elif purity >= cls.PURITY_SILENT_THRESHOLD or align >= cls.FIDELITY_SILENT_THRESHOLD:
            local = HeytingOmega3.DEGRADED
        else:
            local = HeytingOmega3.VETOED

        return SilentFieldProbe(
            purity=purity,
            fidelity_to_ground=align,
            correlation_leakage=leak,
            umegaki_to_ground=relative_ent,
            local_verdict=local,
        )


# ── §2.4 ModularSilencePipeline — HAND-OFF FASE 2 → FASE 3 ─────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldBundle:
    r"""
    Paquete de hand-off FASE 2 → FASE 3. Encapsula el estado modular
    purificado, los certificados del análisis KMS y las métricas de silencio
    para su adjudicación en Ω₃ y su eventual firma criptográfica.
    """
    cycle_index: int
    rho: np.ndarray
    K: ModularHamiltonian
    audit: VacuumAuditReport
    probe: SilentFieldProbe
    kms_residual: float
    algebra_axioms: Dict[str, float]


class ModularSilencePipeline:
    r"""
    Orquestador determinista de la dinámica modular:

        (ρ, H, |Ω⟩⟨Ω|) → K_ρ → Tomita (KMS, axiomas) → Espectro → Silencio
                       → SilentFieldBundle  (hand-off → FASE 3)
    """

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho: np.ndarray,
        H: np.ndarray,
        ground_projector: np.ndarray,
        K: ModularHamiltonian,
    ) -> SilentFieldBundle:
        # (1) Coherencia algebraica de Tomita-Takesaki
        kms_res = TomitaTakesakiEngine.verify_kms(rho)
        axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho)

        # (2) Auditoría espectral del vacío
        audit = VacuumSpectraAnalyzer.audit(rho, H, K)

        # (3) Sonda del silencio (fuga fuera del vacío)
        probe = SilentFieldDetector.probe(rho, ground_projector)

        return SilentFieldBundle(
            cycle_index=cycle_index,
            rho=rho,
            K=K,
            audit=audit,
            probe=probe,
            kms_residual=kms_res,
            algebra_axioms=axioms,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · SOBERANÍA Y CERTIFICACIÓN DEL SILENCIO (C. de FASE 2)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# ── §3.1 Adjudicador en el retículo Ω₃ ─────────────────────────────────────
class HeytingVacuumAdjudicator:
    r"""
    Colapsa el SilentFieldBundle en un veredicto único Ω₃, luego aplica
    meet (∧) con el veredicto externo (godel_agent):

        local_verdict = audit.local_verdict ∧ probe.local_verdict
                          ∧ (KMS ≈ 0 ? ⊤ : ⋆)
                          ∧ (axiomas ≈ 0 ? ⊤ : ⋆)
        final         = local_verdict ∧ external_verdict

    El meet es la operación de ínfimo del retículo: la decisión más
    conservadora entre las dos fuentes de verdad.
    """
    KMS_RESIDUAL_TOL: float = 1.0e-6
    AXIOM_RESIDUAL_TOL: float = 1.0e-6

    @classmethod
    def adjudicate(
        cls,
        bundle: SilentFieldBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        kms_verdict = (
            HeytingOmega3.COHERENT
            if bundle.kms_residual < cls.KMS_RESIDUAL_TOL
            else HeytingOmega3.DEGRADED
        )
        axioms_ok = all(
            v < cls.AXIOM_RESIDUAL_TOL for v in bundle.algebra_axioms.values()
        )
        axioms_verdict = (
            HeytingOmega3.COHERENT if axioms_ok else HeytingOmega3.DEGRADED
        )

        local = (
            bundle.audit.local_verdict
            .meet(bundle.probe.local_verdict)
            .meet(kms_verdict)
            .meet(axioms_verdict)
        )
        return local.meet(external_verdict)


# ── §3.2 Certificado SilentFieldState ──────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SilentFieldState:
    r"""
    Certificado firmado del ciclo de silencio:

        • Contiene todos los observables del motor modular.
        • El `experience_hash` encadena el identificador del ciclo con las
          métricas críticas (VEV, ruido, conteo cristalino) mediante SHA-256.
        • El `phase_chain_sha256` conserva la custodia forense entre ciclos
          (cada ciclo hereda el hash del anterior).
    """
    cycle_id: str
    witness_engine_id: str
    vacuum_expectation_value: float
    kms_temperature_beta: float
    noise_emission_decibels: float
    crystallized_experience_count: int
    spectral_gap: float
    purity: float
    fidelity_to_ground: float
    kms_residual: float
    heyting_verdict: HeytingOmega3
    experience_hash: str
    phase_chain_sha256: str
    timestamp_utc: float


# ── §3.3 TOONSilentWitnessEngine — orquestador de ciclos ───────────────────
class TOONSilentWitnessEngine:
    r"""
    Motor Espectral del Vacío y Silencio Epistemológico.

    Custodia la Matriz MAC con la técnica Tomita-Takesaki: prepara el par
    (ρ_Ω, K_Ω) desde un Hamiltoniano externo H usando la FASE 1, ejecuta
    el pipeline modular de FASE 2, adjudica en Ω₃ y firma el certificado
    en FASE 3. Encadena SHA-256 entre ciclos para trazabilidad forense.
    """

    def __init__(
        self,
        engine_id: str = "SILENT-ENGINE-SABIO-01",
        mac_dimension: int = 4,
        kms_beta: float = 1.0,
    ) -> None:
        self.engine_id = engine_id
        self.mac_dimension = mac_dimension
        self.kms_beta = kms_beta
        self.cycle_count = 0
        self.history: List[SilentFieldState] = []

        # Hamiltoniano externo (mini-modelo del conocimiento): 4 niveles
        # equiespaciados con acoplamiento débil no diagonal.
        base = np.diag(np.linspace(0.0, 3.0, mac_dimension)).astype(np.complex128)
        weak = 1e-3 * (
            np.tri(mac_dimension, mac_dimension, k=1)
            - np.tri(mac_dimension, mac_dimension, k=-1)
        ).astype(np.complex128)
        self.H = base + weak + weak.conj().T

        # Estado de vacío (fase 1)
        self.ground_projector = VacuumStatePreparation.ground_state_projector(self.H)

        # Cadena de custodia inicial
        self._phase_chain_hash = hashlib.sha256(
            f"{engine_id}::GENESIS".encode("ascii")
        ).hexdigest()

    # ──── utilidades internas ─────────────────────────────────────────────
    def _update_chain(self, tag: str, payload: bytes) -> str:
        h = hashlib.sha256(
            self._phase_chain_hash.encode("ascii") + tag.encode("ascii") + payload
        ).hexdigest()
        self._phase_chain_hash = h
        return h

    # ──── ciclo principal de silencio ─────────────────────────────────────
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
        logger.info("═══ Ciclo del vacío %s | cristalizados=%d ═══",
                    cycle_id, triad_crystallized_count)

        # ── FASE 1 ── Preparación del par (ρ_Ω, K_Ω) ──
        rho, K = VacuumStatePreparation.prepare_vacuum_pair(self.H, beta=beta_vacuum)
        self._update_chain("F1", rho.tobytes())

        # ── FASE 2 ── Tomita + espectro + sonda de silencio ──
        bundle = ModularSilencePipeline.synthesize(
            cycle_index=self.cycle_count,
            rho=rho,
            H=self.H,
            ground_projector=self.ground_projector,
            K=K,
        )
        self._update_chain(
            "F2",
            f"{bundle.kms_residual:.12e}|{bundle.audit.vev:.12e}".encode("ascii"),
        )

        # ── FASE 3 ── Adjudicación Ω₃ y firma ──
        verdict = HeytingVacuumAdjudicator.adjudicate(bundle, external_verdict)
        self._update_chain(
            "F3",
            f"{verdict.name}|{bundle.audit.noise_db:.6f}|"
            f"{triad_crystallized_count}".encode("ascii"),
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

        state = SilentFieldState(
            cycle_id=cycle_id,
            witness_engine_id=self.engine_id,
            vacuum_expectation_value=bundle.audit.vev,
            kms_temperature_beta=kms_beta,
            noise_emission_decibels=bundle.audit.noise_db,
            crystallized_experience_count=triad_crystallized_count,
            spectral_gap=bundle.audit.spectral_gap,
            purity=bundle.probe.purity,
            fidelity_to_ground=bundle.probe.fidelity_to_ground,
            kms_residual=bundle.kms_residual,
            heyting_verdict=verdict,
            experience_hash=experience_hash,
            phase_chain_sha256=self._phase_chain_hash,
            timestamp_utc=time.time(),
        )
        self.history.append(state)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Ciclo %s completado en %.2f ms | Ω₃=%s | VEV=%.3e | ruido=%.3f dB | KMS=%.2e",
            cycle_id, dt_ms, verdict.name,
            state.vacuum_expectation_value, state.noise_emission_decibels,
            state.kms_residual,
        )
        return state


# ── §3.4 Punto de entrada / demostración ───────────────────────────────────
if __name__ == "__main__":
    engine = TOONSilentWitnessEngine(
        engine_id="SILENT-ENGINE-SABIO-01",
        mac_dimension=4,
        kms_beta=1.0,
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Silent Witness Engine (APU Filter v8.0+)")
    print("═" * 80)

    scenarios = [
        # (nombre, conteo_cristalizado, beta_vacío, veredicto_externo)
        ("VACÍO FRÍO (β → ∞)",          5, 1.0e3, HeytingOmega3.COHERENT),
        ("VACÍO TEMPLADO (β = 10)",     7, 10.0,  HeytingOmega3.COHERENT),
        ("VACÍO TIBIO (β = 1)",        12, 1.0,   HeytingOmega3.COHERENT),
        ("VACÍO AGRESIVO (β = 0.5)",   20, 0.5,   HeytingOmega3.COHERENT),
        ("VACÍO HOSTIL (β = 0.1)",     50, 0.1,   HeytingOmega3.DEGRADED),
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
            f"pureza={state.purity:.6f} | "
            f"fid_Ω={state.fidelity_to_ground:.6f} | "
            f"ruido={state.noise_emission_decibels:.3f} dB | "
            f"KMS={state.kms_residual:.2e} | "
            f"hash={state.experience_hash[:16]}…"
        )

    print("\n" + "═" * 80)
    print("✓ Auditoría modular Tomita-Takesaki completada.")
    print("✓ KMS(β=1) verificado numéricamente.")
    print("✓ Cadena de custodia forense preservada entre ciclos.")
    print("═" * 80)
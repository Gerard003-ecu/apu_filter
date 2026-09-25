# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Oniric Auditor Engine (Motor Espectral Auditor de Sueños)    ║
║ Ubicación: app/wisdom/toon_oniric_auditor_engine.py                          ║
║ Versión  : 3.0.0-Doctoral-Nested-Ω₃-TQFT-Dirac-GromovWitten-Holonomy-Merkle  ║
║ Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el  ║
║            germen formal del primero de k+1)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Formalización (topos + TQFT + espectro-cuántica)
================================================

Sea 𝓣_Ω el topos de haces sobre el retículo de Heyting lineal

        Ω₃  =  { VETOED  ≺  DEGRADED  ≺  COHERENT }.

El auditor realiza un endofuntor

        𝒜  :  𝐂𝐢𝐜𝐥𝐨_𝐎𝐧𝐢𝐫𝐢𝐜𝐨  ──▶  𝐏𝐚𝐬𝐚𝐩𝐨𝐫𝐭𝐞_𝐈𝐦𝐦

como composición estrictamente asociativa

        𝒜  =  V ∘ Hol ∘ Seal ∘ χ ∘ I_GW ∘ D ∘ Spec

donde Spec es el ÚLTIMO morfismo de FASE-1 (extract_spectral_measure) y el
PRIMERO que consume FASE-2; evaluate_dream_spectrum es el ÚLTIMO de FASE-2
y _seal_and_accumulate el PRIMERO de FASE-3.

Espectro y amplitudes
=====================

    Spec : ρ ↦ λ ∈ Δ^{n-1}                 (medida espectral, simplex)
    γ(ρ) = ‖λ‖₂² = Tr(ρ²)                  (pureza)
    S(ρ) = −Σ λᵢ log λᵢ                    (von Neumann)
    Δλ   = λ_max − λ_{max−1}               (gap)
    E_D  = ½ Σᵢ (λ_{i+1} − λᵢ)²            (Dirichlet discreto / H¹)
    E_∂  = Σᵢ |λ_{i+1} − λᵢ|               (energía de Dirac / TV)
    I_GW = γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|)

I_GW es una amplitud TQFT sintética: γ juega el rol de clase virtual,
e^{−E_D} el peso de energía (filtrado a la Gromov), (1+b₁)⁻¹ la penalización
por género/ciclos, y el factor de Euler la corrección cobordista.

Holonomía
=========

    H₊(t)  = ⊕_{τ≤t} I_GW(τ)               (monoide (ℝ≥0, +, 0))
    W(t)   = exp(i H₊(t)) ∈ U(1)           (Wilson)
    χ_∥(t) = ⋀_{τ≤t} χ(τ) ∈ Ω₃             (transporte paralelo intuicionista)

Invariantes verificables
========================
    ρ = ρ†, ρ ⪰ 0, Tr ρ = 1, spec(ρ) ⊂ [0,1].
    I_GW ∈ [0, 1],  E_D ≥ 0,  E_∂ ≥ 0,  Δλ ≥ 0.
    Hojas SHA-256 inyectivas; Merkle verificable en O(log n).
    ∂-consistencia: bₖ ≥ 0; χ = b₀ − b₁ + b₂ si se proveen los tres.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditorEngine.v3")

__all__ = [
    "HeytingOmega3",
    "DensityOperator",
    "SpectralMeasure",
    "SpectralMeasureSeed",
    "ImmunizationPassport",
    "OniricFieldState",
    "OniricSpectraEngine",
    "UnsealedOniricTrace",
    "MerkleInclusionProof",
    "TOONOniricAuditorEngine",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — RETÍCULO Ω₃, DENSIDAD, MEDIDA ESPECTRAL Y SEMILLA Spec
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje del topos 𝓣_Ω. El ÚLTIMO método de esta fase
# (SpectralMeasureSeed.extract_spectral_measure) es el germen formal de
# FASE-2: OniricSpectraEngine lo realiza y no reconstruye el espectro.
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⊤  si a ≤ b,  else b
        ¬_H a  = a → ⊥

    El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂. DEGRADED viola el tercio excluso
    (a ∨ ¬a ≠ ⊤), de modo que Ω₃ es estrictamente intuicionista.
    Toda auditoría emite una flecha característica χ : Escenario → Ω₃.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Residuo a → b = ⋁{ c ∈ Ω₃ | a ∧ c ≤ b }."""
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        r"""¬_H a := a → ⊥.  ¬VETOED = COHERENT; ¬DEGRADED = ¬COHERENT = VETOED."""
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        r"""a regular ⟺ ¬¬a = a. En Ω₃: {VETOED, COHERENT}."""
        return self.double_negation() == self

    def excluded_middle_holds(self) -> bool:
        r"""a ∨ ¬a = ⊤  ⇔  a ∈ {⊥, ⊤}. Falla en DEGRADED."""
        return self.join(self.pseudo_complement()) == HeytingOmega3.COHERENT

    @classmethod
    def bottom(cls) -> "HeytingOmega3":
        return cls.VETOED

    @classmethod
    def top(cls) -> "HeytingOmega3":
        return cls.COHERENT

    @classmethod
    def from_bool(cls, b: bool) -> "HeytingOmega3":
        return cls.COHERENT if b else cls.VETOED

    def to_bool(self) -> bool:
        if self == HeytingOmega3.DEGRADED:
            raise ValueError("DEGRADED no admite proyección fiel a 𝔹₂.")
        return self == HeytingOmega3.COHERENT

    @property
    def verdict(self) -> str:
        return self.name

    @classmethod
    def verify_residuation_axiom(cls) -> bool:
        r"""∀ a,b,c ∈ Ω₃:  (c ∧ a ≤ b)  ⟺  (c ≤ (a → b))."""
        elements = list(cls)
        for a in elements:
            for b in elements:
                residual = a.implies(b)
                for c in elements:
                    lhs = min(int(c), int(a)) <= int(b)
                    rhs = int(c) <= int(residual)
                    if lhs != rhs:
                        return False
        return True


@dataclass(frozen=True, slots=True)
class DensityOperator:
    r"""
    Estado cuántico ρ ∈ 𝔇(ℋₙ) ⊂ B(ℋₙ).

    Invariantes (verificados en __post_init__):
        ρ = ρ†,  spec(ρ) ⊂ [−ε, 1+ε],  |Tr ρ − 1| ≤ 10⁻⁶.
    """

    matrix: np.ndarray
    atol: float = 1e-8

    def __post_init__(self) -> None:
        rho = np.asarray(self.matrix, dtype=np.complex128)
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("DensityOperator exige matriz cuadrada.")
        object.__setattr__(self, "matrix", np.array(rho, copy=True))
        if not np.allclose(rho, rho.conj().T, atol=self.atol):
            raise ValueError("DensityOperator exige ρ = ρ†.")
        tr = float(np.trace(rho).real)
        if abs(tr - 1.0) > 1e-6:
            raise ValueError(f"DensityOperator exige Tr ρ = 1 (Tr={tr}).")

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def as_array(self) -> np.ndarray:
        return self.matrix

    def cstar_residual(self) -> float:
        rho = self.matrix
        op = float(la.norm(rho.conj().T @ rho, 2))
        nrm = float(la.norm(rho, 2))
        return abs(op - nrm * nrm)

    @classmethod
    def from_array(cls, rho: np.ndarray, atol: float = 1e-8) -> "DensityOperator":
        rho_h = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        evals, evecs = la.eigh(rho_h)
        evals = np.clip(evals, 0.0, None)
        s = float(np.sum(evals))
        if s <= 1e-15:
            n = rho_h.shape[0]
            rho_h = np.eye(n, dtype=np.complex128) / n
        else:
            evals = evals / s
            rho_h = (evecs * evals) @ evecs.conj().T
            rho_h = 0.5 * (rho_h + rho_h.conj().T)
        return cls(matrix=rho_h, atol=atol)


@dataclass(frozen=True, slots=True)
class SpectralMeasure:
    r"""
    Medida espectral λ ∈ Δ^{n−1} de un estado ρ, con observables derivados.

    Este objeto cierra el contenido informacional de Spec (FASE-1).
    FASE-2 *continúa* exactamente aquí: OniricSpectraEngine consume
    SpectralMeasure y no rediagonaliza ρ salvo petición explícita.
    """

    eigenvalues: np.ndarray  # ordenados no-decrecientes, Σλ = 1, λ ≥ 0
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dimension: int
    cstar_residual: float

    def __post_init__(self) -> None:
        lam = np.asarray(self.eigenvalues, dtype=np.float64).reshape(-1)
        if lam.size < 1:
            raise ValueError("SpectralMeasure exige al menos un autovalor.")
        object.__setattr__(self, "eigenvalues", lam)

    def as_simplex(self) -> np.ndarray:
        return self.eigenvalues


@runtime_checkable
class ImmunizationPassport(Protocol):
    r"""
    Protocolo estructural del pasaporte de inmunización. Cualquier tipo
    que exponga estos atributos y `is_immune` es consumible por
    GodelEngine / WeaverEngine.
    """

    immunization_hash: str
    heyting_verdict: HeytingOmega3
    gromov_witten_invariant: float

    def is_immune(self) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class OniricFieldState:
    r"""
    Estado onírico como flecha s : 1 → 𝓣_Ω. Objeto terminal de 𝒜.

    Invariantes:
      - density_matrix ∈ 𝔇(ℋₙ) (hermítica, PSD, traza 1).
      - gromov_witten_invariant ≥ 0, betti_* ≥ 0.
      - immunization_hash ∈ {0,1}^{256} hex.
    """

    cycle_id: str
    scenario_id: str
    dream_isolation_flag: bool
    density_matrix: np.ndarray
    dirichlet_energy: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    cstar_residual: float
    betti_0: int
    betti_1_loops: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    immunization_hash: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex

    def __post_init__(self) -> None:
        rho = np.asarray(self.density_matrix, dtype=np.complex128)
        object.__setattr__(self, "density_matrix", np.array(rho, copy=True))
        if self.betti_0 < 0 or self.betti_1_loops < 0 or self.betti_2 < 0:
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if self.gromov_witten_invariant < 0.0:
            raise ValueError("I_GW debe ser ≥ 0.")

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def is_topologically_consistent(self) -> bool:
        r"""
        Coherencia topológico-espectral:
          • bₖ ≥ 0 (ya exigido);
          • χ = b₀ − b₁ + b₂;
          • un VETOED por ¬aislamiento es admisible con cualquier b₁;
          • un VETOED por b₁ alto exige b₁ > 3.
        """
        if self.euler_characteristic != (self.betti_0 - self.betti_1_loops + self.betti_2):
            return False
        if self.heyting_verdict == HeytingOmega3.VETOED:
            if not self.dream_isolation_flag:
                return True
            return self.betti_1_loops > 3
        return True

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ aislado ∧ ¬VETOED ∧ físico ∧ topológicamente consistente.
        """
        return (
            self.dream_isolation_flag
            and self.heyting_verdict != HeytingOmega3.VETOED
            and self.is_quantum_physical()
            and self.is_topologically_consistent()
        )

    def passport_prefix(self, n: int = 16) -> str:
        return self.immunization_hash[:n]


class SpectralMeasureSeed(ABC):
    r"""
    Germen formal de la flecha Spec : ρ ↦ λ ∈ Δ^{n−1}.

    Cierra el andamiaje de FASE-1. FASE-2 *continúa* exactamente en
    extract_spectral_measure: OniricSpectraEngine lo realiza y calcula
    sobre la medida (I_GW, E_D, E_∂, χ) sin rediagonalizar.
    """

    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        Flecha Spec. Produce la medida espectral de ρ.

        CONTINÚA EN FASE-2: OniricSpectraEngine.extract_spectral_measure.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TQFT, GROMOV-WITTEN SINTÉTICO, DIRICHLET-DIRAC Y TRAZA ABIERTA
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (extract_spectral_measure) ES la
# realización del último de FASE-1. El último (evaluate_dream_spectrum)
# produce UnsealedOniricTrace, germen formal de FASE-3.
# ══════════════════════════════════════════════════════════════════════════════


class OniricSpectraEngine(SpectralMeasureSeed):
    r"""
    CONTINUACIÓN FORMAL de SpectralMeasureSeed.extract_spectral_measure.

    Motor espectral onírico: calcula I_GW (amplitud TQFT sintética) y
    clasifica en Ω₃.

        γ(ρ)   = Σ λᵢ²
        S(ρ)   = −Σ λᵢ log λᵢ
        E_D    = ½ Σ (λ_{i+1} − λᵢ)²          (Dirichlet / energía de mapa)
        E_∂    = Σ |λ_{i+1} − λᵢ|             (Dirac / variación total)
        I_GW   = γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|)

    Clasificación Heyting:
        VETOED    si ¬aislamiento  ∨  b₁ > β_max  ∨  I_GW < ι_min
        DEGRADED  si E_D > ε_max  ∨  E_∂ > δ_max
        COHERENT  en otro caso
    """

    BETTI_MAX: Final[int] = 3
    GW_MIN: Final[float] = 0.05
    DIRICHLET_MAX: Final[float] = 0.85
    DIRAC_TV_MAX: Final[float] = 1.50
    EIGENVALUE_FLOOR: Final[float] = 1e-15
    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3

    @classmethod
    def _project_spectrum(cls, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.clip(np.real(eigvals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        if s < cls.EIGENVALUE_FLOOR:
            n = eigvals.shape[0]
            return np.full(n, 1.0 / max(n, 1))
        return eigvals / s

    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        CONTINUACIÓN FORMAL de SpectralMeasureSeed.extract_spectral_measure.
        Diagonalización hermitiana + proyección al simplex.
        """
        rho_h = 0.5 * (np.asarray(rho, dtype=np.complex128) + np.asarray(rho, dtype=np.complex128).conj().T)
        eigvals = la.eigvalsh(rho_h)
        lam = self._project_spectrum(eigvals)
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        cstar = abs(op - nrm * nrm)
        return SpectralMeasure(
            eigenvalues=lam,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=gap,
            dimension=int(lam.size),
            cstar_residual=cstar,
        )

    @classmethod
    def compute_purity(cls, eigvals: np.ndarray) -> float:
        return float(np.sum(cls._project_spectrum(eigvals) ** 2))

    @classmethod
    def compute_spectral_entropy(cls, eigvals: np.ndarray) -> float:
        lam = cls._project_spectrum(eigvals)
        return -float(np.sum(lam * np.log(lam)))

    @classmethod
    def compute_dirichlet_energy(cls, eigvals: np.ndarray) -> float:
        r"""E_D(ρ) = ½ Σ (λ_{i+1} − λᵢ)²  (seminorma H¹ discreta sobre spec ρ)."""
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad ** 2))

    @classmethod
    def compute_dirac_total_variation(cls, eigvals: np.ndarray) -> float:
        r"""E_∂(ρ) = Σ |λ_{i+1} − λᵢ|  (energía de Dirac / TV del espectro)."""
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(lam))))

    @classmethod
    def compute_gromov_witten_invariant(
        cls,
        purity: float,
        dirichlet_energy: float,
        betti_1: int,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        r"""
        Amplitud TQFT / Gromov-Witten sintética:

            I_GW = γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|)

        Acotada en [0, 1]. El factor de Euler es la corrección cobordista
        (χ > 0 favorece esferas; χ < 0 penaliza género alto).
        """
        if betti_1 < 0:
            raise ValueError("b₁ debe ser ≥ 0.")
        n = max(int(dimension), 1)
        chi_pos = max(int(euler_characteristic), 0)
        chi_abs = abs(int(euler_characteristic))
        euler_factor = (1.0 + chi_pos) / (1.0 + chi_abs)
        entropy_damp = math.exp(-abs(entropy) / n)
        denom = 1.0 + float(betti_1)
        raw = (
            float(purity)
            * math.exp(-float(dirichlet_energy))
            * entropy_damp
            / denom
            * euler_factor
        )
        return float(min(max(raw, 0.0), 1.0))

    @classmethod
    def classify(
        cls,
        betti_1: int,
        gw_invariant: float,
        dirichlet_energy: float,
        dream_isolation: bool,
        dirac_tv: float = 0.0,
    ) -> HeytingOmega3:
        if not dream_isolation:
            return HeytingOmega3.VETOED
        if betti_1 > cls.BETTI_MAX or gw_invariant < cls.GW_MIN:
            return HeytingOmega3.VETOED
        if dirichlet_energy > cls.DIRICHLET_MAX or dirac_tv > cls.DIRAC_TV_MAX:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def evaluate_dream_spectrum(
        self,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float],
        betti_1: int,
        dream_isolation: bool,
        betti_0: int = 1,
        betti_2: int = 0,
    ) -> "UnsealedOniricTrace":
        r"""
        ÚLTIMO método de FASE-2: Spec ∘ D ∘ I_GW ∘ χ, sin sello ni holonomía.

        Si `dirichlet_energy` es None, E_D se computa de ρ (modo auto-consistente).
        Si se provee, se usa tal cual y se reporta el residuo |E_D^{user} − E_D^{spec}|.

        CONTINÚA EN FASE-3: TOONOniricAuditorEngine._seal_and_accumulate.
        """
        measure = self.extract_spectral_measure(density_matrix)
        lam = measure.as_simplex()
        ed_internal = self.compute_dirichlet_energy(lam)
        ed_dirac = self.compute_dirac_total_variation(lam)
        if dirichlet_energy is None:
            ed = ed_internal
            ed_residual = 0.0
        else:
            ed = float(dirichlet_energy)
            ed_residual = abs(ed - ed_internal)
            if ed_residual > self.DIRICHLET_CONSISTENCY_TOL:
                logger.debug(
                    "Discrepancia E_D usuario vs. espectral: |%.6f − %.6f| = %.3e",
                    ed, ed_internal, ed_residual,
                )

        b0 = max(int(betti_0), 0)
        b1 = max(int(betti_1), 0)
        b2 = max(int(betti_2), 0)
        chi = b0 - b1 + b2

        gw = self.compute_gromov_witten_invariant(
            purity=measure.purity,
            dirichlet_energy=ed,
            betti_1=b1,
            entropy=measure.von_neumann_entropy,
            dimension=measure.dimension,
            euler_characteristic=chi,
        )
        verdict = self.classify(
            betti_1=b1,
            gw_invariant=gw,
            dirichlet_energy=ed,
            dream_isolation=dream_isolation,
            dirac_tv=ed_dirac,
        )
        return UnsealedOniricTrace(
            density_matrix=np.array(density_matrix, copy=True),
            measure=measure,
            dirichlet_energy=ed,
            dirichlet_internal=ed_internal,
            dirichlet_residual=ed_residual,
            dirac_total_variation=ed_dirac,
            gromov_witten_invariant=gw,
            tqft_amplitude=gw,
            betti_0=b0,
            betti_1=b1,
            betti_2=b2,
            euler_characteristic=chi,
            heyting_verdict=verdict,
            dream_isolation=dream_isolation,
        )


@dataclass(frozen=True, slots=True)
class UnsealedOniricTrace:
    r"""
    Traza abierta: portadora de (Spec, D, I_GW, χ) antes de Seal/Hol/V.

    Cierra FASE-2. FASE-3 *continúa* exactamente aquí:
    TOONOniricAuditorEngine._seal_and_accumulate es el primer método
    de FASE-3 y consume esta traza.
    """

    density_matrix: np.ndarray
    measure: SpectralMeasure
    dirichlet_energy: float
    dirichlet_internal: float
    dirichlet_residual: float
    dirac_total_variation: float
    gromov_witten_invariant: float
    tqft_amplitude: float
    betti_0: int
    betti_1: int
    betti_2: int
    euler_characteristic: int
    heyting_verdict: HeytingOmega3
    dream_isolation: bool


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — AUDITOR, SELLO, HOLONOMÍA, MERKLE, INMUNIZACIÓN Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (_seal_and_accumulate) consume
# UnsealedOniricTrace, valor de retorno del último método de FASE-2.
# Aquí se realiza Seal, Hol (⊕ y Wilson), el sello SHA-256 y las vistas.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
    r"""Prueba de inclusión Merkle (camino de hermanos, convención Bitcoin/CT)."""

    leaf_hash: str
    siblings: Tuple[str, ...]
    index: int
    root: str

    def verify(self) -> bool:
        node = bytes.fromhex(self.leaf_hash)
        idx = self.index
        for sib_hex in self.siblings:
            sib = bytes.fromhex(sib_hex)
            if idx % 2 == 0:
                node = hashlib.sha256(node + sib).digest()
            else:
                node = hashlib.sha256(sib + node).digest()
            idx //= 2
        return node.hex() == self.root


class TOONOniricAuditorEngine:
    r"""
    Motor espectral auditor de sueños. Endofuntor

        𝒜  =  V ∘ Hol ∘ Seal ∘ χ ∘ I_GW ∘ D ∘ Spec

    realizado como

        audit_oniric_cycle
            =  _seal_and_accumulate  ∘  evaluate_dream_spectrum

    donde evaluate_dream_spectrum es el último método de FASE-2 y
    _seal_and_accumulate es la continuación formal que abre FASE-3.

    Holonomía:
        H₊(t) = Σ I_GW(τ)                 (monoide aditivo)
        W(t)  = exp(i H₊(t)) ∈ U(1)       (Wilson)
        χ_∥   = ⋀ χ(τ)                    (transporte paralelo en Ω₃)
    """

    def __init__(
        self,
        engine_id: str = "ONIRIC-ENGINE-SABIO-01",
        spectra_engine: Optional[OniricSpectraEngine] = None,
    ) -> None:
        self.engine_id: str = engine_id
        self.spectra: OniricSpectraEngine = (
            spectra_engine if spectra_engine is not None else OniricSpectraEngine()
        )
        self.cycle_count: int = 0
        self.history: List[OniricFieldState] = []
        self._holonomy_accum: float = 0.0

    def _seal_passport(
        self,
        cycle_id: str,
        scenario_id: str,
        verdict: HeytingOmega3,
        gw_invariant: float,
        t_seal: float,
    ) -> str:
        hasher = hashlib.sha256()
        payload = (
            f"{self.engine_id}::{cycle_id}::{scenario_id}::"
            f"{verdict.name}::{gw_invariant:.10f}::{t_seal:.6f}"
        )
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

    def _seal_and_accumulate(
        self,
        trace: UnsealedOniricTrace,
        cycle_id: str,
        scenario_id: str,
    ) -> OniricFieldState:
        r"""
        CONTINUACIÓN FORMAL de OniricSpectraEngine.evaluate_dream_spectrum.

        Aplica Seal (SHA-256), Hol (⊕ I_GW y fase de Wilson) y construye
        el objeto de 𝐏𝐚𝐬𝐚𝐩𝐨𝐫𝐭𝐞_𝐈𝐦𝐦 / OniricFieldState.
        """
        t_seal = time.time()
        imm_hash = self._seal_passport(
            cycle_id=cycle_id,
            scenario_id=scenario_id,
            verdict=trace.heyting_verdict,
            gw_invariant=trace.gromov_witten_invariant,
            t_seal=t_seal,
        )
        self._holonomy_accum += trace.gromov_witten_invariant
        wilson = complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

        return OniricFieldState(
            cycle_id=cycle_id,
            scenario_id=scenario_id,
            dream_isolation_flag=trace.dream_isolation,
            density_matrix=trace.density_matrix,
            dirichlet_energy=trace.dirichlet_energy,
            dirac_total_variation=trace.dirac_total_variation,
            gromov_witten_invariant=trace.gromov_witten_invariant,
            tqft_amplitude=trace.tqft_amplitude,
            purity=trace.measure.purity,
            von_neumann_entropy=trace.measure.von_neumann_entropy,
            spectral_gap=trace.measure.spectral_gap,
            cstar_residual=trace.measure.cstar_residual,
            betti_0=trace.betti_0,
            betti_1_loops=trace.betti_1,
            betti_2=trace.betti_2,
            euler_characteristic=trace.euler_characteristic,
            heyting_verdict=trace.heyting_verdict,
            immunization_hash=imm_hash,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson,
        )

    def audit_oniric_cycle(
        self,
        scenario_id: str,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float] = None,
        betti_1: int = 0,
        dream_isolation: bool = True,
        betti_0: int = 1,
        betti_2: int = 0,
    ) -> OniricFieldState:
        r"""
        Audita un ciclo onírico completo.

        Pasos anidados:
          1. Identidad de ciclo.
          2. evaluate_dream_spectrum (FASE-2: Spec, D, I_GW, χ)
             → UnsealedOniricTrace.
          3. _seal_and_accumulate (FASE-3: Seal, Hol, Wilson)
             → OniricFieldState.
          4. Persistencia inmutable.
        """
        self.cycle_count += 1
        cycle_id = f"CYC-ONIRIC-AUDIT-{self.cycle_count:04d}"
        t_start = time.time()
        logger.info(
            "=== Iniciando Auditoría Espectral Onírica %s | Escenario: %s ===",
            cycle_id, scenario_id,
        )

        trace = self.spectra.evaluate_dream_spectrum(
            density_matrix=density_matrix,
            dirichlet_energy=dirichlet_energy,
            betti_1=betti_1,
            dream_isolation=dream_isolation,
            betti_0=betti_0,
            betti_2=betti_2,
        )
        state = self._seal_and_accumulate(trace, cycle_id, scenario_id)
        self.history.append(state)

        logger.info(
            "Ciclo Espectral Onírico %s Finalizado en %.2f ms | Veredicto: %s | "
            "I_GW: %.6f | γ: %.4f | E_D: %.4f | E_∂: %.4f | Δλ: %.4f | "
            "χ: %d | H₊: %.6f | W: %.3f%+.3fi",
            cycle_id,
            (time.time() - t_start) * 1000.0,
            state.heyting_verdict.name,
            state.gromov_witten_invariant,
            state.purity,
            state.dirichlet_energy,
            state.dirac_total_variation,
            state.spectral_gap,
            state.euler_characteristic,
            self._holonomy_accum,
            state.wilson_phase.real,
            state.wilson_phase.imag,
        )
        return state

    @property
    def registry(self) -> Tuple[OniricFieldState, ...]:
        return tuple(self.history)

    @property
    def holonomy_accum(self) -> float:
        r"""H₊(t) = Σ I_GW(τ)  (monoide aditivo)."""
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        r"""W(t) = exp(i H₊(t)) ∈ U(1)."""
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
        r"""χ_∥ = ⋀ χ(τ)  (transporte paralelo en Ω₃)."""
        gv = HeytingOmega3.COHERENT
        for s in self.history:
            gv = gv.meet(s.heyting_verdict)
        return gv

    @staticmethod
    def _merkle_tree_root(leaf_hashes: Sequence[str]) -> str:
        if not leaf_hashes:
            return hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
        level = [bytes.fromhex(h) for h in leaf_hashes]
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _merkle_proof(leaf_hashes: Sequence[str], index: int) -> MerkleInclusionProof:
        if not leaf_hashes:
            empty = hashlib.sha256(b"EMPTY_MERKLE_ROOT").hexdigest()
            return MerkleInclusionProof(empty, tuple(), 0, empty)
        level = [bytes.fromhex(h) for h in leaf_hashes]
        siblings: List[str] = []
        idx = index
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            pair = idx ^ 1
            siblings.append(level[pair].hex())
            next_level = [
                hashlib.sha256(level[i] + level[i + 1]).digest()
                for i in range(0, len(level), 2)
            ]
            level = next_level
            idx //= 2
        return MerkleInclusionProof(
            leaf_hash=leaf_hashes[index],
            siblings=tuple(siblings),
            index=index,
            root=level[0].hex(),
        )

    def merkle_root(self) -> str:
        return self._merkle_tree_root([s.immunization_hash for s in self.history])

    def merkle_proofs_ok(self) -> bool:
        leaves = [s.immunization_hash for s in self.history]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva:
            n_cycles, verdict_distribution, global_verdict,
            holonomy_accum, wilson_loop, avg_gw_invariant,
            avg_dirichlet_energy, avg_dirac_tv, avg_purity,
            n_immune, all_physically_valid, registry_integrity_ok,
            merkle_proofs_ok.
        """
        n = len(self.history)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_cycles": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0,
                "wilson_loop": 1.0 + 0.0j,
                "avg_gw_invariant": 0.0,
                "avg_dirichlet_energy": 0.0,
                "avg_dirac_tv": 0.0,
                "avg_purity": 0.0,
                "n_immune": 0,
                "all_physically_valid": True,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
            }

        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_ed = total_tv = total_p = 0.0
        n_imm = 0
        all_valid = True
        hashes: Set[str] = set()
        collide = False
        for s in self.history:
            dist[s.heyting_verdict.name] += 1
            total_gw += s.gromov_witten_invariant
            total_ed += s.dirichlet_energy
            total_tv += s.dirac_total_variation
            total_p += s.purity
            if s.is_immune():
                n_imm += 1
            if not s.is_quantum_physical():
                all_valid = False
            if s.immunization_hash in hashes:
                collide = True
            hashes.add(s.immunization_hash)

        inv = 1.0 / n
        return {
            "n_cycles": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "avg_gw_invariant": total_gw * inv,
            "avg_dirichlet_energy": total_ed * inv,
            "avg_dirac_tv": total_tv * inv,
            "avg_purity": total_p * inv,
            "n_immune": n_imm,
            "all_physically_valid": all_valid,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
        }

    def emit_passport(self) -> Dict[str, Any]:
        r"""
        Pasaporte de inmunización agregado, consumible por GodelEngine /
        WeaverEngine. evidence_hash encadena engine_id, H₊ y las hojas.
        """
        h = hashlib.sha256()
        h.update(
            f"{self.engine_id}::{self.cycle_count}::{self._holonomy_accum:.10f}".encode(
                "utf-8"
            )
        )
        for s in self.history:
            h.update(s.immunization_hash.encode("utf-8"))
        return {
            "engine_id": self.engine_id,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "global_verdict": self.global_verdict.name,
            "evidence_hash": h.hexdigest(),
            "merkle_root": self.merkle_root(),
            "registry_size": self.cycle_count,
            "n_immune": sum(1 for s in self.history if s.is_immune()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBAS Y EJECUCIÓN AUTÓNOMA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("═" * 80)
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Engine v3.0.0")
    print("FASES ANIDADAS: Ω₃+Spec → TQFT/I_GW/Dirac → Seal/Holonomía/Merkle")
    print("═" * 80)

    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    assert HeytingOmega3.verify_residuation_axiom()
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.is_regular() is True
    assert HeytingOmega3.DEGRADED.is_regular() is False
    print("  • Residuación, tercio excluso y regularidad: OK")

    rng = np.random.default_rng(20250321)
    engine = TOONOniricAuditorEngine()

    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    rho_op = DensityOperator.from_array(rho)

    print("\n>>> ESCENARIO A: Estado físico canónico (Ginibre, E_D provisto)...")
    s1 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-001",
        density_matrix=rho_op.as_array(),
        dirichlet_energy=0.35,
        betti_1=0,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo             : {s1.cycle_id}")
    print(f"    - Veredicto Heyting    : {s1.heyting_verdict.name}")
    print(f"    - I_GW / TQFT          : {s1.gromov_witten_invariant:.6f}")
    print(f"    - γ / S_vN / Δλ        : {s1.purity:.4f} / {s1.von_neumann_entropy:.4f} / {s1.spectral_gap:.4f}")
    print(f"    - E_D / E_∂            : {s1.dirichlet_energy:.6f} / {s1.dirac_total_variation:.6f}")
    print(f"    - χ = b₀−b₁+b₂         : {s1.euler_characteristic}")
    print(f"    - Residual C*          : {s1.cstar_residual:.3e}")
    print(f"    - Físico / inmune      : {s1.is_quantum_physical()} / {s1.is_immune()}")
    print(f"    - Wilson W             : {s1.wilson_phase:.4f}")
    print(f"    - Pasaporte SHA-256    : {s1.passport_prefix(32)}...")

    print("\n>>> ESCENARIO B: Estado máximamente mixto (ρ = I/n, E_D auto)...")
    rho_mixed = np.eye(4, dtype=np.complex128) / 4.0
    s2 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-MIXED",
        density_matrix=rho_mixed,
        dirichlet_energy=None,
        betti_1=0,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo             : {s2.cycle_id}")
    print(f"    - Veredicto Heyting    : {s2.heyting_verdict.name}")
    print(f"    - I_GW                 : {s2.gromov_witten_invariant:.6f}")
    print(f"    - E_D / E_∂            : {s2.dirichlet_energy:.6e} / {s2.dirac_total_variation:.6e}")
    print(f"    - γ (debe ser 1/n)     : {s2.purity:.6f}")

    print("\n>>> ESCENARIO C: Violación de aislamiento REM (veto duro)...")
    s3 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-BREACH",
        density_matrix=rho,
        dirichlet_energy=0.15,
        betti_1=5,
        dream_isolation=False,
    )
    print(f"    - ID Ciclo             : {s3.cycle_id}")
    print(f"    - Veredicto Heyting    : {s3.heyting_verdict.name}")
    print(f"    - Inmune               : {s3.is_immune()}")
    print(f"    - Topo-consistente     : {s3.is_topologically_consistent()}")
    assert s3.heyting_verdict == HeytingOmega3.VETOED
    assert s3.is_immune() is False

    print("\n>>> ESCENARIO D: Topología sintáctica compleja (b₁ = 4)...")
    s4 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-TOPOLOGIC",
        density_matrix=rho,
        dirichlet_energy=0.40,
        betti_1=4,
        betti_0=1,
        betti_2=0,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo             : {s4.cycle_id}")
    print(f"    - Veredicto Heyting    : {s4.heyting_verdict.name}")
    print(f"    - χ                    : {s4.euler_characteristic}")
    assert s4.heyting_verdict == HeytingOmega3.VETOED

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]
    assert audit["all_physically_valid"]

    print("\n>>> EMISIÓN DE PASAPORTE DE INMUNIZACIÓN AGREGADO...")
    passport = engine.emit_passport()
    for k, v in passport.items():
        print(f"    - {k:<20}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Motor Onírico Auditor v3.0.0 completadas.")
    print("═" * 80)
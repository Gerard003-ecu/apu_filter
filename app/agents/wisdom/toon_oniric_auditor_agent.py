# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Oniric Auditor Agent (Soberano Auditor de Escenarios Oníricos)║
║ Ubicación: app/agents/wisdom/toon_oniric_auditor_agent.py                     ║
║ Versión  : 3.0.0-Doctoral-Nested-Ω₃-TQFT-Isolation-Holonomy-Merkle            ║
║ Fases    : FASE-1 → FASE-2 → FASE-3  (anidadas: el último método de k es el   ║
║            germen formal del primero de k+1)                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Formalización categorial
========================

Sea 𝓣_Ω el topos de haces sobre el retículo de Heyting lineal

        Ω₃  =  { VETOED  ≺  DEGRADED  ≺  COHERENT }.

El agente realiza un funtor soberano

        𝒜  :  𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨  ──▶  𝐂𝐞𝐫𝐭_𝐈𝐦𝐦

como composición estrictamente asociativa

        𝒜  =  V ∘ Seal₂ ∘ Seal₁ ∘ Isol ∘ I_GW ∘ D ∘ Spec

donde Spec es el ÚLTIMO morfismo de FASE-1 (extract_spectral_measure) y el
PRIMERO que consume FASE-2; compose_audit_arrows es el ÚLTIMO de FASE-2 y
_seal_and_classify el PRIMERO de FASE-3.

Espectro y amplitudes
=====================

    Spec : ρ ↦ λ ∈ Δ^{n−1}
    γ(ρ) = ‖λ‖₂² = Tr(ρ²)
    S(ρ) = −Σ λᵢ log λᵢ
    E_D  = ½ Σ (λ_{i+1} − λᵢ)²              (Dirichlet / H¹)
    E_∂  = Σ |λ_{i+1} − λᵢ|                 (Dirac / TV)
    I_GW = γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|)

Aislamiento (guardián homológico)
=================================

    dream_verified     := DREAM_STATE_FLAG
    hardware_leak      := ¬dream ∧ (risk > θ_leak)
    fully_isolated     := dream ∧ ¬leak
    Λ_isolation        : 𝔽₂ × 𝔽₂ → Ω₃ ,   (0,·) ↦ ⊥

Holonomía y sello dual
======================

    H₊(t) = ⊕_{τ≤t} I_GW(τ)
    W(t)  = exp(i H₊(t)) ∈ U(1)
    Seal₁ = SHA-256(semántica del payload)
    Seal₂ = SHA-256(agente ‖ id ‖ Seal₁ ‖ t)

Invariantes verificables
========================
    ρ = ρ†, ρ ⪰ 0, Tr ρ = 1.
    I_GW ∈ [0,1], E_D ≥ 0, E_∂ ≥ 0, bₖ ≥ 0.
    is_fully_isolated ≡ dream_verified ∧ ¬hardware_leak.
    Hojas SHA-256 inyectivas; Merkle verificable en O(log n).
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

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditor.v3")

__all__ = [
    "HeytingOmega3",
    "DensityOperator",
    "SpectralMeasure",
    "OniricAuditSeed",
    "OniricScenarioPayload",
    "OniricIsolationCertificate",
    "ImmunizationCertificate",
    "IsolationAuditor",
    "GWInvariantEvaluator",
    "GromovWittenOniricAuditor",
    "OniricIsolationGuard",
    "UnsealedOniricAuditTrace",
    "OniricAuditArrowComposer",
    "MerkleInclusionProof",
    "OniricDreamAuditorAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Ω₃, DENSIDAD, PAYLOAD, CERTIFICADOS Y SEMILLA Spec
# ══════════════════════════════════════════════════════════════════════════════
# Andamiaje del topos 𝓣_Ω. El ÚLTIMO método de esta fase
# (OniricAuditSeed.extract_spectral_measure) es el germen formal de FASE-2:
# GromovWittenOniricAuditor lo realiza y no reconstruye el espectro.
# ══════════════════════════════════════════════════════════════════════════════


class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⊤  si a ≤ b,  else b
        ¬_H a  = a → ⊥

    El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂. DEGRADED viola el tercio excluso,
    de modo que Ω₃ es estrictamente intuicionista. Todo escenario induce
    una flecha característica χ : 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨 → Ω₃.
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

    Invariantes: ρ = ρ†, spec(ρ) ⊂ [−ε, 1+ε], |Tr ρ − 1| ≤ 10⁻⁶.
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
        raw = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (raw + raw.conj().T)
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

    Cierra el contenido informacional de Spec (FASE-1). FASE-2 *continúa*
    exactamente aquí: GromovWittenOniricAuditor consume SpectralMeasure y
    no rediagonaliza ρ salvo petición explícita.
    """

    eigenvalues: np.ndarray
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirichlet_energy: float
    dirac_total_variation: float
    dimension: int
    cstar_residual: float

    def __post_init__(self) -> None:
        lam = np.asarray(self.eigenvalues, dtype=np.float64).reshape(-1)
        if lam.size < 1:
            raise ValueError("SpectralMeasure exige al menos un autovalor.")
        object.__setattr__(self, "eigenvalues", lam)

    def as_simplex(self) -> np.ndarray:
        return self.eigenvalues


@dataclass(frozen=True, slots=True)
class OniricScenarioPayload:
    r"""
    Payload de un escenario onírico contrafactual. Objeto de 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨.

    Invariantes:
      - density_matrix hermítica, PSD, Tr = 1 (se proyecta al simplex).
      - dirichlet_energy ≥ 0, betti_* ≥ 0, risk ∈ [0, 1].
      - scenario_id y synthetic_cartridge_id no vacíos.
    """

    scenario_id: str
    dream_state_flag: bool
    synthetic_cartridge_id: str
    density_matrix: np.ndarray
    dirichlet_energy: float
    betti_1_loop_count: int
    betti_0_components: int
    betti_2_cavities: int
    simulated_risk_factor: float
    timestamp_utc: float

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id no puede ser vacío.")
        if not self.synthetic_cartridge_id:
            raise ValueError("synthetic_cartridge_id no puede ser vacío.")
        if self.dirichlet_energy < 0.0:
            raise ValueError("dirichlet_energy debe ser ≥ 0.")
        if self.betti_1_loop_count < 0 or self.betti_0_components < 0 or self.betti_2_cavities < 0:
            raise ValueError("Los números de Betti deben ser ≥ 0.")
        if not (0.0 <= self.simulated_risk_factor <= 1.0):
            raise ValueError("simulated_risk_factor debe estar en [0, 1].")
        rho = DensityOperator.from_array(self.density_matrix).as_array()
        object.__setattr__(self, "density_matrix", rho)

    def is_quantum_physical(self, atol: float = 1e-9) -> bool:
        rho = self.density_matrix
        if not np.allclose(rho, rho.conj().T, atol=atol):
            return False
        if np.any(la.eigvalsh(rho) < -atol):
            return False
        return abs(float(np.trace(rho).real) - 1.0) < atol

    def euler_characteristic(self) -> int:
        r"""χ = b₀ − b₁ + b₂  (Euler–Poincaré)."""
        return (
            int(self.betti_0_components)
            - int(self.betti_1_loop_count)
            + int(self.betti_2_cavities)
        )

    def topological_class(self) -> str:
        r"""
        Clasificación discreta del 1-esqueleto:
            POINT        : b₀ = 1, b₁ = 0
            ARC          : b₀ ≥ 2, b₁ = 0
            LOOPED_LIGHT : b₁ = 1
            LOOPED_DENSE : b₁ ≥ 2
        """
        if self.betti_1_loop_count == 0:
            return "POINT" if self.betti_0_components == 1 else "ARC"
        if self.betti_1_loop_count == 1:
            return "LOOPED_LIGHT"
        return "LOOPED_DENSE"

    def density_operator(self) -> DensityOperator:
        return DensityOperator(matrix=self.density_matrix)


@dataclass(frozen=True, slots=True)
class OniricIsolationCertificate:
    r"""
    Certificado de aislamiento homológico. Axioma lógico:

        is_fully_isolated  ≡  dream_state_verified  ∧  ¬hardware_leak_risk
    """

    is_fully_isolated: bool
    dream_state_verified: bool
    hardware_leak_risk: bool
    proof_hash: str
    leak_threshold: float

    def logical_consistency(self) -> bool:
        expected = self.dream_state_verified and (not self.hardware_leak_risk)
        return self.is_fully_isolated == expected

    def as_heyting(self) -> HeytingOmega3:
        r"""Λ_isolation : 𝔽₂×𝔽₂ → Ω₃.  ¬dream ↦ ⊥; leak ↦ ⊥; else ⊤."""
        return HeytingOmega3.from_bool(self.is_fully_isolated)


@dataclass(frozen=True, slots=True)
class ImmunizationCertificate:
    r"""
    Objeto terminal del funtor 𝒜 : 𝐒𝐜𝐞𝐧𝐚𝐫𝐢𝐨 → 𝐂𝐞𝐫𝐭_𝐈𝐦𝐦.
    Sello dual: payload_hash (semántica) y digital_signature (procedencia).
    """

    immunization_id: str
    scenario_id: str
    heyting_verdict: HeytingOmega3
    gromov_witten_invariant: float
    tqft_amplitude: float
    purity: float
    von_neumann_entropy: float
    spectral_gap: float
    dirac_total_variation: float
    cstar_residual: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    euler_characteristic: int
    topological_class: str
    isolation_cert: OniricIsolationCertificate
    immunization_payload_hash: str
    digital_signature_sha256: str
    timestamp_utc: float
    holonomy_partial: float
    wilson_phase: complex

    def is_vetoed(self) -> bool:
        return self.heyting_verdict == HeytingOmega3.VETOED

    def is_immune(self) -> bool:
        r"""
        Inmune ⇔ ¬VETOED ∧ aislamiento ∧ Dirichlet válido ∧ consistencia lógica.
        """
        return (
            self.heyting_verdict != HeytingOmega3.VETOED
            and self.isolation_cert.is_fully_isolated
            and self.isolation_cert.logical_consistency()
            and self.dirichlet_bound_valid
        )

    def signature_prefix(self, n: int = 16) -> str:
        return self.digital_signature_sha256[:n]


@runtime_checkable
class IsolationAuditor(Protocol):
    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        ...


@runtime_checkable
class GWInvariantEvaluator(Protocol):
    def compute_gw_invariant(
        self,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        ...


class OniricAuditSeed(ABC):
    r"""
    Germen formal de la flecha Spec : ρ ↦ λ ∈ Δ^{n−1}.

    Cierra el andamiaje de FASE-1. FASE-2 *continúa* exactamente en
    extract_spectral_measure: GromovWittenOniricAuditor lo realiza y
    calcula I_GW, E_D, E_∂ sobre la medida, sin rediagonalizar.
    """

    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        r"""
        Flecha Spec. Produce la medida espectral de ρ.

        CONTINÚA EN FASE-2: GromovWittenOniricAuditor.extract_spectral_measure.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — GW/TQFT, GUARDIÁN DE AISLAMIENTO Y TRAZA ABIERTA
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (extract_spectral_measure) ES la
# realización del último de FASE-1. El último (compose_audit_arrows)
# produce UnsealedOniricAuditTrace, germen formal de FASE-3.
# ══════════════════════════════════════════════════════════════════════════════


class GromovWittenOniricAuditor(OniricAuditSeed):
    r"""
    CONTINUACIÓN FORMAL de OniricAuditSeed.extract_spectral_measure.

    Auditor del invariante de Gromov-Witten / amplitud TQFT:

        I_GW = γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|)

    Si dirichlet_energy es None (forma reducida): I_GW = γ / (1+b₁).
    Si se provee (forma TQFT completa): se atenúa por e^{−E_D} y e^{−S/n}.
    """

    EIGENVALUE_FLOOR: Final[float] = 1e-15

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
        CONTINUACIÓN FORMAL de OniricAuditSeed.extract_spectral_measure.
        Diagonalización hermitiana + proyección al simplex + E_D, E_∂.
        """
        raw = np.asarray(rho, dtype=np.complex128)
        rho_h = 0.5 * (raw + raw.conj().T)
        eigvals = la.eigvalsh(rho_h)
        lam = np.sort(self._project_spectrum(eigvals))
        purity = float(np.sum(lam ** 2))
        entropy = -float(np.sum(lam * np.log(lam)))
        gap = float(lam[-1] - lam[-2]) if lam.size >= 2 else 0.0
        if lam.size < 2:
            ed = 0.0
            tv = 0.0
        else:
            diff = np.diff(lam)
            ed = 0.5 * float(np.sum(diff ** 2))
            tv = float(np.sum(np.abs(diff)))
        op = float(la.norm(rho_h.conj().T @ rho_h, 2))
        nrm = float(la.norm(rho_h, 2))
        return SpectralMeasure(
            eigenvalues=lam,
            purity=purity,
            von_neumann_entropy=entropy,
            spectral_gap=gap,
            dirichlet_energy=ed,
            dirac_total_variation=tv,
            dimension=int(lam.size),
            cstar_residual=abs(op - nrm * nrm),
        )

    @classmethod
    def compute_purity(cls, density_matrix: np.ndarray) -> float:
        eigvals = la.eigvalsh(density_matrix)
        lam = cls._project_spectrum(eigvals)
        return float(np.sum(lam ** 2))

    def compute_gw_invariant(
        self,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        r"""Adaptador Protocol/GWInvariantEvaluator."""
        measure = self.extract_spectral_measure(density_matrix)
        ed = measure.dirichlet_energy if dirichlet_energy is None else float(dirichlet_energy)
        return self.compute_gromov_witten_invariant(
            density_matrix,
            betti_1,
            dirichlet_energy=ed,
            entropy=measure.von_neumann_entropy if entropy == 0.0 else entropy,
            dimension=measure.dimension if dimension == 1 else dimension,
            euler_characteristic=euler_characteristic,
        )

    @classmethod
    def compute_gromov_witten_invariant(
        cls,
        density_matrix: np.ndarray,
        betti_1: int,
        *,
        dirichlet_energy: Optional[float] = None,
        entropy: float = 0.0,
        dimension: int = 1,
        euler_characteristic: int = 1,
    ) -> float:
        r"""
        Forma reducida (dirichlet_energy is None):  γ / (1+b₁).
        Forma TQFT completa: γ · e^{−E_D} · e^{−S/n} / (1+b₁) · (1+χ₊)/(1+|χ|).
        Acotada en [0, 1].
        """
        if betti_1 < 0:
            raise ValueError("b₁ debe ser ≥ 0.")
        gamma = cls.compute_purity(density_matrix)
        denom = 1.0 + float(betti_1)
        if dirichlet_energy is None:
            return float(min(max(gamma / denom, 0.0), 1.0))
        n = max(int(dimension), 1)
        chi_pos = max(int(euler_characteristic), 0)
        chi_abs = abs(int(euler_characteristic))
        euler_factor = (1.0 + chi_pos) / (1.0 + chi_abs)
        entropy_damp = math.exp(-abs(entropy) / n)
        raw = (
            gamma
            * math.exp(-float(dirichlet_energy))
            * entropy_damp
            / denom
            * euler_factor
        )
        return float(min(max(raw, 0.0), 1.0))


class OniricIsolationGuard:
    r"""
    Guardián de aislamiento homológico.

        dream_verified     := payload.dream_state_flag
        hardware_leak_risk := ¬dream_verified ∧ (risk > θ_leak)
        is_fully_isolated  := dream_verified ∧ ¬hardware_leak_risk

    θ_leak = 0.8.  ¬dream implica siempre ¬isolated; el flag de fuga
    hardware es el refinamiento forense (riesgo contractual alto).
    """

    LEAK_RISK_THRESHOLD: Final[float] = 0.8

    def audit_isolation(self, payload: OniricScenarioPayload) -> OniricIsolationCertificate:
        dream_verified = bool(payload.dream_state_flag)
        hardware_leak = (not dream_verified) and (
            payload.simulated_risk_factor > self.LEAK_RISK_THRESHOLD
        )
        is_isolated = dream_verified and (not hardware_leak)

        h = hashlib.sha256()
        payload_bytes = (
            f"{payload.scenario_id}::{payload.dream_state_flag}::"
            f"{is_isolated}::{payload.synthetic_cartridge_id}::"
            f"{payload.simulated_risk_factor:.6f}"
        )
        h.update(payload_bytes.encode("utf-8"))

        cert = OniricIsolationCertificate(
            is_fully_isolated=is_isolated,
            dream_state_verified=dream_verified,
            hardware_leak_risk=hardware_leak,
            proof_hash=h.hexdigest(),
            leak_threshold=self.LEAK_RISK_THRESHOLD,
        )
        if not cert.logical_consistency():
            raise RuntimeError("Violación del axioma de aislamiento homológico.")
        return cert


@dataclass(frozen=True, slots=True)
class UnsealedOniricAuditTrace:
    r"""
    Traza abierta: portadora de (Spec, D, I_GW, Isol) antes de V/Seal.

    Cierra FASE-2. FASE-3 *continúa* exactamente aquí:
    OniricDreamAuditorAgent._seal_and_classify es el primer método de
    FASE-3 y consume esta traza.
    """

    payload: OniricScenarioPayload
    measure: SpectralMeasure
    gromov_witten_invariant: float
    tqft_amplitude: float
    dirichlet_bound_valid: bool
    dirichlet_residual: float
    isolation_cert: OniricIsolationCertificate


class OniricAuditArrowComposer:
    r"""
    Compositor de las flechas Isol ∘ I_GW ∘ D ∘ Spec.

    ÚLTIMO método de FASE-2: compose_audit_arrows.
    CONTINÚA EN FASE-3: OniricDreamAuditorAgent._seal_and_classify.
    """

    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3

    def __init__(
        self,
        spectra: GromovWittenOniricAuditor,
        isolation_auditor: IsolationAuditor,
        energy_threshold: float,
        include_dirichlet_attenuation: bool,
    ) -> None:
        self.spectra = spectra
        self.isolation_auditor = isolation_auditor
        self.energy_threshold = float(energy_threshold)
        self.include_dirichlet_attenuation = bool(include_dirichlet_attenuation)

    def compose_audit_arrows(
        self, payload: OniricScenarioPayload
    ) -> UnsealedOniricAuditTrace:
        r"""
        Realiza Spec, D, I_GW, Isol y emite la traza abierta.

        CONTINÚA EN FASE-3 (clasificación V, sello dual, holonomía).
        """
        measure = self.spectra.extract_spectral_measure(payload.density_matrix)
        ed_residual = abs(payload.dirichlet_energy - measure.dirichlet_energy)
        if ed_residual > self.DIRICHLET_CONSISTENCY_TOL:
            logger.debug(
                "Discrepancia E_D payload vs. espectral: |%.6f − %.6f| = %.3e",
                payload.dirichlet_energy,
                measure.dirichlet_energy,
                ed_residual,
            )

        isolation = self.isolation_auditor.audit_isolation(payload)

        if not isolation.is_fully_isolated:
            return UnsealedOniricAuditTrace(
                payload=payload,
                measure=measure,
                gromov_witten_invariant=0.0,
                tqft_amplitude=0.0,
                dirichlet_bound_valid=False,
                dirichlet_residual=ed_residual,
                isolation_cert=isolation,
            )

        ed_for_gw: Optional[float]
        if self.include_dirichlet_attenuation:
            ed_for_gw = payload.dirichlet_energy
        else:
            ed_for_gw = None

        gw = GromovWittenOniricAuditor.compute_gromov_witten_invariant(
            payload.density_matrix,
            payload.betti_1_loop_count,
            dirichlet_energy=ed_for_gw,
            entropy=measure.von_neumann_entropy,
            dimension=measure.dimension,
            euler_characteristic=payload.euler_characteristic(),
        )
        dirichlet_valid = payload.dirichlet_energy <= self.energy_threshold

        return UnsealedOniricAuditTrace(
            payload=payload,
            measure=measure,
            gromov_witten_invariant=gw,
            tqft_amplitude=gw,
            dirichlet_bound_valid=dirichlet_valid,
            dirichlet_residual=ed_residual,
            isolation_cert=isolation,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SOBERANO AUDITOR, SELLO DUAL, HOLONOMÍA, MERKLE Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════
# Anidación: el primer método operativo (_seal_and_classify) consume
# UnsealedOniricAuditTrace, valor de retorno del último método de FASE-2.
# Aquí se realiza V (meet Ω₃), Seal₁/Seal₂, Hol (⊕ y Wilson) y el registro.
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


class OniricDreamAuditorAgent:
    r"""
    Soberano auditor de escenarios contrafactuales del Onírico.

        𝒜  =  V ∘ Seal₂ ∘ Seal₁ ∘ Isol ∘ I_GW ∘ D ∘ Spec

    realizado como

        audit_dream_scenario
            =  _seal_and_classify  ∘  compose_audit_arrows

    donde compose_audit_arrows es el último método de FASE-2 y
    _seal_and_classify es la continuación formal que abre FASE-3.

    Constantes de calibración (Final):
        energy_threshold = 0.85     (cota superior de E_D admisible)
        gw_tolerance     = 0.15     (umbral inferior de I_GW)
        betti_max        = 3        (bucles máximos tolerados)
        dirac_tv_max     = 1.50     (variación total de Dirac)
    """

    _DEFAULT_ENERGY_THRESHOLD: Final[float] = 0.85
    _DEFAULT_GW_TOLERANCE: Final[float] = 0.15
    _DEFAULT_BETTI_MAX: Final[int] = 3
    _DEFAULT_DIRAC_TV_MAX: Final[float] = 1.50

    def __init__(
        self,
        agent_id: str = "ONIRIC-AUDITOR-SABIO-01",
        energy_threshold: float = _DEFAULT_ENERGY_THRESHOLD,
        gw_tolerance: float = _DEFAULT_GW_TOLERANCE,
        betti_max: int = _DEFAULT_BETTI_MAX,
        dirac_tv_max: float = _DEFAULT_DIRAC_TV_MAX,
        isolation_auditor: Optional[IsolationAuditor] = None,
        gw_evaluator: Optional[GromovWittenOniricAuditor] = None,
        *,
        include_dirichlet_attenuation: bool = True,
    ) -> None:
        if not (0.0 < energy_threshold <= 2.0):
            raise ValueError("energy_threshold debe estar en (0, 2].")
        if not (0.0 <= gw_tolerance <= 1.0):
            raise ValueError("gw_tolerance debe estar en [0, 1].")
        if betti_max < 0:
            raise ValueError("betti_max debe ser ≥ 0.")
        if dirac_tv_max < 0.0:
            raise ValueError("dirac_tv_max debe ser ≥ 0.")

        self.agent_id: str = agent_id
        self.energy_threshold: float = float(energy_threshold)
        self.gw_tolerance: float = float(gw_tolerance)
        self.betti_max: int = int(betti_max)
        self.dirac_tv_max: float = float(dirac_tv_max)
        self._include_dirichlet_attenuation: bool = bool(include_dirichlet_attenuation)

        spectra = gw_evaluator if gw_evaluator is not None else GromovWittenOniricAuditor()
        isolator: IsolationAuditor = (
            isolation_auditor if isolation_auditor is not None else OniricIsolationGuard()
        )
        self.composer: OniricAuditArrowComposer = OniricAuditArrowComposer(
            spectra=spectra,
            isolation_auditor=isolator,
            energy_threshold=self.energy_threshold,
            include_dirichlet_attenuation=self._include_dirichlet_attenuation,
        )
        self._isolation_auditor = isolator
        self._gw_evaluator = spectra

        self.audit_count: int = 0
        self.immunization_registry: List[ImmunizationCertificate] = []
        self._holonomy_accum: float = 0.0

    def _seal_payload(
        self,
        payload: OniricScenarioPayload,
        verdict: HeytingOmega3,
        gw_inv: float,
        dirichlet_valid: bool,
    ) -> str:
        h = hashlib.sha256()
        payload_str = (
            f"{payload.scenario_id}::{payload.synthetic_cartridge_id}::"
            f"{verdict.name}::{gw_inv:.8f}::{dirichlet_valid}::"
            f"{payload.betti_1_loop_count}::{payload.betti_0_components}::"
            f"{payload.betti_2_cavities}"
        )
        h.update(payload_str.encode("utf-8"))
        return h.hexdigest()

    def _seal_signature(
        self, immunization_id: str, payload_hash: str, t_seal: float
    ) -> str:
        h = hashlib.sha256()
        h.update(
            f"{self.agent_id}::{immunization_id}::{payload_hash}::{t_seal:.6f}".encode(
                "utf-8"
            )
        )
        return h.hexdigest()

    def _classify(
        self,
        *,
        betti_1: int,
        gw_inv: float,
        dirichlet_valid: bool,
        isolation: OniricIsolationCertificate,
        dirac_tv: float,
    ) -> HeytingOmega3:
        r"""
        Clasificación Heyting:
            VETOED    si ¬isolated  ∨  b₁ > betti_max  ∨  I_GW < gw_tolerance
            DEGRADED  si ¬dirichlet_valid  ∨  E_∂ > dirac_tv_max
            COHERENT  en otro caso
        """
        if not isolation.is_fully_isolated:
            return HeytingOmega3.VETOED
        if betti_1 > self.betti_max or gw_inv < self.gw_tolerance:
            return HeytingOmega3.VETOED
        if not dirichlet_valid or dirac_tv > self.dirac_tv_max:
            return HeytingOmega3.DEGRADED
        return HeytingOmega3.COHERENT

    def _seal_and_classify(
        self, trace: UnsealedOniricAuditTrace
    ) -> ImmunizationCertificate:
        r"""
        CONTINUACIÓN FORMAL de OniricAuditArrowComposer.compose_audit_arrows.

        Aplica V (meet Ω₃), Seal₁/Seal₂, Hol (⊕ I_GW y Wilson) y construye
        el objeto de 𝐂𝐞𝐫𝐭_𝐈𝐦𝐦.
        """
        payload = trace.payload
        isolation = trace.isolation_cert
        measure = trace.measure

        verdict = self._classify(
            betti_1=payload.betti_1_loop_count,
            gw_inv=trace.gromov_witten_invariant,
            dirichlet_valid=trace.dirichlet_bound_valid,
            isolation=isolation,
            dirac_tv=measure.dirac_total_variation,
        )

        imm_id = f"IMM-ONIRIC-{self.audit_count:04d}"
        t_seal = time.time()
        payload_hash = self._seal_payload(
            payload, verdict, trace.gromov_witten_invariant, trace.dirichlet_bound_valid
        )
        signature = self._seal_signature(imm_id, payload_hash, t_seal)

        self._holonomy_accum += trace.gromov_witten_invariant
        wilson = complex(
            math.cos(self._holonomy_accum), math.sin(self._holonomy_accum)
        )

        return ImmunizationCertificate(
            immunization_id=imm_id,
            scenario_id=payload.scenario_id,
            heyting_verdict=verdict,
            gromov_witten_invariant=float(trace.gromov_witten_invariant),
            tqft_amplitude=float(trace.tqft_amplitude),
            purity=measure.purity,
            von_neumann_entropy=measure.von_neumann_entropy,
            spectral_gap=measure.spectral_gap,
            dirac_total_variation=measure.dirac_total_variation,
            cstar_residual=measure.cstar_residual,
            dirichlet_bound_valid=bool(trace.dirichlet_bound_valid),
            dirichlet_residual=float(trace.dirichlet_residual),
            euler_characteristic=payload.euler_characteristic(),
            topological_class=payload.topological_class(),
            isolation_cert=isolation,
            immunization_payload_hash=payload_hash,
            digital_signature_sha256=signature,
            timestamp_utc=t_seal,
            holonomy_partial=self._holonomy_accum,
            wilson_phase=wilson,
        )

    def audit_dream_scenario(
        self, payload: OniricScenarioPayload
    ) -> ImmunizationCertificate:
        r"""
        Audita un escenario onírico completo.

        Pasos anidados:
          1. Identidad de auditoría.
          2. compose_audit_arrows (FASE-2: Spec, D, I_GW, Isol)
             → UnsealedOniricAuditTrace.
          3. _seal_and_classify (FASE-3: V, Seal dual, Hol)
             → ImmunizationCertificate.
          4. Persistencia inmutable.
        """
        self.audit_count += 1
        t_start = time.time()
        logger.info(
            "=== Auditando Escenario Onírico #%d | ID: %s ===",
            self.audit_count,
            payload.scenario_id,
        )

        trace = self.composer.compose_audit_arrows(payload)

        if not trace.isolation_cert.is_fully_isolated:
            logger.critical(
                "¡CRÍTICO! Fuga de aislamiento en escenario onírico %s "
                "(dream=%s, leak=%s)",
                payload.scenario_id,
                trace.isolation_cert.dream_state_verified,
                trace.isolation_cert.hardware_leak_risk,
            )

        cert = self._seal_and_classify(trace)
        self.immunization_registry.append(cert)

        logger.info(
            "Auditoría Onírica Finalizada en %.2f ms | Veredicto: %s | "
            "I_GW=%.6f | γ=%.4f | E_D=%.4f | E_∂=%.4f | χ=%d | clase=%s | "
            "H₊=%.6f | inmune=%s",
            (time.time() - t_start) * 1000.0,
            cert.heyting_verdict.name,
            cert.gromov_witten_invariant,
            cert.purity,
            payload.dirichlet_energy,
            cert.dirac_total_variation,
            cert.euler_characteristic,
            cert.topological_class,
            self._holonomy_accum,
            cert.is_immune(),
        )
        return cert

    @property
    def registry(self) -> Tuple[ImmunizationCertificate, ...]:
        return tuple(self.immunization_registry)

    @property
    def holonomy_accum(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
        gv = HeytingOmega3.COHERENT
        for c in self.immunization_registry:
            gv = gv.meet(c.heyting_verdict)
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
    def _merkle_proof(
        leaf_hashes: Sequence[str], index: int
    ) -> MerkleInclusionProof:
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
        return self._merkle_tree_root(
            [c.digital_signature_sha256 for c in self.immunization_registry]
        )

    def merkle_proofs_ok(self) -> bool:
        leaves = [c.digital_signature_sha256 for c in self.immunization_registry]
        root = self._merkle_tree_root(leaves)
        for i in range(len(leaves)):
            proof = self._merkle_proof(leaves, i)
            if proof.root != root or not proof.verify():
                return False
        return True

    def audit_registry(self) -> Dict[str, Any]:
        r"""
        Auditoría retrospectiva:
            n_certificates, verdict_distribution, global_verdict,
            holonomy_accum, wilson_loop, avg_gw_invariant,
            avg_dirichlet_energy (E_D del payload no se retiene: se usa
            el residuo y la cota), avg_purity, n_immune,
            n_hardware_leak_risk, registry_integrity_ok, merkle_proofs_ok,
            isolation_axioms_ok.
        """
        n = len(self.immunization_registry)
        empty_dist = {v.name: 0 for v in HeytingOmega3}
        if n == 0:
            return {
                "n_certificates": 0,
                "verdict_distribution": empty_dist,
                "global_verdict": HeytingOmega3.COHERENT.name,
                "holonomy_accum": 0.0,
                "wilson_loop": 1.0 + 0.0j,
                "avg_gw_invariant": 0.0,
                "avg_purity": 0.0,
                "avg_dirac_tv": 0.0,
                "avg_dirichlet_residual": 0.0,
                "n_immune": 0,
                "n_hardware_leak_risk": 0,
                "registry_integrity_ok": True,
                "merkle_proofs_ok": True,
                "isolation_axioms_ok": True,
            }

        dist: Dict[str, int] = {v.name: 0 for v in HeytingOmega3}
        total_gw = total_p = total_tv = total_ed_res = 0.0
        immune_count = leak_count = 0
        signatures: Set[str] = set()
        collide = False
        axioms_ok = True
        for c in self.immunization_registry:
            dist[c.heyting_verdict.name] += 1
            total_gw += c.gromov_witten_invariant
            total_p += c.purity
            total_tv += c.dirac_total_variation
            total_ed_res += c.dirichlet_residual
            if c.isolation_cert.hardware_leak_risk:
                leak_count += 1
            if c.is_immune():
                immune_count += 1
            if not c.isolation_cert.logical_consistency():
                axioms_ok = False
            if c.digital_signature_sha256 in signatures:
                collide = True
            signatures.add(c.digital_signature_sha256)

        inv = 1.0 / n
        return {
            "n_certificates": n,
            "verdict_distribution": dist,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "avg_gw_invariant": total_gw * inv,
            "avg_purity": total_p * inv,
            "avg_dirac_tv": total_tv * inv,
            "avg_dirichlet_residual": total_ed_res * inv,
            "n_immune": immune_count,
            "n_hardware_leak_risk": leak_count,
            "registry_integrity_ok": not collide,
            "merkle_proofs_ok": self.merkle_proofs_ok(),
            "isolation_axioms_ok": axioms_ok,
        }

    def emit_immunization_passport(self) -> Dict[str, Any]:
        r"""
        Pasaporte agregado consumible por GodelAgent / TOONWisdomWeaver.
        evidence_hash encadena agent_id, H₊ y las firmas individuales.
        """
        h = hashlib.sha256()
        h.update(
            f"{self.agent_id}::{self.audit_count}::{self._holonomy_accum:.10f}".encode(
                "utf-8"
            )
        )
        for c in self.immunization_registry:
            h.update(c.digital_signature_sha256.encode("utf-8"))
        return {
            "agent_id": self.agent_id,
            "registry_size": self.audit_count,
            "global_verdict": self.global_verdict.name,
            "holonomy_accum": self._holonomy_accum,
            "wilson_loop": self.wilson_loop,
            "n_immune": sum(1 for c in self.immunization_registry if c.is_immune()),
            "merkle_root": self.merkle_root(),
            "evidence_hash": h.hexdigest(),
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
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Agent v3.0.0")
    print("FASES ANIDADAS: Ω₃+Spec → GW/Isol+Traza → V/Sello dual/Holonomía/Merkle")
    print("═" * 80)

    print("\n[§0] VERIFICACIÓN FORMAL DE Ω₃")
    assert HeytingOmega3.verify_residuation_axiom()
    assert HeytingOmega3.DEGRADED.excluded_middle_holds() is False
    assert HeytingOmega3.COHERENT.excluded_middle_holds() is True
    assert HeytingOmega3.VETOED.is_regular() is True
    assert HeytingOmega3.DEGRADED.is_regular() is False
    print("  • Residuación, tercio excluso y regularidad: OK")

    rng = np.random.default_rng(20250321)
    auditor = OniricDreamAuditorAgent(
        agent_id="ONIRIC-AUDITOR-SABIO-01",
        include_dirichlet_attenuation=True,
    )

    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real

    print("\n>>> ESCENARIO 1: Cisne Negro plausible (aislado, E_D moderada)...")
    s1 = OniricScenarioPayload(
        scenario_id="DREAM-BLACK-SWAN-001",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-SYNTH-STRESS-01",
        density_matrix=rho,
        dirichlet_energy=0.42,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.75,
        timestamp_utc=time.time(),
    )
    cert1 = auditor.audit_dream_scenario(s1)
    print(f"    - ID Certificado       : {cert1.immunization_id}")
    print(f"    - Veredicto Heyting    : {cert1.heyting_verdict.name}")
    print(f"    - I_GW / TQFT          : {cert1.gromov_witten_invariant:.6f}")
    print(f"    - γ / S_vN / Δλ        : {cert1.purity:.4f} / {cert1.von_neumann_entropy:.4f} / {cert1.spectral_gap:.4f}")
    print(f"    - E_∂ / residual C*    : {cert1.dirac_total_variation:.6f} / {cert1.cstar_residual:.3e}")
    print(f"    - Dirichlet válido     : {cert1.dirichlet_bound_valid}")
    print(f"    - Aislamiento completo : {cert1.isolation_cert.is_fully_isolated}")
    print(f"    - Axioma aislamiento   : {cert1.isolation_cert.logical_consistency()}")
    print(f"    - Inmune               : {cert1.is_immune()}")
    print(f"    - χ / clase            : {cert1.euler_characteristic} / {cert1.topological_class}")
    print(f"    - Wilson W             : {cert1.wilson_phase:.4f}")
    print(f"    - Firma SHA-256        : {cert1.signature_prefix(32)}...")
    assert s1.is_quantum_physical()
    assert cert1.isolation_cert.logical_consistency()

    print("\n>>> ESCENARIO 2: Fuga de aislamiento (dream=False, risk=0.95)...")
    rho_leak = np.eye(4, dtype=np.complex128) / 4.0
    s2 = OniricScenarioPayload(
        scenario_id="DREAM-LEAK-002",
        dream_state_flag=False,
        synthetic_cartridge_id="CARTRIDGE-LEAK-02",
        density_matrix=rho_leak,
        dirichlet_energy=0.20,
        betti_1_loop_count=0,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.95,
        timestamp_utc=time.time(),
    )
    cert2 = auditor.audit_dream_scenario(s2)
    print(f"    - ID Certificado       : {cert2.immunization_id}")
    print(f"    - Veredicto Heyting    : {cert2.heyting_verdict.name}")
    print(f"    - Hardware leak risk   : {cert2.isolation_cert.hardware_leak_risk}")
    print(f"    - Aislamiento completo : {cert2.isolation_cert.is_fully_isolated}")
    print(f"    - Inmune               : {cert2.is_immune()}")
    assert cert2.heyting_verdict == HeytingOmega3.VETOED
    assert cert2.is_immune() is False
    assert cert2.isolation_cert.hardware_leak_risk is True

    print("\n>>> ESCENARIO 3: Topología sintáctica compleja (b₁ = 5)...")
    s3 = OniricScenarioPayload(
        scenario_id="DREAM-COMPLEX-003",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-COMPLEX-03",
        density_matrix=rho,
        dirichlet_energy=0.30,
        betti_1_loop_count=5,
        betti_0_components=2,
        betti_2_cavities=1,
        simulated_risk_factor=0.30,
        timestamp_utc=time.time(),
    )
    cert3 = auditor.audit_dream_scenario(s3)
    print(f"    - ID Certificado       : {cert3.immunization_id}")
    print(f"    - Veredicto Heyting    : {cert3.heyting_verdict.name}")
    print(f"    - b₁ / χ               : {s3.betti_1_loop_count} / {s3.euler_characteristic()}")
    print(f"    - Clase topológica     : {s3.topological_class()}")
    assert cert3.heyting_verdict == HeytingOmega3.VETOED
    assert s3.euler_characteristic() == 2 - 5 + 1

    print("\n>>> ESCENARIO 4: E_D elevada (borde Dirichlet violado → DEGRADED)...")
    s4 = OniricScenarioPayload(
        scenario_id="DREAM-HIGH-ED-004",
        dream_state_flag=True,
        synthetic_cartridge_id="CARTRIDGE-ED-04",
        density_matrix=rho,
        dirichlet_energy=0.92,
        betti_1_loop_count=1,
        betti_0_components=1,
        betti_2_cavities=0,
        simulated_risk_factor=0.40,
        timestamp_utc=time.time(),
    )
    cert4 = auditor.audit_dream_scenario(s4)
    print(f"    - ID Certificado       : {cert4.immunization_id}")
    print(f"    - Veredicto Heyting    : {cert4.heyting_verdict.name}")
    print(f"    - Dirichlet válido     : {cert4.dirichlet_bound_valid}")
    print(f"    - Clase topológica     : {s4.topological_class()}")
    assert cert4.heyting_verdict == HeytingOmega3.DEGRADED
    assert cert4.dirichlet_bound_valid is False

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = auditor.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]
    assert audit["isolation_axioms_ok"]
    assert audit["n_hardware_leak_risk"] >= 1

    print("\n>>> PASAPORTE AGREGADO DE INMUNIZACIÓN...")
    passport = auditor.emit_immunization_passport()
    for k, v in passport.items():
        print(f"    - {k:<20}: {v}")

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Soberano Auditor Onírico v3.0.0 completadas.")
    print("═" * 80)
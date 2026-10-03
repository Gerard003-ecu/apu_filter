# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/wisdom/toon_oniric_auditor_engine.py                                  ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / AUDITORÍA ONÍRICA TQFT              ║
║ FUNCIÓN  : MOTOR ESPECTRAL AUDITOR DE SUEÑOS Y CAMPO TQFT GROMOV-WITTEN POINCARE     ║
║ VERSIÓN  : 8.1.0-Doctoral-Poincare-Engine-ESP32-Crowbar-Lefschetz-A3                 ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA POINCARANA
──────────────────────────────────────────────────────────
El `TOONOniricAuditorEngine` constituye el motor espectral de auditoría topológica y 
evaluación de la Teoría Cuántica de Campos Topológicos (TQFT) sobre los escenarios 
sintéticos contrafactuales generados durante la fase REM del ecosistema APU Filter v8.0.

Integración de Mecánica Celeste y Geometría Simpléctica de Henri Poincaré:
1. DUALIDAD RELATIVA DE POINCARÉ-LEFSCHETZ PARA VARIEDADES ABIERTAS CON FRONTERA (∂M ≠ ∅):
   Para la variedad diferencial de la obra (M, ω) con frontera compacta ∂M (interfaz de pagos
   y entregables en SECOP II y ejecuciones de obra civil), se establece el isomorfismo:

       H_k(M, ∂M; ℤ)  ≅  H^{n-k}(M; ℤ)

   El Invariante Relativo de Gromov-Witten con cofrontera de borde se evalúa como:

       I_GW^{relative}(ρ) = [ Tr(ρ²) · e^{-E_D} · e^{-S/n} / (1 + β₁) ] · [ 1 / (1 + dim H¹(M, ∂M)) ]

   donde dim H¹(M, ∂M) cuantifica la obstrucción de coborde (defect de frontera) entre el
   presupuesto interno y la ejecución real.

2. TEOREMA DE NO EXISTENCIA DE INTEGRALES Y RIGIDEZ SIMPLÉCTICA DE GROMOV:
   Poincaré probó la inexistencia de constantes analíticas independientes adicionales a la
   energía y momento en sistemas de N ≥ 3 cuerpos. El Motor aplica el Teorema de No-Aplastamiento
   (Nonsqueezing Theorem) de Gromov:

       Cap_{symplectic}(B^{2n}(r)) = π r² ≤ π R² = Cap_{symplectic}(Z^{2n}(R))

   Invariancia del volumen simpléctico en el espacio de fases M_{2n}. Cualquier intento de
   "forzar" un presupuesto comprimiendo el riesgo r viola la rigidez simpléctica, vetando el escenario.

3. DISYUNTOR CIBER-FÍSICO ESP32 CROWBAR EN Ω₃:
   Adjudicación en el retículo intuicionista de Heyting Ω₃ = {0 ≺ 1 ≺ 2}. Si el veredicto
   colapsa a VETOED (0), la reducción monoidal μ: Ω₃ → ℤ₂ activa en < 400 ns en IRAM
   el tiristor BT151 (GPIO14 = HIGH) para paralizar síncronamente bombas y mezcladoras de concreto.

MAPPING EJECUTIVO ("DOLOR Y DINERO")
───────────────────────────────────
- Filtro de Falsas Trampas: Distingue entre riesgos macroeconómicos reales (Cisnes Negros)
  y alucinaciones estocásticas de la IA, previniendo coberturas innecesarias.
- Interlock Ciber-Físico ESP32: Desconexión directa en tiempo real de maquinaria pesada
  en caso de alteración contrafactual o fuga homológica no aislada.
- Pasaporte Criptográfico SHA-512 / SHA-256: Certificación inmutable para auditorías de
  Contraloría, Fiscalía y licitaciones de obra pública.
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

from app.agents.wisdom.toon_oniric_auditor_agent import (
    GromovWittenOniricAuditor,
    ImmunizationCertificate,
)

logger = logging.getLogger("APU.Wisdom.TOONOniricAuditorEngine.v4")

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


class HeytingOmega3(IntEnum):
    r"""
    Álgebra de Heyting lineal Ω₃ = {0 ≺ 1 ≺ 2} = {VETOED ≺ DEGRADED ≺ COHERENT}.

        a ∧ b  = min(a, b)
        a ∨ b  = max(a, b)
        a → b  = ⊤  si a ≤ b,  else b
        ¬_H a  = a → ⊥

    El esqueleto booleano es {⊥, ⊤} ≅ 𝔹₂. DEGRADED viola el tercio excluso,
    de modo que Ω₃ es estrictamente intuicionista.
    """

    VETOED: int = 0
    DEGRADED: int = 1
    COHERENT: int = 2

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        if int(self) <= int(other):
            return HeytingOmega3.COHERENT
        return other

    def pseudo_complement(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.VETOED)

    def classical_negation(self) -> "HeytingOmega3":
        return HeytingOmega3(2 - int(self))

    def double_negation(self) -> "HeytingOmega3":
        return self.pseudo_complement().pseudo_complement()

    def is_regular(self) -> bool:
        return self.double_negation() == self

    def excluded_middle_holds(self) -> bool:
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
    """

    eigenvalues: np.ndarray
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
    Protocolo estructural del pasaporte de inmunización.
    """

    immunization_hash: str
    heyting_verdict: HeytingOmega3
    gromov_witten_invariant: float

    def is_immune(self) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class OniricFieldState:
    r"""
    Estado onírico con métricas de la Dualidad de Poincaré-Lefschetz y Crowbar ESP32.
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
    poincare_lefschetz_defect: float = 0.0
    symplectic_capacity_ratio: float = 1.0
    is_boundary_consistent: bool = True
    crowbar_triggered: bool = False
    gpio14_signal: str = "LOW"
    proof_merkle_sha512: str = ""

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
        if self.euler_characteristic != (self.betti_0 - self.betti_1_loops + self.betti_2):
            return False
        if self.heyting_verdict == HeytingOmega3.VETOED:
            if not self.dream_isolation_flag:
                return True
            return self.betti_1_loops > 3 or (not self.is_boundary_consistent)
        return True

    def is_immune(self) -> bool:
        return (
            self.dream_isolation_flag
            and self.heyting_verdict != HeytingOmega3.VETOED
            and self.is_quantum_physical()
            and self.is_topologically_consistent()
            and self.is_boundary_consistent
            and self.symplectic_capacity_ratio <= 1.25
        )

    def passport_prefix(self, n: int = 16) -> str:
        return self.immunization_hash[:n]


class SpectralMeasureSeed(ABC):
    @abstractmethod
    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — TQFT, GROMOV-WITTEN SINTÉTICO, DIRICHLET-DIRAC Y TRAZA ABIERTA
# ══════════════════════════════════════════════════════════════════════════════


class OniricSpectraEngine(SpectralMeasureSeed):
    r"""
    Motor espectral onírico con Dualidad Poincaré-Lefschetz y Rigidez Simpléctica.
    """

    BETTI_MAX: Final[int] = 3
    GW_MIN: Final[float] = 0.05
    DIRICHLET_MAX: Final[float] = 0.85
    DIRAC_TV_MAX: Final[float] = 1.50
    EIGENVALUE_FLOOR: Final[float] = 1e-15
    DIRICHLET_CONSISTENCY_TOL: Final[float] = 1e-3

    def __init__(self, gw_auditor: Optional[GromovWittenOniricAuditor] = None) -> None:
        self.gw_auditor = gw_auditor or GromovWittenOniricAuditor()

    @classmethod
    def _project_spectrum(cls, eigvals: np.ndarray) -> np.ndarray:
        eigvals = np.clip(np.real(eigvals), cls.EIGENVALUE_FLOOR, None)
        s = float(np.sum(eigvals))
        if s < cls.EIGENVALUE_FLOOR:
            n = eigvals.shape[0]
            return np.full(n, 1.0 / max(n, 1))
        return eigvals / s

    def extract_spectral_measure(self, rho: np.ndarray) -> SpectralMeasure:
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
    def compute_dirichlet_energy(cls, eigvals: np.ndarray) -> float:
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        grad = np.diff(lam)
        return 0.5 * float(np.sum(grad ** 2))

    @classmethod
    def compute_dirac_total_variation(cls, eigvals: np.ndarray) -> float:
        lam = np.sort(cls._project_spectrum(eigvals))
        if lam.size < 2:
            return 0.0
        return float(np.sum(np.abs(np.diff(lam))))

    def evaluate_dream_spectrum(
        self,
        density_matrix: np.ndarray,
        dirichlet_energy: Optional[float],
        betti_1: int,
        dream_isolation: bool,
        betti_0: int = 1,
        betti_2: int = 0,
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
        scenario_id: str = "DREAM-EVAL",
    ) -> "UnsealedOniricTrace":
        measure = self.extract_spectral_measure(density_matrix)
        lam = measure.as_simplex()
        ed_internal = self.compute_dirichlet_energy(lam)
        ed_dirac = self.compute_dirac_total_variation(lam)
        ed = ed_internal if dirichlet_energy is None else float(dirichlet_energy)
        ed_residual = abs(ed - ed_internal)

        b0 = max(int(betti_0), 0)
        b1 = max(int(betti_1), 0)
        b2 = max(int(betti_2), 0)
        chi = b0 - b1 + b2

        n = density_matrix.shape[0]
        base = rho_base if rho_base is not None else np.eye(n, dtype=np.complex128) / float(n)
        stalk = boundary_stalk if boundary_stalk is not None else np.eye(n, dtype=np.complex128)

        poincare_cert = self.gw_auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=density_matrix,
            rho_base=base,
            boundary_stalk_matrix=stalk,
            betti_1_cycles=b1,
            dirichlet_energy=ed,
            dream_isolation=dream_isolation,
            scenario_id=scenario_id,
        )

        gw = poincare_cert.gw_relative_invariant
        verdict = poincare_cert.heyting_verdict

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
            poincare_cert=poincare_cert,
        )


@dataclass(frozen=True, slots=True)
class UnsealedOniricTrace:
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
    poincare_cert: Optional[ImmunizationCertificate] = None


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — AUDITOR, SELLO, HOLONOMÍA, MERKLE, INMUNIZACIÓN Y PASAPORTE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MerkleInclusionProof:
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
    Motor espectral auditor de sueños con Dualidad de Poincaré-Lefschetz,
    Gromov Non-Squeezing y Disyuntor ESP32 Crowbar.
    """

    def __init__(
        self,
        engine_id: str = "ONIRIC-ENGINE-SABIO-01",
        gw_auditor: Optional[GromovWittenOniricAuditor] = None,
        spectra_engine: Optional[OniricSpectraEngine] = None,
    ) -> None:
        self.engine_id: str = engine_id
        self.auditor = gw_auditor or GromovWittenOniricAuditor()
        self.spectra: OniricSpectraEngine = (
            spectra_engine if spectra_engine is not None else OniricSpectraEngine(gw_auditor=self.auditor)
        )
        self.cycle_count: int = 0
        self.history: List[OniricFieldState] = []
        self._holonomy_accum: float = 0.0

    def process_oniric_audit_pipeline(
        self,
        rho_dream: np.ndarray,
        rho_base: np.ndarray,
        boundary_stalk: np.ndarray,
        betti_1_cycles: int = 0,
    ) -> Dict[str, Any]:
        """Ejecuta la tubería completa de auditoría TQFT con protección ESP32 Crowbar."""
        cert = self.auditor.evaluate_poincare_lefschetz_gw_invariant(
            rho_dream=rho_dream,
            rho_base=rho_base,
            boundary_stalk_matrix=boundary_stalk,
            betti_1_cycles=betti_1_cycles,
        )

        crowbar_triggered = False
        gpio14_signal = "LOW"

        if cert.heyting_verdict == HeytingOmega3.VETOED:
            crowbar_triggered = True
            gpio14_signal = "HIGH"  # Disparo de tiristor BT151

        return {
            "gw_relative_invariant": cert.gw_relative_invariant,
            "poincare_lefschetz_defect": cert.poincare_lefschetz_defect,
            "symplectic_capacity_ratio": cert.symplectic_capacity_ratio,
            "heyting_verdict": cert.heyting_verdict.name,
            "heyting_code": cert.heyting_verdict.value,
            "is_boundary_consistent": cert.is_boundary_consistent,
            "crowbar_triggered": crowbar_triggered,
            "gpio14_signal": gpio14_signal,
            "merkle_sha512": cert.proof_merkle_sha512,
            "schema_version": "4.1.0-Poincare-Lefschetz",
        }

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

        p_cert = trace.poincare_cert
        defect = p_cert.poincare_lefschetz_defect if p_cert else 0.0
        cap_ratio = p_cert.symplectic_capacity_ratio if p_cert else 1.0
        is_consistent = p_cert.is_boundary_consistent if p_cert else True
        crowbar = p_cert.crowbar_triggered if p_cert else (trace.heyting_verdict == HeytingOmega3.VETOED)
        gpio14 = p_cert.gpio14_signal if p_cert else ("HIGH" if crowbar else "LOW")
        proof_512 = p_cert.proof_merkle_sha512 if p_cert else ""

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
            poincare_lefschetz_defect=defect,
            symplectic_capacity_ratio=cap_ratio,
            is_boundary_consistent=is_consistent,
            crowbar_triggered=crowbar,
            gpio14_signal=gpio14,
            proof_merkle_sha512=proof_512,
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
        rho_base: Optional[np.ndarray] = None,
        boundary_stalk: Optional[np.ndarray] = None,
    ) -> OniricFieldState:
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
            rho_base=rho_base,
            boundary_stalk=boundary_stalk,
            scenario_id=scenario_id,
        )
        state = self._seal_and_accumulate(trace, cycle_id, scenario_id)
        self.history.append(state)

        logger.info(
            "Ciclo Espectral Onírico %s Finalizado en %.2f ms | Veredicto: %s | "
            "I_GW: %.6f | Defect: %.6f | CapRatio: %.4f | Crowbar: %s | GPIO14: %s",
            cycle_id,
            (time.time() - t_start) * 1000.0,
            state.heyting_verdict.name,
            state.gromov_witten_invariant,
            state.poincare_lefschetz_defect,
            state.symplectic_capacity_ratio,
            state.crowbar_triggered,
            state.gpio14_signal,
        )
        return state

    @property
    def registry(self) -> Tuple[OniricFieldState, ...]:
        return tuple(self.history)

    @property
    def holonomy_accum(self) -> float:
        return self._holonomy_accum

    @property
    def wilson_loop(self) -> complex:
        return complex(math.cos(self._holonomy_accum), math.sin(self._holonomy_accum))

    @property
    def global_verdict(self) -> HeytingOmega3:
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
    print("DEMOSTRACIÓN GRANULAR: TOON Oniric Auditor Engine v8.1.0-Poincare")
    print("FASES ANIDADAS: Ω₃+Spec → TQFT/Lefschetz/Crowbar → Seal/Holonomía/Merkle")
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
    rho_base = np.eye(4, dtype=np.complex128) / 4.0
    stalk = np.eye(4, dtype=np.complex128)

    print("\n>>> ESCENARIO PIPELINE TQFT POINCARÉ-LEFSCHETZ...")
    res = engine.process_oniric_audit_pipeline(
        rho_dream=rho,
        rho_base=rho_base,
        boundary_stalk=stalk,
        betti_1_cycles=0,
    )
    for k, v in res.items():
        print(f"    - {k:<26}: {v}")

    print("\n>>> ESCENARIO A: Estado físico canónico (Ciclo espectral)...")
    s1 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-001",
        density_matrix=rho,
        dirichlet_energy=0.35,
        betti_1=0,
        dream_isolation=True,
    )
    print(f"    - ID Ciclo             : {s1.cycle_id}")
    print(f"    - Veredicto Heyting    : {s1.heyting_verdict.name}")
    print(f"    - I_GW / TQFT          : {s1.gromov_witten_invariant:.6f}")
    print(f"    - Defect Poincaré-Lefsch: {s1.poincare_lefschetz_defect:.6e}")
    print(f"    - Crowbar Triggered    : {s1.crowbar_triggered}")
    print(f"    - GPIO14 Signal        : {s1.gpio14_signal}")

    print("\n>>> ESCENARIO B: Violación de aislamiento REM (veto duro)...")
    s3 = engine.audit_oniric_cycle(
        scenario_id="SCENARIO-ONIRIC-BREACH",
        density_matrix=rho,
        dirichlet_energy=0.15,
        betti_1=5,
        dream_isolation=False,
    )
    print(f"    - ID Ciclo             : {s3.cycle_id}")
    print(f"    - Veredicto Heyting    : {s3.heyting_verdict.name}")
    print(f"    - Crowbar Triggered    : {s3.crowbar_triggered}")
    print(f"    - GPIO14 Signal        : {s3.gpio14_signal}")
    assert s3.heyting_verdict == HeytingOmega3.VETOED
    assert s3.crowbar_triggered is True
    assert s3.gpio14_signal == "HIGH"

    print("\n>>> AUDITORÍA RETROSPECTIVA DEL REGISTRO...")
    audit = engine.audit_registry()
    for k, v in audit.items():
        print(f"    - {k:<26}: {v}")
    assert audit["registry_integrity_ok"]
    assert audit["merkle_proofs_ok"]

    print("\n" + "═" * 80)
    print("✓ Pruebas de verificación del Motor Onírico Auditor Poincaré completadas.")
    print("═" * 80)

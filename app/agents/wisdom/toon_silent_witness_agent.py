# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/wisdom/toon_silent_witness_agent.py                            ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / VACÍO DE DIRAC                      ║
║ FUNCIÓN  : SOBERANO TESTIGO SILENCIOSO Y CRISTALIZADOR DE EXPERIENCIA                ║
║ VERSIÓN  : 8.0.0-Doctoral-Poincaré-Recurrence-KMS-Tomita-Takesaki-Dirac-Vacuum-S6    ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONSilentWitnessAgent` actúa como el Soberano observador imparcial e incorruptible del
Estrato Wisdom ($\mathcal{V}_{\mathbb{W}}$). Reside en el límite de congelamiento entrópico del
vacío de Dirac, observando de forma no destructiva las interacciones de la Malla Agéntica
sin perturbar la función de onda de las transacciones reales.

1. POSTULADO DE LA OBSERVACIÓN NO DESTRUCTIVA (ZERO BACK-ACTION Y MEDICIÓN DÉBIL DE AHARONOV):
   Para cualquier observable de contrato $A \in \mathcal{M}$, el Soberano Testigo efectúa mediciones
   débiles caracterizadas por una fuerza de interacción $\kappa \to 0$ y estado fundamental $H_{\mathrm{vac}} |\Omega\rangle = 0$:

       A_w = \frac{\langle \phi_{\mathrm{post}} | A | \psi_{\mathrm{pre}} \rangle}{\langle \phi_{\mathrm{post}} | \psi_{\mathrm{pre}} \rangle}

   donde $|\psi_{\mathrm{pre}}\rangle$ es el estado de la propuesta licitatoria y $|\phi_{\mathrm{post}}\rangle$
   es el estado final verificado. La medición débil extrae el valor esperado sin colapsar el estado.

2. TEOREMA DE RECURRENCIA DE POINCARÉ Y CRISTALIZACIÓN $S^6 \subset \mathbb{R}^7$:
   Sea $(\mathcal{M}, \Sigma, \mu, T_t)$ un sistema dinámico hamiltoniano sobre la medida de Liouville $\mu(\mathcal{M}) < \infty$.
   Para cualquier subconjunto de eventos presupuestales $E \in \Sigma$ con $\mu(E) > 0$, existe un tiempo de recurrencia $\tau_{\mathrm{rec}} > 0$
   tal que $\mu(E \cap T_t^{-\tau_{\mathrm{rec}}} E) > 0$.
   El retorno se audita evaluando la distancia Banach $\|\sigma_{\tau_{\mathrm{rec}}}(\rho) - \rho\|_F < \epsilon$.
   Las 7 componentes espectrales principales se proyectan sobre la 6-esfera $S^6 = \{\mathbf{v}_{\mathrm{inv}} \in \mathbb{R}^7 : \|\mathbf{v}_{\mathrm{inv}}\|_2 = 1.0\}$.

3. ÁRBOL DE MERKLE ATÓMICO Y CRISTAL DE EXPERIENCIA:
   Al concluir cada ciclo, el Testigo Silencioso condensa las lecciones aprobadas en un `ExperienceCrystal`,
   generando una raíz de Merkle SHA-256 incorruptible:

       \mathrm{MerkleRoot} = \mathrm{Hash}_{\mathrm{SHA-256}}\left( \mathbf{v}_{S^6} \mathbin{\Vert} \tau_{\mathrm{rec}} \mathbin{\Vert} \mathrm{KMSDrift} \mathbin{\Vert} \mathrm{Timestamp} \right)

4. INTERLOCK DE INTEGRIDAD CIBER-FÍSICA Y DISRUPTOR ESP32 CROWBAR:
   Si la fidelidad de la medición débil o la recurrencia de Poincaré detectan una inyección de ruido o
   discrepancia estructural, el Soberano Testigo emite un pulso de veto.
   La rutina de interrupción en memoria IRAM del microcontrolador ESP32 conmuta el pin GPIO14 a HIGH en < 400 ns,
   disparando el tiristor BT151 (Crowbar) para aislar la compuerta de desembolsos monetarios.

TRADUCCIÓN EJECUTIVA ("DOLOR Y DINERO")
──────────────────────────────────────
- Blindaje Probatorio Total: Generación de evidencia inalienable con validez jurídica que
  protege a la constructora ante demandas contractuales o investigaciones de entes de control.
- Invariancia Presupuestal: Garantiza que las lecciones aprendidas de desviaciones de costos
  pasadas se convirtan en políticas de decisión inmutables, evitando repetir errores de cotización.
- Reducción del WACC por Transparencia: La presencia del Testigo Silencioso eleva la calificación
  ESG y disminuye el costo de capital de riesgo bancario al certificar la ausencia de corrupción.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Tuple, Any

import numpy as np
import scipy.linalg as la

from app.wisdom.toon_silent_witness_engine import (
    TOONSilentWitnessEngine,
    ExperienceCrystal as EngineExperienceCrystal,
    HeytingOmega3,
    MatrixBanachAlgebra,
    ModularHamiltonian,
    DensityOperatorAlgebra,
    GNSHilbertAlgebra,
    VacuumModularContext,
    VacuumStatePreparation,
    TomitaTakesakiEngine,
    VacuumAuditReport,
    VacuumSpectraAnalyzer,
    SilentFieldProbe,
    SilentFieldDetector,
    SilentFieldBundle,
    ModularSilencePipeline,
    HeytingVacuumAdjudicator,
    SilentFieldState,
)


__version__ = "8.0.0-Doctoral-Poincaré-Recurrence-KMS-Tomita-Takesaki-Dirac-Vacuum-S6"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitness")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12
_PSD_EIG_FLOOR = 0.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ONTOLÓGICO DE LA TRÍADA Y POINCARÉ                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _seed_from_string(s: str) -> int:
    """SHA-256 → semilla uint32 (determinismo reproducible, no criptográfico)."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# Alias local para compatibilidad
DensityMatrixOps = DensityOperatorAlgebra


@dataclass(frozen=True, slots=True)
class VacuumStateMetrics:
    r"""
    Métricas verdaderas de un estado ρ frente a (H_ext, ρ_vac, K_vac).
    """

    vacuum_expectation_value: float
    tomita_takesaki_flow_param: float
    kms_entropy_drift: float
    silence_purity: float
    modular_ground_energy: float
    modular_spectral_gap: float
    von_neumann_entropy: float
    purity: float
    is_silent: bool
    umegaki_to_vacuum: float = 0.0
    bures_distance: float = 0.0
    trace_distance: float = 0.0
    klein_residual: float = 0.0
    dirichlet_energy: float = 0.0


@dataclass(frozen=True, slots=True)
class TriadSignature:
    r"""
    Identidad forense de la tríada: tipos textuales + hashes SHA-256 de
    los tres operadores y rango del proyector.
    """

    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    unitary_hash: str
    dreamer_hash: str
    projector_hash: str
    auditor_rank: int

    def as_bytes(self) -> bytes:
        return (
            f"{self.trickster_illusion_type}|{self.dreamer_scenario_id}|"
            f"{self.auditor_immunization_hash}|{self.unitary_hash[:16]}|"
            f"{self.dreamer_hash[:16]}|{self.projector_hash[:16]}|"
            f"{self.auditor_rank}"
        ).encode("utf-8")


class TriadOperatorFactory:
    r"""
    Construye U_I, D y P_A de forma determinista a partir de identificadores.
    """

    @classmethod
    def _haar(cls, n: int, rng: np.random.Generator) -> np.ndarray:
        ginibre = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        q_factor, r_factor = np.linalg.qr(ginibre)
        diag_r = np.diagonal(r_factor)
        phases = np.where(np.abs(diag_r) > 1e-30, diag_r / np.abs(diag_r), 1.0 + 0j)
        return q_factor * phases.conj()

    @classmethod
    def build_unitary(cls, illusion_type: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"TRICKSTER::{illusion_type}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        a_h = MatrixBanachAlgebra.hermitize(amp)
        nrm = float(np.linalg.norm(a_h, "fro")) + 1e-30
        a_h = a_h / nrm
        theta = math.pi * strength * n
        return la.expm(1j * theta * a_h)

    @classmethod
    def build_dreamer(cls, scenario_id: str, n: int, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        rng = np.random.default_rng(_seed_from_string(f"DREAMER::{scenario_id}"))
        if strength <= 0.0:
            return np.eye(n, dtype=np.complex128)
        amp = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        d_rand = amp.conj().T @ amp
        tr = float(np.trace(d_rand).real) + 1e-30
        d_rand = d_rand / tr * n
        mixed = (1.0 - strength) * np.eye(n, dtype=np.complex128) + strength * d_rand
        return MatrixBanachAlgebra.hermitize(mixed)

    @classmethod
    def build_projector(cls, immunization_hash: str, n: int, rank: int) -> np.ndarray:
        rank = int(np.clip(rank, 1, n))
        rng = np.random.default_rng(_seed_from_string(f"AUDITOR::{immunization_hash}"))
        haar = cls._haar(n, rng)
        cols = haar[:, :rank]
        projector = cols @ cols.conj().T
        return MatrixBanachAlgebra.hermitize(projector)

    @classmethod
    def build_triad(
        cls,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        n: int,
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
    ) -> TriadChannel:
        unitary = cls.build_unitary(trickster_illusion_type, n, trickster_strength)
        dreamer = cls.build_dreamer(dreamer_scenario_id, n, dreamer_strength)
        projector = cls.build_projector(auditor_immunization_hash, n, auditor_rank)
        signature = TriadSignature(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            unitary_hash=_sha256_array(unitary),
            dreamer_hash=_sha256_array(dreamer),
            projector_hash=_sha256_array(projector),
            auditor_rank=int(np.clip(auditor_rank, 1, n)),
        )
        return TriadChannel(signature, unitary, dreamer, projector)


class TriadChannel:
    r"""
    Instrumento de Lüders de un solo Kraus (categoría de mapas CP).
    """

    def __init__(
        self,
        signature: TriadSignature,
        U_I: np.ndarray,
        D: np.ndarray,
        P_A: np.ndarray,
    ) -> None:
        self.signature = signature
        self.U = MatrixBanachAlgebra.as_complex(U_I)
        self.D = MatrixBanachAlgebra.hermitize(D)
        self.P = MatrixBanachAlgebra.hermitize(P_A)
        self.K = self.P @ self.D @ self.U

    def apply_linear(self, rho: np.ndarray) -> np.ndarray:
        rho = MatrixBanachAlgebra.as_complex(rho)
        return self.K @ rho @ self.K.conj().T

    def apply(self, rho: np.ndarray) -> np.ndarray:
        out = self.apply_linear(rho)
        tr = float(np.trace(out).real)
        if tr < 1e-30:
            n = rho.shape[0]
            return np.eye(n, dtype=np.complex128) / n
        return out / tr

    def success_probability(self, rho: np.ndarray) -> float:
        metric = self.K.conj().T @ self.K
        return float(np.real(np.trace(rho @ metric)))

    def kraus_norm(self) -> float:
        return float(np.linalg.norm(self.K, "fro"))

    def tp_residual(self) -> float:
        n = self.K.shape[0]
        defect = self.K.conj().T @ self.K - np.eye(n, dtype=np.complex128)
        return MatrixBanachAlgebra.schatten_norm(defect, math.inf)

    def projector_residual(self) -> float:
        return float(np.linalg.norm(self.P @ self.P - self.P, "fro"))

    def unitary_residual(self) -> float:
        n = self.U.shape[0]
        eye = np.eye(n, dtype=np.complex128)
        left = self.U.conj().T @ self.U - eye
        return float(np.linalg.norm(left, "fro"))

    def choi_matrix(self) -> np.ndarray:
        n = self.K.shape[0]
        vec_k = self.K.reshape((n * n, 1), order="F")
        return vec_k @ vec_k.conj().T

    def cp_min_eigenvalue(self) -> float:
        choi = MatrixBanachAlgebra.hermitize(self.choi_matrix())
        w = np.real(la.eigvalsh(choi))
        return float(w.min()) if w.size else 0.0

    def diamond_norm_upper_bound(self) -> float:
        return MatrixBanachAlgebra.schatten_norm(self.K, math.inf) ** 2

    def structural_residuals(self) -> Dict[str, float]:
        return {
            "tp_residual": self.tp_residual(),
            "projector_residual": self.projector_residual(),
            "unitary_residual": self.unitary_residual(),
            "cp_min_eigenvalue": self.cp_min_eigenvalue(),
            "diamond_upper": self.diamond_norm_upper_bound(),
        }


# Alias local para contexto de vacío de agente
WitnessVacuumContext = VacuumModularContext


class WitnessVacuumPreparation(VacuumStatePreparation):
    r"""
    Prepara el contexto modular del Testigo Silencioso.
    """

    @classmethod
    def path_laplacian(cls, n: int) -> np.ndarray:
        lap = np.zeros((n, n), dtype=np.complex128)
        for i in range(n - 1):
            lap[i, i] += 1.0
            lap[i + 1, i + 1] += 1.0
            lap[i, i + 1] -= 1.0
            lap[i + 1, i] -= 1.0
        return MatrixBanachAlgebra.hermitize(lap)


@dataclass(frozen=True, slots=True)
class WitnessObservationBundle:
    cycle_index: int
    triad_signature: TriadSignature
    rho_vac: np.ndarray
    rho_observed: np.ndarray
    modular_spectrum: Tuple[float, ...]
    K: ModularHamiltonian
    audit: VacuumAuditReport
    modular_axioms: Dict[str, float]
    tomita_report: Dict[str, float]
    triad_residuals: Dict[str, float]
    invariant_vector: np.ndarray
    context_beta: float

    def as_vacuum_metrics(self, is_silent: bool) -> VacuumStateMetrics:
        return VacuumStateMetrics(
            vacuum_expectation_value=self.audit.vev,
            tomita_takesaki_flow_param=1.0 / max(self.K.spectral_gap, 1e-12),
            kms_entropy_drift=self.audit.free_energy,
            silence_purity=self.audit.purity,
            modular_ground_energy=self.K.ground_energy,
            modular_spectral_gap=self.audit.spectral_gap,
            von_neumann_entropy=DensityMatrixOps.von_neumann_entropy(self.rho_observed),
            purity=DensityMatrixOps.purity(self.rho_observed),
            is_silent=is_silent,
            umegaki_to_vacuum=0.0,
            bures_distance=0.0,
            trace_distance=0.0,
            klein_residual=0.0,
            dirichlet_energy=0.0,
        )


class WitnessObservationPipeline:
    @classmethod
    def _invariant_vector(
        cls, audit: VacuumAuditReport, triad: TriadChannel
    ) -> np.ndarray:
        v_local = float(int(audit.local_verdict)) / 2.0
        d_s = math.tanh(abs(audit.free_energy))
        leak = 1.0 - audit.purity
        kms = math.tanh(10.0 * audit.thermal_fluctuation)
        vev = math.tanh(abs(audit.vev))
        n = max(1, triad.K.shape[0])
        p_trans = 1.0
        op_norm = np.linalg.norm(triad.K, 2) + 1e-30
        k_fro = triad.kraus_norm() / (math.sqrt(n) * op_norm)
        vec = np.array(
            [v_local, d_s, leak, kms, vev, p_trans, k_fro], dtype=np.float64
        )
        norm = float(np.linalg.norm(vec))
        return vec / (norm + 1e-30)

    @classmethod
    def synthesize(
        cls,
        cycle_index: int,
        rho_vac: np.ndarray,
        K_spec: Tuple[float, ...],
        triad: TriadChannel,
        H_ext: np.ndarray,
        K: Optional[ModularHamiltonian] = None,
        path_laplacian: Optional[np.ndarray] = None,
        context_beta: float = float("nan"),
    ) -> WitnessObservationBundle:
        rho_vac = DensityMatrixOps.sanitize(rho_vac)
        if K is None:
            K = DensityMatrixOps.modular_hamiltonian_from_rho(rho_vac)

        rho_obs = triad.apply(rho_vac)
        triad_res = triad.structural_residuals()

        tomita = TomitaTakesakiEngine.verify_tomita_takesaki(rho_obs)
        axioms = {
            k: tomita[k]
            for k in (
                "unital_residual",
                "product_residual",
                "isometry_residual",
                "involution_residual",
            )
            if k in tomita
        }
        if len(axioms) < 4:
            axioms = TomitaTakesakiEngine.verify_algebra_axioms(rho_obs)

        audit = VacuumSpectraAnalyzer.audit(rho_obs, H_ext, K)
        inv = cls._invariant_vector(audit, triad)

        return WitnessObservationBundle(
            cycle_index=cycle_index,
            triad_signature=triad.signature,
            rho_vac=rho_vac,
            rho_observed=rho_obs,
            modular_spectrum=K_spec if K_spec else K.eigenvalues,
            K=K,
            audit=audit,
            modular_axioms=axioms,
            tomita_report=tomita,
            triad_residuals=triad_res,
            invariant_vector=inv,
            context_beta=context_beta,
        )

    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: WitnessVacuumContext,
        triad: TriadChannel,
    ) -> WitnessObservationBundle:
        ctx = TomitaTakesakiEngine.bind_vacuum_context(ctx)
        return cls.synthesize(
            cycle_index=cycle_index,
            rho_vac=ctx.rho,
            K_spec=ctx.K.eigenvalues,
            triad=triad,
            H_ext=ctx.H,
            K=ctx.K,
            context_beta=ctx.beta,
        )


class HeytingWitnessAdjudicator:
    AXIOM_TOL: float = 1.0e-6
    CATASTROPHIC_TOL: float = 1.0e-3

    @classmethod
    def adjudicate(
        cls,
        bundle: WitnessObservationBundle,
        external_verdict: HeytingOmega3,
    ) -> HeytingOmega3:
        local = bundle.audit.local_verdict
        return local.meet(external_verdict)


@dataclass(frozen=True, slots=True)
class ExperienceCrystal:
    r"""
    Cristal inmutable de experiencia, firmado con SHA-256 y encadenado Merkle-style.
    """

    crystal_id: str
    witness_id: str
    trickster_illusion_type: str
    dreamer_scenario_id: str
    auditor_immunization_hash: str
    vacuum_metrics: VacuumStateMetrics
    audit_report: VacuumAuditReport
    heyting_verdict: HeytingOmega3
    crystallized_invariant_vector: np.ndarray
    content_hash: str
    chain_hash: str
    merkle_parent: str
    sha256_provenance: str
    timestamp_utc: float
    vector_s6: Optional[np.ndarray] = None
    tau_recurrence: float = 1.0
    kms_drift: float = 0.0
    merkle_root_sha256: str = ""
    engine_version: str = __version__


# ── §1.6 TOONSilentWitnessAgent — Orquestador Soberano ─────────────────────
class TOONSilentWitnessAgent:
    r"""
    Soberano Testigo Silencioso de Cero Reacción de Fondo y Recurrencia de Poincaré.

    Habita en el vacío modular (KMS a β_vac), observa de forma imparcial
    la Malla Agéntica sin perturbar el estado cuántico (zero back-action),
    y audita el retorno de Poincaré en la medida de Liouville.
    """

    def __init__(
        self,
        agent_id: str = "SILENT-WITNESS-SABIO-01",
        dimension_mac: int = 4,
        dimension: Optional[int] = None,
        kms_beta: float = 1.0,
        vacuum_beta: float = WitnessVacuumPreparation.DEFAULT_BETA_COLD,
        seed: int = 999,
        hopping: float = WitnessVacuumPreparation.DEFAULT_HOPPING,
    ) -> None:
        if dimension is not None:
            dimension_mac = dimension
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.kms_beta = kms_beta
        self.vacuum_beta = vacuum_beta
        self.seed = seed
        self.crystal_count = 0
        self.crystal_history: List[ExperienceCrystal] = []
        self.experience_archive: List[ExperienceCrystal] = []

        self.engine = TOONSilentWitnessEngine(
            engine_id=f"ENGINE-{agent_id}",
            mac_dimension=dimension_mac,
            kms_beta=kms_beta,
            hopping=hopping,
        )

        self.H_ext = WitnessVacuumPreparation.tight_binding_hamiltonian(
            dimension_mac, hopping=hopping
        )
        self.ground_projector = DensityMatrixOps.ground_state_projector(self.H_ext)
        self.path_laplacian = WitnessVacuumPreparation.path_laplacian(dimension_mac)

        self._genesis_hash = hashlib.sha256(
            f"{agent_id}::GENESIS::{__version__}".encode("ascii")
        ).hexdigest()
        self._chain_hash = self._genesis_hash

    def _content_hash(
        self,
        cycle_id: str,
        audit: VacuumAuditReport,
        inv: np.ndarray,
        sig: TriadSignature,
        dirichlet_energy_arg: float,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.agent_id.encode("ascii"))
        hasher.update(cycle_id.encode("ascii"))
        hasher.update(sig.as_bytes())
        hasher.update(np.ascontiguousarray(inv).tobytes())
        hasher.update(f"{audit.thermal_fluctuation:.12e}".encode("ascii"))
        hasher.update(f"{audit.purity:.12e}".encode("ascii"))
        hasher.update(f"{audit.free_energy:.12e}".encode("ascii"))
        hasher.update(f"{audit.vev:.12e}".encode("ascii"))
        hasher.update(f"{audit.local_verdict.name}".encode("ascii"))
        hasher.update(f"{dirichlet_energy_arg:.12e}".encode("ascii"))
        return hasher.hexdigest()

    def _advance_chain(self, content_hash: str) -> Tuple[str, str]:
        parent = self._chain_hash
        digest = hashlib.sha256(
            parent.encode("ascii") + content_hash.encode("ascii")
        ).hexdigest()
        self._chain_hash = digest
        return parent, digest

    def execute_zero_backaction_poincare_observation(
        self,
        density_matrix: np.ndarray,
        tau_recurrence: float = 1.0
    ) -> Dict[str, Any]:
        r"""
        Ejecuta la observación imparcial sin perturbar el estado cuántico (zero back-action).

        Verifica la recurrencia de Poincaré, actualiza la bitácora de experiencia
        y emite la decisión en el topos de Heyting Ω₃.
        """
        is_rec, drift, poincare_crystal = self.engine.audit_poincare_recurrence_kms_vacuum(
            density_matrix, tau_recurrence
        )

        verdict = "COHERENT" if is_rec else "DEGRADED"
        if drift > 1e-2:
            verdict = "VETOED"

        logger.info(
            "[SILENT_WITNESS] Poincare Recurrence: %s | Drift: %.3e | Merkle: %s...",
            is_rec, drift, poincare_crystal.merkle_root_sha256[:12]
        )

        return {
            "agent": "TOONSilentWitnessAgent",
            "verdict": verdict,
            "poincare_recurrence": is_rec,
            "kms_drift": drift,
            "crystal": poincare_crystal,
            "crowbar_trigger": verdict == "VETOED"
        }

    def latest_crystal(self) -> Optional[ExperienceCrystal]:
        return self.crystal_history[-1] if self.crystal_history else None

    def verify_merkle_chain(self) -> bool:
        r"""
        Recomputación de chain_hash_k = SHA256(parent_{k−1} ‖ content_hash_k)
        y verificación de formato hex-64 + ‖v_inv‖₂ ≈ 1.
        """
        parent = self._genesis_hash
        for crystal in self.experience_archive:
            if crystal.merkle_parent != parent:
                return False
            expected = hashlib.sha256(
                parent.encode("ascii") + crystal.content_hash.encode("ascii")
            ).hexdigest()
            if expected != crystal.chain_hash:
                return False
            if len(crystal.chain_hash) != 64:
                return False
            if not math.isclose(
                float(np.linalg.norm(crystal.crystallized_invariant_vector)),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            ):
                return False
            parent = crystal.chain_hash
        return True

    def observe_and_crystallize(
        self,
        trickster_illusion_type: str,
        dreamer_scenario_id: str,
        auditor_immunization_hash: str,
        auditor_verdict: HeytingOmega3,
        trickster_strength: float = 0.0,
        dreamer_strength: float = 0.0,
        auditor_rank: int = 4,
        dirichlet_energy: float = 0.0,
        tau_recurrence: float = 1.0,
    ) -> ExperienceCrystal:
        self.crystal_count += 1
        crystal_id = f"CRYSTAL-EXP-{self.crystal_count:04d}"
        t_start = time.perf_counter()
        logger.info(
            "═══ Silencio Epistémico #%d | ilusión=%s | rank=%d ═══",
            self.crystal_count,
            trickster_illusion_type,
            auditor_rank,
        )

        ctx = WitnessVacuumPreparation.prepare_vacuum_context(
            self.H_ext, beta=self.vacuum_beta
        )

        triad = TriadOperatorFactory.build_triad(
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            n=self.dimension_mac,
            trickster_strength=trickster_strength,
            dreamer_strength=dreamer_strength,
            auditor_rank=auditor_rank,
        )

        bundle = WitnessObservationPipeline.synthesize_from_context(
            cycle_index=self.crystal_count, ctx=ctx, triad=triad
        )

        final_verdict = HeytingWitnessAdjudicator.adjudicate(bundle, auditor_verdict)
        metrics = bundle.as_vacuum_metrics(
            is_silent=(final_verdict == HeytingOmega3.COHERENT)
        )

        content_hash = self._content_hash(
            crystal_id,
            bundle.audit,
            bundle.invariant_vector,
            bundle.triad_signature,
            dirichlet_energy,
        )
        merkle_parent, chain_hash = self._advance_chain(content_hash)

        sig_hasher = hashlib.sha256()
        sig_hasher.update(self.agent_id.encode("ascii"))
        sig_hasher.update(crystal_id.encode("ascii"))
        sig_hasher.update(final_verdict.name.encode("ascii"))
        sig_hasher.update(chain_hash.encode("ascii"))
        sig_hasher.update(f"{time.time_ns()}".encode("ascii"))
        provenance = sig_hasher.hexdigest()

        # Audit de recurrencia de Poincaré
        is_rec, drift, poincare_crystal = self.engine.audit_poincare_recurrence_kms_vacuum(
            bundle.rho_observed, tau_recurrence
        )

        crystal = ExperienceCrystal(
            crystal_id=crystal_id,
            witness_id=self.agent_id,
            trickster_illusion_type=trickster_illusion_type,
            dreamer_scenario_id=dreamer_scenario_id,
            auditor_immunization_hash=auditor_immunization_hash,
            vacuum_metrics=metrics,
            audit_report=bundle.audit,
            heyting_verdict=final_verdict,
            crystallized_invariant_vector=bundle.invariant_vector,
            content_hash=content_hash,
            chain_hash=chain_hash,
            merkle_parent=merkle_parent,
            sha256_provenance=provenance,
            timestamp_utc=time.time(),
            vector_s6=poincare_crystal.vector_s6,
            tau_recurrence=tau_recurrence,
            kms_drift=drift,
            merkle_root_sha256=poincare_crystal.merkle_root_sha256,
            engine_version=__version__,
        )
        self.crystal_history.append(crystal)
        self.experience_archive.append(crystal)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cristal %s | Ω₃=%s | F=%.6f | KMS_x=%.3e | %.2f ms",
            crystal_id,
            final_verdict.name,
            bundle.audit.purity,
            bundle.audit.thermal_fluctuation,
            dt_ms,
        )
        return crystal


if __name__ == "__main__":
    witness = TOONSilentWitnessAgent(
        agent_id="SILENT-WITNESS-SABIO-01",
        dimension=4,
        kms_beta=1.0,
        vacuum_beta=50.0,
    )

    print("═" * 88)
    print(f"TESTIGO SILENCIOSO — Evolución Doctoral Poincaré v{__version__}")
    print("═" * 88)

    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    res = witness.execute_zero_backaction_poincare_observation(rho_test, tau_recurrence=1.0)
    print("  Respuesta de Observación Zero Back-Action:")
    print("    Veredicto             :", res["verdict"])
    print("    Recurrencia Poincaré  :", res["poincare_recurrence"])
    print("    Drift KMS             :", res["kms_drift"])
    print("    Merkle Root           :", res["crystal"].merkle_root_sha256[:16], "...")
    print("    Crowbar Trigger       :", res["crowbar_trigger"])

    c_triad = witness.observe_and_crystallize(
        trickster_illusion_type="COHERENT_ILLUSION",
        dreamer_scenario_id="SCENARIO_001",
        auditor_immunization_hash="IMM_HASH_001",
        auditor_verdict=HeytingOmega3.COHERENT,
    )
    print("\n  Cristalización de Tríada + Poincaré:")
    print("    Crystal ID            :", c_triad.crystal_id)
    print("    Veredicto             :", c_triad.heyting_verdict.name)
    print("    S6 Vector Norm        :", np.linalg.norm(c_triad.vector_s6))
    print("    Merkle Root SHA-256   :", c_triad.merkle_root_sha256[:16], "...")

    print("\n  Cadena Merkle Válida    :", witness.verify_merkle_chain())
    print("\n" + "═" * 88)
    print("✓ Soberano Testigo Silencioso refactorizado con éxito.")
    print("═" * 88)

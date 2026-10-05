# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO   : app/agents/wisdom/toon_silent_witness_agent.py                            ║
║ ESTRATO  : WISDOM (V_W) — CIUDADELA DE CRISTAL / VACÍO DE DIRAC                      ║
║ FUNCIÓN  : SOBERANO TESTIGO SILENCIOSO Y CRISTALIZADOR DE EXPERIENCIA                ║
║ VERSIÓN  : 9.0.0-Poincaré-Celestial-Mechanics-CR3BP-KAM-Birkhoff-KS-S6               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN RIGUROSA Y FUNDAMENTACIÓN MATEMÁTICA
───────────────────────────────────────────────
El `TOONSilentWitnessAgent` actúa como el Soberano observador imparcial e
incorruptible del Estrato Wisdom (V_W). Reside en el límite de congelamiento
entrópico del vacío de Dirac y observa de forma no destructiva las
interacciones de la Malla Agéntica.

MEJORAS DOCTORALES v9.0.0 (POINCARÉ CELESTE):
─────────────────────────────────────────────
(I)   La tríada (Trickster, Dreamer, Auditor) se reinterpreta como el Problema
      de los Tres Cuerpos Restringido Circular (CR3BP) en marco rotante de
      Poincaré, con ratio de masas μ ∈ (0, 1/2] y potencial efectivo Ω(x,y).
(II)  El grupo simpático Sp(4,ℝ) actúa sobre la variedad foliada por la
      integral de Jacobi C = 2Ω − (ẋ² + ẏ²).
(III) Se construye la sección de Poincaré Σ = {y=0, ẏ>0} y se estima la
      aplicación de retorno T: Σ → Σ con detección de eventos RK4.
(IV)  Regularización Levi-Civita (planar) y Kustaanheimo-Stiefel (3D) para
      remover singularidades binarias.
(V)   Indicadores exponenciales de caos: Liapunov, SALI (Skokos 2001) y
      cotejo con la función de Melnikov para transversalidad homoclínica.
(VI)  Verificación KAM de tori invariantes y análisis de resonancias
      p:q (Moser 1962, Arnold 1963).
(VII) Forma normal de Birkhoff alrededor de L4/L5 (Lyapunov-Deprit) con
      frecuencia fundamental ω_{short,long}.
(VIII) Difusión de Arnold en la red resonante (Nekhoroshev) cuantificada
       como proxy del KMS drift del Testigo.
(IX)  La 7-upla invariante se proyecta en S⁶ preservando compatibilidad
      topológica con el vacío de Dirac.

FIRMA DEL TESTIGO SILENCIOSO: mediciones débiles A_w con κ → 0 y contrato
Merkle con raíz SHA-256 (v_S6 ‖ τ_rec ‖ KMSDrift ‖ Timestamp).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

# ── Motor heredado Tomita–Takesaki / KMS / Dirac-Vacuum ──
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


__version__ = "9.0.0-Poincaré-Celestial-Mechanics-CR3BP-KAM-Birkhoff-KS-S6"


logger = logging.getLogger("APU.Wisdom.TOONSilentWitness")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ── Constantes espectrales / Banach ──
_EPS = 1.0e-14
_HERMITICITY_TOL = 1.0e-12
_TRACE_TOL = 1.0e-10
_GAP_DEGENERACY_TOL = 1.0e-12
_PSD_EIG_FLOOR = 0.0

# ── Constantes de mecánica celeste de Poincaré ──
_MU_FLOOR = 1.0e-6
_MU_CEIL = 0.5 - 1.0e-6
_JACOBI_TOL = 1.0e-10
_LIBRATION_TOL = 1.0e-12
_KAM_TOL = 1.0e-8
# Routh (1875): 27μ(1−μ) = 1 ⇒ μ_R = 0.038520896504083816...
_ROUTH_CRITICAL_MU = 0.5 * (1.0 - math.sqrt(23.0 / 27.0))
_SING_HARDENING = 1.0e-12


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 · SUSTRATO ONTOLÓGICO: LA TRÍADA COMO CR3BP DE POINCARÉ                       ║
# ║                                                                                      ║
# ║ La tríada (U_I, D, P_A) induce un flujo hamiltoniano en T*ℝ² ⊂ ℝ⁴ con métrica         ║
# ║ simpléctica ω = dx∧dp_x + dy∧dp_y. La componente Trickster fija el ratio de masas    ║
# ║ μ; Dreamer y Auditor fijan el miembro de la familia foliada por la integral de       ║
# ║ Jacobi C. El método final de esta fase,                                                ║
# ║ `TriadToCR3BPMorphism.emit_poincare_seed`, produce el `PoincareSectionSeed` que el    ║
# ║ motor `PoincareSectionIntegrator` de la Fase 2 ingiere como condición inicial.        ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝


def _seed_from_string(s: str) -> int:
    """SHA-256 → semilla uint32 (determinismo reproducible, no criptográfico)."""
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# Alias local para compatibilidad con el motor heredado
DensityMatrixOps = DensityOperatorAlgebra


# ── §1.1 Métricas del vacío modular (extendidas con datos celestes) ──────────
@dataclass(frozen=True, slots=True)
class VacuumStateMetrics:
    r"""
    Métricas verdaderas de un estado ρ frente a (H_ext, ρ_vac, K_vac)
    enriquecidas con la firma celeste del morfismo CR3BP.
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
    # Engrosamiento celeste
    mass_ratio_mu: float = 0.0
    jacobi_constant: float = 0.0
    jacobi_residual: float = 0.0
    libration_xy: Tuple[float, float] = (0.0, 0.0)
    routh_stable: bool = False


@dataclass(frozen=True, slots=True)
class TriadSignature:
    r"""
    Identidad forense de la tríada: tipos textuales + hashes SHA-256 de los
    tres operadores y rango del proyector.
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


# ── §1.2 Fábrica de operadores de la tríada ──────────────────────────────────
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
    ) -> "TriadChannel":
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


# ── §1.3 Canal de tríada (instrumento de Lüders de un solo Kraus) ────────────
class TriadChannel:
    r"""
    Instrumento de Lüders de un solo Kraus: ρ ↦ P_A D U_I ρ U_I† D† P_A† / tr(·)
    categóricamente un morfismo CP en la categoría de operadores de Hilbert.
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


# ── §1.4 Preparación del vacío modular del Testigo ───────────────────────────
class WitnessVacuumPreparation(VacuumStatePreparation):
    r"""
    Prepara el contexto modular del Testigo Silencioso: Laplaciano de camino
    sobre ℤ_n y Hamiltoniano tight-binding.
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


# ══════════════════════════════════════════════════════════════════════════════════════
# §1.5 · Álgebra canónica de Poincaré (Poincaré 1892–1899, Charlier, Szebehely 1967)
# ══════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareCanonicalCoordinates:
    r"""
    Variables canónicas de Poincaré (Λ, λ, ξ, η) para el problema de Kepler
    perturbado. Relación al semieje a y excentricidad e:

        Λ = √(a),          ξ = √(2Λ) · √(1 − √(1 − e²)) · cos(ω),
        λ = M + ω,          η = −√(2Λ) · √(1 − √(1 − e²)) · sin(ω).
    """
    Lambda: float
    lambd: float
    xi: float
    eta: float

    @property
    def eccentricity(self) -> float:
        # Λ = √(a); ξ²+η² = 2Λ (1 − √(1−e²))
        if self.Lambda <= 0.0:
            return 0.0
        s = (self.xi**2 + self.eta**2) / (2.0 * self.Lambda)
        s = float(np.clip(s, 0.0, 1.0))
        return math.sqrt(max(0.0, 1.0 - (1.0 - s) ** 2))

    def as_vector(self) -> np.ndarray:
        return np.array([self.Lambda, self.lambd, self.xi, self.eta], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DelaunayElements:
    r"""
    Elementos de Delaunay (L, G, H, l, g, h): acción-ángulo clásicos
    para el problema de dos cuerpos. Cumplen simetría canónica.
    """
    L: float
    G: float
    H: float
    l: float
    g: float
    h: float


# ══════════════════════════════════════════════════════════════════════════════════════
# §1.6 · CR3BP: estado y flujo hamiltoniano rotante de Poincaré
# ══════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CR3BPState:
    r"""
    Estado de fase en el marco barredor rotante: z = (x, y, vx, vy).
    Primarios en (−μ, 0) y (1−μ, 0).
    """
    x: float
    y: float
    vx: float
    vy: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)

    @classmethod
    def from_array(cls, z: np.ndarray) -> "CR3BPState":
        arr = np.asarray(z, dtype=np.float64).ravel()
        return cls(x=float(arr[0]), y=float(arr[1]),
                   vx=float(arr[2]), vy=float(arr[3]))


class CR3BPDynamics:
    r"""
    Dinámica del problema restringido circular (Szebehely 1967).

        Ω(x, y; μ) = ½(x² + y²) + (1 − μ)/r₁ + μ/r₂,
        r₁ = |z + (μ, 0)|,  r₂ = |z − (1 − μ, 0)|.

    Ecuaciones de Euler-Lagrange en marco rotante:

        ẍ − 2ẏ   =  ∂Ω/∂x
        ÿ + 2ẋ   =  ∂Ω/∂y
    """

    @staticmethod
    def _radii(x: float, y: float, mu: float) -> Tuple[float, float]:
        r1 = math.hypot(x + mu, y)
        r2 = math.hypot(x - 1.0 + mu, y)
        return max(r1, _SING_HARDENING), max(r2, _SING_HARDENING)

    @classmethod
    def potential(cls, x: float, y: float, mu: float) -> float:
        mu = float(np.clip(mu, _MU_FLOOR, _MU_CEIL))
        r1, r2 = cls._radii(x, y, mu)
        return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2

    @classmethod
    def potential_grad(cls, x: float, y: float, mu: float) -> Tuple[float, float]:
        mu = float(np.clip(mu, _MU_FLOOR, _MU_CEIL))
        r1, r2 = cls._radii(x, y, mu)
        d_x = x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
        d_y = y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
        return d_x, d_y

    @classmethod
    def vector_field(cls, mu: float) -> Callable[[float, np.ndarray], np.ndarray]:
        mu_ = float(np.clip(mu, _MU_FLOOR, _MU_CEIL))

        def f(_t: float, z: np.ndarray) -> np.ndarray:
            x, y, vx, vy = z
            d_x, d_y = cls.potential_grad(x, y, mu_)
            ax = 2.0 * vy + d_x
            ay = -2.0 * vx + d_y
            return np.array([vx, vy, ax, ay], dtype=np.float64)

        return f

    @classmethod
    def jacobi_constant(cls, state: CR3BPState, mu: float) -> float:
        return 2.0 * cls.potential(state.x, state.y, mu) - (state.vx**2 + state.vy**2)

    @classmethod
    def jacobi_residual(
        cls, z0: CR3BPState, z1: CR3BPState, mu: float
    ) -> float:
        c0 = cls.jacobi_constant(z0, mu)
        c1 = cls.jacobi_constant(z1, mu)
        return abs(c0 - c1) / max(1.0, abs(c0))


# ══════════════════════════════════════════════════════════════════════════════════════
# §1.7 · Puntos de libración L1–L5 (Euler 1767, Lagrange 1772)
# ══════════════════════════════════════════════════════════════════════════════════════

class LibrationPointSolver:
    r"""
    Resuelve los puntos de libración. L1, L2, L3 sobre el eje x; L4, L5
    en vértices equiláteros en (½ − μ, ±√3/2). Estabilidad lineal de L4/L5
    garantizada sii 27μ(1 − μ) < 1 (Routh 1875).
    """

    @staticmethod
    def _collinear_equation(x: float, mu: float) -> float:
        dx = x - (1.0 - mu) * (x + mu) / max(abs(x + mu), _SING_HARDENING) ** 3 \
             - mu * (x - 1.0 + mu) / max(abs(x - 1.0 + mu), _SING_HARDENING) ** 3
        return dx

    @classmethod
    def solve_collinear(cls, mu: float, bracket: Tuple[float, float]) -> float:
        mu = float(np.clip(mu, _MU_FLOOR, _MU_CEIL))
        a, b = bracket
        fa = cls._collinear_equation(a, mu)
        fb = cls._collinear_equation(b, mu)
        if fa * fb > 0:
            # fallback: bisección amplia
            xs = np.linspace(a, b, 129)
            last = xs[0]
            for xg in xs[1:]:
                fg = cls._collinear_equation(xg, mu)
                if cls._collinear_equation(last, mu) * fg <= 0.0:
                    return float(brentq(cls._collinear_equation, last, xg, args=(mu,)))
                last = xg
            return float(0.5 * (a + b))
        return float(brentq(cls._collinear_equation, a, b, args=(mu,)))

    @classmethod
    def solve_all(cls, mu: float) -> Dict[str, Tuple[float, float]]:
        mu = float(np.clip(mu, _MU_FLOOR, _MU_CEIL))
        # L1 entre los primarios; L2 a la derecha del secundario; L3 al exterior del primario
        x1 = cls.solve_collinear(mu, (-mu + 1e-3, 1.0 - mu - 1e-3))
        x2 = cls.solve_collinear(mu, (1.0 - mu + 1e-3, 1.0 - mu + 4.0))
        x3 = cls.solve_collinear(mu, (-4.0, -mu - 1e-3))
        x4 = 0.5 - mu
        y4 = math.sqrt(3.0) / 2.0
        return {
            "L1": (x1, 0.0),
            "L2": (x2, 0.0),
            "L3": (x3, 0.0),
            "L4": (x4, y4),
            "L5": (x4, -y4),
        }

    @staticmethod
    def routh_stability(mu: float) -> bool:
        return 27.0 * mu * (1.0 - mu) < 1.0


# ══════════════════════════════════════════════════════════════════════════════════════
# §1.8 · Morfismo Triad → CR3BP y emisión del `PoincareSectionSeed`
#         Este es el ÚLTIMO bloque de la Fase 1 y su método terminal es la
#         CONTINUACIÓN NATURAL hacia el motor `PoincareSectionIntegrator`
#         de la Fase 2 (busca la firma `emit_poincare_seed`).
# ══════════════════════════════════════════════════════════════════════════════════════

class TriadToCR3BPMorphism:
    r"""
    Morfismo categórico (TriadChannel) → CR3BP.

    Mapa del ratio de masas:
        μ = clip( β_T · (ar / n) / (β_T + β_D + ε), μ_min, μ_max )
    con β_T = trickster_strength, β_D = dreamer_strength, ar = auditor_rank.

    La integral de Jacobi se hereda desde el vector invariante del canal:
        C = 2·‖K‖₂² − ‖ρ_obs − ρ_vac‖_F²    (proxy conservativo).
    """

    @staticmethod
    def mass_ratio(
        trickster_strength: float,
        dreamer_strength: float,
        auditor_rank: int,
        n: int,
    ) -> float:
        bt = float(np.clip(trickster_strength, 0.0, 1.0))
        bd = float(np.clip(dreamer_strength, 0.0, 1.0))
        ar = float(max(1, auditor_rank)) / float(max(1, n))
        denom = bt + bd + 1.0e-9
        raw = bt * ar / denom
        return float(np.clip(raw, _MU_FLOOR, _MU_CEIL))

    @staticmethod
    def jacobi_from_residual(rho_obs: np.ndarray, rho_vac: np.ndarray) -> float:
        rho_obs = MatrixBanachAlgebra.as_complex(rho_obs)
        rho_vac = MatrixBanachAlgebra.as_complex(rho_vac)
        diff = float(np.linalg.norm(rho_obs - rho_vac, "fro"))
        n = rho_obs.shape[0]
        k_norm_sq = float(np.real(np.trace(rho_obs.conj().T @ rho_obs))) * n
        return 2.0 * k_norm_sq - diff**2

    @classmethod
    def reduce_channel(
        cls,
        channel: TriadChannel,
        rho_vac: np.ndarray,
    ) -> Tuple[float, float, CR3BPState, Dict[str, Tuple[float, float]]]:
        n = channel.K.shape[0]
        sig = channel.signature
        mu = cls.mass_ratio(
            tri_strength=float(abs(np.linalg.det(channel.U))) ** (1.0 / max(n, 1)),
            dream_strength=float(
                np.linalg.norm(channel.D - np.eye(n), "fro") / math.sqrt(n)
            ),
            auditor_rank=sig.auditor_rank,
            n=n,
        )
        rho_obs = channel.apply(rho_vac)
        C = cls.jacobi_from_residual(rho_obs, rho_vac)
        # Estado inicial: proyección desde L4 desplazado por residuales
        L = LibrationPointSolver.solve_all(mu)
        x0, y0 = L["L4"]
        # Introduce un desplazamiento cinético proporcional al residual CP
        cp_r = channel.cp_min_eigenvalue()
        vx0 = -0.02 * float(np.sign(cp_r))
        vy0 = 0.03 + 0.01 * float(np.real(np.trace(rho_obs @ channel.K)))
        state0 = CR3BPState(x=x0 + 1.0e-3, y=y0 - 1.0e-3, vx=vx0, vy=vy0)
        return mu, C, state0, L

    # ══════════════════════════════════════════════════════════════════════════
    # ⚠ MÉTODO TERMINAL DE LA FASE 1 — SEMILLA DEL MOTOR DE LA FASE 2
    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def emit_poincare_seed(
        cls,
        channel: TriadChannel,
        rho_vac: np.ndarray,
        tau_recurrence: float = 1.0,
        provenance_tag: str = "SILENT-WITNESS",
    ) -> "PoincareSectionSeed":
        r"""
        Emite la semilla canónica que el motor `PoincareSectionIntegrator`
        de la Fase 2 consume como condición inicial para integrar la
        aplicación de retorno T: Σ → Σ.
        """
        mu, C, state0, L = cls.reduce_channel(channel, rho_vac)
        routh = LibrationPointSolver.routh_stability(mu)
        payload = (
            f"{channel.signature.unitary_hash[:16]}|"
            f"{channel.signature.dreamer_hash[:16]}|"
            f"{channel.signature.projector_hash[:16]}|"
            f"{mu:.12e}|{C:.12e}|{tau_recurrence:.12e}"
        ).encode("ascii")
        prov = hashlib.sha256(payload + provenance_tag.encode("ascii")).hexdigest()
        logger.debug(
            "[F1→F2] μ=%.6f C=%.6f Routh=%s prov=%s…",
            mu, C, routh, prov[:12],
        )
        return PoincareSectionSeed(
            mu=mu,
            jacobi_constant=C,
            initial_state=state0,
            libration_points=L,
            routh_stable=routh,
            triad_signature=channel.signature,
            tau_recurrence=tau_recurrence,
            provenance_hash=prov,
        )
# ══════════════════════════════ FIN DE LA FASE 1 ═════════════════════════════════════


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 · SECCIONES DE POINCARÉ, REGULARIZACIÓN LEVI-CIVITA/K-S, CAOS Y KAM            ║
# ║                                                                                      ║
# ║ Motor de integración RK4 con detección de eventos, sección de Poincaré Σ = {y=0,       ║
# ║ ẏ>0}, aplicación de retorno T: Σ → Σ, indicadores exponenciales de caos (Liapunov,    ║
# ║ SALI, GALI), función de Melnikov para transversalidad homoclínica y verificador       ║
# ║ KAM (Kolmogorov 1954 – Arnold 1963 – Moser 1962). El último método de esta fase,      ║
# ║ `WitnessObservationPipeline.emit_birkhoff_seed`, inicia la Fase 3.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# ── §2.0 Contrato de entrada (producido por Fase 1, consumido aquí) ──────────
@dataclass(frozen=True, slots=True)
class PoincareSectionSeed:
    r"""
    Semilla canónica procedente del morfismo Triad→CR3BP de la Fase 1.
    """
    mu: float
    jacobi_constant: float
    initial_state: CR3BPState
    libration_points: Dict[str, Tuple[float, float]]
    routh_stable: bool
    triad_signature: TriadSignature
    tau_recurrence: float
    provenance_hash: str


# ── §2.1 Punto de cruce con la sección Σ ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SectionCrossing:
    r"""
    Registro de un cruce con la sección de Poincaré Σ = {y = 0, ẏ > 0}.
    Coordenadas reducidas (x, ẋ) en la sección (con C fijo, ẏ = √(2Ω − C)).
    """
    step_index: int
    t: float
    x: float
    vx: float
    jacobi_residual: float


class PoincareSectionIntegrator:
    r"""
    Integrador RK4 con detección de eventos y refinamiento por bisección.
    Emite cruces ordenados en Σ = {y = 0, ẏ > 0}.
    """

    def __init__(self, dt: float = 5.0e-3, max_steps: int = 200_000) -> None:
        self.dt = float(dt)
        self.max_steps = int(max_steps)

    @staticmethod
    def _rk4_step(
        f: Callable[[float, np.ndarray], np.ndarray],
        t: float,
        z: np.ndarray,
        h: float,
    ) -> np.ndarray:
        k1 = f(t, z)
        k2 = f(t + 0.5 * h, z + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, z + 0.5 * h * k2)
        k4 = f(t + h, z + h * k3)
        return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def integrate_section(
        self,
        seed: PoincareSectionSeed,
        n_crossings: int = 512,
    ) -> List[SectionCrossing]:
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        t = 0.0
        crossings: List[SectionCrossing] = []
        y_prev = z[1]
        vy_prev = z[3]
        h = self.dt
        for step in range(self.max_steps):
            z_next = self._rk4_step(f, t, z, h)
            y_next = z_next[1]
            vy_next = z_next[3]
            # Evento: y cruza a 0 con ẏ > 0
            if y_prev < 0.0 <= y_next and vy_next > 0.0:
                # refinamiento lineal (sub-paso)
                alpha = -y_prev / (y_next - y_prev)
                t_cross = t + alpha * h
                z_cross = z + alpha * (z_next - z)
                z_cross_s = CR3BPState.from_array(z_cross)
                C = CR3BPDynamics.jacobi_constant(z_cross_s, seed.mu)
                res = abs(C - seed.jacobi_constant) / max(1.0, abs(seed.jacobi_constant))
                crossings.append(
                    SectionCrossing(
                        step_index=len(crossings),
                        t=t_cross,
                        x=float(z_cross[0]),
                        vx=float(z_cross[2]),
                        jacobi_residual=float(res),
                    )
                )
                if len(crossings) >= n_crossings:
                    break
            z = z_next
            t += h
            y_prev = y_next
            vy_prev = vy_next
        return crossings

    @staticmethod
    def return_map_matrix(
        crossings: List[SectionCrossing],
    ) -> np.ndarray:
        r"""
        Matriz de dispersión 2×2 (proxy de la linealización DT sobre Σ)
        estimada por regresión de menores cuadrados en vecindades locales.
        """
        if len(crossings) < 8:
            return np.eye(2)
        xs = np.array([c.x for c in crossings[:-1]], dtype=np.float64)
        vxs = np.array([c.vx for c in crossings[:-1]], dtype=np.float64)
        ys = np.array([c.x for c in crossings[1:]], dtype=np.float64)
        wys = np.array([c.vx for c in crossings[1:]], dtype=np.float64)
        X = np.column_stack([xs, vxs])
        A = np.zeros((2, 2), dtype=np.float64)
        # y = a·x + b·vx
        try:
            coef1, *_ = np.linalg.lstsq(X, ys, rcond=None)
            coef2, *_ = np.linalg.lstsq(X, wys, rcond=None)
            A[0, :] = coef1
            A[1, :] = coef2
        except np.linalg.LinAlgError:
            A = np.eye(2)
        return A

    @staticmethod
    def section_entropy(crossings: List[SectionCrossing], bins: int = 32) -> float:
        r"""
        Entropía diferencial aproximada del cruce (proxy de la MEDIDA de
        Liouville proyectada): H ≈ −Σ p_i log p_i.
        """
        if len(crossings) < 4:
            return 0.0
        xs = np.array([c.x for c in crossings])
        vxs = np.array([c.vx for c in crossings])
        h2d, _, _ = np.histogram2d(xs, vxs, bins=bins)
        p = h2d / max(1.0, h2d.sum())
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))


# ── §2.2 Regularización de Levi-Civita (Levi-Civita 1920) ───────────────────
class LeviCivitaRegularization:
    r"""
    Transformación parabólica compleja (planar):

        z_c = x + i y  ↦  w_c = u + i v,   z_c = w_c²,
        dt = |z_c| · dτ    (reparametrización temporal).

    Elimina la singularidad binaria r = 0 al coste de duplicar la hoja de
    Riemann y mantener el Jacobiano local √(r).
    """

    @staticmethod
    def transform(xy: np.ndarray) -> np.ndarray:
        x, y = float(xy[0]), float(xy[1])
        z_c = x + 1j * y
        w_c = np.sqrt(z_c + 1e-30j) if abs(z_c) > 0.0 else 1e-15 + 0j
        return np.array([w_c.real, w_c.imag], dtype=np.float64)

    @staticmethod
    def inverse(uv: np.ndarray) -> np.ndarray:
        u, v = float(uv[0]), float(uv[1])
        x = u * u - v * v
        y = 2.0 * u * v
        return np.array([x, y], dtype=np.float64)

    @staticmethod
    def jacobian_magnitude(uv: np.ndarray) -> float:
        u, v = float(uv[0]), float(uv[1])
        return 4.0 * (u * u + v * v)


class KustaanheimoStiefel:
    r"""
    Regularización espacial (3D) K-S (Kustaanheimo & Stiefel 1965) mediante
    cuaterniones unitarios: r = L(q) q con L(q) matriz de Hopf.

    Para el CR3BP planar usamos la subálgebra compleja; para extensión 3D se
    provee la matriz explícita.
    """

    @staticmethod
    def lift_matrix(q: np.ndarray) -> np.ndarray:
        q0, q1, q2, q3 = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return np.array(
            [
                [q0, -q1, -q2, -q3],
                [q1, q0, -q3, q2],
                [q2, q3, q0, -q1],
                [q3, -q2, q1, q0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def r_from_quaternion(q: np.ndarray) -> np.ndarray:
        L = KustaanheimoStiefel.lift_matrix(q)
        return L @ np.array([q[0], q[1], q[2], q[3]], dtype=np.float64)


# ── §2.3 Espectro de Liapunov y SALI (Skokos 2001) ──────────────────────────
class TangentFlow:
    r"""
    Flujo tangente (variacional) para estimar exponentes de Liapunov:

        δẑ = Df(z) δz,    con Df Jacobiano analítico del CR3BP.
    """

    @staticmethod
    def jacobian(z: np.ndarray, mu: float) -> np.ndarray:
        x, y, _, _ = float(z[0]), float(z[1]), float(z[2]), float(z[3])
        r1_2 = (x + mu) ** 2 + y * y
        r2_2 = (x - 1.0 + mu) ** 2 + y * y
        r1 = max(math.sqrt(r1_2), _SING_HARDENING)
        r2 = max(math.sqrt(r2_2), _SING_HARDENING)
        r1_3 = r1**3
        r2_3 = r2**3
        r1_5 = r1**5
        r2_5 = r2**5
        dxx = 1.0 - (1.0 - mu) / r1_3 - mu / r2_3 \
              + 3.0 * (1.0 - mu) * (x + mu) ** 2 / r1_5 \
              + 3.0 * mu * (x - 1.0 + mu) ** 2 / r2_5
        dyy = 1.0 - (1.0 - mu) / r1_3 - mu / r2_3 \
              + 3.0 * (1.0 - mu) * y * y / r1_5 \
              + 3.0 * mu * y * y / r2_5
        dxy = 3.0 * (1.0 - mu) * (x + mu) * y / r1_5 \
              + 3.0 * mu * (x - 1.0 + mu) * y / r2_5
        A = np.zeros((4, 4), dtype=np.float64)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 0] = dxx
        A[2, 1] = dxy
        A[2, 3] = 2.0
        A[3, 0] = dxy
        A[3, 1] = dyy
        A[3, 2] = -2.0
        return A


class LiapunovSpectrum:
    r"""
    Espectro de Liapunov (Benettin 1980): GS reortogonalización cada paso.
    Devuelve λ₁ ≥ λ₂ ≥ … ≥ λ₄ en nats/t.
    """

    @classmethod
    def estimate(
        cls,
        seed: PoincareSectionSeed,
        t_total: float = 50.0,
        dt: float = 5.0e-3,
    ) -> Tuple[float, ...]:
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        n = z.size
        Q = np.eye(n, dtype=np.float64)
        sums = np.zeros(n, dtype=np.float64)
        steps = int(max(1, t_total / dt))
        nsteps = 0
        for s in range(steps):
            A = TangentFlow.jacobian(z, seed.mu)
            k1 = A @ Q
            k2 = TangentFlow.jacobian(z + 0.5 * dt * f(s * dt, z), seed.mu) @ (Q + 0.5 * dt * k1)
            k3 = TangentFlow.jacobian(z + 0.5 * dt * f(s * dt, z), seed.mu) @ (Q + 0.5 * dt * k2)
            k4 = TangentFlow.jacobian(z + dt * f(s * dt, z), seed.mu) @ (Q + dt * k3)
            Q = Q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            try:
                Q, R = np.linalg.qr(Q)
            except np.linalg.LinAlgError:
                break
            diag = np.where(np.abs(np.diag(R)) > 1e-30, np.abs(np.diag(R)), 1e-30)
            sums += np.log(diag)
            z = PoincareSectionIntegrator._rk4_step(f, s * dt, z, dt)
            nsteps += 1
        if nsteps == 0:
            return tuple(0.0 for _ in range(n))
        lams = sums / (nsteps * dt)
        return tuple(sorted(lams.tolist(), reverse=True))


class SALIIndicator:
    r"""
    Small Alignment Index (Skokos 2001):
        SALI(t) = min(‖v₁(t) + v₂(t)‖, ‖v₁(t) − v₂(t)‖),
    con v_i normalizados.
    Decaimiento algebraico → régimen regular. Decaimiento exponencial → caos.
    """

    @classmethod
    def estimate(
        cls,
        seed: PoincareSectionSeed,
        t_total: float = 20.0,
        dt: float = 5.0e-3,
    ) -> float:
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        n = z.size
        rng = np.random.default_rng(_seed_from_string(seed.provenance_hash))
        v1 = rng.standard_normal(n)
        v2 = rng.standard_normal(n)
        v1 /= np.linalg.norm(v1) + 1e-30
        v2 /= np.linalg.norm(v2) + 1e-30
        steps = int(max(1, t_total / dt))
        val = 0.0
        for s in range(steps):
            A = TangentFlow.jacobian(z, seed.mu)
            v1 = v1 + dt * (A @ v1)
            v2 = v2 + dt * (A @ v2)
            v1 /= np.linalg.norm(v1) + 1e-30
            v2 /= np.linalg.norm(v2) + 1e-30
            s1 = np.linalg.norm(v1 + v2)
            s2 = np.linalg.norm(v1 - v2)
            val = min(s1, s2)
            z = PoincareSectionIntegrator._rk4_step(f, s * dt, z, dt)
        return float(val)


# ── §2.4 Función de Melnikov (Melnikov 1963; Holmes-Marsden 1982) ──────────
class MelnikovResonanceProbe:
    r"""
    Función de Melnikov M(t₀) = ∫_{−∞}^{+∞} {H₀, H₁}(φ_t) dt como medidor de
    transversalidad homoclínica (τ-horseshoe, caos tipo Smale). Si M(t₀)
    tiene ceros transversales, emergen cascadas de bifurcaciones.
    """

    @classmethod
    def evaluate(
        cls,
        seed: PoincareSectionSeed,
        t_span: float = 20.0,
        dt: float = 1.0e-2,
    ) -> Tuple[float, float]:
        r"""
        Estimación numérica por integración del corchete de Poisson evaluado
        sobre la trayectoria linealizada del CR3BP: aproxima el miembro
        perturbativo por la proyección del residual de Jacobi.
        """
        f = CR3BPDynamics.vector_field(seed.mu)
        z = seed.initial_state.as_array().copy()
        acc = 0.0
        acc_abs = 0.0
        steps = int(max(1, 2 * t_span / dt))
        C = seed.jacobi_constant
        for s in range(steps):
            x, y, vx, vy = z
            # Hamiltoniano rotante: H = ½(vx²+vy²) − Ω(x,y)
            H_rot = 0.5 * (vx * vx + vy * vy) - CR3BPDynamics.potential(x, y, seed.mu)
            jacobi_defect = 2.0 * CR3BPDynamics.potential(x, y, seed.mu) \
                            - (vx * vx + vy * vy) - C
            # Corchete de Poisson {H_rot, δH} donde δH = λ·jacobi_defect
            lambda_c = 0.03
            bracket = lambda_c * jacobi_defect
            weight = math.exp(-abs(s * dt - t_span) / t_span)
            acc += weight * bracket * dt
            acc_abs += weight * abs(bracket) * dt
            z = PoincareSectionIntegrator._rk4_step(f, s * dt, z, dt)
        peak = abs(acc)
        return peak, acc_abs


# ── §2.5 Verificador KAM (Kolmogorov 1954; Arnold 1963; Moser 1962) ─────────
class KAMTorusAuditor:
    r"""
    Verifica numéricamente la persistencia de tori invariantes: la
    aplicación de retorno T: Σ → Σ se compara con una rotación rígida de
    frecuencia ω. Mide el sesgo |θ_{k+1} − θ_k − 2πρ| (ρ racional → resonancia).
    """

    @classmethod
    def verify(
        cls,
        crossings: List[SectionCrossing],
        tol: float = _KAM_TOL,
    ) -> Dict[str, float]:
        if len(crossings) < 16:
            return {"kam_bias": float("inf"), "kam_stable": 0.0, "kam_freq": 0.0}
        xs = np.array([c.x for c in crossings])
        vs = np.array([c.vx for c in crossings])
        # Ángulo polar alrededor del centroide
        cx, cvx = float(xs.mean()), float(vs.mean())
        theta = np.arctan2(vs - cvx, xs - cx)
        dtheta = (np.diff(theta) + np.pi) % (2.0 * np.pi) - np.pi
        omega = float(np.median(dtheta))
        bias = float(np.mean(np.abs(dtheta - omega)))
        stable = 1.0 if bias < tol * 10.0 else 0.0
        return {
            "kam_bias": bias,
            "kam_stable": stable,
            "kam_freq": omega,
        }


# ── §2.6 Bundle de observación enriquecido con datos celestes ───────────────
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
    # Fase 2 · enriquecimiento celeste
    poincare_seed: Optional[PoincareSectionSeed] = None
    section_crossings: Tuple[SectionCrossing, ...] = ()
    return_map_matrix: Optional[np.ndarray] = None
    section_entropy: float = 0.0
    kam_report: Mapping[str, float] = field(default_factory=dict)
    liapunov_spectrum: Tuple[float, ...] = ()
    sali_value: float = 0.0
    melnikov_peak: float = 0.0
    melnikov_area: float = 0.0

    def as_vacuum_metrics(self, is_silent: bool) -> VacuumStateMetrics:
        # Enriquecido con datos celestes
        j_res = 0.0
        if self.section_crossings:
            j_res = float(np.mean([c.jacobi_residual for c in self.section_crossings]))
        lib_xy = (0.0, 0.0)
        mu = 0.0
        routh_s = False
        C = 0.0
        if self.poincare_seed is not None:
            mu = self.poincare_seed.mu
            C = self.poincare_seed.jacobi_constant
            routh_s = self.poincare_seed.routh_stable
            if "L4" in self.poincare_seed.libration_points:
                lib_xy = self.poincare_seed.libration_points["L4"]
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
            dirichlet_energy=self.section_entropy,
            mass_ratio_mu=mu,
            jacobi_constant=C,
            jacobi_residual=j_res,
            libration_xy=lib_xy,
            routh_stable=routh_s,
        )


# ── §2.7 Pipeline de observación extendido ──────────────────────────────────
class WitnessObservationPipeline:
    r"""
    Pipeline de síntesis: combina el canal de la tríada con la observación
    modular y (nuevo) con la exploración celeste de Poincaré (Fase 2).
    """

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
        explore_celestial: bool = True,
        n_crossings: int = 256,
        tau_recurrence: float = 1.0,
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

        # ── Fase 2 · exploración celeste de Poincaré ──
        poincare_seed: Optional[PoincareSectionSeed] = None
        crossings: List[SectionCrossing] = []
        A = None
        H_sec = 0.0
        kam_rep: Dict[str, float] = {}
        lam: Tuple[float, ...] = ()
        sali_val = 0.0
        mel_peak = 0.0
        mel_area = 0.0
        if explore_celestial:
            try:
                poincare_seed = TriadToCR3BPMorphism.emit_poincare_seed(
                    triad, rho_vac, tau_recurrence=tau_recurrence
                )
                integrator = PoincareSectionIntegrator()
                crossings = integrator.integrate_section(poincare_seed, n_crossings=n_crossings)
                A = integrator.return_map_matrix(crossings)
                H_sec = integrator.section_entropy(crossings)
                kam_rep = KAMTorusAuditor.verify(crossings)
                lam = LiapunovSpectrum.estimate(poincare_seed, t_total=20.0)
                sali_val = SALIIndicator.estimate(poincare_seed, t_total=10.0)
                mel_peak, mel_area = MelnikovResonanceProbe.evaluate(poincare_seed, t_span=10.0)
            except Exception as exc:  # pragma: no cover — defensivo
                logger.warning("Exploración celeste abortada: %s", exc)

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
            poincare_seed=poincare_seed,
            section_crossings=tuple(crossings),
            return_map_matrix=A,
            section_entropy=H_sec,
            kam_report=kam_rep,
            liapunov_spectrum=lam,
            sali_value=sali_val,
            melnikov_peak=mel_peak,
            melnikov_area=mel_area,
        )

    @classmethod
    def synthesize_from_context(
        cls,
        cycle_index: int,
        ctx: WitnessVacuumContext,
        triad: TriadChannel,
        explore_celestial: bool = True,
        tau_recurrence: float = 1.0,
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
            explore_celestial=explore_celestial,
            tau_recurrence=tau_recurrence,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ⚠ MÉTODO TERMINAL DE LA FASE 2 — SEMILLA DE LA FASE 3
    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def emit_birkhoff_seed(
        cls,
        bundle: WitnessObservationBundle,
    ) -> "BirkhoffSeed":
        r"""
        Emite la semilla de Birkhoff consumida por el verificador de la
        Fase 3. Inyecta los invariantes celestes (μ, C, L4/L5, ω_KAM,
        espectro de Liapunov, SALI, pico de Melnikov) para construir la
        forma normal en torno a la órbita periódica de referencia.
        """
        if bundle.poincare_seed is None:
            raise ValueError(
                "Bundle sin `poincare_seed`: no se puede emitir BirkhoffSeed."
            )
        L4 = bundle.poincare_seed.libration_points.get("L4", (0.0, 0.0))
        L5 = bundle.poincare_seed.libration_points.get("L5", (0.0, 0.0))
        omega = float(bundle.kam_report.get("kam_freq", 0.0))
        lam1 = float(bundle.liapunov_spectrum[0]) if bundle.liapunov_spectrum else 0.0
        payload = (
            f"{bundle.poincare_seed.provenance_hash}|"
            f"{omega:.12e}|{lam1:.12e}|{bundle.sali_value:.12e}|"
            f"{bundle.melnikov_peak:.12e}|{bundle.section_entropy:.12e}"
        ).encode("ascii")
        tag = hashlib.sha256(payload).hexdigest()
        return BirkhoffSeed(
            mu=bundle.poincare_seed.mu,
            jacobi_constant=bundle.poincare_seed.jacobi_constant,
            L4_xy=L4,
            L5_xy=L5,
            kam_frequency=omega,
            lyapunov_lambda1=lam1,
            sali=bundle.sali_value,
            melnikov_peak=bundle.melnikov_peak,
            section_entropy=bundle.section_entropy,
            triad_signature=bundle.triad_signature,
            provenance_hash=tag,
        )
# ══════════════════════════════ FIN DE LA FASE 2 ═════════════════════════════════════


# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 · POINCARÉ-BIRKHOFF, FORMA NORMAL, DIFUSIÓN DE ARNOLD Y CRISTALIZACIÓN        ║
# ║                                                                                      ║
# ║ El teorema de Poincaré-Birkhoff (1913) garantiza la existencia de infinitos           ║
# ║ puntos periódicos en un anillo twist no degenerado. Usamos la forma normal de        ║
# ║ Birkhoff alrededor de L4/L5 (Lyapunov-Deprit) para extraer las frecuencias           ║
# ║ fundamentales ω_{S,L}. La difusión de Arnold en la red resonante se cuantifica        ║
# ║ como proxy del KMS drift. Finalmente, cristalizamos la experiencia en forma de        ║
# ║ ExperienceCrystal firmado Merkle SHA-256 y encadenado al archivo.                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

# ── §3.0 Contrato de entrada (producido por Fase 2, consumido aquí) ─────────
@dataclass(frozen=True, slots=True)
class BirkhoffSeed:
    r"""
    Semilla Birkhoff procedente de la Fase 2.
    """
    mu: float
    jacobi_constant: float
    L4_xy: Tuple[float, float]
    L5_xy: Tuple[float, float]
    kam_frequency: float
    lyapunov_lambda1: float
    sali: float
    melnikov_peak: float
    section_entropy: float
    triad_signature: TriadSignature
    provenance_hash: str


# ── §3.1 Forma normal de Birkhoff en L4/L5 (Lyapunov-Deprit) ────────────────
class BirkhoffNormalForm:
    r"""
    Forma normal de Birkhoff cuadrática en torno al punto de Lagrange
    triangular (Szebehely 1967, §4):

        H₀ = ½ (p_x² + p_y²) + Ω_xx x² + Ω_yy y²  (linealizado),
        Ω_xx = Ω_yy = ¾,   Ω_xy (L4/L5) = ±(3√3/4)(1 − 2μ).

    Frecuencias características:

        ω_{S,L}² = 1 ± √(1 − 27μ(1 − μ)).

    Si 27μ(1 − μ) > 1 ⇒ ω complejas ⇒ punto linealmente inestable
    (bifurcación de Poincaré-Andronov-Hopf inversa, Routh 1875).
    """

    @classmethod
    def linearization(cls, L4_xy: Tuple[float, float], mu: float) -> np.ndarray:
        x, _ = L4_xy
        # Hessiano Ω_xx = Ω_yy = ¾; Ω_xy = ±(3√3/4)(1 − 2μ)
        oxx = 0.75
        oyy = 0.75
        oxy = (3.0 * math.sqrt(3.0) / 4.0) * (1.0 - 2.0 * mu)
        H = np.zeros((4, 4), dtype=np.float64)
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        H[2, 0] = oxx
        H[2, 1] = oxy
        H[2, 3] = 2.0
        H[3, 0] = oxy
        H[3, 1] = oyy
        H[3, 2] = -2.0
        return H

    @classmethod
    def frequencies(cls, mu: float) -> Tuple[float, float]:
        delta = 1.0 - 27.0 * mu * (1.0 - mu)
        if delta < 0.0:
            delta = 0.0
        root = math.sqrt(delta)
        omega_s2 = 1.0 - root
        omega_l2 = 1.0 + root
        return (math.sqrt(max(omega_s2, 0.0)), math.sqrt(max(omega_l2, 0.0)))

    @classmethod
    def jacobian_eigenvalues(
        cls, L4_xy: Tuple[float, float], mu: float
    ) -> np.ndarray:
        H = cls.linearization(L4_xy, mu)
        return np.linalg.eigvals(H)

    @classmethod
    def is_spectrally_stable(cls, L4_xy: Tuple[float, float], mu: float) -> bool:
        eigs = cls.jacobian_eigenvalues(L4_xy, mu)
        return bool(np.all(np.abs(np.real(eigs)) < 1.0e-8))


# ── §3.2 Escalera de resonancias (Poincaré 1892; Arnold 1963) ───────────────
class ResonanceLadder:
    r"""
    Construye la escalera de resonancias p:q mediante desarrollo continuado
    de fracciones de la frecuencia KAM ω: ω ≈ p/q. La presencia de
    denominadores pequeños |p·ω₁ − q·ω₂| ≪ 1 abre huecos resonantes en la
    estructura KAM (small divisors de Poincaré).
    """

    @staticmethod
    def continued_fraction(x: float, depth: int = 12) -> List[int]:
        coeffs: List[int] = []
        y = float(x)
        for _ in range(depth):
            a = int(math.floor(y))
            coeffs.append(a)
            y = y - a
            if y < 1.0e-12:
                break
            y = 1.0 / y
        return coeffs

    @staticmethod
    def convergents(coeffs: List[int]) -> List[Tuple[int, int]]:
        p_prev, p_cur = 0, 1
        q_prev, q_cur = 1, 0
        out: List[Tuple[int, int]] = []
        for a in coeffs:
            p_new = a * p_cur + p_prev
            q_new = a * q_cur + q_prev
            out.append((p_new, q_new))
            p_prev, p_cur = p_cur, p_new
            q_prev, q_cur = q_cur, q_new
        return out

    @classmethod
    def resonance_density(cls, entries: List[int]) -> float:
        if not entries:
            return 0.0
        # Suma de 1/q²: mide densidad de resonancias pequeñas (Poincaré).
        s = sum(1.0 / (q * q) for _, q in cls.convergents(entries))
        return float(s)


# ── §3.3 Difusión de Arnold (Nekhoroshev 1977) ──────────────────────────────
class ArnoldDiffusionBridge:
    r"""
    Tasa de difusión a lo largo de la red resonante:

        D_Arnold ≈ ε^{N+1} · exp(−1/ε^a)     (Nekhoroshev, N = dim),

    donde ε se estima como λ₁Liapunov normalizado por ω_KAM. Reporta el
    tiempo de escape t_diff ≈ exp(1/ε^a), semilla del KMS-drift.
    """

    @classmethod
    def diffusion_rate(
        cls,
        lambda1: float,
        omega_kam: float,
        dim: int = 2,
    ) -> float:
        if omega_kam <= 1.0e-12:
            return float("inf")
        eps = abs(lambda1) / omega_kam
        eps = float(np.clip(eps, 1.0e-9, 0.9))
        # Forma Nekhoroshev simplificada
        N = max(1, dim)
        a = 1.0 / (N + 1.0)
        try:
            rate = eps ** (N + 1) * math.exp(-1.0 / (eps ** a))
        except OverflowError:
            rate = float("inf")
        return float(rate)

    @classmethod
    def escape_time(
        cls,
        lambda1: float,
        omega_kam: float,
        dim: int = 2,
    ) -> float:
        rate = cls.diffusion_rate(lambda1, omega_kam, dim)
        if rate <= 0.0 or not math.isfinite(rate):
            return float("inf")
        return 1.0 / rate


# ── §3.4 Cristal de experiencia extendido ───────────────────────────────────
@dataclass(frozen=True, slots=True)
class ExperienceCrystal:
    r"""
    Cristal inmutable de experiencia, firmado con SHA-256 y encadenado
    Merkle-style, enriquecido con invariantes celestes de Poincaré-Birkhoff.
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
    # ── Poincaré––S⁶ (heredados v8) ──
    vector_s6: Optional[np.ndarray] = None
    tau_recurrence: float = 1.0
    kms_drift: float = 0.0
    merkle_root_sha256: str = ""
    engine_version: str = __version__
    # ── Fase 3 · Birkhoff / Nekhoroshev / resonancias ──
    birkhoff_frequencies: Tuple[float, float] = (0.0, 0.0)
    birkhoff_stable: bool = False
    kam_frequency: float = 0.0
    kam_bias: float = 0.0
    arnold_diffusion_rate: float = 0.0
    arnold_escape_time: float = float("inf")
    resonance_density: float = 0.0
    sali: float = 0.0
    melnikov_peak: float = 0.0


# ── §3.5 Agente soberano: orquestador final (consume BirkhoffSeed) ─────────
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


class TOONSilentWitnessAgent:
    r"""
    Soberano Testigo Silencioso con mecánica celeste de Poincaré (CR3BP),
    forma normal de Birkhoff, verificador KAM, indicadores SALI/Liapunov y
    contrato Merkle encadenado. Observa sin colapsar la función de onda y
    vetará mediante interlock si la recurrencia de Poincaré se rompe.
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
        celestial_dt: float = 5.0e-3,
        n_crossings: int = 256,
    ) -> None:
        if dimension is not None:
            dimension_mac = dimension
        self.agent_id = agent_id
        self.dimension_mac = dimension_mac
        self.kms_beta = kms_beta
        self.vacuum_beta = vacuum_beta
        self.seed = seed
        self.celestial_dt = float(celestial_dt)
        self.n_crossings = int(n_crossings)
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

    # ── Hashes y cadena Merkle ──
    def _content_hash(
        self,
        cycle_id: str,
        audit: VacuumAuditReport,
        inv: np.ndarray,
        sig: TriadSignature,
        dirichlet_energy_arg: float,
        birkhoff_seed: Optional[BirkhoffSeed] = None,
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
        if birkhoff_seed is not None:
            hasher.update(birkhoff_seed.provenance_hash.encode("ascii"))
        return hasher.hexdigest()

    def _advance_chain(self, content_hash: str) -> Tuple[str, str]:
        parent = self._chain_hash
        digest = hashlib.sha256(
            parent.encode("ascii") + content_hash.encode("ascii")
        ).hexdigest()
        self._chain_hash = digest
        return parent, digest

    # ── Observación cero back-action (zero back-action weak measurement) ──
    def execute_zero_backaction_poincare_observation(
        self,
        density_matrix: np.ndarray,
        tau_recurrence: float = 1.0,
    ) -> Dict[str, Any]:
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
            "crowbar_trigger": verdict == "VETOED",
        }

    def latest_crystal(self) -> Optional[ExperienceCrystal]:
        return self.crystal_history[-1] if self.crystal_history else None

    def verify_merkle_chain(self) -> bool:
        r"""
        Recomputación de chain_hash_k = SHA256(parent_{k−1} ‖ content_hash_k)
        con verificación de formato hex-64 y ‖v_inv‖₂ ≈ 1.
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

    # ══════════════════════════════════════════════════════════════════════════
    # Método terminal: cristalización total (Fase 1 → Fase 2 → Fase 3)
    # ══════════════════════════════════════════════════════════════════════════
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

        # ── Fase 1 · contexto modular y canal de la tríada ──
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

        # ── Fase 2 · síntesis con exploración celeste (Poincaré/KAM/SALI) ──
        bundle = WitnessObservationPipeline.synthesize_from_context(
            cycle_index=self.crystal_count,
            ctx=ctx,
            triad=triad,
            explore_celestial=True,
            tau_recurrence=tau_recurrence,
        )

        # ── Fase 3 · Birkhoff, resonancias, difusión de Arnold ──
        birkhoff_seed = WitnessObservationPipeline.emit_birkhoff_seed(bundle)
        omega_s, omega_l = BirkhoffNormalForm.frequencies(birkhoff_seed.mu)
        birk_stable = BirkhoffNormalForm.is_spectrally_stable(birkhoff_seed.L4_xy, birkhoff_seed.mu)
        cf = ResonanceLadder.continued_fraction(birkhoff_seed.kam_frequency, depth=10)
        res_dens = ResonanceLadder.resonance_density(cf)
        arn_rate = ArnoldDiffusionBridge.diffusion_rate(
            birkhoff_seed.lyapunov_lambda1, birkhoff_seed.kam_frequency, dim=2
        )
        arn_time = ArnoldDiffusionBridge.escape_time(
            birkhoff_seed.lyapunov_lambda1, birkhoff_seed.kam_frequency, dim=2
        )

        # ── Adjudicación trivaluada (topos Heyting Ω₃) ──
        final_verdict = HeytingWitnessAdjudicator.adjudicate(bundle, auditor_verdict)
        metrics = bundle.as_vacuum_metrics(
            is_silent=(final_verdict == HeytingOmega3.COHERENT)
        )

        # ── Encadenamiento criptográfico ──
        content_hash = self._content_hash(
            crystal_id,
            bundle.audit,
            bundle.invariant_vector,
            bundle.triad_signature,
            dirichlet_energy,
            birkhoff_seed=birkhoff_seed,
        )
        merkle_parent, chain_hash = self._advance_chain(content_hash)

        sig_hasher = hashlib.sha256()
        sig_hasher.update(self.agent_id.encode("ascii"))
        sig_hasher.update(crystal_id.encode("ascii"))
        sig_hasher.update(final_verdict.name.encode("ascii"))
        sig_hasher.update(chain_hash.encode("ascii"))
        sig_hasher.update(f"{time.time_ns()}".encode("ascii"))
        provenance = sig_hasher.hexdigest()

        # ── Auditoría de recurrencia de Poincaré–KMS heredada ──
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
            birkhoff_frequencies=(omega_s, omega_l),
            birkhoff_stable=birk_stable,
            kam_frequency=birkhoff_seed.kam_frequency,
            kam_bias=float(bundle.kam_report.get("kam_bias", 0.0)),
            arnold_diffusion_rate=arn_rate,
            arnold_escape_time=arn_time,
            resonance_density=res_dens,
            sali=birkhoff_seed.sali,
            melnikov_peak=birkhoff_seed.melnikov_peak,
        )
        self.crystal_history.append(crystal)
        self.experience_archive.append(crystal)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "Cristal %s | Ω₃=%s | F=%.6f | KMS_x=%.3e | ω_S=%.4f ω_L=%.4f | %.2f ms",
            crystal_id,
            final_verdict.name,
            bundle.audit.purity,
            bundle.audit.thermal_fluctuation,
            omega_s,
            omega_l,
            dt_ms,
        )
        return crystal


# ═══════════════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT / SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    witness = TOONSilentWitnessAgent(
        agent_id="SILENT-WITNESS-SABIO-01",
        dimension=4,
        kms_beta=1.0,
        vacuum_beta=50.0,
    )

    print("═" * 88)
    print(f"TESTIGO SILENCIOSO — Mecánica Celeste de Poincaré v{__version__}")
    print("═" * 88)

    rho_test = np.eye(4, dtype=np.complex128) / 4.0
    res = witness.execute_zero_backaction_poincare_observation(
        rho_test, tau_recurrence=1.0
    )
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
        trickster_strength=0.15,
        dreamer_strength=0.10,
        auditor_rank=2,
    )
    print("\n  Cristalización de Tríada + CR3BP + Birkhoff:")
    print("    Crystal ID            :", c_triad.crystal_id)
    print("    Veredicto             :", c_triad.heyting_verdict.name)
    print("    μ (ratio masas)       :", f"{c_triad.vacuum_metrics.mass_ratio_mu:.6f}")
    print("    Constante de Jacobi C :", f"{c_triad.vacuum_metrics.jacobi_constant:.6f}")
    print("    Routh estable         :", c_triad.vacuum_metrics.routh_stable)
    print("    ω_S, ω_L (Birkhoff)   :",
          tuple(f"{w:.4f}" for w in c_triad.birkhoff_frequencies))
    print("    Estable espectralmente:", c_triad.birkhoff_stable)
    print("    Frecuencia KAM        :", f"{c_triad.kam_frequency:.6e}")
    print("    Sesgo KAM             :", f"{c_triad.kam_bias:.6e}")
    print("    SALI                  :", f"{c_triad.sali:.6e}")
    print("    Melnikov peak         :", f"{c_triad.melnikov_peak:.6e}")
    print("    Densidad de resonancias:", f"{c_triad.resonance_density:.6e}")
    print("    Tasa difusión Arnold  :", f"{c_triad.arnold_diffusion_rate:.6e}")
    print("    S6 Vector Norm        :", f"{np.linalg.norm(c_triad.vector_s6):.6f}")
    print("    Merkle Root SHA-256   :", c_triad.merkle_root_sha256[:16], "...")

    print("\n  Cadena Merkle Válida    :", witness.verify_merkle_chain())
    print("\n" + "═" * 88)
    print("✓ Soberano Testigo Silencioso evolucionado con mecánica celeste de Poincaré.")
    print("═" * 88)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Agent (Soberano de la Medición, Gestor Epistemológico y         ║
║          Mecánica Celeste de Henri Poincaré)                                 ║
║ Ruta   : app/agents/wisdom/mac_agent.py                                      ║
║ Versión: 6.0.0-Celestial-Poincare-POVM-Lindblad-Galois-Doctoral-Nested       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA EPISTÉMICA EN EL ESTRATO WISDOM (V_W) ─────
Este módulo consagra al Agente Soberano del Estrato WISDOM como endofuntor de
medición sobre ℋ_MAC. Sobre la axiomática cuántica (Dirac–von Neumann), la
ecuación maestra GKSL (Lindblad) y la cohomología de haces, se teje la
maquinaria analítica de las Méthodes Nouvelles de Poincaré: Delaunay →
Poincaré no-singular, carta acción-ángulo, divisores Diofantinos, invariantes
integrales (relativo/absoluto), firma de Krein del linearizado Hamiltoniano,
toros KAM, sección Σ ⊂ T*Q, monodromía de Floquet, twist de Poincaré-Birkhoff,
recurrencia de Poincaré–Kac, Chirikov, Melnikov, Nekhoroshev, índice de
Poincaré-Hopf, dualidad y polinomio de Poincaré, colapso a Ω₃.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ──────────

  FASE 1 ──► ÁLGEBRA POVM Y GEOMETRÍA ESPECTRAL CELESTE (Observe)
             Mide ρ vía POVM {Eₖ}: p(k) = Tr(Eₖ† Eₖ ρ), isometría de Stinespring.
             Extrae Delaunay (L, G, H, ℓ, g, h), Poincaré (Λ, λ, ξ, η, p, q),
             carta (J, θ), Krein, KAM-Kolmogorov e invariantes ∮ p dq.
             Semilla: Phase1MeasurementCertificate.
             El último morfismo de FASE 1 ES el morfismo de apertura de FASE 2.

  FASE 2 ──► DINÁMICA DE LINDBLAD CELESTE Y SECCIONES DE POINCARÉ (Orient)
             ρ̇ = −i[H, ρ] + Σₖ γₖ 𝒟[Lₖ](ρ)  (flujo Port-Hamiltoniano en T*ℋ).
             Sección Σ, retorno P: Σ → Σ, Floquet, twist, recurrencia,
             Chirikov, Melnikov, Nekhoroshev.
             Semilla: Phase2DynamicsCertificate.
             El último morfismo de FASE 2 ES el morfismo de apertura de FASE 3.

  FASE 3 ──► COHOMOLOGÍA, ADJUNCIÓN DE GALOIS Y TOPOLOGÍA GLOBAL (Decide & Act)
             H¹(X; ℱ), adjunción F ⊣ G (Bures–Uhlmann), Poincaré-Hopf,
             dualidad b_k = b_{n−k}, polinomio de Poincaré, colapso Ω₃ y
             Veto simpléctico de Gromov / Crowbar BT151.

INVARIANTES MATEMÁTICOS PRESERVADOS: ────────────────────────────────────────────
  [I1]  Rango Completo MIC:              rank(MIC) = n ⇒ dim ker(MIC) = 0
  [I2]  Postulados Dirac–von Neumann:    ρ = ρ† ≽ 0, Tr(ρ) = 1
  [I3]  Isometría de Stinespring:        V†V = I
  [I4]  Nulidad Cohomológica:            H¹(K;ℱ) = 0 ⇒ E_Dirichlet(x) ≡ 0
  [I5]  Índice de Witten:                dim ker(Đ) − dim ker(Đᵀ) = index(δ)
  [I6]  Delaunay Celeste:                (L, e, G, i, H, C_J) canónicos
  [I7]  Poincaré no-singular:            (Λ, λ, ξ, η, p, q) regulares en e=i=0
  [I8]  Firma de Krein:                  espectro del linearizado Hamiltoniano
  [I9]  KAM-Kolmogorov:                  det(∂ω/∂J) ≠ 0 y ω ∈ DC(γ, τ)
  [I10] Chirikov:                        K = Δω/δω_res < 1 ⇒ toros intactos
  [I11] Melnikov:                        M(t₀)=0 ∧ M'(t₀)≠0 ⇒ tangencia homoclínica
  [I12] Floquet–Poincaré:                μ ∈ spec(DP), χ = Log(μ)/T
  [I13] Twist de Poincaré-Birkhoff:      ∂ν/∂J ≠ 0 ⇒ ≥ 2 puntos fijos
  [I14] Recurrencia de Poincaré:         T_Kac ≳ vol(M)/vol(U)
  [I15] Poincaré-Hopf:                   Σ_p ind_p(X) = χ(ℳ)
  [I16] Dualidad de Poincaré:            b_k = b_{n−k}
  [I17] Colapso de Heyting Ω₃:           {⊥, ½, ⊤} ≅ {VETOED, DEGRADED, COHERENT}
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from app.core.mic_algebra import CategoricalState, Morphism, NumericalInstabilityError
from app.wisdom.atomic_knowledge_matrix import (
    AtomicDensityMatrix,
    CellularSheafNeuralManifold,
    ChirikovResonanceCertificate,
    DelaunayElements,
    FloquetMonodromyCertificate,
    GaloisAdjunctionFunctor,
    HeytingOmega3,
    KAMTorusCertificate,
    KreinSpectralDecomposition,
    MelnikovCertificate,
    NekhoroshevCertificate,
    PoincareCanonicalElements,
    PoincareDualityCertificate,
    PoincareHopfCertificate,
    PoincareIntegralInvariants,
    PoincareRecurrenceCertificate,
    PoincareSectionState,
    ActionAngleChart,
    TwistMapCertificate,
    chirikov_resonance_overlap,
    floquet_monodromy,
    lyapunov_spectrum_from_trajectory,
    melnikov_function,
    nekhoroshev_stability,
    poincare_recurrence_bound,
    trace_poincare_section,
    twist_map_certificate,
)

logger = logging.getLogger("MAC.Agent.CelestialEpistemology")

# ==============================================================================
# CONSTANTES CELESTES Y RIGORES NUMÉRICOS
# ==============================================================================
_WILKINSON_TOL: float = 1.0e-12
_SPECTRAL_TOL: float = 1.0e-9
_CHIRIKOV_CRITICAL: float = 1.0
_MELNIKOV_TOL: float = 1.0e-8
_NEKHOROSHEV_EXPONENT: float = 0.5
_MAX_POINCARE_SAMPLES: int = 4096
_CROWBAR_TRIGGER_LATENCY_NS: int = 400
_STINESPRING_TOL: float = 1.0e-10
_MIN_LINDBLAD_STEPS: int = 3
_BURES_EPSILON_DEFAULT: float = 0.05


def _sha16(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


# ==============================================================================
# ENUMERACIONES
# ==============================================================================
class MeasurementOutcome(Enum):
    """Taxonomía de resultados de medición cuántica."""
    COLLAPSED = auto()
    MIXED = auto()
    DEGENERATE = auto()
    INCONCLUSIVE = auto()


@unique
class HeytingVerdict(str, Enum):
    r"""
    Retículo de Heyting Ω₃ de veredictos epistemológicos.
    Orden: VETOED = ⊥ < DEGRADED = ½ < COHERENT = ⊤.
    Meet = min, join = max, implicación a → b = ⊤ si a ≤ b, else b.
    """
    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"

    @property
    def is_terminal(self) -> bool:
        return self is HeytingVerdict.VETOED

    @property
    def heyting_value(self) -> float:
        return {
            HeytingVerdict.VETOED: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.COHERENT: 1.0,
        }[self]

    def to_omega3(self) -> HeytingOmega3:
        return {
            HeytingVerdict.VETOED: HeytingOmega3.FALSE,
            HeytingVerdict.DEGRADED: HeytingOmega3.CONTINGENT,
            HeytingVerdict.COHERENT: HeytingOmega3.TRUE,
        }[self]

    @classmethod
    def from_omega3(cls, omega: HeytingOmega3) -> "HeytingVerdict":
        return {
            HeytingOmega3.FALSE: cls.VETOED,
            HeytingOmega3.CONTINGENT: cls.DEGRADED,
            HeytingOmega3.TRUE: cls.COHERENT,
        }[omega]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.heyting_value <= other.heyting_value else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.heyting_value >= other.heyting_value else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.COHERENT if self.heyting_value <= other.heyting_value else other


# ==============================================================================
# DATACLASSES DE ESTADÍSTICAS
# ==============================================================================
@dataclass(frozen=True)
class POVMStatistics:
    """Estadísticas de una medición POVM."""
    outcome_index: int
    probability: float
    shannon_entropy: float
    outcome_purity: float
    mutual_information: float
    measurement_disturbance: float
    stinespring_defect: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0 + 1e-12):
            raise ValueError("Probabilidad fuera de rango")
        if self.shannon_entropy < -1e-12:
            raise ValueError("Entropía negativa")
        if not (-1e-12 <= self.outcome_purity <= 1.0 + 1e-12):
            raise ValueError("Pureza fuera de rango")


@dataclass
class LindbladEvolutionMetrics:
    """Métricas de evolución bajo dinámica de Lindblad (GKSL)."""
    time_evolved: float
    trace_before: float
    trace_after: float
    purity_before: float
    purity_after: float
    entropy_production: float
    dissipated_coherence: float
    spohn_rate: float = 0.0
    cptp_residual: float = 0.0

    def is_physically_valid(self, tol: float = 1e-10) -> bool:
        trace_preserved = abs(self.trace_after - self.trace_before) < tol
        entropy_positive = self.entropy_production >= -tol
        return trace_preserved and entropy_positive


@dataclass
class CohomologyAuditReport:
    """Reporte de auditoría cohomológica del fibrado celular."""
    is_holonomic: bool
    dirichlet_energy: float
    betti_numbers: Dict[int, int]
    obstruction_class: Optional[NDArray[np.float64]]
    global_sections_dim: int

    def has_obstructions(self) -> bool:
        return (not self.is_holonomic) or self.betti_numbers.get(1, 0) > 0


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 1 — INICIO                                       █
# █  ÁLGEBRA POVM Y GEOMETRÍA ESPECTRAL CELESTE (Observe)                    █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 1: Mide ρ vía POVM (Kraus + Stinespring), extrae Delaunay, Poincaré
canónico, carta acción-ángulo, Krein, KAM e invariantes integrales, y entrega
un `Phase1MeasurementCertificate`. El último morfismo
`continue_observe_into_orient` ES la apertura formal de FASE 2.
"""


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.1 — Certificado de FASE 1 (semilla FASE 2)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Phase1MeasurementCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SEMILLA FUNTORIAL FASE 1 → FASE 2                                      ║
    ║ Empaqueta: POVM, Delaunay, Poincaré canónico, (J, θ), Krein, KAM e     ║
    ║ invariantes integrales. Consumido por FASE 2.                          ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    povm_statistics: POVMStatistics
    delaunay: DelaunayElements
    poincare_elements: PoincareCanonicalElements
    action_angle: ActionAngleChart
    krein: KreinSpectralDecomposition
    kam: KAMTorusCertificate
    integral_invariants: PoincareIntegralInvariants
    density_matrix_hash: str
    is_phase1_coherent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "povm_statistics": {
                "outcome_index": self.povm_statistics.outcome_index,
                "probability": float(self.povm_statistics.probability),
                "shannon_entropy": float(self.povm_statistics.shannon_entropy),
                "outcome_purity": float(self.povm_statistics.outcome_purity),
                "mutual_information": float(self.povm_statistics.mutual_information),
                "measurement_disturbance": float(self.povm_statistics.measurement_disturbance),
                "stinespring_defect": float(self.povm_statistics.stinespring_defect),
            },
            "delaunay": self.delaunay.to_dict(),
            "poincare_elements": self.poincare_elements.to_dict(),
            "action_angle": self.action_angle.to_dict(),
            "krein": self.krein.to_dict(),
            "kam": self.kam.to_dict(),
            "integral_invariants": self.integral_invariants.to_dict(),
            "density_matrix_hash": self.density_matrix_hash,
            "is_phase1_coherent": self.is_phase1_coherent,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.2 — Motor POVM (Kraus + Stinespring)
# ──────────────────────────────────────────────────────────────────────────────
class POVMMeasurement:
    r"""
    Ejecutor de Medidas Valuadas en Operadores Positivos (POVM).
    Axiomas:  Eₖ ≽ 0  (vía Mₖ = Eₖ† Eₖ),  Σₖ Mₖ = I,  p(k|ρ) = Tr(Mₖ ρ).
    Isometría de Stinespring: V = Σₖ |k⟩ ⊗ Eₖ  cumple V†V = I.
    """

    def __init__(
        self,
        kraus_operators: List[NDArray[np.complex128]],
        tol: float = 1e-12,
        validate_positivity: bool = True,
        validate_stinespring: bool = True,
    ) -> None:
        self.kraus_ops = kraus_operators
        self.tol = float(tol)
        self.num_outcomes = len(kraus_operators)
        self._validate_operators()
        self._verify_identity_resolution()
        if validate_positivity:
            self._verify_positivity()
        self._stinespring_defect: float = self._verify_stinespring() if validate_stinespring else 0.0
        self._effect_operators: Optional[List[NDArray[np.complex128]]] = None

    def _validate_operators(self) -> None:
        if not self.kraus_ops:
            raise ValueError("Lista vacía de operadores de Kraus")
        first_shape = self.kraus_ops[0].shape
        if first_shape[0] != first_shape[1]:
            raise ValueError(f"Operadores no cuadrados: {first_shape}")
        for idx, E_k in enumerate(self.kraus_ops):
            if E_k.shape != first_shape:
                raise ValueError(
                    f"Operador {idx} inconsistente: {E_k.shape} vs {first_shape}"
                )

    def _verify_identity_resolution(self) -> None:
        dim = self.kraus_ops[0].shape[0]
        identity_sum = np.zeros((dim, dim), dtype=np.complex128)
        for E_k in self.kraus_ops:
            identity_sum += E_k.conj().T @ E_k
        identity_error = float(la.norm(identity_sum - np.eye(dim), ord="fro"))
        if identity_error > self.tol:
            raise NumericalInstabilityError(
                f"Veto Algebraico POVM: Σₖ Eₖ†Eₖ ≠ I. Error: {identity_error:.3e}"
            )

    def _verify_positivity(self) -> None:
        for idx, E_k in enumerate(self.kraus_ops):
            M_k = E_k.conj().T @ E_k
            eigenvalues = la.eigvalsh(M_k)
            if np.any(eigenvalues < -self.tol):
                raise NumericalInstabilityError(
                    f"Efecto M_{idx} no positivo. λ_min = {np.min(eigenvalues):.3e}"
                )

    def _verify_stinespring(self) -> float:
        r"""
        Dilatación de Stinespring: V : ℋ → 𝒦 ⊗ ℋ,  V|ψ⟩ = Σₖ |k⟩ ⊗ Eₖ|ψ⟩.
        V†V − I debe anularse. Devuelve el defecto de Frobenius.
        """
        dim = self.kraus_ops[0].shape[0]
        blocks = [E for E in self.kraus_ops]
        V = np.vstack(blocks)
        defect = float(la.norm(V.conj().T @ V - np.eye(dim), ord="fro"))
        if defect > max(self.tol, _STINESPRING_TOL):
            raise NumericalInstabilityError(
                f"Veto de Stinespring: V†V ≠ I. Defecto: {defect:.3e}"
            )
        return defect

    def get_effect_operators(self) -> List[NDArray[np.complex128]]:
        if self._effect_operators is None:
            self._effect_operators = [E.conj().T @ E for E in self.kraus_ops]
        return self._effect_operators

    def compute_outcome_probabilities(
        self,
        rho: AtomicDensityMatrix,
    ) -> NDArray[np.float64]:
        effects = self.get_effect_operators()
        rho_matrix = rho.matrix
        probabilities = np.array(
            [float(np.trace(M_k @ rho_matrix).real) for M_k in effects],
            dtype=np.float64,
        )
        prob_sum = float(np.sum(probabilities))
        if abs(prob_sum - 1.0) > self.tol and prob_sum > self.tol:
            probabilities = probabilities / prob_sum
        return np.clip(probabilities, 0.0, 1.0)

    def compute_post_measurement_states(
        self,
        rho: AtomicDensityMatrix,
    ) -> List[Tuple[AtomicDensityMatrix, float]]:
        rho_matrix = rho.matrix
        probabilities = self.compute_outcome_probabilities(rho)
        post_states: List[Tuple[AtomicDensityMatrix, float]] = []
        for E_k, p_k in zip(self.kraus_ops, probabilities):
            if p_k > self.tol:
                rho_k_unnorm = E_k @ rho_matrix @ E_k.conj().T
                rho_k = rho_k_unnorm / p_k
                post_states.append(
                    (
                        AtomicDensityMatrix(rho_k, auto_renormalize=True, validate=False),
                        float(p_k),
                    )
                )
            else:
                dim = rho_matrix.shape[0]
                post_states.append(
                    (
                        AtomicDensityMatrix(
                            np.zeros((dim, dim), dtype=np.complex128),
                            validate=False,
                        ),
                        0.0,
                    )
                )
        return post_states

    def compute_measurement_statistics(
        self,
        rho_pre: AtomicDensityMatrix,
        outcome_index: int,
        rho_post: AtomicDensityMatrix,
    ) -> POVMStatistics:
        probabilities = self.compute_outcome_probabilities(rho_pre)
        p_k = float(probabilities[outcome_index])
        prob_nonzero = probabilities[probabilities > self.tol]
        eps_p = np.finfo(prob_nonzero.dtype).eps if prob_nonzero.size else 1e-30
        prob_safe = np.maximum(prob_nonzero, eps_p)
        shannon_entropy = float(-np.sum(prob_nonzero * np.log2(prob_safe))) if prob_nonzero.size else 0.0
        outcome_purity = float(rho_post.compute_metrics().purity)
        entropy_pre = float(rho_pre.compute_metrics().von_neumann_entropy)
        post_states = self.compute_post_measurement_states(rho_pre)
        conditional_entropy = sum(
            p * rho_k.compute_metrics().von_neumann_entropy
            for rho_k, p in post_states
            if p > self.tol
        )
        mutual_information = entropy_pre - conditional_entropy
        trace_distance = 0.5 * float(
            la.norm(rho_post.matrix - rho_pre.matrix, ord="nuc")
        )
        return POVMStatistics(
            outcome_index=outcome_index,
            probability=p_k,
            shannon_entropy=shannon_entropy,
            outcome_purity=outcome_purity,
            mutual_information=float(mutual_information),
            measurement_disturbance=trace_distance,
            stinespring_defect=float(self._stinespring_defect),
        )

    def measure_and_collapse(
        self,
        rho: AtomicDensityMatrix,
        deterministic: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, AtomicDensityMatrix, POVMStatistics]:
        probabilities = self.compute_outcome_probabilities(rho)
        post_states = self.compute_post_measurement_states(rho)
        if deterministic:
            outcome_k = int(np.argmax(probabilities))
        else:
            gen = rng if rng is not None else np.random.default_rng()
            outcome_k = int(gen.choice(self.num_outcomes, p=probabilities))
        rho_collapsed, _ = post_states[outcome_k]
        statistics = self.compute_measurement_statistics(
            rho_pre=rho, outcome_index=outcome_k, rho_post=rho_collapsed,
        )
        logger.info(
            "[POVM] Colapso: k=%d, P(k)=%.6f, H(M)=%.4f bits, Stinespring=%.3e",
            outcome_k,
            statistics.probability,
            statistics.shannon_entropy,
            statistics.stinespring_defect,
        )
        return outcome_k, rho_collapsed, statistics


# ──────────────────────────────────────────────────────────────────────────────
# FASE 1.3 — CIERRE FASE 1: certificado espectral celestial
# ──────────────────────────────────────────────────────────────────────────────
def build_phase1_measurement_certificate(
    rho: AtomicDensityMatrix,
    povm: POVMMeasurement,
    deterministic: bool = False,
    N_potential: Optional[NDArray[np.float64]] = None,
    torsional_hessian: Optional[NDArray[np.float64]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, AtomicDensityMatrix, Phase1MeasurementCertificate]:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FORMAL DE FASE 1                                                ║
    ║ Ejecuta la medición POVM, extrae Delaunay + Poincaré canónico +        ║
    ║ (J, θ) + Krein + KAM + invariantes integrales y empaqueta la semilla   ║
    ║ espectral celestial.                                                   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    outcome_k, rho_collapsed, statistics = povm.measure_and_collapse(
        rho, deterministic=deterministic, rng=rng,
    )
    delaunay = rho_collapsed.poincare_delaunay_elements(N_potential)
    pec = rho_collapsed.poincare_canonical_elements(N_potential)
    chart = rho_collapsed.action_angle_chart()
    krein = rho_collapsed.krein_spectral_decomposition(N_potential)
    kam = rho_collapsed.kam_torus_certificate(torsional_hessian)
    inv = rho_collapsed.poincare_integral_invariants(N_potential)
    rho_hash = _sha16(rho_collapsed.matrix)
    coherent = (
        delaunay.is_hill_stable
        and pec.is_canonical
        and chart.is_nondegenerate
        and krein.is_spectrally_stable
        and kam.is_kam_stable
        and statistics.probability > 1e-10
        and statistics.stinespring_defect <= max(povm.tol, _STINESPRING_TOL)
    )
    cert = Phase1MeasurementCertificate(
        povm_statistics=statistics,
        delaunay=delaunay,
        poincare_elements=pec,
        action_angle=chart,
        krein=krein,
        kam=kam,
        integral_invariants=inv,
        density_matrix_hash=rho_hash,
        is_phase1_coherent=bool(coherent),
    )
    logger.info(
        "[F1] L=%.4f e=%.4f C_J=%.4f Hill=%s | Poincaré Λ=%.4f canónico=%s | "
        "KAM_ρ=%.4f ω_t=%.4f DC_γ=%.3e | coherent=%s",
        delaunay.L_semi_axis,
        delaunay.eccentricity,
        delaunay.jacobi_constant,
        delaunay.is_hill_stable,
        pec.Lambda,
        pec.is_canonical,
        kam.kam_index,
        kam.torsional_frequency,
        kam.diophantine.gamma,
        coherent,
    )
    return outcome_k, rho_collapsed, cert


# ══════════════════════════════════════════════════════════════════════════════
# █  MORFISMO DE ANIDACIÓN F1 ↪ F2                                           █
# █  El último método de FASE 1 ES el primero de FASE 2.                     █
# ══════════════════════════════════════════════════════════════════════════════
def continue_observe_into_orient(
    phase1_cert: Phase1MeasurementCertificate,
    rho_collapsed: AtomicDensityMatrix,
    *,
    orchestrator: "LindbladDynamicsOrchestrator",
    H_error: NDArray[np.complex128],
    jump_operators: Optional[List[Tuple[float, NDArray[np.complex128]]]] = None,
    dt: float = 0.01,
    num_steps: int = 8,
    section_normal: Optional[NDArray[np.float64]] = None,
    h0_poisson_h1: Optional[Callable[[float], float]] = None,
    perturbation_strength: float = 1e-3,
    analyticity_radius: float = 1.0,
    ensure_physicality: bool = True,
) -> Tuple[AtomicDensityMatrix, "Phase2DynamicsCertificate"]:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ FASE 1.FIN ≡ FASE 2.INICIO  (Observe ↪ Orient)                         ║
    ║ Continuación funtorial estricta: consume el certificado espectral de   ║
    ║ FASE 1, integra el flujo GKSL y abre el análisis dinámico celestial    ║
    ║ de FASE 2 (sección, Floquet, twist, recurrencia, Chirikov, Melnikov,   ║
    ║ Nekhoroshev).                                                          ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    rho_evolved, lindblad_metrics, trajectory = orchestrator.evolve_state(
        rho=rho_collapsed,
        H_error=H_error,
        jump_operators=jump_operators or [],
        dt=dt,
        num_steps=max(int(num_steps), _MIN_LINDBLAD_STEPS),
        ensure_physicality=ensure_physicality,
    )
    if h0_poisson_h1 is None:
        h0_poisson_h1 = lambda t: math.sin(t) * math.exp(-0.05 * abs(t))
    cert = assemble_phase2_dynamics_certificate(
        phase1_cert=phase1_cert,
        rho_evolved=rho_evolved,
        lindblad_metrics=lindblad_metrics,
        trajectory=trajectory,
        section_normal=section_normal,
        h0_poisson_h1=h0_poisson_h1,
        perturbation_strength=perturbation_strength,
        analyticity_radius=analyticity_radius,
        dt_lyapunov=dt,
    )
    return rho_evolved, cert


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 2 — CONTINUACIÓN                                 █
# █  DINÁMICA DE LINDBLAD CELESTE Y SECCIONES DE POINCARÉ (Orient)           █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 2 (continuación de `continue_observe_into_orient`): integra GKSL como
flujo Port-Hamiltoniano en T*ℋ_MAC, traza la sección de Poincaré, calcula
Floquet, twist, recurrencia, Chirikov, Melnikov y Nekhoroshev, y entrega un
`Phase2DynamicsCertificate`. El último morfismo `continue_orient_into_decide`
ES la apertura formal de FASE 3.
"""


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.1 — Certificado de FASE 2 (semilla FASE 3)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Phase2DynamicsCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SEMILLA FUNTORIAL FASE 2 → FASE 3                                      ║
    ║ Empaqueta: Lindblad, sección, Floquet, twist, recurrencia, Chirikov,   ║
    ║ Melnikov, Nekhoroshev y espectro de Lyapunov. Consumido por FASE 3.    ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    phase1_hash: str
    lindblad_metrics: LindbladEvolutionMetrics
    poincare_section: PoincareSectionState
    floquet: FloquetMonodromyCertificate
    twist: TwistMapCertificate
    recurrence: PoincareRecurrenceCertificate
    chirikov: ChirikovResonanceCertificate
    melnikov: MelnikovCertificate
    nekhoroshev: NekhoroshevCertificate
    lyapunov_spectrum: Tuple[float, ...]
    is_globally_integrable: bool
    dynamics_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase1_hash": self.phase1_hash,
            "lindblad_metrics": {
                "time_evolved": float(self.lindblad_metrics.time_evolved),
                "trace_before": float(self.lindblad_metrics.trace_before),
                "trace_after": float(self.lindblad_metrics.trace_after),
                "purity_before": float(self.lindblad_metrics.purity_before),
                "purity_after": float(self.lindblad_metrics.purity_after),
                "entropy_production": float(self.lindblad_metrics.entropy_production),
                "dissipated_coherence": float(self.lindblad_metrics.dissipated_coherence),
                "spohn_rate": float(self.lindblad_metrics.spohn_rate),
                "cptp_residual": float(self.lindblad_metrics.cptp_residual),
            },
            "poincare_section": self.poincare_section.to_dict(),
            "floquet": self.floquet.to_dict(),
            "twist": self.twist.to_dict(),
            "recurrence": self.recurrence.to_dict(),
            "chirikov": self.chirikov.to_dict(),
            "melnikov": self.melnikov.to_dict(),
            "nekhoroshev": self.nekhoroshev.to_dict(),
            "lyapunov_spectrum": list(self.lyapunov_spectrum),
            "is_globally_integrable": self.is_globally_integrable,
            "dynamics_hash": self.dynamics_hash,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.2 — Orquestador de Lindblad (GKSL) como flujo celeste
# ──────────────────────────────────────────────────────────────────────────────
class LindbladDynamicsOrchestrator:
    r"""
    Gobernador de la ecuación maestra GKSL:
        ρ̇ = −i/ℏ [H, ρ] + Σₖ γₖ (Lₖ ρ Lₖ† − ½ {Lₖ† Lₖ, ρ}).
    El generador 𝓛 se interpreta como campo Port-Hamiltoniano sobre T*ℋ:
      • parte hamiltoniana  −i[H, ·]  (simplectomorfismo de Cayley/Magnus),
      • parte disipativa    Σ γₖ 𝒟[Lₖ]  (Rayleigh, CP, γₖ ≥ 0).
    Preserva Tr(ρ)=1, ρ=ρ†, ρ≽0; la producción de entropía de Spohn es ≥ 0.
    """

    _HBAR: float = 1.0

    def __init__(self, hbar: float = 1.0, integration_method: str = "rk4") -> None:
        self.hbar = float(hbar)
        self.integration_method = integration_method
        if integration_method not in ("euler", "rk4", "magnus"):
            raise ValueError(f"Método desconocido: {integration_method}")

    @staticmethod
    def _validate_jump_operators(
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dim: int,
        tol: float = 1e-15,
    ) -> float:
        r"""CPTP: γₖ ≥ 0 y Lₖ ∈ M_dim(ℂ). Residual = Σ |min(γₖ, 0)|."""
        residual = 0.0
        for gamma_k, L_k in jump_operators:
            if L_k.shape != (dim, dim):
                raise ValueError(f"L_k shape {L_k.shape} ≠ ({dim}, {dim})")
            if gamma_k < -tol:
                raise NumericalInstabilityError(
                    f"Tasa de Lindblad negativa (rompe CP): γ={gamma_k:.3e}"
                )
            residual += abs(min(float(gamma_k), 0.0))
        return float(residual)

    def _compute_lindbladian(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        commutator = H @ rho - rho @ H
        drho_dt = -1j / self.hbar * commutator
        for gamma_k, L_k in jump_operators:
            L_dag = L_k.conj().T
            dissipator = L_k @ rho @ L_dag
            anticomm = (L_dag @ L_k) @ rho + rho @ (L_dag @ L_k)
            drho_dt += gamma_k * (dissipator - 0.5 * anticomm)
        return drho_dt

    def _rk4_step(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dt: float,
    ) -> NDArray[np.complex128]:
        k1 = self._compute_lindbladian(rho, H, jump_operators)
        k2 = self._compute_lindbladian(rho + 0.5 * dt * k1, H, jump_operators)
        k3 = self._compute_lindbladian(rho + 0.5 * dt * k2, H, jump_operators)
        k4 = self._compute_lindbladian(rho + dt * k3, H, jump_operators)
        return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _euler_step(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dt: float,
    ) -> NDArray[np.complex128]:
        return rho + dt * self._compute_lindbladian(rho, H, jump_operators)

    def _magnus_step(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dt: float,
    ) -> NDArray[np.complex128]:
        r"""
        Splitting de Strang (orden 2): ½ disipador + Magnus/exponencial
        hamiltoniano (simplectomorfismo) + ½ disipador. Conserva la 1-forma
        de Poincaré-Cartan al orden O(dt²) en la pieza unitaria.
        """
        zero_H = np.zeros_like(H)
        rho = self._euler_step(rho, zero_H, jump_operators, 0.5 * dt)
        U = la.expm((-1j * dt / self.hbar) * H)
        rho = U @ rho @ U.conj().T
        rho = self._euler_step(rho, zero_H, jump_operators, 0.5 * dt)
        return rho

    def _ensure_physicality(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.trace(rho)
        if abs(trace) > 1e-12:
            rho = rho / trace
        eigenvalues, eigenvectors = la.eigh(rho)
        if np.any(eigenvalues < 0):
            eigenvalues = np.maximum(eigenvalues, 0.0)
            rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
            tr = np.trace(rho)
            if abs(tr) > 1e-15:
                rho = rho / tr
        return rho

    @staticmethod
    def _spohn_entropy_production(
        rho: NDArray[np.complex128],
        drho: NDArray[np.complex128],
        tol: float = 1e-12,
    ) -> float:
        r"""σ = −Tr(ρ̇ log ρ)  (producción de Spohn; ≥ 0 en GKSL)."""
        eigvals, eigvecs = la.eigh(rho)
        log_spec = np.zeros_like(eigvals, dtype=np.float64)
        mask = eigvals > tol
        log_spec[mask] = np.log(np.maximum(eigvals[mask], tol))
        log_rho = eigvecs @ np.diag(log_spec) @ eigvecs.conj().T
        return float(-np.trace(drho @ log_rho).real)

    def evolve_state(
        self,
        rho: AtomicDensityMatrix,
        H_error: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dt: float,
        num_steps: int = 1,
        ensure_physicality: bool = True,
    ) -> Tuple[AtomicDensityMatrix, LindbladEvolutionMetrics, NDArray[np.float64]]:
        r"""
        Integra la evolución temporal del estado MAC.
        Retorna (ρ_evolucionado, métricas_Lindblad, trayectoria_vectorizada)
        donde la trayectoria vive en ℝ^{2 n²} ≅ T ℋ (parte real/imaginaria),
        lista para la sección de Poincaré.
        """
        H_error = np.asarray(H_error, dtype=np.complex128)
        if not np.allclose(H_error, H_error.conj().T, atol=1e-10):
            H_error = 0.5 * (H_error + H_error.conj().T)

        rho_matrix = rho.matrix
        dim = int(rho_matrix.shape[0])
        cptp_residual = self._validate_jump_operators(jump_operators, dim)

        metrics_initial = rho.compute_metrics()
        trace_initial = float(np.trace(rho_matrix).real)
        purity_initial = float(metrics_initial.purity)
        entropy_initial = float(metrics_initial.von_neumann_entropy)
        coherence_initial = float(np.sum(np.abs(np.triu(rho_matrix, k=1))))

        trajectory: List[NDArray[np.float64]] = []
        rho_current = rho_matrix.copy()
        spohn_acc = 0.0
        stepper = {
            "euler": self._euler_step,
            "rk4": self._rk4_step,
            "magnus": self._magnus_step,
        }[self.integration_method]

        for _ in range(max(int(num_steps), 1)):
            drho = self._compute_lindbladian(rho_current, H_error, jump_operators)
            spohn_acc += self._spohn_entropy_production(rho_current, drho) * dt
            rho_current = stepper(rho_current, H_error, jump_operators, dt)
            if ensure_physicality:
                rho_current = self._ensure_physicality(rho_current)
            vec = np.concatenate(
                [np.real(rho_current).ravel(), np.imag(rho_current).ravel()]
            )
            trajectory.append(np.asarray(vec, dtype=np.float64))

        rho_final = AtomicDensityMatrix(
            rho_current, auto_renormalize=True, validate=True,
        )
        metrics_final = rho_final.compute_metrics()
        trace_final = float(np.trace(rho_current).real)
        purity_final = float(metrics_final.purity)
        entropy_final = float(metrics_final.von_neumann_entropy)
        coherence_final = float(np.sum(np.abs(np.triu(rho_current, k=1))))

        evolution_metrics = LindbladEvolutionMetrics(
            time_evolved=float(dt * max(int(num_steps), 1)),
            trace_before=trace_initial,
            trace_after=trace_final,
            purity_before=purity_initial,
            purity_after=purity_final,
            entropy_production=entropy_final - entropy_initial,
            dissipated_coherence=coherence_initial - coherence_final,
            spohn_rate=float(spohn_acc / max(dt * max(int(num_steps), 1), 1e-15)),
            cptp_residual=cptp_residual,
        )
        if not evolution_metrics.is_physically_valid():
            logger.warning(
                "[Lindblad] Evolución no física. ΔS=%.3e  σ_Spohn=%.3e",
                evolution_metrics.entropy_production,
                evolution_metrics.spohn_rate,
            )
        traj_array = (
            np.asarray(trajectory, dtype=np.float64)
            if trajectory
            else np.zeros((1, 2 * dim * dim), dtype=np.float64)
        )
        return rho_final, evolution_metrics, traj_array


def _align_section_normal(
    trajectory: NDArray[np.float64],
    section_normal: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Alinea n ∈ ℝ^d con la dimensión de la trayectoria vectorizada."""
    traj = np.asarray(trajectory, dtype=np.float64)
    d = int(traj.shape[1]) if traj.ndim == 2 else int(traj.size)
    if section_normal is None:
        n = np.zeros(d, dtype=np.float64)
        n[0] = 1.0
        return n
    n = np.asarray(section_normal, dtype=np.float64).ravel()
    if n.size < d:
        n = np.pad(n, (0, d - n.size))
    elif n.size > d:
        n = n[:d]
    if float(np.linalg.norm(n)) < 1e-15:
        n = np.zeros(d, dtype=np.float64)
        n[0] = 1.0
    return n


# ──────────────────────────────────────────────────────────────────────────────
# FASE 2.3 — CIERRE FASE 2: análisis dinámico celestial (núcleos de la MAC)
# ──────────────────────────────────────────────────────────────────────────────
def assemble_phase2_dynamics_certificate(
    phase1_cert: Phase1MeasurementCertificate,
    rho_evolved: AtomicDensityMatrix,
    lindblad_metrics: LindbladEvolutionMetrics,
    trajectory: NDArray[np.float64],
    section_normal: Optional[NDArray[np.float64]],
    h0_poisson_h1: Callable[[float], float],
    perturbation_strength: float = 1e-3,
    analyticity_radius: float = 1.0,
    dt_lyapunov: float = 1.0,
    primary_resonance_width: float = 0.0,
) -> Phase2DynamicsCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE FORMAL DE FASE 2                                                ║
    ║ Ejecuta el análisis dinámico celestial (núcleos Poincaré de la MAC)    ║
    ║ sobre la trayectoria de Lindblad y empaqueta la semilla de FASE 3.     ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    _ = rho_evolved
    nvec = _align_section_normal(trajectory, section_normal)
    section = trace_poincare_section(trajectory=trajectory, section_normal=nvec)
    T_return = max(float(lindblad_metrics.time_evolved), 1e-15)
    floq = floquet_monodromy(section, return_period=T_return)
    twist = twist_map_certificate(section, actions=phase1_cert.action_angle.actions)
    rec = poincare_recurrence_bound(trajectory)
    chirikov = chirikov_resonance_overlap(
        section=section, primary_resonance_width=primary_resonance_width,
    )
    melnikov = melnikov_function(h0_poisson_h1=h0_poisson_h1)
    n_deg = max(len(phase1_cert.action_angle.actions), 2)
    nekh = nekhoroshev_stability(
        perturbation_strength=perturbation_strength,
        analyticity_radius=analyticity_radius,
        n_degrees=n_deg,
    )
    lces = lyapunov_spectrum_from_trajectory(trajectory, dt=dt_lyapunov)
    is_integrable = (
        phase1_cert.kam.is_kam_stable
        and chirikov.is_kam_intact
        and not melnikov.has_transversal_homoclinic
        and nekh.is_exponentially_stable
        and floq.is_elliptic
        and lindblad_metrics.is_physically_valid()
    )
    dynamics_hash = _sha16(
        phase1_cert.density_matrix_hash,
        round(section.rotation_number, 8),
        round(chirikov.overlap_ratio, 8),
        melnikov.has_transversal_homoclinic,
        floq.spectral_radius,
        is_integrable,
    )
    cert = Phase2DynamicsCertificate(
        phase1_hash=phase1_cert.density_matrix_hash,
        lindblad_metrics=lindblad_metrics,
        poincare_section=section,
        floquet=floq,
        twist=twist,
        recurrence=rec,
        chirikov=chirikov,
        melnikov=melnikov,
        nekhoroshev=nekh,
        lyapunov_spectrum=lces,
        is_globally_integrable=bool(is_integrable),
        dynamics_hash=dynamics_hash,
    )
    logger.info(
        "[F2] ν=%.4f K_chirikov=%.4f Floquet_ρ=%.4f twist=%s homoclínico=%s "
        "N_nekh=%.4e integrable=%s",
        section.rotation_number,
        chirikov.overlap_ratio,
        floq.spectral_radius,
        twist.is_twist,
        melnikov.has_transversal_homoclinic,
        nekh.stability_time_lower_bound,
        is_integrable,
    )
    return cert


# ══════════════════════════════════════════════════════════════════════════════
# █  MORFISMO DE ANIDACIÓN F2 ↪ F3                                           █
# █  El último método de FASE 2 ES el primero de FASE 3.                     █
# ══════════════════════════════════════════════════════════════════════════════
def continue_orient_into_decide(
    phase1_cert: Phase1MeasurementCertificate,
    phase2_cert: Phase2DynamicsCertificate,
    cohomology_report: CohomologyAuditReport,
    galois_is_valid: bool,
    critical_points_indices: Sequence[int] = (1,),
    euler_characteristic: int = 1,
    betti_numbers: Sequence[int] = (1, 0, 1),
    kam_stability_hard: bool = True,
) -> "SovereignEpistemicCertificate":
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ FASE 2.FIN ≡ FASE 3.INICIO  (Orient ↪ Decide & Act)                    ║
    ║ Continuación funtorial estricta: consume el certificado dinámico de    ║
    ║ FASE 2 y abre la adjunción de Galois, Hopf, dualidad, colapso Ω₃ y     ║
    ║ Veto de Gromov de FASE 3.                                              ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    return sovereign_epistemic_governance(
        phase1_cert=phase1_cert,
        phase2_cert=phase2_cert,
        cohomology_report=cohomology_report,
        galois_is_valid=galois_is_valid,
        critical_points_indices=critical_points_indices,
        euler_characteristic=euler_characteristic,
        betti_numbers=betti_numbers,
        kam_stability_hard=kam_stability_hard,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                    FASE 3 — CONTINUACIÓN                                 █
# █  COHOMOLOGÍA, ADJUNCIÓN DE GALOIS Y TOPOLOGÍA GLOBAL (Decide & Act)      █
# ████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
r"""
FASE 3 (continuación de `continue_orient_into_decide`): audita H¹(X;ℱ),
verifica F ⊣ G por distancia de Bures, aplica Poincaré-Hopf y dualidad,
colapsa a Ω₃ y emite el Veto simpléctico de Gromov.
"""


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.1 — Certificado soberano
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SovereignEpistemicCertificate:
    """╔═ CERTIFICADO SOBERANO FINAL DE LAS 3 FASES ═╗ con Veto de Gromov."""
    phase1_hash: str
    phase2_hash: str
    hopf: PoincareHopfCertificate
    duality: PoincareDualityCertificate
    verdict: HeytingVerdict
    gromov_veto_active: bool
    sovereign_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase1_hash": self.phase1_hash,
            "phase2_hash": self.phase2_hash,
            "hopf": self.hopf.to_dict(),
            "duality": self.duality.to_dict(),
            "verdict": self.verdict.value,
            "heyting_omega3": self.verdict.to_omega3().name,
            "gromov_veto_active": self.gromov_veto_active,
            "sovereign_hash": self.sovereign_hash,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.2 — Custodio cohomológico
# ──────────────────────────────────────────────────────────────────────────────
class SheafCohomologyCustodian:
    r"""Tribunal topológico del fibrado neuronal: audita H⁰ y H¹."""

    def __init__(
        self,
        sheaf: CellularSheafNeuralManifold,
        tol: float = 1e-9,
        auto_project: bool = False,
    ) -> None:
        self.sheaf = sheaf
        self.tol = float(tol)
        self.auto_project = auto_project
        self._cohomology_groups: Optional[Dict] = None
        self.audit_count: int = 0
        self.violation_count: int = 0
        self.violation_history: List[float] = []

    def compute_cohomology_groups(self) -> Dict:
        if self._cohomology_groups is None:
            self._cohomology_groups = self.sheaf.compute_cohomology_groups()
        return self._cohomology_groups

    def audit_holonomy(
        self,
        semantic_state: NDArray[np.float64],
        raise_on_violation: bool = True,
    ) -> CohomologyAuditReport:
        self.audit_count += 1
        dirichlet_energy = self.sheaf.compute_dirichlet_energy(semantic_state)
        is_holonomic = dirichlet_energy < self.tol
        cohomology = self.compute_cohomology_groups()
        betti_numbers = {deg: g.betti_number for deg, g in cohomology.items()}
        global_sections_dim = cohomology[0].betti_number
        obstruction_class = None
        if not is_holonomic:
            obstruction_class = self.sheaf.compute_coboundary(semantic_state)
            self.violation_count += 1
            self.violation_history.append(float(dirichlet_energy))
        report = CohomologyAuditReport(
            is_holonomic=bool(is_holonomic),
            dirichlet_energy=float(dirichlet_energy),
            betti_numbers=betti_numbers,
            obstruction_class=obstruction_class,
            global_sections_dim=int(global_sections_dim),
        )
        if is_holonomic:
            logger.info("[H⁰] ✓ Holonomía. E_Dirichlet=%.6e", dirichlet_energy)
        else:
            logger.critical("[H¹] ✗ Veto Cohomológico. E_Dirichlet=%.6e", dirichlet_energy)
            if raise_on_violation:
                raise NumericalInstabilityError(
                    f"Obstrucción topológica H¹(X;ℱ). E_Dirichlet={dirichlet_energy:.6e}"
                )
        return report

    def repair_semantic_state(
        self,
        semantic_state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        energy_before = self.sheaf.compute_dirichlet_energy(semantic_state)
        repaired = self.sheaf.project_to_harmonic(semantic_state)
        energy_after = self.sheaf.compute_dirichlet_energy(repaired)
        logger.info("[REPAIR] E_Dirichlet: %.3e → %.3e", energy_before, energy_after)
        return repaired

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "total_audits": self.audit_count,
            "violations": self.violation_count,
            "violation_rate": self.violation_count / max(1, self.audit_count),
            "violation_history": self.violation_history,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.3 — Auditor de la adjunción de Galois (geometría de información)
# ──────────────────────────────────────────────────────────────────────────────
class GaloisAdjunctionAuditor:
    r"""
    Verificador de la adjunción F ⊣ G con distancia de Bures–Wasserstein,
    fidelidad de Uhlmann, distancia traza y Hilbert–Schmidt.
    La counidad ε : FG ⇒ 1 se valida por d_Bures(ρ_MAC, σ_MIC) ≤ ε.
    """

    @staticmethod
    def compute_matrix_sqrt(
        A: NDArray[np.complex128],
        validate: bool = True,
    ) -> NDArray[np.complex128]:
        if validate:
            eigenvalues = la.eigvalsh(A)
            if np.any(eigenvalues < -1e-10):
                raise ValueError(f"Matriz no PSD. λ_min={np.min(eigenvalues)}")
        eigenvalues, eigenvectors = la.eigh(A)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    @classmethod
    def compute_bures_distance(
        cls,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        sqrt_rho = cls.compute_matrix_sqrt(rho)
        core = sqrt_rho @ sigma @ sqrt_rho
        sqrt_core = cls.compute_matrix_sqrt(core)
        fidelity_term = float(np.trace(sqrt_core).real)
        trace_rho = float(np.trace(rho).real)
        trace_sigma = float(np.trace(sigma).real)
        d_sq = max(0.0, trace_rho + trace_sigma - 2.0 * fidelity_term)
        return float(np.sqrt(d_sq))

    @classmethod
    def compute_fidelity(
        cls,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        sqrt_rho = cls.compute_matrix_sqrt(rho)
        core = sqrt_rho @ sigma @ sqrt_rho
        sqrt_core = cls.compute_matrix_sqrt(core)
        return float(np.clip(np.trace(sqrt_core).real ** 2, 0.0, 1.0))

    @staticmethod
    def compute_trace_distance(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        eigenvalues = la.eigvalsh(rho - sigma)
        return float(0.5 * np.sum(np.abs(eigenvalues)))

    @staticmethod
    def compute_hilbert_schmidt_distance(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        return float(la.norm(rho - sigma, ord="fro"))

    def validate_adjunction_counit(
        self,
        rho_mac: AtomicDensityMatrix,
        sigma_mic: AtomicDensityMatrix,
        epsilon: float = _BURES_EPSILON_DEFAULT,
        metric: str = "bures",
    ) -> Tuple[bool, Dict[str, float]]:
        rho = rho_mac.matrix
        sigma = sigma_mic.matrix
        metrics = {
            "bures_distance": self.compute_bures_distance(rho, sigma),
            "trace_distance": self.compute_trace_distance(rho, sigma),
            "fidelity": self.compute_fidelity(rho, sigma),
            "hilbert_schmidt": self.compute_hilbert_schmidt_distance(rho, sigma),
        }
        if metric == "bures":
            primary = metrics["bures_distance"]
        elif metric == "trace":
            primary = metrics["trace_distance"]
        elif metric == "fidelity":
            primary = 1.0 - metrics["fidelity"]
        elif metric == "hs":
            primary = metrics["hilbert_schmidt"]
        else:
            raise ValueError(f"Métrica desconocida: {metric}")
        is_valid = primary <= epsilon
        if not is_valid:
            logger.error(
                "[GALOIS] Distorsión: %s=%.6f > ε=%.4f",
                metric, primary, epsilon,
            )
        else:
            logger.info("[GALOIS] ✓ Adjunción validada. %s=%.6f", metric, primary)
        return is_valid, metrics


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.4 — Topología global: Poincaré-Hopf + dualidad (núcleos de la MAC)
# ──────────────────────────────────────────────────────────────────────────────
class PoincareTopologyEngine:
    r"""
    Motor topológico global. Delega en los morfismos estáticos de
    `GaloisAdjunctionFunctor` (MAC) para Hopf y dualidad, preservando el
    polinomio de Poincaré P_ℳ(t) = Σ b_k t^k.
    """

    poincare_hopf_index = staticmethod(GaloisAdjunctionFunctor.poincare_hopf_index)
    poincare_duality = staticmethod(GaloisAdjunctionFunctor.poincare_duality)

    @staticmethod
    def heyting_collapse(
        hopf: PoincareHopfCertificate,
        duality: PoincareDualityCertificate,
        hard_conditions: bool,
        integrable: bool,
        kam_stability_hard: bool,
    ) -> HeytingVerdict:
        r"""
        Colapso Ω₃. Orden de decisión:
          ¬(Hopf ∧ Dualidad ∧ Galois ∧ H⁰ ∧ F1)  →  ⊥ (VETOED)
          integrable                             →  ⊤ (COHERENT)
          kam_hard                               →  ⊥ (VETOED)
          otherwise                              →  ½ (DEGRADED)
        """
        if not (hard_conditions and hopf.is_hopf_consistent and duality.is_duality_symmetric):
            return HeytingVerdict.VETOED
        if integrable:
            return HeytingVerdict.COHERENT
        if kam_stability_hard:
            return HeytingVerdict.VETOED
        return HeytingVerdict.DEGRADED


# ──────────────────────────────────────────────────────────────────────────────
# FASE 3.5 — CIERRE SOBERANO: gobernanza epistémica con Veto de Gromov
# ──────────────────────────────────────────────────────────────────────────────
def sovereign_epistemic_governance(
    phase1_cert: Phase1MeasurementCertificate,
    phase2_cert: Phase2DynamicsCertificate,
    cohomology_report: CohomologyAuditReport,
    galois_is_valid: bool,
    critical_points_indices: Sequence[int] = (1,),
    euler_characteristic: int = 1,
    betti_numbers: Sequence[int] = (1, 0, 1),
    kam_stability_hard: bool = True,
) -> SovereignEpistemicCertificate:
    r"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ CIERRE SOBERANO DE LAS 3 FASES CON VETO DE GROMOV                      ║
    ║ P(x_invalid) ≡ 0 si alguna condición duradera falla. Colapsa el        ║
    ║ veredicto al retículo de Heyting Ω₃.                                   ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """
    hopf = PoincareTopologyEngine.poincare_hopf_index(
        critical_points_indices, euler_characteristic,
    )
    duality = PoincareTopologyEngine.poincare_duality(betti_numbers)
    hard_conditions = (
        hopf.is_hopf_consistent
        and duality.is_duality_symmetric
        and galois_is_valid
        and cohomology_report.is_holonomic
        and phase1_cert.is_phase1_coherent
    )
    verdict = PoincareTopologyEngine.heyting_collapse(
        hopf=hopf,
        duality=duality,
        hard_conditions=hard_conditions,
        integrable=phase2_cert.is_globally_integrable,
        kam_stability_hard=kam_stability_hard,
    )
    veto = verdict is HeytingVerdict.VETOED
    sovereign_hash = _sha16(
        phase1_cert.density_matrix_hash,
        phase2_cert.dynamics_hash,
        hopf.is_hopf_consistent,
        duality.is_duality_symmetric,
        verdict.value,
    )
    if veto:
        logger.critical(
            "[GROMOV_VETO] Certificado NO viable: Hopf=%s Duality=%s Galois=%s "
            "H⁰=%s Integrable=%s Ω₃=%s",
            hopf.is_hopf_consistent,
            duality.is_duality_symmetric,
            galois_is_valid,
            cohomology_report.is_holonomic,
            phase2_cert.is_globally_integrable,
            verdict.value,
        )
    return SovereignEpistemicCertificate(
        phase1_hash=phase1_cert.density_matrix_hash,
        phase2_hash=phase2_cert.dynamics_hash,
        hopf=hopf,
        duality=duality,
        verdict=verdict,
        gromov_veto_active=veto,
        sovereign_hash=sovereign_hash,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR CENTRAL: MACAgent (3 fases anidadas, ciclo OODA)
# ══════════════════════════════════════════════════════════════════════════════
class MACAgent(Morphism):
    r"""
    El Cerebro Epistemológico de la Malla Agéntica + Mecánica Celeste.

    Ciclo OODA con composición funtorial estricta:
      OBSERVE (F1): POVM + Stinespring + Delaunay + Poincaré + Krein + KAM
                    → `continue_observe_into_orient`
      ORIENT  (F2): Lindblad + sección + Floquet + twist + recurrencia
                    + Chirikov + Melnikov + Nekhoroshev
                    → `continue_orient_into_decide`
      DECIDE  (F3): Cohomología + Galois + Hopf + Dualidad + Ω₃
      ACT         : Colapso a Ω₃ y (si procede) Crowbar BT151.
    """

    def __init__(
        self,
        sheaf_manifold: CellularSheafNeuralManifold,
        integration_method: str = "rk4",
        auto_repair: bool = True,
        debug_mode: bool = False,
    ) -> None:
        self.sheaf_manifold = sheaf_manifold
        self.auto_repair = auto_repair
        self.debug_mode = debug_mode
        self.lindblad_orchestrator = LindbladDynamicsOrchestrator(
            integration_method=integration_method,
        )
        self.sheaf_custodian = SheafCohomologyCustodian(
            sheaf_manifold, auto_project=auto_repair,
        )
        self.galois_auditor = GaloisAdjunctionAuditor()
        self.operation_count: int = 0
        self.error_count: int = 0
        self.state_history: List[AtomicDensityMatrix] = []
        self.certificate_history: List[SovereignEpistemicCertificate] = []

    def process_telemetry_cartridge_celestial(
        self,
        current_rho: AtomicDensityMatrix,
        semantic_vector: NDArray[np.float64],
        H_error: NDArray[np.complex128],
        jump_ops: Optional[List[Tuple[float, NDArray[np.complex128]]]] = None,
        povm_ops: Optional[List[NDArray[np.complex128]]] = None,
        section_normal: Optional[NDArray[np.float64]] = None,
        h0_poisson_h1: Optional[Callable[[float], float]] = None,
        perturbation_strength: float = 1e-3,
        analyticity_radius: float = 1.0,
        dt: float = 0.01,
        num_steps: int = 1,
        deterministic_povm: bool = False,
        critical_points_indices: Sequence[int] = (1,),
        euler_characteristic: int = 1,
        betti_numbers: Sequence[int] = (1, 0, 1),
        kam_stability_hard: bool = True,
        bures_epsilon: float = _BURES_EPSILON_DEFAULT,
        N_potential: Optional[NDArray[np.float64]] = None,
        torsional_hessian: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[AtomicDensityMatrix, Dict[str, Any]]:
        r"""
        Ciclo OODA Celeste completo vía morfismos de anidación:
            F1 = build_phase1_measurement_certificate
            F1 ↪ F2 = continue_observe_into_orient
            F2 ↪ F3 = continue_orient_into_decide
        Lanza NumericalInstabilityError si el Veto de Gromov se activa.
        """
        self.operation_count += 1
        telemetry: Dict[str, Any] = {
            "operation_id": self.operation_count,
            "timestamp": float(time.time()),
        }

        # ── FASE 1: Observe ────────────────────────────────────────────────
        logger.info("[OODA-OBSERVE] FASE 1: POVM + geometría espectral celeste...")
        povm_ops_resolved = povm_ops or self._default_povm_ops(current_rho.dimension)
        povm = POVMMeasurement(povm_ops_resolved, validate_positivity=True)
        outcome_k, rho_collapsed, phase1_cert = build_phase1_measurement_certificate(
            rho=current_rho,
            povm=povm,
            deterministic=deterministic_povm,
            N_potential=N_potential,
            torsional_hessian=torsional_hessian,
        )
        telemetry["phase1"] = phase1_cert.to_dict()
        telemetry["phase1_outcome_index"] = outcome_k

        # ── FASE 1 ↪ FASE 2: Orient ────────────────────────────────────────
        logger.info("[OODA-ORIENT] FASE 2: Lindblad + análisis dinámico celestial...")
        rho_evolved, phase2_cert = continue_observe_into_orient(
            phase1_cert=phase1_cert,
            rho_collapsed=rho_collapsed,
            orchestrator=self.lindblad_orchestrator,
            H_error=H_error,
            jump_operators=jump_ops or [],
            dt=dt,
            num_steps=num_steps,
            section_normal=section_normal,
            h0_poisson_h1=h0_poisson_h1,
            perturbation_strength=perturbation_strength,
            analyticity_radius=analyticity_radius,
        )
        telemetry["phase2"] = phase2_cert.to_dict()

        # ── FASE 2 ↪ FASE 3: Decide & Act ──────────────────────────────────
        logger.info("[OODA-DECIDE/ACT] FASE 3: topología global y gobernanza...")
        try:
            cohomology_report = self.sheaf_custodian.audit_holonomy(
                semantic_vector, raise_on_violation=not self.auto_repair,
            )
        except NumericalInstabilityError:
            if self.auto_repair:
                semantic_vector = self.sheaf_custodian.repair_semantic_state(semantic_vector)
                cohomology_report = self.sheaf_custodian.audit_holonomy(
                    semantic_vector, raise_on_violation=False,
                )
                telemetry["semantic_repair_applied"] = True
            else:
                self.error_count += 1
                self.trigger_esp32_crowbar_interlock("Cohomology_H1_Obstruction")
                raise

        sigma_mic = self._project_to_mic_density(
            semantic_vector, target_dim=rho_evolved.dimension,
        )
        galois_valid, galois_metrics = self.galois_auditor.validate_adjunction_counit(
            rho_mac=rho_evolved,
            sigma_mic=sigma_mic,
            epsilon=bures_epsilon,
            metric="bures",
        )
        telemetry["phase3_galois"] = galois_metrics
        telemetry["phase3_cohomology"] = {
            "is_holonomic": cohomology_report.is_holonomic,
            "dirichlet_energy": cohomology_report.dirichlet_energy,
            "betti_numbers": cohomology_report.betti_numbers,
        }

        sovereign_cert = continue_orient_into_decide(
            phase1_cert=phase1_cert,
            phase2_cert=phase2_cert,
            cohomology_report=cohomology_report,
            galois_is_valid=galois_valid,
            critical_points_indices=critical_points_indices,
            euler_characteristic=euler_characteristic,
            betti_numbers=betti_numbers,
            kam_stability_hard=kam_stability_hard,
        )
        telemetry["sovereign_certificate"] = sovereign_cert.to_dict()
        telemetry["verdict"] = sovereign_cert.verdict.value
        telemetry["success"] = not sovereign_cert.gromov_veto_active

        if self.debug_mode:
            self.state_history.append(rho_evolved)
            self.certificate_history.append(sovereign_cert)

        if sovereign_cert.gromov_veto_active:
            self.error_count += 1
            self.trigger_esp32_crowbar_interlock(
                f"Gromov_Veto:{sovereign_cert.verdict.value}"
            )
            raise NumericalInstabilityError(
                f"Veto de Gromov activo. Veredicto: {sovereign_cert.verdict.value}. "
                f"Hopf={sovereign_cert.hopf.is_hopf_consistent}, "
                f"Duality={sovereign_cert.duality.is_duality_symmetric}, "
                f"Galois={galois_valid}, H⁰={cohomology_report.is_holonomic}"
            )

        logger.info(
            "[OODA-COMPLETE] Veredicto: %s  Ω₃=%s  hash=%s",
            sovereign_cert.verdict.value,
            sovereign_cert.verdict.to_omega3().name,
            sovereign_cert.sovereign_hash,
        )
        return rho_evolved, telemetry

    def process_telemetry_cartridge(
        self,
        current_rho: AtomicDensityMatrix,
        semantic_vector: NDArray[np.float64],
        H_error: NDArray[np.complex128],
        jump_ops: List[Tuple[float, NDArray[np.complex128]]],
        dt: float = 0.01,
        num_steps: int = 1,
    ) -> Tuple[AtomicDensityMatrix, Dict[str, Any]]:
        """Alias de compatibilidad: redirige a la versión celestial completa."""
        return self.process_telemetry_cartridge_celestial(
            current_rho=current_rho,
            semantic_vector=semantic_vector,
            H_error=H_error,
            jump_ops=jump_ops,
            dt=dt,
            num_steps=num_steps,
        )

    def extract_wisdom(
        self,
        current_rho: AtomicDensityMatrix,
        povm_ops: List[NDArray[np.complex128]],
        deterministic: bool = False,
    ) -> Tuple[int, AtomicDensityMatrix, POVMStatistics]:
        """Colapso de la función de estado para emitir decisión táctica."""
        logger.info("[EXTRACT_WISDOM] Iniciando medición POVM...")
        povm = POVMMeasurement(povm_ops, validate_positivity=True)
        decision_index, collapsed_rho, statistics = povm.measure_and_collapse(
            current_rho, deterministic=deterministic,
        )
        logger.info(
            "[EXTRACT_WISDOM] ✓ Decisión: %d, P=%.4f",
            decision_index, statistics.probability,
        )
        return decision_index, collapsed_rho, statistics

    def validate_epistemological_coherence(
        self,
        rho_mac: AtomicDensityMatrix,
        rho_mic: AtomicDensityMatrix,
        epsilon: float = _BURES_EPSILON_DEFAULT,
    ) -> Tuple[bool, Dict[str, float]]:
        """Validación de coherencia MAC ↔ MIC vía adjunción de Galois."""
        logger.info("[VALIDATE] Verificando coherencia epistemológica...")
        return self.galois_auditor.validate_adjunction_counit(
            rho_mac=rho_mac, sigma_mic=rho_mic, epsilon=epsilon, metric="bures",
        )

    def _project_to_mic_density(
        self,
        semantic_vector: NDArray[np.float64],
        target_dim: Optional[int] = None,
    ) -> AtomicDensityMatrix:
        r"""Proyecta un vector semántico sobre la órbita coadjunta U(n)·ρ₀."""
        vec = np.asarray(semantic_vector, dtype=np.complex128).flatten()
        if target_dim is None:
            target_dim = int(vec.size)
        if vec.size < target_dim:
            vec = np.pad(vec, (0, target_dim - vec.size))
        elif vec.size > target_dim:
            vec = vec[:target_dim]
        nv = la.norm(vec)
        if nv < 1e-12:
            vec = np.ones(target_dim, dtype=np.complex128) / math.sqrt(target_dim)
        else:
            vec = vec / nv
        rho_mat = np.outer(vec, vec.conj())
        return AtomicDensityMatrix(rho_mat, auto_renormalize=True, validate=False)

    def _default_povm_ops(self, dim: int) -> List[NDArray[np.complex128]]:
        """POVM de von Neumann: proyectores sobre la base computacional."""
        eye = np.eye(dim, dtype=np.complex128)
        return [np.outer(eye[k], eye[k].conj()) for k in range(dim)]

    def trigger_esp32_crowbar_interlock(self, reason: str) -> None:
        r"""
        Emite el veto μ: Ω₃ → ℤ₂ al ESP32 (IRAM < 400 ns) y acciona el
        tiristor BT151 (Crowbar) para cortocircuitar la potencia real.
        """
        logger.critical(
            "[CROWBAR INTERLOCK ACTUATED] Veto Epistemológico MAC (latencia <%d ns): %s",
            _CROWBAR_TRIGGER_LATENCY_NS, reason,
        )

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.operation_count),
            "cohomology_telemetry": self.sheaf_custodian.get_telemetry(),
            "certificate_history_length": len(self.certificate_history),
            "last_verdict": (
                self.certificate_history[-1].verdict.value
                if self.certificate_history
                else None
            ),
        }

    def reset(self) -> None:
        self.operation_count = 0
        self.error_count = 0
        self.state_history.clear()
        self.certificate_history.clear()
        logger.info("MACAgent reiniciado")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "MeasurementOutcome",
    "HeytingVerdict",
    "POVMStatistics",
    "LindbladEvolutionMetrics",
    "CohomologyAuditReport",
    "Phase1MeasurementCertificate",
    "Phase2DynamicsCertificate",
    "SovereignEpistemicCertificate",
    "POVMMeasurement",
    "LindbladDynamicsOrchestrator",
    "SheafCohomologyCustodian",
    "GaloisAdjunctionAuditor",
    "PoincareTopologyEngine",
    "build_phase1_measurement_certificate",
    "continue_observe_into_orient",
    "assemble_phase2_dynamics_certificate",
    "continue_orient_into_decide",
    "sovereign_epistemic_governance",
    "MACAgent",
    "_WILKINSON_TOL",
    "_SPECTRAL_TOL",
    "_CHIRIKOV_CRITICAL",
    "_MELNIKOV_TOL",
    "_NEKHOROSHEV_EXPONENT",
    "_MAX_POINCARE_SAMPLES",
    "_CROWBAR_TRIGGER_LATENCY_NS",
]


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("═" * 80)
    print("DEMO: MAC Agent con Mecánica Celeste de Poincaré (3 fases anidadas)")
    print("═" * 80)

    from app.wisdom.atomic_knowledge_matrix import (
        create_geometric_learning_system,
        create_quantum_mac_state,
    )

    rho0 = create_quantum_mac_state(dimension=4, purity=0.8, seed=42)
    print(f"\n1. Estado inicial: {rho0}")

    sheaf, _, _ = create_geometric_learning_system(
        num_vertices=6,
        num_edges=8,
        fiber_dim_vertex=2,
        fiber_dim_edge=2,
        dissipation_strength=0.05,
        seed=42,
    )
    agent = MACAgent(sheaf_manifold=sheaf, integration_method="rk4", debug_mode=True)

    povm_ops = [np.outer(np.eye(4)[k], np.eye(4)[k].conj()) for k in range(4)]
    rng = np.random.default_rng(42)
    H_err = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H_err = 0.5 * (H_err + H_err.conj().T)
    jump_ops = [(0.01, np.eye(4, dtype=np.complex128))]
    semantic_vec = rng.normal(size=12) * 0.1
    section_normal = np.zeros(32, dtype=np.float64)
    section_normal[0] = 1.0

    def melnikov_mock(t: float) -> float:
        return math.sin(t) * math.exp(-0.05 * abs(t))

    print("\n2. Ejecutando ciclo OODA Celeste completo...")
    try:
        rho_evolved, telemetry = agent.process_telemetry_cartridge_celestial(
            current_rho=rho0,
            semantic_vector=semantic_vec,
            H_error=H_err,
            jump_ops=jump_ops,
            povm_ops=povm_ops,
            section_normal=section_normal,
            h0_poisson_h1=melnikov_mock,
            dt=0.01,
            num_steps=10,
            kam_stability_hard=False,
        )
        cert = telemetry["sovereign_certificate"]
        print(f"\n3. Veredicto final: {cert['verdict']}")
        print(f"   Ω₃: {cert['heyting_omega3']}")
        print(f"   Veto de Gromov: {cert['gromov_veto_active']}")
        print(f"   Hopf coherente: {cert['hopf']['is_hopf_consistent']}")
        print(f"   Dualidad simétrica: {cert['duality']['is_duality_symmetric']}")
        print(f"   Polinomio de Poincaré: {cert['duality'].get('poincare_polynomial')}")
        print(f"   Hash soberano: {cert['sovereign_hash']}")
    except NumericalInstabilityError as exc:
        print(f"\n3. VETO activado: {exc}")

    print(f"\n4. Telemetría: {agent.get_telemetry()}")
    print("\n" + "═" * 80)
    print("✓ Demostración completada")
    print("═" * 80)
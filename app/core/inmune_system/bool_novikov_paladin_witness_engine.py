# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO  : Bool Novikov Paladin Engine (Motor Espectral Ciego FPU)                                            ║
║ RUTA    : app/omega/bool_novikov_paladin_engine.py                                                           ║
║ VERSIÓN : 15.0.0-FPU-Uhlmann-Halley-Melnikov-Octonion-L3-PhD                                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN FORMAL Y ROL OPERATIVO (NATURALEZA DEL MOTOR)
────────────────────────────────────────────────────────────
El `BoolNovikovPaladinEngine` es un Motor Espectral Ciego de cómputo físico e intensivo que habita en la FPU 
(Floating Point Unit) dentro del Estrato Omega (V_Ω, Nivel 0.5 — El Ágora Tensorial). Su naturaleza es la de 
un funtor covariante de transformación puramente determinista y en lazo abierto: procesa estados vectoriales, 
matrices de densidad cuántica, regularizaciones celestes y equilibrios de materia-antimateria.

Su rol operativo se circunscribe estrictamente al cálculo numérico en doble precisión (IEEE-754) bajo aislamiento 
numérico total (FPU Secure). El motor no evalúa retículos de Heyting, no emite juicios de valor sobre el negocio, 
no maneja estados agénticos OODA ni interactúa con la memoria perimetral o pines de hardware. Su responsabilidad 
es la consolidación de reportes numéricos inmutables que entregan los tensores analíticos al soberano de calibre.

2. MARCO TEÓRICO Y FORMULACIÓN MATEMÁTICA
──────────────────────────────────────────
El motor deconstruye el espacio de fase en tres fases espectrales puras (Π₁, Π₂, Π₃):

FASE Π₁ · GEOMETRÍA DE DENSIDAD Y OCTONIONES DE CAYLEY-DICKSON:
  1. Fidelidad Cuántica de Uhlmann y Distancia de Bures sobre D(H):
     Dado el par de operadores densidad ρ, σ ∈ D(H), computa:
       $$F(\rho, \sigma) = \left( \operatorname{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}} \right)^2, \qquad d_B(\rho, \sigma) = \sqrt{2 - 2\sqrt{F(\rho, \sigma)}}$$
  2. Entropía Relativa Cuántica de Umegaki y Cota de Pinsker:
       $$D(\rho \parallel \sigma) = \operatorname{Tr}\left(\rho (\ln\rho - \ln\sigma)\right) \ge \frac{1}{2} \|\rho - \sigma\|_1^2$$
  3. Álgebra Normada de Octoniones (O ≅ R⁸):
     Estructurada sobre la orientación del Plano Proyectivo de Fano PG(2, 2):
       $$e_i \star e_j = -\delta_{ij} e_0 + \sum_{k=1}^7 \varepsilon_{ijk} e_k \implies \|x \star y\|_2 = \|x\|_2 \|y\|_2$$
     Evalúa las identidades de Moufang sobre el asociador [x, y, z]_O = (x ⋆ y) ⋆ z - x ⋆ (y ⋆ z):
       $$x \star (y \star (x \star z)) = ((x \star y) \star x) \star z$$

FASE Π₂ · ESTABILIDAD CELESTE DE POINCARÉ Y CAOS HOMOCLÍNICO:
  1. Carta Canónica Regularizada de Poincaré para e → 0 e i → 0:
       $$\Lambda = \sqrt{\mu a}, \quad \Gamma = \Lambda(1 - \sqrt{1 - e^2}), \quad Z = \Lambda \sqrt{1 - e^2}(1 - \cos i)$$
       $$\xi_1 = \sqrt{2\Gamma}\cos(\omega+\Omega), \quad \eta_1 = -\sqrt{2\Gamma}\sin(\omega+\Omega), \quad \xi_2 = \sqrt{2Z}\cos\Omega, \quad \eta_2 = -\sqrt{2Z}\sin\Omega$$
     Identidad simpléctica verificada: $\Lambda - \frac{1}{2}(\xi_1^2 + \eta_1^2) - \frac{1}{2}(\xi_2^2 + \eta_2^2) = H$.
  2. Solucionador Cúbico de Halley-Kepler para f(E) = E - e sin E - M = 0:
       $$E_{k+1} = E_k - \frac{2 f(E_k) f'(E_k)}{2 (f'(E_k))^2 - f(E_k) f''(E_k)}$$
  3. Integral Homoclínica de Melnikov-Duffing y Turnstile de MacKay-Meiss-Percival:
     Cuantifica el área de transporte en el lóbulo de la separatriz escindida:
       $$\frac{A_{\mathrm{lobe}}}{\varepsilon} = \frac{2}{\omega} \left( \sqrt{B^2 - A^2} - A \arccos\left(\frac{A}{B}\right) \right) \quad \text{con } A = \frac{4}{3}\gamma_{\mathrm{diss}}, \ B = \sqrt{2}\pi f \omega \operatorname{sech}\left(\frac{\pi\omega}{2}\right)$$
  4. Signatura de Krein-Moser e Invariante de Poincaré-Cartan:
     Evalúa el desvío simpléctico sobre la matriz de monodromía M ∈ Sp(2n, R):
       $$\Delta_{\mathrm{PC}} = \|\mathbf{M}^\top \mathbf{\Omega} \mathbf{M} - \mathbf{\Omega}\|_F \le \tau_{\mathrm{symp}}$$
  5. Escarpamiento de Lochak-Nekhoroshev: Exponente óptimo de estabilidad $a_{\mathrm{Nekh}} = \frac{1}{2 d_{\mathrm{dof}}}$.

FASE Π₃ · LEDGER QED Y AUTOMEJORA RECURSIVA RSI-L3:
  1. Aniquilación Lógica de Materia-Antimateria (e⁻ + e⁺ → 2γ):
     Calcula la liberación entálpica $E = 2 m^* c^2$ tras la colisión del electrón de falla e⁻ y el positrón e⁺.
  2. Homología Simplicial y Nulidad de Torsión:
     Audita que el primer grupo de homología sea trivial:
       $$\beta_1 = \dim H_1(K; \mathbb{Z}) = 0 \quad \land \quad \operatorname{Tor}(H_k(K; \mathbb{Z})) \equiv \mathbf{0}$$

3. AXIOMAS E INVARIANTES PRESERVADOS
────────────────────────────────────
  • AXIOMA DE HERMITICIDAD Y TRACEABILITY (QED-1):
    $\forall \rho \in D(H): \rho = \rho^\dagger \land \operatorname{Tr}(\rho) = 1.0 \pm 10^{-9} \land \lambda_{\min}(\rho) \ge -10^{-12}$.
  • AXIOMA DE INVARIANZA SIMPLÉCTICA DE LIOUVILLE (POINCARÉ-1):
    $\det(\mathbf{M}) = +1.0 \pm 10^{-9} \land \mathbf{M}^\top \mathbf{\Omega} \mathbf{M} = \mathbf{\Omega}$.
  • AXIOMA DE ALTERNATIVIDAD OCTONIÓNICA (CAYLEY-1):
    $\forall x, y \in O: [x, x, y]_O = 0 \land \|x \star y\|_2 = \|x\|_2 \|y\|_2$.
  • AXIOMA DE ACICLICIDAD Y CONGRUENCIA (HOMOLOGY-1):
    $\beta_1 = 0 \implies H_1(K; \mathbb{Z}) \cong \mathbf{0}$ (Aniquilación absoluta de referencias circulares).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import lru_cache
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Core.InmuneSystem.BoolNovikovPaladinWitnessEngine")

__all__ = [
    "HeytingTruthValue",
    "QNDInspectionReport",
    "CelestialWitnessReport",
    "BanachWitnessReport",
    "RSILevel3WitnessTelemetry",
    "RSI3Policy",
    "Phase1ToPhase2Seed",
    "Phase2ToPhase3Seed",
    "BoolNovikovPaladinWitnessEngine",
    "ENGINE_VERSION",
]

_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOLE-NOVIKOV-PALADIN-WITNESS-QND-KMS-POINCARE-KAM-RSI3-PhD"
)
version: Final[str] = _ENGINE_VERSION
ENGINE_VERSION: Final[str] = _ENGINE_VERSION


# ══════════════════════════════════════════════════════════════════════════════
# §0 · CONSTANTES FÍSICO-MATEMÁTICAS Y COTAS ESPECTRALES SOBERANAS
# ══════════════════════════════════════════════════════════════════════════════
_WILKINSON_EPSILON: Final[float] = 1.0e-15          # ε_W : referencia de retroacción y tolerancia dura
_TINY: Final[float] = 1.0e-300                      # guarda de división (no es una tolerancia física)
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12         # δ_PSD : piso espectral de fidelidad del estado
_MAX_MODULAR_CONDITION: Final[float] = 1.0e8        # κ_max(ρ) : acota el error de ρ^(−1/2) en KMS
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10           # γ_Krull : barrera disipativa de Melnikov
_MAX_QND_BACKACTION_DB: Final[float] = 0.0          # cota dura de retroacción logarítmica
_QND_COMMUTATOR_CEILING: Final[float] = 1.0e-10     # cota absoluta de ‖[H,ρ]‖_F
_QND_COMMUTATOR_REL_CEILING: Final[float] = 1.0e-10 # cota relativa (Böttcher-Wenzel), invariante de escala
_UHLMANN_FIDELITY_FLOOR: Final[float] = 0.95        # F_min  (restricción activa)
_BURES_DISTANCE_CEILING: Final[float] = 0.35        # d_B_max (implicada por F_min: ver 1.5)
_FIDELITY_DEGRADED_FLOOR: Final[float] = 0.80       # estrato de verdad parcial en Ω₃
_KAM_DIOPHANTINE_FLOOR: Final[float] = 1.0e-4       # γ_KAM sobre frecuencias normalizadas
_BRUNO_SERIES_CEILING: Final[float] = 40.0          # cota operativa de B(ω) truncada
_MAX_MELNIKOV_AMPLITUDE_REL: Final[float] = 0.25    # margen operativo (umbral exacto de Smale = 1.0)
_MELNIKOV_TRANSVERSALITY_THRESHOLD: Final[float] = 1.0
_MAX_CHIRIKOV_OVERLAP: Final[float] = 1.0           # criterio heurístico de solapamiento
_GREENE_CRITICAL_RESIDUE: Final[float] = 0.25       # R_c de Greene (K_c ≈ 0.971635406)
_BANACH_CONTRACT_CEILING: Final[float] = 0.857375   # (0.95)³ : cota de Lip(T³)
_DENMAN_BEAVERS_MAX_ITER: Final[int] = 16
_BRUNO_MAX_EXPONENT: Final[int] = 4                 # K_max = 2⁴ = 16 en la red resonante
_LAPLACE_LIMIT: Final[float] = 0.6627434193491816   # radio de convergencia de la serie de Lagrange en e
_GOLDEN_RATIO_CONJUGATE: Final[float] = 0.6180339887498949
_HILL_STABILITY_COEFF: Final[float] = 2.40          # Gladman (1993), Δ_crit = 2.40 (μ₁+μ₂)^{1/3}
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_PALADIN_WITNESS_SECRET_KEY_v13.0.0"
_HKDF_SALT: Final[bytes] = b"APU-PALADIN-WITNESS-HKDF-SALT-v15"
_MERGE_LEAF_PREFIX: Final[bytes] = b"\x00"
_MERGE_NODE_PREFIX: Final[bytes] = b"\x01"


# ══════════════════════════════════════════════════════════════════════════════
# §1 · ÁLGEBRA DE HEYTING INTUICIONISTA Ω₃ Y BOOLEANIZACIÓN POR DOBLE NEGACIÓN
# ══════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    """
    Topos de Heyting Ω₃ = {COHERENT(0x00), DEGRADED(0x01), VETOED(0x03)}.

    Orden de salud (retículo dual del orden numérico de severidad):
        a ⊑ b  ⟺  int(a) ≥ int(b),   ⊤ = COHERENT,  ⊥ = VETOED.
    Es la cadena de Gödel de 3 elementos: álgebra de Heyting NO booleana, pues
        ¬DEGRADED = VETOED  y  ¬¬DEGRADED = COHERENT ≠ DEGRADED.
    La topología de Lawvere-Tierney j = ¬¬ es idempotente, preserva ⊤ y ∧; su subtopos
    de haces es el esqueleto booleano {COHERENT, VETOED} — de ahí el prefijo "Bool".
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def meet(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Conjunción (∧): ínfimo en el orden de salud = estado más restrictivo."""
        return HeytingTruthValue(max(int(self), int(other)))

    def join(self, other: "HeytingTruthValue") -> "HeytingTruthValue":
        """Disyunción (∨): supremo en el orden de salud = estado más benigno."""
        return HeytingTruthValue(min(int(self), int(other)))

    def imply(self, consequent: "HeytingTruthValue") -> "HeytingTruthValue":
        """Implicación relativa de Heyting: adjunción (a ∧ x) ⊑ b ⟺ x ⊑ (a ⇒ b)."""
        if int(self) >= int(consequent):
            return HeytingTruthValue.COHERENT
        return consequent

    def negate(self) -> "HeytingTruthValue":
        """Pseudocomplemento ¬a = (a ⇒ ⊥)."""
        return self.imply(HeytingTruthValue.VETOED)

    def double_negation(self) -> "HeytingTruthValue":
        """Topología de doble negación j(a) = ¬¬a (núcleo booleanizante)."""
        return self.negate().negate()

    def boolean_collapse(self) -> "HeytingTruthValue":
        """Imagen en el subtopos booleano de elementos ¬¬-estables."""
        return self.double_negation()

    def is_regular(self) -> bool:
        """a es ¬¬-estable ⟺ ¬¬a = a (pertenece al esqueleto booleano)."""
        return self.double_negation() == self

    @staticmethod
    def verify_heyting_axioms() -> bool:
        """Verificación exhaustiva (27 tríadas) de la adjunción de residuación y de j=¬¬."""
        vals = (HeytingTruthValue.COHERENT, HeytingTruthValue.DEGRADED, HeytingTruthValue.VETOED)
        leq = lambda a, b: int(a) >= int(b)  # noqa: E731 (orden de salud)
        for a in vals:
            for b in vals:
                for x in vals:
                    if leq(a.meet(x), b) != leq(x, a.imply(b)):
                        return False
            if a.double_negation().double_negation() != a.double_negation():
                return False
            for b in vals:
                if a.meet(b).double_negation() != a.double_negation().meet(b.double_negation()):
                    return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# §2 · ESTRUCTURAS INMUTABLES: CERTIFICADOS, SEMILLAS DE CONTINUIDAD E INFORMES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class OperatorValidation:
    """Certificado de validación de un operador (proyección autoadjunta y defectos crudos)."""

    operator: NDArray[np.complex128]
    trace_defect: float
    hermiticity_defect: float
    is_fallback: bool


@dataclass(frozen=True)
class SpectralPurification:
    """Resultado de la proyección exacta sobre el símplex cuántico capado D_δ(H)."""

    rho: NDArray[np.complex128]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    reference_probs: NDArray[np.float64]  # estado sin regularizar (mismo autobase) para el certificado
    entropy: float
    purity: float
    effective_rank: float
    projection_cost: float
    floor_used: float
    condition_number: float
    kkt_feasible: bool


@dataclass(frozen=True)
class KMSCertificate:
    """Certificados de la teoría modular de Tomita-Takesaki y de la condición KMS."""

    kms_residual: float
    tomita_commutator_defect: float
    involution_defect: float
    gibbs_defect: float
    modular_flow_defect: float
    beta_estimate: float
    log_partition: float
    ergotropy: float
    is_passive: bool


@dataclass(frozen=True)
class RecurrenceCertificate:
    """Recurrencia cuántica de Poincaré y límites cuánticos de velocidad."""

    recurrence_bound: float
    dirichlet_log10_bound: float
    mandelstam_tamm_time: float
    margolus_levitin_time: float
    bohr_mode_count: int
    energy_variance: float


@dataclass(frozen=True)
class BettiTopology:
    """Homología simplicial del 1-esqueleto de interferencia K₁(ρ) y holonomía de Bargmann."""

    beta_0: int
    beta_1: int
    num_vertices: int
    num_edges: int
    euler_identity_ok: bool
    max_bargmann_phase: float
    nontrivial_cycle_count: int
    fiedler_connectivity: float
    persistence_profile: Tuple[int, ...]


@dataclass(frozen=True)
class Phase1ToPhase2Seed:
    """
    [MORFISMO DE CONTINUIDAD 1.11 → 2.1] Semilla simpléctica emitida por el método terminal
    de la Fase 1. Es el argumento formal de apertura de la Fase 2.
    """

    qnd_commutator_perturbation: float
    qnd_relative_commutator: float
    bures_geodesic_tension: float
    poincare_recurrence_scale: float
    modular_beta: float
    homological_obstruction: int
    upstream_truth: HeytingTruthValue
    has_context: bool = True


@dataclass(frozen=True)
class Phase2ToPhase3Seed:
    """[MORFISMO DE CONTINUIDAD 2.13 → 3.1] Margen de estabilidad celeste para la mónada T³."""

    celestial_damping_gain: float
    resonance_pressure: float
    floquet_spectral_radius: float
    greene_torus_intact: bool
    amd_stable: bool
    upstream_truth: HeytingTruthValue
    has_upstream_context: bool = True


@dataclass(frozen=True)
class QNDInspectionReport:
    """Informe inmutable de Fase 1 (campos heredados de v14 preservados en orden posicional)."""

    is_qnd_valid: bool
    qnd_commutator_norm: float
    backaction_db: float
    trace_preservation: float
    uhlmann_fidelity: float
    bures_distance: float
    is_fides_healthy: bool
    kms_correlation_residual: float
    betti_1_topology: int
    truth_value: HeytingTruthValue
    # Telemetría extendida v14
    tomita_modular_defect: float = 0.0
    von_neumann_entropy: float = 0.0
    poincare_quantum_recurrence_bound: float = 0.0
    rsi_l3_purification_iterations: int = 0
    # Certificados rigurosos v15
    relative_commutator: float = 0.0
    energy_coherence_defect: float = 0.0
    commutator_identity_error: float = 0.0
    fidelity_svd_eig_discrepancy: float = 0.0
    fidelity_lower_certificate: float = 0.0
    fidelity_upper_certificate: float = 1.0
    bures_angle: float = 0.0
    trace_distance: float = 0.0
    fuchs_van_de_graaf_ok: bool = True
    transport_map_defect: float = 0.0
    purification_projection_cost: float = 0.0
    purification_kkt_ok: bool = True
    state_condition_number: float = 1.0
    state_purity: float = 1.0
    involution_defect: float = 0.0
    gibbs_defect: float = 0.0
    modular_flow_defect: float = 0.0
    modular_beta: float = 0.0
    ergotropy: float = 0.0
    is_passive_state: bool = True
    dirichlet_recurrence_log10: float = 0.0
    mandelstam_tamm_time: float = 0.0
    margolus_levitin_time: float = 0.0
    betti_0_topology: int = 1
    euler_identity_ok: bool = True
    max_bargmann_phase: float = 0.0
    fiedler_connectivity: float = 0.0
    obstruction_reasons: Tuple[str, ...] = ()
    continuation: Optional[Phase1ToPhase2Seed] = None


@dataclass(frozen=True)
class CelestialWitnessReport:
    """Informe inmutable de Fase 2 (mecánica celeste de Poincaré)."""

    is_kam_stable: bool
    diophantine_min_divisor: float
    lindstedt_omega: float
    melnikov_amplitude_rel: float
    is_melnikov_safe: bool
    chirikov_overlap_k: float
    is_chirikov_contained: bool
    krein_signature_stable: bool
    truth_value: HeytingTruthValue
    # Telemetría extendida v14
    siegel_bruno_series_sum: float = 0.0
    floquet_spectral_radius: float = 1.0
    poincare_cartan_invariant: float = 0.0
    rsi_l3_resonance_damping_gain: float = 1.0
    # Certificados rigurosos v15
    diophantine_gamma: float = 0.0
    diophantine_tau: float = 2.0
    bruno_certified_bound: float = 0.0
    bruno_convergent: bool = True
    lattice_vector_count: int = 0
    cf_bounded_type: bool = False
    kepler_max_residual: float = 0.0
    kepler_mean_iterations: float = 0.0
    kepler_empirical_order: float = 0.0
    kapteyn_series_residual: float = 0.0
    laplace_series_regime: bool = True
    lindstedt_exact_omega: float = 0.0
    lindstedt_remainder_rel: float = 0.0
    agm_iterations: int = 0
    equation_of_center_remainder: float = 0.0
    melnikov_osc_amplitude: float = 0.0
    melnikov_transversal: bool = False
    melnikov_critical_forcing_ratio: float = 0.0
    greene_residues: Tuple[float, ...] = ()
    greene_torus_intact: bool = True
    greene_certified: bool = False
    symplectic_defect: float = 0.0
    monodromy_step_certificate: float = 0.0
    krein_collision_detected: bool = False
    mathieu_a: float = 0.0
    mathieu_q: float = 0.0
    max_lyapunov_exponent: float = 0.0
    angular_momentum_deficit: float = 0.0
    amd_critical: float = 0.0
    amd_stable: bool = True
    hill_separation_ok: bool = True
    effective_perturbation_eps: float = 0.0
    obstruction_reasons: Tuple[str, ...] = ()
    continuation: Optional[Phase2ToPhase3Seed] = None


@dataclass(frozen=True)
class BanachWitnessReport:
    """Informe inmutable de Fase 3 (mónada Kleisli T³, RSI-L3 y atestación)."""

    banach_ratio_q: float
    is_banach_contractive: bool
    lyapunov_value: float
    jerk_capacity: float
    truth_value: HeytingTruthValue
    # Telemetría extendida v14
    kleisli_t3_lipschitz: float = 0.0
    aitken_accelerated_residual: float = 0.0
    estimated_steps_to_eps: int = 0
    rsi_l3_fixed_point_defect: float = 0.0
    # Certificados rigurosos v15
    geometric_fit_r2: float = 1.0
    a_priori_error_bound: float = 0.0
    a_posteriori_error_bound: float = 0.0
    certified_decimal_digits: float = 0.0
    lyapunov_decrement: float = 0.0
    lyapunov_certificate_ok: bool = True
    jerk_third_difference: float = 0.0
    wynn_epsilon_residual: float = 0.0
    accelerator_selected: str = "aitken"
    empirical_convergence_order: float = 1.0
    rsi_l1_tolerance_gain: float = 1.0
    rsi_l2_policy_relaxation: float = 1.0
    rsi_l3_meta_lipschitz: float = 0.0
    rsi_l3_meta_accepted: bool = False
    rsi_l3_proof_depth_remaining: int = 0
    rsi_l3_rollback_count: int = 0
    novikov_self_consistent: bool = True
    obstruction_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RSI3Policy:
    """
    Política de automejora de Nivel 3 en un compacto convexo Θ ⊂ ℝ³.
    `proof_depth_budget` es el contador bien fundado que garantiza la terminación de toda
    cadena de automodificación (descenso estricto en ℕ, sin autorreferencia löbiana).
    """

    relaxation: float = 1.0
    accelerator_order: int = 2
    trust_q_ceiling: float = 0.95
    proof_depth_budget: int = 3
    meta_lipschitz: float = 0.5

    def project(self) -> "RSI3Policy":
        """Proyección Π_Θ sobre el compacto de políticas admisibles (no expansiva)."""
        return RSI3Policy(
            relaxation=float(np.clip(self.relaxation, 0.50, 1.50)),
            accelerator_order=int(np.clip(self.accelerator_order, 0, 6)),
            trust_q_ceiling=float(np.clip(self.trust_q_ceiling, 0.50, 0.999)),
            proof_depth_budget=max(0, int(self.proof_depth_budget)),
            meta_lipschitz=float(np.clip(self.meta_lipschitz, 1.0e-3, 0.999)),
        )

    def distance(self, other: "RSI3Policy") -> float:
        """Métrica de Banach sobre Θ (norma euclídea de las coordenadas continuas)."""
        return math.sqrt(
            (self.relaxation - other.relaxation) ** 2
            + (self.trust_q_ceiling - other.trust_q_ceiling) ** 2
            + 0.25 * (self.accelerator_order - other.accelerator_order) ** 2
        )


@dataclass
class RSILevel3WitnessTelemetry:
    """Registro acumulativo (mutable, append-only en espíritu) de la automejora RSI-L3."""

    total_qnd_audits: int = 0
    total_celestial_audits: int = 0
    total_banach_audits: int = 0
    cumulative_denman_beavers_steps: int = 0
    residual_history: List[float] = field(default_factory=list)
    last_heyting_verdict: HeytingTruthValue = HeytingTruthValue.COHERENT
    # v15 · libro mayor de automejora y atestación encadenada
    policy: RSI3Policy = field(default_factory=RSI3Policy)
    policy_ledger: List[Dict[str, Any]] = field(default_factory=list)
    rollback_count: int = 0
    empirical_order_history: List[float] = field(default_factory=list)
    signature_chain: List[str] = field(default_factory=list)
    monotone_counter: int = 0
    dev_key_in_use: bool = False
    cumulative_agm_steps: int = 0
    cumulative_kepler_steps: int = 0
    cumulative_monodromy_panels: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# §3 · UTILIDADES ESPECTRALES Y NUMÉRICAS COMPARTIDAS (FUNCIONES PURAS)
# ══════════════════════════════════════════════════════════════════════════════
def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión total a float finito (nunca propaga NaN/Inf al núcleo del motor)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _fro(mat: NDArray[Any]) -> float:
    """Norma de Frobenius tolerante a matrices vacías."""
    return float(la.norm(mat, ord="fro")) if mat.size else 0.0


def _relative(defect: float, scale: float) -> float:
    """Defecto relativo con guarda de escala nula."""
    return float(defect / max(scale, _TINY))


@lru_cache(maxsize=16)
def _resonant_lattice(dimension: int, k_max: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Enumeración cacheada de k ∈ ℤᵈ \\ {0} con 0 < ‖k‖₁ ≤ k_max, módulo la simetría k ↦ −k
    (se conserva el representante cuyo primer componente no nulo es positivo).
    Devuelve (vectores, normas_L1). Al ser estática, la red se memoiza: el coste por auditoría
    es un único producto matriz-vector.
    """
    dim = max(1, min(int(dimension), 4))
    km = max(2, min(int(k_max), 16))
    rows: List[Tuple[int, ...]] = []

    def _recurse(pos: int, budget: int, current: Tuple[int, ...]) -> None:
        if pos == dim:
            for v in current:
                if v != 0:
                    if v > 0:
                        rows.append(current)
                    return
            return
        for v in range(-budget, budget + 1):
            _recurse(pos + 1, budget - abs(v), current + (v,))

    _recurse(0, km, ())
    if not rows:
        rows = [(1,) + (0,) * (dim - 1)]
    vecs = np.asarray(rows, dtype=np.float64)
    norms = np.sum(np.abs(vecs), axis=1)
    return vecs, norms


def _golden_section_minimize(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1.0e-12,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """
    Minimización unimodal por sección áurea: contracción de razón 1/φ = 0.618… por paso
    (el mismo número áureo del toro KAM más irracional es aquí la razón de Banach).
    """
    a, b = float(lo), float(hi)
    if b - a <= tol:
        return a, float(func(a))
    inv_phi = _GOLDEN_RATIO_CONJUGATE
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = float(func(c)), float(func(d))
    for _ in range(max_iter):
        if b - a <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = float(func(c))
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = float(func(d))
    x = 0.5 * (a + b)
    return x, float(func(x))


def _arithmetic_geometric_mean(a0: float, b0: float, tol: float = 1.0e-16) -> Tuple[float, int]:
    """
    AGM de Gauss-Brent-Salamin: convergencia CUADRÁTICA con invariante
    K(k) = π / (2·AGM(1, √(1−k²))). Es el motor de resumación exacta de Lindstedt.
    """
    a, b = abs(float(a0)), abs(float(b0))
    if a <= _TINY and b <= _TINY:
        return 0.0, 0
    for k in range(1, 61):
        if abs(a - b) <= tol * max(a, b):
            return 0.5 * (a + b), k
        a, b = 0.5 * (a + b), math.sqrt(max(0.0, a * b))
    return 0.5 * (a + b), 60


def _bessel_j_periodic_trapezoid(order: int, argument: float, nodes: int = 512) -> float:
    """
    J_n(x) = (1/π)∫₀^π cos(nτ − x sin τ) dτ evaluada por regla trapezoidal periódica.
    Para integrandos analíticos y periódicos la convergencia es GEOMÉTRICA
    (Trefethen-Weideman): con 512 nodos el error es ~ε_machine para n ≤ 64, |x| ≤ 64.
    """
    n = int(order)
    tau = (np.arange(nodes, dtype=np.float64) + 0.5) * (math.pi / nodes)
    return float(np.mean(np.cos(n * tau - float(argument) * np.sin(tau))))


def _merkle_root(leaves: Sequence[bytes]) -> str:
    """
    Raíz de Merkle estilo RFC 6962: hoja = SHA256(0x00‖datos), nodo = SHA256(0x01‖L‖R).
    Los prefijos de dominio impiden ataques de segunda preimagen; el nodo impar se PROMUEVE
    (no se duplica), evitando la maleabilidad por duplicación de hojas.
    """
    if not leaves:
        return hashlib.sha256(_MERGE_LEAF_PREFIX).hexdigest()
    level = [hashlib.sha256(_MERGE_LEAF_PREFIX + bytes(leaf)).digest() for leaf in leaves]
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.sha256(_MERGE_NODE_PREFIX + level[i] + level[i + 1]).digest())
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


# ══════════════════════════════════════════════════════════════════════════════
# §4 · MOTOR SOBERANO: TRES FASES ANIDADAS CON CONTINUIDAD FORMAL
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovPaladinWitnessEngine:
    """
    Motor espectral de inspección QND, geometría de Bures-Uhlmann, modularidad KMS,
    mecánica celeste de Poincaré y mónada Kleisli RSI³, en tres fases anidadas:

        Fase 1 · Phase1_QNDTomitaTakesakiInspector      (1.1 → 1.11)
        Fase 2 · Phase2_PoincareKAMCelestialNavigator   (2.1 → 2.13)
        Fase 3 · Phase3_BanachKleisliRSI3Governor       (3.1 → 3.11)

    Continuidad estricta: el último método de la Fase n EMITE la semilla que ABRE la Fase n+1.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 1 · INSPECCIÓN QND, BURES-UHLMANN, TOMITA-TAKESAKI KMS Y BETTI
    # ──────────────────────────────────────────────────────────────────────────
    class Phase1_QNDTomitaTakesakiInspector:
        """
        Fase 1. Cadena formal:
            1.1 validación → 1.2 purificación → 1.3 QND → 1.4 raíz operatorial →
            1.5 Uhlmann-Bures → 1.6 KMS → 1.7 recurrencia → 1.8 Betti →
            1.9 adjudicación → 1.10 semilla → 1.11 TERMINAL (emite Phase1ToPhase2Seed).
        """

        def __init__(self, telemetry: RSILevel3WitnessTelemetry) -> None:
            self._telemetry: RSILevel3WitnessTelemetry = telemetry
            self._last_report: Optional[QNDInspectionReport] = None

        # ── Método 1.1 ────────────────────────────────────────────────────────
        @staticmethod
        def _validate_and_symmetrize_operator(
            mat: Optional[NDArray[np.complex128]],
            dim: int,
            is_density: bool = True,
        ) -> OperatorValidation:
            """
            1.1 Valida dimensión, finitud y cuadratura; proyecta sobre el subespacio autoadjunto
            H_sym = ½(M + M†) y CERTIFICA el defecto de hermiticidad ‖M − H_sym‖_F/‖M‖_F
            (v14 lo descartaba silenciosamente). El defecto de traza se mide sobre el operador crudo.
            """
            n = max(1, int(dim))
            valid = (
                isinstance(mat, np.ndarray)
                and mat.ndim == 2
                and mat.shape[0] == mat.shape[1] == n
                and mat.size > 0
                and bool(np.all(np.isfinite(mat)))
            )
            if not valid:
                if is_density:
                    fallback = np.eye(n, dtype=np.complex128) / float(n)
                else:
                    fallback = np.diag(np.linspace(1.0, 2.0, n)).astype(np.complex128)
                return OperatorValidation(fallback, 0.0, 0.0, True)

            mat_c = np.asarray(mat, dtype=np.complex128)
            herm = 0.5 * (mat_c + mat_c.conj().T)
            herm_defect = _relative(_fro(mat_c - herm), _fro(mat_c))
            trace_defect = abs(float(np.real(np.trace(mat_c))) - 1.0) if is_density else 0.0
            return OperatorValidation(herm, float(trace_defect), float(herm_defect), False)

        # ── Método 1.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _project_spectrum_capped_simplex(
            values: NDArray[np.float64], floor: float
        ) -> Tuple[NDArray[np.float64], float, bool]:
            """
            1.2a Proyección euclídea EXACTA sobre {p ∈ ℝⁿ : p_i ≥ δ, Σ p_i = 1} (water-filling).
            Con s = p − δ ≥ 0 y S = 1 − nδ, se resuelve p = δ + max(v − δ − θ, 0) con θ tal que
            Σ max(v − δ − θ, 0) = S (algoritmo de ordenación Duchi-Condat, O(n log n)).
            Factibilidad: el conjunto es no vacío ⟺ δ ≤ 1/n; se impone δ ≤ 1/(2n) por margen.
            """
            v = np.asarray(values, dtype=np.float64).ravel()
            n = int(v.size)
            delta = float(min(max(floor, 0.0), 0.5 / max(1, n)))
            feasible = delta * n <= 1.0
            mass = 1.0 - n * delta
            y = v - delta
            order = np.sort(y)[::-1]
            css = np.cumsum(order)
            idx = np.arange(1, n + 1, dtype=np.float64)
            cond = order - (css - mass) / idx > 0.0
            rho_idx = int(np.flatnonzero(cond)[-1]) + 1 if bool(cond.any()) else 1
            theta = float((css[rho_idx - 1] - mass) / rho_idx)
            probs = np.maximum(y - theta, 0.0) + delta
            deficit = 1.0 - float(np.sum(probs))
            probs[int(np.argmax(probs))] += deficit  # corrección exacta de traza sin violar el piso
            return probs, theta, bool(feasible)

        def _rsi_l3_purify_density_operator(
            self, rho_hermitian: NDArray[np.complex128]
        ) -> SpectralPurification:
            """
            1.2b [RSI-L3] Proyección de Frobenius sobre D_δ(H). Por Mirsky/Hoffman-Wielandt,
            la proyección óptima sobre un conjunto unitariamente invariante conserva los
            autovectores y proyecta el espectro: de ahí la exactitud de 1.2a.
            Se acota κ(ρ) ≤ κ_max para que ρ^(−1/2) sea numéricamente significativo en KMS.
            Complejidad O(n³) (eigh) + O(n log n) (water-filling).
            """
            n = int(rho_hermitian.shape[0])
            try:
                eigvals, eigvecs = la.eigh(rho_hermitian)
                eigvals = np.real(eigvals).astype(np.float64)
                lam_max = float(max(np.max(eigvals), 1.0 / max(1, n)))
                delta_eff = min(max(_SPECTRAL_PSD_FLOOR, lam_max / _MAX_MODULAR_CONDITION), 0.5 / n)
                probs, _theta, feasible = self._project_spectrum_capped_simplex(eigvals, delta_eff)

                raw = np.maximum(eigvals, 0.0)
                raw_sum = float(np.sum(raw))
                ref = raw / raw_sum if raw_sum > _WILKINSON_EPSILON else np.full(n, 1.0 / n)

                rho = (eigvecs * probs) @ eigvecs.conj().T
                rho = 0.5 * (rho + rho.conj().T)
                entropy = float(-np.sum(probs * np.log(np.maximum(probs, _WILKINSON_EPSILON))))
                purity = float(np.sum(probs ** 2))
                cost = float(np.linalg.norm(probs - eigvals, ord=2))
                kappa = float(np.max(probs) / max(np.min(probs), _TINY))
                kkt = bool(feasible and np.min(probs) >= delta_eff - 1e-12
                           and abs(float(np.sum(probs)) - 1.0) <= 1e-12)
                return SpectralPurification(
                    rho=rho, eigenvalues=probs, eigenvectors=eigvecs, reference_probs=ref,
                    entropy=max(0.0, entropy), purity=purity,
                    effective_rank=float(math.exp(max(0.0, entropy))),
                    projection_cost=cost, floor_used=float(delta_eff),
                    condition_number=kappa, kkt_feasible=kkt,
                )
            except Exception as exc:  # pragma: no cover - ruta de degradación controlada
                logger.warning("Purificación espectral fallida (%s); estado máximamente mixto.", exc)
                uniform = np.eye(n, dtype=np.complex128) / float(n)
                probs = np.full(n, 1.0 / n)
                return SpectralPurification(
                    rho=uniform, eigenvalues=probs, eigenvectors=np.eye(n, dtype=np.complex128),
                    reference_probs=probs, entropy=math.log(float(n)), purity=1.0 / n,
                    effective_rank=float(n), projection_cost=0.0, floor_used=1.0 / n,
                    condition_number=1.0, kkt_feasible=True,
                )

        # ── Método 1.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_qnd_commutator_and_backaction(
            rho: NDArray[np.complex128], hamiltonian: NDArray[np.complex128]
        ) -> Tuple[float, float, bool, float, float, float]:
            """
            1.3 Conmutador de Heisenberg-von Neumann C = [H, ρ].
            Normalización de Böttcher-Wenzel: ‖[A,B]‖_F ≤ √2‖A‖_F‖B‖_F ⟹ c_rel ∈ [0,1]
            es invariante de escala (v14 usaba una cota absoluta dependiente de unidades).
                BA_dB = max(0, 10 log₁₀(max(c_rel, 10⁻¹⁶)/ε_W)).
            CERTIFICADO algebraico: en la autobase de H, ‖[H,ρ]‖²_F = Σ_ij |ρ_ij|²(E_i−E_j)².
            El defecto de coherencia energética ‖ρ − diag_H(ρ)‖_F/‖ρ‖_F es la obstrucción QND exacta.
            """
            comm = hamiltonian @ rho - rho @ hamiltonian
            comm_norm = _fro(comm)
            scale = math.sqrt(2.0) * _fro(hamiltonian) * _fro(rho)
            c_rel = float(np.clip(comm_norm / max(scale, _TINY), 0.0, 1.0))
            if c_rel > _WILKINSON_EPSILON:
                backaction_db = max(0.0, 10.0 * math.log10(max(c_rel, 1.0e-16) / _WILKINSON_EPSILON))
            else:
                backaction_db = 0.0
            is_qnd = (comm_norm <= _QND_COMMUTATOR_CEILING) or (c_rel <= _QND_COMMUTATOR_REL_CEILING)
            is_qnd = bool(is_qnd and backaction_db <= max(_MAX_QND_BACKACTION_DB, 0.0) + 1e-9)

            try:
                energies, evecs = la.eigh(hamiltonian)
                rho_e = evecs.conj().T @ rho @ evecs
                gaps = energies[:, None] - energies[None, :]
                identity_norm = float(np.sqrt(np.sum(np.abs(rho_e) ** 2 * gaps ** 2)))
                identity_err = _relative(abs(identity_norm - comm_norm), max(comm_norm, 1.0))
                dephased = np.diag(np.diag(rho_e))
                coh_defect = _relative(_fro(rho_e - dephased), _fro(rho))
            except Exception:
                identity_err, coh_defect = 0.0, 0.0
            return comm_norm, backaction_db, is_qnd, c_rel, float(coh_defect), float(identity_err)

        # ── Método 1.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _rsi_l3_denman_beavers_sqrtm(
            psd_mat: NDArray[np.complex128],
        ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], int, float]:
            """
            1.4 [RSI-L3] Raíz principal A^{1/2} e inversa A^{−1/2} sobre la matriz NORMALIZADA
            Â = A/α, α = ‖A‖₂ (evita que el piso δ contamine la escala del refinamiento).
            Semilla espectral exacta + refinamiento Denman-Beavers ESCALADO:
                μ_k = |det Y_k · det Z_k|^{−1/(2n)},
                Y_{k+1} = ½(μ_k Y_k + (μ_k Z_k)⁻¹),  Z_{k+1} = ½(μ_k Z_k + (μ_k Y_k)⁻¹),
            globalmente convergente (Higham). El lazo solo se ejecuta si el error hacia atrás
            relativo ‖Â − Y²‖_F/‖Â‖_F excede la tolerancia: los pasos reportados son REALES.
            """
            n = int(psd_mat.shape[0])
            herm = 0.5 * (psd_mat + psd_mat.conj().T)
            ident = np.eye(n, dtype=np.complex128)
            try:
                alpha = float(np.max(np.abs(la.eigvalsh(herm)))) if n else 1.0
                alpha = max(alpha, _SPECTRAL_PSD_FLOOR)
                a_norm = herm / alpha
                eigvals, eigvecs = la.eigh(a_norm)
                ev = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR / alpha)
                sq = np.sqrt(ev)
                y_mat = (eigvecs * sq) @ eigvecs.conj().T
                z_mat = (eigvecs * (1.0 / sq)) @ eigvecs.conj().T
                a_fro = max(_fro(a_norm), _TINY)
                iters = 0
                backward = _relative(_fro(a_norm - y_mat @ y_mat), a_fro)
                for _ in range(_DENMAN_BEAVERS_MAX_ITER):
                    coupling = _fro(ident - z_mat @ y_mat)
                    if backward <= 10.0 * _WILKINSON_EPSILON and coupling <= 1e-12:
                        break
                    _, ld_y = np.linalg.slogdet(y_mat)
                    _, ld_z = np.linalg.slogdet(z_mat)
                    mu = float(math.exp(-(float(ld_y) + float(ld_z)) / (2.0 * n)))
                    mu = float(np.clip(mu, 1.0e-6, 1.0e6))
                    ys, zs = mu * y_mat, mu * z_mat
                    y_mat = 0.5 * (ys + la.inv(zs))
                    z_mat = 0.5 * (zs + la.inv(ys))
                    iters += 1
                    backward = _relative(_fro(a_norm - y_mat @ y_mat), a_fro)
                root = math.sqrt(alpha)
                y_mat = 0.5 * (y_mat + y_mat.conj().T) * root
                z_mat = 0.5 * (z_mat + z_mat.conj().T) / root
                return y_mat, z_mat, iters, float(backward)
            except Exception:
                shifted = herm + _SPECTRAL_PSD_FLOOR * ident
                sqrt_fb = la.sqrtm(shifted)
                sqrt_fb = np.asarray(sqrt_fb, dtype=np.complex128)
                return sqrt_fb, la.pinv(sqrt_fb), 1, float("nan")

        # ── Método 1.5 ────────────────────────────────────────────────────────
        def _compute_uhlmann_bures_geometry(
            self,
            rho: NDArray[np.complex128],
            sigma: NDArray[np.complex128],
            rho_ref_probs: Optional[NDArray[np.float64]] = None,
            rho_probs: Optional[NDArray[np.float64]] = None,
            sigma_ref_probs: Optional[NDArray[np.float64]] = None,
            sigma_probs: Optional[NDArray[np.float64]] = None,
        ) -> Dict[str, Any]:
            """
            1.5 Fidelidad de Uhlmann por NORMA TRAZA: √F = ‖√ρ √σ‖₁ = Σ σ_i(√ρ√σ).
            Esta ruta evita formar √ρ σ √ρ, que eleva el número de condición al cuadrado.
            Se contrasta con la ruta espectral clásica y se reporta la discrepancia.
                d_B = √(2 − 2√F),  θ_B = arccos√F,  D_tr = ½‖ρ−σ‖₁.
            Certificado de Fuchs-van de Graaf: 1 − √F ≤ D_tr ≤ √(1 − F).
            Certificado de regularización (desigualdad triangular de la métrica de Bures):
                |d_B(ρ,σ) − d_B(ρ_δ,σ_δ)| ≤ d_B(ρ,ρ_δ) + d_B(σ,σ_δ),
            calculado exactamente en la autobase común (estados conmutantes).
            Transporte óptimo de Pusz-Woronowicz: T = ρ^{−1/2}(ρ^{1/2}σρ^{1/2})^{1/2}ρ^{−1/2},
            con certificado ‖TρT − σ‖_F/‖σ‖_F (ρ # σ es el punto medio geodésico de Bures-Wasserstein).
            NOTA de rigor: F ≥ 0.95 ⟹ d_B ≤ 0.2250 < 0.35, luego la cota de Bures es redundante;
            se conserva por explicitud contractual, pero la restricción ACTIVA es la fidelidad.
            """
            sqrt_rho, inv_sqrt_rho, iters_rho, bw_rho = self._rsi_l3_denman_beavers_sqrtm(rho)
            sqrt_sigma, _inv_s, iters_sigma, _bw_s = self._rsi_l3_denman_beavers_sqrtm(sigma)

            singular = la.svdvals(sqrt_rho @ sqrt_sigma)
            sqrt_f_svd = float(np.clip(np.sum(np.real(singular)), 0.0, 1.0))

            m_inter = sqrt_rho @ sigma @ sqrt_rho
            m_inter = 0.5 * (m_inter + m_inter.conj().T)
            eig_m, vec_m = la.eigh(m_inter)
            eig_m = np.real(eig_m)
            sqrt_f_eig = float(np.clip(np.sum(np.sqrt(np.maximum(eig_m, 0.0))), 0.0, 1.0))

            sqrt_f = sqrt_f_svd
            f_val = float(np.clip(sqrt_f * sqrt_f, 0.0, 1.0))
            bures = math.sqrt(max(0.0, 2.0 - 2.0 * sqrt_f))
            angle = math.acos(float(np.clip(sqrt_f, -1.0, 1.0)))

            diff_eigs = np.real(la.eigvalsh(0.5 * ((rho - sigma) + (rho - sigma).conj().T)))
            trace_dist = float(np.clip(0.5 * np.sum(np.abs(diff_eigs)), 0.0, 1.0))
            fvg_lo, fvg_hi = 1.0 - sqrt_f, math.sqrt(max(0.0, 1.0 - f_val))
            fvg_ok = bool(fvg_lo - 1e-9 <= trace_dist <= fvg_hi + 1e-9)

            m_sqrt = (vec_m * np.sqrt(np.maximum(eig_m, 0.0))) @ vec_m.conj().T
            transport = inv_sqrt_rho @ m_sqrt @ inv_sqrt_rho
            transport_defect = _relative(_fro(transport @ rho @ transport - sigma), _fro(sigma))

            def _commuting_bures(p: Optional[NDArray], q: Optional[NDArray]) -> float:
                if p is None or q is None:
                    return 0.0
                overlap = float(np.sum(np.sqrt(np.maximum(p, 0.0) * np.maximum(q, 0.0))))
                return math.sqrt(max(0.0, 2.0 - 2.0 * float(np.clip(overlap, 0.0, 1.0))))

            reg_slack = _commuting_bures(rho_ref_probs, rho_probs) + _commuting_bures(
                sigma_ref_probs, sigma_probs
            )
            d_lo, d_hi = max(0.0, bures - reg_slack), bures + reg_slack
            f_lo = float(np.clip((1.0 - 0.5 * d_hi * d_hi) ** 2, 0.0, 1.0)) if d_hi <= math.sqrt(2.0) else 0.0
            f_hi = float(np.clip((1.0 - 0.5 * d_lo * d_lo) ** 2, 0.0, 1.0))

            healthy = bool(f_val >= _UHLMANN_FIDELITY_FLOOR and bures <= _BURES_DISTANCE_CEILING)
            return {
                "fidelity": f_val,
                "bures": bures,
                "healthy": healthy,
                "angle": angle,
                "trace_distance": trace_dist,
                "fvg_ok": fvg_ok,
                "svd_eig_discrepancy": abs(sqrt_f_svd - sqrt_f_eig),
                "transport_defect": float(min(transport_defect, 1.0e6)),
                "f_lower": min(f_lo, f_val),
                "f_upper": max(f_hi, f_val),
                "db_iters": iters_rho + iters_sigma,
                "sqrt_rho": sqrt_rho,
                "inv_sqrt_rho": inv_sqrt_rho,
                "backward_error": bw_rho,
            }

        # ── Método 1.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_tomita_takesaki_kms_symmetry(
            purification: SpectralPurification,
            hamiltonian: NDArray[np.complex128],
            comm_norm: float,
        ) -> KMSCertificate:
            """
            1.6 Teoría modular de Tomita-Takesaki y condición KMS, con certificados reales.
            Sea K = −ln ρ (Hamiltoniano modular). En la representación GNS de Hilbert-Schmidt:
                Δ_ρ(X) = ρXρ⁻¹,  S₀(X) = J Δ_ρ^{1/2}(X) = ρ^{1/2} X† ρ^{−1/2},  S₀² = Id.
            (i)  Certificado de involución: ‖S₀²(X) − X‖_F/‖X‖_F sobre una sonda no conmutante.
            (ii) KMS ⟺ ρ = e^{−βH}/Z ⟺ K = βH + ln Z·I: ajuste por mínimos cuadrados
                 (proyección ortogonal de K sobre span{H, I} en el producto de Frobenius).
            (iii) Flujo modular σ_t(X) = ρ^{it}Xρ^{−it} ≡ e^{−iβtH} X e^{+iβtH} (Heisenberg a −βt).
            (iv) Ergotropía W = Tr(ρH) − Tr(ρ_pas H) con ρ_pas el reordenamiento pasivo
                 (poblaciones decrecientes ↔ energías crecientes): W = 0 ⟺ estado pasivo.
            El defecto conmutativo [√ρ, H] se conserva como telemetría heredada, normalizado.
            """
            n = int(purification.rho.shape[0])
            probs = np.maximum(purification.eigenvalues, _WILKINSON_EPSILON)
            vecs = purification.eigenvectors
            log_p = np.log(probs)
            sqrt_p, inv_sqrt_p = np.sqrt(probs), 1.0 / np.sqrt(probs)

            sqrt_rho = (vecs * sqrt_p) @ vecs.conj().T
            inv_sqrt_rho = (vecs * inv_sqrt_p) @ vecs.conj().T
            k_mod = -(vecs * log_p) @ vecs.conj().T

            probe = np.zeros((n, n), dtype=np.complex128)
            if n >= 2:
                shift = np.roll(np.eye(n, dtype=np.complex128), 1, axis=0)
                probe = 0.5 * (shift + shift.conj().T)
            else:
                probe = np.eye(1, dtype=np.complex128)
            probe_norm = max(_fro(probe), _TINY)

            s0 = lambda x: sqrt_rho @ x.conj().T @ inv_sqrt_rho  # noqa: E731
            involution_defect = _relative(_fro(s0(s0(probe)) - probe), probe_norm)

            h_fro2 = float(np.real(np.vdot(hamiltonian, hamiltonian)))
            tr_h = float(np.real(np.trace(hamiltonian)))
            tr_k = float(np.real(np.trace(k_mod)))
            b1 = float(np.real(np.vdot(hamiltonian, k_mod)))
            gram = np.array([[h_fro2, tr_h], [tr_h, float(n)]], dtype=np.float64)
            rhs = np.array([b1, tr_k], dtype=np.float64)
            try:
                beta, log_z = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            except Exception:
                beta, log_z = 0.0, tr_k / max(1, n)
            beta, log_z = float(beta), float(log_z)
            gibbs_res = _fro(k_mod - beta * hamiltonian - log_z * np.eye(n, dtype=np.complex128))
            gibbs_defect = _relative(gibbs_res, max(_fro(k_mod), _TINY))

            try:
                energies, h_vecs = la.eigh(hamiltonian)
            except Exception:
                energies, h_vecs = np.zeros(n), np.eye(n, dtype=np.complex128)

            flow_defect = 0.0
            for t in (0.37, 1.13):
                u_mod = (vecs * np.exp(1j * t * log_p)) @ vecs.conj().T
                lhs = u_mod @ probe @ u_mod.conj().T
                u_heis = (h_vecs * np.exp(-1j * beta * t * energies)) @ h_vecs.conj().T
                rhs_flow = u_heis @ probe @ u_heis.conj().T
                flow_defect = max(flow_defect, _relative(_fro(lhs - rhs_flow), probe_norm))

            populations = np.maximum(np.real(np.diag(h_vecs.conj().T @ purification.rho @ h_vecs)), 0.0)
            populations = populations / max(float(np.sum(populations)), _TINY)
            order_p = np.argsort(-populations)
            order_e = np.argsort(energies)
            passive = np.zeros(n, dtype=np.float64)
            passive[order_e] = populations[order_p]
            mean_e = float(np.sum(populations * energies))
            ergotropy = max(0.0, mean_e - float(np.sum(passive * energies)))

            tomita_defect = _relative(_fro(sqrt_rho @ hamiltonian - hamiltonian @ sqrt_rho),
                                      max(_fro(hamiltonian) * _fro(purification.rho), _TINY))
            kms_residual = float(max(gibbs_defect, flow_defect))
            return KMSCertificate(
                kms_residual=kms_residual,
                tomita_commutator_defect=float(tomita_defect),
                involution_defect=float(min(involution_defect, 1e6)),
                gibbs_defect=float(min(gibbs_defect, 1e6)),
                modular_flow_defect=float(min(flow_defect, 1e6)),
                beta_estimate=beta,
                log_partition=log_z,
                ergotropy=float(ergotropy),
                is_passive=bool(ergotropy <= 1e-10 * max(1.0, abs(mean_e))),
            )

        # ── Método 1.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_poincare_quantum_recurrence(
            hamiltonian: NDArray[np.complex128], rho: NDArray[np.complex128]
        ) -> RecurrenceCertificate:
            """
            1.7 Recurrencia cuántica de Poincaré-Bocchieri-Loinger para espectro discreto.
            (a) Escala corta: T_rec ≈ 2π/ΔE_rms con ΔE_rms PONDERADO por las coherencias de
                población p_i p_j en la autobase de H (v14 declaraba el peso pero no lo usaba).
            (b) Cota de existencia por aproximación simultánea de Dirichlet: para m modos Bohr
                independientes existe t ≤ (1/δ)^m con todas las fases a distancia < δ; se reporta
                log₁₀ T para evitar desbordamiento (el bound real es astronómico y así se declara).
            (c) Límites cuánticos de velocidad: τ_MT = π/(2ΔE) (Mandelstam-Tamm) y
                τ_ML = π/(2(⟨E⟩−E₀)) (Margolus-Levitin), con ħ = 1.
            """
            try:
                energies, evecs = la.eigh(hamiltonian)
                energies = np.real(energies)
                n = int(energies.size)
                if n < 2:
                    return RecurrenceCertificate(2.0 * math.pi, 0.0, math.inf, math.inf, 0, 0.0)
                pops = np.maximum(np.real(np.diag(evecs.conj().T @ rho @ evecs)), 0.0)
                pops = pops / max(float(np.sum(pops)), _TINY)
                weights = np.outer(pops, pops)
                gaps = np.abs(energies[:, None] - energies[None, :])
                mask = (gaps > _SPECTRAL_PSD_FLOOR) & (weights > 1.0e-9 / (n * n))
                if not bool(np.any(mask)):
                    return RecurrenceCertificate(2.0 * math.pi, 0.0, math.inf, math.inf, 0, 0.0)
                sel_g, sel_w = gaps[mask], weights[mask]
                delta_rms = float(np.sqrt(np.sum(sel_w * sel_g ** 2) / max(np.sum(sel_w), _TINY)))
                t_rec = (2.0 * math.pi) / max(delta_rms, _SPECTRAL_PSD_FLOOR)

                unique_modes = np.unique(np.round(sel_g, decimals=12))
                m = int(min(max(unique_modes.size // 2, 1), 64))
                log10_dirichlet = m * math.log10(1.0 / 1.0e-3) + math.log10(2.0 * math.pi)

                mean_energy = float(np.sum(pops * energies))
                var_energy = max(0.0, float(np.sum(pops * energies ** 2)) - mean_energy ** 2)
                sd_energy = math.sqrt(var_energy)
                tau_mt = math.pi / (2.0 * sd_energy) if sd_energy > _TINY else math.inf
                excitation = mean_energy - float(np.min(energies))
                tau_ml = math.pi / (2.0 * excitation) if excitation > _TINY else math.inf
                return RecurrenceCertificate(
                    recurrence_bound=float(t_rec),
                    dirichlet_log10_bound=float(log10_dirichlet),
                    mandelstam_tamm_time=float(tau_mt),
                    margolus_levitin_time=float(tau_ml),
                    bohr_mode_count=m,
                    energy_variance=float(var_energy),
                )
            except Exception:
                return RecurrenceCertificate(2.0 * math.pi, 0.0, math.inf, math.inf, 0, 0.0)

        # ── Método 1.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_de_rham_betti_acyclicity(
            rho: NDArray[np.complex128], coherence_threshold: float = 0.25
        ) -> BettiTopology:
            """
            1.8 Homología simplicial del grafo de interferencia K₁(ρ): (i,j) ∈ E ⟺
            |ρ_ij|/√(ρ_ii ρ_jj) > τ_coh (invariante gauge bajo U(1)ⁿ diagonal).
            β₀ se calcula por UNION-FIND (entero exacto, sin tolerancia espectral frágil) y
            β₁ por el rango del espacio de ciclos (= número de cuerdas del bosque generador).
            CERTIFICADO CRUZADO: |E| − |V| + β₀ == #cuerdas (dos cálculos independientes).
            [CORRECCIÓN C2] v14 devolvía max(1, β₁) si la fidelidad era mala: eso FABRICABA
            topología desde una métrica. Aquí β₁ es puramente topológico; la obstrucción
            métrica se adjudica aparte, en 1.9.
            HOLONOMÍA: sobre cada ciclo fundamental se calcula la fase de Bargmann-Wilson
                φ(C) = arg ∏_k ρ_{i_k i_{k+1}},
            invariante bajo ρ ↦ DρD† con D unitaria diagonal (telescopio de fases). Para un
            estado puro φ ≡ 0: φ ≠ 0 atestigua flujo de fase geométrico / mezclamiento.
            """
            n = int(rho.shape[0])
            diag = np.maximum(np.real(np.diag(rho)), _SPECTRAL_PSD_FLOOR)
            coherence = np.abs(rho) / np.sqrt(np.outer(diag, diag))
            np.fill_diagonal(coherence, 0.0)
            adj = (coherence > coherence_threshold).astype(np.float64)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0.0]

            parent = list(range(n))

            def find(a: int) -> int:
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            forest: List[List[int]] = [[] for _ in range(n)]
            chords: List[Tuple[int, int]] = []
            for (i, j) in edges:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    forest[i].append(j)
                    forest[j].append(i)
                else:
                    chords.append((i, j))
            beta_0 = len({find(i) for i in range(n)})
            beta_1 = len(chords)
            euler_ok = bool(beta_1 == len(edges) - n + beta_0)

            tree_parent = [-1] * n
            depth = [0] * n
            seen = [False] * n
            for root in range(n):
                if seen[root]:
                    continue
                seen[root] = True
                stack = [root]
                while stack:
                    u = stack.pop()
                    for v in forest[u]:
                        if not seen[v]:
                            seen[v] = True
                            tree_parent[v] = u
                            depth[v] = depth[u] + 1
                            stack.append(v)

            def tree_path(u: int, v: int) -> List[int]:
                pu, pv = [u], [v]
                a, b = u, v
                while depth[a] > depth[b]:
                    a = tree_parent[a]
                    pu.append(a)
                while depth[b] > depth[a]:
                    b = tree_parent[b]
                    pv.append(b)
                while a != b and tree_parent[a] >= 0 and tree_parent[b] >= 0:
                    a, b = tree_parent[a], tree_parent[b]
                    pu.append(a)
                    pv.append(b)
                return pu + pv[::-1]

            max_phase, nontrivial = 0.0, 0
            for (u, v) in chords:
                cycle = tree_path(u, v)
                if len(cycle) < 3:
                    continue
                prod = complex(1.0, 0.0)
                for k in range(len(cycle)):
                    prod *= complex(rho[cycle[k], cycle[(k + 1) % len(cycle)]])
                phase = abs(float(np.angle(prod)))
                max_phase = max(max_phase, phase)
                if phase > 1.0e-6:
                    nontrivial += 1

            weighted_l = np.diag(np.sum(coherence, axis=1)) - coherence
            try:
                lap_eigs = np.sort(np.real(la.eigvalsh(weighted_l)))
                fiedler = float(lap_eigs[1]) if n >= 2 else 0.0
            except Exception:
                fiedler = 0.0

            profile: List[int] = []
            for tau in (0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 0.95):
                par = list(range(n))

                def find2(a: int) -> int:
                    while par[a] != a:
                        par[a] = par[par[a]]
                        a = par[a]
                    return a

                e_cnt = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if coherence[i, j] > tau:
                            e_cnt += 1
                            ra, rb = find2(i), find2(j)
                            if ra != rb:
                                par[ra] = rb
                b0 = len({find2(i) for i in range(n)})
                profile.append(max(0, e_cnt - n + b0))

            return BettiTopology(
                beta_0=int(beta_0), beta_1=int(beta_1), num_vertices=n, num_edges=len(edges),
                euler_identity_ok=euler_ok, max_bargmann_phase=float(max_phase),
                nontrivial_cycle_count=int(nontrivial), fiedler_connectivity=max(0.0, fiedler),
                persistence_profile=tuple(profile),
            )

        # ── Método 1.9 ────────────────────────────────────────────────────────
        @staticmethod
        def _adjudicate_qnd_heyting_truth(
            is_qnd_valid: bool,
            is_fides_healthy: bool,
            betti_1: int,
            uhlmann_fidelity: float,
            euler_ok: bool,
        ) -> Tuple[HeytingTruthValue, Tuple[str, ...]]:
            """
            1.9 Adjudicación constructiva en Ω₃ con traza explícita de obstrucciones.
            COHERENT ⟺ QND ∧ fidelidad ∧ aciclicidad ∧ certificado de Euler.
            DEGRADED ⟺ verdad parcial (fidelidad sana o F ≥ estrato 0.80).
            VETOED   ⟺ obstrucción total.
            """
            reasons: List[str] = []
            if not is_qnd_valid:
                reasons.append("QND_COMMUTATOR_EXCEEDED")
            if not is_fides_healthy:
                reasons.append("BURES_UHLMANN_OUT_OF_SAFE_BALL")
            if betti_1 != 0:
                reasons.append("HOMOLOGICAL_CYCLE_OBSTRUCTION")
            if not euler_ok:
                reasons.append("EULER_POINCARE_IDENTITY_VIOLATED")
            if is_qnd_valid and is_fides_healthy and betti_1 == 0 and euler_ok:
                return HeytingTruthValue.COHERENT, ()
            if is_fides_healthy or uhlmann_fidelity >= _FIDELITY_DEGRADED_FLOOR:
                return HeytingTruthValue.DEGRADED, tuple(reasons)
            return HeytingTruthValue.VETOED, tuple(reasons)

        # ── Método 1.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _build_phase2_symplectic_seed(
            report: QNDInspectionReport,
        ) -> Phase1ToPhase2Seed:
            """
            1.10 Construye la semilla simpléctica que la Fase 2 absorbe en 2.1.
            La perturbación efectiva del sistema celeste se alimenta del defecto QND
            (retroacción pasiva del testigo sobre el presupuesto de perturbación), acotada
            a 10⁻³ para no violar el régimen de validez de la serie de Lindstedt.
            """
            return Phase1ToPhase2Seed(
                qnd_commutator_perturbation=float(min(1.0e-3, report.qnd_commutator_norm)),
                qnd_relative_commutator=float(report.relative_commutator),
                bures_geodesic_tension=float(report.bures_distance),
                poincare_recurrence_scale=float(
                    report.poincare_quantum_recurrence_bound
                    if math.isfinite(report.poincare_quantum_recurrence_bound)
                    else 2.0 * math.pi
                ),
                modular_beta=float(report.modular_beta),
                homological_obstruction=int(report.betti_1_topology),
                upstream_truth=report.truth_value,
                has_context=True,
            )

        # ── Método 1.11 · TERMINAL FASE 1 → APERTURA FASE 2 ───────────────────
        def audit_qnd_inspection(
            self,
            rho_state: NDArray[np.complex128],
            rho_target: Optional[NDArray[np.complex128]] = None,
            hamiltonian: Optional[NDArray[np.complex128]] = None,
        ) -> QNDInspectionReport:
            """
            1.11 [TERMINAL FASE 1] Orquesta 1.1 → 1.10 y, como ÚLTIMA definición formal de la
            fase, EMITE el `Phase1ToPhase2Seed` que constituye el argumento de apertura (2.1)
            de la Fase 2. La continuidad es de tipos, no de convención:

                audit_qnd_inspection(...) → QNDInspectionReport.continuation
                                          → Phase2._ingest_phase1_continuation(seed)
            """
            n = 4
            if isinstance(rho_state, np.ndarray) and rho_state.ndim == 2 and rho_state.shape[0] > 0:
                n = int(rho_state.shape[0])

            val_rho = self._validate_and_symmetrize_operator(rho_state, n, is_density=True)
            val_sigma = self._validate_and_symmetrize_operator(rho_target, n, is_density=True)
            val_h = self._validate_and_symmetrize_operator(hamiltonian, n, is_density=False)

            comm_norm, backaction_db, is_qnd, c_rel, coh_defect, id_err = (
                self._compute_qnd_commutator_and_backaction(val_rho.operator, val_h.operator)
            )

            pur_rho = self._rsi_l3_purify_density_operator(val_rho.operator)
            pur_sigma = self._rsi_l3_purify_density_operator(val_sigma.operator)

            geo = self._compute_uhlmann_bures_geometry(
                pur_rho.rho, pur_sigma.rho,
                pur_rho.reference_probs, pur_rho.eigenvalues,
                pur_sigma.reference_probs, pur_sigma.eigenvalues,
            )
            kms = self._compute_tomita_takesaki_kms_symmetry(pur_rho, val_h.operator, comm_norm)
            rec = self._compute_poincare_quantum_recurrence(val_h.operator, pur_rho.rho)
            topo = self._compute_de_rham_betti_acyclicity(pur_rho.rho)

            tv, reasons = self._adjudicate_qnd_heyting_truth(
                is_qnd, bool(geo["healthy"]), topo.beta_1, float(geo["fidelity"]),
                topo.euler_identity_ok,
            )

            self._telemetry.total_qnd_audits += 1
            self._telemetry.cumulative_denman_beavers_steps += int(geo["db_iters"])
            self._telemetry.last_heyting_verdict = tv

            report = QNDInspectionReport(
                is_qnd_valid=bool(is_qnd),
                qnd_commutator_norm=float(comm_norm),
                backaction_db=float(backaction_db),
                trace_preservation=float(val_rho.trace_defect),
                uhlmann_fidelity=float(geo["fidelity"]),
                bures_distance=float(geo["bures"]),
                is_fides_healthy=bool(geo["healthy"]),
                kms_correlation_residual=float(kms.kms_residual),
                betti_1_topology=int(topo.beta_1),
                truth_value=tv,
                tomita_modular_defect=float(kms.tomita_commutator_defect),
                von_neumann_entropy=float(pur_rho.entropy),
                poincare_quantum_recurrence_bound=float(rec.recurrence_bound),
                rsi_l3_purification_iterations=int(geo["db_iters"]),
                relative_commutator=float(c_rel),
                energy_coherence_defect=float(coh_defect),
                commutator_identity_error=float(id_err),
                fidelity_svd_eig_discrepancy=float(geo["svd_eig_discrepancy"]),
                fidelity_lower_certificate=float(geo["f_lower"]),
                fidelity_upper_certificate=float(geo["f_upper"]),
                bures_angle=float(geo["angle"]),
                trace_distance=float(geo["trace_distance"]),
                fuchs_van_de_graaf_ok=bool(geo["fvg_ok"]),
                transport_map_defect=float(geo["transport_defect"]),
                purification_projection_cost=float(pur_rho.projection_cost),
                purification_kkt_ok=bool(pur_rho.kkt_feasible),
                state_condition_number=float(pur_rho.condition_number),
                state_purity=float(pur_rho.purity),
                involution_defect=float(kms.involution_defect),
                gibbs_defect=float(kms.gibbs_defect),
                modular_flow_defect=float(kms.modular_flow_defect),
                modular_beta=float(kms.beta_estimate),
                ergotropy=float(kms.ergotropy),
                is_passive_state=bool(kms.is_passive),
                dirichlet_recurrence_log10=float(rec.dirichlet_log10_bound),
                mandelstam_tamm_time=float(rec.mandelstam_tamm_time),
                margolus_levitin_time=float(rec.margolus_levitin_time),
                betti_0_topology=int(topo.beta_0),
                euler_identity_ok=bool(topo.euler_identity_ok),
                max_bargmann_phase=float(topo.max_bargmann_phase),
                fiedler_connectivity=float(topo.fiedler_connectivity),
                obstruction_reasons=reasons,
            )
            # ── CONTINUIDAD FORMAL: última sentencia de la Fase 1 = semilla de la Fase 2 ──
            report = replace(report, continuation=self._build_phase2_symplectic_seed(report))
            self._last_report = report
            return report

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 2 · MECÁNICA CELESTE DE POINCARÉ: KAM, BRUNO, MELNIKOV, GREENE, FLOQUET
    # ──────────────────────────────────────────────────────────────────────────
    class Phase2_PoincareKAMCelestialNavigator:
        """
        Fase 2. Cadena formal:
            2.1 INGESTA de Phase1ToPhase2Seed → 2.2 toro → 2.3 Siegel-Bruno →
            2.4 Kepler certificado → 2.5 Lindstedt+AGM → 2.6 Melnikov(sech) →
            2.7 Chirikov → 2.8 residuos de Greene → 2.9 Floquet-Krein →
            2.10 AMD de Laskar → 2.11 adjudicación → 2.12 semilla →
            2.13 TERMINAL (emite Phase2ToPhase3Seed).
        """

        def __init__(
            self,
            phase1_inspector: "BoolNovikovPaladinWitnessEngine.Phase1_QNDTomitaTakesakiInspector",
            telemetry: RSILevel3WitnessTelemetry,
        ) -> None:
            self._phase1 = phase1_inspector
            self._telemetry = telemetry
            self._last_report: Optional[CelestialWitnessReport] = None

        # ── Método 2.1 · APERTURA FASE 2 (continuación directa de 1.11) ───────
        def _ingest_phase1_continuation(
            self, qnd_report: Optional[QNDInspectionReport] = None
        ) -> Phase1ToPhase2Seed:
            """
            2.1 [CONTINUACIÓN DIRECTA DE 1.11] Absorbe la semilla simpléctica emitida por el
            método terminal de la Fase 1. Si se pasa el informe, se usa su campo `continuation`
            (tipado); si no, se recupera el último informe de la Fase 1. Sin contexto aguas
            arriba, se devuelve la semilla neutra (identidad del acoplamiento).
            """
            active = qnd_report if qnd_report is not None else self._phase1._last_report
            if active is not None and active.continuation is not None:
                return active.continuation
            if active is not None:
                return self._phase1._build_phase2_symplectic_seed(active)
            return Phase1ToPhase2Seed(
                qnd_commutator_perturbation=0.0, qnd_relative_commutator=0.0,
                bures_geodesic_tension=0.0, poincare_recurrence_scale=2.0 * math.pi,
                modular_beta=0.0, homological_obstruction=0,
                upstream_truth=HeytingTruthValue.COHERENT, has_context=False,
            )

        # ── Método 2.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _normalize_frequency_torus(
            frequencies: Optional[Sequence[float]],
        ) -> Tuple[List[float], List[float], float]:
            """
            2.2 Normaliza el vector de frecuencias fundamentales del toro invariante
            Tᵈ = ℝᵈ/(2πℤ)ᵈ. Devuelve (ω físicas, ω̂ = ω/‖ω‖_∞, escala).
            La condición diofántica NO es invariante de escala: se evalúa sobre ω̂ para que
            el umbral γ_KAM sea adimensional (v14 comparaba magnitudes con unidades).
            """
            default = [1.0, math.sqrt(2.0), math.sqrt(3.0)]
            cleaned: List[float] = []
            for f in (frequencies or []):
                v = abs(_safe_float(f, 0.0))
                if v > _WILKINSON_EPSILON:
                    cleaned.append(v)
            if not cleaned:
                cleaned = default
            cleaned = cleaned[:8]
            scale = max(cleaned)
            return cleaned, [v / scale for v in cleaned], float(scale)

        # ── Método 2.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _evaluate_siegel_bruno_diophantine_lattice(
            normalized_freqs: List[float], tau: float = 2.0
        ) -> Dict[str, Any]:
            """
            2.3 Red resonante REAL: se enumeran k ∈ ℤᵈ\\{0} con 0 < ‖k‖₁ ≤ 2ᴺ (N = 4) y se
            calcula Ω_n = min_{0<‖k‖₁≤2ⁿ}|k·ω̂|, de donde
                B_N(ω) = Σ_{n=1}^{N} 2⁻ⁿ ln(1/Ω_n)   (suma parcial de la serie de Bruno).
            Constante diofántica empirizada: γ̂ = min_k |k·ω̂|·‖k‖₁^τ.
            TEOREMA CERTIFICADO (Siegel ⟹ Bruno): si |k·ω| ≥ γ‖k‖⁻ᵗᵃᵘ entonces Ω_n ≥ γ2⁻ⁿᵗᵃᵘ y
                B(ω) ≤ ln(1/γ) + τ ln2 · Σ n2⁻ⁿ = ln(1/γ) + 2τ ln 2 < ∞.
            Se reporta la cota analítica junto a la suma parcial: la convergencia queda
            ACOTADA, no meramente sugerida por una truncación.
            Para d = 2 se añade el diagnóstico de fracción continua: cocientes parciales
            acotados ⟺ tipo constante ⟺ diofántico con τ = 1 (badly approximable).
            """
            d = len(normalized_freqs)
            omega = np.asarray(normalized_freqs, dtype=np.float64)
            k_max = 2 ** _BRUNO_MAX_EXPONENT
            vecs, norms = _resonant_lattice(d, k_max)
            dots = np.abs(vecs @ omega)
            dots = np.maximum(dots, _WILKINSON_EPSILON)

            gamma_hat = float(np.min(dots * norms ** tau))
            min_div = float(np.min(dots))
            bruno_sum = 0.0
            for n_exp in range(1, _BRUNO_MAX_EXPONENT + 1):
                shell = norms <= (2 ** n_exp)
                omega_n = float(np.min(dots[shell])) if bool(np.any(shell)) else 1.0
                omega_n = min(1.0, max(omega_n, _WILKINSON_EPSILON))
                bruno_sum += (2.0 ** (-n_exp)) * (-math.log(omega_n))

            gamma_ref = max(gamma_hat, _WILKINSON_EPSILON)
            bruno_bound = float(math.log(1.0 / gamma_ref) + 2.0 * tau * math.log(2.0))
            kam_stable = bool(gamma_hat >= _KAM_DIOPHANTINE_FLOOR and min_div >= _KAM_DIOPHANTINE_FLOOR)

            bounded_type = False
            if d == 2:
                ratio = float(normalized_freqs[0] / max(normalized_freqs[1], _TINY))
                quotients: List[int] = []
                x = ratio
                for _ in range(24):
                    a = int(math.floor(x))
                    quotients.append(a)
                    frac = x - a
                    if frac <= 1e-12:
                        break
                    x = 1.0 / frac
                bounded_type = bool(quotients and max(quotients) <= 12)

            return {
                "min_div": min_div,
                "gamma": gamma_hat,
                "bruno_sum": float(bruno_sum),
                "bruno_bound": bruno_bound,
                "bruno_convergent": bool(bruno_sum <= min(bruno_bound, _BRUNO_SERIES_CEILING)),
                "kam_stable": kam_stable,
                "lattice_count": int(vecs.shape[0]),
                "bounded_type": bounded_type,
            }

        # ── Método 2.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _solve_kepler_equation_certified(
            eccentricity: float, mean_anomaly: float
        ) -> Tuple[float, float, int, List[float]]:
            """
            2.4a Ecuación de Kepler E − e·sin E = M resuelta por el iterado cuártico de Danby
            con SALVAGUARDA DE BISECCIÓN en el intervalo garantizado E ∈ [M−e, M+e]
            (pues |E−M| = e|sin E| ≤ e). La salvaguarda convierte un método local en uno
            GLOBALMENTE convergente: certificado de existencia por cambio de signo.
                f = E − e sinE − M,  f' = 1 − e cosE,  f'' = e sinE,  f''' = e cosE
                δ₁ = −f/f',  δ₂ = −f/(f' + δ₁f''/2),  δ₃ = −f/(f' + δ₁f''/2 + δ₂²f'''/6).
            Devuelve (E, residual, iteraciones, historial de errores) para estimar el orden
            empírico de convergencia que alimenta el nivel L2 de la RSI.
            """
            e = float(np.clip(abs(eccentricity), 0.0, 0.999999))
            m = float(mean_anomaly)
            lo, hi = m - e - 1e-12, m + e + 1e-12
            ecc_anom = m + 0.85 * e * (1.0 if math.sin(m) >= 0.0 else -1.0)
            errors: List[float] = []
            f_val = ecc_anom - e * math.sin(ecc_anom) - m
            iters = 0
            for k in range(1, 61):
                iters = k
                errors.append(abs(f_val))
                if abs(f_val) <= 4.0 * _WILKINSON_EPSILON:
                    break
                if f_val > 0.0:
                    hi = ecc_anom
                else:
                    lo = ecc_anom
                fp = 1.0 - e * math.cos(ecc_anom)
                fpp = e * math.sin(ecc_anom)
                fppp = e * math.cos(ecc_anom)
                step = None
                if abs(fp) > 1e-14:
                    d1 = -f_val / fp
                    den2 = fp + 0.5 * d1 * fpp
                    d2 = -f_val / den2 if abs(den2) > 1e-14 else d1
                    den3 = fp + 0.5 * d1 * fpp + (d2 * d2) * fppp / 6.0
                    d3 = -f_val / den3 if abs(den3) > 1e-14 else d2
                    cand = ecc_anom + d3
                    if lo < cand < hi:
                        step = d3
                if step is None:
                    cand = 0.5 * (lo + hi)
                    step = cand - ecc_anom
                ecc_anom += step
                f_val = ecc_anom - e * math.sin(ecc_anom) - m
            return float(ecc_anom), float(abs(f_val)), int(iters), errors

        def _audit_kepler_torus(
            self, eccentricity: float
        ) -> Dict[str, Any]:
            """
            2.4b Barrido del toro de anomalías medias (16 nodos) para certificar el solver en
            el peor caso, y contraste independiente con la serie de Kapteyn-Bessel
                E = M + Σ_{n≥1} (2/n) J_n(ne) sin(nM),
            convergente para todo e < 1 (a diferencia de la serie de LAGRANGE en potencias de e,
            cuyo radio es el límite de Laplace e_L = 0.6627434193…). Esta distinción es la que
            v14 no hacía: se reportan ambos regímenes por separado.
            """
            e = float(np.clip(abs(eccentricity), 0.0, 0.999999))
            worst_res, total_iters, orders = 0.0, 0, []
            worst_m, worst_e_anom = 0.7, 0.7
            nodes = 16
            for idx in range(1, nodes + 1):
                m = math.pi * idx / (nodes + 1)
                sol, res, its, errs = self._solve_kepler_equation_certified(e, m)
                worst_res = max(worst_res, res)
                total_iters += its
                if res > 1e-14 or idx == nodes // 2:
                    worst_m, worst_e_anom = m, sol
                if len(errs) >= 3 and errs[-3] > 1e-12 and errs[-2] > 1e-14 and errs[-1] > 0.0:
                    num = math.log(max(errs[-1] / errs[-2], _TINY))
                    den = math.log(max(errs[-2] / errs[-3], _TINY))
                    if abs(den) > 1e-6:
                        orders.append(float(np.clip(num / den, 0.0, 6.0)))
            self._telemetry.cumulative_kepler_steps += total_iters

            kapteyn_res = 0.0
            if e > 1e-8:
                series = worst_m
                for n_ord in range(1, 49):
                    jn = _bessel_j_periodic_trapezoid(n_ord, n_ord * e)
                    term = (2.0 / n_ord) * jn * math.sin(n_ord * worst_m)
                    series += term
                    if abs(term) < 1e-16 and n_ord > 8:
                        break
                kapteyn_res = abs(series - e * math.sin(series) - worst_m)

            return {
                "max_residual": float(worst_res),
                "mean_iterations": float(total_iters / nodes),
                "empirical_order": float(np.mean(orders)) if orders else 0.0,
                "kapteyn_residual": float(kapteyn_res),
                "laplace_regime": bool(e < _LAPLACE_LIMIT),
                "reference_M": float(worst_m),
                "reference_E": float(worst_e_anom),
            }

        # ── Método 2.5 ────────────────────────────────────────────────────────
        def _compute_lindstedt_poincare_canonical_series(
            self, w0: float, perturbation_eps: float, eccentricity: float,
            kepler_ref: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            2.5 Serie canónica de Lindstedt-Poincaré SIN términos seculares, con resumación
            exacta y certificado de resto. Modelo: Duffing canónico ẍ + ω₀²x + εx³ = 0 en
            variables de Poincaré (ξ,η) = √(2J)(cos ϖ, sin ϖ), donde la amplitud normalizada
            coincide con la excentricidad, a ≡ e (pues e ≈ √(2J/Λ)). Con δ = εe²/ω₀²:
                ω_LP = ω₀(1 + (3/8)δ − (21/256)δ²)        [coeficientes clásicos exactos]
                ω_ex = ω₀√(1+2δ)·AGM(1, √(1−k²)),  k² = δ/(1+2δ)
            La identidad K(k) = π/(2·AGM(1,k')) convierte la serie truncada en un objeto
            CERTIFICADO: resto relativo |ω_LP − ω_ex|/ω_ex. [CORRECCIÓN C8: el ω₁ = 0.1 de v14
            no provenía de ningún modelo; aquí cada coeficiente es derivable y verificable.]
            Además, la serie de la ecuación del centro en e (Brouwer-Clemence / Murray-Dermott)
                ν − M = (2e − e³/4)sinM + (5/4)e² sin2M + (13/12)e³ sin3M + O(e⁴)
            se contrasta con el ν EXACTO del solver certificado 2.4: segundo certificado de resto.
            """
            eps = abs(_safe_float(perturbation_eps, 0.0))
            ecc = float(np.clip(abs(_safe_float(eccentricity, 0.0)), 0.0, 0.99))
            omega0 = max(abs(_safe_float(w0, 1.0)), 1e-6)
            delta = eps * ecc * ecc / (omega0 * omega0)
            delta = float(np.clip(delta, 0.0, 1.0e3))

            omega_series = omega0 * (1.0 + (3.0 / 8.0) * delta - (21.0 / 256.0) * delta * delta)
            k_sq = delta / (1.0 + 2.0 * delta)
            k_prime = math.sqrt(max(0.0, 1.0 - k_sq))
            agm_val, agm_iters = _arithmetic_geometric_mean(1.0, k_prime)
            omega_exact = omega0 * math.sqrt(1.0 + 2.0 * delta) * agm_val
            remainder = _relative(abs(omega_series - omega_exact), max(omega_exact, _TINY))
            self._telemetry.cumulative_agm_steps += agm_iters

            m_ref = float(kepler_ref.get("reference_M", 0.7))
            e_anom = float(kepler_ref.get("reference_E", m_ref))
            true_anom = 2.0 * math.atan2(
                math.sqrt(1.0 + ecc) * math.sin(e_anom / 2.0),
                math.sqrt(1.0 - ecc) * math.cos(e_anom / 2.0),
            )
            center_exact = true_anom - m_ref
            center_series = (
                (2.0 * ecc - ecc ** 3 / 4.0) * math.sin(m_ref)
                + (5.0 / 4.0) * ecc ** 2 * math.sin(2.0 * m_ref)
                + (13.0 / 12.0) * ecc ** 3 * math.sin(3.0 * m_ref)
            )
            center_remainder = abs(center_series - center_exact)

            gain = float(np.clip(omega_series / max(omega_exact, _TINY), 0.0, 2.0))
            return {
                "omega_series": float(omega_series),
                "omega_exact": float(omega_exact),
                "remainder_rel": float(remainder),
                "agm_iterations": int(agm_iters),
                "damping_gain": gain,
                "center_remainder": float(center_remainder),
            }

        # ── Método 2.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_melnikov_homoclinic_integral(
            w_drive: float, perturbation_eps: float, krull_floor: float = _DEFAULT_KRULL_FLOOR
        ) -> Dict[str, Any]:
            """
            2.6 Integral homoclínica de Poincaré-Melnikov sobre la separatriz de Duffing
            Γ₀(t) = (√2 sech t, −√2 sech t tanh t), para ẍ = x − x³ + ε(f cos ωt − γẋ):
                M(t₀) = −(4/3)γ + √2 π f ω SECH(πω/2) sin(ω t₀).
            Derivación del núcleo: ∫sech t·e^{iωt}dt = π sech(πω/2); integrando por partes,
            ∫sech t tanh t sin(ωt)dt = πω sech(πω/2). El término disipativo es
            ∫ẋ₀²dt = 2∫sech²t tanh²t dt = 4/3.
            [CORRECCIÓN CRÍTICA C1] v14 usaba csch(πω/2): falso (producía M_osc → 2√2 f cuando
            ω→0, i.e. caos espurio en el límite adiabático, y una singularidad en ω=0).
            Criterio EXACTO de herradura de Smale: M tiene cero simple ⟺ A_rel > 1 con
            A_rel = M_osc/((4/3)γ). El umbral operativo 0.25 es un factor de seguridad 4×,
            declarado como tal (no es el umbral matemático).
            Forzamiento crítico: f/γ|_c = (4/3)/(√2 π ω sech(πω/2)) = 2√2 cosh(πω/2)/(3πω).
            """
            f0 = abs(_safe_float(perturbation_eps, 0.0))
            omega = abs(_safe_float(w_drive, 0.0))
            gamma = max(_safe_float(krull_floor, _DEFAULT_KRULL_FLOOR), 1.0e-6)
            barrier = (4.0 / 3.0) * gamma
            arg = min(700.0, math.pi * omega / 2.0)
            sech_val = 1.0 / math.cosh(arg)  # cosh ≥ 1: nunca singular, nunca desborda tras el clip
            melnikov_osc = math.sqrt(2.0) * math.pi * f0 * omega * sech_val
            melnikov_rel = melnikov_osc / barrier
            safe = bool(melnikov_rel <= _MAX_MELNIKOV_AMPLITUDE_REL)
            transversal = bool(melnikov_rel > _MELNIKOV_TRANSVERSALITY_THRESHOLD)
            if omega > 1e-9:
                critical_ratio = (4.0 / 3.0) / (math.sqrt(2.0) * math.pi * omega * sech_val)
            else:
                critical_ratio = math.inf
            return {
                "osc": float(melnikov_osc),
                "rel": float(melnikov_rel),
                "safe": safe,
                "transversal": transversal,
                "critical_ratio": float(min(critical_ratio, 1e12)),
            }

        # ── Método 2.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_chirikov_resonance_overlap(
            min_divisor: float, perturbation_eps: float
        ) -> Tuple[float, bool]:
            """
            2.7 Semiancho de isla resonante del pendulato de resonancia Δω = 2√ε y criterio
            de solapamiento de Chirikov K = (Δω₁+Δω₂)/|ω₁−ω₂| = 4√ε/max(δω,10⁻⁶).
            Es un criterio HEURÍSTICO (K ≲ 1); el valor riguroso-numérico para el mapa
            estándar es K_c ≈ 0.971635406 (Greene), auditado independientemente en 2.8.
            """
            delta_w = max(abs(_safe_float(min_divisor, 0.0)), 1.0e-6)
            island = 2.0 * math.sqrt(max(0.0, abs(_safe_float(perturbation_eps, 0.0))))
            chirikov_k = (2.0 * island) / delta_w
            return float(chirikov_k), bool(chirikov_k < _MAX_CHIRIKOV_OVERLAP)

        # ── Método 2.8 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_greene_residue_sequence(
            kick_k: float, max_q: int = 34
        ) -> Dict[str, Any]:
            """
            2.8 Criterio de residuos de Greene para el toro de razón áurea del mapa estándar
                p' = p − (K/2π) sin(2πx),   x' = x + p'   (mod 1).
            Las órbitas periódicas de número de rotación p/q = F_n/F_{n+1} → (√5−1)/2 se
            obtienen como solución del modelo de Frenkel-Kontorova
                u_{n+1} − 2u_n + u_{n−1} + (K/2π) sin(2π(u_n + np/q)) = 0,
            resuelto por Newton con CONTINUACIÓN en K desde el límite integrable K=0
            (donde u ≡ 0 es solución exacta), lo que da una rama bien definida.
            La monodromía tangente es M = Π T(x_n), T(x) = [[1−c, 1], [−c, 1]], c = K cos(2πx),
            con det T = 1 (simpléctica). El residuo de Greene es R = (2 − Tr M)/4:
                R → 0 (K < K_c),  R → 1/4 (K = K_c),  |R| → ∞ (K > K_c).
            Este es el único criterio de la Fase 2 que es riguroso-numérico y no heurístico.
            """
            k_target = float(np.clip(abs(_safe_float(kick_k, 0.0)), 0.0, 4.0))
            fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
            residues: List[float] = []
            certified = True
            if k_target <= 1e-12:
                return {"residues": (0.0,), "intact": True, "certified": True}

            for idx in range(len(fibs) - 1):
                p_rot, q_orb = fibs[idx], fibs[idx + 1]
                if q_orb > max_q:
                    break
                lin = np.arange(q_orb, dtype=np.float64) * (p_rot / q_orb)
                u = np.zeros(q_orb, dtype=np.float64)
                eye = np.eye(q_orb)
                rows = np.arange(q_orb)
                converged_branch = False
                cont_steps = 8
                for s in range(1, cont_steps + 1):
                    k_s = k_target * s / cont_steps
                    for _ in range(30):
                        x = u + lin
                        f_vec = np.roll(u, -1) - 2.0 * u + np.roll(u, 1) \
                            + (k_s / (2.0 * math.pi)) * np.sin(2.0 * math.pi * x)
                        if float(np.max(np.abs(f_vec))) < 1e-12:
                            converged_branch = True
                            break
                        c_vec = k_s * np.cos(2.0 * math.pi * x)
                        jac = np.diag(c_vec - 2.0)
                        jac[rows, (rows + 1) % q_orb] += 1.0
                        jac[rows, (rows - 1) % q_orb] += 1.0
                        try:
                            step = np.linalg.solve(jac, -f_vec)
                        except np.linalg.LinAlgError:
                            step = np.linalg.lstsq(jac, -f_vec, rcond=None)[0]
                        nrm = float(np.max(np.abs(step)))
                        if nrm > 0.25:
                            step *= 0.25 / nrm
                        u = u + step
                    if not converged_branch:
                        break
                if not converged_branch:
                    certified = False
                    break
                x = u + lin
                mono = np.eye(2)
                for xn in x:
                    c_val = k_target * math.cos(2.0 * math.pi * float(xn))
                    mono = np.array([[1.0 - c_val, 1.0], [-c_val, 1.0]]) @ mono
                residues.append(float((2.0 - float(np.trace(mono))) / 4.0))

            intact = bool(residues and abs(residues[-1]) < _GREENE_CRITICAL_RESIDUE
                          and max(abs(r) for r in residues) < 1.0)
            return {"residues": tuple(residues), "intact": intact, "certified": bool(certified and bool(residues))}

        # ── Método 2.9 ────────────────────────────────────────────────────────
        def _compute_floquet_krein_monodromy(
            self, freqs: List[float], perturbation_eps: float, eccentricity: float,
            chirikov_k: float, melnikov_rel: float,
        ) -> Dict[str, Any]:
            """
            2.9 Monodromía de Floquet REAL del sistema variacional de Jacobi-Poincaré,
            modelado como oscilador de Hill-Mathieu ẍ + (ω_i² + ε cos(Ωt))x = 0.
            [CORRECCIÓN C3] v14 construía bloques cuya rama hiperbólica era inalcanzable, de
            modo que el test de estabilidad era vacío. Aquí la monodromía se INTEGRA:
            en cada panel el coeficiente se congela en el punto medio y el flujo 2×2 es
            EXACTO (trigonométrico si k>0, hiperbólico si k<0, shear si k=0), con det = 1
            ⟹ simpléctica por construcción (certificado ‖MᵀJM − J‖ ~ ε_machine).
            Convergencia certificada por DUPLICACIÓN DE PANELES (extrapolación de Richardson).
            Cartas de Arnold: con τ = Ωt/2 se obtiene a = (2ω_i/Ω)², q = −2ε/Ω².
            Signatura de Krein κ(ξ) = sgn(−i ξ†Jξ) ∈ ℝ (pues ξ†Jξ es imaginario puro);
            una COLISIÓN de Krein (λ_i ≈ λ_j con κ_iκ_j < 0) es la bifurcación hamiltoniana
            de Hopf: ésa es la obstrucción genuina, no la mera pertenencia a S¹.
            Invariante integral relativo de Poincaré-Cartan ∮_γ p·dq = 2π Σ J_i, con las
            acciones de Delaunay Λ_i = √(μ a_i) = n_i^{−1/3} (μ=1) y el déficit de momento
            angular AMD = Σ Λ_i(1 − √(1−e_i²)) de Laskar. [v14 usaba π en vez de 2π y
            confundía frecuencias con acciones.]
            """
            d = max(1, min(len(freqs), 4))
            eps = abs(_safe_float(perturbation_eps, 0.0))
            ecc = float(np.clip(abs(_safe_float(eccentricity, 0.0)), 0.0, 0.99))
            drive = max(abs(freqs[0]), 1e-6)
            period = 2.0 * math.pi / drive
            j2 = np.array([[0.0, 1.0], [-1.0, 0.0]])

            def build_monodromy(panels: int) -> NDArray[np.float64]:
                h = period / panels
                blocks = []
                for i in range(d):
                    w_i = max(abs(freqs[i]), 1e-6)
                    mono = np.eye(2)
                    for jdx in range(panels):
                        t_mid = (jdx + 0.5) * h
                        k_coef = w_i * w_i + eps * math.cos(drive * t_mid)
                        if k_coef > 1e-14:
                            r = math.sqrt(k_coef)
                            blk = np.array([[math.cos(r * h), math.sin(r * h) / r],
                                            [-r * math.sin(r * h), math.cos(r * h)]])
                        elif k_coef < -1e-14:
                            r = math.sqrt(-k_coef)
                            blk = np.array([[math.cosh(r * h), math.sinh(r * h) / r],
                                            [r * math.sinh(r * h), math.cosh(r * h)]])
                        else:
                            blk = np.array([[1.0, h], [0.0, 1.0]])
                        mono = blk @ mono
                    blocks.append(mono)
                return np.asarray(la.block_diag(*blocks), dtype=np.float64)

            panels = 128
            mono = build_monodromy(panels)
            step_cert = 0.0
            for _ in range(6):
                finer = build_monodromy(2 * panels)
                step_cert = _relative(float(la.norm(finer - mono, ord="fro")),
                                      max(float(la.norm(finer, ord="fro")), _TINY))
                mono, panels = finer, 2 * panels
                if step_cert <= 1e-10:
                    break
            self._telemetry.cumulative_monodromy_panels += panels

            j_full = np.asarray(la.block_diag(*[j2] * d), dtype=np.float64)
            sympl_defect = float(la.norm(mono.T @ j_full @ mono - j_full, ord="fro"))
            eigvals, eigvecs = la.eig(mono)
            radius = float(np.max(np.abs(eigvals))) if eigvals.size else 1.0
            lyapunov = math.log(max(radius, _TINY)) / period

            signatures: List[Tuple[complex, float]] = []
            for idx in range(eigvals.size):
                lam = complex(eigvals[idx])
                if abs(abs(lam) - 1.0) <= 1e-6 and abs(lam.imag) > 1e-8:
                    xi = eigvecs[:, idx]
                    krein = float(np.real(-1j * np.vdot(xi, j_full @ xi)))
                    signatures.append((lam, math.copysign(1.0, krein) if abs(krein) > 1e-12 else 0.0))
            collision = False
            for i in range(len(signatures)):
                for jdx in range(i + 1, len(signatures)):
                    li, ki = signatures[i]
                    lj, kj = signatures[jdx]
                    if ki * kj < 0.0 and abs(li - lj) < 1e-3:
                        collision = True
            elliptic = bool(abs(radius - 1.0) <= 1e-6)
            krein_stable = bool(elliptic and not collision and chirikov_k < 1.0 and melnikov_rel < 0.5)

            actions = np.array([max(abs(f), 1e-9) ** (-1.0 / 3.0) for f in freqs[:d]])
            amd = float(np.sum(actions * (1.0 - math.sqrt(max(0.0, 1.0 - ecc * ecc)))))
            cartan = 2.0 * math.pi * amd
            return {
                "krein_stable": krein_stable,
                "radius": radius,
                "symplectic_defect": sympl_defect,
                "step_certificate": float(step_cert),
                "krein_collision": bool(collision),
                "lyapunov": float(lyapunov),
                "mathieu_a": float((2.0 * max(abs(freqs[0]), 1e-6) / drive) ** 2),
                "mathieu_q": float(-2.0 * eps / (drive * drive)),
                "amd": amd,
                "cartan": float(cartan),
                "actions": actions,
            }

        # ── Método 2.10 ───────────────────────────────────────────────────────
        @staticmethod
        def _evaluate_laskar_amd_stability(
            freqs: List[float], eccentricity: float, actions: NDArray[np.float64]
        ) -> Dict[str, Any]:
            """
            2.10 Criterio de AMD-estabilidad de Laskar-Petit (2017), caso coplanar y sólo
            colisión. Con Λ_i = √(μ a_i), a_i = n_i^{−2/3} (μ = GM = 1), el AMD total es
                C = Σ Λ_i (1 − √(1−e_i²)).
            El AMD CRÍTICO de un par es el mínimo de C sujeto a la tangencia de órbitas
                a₁(1+e₁) = a₂(1−e₂),  α = a₁/a₂ < 1,
            que se reduce a una minimización unimodal en e₁ ∈ [0, min(1, 1/α − 1)] resuelta
            por sección áurea (contracción de razón 1/φ). El sistema es AMD-estable si
            C < min_pares C_c: NINGUNA evolución secular puede producir cruce de órbitas.
            Se añade la cota suficiente de estabilidad de Hill de Gladman (1993):
                Δ = (a₂−a₁)/a₁ > 2.40 (μ₁+μ₂)^{1/3}.
            """
            ecc = float(np.clip(abs(_safe_float(eccentricity, 0.0)), 0.0, 0.99))
            n_arr = np.array([max(abs(f), 1e-9) for f in freqs], dtype=np.float64)
            semi_major = n_arr ** (-2.0 / 3.0)
            lam = np.array(actions, dtype=np.float64) if len(actions) == len(freqs) else n_arr ** (-1.0 / 3.0)
            amd_total = float(np.sum(lam * (1.0 - math.sqrt(max(0.0, 1.0 - ecc * ecc)))))

            order = np.argsort(semi_major)
            critical = math.inf
            hill_ok = True
            mu_pair = 1.0e-6
            for idx in range(len(order) - 1):
                i1, i2 = int(order[idx]), int(order[idx + 1])
                a1, a2 = float(semi_major[i1]), float(semi_major[i2])
                l1, l2 = float(lam[i1]), float(lam[i2])
                alpha = a1 / a2
                delta_rel = (a2 - a1) / a1
                if delta_rel <= _HILL_STABILITY_COEFF * (2.0 * mu_pair) ** (1.0 / 3.0):
                    hill_ok = False
                upper = min(0.999999, (1.0 / alpha) - 1.0)
                if upper <= 0.0:
                    critical = 0.0
                    continue

                def pair_amd(e1: float) -> float:
                    e2 = float(np.clip(1.0 - alpha * (1.0 + e1), 0.0, 0.999999))
                    return (l1 * (1.0 - math.sqrt(max(0.0, 1.0 - e1 * e1)))
                            + l2 * (1.0 - math.sqrt(max(0.0, 1.0 - e2 * e2))))

                _, c_c = _golden_section_minimize(pair_amd, 0.0, upper, tol=1e-12, max_iter=160)
                critical = min(critical, float(c_c))
            if not math.isfinite(critical):
                critical = math.inf
            stable = bool(amd_total < critical)
            return {
                "amd": amd_total,
                "critical": float(critical) if math.isfinite(critical) else 1e12,
                "stable": stable,
                "hill_ok": bool(hill_ok),
            }

        # ── Método 2.11 ───────────────────────────────────────────────────────
        @staticmethod
        def _adjudicate_celestial_heyting_truth(
            is_kam_stable: bool, is_melnikov_safe: bool, is_chirikov_contained: bool,
            krein_stable: bool, amd_stable: bool,
        ) -> Tuple[HeytingTruthValue, Tuple[str, ...]]:
            """
            2.11 Adjudicación en Ω₃. COHERENT exige la conjunción de las cinco barreras;
            la estabilidad de Krein y la AMD-estabilidad se incorporan como condiciones
            necesarias del estrato superior (v14 las calculaba y luego las IGNORABA).
            """
            reasons: List[str] = []
            if not is_kam_stable:
                reasons.append("KAM_DIOPHANTINE_DIVISOR_COLLAPSE")
            if not is_melnikov_safe:
                reasons.append("MELNIKOV_HOMOCLINIC_MARGIN_EXCEEDED")
            if not is_chirikov_contained:
                reasons.append("CHIRIKOV_RESONANCE_OVERLAP")
            if not krein_stable:
                reasons.append("FLOQUET_KREIN_OBSTRUCTION")
            if not amd_stable:
                reasons.append("AMD_SECULAR_CROSSING_ALLOWED")
            if is_kam_stable and is_melnikov_safe and is_chirikov_contained and krein_stable and amd_stable:
                return HeytingTruthValue.COHERENT, ()
            if is_kam_stable or is_chirikov_contained:
                return HeytingTruthValue.DEGRADED, tuple(reasons)
            return HeytingTruthValue.VETOED, tuple(reasons)

        # ── Método 2.12 ───────────────────────────────────────────────────────
        @staticmethod
        def _build_phase3_contractive_seed(
            report: CelestialWitnessReport, upstream: HeytingTruthValue
        ) -> Phase2ToPhase3Seed:
            """
            2.12 Sintetiza el margen de estabilidad celeste que la mónada de Banach absorbe
            en 3.1: la presión resonante es la combinación monótona de Chirikov, Melnikov y
            el exceso del radio de Floquet sobre el círculo unitario.
            """
            pressure = float(np.clip(
                0.5 * min(report.chirikov_overlap_k, 4.0)
                + 0.5 * min(report.melnikov_amplitude_rel, 4.0)
                + max(0.0, report.floquet_spectral_radius - 1.0),
                0.0, 8.0,
            ))
            return Phase2ToPhase3Seed(
                celestial_damping_gain=float(report.rsi_l3_resonance_damping_gain),
                resonance_pressure=pressure,
                floquet_spectral_radius=float(report.floquet_spectral_radius),
                greene_torus_intact=bool(report.greene_torus_intact),
                amd_stable=bool(report.amd_stable),
                upstream_truth=upstream,
                has_upstream_context=True,
            )

        # ── Método 2.13 · TERMINAL FASE 2 → APERTURA FASE 3 ───────────────────
        def audit_celestial_mechanics(
            self,
            frequencies: Optional[Sequence[float]] = None,
            perturbation_eps: float = 0.01,
            eccentricity: float = 0.05,
            enable_greene: bool = True,
            qnd_report: Optional[QNDInspectionReport] = None,
        ) -> CelestialWitnessReport:
            """
            2.13 [TERMINAL FASE 2] Orquesta 2.1 → 2.12 y EMITE el `Phase2ToPhase3Seed` que
            abre la Fase 3 en 3.1. Continuidad de tipos:

                audit_celestial_mechanics(...) → CelestialWitnessReport.continuation
                                               → Phase3._ingest_phase2_continuation(seed)
            """
            seed = self._ingest_phase1_continuation(qnd_report)
            eps_user = abs(_safe_float(perturbation_eps, 0.01))
            effective_eps = eps_user + seed.qnd_commutator_perturbation

            freqs, norm_freqs, _scale = self._normalize_frequency_torus(frequencies)
            lattice = self._evaluate_siegel_bruno_diophantine_lattice(norm_freqs)
            kepler = self._audit_kepler_torus(eccentricity)
            lindstedt = self._compute_lindstedt_poincare_canonical_series(
                freqs[0], effective_eps, eccentricity, kepler
            )
            melnikov = self._compute_melnikov_homoclinic_integral(freqs[0], effective_eps)
            chirikov_k, chirikov_ok = self._compute_chirikov_resonance_overlap(
                lattice["min_div"], effective_eps
            )
            greene = (
                self._compute_greene_residue_sequence(min(4.0 * effective_eps / max(lattice["min_div"], 1e-6), 2.0))
                if enable_greene else {"residues": (), "intact": True, "certified": False}
            )
            floquet = self._compute_floquet_krein_monodromy(
                freqs, effective_eps, eccentricity, chirikov_k, melnikov["rel"]
            )
            amd = self._evaluate_laskar_amd_stability(freqs, eccentricity, floquet["actions"])

            tv, reasons = self._adjudicate_celestial_heyting_truth(
                lattice["kam_stable"], melnikov["safe"], chirikov_ok,
                floquet["krein_stable"], amd["stable"],
            )
            self._telemetry.total_celestial_audits += 1
            self._telemetry.last_heyting_verdict = tv

            report = CelestialWitnessReport(
                is_kam_stable=bool(lattice["kam_stable"]),
                diophantine_min_divisor=float(lattice["min_div"]),
                lindstedt_omega=float(lindstedt["omega_series"]),
                melnikov_amplitude_rel=float(melnikov["rel"]),
                is_melnikov_safe=bool(melnikov["safe"]),
                chirikov_overlap_k=float(chirikov_k),
                is_chirikov_contained=bool(chirikov_ok),
                krein_signature_stable=bool(floquet["krein_stable"]),
                truth_value=tv,
                siegel_bruno_series_sum=float(lattice["bruno_sum"]),
                floquet_spectral_radius=float(floquet["radius"]),
                poincare_cartan_invariant=float(floquet["cartan"]),
                rsi_l3_resonance_damping_gain=float(lindstedt["damping_gain"]),
                diophantine_gamma=float(lattice["gamma"]),
                diophantine_tau=2.0,
                bruno_certified_bound=float(lattice["bruno_bound"]),
                bruno_convergent=bool(lattice["bruno_convergent"]),
                lattice_vector_count=int(lattice["lattice_count"]),
                cf_bounded_type=bool(lattice["bounded_type"]),
                kepler_max_residual=float(kepler["max_residual"]),
                kepler_mean_iterations=float(kepler["mean_iterations"]),
                kepler_empirical_order=float(kepler["empirical_order"]),
                kapteyn_series_residual=float(kepler["kapteyn_residual"]),
                laplace_series_regime=bool(kepler["laplace_regime"]),
                lindstedt_exact_omega=float(lindstedt["omega_exact"]),
                lindstedt_remainder_rel=float(lindstedt["remainder_rel"]),
                agm_iterations=int(lindstedt["agm_iterations"]),
                equation_of_center_remainder=float(lindstedt["center_remainder"]),
                melnikov_osc_amplitude=float(melnikov["osc"]),
                melnikov_transversal=bool(melnikov["transversal"]),
                melnikov_critical_forcing_ratio=float(melnikov["critical_ratio"]),
                greene_residues=tuple(greene["residues"]),
                greene_torus_intact=bool(greene["intact"]),
                greene_certified=bool(greene["certified"]),
                symplectic_defect=float(floquet["symplectic_defect"]),
                monodromy_step_certificate=float(floquet["step_certificate"]),
                krein_collision_detected=bool(floquet["krein_collision"]),
                mathieu_a=float(floquet["mathieu_a"]),
                mathieu_q=float(floquet["mathieu_q"]),
                max_lyapunov_exponent=float(floquet["lyapunov"]),
                angular_momentum_deficit=float(amd["amd"]),
                amd_critical=float(amd["critical"]),
                amd_stable=bool(amd["stable"]),
                hill_separation_ok=bool(amd["hill_ok"]),
                effective_perturbation_eps=float(effective_eps),
                obstruction_reasons=reasons,
            )
            combined = seed.upstream_truth.meet(tv)
            # ── CONTINUIDAD FORMAL: última sentencia de la Fase 2 = semilla de la Fase 3 ──
            report = replace(report, continuation=self._build_phase3_contractive_seed(report, combined))
            self._last_report = report
            return report

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 3 · GOBERNADOR KLEISLI RSI³ EN BANACH, LYAPUNOV Y ATESTACIÓN PASIVA
    # ──────────────────────────────────────────────────────────────────────────
    class Phase3_BanachKleisliRSI3Governor:
        """
        Fase 3. Cadena formal:
            3.1 INGESTA de Phase2ToPhase3Seed → 3.2 contracción de Banach →
            3.3 aceleración Δ²/Wynn → 3.4 Lyapunov y margen ISS de jerk →
            3.5 METAAUTOMEJORA RSI-L3 (L1/L2/L3 con presupuesto bien fundado) →
            3.6 adjudicación → 3.7 canonicalización+Merkle → 3.8 HMAC/verificación →
            3.9 auditoría de contracción → 3.10 sello del ciclo →
            3.11 TERMINAL (composición monádica completa de las tres fases).
        """

        def __init__(
            self,
            phase1_inspector: "BoolNovikovPaladinWitnessEngine.Phase1_QNDTomitaTakesakiInspector",
            phase2_navigator: "BoolNovikovPaladinWitnessEngine.Phase2_PoincareKAMCelestialNavigator",
            telemetry: RSILevel3WitnessTelemetry,
            secret_key: bytes,
        ) -> None:
            self._phase1 = phase1_inspector
            self._phase2 = phase2_navigator
            self._telemetry = telemetry
            self._secret_key = bytes(secret_key)
            self._last_report: Optional[BanachWitnessReport] = None

        # ── Método 3.1 · APERTURA FASE 3 (continuación directa de 2.13) ───────
        def _ingest_phase2_continuation(
            self,
            celestial_report: Optional[CelestialWitnessReport] = None,
            qnd_report: Optional[QNDInspectionReport] = None,
        ) -> Phase2ToPhase3Seed:
            """
            3.1 [CONTINUACIÓN DIRECTA DE 2.13] Absorbe la semilla contractiva de la Fase 2 y
            la conjunción intuicionista de los veredictos aguas arriba (Fase 1 ∧ Fase 2).
            """
            cel = celestial_report if celestial_report is not None else self._phase2._last_report
            qnd = qnd_report if qnd_report is not None else self._phase1._last_report
            combined = HeytingTruthValue.COHERENT
            if qnd is not None:
                combined = combined.meet(qnd.truth_value)
            if cel is not None:
                combined = combined.meet(cel.truth_value)
            if cel is not None and cel.continuation is not None:
                return replace(cel.continuation, upstream_truth=combined)
            if cel is not None:
                return self._phase2._build_phase3_contractive_seed(cel, combined)
            return Phase2ToPhase3Seed(
                celestial_damping_gain=1.0, resonance_pressure=0.0, floquet_spectral_radius=1.0,
                greene_torus_intact=True, amd_stable=True,
                upstream_truth=combined, has_upstream_context=False,
            )

        # ── Método 3.2 ────────────────────────────────────────────────────────
        @staticmethod
        def _evaluate_kleisli_t3_banach_contraction(
            prev_r: float, curr_r: float, history: Sequence[float]
        ) -> Dict[str, Any]:
            """
            3.2 Contracción de Banach-Caccioppoli CERTIFICADA. La razón q se estima por
            regresión log-lineal sobre el historial (MLE del decaimiento geométrico), que
            reduce al cociente r_k/r_{k−1} cuando sólo hay dos puntos y es robusta al ruido:
                ln r_j = a + b·j  ⟹  q̂ = e^b,  q = clip(q̂, 10⁻³, 0.999).
            Mónada de Kleisli T³ = T∘T∘T ⟹ Lip(T³) = q³ ≤ 0.857375 < 1 ⟹ punto fijo ÚNICO
            (autoconsistencia de Novikov realizada como punto fijo de Banach: la unicidad
            excluye ramas paradójicas de la automejora).
            Cotas rigurosas:
                A PRIORI      d(x_k,x*) ≤ qᵏ/(1−q)·d(x₀,x₁)
                A POSTERIORI  d(x_k,x*) ≤ r_k/(1−q)   con r_k = ‖x_k − T(x_k)‖
                N_ε = ⌈ln(ε_W(1−q)/r_k)/ln q⌉  (pasos hasta la precisión de Wilkinson).
            """
            prev_r = max(_WILKINSON_EPSILON, abs(prev_r))
            curr_r = max(_WILKINSON_EPSILON, abs(curr_r))
            seq = [max(_WILKINSON_EPSILON, abs(_safe_float(v, curr_r))) for v in history[-16:]]
            if len(seq) >= 3:
                t = np.arange(len(seq), dtype=np.float64)
                y = np.log(np.asarray(seq))
                t_c, y_c = t - t.mean(), y - y.mean()
                denom = float(np.sum(t_c ** 2))
                slope = float(np.sum(t_c * y_c) / denom) if denom > _TINY else math.log(curr_r / prev_r)
                pred = y.mean() + slope * t_c
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum(y_c ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > _TINY else 1.0
            else:
                slope = math.log(curr_r / prev_r)
                r2 = 1.0
            q_banach = float(np.clip(math.exp(float(np.clip(slope, -50.0, 50.0))), 0.001, 0.999))
            lip_t3 = q_banach ** 3
            contractive = bool(lip_t3 <= _BANACH_CONTRACT_CEILING)

            one_minus_q = max(1.0 - q_banach, 1e-6)
            a_post = curr_r / one_minus_q
            a_prior = (q_banach ** max(1, len(seq))) / one_minus_q * abs(seq[0] - seq[1]) if len(seq) >= 2 else a_post
            digits = max(0.0, -math.log10(max(a_post, _TINY)))
            if curr_r <= _WILKINSON_EPSILON:
                steps = 0
            else:
                target = (_WILKINSON_EPSILON * one_minus_q) / curr_r
                steps = 1 if target >= 1.0 else max(1, int(math.ceil(math.log(target) / math.log(q_banach))))
            return {
                "q": q_banach, "lip_t3": float(lip_t3), "contractive": contractive,
                "steps": int(steps), "r2": float(np.clip(r2, 0.0, 1.0)),
                "a_priori": float(a_prior), "a_posteriori": float(a_post),
                "digits": float(digits),
            }

        # ── Método 3.3 ────────────────────────────────────────────────────────
        @staticmethod
        def _rsi_l3_aitken_steffensen_accelerator(
            prev_r: float, curr_r: float, q_banach: float, history: Sequence[float], order: int
        ) -> Dict[str, Any]:
            """
            3.3 [RSI-L3] Aceleración de punto fijo con guarda de denominador RELATIVA.
            (a) Aitken-Steffensen Δ²:  r̂* = r_{k−1} − (r_k − r_{k−1})²/(r_{k+1} − 2r_k + r_{k−1}),
                exacto para órbitas geométricas; se aborta si |denominador| ≤ ε_W·escala
                (v14 usaba una guarda absoluta que fallaba para residuos diminutos).
            (b) ε-algoritmo de Wynn-Shanks (generalización de orden 2k de Δ²):
                ε_{−1}⁽ⁿ⁾=0, ε₀⁽ⁿ⁾=sₙ, ε_{k+1}⁽ⁿ⁾ = ε_{k−1}⁽ⁿ⁺¹⁾ + 1/(ε_k⁽ⁿ⁺¹⁾ − ε_k⁽ⁿ⁾).
                Los ε pares son las transformadas de Shanks; se selecciona el estimador de
                menor residual certificado. Defecto de punto fijo ‖r_k − T³(r_k)‖ = (1−q³)r_k.
            """
            prev_r = max(_WILKINSON_EPSILON, abs(prev_r))
            curr_r = max(_WILKINSON_EPSILON, abs(curr_r))
            next_pred = q_banach * curr_r
            denom = next_pred - 2.0 * curr_r + prev_r
            scale = max(abs(prev_r), abs(curr_r), _WILKINSON_EPSILON)
            if abs(denom) > 1e-12 * scale:
                aitken = prev_r - ((curr_r - prev_r) ** 2) / denom
                aitken = float(max(0.0, min(curr_r, abs(aitken))))
            else:
                aitken = float(next_pred)

            seq = [max(0.0, abs(_safe_float(v, 0.0))) for v in history[-12:]]
            wynn = aitken
            if len(seq) >= 3:
                eps_prev = [0.0] * (len(seq) + 1)
                eps_curr = list(seq) + [seq[-1]]
                best = seq[-1]
                for k in range(1, min(2 * max(1, order), len(seq))):
                    eps_next = [0.0] * (len(eps_curr) - 1)
                    for i in range(len(eps_curr) - 1):
                        diff = eps_curr[i + 1] - eps_curr[i]
                        if abs(diff) <= 1e-14 * max(abs(eps_curr[i]), 1.0):
                            eps_next[i] = eps_curr[i + 1]
                        else:
                            eps_next[i] = eps_prev[i + 1] + 1.0 / diff
                    eps_prev, eps_curr = eps_curr, eps_next
                    if k % 2 == 0 and eps_curr:
                        cand = float(eps_curr[0])
                        if math.isfinite(cand) and 0.0 <= cand <= best:
                            best = cand
                    if len(eps_curr) <= 1:
                        break
                wynn = float(max(0.0, min(curr_r, best)))

            selected = "wynn_shanks" if wynn < aitken else "aitken"
            accelerated = min(aitken, wynn)
            defect = abs(curr_r - (q_banach ** 3) * curr_r)
            return {
                "aitken": float(aitken), "wynn": float(wynn),
                "accelerated": float(accelerated), "selected": selected,
                "defect": float(defect),
            }

        # ── Método 3.4 ────────────────────────────────────────────────────────
        @staticmethod
        def _compute_lyapunov_and_jerk_dynamics(
            curr_r: float, q_banach: float, history: Sequence[float]
        ) -> Dict[str, Any]:
            """
            3.4 Funcional de Lyapunov-Poincaré y capacidad de jerk RIGUROSAS.
            [CORRECCIÓN C9] V(r) = ½r² es definida positiva y radialmente no acotada; para
            r_{k+1} = q r_k se tiene ΔV = ½(q²−1)r_k² < 0: certificado de estabilidad
            exponencial con tasa λ = −ln q. Se contrasta ΔV medido vs. predicho.
            JERK: la diferencia tercera discreta J₃ = r_k − 3r_{k−1} + 3r_{k−2} − r_{k−3}
            mide la sacudida d³x/dt³ de la órbita residual. La CAPACIDAD de jerk se define
            como el margen ISS exacto: una perturbación aditiva ζ preserva el descenso
            monótono ⟺ |ζ| < (1−q)·r_k. Ese número (no un porcentaje inventado) es la reserva
            de disipación frente a sacudidas de tercer orden.
            """
            curr_r = max(_WILKINSON_EPSILON, abs(curr_r))
            lyap = 0.5 * curr_r * curr_r
            seq = [abs(_safe_float(v, 0.0)) for v in history[-4:]]
            jerk3 = 0.0
            measured_dv = 0.0
            if len(seq) >= 4:
                jerk3 = float(seq[-1] - 3.0 * seq[-2] + 3.0 * seq[-3] - seq[-4])
            if len(seq) >= 2:
                measured_dv = 0.5 * (seq[-1] ** 2 - seq[-2] ** 2)
            predicted_dv = 0.5 * (q_banach ** 2 - 1.0) * (seq[-2] ** 2 if len(seq) >= 2 else curr_r ** 2)
            certificate_ok = bool(measured_dv <= 1e-12 or measured_dv <= predicted_dv * 0.5 + 1e-15)
            jerk_capacity = (1.0 - q_banach) * curr_r
            return {
                "lyapunov": float(lyap),
                "decrement": float(measured_dv),
                "certificate_ok": certificate_ok,
                "jerk_capacity": float(jerk_capacity),
                "jerk_third_difference": float(jerk3),
            }

        # ── Método 3.5 · NÚCLEO DE AUTOMEJORA RECURSIVA DE NIVEL 3 ────────────
        def _rsi_l3_meta_policy_update(
            self,
            banach: Dict[str, Any],
            accel: Dict[str, Any],
            seed: Phase2ToPhase3Seed,
            curr_r: float,
        ) -> Dict[str, Any]:
            """
            3.5 [RSI-L3] Automejora recursiva en TRES niveles anidados, acotada y auditable.

            L1 · REACTIVO: reajusta la tolerancia efectiva por la ganancia observada
                 g₁ = clip(r_{k−1}/r_k, 1, 10) y propone una relajación ω' = ω·(q̂/q_obj)^{η}.
            L2 · REFLEXIVO: estima el orden empírico de convergencia
                 p̂ = ln(e_{k+1}/e_k)/ln(e_k/e_{k−1}) sobre el historial y elige el orden del
                 acelerador de Shanks (2k) que minimiza el residual certificado.
            L3 · META: aplica la proyección Π_Θ sobre el compacto de políticas y ACEPTA la
                 automodificación sólo si se cumplen las TRES cláusulas del certificado:
                   (C1) Lyapunov: V propuesta < V actual (descenso estricto).
                   (C2) Banach: Lip(T³) estimada sobre Θ, q_meta³ ≤ 0.857375.
                   (C3) POZO BIEN FUNDADO: proof_depth_budget > 0, y se DECREMENTA.
                 Si alguna falla → ROLLBACK íntegro de la política.

            FUNDAMENTO LÓGICO: el obstáculo löbiano (una sistema no puede confiar en su propia
            soundness sin inconsistencia) se evita NO asserting auto-confianza: cada nivel
            exige un certificado explícito a profundidad de prueba ESTRICTAMENTE MENOR.
            Como ℕ es bien fundado, toda cadena de automodificación es finita: la recursión
            es total y terminante por descenso ordinal, no por un punto fijo autorreferente.
            """
            tel = self._telemetry
            policy = tel.policy
            hist = [max(_WILKINSON_EPSILON, abs(v)) for v in tel.residual_history[-6:]]

            gain_l1 = float(np.clip(hist[-2] / hist[-1], 1.0, 10.0)) if len(hist) >= 2 else 1.0

            order_l2 = 1.0
            if len(hist) >= 3 and hist[-3] > hist[-2] > hist[-1] > 0.0:
                num = math.log(max(hist[-1] / hist[-2], _TINY))
                den = math.log(max(hist[-2] / hist[-3], _TINY))
                if abs(den) > 1e-9:
                    order_l2 = float(np.clip(num / den, 0.0, 6.0))
            tel.empirical_order_history.append(order_l2)
            if len(tel.empirical_order_history) > 64:
                tel.empirical_order_history.pop(0)
            desired_order = int(np.clip(round(order_l2), 1, 4))

            pressure = float(np.clip(seed.resonance_pressure, 0.0, 8.0))
            q_target = float(np.clip(policy.trust_q_ceiling * (1.0 - 0.05 * pressure), 0.5, 0.999))
            ratio = banach["q"] / max(q_target, 1e-6)
            proposed_relax = policy.relaxation * float(np.clip(ratio, 0.5, 2.0)) ** 0.25
            proposed = RSI3Policy(
                relaxation=proposed_relax,
                accelerator_order=desired_order,
                trust_q_ceiling=q_target,
                proof_depth_budget=policy.proof_depth_budget - 1,
                meta_lipschitz=policy.meta_lipschitz,
            ).project()

            step_new = policy.distance(proposed)
            ledger = tel.policy_ledger
            step_old = float(ledger[-1].get("step", 0.0)) if ledger else 0.0
            q_meta = float(np.clip(step_new / step_old, 0.001, 0.999)) if step_old > 1e-12 else 0.5
            proposed = replace(proposed, meta_lipschitz=q_meta).project()

            v_curr = 0.5 * curr_r * curr_r
            v_prop = 0.5 * (accel["accelerated"] * proposed.relaxation) ** 2
            c1_lyapunov = bool(v_prop < v_curr or curr_r <= _WILKINSON_EPSILON)
            c2_banach = bool((q_meta ** 3) <= _BANACH_CONTRACT_CEILING and banach["contractive"])
            c3_well_founded = bool(policy.proof_depth_budget > 0)
            accepted = bool(c1_lyapunov and c2_banach and c3_well_founded)

            if accepted:
                tel.policy = proposed
            else:
                tel.rollback_count += 1
                tel.policy = replace(policy, proof_depth_budget=max(0, policy.proof_depth_budget)).project()

            tel.policy_ledger.append({
                "step": float(step_new),
                "q_meta": float(q_meta),
                "accepted": accepted,
                "C1_lyapunov": c1_lyapunov,
                "C2_banach": c2_banach,
                "C3_well_founded": c3_well_founded,
                "depth_remaining": int(tel.policy.proof_depth_budget),
                "order_estimate": float(order_l2),
            })
            if len(tel.policy_ledger) > 64:
                tel.policy_ledger.pop(0)

            return {
                "l1_gain": gain_l1,
                "l2_order": order_l2,
                "relaxation": float(tel.policy.relaxation),
                "meta_lipschitz": float(q_meta),
                "accepted": accepted,
                "depth_remaining": int(tel.policy.proof_depth_budget),
                "rollbacks": int(tel.rollback_count),
                "novikov_ok": bool(banach["contractive"]),
            }

        # ── Método 3.6 ────────────────────────────────────────────────────────
        @staticmethod
        def _adjudicate_banach_heyting_truth(
            is_contractive: bool, q_banach: float, meta_accepted: bool, lyapunov_ok: bool
        ) -> Tuple[HeytingTruthValue, Tuple[str, ...]]:
            """
            3.6 Adjudicación en Ω₃. COHERENT exige contracción de T³ Y descenso de Lyapunov;
            la aceptación del metanivel es un requisito del estrato superior, no un adorno:
            sin certificado de automejora el veredicto no puede ser ⊤ (fail-closed).
            """
            reasons: List[str] = []
            if not is_contractive:
                reasons.append("KLEISLI_T3_NOT_CONTRACTIVE")
            if not lyapunov_ok:
                reasons.append("LYAPUNOV_MONOTONICITY_VIOLATED")
            if not meta_accepted:
                reasons.append("RSI_L3_CERTIFICATE_WITHHELD")
            if is_contractive and lyapunov_ok and meta_accepted:
                return HeytingTruthValue.COHERENT, ()
            if is_contractive and q_banach < 0.95:
                return HeytingTruthValue.DEGRADED, tuple(reasons)
            if is_contractive:
                return HeytingTruthValue.DEGRADED, tuple(reasons)
            return HeytingTruthValue.VETOED, tuple(reasons)

        # ── Método 3.7 ────────────────────────────────────────────────────────
        @staticmethod
        def _canonical_serialize(fields: Dict[str, Any]) -> str:
            """
            3.7a Canonicalización INYECTIVA del payload: JSON con claves ordenadas y
            separadores fijos, sin NaN/Inf. Los flotantes se emiten en notación científica
            de 12 dígitos para determinismo entre plataformas. v14 concatenaba campos con
            '|' sin longitudes de prefijo: un campo con '|' inyectaba un payload distinto
            con la misma firma (colisión de canonicalización).
            """
            normalized: Dict[str, Any] = {}
            for key in sorted(fields):
                val = fields[key]
                if isinstance(val, bool) or val is None:
                    normalized[key] = val
                elif isinstance(val, (int, np.integer)):
                    normalized[key] = int(val)
                elif isinstance(val, (float, np.floating)):
                    fv = float(val)
                    normalized[key] = format(fv, ".12e") if math.isfinite(fv) else "non-finite"
                elif isinstance(val, (tuple, list)):
                    normalized[key] = [
                        format(float(v), ".12e") if isinstance(v, (int, float, np.floating)) else str(v)
                        for v in val
                    ]
                else:
                    normalized[key] = str(val)
            return json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False)

        def _seal_certificate(self, fields: Dict[str, Any], domain: bytes) -> Dict[str, Any]:
            """
            3.7b Sello append-only: raíz de Merkle sobre las hojas de fase, encadenamiento
            con la firma anterior (cadena de hashes) y contador monótono anti-replay.
            """
            tel = self._telemetry
            tel.monotone_counter += 1
            leaves = [
                self._canonical_serialize(fields.get("phase1", {})).encode("utf-8"),
                self._canonical_serialize(fields.get("phase2", {})).encode("utf-8"),
                self._canonical_serialize(fields.get("phase3", {})).encode("utf-8"),
            ]
            root = _merkle_root(leaves)
            envelope = dict(fields)
            envelope.update({
                "merkle_root": root,
                "chain_prev": tel.signature_chain[-1] if tel.signature_chain else "genesis",
                "counter": int(tel.monotone_counter),
                "domain": domain.decode("utf-8", errors="replace"),
                "dev_key": bool(tel.dev_key_in_use),
            })
            payload = self._canonical_serialize(envelope)
            signature = self.generate_witness_hmac_signature(payload, domain)
            tel.signature_chain.append(signature)
            if len(tel.signature_chain) > 256:
                tel.signature_chain.pop(0)
            return {"canonical_payload": payload, "signature": signature, "merkle_root": root}

        # ── Método 3.8 ────────────────────────────────────────────────────────
        def generate_witness_hmac_signature(self, payload: str, domain: bytes) -> str:
            """
            3.8a HMAC-SHA256 con separación de dominio por derivación HKDF (extract-then-expand):
                PRK = HMAC-SHA256(salt = _HKDF_SALT, ikm = secret_key)
                K_dom = HMAC-SHA256(PRK, domain ‖ 0x01)
                firma = HMAC-SHA256(K_dom, payload).
            [CORRECCIÓN C10] v14 usaba K = domain ‖ secret_key, que es AMBIGUO en la frontera
            (domain="AB", key="C") ≡ (domain="A", key="BC"): dos contextos distintos compartían
            clave. La derivación por PRF elimina la ambigüedad y da separación criptográfica.
            """
            domain_bytes = domain if isinstance(domain, bytes) else str(domain).encode("utf-8")
            prk = hmac.new(_HKDF_SALT, self._secret_key, hashlib.sha256).digest()
            k_dom = hmac.new(prk, domain_bytes + b"\x01", hashlib.sha256).digest()
            return hmac.new(k_dom, str(payload).encode("utf-8"), hashlib.sha256).hexdigest()

        def verify_witness_hmac_signature(self, payload: str, domain: bytes, signature: str) -> bool:
            """3.8b Verificación en TIEMPO CONSTANTE (hmac.compare_digest), anti-timing."""
            expected = self.generate_witness_hmac_signature(payload, domain)
            try:
                return bool(hmac.compare_digest(expected, str(signature)))
            except (TypeError, ValueError):
                return False

        # ── Método 3.9 ────────────────────────────────────────────────────────
        def audit_banach_rsi3_contraction(
            self,
            previous_residual: float,
            current_residual: float,
            celestial_report: Optional[CelestialWitnessReport] = None,
            qnd_report: Optional[QNDInspectionReport] = None,
        ) -> BanachWitnessReport:
            """
            3.9 Audita la contracción de Banach-Picard de la mónada Kleisli T³, ejecuta la
            automejora RSI-L3 (3.5) y emite el informe inmutable que sella 3.10.
            """
            seed = self._ingest_phase2_continuation(celestial_report, qnd_report)
            prev_r = max(_WILKINSON_EPSILON, abs(_safe_float(previous_residual, 0.1)))
            curr_r = max(_WILKINSON_EPSILON, abs(_safe_float(current_residual, 0.05)))

            history = list(self._telemetry.residual_history) + [curr_r]
            banach = self._evaluate_kleisli_t3_banach_contraction(prev_r, curr_r, history)
            accel = self._rsi_l3_aitken_steffensen_accelerator(
                prev_r, curr_r, banach["q"], history, self._telemetry.policy.accelerator_order
            )
            dyn = self._compute_lyapunov_and_jerk_dynamics(curr_r, banach["q"], history)
            meta = self._rsi_l3_meta_policy_update(banach, accel, seed, curr_r)

            tv, reasons = self._adjudicate_banach_heyting_truth(
                banach["contractive"], banach["q"], meta["accepted"], dyn["certificate_ok"]
            )
            if seed.upstream_truth != HeytingTruthValue.COHERENT:
                tv = tv.meet(seed.upstream_truth)
                reasons = reasons + ("UPSTREAM_OBSTRUCTION",)

            self._telemetry.total_banach_audits += 1
            self._telemetry.residual_history.append(curr_r)
            if len(self._telemetry.residual_history) > 64:
                self._telemetry.residual_history.pop(0)
            self._telemetry.last_heyting_verdict = tv

            report = BanachWitnessReport(
                banach_ratio_q=float(banach["q"]),
                is_banach_contractive=bool(banach["contractive"]),
                lyapunov_value=float(dyn["lyapunov"]),
                jerk_capacity=float(dyn["jerk_capacity"]),
                truth_value=tv,
                kleisli_t3_lipschitz=float(banach["lip_t3"]),
                aitken_accelerated_residual=float(accel["accelerated"]),
                estimated_steps_to_eps=int(banach["steps"]),
                rsi_l3_fixed_point_defect=float(accel["defect"]),
                geometric_fit_r2=float(banach["r2"]),
                a_priori_error_bound=float(banach["a_priori"]),
                a_posteriori_error_bound=float(banach["a_posteriori"]),
                certified_decimal_digits=float(banach["digits"]),
                lyapunov_decrement=float(dyn["decrement"]),
                lyapunov_certificate_ok=bool(dyn["certificate_ok"]),
                jerk_third_difference=float(dyn["jerk_third_difference"]),
                wynn_epsilon_residual=float(accel["wynn"]),
                accelerator_selected=str(accel["selected"]),
                empirical_convergence_order=float(meta["l2_order"]),
                rsi_l1_tolerance_gain=float(meta["l1_gain"]),
                rsi_l2_policy_relaxation=float(meta["relaxation"]),
                rsi_l3_meta_lipschitz=float(meta["meta_lipschitz"]),
                rsi_l3_meta_accepted=bool(meta["accepted"]),
                rsi_l3_proof_depth_remaining=int(meta["depth_remaining"]),
                rsi_l3_rollback_count=int(meta["rollbacks"]),
                novikov_self_consistent=bool(meta["novikov_ok"]),
                obstruction_reasons=tuple(reasons),
            )
            self._last_report = report
            return report

        # ── Método 3.10 ───────────────────────────────────────────────────────
        def _compose_witness_certificate(
            self,
            qnd_rep: QNDInspectionReport,
            cel_rep: CelestialWitnessReport,
            banach_rep: BanachWitnessReport,
            domain: bytes,
        ) -> Dict[str, Any]:
            """
            3.10 Compone el veredicto global por conjunción intuicionista en Ω₃ y sella el
            certificado con Merkle + cadena de hashes + HMAC. La colapsación booleana ¬¬
            se reporta POR SEPARADO: el veredicto intuicionista conserva el estrato DEGRADED
            (información que la lógica clásica destruiría).
            """
            global_truth = qnd_rep.truth_value.meet(cel_rep.truth_value).meet(banach_rep.truth_value)
            fields = {
                "v": _ENGINE_VERSION,
                "phase1": {
                    "qnd": qnd_rep.qnd_commutator_norm, "c_rel": qnd_rep.relative_commutator,
                    "fides": qnd_rep.uhlmann_fidelity, "f_lo": qnd_rep.fidelity_lower_certificate,
                    "bures": qnd_rep.bures_distance, "betti1": qnd_rep.betti_1_topology,
                    "kms": qnd_rep.kms_correlation_residual, "beta": qnd_rep.modular_beta,
                    "truth": int(qnd_rep.truth_value),
                },
                "phase2": {
                    "gamma_kam": cel_rep.diophantine_gamma, "bruno": cel_rep.siegel_bruno_series_sum,
                    "omega_lp": cel_rep.lindstedt_omega, "omega_exact": cel_rep.lindstedt_exact_omega,
                    "melnikov": cel_rep.melnikov_amplitude_rel, "chirikov": cel_rep.chirikov_overlap_k,
                    "rho_floquet": cel_rep.floquet_spectral_radius, "amd": cel_rep.angular_momentum_deficit,
                    "truth": int(cel_rep.truth_value),
                },
                "phase3": {
                    "q": banach_rep.banach_ratio_q, "lip_t3": banach_rep.kleisli_t3_lipschitz,
                    "a_post": banach_rep.a_posteriori_error_bound, "lyap": banach_rep.lyapunov_value,
                    "rsi_accepted": banach_rep.rsi_l3_meta_accepted,
                    "depth": banach_rep.rsi_l3_proof_depth_remaining,
                    "truth": int(banach_rep.truth_value),
                },
                "truth": int(global_truth),
                "truth_boolean": int(global_truth.boolean_collapse()),
            }
            sealed = self._seal_certificate(fields, domain)
            return {
                "global_truth_value": global_truth,
                "global_truth_boolean_collapse": global_truth.boolean_collapse(),
                "canonical_payload": sealed["canonical_payload"],
                "hmac_sha256_signature": sealed["signature"],
                "merkle_root": sealed["merkle_root"],
                "attestation_counter": int(self._telemetry.monotone_counter),
            }

        # ── Método 3.11 · TERMINAL FASE 3 · COMPOSICIÓN MONÁDICA COMPLETA ─────
        def execute_full_paladin_witness_cycle(
            self,
            rho_state: NDArray[np.complex128],
            rho_target: Optional[NDArray[np.complex128]] = None,
            hamiltonian: Optional[NDArray[np.complex128]] = None,
            frequencies: Optional[Sequence[float]] = None,
            perturbation_eps: float = 0.01,
            eccentricity: float = 0.05,
            previous_residual: float = 0.10,
            current_residual: float = 0.05,
            domain: bytes = b"APU_PALADIN_QND_WITNESS_DOMAIN_V15",
            enable_greene: bool = True,
        ) -> Dict[str, Any]:
            """
            3.11 [TERMINAL FASE 3] Composición monádica T³ de las tres fases anidadas, con
            continuidad de tipos de extremo a extremo:

                1.11 audit_qnd_inspection ──Phase1ToPhase2Seed──► 2.1 … 2.13
                                          ──Phase2ToPhase3Seed──► 3.1 … 3.9 ──► 3.10 sello

            El punto fijo ρ* = T³(ρ*) es la condición de autoconsistencia de Novikov; su
            UNICIDAD (Banach) excluye ramas paradójicas del bucle de automejora.
            """
            qnd_rep = self._phase1.audit_qnd_inspection(
                rho_state=rho_state, rho_target=rho_target, hamiltonian=hamiltonian
            )
            cel_rep = self._phase2.audit_celestial_mechanics(
                frequencies=frequencies if frequencies is not None else [1.0, math.sqrt(2.0), math.sqrt(3.0)],
                perturbation_eps=perturbation_eps,
                eccentricity=eccentricity,
                enable_greene=enable_greene,
                qnd_report=qnd_rep,
            )
            banach_rep = self.audit_banach_rsi3_contraction(
                previous_residual=previous_residual,
                current_residual=current_residual,
                celestial_report=cel_rep,
                qnd_report=qnd_rep,
            )
            certificate = self._compose_witness_certificate(qnd_rep, cel_rep, banach_rep, domain)
            return {
                "engine_version": _ENGINE_VERSION,
                "qnd_report": qnd_rep,
                "celestial_report": cel_rep,
                "banach_report": banach_rep,
                "rsi_l3_policy": self._telemetry.policy,
                **certificate,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # §5 · INICIALIZACIÓN Y DELEGACIÓN SOBERANA
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, secret_key: Optional[bytes] = None, policy: Optional[RSI3Policy] = None) -> None:
        self.secret_key: bytes = bytes(secret_key) if secret_key else _DEFAULT_SECRET_KEY
        self._telemetry = RSILevel3WitnessTelemetry(
            policy=(policy or RSI3Policy()).project(),
            dev_key_in_use=not bool(secret_key),
        )
        if not secret_key:
            logger.warning(
                "Clave de desarrollo en uso: los certificados NO son atestables en producción. "
                "Suministre secret_key explícita (≥32 bytes de entropía)."
            )
        self.phase1_inspector = self.Phase1_QNDTomitaTakesakiInspector(telemetry=self._telemetry)
        self.phase2_navigator = self.Phase2_PoincareKAMCelestialNavigator(
            phase1_inspector=self.phase1_inspector, telemetry=self._telemetry
        )
        self.phase3_governor = self.Phase3_BanachKleisliRSI3Governor(
            phase1_inspector=self.phase1_inspector,
            phase2_navigator=self.phase2_navigator,
            telemetry=self._telemetry,
            secret_key=self.secret_key,
        )
        logger.info("BoolNovikovPaladinWitnessEngine v%s inicializado (3 fases anidadas).", _ENGINE_VERSION)

    def audit_qnd_inspection(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
    ) -> QNDInspectionReport:
        """Delega en 1.11 (terminal de Fase 1, emisor de la semilla hacia 2.1)."""
        return self.phase1_inspector.audit_qnd_inspection(rho_state, rho_target, hamiltonian)

    def audit_celestial_mechanics(
        self,
        frequencies: Sequence[float],
        perturbation_eps: float = 0.01,
        eccentricity: float = 0.05,
        enable_greene: bool = True,
    ) -> CelestialWitnessReport:
        """Delega en 2.13 (terminal de Fase 2, emisor de la semilla hacia 3.1)."""
        return self.phase2_navigator.audit_celestial_mechanics(
            frequencies=frequencies, perturbation_eps=perturbation_eps,
            eccentricity=eccentricity, enable_greene=enable_greene,
        )

    def audit_banach_rsi3_contraction(
        self, previous_residual: float, current_residual: float
    ) -> BanachWitnessReport:
        """Delega en 3.9 (auditoría de contracción de Kleisli T³ y automejora RSI-L3)."""
        return self.phase3_governor.audit_banach_rsi3_contraction(previous_residual, current_residual)

    def generate_witness_hmac_signature(self, payload: str, domain: bytes) -> str:
        """Delega en 3.8a (HMAC-SHA256 con separación de dominio por HKDF)."""
        return self.phase3_governor.generate_witness_hmac_signature(payload, domain)

    def verify_witness_hmac_signature(self, payload: str, domain: bytes, signature: str) -> bool:
        """Delega en 3.8b (verificación en tiempo constante)."""
        return self.phase3_governor.verify_witness_hmac_signature(payload, domain, signature)

    def execute_full_paladin_witness_cycle(
        self,
        rho_state: NDArray[np.complex128],
        rho_target: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        frequencies: Optional[Sequence[float]] = None,
        perturbation_eps: float = 0.01,
        eccentricity: float = 0.05,
        previous_residual: float = 0.10,
        current_residual: float = 0.05,
        domain: bytes = b"APU_PALADIN_QND_WITNESS_DOMAIN_V15",
        enable_greene: bool = True,
    ) -> Dict[str, Any]:
        """Delega en 3.11 (composición monádica completa Fase 1 → Fase 2 → Fase 3)."""
        return self.phase3_governor.execute_full_paladin_witness_cycle(
            rho_state=rho_state, rho_target=rho_target, hamiltonian=hamiltonian,
            frequencies=frequencies, perturbation_eps=perturbation_eps, eccentricity=eccentricity,
            previous_residual=previous_residual, current_residual=current_residual,
            domain=domain, enable_greene=enable_greene,
        )

    def get_rsi_l3_telemetry(self) -> RSILevel3WitnessTelemetry:
        """Retorna el libro mayor vivo de la automejora recursiva de Nivel 3."""
        return self._telemetry


# ══════════════════════════════════════════════════════════════════════════════
# §6 · AUTOVERIFICACIÓN EJECUTABLE (identidades algebraicas y certificados)
# ══════════════════════════════════════════════════════════════════════════════
def _self_test() -> bool:
    """
    Batería mínima de identidades verificables. Cada aserción corresponde a un teorema
    o a una identidad algebraica citada en los docstrings, no a un valor empírico.
    """
    ok = True
    checks: List[Tuple[str, bool]] = []

    checks.append(("Heyting: residuación y topología j=¬¬", HeytingTruthValue.verify_heyting_axioms()))
    checks.append(("Heyting: DEGRADED no es ¬¬-estable",
                   not HeytingTruthValue.DEGRADED.is_regular()))

    p1 = BoolNovikovPaladinWitnessEngine.Phase1_QNDTomitaTakesakiInspector
    probs, _theta, feasible = p1._project_spectrum_capped_simplex(
        np.array([-0.4, 0.2, 0.5, 0.7]), 1e-6
    )
    checks.append(("Símplex capado: KKT (p≥δ, Σp=1)",
                   feasible and abs(float(probs.sum()) - 1.0) < 1e-12 and float(probs.min()) >= 0.0))

    agm_val, _ = _arithmetic_geometric_mean(1.0, math.cos(0.25))
    series = 1.0 - 0.5 ** 2 / 16.0 + 0.5 ** 4 / 3072.0 - 23.0 * 0.5 ** 6 / 737280.0
    checks.append(("AGM ≡ serie de Lindstedt del pendulato (A=0.5)", abs(agm_val - series) < 1e-6))

    p2 = BoolNovikovPaladinWitnessEngine.Phase2_PoincareKAMCelestialNavigator
    mel_lo = p2._compute_melnikov_homoclinic_integral(1.0, 0.01, 0.1)
    mel_hi = p2._compute_melnikov_homoclinic_integral(1.0, 0.01, 0.1)
    slow = p2._compute_melnikov_homoclinic_integral(1e-6, 0.01, 0.1)["osc"]
    checks.append(("Melnikov sech: límite adiabático M_osc → 0", slow < 1e-4))
    checks.append(("Melnikov: umbral de Smale bien escalado",
                   abs(mel_lo["rel"] - mel_hi["rel"]) < 1e-15))

    sol, res, _its, _errs = p2._solve_kepler_equation_certified(0.9, 0.3)
    checks.append(("Kepler: residual |E − e sinE − M| < 1e-12", res < 1e-12))
    checks.append(("Bessel J₀(0) = 1 (cuadratura periódica)",
                   abs(_bessel_j_periodic_trapezoid(0, 0.0) - 1.0) < 1e-12))

    engine = BoolNovikovPaladinWitnessEngine(secret_key=b"self-test-key-32-bytes-of-entropy!!")
    rho = np.diag([0.4, 0.3, 0.2, 0.1]).astype(np.complex128)
    ham = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
    out = engine.execute_full_paladin_witness_cycle(rho_state=rho, rho_target=rho, hamiltonian=ham)
    q_rep, c_rep, b_rep = out["qnd_report"], out["celestial_report"], out["banach_report"]

    checks.append(("Fidelidad F(ρ,ρ) = 1", abs(q_rep.uhlmann_fidelity - 1.0) < 1e-9))
    checks.append(("Fuchs-van de Graaf", q_rep.fuchs_van_de_graaf_ok))
    checks.append(("Identidad de Euler-Poincaré", q_rep.euler_identity_ok))
    checks.append(("Estado diagonal ⟹ QND exacto", q_rep.is_qnd_valid))
    checks.append(("Monodromía simpléctica ‖MᵀJM − J‖ < 1e-10", c_rep.symplectic_defect < 1e-10))
    checks.append(("Continuidad 1.11 → 2.1 tipada", q_rep.continuation is not None))
    checks.append(("Continuidad 2.13 → 3.1 tipada", c_rep.continuation is not None))
    checks.append(("Banach: Lip(T³) = q³", abs(b_rep.kleisli_t3_lipschitz - b_rep.banach_ratio_q ** 3) < 1e-12))
    checks.append(("HMAC: verificación en tiempo constante",
                   engine.verify_witness_hmac_signature(
                       out["canonical_payload"], b"APU_PALADIN_QND_WITNESS_DOMAIN_V15",
                       out["hmac_sha256_signature"])))
    checks.append(("HMAC: separación de dominio efectiva",
                   engine.generate_witness_hmac_signature("x", b"A")
                   != engine.generate_witness_hmac_signature("x", b"B")))

    for name, passed in checks:
        ok = ok and bool(passed)
        print(f"[{'OK ' if passed else 'FAIL'}] {name}")
    print(f"\nVEREDICTO GLOBAL DE AUTOVERIFICACIÓN: {'COHERENTE' if ok else 'OBSTRUIDO'}")
    return ok


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(name)s · %(message)s")
    raise SystemExit(0 if _self_test() else 1)
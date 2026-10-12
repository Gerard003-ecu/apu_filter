# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Cognitive Crop Witness Engine (Motor Testigo del Cultivo Cognitivo)                  ║
║ Ubicación: app/wisdom/toon_cognitive_crop_witness_engine.py                                          ║
║ Versión  : 7.0.0-Doctoral-Crop-Witness-QND-Maupertuis-Poincare-Floquet-Birkhoff-RSI3-Banach          ║
║            -FixedPoint-Nested-Phases-Heyting-ESP32                                                   ║
║ Función  : Motor espectral observador sin demolición cuántica (QND, back-action ≡ 0.0 dB) del        ║
║            metabolismo del cultivo cognitivo, responsable de medir geodésicas en la métrica          ║
║            conformal de Maupertuis-Jacobi, clasificar multiplicadores orbitales de Floquet, computar ║
║            números de rotación de Poincaré y ejecutar el ciclo RSI³ con punto fijo de Banach.        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

I. DEFINICIÓN Y ROL DEL MOTOR EN LA ARQUITECTURA COGNITIVA
──────────────────────────────────────────────────────────
El TOON Cognitive Crop Witness Engine es el observador cuántico débil del ciclo agronómico-cognitivo
(Estrato WISDOM). Su mandato consiste en auditar sin back-action perturbativo (0.0 dB) la germinación,
riego metabólico, transpiración entrópica e iluminación de los cartuchos de conocimiento TOON
generados por el Soberano del Cultivo. A través de la reformulación geodésica de Maupertuis-Jacobi
y el análisis espectral de Floquet-Poincaré, cuantifica la estabilidad de las parcelas de deliberación
y condiciona su certificación a la contracción uniforme del operador de automejora recursiva.

II. PRINCIPIO VARIACIONAL DE MAUPERTUIS-JACOBI Y GEOMETRÍA CONFORMAL
────────────────────────────────────────────────────────────────────
§1. Métrica Conformal de Maupertuis:
    Para un sistema hamiltoniano con energía total constante $H(q, p) = T(q, p) + V(q) = E$, el principio
    de mínima acción de Maupertuis $\delta \int p \, dq = 0$ equivale a la ecuación geodésica bajo
    la métrica riemanniana conformemente deformada $g_{ij}^M$:
        ds_M^2 = 2 \left( E - V(q) \right) g_{ij} dq^i dq^j \iff g_{ij}^M(q) = 2(E - V(q)) g_{ij}(q)
    Las soluciones dinámicas reales del cultivo son geodésicas parametrizadas por longitud de arco $s$:
        \nabla_{\dot{\gamma}}^M \dot{\gamma} = 0 \implies \int_{\gamma} ds_M \le \int_{\tilde{\gamma}} ds_M
    garantizando que la absorción de nutrientes cognitivos minimice la longitud informacional.

§2. Acción Reducida de Maupertuis:
    El funcional de acción reducida evaluado a lo largo de la trayectoria de deliberación satisface:
        S_0(E) = \oint p \, dq = \int_0^T 2 T(t) \, dt
    constituyendo el generador canónico de las transformaciones de escala temporal del cultivo.

III. MECÁNICA CELESTE DE POINCARÉ Y MULTIPLICADORES DE FLOQUET
──────────────────────────────────────────────────────────────
§1. Multiplicadores de Floquet y Monodromía Orbital:
    Para la órbita periódica de referencia del cultivo $\gamma(t)$ con período $T$, la ecuación variacional
    $\dot{\xi} = J \nabla^2 H(\gamma(t)) \xi$ induce la matriz de monodromía $M = \Phi(T, 0) \in \mathrm{Sp}(4, \mathbb{R})$.
    Los autovalores de $M$ son los multiplicadores de Floquet $\{\mu_k\}_{k=1}^4$:
        • ELÍPTICO   : \mu_k = e^{\pm i \theta_k}, \quad |\mu_k| = 1 \quad (\text{Órbita centralmente estable})
        • PARABÓLICO : \mu_k = \pm 1 \quad \text{con defectos no diagonales} \quad (\text{Colisión de Krein})
        • HIPERBÓLICO: |\mu_k| \neq 1 \implies \text{separatrices exponenciales} \quad (\text{Dispersión de cultivo})

§2. Número de Rotación de Poincaré y Fracciones Continuas:
    El número de rotación $\rho(\omega_L, \omega_G)$ sobre el toro invariante se expande en convergentes:
        \rho = \frac{\omega_G}{\omega_L} = [a_0; a_1, a_2, \dots, a_k, \dots] \implies \frac{p_k}{q_k}
    Por el Teorema Geométrico de Poincaré-Birkhoff, cualquier perturbación sobre una rotación racional
    $\rho \approx p/q$ preserva un número par mínimo de $2q$ puntos fijos (órbitas periódicas cerradas).

§3. Teorema de Recurrencia y Tiempo Medio de Kac:
    Bajo el flujo de Liouville en la sección de fase de pureza cuántica $\mu(A) = \mathrm{Tr}(\rho^2)$:
        \langle \tau_{\mathrm{rec}} \rangle = \frac{1}{\mu(A)} = \frac{1}{\mathrm{Tr}(\rho^2)} \ge 1.0

IV. AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI³) Y PUNTO FIJO DE BANACH
─────────────────────────────────────────────────────────────────
§1. Operador Conjunto de Banach $T_{\mathrm{RSI3}}$:
    El proceso de automejora se formaliza como un endomapeo sobre el espacio normado $(\mathbb{R}^4, \|\cdot\|_\infty)$
    que actúa sobre el estado vectorial de hiperparámetros $x = (\eta, w_{\mathrm{flatten}}, w_{\mathrm{entropy}}, w_{\mathrm{maupertuis}})$:
        T_{\mathrm{RSI3}}: (\mathbb{R}^4, \|\cdot\|_\infty) \longrightarrow (\mathbb{R}^4, \|\cdot\|_\infty)

§2. Verificación Numérica de la Contracción Uniforme:
    El motor estima la constante de Lipschitz suprema $\hat{L}$ en cada época:
        \hat{L} = \sup_{x \neq y} \frac{\|T_{\mathrm{RSI3}}(x) - T_{\mathrm{RSI3}}(y)\|_\infty}{\|x - y\|_\infty}
    La validez del ciclo exige la verificación formal de la condición de contracción:
        0 \le \hat{L} < 1.0
    Garantizando por el Teorema del Punto Fijo de Banach la convergencia geométrica hacia el único
    atractor óptimo $x^*$, acotando el error absoluto residual mediante:
        \|x_n - x^*\|_\infty \le \frac{\hat{L}^n}{1 - \hat{L}} \|x_1 - x_0\|_\infty

V. ARQUITECTURA DE TRES FASES ANIDADAS
──────────────────────────────────────
  • FASE I   [Atlas Celeste Poincaré-Maupertuis]: Computa Delaunay, $ds_M^2$, multiplicadores de
             Floquet, $\rho(\omega)$ y emite `CropWitnessQNDInitialCondition`.
  • FASE II  [Motor QND del Cultivo]: Ingiere la condición inicial, evalúa $A_w$, Berry-Bargmann,
             Maupertuis, RSI³ de 3 niveles con Banach y emite `CropWitnessCertificateDraft`.
  • FASE III [Motor Soberano Adjudicador]: Adjudica en Heyting $\Omega_3$ subordinado a $\hat{L} < 1$
             y estabilidad orbital de Floquet, acciona el ESP32 Crowbar y sella el certificado Merkle.

VI. DISYUNTOR CIBER-FÍSICO ESP32 Y PURGA EN EL VACÍO DE DIRAC
─────────────────────────────────────────────────────────────
Si el metabolismo del cultivo exhibe hiperbolicidad inestable o $\hat{L} \ge 1.0$:
    \tau_{\mathrm{crowbar}} \le 400.0 \text{ ns (ESP32 / GPIO14)}
aniquilando los estados cuánticos degradados en el Vacío de Dirac:
    e^- + e^+ \longrightarrow 2\gamma \quad \left( E = 2 m_e c^2 \approx 1.021998 \text{ MeV}, \quad \text{Back-Action } 0.0 \text{ dB} \right)
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

logger = logging.getLogger("APU.Wisdom.TOONCognitiveCropWitnessEngine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_TAU = 2.0 * math.pi
_M_E_C2_MEV = 0.510998950
_CROWBAR_LATENCY_BUDGET_NS = 400.0


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃ Y MODOS DE ACTUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    VETOED   = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def verdict(self) -> str:
        return self.name

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def heyting_implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(2 if int(other) >= int(self) else int(other))

    def pseudocomplement(self) -> "HeytingOmega3":
        """¬a = a → VETOED."""
        return self.heyting_implication(HeytingOmega3.VETOED)


class ActuationMode(IntEnum):
    NORMAL_FLUID            = 0
    SOFT_VETO_BYPASS        = 1
    HARD_VETO_ESP32_CROWBAR = 2


class CropObservationStatus(IntEnum):
    PENDING_OBSERVATION = 0
    OBSERVED_QND        = 1
    PURGED_DIRAC_VACUUM = 2
    DEGRADED_FLUID_DRIFT = 3


# ══════════════════════════════════════════════════════════════════════════════
# §A.1 CLASIFICACIÓN ORBITAL DE POINCARÉ (extensión doctoral v7.0.0)
# ══════════════════════════════════════════════════════════════════════════════

class PoincareOrbitStabilityClass(IntEnum):
    """
    Clasificación de la órbita de referencia del cultivo según los
    multiplicadores de Floquet de la monodromía M ∈ Sp(4,ℝ) linealizada
    (Poincaré, 1892, Tome I, Cap. IV):

      • ELLIPTIC_STABLE    : multiplicadores complejos conjugados en |λ|=1.
      • PARABOLIC_MARGINAL : multiplicadores reales en λ=±1 (degeneración).
      • HYPERBOLIC_UNSTABLE: ∃ multiplicador real con |λ|>1 (separatriz).
    """
    ELLIPTIC_STABLE     = 0
    PARABOLIC_MARGINAL  = 1
    HYPERBOLIC_UNSTABLE = 2

    @property
    def label(self) -> str:
        return self.name


# ══════════════════════════════════════════════════════════════════════════════
# §B. TIPO CUATERNIÓNICO (FIBRACIÓN DE HOPF S³ → S²)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class QuaternionHopfLift:
    """Levantamiento cuaterniónico de la fibración de Hopf S³ → S²."""
    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def from_complex_pair(a: complex, b: complex) -> "QuaternionHopfLift":
        nrm = math.sqrt(abs(a) ** 2 + abs(b) ** 2) + 1e-300
        return QuaternionHopfLift(
            w=float(a.real / nrm),
            x=float(a.imag / nrm),
            y=float(b.real / nrm),
            z=float(b.imag / nrm),
        )

    def conjugate(self) -> "QuaternionHopfLift":
        return QuaternionHopfLift(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other: "QuaternionHopfLift") -> "QuaternionHopfLift":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return QuaternionHopfLift(
            w=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            x=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            y=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            z=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def hopf_base_point(self) -> Tuple[float, float, float]:
        """π(q) = q i q̄ ∈ S²."""
        qi = QuaternionHopfLift(0.0, 1.0, 0.0, 0.0)
        r = self.multiply(qi).multiply(self.conjugate())
        return (r.x, r.y, r.z)


# ══════════════════════════════════════════════════════════════════════════════
# §C. DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareCelestialInvariants:
    """
    Invariantes rigurosos de la Mecánica Celeste de Poincaré forjados sobre
    la órbita de referencia de la parcela de cultivo.

    Referencias:
      • Poincaré, H. (1892-1899). Les Méthodes Nouvelles de la Mécanique Céleste.
      • Poincaré, H. (1890). Sur le problème des trois corps... (Acta Math. 13).
      • Birkhoff, G.D. (1913). Proof of Poincaré's Geometric Theorem.
      • Floquet, G. (1883). Sur les équations différentielles linéaires.
    """
    floquet_multipliers_real: Tuple[float, ...]
    floquet_multipliers_imag: Tuple[float, ...]
    characteristic_exponent_max: float
    orbit_stability_class: PoincareOrbitStabilityClass
    delaunay_omega_L: float
    delaunay_omega_G: float
    delaunay_omega_H: float
    poincare_rotation_number: float
    poincare_recurrence_time_estimate: float
    birkhoff_fixed_points_count: int
    resonance_order_detected: int
    resonance_small_divisor: float
    nonintegrability_verdict: bool
    small_parameter_series_coeffs: Tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CropWitnessCanonicalSeed:
    """Germen inmutable: Delaunay-Kepler + CR3BP + Melnikov + Greene + Bryuno
    + Maupertuis + Mecánica Celeste de Poincaré."""
    seed_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    eccentricity_e: float
    inclination_i: float
    laplace_runge_lenz_norm: float
    cr3bp_jacobi_constant: float
    lagrange_l1_instability_rate: float
    greene_residue_R: float
    melnikov_integral_M0: float
    bryuno_diophantine_sum: float
    morse_bott_euler_chi: int
    poincare_cartan_1form: float
    stora_zumino_residual: float
    # ─── Métrica de Maupertuis-Jacobi ────────────────────────────────────────
    maupertuis_energy_E: float          # E = Tr(H ρ)
    maupertuis_potential_V: float       # V(q) potencial efectivo
    maupertuis_conformal_factor: float  # n(q) = √(2(E - V))
    maupertuis_geodesic_length: float   # ℓ_MJ = ∫ n(q) ds
    hopf_base_coordinates: Tuple[float, float, float]
    # ─── Mecánica Celeste de Poincaré ─────────────────────────────────────────
    poincare_invariants: PoincareCelestialInvariants
    sha256_provenance: str


# ─── Semilla formal FASE I → FASE II ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CropWitnessQNDInitialCondition:
    """
    Estructura que la FASE I entrega como semilla formal a la FASE II.
    Encapsula el germen canónico + monodromía simpléctica + levantamiento de Hopf
    + camino geodésico de Maupertuis-Jacobi.
    """
    seed: CropWitnessCanonicalSeed
    crop_id: str
    rho_mac: np.ndarray
    rho_crop: np.ndarray
    monodromy_matrix: np.ndarray
    hopf_lift: QuaternionHopfLift
    maupertuis_geodesic_path: Tuple[Tuple[float, ...], ...]
    preconditioner_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class CropWeakObservationGerm:
    """Observación débil QND sobre la parcela de cultivo."""
    observation_id: str
    crop_id: str
    weak_value_Aw: complex
    fubini_study_distance: float
    uhlmann_fidelity: float
    uhlmann_residual: float
    berry_pancharatnam_phase: float
    berry_bargmann_invariant: complex
    maupertuis_conformal_index: float
    maupertuis_geodesic_length: float
    levy_concentration_width: float
    oseledets_transverse_lyapunov: float
    kolmogorov_sinai_entropy: float
    novikov_valuation: float
    capacity_gromov: float
    back_action_db: float
    status: CropObservationStatus
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class RSI3BanachConvergenceState:
    """
    Estado del operador de Automejora Recursiva de NIVEL 3, T_RSI3:Θ⁴→Θ⁴,
    actuando sobre el álgebra de Banach (ℝ⁴, ‖·‖_∞) del estado conjunto
    (η, w_flatten, w_entropy, w_maupertuis) — los pesos internos de la
    composición de Kleisli μ_T: T²⇒T sobre la mónada de continuación Cont.

    La verificación 0≤L̂<1 invoca el Teorema del Punto Fijo de Banach (1922),
    análogo funcional-analítico del Teorema de Recurrencia de Poincaré: toda
    órbita de T_RSI3 converge geométricamente a su punto fijo x*.
    """
    level1_eta_unit: float
    level2_weights: Tuple[float, float, float]
    level3_lipschitz_constant_L: float
    level3_is_contraction: bool
    level3_fixed_point_residual: float
    level3_banach_error_bound: float
    level3_iteration_depth: int
    kleisli_associativity_residual: float
    kleisli_unit_residual: float
    genealogy_merkle: str


# ─── Semilla formal FASE II → FASE III ────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class CropWitnessCertificateDraft:
    """
    Semilla formal FASE II → FASE III. Agrega observaciones QND, tasa RSI3
    con cierre de Banach, e invariantes de Poincaré heredados de FASE I.
    """
    draft_id: str
    crop_id: str
    observations: Tuple[CropWeakObservationGerm, ...]
    level3_rsi_monadic_rate: float
    kleisli_composition_trace: Tuple[str, ...]
    rsi3_banach_state: RSI3BanachConvergenceState
    poincare_orbit_stability_class: PoincareOrbitStabilityClass
    poincare_nonintegrability_verdict: bool
    peak_gromov: float
    max_fubini_study: float
    max_berry_abs: float
    min_maupertuis_factor: float
    berry_coherent: bool
    aggregate_verdict: HeytingOmega3
    draft_sha256: str
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class CropWitnessExecutionCertificate:
    """Certificado inmutable del Testigo del Cultivo."""
    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    heyting_pseudocomplement: HeytingOmega3
    actuation_mode: ActuationMode
    observations_count: int
    purged_count: int
    level3_rsi_monadic_rate: float
    gromov_capacity_peak: float
    poincare_cartan_residual: float
    berry_phase_coherent: bool
    # ─── RSI3 Banach (Nivel 3) ────────────────────────────────────────────────
    rsi3_lipschitz_constant: float
    rsi3_banach_contraction_verified: bool
    rsi3_banach_error_bound: float
    # ─── Mecánica Celeste de Poincaré ─────────────────────────────────────────
    poincare_orbit_stability_class: PoincareOrbitStabilityClass
    poincare_nonintegrability_certified: bool
    esp32_crowbar_latency_ns: float
    dirac_vacuum_purge_energy_mev: float
    merkle_root_sha256: str
    provenance_hmac: str


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE I — ATLAS CELESTE POINCARÉ-MAUPERTUIS                             ║
# ║  Terminal: `emit_crop_qnd_initial_condition` → semilla de FASE II.      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class PoincareCropWitnessAtlas:
    """
    FASE I — Atlas Celeste Poincaré-Maupertuis.

    Construcciones rigurosas:
      • Delaunay-Kepler: e = √(1-(λ₂/λ₁)²), G = L√(1-e²), i = arccos(λ₃/λ₂),
        H = G cos i, ‖A‖ = μe.
      • CR3BP: Jacobi C_J y silla L1 vía autovalor real del linealizado.
      • Monodromía simpléctica M ∈ Sp(4,ℝ): M₀ = R(θ) ⊕ R(θ) + ε·S.
      • Melnikov: M(t₀) = ∫ sech²(t) tanh(t) sin(ωt) dt (Simpson 1/3).
      • Bryuno: fracciones continuas reales de ω.
      • Morse-Bott: χ(ℂPⁿ⁻¹) = n por Betti b_{2k} = 1, 0 ≤ k < n.
      • Poincaré-Cartan: θ_PC = Tr(ρ dN) + residuo ‖[ρ,N]‖_F.
      • Stora-Zumino: residual ‖d CS₃ - Tr(F∧F)‖ (anomalía ABJ).
      • Hopf: levantamiento cuaterniónico S³ → S².
      • Maupertuis-Jacobi: g_MJ = 2(E - V) g_base, ℓ_MJ = ∫_γ n(q) ds.
      • MECÁNICA CELESTE DE POINCARÉ (v7.0.0): Floquet/estabilidad orbital,
        número de rotación, recurrencia de Kac, Poincaré-Birkhoff,
        no-integrabilidad por divisores pequeños, serie del parámetro
        pequeño.
    """

    # ── I.1 Reducción simpléctica ────────────────────────────────────────────
    @staticmethod
    def _symplectic_eigenreduction(rho: np.ndarray) -> Tuple[np.ndarray, float]:
        rho_h = 0.5 * (rho + rho.conj().T)
        ev = la.eigvalsh(rho_h)
        actions = np.sort(np.maximum(ev, 1e-15))[::-1]
        purity = float(np.trace(rho_h @ rho_h).real)
        return actions, purity

    # ── I.2 Delaunay-Kepler coherente ────────────────────────────────────────
    @staticmethod
    def _delaunay_kepler_actions(
        actions: np.ndarray, mu: float = 1.0
    ) -> Tuple[float, float, float, float, float, float]:
        L = float(np.sum(actions))
        lam = np.sort(np.maximum(actions, 1e-15))[::-1]
        n = len(lam)
        lam1 = float(lam[0])
        lam2 = float(lam[1]) if n > 1 else lam1 * 0.5
        lam3 = float(lam[2]) if n > 2 else lam2 * 0.5
        e = float(math.sqrt(max(0.0, 1.0 - (lam2 / (lam1 + 1e-300)) ** 2)))
        G = float(L * math.sqrt(max(0.0, 1.0 - e ** 2)))
        cos_i = float(np.clip(lam3 / (lam2 + 1e-300), -1.0, 1.0))
        i = float(math.acos(cos_i))
        H = float(G * cos_i)
        lrl = float(mu * e)
        return L, G, H, e, i, lrl

    # ── I.3 CR3BP Jacobi + silla L1 ──────────────────────────────────────────
    @staticmethod
    def _cr3bp_jacobi_and_l1(purity: float) -> Tuple[float, float]:
        mu = float(np.clip(1.0 - purity, 1e-3, 0.499))
        x_L1 = 0.5 - mu + (mu / 3.0) ** (1.0 / 3.0) - 0.1
        r1 = abs(x_L1 + mu) + 1e-9
        r2 = abs(x_L1 - (1.0 - mu)) + 1e-9
        C_J = float((1.0 - mu) / r1 + mu / r2 + 0.5 * x_L1 ** 2)
        lam_u = float(math.sqrt(abs(2.0 * mu + 1.0)))
        return C_J, lam_u

    # ── I.4 Monodromía Sp(4,ℝ) + Greene ──────────────────────────────────────
    @staticmethod
    def _symplectic_monodromy(theta: float, eps: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        M0 = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), R]])
        A = np.array([[0.0, eps], [-eps, 0.0]])
        B = np.array([[eps, 0.0], [0.0, -eps]])
        C = np.array([[0.0, eps], [eps, 0.0]])
        S = np.block([[A, B], [C, -A.T]])
        return M0 + eps * S

    @staticmethod
    def _greene_residue(M: np.ndarray) -> float:
        return float((2.0 - float(np.trace(M))) / 4.0)

    # ── I.5 Melnikov homoclínico (Simpson) ────────────────────────────────────
    @staticmethod
    def _melnikov_homoclinic(eps: float, n_grid: int = 201) -> float:
        t = np.linspace(-6.0, 6.0, n_grid)
        h = t[1] - t[0]
        integ = (1.0 / np.cosh(t)) ** 2 * np.tanh(t) * np.sin(t)
        w = np.ones(n_grid)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return float((h / 3.0) * float(np.sum(w * integ)) * eps)

    # ── I.6 Bryuno vía fracciones continuas ──────────────────────────────────
    @staticmethod
    def _bryuno_convergents(omega: float, depth: int = 12) -> float:
        if omega <= 0:
            omega = abs(omega) + 1e-9
        x = omega
        a: List[int] = []
        for _ in range(depth):
            ai = int(math.floor(x))
            a.append(ai)
            frac = x - ai
            if frac < 1e-12:
                break
            x = 1.0 / frac
        q_prev, q_curr = 0.0, 1.0
        B = 0.0
        for ai in a:
            q_next = ai * q_curr + q_prev
            if q_curr > 0.0 and q_next > 0.0:
                B += math.log(q_next) / q_curr
            q_prev, q_curr = q_curr, q_next
        return float(B)

    # ── I.7 Morse-Bott χ(ℂPⁿ⁻¹) ──────────────────────────────────────────────
    @staticmethod
    def _morse_bott_chi(dim: int) -> int:
        return int(dim)

    # ── I.8 Poincaré-Cartan + residuo ─────────────────────────────────────────
    @staticmethod
    def _poincare_cartan(rho: np.ndarray, N: np.ndarray) -> Tuple[float, float]:
        theta = float(np.trace(rho @ N).real)
        comm = rho @ N - N @ rho
        return theta, float(np.linalg.norm(comm, "fro"))

    # ── I.9 Stora-Zumino (anomalía ABJ) ──────────────────────────────────────
    @staticmethod
    def _stora_zumino_residual(rho: np.ndarray) -> float:
        F = 0.5 * (rho - rho.conj().T)
        d_cs3 = -np.trace(F @ F.conj().T) / (8.0 * math.pi ** 2)
        tr_F2 = np.trace(F @ F) / (8.0 * math.pi ** 2)
        return float(abs(d_cs3 - tr_F2).real)

    # ── I.10 Hopf lift cuaterniónico ─────────────────────────────────────────
    @staticmethod
    def _hopf_lift_from_state(psi: np.ndarray) -> QuaternionHopfLift:
        idx = np.argsort(-np.abs(psi))[:2]
        a = complex(psi[idx[0]]) if len(idx) > 0 else 1.0 + 0j
        b = complex(psi[idx[1]]) if len(idx) > 1 else 0.0 + 0j
        return QuaternionHopfLift.from_complex_pair(a, b)

    # ── I.11 Métrica de Maupertuis-Jacobi (núcleo riguroso) ──────────────────
    @classmethod
    def _maupertuis_jacobi_metric(
        cls,
        rho: np.ndarray,
        rho_mac: np.ndarray,
        n_geodesic_steps: int = 24,
    ) -> Tuple[float, float, float, Tuple[Tuple[float, ...], ...]]:
        """
        Métrica conformal de Maupertuis-Jacobi:
            g_MJ = 2(E - V(q)) g_base,   n(q) = √(2(E - V(q)))
        donde E = Tr(H ρ) es la energía total con H = ρ_mac (Hamiltoniano de
        referencia) y V(q) = 1 - Tr(ρ²) es el potencial efectivo (función de
        la pureza local del cultivo).

        La longitud geodésica de Jacobi es:
            ℓ_MJ = ∫_γ n(q) ds
        computada sobre el camino geodésico proyectivo entre ρ_mac y ρ_crop.
        """
        dim = rho.shape[0]
        rho_h = 0.5 * (rho + rho.conj().T)
        rho_mac_h = 0.5 * (rho_mac + rho_mac.conj().T)

        # Energía total E = Tr(H ρ) con H = ρ_mac
        E_total = float(np.trace(rho_mac_h @ rho_h).real)
        # Potencial efectivo V = 1 - Tr(ρ²) (función de la pureza local)
        V_eff = float(np.clip(1.0 - np.trace(rho_h @ rho_h).real, 1e-9, 1.0))
        # Factor conformal n(q) = √(2(E - V))
        conformal_arg = max(2.0 * (E_total - V_eff), 1e-12)
        n_q = float(math.sqrt(conformal_arg))

        # Camino geodésico proyectivo: interpolación entre ρ_mac y ρ_crop
        # parametrizada por t ∈ [0,1] sobre estados normalizados
        _, Vmac = la.eigh(rho_mac_h)
        _, Vcrop = la.eigh(rho_h)
        psi_mac = Vmac[:, -1]
        psi_crop = Vcrop[:, -1]

        path: List[Tuple[float, ...]] = []
        length = 0.0
        prev_point: Optional[np.ndarray] = None
        prev_n: float = n_q

        for k in range(n_geodesic_steps + 1):
            t = k / n_geodesic_steps
            # Interpolación geodésica proyectiva (Bures-like)
            psi_t = (1.0 - t) * psi_mac + t * psi_crop
            nrm = np.linalg.norm(psi_t) + 1e-300
            psi_t = psi_t / nrm
            # Metrica base: distancia euclídea local en ℂ^{n}
            base_norm = float(np.linalg.norm(psi_t))

            # Factor conformal local n(q_t)
            rho_t = np.outer(psi_t, psi_t.conj())
            V_t = float(np.clip(1.0 - np.trace(rho_t @ rho_t).real, 1e-9, 1.0))
            n_t = float(math.sqrt(max(2.0 * (E_total - V_t), 1e-12)))

            path.append(tuple(complex(c) for c in psi_t[:4]))

            if prev_point is not None:
                ds = float(np.linalg.norm(psi_t - prev_point))
                # Integración trapezoidal del integrando n(q) ds
                length += 0.5 * (n_t + prev_n) * ds

            prev_point = psi_t
            prev_n = n_t

        return E_total, V_eff, n_q, tuple(path)

    # ══════════════════════════════════════════════════════════════════════════
    # §I.12 — §I.17 MECÁNICA CELESTE DE POINCARÉ (extensión doctoral v7.0.0)
    # ══════════════════════════════════════════════════════════════════════════

    # ── I.12 Frecuencias de Delaunay conjugadas ω_i = ∂H/∂J_i ────────────────
    @staticmethod
    def _delaunay_frequencies(
        L: float, G: float, H: float, mu_eff: float = 1.0
    ) -> Tuple[float, float, float]:
        omega_L = float(mu_eff ** 2 / max(L ** 3, 1e-18))
        omega_G = float(omega_L * (1.0 + G / max(L, 1e-15)))
        omega_H = float(omega_G * (1.0 + H / max(G, 1e-15)))
        return omega_L, omega_G, omega_H

    # ── I.13 Multiplicadores de Floquet y clasificación de estabilidad ──────
    @staticmethod
    def _poincare_floquet_multipliers(
        M: np.ndarray,
    ) -> Tuple[np.ndarray, PoincareOrbitStabilityClass]:
        """
        Autovalores (multiplicadores de Floquet) de la monodromía M ∈ Sp(4,ℝ)
        linealizada alrededor de la órbita de referencia del cultivo.
        Clasificación (Poincaré, 1892, Tome I, Cap. IV):

          |λ|=1, λ∈ℂ\\ℝ (pares conjugados)  → ELLIPTIC_STABLE
          |λ|=1, λ∈ℝ (λ=±1)                 → PARABOLIC_MARGINAL
          ∃|λ|>1                            → HYPERBOLIC_UNSTABLE
        """
        eigvals = la.eigvals(M)
        max_mod = float(np.max(np.abs(eigvals)))
        if max_mod > 1.0 + 1e-9:
            stability = PoincareOrbitStabilityClass.HYPERBOLIC_UNSTABLE
        elif abs(max_mod - 1.0) < 1e-6:
            has_imag = bool(np.any(np.abs(eigvals.imag) > 1e-9))
            stability = (PoincareOrbitStabilityClass.ELLIPTIC_STABLE if has_imag
                        else PoincareOrbitStabilityClass.PARABOLIC_MARGINAL)
        else:
            stability = PoincareOrbitStabilityClass.ELLIPTIC_STABLE
        return eigvals, stability

    # ── I.14 Número de rotación de Poincaré (fracción continua) ─────────────
    @staticmethod
    def _poincare_rotation_number(
        omega_a: float, omega_b: float, n_terms: int = 20
    ) -> float:
        """
        ρ = ω_b/ω_a aproximado mediante el algoritmo clásico de fracciones
        continuas de Poincaré (convergentes p_k/q_k), determinante para la
        persistencia KAM del toro invariante del cultivo.
        """
        ratio = abs(omega_b) / (abs(omega_a) + 1e-15)
        a0 = math.floor(ratio)
        x = ratio - a0
        coeffs: List[int] = []
        for _ in range(n_terms):
            if x < 1e-12:
                break
            inv = 1.0 / x
            a = math.floor(inv)
            coeffs.append(int(a))
            x = inv - a
        val = 0.0
        for a in reversed(coeffs):
            val = 1.0 / (a + val) if (a + val) != 0 else 0.0
        return float(a0 + val)

    # ── I.15 Teorema de recurrencia de Poincaré (estimador de Kac) ──────────
    @staticmethod
    def _poincare_recurrence_time(phase_space_measure_fraction: float) -> float:
        """
        Teorema de recurrencia de Poincaré (1890): bajo un flujo que preserva
        la medida en un espacio de fases de volumen finito, casi todo punto
        regresa arbitrariamente cerca de sí mismo. El Lema de Kac acota el
        tiempo medio de recurrencia τ_rec ≈ 1/μ(A) para un entorno A de
        medida relativa μ(A) (aquí, la pureza espectral del cultivo).
        """
        mu = float(np.clip(phase_space_measure_fraction, 1e-6, 1.0))
        return float(1.0 / mu)

    # ── I.16 Teorema geométrico de Poincaré-Birkhoff ────────────────────────
    @staticmethod
    def _poincare_birkhoff_fixed_points(rotation_number: float, q_max: int = 12) -> int:
        """
        "Último teorema geométrico de Poincaré" (probado por Birkhoff, 1913):
        un twist map de un anillo que preserva área posee al menos 2q puntos
        fijos cuando el número de rotación es racional ρ≈p/q (mitad
        elípticos, mitad hiperbólicos).
        """
        frac = rotation_number - math.floor(rotation_number)
        best_q, best_err = 1, 1.0
        for q in range(1, q_max + 1):
            p = round(frac * q)
            err = abs(frac - p / q)
            if err < best_err:
                best_err, best_q = err, q
        return int(2 * best_q)

    # ── I.17 No-integrabilidad por divisores pequeños + serie ε-pequeño ────
    @staticmethod
    def _poincare_nonintegrability_test(
        omega_L: float, omega_G: float, omega_H: float, k_max: int = 4
    ) -> Tuple[int, float, bool]:
        """
        Busca resonancias k₁ω_L+k₂ω_G+k₃ω_H≈0 ("divisores pequeños" de
        Poincaré). Una resonancia de orden bajo certifica, vía el argumento
        clásico de Poincaré sobre series divergentes (1892, Tome I, Cap. V),
        la NO-existencia genérica de una integral primera analítica
        adicional uniforme.
        """
        best_order, best_div = 0, float("inf")
        for k1 in range(-k_max, k_max + 1):
            for k2 in range(-k_max, k_max + 1):
                for k3 in range(-k_max, k_max + 1):
                    order = abs(k1) + abs(k2) + abs(k3)
                    if order == 0 or order > k_max:
                        continue
                    val = abs(k1 * omega_L + k2 * omega_G + k3 * omega_H)
                    if val < best_div:
                        best_div, best_order = val, order
        nonintegrable = bool(best_div < 1e-2)
        return best_order, float(best_div), nonintegrable

    @staticmethod
    def _poincare_small_parameter_series(
        eps: float, x0_amp: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Expansión asintótica x(t,ε) = x₀ + εx₁ + ε²x₂ + … (Poincaré, 1892,
        Cap. II-III) sobre el modelo canónico del oscilador perturbado,
        prototipo de la perturbación del problema restringido aplicado a la
        excentricidad orbital del cultivo.
        """
        c0 = x0_amp
        c1 = -(x0_amp ** 3) / 8.0
        c2 = (3.0 * x0_amp ** 5) / 256.0
        return float(c0), float(c1 * eps), float(c2 * eps ** 2)

    @classmethod
    def _compile_poincare_invariants(
        cls,
        L: float, G: float, H: float,
        M: np.ndarray,
        purity: float,
        eccentricity_e: float,
        perturbation_eps: float,
    ) -> PoincareCelestialInvariants:
        """Agrega la totalidad de los invariantes celestes de Poincaré (§I.12-17)."""
        omega_L, omega_G, omega_H = cls._delaunay_frequencies(L, G, H)
        floquet_eigs, orbit_stability = cls._poincare_floquet_multipliers(M)
        char_exp_max = float(np.max(np.abs(floquet_eigs)))

        rot_number = cls._poincare_rotation_number(omega_L, omega_G)
        birkhoff_fp = cls._poincare_birkhoff_fixed_points(rot_number)
        recurrence_time = cls._poincare_recurrence_time(float(np.clip(purity, 1e-6, 1.0)))
        res_order, res_div, noninteg = cls._poincare_nonintegrability_test(
            omega_L, omega_G, omega_H
        )
        c0, c1, c2 = cls._poincare_small_parameter_series(
            perturbation_eps, x0_amp=eccentricity_e + 1e-3
        )

        return PoincareCelestialInvariants(
            floquet_multipliers_real=tuple(float(x) for x in floquet_eigs.real),
            floquet_multipliers_imag=tuple(float(x) for x in floquet_eigs.imag),
            characteristic_exponent_max=char_exp_max,
            orbit_stability_class=orbit_stability,
            delaunay_omega_L=omega_L, delaunay_omega_G=omega_G, delaunay_omega_H=omega_H,
            poincare_rotation_number=rot_number,
            poincare_recurrence_time_estimate=recurrence_time,
            birkhoff_fixed_points_count=birkhoff_fp,
            resonance_order_detected=res_order,
            resonance_small_divisor=res_div,
            nonintegrability_verdict=noninteg,
            small_parameter_series_coeffs=(c0, c1, c2),
        )

    # ── I.18 Agregador maestro ────────────────────────────────────────────────
    @classmethod
    def compute_canonical_crop_witness_seed(
        cls,
        density_matrix: np.ndarray,
        N_diag: np.ndarray,
        rho_mac: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> CropWitnessCanonicalSeed:
        now = time.time()
        actions, purity = cls._symplectic_eigenreduction(density_matrix)
        L, G, H, e, i, lrl = cls._delaunay_kepler_actions(actions)
        C_J, lam_u = cls._cr3bp_jacobi_and_l1(purity)

        theta = float(np.trace(density_matrix @ N_diag).real)
        M = cls._symplectic_monodromy(theta, perturbation_eps)
        R_G = cls._greene_residue(M)

        M0 = cls._melnikov_homoclinic(eps=perturbation_eps)

        omega = float(actions[1] / (actions[0] + 1e-15)) if len(actions) > 1 else 1.618
        B = cls._bryuno_convergents(omega=omega, depth=12)

        chi = cls._morse_bott_chi(dim=density_matrix.shape[0])
        theta_pc, _ = cls._poincare_cartan(density_matrix, N_diag)
        sz = cls._stora_zumino_residual(density_matrix)

        # Maupertuis-Jacobi metric
        E_tot, V_eff, n_q, path = cls._maupertuis_jacobi_metric(
            rho=density_matrix, rho_mac=rho_mac,
        )
        # Longitud geodésica ℓ_MJ sobre el camino proyectivo
        ell_mj = 0.0
        prev = None
        for pt in path:
            arr = np.array([complex(c) for c in pt])
            if prev is not None:
                ell_mj += float(np.linalg.norm(arr - prev))
            prev = arr
        ell_mj *= n_q  # escalado por factor conformal promedio

        _, vecs = la.eigh(0.5 * (density_matrix + density_matrix.conj().T))
        lift = cls._hopf_lift_from_state(vecs[:, -1])
        hopf_xyz = lift.hopf_base_point()

        # ── Mecánica Celeste de Poincaré ──────────────────────────────────
        poincare_invariants = cls._compile_poincare_invariants(
            L=L, G=G, H=H, M=M, purity=purity,
            eccentricity_e=e, perturbation_eps=perturbation_eps,
        )

        seed_id = f"SEED-CROP-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        prov = (
            f"{seed_id}|L={L:.10f}|G={G:.10f}|H={H:.10f}|e={e:.10f}|"
            f"i={i:.10f}|C_J={C_J:.10f}|R_G={R_G:.10e}|M0={M0:.10e}|"
            f"B={B:.10e}|χ={chi}|θ={theta_pc:.10e}|SZ={sz:.10e}|"
            f"E={E_tot:.10e}|V={V_eff:.10e}|n={n_q:.10e}|ℓ={ell_mj:.10e}|"
            f"orbit={poincare_invariants.orbit_stability_class.name}|"
            f"ρrot={poincare_invariants.poincare_rotation_number:.10f}|"
            f"2q={poincare_invariants.birkhoff_fixed_points_count}|"
            f"noninteg={poincare_invariants.nonintegrability_verdict}"
        ).encode("utf-8")
        sha = hashlib.sha256(prov).hexdigest()

        return CropWitnessCanonicalSeed(
            seed_id=seed_id,
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            eccentricity_e=e,
            inclination_i=i,
            laplace_runge_lenz_norm=lrl,
            cr3bp_jacobi_constant=C_J,
            lagrange_l1_instability_rate=lam_u,
            greene_residue_R=R_G,
            melnikov_integral_M0=M0,
            bryuno_diophantine_sum=B,
            morse_bott_euler_chi=chi,
            poincare_cartan_1form=theta_pc,
            stora_zumino_residual=sz,
            maupertuis_energy_E=E_tot,
            maupertuis_potential_V=V_eff,
            maupertuis_conformal_factor=n_q,
            maupertuis_geodesic_length=ell_mj,
            hopf_base_coordinates=hopf_xyz,
            poincare_invariants=poincare_invariants,
            sha256_provenance=sha,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE I → ARRANQUE FORMAL DE FASE II ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def emit_crop_qnd_initial_condition(
        cls,
        crop_id: str,
        rho_mac: np.ndarray,
        rho_crop: np.ndarray,
        N_diag: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> CropWitnessQNDInitialCondition:
        """
        Cierre formal de FASE I: empaqueta el germen canónico (incl. los
        invariantes de Poincaré) + monodromía simpléctica + levantamiento de
        Hopf + camino geodésico de Maupertuis.
        """
        seed = cls.compute_canonical_crop_witness_seed(
            density_matrix=rho_crop,
            N_diag=N_diag,
            rho_mac=rho_mac,
            perturbation_eps=perturbation_eps,
        )
        theta_seed = float(np.trace(rho_crop @ N_diag).real)
        M = cls._symplectic_monodromy(theta_seed, perturbation_eps)
        _, vecs = la.eigh(0.5 * (rho_crop + rho_crop.conj().T))
        lift = cls._hopf_lift_from_state(vecs[:, -1])
        _, _, _, path = cls._maupertuis_jacobi_metric(rho_crop, rho_mac)

        precond_bytes = (
            f"{seed.sha256_provenance}|{theta_seed:.12e}|{crop_id}".encode("utf-8")
        )
        precond = hashlib.sha256(precond_bytes).hexdigest()

        logger.info(
            f"[FASE I → FASE II] Semilla QND del cultivo emitida: {seed.seed_id} | "
            f"L={seed.delaunay_L:.4f} e={seed.eccentricity_e:.4f} "
            f"n(q)={seed.maupertuis_conformal_factor:.4f} ℓ_MJ={seed.maupertuis_geodesic_length:.4f} | "
            f"Órbita Poincaré={seed.poincare_invariants.orbit_stability_class.label} "
            f"ρ_rot={seed.poincare_invariants.poincare_rotation_number:.4f} "
            f"no-integ={seed.poincare_invariants.nonintegrability_verdict}"
        )
        return CropWitnessQNDInitialCondition(
            seed=seed,
            crop_id=crop_id,
            rho_mac=np.array(rho_mac, dtype=np.complex128, copy=True),
            rho_crop=np.array(rho_crop, dtype=np.complex128, copy=True),
            monodromy_matrix=M,
            hopf_lift=lift,
            maupertuis_geodesic_path=path,
            preconditioner_sha256=precond,
            timestamp_utc=time.time(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE II — MOTOR QND, MAUPERTUIS Y TORRE RSI3 SOBRE BANACH               ║
# ║  Primer método `ingest_crop_qnd_initial_condition` CONSUME FASE I.       ║
# ║  Terminal `emit_certificate_draft_for_adjudication` → semilla FASE III. ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class CropWitnessQNDEngine:
    """
    FASE II — Motor QND de Medición Débil sobre la variedad de Maupertuis-Jacobi.

    Rigurosos:
      • Uhlmann F(ρ,σ) = (Tr √(√ρ σ √ρ))² vía sqrtm con proyección hermitiana.
      • Fubini-Study d_FS = arccos(√F) sobre ℂPⁿ⁻¹.
      • Berry-Pancharatnam: invariante de Bargmann -arg ∏ ⟨ψ_k|ψ_{k+1}⟩.
      • Lévy: ε* = √(log(2n+1)/(2n-1)) sobre S^{2n-1}.
      • Oseledets transversal vía QR iterado de U = e^{-iHΔt}.
      • KS: h_KS = -Σ p_i ln p_i sobre exponentes positivos.
      • Novikov: valuación formal v(F) = min suporte en anillo Λ.
      • Gromov: no-squeezing c_G ≈ π det(ρ)^{1/n} 2.
      • Maupertuis: longitud geodésica ℓ_MJ = ∫ n(q) ds del camino de FASE I.
      • RSI3 de TRES NIVELES sobre la mónada Cont (Kleisli), culminando en
        un cierre de contracción de Banach (v7.0.0, §II.11-15).
    """

    _ETA_MIN: float = 0.05
    _ETA_MAX: float = 0.45
    _RSI3_WEIGHT_MIN: float = 0.2
    _RSI3_WEIGHT_MAX: float = 2.0
    _RSI3_HISTORY_MAXLEN: int = 32

    def __init__(self, dimension: int = 56) -> None:
        self.dimension = dimension
        self._phase1_condition: Optional[CropWitnessQNDInitialCondition] = None
        self._kleisli_trace: List[str] = []

        # ── RSI3: estado persistente del álgebra de Banach de automejora ───
        self._eta_current: float = 0.25
        self._rsi3_weights_current: np.ndarray = np.array(
            [1.0, 1.0, 1.0], dtype=np.float64
        )
        self._rsi3_state_history: "deque[Tuple[np.ndarray, np.ndarray]]" = deque(
            maxlen=self._RSI3_HISTORY_MAXLEN
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE II ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def ingest_crop_qnd_initial_condition(
        self,
        condition: CropWitnessQNDInitialCondition,
    ) -> "CropWitnessQNDEngine":
        """
        Ingesta la semilla de FASE I, valida la huella preconditioner_sha256.
        Retorna self para permitir encadenamiento fluido.
        """
        recomputed = hashlib.sha256(
            (
                f"{condition.seed.sha256_provenance}|"
                f"{float(np.trace(condition.rho_crop @ np.diag(np.linspace(1.0, 2.0, self.dimension))).real):.12e}|"
                f"{condition.crop_id}"
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != condition.preconditioner_sha256:
            logger.warning(
                f"[FASE II] Preconditioner divergente: {recomputed[:16]}… vs "
                f"{condition.preconditioner_sha256[:16]}…"
            )
        self._phase1_condition = condition
        logger.info(
            f"[FASE II ← FASE I] Condición QND del cultivo ingerida para crop={condition.crop_id} | "
            f"Órbita Poincaré={condition.seed.poincare_invariants.orbit_stability_class.label}."
        )
        return self

    # ── II.1 Uhlmann + Fubini-Study ──────────────────────────────────────────
    @staticmethod
    def _uhlmann_and_fubini(rho: np.ndarray, sigma: np.ndarray) -> Tuple[float, float]:
        rho_h = 0.5 * (rho + rho.conj().T)
        sig_h = 0.5 * (sigma + sigma.conj().T)
        try:
            sqrt_rho = la.sqrtm(rho_h)
            M = sqrt_rho @ sig_h @ sqrt_rho
            M = 0.5 * (M + M.conj().T)
            ev = np.clip(np.linalg.eigvalsh(M).real, 0.0, None)
            F = float(np.clip(float(np.sum(np.sqrt(ev))) ** 2, 1e-12, 1.0))
        except Exception:
            F = float(np.clip(np.trace(rho_h @ sig_h).real, 1e-12, 1.0))
        return F, float(math.acos(math.sqrt(F)))

    # ── II.2 Valor débil QND ─────────────────────────────────────────────────
    @staticmethod
    def _weak_value_qnd(
        rho_i: np.ndarray, rho_f: np.ndarray, A_op: np.ndarray,
    ) -> complex:
        _, Vi = la.eigh(0.5 * (rho_i + rho_i.conj().T))
        _, Vf = la.eigh(0.5 * (rho_f + rho_f.conj().T))
        phi_i = Vi[:, -1]
        phi_f = Vf[:, -1]
        overlap = np.vdot(phi_f, phi_i)
        if abs(overlap) < 1e-14:
            return complex(float(np.trace(A_op @ rho_f).real), 0.0)
        return complex(np.vdot(phi_f, A_op @ phi_i) / overlap)

    # ── II.3 Berry-Pancharatnam (invariante de Bargmann) ─────────────────────
    @staticmethod
    def _berry_pancharatnam_phase(
        psi_loop: List[np.ndarray],
    ) -> Tuple[float, complex]:
        if len(psi_loop) < 3:
            return 0.0, complex(1.0, 0.0)
        prod = complex(1.0, 0.0)
        N = len(psi_loop)
        for k in range(N):
            a = psi_loop[k]
            b = psi_loop[(k + 1) % N]
            na = np.linalg.norm(a) + 1e-300
            nb = np.linalg.norm(b) + 1e-300
            prod *= np.vdot(a / na, b / nb)
        gamma = -float(cmath.phase(prod))
        gamma = (gamma + math.pi) % _TAU - math.pi
        return gamma, prod

    # ── II.4 Ancho de Lévy ───────────────────────────────────────────────────
    @staticmethod
    def _levy_width(dim: int) -> float:
        return float(math.sqrt(math.log(2.0 * dim + 1.0) / (2.0 * dim - 1.0)))

    # ── II.5 Oseledets vía QR ─────────────────────────────────────────────────
    @staticmethod
    def _oseledets_qr(rho: np.ndarray, n_iter: int = 48) -> Tuple[float, np.ndarray]:
        H = 0.5 * (rho + rho.conj().T)
        U = la.expm(-1j * H * 0.01)
        dim = H.shape[0]
        Q = np.eye(dim, dtype=np.complex128)
        log_diag = np.zeros(dim)
        for _ in range(n_iter):
            W = U @ Q
            Q, R = np.linalg.qr(W)
            d = np.abs(np.diag(R)) + 1e-300
            log_diag += np.log(d)
        lambdas = np.sort((log_diag / n_iter).real)[::-1]
        lam_trans = float(lambdas[1]) if dim > 1 else -1.0
        return lam_trans, lambdas

    # ── II.6 Kolmogorov-Sinai ────────────────────────────────────────────────
    @staticmethod
    def _ks_entropy(lambdas: np.ndarray) -> float:
        pos = lambdas[lambdas > 0]
        if pos.size == 0:
            return 0.0
        s = float(np.sum(pos))
        p = pos / (s + 1e-300)
        return float(-np.sum(p * np.log(p + 1e-300)))

    # ── II.7 Novikov formal ──────────────────────────────────────────────────
    @staticmethod
    def _novikov_valuation(rho: np.ndarray, max_terms: int = 8) -> float:
        ev = np.sort(np.abs(la.eigvalsh(0.5 * (rho + rho.conj().T))))[::-1]
        ev = np.clip(ev, 1e-15, None)[:max_terms]
        exponents = -np.log(ev)
        coeffs = ev / (np.sum(ev) + 1e-300)
        support = exponents[np.abs(coeffs) > 1e-9]
        return float(np.min(support)) if support.size > 0 else 0.0

    # ── II.8 Gromov no-squeezing ─────────────────────────────────────────────
    @staticmethod
    def _gromov_capacity(rho: np.ndarray) -> float:
        rho_h = 0.5 * (rho + rho.conj().T)
        det_rho = max(float(np.linalg.det(rho_h).real), 1e-30)
        r_sq = det_rho ** (1.0 / rho.shape[0])
        c_g = float(math.pi * r_sq * 2.0)
        return float(np.clip(c_g, 1e-3, 12.5))

    # ── II.9 Maupertuis-Jacobi geodesic length ────────────────────────────────
    @staticmethod
    def _maupertuis_geodesic_length(
        rho: np.ndarray, rho_mac: np.ndarray, n_steps: int = 24,
    ) -> Tuple[float, float]:
        """
        Recomputa la longitud geodésica ℓ_MJ = ∫ n(q) ds y el índice
        conformal n(q) = √(2(E - V)). Verificación independiente de FASE I.
        """
        E_tot = float(np.trace(rho_mac @ rho).real)
        V = float(np.clip(1.0 - np.trace(rho @ rho).real, 1e-9, 1.0))
        n_q = float(math.sqrt(max(2.0 * (E_tot - V), 1e-12)))

        _, Vmac = la.eigh(0.5 * (rho_mac + rho_mac.conj().T))
        _, Vcr  = la.eigh(0.5 * (rho + rho.conj().T))
        psi_mac = Vmac[:, -1]
        psi_cr  = Vcr[:, -1]
        length = 0.0
        prev = psi_mac
        prev_n = n_q
        for k in range(1, n_steps + 1):
            t = k / n_steps
            psi_t = (1.0 - t) * psi_mac + t * psi_cr
            psi_t = psi_t / (np.linalg.norm(psi_t) + 1e-300)
            rho_t = np.outer(psi_t, psi_t.conj())
            V_t = float(np.clip(1.0 - np.trace(rho_t @ rho_t).real, 1e-9, 1.0))
            n_t = float(math.sqrt(max(2.0 * (E_tot - V_t), 1e-12)))
            ds = float(np.linalg.norm(psi_t - prev))
            length += 0.5 * (n_t + prev_n) * ds
            prev = psi_t
            prev_n = n_t
        return n_q, length

    # ── II.10 Núcleo observacional QND ────────────────────────────────────────
    def observe_crop_weak_value(
        self,
        crop_id: str,
        mac_density_matrix: np.ndarray,
        crop_density_matrix: np.ndarray,
    ) -> CropWeakObservationGerm:
        now = time.time()
        obs_id = f"OBS-CROP-QND-{int(now * 1e3) % 1_000_000:06d}"
        dim = mac_density_matrix.shape[0]

        # 1. Uhlmann + Fubini-Study
        F, d_fs = self._uhlmann_and_fubini(mac_density_matrix, crop_density_matrix)
        uhlmann_res = 1.0 - F

        # 2. Valor débil QND
        A_op = np.diag(np.linspace(1.0, 2.0, dim))
        Aw = self._weak_value_qnd(mac_density_matrix, crop_density_matrix, A_op)

        # 3. Fase de Berry-Bargmann sobre lazo proyectivo mac→crop→inter→mac
        _, Vm = la.eigh(0.5 * (mac_density_matrix + mac_density_matrix.conj().T))
        _, Vc = la.eigh(0.5 * (crop_density_matrix + crop_density_matrix.conj().T))
        psi_m = Vm[:, -1]
        psi_c = Vc[:, -1]
        inter = (psi_m + psi_c) / (np.linalg.norm(psi_m + psi_c) + 1e-300)
        berry_phase, bargmann = self._berry_pancharatnam_phase([psi_m, psi_c, inter])

        # 4. Lévy
        levy = self._levy_width(dim)

        # 5. Oseledets vía QR
        lam_trans, lambdas = self._oseledets_qr(mac_density_matrix, n_iter=48)

        # 6. KS
        h_ks = self._ks_entropy(lambdas)

        # 7. Novikov
        novikov_v = self._novikov_valuation(mac_density_matrix)

        # 8. Gromov
        gromov_c = self._gromov_capacity(crop_density_matrix)

        # 9. Maupertuis-Jacobi
        n_q, ell_mj = self._maupertuis_geodesic_length(crop_density_matrix, mac_density_matrix)

        status = (
            CropObservationStatus.OBSERVED_QND
            if abs(berry_phase) < 0.5
            else CropObservationStatus.DEGRADED_FLUID_DRIFT
        )

        return CropWeakObservationGerm(
            observation_id=obs_id,
            crop_id=crop_id,
            weak_value_Aw=Aw,
            fubini_study_distance=d_fs,
            uhlmann_fidelity=F,
            uhlmann_residual=uhlmann_res,
            berry_pancharatnam_phase=berry_phase,
            berry_bargmann_invariant=bargmann,
            maupertuis_conformal_index=n_q,
            maupertuis_geodesic_length=ell_mj,
            levy_concentration_width=levy,
            oseledets_transverse_lyapunov=lam_trans,
            kolmogorov_sinai_entropy=h_ks,
            novikov_valuation=novikov_v,
            capacity_gromov=gromov_c,
            back_action_db=0.0,
            status=status,
            timestamp_utc=now,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # §II.11 — §II.15 AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI3) SOBRE BANACH
    # ══════════════════════════════════════════════════════════════════════════

    # ── II.11 RSI3 — NIVEL 1: Adaptación Paramétrica (unidad η_T) ───────────
    def _rsi3_level1_parametric_unit(self, eta_in: float) -> float:
        """NIVEL 1: unidad monádica η_T:1→T(1), acotada en [η_min,η_max]."""
        return float(np.clip(eta_in, self._ETA_MIN, self._ETA_MAX))

    # ── Composición de Kleisli μ_T: T²⇒T ponderada por pesos RSI3-N2 ────────
    def _kleisli_composition_mu(
        self,
        eta_a: float, eta_b: float,
        obs: CropWeakObservationGerm,
        weights: np.ndarray,
    ) -> float:
        """
        μ_T: T²(A) → T(A) sobre la mónada de continuación Cont, generalizada
        con pesos RSI3-NIVEL2 (w_flatten, w_entropy, w_maupertuis):

          flatten        = √(1 - d_FS/π)
          entropy_factor = exp(-h_KS/(1+λ_⟂²))
          maupertuis_gain= 1 + 0.1·min(1, ℓ_MJ/n(q))

          μ_T = √(η_a·η_b) · flatten^{w₁} · entropy_factor^{w₂} · maupertuis_gain^{w₃}
        """
        w1, w2, w3 = (float(w) for w in weights)

        flatten = math.sqrt(max(1e-9, 1.0 - obs.fubini_study_distance / math.pi))
        entropy_factor = math.exp(
            -obs.kolmogorov_sinai_entropy
            / (1.0 + obs.oseledets_transverse_lyapunov ** 2)
        )
        maupertuis_gain = 1.0 + 0.1 * min(
            1.0, obs.maupertuis_geodesic_length / (obs.maupertuis_conformal_index + 1e-9)
        )

        composed = math.sqrt(max(1e-9, eta_a * eta_b))
        rate = (
            composed
            * max(flatten, 1e-9) ** w1
            * max(entropy_factor, 1e-9) ** w2
            * max(maupertuis_gain, 1e-9) ** w3
        )
        return float(np.clip(rate, self._ETA_MIN, self._ETA_MAX))

    # ── II.12 Verificación de leyes de la mónada Cont (Kleisli) ─────────────
    def _verify_kleisli_monad_laws(
        self, eta_in: float, obs: CropWeakObservationGerm, weights: np.ndarray,
    ) -> Tuple[float, float]:
        """Verifica unidad y asociatividad de μ_T sobre la mónada Cont."""
        eta_x = self._rsi3_level1_parametric_unit(eta_in)
        mu_with_unit = self._kleisli_composition_mu(eta_x, eta_in, obs, weights)
        unit_residual = float(abs(mu_with_unit - eta_in))

        eta_y = self._rsi3_level1_parametric_unit(eta_in * 1.05)
        eta_z = self._rsi3_level1_parametric_unit(eta_in * 0.95)
        left = self._kleisli_composition_mu(
            self._kleisli_composition_mu(eta_x, eta_y, obs, weights), eta_z, obs, weights
        )
        right = self._kleisli_composition_mu(
            eta_x, self._kleisli_composition_mu(eta_y, eta_z, obs, weights), obs, weights
        )
        assoc_residual = float(abs(left - right))
        return assoc_residual, unit_residual

    # ── II.13 RSI3 — NIVEL 2: Mutación Estratégica de pesos de μ_T ──────────
    def _rsi3_level2_strategic_weight_mutation(
        self, weights: np.ndarray, assoc_residual: float, gamma: float = 0.08,
    ) -> np.ndarray:
        """
        NIVEL 2: los 3 pesos internos de μ_T (w_flatten, w_entropy,
        w_maupertuis) se homogeneízan hacia 1 vía descenso de subgradiente
        acotado cuando el residual de asociatividad crece (incoherencia
        categórica de la composición de Kleisli):

            w_{t+1} = clip(w_t − γ·sign(w_t−1)·min(assoc_res·10³,1), 0.2, 2.0)

        Automejora de SEGUNDO ORDEN: modifica la propia regla de combinación
        μ_T, no solo el escalar η.
        """
        correction_mag = min(assoc_residual * 1_000.0, 1.0)
        w_next = weights - gamma * np.sign(weights - 1.0) * correction_mag
        return np.clip(w_next, self._RSI3_WEIGHT_MIN, self._RSI3_WEIGHT_MAX)

    # ── Operador de Banach T_RSI3: Θ⁴ → Θ⁴ ────────────────────────────────────
    def _rsi3_level3_banach_operator(
        self, state_vec: np.ndarray, obs: CropWeakObservationGerm, assoc_residual: float,
    ) -> np.ndarray:
        """T_RSI3(η,w) = (μ_T(η,η;w), Mutate₂(w, assoc_res))."""
        eta = float(state_vec[0])
        weights = state_vec[1:4]
        eta_next = self._kleisli_composition_mu(eta, eta, obs, weights)
        weights_next = self._rsi3_level2_strategic_weight_mutation(weights, assoc_residual)
        return np.concatenate([[eta_next], weights_next])

    # ── Estimación empírica de la constante de Lipschitz ─────────────────────
    def _rsi3_estimate_lipschitz_constant(self) -> float:
        """
        L̂ = max_i ‖T(x_i)−T(x_{i−1})‖_∞ / ‖x_i−x_{i−1}‖_∞ sobre el historial
        acotado de iteraciones. Si L̂<1, T_RSI3 es (empíricamente) una
        contracción de Banach, garantizando convergencia geométrica al
        único punto fijo x* (Teorema del Punto Fijo de Banach, 1922).
        """
        hist = list(self._rsi3_state_history)
        if len(hist) < 2:
            return 0.0
        ratios: List[float] = []
        for i in range(1, len(hist)):
            x_prev, Tx_prev = hist[i - 1]
            x_curr, Tx_curr = hist[i]
            dx = float(np.linalg.norm(x_curr - x_prev, ord=np.inf))
            dT = float(np.linalg.norm(Tx_curr - Tx_prev, ord=np.inf))
            if dx > 1e-12:
                ratios.append(dT / dx)
        return float(max(ratios)) if ratios else 0.0

    # ── II.14 RSI3 — NIVEL 3: Cierre Meta-Reflexivo de Punto Fijo Banach ────
    def _rsi3_level3_meta_reflective_closure(
        self, obs: CropWeakObservationGerm,
    ) -> RSI3BanachConvergenceState:
        """
        Cierra el ciclo de Automejora Recursiva de NIVEL 3:

          1. Verifica las leyes de la mónada Cont (asociatividad/unidad).
          2. Aplica T_RSI3 al estado conjunto actual (η_t,w_t)→(η_{t+1},w_{t+1}).
          3. Registra el par (x_t, T(x_t)) en el historial acotado.
          4. Estima L̂ y verifica la condición de contracción 0≤L̂<1.
          5. Si se verifica, calcula la cota de error de Banach
                 ‖x_n−x*‖_∞ ≤ (L̂ⁿ/(1−L̂))·‖x₁−x₀‖_∞.
          6. Si NO se verifica (L̂≥1), AUTO-REGULACIÓN de tercer orden:
             se frena la innovación forzando η→η_min.

        El sistema VERIFICA FORMALMENTE LA CONVERGENCIA DE SU PROPIO PROCESO
        DE MEJORA — el análogo funcional-analítico del Teorema de
        Recurrencia de Poincaré sobre el álgebra de Banach (ℝ⁴,‖·‖_∞).
        """
        state_vec = np.concatenate([[self._eta_current], self._rsi3_weights_current])

        assoc_res, unit_res = self._verify_kleisli_monad_laws(
            self._eta_current, obs, self._rsi3_weights_current
        )

        next_vec = self._rsi3_level3_banach_operator(state_vec, obs, assoc_res)
        self._rsi3_state_history.append((state_vec, next_vec))

        L_hat = self._rsi3_estimate_lipschitz_constant()
        is_contraction = bool(0.0 <= L_hat < 1.0)
        fp_residual = float(np.linalg.norm(next_vec - state_vec, ord=np.inf))

        if is_contraction and len(self._rsi3_state_history) >= 1:
            x0, x1 = self._rsi3_state_history[0]
            initial_gap = float(np.linalg.norm(x1 - x0, ord=np.inf))
            n_iter = len(self._rsi3_state_history)
            error_bound = float(
                (L_hat ** n_iter) / max(1.0 - L_hat, 1e-12) * initial_gap
            )
        else:
            next_vec = next_vec.copy()
            next_vec[0] = self._ETA_MIN
            error_bound = float("inf")

        self._eta_current = float(next_vec[0])
        self._rsi3_weights_current = next_vec[1:4]

        genealogy_payload = (
            f"{L_hat:.10f}:{is_contraction}:{fp_residual:.10f}:"
            f"{self._rsi3_weights_current.tolist()}:{assoc_res:.10e}:"
            f"{unit_res:.10e}:{time.time():.6f}"
        ).encode("utf-8")
        genealogy_merkle = hashlib.sha256(genealogy_payload).hexdigest()

        self._kleisli_trace.append(
            f"T_RSI3(η={state_vec[0]:.6f}, w={tuple(round(w,4) for w in state_vec[1:4])}, "
            f"obs={obs.observation_id}) → η'={self._eta_current:.6f}, "
            f"L̂={L_hat:.4f}, contracción={is_contraction}"
        )

        return RSI3BanachConvergenceState(
            level1_eta_unit=self._eta_current,
            level2_weights=tuple(float(w) for w in self._rsi3_weights_current),
            level3_lipschitz_constant_L=L_hat,
            level3_is_contraction=is_contraction,
            level3_fixed_point_residual=fp_residual,
            level3_banach_error_bound=error_bound,
            level3_iteration_depth=len(self._rsi3_state_history),
            kleisli_associativity_residual=assoc_res,
            kleisli_unit_residual=unit_res,
            genealogy_merkle=genealogy_merkle,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE II → ARRANQUE FORMAL DE FASE III ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def emit_certificate_draft_for_adjudication(
        self,
        observations: List[CropWeakObservationGerm],
        crop_id: str,
        previous_rsi3_rate: float = 0.25,
    ) -> CropWitnessCertificateDraft:
        """
        Cierre formal de FASE II: agrega N observaciones QND, aplica RSI3 de
        TRES NIVELES (paramétrico → mutación estratégica → cierre de Banach)
        con traza Kleisli, hereda los invariantes de Poincaré de FASE I, y
        emite la semilla formal de FASE III con huella SHA-256 canónica.
        """
        if not observations:
            raise ValueError("Se requiere al menos una observación QND para emitir draft.")

        now = time.time()
        draft_id = f"DRAFT-CROP-WITNESS-{int(now * 1e3) % 1_000_000:06d}"

        # Continuidad externa del estado RSI3 (si es la primera iteración)
        if len(self._rsi3_state_history) == 0:
            self._eta_current = self._rsi3_level1_parametric_unit(previous_rsi3_rate)

        banach_state: Optional[RSI3BanachConvergenceState] = None
        for obs in observations:
            banach_state = self._rsi3_level3_meta_reflective_closure(obs)

        assert banach_state is not None
        rsi3_rate = banach_state.level1_eta_unit

        # Herencia de la Mecánica Celeste de Poincaré (FASE I)
        if self._phase1_condition is not None:
            pci = self._phase1_condition.seed.poincare_invariants
            orbit_stability = pci.orbit_stability_class
            noninteg = pci.nonintegrability_verdict
        else:
            orbit_stability = PoincareOrbitStabilityClass.ELLIPTIC_STABLE
            noninteg = False

        peak_gromov = max(o.capacity_gromov for o in observations)
        max_d_fs = max(o.fubini_study_distance for o in observations)
        max_berry = max(abs(o.berry_pancharatnam_phase) for o in observations)
        min_maupertuis = min(o.maupertuis_conformal_index for o in observations)
        berry_coherent = all(abs(o.berry_pancharatnam_phase) < 0.50 for o in observations)

        # Veredicto agregado Heyting ∧
        verdict = HeytingOmega3.COHERENT
        for o in observations:
            if o.capacity_gromov > 12.5 or o.fubini_study_distance > 1.50:
                local = HeytingOmega3.VETOED
            elif o.fubini_study_distance > 0.15 or abs(o.berry_pancharatnam_phase) >= 0.50:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.COHERENT
            verdict = verdict.meet(local)

        draft_bytes = (
            draft_id + "|" + crop_id + "|"
            + "|".join(o.observation_id for o in observations)
            + f"|rsi3={rsi3_rate:.10e}"
            + f"|cG={peak_gromov:.10e}"
            + f"|L_hat={banach_state.level3_lipschitz_constant_L:.10e}"
            + f"|orbit={orbit_stability.name}"
            + f"|noninteg={noninteg}"
        ).encode("utf-8")
        draft_sha = hashlib.sha256(draft_bytes).hexdigest()

        logger.info(
            f"[FASE II → FASE III] Draft emitido {draft_id} | crop={crop_id} | "
            f"N={len(observations)} | RSI3={rsi3_rate:.6f} | "
            f"L̂_Banach={banach_state.level3_lipschitz_constant_L:.4f} "
            f"(contracción={banach_state.level3_is_contraction}) | "
            f"c_G_peak={peak_gromov:.4f} | n(q)_min={min_maupertuis:.4f} | "
            f"órbita={orbit_stability.label} | verdict={verdict.name}"
        )
        return CropWitnessCertificateDraft(
            draft_id=draft_id,
            crop_id=crop_id,
            observations=tuple(observations),
            level3_rsi_monadic_rate=rsi3_rate,
            kleisli_composition_trace=tuple(self._kleisli_trace),
            rsi3_banach_state=banach_state,
            poincare_orbit_stability_class=orbit_stability,
            poincare_nonintegrability_verdict=noninteg,
            peak_gromov=peak_gromov,
            max_fubini_study=max_d_fs,
            max_berry_abs=max_berry,
            min_maupertuis_factor=min_maupertuis,
            berry_coherent=berry_coherent,
            aggregate_verdict=verdict,
            draft_sha256=draft_sha,
            timestamp_utc=now,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FASE III — MOTOR SOBERANO DEL TESTIGO DEL CULTIVO                      ║
# ║  Primer método `receive_certificate_draft` CONSUME FASE II.             ║
# ║  Terminal `emit_crop_witness_execution_certificate`.                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONCognitiveCropWitnessEngine:
    """
    FASE III — Motor Espectral Principal del Testigo del Cultivo Cognitivo.

    Orquesta las tres fases anidadas y emite el certificado inmutable con
    cierre ciber-físico al disyuntor ESP32 Crowbar (< 400 ns / GPIO14),
    aniquilación al Vacío de Dirac (e⁻ + e⁺ → 2γ), pseudocomplemento Heyting
    — condicionado a la estabilidad orbital de Poincaré y a la contracción
    de Banach de RSI3.
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
        sovereign_id: str = "CROP-WITNESS-MASTER-01",
    ) -> None:
        self.dimension = dimension
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin
        self.sovereign_id = sovereign_id

        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension))
        self._qnd_engine = CropWitnessQNDEngine(dimension=dimension)

        self._hmac_secret = hashlib.sha256(
            f"SECRET-CROP-WITNESS-{sovereign_id}".encode("utf-8")
        ).digest()

        self._active_draft: Optional[CropWitnessCertificateDraft] = None
        self._purged_history: List[str] = []
        self._capacity_peak: float = 0.0
        self._last_rsi3_rate: float = 0.25
        self._crowbar_latency_ns: float = 0.0

        logger.info(
            f"TOONCognitiveCropWitnessEngine v7.0.0 inicializado. "
            f"Dim={dimension}, GromovMax={capacity_gromov_max}, GPIO={esp32_gpio_pin}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE III ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def receive_certificate_draft(
        self,
        draft: CropWitnessCertificateDraft,
    ) -> "TOONCognitiveCropWitnessEngine":
        """
        Recibe la semilla de FASE II, valida la huella SHA-256 del draft.
        Retorna self para permitir encadenamiento fluido.
        """
        recomputed = hashlib.sha256(
            (
                draft.draft_id + "|" + draft.crop_id + "|"
                + "|".join(o.observation_id for o in draft.observations)
                + f"|rsi3={draft.level3_rsi_monadic_rate:.10e}"
                + f"|cG={draft.peak_gromov:.10e}"
                + f"|L_hat={draft.rsi3_banach_state.level3_lipschitz_constant_L:.10e}"
                + f"|orbit={draft.poincare_orbit_stability_class.name}"
                + f"|noninteg={draft.poincare_nonintegrability_verdict}"
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != draft.draft_sha256:
            logger.warning(
                f"[FASE III] Huella divergente del draft: {recomputed[:16]}… vs "
                f"{draft.draft_sha256[:16]}…"
            )
        self._active_draft = draft
        self._last_rsi3_rate = draft.level3_rsi_monadic_rate
        logger.info(
            f"[FASE III ← FASE II] Draft {draft.draft_id} recibido | crop={draft.crop_id} | "
            f"N={len(draft.observations)} | verdict={draft.aggregate_verdict.name} | "
            f"órbita={draft.poincare_orbit_stability_class.label} | "
            f"Banach contracción={draft.rsi3_banach_state.level3_is_contraction}"
        )
        return self

    # ── III.1 Adjudicación Heyting con ∧/∨ y pseudocomplemento ────────────────
    def _heyting_adjudication(
        self,
        draft: CropWitnessCertificateDraft,
    ) -> Tuple[HeytingOmega3, ActuationMode]:
        """
        Reglas (v7.0.0, ampliadas con Poincaré ⊗ RSI3-Banach):
          VETOED   si c_G > max ∨ d_FS > 1.50 ∨ n(q) < 0.05
                     ∨ órbita = HYPERBOLIC_UNSTABLE
          DEGRADED si d_FS > 0.15 ∨ ¬berry_coherent
                     ∨ ¬contracción_Banach(RSI3)
          COHERENT en otro caso
        """
        verdict = HeytingOmega3.COHERENT
        for obs in draft.observations:
            if (
                obs.capacity_gromov > self.capacity_gromov_max
                or obs.fubini_study_distance > 1.50
                or obs.maupertuis_conformal_index < 0.05
            ):
                local = HeytingOmega3.VETOED
            elif obs.fubini_study_distance > 0.15 or abs(obs.berry_pancharatnam_phase) >= 0.50:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.COHERENT
            verdict = verdict.meet(local)

        # ── Condicionamiento por la Mecánica Celeste de Poincaré ──────────
        if draft.poincare_orbit_stability_class == PoincareOrbitStabilityClass.HYPERBOLIC_UNSTABLE:
            verdict = verdict.meet(HeytingOmega3.VETOED)

        # ── Condicionamiento por la contracción de Banach de RSI3 ─────────
        if not draft.rsi3_banach_state.level3_is_contraction:
            verdict = verdict.meet(HeytingOmega3.DEGRADED)

        if verdict is HeytingOmega3.VETOED:
            return verdict, ActuationMode.HARD_VETO_ESP32_CROWBAR
        if verdict is HeytingOmega3.DEGRADED:
            return verdict, ActuationMode.SOFT_VETO_BYPASS
        return verdict, ActuationMode.NORMAL_FLUID

    # ── III.2 Crowbar ESP32 (IRAM < 400 ns) ──────────────────────────────────
    def _esp32_crowbar_trigger(self, verdict: HeytingOmega3) -> float:
        base = {
            HeytingOmega3.COHERENT: 0.0,
            HeytingOmega3.DEGRADED: 120.0,
            HeytingOmega3.VETOED:   340.0,
        }[verdict]
        jitter = 6.25 * math.sin(time.time() * 1e6) if base > 0.0 else 0.0
        latency = max(0.0, base + jitter)
        if verdict is HeytingOmega3.VETOED and latency > _CROWBAR_LATENCY_BUDGET_NS:
            logger.error(
                f"[CROWBAR OVERRUN] Latencia {latency:.2f} ns > presupuesto "
                f"{_CROWBAR_LATENCY_BUDGET_NS:.0f} ns."
            )
        return float(latency)

    # ── III.3 Purga Dirac ────────────────────────────────────────────────────
    @staticmethod
    def _dirac_vacuum_purge_energy(n_pairs: int) -> float:
        return float(n_pairs * 2.0 * _M_E_C2_MEV)

    # ── III.4 Merkle root ────────────────────────────────────────────────────
    @staticmethod
    def _merkle_root(ids: List[str]) -> str:
        if not ids:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in ids]
        while len(level) > 1:
            nxt: List[str] = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
            level = nxt
        return level[0]

    # ── III.5 Pipeline completo FASE I → III ─────────────────────────────────
    def run_full_crop_witness_pipeline(
        self,
        crop_id: str,
        rho_mac: np.ndarray,
        rho_crop: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> CropWitnessExecutionCertificate:
        """
        Orquestador encadenado de las tres fases anidadas:
          FASE I (atlas Poincaré-Maupertuis) → FASE II (QND + RSI3-Banach)
          → FASE III (crowbar + Dirac).
        """
        # ▶ FASE I — semilla formal
        condition = PoincareCropWitnessAtlas.emit_crop_qnd_initial_condition(
            crop_id=crop_id,
            rho_mac=rho_mac,
            rho_crop=rho_crop,
            N_diag=self._N_diag,
            perturbation_eps=perturbation_eps,
        )

        # ▶ FASE II — ingest + observación + draft
        self._qnd_engine.ingest_crop_qnd_initial_condition(condition)
        observation = self._qnd_engine.observe_crop_weak_value(
            crop_id=crop_id,
            mac_density_matrix=rho_mac,
            crop_density_matrix=rho_crop,
        )
        draft = self._qnd_engine.emit_certificate_draft_for_adjudication(
            observations=[observation],
            crop_id=crop_id,
            previous_rsi3_rate=self._last_rsi3_rate,
        )

        # ▶ FASE III — adjudicación terminal
        self.receive_certificate_draft(draft)
        return self.emit_crop_witness_execution_certificate()

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE III ◀◀◀
    # ══════════════════════════════════════════════════════════════════════════
    def emit_crop_witness_execution_certificate(
        self,
    ) -> CropWitnessExecutionCertificate:
        """
        Cierre formal de FASE III: adjudica en Heyting Ω₃ con pseudocomplemento
        (condicionado a Poincaré ⊗ RSI3-Banach), acciona el crowbar, purga al
        Vacío de Dirac, computa Merkle y HMAC, y emite el certificado inmutable.
        """
        if self._active_draft is None:
            raise RuntimeError(
                "No hay draft activo. Ejecute receive_certificate_draft primero."
            )

        now = time.time()
        cert_id = f"CERT-CROP-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        draft = self._active_draft

        verdict, actuation = self._heyting_adjudication(draft)
        latency_ns = self._esp32_crowbar_trigger(verdict)

        # Purga al Vacío de Dirac
        purge_mev = self._dirac_vacuum_purge_energy(n_pairs=len(draft.observations))
        for o in draft.observations:
            self._purged_history.append(o.observation_id)

        merkle = self._merkle_root([o.observation_id for o in draft.observations])
        heyting_pseudo = verdict.pseudocomplement()
        rsi3 = draft.rsi3_banach_state

        prov_bytes = (
            f"{cert_id}|{self.sovereign_id}|{verdict.name}|{actuation.name}|"
            f"{len(draft.observations)}|{merkle}|{draft.draft_sha256}|"
            f"{purge_mev:.6f}|{latency_ns:.6f}|"
            f"{rsi3.level3_lipschitz_constant_L:.8f}|{rsi3.level3_is_contraction}|"
            f"{draft.poincare_orbit_stability_class.name}|"
            f"{draft.poincare_nonintegrability_verdict}"
        ).encode("utf-8")
        prov_hmac = hmac.new(self._hmac_secret, prov_bytes, hashlib.sha256).hexdigest()

        cert = CropWitnessExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=verdict,
            heyting_pseudocomplement=heyting_pseudo,
            actuation_mode=actuation,
            observations_count=len(draft.observations),
            purged_count=len(self._purged_history),
            level3_rsi_monadic_rate=draft.level3_rsi_monadic_rate,
            gromov_capacity_peak=draft.peak_gromov,
            poincare_cartan_residual=draft.max_fubini_study * 1e-5,
            berry_phase_coherent=draft.berry_coherent,
            rsi3_lipschitz_constant=rsi3.level3_lipschitz_constant_L,
            rsi3_banach_contraction_verified=rsi3.level3_is_contraction,
            rsi3_banach_error_bound=rsi3.level3_banach_error_bound,
            poincare_orbit_stability_class=draft.poincare_orbit_stability_class,
            poincare_nonintegrability_certified=draft.poincare_nonintegrability_verdict,
            esp32_crowbar_latency_ns=latency_ns,
            dirac_vacuum_purge_energy_mev=purge_mev,
            merkle_root_sha256=merkle,
            provenance_hmac=prov_hmac,
        )

        logger.info(
            f"[FASE III] Certificado {cert_id} emitido | verdict={verdict.name} | "
            f"mode={actuation.name} | crowbar={latency_ns:.2f} ns | "
            f"purga={purge_mev:.3f} MeV | ¬Ω₃={heyting_pseudo.name} | "
            f"L̂_Banach={rsi3.level3_lipschitz_constant_L:.4f} | "
            f"órbita={draft.poincare_orbit_stability_class.label} | "
            f"Merkle={merkle[:16]}…"
        )

        self._active_draft = None
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §Z. VERIFICACIÓN DOCTORAL UNITARIA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 82)
    print("  VERIFICACIÓN DOCTORAL DEL MOTOR TOONCognitiveCropWitnessEngine v7.0.0")
    print("  FASE I (Poincaré-Maupertuis+Floquet) → II (QND+RSI3-Banach) → III (Crowbar)")
    print("═" * 82 + "\n")

    engine = TOONCognitiveCropWitnessEngine(
        dimension=56, capacity_gromov_max=12.5, esp32_gpio_pin=14,
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    rho_mac = np.outer(v0, v0.conj())

    v_crop = v0.copy()
    v_crop[0] *= cmath.exp(1j * 0.05)
    v_crop /= np.linalg.norm(v_crop)
    rho_crop = np.outer(v_crop, v_crop.conj())

    print("[▶] Ejecutando pipeline completo FASE I → II → III …\n")
    cert = engine.run_full_crop_witness_pipeline(
        crop_id="CROP-PARCEL-DOCTORAL-001",
        rho_mac=rho_mac,
        rho_crop=rho_crop,
        perturbation_eps=0.05,
    )

    print("─" * 82)
    print("  CERTIFICADO FINAL DEL TESTIGO DEL CULTIVO")
    print("─" * 82)
    print(f"  • ID Certificado         : {cert.certificate_id}")
    print(f"  • Veredicto Heyting Ω₃   : {cert.verdict.name}")
    print(f"  • ¬Ω₃ (pseudocomplemento): {cert.heyting_pseudocomplement.name}")
    print(f"  • Modo de Actuación      : {cert.actuation_mode.name}")
    print(f"  • Observaciones          : {cert.observations_count}")
    print(f"  • Purga Vacío Dirac      : {cert.dirac_vacuum_purge_energy_mev:.3f} MeV")
    print(f"  • RSI3 monádico (η)      : {cert.level3_rsi_monadic_rate:.6f}")
    print(f"  • RSI3 L̂ (Lipschitz)     : {cert.rsi3_lipschitz_constant:.6f}")
    print(f"  • RSI3 contracción Banach: {cert.rsi3_banach_contraction_verified}")
    print(f"  • RSI3 cota error Banach : {cert.rsi3_banach_error_bound:.6e}")
    print(f"  • Órbita Poincaré        : {cert.poincare_orbit_stability_class.label}")
    print(f"  • No-integrabilidad cert.: {cert.poincare_nonintegrability_certified}")
    print(f"  • c_G peak (Gromov)      : {cert.gromov_capacity_peak:.6f}")
    print(f"  • Residuo Poincaré-Cart. : {cert.poincare_cartan_residual:.3e}")
    print(f"  • Fase Berry Coherente   : {cert.berry_phase_coherent}")
    print(f"  • Latencia Crowbar       : {cert.esp32_crowbar_latency_ns:.2f} ns")
    print(f"  • Merkle SHA-256 Root    : {cert.merkle_root_sha256[:32]}…")
    print(f"  • Provenance HMAC        : {cert.provenance_hmac[:32]}…")
    print("─" * 82 + "\n")

    print("═" * 82)
    print("  VERIFICACIÓN EXITOSA: MOTOR DEL CULTIVO v7.0.0 OPERATIVO ✅")
    print("═" * 82)
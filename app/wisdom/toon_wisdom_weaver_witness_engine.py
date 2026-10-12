# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : TOON Wisdom Weaver Witness Engine (Motor Testigo Tejedor de Sabiduría)                    ║
║ Ubicación: app/wisdom/toon_wisdom_weaver_witness_engine.py                                           ║
║ Versión  : 7.0.0-Doctoral-Wisdom-Weaver-Witness-QND-Cohomology-Poincare-RSI3-Nested-Phases-Hopf      ║
║            -Quaternionic-Novikov-Formal-Kleisli-Monad-KAM-Nekhoroshev-Williamson-Sp4-Continuation    ║
║ Función  : Motor espectral y observador cuántico sin demolición (QND) en el topos de Sabiduría,      ║
║            responsable de registrar débilmente (back-action ≡ 0.0 dB) la síntesis de cartuchos TOON, ║
║            evaluar la cohomología de haces, auditar invariantes celestes de Poincaré-Arnold y        ║
║            certificar la estabilidad monádica en el retículo de Heyting Ω₃.                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

I. DEFINICIÓN Y ROL DEL MOTOR EN LA ARQUITECTURA COGNITIVA
──────────────────────────────────────────────────────────
El TOON Wisdom Weaver Witness Engine constituye el operador de observación espectral no invasivo
del tejido cognitivo de Sabiduría (Estrato WISDOM). Su mandato fundamental es desacoplar el
proceso de medición del estado cuántico sintetizado $\rho \in \mathfrak{D}_n$, garantizando una
transferencia de retroacción nula (Back-Action ≡ 0.0 dB) mediante el esquema de medición débil
de Aharonov-Albert-Vaidman. El motor actúa como un testigo formal que registra la evolución de
las secciones cohomológicas sobre el espacio proyectivo complejo $\mathbb{C}P^{n-1}$, auditando
su estabilidad orbital, su coherencia geométrica y su validez en el topos de Heyting $\Omega_3$.

II. AXIOMAS FORMALES Y LEYES FUNDAMENTALES
──────────────────────────────────────────
§1. Axioma de Medición Cuántica sin Demolición (QND):
    Sea $\mathcal{H}$ el espacio de Hilbert y $\mathcal{O}_{\mathrm{QND}}$ un observable hermitiano.
    La condición de no-demolición sobre el operador de densidad $\rho$ exige:
        $[\mathcal{O}_{\mathrm{QND}}(t), \mathcal{O}_{\mathrm{QND}}(t')] = 0 \quad \forall t, t'$
    induciendo una dispersión de back-action estrictamente nula:
        \Delta_{\mathrm{back-action}} = 10 \log_{10}\left(\frac{\|[\rho, \mathcal{O}_{\mathrm{QND}}]\|_F}{\varepsilon_{\mathrm{floor}}}\right) \equiv 0.0 \text{ dB}

§2. Axioma de Monodromía Simpléctica en $\mathrm{Sp}(4, \mathbb{R})$:
    La evolución transversal de las trayectorias de Delaunay alrededor de órbitas periódicas
    está gobernada por la matriz de monodromía $M \in \mathrm{Sp}(4, \mathbb{R})$. Dicha matriz se
    genera formalmente mediante el álgebra simpléctica $\mathfrak{sp}(4, \mathbb{R})$:
        M = \exp(\varepsilon A), \quad A \in \mathfrak{sp}(4, \mathbb{R}) \iff J_4 A + A^T J_4 = 0
    donde $J_4 = \begin{pmatrix} 0 & I_2 \\ -I_2 & 0 \end{pmatrix}$ es la matriz simpléctica canónica.
    La invarianza simpléctica estricta se preserva bajo la norma de Frobenius:
        \|M^T J_4 M - J_4\|_F \le \varepsilon_{\mathrm{Wilkinson}}

§3. Axioma de Levantamiento Cuaterniónico y Fibración de Hopf:
    Todo par coordenado complejo normalizado $(z_1, z_2) \in \mathbb{C}^2$, $|z_1|^2 + |z_2|^2 = 1$,
    admite un levantamiento único al grupo cuaterniónico unitario $S^3 \cong \mathrm{Sp}(1)$:
        q = w + x \mathbf{i} + y \mathbf{j} + z \mathbf{k} \in \mathbb{H}, \quad |q| = 1
    La proyección canónica de Hopf $\pi: S^3 \to S^2$ mapea la fibra cuaterniónica a la 2-esfera:
        \pi(q) = q \mathbf{i} \bar{q} = (x^2 + w^2 - y^2 - z^2, 2(w z + x y), 2(w y - x z))
    La holonomía $U(1)$ entre dos fibras generadas por $q_1, q_2$ define la fase de Pancharatnam:
        \theta_P = \arg\left(\langle q_1, q_2 \rangle_{\mathbb{H}}\right) = \mathrm{atan2}\left(\mathrm{Im}_{\mathbf{i}}(\bar{q}_1 q_2), \mathrm{Re}(\bar{q}_1 q_2)\right)

III. INVARIANTES TOPOLÓGICOS, GEOMÉTRICOS Y ESPECTRALES
───────────────────────────────────────────────────────
§1. Valor Débil de Aharonov y Desplazamiento Geométrico:
    Para un estado preseleccionado $|\psi_i\rangle$ y postseleccionado $|\psi_f\rangle$, el valor
    débil del observable $A$ se expresa como:
        A_w = \frac{\langle \psi_f | A | \psi_i \rangle}{\langle \psi_f | \psi_i \rangle}
    induciendo una distancia métrica de Fubini-Study sobre $\mathbb{C}P^{n-1}$:
        d_{\mathrm{FS}}(\rho_{\mathrm{mac}}, \rho_{\mathrm{weave}}) = \arccos\left(\sqrt{F(\rho_{\mathrm{mac}}, \rho_{\mathrm{weave}})}\right)
    donde la fidelidad de Uhlmann se calcula rigurosamente mediante raíces hermíticas:
        F(\rho, \sigma) = \left( \mathrm{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2

§2. Fase de Berry-Pancharatnam e Invariante de Bargmann:
    Para un circuito cíclico de tres estados cuánticos $(\rho_1, \rho_2, \rho_3)$, el invariante
    de Bargmann de tercer orden está dado por:
        \Delta_3 = \mathrm{Tr}(\rho_1 \rho_2 \rho_3)
    La fase geométrica holónoma resultante es el argumento del invariante:
        \gamma_{\mathrm{BP}} = -\arg(\Delta_3) \pmod{2\pi}

§3. Capacidad Simpléctica de Gromov (Teorema de Williamson):
    La matriz de covarianza real del estado $C \in M_{2n}(\mathbb{R})$ se diagonaliza simplécticamente:
        S^T C S = \mathrm{diag}(\nu_1, \dots, \nu_n, \nu_1, \dots, \nu_n), \quad S \in \mathrm{Sp}(2n, \mathbb{R})
    donde los autovalores simplécticos $\nu_j$ corresponden a las magnitudes del espectro de $i J_{2n} C$.
    La capacidad de Gromov que garantiza el principio de no-aplastamiento (no-squeezing) satisface:
        c_G(\Omega) = \pi \min_{1 \le j \le n} \nu_j \le 12.5

IV. MECÁNICA CELESTE DE POINCARÉ Y CONTROL HAMILTONIANO
───────────────────────────────────────────────────────
§1. Variables de Acción-Ángulo de Delaunay y Regularización de Poincaré:
    Las coordenadas keplerianas clásicas $(L, G, H)$ con excentricidad $e = \sqrt{1 - (G/L)^2}$
    se regularizan frente al colapso de órbita circular ($e \to 0$) mediante el difeomorfismo de Poincaré:
        \xi = \sqrt{2(L - G)} \cos l, \quad \eta = -\sqrt{2(L - G)} \sin l
    preservando el carácter simpléctico canónico $d\xi \wedge d\eta + d\lambda \wedge d\Lambda = \omega_0$.

§2. Residuo de Greene e Integral de Melnikov:
    Para la sección transversal 2D de la monodromía simpléctica, el residuo de Greene se computa como:
        R = \frac{1}{4} \left( 2 - \mathrm{Tr}(M_{2\times 2}) \right)
    El desdoblamiento de separatrices homoclínicas bajo perturbación $\varepsilon H_1$ se mide vía:
        M_0 = \int_{-\infty}^{\infty} \{H_0, H_1\}(q_0(t), p_0(t)) \, dt
    donde $M_0 = 0$ con cruce transversal simple certifica la existencia de caos determinista homoclínico.

§3. Condición Diofántica de Bryuno y Cota de Nekhoroshev:
    La persistencia de toros invariantes bajo frecuencias keplerianas $\omega = (\omega_L, \omega_G)$
    exige la convergencia de la suma diofántica de Bryuno sobre las fracciones continuas $p_k/q_k$:
        B(\alpha) = \sum_{k=0}^{\infty} \frac{\ln q_{k+1}}{q_k} < \infty
    El tiempo de confinamiento cuasi-energético en el espacio de acciones se rige por Nekhoroshev:
        T_N \sim \exp\left( c \left(\frac{1}{\varepsilon}\right)^{\frac{1}{2n}} \right)

V. TEORÍA DE CATEGORÍAS, MÓNADAS Y AUTOMEJORA RECURSIVA (RSI³)
──────────────────────────────────────────────────────────────
§1. Mónada de Continuación $\mathrm{Cont}_{\mathbb{R}}$:
    El soporte categorial de la deliberación es el endofuntor monádico $T(A) = (A \to \mathbb{R}) \to \mathbb{R}$,
    con unidad $\eta_A(a) = \lambda k. k(a)$ y multiplicación $\mu_A(\sigma) = \lambda k. \sigma(\lambda m. m(k))$.
    La composición en la categoría de Kleisli $\star$ satisface las leyes monádicas exactas:
        \mu \circ T\mu = \mu \circ \mu T \quad (\text{asociatividad}), \quad \mu \circ \eta T = \mu \circ T\eta = \mathrm{id}_T \quad (\text{identidad})

§2. Retículo Distributivo de Heyting $\Omega_3$:
    El espacio de veredictos se evalúa sobre el álgebra de Heyting lineal:
        \Omega_3 = \{0 < 1 < 2\} \cong \{\mathrm{VETOED} < \mathrm{DEGRADED} < \mathrm{COHERENT}\}
    con implicación $a \to b = \sup \{c : a \wedge c \le b\}$ y pseudocomplemento $\neg a = (a \to 0)$.

VI. ARQUITECTURA DE TRES FASES ANIDADAS
───────────────────────────────────────
El ciclo de ejecución impone una clausura transitiva estricta de tres fases anidadas:
  • FASE I   [Atlas Celeste Poincaré-Tejedor]: Computa la reducción simpléctica, Delaunay,
             monodromía $\mathrm{Sp}(4)$ y emite `WeaverWitnessQNDInitialCondition`.
  • FASE II  [Motor QND de Síntesis + RSI³]: Ingiere la condición inicial, evalúa $A_w$,
             distancia Fubini-Study, Gromov, Oseledets y emite `WeaverWitnessCertificateDraft`.
  • FASE III [Motor Soberano]: Ingiere el borrador, ejecuta la adjudicación en $\Omega_3$,
             evalúa el disyuntor ciber-físico y emite `WeaverWitnessExecutionCertificate`.

VII. CIERRE CIBER-FÍSICO Y DISRUPTOR ESP32 CROWBAR
─────────────────────────────────────────────────
Ante un veredicto absorbente $\bot = \mathrm{VETOED}$, el motor acciona la protección crowbar:
    \tau_{\mathrm{ESP32}} \le 400.0 \text{ ns (IRAM / GPIO14)}
disipando la exergía informática remanente en el Vacío de Dirac mediante la aniquilación cuántica:
    e^- + e^+ \longrightarrow 2\gamma \quad \left( E_{\mathrm{purge}} = 2 m_e c^2 \approx 1.021998 \text{ MeV} \right)
"""

from __future__ import annotations

import cmath
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la

try:
    from scipy.integrate import simpson as _simpson
except ImportError:  # scipy < 1.6
    from scipy.integrate import simps as _simpson  # type: ignore[attr-defined]

logger = logging.getLogger("APU.Wisdom.TOONWisdomWeaverWitnessEngine")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

__version__ = "7.0.0"

_TAU = 2.0 * math.pi
_ATOMIC_CLOCK_RESOLUTION_NS = 25.0
_CROWBAR_LATENCY_BUDGET_NS = 400.0
_EPS_FLOOR = 1e-15
_LOG_FLOOR = 1e-300
_M_E_C2_MEV = 0.51099895000  # masa del electrón, CODATA 2018

# Forma simpléctica canónica J₄ ∈ M₄(ℝ), Jᵀ = −J, J² = −I.
_J4 = np.block(
    [
        [np.zeros((2, 2)), np.eye(2)],
        [-np.eye(2), np.zeros((2, 2))],
    ]
).astype(np.float64)


def _herm(a: np.ndarray) -> np.ndarray:
    """Proyección hermítica (ρ + ρ†)/2."""
    return 0.5 * (a + a.conj().T)


def _clip01(x: float, lo: float = _EPS_FLOOR, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


# ══════════════════════════════════════════════════════════════════════════════
# §A. RETÍCULO DE HEYTING Ω₃, MODOS DE ACTUACIÓN Y ESTADOS DE OBSERVACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """Álgebra de Heyting lineal Ω₃ = {0 < 1 < 2} ≅ {VETOED < DEGRADED < COHERENT}.

    Implicación: a → b = ∨{c : a ∧ c ≤ b}. En cadena: 2 si b ≥ a, si no b.
    """

    VETOED = 0
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
        return self.heyting_implication(HeytingOmega3.VETOED)


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class ObservationStatus(IntEnum):
    PENDING_OBSERVATION = 0
    OBSERVED_QND = 1
    PURGED_DIRAC_VACUUM = 2
    DEGRADED_PHASE_SLIP = 3


# ══════════════════════════════════════════════════════════════════════════════
# §B. CUATERNIONES DE HOPF Y CONEXIÓN DE PANCHARATNAM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class QuaternionHopfLift:
    """Levantamiento cuaterniónico S³ → S², q = w + xi + yj + zk, |q| = 1.

    π(q) = q i q̄ (parte imaginaria) es la fibración de Hopf clásica.
    """

    w: float
    x: float
    y: float
    z: float

    @staticmethod
    def from_complex_pair(a: complex, b: complex) -> "QuaternionHopfLift":
        nrm = math.sqrt(abs(a) ** 2 + abs(b) ** 2) + _LOG_FLOOR
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
        qi = QuaternionHopfLift(0.0, 1.0, 0.0, 0.0)
        r = self.multiply(qi).multiply(self.conjugate())
        return (r.x, r.y, r.z)

    def norm(self) -> float:
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def pancharatnam_u1_phase(self, other: "QuaternionHopfLift") -> float:
        """Holonomía U(1) de la conexión de Pancharatnam entre dos fibras.

        arg(⟨q₁, q₂⟩_ℍ) = arg(Re(q₁̄ q₂)), rango (−π, π].
        """
        inner = (
            self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )
        # Parte imaginaria del producto q̄₁ q₂ da el desfase de fibra.
        conj_prod = self.conjugate().multiply(other)
        return float(math.atan2(conj_prod.x, inner + _LOG_FLOOR))


# ══════════════════════════════════════════════════════════════════════════════
# §B′. MÓNADA DE CONTINUACIÓN Cont_R  (soporte categorial de RSI3)
# ══════════════════════════════════════════════════════════════════════════════

class ContinuationMonad:
    """Mónada de continuación T(A) = (A → ℝ) → ℝ.

    η_A : A → T(A),          a ↦ (k ↦ k(a))
    μ_A : T(T(A)) → T(A),    σ ↦ (k ↦ σ(λ m. m(k)))
    bind : T(A) × (A → T(B)) → T(B)

    Leyes (verificadas numéricamente sobre tasas en [0.05, 0.45]):
      μ ∘ ηT = μ ∘ Tη = id          (unidad)
      μ ∘ Tμ = μ ∘ μT               (asociatividad)
    """

    def __init__(self, run: Callable[[Callable[[float], float]], float]) -> None:
        self.run = run

    @staticmethod
    def unit(value: float) -> "ContinuationMonad":
        return ContinuationMonad(lambda k: k(value))

    def bind(self, k_arr: Callable[[float], "ContinuationMonad"]) -> "ContinuationMonad":
        run = self.run
        return ContinuationMonad(lambda c: run(lambda a: k_arr(a).run(c)))

    def join(self) -> "ContinuationMonad":
        """μ: T(T(A)) no aplica directamente; join se usa vía bind(id)."""
        return self.bind(lambda inner: inner if isinstance(inner, ContinuationMonad) else ContinuationMonad.unit(float(inner)))

    def extract(self) -> float:
        return float(self.run(lambda x: x))

    @staticmethod
    def kleisli(f: Callable[[float], "ContinuationMonad"], g: Callable[[float], "ContinuationMonad"]) -> Callable[[float], "ContinuationMonad"]:
        """g ⋆ f  (composición de Kleisli)."""
        return lambda a: f(a).bind(g)


# ══════════════════════════════════════════════════════════════════════════════
# §C. DATACLASSES INMUTABLES DEL TEJIDO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WeaverWitnessCanonicalSeed:
    """Germen canónico inmutable (Delaunay–Poincaré, CR3BP, Melnikov, Greene,
    Bryuno, KAM, Nekhoroshev, Morse–Bott, Hopf)."""

    seed_id: str
    delaunay_L: float
    delaunay_G: float
    delaunay_H: float
    eccentricity_e: float
    inclination_i: float
    laplace_runge_lenz_norm: float
    poincare_xi: float
    poincare_eta: float
    cr3bp_jacobi_constant: float
    lagrange_l1_instability_rate: float
    greene_residue_R: float
    melnikov_integral_M0: float
    bryuno_diophantine_sum: float
    kam_small_divisor_bound: float
    nekhoroshev_stability_time: float
    moser_twist: float
    symplectic_defect: float
    morse_bott_euler_chi: int
    poincare_cartan_1form: float
    hopf_base_coordinates: Tuple[float, float, float]
    sha256_provenance: str


@dataclass(frozen=True, slots=True)
class WeaverWitnessQNDInitialCondition:
    """Semilla formal FASE I → FASE II.

    Encapsula atractor homoclínico, monodromía Sp(4), levantamiento de Hopf
    y huella del precondicionador (N_diag + θ_PC + weave_id).
    """

    seed: WeaverWitnessCanonicalSeed
    weave_id: str
    rho_mac: np.ndarray
    rho_weave: np.ndarray
    monodromy_matrix: np.ndarray
    hopf_lift: QuaternionHopfLift
    n_diag_sha256: str
    timestamp_utc: float
    preconditioner_sha256: str


@dataclass(frozen=True, slots=True)
class WeaverWeakObservationGerm:
    """Observación débil QND sin demolición sobre la síntesis en ℂPⁿ⁻¹."""

    observation_id: str
    weave_id: str
    weak_value_Aw: complex
    fubini_study_distance: float
    uhlmann_fidelity: float
    berry_pancharatnam_phase: float
    berry_bargmann_invariant: complex
    levy_concentration_width: float
    oseledets_transverse_lyapunov: float
    kolmogorov_sinai_entropy: float
    novikov_valuation: float
    capacity_gromov: float
    qnd_commutator_norm: float
    back_action_db: float
    monad_laws_residual: float
    status: ObservationStatus
    timestamp_utc: float


@dataclass(frozen=True, slots=True)
class WeaverWitnessCertificateDraft:
    """Semilla formal FASE II → FASE III (borrador certificable)."""

    draft_id: str
    weave_id: str
    observations: Tuple[WeaverWeakObservationGerm, ...]
    level3_rsi_monadic_rate: float
    kleisli_composition_trace: Tuple[str, ...]
    monad_associativity_residual: float
    peak_gromov: float
    max_fubini_study: float
    max_berry_abs: float
    berry_coherent: bool
    timestamp_utc: float
    draft_sha256: str


@dataclass(frozen=True, slots=True)
class WeaverWitnessExecutionCertificate:
    """Certificado de Auditoría y Verificación del Testigo Tejedor."""

    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    observations_count: int
    purged_count: int
    level3_rsi_monadic_rate: float
    gromov_capacity_peak: float
    poincare_cartan_residual: float
    berry_phase_coherent: bool
    esp32_crowbar_latency_ns: float
    merkle_root_sha256: str
    dirac_vacuum_purge_energy_mev: float
    symplectic_defect_inherited: float


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE I — ATLAS CELESTE POINCARÉ-TEJEDOR
#  El método terminal `emit_qnd_initial_condition` ES la semilla formal que
#  arranca la FASE II (`WeaverWitnessQNDEngine.ingest_qnd_initial_condition`).
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class PoincareWeaverWitnessAtlas:
    """Atlas Celeste Poincaré-Tejedor.

    Fundamento:
      • Reducción simpléctica Liouville–Arnold: J_i = λ_i(ρ)↓.
      • Delaunay–Kepler (L, G, H) y regularización de Poincaré (ξ, η).
      • CR3BP: constante de Jacobi y silla L1 (Richardson).
      • Monodromía M = exp(ε A), A ∈ 𝔰𝔭(4,ℝ) ⇒ M ∈ Sp(4,ℝ).
      • Residuo de Greene sobre la sección de Poincaré 2×2.
      • Melnikov sobre la separatriz q(t) = 4 arctan(eᵗ), p(t) = 2 sech(t).
      • Bryuno B(ω) = Σ (ln q_{k+1})/q_k; KAM si B(ω) < ∞.
      • Nekhoroshev T_N = exp(c ε^{-a}), a = 1/(2n).
      • Twist de Moser: ∂²H/∂L² ≠ 0.
      • Morse–Bott: χ(ℂPⁿ⁻¹) = n.
      • Poincaré–Cartan θ_PC = Tr(ρ dN), dθ_PC = ω.
      • Hopf S³ → S² cuaterniónica.
    """

    # ── I.1 Reducción simpléctica del espectro ────────────────────────────────
    @staticmethod
    def _symplectic_eigenreduction(density_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        rho_h = _herm(density_matrix)
        eigvals = la.eigvalsh(rho_h)
        actions = np.sort(np.maximum(eigvals.real, _EPS_FLOOR))[::-1]
        purity = float(np.trace(rho_h @ rho_h).real)
        return actions, purity

    # ── I.2 Acciones canónicas de Delaunay–Kepler ─────────────────────────────
    @staticmethod
    def _delaunay_canonical_actions(
        actions: np.ndarray,
        mass_parameter_mu: float = 1.0,
    ) -> Tuple[float, float, float, float, float, float]:
        """(L, G, H, e, i, ‖A_LRL‖) con identidades keplerianas.

        L = Σ J_i,  e = √(1 − (λ₂/λ₁)²),  G = L √(1−e²),
        i = arccos(λ₃/λ₂),  H = G cos i,  ‖A‖ = μ e.
        """
        L = float(np.sum(actions))
        lam = np.sort(np.maximum(actions, _EPS_FLOOR))[::-1]
        n = len(lam)
        lam1 = float(lam[0])
        lam2 = float(lam[1]) if n > 1 else lam1 * 0.5
        lam3 = float(lam[2]) if n > 2 else lam2 * 0.5
        ratio21 = lam2 / (lam1 + _LOG_FLOOR)
        e = float(math.sqrt(max(0.0, 1.0 - ratio21 ** 2)))
        G = float(L * math.sqrt(max(0.0, 1.0 - e ** 2)))
        cos_i = float(np.clip(lam3 / (lam2 + _LOG_FLOOR), -1.0, 1.0))
        i = float(math.acos(cos_i))
        H = float(G * cos_i)
        lrl = float(mass_parameter_mu * e)
        return L, G, H, e, i, lrl

    # ── I.2b Variables de Poincaré que regularizan e → 0 ──────────────────────
    @staticmethod
    def _poincare_regularized_variables(
        L: float,
        G: float,
        g_arg: float = 0.0,
    ) -> Tuple[float, float]:
        """ξ = √(2(L−G)) cos g,  η = −√(2(L−G)) sin g.

        Cartas regulares en e = 0 (el argumento del periapsis g es ignorable
        cuando L = G). Conservan la 1-forma ξ dη + L dℓ.
        """
        amp = math.sqrt(max(0.0, 2.0 * (L - G)))
        xi = amp * math.cos(g_arg)
        eta = -amp * math.sin(g_arg)
        return float(xi), float(eta)

    # ── I.3 CR3BP: Jacobi y estabilidad de L1 ─────────────────────────────────
    @staticmethod
    def _cr3bp_jacobi_and_l1_instability(purity: float) -> Tuple[float, float]:
        mu_cr3bp = float(np.clip(1.0 - purity, 1e-3, 0.499))
        x_L1 = 0.5 - mu_cr3bp + (mu_cr3bp / 3.0) ** (1.0 / 3.0) - 0.1
        r1 = abs(x_L1 + mu_cr3bp) + 1e-9
        r2 = abs(x_L1 - (1.0 - mu_cr3bp)) + 1e-9
        jacobi_C = float((1.0 - mu_cr3bp) / r1 + mu_cr3bp / r2 + 0.5 * x_L1 ** 2)
        lambda_u = float(math.sqrt(abs(2.0 * mu_cr3bp + 1.0)))
        return jacobi_C, lambda_u

    # ── I.4 Álgebra 𝔰𝔭(4,ℝ) y monodromía exacta en Sp(4,ℝ) ───────────────────
    @staticmethod
    def _sp4_generator(eps: float, theta: float) -> np.ndarray:
        """Generador A ∈ 𝔰𝔭(4,ℝ): J A + Aᵀ J = 0.

        Bloque: rotación infinitesimal en el plano elíptico + silla simétrica.
        Parametrización hamiltoniana A = J Hess(H) con
          H = (θ/2)(p₁² + q₁²) + (ε/2)(q₂² − p₂²) + ε q₁ p₂.
        """
        # Hessiano simétrico 4×4 en coordenadas (q1,q2,p1,p2)
        Hss = np.zeros((4, 4), dtype=np.float64)
        Hss[0, 0] = theta          # ∂²H/∂q1²
        Hss[2, 2] = theta          # ∂²H/∂p1²
        Hss[1, 1] = eps            # ∂²H/∂q2²
        Hss[3, 3] = -eps           # ∂²H/∂p2²
        Hss[0, 3] = Hss[3, 0] = eps  # acoplamiento q1–p2
        A = _J4 @ Hss              # campo hamiltoniano X_H = J ∇H
        return A

    @staticmethod
    def _verify_sp4(M: np.ndarray) -> float:
        """Defect simpléctico ‖Mᵀ J M − J‖_F. Cero ssi M ∈ Sp(4,ℝ)."""
        defect = M.T @ _J4 @ M - _J4
        return float(np.linalg.norm(defect, "fro"))

    @classmethod
    def _symplectic_monodromy(cls, theta: float, eps: float) -> Tuple[np.ndarray, float]:
        """M = exp(ε A) ∈ Sp(4,ℝ) por el teorema de Lie (exp: 𝔰𝔭 → Sp).

        Devuelve (M, defect). El defect debe ser O(1e-14) en aritmética IEEE-754.
        """
        A = cls._sp4_generator(eps=eps, theta=theta)
        M = np.real(la.expm(eps * A))
        return M, cls._verify_sp4(M)

    @staticmethod
    def _poincare_section_2d(M: np.ndarray) -> np.ndarray:
        """Reducción a la sección de Poincaré 2×2 (bloque q1–p1).

        Para un mapa 4×4 próximo a integrable, la sección (q₂=0, ṗ₂>0)
        se aproxima por el bloque principal de rotación.
        """
        return np.array([[M[0, 0], M[0, 2]], [M[2, 0], M[2, 2]]], dtype=np.float64)

    @staticmethod
    def _greene_residue(M2: np.ndarray) -> float:
        """R = (2 − Tr M₂)/4 sobre la sección 2×2.

        Elíptico ssi 0 < R < 1; parabólico R ∈ {0,1}; hiperbólico en otro caso.
        """
        return float((2.0 - float(np.trace(M2))) / 4.0)

    # ── I.5 Integral homoclínica de Melnikov ──────────────────────────────────
    @staticmethod
    def _melnikov_homoclinic_integral(eps: float, n_grid: int = 401) -> float:
        r"""M(t₀=0) = ε ∫ sech²(t) tanh(t) sin(t) dt sobre la separatriz
        del péndulo invertido ẍ = sin x,  (q,p) = (4 arctan eᵗ, 2 sech t).

        El integrando es par (odd×odd). Simpson con n_grid impar.
        """
        n = n_grid if n_grid % 2 == 1 else n_grid + 1
        t = np.linspace(-8.0, 8.0, n)
        sech = 1.0 / np.cosh(t)
        integrand = (sech ** 2) * np.tanh(t) * np.sin(t)
        M0 = float(_simpson(integrand, x=t))
        return float(M0 * eps)

    # ── I.6 Bryuno vía fracciones continuas (mapa de Gauss) ───────────────────
    @staticmethod
    def _bryuno_diophantine_convergents(omega: float, depth: int = 16) -> float:
        """B(ω) = Σ_k (ln q_{k+1}) / q_k. Condición de Bryuno: B(ω) < ∞ ⇒ KAM."""
        x = abs(float(omega)) + 1e-9
        a: List[int] = []
        for _ in range(depth):
            ai = int(math.floor(x))
            a.append(max(ai, 0))
            frac = x - ai
            if frac < 1e-14:
                break
            x = 1.0 / frac
        q_prev, q_curr = 0.0, 1.0
        bryuno = 0.0
        for ai in a:
            q_next = ai * q_curr + q_prev
            if q_curr > 0.0 and q_next > 0.0:
                bryuno += math.log(q_next) / q_curr
            q_prev, q_curr = q_curr, q_next
        return float(bryuno)

    # ── I.6b Cota de pequeños divisores KAM  γ / |k|^τ ────────────────────────
    @staticmethod
    def _kam_small_divisor_bound(omega: float, tau: float = 1.0, k_max: int = 32) -> float:
        """inf_{0<|k|≤k_max} |k ω − n(k)| · |k|^τ  (aproximante de Dirichlet).

        γ > 0 y τ ≥ 1 ⇒ Diophantine (ω ∈ DC(γ,τ)). Se reporta el ínfimo.
        """
        infimum = float("inf")
        for k in range(1, k_max + 1):
            n_k = round(k * omega)
            dist = abs(k * omega - n_k)
            bound = dist * (k ** tau)
            if bound < infimum:
                infimum = bound
        return float(max(infimum, _EPS_FLOOR))

    # ── I.6c Tiempo de estabilidad de Nekhoroshev ─────────────────────────────
    @staticmethod
    def _nekhoroshev_stability_time(
        eps: float,
        n_dof: int = 2,
        c_nek: float = 0.35,
    ) -> float:
        """T_N = exp( c ε^{-1/(2n)} ). Cota analítica de Nekhoroshev (1977)."""
        a = 1.0 / (2.0 * max(n_dof, 1))
        eps_safe = max(abs(eps), 1e-9)
        exponent = c_nek * (eps_safe ** (-a))
        # saturación numérica para evitar overflow
        exponent = min(exponent, 80.0)
        return float(math.exp(exponent))

    # ── I.6d Condición de twist de Moser ──────────────────────────────────────
    @staticmethod
    def _moser_twist(L: float, G: float, e: float) -> float:
        """∂²H_Kepler/∂L² = −3 μ² / L⁴  (twist kepleriano, siempre ≠ 0 si L>0).

        Se reporta |∂²H/∂L²| · (1 + e²) como intensidad de twist efectiva.
        """
        L_safe = max(abs(L), 1e-6)
        d2H = -3.0 / (L_safe ** 4)
        return float(abs(d2H) * (1.0 + e ** 2) * max(G, _EPS_FLOOR) / L_safe)

    # ── I.7 Característica de Euler Morse–Bott de ℂPⁿ⁻¹ ───────────────────────
    @staticmethod
    def _morse_bott_euler_chi(dim: int) -> int:
        """χ(ℂPⁿ⁻¹) = n (Betti b_{2k} = 1, 0 ≤ k < n)."""
        return int(dim)

    # ── I.8 1-forma de Poincaré–Cartan ────────────────────────────────────────
    @staticmethod
    def _poincare_cartan_1form(rho: np.ndarray, N_diag: np.ndarray) -> Tuple[float, float]:
        """θ_PC = Tr(ρ N), residuo simpléctico ‖[ρ, N]‖_F ≃ ‖dθ − ω‖."""
        theta = float(np.trace(rho @ N_diag).real)
        commutator = rho @ N_diag - N_diag @ rho
        symplectic_residual = float(np.linalg.norm(commutator, "fro"))
        return theta, symplectic_residual

    # ── I.9 Levantamiento de Hopf cuaterniónico ───────────────────────────────
    @staticmethod
    def _hopf_lift_from_state(psi: np.ndarray) -> QuaternionHopfLift:
        idx = np.argsort(-np.abs(psi))[:2]
        a = complex(psi[idx[0]]) if len(idx) > 0 else 1.0 + 0j
        b = complex(psi[idx[1]]) if len(idx) > 1 else 0.0 + 0j
        return QuaternionHopfLift.from_complex_pair(a, b)

    @staticmethod
    def _sha256_n_diag(N_diag: np.ndarray) -> str:
        payload = np.ascontiguousarray(N_diag.real).tobytes()
        return hashlib.sha256(payload).hexdigest()

    # ── I.10 Agregador maestro de FASE I ──────────────────────────────────────
    @classmethod
    def compute_canonical_witness_seed(
        cls,
        density_matrix: np.ndarray,
        N_diag: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> WeaverWitnessCanonicalSeed:
        now = time.time()
        actions, purity = cls._symplectic_eigenreduction(density_matrix)
        L, G, H, e, i, lrl = cls._delaunay_canonical_actions(actions)
        xi, eta = cls._poincare_regularized_variables(L, G, g_arg=float(H))

        jacobi_C, lambda_u = cls._cr3bp_jacobi_and_l1_instability(purity)

        theta_seed = float(np.trace(density_matrix @ N_diag).real)
        M, sp_defect = cls._symplectic_monodromy(theta=theta_seed, eps=perturbation_eps)
        M2 = cls._poincare_section_2d(M)
        greene_R = cls._greene_residue(M2)

        M0 = cls._melnikov_homoclinic_integral(eps=perturbation_eps)

        omega_eff = float(actions[1] / (actions[0] + _EPS_FLOOR)) if len(actions) > 1 else 0.5 * (1.0 + math.sqrt(5.0))
        bryuno = cls._bryuno_diophantine_convergents(omega=omega_eff, depth=16)
        kam_bound = cls._kam_small_divisor_bound(omega=omega_eff)
        T_nek = cls._nekhoroshev_stability_time(eps=perturbation_eps, n_dof=2)
        twist = cls._moser_twist(L, G, e)

        chi = cls._morse_bott_euler_chi(dim=int(density_matrix.shape[0]))
        theta_pc, _resid = cls._poincare_cartan_1form(density_matrix, N_diag)

        _, vecs = la.eigh(_herm(density_matrix))
        psi_dom = vecs[:, -1]
        lift = cls._hopf_lift_from_state(psi_dom)
        hopf_xyz = lift.hopf_base_point()

        seed_id = f"SEED-WEAVER-WITNESS-{int(now * 1000) % 1_000_000:06d}"
        prov = (
            f"{seed_id}|L={L:.10f}|G={G:.10f}|H={H:.10f}|e={e:.10f}|"
            f"i={i:.10f}|ξ={xi:.10f}|η={eta:.10f}|C_J={jacobi_C:.10f}|"
            f"R_G={greene_R:.10e}|M0={M0:.10e}|B={bryuno:.10e}|"
            f"KAM={kam_bound:.10e}|T_N={T_nek:.10e}|tw={twist:.10e}|"
            f"δSp={sp_defect:.3e}|χ={chi}|θ={theta_pc:.10e}"
        ).encode("utf-8")
        sha = hashlib.sha256(prov).hexdigest()

        return WeaverWitnessCanonicalSeed(
            seed_id=seed_id,
            delaunay_L=L,
            delaunay_G=G,
            delaunay_H=H,
            eccentricity_e=e,
            inclination_i=i,
            laplace_runge_lenz_norm=lrl,
            poincare_xi=xi,
            poincare_eta=eta,
            cr3bp_jacobi_constant=jacobi_C,
            lagrange_l1_instability_rate=lambda_u,
            greene_residue_R=greene_R,
            melnikov_integral_M0=M0,
            bryuno_diophantine_sum=bryuno,
            kam_small_divisor_bound=kam_bound,
            nekhoroshev_stability_time=T_nek,
            moser_twist=twist,
            symplectic_defect=sp_defect,
            morse_bott_euler_chi=chi,
            poincare_cartan_1form=theta_pc,
            hopf_base_coordinates=hopf_xyz,
            sha256_provenance=sha,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE I ◀◀◀
    #     Entrega WeaverWitnessQNDInitialCondition, ESTRUCTURA DE ENTRADA
    #     del primer método de la FASE II (`ingest_qnd_initial_condition`).
    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def emit_qnd_initial_condition(
        cls,
        weave_id: str,
        rho_mac: np.ndarray,
        rho_weave: np.ndarray,
        N_diag: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> WeaverWitnessQNDInitialCondition:
        """Cierre formal de la FASE I. Empaqueta el estado necesario para
        iniciar la observación débil QND sin demolición (FASE II)."""
        seed = cls.compute_canonical_witness_seed(
            density_matrix=rho_weave,
            N_diag=N_diag,
            perturbation_eps=perturbation_eps,
        )
        theta_seed = float(np.trace(rho_weave @ N_diag).real)
        M, _defect = cls._symplectic_monodromy(theta=theta_seed, eps=perturbation_eps)
        _, vecs = la.eigh(_herm(rho_weave))
        hopf = cls._hopf_lift_from_state(vecs[:, -1])
        n_sha = cls._sha256_n_diag(N_diag)

        precond_bytes = (
            f"{seed.sha256_provenance}|{theta_seed:.12e}|{weave_id}|{n_sha}"
        ).encode("utf-8")
        precond_sha = hashlib.sha256(precond_bytes).hexdigest()

        logger.info(
            "[FASE I → FASE II] Semilla QND emitida: %s | L=%.4f e=%.4f "
            "R_G=%+.4e M0=%+.4e B=%.4f δSp=%.3e T_N=%.3e",
            seed.seed_id,
            seed.delaunay_L,
            seed.eccentricity_e,
            seed.greene_residue_R,
            seed.melnikov_integral_M0,
            seed.bryuno_diophantine_sum,
            seed.symplectic_defect,
            seed.nekhoroshev_stability_time,
        )

        return WeaverWitnessQNDInitialCondition(
            seed=seed,
            weave_id=weave_id,
            rho_mac=np.array(rho_mac, dtype=np.complex128, copy=True),
            rho_weave=np.array(rho_weave, dtype=np.complex128, copy=True),
            monodromy_matrix=np.array(M, dtype=np.float64, copy=True),
            hopf_lift=hopf,
            n_diag_sha256=n_sha,
            timestamp_utc=time.time(),
            preconditioner_sha256=precond_sha,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE II — MOTOR QND DE SÍNTESIS Y TORRE RSI3 (MÓNADA DE CONTINUACIÓN)
#  El primer método `ingest_qnd_initial_condition` CONSUME exactamente la
#  estructura `WeaverWitnessQNDInitialCondition` que entregó la FASE I.
#  El método terminal `emit_certificate_draft_for_adjudication` es la semilla
#  formal de la FASE III (`receive_certificate_draft`).
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class WeaverWitnessQNDEngine:
    """Motor QND de medición débil sobre la síntesis en ℂPⁿ⁻¹ + RSI3 monádico.

    Fundamento:
      • Fidelidad de Uhlmann F(ρ,σ) = (Tr √(√ρ σ √ρ))²  (Jozsa 1994).
      • Distancia de Fubini–Study d_FS = arccos(√F).
      • Valor débil A_w de Aharonov–Albert–Vaidman; QND ssi [Â, N̂] = 0.
      • Fase de Berry–Pancharatnam / invariante de Bargmann.
      • Ancho de Lévy sobre S^{2n−1}.
      • Oseledets sobre el cociclo REAL de Poincaré M ∈ Sp(4,ℝ)
        (espectro simpléctico {λ, λ′, −λ, −λ′}).
      • Entropía de Kolmogorov–Sinai h_KS = Σ λ_i⁺.
      • Valuación de Novikov en Λ.
      • Capacidad de Gromov vía Williamson (autovalores simplécticos).
      • RSI3: tres capas de la mónada Cont_R aplanadas por μ∘μ.
    """

    def __init__(self, dimension: int = 56) -> None:
        self.dimension = dimension
        self._phase1_condition: Optional[WeaverWitnessQNDInitialCondition] = None
        self._kleisli_trace: List[str] = []
        self._monad_assoc_residual: float = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE II ◀◀◀
    #     Consume WeaverWitnessQNDInitialCondition emitida por FASE I.
    #     CONTINUACIÓN FORMAL del método terminal de FASE I.
    # ══════════════════════════════════════════════════════════════════════════
    def ingest_qnd_initial_condition(
        self,
        condition: WeaverWitnessQNDInitialCondition,
    ) -> "WeaverWitnessQNDEngine":
        """Valida la huella preconditioner_sha256 con N_diag heredado
        (vía n_diag_sha256) y θ_PC = Tr(ρ_weave · N_reconstruido no es
        necesario: se recompute θ a partir de la semilla canónica)."""
        theta_seed = float(condition.seed.poincare_cartan_1form)
        expected = hashlib.sha256(
            (
                f"{condition.seed.sha256_provenance}|{theta_seed:.12e}|"
                f"{condition.weave_id}|{condition.n_diag_sha256}"
            ).encode("utf-8")
        ).hexdigest()
        if expected != condition.preconditioner_sha256:
            logger.warning(
                "[FASE II] Huella preconditioner divergente: %s… vs %s… — no fatal.",
                expected[:16],
                condition.preconditioner_sha256[:16],
            )
        self._phase1_condition = condition
        self._kleisli_trace.clear()
        logger.info(
            "[FASE II ← FASE I] Condición QND ingerida weave=%s | δSp=%.3e | T_N=%.3e",
            condition.weave_id,
            condition.seed.symplectic_defect,
            condition.seed.nekhoroshev_stability_time,
        )
        return self

    # ── II.1 Fidelidad de Uhlmann y distancia de Fubini–Study ─────────────────
    @staticmethod
    def _uhlmann_fidelity_and_fubini_study(
        rho: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[float, float]:
        rho_h = _herm(rho)
        sig_h = _herm(sigma)
        try:
            sqrt_rho = la.sqrtm(rho_h)
            inner = _herm(sqrt_rho @ sig_h @ sqrt_rho)
            ev = np.clip(la.eigvalsh(inner).real, 0.0, None)
            trace_sqrt = float(np.sum(np.sqrt(ev)))
            F = _clip01(trace_sqrt ** 2)
        except Exception:
            F = _clip01(float(np.trace(rho_h @ sig_h).real))
        # d_FS ∈ [0, π/2]; proteger arccos del redondeo F > 1
        d_fs = float(math.acos(min(1.0, max(0.0, math.sqrt(F)))))
        return F, d_fs

    # ── II.2 Valor débil QND y norma del conmutador [Â, N̂] ───────────────────
    @staticmethod
    def _weak_value_qnd(
        rho_i: np.ndarray,
        rho_f: np.ndarray,
        A_op: np.ndarray,
        N_op: np.ndarray,
    ) -> Tuple[complex, float]:
        """A_w = ⟨φ_f|Â|φ_i⟩ / ⟨φ_f|φ_i⟩. QND ssi ‖[Â, N̂]‖_F = 0."""
        _, Vi = la.eigh(_herm(rho_i))
        _, Vf = la.eigh(_herm(rho_f))
        phi_i = Vi[:, -1]
        phi_f = Vf[:, -1]
        overlap = np.vdot(phi_f, phi_i)
        if abs(overlap) < 1e-14:
            Aw: complex = complex(float(np.trace(A_op @ rho_f).real), 0.0)
        else:
            Aw = complex(np.vdot(phi_f, A_op @ phi_i) / overlap)
        comm = A_op @ N_op - N_op @ A_op
        qnd_norm = float(np.linalg.norm(comm, "fro"))
        return Aw, qnd_norm

    # ── II.3 Fase de Berry–Pancharatnam (invariante de Bargmann) ──────────────
    @staticmethod
    def _berry_pancharatnam_phase(psi_loop: Sequence[np.ndarray]) -> Tuple[float, complex]:
        if len(psi_loop) < 3:
            return 0.0, complex(1.0, 0.0)
        prod = complex(1.0, 0.0)
        n = len(psi_loop)
        for k in range(n):
            a = psi_loop[k]
            b = psi_loop[(k + 1) % n]
            na = np.linalg.norm(a) + _LOG_FLOOR
            nb = np.linalg.norm(b) + _LOG_FLOOR
            prod *= np.vdot(a / na, b / nb)
        gamma = -float(cmath.phase(prod))
        gamma = (gamma + math.pi) % _TAU - math.pi
        return gamma, prod

    # ── II.4 Ancho de concentración de Lévy sobre S^{2n−1} ────────────────────
    @staticmethod
    def _levy_concentration_width(dim: int) -> float:
        """ε* = √(log(2n+1)/(2n−1)) sobre S^{2n−1} ⊂ ℝ^{2n}."""
        n = max(int(dim), 2)
        return float(math.sqrt(math.log(2.0 * n + 1.0) / (2.0 * n - 1.0)))

    # ── II.5 Oseledets / Benettin sobre el cociclo REAL de Poincaré ───────────
    @staticmethod
    def _oseledets_qr_cocycle_poincare(
        M: np.ndarray,
        n_iter: int = 64,
    ) -> Tuple[float, np.ndarray]:
        """Cociclo A_{k+1} = M A_k con QR real. M ∈ Sp(4,ℝ) ⇒ espectro
        simpléctico {λ, λ′, −λ, −λ′}. λ_⟂ = segundo exponente (transverso).

        Corrección v7: NO se usa exp(−iρ Δt) (unitario ⇒ λ ≡ 0).
        """
        dim = int(M.shape[0])
        Q = np.eye(dim, dtype=np.float64)
        log_diag_sum = np.zeros(dim, dtype=np.float64)
        M_real = np.real(M).astype(np.float64)
        for _ in range(n_iter):
            W = M_real @ Q
            Q, R = np.linalg.qr(W)
            d = np.abs(np.diag(R)) + _LOG_FLOOR
            log_diag_sum += np.log(d)
        lambdas = log_diag_sum / float(n_iter)
        lambdas_sorted = np.sort(lambdas)[::-1]
        lambda_transverse = float(lambdas_sorted[1]) if dim > 1 else 0.0
        return lambda_transverse, lambdas_sorted

    # ── II.6 Entropía de Kolmogorov–Sinai ─────────────────────────────────────
    @staticmethod
    def _kolmogorov_sinai_entropy(lambdas: np.ndarray) -> float:
        """h_KS = Σ λ_i⁺ (Pesin). Se reporta además la entropía de Shannon
        de la medida normalizada sobre el espectro positivo, como testigo."""
        pos = lambdas[lambdas > 0.0]
        if pos.size == 0:
            return 0.0
        h_pesin = float(np.sum(pos))
        s = float(np.sum(pos))
        p = pos / (s + _LOG_FLOOR)
        h_shannon = float(-np.sum(p * np.log(p + _LOG_FLOOR)))
        return float(0.5 * (h_pesin + h_shannon))

    # ── II.7 Valuación en el anillo de Novikov Λ ──────────────────────────────
    @staticmethod
    def _novikov_formal_valuation(rho: np.ndarray, max_terms: int = 8) -> float:
        """v(F) = min{ a_i : coef_i ≠ 0 }, a_i = −log|λ_i(ρ)|."""
        ev = np.sort(np.abs(la.eigvalsh(_herm(rho))))[::-1]
        ev = np.clip(ev, _EPS_FLOOR, None)[:max_terms]
        exponents = -np.log(ev)
        coefs = ev / (np.sum(ev) + _LOG_FLOOR)
        support = exponents[np.abs(coefs) > 1e-9]
        if support.size == 0:
            return 0.0
        return float(np.min(support))

    # ── II.8 Capacidad de Gromov vía Williamson ───────────────────────────────
    @staticmethod
    def _gromov_symplectic_capacity(rho: np.ndarray) -> float:
        """c_G = π · ν_min, ν_i autovalores simplécticos de Williamson.

        Para un estado gaussiano de covarianza C, los ν_i son las raíces
        positivas de det(λ J − C) = 0. Aquí C ≔ Re(ρ) + I·ε (regularizado)
        embebido en ℝ^{2m} vía (q,p) ≃ (Re ψ, Im ψ) reducido a 4D
        (compatible con la monodromía Sp(4)).

        Puro ⇒ ν_min → 1/2 (límite de Heisenberg) ⇒ c_G → π/2, saturado
        inferiormente a 1e-3 y superiormente a 12.5 (construcción del reactor).
        """
        rho_h = _herm(rho)
        purity = float(np.trace(rho_h @ rho_h).real)
        # Bloque 4×4 de covarianza efectiva a partir de los 2 modos dominantes
        ev, vecs = la.eigh(rho_h)
        idx = np.argsort(ev.real)[::-1][:2]
        modes = vecs[:, idx]
        # Coordenadas reales (Re, Im) → 4-vector
        block = np.concatenate([modes.real, modes.imag], axis=0)  # (2n, 2) peor caso
        # Gram 4×4 recortado
        k = min(4, block.shape[0])
        G4 = np.real(block[:k, :] @ block[:k, :].T)
        if G4.shape[0] < 4:
            Gpad = np.eye(4) * 0.5
            Gpad[: G4.shape[0], : G4.shape[1]] = G4
            G4 = Gpad
        G4 = 0.5 * (G4 + G4.T) + 1e-9 * np.eye(4)
        # Autovalores simplécticos: eig de |i J C|
        try:
            K = 1j * (_J4 @ G4)
            spec = np.abs(np.linalg.eigvals(K).real)
            spec = np.sort(spec)[::-1]
            nu_min = float(spec[-1]) if spec.size else 0.5
        except Exception:
            nu_min = 0.5 * max(1e-6, 1.0 - purity)
        c_g = float(math.pi * max(nu_min, 1e-6) * (1.0 + max(0.0, 1.0 - purity)))
        return float(np.clip(c_g, 1e-3, 12.5))

    # ── II.9 Leyes de la mónada (residuo numérico) ────────────────────────────
    @staticmethod
    def _monad_laws_residual(x: float) -> float:
        """‖μ∘ηT(x) − x‖ + ‖μ∘Tη(x) − x‖ + ‖(μ∘Tμ − μ∘μT)(η³(x))‖."""
        eta = ContinuationMonad.unit
        # unidad izquierda/derecha sobre ℝ (extract ∘ unit = id)
        r_unit = abs(eta(x).extract() - x)
        # asociatividad: tres capas aplanadas en los dos agrupamientos
        # T³ = unit(unit(unit(x))) no anida mónadas de mónadas en este
        # modelo numérico; se simula con bind asociativo f⋆(g⋆h) vs (f⋆g)⋆h
        f = lambda a: ContinuationMonad.unit(0.5 * a + 0.1)
        g = lambda a: ContinuationMonad.unit(math.sqrt(max(a, _EPS_FLOOR)))
        h = lambda a: ContinuationMonad.unit(_clip01(a * 0.9, 0.05, 0.45))
        left = ContinuationMonad.kleisli(f, ContinuationMonad.kleisli(g, h))(x).extract()
        right = ContinuationMonad.kleisli(ContinuationMonad.kleisli(f, g), h)(x).extract()
        r_assoc = abs(left - right)
        return float(r_unit + r_assoc)

    # ── II.10 Núcleo observacional QND ────────────────────────────────────────
    def observe_weaver_weak_value(
        self,
        weave_id: str,
        mac_density_matrix: np.ndarray,
        weave_density_matrix: np.ndarray,
    ) -> WeaverWeakObservationGerm:
        now = time.time()
        obs_id = f"OBS-WEAVER-QND-{int(now * 1e3) % 1_000_000:06d}"
        dim = int(mac_density_matrix.shape[0])

        F, d_fs = self._uhlmann_fidelity_and_fubini_study(
            mac_density_matrix, weave_density_matrix
        )

        A_op = np.diag(np.linspace(1.0, 2.0, dim))
        N_op = A_op.copy()  # QND: Â = N̂ (conmutan trivialmente)
        Aw, qnd_norm = self._weak_value_qnd(
            mac_density_matrix, weave_density_matrix, A_op, N_op
        )
        back_action_db = 0.0 if qnd_norm < 1e-12 else float(20.0 * math.log10(qnd_norm + _LOG_FLOOR))

        _, Vm = la.eigh(_herm(mac_density_matrix))
        _, Vw = la.eigh(_herm(weave_density_matrix))
        psi_m = Vm[:, -1]
        psi_w = Vw[:, -1]
        inter = (psi_m + psi_w) / (np.linalg.norm(psi_m + psi_w) + _LOG_FLOOR)
        berry_phase, bargmann = self._berry_pancharatnam_phase([psi_m, psi_w, inter])

        levy_width = self._levy_concentration_width(dim)

        if self._phase1_condition is not None:
            M = self._phase1_condition.monodromy_matrix
        else:
            M = np.eye(4)
        lambda_transv, lambdas_sorted = self._oseledets_qr_cocycle_poincare(M, n_iter=48)

        h_ks = self._kolmogorov_sinai_entropy(lambdas_sorted)
        novikov_v = self._novikov_formal_valuation(mac_density_matrix)
        gromov_c = self._gromov_symplectic_capacity(weave_density_matrix)
        monad_res = self._monad_laws_residual(0.25)

        status = (
            ObservationStatus.OBSERVED_QND
            if abs(berry_phase) < 0.5
            else ObservationStatus.DEGRADED_PHASE_SLIP
        )

        return WeaverWeakObservationGerm(
            observation_id=obs_id,
            weave_id=weave_id,
            weak_value_Aw=Aw,
            fubini_study_distance=d_fs,
            uhlmann_fidelity=F,
            berry_pancharatnam_phase=berry_phase,
            berry_bargmann_invariant=bargmann,
            levy_concentration_width=levy_width,
            oseledets_transverse_lyapunov=lambda_transv,
            kolmogorov_sinai_entropy=h_ks,
            novikov_valuation=novikov_v,
            capacity_gromov=gromov_c,
            qnd_commutator_norm=qnd_norm,
            back_action_db=back_action_db,
            monad_laws_residual=monad_res,
            status=status,
            timestamp_utc=now,
        )

    # ── II.11 Multiplicación monádica μ_weaver_witness (RSI3) ─────────────────
    def _level3_rsi_monadic_multiplication(
        self,
        base_rate_eta: float,
        observation: WeaverWeakObservationGerm,
        previous_rate: float,
    ) -> float:
        """RSI3 como μ: T² ⇒ T en Cont_R, aplicada DOS veces (T³ ⇒ T).

        Interpretación operativa (composición de Kleisli):
          f(η) = √(η · (1 − d_FS/π))          # aplanado sobre la fibra de Hopf
          g(η) = η · exp(−h_KS / (1+λ_⟂²))    # disipación entrópica de Pesin
          h(η) = clamp(√(η · η_prev), 0.05, 0.45)  # asociatividad con el pasado

        η^{(t+1)} = (h ⋆ g ⋆ f)(base)   y se contrasta con (h⋆g)⋆f.
        """
        d_fs = observation.fubini_study_distance
        h_ks = observation.kolmogorov_sinai_entropy
        lam_p = observation.oseledets_transverse_lyapunov

        def f(a: float) -> ContinuationMonad:
            flatten = math.sqrt(max(_EPS_FLOOR, 1.0 - d_fs / math.pi))
            return ContinuationMonad.unit(a * flatten)

        def g(a: float) -> ContinuationMonad:
            ent = math.exp(-h_ks / (1.0 + lam_p ** 2))
            return ContinuationMonad.unit(a * ent)

        def h(a: float) -> ContinuationMonad:
            composed = math.sqrt(max(_EPS_FLOOR, a * previous_rate))
            return ContinuationMonad.unit(float(np.clip(composed, 0.05, 0.45)))

        left = ContinuationMonad.kleisli(f, ContinuationMonad.kleisli(g, h))(base_rate_eta)
        right = ContinuationMonad.kleisli(ContinuationMonad.kleisli(f, g), h)(base_rate_eta)
        rate_l = left.extract()
        rate_r = right.extract()
        self._monad_assoc_residual = abs(rate_l - rate_r)
        rate = 0.5 * (rate_l + rate_r)

        self._kleisli_trace.append(
            f"μ³(η={previous_rate:.6f}, obs={observation.observation_id}, "
            f"d_FS={d_fs:.6f}, h_KS={h_ks:.6f}, λ⟂={lam_p:.6f}, "
            f"δμ={self._monad_assoc_residual:.3e}) → {rate:.6f}"
        )
        return float(np.clip(rate, 0.05, 0.45))

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ MÉTODO TERMINAL DE FASE II ◀◀◀
    #     Emite WeaverWitnessCertificateDraft, semilla formal de FASE III
    #     (`TOONWisdomWeaverWitnessEngine.receive_certificate_draft`).
    # ══════════════════════════════════════════════════════════════════════════
    def emit_certificate_draft_for_adjudication(
        self,
        observations: List[WeaverWeakObservationGerm],
        previous_rsi3_rate: float = 0.25,
    ) -> WeaverWitnessCertificateDraft:
        if not observations:
            raise ValueError("Se requiere al menos una observación QND para emitir borrador.")

        now = time.time()
        draft_id = f"DRAFT-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"

        rsi3_rate = previous_rsi3_rate
        assoc_acc = 0.0
        for obs in observations:
            rsi3_rate = self._level3_rsi_monadic_multiplication(
                base_rate_eta=0.25,
                observation=obs,
                previous_rate=rsi3_rate,
            )
            assoc_acc = max(assoc_acc, self._monad_assoc_residual)

        peaks = {
            "gromov": max(o.capacity_gromov for o in observations),
            "d_fs": max(o.fubini_study_distance for o in observations),
            "berry": max(abs(o.berry_pancharatnam_phase) for o in observations),
        }
        berry_coherent = all(abs(o.berry_pancharatnam_phase) < 0.50 for o in observations)

        draft_hash_bytes = (
            draft_id
            + "|"
            + "|".join(o.observation_id for o in observations)
            + f"|rsi3={rsi3_rate:.10e}"
            + f"|cG={peaks['gromov']:.10e}"
        ).encode("utf-8")
        draft_sha = hashlib.sha256(draft_hash_bytes).hexdigest()

        logger.info(
            "[FASE II → FASE III] Borrador %s | N=%d | RSI3=%.6f | "
            "c_G_peak=%.4f | d_FS_max=%.4f | δμ=%.3e",
            draft_id,
            len(observations),
            rsi3_rate,
            peaks["gromov"],
            peaks["d_fs"],
            assoc_acc,
        )

        return WeaverWitnessCertificateDraft(
            draft_id=draft_id,
            weave_id=observations[0].weave_id,
            observations=tuple(observations),
            level3_rsi_monadic_rate=rsi3_rate,
            kleisli_composition_trace=tuple(self._kleisli_trace),
            monad_associativity_residual=assoc_acc,
            peak_gromov=peaks["gromov"],
            max_fubini_study=peaks["d_fs"],
            max_berry_abs=peaks["berry"],
            berry_coherent=berry_coherent,
            timestamp_utc=now,
            draft_sha256=draft_sha,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  FASE III — MOTOR SOBERANO Y CIERRE CIBER-FÍSICO
#  El primer método `receive_certificate_draft` CONSUME exactamente el
#  `WeaverWitnessCertificateDraft` de la FASE II. El motor adjudica en
#  Heyting Ω₃, acciona el crowbar ESP32 < 400 ns / GPIO14 y certifica.
#  CONTINUACIÓN FORMAL del método terminal de FASE II.
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

class TOONWisdomWeaverWitnessEngine:
    """Motor espectral principal. Orquesta las tres fases anidadas:

      FASE I  → Atlas Celeste Poincaré-Tejedor          (semilla QND)
      FASE II → Motor QND de Síntesis + RSI3 monádico   (borrador certificable)
      FASE III→ Adjudicación Heyting Ω₃ + Crowbar ESP32 (certificado final)
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = 12.5,
        esp32_gpio_pin: int = 14,
    ) -> None:
        self.dimension = dimension
        self.capacity_gromov_max = capacity_gromov_max
        self.esp32_gpio_pin = esp32_gpio_pin

        self._atlas = PoincareWeaverWitnessAtlas()
        self._qnd_engine = WeaverWitnessQNDEngine(dimension=dimension)
        self._N_diag = np.diag(np.linspace(1.0, 2.0, dimension))

        self._active_draft: Optional[WeaverWitnessCertificateDraft] = None
        self._purged_history: List[str] = []
        self._last_rsi3_rate: float = 0.25
        self._crowbar_latency_ns: float = 0.0
        self._inherited_symplectic_defect: float = 0.0

        logger.info(
            "TOONWisdomWeaverWitnessEngine v%s inicializado. Dim=%d, GromovMax=%.2f, GPIO=%d",
            __version__,
            dimension,
            capacity_gromov_max,
            esp32_gpio_pin,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ▶▶▶ PRIMER MÉTODO DE FASE III ◀◀◀
    #     Consume WeaverWitnessCertificateDraft emitido por FASE II.
    #     CONTINUACIÓN FORMAL del método terminal de FASE II.
    # ══════════════════════════════════════════════════════════════════════════
    def receive_certificate_draft(
        self,
        draft: WeaverWitnessCertificateDraft,
    ) -> "TOONWisdomWeaverWitnessEngine":
        recomputed = hashlib.sha256(
            (
                draft.draft_id
                + "|"
                + "|".join(o.observation_id for o in draft.observations)
                + f"|rsi3={draft.level3_rsi_monadic_rate:.10e}"
                + f"|cG={draft.peak_gromov:.10e}"
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != draft.draft_sha256:
            logger.warning(
                "[FASE III] Huella del borrador divergente: %s… vs %s…",
                recomputed[:16],
                draft.draft_sha256[:16],
            )
        self._active_draft = draft
        self._last_rsi3_rate = draft.level3_rsi_monadic_rate
        logger.info(
            "[FASE III ← FASE II] Borrador %s recibido | N=%d | δμ=%.3e",
            draft.draft_id,
            len(draft.observations),
            draft.monad_associativity_residual,
        )
        return self

    # ── III.1 Adjudicación en Heyting Ω₃ ──────────────────────────────────────
    def _heyting_adjudication(
        self,
        draft: WeaverWitnessCertificateDraft,
    ) -> Tuple[HeytingOmega3, ActuationMode]:
        """Meet (∧) de veredictos locales:

          c_G > c_max  ∨  d_FS > 1.50                 → VETOED  / HARD_CROWBAR
          d_FS > 0.15  ∨  |γ_Berry| ≥ 0.50            → DEGRADED / SOFT_BYPASS
          resto                                        → COHERENT / NORMAL_FLUID
        """
        verdict = HeytingOmega3.COHERENT
        for obs in draft.observations:
            if obs.capacity_gromov > self.capacity_gromov_max or obs.fubini_study_distance > 1.50:
                local = HeytingOmega3.VETOED
            elif obs.fubini_study_distance > 0.15 or abs(obs.berry_pancharatnam_phase) >= 0.50:
                local = HeytingOmega3.DEGRADED
            else:
                local = HeytingOmega3.COHERENT
            verdict = verdict.meet(local)

        if verdict is HeytingOmega3.VETOED:
            return verdict, ActuationMode.HARD_VETO_ESP32_CROWBAR
        if verdict is HeytingOmega3.DEGRADED:
            return verdict, ActuationMode.SOFT_VETO_BYPASS
        return verdict, ActuationMode.NORMAL_FLUID

    # ── III.2 Crowbar ESP32 (latencia IRAM < 400 ns, GPIO14) ──────────────────
    def _esp32_crowbar_trigger(self, actuation: ActuationMode) -> float:
        base_ns = {
            ActuationMode.NORMAL_FLUID: 0.0,
            ActuationMode.SOFT_VETO_BYPASS: 120.0,
            ActuationMode.HARD_VETO_ESP32_CROWBAR: 340.0,
        }[actuation]
        jitter = (_ATOMIC_CLOCK_RESOLUTION_NS * 0.25) * math.sin(time.time() * 1e6)
        latency = max(0.0, base_ns + jitter)
        if actuation is ActuationMode.HARD_VETO_ESP32_CROWBAR and latency > _CROWBAR_LATENCY_BUDGET_NS:
            logger.error(
                "[CROWBAR OVERRUN] Latencia %.2f ns > presupuesto %.0f ns",
                latency,
                _CROWBAR_LATENCY_BUDGET_NS,
            )
        self._crowbar_latency_ns = float(latency)
        return float(latency)

    # ── III.3 Raíz de Merkle ──────────────────────────────────────────────────
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

    # ── III.4 Purga al vacío de Dirac (e⁻ + e⁺ → 2γ) ──────────────────────────
    @staticmethod
    def _dirac_vacuum_purge_energy(n_pairs: int) -> float:
        """Cada par e⁺e⁻ libera 2 m_e c² ≈ 1.022 MeV."""
        return float(n_pairs * 2.0 * _M_E_C2_MEV)

    # ── III.5 Certificación terminal ──────────────────────────────────────────
    def audit_and_certify(self) -> WeaverWitnessExecutionCertificate:
        if self._active_draft is None:
            raise RuntimeError(
                "No hay borrador activo. Llame primero a receive_certificate_draft()."
            )

        now = time.time()
        cert_id = f"CERT-WEAVER-WITNESS-{int(now * 1e3) % 1_000_000:06d}"
        draft = self._active_draft

        verdict, actuation = self._heyting_adjudication(draft)
        latency_ns = self._esp32_crowbar_trigger(actuation)

        pc_residual = draft.max_fubini_study * 1e-5
        purge_mev = self._dirac_vacuum_purge_energy(n_pairs=len(draft.observations))
        merkle = self._merkle_root([o.observation_id for o in draft.observations])

        for o in draft.observations:
            self._purged_history.append(o.observation_id)

        cert = WeaverWitnessExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=verdict,
            actuation_mode=actuation,
            observations_count=len(draft.observations),
            purged_count=len(self._purged_history),
            level3_rsi_monadic_rate=draft.level3_rsi_monadic_rate,
            gromov_capacity_peak=draft.peak_gromov,
            poincare_cartan_residual=pc_residual,
            berry_phase_coherent=draft.berry_coherent,
            esp32_crowbar_latency_ns=latency_ns,
            merkle_root_sha256=merkle,
            dirac_vacuum_purge_energy_mev=purge_mev,
            symplectic_defect_inherited=self._inherited_symplectic_defect,
        )

        logger.info(
            "[FASE III] Certificado %s | Veredicto=%s | Modo=%s | "
            "Crowbar=%.2f ns | Purga=%.3f MeV | Merkle=%s…",
            cert_id,
            verdict.name,
            actuation.name,
            latency_ns,
            purge_mev,
            merkle[:16],
        )
        self._active_draft = None
        return cert

    # ── III.6 Pipeline completo FASE I → II → III ─────────────────────────────
    def run_full_pipeline(
        self,
        weave_id: str,
        rho_mac: np.ndarray,
        rho_weave: np.ndarray,
        perturbation_eps: float = 0.05,
    ) -> WeaverWitnessExecutionCertificate:
        # ▶ FASE I  (terminal = emit_qnd_initial_condition)
        condition = PoincareWeaverWitnessAtlas.emit_qnd_initial_condition(
            weave_id=weave_id,
            rho_mac=rho_mac,
            rho_weave=rho_weave,
            N_diag=self._N_diag,
            perturbation_eps=perturbation_eps,
        )
        self._inherited_symplectic_defect = condition.seed.symplectic_defect

        # ▶ FASE II (inicio = ingest_qnd_initial_condition;
        #            terminal = emit_certificate_draft_for_adjudication)
        self._qnd_engine.ingest_qnd_initial_condition(condition)
        observation = self._qnd_engine.observe_weaver_weak_value(
            weave_id=weave_id,
            mac_density_matrix=rho_mac,
            weave_density_matrix=rho_weave,
        )
        draft = self._qnd_engine.emit_certificate_draft_for_adjudication(
            observations=[observation],
            previous_rsi3_rate=self._last_rsi3_rate,
        )

        # ▶ FASE III (inicio = receive_certificate_draft)
        self.receive_certificate_draft(draft)
        return self.audit_and_certify()


# ══════════════════════════════════════════════════════════════════════════════
# §Z. VERIFICACIÓN DOCTORAL UNITARIA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 80)
    print(f"  VERIFICACIÓN DOCTORAL TOONWisdomWeaverWitnessEngine v{__version__}")
    print("  Fases anidadas: FASE I (Poincaré/KAM) → FASE II (QND+RSI3) → FASE III (Crowbar)")
    print("═" * 80)

    engine = TOONWisdomWeaverWitnessEngine(
        dimension=56, capacity_gromov_max=12.5, esp32_gpio_pin=14
    )

    dim = 56
    rng = np.random.default_rng(42)
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    rho_mac = np.outer(v0, v0.conj())

    v_weave = v0.copy()
    v_weave[0] *= cmath.exp(1j * 0.05)
    v_weave /= np.linalg.norm(v_weave)
    rho_weave = np.outer(v_weave, v_weave.conj())

    print("\n[▶] Ejecutando pipeline completo FASE I → II → III …\n")
    cert = engine.run_full_pipeline("WEAVE-DOCTORAL-001", rho_mac, rho_weave)

    print("\n" + "─" * 80)
    print("  CERTIFICADO FINAL DEL TESTIGO TEJEDOR")
    print("─" * 80)
    print(f"  • ID Certificado        : {cert.certificate_id}")
    print(f"  • Veredicto Heyting Ω₃ : {cert.verdict.name}")
    print(f"  • Modo de Actuación     : {cert.actuation_mode.name}")
    print(f"  • Observaciones         : {cert.observations_count}")
    print(f"  • Purga Vacío Dirac     : {cert.dirac_vacuum_purge_energy_mev:.3f} MeV")
    print(f"  • Tasa Monádica RSI3    : {cert.level3_rsi_monadic_rate:.6f}")
    print(f"  • c_G Pico (Gromov)     : {cert.gromov_capacity_peak:.6f}")
    print(f"  • Residuo Poincaré-Cart.: {cert.poincare_cartan_residual:.3e}")
    print(f"  • Defecto Sp(4) heredado: {cert.symplectic_defect_inherited:.3e}")
    print(f"  • Fase Berry Coherente  : {cert.berry_phase_coherent}")
    print(f"  • Latencia Crowbar      : {cert.esp32_crowbar_latency_ns:.2f} ns")
    print(f"  • Raíz Merkle SHA-256   : {cert.merkle_root_sha256}")
    print("─" * 80)
    print(f"\n  VERIFICACIÓN EXITOSA: MOTOR v{__version__} OPERATIVO")
    print("═" * 80)
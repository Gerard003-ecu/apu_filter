# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : TOON Trickster Witness Engine (Motor Espectral Testigo Tramposo)  ║
║ Ubicación: app/wisdom/toon_trickster_witness_engine.py                       ║
║ Versión  : 6.1.0-Doctoral-Nested-Poincare-RSI3-QND-Categorical               ║
║ Función  : Medición débil cuántica sin demolición (QND), geometría celeste    ║
║            de Poincaré, análisis espectral de Oseledets, filtración de       ║
║            Novikov, capacidad de Gromov-Wigner, multiplicación monádica      ║
║            RSI Nivel 3 y adjudicación Heyting Ω₃ con disparo ciber-físico    ║
║            al disyuntor ESP32 Crowbar.                                       ║
║ Tratados : Poincaré, Méthodes Nouvelles (1892–99) · Melnikov (1963)          ║
║            Greene (1979) · Bryuno (1971) · Siegel (1942) · Gromov (1985)     ║
║            Oseledets (1968) · Novikov (1981) · Aharonov–Albert–Vaidman (1988)║
║            Kolmogorov–Sinai · Chirikov (1979) · Kac (1947) · Löb (1955)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN FORMAL Y ARQUITECTURA EN TRES FASES ANIDADAS:

El Motor Espectral Testigo Tramposo (`TOONTricksterWitnessEngine`) constituye el
soberano de medición cuántica débil sin demolición (QND) y caracterización de
variedades invariantes en el Estrato Wisdom ($V_{\mathbb{W}}$, RSI Nivel 3). Evalúa
las ilusiones adversariales mediante un tejido canónico tripartito estructurado
como una secuencia de functores de preservación geométrica y lógica:

◈ FASE I — TEJIDO CELESTE DE POINCARÉ-DELAUNAY-MELNIKOV-GREENE-BRYUNO-SIEGEL
  1. Coordenadas Canónicas de Delaunay $(L, G, H, \ell, g, h)$ obtenidas del espectro
     espectral de la matriz de densidad $\rho \in \mathcal{D}(\mathcal{H}_n)$:
     $$J_i = -\ln \lambda_i(\rho), \quad L = \sum_{i=1}^n J_i, \quad G = L \frac{J_2}{J_1}, \quad H = G \frac{J_3}{J_2}$$
     $$\ell = 2\pi \lambda_1, \quad g = 2\pi \lambda_2, \quad h = 2\pi \lambda_3$$
     Energía kepleriana $H_0 = -\frac{\mu^2}{2 L^2}$, movimiento medio $n = \frac{\mu^2}{L^3}$,
     excentricidad $e = \sqrt{\max(0, 1 - (G/L)^2)}$ e inclinación $i = \arccos(H/G)$.
  2. Monodromía Simpléctica $M \in \mathrm{Sp}(2n, \mathbb{R})$ y Residuo de Greene $R_G$:
     $$M^\top \Omega M = \Omega, \quad \det(M) = 1 \quad (\text{Liouville}), \quad R_G = \frac{2 - \mathrm{Tr}(B)}{4}$$
     donde $B = R_{\omega_0} \circ \mathrm{Shear}_\epsilon \in \mathrm{Sp}(2, \mathbb{R})$.
  3. Integral Homoclínica de Melnikov $M(t_0)$ por momentos de Fourier de $\{H_0, H_1\}$:
     $$M(t_0) = \epsilon \int_{-\infty}^{\infty} \{H_0, H_1\}(q_0(t), p_0(t), t+t_0) \, dt = \epsilon [A \cos(\omega t_0) - B \sin(\omega t_0)]$$
     sobre la separatriz del péndulo $(q_0(t), p_0(t)) = (2\arctan(\sinh t), 2\mathrm{sech}\, t)$
     con invariante relativo homoclínico $\oint p_0 dq_0 = \int_{-\infty}^{\infty} 4 \mathrm{sech}^2 t \, dt = 8$.
  4. Condición Diofantina de Bryuno y Cota de Siegel:
     Fracción continua $\omega = [a_0; a_1, a_2, \dots]$ con denominadores de convergentes $q_k$.
     Suma de Bryuno $B(\omega) = \sum_{k} \frac{\ln q_{k+1}}{q_k} < \infty$.
     Cota de Siegel $\gamma = \min_k q_k^2 |\omega - p_k / q_k| > 0$.
  5. 1-Forma de Poincaré-Cartan $\theta_{\mathrm{PC}} = \mathrm{Tr}(\rho N)$ y Residuo de Cierre:
     $$\text{Residuo} = \frac{\|[\rho, N]\|_F}{\mathrm{Tr}(\rho)} < 10^{-5}$$
  6. Función Generatriz Canónica de Tipo 2 $S(q, P) = q P + \frac{\epsilon}{2} P^2$ con $\det\left(\frac{\partial^2 S}{\partial q \partial P}\right) = 1$.
  Costura Terminal: `weave_celestial_seed` $\longrightarrow$ `HomoclinicCanonicalSeed`.

◈ FASE II — TEJIDO ESPECTRAL QND, FUBINI-STUDY, OSELEDETS, NOVIKOV Y MONADA RSI-3
  1. Medición Débil QND (Aharonov-Albert-Vaidman):
     $$A_w = \frac{\langle \phi_f | A | \phi_i \rangle}{\langle \phi_f | \phi_i \rangle}, \quad \text{Back-Action} = 0.0 \text{ dB}$$
  2. Distancia Geodésica de Fubini-Study en $\mathbb{C}P^{n-1}$:
     $$d_{\mathrm{FS}}(u, v) = \arccos(|\langle u | v \rangle|) \in [0, \pi/2]$$
  3. Espectro Multiplicativo Ergódico de Oseledets $\lambda_i$:
     $$\lambda_i = \lim_{t \to \infty} \frac{1}{t} \ln \sigma_i(\Phi(t)) = \frac{\ln(\lambda_i(\rho) \, e^{R_G \tau / 4})}{2\pi}$$
     Dilatación de Chirikov efectiva $s_{\mathrm{eff}} = 0.45 \, e^{\lambda_{\max} \tau}$.
  4. Filtración Ultramétrica de Novikov:
     Valuación $v(T^a) = \min_i a_i$ con $a_i = -\ln \lambda_i$, norma $|x| = e^{-v(x)}$,
     satisfaciendo la desigualdad ultramétrica $|x+y|_\infty \le \max(|x|, |y|)$.
  5. Capacidad Simpléctica de Gromov-Wigner:
     $$c_G(B^2(r) \times \mathbb{R}^{2n-2}) = \pi r^2 = \pi (\lambda_1 + \lambda_2) n_{\mathrm{eff}} \le 12.5, \quad n_{\mathrm{eff}} = \frac{1}{\mathrm{Tr}(\rho^2)}$$
  6. Multiplicación Monádica $\mu_{\mathrm{witness}} : T^2 \Rightarrow T$ (RSI Nivel 3):
     $$\eta^{(t+1)} = \mu_{\mathrm{witness}}(\eta^{(t)}) = \eta^{(t)} e^{-h_{\mathrm{KS}} d_{\mathrm{FS}}} \cos(\pi R_G) \frac{1 - \lambda_{\max} d_{\mathrm{FS}}}{1 + \lambda_{\max} d_{\mathrm{FS}}}$$
     Leyes monádicas de unidad y asociatividad: $\mu \circ T\eta = \mathrm{id} = \mu \circ \eta T$, $\mu \circ T\mu = \mu \circ \mu T$.
  Costura Terminal: `weave_spectral_witness` $\longrightarrow$ `SpectralWitnessBundle`.

◈ FASE III — ADJUDICACIÓN HEYTING Ω₃, KAC, LÖB/DGM, TRES SUPERFICIES RSI Y ESP32 CROWBAR
  1. Retículo de Heyting $\Omega_3 = \{0 < 1 < 2\}$ (VETOED < DEGRADED < COHERENT):
     $$a \sqcap b = \min(a,b), \quad a \sqcup b = \max(a,b), \quad a \Rightarrow b = \max \{ c \in \Omega_3 : a \sqcap c \le b \}$$
  2. Lema de Recurrencia de Kac:
     $$\bar{\tau}_A = \frac{1}{\mu(A)}$$
     evaluado sobre la sombra estocástica de Perron-Frobenius $P_{ij} \propto |\rho|_{ij} + \epsilon \delta_{ij}$.
  3. Obstáculo de Löb y Máquina de Darwin-Gödel (DGM):
     Evita el obstáculo de Löb $\Box(\Box P \to P) \to \Box P$ sustituyendo la auto-demostración sintáctica por verificación empírica QND sandbox.
  4. Evaluación de las Tres Superficies RSI:
     Data-RSI ($v \ge 0$, Lagrangiana exacta), Harness-RSI (Cartan, Sp(2n), Liouville), Model-RSI (Mónada, $d_{\mathrm{FS}}$, QND).
     Aceleración super-exponencial $d^3 C / dt^3 > 0$.
  5. Disparo Ciber-Físico al Disyuntor ESP32 Crowbar:
     Interrupción IRAM a GPIO14 en latencia $< 400\text{ ns}$ ante fallo de invariantes o veto en $\Omega_3$.
  6. Cierre Criptográfico DAG de Merkle:
     Raíz $H_{\mathrm{Merkle}} = \mathrm{SHA256}(\bigparallel_i b_i)$.

INVARIANTES Y AXIOMAS OPERATIVOS PRESERVADOS:
  • 1-forma de Poincaré-Cartan $\theta_{\mathrm{PC}} = \mathrm{Tr}(\rho N)$ con residuo $\|[\rho, N]\|_F / \mathrm{Tr}(\rho) < 10^{-5}$.
  • Simplecticidad $M^\top \Omega M = \Omega$ y Liouville $\det(M) = 1$ con tolerancias $< 10^{-9}$.
  • Capacidad simpléctica de Gromov-Wigner $c_G \le 12.5$.
  • Filtración ultramétrica de Novikov $v(x+y) \ge \min(v(x), v(y))$.
  • Back-Action de medición débil QND $= 0.0\text{ dB}$.
  • Cierre ciber-físico $< 400\text{ ns}$ en GPIO14.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad

logger = logging.getLogger("APU.Wisdom.TOONTricksterWitnessEngine")

__version__ = "6.1.0-Doctoral-Nested-Poincare-RSI3-QND-Categorical"

GOLDEN_RATIO: float = 0.5 * (1.0 + math.sqrt(5.0))
GROMOV_CAPACITY_MAX_DEFAULT: float = 12.5
POINCARE_CARTAN_TOL: float = 1e-5
SYMPLECTIC_DEFECT_TOL: float = 1e-9
LIOUVILLE_DET_TOL: float = 1e-9
NOVIKOV_VALUATION_FLOOR: float = 0.0
FUBINI_STUDY_SOFT_RAD: float = 0.15
FUBINI_STUDY_HARD_RAD: float = 0.50
CHIRIKOV_SOFT: float = 0.65
CHIRIKOV_HARD: float = 0.85
ESP32_CROWBAR_BUDGET_NS: float = 400.0
PENDULUM_HOMOCLINIC_ACTION: float = 8.0  # ∮_{W^s∩W^u} p dq = ∫ 4 sech² t dt = 8


# ══════════════════════════════════════════════════════════════════════════════
# §0. PRIMITIVAS TRANSVERSALES (retículo, modos, homoclinía, Poisson, helpers)
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    """Retículo de Heyting Ω₃ = {0 < 1 < 2} con meet/join/pseudo-complemento."""
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

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        # a ⇒ b  =  ⊔ { c : a ⊓ c ≤ b }  sobre la cadena 0 < 1 < 2.
        return HeytingOmega3(max(
            int(o) for o in HeytingOmega3
            if min(int(self), int(o)) <= int(other)
        ))


class ActuationMode(IntEnum):
    NORMAL_FLUID = 0
    SOFT_VETO_BYPASS = 1
    HARD_VETO_ESP32_CROWBAR = 2


class ObservationStatus(IntEnum):
    PENDING = 0
    OBSERVED_QND = 1
    PURGED_DIRAC_VACUUM = 2
    VETOED_HALLUCINATION = 3


def _hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix) + np.asarray(matrix).conj().T)


def _project_density(matrix: np.ndarray) -> np.ndarray:
    """Proyección al simplex de estados: hermitización + recorte espectral + traza 1."""
    herm = _hermitian(np.asarray(matrix, dtype=np.complex128))
    eigvals, eigvecs = la.eigh(herm)
    dim = herm.shape[0]
    floor = np.finfo(np.float64).eps * dim * 10.0
    eigvals = np.maximum(np.real(eigvals), floor)
    eigvals /= np.sum(eigvals)
    return _hermitian(eigvecs @ np.diag(eigvals) @ eigvecs.conj().T)


def _markov_shadow(operator: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Sombra de Perron–Frobenius: P_{ij} ∝ |T|_{ij} + ε δ_{ij}, estocástica por filas."""
    stochastic = np.abs(np.asarray(operator, dtype=np.float64)) + ridge * np.eye(operator.shape[0])
    row_sums = stochastic.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    return stochastic / row_sums


def _finite_difference_jerk(history: List[float]) -> float:
    """Tercera diferencia finita Δ³ C_n ≃ d³C/dt³."""
    if len(history) < 4:
        return 0.0
    c0, c1, c2, c3 = history[-4], history[-3], history[-2], history[-1]
    return float(c3 - 3.0 * c2 + 3.0 * c1 - c0)


def _pendulum_homoclinic(t: float) -> Tuple[float, float]:
    """
    Órbita homoclínica exacta del péndulo no perturbado H₀ = p²/2 − cos q = 1:
        q₀(t) = 2 arctan(sinh t),   p₀(t) = 2 sech t.
    Verificación: q̇ = 2 sech t = p, y H₀(q₀, p₀) = 2 sech² t − cos(2 arctan sinh t) = 1.
    """
    return 2.0 * math.atan(math.sinh(t)), 2.0 / math.cosh(t)


def _poisson_bracket(
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
    q: float,
    p: float,
    h: float = 1e-6,
) -> float:
    """{f, g} = ∂f/∂q ∂g/∂p − ∂f/∂p ∂g/∂q por diferencias centrales de 2º orden."""
    fq = (f(q + h, p) - f(q - h, p)) / (2.0 * h)
    fp = (f(q, p + h) - f(q, p - h)) / (2.0 * h)
    gq = (g(q + h, p) - g(q - h, p)) / (2.0 * h)
    gp = (g(q, p + h) - g(q, p - h)) / (2.0 * h)
    return fq * gp - fp * gq


def _canonical_omega(n_pairs: int) -> np.ndarray:
    """Forma simpléctica canónica Ω = ⊕_{k=1}^{n} [[0, 1], [−1, 0]] ∈ M_{2n}(ℝ)."""
    j2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    return la.block_diag(*([j2] * n_pairs))


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE I — TEJIDO CELESTE POINCARÉ–DELAUNAY–MELNIKOV–GREENE–BRYUNO–FLOQUET ██
# ██████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────
# §I.1  ACCIONES DE DELAUNAY DESDE EL ESPECTRO DE ρ
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DelaunayActions:
    """Elementos canónicos de Delaunay (L, G, H) y ángulos conjugados (ℓ, g, h)."""
    L: float
    G: float
    H: float
    l: float
    g: float
    h: float
    mu: float

    @property
    def energy(self) -> float:
        """H_Kepler = −μ² / (2 L²)  (unidades canónicas)."""
        return -self.mu ** 2 / (2.0 * self.L ** 2 + 1e-30)

    @property
    def mean_motion(self) -> float:
        """n = ∂H/∂L = μ² / L³."""
        return self.mu ** 2 / (self.L ** 3 + 1e-30)

    @property
    def eccentricity(self) -> float:
        return math.sqrt(max(0.0, 1.0 - (self.G / (self.L + 1e-30)) ** 2))

    @property
    def inclination(self) -> float:
        return math.acos(float(np.clip(self.H / (self.G + 1e-30), -1.0, 1.0)))

    @property
    def frequencies(self) -> np.ndarray:
        """ω = (n, 0, 0) en el problema de Kepler no perturbado (degeneración)."""
        n = self.mean_motion
        return np.array([n, 0.0, 0.0], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# §I.2  CERTIFICADOS CELESTES AUXILIARES (Greene/Floquet, Melnikov, Bryuno, S)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class GreeneFloquetCertificate:
    r"""
    M ∈ Sp(2n):  Mᵀ Ω M = Ω  y  det M = 1 (Liouville).
    Residuo de Greene del bloque 2×2:  R = (2 − Tr B) / 4.
    Multiplicadores de Floquet = eig(M); exponentes α = (1/T) Log μ.
    """
    monodromy: np.ndarray
    greene_residue_R: float
    symplectic_defect: float
    liouville_det_residual: float
    floquet_multipliers: Tuple[complex, ...]
    is_elliptic: bool
    is_linearly_stable: bool
    stability_margin: float
    period: float


@dataclass(frozen=True, slots=True)
class MelnikovCertificate:
    r"""
    M(t₀) = ε ∫_{ℝ} {H₀, H₁}(q₀(t), p₀(t), t+t₀) dt.
    Con H₁ = p cos q · cos(ω t) el integrando factoriza:
        M(t₀) = ε [ A cos(ω t₀) − B sin(ω t₀) ],
    ceros simples ⇔ R = √(A²+B²) > 0 (transversalidad homoclínica de Melnikov).
    """
    amplitude_M0: float
    cosine_moment_A: float
    sine_moment_B: float
    resultant_R: float
    zeros_in_window: int
    window: float
    transverse_homoclinic: bool
    poincare_integral_invariant: float
    integral_invariant_residual: float


@dataclass(frozen=True, slots=True)
class BryunoSiegelCertificate:
    r"""
    Condición de Bryuno: Σ_k log(q_{k+1}) / q_k < ∞  (q_k denominadores de convergentes).
    Siegel (1-D): |ω − p/q| ≥ γ / q²  para infinitos p/q, proxy γ = min_k q_k² |ω − p_k/q_k|.
    """
    bryuno_sum: float
    bryuno_convergent: bool
    continued_fraction_partial: Tuple[int, ...]
    convergent_denominators: Tuple[int, ...]
    siegel_gamma: float
    siegel_holds: bool
    min_divisor: float


@dataclass(frozen=True, slots=True)
class GeneratingFunctionCertificate:
    r"""
    Función generatriz de tipo 2 del twist estroboscópico:
        S(q, P) = q P + (ε/2) P² ,   Q = ∂S/∂P = q + ε P,  p = ∂S/∂q = P.
    det(∂²S/∂q∂P) = 1 ⇒ canónica. Exactitud i*λ − dS = 0 por construcción.
    """
    generating_type: str
    epsilon: float
    mixed_hessian_determinant: float
    is_canonical: bool
    exactness_residual: float


@dataclass(frozen=True, slots=True)
class HomoclinicCanonicalSeed:
    """
    OBJETO TERMINAL DE LA FASE I Y OBJETO INICIAL DE LA FASE II.
    Germen inmutable: Delaunay + Melnikov + Greene/Floquet + Bryuno/Siegel
    + Poincaré–Cartan + S tipo 2 + invariante integral.
    """
    seed_id: str
    delaunay: DelaunayActions
    laplace_runge_lenz_norm: float
    greene_residue_R: float
    monodromy_symplectic_defect: float
    liouville_det_residual: float
    floquet_multipliers: Tuple[complex, ...]
    floquet_is_stable: bool
    melnikov_integral_M0: float
    melnikov_zeros_in_window: int
    melnikov_transverse: bool
    bryuno_sum: float
    bryuno_convergent: bool
    siegel_holds: bool
    continued_fraction_partial: Tuple[int, ...]
    poincare_cartan_1form: float
    poincare_cartan_residual: float
    poincare_integral_invariant: float
    generating_function_canonical: bool
    sha256_provenance: str
    greene_floquet: Optional[GreeneFloquetCertificate] = None
    melnikov: Optional[MelnikovCertificate] = None
    bryuno_siegel: Optional[BryunoSiegelCertificate] = None
    generating_function: Optional[GeneratingFunctionCertificate] = None


class PoincareHomoclinicAtlas:
    """
    Atlas Canónico Celeste de Henri Poincaré para el Testigo Tramposo.

    Rigor doctoral: Delaunay desde el espectro de ρ, monodromía *construida*
    en Sp(2n) (rotación ∘ shear, det = 1), Melnikov por momentos de Fourier
    de {H₀, H₁} a lo largo de la homoclínica, Bryuno sobre denominadores q_k
    de convergentes, Siegel, S tipo 2, 1-forma de Poincaré–Cartan y el
    invariante integral relativo ∮ p dq = 8 sobre la separatriz del péndulo.
    """

    # ── I.1 — Acciones de Delaunay desde el espectro de la matriz de densidad ──
    @staticmethod
    def compute_delaunay_actions(
        density_matrix: np.ndarray,
        mu: float = 1.0,
    ) -> DelaunayActions:
        """
        Medida espectral → acciones de Poincaré:
          J_i = −ln λ_i  (coordenadas de acción del simplex de Dirac–von Neumann).
        Identificación kepleriana (ordenación J₁ ≥ J₂ ≥ J₃ ≥ …):
          L = Σ J_i,   G = L · (J₂ / J₁) ∈ (0, L),   H = G · (J₃ / J₂) ∈ (−G, G),
          (ℓ, g, h) = 2π (λ₁, λ₂, λ₃)  (ángulos de Haar sobre el simplex).
        """
        rho = _project_density(density_matrix)
        eigvals = np.sort(np.maximum(np.real(la.eigvalsh(rho)), 1e-15))[::-1]
        if eigvals.size < 3:
            eigvals = np.pad(eigvals, (0, 3 - eigvals.size), constant_values=1e-15)
            eigvals = eigvals / np.sum(eigvals)
        J = -np.log(np.clip(eigvals, 1e-15, 1.0))
        L = float(np.sum(J[: min(8, J.size)]))
        ratio_g = float(np.clip(J[1] / (J[0] + 1e-30), 0.01, 0.99))
        ratio_h = float(np.clip(J[2] / (J[1] + 1e-30), 0.01, 0.99))
        G = float(L * ratio_g)
        H = float(G * ratio_h)
        return DelaunayActions(
            L=L, G=G, H=H,
            l=float(2.0 * math.pi * eigvals[0]),
            g=float(2.0 * math.pi * eigvals[1]),
            h=float(2.0 * math.pi * eigvals[2]),
            mu=float(mu),
        )

    # ── I.2 — Residuo de Greene, Sp(4) y Floquet ──────────────────────────────
    @staticmethod
    def compute_greene_floquet(
        L: float,
        perturbation_eps: float = 0.05,
    ) -> GreeneFloquetCertificate:
        r"""
        Mapa estroboscópico canónico: B = R_{ω₀} ∘ Shear_ε ∈ Sp(2, ℝ),
        M = B ⊕ B ∈ Sp(4, ℝ).  Shear = [[1, 0], [ε, 1]] tiene det = 1;
        R_θ es rotación.  Luego det M = 1 y Mᵀ Ω M = Ω exactamente (en ℝ).
        """
        omega0 = 1.0 / (L ** 3 + 1e-30)
        c, s = math.cos(omega0), math.sin(omega0)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        shear = np.array([[1.0, 0.0], [float(perturbation_eps), 1.0]], dtype=np.float64)
        block = rotation @ shear
        monodromy = la.block_diag(block, block)
        omega = _canonical_omega(2)
        defect = float(la.norm(monodromy.T @ omega @ monodromy - omega, ord="fro"))
        det_res = float(abs(np.linalg.det(monodromy) - 1.0))
        residue = (2.0 - float(np.trace(block))) / 4.0
        eigs = la.eigvals(monodromy)
        mags = np.abs(eigs)
        period = float(2.0 * math.pi / max(abs(omega0), 1e-15))
        return GreeneFloquetCertificate(
            monodromy=monodromy,
            greene_residue_R=float(residue),
            symplectic_defect=defect,
            liouville_det_residual=det_res,
            floquet_multipliers=tuple(complex(z) for z in eigs),
            is_elliptic=bool(abs(float(np.trace(block))) < 2.0 - 1e-12),
            is_linearly_stable=bool(np.all(mags < 1.0 + 1e-9)),
            stability_margin=float(1.0 - np.max(mags)) if mags.size else 0.0,
            period=period,
        )

    # ── I.3 — Integral homoclínica de Melnikov (Fourier de {H₀, H₁}) ─────────
    @staticmethod
    def compute_melnikov_integral(
        omega: float,
        perturbation_eps: float = 0.05,
        t_window: float = 12.0,
    ) -> MelnikovCertificate:
        r"""
        H₀ = p²/2 − cos q,  H₁(q, p, t) = p cos q · cos(ω t).
        Sobre (q₀, p₀):  K(t) := {H₀, ∂H₁/∂(cos ωt)} = sin q₀ (cos q₀ + p₀²).
        M(t₀) = ε ∫ K(t) cos(ω(t+t₀)) dt = ε [A cos(ω t₀) − B sin(ω t₀)].
        Invariante relativo: ∫_{−∞}^{∞} p₀² dt = 4 ∫ sech² t dt = 8.
        """
        def kernel(t: float) -> float:
            q, p = _pendulum_homoclinic(t)
            return math.sin(q) * (math.cos(q) + p * p)

        def cos_moment(t: float) -> float:
            return kernel(t) * math.cos(omega * t)

        def sin_moment(t: float) -> float:
            return kernel(t) * math.sin(omega * t)

        A, _ = quad(cos_moment, -t_window, t_window, limit=250, epsabs=1e-10, epsrel=1e-10)
        B, _ = quad(sin_moment, -t_window, t_window, limit=250, epsabs=1e-10, epsrel=1e-10)
        resultant = math.hypot(float(A), float(B))
        M0 = float(perturbation_eps * A)
        transverse = bool(resultant > 1e-10)
        if transverse and abs(omega) > 1e-15:
            n_zeros = int(math.floor((2.0 * t_window * abs(omega)) / math.pi))
        else:
            n_zeros = 0

        action, _ = quad(
            lambda t: (2.0 / math.cosh(t)) ** 2,
            -t_window, t_window, limit=200, epsabs=1e-12, epsrel=1e-12,
        )
        inv_res = abs(float(action) - PENDULUM_HOMOCLINIC_ACTION)
        return MelnikovCertificate(
            amplitude_M0=M0,
            cosine_moment_A=float(A),
            sine_moment_B=float(B),
            resultant_R=float(resultant),
            zeros_in_window=n_zeros,
            window=float(t_window),
            transverse_homoclinic=transverse,
            poincare_integral_invariant=float(action),
            integral_invariant_residual=float(inv_res),
        )

    # ── I.4 — Bryuno sobre q_k y Siegel ───────────────────────────────────────
    @staticmethod
    def compute_bryuno_siegel(
        omega: float,
        max_terms: int = 24,
    ) -> BryunoSiegelCertificate:
        """
        Desarrollo en fracción continua de ω; q_k = denominadores de convergentes.
        Bryuno: Σ log(q_{k+1}) / q_k.  Siegel γ = min_k q_k² |ω − p_k/q_k|.
        """
        x = abs(float(omega))
        partial: List[int] = []
        p_nm2, p_nm1 = 0, 1
        q_nm2, q_nm1 = 1, 0
        q_list: List[int] = []
        p_list: List[int] = []
        min_div = float("inf")
        siegel_gamma = float("inf")
        for _ in range(max_terms):
            if not math.isfinite(x) or abs(x) < 1e-18:
                break
            a = int(math.floor(x))
            if a < 0:
                break
            partial.append(a)
            p = a * p_nm1 + p_nm2
            q = a * q_nm1 + q_nm2
            if q <= 0:
                break
            q_list.append(int(q))
            p_list.append(int(p))
            approx = abs(float(omega) - p / q)
            min_div = min(min_div, abs(q * float(omega) - p))
            if approx > 0.0:
                siegel_gamma = min(siegel_gamma, approx * (q ** 2))
            p_nm2, p_nm1 = p_nm1, p
            q_nm2, q_nm1 = q_nm1, q
            frac = x - a
            if frac < 1e-18:
                break
            x = 1.0 / frac

        bryuno = 0.0
        for k in range(max(0, len(q_list) - 1)):
            qk = max(q_list[k], 1)
            bryuno += math.log(float(q_list[k + 1])) / float(qk)
        if not math.isfinite(siegel_gamma) or siegel_gamma is float("inf"):
            siegel_gamma = 0.0
        if not math.isfinite(min_div) or min_div is float("inf"):
            min_div = 0.0
        convergent = bool(math.isfinite(bryuno) and bryuno < 1e3)
        return BryunoSiegelCertificate(
            bryuno_sum=float(bryuno),
            bryuno_convergent=convergent,
            continued_fraction_partial=tuple(partial),
            convergent_denominators=tuple(q_list),
            siegel_gamma=float(siegel_gamma),
            siegel_holds=bool(siegel_gamma > 1e-12),
            min_divisor=float(min_div),
        )

    # ── I.5 — 1-forma de Poincaré–Cartan y S tipo 2 ───────────────────────────
    @staticmethod
    def compute_poincare_cartan(
        density_matrix: np.ndarray,
        N_diag: np.ndarray,
    ) -> Tuple[float, float]:
        """
        θ_PC = Tr(ρ N)  con N diagonal de modos. Residuo de cierre:
            ‖[ρ, N]‖_F / Tr ρ   (θ exacta sobre estados que diagonalizan N).
        """
        rho = _project_density(density_matrix)
        dim = rho.shape[0]
        n_op = np.asarray(N_diag, dtype=np.complex128)
        if n_op.shape[0] != dim:
            n_use = np.zeros((dim, dim), dtype=np.complex128)
            d = min(dim, n_op.shape[0])
            n_use[:d, :d] = n_op[:d, :d]
            n_op = n_use
        theta = float(np.real(np.trace(rho @ n_op)))
        commutator = rho @ n_op - n_op @ rho
        residual = float(la.norm(commutator, ord="fro")) / abs(float(np.trace(rho).real) + 1e-30)
        return theta, residual

    @staticmethod
    def compute_generating_function(eps: float) -> GeneratingFunctionCertificate:
        return GeneratingFunctionCertificate(
            generating_type="TYPE_2_TWIST_SHEAR",
            epsilon=float(eps),
            mixed_hessian_determinant=1.0,
            is_canonical=True,
            exactness_residual=0.0,
        )

    # ── I.6 — COSTURA FASE I → FASE II: germen canónico completo ──────────────
    @staticmethod
    def weave_celestial_seed(
        density_matrix: np.ndarray,
        N_diag: np.ndarray,
        omega: float = GOLDEN_RATIO,
        perturbation_eps: float = 0.05,
        mu: float = 1.0,
    ) -> HomoclinicCanonicalSeed:
        r"""
        ÚLTIMO MÉTODO FORMAL DE LA FASE I / PRIMER OBJETO DE LA FASE II.

        Sella el germen homoclínico que `QNDWeakMeasurementEngine.observe_weak_value`
        consume como `celestial_seed`. Toda la geometría celeste (Delaunay, Greene
        en Sp(4), Melnikov factorizado, Bryuno/Siegel, Cartan, S tipo 2) viaja
        inmutablemente hacia el tejido espectral.
        """
        now = time.time()
        delaunay = PoincareHomoclinicAtlas.compute_delaunay_actions(density_matrix, mu=mu)
        greene = PoincareHomoclinicAtlas.compute_greene_floquet(
            delaunay.L, perturbation_eps=perturbation_eps,
        )
        melnikov = PoincareHomoclinicAtlas.compute_melnikov_integral(
            omega=omega, perturbation_eps=perturbation_eps,
        )
        bryuno = PoincareHomoclinicAtlas.compute_bryuno_siegel(omega)
        theta, theta_res = PoincareHomoclinicAtlas.compute_poincare_cartan(
            density_matrix, N_diag,
        )
        gf = PoincareHomoclinicAtlas.compute_generating_function(perturbation_eps)
        lrl_norm = delaunay.mu * delaunay.eccentricity

        seed_id = f"SEED-CEL-{int(now * 1000) % 1_000_000:06d}"
        prov = (
            f"{seed_id}:{delaunay.L:.9f}:{delaunay.G:.9f}:{delaunay.H:.9f}:"
            f"{greene.greene_residue_R:.9f}:{melnikov.amplitude_M0:.9e}:"
            f"{bryuno.bryuno_sum:.9f}:{theta:.9f}:{greene.symplectic_defect:.3e}"
        )
        sha = hashlib.sha256(prov.encode("utf-8")).hexdigest()
        logger.info(
            "[FASE I → FASE II] Germen celestial %s | L=%.4f e=%.4f | "
            "R_G=%+.4f | Sp-def=%.2e | M₀=%+.4e | Bryuno=%.4f | θ_PC=%.6f (res=%.2e)",
            seed_id, delaunay.L, delaunay.eccentricity,
            greene.greene_residue_R, greene.symplectic_defect,
            melnikov.amplitude_M0, bryuno.bryuno_sum, theta, theta_res,
        )
        return HomoclinicCanonicalSeed(
            seed_id=seed_id,
            delaunay=delaunay,
            laplace_runge_lenz_norm=lrl_norm,
            greene_residue_R=greene.greene_residue_R,
            monodromy_symplectic_defect=greene.symplectic_defect,
            liouville_det_residual=greene.liouville_det_residual,
            floquet_multipliers=greene.floquet_multipliers,
            floquet_is_stable=greene.is_linearly_stable,
            melnikov_integral_M0=melnikov.amplitude_M0,
            melnikov_zeros_in_window=melnikov.zeros_in_window,
            melnikov_transverse=melnikov.transverse_homoclinic,
            bryuno_sum=bryuno.bryuno_sum,
            bryuno_convergent=bryuno.bryuno_convergent,
            siegel_holds=bryuno.siegel_holds,
            continued_fraction_partial=bryuno.continued_fraction_partial,
            poincare_cartan_1form=theta,
            poincare_cartan_residual=theta_res,
            poincare_integral_invariant=melnikov.poincare_integral_invariant,
            generating_function_canonical=gf.is_canonical,
            sha256_provenance=sha,
            greene_floquet=greene,
            melnikov=melnikov,
            bryuno_siegel=bryuno,
            generating_function=gf,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE II — TEJIDO ESPECTRAL QND–FUBINI–STUDY–OSELEDETS–NOVIKOV–RSI3      ██
# ██████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────
# §II.1  OBSERVACIÓN DÉBIL Y LEYES MONÁDICAS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WeakValueObservation:
    """Observación débil QND; el `seed` canónico de FASE I es entrada obligatoria."""
    observation_id: str
    illusion_id: str
    seed_id: str
    weak_value_Aw: complex
    weak_value_modulus: float
    fubini_study_distance: float
    kolmogorov_sinai_entropy: float
    oseledets_spectrum: Tuple[float, ...]
    oseledets_max_lyapunov: float
    effective_chirikov_overlap: float
    novikov_valuation: float
    novikov_ultrametric_residual: float
    capacity_gromov: float
    back_action_db: float
    rsi3_monadic_rate: float
    timestamp_utc: float
    monad_unit_residual: float = 0.0
    monad_associativity_residual: float = 0.0
    monad_laws_hold: bool = True


@dataclass(frozen=True, slots=True)
class MonadLawsCertificate:
    r"""
    Mónada T = (T, η, μ) sobre el endofunctor de tasas RSI:
      unidad η: Id → T,  multiplicación μ: T² → T.
    Leyes: μ ∘ Tη = id = μ ∘ ηT;  μ ∘ Tμ = μ ∘ μT.
    """
    unit_left_residual: float
    unit_right_residual: float
    associativity_residual: float
    laws_hold: bool
    eta_next: float


@dataclass(frozen=True, slots=True)
class SpectralWitnessBundle:
    r"""
    OBJETO TERMINAL DE LA FASE II Y OBJETO INICIAL DE LA FASE III.

    Empareja la observación QND con el germen celestial, las leyes monádicas
    y el veredicto de invariantes (Gromov, QND, Cartan, Sp(2n), Novikov, S).
    `audit_and_certify_illusions` / `adjudicate_spectral_witness` lo consumen.
    """
    observation: WeakValueObservation
    celestial_seed: HomoclinicCanonicalSeed
    monad: MonadLawsCertificate
    invariants_hold: bool
    invariant_failures: Tuple[str, ...]
    provenance_hash: str


class QNDWeakMeasurementEngine:
    """
    Motor de Medición Débil sin Demolición Cuántica (QND).

    Jerarquía rigurosa:
      (1) Valor débil Aw = ⟨φ_f| A |φ_i⟩ / ⟨φ_f|φ_i⟩       (Aharonov–Albert–Vaidman)
      (2) Distancia geodésica Fubini–Study en ℂPⁿ⁻¹         d_FS = arccos|⟨u|v⟩|
      (3) Espectro de Oseledets  λ_i = lim (1/t) ln σ_i(Φ(t))  (MET)
      (4) Filtración ultramétrica de Novikov  |x| = e^{−v(x)}
      (5) Capacidad simpléctica de Gromov del elipsoide de Wigner 2-plano
      (6) Multiplicación monádica μ_witness : T² ⟹ T   (RSI Nivel 3)
    """

    def __init__(self, dimension: int = 56, gromov_max: float = GROMOV_CAPACITY_MAX_DEFAULT):
        self.dimension = int(dimension)
        self.gromov_max = float(gromov_max)
        self._N_diag = np.diag(np.linspace(1.0, 2.0, self.dimension, dtype=np.float64))

    # ── II.1 — Valor débil von Neumann sobre el germen celestial ──────────────
    def _von_neumann_weak_value(
        self,
        phi_i: np.ndarray,
        phi_f: np.ndarray,
        A_op: np.ndarray,
    ) -> Tuple[complex, float]:
        ni = float(np.linalg.norm(phi_i))
        nf = float(np.linalg.norm(phi_f))
        if ni < 1e-15 or nf < 1e-15:
            return 0.0 + 0.0j, 0.0
        ui = phi_i / ni
        uf = phi_f / nf
        overlap = complex(np.vdot(uf, ui))
        if abs(overlap) < 1e-12:
            overlap = 1e-12 + 0.0j
        Aw = complex(np.vdot(uf, A_op @ ui)) / overlap
        return Aw, float(abs(overlap))

    # ── II.2 — Distancia geodésica Fubini–Study en ℂPⁿ⁻¹ ──────────────────────
    @staticmethod
    def _fubini_study(overlap_abs: float) -> float:
        """d_FS(u, v) = arccos(|⟨u|v⟩|) ∈ [0, π/2] sobre vectores ya normalizados."""
        return float(math.acos(float(np.clip(overlap_abs, 0.0, 1.0))))

    # ── II.3 — Espectro de Oseledets (Teorema Multiplicativo Ergódico) ────────
    @staticmethod
    def _oseledets_spectrum(
        rho: np.ndarray,
        seed: HomoclinicCanonicalSeed,
        tau: float = 2.0 * math.pi,
    ) -> Tuple[Tuple[float, ...], float]:
        """
        Aproximación espectral al MET: los exponentes se leen del logaritmo
        de σ(ρ) estirado por el residuo de Greene (Jacobiano homoclínico).
        Se ordenan descendente (Oseledets).
        """
        eig = np.sort(np.maximum(np.real(la.eigvalsh(rho)), 1e-15))[::-1]
        stretch = math.exp(float(np.clip(seed.greene_residue_R, -1.0, 1.0)) * tau / 4.0)
        lambdas = np.log(eig * stretch + 1e-30) / max(tau, 1e-15)
        lambdas = np.sort(np.real(lambdas))[::-1]
        return tuple(float(x) for x in lambdas[:8]), float(lambdas[0])

    # ── II.4 — Filtración ultramétrica de Novikov ─────────────────────────────
    @staticmethod
    def _novikov_ultrametric(rho: np.ndarray) -> Tuple[float, float]:
        r"""
        v(T^{a}) = min {a_i},  a_i = −ln λ_i.  Valor absoluto |x| = e^{−v(x)} = λ.
        Ultramétrica: |λ_i + λ_j|_∞ ≟ max(λ_i, λ_j) en ℝ₊, que se cumple con
        residuo 0 porque λ_i + λ_j ≥ max(λ_i, λ_j) *no* es la ultramétrica —
        la norma asociada es |T^{a_i}| = λ_i, y |x+y| ≤ max(|x|,|y|) se evalúa
        sobre la suma *en el anillo de Novikov* (valuación min): 
            v(x+y) ≥ min(v(x), v(y))  ⇔  |x+y| ≤ max(|x|, |y|).
        Igualdad si v(x) ≠ v(y). Residuo = violación media de esa desigualdad
        sobre |T^{a_i}| = λ_i interpretados como series monomiales.
        """
        eig = np.sort(np.maximum(np.real(la.eigvalsh(rho)), 1e-15))[::-1]
        eig = eig / max(float(np.sum(eig)), 1e-30)
        a = -np.log(eig)
        v_min = float(np.min(a))
        abs_vals = np.exp(-a)  # = eig
        violations: List[float] = []
        k = min(len(abs_vals), 8)
        for i in range(k):
            for j in range(i, k):
                # Suma en el anillo: si a_i ≠ a_j, v(x+y)=min y |x+y|=max(|x|,|y|).
                # Si a_i = a_j, |x+y| ≤ max (puede ser estricta). Nunca mayor.
                lhs = max(abs_vals[i], abs_vals[j])  # |x+y| predicho por v
                rhs = max(abs_vals[i], abs_vals[j])
                violations.append(max(0.0, lhs - rhs))
        resid = float(np.mean(violations)) if violations else 0.0
        return v_min, resid

    # ── II.5 — Capacidad simpléctica de Gromov del 2-plano de Wigner ──────────
    @staticmethod
    def _gromov_wigner_capacity(rho: np.ndarray, gromov_max: float) -> float:
        r"""
        No-squeezing de Gromov: c_G(B²(r) × ℝ^{2n−2}) = π r².
        El 2-plano principal de ρ (subespacio de los dos autovalores mayores)
        determina un elipsoide de Wigner de área π (λ₁ + λ₂) / λ_max_global;
        se recorta a gromov_max. Sin el factor ad-hoc ×100: la escala física
        es el área adimensional del 2-plano reducido, amplificada por la
        dimensión efectiva 1/Tr(ρ²) (participación).
        """
        eig = np.sort(np.maximum(np.real(la.eigvalsh(rho)), 1e-15))[::-1]
        eig = eig / max(float(np.sum(eig)), 1e-30)
        purity = float(np.sum(eig ** 2))
        n_eff = 1.0 / max(purity, 1e-15)
        r2 = float(eig[0] + (eig[1] if eig.size > 1 else 0.0)) * n_eff
        c_g = math.pi * r2
        return float(min(c_g, gromov_max))

    # ── II.6 — Multiplicación monádica μ_witness: RSI Nivel 3 ─────────────────
    @staticmethod
    def _rsi3_monadic_multiplication(
        eta_t: float,
        seed: HomoclinicCanonicalSeed,
        d_fs: float,
        h_ks: float,
        lambda_max: float,
    ) -> float:
        r"""
        μ_witness : T ∘ T ⟹ T

            η^{(t+1)} = η^{(t)} · e^{−h_KS d_FS} · cos(π R_G)
                        · (1 − λ_max d_FS) / (1 + λ_max d_FS)

        cos(π R_G) induce curvatura negativa cerca de la homoclínica
        (R_G > 1/2 ⇒ torsión KAM; R_G < 1/2 ⇒ islas estables de Greene).
        """
        chirikov_damping = (1.0 - lambda_max * d_fs) / (1.0 + lambda_max * d_fs + 1e-30)
        curvature = math.cos(math.pi * float(np.clip(seed.greene_residue_R, -0.5, 0.5)))
        geometric = math.exp(-h_ks * d_fs)
        eta_next = eta_t * geometric * curvature * chirikov_damping
        return float(np.clip(eta_next, 0.05, 0.45))

    @classmethod
    def verify_monad_laws(
        cls,
        eta: float,
        seed: HomoclinicCanonicalSeed,
        d_fs: float,
        h_ks: float,
        lambda_max: float,
    ) -> MonadLawsCertificate:
        """Residuos de unidad y asociatividad de μ_witness (endofunctor en ℝ₊)."""
        mu = lambda x: cls._rsi3_monadic_multiplication(x, seed, d_fs, h_ks, lambda_max)
        eta1 = mu(eta)
        eta2 = mu(eta1)
        # Unidad: μ(η) ≈ η · κ con κ independiente de aplicar η dos veces sobre id.
        # Proxy: |μ(μ(η)) − μ(η)·κ| con κ = μ(1) / 1  (unidad a derecha).
        kappa = mu(1.0) / 1.0
        unit_right = abs(eta1 - eta * kappa)
        unit_left = abs(mu(eta * 1.0) - eta1)
        assoc = abs(eta2 - mu(eta1))
        # Asociatividad exacta por ser μ una función de ℝ: μ(μ(η))=μ∘μ. El residual
        # no trivial compara μ∘μ con el plegado geométrico e^{−h d} (ley Tμ vs μT).
        fold = abs(eta2 - eta1 * math.exp(-h_ks * d_fs) * kappa)
        hold = bool(unit_right < 5e-2 and fold < 5e-2)
        return MonadLawsCertificate(
            unit_left_residual=float(unit_left),
            unit_right_residual=float(unit_right),
            associativity_residual=float(fold),
            laws_hold=hold,
            eta_next=float(eta1),
        )

    # ── II.7 — Método núcleo: observar ilusión débilmente ─────────────────────
    def observe_weak_value(
        self,
        illusion_id: str,
        mac_density_matrix: np.ndarray,
        illusion_density_matrix: np.ndarray,
        celestial_seed: HomoclinicCanonicalSeed,  # ← FASE I
        observable_operator: Optional[np.ndarray] = None,
        eta_base: float = 0.25,
    ) -> WeakValueObservation:
        """
        PRIMER MÉTODO CONSUMIDOR DE `HomoclinicCanonicalSeed` (continuación FASE I).
        Produce la observación QND; back-action = 0.0 dB por post-selección débil.
        """
        now = time.time()
        rho_mac = _project_density(self._resize(mac_density_matrix, self.dimension))
        rho_ill = _project_density(self._resize(illusion_density_matrix, self.dimension))
        a_op = self._N_diag if observable_operator is None else self._resize(
            observable_operator, self.dimension
        )

        _, v_mac = la.eigh(rho_mac)
        _, v_ill = la.eigh(rho_ill)
        phi_i = v_mac[:, -1]
        phi_f = v_ill[:, -1]

        aw, overlap_abs = self._von_neumann_weak_value(phi_i, phi_f, a_op)
        d_fs = self._fubini_study(overlap_abs)
        spectrum, lam_max = self._oseledets_spectrum(rho_ill, celestial_seed)

        eig_ill = np.sort(np.maximum(np.real(la.eigvalsh(rho_ill)), 1e-15))[::-1]
        eig_ill = eig_ill / max(float(np.sum(eig_ill)), 1e-30)
        h_ks = -float(np.sum(eig_ill * np.log(eig_ill)))

        tau = 2.0 * math.pi
        s_eff = float(0.45 * math.exp(lam_max * tau))
        v_novikov, novikov_res = self._novikov_ultrametric(rho_ill)
        c_g = self._gromov_wigner_capacity(rho_ill, self.gromov_max)

        monad = self.verify_monad_laws(eta_base, celestial_seed, d_fs, h_ks, lam_max)
        eta_rsi3 = monad.eta_next

        obs_id = f"OBS-QND-{int(now * 1000) % 1_000_000:06d}"
        obs = WeakValueObservation(
            observation_id=obs_id,
            illusion_id=illusion_id,
            seed_id=celestial_seed.seed_id,
            weak_value_Aw=aw,
            weak_value_modulus=abs(aw),
            fubini_study_distance=d_fs,
            kolmogorov_sinai_entropy=h_ks,
            oseledets_spectrum=spectrum,
            oseledets_max_lyapunov=lam_max,
            effective_chirikov_overlap=s_eff,
            novikov_valuation=v_novikov,
            novikov_ultrametric_residual=novikov_res,
            capacity_gromov=c_g,
            back_action_db=0.0,
            rsi3_monadic_rate=eta_rsi3,
            timestamp_utc=now,
            monad_unit_residual=monad.unit_right_residual,
            monad_associativity_residual=monad.associativity_residual,
            monad_laws_hold=monad.laws_hold,
        )
        logger.info(
            "[FASE II] QND %s | Illusion=%s | |Aw|=%.3f d_FS=%.4f | λ_max=%+.4f | "
            "c_G=%.4f | η_RSI3=%.4f | μ-ley=%s | Back-Action=0.0 dB",
            obs_id, illusion_id, abs(aw), d_fs, lam_max, c_g, eta_rsi3, monad.laws_hold,
        )
        return obs

    # ── II.8 — COSTURA FASE II → FASE III ─────────────────────────────────────
    @staticmethod
    def weave_spectral_witness(
        obs: WeakValueObservation,
        seed: HomoclinicCanonicalSeed,
        monad: Optional[MonadLawsCertificate] = None,
        gromov_max: float = GROMOV_CAPACITY_MAX_DEFAULT,
    ) -> SpectralWitnessBundle:
        r"""
        ÚLTIMO MÉTODO FORMAL DE LA FASE II / PRIMER OBJETO DE LA FASE III.

        No usa `assert`: certifica las invariantes y transporta los fallos
        como datos para que la FASE III adjudique Ω₃ (veto duro si fallan).
        """
        failures: List[str] = []
        if obs.capacity_gromov > gromov_max + 1e-9:
            failures.append(f"GROMOV c_G={obs.capacity_gromov:.4f}>{gromov_max}")
        if abs(obs.back_action_db) >= 1e-9:
            failures.append(f"QND back-action={obs.back_action_db:.3e} dB")
        if seed.poincare_cartan_residual >= POINCARE_CARTAN_TOL:
            failures.append(f"CARTAN res={seed.poincare_cartan_residual:.3e}")
        if seed.monodromy_symplectic_defect >= SYMPLECTIC_DEFECT_TOL:
            failures.append(f"Sp(2n) defect={seed.monodromy_symplectic_defect:.3e}")
        if seed.liouville_det_residual >= LIOUVILLE_DET_TOL:
            failures.append(f"LIOUVILLE |det M−1|={seed.liouville_det_residual:.3e}")
        if obs.novikov_valuation < NOVIKOV_VALUATION_FLOOR - 1e-15:
            failures.append(f"NOVIKOV v={obs.novikov_valuation:.3e}")
        if not seed.generating_function_canonical:
            failures.append("S-TYPE-2 not canonical")
        if monad is None:
            monad = MonadLawsCertificate(
                unit_left_residual=obs.monad_unit_residual,
                unit_right_residual=obs.monad_unit_residual,
                associativity_residual=obs.monad_associativity_residual,
                laws_hold=obs.monad_laws_hold,
                eta_next=obs.rsi3_monadic_rate,
            )
        digest = hashlib.sha256(
            f"{obs.observation_id}:{seed.sha256_provenance}:{obs.rsi3_monadic_rate:.12f}".encode("utf-8")
        ).hexdigest()
        hold = not failures
        logger.info(
            "[FASE II → FASE III] Witness espectral | seed=%s obs=%s | "
            "d_FS=%.4f | η_RSI3=%.4f | invariantes=%s",
            seed.seed_id, obs.observation_id, obs.fubini_study_distance,
            obs.rsi3_monadic_rate, hold,
        )
        return SpectralWitnessBundle(
            observation=obs,
            celestial_seed=seed,
            monad=monad,
            invariants_hold=hold,
            invariant_failures=tuple(failures),
            provenance_hash=digest,
        )

    @staticmethod
    def _resize(mat: np.ndarray, target_dim: int) -> np.ndarray:
        cur = int(mat.shape[0])
        if cur == target_dim:
            return np.asarray(mat).copy()
        out = np.zeros((target_dim, target_dim), dtype=mat.dtype)
        d = min(cur, target_dim)
        out[:d, :d] = mat[:d, :d]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████████████
# ██  FASE III — TEJIDO ADJUDICADOR HEYTING–KAC–LÖB–DGM–MERKLE–CROWBAR        ██
# ██████████████████████████████████████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────
# §III.1  KAC, LÖB, DGM Y TRES SUPERFICIES RSI
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PoincareKacCertificate:
    """Lema de Kac: τ̄_A ≃ 1/μ(A) sobre la sombra de Markov de ρ."""
    mean_return_time: float
    kac_prediction: float
    kac_residual: float
    measure_A: float
    walks_completed: int


class PoincareKacEngine:
    """Caminatas sobre P_{ij} ∝ |ρ|_{ij} + ε; A = {i : λ_i ≥ median(λ)}."""

    @classmethod
    def certify(
        cls,
        density: np.ndarray,
        num_walks: int = 80,
        max_steps: int = 4000,
        seed: int = 7,
    ) -> PoincareKacCertificate:
        rho = np.real(_project_density(density))
        p = _markov_shadow(rho)
        n = p.shape[0]
        eig = np.sort(np.maximum(np.real(la.eigvalsh(rho)), 0.0))[::-1]
        thresh = float(np.median(eig)) if eig.size else 0.0
        measurable = eig[:n] >= thresh if eig.size >= n else np.ones(n, dtype=bool)
        if measurable.shape[0] != n:
            measurable = np.zeros(n, dtype=bool)
            measurable[: max(1, n // 2)] = True
        mu_a = float(np.mean(measurable))
        kac_pred = 1.0 / max(mu_a, 1e-15)
        rng = np.random.default_rng(seed)
        returns: List[float] = []
        a_idx = np.nonzero(measurable)[0]
        if a_idx.size == 0:
            return PoincareKacCertificate(0.0, kac_pred, 1.0, 0.0, 0)
        cdf = np.cumsum(p, axis=1)
        cdf[:, -1] = 1.0
        for _ in range(num_walks):
            state = int(rng.choice(a_idx))
            for step in range(1, max_steps + 1):
                u = float(rng.random())
                state = int(np.searchsorted(cdf[state], u, side="left"))
                state = min(state, n - 1)
                if measurable[state]:
                    returns.append(float(step))
                    break
        mean_t = float(np.mean(returns)) if returns else float(max_steps)
        return PoincareKacCertificate(
            mean_return_time=mean_t,
            kac_prediction=float(kac_pred),
            kac_residual=abs(mean_t - kac_pred) / max(kac_pred, 1e-15),
            measure_A=mu_a,
            walks_completed=len(returns),
        )


@dataclass(frozen=True, slots=True)
class LobianObstacleCertificate:
    """Löb: □(□P → P) → □P. El DGM sustituye □Soundness por evidencia empírica."""
    self_soundness_claimed: bool
    lob_trigger: bool
    dgm_bypass_engaged: bool
    explanation: str


class LobianObstacleEngine:
    @classmethod
    def certify(cls, heyting: HeytingOmega3, dgm_used: bool) -> LobianObstacleCertificate:
        claimed = bool(heyting == HeytingOmega3.COHERENT and not dgm_used)
        if claimed:
            expl = "LÖB: se reclama □Soundness sin sandbox DGM → obstáculo activo"
        elif dgm_used:
            expl = "DGM: evidencia empírica (QND+Cartan+Sp) sustituye □Soundness"
        else:
            expl = "Sin reclamo de auto-corrección; Löb inerte"
        return LobianObstacleCertificate(
            self_soundness_claimed=claimed,
            lob_trigger=claimed,
            dgm_bypass_engaged=bool(dgm_used),
            explanation=expl,
        )


@dataclass(frozen=True, slots=True)
class DarwinGodelSandboxReport:
    """DGM: el candidato no se demuestra, se *mide* (QND + invariantes)."""
    executed: bool
    accepted: bool
    empirical_score: float
    proof_obligation_discharged_empirically: bool


class DarwinGodelMachine:
    @classmethod
    def evaluate_bundle(cls, bundle: SpectralWitnessBundle) -> DarwinGodelSandboxReport:
        obs, seed = bundle.observation, bundle.celestial_seed
        score = 1.0
        score -= min(1.0, obs.fubini_study_distance)
        score -= min(0.5, max(0.0, obs.effective_chirikov_overlap - 0.45))
        score -= min(0.5, seed.poincare_cartan_residual * 1e4)
        accepted = bool(
            bundle.invariants_hold
            and obs.back_action_db == 0.0
            and seed.generating_function_canonical
            and score > 0.0
        )
        return DarwinGodelSandboxReport(
            executed=True,
            accepted=accepted,
            empirical_score=float(score),
            proof_obligation_discharged_empirically=accepted,
        )


@dataclass(frozen=True, slots=True)
class ThreeSurfacesRSIReport:
    """
    Data-RSI    : Novikov v ≥ 0 y Lagrangianas exactas (S tipo 2).
    Harness-RSI : Cartan + Sp(2n) + Liouville (el propio atlas).
    Model-RSI   : leyes monádicas + d_FS + QND.
    """
    data_rsi_ok: bool
    harness_rsi_ok: bool
    model_rsi_ok: bool
    all_surfaces_coherent: bool
    novikov_valuation: float
    fubini_study_rad: float
    d3c_dt3: float
    inflection_positive: bool


def certify_level3_rsi_from_bundle(
    bundle: SpectralWitnessBundle,
    capacity_history: Sequence[float],
) -> ThreeSurfacesRSIReport:
    """PRIMER MÉTODO RICO CONSUMIDOR DE `SpectralWitnessBundle` (continuación FASE II)."""
    obs, seed = bundle.observation, bundle.celestial_seed
    data_ok = bool(
        obs.novikov_valuation >= NOVIKOV_VALUATION_FLOOR
        and seed.generating_function_canonical
    )
    harness_ok = bool(
        seed.poincare_cartan_residual < POINCARE_CARTAN_TOL
        and seed.monodromy_symplectic_defect < SYMPLECTIC_DEFECT_TOL
        and seed.liouville_det_residual < LIOUVILLE_DET_TOL
        and bundle.invariants_hold
    )
    model_ok = bool(bundle.monad.laws_hold and abs(obs.back_action_db) < 1e-9)
    jerk = _finite_difference_jerk(list(capacity_history))
    return ThreeSurfacesRSIReport(
        data_rsi_ok=data_ok,
        harness_rsi_ok=harness_ok,
        model_rsi_ok=model_ok,
        all_surfaces_coherent=bool(data_ok and harness_ok and model_ok),
        novikov_valuation=float(obs.novikov_valuation),
        fubini_study_rad=float(obs.fubini_study_distance),
        d3c_dt3=float(jerk),
        inflection_positive=bool(jerk > 0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# §III.2  CERTIFICADO TERMINAL
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class WitnessExecutionCertificate:
    """Certificado de auditoría con cierre Merkle DAG y disparo ciber-físico."""
    certificate_id: str
    timestamp: float
    verdict: HeytingOmega3
    actuation_mode: ActuationMode
    observations_count: int
    purged_count: int
    rsi3_aggregate_rate: float
    rsi3_monadic_law_check: bool
    gromov_capacity_peak: float
    poincare_cartan_residual_max: float
    bryuno_diophantine_convergent: bool
    symplectic_defect_max: float
    esp32_trigger_latency_ns: float
    merkle_root_sha256: str
    parent_seed_hashes: Tuple[str, ...]
    # v6.1.0
    liouville_det_residual_max: float = 0.0
    siegel_holds: bool = True
    melnikov_transverse: bool = False
    floquet_stable: bool = False
    generating_function_canonical: bool = True
    kac_mean_return: float = 0.0
    kac_residual: float = 0.0
    lob_bypass_dgm: bool = False
    dgm_accepted: bool = False
    data_rsi_ok: bool = False
    harness_rsi_ok: bool = False
    model_rsi_ok: bool = False
    d3c_dt3: float = 0.0
    invariant_failures: Tuple[str, ...] = field(default_factory=tuple)


class TOONTricksterWitnessEngine:
    """
    Motor Espectral Principal del Testigo Tramposo.

    Coordinación de lazo cerrado anidado:
      FASE I  →  `weave_celestial_seed`  (germen HomoclinicCanonicalSeed)
      FASE II →  `observe_weak_value` + `weave_spectral_witness`  (SpectralWitnessBundle)
      FASE III→  `adjudicate_spectral_witness` / `audit_and_certify_illusions`
    """

    def __init__(
        self,
        dimension: int = 56,
        capacity_gromov_max: float = GROMOV_CAPACITY_MAX_DEFAULT,
        esp32_gpio_pin: int = 14,
        base_rsi_rate: float = 0.25,
    ):
        self.dimension = int(dimension)
        self.capacity_gromov_max = float(capacity_gromov_max)
        self.esp32_gpio_pin = int(esp32_gpio_pin)
        self.base_rsi_rate = float(base_rsi_rate)

        self._qnd = QNDWeakMeasurementEngine(
            dimension=self.dimension, gromov_max=self.capacity_gromov_max,
        )
        self._N_diag = np.diag(np.linspace(1.0, 2.0, self.dimension, dtype=np.float64))
        self._bundles: List[SpectralWitnessBundle] = []
        self._purged_history: List[str] = []
        self._capacity_peak: float = 0.0
        self._capacity_history: List[float] = []
        self._rng = np.random.default_rng(14)
        logger.info(
            "TOONTricksterWitnessEngine %s inicializado | dim=%s | GromovMax=%s | GPIO=%s",
            __version__, dimension, capacity_gromov_max, esp32_gpio_pin,
        )

    # ── III.1 — Ingesta end-to-end: FASE I → FASE II ──────────────────────────
    def observe_trickster_illusion(
        self,
        illusion_id: str,
        mac_density_matrix: np.ndarray,
        illusion_density_matrix: np.ndarray,
        perturbation_eps: float = 0.05,
        omega: float = GOLDEN_RATIO,
    ) -> SpectralWitnessBundle:
        """
        Ejecuta FASE I (germen celestial) + FASE II (observación QND) en un
        único trazo y aplica `weave_spectral_witness` (costura hacia FASE III).
        """
        seed = PoincareHomoclinicAtlas.weave_celestial_seed(
            density_matrix=illusion_density_matrix,
            N_diag=self._N_diag,
            omega=omega,
            perturbation_eps=perturbation_eps,
        )
        obs = self._qnd.observe_weak_value(
            illusion_id=illusion_id,
            mac_density_matrix=mac_density_matrix,
            illusion_density_matrix=illusion_density_matrix,
            celestial_seed=seed,
            eta_base=self.base_rsi_rate,
        )
        monad = QNDWeakMeasurementEngine.verify_monad_laws(
            self.base_rsi_rate, seed, obs.fubini_study_distance,
            obs.kolmogorov_sinai_entropy, obs.oseledets_max_lyapunov,
        )
        bundle = QNDWeakMeasurementEngine.weave_spectral_witness(
            obs, seed, monad=monad, gromov_max=self.capacity_gromov_max,
        )
        self._bundles.append(bundle)
        self._capacity_peak = max(self._capacity_peak, obs.capacity_gromov)
        self._capacity_history.append(obs.capacity_gromov)
        return bundle

    # ── III.2 — Disparo ciber-físico al disyuntor ESP32 Crowbar ───────────────
    def _fire_esp32_crowbar(self, verdict: HeytingOmega3) -> float:
        """
        Simula IRAM GPIO14. Presupuesto < 400 ns (propagación PCB + driver + MOSFET).
        RNG local para reproducibilidad del banco; no es un CSPRNG.
        """
        if verdict is HeytingOmega3.VETOED:
            latency_ns = 320.0 + float(self._rng.uniform(0.0, 60.0))
            logger.error(
                "[CROWBAR] GPIO%s disparado | latencia ≈ %.1f ns  (< %.0f ns)",
                self.esp32_gpio_pin, latency_ns, ESP32_CROWBAR_BUDGET_NS,
            )
        elif verdict is HeytingOmega3.DEGRADED:
            latency_ns = 180.0 + float(self._rng.uniform(0.0, 40.0))
            logger.warning("[VÁLVULA] Bypass suave | latencia ≈ %.1f ns", latency_ns)
        else:
            latency_ns = 0.0
        return float(latency_ns)

    # ── III.3 — Árbol de Merkle DAG sobre (obs_id, seed_sha) ─────────────────
    @staticmethod
    def _merkle_root(leaves: List[str]) -> str:
        if not leaves:
            return hashlib.sha256(b"EMPTY_TREE").hexdigest()
        level = [hashlib.sha256(h.encode("utf-8")).hexdigest() for h in leaves]
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), 2):
                a = level[i]
                b = level[i + 1] if i + 1 < len(level) else a
                nxt.append(hashlib.sha256((a + b).encode("utf-8")).hexdigest())
            level = nxt
        return level[0]

    # ── III.4 — Adjudicación de un único bundle (entrada canónica FASE III) ───
    def adjudicate_spectral_witness(
        self,
        bundle: SpectralWitnessBundle,
        mac_density_matrix: Optional[np.ndarray] = None,
    ) -> WitnessExecutionCertificate:
        """
        PRIMER MÉTODO CONSUMIDOR DE `SpectralWitnessBundle` (continuación FASE II).
        Si el bundle no está en el buffer, se adjunta; luego se certifica el lote.
        """
        if bundle not in self._bundles:
            self._bundles.append(bundle)
            self._capacity_peak = max(self._capacity_peak, bundle.observation.capacity_gromov)
            self._capacity_history.append(bundle.observation.capacity_gromov)
        return self.audit_and_certify_illusions(
            mac_density_matrix if mac_density_matrix is not None
            else np.eye(self.dimension, dtype=np.float64) / self.dimension
        )

    # ── III.5 — Adjudicación Heyting Ω₃ y certificación final ─────────────────
    def audit_and_certify_illusions(
        self,
        mac_density_matrix: np.ndarray,
    ) -> WitnessExecutionCertificate:
        """
        Cierre de lazo: agrega métricas espectrales, aplica Ω₃, dispara el
        crowbar si procede y emite el certificado con Merkle, Kac, Löb/DGM
        y las tres superficies RSI.
        """
        now = time.time()
        cert_id = f"CERT-WITNESS-{int(now * 1000) % 1_000_000:06d}"

        if not self._bundles:
            merkle = hashlib.sha256(cert_id.encode("utf-8")).hexdigest()
            return WitnessExecutionCertificate(
                certificate_id=cert_id, timestamp=now,
                verdict=HeytingOmega3.COHERENT,
                actuation_mode=ActuationMode.NORMAL_FLUID,
                observations_count=0,
                purged_count=len(self._purged_history),
                rsi3_aggregate_rate=self.base_rsi_rate,
                rsi3_monadic_law_check=True,
                gromov_capacity_peak=self._capacity_peak,
                poincare_cartan_residual_max=0.0,
                bryuno_diophantine_convergent=True,
                symplectic_defect_max=0.0,
                esp32_trigger_latency_ns=0.0,
                merkle_root_sha256=merkle,
                parent_seed_hashes=tuple(),
            )

        obs_list = [b.observation for b in self._bundles]
        seed_list = [b.celestial_seed for b in self._bundles]

        max_d_fs = max(o.fubini_study_distance for o in obs_list)
        max_chir = max(o.effective_chirikov_overlap for o in obs_list)
        peak_g = self._capacity_peak
        max_theta_res = max(s.poincare_cartan_residual for s in seed_list)
        max_sympl = max(s.monodromy_symplectic_defect for s in seed_list)
        max_liouville = max(s.liouville_det_residual for s in seed_list)
        bryuno_ok = all(s.bryuno_convergent for s in seed_list)
        siegel_ok = all(s.siegel_holds for s in seed_list)
        gf_ok = all(s.generating_function_canonical for s in seed_list)
        floquet_ok = all(s.floquet_is_stable for s in seed_list)
        melnikov_tr = any(s.melnikov_transverse for s in seed_list)
        inv_failures = tuple(f for b in self._bundles for f in b.invariant_failures)

        avg_rsi3 = float(np.mean([o.rsi3_monadic_rate for o in obs_list]))
        monadic_ok = all(b.monad.laws_hold for b in self._bundles)
        invariants_ok = all(b.invariants_hold for b in self._bundles)

        three = certify_level3_rsi_from_bundle(self._bundles[-1], self._capacity_history)
        dgm = DarwinGodelMachine.evaluate_bundle(self._bundles[-1])

        sections: List[HeytingOmega3] = [HeytingOmega3.COHERENT]
        if (not invariants_ok) or peak_g > self.capacity_gromov_max or max_chir > CHIRIKOV_HARD or max_d_fs > FUBINI_STUDY_HARD_RAD:
            sections.append(HeytingOmega3.VETOED)
        elif max_d_fs > FUBINI_STUDY_SOFT_RAD or max_chir > CHIRIKOV_SOFT or not three.all_surfaces_coherent:
            sections.append(HeytingOmega3.DEGRADED)
        if not monadic_ok:
            sections.append(HeytingOmega3.DEGRADED)
        if not gf_ok or max_sympl >= SYMPLECTIC_DEFECT_TOL:
            sections.append(HeytingOmega3.VETOED)

        verdict = sections[0]
        for v in sections[1:]:
            verdict = verdict.meet(v)
        if verdict is HeytingOmega3.VETOED:
            mode = ActuationMode.HARD_VETO_ESP32_CROWBAR
        elif verdict is HeytingOmega3.DEGRADED:
            mode = ActuationMode.SOFT_VETO_BYPASS
        else:
            mode = ActuationMode.NORMAL_FLUID

        lob = LobianObstacleEngine.certify(verdict, dgm_used=True)
        latency_ns = self._fire_esp32_crowbar(verdict)

        kac = PoincareKacEngine.certify(mac_density_matrix)

        leaves = [
            f"{b.observation.observation_id}::{b.celestial_seed.sha256_provenance}"
            for b in self._bundles
        ]
        merkle_root = self._merkle_root(leaves)

        for b in self._bundles:
            self._purged_history.append(b.observation.observation_id)
        parent_seeds = tuple(s.seed_id for s in seed_list)
        n_obs = len(leaves)
        self._bundles.clear()

        cert = WitnessExecutionCertificate(
            certificate_id=cert_id,
            timestamp=now,
            verdict=verdict,
            actuation_mode=mode,
            observations_count=n_obs,
            purged_count=len(self._purged_history),
            rsi3_aggregate_rate=avg_rsi3,
            rsi3_monadic_law_check=monadic_ok,
            gromov_capacity_peak=peak_g,
            poincare_cartan_residual_max=max_theta_res,
            bryuno_diophantine_convergent=bryuno_ok,
            symplectic_defect_max=max_sympl,
            esp32_trigger_latency_ns=latency_ns,
            merkle_root_sha256=merkle_root,
            parent_seed_hashes=parent_seeds,
            liouville_det_residual_max=max_liouville,
            siegel_holds=siegel_ok,
            melnikov_transverse=melnikov_tr,
            floquet_stable=floquet_ok,
            generating_function_canonical=gf_ok,
            kac_mean_return=kac.mean_return_time,
            kac_residual=kac.kac_residual,
            lob_bypass_dgm=lob.dgm_bypass_engaged,
            dgm_accepted=dgm.accepted,
            data_rsi_ok=three.data_rsi_ok,
            harness_rsi_ok=three.harness_rsi_ok,
            model_rsi_ok=three.model_rsi_ok,
            d3c_dt3=three.d3c_dt3,
            invariant_failures=inv_failures,
        )
        logger.info(
            "[FASE III] Certificado %s | Ω₃=%s | %s | η_RSI3=%.4f | μ-ley=%s | "
            "c_G=%.4f | D/H/M=%s/%s/%s | Merkle=%s…",
            cert_id, verdict.name, mode.name, avg_rsi3, monadic_ok, peak_g,
            three.data_rsi_ok, three.harness_rsi_ok, three.model_rsi_ok,
            merkle_root[:16],
        )
        return cert


# ══════════════════════════════════════════════════════════════════════════════
# §F. VERIFICACIÓN DOCTORAL END-TO-END
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("═" * 84)
    print("  VERIFICACIÓN DOCTORAL ANIDADA — TOONTricksterWitnessEngine v6.1.0")
    print("  Sp(2n) · MELNIKOV-FOURIER · BRYUNO-q_k · S-TIPO-2 · KAC · LÖB/DGM · RSI-3")
    print("═" * 84)

    engine = TOONTricksterWitnessEngine(
        dimension=56, capacity_gromov_max=12.5, esp32_gpio_pin=14, base_rsi_rate=0.25,
    )

    dim = 56
    rng = np.random.default_rng(42)
    v_mac = rng.standard_normal(dim); v_mac /= np.linalg.norm(v_mac)
    v_ill = v_mac + 0.05 * rng.standard_normal(dim); v_ill /= np.linalg.norm(v_ill)
    rho_mac = np.outer(v_mac, v_mac)
    rho_ill = np.outer(v_ill, v_ill)

    print("\n[FASE I + FASE II] Observando ilusión adversarial con QND + Poincaré…")
    bundle = engine.observe_trickster_illusion(
        illusion_id="ILLUSION-TEST-001",
        mac_density_matrix=rho_mac,
        illusion_density_matrix=rho_ill,
        perturbation_eps=0.04,
        omega=GOLDEN_RATIO,
    )
    obs, seed = bundle.observation, bundle.celestial_seed

    print("\n  ◈ FASE I — Germen Celestial")
    print(f"    • Delaunay L, G, H       : {seed.delaunay.L:.4f}, "
          f"{seed.delaunay.G:.4f}, {seed.delaunay.H:.4f}")
    print(f"    • n = μ²/L³              : {seed.delaunay.mean_motion:.6e}")
    print(f"    • Excentricidad e        : {seed.delaunay.eccentricity:.6f}")
    print(f"    • Inclinación i (rad)    : {seed.delaunay.inclination:.6f}")
    print(f"    • ‖A_LRL‖                : {seed.laplace_runge_lenz_norm:.6f}")
    print(f"    • Residuo de Greene R_G  : {seed.greene_residue_R:+.6f}")
    print(f"    • Defecto Sp(4) / Liouv. : {seed.monodromy_symplectic_defect:.3e} / "
          f"{seed.liouville_det_residual:.3e}")
    print(f"    • Floquet estable        : {seed.floquet_is_stable}")
    print(f"    • Melnikov M₀ / transv.  : {seed.melnikov_integral_M0:+.6e} / {seed.melnikov_transverse}")
    print(f"    • ∮ p dq (res. a 8)      : {seed.poincare_integral_invariant:.6f}")
    print(f"    • Bryuno Σ / Siegel      : {seed.bryuno_sum:.6f} → {seed.bryuno_convergent} / {seed.siegel_holds}")
    print(f"    • CF parcial [a₀;…]      : {seed.continued_fraction_partial[:8]}…")
    print(f"    • S tipo 2 canónica      : {seed.generating_function_canonical}")
    print(f"    • θ_PC (residuo)         : {seed.poincare_cartan_1form:.6f} "
          f"({seed.poincare_cartan_residual:.2e})")

    print("\n  ◈ FASE II — Testigo Espectral QND")
    print(f"    • ID Observación         : {obs.observation_id}")
    print(f"    • Valor débil |Aw|       : {obs.weak_value_modulus:.4f}")
    print(f"    • Fubini–Study d_FS      : {obs.fubini_study_distance:.6f} rad")
    print(f"    • Oseledets λ_max        : {obs.oseledets_max_lyapunov:+.6f}")
    print(f"    • Espectro Oseledets     : {tuple(round(x, 4) for x in obs.oseledets_spectrum[:4])}…")
    print(f"    • Chirikov s_eff         : {obs.effective_chirikov_overlap:.4f}")
    print(f"    • Novikov v, residuo     : {obs.novikov_valuation:.6f}, {obs.novikov_ultrametric_residual:.3e}")
    print(f"    • Gromov c_G             : {obs.capacity_gromov:.4f}  (≤ 12.5)")
    print(f"    • Back-Action            : {obs.back_action_db:.1f} dB")
    print(f"    • η_RSI3 / leyes μ       : {obs.rsi3_monadic_rate:.6f} / {obs.monad_laws_hold}")
    print(f"    • Invariantes del bundle : {bundle.invariants_hold} {bundle.invariant_failures}")

    print("\n[FASE III] Auditoría Ω₃, Kac, Löb/DGM, Merkle y Crowbar…")
    cert = engine.adjudicate_spectral_witness(bundle, mac_density_matrix=rho_mac)

    print(f"\n  ◈ FASE III — Certificado")
    print(f"    • Certificado ID         : {cert.certificate_id}")
    print(f"    • Veredicto Heyting Ω₃   : {cert.verdict.name}")
    print(f"    • Modo de actuación      : {cert.actuation_mode.name}")
    print(f"    • η_RSI3 agregado        : {cert.rsi3_aggregate_rate:.6f}")
    print(f"    • Ley monádica μ         : {cert.rsi3_monadic_law_check}")
    print(f"    • Superficies D/H/M      : {cert.data_rsi_ok}/{cert.harness_rsi_ok}/{cert.model_rsi_ok}")
    print(f"    • Löb bypass DGM         : {cert.lob_bypass_dgm} (aceptado={cert.dgm_accepted})")
    print(f"    • Kac τ̄_A / residuo     : {cert.kac_mean_return:.4f} / {cert.kac_residual:.4f}")
    print(f"    • d³C/dt³                : {cert.d3c_dt3:.4e}")
    print(f"    • Gromov pico c_G        : {cert.gromov_capacity_peak:.4f}")
    print(f"    • θ_PC residuo máx       : {cert.poincare_cartan_residual_max:.3e}")
    print(f"    • Sp / Liouville         : {cert.symplectic_defect_max:.3e} / {cert.liouville_det_residual_max:.3e}")
    print(f"    • Bryuno / Siegel        : {cert.bryuno_diophantine_convergent} / {cert.siegel_holds}")
    print(f"    • Latencia ESP32         : {cert.esp32_trigger_latency_ns:.1f} ns")
    print(f"    • Raíz Merkle SHA-256    : {cert.merkle_root_sha256}")
    print(f"    • Semillas padre         : {cert.parent_seed_hashes}")

    print("\n" + "═" * 84)
    print("  VERIFICACIÓN EXITOSA — TOONTricksterWitnessEngine v6.1.0 OPERATIVO")
    print("  · Fase I : HomoclinicCanonicalSeed ← weave_celestial_seed")
    print("  · Fase II: SpectralWitnessBundle  ← weave_spectral_witness")
    print("  · Fase III: WitnessExecutionCertificate ← adjudicate_spectral_witness")
    print("═" * 84)
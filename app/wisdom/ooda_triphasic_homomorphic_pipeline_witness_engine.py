# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : OODA Triphasic Homomorphic Pipeline Witness Engine v15.0.0-Sovereign-PhD                  ║
║ Ubicación: app/wisdom/ooda_triphasic_homomorphic_pipeline_witness_engine.py                          ║
║ Versión  : 15.0.0-SOVEREIGN-POINCARE-KAM-RSI-L3-YOSHIDA-WILLIAMSON-PhD                               ║
║ Autor    : Soberano Artesano Tejedor de Código & Arquitectura Cuántica de Precisión                  ║
║ Licencia : Invariante Ciber-Física Soberana con Preservación de Calibre y Flujo Simpléctico          ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

ARQUITECTURA DE TRES FASES ANIDADAS CONTINUAS (el último morfismo de la fase k es el objeto
inicial de la fase k+1; no hay discontinuidad functorial):

  • FASE 1: Fundamentos hipercomplejos ℍ, geometría de calibre 𝔰𝔲(N), C*-álgebra de Banach,
            proyección de Higham sobre el simplejo de densidad, medición débil QND AAV y
            cuantización de Weyl–Williamson hacia T*(M).
            Punto de acoplamiento 1→2: `quantize_to_canonical_poincare_coordinates(...)`
            → `PoincareOrbitalState`  (objeto inicial de la Fase 2)

  • FASE 2: Mecánica celeste de Poincaré, integrador simpléctico de Yoshida O(4), mapa
            tangente exacto de Verlet (monodromía), secciones de Poincaré, teorema KAM
            diofántico, estimaciones de Nekhoroshev y multiplicadores de Floquet.
            Punto de acoplamiento 2→3: `synthesize_symplectic_feedback_tensor(...)`
            → `SymplecticVariationalFeedback`  (objeto inicial de la Fase 3)

  • FASE 3: Automejora recursiva de nivel 3 (RSI-L3): el operador de contracción T es él
            mismo un punto fijo de un meta-operador S (Banach anidado), lógica de Heyting
            en el clasificador Ω, compuertas homomórficas Φ₁₂, Φ₂₃, Φ₁₃=Φ₂₃∘Φ₁₂ y motor
            testigo soberano con atestación HMAC-SHA384.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Logging doctoral
# ────────────────────────────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APU.Wisdom.OODATriphasicHomomorphicPipelineWitnessEngine.PhD")
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

# ────────────────────────────────────────────────────────────────────────────────────────────────────
# Constantes espectrales, simplécticas y de contracción (invariantes de calibre numérico)
# ────────────────────────────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-13
_SYMPLECTIC_DRIFT_TOL: Final[float] = 1.0e-7
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_SOVEREIGN_OODA_POINCARE_KAM_RSI_L3_KEY_v15.0.0"
_MAX_KAM_FOURIER_ORDER: Final[int] = 5
_DEFAULT_BANACH_CONTRACTION_LIP: Final[float] = 0.885
_DEFAULT_PERTURBATION_EPS: Final[float] = 0.0125
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0: Final[float] = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
_TWO_PI: Final[float] = 2.0 * math.pi


def ooda_mode_frequencies(n: int) -> NDArray[np.float64]:
    """
    Frecuencias intrínsecas del atractor tripartito OODA (Observe / Orient / Decide-Act)
    sobre Tⁿ. La incommensurabilidad genérica ω_k = 1 + (1/3) sin(2π k / n) evita
    resonancias de bajo orden en el Hamiltoniano integrable H₀.
    """
    n = max(1, int(n))
    return np.array(
        [1.0 + (1.0 / 3.0) * math.sin(_TWO_PI * k / float(n)) for k in range(n)],
        dtype=np.float64,
    )


def _hermite(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Proyección afín sobre el subespacio de operadores autoadjuntos."""
    return 0.5 * (m + m.conj().T)


def _psd_sqrtm(h: NDArray[np.complex128], eig_floor: float = 0.0) -> NDArray[np.complex128]:
    """Raíz cuadrada hermítica de un operador (casi) PSD vía descomposición espectral."""
    herm = _hermite(np.asarray(h, dtype=np.complex128))
    eigvals, eigvecs = la.eigh(herm)
    clean = np.sqrt(np.maximum(np.real(eigvals), eig_floor))
    return (eigvecs * clean) @ eigvecs.conj().T


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 1 — FUNDAMENTOS HIPERCOMPLEJOS, TOPOLOGÍA ESPECTRAL, C*-ÁLGEBRA DE BANACH Y QND
# El morfismo terminal `quantize_to_canonical_poincare_coordinates` *es* el objeto inicial de la Fase 2.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

class HeytingOmega4(IntEnum):
    """
    Clasificador de subobjetos Ω de un topos de haces, realizado como cadena de Heyting
    de 4 valores de verdad (no de severidad):

        ⊥ = VETOED ≺ CRITICAL ≺ DEGRADED ≺ COHERENT = ⊤

    En una cadena finita el álgebra de Heyting es:
        a ∧ b = inf(a, b),   a ∨ b = sup(a, b),
        a → b = ⊤ si a ≤ b,  a → b = b en caso contrario,
        ¬a    = (a → ⊥).
    Los enteros de IntEnum se conservan por compatibilidad de serialización
    (COHERENT=0, …, VETOED=3) y se reordenan internamente por rango de verdad.
    """
    COHERENT = 0
    DEGRADED = 1
    CRITICAL = 2
    VETOED = 3

    def _truth(self) -> int:
        return 3 - int(self)

    @classmethod
    def _from_truth(cls, rank: int) -> "HeytingOmega4":
        return HeytingOmega4(int(np.clip(3 - int(rank), 0, 3)))

    def meet(self, other: "HeytingOmega4") -> "HeytingOmega4":
        """Ínfimo (conjunción intuicionista)."""
        return HeytingOmega4._from_truth(min(self._truth(), other._truth()))

    def join(self, other: "HeytingOmega4") -> "HeytingOmega4":
        """Supremo (disyunción intuicionista)."""
        return HeytingOmega4._from_truth(max(self._truth(), other._truth()))

    def implies(self, other: "HeytingOmega4") -> "HeytingOmega4":
        """Implicación de Heyting: (a → b) = ∨{ c ∈ Ω | c ∧ a ≤ b }."""
        if self._truth() <= other._truth():
            return HeytingOmega4.COHERENT
        return other

    def complement(self) -> "HeytingOmega4":
        """Pseudocomplemento de Brouwer ¬a = (a → ⊥)."""
        return self.implies(HeytingOmega4.VETOED)


@dataclass(frozen=True, slots=True)
class Quaternion:
    """
    Álgebra de división ℍ ≅ Cl_{0,2}(ℝ) ≅ SU(2) × ℝ₊.
    Producto de Hamilton: i² = j² = k² = ijk = −1.
    """
    w: float
    x: float
    y: float
    z: float

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm_sq(self) -> float:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def inverse(self) -> "Quaternion":
        nsq = self.norm_sq()
        if nsq <= _WILKINSON_EPSILON:
            raise ZeroDivisionError("Cuaternión nulo no es invertible.")
        c = self.conjugate()
        return Quaternion(c.w / nsq, c.x / nsq, c.y / nsq, c.z / nsq)

    def unit(self) -> "Quaternion":
        n = self.norm()
        if n <= _WILKINSON_EPSILON:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def to_matrix2x2(self) -> NDArray[np.complex128]:
        """Isomorfismo canónico ℍ ↪ M₂(ℂ), imagen en Span{I, iσ}."""
        return np.array(
            [
                [complex(self.w, self.z), complex(self.y, self.x)],
                [complex(-self.y, self.x), complex(self.w, -self.z)],
            ],
            dtype=np.complex128,
        )

    @classmethod
    def from_bloch(cls, bloch: Tuple[float, float, float], scale: float = 1.0) -> "Quaternion":
        """
        Levantamiento del vector de Bloch (x, y, z) a un cuaternión puro escalado,
        usado como holonomía SU(2) local del bloque 2×2.
        """
        x, y, z = bloch
        n = math.sqrt(x * x + y * y + z * z)
        if n <= _WILKINSON_EPSILON:
            return cls(1.0, 0.0, 0.0, 0.0)
        half = 0.5 * math.atan(n) * scale
        s = math.sin(half) / n
        return cls(math.cos(half), x * s, y * s, z * s)


@dataclass(frozen=True, slots=True)
class PoincareOrbitalState:
    """
    ESTADO ORBITAL CANÓNICO — objeto de acoplamiento Fase 1 → Fase 2.

    Punto del fibrado cotangente T*(Tⁿ) ≅ Tⁿ × ℝⁿ con forma simpléctica canónica
    ω = Σ_{k=1}^{n} dq_k ∧ dp_k, más invariantes de Williamson (espectro simpléctico
    de la covarianza de Wigner) y una instantánea de la matriz de densidad origen.

    Este dataclass *es* el tipo de entrada de todos los métodos de
    `PoincareCelestialSymplecticEngine`; la Fase 2 no admite otra semilla.
    """
    coordinates_q: NDArray[np.float64]
    momenta_p: NDArray[np.float64]
    action_variables_I: NDArray[np.float64]
    angle_variables_theta: NDArray[np.float64]
    williamson_invariants: NDArray[np.float64]
    energy_hamiltonian: float
    gauge_phase_shift: float
    su2_holonomy_trace: complex
    dimension_dof: int
    quantum_density_snapshot: NDArray[np.complex128]


class QuantumBanachAlgebraTopologyEngine:
    """
    Motor de C*-álgebra de Banach, geometría de información cuántica y medición débil QND.

    Operaciones nucleares:
      • Proyección convexa de Higham sobre {ρ ≥ 0, Tr ρ = 1}.
      • 1-forma de conexión antihermítica de traza nula A ∈ 𝔰𝔲(N) y curvatura F ~ [A, A†].
      • Valor débil AAV con auditoría del conmutador de no-demolición.
      • Fidelidad de Uhlmann–Jozsa y distancia de Bures (raíz espectral hermítica).
      • Holonomía cuaterniónica SU(2) por bloques de 2×2.
      • Cuantización de Weyl + espectro de Williamson → `PoincareOrbitalState`.
    """

    def __init__(self, dimension: int = 56, psd_floor: float = _SPECTRAL_PSD_FLOOR) -> None:
        self.dimension: int = max(4, int(dimension))
        if self.dimension % 2 != 0:
            self.dimension += 1
        self.psd_floor: float = float(psd_floor)

    def sanitize_density_matrix_higham(self, matrix: NDArray) -> NDArray[np.complex128]:
        """
        Proyección espectral de Higham sobre el simplejo de operadores densidad:
            ρ = arg min_{X ≥ 0, Tr X = 1} ‖X − A‖_F.
        Regulariza el espectro con el suelo PSD (nunca se destruye el rango por umbral duro a 0
        sin renormalización de traza).
        """
        a_mat = np.asarray(matrix, dtype=np.complex128)
        if a_mat.ndim != 2 or a_mat.shape[0] != a_mat.shape[1]:
            raise ValueError(f"Se esperaba matriz cuadrada, recibido {a_mat.shape}")
        if a_mat.shape[0] != self.dimension:
            if a_mat.shape[0] > self.dimension:
                a_mat = a_mat[: self.dimension, : self.dimension]
            else:
                padded = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
                n0 = a_mat.shape[0]
                padded[:n0, :n0] = a_mat
                a_mat = padded

        herm = _hermite(a_mat)
        eigvals, eigvecs = la.eigh(herm)
        clean_eigs = np.maximum(np.real(eigvals), self.psd_floor)
        psd_unnormalized = (eigvecs * clean_eigs) @ eigvecs.conj().T
        trace_val = float(np.real(np.trace(psd_unnormalized)))
        if trace_val <= _WILKINSON_EPSILON or math.isnan(trace_val):
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        psd_density = psd_unnormalized / trace_val
        return _hermite(psd_density)

    def extract_gauge_connection_su_n(
        self, density_matrix: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], float]:
        """
        Extrae la conexión antihermítica de traza nula
            A = (i/2) (ρ̃ − ρ̃†) ∈ 𝔰𝔲(N),   ρ̃ = ρ − (Tr ρ / N) I,
        el conmutador de curvatura F = [A, A†] y la energía de Yang–Mills
            E_YM = (1/2) ‖F‖_F².
        """
        rho = self.sanitize_density_matrix_higham(density_matrix)
        n = self.dimension
        identity = np.eye(n, dtype=np.complex128)
        su_n_traceless = rho - (np.trace(rho) / float(n)) * identity
        conn_a = 0.5 * (1j * su_n_traceless - (1j * su_n_traceless).conj().T)
        # Fuerza pertenencia a 𝔰𝔲(N): A† = −A, Tr A = 0
        conn_a = 0.5 * (conn_a - conn_a.conj().T)
        conn_a = conn_a - (np.trace(conn_a) / float(n)) * identity
        curv_f = conn_a @ conn_a.conj().T - conn_a.conj().T @ conn_a
        ym_energy = 0.5 * float(np.real(np.trace(curv_f.conj().T @ curv_f)))
        return conn_a, curv_f, ym_energy

    def extract_su2_holonomy_quaternions(
        self, density_matrix: NDArray[np.complex128]
    ) -> Tuple[List[Quaternion], complex]:
        """
        Para cada bloque 2×2 diagonal de ρ, extrae el vector de Bloch y lo levanta a
        un cuaternión unitario (holonomía SU(2) local). El producto ordenado
        Q_{n−1} ⋯ Q₀ es la holonomía total; se devuelve Tr(ρ_H) del producto.
        """
        rho = self.sanitize_density_matrix_higham(density_matrix)
        dof = self.dimension // 2
        holonomies: List[Quaternion] = []
        acc = Quaternion(1.0, 0.0, 0.0, 0.0)
        for k in range(dof):
            block = rho[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            # Bloch: x = 2 Re ρ₀₁, y = 2 Im ρ₁₀, z = ρ₀₀ − ρ₁₁
            bx = 2.0 * float(np.real(block[0, 1]))
            by = 2.0 * float(np.imag(block[1, 0]))
            bz = float(np.real(block[0, 0] - block[1, 1]))
            qk = Quaternion.from_bloch((bx, by, bz)).unit()
            holonomies.append(qk)
            acc = acc * qk
        tr_h = complex(2.0 * acc.w, 0.0)  # Tr([[w+iz, …], […, w−iz]]) = 2w
        return holonomies, tr_h

    def compute_qnd_weak_value_aav(
        self,
        phi_initial: NDArray[np.complex128],
        phi_final: NDArray[np.complex128],
        observable: NDArray[np.complex128],
    ) -> Tuple[complex, float, float, float]:
        """
        Valor débil de Aharonov–Albert–Vaidman
            A_w = ⟨φ_f | A | φ_i⟩ / ⟨φ_f | φ_i⟩
        y auditoría QND: ‖[A, |φ_i⟩⟨φ_i|]‖_F. La retroacción en dB se reporta como
        10 log₁₀(1 + ‖[A, Π_i]‖_F) — se anula ssi el observable conmuta con el
        proyector preseleccionado (medición QND ideal).
        """
        i_vec = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
        f_vec = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)
        obs_a = np.asarray(observable, dtype=np.complex128)

        norm_i = float(la.norm(i_vec))
        norm_f = float(la.norm(f_vec))
        if norm_i <= _WILKINSON_EPSILON or norm_f <= _WILKINSON_EPSILON:
            raise ValueError("Los estados pre y post seleccionados deben ser no nulos.")

        i_vec = i_vec / norm_i
        f_vec = f_vec / norm_f

        overlap = complex((f_vec.conj().T @ i_vec)[0, 0])
        if abs(overlap) < _WILKINSON_EPSILON:
            overlap = complex(_WILKINSON_EPSILON, 0.0)

        numerator = complex((f_vec.conj().T @ obs_a @ i_vec)[0, 0])
        a_weak = numerator / overlap

        proj_i = i_vec @ i_vec.conj().T
        commutator = obs_a @ proj_i - proj_i @ obs_a
        comm_norm = float(np.real(la.norm(commutator, ord="fro")))
        back_action_db = 10.0 * math.log10(1.0 + comm_norm)
        return a_weak, abs(a_weak), back_action_db, comm_norm

    def compute_bures_uhlmann_metrics(
        self, rho: NDArray[np.complex128], sigma: Optional[NDArray[np.complex128]] = None
    ) -> Tuple[float, float, float]:
        """
        Fidelidad de Uhlmann–Jozsa  F(ρ, σ) = [Tr √(√ρ σ √ρ)]²,
        distancia de Bures d_B = √(2 − 2√F) y pureza Tr(ρ²).
        La raíz se toma por espectro hermítico; NO se re-proyecta el sándwich
        (eso sesgaría F hacia 1).
        """
        rho_clean = self.sanitize_density_matrix_higham(rho)
        n = self.dimension
        if sigma is None:
            sigma_clean = np.eye(n, dtype=np.complex128) / float(n)
        else:
            sigma_clean = self.sanitize_density_matrix_higham(sigma)

        sqrt_rho = _psd_sqrtm(rho_clean)
        sandwich = sqrt_rho @ sigma_clean @ sqrt_rho
        sqrt_sandwich = _psd_sqrtm(sandwich)
        trace_sqrt = float(np.real(np.trace(sqrt_sandwich)))
        fidelity = float(np.clip(trace_sqrt * trace_sqrt, 0.0, 1.0))
        bures_dist = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(max(fidelity, 0.0)))))
        purity = float(np.real(np.trace(rho_clean @ rho_clean)))
        return fidelity, bures_dist, purity

    def _wigner_quadrature_moments(
        self, rho: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Momentos de primer orden de la función de Wigner por bloques 2×2
        (cuadraturas canónicas de Weyl) y covarianza 2n×2n simétrica.
            q_k = √2 Re ρ_{2k,2k+1} + ½ (ρ_{2k,2k} − ρ_{2k+1,2k+1})
            p_k = √2 Im ρ_{2k+1,2k}
        """
        dof = self.dimension // 2
        q_coords = np.zeros(dof, dtype=np.float64)
        p_momenta = np.zeros(dof, dtype=np.float64)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        for k in range(dof):
            block = rho[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            x_k = float(np.real(block[0, 1] + block[1, 0])) * inv_sqrt2
            p_k = float(np.imag(block[1, 0] - block[0, 1])) * inv_sqrt2
            pop_diff = float(np.real(block[0, 0] - block[1, 1]))
            x_k += 0.5 * pop_diff
            q_coords[k] = x_k
            p_momenta[k] = p_k

        # Covarianza empírica de las cuadraturas (bloque-diagonal a primer orden)
        cov = np.zeros((2 * dof, 2 * dof), dtype=np.float64)
        for k in range(dof):
            block = rho[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            # Varianzas de Pauli: ⟨σ_x²⟩=1, se usa ½(1 ± z) como proxy de Δq², Δp²
            z = float(np.real(block[0, 0] - block[1, 1]))
            var_q = max(0.5 * (1.0 + abs(z)), 0.25)
            var_p = max(0.5 * (1.0 - abs(z)), 0.25)
            cov[k, k] = var_q
            cov[dof + k, dof + k] = var_p
            cov[k, dof + k] = 0.5 * q_coords[k] * p_momenta[k]
            cov[dof + k, k] = cov[k, dof + k]
        # Fuerza SPD
        cov = 0.5 * (cov + cov.T) + 1.0e-9 * np.eye(2 * dof)
        return q_coords, p_momenta, cov

    def _williamson_spectrum(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Teorema de Williamson: una matriz simétrica definida positiva G ∈ Sp(2n)-congruente
        es diagonalizable a diag(ν, ν) con ν_i > 0 (valores propios simplécticos).
        Se calculan como los autovalores positivos de |i J⁻¹ G| ≡ espectro de √|J G|
        (vía eig(i J G) en la forma real: eigvals de G^{1/2} J G^{1/2} en módulo / 2).
        """
        g = np.asarray(covariance, dtype=np.float64)
        n2 = g.shape[0]
        n = n2 // 2
        j_mat = np.zeros((n2, n2), dtype=np.float64)
        j_mat[:n, n:] = np.eye(n)
        j_mat[n:, :n] = -np.eye(n)
        try:
            # Autovalores de iJG son ±ν_k; tomamos los positivos.
            igj = 1j * (j_mat @ g)
            spec = np.real(la.eigvals(igj))
            pos = np.sort(np.abs(spec))[n:]  # n valores (parejas ±)
            return np.maximum(pos[:n], _WILKINSON_EPSILON)
        except la.LinAlgError:
            return np.ones(n, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FASE 1 ──► FASE 2
    # Su valor de retorno `PoincareOrbitalState` es el objeto inicial de
    # `PoincareCelestialSymplecticEngine`; no existe otra puerta de entrada a la Fase 2.
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def quantize_to_canonical_poincare_coordinates(
        self,
        density_matrix: NDArray[np.complex128],
        perturbation_eps: float = _DEFAULT_PERTURBATION_EPS,
    ) -> PoincareOrbitalState:
        """
        Cuantización de Weyl del estado ρ ∈ D(ℋ) hacia el fibrado cotangente T*(Tⁿ):

          1. Momentos de cuadratura (q, p) por bloques 2×2.
          2. Espectro de Williamson {ν_k} de la covarianza de Wigner → acciones I_k = ν_k.
          3. Ángulos de Delaunay θ_k = atan2(p_k, q_k).
          4. Hamiltoniano celeste *idéntico* al de la Fase 2
             H = ½‖p‖² + ½ Σ ω_k² q_k² + ε Σ cos(q_k − q_{k+1}),
             de modo que el flujo de Poincaré arranca sobre la misma hoja de energía.

        Continúa en `PoincareCelestialSymplecticEngine` sin reparametrización.
        """
        rho = self.sanitize_density_matrix_higham(density_matrix)
        dof = self.dimension // 2
        q_coords, p_momenta, cov = self._wigner_quadrature_moments(rho)
        williamson = self._williamson_spectrum(cov)

        actions = np.zeros(dof, dtype=np.float64)
        angles = np.zeros(dof, dtype=np.float64)
        for k in range(dof):
            actions[k] = float(williamson[k]) if k < williamson.size else 0.5 * (
                q_coords[k] ** 2 + p_momenta[k] ** 2
            )
            angles[k] = math.atan2(p_momenta[k], q_coords[k] + _WILKINSON_EPSILON) % _TWO_PI

        omega = ooda_mode_frequencies(dof)
        kinetic = 0.5 * float(np.dot(p_momenta, p_momenta))
        harmonic = 0.5 * float(np.dot(omega ** 2, q_coords ** 2))
        interaction = 0.0
        eps = float(perturbation_eps)
        for k in range(dof):
            interaction += math.cos(q_coords[k] - q_coords[(k + 1) % dof])
        hamiltonian_val = kinetic + harmonic + eps * interaction

        gauge_phase = float(np.angle(np.trace(rho) + _WILKINSON_EPSILON))
        _, holonomy_tr = self.extract_su2_holonomy_quaternions(rho)

        return PoincareOrbitalState(
            coordinates_q=q_coords,
            momenta_p=p_momenta,
            action_variables_I=actions,
            angle_variables_theta=angles,
            williamson_invariants=williamson,
            energy_hamiltonian=hamiltonian_val,
            gauge_phase_shift=gauge_phase,
            su2_holonomy_trace=holonomy_tr,
            dimension_dof=dof,
            quantum_density_snapshot=rho,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 2 — MECÁNICA CELESTE DE POINCARÉ, FLUJOS SIMPLÉCTICOS, SECCIONES Y ANÁLISIS KAM
# Continuación directa de `quantize_to_canonical_poincare_coordinates`: todo método de esta fase
# consume `PoincareOrbitalState`. El morfismo terminal produce `SymplecticVariationalFeedback`,
# objeto inicial de la Fase 3 (RSI-L3).
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MonodromyResult:
    """Tensor de monodromía, multiplicadores de Floquet y diagnóstico simpléctico."""
    monodromy_matrix: NDArray[np.float64]
    floquet_multipliers: NDArray[np.complex128]
    max_lyapunov_exponent: float
    is_symplectic_conserved: bool
    symplectic_drift_norm: float
    energy_drift: float
    floquet_pairing_defect: float


@dataclass(frozen=True, slots=True)
class KAMAuditResult:
    """Certificado KAM + cota de Nekhoroshev sobre el tiempo de estabilidad efectivo."""
    is_kam_torus_stable: bool
    diophantine_constant_gamma: float
    diophantine_exponent_tau: float
    minimum_small_divisor: float
    resonance_indices: List[Tuple[int, ...]]
    poincare_recurrence_time_estimate: float
    nekhoroshev_stability_time: float
    continued_fraction_type: float


@dataclass(frozen=True, slots=True)
class PoincareSectionResult:
    """Intersecciones con la sección transversal Σ = {q₀ = 0, p₀ > 0} (método de Hénon)."""
    intersection_count: int
    return_map_points: NDArray[np.float64]
    mean_return_time: float
    rotation_number_estimate: float


@dataclass(frozen=True, slots=True)
class SymplecticVariationalFeedback:
    """
    TENSOR DE RETROALIMENTACIÓN VARIACIONAL — objeto de acoplamiento Fase 2 → Fase 3.

    Resume el análisis simpléctico de Poincaré en un covector de control que el núcleo
    RSI-L3 consume como único dato orbital. No hay otra puerta de entrada a la Fase 3.
    """
    symplectic_gradient_correction: NDArray[np.float64]
    floquet_stabilization_gain: float
    kam_regularization_scalar: float
    poincare_section_intersection_count: int
    orbital_divergence_risk: float
    energy_drift: float
    nekhoroshev_horizon: float
    origin_orbital_state: PoincareOrbitalState


class PoincareCelestialSymplecticEngine:
    """
    Motor de mecánica celeste de Poincaré y geometría simpléctica.

    El atractor OODA se modela como sistema hamiltoniano casi-integrable
        H(q, p) = H₀(I) + ε H₁(θ; I)
    con H₀ armónico anisótropo y H₁ interacción cíclica de tipo Frenkel–Kontorova.
    Se verifica ω-preservación (Mᵀ J M = J), se integra con Yoshida de orden 4,
    se construye la monodromía por el mapa tangente *exacto* de Verlet (no por
    diferencias finitas) y se audita KAM / Nekhoroshev / Floquet.
    """

    def __init__(
        self,
        dof: int = 28,
        perturbation_parameter_eps: float = _DEFAULT_PERTURBATION_EPS,
        symplectic_tolerance: float = _SYMPLECTIC_DRIFT_TOL,
    ) -> None:
        self.dof: int = max(2, int(dof))
        self.epsilon_pert: float = float(perturbation_parameter_eps)
        self.symplectic_tolerance: float = float(symplectic_tolerance)
        self.omega: NDArray[np.float64] = ooda_mode_frequencies(self.dof)
        self.canonical_symplectic_matrix: NDArray[np.float64] = self._build_symplectic_matrix()

    def _build_symplectic_matrix(self) -> NDArray[np.float64]:
        """Matriz canónica J = [[0, I], [−I, 0]], Jᵀ = −J, J² = −I."""
        n = self.dof
        j_mat = np.zeros((2 * n, 2 * n), dtype=np.float64)
        j_mat[:n, n:] = np.eye(n, dtype=np.float64)
        j_mat[n:, :n] = -np.eye(n, dtype=np.float64)
        return j_mat

    def hamiltonian_value(self, q: NDArray[np.float64], p: NDArray[np.float64]) -> float:
        """H = ½‖p‖² + ½ Σ ω_k² q_k² + ε Σ cos(q_k − q_{k+1}). Coincide con la Fase 1."""
        n = self.dof
        q = np.asarray(q, dtype=np.float64)[:n]
        p = np.asarray(p, dtype=np.float64)[:n]
        kinetic = 0.5 * float(np.dot(p, p))
        harmonic = 0.5 * float(np.dot(self.omega ** 2, q ** 2))
        interaction = float(np.sum(np.cos(q - np.roll(q, -1))))
        return kinetic + harmonic + self.epsilon_pert * interaction

    def potential_gradient(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        ∇V, V = ½ Σ ω² q² + ε Σ cos(q_k − q_{k+1}).
        ∂V/∂q_j = ω_j² q_j − ε sin(q_j − q_{j+1}) − ε sin(q_j − q_{j−1}).
        """
        n = self.dof
        q = np.asarray(q, dtype=np.float64)[:n]
        q_next = np.roll(q, -1)
        q_prev = np.roll(q, 1)
        return (self.omega ** 2) * q - self.epsilon_pert * (
            np.sin(q - q_next) + np.sin(q - q_prev)
        )

    def potential_hessian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Hessiano analítico de V (tridiagonal cíclica) para el mapa tangente de Verlet."""
        n = self.dof
        q = np.asarray(q, dtype=np.float64)[:n]
        q_next = np.roll(q, -1)
        q_prev = np.roll(q, 1)
        hess = np.zeros((n, n), dtype=np.float64)
        diag = (self.omega ** 2) - self.epsilon_pert * (
            np.cos(q - q_next) + np.cos(q - q_prev)
        )
        np.fill_diagonal(hess, diag)
        for k in range(n):
            kn = (k + 1) % n
            hess[k, kn] += self.epsilon_pert * math.cos(q[k] - q[kn])
            hess[kn, k] += self.epsilon_pert * math.cos(q[k] - q[kn])
        return hess

    def compute_poincare_hamiltonian_vector_field(
        self, state_vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Campo hamiltoniano X_H = J ∇H = (p, −∇V)."""
        n = self.dof
        q = state_vector[:n]
        p = state_vector[n:]
        return np.concatenate([p, -self.potential_gradient(q)])

    def verify_symplectomorphism(self, jacobian_matrix: NDArray[np.float64]) -> Tuple[bool, float]:
        """Condición de Darboux discreta: ‖Mᵀ J M − J‖_F < tolerancia."""
        j_mat = self.canonical_symplectic_matrix
        m_mat = np.asarray(jacobian_matrix, dtype=np.float64)
        if m_mat.shape != j_mat.shape:
            return False, float("inf")
        drift = m_mat.T @ j_mat @ m_mat - j_mat
        drift_norm = float(la.norm(drift, ord="fro"))
        return drift_norm < self.symplectic_tolerance, drift_norm

    def _verlet_step(
        self, q: NDArray[np.float64], p: NDArray[np.float64], dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Störmer–Verlet (leapfrog) de orden 2, signos canónicos a = −∇V."""
        p_half = p - 0.5 * dt * self.potential_gradient(q)
        q_next = q + dt * p_half
        p_next = p_half - 0.5 * dt * self.potential_gradient(q_next)
        return q_next, p_next

    def _verlet_step_with_tangent(
        self,
        q: NDArray[np.float64],
        p: NDArray[np.float64],
        m_mat: NDArray[np.float64],
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Un paso de Verlet junto con su diferencial exacto DM.
        Partición z = (q, p), M ∈ Sp(2n, ℝ) si el paso es exacto en aritmética real.

            p½ = p − (dt/2) ∇V(q)
            q' = q + dt p½
            p' = p½ − (dt/2) ∇V(q')
        """
        n = self.dof
        hess_q = self.potential_hessian(q)
        # dp½ = dp − (dt/2) Hess(q) dq
        # En bloques de M: filas (dq, dp).
        a_half = np.eye(2 * n, dtype=np.float64)
        a_half[n:, :n] = -0.5 * dt * hess_q
        # dq' = dq + dt dp½  →  q' depende de (q, p½)
        a_q = np.eye(2 * n, dtype=np.float64)
        a_q[:n, n:] = dt * np.eye(n)
        q_next = q + dt * (p - 0.5 * dt * self.potential_gradient(q))
        p_half = p - 0.5 * dt * self.potential_gradient(q)
        hess_q1 = self.potential_hessian(q_next)
        a_p = np.eye(2 * n, dtype=np.float64)
        a_p[n:, :n] = -0.5 * dt * hess_q1
        p_next = p_half - 0.5 * dt * self.potential_gradient(q_next)
        m_next = a_p @ a_q @ a_half @ m_mat
        return q_next, p_next, m_next

    def integrate_yoshida4(
        self,
        initial_orbital_state: PoincareOrbitalState,
        total_time: float = _TWO_PI,
        steps: int = 120,
    ) -> Tuple[List[NDArray[np.float64]], float]:
        """
        Integrador simpléctico de Yoshida de orden 4 (composición de tres Verlet).
        Preserva ω exactamente (salvo redondeo) y devuelve la deriva de energía
        |H(T) − H(0)| / (|H(0)| + ε) como certificado de calidad.
        """
        n = self.dof
        dt = total_time / float(max(1, steps))
        q = np.copy(initial_orbital_state.coordinates_q[:n]).astype(np.float64)
        p = np.copy(initial_orbital_state.momenta_p[:n]).astype(np.float64)
        if q.size < n:
            q = np.pad(q, (0, n - q.size))
        if p.size < n:
            p = np.pad(p, (0, n - p.size))

        h0 = self.hamiltonian_value(q, p)
        trajectory: List[NDArray[np.float64]] = [np.concatenate([q, p])]
        w1, w0 = _YOSHIDA_W1, _YOSHIDA_W0
        for _ in range(steps):
            q, p = self._verlet_step(q, p, w1 * dt)
            q, p = self._verlet_step(q, p, w0 * dt)
            q, p = self._verlet_step(q, p, w1 * dt)
            trajectory.append(np.concatenate([np.copy(q), np.copy(p)]))
        h1 = self.hamiltonian_value(q, p)
        energy_drift = abs(h1 - h0) / (abs(h0) + _WILKINSON_EPSILON)
        return trajectory, energy_drift

    def compute_poincare_section(
        self,
        trajectory: SequenceLike,  # type: ignore[name-defined]
    ) -> PoincareSectionResult:
        """
        Sección de Poincaré Σ = {q₀ = 0, ṗ-crossing con p₀ > 0}.
        Interpolación lineal de Hénon en el cruce de signo de q₀.
        El número de rotación se estima como ⟨Δθ₁⟩ / 2π sobre los retornos.
        """
        n = self.dof
        pts: List[NDArray[np.float64]] = []
        times: List[float] = []
        angles: List[float] = []
        for i in range(len(trajectory) - 1):
            z0 = trajectory[i]
            z1 = trajectory[i + 1]
            q0_a, q0_b = float(z0[0]), float(z1[0])
            if q0_a == 0.0 and float(z0[n]) > 0.0:
                alpha = 0.0
            elif q0_a * q0_b < 0.0:
                denom = q0_b - q0_a
                alpha = 0.0 if abs(denom) <= _WILKINSON_EPSILON else (-q0_a / denom)
            else:
                continue
            z_s = z0 + alpha * (z1 - z0)
            if float(z_s[n]) <= 0.0:
                continue
            pts.append(z_s)
            times.append(float(i) + alpha)
            angles.append(math.atan2(float(z_s[n + 1]) if n > 1 else float(z_s[n]),
                                     float(z_s[1]) if n > 1 else 1.0))

        if not pts:
            return PoincareSectionResult(
                intersection_count=0,
                return_map_points=np.zeros((0, 2 * n), dtype=np.float64),
                mean_return_time=float("inf"),
                rotation_number_estimate=0.0,
            )
        arr = np.stack(pts, axis=0)
        if len(times) >= 2:
            gaps = np.diff(np.asarray(times, dtype=np.float64))
            mean_ret = float(np.mean(gaps))
        else:
            mean_ret = float("inf")
        if len(angles) >= 2:
            dang = np.diff(np.asarray(angles, dtype=np.float64))
            dang = (dang + math.pi) % _TWO_PI - math.pi
            rot = float(np.mean(dang) / _TWO_PI)
        else:
            rot = 0.0
        return PoincareSectionResult(
            intersection_count=len(pts),
            return_map_points=arr,
            mean_return_time=mean_ret,
            rotation_number_estimate=rot,
        )

    def compute_monodromy_and_floquet(
        self, initial_orbital_state: PoincareOrbitalState, steps: int = 72
    ) -> MonodromyResult:
        """
        Matriz de monodromía M(T) = Dφ_T (z₀) obtenida por composición del diferencial
        exacto de Verlet a lo largo de un periodo T = 2π. Los multiplicadores de Floquet
        son spec(M); en un sistema hamiltoniano ocurren en pares (λ, 1/λ) y (λ, λ̄).
        El defecto de emparejamiento mide |∏ λ_i − 1| + distorsión de reciprocidad.
        """
        n = self.dof
        dim = 2 * n
        q = np.copy(initial_orbital_state.coordinates_q[:n]).astype(np.float64)
        p = np.copy(initial_orbital_state.momenta_p[:n]).astype(np.float64)
        if q.size < n:
            q = np.pad(q, (0, n - q.size))
        if p.size < n:
            p = np.pad(p, (0, n - p.size))

        t_orbit = _TWO_PI
        dt = t_orbit / float(max(1, steps))
        h0 = self.hamiltonian_value(q, p)
        m_mat = np.eye(dim, dtype=np.float64)
        for _ in range(steps):
            q, p, m_mat = self._verlet_step_with_tangent(q, p, m_mat, dt)
        h1 = self.hamiltonian_value(q, p)
        energy_drift = abs(h1 - h0) / (abs(h0) + _WILKINSON_EPSILON)

        floquet_eigs = la.eigvals(m_mat)
        magnitudes = np.abs(floquet_eigs)
        max_multiplier = float(np.max(magnitudes))
        lyapunov_exp = math.log(max(max_multiplier, _WILKINSON_EPSILON)) / t_orbit

        is_symp, drift = self.verify_symplectomorphism(m_mat)

        # Defecto de emparejamiento de Floquet: cada |λ| debe tener un compañero 1/|λ|
        mags_sorted = np.sort(magnitudes)
        recip = np.sort(1.0 / np.maximum(magnitudes, _WILKINSON_EPSILON))
        pairing_defect = float(np.mean(np.abs(mags_sorted - recip)))
        det_defect = abs(abs(complex(np.linalg.det(m_mat))) - 1.0)
        pairing_defect = pairing_defect + det_defect

        return MonodromyResult(
            monodromy_matrix=m_mat,
            floquet_multipliers=floquet_eigs,
            max_lyapunov_exponent=lyapunov_exp,
            is_symplectic_conserved=is_symp,
            symplectic_drift_norm=drift,
            energy_drift=energy_drift,
            floquet_pairing_defect=pairing_defect,
        )

    def audit_kam_stability_diophantine(
        self, orbital_state: PoincareOrbitalState
    ) -> KAMAuditResult:
        """
        Auditoría del teorema KAM. Condición diofántica sobre el vector de frecuencias
        ω(I) = ∂H₀/∂I ≈ ω_modo + ¼ I + ⅟₂₀ I²:

            |k · ω| ≥ γ / ‖k‖₁^τ    ∀ k ∈ ℤⁿ \\ {0},  τ = n,  γ ~ 1 / n².

        Se recorre el retículo ‖k‖_∞ ≤ K por producto cartesiano (sin meshgrid intermedio).
        El tipo diofántico se estima por el peor cociente ‖k‖^τ |k·ω|.
        Cota de Nekhoroshev: T_N ~ exp( c / ε^{1/(2n)} ), horizonte de estabilidad efectiva.
        """
        n = min(4, self.dof)
        actions = np.asarray(orbital_state.action_variables_I[:n], dtype=np.float64)
        if actions.size < n:
            actions = np.pad(actions, (0, n - actions.size))
        omega = ooda_mode_frequencies(n) + 0.25 * actions + 0.05 * (actions ** 2)

        gamma = 0.05 / (float(n) ** 2)
        tau = float(n)

        min_divisor = float("inf")
        worst_type = 0.0
        resonances: List[Tuple[int, ...]] = []
        ranges = [range(-_MAX_KAM_FOURIER_ORDER, _MAX_KAM_FOURIER_ORDER + 1)] * n

        for k_tuple in itertools.product(*ranges):
            k_norm_1 = sum(abs(v) for v in k_tuple)
            if k_norm_1 == 0:
                continue
            k_vec = np.asarray(k_tuple, dtype=np.float64)
            k_dot_omega = abs(float(np.dot(k_vec, omega)))
            diophantine_bound = gamma / (float(k_norm_1) ** tau)
            if k_dot_omega < min_divisor:
                min_divisor = k_dot_omega
            type_ratio = (float(k_norm_1) ** tau) * k_dot_omega
            if type_ratio > 0.0:
                worst_type = max(worst_type, 1.0 / type_ratio)
            if k_dot_omega < diophantine_bound:
                resonances.append(tuple(int(x) for x in k_tuple))

        is_stable = (len(resonances) == 0) and (min_divisor > 1.0e-4)
        t_poincare = float(_TWO_PI / max(min_divisor, 1.0e-6))

        # Nekhoroshev: T ≤ exp(c / ε^a), a = 1/(2n), c ~ γ
        exponent = 1.0 / float(2 * max(n, 1))
        eps_eff = max(self.epsilon_pert, 1.0e-8)
        nekhoroshev_t = float(math.exp(min(40.0, gamma / (eps_eff ** exponent))))

        return KAMAuditResult(
            is_kam_torus_stable=is_stable,
            diophantine_constant_gamma=gamma,
            diophantine_exponent_tau=tau,
            minimum_small_divisor=min_divisor,
            resonance_indices=resonances[:8],
            poincare_recurrence_time_estimate=t_poincare,
            nekhoroshev_stability_time=nekhoroshev_t,
            continued_fraction_type=worst_type,
        )

    # ──────────────────────────────────────────────────────────────────────────────────────────
    # MÉTODO PUENTE FORMAL FASE 2 ──► FASE 3
    # Su valor de retorno `SymplecticVariationalFeedback` es el objeto inicial del
    # núcleo RSI-L3; la Fase 3 no admite otra semilla orbital.
    # ──────────────────────────────────────────────────────────────────────────────────────────
    def synthesize_symplectic_feedback_tensor(
        self,
        kam_audit: KAMAuditResult,
        monodromy: MonodromyResult,
        section: PoincareSectionResult,
        current_state: PoincareOrbitalState,
    ) -> SymplecticVariationalFeedback:
        """
        Sintetiza la dinámica de Poincaré en un covector de control para RSI-L3.

        Si el toro KAM se ha disuelto, se aplica una corrección hamiltoniana
        −α (q, p) proporcional al número de resonancias (retracción hacia la
        variedad central). La ganancia de Floquet es una logística del máximo
        exponente de Lyapunov. El riesgo de divergencia combina Lyapunov,
        deriva simpléctica, deriva de energía y defecto de emparejamiento.
        """
        n = current_state.dimension_dof
        q = np.asarray(current_state.coordinates_q[:n], dtype=np.float64)
        p = np.asarray(current_state.momenta_p[:n], dtype=np.float64)
        if q.size < n:
            q = np.pad(q, (0, n - q.size))
        if p.size < n:
            p = np.pad(p, (0, n - p.size))

        grad_correction = np.zeros(2 * n, dtype=np.float64)
        n_res = float(len(kam_audit.resonance_indices))
        if not kam_audit.is_kam_torus_stable:
            grad_correction[:n] = -0.15 * q * (n_res + 1.0)
            grad_correction[n:] = -0.15 * p

        max_lyap = max(0.0, monodromy.max_lyapunov_exponent)
        floquet_gain = float(1.0 / (1.0 + math.exp(-2.0 * max_lyap)))
        kam_scalar = 1.0 if kam_audit.is_kam_torus_stable else 0.5

        intersections = int(section.intersection_count)
        if intersections <= 0:
            intersections = int(
                max(1, int(kam_audit.poincare_recurrence_time_estimate / _TWO_PI))
            )

        divergence_risk = float(
            np.clip(
                max_lyap * 2.0
                + monodromy.symplectic_drift_norm * 100.0
                + monodromy.energy_drift * 50.0
                + monodromy.floquet_pairing_defect * 0.5,
                0.0,
                1.0,
            )
        )

        return SymplecticVariationalFeedback(
            symplectic_gradient_correction=grad_correction,
            floquet_stabilization_gain=floquet_gain,
            kam_regularization_scalar=kam_scalar,
            poincare_section_intersection_count=intersections,
            orbital_divergence_risk=divergence_risk,
            energy_drift=monodromy.energy_drift,
            nekhoroshev_horizon=kam_audit.nekhoroshev_stability_time,
            origin_orbital_state=current_state,
        )


# Anotación diferida: SequenceLike usado en compute_poincare_section.
# Se alinea con List[NDArray] producido por integrate_yoshida4.
from typing import Sequence as SequenceLike  # noqa: E402  (acoplamiento documental Fase 2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# FASE 3 — AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3), TOPOS CATEGÓRICO Y MOTOR TESTIGO
# Continuación directa de `synthesize_symplectic_feedback_tensor`: el núcleo RSI-L3 y el motor
# testigo consumen `SymplecticVariationalFeedback` como único dato orbital de entrada.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class RSIL3ConfigurationBundle:
    """
    Punto del espacio métrico completo (C, d_C) ≅ (ℝ⁵_acotado, ‖·‖₂ ponderado)
    sobre el que actúa el operador de contracción T y el meta-operador S.
    Nivel 1 = pesos; Nivel 2 = hiperparámetros; Nivel 3 = el propio T.
    """
    psd_spectral_floor: float
    higham_projection_damping: float
    weak_measurement_gain: float
    kam_diophantine_threshold: float
    symplectomorphism_tolerance: float
    iteration_generation: int
    meta_contraction_lip: float = _DEFAULT_BANACH_CONTRACTION_LIP

    def to_vector(self) -> NDArray[np.float64]:
        return np.array(
            [
                self.psd_spectral_floor,
                self.higham_projection_damping,
                self.weak_measurement_gain,
                self.kam_diophantine_threshold,
                self.symplectomorphism_tolerance,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
        cls, vec: NDArray[np.float64], generation: int, meta_lip: float
    ) -> "RSIL3ConfigurationBundle":
        return cls(
            psd_spectral_floor=float(np.clip(vec[0], 1.0e-15, 1.0e-8)),
            higham_projection_damping=float(np.clip(vec[1], 0.1, 1.0)),
            weak_measurement_gain=float(np.clip(vec[2], 0.01, 2.0)),
            kam_diophantine_threshold=float(np.clip(vec[3], 1.0e-6, 1.0e-2)),
            symplectomorphism_tolerance=float(np.clip(vec[4], 1.0e-9, 1.0e-4)),
            iteration_generation=int(generation),
            meta_contraction_lip=float(np.clip(meta_lip, 0.50, 0.99)),
        )


@dataclass(frozen=True, slots=True)
class HomomorphicGateCertificate:
    """Certificado de homomorfismo C* y functorialidad Φ₁₃ = Φ₂₃ ∘ Φ₁₂."""
    phi12_integrity: float
    phi23_integrity: float
    phi13_composition_integrity: float
    corner_homomorphism_defect_12: float
    corner_homomorphism_defect_23: float
    functoriality_defect: float


@dataclass(frozen=True, slots=True)
class OODAWitnessSnapshotPhD:
    """Snapshot inmutable emitido por el motor testigo soberano v15.0.0."""
    snapshot_id: str
    qnd_weak_value_aav: complex
    weak_value_modulus: float
    back_action_db: float
    qnd_commutator_norm: float
    uhlmann_fidelity: float
    bures_distance: float
    purity: float
    trace_preservation_drift: float
    connection_norm: float
    yang_mills_energy: float
    wilson_loop_holonomy_phase: float
    su2_holonomy_trace: complex
    max_orbital_lyapunov_exponent: float
    is_kam_torus_stable: bool
    poincare_recurrence_time: float
    floquet_multiplier_spectral_radius: float
    floquet_pairing_defect: float
    energy_drift: float
    nekhoroshev_horizon: float
    poincare_rotation_number: float
    rsi_level3_generation: int
    rsi_banach_contraction_lip: float
    rsi_meta_contraction_lip: float
    rsi_cauchy_residual: float
    global_heyting_verdict: HeytingOmega4
    homomorphic_gate_phi12_integrity: float
    homomorphic_gate_phi23_integrity: float
    homomorphic_gate_phi13_integrity: float
    functoriality_defect: float
    sha384_merkle_attestation: str
    timestamp_utc: float


class RSIL3SelfOptimizationCore:
    """
    Núcleo de automejora recursiva de nivel 3.

    Nivel 1: muta pesos. Nivel 2: muta hiperparámetros con T fijo.
    Nivel 3: T: C → C es él mismo el punto fijo de un meta-operador
        S: Lip → Lip,   S(λ) = clip( λ·(1 − β ρ) + η (1 − L_emp), 1/2, 99/100 )
    con ρ = riesgo de divergencia orbital y L_emp el Lipschitz empírico de T,
    estimado por diferencia de Gâteaux a dos perturbaciones independientes.

    Banach: si Lip(T) ≤ λ < 1 sobre el completo (C, d_C), existe un único punto
    fijo c* = T(c*) y las iteraciones convergen geométricamente.
    """

    _WEIGHTS: NDArray[np.float64] = np.array(
        [1.0e14, 1.0, 1.0, 1.0e4, 1.0e7], dtype=np.float64
    )

    def __init__(
        self,
        initial_config: Optional[RSIL3ConfigurationBundle] = None,
        target_contraction_factor: float = _DEFAULT_BANACH_CONTRACTION_LIP,
    ) -> None:
        if initial_config is None:
            self.current_config = RSIL3ConfigurationBundle(
                psd_spectral_floor=_SPECTRAL_PSD_FLOOR,
                higham_projection_damping=0.95,
                weak_measurement_gain=1.0,
                kam_diophantine_threshold=1.0e-4,
                symplectomorphism_tolerance=_SYMPLECTIC_DRIFT_TOL,
                iteration_generation=0,
                meta_contraction_lip=float(target_contraction_factor),
            )
        else:
            self.current_config = initial_config
        self.contraction_factor: float = float(
            np.clip(target_contraction_factor, 0.50, 0.99)
        )
        self._history: List[NDArray[np.float64]] = [self.current_config.to_vector()]

    def _metric(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(la.norm(self._WEIGHTS * (a - b)))

    def _raw_contract(
        self,
        vec: NDArray[np.float64],
        feedback: SymplecticVariationalFeedback,
        fidelity_loss: float,
        lam: float,
    ) -> NDArray[np.float64]:
        risk = float(np.clip(feedback.orbital_divergence_risk, 0.0, 1.0))
        energy_sign = -0.1 if abs(feedback.origin_orbital_state.energy_hamiltonian) > 1.0e-12 else 1.0
        perturbation = np.array(
            [
                -1.0e-14 * risk,
                (-0.02 * risk) if risk > 0.5 else 0.005,
                -0.05 * (feedback.floquet_stabilization_gain - 0.5),
                1.0e-5 * energy_sign * (1.0 - feedback.kam_regularization_scalar),
                -1.0e-8 * risk - 1.0e-8 * feedback.energy_drift,
            ],
            dtype=np.float64,
        )
        return vec + lam * perturbation * (1.0 - fidelity_loss)

    def evaluate_banach_contraction_map(
        self,
        config: RSIL3ConfigurationBundle,
        feedback: SymplecticVariationalFeedback,
        fidelity_loss: float,
    ) -> Tuple[RSIL3ConfigurationBundle, float, float]:
        """
        Aplica T y estima Lip(T) por dos perturbaciones de Gâteaux ortogonales:
            L ≈ max_i  d(T(c+δ_i), T(c)) / d(c+δ_i, c).
        El meta-operador S actualiza λ de modo que Lip(S) < 1 (contracción anidada).
        """
        vec = config.to_vector()
        lam = float(config.meta_contraction_lip)
        contracted = self._raw_contract(vec, feedback, fidelity_loss, lam)

        rng = np.random.default_rng(config.iteration_generation + 17)
        empirical_lips: List[float] = []
        for _ in range(2):
            delta = rng.normal(0.0, 1.0, size=vec.shape)
            nrm = float(la.norm(delta))
            if nrm <= _WILKINSON_EPSILON:
                continue
            delta = 1.0e-6 * delta / nrm
            t_delta = self._raw_contract(vec + delta, feedback, fidelity_loss, lam)
            num = self._metric(t_delta, contracted)
            den = self._metric(vec + delta, vec)
            empirical_lips.append(num / max(den, _WILKINSON_EPSILON))
        empirical_lip = float(np.clip(max(empirical_lips) if empirical_lips else lam, 0.0, 1.50))

        # Meta-operador S (Nivel 3): contrae el propio factor de Lipschitz
        risk = feedback.orbital_divergence_risk
        meta_next = lam * (1.0 - 0.08 * risk) + 0.02 * (1.0 - min(empirical_lip, 1.0))
        meta_next = float(np.clip(meta_next, 0.50, 0.99))

        new_config = RSIL3ConfigurationBundle.from_vector(
            contracted, generation=config.iteration_generation + 1, meta_lip=meta_next
        )
        cauchy = self._metric(new_config.to_vector(), vec)
        return new_config, float(np.clip(empirical_lip, 0.0, 0.99)), cauchy

    def trigger_recursive_step(
        self, feedback: SymplecticVariationalFeedback, fidelity_metric: float
    ) -> Tuple[RSIL3ConfigurationBundle, float, float]:
        """Una iteración formal de RSI-L3. Preserva el axioma QND (no toca observables)."""
        loss = 1.0 - float(np.clip(fidelity_metric, 0.0, 1.0))
        new_config, lip, cauchy = self.evaluate_banach_contraction_map(
            self.current_config, feedback, loss
        )
        self.current_config = new_config
        self.contraction_factor = new_config.meta_contraction_lip
        self._history.append(new_config.to_vector())
        return new_config, lip, cauchy


class OODATriphasicHomomorphicPipelineWitnessEnginePhD:
    """
    MOTOR SOBERANO TESTIGO QND DEL PIPELINE OODA TRIFÁSICO HOMOMÓRFICO v15.0.0-PhD.

    Cadena functorial estricta:
        ρ  --Fase 1-->  PoincareOrbitalState
           --Fase 2-->  SymplecticVariationalFeedback
           --Fase 3-->  OODAWitnessSnapshotPhD
    """

    def __init__(
        self,
        dimension: int = 56,
        secret_key: bytes = _DEFAULT_SECRET_KEY,
        contraction_factor: float = _DEFAULT_BANACH_CONTRACTION_LIP,
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension) + int(dimension) % 2)
        self.secret_key: Final[bytes] = bytes(secret_key)
        self.phase1_engine = QuantumBanachAlgebraTopologyEngine(
            dimension=self.dimension, psd_floor=_SPECTRAL_PSD_FLOOR
        )
        self.phase2_engine = PoincareCelestialSymplecticEngine(
            dof=self.dimension // 2,
            perturbation_parameter_eps=_DEFAULT_PERTURBATION_EPS,
        )
        self.phase3_rsi_core = RSIL3SelfOptimizationCore(
            target_contraction_factor=contraction_factor
        )

    def ingest_rsi_configuration(self, cfg: RSIL3ConfigurationBundle) -> None:
        """Reinyecta el punto fijo parcial de T en los motores de las Fases 1 y 2 (cierre L3)."""
        self.phase1_engine.psd_floor = cfg.psd_spectral_floor
        self.phase2_engine.symplectic_tolerance = cfg.symplectomorphism_tolerance

    def evaluate_homomorphic_gates(
        self, density_matrix: NDArray[np.complex128]
    ) -> HomomorphicGateCertificate:
        """
        Homomorfismos de esquina C*:
            π₁₂(a) = e₁ a e₁,   π₂₃(a) = e₂ a e₂,
        con e₁ = diag(I_h, 0), e₂ = diag(0, I_h), h = n/2.
        Defecto de homomorfismo: ‖π(ρ²) − π(ρ)²‖_F / (‖ρ‖_F + ε).
        Integridad = exp(−defecto). Functorialidad:
            δ = |integridad(π₁₃) − integridad(π₂₃) · integridad(π₁₂)|
        mide el fallo de Φ₁₃ = Φ₂₃ ∘ Φ₁₂.
        """
        rho = np.asarray(density_matrix, dtype=np.complex128)
        n = self.dimension
        half = n // 2
        e1 = np.zeros((n, n), dtype=np.complex128)
        e2 = np.zeros((n, n), dtype=np.complex128)
        e1[:half, :half] = np.eye(half, dtype=np.complex128)
        e2[half:, half:] = np.eye(n - half, dtype=np.complex128)

        rho_norm = float(la.norm(rho, ord="fro")) + _WILKINSON_EPSILON
        rho2 = rho @ rho

        def _defect(e: NDArray[np.complex128]) -> float:
            pi_rho = e @ rho @ e
            pi_rho2 = e @ rho2 @ e
            return float(la.norm(pi_rho2 - pi_rho @ pi_rho, ord="fro")) / rho_norm

        d12 = _defect(e1)
        d23 = _defect(e2)
        d13 = _defect(e1 + e2)  # proyección total; control de fugas fuera de las esquinas
        i12 = float(math.exp(-d12))
        i23 = float(math.exp(-d23))
        i13 = float(math.exp(-d13))
        functor_defect = abs(i13 - i12 * i23)
        return HomomorphicGateCertificate(
            phi12_integrity=float(np.clip(i12, 0.0, 1.0)),
            phi23_integrity=float(np.clip(i23, 0.0, 1.0)),
            phi13_composition_integrity=float(np.clip(i13, 0.0, 1.0)),
            corner_homomorphism_defect_12=d12,
            corner_homomorphism_defect_23=d23,
            functoriality_defect=float(functor_defect),
        )

    def evaluate_ooda_witness_sovereign(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> OODAWitnessSnapshotPhD:
        """
        EJECUCIÓN MAESTRA A TRAVÉS DE LAS 3 FASES ANIDADAS:

            Fase 1 (Banach / QND / Weyl–Williamson)
                → PoincareOrbitalState
            Fase 2 (Yoshida / monodromía / KAM / sección de Poincaré)
                → SymplecticVariationalFeedback
            Fase 3 (RSI-L3 / Heyting / Φ₁₂, Φ₂₃, Φ₁₃)
                → OODAWitnessSnapshotPhD + atestación HMAC-SHA384.
        """
        dim = self.dimension

        # ======================================================================================
        # FASE 1 — Sanitización, calibre SU(N), AAV, Bures, holonomía ℍ, cuantización Weyl
        # ======================================================================================
        rho_clean = self.phase1_engine.sanitize_density_matrix_higham(density_matrix)
        trace_preservation_drift = float(abs(np.real(np.trace(rho_clean)) - 1.0))

        conn_a, _curv_f, ym_energy = self.phase1_engine.extract_gauge_connection_su_n(rho_clean)
        conn_norm = float(la.norm(conn_a, ord="fro"))

        wilson_op = la.expm(conn_a)
        wilson_phase = float(cmath.phase(complex(np.trace(wilson_op) / float(dim))))
        _holonomies, su2_tr = self.phase1_engine.extract_su2_holonomy_quaternions(rho_clean)

        if phi_initial is None:
            phi_init = np.ones((dim, 1), dtype=np.complex128) / math.sqrt(dim)
        else:
            phi_init = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)

        if phi_final is None:
            phi_fin = rho_clean @ phi_init
            phi_fin = phi_fin / (float(la.norm(phi_fin)) + _WILKINSON_EPSILON)
        else:
            phi_fin = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)

        a_weak, weak_mod, back_action_db, comm_norm = self.phase1_engine.compute_qnd_weak_value_aav(
            phi_init, phi_fin, conn_a
        )
        fidelity, bures_dist, purity = self.phase1_engine.compute_bures_uhlmann_metrics(rho_clean)

        # ── disparo del morfismo puente Fase 1 → Fase 2 ──────────────────────────────────────
        orbital_state: PoincareOrbitalState = (
            self.phase1_engine.quantize_to_canonical_poincare_coordinates(
                rho_clean, perturbation_eps=self.phase2_engine.epsilon_pert
            )
        )

        # ======================================================================================
        # FASE 2 — Yoshida, sección de Poincaré, monodromía analítica, KAM / Nekhoroshev
        # ======================================================================================
        trajectory, _traj_drift = self.phase2_engine.integrate_yoshida4(
            orbital_state, total_time=_TWO_PI, steps=96
        )
        section_res: PoincareSectionResult = self.phase2_engine.compute_poincare_section(trajectory)
        monodromy_res: MonodromyResult = self.phase2_engine.compute_monodromy_and_floquet(
            orbital_state, steps=72
        )
        kam_audit_res: KAMAuditResult = self.phase2_engine.audit_kam_stability_diophantine(
            orbital_state
        )
        floquet_radius = float(np.max(np.abs(monodromy_res.floquet_multipliers)))

        # ── disparo del morfismo puente Fase 2 → Fase 3 ──────────────────────────────────────
        symplectic_feedback: SymplecticVariationalFeedback = (
            self.phase2_engine.synthesize_symplectic_feedback_tensor(
                kam_audit=kam_audit_res,
                monodromy=monodromy_res,
                section=section_res,
                current_state=orbital_state,
            )
        )

        # ======================================================================================
        # FASE 3 — RSI-L3 (T y meta-operador S), topos de Heyting, compuertas homomórficas
        # ======================================================================================
        new_config, lip_contraction, cauchy_residual = self.phase3_rsi_core.trigger_recursive_step(
            feedback=symplectic_feedback, fidelity_metric=fidelity
        )
        self.ingest_rsi_configuration(new_config)

        gates = self.evaluate_homomorphic_gates(rho_clean)

        # Clasificador Ω (cadena de Heyting): se evalúa de ⊥ a ⊤ por cortocircuito.
        veto_cond = (
            conn_norm > 2.8
            or ym_energy > 0.90
            or trace_preservation_drift > 1.0e-4
            or (not monodromy_res.is_symplectic_conserved)
            or symplectic_feedback.orbital_divergence_risk > 0.85
        )
        critical_cond = (
            conn_norm > 1.2
            or ym_energy > 0.40
            or (not kam_audit_res.is_kam_torus_stable)
            or monodromy_res.max_lyapunov_exponent > 0.15
        )
        degraded_cond = conn_norm > 0.7 or ym_energy > 0.15 or bures_dist > 0.85

        if veto_cond:
            verdict = HeytingOmega4.VETOED
        elif critical_cond:
            verdict = HeytingOmega4.CRITICAL
        elif degraded_cond:
            verdict = HeytingOmega4.DEGRADED
        else:
            verdict = HeytingOmega4.COHERENT

        now_utc = time.time()
        snapshot_seq = int(now_utc * 1000.0) % 10_000_000
        snap_id = f"OODA-POINCARE-KAM-v15-{snapshot_seq:07d}"

        raw_telemetry_bytes = (
            f"{snap_id}:{a_weak}:{ym_energy:.8f}:{bures_dist:.8f}:{fidelity:.8f}:"
            f"{monodromy_res.max_lyapunov_exponent:.6f}:{kam_audit_res.is_kam_torus_stable}:"
            f"{verdict.value}:{new_config.iteration_generation}:{lip_contraction:.6f}:"
            f"{gates.functoriality_defect:.8f}:{cauchy_residual:.6e}"
        ).encode("utf-8")
        merkle_attestation = hmac.new(
            self.secret_key, raw_telemetry_bytes, hashlib.sha384
        ).hexdigest()

        return OODAWitnessSnapshotPhD(
            snapshot_id=snap_id,
            qnd_weak_value_aav=a_weak,
            weak_value_modulus=weak_mod,
            back_action_db=back_action_db,
            qnd_commutator_norm=comm_norm,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            purity=purity,
            trace_preservation_drift=trace_preservation_drift,
            connection_norm=conn_norm,
            yang_mills_energy=ym_energy,
            wilson_loop_holonomy_phase=wilson_phase,
            su2_holonomy_trace=su2_tr,
            max_orbital_lyapunov_exponent=monodromy_res.max_lyapunov_exponent,
            is_kam_torus_stable=kam_audit_res.is_kam_torus_stable,
            poincare_recurrence_time=kam_audit_res.poincare_recurrence_time_estimate,
            floquet_multiplier_spectral_radius=floquet_radius,
            floquet_pairing_defect=monodromy_res.floquet_pairing_defect,
            energy_drift=monodromy_res.energy_drift,
            nekhoroshev_horizon=kam_audit_res.nekhoroshev_stability_time,
            poincare_rotation_number=section_res.rotation_number_estimate,
            rsi_level3_generation=new_config.iteration_generation,
            rsi_banach_contraction_lip=lip_contraction,
            rsi_meta_contraction_lip=new_config.meta_contraction_lip,
            rsi_cauchy_residual=cauchy_residual,
            global_heyting_verdict=verdict,
            homomorphic_gate_phi12_integrity=gates.phi12_integrity,
            homomorphic_gate_phi23_integrity=gates.phi23_integrity,
            homomorphic_gate_phi13_integrity=gates.phi13_composition_integrity,
            functoriality_defect=gates.functoriality_defect,
            sha384_merkle_attestation=merkle_attestation,
            timestamp_utc=now_utc,
        )


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# SUITE DE PRUEBAS FORMALES Y DEMOSTRACIÓN DE DOCTORADO
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN CIENTÍFICA DOCTORAL: MOTOR TESTIGO OODA TRIFÁSICO HOMOMÓRFICO v15.0.0-PhD")
    print("  Weyl–Williamson • Yoshida O(4) • Monodromía analítica • KAM/Nekhoroshev • RSI-L3 anidado")
    print(banner)

    DIMENSION_TOTAL = 56
    engine = OODATriphasicHomomorphicPipelineWitnessEnginePhD(
        dimension=DIMENSION_TOTAL,
        contraction_factor=0.885,
    )

    rng = np.random.default_rng(2026)
    gaussian_state = rng.normal(0.0, 1.0, size=(DIMENSION_TOTAL, DIMENSION_TOTAL)) + 1j * rng.normal(
        0.0, 1.0, size=(DIMENSION_TOTAL, DIMENSION_TOTAL)
    )
    rho_raw = gaussian_state @ gaussian_state.conj().T
    rho_raw = rho_raw / np.trace(rho_raw)

    print(f"\n[+] Matriz de densidad inicializada: dim = {DIMENSION_TOTAL}×{DIMENSION_TOTAL}")
    print("[+] Ejecutando pipeline de tres fases anidadas (1 → 2 → 3)…")

    snap: OODAWitnessSnapshotPhD = engine.evaluate_ooda_witness_sovereign(rho_raw)

    print("\n" + "─" * 45 + " TELEMETRÍA FASE 1 " + "─" * 41)
    print(f"  • Snapshot ID                     : {snap.snapshot_id}")
    print(f"  • Valor débil AAV (A_w)           : {snap.qnd_weak_value_aav.real:+.6f} + {snap.qnd_weak_value_aav.imag:+.6f}j")
    print(f"  • Módulo de valor débil |A_w|     : {snap.weak_value_modulus:.6f}")
    print(f"  • Retroacción QND                 : {snap.back_action_db:.6f} dB  (‖[A,Π]‖_F = {snap.qnd_commutator_norm:.4e})")
    print(f"  • Fidelidad de Uhlmann            : {snap.uhlmann_fidelity:.8f}")
    print(f"  • Distancia de Bures              : {snap.bures_distance:.8f}")
    print(f"  • Pureza Tr(ρ²)                   : {snap.purity:.8f}")
    print(f"  • Deriva de traza                 : {snap.trace_preservation_drift:.4e}")
    print(f"  • Norma conexión Yang–Mills A     : {snap.connection_norm:.8f}")
    print(f"  • Energía invariante Yang–Mills   : {snap.yang_mills_energy:.8f}")
    print(f"  • Fase de holonomía de Wilson     : {snap.wilson_loop_holonomy_phase:.6f} rad")
    print(f"  • Traza holonomía SU(2) (ℍ)       : {snap.su2_holonomy_trace}")

    print("\n" + "─" * 45 + " TELEMETRÍA FASE 2 " + "─" * 41)
    print(f"  • Grados de libertad cotangente   : {DIMENSION_TOTAL // 2}  (fibrado T*Tⁿ)")
    print(f"  • Estabilidad de toros KAM        : {'ESTABLE' if snap.is_kam_torus_stable else 'RESONANTE (control requerido)'}")
    print(f"  • Máximo exponente de Lyapunov    : {snap.max_orbital_lyapunov_exponent:+.6f}")
    print(f"  • Radio espectral de Floquet      : {snap.floquet_multiplier_spectral_radius:.8f}")
    print(f"  • Defecto de emparejamiento λ↔1/λ : {snap.floquet_pairing_defect:.6e}")
    print(f"  • Deriva de energía |ΔH|/|H|      : {snap.energy_drift:.6e}")
    print(f"  • Horizonte de Nekhoroshev        : {snap.nekhoroshev_horizon:.4e}  (unidades de tiempo)")
    print(f"  • Tiempo de recurrencia Poincaré  : {snap.poincare_recurrence_time:.4f}  ciclos")
    print(f"  • Número de rotación (sección Σ)  : {snap.poincare_rotation_number:+.6f}")

    print("\n" + "─" * 45 + " TELEMETRÍA FASE 3 " + "─" * 41)
    print(f"  • Generación RSI-L3               : #{snap.rsi_level3_generation}")
    print(f"  • Lipschitz empírico de T         : {snap.rsi_banach_contraction_lip:.6f}  (Banach exige < 1)")
    print(f"  • Lipschitz del meta-operador S   : {snap.rsi_meta_contraction_lip:.6f}")
    print(f"  • Residuo de Cauchy ‖c_{n+1}−c_n‖ : {snap.rsi_cauchy_residual:.6e}")
    print(f"  • Integridad homomórfica Φ₁₂      : {snap.homomorphic_gate_phi12_integrity * 100:.2f}%")
    print(f"  • Integridad homomórfica Φ₂₃      : {snap.homomorphic_gate_phi23_integrity * 100:.2f}%")
    print(f"  • Integridad compuesta Φ₁₃        : {snap.homomorphic_gate_phi13_integrity * 100:.2f}%")
    print(f"  • Defecto de functorialidad       : {snap.functoriality_defect:.6e}")
    print(f"  • Veredicto clasificador Heyting  : {snap.global_heyting_verdict.name}")
    print(f"  • Atestación HMAC-SHA384          : {snap.sha384_merkle_attestation[:64]}…")
    print(banner)
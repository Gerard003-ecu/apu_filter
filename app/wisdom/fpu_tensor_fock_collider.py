# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : FPU Tensor Fock Collider + Poincaré + KAM + Automejora L3                                 ║
║ Ubicación: app/wisdom/fpu_tensor_fock_collider.py                                                 ║
║ Versión  : 11.0.0-SOVEREIGN-FPU-POINCARE-KAM-L3-PhD                                                ║
║ Función  : Operatoria CAR (Jordan-Wigner exacto), CCR (escalera truncada), Dirac e⁻+e⁺→2γ,         ║
║            Lindblad-GKSL, Spohn, sección de Poincaré sobre 𝔇_n, invariante de Poincaré-Cartan,     ║
║            Birkhoff-twist, KAM diofantino proyectado, resonancias de movimiento medio y             ║
║            Automejora recursiva L3 (jerk metacognitivo + Banach + Lyapunov).                        ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.     ║
║   FASE 1 → ÁLGEBRA CORREGIDA (CAR-JW, CCR-trunc, GKSL, Dirac) + semilla canónica                   ║
║   FASE 2 → FockStateSeed + Σ Poincaré + Birkhoff + KAM + mecánica celeste + Operador L0..L3        ║
║   FASE 3 → FockTrajectory + SovereignFockAuditor + Reporte enriquecido                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.FPUTensorFockCollider")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, algebraicas y de Poincaré / KAM / L3
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ALGEBRA_RESIDUAL_TOL: Final[float] = 1.0e-10
_POSITRON_REST_MASS_KEV: Final[float] = 1022.0
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_FPU_TENSOR_DIRAC_SOVEREIGN_KEY_v11.0.0"
_DIOPHANTINE_TAU: Final[float] = 6.0          # τ ≥ n_eff − 1  (toro reducido T^{n_eff})
_KAM_MAX_K: Final[int] = 8
_KAM_EFFECTIVE_DIM: Final[int] = 6            # proyección KAM sobre T^6 ⊂ T^n
_L3_DEFAULT_ITERS: Final[int] = 6
_FERMIONIC_SUBSPACE_DIM: Final[int] = 4       # 2 modos JW → 2^2 = 4
_BOSONIC_TRUNCATION_N: Final[int] = 6         # 7-dim truncated Fock para CCR
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_RESONANCE_THRESHOLD: Final[float] = 1.0e-3
_GRAZING_SECTION_TOL: Final[float] = 1e-14
_BANACH_KAPPA_MAX: Final[float] = 1.0 - 1e-9
_LYAPUNOV_ARMJO_ITERS: Final[int] = 8
_HOMOCLINIC_HS_TOL: Final[float] = 1.0e-4


def _higham_density_projection(matrix: NDArray) -> NDArray[np.complex128]:
    r"""
    Proyección convexa de Higham (1988) sobre el simplejo de estados:
        Hermitización → recorte espectral PSD → renormalización Tr(ρ) ≡ 1.
    Es 1-Lipschitz sobre el cono PSD en norma de Frobenius; la renormalización
    preserva el simplejo compacto convexo 𝔇_n.
    """
    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("La proyección de Higham exige una matriz cuadrada.")
    herm = 0.5 * (a + a.conj().T)
    eigvals, eigvecs = la.eigh(herm)
    clean_eigs = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
    psd = (eigvecs * clean_eigs) @ eigvecs.conj().T
    tr = float(np.real(np.trace(psd)))
    n = psd.shape[0]
    if tr <= _WILKINSON_EPSILON:
        return np.eye(n, dtype=np.complex128) / float(n)
    return psd / tr


def _von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
    """S(ρ) = −Tr(ρ log ρ) sobre el soporte numérico (autovalores > piso espectral)."""
    herm = 0.5 * (rho + rho.conj().T)
    eigs = np.real(la.eigvalsh(herm))
    eigs = eigs[eigs > _SPECTRAL_PSD_FLOOR]
    if eigs.size == 0:
        return 0.0
    return float(-np.sum(eigs * np.log(eigs)))


def _purity(rho: NDArray[np.complex128]) -> float:
    """Π(ρ) = Tr(ρ²) ∈ [1/n, 1]. Invariante unitario; analogía celeste de 'energía interna'."""
    return float(np.real(np.trace(rho @ rho)))


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: ÁLGEBRA CORREGIDA (CAR-JW, CCR-trunc, GKSL, Dirac) + Report extendido                  ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La ÚLTIMA definición de FASE 1 (export_fock_state_seed) retorna un FockStateSeed, que es la    ║
# ║  PRIMERA definición consumida por FASE 2.                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

class ParticleStatisticsCategory(IntEnum):
    """Categoría estadística cuántica para las 12 partículas fundamentales."""
    FERMION_CAR = 0x01   # Pauli CAR: {a_i, a_j†} = δ_ij I
    BOSON_CCR   = 0x02   # Boson CCR: [b_i, b_j†] = δ_ij I
    ANTIMATTER  = 0x03   # Dirac: e⁻ + e⁺ ➔ 2γ (1022 keV)


@dataclass(frozen=True, slots=True)
class AnnihilationReactionReport:
    """
    Reporte inmutable de reacción de aniquilación en el espacio de Fock.
    Extendido con campos Poincaré / KAM / L3 / mecánica celeste (defaults = retrocompatibilidad).
    """
    reaction_id: str
    electron_anomalies_purged: int
    positron_auth_valid: bool
    released_energy_kev: float
    back_action_db: float
    spohn_entropy_production: float
    gksl_dissipation_rate: float
    sha256_merkle_digest: str
    timestamp_utc: float
    # ── Campos FASE 2/3 con defaults ──
    car_residual: float = 0.0
    ccr_residual: float = 0.0
    car_valid: bool = True
    ccr_valid: bool = True
    car_nilpotent_residual: float = 0.0
    lyapunov_exponent: float = 0.0
    spectral_radius: float = 1.0
    birkhoff_classification: str = "N/A"
    kam_diophantine_gamma: float = 0.0
    kam_stable: bool = True
    poincare_crossings: int = 0
    fock_winding_numbers: Tuple[float, ...] = field(default_factory=tuple)
    contraction_final: float = 0.0
    meta_learning_final: float = 0.0
    l3_certified: bool = False
    von_neumann_entropy: float = 0.0
    purity_final: float = 1.0
    poincare_rotation_number: float = 0.0
    poincare_cartan_invariant: float = 0.0
    banach_kappa: float = 1.0
    banach_certified: bool = False
    lyapunov_function_final: float = 0.0
    twist_coefficient: float = 0.0
    homoclinic_detected: bool = False
    mean_motion_resonances: int = 0
    characteristic_exponent_max: float = 0.0


class FPUTensorFockCollider:
    """
    Motor tensorial FPU sobre el Espacio de Fock ℱ(ℋ).
    Implementa CAR (Jordan-Wigner exacto de 2 modos), CCR (escalera truncada),
    Lindblad-GKSL (CPTP infinitesimal) y aniquilación de Dirac.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        self.dimension: Final[int] = max(_FERMIONIC_SUBSPACE_DIM, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa PSD de Higham y normalización Tr(ρ) ≡ 1."""
        rho = _higham_density_projection(matrix)
        if rho.shape[0] != self.dimension:
            embedded = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
            d = min(self.dimension, rho.shape[0])
            embedded[:d, :d] = rho[:d, :d]
            return _higham_density_projection(embedded)
        return rho

    # ─── 1. CONSTRUCCIÓN EXACTA DE OPERADORES CAR (JORDAN-WIGNER) ──────────────
    def _construct_fermionic_operators(self) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        [v11 CORRECCIÓN] Construye 2 operadores de aniquilación fermiónicos de Jordan-Wigner
        en la convención (modo 0 = primer factor de Kronecker):

            c_0 = σ_- ⊗ I₂
            c_1 = σ_z ⊗ σ_-

        con σ_z = diag(1, −1) y σ_- = |0⟩⟨1| en la base de ocupación {|0⟩, |1⟩}.
        El string de JW (σ_z en modos precedentes) garantiza EXACTAMENTE

            {c_i, c_j†} = δ_ij I_4 ,     {c_i, c_j} = 0 ,     c_i² = 0

        sobre el subespacio fermiónico de 4 dimensiones. Se embeben en ℂ^{dim×dim}
        con ceros fuera del subespacio físico.
        """
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)  # |0⟩⟨1|
        i2 = np.eye(2, dtype=np.complex128)

        c0_small = np.kron(sm, i2)   # 4×4  — modo 0
        c1_small = np.kron(sz, sm)   # 4×4  — modo 1 (string JW)

        c0 = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        c1 = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        c0[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM] = c0_small
        c1[:_FERMIONIC_SUBSPACE_DIM, :_FERMIONIC_SUBSPACE_DIM] = c1_small
        return c0, c1

    def verify_car_anti_commutation(
        self,
        state_a: Optional[NDArray] = None,
        state_b: Optional[NDArray] = None,
    ) -> Tuple[bool, float]:
        r"""
        Verifica {c_i, c_j†} = δ_ij I_4 sobre el subespacio fermiónico de 4 dimensiones.
        La firma `state_a, state_b` se conserva por retrocompatibilidad y NO interviene
        (los proyectores de rango 1 no realizan el álgebra CAR).
        """
        del state_a, state_b
        c0, c1 = self._construct_fermionic_operators()
        d = _FERMIONIC_SUBSPACE_DIM
        c0_s, c1_s = c0[:d, :d], c1[:d, :d]
        c0_d, c1_d = c0_s.conj().T, c1_s.conj().T
        i_d = np.eye(d, dtype=np.complex128)

        ac_00 = c0_s @ c0_d + c0_d @ c0_s
        ac_01 = c0_s @ c1_d + c1_d @ c0_s
        ac_10 = c1_s @ c0_d + c0_d @ c1_s
        ac_11 = c1_s @ c1_d + c1_d @ c1_s

        residual = max(
            float(np.linalg.norm(ac_00 - i_d, ord="fro")),
            float(np.linalg.norm(ac_01, ord="fro")),
            float(np.linalg.norm(ac_10, ord="fro")),
            float(np.linalg.norm(ac_11 - i_d, ord="fro")),
        )
        return residual < _ALGEBRA_RESIDUAL_TOL, residual

    def verify_car_nilpotency_and_mutual_anticommutator(self) -> Tuple[bool, float]:
        r"""
        Completa el álgebra CAR:
            c_i² = 0 ,   {c_i, c_j} = 0   (i ≠ j).
        Residual = máx de las normas de Frobenius de los defectos.
        """
        c0, c1 = self._construct_fermionic_operators()
        d = _FERMIONIC_SUBSPACE_DIM
        c0_s, c1_s = c0[:d, :d], c1[:d, :d]
        residual = max(
            float(np.linalg.norm(c0_s @ c0_s, ord="fro")),
            float(np.linalg.norm(c1_s @ c1_s, ord="fro")),
            float(np.linalg.norm(c0_s @ c1_s + c1_s @ c0_s, ord="fro")),
        )
        return residual < _ALGEBRA_RESIDUAL_TOL, residual

    # ─── 2. CONSTRUCCIÓN EXACTA DE OPERADORES CCR (ESCALERA TRUNCADA) ──────────
    def _construct_bosonic_operators(
        self, n_max: int = _BOSONIC_TRUNCATION_N
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""
        Escalera bosónica truncada (Fock de dimensión n_max+1):
            b|n⟩ = √n |n−1⟩          (n ≥ 1)
            b†|n⟩ = √(n+1) |n+1⟩     (n < n_max)
            b†|n_max⟩ = 0
        La relación  [b, b†] = I − (n_max+1)|n_max⟩⟨n_max|  es EXACTA (no aproximada).
        """
        d = min(n_max + 1, self.dimension)
        b_small = np.zeros((d, d), dtype=np.complex128)
        for n in range(1, d):
            b_small[n - 1, n] = np.sqrt(float(n))
        b = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        b[:d, :d] = b_small
        return b, b.conj().T

    def verify_ccr_commutation(
        self, n_max: int = _BOSONIC_TRUNCATION_N
    ) -> Tuple[bool, float]:
        r"""
        Verifica la fórmula truncada:
            [b, b†] = I_d − d |n_max⟩⟨n_max|    (d = n_max+1)
        sobre las d dimensiones de la escalera bosónica.
        """
        b, b_dag = self._construct_bosonic_operators(n_max)
        d = min(n_max + 1, self.dimension)
        comm_small = b[:d, :d] @ b_dag[:d, :d] - b_dag[:d, :d] @ b[:d, :d]

        i_d = np.eye(d, dtype=np.complex128)
        e_top = np.zeros((d, 1), dtype=np.complex128)
        e_top[d - 1, 0] = 1.0
        truncation_correction = float(d) * (e_top @ e_top.conj().T)
        expected = i_d - truncation_correction
        residual = float(np.linalg.norm(comm_small - expected, ord="fro"))
        return residual < _ALGEBRA_RESIDUAL_TOL, residual

    # ─── 3. EVOLUCIÓN GKSL Y ENTROPÍA DE SPOHN ────────────────────────────────
    def compute_lindblad_gksl_dissipation_and_spohn_entropy(
        self, density_matrix: NDArray, hamiltonian_eff: Optional[NDArray] = None
    ) -> Tuple[NDArray[np.complex128], float, float]:
        r"""
        Generador GKSL (forma de Lindblad, CPTP infinitesimal, Tr ℒ(ρ) = 0):

            ℒ(ρ) = −i[H, ρ] + 𝒟(ρ),
            𝒟(ρ) = LρL† − ½{L†L, ρ}.

        Salto físico: L = √γ c_0  (decaimento fermiónico del modo JW 0).
        Producción de entropía de Spohn respecto del estado invariante de T→∞
        (ρ_∞ = I/n, log ρ_∞ ∝ I ⇒ Tr(log ρ_∞ · ℒ(ρ)) = 0):

            σ_Spohn(ρ‖ρ_∞) = −Tr(log ρ · ℒ(ρ)) ≥ 0.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]

        if hamiltonian_eff is None:
            hamiltonian_eff = 0.1 * np.eye(dim, dtype=np.complex128)
        h = np.asarray(hamiltonian_eff, dtype=np.complex128)
        unitary = -1j * (h @ rho - rho @ h)

        c0, _ = self._construct_fermionic_operators()
        jump_l = 0.05 * c0[:dim, :dim]
        jump_l_dag = jump_l.conj().T
        dissipator = (
            jump_l @ rho @ jump_l_dag
            - 0.5 * (jump_l_dag @ jump_l @ rho + rho @ jump_l_dag @ jump_l)
        )
        d_rho = unitary + dissipator

        eigvals, eigvecs = la.eigh(0.5 * (rho + rho.conj().T))
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        log_rho = (eigvecs * log_eigs) @ eigvecs.conj().T

        spohn_raw = -float(np.real(np.trace(log_rho @ d_rho)))
        spohn_entropy = max(0.0, spohn_raw)
        gksl_rate = float(np.linalg.norm(dissipator, ord="fro"))
        return d_rho, spohn_entropy, gksl_rate

    # ─── 4. COLISIONADOR Y ANIQUILACIÓN DE DIRAC (e⁻ + e⁺ ➔ 2γ) ───────────────
    def collide_and_annihilate_dirac(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
    ) -> AnnihilationReactionReport:
        """Aniquilación e⁻ + e⁺ ➔ 2γ (1022.0 keV) previa autenticación HMAC del positrón."""
        now = time.time()
        rho = self.sanitize_density_matrix(density_matrix)
        _, spohn_entropy, gksl_rate = self.compute_lindblad_gksl_dissipation_and_spohn_entropy(rho)

        if electron_anomalies <= 0:
            rxn_id = f"DIRAC-NOMINAL-{int(now * 1000.0) % 1000000:06d}"
            seed_bytes = f"{rxn_id}:NOMINAL:0.0:{now}".encode("utf-8")
            return AnnihilationReactionReport(
                reaction_id=rxn_id,
                electron_anomalies_purged=0,
                positron_auth_valid=True,
                released_energy_kev=0.0,
                back_action_db=0.0,
                spohn_entropy_production=spohn_entropy,
                gksl_dissipation_rate=gksl_rate,
                sha256_merkle_digest=hashlib.sha256(seed_bytes).hexdigest(),
                timestamp_utc=now,
                von_neumann_entropy=_von_neumann_entropy(rho),
                purity_final=_purity(rho),
            )

        expected_pos_hmac = hmac.new(
            self.secret_key, transaction_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        auth_valid = hmac.compare_digest(positron_auth_signature, expected_pos_hmac)

        if auth_valid:
            released_kev = electron_anomalies * _POSITRON_REST_MASS_KEV
            back_action_db = 0.0
            purged = electron_anomalies
        else:
            released_kev = 0.0
            back_action_db = 12.5
            purged = 0

        rxn_id = f"DIRAC-ANNIHILATION-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{rxn_id}:{purged}:{auth_valid}:{released_kev:.2f}:{now}".encode("utf-8")
        )
        merkle_digest = hashlib.sha256(seed_bytes).hexdigest()
        return AnnihilationReactionReport(
            reaction_id=rxn_id,
            electron_anomalies_purged=purged,
            positron_auth_valid=auth_valid,
            released_energy_kev=released_kev,
            back_action_db=back_action_db,
            spohn_entropy_production=spohn_entropy,
            gksl_dissipation_rate=gksl_rate,
            sha256_merkle_digest=merkle_digest,
            timestamp_utc=now,
            von_neumann_entropy=_von_neumann_entropy(rho),
            purity_final=_purity(rho),
        )

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (export_fock_state_seed).                                         ║
    # ║  ÚLTIMO método de FASE 1. Retorna FockStateSeed → PRIMER objeto de FASE 2.                  ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def export_fock_state_seed(
        self,
        density_matrix: NDArray,
        car_ok: bool,
        car_residual: float,
        ccr_ok: bool,
        ccr_residual: float,
        spohn_entropy: float,
        gksl_rate: float,
    ) -> "FockStateSeed":
        r"""
        Comprime la auditoría algebraica (CAR/CCR/GKSL) en una semilla canónica de
        mecánica celeste de Poincaré (variables de Delaunay cuánticas):

          • J_k = eigvals(ρ) ordenadas ↓ y normalizadas     (acciones)
          • θ_k = fases del autovector dominante            (ángulos)
          • ω_k = √(1+k) + φ/(k+2)                          (frecuencias no resonantes)
          • Target: estado máximamente mixto (equilibrio térmico T→∞)
          • Probe H_probe hermitiano determinista por hash  (observable de sección Σ)

        La terna (J, θ, ω) es el análogo Fock de las variables de Delaunay (L, ℓ, n)
        del problema de Kepler: acciones, anomalías medias y movimientos medios.
        """
        rho = self.sanitize_density_matrix(density_matrix)
        dim = rho.shape[0]

        herm = 0.5 * (rho + rho.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        j_sorted = np.sort(np.maximum(np.real(eigvals), 0.0))[::-1]
        j_total = float(j_sorted.sum())
        j_norm = j_sorted / (j_total + _WILKINSON_EPSILON)

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + _GOLDEN_RATIO / (k_idx + 2.0)
        dominant_vec = eigvecs[:, -1]
        theta = np.angle(dominant_vec)

        target = np.eye(dim, dtype=np.complex128) / float(dim)

        rho_hash = hashlib.sha256(np.asarray(rho).tobytes()).hexdigest()[:8]
        seed_val = int(rho_hash, 16)
        rng = np.random.default_rng(seed_val)
        probe_raw = rng.normal(0.0, 1.0, (dim, dim)) + 1j * rng.normal(0.0, 1.0, (dim, dim))
        probe = 0.5 * (probe_raw + probe_raw.conj().T)

        return FockStateSeed(
            source_merkle=rho_hash,
            rho_vector=tuple(complex(x) for x in rho.flatten()),
            target_vector=tuple(complex(x) for x in target.flatten()),
            probe_vector=tuple(complex(x) for x in probe.flatten()),
            action_vector=tuple(float(x) for x in j_norm),
            angle_vector=tuple(float(x) for x in theta),
            frequency_vector=tuple(float(x) for x in omega),
            car_residual=car_residual,
            ccr_residual=ccr_residual,
            car_valid=car_ok,
            ccr_valid=ccr_ok,
            spohn_entropy=spohn_entropy,
            gksl_rate=gksl_rate,
            dimension=dim,
            von_neumann_entropy=_von_neumann_entropy(rho),
            purity=_purity(rho),
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: FockStateSeed + Σ Poincaré + Birkhoff + KAM + mecánica celeste + Operador L0..L3       ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (FockStateSeed) consume export_fock_state_seed (FASE 1).                         ║
# ║  Última def. (RecursiveFockImprover.iterate_level_3) → FockTrajectory (PRIMER objeto FASE 3).  ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class FockStateSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRY: continúa de export_fock_state_seed.                                        │
    │  Semilla canónica con ρ, ρ*, probe hermitiano, acciones de Delaunay J, ángulos θ,           │
    │  movimientos medios ω y residuales CAR/CCR/Spohn/GKSL.                                      │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    source_merkle: str
    rho_vector: Tuple[complex, ...]
    target_vector: Tuple[complex, ...]
    probe_vector: Tuple[complex, ...]
    action_vector: Tuple[float, ...]
    angle_vector: Tuple[float, ...]
    frequency_vector: Tuple[float, ...]
    car_residual: float
    ccr_residual: float
    car_valid: bool
    ccr_valid: bool
    spohn_entropy: float
    gksl_rate: float
    dimension: int
    von_neumann_entropy: float = 0.0
    purity: float = 1.0

    def get_rho(self) -> NDArray[np.complex128]:
        return np.array(self.rho_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_target(self) -> NDArray[np.complex128]:
        return np.array(self.target_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )

    def get_probe(self) -> NDArray[np.complex128]:
        return np.array(self.probe_vector, dtype=np.complex128).reshape(
            (self.dimension, self.dimension)
        )


class FockPoincareSection:
    r"""
    Sección transversal de Poincaré sobre el simplejo 𝔇_n:

        Σ = { ρ ∈ 𝔇_n : g(ρ) = Re⟨H_probe, ρ⟩_HS − c = 0 }.

    La transversabilidad se expresa como la derivada de Lie ℒ_X g ≠ 0, con X el
    campo discreto de automejora (analogía del campo hamiltoniano de Poincaré).
    """

    def __init__(self, probe: NDArray[np.complex128], offset: float) -> None:
        herm = 0.5 * (np.asarray(probe, dtype=np.complex128) + np.asarray(probe, dtype=np.complex128).conj().T)
        self.probe = herm
        self.offset = float(offset)

    def g(self, rho: NDArray[np.complex128]) -> float:
        return float(np.real(np.trace(self.probe @ rho))) - self.offset

    def lie_derivative(self, d_rho: NDArray[np.complex128]) -> float:
        r"""ℒ_X g = dg(X) = Re Tr(H_probe · dρ)."""
        return float(np.real(np.trace(self.probe @ d_rho)))

    def transversality(self, d_rho: NDArray[np.complex128], tol: float = 1e-10) -> bool:
        return abs(self.lie_derivative(d_rho)) > tol


class FockReturnMap:
    """
    Mapa de retorno P: Σ → Σ de Poincaré.
    Interpolación lineal en el cruce de signo de g, filtrado de órbitas rasantes
    y re-proyección de Higham para permanecer en 𝔇_n ∩ Σ.
    """

    def __init__(self, section: FockPoincareSection) -> None:
        self.section = section

    def apply(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        crossings: List[NDArray[np.complex128]] = []
        if len(rho_sequence) < 2:
            return crossings
        g_vals = [self.section.g(r) for r in rho_sequence]
        for i in range(len(g_vals) - 1):
            gi, gj = g_vals[i], g_vals[i + 1]
            if gi * gj >= 0.0:
                continue
            denom = gi - gj
            if abs(denom) < _GRAZING_SECTION_TOL:
                continue
            t = gi / denom
            if t < 0.0 or t > 1.0:
                continue
            d_rho = rho_sequence[i + 1] - rho_sequence[i]
            if not self.section.transversality(d_rho, tol=_GRAZING_SECTION_TOL):
                continue
            rho_cross = (1.0 - t) * rho_sequence[i] + t * rho_sequence[i + 1]
            crossings.append(_higham_density_projection(rho_cross))
        return crossings

    def rotation_number(self, angle_sequence: Sequence[float]) -> float:
        r"""
        Número de rotación de Poincaré:
            ρ_rot = (θ_T − θ_0) / (2π T) ∈ ℝ/ℤ  (levantamiento continuo no exigido;
            se usa la diferencia neta de la fase de sección).
        """
        if len(angle_sequence) < 2:
            return 0.0
        delta = float(angle_sequence[-1] - angle_sequence[0])
        periods = float(len(angle_sequence) - 1)
        return delta / (2.0 * math.pi * periods)


class BirkhoffFockForm:
    r"""
    Forma normal de Birkhoff en el punto fijo del ciclo L3 sobre 𝔇_n.
    Clasifica el espectro del jacobiano discreto y estima el twist ∂ω/∂J
    (condición de Poincaré-Birkhoff para existencia de órbitas periódicas).
    """

    def __init__(
        self,
        fixed_rho: NDArray[np.complex128],
        jacobian_spectrum: NDArray[np.complex128],
        twist_coefficient: float = 0.0,
    ) -> None:
        self.fixed_rho = fixed_rho
        self.jacobian_spectrum = jacobian_spectrum
        self.twist_coefficient = float(twist_coefficient)

    def classify(self) -> str:
        eigs = self.jacobian_spectrum
        if eigs.size == 0:
            return "DEGENERATE"
        moduli = np.abs(eigs)
        if np.all(moduli < 1.0 - 1e-8):
            return "ATTRACTING"
        if np.all(moduli > 1.0 + 1e-8):
            return "REPELLING"
        if np.all(np.abs(moduli - 1.0) < 1e-8):
            return "ELLIPTIC_CENTER"
        return "SADDLE_HYPERBOLIC"

    def spectral_radius(self) -> float:
        if self.jacobian_spectrum.size == 0:
            return 0.0
        return float(np.max(np.abs(self.jacobian_spectrum)))

    def lyapunov_exponent(self) -> float:
        radius = self.spectral_radius()
        return float(np.log(max(radius, _WILKINSON_EPSILON)))

    def characteristic_exponents(self) -> Tuple[float, ...]:
        """Exponentes característicos de Poincaré: χ_i = log|λ_i| (Floquet)."""
        if self.jacobian_spectrum.size == 0:
            return (0.0,)
        return tuple(float(np.log(max(abs(z), _WILKINSON_EPSILON))) for z in self.jacobian_spectrum)

    def poincare_birkhoff_twist_holds(self, tol: float = 1e-12) -> bool:
        """Condición de twist: ω'(J) ≠ 0 ⇒ el mapa de anillo admite órbitas p/q."""
        return abs(self.twist_coefficient) > tol


class KAMFockTorus:
    r"""
    Toro invariante T^{n_eff} de robustez disipativa (condición diofantina proyectada):

        |⟨k, ω⟩| ≥ γ / |k|_1^τ    ∀ k ∈ ℤ^{n_eff}\{0}  con  ‖k‖_0 ≤ 2, τ ≥ n_eff−1.

    Se auditan resonancias de movimiento medio de 1 y 2 frecuencias — el núcleo
    de la mecánica celeste de Poincaré (small divisors del problema de 3 cuerpos).
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: Optional[float] = None
    ) -> float:
        n_eff = min(self.n, _KAM_EFFECTIVE_DIM)
        if n_eff == 0:
            return 0.0
        omega = np.asarray(self.frequencies[:n_eff], dtype=np.float64)
        tau_eff = float(_DIOPHANTINE_TAU if tau is None else tau)
        gamma = math.inf

        for i in range(n_eff):
            for k in range(1, max_k + 1):
                inner = abs(k * omega[i])
                denom = float(k ** tau_eff)
                if denom > 0.0:
                    gamma = min(gamma, inner * denom)

        for i in range(n_eff):
            for j in range(i + 1, n_eff):
                for p in range(-max_k, max_k + 1):
                    for q in range(-max_k, max_k + 1):
                        if p == 0 and q == 0:
                            continue
                        inner = abs(p * omega[i] + q * omega[j])
                        l1 = float(abs(p) + abs(q))
                        denom = l1 ** tau_eff
                        if denom > 0.0:
                            gamma = min(gamma, inner * denom)

        if not math.isfinite(gamma):
            return 0.0
        return float(gamma)

    def is_kam_stable(
        self, perturbation_norm: float, gamma: Optional[float] = None
    ) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        return gamma > perturbation_norm

    def mean_motion_resonance_count(
        self,
        max_order: int = 5,
        threshold: float = _RESONANCE_THRESHOLD,
    ) -> int:
        r"""
        Cuenta resonancias de movimiento medio p ω_i − q ω_j ≈ 0
        (analogía de las conmensurabilidades de Poincaré en el problema lunar).
        """
        n_eff = min(self.n, _KAM_EFFECTIVE_DIM)
        omega = np.asarray(self.frequencies[:n_eff], dtype=np.float64)
        count = 0
        for i in range(n_eff):
            for j in range(i + 1, n_eff):
                hit = False
                for p in range(1, max_order + 1):
                    for q in range(1, max_order + 1):
                        divisor = abs(p * omega[i] - q * omega[j]) / float(p + q)
                        if divisor < threshold:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    count += 1
        return count

    def winding_numbers(self, phases: Sequence[float]) -> Tuple[float, ...]:
        """Números de enrollamiento topológico ω_k / Δθ sobre la fase de sección."""
        n_eff = min(self.n, 4)
        if len(phases) < 2:
            return tuple(0.0 for _ in range(n_eff))
        delta_phi = float(phases[-1] - phases[0])
        if abs(delta_phi) < _WILKINSON_EPSILON:
            return tuple(0.0 for _ in range(n_eff))
        return tuple(float(w / delta_phi) for w in self.frequencies[:n_eff])


class PoincareCelestialMechanics:
    r"""
    Mecánica celeste de Poincaré sobre la órbita coadjunta de Fock.

    • Invariante integral de Poincaré-Cartan:  ∮ ∑ J_k dθ_k
    • Función generatriz de tipo II:           S(θ, J') = ⟨θ, J'⟩
    • Twist de Birkhoff:                       ∂ω_eff / ∂J  (diferencias finitas)
    • Exponentes característicos:              χ = log|spec(DP)|
    • Heurística homoclínica:                  cruces próximos + λ > 0
    """

    @staticmethod
    def poincare_cartan_invariant(
        action_sequence: Sequence[float],
        angle_sequence: Sequence[float],
    ) -> float:
        r"""
        Discretización del invariante de Poincaré-Cartan:
            I_PC = ∑_t J_t (θ_{t+1} − θ_t).
        En un flujo hamiltoniano exacto I_PC es independiente de la sección;
        aquí certifica la 'área' barrida en el plano de Delaunay (J, θ).
        """
        n = min(len(action_sequence), len(angle_sequence))
        if n < 2:
            return 0.0
        acc = 0.0
        for t in range(n - 1):
            acc += float(action_sequence[t]) * float(angle_sequence[t + 1] - angle_sequence[t])
        return float(acc)

    @staticmethod
    def generating_function_type2(
        angles: Sequence[float],
        new_actions: Sequence[float],
    ) -> float:
        """S(θ, J') = ⟨θ, J'⟩  (identidad canónica; germen de perturbación KAM)."""
        n = min(len(angles), len(new_actions))
        return float(sum(float(angles[i]) * float(new_actions[i]) for i in range(n)))

    @staticmethod
    def twist_coefficient(
        leading_actions: Sequence[float],
        angle_sequence: Sequence[float],
    ) -> float:
        r"""
        Twist discreto ω'(J) ≈ Δ(Δθ) / ΔJ sobre la acción dominante
        (autovalor máximo de ρ, analogía del semieje mayor de Kepler).
        """
        n = min(len(leading_actions), len(angle_sequence))
        if n < 3:
            return 0.0
        d_theta = [
            float(angle_sequence[t + 1] - angle_sequence[t]) for t in range(n - 1)
        ]
        twists: List[float] = []
        for t in range(len(d_theta) - 1):
            d_j = float(leading_actions[t + 1] - leading_actions[t])
            if abs(d_j) > _WILKINSON_EPSILON:
                twists.append((d_theta[t + 1] - d_theta[t]) / d_j)
        if not twists:
            return 0.0
        return float(np.mean(np.asarray(twists, dtype=np.float64)))

    @staticmethod
    def homoclinic_signature(
        crossings: Sequence[NDArray[np.complex128]],
        lyapunov: float,
        hs_tol: float = _HOMOCLINIC_HS_TOL,
    ) -> bool:
        """
        Firma homoclínica de Poincaré: dos cruces no consecutivos próximos en
        norma HS y estiramiento hiperbólico (λ > 0). Detecta gérmenes de trama
        homoclínica (no una prueba topológica completa).
        """
        if lyapunov <= 0.0 or len(crossings) < 3:
            return False
        for i in range(len(crossings)):
            for j in range(i + 2, len(crossings)):
                dist = float(np.linalg.norm(crossings[i] - crossings[j], ord="fro"))
                if dist < hs_tol:
                    return True
        return False


class RecursiveFockImprover:
    r"""
    Operador de automejora recursiva I: 𝔇_n → 𝔇_n  (endofunctor del simplejo).

      L0: identidad
      L1: I_1(ρ) = Higham((1−θ)ρ + θρ*)  con backtracking de Lyapunov
          V(ρ) = ‖ρ − ρ*‖_F²  (descenso certificado tipo Armijo)
      L2: θ ← clip(θ · (1 + 0.1(ratio_err − 1)))   (adaptación por residuo)
      L3: metacognición de la adaptación:
            • η_meta ← η_meta · {0.90 / 1.05} según ρ_espectral(DI)
            • jerk de θ (tercera diferencia finita) amortigua η si la ley de
              aprendizaje acelera de forma inestable
            • θ ← clip(θ + 0.05(η_meta − 1/2))

    Certificación Banach: κ̂ = máx_t ‖ρ_{t+1}−ρ*‖ / ‖ρ_t−ρ*‖  < 1
    implica punto fijo único de I en el completo (𝔇_n, ‖·‖_F).
    """

    def __init__(self, contraction: float = 0.5, meta_learning: float = 0.3) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self._contraction_history: List[float] = []
        self._meta_history: List[float] = []

    def apply_level_0(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho.copy()

    def apply_level_1(
        self, rho: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        v0 = self._lyapunov_function(rho, target)
        theta = float(self.contraction)
        candidate = rho
        for _ in range(_LYAPUNOV_ARMJO_ITERS):
            mixed = (1.0 - theta) * rho + theta * target
            candidate = _higham_density_projection(mixed)
            v1 = self._lyapunov_function(candidate, target)
            if v1 <= v0 * (1.0 - 1e-12) or theta < 0.06:
                return candidate
            theta *= 0.5
        return candidate

    def apply_level_2(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(history) >= 2:
            err = float(np.linalg.norm(rho - target, ord="fro"))
            prev = float(np.linalg.norm(history[-2] - target, ord="fro"))
            if prev > _WILKINSON_EPSILON:
                ratio = err / prev
                self.contraction = float(
                    np.clip(self.contraction * (1.0 + 0.1 * (ratio - 1.0)), 0.05, 0.95)
                )
        return self.apply_level_1(rho, target)

    def apply_level_3(
        self,
        rho: NDArray[np.complex128],
        target: NDArray[np.complex128],
        history: List[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        if len(history) >= 3:
            rho_spec = self._estimate_local_spectral_radius(history[-3:])
            if rho_spec > 1.0:
                self.meta_learning = float(np.clip(self.meta_learning * 0.90, 0.05, 0.90))
            elif 0.0 < rho_spec < 0.5:
                self.meta_learning = float(np.clip(self.meta_learning * 1.05, 0.05, 0.90))
            self.contraction = float(np.clip(
                self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95
            ))

        # Jerk metacognitivo: tercera diferencia finita de θ (L3 genuino).
        hist = self._contraction_history
        if len(hist) >= 3:
            jerk = hist[-1] - 2.0 * hist[-2] + hist[-3]
            if abs(jerk) > 0.05:
                self.meta_learning = float(np.clip(self.meta_learning * 0.92, 0.05, 0.90))

        evolved = self.apply_level_2(rho, target, history)
        self._contraction_history.append(float(self.contraction))
        self._meta_history.append(float(self.meta_learning))
        return evolved

    @staticmethod
    def _estimate_local_spectral_radius(recent: Sequence[NDArray[np.complex128]]) -> float:
        if len(recent) < 3:
            return 0.5
        d1 = float(np.linalg.norm(recent[-1] - recent[-2], ord="fro"))
        d2 = float(np.linalg.norm(recent[-2] - recent[-3], ord="fro"))
        if d1 < _WILKINSON_EPSILON or d2 < _WILKINSON_EPSILON:
            return 0.5
        return d1 / d2

    @staticmethod
    def _lyapunov_function(
        rho: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> float:
        """V(ρ) = ‖ρ − ρ*‖_F².  Definita positiva, C^∞, mínimo único en ρ*."""
        delta = rho - target
        return float(np.real(np.vdot(delta.ravel(), delta.ravel())))

    @staticmethod
    def _banach_kappa(
        rho_sequence: Sequence[NDArray[np.complex128]],
        target: NDArray[np.complex128],
    ) -> float:
        ratios: List[float] = []
        for t in range(len(rho_sequence) - 1):
            d0 = float(np.linalg.norm(rho_sequence[t] - target, ord="fro"))
            d1 = float(np.linalg.norm(rho_sequence[t + 1] - target, ord="fro"))
            if d0 > _WILKINSON_EPSILON:
                ratios.append(d1 / d0)
        if not ratios:
            return 1.0
        return float(max(ratios))

    @staticmethod
    def _serialize(rho: NDArray[np.complex128]) -> Tuple[complex, ...]:
        return tuple(complex(x) for x in rho.flatten())

    @staticmethod
    def deserialize(vec: Sequence[complex], dim: int) -> NDArray[np.complex128]:
        return np.array(vec, dtype=np.complex128).reshape((dim, dim))

    @staticmethod
    def _estimate_global_jacobian_spectrum(
        rho_sequence: Sequence[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """
        Espectro de monodromía reducido: cocientes de incrementos HS sucesivos,
        proyección al fibrado tangente (hermítico de traza nula) vía Δρ.
        """
        if len(rho_sequence) < 2:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        multipliers: List[complex] = []
        for t in range(1, len(rho_sequence) - 1):
            d_prev = rho_sequence[t] - rho_sequence[t - 1]
            d_next = rho_sequence[t + 1] - rho_sequence[t]
            n_prev = float(np.linalg.norm(d_prev, ord="fro"))
            if n_prev > _WILKINSON_EPSILON:
                multipliers.append(complex(float(np.linalg.norm(d_next, ord="fro")) / n_prev, 0.0))
        if not multipliers:
            d_last = float(np.linalg.norm(rho_sequence[-1] - rho_sequence[-2], ord="fro"))
            return np.array([complex(d_last, 0.0)], dtype=np.complex128)
        return np.asarray(multipliers, dtype=np.complex128)

    @staticmethod
    def _leading_action(rho: NDArray[np.complex128]) -> float:
        eigs = np.real(la.eigvalsh(0.5 * (rho + rho.conj().T)))
        return float(np.max(eigs)) if eigs.size else 0.0

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (iterate_level_3).                                                ║
    # ║  ÚLTIMO método de FASE 2. Retorna FockTrajectory → PRIMER objeto de FASE 3.                 ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: FockStateSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> "FockTrajectory":
        r"""
        Itera I_L3 sobre ρ_seed con target ρ*. Devuelve la trayectoria Fock con
        estructura de mecánica celeste de Poincaré:

          • Secuencia ρ_0 → ρ_T  (Tr ≡ 1, PSD)
          • Acciones J_t = Tr(ρ_t) ≡ 1  y acción dominante λ_max(ρ_t)
          • Ángulos θ_t = arg Tr(H_probe ρ_t)  (fase de sección)
          • λ de Lyapunov (log-ratio de ‖Δρ‖_F)
          • γ diofantina, winding, twist, invariante de Poincaré-Cartan
          • Certificación Banach κ̂ < 1 y V(ρ_T)
          • Clasificación de Birkhoff del punto fijo final
        """
        dim = seed.dimension
        rho = seed.get_rho()
        target = seed.get_target()
        probe = seed.get_probe()

        rho_sequence: List[NDArray[np.complex128]] = [rho.copy()]
        action_sequence: List[float] = [float(np.real(np.trace(rho)))]
        angle_sequence: List[float] = [float(np.angle(np.trace(probe @ rho)))]
        leading_actions: List[float] = [self._leading_action(rho)]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max_iters):
            history.append(rho.copy())
            rho = self.apply_level_3(rho, target, history)
            rho_sequence.append(rho.copy())
            action_sequence.append(float(np.real(np.trace(rho))))
            angle_sequence.append(float(np.angle(np.trace(probe @ rho))))
            leading_actions.append(self._leading_action(rho))

        diffs = [
            float(np.linalg.norm(rho_sequence[i + 1] - rho_sequence[i], ord="fro"))
            for i in range(len(rho_sequence) - 1)
        ]
        if len(diffs) >= 2 and all(d > _WILKINSON_EPSILON for d in diffs):
            log_ratios = np.log(np.array(diffs[1:]) / np.array(diffs[:-1]))
            lyapunov = float(np.mean(log_ratios))
        else:
            lyapunov = -20.0
        convergence = diffs[-1] if diffs else 0.0

        torus = KAMFockTorus(seed.frequency_vector)
        gamma = torus.diophantine_constant()
        kam_stable = torus.is_kam_stable(convergence, gamma)
        winding = torus.winding_numbers(angle_sequence)
        mmr = torus.mean_motion_resonance_count()

        jacobian_spectrum = self._estimate_global_jacobian_spectrum(rho_sequence)
        twist = PoincareCelestialMechanics.twist_coefficient(leading_actions, angle_sequence)
        birkhoff = BirkhoffFockForm(rho_sequence[-1], jacobian_spectrum, twist_coefficient=twist)
        birkhoff_class = birkhoff.classify()
        char_exps = birkhoff.characteristic_exponents()

        kappa = self._banach_kappa(rho_sequence, target)
        v_final = self._lyapunov_function(rho_sequence[-1], target)
        cartan = PoincareCelestialMechanics.poincare_cartan_invariant(
            leading_actions, angle_sequence
        )
        rot_num = FockReturnMap(FockPoincareSection(probe, 0.0)).rotation_number(angle_sequence)

        spectral_radius = (
            float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0
        )

        return FockTrajectory(
            source_merkle=seed.source_merkle,
            dimension=dim,
            rho_sequence=tuple(self._serialize(r) for r in rho_sequence),
            action_sequence=tuple(action_sequence),
            angle_sequence=tuple(angle_sequence),
            leading_action_sequence=tuple(leading_actions),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            spectral_radius=spectral_radius,
            is_kam_stable=kam_stable,
            winding_numbers=winding,
            diophantine_gamma=gamma,
            final_contraction=self.contraction,
            final_meta_learning=self.meta_learning,
            birkhoff_classification=birkhoff_class,
            von_neumann_entropy_final=_von_neumann_entropy(rho_sequence[-1]),
            purity_final=_purity(rho_sequence[-1]),
            poincare_rotation_number=rot_num,
            poincare_cartan_invariant=cartan,
            banach_kappa=kappa,
            banach_certified=bool(kappa < _BANACH_KAPPA_MAX),
            lyapunov_function_final=v_final,
            twist_coefficient=twist,
            mean_motion_resonances=mmr,
            characteristic_exponent_max=max(char_exps) if char_exps else 0.0,
        )


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: FockTrajectory + SovereignFockAuditor                                                  ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (FockTrajectory) consume iterate_level_3 de FASE 2.                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class FockTrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRY: continúa de iterate_level_3 (FASE 2).                                      │
    │  Trayectoria L3 del ciclo Fock con Poincaré + Birkhoff + KAM + Delaunay embebidos.          │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    source_merkle: str
    dimension: int
    rho_sequence: Tuple[Tuple[complex, ...], ...]
    action_sequence: Tuple[float, ...]
    angle_sequence: Tuple[float, ...]
    lyapunov_exponent: float
    convergence_residual: float
    spectral_radius: float
    is_kam_stable: bool
    winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    final_contraction: float
    final_meta_learning: float
    birkhoff_classification: str
    leading_action_sequence: Tuple[float, ...] = field(default_factory=tuple)
    von_neumann_entropy_final: float = 0.0
    purity_final: float = 1.0
    poincare_rotation_number: float = 0.0
    poincare_cartan_invariant: float = 0.0
    banach_kappa: float = 1.0
    banach_certified: bool = False
    lyapunov_function_final: float = 0.0
    twist_coefficient: float = 0.0
    mean_motion_resonances: int = 0
    characteristic_exponent_max: float = 0.0

    def get_rho_sequence_arrays(self) -> List[NDArray[np.complex128]]:
        return [
            RecursiveFockImprover.deserialize(s, self.dimension) for s in self.rho_sequence
        ]

    def build_poincare_section(
        self, probe: NDArray[np.complex128]
    ) -> FockPoincareSection:
        rho_arr = self.get_rho_sequence_arrays()
        herm_probe = 0.5 * (probe + probe.conj().T)
        g_vals = [float(np.real(np.trace(herm_probe @ r))) for r in rho_arr]
        offset = float(np.mean(g_vals)) if g_vals else 0.0
        return FockPoincareSection(probe=herm_probe, offset=offset)

    def is_unstable_fock(self) -> bool:
        return (
            self.lyapunov_exponent > 0.0
            or self.birkhoff_classification in ("REPELLING", "SADDLE_HYPERBOLIC")
        )

    def is_meta_stable_fock(self) -> bool:
        return (
            self.lyapunov_exponent < 0.0
            and self.is_kam_stable
            and self.birkhoff_classification in ("ATTRACTING", "ELLIPTIC_CENTER")
            and self.banach_certified
        )


class SovereignFockAuditor:
    """
    Auditor soberano que orquesta:
      1) Auditoría algebraica CAR/CCR (+ nilpotencia)
      2) Aniquilación de Dirac base
      3) Semilla canónica (FASE 1 → FASE 2)
      4) Trayectoria L3 (FASE 2 → FASE 3)
      5) Poincaré crossings + firma homoclínica
      6) Reporte enriquecido (AnnihilationReactionReport con campos L3 / celestes)
    """

    def __init__(
        self,
        collider: FPUTensorFockCollider,
        improver: Optional[RecursiveFockImprover] = None,
    ) -> None:
        self.collider = collider
        self.improver = improver or RecursiveFockImprover(
            contraction=0.5, meta_learning=0.3
        )

    def full_audit(
        self,
        electron_anomalies: int,
        positron_auth_signature: str,
        transaction_payload: str,
        density_matrix: NDArray,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> AnnihilationReactionReport:
        # (1) Auditoría algebraica
        car_ok, car_res = self.collider.verify_car_anti_commutation()
        ccr_ok, ccr_res = self.collider.verify_ccr_commutation()
        _, car_nil = self.collider.verify_car_nilpotency_and_mutual_anticommutator()

        # (2) Aniquilación base
        base_report = self.collider.collide_and_annihilate_dirac(
            electron_anomalies, positron_auth_signature, transaction_payload, density_matrix
        )

        # (3) Semilla FASE 1 → FASE 2
        rho_sane = self.collider.sanitize_density_matrix(density_matrix)
        seed = self.collider.export_fock_state_seed(
            rho_sane,
            car_ok,
            car_res,
            ccr_ok,
            ccr_res,
            base_report.spohn_entropy_production,
            base_report.gksl_dissipation_rate,
        )

        # (4) Trayectoria L3 FASE 2 → FASE 3
        traj = self.improver.iterate_level_3(seed, max_iters=max_iters)

        # (5) Poincaré crossings + homoclínica
        section = traj.build_poincare_section(seed.get_probe())
        prm = FockReturnMap(section)
        crossings = prm.apply(traj.get_rho_sequence_arrays())
        homoclinic = PoincareCelestialMechanics.homoclinic_signature(
            crossings, traj.lyapunov_exponent
        )

        # (6) Reporte enriquecido
        l3_cert = (
            traj.is_meta_stable_fock()
            and car_ok
            and ccr_ok
            and base_report.positron_auth_valid
            and not homoclinic
        )
        return replace(
            base_report,
            car_residual=car_res,
            ccr_residual=ccr_res,
            car_valid=car_ok,
            ccr_valid=ccr_ok,
            car_nilpotent_residual=car_nil,
            lyapunov_exponent=traj.lyapunov_exponent,
            spectral_radius=traj.spectral_radius,
            birkhoff_classification=traj.birkhoff_classification,
            kam_diophantine_gamma=traj.diophantine_gamma,
            kam_stable=traj.is_kam_stable,
            poincare_crossings=len(crossings),
            fock_winding_numbers=traj.winding_numbers[:4],
            contraction_final=traj.final_contraction,
            meta_learning_final=traj.final_meta_learning,
            l3_certified=l3_cert,
            von_neumann_entropy=traj.von_neumann_entropy_final,
            purity_final=traj.purity_final,
            poincare_rotation_number=traj.poincare_rotation_number,
            poincare_cartan_invariant=traj.poincare_cartan_invariant,
            banach_kappa=traj.banach_kappa,
            banach_certified=traj.banach_certified,
            lyapunov_function_final=traj.lyapunov_function_final,
            twist_coefficient=traj.twist_coefficient,
            homoclinic_detected=homoclinic,
            mean_motion_resonances=traj.mean_motion_resonances,
            characteristic_exponent_max=traj.characteristic_exponent_max,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 100
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — v11.0.0-SOVEREIGN-FPU-POINCARE-KAM-L3-PhD")
    print("  CAR-JW exacto + CCR-trunc + Dirac + GKSL + Spohn + Poincaré-Cartan + KAM + L3")
    print(banner)

    collider = FPUTensorFockCollider(dimension=56)
    auditor = SovereignFockAuditor(collider)
    rng = np.random.default_rng(2026)

    state_a = rng.normal(0.0, 1.0, size=(56, 1)) + 1j * rng.normal(0.0, 1.0, size=(56, 1))
    density_sample = state_a @ state_a.conj().T
    density_sample /= np.trace(density_sample)

    car_ok, car_res = collider.verify_car_anti_commutation()
    ccr_ok, ccr_res = collider.verify_ccr_commutation()
    nil_ok, nil_res = collider.verify_car_nilpotency_and_mutual_anticommutator()

    print(f"\n  ─── FASE 1: Auditoría Algebraica (JW EXACTO) ─────────────────────")
    print(f"  • CAR {{c_i, c_j†}} = δ_ij I (2 modos, 4-dim) : Valid = {car_ok} (residual = {car_res:.3e})")
    print(f"  • CAR c_i² = 0, {{c_0, c_1}} = 0              : Valid = {nil_ok} (residual = {nil_res:.3e})")
    print(f"  • CCR (escalera truncada, N=6 → 7-dim)        : Valid = {ccr_ok} (residual = {ccr_res:.3e})")

    d_rho, spohn_s, gksl_r = collider.compute_lindblad_gksl_dissipation_and_spohn_entropy(
        density_sample
    )
    print(f"  • Spohn σ_Spohn(ρ‖I/n)                        : {spohn_s:.6e}")
    print(f"  • GKSL rate ‖𝒟(ρ)‖_F                         : {gksl_r:.6f}")
    print(f"  • von Neumann S(ρ)                            : {_von_neumann_entropy(collider.sanitize_density_matrix(density_sample)):.6f}")

    payload = "TRANSACTION-APU-PRECIO-SOBRECOSTO-001"
    valid_hmac = hmac.new(
        collider.secret_key, payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    report = auditor.full_audit(
        electron_anomalies=2,
        positron_auth_signature=valid_hmac,
        transaction_payload=payload,
        density_matrix=density_sample,
    )

    print(f"\n  ─── FASE 2 → FASE 3: Poincaré / Birkhoff / KAM / Delaunay / L3 ──")
    print(f"  • Reacción                                    : {report.reaction_id}")
    print(f"  • Electrones Purgados (e⁻)                    : {report.electron_anomalies_purged}")
    print(f"  • Positrón Criptográfico (e⁺)                 : Válido = {report.positron_auth_valid}")
    print(f"  • Energía Liberada (2γ)                       : {report.released_energy_kev:.1f} keV")
    print(f"  • Retroacción en RAM                          : {report.back_action_db:.1f} dB")
    print(f"  • CAR residual / nilpotente                   : {report.car_residual:.3e} / {report.car_nilpotent_residual:.3e}")
    print(f"  • CCR residual                                : {report.ccr_residual:.3e}")
    print(f"  • Lyapunov λ                                  : {report.lyapunov_exponent:+.6f}")
    print(f"  • Radio Espectral ρ(DI)                       : {report.spectral_radius:.6f}")
    print(f"  • Exponente característico máx (Poincaré)     : {report.characteristic_exponent_max:+.6f}")
    print(f"  • Birkhoff Class                              : {report.birkhoff_classification}")
    print(f"  • Twist ∂ω/∂J                                 : {report.twist_coefficient:+.6e}")
    print(f"  • KAM γ Diofantina                            : {report.kam_diophantine_gamma:.6e}")
    print(f"  • KAM Estable                                 : {report.kam_stable}")
    print(f"  • Resonancias de movimiento medio             : {report.mean_motion_resonances}")
    print(f"  • Cruces Poincaré Σ                           : {report.poincare_crossings}")
    print(f"  • Número de rotación ρ_rot                    : {report.poincare_rotation_number:+.6f}")
    print(f"  • Invariante Poincaré-Cartan                  : {report.poincare_cartan_invariant:+.6e}")
    print(f"  • Homoclínica detectada                       : {report.homoclinic_detected}")
    w = report.fock_winding_numbers
    w0 = w[0] if len(w) > 0 else 0.0
    w1 = w[1] if len(w) > 1 else 0.0
    w2 = w[2] if len(w) > 2 else 0.0
    w3 = w[3] if len(w) > 3 else 0.0
    print(f"  • Enrollamiento ω (4)                         : ({w0:.6f}, {w1:.6f}, {w2:.6f}, {w3:.6f})")
    print(f"  • θ_final (contracción)                       : {report.contraction_final:.6f}")
    print(f"  • η_meta_final (L3)                           : {report.meta_learning_final:.6f}")
    print(f"  • Banach κ̂ / certificado                     : {report.banach_kappa:.6f} / {report.banach_certified}")
    print(f"  • V(ρ_T) Lyapunov                             : {report.lyapunov_function_final:.6e}")
    print(f"  • S(ρ_T) / Π(ρ_T)                             : {report.von_neumann_entropy:.6f} / {report.purity_final:.6f}")
    print(f"  • L3 Certificado                              : {report.l3_certified}")
    print(f"  • Raíz Merkle SHA-256                         : {report.sha256_merkle_digest[:32]}...")

    print(f"\n  ─── Test de firma INVÁLIDA (defensa de la aniquilación) ────────")
    bad_report = auditor.full_audit(
        electron_anomalies=2,
        positron_auth_signature="BAD_SIGNATURE",
        transaction_payload=payload,
        density_matrix=density_sample,
    )
    print(f"  • Auth válido                                 : {bad_report.positron_auth_valid}")
    print(f"  • Electrones purgados                         : {bad_report.electron_anomalies_purged}")
    print(f"  • Retroacción                                 : {bad_report.back_action_db:.1f} dB")
    print(f"  • L3 Certificado                              : {bad_report.l3_certified}")

    print(f"\n  ─── Trayectoria L3 explícita (ρ_t, J_t, λ_max, θ_t) ────────────")
    seed_dbg = collider.export_fock_state_seed(
        density_sample, car_ok, car_res, ccr_ok, ccr_res, spohn_s, gksl_r
    )
    improver = RecursiveFockImprover()
    traj = improver.iterate_level_3(seed_dbg, max_iters=6)
    for t, (j_tr, ang) in enumerate(zip(traj.action_sequence, traj.angle_sequence)):
        lam = traj.leading_action_sequence[t] if t < len(traj.leading_action_sequence) else 0.0
        print(f"    t={t:02d} | Tr(ρ) = {j_tr:.8f} | λ_max = {lam:.8f} | θ_t = {ang:+.6f} rad")
    print(
        f"    → λ = {traj.lyapunov_exponent:+.6f} | "
        f"KAM γ = {traj.diophantine_gamma:.6e} | "
        f"Birkhoff = {traj.birkhoff_classification} | "
        f"κ̂ = {traj.banach_kappa:.4f} | "
        f"I_PC = {traj.poincare_cartan_invariant:+.3e}"
    )

    print("\n" + banner)
    print("  ✓ FASE 1 → FASE 2: export_fock_state_seed → FockStateSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveFockImprover.iterate_level_3 → FockTrajectory")
    print("  ✓ CAR corregido: c_0 = σ_-⊗I , c_1 = σ_z⊗σ_-  (anticommutación exacta)")
    print("  ✓ CCR truncado con [b,b†] = I − (N+1)|N⟩⟨N|  (verificación algebraica real)")
    print("  ✓ Poincaré-Cartan + twist de Birkhoff + resonancias de movimiento medio")
    print("  ✓ Automejora L3: Armijo-Lyapunov + jerk metacognitivo + certificado Banach")
    print(banner)
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : De Rham-Galois Adjunction Bridge + Poincaré + KAM/BHT + Automejora L3                 ║
║ Ubicación: app/wisdom/de_rham_galois_adjunction_bridge.py                                        ║
║ Versión  : 11.0.0-SOVEREIGN-DERHAM-GALOIS-POINCARE-KAM-BHT-L3-PhD                                ║
║ Función  : Isomorfismo categórico F ⊣ G (Hom_𝒟(F(X),Y) ≅ Hom_𝒞(X,G(Y))) enriquecido con         ║
║            complejo de de Rham matricial, holonomía de Pancharatnam–Berry, mecánica celeste      ║
║            de Poincaré, tori KAM diofantinos (criterio BHT disipativo) y operador de             ║
║            automejora recursiva Nivel 3 (metacognición metacognitiva sobre el ciclo de           ║
║            adjunción, realimentada por el mapa de retorno).                                      ║
║ Estructura: 3 Fases Anidadas — la ÚLTIMA def. de fase n es la FRONTERA de la PRIMERA def. n+1.   ║
║   FASE 1 → Fundación Categórica (F, G, η, ε, Δ_△, A_μ, F_dR, ρ_MAC) + Uhlmann + Report          ║
║   FASE 2 → Semilla Canónica + Poincaré Σ + Twist P-B + KAM/BHT + Operador L3                     ║
║   FASE 3 → Trayectoria + Auditor Soberano + Reporte Enriquecido                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ── Importación resiliente de la base inmutable de partículas TOON (v11 / v10 / v9 / fallback)
_TOON_AVAILABLE: bool
_TOON_V10: bool
try:
    from toon_cartridge_base import ToonCartridge, ToonParticleFactory  # type: ignore

    _TOON_AVAILABLE = True
    _TOON_V10 = hasattr(ToonParticleFactory, "create_polaron")
except ImportError:
    try:
        from app.wisdom.toon_cartridge_base import (  # type: ignore
            ToonCartridge,
            ToonParticleFactory,
        )

        _TOON_AVAILABLE = True
        _TOON_V10 = hasattr(ToonParticleFactory, "create_polaron")
    except ImportError:
        _TOON_AVAILABLE = False
        _TOON_V10 = False

        @dataclass(frozen=True, slots=True)
        class ToonCartridge:  # type: ignore[no-redef]
            cartridge_id: str
            cartridge_type: str
            particle_name: str
            mass_energy_kev: float
            signature_hmac: str

            @property
            def quantum_state_hash(self) -> str:
                payload = (
                    f"{self.cartridge_id}:{self.cartridge_type}:{self.particle_name}:"
                    f"{self.mass_energy_kev:.6f}:{self.signature_hmac}"
                )
                return hashlib.sha256(payload.encode("utf-8")).hexdigest()

            def to_hilbert_state_vector(self, dimension: int = 56) -> NDArray[np.complex128]:
                dim = max(4, int(dimension))
                seed = int(self.quantum_state_hash[:16], 16)
                rng = np.random.default_rng(seed)
                v = rng.normal(0.0, 1.0, size=(dim, 1)) + 1j * rng.normal(0.0, 1.0, size=(dim, 1))
                nrm = float(np.linalg.norm(v))
                return v / (nrm if nrm > 1e-15 else 1.0)


logger = logging.getLogger("APU.Wisdom.DeRhamGaloisAdjunctionBridge")

__version__: Final[str] = "11.0.0-SOVEREIGN-DERHAM-GALOIS-POINCARE-KAM-BHT-L3-PhD"

# ──────────────────────────────────────────────────────────────────────────────
# Constantes algebraicas, espectrales, de Rham y de Poincaré / KAM / BHT
# ──────────────────────────────────────────────────────────────────────────────
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_GALOIS_GAP_MAX_TOLERANCE: Final[float] = 1.0e-6
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_DERHAM_GALOIS_SOVEREIGN_KEY"
_DIOPHANTINE_TAU: Final[float] = 2.0
_KAM_MAX_K: Final[int] = 12
_KAM_LATTICE_MODES: Final[int] = 6
_L3_DEFAULT_ITERS: Final[int] = 8
_GOLDEN_RATIO: Final[float] = 0.5 * (1.0 + math.sqrt(5.0))
_TWO_PI: Final[float] = 2.0 * math.pi
_TRANSVERSALITY_TOL: Final[float] = 1.0e-10
_RESONANCE_MAX_Q: Final[int] = 8
_RESONANCE_TOL: Final[float] = 0.02
_BHT_DISSIPATION_CEILING: Final[float] = 0.95
_DEFAULT_HILBERT_DIM: Final[int] = 56


def _is_near_resonance(
    rho: float, max_q: int = _RESONANCE_MAX_Q, tol: float = _RESONANCE_TOL
) -> bool:
    """Aproximación de Farey: ρ ≈ p/q con q ≤ max_q (divisores pequeños KAM)."""
    if not math.isfinite(rho):
        return True
    frac = rho - math.floor(rho)
    for q in range(1, max_q + 1):
        p = round(frac * q)
        if abs(frac - (p / q)) < (tol / q):
            return True
    return False


def _hermitian(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    a = np.asarray(m, dtype=np.complex128)
    return 0.5 * (a + a.conj().T)


def _antihermitian(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    a = np.asarray(m, dtype=np.complex128)
    return 0.5 * (a - a.conj().T)


def _hs_inner(a: NDArray[np.complex128], b: NDArray[np.complex128]) -> complex:
    """Producto Hilbert–Schmidt ⟨A, B⟩_HS = Tr(A† B)."""
    return complex(np.trace(np.asarray(a, dtype=np.complex128).conj().T @ np.asarray(b, dtype=np.complex128)))


def _safe_eigh(herm: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    h = _hermitian(herm)
    try:
        evals, evecs = la.eigh(h)
    except la.LinAlgError:
        evals, evecs = np.linalg.eigh(h)
    return np.real(evals).astype(np.float64), np.asarray(evecs, dtype=np.complex128)


def _matrix_sqrt_psd(psd: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Raíz cuadrada PSD vía espectro (más estable que sqrtm genérico)."""
    evals, evecs = _safe_eigh(psd)
    sqrt_e = np.sqrt(np.maximum(evals, 0.0))
    return (evecs * sqrt_e) @ evecs.conj().T


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1: FUNDACIÓN CATEGÓRICA (F, G, η, ε, Δ_△, A_μ, F_dR, ρ_MAC) + CORRECCIÓN MATEMÁTICA      ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  La ÚLTIMA definición de FASE 1 (to_categorical_state_seed) retorna un CategoricalStateSeed,     ║
# ║  que es la PRIMERA definición consumida por FASE 2.                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


class AdjunctionCoherenceStatus(IntEnum):
    """Clasificación del Estado de Coherencia Categórica F ⊣ G."""

    PERFECT_ISOMORPHISM = 0  # ⊤  (Fid ≥ 0.95, Δ_Galois ≤ 1e-6)
    DEGRADED_DISCREPANCY = 1  # ★  (Deriva menor; d_B ≤ 0.35)
    HOLONOMY_BREAKAGE = 2  # ⊥  (Ruptura de adjunción)
    META_STABLE_L3 = 3  # ⊕  (Adjunción estabilizada por L3: λ<0 ∧ KAM/BHT ✓)


@dataclass(frozen=True, slots=True)
class DeRhamGaloisAdjunctionReport:
    """Reporte inmutable de la adjunción Hom_𝒟(F(X), Y) ≅ Hom_𝒞(X, G(Y))."""

    report_id: str
    dimension_N: int
    particle_count: int
    trace_density: float
    uhlmann_fidelity: float
    bures_distance: float
    galois_gap_delta: float
    unit_eta_norm: float
    counit_epsilon_norm: float
    connection_A_norm: float
    coherence_status: AdjunctionCoherenceStatus
    sha256_merkle_root: str
    timestamp_utc: float
    # ── Campos Poincaré / KAM / BHT / L3 / de Rham (FASE 2–3; defaults = compat) ──
    lyapunov_exponent: float = 0.0
    spectral_radius: float = 1.0
    kam_diophantine_gamma: float = 0.0
    kam_stable: bool = True
    is_bht_stable: bool = True
    poincare_crossings: int = 0
    poincare_rotation_number: float = 0.0
    twist_condition: bool = False
    adjunction_winding: Tuple[float, ...] = field(default_factory=tuple)
    meta_learning_final: float = 0.0
    contraction_final: float = 0.0
    birkhoff_classification: str = "N/A"
    holonomy_phase: float = 0.0
    yang_mills_action: float = 0.0
    triangle_identity_defect: float = 0.0
    monad_idempotency_defect: float = 0.0
    uhlmann_fidelity_l3: float = 0.0
    resonance_hits: int = 0


class DiscreteDeRhamComplex:
    r"""
    Complejo de de Rham matricial sobre 𝔲(N) (trivialización global de 𝔇_n):

        Ω⁰ ──d──► Ω¹ ──d──► Ω²
         ℝ         𝔲(N)      [𝔲(N), 𝔲(N)]

    En la trivialización de Maurer–Cartan (dA ≡ 0 a nivel de carta),
        F = dA + A ∧ A = A² ,
    y la identidad de Bianchi d_A F = dF + [A, F] se verifica algebraicamente
    para F = A²: [A, A²] = 0.

    Invariantes de Chern–Weil:
        YM(A)  = ½ ‖F‖_HS²          (acción de Yang–Mills)
        ch₁    = (i / 2π) Tr(F)     (1-forma de Chern; 0 si F ∈ 𝔰𝔲(N))
    """

    def __init__(self, connection_A: NDArray[np.complex128]) -> None:
        self.A = _antihermitian(connection_A)

    def curvature(self) -> NDArray[np.complex128]:
        """2-forma de curvatura F = A ∧ A = A² (Maurer–Cartan)."""
        return self.A @ self.A

    def bianchi_residue(self) -> float:
        """‖[A, F]‖_HS — debe ser ~ 0 para F = A²."""
        F = self.curvature()
        comm = self.A @ F - F @ self.A
        return float(np.linalg.norm(comm, ord="fro"))

    def yang_mills_action(self) -> float:
        F = self.curvature()
        return 0.5 * float(np.real(_hs_inner(F, F)))

    def chern_1_proxy(self) -> float:
        F = self.curvature()
        return float(np.imag(np.trace(F)) / _TWO_PI)


class DeRhamGaloisAdjunctionBridge:
    r"""
    Puente de Adjunción F ⊣ G entre:
      • 𝒞: MIC discreta (ℤ₂^N / minitérminos)
      • 𝒟: MAC continua (𝔇_N ⊂ ℬ(ℋ), operadores densidad ρ)

    Garantiza (hasta defecto medible) Hom_𝒟(F(X), Y) ≅ Hom_𝒞(X, G(Y)),
    construye ρ_MAC con Tr(ρ) ≡ 1 y extrae la conexión de gauge A_μ ∈ 𝔰𝔲(N)
    junto con su curvatura de de Rham.
    """

    def __init__(
        self, dimension: int = _DEFAULT_HILBERT_DIM, secret_key: bytes = _DEFAULT_SECRET_KEY
    ) -> None:
        self.dimension: Final[int] = max(4, int(dimension))
        self.secret_key: Final[bytes] = bytes(secret_key)

    def _psd_higham_projection(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Proyección convexa de Higham al cono PSD hermitiano."""
        herm = _hermitian(np.asarray(matrix, dtype=np.complex128))
        eigvals, eigvecs = _safe_eigh(herm)
        clean = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)
        return (eigvecs * clean) @ eigvecs.conj().T

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        """Sanitiza y fuerza traza unitaria estricta Tr(ρ) = 1."""
        psd = self._psd_higham_projection(matrix)
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    def functor_F_elevation(
        self, cartridges: Sequence[ToonCartridge]
    ) -> NDArray[np.complex128]:
        r"""
        F(X) = arg min_{M=M† ⪰ 0, Tr=1} ‖M − Σ_i |v_i⟩⟨v_i|‖_F   (proyección Higham).

        F es el funtor de elevación 𝒞 → 𝒟 (left adjoint).
        """
        dim = self.dimension
        if not cartridges:
            return np.eye(dim, dtype=np.complex128) / float(dim)

        rho_accum = np.zeros((dim, dim), dtype=np.complex128)
        for cartridge in cartridges:
            v_i = np.asarray(cartridge.to_hilbert_state_vector(dim), dtype=np.complex128)
            if v_i.ndim == 1:
                v_i = v_i.reshape((dim, 1))
            elif v_i.shape != (dim, 1):
                v_i = v_i.reshape((dim, 1))
            rho_accum += v_i @ v_i.conj().T
        return self.sanitize_density_matrix(rho_accum)

    def functor_G_forgetful(
        self, density_matrix: NDArray, threshold: Optional[float] = None
    ) -> NDArray[np.float64]:
        r"""
        G(ρ) = 1{ Re diag(ρ) ≥ 1/N } ∈ {0,1}^N.

        El umbral de equipartición 1/N es el punto fijo de la medida de Haar
        sobre el simplejo (funtor de olvido 𝒟 → 𝒞, right adjoint).
        """
        if threshold is None:
            threshold = 1.0 / float(self.dimension)
        rho = self.sanitize_density_matrix(density_matrix)
        diag_real = np.real(np.diag(rho))
        return np.where(diag_real >= threshold, 1.0, 0.0).astype(np.float64)

    def compute_unit_eta_and_counit_epsilon(
        self,
        cartridges: Sequence[ToonCartridge],
        density_matrix: NDArray,
    ) -> Tuple[float, float, float]:
        r"""
        Unidad η : 1_𝒞 ⇒ G∘F ,  Counidad ε : F∘G ⇒ 1_𝒟.

        Galois gap (reciprocidad de Frobenius numérica):
            Δ = | ⟨F(X), Y⟩_HS − ⟨X, G(Y)⟩_ℝ |
              = | Tr(ρ_F ρ_Y) − ⟨diag(ρ_F), diag(ρ_Y)⟩ |
        """
        rho_F = self.functor_F_elevation(cartridges)
        rho_Y = self.sanitize_density_matrix(density_matrix)
        v_G = self.functor_G_forgetful(rho_Y)
        v_diag_F = np.real(np.diag(rho_F))

        unit_eta_norm = float(np.linalg.norm(v_diag_F - v_G))

        rho_counit = np.diag(v_G).astype(np.complex128)
        counit_epsilon_norm = float(np.linalg.norm(rho_F - rho_counit, ord="fro"))

        pairing_continuous = float(np.real(np.trace(rho_F @ rho_Y)))
        v_target_diag = np.real(np.diag(rho_Y))
        pairing_discrete = float(np.dot(v_diag_F, v_target_diag))
        galois_gap = float(abs(pairing_continuous - pairing_discrete))
        return unit_eta_norm, counit_epsilon_norm, galois_gap

    def triangle_identity_defects(
        self, cartridges: Sequence[ToonCartridge]
    ) -> Tuple[float, float]:
        r"""
        Identidades del triángulo de la adjunción:

            (ε F) ∘ (F η) = id_F     (izquierda)
            (G ε) ∘ (η G) = id_G     (derecha)

        Proxies numéricos sobre 𝔇_N / {0,1}^N:
            Δ_L = ‖ ρ_F − F(G(ρ_F)) ‖_F
            Δ_R = ‖ G(ρ_F) − G(F(G(ρ_F))) ‖_2
        """
        rho_F = self.functor_F_elevation(cartridges)
        bits = self.functor_G_forgetful(rho_F)
        rho_FGF = self.sanitize_density_matrix(np.diag(bits).astype(np.complex128))
        bits2 = self.functor_G_forgetful(rho_FGF)
        delta_left = float(np.linalg.norm(rho_F - rho_FGF, ord="fro"))
        delta_right = float(np.linalg.norm(bits - bits2))
        return delta_left, delta_right

    def monad_idempotency_defect(self, cartridges: Sequence[ToonCartridge]) -> float:
        r"""
        Mónada T = G∘F sobre 𝒞. Para una reflexión (adjunción idempotente)
        se exige T² = T, i.e. G F G F = G F.
        """
        _, delta_right = self.triangle_identity_defects(cartridges)
        return delta_right

    def extract_gauge_connection_A(
        self, density_matrix: NDArray
    ) -> NDArray[np.complex128]:
        r"""
        A = i (ρ − (Tr ρ / N) I) ∈ 𝔰𝔲(N), anti-hermítica y de traza nula.
        Es la desviación infinitesimal respecto del estado máximamente mixto
        (conexión de Maurer–Cartan de la órbita unitaria).
        """
        rho = self.sanitize_density_matrix(density_matrix)
        n = rho.shape[0]
        tr_val = np.trace(rho) / float(n)
        su_n_part = rho - tr_val * np.eye(n, dtype=np.complex128)
        return _antihermitian(1j * su_n_part)

    def de_rham_curvature_bundle(
        self, density_matrix: NDArray
    ) -> DiscreteDeRhamComplex:
        """Complejo de de Rham anclado en la conexión A_μ extraída de ρ."""
        return DiscreteDeRhamComplex(self.extract_gauge_connection_A(density_matrix))

    def evaluate_uhlmann_fidelity_and_bures(
        self, rho_a: NDArray, rho_b: NDArray
    ) -> Tuple[float, float]:
        r"""
        Fidelidad de Uhlmann (Nielsen–Chuang / Uhlmann 1976):

            F(ρ, σ) = ‖ √ρ √σ ‖_1² = [ Tr √(√ρ σ √ρ) ]²

        Implementación estable: suma de valores singulares de √ρ √σ
        (norma nuclear), sin renormalizar el operador intermedio.

        Distancia de Bures: d_B(ρ, σ) = √(2 − 2 √F).
        """
        r_a = self.sanitize_density_matrix(rho_a)
        r_b = self.sanitize_density_matrix(rho_b)
        sqrt_a = _matrix_sqrt_psd(r_a)
        sqrt_b = _matrix_sqrt_psd(r_b)
        try:
            svals = np.linalg.svd(sqrt_a @ sqrt_b, compute_uv=False)
        except np.linalg.LinAlgError:
            middle = _hermitian(sqrt_a @ r_b @ sqrt_a)
            evals, _ = _safe_eigh(middle)
            svals = np.sqrt(np.maximum(evals, 0.0))
        trace_sqrt = float(np.sum(np.real(svals)))
        if not math.isfinite(trace_sqrt):
            trace_sqrt = 0.0
        fidelity = min(max(trace_sqrt**2, 0.0), 1.0)
        bures_dist = float(math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fidelity))))
        return fidelity, bures_dist

    def evaluate_adjunction_coherence(
        self,
        cartridges: Sequence[ToonCartridge],
        target_density_matrix: Optional[NDArray] = None,
    ) -> DeRhamGaloisAdjunctionReport:
        """
        Auditoría categórica base SIN el análisis Poincaré/KAM/L3
        (esos campos quedan en defaults; se rellenan en FASE 3 vía SovereignAuditor).
        """
        now = time.time()
        dim = self.dimension
        rho_F = self.functor_F_elevation(cartridges)
        target = (
            rho_F
            if target_density_matrix is None
            else self.sanitize_density_matrix(target_density_matrix)
        )

        unit_eta, counit_eps, galois_gap = self.compute_unit_eta_and_counit_epsilon(
            cartridges, target
        )
        fidelity, bures_dist = self.evaluate_uhlmann_fidelity_and_bures(rho_F, target)
        conn_A = self.extract_gauge_connection_A(target)
        conn_norm = float(np.linalg.norm(conn_A, ord="fro"))
        tr_val = float(np.real(np.trace(target)))
        dR = DiscreteDeRhamComplex(conn_A)
        ym = dR.yang_mills_action()
        tri_L, tri_R = self.triangle_identity_defects(cartridges)
        tri_def = max(tri_L, tri_R)

        if fidelity >= _UHLMANN_FIDELITY_MIN and galois_gap <= _GALOIS_GAP_MAX_TOLERANCE:
            status = AdjunctionCoherenceStatus.PERFECT_ISOMORPHISM
        elif fidelity >= 0.85 and bures_dist <= 0.35:
            status = AdjunctionCoherenceStatus.DEGRADED_DISCREPANCY
        else:
            status = AdjunctionCoherenceStatus.HOLONOMY_BREAKAGE

        rep_id = f"DERHAM-GALOIS-ADJUNCTION-{int(now * 1000.0) % 1000000:06d}"
        seed_bytes = (
            f"{rep_id}:{fidelity:.8f}:{bures_dist:.8f}:{galois_gap:.8f}:{status.value}"
        ).encode("utf-8")
        merkle_root = hashlib.sha256(seed_bytes).hexdigest()

        return DeRhamGaloisAdjunctionReport(
            report_id=rep_id,
            dimension_N=dim,
            particle_count=len(cartridges),
            trace_density=tr_val,
            uhlmann_fidelity=fidelity,
            bures_distance=bures_dist,
            galois_gap_delta=galois_gap,
            unit_eta_norm=unit_eta,
            counit_epsilon_norm=counit_eps,
            connection_A_norm=conn_norm,
            coherence_status=status,
            sha256_merkle_root=merkle_root,
            timestamp_utc=now,
            yang_mills_action=ym,
            triangle_identity_defect=tri_def,
            monad_idempotency_defect=tri_R,
        )

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 1 → FASE 2: FRONTERA (to_categorical_state_seed).                                     ║
    # ║  ÚLTIMO método de FASE 1. Retorna CategoricalStateSeed → PRIMERA def. de FASE 2.            ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def to_categorical_state_seed(
        self,
        cartridges: Sequence[ToonCartridge],
        target_density_matrix: Optional[NDArray] = None,
    ) -> "CategoricalStateSeed":
        r"""
        Comprime la coherencia categórica (η, ε, Δ_Galois, Δ_△, ρ_F, A_μ, F_dR)
        en una semilla canónica que alimenta el análisis Poincaré / KAM / L3:

          • ρ_F (vector plano) + ρ_target (ideal: proyección diagonal de ρ_F)
          • Probe hermítico (observable de sección de Poincaré: g(ρ)=⟨H,ρ⟩_HS − c)
          • Espectro canónico:
                J_k = autovalores de ρ_F  (acciones)
                θ_k = arg(ψ_dom)_k + 0.1 (A_μ)_{kk}   (ángulos + corrección de gauge)
                ω_k = √(1+k) + (φ−1)/(1+k)            (FPU ⊕ cascade áureo)

        Retorna el CategoricalStateSeed que INICIA FASE 2.
        """
        rho_F = self.functor_F_elevation(cartridges)
        if target_density_matrix is None:
            rho_target = np.diag(np.real(np.diag(rho_F))).astype(np.complex128)
            rho_target = self.sanitize_density_matrix(rho_target)
        else:
            rho_target = self.sanitize_density_matrix(target_density_matrix)

        dim = self.dimension
        eigvals, eigvecs = _safe_eigh(rho_F)
        order = np.argsort(eigvals)[::-1]
        J_sorted = np.maximum(eigvals[order], 0.0)
        J_total = float(J_sorted.sum())
        J_norm = J_sorted / (J_total + _WILKINSON_EPSILON)

        k_idx = np.arange(dim, dtype=np.float64)
        omega = np.sqrt(1.0 + k_idx) + (_GOLDEN_RATIO - 1.0) / (1.0 + k_idx)

        dominant_vec = eigvecs[:, order[0]]
        conn_A = self.extract_gauge_connection_A(rho_F)
        conn_corr = np.real(np.diag(1j * conn_A))
        theta = np.angle(dominant_vec) + 0.1 * conn_corr

        seed_val = int(
            hashlib.sha256(f"{self._hash_cartridges(cartridges)}:{dim}".encode()).hexdigest()[:16],
            16,
        )
        rng = np.random.default_rng(seed_val)
        H_raw = rng.normal(0.0, 1.0, (dim, dim))
        probe = _hermitian(H_raw.astype(np.complex128))
        tr_p = float(np.real(np.trace(probe)))
        probe = probe - (tr_p / float(dim)) * np.eye(dim, dtype=np.complex128)
        pnorm = float(np.linalg.norm(probe, ord="fro"))
        if pnorm > _WILKINSON_EPSILON:
            probe = probe / pnorm

        unit_eta, counit_eps, galois_gap = self.compute_unit_eta_and_counit_epsilon(
            cartridges, rho_target
        )
        fid, bures = self.evaluate_uhlmann_fidelity_and_bures(rho_F, rho_target)
        tri_L, tri_R = self.triangle_identity_defects(cartridges)
        dR = DiscreteDeRhamComplex(conn_A)

        return CategoricalStateSeed(
            source_merkle=self._hash_cartridges(cartridges),
            rho_vector=tuple(complex(x) for x in rho_F.flatten()),
            target_vector=tuple(complex(x) for x in rho_target.flatten()),
            probe_vector=tuple(complex(x) for x in probe.flatten()),
            action_vector=tuple(float(x) for x in J_norm),
            angle_vector=tuple(float(x) for x in theta),
            frequency_vector=tuple(float(x) for x in omega),
            eta_norm=unit_eta,
            epsilon_norm=counit_eps,
            galois_gap=galois_gap,
            fidelity=fid,
            bures_distance=bures,
            triangle_left=tri_L,
            triangle_right=tri_R,
            yang_mills_action=dR.yang_mills_action(),
            bianchi_residue=dR.bianchi_residue(),
            dimension=dim,
        )

    @staticmethod
    def _hash_cartridges(cartridges: Sequence[ToonCartridge]) -> str:
        acc = hashlib.sha256()
        if not cartridges:
            acc.update(b"EMPTY_CARTRIDGE_SET")
        for c in cartridges:
            acc.update(c.quantum_state_hash.encode("utf-8"))
        return acc.hexdigest()


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 2: SEMILLA CANÓNICA + POINCARÉ Σ + BIRKHOFF-TWIST + KAM/BHT + OPERADOR L3                 ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (CategoricalStateSeed) consume el retorno de to_categorical_state_seed (FASE 1).  ║
# ║  Última def. (RecursiveAdjunctionImprover.iterate_level_3) → AdjunctionTrajectory (FASE 3).     ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class CategoricalStateSeed:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 2 — ENTRY: continúa de to_categorical_state_seed.                                      │
    │  Semilla canónica (ρ, ρ*, H_probe, J, θ, ω) que ancla Poincaré / KAM / L3.                   │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
    """

    source_merkle: str
    rho_vector: Tuple[complex, ...]
    target_vector: Tuple[complex, ...]
    probe_vector: Tuple[complex, ...]
    action_vector: Tuple[float, ...]
    angle_vector: Tuple[float, ...]
    frequency_vector: Tuple[float, ...]
    eta_norm: float
    epsilon_norm: float
    galois_gap: float
    fidelity: float
    bures_distance: float
    triangle_left: float
    triangle_right: float
    yang_mills_action: float
    bianchi_residue: float
    dimension: int

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

    def purity(self) -> float:
        rho = self.get_rho()
        return float(np.real(np.trace(rho @ rho)))

    def von_neumann_entropy(self) -> float:
        ev = np.maximum(np.real(_safe_eigh(self.get_rho())[0]), _WILKINSON_EPSILON)
        ev = ev / float(ev.sum())
        return float(-np.sum(ev * np.log(ev)))

    def poincare_cartan_invariant(self) -> float:
        r"""I₁ = Σ_k J_k = Tr(ρ) = 1  (invariante integral de Poincaré–Cartan)."""
        return float(sum(self.action_vector))


class AdjunctionPoincareSection:
    r"""
    Sección de Poincaré transversal sobre el simplejo de densidad:
        Σ = { ρ ∈ 𝔇_N : g(ρ) = Re⟨H_probe, ρ⟩_HS − c = 0 }.
    H_probe hermítico ⇒ g es un funcional lineal real (observable).
    Requiere transversalidad ∇g · T_{L3} ≠ 0.
    """

    def __init__(self, probe: NDArray[np.complex128], offset: float, label: str = "Σ") -> None:
        self.probe = _hermitian(probe)
        self.offset = float(offset)
        self.label = label

    def g(self, rho: NDArray[np.complex128]) -> float:
        return float(np.real(_hs_inner(self.probe, rho))) - self.offset

    def transversality(
        self, d_rho: NDArray[np.complex128], tol: float = _TRANSVERSALITY_TOL
    ) -> bool:
        """∇g · ṙ = Re⟨H, ṙ⟩_HS ≠ 0 (corte transversal)."""
        return abs(float(np.real(_hs_inner(self.probe, d_rho)))) > tol

    def oriented_flux(self, d_rho: NDArray[np.complex128]) -> float:
        return float(np.real(_hs_inner(self.probe, d_rho)))


class AdjunctionReturnMap:
    r"""
    Mapa de retorno P : Σ → Σ inducido por la órbita L3 de la adjunción.
    Cruces g(ρ_i)·g(ρ_{i+1}) < 0 con interpolación lineal, aceptados sólo si
    el campo discreto Δρ es transversal a Σ. Análogo celeste sobre 𝔇_N.
    """

    def __init__(self, section: AdjunctionPoincareSection) -> None:
        self.section = section

    def detect_crossings(
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
            if abs(denom) <= _WILKINSON_EPSILON:
                continue
            t = float(np.clip(gi / denom, 0.0, 1.0))
            flow = rho_sequence[i + 1] - rho_sequence[i]
            if not self.section.transversality(flow):
                continue
            rho_cross = (1.0 - t) * rho_sequence[i] + t * rho_sequence[i + 1]
            tr_val = float(np.real(np.trace(rho_cross)))
            if tr_val > _WILKINSON_EPSILON:
                rho_cross = rho_cross / tr_val
            crossings.append(rho_cross)
        return crossings

    def apply(
        self, rho_sequence: Sequence[NDArray[np.complex128]]
    ) -> List[NDArray[np.complex128]]:
        return self.detect_crossings(rho_sequence)

    @staticmethod
    def birkhoff_rotation_number(
        crossings: Sequence[NDArray[np.complex128]], probe: NDArray[np.complex128]
    ) -> float:
        r"""
        Número de rotación de Birkhoff del mapa de Poincaré:
            ρ = (θ_N − θ_0) / (2π N),   θ = arg Tr(H† ρ),
        con unwrap para eliminar saltos de rama.
        """
        if len(crossings) < 2:
            return 0.0
        H = _hermitian(probe)
        raw = [float(np.angle(_hs_inner(H, r))) for r in crossings]
        angles = np.unwrap(np.array(raw, dtype=np.float64))
        n_steps = len(angles) - 1
        if n_steps <= 0:
            return 0.0
        return float((angles[-1] - angles[0]) / (_TWO_PI * n_steps))


class AdjunctionPoincareBirkhoffTwist:
    r"""
    Teorema geométrico de Poincaré–Birkhoff sobre el anillo (J_dom, θ_dom):
    un mapa que preserva área con twist ∂(Δθ)/∂J ≠ 0 posee ≥ 2 puntos fijos.
    """

    def __init__(self, seed: CategoricalStateSeed, delta_J: float = 1.0e-4) -> None:
        self.seed = seed
        self.delta_J = float(delta_J)

    def twist_derivative(self) -> float:
        J = np.array(self.seed.action_vector, dtype=np.float64)
        theta = np.array(self.seed.angle_vector, dtype=np.float64)
        omega = np.array(self.seed.frequency_vector, dtype=np.float64)
        if J.size == 0:
            return 0.0
        dom = int(np.argmax(J))
        dJ = self.delta_J

        def advance(J_vec: NDArray[np.float64]) -> float:
            return float((theta[dom] + omega[dom] * J_vec[dom]) % _TWO_PI)

        J_plus, J_minus = J.copy(), J.copy()
        J_plus[dom] = min(1.0, J[dom] + dJ)
        J_minus[dom] = max(_WILKINSON_EPSILON, J[dom] - dJ)
        d_theta = advance(J_plus) - advance(J_minus)
        d_theta = (d_theta + math.pi) % _TWO_PI - math.pi
        denom = float(J_plus[dom] - J_minus[dom])
        if abs(denom) <= _WILKINSON_EPSILON:
            return 0.0
        return float(d_theta / denom)

    def satisfies_twist(self, tol: float = 1.0e-12) -> bool:
        return abs(self.twist_derivative()) > tol


class BirkhoffAdjunctionForm:
    r"""
    Forma normal de Birkhoff en el punto fijo de la iteración L3:
        T(ρ) = ρ* + DT(ρ*)·(ρ − ρ*) + O(‖ρ − ρ*‖²).
    Clasificación espectral y exponente de Lyapunov local.
    """

    def __init__(
        self,
        fixed_rho: NDArray[np.complex128],
        jacobian_spectrum: NDArray[np.complex128],
    ) -> None:
        self.fixed_rho = fixed_rho
        self.jacobian_spectrum = np.asarray(jacobian_spectrum, dtype=np.complex128)

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
        return (
            float(np.max(np.abs(self.jacobian_spectrum))) if self.jacobian_spectrum.size else 0.0
        )

    def lyapunov_exponent(self) -> float:
        rho = self.spectral_radius()
        return float(np.log(max(rho, _WILKINSON_EPSILON)))

    def twist_eigenvalue_condition(self) -> bool:
        """Elíptica no degenerada: ∃ λ con |λ|=1 y λ ≠ ±1."""
        for lam in self.jacobian_spectrum:
            mod = abs(complex(lam))
            if abs(mod - 1.0) < 1e-8 and abs(lam - 1.0) > 1e-8 and abs(lam + 1.0) > 1e-8:
                return True
        return False


class KAMCoherenceTorus:
    r"""
    Toro invariante 𝕋^n de coherencia F⊣G con frecuencias ω ∈ ℝ^n.

    Condición diofantina (Kolmogorov):
        |⟨k, ω⟩| ≥ γ / |k|_1^τ ,   ∀ k ∈ ℤ^n \ {0},   τ > n − 1.

    Criterio BHT (Broer–Huitema–Takens) en el régimen disipativo L3:
        el toro persiste como NHIM si γ > ‖ε‖ y la disipación no satura γ.
    """

    def __init__(self, frequencies: Sequence[float]) -> None:
        self.frequencies = tuple(float(f) for f in frequencies)
        self.n = len(self.frequencies)

    def _lattice_vectors(self, max_k: int, modes: int) -> Iterable[NDArray[np.float64]]:
        m = max(1, min(self.n, modes))
        seen: Set[Tuple[int, ...]] = set()

        def emit(vec: NDArray[np.float64]) -> Iterable[NDArray[np.float64]]:
            key = tuple(int(x) for x in vec)
            if any(key) and key not in seen:
                seen.add(key)
                yield vec

        for i in range(m):
            for s in (-1, 1):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = s
                yield from emit(kvec)

        for i, j in itertools.combinations(range(m), 2):
            for si, sj in itertools.product((-1, 1), repeat=2):
                kvec = np.zeros(m, dtype=np.float64)
                kvec[i] = si
                kvec[j] = sj
                yield from emit(kvec)

        cube = min(m, 4)
        bound = min(2, max(1, max_k // 6))
        for coeffs in itertools.product(range(-bound, bound + 1), repeat=cube):
            if all(c == 0 for c in coeffs):
                continue
            kvec = np.zeros(m, dtype=np.float64)
            kvec[:cube] = np.array(coeffs, dtype=np.float64)
            yield from emit(kvec)

    def diophantine_constant(
        self, max_k: int = _KAM_MAX_K, tau: float = _DIOPHANTINE_TAU
    ) -> float:
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        gammas: List[float] = []
        for kvec in self._lattice_vectors(max_k=max_k, modes=m):
            inner = abs(float(np.dot(kvec, omega)))
            l1 = float(np.sum(np.abs(kvec)))
            if l1 <= 0.0:
                continue
            gammas.append(inner * (l1**tau))
        return float(min(gammas)) if gammas else 0.0

    def small_divisors(self, max_k: int = _KAM_MAX_K) -> Tuple[float, ...]:
        m = max(1, min(self.n, _KAM_LATTICE_MODES))
        omega = np.array(self.frequencies[:m], dtype=np.float64)
        divs = [abs(float(np.dot(kvec, omega))) for kvec in self._lattice_vectors(max_k, m)]
        divs = [d for d in divs if d > 0.0]
        divs.sort()
        return tuple(divs[:8])

    def is_kam_stable(self, perturbation_norm: float, gamma: Optional[float] = None) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        return gamma > perturbation_norm

    def is_dissipative_kam_stable(
        self,
        perturbation_norm: float,
        dissipation: float,
        gamma: Optional[float] = None,
    ) -> bool:
        if gamma is None:
            gamma = self.diophantine_constant()
        dissip = abs(float(dissipation))
        return (gamma > perturbation_norm) and (dissip < min(gamma, _BHT_DISSIPATION_CEILING))

    def winding_numbers(self, phases: Sequence[float]) -> Tuple[float, ...]:
        if len(phases) < 2:
            return tuple(0.0 for _ in range(self.n))
        unwrapped = np.unwrap(np.array(phases, dtype=np.float64))
        delta_phi = float(unwrapped[-1] - unwrapped[0])
        if abs(delta_phi) < _WILKINSON_EPSILON:
            return tuple(0.0 for _ in range(self.n))
        return tuple(float(w / delta_phi) for w in self.frequencies)


def pancharatnam_berry_holonomy(rho_sequence: Sequence[NDArray[np.complex128]]) -> float:
    r"""
    Holonomía discreta de Pancharatnam–Berry a lo largo de {ρ_t}:
        γ_geom = Σ_t arg ⟨ψ_t | ψ_{t+1}⟩ ,
    con gauge de transporte paralelo (ψ ← ψ · ⟨ψ_prev|ψ⟩* / |⟨·⟩|).
    """
    if len(rho_sequence) < 2:
        return 0.0
    acc = 0.0
    prev: Optional[NDArray[np.complex128]] = None
    for rho in rho_sequence:
        evals, evecs = _safe_eigh(rho)
        psi = evecs[:, int(np.argmax(evals))]
        if prev is not None:
            overlap = np.vdot(prev, psi)
            acc += float(np.angle(overlap))
            mag = abs(overlap)
            if mag > _WILKINSON_EPSILON:
                psi = psi * (np.conj(overlap) / mag)
        prev = psi
    return float(acc)


class RecursiveAdjunctionImprover:
    r"""
    Operador de automejora recursiva I : 𝔇_N → 𝔇_N sobre el ciclo F ⊣ G.

      L0: I_0(ρ) = ρ                                                 (identidad)
      L1: I_1(ρ) = Higham((1−θ)ρ + θ ρ*)                             (contracción)
      L2: θ ← θ·(1 + 0.1(ratio_err − 1))                             (residuo)
      L3: η_meta ← η_meta · {0.90 / 1.05} según ρ_espectral(DI)
          ⊕ realimentación del número de rotación de Poincaré
          ⊕ penalización de resonancias p/q (divisores pequeños KAM)
          ⊕ proyección PSD + Tr = 1
        = metacognición metacognitiva con observable celeste de 2º orden.
    """

    def __init__(self, contraction: float = 0.5, meta_learning: float = 0.3) -> None:
        self.contraction: float = float(contraction)
        self.meta_learning: float = float(meta_learning)
        self._last_rotation: float = 0.0
        self._resonance_hits: int = 0

    def apply_level_0(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.asarray(rho, dtype=np.complex128).copy()

    def apply_level_1(
        self, rho: NDArray[np.complex128], target: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        theta = self.contraction
        mixed = (1.0 - theta) * rho + theta * target
        return self._project_psd_normalize(mixed)

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
        poincare_rotation: Optional[float] = None,
    ) -> NDArray[np.complex128]:
        if len(history) >= 3:
            rho_spec = self._estimate_local_spectral_radius(history[-3:])
            if rho_spec > 1.0:
                self.meta_learning = float(np.clip(self.meta_learning * 0.90, 0.05, 0.90))
            elif 0.0 < rho_spec < 0.5:
                self.meta_learning = float(np.clip(self.meta_learning * 1.05, 0.05, 0.90))

        if poincare_rotation is not None and math.isfinite(poincare_rotation):
            self._last_rotation = float(poincare_rotation)
            if _is_near_resonance(poincare_rotation):
                self._resonance_hits += 1
                self.contraction = float(np.clip(self.contraction * 1.08, 0.05, 0.95))
                self.meta_learning = float(np.clip(self.meta_learning * 0.92, 0.05, 0.90))
            else:
                self.contraction = float(np.clip(self.contraction * 0.98, 0.05, 0.95))

        self.contraction = float(
            np.clip(self.contraction + 0.05 * (self.meta_learning - 0.5), 0.05, 0.95)
        )
        return self.apply_level_2(rho, target, history)

    @staticmethod
    def _estimate_local_spectral_radius(recent: Sequence[NDArray[np.complex128]]) -> float:
        r"""Cociente de Rayleigh HS: λ ≈ ⟨Δ₂, Δ₁⟩_HS / ‖Δ₂‖²."""
        if len(recent) < 3:
            return 0.5
        d1 = recent[-1] - recent[-2]
        d2 = recent[-2] - recent[-3]
        denom = _hs_inner(d2, d2)
        if abs(denom) < _WILKINSON_EPSILON:
            return 0.5
        return float(abs(_hs_inner(d2, d1) / denom))

    @staticmethod
    def _project_psd_normalize(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
        herm = _hermitian(m)
        eigs, vecs = _safe_eigh(herm)
        clean = np.maximum(eigs, _SPECTRAL_PSD_FLOOR)
        psd = (vecs * clean) @ vecs.conj().T
        tr = float(np.real(np.trace(psd)))
        if tr < _WILKINSON_EPSILON:
            n = psd.shape[0]
            return np.eye(n, dtype=np.complex128) / float(n)
        return psd / tr

    def _poincare_rotation_from_history(
        self,
        history: Sequence[NDArray[np.complex128]],
        probe: NDArray[np.complex128],
    ) -> Optional[float]:
        if len(history) < 4:
            return None
        section = AdjunctionPoincareSection(probe=probe, offset=0.0, label="Σ_L3")
        prm = AdjunctionReturnMap(section)
        crossings = prm.detect_crossings(list(history))
        if len(crossings) < 2:
            return None
        return AdjunctionReturnMap.birkhoff_rotation_number(crossings, probe)

    # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 → FASE 3: FRONTERA (iterate_level_3).                                                ║
    # ║  ÚLTIMO método de FASE 2. Retorna AdjunctionTrajectory → PRIMER objeto de FASE 3.            ║
    # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
    def iterate_level_3(
        self,
        seed: CategoricalStateSeed,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> "AdjunctionTrajectory":
        r"""
        Itera I_L3 sobre ρ_seed con target ρ*_seed. Devuelve:
          • Secuencia ρ_0 → ρ_T
          • Acciones J_t = Tr(ρ_t) ≡ 1 (invariante de traza / Poincaré–Cartan)
          • Ángulos θ_t = arg Tr(H_probe† ρ_t)  (fase de Poincaré)
          • Exponente de Lyapunov λ (log-ratio de ‖Δρ‖_F)
          • Constante diofantina γ y estabilidad KAM / BHT
          • Números de enrollamiento, rotación de Birkhoff, twist P–B
          • Holonomía de Pancharatnam–Berry a lo largo de la órbita
        """
        dim = seed.dimension
        rho = seed.get_rho()
        target = seed.get_target()
        probe = seed.get_probe()

        rho_sequence: List[NDArray[np.complex128]] = [rho.copy()]
        action_sequence: List[float] = [float(np.real(np.trace(rho)))]
        angle_sequence: List[float] = [float(np.angle(_hs_inner(probe, rho)))]
        history: List[NDArray[np.complex128]] = []

        for _ in range(max_iters):
            history.append(rho.copy())
            rho_p = self._poincare_rotation_from_history(history, probe)
            rho = self.apply_level_3(rho, target, history, poincare_rotation=rho_p)
            rho_sequence.append(rho.copy())
            action_sequence.append(float(np.real(np.trace(rho))))
            angle_sequence.append(float(np.angle(_hs_inner(probe, rho))))

        diffs = [
            float(np.linalg.norm(rho_sequence[i + 1] - rho_sequence[i], ord="fro"))
            for i in range(len(rho_sequence) - 1)
        ]
        if len(diffs) >= 2 and all(d > _WILKINSON_EPSILON for d in diffs):
            log_ratios = np.log(np.array(diffs[1:]) / np.maximum(np.array(diffs[:-1]), _WILKINSON_EPSILON))
            finite = np.isfinite(log_ratios)
            lyapunov = float(np.mean(log_ratios[finite])) if np.any(finite) else 0.0
        elif diffs and all(d <= _WILKINSON_EPSILON for d in diffs):
            lyapunov = -20.0
        else:
            lyapunov = -20.0 if not diffs else float(np.log(max(diffs[-1], _WILKINSON_EPSILON)))
        if not math.isfinite(lyapunov):
            lyapunov = -20.0 if lyapunov < 0 else 0.0
        convergence = diffs[-1] if diffs else 0.0

        torus = KAMCoherenceTorus(seed.frequency_vector)
        gamma = torus.diophantine_constant()
        dissipation = abs(self.contraction)
        kam_stable = torus.is_kam_stable(convergence, gamma)
        bht_stable = torus.is_dissipative_kam_stable(convergence, dissipation, gamma)
        winding = torus.winding_numbers(angle_sequence)

        jacobian_spectrum = self._estimate_global_jacobian_spectrum(rho_sequence)
        birkhoff = BirkhoffAdjunctionForm(rho_sequence[-1], jacobian_spectrum)
        birkhoff_class = birkhoff.classify()

        live_section = AdjunctionPoincareSection(probe=probe, offset=0.0, label="Σ_flow")
        prm = AdjunctionReturnMap(live_section)
        crossings = prm.apply(rho_sequence)
        rotation = (
            AdjunctionReturnMap.birkhoff_rotation_number(crossings, probe)
            if len(crossings) >= 2
            else 0.0
        )
        twist = AdjunctionPoincareBirkhoffTwist(seed)
        holonomy = pancharatnam_berry_holonomy(rho_sequence)

        spectral_radius = float(abs(math.exp(lyapunov))) if math.isfinite(lyapunov) else 1.0

        return AdjunctionTrajectory(
            source_merkle=seed.source_merkle,
            dimension=dim,
            rho_sequence=tuple(self._serialize(r) for r in rho_sequence),
            action_sequence=tuple(action_sequence),
            angle_sequence=tuple(angle_sequence),
            lyapunov_exponent=lyapunov,
            convergence_residual=convergence,
            spectral_radius=spectral_radius,
            is_kam_stable=kam_stable,
            is_bht_stable=bht_stable,
            winding_numbers=winding,
            diophantine_gamma=gamma,
            final_contraction=self.contraction,
            final_meta_learning=self.meta_learning,
            birkhoff_classification=birkhoff_class,
            poincare_rotation_number=rotation,
            poincare_crossing_count=len(crossings),
            twist_condition=twist.satisfies_twist(),
            holonomy_phase=holonomy,
            resonance_hits=self._resonance_hits,
            triangle_left=seed.triangle_left,
            triangle_right=seed.triangle_right,
            yang_mills_action=seed.yang_mills_action,
        )

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
        if len(rho_sequence) < 2:
            return np.array([0.5 + 0.0j], dtype=np.complex128)
        d_last = rho_sequence[-1] - rho_sequence[-2]
        if len(rho_sequence) >= 3:
            d_prev = rho_sequence[-2] - rho_sequence[-3]
            denom = _hs_inner(d_prev, d_prev)
            if abs(denom) > _WILKINSON_EPSILON:
                rayleigh = _hs_inner(d_prev, d_last) / denom
                return np.array([complex(rayleigh)], dtype=np.complex128)
        nrm = float(np.linalg.norm(d_last, ord="fro"))
        return np.array([complex(nrm, 0.0)], dtype=np.complex128)


# ╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 3: TRAYECTORIA + AUDITOR SOBERANO + REPORTE ENRIQUECIDO                                   ║
# ║  ─────────────────────────────────────────────────────────────────────────────────────────────  ║
# ║  Primera def. (AdjunctionTrajectory) consume iterate_level_3 de FASE 2.                         ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True, slots=True)
class AdjunctionTrajectory:
    r"""
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │  FASE 3 — ENTRY: continúa de iterate_level_3 (FASE 2).                                       │
    │  Trayectoria L3 sobre F⊣G, con Poincaré, twist P–B, KAM/BHT y holonomía de Rham.            │
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
    is_bht_stable: bool
    winding_numbers: Tuple[float, ...]
    diophantine_gamma: float
    final_contraction: float
    final_meta_learning: float
    birkhoff_classification: str
    poincare_rotation_number: float
    poincare_crossing_count: int
    twist_condition: bool
    holonomy_phase: float
    resonance_hits: int
    triangle_left: float
    triangle_right: float
    yang_mills_action: float

    def get_rho_sequence_arrays(self) -> List[NDArray[np.complex128]]:
        return [
            RecursiveAdjunctionImprover.deserialize(s, self.dimension) for s in self.rho_sequence
        ]

    def build_poincare_section(self, probe: NDArray[np.complex128]) -> AdjunctionPoincareSection:
        rho_arr = self.get_rho_sequence_arrays()
        g_vals = [float(np.real(_hs_inner(probe, r))) for r in rho_arr]
        offset = float(np.mean(g_vals)) if g_vals else 0.0
        return AdjunctionPoincareSection(
            probe=probe, offset=offset, label=f"Σ[{self.source_merkle[:8]}]"
        )

    def is_stable_meta_adjunction(self) -> bool:
        """Criterio L3 soberano: λ < 0 ∧ (KAM ∨ BHT) ∧ Birkhoff ∈ {ATTRACTING, ELLIPTIC_CENTER}."""
        return (
            self.lyapunov_exponent < 0.0
            and (self.is_kam_stable or self.is_bht_stable)
            and self.birkhoff_classification in ("ATTRACTING", "ELLIPTIC_CENTER")
        )

    def topological_index(self) -> int:
        if not math.isfinite(self.poincare_rotation_number):
            return 0
        return int(round(self.poincare_rotation_number))

    def final_rho(self) -> NDArray[np.complex128]:
        seq = self.get_rho_sequence_arrays()
        return seq[-1] if seq else np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)


class SovereignAdjunctionAuditor:
    r"""
    Auditor soberano que combina:
      • El reporte categórico base (FASE 1: F, G, η, ε, Δ_△, A_μ, F_dR, ρ)
      • La trayectoria L3 con Poincaré / KAM / BHT / Birkhoff (FASE 2)
    y produce un DeRhamGaloisAdjunctionReport enriquecido.

    Clasificación META_STABLE_L3 cuando la órbita L3 es estable y la
    fidelidad de Uhlmann *post-L3* alcanza el umbral de isomorfismo:
    la automejora recursiva repara la adjunción.
    """

    def __init__(
        self,
        bridge: DeRhamGaloisAdjunctionBridge,
        improver: Optional[RecursiveAdjunctionImprover] = None,
    ) -> None:
        self.bridge = bridge
        self.improver = improver or RecursiveAdjunctionImprover(
            contraction=0.5, meta_learning=0.3
        )

    def full_audit(
        self,
        cartridges: Sequence[ToonCartridge],
        target_density_matrix: Optional[NDArray] = None,
        max_iters: int = _L3_DEFAULT_ITERS,
    ) -> DeRhamGaloisAdjunctionReport:
        base = self.bridge.evaluate_adjunction_coherence(cartridges, target_density_matrix)
        seed = self.bridge.to_categorical_state_seed(cartridges, target_density_matrix)
        traj = self.improver.iterate_level_3(seed, max_iters=max_iters)

        section = traj.build_poincare_section(seed.get_probe())
        prm = AdjunctionReturnMap(section)
        crossings = prm.apply(traj.get_rho_sequence_arrays())

        fid_l3, _bures_l3 = self.bridge.evaluate_uhlmann_fidelity_and_bures(
            traj.final_rho(), seed.get_target()
        )

        if traj.is_stable_meta_adjunction() and fid_l3 >= _UHLMANN_FIDELITY_MIN:
            status = AdjunctionCoherenceStatus.META_STABLE_L3
        else:
            status = base.coherence_status

        extended_seed = (
            f"{base.report_id}:{base.uhlmann_fidelity:.8f}:"
            f"{traj.lyapunov_exponent:.8f}:{traj.diophantine_gamma:.8f}:"
            f"{traj.birkhoff_classification}:{status.value}:{fid_l3:.8f}:"
            f"{traj.holonomy_phase:.8f}"
        ).encode("utf-8")
        merkle = hashlib.sha256(extended_seed).hexdigest()

        winding = traj.winding_numbers[:4] if traj.winding_numbers else tuple()

        return DeRhamGaloisAdjunctionReport(
            report_id=base.report_id,
            dimension_N=base.dimension_N,
            particle_count=base.particle_count,
            trace_density=base.trace_density,
            uhlmann_fidelity=base.uhlmann_fidelity,
            bures_distance=base.bures_distance,
            galois_gap_delta=base.galois_gap_delta,
            unit_eta_norm=base.unit_eta_norm,
            counit_epsilon_norm=base.counit_epsilon_norm,
            connection_A_norm=base.connection_A_norm,
            coherence_status=status,
            sha256_merkle_root=merkle,
            timestamp_utc=base.timestamp_utc,
            lyapunov_exponent=traj.lyapunov_exponent,
            spectral_radius=traj.spectral_radius,
            kam_diophantine_gamma=traj.diophantine_gamma,
            kam_stable=traj.is_kam_stable,
            is_bht_stable=traj.is_bht_stable,
            poincare_crossings=len(crossings),
            poincare_rotation_number=traj.poincare_rotation_number,
            twist_condition=traj.twist_condition,
            adjunction_winding=winding,
            meta_learning_final=traj.final_meta_learning,
            contraction_final=traj.final_contraction,
            birkhoff_classification=traj.birkhoff_classification,
            holonomy_phase=traj.holonomy_phase,
            yang_mills_action=traj.yang_mills_action,
            triangle_identity_defect=max(traj.triangle_left, traj.triangle_right),
            monad_idempotency_defect=traj.triangle_right,
            uhlmann_fidelity_l3=fid_l3,
            resonance_hits=traj.resonance_hits,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA DE AUDITORÍA DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
def _build_demo_cartridges() -> List[ToonCartridge]:
    """Construye las 12 partículas con dispatch entre v11 / v10 / v9 / fallback."""
    if _TOON_V10:
        return [
            ToonParticleFactory.create_polaron("POL-001", 12.5),
            ToonParticleFactory.create_torsion("TOR-002", 0.045),
            ToonParticleFactory.create_electron("ELE-003", 15_000_000.0),
            ToonParticleFactory.create_proton("PRO-004", 45.0),
            ToonParticleFactory.create_householder("HOU-005", "REV-2026-v9"),
            ToonParticleFactory.create_photon("PHO-006", "SECOP-2026-8891"),
            ToonParticleFactory.create_riemannian_focal("RFB-007", 0.0012),
            ToonParticleFactory.create_magnon("MAG-008", True),
            ToonParticleFactory.create_swelling_plasmon("SWP-009", 0.015),
            ToonParticleFactory.create_yielding_phonon("YPH-010", "CCT-2026-CIVIL"),
            ToonParticleFactory.create_liquefaction_soliton("LQS-011", "ACTA-ETHNIC-001"),
            ToonParticleFactory.create_positron("POS-012", "HMAC-AUTH-OK-2026"),
        ]
    if _TOON_AVAILABLE:
        try:
            return ToonParticleFactory.create_all_12_fundamental_cartridges(  # type: ignore[attr-defined]
                secret_key=b"SOVEREIGN_KEY_2026"
            )
        except Exception:
            logger.warning("ToonParticleFactory v9 dispatch falló; usando fallback.")
    return [
        ToonCartridge(  # type: ignore[call-arg, misc]
            f"CART-{i:02d}",
            "FERMION_CAR" if i < 5 else "BOSON_CCR",
            f"Particle-{i}",
            511.0,
            f"SIGN-{i}",
        )
        for i in range(12)
    ]


if __name__ == "__main__":
    banner = "═" * 105
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — v11.0.0-SOVEREIGN-DERHAM-GALOIS-POINCARE-KAM-BHT-L3-PhD")
    print("  F⊣G + de Rham + Uhlmann + Poincaré Σ + Twist P-B + KAM/BHT + Automejora L3")
    print(banner)

    cartridges_12 = _build_demo_cartridges()

    bridge = DeRhamGaloisAdjunctionBridge(dimension=56)
    auditor = SovereignAdjunctionAuditor(bridge)
    report = auditor.full_audit(cartridges_12)

    print("\n  ─── Reporte Categórico Base (FASE 1) ───────────────────────────────────────")
    print(f"  • Report ID             : {report.report_id}")
    print(f"  • Dimensión N           : {report.dimension_N}")
    print(f"  • Partículas Elevadas   : {report.particle_count}")
    print(f"  • Traza ρ_MAC           : {report.trace_density:.6f}")
    print(f"  • Fidelidad Uhlmann F   : {report.uhlmann_fidelity:.6f}  ← ‖√ρ√σ‖₁²")
    print(f"  • Fidelidad Uhlmann L3  : {report.uhlmann_fidelity_l3:.6f}  ← post-automejora")
    print(f"  • Distancia Bures d_B   : {report.bures_distance:.6f}")
    print(f"  • Galois Gap Δ_Frob     : {report.galois_gap_delta:.8e}  ← reciprocidad")
    print(f"  • Defecto triángulo Δ_△ : {report.triangle_identity_defect:.8e}")
    print(f"  • Defecto mónada T²−T   : {report.monad_idempotency_defect:.8e}")
    print(f"  • Norma Unidad η        : {report.unit_eta_norm:.6f}")
    print(f"  • Norma Counidad ε      : {report.counit_epsilon_norm:.6f}")
    print(f"  • Norma Conexión A_μ    : {report.connection_A_norm:.6f}")
    print(f"  • Acción Yang–Mills     : {report.yang_mills_action:.6e}")

    w0 = report.adjunction_winding[0] if len(report.adjunction_winding) > 0 else 0.0
    w1 = report.adjunction_winding[1] if len(report.adjunction_winding) > 1 else 0.0
    w2 = report.adjunction_winding[2] if len(report.adjunction_winding) > 2 else 0.0

    print("\n  ─── Análisis Poincaré / KAM / BHT / L3 (FASE 2 → FASE 3) ───────────────────")
    print(f"  • Exponente Lyapunov λ  : {report.lyapunov_exponent:+.6f}")
    print(f"  • Radio Espectral ρ(DI) : {report.spectral_radius:.6f}")
    print(f"  • Birkhoff Class        : {report.birkhoff_classification}")
    print(f"  • KAM γ Diofantina      : {report.kam_diophantine_gamma:.6e}")
    print(f"  • KAM Estable           : {report.kam_stable}")
    print(f"  • BHT (KAM disipativo)  : {report.is_bht_stable}")
    print(f"  • Cruces Poincaré Σ     : {report.poincare_crossings}")
    print(f"  • Rotación Birkhoff ρ   : {report.poincare_rotation_number:+.6f}")
    print(f"  • Twist Poincaré–Birkh. : {report.twist_condition}")
    print(f"  • Holonomía Berry γ     : {report.holonomy_phase:+.6f} rad")
    print(f"  • Hits resonancia p/q   : {report.resonance_hits}")
    print(f"  • Enrollamiento ω (3)   : ({w0:.6f}, {w1:.6f}, {w2:.6f})")
    print(f"  • θ_final (contracción) : {report.contraction_final:.6f}")
    print(f"  • η_meta_final (L3)     : {report.meta_learning_final:.6f}")
    print(f"  • Estado Coherencia F⊣G : {report.coherence_status.name}")
    print(f"  • Raíz Merkle SHA-256   : {report.sha256_merkle_root[:32]}...")

    print("\n  ─── Trayectoria L3 (ρ_t, J_t, θ_t) ─────────────────────────────────────────")
    seed = bridge.to_categorical_state_seed(cartridges_12)
    improver = RecursiveAdjunctionImprover()
    traj = improver.iterate_level_3(seed, max_iters=6)
    dR = bridge.de_rham_curvature_bundle(seed.get_rho())
    twist = AdjunctionPoincareBirkhoffTwist(seed)
    torus = KAMCoherenceTorus(seed.frequency_vector)

    print(f"    I₁ Poincaré–Cartan    : {seed.poincare_cartan_invariant():.6f}")
    print(f"    Pureza Tr(ρ²)         : {seed.purity():.6f}")
    print(f"    Entropía von Neumann  : {seed.von_neumann_entropy():.6f}")
    print(f"    Residuo Bianchi ‖[A,F]‖: {dR.bianchi_residue():.3e}")
    print(f"    Twist ∂(Δθ)/∂J        : {twist.twist_derivative():+.6e}")
    print(f"    Divisores pequeños (4): {torus.small_divisors()[:4]}")

    for t, (J, ang) in enumerate(zip(traj.action_sequence, traj.angle_sequence)):
        if t == 0:
            dlt = 0.0
        else:
            r_t = RecursiveAdjunctionImprover.deserialize(traj.rho_sequence[t], seed.dimension)
            r_p = RecursiveAdjunctionImprover.deserialize(traj.rho_sequence[t - 1], seed.dimension)
            dlt = float(np.linalg.norm(r_t - r_p, ord="fro"))
        print(
            f"    t={t:02d} | J_t = {J:.8f} | θ_t = {ang:+.6f} rad | Δρ = {dlt:.6e}"
        )
    print(
        f"    → λ = {traj.lyapunov_exponent:+.6f} | "
        f"meta-estable = {traj.is_stable_meta_adjunction()} | "
        f"índice topológico = {traj.topological_index()}"
    )

    print("\n" + banner)
    print("  ✓ FASE 1 → FASE 2: to_categorical_state_seed → CategoricalStateSeed")
    print("  ✓ FASE 2 → FASE 3: RecursiveAdjunctionImprover.iterate_level_3 → AdjunctionTrajectory")
    print("  ✓ Fidelidad de Uhlmann = ‖√ρ √σ‖₁² (norma nuclear, sin renormalizar middle)")
    print("  ✓ Galois Gap = defecto de reciprocidad de Frobenius ⟨F(X),Y⟩ ↔ ⟨X,G(Y)⟩")
    print("  ✓ Identidades del triángulo Δ_△ y mónada T = G∘F (T² − T)")
    print("  ✓ Complejo de de Rham matricial: F = A∧A, Bianchi, Yang–Mills, holonomía Berry")
    print("  ✓ Poincaré Σ transversal + rotación de Birkhoff + twist P–B + KAM/BHT + L3")
    print(banner)